/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include <mutex>
#include <vector>
#include <unordered_set>
#include <assert.h>

#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_camera_manager.h"
#include "rtx_options.h"
#include "rtx_materials.h"
#include "rtx_terrain_baker.h"

#include "rtx_cb_types.h"
#include "rtx_matrix_helpers.h"
#include "dxvk_scoped_annotation.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/concept/surface_material/surface_material_hitgroup.h"
#include "rtx/pass/instance_definitions.h"

namespace dxvk {
  
  static bool isMirrorTransform(const Matrix4& m) {
    // Note: Identify if the winding is inverted by checking if the z axis is ever flipped relative to what it's expected to be for clockwise vertices in a lefthanded space
    // (x cross y) through the series of transformations
    Vector3 x(m[0].data), y(m[1].data), z(m[2].data);
    return dot(cross(x, y), z) < 0;
  }

  static uint32_t determineInstanceFlags(const DrawCallState& drawCall, const RtSurface& surface) {
    // [SpawnGeomDiag] TF2 floor backface fix:
    // Empirically, on this dxvk-remix-DX11 branch, BLAS source vertex
    // ordering for D3D11 draws is opposite Vulkan ray tracing's default
    // front-face interpretation. With CULL_BACK active, every BSP world
    // surface (floor, walls, props) was being culled by primary rays.
    //
    // Forcing TRIANGLE_FACING_CULL_DISABLE_BIT_KHR on merged-bucket TLAS
    // instances made geometry appear (both sides hit). Forcing
    // TRIANGLE_FLIP_FACING_BIT_KHR (so CCW = front) made geometry render
    // correctly in CULL_BACK. Both confirm BLAS winding interpretation needs
    // inversion vs the OLD `drawClockwise == worldToProjectionMirrored` rule.
    //
    // Old rule's failure mode: TF2 main camera's projection determinant was
    // positive (mirror=0). drawClockwise=1, mirror=0 -> 1==0 false -> no
    // FLIP. So under the main camera, BSP draws were never flipped. Floor
    // invisible. The condition must be inverted: FLIP when drawClockwise
    // disagrees with the (post-instance-transform) winding parity.
    //
    // Switching the mirror basis to objectToWorld didn't help in isolation
    // (BSP o2w is identity). What works is the inverse comparison: FLIP when
    // `drawClockwise != objectToWorldMirrored`. For TF2 BSP main camera
    // (drawClockwise=1, o2wMirror=0): 1 != 0 -> FLIP. Floor renders.
    // For mirrored instances (drawClockwise=1, o2wMirror=1): 1 != 1 -> no
    // FLIP, which is correct because the negative-scale o2w has already
    // reversed the apparent winding.
    const Matrix4& objectToWorld = drawCall.getTransformData().objectToWorld;
    const bool objectToWorldMirrored = isMirrorTransform(objectToWorld);

    // Note: Vulkan ray tracing defaults to defining the front face based on clockwise vertex order when viewed from a left-handed coordinate system. The front face
    // should therefore be flipped if a counterclockwise ordering is used in this normal case, or the inverse logic if the BLAS instance transform inverts winding.
    // See: https://www.khronos.org/registry/vulkan/specs/1.1-khr-extensions/html/chap33.html#ray-traversal-culling-face
    const bool drawClockwise = drawCall.getGeometryData().frontFace == VkFrontFace::VK_FRONT_FACE_CLOCKWISE;

    uint32_t flags = 0;

    // Note: Flip front face by setting the front face to counterclockwise, which is the opposite of Vulkan ray tracing's clockwise default.
    if (drawClockwise != objectToWorldMirrored)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;

    if (!RtxOptions::enableCulling())
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    // [SpawnGeomDiag] Capture the inputs determineInstanceFlags decided on, per-draw,
    // to diagnose the TF2 BSP backface-cull issue. Logs both the OLD basis
    // (worldToProjection mirror) and the NEW basis (objectToWorld mirror) so
    // we can audit how often they disagree across all submissions and confirm
    // no unexpected regressions for non-BSP draws.
    // Throttled per (vsHash,matHash,frontFace,cullMode,o2wMirror,w2pMirror).
    {
      const Matrix4 worldToProjection = drawCall.getTransformData().viewToProjection * drawCall.getTransformData().worldToView;
      const bool worldToProjectionMirrored = isMirrorTransform(worldToProjection);
      const uint32_t vsHash = uint32_t(uint64_t(drawCall.getTransformData().vertexShaderHash) & 0xffffffffu);
      const uint32_t matHash = uint32_t(uint64_t(drawCall.getMaterialData().getHash()) & 0xffffffffu);
      const uint32_t cullMode = uint32_t(drawCall.getGeometryData().cullMode);
      const uint64_t key =
        (uint64_t(vsHash) << 32) ^ uint64_t(matHash) ^
        (uint64_t(drawClockwise ? 1 : 0) << 1) ^
        (uint64_t(objectToWorldMirrored ? 1 : 0) << 2) ^
        (uint64_t(worldToProjectionMirrored ? 1 : 0) << 3) ^
        (uint64_t(cullMode) << 8);
      static std::unordered_set<uint64_t> s_loggedFlagDecide;
      if (s_loggedFlagDecide.insert(key).second && s_loggedFlagDecide.size() <= 256) {
        Logger::warn(str::format("[SpawnGeomDiag.FlagDecide] vsHash=0x", std::hex, vsHash,
          " matHash=0x", matHash,
          " frontFace=", std::dec, uint32_t(drawCall.getGeometryData().frontFace),
          " (cw=", drawClockwise ? 1 : 0, ")",
          " o2wMirror=", objectToWorldMirrored ? 1 : 0,
          " w2pMirror=", worldToProjectionMirrored ? 1 : 0,
          " cullMode=", cullMode,
          " blend=", drawCall.getMaterialData().blendMode.enableBlending ? 1 : 0,
          " decal=", surface.alphaState.isDecal ? 1 : 0,
          " forceCullBit=", drawCall.getGeometryData().forceCullBit ? 1 : 0,
          " enableCulling=", RtxOptions::enableCulling() ? 1 : 0,
          " preFlags=0x", std::hex, flags, std::dec));
      }
    }

    // This check can be overridden by replacement assets.
    if (drawCall.getMaterialData().blendMode.enableBlending && !surface.alphaState.isDecal && !drawCall.getGeometryData().forceCullBit)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    // Disable culling for baked terrain instances when the option is enabled
    // Terrain with back face culling enabled may flicker in some circumstances.  
    // Forcing the geometry to be double-sided fixes the flicker, but may be undesireable in some games.
    if (TerrainBaker::disableBackFaceCulling() && drawCall.testCategoryFlags(InstanceCategories::Terrain)) {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    switch (drawCall.getGeometryData().cullMode) {
    case VkCullModeFlagBits::VK_CULL_MODE_NONE:
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_FRONT_BIT:
      // Note: Invert front face flag once more if front face culling is desired to make the current front face the backface (as we simply assume that any culling
      // desired will be backface via gl_RayFlagsCullBackFacingTrianglesEXT which helps simplify GPU-side logic).
      flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT:
      // Default in shader (gl_RayFlagsCullBackFacingTrianglesEXT)
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_FRONT_AND_BACK:
      assert(0); // this should already be filtered out up stack
      break;
    }

    return flags;
  }

  RtInstance::RtInstance(const uint64_t id, uint32_t instanceVectorId)
    : m_id(id)
    , m_instanceVectorId(instanceVectorId)
    , m_surfaceIndex(SURFACE_INDEX_INVALID)
    , m_previousSurfaceIndex(SURFACE_INDEX_INVALID) { }

  // Makes a copy of an instance
  RtInstance::RtInstance(const RtInstance& src, uint64_t id, uint32_t instanceVectorId)
    : surface(src.surface)
    , m_id(id)
    , m_instanceVectorId(instanceVectorId)
    , m_seenCameraTypes(src.m_seenCameraTypes)
    , m_materialType(src.m_materialType)
    , m_albedoOpacityTextureIndex(src.m_albedoOpacityTextureIndex)
    , m_samplerIndex(src.m_samplerIndex)
    , m_secondaryOpacityTextureIndex(src.m_secondaryOpacityTextureIndex)
    , m_secondarySamplerIndex(src.m_secondarySamplerIndex)
    , m_isAnimated(src.m_isAnimated)
    , m_opacityMicromapInstanceData(src.m_opacityMicromapInstanceData)
    , m_surfaceIndex(src.m_surfaceIndex)
    , m_previousSurfaceIndex(src.m_previousSurfaceIndex)
    , m_isHidden(src.m_isHidden)
    , m_isPlayerModel(src.m_isPlayerModel)
    , m_isWorldSpaceUI(src.m_isWorldSpaceUI)
    , m_isUnordered(src.m_isUnordered)
    , m_isObjectToWorldMirrored(src.m_isObjectToWorldMirrored)
    , m_linkedBlas(src.m_linkedBlas)
    , m_materialHash(src.m_materialHash)
    , m_materialDataHash(src.m_materialDataHash)
    , m_texcoordHash(src.m_texcoordHash)
    , m_indexHash(src.m_indexHash)
    , m_vkInstance(src.m_vkInstance)
    , m_geometryFlags(src.m_geometryFlags)
    , m_firstBillboard(src.m_firstBillboard)
    , m_billboardCount(src.m_billboardCount)
    , m_categoryFlags(src.m_categoryFlags) {
    // Members for which state carry over is intentionally skipped
    /*
       m_isMarkedForGC
       m_isUnlinkedForGC
       m_isInsideFrustum
       m_frameLastUpdated
       m_frameCreated
       m_isCreatedByRenderer
       m_spatialCacheHash
       m_primInstanceOwner
       buildGeometries
       buildRanges
       billboardIndices
       indexOffsets
     */
  }
  // Ensure the copy ctor copies all needed members when size changes, and update the object size check.
  // Note: The object has a different size on Debug builds. 
  //       Checking the non-Debug flavors is good enough for the sake of convenience of tracking just a single size.
 #if defined(DEBUG_OPTIMIZED) || defined(NDEBUG)
  namespace {
    template<int RtInstanceSize> struct CheckRtInstanceSize {
      // The second line of the build error should contain the new size of RtInstance in the template argument, i.e. `dxvk::CheckRtInstanceSize<newSize>`
      static_assert(RtInstanceSize == 768, "RtInstance size has changed.  Fix the copy constructor above this message, then update the expected size.");
    };
    CheckRtInstanceSize<sizeof(RtInstance)> _rtInstanceSizeTest;
  }
 #endif

  void RtInstance::setBlas(BlasEntry& blas) {
    m_linkedBlas = &blas;
  }

  void RtInstance::onTransformChanged() {
    // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
    // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
    const auto t = transpose(surface.objectToWorld);
    memcpy(&m_vkInstance.transform, &t, sizeof(VkTransformMatrixKHR));

    if (!m_isCreatedByRenderer) {
      // NOTE: This code would cache instances based on predicted position instead of current position, but in testing it fails too frequently
      // const Vector3 newPos = 2.f * surface.objectToWorld[3].xyz() - surface.prevObjectToWorld[3].xyz();

      // Cache based on current position.
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();
      const Vector3 newPos = getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().move(m_spatialCacheHash, newPos, firstInstanceObjectToWorld, this);
    }
  }

  bool RtInstance::teleport(const Matrix4& objectToWorld) {
    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = objectToWorld;
    if (!m_isCreatedByRenderer) {
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();
      const Vector3 centroid = getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().insert(centroid, firstInstanceObjectToWorld, this);
    }
    
    // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
    // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
    const auto t = transpose(surface.objectToWorld);
    memcpy(&m_vkInstance.transform, &t, sizeof(VkTransformMatrixKHR));
    
    return false; // freshly teleported instances are always treated as still.
  }

  bool RtInstance::teleport(const Matrix4& objectToWorld, const Matrix4& prevObjectToWorld) {
    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = prevObjectToWorld;
    onTransformChanged();

    return memcmp(surface.prevObjectToWorld.data, surface.objectToWorld.data, sizeof(Matrix4)) != 0;
  }

  void RtInstance::teleportWithHistory(const Matrix4& oldToNew) {
    surface.objectToWorld = oldToNew * surface.objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = oldToNew * surface.prevObjectToWorld;
    onTransformChanged();

    if (m_primInstanceOwner.isRoot(this)) {
      // this is the root of a replacement - need to update the transform history for all the instances in the replacement.
      for (size_t i = 0; i < m_primInstanceOwner.getReplacementInstance()->prims.size(); i++) {
        RtInstance* instance = m_primInstanceOwner.getReplacementInstance()->prims[i].getInstance();
        if (instance != nullptr && instance != this) {
          instance->teleportWithHistory(oldToNew);
        }
      }
    }
  }
  
  bool RtInstance::move(const Matrix4& objectToWorld) {
    surface.prevObjectToWorld = surface.objectToWorld;
    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(objectToWorld)));
    onTransformChanged();

    // See if the transform has changed even a tiny bit.
    // The result is used for the 'isStatic' surface flag, which is in turn used to skip motion vector calculation
    // on the GPU. We need nonzero motion vectors on objects moving even slightly to make RTXDI temporal bias correction work.
    // This comparison is not robust if the transforms are reconstructed from baked object-to-view matrices,
    // but it works well e.g. in Portal. Even if it detects truly static objects as moving, that's fine because that will only
    // have a minor performance effect of calculation extra motion vectors.
    return memcmp(surface.prevObjectToWorld.data, surface.objectToWorld.data, sizeof(Matrix4)) != 0;
  }

  bool RtInstance::moveAgain(const Matrix4& objectToWorld) {
    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(objectToWorld)));
    onTransformChanged();

    // See comment in move()
    return memcmp(surface.prevObjectToWorld.data, surface.objectToWorld.data, sizeof(Matrix4)) != 0;
  }

  void RtInstance::setFrameCreated(const uint32_t frameIndex) {
    m_frameCreated = frameIndex;
  }

  // Sets frame id of last update, if this is the first time the frame id is set
  // instance's per frame state is reset as well
  // Returns true if this is the first update this frame
  bool RtInstance::setFrameLastUpdated(const uint32_t frameIndex) {
    if (m_frameLastUpdated != frameIndex) {
      m_seenCameraTypes.clrAll();

      m_frameLastUpdated = frameIndex;

      return true;
    }

    return false;
  }

  void RtInstance::markForGarbageCollection() const {
    m_isMarkedForGC = true;
  }

  void RtInstance::markAsUnlinkedFromBlasEntryForGarbageCollection() const {
    m_isUnlinkedForGC = true;
  }

  void RtInstance::markAsInsideFrustum() const {
    m_isInsideFrustum = true;
  }

  void RtInstance::markAsOutsideFrustum() const {
    m_isInsideFrustum = false;
  }

  bool RtInstance::registerCamera(CameraType::Enum cameraType, uint32_t frameIndex) {
    const bool settingNewCameraType = !m_seenCameraTypes.test(cameraType);

    if (settingNewCameraType)
      m_seenCameraTypes.set(cameraType);

    return settingNewCameraType;
  }

  bool RtInstance::isCameraRegistered(CameraType::Enum cameraType) const {
    return m_seenCameraTypes.test(cameraType);
  }

  void RtInstance::setCustomIndexBit(uint32_t oneBitMask, bool value) {
    m_vkInstance.instanceCustomIndex = setBit(m_vkInstance.instanceCustomIndex, value, oneBitMask);
  }

  bool RtInstance::getCustomIndexBit(uint32_t oneBitMask) const {
    return m_vkInstance.instanceCustomIndex & oneBitMask;
  }

  bool RtInstance::isOpaque() const {
    return getMaterialType() == MaterialDataType::Opaque;
  }

  bool RtInstance::isViewModel() const {
    return getCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL);
  }

  bool RtInstance::isViewModelNonReference() const {
    return m_vkInstance.mask != 0 && isViewModel();
  }

  bool RtInstance::isViewModelReference() const { 
    return m_vkInstance.mask == 0 && isViewModel();
    }

  bool RtInstance::isViewModelVirtual() const {
    return m_vkInstance.mask & OBJECT_MASK_VIEWMODEL_VIRTUAL;
  }

  void RtInstance::printDebugInfo() const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "=== RtInstance Debug Info ===\n",
      "ID: ", m_id, "\n",
      "Vector Index: ", m_instanceVectorId, "\n",
      "Frame Created: ", m_frameCreated, "\n",
      "Frame Last Updated: ", m_frameLastUpdated, "\n",
      "Frame Age: ", getFrameAge(), "\n",
      "\n",
      "=== Transform Info ===\n",
      "World Position: (", getWorldPosition().x, ", ", getWorldPosition().y, ", ", getWorldPosition().z, ")\n",
      "Transform Matrix:\n",
      "  [", getTransform()[0][0], ", ", getTransform()[0][1], ", ", getTransform()[0][2], ", ", getTransform()[0][3], "]\n",
      "  [", getTransform()[1][0], ", ", getTransform()[1][1], ", ", getTransform()[1][2], ", ", getTransform()[1][3], "]\n",
      "  [", getTransform()[2][0], ", ", getTransform()[2][1], ", ", getTransform()[2][2], ", ", getTransform()[2][3], "]\n",
      "  [", getTransform()[3][0], ", ", getTransform()[3][1], ", ", getTransform()[3][2], ", ", getTransform()[3][3], "]\n",
      "Previous World Position: (", getPrevWorldPosition().x, ", ", getPrevWorldPosition().y, ", ", getPrevWorldPosition().z, ")\n",
      "\n",
      "=== BLAS Info ===\n",
      "Linked BLAS: ", m_linkedBlas ? "Valid" : "Null"));
    
    if (m_linkedBlas) {
      Logger::warn("=== BLAS Entry Debug Info ===");
      m_linkedBlas->printDebugInfo("(from RtInstance)");
      Logger::warn("=== End BLAS Entry Debug Info ===");
      
      // Print DrawCallState info
      Logger::warn("=== DrawCallState Debug Info ===");
      m_linkedBlas->input.printDebugInfo("(from RtInstance)");
      Logger::warn("=== End DrawCallState Debug Info ===");
    }
    
    // Print RtSurface info
    Logger::warn("=== RtSurface Debug Info ===");
    surface.printDebugInfo("(from RtInstance)");
    Logger::warn("=== End RtSurface Debug Info ===");
    
    Logger::warn(str::format(
      "=== Hash Info ===\n",
      "Material Hash: 0x", std::hex, m_materialHash, std::dec, "\n",
      "Material Data Hash: 0x", std::hex, m_materialDataHash, std::dec, "\n",
      "Texcoord Hash: 0x", std::hex, m_texcoordHash, std::dec, "\n",
      "Index Hash: 0x", std::hex, m_indexHash, std::dec, "\n",
      "Spatial Cache Hash: 0x", std::hex, m_spatialCacheHash, std::dec, "\n",
      "\n",
      "=== Vulkan Instance Info ===\n",
      "VK Instance Mask: ", m_vkInstance.mask, "\n",
      "VK Instance Flags: ", m_vkInstance.flags, "\n",
      "VK Instance Custom Index: ", m_vkInstance.instanceCustomIndex, "\n",
      "VK Instance SBT Record Offset: ", m_vkInstance.instanceShaderBindingTableRecordOffset, "\n",
      "\n",
      "=== Material Info ===\n",
      "Material Type: ", static_cast<int>(m_materialType), "\n",
      "Albedo Opacity Texture Index: ", m_albedoOpacityTextureIndex, "\n",
      "Sampler Index: ", m_samplerIndex, "\n",
      "Secondary Opacity Texture Index: ", m_secondaryOpacityTextureIndex, "\n",
      "Secondary Sampler Index: ", m_secondarySamplerIndex, "\n",
      "\n",
      "=== Surface Info ===\n",
      "Surface Index: ", m_surfaceIndex, "\n",
      "Previous Surface Index: ", m_previousSurfaceIndex, "\n",
      "\n",
      "=== Billboard Info ===\n",
      "First Billboard Index: ", m_firstBillboard, "\n",
      "Billboard Count: ", m_billboardCount, "\n",
      "\n",
      "=== Geometry Info ===\n",
      "Geometry Flags: ", m_geometryFlags, "\n",
      "\n",
      "=== Boolean Flags ===\n",
      "Is Hidden: ", m_isHidden ? "true" : "false", "\n",
      "Is Player Model: ", m_isPlayerModel ? "true" : "false", "\n",
      "Is World Space UI: ", m_isWorldSpaceUI ? "true" : "false", "\n",
      "Is Unordered: ", m_isUnordered ? "true" : "false", "\n",
      "Is Object To World Mirrored: ", m_isObjectToWorldMirrored ? "true" : "false", "\n",
      "Is Created By Renderer: ", m_isCreatedByRenderer ? "true" : "false", "\n",
      "Is Animated: ", m_isAnimated ? "true" : "false", "\n",
      "Is Front Face Flipped: ", isFrontFaceFlipped ? "true" : "false", "\n",
      "\n",
      "=== Garbage Collection Flags ===\n",
      "Is Marked For GC: ", m_isMarkedForGC ? "true" : "false", "\n",
      "Is Unlinked For GC: ", m_isUnlinkedForGC ? "true" : "false", "\n",
      "Is Inside Frustum: ", m_isInsideFrustum ? "true" : "false", "\n",
      "\n",
      "=== View Model Flags ===\n",
      "Is View Model: ", isViewModel() ? "true" : "false", "\n",
      "Is View Model Non Reference: ", isViewModelNonReference() ? "true" : "false", "\n",
      "Is View Model Reference: ", isViewModelReference() ? "true" : "false", "\n",
      "Is View Model Virtual: ", isViewModelVirtual() ? "true" : "false", "\n",
      "\n",
      "=== Category Info ===\n",
      "Category Flags: ", m_categoryFlags.raw(), "\n",
      "\n",
      "=== Camera Types ===\n",
      "Seen Camera Types Mask: ", m_seenCameraTypes.raw()));

    for (uint32_t type = 0; type < CameraType::Count; ++type) {
      if (m_seenCameraTypes.test(static_cast<CameraType::Enum>(type))) {
        Logger::warn(str::format("  Camera Type: ", type));
      }
    }
    
    Logger::warn(str::format(
      "\n=== Billboard Indices ===\n",
      "Billboard Indices Count: ", billboardIndices.size()));
    
    for (size_t i = 0; i < std::min(billboardIndices.size(), size_t(5)); ++i) {
      Logger::warn(str::format("  Billboard Index ", i, ": ", billboardIndices[i]));
    }
    if (billboardIndices.size() > 5) {
      Logger::warn(str::format("  ... and ", billboardIndices.size() - 5, " more"));
    }
    
    Logger::warn(str::format(
      "\n=== Index Offsets ===\n",
      "Index Offsets Count: ", indexOffsets.size()));
    
    for (size_t i = 0; i < std::min(indexOffsets.size(), size_t(5)); ++i) {
      Logger::warn(str::format("  Index Offset ", i, ": ", indexOffsets[i]));
    }
    if (indexOffsets.size() > 5) {
      Logger::warn(str::format("  ... and ", indexOffsets.size() - 5, " more"));
    }
    
    Logger::warn("=== End RtInstance Debug Info ===");
#endif
}

  InstanceManager::InstanceManager(DxvkDevice* device, ResourceCache* pResourceCache)
    : CommonDeviceObject(device)
    , m_pResourceCache(pResourceCache) {
    m_previousViewModelState = RtxOptions::ViewModel::enable();
  }

  InstanceManager::~InstanceManager() {
  }

  void InstanceManager::removeEventHandler(void* eventHandlerOwnerAddress) {
    for (auto eventIter = m_eventHandlers.begin(); eventIter != m_eventHandlers.end(); eventIter++) {
      if (eventIter->eventHandlerOwnerAddress == eventHandlerOwnerAddress) {
        m_eventHandlers.erase(eventIter);
        break;
      }
    }
  }

  void InstanceManager::clear() {
    for (RtInstance* instance : m_instances) {
      removeInstance(instance);
      delete instance;
    }

    m_instances.clear();
    m_viewModelCandidates.clear();
    m_playerModelInstances.clear();
  }  

  void InstanceManager::garbageCollection() {
    // Can be configured per game: 'rtx.numFramesToKeepInstances'
    const uint32_t numFramesToKeepInstances = RtxOptions::numFramesToKeepInstances();

    // NV-DXVK [GcEntry probe]: log the value GC actually uses at decision
    // time. The Rm2904 probe in removeInstance reads RtxOptions later in
    // the call stack and may see a different (post-apply) value if
    // RtxOptionManager::applyPendingValues races between the two reads.
    //
    // Also track removed-this-pass count so we can correlate spikes in
    // removeInstance calls with specific GC passes.
    uint32_t probeRemovedThisPass = 0;
    uint32_t probeKeptThisPass    = 0;
    uint32_t probeRemovedMarked   = 0;
    uint32_t probeRemovedLifetime = 0;
    uint32_t probeRemovedForce    = 0;
    const uint32_t probeFrame     = m_device->getCurrentFrameId();
    {
      Logger::info(str::format(
        "[GcEntry] f=", probeFrame,
        " keepInstances=", numFramesToKeepInstances,
        " antiCullEnabled=", (RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1 : 0),
        " instCount=", m_instances.size()));
    }

    // Remove instances past their lifetime or marked for GC explicitly
    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // Need to release all instances when ViewModel enablement changes
    // This is a big hammer but it's fine, it's a debugging feature
    const bool isViewModelEnabled = RtxOptions::ViewModel::enable();
    if (isViewModelEnabled != m_previousViewModelState) {
      clear();
      m_previousViewModelState = isViewModelEnabled;
    }

    const bool forceGarbageCollection = (m_instances.size() >= RtxOptions::AntiCulling::Object::numObjectsToKeep());
    for (uint32_t i = 0; i < m_instances.size();) {
      // Must take a ref here since we'll be swapping
      RtInstance*& pInstance = m_instances[i];
      assert(pInstance != nullptr);

      const bool enableGarbageCollection =
        !RtxOptions::AntiCulling::isObjectAntiCullingEnabled() || // It's always True if anti-culling is disabled
        (pInstance->m_isInsideFrustum) ||
        (pInstance->getBlas()->input.getSkinningState().numBones > 0) ||
        (pInstance->m_isAnimated) ||
        (pInstance->m_isPlayerModel);

      // NV-DXVK [GC clause probe]: capture WHICH side of the OR fired and
      // whether this is a sub-view mountain instance. Lets us correlate
      // [Rm2904] entries that show "all conditions false" with the actual
      // GC-time evaluation.
      const bool clauseLifetime = (forceGarbageCollection || enableGarbageCollection) &&
                                  (pInstance->m_frameLastUpdated + numFramesToKeepInstances <= currentFrame);
      const bool clauseMarked = pInstance->m_isMarkedForGC;
      const bool shouldRemove = clauseLifetime || clauseMarked;
      if (shouldRemove) {
        probeRemovedThisPass += 1;
        if (clauseMarked)   probeRemovedMarked   += 1;
        if (clauseLifetime) probeRemovedLifetime += 1;
        if (forceGarbageCollection && !enableGarbageCollection && clauseLifetime) probeRemovedForce += 1;

        // Per-instance log for VS_2904d2: capture exactly what GC saw.
        {
          const BlasEntry* pBlasGC = pInstance->getBlas();
          if (pBlasGC != nullptr
              && pBlasGC->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
            thread_local uint32_t sGcVsProbe = 0;
            if (sGcVsProbe < 16 || (sGcVsProbe & 0xFF) == 0) {
              Logger::info(str::format(
                "[Gc2904Decide] #", sGcVsProbe,
                " f=", currentFrame,
                " lastUpd=", pInstance->m_frameLastUpdated,
                " keepN=", numFramesToKeepInstances,
                " force=", (forceGarbageCollection ? 1 : 0),
                " enable=", (enableGarbageCollection ? 1 : 0),
                " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
                " markedGC=", (clauseMarked ? 1 : 0),
                " clauseLifetime=", (clauseLifetime ? 1 : 0)));
            }
            sGcVsProbe += 1;
          }
        }

        // Note: Pop and swap for performance, index not incremented to process swapped instance on next iteration
        removeInstance(pInstance);

        // NOTE: pInstance is now the (previously) last element
        std::swap(pInstance, m_instances.back());

        m_instances[i]->m_instanceVectorId = i;

        delete m_instances.back();

        // Remove the last element
        m_instances.pop_back();
        continue;
      }
      probeKeptThisPass += 1;
      ++i;
    }

    // [GcExit]: summarize this GC pass. If probeRemovedThisPass > 0 but
    // probeRemovedMarked == 0 && probeRemovedLifetime == 0, removeInstance
    // must be running from somewhere ELSE — telling us to look at another
    // call site.
    Logger::info(str::format(
      "[GcExit] f=", probeFrame,
      " kept=", probeKeptThisPass,
      " removed=", probeRemovedThisPass,
      " viaMarked=", probeRemovedMarked,
      " viaLifetime=", probeRemovedLifetime,
      " viaForce=", probeRemovedForce));
  }

  void InstanceManager::onFrameEnd() {
    m_viewModelCandidates.clear();
    m_playerModelInstances.clear();
    resetSurfaceIndices();
    m_billboards.clear();
    // reset decal counter
    m_decalSortOrderCounter = 0;
  }

  RtInstance* InstanceManager::processSceneObject(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance) {

    // If the RtInstance represents multiple instances, use the full transform of the first copy for the spatial map.
    // this prevents a bad de-duplication when the same replacement asset is used in multiple GeomPointInstancer prims.
    Matrix4 firstInstanceObjectToWorld = drawCall.getTransformData().calcFirstInstanceObjectToWorld();

    // If we already know which instance to use, just use that.
    RtInstance* currentInstance = existingInstance;

    // Search for an existing instance matching our input
    if (currentInstance == nullptr) {
      currentInstance = findSimilarInstance(blas, materialData, firstInstanceObjectToWorld, drawCall.cameraType, rayPortalManager);
    }

    // NV-DXVK [SubView phantom check]: per-VS-per-frame counters of
    // "found similar existing instance" vs "created new". If sub-view
    // VSes show new>0 in steady state after the smoothed-scale fix,
    // phantoms are still being spawned each frame and we need to look
    // at why findSimilarInstance is missing despite stable o2w.
    //
    // Note: this fires for ALL VSes, but the [SubViewVar] log already
    // identifies which hashes correspond to sub-view content — match
    // those hashes against this log's vs= field.
    const bool foundSimilar = (currentInstance != nullptr);
    {
      const XXH64_hash_t vsHash = drawCall.getTransformData().vertexShaderHash;
      char vsKey[24] = "VS_";
      static const char hex[] = "0123456789abcdef";
      for (int i = 0; i < 8; ++i) {
        const uint8_t b = static_cast<uint8_t>(vsHash >> ((7 - i) * 8));
        vsKey[3 + i*2 + 0] = hex[b >> 4];
        vsKey[3 + i*2 + 1] = hex[b & 0xF];
      }
      vsKey[19] = '\0';
      struct InstStats {
        uint32_t frameId  = 0;
        uint32_t reused   = 0;
        uint32_t created  = 0;
      };
      static thread_local std::unordered_map<std::string, InstStats> sStats;
      static thread_local uint32_t sLastReportedFrame = UINT32_MAX;
      const uint32_t curFrame = m_device->getCurrentFrameId();
      auto& s = sStats[vsKey];
      if (s.frameId != curFrame) {
        s.frameId = curFrame;
        s.reused = 0;
        s.created = 0;
      }
      if (foundSimilar) s.reused += 1;
      else              s.created += 1;
      // Emit a per-VS summary once per frame, on the first observed
      // draw of the new frame (across any VS). Keeps the log size
      // bounded — one [InstCounts] line per active VS per frame.
      if (sLastReportedFrame != curFrame) {
        sLastReportedFrame = curFrame;
        for (auto& kv : sStats) {
          if (kv.second.frameId + 1 == curFrame
              && (kv.second.reused + kv.second.created) > 0) {
            Logger::info(str::format(
              "[InstCounts] frame=", kv.second.frameId,
              " vs=", kv.first,
              " reused=", kv.second.reused,
              " created=", kv.second.created));
          }
        }
      }
    }

    // NV-DXVK [VS_2904d2 trace]: this VS shows up in [InstCounts] as
    // created=48/frame (the biggest phantom source) but is invisible
    // to our d3d11-side probes [PhantomProbe] and [Cb2Track]. That
    // means it enters the instance manager via a path that bypasses
    // D3D11Rtx::SubmitDraw — likely a replacement-asset / GeomPoint
    // Instancer route — so our SetSkyCategoryFromCb2 reproject never
    // gets a shot at it.
    //
    // Goal of this probe: figure out which d3d11 VS it ACTUALLY
    // corresponds to (its raw XXH64 hash at this layer differs from
    // the d3d11 shader key), and watch the per-draw
    // firstInstanceObjectToWorld.translation so we can tell:
    //   - if it lands at far-distance main-world coords already
    //     (somebody upstream reprojected it for us), or
    //   - if it lands at sub-view-local coords < 100u from skyCam
    //     (untouched sub-view data → we need to reproject here),
    //   - whether it shifts frame-to-frame with the moving ship
    //     (drives the dedup miss → phantom trail), or
    //   - whether it's stable when the ship is stationary (the
    //     reused=48/created=0 window at frames 884-887 suggests
    //     this is the case).
    {
      const XXH64_hash_t vsHash = drawCall.getTransformData().vertexShaderHash;
      if (vsHash == 0x2904d2163ef31a17ull) {
        thread_local uint32_t sLastFrame = UINT32_MAX;
        thread_local uint32_t sDrawIdx   = 0;
        const uint32_t curFrame = m_device->getCurrentFrameId();
        if (sLastFrame != curFrame) {
          sLastFrame = curFrame;
          sDrawIdx   = 0;
        } else {
          sDrawIdx  += 1;
        }
        // Only log first ~12 draws per frame to keep volume bounded —
        // 48 draws × every frame would spam the log.
        if (sDrawIdx < 12) {
          const Matrix4& m = firstInstanceObjectToWorld;
          const uint32_t camType = static_cast<uint32_t>(drawCall.cameraType);
          Logger::info(str::format(
            "[VS2904Trace] frame=", curFrame, " draw=", sDrawIdx,
            " vsHash=0x", std::hex, vsHash, std::dec,
            " camType=", camType,
            " firstO2W.t=(", float(m[3][0]), ",",
                              float(m[3][1]), ",",
                              float(m[3][2]), ")",
            " foundSimilar=", (foundSimilar ? 1 : 0)));
        }
      }
    }

    if (currentInstance == nullptr) {
      // No existing match - so need to create one
      currentInstance = addInstance(blas);
    }

    updateInstance(*currentInstance, cameraManager, blas, drawCall, materialData);

    return currentInstance;
  }

  RtSurface::AlphaState InstanceManager::calculateAlphaState(const DrawCallState& drawCall, const MaterialData& materialData) {
    RtSurface::AlphaState out{};

    // Handle Alpha State for non-Opaque materials

    if (materialData.getType() == MaterialDataType::Translucent) {
      // Note: Explicitly ensure translucent materials are not considered fully opaque (even though this is the
      // default in the alpha state).
      out.isFullyOpaque = false;

      return out;
    } else if (materialData.getType() != MaterialDataType::Opaque) {
      return out;
    }

    // Determine if the Legacy Alpha State should be used based on the material data
    // Note: The Material Data may be either Legacy or Opaque here, both use the Opaque Surface Material.

    const auto& opaqueMaterialData = materialData.getOpaqueMaterialData();

    const bool useLegacyAlphaState = opaqueMaterialData.getUseLegacyAlphaState();

    // Handle Alpha Test State

    // Note: Even if the Alpha Test enable flag is set, we consider it disabled if the actual test type is set to always.
    const bool forceAlphaTest = drawCall.getCategoryFlags().test(InstanceCategories::AlphaBlendToCutout);
    const bool alphaTestEnabled = forceAlphaTest || (AlphaTestType)drawCall.getMaterialData().alphaTestCompareOp != AlphaTestType::kAlways;

    // Note: Use the Opaque Material Data's alpha test state information directly if requested,
    // otherwise derive the alpha test state from the drawcall (via its legacy material data).
    if (forceAlphaTest) {
      out.alphaTestType = AlphaTestType::kGreater;
      out.alphaTestReferenceValue = static_cast<uint8_t>(RtxOptions::forceCutoutAlpha() * 255.0);
    } else if (!useLegacyAlphaState) {
      out.alphaTestType = opaqueMaterialData.getAlphaTestType();
      out.alphaTestReferenceValue = opaqueMaterialData.getAlphaTestReferenceValue();
    } else if (alphaTestEnabled) {
      out.alphaTestType = (AlphaTestType)drawCall.getMaterialData().alphaTestCompareOp;
      out.alphaTestReferenceValue = drawCall.getMaterialData().alphaTestReferenceValue;
    }

    // Handle Alpha Blend State

    bool blendEnabled = false;
    BlendType blendType = BlendType::kColor;
    bool invertedBlend = false;

    // Note: Use the Opaque Material Data's blend state information directly if requested,
    // otherwise derive the alpha blend state from the drawcall (via its legacy material data).
    if (forceAlphaTest) {
      blendEnabled = false;
    } else if (!useLegacyAlphaState) {
      blendEnabled = opaqueMaterialData.getBlendEnabled();
      blendType = opaqueMaterialData.getBlendType();
      invertedBlend = opaqueMaterialData.getInvertedBlend();
    } else if (drawCall.getMaterialData().blendMode.enableBlending) {
      const auto srcColorBlendFactor = drawCall.getMaterialData().blendMode.colorSrcFactor;
      const auto dstColorBlendFactor = drawCall.getMaterialData().blendMode.colorDstFactor;
      const auto colorBlendOp = drawCall.getMaterialData().blendMode.colorBlendOp;

      blendEnabled = true; // Note: Set to false later for cases which don't need it

      if (colorBlendOp == VkBlendOp::VK_BLEND_OP_ADD) {
        if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) {
          // Opaque Alias
          blendEnabled = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Standard Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Inverted Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Standard Reverse Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Premultiplied Alpha vs Inverted Reverse Emissive Alpha Blending
          const auto srcAlphaBlendFactor = drawCall.getMaterialData().blendMode.alphaSrcFactor;
          const auto dstAlphaBlendFactor = drawCall.getMaterialData().blendMode.alphaDstFactor;
          const auto alphaBlendOp = drawCall.getMaterialData().blendMode.alphaBlendOp;
          const auto colorWriteMask = drawCall.getMaterialData().blendMode.writeMask;
          const bool alphaWritesDisabled = (colorWriteMask & VK_COLOR_COMPONENT_A_BIT) == 0;

          const bool looksPremultiplied =
            (alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
             srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
             dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
            alphaWritesDisabled;

          if (looksPremultiplied) {
            // Premultiplied Alpha (ONE, ONE_MINUS_SRC_ALPHA)
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
            invertedBlend = false;
          } else {
            // Inverted Reverse Emissive Alpha Blending
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
            invertedBlend = true;
          }
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Standard Color Blending
          blendType = BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Inverted Color Blending
          blendType = BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Standard Reverse Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseColorEmissive : BlendType::kReverseColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Inverted Reverse Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseColorEmissive : BlendType::kReverseColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Emissive Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) ||
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR)
          ) {
          // Standard Multiplicative Blending
          blendType = BlendType::kMultiplicative;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Double Multiplicative Blending
          blendType = BlendType::kDoubleMultiplicative;
          invertedBlend = false;
        } else {
          blendEnabled = false;
        }
      } else {
        blendEnabled = false;
      }
    }

    // Special case for the player model eyes in Portal:
    // They are rendered with blending enabled but 1.0 is added to alpha from the texture.
    // Detect this case here and turn such geometry into non-alpha-blended, otherwise
    // the eyes end up in the unordered TLAS and are not rendered correctly.
    const auto& drawMaterialData = drawCall.getMaterialData();
    if (blendEnabled && blendType == BlendType::kAlpha && !invertedBlend &&
        drawMaterialData.textureAlphaOperation == DxvkRtTextureOperation::Add &&
        drawMaterialData.textureAlphaArg1Source == RtTextureArgSource::Texture &&
        drawMaterialData.textureAlphaArg2Source == RtTextureArgSource::TFactor &&
        (drawMaterialData.tFactor >> 24) == 0xff) {
      blendEnabled = false;
    }

    if (blendEnabled) {
      out.blendType = blendType;
      out.invertedBlend = invertedBlend;
      // Note: Emissive blend flag must match which blend types are expected to use emissive override in the shader to appear emissive.
      out.emissiveBlend = isBlendTypeEmissive(blendType);

      // Handle Particle/Decal Flags
      // Note: Particles/Decals currently require blending be enabled, be it through the game's original draw call (if legacy alpha state is used),
      // or through the manually specified alpha state.

      // Note: Particles are differentiated from typical objects with opacity by labeling their source material textures as being particle textures.
      out.isParticle = drawCall.testCategoryFlags(InstanceCategories::Particle);
      out.isDecal = drawCall.testCategoryFlags(DECAL_CATEGORY_FLAGS);
    } else {
      out.invertedBlend = false;
      out.emissiveBlend = false;
    }
    
    // Set the fully opaque flag
    // Note: Fully opaque surfaces can only be signaled when no blending or alpha testing is done as well as no translucency material wise is used.
    // This is important for signaling when to not use the opacity channel in materials when it is not being used for anything.

    out.isFullyOpaque = !blendEnabled && out.alphaTestType == AlphaTestType::kAlways; // use the blend/test type from the output, rather than legacy for this so replacements can override
    out.isBlendingDisabled = !blendEnabled;

    return out;
  }

  void InstanceManager::mergeInstanceHeuristics(RtInstance& instanceToModify, const DrawCallState& drawCall, const RtSurface::AlphaState& alphaState) const {
    // "Opaqueness" takes priority!
    if (
      (alphaState.isFullyOpaque || alphaState.alphaTestType == AlphaTestType::kAlways) &&
      !(instanceToModify.surface.alphaState.isFullyOpaque || instanceToModify.surface.alphaState.alphaTestType == AlphaTestType::kAlways)
    ) {
      instanceToModify.surface.alphaState = alphaState;
    }

    // NOTE: In the future we could extend this with heuristics as needed...
  }

  RtInstance* InstanceManager::findSimilarInstance(BlasEntry& blas, const MaterialData& material, const Matrix4& firstInstanceObjectToWorld, CameraType::Enum cameraType, const RayPortalManager& rayPortalManager) {

    // Disable temporal correlation between instances so that duplicate instances are not created
    // should a developer option change instance enough for it not to match anymore
    if (RtxOptions::enableInstanceDebuggingTools()) {
      return nullptr;
    }

    RtInstance* result = nullptr;

    const uint32_t currentFrameIdx = m_device->getCurrentFrameId();
    const Vector3 worldPosition = blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
    
    const float uniqueObjectDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();

    RtInstance* pSimilar = nullptr;
    float nearestDistSqr = FLT_MAX;

    // NV-DXVK [VS_2904d2 findSimilar probe]: diagnose why dedup fails for
    // the sub-view mountain content. After our reproject is locked (scale
    // and anchor pinned to map constants), T_reproject is bytewise stable
    // per frame, so post-reproject world positions SHOULD be byte-identical
    // across frames for static props — and getDataAtTransform's exact-hash
    // lookup should hit every time. Yet InstCounts shows created=48 each
    // frame. We need to know per-stage what's failing.
    const XXH64_hash_t vsHashProbe = blas.input.getTransformData().vertexShaderHash;
    const bool isProbeVS = (vsHashProbe == 0x2904d2163ef31a17ull);

    // Search the BLAS for an instance matching ours
    {
      // Search for an exact match
      result = const_cast<RtInstance*>(blas.getSpatialMap().getDataAtTransform(firstInstanceObjectToWorld));
      const bool exactHit = (result != nullptr);
      if (result != nullptr) {
        if (isProbeVS) {
          thread_local uint32_t sExactProbe = 0;
          if (sExactProbe < 8 || (sExactProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[FindSim2904] #", sExactProbe,
              " stage=exact hit=1 spatialMapSize=", blas.getSpatialMap().size()));
          }
          sExactProbe += 1;
        }
        return result;
      }

      // No exact match, so find the closest match in the region
      // (need to check a 2x2x2 patch of cells to account for positions close to a border)
      // Probe: count how many instances passed/failed each filter clause.
      uint32_t probeNumCellInstances = 0;
      uint32_t probeRejFrameUpdated  = 0;
      uint32_t probeRejMaterialHash  = 0;
      uint32_t probeRejSubPrim       = 0;
      uint32_t probePassedFilter     = 0;
      result = const_cast<RtInstance*>(blas.getSpatialMap().getNearestData(worldPosition, uniqueObjectDistanceSqr, nearestDistSqr,
        [&] (const RtInstance* instance) {
          // Filter out instances by returning false if the instance:
          // - has already been updated this frame
          // - doesn't use the same material
          // - is a sub prim of a replacement instance
          if (isProbeVS) {
            probeNumCellInstances += 1;
            const bool okFrame = (instance->m_frameLastUpdated != currentFrameIdx);
            const bool okMat   = (instance->m_materialHash == material.getHash());
            const bool okSub   = !instance->m_primInstanceOwner.isSubPrim();
            if (!okFrame) probeRejFrameUpdated += 1;
            if (!okMat)   probeRejMaterialHash += 1;
            if (!okSub)   probeRejSubPrim      += 1;
            if (okFrame && okMat && okSub) probePassedFilter += 1;
            return okFrame && okMat && okSub;
          }
          return instance->m_frameLastUpdated != currentFrameIdx && instance->m_materialHash == material.getHash() && !instance->m_primInstanceOwner.isSubPrim();
        }
      ));
      if (isProbeVS) {
        thread_local uint32_t sNearestProbe = 0;
        if (sNearestProbe < 8 || (sNearestProbe & 0x3FF) == 0) {
          Logger::info(str::format(
            "[FindSim2904] #", sNearestProbe,
            " stage=nearest exactHit=0 hit=", (result != nullptr ? 1 : 0),
            " nearestDistSqr=", nearestDistSqr,
            " maxDistSqr=", uniqueObjectDistanceSqr,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " cellInsts=", probeNumCellInstances,
            " rejFrame=", probeRejFrameUpdated,
            " rejMat=", probeRejMaterialHash,
            " rejSub=", probeRejSubPrim,
            " passed=", probePassedFilter,
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")",
            " curMatHash=0x", std::hex, material.getHash(), std::dec));
        }
        sNearestProbe += 1;
      }
      if (nearestDistSqr == 0.0f && result != nullptr) {
        // Not going to find anything closer
        return result;
      }
    }

    // For portal gun and other objects that were drawn in the ViewModel, need to check the
    // virtual version of the instance from previous frame.
    if (nearestDistSqr > 0.0f &&
        cameraType == CameraType::ViewModel && 
        RtxOptions::useRayPortalVirtualInstanceMatching() ) {
      const Matrix4* teleportMatrix = nullptr;
      for (const RtInstance* instance : blas.getLinkedInstances()) {
        if (instance->m_frameLastUpdated != currentFrameIdx - 1 || 
            instance->m_materialHash != material.getHash()) {
          continue;
        }
        
        // Compare against virtual position of a predicted instance's position in the current frame
        const Vector3& prevPrevInstanceWorldPosition = instance->getPrevWorldPosition();
        const Vector3& prevInstanceWorldPosition = instance->getWorldPosition();
        const Vector3 predictedInstanceWorldPosition = prevInstanceWorldPosition +
          (prevInstanceWorldPosition - prevPrevInstanceWorldPosition);
      
        // Check all portal pairs
        for (auto& rayPortalPair : rayPortalManager.getRayPortalPairInfos()) {
          if (rayPortalPair.has_value()) {
            for (uint32_t i = 0; i < 2; i++) {
              const auto& rayPortal = rayPortalPair->pairInfos[i];

              const Vector3 virtualPredictedInstanceWorldPosition =
                rayPortalManager.getVirtualPosition(predictedInstanceWorldPosition, rayPortal.portalToOpposingPortalDirection);

              // Distance of the object from the predicted virtual position of an instance
              const float virtualDistSqr = lengthSqr(virtualPredictedInstanceWorldPosition - worldPosition);

              // Is the instance is similar, and within range?  We already know the BLAS is shared, due to the for loop
              if (virtualDistSqr <= uniqueObjectDistanceSqr && virtualDistSqr < nearestDistSqr) {
                nearestDistSqr = virtualDistSqr;
                result = const_cast<RtInstance*>(instance);
                teleportMatrix = &rayPortal.portalToOpposingPortalDirection;
                if (virtualDistSqr == 0.0f) {
                  // Not going to find anything closer.
                  break;
                }
              }
            }
          }
        }
      }
      
      // If the match was against a virtual equivalent of the instance from previous frame, 
      // update the instance's transform to that of the virtual one
      if (teleportMatrix) {
        result->teleportWithHistory(*teleportMatrix);
      }
    }


    return result; 
  }

  RtInstance* InstanceManager::addInstance(BlasEntry& blas) {
    const uint32_t currentFrameIdx = m_device->getCurrentFrameId();

    const uint32_t instanceIdx = m_instances.size();
    RtInstance* newInst = new RtInstance(m_nextInstanceId++, instanceIdx);
    m_instances.push_back(newInst);

    RtInstance* currentInstance = m_instances[instanceIdx];

    currentInstance->m_frameCreated = currentFrameIdx;
    
    // Set Instance Vulkan AS Instance information
    {
      currentInstance->m_vkInstance.mask = 0;
      currentInstance->m_vkInstance.flags = 0;
      currentInstance->m_vkInstance.instanceCustomIndex = 0;
      currentInstance->m_vkInstance.instanceShaderBindingTableRecordOffset = 0;
      currentInstance->setBlas(blas);
    }

    // Rest of the setup happens in updateInstance()

    // Notify events after instance has been added
    for (auto& event : m_eventHandlers)
      event.onInstanceAddedCallback(*currentInstance);

    // onInstanceAddedCallback will link current instance to the BLAS
    currentInstance->m_isUnlinkedForGC = false;

    return currentInstance;
  }

  // Creates a copy of an instance
  // If the copy is temporary and is not tracked via callbacks/externally, it doesn't need
  // a valid unique instance ID. In that case, set generateValidID to false to avoid overflowing the ID value
  RtInstance* InstanceManager::createInstanceCopy(const RtInstance& reference, bool generateValidID) {

    const uint32_t instanceIdx = m_instances.size();

    uint64_t id = generateValidID ? m_nextInstanceId++ : kInvalidInstanceId;
    RtInstance* newInstance = new RtInstance(reference, id, instanceIdx);
    newInstance->m_isCreatedByRenderer = true;
    m_instances.push_back(newInstance);

    return newInstance;
  }

  void InstanceManager::processInstanceBuffers(const BlasEntry& blas, RtInstance& currentInstance) const {
    currentInstance.surface.positionBufferIndex = blas.modifiedGeometryData.positionBufferIndex;
    currentInstance.surface.positionOffset = blas.modifiedGeometryData.positionBuffer.offsetFromSlice();
    currentInstance.surface.positionStride = blas.modifiedGeometryData.positionBuffer.stride();
    currentInstance.surface.normalBufferIndex = blas.modifiedGeometryData.normalBufferIndex;
    currentInstance.surface.normalOffset = blas.modifiedGeometryData.normalBuffer.offsetFromSlice();
    currentInstance.surface.normalStride = blas.modifiedGeometryData.normalBuffer.stride();
    currentInstance.surface.normalFormat = blas.modifiedGeometryData.normalBuffer.vertexFormat();
    currentInstance.surface.color0BufferIndex = blas.modifiedGeometryData.color0BufferIndex;
    currentInstance.surface.color0Offset = blas.modifiedGeometryData.color0Buffer.offsetFromSlice();
    currentInstance.surface.color0Stride = blas.modifiedGeometryData.color0Buffer.stride();
    currentInstance.surface.texcoordBufferIndex = blas.modifiedGeometryData.texcoordBufferIndex;
    currentInstance.surface.texcoordOffset = blas.modifiedGeometryData.texcoordBuffer.offsetFromSlice();
    currentInstance.surface.texcoordStride = blas.modifiedGeometryData.texcoordBuffer.stride();
    // NV-DXVK: derive texcoordEncoding from the BUFFER's actual vertex
    // format here — the ground truth of how the bytes are stored.
    // Previously the encoding was set later from
    // drawCall.getTransformData().texcoordEncoding (VS-ISGN-derived).
    // That broke when the same BLAS was referenced by draws of multiple
    // VS classes (geometry-hash dedup): the BLAS's texcoord buffer is
    // built from one VS's VB, but later draws with a different VS class
    // would overwrite the encoding to a value that doesn't match the
    // buffer. Probe 8 confirmed surfaces ending up with stride=20
    // (R32G32_FLOAT layout) AND encoding=1 (TF2BspUintPacked decode) —
    // an impossible combination producing 8000-range garbage UVs from
    // float-bit reinterpretation. Tying encoding to buffer.vertexFormat()
    // makes the two always consistent because they come from the same
    // source.
    const VkFormat tcFmt = blas.modifiedGeometryData.texcoordBuffer.defined()
        ? blas.modifiedGeometryData.texcoordBuffer.vertexFormat()
        : VK_FORMAT_UNDEFINED;
    currentInstance.surface.texcoordEncoding =
        (tcFmt == VK_FORMAT_R32G32_UINT || tcFmt == VK_FORMAT_R32G32_SINT)
        ? RtSurface::TexcoordEncoding::TF2BspUintPacked
        : RtSurface::TexcoordEncoding::Float;
    // NV-DXVK: propagate lightmap-UV presence so surface_interaction can
    // read TEXCOORD1 from the second 8-byte slot the interleaver wrote
    // alongside TEXCOORD0. The flag is the single source of truth for
    // both the per-hit interpolation gate and the material code's
    // lightmap-sampler UV lookup.
    currentInstance.surface.hasLightmap = blas.modifiedGeometryData.hasTexcoord1;
    // NV-DXVK: TF2 worldspace VGUI extras — the slang VGUI evaluator reads
    // 8 floats per vertex starting at element index vguiOffset. Both fields
    // were stamped by RtxGeometryUtils::processGeometryBuffers (slow path)
    // when input.vguiLayoutEnable was true; a non-VGUI surface sees
    // hasVgui=false and the dispatch in opaque_surface_material_interaction
    // is skipped.
    currentInstance.surface.isVgui = blas.modifiedGeometryData.hasVgui;
    currentInstance.surface.vguiOffset = blas.modifiedGeometryData.vguiOffset;
    // NV-DXVK: 3 VGUI structured-buffer bindless indices. Truncating to
    // uint16_t is safe because BindlessResourceManager::kMaxBindlessResources
    // is 64K (matches uint16_t range). kSurfaceInvalidBufferIndex (UINT32_MAX)
    // truncates to 0xFFFF, which the slang side compares against
    // BINDING_INDEX_INVALID(0xFFFF) — see surface_interaction.slangh:853 for
    // the same convention applied to indexBufferIndex.
    currentInstance.surface.vguiFontBoundsBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiFontBoundsBufferIndex);
    currentInstance.surface.vguiImgBoundsBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiImgBoundsBufferIndex);
    currentInstance.surface.vguiStylesBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiStylesBufferIndex);
    {
      static std::unordered_set<uint64_t> sLoggedVguiSurfaceKeys;
      static std::mutex sLoggedVguiSurfaceMu;
      if (currentInstance.surface.isVgui) {
        const uint64_t key =
            (uint64_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFFu)
          | ((uint64_t(currentInstance.surface.vguiOffset) & 0xFFFFu) << 16);
        bool firstSeen = false;
        {
          std::lock_guard<std::mutex> lk(sLoggedVguiSurfaceMu);
          firstSeen = sLoggedVguiSurfaceKeys.insert(key).second;
        }
        if (firstSeen) {
          Logger::info(str::format(
            "[VguiSurface] isVgui=1",
            " texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
            " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
            " texStride=", uint32_t(currentInstance.surface.texcoordStride),
            " vguiOffset(floatUnits)=", uint32_t(currentInstance.surface.vguiOffset),
            " — VGUI evaluator reads (vguiOffset .. vguiOffset+8)"));
        }
      }
    }
    // Log unique (texBufIdx, texOffset, texStride, hasLightmap) tuples so
    // we can confirm the lightmap-UV flag actually lands on instances.
    // [TC1Detect] (IA capture) and [TC1Interleave] (interleaver dispatch)
    // are the upstream confirmations; this is the per-instance one. If
    // upstream logs fire but this one shows hasLightmap=0 across the
    // board, the propagation between RaytraceGeometry and the surface
    // metadata is broken.
    {
      static std::unordered_set<uint64_t> sLoggedTc1SurfaceKeys;
      static std::mutex sLoggedTc1SurfaceMu;
      const uint64_t key =
          (uint64_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFFu)
        | ((uint64_t(currentInstance.surface.texcoordOffset) & 0xFFu) << 16)
        | ((uint64_t(currentInstance.surface.texcoordStride) & 0xFFu) << 24)
        | ((uint64_t(currentInstance.surface.hasLightmap ? 1u : 0u)) << 32);
      bool firstSeen = false;
      {
        std::lock_guard<std::mutex> lk(sLoggedTc1SurfaceMu);
        firstSeen = sLoggedTc1SurfaceKeys.insert(key).second;
      }
      if (firstSeen) {
        Logger::info(str::format(
          "[TC1Surface] hasLightmap=", (currentInstance.surface.hasLightmap ? 1 : 0),
          " texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
          " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
          " texStride=", uint32_t(currentInstance.surface.texcoordStride),
          " — lightmap UV (when set) read at element index (texOffset/4 + 2)"));
      }
    }
    currentInstance.surface.previousPositionBufferIndex = blas.modifiedGeometryData.previousPositionBufferIndex;
    currentInstance.surface.indexBufferIndex = blas.modifiedGeometryData.indexBufferIndex;
    currentInstance.surface.indexStride = blas.modifiedGeometryData.indexBuffer.stride();

    // NV-DXVK: per-instance buffer-layout dump for diagnosing the BSP-wall
    // degenerate-UV bug (probes confirmed t1==t2 in UV-space on a real-area
    // world triangle). If two adjacent triangles on the SAME instance show
    // wildly different texcoord layouts, or if texcoordStride doesn't match
    // posStride for known interleaved sources, the interleaver is feeding
    // wrong data. Logged once per (texBufIdx, texStride, texOffset, idxStride)
    // tuple to avoid spam.
    {
      const uint32_t key =
        (uint32_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFu)
        | ((uint32_t(currentInstance.surface.texcoordStride) & 0xFFu) << 12)
        | ((uint32_t(currentInstance.surface.texcoordOffset) & 0xFFu) << 20)
        | ((uint32_t(currentInstance.surface.indexStride) & 0xFu) << 28);
      static std::unordered_set<uint32_t> seenBlasGeom;
      if (seenBlasGeom.insert(key).second) {
        Logger::info(str::format(
          "[BlasGeom] texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
          " texStride=", uint32_t(currentInstance.surface.texcoordStride),
          " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
          " posBufIdx=", uint32_t(currentInstance.surface.positionBufferIndex),
          " posStride=", uint32_t(currentInstance.surface.positionStride),
          " posOffset=", uint32_t(currentInstance.surface.positionOffset),
          " idxBufIdx=", uint32_t(currentInstance.surface.indexBufferIndex),
          " idxStride=", uint32_t(currentInstance.surface.indexStride),
          " normBufIdx=", uint32_t(currentInstance.surface.normalBufferIndex),
          " normStride=", uint32_t(currentInstance.surface.normalStride)));
      }
    }
  }

  // Returns true if the instance was modified
  bool InstanceManager::applyDeveloperOptions(RtInstance& currentInstance, const DrawCallState& drawCall) {
    if (!RtxOptions::enableInstanceDebuggingTools()) {
      return false;
    }

    if ((
      currentInstance.m_instanceVectorId >= RtxOptions::instanceOverrideInstanceIdx() &&
      currentInstance.m_instanceVectorId < RtxOptions::instanceOverrideInstanceIdx() + RtxOptions::instanceOverrideInstanceIdxRange())) {

      if (RtxOptions::instanceOverrideSelectedInstancePrintMaterialHash())
        Logger::info(str::format("Draw Call Material Hash: ", drawCall.getMaterialData().getHash()));

      // Apply world offset
      Vector3 worldOffset = RtxOptions::instanceOverrideWorldOffset();
      currentInstance.teleportWithHistory(translationMatrix(worldOffset));

      return true;
    }

    return false;
  }

  void InstanceManager::bindMaterial(RtInstance& instance, const RtSurfaceMaterial& material) {
    if (material.getType() == RtSurfaceMaterialType::Opaque) {
      instance.m_albedoOpacityTextureIndex = material.getOpaqueSurfaceMaterial().getAlbedoOpacityTextureIndex();
      instance.m_samplerIndex = material.getOpaqueSurfaceMaterial().getSamplerIndex();
    } else if (material.getType() == RtSurfaceMaterialType::RayPortal) {
      instance.m_albedoOpacityTextureIndex = material.getRayPortalSurfaceMaterial().getMaskTextureIndex();
      instance.m_samplerIndex = material.getRayPortalSurfaceMaterial().getSamplerIndex();
      instance.m_secondaryOpacityTextureIndex = material.getRayPortalSurfaceMaterial().getMaskTextureIndex2();
      instance.m_secondarySamplerIndex = material.getRayPortalSurfaceMaterial().getSamplerIndex2();
    }

    instance.m_vkInstance.instanceCustomIndex = (instance.m_vkInstance.instanceCustomIndex & ~(surfaceMaterialTypeMask << CUSTOM_INDEX_MATERIAL_TYPE_BIT));
    instance.m_vkInstance.instanceCustomIndex |= ((uint32_t)material.getType() << CUSTOM_INDEX_MATERIAL_TYPE_BIT);

    // Fetch the material from the cache
    m_pResourceCache->find(material, instance.surface.surfaceMaterialIndex);
  }

  // Updates the state of the instance with the draw call inputs
  // It handles multiple draw calls called for a same instance within a frame
  // To be called on every draw call
  void InstanceManager::updateInstance(RtInstance& currentInstance,
                                       const CameraManager& cameraManager,
                                       const BlasEntry& blas,
                                       const DrawCallState& drawCall,
                                       MaterialData& materialData) {
    currentInstance.m_categoryFlags = drawCall.getCategoryFlags();
    currentInstance.surface.instancesToObject = drawCall.getTransformData().instancesToObject;
    // NV-DXVK: Couple lifetime of the transform vector to this RtInstance so the
    // raw pointer above cannot dangle once the draw-call source's own ring-buffer
    // releases its reference. Null for sources with externally-owned storage
    // (USD replacements, etc.) — that's fine, they already manage lifetime.
    currentInstance.surface.instancesToObjectOwner = drawCall.getTransformData().instancesToObjectOwner;

    // setFrameLastUpdated() must be called first as it resets instance's state on a first call in a frame
    const bool isFirstUpdateThisFrame = currentInstance.setFrameLastUpdated(m_device->getCurrentFrameId());

    // These can change in the Runtime UI so need to check during update
    currentInstance.m_isHidden = currentInstance.testCategoryFlags(InstanceCategories::Hidden);
    currentInstance.m_isPlayerModel = currentInstance.testCategoryFlags(InstanceCategories::ThirdPersonPlayerModel);
    currentInstance.m_isWorldSpaceUI = currentInstance.testCategoryFlags(InstanceCategories::WorldUI);

    // Hide the sky instance since it is not raytraced.
    // Sky mesh and material are only good for capture and replacement purposes.
    if (drawCall.cameraType == CameraType::Sky) {
      currentInstance.m_isHidden = true;
    }

    // Register camera
    bool isNewCameraSet = currentInstance.registerCamera(drawCall.cameraType, m_device->getCurrentFrameId());

    const bool overridePreviousCameraUpdate = isNewCameraSet &&
      (drawCall.cameraType == CameraType::Main ||
       // Don't overwrite transform from when the instance was seen with the main camera
       !currentInstance.isCameraRegistered(CameraType::Main));

    const RtSurface::AlphaState alphaState = calculateAlphaState(drawCall, materialData);
    bool hasTransformChanged = false;
    bool hasPreviousPositions = false;

    if (!isFirstUpdateThisFrame) {
      // This is probably the same instance, being drawn twice!  Merge it
      mergeInstanceHeuristics(currentInstance, drawCall, alphaState);
    }
    
    // Updates done only once a frame unless overriden due to an explicit state
    if (isFirstUpdateThisFrame || overridePreviousCameraUpdate) {

      if (isFirstUpdateThisFrame) {
        processInstanceBuffers(blas, currentInstance);

        currentInstance.m_materialType = materialData.getType();

        const XXH64_hash_t materialInstanceHash = materialData.getHash();
        currentInstance.m_materialDataHash = drawCall.getMaterialData().getHash();
        currentInstance.surface.hasMaterialChanged = currentInstance.m_materialHash != kEmptyHash && currentInstance.m_materialHash != materialInstanceHash;
        currentInstance.m_materialHash = materialInstanceHash;

        currentInstance.m_texcoordHash = drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord];
        currentInstance.m_indexHash = drawCall.getGeometryData().hashes[HashComponents::Indices];

        // Surface meta data
        currentInstance.surface.isEmissive = false;
        currentInstance.surface.isMatte = false;
        currentInstance.surface.textureColorArg1Source = drawCall.getMaterialData().textureColorArg1Source;
        currentInstance.surface.textureColorArg2Source = drawCall.getMaterialData().textureColorArg2Source;
        currentInstance.surface.textureColorOperation = drawCall.getMaterialData().textureColorOperation;
        currentInstance.surface.textureAlphaArg1Source = drawCall.getMaterialData().textureAlphaArg1Source;
        currentInstance.surface.textureAlphaArg2Source = drawCall.getMaterialData().textureAlphaArg2Source;
        currentInstance.surface.textureAlphaOperation = drawCall.getMaterialData().textureAlphaOperation;
        currentInstance.surface.texgenMode = drawCall.getTransformData().texgenMode; // NOTE: Make it material data...
        // NV-DXVK: texcoordEncoding intentionally NOT set from drawCall
        // here. processInstanceBuffers() above derives it from the BLAS's
        // texcoord buffer.vertexFormat() — the only source consistent
        // with the actual buffer bytes. See the SurfaceEncMismatch fix
        // log in writeGPUData for the bug history. Per-draw assignment
        // here used to clobber the correct value when the same BLAS was
        // referenced by mixed-VS-class draws.
        currentInstance.surface.tFactor = drawCall.getMaterialData().tFactor;
        currentInstance.surface.alphaState = alphaState;
        currentInstance.surface.isAnimatedWater = currentInstance.testCategoryFlags(InstanceCategories::AnimatedWater);
        currentInstance.surface.associatedGeometryHash = drawCall.getHash(RtxOptions::geometryAssetHashRule());
        currentInstance.surface.isTextureFactorBlend = drawCall.getMaterialData().isTextureFactorBlend;
        currentInstance.surface.isVertexColorBakedLighting = drawCall.getMaterialData().isVertexColorBakedLighting;
        currentInstance.surface.isMotionBlurMaskOut = currentInstance.testCategoryFlags(InstanceCategories::IgnoreMotionBlur);
        currentInstance.surface.ignoreTransparencyLayer = currentInstance.testCategoryFlags(InstanceCategories::IgnoreTransparencyLayer);

        // Note: Skip the spritesheet adjustment logic in the surface interaction when using Ray Portal materials as this logic
        // is done later in the Surface Material Interaction (and doing it in both places will just double up the animation).
        currentInstance.surface.skipSurfaceInteractionSpritesheetAdjustment = (currentInstance.m_materialType == MaterialDataType::RayPortal);
        currentInstance.surface.isInsideFrustum = RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? currentInstance.m_isInsideFrustum : true;

        currentInstance.surface.blendModeState = drawCall.getMaterialData().blendMode;

        if (drawCall.isEye()) {
          // assume that the texture transform has eye parameters encoded
          const Matrix4& texTransform = drawCall.getTransformData().textureTransform;
          RtEyeParams eyeParams{};
          eyeParams.eyeballOrigin = Vector3{ texTransform.data[0].w, texTransform.data[1].w, texTransform.data[2].w };
          eyeParams.eyeRightU = Vector3{ texTransform.data[0].x, texTransform.data[1].x, texTransform.data[2].x };
          eyeParams.eyeUpV = Vector3{ texTransform.data[0].y, texTransform.data[1].y, texTransform.data[2].y };
          currentInstance.surface.eyeParams = eyeParams;
        }

        uint8_t spriteSheetRows = 0, spriteSheetCols = 0, spriteSheetFPS = 0;
        materialData.getSpriteSheetData(currentInstance.surface.spriteSheetRows, currentInstance.surface.spriteSheetCols, currentInstance.surface.spriteSheetFPS);
        currentInstance.m_isAnimated = currentInstance.surface.spriteSheetFPS != 0;
        currentInstance.surface.objectPickingValue = drawCall.drawCallID;

        // Note: Extract spritesheet information from the associated material data as it ends up stored in the Surface
        // not in the Surface Material like most material information.
        switch (materialData.getType()) {
        case MaterialDataType::Opaque:
        {
          spriteSheetRows = materialData.getOpaqueMaterialData().getSpriteSheetRows();
          spriteSheetCols = materialData.getOpaqueMaterialData().getSpriteSheetCols();
          spriteSheetFPS = materialData.getOpaqueMaterialData().getSpriteSheetFPS();

          const bool useLegacyAlphaState = materialData.getOpaqueMaterialData().getUseLegacyAlphaState();

          // NV-DXVK: TF2 worldspace VGUI/HUD shaders are inherently unlit —
          // the PS writes the final composed UI color directly. Setting
          // surface.isMatte=true makes the slang shader zero out albedo
          // and baseReflectivity, leaving only the emissive contribution
          // (the picked color texture forwarded by LegacyMaterialData::
          // as<OpaqueMaterialData>()) so the UI texture is rendered as-is
          // without world lighting. Detection happens upstream in
          // d3d11_rtx.cpp::FillMaterialData via PS RDEF resource names.
          if (materialData.getOpaqueMaterialData().getIsUnlitOutput()) {
            currentInstance.surface.isMatte = true;
          }

          // NV-DXVK: emission is now driven by the PS's own CBuffer signal
          // (CBufUberStatic.c_emissiveTint marked D3D_SVF_USED, read in
          // D3D11Rtx::FillMaterialData and forwarded through LegacyMaterial
          // Data::as<OpaqueMaterialData>()). The previous heuristics that
          // lived here — promoting on `m_isWorldSpaceUI` or on detected
          // emissive-blend Vulkan blend states — were both wrong-by-
          // construction in the absence of the ground-truth signal: the
          // blend-state path mis-classified TF2 refract/water/decal layers
          // as light sources (24/32 distinct materials in observed logs),
          // producing the "lighting bouncing off surfaces" bug.
          //
          // Worldspace UI is left unhandled here intentionally — TF2 doesn't
          // categorise any draws as InstanceCategories::WorldUI in this
          // fork (verified via [EmissivePromote.WorldUI] zero hits across
          // multiple gameplay sessions). If a future map does, the right
          // place to resurrect that promotion is alongside the PS-CB read,
          // not as a pre-material-data mutation. Log a one-shot if it
          // ever fires so we notice.
          if (currentInstance.m_isWorldSpaceUI) {
            const XXH64_hash_t matHash = drawCall.getMaterialData().getHash();
            static std::unordered_set<XXH64_hash_t> sUiSeenWarn;
            if (sUiSeenWarn.insert(matHash).second) {
              Logger::warn(str::format(
                "[EmissivePromote.WorldUI.Unhandled] matHash=0x",
                std::hex, matHash, std::dec,
                " — TF2 fork no longer auto-promotes WorldUI to emissive;"
                " if this material should glow, route via PS CBuffer signal"
                " (see d3d11_rtx.cpp::FillMaterialData)."));
            }
          }

          currentInstance.m_isSubsurface = materialData.getOpaqueMaterialData().getSubsurfaceDiffusionProfile();

          break;
        }
        case MaterialDataType::Translucent:
          spriteSheetRows = materialData.getTranslucentMaterialData().getSpriteSheetRows();
          spriteSheetCols = materialData.getTranslucentMaterialData().getSpriteSheetCols();
          spriteSheetFPS = materialData.getTranslucentMaterialData().getSpriteSheetFPS();

          break;
        case MaterialDataType::RayPortal:
          spriteSheetRows = materialData.getRayPortalMaterialData().getSpriteSheetRows();
          spriteSheetCols = materialData.getRayPortalMaterialData().getSpriteSheetCols();
          spriteSheetFPS = materialData.getRayPortalMaterialData().getSpriteSheetFPS();

          break;
        case MaterialDataType::Count:
        case MaterialDataType::Invalid:
          assert(0);
          break;
        }
      }

      // Update transform
      {
        // Heuristic for MS5 - motion vectors on translucent surfaces cannot be trusted.  This will help with IQ, but need a longer term solution [TREX-634]
        const bool isMotionUnstable = currentInstance.m_materialType == MaterialDataType::Translucent
                                   || currentInstance.testCategoryFlags(InstanceCategories::Particle)
                                   || currentInstance.testCategoryFlags(InstanceCategories::WorldUI);

        hasPreviousPositions = blas.modifiedGeometryData.previousPositionBuffer.defined() && !isMotionUnstable;
        const bool isFirstUpdateAfterCreation = currentInstance.isCreatedThisFrame(m_device->getCurrentFrameId()) && isFirstUpdateThisFrame;

        // Note: objectToView is aliased on updates, since findSimilarInstance() doesn't discern it
        Matrix4 objectToWorld = drawCall.getTransformData().objectToWorld;

        // Hack for TREX-2272. In Portal, in the GLaDOS chamber, the monitors show a countdown timer with background, and the digits and background are coplanar.
        // We cannot reliably determine the digits material because it's a dynamic texture rendered by vgui that contains all kinds of UI things.
        // So instead of offsetting the digits or making them live in unordered TLAS (either of which would solve the problem), we offset the screen background backwards.
        const float worldSpaceUiBackgroundOffset = RtxOptions::worldSpaceUiBackgroundOffset();
        if (worldSpaceUiBackgroundOffset != 0.f && currentInstance.testCategoryFlags(InstanceCategories::WorldMatte)) {
          objectToWorld[3] += objectToWorld[2] * worldSpaceUiBackgroundOffset;
        }

        // Update the transform based on what state we're in
        if (isFirstUpdateAfterCreation) {
          hasTransformChanged = currentInstance.teleport(objectToWorld);
        } else if (isFirstUpdateThisFrame) {
          hasTransformChanged = currentInstance.move(objectToWorld);
        } else {
          hasTransformChanged = currentInstance.moveAgain(objectToWorld);
        }

        currentInstance.surface.textureTransform = drawCall.getTransformData().textureTransform;
        // NV-DXVK: log ONE entry per (hash-of-matrix, vsHash) combo so we see
        // every distinct transform any draw uses, paired with the VS hash so
        // we can correlate to wall draws. Key: we want to know the scale on
        // the WALL surfaces specifically — runtime probe sees uvLen ~1000
        // post-transform; if the wall transform is identity then the runtime
        // value IS what the BSP+VS-decode produces; if it's a strong scale,
        // the source decode is at a different magnitude.
        {
          const auto& m = currentInstance.surface.textureTransform;
          const bool isIdent = (m == Matrix4());
          // Hash of the four 2D-relevant entries (col0.x, col0.y, col1.x, col1.y, col3.x, col3.y)
          uint64_t key = 0;
          auto hashF = [&key](float v) {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            key = key * 0x100000001b3ull ^ uint64_t(bits);
          };
          hashF(m.data[0].x); hashF(m.data[0].y);
          hashF(m.data[1].x); hashF(m.data[1].y);
          hashF(m.data[3].x); hashF(m.data[3].y);
          // Combine with the geometry's texcoord hash so per-mesh-distinct
          // transforms log separately. Texcoord hash is a stable identifier
          // for which mesh's UVs we're seeing (same mesh = same hash).
          const uint64_t txcHash = uint64_t(currentInstance.m_texcoordHash);
          const uint64_t comboKey = key ^ (txcHash * 0x9E3779B97F4A7C15ull);
          static std::unordered_set<uint64_t> seen;
          if (seen.insert(comboKey).second) {
            Logger::info(str::format(
              "[RTX-InstMgr.UVx] txcHash=0x", std::hex, txcHash, std::dec,
              " ident=", isIdent ? "1" : "0",
              " col0=(", m.data[0].x, ",", m.data[0].y, ")",
              " col1=(", m.data[1].x, ",", m.data[1].y, ")",
              " col3=(", m.data[3].x, ",", m.data[3].y, ")"));
          }
        }

        currentInstance.surface.isStatic = !(hasTransformChanged || hasPreviousPositions) || currentInstance.m_materialType == MaterialDataType::RayPortal;

        currentInstance.surface.isClipPlaneEnabled = drawCall.getTransformData().enableClipPlane;
        currentInstance.surface.clipPlane = drawCall.getTransformData().clipPlane;

        // Apply developer options
        if (isFirstUpdateThisFrame)
          applyDeveloperOptions(currentInstance, drawCall);
      }
    }

    // We only have 1 hit shader.
    currentInstance.m_vkInstance.instanceShaderBindingTableRecordOffset = 0;

    // Update instance flags.
    // Note: this should happen on instance updates and not creation because the same geometry can be drawn
    // with different flags, and the instance manager can match an old instance of a geometry to a new one with different draw mode.
    currentInstance.m_vkInstance.flags = determineInstanceFlags(drawCall, currentInstance.surface);
    currentInstance.isFrontFaceFlipped = (currentInstance.m_vkInstance.flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR) != 0;

    // Apply the decal sort index for this instance so we can approximate order correctness on the GPU in AHS
    if (currentInstance.surface.alphaState.isDecal) {
      currentInstance.surface.decalSortOrder = m_decalSortOrderCounter++;
#if !NDEBUG
      if (m_decalSortOrderCounter > 255) {
        ONCE(Logger::err("Too many decals in this scene to sort correctly, may see some decal corruption issues."));
      }
#endif
    }

    // Update the geometry and instance flags
    if (currentInstance.isOpaque() && drawCall.isUsingRaytracedRenderTarget) {
      // render target texture - need this to be in the opaque pass, even if alphaState.isFullyOpaque is false.
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    } else if (
      (!currentInstance.surface.alphaState.isFullyOpaque && currentInstance.surface.alphaState.isParticle) ||
      (currentInstance.surface.alphaState.isDecal) ||
      // Note: include alpha blended geometry on the player model into the unordered TLAS. This is hacky as there might be
      // suitable geometry outside of the player model, but we don't have a way to distinguish it from alpha blended geometry
      // that should be alpha tested instead, like some metallic stairs in Portal -- those should be resolved normally.
      (!currentInstance.surface.alphaState.isFullyOpaque && !currentInstance.surface.alphaState.isBlendingDisabled && currentInstance.m_isPlayerModel) ||
      currentInstance.surface.alphaState.emissiveBlend
    ) {
      // Alpha-blended and emissive particles go to the separate "unordered" TLAS as non-opaque geometry
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      currentInstance.m_isUnordered = true;
      // Unordered resolve only accumulates via any-hits and ignores opaque hits, therefore force 
      // the opaque hits resolve via OMMs to be turned into any-hits.
      // Note: this has unexpected effect even with OMM off and results in minor visual changes in Portal MF A DLSS test
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque && currentInstance.surface.alphaState.isBlendingDisabled) {
      // Alpha-tested geometry goes to the primary TLAS as non-opaque geometry with potential duplicate hits.
      currentInstance.m_geometryFlags = 0;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque) {
      // Alpha-blended geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      // Treat all non-transparent hits as any-hits
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::Translucent) {
      // Translucent (e.g. glass) geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
      // Portals go to the primary TLAS as opaque.
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    } else if (currentInstance.surface.isClipPlaneEnabled) {
      // Use non-opaque hits to process clip planes on visibility rays.
      // To handle cases when the same *static* object is used both with and without clip planes,
      // use the force bit to avoid BLAS confusion (because the geometry flags are baked into BLAS).
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else {
      // All other fully opaques go to the primary TLAS as opaque.
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    }
    
    // Enable backface culling for Portals to avoid additional hits to the back of Portals
    if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
      currentInstance.m_vkInstance.flags &= ~VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    // Update mask
    {
      uint mask = isFirstUpdateThisFrame ? 0 : currentInstance.m_vkInstance.mask;

      if (currentInstance.m_isPlayerModel && drawCall.cameraType != CameraType::ViewModel) {
        mask |= OBJECT_MASK_PLAYER_MODEL;
        // Lazy-clear stale instances if onFrameEnd() was skipped last frame (e.g. device loss on alt+tab)
        const uint32_t currentFrameId = m_device->getCurrentFrameId();
        if (m_playerModelInstancesFrameId != currentFrameId) {
          m_playerModelInstances.clear();
          m_playerModelInstancesFrameId = currentFrameId;
        }
        m_playerModelInstances.push_back(&currentInstance);
      } else {
        currentInstance.m_isPlayerModel = false;
        if (currentInstance.m_isUnordered && RtxOptions::enableSeparateUnorderedApproximations()) {
          if (currentInstance.surface.alphaState.isDecal) {
            mask = OBJECT_MASK_UNORDERED_ALL_BLENDED;
          } else {
            // Separate set of mask bits for the unordered TLAS
            if (currentInstance.surface.alphaState.emissiveBlend)
              mask |= OBJECT_MASK_UNORDERED_ALL_EMISSIVE;
            else
              mask |= OBJECT_MASK_UNORDERED_ALL_BLENDED;
          }
        }
        else {
          if (currentInstance.m_materialType == MaterialDataType::Translucent) {
            // Translucent material
            mask |= OBJECT_MASK_TRANSLUCENT;
          } else if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
            // Portal
            mask |= OBJECT_MASK_PORTAL;
          } else {
            mask |= currentInstance.surface.alphaState.isBlendingDisabled ? OBJECT_MASK_OPAQUE : OBJECT_MASK_ALPHA_BLEND;
          }
        }
      }

      if (currentInstance.m_isHidden)
        mask = 0;

      currentInstance.m_vkInstance.mask = mask;
    }
    // This flag translates to a flip of VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR when the instance
    // is a separate BLAS instance, and to nothing if it's a part of a merged BLAS.
    // The reason is in this bit of Vulkan spec:
    //     VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR indicates that the facing determination for geometry in this instance
    //     is inverted. Because the facing is determined in object space, an instance transform does not change the winding,
    //     but a geometry transform does.
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGeometryInstanceFlagBitsNV.html 
    currentInstance.m_isObjectToWorldMirrored = isMirrorTransform(drawCall.getTransformData().objectToWorld);

    bool billboardsGotGenerated = false;
    currentInstance.m_billboardCount = 0;
    
    if (drawCall.cameraType == CameraType::ViewModel && !currentInstance.m_isHidden && isFirstUpdateThisFrame) {
      // Lazy-clear stale candidates if onFrameEnd() was skipped last frame (e.g. device loss on alt+tab)
      const uint32_t currentFrameId = m_device->getCurrentFrameId();
      if (m_viewModelCandidatesFrameId != currentFrameId) {
        m_viewModelCandidates.clear();
        m_viewModelCandidatesFrameId = currentFrameId;
      }
      m_viewModelCandidates.push_back(&currentInstance);
      Logger::info(str::format(
        "[VM.candidate] f=", currentFrameId,
        " candidates=", m_viewModelCandidates.size()));
    }
    // NV-DXVK [VM.instance]: log every draw that reaches instance update so
    // we can see whether ViewModel-classified draws arrive here (proves
    // SceneManager → InstanceManager plumbing works).
    if (drawCall.cameraType == CameraType::ViewModel) {
      static uint32_t sLastF = 0;
      static uint32_t sCount = 0;
      const uint32_t fid = m_device->getCurrentFrameId();
      if (fid != sLastF) { sLastF = fid; sCount = 0; }
      if (sCount < 16) {
        ++sCount;
        Logger::info(str::format(
          "[VM.instance] f=", fid,
          " isHidden=", (currentInstance.m_isHidden ? 1 : 0),
          " firstUpdate=", (isFirstUpdateThisFrame ? 1 : 0),
          " mask=", currentInstance.getVkInstance().mask));
      }
    }

    if (RtxOptions::enableSeparateUnorderedApproximations() &&
        (drawCall.cameraType == CameraType::Main || drawCall.cameraType == CameraType::ViewModel) &&
        currentInstance.m_isUnordered &&
        !currentInstance.m_isHidden &&
        currentInstance.getVkInstance().mask != 0) {

      if (currentInstance.testCategoryFlags(InstanceCategories::Beam)) {
        createBeams(currentInstance);
      } else if(!currentInstance.surface.alphaState.isDecal) {
        createBillboards(currentInstance, cameraManager.getMainCamera().getDirection(false));
      }

      billboardsGotGenerated = currentInstance.m_billboardCount != 0;
    }

    // Updates done only once a frame unless overriden due to an explicit state
    if (isFirstUpdateThisFrame || overridePreviousCameraUpdate ||
        (billboardsGotGenerated && RtxOptions::getEnableOpacityMicromap())) {
      // Inform the listeners
      for (auto& event : m_eventHandlers) {
        event.onInstanceUpdatedCallback(currentInstance, drawCall, materialData, hasTransformChanged, hasPreviousPositions, isFirstUpdateThisFrame);
      }
    }
  }

  void InstanceManager::removeInstance(RtInstance* instance) {
    // NV-DXVK [VS_2904d2 removeInstance probe]: log when sub-view mountain
    // instances get destroyed. There are three paths to removal:
    //   (1) Lifetime expiry: instance->m_frameLastUpdated + numFramesToKeep
    //       Instances <= currentFrame (the standard GC pass).
    //   (2) m_isMarkedForGC=true (BLAS destroyed via onSceneObjectDestroyed,
    //       or viewModel/clone/virtual instance lifecycle).
    //   (3) forceGarbageCollection (instance count over numObjectsToKeep cap).
    // Logging which path fires tells us WHY dedup misses despite our reproject
    // producing byte-identical world positions across frames.
    {
      const BlasEntry* pBlas = instance->getBlas();
      if (pBlas != nullptr) {
        const XXH64_hash_t vsHash = pBlas->input.getTransformData().vertexShaderHash;
        if (vsHash == 0x2904d2163ef31a17ull) {
          thread_local uint32_t sRemoveProbe = 0;
          if (sRemoveProbe < 32 || (sRemoveProbe & 0x3FF) == 0) {
            const uint32_t curFrame = m_device->getCurrentFrameId();
            const uint32_t lastUpdated = instance->m_frameLastUpdated;
            const uint32_t lifetimeOpt = RtxOptions::numFramesToKeepInstances();
            const bool lifetimeExpired = (lastUpdated + lifetimeOpt <= curFrame);
            Logger::info(str::format(
              "[Rm2904] #", sRemoveProbe,
              " f=", curFrame,
              " lastUpd=", lastUpdated,
              " markedGC=", (instance->m_isMarkedForGC ? 1 : 0),
              " unlinkedGC=", (instance->m_isUnlinkedForGC ? 1 : 0),
              " lifetimeExp=", (lifetimeExpired ? 1 : 0),
              " keepN=", lifetimeOpt,
              " insideFrustum=", (instance->m_isInsideFrustum ? 1 : 0),
              " isHidden=", (instance->m_isHidden ? 1 : 0),
              " createdByRenderer=", (instance->m_isCreatedByRenderer ? 1 : 0)));
          }
          sRemoveProbe += 1;
        }
      }
    }

    // Always clean up replacement instance references, even for renderer-created instances
    // to avoid use-after-free bugs in ReplacementInstance.prims
    instance->getPrimInstanceOwner().setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex, instance, PrimInstance::Type::Instance);
    instance->removeFromSpatialCache();
    
    // In these cases we skip calling onInstanceDestroyed:
    //   Some view model and player instances are created in the renderer and don't have onInstanceAdded called,
    //   so not call onInstanceDestroyed either.
    if (instance->m_isCreatedByRenderer) {
      return;
    }

    
    for (auto& event : m_eventHandlers) {
      event.onInstanceDestroyedCallback(*instance);
    }
  }

  RtInstance* InstanceManager::createViewModelInstance(Rc<DxvkContext> ctx,
                                                       const RtInstance& reference,
                                                       const Matrix4d& perspectiveCorrection,
                                                       const Matrix4d& prevPerspectiveCorrection) {

    // Create a view model instance corresponding to the reference instance, for one frame 

    // Don't pollute global instance id with View Models since they're not tracked in game capturer
    const bool needValidGlobalInstanceId = false;

    RtInstance* viewModelInstance = createInstanceCopy(reference, needValidGlobalInstanceId);

    const uint32_t frameId = m_device->getCurrentFrameId();
    viewModelInstance->setFrameCreated(frameId);
    viewModelInstance->setFrameLastUpdated(frameId);
    viewModelInstance->m_vkInstance.mask = OBJECT_MASK_VIEWMODEL;
    viewModelInstance->setCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL, true);

    // View model instances are recreated every frame
    viewModelInstance->markForGarbageCollection();

    if (RtxOptions::ViewModel::perspectiveCorrection()) {
      // A transform that looks "correct" only from a main camera's point of view
      const auto corrected = perspectiveCorrection * reference.getTransform();
      const auto prevCorrected = prevPerspectiveCorrection * reference.getPrevTransform();

      auto isOrdinary = [](const Matrix4d& m) {
        auto isCloseTo = [](auto a, auto b) {
          return std::abs(a - b) < 0.001;
        };
        return isCloseTo(m[0][3], 0.0)
          && isCloseTo(m[1][3], 0.0)
          && isCloseTo(m[2][3], 0.0)
          && isCloseTo(m[3][3], 1.0);
      };

      // If matrices are not convoluted, don't modify the vertex data: just set the transforms directly
      if (isOrdinary(corrected) && isOrdinary(prevCorrected)) {
        viewModelInstance->teleport(corrected, prevCorrected);
      } else {
        ONCE(Logger::info("[RTX-Compatibility-Info] Unexpected values in the perspective-corrected transform of a view model. Fallback to geometry modification"));
        // Only need to run this on BVH op (maybe this could be moved to geometry processing?)
        if (viewModelInstance->getBlas()->frameLastUpdated == frameId) {
          const auto worldToObject = inverse(reference.getTransform());
          const auto instancePositionTransform = worldToObject * perspectiveCorrection * reference.getTransform();

          ctx->getCommonObjects()->metaGeometryUtils().dispatchViewModelCorrection(ctx,
            viewModelInstance->getBlas()->modifiedGeometryData, instancePositionTransform);
        }
      }
    }

    // ViewModel should never be considered static
    viewModelInstance->surface.isStatic = false;

    // Note this is an instance copy of a input reference. It is unknown to the source engine, so we don't call onInstanceAdded callbacks for it
    // It also results in this instance not being linked to reference instance BLAS and thus not considered in findSimilarInstances' lookups
    // This is desired as ViewModel instances are not to be linked frame to frame

    // NV-DXVK [VM.final]: log the final viewmodel instance's transform +
    // mask so we can see where it'd render and whether the BVH/TLAS upload
    // will accept it.
    {
      const auto& t = viewModelInstance->getTransform();
      Logger::info(str::format(
        "[VM.final] f=", frameId,
        " mask=0x", std::hex, (uint32_t)viewModelInstance->m_vkInstance.mask, std::dec,
        " pc=", (RtxOptions::ViewModel::perspectiveCorrection() ? 1 : 0),
        " T=(", t[3][0], ",", t[3][1], ",", t[3][2], ")",
        " diag=(", t[0][0], ",", t[1][1], ",", t[2][2], ")"));
    }

    return viewModelInstance;
  }

  void InstanceManager::createViewModelInstances(Rc<DxvkContext> ctx,
                                                 const CameraManager& cameraManager,
                                                 const RayPortalManager& rayPortalManager) {
    ScopedGpuProfileZone(ctx, "ViewModel");

    const uint32_t fid = m_device->getCurrentFrameId();
    const bool vmEnable = RtxOptions::ViewModel::enable();
    const bool vmCamValid = cameraManager.isCameraValid(CameraType::ViewModel);
    Logger::info(str::format(
      "[VM.create] f=", fid,
      " enable=", (vmEnable ? 1 : 0),
      " camValid=", (vmCamValid ? 1 : 0),
      " candidates=", m_viewModelCandidates.size()));

    if (!vmEnable)
      return;

    if (!vmCamValid)
      return;

    // If the first person player model is enabled, hide the view model.
    if (RtxOptions::PlayerModel::enableInPrimarySpace()) {
      for (auto* candidateInstance : m_viewModelCandidates) {
        candidateInstance->m_vkInstance.mask = 0;
      }
      return;
    }

    const RtCamera& camera = cameraManager.getMainCamera();
    const RtCamera& viewModelCamera = cameraManager.getCamera(CameraType::ViewModel);

    // Use the FOV (XY scaling) from the view-model matrix and the near/far planes (ZW scaling) from the main matrix.
    // The view-model camera has different near/far planes, so if that projection matrix is used naively,
    // the gun ends up being scaled up by a factor of 7 or so (in Portal).
    const auto& mainProjectionMatrix = camera.getViewToProjection();
    auto viewModelProjectionMatrix = viewModelCamera.getViewToProjection();
    viewModelProjectionMatrix[2][2] = mainProjectionMatrix[2][2];
    viewModelProjectionMatrix[2][3] = mainProjectionMatrix[2][3];
    viewModelProjectionMatrix[3][2] = mainProjectionMatrix[3][2];

    const auto& mainPreviousProjectionMatrix = camera.getPreviousViewToProjection();
    auto previousViewModelProjectionMatrix = viewModelCamera.getPreviousViewToProjection();
    previousViewModelProjectionMatrix[2][2] = mainPreviousProjectionMatrix[2][2];
    previousViewModelProjectionMatrix[2][3] = mainPreviousProjectionMatrix[2][3];
    previousViewModelProjectionMatrix[3][2] = mainPreviousProjectionMatrix[3][2];

    // Apply an extra scaling matrix to the view-space positions of view model to make it less likely to interact with world geometry.
    Matrix4d scaleMatrix {};
    scaleMatrix[0][0] = scaleMatrix[1][1] = scaleMatrix[2][2] = RtxOptions::ViewModel::scale();
    scaleMatrix[3][3] = 1.0;

    // Compute the view-model perspective correction matrix.
    // This expression (read right-to-left) is a solution to the following equation:
    //   (mainProjection * mainView * objectToWorld) * transformedPosition = (viewModelProjection * viewModelView * objectToWorld) * position
    // where 'position' is the original vertex data supplied by the game, and 'transformedPosition' is what we need to compute in order to make
    // the view model project into the same screen positions using the main camera.
    // The 'objectToWorld' matrices are applied later, in createViewModelInstance, because they're different per-instance.
    const auto perspectiveCorrection = camera.getViewToWorld(false) * (camera.getProjectionToView() * viewModelProjectionMatrix * scaleMatrix) * viewModelCamera.getWorldToView(false);
    const auto prevPerspectiveCorrection = camera.getPreviousViewToWorld(false) * (camera.getPreviousProjectionToView() * previousViewModelProjectionMatrix * scaleMatrix) * viewModelCamera.getPreviousWorldToView(false);

    // Create any valid view model instances from the list of candidates
    std::vector<RtInstance*> viewModelInstances;
    uint32_t vmSkippedMulticam = 0, vmSkippedNoCam = 0, vmCreated = 0;
    for (auto* candidateInstance : m_viewModelCandidates) {

      // Valid view model instances must be associated only with the view model camera
      // Check: exactly one bit set (power-of-two check via raw bitmask)
      const auto seenMask = candidateInstance->m_seenCameraTypes.raw();
      if (seenMask == 0) { ++vmSkippedNoCam; continue; }
      if ((seenMask & (seenMask - 1)) != 0) { ++vmSkippedMulticam; continue; }

      // Hide the reference instance since we'll create a separate instance for the view model
      candidateInstance->m_vkInstance.mask = 0;

      // Tag the instance as ViewModel so it can be checked for it being a reference view model instance
      candidateInstance->setCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL, true);

      viewModelInstances.push_back(createViewModelInstance(ctx, *candidateInstance, perspectiveCorrection, prevPerspectiveCorrection));
      ++vmCreated;
    }
    Logger::info(str::format(
      "[VM.created] f=", fid,
      " total=", m_viewModelCandidates.size(),
      " created=", vmCreated,
      " skipNoCam=", vmSkippedNoCam,
      " skipMultiCam=", vmSkippedMulticam));

    // Create virtual instances for the view model instances
    createRayPortalVirtualViewModelInstances(viewModelInstances, cameraManager, rayPortalManager);
  }

  static bool isInsidePlayerModel(const Vector3& playerModelPosition, const Vector3& instancePosition) {
    const Vector3 playerToInstance = instancePosition - playerModelPosition;
    const float horizontalDistance = length(Vector2(playerToInstance.x, playerToInstance.y));
    const float verticalDistance = fabs(playerToInstance.z);

    // Distance thresholds determined experimentally to match the portal gun held in player's hands
    // but not match the gun on the pedestals.
    const float maxHorizontalDistance = RtxOptions::PlayerModel::horizontalDetectionDistance();
    const float maxVerticalDistance = RtxOptions::PlayerModel::verticalDetectionDistance();

    return (horizontalDistance <= maxHorizontalDistance) && (verticalDistance <= maxVerticalDistance);
  }

  void InstanceManager::filterPlayerModelInstances(const Vector3& playerModelPosition, const RtInstance* bodyInstance) {
    for (size_t i = 0; i < m_playerModelInstances.size(); ++i) {
      RtInstance* instance = m_playerModelInstances[i];

      // Don't compare the body to itself.
      if (instance == bodyInstance)
        continue;

      if (instance->m_isUnordered) {
        // Particles don't have a valid position in the instance matrix and often combine many particles
        // in one instance. So we rely on the analysis done for billboard creation earlier and see if the billboards
        // intersect with the player model.

        // Start assuming that the instance is actually part of the player model.
        bool isPlayerModelInstance = true;

        if (instance->m_billboardCount > 0) {
          // Check if the billboards are used as intersection primitives. 
          // Note: If one billboard is used as an intersection primitive, all of them are
          if (m_billboards[instance->m_firstBillboard].allowAsIntersectionPrimitive) {
            // If there are billboards, look at their centers, and if any of them are outside of the player model
            // limits, consider the entire instance non-player-model.
            // Opposite approach is possible, too, not entirely sure what's better.
            for (uint32_t billboardIndex = 0; billboardIndex < instance->m_billboardCount; ++billboardIndex) {
              const IntersectionBillboard& billboard = m_billboards[billboardIndex + instance->m_firstBillboard];
              if (!isInsidePlayerModel(playerModelPosition, billboard.center)) {
                isPlayerModelInstance = false;
                break;
              }
            }
          }
        }

        if (isPlayerModelInstance) {
          if (instance->m_billboardCount > 0) {
            // If this instance contains particles and is part of the player model,
            // assign the PLAYER_MODEL mask to its billboards and hide the original instance.
            for (uint32_t billboardIndex = 0; billboardIndex < instance->m_billboardCount; ++billboardIndex) {
              IntersectionBillboard& billboard = m_billboards[billboardIndex + instance->m_firstBillboard];
              billboard.instanceMask = OBJECT_MASK_PLAYER_MODEL;
            }

            instance->getVkInstance().mask = 0;
          }
        } else {
          // Remove the instance from the list to avoid creating virtual instances for it.
          m_playerModelInstances.erase(m_playerModelInstances.begin() + i);
          --i;
        }
      } else {
        const Vector3 instancePosition = instance->getTransform()[3].xyz();

        if (!isInsidePlayerModel(playerModelPosition, instancePosition)) {
          // Note: just use the OPAQUE flag here, which works for Portal with current assets.
          // Might want to apply more complex logic if that is insufficient one day.
          instance->getVkInstance().mask = OBJECT_MASK_OPAQUE;

          // Remove this instance from the player model list.
          m_playerModelInstances.erase(m_playerModelInstances.begin() + i);
          --i;
        }
      }
    }
  }

  void InstanceManager::detectIfPlayerModelIsVirtual(
    const CameraManager& cameraManager,
    const RayPortalManager& rayPortalManager,
    const Vector3& playerModelPosition,
    bool* out_PlayerModelIsVirtual,
    const SingleRayPortalDirectionInfo** out_NearPortalInfo,
    const SingleRayPortalDirectionInfo** out_FarPortalInfo) const {
    auto& rayPortalPair = *rayPortalManager.getRayPortalPairInfos().begin();

    *out_PlayerModelIsVirtual = false;
    int portalIndexForVirtualInstances = -1;

    if (rayPortalPair.has_value()) {

      // Estimate the position of the player model's eyes (where the camera normally is), ignoring crouching.
      // Note that in Portal, the player model is always upright, even if the player is flying out of a floor portal upside down.
      // This makes the detection of whether the player model is virtual more robust.

      Vector3 playerModelEyePosition = playerModelPosition;
      playerModelEyePosition.z += RtxOptions::PlayerModel::eyeHeight();

      // Find the portal that is closest to the model

      float distanceOfModelPortal = FLT_MAX;
      int playerModelNearPortalIndex = 0;

      for (int portalIndex = 0; portalIndex < 2; ++portalIndex) {
        const RayPortalInfo& portalInfo = rayPortalPair->pairInfos[portalIndex].entryPortalInfo;
        const float distanceToModel = length(portalInfo.centroid - playerModelEyePosition);
        if (distanceToModel < distanceOfModelPortal) {
          distanceOfModelPortal = distanceToModel;
          playerModelNearPortalIndex = portalIndex;
        }
      }

      const Vector3& camPos = cameraManager.getCamera(CameraType::Main).getPosition(/* freecam = */ false);

      // Find the portal that the imaginary player (i.e. a blob around the camera, or camera volume) is currently intersecting

      uint32_t cameraVolumePortalIntersectionMask = 0;

      for (uint i = 0; i < 2; i++) {
        const auto& rayPortal = rayPortalPair->pairInfos[i];
        const Vector3 dirToPortalCentroid = rayPortal.entryPortalInfo.centroid - camPos;

        // Approximate the player collision model with this capsule-like shape
        const float maximumNormalDistance = lerp(RtxOptions::PlayerModel::intersectionCapsuleRadius(),
                                                 RtxOptions::PlayerModel::intersectionCapsuleHeight(),
                                                 clamp(rayPortal.entryPortalInfo.planeNormal.z, 0.f, 1.f));

        // Test if that shape intersects with the portal and if the camera is in front of it
        const float planeDistanceNormal = -dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeNormal);
        const float planeDistanceX = dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeBasis[0]);
        const float planeDistanceY = dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeBasis[1]);
        const bool cameraVolumeIntersectsPortal = 0.f < planeDistanceNormal && planeDistanceNormal < maximumNormalDistance
          && std::abs(planeDistanceX) < rayPortal.entryPortalInfo.planeHalfExtents.x
          && std::abs(planeDistanceY) < rayPortal.entryPortalInfo.planeHalfExtents.y;

        if (cameraVolumeIntersectsPortal) {
          portalIndexForVirtualInstances = i;
          cameraVolumePortalIntersectionMask |= (1 << i);
        }
      }

      // If the camera volume intersects exactly one portal, and the player model is closer to another portal,
      // that must mean the game is rendering the model at the other side of a portal (i.e. the player model is virtual/ghost).
      // This excludes the case when the camera intersects both portals.
      // De-virtualize the player model using the same portal that was used to virtualize it.
      const int playerModelFarPortalIndex = !playerModelNearPortalIndex;
      // Additional heuristic that tells if the player model eyes become closer to the camera if it's de-virtualized.
      // Fixes false virtual player model detections when there is one portal on a wall and another on the floor right next to it,
      // and you stand between these portals (see TREX-2254).
      const float playerModelEyeDistanceToCamera = length(playerModelEyePosition - camPos);
      const Vector3 devirtualizedPlayerModelEyePosition = (rayPortalPair->pairInfos[playerModelNearPortalIndex].portalToOpposingPortalDirection * Vector4(playerModelEyePosition, 1.f)).xyz();
      const float devirtualizedPlayerModelEyeDistanceToCamera = length(devirtualizedPlayerModelEyePosition - camPos);
      if (cameraVolumePortalIntersectionMask == (1 << playerModelFarPortalIndex) && devirtualizedPlayerModelEyeDistanceToCamera < playerModelEyeDistanceToCamera) {
        *out_PlayerModelIsVirtual = true;
        portalIndexForVirtualInstances = !portalIndexForVirtualInstances;
      }
      // In other (regular) situations, if the camera volume intersects at least one volume, make sure to use
      // the same portal for virtual player model as the one used for the virtual view model,
      // to avoid inconsistencies in tracing.
      else if (m_virtualInstancePortalIndex >= 0 && portalIndexForVirtualInstances >= 0) {
        portalIndexForVirtualInstances = m_virtualInstancePortalIndex;
      }
    }

    *out_NearPortalInfo = (portalIndexForVirtualInstances >= 0) ? &rayPortalPair->pairInfos[portalIndexForVirtualInstances] : nullptr;
    *out_FarPortalInfo = (portalIndexForVirtualInstances >= 0) ? &rayPortalPair->pairInfos[!portalIndexForVirtualInstances] : nullptr;
  }

  void InstanceManager::createPlayerModelVirtualInstances(Rc<DxvkContext> ctx, const CameraManager& cameraManager, const RayPortalManager& rayPortalManager) {
    if (m_playerModelInstances.empty())
      return;

    // Sometimes, the game renders the player model on the other side of the portal
    // that is closest to the camera. To detect that, we look at the model position.
    // Here, we also detect the instances of the portal gun that are rendered in the world
    // using the same mesh and texture as the held portal gun but should not be considered
    // a part of the player model. Those are detected by comparing their position to the body.
    
    // Find the instance marked with the "playerBody" material
    const RtInstance* bodyInstance = nullptr;
    for (RtInstance* instance : m_playerModelInstances) {
      if (instance->testCategoryFlags(InstanceCategories::ThirdPersonPlayerBody))
        bodyInstance = instance;
    }

    if (!bodyInstance)
      return;

    // Get the position from the transform matrix - works for Portal
    Vector3 playerModelPosition = bodyInstance->getTransform()[3].xyz();

    // Detect instances that are too far away from the body, make them regular objects.
    // This fixes the guns placed on pedestals to be picked up.
    filterPlayerModelInstances(playerModelPosition, bodyInstance);

    // Detect if the player model rendered by the game is virtual or not
    bool playerModelIsVirtual = false;
    // Near portal is where the original instance is
    const SingleRayPortalDirectionInfo* nearPortalInfo = nullptr;
    // Far portal is where the cloned instance will be
    const SingleRayPortalDirectionInfo* farPortalInfo = nullptr;
    detectIfPlayerModelIsVirtual(cameraManager, rayPortalManager, playerModelPosition, &playerModelIsVirtual, &nearPortalInfo, &farPortalInfo);
        
    const uint32_t frameId = m_device->getCurrentFrameId();

    // Set up the math to offset the player model backwards if it's to be shown in primary space
    float backwardOffset = RtxOptions::PlayerModel::backwardOffset();

    const bool createVirtualInstances = RtxOptions::PlayerModel::enableVirtualInstances() && (nearPortalInfo != nullptr);

    // The loop below creates virtual instances and applies the offset. Exit if neither is necessary.
    if (!createVirtualInstances && backwardOffset == 0.f)
      return;

    // Calculate the offset vector
    Vector3 backwardOffsetVector = cameraManager.getMainCamera().getHorizontalForwardDirection();
    backwardOffsetVector *= -backwardOffset;

    if (playerModelIsVirtual && farPortalInfo) {
      // Transform the offset vector into portal space
      backwardOffsetVector = (farPortalInfo->portalToOpposingPortalDirection * Vector4(backwardOffsetVector, 0.f)).xyz();
    }

    const Matrix4 backwardOffsetMatrix {
      Vector4{ 1.f, 0.f, 0.f, 0.f },
      Vector4{ 0.f, 1.f, 0.f, 0.f },
      Vector4{ 0.f, 0.f, 1.f, 0.f },
      Vector4(backwardOffsetVector, 1.f)
    };
    
    // Create virtual instances for player model instances that are close to portals.
    // Offset both real and virtual instances by backwardOffset units if enabled.
    for (RtInstance* originalInstance : m_playerModelInstances) {

      if (backwardOffset != 0.f) {
        // Offset the original instance
        originalInstance->teleportWithHistory(backwardOffsetMatrix);

        // Offset the original instance particles
        for (uint32_t i = 0; i < originalInstance->m_billboardCount; ++i) {
          m_billboards[originalInstance->m_firstBillboard + i].center += backwardOffsetVector;
        }
      }

      if (!createVirtualInstances)
        continue;
      
      // Don't pollute global instance id with Player Models since they're not tracked in game capturer
      const bool needValidGlobalInstanceId = false;

      RtInstance* clonedInstance = createInstanceCopy(*originalInstance, needValidGlobalInstanceId);
      
      clonedInstance->setFrameCreated(frameId);
      clonedInstance->setFrameLastUpdated(frameId);

      // Cloned player model instances are recreated every frame
      clonedInstance->markForGarbageCollection();

      // Compute the instance masks for both original and cloned instances.
      // When the original instance is real (which is the case normally), the cloned one is virtual and located on the other side of a portal.
      // When the original instance is virtual (rendered by the game on the other side of a portal), the cloned one is not.
      const uint32_t originalInstanceMask = playerModelIsVirtual ? OBJECT_MASK_PLAYER_MODEL_VIRTUAL : OBJECT_MASK_PLAYER_MODEL;
      const uint32_t clonedInstanceMask = playerModelIsVirtual ? OBJECT_MASK_PLAYER_MODEL : OBJECT_MASK_PLAYER_MODEL_VIRTUAL;

      if (originalInstance->m_billboardCount > 0) {
        // If this is a translucent instance with billboards, clone the billboards and hide the original instance.
        
        // Allocate some billboard entries first
        clonedInstance->m_firstBillboard = m_billboards.size();
        clonedInstance->m_billboardCount = originalInstance->m_billboardCount;
        m_billboards.resize(m_billboards.size() + originalInstance->m_billboardCount);

        // Copy the billboards to the new location and patch them
        for (uint32_t i = 0; i < originalInstance->m_billboardCount; ++i) {
          IntersectionBillboard* originalBillboard = &m_billboards[originalInstance->m_firstBillboard + i];
          IntersectionBillboard* clonedBillboard = &m_billboards[clonedInstance->m_firstBillboard + i];

          *clonedBillboard = *originalBillboard;
          clonedBillboard->instance = clonedInstance;

          // Update the instance masks of both instances
          originalBillboard->instanceMask = originalInstanceMask;
          clonedBillboard->instanceMask = clonedInstanceMask;

          // Update the center.
          // The orientation is irrelevant because the GPU will re-derive it for each ray.
          clonedBillboard->center = (nearPortalInfo->portalToOpposingPortalDirection * Vector4(originalBillboard->center, 1.0f)).xyz();
        }

        // Hide the geometric instances but keep them in the list so that surface data is generated for them.
        originalInstance->m_vkInstance.mask = 0;
        clonedInstance->m_vkInstance.mask = 0;
      }
      else {
        // Update the instance masks of both instances
        originalInstance->m_vkInstance.mask = originalInstanceMask;
        clonedInstance->m_vkInstance.mask = clonedInstanceMask;
      }
      
      // Update cloned instance transforms given the reference and the portal transform
      {
        clonedInstance->teleportWithHistory(nearPortalInfo->portalToOpposingPortalDirection);
      }

      // Use a clip plane to make sure that the cloned instance doesn't stick through a slab
      // that the other portal might be placed on.
      clonedInstance->surface.isClipPlaneEnabled = true;
      clonedInstance->surface.clipPlane = Vector4(farPortalInfo->entryPortalInfo.planeNormal,
        -dot(farPortalInfo->entryPortalInfo.planeNormal, farPortalInfo->entryPortalInfo.centroid));
      // Use the FORCE_NO_OPAQUE flag to enable any-hit processing in the visiblity rays for this clipped instance.
      clonedInstance->m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;

      // Same clip plane logic for the original instance, only using the near portal.
      originalInstance->surface.isClipPlaneEnabled = true;
      originalInstance->surface.clipPlane = Vector4(nearPortalInfo->entryPortalInfo.planeNormal,
        -dot(nearPortalInfo->entryPortalInfo.planeNormal, nearPortalInfo->entryPortalInfo.centroid));
      originalInstance->m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    }
  }

  void InstanceManager::findPortalForVirtualInstances(const CameraManager& cameraManager, const RayPortalManager& rayPortalManager) {
    m_virtualInstancePortalIndex = -1;

    // Virtual instances for the view model and the player model are generated for the closest portal to the camera.

    static_assert(maxRayPortalCount == 2);
    auto& rayPortalPair = *rayPortalManager.getRayPortalPairInfos().begin();

    if (!rayPortalPair.has_value())
      return;

    const Vector3& camPos = cameraManager.getCamera(CameraType::Main).getPosition(/* freecam = */ false);

    const float kMaxDistanceToPortal = RtxOptions::ViewModel::rangeMeters() * RtxOptions::getMeterToWorldUnitScale();

    // Find the closest valid portal to generate the instances for since we can generate 
    // virtual instances only for one of the portals due to instance mask bit allocation.
    // This will result in missing virtual viewModel geo for some corner cases, 
    // such as when portals are close to each other in a corner arrangement
    float minDistanceToPortal = FLT_MAX;

    for (uint i = 0; i < 2; i++) {
      const auto& rayPortal = rayPortalPair->pairInfos[i];
      const Vector3 dirToPortalCentroid = rayPortal.entryPortalInfo.centroid - camPos;
      const float distanceToPortal = length(dirToPortalCentroid);

      if (distanceToPortal <= kMaxDistanceToPortal &&
          distanceToPortal < minDistanceToPortal) {
        minDistanceToPortal = distanceToPortal;
        m_virtualInstancePortalIndex = rayPortal.entryPortalInfo.portalIndex;
      }
    }

  }

  void InstanceManager::createRayPortalVirtualViewModelInstances(const std::vector<RtInstance*>& viewModelReferenceInstances,
                                                                 const CameraManager& cameraManager,
                                                                 const RayPortalManager& rayPortalManager) {
    // Early out if there is no eligible portal
    if (m_virtualInstancePortalIndex < 0)
      return;

    if (rayPortalManager.getRayPortalPairInfos().empty()) {
      assert(!"There must be a portal pair in createRayPortalVirtualViewModelInstances if m_virtualInstancePortalIndex is defined");
      return;
    }

    if (!RtxOptions::ViewModel::enableVirtualInstances())
      return;

    const SingleRayPortalDirectionInfo& closestPortalInfo = rayPortalManager.getRayPortalPairInfos()[0]->pairInfos[m_virtualInstancePortalIndex];
    
    const uint32_t frameId = m_device->getCurrentFrameId();

    // Create virtual instances for view model instances that are close to portals
    for (RtInstance* referenceInstance : viewModelReferenceInstances) {

      // Create a view model virtual instance corresponding to the view model instance, for one frame

      // Don't pollute global instance id with View Models since they're not tracked in game capturer
      const bool needValidGlobalInstanceId = false;

      RtInstance* virtualInstance = createInstanceCopy(*referenceInstance, needValidGlobalInstanceId);

      virtualInstance->setFrameCreated(frameId);
      virtualInstance->setFrameLastUpdated(frameId);

      // Virtual view model instances are recreated every frame
      virtualInstance->markForGarbageCollection();

      // Virtual instances are to be visible only in their corresponding portal spaces
      static_assert(maxRayPortalCount == 2);
      // View model virtual instance
      virtualInstance->m_vkInstance.mask = OBJECT_MASK_VIEWMODEL_VIRTUAL;
    
      // Update virtual instance transforms given the reference and the portal transform
      {
        virtualInstance->teleportWithHistory(closestPortalInfo.portalToOpposingPortalDirection);
      }

      // Note this is an instance copy of an input reference. It is unknown to the source engine, so we don't call onInstanceAdded callbacks for it
      // It also results in this instance not being linked to reference instance BLAS and thus not considered in findSimilarInstances' lookups
      // This is desired as ViewModel instances are not to be linked frame to frame
    }
  }

  void InstanceManager::resetSurfaceIndices() {
    for (auto instance : m_instances)
      instance->m_surfaceIndex = SURFACE_INDEX_INVALID;
  }

  inline bool isFpSpecial(float x) {
    const uint32_t u = *(uint32_t*) &x;
    return (u & 0x7f800000) == 0x7f800000;
  }

  void InstanceManager::createBillboards(RtInstance& instance, const Vector3& cameraViewDirection)
  {
    const RasterGeometry& geometryData = instance.getBlas()->input.getGeometryData();

    constexpr uint32_t indicesPerQuad = 6;

    // Check if this is a supported geometry first
    if (geometryData.indexCount < indicesPerQuad || 
        (geometryData.indexCount % indicesPerQuad) != 0 ||
        geometryData.indexBuffer.indexType() != VK_INDEX_TYPE_UINT16 ||
        geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      return;
    
    const GeometryBufferData bufferData(geometryData);

    // Check if the necessary buffers exist
    // Warning: do not generate billboards for instances without indices as other code sections using billboards expect indices to be present
    if (!bufferData.indexData || !bufferData.positionData || !bufferData.texcoordData)
      return;

    const bool hasNonIdentityTextureTransform = instance.surface.textureTransform != Matrix4();
    bool bSuccess = true;
    bool areAllBillboardsValidIntersectionCandidates = true;
    uint32_t billboardCount = 0;
    instance.m_firstBillboard = m_billboards.size();

    const Matrix4 instanceTransform = instance.getTransform();

    // Go over all quads in this draw call.
    // Note: decals are often batched into a few draw calls, and we want to offset each decal separately.
    for (int indexOffset = 0; indexOffset + indicesPerQuad <= geometryData.indexCount; indexOffset += indicesPerQuad) {
      // Load indices for a quad
      uint16_t indices[indicesPerQuad];
      for (size_t idx = 0; idx < indicesPerQuad; ++idx) {
        indices[idx] = bufferData.getIndex(idx + indexOffset);
      }


      // Make sure that these indices follow a known quad pattern: A, B, C, A, C, D
      // If they don't, we can't process this "quad" - so, cancel the whole instance.
      if (indices[0] != indices[3] || indices[2] != indices[4]) {
        ONCE(Logger::info("[RTX] InstanceManager: detected unsupported quad index layout for billboard creation"));
        // This quad is incompatible altogether. Abort processing billboards for this instance and skip billboard processing for it
        bSuccess = false;
        break;
      }
      
      // Load data for a triangle
      Vector3 positions[3];
      Vector2 texcoords[4];
      uint8_t vertexOpacities8bit[4] = {};
      
      for (size_t idx = 0; idx < 3; ++idx) {
        const uint16_t currentIndex = indices[idx];

        Vector4 objectSpacePosition = Vector4(bufferData.getPosition(currentIndex), 1.0f);

        positions[idx] = (instanceTransform * objectSpacePosition).xyz();

        texcoords[idx] = bufferData.getTexCoord(currentIndex);

        if (hasNonIdentityTextureTransform)
          texcoords[idx] = (instance.surface.textureTransform * Vector4(texcoords[idx].x, texcoords[idx].y, 0.f, 1.f)).xy();

        if (bufferData.vertexColorData)
          vertexOpacities8bit[idx] = bufferData.getVertexColor(indices[idx]) >> 24;
      }

      // Load one vertex color - assuming that the entire billboard uses the same color
      uint32_t vertexColor = ~0u;
      if (bufferData.vertexColorData)
        vertexColor = bufferData.getVertexColor(indices[0]);

      // Compute the normal
      const Vector3 xVector { positions[2] - positions[1] };
      const Vector3 yVector { positions[1] - positions[0] };
      const Vector3 center { (positions[2] + positions[0]) * 0.5f };

      IntersectionBillboard billboard;

      const bool centerIsSpecial = isFpSpecial(center.x) || isFpSpecial(center.y) || isFpSpecial(center.z);
      if (centerIsSpecial) {
        areAllBillboardsValidIntersectionCandidates = false;
      }

      const float xLength = length(xVector);
      const float yLength = length(yVector);
      const float dotAxes = dot(xVector, yVector) / (xLength * yLength);
      // Note: This could probably be handled in a better way (like skipping this quad) rather than just assigning
      // a fallback normal, but this is simple enough.
      const Vector3 normal = safeNormalize(cross(xVector, yVector), Vector3(0.0f, 0.0f, 1.0f));
      const float normalDotCamera = dot(normal, cameraViewDirection);


      // Limit the set of particles that are turned into intersection primitives:
      // - Must be roughly square
      const bool isSquare = xLength <= yLength * 1.5f && yLength <= xLength * 1.5f;
      // - The original quad must have perpendicular sides
      const bool hasPerpendicularSides = std::abs(dotAxes) < 0.01f;
      // - Must be in the camera view plane, i.e. only auto-oriented particles, not world-space ones
      //   (except player model particles, which are oriented towards the camera and not in the view plane)
      const bool isInViewPlane = std::abs(normalDotCamera) > 0.99f;
      // Assume that all billboards on the player model are camera facing
      const bool isCameraFacing = instance.m_isPlayerModel;
      if (!isSquare || !hasPerpendicularSides || !isInViewPlane && !isCameraFacing) {
        areAllBillboardsValidIntersectionCandidates = false;
      }

      const Vector2 xVectorUV { texcoords[2] - texcoords[1] };
      const Vector2 yVectorUV { texcoords[1] - texcoords[0] };
      const Vector2 centerUV { (texcoords[2] + texcoords[0]) * 0.5f };

      // Fill in data for the quad's last/4th vertex
      texcoords[3] = bufferData.getTexCoord(indices[5]);
      if (bufferData.vertexColorData)
        vertexOpacities8bit[3] = bufferData.getVertexColor(indices[5]) >> 24;

      billboard.center = center;
      billboard.xAxis = xVector / xLength;
      billboard.width = xLength;
      billboard.yAxis = yVector / yLength;
      billboard.height = yLength;
      billboard.xAxisUV = xVectorUV * 0.5f;
      billboard.yAxisUV = yVectorUV * 0.5f;
      billboard.centerUV = centerUV;
      billboard.instance = &instance;
      billboard.vertexColor = vertexColor;
      billboard.instanceMask = instance.getVkInstance().mask & OBJECT_MASK_UNORDERED_ALL_INTERSECTION_PRIMITIVE;
      billboard.texCoordHash = XXH64(texcoords, sizeof(texcoords), kEmptyHash);
      billboard.vertexOpacityHash = XXH64(vertexOpacities8bit, sizeof(vertexOpacities8bit), kEmptyHash);
      billboard.allowAsIntersectionPrimitive = true;
      billboard.isBeam = false;
      billboard.isCameraFacing = isCameraFacing;
      m_billboards.push_back(billboard);
      ++billboardCount;
    }

    if (bSuccess) {
      instance.m_billboardCount = billboardCount;

      if (areAllBillboardsValidIntersectionCandidates) {
        // Update the instance mask to hide it from rays that look only for intersection billboards.
        instance.getVkInstance().mask &= OBJECT_MASK_UNORDERED_ALL_GEOMETRY;
      } else {
        // Disable the rest of the billboards as intersection primitives since only a single mask can be used
        // per instance
        for (uint32_t i = m_billboards.size() - instance.m_billboardCount; i < m_billboards.size(); i++) {
          IntersectionBillboard& billboard = m_billboards[i];
          billboard.allowAsIntersectionPrimitive = false;
        }
      }
    } else {
      // Revert the billboards that were created successfully before the first failure,
      // because one of the failed to be created
      m_billboards.erase(m_billboards.end() - billboardCount, m_billboards.end());
    }
  }

  void InstanceManager::createBeams(RtInstance& instance) {
    const RasterGeometry& geometryData = instance.getBlas()->input.getGeometryData();

    // Check if this is a supported geometry first
    if (geometryData.indexCount < 4 ||
        (geometryData.indexCount % 2) != 0 ||
        geometryData.indexBuffer.indexType() != VK_INDEX_TYPE_UINT16 ||
        geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
      return;

    const GeometryBufferData bufferData(geometryData);

    // Check if the necessary buffers exist
    if (!bufferData.indexData || !bufferData.positionData || !bufferData.texcoordData)
      return;

    // Extract the beams from the triangle strip.
    // Start by loading the first 2 indices.
    uint16_t indices[4];
    indices[0] = bufferData.getIndex(0);
    indices[1] = bufferData.getIndex(1);

    for (int index = 2; index < geometryData.indexCount - 1; index += 2) {
      // When there are multiple beams packed into one triangle strip, they are separated
      // by a pair of repeating indices, such as: (0 1 2 3) 3 4 (4 5 6 7)
      // We want to keep looking at indices until either the end of the strip is reached,
      // or until we detect such a repeating pair. In the latter case, we skip the pair
      // at the end of this loop.
      const bool endOfStrip = index >= geometryData.indexCount - 2;
      const bool restart = !endOfStrip && (bufferData.getIndex(index + 1) == bufferData.getIndex(index + 2));

      if (!endOfStrip && !restart)
        continue;

      // Load the indices of the last 2 vertices of the beam.
      indices[2] = bufferData.getIndex(index);
      indices[3] = bufferData.getIndex(index + 1);

      // Load the source data for the 4 vertices that define our beam.
      Vector3 positions[4];
      Vector2 texcoords[4];
      for (int i = 0; i < 4; ++i) {
        positions[i] = bufferData.getPosition(indices[i]);
        texcoords[i] = bufferData.getTexCoord(indices[i]);
      }

      // Load one vertex color - assuming that the entire beam uses the same color
      uint32_t vertexColor = ~0u;
      if (bufferData.vertexColorData)
        vertexColor = bufferData.getVertexColor(indices[0]);

      // Extract the beam cylinder axis, length and width from the vertices.
      // Note that the 4 vertices are not necessarily coplanar: the beam is tessellated
      // in the axial direction, and each segment is rotated separately to face the camera.
      // The vertices are laid out in a triangle strip order:
      //     0-2
      //  -- |/| --> axis
      //     1-3
      const Vector3 startPosition = (positions[0] + positions[1]) * 0.5f;
      const Vector3 endPosition = (positions[2] + positions[3]) * 0.5f;
      const float beamWidth = length(positions[1] - positions[0]);
      const float beamLength = length(endPosition - startPosition);

      // Fill out the billboard struct.
      IntersectionBillboard billboard;
      billboard.center = (startPosition + endPosition) * 0.5f;
      billboard.xAxis = normalize(positions[1] - positions[0]);
      billboard.width = beamWidth;
      billboard.yAxis = normalize(endPosition - startPosition);
      billboard.height = beamLength;
      billboard.xAxisUV = (texcoords[1] - texcoords[0]) * 0.5f;
      billboard.yAxisUV = (texcoords[2] - texcoords[0]) * 0.5f;
      billboard.centerUV = (texcoords[0] + texcoords[3]) * 0.5f;
      billboard.vertexColor = vertexColor;
      billboard.instanceMask = instance.getVkInstance().mask & OBJECT_MASK_UNORDERED_ALL_INTERSECTION_PRIMITIVE;
      billboard.instance = &instance;
      billboard.texCoordHash = 0;
      billboard.vertexOpacityHash = 0;
      billboard.allowAsIntersectionPrimitive = true;
      billboard.isBeam = true;
      billboard.isCameraFacing = false;
      m_billboards.push_back(billboard);

      // If there are enough vertices left in the strip to fit one more beam, after the separator pair,
      // skip the separator and load the first two indices of the next beam.
      if (index <= geometryData.indexCount - 8) {
        index += 4;
        indices[0] = bufferData.getIndex(index);
        indices[1] = bufferData.getIndex(index + 1);
      }
    }

    instance.getVkInstance().mask &= OBJECT_MASK_UNORDERED_ALL_GEOMETRY;

    // Note: setting the instance's billboardCount to 0 here because we don't need either of the uses of that count:
    // - Beams cannot be parts of a player model;
    // - Beams should not be split into quads for OMM reuse.
    instance.m_billboardCount = 0;
  }

  const XXH64_hash_t RtInstance::calculateAntiCullingHash() const {
    if (RtxOptions::AntiCulling::isObjectAntiCullingEnabled()) {
      const Vector3 pos = getWorldPosition();
      const XXH64_hash_t posHash = XXH3_64bits(&pos, sizeof(pos));
      XXH64_hash_t antiCullingHash = XXH3_64bits_withSeed(&m_materialDataHash, sizeof(XXH64_hash_t), posHash);

      if (RtxOptions::AntiCulling::Object::hashInstanceWithBoundingBoxHash() &&
          RtxOptions::needsMeshBoundingBox()) {
        const AxisAlignedBoundingBox& boundingBox = getBlas()->input.getGeometryData().boundingBox;
        const XXH64_hash_t bboxHash = boundingBox.calculateHash();
        antiCullingHash = XXH3_64bits_withSeed(&bboxHash, sizeof(antiCullingHash), antiCullingHash);
      }
      return antiCullingHash;
    }

    return XXH64_hash_t();
  }
}  // namespace dxvk
