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
#include <atomic>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <assert.h>

#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_draw_call_cache.h"
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

  // Forward decl for the engine-hook main-cam capture counter (defined in
  // rtx_camera_manager.cpp at dxvk::tf2 scope). Used as a gameplay-active
  // gate for the PropCensus probe so thread-local accumulators don't fill
  // with menu-phase noise.
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;
  }

  // NV-DXVK [VsColor]: session-stable vertex-shader -> small-id table backing
  // RtSurface::vsDebugId and DEBUG_VIEW_VERTEX_SHADER_ID.
  //
  // Why session-stable rather than per-frame: the whole point is to watch one
  // object over time, so its colour must not move between frames. A per-frame
  // renumbering would repaint the scene every frame and be exactly as useless
  // as surfaceIndex is for cross-frame attribution.
  //
  // Cost: one hash lookup per draw against a table that stops growing after
  // the first few frames (this level has ~31 distinct VSes), and ONE log line
  // per VS for the entire run -- not per frame, and not per draw. There is no
  // throttle and no sampling: every draw gets a correct id every frame, the
  // identification is simply built so that being complete is already cheap.
  //
  // Ids start at 1; 0 is reserved for "unassigned" and paints black.
  static uint16_t acquireVsDebugId(XXH64_hash_t vsHash) {
    if (vsHash == 0) {
      return 0;
    }

    static std::mutex sVsIdMu;
    static std::unordered_map<XXH64_hash_t, uint16_t> sVsIds;
    static uint16_t sNextVsId = 1;

    uint16_t assigned = 0;
    bool isNew = false;
    {
      std::lock_guard<std::mutex> g(sVsIdMu);
      auto [it, inserted] = sVsIds.emplace(vsHash, sNextVsId);
      if (inserted) {
        // The colour lattice holds 124 usable codes (see vsDebugIdToColor).
        // Saturate rather than wrap: a run that somehow exceeds it degrades
        // into "everything past 124 shares id 124" instead of silently
        // aliasing onto an already-legended colour, which would make the
        // legend quietly wrong rather than obviously incomplete.
        if (sNextVsId < 124) {
          ++sNextVsId;
        }
        isNew = true;
      }
      assigned = it->second;
    }

    if (isNew) {
      // Reproduce vsDebugIdToColor() from geometry_resolver.slangh exactly.
      // The 5-level lattice quantises to these values on both sides, so the
      // logged RGB is literally what the shader paints -- and, unlike the hue
      // sweep this replaced, a channel misread by up to ~30 still decodes to
      // the right id.
      static constexpr uint32_t kLevels[5] = { 0u, 64u, 128u, 191u, 255u };
      const uint32_t r = kLevels[assigned % 5];
      const uint32_t g = kLevels[(assigned / 5) % 5];
      const uint32_t b = kLevels[(assigned / 25) % 5];

      Logger::info(str::format(
        "[VsColor] id=", assigned,
        " vs=0x", std::hex, static_cast<uint64_t>(vsHash), std::dec,
        " rgb=(", r, ",", g, ",", b, ")",
        " — DEBUG_VIEW_VERTEX_SHADER_ID (861) paints this VS this colour;"
        " decode a screenshot by rounding each channel to {0,64,128,191,255}"
        " -> levels, then id = lr + 5*lg + 25*lb"));
    }

    return assigned;
  }

  // NV-DXVK [DropTrace]: per-frame dropship submit counter, defined at
  // namespace dxvk scope in rtx_scene_manager.cpp. Must be declared at
  // namespace scope (NOT block scope) so it resolves to dxvk::g_dropTrace*
  // rather than the global ::g_dropTrace* (which fails to link).
  extern std::atomic<uint32_t> g_dropTraceFrame;
  extern std::atomic<uint32_t> g_dropTraceSubmits;
  // NV-DXVK [DropTrace] RAW: dropship draws entering D3D11Rtx::SubmitDraw (set
  // in d3d11_rtx.cpp); compared against g_dropTraceSubmits (draws that survived
  // to submitDrawState) to split engine-cull vs Remix-SubmitDraw-drop.
  extern std::atomic<uint32_t> g_dropTraceRawFrame;
  extern std::atomic<uint32_t> g_dropTraceRawSubmits;

  // NV-DXVK [VanishEdge]: Main-camera pose stashed by [CullCmp] (rtx_camera_manager.cpp),
  // read by the ship-vanish edge detector below.
  extern float g_veCamPx, g_veCamPy, g_veCamPz, g_veCamDx, g_veCamDy, g_veCamDz, g_veCamFov;
  extern std::atomic<uint32_t> g_veCamFrame;

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

    // NV-DXVK [tf2StableBackfaceCull]: by default the winding basis is the
    // objectToWorld mirror parity alone. That is unstable for sub-view
    // billboard/skinned cards whose o2w determinant sign flips as they reorient
    // (toggling which face is culled → a dark back face leaks in). When the
    // option is on, use the NET object->projection winding parity instead,
    // which is what the raster game culls on and is stable. det(o2w*w2p) sign
    // == det(o2w) XOR det(w2p), so this is identical to the current rule when
    // the projection is non-mirrored (w2pMirror=0) — all normal main-camera
    // geometry is unchanged; only mirrored/sub-view projections differ.
    bool windingMirrorBasis = objectToWorldMirrored;
    if (RtxOptions::tf2StableBackfaceCull()) {
      const Matrix4 worldToProjection =
        drawCall.getTransformData().viewToProjection * drawCall.getTransformData().worldToView;
      windingMirrorBasis = objectToWorldMirrored != isMirrorTransform(worldToProjection);
    }

    // Note: Flip front face by setting the front face to counterclockwise, which is the opposite of Vulkan ray tracing's clockwise default.
    if (drawClockwise != windingMirrorBasis)
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
          " ignAC=", drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0,
          " catFlagsRaw=0x", std::hex, drawCall.getCategoryFlags().raw(), std::dec,
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

    // NV-DXVK [sub-view skybox cull]: 3D-skybox geometry reprojected by
    // SetSkyCategoryFromCb2 is tagged IgnoreAntiCulling. It passes through a
    // 90-degree per-instance rotation + scale-1000 reproject, but the
    // FLIP_FACING decision above is computed from objectToWorld alone and
    // does not model the per-instance transform or the reproject — so opaque
    // (blend=0, cullMode=BACK) skybox mountains end up with inverted winding
    // and get fully backface-culled (invisible). Force them double-sided: a
    // skybox backdrop being double-sided is standard and harmless, and it
    // sidesteps the winding ambiguity entirely.
    if (drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling)) {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    // NV-DXVK [flipSubViewSkyboxNormals]: the sub-view reproject submits 3D-skybox
    // geometry with INVERTED WINDING. These draws have no vertex-normal buffer
    // (confirmed via [FlipNormalDiag] hasNormalBuffer=0), so the shading normal is
    // the GEOMETRIC/face normal derived from winding — which therefore points the
    // wrong way (N·L<0 -> black despite unshadowed sun + valid albedo). Toggle the
    // FLIP_FACING bit to correct the winding so the geometric normal faces the sun.
    // (Culling is already disabled for this geometry, so this only affects the
    // shading-side front/back determination, not visibility.)
    if (RtxOptions::flipSubViewSkyboxNormals() &&
        (drawCall.getTransformData().isSubView || drawCall.getTransformData().isSubViewSkybox)) {
      flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
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
    // NV-DXVK: must travel WITH m_previousSurfaceIndex — the two are one
    // value (base + length of the surface-slot range owned last frame). A copy
    // that inherited the base but reset the length to its default would map
    // only the first slot of a PointInstancer range for a frame, which is the
    // exact bug the range mapping in AccelManager was added to fix.
    , m_previousSurfaceCount(src.m_previousSurfaceCount)
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
    , m_categoryFlags(src.m_categoryFlags)
    // NV-DXVK: was missing. m_stablePropId is the SpatialMap dedup key (see its
    // declaration). Left at 0 the clone falls back to XXH64(matrix), which is
    // exactly the drift-prone path the stable-prop ID was added to replace, and
    // clones are sub-view-reprojected content - the case it was added FOR.
    , m_stablePropId(src.m_stablePropId)
    // NV-DXVK: was missing. This mirrors the FLIP_FACING bit of m_vkInstance.flags
    // (set from it at updateInstance), and m_vkInstance IS copied - so leaving
    // this false made the clone internally inconsistent with its own flags.
    // Currently only read by diagnostics and the game capturer, so copying it is
    // low risk and removes the contradiction.
    , isFrontFaceFlipped(src.isFrontFaceFlipped)
    // NV-DXVK [FirstBakeHold]: clones inherit the hold. m_linkedBlas is copied
    // above, so the clone renders from the same (possibly still source-pending)
    // entry as its source — without the stash it would show the collapsed
    // first bake for a frame, the exact artifact the hold exists to prevent.
    // Rc copy just addrefs; the stamp site releases it per-instance once the
    // bake lands.
    , m_prevBlasKeepAlive(src.m_prevBlasKeepAlive) {
    // NV-DXVK [2026-07-26]: m_isSubsurface is skipped DELIBERATELY, and not
    // because it is harmless. It is a genuine divergence: createInstanceCopy
    // never calls updateInstance, so a clone of a subsurface instance reports
    // isSubsurface()==false and is routed as non-subsurface.
    //
    // Copying it was tried and FROZE THE GAME (hard GPU hang during load).
    // Reason: isSubsurface() does not merely report. It
    //   - gates BLAS bucket compatibility          (rtx_accel_manager.cpp:209)
    //   - ALSO pushes the instance into m_mergedInstances[Tlas::SSS]   (:2184)
    //   - reserves an EXTRA PointInstancerBatch with tlasType=Tlas::SSS,
    //     bumping m_pointInstancerSlotsPerType[Tlas::SSS]              (:6193)
    // That last one shifts every per-type byte offset that
    // dispatchPointInstancerCulling writes into m_vkInstanceBuffer, so turning
    // it on for clones corrupts AS instance data unless the SSS slot accounting
    // and TLAS build are verified to cover them - cf. the null-AS ->
    // VK_ERROR_DEVICE_LOST warning in rtx_context.cpp:2431.
    //
    // So this is a KNOWN latent bug, not an oversight. Fixing it properly means
    // auditing the SSS region sizing in dispatchPointInstancerCulling FIRST and
    // confirming Tlas::SSS is actually built when clones land in it. Do not
    // simply add it to the init list above.
    //
    // Members for which state carry over is intentionally skipped
    /*
       m_isSubsurface  (see the note above - deliberate, and load-bearing)
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
      // 768 -> 792 on 2026-07-26, the first time a non-Debug build compiled this.
      // Audited all members against the copy ctor and the skip list below it:
      // m_isSubsurface, m_stablePropId and isFrontFaceFlipped were in NEITHER.
      // m_stablePropId and isFrontFaceFlipped are now copied. m_isSubsurface is
      // NOT - copying it froze the game; see the long note in the copy ctor.
      //
      // 792 -> 800 on 2026-07-29: RtSurface gained uint16_t vsDebugId ([VsColor],
      // DEBUG_VIEW_VERTEX_SHADER_ID). No copy-ctor change was needed - the ctor
      // takes the whole surface by value via `surface(src.surface)`, so every
      // RtSurface member including this one is already carried to clones. The
      // GPU-side surface is NOT affected: vsDebugId is OR'd into the flags0 word
      // writeGPUData already emitted, so no extra bytes are uploaded per surface.
      //
      // 800 -> 808 on 2026-07-31: added uint32_t m_previousSurfaceCount, the
      // length of the surface-slot range this instance owned last frame. Only
      // 4 bytes of payload; the other 4 are alignment padding for the members
      // that follow it.
      // COPY CTOR: DONE - it is copied, deliberately and necessarily. It forms
      // a single logical value with m_previousSurfaceIndex (base + length of
      // the range), and a clone that inherited the base while defaulting the
      // length to 1 would map only the first slot of a PointInstancer range,
      // which is the precise bug the range mapping in AccelManager::
      // uploadSurfaceData was added to fix. The two must always travel
      // together - if you ever touch one, touch the other.
      //
      // 808 -> 816 on 2026-08-02: added Rc<PooledBlas> m_prevBlasKeepAlive
      // ([FirstBakeHold] — hold the previous BLAS across a relink whose
      // destination bake is still source-pending; the geometry-flicker fix).
      // COPY CTOR: DONE - clones inherit the hold (see the note in the init
      // list); a clone rendering the collapsed first bake is the artifact the
      // member exists to prevent.
      static_assert(RtInstanceSize == 816, "RtInstance size has changed.  Fix the copy constructor above this message, then update the expected size.");
    };
    CheckRtInstanceSize<sizeof(RtInstance)> _rtInstanceSizeTest;
  }
 #endif

  void RtInstance::setBlas(BlasEntry& blas) {
    m_linkedBlas = &blas;
  }

  // NV-DXVK [MapGate]: why is the SpatialMap never written?
  //
  // [MapWrite] logs ZERO lines across an entire run, from either write site,
  // while [InstReap]/[FindSim]/[MtnDedup] in this same file log normally and
  // the tag is not on the denylist. Both write sites sit inside
  // `if (!m_isCreatedByRenderer)`, and the [MapWrite] call is inside that block
  // too - so zero lines cannot distinguish:
  //     (a) these functions are never called for these instances, from
  //     (b) they are called and m_isCreatedByRenderer is always true.
  // Those are different bugs in different places, and guessing between them is
  // what this probe exists to avoid.
  //
  // Consequences of the map staying empty, all measured: getNearestData has
  // zero candidates on 100% of 17402 lookups (spatialMapSize=0, every rejection
  // counter 0), nearest matching hits 0.00%, 93% of reaps carry propId=0x0 so
  // exact matching cannot cover for it, every draw therefore builds a NEW
  // instance, the old one is never touched, age reaches exactly 1.00 and
  // keepN=1 reaps it - ~31 fresh instance ids per frame, and 58% of a stable
  // population replaced every frame.
  //
  // COUNTERS, not per-call lines: these functions run per instance per frame
  // (hundreds), and the log is already 127 MB. One summary line per frame
  // answers the question without adding to that. No VS gate either - this
  // file's own history records every VS gate here turning into a blind spot,
  // because it only sees writes made while the instance is linked to that VS.
  //
  // mapWritesExpected is the field to read: it counts calls that SHOULD have
  // reached the write. If it is > 0 while [MapWrite] stays 0, the write is
  // being skipped for some third reason and (a)/(b) are both wrong. If it is 0
  // with calls > 0, m_isCreatedByRenderer is the gate. If calls are 0, nothing
  // ever asks these instances to move.
  //
  // Counters are atomic; the flush uses exchange, so a concurrent increment can
  // be lost across a frame boundary. That costs a unit or two on a count in the
  // hundreds and is not worth a lock on a per-instance path.
  namespace {
    std::atomic<uint32_t> s_mgFrame       { UINT32_MAX };
    std::atomic<uint32_t> s_mgOtcCalls    { 0u };
    std::atomic<uint32_t> s_mgOtcRenderer { 0u };
    std::atomic<uint32_t> s_mgTpCalls     { 0u };
    std::atomic<uint32_t> s_mgTpRenderer  { 0u };
    std::atomic<uint32_t> s_mgNullBlas    { 0u };
    std::atomic<uint32_t> s_mgUnsetFrame  { 0u };

    void mapGateAccount(uint32_t frame, bool isTeleport, bool isRenderer, bool blasNull) {
      // kInvalidFrameIndex IS UINT32_MAX - the same value used as the "no frame
      // seen yet" seed for s_mgFrame - and a brand-new instance carries it
      // until its first update. Letting it drive a frame transition would flip
      // s_mgFrame to the seed and make the NEXT real frame discard the whole
      // bucket as unseeded. With ~31 fresh instances arriving per frame that
      // would fire constantly and quietly zero the measurement. Such calls are
      // still COUNTED (into the current bucket, and separately as unsetFrame);
      // they just never flush.
      if (frame != kInvalidFrameIndex) {
        uint32_t observed = s_mgFrame.load(std::memory_order_relaxed);
        if (observed != frame && s_mgFrame.compare_exchange_strong(observed, frame)) {
          const uint32_t otc  = s_mgOtcCalls.exchange(0u);
          const uint32_t otcR = s_mgOtcRenderer.exchange(0u);
          const uint32_t tp   = s_mgTpCalls.exchange(0u);
          const uint32_t tpR  = s_mgTpRenderer.exchange(0u);
          const uint32_t nb   = s_mgNullBlas.exchange(0u);
          const uint32_t uf   = s_mgUnsetFrame.exchange(0u);
          if (observed != kInvalidFrameIndex) {
            Logger::info(str::format(
              "[MapGate] f=", observed,
              " onTransformChanged=", otc,
              " otcIsRenderer=", otcR,
              " teleport1=", tp,
              " tpIsRenderer=", tpR,
              " nullBlas=", nb,
              " unsetFrame=", uf,
              " mapWritesExpected=", (otc - otcR) + (tp - tpR)));
          }
        }
      } else {
        s_mgUnsetFrame.fetch_add(1u, std::memory_order_relaxed);
      }

      if (isTeleport) {
        s_mgTpCalls.fetch_add(1u, std::memory_order_relaxed);
        if (isRenderer) {
          s_mgTpRenderer.fetch_add(1u, std::memory_order_relaxed);
        }
      } else {
        s_mgOtcCalls.fetch_add(1u, std::memory_order_relaxed);
        if (isRenderer) {
          s_mgOtcRenderer.fetch_add(1u, std::memory_order_relaxed);
        }
      }
      if (blasNull) {
        s_mgNullBlas.fetch_add(1u, std::memory_order_relaxed);
      }
    }
  }

  void RtInstance::onTransformChanged() {
    // NV-DXVK [MapGate]: account BEFORE the m_isCreatedByRenderer branch, which
    // is the whole point - inside it, this call would be as invisible as
    // [MapWrite] already is.
    mapGateAccount(m_frameLastUpdated, /*isTeleport*/ false,
                   m_isCreatedByRenderer, m_linkedBlas == nullptr);

    // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
    // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
    const auto t = transpose(surface.objectToWorld);
    memcpy(&m_vkInstance.transform, &t, sizeof(VkTransformMatrixKHR));

    if (!m_isCreatedByRenderer) {
      // NOTE: This code would cache instances based on predicted position instead of current position, but in testing it fails too frequently
      // const Vector3 newPos = 2.f * surface.objectToWorld[3].xyz() - surface.prevObjectToWorld[3].xyz();

      // Cache based on current position. Pass m_stablePropId so sub-view-
      // reprojected content (where per-frame transform drift would defeat
      // matrix-bytes hashing) keeps a stable cache key tied to per-prop
      // identity. Default 0 = original matrix-hash behavior.
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();
      const Vector3 newPos = getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);

      // NV-DXVK [Otc2904]: pre-move state for VS_2904d2 instances. If we
      // see m_spatialCacheHash != m_stablePropId here, this move() will
      // erase the propId slot and re-insert at a new slot — that's how an
      // instance can "exist" yet its propId lookup miss. Log first 32 +
      // every 256th to bound output. Only fires when divergence is real.
      const BlasEntry* pBlasOtc = m_linkedBlas;
      if (pBlasOtc != nullptr
          && pBlasOtc->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull
          && m_spatialCacheHash != static_cast<XXH64_hash_t>(m_stablePropId)
          && m_stablePropId != 0) {
        static thread_local uint32_t sOtcProbe = 0;
        if (sOtcProbe < 32 || (sOtcProbe & 0xFF) == 0) {
          Logger::warn(str::format(
            "[Otc2904] #", sOtcProbe,
            " spatialCacheHash=0x", std::hex, m_spatialCacheHash, std::dec,
            " stablePropId=0x", std::hex, m_stablePropId, std::dec,
            " DIVERGENCE — about to erase old slot + insert new"));
        }
        sOtcProbe += 1;
      }

      // NV-DXVK [MapWrite]: log every SpatialMap write for the probe VSes, from
      // the WRITE side. The tree-billboard flicker is caused by the tree's map
      // holding a SKY-SPACE entry (~29000 units from the main-world query, owner
      // = the tree's own VS, ownerFrame = curFrame-1) which the 300-unit
      // distance test then rejects, spawning a fresh instance.
      //
      // The writer could not be found from the read side: [MtnDedup] fires on
      // every draw of this VS and logged ZERO sky-space centroids across 1630
      // rows, and the m_delayedRayTracedSky reproject path -- the candidate the
      // handoff named -- is dead code here, since skyReprojectToMainCameraSpace
      // defaults false, making l_forceRaster() unconditionally true so the push
      // is never reached ([SkyTrace.delayPush] fires 0 times with the log filter
      // now un-blanketed). Whatever writes the entry therefore does not pass
      // through findSimilarInstance, so instrument insert/move themselves.
      //
      // isRenderer distinguishes an instance the renderer synthesised from one
      // built from a game draw; cam names the camera the instance was submitted
      // under. Together they identify the path without guessing.
      // Log AFTER the call so the resulting cache KEY can be printed. Joining
      // [MapWrite] to [MapDump2] by blasPtr is unsound -- BlasEntries are
      // destroyed and reallocated constantly, so one address refers to many
      // different objects over a run. The key is derived from the transform
      // bytes (propId is 0 for these props), so a sky-space transform and a
      // main-world one cannot share a key. A key in [MapDump2] with no matching
      // [MapWrite] proves a writer outside these two sites.
      const XXH64_hash_t oldKey = m_spatialCacheHash;
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().move(
          m_spatialCacheHash, newPos, firstInstanceObjectToWorld, this, m_stablePropId);

      // UNGATED per-frame census: one line per instance per frame. No VS gate
      // (every VS gate in this investigation was a blind spot -- it only saw
      // writes made while the instance was linked to that VS's BlasEntry) and no
      // visibility gate either, because m_surfaceIndex / mask are assigned later
      // in the frame by AccelManager and gating on them emitted nothing at all.
      // hidden/frustum/mask/surfIdx are logged as FIELDS so the on-screen subset
      // can be selected offline without dropping objects at capture time.
      {
        Logger::info(str::format(
          "[MapWrite] f=", m_frameLastUpdated,
          " hidden=", (censusHidden() ? 1 : 0),
          " inFrustum=", (censusInFrustum() ? 1 : 0),
          " mask=0x", std::hex, censusMask(), std::dec,
          " surfIdx=", censusSurfaceIndex(),
          " op=move vs=0x", std::hex,
            static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash), std::dec,
          " seenCams=0x", std::hex, static_cast<uint32_t>(m_seenCameraTypes.raw()), std::dec,
          " isRenderer=", (m_isCreatedByRenderer ? 1 : 0),
          " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
          " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(m_linkedBlas), std::dec,
          " oldKey=0x", std::hex, static_cast<uint64_t>(oldKey), std::dec,
          " key=0x", std::hex, static_cast<uint64_t>(m_spatialCacheHash), std::dec,
          " mapSz=", m_linkedBlas->getSpatialMap().size(),
          " newPos=(", newPos.x, ",", newPos.y, ",", newPos.z, ")",
          " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
            firstInstanceObjectToWorld[3][1], ",",
            firstInstanceObjectToWorld[3][2], ")"));
      }
    }
  }

  bool RtInstance::teleport(const Matrix4& objectToWorld) {
    // NV-DXVK [MapGate]: the OTHER write site, and the only one that can seed a
    // brand-new entry. Accounted before its own m_isCreatedByRenderer branch,
    // for the same reason as onTransformChanged.
    mapGateAccount(m_frameLastUpdated, /*isTeleport*/ true,
                   m_isCreatedByRenderer, m_linkedBlas == nullptr);

    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = objectToWorld;
    if (!m_isCreatedByRenderer) {
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();
      const Vector3 centroid = getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
      // NV-DXVK [MapWrite]: see the note at the move() site in
      // onTransformChanged. teleport() is the OTHER way an entry enters the map,
      // and unlike move() it can seed a brand-new entry outright.
      // See the note at the move() site: log after the call so the resulting
      // cache key is available for an exact join against [MapDump2].
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().insert(
          centroid, firstInstanceObjectToWorld, this, m_stablePropId);

      // See the move() site: ungated census, visibility logged as fields.
      {
        Logger::info(str::format(
          "[MapWrite] f=", m_frameLastUpdated,
          " hidden=", (censusHidden() ? 1 : 0),
          " inFrustum=", (censusInFrustum() ? 1 : 0),
          " mask=0x", std::hex, censusMask(), std::dec,
          " surfIdx=", censusSurfaceIndex(),
          " op=teleport vs=0x", std::hex,
            static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash), std::dec,
          " seenCams=0x", std::hex, static_cast<uint32_t>(m_seenCameraTypes.raw()), std::dec,
          " isRenderer=", (m_isCreatedByRenderer ? 1 : 0),
          " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
          " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(m_linkedBlas), std::dec,
          " key=0x", std::hex, static_cast<uint64_t>(m_spatialCacheHash), std::dec,
          " mapSz=", m_linkedBlas->getSpatialMap().size(),
          " newPos=(", centroid.x, ",", centroid.y, ",", centroid.z, ")",
          " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
            firstInstanceObjectToWorld[3][1], ",",
            firstInstanceObjectToWorld[3][2], ")"));
      }
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
    // NV-DXVK [InstClearProbe]: gameplay-only. SceneManager::clear() is the
    // only known external caller and has its own [SceneClearProbe], but
    // 109 instances vanished between f=1176 and f=1178 with no probe
    // firing — log here so we catch any third caller, plus a stack-trace
    // hint (the frame ID + pre-clear size) for correlation. Skip the load
    // window where clears are expected and high-volume.
    if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      const uint32_t curF = m_device->getCurrentFrameId();
      Logger::warn(str::format(
        "[InstClearProbe] f=", curF,
        " preSize=", m_instances.size(),
        " — InstanceManager::clear() entered during gameplay"));
    }

    for (RtInstance* instance : m_instances) {
      removeInstance(instance);
      delete instance;
    }

    m_instances.clear();
    m_viewModelCandidates.clear();
    m_playerModelInstances.clear();
  }

  void InstanceManager::garbageCollection() {
    // NV-DXVK [perf]: master gate for the per-frame instance-census diagnostics
    // (GcEntry / SubViewVsCensus / MtnCensus / HullCensus / DropTrace / RiddenTrace)
    // left over from the mountain / vanishing-ship / dropship investigations. Each is
    // a full O(N) walk over m_instances plus str::format+Logger EVERY frame on the
    // RT-branch thread — measured at gc=67-87ms/frame for ~558 instances in
    // [Perf.PrepScene], i.e. the single biggest chunk of prepareSceneData. The walks
    // are read-only (no effect on the real reaper below); flip to true to re-enable
    // any of these investigations.
    constexpr bool kEnableGcCensus = false;

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

    // NV-DXVK [ReapJoin] — THE PER-MESH DRAW-ARRIVAL JOIN.
    //
    // The open question after the 2026-07-31 05:00 measurement: 437 on-screen
    // meshes are destroyed and recreated on a ~5-frame cycle, continuously,
    // and joining [InstReap] against [BulkPush] showed the SHADER was submitted
    // on 100% of reap frames. But BulkPush is per-VS, and a VS draws many
    // meshes, so that could not prove THIS mesh's draw arrived.
    //
    // BlasEntry::frameLastTouched closes it exactly. It is set unconditionally
    // in SceneManager::processDrawCallState (rtx_scene_manager.cpp:2952) for
    // every arriving draw, keyed on the geometry hash — i.e. it means "a draw
    // for THIS geometry arrived this frame", per mesh, not per shader.
    //
    // ORDERING IS VERIFIED, and it is the whole reason this probe is valid:
    // garbageCollection() runs from inside prepareSceneData
    // (rtx_scene_manager.cpp:4009), which runs AFTER the frame's draws have
    // been processed. So frameLastTouched is already stamped with the current
    // frame by the time we reap. If GC ran first this would read one frame
    // stale on every line and report a false negative every time.
    //
    // READ IT:
    //   drew=1  the draw for this exact mesh ARRIVED this frame and we
    //           destroyed the instance anyway => the draw was not matched to
    //           the existing instance. A MATCHING bug, in findSimilarInstance
    //           / the spatial map, and it is ours.
    //   drew=0  no draw for this geometry this frame => a genuine per-mesh
    //           submission gap, upstream of Remix.
    // Mixed => split by vs= and treat the two populations separately; do not
    // average them.
    uint32_t probeReapDrew   = 0;   // reaped WITH a draw this frame (matching bug)
    uint32_t probeReapNoDraw = 0;   // reaped with no draw this frame (submission gap)
    if constexpr (kEnableGcCensus) {
      Logger::info(str::format(
        "[GcEntry] f=", probeFrame,
        " keepInstances=", numFramesToKeepInstances,
        " antiCullEnabled=", (RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1 : 0),
        " instCount=", m_instances.size()));
    }

    // NV-DXVK [InstDriftProbe]: catch silent removals between GC passes.
    // Tracking is UNCONDITIONAL so that the first gameplay GC has valid
    // history from the last pre-gameplay GC (otherwise sLastGcExitSize
    // stays UINT32_MAX and we miss the very first drift event). Only the
    // LOG is gameplay-gated. Drift events are naturally rare (one per
    // size-shrink between GCs), so no rate-limit needed.
    static uint32_t sLastGcExitSize  = UINT32_MAX;
    static uint32_t sLastGcExitFrame = UINT32_MAX;
    {
      const uint32_t currentSize = static_cast<uint32_t>(m_instances.size());
      const bool inGameplay = tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      if (inGameplay && sLastGcExitSize != UINT32_MAX && currentSize < sLastGcExitSize) {
        const int32_t drift = static_cast<int32_t>(currentSize) - static_cast<int32_t>(sLastGcExitSize);
        Logger::warn(str::format(
          "[InstDriftProbe] f=", probeFrame,
          " prevGcExitFrame=", sLastGcExitFrame,
          " prevGcExitSize=", sLastGcExitSize,
          " currentSize=", currentSize,
          " drift=", drift,
          " frameJump=", (probeFrame - sLastGcExitFrame),
          " — m_instances shrunk between GC passes; clear() or external removal path active"));
      }
      // sLastGcExit* are updated unconditionally at GcExit below.
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

    // NV-DXVK [SubViewVsCensus]: per-VS bucket counts across ALL
    // sub-view RtInstances (every instance with IgnoreAntiCulling set
    // — that flag is exclusively applied by SetSkyCategoryFromCb2's
    // reproject branch, so it's a clean filter for "sub-view content").
    // The mountain investigation initially scoped to VS_2904d2, but
    // the PropIdTrace log showed 7 distinct VS hashes participating
    // in the sub-view pass:
    //   VS_1baf78e08c4e8fed  (~1933 reproject hits)
    //   VS_2094e03a7a19c026  (~1304)
    //   VS_2f543cd750faaf2d  (~512)
    //   VS_eda5efc125a8ed9c  (~163 — sky dome)
    //   VS_aa5c8f7e8788e1d2  (~17)
    //   VS_c10aa132da51c65b  (~5)
    //   VS_1ddf42076ed0b6d1  (~2)
    //   VS_2904d2163ef31a17  (the prop-mountain shader)
    // The user's "missing mountains" may be in any of these — distant
    // terrain often renders under a different VS than near props.
    // This probe counts INSTANCES per VS hash, so a sudden drop in
    // any bucket is a candidate for the missing-mountain VS.
    if constexpr (kEnableGcCensus) {
      static thread_local uint32_t sSubViewVsLastFrame = UINT32_MAX;
      if (sSubViewVsLastFrame != currentFrame) {
        sSubViewVsLastFrame = currentFrame;
        std::unordered_map<uint64_t, uint32_t> byVsTotal;
        std::unordered_map<uint64_t, uint32_t> byVsHidden;
        std::unordered_map<uint64_t, uint32_t> byVsOutFr;
        for (const RtInstance* pInst : m_instances) {
          if (pInst == nullptr) continue;
          if (!pInst->testCategoryFlags(InstanceCategories::IgnoreAntiCulling)) continue;
          const BlasEntry* pBl = pInst->getBlas();
          if (pBl == nullptr) continue;
          const uint64_t h = static_cast<uint64_t>(
            pBl->input.getTransformData().vertexShaderHash);
          byVsTotal[h] += 1;
          if (pInst->m_isHidden)        byVsHidden[h] += 1;
          if (!pInst->m_isInsideFrustum) byVsOutFr[h] += 1;
        }
        for (const auto& kv : byVsTotal) {
          Logger::info(str::format(
            "[SubViewVsCensus] f=", currentFrame,
            " vs=0x", std::hex, kv.first, std::dec,
            " total=", kv.second,
            " hidden=", byVsHidden[kv.first],
            " notInFr=", byVsOutFr[kv.first]));
        }
      }
    }

    // NV-DXVK [MtnCensus]: per-frame visibility census of ALL VS_2904d2
    // mountain instances. The user reports that some mountains are
    // missing while others are visible, in a way that doesn't match
    // simple frustum culling (some missing ones stay missing). The
    // existing per-instance probes are rate-limited to 5/frame, so
    // they only sample ~10% of the 48 mountain population — not enough
    // to spot a consistent missing subset.
    //
    // This probe iterates the FULL m_instances list once per frame,
    // counts every VS_2904d2 instance by visibility-state bucket, and
    // dumps a summary line. Every 60 frames it ALSO dumps each
    // instance's per-instance state so we can see which specific
    // mountains (by propId) are stuck in a non-visible state across
    // many frames. Correlate propId persistence with what the user
    // sees on screen: a propId that's `hidden=1` or `inFrustum=0` in
    // every dump but visually "should be visible" tells us which
    // pipeline stage is dropping it.
    //
    // Cost: O(N) loop over m_instances, ~3000 entries × cheap field
    // reads — negligible vs the GC pass's own loop next.
    //
    // Buckets:
    //   total           — every VS_2904d2 instance
    //   hidden          — m_isHidden==true (sky-classified or
    //                     replacement-asset hidden)
    //   notInFrustum    — m_isInsideFrustum==false (anti-culling-dedup
    //                     decided it's outside the main-cam frustum)
    //   markedGC        — m_isMarkedForGC==true (queued for removal
    //                     this pass)
    //   ignAntiCull     — has InstanceCategories::IgnoreAntiCulling
    //                     category set (sub-view sticky-preserve)
    //   staleN          — m_frameLastUpdated < currentFrame (not
    //                     touched THIS frame). N is the gap.
    if constexpr (kEnableGcCensus) {
      static thread_local uint32_t sCensusLastDumpFrame = UINT32_MAX;
      const bool dumpPerInstance =
        (currentFrame % 60u == 0u) && (sCensusLastDumpFrame != currentFrame);

      uint32_t total       = 0;
      uint32_t hidden      = 0;
      uint32_t notInFr     = 0;
      uint32_t markedGC    = 0;
      uint32_t ignAntiCull = 0;
      uint32_t staleAny    = 0;   // updated < currentFrame
      uint32_t staleMany   = 0;   // gap > 1

      uint32_t perInstIdx = 0;
      for (const RtInstance* pInst : m_instances) {
        if (pInst == nullptr) continue;
        const BlasEntry* pBlasCensus = pInst->getBlas();
        if (pBlasCensus == nullptr) continue;
        if (pBlasCensus->input.getTransformData().vertexShaderHash != 0x2904d2163ef31a17ull) continue;

        total += 1;
        const bool isHidden_   = pInst->m_isHidden;
        const bool inFr_       = pInst->m_isInsideFrustum;
        const bool gcMark_     = pInst->m_isMarkedForGC;
        const bool ignAC_      = pInst->testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
        const uint32_t gap     = (currentFrame > pInst->m_frameLastUpdated)
                                 ? (currentFrame - pInst->m_frameLastUpdated) : 0u;
        if (isHidden_) hidden      += 1;
        if (!inFr_)    notInFr     += 1;
        if (gcMark_)   markedGC    += 1;
        if (ignAC_)    ignAntiCull += 1;
        if (gap > 0u)  staleAny    += 1;
        if (gap > 1u)  staleMany   += 1;

        if (dumpPerInstance) {
          // Pull o2w.translation directly from the instance's current
          // worldToObject inverse — that's the TLAS-bound position.
          // If reproject worked, |t.z| should be large (>10000); if
          // the multi-shader-variant fanout clobbered it with raw
          // sub-view-local data, |t.z| stays ~15600 (the sky-anchor)
          // and the mountain renders below the visible world.
          const Matrix4& o2w = pInst->getTransform();
          const float tx = float(o2w[3][0]);
          const float ty = float(o2w[3][1]);
          const float tz = float(o2w[3][2]);
          // Reproject sanity flag: a correctly-reprojected sub-view
          // mountain ends up many thousands of units from origin in
          // main-world space (the sub-view layout × scale). Raw
          // sub-view-local coords are near the engine-sky-anchor
          // (small XY, Z near -15616). If |tz| < 20000 the o2w is
          // suspiciously close to the anchor — flag for inspection.
          const bool tzLooksRaw = (std::abs(tz) < 20000.0f);
          // NV-DXVK [renderability triad]: the user reports SOME mountain
          // instances render correctly and others are entirely absent.
          // All 22 instances exist (not hidden / GC'd), so the missing
          // ones fail BETWEEN instance and screen. Log the three gates
          // that decide it: VkInstance mask (0 => no ray can hit it),
          // BLAS primitive count (0 => BLAS not built / empty geometry),
          // and surfaceIndex (kInvalid => not bound into the surface
          // table). Whichever differs between present and absent
          // mountains is the actual cause.
          const uint32_t mtnMask = pInst->getVkInstance().mask;
          uint32_t mtnBlasPrim = 0u;
          if (pBlasCensus != nullptr && !pBlasCensus->buildRanges.empty()) {
            mtnBlasPrim = pBlasCensus->buildRanges[0].primitiveCount;
          }
          const uint32_t mtnVtx = (pBlasCensus != nullptr)
            ? pBlasCensus->modifiedGeometryData.vertexCount : 0u;
          Logger::info(str::format(
            "[MtnCensus.Inst] f=", currentFrame,
            " #", perInstIdx,
            " propId=0x", std::hex, pInst->m_stablePropId, std::dec,
            " hidden=", (isHidden_ ? 1 : 0),
            " inFrustum=", (inFr_ ? 1 : 0),
            " ignAntiCull=", (ignAC_ ? 1 : 0),
            " markedGC=", (gcMark_ ? 1 : 0),
            " lastUpdGap=", gap,
            " mask=0x", std::hex, mtnMask, std::dec,
            " blasPrim=", mtnBlasPrim,
            " vtx=", mtnVtx,
            " surfIdx=", pInst->getSurfaceIndex(),
            " category=0x", std::hex,
              static_cast<uint64_t>(pInst->getCategoryFlags().raw()),
              std::dec,
            " o2w.t=(", tx, ",", ty, ",", tz, ")",
            " tzLooksRaw=", (tzLooksRaw ? 1 : 0)));
          perInstIdx += 1;
        }
      }

      // Summary line every frame — cheap to skim.
      if (total > 0u) {
        Logger::info(str::format(
          "[MtnCensus] f=", currentFrame,
          " total=", total,
          " hidden=", hidden,
          " notInFrustum=", notInFr,
          " markedGC=", markedGC,
          " ignAntiCull=", ignAntiCull,
          " staleAny=", staleAny,
          " staleMany=", staleMany));
      }
      if (dumpPerInstance) sCensusLastDumpFrame = currentFrame;
    }

    // NV-DXVK [HullCensus]: the "vanishing ship" is a PRIMARY-VISIBILITY problem — it
    // disappears in the raw Diffuse Albedo debug view, which is written in the GBuffer pass
    // BEFORE path tracing and BEFORE the denoiser. So the denoiser/confidence/MV chase was the
    // wrong layer. A view-direction-gated, per-frame, instant G-buffer vanish means the hull is
    // either (a) dropped from the TLAS as an instance, or (b) in the TLAS but its triangles are
    // backface-culled at ray time depending on view angle. This census discriminates the two,
    // per frame, for the Widow hull's two vertex shaders. Correlate by frame with [ShipDir] f=N
    // (camera forward) to see which mechanism flips with the good/bad look direction.
    //
    //   present-but-vanished  -> mask!=0 & inFrustum & !hidden & blasPrim>0 & surfIdx valid
    //                            => ray-time BACKFACE cull (FLIP/CULLDISABLE decision wrong).
    //   instance-level cull   -> hidden=1 / inFrustum=0 / mask=0 / surfIdx invalid
    //                            => frustum/categorization/GC dropping it (wrong-camera suspect).
    //
    // VS aggregates are useless here (these VS hashes are shared with sky+floor); this iterates
    // INSTANCES, and each hull instance's world position (o2w.t) lets us track a specific piece
    // across frames regardless of the shared shader.
    if constexpr (kEnableGcCensus) {
      // NV-DXVK [StudioModelHook] re-gate: census now keys on the Widow engine
      // model (pBl->input.isWidowModel) — the shared VS-hash constants are gone.
      uint32_t hTotal = 0u, hHidden = 0u, hNotFr = 0u, hMaskZero = 0u, hSurfBad = 0u, hCullActive = 0u;
      uint32_t hIdx = 0u;
      // NV-DXVK [DropTrace]: dropship-specific per-frame aggregation, paired
      // with the submitDrawState arrival counter (g_dropTrace*, declared at
      // namespace scope above, defined in rtx_scene_manager.cpp).
      uint32_t dropInst = 0u, dropPresent = 0u, dropMaxBlas = 0u;
      // [RiddenTrace] SESSION-J: the RIDDEN ship = widow/Crow submeshes at o2w.t≈origin
      // (you're inside it; Remix recenters the world at the camera, so it lands at the
      // origin — the formation ship is 1000+ units away). One self-contained line per GC
      // walk so localization needs NO cross-line frame matching (the f= counter aliases
      // under heavy logging). It tells exactly where the ship drops:
      //   blasPrim0>0            -> geometry->BLAS dropped it (empty BLAS)
      //   hidden>0 / maskZero>0  -> excluded from the TLAS (won't be ray-traced)
      //   notInFrustum>0         -> Remix frustum-culled
      //   renderable>0 yet still invisible on screen -> it IS in the TLAS => the vanish
      //                            is shading/a pass/denoiser, NOT geometry.
      uint32_t rN = 0u, rRender = 0u, rHidden = 0u, rMaskZero = 0u, rNotFr = 0u, rBlas0 = 0u, fN = 0u;
      uint32_t rMinBlas = 0xFFFFFFFFu, rMaxBlas = 0u;
      for (const RtInstance* pInst : m_instances) {
        if (pInst == nullptr) continue;
        const BlasEntry* pBl = pInst->getBlas();
        if (pBl == nullptr) continue;
        const uint64_t vs = static_cast<uint64_t>(pBl->input.getTransformData().vertexShaderHash);
        // NV-DXVK [HullCensus] re-widen: the VISIBLE ship the user stands in is
        // VS=0x292b / name=(none) — WORLD geometry, NOT the studio Widow — and
        // it vanishes by dropping out of the TLAS (box rays hit the 3D skybox,
        // [ShipBox] meanViewZ jumps 1->200000 svSky=1). Census ALL 0x292b
        // instances (plus any widow-tagged) with name/mat/tex per line so a
        // good-vs-vanished diff shows which instance drops out and how
        // (hidden/mask/inFrustum/markedGC). Capped per frame to bound volume.
        if (vs != 0x292b6ba0d1854f28ull && !pBl->input.isWidowModel) continue;
        if (hIdx >= 120u) break;

        const bool isHidden_ = pInst->m_isHidden;
        const bool inFr_     = pInst->m_isInsideFrustum;
        const bool gcMark_   = pInst->m_isMarkedForGC;
        const uint32_t mask  = pInst->getVkInstance().mask;
        const uint32_t iflags= pInst->getVkInstance().flags;  // VkGeometryInstanceFlagBitsKHR
        const bool flipFace  = (iflags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR) != 0u;
        const bool cullOff   = (iflags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR) != 0u;
        const bool cullActive= !cullOff;  // if true, back-facing tris ARE culled at ray time -> directional vanish possible
        uint32_t blasPrim = 0u;
        if (!pBl->buildRanges.empty()) blasPrim = pBl->buildRanges[0].primitiveCount;
        // NV-DXVK [DropTrace]: aggregate the dropship sub-meshes specifically.
        if (pBl->input.studioModelName[0] != '\0'
            && (std::strstr(pBl->input.studioModelName, "Crow_dropship") != nullptr
             || std::strstr(pBl->input.studioModelName, "widow") != nullptr)) {
          dropInst++;
          if (blasPrim > 0u) dropPresent++;
          if (blasPrim > dropMaxBlas) dropMaxBlas = blasPrim;
        }
        const uint32_t vtx = pBl->modifiedGeometryData.vertexCount;
        // NV-DXVK [GeoChain] SESSION-E: the dropship is NOT culled — the engine
        // submits full geometry every frame ([DropGeo] ext01 count=80988 even at
        // present=0) but Remix builds a 0-primitive BLAS in the bad view
        // (maxBlasPrim 0<->26996 on identical input). Localize WHERE the count
        // collapses by logging the whole chain: engine-submitted raster counts
        // (input) -> raytrace-ready counts (modified) -> BLAS primitiveCount.
        //   inIdx>0 & modIdx==0  -> Remix's triangle-list/index gen zeroes it
        //   inIdx>0 & modIdx>0 & blasPrim==0 -> BLAS build drops all (degenerate tris)
        //   objBBvalid=0 / collapsed extent -> skinned verts collapsed (NaN/degenerate)
        const uint32_t inVtx  = pBl->input.getGeometryData().vertexCount;
        const uint32_t inIdx  = pBl->input.getGeometryData().indexCount;
        const uint32_t modIdx = pBl->modifiedGeometryData.indexCount;
        const uint32_t surfIdx = pInst->getSurfaceIndex();
        const bool surfBad = (surfIdx == SURFACE_INDEX_INVALID);
        const Matrix4& o2w = pInst->getTransform();
        const float tx = float(o2w[3][0]), ty = float(o2w[3][1]), tz = float(o2w[3][2]);
        // NV-DXVK [HullCensus] WORLD coordinates. o2w.t alone is useless for BSP-style geometry
        // (identity transform, verts carry world coords). Transform the object-space geometry
        // AABB by o2w to get the instance's actual WORLD position + extent. Diff between visible
        // and vanish frames: if the ship's world centroid/AABB MOVES when it vanishes, the
        // geometry is being teleported; if it's identical, the geometry sits still and the vanish
        // is purely a render/occlusion issue.
        const AxisAlignedBoundingBox& objBB = pBl->input.getGeometryData().boundingBox;
        const Vector3 wCen = objBB.getTransformedCentroid(o2w);
        const Vector3 wMinC = objBB.isValid() ? (o2w * Vector4(objBB.minPos, 1.0f)).xyz() : Vector3(0.f);
        const Vector3 wMaxC = objBB.isValid() ? (o2w * Vector4(objBB.maxPos, 1.0f)).xyz() : Vector3(0.f);
        const uint32_t gap = (currentFrame > pInst->m_frameLastUpdated)
                             ? (currentFrame - pInst->m_frameLastUpdated) : 0u;

        hTotal++;
        if (isHidden_) hHidden++;
        if (!inFr_)    hNotFr++;
        if (mask == 0u) hMaskZero++;
        if (surfBad)   hSurfBad++;
        if (cullActive) hCullActive++;

        // [RiddenTrace] accumulation: widow/Crow studio submesh at the recentered origin
        // is the ridden ship; far ones are the formation ship. Squared dist (no sqrt).
        {
          const bool isDropName = (pBl->input.studioModelName[0] != '\0'
              && (std::strstr(pBl->input.studioModelName, "Crow_dropship") != nullptr
               || std::strstr(pBl->input.studioModelName, "widow") != nullptr));
          if (isDropName) {
            const float o2wDist2 = tx * tx + ty * ty + tz * tz;
            if (o2wDist2 < 40000.0f) {                 // < 200u from recentered origin = ridden
              rN++;
              if (isHidden_)   rHidden++;
              if (mask == 0u)  rMaskZero++;
              if (!inFr_)      rNotFr++;
              if (blasPrim == 0u) rBlas0++;
              if (blasPrim < rMinBlas) rMinBlas = blasPrim;
              if (blasPrim > rMaxBlas) rMaxBlas = blasPrim;
              if (!isHidden_ && mask != 0u && blasPrim > 0u && inFr_) rRender++;
            } else {
              fN++;                                     // formation ship (far)
            }
          }
        }

        // Hull has few instances — log each every frame (not throttled). The per-instance
        // line is what we diff between the good and bad look direction.
        Logger::info(str::format(
          "[HullCensus.Inst] f=", currentFrame,
          " #", hIdx,
          " vs=0x", std::hex, vs, std::dec,
          " name=", (pBl->input.studioModelName[0] ? pBl->input.studioModelName : "(none)"),
          " mat=0x", std::hex, static_cast<uint64_t>(pBl->input.getMaterialData().getHash()), std::dec,
          // NV-DXVK: do NOT read getColorTexture().getImageHash() here. This
          // census runs inside InstanceManager::garbageCollection, concurrent
          // with the texture-streaming/GC thread. The isValid()+!isImageEmpty()
          // guard is NOT sufficient: streaming can free ManagedTexture::
          // m_currentMipView BETWEEN the isImageEmpty() check and getImageHash(),
          // so getImageHash() then derefs a dangling Rc<DxvkImage> → AV in
          // Rc::operator-> (util_rc_ptr.h:92). Confirmed by VS callstack:
          // getImageHash (rtx_texture.h:133) ← garbageCollection:963. A strong-
          // Rc copy does NOT help — the copy's incRef is itself the crashing
          // deref. mat=getHash() above is a plain XXH64 value (no image deref)
          // and is safe; the per-instance texture hash is simply omitted from
          // the GC-time walk. (See memory: getImageHash GC crash.)
          " tex=<skip-gc-uaf>",
          " hidden=", (isHidden_ ? 1 : 0),
          " inFrustum=", (inFr_ ? 1 : 0),
          " markedGC=", (gcMark_ ? 1 : 0),
          " lastUpdGap=", gap,
          " mask=0x", std::hex, mask, std::dec,
          " blasPrim=", blasPrim,
          " vtx=", vtx,
          // [GeoChain] engine-submitted -> raytrace-ready -> BLAS, + object-space
          // AABB validity/extent (collapsed => skinned verts degenerate).
          " inVtx=", inVtx, " inIdx=", inIdx, " modIdx=", modIdx,
          " objBBvalid=", (objBB.isValid() ? 1 : 0),
          " objBBmin=(", objBB.minPos.x, ",", objBB.minPos.y, ",", objBB.minPos.z, ")",
          " objBBmax=(", objBB.maxPos.x, ",", objBB.maxPos.y, ",", objBB.maxPos.z, ")",
          " surfIdx=", surfIdx,
          " cullActive=", (cullActive ? 1 : 0),
          " flipFace=", (flipFace ? 1 : 0),
          " cullDisable=", (cullOff ? 1 : 0),
          " category=0x", std::hex, static_cast<uint64_t>(pInst->getCategoryFlags().raw()), std::dec,
          " o2w.t=(", tx, ",", ty, ",", tz, ")",
          " worldCen=(", wCen.x, ",", wCen.y, ",", wCen.z, ")",
          " worldAABBmin=(", wMinC.x, ",", wMinC.y, ",", wMinC.z, ")",
          " worldAABBmax=(", wMaxC.x, ",", wMaxC.y, ",", wMaxC.z, ")"));
        hIdx++;
      }
      if (hTotal > 0u) {
        Logger::info(str::format(
          "[HullCensus] f=", currentFrame,
          " total=", hTotal,
          " hidden=", hHidden,
          " notInFrustum=", hNotFr,
          " maskZero=", hMaskZero,
          " surfIdxInvalid=", hSurfBad,
          " cullActive=", hCullActive));
      }
      // NV-DXVK [DropTrace]: per-frame dropship fate (logged every frame).
      //   submits     = dropship (Crow/Widow) draws that reached
      //                 submitDrawState THIS frame (runs before this GC)
      //   instances   = dropship instances currently in m_instances
      //   present     = those with blasPrim>0 (i.e. will render)
      //   maxBlasPrim = largest built prim count among them
      // Spin to sustain the despawn, then diff:
      //   submits>0 & present=0 -> draw arrives but BLAS/instance lost
      //                            downstream (scene/accel side)
      //   submits=0            -> draw stops reaching the scene under motion
      //                            (dropped in SubmitDraw / not submitted)
      {
        const uint32_t dtSubmits =
          (g_dropTraceFrame.load(std::memory_order_relaxed) == currentFrame)
            ? g_dropTraceSubmits.load(std::memory_order_relaxed) : 0u;
        // raw = dropship draws that ENTERED SubmitDraw this frame (pre-cascade).
        //   raw==submits -> engine stopped submitting (game-side cull)
        //   raw >submits -> Remix's SubmitDraw cascade dropped them
        const uint32_t dtRaw =
          (g_dropTraceRawFrame.load(std::memory_order_relaxed) == currentFrame)
            ? g_dropTraceRawSubmits.load(std::memory_order_relaxed) : 0u;
        Logger::info(str::format(
          "[DropTrace] f=", currentFrame,
          " raw=", dtRaw,
          " submits=", dtSubmits,
          " instances=", dropInst,
          " present=", dropPresent,
          " maxBlasPrim=", dropMaxBlas));

        // NV-DXVK [VanishEdge]: dump the full state at every ship appear/disappear
        // EDGE (when dropPresent crosses the visible<->gone threshold), so we can
        // characterize the TRIGGER across many transitions instead of fitting one.
        // dropPresent = ship submeshes with blasPrim>0 (actually rendering). Hysteresis
        // (>=4 present, <=2 gone) avoids flicker. Pairs the edge with the Main camera
        // pose ([CullCmp]/RtCamera, stashed in g_veCam*) + the ship counts. The player
        // RIDES the ship, so camPos is the ship's flight path; a fixed world band where
        // it vanishes shows up as the same camPos range each transition.
        {
          static int s_veState = -1;                       // -1 unknown, 0 gone, 1 present
          const int veState = (dropPresent >= 4u) ? 1
                            : (dropPresent <= 2u) ? 0
                            : s_veState;
          if (s_veState != -1 && veState != s_veState) {
            const bool poseFresh = (g_veCamFrame.load(std::memory_order_acquire) != 0xFFFFFFFFu);
            Logger::warn(str::format(
              "[VanishEdge] f=", currentFrame, " ", (veState ? "APPEAR" : "VANISH"),
              " present=", dropPresent, " submits=", dtSubmits,
              " instances=", dropInst, " maxBlasPrim=", dropMaxBlas,
              " camPos=(", g_veCamPx, ",", g_veCamPy, ",", g_veCamPz, ")",
              " camFwd=(", g_veCamDx, ",", g_veCamDy, ",", g_veCamDz, ")",
              " fov=", g_veCamFov, " camFrame=", g_veCamFrame.load(std::memory_order_relaxed),
              " poseFresh=", poseFresh ? 1 : 0));
          }
          if (veState != -1) s_veState = veState;
        }
        // [RiddenTrace]: one self-contained line — everything about the RIDDEN ship in a
        // single Logger call, so no cross-line frame matching. renderable==0 with rN>0
        // pinpoints the stage; renderable>0 while it's gone on screen => shading/pass.
        Logger::info(str::format(
          "[RiddenTrace] f=", currentFrame,
          " riddenSubmeshes=", rN,
          " renderable=", rRender,
          " blasPrim0=", rBlas0,
          " hidden=", rHidden,
          " maskZero=", rMaskZero,
          " notInFrustum=", rNotFr,
          " minBlas=", (rN ? rMinBlas : 0u),
          " maxBlas=", rMaxBlas,
          " formationSubmeshes=", fN,
          " dropPresent=", dropPresent));
      }
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

      // NV-DXVK [GcKeep2904]: log every VS_2904d2 instance evaluated at GC,
      // not just removed ones. Rate-limit per-frame: log first 5 mountain
      // instances per GC pass so we get a representative slice without
      // 48/frame spam. Tells us whether m_isInsideFrustum is correctly
      // false (IgnoreAntiCulling working) or stuck true (BLAS frustum
      // loop didn't re-evaluate after a frame skip), and whether the
      // sticky-IgnoreAntiCulling preserve from updateInstance survived.
      {
        const BlasEntry* pBlasInspect = pInstance->getBlas();
        if (pBlasInspect != nullptr
            && pBlasInspect->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
          static thread_local uint32_t sKeepProbeFrame = UINT32_MAX;
          static thread_local uint32_t sKeepProbeCount = 0;
          if (sKeepProbeFrame != currentFrame) {
            sKeepProbeFrame = currentFrame;
            sKeepProbeCount = 0;
          }
          if (sKeepProbeCount < 5) {
            Logger::info(str::format(
              "[GcKeep2904] #", sKeepProbeCount,
              " f=", currentFrame,
              " lastUpd=", pInstance->m_frameLastUpdated,
              " gap=", (currentFrame - pInstance->m_frameLastUpdated),
              " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
              " ignAntiCull=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
              " enableGC=", (enableGarbageCollection ? 1 : 0),
              " skinned=", (pInstance->getBlas()->input.getSkinningState().numBones > 0 ? 1 : 0),
              " animated=", (pInstance->m_isAnimated ? 1 : 0),
              " playerMdl=", (pInstance->m_isPlayerModel ? 1 : 0),
              " markedGC=", (pInstance->m_isMarkedForGC ? 1 : 0),
              " propId=0x", std::hex, pInstance->m_stablePropId, std::dec));
            sKeepProbeCount += 1;
          }
        }
      }

      // NV-DXVK [GC clause probe]: capture WHICH side of the OR fired and
      // whether this is a sub-view mountain instance. Lets us correlate
      // [Rm2904] entries that show "all conditions false" with the actual
      // GC-time evaluation.
      //
      // NV-DXVK [SubViewKeepLong]: instances tagged IgnoreAntiCulling
      // (TF2 3D-skybox sub-view content) get the longer
      // numFramesToKeepSubViewInstances grant. TF2's engine throttles
      // sub-view rendering — on frames where it doesn't redraw the sub-
      // view fan, those instances aren't touched. With the default
      // numFramesToKeepInstances=1, ONE skipped frame retires the entire
      // sub-view fan together; their ordered-surface slots get
      // reallocated; previous-frame GBuffer pixels referencing the now-
      // retired slots render with the wrong content (large black blocks
      // on mountains / dome). Verified via [Coverage] priorOwnerFrame
      // data: an entire 800k-pixel stale event traced to one frame
      // whose SubViewGateCounts showed 90 fewer candidates than steady
      // state. The longer keep absorbs engine LOD skips up to ~16 frames.
      //
      // NV-DXVK [PropIdKeepLong attempt reverted]: a previous version of
      // this gate extended the long keep to any instance with a non-zero
      // stablePropId. That made things WORSE for the path-10 bone-
      // instanced fanout (VS_2947c6 ~7787 PI slots/frame): its
      // MakeBoneStablePropId propId isn't actually stable across frames
      // (rolls at the i2o[0].T 1u rounding boundary or whenever the
      // engine rotates the per-instance buffer arena), so dedup misses
      // were creating new RtInstances each frame. With keepN=1 the old
      // ones retired fast; with keepN=16 they all stayed alive AND every
      // touched-this-frame instance still calls addPointInstancerBlas →
      // m_reorderedSurfaces doubled to 17155, every subsequent collapse
      // brought DOWN 17000 slots' worth of stale pixels instead of 8500.
      // The right fix lives at the propId producer (stabilize
      // MakeBoneStablePropId across the rolling input), not here. Until
      // that lands, keep the gate strict to IgnoreAntiCulling.
      // NV-DXVK [keepStablePropIdInstancesLong]: shadow-sourced fanout terrain
      // (VS_2947c6) is submitted only via TF2's spot-shadow pass; when that
      // pass culls it, it stops being submitted and would retire at the
      // default keepN=1. With a STABLE identity
      // (rtx.boneStablePropIdFanoutPositionOnly) the longer keep is safe and
      // retains it across the shadow-off frames. Continuously-submitted props
      // are unaffected (their keep window never elapses).
      const bool keepLongForProp =
        RtxOptions::keepStablePropIdInstancesLong()
        && pInstance->m_stablePropId != 0ull;
      const uint32_t instanceKeepN =
        (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) || keepLongForProp)
          ? RtxOptions::numFramesToKeepSubViewInstances()
          : numFramesToKeepInstances;
      const bool clauseLifetime = (forceGarbageCollection || enableGarbageCollection) &&
                                  (pInstance->m_frameLastUpdated + instanceKeepN <= currentFrame);
      const bool clauseMarked = pInstance->m_isMarkedForGC;
      const bool shouldRemove = clauseLifetime || clauseMarked;
      if (shouldRemove) {
        probeRemovedThisPass += 1;
        if (clauseMarked)   probeRemovedMarked   += 1;
        if (clauseLifetime) probeRemovedLifetime += 1;
        if (forceGarbageCollection && !enableGarbageCollection && clauseLifetime) probeRemovedForce += 1;

        // NV-DXVK [ReapJoin]: tally the draw-arrival split for EVERY reap.
        // Deliberately OUTSIDE the [InstReap] detail block below, which is
        // gated on the engine-hook capture counter — the totals must describe
        // every reap in the pass, not only the ones that got a detail line.
        // A summary that silently counts a subset is how the 24-line cap
        // misreported this same churn on 2026-07-29.
        {
          const BlasEntry* pBlasJoin = pInstance->getBlas();
          if (pBlasJoin != nullptr && pBlasJoin->frameLastTouched == currentFrame) {
            probeReapDrew += 1;
          } else {
            probeReapNoDraw += 1;
          }
        }

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
                " keepN=", instanceKeepN,
                " keepNbase=", numFramesToKeepInstances,
                " force=", (forceGarbageCollection ? 1 : 0),
                " enable=", (enableGarbageCollection ? 1 : 0),
                " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
                " ignAntiCull=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
                " markedGC=", (clauseMarked ? 1 : 0),
                " clauseLifetime=", (clauseLifetime ? 1 : 0)));
            }
            sGcVsProbe += 1;
          }
        }

        // NV-DXVK [InstReap]: per-removal detail for ALL VSes — the s2s
        // "two views" flip probe. Theory: view 1's dome + s2s superstructure
        // are intro-era (f~527-531) sub-view instances coasting on the
        // 16-frame numFramesToKeepSubViewInstances grant after the steady
        // stream stopped refreshing them; the one-frame view-2 flip
        // (sky-miss 0%->45%, BlasDestroyed wave peaking x25) is their
        // collective expiry. Readout at the next flip:
        //   keepN=16, age 16-17, lastUpd in the intro band, on the
        //     big-coverage vsHashes (0x2859d250/0x292b6ba0/...) ->
        //     expiry CONFIRMED; the fix is making steady draws resolve to
        //     these instances (or their propId producer), NOT a longer keep.
        //   keepN=1, age~2 on those vsHashes -> the draw stream itself
        //     dropped/rekeyed them (churn) — look upstream of GC.
        // pos= identifies dome (|pos| huge, reprojected) vs deck (~y=-10860).
        // Gameplay-gated; 24 detail lines/frame ([GcExit] keeps the totals).
        if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
          // UNCAPPED. This was 24/frame and it SATURATED on every single frame
          // of the 2026-07-29 capture — 24 reaps/frame, unbroken — so the
          // number in the log was the cap reporting itself, not the scene, and
          // the real churn magnitude was never measured. Every reap is logged
          // now. A count that is the answer must not be clipped by the probe
          // that measures it.
          {
            const BlasEntry* pBlasRp = pInstance->getBlas();
            const XXH64_hash_t vsRp = (pBlasRp != nullptr)
              ? pBlasRp->input.getTransformData().vertexShaderHash : 0ull;
            const Vector3 posRp = pInstance->getWorldPosition();
            // NV-DXVK: blasPtr + that BlasEntry's SpatialMap size at reap time.
            // Two props that share a mesh share a geometry hash and therefore a
            // BlasEntry, and the 2026-07-29 capture showed such a pair — one at
            // a main-world transform, one at a 3D-skybox transform ~26000u away
            // — alternating CREATE every frame on one blasPtr while the map
            // never grows past 1. SpatialMap::insert does not evict (it bumps
            // on hash collision, and [SpatialBump] fired 0 times), so the only
            // way the loser leaves the map is this reap. blasPtr groups the
            // pair; mapSz says whether the survivor was already alone when the
            // reap ran; age vs keepN says whether this is ordinary lifetime
            // expiry (i.e. that prop's draw never arrived) or something else.
            const size_t mapSzRp = (pBlasRp != nullptr)
              ? pBlasRp->getSpatialMap().size() : 0u;
            // mat= is the only field that ties a reap back to a group in the
            // auto_scene OBJ (whose names are vs+mat). vs alone is ambiguous —
            // one VS draws several unrelated meshes.
            const uint64_t matRp = (pBlasRp != nullptr)
              ? static_cast<uint64_t>(pBlasRp->input.getMaterialData().getHash()) : 0ull;
            // v= is the [MeshTrace] identity's second half. vs alone cannot
            // name a mesh (one shader draws many) and mat is not run-stable,
            // so without this a reap cannot be joined to the mesh that
            // vanished from the TLAS.
            const uint32_t vertsRp = (pBlasRp != nullptr)
              ? pBlasRp->modifiedGeometryData.vertexCount : 0u;
            Logger::info(str::format(
              "[InstReap] f=", currentFrame,
              " vs=0x", std::hex, static_cast<uint64_t>(vsRp),
              " mat=0x", matRp, std::dec,
              " v=", vertsRp, std::hex,
              " propId=0x", static_cast<uint64_t>(pInstance->m_stablePropId), std::dec,
              " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(pBlasRp), std::dec,
              " mapSz=", mapSzRp,
              // NV-DXVK [ReapJoin] per-line fields. drew= is the whole answer:
              // 1 = a draw for THIS mesh arrived this frame and we reaped the
              // instance anyway (matching bug), 0 = no draw arrived
              // (submission gap). See the [ReapJoin] block at the top of
              // garbageCollection for why frameLastTouched is the right field
              // and why the call ordering makes it trustworthy.
              " drew=", (pBlasRp != nullptr && pBlasRp->frameLastTouched == currentFrame) ? 1 : 0,
              " blasTouch=", (pBlasRp != nullptr ? pBlasRp->frameLastTouched : 0u),
              // blasUpd advances only on a REAL geometry change (kUpdateBVH /
              // KBuildBVH), so blasTouch==f with blasUpd lagging is the normal
              // static-mesh case, not a defect. Printed so the two are never
              // confused — blasUpd is NOT a draw-arrival signal.
              " blasUpd=", (pBlasRp != nullptr ? pBlasRp->frameLastUpdated : 0u),
              // How many instances still hold this geometry. If drew=1 and a
              // sibling exists, the arriving draw went to a DIFFERENT instance
              // than the one being reaped — that is the matching failure made
              // visible, and linked= names how many candidates were in play.
              " linked=", (pBlasRp != nullptr ? pBlasRp->getLinkedInstances().size() : 0u),
              " lastUpd=", pInstance->m_frameLastUpdated,
              " age=", (currentFrame - pInstance->m_frameLastUpdated),
              " keepN=", instanceKeepN,
              " ignAC=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
              " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
              " marked=", (clauseMarked ? 1 : 0),
              " pos=(", posRp.x, ",", posRp.y, ",", posRp.z, ")"));
          }
        }

        // NV-DXVK [InstReapWave]: UNGATED mass-removal detail. The view-2 flip
        // is a one-frame mass lifetime-GC (f=662: 566 removed, f=663: 322 —
        // ~60% of the scene re-keyed) but [InstReap] above is gated on the
        // engine-hook capture counter, which only crosses 16 AFTER the flip
        // (the hook feed engages AT the flip — fanoutFpAddr first capture ==
        // flip frame). This logs removals #65..96 of any pass that removes
        // >64 instances, no gate: zero noise in steady state (passes remove
        // 0-47), 32 named victims at the wave. vs/propId/pos identify the
        // population (dome? structure? camera-relative recon draws?).
        if (probeRemovedThisPass > 64u && probeRemovedThisPass <= 96u) {
          const BlasEntry* pBlasWv = pInstance->getBlas();
          const XXH64_hash_t vsWv = (pBlasWv != nullptr)
            ? pBlasWv->input.getTransformData().vertexShaderHash : 0ull;
          const uint64_t matWv = (pBlasWv != nullptr)
            ? static_cast<uint64_t>(pBlasWv->input.getMaterialData().getHash()) : 0ull;
          const uint32_t vertsWv = (pBlasWv != nullptr)
            ? pBlasWv->modifiedGeometryData.vertexCount : 0u;
          const Vector3 posWv = pInstance->getWorldPosition();
          Logger::warn(str::format(
            "[InstReapWave] f=", currentFrame,
            " n=", probeRemovedThisPass,
            " vs=0x", std::hex, static_cast<uint64_t>(vsWv),
            " mat=0x", matWv, std::dec,
            " v=", vertsWv, std::hex,
            " propId=0x", static_cast<uint64_t>(pInstance->m_stablePropId), std::dec,
            " lastUpd=", pInstance->m_frameLastUpdated,
            " keepN=", instanceKeepN,
            " ignAC=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
            " pos=(", posWv.x, ",", posWv.y, ",", posWv.z, ")"));
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

    // NV-DXVK [ReapJoin] — one line per frame, the decisive readout.
    //
    //   drew=N     reaps where the mesh's own draw ARRIVED this frame and we
    //              destroyed the instance regardless => MATCHING BUG (ours).
    //   noDraw=N   reaps with no draw for that geometry => submission gap
    //              (upstream of Remix).
    //
    // Emitted on every GC pass including empty ones: a frame with removed=0
    // is data (it says the churn stopped), and a probe that only prints when
    // it has something to say cannot show you an absence.
    //
    // Cost is two increments per reap plus one line per frame — safe to leave
    // on. It is NOT gated on the engine-hook capture counter, unlike the
    // [InstReap] detail lines, so the totals cover every reap in the pass.
    if (probeRemovedThisPass > 0 || probeKeptThisPass > 0) {
      Logger::info(str::format(
        "[ReapJoin] f=", probeFrame,
        " removed=", probeRemovedThisPass,
        " drew=", probeReapDrew,
        " noDraw=", probeReapNoDraw,
        " pctDrew=", (probeRemovedThisPass > 0
          ? (100u * probeReapDrew) / probeRemovedThisPass : 0u),
        " kept=", probeKeptThisPass,
        " live=", static_cast<uint32_t>(m_instances.size())));
    }

    // NV-DXVK [InstDriftProbe]: record post-GC size UNCONDITIONALLY so the
    // first gameplay GC has valid history from the last pre-gameplay GC.
    // Tracking is cheap (two uint32 writes); only the log is gated.
    sLastGcExitSize  = static_cast<uint32_t>(m_instances.size());
    sLastGcExitFrame = probeFrame;
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
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache) {

    // If the RtInstance represents multiple instances, use the full transform of the first copy for the spatial map.
    // this prevents a bad de-duplication when the same replacement asset is used in multiple GeomPointInstancer prims.
    Matrix4 firstInstanceObjectToWorld = drawCall.getTransformData().calcFirstInstanceObjectToWorld();

    // If we already know which instance to use, just use that.
    RtInstance* currentInstance = existingInstance;

    // Search for an existing instance matching our input
    if (currentInstance == nullptr) {
      currentInstance = findSimilarInstance(blas, materialData, firstInstanceObjectToWorld, drawCall.cameraType, rayPortalManager, drawCall.getTransformData().stablePropId, drawCallCache);
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
      // NV-DXVK [MtnDedup]: for the path-10 mountain shaders, log whether
      // findSimilarInstance reused an existing RtInstance or a new one is
      // about to be spawned, with the lookup propId + transform.
      //
      // Logged EVERY mountain draw, EVERY frame (no cap, no frame sampling):
      // the game runs at a few FPS so any frame-modulo gate would almost
      // never land. Correlate per frame with the same-frame [MtnPIAdd] batch
      // census — together they show whether the TLAS over-instancing comes
      // from too many draws/RtInstances or from per-batch fanout.
      const XXH64_hash_t vsHashMd = drawCall.getTransformData().vertexShaderHash;
      // Also driven by rtx.findSimilarProbeVsHashes. This is the only probe on
      // this path that fires for EVERY draw rather than sampling, which is what
      // is needed to catch a draw that has never once appeared as a lookup:
      // [FindSim] has only ever logged main-world queries for the tree VS, yet
      // its SpatialMap keeps ending up holding a sky-coordinate entry. Whatever
      // creates that entry has to pass through here.
      const bool isOptDedupVs =
        lookupHash(RtxOptions::findSimilarProbeVsHashes(), vsHashMd);
      if (vsHashMd == 0x29146e1dd50b0314ull
          || vsHashMd == 0x28f7ffa90d189017ull
          || isOptDedupVs) {
        // UNCAPPED, for both the hardcoded VSes and the option path. The
        // option path used to carry a global 4000-line budget; aimed at the
        // high-churn VSes (0x2859d250 / 0x28d6baea / 0x292b6ba0) that draw
        // hundreds of times per frame, any budget burns out early and blinds
        // the rest of the run. Same failure as [InstReap]'s 24/frame cap.
        {
          // Centroid is what the SpatialMap actually keys/cells on, and it is
          // NOT o2w.T — log both so a divergence is visible directly.
          const Vector3 mdCentroid =
            blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
          const float mdCol0 = length(Vector3(firstInstanceObjectToWorld[0][0],
                                              firstInstanceObjectToWorld[0][1],
                                              firstInstanceObjectToWorld[0][2]));
          // The DRAW's geometry, not blas.input's: the draw's hash is what
          // selected (or failed to select) the BlasEntry, so it is the input
          // to the routing decision we are diffing. blas.input is the result.
          const auto& mdGeo = drawCall.getGeometryData();
          Logger::info(str::format(
            "[MtnDedup] vsXxh=0x", std::hex, static_cast<uint64_t>(vsHashMd), std::dec,
            " frame=", m_device->getCurrentFrameId(),
            " cam=", static_cast<uint32_t>(drawCall.cameraType),
            " foundSimilar=", (foundSimilar ? 1 : 0),
            " hadExisting=", (existingInstance != nullptr ? 1 : 0),
            " propId=0x", std::hex,
              static_cast<uint64_t>(drawCall.getTransformData().stablePropId), std::dec,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
            " o2wCol0Len=", mdCol0,
            " centroid=(", mdCentroid.x, ",", mdCentroid.y, ",", mdCentroid.z, ")",
            " lookup.o2w.T=(", firstInstanceObjectToWorld[3][0], ",",
              firstInstanceObjectToWorld[3][1], ",",
              firstInstanceObjectToWorld[3][2], ")",
            // NV-DXVK: per-component geometry hashes. The 2026-07-29 [MapDump]
            // proved every tree miss is the FIRST SIGHTING of a new blasPtr
            // with an empty map — a forced miss, not a keying failure — while
            // the tree VS was reaped 0 times in 17078 reaps. So the defect is
            // BlasEntry churn: the same billboard hashing differently on some
            // frames. BlasEntry identity is the geometry hash, so log each
            // COMPONENT rather than the composite rules: FullGeometryHash
            // changing only says "something moved", whereas diffing two
            // consecutive frames' components names the single unstable field.
            //   VertexPosition/Texcoord -> content genuinely rewritten
            //     (camera-facing billboard verts rebaked per frame)
            //   Indices/GeometryDescriptor -> topology or draw-range changed
            //   VertexLayout/VertexShader -> binding state changed, content
            //     identical — that would be the dynamic-buffer arena rotating
            //     (d3d11_rtx.cpp:30262) and is fixable without touching content
            " hPos=0x", std::hex,
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexPosition]),
            " hUv=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexTexcoord]),
            " hIdx=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::Indices]),
            " hDesc=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::GeometryDescriptor]),
            " hLayout=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexLayout]),
            " hVs=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexShader]),
            " hFull=0x",
              static_cast<uint64_t>(mdGeo.getHashForRule<rules::FullGeometryHash>()),
            std::dec,
            " vtxCount=", mdGeo.vertexCount,
            " idxCount=", mdGeo.indexCount,
            // NV-DXVK: texture identity. findSimilarProbeVsHashes selects a
            // SHADER, and 0x29382bf838fda043 is not tree-specific -- it is a
            // shared prop VS (the 2026-07-29 16:59 manifest has it drawing a
            // 128x128 industrial panel atlas, 8 instances, while [SkyDiag] shows
            // 8 distinct albedos on it). Without a texture id every tree line is
            // mixed in with unrelated props and the per-component hash diff
            // above cannot be restricted to the billboard. albHash matches the
            // <hash>_albedo.dds naming in onscreen_albedo_dump/manifest.txt, so a
            // dumped texture maps straight to the draws that used it.
            " albHash=0x", std::hex,
              static_cast<uint64_t>(drawCall.getMaterialData().getColorTexture().isValid()
                ? drawCall.getMaterialData().getColorTexture().getImageHash()
                : 0ull),
            " mat=0x", static_cast<uint64_t>(drawCall.getMaterialData().getHash()), std::dec,
            // The 2026-07-29 17:21 capture showed this material hash ALTERNATING
            // every frame on a stationary tree whose geometry hashes and albedo
            // were byte-stable (10 distinct mat values at one position, e.g.
            // fc2c10c1ed54 / b64d0af14ff5 flipping f2705..f2709). Dump the hash
            // inputs so the responsible field is named rather than inferred.
            " | ", drawCall.getMaterialData().debugHashInputs()));
        }
      }
    }
    // NV-DXVK [SubViewKey]: per-frame census of sub-view-fan instance
    // keying — the second half of the two-views probe pair ([InstReap]
    // shows WHAT dies at the flip; this shows WHY the steady-state stream
    // didn't keep it alive). A draw counts if it carries a stablePropId or
    // a reprojected x1000-scale o2w (col0 length > 100; normal draws are
    // ~1). Readout, intro frame vs steady frame:
    //   steady: hit high, create=0, but the intro-era propIds never appear
    //     in any later create/hit -> the engine stopped issuing those
    //     draws; fix at keep/refresh policy or the draw source.
    //   steady: create>0 EVERY frame with new propIds -> keying churn; fix
    //     at the propId producer (cf. the reverted PropIdKeepLong note in
    //     garbageCollection — MakeBoneStablePropId instability).
    // Aggregate 1 line/frame + create detail capped 12/frame, gameplay-gated.
    {
      const bool svGameplay =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      const uint64_t svPropId =
        static_cast<uint64_t>(drawCall.getTransformData().stablePropId);
      const float svSc0Sq =
          firstInstanceObjectToWorld[0][0] * firstInstanceObjectToWorld[0][0]
        + firstInstanceObjectToWorld[0][1] * firstInstanceObjectToWorld[0][1]
        + firstInstanceObjectToWorld[0][2] * firstInstanceObjectToWorld[0][2];
      const bool svScaled = svSc0Sq > 10000.0f;  // col0 len > 100 => reprojected
      if (svGameplay && (svPropId != 0ull || svScaled)) {
        struct SvKeyAgg {
          uint32_t frame = 0xFFFFFFFFu;
          uint32_t cand = 0, hit = 0, create = 0, scaled = 0, detail = 0;
        };
        static thread_local SvKeyAgg sSvKey;
        const uint32_t svFrame = m_device->getCurrentFrameId();
        if (sSvKey.frame != svFrame) {
          if (sSvKey.frame != 0xFFFFFFFFu && sSvKey.cand > 0u) {
            Logger::info(str::format(
              "[SubViewKey] f=", sSvKey.frame,
              " cand=", sSvKey.cand,
              " hit=", sSvKey.hit,
              " create=", sSvKey.create,
              " scaled=", sSvKey.scaled));
          }
          sSvKey = SvKeyAgg{};
          sSvKey.frame = svFrame;
        }
        sSvKey.cand += 1;
        if (svScaled) sSvKey.scaled += 1;
        if (foundSimilar) {
          sSvKey.hit += 1;
        } else {
          sSvKey.create += 1;
          // Cap raised 12 -> 48. The 12/frame budget is shared across every
          // sub-view VS with no prioritisation, and it measurably truncated:
          // in the 2026-07-29 capture 30 of 689 frames saturated at exactly
          // 12, including f=9510 and f=9564 — the two frames carrying the
          // propId-collision reaps this probe exists to explain. Observed
          // distribution is <=5 creates on most frames, so 48 costs little
          // while making the busy frames (the interesting ones) complete.
          if (sSvKey.detail < 48u) {
            sSvKey.detail += 1;
            // NV-DXVK: blasPtr + the BlasEntry's OWN recorded VS, alongside
            // the DRAW's VS. The SpatialMap is per-BlasEntry, so [FindSim]'s
            // ownerVs — read as dbgOwner->getBlas()->input...vertexShaderHash
            // — can only ever echo the VS of the blas being queried. It is
            // structurally incapable of detecting two shaders sharing one
            // BlasEntry, which is the case it was written to test. Here the
            // draw's VS and the blas's VS are independent reads, so
            // blasVs != vs on a single line IS the shared-BlasEntry proof.
            // blasPtr cross-references directly against [MtnDedup]'s blasPtr.
            Logger::info(str::format(
              "[SubViewKey.create] f=", svFrame,
              " vs=0x", std::hex,
                static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash),
              " propId=0x", svPropId,
              " blasPtr=0x", reinterpret_cast<uintptr_t>(&blas),
              " blasVs=0x",
                static_cast<uint64_t>(blas.input.getTransformData().vertexShaderHash),
                std::dec,
              " mapSz=", blas.getSpatialMap().size(),
              " scaledCol0=", (svScaled ? 1 : 0),
              " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
                         firstInstanceObjectToWorld[3][1], ",",
                         firstInstanceObjectToWorld[3][2], ")"));
          }
        }
      }
    }
    // NV-DXVK [PassCensus]: per-frame inventory of EVERY instance, bucketed by
    // (VS hash, cameraType). Answers the architecture question the retention
    // patches dodged: "where does the geometry that's missing in view 2 come
    // from — a real camera or the shadow fanout?" For each bucket: count, how
    // many were reprojected (col0 scale > 100 == sky_camera sub-view content),
    // and the world-translation AABB. Flush one line per bucket at frame
    // transition, gameplay-gated, <=40 lines/frame.
    //
    // Diff a VIEW-1 frame against a VIEW-2 frame:
    //   a VS present in view 1, GONE in view 2  == the missing geometry. Its
    //   cameraType is the verdict:
    //     - same cameraType as the reprojected sky_camera sub-view (scaled>0)
    //       -> the real sky_camera DOES carry it; conversion drops it in
    //          view 2 -> proper fix lives in the sub-view conversion path.
    //     - only ever the shadow-fanout cameraType (the 0x2947c634 family,
    //       scaled==0, cam==6) -> NO real-camera source exists; the horizon
    //       is sourced from a shadow render -> the fanout dependency is
    //       structural and must be re-sourced, not retained.
    // GATE CHANGE (2026-06-13): was gated on g_engineHookCaptureCount>16, which
    // only opens AFTER the view-1->view-2 flip, so view 1 was never captured and
    // a clean cross-view diff was impossible. Now accumulate EVERY frame and
    // flush only "real" scene frames (>=100 instances), which captures the intro
    // (view 1, lots of geometry) AND gameplay (view 2) while skipping menu/load
    // frames (few instances). One run then yields the definitive view-1 vs
    // view-2 per-VS diff: a VS instanced in view 1 but absent in view 2 is the
    // dropped sky-camera sub-view content (the displaced/missing geometry).
    {
      {
        struct PcEntry {
          uint32_t count = 0, scaled = 0; int camType = -1;
          float mn[3] = { 1e30f, 1e30f, 1e30f };
          float mx[3] = { -1e30f, -1e30f, -1e30f };
        };
        static thread_local uint32_t sPcFrame = 0xFFFFFFFFu;
        static thread_local uint32_t sPcTotal = 0;
        static thread_local std::unordered_map<uint64_t, PcEntry> sPc;
        const uint32_t pcF = m_device->getCurrentFrameId();
        if (sPcFrame != pcF) {
          if (sPcFrame != 0xFFFFFFFFu && sPcTotal >= 100u) {
            uint32_t printed = 0;
            for (const auto& kv : sPc) {
              if (printed >= 40u) break;
              const PcEntry& e = kv.second;
              if (e.count < 2u) continue;  // skip per-frame singleton noise
              ++printed;
              Logger::info(str::format(
                "[PassCensus] f=", sPcFrame,
                " total=", sPcTotal,
                " vs=0x", std::hex, kv.first, std::dec,
                " cam=", e.camType,
                " count=", e.count,
                " scaled=", e.scaled,
                " wMin=(", e.mn[0], ",", e.mn[1], ",", e.mn[2], ")",
                " wMax=(", e.mx[0], ",", e.mx[1], ",", e.mx[2], ")"));
            }
          }
          sPc.clear();
          sPcTotal = 0;
          sPcFrame = pcF;
        }
        const uint64_t pcVs =
          static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
        PcEntry& e = sPc[pcVs];
        e.count += 1;
        sPcTotal += 1;
        e.camType = static_cast<int>(drawCall.cameraType);
        const float pcSc0Sq =
            firstInstanceObjectToWorld[0][0] * firstInstanceObjectToWorld[0][0]
          + firstInstanceObjectToWorld[0][1] * firstInstanceObjectToWorld[0][1]
          + firstInstanceObjectToWorld[0][2] * firstInstanceObjectToWorld[0][2];
        if (pcSc0Sq > 10000.0f) e.scaled += 1;
        for (int a = 0; a < 3; ++a) {
          const float t = firstInstanceObjectToWorld[3][a];
          e.mn[a] = std::min(e.mn[a], t);
          e.mx[a] = std::max(e.mx[a], t);
        }
      }
    }
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

      // NV-DXVK [PropCensus]: for VS_2904d2 sub-view mountains, track which
      // distinct world-position keys appear in this frame's submissions and
      // log the set every 30 frames. Goal: if one mountain is permanently
      // missing while the rest render, that prop's position will be
      // consistently absent from the census. Pinpoints which charIdx /
      // world-pos pair is dropping out.
      //
      // [gameplay gate] Only run during gameplay (after engine-hook main-
      // cam has fired >16 times). Without this gate the accumulating
      // thread-local maps fill with menu-phase noise before gameplay even
      // starts, polluting the "first seen / last seen" stats.
      const bool gameplayActive =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16;
      if (gameplayActive && vsHash == 0x2904d2163ef31a17ull) {
        const float wpx = float(firstInstanceObjectToWorld[3][0]);
        const float wpy = float(firstInstanceObjectToWorld[3][1]);
        const float wpz = float(firstInstanceObjectToWorld[3][2]);
        const Vector3 wp{wpx, wpy, wpz};
        // Quantize to 5000u so per-frame snap drift doesn't make the
        // same prop look like multiple distinct keys.
        const int kx = int(std::round(wp.x / 5000.f));
        const int ky = int(std::round(wp.y / 5000.f));
        const int kz = int(std::round(wp.z / 5000.f));
        static thread_local std::unordered_map<uint64_t, uint32_t> sPropFirstSeen;
        static thread_local std::unordered_map<uint64_t, uint32_t> sPropLastSeen;
        const uint64_t key = (uint64_t(uint32_t(kx)) & 0x1FFFFFu)
                           | ((uint64_t(uint32_t(ky)) & 0x1FFFFFu) << 21)
                           | ((uint64_t(uint32_t(kz)) & 0x1FFFFFu) << 42);
        const uint32_t curF = m_device->getCurrentFrameId();
        auto first = sPropFirstSeen.find(key);
        if (first == sPropFirstSeen.end()) {
          sPropFirstSeen.emplace(key, curF);
          Logger::info(str::format(
            "[PropCensus] newKey f=", curF,
            " worldPos=(", wp.x, ",", wp.y, ",", wp.z, ")",
            " key=(", kx, ",", ky, ",", kz, ")"));
        }
        sPropLastSeen[key] = curF;

        // Every 60 frames, dump props whose lastSeen is 2+ frames behind
        // current frame — those are the "missing" ones.
        static thread_local uint32_t sLastReportF = UINT32_MAX;
        if (sLastReportF != curF && (curF % 60) == 0) {
          sLastReportF = curF;
          for (const auto& [k, lastF] : sPropLastSeen) {
            if (lastF + 2 <= curF && sPropFirstSeen[k] + 30 < curF) {
              // Decode position from key
              const int ux = int(k & 0x1FFFFFu);
              const int uy = int((k >> 21) & 0x1FFFFFu);
              const int uz = int((k >> 42) & 0x1FFFFFu);
              const int sx = (ux & 0x100000) ? (ux | ~int(0x1FFFFF)) : ux;
              const int sy = (uy & 0x100000) ? (uy | ~int(0x1FFFFF)) : uy;
              const int sz = (uz & 0x100000) ? (uz | ~int(0x1FFFFF)) : uz;
              Logger::warn(str::format(
                "[PropCensus] DROPPED f=", curF,
                " firstSeen=", sPropFirstSeen[k],
                " lastSeen=", lastF,
                " framesSinceSeen=", (curF - lastF),
                " posKey=(", sx*5000, ",", sy*5000, ",", sz*5000, ")"));
            }
          }
        }
      }

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

          // A genuine premultiplied-alpha draw composites the alpha channel
          // with the OVER operator too (not just color). There are two
          // algebraically-identical ways to spell that alpha OVER:
          //   a = src.a*1          + dst.a*(1-src.a)   [ONE, ONE_MINUS_SRC_ALPHA]
          //   a = src.a*(1-dst.a)  + dst.a*1           [ONE_MINUS_DST_ALPHA, ONE]
          // Either one means the draw is maintaining a correct framebuffer
          // alpha channel — i.e. it is doing real translucent compositing,
          // not a fire-and-forget additive glow. TF2's sky cloud billboards
          // use the second form; the original check only knew the first, so
          // the clouds fell through to the "inverted reverse emissive"
          // branch, were tagged emissive, and rendered as glowing rectangles.
          const bool alphaIsOverComposite =
            alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
            ((srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
              dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
             (srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA &&
              dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE));

          const bool looksPremultiplied =
            (alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
             srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
             dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
            alphaWritesDisabled;

          if (alphaIsOverComposite) {
            // Genuine premultiplied-alpha translucency — OVER on both color
            // and alpha. This is compositing, never emission, so classify it
            // as kAlpha even when emissive blend translation is enabled.
            // (Real additive glows use ONE,ONE or SRC_ALPHA,ONE and are
            // caught by the emissive branches above; they do not maintain an
            // OVER-composited alpha channel, so they never reach here.)
            blendType = BlendType::kAlpha;
            invertedBlend = false;
          } else if (looksPremultiplied) {
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

  RtInstance* InstanceManager::findSimilarInstance(BlasEntry& blas, const MaterialData& material, const Matrix4& firstInstanceObjectToWorld, CameraType::Enum cameraType, const RayPortalManager& rayPortalManager, uint64_t stablePropId, DrawCallCache* drawCallCache) {

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
    // Hardcoded hash kept so the original 2904d2 capture still reproduces;
    // rtx.findSimilarProbeVsHashes aims the same probe at anything else
    // (e.g. the tree-billboard VS) without a rebuild.
    const bool isProbeVS = (vsHashProbe == 0x2904d2163ef31a17ull)
                        || lookupHash(RtxOptions::findSimilarProbeVsHashes(), vsHashProbe);
    // NV-DXVK [FindSimMtn]: the s2s "two views" black-sky root. The reprojected
    // 3D-skybox terrain VS 0x29146e1d places 14 panels every frame in both
    // views (MtnPlace=14, all reproject gates PASS), but only 13 survive to the
    // sub-view in view1 vs 9 in view2 — 4-5 panels collapse here in dedup
    // (hidden=0/notInFr=0, no reap). This probe logs, per panel, whether it
    // collapses via EXACT (getDataAtTransform on propId/transform) or NEAREST
    // (getNearestData within uniqueObjectDistanceSqr) — telling us if view2's
    // extra collapses are propId collisions (engine reused a stablePropId for
    // distinct panels) or distance merges (panels fall within the dedup radius
    // after reproject). Own counters so it captures full frames of all 14.
    const bool isMtnProbe = (vsHashProbe == 0x29146e1dd50b0314ull);

    // Search the BLAS for an instance matching ours
    {
      // Search for an exact match. stablePropId (passed in from the
      // caller's drawCall.transformData) overrides the matrix-bytes
      // cache key when non-zero — anchors dedup to per-prop identity.
      result = const_cast<RtInstance*>(blas.getSpatialMap().getDataAtTransform(firstInstanceObjectToWorld, stablePropId));
      const bool exactHit = (result != nullptr);
      if (result != nullptr) {
        if (isProbeVS) {
          thread_local uint32_t sExactProbe = 0;
          if (sExactProbe < 8 || (sExactProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[FindSim] #", sExactProbe, " vs=0x", std::hex, vsHashProbe, std::dec,
              " stage=exact hit=1 propId=0x", std::hex, stablePropId, std::dec,
              " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
              " spatialMapSize=", blas.getSpatialMap().size(),
              " linkedInst=", blas.getLinkedInstances().size()));
          }
          sExactProbe += 1;
        }
        if (isMtnProbe) {
          thread_local uint32_t sMtnExact = 0;
          if (sMtnExact < 80u || (sMtnExact & 0xFFu) == 0u) {
            Logger::info(str::format(
              "[FindSimMtn] #", sMtnExact, " f=", currentFrameIdx,
              " outcome=COLLAPSE stage=exact propId=0x", std::hex, stablePropId, std::dec,
              " spatialMapSize=", blas.getSpatialMap().size(),
              " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
          }
          sMtnExact += 1;
        }
        return result;
      }
      // NV-DXVK [FindSim2904 exact-miss]: log propId we looked up. Pair
      // against [PropIdTrace] at the same frame to see if lookup propId
      // matches insert propId for the same physical prop. If they match
      // and the map is non-empty, the insert went into a bumped slot
      // (collision); if they differ, the rounding boundary is flipping.
      if (isProbeVS) {
        thread_local uint32_t sExactMissProbe = 0;
        if (sExactMissProbe < 32 || (sExactMissProbe & 0x3FF) == 0) {
          // NV-DXVK: include BlasEntry pointer + linked-instance count so we
          // can detect whether the lookup is going to a different BlasEntry
          // than the one holding the instance. If blasPtr changes between
          // calls for the same physical mountain, drawCallCache is routing
          // to a different BLAS — root cause of "instance alive, lookup
          // misses".
          Logger::info(str::format(
            "[FindSim] #", sExactMissProbe,
            " vs=0x", std::hex, vsHashProbe, std::dec,
            " stage=exact hit=0 propId=0x", std::hex, stablePropId, std::dec,
            " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " linkedInst=", blas.getLinkedInstances().size(),
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
        }
        sExactMissProbe += 1;
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
          if (isProbeVS || isMtnProbe) {
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
        // Throttle raised from 8 to 256: the mapSz>0 exact-misses — the only
        // interesting case — all landed past #8 and were never logged.
        thread_local uint32_t sNearestProbe = 0;
        // UNCAPPED. The old "first 256 then every 1024th" throttle aliased the
        // exact events being hunted — a miss is rare, so sampling makes it
        // likely to be the one that is skipped.
        {
          // Distinguish "the cached entry is genuinely elsewhere" from "the
          // cell grid has lost track of an entry that is right here".
          Vector3 dbgNearestPos { 0.f, 0.f, 0.f };
          const RtInstance* dbgOwner = nullptr;
          const float dbgCacheDistSqr =
            blas.getSpatialMap().debugClosestCachedDistSqr(worldPosition, dbgNearestPos, &dbgOwner);
          const size_t dbgCellEntries = blas.getSpatialMap().debugCellEntryCount();
          // Identify who owns the far-away entry. If its VS differs from ours,
          // two different draws are sharing one BlasEntry and evicting each
          // other; if it matches, the same draw is being placed in two spaces.
          uint64_t dbgOwnerVs = 0ull;
          uint32_t dbgOwnerFrame = 0u;
          uint32_t dbgOwnerCam = 0u;
          if (dbgOwner != nullptr) {
            dbgOwnerFrame = dbgOwner->m_frameLastUpdated;
            const BlasEntry* ownerBlas = dbgOwner->getBlas();
            if (ownerBlas != nullptr) {
              dbgOwnerVs = uint64_t(ownerBlas->input.getTransformData().vertexShaderHash);
              dbgOwnerCam = uint32_t(ownerBlas->input.cameraType);
            }
          }
          // NV-DXVK [MapDump2]: on an actual MISS, dump EVERY live entry rather
          // than just the nearest one. [MapWrite] instruments BOTH SpatialMap
          // write sites (insert via teleport, move via onTransformChanged) and
          // logged ZERO sky-space centroids for this VS across 2231 writes --
          // yet debugClosestCachedDistSqr reports a sky-space nearest entry with
          // ownerVs equal to this same VS. Those cannot both be true, and a
          // single-nearest readback cannot show which observation is incomplete.
          // ownerBlas is printed alongside the queried blas: if they differ, the
          // owning instance has been relinked to another BlasEntry while its
          // cache entry stayed behind here, which no write-site probe would see.
          // Misses are rare, so this is uncapped and prints the whole map.
          if (result == nullptr) {
            uint32_t dumpIdx = 0;
            blas.getSpatialMap().debugForEachEntry(
              [&](auto key, const Vector3& c, const RtInstance* owner) {
                uint64_t  oVs = 0ull;
                uint32_t  oFrame = 0u;
                uintptr_t oBlas = 0u;
                if (owner != nullptr) {
                  oFrame = owner->m_frameLastUpdated;
                  const BlasEntry* ob = owner->getBlas();
                  if (ob != nullptr) {
                    oVs   = uint64_t(ob->input.getTransformData().vertexShaderHash);
                    oBlas = reinterpret_cast<uintptr_t>(ob);
                  }
                }
                Logger::info(str::format(
                  "[MapDump2] f=", currentFrameIdx,
                  " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
                  " #", dumpIdx++,
                  " key=0x", std::hex, static_cast<uint64_t>(key), std::dec,
                  " centroid=(", c.x, ",", c.y, ",", c.z, ")",
                  " ownerVs=0x", std::hex, oVs, std::dec,
                  " ownerBlas=0x", std::hex, oBlas, std::dec,
                  " ownerFrame=", oFrame,
                  " query=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
              });
          }

          // worldPosition is boundingBox.getTransformedCentroid(o2w), NOT
          // o2w.T. With the sub-view reproject baking scale=1000 into o2w,
          // the object-space bbox centre is amplified 1000x on its way to the
          // centroid — so a ~26-unit local offset lands ~26000 units away.
          // Log the decomposition so the arithmetic is visible rather than
          // inferred: local bbox, the o2w column lengths (the actual applied
          // scale) and o2w.T next to the resulting centroid.
          const AxisAlignedBoundingBox& dbgBox = blas.input.getGeometryData().boundingBox;
          const Vector3 dbgBoxCentre = (dbgBox.minPos + dbgBox.maxPos) * 0.5f;
          const Matrix4& dbgO2w = firstInstanceObjectToWorld;
          const float dbgScaleX = length(Vector3(dbgO2w[0][0], dbgO2w[0][1], dbgO2w[0][2]));
          const float dbgScaleY = length(Vector3(dbgO2w[1][0], dbgO2w[1][1], dbgO2w[1][2]));
          const float dbgScaleZ = length(Vector3(dbgO2w[2][0], dbgO2w[2][1], dbgO2w[2][2]));
          Logger::info(str::format(
            "[FindSim] #", sNearestProbe, " vs=0x", std::hex, vsHashProbe, std::dec,
            " boxLocalCentre=(", dbgBoxCentre.x, ",", dbgBoxCentre.y, ",", dbgBoxCentre.z, ")",
            " boxMin=(", dbgBox.minPos.x, ",", dbgBox.minPos.y, ",", dbgBox.minPos.z, ")",
            " boxMax=(", dbgBox.maxPos.x, ",", dbgBox.maxPos.y, ",", dbgBox.maxPos.z, ")",
            " o2wScale=(", dbgScaleX, ",", dbgScaleY, ",", dbgScaleZ, ")",
            " o2wT=(", float(dbgO2w[3][0]), ",", float(dbgO2w[3][1]), ",", float(dbgO2w[3][2]), ")",
            " cacheNearestDistSqr=", dbgCacheDistSqr,
            " cacheNearestPos=(", dbgNearestPos.x, ",", dbgNearestPos.y, ",", dbgNearestPos.z, ")",
            " ownerVs=0x", std::hex, dbgOwnerVs, std::dec,
            " ownerCam=", dbgOwnerCam,
            " ownerFrame=", dbgOwnerFrame,
            " curFrame=", currentFrameIdx,
            " curCam=", static_cast<uint32_t>(cameraType),
            " cellEntries=", dbgCellEntries,
            " stage=nearest exactHit=0 hit=", (result != nullptr ? 1 : 0),
            " propId=0x", std::hex, stablePropId, std::dec,
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
          // NV-DXVK [MapDump]: on a MISS against a NON-EMPTY map, dump every
          // live entry beside the query. The keepN=4 experiment refuted the
          // lifetime explanation (maps grew to 13 entries, misses continued),
          // so the entry the query wants may well be present — the question is
          // why the lookup cannot reach it. Two reachable answers:
          //   entryKey == queryKey but the exact stage still missed  -> the
          //     exact lookup is keyed on something else than we think
          //   entryCell != queryCell while |entryCentroid - queryCentroid| is
          //     small -> the cell grid and the distance test disagree, i.e. a
          //     border/rounding fault, not a coordinate-space fault
          //   entryCentroid genuinely far -> confirms the squatter reading
          // queryKey is formed exactly as getDataAtTransform does: propId when
          // non-zero, else XXH64 over the matrix bytes.
          if (result == nullptr && blas.getSpatialMap().size() > 0u) {
            // UNCAPPED. A miss against a non-empty map is the event this whole
            // probe exists to catch; clipping it to 4/frame risks dropping the
            // one that matters.
            {
              const uint64_t queryKey = (stablePropId != 0ull)
                ? stablePropId
                : static_cast<uint64_t>(XXH64(&firstInstanceObjectToWorld,
                                              sizeof(firstInstanceObjectToWorld), 0));
              const auto qCell = blas.getSpatialMap().debugCellPosOf(worldPosition);
              Logger::info(str::format(
                "[MapDump] f=", currentFrameIdx,
                " vs=0x", std::hex, vsHashProbe, std::dec,
                " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
                " QUERY key=0x", std::hex, queryKey, std::dec,
                " propId=0x", std::hex, stablePropId, std::dec,
                " centroid=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")",
                " cell=(", qCell.x, ",", qCell.y, ",", qCell.z, ")",
                " cellSize=", blas.getSpatialMap().debugCellSize(),
                " maxDistSqr=", uniqueObjectDistanceSqr,
                " entries=", blas.getSpatialMap().size()));
              uint32_t dumped = 0;
              blas.getSpatialMap().debugForEachEntry(
                [&](XXH64_hash_t entryKey, const Vector3& entryCentroid,
                    const RtInstance* entryData) {
                  dumped += 1;   // uncapped: dump every entry in the map
                  const auto eCell = blas.getSpatialMap().debugCellPosOf(entryCentroid);
                  const Vector3 delta = entryCentroid - worldPosition;
                  const float dSq = lengthSqr(delta);
                  Logger::info(str::format(
                    "[MapDump]   entry#", dumped,
                    " key=0x", std::hex, static_cast<uint64_t>(entryKey), std::dec,
                    " keyMatchesQuery=", (static_cast<uint64_t>(entryKey) == queryKey ? 1 : 0),
                    " centroid=(", entryCentroid.x, ",", entryCentroid.y, ",", entryCentroid.z, ")",
                    " cell=(", eCell.x, ",", eCell.y, ",", eCell.z, ")",
                    " sameCell=", ((eCell.x == qCell.x && eCell.y == qCell.y && eCell.z == qCell.z) ? 1 : 0),
                    " distSq=", dSq,
                    " withinMaxDist=", (dSq < uniqueObjectDistanceSqr ? 1 : 0),
                    // The three filter clauses getNearestData applies, so a
                    // reachable-but-rejected entry is distinguishable from an
                    // unreachable one.
                    " ownerPropId=0x", std::hex,
                      (entryData != nullptr ? entryData->m_stablePropId : 0ull), std::dec,
                    " ownerLastUpd=", (entryData != nullptr ? entryData->m_frameLastUpdated : 0u),
                    " okFrame=", ((entryData != nullptr
                                   && entryData->m_frameLastUpdated != currentFrameIdx) ? 1 : 0),
                    " okMat=", ((entryData != nullptr
                                 && entryData->m_materialHash == material.getHash()) ? 1 : 0),
                    " okSub=", ((entryData != nullptr
                                 && !entryData->m_primInstanceOwner.isSubPrim()) ? 1 : 0)));
                });
            }
          }
        }
        sNearestProbe += 1;
      }
      if (isMtnProbe) {
        thread_local uint32_t sMtnNear = 0;
        if (sMtnNear < 80u || (sMtnNear & 0xFFu) == 0u) {
          Logger::info(str::format(
            "[FindSimMtn] #", sMtnNear, " f=", currentFrameIdx,
            " outcome=", (result != nullptr ? "COLLAPSE" : "KEEP-NEW"),
            " stage=nearest exactMissed=1",
            " propId=0x", std::hex, stablePropId, std::dec,
            " nearestDistSqr=", nearestDistSqr,
            " maxDistSqr=", uniqueObjectDistanceSqr,
            " cellInsts=", probeNumCellInstances,
            " passedFilter=", probePassedFilter,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
        }
        sMtnNear += 1;
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

    // NV-DXVK [MatBind identity] cross-entry instance relink.
    //
    // For re-batched geometry (world-space vertices rewritten per frame) the
    // DrawCallCache correctly allocates a NEW BlasEntry whenever the index
    // data changes (topology is immutable on a living entry — reusing one
    // across a topology change caused VK_ERROR_DEVICE_LOST, 2026-08-02). But
    // the OBJECT still exists: last frame's instance sits in a sibling
    // entry's SpatialMap under the SAME transform key (these draws carry the
    // identity transform, and the exact key is hashed from transform bytes,
    // not position). Without this step that instance goes unpaired, dies at
    // numFramesToKeepInstances=1, and a fresh id is created here — measured
    // at ~10 of 17 draws/frame on vs 0x2859d250: the geometry flicker churn.
    //
    // So: on a full miss, search the engine-class siblings (same matsys
    // IMaterial* + shader, via DrawCallCache::m_engineClassIndex) for an
    // un-updated instance at this exact transform key, and MIGRATE it to
    // this entry. updateInstance then rebinds every surface buffer index
    // from the new entry (processInstanceBuffers runs on every first update
    // of a frame), so the instance renders the new entry's geometry while
    // keeping its id, TLAS slot continuity, and temporal history.
    if (result == nullptr && drawCallCache != nullptr && blas.engineClassKey != 0) {
      RtInstance* migrated = nullptr;
      BlasEntry* migratedFrom = nullptr;
      drawCallCache->forEachEngineClassSibling(blas.engineClassKey, [&](BlasEntry& sibling) {
        if (&sibling == &blas || migrated != nullptr) {
          return;
        }
        const RtInstance* cand =
          sibling.getSpatialMap().getDataAtTransform(firstInstanceObjectToWorld, stablePropId);
        if (cand == nullptr) {
          return;
        }
        RtInstance* c = const_cast<RtInstance*>(cand);
        // Same acceptance filters as the nearest-stage search above.
        if (c->m_frameLastUpdated == currentFrameIdx
            || c->m_materialHash != material.getHash()
            || c->m_primInstanceOwner.isSubPrim()
            || c->m_isCreatedByRenderer) {
          return;
        }
        migrated = c;
        migratedFrom = &sibling;
      });
      if (migrated != nullptr) {
        // NV-DXVK [FirstBakeHold — flicker fix]: if the DESTINATION entry's
        // first bake is still source-pending, its BLAS content is not
        // trustworthy this frame — stash the FROM-entry's built BLAS so
        // AccelManager can render the previous geometry for the handover
        // frame instead of a collapsed bake (see RtInstance member comment).
        // If the instance already carries a stash (chained re-batches on
        // consecutive frames — the observed paired dedup-miss pattern), only
        // overwrite it when the FROM entry's own bake is NOT pending:
        // otherwise we would replace a known-good BLAS with a garbage one.
        if (blas.modifiedGeometryData.pendingSrcBake
            && migratedFrom->dynamicBlas.ptr() != nullptr
            && migratedFrom->dynamicBlas->accelerationStructureReference != 0) {
          if (migrated->m_prevBlasKeepAlive.ptr() == nullptr
              || !migratedFrom->modifiedGeometryData.pendingSrcBake) {
            migrated->m_prevBlasKeepAlive = migratedFrom->dynamicBlas;
          }
        } else {
          migrated->m_prevBlasKeepAlive = nullptr;
        }
        // Order matters: the spatial-cache erase must run while the instance
        // still points at the OLD entry (it erases from m_linkedBlas's map).
        migrated->removeFromSpatialCache();
        migratedFrom->unlinkInstance(migrated);
        migrated->setBlas(blas);
        blas.linkInstance(migrated);
        const Vector3 newCentroid =
          blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
        migrated->m_spatialCacheHash =
          blas.getSpatialMap().insert(newCentroid, firstInstanceObjectToWorld, migrated, stablePropId);
        if (isProbeVS) {
          static thread_local uint32_t sRelinkProbe = 0;
          if (sRelinkProbe < 64 || (sRelinkProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[ClassRelink] #", sRelinkProbe, " f=", currentFrameIdx,
              " vs=0x", std::hex, vsHashProbe, std::dec,
              " instId=", migrated->getId(),
              " fromBlas=0x", std::hex, reinterpret_cast<uintptr_t>(migratedFrom),
              " toBlas=0x", reinterpret_cast<uintptr_t>(&blas), std::dec,
              " newPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
          }
          sRelinkProbe += 1;
        }
        result = migrated;
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
  // [ZigGun] handoff: updateInstance has no DxvkContext (can't GPU-readback), so
  // we tag the near-eye high-vtx gun instance here and read it back in
  // createViewModelInstances (which has ctx). Same frame, set before the accel
  // build, so the pointer is valid when consumed.
  static RtInstance* s_zigGunInstance = nullptr;
  // NV-DXVK: frame id when s_zigGunInstance was last tagged. The tag is a RAW
  // pointer into the pooled RtInstance storage and is only safe to dereference in
  // the SAME frame it was set. A tag left over from a previous frame can dangle
  // after the instance is freed (e.g. device loss on alt+tab skips the
  // consume+null in createViewModelInstances) -> use-after-free: getBlas() reads
  // the freed (0xDD-filled) instance's m_blas as a wild pointer, then
  // ->frameLastUpdated dereferences it and AVs. Gate every deref on this == fid.
  static uint32_t    s_zigGunInstanceFrameId = UINT32_MAX;

  void InstanceManager::updateInstance(RtInstance& currentInstance,
                                       const CameraManager& cameraManager,
                                       const BlasEntry& blas,
                                       const DrawCallState& drawCall,
                                       MaterialData& materialData) {
    // NV-DXVK [sticky IgnoreAntiCulling]: preserve IgnoreAntiCulling across
    // shader-variant draws that update the same RtInstance within a frame.
    // The mountain mesh is drawn with multiple d3d11 shader variants (e.g.
    // VS_c10aa, VS_2f543cd, VS_2904d2 SHA1 prefixes for the same logical
    // shader). Only the variants that pass SetSkyCategoryFromCb2's reproject
    // gate carry IgnoreAntiCulling on dcs; non-reproject variants arrive
    // with the flag unset, and an unconditional assignment would clobber
    // the flag that an earlier draw set this same frame. OR-preserving
    // means: once any draw this frame tagged the instance as sub-view,
    // it stays tagged for the whole frame's GC pass.
    const CategoryFlags prevFlags = currentInstance.m_categoryFlags;
    currentInstance.m_categoryFlags = drawCall.getCategoryFlags();
    if (prevFlags.test(InstanceCategories::IgnoreAntiCulling)) {
      currentInstance.m_categoryFlags.set(InstanceCategories::IgnoreAntiCulling);
    }

    // NV-DXVK [UpdInst2904 probe]: for VS_2904d2 draws, log the incoming
    // drawCall's categoryFlags and stablePropId so we know whether the
    // SetSkyCategoryFromCb2 reproject tagging is reaching updateInstance at
    // all. If incomingIgnAC=0 here despite PropIdTrace firing for the same
    // logical shader, something between d3d11_rtx and processSceneObject
    // is clearing the bit (replacement path, anonymous dcs copy, etc).
    if (blas.input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
      thread_local uint32_t sUpdProbe = 0;
      if (sUpdProbe < 16 || (sUpdProbe & 0x3FF) == 0) {
        const auto incomingFlags = drawCall.getCategoryFlags();
        Logger::info(str::format(
          "[UpdInst2904] #", sUpdProbe,
          " f=", m_device->getCurrentFrameId(),
          " incomingIgnAC=", (incomingFlags.test(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
          " incomingPropId=0x", std::hex, drawCall.getTransformData().stablePropId, std::dec,
          " incomingFlagsRaw=0x", std::hex, incomingFlags.raw(), std::dec,
          " prevInstFlagsRaw=0x", std::hex, prevFlags.raw(), std::dec,
          " finalInstFlagsRaw=0x", std::hex, currentInstance.m_categoryFlags.raw(), std::dec));
      }
      sUpdProbe += 1;
    }
    currentInstance.surface.instancesToObject = drawCall.getTransformData().instancesToObject;
    // NV-DXVK: Couple lifetime of the transform vector to this RtInstance so the
    // raw pointer above cannot dangle once the draw-call source's own ring-buffer
    // releases its reference. Null for sources with externally-owned storage
    // (USD replacements, etc.) — that's fine, they already manage lifetime.
    currentInstance.surface.instancesToObjectOwner = drawCall.getTransformData().instancesToObjectOwner;

    // NV-DXVK: flag sub-view (3D-skybox) geometry so WriteGPUData negates its
    // shading normal — corrects the reproject's inverted-winding normal flip that
    // leaves the distant mountains facing away from the sun (N·L<0 -> black).
    // Gate widened to isSubView OR isSubViewSkybox (the dome is tagged the latter
    // and may not carry isSubView at instance time).
    const bool svFlip = drawCall.getTransformData().isSubView || drawCall.getTransformData().isSubViewSkybox;
    currentInstance.surface.flipShadingNormal = RtxOptions::flipSubViewSkyboxNormals() && svFlip;
    // NV-DXVK [FlipNormalDiag]: one line per VS — confirms WHICH draws get the flip
    // and whether they even have a vertex-normal buffer (if not, the shading normal
    // is the geometric/face normal and negating normalInstanceToWorld is a no-op).
    if (svFlip) {
      static std::mutex sFnMu;
      static std::unordered_set<XXH64_hash_t> sFnLog;
      const XXH64_hash_t vsFn = drawCall.getTransformData().vertexShaderHash;
      bool firstFn = false;
      { std::lock_guard<std::mutex> g(sFnMu); firstFn = sFnLog.insert(vsFn).second; }
      if (firstFn) {
        Logger::info(str::format(
          "[FlipNormalDiag] vsXxh=0x", std::hex, uint64_t(vsFn), std::dec,
          " isSubView=", drawCall.getTransformData().isSubView ? 1 : 0,
          " isSubViewSkybox=", drawCall.getTransformData().isSubViewSkybox ? 1 : 0,
          " hasNormalBuffer=", drawCall.getGeometryData().normalBuffer.defined() ? 1 : 0,
          " flipApplied=", currentInstance.surface.flipShadingNormal ? 1 : 0));
      }
    }

    // NV-DXVK [Stable prop ID]: mirror the drawCall's per-prop identity onto
    // the instance so subsequent spatial-map operations from onTransformChanged
    // and teleport (which see no drawCall reference) can use the same cache
    // key. ONLY OVERWRITE WHEN NON-ZERO — a non-reproject draw variant for
    // the same logical mountain arrives with stablePropId=0; clobbering the
    // existing non-zero value would erase the SpatialMap entry at the propId
    // slot (via move(oldHash=propIdA, overrideHash=0) → erase+insert at
    // matrix-hash slot), guaranteeing the next frame's propId lookup misses.
    if (drawCall.getTransformData().stablePropId != 0) {
      currentInstance.m_stablePropId = drawCall.getTransformData().stablePropId;
    }

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

        // NV-DXVK [MtnAlphaState]: confirm the RESOLVED alpha state (post calculateAlphaState,
        // which can override the raw compare op) for the two SKY_MOUNTAIN VS in the black-tops
        // investigation. Prediction: culprit 0x28f7ffa9 has alphaTestType != kAlways(7) =>
        // isFullyOpaque=0 (goes through the binary cutout path -> can flip the integration
        // surface off -> demodulate-zero -> black); clean 0x29146e1d is kAlways/fullyOpaque.
        // Targeted (2 hashes) + first-update-only so it never spams.
        if (RtxOptions::logSurfaceCoverage()) {
          const uint64_t vsH = (uint64_t) drawCall.getTransformData().vertexShaderHash;
          if (vsH == 0x28f7ffa90d189017ull || vsH == 0x29146e1dd50b0314ull) {
            Logger::info(str::format(
              "[MtnAlphaState] vs=0x", std::hex, vsH, std::dec,
              " alphaTestType=", (uint32_t) alphaState.alphaTestType,
              " alphaRef=", (uint32_t) alphaState.alphaTestReferenceValue,
              " blendDisabled=", alphaState.isBlendingDisabled ? 1 : 0,
              " fullyOpaque=", alphaState.isFullyOpaque ? 1 : 0,
              " invertedBlend=", alphaState.invertedBlend ? 1 : 0,
              " emissiveBlend=", alphaState.emissiveBlend ? 1 : 0,
              " isParticle=", alphaState.isParticle ? 1 : 0,
              " isDecal=", alphaState.isDecal ? 1 : 0));
          }
        }
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
        // NV-DXVK [VsColor]: stamp the per-pixel vertex-shader identity. Taken
        // from the DRAW (not the linked BlasEntry): when several draws share a
        // BlasEntry the blas reports only whichever VS created it, which is the
        // exact aliasing that made every VS-gated probe in this investigation a
        // blind spot. The draw's own hash is what actually produced the pixel.
        currentInstance.surface.vsDebugId =
          acquireVsDebugId(drawCall.getTransformData().vertexShaderHash);

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

          // NV-DXVK: TF2 3D-skybox cloud billboard handling. A cloud
          // billboard = an instance in the 3D skybox (IgnoreAntiCulling,
          // applied by SetSkyCategoryFromCb2) whose material is the
          // fog-synthesizing premultiplied uber shader (Tf2SkyboxFog). The
          // material flag alone is too broad — that PS family is also used
          // by playable-world smoke / effects / props — so it is ANDed with
          // the 3D-skybox category.
          //
          // Switched by rtx.enableTf2SkyboxCloudFog:
          //  - enabled:  flag Surface::isTf2SkyboxFog so the opaque material
          //              interaction reconstructs the game's fog-blend
          //              colour (the alpha-blend unlit-cloud composite keys
          //              off the same flag).
          //  - disabled: the cloud texture is a near-black coverage map with
          //              no colour of its own, so leaving the billboards in
          //              renders them solid black. Hide them entirely
          //              instead — m_isHidden forces the instance mask to 0
          //              (see below), so the rays miss them and the sky
          //              shows through cleanly.
          const bool ignoreAntiCullCat =
            currentInstance.testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
          const bool matTf2Fog =
            materialData.getOpaqueMaterialData().getTf2SkyboxFog();
          const bool isTf2Cloud = ignoreAntiCullCat && matTf2Fog;
          const bool tf2CloudFogEnabled = RtxOptions::enableTf2SkyboxCloudFog();
          currentInstance.surface.isTf2SkyboxFog = isTf2Cloud && tf2CloudFogEnabled;
          // [TF2 fog-pipeline hide] When the user has fog reconstruction
          // disabled, we hide every surface tagged matTf2Fog — not just
          // the sub-view ones the original isTf2Cloud check caught.
          // Bisect data: with isTf2Cloud-only the hide-list correctly
          // dropped the four sub-view cloud VSes (0x2904, 0x290deec3,
          // 0x296dc3ae, 0x2a904f3d) but three MAIN-WORLD matTf2Fog
          // surfaces (0x29a262d2, 0x29566a60, 0x28d6a5dc) were left
          // visible — they render through the premult-encode bypass
          // without the fog reconstruction the original PS expected,
          // producing the dark sky-region corruption. Same intent as
          // the cloud hide: "we can't reconstruct the fog math, so
          // don't render the fog-dependent surface". The
          // IgnoreAntiCulling restriction in the original gate was
          // incidental to the cloud-billboard case and excluded
          // legitimate fog-pipeline targets that happen to live in
          // main-world coords.
          if (matTf2Fog && !tf2CloudFogEnabled) {
            currentInstance.m_isHidden = true;
          }

          // NV-DXVK [FogHideProbe]: the visible garbage is ~29 stacked VS
          // 0x29566a60 draws at one origin, flickering. [Tf2CloudClass] is
          // one-shot-per-VS so it can't reveal whether EVERY stacked draw gets
          // matTf2Fog/hidden. Log each distinct (ignoreAntiCull,matTf2Fog,
          // m_isHidden) outcome for this VS — if any line shows matTf2Fog=0 (or
          // m_isHidden=0), those draws are NOT hidden → they render and flicker.
          {
            const uint64_t vsHfh = uint64_t(drawCall.getTransformData().vertexShaderHash);
            if (vsHfh == 0x29566a60d473af50ull) {
              static std::mutex sFhMu;
              static std::unordered_set<uint32_t> sFhSeen;
              const uint32_t key = (ignoreAntiCullCat ? 4u : 0u)
                                 | (matTf2Fog ? 2u : 0u)
                                 | (currentInstance.m_isHidden ? 1u : 0u);
              bool firstFh = false;
              { std::lock_guard<std::mutex> g(sFhMu);
                if (sFhSeen.insert(key).second) firstFh = true; }
              if (firstFh)
                Logger::warn(str::format(
                  "[FogHideProbe] VS=0x29566a60 matTf2Fog=", (matTf2Fog ? 1 : 0),
                  " m_isHidden=", (currentInstance.m_isHidden ? 1 : 0),
                  " ignoreAntiCull=", (ignoreAntiCullCat ? 1 : 0),
                  " tf2CloudFogEnabled=", (tf2CloudFogEnabled ? 1 : 0)));
            }
          }

          // NV-DXVK [SV_Coverage hide]: PSes that write SV_Coverage
          // (oMask) implement smooth alpha via MSAA sub-pixel sample
          // masking. The path tracer can't honor oMask — full RGBA
          // hits every pixel, producing the visible BOXY corruption
          // in TF2's 3D-skybox (VS_95da0b01 + FS_e508ad41 = the
          // single-tri sky-noise overlay flooding the sky speckling
          // pattern the user reported). Hide unconditionally; there
          // is no path-tracer-compatible rendering of these draws.
          // Flag set in d3d11_rtx.cpp::FillMaterialData by walking
          // the PS OSGN for systemValueType == D3D_NAME_COVERAGE.
          if (drawCall.getMaterialData().sourcePsWritesCoverageMask) {
            currentInstance.m_isHidden = true;
          }

          // NV-DXVK [Tf2CloudClass]: one-shot per VS hash, log the
          // classification result so we can debug "cloud not being
          // tagged Tf2Cloud" vs "tagged but rendering wrong" without
          // recompile. Caps at 32 distinct VSes per session.
          // Gameplay-gated via captureCount > 16.
          {
            if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u
                && (ignoreAntiCullCat || matTf2Fog)) {
              static std::mutex sTf2CcMu;
              static std::unordered_set<uint64_t> sTf2CcSeen;
              const uint64_t vsHashCc = uint64_t(drawCall.getTransformData().vertexShaderHash);
              bool firstCc = false;
              {
                std::lock_guard<std::mutex> g(sTf2CcMu);
                if (sTf2CcSeen.size() < 32u
                    && sTf2CcSeen.insert(vsHashCc).second) {
                  firstCc = true;
                }
              }
              if (firstCc) {
                Logger::warn(str::format(
                  "[Tf2CloudClass] vsXxh=0x", std::hex, vsHashCc, std::dec,
                  " ignoreAntiCull=", (ignoreAntiCullCat ? 1 : 0),
                  " matTf2Fog=", (matTf2Fog ? 1 : 0),
                  " isTf2Cloud=", (isTf2Cloud ? 1 : 0),
                  " tf2CloudFogEnabled=", (tf2CloudFogEnabled ? 1 : 0),
                  " → surface.isTf2SkyboxFog=", (currentInstance.surface.isTf2SkyboxFog ? 1 : 0),
                  " m_isHidden=", (currentInstance.m_isHidden ? 1 : 0)));
              }
            }
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
        int mtnMovePath = -1;  // 0=teleport(prev=cur,zero motion) 1=move(prev=old,correct) 2=moveAgain(prev STALE)
        if (isFirstUpdateAfterCreation) {
          hasTransformChanged = currentInstance.teleport(objectToWorld);
          mtnMovePath = 0;
        } else if (isFirstUpdateThisFrame) {
          hasTransformChanged = currentInstance.move(objectToWorld);
          mtnMovePath = 1;
        } else {
          hasTransformChanged = currentInstance.moveAgain(objectToWorld);
          mtnMovePath = 2;
        }

        // NV-DXVK [MtnMotion]: for the black/streaking SKY_MOUNTAIN VS, log how motion is being
        // tracked. The streak + low denoiser confidence point at a motion-vector fault. This
        // shows, per draw: which update path ran (teleport/move/moveAgain), whether the engine
        // marked a transform change, the material type + isMotionUnstable gate (translucent kills
        // prev-positions), and the actual prev-vs-cur world translation delta. If delta~0 while
        // the camera moves, OR path=0/2 in steady state, motion is being dropped -> streak + the
        // gradient spike that collapses RTXDI confidence -> NRD black.
        if (RtxOptions::logSurfaceCoverage()) {
          const uint64_t vsHmm = (uint64_t) drawCall.getTransformData().vertexShaderHash;
          if (vsHmm == 0x28f7ffa90d189017ull || vsHmm == 0x29146e1dd50b0314ull) {
            const Vector4 cT = currentInstance.surface.objectToWorld[3];
            const Vector4 pT = currentInstance.surface.prevObjectToWorld[3];
            const float dT = std::sqrt(float((cT.x-pT.x)*(cT.x-pT.x) + (cT.y-pT.y)*(cT.y-pT.y) + (cT.z-pT.z)*(cT.z-pT.z)));
            // NV-DXVK [MtnMotion] streak hunt: the motion vector is correct ONLY if this
            // frame's prevObjectToWorld == LAST frame's reprojected objectToWorld. The
            // ssmv probe shows the screen MV pops every ~3 frames (=streak); test whether
            // that pop is a STALE PREV (prev didn't follow the reproject drift) or a
            // genuine restream gap. Keyed by stable instance id so fanout (instCount=2)
            // doesn't mix the two instances. Logs:
            //  fsl  = frames since this instance was last seen here (1=every frame; >1=restream gap / skipped frame)
            //  rep  = |curT - lastCurT|   how far the reproject moved this instance's world pos vs last frame (T_reproject change)
            //  pml  = |prevT - lastCurT|  prev-matches-last; ~0 = prev correctly == last frame's cur (good); large = STALE PREV (bad)
            // DECIDER on an ssmv-spike frame: pml large => fix prev-tracking (move()/prev not getting last reproject);
            //                                 pml ~0 but screen MV still spikes => camera-phase mismatch (GPU prev-cam != prev-o2w frame).
            const uint32_t curFrameMM = m_device->getCurrentFrameId();
            const uint64_t instId = currentInstance.getId();
            struct LastSeen { uint32_t frame; float x, y, z; };
            static std::mutex sMtnMtx;
            static std::unordered_map<uint64_t, LastSeen> sLastSeen;
            int      fsl = -1;
            float    rep = -1.0f, pml = -1.0f;
            {
              std::lock_guard<std::mutex> lk(sMtnMtx);
              auto it = sLastSeen.find(instId);
              if (it != sLastSeen.end() && isFirstUpdateThisFrame) {
                const LastSeen& ls = it->second;
                fsl = int(curFrameMM) - int(ls.frame);
                rep = std::sqrt((float(cT.x)-ls.x)*(float(cT.x)-ls.x) + (float(cT.y)-ls.y)*(float(cT.y)-ls.y) + (float(cT.z)-ls.z)*(float(cT.z)-ls.z));
                pml = std::sqrt((float(pT.x)-ls.x)*(float(pT.x)-ls.x) + (float(pT.y)-ls.y)*(float(pT.y)-ls.y) + (float(pT.z)-ls.z)*(float(pT.z)-ls.z));
              }
              if (isFirstUpdateThisFrame)
                sLastSeen[instId] = LastSeen { curFrameMM, float(cT.x), float(cT.y), float(cT.z) };
            }
            Logger::info(str::format(
              "[MtnMotion] f=", curFrameMM, " vs=0x", std::hex, vsHmm, std::dec,
              " id=", instId,
              " path=", mtnMovePath, " xfChanged=", hasTransformChanged ? 1 : 0,
              " matType=", (uint32_t) currentInstance.m_materialType,
              " motionUnstable=", isMotionUnstable ? 1 : 0,
              " hasPrevPos=", hasPreviousPositions ? 1 : 0,
              " firstThisFrame=", isFirstUpdateThisFrame ? 1 : 0,
              " fsl=", fsl, " rep=", rep, " pml=", pml,
              " curT=(", float(cT.x), ",", float(cT.y), ",", float(cT.z), ")",
              " prevDelta=", dT));
          }
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

    // NV-DXVK [Perf.NonOpaque]: which branch below claimed this instance. Only
    // kOpaque avoids the primary-ray resolve loop; every other value means a ray
    // that hits this instance re-traces the whole TLAS at least once more.
    enum NonOpaqueReason : uint8_t {
      kRTRenderTarget = 0,  // forced opaque-pass, any-hit geometry flags
      kUnorderedBlend = 1,  // particle/decal/player-blend/emissive -> unordered TLAS + FORCE_NO_OPAQUE
      kAlphaTested    = 2,  // primary TLAS, duplicate any-hits allowed
      kAlphaBlended   = 3,  // primary TLAS, FORCE_NO_OPAQUE
      kTranslucent    = 4,
      kRayPortal      = 5,
      kClipPlane      = 6,  // FORCE_NO_OPAQUE purely to run clip planes in any-hit
      kOpaque         = 7,
      kReasonCount    = 8
    };
    uint8_t nonOpaqueReason = kOpaque;

    // Update the geometry and instance flags
    if (currentInstance.isOpaque() && drawCall.isUsingRaytracedRenderTarget) {
      nonOpaqueReason = kRTRenderTarget;
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
      nonOpaqueReason = kUnorderedBlend;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      currentInstance.m_isUnordered = true;
      // Unordered resolve only accumulates via any-hits and ignores opaque hits, therefore force 
      // the opaque hits resolve via OMMs to be turned into any-hits.
      // Note: this has unexpected effect even with OMM off and results in minor visual changes in Portal MF A DLSS test
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque && currentInstance.surface.alphaState.isBlendingDisabled) {
      // Alpha-tested geometry goes to the primary TLAS as non-opaque geometry with potential duplicate hits.
      nonOpaqueReason = kAlphaTested;
      currentInstance.m_geometryFlags = 0;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque) {
      // Alpha-blended geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      nonOpaqueReason = kAlphaBlended;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      // Treat all non-transparent hits as any-hits
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::Translucent) {
      // Translucent (e.g. glass) geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      nonOpaqueReason = kTranslucent;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
      // Portals go to the primary TLAS as opaque.
      nonOpaqueReason = kRayPortal;
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    } else if (currentInstance.surface.isClipPlaneEnabled) {
      nonOpaqueReason = kClipPlane;
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

    // [ZigGun] RE-ANCHORED on the POSITIVELY IDENTIFIED gun VS hash. The old
    // near-eye/vtx>1000 heuristic tagged the WRONG object (a 3-vtx decoy / player
    // body), so every prior [Zig*] measurement was of a non-gun — the whole v2/v3
    // dead-end. The first-person weapon is VS=0x292b6ba0d1854f28, confirmed via the
    // PickRegion coverage probe (largest stable center-bottom surface) AND a hide
    // test: rtx.debug.hideVertexShaders=0x292b6ba0d1854f28 removes the weapon.
    // Selecting s_zigGunInstance by this hash makes the downstream [ZigGunRB] /
    // [ZigNDC] skinned-vertex readback finally measure the thing that zig-zags.
    //
    // The per-frame line below localizes the horizontal sawtooth. The gun BLAS is
    // LOCAL space + an objectToWorld, so its world origin is the o2w translation;
    // we project that through the Main worldToView -> viewToProjection to a screen
    // ndcX. Across a walk cycle:
    //   sawtooth in wp/o2wT.x        -> the INSTANCE TRANSFORM jitters (fix there)
    //   smooth wp but sawtooth ndcX  -> the Main worldToView/camera is the source
    //   both smooth, gun still jerks -> the SKELETON/BONES (watch [ZigNDC], which
    //                                   reads the actual skinned verts, not o2w)
    // Gameplay-gated + throttled per project logging conventions. NOTE: to measure
    // the gun you must REMOVE it from rtx.debug.hideVertexShaders (hidden -> mask=0,
    // excluded from the AS build the readback consumes).
    {
      static constexpr XXH64_hash_t kGunVsHash = 0x292b6ba0d1854f28ull;
      if (drawCall.getTransformData().vertexShaderHash == kGunVsHash
          && !currentInstance.m_isHidden
          && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
        const uint32_t vtx = currentInstance.getBlas() ? currentInstance.getBlas()->modifiedGeometryData.vertexCount : 0u;
        // NV-DXVK: drop a stale tag from a previous frame BEFORE dereferencing it
        // below. The pointed-to RtInstance may have been freed since it was tagged
        // (device loss on alt+tab), so the getBlas() comparison would be a
        // use-after-free (rax=0xDD). Same-frame retags stay valid.
        const uint32_t zigFid = m_device->getCurrentFrameId();
        if (s_zigGunInstanceFrameId != zigFid) {
          s_zigGunInstance = nullptr;
          s_zigGunInstanceFrameId = zigFid;
        }
        // Tag the highest-vtx gun draw (the body, if the weapon is split across
        // sub-meshes) for the ctx-readback in createViewModelInstances.
        if (s_zigGunInstance == nullptr
            || (s_zigGunInstance->getBlas() && vtx > s_zigGunInstance->getBlas()->modifiedGeometryData.vertexCount)) {
          s_zigGunInstance = &currentInstance;
        }
        static uint32_t sGunF = 0; static uint32_t sGunC = 0;
        const uint32_t fid = m_device->getCurrentFrameId();
        if (fid != sGunF) { sGunF = fid; sGunC = 0; }
        if (sGunC < 6) {
          ++sGunC;
          const auto& mainCam = cameraManager.getMainCamera();
          const Matrix4 w2v = mainCam.getWorldToView(false);
          const Matrix4 v2p = mainCam.getViewToProjection();
          const Vector3 wp = currentInstance.getWorldPosition();
          const Matrix4 o2w = currentInstance.getTransform();
          const Vector4 vpos = w2v * Vector4(wp, 1.0f);
          const Vector4 ppos = v2p * vpos;
          const float ndcX = (ppos.w != 0.0f) ? (ppos.x / ppos.w) : 0.0f;
          Logger::info(str::format(
            "[ZigGun] f=", fid,
            " camType=", (uint32_t)drawCall.cameraType,
            " vtx=", vtx,
            " wp=(", wp.x, ",", wp.y, ",", wp.z, ")",
            " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
            " viewX=", vpos.x, " viewZ=", vpos.z,
            " ndcX=", ndcX,
            " mask=0x", std::hex, (uint32_t)currentInstance.getVkInstance().mask, std::dec));
        }
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

    // NV-DXVK [Perf.NonOpaque]: census of the classification above, over the
    // instances that actually end up in a TLAS (mask != 0, not hidden).
    //
    // Why this and not [Perf.Tlas]/[Perf.TlasOverlap]: those count PRIMITIVES,
    // and the primitive share of any-hit geometry (3.1%) was used to retire the
    // opacity theory. That is the wrong denominator. The primary ray runs with
    // RAY_FLAG_FORCE_OPAQUE, so non-opaque geometry never costs any-hit
    // traversal - what it costs is an extra full-TLAS re-trace per pixel from
    // the resolve loop, and that fires once per *surface crossed*, regardless of
    // how many triangles that surface has. A 2-triangle fullscreen haze quad and
    // a 100k-triangle wall cost the resolve loop exactly the same. So the
    // denominator has to be instances (and, better, screen coverage), which is
    // where the 39%-of-instances number came from and why it disagrees so hard
    // with the 3.1%-of-primitives number.
    //
    // Reported once per second so it is safe to leave on during a timing run.
    if (RtxOptions::perfNonOpaqueCensus()
        && !currentInstance.m_isHidden
        && currentInstance.getVkInstance().mask != 0) {
      struct NonOpaqueCensus {
        std::mutex mu;
        uint64_t frames = 0;
        uint64_t instances = 0;
        uint64_t byReason[kReasonCount] = {};
        uint64_t primsByReason[kReasonCount] = {};
        uint64_t forceNoOpaque = 0;
        uint64_t unorderedTlas = 0;
        uint32_t lastFrame = UINT32_MAX;
        // Which vertex shaders produce the resolve-loop drivers, tallied by
        // INSTANCE count rather than primitive count. Primitive count is the
        // wrong weight for this mechanism by the same argument as above: the
        // resolve loop pays per surface crossed, not per triangle. Naming the
        // vertex shader is what makes the number actionable, since it maps back
        // to a specific piece of TF2 content.
        struct Driver { uint64_t instances = 0; uint64_t prims = 0; uint8_t reason = kOpaque; };
        std::unordered_map<uint64_t /*vsHash*/, Driver> byVs;
        std::chrono::steady_clock::time_point lastLog {};
      };
      static NonOpaqueCensus s_census;

      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t prims = drawCall.getGeometryData().calculatePrimitiveCount();
      const uint64_t vsHash = uint64_t(drawCall.getTransformData().vertexShaderHash);
      const bool fno = (currentInstance.getVkInstance().flags
                        & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) != 0;

      std::lock_guard<std::mutex> g(s_census.mu);

      if (s_census.lastFrame != frameId) {
        s_census.lastFrame = frameId;
        ++s_census.frames;
      }
      ++s_census.instances;
      ++s_census.byReason[nonOpaqueReason];
      s_census.primsByReason[nonOpaqueReason] += prims;
      if (fno)
        ++s_census.forceNoOpaque;
      if (currentInstance.m_isUnordered)
        ++s_census.unorderedTlas;

      const bool isDriver = nonOpaqueReason == kUnorderedBlend
                         || nonOpaqueReason == kAlphaTested
                         || nonOpaqueReason == kAlphaBlended
                         || nonOpaqueReason == kTranslucent
                         || nonOpaqueReason == kClipPlane;
      if (isDriver) {
        // Cap only new keys, so an already-tracked shader keeps accumulating
        // instead of freezing at whatever count it had when the map filled.
        auto it = s_census.byVs.find(vsHash);
        if (it == s_census.byVs.end() && s_census.byVs.size() < 512)
          it = s_census.byVs.emplace(vsHash, NonOpaqueCensus::Driver {}).first;
        if (it != s_census.byVs.end()) {
          ++it->second.instances;
          it->second.prims += prims;
          it->second.reason = nonOpaqueReason;
        }
      }

      const auto now = std::chrono::steady_clock::now();
      if (s_census.lastLog.time_since_epoch().count() == 0)
        s_census.lastLog = now;
      else if (now - s_census.lastLog >= std::chrono::seconds(1) && s_census.frames > 0) {
        const double ff = double(s_census.frames);
        const double perFrameInst = double(s_census.instances) / ff;
        // Only these five branches produce a surface whose material can resolve
        // below resolveOpaquenessThreshold and therefore re-arm continueResolving.
        // kRayPortal and kRTRenderTarget are deliberately excluded: both take an
        // early-out path in resolveVertex, so counting them here would inflate the
        // headline number with instances that do not drive the loop.
        const uint64_t resolveLoopDrivers =
            s_census.byReason[kUnorderedBlend]
          + s_census.byReason[kAlphaTested]
          + s_census.byReason[kAlphaBlended]
          + s_census.byReason[kTranslucent]
          + s_census.byReason[kClipPlane];
        const double driverPct = s_census.instances
          ? 100.0 * double(resolveLoopDrivers) / double(s_census.instances) : 0.0;

        static const char* kReasonNames[kReasonCount] = {
          "rtTarget", "unorderedBlend", "alphaTest", "alphaBlend",
          "translucent", "portal", "clipPlane", "opaque"
        };

        std::string byReason;
        for (uint32_t r = 0; r < kReasonCount; ++r) {
          byReason += str::format(" ", kReasonNames[r], "=",
                                  double(s_census.byReason[r]) / ff,
                                  "(", double(s_census.primsByReason[r]) / ff, "p)");
        }

        // Top vertex shaders by driver-instance count over the window.
        std::vector<std::pair<uint64_t, NonOpaqueCensus::Driver>> ranked(
          s_census.byVs.begin(), s_census.byVs.end());
        std::partial_sort(
          ranked.begin(),
          ranked.begin() + std::min<size_t>(5, ranked.size()),
          ranked.end(),
          [](const auto& a, const auto& b) { return a.second.instances > b.second.instances; });

        std::string offenders;
        for (size_t i = 0; i < ranked.size() && i < 5; ++i) {
          offenders += str::format(" vs=0x", std::hex, ranked[i].first, std::dec,
                                   ":", kReasonNames[ranked[i].second.reason],
                                   ":", double(ranked[i].second.instances) / ff, "inst",
                                   ":", double(ranked[i].second.prims) / ff, "p");
        }

        Logger::warn(str::format(
          "[Perf.NonOpaque] f=", frameId, " frames=", s_census.frames,
          " instPerFrame=", perFrameInst,
          " resolveLoopDrivers=", double(resolveLoopDrivers) / ff, " (", driverPct, "%)",
          " forceNoOpaque=", double(s_census.forceNoOpaque) / ff,
          " unorderedTlas=", double(s_census.unorderedTlas) / ff,
          " | byReason(count(prims)):", byReason.c_str(),
          " | topDriversByInstances:", offenders.c_str()));

        // Field-wise reset: the struct holds a std::mutex (which this scope still
        // owns the lock on), so it is neither copy- nor move-assignable.
        s_census.frames = 0;
        s_census.instances = 0;
        s_census.forceNoOpaque = 0;
        s_census.unorderedTlas = 0;
        for (uint32_t r = 0; r < kReasonCount; ++r) {
          s_census.byReason[r] = 0;
          s_census.primsByReason[r] = 0;
        }
        s_census.byVs.clear();
        s_census.lastLog = now;
        // Not frameId: the rest of THIS frame still has instances to report, and
        // they would otherwise land in a window whose frame count is zero.
        s_census.lastFrame = UINT32_MAX;
      }
    }

    // [CloudRoute] How are the 3D-skybox blended billboards (TF2 sky cloud
    // cards) routed? They reach here IgnoreAntiCulling-tagged + blend-enabled.
    // Logged AFTER billboard generation so m_billboardCount is final. route=
    // decodes the final object mask into the BVH/pass the geometry traces in:
    //   OPAQUE      -> opaque BVH, drawn solid (the bug, if a cloud lands here)
    //   ALPHA_BLEND -> alpha-blend BVH (ordered translucent)
    //   TRANSLUCENT -> translucent material pass
    //   U_BLENDED/U_EMISSIVE -> unordered TLAS (m_isUnordered billboard path)
    //   <none/hidden> -> mask==0, not traced at all
    // Keyed per vsHash, gameplay-gated.
    if (drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling)
        && !currentInstance.surface.alphaState.isBlendingDisabled
        && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      const uint32_t vsHashCR = uint32_t(uint64_t(drawCall.getTransformData().vertexShaderHash) & 0xffffffffu);
      static std::mutex s_cloudRouteMu;
      static std::unordered_set<uint32_t> s_cloudRouteSeen;
      bool firstCR = false;
      {
        std::lock_guard<std::mutex> g(s_cloudRouteMu);
        if (s_cloudRouteSeen.size() < 128 && s_cloudRouteSeen.insert(vsHashCR).second)
          firstCR = true;
      }
      if (firstCR) {
        const auto& as = currentInstance.surface.alphaState;
        const auto& bm = drawCall.getMaterialData().blendMode;
        const uint32_t m = currentInstance.getVkInstance().mask;
        // The unordered TLAS uses a SEPARATE bit namespace from the standard
        // mask - decode with the right one (matches the routing if/else above).
        const bool unorderedNs = currentInstance.m_isUnordered
                              && RtxOptions::enableSeparateUnorderedApproximations();
        std::string route;
        if (m == 0) {
          route = "<none/hidden>";
        } else if (unorderedNs) {
          if (m & OBJECT_MASK_UNORDERED_EMISSIVE_GEOMETRY)               route += "U_EMISSIVE_GEO ";
          if (m & OBJECT_MASK_UNORDERED_BLENDED_GEOMETRY)                route += "U_BLENDED_GEO ";
          if (m & OBJECT_MASK_UNORDERED_EMISSIVE_INTERSECTION_PRIMITIVE) route += "U_EMISSIVE_PRIM ";
          if (m & OBJECT_MASK_UNORDERED_BLENDED_INTERSECTION_PRIMITIVE)  route += "U_BLENDED_PRIM ";
        } else {
          if (m & OBJECT_MASK_TRANSLUCENT)          route += "TRANSLUCENT ";
          if (m & OBJECT_MASK_PORTAL)               route += "PORTAL ";
          if (m & OBJECT_MASK_ALPHA_BLEND)          route += "ALPHA_BLEND ";
          if (m & OBJECT_MASK_OPAQUE)               route += "OPAQUE ";
          if (m & OBJECT_MASK_VIEWMODEL)            route += "VIEWMODEL ";
          if (m & OBJECT_MASK_VIEWMODEL_VIRTUAL)    route += "VIEWMODEL_VIRTUAL ";
          if (m & OBJECT_MASK_PLAYER_MODEL)         route += "PLAYER_MODEL ";
          if (m & OBJECT_MASK_PLAYER_MODEL_VIRTUAL) route += "PLAYER_MODEL_VIRTUAL ";
        }
        if (route.empty()) route = "<unknown>";
        Logger::warn(str::format("[CloudRoute] vs=0x", std::hex, vsHashCR, std::dec,
          " cam=", static_cast<int>(drawCall.cameraType),
          " route=", route.c_str(),
          "(mask=0x", std::hex, m, std::dec, ")",
          " isUnordered=", currentInstance.m_isUnordered ? 1 : 0,
          " billboards=", currentInstance.m_billboardCount,
          " billboardsGen=", billboardsGotGenerated ? 1 : 0,
          " isHidden=", currentInstance.m_isHidden ? 1 : 0,
          " isPlayerModel=", currentInstance.m_isPlayerModel ? 1 : 0,
          " matType=", static_cast<int>(currentInstance.m_materialType),
          " | alphaState: fullyOpaque=", as.isFullyOpaque ? 1 : 0,
          " blendingDisabled=", as.isBlendingDisabled ? 1 : 0,
          " blendType=", static_cast<int>(as.blendType),
          " invertedBlend=", as.invertedBlend ? 1 : 0,
          " emissiveBlend=", as.emissiveBlend ? 1 : 0,
          " isParticle=", as.isParticle ? 1 : 0,
          " isDecal=", as.isDecal ? 1 : 0,
          " alphaTestType=", static_cast<int>(as.alphaTestType),
          // VkBlendFactor: 0=ZERO 1=ONE 6=SRC_ALPHA 7=ONE_MINUS_SRC_ALPHA;
          // VkBlendOp: 0=ADD. Used to see why looksPremultiplied failed in
          // calculateAlphaState (the ONE,ONE_MINUS_SRC_ALPHA disambiguation).
          " | blend: colorSrc=", static_cast<int>(bm.colorSrcFactor),
          " colorDst=", static_cast<int>(bm.colorDstFactor),
          " colorOp=", static_cast<int>(bm.colorBlendOp),
          " alphaSrc=", static_cast<int>(bm.alphaSrcFactor),
          " alphaDst=", static_cast<int>(bm.alphaDstFactor),
          " alphaOp=", static_cast<int>(bm.alphaBlendOp),
          " writeMask=0x", std::hex, static_cast<uint32_t>(bm.writeMask), std::dec));
      }
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

    // NV-DXVK [zig-zag ROOT FIX]: for the TF2 engine-hook viewmodel the per-draw
    // o2w (reference.getTransform()) was sampled from the camera-manager Main on
    // the DRAW thread — a frame out of step with the perspectiveCorrection built
    // on the CS thread (proven: [ZigSync] renderCam lagged [ZigPC] mV2W_T by one
    // frame, so mainW2V*o2w != I and the residual sawtoothed into the horizontal
    // zig-zag). For the Main terms to cancel (clip = vmProj*scale*v) the o2w MUST
    // be the IDENTICAL matrix perspectiveCorrection used. Re-source it here from
    // the SAME camera-manager Main — same CS thread, same frame as the pc built
    // by the caller — and teleport the instance so the geometry-correction path
    // also renders with it. Gated to the engine-hook path so non-TF2 viewmodels
    // keep their real per-draw o2w.
    // [ZigProbe] instrumentation state — what happened to THIS viewmodel instance.
    int  zigBranch = -1;            // 0=ordinary-teleport, 1=geometry-dispatched, 2=geom-gated-out
    bool zigDispatched = false;
    uint64_t zigPosAddrBefore = 0, zigPosAddrAfter = 0;
    uint32_t zigVtxCount = 0;
    const void* zigBlasPtr = (const void*)viewModelInstance->getBlas();
    if (viewModelInstance->getBlas() != nullptr) {
      const auto& pb0 = viewModelInstance->getBlas()->modifiedGeometryData.positionBuffer;
      if (pb0.defined()) { zigPosAddrBefore = (uint64_t)pb0.getDeviceAddress() + pb0.offsetFromSlice(); }
      zigVtxCount = viewModelInstance->getBlas()->modifiedGeometryData.vertexCount;
    }

    Matrix4 refXform = reference.getTransform();
    Matrix4 refPrevXform = reference.getPrevTransform();
    if (RtxOptions::useEngineHookMainCamera()) {
      const auto& mainCam =
        ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
      refXform = mainCam.getViewToWorld(false);
      refPrevXform = mainCam.getPreviousViewToWorld(false);
      viewModelInstance->teleport(refXform, refPrevXform);
    }

    if (RtxOptions::ViewModel::perspectiveCorrection()) {
      // A transform that looks "correct" only from a main camera's point of view
      const auto corrected = perspectiveCorrection * refXform;
      const auto prevCorrected = prevPerspectiveCorrection * refPrevXform;

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
      // [zig-zag FIX] GROUND TRUTH ([ZigNDC]): v0 IS the gun's world position
      // (v0-camMain=(0,0,-60)); mainW2V*v0 gives view-space pos with constant Y
      // but SAWTOOTHING X = the horizontal zig-zag. v0 lags because it's baked in
      // the ViewModel-camera frame, which phase-lags Main. The RT renders v0 as
      // world (instance transform is effectively ignored — proven: applying it
      // throws v0 14000u away yet the gun renders glued). So the fix must modify
      // v0, and we FORCE the geometry path to do so for the engine-hook viewmodel.
      if (isOrdinary(corrected) && isOrdinary(prevCorrected) && !RtxOptions::useEngineHookMainCamera()) {
        viewModelInstance->teleport(corrected, prevCorrected);
        zigBranch = 0;
      } else {
        ONCE(Logger::info("[RTX-Compatibility-Info] Unexpected values in the perspective-corrected transform of a view model. Fallback to geometry modification"));
        // Only need to run this on BVH op (maybe this could be moved to geometry processing?)
        if (viewModelInstance->getBlas()->frameLastUpdated == frameId) {
          Matrix4d instancePositionTransform;
          if (RtxOptions::useEngineHookMainCamera()) {
            // Reproject the lagging world verts so they are GLUED to the Main
            // camera: v_new = mainV2W * vmW2V * v. Then mainW2V*v_new = vmW2V*v =
            // the gun relative to its OWN (same-frame) bake camera = constant, no
            // lag. Verify: [ZigNDC] viewDirect.x must go flat (was -7..+68 sawtooth).
            auto& cm = ctx->getCommonObjects()->getSceneManager().getCameraManager();
            const Matrix4 mainV2W = cm.getCamera(CameraType::Main).getViewToWorld(false);
            const Matrix4 vmW2V   = cm.getCamera(CameraType::ViewModel).getWorldToView(false);
            instancePositionTransform = Matrix4d(mainV2W * vmW2V);
          } else {
            const auto worldToObject = inverse(refXform);
            instancePositionTransform = worldToObject * perspectiveCorrection * refXform;
          }

          ctx->getCommonObjects()->metaGeometryUtils().dispatchViewModelCorrection(ctx,
            viewModelInstance->getBlas()->modifiedGeometryData, instancePositionTransform);
          zigBranch = 1; zigDispatched = true;

          // [zig-zag FIX] the verts are now WORLD-space (reprojected onto Main).
          // The instance transform must be IDENTITY so the RT renders them as-is
          // instead of re-applying mainV2W (=refXform from line 2724), which would
          // re-displace the now-world geometry. This is the completing half of the
          // geometry-path fix.
          if (RtxOptions::useEngineHookMainCamera()) {
            viewModelInstance->teleport(Matrix4());
          }
        } else {
          zigBranch = 2;
        }
      }
    }

    // [ZigProbe] DECISIVE per-instance dump: which BLAS, which buffer, branch,
    // whether the correction ran, the buffer device-address (match this against
    // [ZigBlas] in accel_manager to see if the RT builds from THIS buffer), and
    // the final instance transform. Logged for every viewmodel instance, every
    // frame, so multi-instance / wrong-BLAS / gated-out cases are all visible.
    {
      if (viewModelInstance->getBlas() != nullptr) {
        const auto& pb1 = viewModelInstance->getBlas()->modifiedGeometryData.positionBuffer;
        if (pb1.defined()) { zigPosAddrAfter = (uint64_t)pb1.getDeviceAddress() + pb1.offsetFromSlice(); }
      }
      const auto& ft = viewModelInstance->getTransform();
      Logger::info(str::format(
        "[ZigInst] f=", frameId,
        " inst=", (const void*)viewModelInstance,
        " blas=", zigBlasPtr,
        " branch=", zigBranch, " dispatched=", (zigDispatched ? 1 : 0),
        " vtx=", zigVtxCount,
        " posAddrBefore=", zigPosAddrBefore,
        " posAddr=", zigPosAddrAfter,
        " mask=0x", std::hex, (uint32_t)viewModelInstance->m_vkInstance.mask, std::dec,
        " o2wT=(", ft[3][0], ",", ft[3][1], ",", ft[3][2], ")",
        " blasFrameUpd=", (viewModelInstance->getBlas() ? viewModelInstance->getBlas()->frameLastUpdated : 0u),
        " thisFrame=", frameId));
    }

    // NV-DXVK [ZigScreen]: measure the gun's ACTUAL on-screen position of a
    // FIXED object-space point through the exact effective chain it renders with
    // (mainCam.viewToProj * mainCam.worldToView * perspectiveCorrection * refXform).
    // A fixed point removes the bones from the equation:
    //   ndc sawtooths during smooth motion -> transform chain is the culprit
    //                                          (vmProj / cancellation not holding)
    //   ndc smooth                          -> transform OK, sawtooth is in bones v
    if (RtxOptions::useEngineHookMainCamera()) {
      const uint32_t zsFrame = m_device->getCurrentFrameId();
      static uint32_t s_zsLastFrame = UINT32_MAX;
      if (zsFrame != s_zsLastFrame) {
        s_zsLastFrame = zsFrame;
        const auto& mc =
          ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
        const Matrix4d eff =
          mc.getViewToProjection() * (mc.getWorldToView(false) * (perspectiveCorrection * refXform));
        // Object origin is the eye (view-local gun) → w≈0 there, useless. Instead
        // report where FORWARD points land: ndc.x(z) = (eff[2][0]*z + eff[3][0]) /
        // (eff[2][3]*z + eff[3][3]); as z→∞ this → eff[2][0]/eff[2][3], i.e. the
        // gun's horizontal screen anchor — independent of the unknown depth/scale.
        // If this ratio sawtooths during smooth motion, the gun sawtooths
        // horizontally and the cause is the transform chain, not the bones.
        const double c20 = eff[2][0], c21 = eff[2][1], c23 = eff[2][3]; // depth col: x,y,w
        Logger::info(str::format(
          "[ZigScreen] f=", zsFrame,
          " ndcX_fwd=", (c23 != 0.0 ? c20 / c23 : 0.0),
          " ndcY_fwd=", (c23 != 0.0 ? c21 / c23 : 0.0),
          " c20=", c20, " c21=", c21, " c23=", c23));
      }
    }

    // NV-DXVK [ZigVB]: ground-truth readback of the gun's final vertices, run
    // every frame regardless of teleport-vs-geometry path (unlike
    // dispatchViewModelCorrection which is gated by the BLAS-update). Also
    // reports whether the BLAS was rebuilt this frame.
    if (RtxOptions::useEngineHookMainCamera() && viewModelInstance->getBlas() != nullptr) {
      const uint32_t blasUpd =
        (viewModelInstance->getBlas()->frameLastUpdated == frameId) ? 1u : 0u;
      // [ZigNDC] pass the EXACT chain the ray tracer uses: the instance's final
      // objectToWorld (post-teleport) + the Main render camera's worldToView and
      // viewToProjection, so the readback can compute the gun's true on-screen NDC.
      const auto& ndcMainCam =
        ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
      ctx->getCommonObjects()->metaGeometryUtils().debugReadbackViewModelVerts(
        ctx, viewModelInstance->getBlas()->modifiedGeometryData, blasUpd,
        Matrix4(viewModelInstance->getTransform()),
        ndcMainCam.getWorldToView(false),
        ndcMainCam.getViewToProjection());
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
      // NV-DXVK [ZigGeo]: split the gun's ±1u horizontal wobble into its two
      // inputs. corrected (=T) = perspectiveCorrection * reference.getTransform().
      // Cameras are PROVEN stable ([ZigCam]: main/vm/engineEye all steady at
      // -25.60), so if T wobbles the noise is in ONE of these two:
      //   refT = reference.getTransform() = the gun's RAW per-draw objectToWorld
      //          (pure geometry — if THIS wobbles, the source is the draw's o2w).
      //   pcT  = perspectiveCorrection translation (camera-derived — if THIS
      //          wobbles despite stable cameras, the correction math is unstable).
      // Watch the .x of refT vs pcT across frames against T.x's ±1u sawtooth.
      const auto& refT = reference.getTransform();
      const auto& pcT  = perspectiveCorrection;
      Logger::info(str::format(
        "[VM.final] f=", frameId,
        " mask=0x", std::hex, (uint32_t)viewModelInstance->m_vkInstance.mask, std::dec,
        " pc=", (RtxOptions::ViewModel::perspectiveCorrection() ? 1 : 0),
        " T=(", t[3][0], ",", t[3][1], ",", t[3][2], ")",
        " refT=(", refT[3][0], ",", refT[3][1], ",", refT[3][2], ")",
        " pcT=(", pcT[3][0], ",", pcT[3][1], ",", pcT[3][2], ")",
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

    // [ZigGun] readback the REAL gun (tagged in updateInstance) through the same
    // viewDirect machinery used on the 3-vtx decoy. Runs first this frame so the
    // readback ring captures the GUN, not the viewmodel triangle. Tells us
    // whether the gun's geometry lags (viewDirect.x sawtooth) and its vertex space.
    // NV-DXVK: only consume the tag if it was set THIS frame. On a device-loss
    // frame the gun isn't drawn (updateInstance never retags) yet this code still
    // runs; the leftover tag points at a freed RtInstance -> use-after-free
    // (rax=0xDD AV at getBlas()->frameLastUpdated). The frame-id gate makes the
    // raw cross-frame pointer safe.
    if (s_zigGunInstance != nullptr && s_zigGunInstanceFrameId == fid && s_zigGunInstance->getBlas() != nullptr) {
      const auto& gunMain = cameraManager.getMainCamera();
      const uint32_t gunBlasUpd =
        (s_zigGunInstance->getBlas()->frameLastUpdated == fid) ? 1u : 0u;
      // NV-DXVK [ShipXform]: full o2w + the Main worldToView it's built from. The ship's verts
      // are static (o2wT=0) but in the vanish view direction a WRONG o2w teleports them behind the
      // camera (world (-199.582,-15364,10187.6), view Z +100). Dump the full matrices so we can
      // see whether the bad o2w is the rotation block being corrupted, a camera matrix leaking in,
      // or worldToView itself being wrong for that yaw — and trace the source.
      const Matrix4 zo2w = Matrix4(s_zigGunInstance->getTransform());
      const Matrix4 zw2v = gunMain.getWorldToView(false);
      // ========================================================================
      // !!! WARNING: s_zigGunInstance RE-TAGS BETWEEN DIFFERENT INSTANCES !!!
      // ------------------------------------------------------------------------
      // Do NOT trust [ZigNDC]/[ShipXform]/[ZigGunRB] world position as "one mesh
      // moving." s_zigGunInstance is set per-frame to whichever 0x292b draw was
      // tagged last, and across the "vanish" it FLIPS between distinct instances:
      // the 15817-vtx mesh (reads near-camera) and the 25537-vtx mesh (reads at
      // the fixed -199.582 anchor). posHash changes with the flip. So the dramatic
      // "world teleports behind camera" is largely a TAG-TRACKING ARTIFACT of this
      // probe, not a proven single-mesh teleport. (Earlier handoffs built a whole
      // engine-vs-Remix-bake theory on this probe — see the DEAD-END note in
      // rtx_geometry_utils.cpp interleaveGeometry: the skinning is verified CORRECT
      // (blend3 match=1). Don't chase bones/cb3/o2w via this probe again.)
      // To measure a real teleport, pin to ONE stable instance (filter by vtx +
      // a persistent key), don't consume the last-wins s_zigGunInstance tag.
      // ========================================================================
      // [ZigGunRB] instance-identity fields: inst ptr / vtx / posHash / blasUpd —
      //   posHash SAME across the jump  -> source object verts unchanged; the teleport is a
      //     transform/bake-application bug (Remix applies the wrong space when baking world).
      //   posHash CHANGES across the jump -> the source vertex data itself changed (engine
      //     re-uploaded the packed VB in a different space / pose).
      // blasUpd=1 means the BLAS was rebuilt THIS frame (modifiedGeometryData refreshed).
      const auto& zGeo = s_zigGunInstance->getBlas()->modifiedGeometryData;
      Logger::info(str::format(
        "[ZigGunRB] f=", fid,
        " inst=", (const void*)s_zigGunInstance,
        " vtx=", zGeo.vertexCount,
        " blasUpd=", gunBlasUpd,
        std::hex,
        " posHash=0x", zGeo.hashes[HashComponents::VertexPosition],
        " idxHash=0x", zGeo.hashes[HashComponents::Indices],
        std::dec,
        " o2wT=(", zo2w[3][0], ",", zo2w[3][1], ",", zo2w[3][2], ")"));
      Logger::info(str::format(
        "[ShipXform] f=", fid,
        " o2w_r0=(", zo2w[0][0], ",", zo2w[0][1], ",", zo2w[0][2], ",", zo2w[0][3], ")",
        " o2w_r1=(", zo2w[1][0], ",", zo2w[1][1], ",", zo2w[1][2], ",", zo2w[1][3], ")",
        " o2w_r2=(", zo2w[2][0], ",", zo2w[2][1], ",", zo2w[2][2], ",", zo2w[2][3], ")",
        " o2w_r3=(", zo2w[3][0], ",", zo2w[3][1], ",", zo2w[3][2], ",", zo2w[3][3], ")"));
      Logger::info(str::format(
        "[ShipXform] f=", fid,
        " w2v_r0=(", zw2v[0][0], ",", zw2v[0][1], ",", zw2v[0][2], ",", zw2v[0][3], ")",
        " w2v_r1=(", zw2v[1][0], ",", zw2v[1][1], ",", zw2v[1][2], ",", zw2v[1][3], ")",
        " w2v_r2=(", zw2v[2][0], ",", zw2v[2][1], ",", zw2v[2][2], ",", zw2v[2][3], ")",
        " w2v_r3=(", zw2v[3][0], ",", zw2v[3][1], ",", zw2v[3][2], ",", zw2v[3][3], ")"));
      ctx->getCommonObjects()->metaGeometryUtils().debugReadbackViewModelVerts(
        ctx, s_zigGunInstance->getBlas()->modifiedGeometryData, gunBlasUpd,
        Matrix4(s_zigGunInstance->getTransform()),
        gunMain.getWorldToView(false),
        gunMain.getViewToProjection());
    }
    s_zigGunInstance = nullptr;

    // NV-DXVK [HullWorldRB]: STABLE-INSTANCE world probe — the correct successor to the
    // re-tagging s_zigGunInstance/[ZigNDC] readback (see warning above). Enumerate ALL
    // big 0x292b instances this frame and batch a GPU readback of each one's vertex-0
    // world position, keyed by stable BLAS ptr. Diff across a visible->vanish pair: a
    // single blas key teleporting to -199.582 = REAL teleport (then chase BLAS-merge /
    // draw-call-cache); a near-cam key AND a -199.582 key BOTH present every frame =
    // the vanish was the tag artifact. Throttled to once per frame (heavy: waitForIdle).
    // Gated behind tf2HeavyProbes (default OFF): this readback does a per-frame
    // flush+waitForIdle — a primary Aftermath device-loss (freeze→crash) driver.
    if (RtxOptions::tf2HeavyProbes()) {
      static uint32_t s_hullRbLastFid = UINT32_MAX;
      if (fid != s_hullRbLastFid) {
        s_hullRbLastFid = fid;
        std::vector<RtxGeometryUtils::HullReadbackItem> items;
        for (const RtInstance* pInst : m_instances) {
          if (pInst == nullptr) continue;
          const BlasEntry* pBl = pInst->getBlas();
          if (pBl == nullptr) continue;
          // NV-DXVK [StudioModelHook] re-gate: enumerate the actual Widow
          // engine model (precise) instead of the shared VS hash 0x292b
          // (which also matched sky/world/weapon). Requires rtx.tf2DetectWidow
          // (or tf2HideWidow/tf2IsolateWidow). vertexCount floor dropped — all
          // Widow sub-meshes are interesting now, and the 16-item cap below
          // bounds the heavy readback.
          if (!pBl->input.isWidowModel) continue;
          const RaytraceGeometry& g = pBl->modifiedGeometryData;
          if (!g.positionBuffer.defined()) continue;
          RtxGeometryUtils::HullReadbackItem it;
          it.key     = static_cast<const void*>(pBl);
          it.vtx     = g.vertexCount;
          it.posHash = g.hashes[HashComponents::VertexPosition];
          it.matHash = static_cast<uint64_t>(pBl->input.getMaterialData().getHash());
          // NV-DXVK: same GC/streaming UAF as the HullCensus line ~963 —
          // isImageEmpty()→getImageHash() is not atomic vs the streaming
          // thread freeing m_currentMipView, so getImageHash() can deref a
          // dangling Rc<DxvkImage> (AV in Rc::operator->). This probe is
          // gated off by default (tf2HeavyProbes), but omit the image-hash
          // read here too so enabling it can't reintroduce the crash. mat
          // hash above is a plain value and is safe.
          it.texHash = 0ull;
          it.buffer  = g.positionBuffer.buffer();
          it.offset  = g.positionBuffer.offsetFromSlice();
          it.o2w     = Matrix4(pInst->getTransform());
          items.push_back(it);
          if (items.size() >= 16u) break;
        }
        if (!items.empty()) {
          ctx->getCommonObjects()->metaGeometryUtils().debugReadbackHullWorldPositions(ctx, fid, items);
        }
      }
    }

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
    // NV-DXVK [zig-zag fix pt2]: use the MAIN camera's (engine-locked, stable)
    // worldToView for the view part of the correction instead of the ViewModel
    // camera's. [ZigPC] proved vmW2V is stale (updates every ~3 frames, jumps
    // ~130u) while Main advances every frame; that fresh-vs-stale mismatch is
    // what makes pcT (and the gun) sawtooth ±60u. The viewmodel sits at the
    // SAME eye as Main (ZigCam/ZigPC: both at -25.60) so the view matrices
    // should be identical — only the projection (FOV/depth) legitimately
    // differs, and that is already handled by viewModelProjectionMatrix above.
    // Using the stable Main view makes the correction a stable similarity
    // transform (mainV2W * projRemap * mainW2V) instead of a stale-vs-fresh mix.
    const auto perspectiveCorrection = camera.getViewToWorld(false) * (camera.getProjectionToView() * viewModelProjectionMatrix * scaleMatrix) * camera.getWorldToView(false);
    const auto prevPerspectiveCorrection = camera.getPreviousViewToWorld(false) * (camera.getPreviousProjectionToView() * previousViewModelProjectionMatrix * scaleMatrix) * camera.getPreviousWorldToView(false);

    // NV-DXVK [ZigPC]: refT (gun o2w) is smooth after the engine-view fix, but
    // [VM.final] T still oscillates → the residual jerk is in perspectiveCorrection
    // itself (its translation pcT.y sawtooths ±63u). Isolate which of the 4
    // factors wobbles, per frame. Main (camera.*) is engine-locked → mV2W_T /
    // mP2V_XY expected steady. The ViewModel camera is NOT engine-suppressed, so
    // its worldToView (vmW2V_T) and projection (vmProjXY) are the suspects.
    {
      const uint32_t pcFrame = m_device->getCurrentFrameId();
      static uint32_t s_pcLastFrame = UINT32_MAX;
      if (pcFrame != s_pcLastFrame) {
        s_pcLastFrame = pcFrame;
        const auto mV2W  = camera.getViewToWorld(false);
        const auto vmW2V = viewModelCamera.getWorldToView(false);
        const auto mP2V  = camera.getProjectionToView();
        Logger::info(str::format(
          "[ZigPC] f=", pcFrame,
          " mV2W_T=(", mV2W[3][0], ",", mV2W[3][1], ",", mV2W[3][2], ")",
          " vmW2V_T=(", vmW2V[3][0], ",", vmW2V[3][1], ",", vmW2V[3][2], ")",
          " vmProjXY=(", viewModelProjectionMatrix[0][0], ",", viewModelProjectionMatrix[1][1], ")",
          " mP2V_XY=(", mP2V[0][0], ",", mP2V[1][1], ")",
          " pcT=(", perspectiveCorrection[3][0], ",", perspectiveCorrection[3][1], ",", perspectiveCorrection[3][2], ")"));
      }
    }

    // Create any valid view model instances from the list of candidates
    std::vector<RtInstance*> viewModelInstances;
    uint32_t vmSkippedMulticam = 0, vmSkippedNoCam = 0, vmCreated = 0;
    for (auto* candidateInstance : m_viewModelCandidates) {

      // Valid view model instances must be associated only with the view model camera
      // Check: exactly one bit set (power-of-two check via raw bitmask)
      const auto seenMask = candidateInstance->m_seenCameraTypes.raw();
      // [ZigCand] every candidate: its BLAS, vtx count, seenMask, current mask,
      // and whether it'll be skipped. Match blas/vtx against [ZigInst]/[ZigBlas]
      // to find whether the VISIBLE gun is a candidate we actually correct, or a
      // skipped/other instance. Also dumps the reference's pre-hide mask.
      {
        const void* candBlas = (const void*)candidateInstance->getBlas();
        const uint32_t candVtx = candidateInstance->getBlas() ? candidateInstance->getBlas()->modifiedGeometryData.vertexCount : 0u;
        Logger::info(str::format(
          "[ZigCand] f=", fid,
          " cand=", (const void*)candidateInstance,
          " blas=", candBlas, " vtx=", candVtx,
          " seenMask=0x", std::hex, (uint32_t)seenMask,
          " mask=0x", (uint32_t)candidateInstance->m_vkInstance.mask, std::dec,
          " skip=", (seenMask == 0 ? "noCam" : ((seenMask & (seenMask - 1)) != 0 ? "multiCam" : "no"))));
      }
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
