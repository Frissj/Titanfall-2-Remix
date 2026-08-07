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
#pragma once

#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "../util/rc/util_rc_ptr.h"
#include "rtx_types.h"
#include "../util/util_vector.h"
#include "../util/util_flags.h"
#include "../util/util_matrix.h"
#include "rtx_camera_manager.h"
#include "dxvk_cmdlist.h"
#include "rtx_opacity_micromap_manager.h"

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
class ResourceCache;
class CameraManager;
class DrawCallCache;

// RtInstance defines a SceneObjects placement/parameterization within the current scene.
class RtInstance {
public:
  RtSurface surface;

  // NV-DXVK [FirstBakeHold — flicker fix]: when a relink migrates this
  // instance onto a BlasEntry whose FIRST bake ran while the engine's source
  // upload was still in flight (RaytraceGeometry::pendingSrcBake — the s2s
  // FIX B condition), the new BLAS content is collapsed/garbage for that
  // frame and the object blinks out for exactly one frame (the TF2 geometry
  // flicker: census dropouts sit on re-batch transitions, "objects don't
  // appear on the first frame"). The relink site stashes the FROM-entry's
  // built BLAS here; AccelManager stamps this reference into the TLAS instead
  // of the pending one until the destination bake lands (pendingSrcBake
  // clears), then releases it. Renders one frame of the previous geometry
  // instead of one frame of nothing. The Rc keeps the old PooledBlas alive
  // for the handover; nulled by the stamp site, GC-safe.
  Rc<PooledBlas> m_prevBlasKeepAlive;

  RtInstance() = delete;
  RtInstance(const uint64_t id, uint32_t instanceVectorId);
  RtInstance(const RtInstance& src, uint64_t id, uint32_t instanceVectorId);

  uint64_t getId() const { return m_id; }
  uint32_t getVectorIdx() const { return m_instanceVectorId; }
  const VkAccelerationStructureInstanceKHR& getVkInstance() const { return m_vkInstance; }
  VkAccelerationStructureInstanceKHR& getVkInstance() { return m_vkInstance; }
  bool isObjectToWorldMirrored() const { return m_isObjectToWorldMirrored; }

  BlasEntry* getBlas() const { return m_linkedBlas; }
  const XXH64_hash_t& getMaterialHash() const { return m_materialHash; }
  const XXH64_hash_t& getMaterialDataHash() const { return m_materialDataHash; }
  const XXH64_hash_t& getTexcoordHash() const { return m_texcoordHash; }
  const XXH64_hash_t& getIndexHash() const { return m_indexHash; }
  const XXH64_hash_t calculateAntiCullingHash() const;
  Matrix4 getTransform() const { return transpose(dxvk::Matrix4(m_vkInstance.transform)); }
  const Matrix4& getPrevTransform() const { return surface.prevObjectToWorld; }
  Vector3 getWorldPosition() const { return Vector3{ m_vkInstance.transform.matrix[0][3], m_vkInstance.transform.matrix[1][3], m_vkInstance.transform.matrix[2][3] }; }
  const Vector3& getPrevWorldPosition() const { return surface.prevObjectToWorld.data[3].xyz(); }

  void removeFromSpatialCache() {
    if (m_isCreatedByRenderer || !m_linkedBlas || m_isUnlinkedForGC || m_spatialCacheHash == kEmptyHash) {
      return;
    }
    m_linkedBlas->getSpatialMap().erase(m_spatialCacheHash);
    m_spatialCacheHash = kEmptyHash;
  }

  bool isCreatedThisFrame(uint32_t frameIndex) const { return frameIndex == m_frameCreated; }

  // Bind a BLAS object to this instance
  void setBlas(BlasEntry& blas);

  // Sets current and previous transforms explicitly
  bool teleport(const Matrix4& objectToWorld);
  bool teleport(const Matrix4& objectToWorld, const Matrix4& prevObjectToWorld);
  // Changes all transform data from an old context to a new context (i.e. when an instance moves through a portal).
  void teleportWithHistory(const Matrix4& oldToNew);
  
  // Move to the new transform and retain previous transforms as history (call the first time a transform changes per frame)
  bool move(const Matrix4& objectToWorld);
  // Move to the new transform without changing history (call if the transform is changed multiple times per frame)
  bool moveAgain(const Matrix4& objectToWorld);

  void setFrameCreated(const uint32_t frameIndex);
  // Returns if this is the first occurence in a given frame
  bool setFrameLastUpdated(const uint32_t frameIndex);
  uint32_t getFrameLastUpdated() const { return m_frameLastUpdated; } 
  uint32_t getFrameAge() const { return m_frameLastUpdated - m_frameCreated; }
  // Signal this object should be collected on the next GC pass
  void markForGarbageCollection() const;
  void markAsUnlinkedFromBlasEntryForGarbageCollection() const;
  void markAsInsideFrustum() const;
  void markAsOutsideFrustum() const;
  // Returns true if a new camera type was registered
  bool registerCamera(CameraType::Enum cameraType, uint32_t frameIndex);
  bool isCameraRegistered(CameraType::Enum cameraType) const;
  void setCustomIndexBit(uint32_t oneBitMask, bool value);
  bool getCustomIndexBit(uint32_t oneBitMask) const;
  bool isHidden() const { return m_isHidden; }
  void setHidden(bool value) { m_isHidden = value; }

  bool usesUnorderedApproximations() const { return m_isUnordered; }
  MaterialDataType getMaterialType() const {
    return m_materialType;
  }
  bool isOpaque() const;

  uint32_t getAlbedoOpacityTextureIndex() const { return m_albedoOpacityTextureIndex; }
  uint32_t getSamplerIndex() const { return m_samplerIndex; }
  uint32_t getSecondaryOpacityTextureIndex() const { return m_secondaryOpacityTextureIndex; }
  uint32_t getSecondarySamplerIndex() const { return m_secondarySamplerIndex; }

  bool isAnimated() const {
    return m_isAnimated;
  }
  void setSurfaceIndex(uint32_t surfaceIndex) {
    m_surfaceIndex = surfaceIndex;
  }
  uint32_t getSurfaceIndex() const {
    return m_surfaceIndex;
  }

  // NV-DXVK [MapWrite census]: "is this instance putting pixels on screen?",
  // as closely as can be answered on the CPU without a GPU readback.
  //
  //   !m_isHidden          - not explicitly suppressed
  //   m_isInsideFrustum    - survives frustum culling
  //   mask != 0            - actually included in a ray-tracing pass; a zero
  //                          instance mask is traced by nothing
  //   surfaceIndex valid   - has a surface the tracer can shade
  //
  // NOT occlusion-aware: an instance entirely behind a wall still reports true,
  // so this is an upper bound on what is visible. Use it to enumerate scene
  // candidates, not to prove a specific object was seen.
  // NOT usable as a gate at SpatialMap-write time: m_surfaceIndex and the
  // instance mask are assigned later in the frame by AccelManager, so both are
  // stale-or-unset while onTransformChanged()/teleport() run. Gating the census
  // on this produced zero lines. Kept for callers that run AFTER accel build.
  // SURFACE_INDEX_INVALID is 0x001FFFFF, not BINDING_INDEX_INVALID's 0xFFFF.
  bool isOnScreen() const {
    return !m_isHidden
        && m_isInsideFrustum
        && m_vkInstance.mask != 0
        && m_surfaceIndex != SURFACE_INDEX_INVALID;
  }

  // Visibility state as it stands AT WRITE TIME. hidden/frustum are already
  // meaningful there; mask and surfaceIndex are last frame's values. Logged as
  // fields rather than used as a gate so the census can be filtered offline
  // without silently dropping objects.
  bool censusHidden() const { return m_isHidden; }
  bool censusInFrustum() const { return m_isInsideFrustum; }
  uint32_t censusMask() const { return m_vkInstance.mask; }
  uint32_t censusSurfaceIndex() const { return m_surfaceIndex; }
  // NV-DXVK [SurfaceIndexStability]: public getter so AccelManager can
  // sort instances by stablePropId before assigning surfaceIndex. With
  // GC's swap+pop_back removal pattern, the m_instances iteration order
  // shifts every frame as instances retire — and surfaceIndex is just
  // iteration position. Sub-view content with stablePropId set should
  // keep the SAME surfaceIndex across frames as long as the set of
  // propId-tagged instances is stable; sorting by propId before the
  // surfaceIndex-assignment loop achieves that.
  uint64_t getStablePropId() const { return m_stablePropId; }
  void setPreviousSurfaceIndex(uint32_t surfaceIndex) {
    m_previousSurfaceIndex = surfaceIndex;
  }
  uint32_t getPreviousSurfaceIndex() const {
    return m_previousSurfaceIndex;
  }
  // NV-DXVK: how many CONSECUTIVE surface slots this instance owned last
  // frame, so the previous->current mapping can cover all of them.
  //
  // Normal instances own exactly one slot and this stays 1. A PointInstancer
  // instance reserves N (accel_manager: m_reorderedSurfaces.insert(end,
  // instanceCount, rtInstance)) and the GPU culling shader writes
  // surfaceBuffer[baseSurfaceIndex + instanceIdx] for each. Before this
  // existed, only the BASE slot was ever entered into surfaceIndexMapping —
  // the remap loop gates on getSurfaceIndex()==SURFACE_INDEX_INVALID, which
  // is true only for the first of the N duplicate entries, and the general
  // branch below it explicitly excludes PI content. So a 259-instance
  // PointInstancer contributed 17 mapping entries for 259 slots and the other
  // 93% had no temporal identity at all.
  //
  // That only became visible once [PIWatch] measured baseSurfaceIndex moving
  // on 67-86 of 86 batches EVERY frame (0 quiet frames in 2711): surfaceIndex
  // is just position in m_reorderedSurfaces, and instance GC reshuffles that
  // order continuously. Unmapped slots therefore resolve to whatever object
  // now occupies that absolute index.
  void setPreviousSurfaceCount(uint32_t count) {
    m_previousSurfaceCount = count;
  }
  uint32_t getPreviousSurfaceCount() const {
    return m_previousSurfaceCount;
  }
  OpacityMicromapInstanceData& getOpacityMicromapInstanceData() { return m_opacityMicromapInstanceData; }
  const OpacityMicromapInstanceData& getOpacityMicromapInstanceData() const { return m_opacityMicromapInstanceData; }

uint32_t getFirstBillboardIndex() const { return m_firstBillboard; }
  uint32_t getBillboardCount() const { return m_billboardCount; }

  VkGeometryFlagsKHR getGeometryFlags() const { return m_geometryFlags; }

  template<typename... InstanceCategories>
  bool testCategoryFlags(InstanceCategories... cat) const { return m_categoryFlags.any(cat...); }
  CategoryFlags getCategoryFlags() const { return m_categoryFlags; }

  bool isViewModel() const;
  bool isViewModelNonReference() const;
  bool isViewModelReference() const;
  bool isViewModelVirtual() const;
  bool isSubsurface() const { return m_isSubsurface; }

  bool isUnlinkedForGC() const { return m_isUnlinkedForGC; }

  PrimInstanceOwner& getPrimInstanceOwner() { return m_primInstanceOwner; }
  
  void printDebugInfo() const;

private:

  Matrix4 calcFirstInstanceObjectToWorld() {
    if (surface.instancesToObject) {
      return surface.objectToWorld * (*surface.instancesToObject)[0]; 
    }
    return surface.objectToWorld;
  }
  // NV-DXVK [perf] handoff v5 sec 4b(a): objectToWorldChanged lets a caller that
  // has already proven surface.objectToWorld is byte-identical skip the
  // transpose + 48-byte memcpy into m_vkInstance.transform. move()/moveAgain()
  // compute exactly that memcmp for their own return value, so the test is free.
  // Defaults true: a caller that cannot prove it gets the old behaviour.
  void onTransformChanged(bool objectToWorldChanged = true);
  friend class InstanceManager;

  // Unique ID of the RtInstance.
  // Sentinel value UINT64_MAX indicates that such RtInstance is a "virtual" instance, and is ignored by some features,
  // most notably the GameCapturer
  const uint64_t m_id;
  mutable uint32_t m_instanceVectorId; // Index within instance vector in instance manager

  mutable bool m_isMarkedForGC = false;
  mutable bool m_isUnlinkedForGC = false;
  mutable bool m_isInsideFrustum = true;
  mutable uint32_t m_frameLastUpdated = kInvalidFrameIndex;
  mutable uint32_t m_frameCreated = kInvalidFrameIndex;

  Flags<CameraType::Enum> m_seenCameraTypes;  // Camera types with which the instance has been originally rendered with

  MaterialDataType m_materialType = MaterialDataType::Invalid;
  uint32_t m_albedoOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_samplerIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_secondaryOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_secondarySamplerIndex = kSurfaceMaterialInvalidTextureIndex;

  // Extra instance meta data needed for Opacity Micromap Manager, generally describes if animated spritesheets are in use
  // on a given instance (though the applicability to OMMs are only relevant for Opaque and Ray Portal materials currently
  // where cutout opacity can be animated, translucent materials do not have any relation right now to OMMs).
  bool m_isAnimated = false;
  // Object with Opacity Micromap per-instance data maintained by Opacity Micromap Manager.
  // Stored in instance object to avoid indirection of looking it up for an instance
  OpacityMicromapInstanceData m_opacityMicromapInstanceData;

  uint32_t m_surfaceIndex;        // Material surface index for reordered surfaces by AccelManager
  uint32_t m_previousSurfaceIndex;
  // Consecutive slots owned last frame; 1 for everything except PointInstancer
  // instances, which own instanceCount. See setPreviousSurfaceCount.
  uint32_t m_previousSurfaceCount = 1;

  bool m_isHidden = false;
  bool m_isPlayerModel = false;
  bool m_isWorldSpaceUI = false;
  bool m_isUnordered = false;
  bool m_isObjectToWorldMirrored = false;
  bool m_isCreatedByRenderer = false;
  bool m_isSubsurface = false;
  BlasEntry* m_linkedBlas = nullptr;
  XXH64_hash_t m_materialHash = kEmptyHash;
  XXH64_hash_t m_materialDataHash = kEmptyHash;
  XXH64_hash_t m_texcoordHash = kEmptyHash;
  XXH64_hash_t m_indexHash = kEmptyHash;
  VkAccelerationStructureInstanceKHR m_vkInstance;
  VkGeometryFlagsKHR m_geometryFlags = 0;
  uint32_t m_firstBillboard = 0;
  uint32_t m_billboardCount = 0;

  CategoryFlags m_categoryFlags;

  XXH64_hash_t m_spatialCacheHash = kEmptyHash;

  // NV-DXVK [Stable prop ID]: when non-zero, used as the SpatialMap cache
  // key (via the overrideHash param to insert/move/erase) instead of
  // XXH64(matrix). Anchors dedup identity to the draw's per-prop ID
  // rather than per-frame transform bytes, eliminating drift-induced
  // dedup misses for sub-view-reprojected content. Mirrored from
  // DrawCallState.transformData.stablePropId at instance creation /
  // updateInstance so onTransformChanged + teleport (which don't have
  // a drawCall reference) can use the same key.
  uint64_t m_stablePropId = 0;

  // This can be used to access all lights and instances that originate from the same draw call.
  // Left as nullptr if the draw call does not have replacement data.
  PrimInstanceOwner m_primInstanceOwner;

public:
  bool isFrontFaceFlipped = false;

  std::vector<uint32_t> billboardIndices;
  std::vector<uint32_t> indexOffsets;
};

// Optional notification callbacks that can be implemented to "opt-in" to InstanceManager events
struct InstanceEventHandler {
  void* eventHandlerOwnerAddress;

  // Callback triggered whenever a new instance has been added to the database
  std::function<void(RtInstance&)> onInstanceAddedCallback;
  // Callback triggered whenever instance metadata is updated - the boolean flags 
  //   signal if the transform and/or vertex positions have changed (respectively)
  std::function<void(RtInstance&, const DrawCallState& drawCall, const MaterialData&, bool, bool, bool)> onInstanceUpdatedCallback;
  // Callback triggered whenever an instance has been removed from the database
  std::function<void(RtInstance&)> onInstanceDestroyedCallback;

  InstanceEventHandler() = delete;
  InstanceEventHandler(void* _eventHandlerOwnerAddress) : eventHandlerOwnerAddress(_eventHandlerOwnerAddress) { }
};

struct IntersectionBillboard {
  Vector3 center;
  Vector3 xAxis;
  float width;
  Vector3 yAxis;
  float height;
  Vector2 xAxisUV;
  Vector2 yAxisUV;
  Vector2 centerUV;
  uint32_t vertexColor;
  uint32_t instanceMask;
  const RtInstance* instance;
  XXH64_hash_t texCoordHash;
  XXH64_hash_t vertexOpacityHash;
  bool allowAsIntersectionPrimitive;
  bool isBeam; // if true, the billboard's Y axis is fixed and the billboard is free to rotate around it
  bool isCameraFacing; // if true, the billboard should always orient the normal toward the camera, don't use the transform matrix
};

// InstanceManager is responsible for maintaining the active set of scene instances
//  and the GPU buffers which are required by VK for instancing.
class InstanceManager : public CommonDeviceObject {
public:
  InstanceManager(InstanceManager const&) = delete;
  InstanceManager& operator=(InstanceManager const&) = delete;

  InstanceManager(DxvkDevice* device, ResourceCache* pResourceCache);
  ~InstanceManager();

  // Return a list of instances currently active in the scene
  const std::vector<RtInstance*>& getInstanceTable() const { return m_instances; }

  // Returns the active number of instances in scene
  const uint32_t getActiveCount() const { return m_instances.size(); }
  
  void onFrameEnd();

  // Optional notification callbacks that can be implemented to "opt-in" to InstanceManager events
  void addEventHandler(const InstanceEventHandler& events) {
    m_eventHandlers.push_back(events);
  }
  
  void removeEventHandler(void* eventHandlerOwnerAddress);

  // Clear all instances currently tracked by manager
  void clear();

  // Clean up instances which are deemed as no longer required
  void garbageCollection();
  
  // Takes a scene object entry (blas + drawcall) and generates/finds the instance data internally
  // NV-DXVK [MatBind identity]: drawCallCache (optional) enables cross-entry
  // instance relink for engine-class siblings — see findSimilarInstance.
  RtInstance* processSceneObject(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache = nullptr);

  // NV-DXVK [fanout split]: same as processSceneObject, but for a draw whose
  // transformData carries isFanoutBatch — it resolves ONE RtInstance PER ELEMENT
  // of instancesToObject rather than one for the whole batch, and appends them to
  // out_instances in batch order.
  //
  // The caller gets every instance rather than a "representative" one on purpose:
  // effect lights, object picking and particle emission all key off the returned
  // instance, and silently applying them to only the first prop of a ~54-prop
  // batch would be a real (and invisible) regression. out_instances is cleared
  // first; it can end up shorter than the batch if two placements resolve to the
  // same RtInstance (see the collide= field of the [FanoutSplit] log — that means
  // two props share a propId and are merging, which is a defect, not a saving).
  void processSceneObjectFanout(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    DrawCallCache* drawCallCache, std::vector<RtInstance*>& out_instances);

  // Binds a raytracing material to the specified instance.
  // NV-DXVK [perf]: indexInCache lets the caller skip the resource-cache lookup
  // below when it already knows the answer -- SceneManager::createSurfaceMaterial
  // returns it through out_indexInCache. UINT32_MAX keeps the old behaviour for
  // any caller that cannot supply one.
  void bindMaterial(RtInstance& instance, const RtSurfaceMaterial& material, uint32_t indexInCache = UINT32_MAX);

  // Creates a copy of a reference instance and adds it to the instance pool
  // Temporary single frame instances generated every frame should disable valid id generation to avoid overflowing it
  RtInstance* createInstanceCopy(const RtInstance& reference, bool generateValidID = true);

  // Creates a view model instance from the reference and adds it to the instance pool
  RtInstance* createViewModelInstance(Rc<DxvkContext> ctx, const RtInstance& reference, const Matrix4d& perspectiveCorrection, const Matrix4d& prevPerspectiveCorrection);

  // Creates view model instances and their virtual counterparts
  void createViewModelInstances(Rc<DxvkContext> ctx, const CameraManager& cameraManager, const RayPortalManager& rayPortalManager);

  void createPlayerModelVirtualInstances(Rc<DxvkContext> ctx, const CameraManager& cameraManager, const RayPortalManager& rayPortalManager);

  void findPortalForVirtualInstances(const CameraManager& cameraManager, const RayPortalManager& rayPortalManager);

  int getVirtualInstancePortalIndex() const { return m_virtualInstancePortalIndex; }
    
  // Creates ray portal virtual instances for viewModel instances for a closest portal within range
  void createRayPortalVirtualViewModelInstances(const std::vector<RtInstance*>& viewModelInstances, const CameraManager& cameraManager, const RayPortalManager& rayPortalManager);

  void resetSurfaceIndices();

  const std::vector<IntersectionBillboard>& getBillboards() const { return m_billboards; }
  
private:
  // NV-DXVK [fanout split]: the per-prop overrides that turn one element of a
  // fanout batch into an ordinary, independently-identified scene object. When a
  // FanoutSplit* is non-null every place that would otherwise read the draw
  // call's batch-wide transform identity reads this instead; when it is null the
  // code path is bit-identical to the pre-split behaviour.
  struct FanoutSplit {
    // drawCall.objectToWorld * instancesToObject[i]. This is exactly the matrix
    // RtSurface::writeGPUData would have derived for slot i of the point
    // instancer, so the rendered placement is unchanged by the split.
    //
    Matrix4 objectToWorld;

    // Always 0 for a split fanout placement, which selects the SpatialMap's
    // composed-matrix-bytes key. The field is kept because processSceneObjectImpl
    // and updateInstance share their bodies with the non-split path, where a draw
    // call can legitimately carry a stablePropId; a split placement must never
    // inherit the BATCH's propId, and 0 here is what prevents that.
    //
    // An engine handle was tried here and removed: charIdx indexes a PER-DRAW
    // scratch buffer of 2..96 entries, so it is an array position, not a prop
    // name, and it merged unrelated props across draws. Motion is handled by
    // prevObjectToWorld below instead — see the note above
    // InstanceManager::processSceneObjectFanout.
    uint64_t stablePropId = 0;

    // NV-DXVK [fanout prev-transform identity] 2026-08-05: the composed matrix
    // this placement had LAST frame, from the engine's own per-instance history
    // (DrawCallTransforms::prevInstancesToObject).
    //
    // An instance is filed in the SpatialMap under the hash of its transform at
    // the time it was inserted. For a prop that moved, this frame's transform is
    // not that hash — but last frame's is, exactly. So this is a second key to
    // try on the exact stage, and it succeeds for movers without any distance
    // tolerance, which is what makes it different from every prior candidate.
    //
    // Equal to objectToWorld when the engine offered no history (a prop's first
    // frame). The probe then degenerates to a repeat of the current-transform
    // probe, which is harmless and needs no special case.
    Matrix4 prevObjectToWorld;
    bool hasPrevObjectToWorld = false;
  };

  // NV-DXVK [fanout prev-transform identity]: how often the engine's history
  // resolved a placement that its current transform could not. This is the claim
  // the whole plumb rests on, so it is counted rather than assumed:
  //   hit ~= (nInst - mtxStable), miss ~= 0  -> the history is bit-exact and the
  //     movers now resolve on the exact stage instead of the nearest search.
  //   hit ~= 0 while created/missProp persist -> the history is NOT reproducing
  //     last frame's key; the plumb is dead weight and should be said to be.
  // Atomic because findSimilarInstance is reachable from the scene-manager's
  // draw-processing threads, and a torn counter would misreport the verdict.
  std::atomic<uint32_t> m_fanoutPrevHitCount { 0 };
  std::atomic<uint32_t> m_fanoutPrevMissCount { 0 };

  ResourceCache* m_pResourceCache;

  // Start at 1 to avoid using 0 - makes it easier to detect a 0 initialized RtInstance (which is invalid)
  uint64_t m_nextInstanceId = 1;

  std::vector<RtInstance*> m_instances; 
  std::vector<RtInstance*> m_viewModelCandidates;
  uint32_t m_viewModelCandidatesFrameId = kInvalidFrameIndex;
  std::vector<RtInstance*> m_playerModelInstances;
  uint32_t m_playerModelInstancesFrameId = kInvalidFrameIndex;
  std::vector<IntersectionBillboard> m_billboards;

  bool m_previousViewModelState = false;
  RtInstance* targetInstance = nullptr;

  uint32_t m_decalSortOrderCounter = 0;  // monotonically incrementing value indicating the draw call order of this decal on the frame

  // Controls active portal space for which virtual view model or player model instances have been generated for.
  // Negative values mean there is no portal that's close enough to the camera.
  int m_virtualInstancePortalIndex = 0;    

  std::vector<InstanceEventHandler> m_eventHandlers;

  // Handles the case of when two (or more) identical geometries+textures draw calls have been submitted in a single frame (typically used for two-pass rendering in FF)
  void mergeInstanceHeuristics(RtInstance& instanceToModify, const DrawCallState& drawCall, const RtSurface::AlphaState& alphaState) const;

  // Finds the "closest" matching instance to a set of inputs, returns a pointer (can be null if not found) to closest instance
  // stablePropId: if non-zero, used as the SpatialMap cache key instead
  // of XXH64(matrix). Anchors dedup to per-prop identity. Default 0
  // preserves the original matrix-hash behavior. Source is the current
  // drawCall's transformData.stablePropId.
  // prevObjectToWorld: if non-null, where this object stood LAST frame. Tried as
  // a SECOND exact-stage key when the current transform misses, which resolves a
  // moved object to the instance it was filed under last frame without involving
  // the nearest-neighbour search at all. Null preserves the original behaviour.
  RtInstance* findSimilarInstance(BlasEntry& blas, const MaterialData& material, const Matrix4& firstInstanceObjectToWorld, CameraType::Enum cameraType, const RayPortalManager& rayPortalManager, uint64_t stablePropId = 0, DrawCallCache* drawCallCache = nullptr, const Matrix4* prevObjectToWorld = nullptr);

  // Shared body of processSceneObject / processSceneObjectFanout. split is null
  // for an ordinary draw and non-null for one placement of a split fanout batch.
  RtInstance* processSceneObjectImpl(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache, const FanoutSplit* split);

  RtInstance* addInstance(BlasEntry& blas);
  void processInstanceBuffers(const BlasEntry& blas, RtInstance& currentInstance) const;

  void updateInstance(
    RtInstance& currentInstance, const CameraManager& cameraManager,
    const BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    const FanoutSplit* split = nullptr);

  void removeInstance(RtInstance* instance);

  static RtSurface::AlphaState calculateAlphaState(const DrawCallState& drawCall, const MaterialData& materialData);

  // Modifies an instance given active developer options. Returns true if the instance was modified
  bool applyDeveloperOptions(RtInstance& currentInstance, const DrawCallState& drawCall);

  void createBillboards(RtInstance& instance, const Vector3& cameraViewDirection);

  void createBeams(RtInstance& instance);

  void filterPlayerModelInstances(const Vector3& playerModelPosition, const RtInstance* bodyInstance);

  void detectIfPlayerModelIsVirtual(
    const CameraManager& cameraManager,
    const RayPortalManager& rayPortalManager,
    const Vector3& playerModelPosition,
    bool* out_PlayerModelIsVirtual,
    const struct SingleRayPortalDirectionInfo** out_NearPortalInfo,
    const struct SingleRayPortalDirectionInfo** out_FarPortalInfo) const;
};

}  // namespace dxvk

