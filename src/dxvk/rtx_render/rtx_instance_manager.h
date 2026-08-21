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
// NV-DXVK [ResidentScene]: forward-declares RtInstance only, so this include
// cannot cycle back on us even though ResidentScene's .cpp includes this file.
#include "rtx_resident_scene.h"

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
class ResourceCache;
class CameraManager;
class DrawCallCache;

// NV-DXVK [perf] handoff v7 sec 4a: a SpatialMap key that has already been paid
// for this frame, carried from the lookup that computed it to the write that
// would otherwise compute it again.
//
// THE REDUNDANCY. findSimilarInstance hashes the composed object-to-world matrix
// to do its exact lookup. A few hundred lines later the same instance reaches
// onTransformChanged, which hashes a matrix again to decide whether its spatial
// cache entry has to be re-filed. [MapGate] measures 15,447 of those writes per
// frame against 15,488 lookups, and for a static prop -- 97% of the population --
// the two matrices are the same 64 bytes. That is ~15,400 redundant XXH64s a
// frame, on the thread that owns the frame.
//
// WHY THE MATRIX POINTER IS HERE TOO, and why this is not just an XXH64_hash_t.
// The precondition for reusing the hash is that the bytes hashed are the bytes
// that would have been hashed, and that is NOT structurally guaranteed:
//
//   - the lookup keys on drawCall.getTransformData().calcFirstInstanceObjectToWorld(),
//     which checks instancesToObject->empty(); RtInstance::calcFirstInstanceObjectToWorld
//     does not, so the two can compose differently
//   - updateInstance mutates its local copy of objectToWorld for WorldMatte
//     instances (rtx.worldSpaceUiBackgroundOffset) after the lookup has run
//   - a split placement, a replacement root and an ordinary draw each reach the
//     write through a different transform
//
// So the consumer memcmps `matrix` against the matrix it is actually about to
// hash and falls back to hashing when they differ. A 64-byte memcmp is a couple
// of SIMD compares; XXH64 over the same 64 bytes is several times that, and the
// compare is what makes the reuse provable instead of assumed.
//
// A default-constructed hint (hash == 0) means "nothing precomputed" and
// restores the original behaviour exactly -- which is also what an instance that
// never went through findSimilarInstance gets.
struct SpatialKeyHint {
  // The matrix `hash` was computed over. Must outlive the call it is passed to;
  // in practice it is a local of the caller further up the same stack.
  const Matrix4* matrix = nullptr;
  // XXH64 of *matrix, or 0 when unavailable. Never a propId override key --
  // see getDataAtTransform's outMatrixHash contract.
  XXH64_hash_t hash = 0;

  bool isUsable() const { return hash != 0 && matrix != nullptr; }
};

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

  // NV-DXVK [Perf.CullAabbCache] 2026-08-08g: cached world AABB for
  // AccelManager's SceneCull. The world box is a pure function of
  // (vkInstance.transform bits, object-space box); the cull re-derived it
  // with 8 Matrix4*Vector4 corner transforms per instance per frame
  // (~8.9k tested instances -> ~71k transforms/frame in the merge loop)
  // even though most instances are static. Keyed on the RAW 48-byte
  // transform bits, so a hit also skips getTransform()'s per-call
  // transpose. Written and read only by the SceneCull block (CS thread).
  struct CullAabbCache {
    float   xform[3][4];      // VkTransformMatrixKHR bits at capture
    Vector3 boxMin, boxMax;   // object box at capture
    Vector3 lo, hi;           // cached world AABB
    bool    valid = false;
  };
  CullAabbCache cullAabbCache;

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
  // keyHint: see SpatialKeyHint. Purely an optimisation; omitting it is correct.
  bool move(const Matrix4& objectToWorld, const SpatialKeyHint& keyHint = SpatialKeyHint());
  // Move to the new transform without changing history (call if the transform is changed multiple times per frame)
  bool moveAgain(const Matrix4& objectToWorld, const SpatialKeyHint& keyHint = SpatialKeyHint());

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
  // The whole set at once, for ResidentScene::build to capture and
  // ResidentScene::touch to replay -- setFrameLastUpdated() clears it on the
  // first stamp of a frame, so a keep-alive that does not put it back loses it.
  uint32_t getSeenCameraMask() const { return m_seenCameraTypes.raw(); }
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
  void onTransformChanged(bool objectToWorldChanged = true,
                          const SpatialKeyHint& keyHint = SpatialKeyHint());
  friend class InstanceManager;
  // NV-DXVK [ResidentScene]: reads and clears m_residentKey to maintain the
  // back-pointer contract documented on that field. Same access InstanceManager
  // needs for m_batchRecordKey, and for the same reason.
  friend class ResidentScene;

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

  // NV-DXVK [Perf.PushInst] PHASE 2: which fanout batch record, if any, is
  // currently holding a raw pointer to this instance. 0 means none.
  //
  // THIS FIELD IS THE LIFETIME CONTRACT, not a convenience. A batch record
  // caches RtInstance* across frames, and GC deletes instances -- so without a
  // back-pointer the record is a dangling-pointer generator, which is the exact
  // failure class this file has already shipped twice (the file-static
  // s_zigGunInstance deref and the GC-walk incRef race). Carrying the key here
  // makes removeInstance's invalidation O(1) and, more importantly, TOTAL: an
  // instance cannot be destroyed without the record that references it being
  // invalidated in the same call, because the destroy path has to come through
  // removeInstance to reach the event handlers at all.
  mutable uint64_t m_batchRecordKey = 0ull;

  // NV-DXVK [ResidentScene]: which resident record, if any, currently holds a
  // raw pointer to this instance. 0 means none. See rtx_resident_scene.h.
  //
  // SAME LIFETIME CONTRACT AS m_batchRecordKey ABOVE, AND FOR THE SAME REASON:
  // a resident record caches RtInstance* across an UNBOUNDED number of frames
  // (that is the entire point of residency), so without a back-pointer it is a
  // dangling-pointer generator -- the failure class this file has already
  // shipped twice, in the file-static s_zigGunInstance deref and the GC-walk
  // incRef race. Carrying the key here makes ResidentScene::invalidateFor()
  // O(1) and TOTAL: an instance cannot be destroyed without coming through
  // removeInstance, and removeInstance invalidates in the same call.
  //
  // DELIBERATELY A SECOND KEY, NOT A REUSE OF m_batchRecordKey. The fanout
  // record and the resident record answer different questions over different
  // populations (one batch of placements versus one object across its life),
  // an instance can legitimately be in both, and sharing the field would make
  // either invalidation silently clear the other's claim.
  mutable uint64_t m_residentKey = 0ull;

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
  // NV-DXVK [perf] handoff v5 sec 4c: ONE digest covering every input that BOTH
  // the `surf` block and the event-fanout gate of updateInstance read -- material
  // and legacy-material hashes, geometry asset hash, category flags, texgen, alpha
  // state, the texture arg/op stages, the three samplers, and the ResourceCache
  // binding epoch. See the notes in rtx_instance_manager.cpp for what each
  // material hash does and does not cover, and why one shared key beats two.
  //
  // NV-DXVK [perf] 2026-08-07: built in TWO stages now. Everything above except
  // the category flags is draw-scoped, so computeDrawStateKey() digests it once
  // per draw into DrawScopedState::stateKey, and mixInstStateKey() folds in this
  // instance's category flags. Same equivalence classes, one fifteenth of the
  // gathers. The absolute value differs from the pre-split scheme, which is
  // harmless: it is only ever compared against this same instance's value from
  // the previous frame.
  //
  // kEmptyHash means "never written", which forces the first update of a fresh
  // instance through both full paths regardless of what the incoming draw hashes
  // to. Eye draws also park kEmptyHash here, so they never skip.
  XXH64_hash_t m_instStateKey = kEmptyHash;
  // NV-DXVK [perf] fastInstanceUpdate: the per-draw inputs the fast path cannot
  // key (winding, projection parity, RT-target, sub-view flags), captured at the
  // last FULL update. DrawScopedState::fastDrawBits must match this byte before
  // the fast path may retain m_vkInstance.flags / m_geometryFlags / mask.
  // 0xFF never matches a real bit set (only the low 5 bits are used), so a fresh
  // instance always takes the full path at least once.
  uint8_t m_fastDrawBits = 0xFF;
  VkAccelerationStructureInstanceKHR m_vkInstance;
  VkGeometryFlagsKHR m_geometryFlags = 0;
  uint32_t m_firstBillboard = 0;
  uint32_t m_billboardCount = 0;

  CategoryFlags m_categoryFlags;

  XXH64_hash_t m_spatialCacheHash = kEmptyHash;

  // NV-DXVK [Phase2b]: frame id of the last DEFERRED spatial-map op recorded for
  // this instance (sharded instance phase only). Lets onTransformChanged detect
  // that an earlier op of the same frame is still pending, in which case its own
  // key-unchanged skip test is unsound (m_spatialCacheHash is stale until the
  // ordered tail applies the chain) and the op must be recorded unconditionally.
  // Cleared by InstanceManager::applyDeferredSpatialOp.
  uint32_t m_spatialOpPendingFrame = kInvalidFrameIndex;

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

// NV-DXVK [Phase2b sharded instance processing]: per-thread context for the
// flush-side parallel shard phase (see PHASE2B_IMPLEMENTATION_SPEC.md).
//
// `info` is non-null exactly while a worker (or the game thread's inline
// fallback) is running one DrawWorkItem's geom-decide + instance work inside
// SceneManager::processDeferredDrawBatch; it points at that item's sidecar so
// deep callees (RtInstance::onTransformChanged / teleport, updateInstance,
// findSimilarInstance) can record deferred work without new plumbing through a
// dozen signatures. When null — the CS thread, the ordered tail, the legacy
// path — every divergence point below compiles to a null test and the code is
// byte-identical to the pre-Phase2b behavior.
//
// `deferredThisDraw` is the miss sentinel: findSimilarInstance sets it (instead
// of adding/migrating/teleporting on a worker) and processSceneObjectImpl
// returns nullptr without running mid/add/update; the caller marks the item for
// the ordered tail, which re-runs the sequential path verbatim.
struct ShardedInstancePhase {
  ShardedDrawInfo* info = nullptr;
  bool deferredThisDraw = false;
  // Ordered-tail mode: the phase machinery stays ON (CS-domain work — buffer
  // binds, billboards, OMM, portals, map writes — still defers into the
  // sidecar, applied/replayed later), but the MISS paths (addInstance,
  // migration, portal teleport) run inline: the tail is single-threaded and is
  // exactly where those are supposed to resolve.
  bool allowMiss = false;
  // The pending-ops entry for the updateInstance call currently on this thread's
  // stack — pushed at updateInstance entry, written by the divergence sites
  // (buffer bind, billboard, OMM) as they are reached. Only ever dereferenced
  // during that same call, so vector growth from LATER pushes cannot dangle it.
  ShardedDrawInfo::PendingInstanceOps* currentOps = nullptr;
};
extern thread_local ShardedInstancePhase t_shardPhase;
inline bool inShardedInstancePhase() { return t_shardPhase.info != nullptr; }

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

  // NV-DXVK [perf] handoff v5 sec 4c: may this handler's onInstanceUpdated be
  // skipped when the instance's material binding is provably unchanged?
  //
  // Set by the REGISTRAR, because only the handler's owner knows what its callback
  // depends on. SceneManager's handler is skippable: for an unchanged binding it
  // re-derives the same surfaceMaterialIndex the instance already holds.
  // OpacityMicromapManager's is NOT: it does per-instance staging bookkeeping that
  // has nothing to do with the material, so it must see every instance.
  //
  // Defaults to false so a handler added later is correct by default and opts in
  // deliberately -- the failure direction for a wrong `true` here is a handler
  // silently not running.
  bool skippableWhenBindingUnchanged = false;

  // NV-DXVK [perf] 2026-08-08 (handoff d §3): may this handler's
  // onInstanceUpdated ALSO be skipped on the fastInstanceUpdate commit path
  // (instance provably binding-unchanged AND frameAge != 0) when the
  // instance's OMM pending-work flag
  // (getOpacityMicromapInstanceData().hasPendingNumTexelsCalculation())
  // is clear?
  //
  // CONTRACT, set by the registrar: `true` asserts that for such an instance
  // the callback's entire body is a no-op unless that flag is set. OMM's
  // handler is exactly this shape: its first-sight staging arm requires
  // frameAge == 0 (impossible on the fast path -- kFastCreated rejects those),
  // leaving only the flag-gated numTexelsPerMicroTriangle calculation, and
  // the flag lives on the RtInstance where the fast path can read it for the
  // cost of a member load instead of a std::function dispatch + profile zone.
  // ~14k fast-path instances/frame make that indirection the fast path's
  // floor (handoff §3). Same default-false reasoning as the field above.
  bool skippableWhenNoPendingOmmWork = false;

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

  // NV-DXVK [Phase2b]: the ONE escape lock of THE_OPTIMISATION_PLAN_2.md Sec 6.
  // Taken only on cold paths during the sharded instance phase (new-material
  // creation, capturer flags, player/view-model vector pushes, fanout-record
  // bookkeeping). Never taken on the memo-hit or find-hit paths. Owned here so
  // both InstanceManager and SceneManager (which owns this manager) reach the
  // same mutex without a second object.
  std::mutex& shardEscapeMutex() { return m_shardEscapeMutex; }

  // NV-DXVK [Phase2b]: apply one deferred spatial-map write recorded by a worker
  // (see DeferredSpatialOp). Called from the ordered tail, single-threaded, in
  // arena order. Reads/writes instance->m_spatialCacheHash at apply time so
  // multi-op chains on one instance resolve exactly like the inline code did.
  void applyDeferredSpatialOp(const DeferredSpatialOp& op);

  // NV-DXVK [Phase2b]: assign the order-sensitive decal sort order for an
  // instance whose updateInstance ran on a worker. Called from the ordered tail
  // in arena order, so the assigned values match the sequential path.
  void assignDecalSortOrder(RtInstance& instance) {
    instance.surface.decalSortOrder = m_decalSortOrderCounter++;
  }

  // NV-DXVK [Phase2b]: CS-record-step half of processInstanceBuffers. Rebinds the
  // instance's surface buffer indices from the (post-bake, post-updateBufferCache)
  // BlasEntry — this frame's bake is what decides which slots the geometry holds,
  // and it happens on the CS thread, so this write cannot happen on the flush
  // side. Same body as the non-hoisted processInstanceBuffers path.
  void bindInstanceBuffersFromBlas(const BlasEntry& blas, RtInstance& instance) const;

  // NV-DXVK [Phase2b]: CS-record-step half of the updateInstance billboard stage
  // (createBillboards/createBeams read mapped buffer contents and append to the
  // CS-domain m_billboards vector). Returns whether billboards were generated,
  // which gates the deferred OMM callback exactly like billboardsGotGenerated
  // did inline. cameraDir = main camera direction, as at the inline site.
  bool runBillboardStage(RtInstance& instance, const Vector3& cameraDir);

  // NV-DXVK [Phase2b]: CS-record-step replay of the deferred (non-skippable) OMM
  // event-handler dispatch for an instance whose updateInstance ran on a worker.
  // Fires only handlers that declared skippableWhenNoPendingOmmWork (the
  // contract that identifies the OMM handler without naming its type) — the
  // skippable (SceneManager) handler already ran on the flush side.
  void fireDeferredOmmCallbacks(RtInstance& instance, const DrawCallState& drawCall,
                                const MaterialData& materialData, bool hasTransformChanged,
                                bool hasPreviousPositions, bool isFirstUpdateThisFrame);

  // NV-DXVK [Phase2b]: ordered-tail continuation for fanout placements whose
  // find deferred on a worker. Rebuilds each placement's FanoutSplit with the
  // SAME composition rules as processSceneObjectFanout's loop (identity-exact
  // base skip, engine prev-transform history) and runs the sequential impl —
  // find repeats against the now-current maps, then migrate/add/update inline.
  // Appends produced instances to out_instances.
  void processDeferredFanoutPlacements(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    DrawCallCache* drawCallCache, const std::vector<uint32_t>& placements,
    std::vector<RtInstance*>& out_instances);

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

  // ================================================================
  // NV-DXVK [Perf.PushInst] PHASE 2 -- PUSH, NOT POLL.
  //
  // THE MEASUREMENT THIS EXISTS FOR. [Perf.UpdInst] reads static=97%,
  // REDUNDANT=97%, xfChg=1%, matChg=0%, and reachPct entry=100 fastRet=93. So
  // ~15.5k instances a frame are each asked "did anything change" and 97% of
  // them answer no. entry 2.73 ms + fastRet 4.15 ms = 6.88 ms/frame of dxvk-cs
  // spent ASKING; the stages that do real work reach 6% and cost ~1.9 ms. We
  // pay 3.6x more to ask the question than to do the work.
  //
  // WHY A BATCH RECORD AND NOT A PER-INSTANCE DIRTY FLAG. The visits are not
  // per draw -- 1,349 draws produce ~15.5k instance visits, because fanout
  // draws carry a placement array. The batch is therefore the natural unit:
  // one fingerprint decides the fate of every placement in it, so the common
  // case costs one hash and one bulk stamp instead of N finds and N updates.
  //
  // WHAT MAKES THE SKIP LEGAL. An unchanged instance cannot simply be omitted:
  // garbageCollection reaps on
  //     m_frameLastUpdated + instanceKeepN <= currentFrame
  // and nothing else. So "skip" must mean KEEP ALIVE WITHOUT REPROCESSING, and
  // that is exactly a frame-id stamp -- which is why this is expressible as a
  // bulk write over a contiguous list rather than needing a redesign of GC.
  //
  // WHAT IS NOT SAFE TO ASSUME, AND IS THEREFORE HASHED OR REPLAYED:
  //   - setFrameLastUpdated() CLEARS m_seenCameraTypes on the first stamp of a
  //     frame, so the skip path must re-register the draw's camera or portal
  //     and view-model logic silently loses its camera set.
  //   - the record holds RAW RtInstance*. RtInstance::m_batchRecordKey is the
  //     back-pointer that makes removeInstance invalidate this in O(1); see
  //     that field for why a back-pointer rather than a scan.
  //
  // THE ORDER OF OPERATIONS IS THE SAME ONE THE SPLIT-TRANSFORM CACHE AND THE
  // EXTRACT MEMO BOTH USED, AND IT IS NOT OPTIONAL: verify first, skip second.
  // rtx.pushInstanceRecordsVerify defaults ON and runs both paths, scoring the
  // prediction without acting on it. Nothing is skipped until FAIL reads 0.
  struct FanoutBatchRecord {
    uint64_t inputHash        = 0ull;   // fingerprint of everything resolution reads
    uint32_t frameLastServed  = kInvalidFrameIndex;
    uint32_t frameLastBuilt   = kInvalidFrameIndex;
    bool     valid            = false;  // cleared by removeInstance / BLAS teardown
    std::vector<RtInstance*> instances; // exactly what the placement loop produced
  };
  std::unordered_map<uint64_t, FanoutBatchRecord> m_fanoutRecords;
  // NV-DXVK [Perf.PushInst] OCCURRENCE ORDINAL, added 2026-08-14 after the first
  // verify run read FAIL=3367 with builtFrame == currentFrame on 100% of the
  // failures.
  //
  // WHAT WENT WRONG. fanoutRecordKey names (vertex shader, geometry), and
  // SEVERAL DRAWS PER FRAME SHARE THAT IDENTITY. Draw 1 stored the record; draw
  // 2 found it valid with a matching fingerprint and predicted a hit -- but the
  // two resolve to DIFFERENT instances, because resolution is stateful within a
  // frame: draw 1's instances are already claimed, so findSimilarInstance hands
  // draw 2 a fresh set. The record was describing the previous DRAW, not the
  // previous FRAME. It also meant every same-key draw clobbered the record, so
  // the next frame compared against whichever draw happened to be last -- which
  // is where the 86% input-miss rate came from. One defect, both symptoms.
  //
  // THE FIX. The Nth occurrence of a batch identity within a frame gets its own
  // record, so draw N is only ever compared against draw N of a previous frame.
  // Counted here rather than taken from any draw index, for the reason
  // [DrawRedund] records about its own ordinal: the draw counters only advance
  // for draws surviving the filters, so several draws share one value.
  std::unordered_map<uint64_t, uint32_t> m_fanoutOrdinals;
  // Fingerprint of every input the placement resolution reads. Declared here so
  // the contract above sits next to it; defined in the .cpp beside its caller.
  uint64_t fanoutRecordFingerprint(const DrawCallState& drawCall,
                                   const BlasEntry& blas,
                                   const MaterialData& materialData,
                                   const std::vector<Matrix4>* transforms,
                                   const std::vector<Matrix4>* prevTransforms) const;
  uint64_t fanoutRecordKey(const DrawCallState& drawCall, const BlasEntry& blas) const;
  // Drop any record referencing this instance. Called from removeInstance.
  void invalidateFanoutRecordFor(const RtInstance* instance);

public:
  // NV-DXVK [ResidentScene]: the RT-side half of the resident scene. Lives here
  // because this is the object that owns instance lifetime -- residency is a
  // statement about when an instance may be reaped, so the record store and the
  // reaper have to be in the same place or they will disagree.
  //
  // Touched ONLY from the RT/CS side. The frame thread's half is
  // D3D11Rtx::ResidentGateIndex and the two never share a structure; see
  // rtx_resident_scene.h for why that decomposition removes the need for a lock.
  ResidentScene& getResidentScene() { return m_residentScene; }
  const ResidentScene& getResidentScene() const { return m_residentScene; }
private:
  ResidentScene m_residentScene;
  // [Perf.PushInst] tallies, dxvk-cs only.
  uint32_t m_piBatches = 0, m_piHit = 0, m_piMissKey = 0, m_piMissInput = 0;
  uint32_t m_piMissInvalid = 0, m_piServedInst = 0, m_piFail = 0;
  // Predictions vs SKIPS. m_piHit counts skips, which are zero by construction
  // while verify is on -- so on the first run the ceiling was invisible and the
  // line read hit=0 whether the record was working perfectly or not at all.
  // m_piPredict counts records that WOULD have served, which is the number that
  // says whether this is worth turning on. m_piMissSameFrame counts predictions
  // refused because the record was built this same frame; see m_fanoutOrdinals.
  uint32_t m_piPredict = 0, m_piMissSameFrame = 0, m_piPredictInst = 0;
  // guard  = predictions REFUSED by the pre-stamp instance validation, i.e.
  //          wrong answers downgraded to cache misses. Expected non-zero and
  //          harmless; it is the mechanism working, not a fault.
  // capped = stores refused because the map was at the ceiling after the sweep.
  //          Should be 0; non-zero means raise pushInstanceRecordsMaxBatches.
  // swept  = records retired by the per-frame age sweep.
  uint32_t m_piGuard = 0, m_piCapped = 0, m_piSwept = 0;
  // stale = predictions refused because the record was built more than one frame
  //         ago, i.e. the batch skipped frames and the global resolution state
  //         had time to drift. This was the entire residue of the second verify
  //         run (FAIL=7, all four discriminator bits zero, gaps of 3-11 frames).
  uint32_t m_piMissStale = 0;
  uint32_t m_piFrame = kInvalidFrameIndex;
  // ================================================================

  std::vector<RtInstance*> m_instances;

  // NV-DXVK [Phase2b]: see shardEscapeMutex() above. Leaf lock — nothing is
  // called out of any hold scope except allocator/logging, so it cannot deadlock
  // with the probe mutexes on the same paths.
  std::mutex m_shardEscapeMutex;
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

  // NV-DXVK [perf] fastInstanceUpdate: once-per-frame digest of every runtime
  // option the fast path's skipped region reads. If the digest differs from the
  // previous frame's, the fast path is disabled for THIS frame, so every
  // instance re-derives its state under the new option values (self-healing on
  // option flips, zero per-instance cost). Mutable because it is maintained
  // lazily from const computeDrawScopedState; the mutex is only touched on the
  // first draw of each frame (and by racing threads at that boundary).
  mutable std::mutex m_fastOptMutex;
  mutable std::atomic<uint32_t> m_fastOptFrame { kInvalidFrameIndex };
  mutable uint64_t m_fastOptDigest = 0;
  mutable bool m_fastOptStable = false;
  bool fastPathOptionsStable() const;

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
  // outQueryMatrixHash: if non-null, receives XXH64(firstInstanceObjectToWorld)
  // as computed by the exact stage -- or 0 when that stage keyed on stablePropId
  // instead and no matrix hash was taken. Written on every path, hit or miss, so
  // the caller can feed it to SpatialKeyHint regardless of the outcome.
  RtInstance* findSimilarInstance(BlasEntry& blas, const MaterialData& material, const Matrix4& firstInstanceObjectToWorld, CameraType::Enum cameraType, const RayPortalManager& rayPortalManager, uint64_t stablePropId = 0, DrawCallCache* drawCallCache = nullptr, const Matrix4* prevObjectToWorld = nullptr, XXH64_hash_t* outQueryMatrixHash = nullptr);

  // NV-DXVK [perf] 2026-08-07: the per-DRAW inputs an instance update needs.
  //
  // A fanout batch turns one draw into ~15 placements ([Perf.SceneObj]
  // callsPerFrame=15,665 against [ProcDCS] draws=1,060), and both of these used
  // to be rebuilt from scratch by every one of them even though neither can
  // differ between placements: alphaState is a pure function of (drawCall,
  // materialData), and stateKey digests only draw-, material- and frame-scoped
  // inputs. Built once by the caller that owns the draw and handed down.
  //
  // If you add a field here, it must be constant across a draw's placements. The
  // per-placement half of the state key is mixInstStateKey().
  // NV-DXVK [perf] 2026-08-07: the BLAS buffer binding processInstanceBuffers
  // copies into RtSurface, resolved once instead of once per placement.
  //
  // Every field is a verbatim read of blas.modifiedGeometryData (or, for
  // texcoordEncoding, a two-line derivation from its texcoord buffer's vertex
  // format). It is a plain POD so the per-instance side is a struct read
  // followed by the same ~23 writes it always did -- the writes cannot be
  // eliminated, only the re-derivation can.
  //
  // Membership rule, and it is the load-bearing one: this may only hold values
  // that are immutable across a draw's placements. That holds because
  // modifiedGeometryData is written exclusively by SceneManager::
  // processDrawCallState, which finishes every write before it calls
  // processSceneObject/processSceneObjectFanout. See rtx.hoistSurfaceBufferBinding
  // for why the per-BLAS version of this idea is unsafe and this one is not.
  struct SurfaceBufferBinding {
    uint32_t positionBufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t positionOffset = 0;
    uint32_t positionStride = 0;
    uint32_t previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t normalBufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t normalOffset = 0;
    uint32_t normalStride = 0;
    VkFormat normalFormat = VK_FORMAT_UNDEFINED;
    uint32_t color0BufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t color0Offset = 0;
    uint32_t color0Stride = 0;
    uint32_t texcoordBufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t texcoordOffset = 0;
    uint32_t texcoordStride = 0;
    uint32_t indexBufferIndex = kSurfaceInvalidBufferIndex;
    uint32_t indexStride = 0;
    RtSurface::TexcoordEncoding texcoordEncoding = RtSurface::TexcoordEncoding::Float;
    bool hasLightmap = false;
    bool isVgui = false;
    uint32_t vguiOffset = 0;
    uint16_t vguiFontBoundsBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
    uint16_t vguiImgBoundsBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
    uint16_t vguiStylesBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
  };

  struct DrawScopedState {
    RtSurface::AlphaState alphaState {};
    // Digest of every draw-scoped instance-state key input, or kEmptyHash when
    // keyEligible is false.
    XXH64_hash_t stateKey = kEmptyHash;
    // Resolved once per draw; consumed by processInstanceBuffers. Left at its
    // defaults when rtx.hoistSurfaceBufferBinding is off, in which case
    // processInstanceBuffers re-derives per instance and never reads this.
    SurfaceBufferBinding buffers {};
    // False for eye draws, which never take the surf/tail skip.
    bool keyEligible = false;
    // NV-DXVK [perf] fastInstanceUpdate: may placements of this draw take the
    // updateInstance fast path at all this frame. Folds in the option itself,
    // keyEligible, and the once-per-frame option-digest stability check (see
    // fastPathOptionsStable) so a runtime option flip forces one full frame.
    bool fastPathAllowed = false;
    // Per-draw inputs the fast path checks against RtInstance::m_fastDrawBits:
    // bit0 drawClockwise, bit1 isUsingRaytracedRenderTarget, bit2 isSubView,
    // bit3 isSubViewSkybox, bit4 tf2StableBackfaceCull projection parity.
    uint8_t fastDrawBits = 0;
  };

  // blas is read for SurfaceBufferBinding only. Both callers already hold the
  // same BlasEntry reference they later hand to processSceneObjectImpl, so the
  // binding resolved here is the one every placement of this draw will bind.
  DrawScopedState computeDrawScopedState(const BlasEntry& blas,
                                         const DrawCallState& drawCall,
                                         const MaterialData& materialData) const;

  // Shared body of processSceneObject / processSceneObjectFanout. split is null
  // for an ordinary draw and non-null for one placement of a split fanout batch.
  // drawState is shared by every placement of the draw - see DrawScopedState.
  RtInstance* processSceneObjectImpl(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache, const FanoutSplit* split, const DrawScopedState& drawState);

  RtInstance* addInstance(BlasEntry& blas);
  // Binds currentInstance.surface to blas's buffers. blas is still passed
  // because rtx.hoistSurfaceBufferBinding=false re-derives from it per
  // instance; on the default path only drawState.buffers is read.
  void processInstanceBuffers(const BlasEntry& blas, const DrawScopedState& drawState,
                              RtInstance& currentInstance) const;

  // The whole read side of processInstanceBuffers. Called once per draw from
  // computeDrawScopedState, or once per instance when the hoist is off.
  static SurfaceBufferBinding resolveSurfaceBufferBinding(const BlasEntry& blas);

  // The whole write side. Split from the read so the two paths above share one
  // copy of the ~23 assignments rather than duplicating them.
  void applySurfaceBufferBinding(const SurfaceBufferBinding& binding, RtInstance& currentInstance) const;

  void updateInstance(
    RtInstance& currentInstance, const CameraManager& cameraManager,
    const BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    const DrawScopedState& drawState,
    const FanoutSplit* split = nullptr, const SpatialKeyHint& keyHint = SpatialKeyHint());

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

