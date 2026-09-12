/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "../util/rc/util_rc_ptr.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_gpu_crash_recorder.h"
#include "rtx_staging.h"
#include "rtx_point_instancer_system.h"
#include "../util/util_vector.h"
#include "../util/util_matrix.h"
#include "../util/util_struct_hash.h"

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
class ResourceCache;
class CameraManager;
class OpacityMicromapManager;

// AccelManager is responsible for maintaining the acceleration structures (BLAS and TLAS)
class AccelManager : public CommonDeviceObject {
  class BlasBucket {
  public:
    std::vector<VkAccelerationStructureGeometryKHR> geometries {};
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges {};
    std::vector<RtInstance*> originalInstances {};
    std::vector<uint32_t> primitiveCounts {};
    std::vector<uint32_t> instanceBillboardIndices {};  // Billboard index within an instance's billboard array
    std::vector<uint32_t> indexOffsets {};              // Index offsets within geometry
    uint8_t instanceMask = 0;
    uint32_t instanceShaderBindingTableRecordOffset = 0;
    uint32_t customIndexFlags = 0;
    VkGeometryInstanceFlagsKHR instanceFlags = 0;
    bool usesUnorderedApproximations = false;
    uint32_t reorderedSurfacesOffset = UINT32_MAX;
    bool hasOmmInstances = false;
    bool hasSssInstances = false;

    // The PooledBlas assigned to this bucket by createBlasBuffersAndInstances.
    // Stored here so the per-bucket cache can capture it after buildBlases.
    PooledBlas* assignedBlas = nullptr;
    
    // Tries to add a geometry instance to the bucket. The addition is successful if either:
    //   a) the bucket is empty,
    //   b) the instance has the same mask etc. as all other instances in the bucket.
    bool tryAddInstance(RtInstance* instance);
  };

  // Key for O(1) bucket lookup in the merged-BLAS path.
  // Two instances can share a merged BLAS bucket iff they have identical keys.
  struct BlasBucketKey {
    uint32_t instanceShaderBindingTableRecordOffset = 0;
    uint32_t customIndexFlags = 0;
    VkGeometryInstanceFlagsKHR instanceFlags = 0;
    uint8_t instanceMask = 0;
    bool usesUnorderedApproximations = false;
    bool isSubsurface = false;
    uint8_t pad = 0;

    bool operator==(const BlasBucketKey& other) const {
      return instanceMask == other.instanceMask &&
             instanceShaderBindingTableRecordOffset == other.instanceShaderBindingTableRecordOffset &&
             customIndexFlags == other.customIndexFlags &&
             instanceFlags == other.instanceFlags &&
             usesUnorderedApproximations == other.usesUnorderedApproximations &&
             isSubsurface == other.isSubsurface;
    }
  };

  struct BlasBucketKeyHash {
    size_t operator()(const BlasBucketKey& k) const {
      return static_cast<size_t>(hashStructByMemory<BlasBucketKey,
          &BlasBucketKey::instanceShaderBindingTableRecordOffset,
          &BlasBucketKey::customIndexFlags,
          &BlasBucketKey::instanceFlags,
          &BlasBucketKey::instanceMask,
          &BlasBucketKey::usesUnorderedApproximations,
          &BlasBucketKey::isSubsurface,
          &BlasBucketKey::pad>(k));
    }
  };

  struct UniqueBlasInstances {
    BlasEntry* blasEntry = nullptr;
    std::vector<RtInstance*> instances;
  };

public:
  AccelManager(AccelManager const&) = delete;
  AccelManager& operator=(AccelManager const&) = delete;

  explicit AccelManager(DxvkDevice* device);

  // Returns a GPU buffer containing the surface data for active instances
  const Rc<DxvkBuffer> getSurfaceBuffer() const { return m_surfaceBuffer; }

  const Rc<DxvkBuffer> getSurfaceMappingBuffer() const { return m_surfaceMappingBuffer; }

  const Rc<DxvkBuffer> getCurrentFramePrimitiveIDPrefixSumBuffer() const {
    return m_primitiveIDPrefixSumBuffer;
  }

  const Rc<DxvkBuffer> getLastFramePrimitiveIDPrefixSumBuffer() const {
    return m_primitiveIDPrefixSumBufferLastFrame;
  }

  const Rc<DxvkBuffer> getBillboardsBuffer() const { return m_billboardsBuffer; }

  // Clear all instances currently tracked by manager
  void clear();

  // Clean up instances which are deemed as no longer required
  void garbageCollection();

  // Prepares instance buffers for rendering by the GPU
  void prepareSceneData(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers, InstanceManager& instanceManager);

  // Uploads instances' surface data to the GPU
  void uploadSurfaceData(Rc<DxvkContext> ctx);

  // Merges the RtInstance's into a set of BLAS. Some of the BLAS will contain multiple geometries/instances,
  // and some other BLAS will be dedicated to instances with static geometries.
  void mergeInstancesIntoBlas(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers,
                              const std::vector<TextureRef>& textures, const CameraManager& cameraManager, 
                              InstanceManager& instanceManager, OpacityMicromapManager* opacityMicromapManager);

  // Dispatches GPU compute culling for all PointInstancer batches recorded during
  // mergeInstancesIntoBlas. Must be called after prepareSceneData (placeholders uploaded)
  // and before buildTlas.
  void dispatchPointInstancerCulling(Rc<DxvkContext> ctx, const CameraManager& cameraManager,
                                     const Rc<DxvkBuffer>& surfaceMaterialBuffer);

  void buildTlas(Rc<DxvkContext> ctx);

  void dumpCrashState(const char* reason) const { m_gpuCrashRecorder.dump(reason); }

  // Returns the number of live BLAS objects
  static uint32_t getBlasCount();

  uint32_t getSurfaceCount() const { return m_reorderedSurfaces.size(); }
  const std::vector<RtInstance*>& getOrderedInstances() const { return m_reorderedSurfaces; }

  // Returns true if the last mergeInstancesIntoBlas call took the fast-skip
  // path (scene generation unchanged).  When true, m_reorderedSurfaces and
  // all BLAS/surface data are identical to the previous frame, so callers
  // can skip redundant GPU uploads (e.g. surface materials).
  bool wasSceneUnchangedThisFrame() const { return m_sceneUnchangedThisFrame; }

  void removeInstanceFromBucketCache(RtInstance* instance);
  void invalidateOpacityMicromapBindings() { m_ommBindPending = true; }
  // NV-DXVK [CamProbe prevSurf]: THE cross-frame surface identity, parallel to
  // m_reorderedSurfaces. Entry i is the slot that slot i occupied LAST frame,
  // or SURFACE_INDEX_INVALID if it had no predecessor.
  //
  // Nothing else on a probe line is durable. surf= is a slot and slots are
  // reassigned every TLAS build (721 slots carried one VS's ~259 surfaces in a
  // single run). RtInstance::getId() is not durable either - measured on the
  // 16:51 run, ids 2604 and 4586 carry BIT-IDENTICAL centroids and partition
  // the run between them (498 + 516 of 1022 frames), so an instance that is
  // destroyed and recreated reads as two different objects. And
  // RtSurface::firstIndex, which the probe used for this until now, is
  // assigned nowhere in the tree: it is default-0 at rtx_materials.h:406 and
  // only ever mutated by the +=/-= pair around writeGPUData below, so it
  // logged 0 on all 17462 lines of that run. Even reading the array that pair
  // adds - m_reorderedSurfacesFirstIndexOffset - would not help, because that
  // is filled with zeros on every path except the merged-BLAS bucket (:1976),
  // and the geometry under investigation is point-instanced.
  //
  // This snapshot is the previous->current mapping uploadSurfaceData already
  // computes for surfaceIndexMapping, captured BEFORE the same loop overwrites
  // m_previousSurfaceIndex with the current slot (:7071/:7085). Read at any
  // later point in the frame - dispatchTlasProbe included - the accessor is
  // already the current slot and the field is silently useless.
  const std::vector<uint32_t>& getProbePrevSurfaceSlots() const { return m_probePrevSurfaceSlot; }

  // NV-DXVK [TlasBind]: what buildTlas last built, per TLAS type.
  //
  // The last untested layer. Every measurement so far — the surface table, the
  // instance entries, the BLAS references and contents — reads the data that
  // FEEDS the TLAS build, and all of it is identical on frames where geometry
  // renders and frames where it vanishes. Nothing has checked that the
  // acceleration structure the ray tracer actually binds is the one built from
  // that data this frame. Recorded here at the build so the bind site can
  // compare directly, instead of the reader hand-joining two log tags.
  //
  // Note Tlas::Opaque swaps accelStructure/previousAccelStructure every frame
  // (buildTlas), so the object pointer alternating between two values is
  // CORRECT and expected; the defect signature would be the bound object not
  // matching the one built on the SAME frame, or builtFrame lagging.
  struct TlasBuildRecord {
    uint32_t builtFrame = kInvalidFrameIndex;
    uint64_t dstHandle = 0ull;     // VkAccelerationStructureKHR built into
    uint64_t tlasObj = 0ull;       // DxvkAccelStructure* that handle came from
    uint32_t numInstances = 0u;
  };
  const TlasBuildRecord& getTlasBuildRecord(Tlas::Type type) const { return m_tlasBuildRecord[type]; }

private:
  TlasBuildRecord m_tlasBuildRecord[Tlas::Count];

  struct SurfaceInfo {
    uint32_t surfaceMaterialIndex;
    Vector3 worldPosition;
  };

  // Persistent containers to reduce frame to frame reallocations in ::buildParticleSurfaceMapping()
  struct {
    std::vector<AccelManager::SurfaceInfo> surfaceInfoLists[2];   // Two containers for subsequent frames, ping-pong framed to frame
    uint32_t currIndex = 0;
    uint32_t prevIndex = 1;
  } buildParticleSurfaceMappingFuncState;

  // Persistent containers to reduce frame to frame reallocations in ::uploadSurfaceData()
  struct {
    std::vector<unsigned char> surfacesGPUData;
    // NV-DXVK [SurfaceDelta] §9 item 5: last frame's copy, for the changed-byte
    // count. Only allocated while rtx.logSurfaceDelta is on.
    std::vector<unsigned char> prevSurfacesGPUData;
    uint32_t surfaceDeltaLastLogFrame = 0u;
    std::vector<uint32_t> surfaceIndexMapping;
    uint32_t previousFrameSurfaceCount = 0; // Tracks last frame's surface count for mapping coverage
  } uploadSurfaceDataFuncState;

  void buildBlases(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers,
                   const CameraManager& cameraManager, OpacityMicromapManager* opacityMicromapManager, const InstanceManager& instanceManager,
                   const std::vector<TextureRef>& textures, const std::vector<RtInstance*>& instances,
                   const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets, 
                   std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                   std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                   const std::vector<VkTransformMatrixKHR>& instanceTransforms,
                   size_t& currentScratchOffset);
  
  void addBlas(RtInstance* instance, BlasEntry* blasEntry, const Matrix4* instanceToObject);
  void addPointInstancerBlas(RtInstance* rtInstance, BlasEntry* blasEntry);

  void createBlasBuffersAndInstances(Rc<DxvkContext> ctx, 
                                     const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                     std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                     std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                     size_t& currentScratchOffset);
  template<Tlas::Type type>
  void internalBuildTlas(Rc<DxvkContext> ctx, size_t& totalScratchSize);

  void buildParticleSurfaceMapping(std::vector<uint32_t>& surfaceIndexMapping);

  bool validateUpdateMode(const VkAccelerationStructureBuildGeometryInfoKHR& oldInfo, const VkAccelerationStructureBuildGeometryInfoKHR& newInfo);

  std::vector<RtInstance*> m_reorderedSurfaces;
  std::vector<uint32_t> m_reorderedSurfacesFirstIndexOffset;
  // NV-DXVK [perf] 2026-08-08 (handoff d §3, merge loop): scratch for the
  // stable-partition ordering in mergeInstancesIntoBlas — members so their
  // ~15.5k-pointer capacity survives across frames instead of reallocating.
  std::vector<RtInstance*> m_mergeSortScratch;
  std::vector<RtInstance*> m_mergeUntaggedScratch;
  // NV-DXVK [Perf.MergeP] 2026-08-08f: persistent-bucket cache. The buckets
  // themselves persist in m_persistBuckets (aliased as `blasBuckets` inside
  // mergeInstancesIntoBlas); m_persistMembers is the ordered merged-instance
  // sequence (+ per-instance fingerprint of every bucket-shaping input) they
  // were built from. A frame whose sequence matches reuses the buckets
  // verbatim; any difference rebuilds them exactly like the legacy path.
  // See the [Perf.MergeP] block in mergeInstancesIntoBlas for the contract.
  struct MergePersistMember {
    RtInstance* inst = nullptr;
    BlasEntry* blas = nullptr;
    uint64_t fp = 0;
  };
  std::vector<std::unique_ptr<BlasBucket>> m_persistBuckets;
  std::vector<MergePersistMember> m_persistMembers;
  std::vector<MergePersistMember> m_persistScratch;
  uint64_t m_persistEpoch = 0;        // option/global fingerprint at capture
  bool m_persistValid = false;
  bool m_persistQuarantined = false;  // verify failure: off for the session
  uint32_t m_persistReuseN = 0, m_persistRebuildN = 0;
  uint32_t m_persistWhyCount = 0, m_persistWhySeq = 0, m_persistWhyEpoch = 0;
  uint32_t m_persistVerifyN = 0, m_persistVerifyFailN = 0;
  // NV-DXVK [CamProbe prevSurf]: see getProbePrevSurfaceSlots().
  std::vector<uint32_t> m_probePrevSurfaceSlot;
  std::vector<uint32_t> m_reorderedSurfacesPrimitiveIDPrefixSum;              // Exclusive prefix sum for this frame's surface primitive count array
  std::vector<uint32_t> m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame;     // Exclusive prefix sum for last frame's surface primitive count array
  std::vector<VkAccelerationStructureInstanceKHR> m_mergedInstances[Tlas::Count];
  std::vector<Rc<PooledBlas>> m_blasPool;

  // GPU-driven PointInstancer culling batches, recorded per frame in mergeInstancesIntoBlas
  std::vector<PointInstancerBatch> m_pointInstancerBatches;

  // Number of VkAccelerationStructureInstanceKHR slots reserved for PointInstancer
  // instances in each TLAS type.  These slots are NOT stored in m_mergedInstances —
  // the GPU culling shader fills them directly in the instance buffer.
  uint32_t m_pointInstancerSlotsPerType[Tlas::Count] = {};

  // NV-DXVK debug (BLAS-BUILD-INPUT probe): parallel arrays to blasToBuild/blasRangesToBuild,
  // populated at each push_back. Used by the inline dump right before vkCmdBuildAccelerationStructuresKHR
  // so we can look up the source vertex/index DxvkBuffer and the owning PooledBlas per BLAS entry.
  // Cleared at start of buildBlases. nullptr entry means "no side info available" (e.g. merged bucket path).
  std::vector<struct BlasEntry*>  m_debugBlasBuildEntries;
  std::vector<struct PooledBlas*> m_debugBlasBuildDstBlas;

  // NV-DXVK (debug probe B): per-frame routing counters, reset at frame start.
  static inline uint32_t s_probeB_addBlasCount = 0;
  static inline uint32_t s_probeB_addPICount = 0;
  static inline uint32_t s_probeB_addPIInstances = 0;

  // NV-DXVK (debug probe E): handoff of the first PI batch's interleaved BLAS
  // position buffer ref + meta from addPointInstancerBlas to dispatchPointInstancerCulling
  // for GPU readback. Reset each frame at the start of mergeInstancesIntoBlas.
  static inline Rc<DxvkBuffer> s_probeE_posBuffer;
  static inline VkDeviceSize   s_probeE_posSliceOff = 0; // base offset of the slice
  static inline uint32_t       s_probeE_posElemOff  = 0; // offsetFromSlice
  static inline uint32_t       s_probeE_posStride   = 0;
  static inline uint32_t       s_probeE_vertexCount = 0;
  static inline VkFormat       s_probeE_posFormat   = VK_FORMAT_UNDEFINED;
  static inline uint64_t       s_probeE_blasRef     = 0; // for cross-referencing with PI-batch logs

  // NV-DXVK (debug probe F): baseSurfaceIndex of the probeE batch, so we can
  // read back the corresponding surface template from m_surfaceBuffer.
  static inline uint32_t       s_probeF_baseSurfaceIndex = 0;
  static inline bool           s_probeF_valid = false;
  // --- Incremental BLAS build caching ---
  // Scene generation from InstanceManager when the BLAS was last built.
  uint64_t m_lastProcessedGeneration = UINT64_MAX;
  bool m_sceneUnchangedThisFrame = false;
  // Set when newly built OMMs need to be bound to BLASes.  Forces all buckets
  // dirty on the next incremental rebuild so tryBindOpacityMicromap runs.
  bool m_ommBindPending = false;

  // Number of non-billboard entries in m_mergedInstances per TLAS type.
  // prepareSceneData() truncates to this baseline before appending fresh billboard instances.
  uint32_t m_mergedInstancesBaselineCount[Tlas::Count] = {};

  // Tracks all dynamic BLAS references from the last full build so they can be
  // touched during the cached (skip) path to prevent GC from collecting them.
  std::vector<Rc<PooledBlas>> m_activeDynamicBlases;

  // --- Dynamics-only rebuild cached state ---
  // Per-bucket cache from the last build.  Each entry corresponds to one merged
  // BLAS bucket and stores all the data needed to restore its portion of
  // m_reorderedSurfaces and m_mergedInstances without re-running the bucket pipeline.
  struct CachedBucketState {
    // The instances that contributed geometry to this bucket (for dirty checking)
    std::vector<RtInstance*> instances;
    std::vector<uint64_t> instanceCacheIdentities;
    // Surface data for m_reorderedSurfaces
    std::vector<RtInstance*> surfaces;
    std::vector<uint32_t> indexOffsets;
    // The PooledBlas assigned to this bucket (kept alive via Rc)
    Rc<PooledBlas> assignedBlas;
    // TLAS instance template (surface offset must be adjusted each frame)
    VkAccelerationStructureInstanceKHR tlasInstance {};
    // Which TLAS type(s) this bucket was emitted to
    bool isUnordered = false;
    bool hasSssInstances = false;
  };
  std::vector<CachedBucketState> m_cachedBuckets;

  // Maps a merged instance pointer to its bucket index in m_cachedBuckets.
  // Allows O(1) "is this instance in a clean bucket?" check in the main loop.
  std::unordered_map<RtInstance*, uint32_t> m_instanceBucketIndex;

  // Set of BlasEntry* that went to the dynamic path on the last full rebuild.
  // Used for quick O(1) per-instance classification on the dynamics-only path.
  std::unordered_set<BlasEntry*> m_cachedDynamicBlasEntries;

  // Scratch containers for collecting dynamic BLAS groups during mergeInstancesIntoBlas().
  // The map is lookup-only; the vector preserves first-seen emission order.
  std::vector<UniqueBlasInstances> m_uniqueDynamicBlas;
  uint32_t m_uniqueDynamicBlasCount = 0;
  std::unordered_map<BlasEntry*, uint32_t> m_uniqueDynamicBlasIndex;

  void resetUniqueDynamicBlasGroups();

  Rc<DxvkBuffer> m_vkInstanceBuffer; // Note: Holds Vulkan AS Instances, not RtInstances
  Rc<DxvkBuffer> m_surfaceBuffer;
  Rc<DxvkBuffer> m_surfaceMappingBuffer;
  Rc<DxvkBuffer> m_transformBuffer;
  Rc<DxvkBuffer> m_primitiveIDPrefixSumBuffer;
  Rc<DxvkBuffer> m_primitiveIDPrefixSumBufferLastFrame;

  int getCurrentFramePrimitiveIDPrefixSumBufferID() const;

  Rc<PooledBlas> m_intersectionBlas;
  Rc<DxvkBuffer> m_aabbBuffer;
  Rc<DxvkBuffer> m_billboardsBuffer;
  void createAndBuildIntersectionBlas(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers);
  
  Rc<DxvkBuffer> getScratchMemory(const size_t requiredScratchAllocSize);
  Rc<PooledBlas> createPooledBlas(size_t bufferSize, const char* name) const;

  // The recorder currently lives here because it only captures acceleration
  // structure state. Move it higher if crash recording expands beyond AS data.
  RtxGpuCrashRecorder m_gpuCrashRecorder;
  VkDeviceSize m_scratchAlignment;
  Rc<DxvkBuffer> m_scratchBuffer;
};

}  // namespace dxvk

