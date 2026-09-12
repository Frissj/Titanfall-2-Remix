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
#include "rtx_gpu_scene.h"
#include "rtx_scene_cull.h"
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

  // NV-DXVK [GpuScene] slice 8: m_reorderedSurfaces is now the SLOT table --
  // index = persistent surface slot, value = owning instance, nullptr = a hole
  // no run owns. getSurfaceCount() is the high-water mark, so every per-slot
  // buffer and every shader bound is sized exactly as before. Callers that
  // dereference entries must skip nullptr.
  uint32_t getSurfaceCount() const { return m_reorderedSurfaces.size(); }
  const std::vector<RtInstance*>& getOrderedInstances() const { return m_reorderedSurfaces; }

  // [GpuScene] read side, for SceneManager's surface-material table (which is
  // indexed by the same slots) and the one [GpuScene] stats line.
  const SurfaceSlotTable& getSurfaceSlots() const { return m_surfaceSlots; }
  const DeltaUploadTable& getSurfaceDelta() const { return m_surfaceDelta; }
  const DeltaUploadTable& getTransformDelta() const { return m_transformDelta; }
  bool wereSurfaceSlotsCompacted() const { return m_surfaceSlotsCompacted; }
  struct GpuSceneVerify {
    uint32_t frames = 0;        // frames the structural verify ran
    uint32_t tlasRefs = 0;      // TLAS surface indices checked
    uint32_t piRanges = 0;      // PointInstancer ranges checked
    uint32_t structFail = 0;    // CUMULATIVE. Must stay 0.
    uint32_t doubleClaim = 0;   // CUMULATIVE. A slot handed to two owners in one walk.
  };
  const GpuSceneVerify& getGpuSceneVerify() const { return m_gsVerify; }
  // Merged-bucket BLAS selection, summed since the last resetGpuSceneStats().
  struct MergedBlasStats {
    uint32_t frames = 0;
    uint32_t buckets = 0;   // bucket-frames
    uint32_t sizeHit = 0;   // size query served from the per-bucket cache
    uint32_t pinHit = 0;    // bucket built into one of its own pinned BLASes
    uint32_t created = 0;   // new BLAS allocated for a bucket (first sight, growth)
    uint32_t skip = 0;      // content unchanged: no GPU build
    uint32_t update = 0;
    uint32_t build = 0;
  };
  const MergedBlasStats& getMergedBlasStats() const { return m_mergedBlasStats; }
  const DeltaUploadTable& getInstanceDelta() const { return m_instanceDelta; }
  // G5 TLAS refit, per build summed over types, since the last reset.
  struct TlasRefitStats {
    uint32_t refit = 0;          // builds done as UPDATE
    uint32_t build = 0;          // full builds
    uint32_t whyCadence = 0;     // full build forced by rtx.gpuScene.tlasRefitMaxFrames
    uint32_t whyTopology = 0;    // region split, count, flags or active status changed
    uint32_t whyRealloc = 0;     // destination AS was (re)created this build
  };
  const TlasRefitStats& getTlasRefitStats() const { return m_tlasRefitStats; }
  void resetGpuSceneStats();

  // NV-DXVK [SceneCull] slice 9: the buffer the TLAS is built from this frame.
  // With the GPU scene cull running it is the cull pass's output (the instance
  // table with the verdict's masks, PointInstancer entries written into it by
  // the PI pass); otherwise the instance table itself.
  const Rc<DxvkBuffer>& tlasInstanceBuffer() const {
    return m_sceneCull.used() ? m_sceneCull.culledBuffer() : m_vkInstanceBuffer;
  }

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
    std::vector<uint32_t> surfaceIndexMapping;
    // NV-DXVK [GpuScene]: what the mapping buffer currently holds, so an
    // identical mapping (the settled-scene case, now that slots persist) is not
    // re-sent.
    std::vector<uint32_t> uploadedSurfaceIndexMapping;
    uint32_t previousFrameSurfaceCount = 0; // Tracks last frame's surface count for mapping coverage
  } uploadSurfaceDataFuncState;

  // ------------------------------------------------------------------------
  // NV-DXVK [GpuScene] -- ARCHITECTURE_OVERHAUL.md slice 8. See rtx_gpu_scene.h.
  // ------------------------------------------------------------------------
  SurfaceSlotTable m_surfaceSlots;
  bool m_surfaceSlotsCompacted = false;
  DeltaUploadTable m_surfaceDelta { uint32_t(kSurfaceGPUSize), "surface" };
  DeltaUploadTable m_transformDelta { uint32_t(sizeof(VkTransformMatrixKHR)), "transform" };
  // NV-DXVK [GpuScene] G5: the CPU half of m_vkInstanceBuffer, and what a TLAS
  // refit needs to know about it. m_tlasInstSig[t][i] = (instance flags << 1)
  // | active, per CPU entry of type t, as of the last upload.
  DeltaUploadTable m_instanceDelta { uint32_t(sizeof(VkAccelerationStructureInstanceKHR)), "instance" };
  bool m_instanceBufferReplaced = false;
  std::vector<uint32_t> m_tlasInstSig[Tlas::Count];
  uint32_t m_tlasPiSlotsLast[Tlas::Count] = {};
  uint64_t m_tlasPiSigLast[Tlas::Count] = {};
  VkBuildAccelerationStructureFlagsKHR m_tlasLastFlags[Tlas::Count] = {};
  bool m_tlasTopologySame[Tlas::Count] = {};
  uint32_t m_tlasRefitRun[Tlas::Count] = {};   // consecutive refits since the last full build
  TlasRefitStats m_tlasRefitStats;
  // Set when the device buffer behind a delta table was (re)created this frame.
  bool m_surfaceBufferReplaced = false;
  bool m_transformBufferReplaced = false;
  // Per-slot claim stamp for the double-claim check (verify only).
  std::vector<uint32_t> m_slotClaimEpoch;
  uint32_t m_slotClaimWalk = 0;
  GpuSceneVerify m_gsVerify;
  // Acquire a run and publish its owner + firstIndex offsets into the slot
  // table. Returns the base slot or SurfaceSlotTable::kNoSlot.
  uint32_t acquireSurfaceRun(uint64_t key, uint32_t count, RtInstance* uniformOwner,
                             RtInstance* const* perSlotOwners,
                             const uint32_t* firstIndexOffsets = nullptr);
  void verifyGpuSceneStructure();
  // The bucket's compatibility key -- the same key its surface run is filed
  // under, so a bucket's slots and its BLAS pair share one identity.
  static uint64_t bucketCompatKey(const BlasBucket& bucket);

  // NV-DXVK [GpuScene] merged-bucket BLAS pinning. There is exactly one bucket
  // per compatibility key (tryAddInstance has no size cap), so the key is a
  // stable identity for the bucket's BLAS as it already is for its slots.
  // Before this, every bucket every frame re-ran the driver size query and
  // best-fit scanned the whole m_blasPool, which could hand a bucket another
  // bucket's BLAS -- whose topology/content hashes are not this bucket's, so
  // the UPDATE and build-skip paths were defeated by a pool shuffle.
  //
  // A PAIR, because with rtx.enablePreviousTLAS the BLAS built last frame is
  // still referenced by the previous TLAS and may not be written this frame;
  // the bucket alternates between two. The pins are SECOND references: every
  // pinned BLAS is also in m_blasPool, so GC, the resource tracking in
  // buildTlas and the crash recorder see it exactly as before, and a pin is
  // dropped on the same evidence the pool evicts on.
  struct MergedBucketBlas {
    Rc<PooledBlas> pinned[2];
    uint64_t sizeKey = 0;  // 0 = no cached size
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
    uint32_t lastFrame = kInvalidFrameIndex;  // frame this entry was last claimed
  };
  std::unordered_map<uint64_t, MergedBucketBlas> m_mergedBucketBlas;
  MergedBlasStats m_mergedBlasStats;

  // NV-DXVK [SceneCull] slice 9 -- see rtx_scene_cull.h. m_mergedSources[t][i]
  // is what produced m_mergedInstances[t][i], so the entry's cull record can be
  // derived: a dynamic-BLAS instance (the entry carries the instance's
  // transform; the record is its BLAS's object box), a merged bucket (identity
  // transform; the record is the union of its members' world boxes), or
  // neither (a billboard: never tested). Pushed beside every m_mergedInstances
  // push, cleared and truncated with it. Valid for the frame it was pushed in:
  // the bucket lives in m_persistBuckets until the next merge.
  struct MergedEntrySource {
    RtInstance* instance = nullptr;
    const BlasEntry* blas = nullptr;
    const BlasBucket* bucket = nullptr;
  };
  std::vector<MergedEntrySource> m_mergedSources[Tlas::Count];
  SceneCullPass m_sceneCull;
  void packSceneCullRecord(const MergedEntrySource& src, SceneCullRecord& record);

  // The primitive-ID prefix sum is double-buffered by SWAPPING the two device
  // buffers rather than re-uploading last frame's array into the second one:
  // after the swap the "last frame" buffer already holds last frame's data.
  // m_prefixSumHeld[i] is what buffer i holds, so a repeat is not re-sent.
  std::vector<uint32_t> m_prefixSumHeldCurrent;
  std::vector<uint32_t> m_prefixSumHeldLast;

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

