#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include "../dxvk_buffer.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;

  // ==========================================================================
  // THE GPU SCENE -- ARCHITECTURE_OVERHAUL.md slice 8 (sec 5.1).
  //
  // WHAT THIS REPLACES. m_reorderedSurfaces was cleared and rebuilt every frame
  // and a surface's slot was its POSITION in that frame's array. That is the
  // same shape as the input-byte object key and the m_bufferCache tape (sec 5.1
  // calls it the third instance): an index that means "position i in this
  // frame", papered over with a stable partition so the positions happened to
  // repeat. It forced the whole table -- surfaces, surface materials, prefix
  // sums, mapping -- to be re-packed and re-uploaded every frame, O(scene),
  // and residency makes the scene bigger in exact proportion to how well it
  // works (sec 0.1 item 7).
  //
  // WHAT IT IS NOW. Two pieces, deliberately separate:
  //
  //   SurfaceSlotTable  WHERE a surface lives. A run of contiguous slots per
  //                     key -- one per RtInstance (by cacheIdentity, which is
  //                     unique per allocation, so no ABA), one per merged BLAS
  //                     bucket (by its compatibility key), N for a
  //                     PointInstancer batch. A run keeps its base across
  //                     frames while its key is acquired every frame with the
  //                     same count. This is sec 1.3's third level -- the
  //                     GPUInstance -- given the stable identity it lacked.
  //
  //   DeltaUploadTable  WHAT the GPU copy holds. An element-granular CPU mirror
  //                     of a device buffer. The frame packs every live element
  //                     (the OUTPUT TEST of sec 2 -- it is cheap, per element,
  //                     and it is the only reliable change signal in a fork
  //                     that mutates instances on paths notifySceneChanged()
  //                     never sees); only elements whose bytes differ are
  //                     transferred, coalesced into one multi-region copy.
  //
  // CONTIGUITY IS A HARD CONSTRAINT, AND IT IS WHY THIS IS RUN-BASED. A merged
  // BLAS bucket addresses its surfaces as instanceCustomIndex + GeometryIndex(),
  // and a PointInstancer batch as base + instanceIdx. Both need their slots
  // consecutive, so the allocator hands out ranges, never scattered slots.
  //
  // FREED ON EVIDENCE, NOT AGE (I2, I7). A run is freed at endFrame() when this
  // frame's walk did not acquire it -- the same evidence the old per-frame
  // rebuild used, so no instance lifetime is shortened. A range freed during a
  // frame becomes allocatable only after endFrame(), so no two owners ever hold
  // one slot within a frame. The previous->current surface mapping carries
  // temporal continuity across a move exactly as it did when every slot moved.
  //
  // THREADING. CS thread only, like the rest of AccelManager.
  // ==========================================================================

  class SurfaceSlotTable {
  public:
    static constexpr uint32_t kNoSlot = ~0u;

    // Run keys are tagged in the low bit so an instance run can never collide
    // with a bucket run.
    static uint64_t instanceRunKey(uint64_t cacheIdentity) { return cacheIdentity << 1; }
    static uint64_t bucketRunKey(uint64_t compatHash)      { return (compatHash << 1) | 1ull; }

    // Opens a walk. Compacts (relocates every run) when holes exceed both
    // compactSlack and the live slot count; returns true when it did, so
    // callers know every slot moved this frame.
    bool beginFrame(uint32_t compactSlack);

    // Base of a run of `count` contiguous slots for `key`, or kNoSlot when the
    // run would cross SURFACE_INDEX_MAX_VALUE. Acquiring the same key twice in
    // one walk yields two distinct runs (the second is re-keyed), because the
    // old per-push behaviour gave every push its own slot.
    uint32_t acquire(uint64_t key, uint32_t count, uint32_t maxSlotValue);

    // Closes the walk: frees every run this walk did not acquire, then trims
    // trailing free space off the high-water mark.
    void endFrame();

    void clear();

    uint32_t highWater() const { return m_highWater; }
    uint32_t liveSlots() const { return m_liveSlots; }
    size_t   runCount()  const { return m_runs.size(); }

    struct Stats {
      uint32_t kept = 0;       // runs acquired at the base they held last walk
      uint32_t created = 0;    // runs with no previous base
      uint32_t moved = 0;      // runs that changed base (resize that could not stay put)
      uint32_t resized = 0;    // runs whose count changed (in place or moved)
      uint32_t freed = 0;      // runs retired at endFrame
      uint32_t rekeyed = 0;    // duplicate acquires of one key in one walk
      uint32_t overflow = 0;   // acquires refused at the surface-index ceiling
      uint32_t compactions = 0;
    };
    const Stats& stats() const { return m_stats; }
    void resetStats() { m_stats = Stats(); }

  private:
    struct Run {
      uint32_t base = 0;
      uint32_t count = 0;
      uint32_t epoch = 0;
    };

    uint32_t allocRange(uint32_t count, uint32_t maxSlotValue);
    void     insertFree(std::vector<std::pair<uint32_t, uint32_t>>& list, uint32_t base, uint32_t count);

    std::unordered_map<uint64_t, Run> m_runs;
    // (base, count), sorted by base, coalesced. m_free is allocatable this
    // walk; m_freedThisWalk joins it at endFrame().
    std::vector<std::pair<uint32_t, uint32_t>> m_free;
    std::vector<std::pair<uint32_t, uint32_t>> m_freedThisWalk;
    uint32_t m_highWater = 0;
    uint32_t m_liveSlots = 0;
    uint32_t m_epoch = 0;
    Stats m_stats;
  };

  class DeltaUploadTable {
  public:
    DeltaUploadTable(uint32_t elementSize, const char* name);

    // elementCount elements will be live in `buffer` this frame. bufferReplaced
    // means the device buffer is new and holds nothing the mirror can vouch for.
    void beginFrame(uint32_t elementCount, bool bufferReplaced);

    // Pack target for one element; consumed by the next commit().
    uint8_t* scratch() { return m_scratch.data(); }

    // Element `index` is CPU-owned this frame and its bytes are in scratch().
    // Must be called in ascending index order within a frame.
    //
    // gpuRewritesAfter: a GPU pass later this frame writes this element, so
    // after this frame the mirror no longer vouches for it. The PointInstancer
    // template is the case: the culling shader writes its instance-0 transform
    // back over it, computed in GPU float, which is not bit-equal to the CPU's.
    void commit(uint32_t index, bool gpuRewritesAfter = false);

    // A GPU pass owns this element; the CPU does not write it and the mirror
    // stops vouching for it (PointInstancer duplicate slots).
    void gpuOwned(uint32_t index) { if (index < m_valid.size()) m_valid[index] = 0; }

    // Forget everything the mirror vouches for (scene clear).
    void invalidateAll();

    // Records ONE multi-region transfer for every element committed dirty.
    void upload(DxvkContext* ctx, const Rc<DxvkBuffer>& buffer);

    // THE GATE (I8). Copies the whole device buffer to host memory together
    // with a snapshot of the mirror and of which elements it vouched for, and
    // compares them once the GPU has finished -- no stall, no frame-count
    // guess: harvest() waits on the staging buffer's own use count. Only
    // elements the mirror vouched for are compared, so GPU-owned slots are
    // excluded by construction rather than by a list.
    void scheduleVerify(DxvkContext* ctx, DxvkDevice* device, const Rc<DxvkBuffer>& buffer, uint32_t frameId);
    void harvestVerify(uint32_t frameId);

    struct Stats {
      uint32_t frames = 0;
      uint64_t elements = 0;      // live elements packed
      uint64_t changed = 0;       // elements whose bytes differed (uploaded)
      uint64_t regions = 0;       // copy regions after coalescing
      uint64_t uploadBytes = 0;   // bytes actually transferred
      uint64_t fullBytes = 0;     // what the old whole-table upload would have sent
      uint32_t replaced = 0;      // frames the device buffer was new
      uint32_t verifyReadbacks = 0;
      uint64_t verifyElements = 0;
      uint32_t verifyFail = 0;    // elements that did not match -- MUST stay 0
      uint32_t verifyFailFrame = 0;
    };
    const Stats& stats() const { return m_stats; }
    void resetStats();  // keeps the verify totals, which are cumulative
    const char* name() const { return m_name; }

  private:
    const uint32_t m_elementSize;
    const char* m_name;

    uint32_t m_count = 0;
    std::vector<uint8_t> m_mirror;
    std::vector<uint8_t> m_valid;       // 1: the device buffer holds exactly the mirror's bytes
    std::vector<uint32_t> m_dirty;      // ascending element indices
    std::vector<uint8_t> m_scratch;
    std::vector<uint8_t> m_packed;
    std::vector<VkBufferCopy> m_regions;

    struct VerifySlot {
      Rc<DxvkBuffer> staging;
      std::vector<uint8_t> mirror;
      std::vector<uint8_t> valid;
      uint32_t count = 0;
      uint32_t frame = 0;
      bool pending = false;
    };
    std::array<VerifySlot, 2> m_verify;

    Stats m_stats;
  };

}
