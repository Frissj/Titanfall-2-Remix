#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>

#include "rtx_constants.h"

namespace dxvk {

  class RtInstance;

  // ==========================================================================
  // THE RESIDENT SCENE -- RT-SIDE HALF. See RESIDENT_SCENE_PLAN.md.
  //
  // WHAT THIS IS FOR. With engine culling back on (rtx.cullOff.enable=False) a
  // draw stops arriving for anything off-screen, and garbageCollection reaps on
  //     m_frameLastUpdated + instanceKeepN <= currentFrame
  // and nothing else -- so one missed frame retires the instance and the
  // geometry leaves the RT scene entirely, taking its shadow, its reflection
  // and its indirect contribution with it. Measurable today as
  // [ReapJoin] respawn=0 starved=1..22/frame, which is this file's own
  // definition of "the geometry received fewer draws than it had instances".
  //
  // Because reaping reads ONLY that frame id, "keep alive without reprocessing"
  // is expressible as a bulk frame-id stamp rather than a redesign of GC. That
  // is exactly what FanoutBatchRecord already does for the fanout path; this is
  // the same mechanism under a second key, covering every draw class.
  //
  // NOT ANTI-CULLING. Nothing here reads or writes rtx.antiCulling.*,
  // AntiCulling::isObjectAntiCullingEnabled, or
  // InstanceCategories::IgnoreAntiCulling. The IgnoreAntiCulling long-keep
  // clause in garbageCollection is untouched; residency adds its own clause
  // beside it, keyed on m_residentKey.
  //
  // WHY TWO MAPS, ONE PER THREAD, AND NO LOCK. The decision ("is this draw
  // unchanged?") and the action ("keep its instances alive") happen on
  // different threads and need different data:
  //
  //   frame thread   D3D11Rtx::ResidentGateIndex   key -> { srcGenHash, frameLastSeen }
  //                  Sees every draw. Owns the hit/miss verdict. Never touches
  //                  an RtInstance -- instances are CS/RT-owned and reading one
  //                  from the frame thread is the getImageHash/s_zigGunInstance
  //                  race class all over again.
  //
  //   RT / CS side   ResidentScene (this file)     key -> { instances, valid }
  //                  Owns the RtInstance* list and the stamp.
  //
  // The two are joined only by a uint64 key travelling through the existing
  // EmitCs stream, so neither map is ever touched by two threads and there is
  // nothing to lock.
  //
  // THE LIFETIME CONTRACT IS THE BACK-POINTER, NOT A SCAN. A record caches raw
  // RtInstance* across frames and GC deletes instances, so without a
  // back-pointer this is a dangling-pointer generator -- the exact failure this
  // tree has already shipped twice (the file-static s_zigGunInstance deref and
  // the GC-walk incRef race). RtInstance::m_residentKey carries the key back,
  // which makes invalidateFor() O(1) and, more importantly, TOTAL: an instance
  // cannot be destroyed without coming through removeInstance, and
  // removeInstance invalidates in the same call.
  //
  // INVALIDATE THE WHOLE RECORD, NEVER ONE ELEMENT. The list is only meaningful
  // as the complete output of one resolution pass; a list with a hole in it
  // would stamp some of an object's instances and silently let the rest retire.
  // Cheap to rebuild, impossible to half-trust. Same rule as
  // invalidateFanoutRecordFor().
  // ==========================================================================
  class ResidentScene {
  public:
    struct Record {
      // The generation fold this record was last BUILT against. The frame
      // thread holds its own copy for the gate; this one exists so a rebuild
      // can be distinguished from a touch when reading the stats.
      uint64_t srcGenHash = 0ull;
      uint32_t frameLastSeen  = kInvalidFrameIndex;
      uint32_t frameLastBuilt = kInvalidFrameIndex;
      // Cleared by invalidateFor() / BLAS teardown. An invalid record is kept
      // (not erased) until the LRU takes it, so the frame thread's index and
      // this map cannot disagree about whether a key exists.
      bool valid = false;
      std::vector<RtInstance*> instances;
    };

    ResidentScene() = default;

    // -- record store, RT side only ------------------------------------------

    Record* find(uint64_t key);
    const Record* find(uint64_t key) const;

    // Creates on miss. Returned reference is valid until the next obtain() or
    // onFrameEnd(); callers must not hold it across either.
    Record& obtain(uint64_t key);

    // Replaces the record's instance list wholesale and stamps the back-pointer
    // on every instance in it. Called after a full-path draw resolves.
    void build(uint64_t key,
               uint64_t srcGenHash,
               uint32_t frame,
               std::vector<RtInstance*>&& instances);

    // THE TOUCH. Keep-alive without reprocessing: stamps frameLastUpdated on
    // every instance in the record and re-registers the camera. Returns false
    // if the key is unknown or the record was invalidated, in which case the
    // caller MUST fall back to the full path -- a failed touch that is treated
    // as a success is a silent retirement.
    bool touch(uint64_t key, uint32_t frame, uint32_t cameraType);

    // O(1) and TOTAL -- see the class comment.
    void invalidateFor(const RtInstance* instance);

    // Erases invalidated records, then evicts down to rtx.residentScene.maxRecords
    // on the age-rung ladder (the [Perf.SplitXf] evict{} policy -- see the body
    // for why a single fixed age bound cannot fire and a sort-LRU is wasted work).
    // Call once per frame from the RT side.
    void onFrameEnd(uint32_t frame);

    void clear();

    // -- stats ----------------------------------------------------------------

    struct Stats {
      uint32_t live       = 0;
      uint32_t built      = 0;   // full-path resolutions that (re)built a record
      uint32_t touched    = 0;   // gate hits that stamped without reprocessing
      uint32_t touchMiss  = 0;   // touch() returned false -- fell back to full path
      uint32_t invalidated = 0;
      uint32_t evicted    = 0;
      // CUMULATIVE across the session and deliberately NOT reset with the rest:
      // a wipe that happens and then reads 0 by the time the line comes out is
      // how an eviction policy hides. Non-zero = maxRecords too small for the
      // scene, which is the OPPOSITE finding from key churn.
      uint32_t wiped      = 0;
      uint32_t instancesStamped = 0;
    };

    const Stats& stats() const { return m_stats; }
    void resetStats() {
      const uint32_t keepWiped = m_stats.wiped;
      m_stats = Stats();
      m_stats.wiped = keepWiped;
    }

    size_t size() const { return m_records.size(); }

  private:
    std::unordered_map<uint64_t, Record> m_records;
    Stats m_stats;
  };

}
