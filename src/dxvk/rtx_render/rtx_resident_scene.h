#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include "rtx_constants.h"

namespace dxvk {

  class RtInstance;

  // Bumped by SceneManager::clear(), which is the tree's existing "the scene
  // this was all built from is gone" event -- a level change reaches it through
  // the camera-cut delayed clear, and a replacement reload reaches it directly.
  //
  // IT EXISTS FOR THE SEED PASS, and the seed pass is the reason residency is
  // invariant rather than view-history dependent. Residency can only ever hold
  // objects that were drawn at least once, so left alone the resident set is
  // "everything the player happened to look at" -- which grows under a
  // pitch-and-yaw sweep instead of staying flat, and fails the plan's own
  // acceptance gate. Seeding replaces that with one bounded unculled burst at
  // load, and this counter is how the frame thread learns a burst is due.
  //
  // An epoch rather than a flag: the frame thread compares against the value it
  // last acted on, so a clear that happens while it is not looking cannot be
  // missed, and two clears in quick succession arm the seed exactly once.
  extern std::atomic<uint32_t> g_sceneEpoch;

  // THE DEATH SIGNAL, and it is load-bearing rather than tidy.
  //
  // A resident instance is exempt from lifetime expiry: that is the entire
  // feature, and it is correct for world geometry and static props, which cannot
  // cease to exist while the level is loaded. It is WRONG for anything that can
  // be destroyed. Without a death signal a killed entity's geometry would stay
  // in the ray-traced scene, casting shadows and appearing in reflections, until
  // its record happened to be evicted -- a ghost, with nothing in the log to say
  // so, which is the worst shape of bug this feature could produce.
  //
  // The engine freeing an object's vertex or index buffer IS that object ceasing
  // to exist, and unlike anything else about the object's fate, DXVK observes it
  // directly: the D3D11Buffer is refcounted and its destructor runs. So this is
  // reported from ~D3D11Buffer, batched, and drained once per frame by
  // ResidentScene::onFrameEnd, which invalidates every record built from that
  // buffer. Their instances then retire on the ordinary lifetime clause.
  //
  // CALLED FROM A DESTRUCTOR, so it must be safe during process teardown and
  // must cost nothing when residency has never been armed -- see the body.
  void noteResidentSourceBufferDestroyed(uint64_t bufferPtr);

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
  //   frame thread   D3D11Rtx::m_residentGate      key -> { folds, frameLastSeen }
  //                  Sees every draw. Owns the PREDICTION -- is this draw asking
  //                  for anything different from last frame. Never touches an
  //                  RtInstance: instances are CS/RT-owned and reading one from
  //                  the frame thread is the getImageHash / s_zigGunInstance race
  //                  class all over again.
  //
  //   RT / CS side   ResidentScene (this file)     key -> { instances, valid }
  //                  Owns the RtInstance* list, the stamp, and the DECISION.
  //                  A prediction is acted on only if a valid record is actually
  //                  found here, which is why a draw that commits for a side
  //                  effect and resolves to no instance can never be skipped.
  //
  // The two are joined only by a uint64 key travelling on the DrawCallState
  // through the existing EmitCs stream, so neither map is ever touched by two
  // threads and there is nothing to lock. The one piece of state that IS shared
  // across threads is the buffer-death queue behind
  // noteResidentSourceBufferDestroyed, which carries its own lock because it is
  // written from whichever thread happens to release a D3D11 buffer.
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
      // THE CAMERA SET THE INSTANCES CARRIED WHEN THE RECORD WAS BUILT.
      //
      // setFrameLastUpdated() CLEARS m_seenCameraTypes on the first stamp of a
      // frame (trap 1), so the touch has to put it back or portal and view-model
      // logic silently loses the instance. The frame thread cannot supply it:
      // the camera is classified on the CS side from the DrawCallState, and the
      // gate deliberately runs before any of that exists. So the record carries
      // it, captured off the instances themselves at build time.
      //
      // Replaying the last known set is not an approximation of a better answer
      // -- it is the only answer there is. A draw that did not arrive cannot say
      // which camera it would have been classified under, and the instance is
      // being kept exactly as it last was.
      uint32_t cameraMask = 0u;
      // The engine buffers this record's draw was made of, as raw addresses
      // rather than references -- see noteResidentSourceBufferDestroyed. Holding
      // a reference would keep the very object alive whose death is the signal.
      // Both are kept because either one dying means the object is gone.
      uint64_t srcVertexBuffer = 0ull;
      uint64_t srcIndexBuffer = 0ull;
      // NOT EVERY DRAW IS SKIPPABLE, AND THE RECORD IS WHERE THAT IS DECIDED.
      //
      // The gate proves a draw is asking for the same GEOMETRY, TRANSFORM and
      // MATERIAL as last frame. That is not the same as proving the draw has
      // nothing else to do. Three things this tree rebuilds from scratch every
      // frame are attached to the instance rather than to any of those three,
      // so an instance carrying one of them must go down the full path however
      // unchanged its inputs are:
      //
      //   billboards   createBillboards / createBeams append into a per-frame
      //                vector that is cleared each frame, so a skipped particle
      //                or beam simply is not drawn.
      //   ray portals  processRayPortalData registers the portal each frame.
      //   decals       decalSortOrder is a per-frame counter that approximates
      //                draw order on the GPU; a stale one misorders the decal.
      //
      // Measured off the instances at build time rather than guessed from the
      // draw, and stored so the touch can refuse in O(1). This is the plan's own
      // Transient class, derived from what the instances turned out to be
      // instead of declared from what a shader looks like.
      bool skipUnsafe = false;
      std::vector<RtInstance*> instances;
    };

    ResidentScene() = default;

    // -- record store, RT side only ------------------------------------------

    Record* find(uint64_t key);
    const Record* find(uint64_t key) const;

    // Replaces the record's instance list wholesale and stamps the back-pointer
    // on every instance in it. Called after a full-path draw resolves.
    //
    // Takes a const reference and copies into the record's existing vector
    // rather than taking ownership of a fresh one: this runs per full-path draw
    // (~538/frame today, and every draw while verify is on), and assigning into
    // the record reuses its capacity instead of allocating each time.
    void build(uint64_t key,
               uint64_t srcGenHash,
               uint64_t srcVertexBuffer,
               uint64_t srcIndexBuffer,
               uint32_t frame,
               const std::vector<RtInstance*>& instances);

    // THE TOUCH. Keep-alive without reprocessing: stamps frameLastUpdated on
    // every instance in the record, replays the camera set the build captured,
    // and stamps the geometry entry each instance is linked to. Returns false
    // if the key is unknown or the record was invalidated, in which case the
    // caller MUST fall back to the full path -- a failed touch that is treated
    // as a success is a silent retirement.
    bool touch(uint64_t key, uint32_t frame);

    // VERIFY SCORING. Answers the only question that matters while verify is on:
    // if the gate had skipped this draw, would the record it served have named
    // the instances the full path actually produced? Returns true when they
    // agree. A disagreement is a FAIL -- the touch would have kept the wrong
    // instances alive and let the right ones retire, which is the "resident but
    // not refreshed" failure with nothing else to catch it.
    bool score(uint64_t key, const std::vector<RtInstance*>& produced);

    // Is this instance currently held resident? O(1).
    //
    // THE GEOMETRY HAS TO OUTLIVE THE INSTANCE, and this is what lets the BLAS
    // garbage collector ask. rtx.numFramesToKeepBLAS is 1, so a BlasEntry is
    // destroyed the frame after its last draw -- and onSceneObjectDestroyed
    // marks every linked instance for collection, which residency's keep clause
    // deliberately does NOT override (an instance must not outlive its
    // geometry). Without this, an off-screen object would have its instances
    // faithfully exempted from lifetime expiry and then reaped anyway one frame
    // later, through the geometry, and residency would do nothing at all for the
    // case it exists for.
    bool holdsInstance(const RtInstance* instance) const;

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
      // The split that makes touchMiss readable -- see ResidentScene::touch.
      // unknown is ordinary and self-correcting, and also covers the draws that
      // have no record by design; invalid means a record's instance died and is
      // the one the acceptance gate is about.
      uint32_t touchMissUnknown = 0;
      uint32_t touchMissInvalid = 0;
      // Refused because the record's instances carry per-frame work the gate
      // cannot speak for -- billboards, ray portals or decals. Not a defect and
      // not self-correcting either: it is a permanent property of those draws.
      uint32_t touchMissUnsafe = 0;
      uint32_t invalidated = 0;
      uint32_t evicted    = 0;
      // CUMULATIVE across the session and deliberately NOT reset with the rest:
      // a wipe that happens and then reads 0 by the time the line comes out is
      // how an eviction policy hides. Non-zero = maxRecords too small for the
      // scene, which is the OPPOSITE finding from key churn.
      uint32_t wiped      = 0;
      uint32_t instancesStamped = 0;
      // VERIFY TALLIES. predicted counts draws the frame-thread gate said it
      // could have skipped; fail counts those where the record disagreed with
      // what the full path actually resolved. failNoRecord is the subset where
      // there was no valid record to serve at all, which is a different defect
      // (the gate and the record store disagree about what exists) from a
      // record naming the wrong instances.
      //
      // FAIL=0 ACROSS A PITCH-AND-YAW SWEEP IS THE GATE FOR TURNING verify OFF.
      // Not a low rate -- zero. Every failure is an object that would have been
      // held resident on stale contents.
      uint32_t predicted  = 0;
      uint32_t fail       = 0;
      uint32_t failNoRecord = 0;
      // Records retired because the engine freed the buffers they were built
      // from. CUMULATIVE, like wiped: this is the signal that keeps a destroyed
      // object from ghosting, and a policy that never fires needs to be visible
      // as never having fired rather than as a zero in the current window.
      uint32_t sourceDestroyed = 0;
    };

    const Stats& stats() const { return m_stats; }
    void resetStats() {
      const uint32_t keepWiped = m_stats.wiped;
      const uint32_t keepSourceDestroyed = m_stats.sourceDestroyed;
      m_stats = Stats();
      m_stats.wiped = keepWiped;
      m_stats.sourceDestroyed = keepSourceDestroyed;
    }

    size_t size() const { return m_records.size(); }

  private:
    std::unordered_map<uint64_t, Record> m_records;
    Stats m_stats;
  };

}
