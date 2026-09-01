#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

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
  // EXISTENCE AUTHORITY vs VISIBILITY OBSERVATION.
  // ARCHITECTURE_OVERHAUL.md sec 3.1, invariant I9, build-order slice 5b.
  //
  // THE HAZARD, AND IT IS THE ONE THAT CAN EMPTY THE SCENE.
  //
  // RESIDENT_SCENE_PLAN sec 7.3 measured sub_1801A8350's return as POST-CULL:
  // Gate 1, the view-direction-dependent leaf-frustum test, runs inside it, so
  // its output list is what survived culling. Section 7.5 slice D then
  // specifies that onFrameEnd "touches every record whose handle is listed,
  // draw or no draw, and ABSENT HANDLES INVALIDATE".
  //
  // Those two sentences in the same document are a scene-emptying bug. Wire
  // slice D to a post-cull list and every off-screen object is retired every
  // frame -- residency inverted into a faster version of the exact starvation
  // it was built to fix.
  //
  // AND IT WOULD NOT LOOK LIKE ONE. The drain happens through the RETIREMENT
  // path rather than the reap path, so [ReapJoin] starved= -- the plan's own
  // headline acceptance gate -- would read IMPROVED while the scene emptied.
  // A bug that makes its own detector read better is why this is a type and
  // not a comment.
  //
  // WHY A VERIFICATION STEP IS NOT SUFFICIENT. "We confirmed the list is
  // pre-cull" is a fact about one capture, on one map, at one position.
  // RESIDENT_SCENE_PLAN sec 0.0 already names a sixth cull with no flag at all
  // -- sub_1802EAD60's position-keyed area order list, 30 of 179 areas -- which
  // no configuration defeats and whose behaviour is not uniform across maps. A
  // discipline that has to be re-established per map is a discipline that will
  // be skipped once.
  //
  // SO THE DISTINCTION IS CARRIED BY THE TYPE:
  //
  //   ExistenceSource    may assert ALIVE   may assert DEAD    proven pre-cull
  //   VisibilitySource   may assert VISIBLE MAY NEVER say DEAD everything else
  //
  // invalidateAbsent() takes an ExistenceSource and nothing else, and
  // ExistenceSource has no public constructor. A post-cull list therefore
  // cannot reach the retirement path -- not "must not", CANNOT. This is the
  // sec 2.1 lesson applied again: find the place where the question answers
  // itself rather than writing a rule somebody has to remember.
  //
  // DO NOT DISCARD A POST-CULL SOURCE -- TYPE IT AND USE IT. A VisibilitySource
  // is the engine's own visibility answer, which is strictly better information
  // than sceneCull's frustum-and-light-influence test (sec 1.4: culled=0,
  // because lightAllKeep covers all 1298 off-screen instances). It is worth
  // having as a culling input. It is just not allowed to kill anything.
  // ==========================================================================
  class EnumerationSource {
  public:
    const char* name() const { return m_name; }

    // Start a frame's list. The previous frame's contents are dropped -- a
    // union across frames would make "absent" unreachable, which is the one
    // question the list exists to answer.
    void beginFrame(uint32_t frame);

    // The engine named this handle this frame.
    void note(uint64_t handle);

    bool listed(uint64_t handle) const;
    uint32_t listedCount() const { return static_cast<uint32_t>(m_listed.size()); }
    uint32_t frame() const { return m_frame; }

  protected:
    explicit EnumerationSource(const char* name) : m_name(name) { }

    const char* m_name = "";
    uint32_t m_frame = kInvalidFrameIndex;
    std::unordered_set<uint64_t> m_listed;
  };

  // Post-cull, or pre-cull but unproven. Constructible by anyone, because the
  // safe classification must be the easy one.
  class VisibilitySource final : public EnumerationSource {
  public:
    explicit VisibilitySource(const char* name) : EnumerationSource(name) { }
  };

  // Proven pre-cull. The ONLY type invalidateAbsent() accepts.
  //
  // No public constructor and no conversion from VisibilitySource. The only way
  // to obtain one is promoteToExistenceSource(), which will not hand one back
  // until the flatness evidence has actually been collected -- so "we believe
  // this list is pre-cull" cannot be expressed in code, only "this list has
  // been measured flat for N frames".
  class ExistenceSource final : public EnumerationSource {
  public:
    ExistenceSource(const ExistenceSource&) = delete;
    ExistenceSource& operator=(const ExistenceSource&) = delete;

  private:
    explicit ExistenceSource(const char* name) : EnumerationSource(name) { }
    friend class ExistenceSourcePromotion;
  };

  // THE PROMOTION GATE, and it is the one slice B already has: listed= FLAT
  // across a fixed-position pitch-and-yaw sweep.
  //
  // A fixed camera position means the true object population cannot change, so
  // any movement in listedCount() is culling leaking into the list. Flat across
  // the sweep is therefore direct evidence the list is pre-cull; anything else
  // is direct evidence it is not. Until it passes ON THE MAP UNDER TEST the
  // source stays a VisibilitySource and residency keeps whatever liveness it
  // had.
  class ExistenceSourcePromotion {
  public:
    // Frames of unchanging listedCount required before promotion. Long enough
    // that a sweep genuinely covers pitch and yaw rather than a pause in the
    // middle of one.
    static constexpr uint32_t kRequiredFlatFrames = 600u;

    // Feed one frame's count while the camera sweeps from a FIXED POSITION.
    // Moving the camera invalidates the evidence, which is why this is fed by
    // the sweep harness rather than by the enumeration hook itself.
    void observe(uint32_t listedCount);

    bool flat() const { return m_flatFrames >= kRequiredFlatFrames; }
    uint32_t flatFrames() const { return m_flatFrames; }
    uint32_t breaks() const { return m_breaks; }

    // Hands back an ExistenceSource ONLY once flat() holds. Returns nullptr
    // otherwise, and a caller that does not check gets no source rather than an
    // unproven one.
    std::unique_ptr<ExistenceSource> promote(const char* name) const;

    void reset();

  private:
    uint32_t m_lastCount = ~0u;
    uint32_t m_flatFrames = 0u;
    uint32_t m_breaks = 0u;
  };

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
      // A FOURTH WAS ADDED BY MEASUREMENT, and it is excluded for a different
      // reason than the three above:
      //
      //   blending on  a translucent surface's appearance is a function of the
      //                scene BEHIND it, and residency retains only the sparse
      //                subset of the scene that was actually drawn. Held
      //                translucent geometry therefore composites against
      //                whatever survived rather than against the scene it was
      //                authored over. Nothing is rebuilt per frame here -- the
      //                dependency is on the rest of the frame, which a record
      //                cannot hold by definition.
      //
      // Measured off the instances at build time rather than guessed from the
      // draw, and stored so the touch can refuse in O(1). This is the plan's own
      // Transient class, derived from what the instances turned out to be
      // instead of declared from what a shader looks like -- and the fourth
      // entry is what that design was FOR: it arrived as a visual bug, was named
      // by [HeldRaw] dumping properties rather than by a shader allowlist, and
      // cost one clause.
      bool skipUnsafe = false;
      // DIAGNOSTIC BASELINE for [HeldRaw], captured off the first instance's
      // geometry at build time.
      //
      // The question they answer cannot be answered without a baseline: a held
      // instance keeps pointing at its BLAS, but nothing here proves the BLAS's
      // vertex CONTENT is still the content the record was built against. If
      // the geometry is regenerated each frame -- skinning being the obvious
      // case, since a skinned mesh's positions are produced per draw -- then a
      // held instance with no draw renders whatever was last written into that
      // buffer, which need not be its own mesh.
      //
      // builtBoneHash non-zero identifies the skinned case directly;
      // builtPosHash compared against the live hash catches any other producer
      // of the same failure without having to name it first.
      uint64_t builtPosHash = 0ull;
      uint64_t builtBoneHash = 0ull;
      // WHICH BlasEntry the baseline above was taken from. The content check is
      // only meaningful against that same entry: a record whose instances span
      // several BLASes would otherwise have one instance's hash validating all
      // of them, which passes every instance except the one it was measured on.
      // Held as a raw address for comparison only and never dereferenced.
      const void* builtBlas = nullptr;
      // NV-DXVK [ExistenceSource] slice 5b: the engine handle this record's
      // object was enumerated under, or 0 when none is known.
      //
      // 0 IS NOT "DEAD" AND MUST NEVER BE TREATED AS IT. Until sec 7 slice B
      // lands there is no handle producer at all, so every record carries 0 --
      // and invalidateAbsent() therefore skips every record, which is exactly
      // the right behaviour for a scene nothing can vouch for. A record with no
      // handle is OUTSIDE the enumeration's authority, which is a different
      // statement from being absent from its list.
      uint64_t engineHandle = 0ull;
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
    bool score(uint64_t key, const std::vector<RtInstance*>& produced, uint32_t ordinal);

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

    // NV-DXVK [ExistenceSource] slice 5b: RETIRE WHAT THE ENGINE NO LONGER
    // LISTS -- RESIDENT_SCENE_PLAN sec 7.5 slice D, with sec 3.1's type on it.
    //
    // Takes an ExistenceSource BY TYPE, not by convention. That is the entire
    // point of the parameter: a VisibilitySource will not convert, so the
    // scene-emptying wiring sec 3.1 describes DOES NOT COMPILE. If you are
    // holding a VisibilitySource and want to call this, the answer is not a
    // cast -- it is to run the fixed-position pitch-and-yaw sweep and promote
    // it through ExistenceSourcePromotion.
    //
    // Records with engineHandle == 0 are SKIPPED, not invalidated -- see the
    // field. That makes this a no-op today, which is correct: there is no
    // handle producer until sec 7 slice B, and a source that cannot identify a
    // record has no authority to retire it.
    //
    // Returns how many records were invalidated.
    uint32_t invalidateAbsent(const ExistenceSource& source, uint32_t frame);

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
      // what the full path actually resolved.
      //
      // THE ARMING GATE IS fail - failNoRecEmpty, NOT fail, AND THE DIFFERENCE
      // IS NOT A TOLERANCE. A draw that resolves to no instance is never filed:
      // processDrawCallState skips build() on an empty list, deliberately,
      // because a record serving an empty touch reads as a hit and does nothing.
      // The inputs of such a draw are perfectly stable, so the gate predicts a
      // hit on it every frame and score() finds no record every frame. That
      // population -- the sky pass, the fog registration, the terrain bake --
      // contributes a fixed non-zero FAIL for as long as the level is loaded,
      // and no change to the key, the folds or the record store can move it.
      // Reading the unsplit total as the arming gate means waiting for a zero
      // that cannot arrive.
      //
      // So the four causes are counted apart, because they want four different
      // responses:
      //
      //   failNoRecEmpty  no record, and the full path produced no instances
      //                   either -- the side-effect class. BENIGN AND
      //                   PERMANENT: the live path already handles it, since
      //                   touch() finds nothing and the draw commits in full.
      //                   Expected steady and non-zero.
      //   failNoRecLost   no record, but the full path DID produce instances.
      //                   A record was filed and went away between the two
      //                   frames. Read evicted=, wiped= and invalidated= on the
      //                   same line first: if those are moving, the eviction
      //                   policy is the story rather than the key.
      //   failSize        a record was there and named a DIFFERENT NUMBER of
      //                   instances than the draw just produced. This is the
      //                   occurrence ordinal sliding when engine culling removes
      //                   one copy of a multi-copy identity (trap 2).
      //   failMember      same count, different instances, position for
      //                   position. The strictest failure, and the one the
      //                   ordinal exists to prevent.
      //
      // failSize + failMember are the two that would have kept the wrong
      // objects alive and let the right ones retire. THOSE must reach zero
      // across a full pitch-and-yaw sweep before verify goes off.
      uint32_t predicted  = 0;
      uint32_t fail       = 0;
      uint32_t failNoRecEmpty = 0;
      uint32_t failNoRecLost  = 0;
      uint32_t failSize       = 0;
      uint32_t failMember     = 0;
      // The two second-level splits, each answering a question its parent count
      // cannot -- see the bodies in score() for what each one implies.
      //
      // failLostErased is the one to read sceptically: it is the count of
      // records that were filed and then erased, which while verify is on is
      // largely residency being disabled rather than residency being wrong.
      uint32_t failLostErased = 0;
      uint32_t failLostNever  = 0;
      // AND THE SPLIT THAT DECIDES WHETHER failSize IS A FAILURE AT ALL.
      //
      // The paragraph above says failSize + failMember "would have kept the
      // wrong objects alive AND let the right ones retire". Those are two
      // different errors and failSize does not separate them, but the sign of
      // the count difference does, and touch() is what makes the sign mean
      // something: it stamps every instance the RECORD names.
      //
      //   failSizeOver   the record names MORE than the draw produced, so the
      //                  extra instances get stamped and survive a frame no
      //                  draw touched them. That is the FEATURE, stated in
      //                  rtx.residentScene.enable's own description, not a
      //                  defect -- unless the record belongs to another object,
      //                  which ord= and shared= are what rule out.
      //   failSizeUnder  the record names FEWER, so instances the draw DID
      //                  resolve to are never stamped -- and a gate hit skips
      //                  the full path, so nothing else stamps them either.
      //                  They age and retire. That is the real defect.
      //
      // COUNTED, NOT SUBTRACTED. realFail deliberately still includes both:
      // narrowing a correctness gate is how a gate gets met without the bug
      // being fixed. These exist so the decision to narrow it can be argued
      // from numbers, and so it is somebody's decision rather than a silent
      // redefinition. Measured 2026-08-23 before the split: 187 over against
      // 210 under, on failures that were 99% ord=0 and ~95% overlapping.
      uint32_t failSizeOver   = 0;
      uint32_t failSizeUnder  = 0;
      // NOT A FAILURE, and it is outside `fail` on purpose -- see score(). The
      // record named the same instances in a different order, and touch()
      // consumes the list as a set, so the right objects are kept alive. It is
      // reported because unstable resolution order is worth knowing about, and
      // because it is the number that would matter if a consumer ever became
      // positional.
      uint32_t memberPerm = 0;
      // Records retired because the engine freed the buffers they were built
      // from. CUMULATIVE, like wiped: this is the signal that keeps a destroyed
      // object from ghosting, and a policy that never fires needs to be visible
      // as never having fired rather than as a zero in the current window.
      uint32_t sourceDestroyed = 0;
      // THE TWO NUMBERS THAT MAKE sourceDestroyed READABLE. Alone it reads 0 for
      // two opposite reasons and cannot distinguish them:
      //
      //   srcNotices=0                   ~D3D11Buffer never runs for this
      //                                  geometry. TF2 pools and reuses its
      //                                  buffers, so the object dies and the
      //                                  buffer does not -- which makes buffer
      //                                  death the wrong death signal for this
      //                                  engine, not a broken one.
      //   srcNotices>0, srcDrained>0,    buffers die constantly and none of them
      //   sourceDestroyed=0              is one a record was built from. The
      //                                  join is broken and is a bug to find.
      //
      // Both CUMULATIVE, like wiped and sourceDestroyed: a policy that never
      // fires has to be visible as never having fired rather than as a zero in
      // the current window.
      uint32_t srcNotices = 0;
      uint32_t srcDrained = 0;
      // NV-DXVK [ExistenceSource] slice 5b. CUMULATIVE, same argument as wiped
      // and sourceDestroyed.
      //
      // WATCH THIS ONE HARDER THAN THE OTHERS. It is the counter for the only
      // path in this file that can empty the scene, and sec 3.1's whole point
      // is that the failure would make [ReapJoin] starved= read BETTER while it
      // happened. absentRetired climbing while starved= falls is that failure,
      // and it is the only place the two can be compared.
      //
      // absentSkipped is its denominator: records the source had no authority
      // over because they carry no engine handle. While it equals the record
      // count, invalidateAbsent is a no-op and cannot have done any harm.
      uint32_t absentRetired = 0;
      uint32_t absentSkipped = 0;
    };

    const Stats& stats() const { return m_stats; }
    void resetStats() {
      const uint32_t keepWiped = m_stats.wiped;
      const uint32_t keepSourceDestroyed = m_stats.sourceDestroyed;
      const uint32_t keepSrcNotices = m_stats.srcNotices;
      const uint32_t keepSrcDrained = m_stats.srcDrained;
      const uint32_t keepAbsentRetired = m_stats.absentRetired;
      const uint32_t keepAbsentSkipped = m_stats.absentSkipped;
      m_stats = Stats();
      m_stats.wiped = keepWiped;
      m_stats.sourceDestroyed = keepSourceDestroyed;
      m_stats.srcNotices = keepSrcNotices;
      m_stats.srcDrained = keepSrcDrained;
      m_stats.absentRetired = keepAbsentRetired;
      m_stats.absentSkipped = keepAbsentSkipped;
    }

    size_t size() const { return m_records.size(); }

  private:
    // Records this store erased, and the frame it happened on. DIAGNOSTIC ONLY:
    // it exists so score() can tell "this key's record was filed and then went
    // away" from "this key never had a record", which are opposite findings that
    // the missing-record count alone reports identically.
    //
    // Kept for a few frames and no longer. The question it answers is always
    // about the immediately preceding frame -- the gate only predicts on a key
    // it judged last frame -- so a long history would cost memory to answer
    // nothing, and an unbounded one would be a leak in exactly the churning
    // scene where it fills fastest.
    static constexpr uint32_t kTombstoneFrames = 4u;
    void recordTombstone(uint64_t key, uint32_t frame);

    std::unordered_map<uint64_t, Record> m_records;
    std::unordered_map<uint64_t, uint32_t> m_tombstones;
    Stats m_stats;
  };

}
