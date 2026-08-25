#include "rtx_resident_scene.h"

#include <algorithm>
#include <iterator>
#include <mutex>
#include <unordered_set>

#include "../../util/thread.h"

#include "rtx_instance_manager.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  std::atomic<uint32_t> g_sceneEpoch { 0u };

  namespace {
    struct SourceBufferDeaths {
      dxvk::mutex mutex;
      std::unordered_set<uint64_t> pending;
    };

    // LEAKED ON PURPOSE, and the reason is written into ~D3D11Buffer already:
    // buffer destructors run late in process teardown, after namespace-scope
    // objects in another translation unit have been destroyed, and that hazard
    // has bitten this tree once (the vanish_diag counter note). An allocation
    // that is never destroyed cannot be used after free. The function-local
    // static holding it is a plain pointer, so its own teardown is trivial.
    SourceBufferDeaths& sourceBufferDeaths() {
      static SourceBufferDeaths* s_deaths = new SourceBufferDeaths();
      return *s_deaths;
    }

    // Non-zero once any record has been built, i.e. once residency is genuinely
    // in use. ~D3D11Buffer runs on the game thread for every buffer the engine
    // frees, which during streaming is a hot path, so with residency off the
    // notice below must cost one relaxed load and nothing else -- no lock, no
    // allocation, no set insertion. Constant-initialised with a trivial
    // destructor, so it is readable at any point in the process's life.
    std::atomic<uint32_t> g_residentTracking { 0u };

    // Cumulative count of death notices that got past the armed guard, i.e.
    // buffers the engine actually freed while residency was live. Read against
    // Stats::sourceDestroyed to tell "the signal never arrives" from "the signal
    // arrives and never matches" -- see noteResidentSourceBufferDestroyed.
    std::atomic<uint32_t> g_srcDeathNotices { 0u };
  }

  void noteResidentSourceBufferDestroyed(uint64_t bufferPtr) {
    if (bufferPtr == 0ull || g_residentTracking.load(std::memory_order_relaxed) == 0u) {
      return;
    }
    SourceBufferDeaths& deaths = sourceBufferDeaths();
    std::lock_guard<dxvk::mutex> lock(deaths.mutex);
    // Bounded: if the drain has stopped happening (residency turned off with
    // records still standing, or a level teardown freeing everything at once)
    // this must not grow without limit. Dropping notices is safe in the only
    // direction that matters -- a missed death leaves a record standing until
    // the LRU takes it, where an unbounded set would be a memory leak.
    // COUNTED HERE, PAST THE ARMED GUARD, and that placement is the point.
    //
    // srcDied has read 0 for entire sessions and that single number cannot say
    // which of two opposite things is true: the destructor never runs for these
    // buffers, or it runs constantly and none of them belongs to a record. The
    // first would mean TF2 pools its geometry and never frees it, which makes
    // ~D3D11Buffer the wrong death signal for this engine entirely. The second
    // would mean the join is broken and is a bug to find.
    //
    // Past the g_residentTracking guard rather than before it, so a run with
    // residency off still pays nothing -- this is called for every buffer the
    // engine frees, which during streaming is a hot path. The guard is only
    // false before the first record exists, so nothing that matters is missed.
    g_srcDeathNotices.fetch_add(1u, std::memory_order_relaxed);
    if (deaths.pending.size() < 65536u) {
      deaths.pending.insert(bufferPtr);
    }
  }

  ResidentScene::Record* ResidentScene::find(uint64_t key) {
    if (key == 0ull) {
      return nullptr;
    }
    const auto it = m_records.find(key);
    return (it == m_records.end()) ? nullptr : &it->second;
  }

  const ResidentScene::Record* ResidentScene::find(uint64_t key) const {
    if (key == 0ull) {
      return nullptr;
    }
    const auto it = m_records.find(key);
    return (it == m_records.end()) ? nullptr : &it->second;
  }

  void ResidentScene::build(uint64_t key,
                            uint64_t srcGenHash,
                            uint64_t srcVertexBuffer,
                            uint64_t srcIndexBuffer,
                            uint32_t frame,
                            const std::vector<RtInstance*>& instances) {
    if (key == 0ull) {
      // 0 is the "no record" sentinel on RtInstance::m_residentKey, exactly as
      // it is on m_batchRecordKey. A record under key 0 could never be
      // invalidated through the back-pointer, so it must not exist.
      return;
    }

    Record& rec = m_records[key];

    // Drop the back-pointers of the PREVIOUS occupants first. An instance that
    // was in the old list and is not in the new one would otherwise keep
    // naming this key, and a later removeInstance would invalidate a record
    // that no longer describes it -- which is harmless on its own, but it also
    // means the instance itself never gets its stale key cleared, so the key
    // survives into whatever record later hashes to it.
    for (RtInstance* inst : rec.instances) {
      if (inst != nullptr && inst->m_residentKey == key) {
        inst->m_residentKey = 0ull;
      }
    }

    rec.instances.assign(instances.begin(), instances.end());
    rec.srcGenHash = srcGenHash;
    rec.srcVertexBuffer = srcVertexBuffer;
    rec.srcIndexBuffer = srcIndexBuffer;
    rec.frameLastSeen = frame;
    rec.frameLastBuilt = frame;
    rec.valid = true;

    // Arm the death notice. Doing it here rather than at construction means a
    // run with residency off never pays for it, and a run with residency on
    // starts paying exactly when there is something for it to protect. Monotonic
    // by design: it is a "has ever been used" flag, not a live count, because a
    // notice arriving after the last record was erased is harmless while one
    // suppressed before the next record is built is a ghost.
    g_residentTracking.store(1u, std::memory_order_relaxed);

    // Capture the camera set off the instances, for the touch to replay. Taken
    // here rather than passed in because this is the one place the answer is
    // known: the instances have just been through the full path, so their
    // m_seenCameraTypes is exactly what this draw registered.
    rec.cameraMask = 0u;
    rec.skipUnsafe = false;
    rec.builtPosHash = 0ull;
    rec.builtBoneHash = 0ull;
    rec.builtBlas = nullptr;
    for (RtInstance* inst : rec.instances) {
      if (inst != nullptr) {
        inst->m_residentKey = key;
        rec.cameraMask |= inst->getSeenCameraMask();
        // See Record::skipUnsafe. The first three name per-frame work that hangs
        // off the instance rather than off the geometry, the transform or the
        // material, so the gate cannot speak for it.
        //
        // BLENDING IS THE FOURTH, AND IT IS A DIFFERENT ARGUMENT FROM THE OTHER
        // THREE. Those are excluded because something is rebuilt each frame that
        // a record does not hold. A blended surface is excluded because of what
        // it IS: its appearance is a function of the scene BEHIND it, and
        // residency by construction retains a sparse subset of the scene -- only
        // what has actually been drawn. So a held translucent surface composites
        // against whatever else happened to survive rather than against the
        // scene it was authored over, and the error grows the longer it is held.
        // An opaque surface has no such dependency, which is exactly why it is
        // safe to keep and a blended one is not.
        //
        // MEASURED, not anticipated. [HeldRaw] over 2136 held instances: 250
        // were blended with particle=0 decal=0 bb=0, so every existing test
        // passed them; 226 of those were under 100 triangles and 63 were single
        // quads, concentrated in three materials. They are the floating planes.
        //
        // This does over-exclude genuinely static translucent world geometry --
        // glass, for one -- and that is the right direction to be wrong in: such
        // geometry falls back to the pre-residency behaviour it has always had,
        // which is a cost in coverage rather than a visible defect. A narrower
        // test would need to distinguish static glass from effect geometry, and
        // nothing on the instance says which is which.
        if (inst->getBillboardCount() != 0u
            || inst->getMaterialType() == MaterialDataType::RayPortal
            || inst->surface.alphaState.isDecal
            || !inst->surface.alphaState.isBlendingDisabled) {
          rec.skipUnsafe = true;
        }
        // Baseline for [HeldRaw], off the first instance that has geometry AND
        // a live entry to read it from -- isUnlinkedForGC() means m_linkedBlas
        // still points at an erased BlasEntry, so getBlas() would hand back a
        // dangling pointer the null test cannot see. Skipping such an instance
        // simply moves the baseline to the next one; if none qualifies,
        // builtPosHash stays 0 and holdsInstance refuses the hold, which is the
        // direction this file already chose to be wrong in.
        if (rec.builtPosHash == 0ull && !inst->isUnlinkedForGC()) {
          if (const BlasEntry* blas = inst->getBlas()) {
            rec.builtPosHash = blas->modifiedGeometryData.hashes[HashComponents::VertexPosition];
            rec.builtBoneHash = blas->modifiedGeometryData.lastBoneHash;
            rec.builtBlas = static_cast<const void*>(blas);
          }
        }
      }
    }

    // SKINNED GEOMETRY IS THE FIFTH UNSAFE CLASS, and it is the one the
    // [HeldRaw] block predicted would "announce itself as a visual bug rather
    // than as a counter".
    //
    // It did. Holding a skinned instance turns part of a mesh black: the
    // Titanfall viewmodel -- the weapon and the hands -- shaded correctly on one
    // side and black on the other for as long as the hold lasted, while every
    // residency counter read healthy.
    //
    // WHY THE OTHER FOUR TESTS CANNOT CATCH IT. A skinned mesh has no
    // billboards, is not a ray portal, is not a decal and is not blended, so it
    // passes all of skipUnsafe. Its positions are REGENERATED PER DRAW from the
    // bone palette, so a held instance that misses a draw renders whatever the
    // skinning pass last wrote into that buffer -- stale normals, hence a
    // shading artefact on part of the mesh rather than a missing object.
    //
    // AND WHY builtPosHash CANNOT CATCH IT EITHER, which is the part worth
    // reading twice. holdsInstance compares hashes[VertexPosition] against
    // builtPosHash and refuses when they differ. But that hash is written in
    // exactly one place, processGeometryInfo, and processGeometryInfo runs when
    // a DRAW ARRIVES. A held instance with no draw never updates it, so the
    // hash still matches the baseline while the buffer underneath has been
    // rewritten. The check passes precisely in the case that is unsafe.
    //
    // builtBoneHash was already captured above and was read by nothing. A
    // non-zero lastBoneHash is the tree's own definition of "this geometry is
    // bone-driven", used the same way by the [HeldRaw] bone= column.
    //
    // Disqualifying through skipUnsafe rather than a new flag is deliberate: it
    // is the existing meaning of "this record must not be held", and it also
    // stops the record being SERVED, which a skinned draw must not be either,
    // for the same reason.
    if (rec.builtBoneHash != 0ull) {
      rec.skipUnsafe = true;
    }

    m_stats.built += 1;
  }

  bool ResidentScene::touch(uint64_t key, uint32_t frame) {
    Record* rec = find(key);

    // A failed touch treated as a success is a SILENT RETIREMENT: the caller
    // would skip the full path believing the instances were kept alive, and they
    // would age out one frame later with nothing to catch it. So the caller acts
    // on the RETURN VALUE, not on its own prediction, and a miss simply commits
    // the draw in full -- the behaviour without residency.
    //
    // THE TWO MISSES ARE COUNTED APART BECAUSE ONLY ONE OF THEM IS A DEFECT.
    //
    //   unknown   no record under this key. Ordinary and self-correcting: the
    //             full path that follows builds one, and the next frame hits.
    //             It also covers the case that has no record BY DESIGN -- a draw
    //             that commits for a side effect and resolves to no instance at
    //             all never files one, and must never be skipped. A steady
    //             non-zero reading here is that population, not a fault.
    //   invalid   a record existed and was invalidated or emptied, which means
    //             an instance it named was destroyed. THIS is the one the plan's
    //             "touchMiss ~0" gate is about, and it wants finding.
    if (rec == nullptr) {
      m_stats.touchMissUnknown += 1;
      m_stats.touchMiss += 1;
      return false;
    }
    if (!rec->valid || rec->instances.empty()) {
      m_stats.touchMissInvalid += 1;
      m_stats.touchMiss += 1;
      return false;
    }
    // Billboards, ray portals and decals rebuild per-frame state that the gate's
    // three tests say nothing about -- see Record::skipUnsafe.
    //
    // OPACITY MICROMAPS ARE THE FOURTH, and read from the option rather than
    // latched at build time because it is a runtime switch: the micromap manager
    // subscribes to the instance-update event and keeps its per-instance
    // bookkeeping on the assumption that a live instance is updated every frame.
    // Skipping the draw stops delivering that event, so with micromaps on the
    // manager would age out data for objects that are still on screen. It is a
    // cost rather than a correctness failure, but it is the opposite of what
    // this feature exists to do, and residency and micromaps working together
    // wants its own measurement rather than an assumption here.
    //
    // Counted on their own so a scene reading touched=0 says WHY rather than
    // merely reading zero.
    if (rec->skipUnsafe || RtxOptions::getEnableOpacityMicromap()) {
      m_stats.touchMissUnsafe += 1;
      m_stats.touchMiss += 1;
      return false;
    }

    for (RtInstance* inst : rec->instances) {
      if (inst == nullptr) {
        continue;
      }
      // THE STAMP IS THE WHOLE MECHANISM. garbageCollection reaps on
      //   m_frameLastUpdated + instanceKeepN <= currentFrame
      // and nothing else, which is why keep-alive without reprocessing is
      // expressible at all.
      inst->setFrameLastUpdated(frame);

      // AND THE CAMERA MUST BE RE-REGISTERED. setFrameLastUpdated() CLEARS
      // m_seenCameraTypes on the first stamp of a frame (see its body), so a
      // skip path that omits this silently loses the instance's camera set and
      // breaks portal / view-model logic downstream. Trap 1 in
      // RESIDENT_SCENE_PLAN.md, already paid for once by pushInstanceRecords.
      for (uint32_t c = 0; c < static_cast<uint32_t>(CameraType::Count); ++c) {
        if ((rec->cameraMask & (1u << c)) != 0u) {
          inst->registerCamera(static_cast<CameraType::Enum>(c), frame);
        }
      }

      // AND THE GEOMETRY HAS TO SURVIVE WITH THE INSTANCE, which is a second
      // lifetime the plan does not name and which would have defeated the whole
      // feature quietly.
      //
      // SceneManager::garbageCollection destroys a BlasEntry on
      //     frameLastTouched < currentFrame - numFramesToKeepGeometryData
      // and that field is stamped ONLY by processDrawCallState -- the path a
      // gate hit exists to skip. So a skipped draw leaves its geometry ageing,
      // the entry is destroyed a few frames later, onSceneObjectDestroyed marks
      // every linked instance for collection, and m_isMarkedForGC is not
      // something residency's keep clause overrides (deliberately: an instance
      // must not outlive its geometry). The instances would therefore retire on
      // schedule no matter how faithfully they were being stamped.
      //
      // Reached through the instance's own link rather than by holding a
      // BlasEntry* in the record: the entry can be destroyed while the record
      // lives, and a raw pointer here would be a second lifetime to police for
      // no benefit.
      //
      // AND m_linkedBlas IS NOT CLEARED WHEN THE INSTANCE IS UNLINKED, which is
      // the whole reason isUnlinkedForGC() is tested here.
      // markAsUnlinkedFromBlasEntryForGarbageCollection() sets that flag and
      // leaves m_linkedBlas pointing at the erased entry, so getBlas() returns a
      // DANGLING pointer that a null check cannot see -- and this site writes
      // through it, which is worse than the read that faulted at +0x278 in the
      // instance-manager reap walk. The flag is the only evidence available: it
      // is set in exactly one place and means "my entry is gone".
      //
      // Skipping the stamp is also the right ANSWER and not merely the safe one.
      // An unlinked instance has no geometry left to keep alive, and it is
      // already marked for collection -- which residency's keep clause
      // deliberately does not override, because an instance must not outlive its
      // geometry.
      BlasEntry* blas = inst->isUnlinkedForGC() ? nullptr : inst->getBlas();
      if (blas != nullptr) {
        blas->frameLastTouched = frame;
      }
    }

    rec->frameLastSeen = frame;
    m_stats.touched += 1;
    m_stats.instancesStamped += static_cast<uint32_t>(rec->instances.size());
    return true;
  }

  bool ResidentScene::score(uint64_t key, const std::vector<RtInstance*>& produced, uint32_t ordinal) {
    m_stats.predicted += 1;

    const Record* rec = find(key);
    if (rec == nullptr || !rec->valid || rec->instances.empty()) {
      // The gate said "serve from the record" and there is no record to serve.
      // Counted apart because it is not the same finding as a record naming the
      // wrong instances, and it has two quite different causes:
      //
      //   the draw resolves to no instance at all -- it commits for a side
      //   effect rather than for geometry -- so nothing was ever filed. Benign,
      //   and the live path handles it by construction: touch() finds nothing
      //   and the draw commits in full.
      //
      //   the record was evicted or invalidated between the two frames. Read
      //   evicted= and wiped= on the same line before concluding anything: if
      //   those are moving, the cap is the story, not the key.
      //
      // Either way no amount of tightening the dirty test would change it, which
      // is why it is not folded in with the FAILs that do want that response.
      //
      // AND THE TWO ARE TOLD APART BY WHAT THE FULL PATH JUST PRODUCED, which
      // is the one piece of evidence that separates them and is already in hand.
      // An empty produced list means the draw resolves to nothing, so nothing
      // was ever filed and nothing ever will be -- processDrawCallState skips
      // build() on an empty list by design. That case is permanent and cannot
      // be driven to zero, so counting it inside the arming gate makes the gate
      // unsatisfiable. A non-empty produced list means a record DID exist at
      // some point and has gone missing, which is a real defect.
      if (produced.empty()) {
        m_stats.failNoRecEmpty += 1;
      } else {
        m_stats.failNoRecLost += 1;
        // AND WHICH KIND OF MISSING, because the two want opposite responses and
        // the difference is invisible from the count alone.
        //
        // A record that was FILED AND THEN ERASED is this measurement fighting
        // itself: while verify is on the keep clause is off, so an instance that
        // misses a frame retires on numFramesToKeepInstances, invalidateFor
        // clears its record and onFrameEnd erases it. That failure is caused by
        // residency being disabled, and arming the keep is what removes it --
        // so counting it against the gate that guards arming is circular.
        //
        // A key that NEVER had a record is a real disagreement: the frame thread
        // says it judged this draw last frame, and the CS side never filed
        // anything for it. That one no amount of arming would fix.
        //
        // The tombstone map is what tells them apart -- see recordTombstone.
        const auto tomb = m_tombstones.find(key);
        if (tomb != m_tombstones.end()) {
          m_stats.failLostErased += 1;
        } else {
          m_stats.failLostNever += 1;
        }
      }
      m_stats.fail += 1;
      return false;
    }

    // ORDER MATTERS AND IS NOT INCIDENTAL. The list is the output of one
    // resolution pass, and a fanout draw's placements are positional -- entry i
    // is placement i. Two lists with the same members in a different order
    // describe two different assignments of placements to instances, so an
    // order-insensitive comparison would pass exactly the case that renders
    // every copy of a mesh at another copy's transform.
    //
    // A COUNT MISMATCH AND A MEMBER MISMATCH ARE DIFFERENT FINDINGS. The count
    // changing says this draw resolved to a different NUMBER of objects than the
    // one that filed the record, which is what happens when the occurrence
    // ordinal slides: engine culling removes copy 1 of a multi-copy identity and
    // copies 2 and 3 shift down onto records built for their neighbours. The
    // count matching while the members differ says the ordinal held and the
    // resolution still landed elsewhere, which the ordinal was introduced to
    // prevent and would mean it does not.
    if (rec->instances.size() != produced.size()) {
      m_stats.failSize += 1;
      m_stats.fail += 1;
      // WHICH WAY, because the two directions are different findings and only
      // one of them is a defect. See Stats::failSizeOver / failSizeUnder.
      if (rec->instances.size() > produced.size()) {
        m_stats.failSizeOver += 1;
      } else {
        m_stats.failSizeUnder += 1;
      }
      // NV-DXVK [RsFailSize]: the counter says HOW MANY size failures, and the
      // two things that decide what a size failure MEANS are not in it.
      //
      // THE SIGN. failSize covers both "the record names more than the draw
      // produced" and "fewer", and those are opposite defects -- a record left
      // holding objects that no longer resolve, versus a draw resolving to
      // objects the record never had. Summed into one counter they are
      // indistinguishable, and the comment above commits to a mechanism
      // (culling removes a copy, later copies shift down) that predicts one
      // sign and not the other.
      //
      // THE OVERLAP. The ordinal-slide story says the draw lands on a
      // NEIGHBOUR's instances, so produced should still be mostly drawn from
      // the same pool the record names. A resolution that went somewhere
      // unrelated shares nothing with it. Same count, same failure, completely
      // different cause, and overlap is what separates them. O(n*m) on a
      // failure path only, over lists the file already documents as a handful
      // of entries -- the same trade the permutation scan below makes.
      if (RtxOptions::ResidentScene::logStats()) {
        static std::atomic<uint32_t> sLines { 0 };
        constexpr uint32_t kMaxLines = 400;
        if (sLines.fetch_add(1, std::memory_order_relaxed) < kMaxLines) {
          uint32_t shared = 0;
          uint32_t fresh = 0;
          for (RtInstance* p : produced) {
            if (std::find(rec->instances.begin(), rec->instances.end(), p) != rec->instances.end()) {
              ++shared;
            }
            // RESPAWNED SINCE THE RECORD WAS BUILT. ord= established that these
            // failures are NOT key collisions -- 99% of them carry ord=0, so the
            // draw's identity is already unique. A uniquely identified draw that
            // resolves to instances the record does not name means the instance
            // set underneath changed, not that the key picked the wrong record.
            // An instance created after the record was last seen is exactly that:
            // the object was reaped and respawned, so the record names pointers
            // that no longer exist. High fresh= makes these failures entry churn
            // surfacing in residency rather than a residency defect.
            if (p->m_frameCreated > rec->frameLastSeen) {
              ++fresh;
            }
          }
          Logger::info(str::format(
            "[RsFailSize] key=0x", std::hex, key, std::dec,
            " ord=", ordinal,
            " recN=", static_cast<uint32_t>(rec->instances.size()),
            " prodN=", static_cast<uint32_t>(produced.size()),
            " delta=", static_cast<int32_t>(produced.size()) - static_cast<int32_t>(rec->instances.size()),
            " shared=", shared,
            " fresh=", fresh,
            " recLastSeen=", rec->frameLastSeen));
        }
      }
      return false;
    }
    for (size_t i = 0; i < produced.size(); ++i) {
      if (rec->instances[i] != produced[i]) {
        // A PERMUTATION AND A DIFFERENT SET ARE DIFFERENT FINDINGS, and only one
        // of them is a failure.
        //
        // Same instances in a different order means resolution is not
        // order-stable: two draws sharing one identity claimed each other's
        // instances this frame. A genuinely different set means the draw
        // resolved to objects the record never named, which is the serious
        // reading and the one that would keep the wrong geometry resident.
        //
        // O(n^2), and deliberately: this runs only on a failure, the lists are
        // a handful of entries, and a set allocation per failure would cost more
        // than the scan it replaces.
        bool permutation = true;
        for (RtInstance* p : produced) {
          if (std::find(rec->instances.begin(), rec->instances.end(), p) == rec->instances.end()) {
            permutation = false;
            break;
          }
        }
        if (permutation) {
          // A PERMUTATION IS NOT A FAILURE, BECAUSE OF WHAT touch() DOES WITH
          // THE LIST. It walks every entry and stamps it -- setFrameLastUpdated,
          // the camera replay, the BLAS frameLastTouched -- and reads position i
          // for nothing. The list is consumed as a SET, so a record naming the
          // same instances in a different order keeps exactly the right objects
          // alive.
          //
          // The comment above about placements being positional describes how
          // the list was PRODUCED, not how it is used, and applying it here made
          // score() stricter than the only consumer requires. Measured cost of
          // that strictness: 113 of 730 failures, 15%, none of which would have
          // kept a wrong object.
          //
          // STILL COUNTED, because the day a consumer does become positional
          // this is the number that says how often it would be wrong -- and
          // because unstable resolution order is worth knowing about on its own.
          m_stats.memberPerm += 1;
          return true;
        }
        m_stats.failMember += 1;
        m_stats.fail += 1;
        // NV-DXVK [RsFailMember]: same count, not a permutation -- so the draw
        // resolved to instances the record does not name. The counter cannot
        // say HOW far off, and the two ends of that range are different bugs.
        //
        // shared=0 means the resolution landed on a completely different set:
        // the key collided, or the identity behind it is not the one the record
        // was built for. 0<shared<n means the lists PARTIALLY agree, which is
        // the neighbour-bleed the ordinal was introduced to prevent and points
        // at the same mechanism [RsFailSize] is testing. firstBad says where
        // the two lists first diverge, so a list that agrees on a prefix and
        // then slides is visible as such rather than as a flat mismatch.
        if (RtxOptions::ResidentScene::logStats()) {
          static std::atomic<uint32_t> sLines { 0 };
          constexpr uint32_t kMaxLines = 400;
          if (sLines.fetch_add(1, std::memory_order_relaxed) < kMaxLines) {
            uint32_t shared = 0;
            uint32_t fresh = 0;
            for (RtInstance* p : produced) {
              if (std::find(rec->instances.begin(), rec->instances.end(), p) != rec->instances.end()) {
                ++shared;
              }
              // See the same field on [RsFailSize]: an instance created after
              // the record was last seen was respawned since, so the record
              // names pointers that are gone. That is entry churn showing up
              // here, not the residency key choosing wrongly.
              if (p->m_frameCreated > rec->frameLastSeen) {
                ++fresh;
              }
            }
            // WAS THE RECORD'S INSTANCE TAKEN, OR LEFT BEHIND? The two are
            // opposite findings and shared=/fresh= cannot tell them apart.
            //
            // Every one of these failures reaches here with the record VALID,
            // and invalidateFor is total -- an instance cannot be destroyed
            // without going through removeInstance, which invalidates in the
            // same call. So the instances the record names are all still ALIVE
            // while the draw resolved to different ones, and the question is
            // what happened to them this frame:
            //
            //   claimed   another draw updated it this frame. The record's
            //             instances still belong to something, so this is a
            //             re-batching or a key naming another object's record.
            //   unclaimed nothing updated it. The draw that owned it spent its
            //             resolution on a NEW instance instead -- a dedup miss
            //             in findSimilarInstance -- and this one now ages out
            //             while its replacement renders. That is the respawn
            //             verdict the reap census already names, arriving here
            //             through a different door.
            //
            // The current frame is taken off a produced instance rather than
            // passed in: those were resolved by the full path moments ago, so
            // their stamp IS this frame, and reading it costs no signature
            // change through a header three files include.
            const uint32_t curFrame = produced.empty()
              ? rec->frameLastSeen
              : produced[0]->getFrameLastUpdated();
            uint32_t recClaimed = 0;
            uint32_t recUnclaimed = 0;
            uint32_t recAgeMax = 0;
            for (const RtInstance* r : rec->instances) {
              if (r == nullptr) {
                continue;
              }
              const uint32_t upd = r->getFrameLastUpdated();
              if (upd == curFrame) {
                ++recClaimed;
              } else {
                ++recUnclaimed;
                if (curFrame > upd && (curFrame - upd) > recAgeMax) {
                  recAgeMax = curFrame - upd;
                }
              }
            }
            // WHICH DRAW, so the failing shaders can be named. Taken off a
            // PRODUCED instance: those were resolved by the full path moments
            // ago, so their entry is live -- unlike the record's, which is what
            // isUnlinkedForGC guards against elsewhere in this file. Guarded
            // anyway, because "freshly resolved" is an argument and the flag is
            // evidence.
            uint64_t vsHash = 0ull;
            // DID THE GEOMETRY ENTRY CHANGE UNDER THE DRAW? This is the test
            // that decides where the fix goes, and the record already carries
            // the evidence.
            //
            // A predictHit requires o2wMatch, so the transform behind these
            // draws is byte-identical to last frame's, and the material fold
            // matched too. A stable transform and a stable material that still
            // resolve to a DIFFERENT instance leaves the BlasEntry: if the draw
            // was re-batched onto a new entry, findSimilarInstance searches that
            // entry's linked instances, finds none, and mints a fresh one --
            // while the old instance sits unclaimed on the old entry and ages
            // out. That is entry churn arriving here rather than a residency
            // fault, and blasSame=0 is what says so.
            //
            // builtBlas is compared as a raw address and never dereferenced,
            // which is the contract it was stored under.
            int blasSame = -1;
            if (!produced.empty() && produced[0] != nullptr
                && !produced[0]->isUnlinkedForGC()) {
              if (const BlasEntry* pb = produced[0]->getBlas()) {
                vsHash = static_cast<uint64_t>(
                  pb->input.getTransformData().vertexShaderHash);
                blasSame = (rec->builtBlas == static_cast<const void*>(pb)) ? 1 : 0;
              }
            }
            Logger::info(str::format(
              "[RsFailMember] key=0x", std::hex, key,
              " vs=0x", vsHash, std::dec,
              " blasSame=", blasSame,
              " ord=", ordinal,
              " n=", static_cast<uint32_t>(produced.size()),
              " firstBad=", static_cast<uint32_t>(i),
              " shared=", shared,
              " fresh=", fresh,
              " curF=", curFrame,
              " recClaimed=", recClaimed,
              " recUnclaimed=", recUnclaimed,
              " recAgeMax=", recAgeMax,
              " recLastSeen=", rec->frameLastSeen));
          }
        }
        return false;
      }
    }
    return true;
  }

  bool ResidentScene::holdsInstance(const RtInstance* instance) const {
    if (instance == nullptr || instance->m_residentKey == 0ull) {
      return false;
    }
    // AN INSTANCE WHOSE ENTRY IS ALREADY GONE CANNOT HOLD GEOMETRY ALIVE, and
    // asking further questions about it reads freed memory.
    //
    // This runs from blasEntryGarbageCollection, inside the same
    // SceneManager::garbageCollection pass that erases BlasEntries.
    // markAsUnlinkedFromBlasEntryForGarbageCollection() sets this flag and
    // leaves m_linkedBlas pointing at the erased entry, so the getBlas() below
    // can return a dangling pointer that survives a null test -- and the content
    // check dereferences it. The builtBlas comparison above it is NOT a guard
    // against that: it compares pointer VALUES, and a stale pointer compares
    // equal to the stale value the record recorded, so it passes exactly when it
    // should not.
    //
    // Refusing is also the right answer rather than merely the safe one. The
    // instance is already marked for collection, and residency's keep clause
    // deliberately does not override that, because an instance must not outlive
    // its geometry.
    if (instance->isUnlinkedForGC()) {
      return false;
    }
    const Record* rec = find(instance->m_residentKey);
    if (rec == nullptr || !rec->valid) {
      return false;
    }
    // skipUnsafe DISQUALIFIES A RECORD FROM HOLDING, not just from being
    // skipped, and the two were only ever the same question by accident.
    //
    // While the keep and the skip armed together, a skipUnsafe record was held
    // alive but its draw always arrived to rebuild the per-frame work -- the
    // billboard vector, the portal registration, the decal sort order -- so
    // holding it was harmless. Arming the keep on its own breaks that: the
    // instance survives a frame with no draw, and nothing reproduces any of it.
    // A held particle, beam or decal is then a frozen one, which is worse than
    // the retirement it was exempted from.
    //
    // OPACITY MICROMAPS ARE NOT TESTED HERE, AND touch() IS RIGHT TO TEST THEM.
    // The two are asking different questions and the option was briefly copied
    // across, which disabled the keep outright on any build where micromaps are
    // supported -- this one. Residency held nothing and liveInst sat at its
    // pre-residency baseline while every counter read healthy.
    //
    // The difference is what the game is doing. touch() stands in for a draw
    // that ARRIVED: the object is actively rendering, the micromap manager is
    // expecting the instance-update event, and suppressing it starves live
    // bookkeeping. The keep covers an object that got NO DRAW AT ALL, which
    // before residency was simply destroyed. Its manager state going stale is a
    // quality cost on an off-screen caster, and onInstanceDestroyed still fires
    // when the record finally ages out, so nothing is left dangling.
    //
    // skipUnsafe applies to both because it is not about the draw -- billboards,
    // beams, portals and decals are rebuilt from scratch each frame, so an
    // instance carrying one is wrong on screen whether it was skipped or merely
    // held.
    if (rec->skipUnsafe) {
      return false;
    }

    // AND THE GEOMETRY MUST STILL BE THE GEOMETRY THE RECORD WAS BUILT AGAINST.
    //
    // Holding an instance keeps its BLAS alive; it does NOT keep the BLAS's
    // vertex CONTENT from being rewritten. When the buffer behind it is reused
    // for another mesh, the held instance renders that mesh's vertices through
    // its own indices and transform, which on screen is a quad stretched into a
    // diagonal sliver.
    //
    // MEASURED. [HeldRaw] posMoved=1 on 90 of 1080 held instances, and the
    // population is completely uniform: every one is tri=4, one material, an
    // alpha-tested cutout, held between 10 and 450 frames. Small quads are the
    // ones that show it because a rewritten 8-vertex buffer produces a shape
    // with no resemblance to the original.
    //
    // THIS IS A CONTENT TEST, NOT A CLASS TEST, which is why it belongs here
    // rather than in skipUnsafe. skipUnsafe asks "what kind of thing is this",
    // and no answer to that question would have caught this: the geometry is
    // ordinary opaque cutout world geometry right up until its buffer is
    // recycled. Asking whether the content still matches is the only test that
    // catches a producer nobody has enumerated, and it self-corrects -- the next
    // draw rebuilds the record against the new content and holding resumes.
    // NO BASELINE MEANS NO HOLD. builtPosHash is 0 when the record's first
    // instance had no BLAS at build time, and the old form of this check skipped
    // validation entirely in that case -- so the records that could not be
    // verified were exactly the ones held unconditionally. Refusing is the safe
    // direction: the instance falls back to the lifetime it had before residency.
    if (rec->builtPosHash == 0ull) {
      return false;
    }
    // AND ONLY AGAINST THE ENTRY THE BASELINE CAME FROM. A record whose
    // instances span several BLASes would otherwise have one instance's hash
    // clear all of them, which validates every instance except the one it was
    // actually measured on.
    if (rec->builtBlas != static_cast<const void*>(instance->getBlas())) {
      return false;
    }
    {
      if (const BlasEntry* blas = instance->getBlas()) {
        // Baseline came from the record's first instance with geometry. For a
        // fanout record every placement shares one mesh, so one baseline is the
        // right one; if that ever stops being true the mismatch refuses the
        // hold, which is the safe direction -- the instance falls back to the
        // ordinary lifetime it had before residency.
        if (blas->modifiedGeometryData.hashes[HashComponents::VertexPosition] != rec->builtPosHash) {
          return false;
        }
      }
    }
    return true;
  }

  void ResidentScene::invalidateFor(const RtInstance* instance) {
    if (instance == nullptr) {
      return;
    }
    const uint64_t key = instance->m_residentKey;
    if (key == 0ull) {
      return;
    }
    // Clear this instance's own key first and unconditionally, so the loop
    // below skips this (dying) entry for free.
    instance->m_residentKey = 0ull;

    const auto it = m_records.find(key);
    if (it != m_records.end()) {
      // INVALIDATE THE WHOLE RECORD, never erase one element. The list is only
      // meaningful as the complete output of one resolution pass; a list with a
      // hole would stamp some of an object's instances and let the rest retire,
      // which is the s2s "two views" failure shape -- alive but not refreshed,
      // with no FAIL to catch it.
      it->second.valid = false;
      m_stats.invalidated += 1;

      // NV-DXVK [FanoutUAF, 2026-08-21]: and DROP THE POINTERS, which marking
      // the record invalid does not do. Three places walk rec.instances and
      // dereference every element -- the rebuild in insert(), the eviction in
      // the age sweep, and dropRecord() -- none of them checking `valid`. A
      // pointer left here by a destroyed instance is read by whichever runs
      // first. The `inst != nullptr` guards at those sites do not help: a freed
      // instance is not a null pointer.
      //
      // This is the same defect that crashed the FANOUT record path at
      // InstanceManager::onFrameEnd (rtx_instance_manager.cpp:3465) on
      // 2026-08-21. It has not fired here only because residency is inert; the
      // removeInstance comment is right that it would matter MORE once armed,
      // since a resident record is designed to outlive an unbounded number of
      // frames without being rebuilt.
      //
      // Safe here for the same reason as the fanout fix: removeInstance calls
      // this BEFORE destroying the instance, so every pointer in the list is
      // still live, and a sibling destroyed earlier in the same GC pass already
      // ran this and emptied the list (we then return early on key == 0).
      for (RtInstance* other : it->second.instances) {
        if (other != nullptr && other->m_residentKey == key) {
          other->m_residentKey = 0ull;
        }
      }
      it->second.instances.clear();
    }
  }

  void ResidentScene::onFrameEnd(uint32_t frame) {
    m_stats.live = static_cast<uint32_t>(m_records.size());

    const uint32_t maxRecords = RtxOptions::ResidentScene::maxRecords();
    const uint32_t keepFrames = RtxOptions::ResidentScene::keepFrames();

    // THE DEATH SIGNAL. Take the buffers the engine freed since the last frame,
    // for pass 1 below to invalidate every record built from one of them.
    //
    // WHY THIS PASS EXISTS AT ALL: a resident instance is exempt from lifetime
    // expiry, which is right for world geometry and static props and wrong for
    // anything that can be destroyed. Without this, a killed entity's geometry
    // would keep casting shadows and appearing in reflections until its record
    // happened to be evicted. The engine freeing the vertex or index buffer IS
    // the object ceasing to exist, and it is the one part of an object's fate
    // DXVK observes directly.
    //
    // A SET AND NO INDEX FROM BUFFER TO RECORDS. Pass 1 below already walks
    // every record every frame, so the test rides along inside it: a pair of
    // lookups per record on the frames where anything died, and one empty()
    // check on every other frame. An index would be faster per death and would
    // have to be kept in step through build, invalidate, age-eviction and wipe
    // -- four places to keep consistent, to speed up a walk that is already
    // being paid for.
    std::unordered_set<uint64_t> dead;
    {
      SourceBufferDeaths& deaths = sourceBufferDeaths();
      std::lock_guard<dxvk::mutex> lock(deaths.mutex);
      dead.swap(deaths.pending);
    }
    // The three numbers that make srcDied readable, published together so they
    // cannot be read from different windows: notices ever received, buffers
    // drained this frame, and (below, as sourceDestroyed) records those buffers
    // actually matched.
    m_stats.srcNotices = g_srcDeathNotices.load(std::memory_order_relaxed);
    m_stats.srcDrained += static_cast<uint32_t>(dead.size());

    // Pass 1: erase records that were invalidated. They cannot serve, and
    // leaving them resident makes the map grow without bound in exactly the
    // scene where identity is churning -- i.e. the case we most need to see in
    // the stats rather than absorb silently.
    for (auto it = m_records.begin(); it != m_records.end(); ) {
      // The death signal, folded in. Invalidate rather than erase directly, so
      // that the back-pointer clearing below runs exactly once for every way a
      // record can leave -- a record erased without it leaves instances naming a
      // key that no longer exists, and that stale key collides with whatever
      // later hashes into the same slot.
      if (it->second.valid && !dead.empty()) {
        const Record& rec = it->second;
        const bool vbDead = rec.srcVertexBuffer != 0ull && dead.count(rec.srcVertexBuffer) != 0;
        const bool ibDead = rec.srcIndexBuffer != 0ull && dead.count(rec.srcIndexBuffer) != 0;
        if (vbDead || ibDead) {
          it->second.valid = false;
          m_stats.sourceDestroyed += 1;
        }
      }

      bool erase = !it->second.valid;

      // keepFrames is 0 = UNBOUNDED by design; residency ends lifetime-by-age
      // rather than lengthening it. Non-zero exists only to bisect a residency
      // bug against the old behaviour, so the age test is guarded on it.
      // keepFrames is 0 = UNBOUNDED by design; residency ends lifetime-by-age
      // rather than lengthening it. Non-zero exists only to bisect a residency
      // bug against the old behaviour, so the age test is guarded on it.
      //
      // DO NOT USE THIS TO BOUND GHOSTS. It is a single fixed age bound, which
      // the eviction note in pass 2 already identifies as the wrong shape, and a
      // measurement confirmed it: with keepFrames=600 the whole load-time cohort
      // aged out on one frame (evicted 25 -> 885 -> 694 while invalidated stayed
      // at ~20), liveInst fell 4144 -> 1200 and never recovered, because engine
      // culling means nothing redraws those objects. Bound the set with
      // maxRecords instead and let the age-rung LADDER below do it: that policy
      // is demand-driven, evicts least-recently-seen first, and stops the moment
      // it is under the cap, so a cohort drains gradually instead of falling off
      // a cliff.
      if (!erase && keepFrames != 0u
          && it->second.frameLastSeen != kInvalidFrameIndex
          && frame > it->second.frameLastSeen
          && (frame - it->second.frameLastSeen) > keepFrames) {
        erase = true;
      }

      if (erase) {
        for (RtInstance* inst : it->second.instances) {
          if (inst != nullptr && inst->m_residentKey == it->first) {
            inst->m_residentKey = 0ull;
          }
        }
        recordTombstone(it->first, frame);
        it = m_records.erase(it);
        m_stats.evicted += 1;
      } else {
        ++it;
      }
    }

    // Pass 2: down to the ceiling, on the AGE-RUNG LADDER.
    //
    // SAME POLICY AS [Perf.SplitXf]'s evict{} and the [Perf.PushInst] age
    // sweep, copied rather than reinvented -- this tree has already worked out
    // what the alternatives cost:
    //
    //   a flat clear      destroys the long-lived entries that were about to
    //                     pay off along with the churn; the eviction policy
    //                     itself becomes the second-order loss.
    //   a std::sort LRU   O(n log n) per frame on the RT path, to produce an
    //                     ordering that age already gives for free.
    //   one fixed age     CANNOT FIRE at a high fill rate: by the time the cap
    //                     is reached the oldest entry is younger than the bound
    //                     and the test matches nothing. That is exactly how the
    //                     first split-transform sweep freed zero for weeks
    //                     while reading as a working policy.
    //
    // So ASK rather than assume: walk the ladder and stop at the first bound
    // that gets under the cap, keeping as much history as the cap can afford.
    // frameLastSeen is stamped on every build and every touch, so a record
    // serving each frame is permanently age 0-1 and survives every rung, while
    // a key minted once and abandoned ages without bound. Any rung separates
    // those two populations; the ladder only decides how much of the
    // recently-but-not-currently-used middle survives -- and that middle is the
    // culled-then-visible object this whole feature exists for.
    //
    // AND THE CEILING BEING HIT AT ALL IS THE FINDING. In a stationary scene it
    // means the KEY is churning (trap 3), not that the scene is large. Read
    // [ResidentGate] newKeys before touching maxRecords.
    const auto dropRecord = [](std::unordered_map<uint64_t, Record>::iterator it) {
      // Clear the back-pointers before erasing. An instance whose m_residentKey
      // still names an erased record would never be invalidated again, and that
      // stale key would collide with whatever later hashes into the same slot.
      for (RtInstance* inst : it->second.instances) {
        if (inst != nullptr && inst->m_residentKey == it->first) {
          inst->m_residentKey = 0ull;
        }
      }
    };

    // AND THE SWEEP IS BOUNDED BY THE OVERAGE, NOT BY THE RUNG. The rung says
    // WHICH records may go; the cap says HOW MANY. Without the size test in the
    // inner loop the first rung that fires removes EVERY record past it,
    // however far under the cap that lands -- and the paragraph above, which
    // claims the ladder "stops the moment it is under" and "cannot produce a
    // cliff", described that intent rather than the code.
    //
    // AN EVICTION HERE DESTROYS GEOMETRY, which is what makes the overshoot a
    // defect rather than a cache inefficiency. dropRecord() zeroes
    // m_residentKey, holdsInstance() then refuses, and with engine culling on
    // nothing redraws an off-screen object to rebuild the record -- so the
    // instances starve on numFramesToKeepInstances and do not come back.
    //
    // MEASURED, 2026-08-24, cap 12000:
    //   f=8214  records=12000  invalidated=0  evicted=0     liveInst=14057
    //   f=8216  [ReapJoin] removed=12265 starved=12265      liveInst=1790
    //   f=8224  records=5920   invalidated=0  evicted=6133  liveInst=1787
    // The store crossed the cap by about one record and lost 6,133, taking 87%
    // of the resident scene with it, permanently. invalidated=0 across the
    // window is what fixes the direction: the records went first and the
    // instances starved after, not the reverse.
    if (maxRecords != 0u && m_records.size() > maxRecords) {
      static constexpr uint32_t kAgeRungs[] = { 300u, 60u, 8u, 2u };
      for (uint32_t r = 0; r < std::size(kAgeRungs) && m_records.size() > maxRecords; ++r) {
        const uint32_t maxAge = kAgeRungs[r];
        for (auto it = m_records.begin();
             it != m_records.end() && m_records.size() > maxRecords; ) {
          if (it->second.frameLastSeen != kInvalidFrameIndex
              && frame > it->second.frameLastSeen
              && (frame - it->second.frameLastSeen) > maxAge) {
            dropRecord(it);
            recordTombstone(it->first, frame);
            it = m_records.erase(it);
            m_stats.evicted += 1;
          } else {
            ++it;
          }
        }
      }

      // Still over after the tightest rung means the cap's worth of records
      // were ALL touched within 2 frames -- a working set that genuinely
      // exceeds the cap rather than churn. A wipe is then the honest answer:
      // silently keeping an unbounded map would trade a frame-time bug for a
      // memory one. It does NOT cost "one frame of rebuilding", which is what
      // this said before the measurement above: only the records whose draws
      // arrive get rebuilt, and the off-screen set has no draw to rebuild it.
      // A wipe is the cliff at full width, so raise maxRecords rather than
      // letting it fire. m_stats.wiped reading
      // non-zero means maxRecords is too small for the scene, which is a
      // DIFFERENT finding from key churn and wants the opposite response.
      if (m_records.size() > maxRecords) {
        for (auto it = m_records.begin(); it != m_records.end(); ++it) {
          dropRecord(it);
        }
        m_stats.evicted += static_cast<uint32_t>(m_records.size());
        m_records.clear();
        m_stats.wiped += 1;
        // NOT TOMBSTONED, and wiped= is what says so. A wipe erases the whole
        // store at once, so tombstoning it would mint one entry per record and
        // then answer "erased" for every key in the scene for the next few
        // frames -- a diagnostic loud enough to drown the finding it exists to
        // report. Read wiped= alongside failLostNever instead: a non-zero wipe
        // is the reason for the burst that follows it.
      }
    }

    // Age out the tombstones, in the same pass that produced them. The window is
    // kTombstoneFrames because the only question asked of this map is about the
    // immediately preceding frame.
    for (auto it = m_tombstones.begin(); it != m_tombstones.end(); ) {
      if (frame > it->second && (frame - it->second) > kTombstoneFrames) {
        it = m_tombstones.erase(it);
      } else {
        ++it;
      }
    }
  }

  void ResidentScene::recordTombstone(uint64_t key, uint32_t frame) {
    // Bounded for the same reason the buffer-death queue is: this fills fastest
    // in exactly the churning scene where it would hurt most, and dropping a
    // tombstone only costs the diagnostic its precision -- a dropped entry reads
    // as failLostNever, which is the conservative direction because it reports
    // the more serious of the two findings rather than the excusable one.
    if (m_tombstones.size() < 131072u) {
      m_tombstones[key] = frame;
    }
  }

  void ResidentScene::clear() {
    for (auto& [k, rec] : m_records) {
      for (RtInstance* inst : rec.instances) {
        if (inst != nullptr && inst->m_residentKey == k) {
          inst->m_residentKey = 0ull;
        }
      }
    }
    m_records.clear();
    m_tombstones.clear();
  }

}
