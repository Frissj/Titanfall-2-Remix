#include "rtx_resident_scene.h"

#include <iterator>
#include <mutex>
#include <unordered_set>

#include "../../util/thread.h"

#include "rtx_instance_manager.h"
#include "rtx_options.h"

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
    for (RtInstance* inst : rec.instances) {
      if (inst != nullptr) {
        inst->m_residentKey = key;
        rec.cameraMask |= inst->getSeenCameraMask();
        // See Record::skipUnsafe. Each of these three names per-frame work that
        // hangs off the instance rather than off the geometry, the transform or
        // the material, so the gate cannot speak for it.
        if (inst->getBillboardCount() != 0u
            || inst->getMaterialType() == MaterialDataType::RayPortal
            || inst->surface.alphaState.isDecal) {
          rec.skipUnsafe = true;
        }
      }
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
      // no benefit. m_linkedBlas is cleared when the instance is unlinked.
      if (BlasEntry* blas = inst->getBlas()) {
        blas->frameLastTouched = frame;
      }
    }

    rec->frameLastSeen = frame;
    m_stats.touched += 1;
    m_stats.instancesStamped += static_cast<uint32_t>(rec->instances.size());
    return true;
  }

  bool ResidentScene::score(uint64_t key, const std::vector<RtInstance*>& produced) {
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
      m_stats.failNoRecord += 1;
      m_stats.fail += 1;
      return false;
    }

    // ORDER MATTERS AND IS NOT INCIDENTAL. The list is the output of one
    // resolution pass, and a fanout draw's placements are positional -- entry i
    // is placement i. Two lists with the same members in a different order
    // describe two different assignments of placements to instances, so an
    // order-insensitive comparison would pass exactly the case that renders
    // every copy of a mesh at another copy's transform.
    if (rec->instances.size() != produced.size()) {
      m_stats.fail += 1;
      return false;
    }
    for (size_t i = 0; i < produced.size(); ++i) {
      if (rec->instances[i] != produced[i]) {
        m_stats.fail += 1;
        return false;
      }
    }
    return true;
  }

  bool ResidentScene::holdsInstance(const RtInstance* instance) const {
    if (instance == nullptr || instance->m_residentKey == 0ull) {
      return false;
    }
    const Record* rec = find(instance->m_residentKey);
    return rec != nullptr && rec->valid;
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

    if (maxRecords != 0u && m_records.size() > maxRecords) {
      static constexpr uint32_t kAgeRungs[] = { 300u, 60u, 8u, 2u };
      for (uint32_t r = 0; r < std::size(kAgeRungs) && m_records.size() > maxRecords; ++r) {
        const uint32_t maxAge = kAgeRungs[r];
        for (auto it = m_records.begin(); it != m_records.end(); ) {
          if (it->second.frameLastSeen != kInvalidFrameIndex
              && frame > it->second.frameLastSeen
              && (frame - it->second.frameLastSeen) > maxAge) {
            dropRecord(it);
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
      // it costs one frame of rebuilding, and silently keeping an unbounded map
      // would trade a frame-time bug for a memory one. m_stats.wiped reading
      // non-zero means maxRecords is too small for the scene, which is a
      // DIFFERENT finding from key churn and wants the opposite response.
      if (m_records.size() > maxRecords) {
        for (auto it = m_records.begin(); it != m_records.end(); ++it) {
          dropRecord(it);
        }
        m_stats.evicted += static_cast<uint32_t>(m_records.size());
        m_records.clear();
        m_stats.wiped += 1;
      }
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
  }

}
