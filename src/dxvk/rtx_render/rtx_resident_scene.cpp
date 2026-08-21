#include "rtx_resident_scene.h"

#include <iterator>

#include "rtx_instance_manager.h"
#include "rtx_options.h"

namespace dxvk {

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

  ResidentScene::Record& ResidentScene::obtain(uint64_t key) {
    return m_records[key];
  }

  void ResidentScene::build(uint64_t key,
                            uint64_t srcGenHash,
                            uint32_t frame,
                            std::vector<RtInstance*>&& instances) {
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

    rec.instances = std::move(instances);
    rec.srcGenHash = srcGenHash;
    rec.frameLastSeen = frame;
    rec.frameLastBuilt = frame;
    rec.valid = true;

    for (RtInstance* inst : rec.instances) {
      if (inst != nullptr) {
        inst->m_residentKey = key;
      }
    }

    m_stats.built += 1;
  }

  bool ResidentScene::touch(uint64_t key, uint32_t frame, uint32_t cameraType) {
    Record* rec = find(key);
    if (rec == nullptr || !rec->valid || rec->instances.empty()) {
      // A failed touch treated as a success is a SILENT RETIREMENT: the caller
      // would skip the full path believing the instances were kept alive, and
      // they would age out one frame later with nothing to catch it. Report the
      // miss and let the caller fall back.
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
      // breaks portal / view-model logic downstream. Documented as trap 1 in
      // RESIDENT_SCENE_PLAN.md and already paid for by pushInstanceRecords.
      inst->registerCamera(static_cast<CameraType::Enum>(cameraType), frame);
    }

    rec->frameLastSeen = frame;
    m_stats.touched += 1;
    m_stats.instancesStamped += static_cast<uint32_t>(rec->instances.size());
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
    const auto it = m_records.find(key);
    if (it != m_records.end()) {
      // INVALIDATE THE WHOLE RECORD, never erase one element. The list is only
      // meaningful as the complete output of one resolution pass; a list with a
      // hole would stamp some of an object's instances and let the rest retire,
      // which is the s2s "two views" failure shape -- alive but not refreshed,
      // with no FAIL to catch it.
      it->second.valid = false;
      m_stats.invalidated += 1;
    }
    instance->m_residentKey = 0ull;
  }

  void ResidentScene::onFrameEnd(uint32_t frame) {
    m_stats.live = static_cast<uint32_t>(m_records.size());

    const uint32_t maxRecords = RtxOptions::ResidentScene::maxRecords();
    const uint32_t keepFrames = RtxOptions::ResidentScene::keepFrames();

    // Pass 1: erase records that were invalidated. They cannot serve, and
    // leaving them resident makes the map grow without bound in exactly the
    // scene where identity is churning -- i.e. the case we most need to see in
    // the stats rather than absorb silently.
    for (auto it = m_records.begin(); it != m_records.end(); ) {
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
