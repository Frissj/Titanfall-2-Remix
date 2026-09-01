#include "rtx_render_object.h"

#include <algorithm>
#include <utility>

namespace dxvk {

  // The occurrence is folded into the low bits rather than hashed with the
  // identity, so that a debugger and a log line can both read the identity back
  // out of the key. Multiplying instead of XOR-ing keeps two different
  // (identity, occurrence) pairs from colliding when the occurrence is small,
  // which it always is -- sec 1.3 measures 4-6 draws per renderable and the
  // multi-copy identities in the ladder's rung 4 note are 3 copies.
  //
  // 0x9E3779B97F4A7C15 is the 64-bit golden-ratio constant used elsewhere in
  // this tree for the same purpose.
  uint64_t RenderObjectDB::primitiveKey(uint64_t iaIdentity, uint32_t occurrence) {
    return iaIdentity ^ (static_cast<uint64_t>(occurrence) * 0x9E3779B97F4A7C15ull);
  }

  // --------------------------------------------------------------------------
  // Slot allocation. The generation is bumped on FREE so that every outstanding
  // handle to a slot goes stale the moment the slot is returned, whether or not
  // it is ever handed out again. Bumping on allocate would leave a handle to a
  // freed-but-unreused slot reading as valid, which is the whole hazard.
  // --------------------------------------------------------------------------

  RenderObjectId RenderObjectDB::allocObject() {
    uint32_t idx;
    if (!m_freeObjects.empty()) {
      idx = m_freeObjects.back();
      m_freeObjects.pop_back();
      m_objects[idx] = RenderObject();
    } else {
      idx = static_cast<uint32_t>(m_objects.size());
      m_objects.emplace_back();
      m_objectGen.push_back(0u);
    }

    RenderObjectId id;
    id.index = idx;
    id.generation = m_objectGen[idx];
    m_objects[idx].id = id;
    // sec 1.3 measures 4-6 primitives per renderable. Reserving here rather
    // than on first push means the grow does not happen on the frame slice 2
    // starts merging, which is the frame the numbers have to be readable.
    m_objects[idx].primitives.reserve(4);
    return id;
  }

  RenderPrimitiveId RenderObjectDB::allocPrimitive() {
    uint32_t idx;
    if (!m_freePrimitives.empty()) {
      idx = m_freePrimitives.back();
      m_freePrimitives.pop_back();
      m_primitives[idx] = RenderPrimitive();
    } else {
      idx = static_cast<uint32_t>(m_primitives.size());
      m_primitives.emplace_back();
      m_primitiveGen.push_back(0u);
    }

    RenderPrimitiveId id;
    id.index = idx;
    id.generation = m_primitiveGen[idx];
    m_primitives[idx].id = id;
    return id;
  }

  void RenderObjectDB::freeObject(RenderObjectId id) {
    if (!id.valid() || id.index >= m_objects.size()) {
      return;
    }
    if (m_objectGen[id.index] != id.generation) {
      return;   // already freed and possibly reissued -- this handle is stale
    }

    RenderObject& o = m_objects[id.index];
    // Drop the identity indices BEFORE the generation bump, while the object
    // still knows what it was filed under. An index left pointing at a freed
    // slot would resolve to a stale handle on the next lookup, which the
    // generation check would then reject -- correct, but it would look like
    // churn rather than like a leak, and those prescribe opposite responses.
    if (o.iaIdentity != 0ull) {
      const auto it = m_objectsByIa.find(o.iaIdentity);
      if (it != m_objectsByIa.end() && it->second == id) {
        m_objectsByIa.erase(it);
      }
    }
    if (o.engineHandle != 0ull) {
      const auto it = m_objectsByHandle.find(o.engineHandle);
      if (it != m_objectsByHandle.end() && it->second == id) {
        m_objectsByHandle.erase(it);
      }
    }

    o = RenderObject();
    ++m_objectGen[id.index];
    m_freeObjects.push_back(id.index);
  }

  void RenderObjectDB::freePrimitive(RenderPrimitiveId id) {
    if (!id.valid() || id.index >= m_primitives.size()) {
      return;
    }
    if (m_primitiveGen[id.index] != id.generation) {
      return;
    }

    RenderPrimitive& p = m_primitives[id.index];

    if (p.residentKey != 0ull) {
      const auto it = m_primitivesByResidentKey.find(p.residentKey);
      if (it != m_primitivesByResidentKey.end() && it->second == id) {
        m_primitivesByResidentKey.erase(it);
      }
    }

    // Unlink from the owner. The owner keeps its OTHER primitives -- this is
    // not ResidentScene's "invalidate the whole record, never one element"
    // rule, and the difference is real: a Record's instance list is only
    // meaningful as the complete output of one resolution pass, whereas an
    // object's primitive list is a set of independently-resolved draws that can
    // legitimately come and go one at a time (a model dropping an LOD mesh, a
    // material variant that stopped being submitted).
    RenderObject* owner = getObject(p.owner);
    if (owner != nullptr) {
      auto& v = owner->primitives;
      v.erase(std::remove(v.begin(), v.end(), id), v.end());
    }

    p = RenderPrimitive();
    ++m_primitiveGen[id.index];
    m_freePrimitives.push_back(id.index);
  }

  // --------------------------------------------------------------------------

  RenderObject* RenderObjectDB::getObject(RenderObjectId id) {
    if (!id.valid() || id.index >= m_objects.size() ||
        m_objectGen[id.index] != id.generation) {
      return nullptr;
    }
    return &m_objects[id.index];
  }

  const RenderObject* RenderObjectDB::getObject(RenderObjectId id) const {
    if (!id.valid() || id.index >= m_objects.size() ||
        m_objectGen[id.index] != id.generation) {
      return nullptr;
    }
    return &m_objects[id.index];
  }

  RenderPrimitive* RenderObjectDB::getPrimitive(RenderPrimitiveId id) {
    if (!id.valid() || id.index >= m_primitives.size() ||
        m_primitiveGen[id.index] != id.generation) {
      return nullptr;
    }
    return &m_primitives[id.index];
  }

  const RenderPrimitive* RenderObjectDB::getPrimitive(RenderPrimitiveId id) const {
    if (!id.valid() || id.index >= m_primitives.size() ||
        m_primitiveGen[id.index] != id.generation) {
      return nullptr;
    }
    return &m_primitives[id.index];
  }

  // --------------------------------------------------------------------------
  // THE RESOLVER. sec 2: `handle ?: iaIdentity+occurrence`, and it is the only
  // place identity is decided.
  // --------------------------------------------------------------------------

  RenderPrimitiveId RenderObjectDB::resolve(uint64_t iaIdentity,
                                            uint32_t occurrence,
                                            uint64_t engineHandle,
                                            uint32_t frame) {
    // NO IDENTITY, NO OBJECT. Mirrors residentDrawKey's two early returns and
    // for the same reason: a draw with no stable key would be filed under
    // something no other draw agrees with, minting a fresh object every frame.
    // Routing it to the full path forever is the correct answer, not a
    // degradation.
    if (iaIdentity == 0ull) {
      ++m_stats.noIdentity;
      return RenderPrimitiveId();
    }

    ++m_stats.resolves;

    const uint64_t pk = primitiveKey(iaIdentity, occurrence);

    const auto pit = m_primitivesByKey.find(pk);
    if (pit != m_primitivesByKey.end()) {
      RenderPrimitive* p = getPrimitive(pit->second);
      if (p != nullptr) {
        // THE GOOD CASE, and the one slice 1's gate is counting. A hit here is
        // the same draw resolving to the same id it had last frame.
        ++m_stats.hits;

        // A handle arriving for a primitive that already exists is slice 2's
        // path: promote the object's identity without disturbing the primitive.
        if (engineHandle != 0ull) {
          RenderObject* o = getObject(p->owner);
          if (o != nullptr && o->engineHandle != engineHandle) {
            o->engineHandle = engineHandle;
            m_objectsByHandle[engineHandle] = p->owner;
          }
        }

        noteObserved(pit->second, frame);
        return pit->second;
      }
      // Stale handle in the index: the slot was freed and the index outlived
      // it. Drop it and fall through to a fresh resolution rather than
      // returning invalid -- the draw is real and still wants an id.
      m_primitivesByKey.erase(pit);
    }

    // ---- a new primitive. Find or create the object that owns it. ----------

    RenderObjectId owner;
    bool ownerIsNew = false;

    // 1. THE HANDLE WINS WHEN PRESENT. This is the branch that makes the object
    //    level many-to-one, and it is written now so slice 2 only has to start
    //    supplying a non-zero handle.
    if (engineHandle != 0ull) {
      const auto hit = m_objectsByHandle.find(engineHandle);
      if (hit != m_objectsByHandle.end() && getObject(hit->second) != nullptr) {
        owner = hit->second;
      }
    }

    // 2. Otherwise one object per primitive -- see the header on why no
    //    grouping is safe before sec 7 slice B.
    const bool iaKnown = m_objectsByIa.find(iaIdentity) != m_objectsByIa.end();

    if (!owner.valid()) {
      owner = allocObject();
      ownerIsNew = true;

      RenderObject& o = m_objects[owner.index];
      o.iaIdentity  = iaIdentity;
      o.occurrence  = occurrence;
      o.engineHandle = engineHandle;
      o.frameLastObserved = frame;

      if (engineHandle != 0ull) {
        m_objectsByHandle[engineHandle] = owner;
      }
      if (!iaKnown) {
        m_objectsByIa[iaIdentity] = owner;
      }
    }

    // THE SPLIT THAT MAKES THE GATE READABLE, and it is the reason these are
    // two counters instead of one.
    //
    //   newObjects    an iaIdentity never seen before. On a settled scene this
    //                 must fall to ~0 and STAY there with the camera moving. If
    //                 it does not, the identity is chasing something -- which
    //                 is rungs 1 and 2 of sec 1.2's ladder failing again, and
    //                 the only failure slice 1 exists to detect.
    //
    //   ordinalShift  a known iaIdentity turning up under a new occurrence.
    //                 EXPECTED to be non-zero and NOT a defect in the identity:
    //                 sec 1.2 rung 4 records that engine culling removing copy
    //                 1 of a 3-copy identity renumbers the survivors, so the
    //                 ordinal moves under objects that did not. It says "wait
    //                 for the engine handle", where newObjects says "the key is
    //                 wrong". Opposite responses, so they are counted apart.
    //
    // In the 1:1 regime these partition newPrimitives exactly. Once slice 2
    // merges primitives under a handle they stop partitioning it, because a
    // second primitive can join an existing object without being either.
    if (ownerIsNew) {
      if (iaKnown) {
        ++m_stats.ordinalShift;
      } else {
        ++m_stats.newObjects;
      }
    }

    const RenderPrimitiveId pid = allocPrimitive();
    ++m_stats.newPrimitives;

    RenderPrimitive& p = m_primitives[pid.index];
    p.owner = owner;
    p.lookupKey = pk;
    p.frameLastObserved = frame;

    m_primitivesByKey[pk] = pid;

    RenderObject* o = getObject(owner);
    if (o != nullptr) {
      o->primitives.push_back(pid);
      o->frameLastObserved = frame;
    }

    return pid;
  }

  void RenderObjectDB::bindResidentKey(RenderPrimitiveId id, uint64_t residentKey) {
    RenderPrimitive* p = getPrimitive(id);
    if (p == nullptr || residentKey == 0ull) {
      return;
    }

    if (p->residentKey == residentKey) {
      return;
    }

    // Rebinding: drop the old index entry first, or the map keeps naming this
    // primitive under a key it no longer has.
    if (p->residentKey != 0ull) {
      const auto it = m_primitivesByResidentKey.find(p->residentKey);
      if (it != m_primitivesByResidentKey.end() && it->second == id) {
        m_primitivesByResidentKey.erase(it);
      }
    }

    p->residentKey = residentKey;
    m_primitivesByResidentKey[residentKey] = id;
  }

  RenderPrimitiveId RenderObjectDB::findByResidentKey(uint64_t residentKey) const {
    const auto it = m_primitivesByResidentKey.find(residentKey);
    if (it == m_primitivesByResidentKey.end()) {
      return RenderPrimitiveId();
    }
    // Generation-checked like every other lookup: an index entry can outlive
    // the slot it names.
    if (getPrimitive(it->second) == nullptr) {
      return RenderPrimitiveId();
    }
    return it->second;
  }

  void RenderObjectDB::noteObserved(RenderPrimitiveId id, uint32_t frame) {
    RenderPrimitive* p = getPrimitive(id);
    if (p == nullptr) {
      return;
    }
    p->frameLastObserved = frame;

    RenderObject* o = getObject(p->owner);
    if (o != nullptr) {
      o->frameLastObserved = frame;
    }
  }

  void RenderObjectDB::noteChanged(RenderPrimitiveId id, uint32_t frame) {
    RenderPrimitive* p = getPrimitive(id);
    if (p == nullptr) {
      return;
    }
    p->frameLastChanged = frame;
    // Observing is implied by changing -- a test cannot find a primitive
    // different without having looked at it. Stamping both here means a caller
    // that only knows about the change cannot accidentally let the primitive
    // age out.
    p->frameLastObserved = frame;

    RenderObject* o = getObject(p->owner);
    if (o != nullptr) {
      o->frameLastChanged  = frame;
      o->frameLastObserved = frame;
    }
  }

  // --------------------------------------------------------------------------

  void RenderObjectDB::onFrameEnd(uint32_t frame, uint32_t quietFrames, uint32_t maxObjects) {
    // Retire primitives nothing has observed for quietFrames.
    //
    // The comparison is written forward rather than as a subtraction, exactly
    // as BufferSlotTable::reclaim writes it, so a frame id that has not
    // advanced -- or that went backwards across a device reset -- cannot wrap
    // into a very large age and retire the whole store.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_primitives.size()); ++i) {
      RenderPrimitive& p = m_primitives[i];
      if (!p.id.valid()) {
        continue;   // free slot
      }
      if (p.frameLastObserved == kInvalidFrameIndex) {
        continue;   // never observed; nothing to age
      }
      if (frame <= p.frameLastObserved ||
          (frame - p.frameLastObserved) <= quietFrames) {
        continue;
      }

      // Drop the identity index entry too, using the key the primitive was
      // actually filed under -- see RenderPrimitive::lookupKey for why this
      // must not be recomputed from the owner.
      const auto it = m_primitivesByKey.find(p.lookupKey);
      if (it != m_primitivesByKey.end() && it->second == p.id) {
        m_primitivesByKey.erase(it);
      }

      freePrimitive(p.id);
      ++m_stats.retiredPrimitives;
    }

    // Then retire objects left with no primitives. An object outliving its last
    // primitive is not automatically wrong once slice 2 lands -- an enumeration
    // can assert an object exists with no draw behind it, which is sec 1.4's
    // whole point -- so this is deliberately conditioned on having HAD
    // primitives and lost them, rather than on being empty.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_objects.size()); ++i) {
      RenderObject& o = m_objects[i];
      if (!o.id.valid() || !o.primitives.empty()) {
        continue;
      }
      if (o.frameLastObserved == kInvalidFrameIndex) {
        continue;
      }
      if (frame <= o.frameLastObserved ||
          (frame - o.frameLastObserved) <= quietFrames) {
        continue;
      }
      freeObject(o.id);
      ++m_stats.retiredObjects;
    }

    // HARD CEILING. Same argument as ResidentScene's maxRecords: an unbounded
    // store in a churning scene is a leak, and this is the counter that makes
    // the ceiling visible when it fires. Oldest-observed first.
    //
    // If this ever fires in steady state the ceiling is too small for the
    // scene, which is the OPPOSITE finding from identity churn -- read it
    // against newObjects before changing anything.
    if (maxObjects != 0u && objectCount() > maxObjects) {
      std::vector<std::pair<uint32_t, uint32_t>> byAge;   // (frameLastObserved, index)
      byAge.reserve(m_objects.size());
      for (uint32_t i = 0; i < static_cast<uint32_t>(m_objects.size()); ++i) {
        if (m_objects[i].id.valid()) {
          byAge.emplace_back(m_objects[i].frameLastObserved, i);
        }
      }
      std::sort(byAge.begin(), byAge.end());

      size_t excess = objectCount() - maxObjects;
      for (const auto& e : byAge) {
        if (excess == 0u) {
          break;
        }
        RenderObject& o = m_objects[e.second];
        if (!o.id.valid()) {
          continue;
        }
        // Take the object's primitives with it. Leaving them orphaned would
        // strand their index entries and make the next resolve of that draw
        // look like new identity churn rather than like eviction.
        const std::vector<RenderPrimitiveId> owned = o.primitives;
        for (const RenderPrimitiveId& pid : owned) {
          const RenderPrimitive* p = getPrimitive(pid);
          if (p != nullptr) {
            const auto it = m_primitivesByKey.find(p->lookupKey);
            if (it != m_primitivesByKey.end() && it->second == pid) {
              m_primitivesByKey.erase(it);
            }
          }
          freePrimitive(pid);
        }
        freeObject(o.id);
        ++m_stats.evicted;
        --excess;
      }
    }
  }

  void RenderObjectDB::clear() {
    m_objects.clear();
    m_primitives.clear();
    m_objectGen.clear();
    m_primitiveGen.clear();
    m_freeObjects.clear();
    m_freePrimitives.clear();
    m_primitivesByKey.clear();
    m_objectsByHandle.clear();
    m_objectsByIa.clear();
    m_primitivesByResidentKey.clear();
    // Stats survive a clear on purpose, minus the per-window fields that
    // resetStats already handles: a level change is exactly when the cumulative
    // eviction count is most worth having.
  }

}
