#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>

#include "rtx_constants.h"

namespace dxvk {

  class RtInstance;

  // ==========================================================================
  // THE RENDER OBJECT DATABASE -- ARCHITECTURE_OVERHAUL.md slice 1.
  //
  // WHAT IS MISSING, AND IT IS ONLY THIS. The tree already has a geometry
  // database (DrawCallCache), a material database (SparseUniqueCache), a
  // sampler database, a buffer database (BufferSlotTable), an instance database
  // (InstanceManager) and a residency database (ResidentScene). What it does
  // not have is a record that says "this is one thing in the world" whose
  // identity does not depend on where that thing is. Every recurring defect in
  // ARCHITECTURE_OVERHAUL sec 1.2's identity ladder is a consequence of the
  // absence: with no object, the pipeline re-derives which thing a draw is from
  // the only evidence it has, which is where the draw put it.
  //
  // SLICE 1 IS A MIRROR AND NOTHING ELSE. This file adds no behaviour. It runs
  // beside ResidentScene, is fed from the same resolution point, and exists to
  // prove ONE property before anything is built on it:
  //
  //     the same draw resolves to the same RenderObjectId across frames
  //     and across camera motion
  //
  // That is slice 1's entire acceptance gate. Nothing downstream may consume an
  // id until newObjects/newPrimitives read ~0 on a settled scene and stay flat
  // through a pitch-and-yaw sweep, which is the same discipline and the same
  // sweep the resident gate was armed under.
  //
  // ------------------------------------------------------------------------
  // WHY THE OBJECT LEVEL IS 1:1 WITH THE PRIMITIVE TODAY, AND WHY THAT IS NOT
  // A SHORTCUT.
  //
  // ARCHITECTURE_OVERHAUL sec 1.3 specifies three levels and forces the middle
  // one by measurement: [Join] reads 64-137 gate calls per frame against
  // 403-577 draws, i.e. four to six draws per renderable, so one object maps to
  // many (geometry, material) pairs and a flat record would grow a variant per
  // draw. That is right, and the grouping still cannot be built yet, because
  // THERE IS NO SAFE GROUPING SIGNAL IN THE TREE.
  //
  // The obvious candidate is the buffer identity -- group primitives that share
  // (vbPtr, ibPtr, ilPtr) and differ only in draw range. residentDrawKey's own
  // comment refutes it in one line: "TF2 sub-allocates many meshes out of one
  // pooled buffer and the pointer alone would merge them"
  // (d3d11_rtx.cpp, residentDrawKey, the field-list comment). Distinct objects
  // share a pooled buffer, so that grouping merges objects rather than meshes,
  // and it fails silently -- which is the exact shape of every rung on the
  // identity ladder that has already broken.
  //
  // So the honest position is: the object level EXISTS, is allocated one per
  // primitive, and is populated with a grouping only when sec 7 slice B lands
  // an engine handle. sec 1.3's own priority list says the same thing --
  // engineHandle is "authoritative when present" and occurrence is used "ONLY
  // while engineHandle is absent". Slice 2 fills RenderObject::engineHandle and
  // starts merging primitives under one object; nothing here has to change
  // shape for that to happen, which is the reason to build the level now rather
  // than to flatten it and reintroduce it later.
  //
  // ------------------------------------------------------------------------
  // WHAT IS DELIBERATELY NOT HERE.
  //
  // NOT AN ECS. Two record types and two owning vectors. No components, no
  // archetypes, no systems, no generic render world. sec 8.
  //
  // NO POSE FIELDS YET. sec 1.3 shows objectToWorld / prevObjectToWorld /
  // worldBounds on RenderObject. They are omitted until slice 7 has a producer
  // for them: a Matrix4 pair plus an AABB per object that nothing writes is
  // memory proportional to the resident set bought for nothing, and a field
  // that is present but always identity is worse than an absent one because it
  // reads as populated. Add them with the job that fills them.
  //
  // NO INSTANCE LIST. ResidentScene::Record already owns the RtInstance* list,
  // the back-pointer contract that keeps it from dangling, and the invalidation
  // path. Duplicating it here would mean two lifetime contracts over the same
  // pointers, which is the failure class RtInstance::m_residentKey exists to
  // close. RenderPrimitive joins to that record by key instead.
  //
  // THREADING. Same rule as ResidentScene: RT/CS side only, no lock. The
  // resolver is fed from processDrawCallState, which is already the CS-side
  // point where "did this draw resolve to any instance" is answerable, and that
  // is the only safe evidence -- see RESIDENT_SCENE_PLAN sec 2.1.
  // ==========================================================================

  // Generational handles. sec 0.1 item 4 established that a bare index into a
  // recycled slot is an ABA generator, and invariant I2 requires every
  // cross-frame reference to be either a back-pointer its owner clears or an
  // index carrying a generation. These are the second kind. The generation is
  // bumped on FREE, not on allocate, so a handle to a freed slot is stale even
  // if the slot is never reused.
  struct RenderObjectId {
    static constexpr uint32_t kInvalidIndex = ~0u;

    uint32_t index = kInvalidIndex;
    uint32_t generation = 0u;

    bool valid() const { return index != kInvalidIndex; }
    bool operator==(const RenderObjectId& o) const {
      return index == o.index && generation == o.generation;
    }
    bool operator!=(const RenderObjectId& o) const { return !(*this == o); }
  };

  struct RenderPrimitiveId {
    static constexpr uint32_t kInvalidIndex = ~0u;

    uint32_t index = kInvalidIndex;
    uint32_t generation = 0u;

    bool valid() const { return index != kInvalidIndex; }
    bool operator==(const RenderPrimitiveId& o) const {
      return index == o.index && generation == o.generation;
    }
    bool operator!=(const RenderPrimitiveId& o) const { return !(*this == o); }
  };

  // ONE per engine handle -- and, until sec 7 slice B, one per primitive. See
  // the class comment for why that is not a flattening.
  struct RenderObject {
    RenderObjectId id;

    // WHERE IDENTITY COMES FROM, in sec 1.3's priority order. The resolver
    // fills the best one available and records which; the object is the join.
    //
    // engineHandle  sec 7 slice B. Authoritative when present. 0 = absent.
    // iaIdentity    residentDrawKey's baseKey -- the XXH64 of ResidentKeyHead.
    //               Always available, camera- and animation-invariant BY
    //               CONSTRUCTION (the fields are engine allocations made at
    //               level/entity spawn), which is exactly the property rungs 1
    //               and 2 of the ladder lacked.
    // occurrence    which copy of a multi-copy identity this is. Only
    //               meaningful while engineHandle is absent, and it is the
    //               known-weak part: sec 1.2 rung 4 records that the ordinal
    //               shifts when engine culling removes copy 1 of a 3-copy
    //               identity. Carried, not trusted.
    uint64_t engineHandle = 0ull;
    uint64_t iaIdentity   = 0ull;
    uint32_t occurrence   = 0u;

    // A draw or an enumeration touched it. Note the asymmetry that sec 1.4
    // makes a rule: a draw is an OBSERVATION, an enumeration is an ASSERTION,
    // and neither is the object. Both land here.
    uint32_t frameLastObserved = kInvalidFrameIndex;
    // An OUTPUT test said something about it differed. Distinct from observed
    // on purpose -- I4 sizes work by what changed, not by what was submitted,
    // and the two numbers are what make that expressible.
    uint32_t frameLastChanged  = kInvalidFrameIndex;

    // The primitives this object draws with. One entry until slice 2; sized for
    // the 4-6 that sec 1.3 measures per renderable so the grow does not happen
    // on the frame slice 2 lands.
    std::vector<RenderPrimitiveId> primitives;
  };

  // ONE per (geometry, material) the object draws with -- the unit the geometry
  // and material caches are already keyed on, and the level ResidentScene's
  // Record is already at.
  struct RenderPrimitive {
    RenderPrimitiveId id;
    RenderObjectId    owner;

    // THE JOIN TO ResidentScene, and it is a key rather than a pointer on
    // purpose. Record lives in an unordered_map that rehashes, so a Record*
    // held across frames is a dangling pointer the first time the map grows.
    // The key is stable and the lookup is O(1).
    uint64_t residentKey = 0ull;

    // THE KEY THIS PRIMITIVE IS FILED UNDER in m_primitivesByKey, stored rather
    // than recomputed from the owner.
    //
    // Recomputing it as primitiveKey(owner->iaIdentity, owner->occurrence) is
    // correct ONLY in today's 1:1 regime and silently wrong the moment slice 2
    // merges several primitives under one object: the owner then carries one of
    // its primitives' occurrences and the retire path would erase the wrong
    // index entry -- or none -- leaving a stale entry that makes the next
    // resolve of that draw read as fresh identity churn. That is precisely the
    // reading slice 1 exists to make trustworthy, so it must not be possible to
    // corrupt it by landing slice 2.
    uint64_t lookupKey = 0ull;

    uint32_t frameLastObserved = kInvalidFrameIndex;
    uint32_t frameLastChanged  = kInvalidFrameIndex;
  };

  class RenderObjectDB {
  public:
    RenderObjectDB() = default;

    // ----------------------------------------------------------------------
    // THE RESOLVER. This is the only place identity is decided, which is the
    // whole point of sec 2's diagram: an observation goes in, an id comes out,
    // and nothing downstream re-derives it.
    //
    // Returns an invalid id for iaIdentity == 0, which is residentDrawKey's
    // no-identity sentinel (no vertex buffer bound, or a DYNAMIC one the CPU
    // rewrites). Those draws have nothing stable to key on and minting an id
    // for them would file them under something no other draw agrees with --
    // the same reasoning, and the same answer, as residentDrawKey's own two
    // early returns.
    RenderPrimitiveId resolve(uint64_t iaIdentity,
                              uint32_t occurrence,
                              uint64_t engineHandle,
                              uint32_t frame);

    // Attach the ResidentScene key a resolved draw was filed under. Separate
    // from resolve() because the two are known at different points: the
    // identity is available at the gate, the resident key only after the draw
    // has actually resolved to instances.
    void bindResidentKey(RenderPrimitiveId id, uint64_t residentKey);

    RenderPrimitiveId findByResidentKey(uint64_t residentKey) const;

    // ----------------------------------------------------------------------
    // Accessors. Return nullptr on a stale handle -- that IS the generation
    // check, and it is the reason the handles carry one.
    RenderObject*    getObject(RenderObjectId id);
    const RenderObject* getObject(RenderObjectId id) const;
    RenderPrimitive* getPrimitive(RenderPrimitiveId id);
    const RenderPrimitive* getPrimitive(RenderPrimitiveId id) const;

    // ----------------------------------------------------------------------
    // The two halves of I4's "work is sized by what changed". noteObserved says
    // a draw or an enumeration named this primitive; noteChanged says an output
    // test found it different. Both stamp the owning object as well, because
    // the object's frameLast* are the max over its primitives and slice 7 reads
    // them per object.
    void noteObserved(RenderPrimitiveId id, uint32_t frame);
    void noteChanged(RenderPrimitiveId id, uint32_t frame);

    // ----------------------------------------------------------------------
    // Retire primitives nothing has observed for quietFrames, then their
    // objects once empty. Evidence-based, NOT age-based: quietFrames is a bound
    // on how long an observation stays good, and the observation itself is what
    // resets it. That distinction is the one sec 0.1 item 4 got wrong about
    // BufferSlotTable, and it is worth being explicit about here so the next
    // reader does not have to re-derive it.
    //
    // maxObjects is a hard ceiling for the same reason ResidentScene has one:
    // an unbounded store in a churning scene is a leak, and a policy that never
    // fires has to be visible as never having fired.
    void onFrameEnd(uint32_t frame, uint32_t quietFrames, uint32_t maxObjects);

    void clear();

    // ----------------------------------------------------------------------
    struct Stats {
      uint32_t resolves = 0;      // calls that produced a valid id
      uint32_t noIdentity = 0;    // iaIdentity == 0, routed to the full path

      // THE ACCEPTANCE GATE. Read exactly as [ResidentGate] newKeys is read: on
      // a settled scene both must be ~0, and they must stay ~0 with the camera
      // moving. A climbing newPrimitives means the same real draw is being
      // given a fresh identity every frame, which is rungs 1 and 2 of the
      // ladder failing again and is the ONLY thing slice 1 is trying to detect.
      uint32_t newObjects = 0;
      uint32_t newPrimitives = 0;

      // Resolves that found an existing primitive -- the good case.
      uint32_t hits = 0;

      // A primitive whose occurrence changed under a stable iaIdentity. This is
      // sec 1.2 rung 4's known defect made countable: engine culling removing
      // one copy of a multi-copy identity renumbers the survivors. NOT folded
      // into newPrimitives, because the response is different -- a high
      // ordinalShift says wait for the engine handle, a high newPrimitives says
      // the identity itself is churning.
      uint32_t ordinalShift = 0;

      uint32_t retiredPrimitives = 0;
      uint32_t retiredObjects = 0;
      // CUMULATIVE and deliberately not reset with the rest, same argument as
      // ResidentScene::Stats::wiped: an eviction that happens and then reads 0
      // by the time the line comes out is how a policy hides.
      uint32_t evicted = 0;
    };

    const Stats& stats() const { return m_stats; }
    void resetStats() {
      const uint32_t keepEvicted = m_stats.evicted;
      m_stats = Stats();
      m_stats.evicted = keepEvicted;
    }

    size_t objectCount() const { return m_objects.size() - m_freeObjects.size(); }
    size_t primitiveCount() const { return m_primitives.size() - m_freePrimitives.size(); }

  private:
    RenderObjectId    allocObject();
    RenderPrimitiveId allocPrimitive();
    void freeObject(RenderObjectId id);
    void freePrimitive(RenderPrimitiveId id);

    // The identity a primitive is looked up by while engineHandle is absent.
    // Folding the occurrence into the map key rather than searching a per
    // identity list keeps resolve() O(1), and it is what makes ordinalShift
    // detectable: a shifted ordinal is a miss on this map with a hit on
    // m_objectsByIa.
    static uint64_t primitiveKey(uint64_t iaIdentity, uint32_t occurrence);

    // Parallel arrays, index-addressed, generation-checked. Slots are reused
    // through the free lists; the generation on the SLOT is bumped when it is
    // freed, so every outstanding handle to it becomes stale at that moment.
    std::vector<RenderObject>    m_objects;
    std::vector<RenderPrimitive> m_primitives;
    std::vector<uint32_t> m_objectGen;
    std::vector<uint32_t> m_primitiveGen;
    std::vector<uint32_t> m_freeObjects;
    std::vector<uint32_t> m_freePrimitives;

    std::unordered_map<uint64_t, RenderPrimitiveId> m_primitivesByKey;

    // THE MERGE ANCHOR, and the reason resolve() will not need rewriting when
    // slice 2 lands. sec 2's resolver is `handle ?: iaIdentity+occurrence`, and
    // that is written out in full below even though nothing supplies a handle
    // yet: when one arrives, several primitives find the SAME object here and
    // the 1:1 relation becomes the many-to-one sec 1.3 specifies, with no
    // change to the resolver's shape and no migration of the stored records.
    //
    // Empty today. That is the honest state, not an oversight -- see the class
    // comment on why no grouping signal exists before sec 7 slice B.
    std::unordered_map<uint64_t, RenderObjectId>    m_objectsByHandle;

    // Every iaIdentity ever resolved, to the object of its first-seen
    // occurrence. Used for two things and neither is grouping: telling a
    // genuinely NEW identity from a new/renumbered COPY of a known one (which
    // is what splits newObjects from ordinalShift), and giving slice 2 an
    // anchor to merge against when a handle first appears for an identity that
    // already has primitives.
    std::unordered_map<uint64_t, RenderObjectId>    m_objectsByIa;

    std::unordered_map<uint64_t, RenderPrimitiveId> m_primitivesByResidentKey;

    Stats m_stats;
  };

}
