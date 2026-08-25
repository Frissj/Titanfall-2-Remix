# THE ARCHITECTURE OVERHAUL

**Written 2026-08-25.** Assumes RESIDENT_SCENE_PLAN.md is finished — slices 1-8
landed, `verify` flipped off, `FAIL=0` held across a pitch-and-yaw sweep, and
§7 slices B-D landed so liveness comes from an engine enumeration rather than
from a draw arriving. Nothing below is worth starting before that, because every
one of these changes assumes the scene outlives the frame.

This document supersedes nothing. RESIDENT_SCENE_PLAN is the *scene lifetime*
plan and stays authoritative for it; OPTIMISATION_PLAN_3 is the *cost* plan.
This is the *shape* plan: what the pipeline should be made of once objects
persist, and it is deliberately written without regard to which line is hot
today.

Every claim here that is about this tree carries the file and line it was read
from. Claims taken from elsewhere are marked as such and are not load-bearing.

---

## 0. THE VERDICT, IN ONE PARAGRAPH

The external advice is right about the destination and wrong about the distance.
"Move from a draw-centric renderer to a persistent render database" is correct,
and it is also mostly **already built** — this tree has a geometry database
(`DrawCallCache`), a material database (`SparseUniqueCache<RtSurfaceMaterial>`),
a sampler database, a buffer database (`BufferSlotTable`), an instance database
(`InstanceManager`) and now a residency database (`ResidentScene`). What it does
not have, and what every recurring defect in the last four months traces back
to, is **an object identity that does not depend on where the object is**.
Instance identity is currently a hash of the composed object-to-world matrix.
That is the single missing abstraction. Everything else in this document is
either a consequence of adding it or a deletion it makes possible.

---

## 0.1 WHAT WAS VERIFIED, AND WHERE THE ADVICE WAS WRONG

Read this before acting on any external suggestion. Six of them do not survive
contact with the code, and two of the six would have cost a rewrite.

### Wrong, and the correction matters

**1. `ExtractTransforms()` is not ~4,950 lines. It is 9,252.**
`src/d3d11/d3d11_rtx.cpp:19047-28298`. Of those, 3,655 are comment-only and 162
blank, so ~5,435 are code — which is where the 4,950 figure came from and why it
reads as roughly right while understating the file by 2x. The comments are not
padding: this tree's convention is that a comment records why the obvious
approach was rejected, and several of them are the only surviving record of a
refuted hypothesis. **Do not "clean up" the comments as part of a split.**

The whole per-draw path is larger than any of the advice assumes:

```
SubmitDrawDeferred   9,922 lines   d3d11_rtx.cpp:54068
EndFrame             9,882         d3d11_rtx.cpp:67009
ExtractTransforms    9,259         d3d11_rtx.cpp:19047
SubmitDrawTail       6,180         d3d11_rtx.cpp:47888
SubmitDraw           5,542         d3d11_rtx.cpp:42346
                    ------
                    40,785 lines in five functions
```

`d3d11_rtx.cpp` is 76,898 lines. Five functions are 53% of it. Any plan that
says "split `ExtractTransforms`" and stops there has addressed 23% of the
problem.

**2. `DrawSnapshot` and `ResidentScene` are not a duplication to be merged.**
The advice reads their separation as "an implementation-stage separation, not
the ideal final architecture". It is neither. The gate deliberately runs off
live `m_context->m_state` *before* any snapshot exists, because the whole point
is to reach a verdict without paying for the capture; and RESIDENT_SCENE_PLAN
§2.1 then corrected itself to move the *decision* to the CS thread
(`rtx_context.cpp:5508`) because "did this draw resolve to any instance" is the
only safe skip evidence and only CS can see it. The two-map, two-thread split is
load-bearing and the reason is written down. Leave it.

What IS true is narrower: the frame-thread gate computes an identity
(`residentDrawKey`, `d3d11_rtx.cpp:40063`) that duplicates what `DrawSnapshot`
already holds, and both duplicate what the engine handle will supply after §7
slice C. That converges on its own — see §2.

**3. "Stop copying constant-buffer bytes, reference stable storage instead."**
The code has already reasoned this out and left the argument in place at
`d3d11_rtx.h:185-208`. Two mechanisms make a deferred read return different
bytes than the draw used, silently, as a wrong transform:

- `Map(D3D11_MAP_WRITE_NO_OVERWRITE)` hands the game back the same `mapPtr` and
  it writes in place, bumping neither the pointer nor `contentGen`;
- a renamed-away slice can return to the allocator and be handed out again
  inside the same frame.

Neither is settleable by argument, which is why `CbRange::srcPtr` and the
PinDefer probe exist: record where the read happened, re-read at frame end, and
bit-compare. **This is a measurement that has not been taken, not an
optimisation waiting to be applied.** Take it before believing the idea.

**4. "Generational handles everywhere."** Mostly already solved, by a better
mechanism, and the one place that genuinely needs a generation is not the one
named. `BufferSlotTable::retain()` (`rtx_utils.h:262`) does not compare against a
remembered buffer — it asks the table whether the slot still holds *this* buffer:

```cpp
if (idx >= m_table.size() || !m_table[idx].defined() || !m_table[idx].matches(b))
  return false;
```

That is strictly stronger than a generation counter for the ABA it prevents, and
cheaper to reason about. **But `reclaim()` (`rtx_utils.h:305`) frees a retired
slot on frame age** — `(frameId - m_lastRefFrame[idx]) > quietFrames` — and
pushes the index onto `m_freeSlots` with no generation bump. A surface uploaded
with a stale index into a slot that has since been reclaimed and re-issued reads
the wrong buffer, and the only thing that catches it is the `[StaleTape] dead`
counter, i.e. a diagnostic rather than a structure.

That is also an internal contradiction worth naming: RESIDENT_SCENE_PLAN §0.2
says "recycling a slot on frame age would eventually hand a live wrong buffer to
an instance that outlived the guess, and residency's whole premise is that an
instance's age is not a bound on its life" — and then reclaim recycles on frame
age. One 32-bit generation on the slot, packed into the surface's buffer index,
turns `[StaleTape] dead` from a thing you watch into a thing that cannot happen.
**One site. Not "everywhere".**

**5. "Introduce a job/task graph."** This one is correct and is genuinely
absent. Confirmed: `WorkerThreadPool::Schedule` (`util/util_threadpool.h:346`)
takes a closure and returns a bare `Future<R>`. There is no dependency counter,
no continuation, no way for a job to make another job runnable.
`flushGeometryBatch` (`d3d11_rtx.cpp:34569`) is one chunked fork-join — schedule
N chunks, then `futs[fi].get()` in order, a busy spin with no timeout — plus a
full `SynchronizeCsThread(SynchronizeAll)` drain at `d3d11_rtx.cpp:34813`. Every
phase is separated by a global barrier. See §4.

**6. "Add GPU-driven culling."** It already exists, works, and is in the tree.
`rtx_point_instancer_system.h:40-60` describes it exactly: the CPU pushes
`mask=0` placeholder instances, a compute shader evaluates each transform
against the camera and overwrites the visible placeholders with full
`VkAccelerationStructureInstanceKHR` entries, culled entries stay `mask=0` and
the RT hardware skips them. Its own comment closes with *"No CPU-side transform
iteration occurs."* It is scoped to USD PointInstancer replacements. Generalising
it to the resident set is an extension of working code, not new architecture.
See §5.3.

### Right, and understated

**7. The GPU scene is rebuilt from scratch every frame, and residency makes that
worse in exact proportion to how well residency works.**

```
rtx_accel_manager.cpp:7946   ctx->writeToBuffer(m_surfaceBuffer, 0, surfacesGPUData.size(), ...)
rtx_accel_manager.cpp:2718   ctx->writeToBuffer(m_transformBuffer, 0, all transforms, ...)
rtx_accel_manager.cpp:9380   buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR   // TLAS
```

`kSurfaceGPUSize` is 256 B (`rtx_materials.h:45`). The surface table is filled by
a serial CPU loop over `m_reorderedSurfaces` and uploaded whole, every frame; the
transform buffer likewise; the TLAS is a full rebuild, never a refit. All three
are O(scene), not O(changed).

This is the same shape as the §0.2 buffer-identity bug and it has the same
property: **the better residency works, the bigger the resident set, and the
more this costs.** Today `m_reorderedSurfaces` is measured in the thousands —
`rtx_accel_manager.cpp:1120` describes "all 15.5k pointers every frame to order
63 of them". Residency's stated goal is to restore the whole level's off-screen
population. RESIDENT_SCENE_PLAN does not mention this anywhere, and it is the
most likely way a fully-landed residency measures as a regression.

**8. Instance identity is a hash of the composed object-to-world matrix.**
`rtx_instance_manager.h:49-80`: *"findSimilarInstance hashes the composed
object-to-world matrix to do its exact lookup"*, and `[MapGate]` measures 15,447
spatial-map writes per frame against 15,488 lookups. `SpatialMap<RtInstance>`
(`util/util_spatial_map.h:40`) keys entries on `transformHash`.

This is §1's whole argument and it is not a design accident — it is what you do
when the only thing you know about an object is where it was drawn. It stops
being necessary the moment §7's engine handle exists.

The tree already knows this is failing. `util_spatial_map.h:62-81` carries a
2026-08-24 write-side ledger built for exactly one open bug: *"a stationary prop
whose exact lookup misses even though its transform has not changed. Two
read-side theories have already been refuted."*

**9. There are 893 distinct diagnostic probe tags and 910 `Logger::` calls in
`d3d11_rtx.cpp` alone.** 75 distinct tags inside `ExtractTransforms` by itself.
Most name investigations this tree's own records mark as RESOLVED. This is not a
style complaint: it is why the five functions are 10,000 lines each, and it is
the largest single obstacle to any of the splits below. See §6.

### Not verifiable here

**10. The SER claim.** The recollection is that "RE Engine restructured to use
SER and path tracing went 14 ms → 3 ms". I cannot confirm that figure or that
engine and will not repeat it. The well-published case is Cyberpunk 2077's
RT: Overdrive mode, where NVIDIA reported roughly 44% faster path tracing from
Shader Execution Reordering — a large win, and one that came from restructuring
so there was divergent work *after* the reorder point and useful coherence hints
*at* it, not from switching a flag on.

**The actionable half is in this tree, and it is not a flag.**
`rtx.enableShaderExecutionReorderingInPathtracerGbuffer` is set `True` in the
live `rtx.conf` and does nothing, because both G-buffer payload states hard-stub
the predicate:

```
shaders/rtx/algorithm/geometry_resolver_state.slangh:182   GeometryResolverState::shouldReorder    -> return false;
shaders/rtx/algorithm/geometry_resolver_state.slangh:366   GeometryPSRResolverState::shouldReorder -> return false;
```

The option's own doc string says so — *"(Note: Hard disabled in shader code)"*
(`rtx_options.h:2628`) — and its default is `false`; the conf overrides the
default to `True` and the shader ignores it. So the G-buffer pass, the one with
the divergence problem, never calls `ReorderThread`. Only `PathState::shouldReorder`
(`path_state.slangh:487`, integrate-indirect) returns true, and it emits at most
two coherence-hint bits: separate-unordered-approximations-active, and
`isNrcUpdate`. Neither is a material or a hit/miss classification.

See §5.4. This is the highest-value GPU item in the document and it is one
predicate function, not a rewrite.

---

## 1. THE ONE MISSING ABSTRACTION

### 1.1 The databases exist. The object does not.

```
GeometryDB    DrawCallCache                     hash -> BlasEntry, stable ptrs    rtx_draw_call_cache.h:41
MaterialDB    SparseUniqueCache<RtSurfaceMaterial>                                rtx_scene_manager.h:122
SamplerDB     SparseUniqueCache<Rc<DxvkSampler>>                                  rtx_scene_manager.h:155
BufferDB      BufferSlotTable<RaytraceBuffer>   stable slots, evidence-freed      rtx_scene_manager.h:115
InstanceDB    InstanceManager::m_instances                                        rtx_instance_manager.h
ResidencyDB   ResidentScene                     key -> instances, valid           rtx_resident_scene.h

RenderObject  -- DOES NOT EXIST --
```

There is no record that says *"this is one thing in the world; here is its
geometry, its material, its pose, its lifetime"*. So every frame the pipeline
re-derives which thing a draw is, from the only evidence it has, which is where
the draw put it.

### 1.2 The identity ladder, and why every rung broke the same way

This tree has now tried five identities. The order is the finding.

| # | identity | where | how it failed |
|---|---|---|---|
| 1 | hash of input constant bytes | (abandoned) | *"the bytes track the camera while the matrix they produce does not"* — ~97% of churn was a key chasing something its own value did not do. `d3d11_rtx.cpp:29790` |
| 2 | hash of composed `objectToWorld` | `SpatialMap`, **still live** | identity is a function of position; a mover re-keys, a jitter re-keys, and a stationary prop can still miss (`util_spatial_map.h:65`) |
| 3 | `stablePropId` | `rtx_instance_manager.h:1085` | `noProp=100%`, 27,274 of 27,274 draws — producers only cover bone-anim and sky, and `ExtractTransforms` runs before any of them |
| 4 | `IdentHead`/`KeyHead` + occurrence ordinal | `d3d11_rtx.cpp:29815`, gate at `:40063` | camera-invariant and good — 46-93 new keys/frame vs 933-959 for the byte key — but the ordinal shifts when engine culling removes copy 1 of a 3-copy identity, and then copies 2 and 3 serve records built for different physical objects (RESIDENT_SCENE_PLAN §7.6 item 4) |
| 5 | engine handle (`IClientRenderable*`, static-prop index) | §7 slice B/C, **not built** | the enumeration site is unsolved: `sub_1801A8350`'s return is post-cull (§7.3) |

Rungs 1 and 2 fail for one reason: **the key is derived from a value that
changes.** Rung 3 fails because the producer runs after the consumer. Rung 4 is
the best available without the engine and its residual defect is *created by the
thing residency exists to work around* — engine culling perturbing the
occurrence numbering.

Rung 5 is the only one whose failure mode is "we have not found the list yet"
rather than "the key is chasing something".

**So the architecture's foundation stone is RESIDENT_SCENE_PLAN §7 slice B.** Not
the job graph, not the GPU scene, not the shader split. Every other item in this
document gets better with a stable object identity and several of them are not
worth building without one.

### 1.3 What a RenderObject is here, and what it deliberately is not

Not an ECS. Not components, not archetypes, not a generic render world. One
concrete record, TF2-shaped, sized to what this tree already needs:

```cpp
struct RenderObjectId { uint32_t index; uint32_t generation; };

struct RenderObject {
  RenderObjectId   id;

  // WHERE IDENTITY COMES FROM, in priority order. The resolver fills the best
  // one available and records which; the object is the join of all of them.
  EngineHandle     engineHandle;   // §7 slice B. authoritative when present
  DrawIdentity     iaIdentity;     // KeyHead. always available, from the draw
  uint32_t         occurrence;     // ONLY while engineHandle is absent

  GeometryId       geometry;       // -> DrawCallCache / BlasEntry
  MaterialId       material;       // -> SparseUniqueCache slot
  Matrix4          objectToWorld;
  Matrix4          prevObjectToWorld;
  AABB             worldBounds;

  uint32_t         frameLastObserved;   // a draw or an enumeration touched it
  uint32_t         frameLastChanged;    // an OUTPUT test said it differed
  ObjectFlags      flags;               // skipUnsafe, everMoved, seenCameraMask
  InstanceRef      instances;           // what ResidentScene::Record holds today
};
```

`ResidentScene::Record` is 90% of this already — it holds instances, a validity
bit, a camera mask, source buffer addresses and `skipUnsafe`. **The migration is
to give that record an identity that is not a draw key, and then let everything
else point at it.** That is a rename plus a resolver, not a rewrite.

### 1.4 The inversion, stated as a rule you can check code against

RESIDENT_SCENE_PLAN §0 demoted the draw to a change signal. §7.8 went further:
*"once the output test exists, absence of a draw does not need to say anything."*
The general rule that falls out, and the one to hold every later change to:

```
A draw is an OBSERVATION of an object.
An enumeration is an ASSERTION that an object exists.
Neither is the object.

Nothing downstream of the resolver may take the draw as its unit of work.
```

Concretely, three things violate that today and are the slice-7 work list:

- geometry decisions are made per draw, so one mesh appearing in 15 draws makes
  15 geometry decisions (`DrawCallCache::get` per `DrawCallState`);
- material resolution runs per draw (`FillMaterialData`, 2,811 lines,
  `d3d11_rtx.cpp:35273`), and its result is content-hashed into a unique cache
  afterwards — i.e. the work is done and then discovered to be redundant;
- instance resolution runs per draw and re-derives identity from position.

---

## 2. THE TARGET ARCHITECTURE

```
                              TITANFALL 2
                                   │
              ┌────────────────────┴────────────────────┐
              │                                          │
     ENGINE ENUMERATION                         D3D11 INTERCEPTION
     pre-cull object list                       DrawIndexed / Draw
     (§7 slice B/F)                                      │
     handle · pose · liveness                            │
              │                                          ▼
              │                              ┌───────────────────────┐
              │                              │  OBSERVE (frame thd)  │
              │                              │  capture, derive      │
              │                              │  NOTHING, decide      │
              │                              │  NOTHING              │
              │                              │  -> DrawObservation   │
              │                              └───────────┬───────────┘
              │                                          │
              └──────────────────┬───────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │    OBJECT RESOLVER    │   handle ?: iaIdentity+occurrence
                     │  observation -> id    │   the ONLY place identity is decided
                     └───────────┬───────────┘
                                 ▼
              ╔══════════════════════════════════════════╗
              ║          THE PERSISTENT SCENE            ║
              ║                                          ║
              ║  RenderObjectDB   <- the new part        ║
              ║  GeometryDB       DrawCallCache          ║
              ║  MaterialDB       SparseUniqueCache      ║
              ║  BufferDB         BufferSlotTable        ║
              ║  InstanceDB       InstanceManager        ║
              ║  BlasDB           PooledBlas / BlasEntry ║
              ╚══════════════════╤═══════════════════════╝
                                 │
                          THE OUTPUT TEST
                    geometry content · transform ·
                    material · destruction   (§7.8)
                                 │
                        the CHANGED set only
                                 ▼
                     ┌───────────────────────┐
                     │       JOB GRAPH       │   dependency counters, §4
                     │  transform · bounds   │   composable, spawns children
                     │  skin · material      │   ordering is a DEPENDENCY,
                     │  geometry · instance  │   never a barrier
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │      GPU SCENE        │   §5
                     │  surfaces (delta)     │   upload what changed
                     │  transforms (delta)   │   not the whole table
                     │  instance records     │
                     └───────────┬───────────┘
                                 ▼
                        GPU VISIBILITY               generalised from
                        mask = 0 for culled          rtx_point_instancer_system
                                 │
                                 ▼
                        TLAS  (refit where legal)
                                 │
                                 ▼
                        PATH TRACE  (SER re-enabled, §5.4)
```

Three boundaries, and they are the whole design:

```
the SCENE       says WHAT EXISTS         persistent, survives frames
the OUTPUT TEST says WHAT CHANGED        per frame, cheap, per object
the FRAME       says WHAT MUST RUN       jobs + GPU work, sized by the changed set
```

Today all three are entangled inside `SubmitDraw`.

---

## 3. THE INVARIANTS

These are what make an implementation correct rather than merely faster. Each is
checkable, and each has a named failure this tree has already shipped.

**I1 — Identity is never derived from a value that changes.**
Not the transform, not the constant bytes, not the frame's draw ordinal once a
handle exists. *Broke as:* the o2w object key (~97% churn), the SpatialMap
transform hash (still live).

**I2 — A handle's validity is structural, not observed.**
Every cross-frame reference is either a back-pointer that its owner's destructor
clears (`RtInstance::m_residentKey`, `m_batchRecordKey`) or an index carrying a
generation. Never a raw pointer plus a frame-age guess. *Broke as:*
`s_zigGunInstance` UAF, the `getImageHash` GC-walk race, and — still open —
`BufferSlotTable::reclaim()`.

**I3 — Capture is a copy or a proof, never a hope.**
Anything read off the frame thread and consumed later is either copied at
capture, or the code carries a proof that the source is immutable until
consumption. *Broke as:* the dropship `COLOR1.y` `WRITE_DISCARD` rename. This is
why `DrawSnapshot` copies bytes and why §0.1 item 3 is a measurement rather than
a refactor.

**I4 — Work is sized by what changed, not by what was submitted.**
One mesh in 15 draws is one geometry decision. One unchanged object is zero
work, CPU and GPU. *Broke as:* full surface-table upload, full TLAS rebuild,
per-draw material resolution feeding a dedup cache.

**I5 — Ordering is a dependency, not a barrier.**
A draw that depends on the previous draw's camera state gets an edge. It does
not get "run this phase on the frame thread". *Broke as:*
`SynchronizeCsThread(SynchronizeAll)` at every flush.

**I6 — A class test may be a SAFETY floor and may never be a liveness rule.**
`skipUnsafe` asks the instances what they turned out to be, which is correct. A
list of shader hashes or a "Dynamic class" check is not. *Stated already* at
RESIDENT_SCENE_PLAN §2.2 and §7.8; promoted here because §5's GPU work will be
tempted to re-introduce one.

**I7 — Every exemption must name the shorter lifetimes running underneath it.**
*Broke as:* `numFramesToKeepBLAS=1` silently reaping the instances residency had
just exempted (RESIDENT_SCENE_PLAN §2.3.1, trap 8). Applies to every new
persistent thing below: a GPU surface slot, a delta-upload dirty bit, a resident
BLAS.

**I8 — Verify first, skip second.**
Every stage lands with a verify mode that runs both paths and scores the
prediction, and nothing is skipped until FAIL reads 0 across a full pitch-and-yaw
sweep. This tree has used it four times and it has caught something every time.

**I9 — Only a proven pre-cull source may assert that an object is dead.**
Existence and visibility are two different claims and the code must not be able
to confuse them. See §3.1 — this is the one invariant here that can empty a
scene, and RESIDENT_SCENE_PLAN as written walks into it.

### 3.1 EXISTENCE AUTHORITY vs VISIBILITY OBSERVATION

**The hazard, stated exactly.** RESIDENT_SCENE_PLAN §7.3 measured
`sub_1801A8350`'s return as **post-cull** — Gate 1, the view-direction-dependent
leaf-frustum test, runs INSIDE it, so its output list is what survived culling.
§7.5 slice D then specifies *"`onFrameEnd` touches every record whose handle is
listed, draw or no draw, and **absent handles invalidate**"*.

Those two sentences in the same document are a scene-emptying bug. Wire slice D
to a post-cull list and every off-screen object is retired every frame: residency
inverted into a faster version of the exact starvation it was built to fix. And
because it happens through the RETIREMENT path rather than the reap path,
`[ReapJoin] starved` — the plan's own headline acceptance gate — would read
**improved** while the scene drained.

**A verification step is not sufficient, and that is the important part.**
"We confirmed the list is pre-cull" is a fact about one capture, on one map, at
one position. RESIDENT_SCENE_PLAN §0.0 already names a sixth cull with no flag at
all — `sub_1802EAD60`'s position-keyed area order list, 30 of 179 areas — which
no configuration defeats and whose behaviour is not uniform across maps. A
discipline that has to be re-established per map is a discipline that will be
skipped once.

**So the distinction is carried by the type, and only one of them is wired to
invalidation:**

```
ExistenceSource     may assert ALIVE     may assert DEAD        requires a PROVEN pre-cull list
VisibilitySource    may assert VISIBLE   MAY NEVER assert DEAD  post-cull lists get this, only this
```

An enumeration hook declares which it is at its construction site. Invalidation
takes an `ExistenceSource` and nothing else, so a post-cull list physically
cannot reach the retirement path — the §2.1 lesson applied again: *find the place
where the question answers itself* rather than writing a rule somebody has to
remember.

**The promotion gate is the one slice B already has.** `listed=` FLAT across a
fixed-position pitch-and-yaw sweep is what promotes a source from
`VisibilitySource` to `ExistenceSource`. Until it passes on the map under test,
the source stays a `VisibilitySource` and residency keeps whatever liveness it
had.

**Do not discard a post-cull source — type it and use it.** A `VisibilitySource`
is the engine's own visibility answer, which is strictly better information than
`sceneCull`'s frustum-and-light-influence test (§1.4: `culled=0`, because
`lightAllKeep` covers all 1298 off-screen instances). Feed it to slice 9 as a
culling input. It is worth having; it is just not allowed to kill anything.

---

## 4. THE CPU HALF — FROM PHASES TO A DEPENDENCY GRAPH

### 4.1 What is actually there

`WorkerThreadPool` (`util/util_threadpool.h:282`) is a work-stealing pool of
SPSC queues. `Schedule` returns a `Future<R>`. That is the entire vocabulary:
there is no way to express "this job becomes runnable when those two finish".

So the frame is a sequence of phases separated by global barriers:

```
capture (frame thread, serial, per draw)
   │
   ▼  flushGeometryBatch
chunked parallel-for  ────────────────► SynchronizeCsThread(SynchronizeAll)
   │                                     full CS drain, d3d11_rtx.cpp:34813
   ▼  futs[fi].get() in order, busy spin
ordered emit
   │
   ▼
dxvk-cs: commitGeometryToRT per draw
   │
   ▼
injectRTX: GC, accel build, TLAS
```

The chunking is right and was hard-won — one task per shard produced scheduler
and condvar overhead, packed bundles fixed it. **Keep the packing.** What to
change is that the packing is currently the *only* structure: because there is
no way to say "wait for that one thing", everything waits for everything.

### 4.2 The change

Add dependency counters to the pool. The shape, which is the CD Projekt RED
REDengine 4 model and is a good fit here:

```cpp
struct JobCounter {
  std::atomic<uint32_t> pending;
  // waiters guarded by a spinlock; released when pending hits 0
};

Job::dependsOn(JobCounter&);           // increments
Job::onComplete();                     // decrement; at 0, enqueue waiters
```

Three properties to get right, because each one has bitten someone:

**Composable — a job must be able to spawn children.** Geometry gather does not
know it will produce 384 instances until it has run. So it spawns a parallel-for
of chunks and the downstream job depends on the *counter*, not on the gather.
Without this the fanout path either guesses a chunk count up front or serialises.

**An empty parallel-for still completes its counter.** Zero instances, zero
bones, no fanout, a replay miss — all of these must leave the graph shape
unchanged, or every "no work" case becomes a special path through the renderer.
This tree has ~20 such cases and they are why `SubmitDrawDeferred` is 9,922
lines.

**The carrier chains become edges, not phases.** `stageDepCarrierGroups`
(`d3d11_rtx.cpp:38372`) already enumerates the cross-draw dependency groups —
camera, camera-smoothing, static, bone, cbLoc, route. Today a draw that touches
one of those is forced onto the frame thread. With counters it gets an edge to
the previous draw in *its* group, and every draw in no group runs concurrently:

```
Cam100 ──► Draw100 ──► Cam101 ──► Draw101        one chain, serial, correct

Draw200 ─┐
Draw201 ─┼── no group membership, all concurrent, no barrier between them
Draw202 ─┘
```

That is the change that makes the worker count matter. Today adding cores adds
nothing past the widest phase.

### 4.2.1 THE LIFETIME CONTRACT, AND WHY THE POOL CANNOT SUPPLY IT

**The counters are right and `Future` cannot carry them.** The existing pool is a
bounded task ring with recycled ids, and a `Future` is a raw pointer into it:

```
util_threadpool.h:362   TaskId taskId = m_taskId++ & (m_taskCount - 1);   // wrapping ring index
util_threadpool.h:259   mutable Task* task = nullptr;                     // Future IS a raw slot pointer
util_threadpool.h:158   result.reset();                                   // capture() clears hasResult
util_threadpool.h:229   valid() { return task != nullptr && task->valid(); }
```

`m_taskCount` is `NumTasksPerThread * numThreads` rounded up to a power of two.
`capture()` placement-news the lambda into the slot's inline storage and calls
`result.reset()`, which clears `hasResult` — under any holder still sitting in
`get()`. `valid()` cannot detect that the slot was re-issued, because there is no
generation on the `Task`.

**So a retained `Future` is I2 being violated inside the primitive the graph
would be built on.** It is safe today only by accident: `flushGeometryBatch`
schedules ~worker-count chunks and joins them immediately, so `m_taskId` never
advances a full `m_taskCount` between capture and get. A graph that holds
completion tokens across phases — which is the entire point of §4.2 — removes
that accident.

**Two more, and both are silent deadlocks rather than corruption.**

*Cancellation never completes.* `cancel()` sets `isDisposed = true`
(`util_threadpool.h:107`) and the capture thunk is
`if (!result.disposed()) { ... result.set(...) }` (`:145`). A cancelled task
never calls `set()`. A dependency counter decremented inside `set()` never fires,
and every job waiting on it is unreachable. The pool's own destructor uses this
path — `m_tasks[taskId].cancel(); m_tasks[taskId]();` at `:335` — so shutdown
with a live graph hangs.

*A full queue never completes either.* `Schedule` returns a default-constructed
`Future` when `isFull()`: `valid()` is false and there is no task at all.
`flushGeometryBatch` survives only because the CALL SITE checks `f.valid()` and
runs the range inline (`d3d11_rtx.cpp:34787`). A graph that assumes `Schedule`
yields a token that will complete deadlocks the first time a worker queue fills.

**The contract, therefore, and it is graph-owned rather than pool-owned:**

```cpp
struct JobHandle { uint32_t index; uint32_t generation; };   // into a graph-owned node array

// Every node completes its dependents EXACTLY ONCE, on every exit path:
//   ran to completion · cancelled · threw · ran inline because the queue was full
struct NodeScope {
  ~NodeScope() { node.signalComplete(); }   // decrement + release waiters, idempotent
};
```

`Task` / `Future` are demoted to "a way to get a worker to run this node's body".
Nothing outside the pool holds a `Future` across a phase boundary. Enqueue-or-run-
inline moves INTO the graph rather than being re-implemented at each call site.

**Gate:** run the graph with a deliberately undersized queue (`NumTasksPerThread`
forced low) and with cancellation injected at random nodes. Every dependent must
still become runnable, and the node array's generation must catch any handle that
outlived its node. If either needs a call-site check to hold, the contract is not
in the graph yet.

### 4.3 The explicit ordering key, and why to add it now

RE Engine's published approach (Capcom, GDC 2019) is worth taking almost
literally: generate intermediate draw commands on many threads, then sort them by
a 64-bit priority before translating to API calls. This tree has the sort key
already — `DrawSnapshot::drawIndex` *is* the submission-order key
(`d3d11_rtx.h:211-216`, and its comment names UE's `FMeshDrawCommandSortKey`).

Make it explicit and widen it:

```
[ frameId : 20 ][ pass : 4 ][ dependency class : 8 ][ submission order : 32 ]
```

Then workers may finish in any order and only the final ordered emit needs the
original sequence. Cheap, and it removes the last reason for the in-order
`futs[fi].get()` spin.

### 4.4 What NOT to do to the CPU half

**Do not put `ExtractTransforms` on a worker as one job.** It is one unit of
parallelism, and the decomposition already exists in the perf map: `xt_setup`,
projection scan, view derivation, world derivation, bones, classification, tail.
Those have different dependency properties. A single job hides that.

**Do not build a generic task-graph framework.** Counters, a parallel-for that
can spawn, and an empty-completion job. Three primitives.
`rules/subtractive-engineering.md` applies to the scheduler as much as to the
renderer.

---

## 5. THE GPU HALF

### 5.1 The changed set must reach the TOP of the accel chain, not the upload

The obvious statement of this item is "delta-upload the surface buffer". That is
true and it is not enough, because the upload is the LAST of four O(scene) steps
and the three above it would rebuild what the delta was meant to preserve.

```
mergeInstancesIntoBlas   walks the full instance table        rtx_accel_manager.cpp:899, :1129
                         m_reorderedSurfaces.clear()          :964   <- rebuilt from scratch
                         partition + stable_sort              :1082-1140
prepareSceneData         consumes m_mergedInstances           :3310
uploadSurfaceData        fills + uploads EVERY surface        :7799, :7946
buildTlas                MODE_BUILD_KHR                       :9380
```

**A changed-object set that is introduced at `prepareSceneData` is introduced
one step too late.** The surface array it would delta against has already been
cleared and rebuilt by `mergeInstancesIntoBlas`, so "which slot did this object
occupy last frame" has no answer by then. The delta has to start at the walk.

**And the tree has already hit this and worked around it.**
`[SurfaceIndexStability]` at `rtx_accel_manager.cpp:1082` exists to make the
ORDER of a per-frame-rebuilt array reproducible frame to frame — a stable-ORDER
substitute for stable SLOTS, implemented as a partition plus a `stable_sort` over
the tagged block, with its own comment noting that the naive version sorted "all
15.5k pointers every frame to order 63 of them".

That is the RESIDENT_SCENE_PLAN §0.2 buffer-tape bug one level up: a slot index
that means "position i in this frame's array". It is the **third** instance of
the same shape in this tree — the input-byte object key, the `m_bufferCache`
tape, and now the surface array — and it was papered over with a sort rather than
fixed, because at the time nothing needed a surface slot to outlive its frame.
Residency does.

**So the change is not an uploader change.** `m_reorderedSurfaces` becomes a
`BufferSlotTable`-shaped persistent array: one stable slot per `RenderObject`,
allocated on first appearance, freed on evidence rather than on frame age, and
carrying a generation (I2). Once it is:

```
per object:  frameLastChanged  vs  frameLastUploaded
             equal      -> its surface slot is already correct on the GPU
             different  -> stage it into a coalesced dirty-range upload
```

and the clear-and-rebuild at `:964`, the partition-and-sort at `:1082-1140` and
the whole-array upload at `:7946` all fall out together. The sort exists only to
make an unstable thing look stable; with real slots there is nothing to stabilise.

**One stale comment to ignore while reading that function.**
`prepareSceneData`'s header comment says `rtx.logTlasSet` and
`rtx.debugInstanceUploadProbe` "default to TRUE" and do work proportional to the
instance count on every frame. They no longer default true — `rtx_options.h:4106`
and `:3984` are both `false`, and the live conf pins them False with the ~4.8 ms
measurement recorded beside them. That one is closed; the comment predates the
fix. The structural O(scene) work above it is what remains.

**Gate:** a stationary scene must upload ~0 surface bytes per frame AND must not
clear `m_reorderedSurfaces`. If it cannot do the second, the first is unreachable.

### 5.2 TLAS refit

`buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR`
(`rtx_accel_manager.cpp:9380`) — full rebuild, every frame, per TLAS type. The
BLAS path already knows how to update (`:2635`, `:3057`), so the machinery and
the flags discipline exist.

A full rebuild is correct and sometimes the right choice; a refit is legal only
if the instance set has not changed topologically. With a persistent scene it
usually has not. Worth a measurement before a change: refit quality degrades and
a degraded TLAS costs traversal, so this is a trade, not a win. **Lower priority
than 5.1 and 5.4.** Listed so it is not rediscovered as free.

### 5.3 Generalise the PointInstancer culling to the resident set

The machinery is written and shipped. `rtx_point_instancer_system.h` pushes
`mask=0` placeholders, dispatches a compute shader that tests each transform
against the camera, and writes full instance entries for survivors.

`sceneCull` (`rtx_accel_manager.cpp:1226-1400`) does the same job on the CPU,
serially, over the instance table, and RESIDENT_SCENE_PLAN §1.4 already
established that it is inert today because `lightAllKeep` fires when `lights=0`
and keeps all 1298 off-screen instances. Once residency restores the off-screen
population, `sceneCull` becomes the whole GPU-side lever — and at that point it
is a serial CPU loop over the entire level, deciding something a compute shader
already decides for point instancers.

**Move it.** The two-stage model that results is the clean one:

```
ENGINE / CPU     which objects exist?          §7 enumeration
SCENE            persistent, level-sized
GPU              which of them matter to this view?    mask = 0
TLAS             resident ∩ visible
```

Do this AFTER 5.1. Delta upload is what makes the GPU's copy of the scene
authoritative enough to cull against.

### 5.4 SER: un-stub the G-buffer predicate — the highest-value GPU item

Both G-buffer payload states return `false` unconditionally
(`geometry_resolver_state.slangh:182` and `:366`). The option that appears to
control this is `True` in the live conf and inert. So the pass with the
divergence problem — one uber-shader carrying opaque, translucent, WBOIT, NRC,
PSR and portal paths — never reorders.

The work is not "return true". SER pays when there is divergent work *after* the
reorder point and a hint that predicts it. So:

1. **Un-stub it and measure with zero hints first.** Even `numCoherenceHints=0`
   reorders by hit/miss and by hit-group, which is most of the primary-ray
   divergence. This is the baseline and it costs one line.
2. **Then add hints, in the order the payload can cheaply compute them:** surface
   material class (opaque / alpha-tested / translucent), then PSR-vs-primary,
   then portal-crossed. Two to four bits.
3. **Then, and only then, consider the material-class split.** Classifying pixels
   into per-material dispatches is the same idea implemented in software, and it
   is a much larger change. Reordering may make it unnecessary — measure before
   building it.

There is a hard precondition and it is why this is not already done: SER lives
under `#if defined(RAY_PIPELINE)`, so it does nothing in the RayQuery/compute
G-buffer mode. `rtx.renderPassGBufferRaytraceMode` defaults to `RayQuery`
(`rtx_options.h:1568`) and the live conf sets `2`. **Confirm which enumerator `2`
is before spending any time here** — if the G-buffer is running as compute, step
1 is inert and the honest first move is the A/B between the compute and
ray-pipeline G-buffer, not the predicate.

### 5.5 What NOT to do to the GPU half

**Do not move the CPU's instance matching to the GPU.** The suggestion to compute
`findSimilarInstance` in a compute shader is solving the symptom. With a
`RenderObject`, there is nothing to match — the observation carries the id. §1
deletes this problem; do not port it.

**Do not read anything back.** A GPU result consumed by the CPU next frame is a
sync point wearing a compute shader. Every item above produces a result the GPU
itself consumes.

---

## 6. DELETE FIRST

`rules/subtractive-engineering.md` says question, delete, simplify, accelerate,
automate, in that order. Applied here, the first two are large and neither is on
anyone's list.

### 6.1 The diagnostic estate

893 distinct probe tags. 910 `Logger::` calls in `d3d11_rtx.cpp`. 75 tags inside
`ExtractTransforms` alone.

They are the reason this tree solved as many bugs as it has and most of them
should not be deleted on sight. But a probe built for a bug the record marks as
RESOLVED is scaffolding left on a finished building, and it is the direct cause
of the 10,000-line functions that make §1 through §5 hard.

The procedure, which is the fence procedure from `rules/simple.md`:

1. For each tag, find the commit note that created it and the one that closed the
   bug.
2. **Closed and never reopened → delete the probe with its state.** Not gated off
   — deleted. A `kEnable`-gated probe still occupies the function and still has
   to be read past.
3. **Closed but the bug could recur → keep, and give it an assertion instead of a
   log line.** `[StaleTape]` is the model: silent when clean.
4. **Open → keep, unchanged.**
5. **Cannot determine why it exists → keep.** An unexplained part is a fence.

Expect to remove most of what you look at, per the rule. Do this **before** the
splits in slices 4-7, because splitting a function that is 40% dead
instrumentation carries the instrumentation into both halves.

### 6.2 Options

654 `RTX_OPTION` declarations (`rtx_options.h`). Each one is a branch, a
configuration the code claims to support, and a rebuild of nearly the whole tree
when the header is touched.

Two specific deletions already identified and still not done:

- `rtx.residentScene.foldCbGenerations` — deleted from `rtx_options.h`, still set
  in `rtx.conf`, inert (RESIDENT_SCENE_PLAN §7.6 item 5).
- `rtx.enableShaderExecutionReorderingInPathtracerGbuffer` — set `True`,
  hard-disabled in shader. Either §5.4 makes it real or it goes.

The general test, and it is the one this tree has already used to kill
`foldCbGenerations`: **a flag that exists so a deletion did not have to happen is
that deletion deferred forever.**

### 6.3 The fallback paths

`shardInstanceProcessing` is `False` in the live conf; `batchSubmitDrawStages` is
`True`. Both have a "set false for the proven path" escape, so the tree carries
two implementations of the instance pipeline and two of the batch pipeline. That
was right while they were being proven. Once §4's graph lands they are three
implementations, and the old two are dead weight in the exact functions slices
4-7 want to split.

Pick one per axis and delete the other. Not gated — deleted.

---

## 7. BUILD ORDER

Nothing here starts before RESIDENT_SCENE_PLAN §7 slice D. That is not a
sequencing preference; §1.2 is the reason.

| # | slice | files | gate |
|---|---|---|---|
| **0** | **Probe cull** (§6.1) — the fence procedure over all 893 tags | `d3d11_rtx.cpp`, `rtx_render/*.cpp` | line count down; every remaining tag traceable to an OPEN bug or an assertion |
| **0b** | Dead-flag and dead-path deletion (§6.2, §6.3) | `rtx_options.h` once, conf, `d3d11_rtx.cpp` | one instance pipeline, one batch pipeline |
| **1** | **`RenderObjectDB` mirroring `ResidentScene`** — no behaviour change, no perf target | `rtx_render_object.{h,cpp}` (new), `rtx_resident_scene.*` | same draw -> same `RenderObjectId` across frames and camera motion. This IS the invariant |
| **2** | **Object resolver** — one place decides identity: handle, else `KeyHead`+occurrence | `d3d11_rtx.cpp`, resolver | `newKeys ~0` stationary; FLAT across a pitch-and-yaw sweep |
| **3** | Slot generation on `BufferSlotTable` (§0.1 item 4) | `rtx_utils.h` | `[StaleTape] dead` becomes unreachable, not merely zero |
| **4** | **Observe/derive split** — `SubmitDraw` captures and enqueues, derives nothing | `d3d11_rtx.cpp` | no live D3D11 read past capture; `XfLiveSite` escape counts all 0 |
| **5** | **`ExtractTransforms` -> pure classifier** — `CaptureTransformInputs()` then `TransformClassifier(input) -> result`, deterministic, no hidden cross-draw state | `d3d11_rtx.cpp` | same inputs -> same output, run twice, bit-identical |
| **5b** | **`ExistenceSource` / `VisibilitySource` types** (§3.1) — invalidation takes the first only | `rtx_resident_scene.*`, enumeration hooks | a post-cull source cannot compile against the retirement path. Do this BEFORE RESIDENT_SCENE_PLAN slice D is wired |
| **6** | **Job graph: counters + graph-owned `JobHandle` + composable parallel-for + exactly-once completion** (§4.2, §4.2.1) | `util_threadpool.h`, new graph, `d3d11_rtx.cpp` | undersized-queue + injected-cancellation run completes every dependent; `SynchronizeCsThread(SynchronizeAll)` removed from the flush; carrier groups are edges |
| **7** | **Dirty-object jobs** — the graph is fed by the changed set, not the draw list | `d3d11_rtx.cpp`, resolver | one mesh in 15 draws -> one geometry decision |
| **8** | **Persistent `m_reorderedSurfaces` slots, then delta upload** (§5.1) | `rtx_accel_manager.cpp` | `m_reorderedSurfaces.clear()` gone; `[SurfaceIndexStability]` sort deleted; stationary scene uploads ~0 surface bytes/frame |
| **9** | **GPU scene cull generalised from PointInstancer** (§5.3) | `rtx_point_instancer_system.*`, `rtx_accel_manager.cpp` | `sceneCull` CPU loop gone; culled set matches the CPU verdict exactly under verify |
| **10** | **SER: un-stub, measure, then hint** (§5.4) | `geometry_resolver_state.slangh` | G-buffer ms, with the RayQuery-vs-ray-pipeline A/B taken first |
| **11** | TLAS refit where legal (§5.2) | `rtx_accel_manager.cpp` | traversal cost not worse; build cost lower |
| **12** | G-buffer material-class split | shaders | **only** if 10 leaves a divergence problem |

Slices 0 and 0b are first for a reason and it is not tidiness: slices 4, 5 and 7
are surgery on the five functions that §6.1 is removing 40% of.

`rtx_options.h` is touched **once**, at slice 0b, with every option this work
will ever need. It is included nearly everywhere and any touch is a ~20 minute
rebuild.

---

## 8. WHAT NOT TO BUILD

**Not an ECS.** One `RenderObject` struct and six databases that already exist.
Entities, components and archetypes are a generality tax on a project that is
reverse-engineering one 2016 game.

**Not a rewrite.** Every slice above is additive or subtractive against the
current tree. `DrawSnapshot` survives as the observation packet — which is what
its own comment already says it is. `ResidentScene` survives as the persistent
record. Phase 2B's chunk packing survives as the job system's batching policy.
The Titanfall-specific reversing in `CULLING_BIBLE.md` is the expensive part and
none of it is touched.

**Not id Tech 8's renderer.** Visibility buffers, deferred texturing and compute
material dispatch are a rasteriser's answer to a problem a path tracer solves
with SER and a well-built TLAS. Take the *principle* — the CPU says what changed,
the GPU decides what needs processing — and skip the pipeline.

**Not another round of per-draw micro-optimisation inside `SubmitDraw`.** This is
the one the measurements are loudest about: engine culling took 1338 draws to 538
and 57.99 ms to 19.97 ms by changing the *population*, not the per-item cost.
Reducing what enters the pipeline is worth more than any amount of shaving inside
it, and it is what §1 through §5 are for.

**Not a class-keyed exclusion anywhere.** I6. It will be tempting at slices 7, 9
and 12 and it is wrong at all three.

---

## 9. OPEN QUESTIONS

Each of these can invalidate a reading taken from a slice above.

**1. `noTail` — the fifth of the frame the gate never sees.** Four shaders report
`o2w{n=0}` across ~965 draws/frame; their draws are judged at the head of
`SubmitDraw` and never reach `SubmitDrawTail`. Whether they are RT-captured at all
is still unknown. **An object identity that never reaches the resolver is an
object the whole of §1 does not cover**, and no amount of enumeration fixes it —
RESIDENT_SCENE_PLAN §7.6 item 1 says exactly this. Read `noTail` before believing
any coverage number from slices 1, 2 or 7.

**2. The G-buffer raytrace mode.** `rtx.renderPassGBufferRaytraceMode = 2` in the
conf against a `RayQuery` default. If that resolves to a compute mode, §5.4 step 1
is inert and slice 10 changes shape entirely. One enum lookup; take it first.

**3. The PinDefer measurement.** `CbRange::srcPtr` exists to settle whether the CB
byte copy can become a pointer. Until it is run, "make capture cheaper by
referencing instead of copying" is a hypothesis with two known counterexamples.

**4. `shardInstanceProcessing` A/B.** RESIDENT_SCENE_PLAN §4.2 and §7.6 item 3 both
ask for it and it has not been taken. The live conf pins it `False`, which is not
the default either section was written against. Slice 0b deletes one of the two
paths and that decision needs the number.

**5. Whether the surface-table scaling problem is real yet.** §5.1 argues from the
code that a level-sized resident set makes the full upload expensive. That is a
prediction. Measure `m_reorderedSurfaces.size()` and the fill-loop time before and
after residency is fully armed, and if it does not move, slice 8 drops down the
list. It is still an I4 violation; it may not be an urgent one.

**6. The occurrence ordinal's residual.** It survives for world geometry and static
props until §7 slices F and H, which is ~80% of the draw stream, and engine culling
perturbs it by construction. `FAIL=0` has to be re-established across a full sweep
after every slice here, not once at the start.

**7. Which enumeration source, if any, is ever promotable.** §3.1 makes a post-cull
list harmless, and that is a safety property, not a solution — a `VisibilitySource`
supplies no positive liveness, so residency keeps eviction-by-output-test and
§7.8's population problem stays open. RESIDENT_SCENE_PLAN §7.5 lists three
candidate sites for a genuine pre-cull list and all three are unproven. **If none
of them is promotable, the architecture still works** — §7.8's four output tests
are the eviction rule and three of the four need no object — but slice E (poses
without draws) and the precise death signal do not arrive, and the plan should say
so rather than assume a pre-cull list exists. Settle this before slices 1 and 2
decide how much weight `engineHandle` carries in the resolver.

**8. Whether the task ring is already being overrun.** §4.2.1 argues it is safe
today by accident. That is an argument, not a measurement, and it is cheap to
settle: add a monotonic per-slot capture counter and assert that no `Future::get()`
observes a slot whose counter moved since `capture()`. If it fires today, slice 6
is a bug fix before it is an architecture change.

---

## 10. THE SHORT VERSION

```
The scene already persists.            RESIDENT_SCENE_PLAN did that.
The databases already exist.           DrawCallCache, SparseUniqueCache, BufferSlotTable.
The GPU culling already exists.        rtx_point_instancer_system.
The batching already exists.           Phase 2B's packed chunks.

What is missing is an OBJECT --
an identity that does not move when the object does --
and everything downstream of the resolver still taking the DRAW as its unit.

Add the object. Feed the frame from the changed set.
Then delete the four fallback paths and most of the 893 probes,
and the five 10,000-line functions become tractable for the first time.
```
