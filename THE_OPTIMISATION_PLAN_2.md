# THE OPTIMISATION PLAN 2 -- THE ARCHITECTURE, AND HOW TO BUILD IT

**Written:** 2026-08-15
**Branch:** `architecture-overhaul`, uncommitted
**Extends:** `THE_OPTIMISATION_PLAN.md` (v1). Closes v1 Phase 6e. Specifies Phase 2b for build.
**This document is an implementation plan, not an analysis.** Numbers appear only where they
constrain the build.

`[M]` measured this session . `[I]` inferred . `[U]` unknown

---

# PART I -- THE TARGET

## 0. THE ARCHITECTURE, IN ONE PAGE

v1 Sec 0.1 states it: **one capture, one parallel pass, one ordered handoff.**

```
GAME THREAD, per draw
  captureDrawSnapshot          capture inputs, derive nothing          BUILT
  collect into m_geoBatch      one DrawWorkItem per draw               BUILT

GAME THREAD, once at EndFrame -- flushGeometryBatch()
  [ORDERED pre-pass]   resolve BlasEntry per draw, build shards        BUILD THIS
  [PARALLEL]           material / hashes / bbox / skinning             BUILT
  [PARALLEL]           geom -- onSceneObject{Added,Updated}            BUILD THIS
  [PARALLEL by shard]  find / mid / add / update                       BUILD THIS
  [ORDERED tail]       migration, creation merge, new materials        BUILD THIS

DXVK-CS, per draw, in draw order
  the index-stash copyBuffer + VB/IB rebind onto gpuCapture            LEAVE ALONE
```

Everything on dxvk-cs except that last line moves to the parallel pass. What stays is positional
and cheap: one `copyBuffer` per draw, 64 vkDraws per frame.

## 0.1 WHY THE SEAM IS WHERE IT IS

From `commitGeometryToRT`'s own header (`rtx_context.cpp:4951-4959`):

> this lambda replays **IN-ORDER on the CS stream**, after this draw's bindings and before any
> later Map(DISCARD) rename replay, so the logical buffer still resolves to the physical slice
> the rasterized draw consumed

**Anything that reads buffer CONTENTS is positional.** Run it out of order and you read bytes the
engine has already renamed -- the failure hit twice in this project (the mid-upload bake, the
renamed dynamic IB). The instance/scene bookkeeping that follows reads no buffers and has no such
dependency. **That is the seam. Cut there.**

---

# PART II -- WHAT IS ALREADY BUILT (do not redo)

## 1. The parallel machinery, live today

| piece | where |
|---|---|
| `WorkerThreadPool` (work-stealing, 30 workers, `LowLatency=false`) | `util_threadpool.h` |
| `GeometryBatchArena m_geoBatch` -- per-frame arena of `DrawWorkItem` | `d3d11_rtx.h:903` |
| `flushGeometryBatch()` -- the parallel-for + join + ordered emit | `d3d11_rtx.cpp` |
| four stages already inside it | material, hashes, bbox, skinning |
| `DrawSnapshot` -- why deferral is legal at all | `d3d11_rtx.h:65-87` |
| `rtx.pushInstanceRecords` -- Phase 2 persistent instance records | `rtx_instance_manager.*` |

**The pool is not the constraint.** `[BatchJoinSplit]` 2026-08-15 `[M]`:

```
disp=120us   JOINIDLE=1us   wake{avg=170us}   ->  taxPerPass ~280us
```

Two more parallel passes cost ~0.6 ms. **v1 Phase 6e is closed -- build against this pool freely.**

## 2. Built this session, 2026-08-15 `[M]`

| # | change | file | result |
|---|---|---|---|
| 1 | `[BatchJoinSplit]` timer split | `d3d11_rtx.cpp` | closed Phase 6e |
| 2 | tail chunk scheduled, not kept for the game thread | `d3d11_rtx.cpp` | see Sec 8 -- amends v1 A.8 |
| 3 | `[BatchJoinCensus]` / `[BatchJoinPeak]` | `d3d11_rtx.cpp` | found the BAR stall |
| 4 | **`RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS`** | `d3d11_buffer.cpp` | **hash 17.6 ms -> 0.5 ms** |
| 5 | per-worker `m_lastSurfaceMaterial` slots | `rtx_scene_manager.{h,cpp}` | **Phase 2b prerequisite, DONE** |
| 6 | `kBjItemProbe` compile gate | `d3d11_rtx.cpp` | probe off |

Frame-end stage, before -> after: `sumWork` 29.0 -> 8.5 ms, `tail` 6.5 -> 0.69 ms,
`parallelForMsPerFrame` 6 -> 0.

---

# PART III -- BUILD IT

## 3. STEP 1 -- ORDERED PRE-PASS: resolve BlasEntry, build shards

**Where:** new stage at the top of `flushGeometryBatch()`, before the existing parallel-for.

**What:** for each `DrawWorkItem` in arena order, do only the cache lookup that
`processDrawCallState` does today at `rtx_scene_manager.cpp:3518`:

```cpp
m_drawCallCache.get(drawCallState, &pBlas)   // kExisted -> Updated, else -> Added
```

Store `pBlas` on the item. Then group item indices by `pBlas` into shards.

**Why ordered:** `m_drawCallCache.get()` **mutates** the cache (it inserts on miss). It is the one
step that cannot be parallel. It is also cheap -- a hash lookup per draw.

**Shape `[M]`:** ~450-465 unique BLASes over ~1350 draws => ~460 independent shards. Ample
parallelism; no shard is large enough to be a straggler on its own.

**Also do here:** `pBlas->frameLastTouched = frameId` and `pBlas->noteDraw(frameId)`
(`rtx_scene_manager.cpp:3530-3535`). Both are per-BlasEntry writes -- legal in the ordered pass,
a race in the parallel one.

**Verify:** shard count ~= `uniqueBlas` from `[Perf.Report]`; sum of shard sizes == `drawsCommit`.

## 4. STEP 2 -- PARALLEL: geom

**Where:** a stage in the existing parallel-for.

**What:** `onSceneObjectAdded` / `onSceneObjectUpdated` (`rtx_scene_manager.cpp:3519-3521`) --
geometry interleave, BLAS input, hashing. **5-6 ms/frame `[M]`.**

**Why it is safe:** each call targets one `BlasEntry`, and Step 1 has already assigned exactly one
owner per entry. Run it **inside the shard task**, not as a separate per-draw pass, so the
exclusive-owner guarantee covers it.

**Trap:** the geometry cache is shared. Anything that allocates from it needs the Sec 6 lock.

## 5. STEP 3 -- PARALLEL BY SHARD: find / mid / add / update

**This is the load-bearing step. Everything else depends on it.**

**Where:** the same shard task as Step 2, immediately after geom for that draw.

**What:** `InstanceManager::processSceneObject` (`rtx_scene_manager.cpp:3717`) and
`processSceneObjectFanout` (`:3709`), i.e. `processSceneObjectImpl` --
`find` (`rtx_instance_manager.cpp:4599`), `mid` (`:4964`), `add` (`:4968`), `update` (`:4978`).

**Measured shape `[M]` 2026-08-15:**

```
pct  find=32-34   mid=7-8   add=4   update=53-55       addedPct=0
ms   find=2.9-4.0 mid=0.7-0.9 add=0.4-0.5 update=4.4-6.1
```

**Why sharding by BlasEntry is sound:** an instance belongs to exactly one BLAS. All of
`find`/`update` operate within `blas.getSpatialMap()` and that BLAS's instance list, so two shards
never touch the same state. `getDataAtTransform` is `const` (`util_spatial_map.h`) -- probes are
pure reads.

**`addedPct=0` is the safety result `[M]`.** In steady state **no instance is created**, so both
partition escapes (Sec 6) are cold. A lock on a path that never fires costs nothing.

## 6. THE ESCAPES -- put these behind one lock

Three things break the BLAS partition. All are on the **miss** path, all are cold at `addedPct=0`.
Give them **one** `std::mutex` on SceneManager/InstanceManager and take it only on these paths:

| escape | where | why it escapes |
|---|---|---|
| cross-BLAS migration | `rtx_instance_manager.cpp` engine-class sibling steal | takes an instance from **another** shard's BLAS |
| `m_instances.push_back` | `addInstance` | the global instance vector, not per-BLAS |
| new surface material | `createSurfaceMaterial` -> `m_surfaceMaterialCache` / `m_preCreationSurfaceMaterialMap` | global caches, and the index is handed out |

**Do not** take the lock on the hit path. If profiling shows contention, that means `addedPct` is
no longer 0 and the assumption above has changed -- re-measure before optimising the lock.

## 7. STEP 4 -- ORDERED TAIL

**Where:** after the join, before the existing Phase C emit loop.

**What must be ordered:**
- SpatialMap **moves/inserts** (reads are const and parallel-safe; writes are not)
- creation merge -- anything the Sec 6 lock deferred
- new material registration, so cache indices are assigned deterministically

**Then the existing Phase C emit is unchanged:** walk the arena in draw order, `EmitCs` the
`commitGeometryToRT` lambda carrying only the positional `copyBuffer` + rebind.

---

# PART IV -- TRAPS THAT WILL BITE

## 8. `m_lastSurfaceMaterial` -- DONE, and why it is not a `thread_local`

**Already built this session.** One slot per worker, claimed on first use
(`rtx_scene_manager.h`, `surfaceMaterialMemoSlot()` / `invalidateSurfaceMaterialMemo()`).

It stores an **index into `m_surfaceMaterialCache`**, and both invalidation sites exist because a
cache clear renumbers every material. A `thread_local` would leave each worker's entry alive across
a reset, still naming an index that now points at a **different material** -- silent wrong-material
corruption, not a stale miss. Hence: slot array + invalidate-all.

It also closes a torn-read failure: with one shared slot a reader could match some fields against
one writer's entry and the rest against another's **and pass**, serving the wrong material.

## 9. `setFrameLastUpdated` clears `m_seenCameraTypes`

From v1 Sec 2.1: the first stamp of a frame clears the camera set, so **any skip path must
re-register the draw's camera**. `garbageCollection` reaps on `m_frameLastUpdated` and nothing
else, so "skip" means *stamp and re-register*, never *omit*.

## 10. Do not reorder the emit

`commitGeometryToRT`'s `copyBuffer` depends on its position in the CS stream
(`rtx_context.cpp:4948-4959`). **Correctness, not performance.** The arena is naturally in draw
order -- keep it that way through the tail.

## 11. AMENDMENT TO v1 APPENDIX A.8 `[M]`

v1 A.8 says:

> **Do not remove the tail-chunk-on-game-thread or the inline fallback.** They are what stop a
> saturated pool from dropping work or idling the caller.

**The inline-fallback half stands and is preserved.** That is the half that stops work being
dropped when a worker queue is full.

**The tail-chunk half is refuted and has been removed.** Measured: the game thread's tail chunk ran
19-24 items at ~320 us/item against the workers' ~24 us/item, so it finished ~6 ms after every
worker was already idle -- it *was* the straggler. The join it existed to fill costs ~1 us
(`JOINIDLE`). Scheduling it instead moved `self` 6000us -> `tail`, and after Sec 12 the whole stage
is 0.69 ms.

## 12. THE MEMORY-TYPE CLASS -- the reason the stage was ever 6 ms

**Fixed this session; recorded because the same shape will recur.**

`[BatchJoinCensus]` showed 72 of 1070 draws carrying 74% of the stage, and the peak draw hashing
**23 KB in 2.68 ms = 8.7 MB/s** against XXH3's 10-30 GB/s. The byte counts proved the range clamps
correct; `memFlags()` read `0x7` = `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`, `HOST_CACHED` clear
-- **BAR memory, every read a PCIe transaction.**

Fix: `RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS` in `d3d11_buffer.cpp` -- the same three lines
already applied to DYNAMIC index buffers (2026-08-06) and DYNAMIC SRV buffers (2026-08-07).

**The rule (Sec 16 R8): before optimising a loop, check what memory it reads.** No amount of
caching the result, balancing the schedule, or deleting redundant calls touches a stall.

**Still open `[U]`:** the GPU half of that trade. Vertex buffers are fetched per-draw by the
rasteriser and copied for BLAS builds, so system RAM puts real traffic on PCIe. Bounded by only 72
draws being affected and the GPU sitting at 13.2 ms with 43-46 ms idle. **To close:** flip the
define to `false`, capture, compare `[Perf.GpuPass] totalMs` distributions across a run. **If it
loses:** do not revert -- capture a cached shadow at `Map()` time instead (the game writes those
bytes through the CPU anyway; WC *writes* are fast). Keeps the buffer in VRAM, no GPU exposure.

## 13. TOMBSTONE -- chunk oversubscription. DO NOT RETRY. `[M]`

`kChunksPerWorker` 1 -> 4 (30 -> 119 chunks):

| | 30 | 119 |
|---|---|---|
| `tail` | 6500 us | **7300 us** |
| `wake` avg | 155 us | **800 us** |
| `sum(work)` | ~30 ms | **~47 ms** |
| frames / 3 s | 54 | **17** |

119 `Schedule()` calls are 119 mutex round trips and 119 `notify_one`s into 30 condvar-sleeping
workers. Aggregate worker time rose 58% and that CPU competes with the game and CS threads, so the
**frame rate** fell. Same failure the `GeometryProcessor` header records from 2026-07-28.
**Chunk count is not a lever on this pool.** Reverted to 1.

Related, unexploited `[M]`: `Schedule()` defaults `Affinity=0xFF` and
`affinityMask = min(popcnt(0xFF)=8, 30)`, so **only queues 0-7 are ever pushed to**; the other 22
workers acquire work solely by stealing. It works (`wait` ~0) -- a note, not a defect.

---

# PART V -- ORDER OF WORK AND VERIFICATION

## 14. BUILD ORDER

| # | step | size `[M]` | risk |
|---|---|---|---|
| 0 | ~~`m_lastSurfaceMaterial` per-worker~~ | -- | **DONE** |
| 1 | ordered pre-pass: BlasEntry resolve + shard build | -- | low, no behaviour change |
| 2 | **sharded claim** -- shard tasks own their BLAS | -- | **highest. everything depends on it** |
| 3 | move `update` into the shard task | 4.4-6.1 ms | medium |
| 4 | move `find`/`mid`/`add` into the shard task | 3.3-4.9 ms | medium |
| 5 | move `geom` into the shard task | 5-6 ms | low |
| 6 | ordered tail + escapes behind the lock | ~1.2 ms stays ordered | low |

**Do 1 and 2 first even though they move no time.** They are what makes 3-5 legal. Landing 3
before 2 is a data race on the instance manager.

## 15. VERIFY AT EACH STEP

Correctness first -- these are the ones that fail silently:

| check | where | must read |
|---|---|---|
| instance count stable | `[Perf.Report]` `inst=` | ~15.4k, unchanged |
| no instances reaped by accident | `[ReapJoin]` `removed=` / `starved=` | not rising |
| identity still resolving | `[FindStage]` `exact=` | ~97% (un-deny the tag first) |
| no new materials appearing | `[Perf.Report]` `matNew=` | **0** |
| nothing lost between stages | `[MeshTrace]` `BatchEmitted` vs `Submitted` | equal |
| draws still committing | `drawsIn` vs `drawsCommit` | ratio unchanged |

Then timing: `[ProcDCS] instMs` / `geomMs` should fall; `[BatchJoinSplit] tail` will rise as work
moves in -- **that is expected**, it is the parallel-for absorbing the CS thread's work.

**Do not measure with `rtx.perfSceneObjSplit=True`** -- it taxes both sides. Turn it on only to
attribute find/mid/add/update, then off.

**Check `clkNs` before quoting any absolute ms** (v1 C.3b). Sane is ~32-49 with `xMin=1.00`. This
session's windows read `clkNs=32-46 xMin=1.00-1.31 cpuSlowX=1` -- sane.

## 16. RULES ADDED THIS SESSION

**R8. Before optimising a loop, check what memory it reads.** Throughput orders of magnitude below
cached RAM is a memory-type problem. `DEVICE_LOCAL|HOST_VISIBLE` without `HOST_CACHED` is BAR and
every CPU read is a PCIe transaction. Check `memFlags()`.

**R9. One aggregate timer is not a diagnosis.** `parallelForMsPerFrame=6` contained dispatch, wake,
real work on the calling thread, and stall, in unknown proportion -- and it blocked Phase 2b as a
prerequisite for a week. Split a timer by *mechanism* before acting on it.

**R10. A probe that can only look in one place will keep confirming that place.** The tail-only
per-item probe kept reporting `@idx1045-1090` because that was the only range it timed. Widened,
the expensive draws were everywhere. Size the *population* before believing a *peak*.

**R11. Aggregate worker time is not wall time.** Divide `sum(work)` by the worker count and check
against the critical path before targeting it. `mat` is 92% of the frame-end stage and ~0.3-0.7 ms
of its critical path.

**R12. In-tree comments decay silently.** Three load-bearing claims were refuted this session, all
true when written: "vertex buffers are genuinely write-only", "the last range runs on the game
thread so it is not idle during the join", and the handoff's `vbPtr`-rotates note. **When a comment
justifies not measuring something, measure it.**

## 17. INSTRUMENTATION STATE

| switch | state | notes |
|---|---|---|
| `kBjItemProbe` (`d3d11_rtx.cpp`) | **false** | per-item 4-way probe, ~220 us/frame. `true` for `[BatchJoinPeak]`/`[BatchJoinCensus]` |
| `[BatchJoinSplit]` | **on** | per-chunk only, ~60 clock reads/frame. Leave on |
| `RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS` | **true** | `false` for the Sec 12 A/B |
| `rtx.perfSceneObjSplit` | **False** | on only to attribute find/mid/add/update |
| `rtx.logDenyTags` | `[FindStage]` added | **remove it before the Sec 15 identity check** |
| `rtx.pushInstanceRecords` / `...Verify` | True / False | Phase 2, shipped |

Carried from the SRV case: the `HOST_CACHED` substitution needs the `HostVisibleSmallPool` fix in
`dxvk_memory.cpp` (present) and produces a once-per-frame VRAM sawtooth against the geometry-cache
trim. **Not a leak** -- see the note in `d3d11_buffer.cpp`.

---

## 18. WHAT THIS BUYS

`commitGeometryToRT` is 23-32 ms/frame, of which `submitDrawState` is 20-29 and
`processDrawCallState` 18-26 (`geom` 5-6, `inst` 11-17, other 1-2) `[M]`.

Steps 2-5 move `geom` + `inst` -- **16-23 ms/frame** -- off dxvk-cs into a pool whose measured tax
is ~280 us/pass. What remains ordered is the `copyBuffer` + rebind, plus the ~1.2 ms tail.

**v1 Appendix B.6's conclusion is the point:** once ordered GPU recording is the only thing left on
dxvk-cs, that thread never becomes the bottleneck again at any frame rate, and the question of
widening it never has to be asked. That is why this is the architecture and not an optimisation.

---

# ANNEX — BUILT 2026-08-15 (night)

Steps 0-6 are implemented, gated on `rtx.shardInstanceProcessing` (default True; set False in
rtx.conf for the byte-identical pre-2b path, no rebuild). Design record + as-built deltas:
`PHASE2B_IMPLEMENTATION_SPEC.md`. The load-bearing addition beyond this plan: a full CS-thread
drain at flush start (overlapped with Phase B) that gives strict mutate/read alternation between
the game thread and dxvk-cs — it is what makes Steps 2-5 legal with one cold lock and no hit-path
synchronization. Sky/terrain/replacement/unready draws take the unchanged legacy CS path in draw
order (`[Shard2b] legacy=` counts them).

**Where the build DIVERGES from Part III — read this before reading the timings.** `[I]` unless marked.

1. **Step 5 (`geom`, 5-6 ms) did not move, and cannot.** Only the cache-state DECISION is on the
   flush side; `processGeometryInfo`'s bake — interleave, dispatches, `updateBufferCache` — still
   records on dxvk-cs at the draw's stream position, because it reads mapped buffer contents and
   emits GPU commands. **Sec 0.1 is the reason: that is the positional half of the seam.** So
   Sec 18's "Steps 2-5 move geom + inst -- 16-23 ms/frame" overstates this phase. What moves is
   `inst` (11-17 ms) minus the CS record residue (per-instance buffer binds, billboards, OMM,
   ray portals). `[ProcDCS] geomMs` staying flat is CORRECT, not a failure.
2. **The drain is a new serialization point this plan never costed.** Pre-2b the game thread and
   dxvk-cs overlapped; the drain makes them strictly alternate, so whatever remains on CS —
   including the previous frame's `injectRTX` and GC — is now exposed to the game thread as
   `[BatchSubmitDraw] csDrainUs`. The net win is (work removed from CS) minus (csDrainUs + the
   ordered pre-pass). Read csDrainUs before any other number.
3. **The pre-pass does more than Sec 3 specifies.** Beyond `drawCallCache.get` it runs
   `finalizePendingFutures`, `processCameraData`, the fog block, up to three replacement probes
   and `determineMaterialData` — 1350x, single-threaded, on the game thread. Correct placement
   (they mutate shared caches) but a new serial cost, reported as `[Shard2b] preUs`.
4. **Sec 6's escapes 1 and 2 are implemented as DEFERRAL, not locking.** Migration and
   `addInstance` never run on a worker at all; a miss defers the draw to the ordered tail, which
   re-runs `find` a second time. Safer than a lock, and still cold at addedPct=0 — but a rise in
   `[Shard2b] deferred=` now costs double finds, not just contention.
5. **Sec 7's "new material registration in the tail" is not built.** `createSurfaceMaterial`
   misses run inside the shard under the escape lock, so new-material cache indices follow shard
   completion order, not draw order. Believed benign (lookup is by hash); see the correction in
   the spec's Sec 3.2.
6. **Sec 13's tombstone applies to the shard dispatch and was violated by the first build.**
   ~460 shards scheduled as ~460 tasks is 4x the count Sec 13 measured as catastrophic. Fixed
   2026-08-18: shards are the ownership unit, BUNDLES (one per worker) are the scheduling unit.
   `[Shard2b] bundles=` must track the worker count. **Sec 13 is not just about `kChunksPerWorker`
   — it is about the number of `Schedule()` calls per frame, whatever produces them.**

**Verification state — be honest with yourself before trusting a run:** built without compiling
(user compiles); mapped by a 6-agent deep-read before design; self-reviewed + brace-delta-checked
against HEAD after. The adversarial review fleet died on a session usage cap with ZERO findings
REPORTED — that is "not reviewed", not "clean". First build: check `[Shard2b]` appears,
`deferred~0`, then the full Sec 15 table. First anomaly: `rtx.shardInstanceProcessing = False`
and diff the two runs' logs.
