# THE OPTIMISATION PLAN

**Written:** 2026-08-11
**Branch:** `parallel-clean` @ `4681ecca` (the XfOverlap work is on `stable` @ `470eaf4b`, pushed, message "broken")
**Goal:** 30 fps = 33.3 ms/frame
**Source of every number below:** `[Perf.Report]` block at 02:47:54 in `remix-dxvk.log`, unless stated otherwise.

Confidence markers used throughout:
`[M]` measured this session · `[I]` inferred from measured data, not itself measured · `[U]` unknown, needs a capture

---

## 0. THE ONE-LINE ANSWER

The scene is 97% static and we rebuild it every frame. **Stop doing redundant work — do not
parallelise it.** Parallelism divides cost by 30. Redundancy elimination multiplies it by 0.03,
and unlike parallelism it works on `dxvk-cs`, which cannot be fanned out at all.

## 0.1 THE TARGET ARCHITECTURE, IN FOUR LINES

**One capture, one parallel pass, one ordered handoff.**

1. **`DrawSnapshot` at draw entry** — capture inputs, derive nothing. Already built.
2. **One frame-end parallel-for** (`m_geoBatch` / `flushGeometryBatch` / `m_pGeometryWorkers`)
   running *every* per-draw stage: material, hashes, bbox, skinning (already there), plus
   transforms, plus instance find/mid/add — and gated so the static 97% never enter it.
3. **Ordered emit to dxvk-cs** in `drawIndex` order carrying **only the positional GPU work** —
   the index-stash copy and capture rebinding that must sit at a specific point in the stream.
4. **Nothing else.**

Delete: the replay tier, the XfOverlap worker, and the defer gate once purity hits 100% (it is a
progress meter, not a mechanism).

One arena, one pass. **Four of the six per-draw stages are already in it today** (A.5). Getting
the other two in is Phase B (transforms) and Phase 2b (instance/scene work).

---

## 1. BASELINE `[M]`

```
FRAME 77.44 ms (12.9 fps)   inst=15431  uniqueBlas=457  drawsIn=1338  drawsCommit=1042
                            matNew=0    texNew=0        created=0

frame thread (PRESENT)   74.76 ms   busy 96.5%   blocked 2.68 ms    <- POLE
dxvk-cs                  45.45 ms   busy 66.4%   idle   23.04 ms
GPU                      15.71 ms                idle   56.86 ms (73%)

SLACK = 74.76 - 45.45 = 29.31 ms
```

`csSyncMs=0 gpuSyncMs=0` — nobody blocks on anybody. This is pure CPU work on two threads.
The GPU is idle because it is **starved**, not because it is fast. It has ~61.7 ms of headroom,
so nothing in this plan is GPU-limited.

### Budget to 30 fps

| thread | now | target | must cut |
|---|---|---|---|
| frame thread | 74.76 | 33.3 | **41.5 ms** |
| dxvk-cs | 45.45 | 33.3 | **12.2 ms** |
| GPU | 15.71 | — | none |

**Both threads must be fixed.** `frame = max(frameThread, dxvk-cs)`, so cutting only one
stops paying the moment it crosses the other.

### Frame thread composition `[M]`

```
D3D11 entry points               58.42 ms  75%
  -> Remix OnDraw* hook          53.01 ms  68%   <- OUR CODE
between entry points             19.21 ms  25%   <- Source engine, untouchable
```

Our injection is **71% of the pole**. Floor with injection at zero is ~22 ms (~45 fps).
30 fps therefore requires cutting 53.01 -> ~11.5 ms, i.e. keeping 22% of current per-draw cost.

Named leaves inside SubmitDraw:

| stage | ms | note |
|---|---|---|
| `pfs_guard` | **14.92** | **largest item on the pole, completely unattributed** `[U]` |
| `bt_extractXf` | 11.07 | the transform derivation |
| `tail_emit` | 4.93 | incl. `te_census` 1.01, `te_camDiag` 0.83, `te_propId` 0.37 (diagnostics) |
| `tail_capture` | 1.34 | |
| `cbc_rawUv` | 1.33 | |
| `vsAnalysis` | 0.93 | |
| `bt_cullVtx` | 0.83 | |
| `skyClassify` | 0.63 | |
| `filters` | 0.56 | |
| `o2w_t31` | 0.48 | |
| `w2vw_cb3` | 0.25 | |
| `bonePalette` | 0.11 | |

Fanout, which lives **outside** SubmitDraw: `SubmitInstancedDraw` 24.67 ms
(`inner SubmitDraw` 17.51 nested; `build` 7.15 → `setupPost` 3.76 incl. **UNATTRIBUTED 2.04** `[U]`
and `dbgTrack` 0.39 diagnostic; `loop` 3.40 → `inst_loop` 2.84, `t31_copy` 0.60, `t31_gather` 0.15).

Replay tier tax is now **0.09 ms** — it is off (`rp_key=0`), confirmed.

### dxvk-cs composition `[M]`

```
commitGeometryToRT              26.39
  submitDrawState               23.83
    processDrawCallState        21.46
      processSceneObject        12.78
        ui_fastRet   3.30   ui_entry  2.20   ui_tail 0.65   ui_xform 0.57
        ui_flags     0.28   ui_surf   0.24   ui_rest 0.35
        find / mid / addInstance    ~5.2 residual, NOT MEASURED  [U]
      geom                       5.98
  finalize (worker join)         1.16
the once-per-frame fat chunk    15.36
  prepareSceneData              14.75
    merge                       11.25   (loop 5.10, buildBlases 3.84, dynBlas 1.82, setup 0.37)
    surfMat 1.59  gc 1.24  setup1+instSetup 0.53  tlas+accel+light 0.13
```

---

## 2. THE CORE FINDING: `REDUNDANT=97%` `[M]`

From `[Perf.UpdInst]`:

```
reachPct:  entry=100  fastRet=93  surf=6  xform=6  flags=6  tail=6
counters:  first=99%  xfChg=1%  matChg=0%  prevPos=0%  static=97%
           REDUNDANT=97%  surfSkip=99%  tailSkip=97%
estMsPerFrame: entry=2.73  fastRet=4.15  xform=0.70  tail=0.54  flags=0.37  surf=0.33
```

Corroborated by `[Perf.FastInst]`: `fast=14327  created=0  keyMiss=19  spatialMiss=9.6`
and `[FindStage]`: `calls=15537 exact=15140` (**97.4% resolve by exact key**).

**Read it like this:** 6.88 ms/frame (`entry` 2.73 + `fastRet` 4.15) is spent *asking* 15,612
instances whether anything changed. The stages that do real work reach 6% of instances and cost
~1.9 ms combined. We pay 3.6x more to ask the question than to do the work.

Persistent identity is already solved — `created=0`, exact-key hit 97.4%, the spatial map fires
~10 times a frame. `findSimilarInstance` and the spatial fallback are effectively vestigial on
the hot path. **Deleting them is hygiene, not milliseconds.**

---

## 3. WHAT UE DOES, AND WHERE WE DIVERGE

UE never asks. `FMeshDrawCommand`s are **cached at `AddToScene` time**
(`Engine/Source/Runtime/Renderer/Public/MeshPassProcessor.h:1275`) and are not rebuilt per frame.
For a static scene UE's per-frame cost is *visibility + sort*. Command construction is absent.

Ours is a **poll**: every instance, every frame, pays entry+fastRet to discover it is static.
UE's is a **push**: the 1% that moved mark themselves dirty; the 97% are never enumerated.

| stage | UE | us |
|---|---|---|
| per-draw derivation | **does not exist** — cached at scene-add | 53 ms/frame |
| visibility / culling | parallel per-primitive, culls hard via screen-size payload (`MeshPassProcessor.h:1672-1699`) | `culled=0` of 8,913 — barely exists |
| command submit | parallel, N RHI command lists (`MeshDrawCommands.cpp:1703-1721`) | serial on dxvk-cs, cannot fan out |

**So we have been building a 30-core solution for a pile of work that in UE is zero**, while the
two stages UE actually parallelises are ones we either don't have or can't.

Other UE mechanisms worth stealing later, in order of value:

1. **Redundant-state elimination at submit** — `FMeshDrawCommandStateCache`
   (`MeshPassProcessor.cpp:1254-1327`). Before issuing each command UE compares against the state
   it last set and skips anything unchanged: PSO only if `CachedPipelineId != StateCache.PipelineId`
   (`:1264`), stencil ref only if it moved (`:1302`), vertex streams compared per slot
   (`:1318-1323`), shader bindings compared per frequency inside `SetOnCommandList` (`:974-993`).
   A fresh `StateCache` is constructed per range (`:1663`), so each parallel task keeps its own
   and stays correct.
   **The half that matters: this only pays off because of the sort.** `GetSortKey_ByState`
   (`MeshDrawCommands.cpp:316`) packs pipeline id and state bucket into the key so that
   same-state commands end up adjacent, which is what makes the cache hit. Sort and cache are
   one mechanism, not two.
   **Open check `[U]`: how much redundant state DXVK already filters on its way to Vulkan has not
   been verified.** Do that before building anything here — it may already be handled.

2. **Merge, don't drop** — `bDynamicInstancing` merges identical commands into one instanced draw
   instead of discarding them, so nothing is lost visually and the CS-side cost still collapses.
   We drop instead (`filterDupSameDraws`: `drawsIn=1338 -> drawsCommit=1042`, 22%).
   **Open check `[U]`: how much duplicate commit survives the existing fanout/instancing path is
   unknown.** First move is measuring that, not writing code — the instance manager may already
   be de-duplicating most of what the filter drops.

3. **Sort by state before commit — OPEN DESIGN QUESTION, unresolved `[U]`.**
   Our `drawIndex` sort preserves issue order and does nothing else. UE sorts by state, and that
   is what makes mechanism 1 work. **We may have more freedom here than UE does:** UE's key
   carries masked / background / translucency-distance bits precisely because raster output
   depends on draw order, and we are building a *scene* for a ray tracer, not compositing a
   framebuffer. If commit order genuinely does not affect the RT scene, sorting by
   pipeline/material identity is open to us in a way it is not to UE.
   **But it may matter somewhere**: the DupFilter's first-wins behaviour and decal ordering both
   hint that order is load-bearing in at least some paths. This is a question about *our*
   renderer that UE's source cannot answer. Resolve it before assuming the freedom exists.

4. **GPU Scene** (`FInstanceCullingContext`, `MeshDrawCommands.cpp:1168, 1754`) — **rejected.**
   It targets dxvk-cs, which is not the pole today, and it presupposes a persistent scene
   representation we reconstruct per frame. The cheap conversion covers the 12.2 ms we need.

5. **Parallel command-list recording** — **rejected, see Appendix B.6.** Not merely large:
   pointless, because there are only 64 Vulkan draws per frame to record.

---

## 4. WHY REDUNDANCY BEATS PARALLELISM HERE

1. **It is the only lever that works on dxvk-cs.** One `DxvkContext`, one ordered stream, and the
   ordering is load-bearing: `commitGeometryToRT` issues a `copyBuffer` whose position in the
   stream decides which bytes it copies (`rtx_context.cpp:4948-4959`). Fan-out there is wrong,
   not merely hard. Redundancy elimination needs no parallelism at all.
2. **One mechanism serves both threads.** A draw proven unchanged is not derived (frame thread)
   *and* not emitted (dxvk-cs). The batch plan only ever helps the frame thread, because its
   output still goes through `EmitCs` in order.
3. **The multiplier is bigger.** 1/30 vs 0.03, and the 0.03 is measured, not hoped for.

### The floor is not zero `[M]`

An unchanged instance cannot simply be omitted: `[ReapJoin] removed=14 starved=14` shows the
instance manager reaps anything that stops being touched. "Skip" must mean **keep alive without
reprocessing**. That is semantically a frame-id stamp — and today we pay a call, an entry check
and a fast return per instance (0.46 us x 15.6k = 6.88 ms) to achieve it.

**The radical form: replace the per-instance visit with a bulk stamp over the unchanged set.**
Writing a frame id across a contiguous 15k array is microseconds. The 97% never enter
`updateInstance`; the dirty 3% do.

---

## 5. THE PLAN

### Phase 0 — free, today. **-2.6 ms on the pole**
Delete diagnostics sitting on the frame thread: `te_census` 1.01, `te_camDiag` 0.83,
`te_propId` 0.37, fanout `dbgTrack` 0.39. The report flags them itself as "(diagnostics?)".
No architecture, no risk.

### Phase 1 — MEASURE. **BLOCKING. Do not skip.**
Three unknowns decide whether 30 fps is reachable at all:

1. **`pfs_guard` sub-buckets** — 14.92 ms, 19% of the pole, contents unknown. `markStg` split.
   **Hypothesis `[I]`, unverified: it is largely the `DrawSnapshot` capture.** `drawSnap_ns` reads
   very large in the raw accumulator yet appears in no named leaf. This matters because capture is
   the *floor* of the whole caching approach — if capture is 15 ms, "skip unchanged draws" bottoms
   out at 15 ms and the arithmetic below fails.
2. **Draw-level redundancy** — hash `DrawSnapshot` frame to frame and report the % identical.
   Instance-level is measured at 97%; draw-level is `[I]` and everything in Phase 3 depends on it.
3. **`rtx.perfSceneObjSplit`** — attributes the ~5.2 ms of find/mid/add. Note: the old handoff
   names `rtx.perfUpdateInstSplit` for this; that is the **wrong switch** — it splits the `update`
   quarter, which is already attributed. `perfSceneObjSplit` is declared at `rtx_options.h:675`.

### Phase 2 — push-not-poll on dxvk-cs. **-7 to -10 ms on dxvk-cs**
Persistent per-instance records; the draw side marks dirty when `xfChg`/`matChg` fire (1% / 0%).
Per-frame loop iterates the dirty set only. Unchanged instances get a **bulk frame-id stamp**
instead of a call. Kills most of `entry` 2.73 + `fastRet` 4.15, and should reach into
find/mid/add once Phase 1.3 has attributed it.

Feasibility note: identity is already exact-keyed with `created=0`, so this is a read-mostly map
plus a dirty list — comparable to Phase 3 in difficulty, **not** the "structurally hostile"
problem earlier drafts of this plan claimed. Steady state has `created=0`; the creation path
during streaming and level transitions still needs handling.

### Phase 2b — split `commitGeometryToRT`: relocate what survives. **up to -21 ms on dxvk-cs**
A **different lever from Phase 2, and they compose.** Phase 2 deletes redundant work; this moves
whatever is genuinely per-frame off the CS thread entirely.

`commitGeometryToRT` is two functions wearing one name:

| half | what | must it stay on dxvk-cs? |
|---|---|---|
| **ordered GPU work** | index-stash `copyBuffer` (`rtx_context.cpp:4975`), rebinding VB/IB onto the pooled `gpuCapture` copies (`:4995-5003`) | **YES — positional** |
| **CPU scene work** | `processDrawCallState` 21.46 ms — `processSceneObject` 12.78 + `geom` 5.98 | **no reason found** |

The ordering constraint is real and load-bearing, and it applies only to the first half. From the
function's own header (`rtx_context.cpp:4951-4959`):

> this lambda replays **IN-ORDER on the CS stream**, after this draw's bindings and before any
> later Map(DISCARD) rename replay, so the logical buffer still resolves to the physical slice
> the rasterized draw consumed

So anything that **reads buffer contents** is positional — run it out of order and you copy bytes
from after the engine renamed the buffer. That is the failure mode already hit twice in this
project (the mid-upload bake, the renamed dynamic IB). The instance/scene bookkeeping that
follows has no such dependency.

**The move:** cut the seam where the positional dependency ends. Ordered GPU work stays on
dxvk-cs (it is one `copyBuffer` — cheap). The CPU scene half becomes another stage in the
frame-end parallel-for, same pattern as Phase B: make it a pure function of a per-draw record,
then batch it.

**Why this matters beyond the milliseconds:** it makes the "can we widen dxvk-cs" question moot.
If the only thing left on that thread is ordered GPU recording, it never becomes the bottleneck
again regardless of frame rate, and no amount of DXVK-core surgery (Appendix B) is ever needed.

**Unverified `[U]`:** only the head of `commitGeometryToRT` (`:4948-5003`) was read. **What
fraction is ordered-GPU versus pure-CPU has not been measured.** If the CPU half turns out to be
2 ms rather than 15, the seam is not worth cutting. Phase 1.3 (`perfSceneObjSplit`) plus a read
of the full function body settles it — do that before writing any code here.

**Order relative to Phase 2:** redundancy first. Deleting work beats moving it, and moving work
you were about to delete is wasted effort. Phase 2b applies to the residue.

### Phase 3 — skip derivation for unchanged draws. **-11 to -35 ms on the pole**
Same test, other thread. If a draw's captured inputs are byte-identical to last frame's, do not
derive it: no `bt_extractXf` (11.07), no material fill, no hashing, no bbox. Emit a keep-alive
instead of a commit. Ceiling depends entirely on Phase 1.1.

> **READ THIS BEFORE WRITING THE TEST — there is a precedent and it shipped a visible bug.**
> `rtx.filterDupSameDraws` is the same idea at a smaller scale, and its v1 key was too weak
> (`d3d11_rtx.cpp:42264-42280`). The viewmodel (gun + gauntlet hands) re-draws the **same
> geometry, same `objectToWorld`, same textures, with a DIFFERENT bone palette per pass** —
> byte-identical under a key that doesn't hash bones, visually different. The skipped
> alternate-pose commits made the viewmodel **flicker**, because temporal accumulation read the
> gaps as translucency. Bones live in `skinningData` / the bone buffers and are deliberately not
> hashed (12KB+ per draw), so the fix was to exclude skinned draws entirely.
>
> Existing scope guards on that filter, all of which Phase 3 inherits:
> - **fanout draws** (per-instance transform arrays) — never skipped
> - **blending-enabled draws** — never skipped
> - **bone-transformed draws** — never skipped (`m_currentDrawIsBoneTransformed`,
>   `skinningData.numBones > 0`, or a defined `boneMatrixBuffer`)
>
> A "nothing changed" test that hashes only what is cheap to hash will be wrong for exactly the
> things that are expensive to hash. Decide per class whether to hash it or exclude it — never
> assume absence of evidence.

### Phase 4 — fanout. **-4.9 ms on the pole**
Attribute the 2.04 ms `UNATTRIBUTED residual` in `setupPost`. Get `inst_loop` (2.84) genuinely
parallel — `rtx.parallelInstanceFanout=True` is set but `parallelInstanceFanoutMinInstances=32`
may be gating most batches out.

### Phase 5 — BLAS. **-? on dxvk-cs**
`buildBlases` 3.84 + `dynBlas` 1.82 = **5.66 ms building acceleration structures for a scene
reporting `created=0` and a flat `uniqueBlas=457`.** That is suspicious on its face and worth a
look independent of everything else.

### Phase 6 — batch/parallel for the residue
The batch system is **already built and running** — see **Appendix A** for its full anatomy,
current contents, and the exact conversion work still outstanding. It runs 1055 items in 6 ms on
30 workers today, carrying four of the six per-draw stages.

After redundancy elimination it becomes the vehicle for whatever genuinely dynamic work remains
(the dirty 1-3%), plus the once-per-frame `prepareSceneData` chunk (14.75 ms). **It is no longer
the headline** — but it is the thing that makes the dirty-set work cheap, and Phase 3's
"skip unchanged draws" test is only sound *because* the snapshot exists to hash.

### Phase 7 — delete
- **Replay tier** (~957 lines, 78 regions). 0 hits in 6.5M draws across a full session; tax
  already 0.09 ms with it off. It is also the sole consumer of `m_lastO2wCamValid` /
  `m_lastO2wCamFromFanout`, the only two fields that fail XfOverlap's verify.
- **XfOverlap worker** (~1,555 lines, uncommitted-then-committed on `stable`). One bespoke thread
  duplicating a work-stealing pool that already exists; `ovJoin` measured 9.22 us/draw of spinning
  against a 17.4 us derivation; crashes with a null deref in `ExtractTransforms` on the
  `rtx-xf-overlap` thread (symbolised from `remix-dxvk.log.prev`, 17:52:42).
- **Defer gate** (`safeToDefer` / `[DrawPure]` / `StageDep`, ~453 lines) — keep as a progress
  meter until purity work is done, then delete. Nothing gates on it at runtime.
- **Spatial-map fallback in `findSimilarInstance`** — vestigial (9.6 calls/frame). Hygiene only.

---

## 6. DOES IT CLOSE?

```
frame thread  74.76
  Phase 0     -2.60
  Phase 4     -4.90
  Phase 3    -11.07  (bt_extractXf alone; more if draw redundancy is high)
  subtotal    56.19   (~17 fps)
  pfs_guard  -14.92   IF recoverable
  target      41.27   -> still short of 33.3
```

**It does not close on the numbers currently in hand.** Reaching 33.3 ms on the frame thread needs
Phase 3 to take not just `bt_extractXf` but most of the 53 ms hook — which is exactly what
draw-level redundancy would deliver *if* it matches the instance-level 97%, and *if* the capture
floor (`pfs_guard`) is small.

dxvk-cs closes comfortably: 45.45, needs -12.2, and Phase 2 alone targets 7-10 with Phase 5 on top.

**Both unknowns are in Phase 1. That is why Phase 1 is blocking and everything else is
provisional.**

---

## 7. RULES THAT MUST NOT BE BROKEN

- **`frame ~= max(frameThread, dxvk-cs)`**; realised saving = `min(size, poleMs - secondMs)`.
  Slack is 29.31 ms today: frame-thread savings past that buy nothing until dxvk-cs comes down.
  (`rtx_options.h:380-381`)
- **Do not reorder the commit.** `commitGeometryToRT` replays in-order on the CS stream and its
  `copyBuffer` depends on that position (`rtx_context.cpp:4948-4959`). This is correctness, not
  performance.
- **Keep-alive, not omit.** `[ReapJoin]` reaps untouched instances.
- **Do not take perf numbers from a diagnostic-heavy build.** The 2026-08-10 18:25 run read
  `geomUsPerDraw=1074`; clean it is 16-20. A 55x distortion produced one wrong conclusion in this
  session already.
- **Never judge overlap perf with `extractOverlapVerify=True`** (double derivation) — moot once
  Phase 7 lands.

---

## 8. CONFIG STATE (set 2026-08-11, backup `rtx.conf.bak2-before-arch-off`)

Target architecture, on:
`useDrawSnapshot=True` · `batchSubmitDrawStages=True` · `geometryWorkerThreads=0` (auto,
hw_concurrency-2) · `parallelInstanceFanout=True` · `filterDupSameDraws=True` ·
`batchHashes`/`batchBoundingBox`/`batchSkinning` at default `true`

Dead ends, off:
`replayExtractTransforms=False` · `replayExtractVerify=False` · `perfStageDepCensus=False` ·
`extractOverlap` / `extractOverlapVerify` keys removed (do not exist at `4681ecca`)

Still to change when taking measurements:
`perfUpdateInstSplit=True` -> the counters have served their purpose (REDUNDANT=97% is captured
above); swap to `perfSceneObjSplit=True` for Phase 1.3. The broader diagnostic set
(`logSubmitStall`, `perfThreadCensus`, `logPrepSceneSplit`, `sceneCull.logStats`,
`debug.dumpVertexShaders`) should be stripped before any final frame-time claim.

---

# APPENDIX A — THE BATCH PARALLEL SYSTEM, AS BUILT

Everything here exists and runs today. This appendix is the record of what it is, so the plan
above can refer to it without re-deriving it. Nothing in this appendix is speculative unless
marked.

## A.1 The pool

`WorkerThreadPool<NumTasksPerThread, WorkStealing, LowLatency>` — `src/util/util_threadpool.h`
(~570 lines). Work-stealing, one SPSC ring queue per worker, `Schedule()` returns `Future<R>`
(`:348`), workers spawned at `:310`, destructor drains and cancels outstanding tasks (`:317-344`).

Three pools instantiate it in-tree:
- `m_pGeometryWorkers` — the one that matters (below)
- `dxvk_raytracing.cpp:109` — deferred BLAS builds
- `rtx_asset_exporter.h:73` — asset export

**It is not stock upstream code.** Three scaling defects were fixed in it on 2026-07-28,
documented at `d3d11_rtx.h:889-898`: one pool-wide steal spinlock split into per-queue
cache-line-padded locks; `notify` moved out from under the mutex in `Schedule`; a failed steal
scan made to yield instead of spin whenever any queue held work. Before those landed, **raising
the worker count made the pool slower**, which is why `rtx.geometryWorkerThreads` exists as an
escape hatch.

`LowLatency=false` deliberately (`d3d11_rtx.h:880-888`): the pool runs its parallel-for once per
frame then idles, and the default spin mode had every idle worker busy-looping the steal scan at
100% CPU for ~95% of every frame — a uniform ~40% inflation of *all* serial work. Matches the
other two pools.

```
using GeometryProcessor = WorkerThreadPool<kMaxConcurrentDraws, /*WorkStealing*/ true,
                                           /*LowLatency*/ false>;          // d3d11_rtx.h:899
std::unique_ptr<GeometryProcessor> m_pGeometryWorkers;                     // d3d11_rtx.h:902
```

Width: `rtx.geometryWorkerThreads = 0` = auto = `hardware_concurrency - 2`, leaving one core for
the game thread and one for dxvk-cs. Measured at **30 workers** on this machine.

## A.2 The arena

`GeometryBatchArena m_geoBatch` (`d3d11_rtx.h:903-909`) — a per-frame arena of `DrawWorkItem`,
each holding that draw's `DrawCallState`, its `MatSnapshot`, and optional hash / bbox / skin jobs.

Two scope rules, both load-bearing:
- **Immediate context only.** Deferred contexts never call `EndFrame`, so they keep the per-draw
  `EmitCs` path and cannot orphan the arena.
- **Owning (game) thread only** while collecting, so no lock is needed.

## A.3 The parallel-for — `flushGeometryBatch()`

`d3d11_rtx.cpp:~25490-25615`. Per item, `runRange(begin,end)` does (`:25505-25530`):

```
FillMaterialData(it.matSnap.resultMat, it.matSnap);   -> it.dcs.materialData; releasePins()
if (it.hasHashJob)  it.dcs.geometryData.hashes       = runBatchHashJob(it.hashJob)
if (it.hasBboxJob)  it.dcs.geometryData.boundingBox  = runBatchBboxJob(it.bboxJob)
if (it.hasSkinJob)  it.dcs.skinningData.pBoneMatrices.adopt(runBatchSkinJob(it.skinJob))
```

Dispatch (`:25533-25547`) — four properties worth preserving in any rewrite:
1. chunked into `chunks` ranges of `chunkSz`
2. **the game thread runs the tail chunk itself** rather than scheduling it and idling
3. **inline fallback**: if `Schedule` returns an invalid future (worker queue full), the range is
   run inline rather than dropped
4. one `Future<void>` per scheduled chunk, collected into `futs`

Join (`:25549-25567`): a single barrier, `futs[fi].get()` in order. Because `Future::get()`
busy-spins with **no timeout**, a `[BatchJoin]` watchdog publishes `g_bjFrame`, `g_bjItems`,
`g_bjChunks`, `g_bjFutCount`, `g_bjIndex`, `g_bjStartNs`, `g_bjActive` before the join, and
`g_bjItemsDone` is bumped per item by whichever thread runs the range — so a hang can be told
from a slow worker.

## A.4 The ordered handoff ("Phase C" in the code)

`d3d11_rtx.cpp:25571-25588`. After the barrier, items are walked **in arena order — which is
draw order** — and each is handed to dxvk-cs:

```
m_context->EmitCs([params, dcs = std::move(it.dcs)](DxvkContext* ctx) mutable {
  static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, dcs);
});
```

`[MeshTrace] BatchEmitted` is recorded before the move, while `it.dcs` is still readable, so a
draw lost between collect and emit can be distinguished from one lost after `commitGeometryToRT`.
`items.clear()` afterwards keeps capacity (grows once).

Heartbeat every 3 s: `[BatchSubmitDraw] frames= itemsPerFrame= parallelForMsPerFrame= workers=`.
Current reading: **`itemsPerFrame=1055 parallelForMsPerFrame=6 workers=30`**.

## A.5 What is in the batch today, and what is not

| stage | in the batch? | flag |
|---|---|---|
| `FillMaterialData` | **yes** | `rtx.deferMaterialCompute` (`rtx_options.h:351`) |
| geometry hashing | **yes** | `rtx.batchHashes` (`:357`) |
| object-space bbox scan | **yes** | `rtx.batchBoundingBox` (`:358`) |
| bone-palette build | **yes** | `rtx.batchSkinning` (`:359`) |
| **`ExtractTransforms`** | **NO** | blocked on Phase B purity — see A.7 |
| **instance find/mid/add** | **NO** | on dxvk-cs; Phase 2 target |

Parent flag `rtx.batchSubmitDrawStages` (`:353`). Sub-flags default `true` so enabling the parent
gives the fully coherent batch; disable one to bisect a stage back to its per-draw path.

## A.6 DrawSnapshot — why any of this is legal

`rtx.useDrawSnapshot` (`rtx_options.h:682`, default `false`, **set True in conf**).

The design rationale is in the header block at `d3d11_rtx.h:65-87` and is the single most
important paragraph in the codebase: SubmitDraw's derivation reads **live** context state —
~460 read sites across 24 distinct `m_context->m_state` paths — *"so they can only run on the
frame thread, in draw order, before the game rebinds for the next draw. That is the ONLY reason
the derivation is serial."*

`DrawSnapshot` is the fix, modelled explicitly on UE's `FMeshDrawCommand`
(`MeshPassProcessor.h:1281`): capture once at the draw entry point, then every derivation stage
becomes a pure function of it.

Supporting machinery, all live:
- **Accessors** that read the record instead of the context: `drawVertexShaderCom()` (x36 at
  `4681ecca`), `drawCbSpan()` (x31), `drawVertexBuffer(slot)`, `drawInputLayout()`, `drawVsSrv(slot)`
- **Per-VS span manifest** — learns which cbuffer byte ranges a given vertex shader's draws
  actually need, so capture copies those and nothing more
- **Carrier groups** — `SdepGroup` enum (`d3d11_rtx.h:56-63`): `kSdepCam`, `kSdepRoute`,
  `kSdepCbLoc`, `kSdepBone`, `kSdepStatic`; `DrawSnapshot` stores one hash **per group**
- **`[DrawSnap]` metrics**: `draws / resolved% / meanRanges / manN / ovf / cbOff / wcMiss / wcHit`

**This hash-per-group is what makes Phase 3 possible.** Skipping an unchanged draw needs a cheap,
sound "identical to last frame" test, and a hash of the captured record is exactly that. Without
the snapshot the only way to answer it is to re-read live state — i.e. to pay the cost you are
trying to avoid.

## A.7 Phase B — the conversion work still outstanding

The consumer half (join -> `SetSkyCategoryFromCb2`, ~9,240 lines) was censused at
**77 direct `m_state` reads + ~27 indirect**. Batch 1 converted 45 sites.

> **Branch warning:** Batch 1 lives on `stable` (`470eaf4b`), **not** on `parallel-clean`.
> At `4681ecca`, `drawIndexBuffer` and `drawPixelShaderCom` have **zero** uses. Cherry-pick those
> 45 sites out of `xfoverlap-wip`/`stable` before continuing — they are independent of the
> XfOverlap worker and behaviour-neutral.

**32 residual live reads remain**, in three classes:

*Batch 2 — identity reads, need new snapshot fields/accessors:*
`rs.viewports` x9, `rs.numViewports` x4 (check how many sites actually want viewport 0),
`om.cbState` x3, `om.dsState` x2, `rs.state` x1 (blend/depth/raster object pointers — note the
existing BlendCache/DepthCache decode caches already key on these pointers),
`ps.shaderResources` x2, `ps.constantBuffers` x2, plus `GetCurrentVsPsHashes` x12 (add one
snapshot-hash accessor).

*Batch 3 — CONTENT reads, the careful class:*
the 14 `GetMappedSlice` sites, plus 4 `vs.constantBuffers` binding+content sites. Route through
`drawCbSpan` / the manifest like the nine already-converted consumers.
**Rule: never convert a content-read site's binding without its bytes.** Doing so mixes two
instants and is the exact bug class the record exists to prevent.

*Purity-neutral, leave alone:*
4x `vertexBuffers.size()` bound checks — `std::array`, constexpr, not a live read.

**When those 32 are closed**, adding `ExtractTransforms` to the batch is small: one more stage in
`DrawWorkItem`, one `rtx.batchExtractTransforms` sub-flag beside `batchHashes`/`batchBoundingBox`/
`batchSkinning`, and the existing dispatch/join/emit path is unchanged. `[DrawPure]`'s
`cbLive`/`geoLive` counters going to 0 is the progress meter.

## A.8 What must NOT be done to this system

- **Do not reorder the emit.** `commitGeometryToRT` replays in-order on the CS stream and its
  `copyBuffer` depends on that position (`rtx_context.cpp:4948-4959`). Correctness, not perf.
- **Do not remove the tail-chunk-on-game-thread or the inline fallback.** They are what stop a
  saturated pool from dropping work or idling the caller.
- **Do not remove the `[BatchJoin]` watchdog** while `Future::get()` still spins without a timeout.
- **Do not half-convert a content read** (A.7, batch 3).
- **Do not re-enable `LowLatency=true`** on this pool (A.1).

## A.9 Deleted / rejected around this system, for the record

- **XfOverlap** — a bespoke single `std::thread` + one job slot + cv + spin join
  (`d3d11_rtx.h:2234-2245`, `d3d11_rtx.cpp:24012-24243`), duplicating the pool that already
  existed, one job in flight, joined inside the same SubmitDraw. Measured `ovSched` 1.02 us/draw
  and **`ovJoin` 9.22 us/draw of spinning** against a 17.4 us derivation — the shadow was only
  ~8.8 us, so it could hide at best ~half. Crashes: null read at +0x78 in `ExtractTransforms` on
  the `rtx-xf-overlap` thread (symbolised from `remix-dxvk.log.prev` 17:52:42, `std::thread::
  _Invoke` at frame[10]). Verify mode fails 8/8 on `m_lastO2wCamValid` + `m_lastO2wCamFromFanout`,
  whose **only** consumer is the replay tier. Lives on `stable`; delete rather than fix.
- **Replay tier** — memoise-the-derivation. 0 hits / 0 carries across 6,542,782 draws in a full
  21-minute session, 99.94% ineligible, `mapSize=4`, `rp_commit=0`. Cost was ~691 ns/draw for
  nothing. Now off; tier tax 0.09 ms.
- **EndFrame deferral plan** — never was code.
- **cb3 predictor / sentinel seeds / frame-scoped hint** — tombstoned; the frame-scoped hint
  attempt broke the replay tier (`hit/full 302k/22k -> 40k/284k`) and is documented at
  `d3d11_rtx.cpp:17403-17409` and `:28843-28849`. Do not retry.

---

# APPENDIX B — DXVK-CS: WHY IT CANNOT BE FANNED OUT

This is written down because "just use secondary command buffers" is the obvious idea and it is
wrong for three independent reasons. `[M]` all three verified in-tree.

**1. DXVK does not use secondary command buffers at all.** Every allocation in the tree is
`VK_COMMAND_BUFFER_LEVEL_PRIMARY` (`dxvk_cmdlist.cpp:44`, `:51`). There is no
`vkCmdExecuteCommands` and no `VkCommandBufferInheritanceInfo` anywhere.

**2. The CS "command stream" is not a Vulkan command buffer.** It is `DxvkCsChunk`
(`dxvk_cs.h:167`) — a queue of deferred **C++ calls**. The CS thread pops a chunk and invokes
`DxvkContext` methods; *those* record Vulkan. Two layers, and the single-threading lives in the
upper one.

**3. The serialisation point is `DxvkContext`, not the command buffer.** It holds the whole
D3D11->Vulkan translation state — bound pipeline, descriptor sets, barrier tracking, render-pass
state. Two threads replaying chunks into one `DxvkContext` corrupt that translation state long
before the command buffer matters. Secondary command buffers give parallel *recording*; they do
not give a second `DxvkContext`, so they do not touch the constraint.

On top of that, the work is the wrong shape: what we hand dxvk-cs per draw is
`commitGeometryToRT` = one `copyBuffer` plus a body of CPU scene work. Secondaries would
parallelise the `copyBuffer` and nothing else.

## B.6 "N CS threads, each with its own DxvkContext" — the honest analysis

This is the genuine DXVK-native version of parallel command lists, and it is the only structure
that would allow **one fused worker doing derivation AND commit for a draw**. It deserves a real
answer rather than "too big".

**Would the ordering constraint survive?** Yes, if split correctly. Splitting the CS stream into
**contiguous draw ranges** (not round-robin) and submitting the primary buffers in range order
preserves the relationship between a draw's commit and any later `Map(DISCARD)` rename, because
renames land in whichever range they belong to and cross-range order is the submit order. This is
the same reasoning as UE's fresh-`StateCache`-per-range (`MeshPassProcessor.cpp:1663`).
**Interleaved splitting would break it.**

**So why is it still rejected? Because there is nothing to record.** `[Perf.Block]` per frame:

```
csChunks=1299   submits=11   vkDraws=64   barriers=340
```

**64 Vulkan draws per frame.** This is a path tracer — the D3D11 draw stream becomes *scene data*,
not draw commands. Parallel command-buffer recording would parallelise ~64 draws and ~340
barriers. dxvk-cs's 45.45 ms is **not** command recording; it is `commitGeometryToRT` scene work
(26.39) plus `prepareSceneData` (15.36).

**And it would not enable the fused worker anyway.** The scene work races on shared RT state —
the instance manager, BLAS cache, geometry cache — not on the command buffer. N `DxvkContext`s
give N command buffers; they do not make `processSceneObject` thread-safe. You would do the entire
DXVK-core rewrite and still be blocked on exactly the purity problem Phase 2 / 2b solve. Plus
cross-context barrier reasoning at every range boundary, which is the genuinely hard part.

**The fused worker is reachable without any of this — it is Phase 2b.** Move the CPU scene half
to the pool and a worker already does derivation + scene work for a draw. What stays on dxvk-cs is
one `copyBuffer` per draw and 64 vkDraws per frame. That is the fused worker for everything that
costs anything, at the price of a conversion instead of a rewrite.

**Verdict: not "too big" — wrong target.** Do not revisit.

**Therefore: dxvk-cs can be made cheaper or emptier, never wider.** Two levers, and they compose:

- **cheaper** — Phase 2, redundancy elimination. The only lever that needs nothing from DXVK.
- **emptier** — Phase 2b, split `commitGeometryToRT` and relocate the CPU scene half to the
  worker pool, leaving only the ordered GPU recording behind.

**Phase 2b is what makes this appendix permanently irrelevant.** Recording is cheap; if it is the
only thing left on that thread, dxvk-cs never becomes the bottleneck again at any frame rate, and
the question of widening it never has to be asked. That is a better outcome than any amount of
DXVK-core surgery, and it costs a conversion instead of a rewrite.

---

# APPENDIX C — MEASUREMENT NOTES AND PROBE RELIABILITY

Read this before trusting any number, including the ones in this document.

## C.1 The present thread IS the draw thread

`[Perf.Busy] tid=21212` and `[Perf.SdThreads] tid=21212 draws=88261` are the **same thread**.
The `(PRESENT)` tag in `[ThreadCensus]` therefore does not mark some untouchable Source engine
thread — it marks the thread `SubmitDraw` and the RT injection run on.

**This produced a wrong conclusion once in this session.** `[ThreadCensus]` also lists
`Titanfall2.exe+0x2280` at ~10% of a core; reading *that* as "the draw thread" led to the
conclusion that the frame thread had slack and frame-thread optimisation was pointless. It is not
the draw thread. Do not repeat it.

## C.2 The present thread is BUSY, not blocked — and this changed

Six consecutive `[Perf.Busy]` windows at 02:47:
```
wallMs=94.4 cpuMs=89.6 busy=94.9% blocked=4.8      wallMs=85.8 cpuMs=82.9 busy=96.6% blocked=3.0
wallMs=86.5 cpuMs=81.9 busy=94.7% blocked=4.6      wallMs=77.4 cpuMs=74.8 busy=96.5% blocked=2.7
wallMs=103.4 cpuMs=99.8 busy=96.5% blocked=3.6     wallMs=77.0 cpuMs=72.6 busy=94.3% blocked=4.4
```
**94-97% busy, 3-5 ms blocked.** It is computing, not waiting on present/vsync/GPU/CS.

The `[ThreadCensus]` header (`d3d11_rtx.cpp:5667-5674`) records that on **2026-08-06** this same
probe read **66.6 ms BLOCKED of a 102.3 ms frame** — asleep on the engine's condvar. The
situation has **inverted** since. Any reasoning inherited from notes written before 08-06 about
the frame thread waiting on engine threads is stale.

## C.3 Throttling is ruled out `[M]`

`cpuSlowX` in `[Perf.Frame]` is the CPU-downclock detector; it "pins at 1.00" at full speed
(`rtx_cpu_stall_probe.h:19`). Across the run it reads
`1.00 2.21 2.77 3.41 3.68 3.68 2.44 2.22 2.15 1.65 3.54 6.18 4.41 3.56 1.33 1.00 2.56 ...` —
it returns to **1.00 ninety seconds in**, then rises again. Thermal throttling has a time
constant of minutes and does not recover for a single sample. This is scatter from the
calibration thread being **descheduled** under contention (`busySum=280%core`, 6 active threads),
not clock drop. Corroborated: the GPU is *idle* (56.86 ms), not busy-and-slow.

## C.4 Probes that are currently unreliable

- **`[Perf.SdThreads]` is degenerate.** It reports `cpuUs=0 stallUs=2045802` with `stall == wall`.
  It is not filling `cpuUs`. **Ignore its stall figure**; `[Perf.Busy]` carries the real
  busy/blocked split. The two "disagree" because SdThreads is broken, not because the thread is
  both.
- **Three `[Perf.Report]` cross-validation rows read `SKIP - source not published`**, including
  **both dxvk-cs rows** (`SceneObj vs ProcDCS instMs`, `UpdInst sum vs SceneObj update`) and the
  SubmitDraw `SdStall` row. So the **Phase 2 / Phase 4 dxvk-cs numbers are the least solid in
  this document.** `rtx.perfSceneObjSplit` is what firms them up.
- Rows that DO pass: `frame thread Entry+Gap vs wall 77.63 vs 77.44 (0.2%)`,
  `fanout InstDraw vs DrawIdxInst entry 24.67 vs 25.31 (2.5%)`,
  `dxvk-cs PrepScene vs CsSplit fat chunk 14.75 vs 15.36 (3.9%)`. Frame-thread accounting is sound.

## C.5 Diagnostics distort by more than the thing being measured

Same scene, dirty vs clean build:
```
dirty (2026-08-10 18:25):  [ProcDCS] perFrameMs=1538  geomMs=1296  geomUsPerDraw=1074
clean (2026-08-11 02:47):  [ProcDCS] perFrameMs=57-67 geomMs=17-21 geomUsPerDraw=16-20
```
**A 55x distortion, concentrated in whichever subsystem is most instrumented.** It produced a
wrong priority call in this session (geometry appeared to dominate the instance manager by 24x;
clean, the instance manager is ~2x geometry). Strip diagnostics before any comparison.

## C.6 Where the data lives

- Runtime log: `<Titanfall2>/rtx-remix/logs/remix-dxvk.log`, previous run `.log.prev`
- The `[Perf.Report]` block is emitted every `rtx.perfReportFrames` (50) but **interleaves with
  other threads' output** — filter to `[Perf.Report]` lines only and take the span between the
  last two `====` separators.
- `rtx.conf` backups from this session: `rtx.conf.bak-2026-08-11-parallel-clean` (before the
  `extractOverlap` key removal) and `rtx.conf.bak2-before-arch-off` (before replay/StageDep off).

---

# APPENDIX D — CODE INVENTORY AND BRANCH STATE

## D.1 Size of each system, current tree `[M]`

Region footprint = contiguous runs of subsystem-owned lines, gaps <= 15.

| system | lines | regions | verdict |
|---|---|---|---|
| DrawSnapshot + manifest | 1,130 | 169 | **keep — keystone** |
| XfOverlap | 1,555 | 171 | delete (on `stable` only) |
| Replay tier | 957 | 78 | delete |
| Defer gate (`safeToDefer`/`DrawPure`/`StageDep`) | 453 | 44 | keep as progress meter, then delete |

The replay tier's 78 regions are woven **through** `ExtractTransforms`, which is why removing it
is surgery rather than a flag flip. Consider defaulting `replayExtractTransforms=false` first
(already done in conf), confirming nothing downstream depended on it, then deleting.

## D.2 When each system was introduced `[M]`

| symbol | commit | date |
|---|---|---|
| `WorkerThreadPool` | `2796e2a8` | initial (upstream) |
| `GeometryBatchArena`, `batchSubmitDrawStages` | `610950c5` | **2026-07-24** |
| `deferMaterialCompute` | `fe1a7fe9` | 2026-07-24 |
| `parallelInstanceFanout` | `f232312c` | 2026-08-07 |
| `replayExtractTransforms` | `2cfb912d` | 2026-08-09 |
| `useDrawSnapshot`, `struct DrawSnapshot`, `perfStageDepCensus` | `36666fef` | 2026-08-10 |
| `safeToDefer` | `4681ecca` | 2026-08-10 |
| `XfCarriers`, `extractOverlap` | `470eaf4b` | 2026-08-11 |

**The batch engine is two weeks older than DrawSnapshot.** The parallel infrastructure was never
the missing piece; purity was.

## D.3 Branches

- **`parallel-clean` @ `4681ecca`** — current. Clean tree, no XfOverlap. Work here.
- **`stable` @ `470eaf4b`** — the XfOverlap commit, message "broken". **Already pushed to
  `origin/stable`**, so do not rewind it without a force-push decision. Contains Phase B batch 1
  (the 45 accessor conversions), which is wanted — cherry-pick, don't discard.
- `main` @ `62378bcf` — "broken", stale.

## D.4 Per-commit subsystem attribution `[M]`

Added lines mentioning each subsystem, showing where effort went:

```
2cfb912d 08-09   replay 226   snapshot 1
eba174d5 08-09   replay  87   snapshot 5
e85456cb 08-09   replay  92   snapshot 1
8fca7e63 08-09   replay  20   snapshot 1
36666fef 08-10   replay   9   snapshot 104   defer 47
4681ecca 08-10   replay   3   snapshot 170   defer 70
470eaf4b 08-11   replay   3   snapshot  68   defer 11   overlap 618
```

08-09 was replay-tier day (425 lines, no snapshot work). 08-10 pivoted to DrawSnapshot. 08-11 was
90% overlap. Two of those three days produced code this plan deletes.
