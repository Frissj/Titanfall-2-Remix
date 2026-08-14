# THE OPTIMISATION PLAN

**Written:** 2026-08-11 · **Last reconciled against the tree:** 2026-08-14
**Branch:** `architecture-overhaul` @ `5ced56c2` (was `4681ecca` when this was written; six commits have landed since)
**Goal:** 30 fps = 33.3 ms/frame
**Source of numbers:** §1 carries TWO baselines. The 2026-08-11 one at `[Perf.Report]` 02:47:54
is the historical anchor most of this document was sized against; the **2026-08-14 marks-off
baseline is the current truth** and is the one to quote. Where a section still reads against the
old anchor it says so.

**READ §8 BEFORE ANY LEAF NUMBER.** Since 2026-08-14 there are TWO instrumentation switches, both
defaulting OFF, and with them off the entire named-leaf table reads zeros *by design*. That is no
longer the lost-env-var false alarm §8 was originally written about — it is now the normal state
of a normal run.

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

## 0.2 STATUS — 2026-08-14. **THE CAPTURE FLOOR WAS NOT A FLOOR.** `[M]`

The single largest correction this document has taken. Phase 1.1 concluded that `pfs_guard` is
95.4% `captureDrawSnapshot`, that the capture floor is therefore ~13 ms, and that §6's
`-14.92 pfs_guard` line had to be **revoked** because "removing it removes Phase 3".

**Measured 2026-08-14, verify off:**

```
pfs_guard    14.92 ms  (24.5% of SubmitDraw, largest row in the report)
          ->  0.66 ms  (1%)
```

The capture was not a floor. It was doing ~4x more work than any consumer read, and the excess
came out without touching what Phase 3 runs on. §6's revocation is itself revoked — see §6.

### What was built

| # | Change | Result `[M]` |
|---|---|---|
| 1 | `[Perf.SplitXf.CbWaste]` — the capture/read cross-tab | 49% of captured **bytes** and 37% of write-combined **transactions** were for spans nothing read |
| 2 | `ObjKeyMask` bit 23 — the **draw range** in the identity key | `keys/f 760->923`, `maxPerKey 48->21`, `FAIL=0` |
| 3 | `splitTransformManifestPathFilter` — per-**path** capture | `wcMiss/draw 1.702 -> 0.433`, waste ~0, `satN 0`, `ovf 0` |
| 4 | `kMaxCbManifest` 6 -> 10 | `satN 18,504 -> 0`, `manN 4.75` |
| 5 | `splitTransformVerify = False` | the cache actually serves: `serve 65%`, `derive` 35% of draws |

Both correctness gates held throughout: `FAIL=0`, `FAILcarrier=0 grp{}` empty.

**AND THE §8 TRAP WAS CHECKED, BECAUSE THIS IS EXACTLY ITS SIGNATURE.** §8 warns that a leaf
reading near zero means `RTX_D3D11_SUBMARK` was lost, not that anything got faster — and it cites
`pfs_guard 0.28 ms` as the worked example of that false win. So, from `[Perf.SubmitDraw.acc]`,
same window, ns:

```
pfs_guard   =  35,796,500   -> 0.80 ms/frame over 45 presentFrames  (report says 0.66)
pfs_studio  =  19,697,700       alive
pfs_drop    =  10,503,100       alive
pf_setup    =  16,934,700       alive
bt_extractXf= 570,567,900       alive
tail_emit   = 329,890,900       alive
```

The whole named-leaf table is live and `pfs_guard`'s own accumulator independently agrees with the
per-frame figure. The sub-markers were on; the drop is real.

### The mechanism, in one line each

- **The capture was keyed per VS; whether a span is read is decided by the o2w PATH**, and 55% of
  draws sit on a VS that takes more than one. So p13's cb3 span was copied onto p1's draws every
  draw, all frame. `p1 dead{3=8184/8063}` — cb3 captured on 100% of p1's draws, read on zero.
- **The object identity key covered the IA binding but not the draw call's own
  `Start`/`BaseVertexLocation`**, so different meshes sub-allocated out of one pooled buffer shared
  a key and evicted each other. That is the collision N-way was built to absorb.

### What this changed about the plan

- **Phase 1.1's "capture floor ~13 ms" is refuted.** It is 0.66 ms.
- **§6's frame-thread arithmetic reopens by ~14.3 ms.** See the corrected block there.
- ~~**The frame thread is no longer where the big number is.** With `pfs_guard` at 1%, the dominant
  CPU cost is `dxvk-cs` — `prepareSceneData 23.22` + `merge 17.89` ~= 41 ms — with **75% GPU idle**.
  That is Phases 2 / 2b, still untouched.~~

  > **WRONG, CORRECTED 2026-08-14 `[M]`. `merge` IS A CHILD OF `prepareSceneData`.** That sum adds
  > a parent to its own child. The report nests them explicitly:
  >
  > ```
  > the once-per-frame fat chunk   22.86
  >   prepareSceneData             23.22   (nested)
  >     merge                      17.89   (nested)
  > ```
  >
  > and its own cross-validation row agrees: `PASS dxvk-cs PrepScene vs CsSplit fat chunk
  > 23.22 vs 22.86 (1.5%)`. The chunk is **22.86 ms, not 41**.
  >
  > **The frame thread is still the pole**, and the report says so in as many words:
  >
  > ```
  > BOTTLENECK: CPU, on the frame thread (95.66 ms). Second is 75.17 ms, so SLACK = 20.50 ms.
  > ```
  >
  > That survives the `clkNs` caveat: against the §1 baseline, dxvk-cs inflated **more** than the
  > frame thread (75.17/45.45 = 1.65x vs 95.66/74.76 = 1.28x), so a clean-clock window widens the
  > gap rather than closing it. Per §7 there are **~20.5 ms of frame-thread savings that realise in
  > full** before dxvk-cs binds.
  >
  > **So do not skip the frame thread for Phases 2 / 2b on this reasoning.** The ordering claim at
  > the end of §6 ("Phases 2 / 2b are now the binding constraint, not Phase 3") rests on the same
  > double-count and is void with it.

### Open, and named honestly

- **`bt_extractXf` did not move** — 11.41 ms against an 11.07 baseline, at 65% serve with verify
  off. It should have dropped substantially. Either the serve is not saving what it should, or the
  compose path costs what the derivation did. **Still unexplained `[U]`, and now UNMEASURABLE by
  default:** with `RTX_PERF_MARKS` off (§8) `bt_extractXf` reads 0. Settling it needs a marks-on
  window plus an A/B of `rtx.splitTransformCache`. Serve is 64% on 2026-08-14
  (`serve=52958 derive=29005`, `FAIL=0`), so the question is unchanged, only the instrument moved.
- ~~**Every absolute ms in the 2026-08-14 report was taken at `clkNs=101`**~~ — **CLOSED. The clean
  window arrived.** Both 2026-08-14 captures (03:48 and 04:47) read `clkNs=32-49`, `xMin=1.00`,
  `cpuSlowX=1`. The absolute milliseconds in §1's current baseline are therefore load-bearing, not
  void. This is the capture that block said was owed.
- **Phase 0 is VOID, not outstanding** — see §5. All four of its targets were checked in source and
  none is a diagnostic. Nothing to do.
- **THE SLACK HAS COLLAPSED, and it changes the ordering of everything below.** 29.31 ms on
  2026-08-11; **2.91 ms** on 2026-08-14. The two CPU threads have converged. Per §7's realisation
  rule, frame-thread work now buys ~2.9 ms before dxvk-cs binds, so the paragraph above
  ("there are ~20.5 ms of frame-thread savings that realise in full") **is spent**. Both columns
  now have to come down roughly together; neither alone moves the frame.

---

## 1. BASELINE `[M]`

### 1a. CURRENT — 2026-08-14, marks OFF, clean clock. **QUOTE THIS ONE.**

```
FRAME 40.56 ms (24.7 fps)   inst=15445  uniqueBlas=462  drawsIn=1349  drawsCommit=1078
                            matNew=0    matTotal=1471   texTotal=2461

frame thread (PRESENT)   38.57 ms   busy 93.6%   blocked 2.63 ms    <- STILL THE POLE
dxvk-cs                  35.66 ms   busy 89.9%   idle    4.02 ms
GPU                      13.96 ms                idle   20.69 ms (35% busy)

SLACK = 38.57 - 35.66 = 2.91 ms
clkNs = 32-49   xMin = 1.00   cpuSlowX = 1        <- machine state SANE, per C.3b
```

**77.44 -> 40.56 ms is real, and it is two effects that must not be conflated.** The
`pfs_guard` work (§0.2) took ~14 ms off the frame thread; the new `RTX_PERF_MARKS` switch
(§8) removes ~1.6 ms of instrument that every earlier number in this document was paying.
Same scene either way: `inst` 15431 -> 15445, `uniqueBlas` 457 -> 462, `drawsIn` 1338 -> 1349.

**THE STRUCTURAL CHANGE IS THE SLACK, NOT THE FRAME TIME.** 29.31 ms -> 2.91 ms. The two CPU
threads have converged, so §7's realisation rule now bites on every line in §6: a frame-thread
saving larger than ~2.9 ms does not appear in the frame time until dxvk-cs comes down with it.
The GPU is still not the limit — 20.69 ms idle, ~26 ms of headroom.

#### Budget to 30 fps, current

| thread | now | target | must cut |
|---|---|---|---|
| frame thread | 38.57 | 33.3 | **5.3 ms** |
| dxvk-cs | 35.66 | 33.3 | **2.4 ms** |
| GPU | 13.96 | — | none |

**30 fps is now a small number away on both threads** — 5.3 and 2.4 ms, against the 41.5 and
12.2 it needed on 2026-08-11. That is the single most important change to this plan's outlook
and it is why the phases below are re-ordered by *realisable* size rather than gross size.

#### Frame-thread composition, current `[M]`

```
D3D11 entry points               24.82 ms  61%
  DrawIndexedInstanced           10.89 ms  27%   (nested)  <- 100% UNATTRIBUTED, see below
  DrawIndexed                    10.48 ms  26%   (nested)
  Draw                            0.52 ms   1%   (nested)
  state-setting calls             2.92 ms   7%   (nested)
between entry points             16.15 ms  40%
  afterQueryEnd                  10.84 ms  27%   (nested)
```

Two things in that block are new and neither is in the older analysis:

- **`between entry points` is 40% of the pole, and it is NOT all Source engine.** The batch
  parallel-for runs from `EndFrame`, on this thread, and `[BatchSubmitDraw]` measures it at
  **`parallelForMsPerFrame=6` with `itemsPerFrame=1068 workers=30`**. That 6-7 ms lands here and
  has been read as engine time. See §5 Phase 6e.
- **`DrawIndexedInstanced` 10.89 ms is the fanout path and it is entirely dark** with marks off —
  `SubmitInstancedDraw` reads 0.00 and the report FAILs its own fanout cross-validation row
  because of it. 28% of the pole cannot currently be seen.

`SubmitDraw` wall is **14.01 ms/frame** (`[Perf.SdThreads] wallUs=1723539 / 123 frames`,
1374 draws/frame => 10.2 us/draw). Of that, `captureDrawSnapshot` is 3.80 and the remaining
~10.2 ms is unattributed while marks are off.

#### `captureDrawSnapshot` is now BOOKKEEPING, not copying `[M]`

Per frame, from the unconditional timers that survive marks-off:

| item | ms | what it is |
|---|---|---|
| `drawSnap` TOTAL | **3.80** | |
| identity block | 1.52 | `dsi_scan 0.76` + `dsi_ident 0.40` + `dsi_t31 0.25` |
| cb span copy | 2.20 | of which **the actual write-combined memcpy is 0.60** |
| — carrier-group hashing | 0.39 | `stageDepCarrierGroups` |
| — named-span locate | 0.68 | |
| — manifest block | 0.75 | |

**The path-filter fix worked and left a different problem behind.** `[Perf.SplitXf.CbWaste]` now
reads `capB/f=0 wasteB/f=0` — nothing dead is copied. But only 27% of the "span copy" bucket is
copying; the rest is *deciding what to copy*. `dsi_scan` 0.76 ms is the 128-SRV + 32-VB walk,
i.e. exactly Phase 8's target, and `[DrawPure] dynSrvSlots{t4=684 t31=103028}` says two slots of
128 ever matter.

### 1b. HISTORICAL — 2026-08-11 `[M]`. The anchor most of this document was sized against.

```
FRAME 77.44 ms (12.9 fps)   inst=15431  uniqueBlas=457  drawsIn=1338  drawsCommit=1042
                            matNew=0    texNew=0        created=0

frame thread (PRESENT)   74.76 ms   busy 96.5%   blocked 2.68 ms    <- POLE
dxvk-cs                  45.45 ms   busy 66.4%   idle   23.04 ms
GPU                      15.71 ms                idle   56.86 ms (73%)

SLACK = 74.76 - 45.45 = 29.31 ms
```

`csSyncMs=0 gpuSyncMs=0` — nobody blocks on anybody. This is pure CPU work on two threads.
The GPU is idle because it is **starved**, not because it is fast.

Budget as it stood then — kept because §6's arithmetic is still written against it:

| thread | then | target | must cut |
|---|---|---|---|
| frame thread | 74.76 | 33.3 | **41.5 ms** |
| dxvk-cs | 45.45 | 33.3 | **12.2 ms** |
| GPU | 15.71 | — | none |

**Both threads must be fixed.** `frame = max(frameThread, dxvk-cs)`, so cutting only one
stops paying the moment it crosses the other. That was true at 29.31 ms of slack and it is
far more binding at 2.91.

### Frame thread composition `[M]`

```
D3D11 entry points               58.42 ms  75%
  -> Remix OnDraw* hook          53.01 ms  68%   <- OUR CODE
between entry points             19.21 ms  25%   <- Source engine, untouchable
```

Our injection is **71% of the pole**. Floor with injection at zero is ~22 ms (~45 fps).
30 fps therefore requires cutting 53.01 -> ~11.5 ms, i.e. keeping 22% of current per-draw cost.

Named leaves inside SubmitDraw — **2026-08-11 values. `pfs_guard` is the only one since
re-measured (14.92 -> 0.66, §0.2); the rest have NOT been re-measured because
`RTX_PERF_MARKS` (§8) now defaults off and reads them all as zero.** Treat every row below
as an 08-11 figure carrying ~1.6 ms of marks overhead spread across it, not as current:

| stage | ms | note |
|---|---|---|
| `pfs_guard` | ~~**14.92**~~ **0.66** | **REALISED, see §0.2.** Now `captureDrawSnapshot` at 3.80 ms measured by its own unconditional timers (§1a) |
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

### dxvk-cs composition — CURRENT, 2026-08-14 `[M]`

```
commitGeometryToRT              20.16
  submitDrawState               18.32
    processDrawCallState        16.72
      processSceneObject        10.58
        updateInstance / find / mid / addInstance   ALL n/a -- perfSceneObjSplit=False  [U]
      geom                       4.82
  finalize (worker join)         0.87
the once-per-frame fat chunk    12.29
  prepareSceneData              10.23   (nested -- do NOT add to the line above, see below)
    merge                        7.57   (loop 3.65, buildBlases 2.32, dynBlas 1.30, setup 0.25)
    surfMat 1.08  gc 1.06  setup1+instSetup 0.40  tlas+accel+light 0.12
```

`merge` is nested inside `prepareSceneData`, which is nested inside the fat chunk. Adding them
double-counts a parent and its own child — a mistake this document made once and had to retract
(§0.2). The chunk is **12.29 ms**, and the report's own cross-validation agrees:
`PASS dxvk-cs PrepScene vs CsSplit fat chunk 10.19 vs 12.35`.

**`processSceneObject` 10.58 ms is now the largest single item on either thread that has no
attribution at all.** `rtx.perfSceneObjSplit=False`, so `updateInstance`, `findSimilarInstance`,
`mid` and `addInstance` all read `n/a`. Phase 1.3 is what makes Phase 2's remaining half sizable.

Historical, 2026-08-11, kept because §6's dxvk-cs arithmetic is written against it:

```
commitGeometryToRT              26.39   (processSceneObject 12.78, geom 5.98)
        ui_fastRet   3.30   ui_entry  2.20   ui_tail 0.65   ui_xform 0.57
        ui_flags     0.28   ui_surf   0.24   ui_rest 0.35
        find / mid / addInstance    ~5.2 residual, NOT MEASURED  [U]
the once-per-frame fat chunk    15.36  (prepareSceneData 14.75, merge 11.25)
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

### Phase 0 — ~~free, today. -2.6 ms on the pole~~ **VOID. REFUTED 2026-08-14 `[M]`.**

> The original text: *"Delete diagnostics sitting on the frame thread: `te_census` 1.01,
> `te_camDiag` 0.83, `te_propId` 0.37, fanout `dbgTrack` 0.39. The report flags them itself as
> '(diagnostics?)'. No architecture, no risk."*
>
> **None of the four is a diagnostic, and the -2.6 ms does not exist.** Checked in source before
> gating anything off:
>
> | leaf | ms | what it actually is |
> |---|---|---|
> | `te_camDiag` | 1.07 | `transformData.sanitize()` (the NaN guard) + the live canonical `worldToView` override (`d3d11_rtx.cpp:47571-47597`). Its diagnostic half is **already gated** behind `RTX_D3D11_DIAG`. |
> | `te_census` | 1.26 | **46% is `objAabb`** — `objAabb_ns=28,068,000` of `te_census=60,774,200` — the object-space bounding-box producer, already coupled to its consumer by `if (RtxOptions::needsMeshBoundingBox())` (`:48024`). The census map is already a gated pointer and the 64-sample world-AABB loop is already `if (tlasE != nullptr)` (`:47920`). |
> | `te_propId` | 0.47 | the `stablePropId` producer — the fix for the black-square corruption. |
> | `dbgTrack` | 0.46 | **the name is a fossil.** `markInst(instDbgTrackNs)` at `:12937` closes over `:12911-12937`, which is `acquirePooledTformVec()` x2 + `reserve(maxInstances)` x2. The diagnostic cluster its own comment describes was already removed. |
>
> **`RTX_D3D11_DIAG` is off** — `[PhantomProbe]`, `[TLASEntry]`, `[MainCamPose]` and `[StudioDump]`
> each emitted **0 lines** across the whole log. The pure diagnostics were already dark, which is
> why these buckets are the size they are: what is left in them is the functional work.
>
> **Where the -2.6 ms came from.** The `[Perf.Report]` verdict row is labelled
> `te_census + te_camDiag (diagnostics?)` — **with a question mark**. The report generator is
> guessing from the leaf's *name*; this plan hardened the guess into "free, today, no risk".
> Acting on it would have deleted a NaN guard, a camera override, a bounding-box producer and a
> corruption fix.
>
> **This is §0.2's lesson with the sign flipped.** There: *a leaf being large does not make it
> irreducible.* Here: **a leaf being named "diag" does not make it removable.** Same fix in both
> directions — cross-tab the leaf against what consumes it before believing the label.

**What was actually free, and is now done `[M]`:** `rtx.perfDrawRedundancy` was left `True`
through the 00:15 frame-time capture (378 `[DrawRedund]` lines in that log). §8 already said
*"it is real time on the pole, so take the redundancy window and the frame-time window separately"*
and the conf's own comment says *"set this back to False before quoting any fps number"* — it was
not. Its question is already answered (§3.0). **Set `False` 2026-08-14.** Every absolute ms in the
00:15 report carries its unbilled per-draw cost, on top of the `clkNs` problem below.

**Two flags in the conf are inert and were mistaken for live `[M]`:** `rtx.memoExtractTransforms`
and `rtx.memoExtractVerify` are both `True`, but `memoOn` requires `!splitHandled`
(`d3d11_rtx.cpp:40370`) and `splitHandled` is set for every draw the split cache claims (`:37924`)
— which `[Perf.SplitXf]` reports as 1309/frame, i.e. all of them. `[Perf.MemoXf]` has printed
**zero lines** in the whole log, and it is not the log filter (`"[Perf.MemoXf"` is not in
`kFilteredTags`). **Phase 3.2b is not running**, verify is not double-deriving, and neither flag
explains anything in `bt_extractXf`. Testing the memo requires turning `splitTransformCache` off
first.

**One decision, not free, left open:** `needsMeshBoundingBox()` is
`objectAntiCulling || lightAntiCulling || terrainBaking || alwaysCalculateAABB || NeeCachePass::enable()`
(`rtx_options.cpp:653-658`). The conf turns off `antiCulling.object.enable` and
`enableAlwaysCalculateAABB` — but **`rtx.neeCache.enable` defaults `true` and is not overridden**,
and `pt_neeCache 0.42 ms` confirms the pass runs. So those two options buy nothing while NeeCache
alone keeps `objAabb`'s ~0.58 ms/frame producer alive. Disabling NeeCache would recover it, but
that is a **quality** change and the GPU is 75% idle — recommend leaving it on. Noted so the
coupling is not rediscovered as a mystery.

### Phase 1 — MEASURE. **BLOCKING. Do not skip.**
Three unknowns decide whether 30 fps is reachable at all:

1. ~~**`pfs_guard` sub-buckets**~~ — **RESOLVED 2026-08-11 22:35 `[M]`. The hypothesis was right,
   and it is the bad answer.** From `[Perf.SubmitDraw.acc]` at 22:35:09.734, same window, ns:

   ```
   pfs_guard        = 960,495,300
   drawSnap_ns      = 916,767,200   <- 95.4% of pfs_guard IS captureDrawSnapshot
     drawSnapId_ns  = 146,741,900   <- 16.0% of capture
     drawSnapAlloc  =   7,421,600   <-  0.8%
     (remainder)    = 762,603,700   <- 83.2%, the named-cbuffer-span copy
   ```

   Scaled to that report's `pfs_guard` 13.77 ms/frame: **capture ~13.1 ms/frame**, of which span
   copy ~10.9, identity/stream/viewport copies ~2.1, arena alloc ~0.11.

   **Consequence: the capture floor is ~13 ms, not small.** This is exactly the failure case this
   item was written to detect. Phase 3 ("skip derivation for unchanged draws") cannot go below it,
   and §6's `-14.92 pfs_guard IF recoverable` line is now mostly **NOT** recoverable — capture is
   the mechanism Phase 3 runs on, not overhead sitting beside it. See §6 for the corrected sum.

   > **REFUTED 2026-08-14 `[M]`. `pfs_guard` is now 0.66 ms — see §0.2.**
   >
   > The reasoning above is sound and the conclusion was still wrong, in a way worth recording
   > because it is a shape this document has fallen into more than once: *"X is 95% of the cost,
   > therefore X is a floor"* skips the question **"is X doing work anyone reads?"** It was not.
   > The capture copied every span any path of a shader had ever read, onto every draw of that
   > shader, because the manifest is keyed per VS while the read is decided by the o2w path.
   > `[Perf.SplitXf.CbWaste]` measured 49% of captured bytes and 37% of write-combined
   > transactions as dead. Keying the capture per path took `wcMiss/draw` 1.702 -> 0.433 and
   > `pfs_guard` 14.92 ms -> 0.66 ms, and it did not remove anything Phase 3 runs on — the
   > record still serves every consumer, it just stops copying what no consumer asks for.
   >
   > The general lesson: **a leaf being large does not make it irreducible. Cross-tab what it
   > produces against what is consumed before calling it a floor.** The probe that settles that
   > question is cheap; the assumption cost this plan a phase.

   Two cheap leads fall out, neither of which needs new instrumentation:
   - `drawSnapId_ns` ~2.1 ms/frame copies identity/stream/viewport fields that the source comment
     at `d3d11_rtx.cpp:9539` says **no consumer reads yet**. Verify, then stop copying them.
   - the 10.9 ms span copy is poorly coalesced: `[DrawSnap] wcMiss=52867 wcHit=56268 (51%)`,
     with `cb3=44513` of the misses. The manifest is the lever, not the capture code.

   The timers that answered this (`drawSnap_ns` / `drawSnapAlloc_ns` / `drawSnapId_ns`) were added
   2026-08-10 at `d3d11_rtx.cpp:9519` and were already live in the log. Nothing needed building.
   They only print while the leaf split is on, which is `RTX_D3D11_SUBMARK=1` in the environment —
   the source default is **off** (`d3d11_rtx.cpp:8774-8779`). Lose that env var and the whole named
   leaf table, `pfs_guard` included, goes dark.
2. **Draw-level redundancy** — hash `DrawSnapshot` frame to frame and report the % identical.
   Instance-level is measured at 97%; draw-level is `[I]` and everything in Phase 3 depends on it.

   **PROBE BUILT 2026-08-11. `rtx.perfDrawRedundancy=True` → `[DrawRedund]`, every 3 s.**
   Nothing is measured yet — this is the instrument, not the answer. Turn it on with
   `useDrawSnapshot=True` (it has nothing to hash otherwise), take a window, turn it off before
   any frame-time number. Code: `D3D11Rtx::noteDrawRedundancy`, called from `SubmitDraw` right
   after the capture, outside `drawSnap_ns` and with `tStg` re-stamped so it is billed to no
   bucket — `pfs_guard` and every leaf around it stay comparable with prior logs.

   **FIRST RUN 2026-08-12 00:06 — the probe was WRONG, not the scene. `[M]`**
   49 windows, ~1,360 draws/frame, and every field read zero:
   `pos{in=74832 ident=0% same=0%} key{hit=0} cheap{ok=0 WRONG=0 miss=0}`.
   The same log window says `static=97% REDUNDANT=97%`, `matNew=0`, `uniqueBlas` flat, and a
   healthy record (`[DrawSnap] resolved=79% ovf=0`, `[DrawPure] pure=98% geoStatic=96%`), so a
   0% draw-identity match is not credible. `pos{in}` ≈ `draws` proves the frame-to-frame
   bookkeeping works — last frame's rows are found. **One folded `ident` hash contained a
   per-frame nonce, and a fold can only ever say "something moved":** the nonce took the whole
   measurement to zero and hid every other answer behind itself. Same defect class as the
   carrier fold that became `carrierGrp[kSdepGroupCount]`, for the same reason.

   **Probe rewritten 2026-08-12 — SIX identity components, tallied separately**, so a lone
   noisy field is named instead of silencing the other five: `sh` (shaders / layout / blend /
   depth / raster / topology), `rtv` (bound RTV — split out because an N-buffer swapchain makes
   it a period-N nonce unrelated to the draw), `dp` (draw params + record flags), `vb`
   (vertex+index bindings), `srv` (mask + SRV pointers), `vp` (viewports); plus `bind` (cbuffer
   bindings **and `contentGen`**), `bytes` (captured cbuffer bytes), `geo` (t31 + COLOR1).

   Two independent readouts, and a raw dump:
   - **`pos{...}`** — the Nth capture of consecutive frames, per component. No key, no
     threshold; **read this first.** A component near 0% while its neighbours are high *is* the
     nonce.
   - **`key{...}`** — keyed on the *stable subset* `sh+dp+vb` + occurrence ordinal, with the
     three excluded components reported over the hits. Keying on the full identity resolved zero
     times, which is why it was narrowed. A keyed hit is **not** proof of "same draw" — only of
     same mesh, shader and range. Judge it against `pos`.
   - **`[DrawRedund.Raw]`** — up to 6 lines/window, fired only where `sh` matched, printing the
     actual before→after values. A percentage cannot distinguish "differs" from "alternates
     between two back buffers", and those want different fixes.

   **SECOND RUN 2026-08-12 00:22 — the split worked and it settles §6. `[M]`**

   ```
   posIn=27081 pos{sh=69% rtv=97% dp=58% vb=42% srv=0% vp=98%
                   bind=0% bytes=20% geo=84% ALL=0%}
   key{hit=16979 (62% of draws) rtv=100% srv=0% vp=100%
       bind=0% bytes=15% geo=85% SAME=0}
   cheap{ok=0 WRONG=0 miss=5508}
   ```

   Three results, in descending order of consequence.

   **(a) THE CHEAP TEST IS DEAD, and this revokes §6's last escape route.**
   `cheap{ok=0 WRONG=0}` on every window. `WRONG=0` means `contentGen` is *sound* — it never
   says "same" when the bytes differ. `ok=0` means it **never fires**: TF2 `Map(WRITE_DISCARD)`s
   its constant buffers every frame as a matter of course, so the generation bumps whether or
   not a byte changed. **It is a rename detector, not a change detector.** Therefore any Phase 3
   skip must compare actual bytes → must copy them → **cannot go below the ~13 ms capture
   floor.** §6's "either draw-level redundancy turns out to skip the capture itself…" is
   answered: it does not. Do not spend more time looking for a pre-test on generations.

   **(b) `srv=0%` was a per-frame GLOBAL, not draw churn.** `[DrawRedund.Raw]` named it: slot 30,
   the bone-matrix SRV, whose *object pointer* TF2 rotates each frame (~7 pointers cycling) and
   binds on essentially every draw — every draw in a frame shows the identical `old->new`
   transition. It carries zero per-draw information and took the whole component to zero.
   `vb0`/`ib`/`rtv`/`vp`/`cnt`/`start`/`base` are byte-identical frame to frame: **the geometry
   bindings are stable.**

   **(c) `bytes=15-20%` vs `geo=85%` — the fold mistake, one level down.** `geo` is pure object
   data (t31/COLOR1) and holds still on 85% of draws; `bytes` folds all ~4.9 captured spans of a
   draw together, so **one view-dependent span poisons the answer for a draw whose object data
   never moved.** Since the camera is in nearly every draw's span set, "is this draw unchanged"
   was being answered by the camera every time.

   > **This is the structural finding, and it reframes Phase 3.** Our record conflates
   > view-dependent and object-dependent inputs. UE's `FMeshDrawCommand` is camera-INDEPENDENT by
   > construction — view state is applied once per view, not baked per draw — which is *why* it
   > can be cached at `AddToScene` time and ours cannot be cached at all. If the per-slot split
   > shows cb3 (object-to-world) static while cb2 (proj/view) moves, then Phase 3 is not "skip
   > unchanged draws" but **"cache the object-space half, recompute the view half"** — a
   > different mechanism with a different and much larger ceiling.

   **Probe extended again 2026-08-12** to answer exactly that: `bytes` is now tallied **per
   cbuffer slot** (`bytesSlot{cb2=… cb3=…}`), `bind` is split into `bindPtr` (buffer identity +
   offsets) vs `bindGen` (generations only), `srv` gains `srvNS` (the same question with the two
   stream slots t30/t31 held out), and a second cross-tab `ptr{ok/WRONG/miss}` asks whether
   binding identity alone predicts the bytes now that generations are known useless.

   **THE WITHIN-FRAME QUESTION — added 2026-08-12, and it is the one that matters most.**

   Everything above is a CROSS-FRAME question, and every cross-frame answer is hostage to the
   scene standing still — it collapses in combat, which is where frame rate matters most. The
   probe now also reports `perFrameDistinct{cb2=… cb3=… wholeDraw=…}`: across the ~1,360 draws
   of **one** frame, how many *distinct* byte values does each cbuffer slot take?

   Expected shape if the view/object split is real:

   | slot | content | expected distinct / frame |
   |---|---|---|
   | cb3 | object-to-world | ~one per object — genuinely per-draw |
   | cb2 | projection / view | **a handful** — main view, viewmodel, 3D-skybox sub-view, sky probe |

   **If cb2 reads ~3 against 1,360 draws, then 1,357 of those captures copied bytes the frame had
   already produced.** That is redundant work *inside a single frame* — it needs no cache, no
   cross-frame identity, and no static-scene assumption to remove. It is precisely UE applying
   view state once per view instead of baking it into every `FMeshDrawCommand`, and unlike
   everything else in this plan **it holds under combat load.**

   **The data model is already shaped for this.** `DrawCallTransforms` (`rtx_types.h:848`) already
   keeps `objectToWorld` apart from `worldToView` / `viewToProjection`, and `m_lastGoodTransforms`
   is already a frame-level latch for the view pair (it is `kSdepStatic` in the carrier census,
   measured at ~65 moves/window — i.e. it already almost never changes per draw). **What is
   per-draw is the DERIVATION, not the data.** So the restructure is:

   - **per view, once per frame** — resolve `worldToView` / `viewToProjection` for each distinct
     camera. `perFrameDistinct{cb2}` is literally the count of how many that is.
   - **per draw** — derive `objectToWorld` from cb3 only.
   - **per draw, trivial** — `objectToView = worldToView * objectToWorld`, one matrix multiply.

   That is `FMeshDrawCommand` + view state, reached by narrowing what the per-draw path reads
   rather than by rewriting the pipeline. **Do not start it before the number is in hand** — the
   whole point of Phase 1 is that this plan has already been wrong twice about which half is
   expensive, and `[DrawSnap] wcMissSlots{cb2=1157 cb3=21708}` is a live hint that the *capture*
   cost is concentrated in the object half, not the view half. If so the win is on the
   **derivation** (`bt_extractXf`) rather than on the capture, which is a smaller but still real
   prize, and the honest version of this section must say which.

   **Read it in this order, and stop at the first one that fails:**

   | field | what it decides |
   |---|---|
   | **`bytesSlot{cb2= cb3=}`** | **THE QUESTION NOW.** cb3 high + cb2 low ⇒ object data is static and the churn is the camera ⇒ Phase 3 becomes "cache the object-space half". cb3 also low ⇒ the object data genuinely moves and there is nothing to cache. |
   | `bindGen=` vs `bindPtr=` | expected ~0% and high respectively. Confirms `contentGen` is a rename detector; if `bindPtr` is *also* low the bindings really do churn and (b) above is wrong. |
   | `ptr{WRONG=}` | whether binding identity alone can stand in for the bytes. Expected large and nonzero (the game rewrites the same buffer) — which would close the last route to skipping the copy. |
   | `srv=` vs `srvNS=` | `srv` ~0 with `srvNS` high confirms the whole SRV churn is the two stream slots and nothing per-draw. |
   | `key{hit=}` | whether a cross-frame key exists **at all**. Reads 62%; `sh=69%` positionally, so keyed is the sounder mechanism. |
   | `pos{ALL=}` / `key{SAME=}` | the ceiling on skipping the *derivation* (capture is paid regardless — see (a)). Bounded by `bt_extractXf` + the unnamed hook residual. |
   | `sameNoSrv` | the sound floor — see the caveat below. |

   The positional ordinal is counted by the probe, **not** `DrawSnapshot::drawIndex` — that is
   `m_drawCallID`, which only advances for draws surviving the filters, so several captures
   share one value.

   **`key{SAME=}` IS A CEILING, NOT A POPULATION.** The hash covers what `DrawSnapshot` covers,
   so it is blind to bone palettes, vertex/index buffer contents and PS state. That is the exact
   blindness that shipped the viewmodel flicker under `filterDupSameDraws` v1 (see the box in
   Phase 3). `sameNoSrv` — redundant draws with no VS SRV bound at all, hence no bone or
   model-instance stream — is the part that is sound without further work. The gap between them
   is the work Phase 3 has to do before it can claim the ceiling.

3. **`rtx.perfSceneObjSplit`** — attributes the ~5.2 ms of find/mid/add. Note: the old handoff
   names `rtx.perfUpdateInstSplit` for this; that is the **wrong switch** — it splits the `update`
   quarter, which is already attributed. `perfSceneObjSplit` is declared at `rtx_options.h:675`.

### Phase 2 — push-not-poll on dxvk-cs. **-7 to -10 ms on dxvk-cs. PARTIALLY BUILT 2026-08-14.**
Persistent per-instance records; the draw side marks dirty when `xfChg`/`matChg` fire (1% / 0%).
Per-frame loop iterates the dirty set only. Unchanged instances get a **bulk frame-id stamp**
instead of a call. Kills most of `entry` 2.73 + `fastRet` 4.15, and should reach into
find/mid/add once Phase 1.3 has attributed it.

Feasibility note: identity is already exact-keyed with `created=0`, so this is a read-mostly map
plus a dirty list — comparable to Phase 3 in difficulty, **not** the "structurally hostile"
problem earlier drafts of this plan claimed. Steady state has `created=0`; the creation path
during streaming and level transitions still needs handling.

#### 2.1 WHAT WAS BUILT `[M]` — `rtx.pushInstanceRecords`, `[Perf.PushInst]`

**The keep-alive rule was read out of the source rather than assumed, and it is the whole design
constraint.** `InstanceManager::garbageCollection` reaps on

```
clauseLifetime = (force||enable) && (m_frameLastUpdated + instanceKeepN <= currentFrame)
```

and **nothing else**. So §4's "skip must mean keep alive without reprocessing" is expressible
exactly as a frame-id stamp, with no change to GC at all.

Built on the **fanout path**, because that is where the visits are: 1,349 draws produce ~15.5k
instance visits, since a fanout draw carries a placement array. The batch is therefore the unit —
one fingerprint decides every placement in it.

| piece | where |
|---|---|
| `FanoutBatchRecord` (input fingerprint + resolved instance list + frame stamps) | `rtx_instance_manager.h` |
| `fanoutRecordKey` / `fanoutRecordFingerprint` | `rtx_instance_manager.cpp` |
| skip = bulk `setFrameLastUpdated` + `registerCamera` per recorded instance | `processSceneObjectFanout` |
| O(1) total invalidation via `RtInstance::m_batchRecordKey` | `removeInstance` |
| `rtx.pushInstanceRecords`, `…Verify` (default **true**), `…MaxBatches` | `rtx_options.h` |

Two traps handled rather than assumed away: `setFrameLastUpdated` **clears `m_seenCameraTypes`**
on the first stamp of a frame, so the skip path must re-register the draw's camera; and a batch
whose loop rejected any placement is never recorded, because those rejections are decided by
per-placement state the fingerprint does not claim to cover.

#### 2.2 FIRST VERIFY RUN 2026-08-14 — `FAIL=3367`. The gate did its job. `[M]`

```
[Perf.PushInst] verify=1 batches=460 hit=0 servedInst=0
                miss{key=4 input=398 invalid=0} records=768 evict=0 FAIL=3367
```

**`builtFrame == currentFrame` on 100% of the FAIL lines**, with the same key repeating
consecutively and `recorded == actual` size diverging at index 0.

**The defect: several draws per frame share one (vertex shader, geometry) identity, and they are
not interchangeable.** Draw 1 stored the record; draw 2 found it valid with a matching
fingerprint and predicted a hit — but the two resolve to *different* instances, because
resolution is stateful within a frame (draw 1's instances are already claimed, so
`findSimilarInstance` hands draw 2 a fresh set). The record was describing the previous **draw**,
not the previous **frame**. The same defect produced the 86% `input` miss rate: every same-key
draw clobbered the record, so the next frame compared against whichever draw happened to be last.
One cause, both symptoms.

**FIXED 2026-08-14, UNVERIFIED:** an **occurrence ordinal** folded into the key, so the Nth
occurrence of a batch identity within a frame gets its own record — the same device
`[DrawRedund]`'s keyed probe needed, and counted the same way (in its own map, never from a draw
index, because those only advance for draws surviving the filters). Plus an independent
`frameLastBuilt == currentFrame` backstop reported as `miss{sameFrame=}`, which must read 0.

Also fixed: `hit=` counts *skips*, which are zero by construction under verify — so a working
record and a broken one printed the same thing. The line now carries `predict=`/`predictInst=`,
which is the ceiling and the per-INSTANCE number that maps onto `entry` 2.73 + `fastRet` 4.15.

**Frame time was unchanged across that run** (40.87 -> 40.56, both threads slightly *up*), which
is exactly what verify predicts: it hashes and compares while skipping nothing. `servedInst=0`.

#### 2.3 WHAT PHASE 2 STILL OWES `[U]`

1. **The fingerprint is O(placements), so this is PULL, not PUSH.** It hashes every placement
   matrix of every batch every frame — ~15.5k x 64 bytes ~= 1 MB/frame of XXH64. Far cheaper than
   a find + update per instance, but it does **not** achieve this plan's stated "the 97% are never
   enumerated". The push form needs the change signal plumbed from the draw side, which is where
   `xfChg`/`matChg` already come from. Size it after 2.2 verifies, not before.
2. **The single-placement path has no record.** `processSceneObject` — ~1,349 draws/frame still
   poll. Small beside fanout's ~15.5k instance visits, which is why fanout went first.
3. **find/mid/add is still unattributed** (§1's dxvk-cs block, `perfSceneObjSplit=False`). The
   claim that Phase 2 "should reach into find/mid/add" has no number behind it until Phase 1.3
   runs.
4. **The creation path** during streaming and level transitions is correct but unaccelerated — a
   changed fingerprint simply falls through to the loop.

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

### Phase 3 — RESTATED 2026-08-12 on measured data. **WITHIN-frame, not cross-frame.**

> **The original Phase 3 was "if a draw's captured inputs are byte-identical to LAST FRAME's, do
> not derive it". Phase 1.2 measured that and it is the weaker half of the truth.** Cross-frame
> whole-draw agreement is 19% (`bytes`), and it is 19% only because the fold drags the object
> half down to the view half — per slot it is cb3 81-87% against cb2 17%. Worse, every
> cross-frame number is hostage to the scene standing still and collapses in combat.
>
> **The within-frame number does not have that weakness, and it is much larger.**

#### 3.0 THE MEASUREMENT `[M]` — `[DrawRedund]`, 2026-08-12 00:41

```
drawsPerFrame = 1367
perFrameDistinct{ cb0=4  cb1=2  cb2=10  cb3=215  wholeDraw=254 }
bytesSlot (cross-frame){ cb0=93%  cb1=92%  cb2=17%  cb3=81-87% }
cheap{ok=0 WRONG=0}          ptr{ok=6604 WRONG=20288 miss=1468}
```

Read across one frame, ~1,367 draws:

| slot | content | distinct/frame | reading |
|---|---|---|---|
| cb2 | projection / view | **10** | not per-draw data at all — main view, viewmodel, 3D-skybox sub-view, sky probe, a few more. ~1,357 of 1,367 captures reproduce a value the frame already holds. |
| cb3 | object-to-world | **215** | the object half, and even it is 6.4:1 redundant. |
| — | whole draw input set | **254** | **the frame contains 254 unique draw inputs and we capture and derive 1,367. 5.4x, inside one frame, with no static-scene assumption.** |

#### 3.1 BOTH CHEAP PRE-TESTS ARE MEASURED DEAD

- `contentGen` — `cheap{ok=0 WRONG=0}` on every window. **Sound but never fires:**
  `Map(WRITE_DISCARD)` bumps the generation every frame regardless of content. Rename detector,
  not change detector.
- binding identity — `ptr{ok=6604 WRONG=20288}`. Predicts the bytes 25% of the time, wrong 3x
  more often than right. The game rewrites the same buffer with new contents; that is the
  expected case, not an anomaly.

**Therefore any correctness-preserving skip must compare actual BYTES.** Do not spend more time
looking for a shortcut around that — two candidates have now been measured and both are dead.

#### 3.2 WHAT TO BUILD

The data model is already shaped for this. `DrawCallTransforms` (`rtx_types.h:848`) keeps
`objectToWorld` apart from `worldToView` / `viewToProjection`, and `m_lastGoodTransforms` is
already a frame-level latch for the view pair (`kSdepStatic`, ~65 moves/window). **What is
per-draw is the DERIVATION, not the data.**

- **3.2a — resolve the view once per view, not once per draw.** `perFrameDistinct{cb2}` is
  literally the count of distinct views: 10. Derive `worldToView`/`viewToProjection` per view id,
  then per draw do only `objectToView = worldToView * objectToWorld` — one matrix multiply.
  This is UE applying view state once per view instead of baking it into every
  `FMeshDrawCommand`.
- **3.2b — memoise the derivation within the frame, keyed on the input bytes.** 254 distinct
  input sets against 1,367 draws is an **81% cut on `bt_extractXf`** and it holds in combat.
  The natural key is `(viewId, objectKey)` where `objectKey` covers cb3's 48-byte o2w block —
  far cheaper to compute than the full span set, which matters because the key must be produced
  *before* the work it is trying to avoid.

**Capture cost is NOT where the prize is, and the earlier draft of this section implied it was.**
`[DrawSnap] wcMissSlots{cb2=16467 cb3=36045}` over 45,761 draws: cb3 is 2.2x cb2, so
deduplicating the view spans recovers roughly a third of the span copy. The prize is the
**derivation**, via 3.2b.

#### 3.3 THE REPLAY TIER'S GATE — ANSWERED 2026-08-12 `[M]`. It was the gate.

**3.2b is the deleted replay tier's idea** (§Phase 7, Appendix A.9), which shipped **0 hits in
6,542,782 draws at 99.94% INELIGIBLE**. That looked like a reason to be careful. It is not: the
tier gated on **ROUTE**, and a route gate is not a duplicate test.

`rr.ineligBits` (`d3d11_rtx.cpp:23151-23203`) is seven rules — `1` fallback/UI, `2` bone,
`4` unsettled axis votes, `8` instanced/fanout, `16` projSlot, `32` viewOk, `64` o2wPath — and
`eligible` requires **all seven clean**. It asks "did this draw take one of a small set of
blessed code paths", never "are these inputs a duplicate". With TF2 binding the bone SRV on
essentially every draw and the fanout path carrying the hulls, near-total rejection is what that
conjunction *should* produce. **The tier did not fail to find duplicates. It never looked.**

**And the gate it needed now exists.** `safeToDefer()` — the three-axis DrawSnapshot purity gate,
built 2026-08-10, *after* the tier was written and never retrofitted to it. Live from
`[Perf.SubmitDraw.acc]`, same window as the `[DrawRedund]` capture above:

```
bt_extractXf = 698,085,600 ns
xfElig       = 474,838,900 ns   xfEligN   = 33,257     -> 14,278 ns/draw
xfInelig     = 223,246,700 ns   xfIneligN = 13,811     -> 16,165 ns/draw
xfElig3      = 474,838,900 ns   xfElig3N  = 33,257     -> elig3 == elig exactly
```

**70.7% of draws pass, and they carry 68% of all `bt_extractXf` time.** The accumulator's own
comment warns that a big bucket cannot distinguish "many cheap draws" from "few expensive ones"
and that the ns/draw is the answer — so: eligible draws cost **14.3 us** against ineligible
**16.2 us**, i.e. **88% as expensive each**. They are not the cheap tail. This is the world that
comment says makes the work worth doing.

Two gates, same function, measured on the same build:

| gate | admits | asks |
|---|---|---|
| replay tier `ineligBits` (7 route rules, ANDed) | **0.06%** | did this draw take a blessed route |
| `safeToDefer()` (3 purity axes) | **70.7%**, carrying 68% of the time | is the derivation a pure function of the record |

**So 3.2b is not the replay tier rebuilt. It is the memo the tier could not have: keyed on
CONTENT, gated on PURITY.** Do not reuse `ineligBits`.

#### 3.3a SIZING IT HONESTLY — and one limit of the 254 figure

`wholeDraw=254` counts distinct **cbuffer-content** sets, not distinct draws: it hashes
`hBytes` only, so two draws with the same object-to-world and camera but *different meshes*
collide in that number.

**For `ExtractTransforms` specifically that is the correct denominator**, because its output
(`DrawCallTransforms`) is a function of the cbuffer content and route state, not of which mesh is
bound. It is the wrong denominator for anything mesh-dependent — material fill, geometry hashing,
bbox — and those are already in the frame-end batch (A.5) anyway.

So the arithmetic, stated with its assumptions visible:

```
bt_extractXf (clean machine, [Perf.Report] 2026-08-12 00:07)      9.11 ms/frame
  x 0.707   passes safeToDefer(), carrying 68% of the time     ~= 6.2  ms
  x 0.81    within-frame duplicate share (254 distinct / 1367) ~= 5.0  ms/frame recoverable
```

**~5 ms/frame on the pole, and it does NOT need a static scene** — that is the whole point of
using the within-frame number. Validate against a combat frame before believing it holds, but
unlike the cross-frame form there is no reason in the mechanism why it should not.

#### 3.3b BUILT 2026-08-12 — `rtx.memoExtractTransforms` / `rtx.memoExtractVerify`

`[Perf.MemoXf]`. The memo sits at the single `dcs.transformData = ExtractTransforms()` call site
(`d3d11_rtx.cpp:~34500`), between the `bt_geoCopy` and `bt_extractXf` marks, so the existing leaf
timings keep their meaning.

- **Key** — `D3D11Rtx::drawMemoKey`. Hashes the whole record: shader/layout/state identity,
  topology, rtv0, all VB/IB/SRV bindings, viewports, cbuffer binding identity, the captured
  cbuffer **bytes**, the t31/COLOR1 entry, **the five carrier-group fingerprints**, and
  `m_currentInstanceIndex`. The carrier fingerprints are what make it sound rather than plausible:
  `ExtractTransforms` *reads* cross-draw state, so two draws with identical D3D11 inputs still
  derive differently if something moved a carrier between them — folding `carrierGrp` in turns
  that into a key change, which costs a hit rather than correctness.
  `contentGen` is deliberately **not** hashed (it bumps every frame, so it would make every key
  unique and the memo would never fire); the bytes subsume it.
- **Scope** — cleared at frame rollover. Within-frame only, which is what makes the key simple
  enough to be complete: the bone-SRV pointer that `[DrawRedund]` showed rotating every frame is
  a *constant* inside the map's lifetime.
- **Store** — on the way **out**, requiring `s_xtEligLast` (`safeToDefer()`). That verdict is a
  statement about what the derivation *did*, so it cannot be consulted beforehand — and that is
  the correct order anyway: a stored entry has a proven-pure, carrier-free producer.
- **Refusals** — any draw whose transforms carry `instancesToObject` / `prevInstancesToObject`
  (or their `shared_ptr` owners) or `isFanoutBatch` is **never stored**. Memoising those would
  serve one draw's per-instance array to another. Counted as `miss{instPtr=}`, not silently
  dropped.
- **One invariant this change breaks, and repairs.** `s_xtEligLast`'s declaration states it is
  never cleared between draws because "the guard is RAII and writes both on every exit, so a
  stale value cannot outlive a draw that ran the derivation". The memo is the **first code that
  can skip the derivation entirely**, so the hit path republishes the verdict explicitly;
  otherwise a memoised draw would bill its `xfNs` to whatever the previous draw was.

**RUN WITH `memoExtractVerify=True` FIRST.** It derives anyway on every hit and bit-compares 17
POD fields, logging `[Perf.MemoXf.FAIL]` with the draw's VS hash and the diverging path ids.
**`FAIL` must be 0 over a real session before the verify flag comes off.** `FAIL>0` means the key
is incomplete — the derivation read an input `drawMemoKey` does not cover — and the failure mode
is a silently wrong transform, i.e. an object in the wrong place. Frame time under verify is
meaningless by construction (two derivations per hit).

#### 3.3c FIRST VERIFY RUN 2026-08-12 01:12 `[M]` — key sound, key too WIDE

```
[Perf.MemoXf] draws=46013 hit=822 (1%) store=31882
              miss{unsafe=13309 (cb=748 carrier=11458 geo=1596) instPtr=0 noSnap=0}
              FAIL=0        (41 windows, ~45k draws each, zero failures)
```

- **`FAIL=0`** — the key is COMPLETE. No draw's derivation read an input `drawMemoKey` misses.
  The verify path works and is the thing that makes the rest of this safe to iterate on.
- **`instPtr=0`** — the ordering tripwire is quiet, as predicted (see 3.3b).
- **`store=69%`** — matches `safeToDefer()`'s measured 70.7%. The gate behaves.
- **`hit=1%` — the key was too WIDE, and §3.3a said so before it was built.** The first version
  hashed the vertex and index buffer bindings, making the key *mesh* identity. There are far more
  meshes per frame than transforms, so every mesh forked the key and the memo degenerated to
  ~unique entries — storing 31,882 to serve 822.

**NARROWED 2026-08-12.** `DrawCallTransforms` is a function of cbuffer content and route state,
not of which mesh is bound — that is exactly why `wholeDraw=254` is the right denominator.

| removed from key | basis | risk |
|---|---|---|
| index buffer | the tree already states `ExtractTransforms` never reads it (`geoBytesStatic` comment: the max-index scan is in `bt_cullVtx`, a different stage) | none |
| vertex buffers | its 5 converted reads exist to produce the COLOR1 per-instance semantic, and that RESULT is hashed as `instSem` — the binding is an input to something already in the key | **judgement call** |
| `rtv0` pointer → presence bit | both consumers only test "is anything bound" | none |

**The vertex-buffer removal is an argument, not a proof, and is not trusted as one.** That is what
verify is for: if a binding feeds the transforms by an uncovered route, `FAIL` goes non-zero and
names the draw. **Keep `memoExtractVerify=True` for the next run.** If FAIL stays 0, the narrowing
ships; if not, restore the VB loop — never weaken the comparison instead.

**SECOND NARROWING 2026-08-12 01:20.** With mesh bindings out: `FAIL=0` held (36 windows) but
`hit` only reached **4-5%**, ~756 distinct keys/frame against a 254 ceiling. Cause found in the
tree rather than guessed: the key hashed all five carrier groups, and `[DrawPure]` sizes
`kSdepCbLoc` at **~5,600 moves/window against cam ~780 and static ~65 — 88% of all carrier
movement**. Every move invalidates every key taken before it, so a *cache* was shredding the memo.
Key now hashes `kSdepCam` + `kSdepStatic` only. `kSdepRoute` seals empty by construction and
`kSdepBone` measures zero, so both were dead weight; dropping `cbLoc` is sound **for the key**
because what it decides is which spans the capture takes — and the span descriptors and their
bytes are already hashed, by a more direct route than the cache's own fingerprint.
**This is a claim about the KEY, not about `safeToDefer()`** — the two reverted attempts to retire
cbLoc from the *carrier set* stay reverted and the gate still requires all five clean.

**NEXT LEVER, sized by this run: `carrier=11458` is 86% of the 13,309 refusals.** The other two
axes are noise (cb 748, geo 1596). If `safeToDefer`'s carrier requirement is over-strict *for a
memo specifically* — draw A wrote carrier X, draw B with an identical key would write the same X,
and anything that moved X between them changes `carrierGrp` and so changes the key — then
relaxing it reaches most of the remaining 29.3%. **Do not relax it on that reasoning.** Publish
the split (done, above), then test it behind its own flag with verify on, the same way this step
was done.

#### 3.4 THE ORIGINAL CROSS-FRAME FORM — keep, demoted

Still worth having for the static case (`cb3` 81-87% frame to frame), but it is now the
*secondary* mechanism: it needs the scene to hold still, and 3.2 does not.

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

### Phase 6 — Phase B purity: finish the half-built batch. **frame thread**
The batch is **built, running, and half-populated**: 1055 items in 6 ms on 30 workers, carrying
**four of the six** per-draw stages. Full anatomy in **Appendix A**. The two missing stages are
transforms (here) and instance/scene work (Phase 2b). Concrete steps:

- **6a. ~~Cherry-pick Phase B batch 1 from `stable`~~ — DONE, AND DO NOT DO IT. `[M]` 2026-08-14.**
  The conversions were **redone independently on this branch** and HEAD is now *ahead* of `stable`.
  Cherry-picking would drag in the XfOverlap worker that Phase 7 deletes, to gain nothing.

  | accessor | `4681ecca` | `stable` `470eaf4b` | **HEAD `5ced56c2`** |
  |---|---|---|---|
  | `drawVertexShaderCom` | 36 | 53 | **54** |
  | `drawPixelShaderCom` | 0 | 5 | **6** |
  | `drawIndexBuffer` | 0 | 13 | **13** |
  | `drawVsSrv` | 12 | 15 | **15** |
  | `drawCbSpan` | 18 | 18 | **19** |
  | `drawViewport*` | 12 | 12 | **27** |
  | `drawVertexBuffer` | 3 | 14 | 4 -> **13** (6f) |
  | `drawInputLayout` | 11 | 13 | 11 -> **13** (6f) |

  The two counts where HEAD looked "behind" were **deliberate, not a regression**:
  `d3d11_rtx.h:1841` records that 10 of the 22 `inputLayout` reads in the file are *outside*
  SubmitDraw, where the record would serve the PREVIOUS draw's bindings. `stable`'s broader pass
  is the reason to read a conversion before lifting it.
- **6b. ~~Batch 2 — identity reads~~ — DONE ON THIS BRANCH, and it never existed on `stable`.**
  `drawBlendState`, `drawDepthState`, `drawRasterState`, `drawPsSrv0to7`, `drawRtv0`, `drawVsCb`,
  `drawT31Entry` and `noteGeoLiveRead` are all live at `d3d11_rtx.h:1860-1869`.
- **6c. Batch 3 — content reads, the careful class.** Still outstanding. `GetMappedSlice` sites +
  `vs.constantBuffers` binding+content sites, routed through `drawCbSpan` / the manifest.
  **Never convert a binding without its bytes** (A.7).
- **6d. Add `ExtractTransforms` to the batch. ~~One more stage~~ — THE SIZING IS WRONG. `[M]`**
  This was written as "one more stage in `DrawWorkItem`, dispatch/join/emit unchanged". The code
  does not allow it. **`dcs.transformData` has ~170 read sites INSIDE SubmitDraw, after the
  derivation** (`d3d11_rtx.cpp:41599` through `:51760`; SubmitDraw runs to `:52039`), and three of
  them *write back into it* — `objectToWorld` at `:41599`, `worldToView`/`viewToProjection` at
  `:42329`, `instancesToObject` at `:43040`. Sampled: sub-view classification, sky-probe gating,
  capture keys, viewmodel. Functional, not diagnostics.

  **The criterion the four batched stages actually meet is not purity — it is having NO
  in-SubmitDraw consumer.** `materialData`, `geometryData.hashes`, `geometryData.boundingBox` and
  `skinningData.pBoneMatrices` are read only on the CS thread. That is why they were easy.
  Deferring transforms means deferring the ~11,000 lines of SubmitDraw that follow it.
- **6e. THE BATCH JOIN IS 6-7 ms OF THE POLE AND NOBODY HAS LOOKED AT IT. `[M]` NEW 2026-08-14.**
  `flushGeometryBatch` (`d3d11_rtx.cpp:29319`) times `tBatch0 -> tBatch1` = dispatch + join **wall
  time on the frame thread**, called from `EndFrame`, so it lands in `between entry points` (§1a)
  and has been read as engine time. `[BatchSubmitDraw] itemsPerFrame=1068 parallelForMsPerFrame=6
  workers=30` — 36 items per worker taking 6 ms is ~170 us/item, roughly 30x what the per-draw
  material path measured. Either the work is far larger than believed or the pool is not
  delivering it (29 `Schedule`+notify calls into a deliberately `LowLatency=false` pool, so every
  idle worker must be woken). **Split the timer into dispatch-only vs join-only vs per-chunk
  before adding a fifth stage to a mechanism whose efficiency is unmeasured.**
- **6f. SubmitDraw-tail conversion — PARTIALLY DONE 2026-08-14. `[M]`**
  Live `m_context->m_state` reads between the derivation (`:39069`) and the batch commit point
  (`:50533`) went **24 -> 12**. Converted: the identity/immutable vertex-buffer and input-layout
  cluster, plus five *aggregate* binds (`const auto& xxIa = m_context->m_state.ia;`) that the
  original census missed because they alias the whole `ia` struct instead of indexing it.
  The remaining 12 are all documented in-tree with their reason, and split two ways: batch-3
  content readers (binding + `GetMappedSlice` must convert together), and two SRV scans whose slot
  comes from RDEF reflection and can be any of 128 while the record's PS window stops at slot 7 —
  widening that is the same 128-slot walk Phase 8 exists to remove.

**Relationship to Phase 3 — do not double-count.** Both target `bt_extractXf` (11.07 ms), by
different means: Phase 3 stops deriving the static 97%; Phase 6 parallelises whatever is left.
You do not get 11.07 ms twice. Phase 3 first if draw-redundancy is high, because deleting work
beats moving it.

**But Phase 6 is not optional, and here is why:** `REDUNDANT=97%` is a measurement of **this
scene, standing still**. In combat — explosions, particles, many moving actors, animated
viewmodels — the dirty fraction will be far higher, and Phase 3's win shrinks in exactly the
frames where frame rate matters most. **Redundancy elimination is the win for the common case;
the batch is the insurance for the worst case.** Ship both.

Phase 6 also gives the batch its residue role: the dirty 1-3% in a static scene, plus the
once-per-frame `prepareSceneData` chunk (14.75 ms). And Phase 3's test is only sound *because*
the snapshot exists to hash — the two systems are mutually enabling, not alternatives.

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

### Phase 8 — bind-time SRV bound-mask. **OPTIONAL. ~-0.9 ms on the pole. `[M]`**

**A decision, not a queued task.** Everything else in this plan is local to the RTX injection;
this one reaches into DXVK's D3D11 state tracking, which is why it is written down rather than done.

**What is left after the 2026-08-11 fix.** `captureDrawSnapshot` walked the 128-slot VS SRV array
three times per draw (`reset()`, the wholesale `Com<>` array copy, the rename scan). The copy and
the scan are now fused into one pass that does refcount work only on slots that changed. Measured,
per draw: `dsi_ident` 1153 -> 466 ns, `dsi_scan` 696 -> 903 ns (it absorbed the copy), net
**-480 ns/draw on the pair, -561 ns on `drawSnapId_ns` (-23%), ~-0.60 ms/frame.** All rename
verdicts unchanged (`geoStatic` 96.5%, `elig3` 100% of eligible, `dynSrvSlots{t4,t31}`).

**The residue is the traversal itself, not the refcounting** — ~903 ns/draw for ~160 fixed
iterations (128 SRV + 32 VB), ~5.6 ns each, a dependent load plus a branch. Fusing removed one of
three walks; it cannot remove the remaining one, because capture has no way to know which slots are
bound without looking at all of them.

**The move:** maintain a bound-slot bitmask where the game *binds* (`VSSetShaderResources` and
friends) instead of discovering it at capture. Capture then iterates set bits — 2, per
`dynSrvSlots{t4=432 t31=65215}` — instead of 128.

**Why it is a decision:**
- it modifies DXVK's D3D11 context state, the one area this project has otherwise left alone;
- every bind path must maintain it or the mask silently under-reports, and an SRV missing from the
  record is the silent-carry class (`DrawSnapshot::vsSrvMask` documents the same trap on the
  capture side);
- ~0.9 ms/frame against a frame-thread gap of ~21 ms. It does not change whether the plan closes.

**Do it only if the frame thread is already close and this is the remainder.** Do not do it before
Phase 1.2, which decides whether any of the frame-thread arithmetic holds.

---

## 6. DOES IT CLOSE?

**Do not double-count.** Phase 3 and Phase 6 both target `bt_extractXf`; Phase 2 and Phase 2b
both target `processSceneObject`. In each pair the first deletes the work and the second moves
whatever survives, so only the larger of the two is counted below.

```
FRAME THREAD                                  DXVK-CS
  74.76  start                                  45.45  start
  -2.60  Phase 0  diagnostics                   -7.00  Phase 2   push-not-poll (7-10)
  -4.90  Phase 4  fanout                        -5.66  Phase 5   BLAS (if recoverable)
 -11.07  Phase 3  bt_extractXf                  ------
         (Phase 6 parallelises the residue,     32.79  -> MEETS 33.3
          not counted again)                           Phase 2b in reserve (up to -21)
  ------
  56.19  (~17 fps)  -> still short
 -14.92  pfs_guard  IF recoverable   <- REVOKED 2026-08-11, see below
  ------
  41.27  -> still short of 33.3
```

**CORRECTION 2026-08-11 `[M]`: strike the `pfs_guard` line.** Phase 1.1 resolved it — 95.4% of
`pfs_guard` is `captureDrawSnapshot`, the machinery Phase 3 is built on. It is not overhead that
can be removed; removing it removes Phase 3. At most ~2.1 ms (`drawSnapId_ns`, fields no consumer
reads) plus whatever manifest coalescing recovers from the 10.9 ms span copy is available here,
not 14.92.

> **THE CORRECTION IS ITSELF CORRECTED, 2026-08-14 `[M]`. PUT THE LINE BACK — IT WAS REALISED.**
>
> `pfs_guard` measured **0.66 ms**, from 14.92. The capture was not a floor; it was copying spans
> no consumer read, because it was keyed per VS while the read is decided by the o2w path. Full
> account in §0.2. Nothing Phase 3 runs on was removed.
>
> Note also that the ~2.1 ms `drawSnapId_ns` lead named above is **refuted separately** — those
> fields have ~70 live consumers (`d3d11_rtx.cpp:9604` records the call-site census). The
> recoverable part was never the identity copies; it was the span copy, and it is now recovered.
>
> ```
> FRAME THREAD
>   74.76  start
>  -14.26  pfs_guard 14.92 -> 0.66            REALISED 2026-08-14
>   ------
>   60.50  realised today
>   -2.60  Phase 0  diagnostics               still outstanding
>   -4.90  Phase 4  fanout                    still outstanding
>  -11.07  Phase 3  bt_extractXf              SEE BELOW - did NOT materialise at 65% serve
>   ------
>   41.93  if Phase 3 lands
> ```
>
> **Phase 3 is the open risk, and it is now the specific one.** With verify off and `serve=65%`,
> `bt_extractXf` read **11.41 ms** against its 11.07 baseline — it did not move. Two thirds of its
> draws skip the derivation entirely and the bucket is unchanged. Either the serve is not saving
> what it should, or the compose path costs what the derivation did. Until that is understood the
> `-11.07` line above is `[U]`, not `[I]`.
>
> ~~**And the frame thread is no longer the binding constraint anyway.** With `pfs_guard` at 1% the
> dominant CPU cost is `dxvk-cs` (`prepareSceneData 23.22` + `merge 17.89` ~= 41 ms) against **75%
> GPU idle**. Phases 2 / 2b, untouched, are now the larger target.~~ Caveat per C.3b: that report ran
> at `clkNs=101`, so its absolute values are void and only the ratios are load-bearing.
>
> > **VOID 2026-08-14 — `merge` is nested INSIDE `prepareSceneData`, so that sum double-counts a
> > parent and its own child. See the corrected block in §0.2.** The fat chunk is 22.86 ms. The
> > report's verdict reads `BOTTLENECK: frame thread (95.66 ms). Second is 75.17 ms, SLACK = 20.50`
> > — the frame thread is still the pole, and dxvk-cs inflated *more* than it did versus baseline
> > (1.65x vs 1.28x), so a clean clock widens the gap. **Frame-thread work still pays, up to
> > ~20.5 ms.**

So the frame thread bottoms out around **54 ms (~18.5 fps)** on everything currently identified,
against a 33.3 ms target. **The gap is ~21 ms and there is no line item for it.** Either draw-level
redundancy (Phase 1.2, still `[U]`) turns out to skip the capture itself — which means the skip test
must run on something cheaper than the snapshot it is trying to avoid building, a harder problem
than the plan has anywhere acknowledged — or 30 fps is not reachable on this architecture and the
honest target is ~20 fps. **Settle Phase 1.2 before writing any more code.**

**dxvk-cs closes.** Phase 2 alone very nearly does it; Phase 5 gives margin; Phase 2b is held in
reserve and would take it far below target.

**The frame thread does not close on the numbers currently in hand.** Reaching 33.3 ms needs
Phase 3 to take not just `bt_extractXf` but most of the 53.01 ms `OnDraw*` hook — which is exactly
what draw-level redundancy delivers *if* it matches the instance-level 97%, and *if* the capture
floor (`pfs_guard`) is small. The ~8.5 ms of unnamed residual inside the hook has to come with it.

**Both of those unknowns are in Phase 1. That is why Phase 1 is blocking and every number on the
frame-thread side is provisional.**

> **UPDATE 2026-08-14 `[M]`: one of those two unknowns is closed, in the good direction.**
> "*and if the capture floor (`pfs_guard`) is small*" — it is. 0.66 ms. The paragraph above treats
> that as a condition on reaching 33.3 ms; the condition is met.
>
> The other one is not. Draw-level redundancy (Phase 1.2) is still `[U]`, and Phase 3's realised
> saving is now *worse* than `[U]` — it is measured at approximately zero (`bt_extractXf` 11.41 vs
> 11.07 at 65% serve). That is the next thing to settle on this side.
>
> **But the ordering rule in §7 now says not to settle it next.** `frame ~= max(frameThread,
> dxvk-cs)`: with `pfs_guard` gone the frame thread has dropped by ~14 ms while `dxvk-cs` sits at
> ~41 ms with 75% GPU idle, so further frame-thread savings buy nothing until `dxvk-cs` comes down.
> **Phases 2 / 2b are now the binding constraint, not Phase 3.**

**And note what the arithmetic assumes: a static scene.** These are steady-state numbers with
`REDUNDANT=97%`, `created=0`, `matNew=0`. Under combat load the Phase 3 line shrinks toward zero
and the Phase 6 line has to carry it. Neither column above is a worst-case budget — validate
against a heavy frame before believing 30 fps is held rather than merely reached.

---

### 6.1 THE ARITHMETIC, REDONE ON THE 2026-08-14 BASELINE `[M]`

Everything above this line is written against the 74.76 / 45.45 baseline. It is kept because the
reasoning and the retractions are worth having, but **this is the sum that is current**:

```
FRAME THREAD                                  DXVK-CS
  38.57  now                                    35.66  now
  33.30  target                                 33.30  target
  ------                                        ------
  -5.27  must cut                               -2.36  must cut
```

**SLACK IS 2.91 ms, and that is the number that governs.** Per §7, a realised saving is
`min(size, poleMs - secondMs)`. So:

- Any single frame-thread win larger than ~2.9 ms is **truncated** — it moves the pole down to
  dxvk-cs and stops paying there.
- Neither thread can be finished alone. The old advice on both sides ("do not skip the frame
  thread", and its opposite "Phases 2 / 2b are the binding constraint") were both written when one
  thread led the other by ~20 ms. **At 2.91 ms apart, they must be worked in alternation**, in
  chunks of roughly the current slack, re-measuring between.

What is actually sized and available on each side today:

| side | item | size | status |
|---|---|---|---|
| frame | batch join (Phase 6e) | **6-7 ms** | unexamined, largest single item anywhere |
| frame | `captureDrawSnapshot` bookkeeping (§1a) | 2.20 + 1.52 | `dsi_scan` 0.76 is Phase 8 |
| frame | fanout (`DrawIndexedInstanced`) | 10.89 | 100% dark until marks on |
| cs | `processSceneObject` | 10.58 | unattributed until Phase 1.3 |
| cs | Phase 2 skip | `[U]` | built, `FAIL` not yet 0 (§2.2) |
| cs | BLAS (Phase 5) | 3.62 | `buildBlases` 2.32 + `dynBlas` 1.30 |

**30 fps is 5.3 and 2.4 ms away.** That is the first time this document has been able to write a
single-digit number in that sentence, and it is why the honest-target paragraph above (~20 fps)
is superseded — not by an argument, but by the 2026-08-14 measurement.

---

## 7. RULES THAT MUST NOT BE BROKEN

- **`frame ~= max(frameThread, dxvk-cs)`**; realised saving = `min(size, poleMs - secondMs)`.
  ~~Slack is 29.31 ms today~~ — **slack is 2.91 ms (2026-08-14).** This rule was a footnote when
  the threads were 29 ms apart and it is now the governing constraint on every phase: no single
  win larger than ~2.9 ms realises in full, and the two threads must be worked in alternation.
  See §6.1. (`rtx_options.h:380-381`)
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

**AMENDED 2026-08-14.** Changes since this section was written, all in the live `rtx.conf` and all
now matching their `rtx_options.h` defaults (checked — the two had drifted on `CamRelObjSpans`):

| Option | Now | Why |
|---|---|---|
| `splitTransformVerify` | **False** | The gate it names is met: `FAIL=0`, `FAILcarrier=0 grp{}`. Until this flipped, the cache served **nothing** — the serve branch is `haveObj && haveView && !xVerify` — so every figure taken before it was a projection. |
| `splitTransformObjKeyMask` | **14094335** | +bit 23 (8388608), the draw range in the identity key |
| `splitTransformManifestPathFilter` | **True** (default) | per-path capture; see §0.2 |
| `splitTransformCamRelObjSpans` | **True** | was already True in conf, default was still `false` — drift corrected 2026-08-14 |

**With verify off, `FAIL` and `FAILcarrier` read 0 because nothing is compared, not because
nothing is wrong.** Verify is what derives the second time. A zero there is no longer a
measurement — the screen is the instrument. To re-measure, set it back to True.

Target architecture, on:
`useDrawSnapshot=True` · `batchSubmitDrawStages=True` · `geometryWorkerThreads=0` (auto,
hw_concurrency-2) · `parallelInstanceFanout=True` · `filterDupSameDraws=True` ·
`batchHashes`/`batchBoundingBox`/`batchSkinning` at default `true`

Dead ends, off:
`replayExtractTransforms=False` · `replayExtractVerify=False` · `perfStageDepCensus=False` ·
`extractOverlap` / `extractOverlapVerify` keys removed (do not exist at `4681ecca`)

**AMENDED AGAIN 2026-08-14** (live conf; backup `rtx.conf.bak-2026-08-14-before-pushinst`):

| Option | Now | Why |
|---|---|---|
| `pushInstanceRecords` | **True** | Phase 2.1 |
| `pushInstanceRecordsVerify` | **True** | pinned explicitly rather than left to the default. Nothing is skipped until `FAIL=0` — see §2.2 for what happened when it was first run |
| `memoExtractTransforms` | True -> **False** | inert all session (zero `[Perf.MemoXf]` lines, because `splitHandled` is true for every draw). Turning the feature off rather than leaving it True-but-dead means it cannot wake up when `splitTransformCache` is next toggled for an unrelated reason |
| `memoExtractVerify` | **True** (unchanged) | deliberately left armed. If the memo is ever re-enabled it must come back with verify already on — the same order `pushInstanceRecords` and `splitTransformCache` both use |
| `perfSceneObjSplit` | **False** (unchanged) | Phase 1.3, wanted, but in its OWN window — §8's one-question-per-capture rule |

### THE MEASUREMENT REGIME CHANGED 2026-08-14 — TWO SWITCHES, BOTH DEFAULT OFF `[M]`

**A named-leaf table of zeros is now the NORMAL state of a normal run, not an alarm.**

| switch | where | default | what it costs when on |
|---|---|---|---|
| `RTX_PERF_MARKS` | env var, `d3d11_rtx.cpp:8869` | **OFF** | ~1.6 ms/frame (~41 ns x ~30 marks x ~1,350 draws) |
| `RTX_D3D11_SUBMARK` | env var, `d3d11_rtx.cpp:8781` | **OFF** | the leaf splits inside those marks |

`RTX_PERF_MARKS` is new (`[Perf.MarksCompileOut]`, 2026-08-14) and is the outer gate: with it off,
all five mark lambdas skip the clock read, and `pfs_guard`, `bt_extractXf`, `tail_emit`, every
`te_*`/`bt_*`/`pff_*` accumulator, the eligibility and serve/derive splits and the whole
`[Perf.SubmitDraw.acc]` line read **0**. There is also a compile-time form (`-DRTX_PERF_MARKS=0`)
that recovers the last ~0.04 ms by discarding the branch entirely; nothing in the build system
defines it, so a normal build gets the runtime switch.

**To read leaves you need BOTH:**

```
set RTX_PERF_MARKS=1
set RTX_D3D11_SUBMARK=1
```

**And do not quote a frame time from that window** — the marks are ~1.6 ms of the very number
being optimised. Ratios from the marks-on window, absolutes from the marks-off window. This is
C.5's lesson (diagnostics distort by more than the thing measured) applied to the instrument that
had been exempt from it.

**What stays valid with marks off:** `FRAME`, the three timelines, `HYGIENE`, the D3D11
entry-point rows, the dxvk-cs and GPU sections, and `captureDrawSnapshot`'s own breakdown — those
come from separate unconditional timers, which is the point.

> **THE ORIGINAL WARNING, still true and still worth reading, because the shape recurs.** The
> 2026-08-12 00:06 run reported `pfs_guard 0.28 ms` against 12.73 the run before. Nothing got
> faster: `markSub` early-returns when the sub-markers are off, so the span silently fell into the
> next `markStg` bucket, and the frame time dropped ~10 ms because ~30 clock reads per draw
> stopped happening. **A leaf reading near zero is the signature of a lost switch, not of a win.**
> The only thing that changed in 2026-08-14 is that the switch is now off *on purpose*, so the
> check is "did I set both env vars" rather than "did I lose one".

Still to change when taking measurements:
`perfDrawRedundancy=True` for the Phase 1.2 capture (built 2026-08-11, never yet run) — needs
`useDrawSnapshot=True`, which is already on. It costs four hashes and two map operations per
draw and is billed to no bucket, so it does not move the leaf table — but it is real time on the
pole, so take the redundancy window and the frame-time window separately, never the same one.
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
| **`ExtractTransforms`** | **NO** | ~~blocked on Phase B purity~~ — blocked on its ~170 IN-SUBMITDRAW CONSUMERS. See Phase 6d |
| **instance find/mid/add** | **NO** | on dxvk-cs; blocked on SHARED MUTABLE STATE, not purity. See below |

Parent flag `rtx.batchSubmitDrawStages` (`:353`). Sub-flags default `true` so enabling the parent
gives the fully coherent batch; disable one to bisect a stage back to its per-draw path.

**THE CRITERION FOR JOINING THIS TABLE IS NOT PURITY. `[M]` 2026-08-14.** Every one of the four
stages above writes a `dcs` field read **only on the CS thread** — `materialData`,
`geometryData.hashes`, `geometryData.boundingBox`, `skinningData.pBoneMatrices`. Nothing inside
SubmitDraw consumes them. That is what made them easy, and it is the test to apply before calling
any future stage "one more entry in `DrawWorkItem`".

Measured against it, the two remaining rows fail for **unrelated** reasons, and neither is the
"finish Phase B purity" story this document told for three days:

- `ExtractTransforms` is pure *enough* now (95% of touched draws, `[DrawPure]`), but
  `dcs.transformData` has ~170 readers inside SubmitDraw and three writers among them (Phase 6d).
- instance find/mid/add is not a purity problem at all. **Appendix B.6 already says so in its own
  words** — the scene work races on the instance manager, BLAS cache and geometry cache. Thirty
  workers cannot touch it until Phase 2 gives instances persistent records. B.6 and the old Phase
  6d text contradicted each other; B.6 was right.

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

> **~~Branch warning~~ — OBSOLETE, and following it would now be a regression. `[M]` 2026-08-14.**
> Batch 1 was redone independently on `architecture-overhaul` and HEAD is ahead of `stable` on
> every accessor except two that were left alone deliberately. Full per-accessor table in Phase
> 6a. Cherry-picking from `stable` would import the XfOverlap worker Phase 7 deletes, for nothing.

**~~32 residual live reads remain~~ — recount 2026-08-14: 12 remain in the window that matters**,
i.e. between the derivation (`:39069`) and the batch commit point (`:50533`). Batch 2 is done;
what is left is batch 3 plus two deliberate exclusions.

*~~Batch 2 — identity reads, need new snapshot fields/accessors~~ — **DONE.*** The accessors exist
at `d3d11_rtx.h:1860-1869`: `drawBlendState`, `drawDepthState`, `drawRasterState`,
`drawPsSrv0to7`, `drawRtv0`, `drawVsCb`, plus `drawT31Entry` and `noteGeoLiveRead`. Viewport reads
went 12 -> 27 uses. None of this existed on `stable`.

*A class the original census missed entirely — **aggregate binds:*** five sites of the form
`const auto& xxIa = m_context->m_state.ia;` that then index `.vertexBuffers[0]` and `.indexBuffer`.
They never matched a `m_state.<stage>.<field>` grep because they alias the whole struct. All five
were identity-only and are converted. **Grep for `m_context->m_state\.[a-z]+;` as well as the
two-dot form when re-censusing.**

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

## C.3b `clkNs` IS THE MACHINE-STATE CHECK. Read it before believing ANY absolute time `[M]`

The 2026-08-12 00:22 run reads `FRAME 186-239 ms (4-5 fps)` against 56-62 ms fifteen minutes
earlier, **on the same scene** (`inst=15454 drawsIn=1331 uniqueBlas=456` vs `15463/1328/462`).
It is not a regression and it is not any probe. `[Perf.SessionState]` reads **`clkNs=104-144`
against the 41 ns this project has always measured** — the cost of a single `steady_clock::now()`
tripled — with `xMin=3.18-4.25` and `cpuSlowX` at 2.8 / 3.1 / 4.0 / 5.25.

Everything scaled together, including things no local change can touch:

| | 00:07 | 00:22 | x |
|---|---|---|---|
| `bt_extractXf` | 9.11 | 25.90 | 2.8 |
| `SubmitInstancedDraw` | 19.94 | 66.90 | 3.4 |
| `commitGeometryToRT` (dxvk-cs) | 24.98 | 88.50 | 3.5 |
| `between entry points` (Source engine) | 17.83 | 38.79 | 2.2 |
| **GPU** | **15.71** | **16.06** | **1.0** |

The GPU is a second clock and did not move, which is what proves it is the CPU and not the work.
**The rule: `clkNs` far above ~41 is a machine-state flag, and every absolute millisecond in that
window is void.** Counts and ratios (`[DrawRedund]`, `REDUNDANT%`, `resolved%`) survive it
untouched, because they are tallies rather than timings — take those from any window, take times
only from a window where `clkNs` is sane.

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

- **`architecture-overhaul` @ `5ced56c2`** — current, clean tree, no XfOverlap. Work here.
  Six commits past the `4681ecca` this document was written at:
  `683d49dd` Plan, `6b0e2e7b` backup, `42d116bf` fixes, `0655e48a` fixed options,
  `8bf0d7f7` + `5ced56c2` optimisations.
- **`stable` @ `470eaf4b`** — the XfOverlap commit, message "broken". A single commit off
  `4681ecca`; `git merge-base --is-ancestor stable HEAD` is **false**, so it holds exactly one
  thing this branch does not: XfOverlap, which Phase 7 deletes. **~~Cherry-pick, do not discard~~
  — NOTHING LEFT TO TAKE. `[M]` 2026-08-14.** Its Phase B batch 1 was redone here and surpassed
  (Phase 6a). Already pushed to `origin/stable`, so do not rewind it without a force-push
  decision, but there is no longer a reason to merge from it.
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
