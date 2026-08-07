# PERF INSTRUMENTATION MAP — what already exists, and which timeline it belongs to

**Purpose.** This repo has ~90 `[Perf.*]` tags plus ~25 more perf-relevant tags. They are
not discoverable by reading the code, because the emit site and the measured site are
usually thousands of lines apart and in different files. Every session so far has burned
time re-deriving instruments that were already built, already enabled, and already in the
log.

**Read this before adding a probe.** Deliberately contains no measured values — numbers go
stale within one build, the map does not.

Confidence: `[V]` verified by reading the emit site · `[I]` inferred from correlation

---

## 0. HOW TO USE THIS

1. Decide which **timeline** your question is about (§1). Most wrong conclusions in this
   repo's history come from comparing two numbers that live on different threads.
2. Find the instrument in §3, check its **gate** in §4.
3. Check §5 before you believe the number.
4. Only then consider writing a new probe.

---

## 1. THE TIMELINES

Six clocks run at once. A per-frame millisecond on one is **not** comparable to a
per-frame millisecond on another, and they do **not** sum to the frame.

### 1a. Frame thread `[V]`
Also called the PRESENT thread, the game thread, or the immediate-context owner. This is
the thread TF2 calls D3D11 on and the one that runs `EndFrame`. Its tid is printed on
`[Perf.Busy]` and as `presentTid` on `[Perf.PresentWall]`, and it appears in
`[ThreadCensus]` labelled `(PRESENT)`.

What runs here: every D3D11 entry point, Remix's whole `OnDraw*` → `SubmitDraw` /
`SubmitInstancedDraw` injection path, and the game's own engine code between our entry
points.

What does **not** run here: any Vulkan command. `D3D11DeviceContext::DrawIndexed` calls
`EmitCs([=](DxvkContext* ctx){ ctx->drawIndexed(...); })` — it enqueues a lambda into a CS
chunk. There is no command buffer on this thread.

### 1b. dxvk-cs `[V]`
Named in `dxvk_cs.cpp`. Consumes CS chunks produced by the frame thread and executes them
(`chunk->executeAll()`), which is where Vulkan commands are actually recorded. Remix's
scene processing rides in these chunks: `commitGeometryToRT` per draw, and the
once-per-frame `injectRTX` / `prepareSceneData` pass.

Runs **concurrently** with the frame thread. The frame is roughly `max(frame thread,
dxvk-cs)`, not their sum.

### 1c. dxvk-queue `[V]`
`DxvkSubmissionQueue::finishCmdLists`, `env::setThreadName("dxvk-queue")` in
`dxvk_queue.cpp`. Its loop has exactly three states: wait on a condvar for a queued
cmdlist → `cmdList->synchronize()` on its fence → release/recycle. Those three are what
`[Perf.Gpu]` reports as `idleMs` / `fenceWaitMs` / `reapMs`.

**This is not the frame thread.** A fence wait here is not the frame thread blocking.

### 1d. GPU `[V]`
Measured by timestamp queries written into the command stream, reported by
`[Perf.GpuPass]`. Independent of every CPU number.

### 1e. d3d11-geometry worker pool `[V]`
`WorkerThreadPool` in `d3d11_rtx.h`, sized by `rtx.geometryWorkerThreads` (0 = auto =
hardware_concurrency − 2). Named `d3d11-geometry(N)` in `[ThreadCensus]`. Runs deferred
geometry hashing, bounding-box scans, material compute, skinning — whatever
`rtx.batchSubmitDrawStages` and `rtx.deferMaterialCompute` route to it.

### 1f. Sampler thread `[V]`
A dedicated diagnostic thread that emits `[ThreadCensus]` and `[GapSampler]` on a ~5 s
cadence for the **whole process**. Started when `rtx.perfThreadCensus` or
`rtx.perfGapSampler` is on.

---

## 2. THE NESTING RULE — READ THIS BEFORE ADDING ANY TWO NUMBERS

Several of the biggest instruments are **nested**, not additive. Summing them
double-counts. The chain, all on dxvk-cs:

```
commitGeometryToRT              [CommitRT] perFrameMs
 └─ submitDrawState             [CommitRT] submitMs
     └─ processDrawCallState    [ProcDCS] perFrameMs   ≈ [Perf.SubmitState] process
         ├─ processSceneObject  [ProcDCS] instMs
         │   └─ find/mid/add/update   [Perf.SceneObj], [Perf.UpdInst]
         └─ geom                [ProcDCS] geomMs
     └─ determineMaterialData   [Perf.SubmitState] material → [Perf.MatData]
```

On the frame thread:

```
D3D11 draw entry point                     [Perf.Entry] DrawIndexed / DrawIdxInst / ...
 ├─ LockContext()                          [Perf.DrawEntry] deLock
 └─ D3D11Rtx::OnDraw*()                    [Perf.DrawEntry] deOnDraw
     ├─ SubmitDraw()                       [Perf.SubmitDraw.acc] wallUs / cpuCycles
     │   └─ ~36 stages + ~45 sub-stages    all other fields on that line
     └─ SubmitInstancedDraw()              [Perf.InstDraw] total_ms   (instanced paths only)
         ├─ fanout build                   [Perf.InstDraw] build_ms
         │   ├─ per-instance loop          [Perf.InstDraw] loop_ms
         │   └─ setupPost                  [Perf.InstDraw] setupPost_ms
         │       └─ map/cam bisections     [Perf.MapCut], [Perf.CamCut]
         └─ inner SubmitDraw calls         [Perf.InstDraw] inner_ms
```

**The trap that has caught every session:** `[Perf.SubmitDraw] wallUs` covers only the
inside of `SubmitDraw`. `SubmitInstancedDraw`'s fanout build sits *outside* it and is
measured **only** by `[Perf.InstDraw] build_ms`. On an instanced-heavy scene that is a
large fraction of the frame thread and it is invisible in every `[Perf.SubmitDraw]` field.

Closure check that proves you have it all:

```
[Perf.DrawEntry] deOnDraw
  == [Perf.InstDraw] total_ms  +  (non-instanced OnDraw* paths)
[Perf.SubmitDraw] wallUs
  == [Perf.InstDraw] inner_ms  +  (inner SubmitDraw of the non-instanced paths)
```

---

## 3. INSTRUMENT INDEX

### 3a. Frame-level / wall

| tag | file | timeline | measures |
|---|---|---|---|
| `[Perf.PresentWall]` | `d3d11_swapchain.cpp` | frame thread | frame wall, time inside vs outside `Present`, `presentTid` |
| `[Perf.Busy]` | `d3d11_rtx.cpp` | frame thread | `GetThreadTimes` kernel+user vs wall → `busyPct`, `blockedMs`. Prints its own `tid` |
| `[Perf.Frame]` | `rtx_context.cpp` | dxvk-cs `[I]` | **one sampled frame**, not a window average. `totalInjectUs`, `tail_prepScene`, per-RT-stage CPU |
| `[Perf.Block]` / `[Perf.GameBlock]` | `d3d11_rtx.cpp` | frame thread | blocking waits: `waitForResource`, `m_csThread.synchronize` |
| `[Perf.CondWait]` | `d3d11_rtx.cpp` | any | splits `[Perf.Busy] blockedMs` by DXVK condvar site, with stacks |

### 3b. Frame thread — D3D11 entry points

| tag | file | measures |
|---|---|---|
| `[Perf.Entry]` | `d3d11_rtx.cpp` | ms + call count **inside** each immediate-context entry point |
| `[Perf.Gap]` | `d3d11_rtx.cpp` | complement of the above: time **between** entry points, attributed to the previous call. Large `afterQueryEnd` = engine code at the frame boundary |
| `[Perf.GapQ]` | `d3d11_rtx.cpp` | the `QueryEnd` gap bucket split by query type |
| `[Perf.Boundary]` | `d3d11_rtx.cpp` | of the TS_DISJOINT gap, how much fell before vs after `Present` |
| `[Perf.DrawEntry]` | `d3d11_rtx.cpp` (globals) / `d3d11_context.cpp` (accum) | **`deLock` vs `deOnDraw`** — separates device-lock contention from Remix's hook. The only instrument that can acquit the lock |
| `[Perf.Query]` | `d3d11_context_imm.cpp` | `GetData` poll count, not-ready %, DONOTFLUSH, distinct query objects, per-type histogram |
| `[Perf.QEvent]` / `[Perf.QEndSite]` | `d3d11_context_imm.cpp` | event-query detail / `QueryEnd` call sites |
| `[Perf.SyncSite]` | `d3d11_context_imm.cpp` | spin-burst stacks. `polls == bursts` means that site **blocks**, not spins |

### 3c. Frame thread — Remix injection (`SubmitDraw`)

| tag | file | measures |
|---|---|---|
| `[Perf.SubmitDraw.acc]` | `d3d11_rtx.cpp` | the big one. ~80 named stage accumulators. **ns**, except `wallUs` (µs) and `cpuCycles` |
| `[Perf.SubmitDraw.max]` | `d3d11_rtx.cpp` | same fields, per-draw max — says whether a stage is uniform or outlier-driven |
| `[Perf.SdThreads]` | `d3d11_rtx.cpp` | per-thread `SubmitDraw` census. Answers "is this one thread or many?" |
| `[Perf.SdStall]` | `d3d11_rtx.cpp` | 8 checkpoints, 1-in-64 draws, wall **vs** `QueryThreadCycleTime` cycles. `wall >> cycles` = that segment blocks. **Structurally blind to `LockContext`** — starts inside `SubmitDraw` |
| `[Perf.CullVtx]` | `d3d11_rtx.cpp` | interior of the `bt_cullVtx` stage, by scan source |
| `[Perf.SrvCache]` / `[Perf.FillMatCache]` / `[Perf.InstBufCache]` / `[Perf.T31Cache]` | `d3d11_rtx.cpp` | hit/miss for the per-draw caches. **Confirm cache wins here, not in a timing bucket** |
| `[Perf.WcCopy]` | `d3d11_rtx.cpp` | write-combined read traffic by call site (bytes + calls) |
| `[Perf.BonePath]` / `[Perf.BlasBounds]` / `[Perf.MemCat]` / `[Perf.OptRead]` / `[Perf.FmtSite]` | `d3d11_rtx.cpp` | narrower per-draw probes |

### 3d. Frame thread — the instanced fanout (`SubmitInstancedDraw`)

| tag | file | measures |
|---|---|---|
| `[Perf.InstDraw]` | `d3d11_rtx.cpp` | **the fanout aggregate.** calls, instances, `total`/`inner`/`build`/`loop`/`setupPost`, plus a least-squares `intercept + slope*instances` fit and `perInst_us` |
| `[Perf.MapCut]` | `d3d11_rtx.cpp` | bisection of the fanout's map bucket: `mapChain` / `t31Map` / `dbgTrack` / `mtnProbe` / `camOrigRest` |
| `[Perf.CamCut]` | `d3d11_rtx.cpp` | bisection of the `camOrigin` span |
| `[Perf.InstFit]` | `d3d11_rtx.cpp` | the regression readout |
| `[InstStall]` | `d3d11_rtx.cpp` | **per-call outliers**: `totalUs`/`buildUs`/`loopUs`/`innerSubmitUs` + VS hash |
| `[SubmitStall]` / `[SubmitStall.Frame]` | `d3d11_rtx.cpp` | any `OnDraw*` over `rtx.submitStallUs`, with frame draw-ordinal, type, VS hash, prim/instance counts; plus a per-frame roll-up |

### 3e. dxvk-cs

| tag | file | measures |
|---|---|---|
| `[Perf.CsSplit]` | `dxvk_cs.cpp` | **chunk-duration histogram** for the thread. `execMs` vs `idleMs` vs `busyPct`, bucketed `<10us / <100us / <1ms / <10ms / >=10ms`. One fat chunk per frame in the top bucket = the once-per-frame Remix pass |
| `[Perf.CsCmd]` | `d3d11_rtx.cpp` | vtable-keyed per-command-type cost — names *which* CS command owns the thread |
| `[CommitRT]` | `rtx_context.cpp` | per-draw probe in `commitGeometryToRT`. `perFrameMs` / `finalizeMs` / `submitMs` / `otherMs` / `avgUsPerDraw`. **`finalizeMs` = blocked on worker futures**; 0 means computing, not stalled |
| `[ProcDCS]` | `rtx_scene_manager.cpp` | split of `processDrawCallState`: `geomMs` / `instMs` / `otherMs` |
| `[Perf.SubmitState]` | `rtx_scene_manager.cpp` | `submitDrawState` stage split: entry / hash / material / process |
| `[Perf.MatData]` / `[Perf.MatCache]` | `rtx_scene_manager.cpp` | `determineMaterialData` and its cache |
| `[Perf.SceneObj]` | `rtx_instance_manager.cpp` | `processSceneObject` split: `find` / `mid` / `add` / `update`, as `usPerCall`, `estMsPerFrame` and `pct`. **`addedPct`** is the regression check |
| `[Perf.UpdInst]` | `rtx_instance_manager.cpp` | inside `updateInstance`; surf-skip rate. Read `surf` **and** `tail` together, never `surf` alone |
| `[Perf.MidWork]` / `[Perf.NonOpaque]` | `rtx_instance_manager.cpp` | narrower instance-path probes |
| `[FindStage]` | `rtx_instance_manager.cpp` | dedup outcomes: `calls` / `exact` / `withPropId` / `propIdMiss`. **The acceptance test for any dedup change** |
| `[MapGate]` / `[MapWrite]` / `[SpatialMove]` / `[SpatialBump]` | `rtx_instance_manager.cpp`, `util_spatial_map.h` | spatial-map write invariants. `mapWrMove` must track `onTransformChanged`; `[SpatialBump]` non-zero = hash collisions |

### 3f. dxvk-cs — the once-per-frame scene pass

| tag | file | measures |
|---|---|---|
| `[Perf.PrepScene]` | `rtx_scene_manager.cpp` | `prepareSceneData` split: `gc` / `setup1` / `instSetup` / `merge` / `accel` / `light` / `surfMat` / `cull` / `tlas` / `tail`, plus `inst` / `surf` **counts**. Values in **µs** |
| `[Perf.PrepSplit]` | `d3d11_rtx.cpp` | companion split |
| `[Perf.Merge]` | `rtx_accel_manager.cpp` | `setup` / `loop` / `dynBlas` / `buildBlases` / `tail` + `uniqueBlas` / `buckets` / `bBuild` / `bUpdate` / `bReuse`. µs |
| `[Perf.MergedBucket]` | `rtx_accel_manager.cpp` | per-bucket BUILD vs reuse, flags fully-static rebuilds |
| `[Perf.Blas]` / `[Perf.BuildBlas]` / `[Perf.Accel]` / `[Perf.Tlas]` / `[Perf.TlasOverlap]` / `[Perf.World]` | `rtx_accel_manager.cpp` | acceleration-structure detail |
| `[Perf.GeoChurn]` / `[Perf.ChunkTrim]` / `[Perf.Alloc]` / `[Perf.Stall]` | `rtx_scene_manager.cpp` | geometry cache churn / trimming / allocation |
| `[MatChurn]` | `rtx_scene_manager.cpp` | **capture-hygiene gate.** `matNew` / `matTotal` / `texTotal`. See §5 |
| `[TrimCache]` / `[SpikeRB]` / `[BlasDestroyed]` | `rtx_scene_manager.cpp`, `rtx_accel_manager.cpp` | BLAS-input cache lifetime and the streaming-race probes |

### 3g. dxvk-queue and GPU

| tag | file | timeline | measures |
|---|---|---|---|
| `[Perf.Gpu]` | `d3d11_rtx.cpp` (reads `DxvkStatCounters`) | **dxvk-queue** | `fenceWaitMs` / `reapMs` / `idleMs` / `cmdLists`. See §5 for why these always sum to the frame |
| `[Perf.FenceSpike]` | `dxvk_queue.cpp` | dxvk-queue | threshold-gated single-submission outlier, names the cmdlist and whether it had a wait semaphore |
| `[Perf.PresentSpike]` | `dxvk_queue.cpp` | present | slow `presentImage` |
| `[Perf.SubmitGap]` | `dxvk_device.cpp` | — | time between submissions |
| `[Perf.GpuPass]` | `rtx_context.cpp` | **GPU** | per-pass GPU timestamps + `outsideRtMs`. **Only trust it when the line says `ALIGNED`**; `*SHIFTED*` means the stage labels are meaningless. `rtx.perfGpuStageSerialize` is needed for clean per-pass attribution |
| `[Perf.GbDispatch]` / `[Perf.GbBranch]` / `[Perf.Occupancy]` | `rtx_pathtracer_gbuffer.cpp` | GPU | gbuffer dispatch detail |
| `[Perf.ShaderClock]` | `rtx_context.cpp` | GPU | in-shader clock instrumentation |
| `[Perf.Omm]` / `[Perf.SkyPrefill]` / `[Perf.AutoExpPlus]` / `[Perf.Resolve]` | various | GPU | subsystem-specific |
| `[Perf.Barrier]` | `d3d11_rtx.cpp` | — | barrier mix by type with resource counts. Distinguishes a full pipeline drain from a narrow transition |

### 3h. Whole-process

| tag | file | measures |
|---|---|---|
| `[ThreadCensus]` | `d3d11_rtx.cpp` | **every thread in the process**, `%core` over the window, sorted, with `busySum`. Names threads by symbol (`tier0.dll+0x...`, `dxvk-cs`, `d3d11-geometry(N)`) and marks `(PRESENT)`. The first thing to read on any "what is the frame" question |
| `[GapSampler]` | `d3d11_rtx.cpp` | stack sampling of the frame thread, bucketed by symbol |
| `[Perf.CpuCalib]` | `d3d11_rtx.cpp`, `rtx_cpu_stall_probe.h` | clock calibration / `cpuSlowX` |
| `[Perf.VidMem]` / `[Perf.TexBudget]` | `d3d11_rtx.cpp`, `rtx_texture_manager.cpp` | memory |
| `[Perf.PipeGfx]` / `[Perf.PipeComp]` / `[Perf.PipeRT]` / `[Perf.Shader]` / `[Perf.StateCache]` | `dxvk_*.cpp` | pipeline + shader compile cost |
| `[Perf.Sweep]` | `rtx_context.cpp` | driver for `rtx.perfAutoSweep*` — automated option sweeps with settle time |

### 3i. Ceiling / feasibility diagnostics (no behaviour change)

These answer "is optimisation X worth building" **before** you build it. All default off.

| tag | option | answers |
|---|---|---|
| `[MemoCeiling]` | `rtx.logMemoCeiling` | of draws reaching commit, how many are identical to last frame's (`geomStable`) and how many are fully replayable (`fullStable`). Upper bound on memoisation |
| `[DupPass]` | `rtx.logDupPass` | within-frame redundant re-injection: `depthOnly`, `dupDepthOnly`, `dupDiffCam`, `dupDiffPs`, `dupSame`. Which passes can be filtered before commit |
| `[Phase0]` / `[Phase0.Summary]` / `[Phase0.MultiMode]` | `rtx.logPhase0Descriptor` | is the per-VS layout bounded and stable enough for GPU-driven injection |
| `[Phase2]` / `[Phase2.MissVS]` | `rtx.capturePhase2` | per-draw capture-record completeness vs injected draws |

---

## 4. GATES

### 4a. `rtx.conf` options

Split probes (default **off** — turning these on costs CPU on the path being measured):
`perfSceneObjSplit`, `perfUpdateInstSplit`, `perfMaterialSplit`, `perfSubmitStateSplit`,
`perfCsCmdProbe`, `perfNonOpaqueCensus`, `perfBlasBoundsProbe`, `logPrepSceneSplit`,
`logFanoutSplitStats`, `logGeomDiag`, `logTlasSet`, `logResolveCensus`,
`logSurfaceCoverage`, `logPrimaryHitCensus`, `perfUnorderedStepCensus`.

Per-draw outlier + ceiling probes (default off): `logSubmitStall` (+ `submitStallUs`,
also the gate for `[Perf.DrawEntry]`, `[Perf.InstDraw]`, `[Perf.MapCut]`, `[InstStall]`),
`logMemoCeiling`, `logDupPass`, `logPhase0Descriptor`, `capturePhase2`.

Process census (default off): `perfThreadCensus`, `perfGapSampler`, `perfShaderClock`
(+ `perfShaderClockLogInterval`), `perfGpuStageSerialize`.

Default **on**: `logMaterialChurn` (`[MatChurn]` — leave it on, it is the hygiene gate),
`deferMaterialCompute`, `batchHashes`, `batchBoundingBox`, `batchSkinning`,
`logResolveCensusRaw`.

Behaviour-changing perf knobs, not probes: `batchSubmitDrawStages`,
`geometryWorkerThreads`, `perfCullInstancesLargerThan`, `perfSkipPom`,
`perfSkipMaterialTextures`, `perfSkipThinFilm`, `perfSkipGeometryFetch`,
`perfGbStopAfter`, `perfUnorderedStopAfter`, `perfMaterialStopAfter`,
`perfSurfaceInteractionStopAfter`, `perfCoherentUnorderedFetch`,
`perfForceSamplerMipBias`, `perfForceSamplerAniso`, `perfCheapTextureGradients`.
Sweep driver: `perfAutoSweep`, `perfAutoSweepSeconds`, `perfAutoSweepSettleSeconds`,
`perfAutoSweepQuick`, `perfAutoSweepExitOnFinish`.

### 4b. Environment variables (no rebuild)

| var | effect |
|---|---|
| `RTX_D3D11_SUBMARK` | **default ON.** `=0` disables the optional `markSub`/`markXtSub` sub-splits (`pfs_*`, `pff_*`, `o2w_*`, `pvr_*`, `w2vs_*`, `fm_*`, `fmm_*`, `tc_*`, `te_*`, `xsf_*`). Parent buckets are unchanged; sub-buckets read 0, meaning *not measured*, not *free* |
| `RTX_D3D11_DIAG` | `=1` re-enables the pure-diagnostic per-draw blocks that are off by default |
| `RTX_FLUSH_MIN_US` / `RTX_FLUSH_INC_US` / `RTX_FLUSH_MAX_PENDING` | implicit-flush policy — forces fewer, larger submissions |
| `RTX_COND_DIAG`, `RTX_BONE_DIAG`, `RTX_SKIN_DIAG`, `RTX_BLIT_DIAG`, `RTX_FMT_DIAG`, `RTX_WCCOPY_DIAG`, `RTX_REJECT_LOG`, `RTX_IDX_CPU_SNAPSHOT`, `RTX_NPC_BONE_BUF`, `RTX_LIGHTMATCH_AUDIT` | subsystem diagnostics |
| `RTX_LOG_UNBUFFERED`, `DXVK_ODS_LOG` | logging behaviour |
| `RTX_DISABLE_ENGINE_HOOKS` | disables the TF2 engine hooks |

### 4c. Compile-time

`static constexpr bool kDiagLogs` in `d3d11_rtx.cpp` — currently `false`, dead-strips a
large family of per-frame dumps (`[TLASFrame]`, `[TLASEntry]`, `[TLASEntry-View]`,
`[SubViewVar]`, `[r8Hist]`, …). **The producers that feed them are not all gated on it**;
check before assuming a census is free.

### 4d. The output filter

`log.cpp` holds a prefix denylist. A tag on it is **silently dropped** even though the
code ran and paid for it. If a new log line does not appear, check this list first —
before concluding the code did not execute.

---

## 5. READING RULES AND KNOWN TRAPS

**Identities that are true by construction and prove nothing.**
`[Perf.Gpu]` `idleMs + fenceWaitMs + reapMs == frame`, always, in every window. Those are
the three states of the dxvk-queue loop; the partition is complete by definition. It is
not evidence that the GPU gates the frame, and it says nothing about any other thread.

**Cross-timeline comparison.** Two numbers of similar magnitude on different threads are a
coincidence until proven otherwise. `[CommitRT] submitMs` (dxvk-cs) and `[Perf.Gpu]
fenceWaitMs` (dxvk-queue) cannot nest — there is no shared thread to nest them on.

**Nesting.** See §2. Adding `[CommitRT]` and `[ProcDCS]` double-counts.

**Instrumented span ≠ function.** `[Perf.SubmitDraw] wallUs` excludes
`SubmitInstancedDraw`'s fanout build. `[Perf.SdStall]` starts inside `SubmitDraw` and is
blind to `LockContext` — its `stall%` cannot acquit lock contention. Use
`[Perf.DrawEntry]` for that.

**Units.** `[Perf.SubmitDraw.acc]` is **ns** except `wallUs` (µs) and `cpuCycles`.
`[Perf.PrepScene]`, `[Perf.Merge]`, `[Perf.Frame]` are **µs**. `[Perf.Busy]`,
`[Perf.Gpu]`, `[Perf.GpuPass]`, `[Perf.CsSplit]` are **ms**. Fields ending `_ns` are ns
wherever they appear.

**Truncation.** Anything accumulated via `duration_cast<microseconds>` per call floors
each sample and therefore always reads **low** — across thousands of draws that is a real
deficit, not rounding noise.

**Window vs sample.** Most lines are per-window totals; divide by the `frames=` count on
`[Perf.Busy]` for the same window to get per-frame. `[Perf.Frame]` is different: it is one
sampled frame, printed every N frames.

**Instrument cost.** `markStg` / `markXt` / `markFm` are **unconditional** clock reads;
only `markSub` / `markXtSub` honour `RTX_D3D11_SUBMARK`. `steady_clock::now()` is QPC,
tens of ns. With dozens of marks per draw across thousands of draws this is a measurable
share of the very path being measured. `QueryThreadCycleTime` was tried per-stage and
**refuted** — an order of magnitude more expensive than QPC.

**Sub-microsecond leaves.** Leaves whose real work is well under a microsecond are below
the resolution of any per-draw timer. Do not conclude a leaf "did not optimise" from a
flat bucket — confirm by **mechanism** instead (a cache-hit line going to ~100%, a call
count going to zero).

**Per-DRAW vs per-INSTANCE.** A probe billed per draw but running per instance is wrong by
the instance-per-draw ratio. This has recurred repeatedly. Always divide the probe's own
call count by the frame's draw count before believing a per-draw figure.

**Attribution direction.** `markStg` *advances* the marker, so a parent bucket reports the
**remainder** after its subdivisions. Parent + children = the original bucket.

---

## 6. CAPTURE HYGIENE — gate every number on all of these

- `[MatChurn]` shows `matNew=0` and `matTotal` flat. Material/texture churn invalidates
  everything downstream.
- No `rtx-asset-exporter` in `[ThreadCensus]`. It runs a significant fraction of a core
  for tens of seconds after spawn and tails off slowly — check the *final* windows of a
  capture, not just the first.
- Same scene and same player position. `dxvk-cs` busy% varies widely within a single run.
- `[Perf.GpuPass]` says `ALIGNED`, or its per-stage labels are meaningless.
- Cross-capture deltas smaller than about a millisecond mean nothing without all of the
  above.

---

## 7. WHEN YOU DO NEED A NEW PROBE

Before writing one:

1. Grep the tag list. `grep -o '"\[[A-Za-z0-9_.-]*' src/d3d11 src/dxvk src/util` and look
   for anything adjacent to your question.
2. Check whether an existing probe is simply **gated off** (§4) rather than absent.
3. Check the `log.cpp` denylist (§4d) — the probe may exist, be enabled, and be filtered.
4. Check the runtime log itself. Several probes are already enabled in `rtx.conf` and have
   been emitting for months.

When you do add one, put its **timeline** and its **nesting parent** in the comment at the
emit site, and add the tag to §3 here.
