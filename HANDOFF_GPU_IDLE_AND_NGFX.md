# Handoff: the frame is half GPU-idle and nobody knows why. Five theories died proving it.

Written 2026-07-28 ~17:50. Everything below is measured unless it says INFERENCE.
Numbers come from four nsys captures (`043325`, `150230`, `153316`, `165811`),
`[Perf.*]` in `remix-dxvk.log`, and dbghelp symbol resolution against
`Titanfall2\bin\x64_retail\d3d11.pdb`.

**Read sec 2 before doing anything.** It is a list of things that look like the
answer, are not, and cost a run each to disprove.

---

## 1. Where the frame actually goes

Steady state, 1920x1080 native, ~16-17 fps:

| | ms | note |
|---|---|---|
| frame | **58-63** | 236-246 usable frames per capture |
| `InjectRTX` (all Remix GPU work) | 51-59 | ~90% of the frame |
| — real Vulkan GPU execution | **21.7-24.9** | submit-busy union |
| — **GPU idle inside InjectRTX** | **25.4-33.9** | **THE OPEN PROBLEM** |
| outside InjectRTX | ~5.4 | present, blit, game raster |

Of the real GPU work, one pass dominates:

| pass | ms | % of real work |
|---|---|---|
| **Primary Rays** | **11.0-12.5** | **~51%** |
| Integrate NEE | 3.0 | 13.8% |
| Integrate Direct Raytracing | 1.4 | 6.3% |
| RTXDI Spatial / Initial+Temporal | 1.0 / 0.8 | 8.2% |
| everything else (~20 passes) | <0.7 ea | ~20% |

`gb_primaryRays` is the single most-confirmed number in this investigation:
**11.0 ms**, agreeing across four runs and two independent clocks (nsys GPU
workload ranges, and the engine's own `[Perf.GpuPass]` at 10.02/10.08/10.47/11.05).

---

## 2. REFUTED — do not re-derive these

Five confident hypotheses, all dead. Each cost at least one full capture or build.

* **"The GPU idle is untraced CUDA work (NRC / DLSS-RR)."** WRONG.
  `--trace=...,cuda` was added and the capture ran with `COLLECT_CUDA_TRACE=true`,
  CUPTI loaded and injected successfully. Result: *"No CUDA events collected"*,
  448 CUPTI events total, no CUDA activity tables at all. Remix and NGX use the
  **Vulkan** backends here; there was never CUDA to find. `cudart64_12.dll` is
  loaded (NRC links it) but dispatches nothing.

* **"The GPU idle is dropped GPU timestamps."** WRONG. Median **882** GPU events
  per frame against nsys's 8192 cap; p95 1080, max 1493. The
  `exceeded ... by 231126` diagnostic is real but belongs to the **freeze**,
  where one "frame" never ends and swallows the whole per-frame budget. Also
  binned frames by missing submit-correlations: idle was 30.49 / 30.15 / 34.04 ms
  for 0 / 1-2 / 3+ missing. Not a coverage artefact.

* **"The geometry WorkerThreadPool's locking is what starves the GPU."** WRONG,
  and this one cost a build. Sampling showed `d3d11.dll` 2.38x over-represented
  in the idle window with `Schedule` blocked in-kernel and workers asleep in
  `processWork`. Three real defects were found and FIXED (sec 4). Result:
  `[BatchSubmitDraw] parallelForMsPerFrame` **6-8 ms before, 6-8 ms after**,
  fps unchanged. The workers were idle because the producer had nothing to give
  them — the serialisation is upstream of the pool, not inside it. Contention
  was a symptom read as a cause.

* **"WBOIT is what drives the gbuffer shader to 255 registers."** WRONG, twice
  over. With `rtx.wboitEnabled = False` the permutation
  `gbuffer_rayquery_nrc_opaque_translucent` still reports **Register Count=255**,
  the **same 144 B/thread spill**, and a marginally LARGER binary (833024 vs
  826880). It is also a large perf REGRESSION: fps ~16 -> 6.8-10.3,
  `pt_integrate` 12.5-14.7 -> 27.9-39.3 ms, `upscaler` 11-13 -> 28.9-35.0 ms.
  Reverted in `rtx.conf`.
  **The trap:** 255 is the hardware CAP, so that counter is SATURATED and cannot
  show partial progress. Removing one feature just lets the compiler spend the
  freed registers elsewhere. Judge register pressure by the **Local Memory
  (spill)** figure, not by Register Count, until the shader is small enough to
  drop under 255 at all.

* **"ngfx's 4294.9 MB PMA buffer / the raised buffer options caused the hang."**
  WRONG. `--allocated-event-buffer-memory-kb 40000` and
  `--allocated-timestamps 400000` had been added on speculation (reasoning by
  analogy from nsys's 8192 ceiling — a different tool's limit). Removing them
  changed nothing: `GPU PMA Buffer Size (MB) = 4294.9` still prints. That is
  ngfx's own default reporting. Both options now default to 0 = "don't pass".

* **"Conflicting Vulkan implicit layers (RenderDoc/RTSS/ReShade) break ngfx."**
  WRONG. This machine really does load **12 implicit layers** into every Vulkan
  instance, and the reasoning was sound — but the A/B says no: with the shader
  profiler off, the capture works **identically** whether the eight
  capture/overlay layers are suppressed or not.

---

## 3. Established facts worth keeping

* **`gb_primaryRays` ~11.0 ms**, ~51% of all real GPU work, two independent
  instruments, four runs. Unmoved by anything tried.
* **The gbuffer shader is at the register cap.** `[Perf.Shader]`:
  ```
  gbuffer_rayquery_nrc_wboit_opaque_translucent        Register Count=255  Binary 826880  LocalMem 144B
  gbuffer_rayquery_nrc_wboit_opaque_translucent_occ256 Register Count=255  Binary 826496  LocalMem 144B  <- the ACTIVE rung
  gbuffer_psr_rayquery_nrc_wboit_opaque_translucent    Register Count=255  Binary 406016  LocalMem 144B
  ```
  These are the ONLY shaders in the tree at 255; the next highest are
  `integrate_nee` and `rtxdi_spatial_reuse` at 168, integrators at 128.
  On AD104: 65536 regs/SM / (255 x 32) => **8 warps/SM of a possible 48 ~= 17%
  occupancy**. With `blockThreads=256` that is exactly one resident block.
* **The occupancy ladder cannot help.** `rtx.perfGbufferBlockThreads` (rungs
  256/384/512, `getOccupancyLadderShader`) changes BLOCK SIZE, but occupancy here
  is REGISTER-bound. At 255 regs the max resident is ~256 threads/SM total, so
  the active 256 rung is already the ceiling and 384/512 cannot be resident.
  This is why every ladder sweep read flat.
* **You are not CPU-bound.** `totalInjectUs` = 3554-5620 us against a ~58 ms
  frame; `GPUIDLEms = 0` and `tail_gpuIdle = 0` on every line. The engine never
  waits for the GPU.
  (`cpuSlowX` 2.4-3.3 seen during captures is **profiler overhead**, not
  throttling — it reads 1.00 on unprofiled runs of the same scene.)
* **`r_no_stalls 1` works.** `[Perf.SyncSite] site0 bursts=0 polls=0` through all
  of gameplay. BUT see sec 6 for site1.
* **SER is unavailable where the cost is.**
  `enableShaderExecutionReorderingInPathtracerGbuffer` is default **false** and
  the option's own text says *"(Note: Hard disabled in shader code)"*. The gbuffer
  also runs `RayQuery(compute)` (`[Perf.GbBranch] executing=RayQuery(compute)`),
  and SER permutations only exist on the RT-pipeline paths. `ser=0` is blocked
  twice over. It IS already enabled for Integrate Indirect (default true).
* **OMM is on and healthy but doing almost nothing.** `[Perf.Omm]`:
  `optEnable=1 supported=1 active=1 enoughMem=1`, but `requestedBinds=28.8/frame`
  vs `boundOMMs=3.8/frame` (**13% coverage**), `blackListed=36`,
  `usedMB=5.99` of `budgetMB=1536` (**0.4%**). Not a cost problem — a missed
  benefit. 87% of alpha-tested binds still pay full any-hit traversal, feeding
  the exact pass that is #1.

---

## 4. Code changed this session

| # | change | file | status |
|---|---|---|---|
| 1 | **Per-queue steal locks** (was ONE pool-wide spinlock taken per pop, so an idle worker grabbed it up to N-1 times per scan while every other worker did the same — O(threads^2) lock traffic). Cache-line padded (`alignas(64)`) or 8 locks share a line and it reverts. | `util_threadpool.h` | landed, **did not move fps** |
| 2 | **Notify outside the mutex** in `Schedule` (was notifying while holding `m_taskMutex`, so the woken worker immediately blocked on it). The empty `{ lock }` scope before the notify is load-bearing — it closes the lost-wakeup race. | `util_threadpool.h` | landed, **did not move fps** |
| 3 | **Yield on failed steal scan in BOTH modes** (the `LowLatency` guard meant this pool never yielded; while any queue held work the condvar predicate stayed true and a worker that lost the race span the whole loop at 100%). | `util_threadpool.h` | landed, **did not move fps** |
| 4 | `AtomicQueue::isEmpty()` — deliberately racy hint so a thief skips a queue without paying its lock. | `util_atomic_queue.h` | landed |
| 5 | **Log filter sweep**, 30 new prefixes, binned by TIME so front-loaded probes were not confused with per-frame ones. 20.04 MB -> 5.50 MB (72.6%), 112190 -> 39293 lines. `[Perf.*]`, `[NsysAuto]`, `[ShaderHashMap]`, `[CamMgr.*]`, `[SubmitStall]`, `[Coverage]` all explicitly preserved. | `util/log/log.cpp` | landed |

1-3 are defensible on their own terms (they are real scalability defects) but are
**unproven as wins here**. If you want them gone, the yield (#3) is the one to
back out first — it trades a spin for a scheduler round trip.

`rtx.conf`: `rtx.geometryWorkerThreads = 0` added + documented (auto = 30 on this
32-core box); `rtx.wboitEnabled = False` reverted with the regression recorded.

---

## 5. Tooling state

**`Capture TF2 (Nsight Systems auto).ps1`** — added `cuda` to `-Trace`; `-Sample
process-tree` at 1000 Hz; `-ResolveSymbols false` (true makes QdstrmImporter
grind for many minutes at "49%" with a 4.7 GB working set — and module+offset is
enough to answer "whose code", which is what mattered).

**`Capture TF2 (Nsight Graphics auto).ps1`** — was a byte-identical COPY of the
Systems script whose `.bat` even invoked the Systems `.ps1`. Now genuinely drives
`ngfx --activity "GPU Trace Profiler"` (Ada / Throughput Metrics / 2000 ms).
Same trigger machinery, three real differences:
  * **ONE hotkey, not two.** `--start-after-hotkey` arms a single trace bounded
    by `--max-duration-ms`; there is no stop key. `CAPTURE-END` is observational.
    A second F11 would arm ANOTHER trace.
  * `--set-gpu-clocks` defaults to `unaltered`, NOT ngfx's `base`. `base` pins the
    GPU below boost so counters repeat, which silently makes every time in the
    capture incomparable to the nsys traces and to in-game fps.
  * Output is a DIRECTORY per run, not a path.

**Two PowerShell traps hit while writing it**, both the same root cause:
`Start-Process -ArgumentList` in PS 5.1 joins `string[]` with spaces and **quotes
nothing**. `--activity "GPU Trace Profiler"` arrived as four arguments
(`Unknown activity name: GPU`). Fixed by `Format-NgfxArgs`, which quotes on
whitespace only. The Systems sibling never tripped this because none of its
values contain spaces.

### The ngfx freeze — SOLVED

`--real-time-shader-profiler` is the **sole** cause. Isolated across four runs:

| shader profiler | layer suppression | result |
|---|---|---|
| on | on | hang |
| on | on, buffers reverted | hang |
| **off** | on | **works** |
| **off** | **off** | **works** |

Presents as ngfx reaching `Session established. Starting activity...` then sitting
on `Waiting for target ready...` until `Activity session destroyed, connection
error encountered`, while the game freezes ~3 s in, still loading shaders.

**Tell:** the INFO line reports `Warp State Sampling Interval = 1024` when the
profiler is on, and **16** when off. If a future run hangs, check that number.

INFERENCE, not proven: source-level attribution requires ngfx to ingest and index
every shader module at creation. This load creates **9,408 unique shader modules**
(DXVK translating the game's D3D11 shaders) plus 81 Remix RT/compute shaders,
several 400-835 KB. That work happens exactly where the log stops. Untested.

The script now defaults `-ShaderProfiler` OFF (opt-in) and `-DisableVkLayers` to
empty. Note what you KEEP by disabling it — per NVIDIA's own text, *"When
disabled, a more detailed list of SM and L1TEX perf counters will be collected."*

---

## 6. Open, in priority order

1. **The 25-34 ms of GPU idle.** Biggest single item in the frame, reproduces
   across every capture, and every explanation the Vulkan trace can test is
   eliminated (sec 2). Sampling says `d3d11.dll` is 2.38x and `engine.dll` 3.06x
   over-represented in the window while `tier0.dll` (thread-pool parking) is
   UNDER-represented at 0.53x — i.e. work concentrates in Remix during exactly
   the window the GPU is starved. Fixing the pool's locking did not help.
   **The instrument not yet used: `-CpuCtxSw process-tree`** (blocked-vs-running,
   not "which module owns the IP"). That distinction is what the pool theory got
   wrong. Requires elevation.
2. **`gb_primaryRays` ~11 ms / 17% occupancy.** The one unambiguous, actionable
   cost. The lever is REGISTER COUNT, i.e. shrinking the
   `nrc + wboit + opaque_translucent` uber-shader permutation — not the block-size
   ladder, and not removing WBOIT (sec 2). A working ngfx GPU Trace with
   Throughput Metrics answers whether it is SM-bound, memory-bound, or
   occupancy-bound, which decides whether that is worth doing.
3. **`Titanfall2_2026_07_27_00_46_15.6422829.ngfx-gputrace`** — 6.6 MB, intact,
   in `Documents\NVIDIA Nsight Graphics\`. Captured successfully BEFORE all of
   the above. Predates the DLSS fix so its frame times are stale, but register
   count and occupancy are properties of the compiled shader, not the resolution.
   Opening a saved trace performs no injection, so the UI launch freeze is
   irrelevant to reading it. **Cheapest available answer to #2.**
4. **OMM at 13% bind coverage, 36 blacklisted.** Free benefit being left on the
   table, feeding pass #1.
5. **`[Perf.SyncSite] site1`** — a SECOND engine sync site `r_no_stalls` does not
   guard. `bursts=7 polls=7 ms=282.57` => `polls == bursts`, so it BLOCKS rather
   than spins, 30-40 ms per burst, ~1 per 15 s. Stack differs from site0:
   `mat+0x1665b -> mat+0x87299`, skipping the `mat+0x19235 | mat+0x18997 |
   mat+0x165f1` chain the convar reads. Hitch source, not throughput.
6. **`r2\cfg\autoexec.cfg` DOES NOT EXIST.** The launcher .bat claims
   `r_no_stalls 1` is set permanently there. It is not — the launch argument is
   the only thing disabling site0. Any other launch path brings it back
   (215-445 ms per 5 s window during load).

---

## 7. Traps that bit this session

**The disk filled to 0 bytes.** A 242 MB nsys report becomes a **4 GB** SQLite
once sampling is on (~80M callchain rows). Two of those plus stale
`nsys-report-*.qdstrm` files took C: to zero, which failed an export with
`database or disk is full` and may have contaminated run-to-run variance before
that (texture streaming, 20 MB/run logs, shader cache). **Export, extract, delete
the .sqlite in the same pass.** The `.nsys-rep` is the irreplaceable artefact; the
`.sqlite` is regenerable.

**A saturated counter cannot show progress.** Register Count=255 is the cap. See
sec 2. The same logic applies to any metric pinned at its maximum.

**Sampling only catches RUNNING threads.** Absence of samples means blocked, not
free. And "which module owns the instruction pointer" is NOT "which module is the
bottleneck" — that conflation is what made the thread-pool fix look obvious and
cost a build.

**Check that a number is attributable to your change before acting on it.** The
4294.9 MB PMA buffer was blamed on options that, when removed, left it unchanged.

**Control your comparisons.** The in-gap module histogram looked damning until it
was compared against out-of-gap: `ntoskrnl` was 92.5% in-gap vs 93.4% out — ratio
0.99, i.e. no signal at all. Only the ratio column was informative.

**`[Perf.GpuPass]` per-pass numbers still need `rtx.perfGpuStageSerialize`**
(currently False) — BOTTOM_OF_PIPE stamping migrates cost to whichever later mark
the GPU reaches late. It happened to agree with nsys on `gb_primaryRays`; do not
generalise that to the other stages.
