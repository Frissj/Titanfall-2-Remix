# HANDOFF — the frame is now GPU-bound; target is `postComposite`

Tree: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
Log:  `C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx-remix\logs\remix-dxvk.log`
Conf: `C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx.conf`
Date: 2026-07-26 (evening). HW: RTX 4080 Laptop, Ryzen 9 7945HX.
1920x1080 native, static camera, standing still.

**Supersedes `HANDOFF_TF2_OUTSIDE_RT_CPU.md`. That document's core premise is now
inverted, not refined.** It said the CPU was the bottleneck (`outsideRtMs` 63.1 ms,
`gpuIdleMs` 46.9 ms, "GPU starved waiting on CPU") and named `tail_prepScene` as
the target. Both halves are gone: `outsideRtMs` is now **0.007–0.06 ms** and
`gpuIdleMs` is **0**. Do not resume its plan.

---

## 1. Where the frame goes now

~156 ms/frame (`presentFrames` 30–35 per 5 s ≈ 6.5 fps). The GPU **is** the frame:

```
[Perf.GpuPass] totalMs=149–162   marks=28..28/28 ALIGNED
  postComposite   ~94      <-- THE TARGET, 60% of the frame
  onFrameBegin    ~24
  pt_gbuffer      ~13.4
  finalBlit       ~7–10
  nrc             ~5.5
  pt_rtxdi        ~3.3
  gb_reflectionPSR ~1.8
  (all others <1 ms each)
```

CPU side is finished — total ~5 ms of a 156 ms frame:

| counter | old handoff | now |
|---|---|---|
| `outsideRtMs` | 63.1 ms | **0.007–0.06 ms** |
| `totalInjectUs` | 27–54 ms | **5.1 ms** |
| `entry_tailToBranch` | 18–42 ms | **3.3 ms** |
| immCtx `totalMs` | 74.3 → 3.95 ms | **1.8 ms** |
| `gpuIdleMs` | 46.9 ms | **0** |

**Do not optimise CPU code.** `uploadSurfaceData`, the merge loop, `cull`, surface
dirty-tracking — all are now worth at most a few hundred µs of a 156 ms frame.

---

## 2. FIRST TASK: split `postComposite`

Its GPU span is the marks at `rtx_context.cpp:2572` → `:2581`, which contains
**exactly two calls**:

```cpp
getCommonObjects()->getTextureManager().copySamplerFeedbackToHost(this);
dispatchObjectPicking(rtOutput, downscaledExtent, targetImage->info().extent);
```

(plus a screenshot path that is inactive unless capturing.)

`copySamplerFeedbackToHost` is the prime suspect: a GPU→CPU readback feeding
texture streaming's mip decisions. A per-frame full-resource copy plus its implied
sync is the right shape for ~94 ms. `dispatchObjectPicking` should be trivial and
is typically only meaningful when a pick is requested.

Two ways to settle it, cheapest first:

1. **Look for a config gate on sampler feedback.** If one exists, flipping it off
   tests the whole hypothesis in one run with no rebuild.
2. **One `markGpuStage()` between the two calls.** Safe for the sweep this time —
   `postComposite` sits *after* `gb_primaryRays` in `kStageNames`, so
   `kGbPrimaryRaysSlot` does not move. You still must insert the name in the
   correct position (see §5, trap 2).

`onFrameBegin` (~24 ms) is the second target and is unexplained. `gpuDrain`
already ruled out "the GPU is draining earlier work" — it reads <0.01 ms.

---

## 3. The three wins that got us here (do not undo)

1. **`r_no_stalls 1` in `r2\cfg\autoexec.cfg`.** `materialsystem_dx11`'s per-frame
   busy-wait fence was burning **2,890 ms of spin per 5 s window — 58% of wall
   clock** — at 426k–567k `GetData` calls/frame. `[Perf.SyncSite]` went
   `bursts=15..27` → **0**. Roughly 4.6 → 6.5 fps. There is also
   `Launch TF2 (r_no_stalls).bat` + a pinnable desktop shortcut for A/B (a launch
   arg overrides the cfg).
2. **Release build.** `build-remixDx11Release.bat` now builds `_Comp64Release`
   with `--buildtype=release`. **5x on the whole prepScene stage** (16.3 → 3.3 ms);
   `uploadSurfaceData` 5,418 → **361 µs (15x)**. Cause was `meson.build:27`
   `b_vscrt=from_buildtype`: Debug links the debug CRT, so
   `_ITERATOR_DEBUG_LEVEL=2` put bounds/iterator checks on every `std::vector`
   index — precisely the two hottest loops. PDBs survive (`meson.build:79-85`
   adds `/Zi /DEBUG` for release explicitly).
3. **`rtx.logGeomDiag=False`** gating the geometry probes. `cull` 6.6 ms → **2.0 ms**
   and its 24 ms spikes are gone. `[SpikeRB]` was the worst offender: two
   `copyBuffer`s + a **full CPU triangle walk** + an unfiltered log write, every
   frame, from a closed investigation.

---

## 4. Instrumentation now available

**`rtx.logPrepSceneSplit`** (~free, safe during timing runs; `True` in rtx.conf).
Emits on the same frame, `fid%10==5`, so all lines read together:
- `[Perf.PrepScene]` — 10 buckets (`cull`/`tlas` are now split)
- `[Perf.Merge]` — `mergeInstancesIntoBlas` split, incl. `tcFlush`
- `[Perf.BuildBlas]` — `upload/omm/census/createBufs/scratch/build`
- `[Perf.Alloc]` — Vulkan allocator: `memAlloc/typeLock/vkAlloc/freeChunks/slice`,
  each with total + **max single occurrence** + count (`dxvk_alloc_probe.h`)
- `[Perf.Stall]` — `wallUs/cpuUs/blockedUs/busyPct/faults/wsMB`, i.e.
  **blocked vs busy** (`rtx_cpu_stall_probe.h`)

Cross-checks are built in: each split's buckets must sum to its parent's field on
the same frame (verified — the residual is the `Logger::warn` itself, ~100–200 µs).

**Always on, cheap:**
- `[Perf.Frame] cpuCalNs= cpuSlowX=` — effective CPU speed, self-normalised to the
  fastest sample in the session. **Limitation: detects drift, not a persistently
  low clock** (see §6).
- `[Perf.GpuPass] marks=N..M/28 ALIGNED|*SHIFTED*` — validates that the positional
  `kStageNames` mapping is actually aligned. Read this before trusting any stage
  label.

---

## 5. TRAPS

1. **Do NOT copy `m_isSubsurface` in `RtInstance`'s copy ctor. It freezes the game.**
   Tried this session; hard GPU hang during load. `isSubsurface()` is not a
   reporting flag — it gates BLAS bucket compatibility
   (`rtx_accel_manager.cpp:209`), **also** pushes into `m_mergedInstances[Tlas::SSS]`
   (`:2184`), and reserves an extra `PointInstancerBatch` with `tlasType=Tlas::SSS`
   (`:6193`), which shifts every per-type byte offset
   `dispatchPointInstancerCulling` writes into `m_vkInstanceBuffer`. It **is** a
   real latent divergence (`createInstanceCopy` never calls `updateInstance`, so
   clones report `false`), but fixing it needs the SSS slot accounting and TLAS
   build audited first. Full note is in the copy ctor.
2. **`kStageNames` is positional.** `accumMs[i]` is named `kStageNames[i-1]`.
   Inserting a mark shifts every later slot, and **four** places must move
   together: the table, `kGbPrimaryRaysSlot`, the `strcmp` guard's index, and its
   message. Marks inserted *after* `gb_primaryRays` don't move the slot. There is
   a runtime guard that shouts if they drift; `marks=` catches count mismatches.
3. **Release-only build blockers.** The `sizeof` `static_assert`s only compile
   under `DEBUG_OPTIMIZED || NDEBUG`, so Debug never saw them.
   `RtInstance == 792` (was 768 — three members had accumulated).
   `RtLight == 264` has not tripped yet but may. **Audit the copy ctor / `copyFrom`
   against the members before changing the number.** Also `werror=true`: with
   asserts gone, variables used only inside `assert()` become unused → errors.
4. **`NDEBUG` removes the asserts**, including
   `assert(offset - oldOffset == kSurfaceGPUSize)` in `RtSurface::writeGPUData`
   and the range check in `writeGPUHelperExplicit`. A surface-packing overrun that
   Debug would have caught now silently corrupts.
5. **Both build scripts deploy to the same `%GAME_RUNTIME_DIR%`** — the game runs
   whichever config was built **last**.
6. **`log.cpp` filters tags at emit, not at format.** `[SpawnGeomDiag.` and
   `[TlasCensus]` are filtered; their `str::format` still runs at full cost. A new
   probe whose tag is on that list will appear to do nothing.
7. **`/O2` risk is unresolved.** `debugoptimized` once made projection scanning
   fail silently (`hasEverFoundProj` stayed 0, CamMgr never latched Main). Release
   is also `/O2`. It has not reproduced in Release so far, but if the screen goes
   black check that first. The two `NDEBUG`-gated sites are compile-time only and
   are **not** that bug.

---

## 6. REFUTED this session — measured, not argued. Do not re-run these.

| hypothesis | verdict |
|---|---|
| `[TlasCensus]` flush owned the spikes | **REFUTED** — 385 µs, 3% of merge |
| CPU throttling caused the 3x bucket inflation | **REFUTED** — `cpuSlowX` pins at **1.00** through gameplay; the 3x readings were all load-phase |
| "uniform 3x degradation over 20 s" | **REFUTED** — averaging artifact of *migrating* single-bucket spikes |
| the Vulkan allocator caused the migrating spikes | **REFUTED** — `[Perf.Alloc]` reads 36–63 µs against a 16 ms spike; `typeLock`/`vkAlloc`/`refills` all **0** |
| the thread was blocked, not busy | **REFUTED** — `busyPct` 95–99%, `faults` 0 |
| `createBufs` was the `buildBlases` leaf | **REFUTED** — 248 µs (4%); `upload` was 90% |
| `totalMs` > frame period ⇒ broken accounting | **REFUTED** — `totalMs × samples` = 97–108% of the window; it is a mean over **RT frames only**, diluted in `presentFrames` by cheap frames |
| stage labels shifted by conditional marks | **REFUTED** — `marks=28..28/28 ALIGNED`, every window |
| `gpuDrain` would be large (GPU draining earlier work) | **REFUTED** — <0.01 ms |
| RTX Mega Geometry would help | **REFUTED on the numbers** — PTLAS targets TLAS build (`buildTlas` = **60 µs**), CLAS targets BLAS build (`bBuild=0`, zero rebuilds). Aims at what is already free, needs extensions absent from this tree, and the GPU is the bottleneck anyway |

Two of my own structural objections (the accounting and the label shift) were
wrong, and acting on either would have aimed work at a phantom. **Validate the
instrument before distrusting it, and distrust reasoning before measurement.**

---

## 7. Open questions

- **`postComposite` ~94 ms: which of the two calls?** §2.
- **`onFrameBegin` ~24 ms** — unexplained; `gpuDrain` ruled out GPU drain.
- **Unreconciled:** an earlier run (21:51) showed `onFrameBegin=91` /
  `postComposite=5.7`, the current one 24 / 94. The workload genuinely differed
  (`volumetrics` 6.9→0.1, `nrc` absent→5.5, `upscaler` present→absent), but that
  is not a complete explanation. The current numbers are validated, reproducible
  across three consecutive windows, and internally consistent — act on those.
- **`finalBlit` 7–10 ms** — the old handoff measured ~70 ms and called it a
  GPU-wait. Much smaller now; revisit only after `postComposite`.

---

## 8. Verification

Read `[Perf.GpuPass]` for the GPU breakdown; `[Perf.PrepScene]` / `[Perf.Merge]` /
`[Perf.BuildBlas]` / `[Perf.Alloc]` / `[Perf.Stall]` for the CPU side.
`presentFrames` per 5 s is still the most trustworthy single number.

Sanity checks that must hold, or something regressed:
- `[Perf.GpuPass]` says **`ALIGNED`** — otherwise stage labels are meaningless
- `[Perf.SyncSite]` `bursts=0` (the engine fence is still open)
- `[Perf.Block]` `gpuSyncMs` / `csSyncMs` are 0
- `[Perf.Entry]` `GetData` is **not** a top entry
- `[Perf.Alloc]` `census=0` and `tcFlush=0` (the geometry probes are still gated)
- `[Perf.CpuCalib]` ~2496 cycles/µs — see below

**The CPU clock cap is deliberate.** `cyclesPerUs=2496` (~2.5 GHz) on a 7945HX
that specs ~5.4 GHz boost. **The user caps it intentionally because the machine
crashes otherwise. Do not suggest unlocking it.** It doubles every CPU number
here versus a stock machine, and `cpuSlowX` reads 1.00 regardless because it
normalises to the fastest sample in the session.

**Camera state matters.** All numbers are a static camera standing still. Compare
like with like, and prefer A/B within one run — `cpuSlowX` and `marks=` exist so
you can tell when two samples are not comparable.
