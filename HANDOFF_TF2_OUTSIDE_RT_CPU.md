# HANDOFF — the CPU frame, `outsideRtMs` / `tail_prepScene`

Tree: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
Log:  `C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx-remix\logs\remix-dxvk.log`
Conf: `C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx.conf`
Date: 2026-07-26. HW: RTX 4080 Laptop (AD104, max clock 3105 MHz), driver 610.62.0.
1920x1080 native, DLSS OFF, static camera, standing still.

**Supersedes `HANDOFF_TF2_PRIMARY_RAY_COST_V4.md` as the active target.** That
document is not wrong about `gb_primaryRays`; it is aiming at the wrong thing.
See §5.

---

## 1. The target, already located

`tail_prepScene` is **99.9% of the entire CPU tail**. From `[Perf.Frame]`, three
consecutive logs:

```
totalInjectUs       27248   32258   54088
entry_tailToBranch  18514   26469   42491
tail_prepScene      18484   26436   42447   <-- THIS
tail_gpuIdle            1       1       2
tail_preTex            10      10      13
tail_texUpload         11      12      14
tail_preScene           3       3       5
```

Everything in the tail except `prepScene` is **microseconds**. There is no
search to do at this level — do not re-split `entry_tailToBranch`, it is solved.
`tail_prepScene` is `SceneManager::prepareSceneData()` in
`src/dxvk/rtx_render/rtx_scene_manager.cpp`.

Note the variance: 18.5 / 26.4 / 42.4 ms across three samples ~5 s apart with a
STATIC camera and near-identical scene counts. That 2.3x swing on unchanging
input is itself a lead — something in there is not proportional to scene size.

---

## 2. Frame context — read before optimising anything

```
frame            ~104 ms   (presentFrames median 48 per 5 s = ~9.6 fps)
outsideRtMs        63.1 ms  <- CPU, the bottleneck
GPU totalMs        49.2 ms
immCtx totalMs      3.95 ms <- ALL D3D11 API calls. Already free, leave it alone.
gpuIdleMs          46.9 ms  <- GPU starved waiting on CPU
gpuSyncMs           0
csSyncMs            0
submits            10
```

GPU pass medians (n=8 windows):

```
gb_primaryRays  21.3   pt_integrate 10.4   prepScene(GPU) 5.3   denoise 4.5
nrc              2.8   pt_rtxdi      1.4   finalBlit      0.8   composite 0.45
(12 further passes, ~1.4 ms combined)
```

`prepScene` appears in BOTH lists: 5.3 ms of GPU time, and 18-42 ms of CPU time.
They are different halves of the same stage. The CPU half is the target.

**Accounting caveat, stated up front:** `outsideRtMs` (63.1) and `totalInjectUs`
(27-54) do not agree, because they are measured on different bases —
`outsideRtMs` is derived from GPU stage timers in `rtx_context.cpp:6076`,
`totalInjectUs` is CPU wall time across the inject. Do not treat either as the
whole CPU frame. `tail_prepScene` dominating `entry_tailToBranch` is solid; the
exact fraction of the 104 ms frame it represents is not.

---

## 3. FIRST TASK: decouple the `[Perf.PrepScene]` gate

The sub-split you need **already exists** at `rtx_scene_manager.cpp:4176` and
prints nine sub-stages:

```
ps_lightMatch  ps_gc  ps_setup1  ps_instSetup  ps_merge
ps_accelLight  ps_surfMat  ps_cullTlas  ps_tail
```

Its condition is `RtxOptions::logSurfaceCoverage()` AND a coarse frame throttle.
Both halves need changing.

**The option half is a trap.** `logSurfaceCoverage` costs **~104 ms/frame** — it
arms the 52-atomic-per-primary-hit coverage block inside
`opaqueSurfaceMaterialInteractionCreate` (rtx.conf documents this;
`gb_primaryRays` goes 27 -> ~140). Turning it on to read the PrepScene split
would quadruple the frame and make every number it prints meaningless. Give
`[Perf.PrepScene]` its own gate: a new
`RTX_OPTION("rtx", bool, logPrepSceneSplit, false, ...)`, or an env read like
the `GetFlushTuning` idiom at the top of `d3d11_context_imm.cpp`.

**The throttle half is too coarse.** Make it fire at least once every ten
frames — `(frameId % 10u) == 5u`. At ~9.6 fps the current gate samples roughly
once per three seconds, and §1 shows this stage swinging 2.3x between samples on
a static camera; three points cannot show a distribution. The log line is one
`Logger::warn` of ~12 short fields, so at ten-frame granularity it is about one
line per second — nothing next to the `[NrcBounds]` line already printing on
every single frame.

This is a small change and it is the prerequisite for everything else in this
document. Until it lands you are guessing about which of the nine sub-stages
owns the 18-42 ms, and this codebase has an unbroken record of punishing
guesses (§5, §6).

---

## 4. What to look at once the split prints

Rank the nine `ps_*` values, then attack the largest — but check these two
properties first, because the 2.3x variance in §1 says at least one of them is
mis-scaling:

- **Does it scale with `inst=` / `surf=`?** Both are printed on the same line
  (`m_instanceManager.getActiveCount()`, `m_accelManager.getSurfaceCount()`).
  A stage that grows while instance count is flat is doing per-frame work that
  should be cached or incremental.
- **Is it rebuilding state that did not change?** The stage name is
  `prepareSceneData` — "rebuild all GPU scene buffers". With a static camera and
  ~450 instances, most of that state is identical frame to frame. Anything
  unconditional in there is the obvious candidate.

`[IdxStashPool]` prints alongside it (`created` / `reused` / `agedOut` /
`freeBytes`). Its own comment states the acceptance test: **`created` must
FLATTEN after warmup.** A `created` that keeps climbing with `reused` near zero
means the index-buffer pool is not recycling, which would be per-draw device
allocation churn landing squarely in this stage. Check that first — it is free,
already instrumented, and has a stated pass/fail.

---

## 5. Do not re-attack the GPU. Here is why.

`gb_primaryRays` is 21.3 ms and 43% of GPU time, which makes it look like the
prize. It is not, for two independent reasons.

**It is behind a 47 ms idle gap.** The GPU already waits half the frame for the
CPU. Shaving GPU work cannot show up in fps until `outsideRtMs` comes down. I
spent most of 2026-07-26 learning this the hard way.

**Every shader-side lever is refuted.** From V4 plus this session, independently
measured null: occupancy (three separate ways), code volume / instruction cache,
L1 capacity, geometry fetch, all 8 material ablations, POM, thin film, sampler
aniso, mip bias, coherent fetch, `primaryRayMaxInteractions`, zero-init hoist,
`-O3`, decal-bin local memory. The ONLY confirmed component is texture gradients
at -1.39 ms (-6%).

If GPU work becomes the bottleneck later, the lever is **not** shader
micro-optimisation, it is `rtx.dlssPreset = 1`. `gb_primaryRays` is genuinely
per-ray (`t = 0.27 + 11.45 x Mpix`, which predicts the measured 24.0 ms at
2.0736 Mpix), so DLSS Quality would take it to ~10.8 ms. It is off in rtx.conf
solely to hold ray count constant for the V4 measurement campaign — that
campaign is over and the condition in that comment is met.

One genuinely untried shader item, for completeness: `resolve.slangh` calls
`surfaceInteractionCreateInto<SurfaceGenerateTangents>` unconditionally, but
tangents are only needed for normal-mapped materials and `SurfaceIgnoreTangents`
already exists (`visibility.slangh` uses it). Never measured. Set expectations
low given the null rate above.

---

## 6. DEAD ENDS — built, measured, refuted, do not repeat

| attempt | result |
|---|---|
| **Shader code volume / register pressure** | **REFUTED.** Compiling out the gbuffer debug views + perf ladders took the kernel 826880 -> 350848 bytes and 255 -> 168 registers. `gb_primaryRays` did not improve (23.89 vs 23.79). At the stock 128-thread block it was 20% WORSE (28.8 ms): 168 regs lets a 3rd block resident, each block drags 33280 B of driver shared memory, 3 x 33280 = 100 KB pins the Ada L1/shared carve-out at its ceiling and leaves ~28 KB of L1. Switches are back at their defaults (True) in `compile_shaders.py`; see `rtx/utility/build_switches.h` for the full table. |
| **`rtx.fastSignalEventQueries`** (Remix-side `r_no_stalls`) | **REFUTED AND REVERTED — the code no longer exists.** Reported EVENT queries signalled without testing the GPU event, to drop the game convar. It worked (`[Perf.SyncSite]` bursts=0, GetData 0.005 ms) but the stall MOVED into `Map()`: 67-82 ms/frame, `gpuSyncMs` 0 -> 77, `presentFrames` 35 -> 32. The query's RETURN VALUE only feeds a spin loop, but the TIMING of the wait is load-bearing for the engine's buffer reuse. `engine.dll` evidently reads `r_no_stalls` too and adapts; DXVK cannot do that from outside. **The fence must be opened engine-side.** |
| Occupancy via `perfGbufferBlockThreads` 384/512 | Refuted — monotonically slower. Forces register cap -> spilling, not occupancy. |
| `-O3` on slangc | Binary grew, registers stayed 255, GPU hung. |

---

## 7. What was fixed on 2026-07-26 (do not undo)

1. **Deleted persistent User env var `RTX_FLUSH_MIN_US=5000`.** A leftover from
   an old flush sweep, applying to every run, multiplying the min flush interval
   6.7x over the 750 us code default. Restore with
   `[Environment]::SetEnvironmentVariable('RTX_FLUSH_MIN_US','5000','User')`.
2. **`d3d11_context_imm.cpp` `GetData`:** the first not-ready poll of a spin
   burst now calls `Flush()` directly instead of `FlushImplicit(FALSE)`.
   `StrongHint` never bypassed the flush TIMER, only the pending-submission
   count, so a spinning app could not get work submitted for a full interval.
3. **`[Perf.SyncSite]` diagnostic** in the same file — one stack capture per spin
   burst (~7/frame, NOT per poll), attributing bursts/polls/ms to each engine
   call site. `RTX_SYNC_DIAG=0` disables. This is what found the sync point.
4. **`r_no_stalls 1` engine-side** (`r2\cfg\autoexec.cfg` or `+r_no_stalls 1`).
   Kills the material system's per-frame busy-wait fence
   (`materialsystem_dx11.dll` `sub_180018940` / `sub_1800191E0` /
   `sub_180087D90` — `_mm_pause` loop, never an OS wait). ConVar 0x1814EDE40,
   default 0, FLAGS = 0 (no cheat/hidden). Cost: input lag, which no counter
   shows.

Session result: **~4.6 -> ~9.6 fps**, immediate-context CPU 74.3 -> 3.95 ms.

---

## 8. Verification

Read `[Perf.Frame]` for `tail_prepScene`, and `[Perf.GpuPass]` /
`[Perf.Block]` for the frame context. `presentFrames` per 5 s is the most
trustworthy single number — it is actual frames on screen, not an instrumented
sub-total, and it is what caught both wins this session.

Sanity checks that must hold, or something regressed:
- `[Perf.SyncSite]` reports `bursts=0` (the fence is still open)
- `[Perf.Block]` `gpuSyncMs` and `csSyncMs` are 0 (nothing stalling in DXVK)
- `[Perf.Entry]` `GetData` is not in the top entries

**Do not run `rtx.perfAutoSweep`** for CPU work — it sweeps shader knobs and
tells you nothing about `prepScene`.

**Camera state matters.** All numbers here are a static camera standing still.
Under movement the GPU passes, the denoiser and NRC all cost more, and NRC
convergence (which is per-FRAME, not per-second) changes the image. Compare
like with like.
