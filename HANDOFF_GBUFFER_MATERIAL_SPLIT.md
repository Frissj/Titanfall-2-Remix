# Handoff: split the gbuffer uber-shader by material class

Written 2026-07-29 ~01:10. Everything below is measured unless it says INFERENCE.
The job is item **#4**: make each gbuffer dispatch carry only the code it needs.

**Read sec 3 before writing any code.** Six things are already refuted, each by a
measurement, and three of them were refuted *this session* after being believed.

---

## 0. The question

Frame is **57-61 ms / 16.4-17.5 fps** at 1920x1080. DLSS is deliberately at
`rtx.qualityDLSS = 5` (FullResolution, no upscaling) — this is a **stress test,
not a misconfiguration**. Do not "fix" it. Do not suggest lowering it.

The CPU is not the problem and this is settled:

```
[Perf.Frame]      totalInjectUs=3853        (3.85 ms of CPU injection/frame)
[Perf.Gpu]        fenceWaitMs=60.90  idleMs=0  reapMs=0.16
[Perf.SubmitGap]  gapMsPerFrame=61.09  inSubmitMsPerFrame=0.025
[Perf.PresentWall] insideMsPerFrame=0.0008
```

The CPU sleeps 60.9 of 61.1 ms waiting on a fence. There is **no GPU idle** —
`idleMs=0`. Earlier handoffs describing "25-34 ms of GPU idle" are not
reproduced; that was CPU-side fence wait seen from the wrong end.

GPU pass budget, `[Perf.GpuPass]`, per frame:

| pass | ms |
|---|---|
| upscaler | 19.8 |
| **gb_primaryRays** | **10.3** |
| restir | 10.3 |
| endFrame | 3.3 |
| prepScene | 2.6 |
| postComposite | 2.0 |
| pt_rtxdi | 1.8 |
| *outsideRt* | 4.7 |
| **total** | **52.4 of 61.1 ms (94% accounted)** |

---

## 1. Why gb_primaryRays is slow

It is **latency-bound, not throughput-bound**. From the Nsight multi-pass
capture (`Desktop\tf2_nsys\tf2_20260728_235240`, 362 MB, complete):

```
SM 19.0%   RTCORE 12.7%   L2 48.8%   VRAM 33.7%
CS occupancy 29.6%   Unallocated warps in active SMs 66.8%
```

Nothing saturates. The RT cores are nearly idle. The GPU is stalling, not
computing.

Top warp-issue stalls:

| range | #1 | #2 |
|---|---|---|
| Primary Rays | Long Scoreboard L1 **7.50%** | No Instruction **4.20%** |
| Integrate Indirect | No Instruction **10.29%** | Long Scoreboard L1 10.19% |
| Ray Reconstruction | Long Scoreboard 5.24% | Not Selected 4.58% |

Nsight's expert system fires `Warp Launch Stalled by CS Register Count` with
`SM Resource Usage Registers CS = 97.3%` and `Active Warps per Cycle CS = 7.78`
(of 48). Its own caveat is *"occupancy increase helps only latency-limited
workloads (Top Throughput% < 85%)"* — top throughput here is **8.2%**, so
occupancy **is** the correct lever.

Compiled shader facts, `[Perf.Shader]`, measured in RayQuery/compute mode:

| shader | regs | binary | spill | shared |
|---|---|---|---|---|
| `gbuffer_rayquery_nrc_wboit_opaque_translucent` | 255 | **808 KB** | 144 B | 6656 |
| `..._occ256` | 255 | 807 KB | 144 B | 13312 |
| `integrate_direct_rayquery` | 128 | **816 KB** | **496 B** | 25088 |
| `gbuffer_psr_rayquery_nrc_wboit_opaque_translucent` | 255 | 396 KB | 144 B | 3584 |

Register math: 255 regs x 32 threads = 8160 regs/warp, 65536 / 8160 = **8
warps/SM** (~17%). Shared memory at 13312 B per 256-thread block allows 4 blocks
at a 64 KB carveout, 7 at 100 KB. **Registers bind 4-7x harder than shared
memory** — do not chase shared memory first.

Two symptoms, one cause: 255 registers gives no latency hiding, and an 800 KB
binary cannot stay in an instruction cache (tens of KB), so divergent paths
refetch through L2 — which is why L2 sits at 48.8% while SM sits at 19.0%.

---

## 2. What #4 is

Stop compiling one shader that contains every material path. Classify pixels by
material class, then issue **one indirect dispatch per class** over a compacted
pixel list.

Why this is not the WBOIT experiment again: removing a feature from a single
uber-shader lets the compiler respend the freed registers on the surviving
paths — measured, sec 3. Separate dispatches mean the opaque shader's worst path
**never contains translucent code at all**, so there is nothing to respend. It
also makes every warp materially coherent by construction, which is the benefit
SER would have bought, without needing SER.

### Files

| file | what |
|---|---|
| `src/dxvk/rtx_render/rtx_pathtracer_gbuffer.cpp` | `dispatch()` at **:301**, the actual dispatch at **:590**, `[Perf.GbDispatch] rayDims` at **:513** |
| same, **:50-95** | the permutation include block — `*_opaque_translucent` names are the uber-shaders to split |
| same, **:744 / :750** | SER raygen variants (`gbuffer_psr_raygen_ser_nrc_wboit` etc.) already exist and are selected by option |
| `src/dxvk/shaders/rtx/pass/gbuffer/gbuffer.slang` | 19.5 KB, the shader body |
| `src/dxvk/rtx_render/rtx_pathtracer_gbuffer.h` | `RaytraceMode { RayQuery=0, RayQueryRayGen=1, TraceRay=2 }` at **:33** |

### Steps

1. **Classification pass.** Write per-pixel material class, then compact into
   per-class pixel index lists plus `VkDispatchIndirectCommand` args. Classes to
   start with: opaque, translucent. PSR is already a separate permutation family
   (`gbuffer_psr_*`) — leave it alone in v1.
2. **Split the permutation.** Build `..._opaque` and `..._translucent` variants
   alongside the existing `..._opaque_translucent`. Keep the combined one behind
   an option so A/B is one config flip, not a rebuild.
3. **Dispatch indirect per class** from `dispatch()` (:301), reading each class's
   pixel list. `ctx->dispatch(...)` at :590 becomes N indirect dispatches.
4. **Measure** — see sec 4, and note the instrumentation caveat there.

### Risks

* Classification costs a pass plus memory traffic. It must come out of the 10.3 ms,
  not on top of it.
* Tail effects: a class with few pixels wastes a dispatch. Consider merging
  classes below a threshold.
* `rtx.perfGbufferBlockThreads = 256` and the `occ256/384/512` variants interact
  with this — one block is currently 8 warps, i.e. exactly the whole SM budget at
  255 regs. Smaller blocks may schedule better once the shader is smaller.
* `rtx.enablePSRR` / `rtx.enablePSTR` are both True, so PSR paths are live.

### Success criteria

* `gb_primaryRays` below **10.3 ms** (`[Perf.GpuPass]`)
* `No Instruction` stall below **4.20%**, `Long Scoreboard L1` below **7.50%**
* per-permutation binary well under 808 KB, spill under 144 B

---

## 3. REFUTED — do not re-derive

* **"Removing WBOIT relieves the register wall."** WRONG, and it made things
  slightly worse. `gbuffer_rayquery_nrc_opaque_translucent` still reported
  Register Count=255, the same 144 B spill, and a **larger** binary (833024 vs
  826880). Register Count is at the hardware cap and is therefore a **pinned,
  blind gauge** — it cannot show partial progress. Judge by spill, binary size
  and runtime.
* **"TraceRay is 4.3x worse."** NOT REPRODUCED on the current build
  (2026-07-29). `gb_primaryRays` 10.27 -> 10.59 ms (+3%), GPU pass total
  52.46 -> 52.40 ms, frame 61.1 -> 58.3 ms. The old "292 ns/ray vs 68" note is
  stale. **`rtx.renderPassGBufferRaytraceMode` is currently 2 (TraceRay) and
  stays there.**
* **"SER isn't on for Integrate Indirect."** WRONG. `VK_NV_ray_tracing_invocation_reorder`
  enabled, `[RTX info] Shader Execution Reordering: enabled`, pass runs
  `Trace Ray (RGS)`. It is on — **and that pass still has the worst
  `No Instruction` stall in the frame (10.29%)**. SER fixes divergence, not code
  size. This is the strongest evidence that #4 beats #5.
* **"Deferred pipeline creation is why RT shader stats are empty."** WRONG.
  Forced `rtx.pipeline.useDeferredOperations = False`, got `deferred=0 result=0`
  and still `execCount=0`.
* **"RT shader stats are obtainable via VK_KHR_pipeline_executable_properties."**
  NO. `countRes=0` (VK_SUCCESS) with `execCount=0` on every RT pipeline.
  NVIDIA does not enumerate executables for ray-tracing pipelines. The extension
  works fine for compute (78 `cs=` lines the same run) and the capture bit *is*
  set at creation. The query code added to `dxvk_raytracing.cpp` is correct —
  keep it, it costs nothing and now logs `NOSTATS reason=` instead of failing
  silently.
* **"The log filter is eating the new lines."** No. `log.cpp` filters by prefix;
  the only `[Perf.*` entry is `[Perf.GeoChurn]`, and no substring filter matches.
  Checked, not assumed.

---

## 4. Measurement caveat — read this or you will work blind

With `renderPassGBufferRaytraceMode = 2` the gbuffer is an RT pipeline, and per
sec 3 **there is no Register Count, Binary Size or spill figure for it in any
log, ever.** `[Perf.Shader]` only emits `cs=` lines.

So judge #4 on:

* **runtime** — `gb_primaryRays` in `[Perf.GpuPass]`
* **stalls** — `No Instruction` / `Long Scoreboard L1` from a Nsight capture

If a compiled-size number is specifically needed, take **one** measurement run at
mode 0 and put it back to 2. Do not leave it at 0.

---

## 5. Tooling

`Capture TF2 (Nsight Graphics auto).bat` — double-click, no arguments. Current
defaults are the working ones; do not "tidy" them:

* `-EventBufferKb 8000` travels as **`GPUTRACE_EVENT_BUFFER_MEMORY_KB`**, an
  undocumented target-side env var found by scanning `WarpVizTarget.dll` for
  `^GPUTRACE_[A-Z0-9_]+`. The documented CLI flag
  `--allocated-event-buffer-memory-kb` is **inert** on 2025.4.
* `-LimitToFrames 12` — multi-pass requires frame-based tracing
  (`"Multi-pass metric sets are only supported with frame-based tracing"`).
* `-PerArchConfig $false` — multi-pass needs the bare `--multi-pass-metrics`
  flag **and** frame-based tracing. The `--per-arch-config-path` JSON does not
  work; its `multi-pass-metrics` key is ignored.
* ngfx's console is captured to `ngfx_console.log` in each run folder. Nsight
  writes no log of its own — that file is the only record of TARGET WARNINGs.

To read expert analysis: open the `.ngfx-gputrace`, **Trace Analysis** tab,
`Export...` → `analysis.yaml` (~5.5 MB, every rule with explanations, suggestions
and metric values). Parsing that file is far faster than driving the GUI.

Nsight 2026.3 does not help: it cannot attach to this game (its interception DLLs
statically import System32 `dxgi.dll`), and its new per-shader occupancy row
answers "which shader limits occupancy", which is already known.

---

## 6. Also open

* **`integrate_direct_rayquery` attribution.** 816 KB, 496 B spill, 25088 B
  shared — the worst-built shader in the frame — but only 2.01 ms in the trace.
  Either it is genuinely cheap, or its cost hides inside `Integrate Raytracing`
  (20.61 ms), which contains it. Resolve from the trace before deciding whether
  it is a 2 ms item or a 20 ms one. No code change needed to answer.
* **#5, SER in the gbuffer.** Requires TraceRay (already set). Both
  `shouldReorder()` bodies in
  `src/dxvk/shaders/rtx/algorithm/geometry_resolver_state.slangh` at **:183** and
  **:366** unconditionally `return false` — that is the
  "(Note: Hard disabled in shader code)" the option comment in `rtx_options.h`
  refers to. `rtx.enableShaderExecutionReorderingInPathtracerGbuffer = True` is
  already set in `rtx.conf`, so this is one shader edit. Do it **after** #4, and
  expect little: see sec 3.
* **2x instrument disagreement, unresolved.** Nsight says Primary Rays is
  20.10 ms; `[Perf.GpuPass]` timestamps say 10.27 ms. Same pass. Nsight's frame
  is 71.2 ms against 61.1 un-instrumented, which does not account for 2x. Both
  agree on ranking, so it does not block #4, but do not mix the two numbers in
  one budget.
