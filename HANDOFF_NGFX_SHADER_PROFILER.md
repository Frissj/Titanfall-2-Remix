# Handoff: chasing register pressure in gb_primaryRays - GPU Trace works, the shader profiler deadlocks in NVIDIA's own code

**Goal: cut frame time from ~58-63 ms. Immediate target: `gb_primaryRays`, 11 ms
and ~51 % of all real GPU work, occupancy-starved at 255 registers / 144 B spill.
Read sec 0 first - it is the question everything else serves.**

Written 2026-07-28 ~22:55. Everything below is measured unless it says INFERENCE.
Supersedes the ngfx half of `HANDOFF_GPU_IDLE_AND_NGFX.md`; the GPU-idle half of
that document is untouched and still current.

**Read sec 3 before doing anything.** Six plausible causes for the shader-profiler
hang are dead, each killed by a measurement, and each cost a run.

---

## 0. WHY ANY OF THIS EXISTS - the actual question

The goal is frame time. TF2 + Remix runs at **~16-17 fps, 58-63 ms/frame**, ~90 %
of it inside `InjectRTX`. That budget splits into two independent problems:

**(A) Half the frame is GPU idle.** Of `InjectRTX`, only **21.7-24.9 ms** is real
Vulkan execution; **25.4-33.9 ms is idle**. Biggest single item in the frame,
still unexplained. This handoff does NOT address it - see
`HANDOFF_GPU_IDLE_AND_NGFX.md` item 1. Its instrument is **Nsight Systems**
(`-CpuCtxSw process-tree`), not Nsight Graphics, and nsys is unaffected by
everything in this document.

**(B) `gb_primaryRays` costs ~11.0 ms = ~51 % of all real GPU work.** Confirmed
across four runs and two independent clocks. The shader
(`gbuffer_rayquery_nrc_wboit_opaque_translucent`) compiles to:

```
Register Count = 255   <- the HARDWARE CAP, i.e. saturated
Local Memory   = 144 B/thread spilled
=> ~8 warps/SM resident of a possible 48  ~= 17 % occupancy
```

**This is the only reason the Nsight Graphics work in this handoff happened.**
The real-time shader profiler was the instrument meant to answer *which source
lines of the `nrc + wboit + opaque_translucent` uber-shader hold those 255
registers live*, so the permutation can be shrunk. You cannot get there by
removing features and watching the counter, because 255 is saturated and shows no
partial progress - removing WBOIT was already tried and moved the count not at
all, while costing 16 fps -> 6.8-10.3.

The occupancy ladder (`rtx.perfGbufferBlockThreads`) also cannot help: occupancy
here is REGISTER-bound, not block-bound, so the active 256 rung is already the
ceiling. That is why every ladder sweep read flat.

**What the 104 MB trace already settled (whole-trace):**

```
SM Throughput 20.1 %   L2 19.4 %   VRAM 17.9 %
Unallocated Warps in Active SMs 61.0 %   CS occupancy 23.7 %
```

Nothing is saturated and most warp slots sit empty. So `gb_primaryRays` is
**not** memory-bound and **not** ALU-bound - it is occupancy/latency-starved,
exactly what a 144 B spill predicts. That confirms register pressure as the lever
and leaves source-line attribution as the one missing piece.

---

## 1. Status

| | state |
|---|---|
| Nsight Graphics **2025.4.0** GPU Trace, shader profiler OFF | **WORKS.** 111 MB trace captured 22:49 |
| Same + `-ShaderProfiler` | **DEADLOCKS** inside `WarpVizTarget.dll`. Not fixable from our side |
| Nsight Graphics **2026.3.0** | **CANNOT ATTACH AT ALL.** Uninstalled. See sec 6 |

Latest good capture:
`Desktop\tf2_nsys\tf2_20260728_224625\Titanfall2_2026_07_28_21_49_14.7879712.ngfx-gputrace`
(111 MB, multi-pass metrics ON, + `GPUTRACE_REGIMES.xls` 46 MB).

Just double-click `Capture TF2 (Nsight Graphics auto).bat`. Every setting below is
already the default; the script takes no arguments by design.

---

## 2. What the working configuration is

```
Activity          = GPU Trace Profiler
ShaderProfiler    = OFF          <- the ONLY thing that deadlocks
MultiPassMetrics  = ON           <- richer counters + Trace Analysis advice
MaxDurationMs     = 2000         (~32 frames at ~16 fps; multi-pass needs frames)
MetricSet         = Throughput Metrics   Architecture = Ada
GpuClocks         = unaltered    <- NOT 'base'; keeps times comparable to nsys
env: DXVK_ENABLE_AFTERMATH=0
     DXVK_ENABLE_PRESENT_METERING=0
     GPUTRACE_SASS_PATCHING_ON_RT_SHADERS_ONLY=1
     NSIGHT_BLOCK_ON_FIRST_INCOMPATIBILITY=0
     DXVK_PERF_EVENTS=1
```

Left ON deliberately (never implicated): shader-pipeline collection, external
shader debug info, NVTX ranges, screenshot, auto-export.

Left OFF deliberately:
* `--time-every-action` - NVIDIA's own text says it costs performance. The whole
  investigation is millisecond-accurate pass costs; a flag that perturbs them
  corrupts the thing it is meant to improve.
* `--set-gpu-clocks base` - repeatable counters, but pins the GPU below boost and
  silently makes every time incomparable to the nsys traces and in-game fps.

---

## 3. REFUTED - do not re-derive

The shader-profiler hang. Each of these was believed, tested, and killed.

* **"Aftermath's device diagnostics fight the SM sampler."** WRONG, and this one
  was believed for most of the session. `dxvk.enableAftermath` was hardcoded
  `true` and chains `VkDeviceDiagnosticsConfigCreateInfoNV` (SHADER_DEBUG_INFO |
  AUTOMATIC_CHECKPOINTS | RESOURCE_TRACKING | SHADER_ERROR_REPORTING) into
  `vkCreateDevice`. Made config/env-driven (sec 5). **Proof it took effect:**
  `VK_NV_device_diagnostics_config` and `VK_NV_device_diagnostic_checkpoints`
  are ABSENT from the "Enabled device extensions" block of `remix-dxvk.log`.
  It still deadlocks. Note the earlier runs that "tested" this were meaningless -
  the env var was ignored by a binary predating the change.
* **"ngfx host-side shader collection is the cost."** WRONG.
  `--disable-collect-shader-pipelines` - log still froze at the same byte offset.
* **"Line-table requests for source correlation."** WRONG.
  `GPUTRACE_DISABLE_LINETABLE_REQUESTS=1` - identical 142 s block. Reverted,
  because it costs source-line mapping and buys nothing.
* **"The SM sampling rate is too aggressive."** WRONG. `-PcSamplesPerPmInterval
  4096` moved `Sample duration` from **3013 ns to 266830 ns** (88x) and
  `Warp State Sampling Interval` 1024 -> 512. Still blocks. The knob works; the
  rate is not the problem.
* **"GPU performance counters need elevation."** WRONG. Re-ran elevated - still
  blocks. (`RmProfilingAdminOnly` being unset is NOT evidence of anything; the
  working traces export `BASE_UNLOCKED` unelevated.)
* **"It is a slow operation, not a deadlock."** WRONG *and* the reverse was also
  wrong at different moments - see sec 7, this cost the most time.

---

## 4. The actual cause

`-ShaderProfiler` deadlocks inside NVIDIA's instrumentation. Stack of the blocked
thread, captured elevated 22:38 (see sec 8 for the tool):

```
tid 29488  cpu=5.42s  parked in an ntdll wait
   ntdll+0x6019b
   ucrtbase+0x50d9
   WarpVizTarget.dll+0x5446c8
   MSVCP140.dll+0x123e0            <- C++ sync primitive
   WarpVizTarget.dll+0x7e76c / +0x73bb5
   d3d11.dll+0x558101              <- Remix/DXVK
   WarpVizTarget.dll+0x5ed41
   d3d11.dll+0x559371 / +0x9315d3
   nvoglv64.dll+0xdda84b           <- NVIDIA driver
   WarpVizTarget.dll+0x37d1c
   nvoglv64.dll+0xf40eb9 / +0xf3acbc
```

A DXVK worker enters GPU Trace's instrumentation, blocks on a lock inside it, and
never returns. The game's main thread (`materialsystem_dx11 <- tier0 <- engine <-
launcher`, tid 18920) then blocks behind it, while ngfx sits forever at
`Waiting for target ready...`. Mutual wait, inside NVIDIA's code.

Caveat on the stack: it is produced by scanning the thread's stack for values
landing in a loaded module, so it is a SUPERSET of the real chain and the ordering
is not a true call sequence. The identification of the components is solid; the
exact frame order is not.

**Only remaining route to source-level shader cost:** GPU Trace a replay of a
saved `.ngfx-capture`. Opening a saved file performs no injection, so the
deadlock cannot occur. `Documents\NVIDIA Nsight Graphics\Titanfall2_2026_04_27_
16_12_26.8133756.ngfx-capture` (1.24 GB, April) already exists. Stale build, but
register allocation and occupancy are properties of the compiled shader.

---

## 5. The one real fix found, and it is worth keeping

`GPUTRACE_SASS_PATCHING_ON_RT_SHADERS_ONLY=1` (undocumented; pulled from strings
in `WarpVizTarget.dll`). Restricts SASS instrumentation to ray-tracing shaders,
which is all this project ever needed - `gb_primaryRays` is an RT shader and the
game's ~9.4k D3D11 raster shaders are noise.

Measured, same run otherwise identical:

| t | without | with |
|---|---|---|
| 19 s | **19.9 cores** | 4.3 |
| 21 s | 19.1 | 1.4 |
| 25 s | 18.2 | 0.2 |
| 27 s+ | 16.6, decaying over 126 s | **0.00** |

That is a ~21-core, two-minute driver shader-recompile storm removed outright, and
it moved the shader-profiler failure point from 50 KB to 140 KB of log. It is a
real fix for a real bug - it is just not the deadlock. Inert while
`-ShaderProfiler` is off; kept for whenever a newer Nsight is retried.

---

## 6. Nsight Graphics 2026.3.0 - cannot attach, uninstalled

Do not reinstall it expecting captures. Three measured outcomes, no working one:

| config | result |
|---|---|
| GPU Trace, stock | game null-derefs its swap chain, `materialsystem_dx11.dll+0x15e2f` |
| Graphics Capture, stock | crashes in `dxgi.dll+0x26b66` via `ngfx-capture-interception.dll` |
| either + DXVK `dxgi.dll` beside the exe | **ngfx never attaches at all** - looks fine, captures nothing |

Root cause: both interception DLLs **statically import `dxgi.dll`** and hook
`IDXGIFactory::CreateSwapChain`. `dxgi.dll` is not a KnownDLL, so the app
directory decides which copy they bind - and it was empty, so they bound
System32's Microsoft dxgi, which cannot make a swap chain for a DXVK device.
Remix's `DxgiFactory::CreateSwapChain` is never entered and the game ignores the
failed HRESULT. Proven by byte-searching for `IID_ID3D12Device`: **0 occurrences
in DXVK's dxgi.dll, 1 in System32's** - the probe frames cannot be DXVK's.

Nsight names it itself: *"An unknown object (IUnknown) was detected being passed
to an API call"*. That object is Remix's device.

**The app-directory redirect is a TRAP.** Putting DXVK's `dxgi.dll` next to the
exe does correct the binding, but then ngfx's interception fails to initialise and
never prints `Attachable process detected` (counted: 1 without the copy, 0 with).
The game runs clean because nothing is instrumenting it. A quiet run is not a
working run.

Also corrected: `WarpVizTarget.dll` is NOT new in 2026.3 - it ships in 2025.4.0
too. The "new component" framing was inferred from release notes, not evidence.

---

## 7. Code changed this session

| file | change |
|---|---|
| `dxvk_options.cpp` / `.h` | `enableAftermath` was hardcoded `true` "for diagnosis". Now `dxvk.enableAftermath` / `DXVK_ENABLE_AFTERMATH`, **default true** so normal play is unchanged. Same for `enableAftermathResourceTracking` |
| `dxvk_options.cpp` / `.h` | new `dxvk.enablePresentMetering` / `DXVK_ENABLE_PRESENT_METERING`, default true |
| `dxvk_adapter.cpp` | DLFG extension list `std::array` -> `std::vector<DxvkExt*>`; `nvPresentMetering` pushed only when enabled. Fixes ngfx 2025.4 halting the game with a modal `Unsupported extension: VK_NV_present_metering` that no unattended run can dismiss. Safe: `DxvkDLFG::supportsPresentMetering()` is a plain read of the extension bit, so consumers already handle absence |
| `d3d11_device.cpp` | new `[DXGI.QIProbe]` in `D3D11DXGIDevice::QueryInterface`: on an unknown riid, capture 8 frames and resolve each **by address** (`GetModuleHandleEx(FROM_ADDRESS)`), logging **full paths** - base names cannot distinguish DXVK's `dxgi.dll` from System32's, which was the entire question. Deduped on (guid, caller), capped at 64 lines |

`rtx.conf` untouched.

---

## 8. Tooling

`Capture TF2 (Nsight Graphics auto).ps1` - defaults are sec 2; takes no arguments.
* Activity names are **version-mapped**: 2025.4 calls it `Frame Debugger`,
  2026.3 calls it `Graphics Capture`. Passing the wrong one is a hard error
  (`Unknown activity name` + `Invalid general options`, ngfx exits, game never
  launches - reads exactly like a capture failure).
* Refuses `GPU Trace Profiler` while a `dxgi.dll` sits beside the exe (sec 6
  trap). `-IKnowGpuTraceCrashes` overrides.
* `Send-FrameDebuggerHotkey` delivers CTRL+Z then SPACE for the legacy
  `--wait-hotkey` activities; `Graphics Capture` and GPU Trace both use F11.

Freeze probe: `scratchpad\Catch3.ps1` (self-contained) launches the capture,
detects the stall by the remix log going quiet, and dumps per-thread stacks by
reading RSP from `GetThreadContext` and resolving stack slots to modules.
`Catch4.ps1` prints a 2-second CPU/log timeline instead - that is what separated
"burning 21 cores" from "blocked at 0 CPU".

---

## 9. Open

1. **Event buffer overflow on the good capture.** The 22:49 run warned twice:
   `Event buffer memory overflow. Trace may be missing some data. Event buffer
   memory that was needed (kB): 3564` (then 3706). Also
   `Some periodic sampling data was dropped ... Re-Tracing with double the sample
   duration: 6986ns` and `samples merged due to too high sample rate`. So the
   111 MB trace is INCOMPLETE. The script exposes `-EventBufferKb` (0 = use
   ngfx's default of 10000); raise it and re-capture before trusting fine detail.
2. **Read the 111 MB trace.** Multi-pass is on this time, so Trace Analysis should
   have real expert advice - the 104 MB predecessor carried the banner
   "works best with the multi-pass metrics enabled".
3. **`gb_primaryRays` is not throughput-bound.** From the 104 MB trace, whole-trace:
   SM 20.1 %, L2 19.4 %, VRAM 17.9 %, `Unallocated Warps in Active SMs` 61.0 %,
   CS occupancy 23.7 %. Nothing saturated, most warp slots empty - consistent with
   the 255-register / 144 B-spill occupancy ceiling, not a memory or ALU limit.
   Confirm per-pass by selecting the primary-rays range rather than the whole trace.
4. **File the deadlock with NVIDIA.** Sec 4 stack + sec 3 elimination table is an
   unusually complete report. The 2026.3 `CreateSwapChain` bug (sec 6) is a
   separate, arguably better report: it faults in their code with a two-line repro
   and does not require them to care about Remix.
5. The GPU-idle work (25-34 ms, `HANDOFF_GPU_IDLE_AND_NGFX.md` item 1) is
   untouched by all of this. **nsys is unaffected** - different layer, no
   WarpVizTarget - so `-CpuCtxSw process-tree` is still the next instrument there.

---

## 10. Traps

**Sample the freeze more than once.** The same stall reads as "blocked, 0 CPU" or
"burning 21 cores" depending on WHEN you look. Catching it at 16 s showed
9.59 s CPU/3 s; catching it at 40 s showed 0.03. A single sample produced two
opposite and equally confident conclusions in this session.

**Check the log's mtime before believing it.** `remix-dxvk.log` is overwritten per
run, so a grep during startup happily matches the PREVIOUS run's crash and reports
it as current. This produced a false "RESULT: CRASHED" that was pure stale data.

**`PARSE OK` is not a syntax check.** PowerShell parses a stray `elseif` as a
command name, so a broken if/else chain parses clean and fails at runtime. Walk
the AST for `CommandAst` named `else`/`elseif` instead.

**`Start-Process -ArgumentList` quotes nothing** in PS 5.1. The capture script's
path has spaces AND parentheses, so an unquoted path arrives as several arguments
and the game silently never launches. The script documents the same trap for
ngfx `--activity`.

**Bitdefender locks scratchpad scripts**, giving `The process cannot access the
file` on the .ps1 and `Add-Type : Access to the path '...dll' is denied` when
compiling P/Invoke. Two runs were lost to this before it was identified.

**A quiet run is not a working run.** See the sec 6 redirect trap: no crash meant
no instrumentation, not compatibility.
