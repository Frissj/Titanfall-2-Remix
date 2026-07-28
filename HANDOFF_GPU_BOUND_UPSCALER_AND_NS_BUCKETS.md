# Handoff: 19 → 58 fps. The upscaler was off, and the CPU work that didn't matter until it wasn't.

Written 2026-07-28 ~04:00. Everything below is measured unless it says INFERENCE.
Numbers come from `[Perf.*]` in `remix-dxvk.log`, gated on `rtx.logSubmitStall`
(currently **True** — set False for normal play).

Read sec 2 first. It is the single fact that moved the frame, and it was
invisible for the entire previous session because it lives in a file the perf
work never looked at.

---

## 1. Result

| | fps | fenceWaitMs | gpuIdleMs | perDraw_onDraw_us |
|---|---|---|---|---|
| session start | **19–20** | 50–58 | 5.7 | 28–31 |
| after DLSS on | **48.8** | 14.6 | 5.7 | 16.3 |
| after CamCatalog fix | **58** | ~14 | **~0.5** | 13.78 |

Two changes did essentially all of it. Everything else in this session was real
CPU reduction into a GPU-bound frame, i.e. worth zero frames at the time.

---

## 2. THE UPSCALER WAS OFF. Check this before any perf work.

`user.conf` overrides `rtx.conf`. It contained:

    rtx.dlssPreset  = 2      <- Custom
    rtx.qualityDLSS = 0
    rtx.graphicsPreset = 1

`DlssPreset::Custom` is **an empty case**:

```cpp
switch (dlssPreset()) {
  case DlssPreset::Off:    upscalerType.setImmediately(UpscalerType::None); ... break;
  case DlssPreset::On:     upscalerType.setImmediately(UpscalerType::DLSS); ... break;
  case DlssPreset::Custom: break;                       // <-- does NOT touch upscalerType
}
```

Custom means "stop managing upscalerType", not "turn DLSS on". Neither conf set
`rtx.upscalerType`, so it stayed `None`. **Fix was one line in `user.conf`:**

    rtx.upscalerType = 1

Result: `[Perf.GbDispatch]` went `downscaled=1920x1080 target=1920x1080 upscaler=0`
→ `downscaled=640x360 target=1920x1080 upscaler=1`. 2.07 → 0.23 Mpix, **19-20 → 48.8 fps**,
`fenceWaitMs` 50-58 → 14.6.

**The authoritative check is `[Perf.GbDispatch]`, not the option echo.** If
`downscaled == target`, you are at native res no matter what the presets say.
It prints ONCE per session at gbuffer setup — do not assume a later line exists.

Second-order effect: with no upscaler, `rtx_context.cpp:2673`'s `else` branch runs a
full-res `copyImage(finalOutput <- compositeOutput)` every frame. See sec 4 for why
"that copy cost 13.4 ms" was NOT established.

---

## 3. What changed in code

| # | change | file | measured |
|---|---|---|---|
| 1 | **CamCatalog session-cap hoisted above the work** | `d3d11_rtx.cpp` ~18542 | `pff_camCat` **2.467 → 0.049 us/draw**. Its 16-tuple cap is a SESSION cap that saturates in the first frames, but was tested AFTER `Desc()` + `GetMappedSlice()` + 3 scalar reads off write-combined cb2. Exact fix — same 16 lines still log. |
| 2 | **markStg / markXt / markFm → nanoseconds** | `d3d11_rtx.cpp` 9093/15097/18081 | Was `duration_cast<microseconds>`, flooring every sub-us stage to 0. At ~13 us/draw over ~30 stages most stages ARE sub-us. Lines now tagged `units=ns`. |
| 3 | **Filtered-tag sweep** (9 sites) | `d3d11_rtx.cpp` | `[T31Stale]` (FNV-hashed every instance matrix, ~1.06M/window), `[FloorTrace]` (24 AABB decodes/frame), `[o2w.fanout]`, `[MainCamPose]`, `[PhantomProbe]`, `[VMPass]`, `[WidowDiag]`, `[ShipSrcVB]`, `[ShipBone]`. DrawIdxInst 4.90-5.05 → 3.87-4.06 ms/frame; `residPerInst_us` 0.34-0.37 → 0.157-0.257. |
| 4 | **maxIdx cache: reverse scan + ring buffer** | `d3d11_buffer.h` 210 | Lookup walked the vector FORWARD while inserts appended to the BACK, so the newest entry was found last (~128×3 compares). `erase(begin())` on overflow was an O(n) memmove. |
| 5 | **`needsMeshBoundingBox()` skip for the index scan** | `d3d11_rtx.cpp` ~21845 | Implements the previous handoff's own release condition. **Inert while NEE cache is on** (`aabbSkip=0`). Proven to work when tested: `aabbSkip=22013/22110`, `wcScan=0`, bt_cullVtx → 0.01 us/draw. |
| 6 | `markSub` + `RTX_D3D11_SUBMARK` | `d3d11_rtx.cpp` | 8 bisection sub-markers, gated off. Returns before the clock read AND before `tStg` advances, so parents report whole spans. **Sub-buckets reading 0 means "not measured", not "free".** |

New instrumentation: **`[Perf.CullVtx]`** (cache hit rates, scan sources, `reduce_ms`
vs `acquire_ms`, MB/s) and `units=ns` on `[Perf.SubmitDraw.acc]`/`.max`.

---

## 4. REFUTED — I was wrong about these, do not re-derive them

* **"The mapping call is the cost, not the reduction"** (bt_cullVtx). WRONG.
  Split timer: `reduce_ms=232.2` of `wc_ms=235.4` — **98.7% is `maxIndexFromWC`**,
  `GetMappedSlice`/`mapPtr` is 1.3%.
* **"`pc_null ≈ 0` proves the 13.4 ms billed to `upscaler` is real work in that span."**
  INVALID. `rtx_options.h:1519` documents this exact trap: GPU timestamps are written
  at BOTTOM_OF_PIPE and do not wait for dispatches to drain, so cost migrates to
  whichever later mark the GPU reaches late — "pc_null and gpuDrain are empty and read
  ~0, while other empty intervals carry the frame." **No `[Perf.GpuPass]` per-pass number
  is trustworthy without `rtx.perfGpuStageSerialize` (env `DXVK_PERF_GPU_STAGE_SERIALIZE`).**
  Use it to find the owner, then turn it off to measure.
* **"`preFilters` dropped 66x (6.37 → 0.096 us/draw)."** Artifact of microsecond
  truncation. A stage going 1.2 → 0.9 us reads as 1 → 0. Fixed by change #2.
* **Widening `maxIndexFromWC` to 128 B / 2 accumulators.** Correct but pointless —
  it targets bandwidth, and bandwidth was never the limit. 12.7 us to reduce 702 bytes
  = **52 MB/s**, ~1.15 us per cache line, far past DRAM latency. That is the CPU
  crossing the bus to host-visible device memory. No SIMD change fixes it.
* **meshoptimizer.** Wrong tool for this frame: `instances=198, blas=74`, and the
  primary-ray pass is ~84% linear in RAY COUNT (resolution), not triangles.
* **`pt_visSurfReadback` being a real readback.** `kEnableRtxDebugProbes = false`
  (`rtx_debug_probes.h:28`) compiles `recordVisibleSurfacesReadback` out entirely.
  Its 3.2-5.3 ms was migrated cost per the BOTTOM_OF_PIPE trap above.
* **`MaybeEarlyInjectForUITexture()`** as the `preFilters` culprit — 0.076 us/draw.

---

## 5. Where the CPU frame goes now (`units=ns`, 165k draws, us/draw)

    bt_cullVtx   2.25   <- WC index scan; see sec 6
    bt_extractXf 1.83   <- ~33 mapped-CB reads, sub-buckets each <=0.3, death by a thousand cuts
    bonePalette  1.41   <- palette read already staged (memcpyFromWC); not the WC defect
    tail_emit    0.95
    pfs_guard    0.78   <- PROFILER, not renderer: QueryThreadCycleTime x2/draw (~0.44us each)
    tail_capture 0.53
    wallUs/draw 11.03

**`gpuIdleMs` is ~0.5 ms. You are GPU-bound again — further CPU work buys ~nothing.**
Before chasing any of the above, confirm `gpuIdleMs > 0` in the scene you care about.

---

## 6. Still on the table

* **GPU side.** `pt_integrate` ~3.6 ms of a ~14.6 ms frame was the largest pass at
  640x360 — but see sec 4, re-measure with `perfGpuStageSerialize` before acting.
  *(GPU-side detail to be attached separately.)*
* **`bt_cullVtx` 2.25 us/draw.** The max-index scan over write-combined memory. Staging
  cannot help — the other seven WC sites read a few fields (one streaming copy fixes them),
  this one must touch every index. **The only fix is not needing it:** GPU-side AABB, or
  anything that makes `needsMeshBoundingBox()` false. Change #5 already implements the
  skip and it is proven — it is just inert while `rtx.neeCache.enable` is true.
  NEE cache costs only ~1.1 ms GPU, so turning it off to win this is a bad trade.
* **`rtx.logSubmitStall = False`** for normal play — drops `pfs_guard` (~0.6 ms/frame).
  Worth checking whether it actually gates `SubmitCpuGuard`; if not, it should.
* `rtx.conf` still has `dlssPreset = 0 / upscalerType = 0`; `user.conf` overrides. Backups:
  `rtx.conf.bak-dlss`, `rtx.conf.bak-meshbbox`, `user.conf.bak`.

---

## 7. Methodology — traps that bit this session

**Normalise per DRAW, never per frame or per window.** Frame counts and window
lengths vary; `bt_cullVtx` "6.0 → 1.8 ms/frame" and "10.37 → 3.13 us/draw" are the
same fact, but only the second survives a frame-rate change. One run this session
had an unexplained 2x GPU slowdown that made every per-frame comparison worthless.

**Nested markers double-count.** `xt_*`/`w2v_*`/`pv_*`/`sf_*` are subdivisions INSIDE
`bt_extractXf`; `fm_*`/`fmm_*` inside `cvr_fillMat`. Summing everything gave 16.03
us/draw against a `wallUs` of 13.46. Sum top-level stages only.

**`markStg` advances the marker** — a parent reports the REMAINDER after its
subdivisions, not the total. Parent + children = the original bucket.

**Bisect; do not name a culprit inside a large bucket.** I did it twice and was
refuted both times (sec 4). The third time I subdivided instead, and `pff_camCat`
fell out immediately at 59% of its span. Sub-markers cost ~41 ns each — gate them
off (`markSub`) once they have served.

**Do not bulk-rewrite source through PowerShell.** `Get-Content -Raw` reads UTF-8 as
ANSI in PS 5.1 and `Set-Content -Encoding utf8` writes it back double-encoded: 1445
mangled em-dashes, a stray BOM, and a 2088-line diff. Repaired by decoding UTF-8 and
re-encoding CP1252. **Use the editor tooling for source edits.**

**A count of zero log lines does not mean a site is free** (still true from the last
handoff — this session found 9 more). And a *saturated* throttle does not mean a probe
is free either: CamCatalog could not log for hours and still cost 16% of every draw.
