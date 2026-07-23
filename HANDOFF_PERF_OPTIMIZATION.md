# HANDOFF: TF2 dxvk-remix CPU performance optimization

Session date: 2026-07-23. Frame time went **~1,300 ms → ~420–570 ms** (scene-dependent, ~3×).
Everything below is grounded in `remix-dxvk.log` probe output — every claim has a log tag you can re-verify.

---

## 1. Ground truth about this machine (do not re-litigate)

- **CPU sustains ~2.4 GHz** (`[Perf.CpuCalib] cyclesPerUs=2496`, measured by busy-spin max over 8×1 ms).
  All µs numbers in the log are real wall time on this slow clock.
  **Open system-level question: why 2.4 GHz?** If this chip nominally boosts higher, a Windows power
  plan / thermal / battery cap is throttling it — fixing that is worth more than any single code change.
- **SubmitDraw does NOT stall.** `[Perf.SdThreads] stallUs` is now computed with the calibrated rate and
  reads 3–4%. `[Perf.SdStall]` (8-checkpoint sampled probe, wallMs/cpuMs per segment) shows wall ≈ cpu
  in every segment. It is ~pure CPU. All "stall/lock/logger contention" theories are DEAD — the old
  huge stallUs was a hardcoded-5 GHz conversion artifact.
- The GPU is mostly idle (`GPUIDLEms` 230–750 ms/frame depending on scene). The path tracer itself is
  2–10 ms. **This whole effort is CPU-side.**

## 2. The one recurring root cause: write-combined (WC) memory reads

D3D11 dynamic buffers are mapped HOST_VISIBLE|HOST_COHERENT **without HOST_CACHED** = write-combined.
CPU **reads** from WC are uncached: every scalar load is a full memory transaction (~300 MB/s effective).
This single mechanism was behind the four biggest wins:

| Fix | Bucket | Before → After |
|---|---|---|
| Coverage dump → GPU compaction | `finalBlit` | ~1,100 ms → 30–80 ms/frame |
| IB max-index scan reads snapshot/staged, not WC | `bt_cullVtx` | ~240 → ~1–2 ms/frame |
| `memcpyFromWC` (movntdqa streaming loads) for IB snapshot | `indexSnap` | ÷4 per draw |
| Staged cbuffer copies (`stagedCbBytes`) | `w2vw_cb3`/`w2vs_cached`/`se2cb2` ÷6, `bt_extractXf` −30% | |

**Reusable tools now in the codebase (`src/d3d11/d3d11_rtx.cpp`, top of file):**
- `memcpyFromWC(dst, src, len)` — streaming-load copy out of WC memory. Use for ANY bulk read of
  mapped slices. Handles the 16-byte source-alignment requirement internally.
- `stagedCbBytes(D3D11Buffer*, size_t& lenOut)` — returns a cached-heap staged copy of a bound
  **constant buffer**, keyed `(buffer ptr, GetContentGeneration())`. Safe ONLY for cbuffers
  (D3D11 11.0 cbuffers are DISCARD-map-only, and DiscardSlice bumps the generation).
  **NEVER use it for vertex/index/SRV buffers** — those allow NO_OVERWRITE writes that don't bump
  the generation. Every call site keeps a raw-mapping fallback when it returns null.
- `memoCamOriginLoc / memoModelInstSlot / memoBoneMatrixSlot` — per-shader-pointer single-entry memos
  for the string-keyed RDEF lookups (`FindCBField`/`FindResourceSlot` heap-allocate temps per call;
  C++17 = no heterogeneous map lookup, see `d3d11_shader.h:75`). Draws batch by VS → ~free.
- `getVsHashShort()` returns **const std::string&** now (was by-value = heap alloc per call, ~10-20/draw).

**Known trap (do NOT do):** dedup/caching keyed on `mapPtr` identity. DXVK's slice pool recycles
mapped addresses within a frame → stale-content aliasing → the exact BLAS-mangle bug class that was
fixed before. Only `GetContentGeneration()` is a safe content key, and only DISCARD bumps it.

## 3. All changes made this session (file → what/why)

- `src/dxvk/shaders/rtx/pass/coverage/coverage_compact.{h,comp.slang}` — NEW compute pass: folds the
  74 MB coverage buffer to nonzero (index,value) pairs on GPU. Auto-discovered by the shader build.
- `src/dxvk/rtx_render/rtx_resources.{h,cpp}` — NEW `m_surfaceCoverageCompactBuffer`
  (8 MB, HOST_CACHED — the point); coverage buffer got STORAGE usage + SHADER_READ access.
- `src/dxvk/rtx_render/rtx_context.cpp` — `recordCoverageCompactDispatch` lambda in
  `dispatchDebugView`, called at BOTH exits, **after** `debugView.dispatch()` (postprocess writes the
  PickRegion slots — recording compact earlier breaks `[PickHash]`/center-VS feed for `[SkinAABB]`).
  CPU dump reads a sparse shadow rebuilt from compact pairs; the end-of-dump memset now explicitly
  clears the REAL GPU buffer (WC writes are fast; the clear must stay or counts accumulate forever).
- `src/d3d11/d3d11_rtx.cpp` — the bulk:
  - `memcpyFromWC` + applied at IB snapshot, cold-scan scratch, bone-palette staging.
  - Max-index cold scan reads `geo.indexDataSnapshot` first (cached heap), never scalar-reads WC.
  - `stagedCbBytes` + ~20 call sites across ExtractTransforms + `SetSkyCategoryFromCb2`.
  - `CaptureEngineSunFromCb`: once-per-frame success latch (was ~16 RDEF lookups + 12 WC reads
    × EVERY draw; publishes a last-write-wins global — once/frame is equivalent).
  - Bone-palette capture stages 12 KB via `memcpyFromWC` before the float3x4 convert.
  - RDEF memos + `getVsHashShort` const-ref (~22 call sites converted to `const std::string&`).
  - Probes: `[Perf.SdStall]` (8 segments, 1/64 draws sampled), `[Perf.CpuCalib]`,
    `sf_cam`/`sf_o2w` sub-buckets inside `xt_srcFb`.
- `src/util/log/log.{h,cpp}` — batched logging: lines buffer in memory; flush on Warn/Error
  (immediately), 64 KB, or 100 ms. Crash path covered: the `UnhandledExceptionFilter` in
  `d3d11_main.cpp` calls `Logger::flush()` which drains the buffer; `~Logger` drains on exit.
  `RTX_LOG_UNBUFFERED=1` restores flush-per-line. (Measured: this was NOT the bottleneck, but it
  removes ~700 OS flushes/s of overhead.)
- `src/dxvk/rtx_render/rtx_types.h` + `rtx_accel_manager.cpp` — **BlasSizeCache**: per-`BlasEntry`
  memo of `vkGetAccelerationStructureBuildSizesKHR` (was 137 driver calls/frame; ~115 entries are
  bReuse). Key = FNV of primCount, vertex format/stride/maxVertex, indexType, transformData
  NULL-ness (spec examines exactly that), geometry+build flags, bound OMM hash.

## 4. Current state & ranked remaining targets (with evidence)

Latest heavy-scene numbers (~570 ms/frame, ~450 committed instances, `[Perf.SdStall]` per ~9-frame window):

1. **`tail_emit`** — 40 µs/draw normal scene, **133 µs/draw heavy** (~70–80 ms/frame). The `emit`
   segment = `commitGeometryToRT` deep-copying DrawCallState into the CS chunk: vectors
   (`skinningData.pBoneMatrices` = up to 16 KB/draw!, futures, Rc refcount traffic — `bt_geoCopy`
   is separate and small, this is the dcs copy). Ideas: move instead of copy where the source dies,
   `shared_ptr` the bone palette like `indexDataSnapshot` already is, pool the chunk allocations.
   START HERE.
2. **`cvr_fillMat`** — 30→101 µs/draw heavy (~58 ms/frame). FillMaterialData scoring. `fm_*`
   sub-buckets exist; `[Perf.FillMatCache]`/`[Perf.SrvCache]` lines report cache hit rates.
   Memory note: per-draw timers can't resolve sub-µs buckets — confirm wins by mechanism.
3. **prepScene `merge` residual** (`[Perf.Merge]`, inject side): `dynBlas` now 18–29 ms (was 38).
   Remaining cost is NOT the size query: it's the per-entry per-instance
   `OpacityMicromapManager::getOpacityMicromapHash` loop (runs every frame, ~137 entries,
   `rtx_accel_manager.cpp` ~1298), `tryBindOpacityMicromap`, and the ~22 real update-build
   recordings. `buildBlases` 17–27 ms: bucket path (only 4 size queries — fine) but check the
   `m_blasPool` linear scan and scratch allocation. `cullTlas` swings 5–46 ms — investigate variance.
4. **`sf_o2w`** — 29 µs/draw: the t31/t30 objectToWorld block. Per draw: COM `GetResource()`
   (AddRef/Release), 5× `GetRtxSemantics()` loops (could be one pass), 12 scalar WC float reads of
   the instance matrix (small `memcpyFromWC` candidate — it's a VB so NO caching, copy per draw only).
5. **`head` segment** — 55 µs/draw heavy (pf_setup + preFilters): entry-cost audit not yet done.
6. **`xt_cls`** — 12–23 µs/draw: V2 classifier override; classify() is cached per shader, the
   rewriters are not.

## 5. How to read the probes (all in `[Perf.*]` log lines, 5 s windows)

- `[Perf.SdStall] ... head=W/C vsIdx=W/C ...` — ×64-extrapolated wallMs/cpuMs per window per segment.
  W≈C = compute-bound (optimize by doing less); W≫C = blocking (find the lock). Currently all W≈C.
- `[Perf.SubmitDraw.acc]` — per-window µs accumulators; **divide by `[Perf.SdThreads]` draws for
  per-draw cost before comparing across scenes** (scenes vary 3k–7.6k draws/window).
- `[Perf.Merge]` `bBuild/bUpdate/bReuse` — dynamic-BLAS mechanism counters; `dynBlas`/`buildBlases`
  µs. `[Perf.PrepScene]` wraps it plus gc/cullTlas/accelLight.
- `sf_cam`/`sf_o2w` — xt_srcFb sub-split (camera reconstruction vs objectToWorld extraction).
- Env knobs: `RTX_LOG_UNBUFFERED=1` (per-line log flush), `RTX_D3D11_DIAG=1` (unfilter diag tags).

## 6. Invariants — do not break

- Coverage compact dispatch must record AFTER `debugView.dispatch()` (PickRegion writes).
- The CPU memset of the real coverage buffer at end-of-dump must stay (GPU atomics accumulate).
- `stagedCbBytes` is cbuffer-only (DISCARD/generation contract). VB/IB reads: `memcpyFromWC` per
  draw, never cached.
- Logger: Warn/Error must keep flushing immediately (crash forensics workflow depends on it).
- BLAS size cache key must include the OMM hash and transformData null-ness.
- The dump/readback "loose sync" semantics (reads N-2 frame data, torn reads tolerated) are
  intentional everywhere in the coverage system — keep entry validation (index < total) on any
  GPU-written data before CPU indexing.
