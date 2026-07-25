# HANDOFF — TF2: gb_primaryRays is ~118 ms and we now know exactly where it goes

Tree: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
Log:  `...\Titanfall-2-...-AnkerGames\Titanfall2\rtx-remix\logs\remix-dxvk.log`
Conf: `...\Titanfall2\rtx.conf`
Date: 2026-07-25. **Supersedes `HANDOFF_TF2_PRIMARY_RAY_COST.md`.**

HW: RTX 4080 Laptop, driver 610.62.0. Debug build (`_Comp64Debug`, `/Od`) — CPU only,
shaders compiled separately.

**Status: cost fully LOCALISED, mechanism NOT proven.** The pass is now split to the
stage level and every structural theory from V1 is dead. What remains is a single
open question (§7) with a cheap next step (§8).

---

## 1. The one-line problem

`gb_primaryRays` costs **~118 ms for 2.0736 Mpix**. 98% of it is one function —
`resolveVertex` (surface + material evaluation) — run 2.21× per pixel. There is no
hotspot inside it.

---

## 2. Established by measurement — trust these

Clean-config baseline (no diagnostics enabled, DLSS off, native 1920x1080):

```
totalMs = 167–184   outsideRtMs = 101–176   → real frame ≈ 270–325 ms (~3.3 fps)

gb_primaryRays  112–128   ← 40% of frame, largest single stage
pt_integrate     16–23
prepScene       6.5–11
pt_rtxdi        8.7–10.6
denoise         4.7–8.2
finalBlit       0.7–1.9
```

### The ablation ladder (§6 `rtx.perfGbStopAfter`)

| rung | `gb_primaryRays` | stage cost | share |
|---|---|---|---|
| 1 — raygen + all 63 G-buffer stores | 0.38 ms | 0.38 | 0.3% |
| 2 — + primary TLAS traversal | 1.9 ms | **traversal 1.5** | 1.3% |
| 3 — + unordered resolve | 2.2 ms | **unordered 0.3** | 0.3% |
| 4 — + `resolveVertex` | 49.5 ms | **material eval 47.3** | 42% |
| 0 — full | 107–118 ms | tail + loop | 54% |

Rungs 2/3/4 each force `continueResolving = false`, so every rung is exactly ONE
iteration. That is what makes the following multiplication legitimate:

```
0.38 + 2.21 × (1.5 + 0.3 + 47.3) = 108.9 ms      measured full: 107–118 ms
```

The post-material tail (medium transmittance, PSR selection, NRC writes) falls out
at ≈ 0. **The remaining 54% is the resolve loop re-running the material evaluation.**

Note: divergence means 2.21 mean iterations costs ~2.21 *full* passes — an inactive
lane does not release the warp.

### Other hard numbers

| fact | evidence |
|---|---|
| mean resolve interactions/pixel = **2.21**, rock steady | `[Perf.Resolve]` view 66 |
| 37–38% of instances are resolve-loop drivers (unorderedBlend 90–101, alphaBlend 29–39, alphaTest ~5.5); opaque 211–225 instances hold 485–539k of the prims | `[Perf.NonOpaque]` |
| 3.07% of PRIMS are any-hit **and** 37% of INSTANCES are — both true, different denominators | `[Perf.TlasOverlap]` + `[Perf.NonOpaque]` on same frames |
| shader: **Register Count=255** (hw cap), **144 B/thread local memory**, 16896 B shared, ~800 KB binary | `[Perf.Shader]` |
| occupancy = 65536/255 = 257 threads = **8 warps of 48 (16.7%)**, register-bound | arithmetic |
| device: `maxComputeSharedMemorySize=49152`, subgroup 32 | `[Perf.Limits]` |
| billboard generation FAILS in TF2 (`unsupported quad index layout`) so unordered instances stay triangles | log |

---

## 3. REFUTED — do not re-test

Everything here died to a direct measurement, not an argument.

| hypothesis | how it died |
|---|---|
| **Triangle count / Mega Geometry / BVH quality** | Traversal is **1.5 ms** of 118. Scene is 667k prims, 185k cam-reachable. Nothing about geometry structure can reach 110 ms that isn't traversal. Also explains why `PREFER_FAST_TRACE` made things worse — it doubled build time to optimise 1.5 ms |
| **TLAS fan-out / map-spanning boxes** (V1 §8) | Real BLAS bounds are TIGHT (§5). The `containAtCamera=45 meanAlongRay=38.7` figures were computed from boxes up to **753×** oversized — mostly false containment |
| **Unordered TLAS / separate unordered approximations** | 0.3 ms. `enableSeparateUnorderedApproximations=False` changed `gb_primaryRays` by ~0 (112 vs 107–110). My original hypothesis was wrong by ~350× |
| **Resolve loop as a large multiplier** | It is 2.21×, not the 5–30× predicted |
| **Texture bandwidth / cache footprint** | Forced mip bias +6 (~4096× smaller texel footprint, verified applied via `lodBias=6` in `[D3D11Rtx.SamplerWrap]`) bought **10%** |
| **Anisotropic tap count** | Dead on inspection: `computeAnisotropicEllipseAxes` builds an **isotropic disc** footprint by design ("matches what raster hardware would compute with aniso=1"), so `maxAnisotropy=16` never engages. Do not spend a run on it |
| **Fork-local gradient setup** (4 clip-space projections + rank-1 detection per pixel) | `rtx.perfCheapTextureGradients=True` → **8.5%** |
| **G-buffer store bandwidth** (63 RW_TEXTURE2D) | Rung 1 = 0.38 ms for raygen + ALL stores |
| **`finalBlit` as the frame ceiling** | **MY OWN ARTIFACT.** See §4 |
| **WBOIT as the register lever** | `wboitEnabled=False` → Register Count **255 → 255**. Zero |
| **`SurfaceInteraction` debug fields as the register lever** | `REMIX_SI_DEBUG_FIELDS=0` → **255 → 255**. Zero (binary did shrink 804224→800512, so it took effect) |
| **Occupancy ballast experiment** | Impossible on this device. 255 regs is already the register-imposed occupancy FLOOR; shared memory is the only other lever, and 48 KB/block × 2 blocks = 96 KB still fits the ~100 KB SM budget. Would need >50 KB/block, above the device cap |

V1 also killed: secondary bounces, integrator swap, PSR, degenerate AABBs, engine
detours, submission fragmentation, OMM, AS build quality, shader `-O3` (hangs GPU),
debug-view switch, NRC/WBOIT permutations as a *timing* factor.

---

## 4. `finalBlit` was a measurement artifact — read this before trusting old logs

V1 and the mid-session notes claimed `finalBlit` ≈ 360 ms and was "the frame ceiling".
**It is 0.7–1.9 ms.**

Every run between 13:11 and 16:13 had `rtx.debugView.debugViewIdx = 66` and
`rtx.debugView.perfResolveStats = True` sitting in `rtx.conf`. The debug-view
statistics path does a host-mapped readback every frame. `finalBlit` "held steady at
~360 ms across five runs" because the *diagnostic* held steady.

The disproof was already in the 13:07 log, before the debug view went in:
`finalBlit = 2.74 / 0.87 / 3.51`.

**Consequence:** the primary-ray pass IS the dominant GPU cost (~40% of frame). Any
note saying otherwise is contaminated. The ablation ladder itself still stands —
`gb_primaryRays` is if anything slightly higher clean (118 vs 110), so the debug view
was not inflating it.

---

## 5. Real defects found on the way (independent of this perf hunt)

**1. `geometryData.boundingBox` is up to 753× oversized.**
Computed over the whole vertex RANGE (`for vi in 0..vertCount`), not the draw's
indexed subset. Measured by `[Perf.BlasBounds]` over 835 draws:

```
ratio = fullDiagWorld / idxDiagWorld
  = 1.0      562 draws        > 10×    80 draws        max 753×
  immutable median 1.00   |   mapPtr median 2.52
worst: VS_1953b6e9cc252e4e (69 draws, 628×)  VS_0f1ada8619685f41 (53, 628×)
       VS_7c6a14cb800791ae (18, 753×)        VS_e7abcf4ea24b0fa7 (25, 230×)
```

Mechanism: draws address a narrow window of a shared VB (`minIdx` large,
`maxIdx == vertCount-1`) and inherit the whole buffer's box. One example: a single
triangle (`indexCount=6, idxVerts=4`) with a real 530-unit box assigned a
**29,631-unit** box. `VS_1953`/`VS_e7ab` are the BSP worldspawn shaders living in a
45–55 MB immutable VB.

**Anti-Culling is the only consumer of this box**, so it is making visibility
decisions on boxes up to 753× too large. A too-large box tests as on-screen more
readily, so the failure direction is keeping instances alive that should be dropped.
The driver builds the TLAS from real BLAS contents, so traversal was never affected.

**2. Every Remix sampler control is dead in this fork.**
`patchSampler` is the sole consumer of `getTotalMipBias()` and
`useAnisotropicFiltering()`, and it is gated on `samplerOverride == nullptr`
(`rtx_scene_manager.cpp:3009`). `[D3D11Rtx.SamplerBranch] case=0 (ovNull=0)` is the
**only** case that ever fires — so `sampler = samplerOverride` stands and
`patchSampler` never runs.

Dead as a result: `rtx.nativeMipBias`, `rtx.upscalingMipBias`,
`rtx.useAnisotropicFiltering`, `rtx.maxAnisotropySamples`, **and the DLSS mip bias**
(`log2(upscaleRatio)`, which exists to sharpen textures when rendering below native).
That last one is a standing visual-quality bug with DLSS on.

**3. `[Perf.Shader]` misreported Local Memory Size.** The driver writes
`0x10_0000_0090`; the high dword `0x10` is constant across every shader and every
build. FIXED — now prints `(raw=0x... lo32=...)`. `lo32` is the real figure. gbuffer
= 144 B/thread, `integrate_direct_rayquery` = 496 B, most others 0.

**4. `rtx.conf` has duplicate keys** — `rtx.debugView.debugViewIdx` (lines 27 and 92)
and `rtx.lights.engineStaticLightIntensityScale`. Later wins silently.

---

## 6. Instrumentation added this session

All default-OFF unless noted. `[Perf.*]` is not dropped by the log.cpp prefix filter.

| tag / flag | file | what |
|---|---|---|
| `rtx.perfGbStopAfter` | `geometry_resolver.slangh` (4 cuts), `raytrace_args.h`, `rtx_context.cpp` | **The ladder.** 0=full, 1=raygen only, 2=+traversal, 3=+unordered, 4=+material. Rungs 2–4 force a single iteration. Garbage image by design |
| `[Perf.NonOpaque]` / `rtx.perfNonOpaqueCensus` (**default ON**) | `rtx_instance_manager.cpp:3311` | Per-branch census of the opaque/non-opaque classification, by INSTANCE (the correct denominator), + top vertex shaders by driver-instance count |
| `[Perf.Resolve]` / `rtx.debugView.perfResolveStats` | `rtx_debug_view.cpp` | Mean resolve interactions per pixel. Needs `rtx.debugView.debugViewIdx = 60` (primary) or `66` (+unordered). **Leaving this on corrupts `finalBlit` — see §4** |
| `[Perf.BlasBounds]` / `rtx.perfBlasBoundsProbe` | `d3d11_rtx.cpp:27645` | Whole-range vs indexed-subset AABB, one-shot per (vs, vertCount, indexCount), capped 2048. Placed AFTER the objAabb timing block so it cannot contaminate it |
| `rtx.perfForceSamplerMipBias` / `rtx.perfForceSamplerAniso` | `rtx_scene_manager.cpp:3028` | Forces mip bias / aniso onto the FINAL material sampler, after the branch, because the normal options never reach it (§5.2). Memoized — `DxvkDevice::createSampler` does not deduplicate |
| `rtx.perfCheapTextureGradients` | `surface_interaction.slangh:1299` | Substitutes the cheap texcoord-diff footprint for `computeAnisotropicEllipseAxes` |
| `REMIX_SI_DEBUG_FIELDS` (**compile-time, default 0**) | `surface.h:57` | Removes `debugPathCode` / `dbgMinClipW` / `dbgMaxClipW` / `dbgHitWorld` from `SurfaceInteraction`. Freed 0 registers but costs nothing; leave off |
| `[Perf.Limits]` | `dxvk_compute.cpp` | `maxComputeSharedMemorySize` etc. — needed to turn Register Count into occupancy |
| `[Perf.Shader]` lmem fix | `dxvk_compute.cpp` | Zeroes the stats array, prints raw+lo32 for 64-bit stats |
| `REMIX_OCCUPANCY_BALLAST_KB` | `gbuffer.slang:308` | **INERT — device cannot support it (§3). Delete or ignore** |
| `[UnhandledException]` | `d3d11_main.cpp` | Now logs faulting module+offset, all registers, and a module-resolved backtrace. Added after an unattributable AV at `0x7FFF4FBD7E17` (read of `0xffffffffffffffff`) in a non-Remix module |

`[Perf.GbDispatch]` is now change-keyed on `gbStopAfter` too, so a mistyped ladder
rung cannot silently report the previous run's timing.

---

## 7. The open question

**Why are those 47 ms slow?** There is no hotspot — texture bandwidth ≤10%, gradient
setup 8.5%, aniso 0%, and three structurally unrelated cuts each returned only their
proportional share. That pattern is the signature of uniformly throttled execution.

The candidate is **occupancy**: 255 registers (hw cap), 144 B/thread spill, 8 warps of
48. It fits everything. It is **not proven**:

- The only supporting data point — POM + thin film removal, 255→168 registers, +22% —
  is confounded, because it removed execution work as well as registers.
- Two attempts to test it by lowering registers (WBOIT, debug fields) both failed to
  move the count off 255.
- The reverse test (lower occupancy, see if time scales) is impossible on this device.

**Register pressure here behaves as a cliff, not a slope.** The compiler wants far
more than 255, gets clamped, and spills. Removing one feature's worth of live state
does not get under the cap, so it buys nothing. POM+thin film crossed it; nothing
else tried has.

---

## 8. NEXT STEP — Option A: find which stage owns the registers

Convert `rtx.perfGbStopAfter` from a runtime `cb` value into a **compile-time**
define. The compiler then dead-code-eliminates the skipped stages and `[Perf.Shader]`
reports a real register count per truncated shader.

1. In `geometry_resolver.slangh`, add near the top:
   ```
   #ifndef REMIX_GBSTOP
   #define REMIX_GBSTOP 0
   #endif
   ```
2. Change the four gates from `cb.perfGbStopAfter` to `REMIX_GBSTOP`:
   - `:2923` `if (cb.perfGbStopAfter != 1)` → `#if REMIX_GBSTOP != 1` … `#endif`
   - `:1448` / `:1533` / `:1576` `if (cb.perfGbStopAfter == N)` → `#if REMIX_GBSTOP == N`
3. Build three times with `REMIX_GBSTOP` = 2, 3, 4 (edit the default, or `-D`).
4. Read the **first** `[Perf.Shader] cs=gbuffer_rayquery_nrc_wboit` line each time.

**No timing runs needed** — the answer is a compile-time number on frame one.

| result | meaning |
|---|---|
| stop=3 well under 255, stop=4 near 255 | material evaluation owns the pressure → split the kernel there |
| stop=2 already near 255 | traversal + setup owns it → the problem is upstream of material eval |
| all near 255 | pressure is diffuse → only a full megakernel split helps |

Ignore stop=1's number; that shader is nearly empty and tells you nothing.

Then **Option B** (the fix, justified regardless of the mechanism): split the
megakernel so material evaluation is its own dispatch from the resolve loop. Each
half fits in registers, and the 2.21× loop re-runs a much smaller kernel — the two
effects compound. Removes no features.

Remaining untried register lever: the VGUI members of `SurfaceInteraction`
(`vguiSecondaryQuadPos`, `vguiGlyphDims`, `vguiPackedIndices`, and the two secondary
gradients — ~10 dwords, ~40 use sites). Deliberately NOT flagged: it is a live
feature and a flag that fails to compile when flipped is worse than none.

---

## 9. Process notes

**Confirm the knob reached the code, not just the config.** Three tests this session
silently did not run:
- `rtx.nativeMipBias` never reaches material samplers (§5.2). Caught only because
  `[D3D11Rtx.SamplerWrap]` prints `lodBias`, which still read 0.
- `rtx.integrateIndirectMode = 1` did not apply — `[Perf.GbDispatch]` reported `nrc=1`
  and the compiled permutation was still `_nrc`. Something outranks the conf layer.
- `debugViewIdx = 66` + `perfResolveStats` were left set for ~3 hours and produced the
  entirely fictional `finalBlit` result in §4.

`[Perf.GbDispatch]`, `[Perf.GbBranch]`, `[Perf.Shader]` and the `lodBias`/`aniso`
fields in `[D3D11Rtx.SamplerWrap]` all exist to make this impossible. Read them
before believing any A/B.

**Prefer compile-time numbers to timing runs where possible.** Register Count answers
a question on frame one with no scene-drift noise. Cross-run timing comparisons in
this session were repeatedly weakened by the camera being in a different place —
within-run stability is ±5%, between-run drift is much larger.

**The ladder technique worked; hypothesis-first did not.** Five mechanism guesses died
(unordered TLAS, fan-out, texture bandwidth, aniso taps, gradient setup). Every one of
them was killed by an ablation, and the ablations are what bounded the search space
down to a single function. Bisect, don't theorise.
