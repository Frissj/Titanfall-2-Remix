# HANDOFF — split the unordered resolve into its own dispatch

Tree: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
Log:  `C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx-remix\logs\remix-dxvk.log`
Conf: `...\Titanfall2\rtx.conf`
Date: 2026-07-25. HW: RTX 4080 Laptop, driver 610.62.0, `_Comp64Debug` (`/Od`, CPU only).
1920x1080 native, DLSS off, static camera.

**Read `HANDOFF_TF2_PRIMARY_RAY_COST_V3.md` first for how the pass got here.**
This document covers only the one remaining idea, and — importantly — the six
that are already dead. Do not re-run them.

---

## 1. The task

`gbuffer_rayquery_nrc_wboit` is **255 registers — the hardware cap — and 794 KB
of code.** 255 registers means 8 warps/SM, the lowest occupancy tier. The pass is
latency-bound with nothing to hide latency behind, which is why every localised
cut measures as noise (§3).

The one structural fix left: **move the unordered resolve out of the gbuffer
shader into its own dispatch**, so each gets its own register budget instead of
sharing one function with the ordered resolve and a 1600-line material evaluator
inlined twice.

This removes no features. That constraint is non-negotiable — a previous attempt
to propose cutting decal bins was rejected, correctly.

---

## 2. Current measured state

Stage split of `gb_primaryRays` (runtime ladder `rtx.perfGbStopAfter`, medians,
reproduced across two full sweeps; baseline drifts 23.8–27 ms with thermals):

```
launch + G-buffer store   0.4 ms    ( 2%)
traversal                 1.8-2.0   ( 8%)
UNORDERED RESOLVE        13.1-13.4  (~56%)   <- the target
material                  7.4-9.3   (~32%)
tail                      0.6-1.6   (under noise floor)
                          ~24 ms total
```

Inside the unordered stage (`rtx.perfUnorderedStopAfter`), consecutive rungs:

```
uno1 traversalOnly        9.99   (see CAVEAT below)
uno2 surfaceInteraction  20.33   +10.34 ms  <- the whole cost sits here
uno3 clipTest            20.13    -0.20     noise
uno4 materialInteraction 22.81    +2.68
uno5 / uno6                        noise
```

**CAVEAT on uno1:** 9.99 ms is *lower* than `gb3_unordered` (15.3) despite having
strictly more stages enabled. Cutting the candidate body changes `opacity`, which
drives `continueResolving`, which collapses the **ordered** loop's trip count too.
Read consecutive differences only; absolute uno values are contaminated.

Census (`rtx.perfUnorderedStepCensus`, needs `logSurfaceCoverage=True` — see §5):

```
stepsPerPixel = 1.20619   interactionsPerPixel = 1.20619   acceptRate = 1.0
```

**1.2 candidates per pixel against a cap of 32.** Not volume-bound, nothing
rejected. Lowering `rtx.primaryRayMaxInteractions` buys nothing.

Shader stats (`[Perf.Shader]`):

```
gbuffer_rayquery_nrc_wboit      255 regs  794112 B  144 B/thread local
gbuffer_psr_rayquery_nrc_wboit  168 regs  403200 B  240 B/thread local
```

Frame context — **read this before optimising anything here**:

```
GPU totalMs 71-75      gb_primaryRays 23.7      outsideRtMs 110
CPU [Perf.Entry] 59-62 ms, of which GetData 54-57 ms / ~400k calls/frame
[Perf.Query] 10.4M polls, 99.9984% notReady, doNotFlush=140
```

The unordered stage is **13 ms of a ~180 ms frame**. Even total success is ~7%.
The EVENT-query spin alone is 4x larger. This split is worth doing only if the
larger items are already handled or explicitly deprioritised.

---

## 3. DEAD ENDS — do not repeat

Every one of these was built, measured, and refuted this session. Each has a
`!! RESULT` comment at its declaration site.

| Attempt | Result | Where documented |
|---|---|---|
| `perfCheapTextureGradients` (bypass `computeAnisotropicEllipseAxes`) | −0.19 ms vs 1.50 ms floor. **Not ALU.** | `raytrace_args.h` |
| `perfCoherentUnorderedFetch=1` (coherent `Surface` load) | −0.216 median / **+0.565 min** → hitch, not cost | `raytrace_args.h` |
| `perfCoherentUnorderedFetch=2` (+ coherent vertex fetch) | +0.195 / +0.454 vs 0.622 floor. **Not locality.** | `raytrace_args.h` |
| Decal bins local-mem → registers | local 144→0 B, but `gbuffer_psr` **168→255 regs**, no timing gain. Compiler was right to spill. | `resolve.slangh` §DecalBins |
| `[noinline]` on `opaqueSurfaceMaterialInteractionCreate` | **Hint ignored.** 794112→789888 B, 255→255 regs. Slang/DXC does not honour it on this path. | `opaque_surface_material_interaction.slangh` |
| Slim material evaluator for the unordered path | Audit killed it pre-build: unused fields (`thinFilmThickness`, `subsurfaceMaterialInteraction`, `normalDetail`) are the cheap tail; `albedo`/`opacity`/`shadingNormal`/roughness need the texture reads + normal-map decode, i.e. the bulk. Also `inout SurfaceInteraction` — POM mutates it and `evaluateOpaqueApproximations` consumes the mutated value. | this doc |
| All 8 material ablations (`skipPom`, `skipTextures`, `skipThinFilm`, `skipAllThree`, `mat_stop1..4`) | every one under the noise floor; `skipPom` came back **slower** (+1.37) | `rtx.conf`, sweep summaries |

**The pattern:** every probe cuts *execution*. Register pressure is an
*allocation* property. That is why they all read as noise, and it is the whole
argument for the split.

Also refuted historically, do not resurrect: the 3.0 / 42.5 / 74.5 stage split
and the ~120 ms pass figure (both pre-coverage-atomic-fix, retracted in V3);
"the cost is material evaluation performed twice" (contradicted by the stage
ladder — material is 7.4–9.3 of 24 — and by the 8 flat material ablations).

The "occupancy is refuted" claim in older comments was measured while the
coverage atomics were burning 104 ms/frame and **does not hold**. Occupancy is
now the best-supported explanation.

---

## 4. The split — design

### Today (one dispatch)

`gbuffer_rayquery_nrc_wboit` does, per pixel:
raygen → ordered TLAS traversal → `resolveVertex` (material eval) →
`resolveVertexUnordered` (traversal + per-candidate material eval + decal binning
+ WBOIT) → PSR → G-buffer store.

Entry points:
- `src/dxvk/shaders/rtx/algorithm/resolve.slangh` — `resolveVertexUnordered`,
  candidate loop at ~line 890 (`for (uint step=0; step<kMaxUnorderedResolveSteps
  && rayQuery.Proceed(); step++)`)
- `src/dxvk/shaders/rtx/algorithm/geometry_resolver.slangh` — the resolver loop
  and the `perfGbStopAfter` cut points
- `src/dxvk/rtx_render/rtx_pathtracer_gbuffer.cpp` — dispatch + permutation key

### Proposed

**Pass 1** — ordered only. raygen + ordered traversal + material + G-buffer store.
Additionally stores what pass 2 needs (see plumbing).

**Pass 2** — unordered resolve, its own compute dispatch, own register budget.
Reads the primary ray + ordered hit distance, runs the unordered TLAS traversal,
candidate loop, decal binning, WBOIT, and composites into the G-buffer.

### Plumbing required

1. **Primary ray per pixel.** Origin/direction reconstructible from camera +
   pixel coord; `coneRadius` / `spreadAngle` are cheap to recompute. `tMax` must
   be the ordered hit distance — already in the G-buffer.
2. **`accumulatedRotation`** (`f16vec4`, PSR/portal virtual space). Needed by the
   billboard branch for primary rays. Either store it or restrict pass 2 to the
   non-PSR case initially.
3. **Decal composite is read-modify-write.** The unordered path mutates the
   surface material that lands in the G-buffer (albedo, normal, reflectivity,
   roughness via `decalMaterialInteractionBlend`). Pass 2 must read the stored
   G-buffer material, blend, and write back.
4. **Radiance/attenuation accumulation** (`currentRadianceAttenuation`,
   `currentEmissiveRadiance`) — currently accumulated into resolver state; needs
   its own target or a G-buffer channel.
5. **`surfaceIndex`** for the topmost decal (`maxSortOrder` path) — written to
   the G-buffer today; pass 2 must own that write.

### Known risks

- **The resolver loop can iterate.** `resolveVertexUnordered` runs inside a loop
  that handles portals and PSR. Splitting assumes one unordered invocation per
  pixel. Verify TF2 never needs more than one (portals are Portal/HL2 features;
  `rtx.enableSecondaryBounces=True` but that is the integrator, not this loop).
  If it can iterate, the split needs a loop count or must handle only iteration 0.
- **The `opacity → continueResolving` coupling** (the uno1 caveat) means ordered
  and unordered are not cleanly separable in the *current* control flow. Confirm
  that the ordered path's trip count does not depend on unordered results before
  assuming pass 1 can run standalone.
- Two dispatches = an extra full-screen read/write of the G-buffer material. At
  2.07 Mpix that is not free; budget for it.

### Verification — in this order

1. `[Perf.Shader]`: **both** new shaders should be well under 255 regs. If the
   gbuffer shader is still 255, the split did not remove enough and the timing
   will not move — stop there, do not proceed to timing.
2. `gb_primaryRays` + the new pass timer should sum to **< 24 ms**. Add a
   `[Perf.GpuPass]` entry for pass 2 or it is invisible.
3. Visual: decals (bullet holes, blood), particles, alpha-blended geometry, and
   emissive-blend surfaces unchanged. `rtx.enableAlphaBlend=False` and
   `rtx.wboitEnabled=True` in the current config — test with alpha blend ON too.

---

## 5. Measurement harness — use it, it is fast now

**`rtx.perfAutoSweepQuick = True`** — 3 steps (baseline / probe / baseline), ~45 s.
The middle step is **passthrough**: it runs with whatever `rtx.perf*` knobs are
set in rtx.conf, so testing one hypothesis needs no code change and no new table
row. Resolution floor came out at **0.087–0.62 ms**, i.e. ~17x more sensitive
than the 6-minute table. Validated against a known answer (rung 1: full sweep
−12.87, quick −14.81).

`rtx.perfAutoSweepQuick = False` → the full 26-step table (rev=3), ~6 min.

**Do not run the full table casually.** Two back-to-back 6-minute runs overheated
this laptop into a hard crash. It saturates the GPU with no idle. Use quick mode.

Verify a probe landed: every step line echoes `applied gb= uno= pom= tex= film=
mat= grad= coh= census=`, captured at apply time. Baselines must show all zeros.

**`rtx.logSurfaceCoverage` is a ~104 ms/frame switch, not a logging toggle.** It
arms 52 atomics per primary hit (the original 131 ms bug). Never True during any
timing run. The sweep's census step needs it, which is why the census must be run
as a separate pass.

**`RTX_FLUSH_MIN_US` / `RTX_FLUSH_INC_US` / `RTX_FLUSH_MAX_PENDING`** are env
vars, no rebuild. Currently `RTX_FLUSH_MIN_US=5000` (stock DXVK is 750).

---

## 6. If you want frame time rather than this stage

Ranked by measured size, all larger than the 13 ms this document is about:

1. **DLSS is off** (`rtx.dlssPreset=0`). Config says worth ~2x; it was disabled
   only to hold ray count constant for shader A/Bs, and that work is finished.
   ~12 ms of GPU for a config edit.
2. **The EVENT-query spin** — 54–57 ms/frame. A rate-limit fix landed this session
   in `d3d11_context_imm.cpp` (`FlushImplicit` was reading the clock on all ~400k
   polls/frame to make a decision possible ~36 times); measure it. The `ScopedCall`
   instrument on `GetData` itself costs a further ~20 ms/frame and is documented
   but unchanged.
3. **~50 ms of `outsideRtMs` is unmeasured.** `[Perf.Entry]` covers the immediate
   context only; `DxvkContext::updateShaderResources` and the rest of the backend
   are in the gap. Add a counter before believing anything about descriptors —
   marked at the site in `dxvk_context.cpp`.
4. **This is a `/Od` debug build** and `outsideRtMs` is CPU.
