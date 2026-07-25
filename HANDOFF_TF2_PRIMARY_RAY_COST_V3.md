# HANDOFF — TF2 primary rays: root cause found and fixed, 131 ms → 27 ms

Tree: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
Log:  `...\Titanfall-2-Digital-Deluxe-Edition-AnkerGames\Titanfall2\rtx-remix\logs\remix-dxvk.log`
Conf: `...\Titanfall2\rtx.conf`   (backup of the pre-sweep state: `rtx.conf.presweep.bak`)
Date: 2026-07-25.

**Supersedes `HANDOFF_TF2_PRIMARY_RAY_COST_V2.md`, which is wrong on every headline
claim and has NOT been edited. Do not work from it.** What it got wrong and why is
in §4 — read that before re-testing anything it lists as refuted.

HW: RTX 4080 Laptop, driver 610.62.0. Debug build (`_Comp64Debug`, `/Od`), CPU only;
shaders compiled separately. All measurements 1920x1080 native, DLSS off, camera
verified static (`curCam=(-14226,-8210.21,-3209.97)`, total spread ~0.03 units).

---

## 1. The result

`gb_primaryRays` was **131 ms**. It is now **27 ms** (25.7–29.3 across 10 samples).
`totalMs` went ~175 → ~78 ms. Same camera, same scene, same build settings.

**Root cause: a diagnostic block, not the renderer.**
`opaque_surface_material_interaction.slangh:1964–2232` runs 52 atomics
(`InterlockedAdd` / `InterlockedMax` / `InterlockedOr`) **per primary hit**. Many of
them target a *single shared address* — the Max/Or calls index
`[REGION * COVERAGE_SURFACE_SLOTS]` with no per-surface offset — so every thread in a
2.07 Mpix dispatch contends for the same word and they serialise.

`rtx.logSurfaceCoverage` gated only the **CPU-side readback**. The GPU writes ran
unconditionally on every primary hit whether anything read them or not. The block's
own comment said so and it was read as intent rather than as a bug.

**Fix:** the block is now wrapped in `if (cb.perfCoverageWrites != 0)`, driven by
`RtxOptions::logSurfaceCoverage()` (`rtx_context.cpp`, next to the other perf
constants). Setting `rtx.logSurfaceCoverage = True` restores the old behaviour
exactly — the instrumentation is unchanged, only conditional. That path is now
load-bearing for coverage debugging; verify it still works next time you need it.

Why it hid for so long: atomic serialisation has **no expensive instruction**, so
every hotspot hunt came back empty and that emptiness was repeatedly interpreted as
"uniformly throttled execution" (V2 §7). It is also `#ifdef RAY_TRACING_PRIMARY_RAY`,
which is why `gb_primaryRays` sat at 135 ms while `integrate_direct_rayquery`, doing
comparable material work, ran at 0.5 ms — a discrepancy V2 never accounted for.

---

## 2. Current frame budget — read before targeting 50 fps

```
gb_primaryRays   27 ms      <- was 131
totalMs         ~78 ms      (all instrumented RT work)
outsideRtMs   134-288 ms    <- everything outside the RT branch
```

**50 fps is 20 ms/frame total. `outsideRtMs` alone is 5-14x that budget.** Even
driving `gb_primaryRays` to zero cannot reach 50 fps. The primary-ray pass is no
longer the frame's bottleneck and optimising it further has a hard ceiling of 27 ms.

The dominant remaining cost is CPU-side and was measured before the fix:

```
[Perf.Entry]  totalMs=143.7   GetData=138.95 ms / 1,063,257 calls   (per frame)
[Perf.Query]  polls=15,880,192  notReady=15,880,120  notReadyPct=99.9995
              type0=15,880,132   (type0 = EVENT)
```

The game polls D3D11 EVENT queries ~1M times per frame and 99.9995% return not-ready.
`doNotFlush=60` out of 15.9M means the game is *not* asking to skip the flush, so it
expects dxvk to push work and let the fence retire.
`flushTuning(min=5000 inc=250 maxPending=6)` throttles the flush to once per 5000+
polls, so each fence takes thousands of spins to clear. **This is a dxvk-side flush
policy, not an inherent cost, and it needs no shader builds.** Re-measure it first —
the numbers above predate the coverage fix.

---

## 3. Instrumentation available (all default-OFF, all already built)

`[Perf.*]` is not dropped by the `log.cpp` prefix filter (`kFilteredTags` is a
denylist and `[Perf.` is not in it — confirmed, don't re-check).

| knob | what |
|---|---|
| `rtx.perfAutoSweep` | **Start here.** Walks every probe below in ONE run, 10 s each, medians them, prints a `[Perf.Sweep] SUMMARY` table, then terminates the process. Waits for gameplay (30 frames with world instances) before starting; freezes its clock if you hit a menu. `perfAutoSweepSeconds` / `perfAutoSweepSettleSeconds` / `perfAutoSweepExitOnFinish` tune it. Step table: `kPerfSweepSteps` in `rtx_context.cpp` |
| `REMIX_GBSTOP` | Compile-time stage ladder in `geometry_resolver.slangh:72`. 0=full (ship default), 1=raygen, 2=+traversal, 3=+unordered, 4=+material. One build per rung; gives register counts via `[Perf.Shader]` as well as timings |
| `rtx.perfUnorderedStopAfter` | Runtime, 1–6, cuts inside `resolveVertexUnordered`'s candidate loop |
| `rtx.perfMaterialStopAfter` | Runtime, 1–4, cuts inside `opaqueSurfaceMaterialInteractionCreate` |
| `rtx.perfSkipPom` / `perfSkipMaterialTextures` / `perfSkipThinFilm` | Independent material feature skips. The texture skip deliberately preserves the albedo/opacity read — see §5 |
| `rtx.perfUnorderedStepCensus` | `[Perf.UnorderedSteps]`: raw candidates/interactions per pixel. Needs `rtx.logSurfaceCoverage=True` AND `rtx.coveragePickRegionOnly=True` |
| `rtx.perfGbStopAfter`, `perfCheapTextureGradients`, `perfForceSampler*`, `perfBlasBoundsProbe` | Pre-existing, from V2 |

**Verify a knob reached the code before believing any A/B.** `[Perf.GbDispatch]`,
`[Perf.GbBranch]`, `[Perf.Shader]` (watch Binary Size change — the full gbuffer
shader is 800512) and the `lodBias`/`aniso` fields in `[D3D11Rtx.SamplerWrap]` exist
for exactly this. Three V2 tests silently never ran.

---

## 4. What V2 got wrong

| V2 claim | reality |
|---|---|
| Cost is `resolveVertex` / material eval, 98% of pass, "no hotspot" | The material function was expensive only because the coverage block lives at the end of it. Genuine material work is a few ms |
| Resolve loop is a 2.21x multiplier holding 54% of the pass | **The loop costs ~0.** Rung 4 (one iteration, tail cut) was within noise of full. V2's `0.38 + 2.21 x 49.1 = 108.9` matched the measurement by cancelling errors: per-iteration understated ~2.5x, multiplier overstated by the same factor |
| Unordered resolve = 0.3 ms | **42.5 ms.** Off by 140x. Not removable via `enableSeparateUnorderedApproximations` — that flag relocates the work rather than deleting it (looks free at rung 3 only because rung 3 already cut the destination) |
| Occupancy / register pressure is the mechanism (§7) | **Refuted.** Rung 4 runs at 168 registers vs 255 full — 12 warps vs 8, +50% occupancy — in identical time |
| POM + thin film removal bought 22%, "confounded" | Confounded in the other direction: with occupancy dead that 22% was execution work. Both now measure as noise; neither is a target |
| `finalBlit` ≈ 360 ms | Already retracted in V2 §4. It was 0.7–1.9 ms; the 360 ms was a debug-view readback left enabled for 3 hours |

Two real defects from V2 that still stand and are **unfixed**:
1. `geometryData.boundingBox` is computed over the whole vertex range, up to **753x
   oversized**. Anti-Culling is its only consumer, so it keeps instances alive that
   should be dropped. Traversal is unaffected (driver builds from real BLAS contents).
2. Every Remix sampler control is dead in this fork — `patchSampler` is gated on
   `samplerOverride == nullptr`, which never holds. Kills `nativeMipBias`,
   `useAnisotropicFiltering`, `maxAnisotropySamples`, **and the DLSS mip bias**
   (a standing visual-quality bug with DLSS on).

---

## 5. Method notes that earned their place

**Bisect, don't theorise.** Every result that survived came from an ablation ladder.
Every retraction came from reasoning about mechanism first. V2 said this and it is
still the single most useful line in it.

**Cross-run comparison is where the errors came from.** The 360 ms `finalBlit`, the
0.3 ms unordered stage, the 2.21x multiplier — all cross-run. `rtx.perfAutoSweep`
exists to make comparisons intra-run; prefer it.

**Use medians, and print min alongside.** The first sweep reported four steps as
*slower than baseline* — impossible for work removal — because single hitched frames
(282 ms, 320 ms) dragged the means 10–20 ms. Minima were already correct. Both
columns are printed now; when median and min disagree the step caught a hitch.

**Watch for probes that relocate work instead of deleting it.** Three separate times:
`enableSeparateUnorderedApproximations` (moves unordered resolve into the ordered
loop); `perfSkipMaterialTextures` before it was fixed (suppressing the albedo read
changed `opacity`, which drives `continueResolving`, which changed the resolve loop's
trip count — it now exempts that one read via a `drivesControlFlow` flag); and the
material-ladder rungs, whose zero-init sets `opacity = 1.0` so **rung 1 collapses the
resolve loop** while rungs 2–4 return after the real opacity is computed and do not.

**There is ~15 ms of upward drift across a 2-minute sweep**, so late steps are
penalised versus early ones. Effects smaller than that are not resolvable with the
current fixed step order. If you need them, make the sweep A-B-A (re-measure baseline
between every probe); the step table is one array in `rtx_context.cpp`.

---

## 6. If you still want primary rays faster

Ceiling is 27 ms and the frame does not care until `outsideRtMs` is dealt with. That
said, the honest remaining breakdown was never re-measured after the fix — **every
number in §4's ladder predates it and is now historical.** Re-run
`rtx.perfAutoSweep` first; the decomposition will have changed completely, and the
old 42.5 / 74.5 split almost certainly evaporates with the atomics gone.

Only then pick a target. Do not port the old ladder's conclusions forward.
