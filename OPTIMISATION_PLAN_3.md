# OPTIMISATION PLAN 3 — RESIDENCY, NOT PARALLELISM

**Written:** 2026-08-20. Branch `architecture-overhaul`.
`[M]` measured `[I]` inferred `[U]` unknown

> **THE ONE-LINE THESIS.** Every plan so far has tried to make ~1,340 draws a
> frame cheaper. The draw count is not a constant — it is a **policy choice this
> project made**, enforced by five byte-patched kill switches, to work around
> `numFramesToKeepInstances = 1`. Fix the residency and the culls can come back
> on, and the draw stream that costs 58 ms collapses. Nothing else on the table
> is worth more than ~5 ms. **This is worth 25–35 ms.**

---

# 0. THE TWO MEASUREMENTS THAT MOTIVATE THIS

## 0.1 The chain A/B — TAKEN 2026-08-20, and it closes 19g §7.2 item 1 `[M]`

Missing since 19f. Both sides all-probes-off, `shardInstanceProcessing=False`,
verify off — one variable.

```
                     chain ON (19g §4 run 3)   chain OFF (03:53)
FRAME                53.91  (18.5 fps)         57.99  (17.2 fps)   +4.08
frame thread         52.92  busy 98.2%         57.11  busy 98.5%
  D3D11 entry points   --                      33.10
  between entry pts    --                      25.02
dxvk-cs              36.11  idle 17.65         41.28  idle 19.70
GPU                  12.35  idle 36.94         12.95  idle 40.17
```

**The chain buys ~4 ms.** Cross-checked a second way: the 03:40 chain-ON window
read 56.32 ms with the probes ON, and the probes cost ~2.5–3 ms, putting
probes-off chain-ON at ~53.5 — agreeing with 53.91 from a different sitting.

Read it honestly. It is a real win and it is **not** transformative. It also
refutes the strong form of the "just move work to workers" thesis: the chain is
the best-case version of that idea in this codebase — a dedicated worker, no
per-draw scheduling overhead — and 4 ms is what it returns. Phase 2b, the other
attempt, cost **+14.5 ms** (R45). The average of the only two data points we
have for moving work off this frame thread is roughly zero.

CAVEAT `[I]`: cross-sitting comparison, and the scene drifts 54–58 ms. Worth
one same-sitting confirm before it goes in a handoff as fact.

## 0.2 Where the frame thread actually goes `[M]` (03:53, probes off, xMin 1.00)

```
frame thread                    57.11 ms   busy 98.5%
  D3D11 entry points            33.10      57%   <- 1,338 draws x 24.7 us
  between entry points          25.02      43%   <- TITANFALL'S OWN CODE
    afterQueryEnd                4.32            (6 gaps -- FIXED cost)
    (rest, ~20.7)                              (per-draw gaps -- SCALES)
```

Complementary, not overlapping — cross-validation PASSes every window.

Everything Optimisation Plans 1 and 2 attacked lives inside the 33.10. The 25.02
has never been addressable **because no plan has questioned the draw stream.**

---

# 1. THE REFRAMING

`drawsIn = 1,338/frame`. Not because TF2 is heavy — because **TF2's culling is
deliberately defeated**. From `CULLING_BIBLE.md` and the live `rtx.conf`, all
default/set ON (`cullOff.X = True` DISABLES cull X):

| geometry class | engine cull | switch | state |
|---|---|---|---|
| entities / renderables | `sub_1801A8350` (client) | `cullOff.frustum` | True |
| walls / terrain / ground | `sub_1802E8DA0` | `cullOff.worldFrustum` + `worldPortal` | True |
| area enqueue / portal flood | `sub_1802EB620` | `cullOff.areaPortal` + `areaClip` + `areaSkipClip` | True |
| area visit / chain merge | `sub_1802ED900` | `cullOff.areaMergeSalvage` | True |
| static props / `.mdl` | `sub_1801B2DD0` (engine vtable) | `cullOff.staticPropFrustum` | True |

**Why they were turned off:** raster culling is not ray-tracing visibility. A
culled object still casts shadows and appears in reflections and indirect light.
When the engine culled it, the draw stopped, and —

```
rtx_options.h:1626   RTX_OPTION("rtx", uint32_t, numFramesToKeepInstances, 1, "");
```

— the instance **retired the very next frame**, so the geometry vanished from
the RT scene entirely. The culls had to go.

**That is a Remix residency limitation being paid for with the engine's entire
culling system.** The engine is being forced to re-submit the whole level every
frame so that Remix does not forget it.

The codebase already knows this is the wrong trade and already built the
exception. `numFramesToKeepSubViewInstances = 16` exists precisely because the
3D-skybox fan is throttled by the engine and one missed frame retired ~90
instances at once. The mechanism is proven; it has simply never been generalised.

Note also (`rtx_options.h:2350`) that Remix's OWN frustum cull, at
`mergeInstancesIntoBlas`, carries the explicit warning that it *"does NOT save
the CPU cost of the draw call itself — cull here to save GPU/BVH work, not
draw-submission CPU."* The GPU is idle 40 ms. Remix-side culling is the wrong
cull. **Engine-side culling is the whole prize, and it is switched off.**

## 1.1 The prize, sized `[I]`

Entry-point cost is ~24.7 µs/draw and scales linearly. `between` is 4.32 ms
fixed + ~20.7 ms per-draw.

| draws culled | entry pts | between | frame thread | fps `[I]` |
|---|---|---|---|---|
| 0% (today) | 33.10 | 25.02 | **57.1** | 17.2 |
| 40% | 19.9 | 16.7 | **~37** | ~27 |
| 50% | 16.6 | 14.7 | **~31** | ~32 |
| 65% | 11.6 | 11.5 | **~23** | ~43 |

Against the rest of the board: chain 4.0, `bt_extractXf` 4.76, `drawSnap` ~5.0,
`sv_keys` 1.26. **The residency lever is an order of magnitude larger than every
other item combined**, and it is the only one that touches the 25 ms of engine
code.

Assumption stated plainly: that cull fraction is `[U]`. Phase 0 measures it
before anything is built.

---

# 2. PHASE 0 — SIZE IT IN ONE SITTING, BUILD NOTHING `[M] required`

Do this before writing a line of code. Every later phase is funded or cancelled
by the number it returns.

Flip the cull switches OFF **one class at a time** (`cullOff.X = False` re-enables
that engine cull). Accept the visual breakage — this is a measurement, not a
build. Per class record `drawsIn`, `FRAME`, `D3D11 entry points`, `between entry
points`, and what visibly breaks.

## 2.1 FIRST ATTEMPT — NO RESULT. THE CAPTURE NEVER REACHED GAMEPLAY `[M]`

> **RETRACTION, same day.** An earlier version of this section read the run
> below as "the area layer collapses the scene" and rewrote the ladder around
> it. That was wrong. Every `[Perf.Report]` in the capture reads `inst=0
> blas=0` — from frame 9000 to the hang — with `drawsIn=1` at 1.30 ms
> (769 fps). That is a MENU/LOADING screen for the entire window. Remix never
> held a scene at any point in it, including before any cull decision could
> matter, so there is no healthy period to compare against and the run says
> NOTHING about culling. The project's own rule (gate instrumentation on
> gameplay, skip menu/loading frames) exists for exactly this and was not
> applied. The claim that `areaPortal`/`areaClip`/`areaSkipClip`/
> `areaMergeSalvage` are load-bearing bug fixes rather than reclaimable culls
> is **UNVERIFIED** and is contradicted by the recollection of whoever wrote
> them (worked fine without areaClip; no hooks needed beyond the camera).
> Do not inherit it. Re-measure in gameplay before believing either way.

```
04:06:22  f 9000-10050  drawsIn=1   1.30 ms (769 fps)  inst=0 blas=0
04:06:47  f 10150       drawsIn=3   1028.65 ms         inst=0 blas=0
04:07:06  f 10400       drawsIn=14  41.79 ms           inst=0 blas=0
04:07:14  f 10450       drawsIn=5   148.09 ms          inst=0 blas=0
```

**The hang is real and is a separate open item.** At the freeze the frame
thread was in a kernel wait with every return address inside `nvoglv64.dll`
(base 0x7FFEA76C0000: +0xDD50E3, +0xEC595B, +0xECCD74, +0xEF3D12, +0xF6B9F5,
+0x12C337C), `dxvk-cs` and the PRESENT thread had both fallen to 0% in
`[ThreadCensus]`, and one `tier0.dll` worker was spinning at ~5.4%. It happened
during load / at the transition into gameplay, not in steady-state gameplay.
Whether it is related to `cullOff.enable=False` at all is `[U]`.

**What IS verified, in code:** `cullOffUpdate()` (d3d11_rtx.cpp:6501) has a
fast path — `if (!master && !s_anyApplied && !g_cullOffRLodForced) return;`.
With `enable=False` set before launch nothing is ever applied and no byte is
ever written, so that configuration is genuinely stock engine. The hang is not
patch damage and there is no half-patched state to suspect.

**Redo the run and get into actual gameplay before reading anything.** Confirm
`inst` and `uniqueBlas` are non-zero and `drawsIn` is in the ~1,300 range in
the baseline before trusting any comparison.

## 2.2 (SUPERSEDED — kept only as the shape of the ladder, NOT its verdicts)

`rtx.cullOff.enable = False` (master off, "the game culls normally"). Result:

```
HYGIENE  drawsIn=14   inst=0  uniqueBlas=0  texTotal=0
HYGIENE  drawsIn=5    inst=0  uniqueBlas=0
FRAME    41.79 -> 148.09 ms, then the process hung
[RenderListProbe] f=10494 viewFlags=0x8 firstPtr=0x0 total=8
```

Draws did not fall from 1,338 to ~600. **They fell to 5, and the RT scene went
empty.** At the hang the frame thread was blocked in a kernel wait with every
return address inside `nvoglv64.dll` (base 0x7FFEA76C0000: +0xDD50E3, +0xEC595B,
+0xECCD74, +0xEF3D12, +0xF6B9F5, +0x12C337C), `dxvk-cs` and the PRESENT thread
had both dropped to 0% in `[ThreadCensus]`, and one `tier0.dll` worker was
spinning at ~5.4%.

**WHY, and it is in CULLING_BIBLE.md all along:** the master switch does not
turn off "culling". It removes ALL the patches, and only two of them
(`frustum`, `staticPropFrustum`) defeat ordinary visibility culls. The other
five defeat **engine defects**: `sub_1802ED900` returns −1 on a degenerate
portal-chain merge and the caller **drops the area and everything behind it**;
`areaPortal`/`areaClip`/`areaSkipClip` are the three area-layer fixes; the
eighth reject `worldPortal` is the pitch-vanish. Unpatched, TF2's area-portal
flood collapses and almost nothing is ever enqueued. That is the original
"look down and everything disappears" bug at full strength.

**So the ceiling run does not exist as a single flip**, and this does not
refute the thesis — it refutes the method. Correct ladder:

```
run 0  baseline, master True, all as shipped              (have it: 57.99)
run 1  cullOff.staticPropFrustum = False    props        <- HEALTHY cull
run 2  cullOff.frustum           = False    entities     <- HEALTHY cull
run 3  runs 1+2 together                                 <- the real ceiling
```

**Leave `worldPortal` / `areaPortal` / `areaClip` / `areaSkipClip` /
`areaMergeSalvage` at True in every run.** They are not culls to reclaim; they
are bug fixes. `worldFrustum` is the ambiguous one — it is a real frustum cull
but §0.2f showed its reject set was not exhaustive and `worldPortal` rides with
it. Try it only after runs 1–3 are understood.

This caps Phase 0's measurable prize at the props + entities share of the draw
stream rather than the whole 1,338. `[U]` how large that share is — which is
exactly what runs 1–3 return.

Keep `rtx.conf` backups per run (`rtx.conf.bak-pre-phase0` is the pre-Phase-0
state). Restore to master True with all individuals as shipped afterwards.

---

# 3. THE ARCHITECTURE — RESIDENCY WITH A STATIC/DYNAMIC SPLIT

The user's framing is a pull architecture, and that is right, but the important
simplification is this:

> **You do not need dirty-tracking for the classes that pay for this.**

World geometry and static props are static **by construction** — the engine's own
names for them say so. They cannot move, so a resident instance can never be
stale. The dirty problem only exists for entities, which are a minority of draws
and can simply keep the current every-frame behaviour.

That splits the work into a tractable Phase 1 and a deferrable Phase 3.

```
                    engine cull ON
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   culled draw                        submitted draw
        │                                   │
        ▼                                   ▼
  no D3D11 call                        SubmitDraw (as today)
  no entry-point cost                        │
  no engine gap cost                         ▼
        │                              touch residency
        ▼                                    │
  RESIDENT INSTANCE                          │
  transform held                             │
  surface slot held  ◄──────────────────────┘
  BLAS/geo held
        │
        └──► TLAS every frame, whether drawn or not
```

## 3.1 Phase 1 — residency for static classes

**Goal:** an instance tagged static survives an unbounded number of untouched
frames without retiring, losing its ordered-surface slot, or having its BLAS
GC'd.

- A residency category + keep policy, generalising the existing
  `IgnoreAntiCulling` / `numFramesToKeepSubViewInstances = 16` path from a magic
  16 to "resident until invalidated".
- `numFramesToKeepBLAS` / geometry / material-texture keeps must extend to match,
  or the instance survives and its geometry does not.
- Entry criterion: the draw's class is world or static-prop. Take that from the
  cull-site classification the `cullOff` sites already have — do **not** invent a
  heuristic (no size thresholds, no allowlists).

**Acceptance:** with `cullOff.worldFrustum = False`, spin 360° and sweep pitch.
No geometry pops. `drawsIn` falls. `FRAME` falls. Reflections of off-screen
geometry still present.

## 3.2 Phase 2 — identity stability, the real blocker `[M] partially known`

Residency is worthless if the key churns: the same object arriving under a new
identity produces a duplicate resident instance instead of a hit, and the scene
grows without bound.

This is **already a known defect** with a documented instance. From
`rtx_options.h:3362`, TF2's path-10 prop fanout round-robins transient VB/IB
buffers, and `MakeBoneStablePropId` keys on `vbPtr`/`ibPtr` — so one static prop
takes a new propId whenever the engine rotates buffers (measured: 342 distinct
fanout positions, 518 distinct hashes). The option comment states the
consequence outright: *"Prereq for granting these instances the longer GC keep."*

So Phase 2 is a named, half-solved problem, not a research project:
- `boneStablePropIdFanoutPositionOnly` (default off) is the existing prototype.
- **Acceptance:** distinct-hash count plateaus to ~distinct-position count, and
  resident instance count is flat over a 5-minute stationary capture. A rising
  instance count is the failure mode and it must be a hard gate.

## 3.3 Phase 3 — dirty tracking, only for what actually moves

Only needed once entities join residency. Two candidate sources, in the order
this project's own rules prescribe:

1. **Logs before debuggers.** `RtlCaptureStackBackTrace` at a dxvk-controlled
   site to find who writes an entity's transform — not IDA first.
2. IDA second, and much of it is already done: `CULLING_BIBLE.md` §5 maps
   renderable → model name, §5.1 the `IClientRenderable` vtable, §5.2 the
   `C_BaseEntity` / `C_BaseAnimating` subobjects.

Until then entities stay non-resident and are drawn every frame. That is correct
and costs only their share of the draws.

---

# 4. THE HAZARDS, ALL OF WHICH THIS CODEBASE HAS ALREADY HIT

Do not rediscover these.

1. **Ordered-surface slot reallocation.** `rtx_options.h:1632` documents it: when
   ~90 sub-view instances retire together their slots are reallocated, and
   GBuffer pixels still referencing the old slots render as the new occupant —
   *"large black/wrong rectangular blocks"*. Residency must hold the **slot**,
   not merely the instance.
2. **Resident ≠ refreshed.** `s2s two views` was exactly this: intro-era
   instances stayed alive while steady-state draws failed to refresh them, and
   the scene showed two different worlds. A resident instance that a later draw
   *should* have updated and didn't is a silent correctness bug with no FAIL to
   catch it.
3. **The culls were disabled for reasons beyond residency.** The pitch-vanish
   and light-leak fixes (§0.2f, §0.2g of the bible) were about geometry RT needs
   for *indirect light and occlusion*, not just direct visibility. Re-enabling
   `areaPortal` may remove an occluder whose absence floods light in even though
   nothing on screen is missing. Judge per class, from Phase 0's observations.
4. **`areaDegen` is permanently OFF and must never be enabled** (bible §0.2g).
5. **Invariance, not a target number.** The bible's verification standard: the
   bug class here is view-dependence, so fix the camera, sweep all pitch and yaw,
   and require the count to be **flat**. A good average over a sweep hides the
   band where it drops.

---

# 5. WHAT THIS PLAN DELIBERATELY DOES NOT DO

**It does not build a task graph.** The UE/RE-Engine proposal is a good north
star and mostly describes what this branch already is (DrawSnapshot ≈
`FMeshDrawCommand`, thread-local draw state, per-frame arena, the chain).
Rejected as the *next* step on three measured grounds:

- Its ceiling is arithmetic: if **all** Remix frame-thread work went to zero,
  57.11 → ~25 ms, because 25.02 ms is engine code it never touches. Residency
  attacks both halves.
- Both attempts to move work off this frame thread returned +4 ms (chain) and
  **−14.5 ms** (2b). The transfer is not free and has never been shown to scale.
- Its best idea — carrier state as dependency edges rather than a frame-thread
  pin — requires enumerating what each draw reads. `ExtractTransforms` touches
  **64 members across 657 references** with **no instrument at all** (19g §7.3).
  You cannot schedule around dependencies you cannot list, and `FAIL = 0` means
  "none fired", not "the set is complete" (R46).

**It does not chase `bt_extractXf` further.** Now fully attributed:

```
bt_extractXf 4.76 = serve 2.38 + derive 2.38
  sv_keys   1.26   <- 53% of the serve half
  sv_lookup 0.39
  sv_compose 0.28  <- the compose is nearly free; §0.2 candidate 1 REFUTED
```

`sv_keys + sv_lookup = 1.65 ms/frame` is the cache deciding whether it may skip
work, against 0.52 ms actually substituting. Content-addressed / incremental key
hashing is a real ~1 ms win and is **the one item worth doing in parallel with
this plan** — it needs no architecture. But note: culling reduces draws, and
`sv_keys` is per-draw, so Phase 0 shrinks this target too. Do it after.

---

# 6. ORDER OF WORK

| # | work | gate | worth `[I]` |
|---|---|---|---|
| 0 | **Cull-class A/B sweep** | none — do it now | sizes everything |
| 1 | Static-class residency | Phase 0 ≥ 15 ms | 15–30 ms |
| 2 | Identity stability | Phase 1 shows churn | correctness gate |
| 3 | Dirty tracking for entities | Phases 1–2 green | 3–8 ms |
| — | Incremental CB key hashing | independent | ~1 ms |
| — | Chain stays ON | measured +4 ms | 4 ms |
| x | Task graph / GPU-driven | **not now** | capped at 25 ms |

## Immediately

Restore `rtx.xfDeferChain = True` — the A/B is done and the chain wins by 4 ms.
Then run Phase 0.

---

# 7. RULES ADDED

**R48. A workload is not a constant until you have checked who chose it.** Three
optimisation plans treated 1,340 draws/frame as the environment. It is a setting
this project wrote, in five byte-patched kill switches, to compensate for a
one-frame instance keep. The largest optimisation available was upstream of
everything being optimised.

**R49. Prefer eliminating work to relocating it — and this codebase has the
receipts.** The split cache (elimination) removes ~3.1 ms/frame net. The two
relocations measured +4 ms and −14.5 ms. Relocation pays the transfer twice: once
to hand the work over and once to keep the halves consistent.

**R50. Residency is the ray-tracing answer to rasterisation culling.** A
raster-culled object still needs to cast shadows and appear in reflections. The
fix is for the RT scene to remember it, not for the engine to keep re-drawing it.
`numFramesToKeepInstances = 1` is what makes the engine's culling unusable, and
`numFramesToKeepSubViewInstances = 16` is the same fix already shipped for one
special case.
