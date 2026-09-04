# PSDT — Perceptual Scene Display Transform

`rtx.tonemap.operator = 8`

A fifth entry in the tonemap operator list, and the only one that is not a tone
curve. The other four are pure functions of one pixel. PSDT is a display
transform: it converts physically reconstructed scene radiance into the image a
viewer adapted to *this* scene, looking at *this* display, would find most
plausible.

The existing operators stay exactly as they are. They are the controls this is
measured against, and that is worth more than the few kilobytes they cost.

---

## Why not just tune GT7

Because the two inputs that most determine what a pixel should look like are not
in the pixel: what the viewer is adapted to, and how much room the display has
left. A per-pixel curve cannot see either, so every per-pixel curve has to guess,
and every guess is a compromise between the dark corridor and the sunlit window.

RTX Remix makes this worse in a specific way. The reconstructed scene carries far
more range than the original game ever did, has ray-traced emissives that reach
values no display can represent, and is temporally reconstructed. A curve tuned
for a car in daylight is being asked a question it was not designed for.

The other reason is stacking. Local exposure, a global curve, chroma compression
and a gamut clamp each independently squeeze the same signal. Each looks
reasonable alone. Together the colour gets squeezed four times, and the image
stops saying "that is a physically bright, saturated red light" and starts saying
"that is approximately red-ish and bright". That is not a curve problem. It is an
architecture problem, and it is the one this addresses.

---

## The contract

Four jobs, four owners, and nothing owns two of them:

```
Auto Exposure    what brightness am I adapted to?          (upstream, unchanged)
PSDT curve       how do I reproduce the adapted scene?     psdt_curve.slangh
Colour volume    how do scene colours fit the display?     psdt_transform.slangh
Glare            how does a very bright source appear?     psdt_transform.slangh
```

Keeping those apart is the whole design. Every stage can be turned off
independently, and when the image is wrong it is possible to say which stage is
wrong.

---

## Pipeline

```
                        RTX RENDERED HDR (linear Rec.709, post-exposure)
                                     |
        +----------------------------+----------------------------+
        |                                                         |
        v                                                         v
   psdt_analysis                                          tonemap histogram
   classify every pixel BODY / SOURCE / GLARE              (already existed)
   reduce each 16x16 block to 4 numbers                            |
   accumulate temporally with reprojection                         |
        |                                                          |
        v                                                          |
   psdt_downsample  x (levels-1)                                   |
   the mip chain IS the adaptation pyramid                         |
   and IS the glare kernel                                         |
        |                                                          |
        +----------------------------+-----------------------------+
                                     v
                              psdt_state  (one workgroup)
                   source-aware anchor, scene white/black, scene
                   intent, asymmetric inertia, cut detection, and
                   every runtime option resolved into 64 floats
                                     |
                                     v
                         tonemapping_apply  ->  psdtApply()
                                     |
     +-------------------------------+-------------------------------+
     |            |            |            |           |            |
     v            v            v            v           v            v
 multi-scale  contrast    exposure-aware  colour     hue         perceptual
 adaptation   budget      curve T(x;E)    volume     trajectory  glare
                          + local          demand    + spatial
                          contrast         vs the    white
                                           boundary
                                     |
                                     v
                            display gamut fit
                                     v
                          colour grading / sRGB / dither
```

Three new compute dispatches, all tiny. Nothing else in the frame moved.

---

## The parts, and what each is for

### Source-aware adaptation

Conventional metering asks "how bright is this frame". That is the wrong question
the moment the scene contains a light, which every Titanfall interior does:

```
   dark corridor  +  small sunlit window

     histogram sees:  dark mass + a huge bright spike
     a viewer sees:   a dark corridor, and a window
```

`psdt_analysis` sorts pixels into three roles. **BODY** — walls, characters,
weapons, ground — determines exposure and nothing else does. **SOURCE** — lamps,
sky, muzzle flashes — is *excluded* from the estimate rather than clamped,
because a clamped sun still drags a mean and an excluded one does not. **GLARE**
is too bright to be a value at all; its energy is kept and spent later as spatial
extent.

`rtx.tonemap.psdt.sourceExclusion` is the blend between the histogram median and
the source-excluded body mean. It is the single setting that most changes how a
dark interior with a bright opening reads.

### Multi-scale adaptation under a contrast budget

The eye does not adapt to a pixel, and it does not adapt to the whole frame
either. Three scales — 16, 64 and 256 pixels — are pooled with scene-dependent
weights, so a large bright room and a small bright lamp are not treated alike.

Then the budget, which is the most important constraint in the system:

```
   shadows      2.00 stops of adaptation allowed
   midtones     0.55
   highlights   0.25
   emitters     0.05
```

Local adaptation with no budget converges on "every region is middle grey". That
improves visibility while destroying the scene's brightness hierarchy — and that
hierarchy carries as much information as the geometry does. The budget is what
lets the local adaptation be strong without the image going flat.

The budget is indexed by the *region's* level, deliberately not by the pixel's
own. Making it per-pixel would let adjacent pixels of different brightness sit
under different adaptation, which is a discontinuity in a signal that has to stay
smooth or it rings.

### An exposure-aware curve

Log2 scene in, log2 display out. An exponential toe, a linear segment, an
exponential shoulder, C1 at both joins, strictly monotonic everywhere, and
asymptotic rather than clamped — the curve itself never clips. Clipping is a
display event, handled once at the end, and what does not fit becomes glare.

Every parameter is resolved per frame from the adaptation state. This is
`T(x; E)`, not `T(x)`: a night scene gets more slope and a longer linear shadow
region, daylight gets more compression, a frame full of explosion gets an earlier
and softer shoulder and hands the rest to the glare model.

### Local contrast, restored before the curve rather than after

The obvious formulation is to map the base and then add the detail back at some
blend of the scene's contrast and the curve's. It has two faults, and both were
found by measurement rather than by looking:

* it is a first-order Taylor expansion of the curve, so a bright object against a
  dark surround extrapolates past the shoulder and clips — grey clipped 2.5 stops
  above the anchor;
* the blend weight varies per pixel, and near the top of the shoulder its
  derivative outruns the curve's own slope, which **inverts the transfer
  function**. Small (8e-5 linear) but real: 123 inversions across the test grid.

Expanding the detail in the scene domain and letting the curve map the result
fixes both, and removes two terms doing it. The boost is a property of the
neighbourhood, so at fixed base the output is a monotone function of a monotone
function and cannot invert anywhere, by construction rather than by measurement.
And a pixel far above its base is pushed *further* into the shoulder, so a lamp
core is never sharpened — not because something prevents it, but because the
mechanism that would sharpen it is the same one that compresses it.

### Colour volume, measured rather than guessed

The question is "how much of this colour can the display reproduce at this
luminance, and what is the least perceptually damaging way to get there".

The measurement is exact and cheap: find the largest `t` for which
`grey + t·(rgb − grey)` still lies inside the display cube. That single number is
how much room is left, and everything else is built on it.

One thing here departs from the obvious formulation, and the departure was forced
by measurement. Writing the pressure as `P = 1 − (1−Py)(1−Pc)(1−Pg)` reads well
and is an **OR**: any one high term takes the result to 1. Chroma is high for
every saturated colour, in gamut or not, so that form strips a mid-luminance red
the display can show perfectly well. Measured: **0.26 of its chroma retained** —
exactly the "everything goes grey" failure the stage exists to prevent. What is
wanted is the AND. So the geometry leads and perception only decides how early
the easing starts.

Compression is a soft knee that asymptotes to *the boundary*, not to zero. A
colour converges on the most colourful thing the display can show at its
luminance — which is the difference between a neon sign that stays a neon sign
and one that becomes a white blob.

### Luminance concession

A saturated red cannot be as bright as a white: the most saturated Rec.709 red
has a luminance of 0.21. When the tone stage asks for a brightness the hue cannot
support, something has to give, and there are only two candidates — bleach the
colour to keep the brightness, or let it be darker to keep the colour. Film lets
it be darker, and that is most of why film highlights read as coloured light
rather than as blown paper.

`luminanceConcession` states that trade instead of assuming it. At the default a
fully saturated colour renders 0.28 stops below a neutral of the same scene
luminance and retains 0.44 of its scene chroma, against 0.35 with the concession
off. Only saturated colours are affected: a neutral is never outside the volume,
so the brightness of the rest of the frame is untouched.

### Hue trajectory

Two paths are computed. One preserves chromaticity exactly; the other is the
per-channel curve, whose dominant channel saturates first and which therefore
bends red → orange → yellow-white on its own. The hue is rotated from the first
towards the second in proportion to how far outside the volume the colour landed,
along the shortest arc so the red/magenta wrap has no jump in it.

That gives a controlled, continuous path to white without handing the whole image
to a per-channel curve. Measured largest hue step across a 1/42-stop sweep:
**0.125°**, against 0.845° for GT7 and 0.576° for Hable.

### Spatial white point

The neutral that chroma compresses towards is the low-frequency local mean colour
rather than D65, so a tungsten-lit interior stays warm as it desaturates instead
of drifting towards daylight white.

### Glare, from the same pyramid

Not bloom. Bloom is a lens artifact applied to the image; glare is how an
intensity that cannot be shown as a value gets shown as an area instead. It is
separate from `rtx.bloom.*` and both can run.

The kernel is the adaptation pyramid, reused — a weighted sum down the mip levels
is a stack of progressively wider box responses, at the cost of a handful of
texture fetches and no extra passes at all. The radius is not a parameter and
cannot be: a dim source drops below the visibility threshold after one or two
levels while a very bright one survives several more, so the halo grows with
roughly the log of source luminance on its own — and inherits the pyramid's
temporal stability while doing it.

The threshold is anchored to the adaptation state, not to an absolute value,
which is the whole reason a car headlight is blinding at night and invisible at
noon.

### Temporal adaptation, and cuts

Measured state carries inertia; resolved parameters do not. So the transform is
deterministic given the state, while the state has a time constant. Get that the
other way round and the image either lags or flickers.

Adaptation is asymmetric in the direction human vision actually is — light
adaptation fast, dark adaptation slow. A player turning 180°, a teleport, a
respawn and a scripted cut all look identical from here: the whole luminance
description of the frame moves at once. Easing through that takes seconds and
reads as the image washing out, so the state re-anchors instead.

---

## A correction this makes, and does not propagate

The renderer's working space is linear **Rec.709 / sRGB primaries**, D65. That is
observable, not assumed: the histogram, the auto exposure and the colour grading
all weight luminance with `calcBt709Luminance`, and the apply shader finishes
with the sRGB OETF and no primary conversion.

`gt7.slangh` and `psycho17.slangh` both treat the framebuffer as Rec.2020. That
is wrong for this pipeline, and it desaturates — Rec.709 values interpreted as
Rec.2020 describe a smaller volume than they actually occupy. PSDT converts
explicitly. **The other two are deliberately left alone**, because they are the
controls, and a control you have quietly edited is not a control. Fixing them is
a separate change with its own before/after.

---

## Measured results

`python3 tools/psdt/psdt_suite.py`. Defaults, SDR sRGB, 100 nit reference.

```
    operator    chroma kept  boundary use  max hue step  midtone contr   clipped   Y inv
    PSDT              0.835         0.799        0.125d          1.150     34.4%       0
    GT7               1.002         0.711        0.845d          1.307     59.3%       0
    Reinhard          0.966         0.910        0.299d          0.840     52.6%       0
    Hable             0.880         0.817        0.576d          0.902     60.3%       0
```

PSDT clips 34.4% of the ladder against 52–60% for the controls, has the smoothest
path to white, and holds a midtone slope of 1.15 where Reinhard and Hable are
below 1.0 — that is, flat. GT7's 1.307 is more midtone contrast than PSDT, bought
with 59% clipping.

Reinhard's high boundary use is not a win: it does not compress chroma at all, it
clips, which is also why its clipping figure is 52.6%.

All seven perceptual invariants hold:

```
  1. brightness ordering preserved            0 inversions
  2. small bright object vs dark surround     4.05 display stops apart
  3. mid-luminance red hue held               -0.00 deg
  4. chroma trajectory continuous             max step 0.00081
  5. deterministic given the state            PASS
  6. adaptation budget respected              0.149 stops shift for a 6-stop brighter neighbourhood
  7. emissive stays as colourful as the       worst boundary use 0.824 at +2 stops
     display allows
```

The curve is monotonic over 40 stops with a worst step of exactly 0, C1 at both
joins, and its analytic derivative matches a central difference to 1.3e-9.

Three defaults are set by sweep rather than by taste — `highlightRolloff` 0.45,
`luminanceConcession` 0.8, `chromaPreservation` 2.5. `psdt_sweep.py` shows 0.45
dominating 0.60 on every axis it trades against.

---

## What this does not do

* **No HDR output.** The transform is display-aware and will grade for a peak
  above diffuse white, but this pipeline ends in an 8-bit sRGB encode, so what
  you would see is an SDR-normalised preview of that grade. Leave
  `displayPeakNits` equal to `displayRefWhiteNits`; that path is exact.
* **The adaptation pyramid is a plain box pyramid, not edge-aware.** Auto Exposure
  Plus builds an edge-aware one for exactly this reason. Measured spatial
  stability is good (0.149 stops of shift for a neighbourhood 6 stops brighter,
  where the budget allows 0.25), so this has not been worth the extra passes —
  but it is the first place to look if haloing ever shows up.
* **Cut detection is histogram-based, not camera-based.** A large simultaneous
  move in the anchor and both percentiles reads as a cut. It is self-contained
  and needs no camera plumbing; a camera-cut hint from `RtCamera` would be
  strictly better and is a small change.
* **It has not been run.** Everything above is measured on the CPU reference in
  `tools/psdt/`, which is a line-for-line port of the shaders. The shaders
  themselves have not been compiled or executed — this branch cross-builds for
  Windows and the toolchain is not on this machine.

---

## Files

```
src/dxvk/shaders/rtx/pass/psdt/
  psdt.h                        bindings, state layout, push constants
  psdt_perceptual_space.slangh  ICtCp / Jzazbz behind one interface, display volume
  psdt_curve.slangh             the luminance curve and its analytic derivative
  psdt_transform.slangh         the per-pixel transform
  psdt_analysis.comp.slang      HDR -> adaptation field mip 0, with classification
  psdt_downsample.comp.slang    the pyramid, per-channel reductions
  psdt_state.comp.slang         histogram + pyramid -> AdaptationState + parameters

src/dxvk/rtx_render/rtx_tone_mapping.{h,cpp}    resources, dispatch, options, UI
src/dxvk/shaders/rtx/pass/tonemap/              operator 8 wired into the apply pass
tools/psdt/                                     CPU reference and test suite
```

PSDT lives inside `DxvkToneMapping` rather than in a pass of its own for one
substantive reason: `psdt_state` has to read the luminance histogram *between*
`dispatchHistogram` and `dispatchToneCurve`, because the tone curve pass zeroes
the histogram on its way out. Everything else follows — the operator selection
already forces the global path, the colour buffer and exposure texture are
already in hand, and no new pass registration or dispatch ordering is introduced.
