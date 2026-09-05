# PSDT — Perceptual Scene Display Transform

`rtx.tonemap.operator = 8`

The only entry in the tonemap operator list that is not a tone curve. The
others are pure functions of one pixel. PSDT is a display transform: it
converts physically reconstructed scene radiance into an image for a viewer
adapted to *this* scene, looking at *this* display.

The existing operators stay exactly as they are, and two more were added
(Reinhard, AgX) for the same reason: they are the controls this is measured
against, and that is worth more than the few kilobytes they cost.

---

## Why not just tune GT7

Because the two inputs that most determine what a pixel should look like are
not in the pixel: what the viewer is adapted to, and how much room the display
has left. A per-pixel curve cannot see either, so every per-pixel curve has to
guess, and every guess is a compromise between the dark corridor and the sunlit
window.

RTX Remix makes this worse in a specific way, and better in a specific way.

Worse: the reconstructed scene carries far more range than the original game
ever did, has ray-traced emissives that reach values no display can represent,
and is temporally reconstructed. A curve tuned for a car in daylight is being
asked a question it was not designed for.

Better: Remix *knows things about the scene that a film pipeline does not*. It
knows which pixels emitted light rather than reflected it. It knows how far
away each one is, and therefore which are sky. It has a demodulated albedo, so
reflectance can be divided out of radiance to leave the colour of the light.
None of that is recoverable from the framebuffer, and all of it is exactly what
a perceptual display transform needs.

The other reason is stacking. Local exposure, a global curve, chroma
compression and a gamut clamp each independently squeeze the same signal. Each
looks reasonable alone. Together the colour gets squeezed four times, and the
image stops saying "that is a physically bright, saturated red light" and starts
saying "that is approximately red-ish and bright". That is not a curve problem.
It is an architecture problem, and it is the one this addresses.

---

## The contract

Four jobs, four owners, and nothing owns two of them:

```
Auto Exposure    what brightness am I adapted to?          (upstream)
PSDT curve       how do I reproduce the adapted scene?     psdt_curve.slangh
Colour volume    how do scene colours fit the display?     psdt_transform.slangh
Glare            how does a very bright source appear?     psdt_transform.slangh
```

Keeping those apart is the whole design. Every stage can be turned off
independently, every stage has a debug view showing what it actually read, and
when the image is wrong it is possible to say which stage is wrong.

The rule that v0.2 broke is the other half of the same sentence: **nothing owns
two of them, and no job has two owners.** Local dynamic range compression had
two, and that is what v0.3 is about.

---

## v0.1 → v0.2: heuristics became measurements

v0.1 had the right architecture with the wrong measurements underneath it. Four
quantities the design described as physical were guesses about brightness:

| the design said | v0.1 actually did | consequence |
|---|---|---|
| source-aware adaptation | `smoothstep` on log luminance | a sunlit white wall metered as a lamp |
| spatial white point | offset the chroma coordinates by the local mean colour | three coloured walls under a neutral light read as a coloured illuminant, and a neutral could leave the display volume |
| display-aware gamut mapping | Rec.709 luminance weights in every gamut | P3 and Rec.2020 cast every gamut ray from a grey point that was not grey |
| perceptual glare | tinted from the scene's low-frequency chroma | a white light near a red wall glared red |

All four are now measured. The renderer signals that make that possible are
`PrimarySurfaceFlags.isEmissive`, `PrimaryLinearViewZ` and `PrimaryAlbedo`, all
of which already exist and none of which cost a pass to produce.

Three further things v0.2 fixed, each found by a test rather than by looking:

* **The cut detector compared a measurement against the filtered state.**
  Downward adaptation is deliberately slow, so during any sustained dimming the
  state falls behind on purpose — and comparing the two turned that intended lag
  into a false cut. A 6-stop fade over half a second tripped it, and the
  exposure snapped part way through a fade that was working correctly. Cuts are
  now detected between successive *measurements*, which is what a cut is.
* **The luminance concession was not monotonic.** It interpolated between two
  rising functions with a rising weight, and the weight could grow faster than
  the film path rose, so the conceded target fell while both its endpoints
  climbed. 18 luminance inversions at `highlightRolloff` 0.90. The gap gates
  itself, so the weight is now constant and the result is a convex combination
  of two monotone functions.
* **The CPU reference diverged from the shader** in two places — the shoulder
  clamp and the black-level epsilon — so the suite was reporting on a curve that
  did not run. `psdt_check.py` now fails the build if that happens again.

---

## v0.2 → v0.3: one pyramid, one owner

v0.2 had two systems doing one job and a scalar apologising for it.

Auto Exposure Plus builds a multi-scale pyramid of local log luminance over the
HDR buffer, accumulates it across frames with motion reprojection, collapses it
with an edge-stopping selector, reconstructs it at full resolution with a guided
filter, and multiplies the buffer by the result. PSDT builds a multi-scale
pyramid of local log luminance over the same buffer, accumulates it across
frames with motion reprojection, pools it, and moves the curve's anchor by the
result. Two pyramids, two temporal filters, two full-resolution passes, both
compressing local dynamic range, running in series.

v0.2's answer was to scale PSDT's local strength and contrast budget by
`1 − |plus.intensity|` and call it coordination. It is not a coordination. It is
a way of maintaining two implementations of one idea and then dividing the work
between them so that neither does it properly.

**v0.3 picks one.** PSDT's field already carries strictly more than Plus's did —
body weight, depth, per-channel source energy, illuminant — and it is already
the glare kernel, so it is the pyramid that survives. While PSDT owns local
adaptation, `DxvkAutoExposurePlus::isEnabled()` returns false: the pass
deactivates, `RtxPass` releases its four resources, and its sixteen dispatches
stop being recorded. PSDT then spends the whole budget rather than 65% of it.

Plus is not deleted and not weakened. It is the local operator for every other
tonemapper in the list — the native curve, Hable, Psycho17, GT7, Reinhard, AgX
— and for all of them it is exactly what it was.

### What came across with it

Suppressing a pass is only honest if what it was contributing comes too, and
Plus was contributing two things PSDT's pooling did not have.

**The edge-stopping collapse.** Plus chose, per pixel, which scale owned the
exposure, by asking whether the scales agreed with each other. PSDT pooled with
frame-resolved weights modulated only by depth agreement. Depth catches a window
in a wall. It does not catch a shadow across the middle of one — same surface,
same distance, two lighting conditions — and that is exactly where local
adaptation is supposed to earn its keep. `scaleCoherence` is that term, in the
same shape as the depth term: a one-stop dead zone, then a squared falloff.

```
   coarse neighbourhood at    +0.5   +1     +2     +3     +4     +6   stops
   pooled, v0.2              0.394  0.788  1.575  2.363  3.150  4.725
   pooled, v0.3              0.394  0.788  1.059  0.092  0.000  0.000
   rejected                   0.0%   0.0%  32.8%  96.1% 100.0% 100.0%
```

**The edge-aware downsample.** Plus reduced its pyramid with a separable
Kuwahara — two one-sided means, lower variance wins — so a level averaged within
features rather than across them. A 2×2 reduction has no "two sides", so what
crossed over is the idea rather than the filter: one Welsch reweight about the
weighted mean, which makes a coarse texel report the population most of its
children are in.

```
   children (stops)          plain   robust   wanted
   [0, 0, 0, +3]             0.750    0.011      0     a shadow across a wall
   [0, 0, 0, +6]             1.500    0.000      0     a window in a wall
   [0, 0, 0, +9]             2.250    0.000      0     a lamp against a wall
   [0, +0.2, -0.3, +0.4]     0.075    0.079      —     ordinary shading, control
   [0, +1, +2, +3]           1.500    1.500      —     a gradient, control
   [0, 0, +6, +6]            3.000    3.000      —     an even split
```

The last row is a limitation, stated rather than hidden: a one-step reweight
cannot break a tie, because both halves sit the same distance from the mean and
get the same weight. What saves it is that the finer level it was reduced from
still holds both halves, is still pooled, and is where a feature that small
belongs.

### And one signal neither of them had

A path-traced frame can put a hundredfold luminance spike in one block for one
frame — a firefly the denoiser did not catch, a speculative fireball, a specular
hit on something that moved. From the framebuffer that is indistinguishable from
a muzzle flash. From the reprojected history it is not.

So how far a block's source energy disagrees with where it was now sizes the
history window it is allowed to escape, and scales the weight it escapes at.
Appearing out of nothing is expensive; persisting is not.

```
                          one frame at 400x        a source that stays
                          halo    of v0.2          90%       halo at 0.1 s
   v0.2 (no test)         6.99    100%             0.367 s   7.84
   v0.3                   3.91     56%             0.500 s   6.41
```

The discrimination is on repetition, not on steadiness, which is the distinction
that matters for a game: a one-frame 4× spike is rejected to about a sixth of its
halo, while a source flickering 4× *every* frame keeps 96% of what a steady one
would. A fire stays lit.

Only the appearing direction is slowed. Release is one frame, unchanged, because
the clamp's upper bound is what handles a falling measurement and a light going
out is not something to be sceptical about.

This is the only part of the transform that knows its input is path traced. It
is one term on one channel, not a variance-aware pipeline, and it is described
that way deliberately.

---

## Pipeline

```
                RTX RENDERED HDR (linear Rec.709, post-exposure)
                                 |
                                 |         gbuffer: emissive bit,
                                 |         linear view Z, albedo
                                 v                    |
                          psdt_analysis  <------------+
              classify every pixel BODY / SOURCE / GLARE / SKY
              estimate the local illuminant as radiance / albedo
              accumulate a BODY-ONLY luminance histogram
              reduce each 16x16 block, reproject, accumulate
              size the source channel's history window by how far it
              disagrees with where it was  (v0.3)
                                 |
              +------------------+------------------+
              |                  |                  |
         body field         source field      illuminant field
       logY, weight,        RGB energy,        illuminant RGB,
       peak, depth          sky weight         confidence
              |                  |                  |
              +------------------+------------------+
                                 v
                      psdt_downsample x (levels-1)
              the mip chain IS the adaptation pyramid
              and IS the glare kernel
              body reduction is robust, not a plain mean  (v0.3)
                                 |
        +------------------------+------------------------+
        |                                                 |
   body histogram                              tonemap histogram
   (scene white/black/median, cuts)            (saliency; already existed)
        |                                                 |
        +------------------------+------------------------+
                                 v
                            psdt_state  (one workgroup)
           source-aware anchor, body-only scene statistics, scene
           intent, camera-aware transition classification, asymmetric
           inertia, and every runtime option resolved into the state
                                 |
                                 v
                     tonemapping_apply  ->  psdtApply()
                                 |
   +---------+---------+---------+---------+---------+---------+
   |         |         |         |         |         |         |
   v         v         v         v         v         v         v
multi-    contrast  exposure-  local    chromatic  colour    perceptual
scale     budget    aware      contrast adaptation volume    glare
adapt.              curve      restore  to the     pressure  (two lobes,
(depth +            T(x;E)              measured   + hue      source-
 light                                  illuminant trajectory coloured)
 edges)
   |         |         |         |         |         |         |
   +---------+---------+---------+---------+---------+---------+
                                 v
                        display gamut fit
                                 v
                    colour grading / sRGB / dither
```

**Dispatch count.** One analysis, `levels - 1` downsamples, one state. At the
default six levels that is **seven compute dispatches**, not three — v0.1's
README said three and was counting the loop as one. The downsamples are a few
thousand threads each; the analysis pass is the only one that touches every
pixel.

Against v0.2 the net is **sixteen fewer**, because Auto Exposure Plus no longer
runs alongside it: one init, twelve separable downsamples, a fuse, a temporal
pass and a full-resolution read-modify-write of the HDR buffer. Two of the three
passes that touched every pixel are now one. None of that is a profile.

---

## The parts, and what each is for

### Scene classification

Conventional metering asks "how bright is this frame". That is the wrong
question the moment the scene contains a light, which every Titanfall interior
does:

```
   dark corridor  +  small sunlit window

     histogram sees:  dark mass + a huge bright spike
     a viewer sees:   a dark corridor, and a window
```

Four roles. **BODY** — walls, characters, weapons, ground — determines exposure.
**SOURCE** is *excluded* from the estimate rather than clamped, because a
clamped sun still drags a mean and an excluded one does not. **GLARE** is too
bright to be a value at all; its energy is kept per channel and spent later as
spatial extent. **SKY** is neither surface nor lamp: it gets a low fixed
adaptation weight, so a frame that is 60% sky still lands most of its weight on
the ground — which is what a viewer walking outdoors actually does — while a
window in a wall gets almost none.

The evidence is five things, not one:

```
emissive bit          the renderer's own answer to "does this emit"
linear view Z         sky is 500001, backdrop 200000, a room is single digits
spatial compactness   a whole bright block is a lit surface; a few bright
                      pixels in a dark block is a lamp or a glint
luminance excess      still there, but as one input rather than the classifier
temporal persistence  the field is reprojected and accumulated, so a
                      classification must survive several frames to fully own
                      the adaptation
```

Measured, at 6 stops over middle grey — the same luminance for every row, so
the only thing separating them is what the renderer said:

```
  neon sign (emissive, compact)     body 0.000   source 1.000
  emissive screen (large area)      body 0.000   source 1.000
  sunlit white wall (large)         body 0.750   source 0.250
  specular glint (compact)          body 0.000   source 1.000
  open sky                          body 0.150   sky 1.0
```

With the gbuffer unavailable the classifier falls back to luminance and
compactness, which still separates a glint from a wall and collapses everything
else — six cases into two. That is v0.1's behaviour, and it is the honest
picture of how much the renderer signals were doing.

### Multi-scale adaptation under a contrast budget, across depth and light edges

Three scales — 16, 64 and 256 pixels — pooled with scene-dependent weights, from
a pyramid whose reduction is robust rather than plain (see v0.3 above: a coarse
texel reports the population most of its children are in, so the levels being
pooled describe something that is actually there).

Then the budget, which is the most important constraint in the system:

```
   shadows      2.00 stops of adaptation allowed
   midtones     0.55
   highlights   0.25
   emitters     0.05
```

Local adaptation with no budget converges on "every region is middle grey".
That improves visibility while destroying the scene's brightness hierarchy —
and that hierarchy carries as much information as the geometry does.

New in v0.2: the pooling refuses to average across a depth discontinuity. A box
pyramid does not know that the bright thing 200 pixels away is the sky through a
window and the dark thing beside it is the wall around it; averaged together
they produce an adaptation value describing neither, and the visible result is a
halo. The field carries each block's body-weighted mean depth as `log2(1 + z)`,
so the disagreement is in octaves of distance and one constant works from a
corridor to a skybox — with a one-octave dead zone (a factor of two in distance
is the same surface) and a squared falloff past it.

Measured, with the luminance term off so the depth term is what is being
measured: a wall with sky six stops brighter behind it shifts by **0.0000
stops**; the same six-stop neighbour at the same depth shifts it by 0.181.

New in v0.3: the pooling also refuses to average across a *lighting*
discontinuity, which is the case depth cannot see. A wall with a shadow across
the middle of it is one surface at one distance under two lighting conditions,
and a coarse level spanning both meters the lit half too dark and the shadowed
half too bright. Same shape as the depth term — a one-stop dead zone, then a
squared falloff — and measured the same way, with the depth term off: a six-stop
lighting edge at the same depth shifts the wall by **0.0000 stops**, while
ordinary half-stop shading still pools normally at 0.256.

Two terms that overlap have to be measured one at a time, or the control case in
each collapses and both tests pass without demonstrating anything. That is why
invariants 8 and 9 each disable the other's term; it is also the bug the v0.2
invariant list would have hidden.

### An exposure-aware curve

Log2 scene in, log2 display out. An exponential toe, a linear segment, an
exponential shoulder, C1 at both joins, strictly monotonic everywhere, and
asymptotic rather than clamped — the curve itself never clips. Clipping is a
display event, handled once at the end, and what does not fit becomes glare.

Every parameter is resolved per frame from the adaptation state. This is
`T(x; E)`, not `T(x)`.

### Local contrast, restored before the curve, rolled off with magnitude

The obvious formulation — map the base, then add the detail back — is a
first-order Taylor expansion of the curve, and near the top of the shoulder its
derivative outruns the curve's own slope and **inverts the transfer function**.
Expanding the detail in the scene domain and letting the curve map the result
fixes that by construction: at fixed base the output is a monotone function of a
monotone function.

v0.2 adds the magnitude rolloff the v0.1 option help promised and the v0.1 code
did not implement. Restoration halves at `detailKnee` stops from the
neighbourhood's own level:

```
   0.25 stops   surface texture         boost 0.35
   1.00 stops   a shadow edge           boost 0.28
   3.00 stops   a bright object         boost 0.09
   6.00 stops   a lamp against a wall   boost 0.01
```

A lamp six stops from the wall behind it is not detail, it is a different
object, and restoring the contrast between them is a halo.

### Colour volume, measured rather than guessed

The question is "how much of this colour can the display reproduce at this
luminance, and what is the least perceptually damaging way to get there".

The measurement is exact and cheap: find the largest `t` for which
`grey + t·(rgb − grey)` still lies inside the display cube. `grey` is located
with **the target gamut's own luminance weights** — v0.1 used Rec.709's in all
three, which is exact in Rec.709 and off by up to 0.036 of full scale in
Rec.2020, on every out-of-gamut colour in the frame. The chroma normaliser is
per-gamut too; using the Rec.709 value in Rec.2020 mis-scaled the chroma term by
27%.

Compression begins before the boundary, not at it, and how early is decided by
four terms — luminance, chroma, gamut distance and **context**. The context term
is new: a colour that is a bright outlier against a much darker neighbourhood is
somewhere the eye is not judging colour fidelity and where a hard clip is very
visible, so it earns an earlier and gentler approach; a colour sitting in a large
field of its own brightness keeps its colour longer, because a hue shift there is
obvious and there is no clipping artifact to trade against.

Compression is a soft knee that asymptotes to *the boundary*, not to zero — the
difference between a neon sign that stays a neon sign and one that becomes a
white blob.

### Luminance concession

A saturated red cannot be as bright as a white: the most saturated Rec.709 red
has a luminance of 0.21. When the tone stage asks for a brightness the hue
cannot support, something has to give — bleach the colour to keep the
brightness, or let it be darker to keep the colour. Film lets it be darker, and
that is most of why film highlights read as coloured light rather than as blown
paper.

The gap between the tone target and the per-channel path *is* the gate: it is
zero for a neutral by construction, and it only opens where the curve is
concave, so the midtones are untouched and the shoulder is where it acts. The
weight on it is constant — see the monotonicity note above.

### Hue trajectory

Two paths. One preserves chromaticity exactly; the other is the per-channel
curve, whose dominant channel saturates first and which therefore bends red →
orange → yellow-white on its own. The hue rotates from the first towards the
second in proportion to how far outside the volume the colour landed, along the
shortest arc so the red/magenta wrap has no jump in it.

Measured largest hue step across a 1/42-stop sweep: **0.073°**, against 0.845°
for GT7 and 0.657° for AgX.

### Chromatic adaptation to a measured illuminant

The neutral that chroma compresses towards is the local illuminant rather than
D65, so a tungsten interior stays warm as it desaturates instead of drifting
towards daylight.

Two things about this are different from v0.1, and both matter.

**It is an adaptation, not an offset.** v0.1 subtracted the local mean chroma,
compressed, and added it back. That is a translation of a colour space, not an
adaptation of an observer, and it has a specific failure: a surface that is
neutral under the local light does not land on the perceptual neutral axis, so
compressing its chroma towards zero does not take it towards the local white,
and the guarantee that a neutral always fits inside the display volume stops
holding. A von Kries transform in CAT02 cone space does mean it — measured, a
local-neutral surface lands on the adapted neutral axis to 1.1e-9, and stays in
gamut under every illuminant tested.

**The illuminant is measured, not averaged.** `radiance / albedo` is the colour
of the light; the mean radiance is the colour of the paint. In a Titanfall
interior of orange panels, rust and ochre trim under a neutral light:

```
   true illuminant       (1.000, 1.000, 1.000)
   grey world (v0.1)     (1.459, 0.915, 0.494)   max error 0.506
   radiance / albedo     (1.000, 1.000, 1.000)   max error 0.000
```

The estimate is weighted by how much albedo a surface has in every channel and
by its luminance, so near-black and fully saturated surfaces contribute nothing
rather than dividing noise by noise, and the frame-level confidence scales the
whole effect — a scene with no evidence adapts to D65, which is the correct
answer when the illuminant is unknown rather than a fallback.

This is a good estimator of the diffuse illuminant, not a measurement of it:
by tonemap time `PrimaryAlbedo` has been scaled by the composite and
ray-reconstruction passes, and the colour buffer contains specular and indirect
light the model does not account for.

### Glare, from the same pyramid

Not bloom. Bloom is a lens artifact applied to the image; glare is how an
intensity that cannot be shown as a value gets shown as an area instead. It is
separate from `rtx.bloom.*` and both can run.

Two things changed from v0.1, and both were things the comment claimed and the
code did not do.

**The threshold is applied per level**, not to the summed energy. That is what
actually produces a radius growing with the log of source luminance: a dim
source's coarse levels fall below the visibility threshold and contribute
nothing, while a bright one stays above threshold several levels further out.
Summing first and thresholding once gives every source the same profile scaled
by brightness — a bloom, not a glare. Measured:

```
   source Y      1      4     16     64    256   1024   4096  16384
   halo radius   0      0     16     32     64    128    256    512  px
```

**The kernel has two lobes.** A single geometric series is a single exponential;
the eye's glare point-spread function is closer to a narrow core on a much
wider, fainter veil. The near-field lobe is that core, and it is also the "near
field" a highlight representation needs between its clipped centre and its halo.

And the halo carries **the source's own colour**, accumulated per channel in the
source field, rather than the scene's average chroma. It is pulled 40% towards
white because ocular scatter is broader-band than the light causing it, which
costs a few degrees of hue on a fully saturated source and is the trade being
made deliberately.

The threshold is anchored to the adaptation state, which is why a car headlight
is blinding at night and invisible at noon — measured at 0.260, 0.201 and 0.000
for anchors at −4, 0 and +4 stops.

### Temporal adaptation, and transitions

Measured state carries inertia; resolved parameters do not. So the transform is
deterministic given the state, while the state has a time constant.

Three things look identical to a histogram and want three different responses,
and v0.2 separates them because the camera knows which is which:

```
   still      normal time constants
   sweeping   3x faster - the scene in front of the player genuinely changed,
              and easing through seconds of it reads as the image washing out
   cut        re-anchor outright (RtCamera::isCameraCut is a position jump
              larger than the scene's own unique-object distance, so it is a
              teleport or a respawn by construction)
```

Measured step response, at 60 fps:

```
   +2 stops                       10/50/90%   0.117 / 0.400 / 1.000 s
   -2 stops                       10/50/90%   0.200 / 0.750 / 2.100 s
   +2 stops during a fast turn    10/50/90%   0.067 / 0.233 / 0.550 s
   doorway, +6 stops over 0.5 s   10/50/90%   0.317 / 0.683 / 1.283 s
   doorway, -6 stops over 0.5 s   10/50/90%   0.433 / 1.033 / 2.383 s
   instant +6 stops (a cut)       10/50/90%   0.000 / 0.000 / 0.017 s
```

No overshoot anywhere, no false cut on either doorway, and a genuine cut lands
in two frames. Under a measurement carrying ±0.5 stops of frame-to-frame noise
the anchor moves by at most 0.0055 stops — a 176× attenuation, and the reason
two filters in series exist at all.

### Ownership of local dynamic range compression

`rtx.tonemap.psdt.localAdaptationOwner`, and the default is PSDT. See the v0.3
section above for why this replaced a scalar.

```
   owner                 PSDT local  budget   Plus dispatches   Plus memory
   PSDT (default)             0.650    2.00                 0      released
   Auto Exposure Plus         0.000    0.00                16       ~14 MiB
   Both (v0.2)                0.423    1.30                16       ~14 MiB
```

The dispatch and memory figures are for 1080p and are counted from
`DxvkAutoExposurePlus::dispatch` and its four resources — read off the code, not
off a GPU, like everything else here.

The other two settings are controls rather than fallbacks. `AutoExposurePlus`
hands the spatial half of the job back to Plus and leaves PSDT's curve anchored
globally, which isolates what PSDT's own local adaptation contributes. `Both`
reproduces v0.2 exactly, so the change is measurable rather than asserted.

---

## Diagnostics: the engine side

`[TonemapProbe]`, enabled with `rtx.logSurfaceCoverage`, reads the tonemapper's
input and output at a sparse grid and logs what the operator did. It used to log
RGB and luma, which answers "did the curve run" and nothing else — the questions
an operator comparison turns on are how much chroma survived, whether the hue
moved, and how much of the frame arrived at a limit, and none of those are
visible in a luma number.

It now carries luminance, ICtCp chroma and hue on both sides, at the same
100-nit reference `tools/psdt/` uses, so a number from the game is comparable
with a number from the harness rather than merely similar to one. One aggregate
line per frame, fixed field order:

```
[TonemapProbe] f=… op=8 plus=1 grade=0 n=24 Yin=[…] Yout=[…]
               clipHi=3/24 clipLo=0/24 negIn=0/24
               dHueMean=0.42 dHueMax=1.83 chromaOutIn=0.71 chromaSamples=19
```

`op=`, `plus=` and `grade=` are on every line so a log can be attributed to a
configuration rather than to a memory of what was set. That is what makes the
operator × Plus matrix tractable: run the same scene under each configuration
with colour grading and bloom off, then diff these lines. `rtx.tf2HeavyProbes`
adds per-sample `[TonemapProbe.px]` lines.

Hue is only accumulated where both sides still carry visible chroma. Averaging
`atan2` noise from samples that correctly arrived at white would make a
well-behaved transform look like one with a hue problem.

The probe reads the final image, so it sees the operator's output after the
operator's own internal clamp. Interior pre-clamp values are not observable from
a readback at all — that needs shader instrumentation, which is the next section.

## Diagnostics: the shader side

`rtx.tonemap.psdt.debugView` replaces the image with one of the transform's own
intermediate values — the value it actually read on that pixel on the way past,
not a re-derivation. A debug view that recomputes the quantity agrees with the
shader right up until the moment something is wrong, which is the only moment it
was needed.

```
 1 adaptation field      pooled adaptation, stops about the anchor
 2 after the budget      what the contrast budget left of it
 3 role classification   green surface, red source, blue sky   <- start here
 4 source energy         the glare kernel's input, per channel
 5 local illuminant      should read grey in a coloured room under a white light
 6 depth and trust       how much of the coarse pooling survived both agreement
                       tests; dark green is pooling on the finest scale  <- second
 7 gamut demand          green in gamut, red out of it, hard edge at the boundary
 8 chroma given up       should be zero over most of the frame
 9 glare only            the halo without the image under it
10 pre-clamp excess      red over peak, blue under zero          <- then here
11 curve slope           where the curve adds contrast and where it takes it away
```

---

## A correction this makes, does not propagate, and now measures

The renderer's working space is linear **Rec.709 / sRGB primaries**, D65. That
is observable, not assumed: the histogram, the auto exposure and the colour
grading all weight luminance with `calcBt709Luminance`, and the apply shader
finishes with the sRGB OETF and no primary conversion.

`gt7.slangh` and `psycho17.slangh` both treat the framebuffer as Rec.2020. PSDT
converts explicitly. **The other two are deliberately left alone**, because they
are the controls, and a control you have quietly edited is not a control.

But "left alone" is only defensible if the cost is known, so `gt7space` in the
suite measures it. GT7's forward and inverse both use the Rec.2020 matrices, so
they are mutually consistent and an untouched colour round-trips exactly — which
is why this is invisible on a grey ramp and why it survived this long. The error
enters only where the ICtCp values are *modified*, which is to say exactly where
the operator does its work:

```
   swatch     Y shipped    Y reference       dY    C shipped   C reference    dHue
   grey          0.8351         0.8351    +0.0%       0.0000        0.0000   n/a
   red           0.3224         0.4090   +26.8%       0.1576        0.1302  +15.4d
   blue          0.4107         0.5436   +32.4%       0.1094        0.0768  +10.9d
   magenta       0.4228         0.4830   +14.2%       0.1659        0.1374   -0.9d
   skin          0.7941         0.8028    +1.1%       0.0343        0.0340   +1.8d

   worst over 11 swatches x 3 luminances:  32.4% luminance, 15.4 deg hue
```

`reference` is GT7 used the way its own reference intends — Rec.709 in,
Rec.2020 for the operator, Rec.709 out.

So: tens of percent of luminance on precisely the saturated colours a
colour-volume comparison is about. **As wired today the GT7 branch is the GT7
algorithm applied to relabelled numbers, not the GT7 reference applied to this
renderer's colours**, and any conclusion drawn from PSDT-versus-GT7 inherits
that. Correcting it changes how the operator looks for anyone using it, which is
a decision rather than a bug fix — hence a measurement here and no edit there.

`psycho17.slangh` has the same assumption and has not been measured; it is the
current default operator, so that is the more consequential of the two.

---

## Measured results

`python3 tools/psdt/psdt_suite.py`. Defaults, SDR sRGB, 100 nit reference. Full
output in `tools/psdt/RESULTS.txt`.

```
  operator  chroma kept  boundary use  max hue step  midtone  hilite bnd use  clipped  Y inv
  PSDT            0.861         0.790        0.143d    1.150           0.819    34.4%      0
  GT7             1.002         0.711        0.845d    1.307           0.237    59.3%      0
  AgX             0.753         0.722        0.657d    0.810           0.913    50.7%      6
  Reinhard        0.862         0.807        0.365d    0.852           0.600    55.0%      0
  Hable           0.880         0.817        0.576d    0.902           0.400    60.3%      0
```

`hilite bnd use` is the one worth reading twice: of the colour the display could
still show at the luminance the operator chose, one to four stops past the
shoulder, PSDT keeps 0.82 and GT7 keeps 0.24. That is the "a neon sign stays a
neon sign" claim, in a number.

The section-33 weighted score, with the weights written down because they are a
choice and cannot be anything else:

```
  S = 0.25*SL + 0.25*SC + 0.20*SH + 0.15*SD + 0.15*ST

  operator      SL      SC      SH      SD      ST       S
  PSDT       0.656   0.790   0.867   0.700   0.947   0.782
  GT7        0.407   0.711   0.430   1.000   1.000   0.665
  Reinhard   0.450   0.807   0.694   0.105   1.000   0.619
  Hable      0.397   0.817   0.562   0.204   1.000   0.597
  AgX        0.000   0.722   0.518   0.020   1.000   0.437
```

AgX scores zero on luminance fidelity because it inverted brightness ordering
six times on the grid, which is disqualifying rather than merely costly. The
stateless operators all score 1 on temporal stability by construction — they
have no state to be unstable, which is also why they cannot adapt to anything.

All nine perceptual invariants hold:

```
  1. brightness ordering preserved         0 inversions
  2. small bright object vs dark surround  4.05 display stops apart
  3. mid-luminance red hue held            -0.05 deg
  4. chroma trajectory continuous          max step 0.00055
  5. colour stable under a noisy input     0.48 of an 8-bit code value
  6. adaptation budget respected           0.150 stops for a 6-stop neighbour
  7. emissive as colourful as the display  worst boundary use 0.822
     allows
  8. a wall is not re-exposed by the sky   0.0000 stops across a depth edge,
     behind it                             0.181 without one
  9. a wall is not re-exposed by the       0.0000 stops across a lighting edge,
     sunlit half of the same wall          0.256 across ordinary shading
```

The nine scores were unchanged by v0.3, and that is the right answer rather than
a disappointing one: the comparison grid runs on an isolated pixel where every
scale reports the frame anchor, so nothing the pooling does can reach it. What
v0.3 changed is measured in `coherence` and `confidence`, which exist because
the existing sections could not see it.

The curve is monotonic over 40 stops with a worst step of exactly 0, C1 at both
joins, and its analytic derivative matches a central difference to 1.3e-9. And
`psdt_sweep.py` now reports **zero luminance inversions at every setting of
every parameter it sweeps**, not just at the defaults.

Six defaults are set by sweep rather than by taste — `highlightRolloff` 0.45,
`luminanceConcession` 0.8, `chromaPreservation` 2.5, and v0.3's
`scaleCoherence` 1.20, `pyramidCoherence` 1.00, `sourceConfidence` 1.50.

The three new ones sweep in `psdt_sweep.py`'s pooling section rather than its
main one, because the main sweep runs `psdt_apply` on an isolated pixel where
the pooling is never exercised. Sweeping them through it would print identical
rows and read as "this parameter does nothing", which is worse than not
sweeping it at all.

`pyramidCoherence` moved from 1.50 to 1.00 on that sweep: at 1.50 a coarse texel
made of three dark children and one three stops brighter still reported 0.13
stops above the dark ones, and at 1.00 it reports 0.011, while the ordinary
shading control moves by 0.004 stops either way. Below about 0.75 the reweight
becomes narrower than the field's plausible residual noise, which is a stability
risk this harness cannot measure — so 1.00 is the value with margin, not the
value with the lowest error.

---

## What this does not do

* **It has not been run on a GPU.** Everything above is measured on the CPU
  reference in `tools/psdt/`, which `psdt_check.py` verifies is still in step
  with the shaders. The shaders themselves have not been compiled or executed —
  this branch cross-builds for Windows and the toolchain is not on this machine.
  That is the single largest gap and no amount of CPU testing closes it.
* **No frame corpus.** A swatch ladder is not a game frame. The next real
  measurement is Titanfall frames through the debug views, starting with the
  role classification.
* **Psycho17 is not in the comparison.** It is 754 lines of vendored operator
  and a hand port could not be verified against the shader that runs; an
  unverified control is worse than a missing one. It is also the current default
  operator, so this is a real gap in the comparison and not a footnote.
* **No HDR output.** The transform is display-aware and will grade for a peak
  above diffuse white, but this pipeline ends in an 8-bit sRGB encode. Leave
  `displayPeakNits` equal to `displayRefWhiteNits`; that path is exact.
* **The illuminant estimate is diffuse-only.** See the caveat above.
* **Performance is unmeasured.** Seven dispatches, all small, plus a per-pixel
  transform that does two to four perceptual conversions and up to two chromatic
  adaptations. The adaptation is skipped entirely when `spatialWhite` is 0. None
  of that is a substitute for a profile — including the sixteen dispatches v0.3
  stopped submitting, which are counted from the code and not from a capture.
* **The source-confidence term is one term on one channel.** It is not a
  variance-aware pipeline and the design note that asked for one wanted more:
  path-tracing variance feeding the exposure, the white point and the colour
  mapping as well. This feeds the glare model, because that is where believing
  one frame of noise costs a visible artifact, and because the body channel
  already has a clamp, a budget and a slow anchor in front of it. Extending it
  is a v0.4 item, not a claim being made here.
* **The robust pyramid reduction cannot break a tie.** Four children split two
  and two produce a value neither pair is at. See the v0.3 section.

### Not a human vision model

The README for v0.1 said PSDT produces "the image a viewer adapted to this scene
would find most plausible". That was an overclaim and it is withdrawn.

This is a perceptually *motivated* model whose inputs are now measured rather
than guessed. A visual model would have photopic/mesopic/scotopic behaviour, rod
and cone interaction, the Purkinje shift, surround induction, visual angle and
display viewing geometry. PSDT has none of those. What it has is a defensible
estimate of what the viewer is adapted to, a defensible estimate of what colour
the light is, and an honest measurement of what the display can still show — and
those three are enough to be worth measuring against the alternatives.

---

## Files

```
src/dxvk/shaders/rtx/pass/psdt/
  psdt.h                        bindings, state layout, push constants
  psdt_perceptual_space.slangh  ICtCp / Jzazbz behind one interface, per-gamut
                                geometry, CAT02 chromatic adaptation
  psdt_curve.slangh             the luminance curve and its analytic derivative
  psdt_transform.slangh         the per-pixel transform and the debug views
  psdt_analysis.comp.slang      HDR + gbuffer -> the three fields, mip 0
  psdt_downsample.comp.slang    the pyramid, three reduction rules
  psdt_state.comp.slang         two histograms + pyramid -> AdaptationState

src/dxvk/shaders/rtx/pass/tonemap/
  reinhard_agx.slangh           two more controls, additive
  tonemap_operators.slangh      operator dispatch (existing operators untouched)

src/dxvk/rtx_render/rtx_tone_mapping.{h,cpp}    resources, dispatch, options, UI
tools/psdt/                                     reference, suite, checker, sweep
```

PSDT lives inside `DxvkToneMapping` rather than in a pass of its own for one
substantive reason: `psdt_state` has to read the luminance histogram *between*
`dispatchHistogram` and `dispatchToneCurve`, because the tone curve pass zeroes
the histogram on its way out.

---

## What v0.4 should be

Everything on v0.3's list that needed a GPU still needs one, and nothing on this
one is a substitute for that.

In order, and the first one is not optional:

1. **Compile and run it.** Nothing below matters until the shaders have been
   through slangc and a frame has gone through them. `psdt_check.py` catches the
   header/C++/shader mismatches that would stop a build — it now also checks the
   `flags` bit packing, which v0.3 rearranged and which nothing else would have
   noticed going wrong. It cannot catch a Slang syntax error, because there is no
   Slang compiler on this machine.
2. **Look at the role classification in-game** before looking at the image. If
   the roles are wrong, every measurement downstream is answering the right
   question about the wrong pixels.
3. **Look at the depth view second.** `debugView` 6 now shows the combined
   agreement of both pooling terms in green, so a frame that is dark there is a
   frame pooling on the finest scale alone. That is the failure mode v0.3
   introduced, and it is one screenshot to rule out.
4. **Run the operator matrix.** The instrumentation is in place, so this is
   mechanical: run the same scene under each operator with
   `rtx.tonemap.colorGradingEnabled=0`, bloom off, and `rtx.logSurfaceCoverage=1`,
   then diff the `[TonemapProbe]` lines.

   ```
   operator 6 (Psycho17)   + Plus off / Plus on
   operator 7 (GT7)        + Plus off / Plus on
   operator 9 (Reinhard)   + Plus off / Plus on
   operator 8 (PSDT)       localAdaptationOwner = PSDT / Plus / Both
   ```

   The last row is the one that establishes whether v0.3 was right. The synthetic
   suite cannot answer it, because it cannot exercise the pooling at all.
5. **Decide the GT7 input-space question**, with the `gt7space` numbers in hand.
   Either convert around the branch and accept that GT7 changes appearance, or
   keep it and state in every comparison that the control is off by up to a
   third of a stop on saturated colours. What is not defensible is leaving it
   undecided, which is where it has been.
6. **Measure Psycho17's input-space assumption too.** It has the same Rec.2020
   comment and it is the current default operator, so it matters more than GT7's.
7. **Profile it**, per stage, with the option toggles that already exist — and
   confirm the sixteen dispatches v0.3 removed are actually gone from a capture
   rather than merely absent from the code path they were counted in.
8. **A Titanfall frame corpus**, captured through the debug views, so
   "does this look better" becomes a measurement.
9. **Psycho17 in the comparison**, via GPU capture rather than a hand port.

Then the things v0.3 deliberately did not do:

10. **Extend confidence past the source channel.** The design note asked for
    path-tracing variance to reach the exposure, the white point and the colour
    mapping. It currently reaches the glare model. The others each already have
    a filter in front of them, so the case for extending it is a measurement
    rather than an assumption — which means it needs the frame corpus first.
11. **A real variance signal rather than a temporal one.** What is measured now
    is disagreement with the reprojected history, which conflates path-tracing
    noise with anything the reprojection got wrong. The denoiser knows the
    difference and does not currently tell the tonemapper.
12. **Scene grading and display grading as separate operations.** The existing
    colour grading runs after PSDT and is a plain RGB operation that can undo the
    hue trajectory PSDT worked to produce. It is off by default, so this is
    latent rather than active.
13. **Bloom and glare are still two systems spreading the same energy.** They are
    conceptually distinct — a lens artifact and an ocular one — and both are on
    by default, which is the same shape of problem v0.3 just fixed one instance
    of. This one is not PSDT's to fix unilaterally, because `rtx.bloom` is
    Remix's and every other operator uses it.
14. **GT7 parameters as options.** Non-blocking, and it touches an operator that
    is deliberately frozen, so it wants a decision before an edit.
