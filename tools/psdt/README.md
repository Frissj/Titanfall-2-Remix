# PSDT tools

CPU reference implementation of the Perceptual Scene Display Transform, and the
objective test suite that sets its defaults.

```bash
python3 tools/psdt/psdt_suite.py            # everything
python3 tools/psdt/psdt_suite.py curve      # one section
python3 tools/psdt/psdt_sweep.py            # parameter sweeps
```

Standard library only. No build, no GPU.

## Why this exists

A screenshot cannot tell you whether the tone curve is monotonic between the two
points you happened to look at, how far a hue moved on its way to white, or how
much of the input's local contrast survived. Those are the questions that decide
whether a display transform is right, and they are all measurable.

Three defects in the shipped transform were found here rather than on screen,
and none of them would have been obvious in a screenshot:

* the base/detail decomposition was a first-order Taylor expansion of the curve,
  so a bright object against a dark surround extrapolated straight past the
  shoulder and clipped. Grey clipped at 2.5 stops over the anchor;
* the colour-volume pressure used `1 - (1-Py)(1-Pc)(1-Pg)`, which is an OR. Any
  saturated colour has a high Pc whether or not it fits, so a mid-luminance red
  the display could show perfectly well was being stripped to 0.26 of its
  chroma - the "everything goes grey" failure the stage exists to prevent;
* chroma was compressed by a factor measured *before* the hue rotation, so the
  result overshot the boundary and the final gamut fit clipped the difference.
  About 9% of the available chroma, at full hue trajectory.

## Layout

| file | what it is |
|---|---|
| `psdt_ref.py` | line-for-line port of the shaders under `src/dxvk/shaders/rtx/pass/psdt/`, plus GT7 / Reinhard / Hable as controls |
| `psdt_suite.py` | the eight measurement sections |
| `psdt_sweep.py` | one-parameter sweeps, for choosing defaults from evidence |

`psdt_ref.py` is deliberately literal rather than idiomatic Python, so that a
divergence from the GPU shows up as a visible diff and not as a judgement call.
When you change a shader, change it here too and re-run the suite.

## Sections

| section | answers |
|---|---|
| `probe` | are the colour matrices and their inverses right, and what is the largest chroma this display can actually produce |
| `curve` | is the curve monotonic and C1 everywhere, and where do the anchors land in code values |
| `response` | the luminance transfer function, against the controls |
| `colour` | chroma retention as a colour runs out of display volume |
| `hue` | the path to white, and whether it has a discontinuity in it |
| `contrast` | how much local contrast survives, by luminance |
| `compare` | the §33 score against GT7, Reinhard and Hable |
| `invariants` | the seven perceptual properties the transform is supposed to hold |

## Reading the metrics

**boundary use** is the honest one. It is the output chroma as a fraction of the
most chroma the display can show at that output luminance and hue. 1.0 means the
transform did as well as physics allows. **chroma kept** is measured against the
scene colour scaled to the same output luminance - which is frequently a colour
no display can show - so it is comparable between operators but is not a target
anyone can reach.

**sat. darkening**, in the sweep, is the price of the luminance concession: how
many stops below a neutral a saturated colour of the same scene luminance
renders. There is no correct value. Less darkening means brighter and more
bleached; more means darker and more colourful. The setting exists so the trade
is stated rather than assumed.
