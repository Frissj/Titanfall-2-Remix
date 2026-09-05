# PSDT harness

Three tools. Between them they answer the two questions a display transform
has to answer before anyone looks at a screenshot: *is the maths right*, and
*does the thing that runs on the GPU match the thing I tested*.

```bash
python3 tools/psdt/psdt_check.py    # shader / header / C++ agree with each other
python3 tools/psdt/psdt_suite.py    # the transform's own behaviour, measured
python3 tools/psdt/psdt_sweep.py    # where a parameter's default should sit
tools/psdt/regen_results.sh         # both of the above into RESULTS.txt
```

`psdt_check.py` exits non-zero on failure and is the one to run first.

---

## `psdt_ref.py` — the CPU reference

A port of the shaders where every constant, clamp and order of operations is
the one the shader uses. It is not an idealised model of the transform; it is
the transform, somewhere it can be measured.

That claim is now enforced rather than asserted. `psdt_check.py` compares the
shared constants, the chroma references, the shoulder clamp and the black-level
epsilon between the two files and fails if any of them have drifted — which is
exactly how v0.1's two divergences got in and stayed in.

Everything in the file that does *not* correspond to shader code is marked
`harness only`. That is the operator controls, and the two temporal filters'
driving loop.

## `psdt_suite.py` — behaviour

Fifteen sections. `python3 tools/psdt/psdt_suite.py <section>` runs one.

| section | what it establishes |
|---|---|
| `probe` | the colour-space constants are what the shipped matrices actually produce, per gamut and per space; the perceptual round trips are lossless; a colour that is the local white lands exactly on the adapted neutral axis |
| `curve` | the luminance curve is monotonic over 40 stops, C1 at both joins, and its analytic derivative matches a central difference to 1e-9 |
| `classify` | the classifier can tell a neon sign from a sunlit wall, an emissive screen from a specular glint, and sky from either — and what it collapses to when the gbuffer is unavailable |
| `illuminant` | radiance over albedo recovers the illuminant of a room whose materials are strongly biased, where grey-world reports the paint; and a surface neutral under a non-D65 light is still inside the display volume |
| `gamut` | what the per-gamut achromatic axis and chroma normaliser fixed, and by how much |
| `gt7space` | what GT7's Rec.2020 input assumption costs on a Rec.709 framebuffer — the number that decides whether GT7 is usable as a control |
| `response` | the luminance transfer and clipping rate, against four controls |
| `colour` | chroma retention as a colour runs out of display volume, and the four terms that decide when compression starts |
| `hue` | the path to white, and whether it has a discontinuity anywhere in a 1/42-stop sweep |
| `contrast` | how much local contrast survives, and the detail-magnitude rolloff that stops a lamp being treated as texture |
| `glare` | the halo radius grows with source luminance, the halo carries the source's colour, both lobes contribute, and the threshold tracks adaptation |
| `coherence` | what came across from Auto Exposure Plus when it stopped running: the robust pyramid reduction, the cross-scale edge stop, and what each of the three ownership settings costs |
| `confidence` | a one-frame spike against a source that persists, a source that flickers, and a light going out — the four cases that have to be told apart from each other |
| `temporal` | step response of the two filters in series: settling times, overshoot, false cuts, and noise attenuation |
| `compare` | the section-33 weighted score, PSDT against GT7, AgX, Reinhard and Hable |
| `invariants` | the nine properties the transform is supposed to hold, as pass/fail |

### What it cannot establish

It runs the CPU reference. So it establishes that the maths is right —
monotonic, continuous, in gamut, settling when it should. It does **not**
establish that the Vulkan implementation is right, because it does not run the
Vulkan implementation, and it does not establish that PSDT looks better in
Titanfall 2, because a swatch ladder is not a game frame.

Both of those need the GPU. The in-image debug views
(`rtx.tonemap.psdt.debugView`) exist for the second one, and the first thing to
check in-game is the *classification* view rather than the final image: if the
roles are wrong, everything downstream is answering the right question about
the wrong pixels.

### Measuring the real renderer

Everything here runs on the CPU. The engine-side counterpart is
`[TonemapProbe]`, which logs luminance, ICtCp chroma, hue, hue delta and
clipping counts for the tonemapper's input/output pair, tagged with the
operator id and whether Auto Exposure Plus and colour grading were on.

That tagging is what makes the operator x Plus comparison tractable: run the
same scene under each configuration with `rtx.tonemap.colorGradingEnabled=0`
and bloom off, then diff the `[TonemapProbe]` lines. Enable it with
`rtx.logSurfaceCoverage`; add `rtx.tf2HeavyProbes` for the per-sample
`[TonemapProbe.px]` lines.

### Psycho17 is not in the comparison

It is a 754-line vendored operator and a hand port could not be verified
against the shader that runs. An unverified control is worse than a missing
one, because its numbers look exactly as authoritative as the verified ones.
Comparing against it needs a GPU capture.

## `psdt_check.py` — consistency

The class of bug a CPU reference structurally cannot catch: a state slot
written under one name and read under another, a push-constant field the C++
never sets, a binding index used twice, an operator id that disagrees between
the enum and the shader header, a constant that drifted between the shader and
the reference.

None of those change the reference's answers. All of them either break the
build or silently produce a different transform on the GPU.

## `psdt_sweep.py` — defaults

For each setting of one parameter, the metrics that parameter actually trades
against each other. A default is defensible when it sits where the curves
cross, not where it looked right in a screenshot.

`highlightRolloff` 0.45, `luminanceConcession` 0.8 and `chromaPreservation` 2.5
are set from this rather than by taste.

## `RESULTS.txt`

A checked-in run of both tools, so a diff shows what a change did to the
numbers. Regenerate with the command at the top of the file.
