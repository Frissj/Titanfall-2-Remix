"""
Parameter sweep. Picks defaults from measurement rather than taste.

Reports, for each setting of one parameter, the metrics that parameter can
plausibly trade against each other. A default is defensible when it sits where
the curves cross, not where it looked nice in a screenshot.
"""
import math
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import psdt_ref as P
from psdt_suite import SWATCHES, LADDER, hdr


def measure(opts):
    st = P.State(opts)
    use = keep = n = 0.0
    for sw in ('red', 'green', 'blue', 'skin', 'neon', 'magenta', 'muzzle'):
        for y in (0.18, 0.5, 1.0, 2.0, 4.0, 8.0):
            rgb = hdr(sw, y)
            out, _ = P.psdt_apply(rgb, st)
            use += 1.0 / P.gamut_limit(out, 1.0)
            oy = max(P.luminance(out), 1e-8)
            ref = [c * oy / max(P.luminance(rgb), 1e-8) for c in rgb]
            ci = P.chroma_of(P.rec709_to_ictcp(ref, 100.0))
            co = P.chroma_of(P.rec709_to_ictcp(out, 100.0))
            keep += min(co / ci, 1.5) if ci > 1e-6 else 1.0
            n += 1

    # How much darker a saturated colour renders than a neutral of the same
    # scene luminance. This is the cost the concession pays.
    dark = 0.0
    dn = 0
    for sw in ('red', 'green', 'blue', 'neon'):
        for y in (0.5, 1.0, 2.0, 4.0):
            a, _ = P.psdt_apply(hdr(sw, y), st)
            b, _ = P.psdt_apply([y, y, y], st)
            dark += math.log2(max(P.luminance(a), 1e-9) / max(P.luminance(b), 1e-9))
            dn += 1

    clip = tot = 0
    for sw in SWATCHES:
        for y in LADDER:
            out, _ = P.psdt_apply(hdr(sw, y), st)
            tot += 1
            if max(out) >= 0.999:
                clip += 1

    inv = 0
    for sw in SWATCHES:
        prev = -1.0
        for i in range(401):
            out, _ = P.psdt_apply(hdr(sw, 2 ** (-10.0 + 24.0 * i / 400)), st)
            oy = P.luminance(out)
            if oy < prev - 1e-6:
                inv += 1
            prev = oy

    # Punch: display stops per scene stop across the anchor. Separation: how
    # many display stops the range from +2 to +5 scene stops still occupies -
    # the difference between highlights that have structure and highlights that
    # are one flat white.
    def dstops(a, b):
        ya = max(P.luminance(P.psdt_apply([2 ** (st.anchor + a)] * 3, st)[0]), 1e-9)
        yb = max(P.luminance(P.psdt_apply([2 ** (st.anchor + b)] * 3, st)[0]), 1e-9)
        return math.log2(yb / ya) / (b - a)

    return dict(use=use / n, keep=keep / n, dark=dark / dn,
                clip=100.0 * clip / tot, inv=inv,
                punch=dstops(-1.0, 1.0), sep=dstops(2.0, 5.0),
                shadow=dstops(-7.0, -4.0))


def sweep(name, values, fmt='{:.2f}'):
    print(f'\n  {name}')
    print(f'    {"value":>8s} {"bnd use":>9s} {"chroma":>8s} {"sat.dark":>10s} '
          f'{"shadow":>8s} {"punch":>8s} {"hi.sep":>8s} {"clip":>7s} {"Yinv":>5s}')
    for v in values:
        m = measure({name: v})
        print(f'    {fmt.format(v):>8s} {m["use"]:9.3f} {m["keep"]:8.3f} {m["dark"]:9.2f}st '
              f'{m["shadow"]:8.3f} {m["punch"]:8.3f} {m["sep"]:8.4f} {m["clip"]:6.1f}% {m["inv"]:5d}')


if __name__ == '__main__':
    print('PSDT parameter sweep')
    print('  boundary use     output chroma / max displayable chroma at that luminance. Higher is better.')
    print('  chroma kept      chroma vs the (often impossible) scene colour at the same luminance.')
    print('  sat. darkening   stops a saturated colour renders below a neutral of the same scene')
    print('                   luminance. This is what the luminance concession costs. Less negative')
    print('                   means brighter and more bleached; more negative means darker and more')
    print('                   colourful. There is no right answer, only a trade to state.')
    print('  shadow/punch/    display stops per scene stop over [-7,-4], [-1,+1] and [+2,+5] stops')
    print('  hi.sep           from the anchor. Punch below 1.0 is a flat midtone; hi.sep near 0 is')
    print('                   a highlight range with no structure left in it.')
    sweep('chromaPreservation', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sweep('luminanceConcession', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sweep('hueTrajectory', [0.0, 0.25, 0.5, 0.6, 0.75, 1.0])
    sweep('midtoneContrast', [0.95, 1.05, 1.15, 1.25, 1.4])
    sweep('highlightRolloff', [0.35, 0.45, 0.6, 0.75, 0.9])
    print()
