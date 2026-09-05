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


def sweep_pooling():
    """
    The v0.3 pooling parameters, which the sweep above cannot see.

    measure() runs psdt_apply on an isolated pixel, so every scale reports the
    frame anchor and the pooling weights are never exercised. Sweeping the
    coherence parameters through it would print identical rows and read as
    "this parameter does nothing", which is worse than not sweeping it.
    """
    print('\n  scaleCoherence   pooled stops for a coarse neighbourhood at +N, depth term off')
    print(f'    {"value":>8s} {"+0.5":>8s} {"+1":>8s} {"+2":>8s} {"+3":>8s} {"+4":>8s}'
          f' {"+6":>8s} {"select":>8s}')
    for v in (0.0, 0.4, 0.8, 1.2, 1.6, 2.4, 4.0):
        st = P.State(opts=dict(depthSensitivity=0.0, scaleCoherence=v))
        base = (st.anchor, 1.0, st.anchor, 0.0)
        row = []
        for offset in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            far = (st.anchor + offset, 1.0, st.anchor + offset, 0.0)
            a = P.sample_adaptation(st, {'S': base, 'M': far, 'L': far})
            row.append(a.pooledLog - st.anchor)
        # Selectivity: how much of ordinary shading survives per unit of
        # lighting edge that does not. Higher is a sharper discriminator; it is
        # unbounded as the edge term goes to zero, so it is capped for display.
        select = min(row[0] / max(row[4], 1e-4), 9999.0)
        print(f'    {v:8.2f} ' + ' '.join(f'{x:8.4f}' for x in row) + f' {select:8.1f}')
    print('    select = (+0.5 kept) / (+4 kept). The knee is where a lighting edge has')
    print('    gone and ordinary shading has not moved at all; past it there is nothing')
    print('    left to buy, because the +0.5 column is already flat.')

    print('\n  pyramidCoherence   reduced value for four children straddling a boundary')
    cases = [('3 dark + 1 lit at +3', [0.0, 0.0, 0.0, 3.0], 0.0),
             ('3 dark + 1 lit at +6', [0.0, 0.0, 0.0, 6.0], 0.0),
             ('shading', [0.0, 0.2, -0.3, 0.4], None)]
    print(f'    {"value":>8s} ' + ' '.join(f'{c[0]:>21s}' for c in cases))
    for v in (0.0, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0):
        cells = []
        for _, stops, wanted in cases:
            children = [(s, 1.0) for s in stops]
            got = P.reduce_body(children, v)
            target = P.reduce_body(children, 0.0) if wanted is None else wanted
            cells.append(f'{got:10.4f} ({abs(got - target):+.4f})')
        print(f'    {v:8.2f} ' + ' '.join(f'{c:>21s}' for c in cells))
    print('    The bracketed figure is the error against what the texel should report.')
    print('    The third column is the control and its error is against the plain mean,')
    print('    which is the right answer where there is no majority to find.')


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
    sweep_pooling()
    print()
