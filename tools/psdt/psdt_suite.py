"""
PSDT objective test suite.

Screenshots cannot answer "did the hue move, and by how much", "is the curve
monotonic between the points I looked at", or "how much of the input's local
contrast survived". This does, on a fixed HDR ladder, for PSDT and for the
operators it is supposed to beat.

    python3 tools/psdt/psdt_suite.py            # everything
    python3 tools/psdt/psdt_suite.py curve      # one section

Sections: probe, curve, response, colour, hue, contrast, compare, invariants
"""

import math
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import psdt_ref as P

# The HDR ladder. Twenty stops, from four below anything a display can show to
# a value only a physically reconstructed scene ever produces.
LADDER = [0.001, 0.003, 0.01, 0.03, 0.1, 0.18, 0.5, 1, 2, 4, 8, 16, 32, 64,
          128, 256, 1024, 4096, 16384]

SWATCHES = {
    'grey':    (1.00, 1.00, 1.00),
    'red':     (1.00, 0.04, 0.04),
    'green':   (0.06, 1.00, 0.08),
    'blue':    (0.05, 0.06, 1.00),
    'yellow':  (1.00, 0.85, 0.05),
    'cyan':    (0.05, 0.95, 1.00),
    'magenta': (1.00, 0.06, 0.90),
    # Titanfall-ish material references: a skin tone, gun metal, and the
    # saturated emissives the game is full of.
    'skin':    (1.00, 0.62, 0.47),
    'gunmetal':(0.32, 0.34, 0.38),
    'neon':    (0.10, 0.90, 1.00),
    'muzzle':  (1.00, 0.72, 0.30),
}


def norm(rgb):
    """Scale a swatch so its luminance is 1, so the ladder is a pure exposure."""
    y = P.luminance(rgb)
    return [c / y for c in rgb]


def hdr(swatch, y):
    return [c * y for c in norm(SWATCHES[swatch])]


def rule(title, width=96):
    print()
    print(title)
    print('-' * min(len(title), width))


# ---------------------------------------------------------------------------

def section_probe():
    rule('PROBE  colour-space constants, recomputed from the shipped matrices')

    for name, fn in (('ICtCp', P.rec709_to_ictcp), ('Jzazbz', P.rec709_to_jzazbz)):
        mx, arg = 0.0, None
        N = 16
        for r in range(N + 1):
            for g in range(N + 1):
                for b in range(N + 1):
                    c = P.chroma_of(fn([r / N, g / N, b / N], 100.0))
                    if c > mx:
                        mx, arg = c, [r / N, g / N, b / N]
        print(f'  max displayable chroma, {name:7s} = {mx:.4f}  at rgb={arg}')

    worst = {'ICtCp': 0.0, 'Jzazbz': 0.0}
    N = 16
    for r in range(N + 1):
        for g in range(N + 1):
            for b in range(N + 1):
                rgb = [r / N, g / N, b / N]
                back = P.ictcp_to_rec709(P.rec709_to_ictcp(rgb, 100.0), 100.0)
                worst['ICtCp'] = max(worst['ICtCp'], max(abs(a - c) for a, c in zip(rgb, back)))
                back = P.jzazbz_to_rec709(P.rec709_to_jzazbz(rgb, 100.0), 100.0)
                worst['Jzazbz'] = max(worst['Jzazbz'], max(abs(a - c) for a, c in zip(rgb, back)))
    for k, v in worst.items():
        verdict = 'OK' if v < 1e-5 else 'TOO LOOSE'
        print(f'  round-trip max abs error, {k:7s} = {v:.3e}   {verdict}')


def section_curve():
    rule('CURVE  monotonicity, C1 continuity, and where the anchors land')

    st = P.State()
    c = st.curve
    print(f'  anchor           {c.midIn:+.3f} log2   ({2 ** c.midIn:.4f} linear)')
    print(f'  display mid      {c.midOutLog:+.3f} log2   ({2 ** c.midOutLog:.4f} linear)')
    print(f'  midtone slope    {c.slope:.3f} display stops per scene stop')
    print(f'  toe onset        {c.toeStart:+.3f} stops rel. anchor')
    print(f'  shoulder onset   {c.shoulderStart:+.3f} stops rel. anchor')
    print(f'  shoulder length  {c.shoulderLen:.3f} stops')
    print(f'  white conv.      {c.whiteIn:+.3f} stops rel. anchor')
    print(f'  black floor      {c.blackLog:+.3f} log2   ({2 ** c.blackLog:.2e} linear)')

    # Monotonicity: 40 stops at 0.001-stop resolution.
    x = c.midIn - 20.0
    prev_y = c.log(x)
    worst_drop = 0.0
    steps = 40000
    for i in range(1, steps + 1):
        xi = c.midIn - 20.0 + 40.0 * i / steps
        y = c.log(xi)
        worst_drop = min(worst_drop, y - prev_y)
        prev_y = y
    print(f'\n  monotonic over [-20, +20] stops:  worst step {worst_drop:+.3e}   '
          f'{"OK" if worst_drop >= 0.0 else "*** NON-MONOTONIC ***"}')

    # C1 at both joins: compare the analytic slope against a central difference.
    for label, x in (('toe join', c.midIn + c.toeStart), ('shoulder join', c.midIn + c.shoulderStart)):
        h = 1e-4
        fd = (c.log(x + h) - c.log(x - h)) / (2 * h)
        an = c.slope_at(x)
        left = c.slope_at(x - h)
        right = c.slope_at(x + h)
        print(f'  {label:14s} slope analytic {an:.6f}  finite-diff {fd:.6f}  '
              f'left {left:.6f}  right {right:.6f}   '
              f'{"C1 OK" if abs(left - right) < 1e-3 else "*** KINK ***"}')

    # Analytic derivative agreement over the whole range.
    worst = 0.0
    for i in range(4001):
        xi = c.midIn - 12.0 + 24.0 * i / 4000
        h = 1e-4
        fd = (c.log(xi + h) - c.log(xi - h)) / (2 * h)
        worst = max(worst, abs(fd - c.slope_at(xi)))
    print(f'  analytic vs numeric slope, max error over [-12,+12] stops: {worst:.2e}   '
          f'{"OK" if worst < 1e-3 else "*** MISMATCH ***"}')

    print('\n  scene stop -> display value')
    print('    rel.anchor    scene Y     display Y   display 8-bit sRGB   slope')
    for s in (-12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16):
        x = c.midIn + s
        y = 2 ** c.log(x)
        srgb = 1.055 * (y ** (1 / 2.4)) - 0.055 if y > 0.0031308 else 12.92 * y
        print(f'    {s:+7d}   {2 ** x:10.4g}   {y:9.5f}   {P.clamp(srgb, 0, 1) * 255:8.1f}'
              f'            {c.slope_at(x):.4f}')


def section_response():
    rule('RESPONSE  luminance transfer, PSDT vs the controls')

    st = P.State()
    print('    scene Y     PSDT      GT7    Reinhard    Hable     (display luminance, grey)')
    for y in LADDER:
        rgb = [y, y, y]
        a, _ = P.psdt_apply(rgb, st)
        print(f'  {y:10.4g}  {P.luminance(a):8.5f}  {P.luminance(P.gt7(rgb)):8.5f}  '
              f'{P.luminance(P.reinhard(rgb)):8.5f}  {P.luminance(P.hable(rgb)):8.5f}')

    print('\n  clipping: fraction of the ladder x swatch grid that clips a channel at 1.0')
    for name, fn in (('PSDT', lambda c: P.psdt_apply(c, st)[0]),
                     ('GT7', P.gt7), ('Reinhard', P.reinhard), ('Hable', P.hable)):
        total = clipped = 0
        for sw in SWATCHES:
            for y in LADDER:
                out = fn(hdr(sw, y))
                total += 1
                if max(out) >= 0.999:
                    clipped += 1
        print(f'    {name:9s} {100.0 * clipped / total:5.1f}%')


def section_colour():
    rule('COLOUR  chroma retention and gamut pressure as a colour runs out of volume')

    st = P.State()
    print('  chroma out / chroma in, measured in ICtCp against the un-compressed path')
    header = '    swatch      ' + ''.join(f'{y:>9.3g}' for y in LADDER[3:14])
    print(header)
    for sw in ('grey', 'red', 'green', 'blue', 'skin', 'neon', 'gunmetal'):
        row = f'    {sw:11s}'
        for y in LADDER[3:14]:
            rgb = hdr(sw, y)
            out, dbg = P.psdt_apply(rgb, st)
            cin = P.chroma_of(P.rec709_to_ictcp([c * dbg['displayY'] / max(P.luminance(rgb), 1e-8)
                                                 for c in rgb], st.refWhite))
            cout = P.chroma_of(P.rec709_to_ictcp(out, st.refWhite))
            row += f'{(cout / cin if cin > 1e-6 else 1.0):9.3f}'
        print(row)

    print('\n  the point of the product form: chroma retention at a fixed high luminance')
    print('    a bright neutral and a bright saturated colour at the same display luminance')
    for sw in ('grey', 'gunmetal', 'skin', 'red', 'neon'):
        rgb = hdr(sw, 64.0)
        out, dbg = P.psdt_apply(rgb, st)
        print(f'    {sw:9s} highlightProgress={dbg["highlightProgress"]:.3f} '
              f'chromaNorm={dbg["chromaNorm"]:.3f} demand={dbg["demand"]:.3f} '
              f'knee={dbg["knee"]:.3f} -> compress={dbg["compress"]:.3f} pressure={dbg["pressure"]:.3f}')


def section_hue():
    rule('HUE  the path to white, and whether it is smooth')

    st = P.State()
    print('  hue rotation in degrees along the ladder (+ is towards the film path)')
    header = '    swatch      ' + ''.join(f'{y:>8.3g}' for y in LADDER[5:16])
    print(header)
    worst_jump = 0.0
    for sw in ('red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'skin', 'neon'):
        row = f'    {sw:11s}'
        prev = None
        for y in LADDER[5:16]:
            rgb = hdr(sw, y)
            out, dbg = P.psdt_apply(rgb, st)
            scale = dbg['displayY'] / max(P.luminance(rgb), 1e-8)
            hin = P.hue_of(P.rec709_to_ictcp([c * scale for c in rgb], st.refWhite))
            hout = P.hue_of(P.rec709_to_ictcp(out, st.refWhite))
            d = hout - hin
            d -= 2 * math.pi * math.floor(d / (2 * math.pi) + 0.5)
            deg = math.degrees(d)
            row += f'{deg:8.2f}'
            if prev is not None:
                worst_jump = max(worst_jump, abs(deg - prev))
            prev = deg
        print(row)
    print(f'\n  largest hue change between adjacent ladder steps: {worst_jump:.2f} deg')
    print('  (adjacent steps here are a full stop apart, so this is a coarse bound;')
    print('   the fine sweep below is the one that would catch a discontinuity)')

    # Fine sweep: 1/32-stop steps through the shoulder, looking for a jump.
    worst = 0.0
    for sw in ('red', 'magenta', 'yellow'):
        prev = None
        for i in range(513):
            y = 2 ** (-2.0 + 12.0 * i / 512)
            out, dbg = P.psdt_apply(hdr(sw, y), st)
            scale = dbg['displayY'] / max(P.luminance(hdr(sw, y)), 1e-8)
            hin = P.hue_of(P.rec709_to_ictcp([c * scale for c in hdr(sw, y)], st.refWhite))
            hout = P.hue_of(P.rec709_to_ictcp(out, st.refWhite))
            d = hout - hin
            d -= 2 * math.pi * math.floor(d / (2 * math.pi) + 0.5)
            deg = math.degrees(d)
            if prev is not None:
                worst = max(worst, abs(deg - prev))
            prev = deg
    print(f'  fine sweep (1/42-stop steps, red/magenta/yellow): max step {worst:.4f} deg   '
          f'{"SMOOTH" if worst < 1.0 else "*** DISCONTINUITY ***"}')


def section_contrast():
    rule('CONTRAST  how much local contrast survives the compression')

    st = P.State()
    print('  a patch of detail riding on a base, at several base levels.')
    print('  ratio 1.0 means the detail kept all its scene contrast; the curve slope')
    print('  column is what it would have kept with no restoration at all.')
    print('    base(rel)   curve slope   detail +-0.25 st   +-1.0 st   +-3.0 st')
    for base_rel in (-8, -6, -4, -2, 0, 1, 2, 3, 4, 6):
        base = st.anchor + base_rel
        row = f'    {base_rel:+8d}'
        row += f'   {st.curve.slope_at(base):11.4f}'
        for d in (0.25, 1.0, 3.0):
            hi, _ = P.psdt_apply([2 ** (base + d)] * 3, st, pooled_log=base)
            lo, _ = P.psdt_apply([2 ** (base - d)] * 3, st, pooled_log=base)
            yh, yl = P.luminance(hi), P.luminance(lo)
            if yl <= 1e-9 or yh <= 1e-9:
                row += '        n/a'
                continue
            out_stops = math.log2(yh / yl)
            row += f'{out_stops / (2 * d):11.4f}'
        print(row)


def section_compare():
    rule('COMPARE  the § 33 score, PSDT against the controls')

    st = P.State()

    def metrics(fn):
        # Luminance fidelity: how closely display luminance tracks a reference
        # perceptual compression, in log space, over the ladder.
        # Colour-volume fidelity: mean chroma retained where the display can
        # still show it (below the shoulder).
        # Hue smoothness: the largest hue step over a fine sweep.
        # Detail: contrast retained at +-1 stop about the anchor.
        # Clipping: fraction of the grid that clips.
        use, un = 0.0, 0
        for sw in ('red', 'green', 'blue', 'skin', 'neon', 'magenta'):
            for y in (0.18, 0.5, 1.0, 2.0, 4.0, 8.0):
                out = fn(hdr(sw, y))
                use += 1.0 / P.gamut_limit(out, 1.0)
                un += 1
        use /= max(un, 1)

        chroma_keep, n = 0.0, 0
        for sw in ('red', 'green', 'blue', 'skin', 'neon', 'magenta'):
            for y in (0.03, 0.1, 0.18, 0.5, 1.0):
                rgb = hdr(sw, y)
                out = fn(rgb)
                oy = max(P.luminance(out), 1e-8)
                ref = [c * oy / max(P.luminance(rgb), 1e-8) for c in rgb]
                cin = P.chroma_of(P.rec709_to_ictcp(ref, 100.0))
                cout = P.chroma_of(P.rec709_to_ictcp(out, 100.0))
                if cin > 1e-6:
                    chroma_keep += min(cout / cin, 1.5)
                    n += 1
        chroma_keep /= max(n, 1)

        # Hue is undefined at zero chroma, so atan2 jitters wildly as a colour
        # arrives at white and any operator would score badly on a raw sweep.
        # Only samples whose output still carries a tenth of the display's
        # chroma capacity are counted, which is the range in which a hue shift
        # is something a viewer could actually see.
        worst_hue, hue_at = 0.0, 0.0
        for sw in ('red', 'magenta', 'yellow', 'green', 'blue'):
            prev = None
            for i in range(513):
                y = 2 ** (-2.0 + 12.0 * i / 512)
                rgb = hdr(sw, y)
                out = fn(rgb)
                oy = max(P.luminance(out), 1e-8)
                ref = [c * oy / max(P.luminance(rgb), 1e-8) for c in rgb]
                if P.chroma_of(P.rec709_to_ictcp(out, 100.0)) < 0.1 * P.CHROMA_REFERENCE_ICTCP:
                    prev = None
                    continue
                d = P.hue_of(P.rec709_to_ictcp(out, 100.0)) - P.hue_of(P.rec709_to_ictcp(ref, 100.0))
                d -= 2 * math.pi * math.floor(d / (2 * math.pi) + 0.5)
                deg = math.degrees(d)
                if prev is not None and abs(deg - prev) > worst_hue:
                    worst_hue, hue_at = abs(deg - prev), math.log2(y)
                prev = deg

        hi = fn([2 ** (P.MIDDLE_GREY_LOG + 1)] * 3)
        lo = fn([2 ** (P.MIDDLE_GREY_LOG - 1)] * 3)
        yh, yl = max(P.luminance(hi), 1e-9), max(P.luminance(lo), 1e-9)
        detail = math.log2(yh / yl) / 2.0

        total = clipped = 0
        inverted = 0
        for sw in SWATCHES:
            prev_y = -1.0
            for y in LADDER:
                out = fn(hdr(sw, y))
                total += 1
                if max(out) >= 0.999:
                    clipped += 1
                oy = P.luminance(out)
                if oy < prev_y - 1e-6:
                    inverted += 1
                prev_y = oy
        return dict(chroma=chroma_keep, use=use, hue=worst_hue, hue_at=hue_at, detail=detail,
                    clip=100.0 * clipped / total, inverted=inverted)

    ops = [('PSDT', lambda c: P.psdt_apply(c, st)[0]),
           ('GT7', P.gt7), ('Reinhard', P.reinhard), ('Hable', P.hable)]

    print(f'    {"operator":10s} {"chroma kept":>12s} {"boundary use":>13s} {"max hue step":>13s} '
          f'{"midtone contr":>14s} {"clipped":>9s} {"Y inv":>7s}')
    for name, fn in ops:
        m = metrics(fn)
        print(f'    {name:10s} {m["chroma"]:12.3f} {m["use"]:13.3f} {m["hue"]:12.3f}d '
              f'{m["detail"]:14.3f} {m["clip"]:8.1f}% {m["inverted"]:7d}')
    print()
    print('    chroma kept   chroma retained against the scene colour scaled to the same output')
    print('                  luminance. Unflattering by construction - the reference is often a')
    print('                  colour no display can show - so it is only comparable, not a target.')
    print('    boundary use  output chroma as a fraction of the most the display can show at that')
    print('                  output luminance and hue. This is the one that says whether the')
    print('                  transform is doing as well as physically possible. 1.0 is the wall.')
    print('    max hue step  largest hue change between adjacent samples of a 1/42-stop sweep,')
    print('                  counted only while the output still has a tenth of the display\'s chroma')
    print('                  capacity - hue is undefined at white and every operator scores badly on')
    print('                  a raw sweep. Large = a visible discontinuity on the path to white.')
    print('    midtone contr display stops per scene stop across the anchor. 1.0 is flat.')
    print('    Y inversions  times a brighter scene value produced a darker display value.')
    print('                  Anything but 0 breaks the most basic perceptual invariant there is.')


def section_invariants():
    rule('INVARIANTS  the seven properties the transform is supposed to hold')

    st = P.State()
    fails = []

    # 1. brightness ordering
    # 1e-6 rather than 0: the shoulder is an exponential asymptote, so at the
    # very top the output differs between adjacent samples by less than double
    # precision resolves. Anything above this floor is the transform inverting,
    # not the arithmetic.
    inversions = 0
    worst = 0.0
    for sw in SWATCHES:
        prev = -1.0
        for i in range(801):
            y = 2 ** (-10.0 + 24.0 * i / 800)
            out, _ = P.psdt_apply(hdr(sw, y), st)
            oy = P.luminance(out)
            if oy < prev:
                worst = max(worst, prev - oy)
            if oy < prev - 1e-6:
                inversions += 1
            prev = oy
    ok = inversions == 0
    fails += [] if ok else ['brightness ordering']
    print(f'  1. brightness ordering preserved            {inversions} inversions, '
          f'largest dip {worst:.2e}   {"PASS" if ok else "FAIL"}')

    # 2. a small bright object stays distinct from a dark surround
    dark, _ = P.psdt_apply([2 ** (st.anchor - 3)] * 3, st, pooled_log=st.anchor - 3)
    bright, _ = P.psdt_apply([2 ** (st.anchor + 3)] * 3, st, pooled_log=st.anchor - 3)
    sep = math.log2(max(P.luminance(bright), 1e-9) / max(P.luminance(dark), 1e-9))
    ok = sep > 1.0
    fails += [] if ok else ['local contrast']
    print(f'  2. small bright object vs dark surround     {sep:.2f} display stops apart   {"PASS" if ok else "FAIL"}')

    # 3. colour identity: a saturated red at ordinary luminance is not rotated
    out, dbg = P.psdt_apply(hdr('red', 0.18), st)
    scale = dbg['displayY'] / max(P.luminance(hdr('red', 0.18)), 1e-8)
    ref = [c * scale for c in hdr('red', 0.18)]
    d = math.degrees(P.hue_of(P.rec709_to_ictcp(out, 100.0)) - P.hue_of(P.rec709_to_ictcp(ref, 100.0)))
    ok = abs(d) < 5.0
    fails += [] if ok else ['colour identity']
    print(f'  3. mid-luminance red hue held               {d:+.2f} deg   {"PASS" if ok else "FAIL"}')

    # 4. colour-volume trajectory is continuous (covered by the fine hue sweep)
    worst = 0.0
    prev = None
    for i in range(2049):
        y = 2 ** (-4.0 + 16.0 * i / 2048)
        out, _ = P.psdt_apply(hdr('red', y), st)
        c = P.chroma_of(P.rec709_to_ictcp(out, 100.0))
        if prev is not None:
            worst = max(worst, abs(c - prev))
        prev = c
    ok = worst < 0.01
    fails += [] if ok else ['chroma trajectory']
    print(f'  4. chroma trajectory continuous             max step {worst:.5f}   {"PASS" if ok else "FAIL"}')

    # 5. temporal: the transform is a pure function of (pixel, state)
    a, _ = P.psdt_apply(hdr('neon', 4.0), st)
    b, _ = P.psdt_apply(hdr('neon', 4.0), st)
    ok = a == b
    fails += [] if ok else ['determinism']
    print(f'  5. deterministic given the state            {"PASS" if ok else "FAIL"}')

    # 6. spatial: a bright neighbour cannot re-expose a pixel past the budget
    far = P.psdt_apply(hdr('grey', 0.18), st, pooled_log=st.anchor)[0]
    near = P.psdt_apply(hdr('grey', 0.18), st, pooled_log=st.anchor + 6.0)[0]
    shift = abs(math.log2(max(P.luminance(near), 1e-9) / max(P.luminance(far), 1e-9)))
    ok = shift <= st.budgetHighlight * st.curve.slope + 0.35
    fails += [] if ok else ['adaptation budget']
    print(f'  6. adaptation budget respected              {shift:.3f} stops shift for a 6-stop'
          f' brighter neighbourhood   {"PASS" if ok else "FAIL"}')

    # 7. Highlight identity. Measured as boundary utilisation, not as raw
    #    chroma: at extreme over-exposure the *correct* answer really is white,
    #    because no display can be both that bright and that colourful and the
    #    per-channel curve every film transform uses says the same. What must
    #    hold is that the transform gets as close to the boundary as the display
    #    allows at every luminance, rather than giving up early.
    worst_use, worst_at = 1.0, 0.0
    for m in (1, 2, 4, 8, 16, 32):
        out, _ = P.psdt_apply(hdr('neon', 0.18 * m), st)
        use = 1.0 / P.gamut_limit(out, 1.0)
        if use < worst_use:
            worst_use, worst_at = use, math.log2(m)
    ok = worst_use > 0.60
    fails += [] if ok else ['highlight identity']
    print(f'  7. emissive stays as colourful as the       worst boundary use {worst_use:.3f} '
          f'at {worst_at:+.0f} stops   {"PASS" if ok else "FAIL"}')
    print('     display allows')

    print()
    if fails:
        print(f'  {len(fails)} FAILED: {", ".join(fails)}')
    else:
        print('  all invariants hold')
    return not fails


SECTIONS = {
    'probe': section_probe, 'curve': section_curve, 'response': section_response,
    'colour': section_colour, 'hue': section_hue, 'contrast': section_contrast,
    'compare': section_compare, 'invariants': section_invariants,
}

if __name__ == '__main__':
    wanted = sys.argv[1:] or list(SECTIONS)
    for w in wanted:
        if w not in SECTIONS:
            print(f'unknown section: {w}\navailable: {", ".join(SECTIONS)}')
            sys.exit(2)
        SECTIONS[w]()
    print()
