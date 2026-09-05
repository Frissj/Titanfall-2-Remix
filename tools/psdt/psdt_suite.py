"""
PSDT objective test suite.

Screenshots cannot answer "did the hue move, and by how much", "is the curve
monotonic between the points I looked at", "how long does the exposure take to
settle after a cut", or "is that halo the pooling crossing a depth edge". This
does, on fixed synthetic input, for PSDT and for the operators it is supposed
to beat.

    python3 tools/psdt/psdt_suite.py            # everything
    python3 tools/psdt/psdt_suite.py curve      # one section

Sections: probe, curve, classify, illuminant, gamut, gt7space, response, colour,
          hue, contrast, glare, temporal, compare, invariants

What this can and cannot establish
----------------------------------
It runs the CPU reference in tools/psdt/psdt_ref.py, which is a line-for-line
port of the shaders. So it establishes that the *maths* is right: monotonic,
continuous, in gamut, settling when it should. It does not establish that the
Vulkan implementation is right, because it does not run the Vulkan
implementation, and it does not establish that PSDT looks better in Titanfall 2,
because a swatch ladder is not a game frame. Both of those need the GPU. The
in-image debug views (rtx.tonemap.psdt.debugView) are the tool for the second
one, and they exist so that the first thing you check in-game is the
classification rather than the final image.
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


def ops_for(st):
    """The operator set every comparison uses."""
    return [('PSDT', lambda c: P.psdt_apply(c, st)[0]),
            ('GT7', P.gt7), ('AgX', P.agx),
            ('Reinhard', P.reinhard), ('Hable', P.hable)]


# ---------------------------------------------------------------------------

def section_probe():
    rule('PROBE  colour-space constants, recomputed from the shipped matrices')

    # Max displayable chroma, per (space, gamut). These are the denominators
    # that turn an absolute chroma into a fraction of the display's capacity,
    # and they are not interchangeable across gamuts - v0.1 used the Rec.709
    # pair for all three, which silently rescaled the chroma term of the
    # colour-volume pressure whenever the gamut was changed.
    print('  max displayable chroma, over each gamut\'s own cube:')
    print(f'    {"gamut":9s} {"ICtCp":>9s} {"Jzazbz":>9s}   (shipped constant in brackets)')
    N = 20
    for gname, gamut in (('Rec709', P.GAMUT_709), ('P3', P.GAMUT_P3), ('Rec2020', P.GAMUT_2020)):
        row = {}
        for sname, space in (('ICtCp', P.SPACE_ICTCP), ('Jzazbz', P.SPACE_JZAZBZ)):
            mx = 0.0
            for i in range(N + 1):
                for j in range(N + 1):
                    for k in range(N + 1):
                        if not (i in (0, N) or j in (0, N) or k in (0, N)):
                            continue
                        c = [i / N, j / N, k / N]
                        rgb = P.from_display_gamut(c, gamut)
                        mx = max(mx, P.chroma_of(P.to_perceptual(rgb, space, 100.0)))
            row[sname] = mx
        print(f'    {gname:9s} {row["ICtCp"]:9.4f} {row["Jzazbz"]:9.4f}   '
              f'[{P.chroma_reference(P.SPACE_ICTCP, gamut):.4f}, '
              f'{P.chroma_reference(P.SPACE_JZAZBZ, gamut):.4f}]')

    # Every gamut's luminance row has to agree with Rec.709's on the same
    # physical colour, or the achromatic axis is in the wrong place and every
    # gamut ray cast from it is tilted.
    worst = 0.0
    for gamut in (P.GAMUT_709, P.GAMUT_P3, P.GAMUT_2020):
        for rgb in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.6, 0.9], [1, 1, 1]):
            y709 = P.luminance(rgb)
            yg = P.display_luminance(P.to_display_gamut(rgb, gamut), gamut)
            worst = max(worst, abs(y709 - yg))
    print(f'\n  per-gamut luminance agreement, max error = {worst:.2e}   '
          f'{"OK" if worst < 1e-6 else "*** MISMATCH ***"}')

    # Perceptual round trip.
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

    # CAT02 round trip, and the property the whole spatial white point rests
    # on: a colour that is the local white must land exactly on the neutral
    # axis after adaptation.
    worst_rt, worst_neutral = 0.0, 0.0
    for white in ([1.0, 1.0, 1.0], [1.35, 1.0, 0.55], [0.72, 0.95, 1.4], [1.1, 1.0, 0.8]):
        gain = P.adaptation_gain(white)
        for rgb in ([0.2, 0.4, 0.6], [1, 1, 1], [0.9, 0.1, 0.1]):
            back = P.adapt_from_d65(P.adapt_to_d65(rgb, gain), gain)
            worst_rt = max(worst_rt, max(abs(a - c) for a, c in zip(rgb, back)))
        wy = P.luminance(white)
        adapted = P.adapt_to_d65([c / wy for c in white], gain)
        worst_neutral = max(worst_neutral, max(adapted) - min(adapted))
    # 1e-7 rather than machine epsilon: the adaptation matrices are baked to
    # nine digits, which is four orders of magnitude finer than an 8-bit code
    # value and is where the round trip should land.
    print(f'  CAT02 round trip, max abs error         = {worst_rt:.3e}   '
          f'{"OK" if worst_rt < 1e-7 else "*** LOSSY ***"}')
    print(f'  local white lands on the neutral axis   = {worst_neutral:.3e} channel spread   '
          f'{"OK" if worst_neutral < 1e-6 else "*** NOT NEUTRAL ***"}')
    print('    (the property v0.1\'s chroma offset did not have, and the reason chroma')
    print('     compression towards zero now actually converges on the local white)')


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

    # The shoulder clamp, which v0.1's shader did not have and its reference
    # did. It keeps a quarter stop of linear segment on each side of the anchor
    # whatever the slope, and it engages once a steep curve has squeezed the
    # highlight range down far enough for the shoulder to start under the mid.
    print('\n  the shoulder clamp, at settings steep enough for it to engage:')
    print(f'    {"midtone":>8s} {"rolloff":>8s} {"unclamped":>10s} {"clamped to":>11s}')
    for slope_opt, rolloff_opt in ((1.15, 0.45), (2.50, 0.45), (2.50, 0.95), (2.50, 0.05)):
        s2 = P.State(dict(midtoneContrast=slope_opt, highlightRolloff=rolloff_opt,
                          sceneAdaptive=False))
        mid_out_log = math.log2(s2.midOut)
        highlight_range = max(s2.peakLog - mid_out_log, 1.0) / s2.curve.slope
        raw = rolloff_opt * highlight_range
        print(f'    {slope_opt:8.2f} {rolloff_opt:8.2f} {raw:10.3f} {s2.curve.shoulderStart:11.3f}'
              f'   {"clamped" if abs(raw - s2.curve.shoulderStart) > 1e-6 else ""}')
    print('    Shader and reference now agree on it; in v0.1 only the reference had it, so')
    print('    the suite was reporting on a curve that did not run.')

    print('\n  scene stop -> display value')
    print('    rel.anchor    scene Y     display Y   display 8-bit sRGB   slope')
    for s in (-12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16):
        x = c.midIn + s
        y = 2 ** c.log(x)
        srgb = 1.055 * (y ** (1 / 2.4)) - 0.055 if y > 0.0031308 else 12.92 * y
        print(f'    {s:+7d}   {2 ** x:10.4g}   {y:9.5f}   {P.clamp(srgb, 0, 1) * 255:8.1f}'
              f'            {c.slope_at(x):.4f}')


def section_classify():
    rule('CLASSIFY  can it tell a lamp from a sunlit wall')

    print('  The question v0.1 could not answer. Its classifier was')
    print('      sourceness = smoothstep(threshold, log2(Y) - middleGrey)')
    print('  so anything bright was a source, and a sunlit white wall metered as a lamp.')
    print()
    print('  Each row is one pixel at 6 stops over middle grey - the same luminance for')
    print('  all of them - differing only in what the renderer says about it.')
    print()
    print(f'    {"case":34s} {"emissive":>9s} {"depth":>9s} {"blockfill":>10s}'
          f' {"body wt":>8s} {"source":>7s} {"sky":>5s}')

    y = 0.18 * 2 ** 6
    cases = [
        ('neon sign (emissive, compact)',   True,  10.0,   0.05),
        ('emissive screen (large area)',    True,  10.0,   1.00),
        ('sunlit white wall (large)',       False, 10.0,   1.00),
        ('specular glint (compact)',        False, 10.0,   0.05),
        ('open sky',                        False, 500001.0, 1.00),
        ('distant backdrop geometry',       False, 200000.0, 1.00),
    ]
    rows = {}
    for name, emissive, view_z, fill in cases:
        bw, src, sky, _, _, _ = P.classify([y, y, y], is_emissive=emissive,
                                           view_z=view_z, bright_fraction=fill)
        rows[name] = (bw, src, sky)
        print(f'    {name:34s} {str(emissive):>9s} {view_z:9.0f} {fill:10.2f}'
              f' {bw:8.3f} {src:7.3f} {sky:5.1f}')

    wall = rows['sunlit white wall (large)']
    neon = rows['neon sign (emissive, compact)']
    ok_wall = wall[0] > 0.6
    ok_neon = neon[1] > 0.9
    print()
    print(f'    sunlit wall stays scene content:  body weight {wall[0]:.3f}   '
          f'{"PASS" if ok_wall else "FAIL"}')
    print(f'    neon sign is fully a source:      sourceness  {neon[1]:.3f}   '
          f'{"PASS" if ok_neon else "FAIL"}')

    print('\n  the same six cases with rendererSignals off (v0.1 behaviour):')
    print(f'    {"case":34s} {"body wt":>8s} {"source":>7s} {"sky":>5s}')
    off = dict(rendererSignals=False)
    collapsed = []
    for name, emissive, view_z, fill in cases:
        bw, src, sky, _, _, _ = P.classify([y, y, y], is_emissive=emissive,
                                           view_z=view_z, bright_fraction=fill, opts=off)
        collapsed.append(round(src, 3))
        print(f'    {name:34s} {bw:8.3f} {src:7.3f} {sky:5.1f}')
    distinct = len(set(collapsed))
    print(f'\n    distinct classifications without the gbuffer: {distinct} of {len(cases)}')
    print('    (compactness still separates a glint from a wall, which is why it is kept as')
    print('     the fallback - but nothing separates a lamp from a wall, or sky from either)')


def section_illuminant():
    rule('ILLUMINANT  the colour of the light, not the colour of the paint')

    print('  A Titanfall interior: orange-painted panels, rust, a bit of grey deck. The')
    print('  materials are strongly biased, the light is not. A grey-world estimate over')
    print('  radiance - which is what v0.1 used as its white point - cannot tell those two')
    print('  apart and reports the paint. Dividing by albedo can.')
    print()

    walls = [('orange panel', (0.52, 0.26, 0.09)),
             ('rust',         (0.40, 0.17, 0.08)),
             ('ochre trim',   (0.55, 0.38, 0.12)),
             ('grey deck',    (0.30, 0.30, 0.31))]

    for lname, light in (('neutral D65', (1.00, 1.00, 1.00)),
                         ('tungsten',    (1.35, 1.00, 0.55))):
        mean_radiance = [0.0, 0.0, 0.0]
        illum_acc = [0.0, 0.0, 0.0]
        illum_w = 0.0
        for _, albedo in walls:
            radiance = [albedo[i] * light[i] * 0.6 for i in range(3)]
            mean_radiance = [mean_radiance[i] + radiance[i] / len(walls) for i in range(3)]
            _, _, _, _, illum, w = P.classify(radiance, albedo=albedo)
            if illum is not None:
                illum_acc = [illum_acc[i] + illum[i] * w for i in range(3)]
                illum_w += w

        gw = [c / max(P.luminance(mean_radiance), 1e-6) for c in mean_radiance]
        est = [c / illum_w for c in illum_acc] if illum_w > 1e-4 else [1.0, 1.0, 1.0]
        est = [c / max(P.luminance(est), 1e-6) for c in est]
        truth = [c / max(P.luminance(light), 1e-6) for c in light]

        gw_err = max(abs(gw[i] - truth[i]) for i in range(3))
        est_err = max(abs(est[i] - truth[i]) for i in range(3))
        print(f'    under {lname}:')
        print(f'      true illuminant       ({truth[0]:.3f}, {truth[1]:.3f}, {truth[2]:.3f})')
        print(f'      grey world (v0.1)     ({gw[0]:.3f}, {gw[1]:.3f}, {gw[2]:.3f})   '
              f'max error {gw_err:.3f}')
        print(f'      radiance / albedo     ({est[0]:.3f}, {est[1]:.3f}, {est[2]:.3f})   '
              f'max error {est_err:.3f}   '
              f'{"BETTER" if est_err < gw_err else "no better"}')

    # The guarantee the adaptation buys: a surface neutral under the local
    # light stays inside the display volume however hard chroma is compressed.
    print('\n  in-gamut guarantee under a non-D65 white point:')
    for wname, white in (('D65', (1.0, 1.0, 1.0)),
                         ('tungsten', (1.35, 1.0, 0.55)),
                         ('cool', (0.72, 0.95, 1.4))):
        st = P.State(dict(spatialWhite=1.0), illuminant=white, illum_confidence=1.0)
        # A surface that is neutral under this illuminant.
        wy = P.luminance(white)
        surface = [0.18 * c / wy for c in white]
        out, dbg = P.psdt_apply(surface, st)
        demand = dbg['demand']
        print(f'    {wname:9s} local-neutral surface -> demand {demand:.4f}   '
              f'{"IN GAMUT" if demand <= 1.0001 else "*** OUT OF GAMUT ***"}')


def section_gamut():
    rule('GAMUT  what the per-gamut geometry fixed')

    print('  v0.1 located the achromatic axis with Rec.709 luminance weights in every')
    print('  gamut. In Rec.709 that is correct; in P3 and Rec.2020 it puts the grey point')
    print('  somewhere that is not grey, and every gamut ray is cast from there.')
    print()
    print('  A neutral cannot show it: every gamut\'s luminance row sums to 1, so R == G == B')
    print('  weights to the same grey either way. It takes a chromatic colour, which is')
    print('  exactly the population the colour-volume stage exists for.')
    print()
    print(f'    {"gamut":9s} {"colour":>22s} {"correct Y":>10s} {"Rec.709 Y":>10s}'
          f' {"demand err":>11s} {"headroom err":>13s}')

    worst_demand, worst_room = 0.0, 0.0
    probes = ([0.90, 0.20, 0.10], [0.15, 0.80, 0.25], [0.20, 0.30, 0.95], [0.85, 0.80, 0.15])
    for gname, gamut in (('Rec709', P.GAMUT_709), ('P3', P.GAMUT_P3), ('Rec2020', P.GAMUT_2020)):
        for rgb in probes:
            correct = P.gamut_demand(rgb, 1.0, gamut)

            # What v0.1 computed: the same ray, cast from a Rec.709-weighted grey.
            grey_wrong = P.clamp(P.luminance(rgb), 0.0, 1.0)
            d = [c - grey_wrong for c in rgb]
            t = 1e6
            for i in range(3):
                if d[i] > 1e-6:
                    t = min(t, (1.0 - grey_wrong) / d[i])
                elif d[i] < -1e-6:
                    t = min(t, -grey_wrong / d[i])
            wrong = 1.0 / max(t, 1e-6)

            grey_right = P.display_luminance(rgb, gamut)
            room_err = abs(grey_right - grey_wrong)
            worst_demand = max(worst_demand, abs(correct - wrong))
            worst_room = max(worst_room, room_err)
            if rgb is probes[0] or gamut != P.GAMUT_709:
                print(f'    {gname:9s} ({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})'
                      f'{"":>4s} {correct:10.4f} {wrong:10.4f} {abs(correct - wrong):11.4f}'
                      f' {room_err:13.4f}')

    print(f'\n  Rec.709 is exact by construction, which is why the bug was invisible on the')
    print(f'  default path. Elsewhere: largest demand error {worst_demand:.4f}, largest')
    print(f'  achromatic-point error {worst_room:.4f} of full scale. The second one is the')
    print(f'  one that matters - it is how far off the "preserve brightness, give up chroma"')
    print(f'  scale was aiming, on every out-of-gamut colour in the frame.')

    # And the chroma normaliser, which was also frozen at the Rec.709 value.
    print('\n  chroma normaliser, as the pressure metric sees it:')
    for gname, gamut in (('Rec709', P.GAMUT_709), ('P3', P.GAMUT_P3), ('Rec2020', P.GAMUT_2020)):
        correct = P.chroma_reference(P.SPACE_ICTCP, gamut)
        v01 = P.chroma_reference(P.SPACE_ICTCP, P.GAMUT_709)
        print(f'    {gname:9s} correct {correct:.4f}   v0.1 used {v01:.4f}   '
              f'chroma term off by {100.0 * (v01 / correct - 1.0):+.0f}%')


def section_gt7space():
    rule('GT7SPACE  what GT7\'s Rec.2020 input assumption costs')

    print('  gt7.slangh says "input is assumed to be linear Rec.2020 frame-buffer RGB",')
    print('  and feeds the framebuffer straight into a Rec.2020 -> LMS matrix. This')
    print('  renderer\'s framebuffer is linear Rec.709 - observably, since the histogram,')
    print('  the auto exposure and the colour grading all weight it with Rec.709 luma and')
    print('  the apply shader ends with the sRGB OETF and no primary conversion.')
    print()
    print('  The forward and inverse inside GT7 use the same (wrong) matrices, so an')
    print('  untouched colour round-trips exactly and the error is invisible on a grey')
    print('  ramp. It enters only where the ICtCp values are modified - the chroma fade')
    print('  and the luminance substitution - which is to say, exactly where the operator')
    print('  does its work.')
    print()
    print('  shipped   = framebuffer fed straight in, as the branch runs today')
    print('  reference = Rec.709 -> Rec.2020 in, operator, Rec.2020 -> Rec.709 out')
    print()
    print(f'    {"swatch":9s} {"scene Y":>8s} {"Y shipped":>10s} {"Y ref":>9s} {"dY":>8s}'
          f' {"C shipped":>10s} {"C ref":>9s} {"dHue":>9s}')

    floor = 0.1 * P.chroma_reference(P.SPACE_ICTCP, P.GAMUT_709)
    worst_dy = worst_dc = worst_dh = 0.0
    worst_dy_at = worst_dh_at = ''
    for sw in ('grey', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
               'skin', 'neon', 'muzzle', 'gunmetal'):
        for y in (0.18, 1.0, 4.0):
            rgb = hdr(sw, y)
            a, b = P.gt7(rgb), P.gt7_in_rec2020(rgb)
            ya, yb = max(P.luminance(a), 1e-9), max(P.luminance(b), 1e-9)
            ca = P.chroma_of(P.rec709_to_ictcp(a, 100.0))
            cb = P.chroma_of(P.rec709_to_ictcp(b, 100.0))
            dy = 100.0 * (yb / ya - 1.0)
            d = P.hue_of(P.rec709_to_ictcp(b, 100.0)) - P.hue_of(P.rec709_to_ictcp(a, 100.0))
            d -= 2 * math.pi * math.floor(d / (2 * math.pi) + 0.5)
            deg = math.degrees(d)

            if abs(dy) > worst_dy:
                worst_dy, worst_dy_at = abs(dy), f'{sw} at {y:g}x'
            worst_dc = max(worst_dc, abs(cb - ca))
            # Hue is undefined at zero chroma, so only count samples where both
            # results still carry something a viewer could see a hue in.
            if min(ca, cb) > floor and abs(deg) > worst_dh:
                worst_dh, worst_dh_at = abs(deg), f'{sw} at {y:g}x'

            if y == 1.0:
                print(f'    {sw:9s} {y:8.2f} {ya:10.4f} {yb:9.4f} {dy:+7.1f}%'
                      f' {ca:10.4f} {cb:9.4f} {deg:+8.2f}d')

    print()
    print(f'  worst over 11 swatches x 3 luminances:')
    print(f'    luminance  {worst_dy:6.1f}%   ({worst_dy_at})')
    print(f'    chroma     {worst_dc:6.4f}')
    print(f'    hue        {worst_dh:6.2f} deg  ({worst_dh_at}), counting only samples')
    print(f'               above a tenth of the display\'s chroma capacity - below that')
    print(f'               hue is undefined and the number is meaningless')
    print()
    print('  Verdict: a neutral is unaffected, so this never shows on a grey ramp, and it')
    print('  is not a subtle bias either - it is tens of percent of luminance on exactly')
    print('  the saturated colours a colour-volume comparison is about. As wired today the')
    print('  GT7 branch is the GT7 algorithm applied to relabelled numbers, not the GT7')
    print('  reference applied to this renderer\'s colours.')
    print()
    print('  This section deliberately does not change gt7.slangh. Correcting it would')
    print('  change how the operator looks for anyone using it, which is a decision rather')
    print('  than a bug fix, and an edited control is not a control. What it does is put a')
    print('  number on the question so the decision can be made with one.')


def section_response():
    rule('RESPONSE  luminance transfer, PSDT vs the controls')

    st = P.State()
    ops = ops_for(st)
    print('    scene Y  ' + ''.join(f'{n:>10s}' for n, _ in ops) + '   (display luminance, grey)')
    for y in LADDER:
        rgb = [y, y, y]
        row = f'  {y:10.4g}'
        for _, fn in ops:
            row += f'{P.luminance(fn(rgb)):10.5f}'
        print(row)

    print('\n  clipping: fraction of the ladder x swatch grid that clips a channel at 1.0')
    for name, fn in ops:
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

    # The context term: the same colour, in a neighbourhood of its own
    # brightness versus as an outlier against a dark one.
    print('\n  the context term, which v0.1\'s blend function did not have:')
    print('    the same neon at the same luminance, in two different neighbourhoods')
    for label, pooled in (('large field of its own brightness', math.log2(P.luminance(hdr('neon', 8.0)))),
                          ('bright outlier against a dark room', st.anchor - 3.0)):
        out, dbg = P.psdt_apply(hdr('neon', 8.0), st, pooled_log=pooled)
        print(f'    {label:36s} context={dbg["context"]:.3f} knee={dbg["knee"]:.3f} '
              f'-> compress={dbg["compress"]:.3f}')


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

    # Fine sweep: 1/42-stop steps through the shoulder, looking for a jump.
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

    print('\n  the detail-magnitude rolloff, which v0.1 documented and did not implement:')
    print('    restoration as a function of how far the pixel is from its neighbourhood')
    print(f'    {"detail (stops)":>15s} {"boost":>8s}   what it is')
    base = st.anchor + 1.0
    for d, what in ((0.25, 'surface texture'), (1.0, 'a shadow edge'),
                    (3.0, 'a bright object'), (6.0, 'a lamp against a wall')):
        _, dbg = P.psdt_apply([2 ** (base + d)] * 3, st, pooled_log=base)
        print(f'    {d:15.2f} {dbg["boost"]:8.4f}   {what}')
    print('    a lamp six stops from the wall behind it is not detail, it is a different')
    print('    object, and restoring the contrast between them is a halo rather than texture.')


def source_levels_for(rgb, area_pixels, levels=6):
    """
    What the source field reports at the centre of a compact source, per level.

    The analysis pass stores the mean source energy over each block, so a
    source covering `area_pixels` contributes L * area / (16 * 2^k)^2 at level
    k. That 1/4^k is the whole mechanism behind the glare radius: a dim source
    falls under the visibility threshold after a level or two while a bright
    one survives several more.
    """
    out = []
    for k in range(levels):
        block = (16 * (2 ** k)) ** 2
        scale = min(area_pixels / block, 1.0)
        out.append([c * scale for c in rgb])
    return out


def section_glare():
    rule('GLARE  radius from luminance, and whose colour the halo is')

    st = P.State()
    print('  A compact source of 256 pixels, at increasing luminance. "levels lit" is how')
    print('  many pyramid levels stayed above the visibility threshold, which is the halo')
    print('  radius: level k covers 16 * 2^k pixels.')
    print()
    print(f'    {"source Y":>10s} {"levels lit":>11s} {"radius (px)":>12s} {"glare Y":>10s}')
    prev_levels = 0
    monotone = True
    for mult in (1, 4, 16, 64, 256, 1024, 4096, 16384):
        src = [c * mult for c in norm(SWATCHES['grey'])]
        levels = source_levels_for(src, 256.0, st.levelCount)
        threshold = 2.0 ** st.glareThreshold
        lit = sum(1 for lv in levels if P.luminance(lv) > threshold)
        g = P.glare(st, levels)
        radius = 16 * (2 ** max(lit - 1, 0)) if lit else 0
        if lit < prev_levels:
            monotone = False
        prev_levels = lit
        print(f'    {mult:10.0f} {lit:11d} {radius:12d} {P.luminance(g.colour):10.5f}')
    print(f'\n    radius grows monotonically with source luminance: '
          f'{"PASS" if monotone else "FAIL"}')

    print('\n  the halo\'s colour is the source\'s, not the scene\'s.')
    print('  v0.1 tinted from the low-frequency scene chroma, so a white light near a red')
    print('  wall glared red. The source field now carries the source\'s own RGB.')
    print()
    print(f'    {"source":10s} {"source chromaticity":>26s} {"glare chromaticity":>26s}'
          f' {"hue err":>9s}')
    for sw in ('grey', 'red', 'neon', 'muzzle'):
        src = [c * 4096.0 for c in norm(SWATCHES[sw])]
        g = P.glare(st, source_levels_for(src, 256.0, st.levelCount))
        # Normalised so the largest channel is 1, which is the readable form -
        # normalising to luminance puts a saturated red's red channel at 4.1.
        sc = [c / max(src) for c in src]
        gc = [c / max(max(g.colour), 1e-6) for c in g.colour]
        # ICtCp is an absolute space, so its hue is not scale-invariant: a
        # 4096-nit source and a dim halo of the same chromaticity do not report
        # the same hue. Comparing at matched luminance isolates what the
        # desaturation did from what PQ did.
        gy = max(P.luminance(g.colour), 1e-9)
        matched = [c * gy / max(P.luminance(src), 1e-9) for c in src]
        hs = P.hue_of(P.rec709_to_ictcp(matched, 100.0))
        hg = P.hue_of(P.rec709_to_ictcp([max(c, 1e-9) for c in g.colour], 100.0))
        d = hg - hs
        d -= 2 * math.pi * math.floor(d / (2 * math.pi) + 0.5)
        print(f'    {sw:10s} ({sc[0]:6.3f},{sc[1]:6.3f},{sc[2]:6.3f})'
              f'      ({gc[0]:6.3f},{gc[1]:6.3f},{gc[2]:6.3f})'
              f' {math.degrees(d):8.2f}d')
    print('    (pulled 40% towards white on purpose - ocular scatter is broader-band than')
    print('     the light causing it - but the source\'s hue survives, which is the point)')

    print('\n  the two lobes. Near field is the core a light has before the veil begins.')
    src = [c * 4096.0 for c in norm(SWATCHES['grey'])]
    for nf in (0.0, 0.4, 0.8):
        stn = P.State(dict(glareNearField=nf))
        g = P.glare(stn, source_levels_for(src, 256.0, stn.levelCount))
        print(f'    nearField {nf:.1f} -> glare luminance {P.luminance(g.colour):.5f}')

    print('\n  and the threshold is relative to adaptation, which is why a headlight is')
    print('  blinding at night and invisible at noon:')
    src = [c * 64.0 for c in norm(SWATCHES['grey'])]
    for label, anchor in (('night  (anchor -4 stops)', P.MIDDLE_GREY_LOG - 4.0),
                          ('indoor (anchor  0 stops)', P.MIDDLE_GREY_LOG),
                          ('noon   (anchor +4 stops)', P.MIDDLE_GREY_LOG + 4.0)):
        sta = P.State(anchor=anchor)
        g = P.glare(sta, source_levels_for(src, 256.0, sta.levelCount))
        print(f'    {label}  glare luminance {P.luminance(g.colour):.5f}')


def section_temporal():
    rule('TEMPORAL  step response of two filters in series')

    print('  PSDT has two temporal filters: the adaptation field accumulates with motion')
    print('  reprojection, and the anchor accumulates again on top of it. That is good for')
    print('  stability and it is also how an image ends up still showing yesterday\'s')
    print('  exposure. v0.1 asserted this was fine; this measures it.')
    print()

    def step_response(from_stops, to_stops, camera_angular=0.0, camera_cut=False,
                      ramp_seconds=0.0, dt=1.0 / 60.0, seconds=6.0):
        field = P.FieldFilter(speed=8.0, value=P.MIDDLE_GREY_LOG + from_stops)
        field.hasHistory = True
        state = P.StateFilter(anchor=P.MIDDLE_GREY_LOG + from_stops)
        state.hasHistory = True

        trace = []
        cuts = 0
        n = int(seconds / dt)
        for i in range(n):
            t = i * dt
            if ramp_seconds > 0.0:
                k = min(t / ramp_seconds, 1.0)
                scene = from_stops + (to_stops - from_stops) * k
            else:
                scene = to_stops
            target = P.MIDDLE_GREY_LOG + scene
            before = state.cutFlag
            measured = field.step(target, dt, before)
            anchor = state.step(measured, dt,
                                camera_angular=camera_angular,
                                camera_cut=camera_cut and i == 0)
            if state.cutFlag >= 1.0 and before < 1.0:
                cuts += 1
            trace.append((t, measured, anchor))
        return trace, cuts

    def settle(trace, index, start, end, fraction):
        goal = start + (end - start) * fraction
        for t, m, a in trace:
            v = (m, a)[index]
            if (end > start and v >= goal) or (end < start and v <= goal):
                return t
        return float('nan')

    def report(label, a, b, **kwargs):
        trace, cuts = step_response(a, b, **kwargs)
        start_a = P.MIDDLE_GREY_LOG + a
        end_a = P.MIDDLE_GREY_LOG + b
        a10 = settle(trace, 1, start_a, end_a, 0.10)
        a50 = settle(trace, 1, start_a, end_a, 0.50)
        a90 = settle(trace, 1, start_a, end_a, 0.90)
        final = trace[-1][2]
        overshoot = 0.0
        for _, _, av in trace:
            overshoot = max(overshoot, (av - end_a) if b > a else (end_a - av))
        print(f'    {label}')
        print(f'      anchor 10/50/90%   {a10:5.3f} / {a50:5.3f} / {a90:5.3f} s'
              f'   overshoot {overshoot:+.4f} st   residual {final - end_a:+.4f} st'
              f'   cuts fired {cuts}')

    print('  Within the cut detector: a 2-stop change, which is a real lighting change and')
    print('  not a scene change. This is the transform\'s actual time constant.')
    report('dark -> bright, +2 stops', 0.0, 2.0)
    report('bright -> dark, -2 stops', 2.0, 0.0)
    report('dark -> bright, +2 stops, mid fast turn', 0.0, 2.0, camera_angular=8.0)

    print()
    print('  A 6-stop change spread over half a second - walking through a doorway. Each')
    print('  frame\'s step is small, so the cut detector correctly stays out of it.')
    report('doorway, +6 stops over 0.5 s', 0.0, 6.0, ramp_seconds=0.5)
    report('doorway, -6 stops over 0.5 s', 6.0, 0.0, ramp_seconds=0.5)

    print()
    print('  And 6 stops in a single frame, which is not a lighting change - nothing in a')
    print('  physical scene does that. Both detectors should fire and the state should')
    print('  re-anchor rather than easing through several seconds of wash-out.')
    report('instant +6 stops (luminance cut)', 0.0, 6.0)
    report('instant +6 stops (camera reports a cut)', 0.0, 6.0, camera_cut=True)

    print()
    print('    Up is faster than down by design - light adaptation is fast, dark adaptation')
    print('    is slow - and that asymmetry is what makes an interior read as dark when you')
    print('    step into it. A fast turn triples the rate, so sweeping your view does not')
    print('    drag seconds of wash-out behind it. A cut re-anchors outright.')
    print()
    print('    Note the field\'s own +-3 stop history clamp, which is doing real work here:')
    print('    it bounds how far behind the measurement the field can ever be, so even the')
    print('    un-cut path cannot lag more than three stops however slow the anchor is.')

    # Frame-to-frame stability under a noisy measurement: the reason the two
    # filters exist at all.
    print('\n  stability: a measurement with +-0.5 stops of frame-to-frame noise')
    import random
    random.seed(20260905)
    field = P.FieldFilter(speed=8.0, value=P.MIDDLE_GREY_LOG)
    field.hasHistory = True
    state = P.StateFilter(anchor=P.MIDDLE_GREY_LOG)
    state.hasHistory = True
    raw_dev, out_dev = 0.0, 0.0
    prev_raw = prev_out = None
    for _ in range(600):
        noisy = P.MIDDLE_GREY_LOG + random.uniform(-0.5, 0.5)
        anchor = state.step(field.step(noisy, 1 / 60.0, state.cutFlag), 1 / 60.0)
        if prev_raw is not None:
            raw_dev = max(raw_dev, abs(noisy - prev_raw))
            out_dev = max(out_dev, abs(anchor - prev_out))
        prev_raw, prev_out = noisy, anchor
    print(f'    largest frame-to-frame step, input  {raw_dev:.4f} stops')
    print(f'    largest frame-to-frame step, anchor {out_dev:.4f} stops   '
          f'({raw_dev / max(out_dev, 1e-6):.0f}x attenuation)')


def section_compare():
    rule('COMPARE  the section 33 score, PSDT against the controls')

    st = P.State()

    def metrics(fn, stateful=False):
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
        floor = 0.1 * P.chroma_reference(P.SPACE_ICTCP, P.GAMUT_709)
        worst_hue, hue_at = 0.0, 0.0
        for sw in ('red', 'magenta', 'yellow', 'green', 'blue'):
            prev = None
            for i in range(513):
                y = 2 ** (-2.0 + 12.0 * i / 512)
                rgb = hdr(sw, y)
                out = fn(rgb)
                oy = max(P.luminance(out), 1e-8)
                ref = [c * oy / max(P.luminance(rgb), 1e-8) for c in rgb]
                if P.chroma_of(P.rec709_to_ictcp(out, 100.0)) < floor:
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
        # Highlight identity: of the colour the display could still show at the
        # luminance the operator chose, how much did it keep. Boundary
        # utilisation rather than raw chroma, and restricted to the range where
        # a bright emissive actually sits - one to four stops past the shoulder.
        #
        # Raw chroma is the wrong measure here and measuring it that way is
        # actively misleading: at 256x middle grey the correct answer really is
        # white, because a display cannot be both at peak luminance and
        # coloured, so raw chroma rewards whichever operator refuses to reach
        # peak. Boundary use asks the question that has an answer - did it get
        # as close to the wall as the wall allows.
        highlight_use, hn = 0.0, 0
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
                if 4 <= y <= 64 and sw not in ('grey', 'gunmetal'):
                    highlight_use += 1.0 / P.gamut_limit(out, 1.0)
                    hn += 1
        highlight_use /= max(hn, 1)

        return dict(chroma=chroma_keep, use=use, hue=worst_hue, hue_at=hue_at, detail=detail,
                    clip=100.0 * clipped / total, inverted=inverted,
                    highlight=highlight_use, stateful=stateful)

    ops = [('PSDT', lambda c: P.psdt_apply(c, st)[0], True),
           ('GT7', P.gt7, False), ('AgX', P.agx, False),
           ('Reinhard', P.reinhard, False), ('Hable', P.hable, False)]

    print(f'    {"operator":10s} {"chroma kept":>12s} {"boundary use":>13s} {"max hue step":>13s} '
          f'{"midtone contr":>14s} {"hilite bnd use":>15s} {"clipped":>9s} {"Y inv":>7s}')
    rows = {}
    for name, fn, stateful in ops:
        m = metrics(fn, stateful)
        rows[name] = m
        print(f'    {name:10s} {m["chroma"]:12.3f} {m["use"]:13.3f} {m["hue"]:12.3f}d '
              f'{m["detail"]:14.3f} {m["highlight"]:15.3f} {m["clip"]:8.1f}% {m["inverted"]:7d}')
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
    print('    hilite bnd use  boundary utilisation restricted to 1-4 stops past the shoulder, which')
    print('                  is where a bright emissive actually sits. Raw chroma is the wrong measure')
    print('                  up there: at 256x middle grey the correct answer really is white, so raw')
    print('                  chroma just rewards whichever operator refuses to reach peak luminance.')
    print('    Y inversions  times a brighter scene value produced a darker display value.')
    print('                  Anything but 0 breaks the most basic perceptual invariant there is.')

    # --- the weighted score ------------------------------------------------
    # Section 33 asked for S = wL*SL + wC*SC + wH*SH + wD*SD + wT*ST. Here it
    # is, with every component normalised to [0,1] and every weight written
    # down. The weights are a choice and cannot be anything else - "which of
    # hue smoothness and midtone punch matters more" has no measurement behind
    # it - so the point of the score is that the choice is explicit and can be
    # argued with, not that it is objective.
    W = dict(luminance=0.25, colour=0.25, hue=0.20, detail=0.15, temporal=0.15)

    # Temporal stability, measured rather than assumed. A pure function of one
    # pixel scores 1 by construction: it has no state to be unstable. That is
    # not a compliment, it is the definition - it also cannot adapt.
    st_field = P.FieldFilter(speed=8.0, value=P.MIDDLE_GREY_LOG)
    st_field.hasHistory = True
    st_state = P.StateFilter(anchor=P.MIDDLE_GREY_LOG)
    st_state.hasHistory = True
    import random
    random.seed(20260905)
    prev = None
    worst_step = 0.0
    for _ in range(600):
        noisy = P.MIDDLE_GREY_LOG + random.uniform(-0.5, 0.5)
        anchor = st_state.step(st_field.step(noisy, 1 / 60.0, st_state.cutFlag), 1 / 60.0)
        if prev is not None:
            worst_step = max(worst_step, abs(anchor - prev))
        prev = anchor
    # A tenth of a stop between frames is the point at which exposure movement
    # becomes visible as movement rather than as adaptation.
    psdt_temporal = math.exp(-worst_step / 0.1)

    def score(m):
        s_l = 1.0 - m['clip'] / 100.0
        if m['inverted'] > 0:
            s_l = 0.0
        s_c = min(m['use'], 1.0)
        s_h = math.exp(-m['hue'] / 1.0)
        s_d = P.saturate((m['detail'] - 0.8) / 0.5)
        s_t = psdt_temporal if m['stateful'] else 1.0
        total = (W['luminance'] * s_l + W['colour'] * s_c + W['hue'] * s_h
                 + W['detail'] * s_d + W['temporal'] * s_t)
        return s_l, s_c, s_h, s_d, s_t, total

    print()
    print(f'  weighted score  S = {W["luminance"]}*SL + {W["colour"]}*SC + {W["hue"]}*SH'
          f' + {W["detail"]}*SD + {W["temporal"]}*ST')
    print(f'    {"operator":10s} {"SL":>7s} {"SC":>7s} {"SH":>7s} {"SD":>7s} {"ST":>7s} {"S":>8s}')
    for name, _, _ in ops:
        s_l, s_c, s_h, s_d, s_t, total = score(rows[name])
        print(f'    {name:10s} {s_l:7.3f} {s_c:7.3f} {s_h:7.3f} {s_d:7.3f} {s_t:7.3f} {total:8.3f}')
    print()
    print('    SL  luminance fidelity: fraction of the grid that did not clip. Zero if the')
    print('        operator ever inverted brightness ordering, because that is disqualifying')
    print('        rather than merely costly.')
    print('    SC  colour-volume fidelity: mean boundary utilisation, capped at 1.')
    print('    SH  hue trajectory smoothness: exp(-maxstep / 1 degree).')
    print('    SD  detail: midtone slope mapped from 0.8 (flat) to 1.3.')
    print('    ST  temporal stability: exp(-worst frame-to-frame anchor step / 0.1 stops).')
    print('        A pure function of one pixel scores 1 by construction - it has no state to')
    print('        be unstable, which is also why it cannot adapt to anything.')
    print()
    print('    Psycho17 is absent. It is a 754-line vendored operator and a hand port could')
    print('    not be verified against the shader that runs; an unverified control is worse')
    print('    than a missing one, because its numbers would look just as authoritative.')
    print('    Comparing against it needs a GPU capture, not this harness.')


def section_invariants():
    rule('INVARIANTS  the eight properties the transform is supposed to hold')

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

    # 5. Temporal stability. v0.1 tested this by calling the same pure function
    #    twice and comparing, which cannot fail and therefore measured nothing.
    #    The actual invariant is the one that was asked for: a colour must not
    #    change because the measurement moved slightly between frames. So the
    #    state is driven with a noisy measurement and the *output colour* is
    #    watched, not the state.
    import random
    random.seed(4242)
    field = P.FieldFilter(speed=8.0, value=P.MIDDLE_GREY_LOG)
    field.hasHistory = True
    filt = P.StateFilter(anchor=P.MIDDLE_GREY_LOG)
    filt.hasHistory = True
    probe = hdr('neon', 4.0)
    prev_rgb = None
    worst_step = 0.0
    for _ in range(300):
        noisy = P.MIDDLE_GREY_LOG + random.uniform(-0.5, 0.5)
        anchor = filt.step(field.step(noisy, 1 / 60.0, filt.cutFlag), 1 / 60.0)
        frame_state = P.State(anchor=anchor)
        out, _ = P.psdt_apply(probe, frame_state)
        if prev_rgb is not None:
            worst_step = max(worst_step, max(abs(a - b) for a, b in zip(out, prev_rgb)))
        prev_rgb = out
    # One 8-bit code value is 1/255. Movement under that cannot be seen.
    ok = worst_step < 1.0 / 255.0
    fails += [] if ok else ['temporal stability']
    print(f'  5. colour stable under a noisy measurement  worst frame-to-frame step '
          f'{worst_step:.5f} ({worst_step * 255:.2f} code values)   {"PASS" if ok else "FAIL"}')

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

    # 8. Spatial stability, which the depth-aware pooling is for: a bright
    #    object must not change the exposure of the surface next to it when the
    #    two are at different depths.
    near_surface = (st.anchor, 1.0, st.anchor, math.log2(1.0 + 5.0))
    sky_behind = (st.anchor + 6.0, 1.0, st.anchor + 6.0, math.log2(1.0 + 200000.0))
    same_depth = (st.anchor + 6.0, 1.0, st.anchor + 6.0, math.log2(1.0 + 5.0))

    a_iso = P.sample_adaptation(st, {'S': near_surface, 'M': near_surface, 'L': near_surface})
    a_sky = P.sample_adaptation(st, {'S': near_surface, 'M': sky_behind, 'L': sky_behind})
    a_same = P.sample_adaptation(st, {'S': near_surface, 'M': same_depth, 'L': same_depth})
    pull_sky = abs(a_sky.adaptedLog - a_iso.adaptedLog)
    pull_same = abs(a_same.adaptedLog - a_iso.adaptedLog)
    ok = pull_sky < 0.02 and pull_same > 4.0 * pull_sky
    fails += [] if ok else ['depth-aware pooling']
    print(f'  8. a wall is not re-exposed by the sky      {pull_sky:.4f} stops across a depth edge, '
          f'{pull_same:.4f} stops without one   {"PASS" if ok else "FAIL"}')
    print('     behind it')

    print()
    if fails:
        print(f'  {len(fails)} FAILED: {", ".join(fails)}')
    else:
        print('  all invariants hold')
    return not fails


SECTIONS = {
    'probe': section_probe, 'curve': section_curve, 'classify': section_classify,
    'gt7space': section_gt7space,
    'illuminant': section_illuminant, 'gamut': section_gamut,
    'response': section_response, 'colour': section_colour, 'hue': section_hue,
    'contrast': section_contrast, 'glare': section_glare, 'temporal': section_temporal,
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
