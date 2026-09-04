"""
CPU reference implementation of PSDT, and of the operators it is measured
against.

This is a line-for-line port of the shaders under
src/dxvk/shaders/rtx/pass/psdt/, kept deliberately literal rather than
idiomatic so that a divergence between this and the GPU is a visible diff and
not a judgement call. Its job is to answer questions that cannot be answered by
looking at a screenshot:

  * is the curve actually monotonic and C1 everywhere, or only where it was
    sampled;
  * what does the transform do to chroma and hue as a colour runs out of
    display volume, in numbers;
  * how much of the input's local contrast survives;
  * and how all of that compares against GT7, Reinhard and Hable on the same
    inputs.

Pure standard library on purpose - it has to run anywhere the repo does.
"""

import math

# ---------------------------------------------------------------------------
# Small matrix helpers
# ---------------------------------------------------------------------------

def mul(M, v):
    return [sum(M[i][k] * v[k] for k in range(3)) for i in range(3)]

def clamp(x, a, b):
    return max(a, min(b, x))

def saturate(x):
    return clamp(x, 0.0, 1.0)

def lerp(a, b, t):
    return a + (b - a) * t

def smoothstep(e0, e1, x):
    t = saturate((x - e0) / (e1 - e0)) if e1 != e0 else (0.0 if x < e0 else 1.0)
    return t * t * (3.0 - 2.0 * t)

# ---------------------------------------------------------------------------
# psdt_perceptual_space.slangh
# ---------------------------------------------------------------------------

LUMA_709 = (0.2126, 0.7152, 0.0722)

REC709_TO_XYZ = [[0.4123907993, 0.3575843394, 0.1804807884],
                 [0.2126390059, 0.7151686788, 0.0721923154],
                 [0.0193308187, 0.1191947798, 0.9505321522]]
XYZ_TO_REC709 = [[3.2409699419, -1.5373831776, -0.4986107603],
                 [-0.9692436363, 1.8759675015, 0.0415550574],
                 [0.0556300797, -0.2039769589, 1.0569715142]]
REC709_TO_REC2020 = [[0.6274039, 0.3292830, 0.0433131],
                     [0.0690973, 0.9195404, 0.0113623],
                     [0.0163914, 0.0880132, 0.8955953]]
REC2020_TO_REC709 = [[1.6604910, -0.5876411, -0.0728499],
                     [-0.1245505, 1.1328999, -0.0083494],
                     [-0.0181508, -0.1005789, 1.1187297]]
REC709_TO_P3 = [[0.8224621, 0.1775380, 0.0],
                [0.0331941, 0.9668058, 0.0],
                [0.0170827, 0.0723974, 0.9105199]]
P3_TO_REC709 = [[1.2249401, -0.2249404, 0.0],
                [-0.0420569, 1.0420571, 0.0],
                [-0.0196376, -0.0786361, 1.0982735]]

REC2020_TO_LMS = [[1688/4096, 2146/4096, 262/4096],
                  [683/4096, 2951/4096, 462/4096],
                  [99/4096, 309/4096, 3688/4096]]
LMS_TO_ICTCP = [[2048/4096, 2048/4096, 0.0],
                [6610/4096, -13613/4096, 7003/4096],
                [17933/4096, -17390/4096, -543/4096]]
ICTCP_TO_LMS = [[1.0, 0.0086090370, 0.1110296250],
                [1.0, -0.0086090370, -0.1110296250],
                [1.0, 0.5600313357, -0.3206271750]]
LMS_TO_REC2020 = [[3.4366066943, -2.5064521187, 0.0698454243],
                  [-0.7913295556, 1.9836004518, -0.1922708962],
                  [-0.0259498997, -0.0989137147, 1.1248636144]]

XYZ_TO_JZLMS = [[0.41478972, 0.579999, 0.0146480],
                [-0.20151000, 1.120649, 0.0531008],
                [-0.01660080, 0.264800, 0.6684799]]
JZLMS_TO_XYZ = [[1.9242264358, -1.0047923126, 0.0376514040],
                [0.3503167621, 0.7264811939, -0.0653844229],
                [-0.0909828110, -0.3127282905, 1.5227665613]]
JZLMS_TO_IAB = [[0.5, 0.5, 0.0],
                [3.524, -4.066708, 0.542708],
                [0.199076, 1.096799, -1.295875]]
JZ_IAB_TO_LMS = [[1.0, 0.1386050433, 0.0580473162],
                 [1.0, -0.1386050433, -0.0580473162],
                 [1.0, -0.0960192421, -0.8118918959]]

PQ_M1 = 0.1593017578125
PQ_M2 = 78.84375
PQ_C1 = 0.8359375
PQ_C2 = 18.8515625
PQ_C3 = 18.6875
PQ_MAX = 10000.0

JZ_B, JZ_G, JZ_D, JZ_D0 = 1.15, 0.66, -0.56, 1.6295499532821566e-11
JZ_EXP = 1.7

SPACE_ICTCP, SPACE_JZAZBZ = 0, 1
GAMUT_709, GAMUT_P3, GAMUT_2020 = 0, 1, 2


def luminance(rgb):
    return sum(c * w for c, w in zip(rgb, LUMA_709))


def pq_encode(v, ref, exp_scale):
    m2 = PQ_M2 * exp_scale
    y = max(v, 0.0) * ref / PQ_MAX
    ym = y ** PQ_M1
    return ((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym)) ** m2


def pq_decode(n, ref, exp_scale):
    m2 = PQ_M2 * exp_scale
    npw = max(n, 0.0) ** (1.0 / m2)
    l = max(npw - PQ_C1, 0.0) / max(PQ_C2 - PQ_C3 * npw, 1e-6)
    l = l ** (1.0 / PQ_M1)
    return l * PQ_MAX / ref


def to_display_gamut(rgb, gamut):
    if gamut == GAMUT_P3:
        return mul(REC709_TO_P3, rgb)
    if gamut == GAMUT_2020:
        return mul(REC709_TO_REC2020, rgb)
    return list(rgb)


def from_display_gamut(rgb, gamut):
    if gamut == GAMUT_P3:
        return mul(P3_TO_REC709, rgb)
    if gamut == GAMUT_2020:
        return mul(REC2020_TO_REC709, rgb)
    return list(rgb)


def rec709_to_ictcp(rgb, ref):
    lms = mul(REC2020_TO_LMS, mul(REC709_TO_REC2020, rgb))
    return mul(LMS_TO_ICTCP, [pq_encode(c, ref, 1.0) for c in lms])


def ictcp_to_rec709(iab, ref):
    lms = [pq_decode(c, ref, 1.0) for c in mul(ICTCP_TO_LMS, iab)]
    return mul(REC2020_TO_REC709, mul(LMS_TO_REC2020, lms))


def rec709_to_jzazbz(rgb, ref):
    xyz = mul(REC709_TO_XYZ, rgb)
    xyzp = [JZ_B * xyz[0] - (JZ_B - 1.0) * xyz[2],
            JZ_G * xyz[1] - (JZ_G - 1.0) * xyz[0],
            xyz[2]]
    lmsp = [pq_encode(c, ref, JZ_EXP) for c in mul(XYZ_TO_JZLMS, xyzp)]
    iab = mul(JZLMS_TO_IAB, lmsp)
    jz = ((1.0 + JZ_D) * iab[0]) / (1.0 + JZ_D * iab[0]) - JZ_D0
    return [jz, iab[1], iab[2]]


def jzazbz_to_rec709(jab, ref):
    jz = jab[0] + JZ_D0
    iz = jz / (1.0 + JZ_D - JZ_D * jz)
    lms = [pq_decode(c, ref, JZ_EXP) for c in mul(JZ_IAB_TO_LMS, [iz, jab[1], jab[2]])]
    xyzp = mul(JZLMS_TO_XYZ, lms)
    x = (xyzp[0] + (JZ_B - 1.0) * xyzp[2]) / JZ_B
    y = (xyzp[1] + (JZ_G - 1.0) * x) / JZ_G
    return mul(XYZ_TO_REC709, [x, y, xyzp[2]])


def to_perceptual(rgb, space, ref):
    return rec709_to_jzazbz(rgb, ref) if space == SPACE_JZAZBZ else rec709_to_ictcp(rgb, ref)


def from_perceptual(iab, space, ref):
    return jzazbz_to_rec709(iab, ref) if space == SPACE_JZAZBZ else ictcp_to_rec709(iab, ref)


def chroma_of(iab):
    return math.hypot(iab[1], iab[2])


def hue_of(iab):
    return math.atan2(iab[2], iab[1])


def gamut_t_max(rgb, peak):
    grey = clamp(luminance(rgb), 0.0, peak)
    t = 1e6
    for c in rgb:
        d = c - grey
        if d > 1e-6:
            t = min(t, (peak - grey) / d)
        elif d < -1e-6:
            t = min(t, -grey / d)
    return t


def gamut_limit(rgb, peak):
    return max(gamut_t_max(rgb, peak), 1e-6)


def gamut_demand(rgb, peak):
    return 1.0 / gamut_limit(rgb, peak)


def gamut_scale(rgb, peak):
    return min(gamut_limit(rgb, peak), 1.0)


# ---------------------------------------------------------------------------
# psdt_curve.slangh
# ---------------------------------------------------------------------------

class Curve:
    __slots__ = ('blackLog', 'peakLog', 'midIn', 'midOutLog', 'slope',
                 'toeStart', 'shoulderStart', 'shoulderLen', 'whiteIn')

    def linear_segment(self, x):
        return self.midOutLog + self.slope * (x - self.midIn)

    def log(self, x):
        xT = self.midIn + self.toeStart
        xS = self.midIn + self.shoulderStart
        if x < xT:
            yT = self.linear_segment(xT)
            span = max(yT - self.blackLog, 1e-3)
            return self.blackLog + span * math.exp((self.slope / span) * (x - xT))
        if x > xS:
            yS = self.linear_segment(xS)
            span = max(self.peakLog - yS, 1e-3)
            return self.peakLog - span * math.exp(-(self.slope / span) * (x - xS))
        return self.linear_segment(x)

    def slope_at(self, x):
        xT = self.midIn + self.toeStart
        xS = self.midIn + self.shoulderStart
        if x < xT:
            yT = self.linear_segment(xT)
            span = max(yT - self.blackLog, 1e-3)
            return self.slope * math.exp((self.slope / span) * (x - xT))
        if x > xS:
            yS = self.linear_segment(xS)
            span = max(self.peakLog - yS, 1e-3)
            return self.slope * math.exp(-(self.slope / span) * (x - xS))
        return self.slope

    def linear(self, y):
        return 2.0 ** self.log(math.log2(max(y, 1e-8)))

    def highlight_progress(self, x):
        xS = self.midIn + self.shoulderStart
        xW = self.midIn + self.whiteIn
        return saturate((x - xS) / max(xW - xS, 1e-3))


# ---------------------------------------------------------------------------
# psdt_state.comp.slang - parameter resolution
# ---------------------------------------------------------------------------

MIDDLE_GREY_LOG = math.log2(0.18)

# kPsdtMaxDetailBoost
MAX_DETAIL_BOOST = 1.0

DEFAULTS = dict(
    displayPeakNits=100.0, displayRefWhiteNits=100.0, displayBlackNits=0.05,
    surroundNits=5.0, displayGamut=GAMUT_709, perceptualSpace=SPACE_ICTCP,
    sceneAdaptive=True, sourceExclusion=0.75,
    localAdaptation=0.65, contrastBudget=1.0,
    midtoneContrast=1.15, shadowDepth=0.55, highlightRolloff=0.45,
    detailStrength=0.55,
    chromaPreservation=2.5, hueTrajectory=0.60, spatialWhite=0.35,
    luminanceConcession=0.8,
    colourfulness=1.0,
    glareStrength=0.5, glareThreshold=5.0, glareFalloff=0.65,
)

# kPsdtChromaReference*, measured over the displayable sRGB cube by
# psdt_suite.py's chroma-reference probe. Space-dependent: the two spaces put
# the maximum in the same place (pure blue) at very different magnitudes.
CHROMA_REFERENCE_ICTCP = 0.2933
CHROMA_REFERENCE_JZAZBZ = 0.1594


def chroma_reference(space):
    return CHROMA_REFERENCE_JZAZBZ if space == SPACE_JZAZBZ else CHROMA_REFERENCE_ICTCP


class State:
    """The resolved AdaptationState, exactly as psdt_state.comp.slang writes it."""

    def __init__(self, opts=None, anchor=MIDDLE_GREY_LOG, scene_white=None,
                 contrast_range=9.0, intent=None, level_count=6):
        o = dict(DEFAULTS)
        if opts:
            o.update(opts)
        self.o = o

        self.anchor = anchor
        self.sceneWhite = scene_white if scene_white is not None else anchor + 7.0
        self.contrastRange = contrast_range
        i = intent or dict(night=0.0, indoor=1.0, outdoor=0.0, bright=0.0)
        self.intent = i
        self.levelCount = level_count

        adaptive = 1.0 if o['sceneAdaptive'] else 0.0

        ref = max(o['displayRefWhiteNits'], 1.0)
        self.refWhite = ref
        self.peakLog = math.log2(max(o['displayPeakNits'], ref) / ref)
        blackLinear = max(o['displayBlackNits'] / ref, 1e-5)
        self.blackLog = math.log2(blackLinear)
        self.gamut = o['displayGamut']
        self.space = o['perceptualSpace']

        surroundRatio = saturate(o['surroundNits'] / max(0.5 * ref, 1.0))
        midOut = 0.18 * lerp(0.92, 1.16, surroundRatio)
        self.midOut = midOut
        midOutLog = math.log2(max(midOut, 1e-6))

        slope = max(o['midtoneContrast'] *
                    lerp(1.0, 1.0 + 0.18 * i['night'] - 0.12 * i['outdoor'], adaptive), 0.30)

        shadowRange = max(midOutLog - self.blackLog, 1.0) / slope
        shadowDepth = saturate(o['shadowDepth'] * lerp(1.0, 1.0 + 0.25 * i['night'], adaptive))
        toeStart = -shadowDepth * shadowRange

        highlightRange = max(self.peakLog - midOutLog, 1.0) / slope
        rolloff = saturate(o['highlightRolloff'] * lerp(1.0, 1.0 - 0.30 * i['bright'], adaptive))
        shoulderStart = clamp(rolloff * highlightRange, 0.25, max(highlightRange - 0.25, 0.30))

        yS = midOutLog + slope * shoulderStart
        shoulderLen = max(self.peakLog - yS, 1e-3) / max(slope, 1e-3)
        whiteIn = min(shoulderStart + shoulderLen * 3.0,
                      max(self.sceneWhite - anchor, shoulderStart + 1.0))

        c = Curve()
        c.blackLog = self.blackLog
        c.peakLog = self.peakLog
        c.midIn = anchor
        c.midOutLog = midOutLog
        c.slope = slope
        c.toeStart = toeStart
        c.shoulderStart = shoulderStart
        c.shoulderLen = shoulderLen
        c.whiteIn = whiteIn
        self.curve = c

        rangeGate = smoothstep(3.0, 8.0, contrast_range)
        self.adaptStrength = saturate(o['localAdaptation']) * lerp(1.0, rangeGate, adaptive)
        fineness = saturate((contrast_range - 6.0) / 8.0) * adaptive
        self.weightL = lerp(0.60, 0.30, fineness)
        self.weightM = 0.30
        self.weightS = lerp(0.10, 0.40, fineness)

        b = max(o['contrastBudget'], 0.0)
        self.budgetShadow, self.budgetMid = b * 2.00, b * 0.55
        self.budgetHighlight, self.budgetEmitter = b * 0.25, b * 0.05

        self.detailStrength = saturate(o['detailStrength'])
        self.detailProtect = 1.0

        self.pressureLuma = 0.50
        self.pressureChroma = chroma_reference(self.space)
        self.pressureGamut = 1.0 - 1.0 / (1.0 + max(o['chromaPreservation'], 0.05))
        self.chromaPreserve = max(o['chromaPreservation'], 0.05)
        self.hueTrajectory = saturate(o['hueTrajectory'])
        self.whiteConverge = 2.0
        self.spatialWhite = saturate(o['spatialWhite'])
        self.colourfulness = max(o['colourfulness'], 0.0)
        self.lumaConcession = saturate(o['luminanceConcession'])

        self.glareStrength = max(o['glareStrength'], 0.0)
        self.glareThreshold = anchor + o['glareThreshold']
        self.glareFalloff = clamp(o['glareFalloff'], 0.05, 0.95)


# ---------------------------------------------------------------------------
# psdt_transform.slangh
# ---------------------------------------------------------------------------

def adaptation_budget(rel, shadow_b, mid_b, high_b, emit_b):
    if rel <= -4.0:
        return shadow_b
    if rel < 0.0:
        return lerp(shadow_b, mid_b, saturate((rel + 4.0) / 4.0))
    if rel < 3.0:
        return lerp(mid_b, high_b, saturate(rel / 3.0))
    if rel < 8.0:
        return lerp(high_b, emit_b, saturate((rel - 3.0) / 5.0))
    return emit_b


def compression_knee(chroma_preservation, urgency, early_onset):
    base = 1.0 - 1.0 / (1.0 + max(chroma_preservation, 0.05))
    return saturate(base * (1.0 - saturate(early_onset) * saturate(urgency)))


def compress_to_boundary(demand, knee):
    if demand <= knee:
        return 1.0
    headroom = max(1.0 - knee, 1e-3)
    compressed = knee + headroom * (1.0 - math.exp(-(demand - knee) / headroom))
    return compressed / max(demand, 1e-4)


def psdt_apply(rgb_in, st, pooled_log=None, local_chroma=(0.0, 0.0), glare=(0.0, 0.0, 0.0)):
    """
    One pixel through the transform.

    `pooled_log` is what the adaptation field would have reported for this
    pixel's neighbourhood. Leaving it None models an isolated pixel in a scene
    the viewer is adapted to, i.e. pooled == anchor, which is the case a
    response curve is actually measuring.
    """
    space, gamut = st.space, st.gamut
    ref, peak_log = st.refWhite, st.peakLog
    peak = 2.0 ** peak_log

    rgb = [max(c, 0.0) for c in rgb_in]
    scene_y = max(luminance(rgb), 1e-8)
    scene_log = math.log2(scene_y)

    pooled = st.anchor if pooled_log is None else pooled_log
    raw = lerp(st.anchor, pooled, st.adaptStrength)
    budget = adaptation_budget(pooled - st.anchor, st.budgetShadow, st.budgetMid,
                               st.budgetHighlight, st.budgetEmitter)
    adapted = st.anchor + clamp(raw - st.anchor, -budget, budget)

    curve = Curve()
    for f in Curve.__slots__:
        setattr(curve, f, getattr(st.curve, f))
    curve.midIn = adapted

    base_log = pooled
    detail_in = scene_log - base_log
    base_slope = curve.slope_at(base_log)
    base_progress = curve.highlight_progress(base_log)
    restore = saturate(st.detailStrength) * (1.0 - saturate(base_progress * st.detailProtect))
    target_slope = lerp(base_slope, 1.0, restore)
    boost = clamp(target_slope / max(base_slope, 1e-3) - 1.0, 0.0, MAX_DETAIL_BOOST)
    out_log = curve.log(base_log + detail_in * (1.0 + boost))
    highlight_progress = curve.highlight_progress(scene_log)
    curve_slope = base_slope

    display_y = clamp(2.0 ** out_log, 2.0 ** curve.blackLog, peak)

    path_film = [curve.linear(c) for c in rgb]
    film_y = max(luminance(path_film), 1e-8)

    strict = [c * display_y / scene_y for c in rgb]
    demand_strict = gamut_demand(to_display_gamut(strict, gamut), peak)
    trouble = saturate(1.0 - 1.0 / max(demand_strict, 1e-4))
    target_y = lerp(display_y, min(film_y, display_y), st.lumaConcession * trouble)

    path_preserve = [c * target_y / scene_y for c in rgb]

    iab_preserve = to_perceptual(path_preserve, space, ref)
    iab_film = to_perceptual(path_film, space, ref)

    chroma_in = chroma_of(iab_preserve)
    hue_in = hue_of(iab_preserve)
    hue_film = hue_of(iab_film)

    chroma_norm = saturate(chroma_in / max(st.pressureChroma, 1e-3))

    convergence = trouble ** max(st.whiteConverge, 1e-3)
    d_hue = hue_film - hue_in
    d_hue -= 2.0 * math.pi * math.floor(d_hue / (2.0 * math.pi) + 0.5)
    hue_out = hue_in + d_hue * st.hueTrajectory * convergence

    rotated = [max(c, 0.0) for c in from_perceptual(
        [iab_preserve[0], chroma_in * math.cos(hue_out), chroma_in * math.sin(hue_out)], space, ref)]
    ry = max(luminance(rotated), 1e-8)
    rotated = [c * target_y / ry for c in rotated]

    demand = gamut_demand(to_display_gamut(rotated, gamut), peak)
    urgency = highlight_progress * chroma_norm
    knee = compression_knee(st.chromaPreserve, urgency, st.pressureLuma)
    compress = compress_to_boundary(demand, knee)
    pressure = saturate(1.0 - compress)

    spatial_white = (local_chroma[0] * st.spatialWhite, local_chroma[1] * st.spatialWhite)
    adapt_offset = st.anchor - adapted
    colourfulness = 1.0 + clamp(adapt_offset * st.colourfulness * 0.1, -0.35, 0.35)

    rel = (chroma_in * math.cos(hue_out) - spatial_white[0],
           chroma_in * math.sin(hue_out) - spatial_white[1])
    compressed = (rel[0] * compress * colourfulness + spatial_white[0] * (1.0 - 0.5 * pressure),
                  rel[1] * compress * colourfulness + spatial_white[1] * (1.0 - 0.5 * pressure))

    out_rgb = from_perceptual([iab_preserve[0], compressed[0], compressed[1]], space, ref)
    out_rgb = [max(c, 0.0) for c in out_rgb]
    oy = max(luminance(out_rgb), 1e-8)
    out_rgb = [c * target_y / oy for c in out_rgb]

    disp = to_display_gamut(out_rgb, gamut)
    fit = gamut_scale(disp, peak)
    grey = clamp(luminance(disp), 0.0, peak)
    disp = [grey + (c - grey) * fit for c in disp]
    out_rgb = from_display_gamut(disp, gamut)

    out_rgb = [max(c + g, 0.0) for c, g in zip(out_rgb, glare)]
    if peak_log > 0.0:
        out_rgb = [c / peak for c in out_rgb]
    return [min(c, 1.0) for c in out_rgb], dict(
        pressure=pressure, demand=demand, knee=knee, compress=compress,
        chromaNorm=chroma_norm,
        highlightProgress=highlight_progress, displayY=display_y,
        targetY=target_y, filmY=film_y, trouble=trouble,
        adapted=adapted, curveSlope=curve_slope, boost=boost)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def reinhard(rgb):
    y = luminance(rgb)
    if y <= 0.0:
        return [0.0, 0.0, 0.0]
    return [saturate(c * ((y / (1.0 + y)) / y)) for c in rgb]


def hable(rgb, exposure_bias=2.0, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, W=4.0):
    def f(x):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    wf = f(W)
    return [saturate(f(c * exposure_bias) / wf) for c in rgb]


GT7_MID, GT7_LINEAR, GT7_TOE = 0.538, 0.444, 1.280
GT7_KA, GT7_KB, GT7_KC = 1.18533333, -1.34943521, -1.34892086
GT7_FADE0, GT7_FADE1, GT7_BLEND = 0.98, 1.16, 0.6
GT7_TARGET_UCS = 0.50381


def _gt7_smoothstep(x, e0, e1):
    if x < e0:
        return 0.0
    if x > e1:
        return 1.0
    t = (x - e0) / (e1 - e0)
    return t * t * (3.0 - 2.0 * t)


def _gt7_curve(x):
    if x < 0.0:
        return 0.0
    wl = _gt7_smoothstep(x, 0.0, GT7_MID)
    if x < GT7_LINEAR:
        toe = GT7_MID * (x / GT7_MID) ** GT7_TOE
        return (1.0 - wl) * toe + wl * x
    return GT7_KA + GT7_KB * math.exp(x * GT7_KC)


def gt7(rgb, ref=100.0):
    """
    The shipped port, faithfully - including its Rec.2020 input assumption,
    which is what makes it a control worth having rather than a second PSDT.
    """
    skewed = [_gt7_curve(c) for c in rgb]
    ucs = rec709_to_ictcp(mul(REC2020_TO_REC709, rgb), ref) if False else _gt7_ictcp(rgb, ref)
    skewed_ucs = _gt7_ictcp(skewed, ref)
    scale = 1.0 - _gt7_smoothstep(ucs[0] / GT7_TARGET_UCS, GT7_FADE0, GT7_FADE1)
    scaled = _gt7_from_ictcp([skewed_ucs[0], ucs[1] * scale, ucs[2] * scale], ref)
    return [min((1.0 - GT7_BLEND) * s + GT7_BLEND * u, 1.0) for s, u in zip(skewed, scaled)]


def _gt7_ictcp(rgb, ref):
    lms = mul(REC2020_TO_LMS, rgb)
    return mul(LMS_TO_ICTCP, [pq_encode(c, ref, 1.0) for c in lms])


def _gt7_from_ictcp(iab, ref):
    lms = [pq_decode(c, ref, 1.0) for c in mul(ICTCP_TO_LMS, iab)]
    return [max(c, 0.0) for c in mul(LMS_TO_REC2020, lms)]
