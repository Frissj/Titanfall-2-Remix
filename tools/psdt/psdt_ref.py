"""
PSDT CPU reference.

A line-for-line port of the PSDT shaders, in the sense that every constant,
every clamp and every order of operations is the one the shader uses. That is
the whole point of it: it is not an idealised model of the transform, it is the
transform, run somewhere it can be measured.

v0.1's reference was described that way and was not quite: it clamped the
shoulder onset where the shader did not, used a different black-level epsilon,
and modelled neither the glare nor the white point nor the multi-scale pooling
at all - psdt_apply took `glare` as a pre-computed argument nobody ever passed.
So three of the stages had no coverage behind the numbers the README quoted.
The divergences are fixed, the three missing stages are here, and the temporal
state is modelled too, so step response can be measured rather than asserted.

Anything in this file that does NOT correspond to shader code is marked
"harness only".
"""

import math

# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def mul(M, v):
    return [M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2] for i in range(3)]


def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


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

GAMUT_709 = 0
GAMUT_P3 = 1
GAMUT_2020 = 2

SPACE_ICTCP = 0
SPACE_JZAZBZ = 1

LUMA_709 = (0.2126390059, 0.7151686788, 0.0721923154)
LUMA_P3 = (0.2289745641, 0.6917385218, 0.0792869141)
LUMA_2020 = (0.2627002120, 0.6779980715, 0.0593017165)

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
REC709_TO_P3 = [[0.8224621, 0.1775380, 0.0000000],
                [0.0331941, 0.9668058, 0.0000000],
                [0.0170827, 0.0723974, 0.9105199]]
P3_TO_REC709 = [[1.2249401, -0.2249404, 0.0000000],
                [-0.0420569, 1.0420571, 0.0000000],
                [-0.0196376, -0.0786361, 1.0982735]]

# kPsdtChromaRef*, measured over each gamut's own displayable cube by
# psdt_suite.py's chroma-reference probe. Not interchangeable across gamuts:
# Rec.709 and Rec.2020 differ by 37% in ICtCp.
CHROMA_REF = {
    (SPACE_ICTCP, GAMUT_709): 0.2933,
    (SPACE_ICTCP, GAMUT_P3): 0.3505,
    (SPACE_ICTCP, GAMUT_2020): 0.4026,
    (SPACE_JZAZBZ, GAMUT_709): 0.1594,
    (SPACE_JZAZBZ, GAMUT_P3): 0.1727,
    (SPACE_JZAZBZ, GAMUT_2020): 0.2133,
}


def chroma_reference(space, gamut):
    return CHROMA_REF[(space, gamut)]


def luminance(rgb):
    return rgb[0] * LUMA_709[0] + rgb[1] * LUMA_709[1] + rgb[2] * LUMA_709[2]


def display_luminance(rgb, gamut):
    w = LUMA_P3 if gamut == GAMUT_P3 else (LUMA_2020 if gamut == GAMUT_2020 else LUMA_709)
    return rgb[0] * w[0] + rgb[1] * w[1] + rgb[2] * w[2]


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


# --- CAT02 chromatic adaptation --------------------------------------------
REC709_TO_CATLMS = [[0.390410370, 0.549917036, 0.008903719],
                    [0.070914464, 0.963129579, 0.001358419],
                    [0.023138990, 0.128015194, 0.936276576]]
CATLMS_TO_REC709 = [[2.858765463, -1.628966853, -0.024822606],
                    [-0.210429557, 1.158388030, 0.000320450],
                    [-0.041879468, -0.118126014, 1.068630117]]
CATLMS_D65 = (0.949231125, 1.035402462, 1.087430760)


def adaptation_gain(white_rgb):
    y = max(luminance(white_rgb), 1e-4)
    lms = mul(REC709_TO_CATLMS, [c / y for c in white_rgb])
    return [CATLMS_D65[i] / max(lms[i], 1e-4) for i in range(3)]


def adapt_to_d65(rgb, gain):
    lms = mul(REC709_TO_CATLMS, rgb)
    return mul(CATLMS_TO_REC709, [lms[i] * gain[i] for i in range(3)])


def adapt_from_d65(rgb, gain):
    lms = mul(REC709_TO_CATLMS, rgb)
    return mul(CATLMS_TO_REC709, [lms[i] / max(gain[i], 1e-6) for i in range(3)])


# --- PQ ---------------------------------------------------------------------
pq_m1 = 2610.0 / 16384.0
pq_m2 = 2523.0 / 4096.0 * 128.0
pq_c1 = 3424.0 / 4096.0
pq_c2 = 2413.0 / 4096.0 * 32.0
pq_c3 = 2392.0 / 4096.0 * 32.0
pq_max = 10000.0


def pq_encode(v, ref, exp_scale):
    m2 = pq_m2 * exp_scale
    y = max(v, 0.0) * ref / pq_max
    ym = y ** pq_m1
    return ((pq_c1 + pq_c2 * ym) / (1.0 + pq_c3 * ym)) ** m2


def pq_decode(n, ref, exp_scale):
    m2 = pq_m2 * exp_scale
    np_ = max(n, 0.0) ** (1.0 / m2)
    l = max(np_ - pq_c1, 0.0) / max(pq_c2 - pq_c3 * np_, 1e-6)
    l = l ** (1.0 / pq_m1)
    return l * pq_max / ref


# --- ICtCp ------------------------------------------------------------------
REC2020_TO_LMS = [[1688 / 4096, 2146 / 4096, 262 / 4096],
                  [683 / 4096, 2951 / 4096, 462 / 4096],
                  [99 / 4096, 309 / 4096, 3688 / 4096]]
LMS_TO_ICTCP = [[2048 / 4096, 2048 / 4096, 0.0],
                [6610 / 4096, -13613 / 4096, 7003 / 4096],
                [17933 / 4096, -17390 / 4096, -543 / 4096]]
ICTCP_TO_LMS = [[1.0, 0.0086090370, 0.1110296250],
                [1.0, -0.0086090370, -0.1110296250],
                [1.0, 0.5600313357, -0.3206271750]]
LMS_TO_REC2020 = [[3.4366066943, -2.5064521187, 0.0698454243],
                  [-0.7913295556, 1.9836004518, -0.1922708962],
                  [-0.0259498997, -0.0989137147, 1.1248636144]]


def rec709_to_ictcp(rgb, ref):
    lms = mul(REC2020_TO_LMS, mul(REC709_TO_REC2020, rgb))
    return mul(LMS_TO_ICTCP, [pq_encode(c, ref, 1.0) for c in lms])


def ictcp_to_rec709(iab, ref):
    lms = [pq_decode(c, ref, 1.0) for c in mul(ICTCP_TO_LMS, iab)]
    return mul(REC2020_TO_REC709, mul(LMS_TO_REC2020, lms))


# --- Jzazbz -----------------------------------------------------------------
JZ_B, JZ_G, JZ_D = 1.15, 0.66, -0.56
JZ_D0 = 1.6295499532821566e-11
JZ_EXP = 1.7
XYZ_TO_JZLMS = [[0.41478972, 0.579999, 0.0146480],
                [-0.20151000, 1.120649, 0.0531008],
                [-0.01660080, 0.264800, 0.6684799]]
JZLMS_TO_XYZ = [[1.9242264358, -1.0047923126, 0.0376514040],
                [0.3503167621, 0.7264811939, -0.0653844229],
                [-0.0909828110, -0.3127282905, 1.5227665613]]
JZLMS_TO_IAB = [[0.5, 0.5, 0.0],
                [3.524, -4.066708, 0.542708],
                [0.199076, 1.096799, -1.295875]]
JZIAB_TO_LMS = [[1.0, 0.1386050433, 0.0580473162],
                [1.0, -0.1386050433, -0.0580473162],
                [1.0, -0.0960192421, -0.8118918959]]


def rec709_to_jzazbz(rgb, ref):
    x = mul(REC709_TO_XYZ, rgb)
    xp = [JZ_B * x[0] - (JZ_B - 1.0) * x[2],
          JZ_G * x[1] - (JZ_G - 1.0) * x[0],
          x[2]]
    lms = mul(XYZ_TO_JZLMS, xp)
    lmsp = [pq_encode(c, ref, JZ_EXP) for c in lms]
    iab = mul(JZLMS_TO_IAB, lmsp)
    jz = ((1.0 + JZ_D) * iab[0]) / (1.0 + JZ_D * iab[0]) - JZ_D0
    return [jz, iab[1], iab[2]]


def jzazbz_to_rec709(jab, ref):
    jz = jab[0] + JZ_D0
    iz = jz / (1.0 + JZ_D - JZ_D * jz)
    lmsp = mul(JZIAB_TO_LMS, [iz, jab[1], jab[2]])
    lms = [pq_decode(c, ref, JZ_EXP) for c in lmsp]
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


def from_polar(intensity, chroma, hue):
    return [intensity, chroma * math.cos(hue), chroma * math.sin(hue)]


# --- display colour volume --------------------------------------------------

def gamut_t_max(rgb, peak, gamut):
    grey = clamp(display_luminance(rgb, gamut), 0.0, peak)
    d = [rgb[i] - grey for i in range(3)]
    t_max = 1e6
    for i in range(3):
        if d[i] > 1e-6:
            t_max = min(t_max, (peak - grey) / d[i])
        elif d[i] < -1e-6:
            t_max = min(t_max, -grey / d[i])
    return max(t_max, 1e-6)


def gamut_limit(rgb, peak, gamut=GAMUT_709):
    return gamut_t_max(rgb, peak, gamut)


def gamut_demand(rgb, peak, gamut=GAMUT_709):
    return 1.0 / gamut_t_max(rgb, peak, gamut)


def gamut_scale(rgb, peak, gamut=GAMUT_709):
    return min(gamut_t_max(rgb, peak, gamut), 1.0)


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

    def linear(self, scene_luminance):
        return 2.0 ** self.log(math.log2(max(scene_luminance, 1e-8)))

    def highlight_progress(self, scene_log2):
        xS = self.midIn + self.shoulderStart
        xW = self.midIn + self.whiteIn
        return saturate((scene_log2 - xS) / max(xW - xS, 1e-3))


# ---------------------------------------------------------------------------
# psdt_state.comp.slang
# ---------------------------------------------------------------------------

MIDDLE_GREY_LOG = math.log2(0.18)

# kPsdtMaxDetailBoost, kPsdtGlareTintFraction, kPsdtDepthDeadzone
MAX_DETAIL_BOOST = 1.0
GLARE_TINT_FRACTION = 0.6
DEPTH_DEADZONE = 1.0

# kSkyAdaptationWeight, kEmissiveThresholdDiscount (psdt_analysis)
SKY_ADAPTATION_WEIGHT = 0.15
EMISSIVE_THRESHOLD_DISCOUNT = 2.5

DEFAULTS = dict(
    displayPeakNits=100.0, displayRefWhiteNits=100.0, displayBlackNits=0.05,
    surroundNits=5.0, displayGamut=GAMUT_709, perceptualSpace=SPACE_ICTCP,
    sceneAdaptive=True,
    # classification
    rendererSignals=True, sourceThreshold=4.0, glareClassThreshold=8.0,
    skyViewZ=100000.0, illuminantMinAlbedo=0.04,
    # adaptation
    sourceExclusion=0.75, localAdaptation=0.65, contrastBudget=1.0,
    depthSensitivity=0.35,
    adaptationSpeedUp=3.0, adaptationSpeedDown=1.2,
    # curve
    midtoneContrast=1.15, shadowDepth=0.55, highlightRolloff=0.45,
    detailStrength=0.55, detailProtect=1.0, detailKnee=3.0,
    # colour volume
    chromaPreservation=2.5, hueTrajectory=0.60, pressureLuma=0.50,
    pressureContext=0.6, whiteConvergence=2.0, spatialWhite=0.35,
    colourfulness=1.0, luminanceConcession=0.8,
    # glare
    glareStrength=0.5, glareThreshold=5.0, glareFalloff=0.65,
    glareNearField=0.8,
)


class State:
    """The resolved AdaptationState, exactly as psdt_state.comp.slang writes it."""

    def __init__(self, opts=None, anchor=MIDDLE_GREY_LOG, scene_white=None,
                 contrast_range=9.0, intent=None, level_count=6,
                 illuminant=(1.0, 1.0, 1.0), illum_confidence=1.0):
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
        self.frameIlluminant = tuple(illuminant)
        self.illumConfidence = saturate(illum_confidence)

        adaptive = 1.0 if o['sceneAdaptive'] else 0.0

        ref = max(o['displayRefWhiteNits'], 1.0)
        self.refWhite = ref
        self.peakLog = math.log2(max(o['displayPeakNits'], ref) / ref)
        # Matches the shader's 1e-7 floor. v0.1's reference used 1e-5.
        blackLinear = max(o['displayBlackNits'] / ref, 1e-7)
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
        # The shader zeroes this when the gbuffer is unavailable, because the
        # depth channel is then meaningless rather than merely coarse.
        self.depthSensitivity = max(o['depthSensitivity'], 0.0) if o['rendererSignals'] else 0.0

        self.detailStrength = saturate(o['detailStrength'])
        self.detailProtect = max(o['detailProtect'], 0.0)
        self.detailKnee = max(o['detailKnee'], 0.05)

        self.pressureLuma = saturate(o['pressureLuma'])
        self.pressureChroma = chroma_reference(self.space, self.gamut)
        self.pressureGamut = 1.0 - 1.0 / (1.0 + max(o['chromaPreservation'], 0.05))
        self.pressureContext = max(o['pressureContext'], 0.0)
        self.chromaPreserve = max(o['chromaPreservation'], 0.05)
        self.hueTrajectory = saturate(o['hueTrajectory'])
        self.whiteConverge = max(o['whiteConvergence'], 0.05)
        self.illumStrength = saturate(o['spatialWhite']) * self.illumConfidence
        self.colourfulness = max(o['colourfulness'], 0.0)
        self.lumaConcession = saturate(o['luminanceConcession'])

        self.glareStrength = max(o['glareStrength'], 0.0)
        self.glareThreshold = anchor + o['glareThreshold']
        self.glareFalloff = clamp(o['glareFalloff'], 0.05, 0.95)
        self.glareNearField = saturate(o['glareNearField'])

        self.levelS, self.levelM, self.levelL = 0, min(2, level_count - 1), min(4, level_count - 1)


# ---------------------------------------------------------------------------
# psdt_state.comp.slang - the temporal half
# ---------------------------------------------------------------------------

class StateFilter:
    """
    The measured half of psdt_state: the anchor's inertia, the asymmetric time
    constants, the sweep multiplier and the cut re-anchor.

    Harness note: this models the state update only. The field's own temporal
    accumulation in psdt_analysis is modelled by FieldFilter below, and the two
    together are what a step-response measurement has to walk through - the
    transform has two temporal filters in series and the whole point of
    measuring the step response is to find out what that pair actually does.
    """

    def __init__(self, opts=None, anchor=MIDDLE_GREY_LOG):
        o = dict(DEFAULTS)
        if opts:
            o.update(opts)
        self.o = o
        self.anchor = anchor
        self.sceneWhite = anchor + 7.0
        self.sceneBlack = anchor - 7.0
        # Previous frame's measurements, before filtering. Only the cut
        # detector reads them - see the note in step().
        self.prevTargetAnchor = anchor
        self.prevTargetWhite = anchor + 7.0
        self.prevTargetBlack = anchor - 7.0
        self.cutFlag = 0.0
        self.hasHistory = False

    def step(self, anchor_target, dt, scene_white=None, scene_black=None,
             camera_angular=0.0, camera_cut=False):
        white = anchor_target + 7.0 if scene_white is None else scene_white
        black = anchor_target - 7.0 if scene_black is None else scene_black

        sweep = saturate(smoothstep(1.5, 6.0, camera_angular))

        # Measurement against measurement, not measurement against state.
        # Downward adaptation is slow on purpose, so during a sustained fade
        # the state falls behind the measurement by design; comparing the two
        # turns that intended lag into a false cut part way through the fade.
        anchor_jump = abs(anchor_target - self.prevTargetAnchor)
        white_jump = abs(white - self.prevTargetWhite)
        black_jump = abs(black - self.prevTargetBlack)
        luminance_cut = self.hasHistory and (
            anchor_jump > 3.0 or (white_jump > 3.5 and black_jump > 3.5))
        is_cut = self.hasHistory and (camera_cut or luminance_cut)

        dt = clamp(dt, 0.0, 0.25)
        base_speed = self.o['adaptationSpeedUp'] if anchor_target > self.anchor \
            else self.o['adaptationSpeedDown']
        speed = base_speed * lerp(1.0, 3.0, sweep)
        alpha = 1.0 - math.exp(-max(speed, 0.01) * dt)
        previous_cut_flag = self.cutFlag
        if not self.hasHistory or is_cut or previous_cut_flag > 0.5:
            alpha = 1.0

        self.anchor = lerp(self.anchor, anchor_target, alpha)
        self.sceneWhite = lerp(self.sceneWhite, white, alpha)
        self.sceneBlack = lerp(self.sceneBlack, black, alpha)
        self.cutFlag = 1.0 if is_cut else max(previous_cut_flag - dt * 4.0, 0.0)
        self.prevTargetAnchor = anchor_target
        self.prevTargetWhite = white
        self.prevTargetBlack = black
        self.hasHistory = True
        return self.anchor


class FieldFilter:
    """
    The adaptation field's own temporal accumulation, from psdt_analysis. One
    texel of it: the history clamp, the cut-driven weight lift, and the
    exponential blend.
    """

    def __init__(self, speed=8.0, value=MIDDLE_GREY_LOG):
        self.speed = speed
        self.value = value
        self.hasHistory = False

    def step(self, measured, dt, cut_flag=0.0):
        per_frame = self.speed * dt
        current_weight = min(1.0, max(0.05, 1.0 - 2.0 ** (-per_frame))) if self.hasHistory else 1.0
        current_weight = lerp(current_weight, 1.0, saturate(cut_flag))
        if self.hasHistory and current_weight < 1.0:
            history = clamp(self.value, measured - 3.0, measured + 3.0)
            self.value = lerp(history, measured, current_weight)
        else:
            self.value = measured
        self.hasHistory = True
        return self.value


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


class Adaptation:
    __slots__ = ('pooledLog', 'anchorLog', 'adaptedLog', 'bodyWeight',
                 'depthLog', 'coarseTrust', 'illuminant')


def sample_adaptation(st, field_levels=None, illum=None):
    """
    psdtSampleAdaptation. `field_levels` is {'S': (logY, weight, peak, depth),
    'M': ..., 'L': ...}; None means every scale reports the frame anchor with
    full body weight and equal depth, which is the isolated-pixel case.
    `illum` is (r, g, b, weight) from the illuminant field, or None for no
    local estimate.
    """
    if field_levels is None:
        f = (st.anchor, 1.0, st.anchor, 0.0)
        field_levels = {'S': f, 'M': f, 'L': f}

    fS, fM, fL = field_levels['S'], field_levels['M'], field_levels['L']
    anchor = st.anchor

    wS, wM, wL = fS[1], fM[1], fL[1]
    aS = lerp(anchor, fS[0], saturate(wS * 4.0))
    aM = lerp(anchor, fM[0], saturate(wM * 4.0))
    aL = lerp(anchor, fL[0], saturate(wL * 4.0))

    w = [st.weightL, st.weightM, st.weightS]
    coarse_trust = 1.0
    if st.depthSensitivity > 1e-3:
        excess_m = max(abs(fM[3] - fS[3]) - DEPTH_DEADZONE, 0.0)
        excess_l = max(abs(fL[3] - fS[3]) - DEPTH_DEADZONE, 0.0)
        agree_m = math.exp(-st.depthSensitivity * excess_m * excess_m)
        agree_l = math.exp(-st.depthSensitivity * excess_l * excess_l)
        w[0] *= agree_l
        w[1] *= agree_m
        coarse_trust = 0.5 * (agree_m + agree_l)
    total = max(w[0] + w[1] + w[2], 1e-4)
    w = [x / total for x in w]

    a = Adaptation()
    a.pooledLog = w[0] * aL + w[1] * aM + w[2] * aS
    a.anchorLog = anchor
    a.bodyWeight = max(wS, wM, wL)
    a.depthLog = fS[3]
    a.coarseTrust = coarse_trust

    raw = lerp(anchor, a.pooledLog, st.adaptStrength)
    budget = adaptation_budget(a.pooledLog - anchor, st.budgetShadow, st.budgetMid,
                               st.budgetHighlight, st.budgetEmitter)
    a.adaptedLog = anchor + clamp(raw - anchor, -budget, budget)

    frame_illum = list(st.frameIlluminant)
    if illum is not None and illum[3] > 1e-4:
        local = [illum[i] / illum[3] for i in range(3)]
        conf = saturate(illum[3] * 4.0)
    else:
        local, conf = frame_illum, 0.0
    a.illuminant = [lerp(frame_illum[i], local[i], conf) for i in range(3)]
    return a


class GlareResult:
    __slots__ = ('colour', 'energy')


def glare(st, source_levels):
    """
    psdtGlare. `source_levels` is a list of (r, g, b) source energies, one per
    pyramid level, as the source field would report at this pixel.
    """
    result = GlareResult()
    result.colour = [0.0, 0.0, 0.0]
    result.energy = 0.0
    if st.glareStrength <= 0.0 or not source_levels:
        return result

    threshold = 2.0 ** st.glareThreshold
    amount = 0.0
    tinted = [0.0, 0.0, 0.0]
    norm = 0.0
    wide = 1.0

    for k in range(min(len(source_levels), st.levelCount)):
        level_rgb = [max(c, 0.0) for c in source_levels[k]]
        level_energy = luminance(level_rgb)
        weight = wide + (st.glareNearField if k < 2 else 0.0)
        wide *= st.glareFalloff
        norm += weight

        over = max(level_energy - threshold, 0.0)
        if over <= 0.0:
            continue
        visible = over / (over + threshold)
        amount += weight * visible * math.log2(1.0 + level_energy)
        tinted = [tinted[i] + weight * visible * level_rgb[i] for i in range(3)]
        result.energy = max(result.energy, level_energy)

    if amount <= 0.0:
        return result

    amount = st.glareStrength * amount / max(norm, 1e-4) * 0.125
    tint_y = max(luminance(tinted), 1e-6)
    chromaticity = [lerp(1.0, tinted[i] / tint_y, GLARE_TINT_FRACTION) for i in range(3)]
    result.colour = [max(chromaticity[i], 0.0) * amount for i in range(3)]
    return result


def compression_knee(chroma_preservation, urgency, early_onset):
    base = 1.0 - 1.0 / (1.0 + max(chroma_preservation, 0.05))
    return saturate(base * (1.0 - saturate(early_onset) * saturate(urgency)))


def compress_to_boundary(demand, knee):
    if demand <= knee:
        return 1.0
    headroom = max(1.0 - knee, 1e-3)
    compressed = knee + headroom * (1.0 - math.exp(-(demand - knee) / headroom))
    return compressed / max(demand, 1e-4)


def psdt_apply(rgb_in, st, pooled_log=None, adapt=None, glare_rgb=(0.0, 0.0, 0.0)):
    """
    One pixel through the transform.

    `adapt` is an Adaptation from sample_adaptation(). `pooled_log` is a
    shorthand for the common case of overriding only the pooled level, which is
    what a response curve measures: an isolated pixel in a scene the viewer is
    adapted to. Passing neither models pooled == anchor.
    """
    space, gamut = st.space, st.gamut
    ref, peak_log = st.refWhite, st.peakLog
    peak = 2.0 ** peak_log

    rgb = [max(c, 0.0) for c in rgb_in]
    scene_y = max(luminance(rgb), 1e-8)
    scene_log = math.log2(scene_y)

    if adapt is None:
        pooled = st.anchor if pooled_log is None else pooled_log
        f = (pooled, 1.0, pooled, 0.0)
        adapt = sample_adaptation(st, {'S': f, 'M': f, 'L': f})

    curve = Curve()
    for f in Curve.__slots__:
        setattr(curve, f, getattr(st.curve, f))
    curve.midIn = adapt.adaptedLog

    base_log = adapt.pooledLog
    detail_in = scene_log - base_log
    base_slope = curve.slope_at(base_log)
    base_progress = curve.highlight_progress(base_log)

    magnitude = abs(detail_in) / max(st.detailKnee, 1e-3)
    magnitude_rolloff = 1.0 / (1.0 + magnitude * magnitude)
    restore = (saturate(st.detailStrength)
               * (1.0 - saturate(base_progress * st.detailProtect))
               * magnitude_rolloff)
    target_slope = lerp(base_slope, 1.0, restore)
    boost = clamp(target_slope / max(base_slope, 1e-3) - 1.0, 0.0, MAX_DETAIL_BOOST)
    out_log = curve.log(base_log + detail_in * (1.0 + boost))
    highlight_progress = curve.highlight_progress(scene_log)

    display_y = clamp(2.0 ** out_log, 2.0 ** curve.blackLog, peak)

    path_film = [curve.linear(c) for c in rgb]
    film_y = max(luminance(path_film), 1e-8)

    strict = [c * display_y / scene_y for c in rgb]
    demand_strict = gamut_demand(to_display_gamut(strict, gamut), peak, gamut)
    trouble = saturate(1.0 - 1.0 / max(demand_strict, 1e-4))
    # Constant weight on a self-gating gap: a convex combination of two
    # monotone functions, which cannot invert. See the note in the shader.
    concession_gap = saturate(1.0 - film_y / max(display_y, 1e-8))
    target_y = display_y * (1.0 - st.lumaConcession * concession_gap)

    path_preserve = [c * target_y / scene_y for c in rgb]

    # --- chromatic adaptation ---
    adapt_white = st.illumStrength > 1e-3
    if adapt_white:
        white = [lerp(1.0, adapt.illuminant[i], st.illumStrength) for i in range(3)]
        gain = adaptation_gain(white)
        work_preserve = adapt_to_d65(path_preserve, gain)
        work_film = adapt_to_d65(path_film, gain)
    else:
        gain = [1.0, 1.0, 1.0]
        work_preserve, work_film = path_preserve, path_film

    iab_preserve = to_perceptual([max(c, 0.0) for c in work_preserve], space, ref)
    iab_film = to_perceptual([max(c, 0.0) for c in work_film], space, ref)

    chroma_in = chroma_of(iab_preserve)
    hue_in = hue_of(iab_preserve)
    hue_film = hue_of(iab_film)

    chroma_norm = saturate(chroma_in / max(st.pressureChroma, 1e-3))

    convergence = trouble ** max(st.whiteConverge, 1e-3)
    d_hue = hue_film - hue_in
    d_hue -= 2.0 * math.pi * math.floor(d_hue / (2.0 * math.pi) + 0.5)
    hue_out = hue_in + d_hue * st.hueTrajectory * convergence

    rotated = [max(c, 0.0) for c in from_perceptual(
        from_polar(iab_preserve[0], chroma_in, hue_out), space, ref)]
    rotated_display = adapt_from_d65(rotated, gain) if adapt_white else rotated
    rotated_display = [max(c, 0.0) for c in rotated_display]
    ry = max(luminance(rotated_display), 1e-8)
    rotated_display = [c * target_y / ry for c in rotated_display]

    demand = gamut_demand(to_display_gamut(rotated_display, gamut), peak, gamut)

    context = saturate((scene_log - base_log) / 4.0)
    early_onset = st.pressureLuma * (1.0 + st.pressureContext * context)
    urgency = highlight_progress * chroma_norm
    knee = compression_knee(st.chromaPreserve, urgency, early_onset)
    compress = compress_to_boundary(demand, knee)
    pressure = saturate(1.0 - compress)

    adapt_offset = adapt.anchorLog - adapt.adaptedLog
    colourfulness = 1.0 + clamp(adapt_offset * st.colourfulness * 0.1, -0.35, 0.35)
    compressed_chroma = chroma_in * compress * colourfulness

    out_rgb = from_perceptual(from_polar(iab_preserve[0], compressed_chroma, hue_out), space, ref)
    if adapt_white:
        out_rgb = adapt_from_d65(out_rgb, gain)
    out_rgb = [max(c, 0.0) for c in out_rgb]
    oy = max(luminance(out_rgb), 1e-8)
    out_rgb = [c * target_y / oy for c in out_rgb]

    disp = to_display_gamut(out_rgb, gamut)
    pre_fit = list(disp)
    fit = gamut_scale(disp, peak, gamut)
    grey = clamp(display_luminance(disp, gamut), 0.0, peak)
    disp = [grey + (c - grey) * fit for c in disp]
    out_rgb = from_display_gamut(disp, gamut)

    out_rgb = [c + g for c, g in zip(out_rgb, glare_rgb)]
    out_rgb = [max(c, 0.0) for c in out_rgb]
    if peak_log > 0.0:
        out_rgb = [c / peak for c in out_rgb]
    return [min(c, 1.0) for c in out_rgb], dict(
        pressure=pressure, demand=demand, knee=knee, compress=compress,
        chromaNorm=chroma_norm, context=context,
        highlightProgress=highlight_progress, displayY=display_y,
        targetY=target_y, filmY=film_y, trouble=trouble,
        adapted=adapt.adaptedLog, pooled=adapt.pooledLog,
        curveSlope=base_slope, boost=boost, preFit=pre_fit,
        coarseTrust=adapt.coarseTrust)


# ---------------------------------------------------------------------------
# psdt_analysis.comp.slang - the classifier
# ---------------------------------------------------------------------------

def classify(rgb, exposure=1.0, is_emissive=False, view_z=10.0, albedo=None,
             bright_fraction=0.0, opts=None):
    """
    One pixel through psdt_analysis's classification, returned as
    (body_weight, sourceness, skyness, source_rgb, illuminant_or_None, weight).

    `bright_fraction` is what the block-level reduction found, which is the
    compactness evidence. The shader computes it from the block; here it is an
    argument so a single pixel can be tested in both a compact and a large
    bright neighbourhood.
    """
    o = dict(DEFAULTS)
    if opts:
        o.update(opts)
    use_signals = o['rendererSignals']

    rgb = [max(c, 0.0) * exposure for c in rgb]
    y = max(luminance(rgb), 1e-5)
    log_y = clamp(math.log2(y), -63.0, 63.0)
    rel = log_y - MIDDLE_GREY_LOG

    if not use_signals:
        is_emissive, view_z, albedo = False, 0.0, None
    is_sky = use_signals and abs(view_z) >= o['skyViewZ']

    threshold = o['sourceThreshold'] - (EMISSIVE_THRESHOLD_DISCOUNT if is_emissive else 0.0)
    brightness = smoothstep(threshold - 0.5, threshold + 0.5, rel)

    compactness = 1.0 - 0.75 * smoothstep(0.35, 0.95, bright_fraction)
    sourceness = 0.0 if is_sky else saturate(brightness * (1.0 if is_emissive else compactness))
    glareness = smoothstep(o['glareClassThreshold'] - 1.0, o['glareClassThreshold'] + 1.0, rel)
    skyness = 1.0 if is_sky else 0.0
    body_weight = SKY_ADAPTATION_WEIGHT if is_sky else (1.0 - sourceness)

    energy_weight = max(sourceness, skyness * glareness) * lerp(0.25, 1.0, glareness)
    source_rgb = [min(c, 16384.0) * energy_weight for c in rgb]

    illum, illum_weight = None, 0.0
    if use_signals and not is_sky and albedo is not None:
        min_albedo = min(albedo)
        albedo_weight = saturate((min_albedo - o['illuminantMinAlbedo'])
                                 / max(o['illuminantMinAlbedo'], 1e-3))
        luminance_weight = saturate(y / (y + 0.045))
        illum_weight = (1.0 - sourceness) * albedo_weight * luminance_weight
        if illum_weight > 1e-3:
            raw = [rgb[i] / max(albedo[i], o['illuminantMinAlbedo']) for i in range(3)]
            iy = max(luminance(raw), 1e-6)
            illum = [clamp(c / iy, 0.125, 8.0) for c in raw]
        else:
            illum_weight = 0.0

    return body_weight, sourceness, skyness, source_rgb, illum, illum_weight


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
# Harness only. These are the operators PSDT is measured against; each is a
# port of the shader that runs in the build, so a number printed beside PSDT's
# is a number from the same pipeline.

def reinhard(rgb, white_point=4.0):
    """Extended Reinhard, matching reinhard_agx.slangh's per-channel form."""
    w2 = max(white_point * white_point, 1e-4)
    return [saturate(max(c, 0.0) * (1.0 + max(c, 0.0) / w2) / (1.0 + max(c, 0.0))) for c in rgb]


def hable(rgb, exposure_bias=2.0, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, W=4.0):
    def f(x):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    wf = f(W)
    return [saturate(f(max(c, 0.0) * exposure_bias) / wf) for c in rgb]


AGX_INSET = [[0.8566271533, 0.1373189729, 0.1118982129],
             [0.0951212405, 0.7612419906, 0.0767994814],
             [0.0482516061, 0.1014390365, 0.8113023057]]
AGX_OUTSET = [[1.1271005818, -0.1413297634, -0.1413297634],
              [-0.1106066501, 1.1578237022, -0.1106066501],
              [-0.0164939043, -0.0164939043, 1.2519364065]]
AGX_MIN_EV, AGX_MAX_EV = -12.47393, 4.026069


def agx(rgb):
    v = mul(AGX_INSET, [max(c, 0.0) for c in rgb])
    v = [clamp(math.log2(max(c, 1e-10)), AGX_MIN_EV, AGX_MAX_EV) for c in v]
    v = [(c - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV) for c in v]

    def sigmoid(x):
        x = saturate(x)
        x2 = x * x
        x4 = x2 * x2
        return (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
                - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)

    v = [max(sigmoid(c), 0.0) ** 2.2 for c in v]
    return [saturate(c) for c in mul(AGX_OUTSET, v)]


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


def _gt7_ictcp(rgb, ref):
    lms = mul(REC2020_TO_LMS, rgb)
    return mul(LMS_TO_ICTCP, [pq_encode(c, ref, 1.0) for c in lms])


def _gt7_from_ictcp(iab, ref):
    lms = [pq_decode(c, ref, 1.0) for c in mul(ICTCP_TO_LMS, iab)]
    return [max(c, 0.0) for c in mul(LMS_TO_REC2020, lms)]


def gt7(rgb, ref=100.0):
    """
    The shipped port, faithfully - including its Rec.2020 input assumption,
    which is what makes it a control worth having rather than a second PSDT.
    """
    skewed = [_gt7_curve(c) for c in rgb]
    ucs = _gt7_ictcp(rgb, ref)
    skewed_ucs = _gt7_ictcp(skewed, ref)
    scale = 1.0 - _gt7_smoothstep(ucs[0] / GT7_TARGET_UCS, GT7_FADE0, GT7_FADE1)
    scaled = _gt7_from_ictcp([skewed_ucs[0], ucs[1] * scale, ucs[2] * scale], ref)
    return [min((1.0 - GT7_BLEND) * s + GT7_BLEND * u, 1.0) for s, u in zip(skewed, scaled)]
