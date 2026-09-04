/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#ifndef PSDT_H
#define PSDT_H

// ===========================================================================
// PSDT - Perceptual Scene Display Transform
// ===========================================================================
//
// PSDT is not a tone curve. It is a display transform: it converts physically
// reconstructed scene radiance into an image for a viewer adapted to this
// scene, looking at this display.
//
// The reason it exists as a separate thing rather than as another entry in
// tonemap_operators.slangh is a contract. Every operator in that file - Hable,
// Psycho17, GT7, Reinhard, AgX - is a pure function of one pixel. That is the
// correct shape for a curve and the wrong shape for a display transform,
// because the two most important inputs to "what should this pixel look like"
// are not in the pixel: what the viewer is adapted to, and how much room the
// display has left. So PSDT splits the job four ways and gives each stage
// exactly one responsibility:
//
//   Auto Exposure  "what brightness am I adapted to?"        (upstream)
//   PSDT curve     "how do I reproduce the adapted scene?"   (psdt_curve)
//   Colour volume  "how do scene colours fit the display?"   (psdt_transform)
//   Glare          "how does a very bright source appear?"   (psdt_transform)
//
// Keeping those apart is the whole point. Stacking them - which is what
// happens when local exposure, a global curve, chroma compression and a gamut
// clamp each independently squeeze the same signal - is how a physically
// correct render ends up looking flat and how a saturated light ends up
// reading as "bright-ish and vaguely red" instead of as a red light.
//
// What v0.2 changed, and why
// --------------------------
// v0.1 had the right architecture with the wrong measurements underneath it.
// Four quantities the design described as physical were heuristics:
//
//   "source"       was a brightness threshold, so a sunlit wall metered as a
//                  lamp;
//   "white point"  was the local mean colour, so three coloured walls under a
//                  neutral light read as a coloured illuminant;
//   "display Y"    used Rec.709 coefficients in every gamut, so P3 and
//                  Rec.2020 had the wrong achromatic axis;
//   "glare colour" came from the scene's mean chroma, so a white light near a
//                  red wall glared red.
//
// All four are now measured from the renderer rather than guessed from the
// framebuffer. Remix has information a film pipeline does not - a per-pixel
// emissive bit, linear view depth that separates sky from geometry, and a
// demodulated albedo that lets reflectance be divided out of radiance - and
// each of those turns one of the heuristics above into a measurement:
//
//   emissive bit + view depth + spatial compactness -> BODY / SOURCE / GLARE
//   radiance / albedo over neutral-ish surfaces     -> local illuminant
//   per-gamut primaries                             -> correct achromatic axis
//   source-only RGB accumulation                    -> glare with the source's
//                                                      own colour
//
// Structure
// ---------
//   psdt_analysis.comp    full-res HDR + gbuffer -> adaptation field mip 0.
//                         Classifies BODY / SOURCE / GLARE / SKY per pixel,
//                         estimates the local illuminant, accumulates a body
//                         only luminance histogram, and reprojects the field.
//   psdt_downsample.comp  field mip k -> mip k+1, for all three field
//                         textures. The mip chain IS the multi-scale
//                         adaptation pyramid and IS the glare kernel.
//   psdt_state.comp       body histogram + saliency histogram + coarse field
//                         -> AdaptationState. One workgroup. Source-aware
//                         anchor, body-only scene white/black, scene intent,
//                         camera-aware cut detection, asymmetric temporal
//                         inertia, and resolution of every runtime parameter.
//   psdt_transform.slangh the per-pixel transform, called from the existing
//                         tonemap apply shader.
//
// Why the parameters live in the state texture
// --------------------------------------------
// MaxPushConstantSize is 128 bytes and ToneMappingApplyToneMappingArgs
// already spends 80 of them. Rather than split PSDT's parameter set across
// the two, psdt_state resolves every runtime option into the state texture
// once per frame. That is not a workaround - the resolution is genuinely
// scene-dependent (an exposure-aware curve has different parameters at night
// than in daylight, per PSDT_STATE_CURVE_*), so it has to happen per frame on
// the GPU where the scene statistics already are.
// ===========================================================================

#include "rtx/utility/shader_types.h"

// ---------------------------------------------------------------------------
// Resolution of the adaptation field
// ---------------------------------------------------------------------------
// Field texel = 16x16 source pixels. At 1920x1080 that is 120x68 texels, which
// is 8160 float4s - small enough that the whole pyramid plus its downsample
// chain costs well under a tenth of a millisecond, and coarse enough that the
// field cannot carry object-scale structure into the adaptation signal. The
// eye does not adapt to a pixel; making the finest scale this coarse is what
// stops "local adaptation" from degenerating into "map everything to grey".
#define PSDT_FIELD_BLOCK_SIZE     16
#define PSDT_GROUP_SIZE           16

// Cap on pyramid levels. Level k covers 16 * 2^k pixels, so 6 levels reaches
// 512 pixels - a quarter of a 1080p frame width, which is already beyond any
// plausible adaptation neighbourhood and is where the glare kernel runs out of
// useful energy too.
#define PSDT_MAX_LEVELS           6

// State texture length, in R32_SFLOAT texels.
#define PSDT_STATE_SIZE           96

// Body-only luminance histogram, accumulated by psdt_analysis and consumed by
// psdt_state. Separate from the tonemapper's own histogram on purpose: that
// one is centre-weighted and counts every pixel, which makes it a good
// saliency estimate and a bad description of what the scene's surfaces are
// doing. 128 buckets over [toneCurveMinStops, toneCurveMaxStops] is 0.25
// stops per bucket at the default 32-stop range.
#define PSDT_BODY_HISTOGRAM_SIZE  128

// Histogram is shared with the native tonemapper (EXPOSURE_HISTOGRAM_SIZE
// buckets over [toneCurveMinStops, toneCurveMaxStops]); psdt_state runs as a
// single workgroup of this width so one thread owns one bucket.
#define PSDT_STATE_GROUP_SIZE     256

// ---------------------------------------------------------------------------
// Adaptation field channel layouts
// ---------------------------------------------------------------------------
// Three RGBA16F textures, all mipped, all reduced by psdt_downsample. Split
// three ways rather than packed into one because the three have different
// reduction rules and different consumers: the body field is a
// weight-normalised mean, the source field is an area mean that has to
// conserve energy, and the illuminant field is a weighted mean that is only
// meaningful after dividing by its own weight.

// Body field. What the viewer is adapting to.
//  .x  body log2 luminance, exposed. Weighted mean over BODY-classified pixels
//      only; the weight is .y. Source pixels are excluded rather than clamped,
//      because a clamped sun still drags the mean and an excluded one does not.
//  .y  body weight in [0,1] - what fraction of the block was ordinary surface.
//      A block that is entirely lamp has .y == 0 and contributes no adaptation.
//  .z  block peak log2 luminance, reduced with max rather than mean, so the
//      scene white estimate survives downsampling.
//  .w  body mean log2(1 + linear view Z). The depth of the surface the
//      adaptation value describes, which is what lets the pooling refuse to
//      average a wall together with the sky behind it.
#define PSDT_FIELD_BODY_LOGY      0
#define PSDT_FIELD_BODY_WEIGHT    1
#define PSDT_FIELD_PEAK_LOGY      2
#define PSDT_FIELD_DEPTH_LOG      3

// Source field. What is too bright to be scene content.
//  .xyz  source energy, linear Rec.709, exposed. Accumulated from SOURCE and
//        GLARE pixels only and stored per channel, so the glare kernel
//        reproduces the source's own colour instead of the scene's average
//        chroma. Stored linearly so that averaging down the mip chain
//        conserves energy.
//  .w    sky weight in [0,1]. Sky is classified from linear view Z rather
//        than from brightness, so an overcast sky that is dimmer than a lamp
//        is still sky.
#define PSDT_SOURCE_R             0
#define PSDT_SOURCE_G             1
#define PSDT_SOURCE_B             2
#define PSDT_SOURCE_SKY_WEIGHT    3

// Illuminant field. What colour the light is.
//  .xyz  weighted illuminant RGB, normalised to luminance 1 before weighting.
//        Estimated per pixel as radiance / albedo over surfaces whose albedo
//        is high enough and neutral enough for the division to mean anything.
//  .w    the weight. Divide .xyz by .w to recover the illuminant; .w == 0
//        means this neighbourhood contained no usable estimator and the
//        consumer should fall back to the frame illuminant or to D65.
#define PSDT_ILLUM_R              0
#define PSDT_ILLUM_G              1
#define PSDT_ILLUM_B              2
#define PSDT_ILLUM_WEIGHT         3

// ---------------------------------------------------------------------------
// AdaptationState slots (R32_SFLOAT 1D texture, PSDT_STATE_SIZE texels)
// ---------------------------------------------------------------------------
// Slots 0-23 are measured and temporally filtered; they carry inertia across
// frames and are the only part of the transform that is not a deterministic
// function of the current frame. Slots 24+ are resolved parameters: pure
// functions of the runtime options and of the measured slots, recomputed every
// frame. The split matters - it is what lets the transform stay deterministic
// given the state while still having a time constant.

// --- measured adaptation state ---
#define PSDT_STATE_ANCHOR_LOGY        0   // Visual anchor. What the viewer is adapted to.
#define PSDT_STATE_SCENE_WHITE_LOGY   1   // Body white, 99.5th percentile of BODY luminance.
#define PSDT_STATE_SCENE_BLACK_LOGY   2   // Body black, 0.5th percentile of BODY luminance.
#define PSDT_STATE_BODY_MEAN_LOGY     3   // Source-excluded mean log luminance, from the field.
#define PSDT_STATE_SOURCE_FRACTION    4   // Fraction of frame classified SOURCE.
#define PSDT_STATE_GLARE_FRACTION     5   // Highlight population: normalised source energy.
#define PSDT_STATE_CONTRAST_RANGE     6   // sceneWhite - sceneBlack, in stops.
#define PSDT_STATE_ADAPT_VELOCITY     7   // d(anchor)/dt, stops per second.
#define PSDT_STATE_CUT_FLAG           8   // 1 on the frame a cut was detected, decaying after.
#define PSDT_STATE_INTENT_NIGHT       9   // Scene intent weights. Soft, sum to 1.
#define PSDT_STATE_INTENT_INDOOR      10
#define PSDT_STATE_INTENT_OUTDOOR     11
#define PSDT_STATE_INTENT_BRIGHT      12  // Explosion / muzzle flash / very high source population.
#define PSDT_STATE_MEDIAN_LOGY        13  // Body histogram median.
#define PSDT_STATE_SALIENCY_LOGY      14  // Centre-weighted median, from the tonemapper histogram.
#define PSDT_STATE_HAS_HISTORY        15  // 0 on the first frame after a reset.
#define PSDT_STATE_SKY_FRACTION       16  // Fraction of frame classified sky, from view depth.
#define PSDT_STATE_FRAME_PEAK_LOGY    17  // Brightest thing in frame.
#define PSDT_STATE_ILLUM_FRAME_R      18  // Frame illuminant, normalised to luminance 1.
#define PSDT_STATE_ILLUM_FRAME_G      19
#define PSDT_STATE_ILLUM_FRAME_B      20
#define PSDT_STATE_ILLUM_CONFIDENCE   21  // How much of the frame produced a usable estimate.
#define PSDT_STATE_CAMERA_ANGULAR     22  // Camera angular speed, radians/second.
#define PSDT_STATE_TRANSITION         23  // 0 still, 1 sweeping, 2 cut. Continuous between.

// --- resolved display model ---
#define PSDT_STATE_DISPLAY_PEAK       24  // Display peak, nits.
#define PSDT_STATE_DISPLAY_BLACK      25  // Display black, nits.
#define PSDT_STATE_DISPLAY_REF_WHITE  26  // Diffuse white / paper white, nits.
#define PSDT_STATE_SURROUND           27  // Viewing environment, nits.
#define PSDT_STATE_DISPLAY_GAMUT      28  // 0 = Rec.709, 1 = DCI-P3, 2 = Rec.2020.
#define PSDT_STATE_DISPLAY_HEADROOM   29  // log2(peak / refWhite). Stops above diffuse white.

// --- resolved perceptual curve. See psdt_curve.slangh. ---
#define PSDT_STATE_CURVE_BLACK        30
#define PSDT_STATE_CURVE_TOE          31
#define PSDT_STATE_CURVE_MID_IN       32  // Scene-referred mid anchor, log2.
#define PSDT_STATE_CURVE_MID_OUT      33  // Display-referred mid anchor, linear [0,1].
#define PSDT_STATE_CURVE_SLOPE        34  // Midtone contrast, display stops per scene stop.
#define PSDT_STATE_CURVE_SHOULDER_IN  35  // Highlight onset, log2 scene, relative to mid.
#define PSDT_STATE_CURVE_SHOULDER_LEN 36  // Shoulder length in scene stops.
#define PSDT_STATE_CURVE_WHITE_IN     37  // Scene white convergence point, log2, relative to mid.

// --- resolved adaptation ---
#define PSDT_STATE_ADAPT_STRENGTH     38
#define PSDT_STATE_ADAPT_WEIGHT_L     39  // Multi-scale pooling weights, sum to 1.
#define PSDT_STATE_ADAPT_WEIGHT_M     40
#define PSDT_STATE_ADAPT_WEIGHT_S     41
#define PSDT_STATE_BUDGET_SHADOW      42  // Adaptation budget, in stops, by luminance role.
#define PSDT_STATE_BUDGET_MID         43
#define PSDT_STATE_BUDGET_HIGHLIGHT   44
#define PSDT_STATE_BUDGET_EMITTER     45
#define PSDT_STATE_DEPTH_SENSITIVITY  46  // How hard the pooling refuses to cross a depth edge.

// --- resolved local contrast restoration ---
#define PSDT_STATE_DETAIL_STRENGTH    47
#define PSDT_STATE_DETAIL_PROTECT     48  // How hard highlights are protected from restoration.
#define PSDT_STATE_DETAIL_KNEE        49  // Detail magnitude, in stops, at which restoration halves.

// --- resolved colour volume ---
#define PSDT_STATE_PRESSURE_LUMA      50  // How early a bright AND saturated colour eases.
#define PSDT_STATE_PRESSURE_CHROMA    51  // Max chroma this display can produce, for normalising.
#define PSDT_STATE_PRESSURE_GAMUT     52  // Resolved soft-knee position. Diagnostic.
#define PSDT_STATE_PRESSURE_CONTEXT   53  // Weight of the local-contrast term in the knee.
#define PSDT_STATE_CHROMA_PRESERVE    54
#define PSDT_STATE_HUE_TRAJECTORY     55
#define PSDT_STATE_WHITE_CONVERGE     56
#define PSDT_STATE_ILLUM_STRENGTH     57  // How far the white point adapts towards the local illuminant.
#define PSDT_STATE_COLOURFULNESS      58
#define PSDT_STATE_LUMA_CONCESSION    59  // Brightness a saturated colour may give up to keep its colour.

// --- resolved glare ---
#define PSDT_STATE_GLARE_STRENGTH     60
#define PSDT_STATE_GLARE_THRESHOLD    61
#define PSDT_STATE_GLARE_FALLOFF      62  // Wide lobe: per-level weight ratio.
#define PSDT_STATE_GLARE_NEARFIELD    63  // Near-field lobe amplitude, relative to the wide lobe.
#define PSDT_STATE_GLARE_LEVELS       64

// --- misc resolved ---
#define PSDT_STATE_PERCEPTUAL_SPACE   65  // 0 = ICtCp, 1 = Jzazbz.
#define PSDT_STATE_FIELD_LEVEL_S      66  // Which mip is A_small / A_medium / A_large.
#define PSDT_STATE_FIELD_LEVEL_M      67
#define PSDT_STATE_FIELD_LEVEL_L      68
#define PSDT_STATE_LEVEL_COUNT        69
#define PSDT_STATE_DEBUG_VIEW         70

// Perceptual space selector values.
static const uint psdtSpaceICtCp  = 0;
static const uint psdtSpaceJzazbz = 1;

// Display gamut selector values.
static const uint psdtGamutRec709 = 0;
static const uint psdtGamutP3     = 1;
static const uint psdtGamutRec2020 = 2;

// Debug views. Written by psdtApply in place of the transformed colour, so
// what is shown is exactly what the transform read - not a re-derivation.
static const uint psdtDebugOff          = 0;
static const uint psdtDebugAdaptation   = 1;   // Pooled adaptation, as stops about the anchor.
static const uint psdtDebugBudget       = 2;   // Adaptation actually used, after the budget clamp.
static const uint psdtDebugRoles        = 3;   // BODY green / SOURCE red / SKY blue.
static const uint psdtDebugSourceEnergy = 4;   // The glare kernel's input, per channel.
static const uint psdtDebugIlluminant   = 5;   // Local illuminant, normalised.
static const uint psdtDebugDepth        = 6;   // Body depth, as the pooling sees it.
static const uint psdtDebugDemand       = 7;   // Colour volume demand. Green in, red out of gamut.
static const uint psdtDebugPressure     = 8;   // Chroma actually given up.
static const uint psdtDebugGlare        = 9;   // Glare contribution alone.
static const uint psdtDebugClipping     = 10;  // Pre-clamp overflow, per channel, before the fit.
static const uint psdtDebugCurveSlope   = 11;  // d(display)/d(scene) at this pixel's base.
static const uint psdtDebugCount        = 12;

// Scene classification of a single pixel. See psdt_analysis.comp.slang.
static const uint psdtRoleBody   = 0;
static const uint psdtRoleSource = 1;
static const uint psdtRoleGlare  = 2;
static const uint psdtRoleSky    = 3;

// ---------------------------------------------------------------------------
// Analysis flag bits (PsdtAnalysisArgs::flags)
// ---------------------------------------------------------------------------
#define PSDT_ANALYSIS_FLAG_AUTO_EXPOSURE   (1u << 0)
#define PSDT_ANALYSIS_FLAG_HAS_HISTORY     (1u << 1)
// Set when the gbuffer inputs are bound to the real thing rather than to a
// stand-in. Without it the classifier falls back to luminance alone, which is
// v0.1 behaviour and is what runs if the gbuffer is unavailable for a frame.
#define PSDT_ANALYSIS_FLAG_RENDERER_SIGNALS (1u << 2)
#define PSDT_ANALYSIS_FLAG_JZAZBZ          (1u << 3)

// ---------------------------------------------------------------------------
// State flag bits (PsdtStateArgs::flags)
// ---------------------------------------------------------------------------
#define PSDT_STATE_FLAG_HAS_HISTORY    (1u << 0)
#define PSDT_STATE_FLAG_SCENE_ADAPTIVE (1u << 1)
#define PSDT_STATE_FLAG_CAMERA_CUT     (1u << 2)
#define PSDT_STATE_FLAG_CAMERA_VALID   (1u << 3)
#define PSDT_STATE_SHIFT_GAMUT         8   // 2 bits
#define PSDT_STATE_SHIFT_SPACE         10  // 1 bit
#define PSDT_STATE_SHIFT_DEBUG         12  // 4 bits
#define PSDT_STATE_SHIFT_LEVELS        16  // 4 bits

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

#define PSDT_ANALYSIS_COLOR_INPUT             0
#define PSDT_ANALYSIS_EXPOSURE_INPUT          1
#define PSDT_ANALYSIS_MOTION_INPUT            2
#define PSDT_ANALYSIS_SURFACE_FLAGS_INPUT     3
#define PSDT_ANALYSIS_VIEW_Z_INPUT            4
#define PSDT_ANALYSIS_ALBEDO_INPUT            5
#define PSDT_ANALYSIS_FIELD_OUTPUT            6
#define PSDT_ANALYSIS_FIELD_HISTORY_INPUT     7
#define PSDT_ANALYSIS_SOURCE_OUTPUT           8
#define PSDT_ANALYSIS_SOURCE_HISTORY_INPUT    9
#define PSDT_ANALYSIS_ILLUM_OUTPUT            10
#define PSDT_ANALYSIS_ILLUM_HISTORY_INPUT     11
#define PSDT_ANALYSIS_BODY_HISTOGRAM_OUTPUT   12
#define PSDT_ANALYSIS_STATE_INPUT             13

#define PSDT_DOWNSAMPLE_FIELD_INPUT           0
#define PSDT_DOWNSAMPLE_FIELD_OUTPUT          1
#define PSDT_DOWNSAMPLE_SOURCE_INPUT          2
#define PSDT_DOWNSAMPLE_SOURCE_OUTPUT         3
#define PSDT_DOWNSAMPLE_ILLUM_INPUT           4
#define PSDT_DOWNSAMPLE_ILLUM_OUTPUT          5

#define PSDT_STATE_HISTOGRAM_INPUT            0
#define PSDT_STATE_BODY_HISTOGRAM_INOUT       1
#define PSDT_STATE_FIELD_INPUT                2
#define PSDT_STATE_SOURCE_INPUT               3
#define PSDT_STATE_ILLUM_INPUT                4
#define PSDT_STATE_STATE_INPUT_OUTPUT         5

// ---------------------------------------------------------------------------
// Constant buffers
// ---------------------------------------------------------------------------

struct PsdtAnalysisArgs
{
  uvec2 colorExtent;
  uvec2 fieldExtent;

  uvec2 motionExtent;
  uvec2 gbufferExtent;

  float exposure;
  uint  flags;
  // Pixels this many stops above the frame's previous body mean are candidates
  // for SOURCE on brightness alone. Absolute thresholds do not work here: a
  // muzzle flash in a dark corridor and a window in a lit room are the same
  // absolute luminance and very different scene events. With the renderer's
  // emissive bit available this is no longer the only evidence, only the
  // strongest of several - see psdt_analysis.comp.slang.
  float sourceThresholdStops;
  // Pixels this many stops above the body mean are GLARE - too bright to be
  // represented as an image at all, so they are represented as spatial extent
  // instead.
  float glareThresholdStops;

  // Blend weight towards the current frame, per frame. Raised towards 1 in
  // proportion to the previous frame's cut flag, which the shader reads from
  // the state texture rather than taking from here - the CPU never reads the
  // state back, and a cut is detected on the GPU.
  float currentWeight;
  float refWhiteNits;
  // Linear view Z above which a pixel is sky rather than geometry. A ray miss
  // writes NRD's missLinearViewZ (500001); the fork clamps far geometry to
  // 200000; ordinary interiors are orders of magnitude below both.
  float skyViewZ;
  // Albedo below this in any channel makes radiance / albedo meaningless, so
  // the pixel contributes nothing to the illuminant estimate.
  float illuminantMinAlbedo;

  float toneCurveMinStops;
  float toneCurveMaxStops;
  uint  pad0;
  uint  pad1;
};

struct PsdtDownsampleArgs
{
  uvec2 srcExtent;
  uvec2 dstExtent;
};

// Exactly 128 bytes, which is MaxPushConstantSize and also the Vulkan
// guaranteed minimum. Every enum, boolean and small integer is packed into
// `flags` for that reason; anything else that needs adding has to displace
// something or become a value resolved from what is already here.
struct PsdtStateArgs
{
  // --- histogram domain, shared with the native tonemapper ---
  float toneCurveMinStops;
  float toneCurveMaxStops;
  float deltaTimeSeconds;
  uint  flags;

  // --- display model ---
  float displayPeakNits;
  float displayBlackNits;
  float displayRefWhiteNits;
  float surroundNits;

  // --- adaptation ---
  float sourceExclusion;
  float localStrength;
  float contrastBudget;
  float depthSensitivity;

  float adaptationSpeedUp;
  float adaptationSpeedDown;
  float cameraAngularSpeed;    // radians / second
  float cameraTranslationSpeed; // world units / second, normalised by the scene scale

  // --- curve ---
  float midtoneContrast;
  float shadowDepth;
  float highlightRolloff;
  float detailStrength;

  float detailProtect;
  float detailKnee;

  // --- colour volume ---
  float chromaPreservation;
  float hueTrajectory;

  float pressureLuma;
  float pressureContext;
  float whiteConvergence;
  float illuminantStrength;

  float colourfulness;
  float luminanceConcession;

  // --- glare ---
  float glareStrength;
  float glareThresholdStops;
};

#endif  // PSDT_H
