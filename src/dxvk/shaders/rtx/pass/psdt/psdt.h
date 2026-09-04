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
// reconstructed scene radiance into the image a viewer adapted to this scene,
// looking at this display, would find most plausible.
//
// The reason it exists as a separate thing rather than as another entry in
// tonemap_operators.slangh is a contract. Every operator in that file - Hable,
// Psycho17, GT7 - is a pure function of one pixel. That is the correct shape
// for a curve and the wrong shape for a display transform, because the two
// most important inputs to "what should this pixel look like" are not in the
// pixel: what the viewer is adapted to, and how much room the display has
// left. So PSDT splits the job four ways and gives each stage exactly one
// responsibility:
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
// Structure
// ---------
//   psdt_analysis.comp    full-res HDR -> adaptation field mip 0, per 16x16
//                         block: body log luminance, body weight, source
//                         energy, block peak. Classifies BODY / SOURCE /
//                         GLARE and accumulates temporally.
//   psdt_downsample.comp  field mip k -> mip k+1. The mip chain IS the
//                         multi-scale adaptation pyramid (A_small / A_medium
//                         / A_large) and simultaneously the glare kernel.
//   psdt_state.comp       histogram + coarse field -> AdaptationState. One
//                         workgroup. Source-aware anchor, scene white, scene
//                         intent classification, asymmetric temporal inertia,
//                         cut detection, and resolution of every runtime
//                         parameter into the state texture.
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
#define PSDT_STATE_SIZE           64

// Histogram is shared with the native tonemapper (EXPOSURE_HISTOGRAM_SIZE
// buckets over [toneCurveMinStops, toneCurveMaxStops]); psdt_state runs as a
// single workgroup of this width so one thread owns one bucket.
#define PSDT_STATE_GROUP_SIZE     256

// ---------------------------------------------------------------------------
// Adaptation field channel layout (RGBA16F)
// ---------------------------------------------------------------------------
// .x  body log2 luminance, exposed, relative to middle grey. Weighted mean
//     over BODY-classified pixels only; the weight is .y. Source pixels are
//     excluded rather than clamped, because a clamped sun still drags the
//     mean and a excluded one does not.
// .y  body weight in [0,1] - what fraction of the block was ordinary surface.
//     A block that is entirely lamp has .y == 0 and contributes no adaptation.
// .z  source linear luminance, exposed. Energy of the SOURCE/GLARE pixels,
//     stored linearly so that averaging down the mip chain conserves it. This
//     is the glare kernel's input.
// .w  block peak log2 luminance, reduced with max rather than mean, so the
//     scene white estimate survives downsampling.
#define PSDT_FIELD_BODY_LOGY      0
#define PSDT_FIELD_BODY_WEIGHT    1
#define PSDT_FIELD_SOURCE_ENERGY  2
#define PSDT_FIELD_PEAK_LOGY      3

// ---------------------------------------------------------------------------
// AdaptationState slots (R32_SFLOAT 1D texture, PSDT_STATE_SIZE texels)
// ---------------------------------------------------------------------------
// Slots 0-15 are measured and temporally filtered; they carry inertia across
// frames and are the only part of the transform that is not a deterministic
// function of the current frame. Slots 16+ are resolved parameters: pure
// functions of the runtime options and of slots 0-15, recomputed every frame.
// The split matters - it is what lets the transform stay deterministic given
// the state while still having a time constant.

// Measured adaptation state.
#define PSDT_STATE_ANCHOR_LOGY        0   // Visual anchor. What the viewer is adapted to.
#define PSDT_STATE_SCENE_WHITE_LOGY   1   // Scene white, ~99.5th percentile of body luminance.
#define PSDT_STATE_SCENE_BLACK_LOGY   2   // Scene black, ~0.5th percentile.
#define PSDT_STATE_BODY_MEAN_LOGY     3   // Source-excluded mean log luminance.
#define PSDT_STATE_SOURCE_FRACTION    4   // Fraction of frame classified SOURCE.
#define PSDT_STATE_GLARE_FRACTION     5   // Fraction classified GLARE (source, and extreme).
#define PSDT_STATE_CONTRAST_RANGE     6   // sceneWhite - sceneBlack, in stops.
#define PSDT_STATE_ADAPT_VELOCITY     7   // d(anchor)/dt, stops per second.
#define PSDT_STATE_CUT_FLAG           8   // 1 on the frame a cut was detected, decaying after.
#define PSDT_STATE_INTENT_NIGHT       9   // Scene intent weights. Soft, sum to 1.
#define PSDT_STATE_INTENT_INDOOR      10
#define PSDT_STATE_INTENT_OUTDOOR     11
#define PSDT_STATE_INTENT_BRIGHT      12  // Explosion / muzzle flash / very high source population.
#define PSDT_STATE_MEDIAN_LOGY        13  // Histogram median, unweighted.
#define PSDT_STATE_SALIENCY_LOGY      14  // Centre-weighted anchor candidate.
#define PSDT_STATE_HAS_HISTORY        15  // 0 on the first frame after a reset.

// Resolved display model.
#define PSDT_STATE_DISPLAY_PEAK       16  // Display peak, nits.
#define PSDT_STATE_DISPLAY_BLACK      17  // Display black, nits.
#define PSDT_STATE_DISPLAY_REF_WHITE  18  // Diffuse white / paper white, nits.
#define PSDT_STATE_SURROUND           19  // Viewing environment, nits.
#define PSDT_STATE_DISPLAY_GAMUT      20  // 0 = Rec.709, 1 = DCI-P3, 2 = Rec.2020.
#define PSDT_STATE_DISPLAY_HEADROOM   21  // log2(peak / refWhite). Stops above diffuse white.

// Resolved perceptual curve. See psdt_curve.slangh for what each one does.
#define PSDT_STATE_CURVE_BLACK        22
#define PSDT_STATE_CURVE_TOE          23
#define PSDT_STATE_CURVE_MID_IN       24  // Scene-referred mid anchor, log2.
#define PSDT_STATE_CURVE_MID_OUT      25  // Display-referred mid anchor, linear [0,1].
#define PSDT_STATE_CURVE_SLOPE        26  // Midtone contrast, display stops per scene stop.
#define PSDT_STATE_CURVE_SHOULDER_IN  27  // Highlight onset, log2 scene, relative to mid.
#define PSDT_STATE_CURVE_SHOULDER_LEN 28  // Shoulder length in scene stops.
#define PSDT_STATE_CURVE_WHITE_IN     29  // Scene white convergence point, log2, relative to mid.

// Resolved adaptation parameters.
#define PSDT_STATE_ADAPT_STRENGTH     30
#define PSDT_STATE_ADAPT_WEIGHT_L     31  // Multi-scale pooling weights, sum to 1.
#define PSDT_STATE_ADAPT_WEIGHT_M     32
#define PSDT_STATE_ADAPT_WEIGHT_S     33
#define PSDT_STATE_BUDGET_SHADOW      34  // Adaptation budget, in stops, by luminance role.
#define PSDT_STATE_BUDGET_MID         35
#define PSDT_STATE_BUDGET_HIGHLIGHT   36
#define PSDT_STATE_BUDGET_EMITTER     37

// Resolved local contrast restoration.
#define PSDT_STATE_DETAIL_STRENGTH    38
#define PSDT_STATE_DETAIL_PROTECT     39  // How hard highlights are protected from restoration.

// Resolved colour volume mapping.
#define PSDT_STATE_PRESSURE_LUMA      40  // How early a bright AND saturated colour eases.
#define PSDT_STATE_PRESSURE_CHROMA    41  // Max chroma this display can produce, for normalising.
#define PSDT_STATE_PRESSURE_GAMUT     42  // Resolved soft-knee position. Diagnostic.
#define PSDT_STATE_CHROMA_PRESERVE    43
#define PSDT_STATE_HUE_TRAJECTORY     44
#define PSDT_STATE_WHITE_CONVERGE     45
#define PSDT_STATE_SPATIAL_WHITE      46
#define PSDT_STATE_COLOURFULNESS      47
#define PSDT_STATE_LUMA_CONCESSION    59  // Brightness a saturated colour may give up
                                          // to keep its colour.

// Resolved glare.
#define PSDT_STATE_GLARE_STRENGTH     48
#define PSDT_STATE_GLARE_THRESHOLD    49
#define PSDT_STATE_GLARE_FALLOFF      50
#define PSDT_STATE_GLARE_LEVELS       51

// Misc resolved.
#define PSDT_STATE_PERCEPTUAL_SPACE   52  // 0 = ICtCp, 1 = Jzazbz.
#define PSDT_STATE_FIELD_LEVEL_S      53  // Which mip is A_small / A_medium / A_large.
#define PSDT_STATE_FIELD_LEVEL_M      54
#define PSDT_STATE_FIELD_LEVEL_L      55
#define PSDT_STATE_LEVEL_COUNT        56
#define PSDT_STATE_DEBUG_VIEW         57
#define PSDT_STATE_FRAME_PEAK_LOGY    58  // Brightest thing in frame. An HDR output
                                          // path would pick its display peak from this.

// Perceptual space selector values.
static const uint psdtSpaceICtCp  = 0;
static const uint psdtSpaceJzazbz = 1;

// Display gamut selector values.
static const uint psdtGamutRec709 = 0;
static const uint psdtGamutP3     = 1;
static const uint psdtGamutRec2020 = 2;

// Scene classification of a single pixel. See psdt_analysis.comp.slang.
static const uint psdtRoleBody   = 0;
static const uint psdtRoleSource = 1;
static const uint psdtRoleGlare  = 2;

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

#define PSDT_ANALYSIS_COLOR_INPUT             0
#define PSDT_ANALYSIS_EXPOSURE_INPUT          1
#define PSDT_ANALYSIS_MOTION_INPUT            2
#define PSDT_ANALYSIS_FIELD_OUTPUT            3
#define PSDT_ANALYSIS_FIELD_HISTORY_INPUT     4
#define PSDT_ANALYSIS_CHROMA_OUTPUT           5
#define PSDT_ANALYSIS_CHROMA_HISTORY_INPUT    6
#define PSDT_ANALYSIS_STATE_INPUT             7

#define PSDT_DOWNSAMPLE_FIELD_INPUT           0
#define PSDT_DOWNSAMPLE_FIELD_OUTPUT          1
#define PSDT_DOWNSAMPLE_CHROMA_INPUT          2
#define PSDT_DOWNSAMPLE_CHROMA_OUTPUT         3

#define PSDT_STATE_HISTOGRAM_INPUT            0
#define PSDT_STATE_FIELD_INPUT                1
#define PSDT_STATE_STATE_INPUT_OUTPUT         2

// ---------------------------------------------------------------------------
// Constant buffers
// ---------------------------------------------------------------------------

struct PsdtAnalysisArgs
{
  uvec2 colorExtent;
  uvec2 fieldExtent;

  uvec2 motionExtent;
  float exposure;
  uint enableAutoExposure;

  // Pixels this many stops above the frame's previous body mean are candidates
  // for SOURCE. Absolute thresholds do not work here: a muzzle flash in a dark
  // corridor and a window in a lit room are the same absolute luminance and
  // very different scene events.
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
  uint hasHistory;

  uint  perceptualSpace;
  float refWhiteNits;
  uint  pad0;
  uint  pad1;
};

struct PsdtDownsampleArgs
{
  uvec2 srcExtent;
  uvec2 dstExtent;
};

// Exactly 128 bytes, which is MaxPushConstantSize and also the Vulkan
// guaranteed minimum. There is no room to add a parameter without removing one;
// `fieldExtent` is the reserved slot to spend if that day comes (psdt_state
// derives everything it needs from the field's own mip chain and does not
// currently read it). Anything larger has to become a resolved value written
// into the state texture instead.
struct PsdtStateArgs
{
  // --- histogram domain, shared with the native tonemapper ---
  float toneCurveMinStops;
  float toneCurveMaxStops;
  float deltaTimeSeconds;
  uint  hasHistory;

  uvec2 fieldExtent;
  uint  levelCount;
  uint  debugView;

  // --- display model ---
  float displayPeakNits;
  float displayBlackNits;
  float displayRefWhiteNits;
  float surroundNits;

  uint  displayGamut;
  uint  perceptualSpace;
  uint  sceneAdaptive;
  float sourceExclusion;

  // --- adaptation ---
  float localStrength;
  float contrastBudget;
  float adaptationSpeedUp;
  float adaptationSpeedDown;

  // --- curve ---
  float midtoneContrast;
  float shadowDepth;
  float highlightRolloff;
  float detailStrength;

  // --- colour volume ---
  float chromaPreservation;
  float hueTrajectory;
  float spatialWhiteStrength;
  float colourfulness;

  // --- glare ---
  float glareStrength;
  float glareThresholdStops;
  float glareFalloff;
  float luminanceConcession;
};

#endif  // PSDT_H
