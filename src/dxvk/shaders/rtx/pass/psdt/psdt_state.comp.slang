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

// ===========================================================================
// PSDT AdaptationState: histogram + adaptation pyramid -> the frame's state
// ===========================================================================
//
// One workgroup, one thread per histogram bucket. It runs between the
// tonemapper's histogram pass and its tone curve pass, because the tone curve
// pass zeroes the histogram when it is done with it.
//
// Two things happen here, and the split between them is deliberate.
//
// Measured state (slots 0..15) carries inertia. It is the only part of the
// whole transform that depends on previous frames. Exposure that reacts
// directly to a per-frame histogram pumps, because a path-traced histogram is
// noisy and because a player turning around changes the histogram completely
// in one frame without the scene having changed at all.
//
// Resolved parameters (slots 16+) are a pure function of the options and of
// the measured state, recomputed from scratch every frame. So the transform
// is deterministic given the state, while the state has a time constant. Get
// this the other way round - inertia in the parameters, or none in the state
// - and the image either lags or flickers.
//
// The anchor
// ----------
// The single most consequential number this pass produces. It answers "what
// part of this scene is the viewer visually adapted to", which is not what a
// histogram mean answers.
//
//   Y_anchor = f(histogram, saliency, source exclusion, temporal, intent)
//
// The histogram already carries a centre weight from the tonemapper's own
// metering, so it contributes saliency for free. The field contributes the
// source-excluded body mean. Blending towards the latter by
// rtx.tonemap.psdt.sourceExclusion is what makes a corridor with a sunlit
// window expose for the corridor - the window is a highlight event, the
// corridor is the adaptation region, and only one of them is scene.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx/pass/psdt/psdt_perceptual_space.slangh"

layout(binding = PSDT_STATE_HISTOGRAM_INPUT)
RWTexture1D<uint> InHistogram;

layout(binding = PSDT_STATE_FIELD_INPUT)
Sampler2D InField;

layout(binding = PSDT_STATE_STATE_INPUT_OUTPUT)
RWTexture1D<float> InOutState;

layout(push_constant)
ConstantBuffer<PsdtStateArgs> cb;

static const uint kBuckets = EXPOSURE_HISTOGRAM_SIZE;
static const float kHistogramScale = 256.0;  // kHistogramFixedPointScale
static const float kMiddleGreyLog = -2.4739311883324122;

groupshared float g_hist[kBuckets];
groupshared float g_scan[kBuckets];
groupshared uint  g_percentile[3];   // 0.5th, 50th, 99.5th

float psdtBucketToLog(uint bucket)
{
  return lerp(cb.toneCurveMinStops, cb.toneCurveMaxStops,
              float(bucket) / float(kBuckets - 1));
}

[shader("compute")]
[numthreads(PSDT_STATE_GROUP_SIZE, 1, 1)]
void main(uint i : SV_GroupIndex)
{

  g_hist[i] = float(InHistogram[int(i)]) / kHistogramScale;
  if (i < 3) { g_percentile[i] = 0; }

  GroupMemoryBarrierWithGroupSync();

  // Inclusive prefix sum over the histogram, Hillis-Steele. Eight steps for
  // 256 buckets; the CDF is what every percentile below is read off.
  g_scan[i] = g_hist[i];
  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint offset = 1; offset < kBuckets; offset <<= 1)
  {
    const float addend = (i >= offset) ? g_scan[i - offset] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    g_scan[i] += addend;
    GroupMemoryBarrierWithGroupSync();
  }

  const float total = g_scan[kBuckets - 1];

  if (total > 1e-4)
  {
    const float previous = (i == 0) ? 0.0 : g_scan[i - 1];
    const float current = g_scan[i];

    // Each thread owns the bucket its own CDF interval covers, so the three
    // percentile searches are a single comparison each rather than a scan.
    const float t005 = total * 0.005;
    const float t500 = total * 0.500;
    const float t995 = total * 0.995;

    if (previous < t005 && current >= t005) { g_percentile[0] = i; }
    if (previous < t500 && current >= t500) { g_percentile[1] = i; }
    if (previous < t995 && current >= t995) { g_percentile[2] = i; }
  }

  GroupMemoryBarrierWithGroupSync();

  if (i != 0)
  {
    return;
  }

  // =====================================================================
  // Everything below is single threaded. It is roughly a hundred scalar
  // operations on numbers that have already been reduced, so a wider
  // reduction would buy nothing and would cost the barriers to coordinate.
  // =====================================================================

  const float sceneBlackLog = psdtBucketToLog(g_percentile[0]);
  const float medianLog = psdtBucketToLog(g_percentile[1]);
  const float sceneWhiteLog = psdtBucketToLog(g_percentile[2]);
  const float contrastRange = max(sceneWhiteLog - sceneBlackLog, 0.0);

  // Coarsest pyramid level, sampled at the frame centre: the whole frame
  // reduced to one texel. .x is the source-excluded geometric mean of body
  // luminance, .y the body fraction, .z the mean source energy, .w the peak.
  const float coarse = max(float(cb.levelCount) - 1.0, 0.0);
  const float4 frame = InField.SampleLevel(float2(0.5, 0.5), coarse);

  const float bodyMeanLog = frame.x;
  const float bodyFraction = saturate(frame.y);
  const float sourceFraction = saturate(1.0 - bodyFraction);
  const float sourceEnergy = max(frame.z, 0.0);
  const float framePeakLog = frame.w;

  // --- the anchor -------------------------------------------------------
  // The histogram median is saliency weighted (the tonemapper's histogram
  // applies a centre weight when it accumulates) but knows nothing about what
  // is a light and what is a wall. The field's body mean is the opposite.
  // sourceExclusion picks the blend; at 1.0 the anchor ignores light sources
  // entirely and exposes purely for surfaces.
  float anchorTarget = lerp(medianLog, bodyMeanLog, saturate(cb.sourceExclusion));

  // --- scene intent (§ statistics, not inference) -----------------------
  // Soft classification from the numbers already in hand. Nothing here is
  // learned or thresholded hard; the weights are smooth so a scene drifting
  // between two classes drifts between two parameter sets rather than
  // switching.
  const float relAnchor = anchorTarget - kMiddleGreyLog;

  float intentNight  = 1.0 - smoothstep(-3.0, 0.0, relAnchor);
  float intentBright = saturate(smoothstep(0.02, 0.30, sourceFraction)
                              + smoothstep(2.0, 32.0, sourceEnergy)) ;
  // Outdoor reads as a bright anchor with a wide range and a substantial
  // population well above it - sky. Indoor is what is left.
  float intentOutdoor = smoothstep(-1.0, 1.5, relAnchor)
                      * smoothstep(6.0, 12.0, contrastRange)
                      * smoothstep(0.01, 0.15, sourceFraction);
  float intentIndoor = max(1.0 - intentNight - intentOutdoor - intentBright, 0.0);

  const float intentSum = max(intentNight + intentIndoor + intentOutdoor + intentBright, 1e-3);
  intentNight  /= intentSum;
  intentIndoor /= intentSum;
  intentOutdoor /= intentSum;
  intentBright /= intentSum;

  // --- temporal adaptation ---------------------------------------------
  const float previousAnchor = InOutState[PSDT_STATE_ANCHOR_LOGY];
  const float previousWhite = InOutState[PSDT_STATE_SCENE_WHITE_LOGY];
  const float previousBlack = InOutState[PSDT_STATE_SCENE_BLACK_LOGY];
  const bool hasHistory = (cb.hasHistory != 0) && (InOutState[PSDT_STATE_HAS_HISTORY] > 0.5);

  // Cut detection. A player turning 180 degrees, a teleport, a respawn and a
  // scripted cut all look the same from here: the whole luminance description
  // of the frame moves at once. Easing through that takes seconds and reads
  // as the image washing out, so the state re-anchors instead.
  const float anchorJump = abs(anchorTarget - previousAnchor);
  const float whiteJump = abs(sceneWhiteLog - previousWhite);
  const float blackJump = abs(sceneBlackLog - previousBlack);
  const bool isCut = hasHistory
                  && (anchorJump > 2.5 || (whiteJump > 3.0 && blackJump > 3.0));

  // Asymmetric, in the direction human vision actually is: light adaptation
  // is fast, dark adaptation is slow. Going from a dark corridor into
  // daylight should settle quickly; the reverse should take a moment, which
  // is also what makes an interior read as dark when you step into it.
  const float dt = clamp(cb.deltaTimeSeconds, 0.0, 0.25);
  const float speed = (anchorTarget > previousAnchor) ? cb.adaptationSpeedUp : cb.adaptationSpeedDown;
  float alpha = 1.0 - exp(-max(speed, 0.01) * dt);
  if (!hasHistory || isCut) { alpha = 1.0; }

  const float anchor = lerp(previousAnchor, anchorTarget, alpha);
  const float sceneWhite = lerp(previousWhite, sceneWhiteLog, alpha);
  const float sceneBlack = lerp(previousBlack, sceneBlackLog, alpha);
  const float velocity = (dt > 1e-4) ? ((anchor - previousAnchor) / dt) : 0.0;

  // Cut flag decays rather than clearing, so the analysis pass keeps its
  // history weight raised for a few frames after a cut while the reprojection
  // is still rebuilding.
  const float previousCut = InOutState[PSDT_STATE_CUT_FLAG];
  const float cutFlag = isCut ? 1.0 : max(previousCut - dt * 4.0, 0.0);

  InOutState[PSDT_STATE_ANCHOR_LOGY]      = anchor;
  InOutState[PSDT_STATE_SCENE_WHITE_LOGY] = sceneWhite;
  InOutState[PSDT_STATE_SCENE_BLACK_LOGY] = sceneBlack;
  InOutState[PSDT_STATE_BODY_MEAN_LOGY]   = bodyMeanLog;
  InOutState[PSDT_STATE_SOURCE_FRACTION]  = sourceFraction;
  InOutState[PSDT_STATE_GLARE_FRACTION]   = saturate(sourceEnergy / (sourceEnergy + 8.0));
  InOutState[PSDT_STATE_CONTRAST_RANGE]   = sceneWhite - sceneBlack;
  InOutState[PSDT_STATE_ADAPT_VELOCITY]   = velocity;
  InOutState[PSDT_STATE_CUT_FLAG]         = cutFlag;
  InOutState[PSDT_STATE_INTENT_NIGHT]     = intentNight;
  InOutState[PSDT_STATE_INTENT_INDOOR]    = intentIndoor;
  InOutState[PSDT_STATE_INTENT_OUTDOOR]   = intentOutdoor;
  InOutState[PSDT_STATE_INTENT_BRIGHT]    = intentBright;
  InOutState[PSDT_STATE_MEDIAN_LOGY]      = medianLog;
  InOutState[PSDT_STATE_SALIENCY_LOGY]    = medianLog;
  InOutState[PSDT_STATE_HAS_HISTORY]      = 1.0;

  // =====================================================================
  // Parameter resolution
  // =====================================================================

  const float refWhite = max(cb.displayRefWhiteNits, 1.0);
  const float peakLog = log2(max(cb.displayPeakNits, refWhite) / refWhite);
  const float blackLinear = max(cb.displayBlackNits / refWhite, 1e-7);
  const float blackLog = log2(blackLinear);

  InOutState[PSDT_STATE_DISPLAY_PEAK]      = cb.displayPeakNits;
  InOutState[PSDT_STATE_DISPLAY_BLACK]     = cb.displayBlackNits;
  InOutState[PSDT_STATE_DISPLAY_REF_WHITE] = refWhite;
  InOutState[PSDT_STATE_SURROUND]          = cb.surroundNits;
  InOutState[PSDT_STATE_DISPLAY_GAMUT]     = float(cb.displayGamut);
  InOutState[PSDT_STATE_DISPLAY_HEADROOM]  = peakLog;

  // Scene-intent modulation. Every term below is a small nudge on a
  // hand-chosen default, not a preset switch - the point is that the same
  // transform behaves differently because the scene is different, rather than
  // because someone changed a setting.
  const float adaptive = (cb.sceneAdaptive != 0) ? 1.0 : 0.0;

  // Surround: a dark viewing environment wants a slightly lower mid and more
  // contrast; a bright one wants the mid lifted or the shadows disappear into
  // the room's own reflected light.
  const float surroundRatio = saturate(cb.surroundNits / max(0.5 * refWhite, 1.0));
  const float midOut = 0.18 * lerp(0.92, 1.16, surroundRatio);

  // Midtone contrast. Night scenes get more slope (shadow separation is what
  // a dark scene is made of); daylight gets less, because the range that has
  // to fit is larger.
  const float slope = max(cb.midtoneContrast
                        * lerp(1.0, 1.0 + 0.18 * intentNight - 0.12 * intentOutdoor, adaptive),
                        0.30);

  const float midOutLog = log2(max(midOut, 1e-6));

  // Toe onset. shadowDepth is the fraction of the available display range
  // below the mid anchor that the linear segment is allowed to use before the
  // toe takes over. Night scenes push it lower, which lengthens the linear
  // shadow region and separates dark detail instead of crushing it.
  const float shadowRange = max(midOutLog - blackLog, 1.0) / slope;
  const float shadowDepth = saturate(cb.shadowDepth * lerp(1.0, 1.0 + 0.25 * intentNight, adaptive));
  const float toeStart = -shadowDepth * shadowRange;

  // Shoulder onset. A frame with a lot of source energy gets an earlier, and
  // therefore longer and softer, shoulder - the highlights are the subject,
  // and what does not fit is handed to the glare model rather than clipped.
  const float highlightRange = max(peakLog - midOutLog, 1.0) / slope;
  const float rolloff = saturate(cb.highlightRolloff * lerp(1.0, 1.0 - 0.30 * intentBright, adaptive));
  const float shoulderStart = rolloff * highlightRange;

  // Derived: stops from the shoulder onset to within 1/e of the peak. The
  // colour-volume stage uses it as the luminance axis of its pressure metric,
  // so it has to describe the curve that is actually running.
  const float yS = midOutLog + slope * shoulderStart;
  const float shoulderLen = max(peakLog - yS, 1e-3) / max(slope, 1e-3);

  // White convergence: three e-foldings is ~95% of the way to peak. Capped by
  // the scene's own white so a frame that contains nothing bright does not
  // reserve display range for highlights it does not have.
  const float whiteIn = min(shoulderStart + shoulderLen * 3.0,
                            max(sceneWhite - anchor, shoulderStart + 1.0));

  InOutState[PSDT_STATE_CURVE_BLACK]        = blackLog;
  InOutState[PSDT_STATE_CURVE_TOE]          = toeStart;
  InOutState[PSDT_STATE_CURVE_MID_IN]       = anchor;
  InOutState[PSDT_STATE_CURVE_MID_OUT]      = midOut;
  InOutState[PSDT_STATE_CURVE_SLOPE]        = slope;
  InOutState[PSDT_STATE_CURVE_SHOULDER_IN]  = shoulderStart;
  InOutState[PSDT_STATE_CURVE_SHOULDER_LEN] = shoulderLen;
  InOutState[PSDT_STATE_CURVE_WHITE_IN]     = whiteIn;

  // --- adaptation -------------------------------------------------------
  // There is nothing to gain from local adaptation in a scene that has little
  // range, and everything to lose: with no range to redistribute, all local
  // adaptation can do is equalise, which is the flat look. So it fades out
  // when the frame's own contrast range is small.
  const float rangeGate = smoothstep(3.0, 8.0, contrastRange);
  InOutState[PSDT_STATE_ADAPT_STRENGTH] = saturate(cb.localStrength) * lerp(1.0, rangeGate, adaptive);

  // Spatial pooling weights. A wide, uniform scene is governed by its coarse
  // scale; a cluttered high-contrast one has real information at finer
  // scales and pooling it all coarsely would throw that away.
  const float fineness = saturate((contrastRange - 6.0) / 8.0) * adaptive;
  InOutState[PSDT_STATE_ADAPT_WEIGHT_L] = lerp(0.60, 0.30, fineness);
  InOutState[PSDT_STATE_ADAPT_WEIGHT_M] = 0.30;
  InOutState[PSDT_STATE_ADAPT_WEIGHT_S] = lerp(0.10, 0.40, fineness);

  // The contrast budget. This is the hierarchy the scene is not allowed to
  // lose: shadows may move a long way to become visible, midtones a little,
  // highlights barely, and an emitter essentially not at all.
  const float budget = max(cb.contrastBudget, 0.0);
  InOutState[PSDT_STATE_BUDGET_SHADOW]    = budget * 2.00;
  InOutState[PSDT_STATE_BUDGET_MID]       = budget * 0.55;
  InOutState[PSDT_STATE_BUDGET_HIGHLIGHT] = budget * 0.25;
  InOutState[PSDT_STATE_BUDGET_EMITTER]   = budget * 0.05;

  // --- local contrast ---------------------------------------------------
  InOutState[PSDT_STATE_DETAIL_STRENGTH] = saturate(cb.detailStrength);
  InOutState[PSDT_STATE_DETAIL_PROTECT]  = 1.0;

  // --- colour volume ----------------------------------------------------
  // How early a bright *and* saturated colour begins easing towards the gamut
  // boundary, relative to where an ordinary colour would. Only the product of
  // brightness and chroma moves this, so a bright neutral is never affected.
  InOutState[PSDT_STATE_PRESSURE_LUMA]   = 0.50;
  InOutState[PSDT_STATE_PRESSURE_CHROMA] = psdtChromaReference(cb.perceptualSpace);
  // Reported for diagnostics: where the soft knee actually sits for an ordinary
  // colour, given chromaPreservation.
  InOutState[PSDT_STATE_PRESSURE_GAMUT]  = 1.0 - 1.0 / (1.0 + max(cb.chromaPreservation, 0.05));
  InOutState[PSDT_STATE_CHROMA_PRESERVE] = max(cb.chromaPreservation, 0.05);
  InOutState[PSDT_STATE_HUE_TRAJECTORY]  = saturate(cb.hueTrajectory);
  InOutState[PSDT_STATE_WHITE_CONVERGE]  = 2.0;
  InOutState[PSDT_STATE_SPATIAL_WHITE]   = saturate(cb.spatialWhiteStrength);
  InOutState[PSDT_STATE_COLOURFULNESS]   = max(cb.colourfulness, 0.0);
  InOutState[PSDT_STATE_LUMA_CONCESSION] = saturate(cb.luminanceConcession);

  // --- glare ------------------------------------------------------------
  // The threshold is anchored to the adaptation state, not to an absolute
  // value: whether a light glares depends on what the viewer is adapted to,
  // which is the entire reason a car headlight is blinding at night and
  // invisible at noon.
  InOutState[PSDT_STATE_GLARE_STRENGTH]  = max(cb.glareStrength, 0.0);
  InOutState[PSDT_STATE_GLARE_THRESHOLD] = anchor + cb.glareThresholdStops;
  InOutState[PSDT_STATE_GLARE_FALLOFF]   = clamp(cb.glareFalloff, 0.05, 0.95);
  InOutState[PSDT_STATE_GLARE_LEVELS]    = float(cb.levelCount);

  // --- misc -------------------------------------------------------------
  InOutState[PSDT_STATE_PERCEPTUAL_SPACE] = float(cb.perceptualSpace);
  InOutState[PSDT_STATE_FIELD_LEVEL_S]    = 0.0;
  InOutState[PSDT_STATE_FIELD_LEVEL_M]    = min(2.0, coarse);
  InOutState[PSDT_STATE_FIELD_LEVEL_L]    = min(4.0, coarse);
  InOutState[PSDT_STATE_LEVEL_COUNT]      = float(cb.levelCount);
  InOutState[PSDT_STATE_DEBUG_VIEW]       = float(cb.debugView);
  InOutState[PSDT_STATE_FRAME_PEAK_LOGY]  = framePeakLog;
}
