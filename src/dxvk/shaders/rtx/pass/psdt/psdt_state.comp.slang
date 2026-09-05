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
// PSDT AdaptationState: two histograms + the pyramid -> the frame's state
// ===========================================================================
//
// One workgroup, 256 threads. It runs between the tonemapper's histogram pass
// and its tone curve pass, because the tone curve pass zeroes the histogram
// when it is done with it.
//
// Two things happen here, and the split between them is deliberate.
//
// Measured state (slots 0..23) carries inertia. It is the only part of the
// whole transform that depends on previous frames. Exposure that reacts
// directly to a per-frame histogram pumps, because a path-traced histogram is
// noisy and because a player turning around changes the histogram completely
// in one frame without the scene having changed at all.
//
// Resolved parameters (slots 24+) are a pure function of the options and of
// the measured state, recomputed from scratch every frame. So the transform
// is deterministic given the state, while the state has a time constant. Get
// this the other way round - inertia in the parameters, or none in the state
// - and the image either lags or flickers.
//
// Two histograms, and why
// -----------------------
// The tonemapper's own histogram applies a radial centre weight as it
// accumulates, and counts every pixel regardless of what it is. That makes it
// an excellent saliency estimate - it answers "how bright is what the player
// is looking at" - and a poor description of the scene's surfaces, because a
// muzzle flash is in it at full strength.
//
// v0.1 read scene white, scene black and the cut detector off that histogram
// while carefully excluding sources from the anchor, so a flash could not
// move the exposure but could still move the definition of white and trip the
// cut detector. That is now fixed by having psdt_analysis accumulate a second
// histogram weighted by body classification, and reading the two for the two
// different questions they actually answer:
//
//   body histogram       scene white, scene black, scene median, cut detection
//   tonemapper histogram saliency
//
// The anchor
// ----------
// The single most consequential number this pass produces. It answers "what
// part of this scene is the viewer visually adapted to", which is not what a
// histogram mean answers.
//
//   Y_anchor = f(body distribution, saliency, source exclusion, intent, time)
//
// All five terms are present. v0.1 had three: it wrote the saliency slot with
// a copy of the median, and computed scene intent *after* the anchor so the
// intent never reached it.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx/pass/psdt/psdt_perceptual_space.slangh"

[[vk::image_format("r32ui")]]
layout(r32ui, binding = PSDT_STATE_BINDING_HISTOGRAM)
RWTexture1D<uint> InHistogram;

[[vk::image_format("r32ui")]]
layout(r32ui, binding = PSDT_STATE_BINDING_BODY_HISTOGRAM)
RWTexture1D<uint> InOutBodyHistogram;

layout(binding = PSDT_STATE_BINDING_FIELD)
Sampler2D InField;

layout(binding = PSDT_STATE_BINDING_SOURCE)
Sampler2D InSource;

layout(binding = PSDT_STATE_BINDING_ILLUM)
Sampler2D InIlluminant;

layout(binding = PSDT_STATE_BINDING_STATE)
RWTexture1D<float> InOutState;

layout(push_constant)
ConstantBuffer<PsdtStateArgs> cb;

static const uint kBuckets = EXPOSURE_HISTOGRAM_SIZE;
static const uint kBodyBuckets = PSDT_BODY_HISTOGRAM_SIZE;
static const float kHistogramScale = 256.0;  // kHistogramFixedPointScale
static const float kBodyScale = 4096.0;      // kFxWeight in psdt_analysis
static const float kMiddleGreyLog = -2.4739311883324122;

groupshared float g_hist[kBuckets];
groupshared float g_scan[kBuckets];
groupshared float g_bodyScan[kBodyBuckets];
groupshared uint  g_saliency;
groupshared uint  g_bodyPercentile[3];   // 0.5th, 50th, 99.5th

float psdtBucketToLog(uint bucket, uint bucketCount)
{
  return lerp(cb.toneCurveMinStops, cb.toneCurveMaxStops,
              float(bucket) / float(bucketCount - 1));
}

[shader("compute")]
[numthreads(PSDT_STATE_GROUP_SIZE, 1, 1)]
void main(uint i : SV_GroupIndex)
{
  g_hist[i] = float(InHistogram[int(i)]) / kHistogramScale;
  if (i < kBodyBuckets)
  {
    g_bodyScan[i] = float(InOutBodyHistogram[int(i)]) / kBodyScale;
  }
  if (i < 3) { g_bodyPercentile[i] = 0; }
  if (i == 0) { g_saliency = 0; }

  GroupMemoryBarrierWithGroupSync();

  // Inclusive prefix sum over the saliency histogram, Hillis-Steele. Eight
  // steps for 256 buckets; the CDF is what the percentile below is read off.
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

  // And over the body histogram, which is half the width. Threads at or above
  // kBodyBuckets sit out the adds but still have to reach every barrier, so
  // the loop is outside the bounds test rather than inside it.
  [unroll]
  for (uint bodyOffset = 1; bodyOffset < kBodyBuckets; bodyOffset <<= 1)
  {
    float bodyAddend = 0.0;
    if (i < kBodyBuckets && i >= bodyOffset) { bodyAddend = g_bodyScan[i - bodyOffset]; }
    GroupMemoryBarrierWithGroupSync();
    if (i < kBodyBuckets) { g_bodyScan[i] += bodyAddend; }
    GroupMemoryBarrierWithGroupSync();
  }

  const float saliencyTotal = g_scan[kBuckets - 1];
  const float bodyTotal = g_bodyScan[kBodyBuckets - 1];

  // Each thread owns the bucket its own CDF interval covers, so every
  // percentile search is a single comparison rather than a scan.
  if (saliencyTotal > 1e-4)
  {
    const float previous = (i == 0) ? 0.0 : g_scan[i - 1];
    const float t500 = saliencyTotal * 0.5;
    if (previous < t500 && g_scan[i] >= t500) { g_saliency = i; }
  }

  if (i < kBodyBuckets && bodyTotal > 1e-4)
  {
    const float previous = (i == 0) ? 0.0 : g_bodyScan[i - 1];
    const float current = g_bodyScan[i];
    const float t005 = bodyTotal * 0.005;
    const float t500 = bodyTotal * 0.500;
    const float t995 = bodyTotal * 0.995;

    if (previous < t005 && current >= t005) { g_bodyPercentile[0] = i; }
    if (previous < t500 && current >= t500) { g_bodyPercentile[1] = i; }
    if (previous < t995 && current >= t995) { g_bodyPercentile[2] = i; }
  }

  GroupMemoryBarrierWithGroupSync();

  // The body histogram is accumulated by psdt_analysis with atomics and has to
  // start each frame at zero. Zeroing it here rather than at the start of
  // analysis costs nothing - these threads exist anyway and have just finished
  // reading it - and avoids either a clear dispatch or a barrier between two
  // halves of the same pass.
  if (i < kBodyBuckets)
  {
    InOutBodyHistogram[int(i)] = 0;
  }

  if (i != 0)
  {
    return;
  }

  // =====================================================================
  // Everything below is single threaded. It is roughly two hundred scalar
  // operations on numbers that have already been reduced, so a wider
  // reduction would buy nothing and would cost the barriers to coordinate.
  // =====================================================================

  const bool bodyValid = bodyTotal > 1e-4;
  const float sceneBlackLog = bodyValid ? psdtBucketToLog(g_bodyPercentile[0], kBodyBuckets) : kMiddleGreyLog - 6.0;
  const float bodyMedianLog = bodyValid ? psdtBucketToLog(g_bodyPercentile[1], kBodyBuckets) : kMiddleGreyLog;
  const float sceneWhiteLog = bodyValid ? psdtBucketToLog(g_bodyPercentile[2], kBodyBuckets) : kMiddleGreyLog + 6.0;
  const float contrastRange = max(sceneWhiteLog - sceneBlackLog, 0.0);
  const float saliencyLog = (saliencyTotal > 1e-4) ? psdtBucketToLog(g_saliency, kBuckets) : bodyMedianLog;

  // Coarsest pyramid level, sampled at the frame centre: the whole frame
  // reduced to one texel of each field.
  const uint levelCount = max((cb.flags >> PSDT_STATE_SHIFT_LEVELS) & 0xFu, 1u);
  const float coarse = float(levelCount) - 1.0;
  const float4 frame = InField.SampleLevel(float2(0.5, 0.5), coarse);
  const float4 frameSource = InSource.SampleLevel(float2(0.5, 0.5), coarse);
  const float4 frameIllum = InIlluminant.SampleLevel(float2(0.5, 0.5), coarse);

  const float bodyMeanLog = frame.x;
  const float bodyFraction = saturate(frame.y);
  const float sourceFraction = saturate(1.0 - bodyFraction);
  const float framePeakLog = frame.z;
  const float sourceEnergy = max(psdtLuminance(frameSource.xyz), 0.0);
  const float skyFraction = saturate(frameSource.w);

  // Frame illuminant. The field stores it premultiplied by confidence, so the
  // divide recovers it and the weight says how much of the frame produced a
  // usable estimate at all. Below a floor there is no evidence and the answer
  // is D65 - which is the honest answer, not a fallback.
  const float illumConfidence = saturate(frameIllum.w * 4.0);
  float3 frameIlluminant = float3(1.0, 1.0, 1.0);
  if (frameIllum.w > 1e-4)
  {
    const float3 raw = frameIllum.xyz / frameIllum.w;
    frameIlluminant = raw / max(psdtLuminance(raw), 1e-4);
  }

  // --- scene intent (statistics, not inference) -------------------------
  // Soft classification from the numbers already in hand. Nothing here is
  // learned or thresholded hard; the weights are smooth so a scene drifting
  // between two classes drifts between two parameter sets rather than
  // switching. Sky fraction is measured from view depth now rather than
  // inferred from "there is something bright up there", which is what makes
  // outdoor separable from a bright interior at all.
  const float bodyAnchor = lerp(bodyMedianLog, bodyMeanLog, 0.35);
  const float relBody = bodyAnchor - kMiddleGreyLog;

  float intentNight  = 1.0 - smoothstep(-3.0, 0.0, relBody);
  float intentBright = saturate(smoothstep(0.02, 0.30, sourceFraction)
                              + smoothstep(2.0, 32.0, sourceEnergy));
  float intentOutdoor = smoothstep(0.02, 0.25, skyFraction)
                      * smoothstep(-2.0, 1.0, relBody);
  float intentIndoor = max(1.0 - intentNight - intentOutdoor - intentBright, 0.0);

  const float intentSum = max(intentNight + intentIndoor + intentOutdoor + intentBright, 1e-3);
  intentNight  /= intentSum;
  intentIndoor /= intentSum;
  intentOutdoor /= intentSum;
  intentBright /= intentSum;

  const float adaptive = ((cb.flags & PSDT_STATE_FLAG_SCENE_ADAPTIVE) != 0) ? 1.0 : 0.0;

  // --- the anchor -------------------------------------------------------
  // sourceExclusion picks how far the anchor moves from "what is in front of
  // the player" towards "what the surfaces are doing". At 1.0 it ignores
  // light sources entirely and exposes purely for surfaces.
  float anchorTarget = lerp(saliencyLog, bodyAnchor, saturate(cb.sourceExclusion));

  // Intent reaches the anchor, which is the point of measuring it. A night
  // scene is anchored slightly above its own mean so that it renders dark
  // rather than being normalised into a grey fog; a frame full of explosion
  // is anchored slightly below so the flash spends the shoulder instead of
  // dragging the whole image down with it. Both are fractions of a stop -
  // they are the difference between two plausible exposures, not a preset.
  anchorTarget += adaptive * (0.35 * intentNight - 0.50 * intentBright);

  // --- temporal adaptation ---------------------------------------------
  const float previousAnchor = InOutState[PSDT_STATE_ANCHOR_LOGY];
  const float previousWhite = InOutState[PSDT_STATE_SCENE_WHITE_LOGY];
  const float previousBlack = InOutState[PSDT_STATE_SCENE_BLACK_LOGY];
  const bool hasHistory = ((cb.flags & PSDT_STATE_FLAG_HAS_HISTORY) != 0)
                        && (InOutState[PSDT_STATE_HAS_HISTORY] > 0.5);

  // Camera-aware transition classification.
  //
  // v0.1 detected cuts from the histogram alone, which conflates three things
  // that want three different responses: a player sweeping their view, a
  // teleport, and a scripted cut. The camera knows which is which, and the
  // renderer already computes it - RtCamera::isCameraCut is a position jump
  // larger than the scene's own unique-object distance, which is a teleport
  // or a respawn by construction.
  //
  //   still     normal time constants
  //   sweeping  faster, because the scene in front of the player genuinely
  //             changed and easing through several seconds of it reads as the
  //             image washing out
  //   cut       re-anchor outright
  const float sweep = saturate(smoothstep(1.5, 6.0, cb.cameraAngularSpeed));
  const bool cameraCut = (cb.flags & PSDT_STATE_FLAG_CAMERA_CUT) != 0;

  // The jump is measured between successive *measurements*, not between this
  // frame's measurement and the filtered state. That distinction is the whole
  // correctness of the detector. Downward adaptation is deliberately slow, so
  // during any sustained dimming the state falls behind the measurement on
  // purpose; comparing the two makes that lag look like a discontinuity, and
  // the exposure snaps part way through a fade that was working correctly.
  // v0.1 compared against the state and did exactly that - psdt_suite's
  // temporal section catches it on a 6-stop fade over half a second.
  //
  // A cut is a discontinuity in what the frame contains. Frame-to-frame, and
  // nothing to do with how far the observer has got round to following it.
  const float prevTargetAnchor = InOutState[PSDT_STATE_PREV_TARGET_ANCHOR];
  const float prevTargetWhite = InOutState[PSDT_STATE_PREV_TARGET_WHITE];
  const float prevTargetBlack = InOutState[PSDT_STATE_PREV_TARGET_BLACK];

  const float anchorJump = abs(anchorTarget - prevTargetAnchor);
  const float whiteJump = abs(sceneWhiteLog - prevTargetWhite);
  const float blackJump = abs(sceneBlackLog - prevTargetBlack);
  // The luminance test is kept as a second opinion, for scripted cuts that do
  // not move the camera - a light going out, a cinematic swap. It is stricter
  // than v0.1's because it no longer has to catch teleports on its own.
  const bool luminanceCut = hasHistory && (anchorJump > 3.0 || (whiteJump > 3.5 && blackJump > 3.5));
  const bool isCut = hasHistory && (cameraCut || luminanceCut);

  // Asymmetric, in the direction human vision actually is: light adaptation
  // is fast, dark adaptation is slow. Going from a dark corridor into
  // daylight should settle quickly; the reverse should take a moment, which
  // is also what makes an interior read as dark when you step into it.
  const float dt = clamp(cb.deltaTimeSeconds, 0.0, 0.25);
  const float baseSpeed = (anchorTarget > previousAnchor) ? cb.adaptationSpeedUp : cb.adaptationSpeedDown;
  const float speed = baseSpeed * lerp(1.0, 3.0, sweep);
  float alpha = 1.0 - exp(-max(speed, 0.01) * dt);
  // The state tracks the measurement outright for as long as the cut flag is
  // still high, not only on the frame the cut was detected. The field is being
  // rebuilt over those same frames - its own history was thrown away and its
  // reprojection has nothing to reproject - so a state that resumed easing
  // immediately would ease towards a field that had not finished converging,
  // and the cut would take half a second to land instead of two frames. The
  // flag decays over about a quarter of a second, which is the window.
  const float previousCutFlag = InOutState[PSDT_STATE_CUT_FLAG];
  if (!hasHistory || isCut || previousCutFlag > 0.5) { alpha = 1.0; }

  const float anchor = lerp(previousAnchor, anchorTarget, alpha);
  const float sceneWhite = lerp(previousWhite, sceneWhiteLog, alpha);
  const float sceneBlack = lerp(previousBlack, sceneBlackLog, alpha);
  const float velocity = (dt > 1e-4) ? ((anchor - previousAnchor) / dt) : 0.0;

  // Cut flag decays rather than clearing, so the analysis pass keeps its
  // history weight raised for a few frames after a cut while the reprojection
  // is still rebuilding.
  const float cutFlag = isCut ? 1.0 : max(previousCutFlag - dt * 4.0, 0.0);

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
  InOutState[PSDT_STATE_MEDIAN_LOGY]      = bodyMedianLog;
  InOutState[PSDT_STATE_SALIENCY_LOGY]    = saliencyLog;
  InOutState[PSDT_STATE_HAS_HISTORY]      = 1.0;
  InOutState[PSDT_STATE_SKY_FRACTION]     = skyFraction;
  InOutState[PSDT_STATE_FRAME_PEAK_LOGY]  = framePeakLog;
  InOutState[PSDT_STATE_ILLUM_FRAME_R]    = frameIlluminant.r;
  InOutState[PSDT_STATE_ILLUM_FRAME_G]    = frameIlluminant.g;
  InOutState[PSDT_STATE_ILLUM_FRAME_B]    = frameIlluminant.b;
  InOutState[PSDT_STATE_ILLUM_CONFIDENCE] = illumConfidence;
  InOutState[PSDT_STATE_CAMERA_ANGULAR]   = cb.cameraAngularSpeed;
  InOutState[PSDT_STATE_TRANSITION]       = isCut ? 2.0 : sweep;
  InOutState[PSDT_STATE_PREV_TARGET_ANCHOR] = anchorTarget;
  InOutState[PSDT_STATE_PREV_TARGET_WHITE]  = sceneWhiteLog;
  InOutState[PSDT_STATE_PREV_TARGET_BLACK]  = sceneBlackLog;

  // =====================================================================
  // Parameter resolution
  // =====================================================================

  const uint gamut = (cb.flags >> PSDT_STATE_SHIFT_GAMUT) & 0x3u;
  const uint space = (cb.flags >> PSDT_STATE_SHIFT_SPACE) & 0x1u;
  const uint debugView = (cb.flags >> PSDT_STATE_SHIFT_DEBUG) & 0xFu;
  const float nearField = float((cb.flags >> PSDT_STATE_SHIFT_NEARFIELD) & 0xFFu) / 255.0;
  const float scaleCoherence = float((cb.flags >> PSDT_STATE_SHIFT_COHERENCE) & 0xFFu)
                             * (PSDT_COHERENCE_MAX / 255.0);

  const float refWhite = max(cb.displayRefWhiteNits, 1.0);
  const float peakLog = log2(max(cb.displayPeakNits, refWhite) / refWhite);
  const float blackLinear = max(cb.displayBlackNits / refWhite, 1e-7);
  const float blackLog = log2(blackLinear);

  InOutState[PSDT_STATE_DISPLAY_PEAK]      = cb.displayPeakNits;
  InOutState[PSDT_STATE_DISPLAY_BLACK]     = cb.displayBlackNits;
  InOutState[PSDT_STATE_DISPLAY_REF_WHITE] = refWhite;
  InOutState[PSDT_STATE_SURROUND]          = cb.surroundNits;
  InOutState[PSDT_STATE_DISPLAY_GAMUT]     = float(gamut);
  InOutState[PSDT_STATE_DISPLAY_HEADROOM]  = peakLog;

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
  //
  // The clamp keeps at least a quarter stop of linear segment on each side of
  // the mid anchor whatever the slope is. Without it a high midtoneContrast
  // shrinks the highlight range enough that the shoulder starts underneath
  // middle grey. v0.1's CPU reference had this clamp and its shader did not,
  // which is the kind of divergence that makes a test suite report on
  // something other than what runs.
  const float highlightRange = max(peakLog - midOutLog, 1.0) / slope;
  const float rolloff = saturate(cb.highlightRolloff * lerp(1.0, 1.0 - 0.30 * intentBright, adaptive));
  const float shoulderStart = clamp(rolloff * highlightRange, 0.25, max(highlightRange - 0.25, 0.30));

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
  InOutState[PSDT_STATE_DEPTH_SENSITIVITY] = max(cb.depthSensitivity, 0.0);

  // Scale coherence. Held back in scenes the transform has already decided are
  // low-range, for the same reason the adaptation strength is: with nothing to
  // redistribute there is no boundary worth protecting, and rejecting the
  // coarse scales over the frame's own gentle gradient would leave the pooling
  // running on the finest scale alone - which is the noisiest of the three and
  // the one that most wants company.
  InOutState[PSDT_STATE_SCALE_COHERENCE] = scaleCoherence * lerp(1.0, rangeGate, adaptive);

  // --- local contrast ---------------------------------------------------
  InOutState[PSDT_STATE_DETAIL_STRENGTH] = saturate(cb.detailStrength);
  InOutState[PSDT_STATE_DETAIL_PROTECT]  = max(cb.detailProtect, 0.0);
  InOutState[PSDT_STATE_DETAIL_KNEE]     = max(cb.detailKnee, 0.05);

  // --- colour volume ----------------------------------------------------
  InOutState[PSDT_STATE_PRESSURE_LUMA]    = saturate(cb.pressureLuma);
  InOutState[PSDT_STATE_PRESSURE_CHROMA]  = psdtChromaReference(space, gamut);
  // Reported for diagnostics: where the soft knee actually sits for an
  // ordinary colour, given chromaPreservation.
  InOutState[PSDT_STATE_PRESSURE_GAMUT]   = 1.0 - 1.0 / (1.0 + max(cb.chromaPreservation, 0.05));
  InOutState[PSDT_STATE_PRESSURE_CONTEXT] = max(cb.pressureContext, 0.0);
  InOutState[PSDT_STATE_CHROMA_PRESERVE]  = max(cb.chromaPreservation, 0.05);
  InOutState[PSDT_STATE_HUE_TRAJECTORY]   = saturate(cb.hueTrajectory);
  InOutState[PSDT_STATE_WHITE_CONVERGE]   = max(cb.whiteConvergence, 0.05);
  // The white point only adapts as far as the evidence supports. A frame
  // where almost nothing had a usable albedo gets a D65 white, which is the
  // right answer when the illuminant is unknown rather than a compromise.
  InOutState[PSDT_STATE_ILLUM_STRENGTH]   = saturate(cb.illuminantStrength) * illumConfidence;
  InOutState[PSDT_STATE_COLOURFULNESS]    = max(cb.colourfulness, 0.0);
  InOutState[PSDT_STATE_LUMA_CONCESSION]  = saturate(cb.luminanceConcession);

  // --- glare ------------------------------------------------------------
  // The threshold is anchored to the adaptation state, not to an absolute
  // value: whether a light glares depends on what the viewer is adapted to,
  // which is the entire reason a car headlight is blinding at night and
  // invisible at noon.
  InOutState[PSDT_STATE_GLARE_STRENGTH]  = max(cb.glareStrength, 0.0);
  InOutState[PSDT_STATE_GLARE_THRESHOLD] = anchor + cb.glareThresholdStops;
  InOutState[PSDT_STATE_GLARE_FALLOFF]   = clamp(cb.glareFalloff, 0.05, 0.95);
  InOutState[PSDT_STATE_GLARE_NEARFIELD] = nearField;
  InOutState[PSDT_STATE_GLARE_LEVELS]    = float(levelCount);

  // --- misc -------------------------------------------------------------
  InOutState[PSDT_STATE_PERCEPTUAL_SPACE] = float(space);
  InOutState[PSDT_STATE_FIELD_LEVEL_S]    = 0.0;
  InOutState[PSDT_STATE_FIELD_LEVEL_M]    = min(2.0, coarse);
  InOutState[PSDT_STATE_FIELD_LEVEL_L]    = min(4.0, coarse);
  InOutState[PSDT_STATE_LEVEL_COUNT]      = float(levelCount);
  InOutState[PSDT_STATE_DEBUG_VIEW]       = float(debugView);
}
