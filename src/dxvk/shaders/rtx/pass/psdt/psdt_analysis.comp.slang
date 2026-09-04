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
// PSDT scene analysis: HDR radiance -> adaptation field, mip 0
// ===========================================================================
//
// One workgroup per field texel, one thread per source pixel. Each 16x16
// block is classified pixel by pixel and reduced to four numbers.
//
// The classification is the interesting part. Conventional exposure metering
// asks "how bright is this frame"; this asks "which parts of this frame is
// the viewer adapted to". Those are different questions the moment a scene
// contains a light source, which every Titanfall interior does:
//
//      dark corridor  +  small sunlit window
//                    |
//        histogram sees: dark mass + huge bright spike
//        viewer sees:    a dark corridor, and a window
//
// A metering system that treats the window as scene content exposes for
// somewhere between the two and renders both wrong. So pixels are sorted into
// three roles and only one of them is allowed to drive adaptation:
//
//   BODY    walls, characters, weapons, ground. Ordinary surfaces reflecting
//           light. These, and only these, determine exposure.
//   SOURCE  lamps, sky, emissive screens, muzzle flashes. Excluded from the
//           adaptation estimate - excluded, not clamped, because a clamped
//           sun still drags a mean and an excluded one does not.
//   GLARE   bright enough that no display can represent it as a value. Its
//           energy is kept, in the field's .z channel, and spent later as
//           spatial extent by the glare model rather than thrown away at a
//           clamp.
//
// The thresholds are in stops above middle grey rather than absolute, which
// only works because the input is post-exposure: the base auto exposure has
// already anchored the frame's mean near 0.18, so "4 stops above middle grey"
// is a statement about this scene and not about a number. With auto exposure
// off the thresholds become absolute and want re-tuning; that is the one
// configuration where rtx.tonemap.psdt.sourceThreshold has to move.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"
#include "rtx/pass/psdt/psdt_perceptual_space.slangh"

layout(binding = PSDT_ANALYSIS_COLOR_INPUT)
Texture2D<float4> InColorBuffer;

layout(binding = PSDT_ANALYSIS_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

layout(binding = PSDT_ANALYSIS_MOTION_INPUT)
Texture2D<float2> PrimaryScreenSpaceMotionVector;

layout(binding = PSDT_ANALYSIS_FIELD_OUTPUT)
RWTexture2D<vec4> OutField;

layout(binding = PSDT_ANALYSIS_FIELD_HISTORY_INPUT)
Sampler2D InFieldHistory;

layout(binding = PSDT_ANALYSIS_CHROMA_OUTPUT)
RWTexture2D<vec4> OutChroma;

layout(binding = PSDT_ANALYSIS_CHROMA_HISTORY_INPUT)
Sampler2D InChromaHistory;

layout(binding = PSDT_ANALYSIS_STATE_INPUT)
RWTexture1D<float> InState;

layout(push_constant)
ConstantBuffer<PsdtAnalysisArgs> cb;

static const uint kThreads = PSDT_GROUP_SIZE * PSDT_GROUP_SIZE;

// Middle grey after exposure. The base auto exposure drives the frame mean
// here, so it is the natural origin for every threshold in this pass.
static const float kMiddleGrey = 0.18;
static const float kMiddleGreyLog = -2.4739311883324122;

// Luminance floor, ~13 stops below middle grey. Bounds how far a black pixel
// can pull the block's geometric mean down.
static const float kMinLuminance = 1e-5;

// f16 storage ceiling for source energy. Well below 65504 so that the mip
// chain's running sums cannot overflow either.
static const float kMaxSourceEnergy = 16384.0;

groupshared float g_bodyLogSum[kThreads];
groupshared float g_bodyWeight[kThreads];
groupshared float g_sourceEnergy[kThreads];
groupshared float g_peakLog[kThreads];
groupshared float g_sumR[kThreads];
groupshared float g_sumG[kThreads];
groupshared float g_sumB[kThreads];

[shader("compute")]
[numthreads(PSDT_GROUP_SIZE, PSDT_GROUP_SIZE, 1)]
void main(uint2 groupId : SV_GroupID, uint2 groupThreadId : SV_GroupThreadID, uint linearIndex : SV_GroupIndex)
{
  const uint2 fieldTexel = groupId;
  const uint2 pixel = fieldTexel * PSDT_FIELD_BLOCK_SIZE + groupThreadId;

  float exposure = cb.exposure;
  if (cb.enableAutoExposure != 0)
  {
    exposure *= InExposure[0].r;
  }

  float bodyLogSum = 0.0;
  float bodyWeight = 0.0;
  float sourceEnergy = 0.0;
  float peakLog = kMiddleGreyLog - 16.0;
  float3 sumRgb = float3(0.0);

  if (all(pixel < cb.colorExtent))
  {
    const float3 rgb = max(InColorBuffer[pixel].xyz, 0.0) * exposure;
    const float y = max(psdtLuminance(rgb), kMinLuminance);
    const float logY = log2(y);
    const float relStops = logY - kMiddleGreyLog;

    peakLog = logY;
    sumRgb = rgb;

    // Soft classification. A hard threshold would make a light that drifts
    // across it pop the whole frame's exposure, because the block's body
    // weight would step. The smoothstep is one stop wide, which is far below
    // the base auto exposure's own time constant, so the transition is never
    // visible as an event.
    const float sourceness = smoothstep(cb.sourceThresholdStops - 0.5,
                                        cb.sourceThresholdStops + 0.5,
                                        relStops);
    const float glareness = smoothstep(cb.glareThresholdStops - 1.0,
                                       cb.glareThresholdStops + 1.0,
                                       relStops);

    bodyWeight = 1.0 - sourceness;
    bodyLogSum = logY * bodyWeight;

    // Everything classified away from BODY carries energy into the glare
    // channel, weighted so that a merely-bright surface contributes little
    // and an actual emitter contributes almost all of itself. Stored linearly
    // so the mip chain conserves it.
    sourceEnergy = min(y, kMaxSourceEnergy) * sourceness * lerp(0.25, 1.0, glareness);
  }

  g_bodyLogSum[linearIndex] = bodyLogSum;
  g_bodyWeight[linearIndex] = bodyWeight;
  g_sourceEnergy[linearIndex] = sourceEnergy;
  g_peakLog[linearIndex] = peakLog;
  g_sumR[linearIndex] = sumRgb.r;
  g_sumG[linearIndex] = sumRgb.g;
  g_sumB[linearIndex] = sumRgb.b;

  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint stride = kThreads / 2; stride > 0; stride >>= 1)
  {
    if (linearIndex < stride)
    {
      g_bodyLogSum[linearIndex]   += g_bodyLogSum[linearIndex + stride];
      g_bodyWeight[linearIndex]   += g_bodyWeight[linearIndex + stride];
      g_sourceEnergy[linearIndex] += g_sourceEnergy[linearIndex + stride];
      g_peakLog[linearIndex]       = max(g_peakLog[linearIndex], g_peakLog[linearIndex + stride]);
      g_sumR[linearIndex]         += g_sumR[linearIndex + stride];
      g_sumG[linearIndex]         += g_sumG[linearIndex + stride];
      g_sumB[linearIndex]         += g_sumB[linearIndex + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (linearIndex != 0 || any(fieldTexel >= cb.fieldExtent))
  {
    return;
  }

  const float weight = g_bodyWeight[0];
  // Mean of log2 luminance is a geometric mean of luminance, which is the
  // right average for an adaptation estimate - it is not dominated by the
  // brightest member the way an arithmetic mean is, and it is linear in the
  // stops the rest of the transform is expressed in.
  const float bodyLog = (weight > 1e-3) ? (g_bodyLogSum[0] / weight) : kMiddleGreyLog;

  float4 current;
  current.x = bodyLog;
  current.y = weight / float(kThreads);
  current.z = min(g_sourceEnergy[0] / float(kThreads), kMaxSourceEnergy);
  current.w = g_peakLog[0];

  // One perceptual conversion per 256 pixels: the block's mean colour, which
  // is what both consumers of the chroma field actually want. Doing it per
  // pixel would cost three PQ encodes on every pixel of the frame to produce
  // a number that is then averaged over 256 of them anyway.
  const float3 meanRgb = float3(g_sumR[0], g_sumG[0], g_sumB[0]) / float(kThreads);
  const float3 iab = psdtToPerceptual(max(meanRgb, 0.0), uint(cb.perceptualSpace), max(cb.refWhiteNits, 1.0));
  float2 currentChroma = iab.yz;

  // --- temporal accumulation -------------------------------------------
  // The field is the most temporally sensitive part of the transform: it sets
  // where the curve's anchor sits, and the input is a denoised path-traced
  // image whose residual noise is worst in exactly the dark regions that get
  // the most adaptation. Left stateless the exposure visibly breathes.
  //
  // The previous frame's cut flag. psdt_state runs after this pass, so what is
  // read here was written a frame ago - which is exactly right: a cut detected
  // last frame means this frame's history describes a scene that is gone. The
  // flag decays over about a quarter of a second rather than clearing, so the
  // field is given several fast frames to re-converge instead of one.
  const float cutFlag = saturate(InState[PSDT_STATE_CUT_FLAG]);
  const float currentWeight = lerp(cb.currentWeight, 1.0, cutFlag);

  if (cb.hasHistory != 0 && currentWeight < 1.0)
  {
    const float2 uv = (float2(fieldTexel) + 0.5) / float2(cb.fieldExtent);

    // Motion vectors are in pixels at raytracing resolution and point from
    // this frame to the previous one; dividing by their own extent gives a
    // resolution independent UV offset. Point sampled, so a silhouette does
    // not blend a foreground and a background vector into one that
    // reprojects to neither.
    const uint2 motionPixel = min(uint2(uv * float2(cb.motionExtent)), cb.motionExtent - 1);
    const float2 motion = PrimaryScreenSpaceMotionVector[motionPixel];
    const float2 previousUv = uv + motion / float2(cb.motionExtent);

    if (all(previousUv > float2(0.0)) && all(previousUv < float2(1.0)))
    {
      float4 history = InFieldHistory.SampleLevel(previousUv, 0);
      float2 historyChroma = InChromaHistory.SampleLevel(previousUv, 0).xy;

      // History is constrained to a window around what this block currently
      // measures. The usual temporal-antialiasing argument: history is
      // trustworthy only insofar as it agrees with something the current
      // frame contains. Anything further out came from geometry that is no
      // longer here. The 3-stop window is fixed rather than derived from a
      // neighbourhood min/max because the neighbours are being written by
      // other workgroups in this same dispatch and cannot be read safely.
      history.x = clamp(history.x, current.x - 3.0, current.x + 3.0);
      history.y = clamp(history.y, current.y - 0.35, current.y + 0.35);
      history.z = clamp(history.z, current.z * 0.25, current.z * 4.0 + 1e-3);
      history.w = clamp(history.w, current.w - 3.0, current.w + 3.0);
      historyChroma = clamp(historyChroma, currentChroma - 0.05, currentChroma + 0.05);

      current = lerp(history, current, currentWeight);
      currentChroma = lerp(historyChroma, currentChroma, currentWeight);
    }
  }

  OutField[fieldTexel] = current;
  OutChroma[fieldTexel] = vec4(currentChroma, 0.0, 0.0);
}
