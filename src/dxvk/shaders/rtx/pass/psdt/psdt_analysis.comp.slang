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
// PSDT scene analysis: HDR radiance + gbuffer -> adaptation field, mip 0
// ===========================================================================
//
// One workgroup per field texel, one thread per source pixel. Each 16x16
// block is classified pixel by pixel and reduced to twelve numbers spread
// across three field textures.
//
// The classification is the interesting part, and it is the part v0.1 got
// wrong. Conventional exposure metering asks "how bright is this frame"; this
// asks "which parts of this frame is the viewer adapted to". Those are
// different questions the moment a scene contains a light source, which every
// Titanfall interior does:
//
//      dark corridor  +  small sunlit window
//                    |
//        histogram sees: dark mass + huge bright spike
//        viewer sees:    a dark corridor, and a window
//
// So pixels are sorted into four roles and only one of them fully drives
// adaptation:
//
//   BODY    walls, characters, weapons, ground. Ordinary surfaces reflecting
//           light. These determine exposure.
//   SOURCE  lamps, emissive screens, muzzle flashes. Excluded from the
//           adaptation estimate - excluded, not clamped, because a clamped
//           sun still drags a mean and an excluded one does not.
//   GLARE   bright enough that no display can represent it as a value. Its
//           energy is kept, per channel, in the source field, and spent later
//           as spatial extent by the glare model rather than thrown away at a
//           clamp.
//   SKY     everything past the far plane. Neither surface nor lamp: it is
//           the illuminant of an outdoor scene and a hole in the wall of an
//           indoor one, and which of those it is depends on how much of the
//           frame it occupies, not on how bright it is.
//
// Where the evidence comes from
// -----------------------------
// v0.1 classified on brightness alone:
//
//     sourceness = smoothstep(threshold, log2(Y) - middleGrey)
//
// which cannot tell a sunlit white wall from a neon sign, because at the
// framebuffer there is no difference between them - both are just bright.
// The difference exists one stage earlier, in the renderer, and Remix keeps
// it. This pass reads it:
//
//   PrimarySurfaceFlags.isEmissive  set by the geometry resolver when the
//                                   surface contributed emissive radiance of
//                                   its own. This is the actual distinction
//                                   between a thing that emits and a thing
//                                   that is lit, and no amount of looking at
//                                   the framebuffer recovers it.
//   PrimaryLinearViewZ              a ray miss writes NRD's missLinearViewZ;
//                                   far backdrop geometry is clamped to
//                                   200000. Either way sky separates from
//                                   room geometry by four orders of
//                                   magnitude, so "is this the sky" stops
//                                   being a guess about brightness.
//   PrimaryAlbedo                   lets reflectance be divided out of
//                                   radiance, which is what turns "the
//                                   average colour around here" into "the
//                                   colour of the light around here".
//   spatial compactness             measured inside the block: how much of
//                                   the neighbourhood is bright. A small
//                                   bright thing in a dark surround is a
//                                   source; a whole block that is bright is a
//                                   lit surface. This is the evidence that
//                                   still works when the gbuffer is
//                                   unavailable, and it is what keeps the
//                                   fallback path sane.
//   temporal persistence            the field is accumulated with motion
//                                   reprojection, so a classification has to
//                                   survive several frames before it fully
//                                   moves the adaptation. A one-frame
//                                   speculative fireball does not re-expose
//                                   the room.
//
// The brightness thresholds remain, but as one input among several rather
// than as the whole classifier. They are in stops above middle grey rather
// than absolute, which only works because the input is post-exposure: the
// base auto exposure has already anchored the frame's mean near 0.18.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"
#include "rtx/pass/psdt/psdt_perceptual_space.slangh"
#include "rtx/pass/post_fx/post_fx_motion_blur_geometry_flags.slangh"

layout(binding = PSDT_ANALYSIS_COLOR_INPUT)
Texture2D<float4> InColorBuffer;

layout(binding = PSDT_ANALYSIS_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

layout(binding = PSDT_ANALYSIS_MOTION_INPUT)
Texture2D<float2> PrimaryScreenSpaceMotionVector;

layout(binding = PSDT_ANALYSIS_SURFACE_FLAGS_INPUT)
Texture2D<uint> PrimarySurfaceFlags;

layout(binding = PSDT_ANALYSIS_VIEW_Z_INPUT)
Texture2D<float> PrimaryLinearViewZ;

layout(binding = PSDT_ANALYSIS_ALBEDO_INPUT)
Texture2D<float4> PrimaryAlbedo;

layout(binding = PSDT_ANALYSIS_FIELD_OUTPUT)
RWTexture2D<vec4> OutField;

layout(binding = PSDT_ANALYSIS_FIELD_HISTORY_INPUT)
Sampler2D InFieldHistory;

layout(binding = PSDT_ANALYSIS_SOURCE_OUTPUT)
RWTexture2D<vec4> OutSource;

layout(binding = PSDT_ANALYSIS_SOURCE_HISTORY_INPUT)
Sampler2D InSourceHistory;

layout(binding = PSDT_ANALYSIS_ILLUM_OUTPUT)
RWTexture2D<vec4> OutIlluminant;

layout(binding = PSDT_ANALYSIS_ILLUM_HISTORY_INPUT)
Sampler2D InIlluminantHistory;

layout(binding = PSDT_ANALYSIS_BODY_HISTOGRAM_OUTPUT)
RWTexture1D<uint> OutBodyHistogram;

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

// How much a sky pixel counts towards adaptation, relative to a surface.
// Not zero and not one, and the reason is the whole sky/window distinction:
// standing outdoors the sky *is* what you are adapted to, while the same sky
// seen through a doorway is not. Rather than deciding that per pixel - which
// is not knowable per pixel - sky is given a low fixed weight and the frame
// decides the rest by area. A frame that is 60% sky still lands most of its
// adaptation weight on the 40% that is ground, which is what a viewer walking
// outdoors actually does; a frame that is 3% sky gives it almost nothing.
static const float kSkyAdaptationWeight = 0.15;

// An emissive surface needs less brightness evidence to read as a source than
// an ordinary one does, because the emissive bit is already direct evidence.
// A dim emissive decal is still scene content, so it is a discount rather than
// a bypass.
static const float kEmissiveThresholdDiscount = 2.5;

// Fixed-point scales for the groupshared reductions. Every accumulator below
// is a uint updated with InterlockedAdd rather than a 256-entry groupshared
// array reduced in log steps: the same result, 11 words of shared memory
// instead of four kilobytes, and no barriers inside the accumulation.
static const float kFxLog = 4096.0;    // log2 luminance, offset by kLogOffset
static const float kFxWeight = 4096.0; // weights in [0,1]
static const float kFxEnergy = 64.0;   // linear source energy
static const float kLogOffset = 64.0;  // makes any plausible log2 luminance positive

groupshared uint g_bodyLogSum;
groupshared uint g_bodyWeight;
groupshared uint g_srcR;
groupshared uint g_srcG;
groupshared uint g_srcB;
groupshared uint g_skyWeight;
groupshared uint g_depthSum;
groupshared uint g_illumR;
groupshared uint g_illumG;
groupshared uint g_illumB;
groupshared uint g_illumWeight;
groupshared uint g_peakLogBits;
groupshared uint g_brightCount;
groupshared uint g_bodyHistogram[PSDT_BODY_HISTOGRAM_SIZE];

// IEEE-754 positive floats compare identically to their bit patterns as
// unsigned integers, so an InterlockedMax over asuint() is an exact float max
// as long as the value cannot be negative. kLogOffset guarantees that.
uint psdtEncodeMax(float v)
{
  return asuint(max(v + kLogOffset, 0.0));
}

float psdtDecodeMax(uint bits)
{
  return asfloat(bits) - kLogOffset;
}

[shader("compute")]
[numthreads(PSDT_GROUP_SIZE, PSDT_GROUP_SIZE, 1)]
void main(uint2 groupId : SV_GroupID, uint2 groupThreadId : SV_GroupThreadID, uint linearIndex : SV_GroupIndex)
{
  const uint2 fieldTexel = groupId;
  const uint2 pixel = fieldTexel * PSDT_FIELD_BLOCK_SIZE + groupThreadId;
  const bool useRendererSignals = (cb.flags & PSDT_ANALYSIS_FLAG_RENDERER_SIGNALS) != 0;

  if (linearIndex == 0)
  {
    g_bodyLogSum = 0; g_bodyWeight = 0;
    g_srcR = 0; g_srcG = 0; g_srcB = 0;
    g_skyWeight = 0; g_depthSum = 0;
    g_illumR = 0; g_illumG = 0; g_illumB = 0; g_illumWeight = 0;
    g_peakLogBits = 0; g_brightCount = 0;
  }
  [unroll] for (uint clearIndex = linearIndex; clearIndex < PSDT_BODY_HISTOGRAM_SIZE; clearIndex += kThreads)
  {
    g_bodyHistogram[clearIndex] = 0;
  }

  GroupMemoryBarrierWithGroupSync();

  // -----------------------------------------------------------------------
  // Per-pixel gather
  // -----------------------------------------------------------------------
  float exposure = cb.exposure;
  if ((cb.flags & PSDT_ANALYSIS_FLAG_AUTO_EXPOSURE) != 0)
  {
    exposure *= InExposure[0].r;
  }

  const bool inFrame = all(pixel < cb.colorExtent);

  float3 rgb = float3(0.0);
  float y = kMinLuminance;
  float logY = kMiddleGreyLog;
  float relStops = 0.0;
  bool isEmissive = false;
  bool isSky = false;
  float viewZ = 0.0;
  float3 albedo = float3(0.0);
  float brightness = 0.0;

  if (inFrame)
  {
    rgb = max(InColorBuffer[pixel].xyz, 0.0) * exposure;
    y = max(psdtLuminance(rgb), kMinLuminance);
    logY = clamp(log2(y), -kLogOffset + 1.0, kLogOffset - 1.0);
    relStops = logY - kMiddleGreyLog;

    if (useRendererSignals)
    {
      // The gbuffer runs at the pre-upscale resolution while the tonemapper's
      // input may be post-upscale, so the two are addressed through a shared
      // normalised coordinate rather than by pixel index.
      const float2 uv = (float2(pixel) + 0.5) / float2(cb.colorExtent);
      const uint2 gb = min(uint2(uv * float2(cb.gbufferExtent)), cb.gbufferExtent - 1);

      const MotionBlurSurfaceFlags flags = motionBlurSurfaceFlagsReadFromGBuffer(PrimarySurfaceFlags[gb]);
      isEmissive = flags.isEmissive;
      viewZ = abs(PrimaryLinearViewZ[gb]);
      isSky = viewZ >= cb.skyViewZ;
      albedo = max(PrimaryAlbedo[gb].xyz, 0.0);
    }

    // Brightness evidence. One stop wide so a light drifting across the
    // threshold never steps the block's weight; the transition is far below
    // the base auto exposure's own time constant and so is never visible as
    // an event.
    const float threshold = cb.sourceThresholdStops
                          - (isEmissive ? kEmissiveThresholdDiscount : 0.0);
    brightness = smoothstep(threshold - 0.5, threshold + 0.5, relStops);

    if (brightness > 0.5)
    {
      InterlockedAdd(g_brightCount, 1u);
    }
  }

  GroupMemoryBarrierWithGroupSync();

  // -----------------------------------------------------------------------
  // Classification, now that the neighbourhood is known
  // -----------------------------------------------------------------------
  // Spatial compactness. A block that is entirely bright is a lit surface -
  // a sunlit wall, a white vehicle, snow - and metering it as a light source
  // is exactly the mistake that makes a bright room render dark. A block with
  // a handful of bright pixels in it is a lamp, a muzzle flash or a specular
  // glint. The renderer's emissive bit settles the case where both are true
  // (a large area light), so compactness is only allowed to *reduce*
  // sourceness, never to create it.
  const float brightFraction = float(g_brightCount) / float(kThreads);
  const float compactness = 1.0 - 0.75 * smoothstep(0.35, 0.95, brightFraction);

  if (inFrame)
  {
    // Emissive surfaces keep their full sourceness however large they are;
    // everything else has to be compact to qualify.
    const float sourceness = isSky
      ? 0.0
      : saturate(brightness * (isEmissive ? 1.0 : compactness));

    const float glareness = smoothstep(cb.glareThresholdStops - 1.0,
                                       cb.glareThresholdStops + 1.0,
                                       relStops);

    // Sky is its own role. It contributes a reduced but nonzero adaptation
    // weight and no source energy, unless it is bright enough to be the sun
    // itself, in which case the glare term picks it up below.
    const float skyness = isSky ? 1.0 : 0.0;
    const float bodyWeight = isSky ? kSkyAdaptationWeight : (1.0 - sourceness);

    InterlockedAdd(g_bodyWeight, uint(bodyWeight * kFxWeight));
    InterlockedAdd(g_bodyLogSum, uint((logY + kLogOffset) * bodyWeight * kFxLog));
    InterlockedAdd(g_skyWeight, uint(skyness * kFxWeight));
    InterlockedMax(g_peakLogBits, psdtEncodeMax(logY));

    // Everything classified away from BODY carries energy into the glare
    // channel, per channel so the halo has the source's colour rather than
    // the scene's. A merely-bright surface contributes a quarter of itself
    // and an actual emitter almost all of it. Sky contributes only through
    // glareness, so an overcast sky adds nothing and the sun adds everything.
    const float energyWeight = max(sourceness, skyness * glareness)
                             * lerp(0.25, 1.0, glareness);
    const float3 sourceRgb = min(rgb, kMaxSourceEnergy) * energyWeight;
    InterlockedAdd(g_srcR, uint(sourceRgb.r * kFxEnergy));
    InterlockedAdd(g_srcG, uint(sourceRgb.g * kFxEnergy));
    InterlockedAdd(g_srcB, uint(sourceRgb.b * kFxEnergy));

    // Depth of the surface this adaptation value describes. log2(1+z) rather
    // than z so that the near field, where depth discontinuities actually
    // matter to adaptation, gets the resolution and a 200000-unit backdrop
    // does not swamp it.
    if (useRendererSignals)
    {
      const float depthLog = log2(1.0 + viewZ);
      InterlockedAdd(g_depthSum, uint(depthLog * bodyWeight * kFxLog));
    }

    // ---------------------------------------------------------------------
    // Local illuminant
    // ---------------------------------------------------------------------
    // radiance = albedo * irradiance for a diffuse surface, so dividing the
    // two gives the colour of the light rather than the colour of the paint.
    // That is the whole difference between this and a grey-world estimate: a
    // red wall, a green wall and a blue wall under a neutral light have a
    // strongly coloured mean radiance and a neutral mean *illuminant*, and
    // only the second one is a white point.
    //
    // The division is only trustworthy where the albedo is substantial in
    // every channel - a near-black surface divides noise by noise, and a
    // fully saturated one has no information at all in its dark channels.
    // Both are handled by weighting rather than by a hard test, so the
    // estimate degrades smoothly to "no opinion" instead of flickering.
    //
    // Caveat, stated because it is real: by the time the tonemapper runs,
    // PrimaryAlbedo has been scaled by the composite and ray-reconstruction
    // passes, and the colour buffer contains specular and indirect light this
    // model does not account for. This is a good estimator of the diffuse
    // illuminant, not a measurement of it, and the confidence weight is what
    // keeps it from being trusted further than that.
    if (useRendererSignals && !isSky)
    {
      const float minAlbedo = min(min(albedo.r, albedo.g), albedo.b);
      const float albedoWeight = saturate((minAlbedo - cb.illuminantMinAlbedo)
                                          / max(cb.illuminantMinAlbedo, 1e-3));
      // Dark pixels are where the denoiser's residual noise lives, so weight
      // by luminance too. The knee is two stops below middle grey.
      const float luminanceWeight = saturate(y / (y + 0.045));
      const float weight = (1.0 - sourceness) * albedoWeight * luminanceWeight;

      if (weight > 1e-3)
      {
        float3 illum = rgb / max(albedo, cb.illuminantMinAlbedo);
        const float illumY = max(psdtLuminance(illum), 1e-6);
        // Normalise to luminance 1 so only the chromaticity survives, then
        // bound it: an 8:1 channel ratio is already far outside any plausible
        // illuminant and anything past it is a specular hit or an albedo the
        // composite pass has rescaled, not a light.
        illum = clamp(illum / illumY, 0.125, 8.0);

        InterlockedAdd(g_illumR, uint(illum.r * weight * kFxLog));
        InterlockedAdd(g_illumG, uint(illum.g * weight * kFxLog));
        InterlockedAdd(g_illumB, uint(illum.b * weight * kFxLog));
        InterlockedAdd(g_illumWeight, uint(weight * kFxWeight));
      }
    }

    // ---------------------------------------------------------------------
    // Body-only luminance histogram
    // ---------------------------------------------------------------------
    // The tonemapper's own histogram is centre-weighted and counts every
    // pixel, which makes it a good saliency estimate and a poor description
    // of the scene's surfaces. v0.1 took scene white, scene black and the cut
    // detector from it, so a muzzle flash moved the definition of white even
    // though the same flash had been carefully excluded from the anchor. This
    // one counts body weight only.
    if (bodyWeight > 1e-3)
    {
      const float span = max(cb.toneCurveMaxStops - cb.toneCurveMinStops, 1e-3);
      const float t = saturate((logY - cb.toneCurveMinStops) / span);
      const uint bucket = min(uint(t * float(PSDT_BODY_HISTOGRAM_SIZE - 1)),
                              uint(PSDT_BODY_HISTOGRAM_SIZE - 1));
      InterlockedAdd(g_bodyHistogram[bucket], uint(bodyWeight * kFxWeight));
    }
  }

  GroupMemoryBarrierWithGroupSync();

  // Merge this block's histogram into the frame's. Most buckets in a 256-pixel
  // block are empty, so the branch removes the great majority of the global
  // atomics rather than issuing 128 per workgroup unconditionally.
  [unroll] for (uint mergeIndex = linearIndex; mergeIndex < PSDT_BODY_HISTOGRAM_SIZE; mergeIndex += kThreads)
  {
    const uint count = g_bodyHistogram[mergeIndex];
    if (count != 0)
    {
      InterlockedAdd(OutBodyHistogram[int(mergeIndex)], count);
    }
  }

  if (linearIndex != 0 || any(fieldTexel >= cb.fieldExtent))
  {
    return;
  }

  // -----------------------------------------------------------------------
  // Reduce to the three field texels
  // -----------------------------------------------------------------------
  const float weight = float(g_bodyWeight) / kFxWeight;
  const float invThreads = 1.0 / float(kThreads);

  // Mean of log2 luminance is a geometric mean of luminance, which is the
  // right average for an adaptation estimate - it is not dominated by the
  // brightest member the way an arithmetic mean is, and it is linear in the
  // stops the rest of the transform is expressed in.
  const float bodyLog = (weight > 1e-3)
    ? ((float(g_bodyLogSum) / kFxLog) / weight - kLogOffset)
    : kMiddleGreyLog;
  const float depthLog = (weight > 1e-3)
    ? ((float(g_depthSum) / kFxLog) / weight)
    : 0.0;

  float4 currentField;
  currentField.x = bodyLog;
  currentField.y = weight * invThreads;
  currentField.z = psdtDecodeMax(g_peakLogBits);
  currentField.w = depthLog;

  float4 currentSource;
  currentSource.x = min(float(g_srcR) / kFxEnergy * invThreads, kMaxSourceEnergy);
  currentSource.y = min(float(g_srcG) / kFxEnergy * invThreads, kMaxSourceEnergy);
  currentSource.z = min(float(g_srcB) / kFxEnergy * invThreads, kMaxSourceEnergy);
  currentSource.w = float(g_skyWeight) / kFxWeight * invThreads;

  // The illuminant is stored pre-multiplied by its own weight, so that a mean
  // taken down the mip chain is a weighted mean without the downsample pass
  // needing to know anything about what it is averaging.
  const float illumWeight = float(g_illumWeight) / kFxWeight * invThreads;
  float4 currentIllum = float4(
    float(g_illumR) / kFxLog * invThreads,
    float(g_illumG) / kFxLog * invThreads,
    float(g_illumB) / kFxLog * invThreads,
    illumWeight);

  // -----------------------------------------------------------------------
  // Temporal accumulation
  // -----------------------------------------------------------------------
  // The field is the most temporally sensitive part of the transform: it sets
  // where the curve's anchor sits, and the input is a denoised path-traced
  // image whose residual noise is worst in exactly the dark regions that get
  // the most adaptation. Left stateless the exposure visibly breathes.
  //
  // It is also where temporal persistence enters the classification. A
  // classification has to survive several frames of reprojection before it
  // fully owns the adaptation, so a single speculative frame of fireball does
  // not re-expose the room, and a lamp that has been a lamp for half a second
  // is trusted completely.
  //
  // The previous frame's cut flag. psdt_state runs after this pass, so what is
  // read here was written a frame ago - which is exactly right: a cut detected
  // last frame means this frame's history describes a scene that is gone. The
  // flag decays over about a quarter of a second rather than clearing, so the
  // field is given several fast frames to re-converge instead of one.
  const float cutFlag = saturate(InState[PSDT_STATE_CUT_FLAG]);
  const float currentWeight = lerp(cb.currentWeight, 1.0, cutFlag);

  if ((cb.flags & PSDT_ANALYSIS_FLAG_HAS_HISTORY) != 0 && currentWeight < 1.0)
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
      // History is constrained to a window around what this block currently
      // measures. The usual temporal-antialiasing argument: history is
      // trustworthy only insofar as it agrees with something the current
      // frame contains. Anything further out came from geometry that is no
      // longer here. The windows are fixed rather than derived from a
      // neighbourhood min/max because the neighbours are being written by
      // other workgroups in this same dispatch and cannot be read safely.
      float4 history = InFieldHistory.SampleLevel(previousUv, 0);
      history.x = clamp(history.x, currentField.x - 3.0, currentField.x + 3.0);
      history.y = clamp(history.y, currentField.y - 0.35, currentField.y + 0.35);
      history.z = clamp(history.z, currentField.z - 3.0, currentField.z + 3.0);
      history.w = clamp(history.w, currentField.w - 2.0, currentField.w + 2.0);
      currentField = lerp(history, currentField, currentWeight);

      float4 sourceHistory = InSourceHistory.SampleLevel(previousUv, 0);
      sourceHistory.xyz = clamp(sourceHistory.xyz,
                                currentSource.xyz * 0.25,
                                currentSource.xyz * 4.0 + 1e-3);
      sourceHistory.w = clamp(sourceHistory.w, currentSource.w - 0.35, currentSource.w + 0.35);
      currentSource = lerp(sourceHistory, currentSource, currentWeight);

      // The illuminant is the slowest-moving thing the field carries and the
      // one that benefits most from a long memory - a room's light does not
      // change colour when the player turns their head, and a per-frame
      // estimate over one 16x16 block is noisy. It is clamped in
      // chromaticity rather than in the premultiplied value, so that a change
      // in confidence alone never looks like a change in colour.
      const float4 illumHistory = InIlluminantHistory.SampleLevel(previousUv, 0);
      const float hw = max(illumHistory.w, 1e-4);
      const float cw = max(currentIllum.w, 1e-4);
      const float3 historyChroma = clamp(illumHistory.xyz / hw,
                                         currentIllum.xyz / cw - 0.35,
                                         currentIllum.xyz / cw + 0.35);
      const float blendedWeight = lerp(illumHistory.w, currentIllum.w, currentWeight);
      const float3 blendedChroma = lerp(historyChroma, currentIllum.xyz / cw,
                                        currentIllum.w > 1e-4 ? currentWeight : 0.0);
      currentIllum = float4(blendedChroma * blendedWeight, blendedWeight);
    }
  }

  OutField[fieldTexel] = currentField;
  OutSource[fieldTexel] = currentSource;
  OutIlluminant[fieldTexel] = currentIllum;
}
