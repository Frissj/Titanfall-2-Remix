/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/auto_exposure_plus/auto_exposure_plus.slangh"

layout(binding = AUTO_EXPOSURE_PLUS_FUSE_PYRAMID_INPUT)
Sampler2D PyramidTexture;

layout(binding = AUTO_EXPOSURE_PLUS_FUSE_COLOR_INPUT)
Texture2D<vec4> InColorBuffer;

layout(binding = AUTO_EXPOSURE_PLUS_FUSE_OUTPUT)
RWTexture2D<vec4> OutFused;

layout(binding = AUTO_EXPOSURE_PLUS_FUSE_EXPOSURE)
RWTexture1D<float> InExposure;

layout(push_constant)
ConstantBuffer<AutoExposurePlusFuseArgs> cb;

float2 samplePyramidLevel(float2 uv, uint level, uint2 pyramidExtent)
{
  const float2 levelExtent = float2(max(pyramidExtent >> uint2(level), uint2(1)));

  if (level >= AUTO_EXPOSURE_PLUS_BSPLINE_LEVEL)
  {
    return autoExposurePlusSampleBSpline(PyramidTexture, uv, levelExtent, float(level));
  }

  return PyramidTexture.SampleLevel(uv, float(level)).xy;
}

// Collapses the pyramid into a single gain field.
//
// This is where the operator differs most from a textbook Mertens fusion, and it is worth being
// precise about why. A Laplacian fusion reconstructs by *summing* band-pass differences from
// coarse to fine; every level contributes everywhere, so an error at a coarse level is spread
// over the whole image and shows up as a halo.
//
// This instead starts at the coarsest level and walks down, at each step *choosing* between the
// running estimate and the next finer level based on how much they disagree about the local
// key. Where they agree, the coarse estimate is kept - large smooth regions get a single
// consistent exposure. Where they disagree, a feature boundary lies between the two scales and
// the finer level takes over, so the gain field breaks cleanly at edges instead of ramping
// across them.
//
// The coarsest level is a handful of texels covering the whole frame. Since the base auto
// exposure has already anchored the frame average to middle grey, that level ends up close to
// zero and the operator's job is genuinely local - it is refining the base exposure, not
// competing with it.
[shader("compute")]
[numthreads(AUTO_EXPOSURE_PLUS_GROUP_SIZE, AUTO_EXPOSURE_PLUS_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.pyramidExtent))
  {
    return;
  }

  const float2 uv = (float2(threadId) + 0.5f) / float2(cb.pyramidExtent);

  float2 fused = samplePyramidLevel(uv, cb.lowestLevel, cb.pyramidExtent);

  for (int level = int(cb.lowestLevel) - 1; level >= 0; --level)
  {
    const float2 finer = samplePyramidLevel(uv, uint(level), cb.pyramidExtent);

    fused = lerp(fused, finer, autoExposurePlusEdgeStop(finer.y, fused.y, cb.equalizationStrength, cb.equalizationPivot));
  }

  // The key is no longer needed past this point, so the second channel is repurposed to carry
  // the guide for the guided-filter upsample in the apply pass. Both must be measured from the
  // same 2x2 reduction of the colour buffer that seeded the pyramid.
  const float avgExposure = getExposure(InExposure, cb.enableAutoExposure, cb.exposure);
  const float3 radiance = autoExposurePlusLoadHalfResColor(InColorBuffer, threadId, cb.colorExtent) * avgExposure;

  OutFused[threadId] = vec4(fused.x, autoExposurePlusGuide(radiance), 0.0f, 0.0f);
}
