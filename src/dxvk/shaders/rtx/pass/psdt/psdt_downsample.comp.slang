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
// PSDT adaptation pyramid: field mip k -> mip k+1
// ===========================================================================
//
// The mip chain is two things at once and neither of them is a mip chain.
//
// It is the multi-scale adaptation pyramid: level k pools 16 * 2^k pixels, so
// levels 0, 2 and 4 are the small, medium and large adaptation neighbourhoods
// the transform pools between. And it is the glare kernel: a weighted sum
// down the levels of the source field is a sum of progressively wider box
// responses, which is a serviceable point-spread function for free.
//
// Three textures, three reduction rules, and the rules are not
// interchangeable:
//
//   body field    a weight-normalised *robust* mean. Averaging the stored
//                 means directly would give a block that is 1% wall and 99%
//                 lamp the same say as one that is all wall - that is what the
//                 weight fixes. The robust part fixes the other half of the
//                 problem: four children that are two dark walls and two
//                 sunlit ones average to a value describing neither, and a
//                 level built out of those values describes a scene that is
//                 not there. See the reweight below.
//   source field  an area mean. It has to conserve energy per unit area down
//                 the chain or the glare kernel's falloff stops being a
//                 property of the source and starts being a property of the
//                 pyramid.
//   illuminant    a plain mean, because the analysis pass already stored it
//                 premultiplied by its own confidence. That is the entire
//                 reason it is stored that way: this pass does not have to
//                 know what it is averaging.
//
// The peak channel is a max, so the scene white estimate survives the whole
// chain instead of being averaged into the mean it is supposed to bound.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"

layout(binding = PSDT_DOWNSAMPLE_FIELD_INPUT)
Texture2D<float4> InField;

layout(binding = PSDT_DOWNSAMPLE_FIELD_OUTPUT)
RWTexture2D<vec4> OutField;

layout(binding = PSDT_DOWNSAMPLE_SOURCE_INPUT)
Texture2D<float4> InSource;

layout(binding = PSDT_DOWNSAMPLE_SOURCE_OUTPUT)
RWTexture2D<vec4> OutSource;

layout(binding = PSDT_DOWNSAMPLE_ILLUM_INPUT)
Texture2D<float4> InIlluminant;

layout(binding = PSDT_DOWNSAMPLE_ILLUM_OUTPUT)
RWTexture2D<vec4> OutIlluminant;

layout(push_constant)
ConstantBuffer<PsdtDownsampleArgs> cb;

[shader("compute")]
[numthreads(PSDT_GROUP_SIZE, PSDT_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.dstExtent))
  {
    return;
  }

  // Odd source extents leave the last row or column without a partner.
  // Clamping rather than skipping keeps the edge texels weighted like the
  // interior ones, which matters because the coarsest levels are only a few
  // texels across and an edge artifact there is an adaptation artifact across
  // a quarter of the screen.
  const uint2 base = threadId * 2;
  const uint2 maxCoord = cb.srcExtent - 1;

  const uint2 c00 = min(base, maxCoord);
  const uint2 c10 = min(base + uint2(1, 0), maxCoord);
  const uint2 c01 = min(base + uint2(0, 1), maxCoord);
  const uint2 c11 = min(base + uint2(1, 1), maxCoord);

  const float4 f00 = InField[c00];
  const float4 f10 = InField[c10];
  const float4 f01 = InField[c01];
  const float4 f11 = InField[c11];

  const float weightSum = f00.y + f10.y + f01.y + f11.y;
  const float logSum = f00.x * f00.y + f10.x * f10.y + f01.x * f01.y + f11.x * f11.y;
  const float depthSum = f00.w * f00.y + f10.w * f10.y + f01.w * f01.y + f11.w * f11.y;

  float4 outField;
  // A block whose four children are all pure light source carries no body
  // weight at all. Dividing by that weight would be a divide by zero; falling
  // back to the unweighted mean of the children keeps the level continuous,
  // and the near-zero .y it also writes tells the transform downstream to
  // disregard this value and use the frame anchor instead.
  if (weightSum > 1e-4)
  {
    outField.x = logSum / weightSum;
    outField.w = depthSum / weightSum;

    // Robust reweight: one Welsch step about the weighted mean.
    //
    // This is Auto Exposure Plus's edge-aware downsample, in the form a 2x2
    // reduction can carry. Plus used a 23-tap separable Kuwahara - two
    // one-sided means, whichever has the lower variance wins - which needs a
    // kernel wide enough for "the two sides" to mean something and has nothing
    // to say about four texels. What survives the translation is the idea
    // rather than the filter: a level should describe the population most of
    // it belongs to, not the average of two populations it straddles.
    //
    // So each child is additionally weighted by how far it sits from its
    // siblings' mean, on a Gaussian of width `coherenceSigma` stops. Four
    // children within a stop of each other are unaffected. Three dark walls
    // and a sunlit one produce a value that is the walls, and the sunlit child
    // is left to be described by the finer level it came from - which still
    // exists, is still pooled, and is where a value that small belongs.
    //
    // One step rather than iterating to convergence, because the initial
    // estimate is a weighted mean of four samples and a second step moves it
    // by less than the field's own temporal noise. Depth is reweighted with
    // the same factors rather than its own: the depth channel exists to say
    // which surface the luminance value describes, so it has to follow the
    // luminance population or it describes a different one.
    if (cb.coherenceSigma > 1e-3)
    {
      const float invSigma = 1.0 / cb.coherenceSigma;
      const float4 d = float4(f00.x, f10.x, f01.x, f11.x) - outField.x;
      const float4 z = d * invSigma;
      const float4 rho = exp(-z * z);

      const float4 w2 = float4(f00.y, f10.y, f01.y, f11.y) * rho;
      const float w2Sum = w2.x + w2.y + w2.z + w2.w;

      // The guard is not decorative: with a narrow sigma and four widely
      // separated children every rho can underflow, and the reduction would
      // divide zero by zero rather than degrade to the mean it started from.
      if (w2Sum > 1e-4)
      {
        outField.x = dot(w2, float4(f00.x, f10.x, f01.x, f11.x)) / w2Sum;
        outField.w = dot(w2, float4(f00.w, f10.w, f01.w, f11.w)) / w2Sum;
      }
    }
  }
  else
  {
    outField.x = 0.25 * (f00.x + f10.x + f01.x + f11.x);
    outField.w = 0.25 * (f00.w + f10.w + f01.w + f11.w);
  }

  outField.y = weightSum * 0.25;
  outField.z = max(max(f00.z, f10.z), max(f01.z, f11.z));

  OutSource[threadId] = 0.25 * (InSource[c00] + InSource[c10] + InSource[c01] + InSource[c11]);
  OutIlluminant[threadId] = 0.25 * (InIlluminant[c00] + InIlluminant[c10]
                                  + InIlluminant[c01] + InIlluminant[c11]);
  OutField[threadId] = outField;
}
