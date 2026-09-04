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
// PSDT adaptation pyramid: field level k -> level k+1
// ===========================================================================
//
// The mip chain is doing two jobs at once and is built to serve both.
//
//   As an adaptation pyramid, level k is the eye's estimate of the
//   surrounding luminance over a neighbourhood 16 * 2^k pixels wide. Three of
//   these levels, pooled with scene-dependent weights, are the local
//   adaptation signal - because adaptation is a property of a neighbourhood,
//   and a single radius cannot tell "a large bright room" from "a small
//   bright lamp".
//
//   As a glare kernel, the same levels are a stack of progressively wider box
//   responses over the source energy. Weighted and summed they approximate
//   the eye's glare point-spread function, so the glare model costs no passes
//   of its own and inherits the pyramid's temporal stability for free.
//
// Each channel is reduced the way its meaning requires, which is the whole
// reason this is not a generic mip generator:
//
//   .x  body log luminance   weighted mean, weighted by .y. An unweighted
//                            mean would let a block that is 95% lamp - and
//                            therefore has almost no opinion - contribute as
//                            much as a block that is entirely wall.
//   .y  body weight          plain mean. It is a fraction of area, so area
//                            averaging is exactly right.
//   .z  source energy        plain mean, and the reason .z is stored linearly
//                            rather than as a log: a mean of linear energy
//                            over an area is energy-conserving, which is what
//                            the glare sum needs. A geometric mean is not.
//   .w  block peak           max, so the scene white estimate survives to the
//                            coarse levels where psdt_state reads it. Averaged
//                            it would decay towards the mean and the state
//                            would lose sight of the brightest thing present.
// ===========================================================================

#include "rtx/pass/psdt/psdt.h"

layout(binding = PSDT_DOWNSAMPLE_FIELD_INPUT)
Texture2D<float4> InField;

layout(binding = PSDT_DOWNSAMPLE_FIELD_OUTPUT)
RWTexture2D<vec4> OutField;

layout(binding = PSDT_DOWNSAMPLE_CHROMA_INPUT)
Texture2D<float4> InChroma;

layout(binding = PSDT_DOWNSAMPLE_CHROMA_OUTPUT)
RWTexture2D<vec4> OutChroma;

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

  // Odd source extents leave the last row or column without a partner. Clamping
  // rather than skipping keeps the edge texels weighted like the interior ones,
  // which matters because the coarsest levels are only a few texels across and
  // an edge artifact there is an adaptation artifact across a quarter of the
  // screen.
  const uint2 base = threadId * 2;
  const uint2 maxCoord = cb.srcExtent - 1;

  float4 sum = float4(0.0);
  float weightedLog = 0.0;
  float weightSum = 0.0;
  float peak = -1e4;
  float2 chromaSum = float2(0.0);

  [unroll]
  for (int dy = 0; dy < 2; ++dy)
  {
    [unroll]
    for (int dx = 0; dx < 2; ++dx)
    {
      const uint2 coord = min(base + uint2(dx, dy), maxCoord);
      const float4 s = InField[coord];

      weightedLog += s.x * s.y;
      weightSum += s.y;
      sum.y += s.y;
      sum.z += s.z;
      peak = max(peak, s.w);
      chromaSum += InChroma[coord].xy;
    }
  }

  float4 outValue;
  // A block whose four children are all pure light source carries no body
  // weight at all. Dividing by that weight would be a divide by zero; falling
  // back to the unweighted mean of the children's log luminance keeps the
  // level continuous, and the near-zero .y it also writes tells the transform
  // downstream to disregard this value and use the frame anchor instead.
  if (weightSum > 1e-4)
  {
    outValue.x = weightedLog / weightSum;
  }
  else
  {
    outValue.x = 0.25 * (InField[min(base, maxCoord)].x
                       + InField[min(base + uint2(1, 0), maxCoord)].x
                       + InField[min(base + uint2(0, 1), maxCoord)].x
                       + InField[min(base + uint2(1, 1), maxCoord)].x);
  }

  outValue.y = sum.y * 0.25;
  outValue.z = sum.z * 0.25;
  outValue.w = peak;

  OutField[threadId] = outValue;
  OutChroma[threadId] = vec4(chromaSum * 0.25, 0.0, 0.0);
}
