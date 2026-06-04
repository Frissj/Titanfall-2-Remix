/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/dof/dof.h"

layout(binding = DOF_COLOR_INPUT)
Sampler2D<float4> InColor;

layout(binding = DOF_VIEWZ_INPUT)
Sampler2D<float> InViewZ;

layout(binding = DOF_COLOR_OUTPUT)
RWTexture2D<float4> OutColor;

layout(push_constant)
ConstantBuffer<DepthOfFieldArgs> cb;

// Golden-angle spiral for an even disk distribution without precomputed kernels.
static const float kGoldenAngle = 2.39996323f;

// Circle of confusion in pixels, computed the way the game's CoC-generation pass
// does: a linear ramp of normalized depth through the harvested c_dof near/far
// coefficients. viewZ is the path tracer's view-space depth (negative into the
// screen). Because the ramp is saturate(depth01*scale + bias), the game's
// "default/off" config yields ~0 CoC at normal gameplay depths (no blur), and
// only the cinematic near/far config blurs — automatically off in free-roam and
// matching the game in cinematics. strength is the user multiplier.
// Exact CoC formula reverse-engineered from the game's DoF pass (fxc disasm of
// FS_ec047dec…): linear view depth selects a (scale,bias) pair by whether it is
// nearer or farther than nearDepthEnd, then CoC = saturate(depth*scale + bias):
//   isNear = viewDepth < nearDepthEnd                       (dofA.x)
//   (scale,bias) = isNear ? worldParams.xy : worldParams.zw (dofB.xy : dofB.zw)
//   CoC = saturate(viewDepth*scale + bias)
// With the harvested config this is NEAR-field only (worldParams.zw = 0), so the
// distance stays sharp and only close foreground blurs — matching the game. The
// game feeds raw linear view depth, so depthScale defaults to 1.0 (|viewZ| is
// already linear view-space Z in Remix); it only needs tuning if the unit scales
// differ.
float cocPixels(float viewZ)
{
  const float dist   = abs(viewZ) * cb.depthScale;
  const bool  isNear = dist < cb.dofA.x;                       // dofA.x = nearDepthEnd
  const float scale  = isNear ? cb.dofB.x : cb.dofB.z;         // worldParams.x : .z
  const float bias   = isNear ? cb.dofB.y : cb.dofB.w;         // worldParams.y : .w
  const float coc    = saturate(dist * scale + bias);          // [0,1], exactly as the game
  return coc * cb.maxBlurRadius * cb.strength;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 ipos : SV_DispatchThreadID)
{
  if (any(ipos >= cb.imageSize))
  {
    return;
  }

  const float2 uv = (float2(ipos) + float2(0.5f)) * cb.imageSizeInverse;

  const float3 sharp     = InColor.SampleLevel(uv, 0).rgb;
  const float  centerCoC = cocPixels(InViewZ.SampleLevel(uv, 0).r);

  // No meaningful blur (also the strength==0 / DoF-inactive path) -> passthrough.
  if (centerCoC < 0.5f)
  {
    OutColor[ipos] = float4(sharp, 1.0f);
    return;
  }

  float3 acc  = sharp;
  float  wsum = 1.0f;
  const uint taps = max(cb.numTaps, 1u);

  for (uint k = 1u; k <= taps; ++k)
  {
    const float  t   = float(k) / float(taps);
    const float  r   = centerCoC * sqrt(t);                 // even disk coverage
    const float  ang = float(k) * kGoldenAngle;
    const float2 off = float2(cos(ang), sin(ang)) * r;
    const float2 tuv = uv + off * cb.imageSizeInverse;

    // A tap contributes only if its own CoC reaches this sample radius. This
    // stops a sharp foreground object from bleeding its color onto focused /
    // background pixels (the classic DoF "intensity leak").
    const float tapCoC = cocPixels(InViewZ.SampleLevel(tuv, 0).r);
    const float w      = saturate(tapCoC - r + 1.0f);

    acc  += InColor.SampleLevel(tuv, 0).rgb * w;
    wsum += w;
  }

  OutColor[ipos] = float4(acc / wsum, 1.0f);
}
