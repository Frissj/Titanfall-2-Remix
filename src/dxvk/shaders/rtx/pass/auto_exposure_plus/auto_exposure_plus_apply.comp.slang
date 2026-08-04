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

[[vk::binding(AUTO_EXPOSURE_PLUS_APPLY_FUSED_INPUT)]]
Sampler2D FusedTexture;

[[vk::binding(AUTO_EXPOSURE_PLUS_APPLY_PYRAMID_INPUT)]]
Sampler2D PyramidTexture;

[[vk::binding(AUTO_EXPOSURE_PLUS_APPLY_COLOR_INPUT_OUTPUT)]]
RWTexture2D<vec4> InOutColorBuffer;

[[vk::binding(AUTO_EXPOSURE_PLUS_APPLY_DEBUG_VIEW_OUTPUT)]]
RWTexture2D<float4> DebugView;

[[vk::binding(AUTO_EXPOSURE_PLUS_APPLY_EXPOSURE)]]
RWTexture1D<float> InExposure;

#include "rtx/utility/debug_view_helpers.slangh"

layout(push_constant)
ConstantBuffer<AutoExposurePlusApplyArgs> cb;

[shader("compute")]
[numthreads(AUTO_EXPOSURE_PLUS_GROUP_SIZE, AUTO_EXPOSURE_PLUS_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.colorExtent))
  {
    return;
  }

  const float2 uv = (float2(threadId) + 0.5f) / float2(cb.colorExtent);

  const float3 radiance = InOutColorBuffer[threadId].xyz;

  // The base exposure is used to *measure* only - it tells the operator where middle grey sits
  // so that the gain field is expressed relative to a correctly exposed frame. It is
  // deliberately not folded into the output: the tonemapper downstream applies it itself, and
  // baking it in here would apply it twice.
  const float avgExposure = getExposure(InExposure, cb.enableAutoExposure, cb.exposure);
  const float3 normalizedRadiance = radiance * avgExposure;

  // Guided upsample of the half resolution gain field.
  //
  // A plain bilinear upsample would let the gain field cross edges that the pyramid was careful
  // not to cross. Instead a local linear model gain = A * guide + B is fitted over a 3x3
  // neighbourhood of the coarse field, then evaluated against this pixel's own full resolution
  // guide - so the reconstructed gain follows the full resolution edges exactly.
  // See https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/
  float momentX = 0.0f;
  float momentY = 0.0f;
  float momentX2 = 0.0f;
  float momentXY = 0.0f;
  float weightSum = 0.0f;

  [unroll]
  for (int dy = -1; dy <= 1; ++dy)
  {
    [unroll]
    for (int dx = -1; dx <= 1; ++dx)
    {
      const float2 fusedSample = FusedTexture.SampleLevel(uv + float2(dx, dy) * cb.fusedTexelSize.zw, 0).xy;

      // x is the guide, y is the gain being reconstructed.
      const float x = fusedSample.y;
      const float y = fusedSample.x;
      const float weight = exp(-float(dx * dx + dy * dy));

      momentX += x * weight;
      momentY += y * weight;
      momentX2 += x * x * weight;
      momentXY += x * y * weight;
      weightSum += weight;
    }
  }

  momentX /= weightSum;
  momentY /= weightSum;
  momentX2 /= weightSum;
  momentXY /= weightSum;

  const float A = (momentXY - momentX * momentY) / (max(momentX2 - momentX * momentX, 0.0f) + 1e-5f);
  const float B = momentY - A * momentX;

  const float guide = autoExposurePlusGuide(normalizedRadiance);

  // The local correction, followed by the global brightness offset.
  //
  // These are deliberately independent: intensity is how strongly regions are pulled towards
  // each other, targetEV is where the whole image sits. Scaling targetEV by the intensity would
  // couple them, and would silently kill the brightness control whenever the local correction
  // was turned off. Here setting intensity to zero leaves targetEV working as a plain global
  // exposure offset.
  float gainEV = A * guide + B;
  gainEV += cb.targetEV;

  // Linear HDR radiance is unbounded, so nothing upstream bounds the regression: a single
  // firefly or a sun disc inside the 3x3 neighbourhood can produce an arbitrarily steep slope A,
  // and the gain that falls out of it is arbitrarily large. Clamping bounds that without
  // altering the operator anywhere the input is sane - at the default this should not engage on
  // well behaved content, so if it is engaging widely something upstream is wrong.
  gainEV = clamp(gainEV, -cb.maxGainEV, cb.maxGainEV);

  // Linear HDR in, linear HDR out. Bloom, post FX and the tone curve all run after this and are
  // untouched by it - they simply receive a better exposed image.
  InOutColorBuffer[threadId] = vec4(radiance * exp2(gainEV), 1.0f);

  // Diagnostics go to the debug view target rather than the colour buffer, because unlike the
  // tonemapper this pass is not the last writer - anything written to the colour buffer here
  // would still be bloomed and tone mapped before it reached the screen.
  if (cb.debugView == DEBUG_VIEW_AUTO_EXPOSURE_PLUS_GAIN)
  {
    // Remapped so that no gain reads as mid grey, brightening as lighter and darkening darker.
    storeInDebugView(threadId, float3(gainEV / (2.0f * cb.maxGainEV) + 0.5f));
  }
  else if (cb.debugView == DEBUG_VIEW_AUTO_EXPOSURE_PLUS_GUIDE)
  {
    storeInDebugView(threadId, float3(guide));
  }
  else if (cb.debugView == DEBUG_VIEW_AUTO_EXPOSURE_PLUS_PYRAMID)
  {
    // Inspect one pyramid level directly. When the output looks wrong the useful question is
    // always which level went wrong, and that is not answerable from the final gain alone.
    // Left half shows the key (log luminance in stops, mid grey = middle grey, one stop per
    // 1/16th of the range); right half shows that level's gain contribution on the same scale
    // as the gain debug view.
    const float2 level = PyramidTexture.SampleLevel(uv, float(cb.debugPyramidLevel)).xy;

    const float displayed = (threadId.x < cb.colorExtent.x / 2)
      ? (level.y / 16.0f + 0.5f)
      : (level.x / (2.0f * cb.maxGainEV) + 0.5f);

    storeInDebugView(threadId, float3(displayed));
  }
}
