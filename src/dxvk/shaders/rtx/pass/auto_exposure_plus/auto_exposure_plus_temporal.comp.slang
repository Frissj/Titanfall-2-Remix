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

layout(binding = AUTO_EXPOSURE_PLUS_TEMPORAL_CURRENT_INPUT)
Texture2D<vec4> CurrentFused;

layout(binding = AUTO_EXPOSURE_PLUS_TEMPORAL_HISTORY_INPUT)
Sampler2D HistoryFused;

layout(binding = AUTO_EXPOSURE_PLUS_TEMPORAL_MOTION_INPUT)
Texture2D<float2> PrimaryScreenSpaceMotionVector;

layout(binding = AUTO_EXPOSURE_PLUS_TEMPORAL_OUTPUT)
RWTexture2D<vec4> OutFused;

layout(push_constant)
ConstantBuffer<AutoExposurePlusTemporalArgs> cb;

// Temporal accumulation of the gain field.
//
// Everything upstream of this pass is stateless - the pyramid is rebuilt and collapsed from
// scratch each frame. That would be fine over a rasterized image, which is already temporally
// stable, but the input here is path traced and denoised, and its residual temporal noise sits
// in exactly the dark regions where this operator applies the most gain. Left stateless the
// exposure shimmers and pumps. Accumulating the collapsed field removes that.
//
// It also gives the local correction a time constant, which it otherwise would not have. The
// base auto exposure eases towards its target over rtx.autoExposure.autoExposureSpeed; without
// this pass the local half would snap instantly, so a bright-to-dark transition would show the
// local part pop and the global part ease in behind it. Matching the two speeds (the default)
// makes the whole operator settle as one.
//
// Only the gain is accumulated. The guide has to describe the frame being rendered right now,
// or the guided-filter regression in the apply pass would be fitting a model against stale
// image structure.
[shader("compute")]
[numthreads(AUTO_EXPOSURE_PLUS_GROUP_SIZE, AUTO_EXPOSURE_PLUS_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.fusedExtent))
  {
    return;
  }

  const float2 current = CurrentFused[threadId].xy;

  float gain = current.x;

  if (cb.hasHistory != 0 && cb.currentWeight < 1.0f)
  {
    const float2 uv = (float2(threadId) + 0.5f) / float2(cb.fusedExtent);

    // Motion vectors are stored in pixels at the raytracing (pre-upscale) resolution and point
    // from this frame to the previous one, so dividing by their own extent yields a resolution
    // independent UV offset that can be applied at the gain field's resolution directly.
    // Point sampled rather than filtered, so that a silhouette does not blend a foreground and
    // a background vector into one that reprojects to neither.
    const uint2 motionPixel = min(uint2(uv * float2(cb.motionExtent)), cb.motionExtent - 1);
    const float2 motion = PrimaryScreenSpaceMotionVector[motionPixel];
    const float2 previousUv = uv + motion / float2(cb.motionExtent);

    if (all(previousUv > float2(0.0f)) && all(previousUv < float2(1.0f)))
    {
      float historyGain = HistoryFused.SampleLevel(previousUv, 0).x;

      // Constrain the history to what is currently observable in this neighbourhood. This is
      // the standard temporal-antialiasing argument: history is only trustworthy insofar as it
      // agrees with something the current frame actually contains. Anything outside that range
      // came from geometry that is no longer here - a disocclusion, a cut, a light switching
      // off - and clamping discards it without needing a threshold to decide.
      float minGain = current.x;
      float maxGain = current.x;

      [unroll]
      for (int dy = -1; dy <= 1; ++dy)
      {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
          const int2 neighbour = clamp(int2(threadId) + int2(dx, dy), int2(0), int2(cb.fusedExtent) - 1);
          const float neighbourGain = CurrentFused[uint2(neighbour)].x;

          minGain = min(minGain, neighbourGain);
          maxGain = max(maxGain, neighbourGain);
        }
      }

      historyGain = clamp(historyGain, minGain, maxGain);

      gain = lerp(historyGain, current.x, cb.currentWeight);
    }
  }

  OutFused[threadId] = vec4(gain, current.y, 0.0f, 0.0f);
}
