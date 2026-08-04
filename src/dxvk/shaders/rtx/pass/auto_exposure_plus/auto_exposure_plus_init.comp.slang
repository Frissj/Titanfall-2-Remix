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

layout(binding = AUTO_EXPOSURE_PLUS_INIT_COLOR_INPUT)
Texture2D<float4> InColorBuffer;

layout(binding = AUTO_EXPOSURE_PLUS_INIT_PYRAMID_OUTPUT)
RWTexture2D<vec4> OutPyramid;

layout(binding = AUTO_EXPOSURE_PLUS_INIT_EXPOSURE)
RWTexture1D<float> InExposure;

layout(push_constant)
ConstantBuffer<AutoExposurePlusInitArgs> cb;

// Seeds level 0 of the pyramid from the linear HDR radiance.
//
// Two channels are carried through the whole pyramid:
//   .y  the key   - log luminance in stops relative to middle grey. Drives the edge-stopping
//                   weights during the collapse; never modified, only filtered.
//   .x  the gain  - the exposure correction this pixel would want in isolation, in stops.
//                   Dark pixels (negative key) ask to be brightened and bright ones to be
//                   darkened, so the field is a partial pull towards a key of 0.
//
// The gain is a fixed *fraction* of the key rather than the whole of it - the 0.5 factor means
// intensity 1.0 pulls a pixel halfway to middle grey, and the default intensity a good deal
// less. Pulling all the way would equalize every pixel to the same brightness, which is a flat
// grey wash, not a photograph. It also means the correction scales with however many stops the
// key spans, so the intensity that looks right depends on the scene's dynamic range.
//
// Note that the base auto exposure has already normalised the frame as a whole to middle grey
// by this point, so what this measures is how far each *region* sits from the frame average -
// which is exactly the quantity the local correction needs.
[shader("compute")]
[numthreads(AUTO_EXPOSURE_PLUS_GROUP_SIZE, AUTO_EXPOSURE_PLUS_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.pyramidExtent))
  {
    return;
  }

  const float avgExposure = getExposure(InExposure, cb.enableAutoExposure, cb.exposure);
  const float3 radiance = autoExposurePlusLoadHalfResColor(InColorBuffer, threadId, cb.colorExtent) * avgExposure;

  const float key = autoExposurePlusLogLuminance(radiance);

  OutPyramid[threadId] = vec4(-key * cb.intensity * 0.5f, key, 0.0f, 0.0f);
}
