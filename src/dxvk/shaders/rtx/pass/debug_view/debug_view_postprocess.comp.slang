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

// Loads debug view input and applies any postprocess transformations.
// Note: this shader code is split from debug_view.slangh
// to support gather operations on the result from that shader.

#include "rtx/pass/debug_view/debug_view_args.h"
#include "rtx/pass/debug_view/debug_view_postprocess_binding_indices.h"
#include "rtx/utility/debug_view_indices.h"
#include "rtx/pass/common_binding_indices.h"

#include "rtx/utility/color.slangh"

// Inputs

layout(binding = DEBUG_VIEW_POSTPROCESS_BINDING_CONSTANTS_INPUT)
ConstantBuffer<DebugViewArgs> cb;

layout(rgba32f, binding = DEBUG_VIEW_POSTPROCESS_BINDING_DEBUG_VIEW_INPUT)
Texture2D<float4> DebugView;

// Outputs

// NV-DXVK: explicit vk::image_format so SPIR-V's OpTypeImage Format is set
// (slang's layout(format) syntax doesn't propagate); required for atomic ops.
[[vk::image_format("r32ui")]]
layout(r32ui, binding = DEBUG_VIEW_POSTPROCESS_BINDING_HDR_WAVEFORM_RED_OUTPUT)
RWTexture2D<uint> HDRWaveformRed;
[[vk::image_format("r32ui")]]
layout(r32ui, binding = DEBUG_VIEW_POSTPROCESS_BINDING_HDR_WAVEFORM_GREEN_OUTPUT)
RWTexture2D<uint> HDRWaveformGreen;
[[vk::image_format("r32ui")]]
layout(r32ui, binding = DEBUG_VIEW_POSTPROCESS_BINDING_HDR_WAVEFORM_BLUE_OUTPUT)
RWTexture2D<uint> HDRWaveformBlue;

layout(rgba32f, binding = DEBUG_VIEW_POSTPROCESS_BINDING_DEBUG_VIEW_OUTPUT)
RWTexture2D<float4> Output;

// NV-DXVK [Coverage DebugViewScan]: ground-truth scan of the displayed buffer.
layout(binding = DEBUG_VIEW_POSTPROCESS_BINDING_SURFACE_COVERAGE)
RWStructuredBuffer<uint> SurfaceCoverageBuffer;

// NV-DXVK [Coverage DebugViewScan]: primary surface index at each pixel, to name
// the VS drawing the displayed red. R32_UINT, same (downscaled) extent as DebugView.
layout(binding = DEBUG_VIEW_POSTPROCESS_BINDING_SHARED_SURFACE_INDEX)
RWTexture2D<uint> SharedSurfaceIndex;

void hdrWaveformOutput(ivec2 threadId, RWTexture2D<uint> HDRWaveformInputOutput, float normalizedOutCoordinate)
{
  // Early out if the coordinate is out of range
  // Note: Using >= 1 as the coordinate value is floored, meaning a value of 1 would be one pixel outside the output texture.

  if (normalizedOutCoordinate >= 1.0f || normalizedOutCoordinate < 0.0f)
  {
    return;
  }

  const uvec2 outCoordinate = uvec2(
    uint(threadId.x / cb.hdrWaveformResolutionScaleFactor),
    uint(normalizedOutCoordinate * cb.hdrWaveformResolution.y));

  uint dummyOldValue;
  InterlockedAdd(HDRWaveformInputOutput[outCoordinate], 1, dummyOldValue);
}

// Color codes a value into BGR, with only one channel being used to color code at a time
// Black: 0 value
// Red: values >= than max value
// Blue/Green (0,1]: the remaining values, with blue representing the lower half of valid range and green the rest
vec3 colorCodeIntoBGRexclusive(uint value, uint maxValue)
{
  if (value == 0)
  {
    return vec3(0);
  }
  else if (value >= maxValue)
  {
    return vec3(1, 0, 0);   
  }

  // Color code the rest, [1, maxValue-1], to a non-black BG color, one channel at a time

  value -= 1;
  const uint numValues = maxValue - 1;

  const float numChannels = 2;
  const uint ceilValuesPerChannel = ceil(numValues / numChannels);

  vec3 color = 0;
  const uint channel = value / ceilValuesPerChannel;
  const uint numValuesInCurrentChannel = min(numValues - channel * ceilValuesPerChannel, ceilValuesPerChannel);

  // Color code as (0,1]
  color[channel] = (float(value % numValuesInCurrentChannel) + 1) / numValuesInCurrentChannel;

  // Reswizzle to show lowest values in Blue and highest in Red
  return color.bgr;
}

void postprocess(ivec2 threadId, inout vec4 outValue)
{
  // Color code inf, nan values
  if (cb.enableInfNanViewFlag)
  { 
    bool nanDetected = false;
    bool infDetected = false;

    // Detect nans and infs in pixel neighborhood
    if (cb.colorCodeRadius > 0) 
    {
      int iRadius = cb.colorCodeRadius;
      for (int y = threadId.y - iRadius; y <= threadId.y + iRadius; y++ )
      {
        for (int x = threadId.x - iRadius; x <= threadId.x + iRadius; x++ )
        {
          if (any(ivec2(x, y) >= cb.debugViewResolution) || any(ivec2(x, y) < 0))
          {
            continue;
          }

          vec4 temp = DebugView[ivec2(x, y)];
          
          // Convert invalid NAN debug view value to NaN for NaN detection logic below.
          // This is done here rather than in debug view comp so that this conversion doesn't make it into the accumulation
          if (cb.debugViewIdx == DEBUG_VIEW_NAN && temp.x != 0.0)
          {
            temp = 0.0 / 0.0;
          }

          nanDetected |= any(isnan(temp));
          infDetected |= any(isinf(temp));
        }
      }
    }

    if (nanDetected || infDetected)
    {
      // Note: Override usual warning colors when pseudo color mode is in use as its visualization will not
      // display these properly.
      const bool usePseudoColorWarningColors =
        (cb.displayType == DebugViewDisplayType::Standard) &&
        (cb.pseudoColorMode != PseudoColorMode::Disabled);
      const vec3 kNanWarningColor = usePseudoColorWarningColors ? vec3(0, 0, 0) : vec3(1, 0, 0);
      const vec3 kInfWarningColor = usePseudoColorWarningColors ? vec3(0, 0, 0) : vec3(0, 0, 1);
      const vec3 kCombinedWarningColor =
        select(nanDetected, kNanWarningColor, vec3(0, 0, 0)) +
        select(infDetected, kInfWarningColor, vec3(0, 0, 0));

      outValue = cb.animationTimeSec > 0 
                ? vec4(sin(cb.animationTimeSec * 8) * kCombinedWarningColor, 1) // flash warning color
                : vec4(kCombinedWarningColor, 1);
      return;
    }
  }

  // Switch between Display mode

  if (cb.displayType == DebugViewDisplayType::Standard)
  {
    outValue *= cb.scale;
    outValue = (outValue - vec4(cb.minValue)) / (vec4(cb.maxValue - cb.minValue));

    if (!cb.enableAlphaChannelFlag)
    {
      outValue.a = 1;
    }

    if (cb.pseudoColorMode != PseudoColorMode::Disabled)
    {
      // Determine the input value to colormap based on the specified mode

      float pseudoColorInput;

      switch (cb.pseudoColorMode)
      {
      case PseudoColorMode::Luminance:
        pseudoColorInput = calcBt709Luminance(outValue.rgb);
        break;
      case PseudoColorMode::Red:
        pseudoColorInput = outValue.r;
        break;
      case PseudoColorMode::Green:
        pseudoColorInput = outValue.g;
        break;
      case PseudoColorMode::Blue:
        pseudoColorInput = outValue.b;
        break;
      case PseudoColorMode::Alpha:
        pseudoColorInput = outValue.a;
        break;
      }

      outValue.rgb = turboColormap(pseudoColorInput);
    }

    if (cb.enableGammaCorrectionFlag)
    {
      outValue.rgb = linearToGamma(outValue.rgb);
    }
  }
  else if (cb.displayType == DebugViewDisplayType::BGRExclusiveColor)
  {
    outValue.rgb = colorCodeIntoBGRexclusive(outValue.r, cb.maxValue);
  }
  else if (cb.displayType == DebugViewDisplayType::EV100)
  {
    const float luminance = calcBt709Luminance(outValue.rgb);
    const float ev100 = calcLuminanceEV100(luminance);
    const float normalizedEV = float(ev100 - cb.evMinValue) / cb.evRange;

    outValue = vec4(turboColormap(normalizedEV), 1.0f);
  }
  else if (cb.displayType == DebugViewDisplayType::HDRWaveform)
  {
    vec3 log10OutValue;

    if (cb.enableLuminanceModeFlag)
    {
      log10OutValue = log10(calcBt709Luminance(outValue.rgb));
    }
    else
    {
      log10OutValue = log10(outValue.rgb);
    }

    const vec3 normalizedOutCoordinate = (log10OutValue - cb.log10MinValue) / cb.log10Range;

    hdrWaveformOutput(threadId, HDRWaveformRed, normalizedOutCoordinate.r);
    hdrWaveformOutput(threadId, HDRWaveformGreen, normalizedOutCoordinate.g);
    hdrWaveformOutput(threadId, HDRWaveformBlue, normalizedOutCoordinate.b);
  }
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.debugViewResolution))
  {
    return;
  }
  
  vec4 value = DebugView[threadId];
  const vec4 inVal = value; // pre-colorization input, for the input-vs-output red split

  // Post-processing and overrides. Run BEFORE the scan so the scan classifies the
  // DISPLAYED (post-colorization) value — exactly what reaches the screen. The
  // earlier version scanned the pre-colorization input and missed the on-screen
  // red (the colorization/normalization, not the raw albedo, is what's red).
  postprocess(threadId, value);

  // NV-DXVK [Coverage DebugViewScan]: classify the ACTUAL displayed value (post
  // colorization). Counts red-dominant / NaN / Inf + screen bbox, and co-samples
  // the surface index at red pixels so the readback names the VS on screen.
  if (cb.debugViewIdx == DEBUG_VIEW_RAW_ALBEDO)
  {
    const uint base = COVERAGE_DVSCAN_REGION * uint(COVERAGE_SURFACE_SLOTS);
    InterlockedAdd(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_TOTAL], 1u);
    // Gate-free per-pixel grid dump: if this thread is a grid sample point, store
    // its exact displayed RGB (no threshold, no averaging). One thread writes
    // each grid cell, so plain stores (no atomics needed).
    {
      const uint strideX = max(cb.debugViewResolution.x / COVERAGE_DVSCAN_GRID_COLS, 1u);
      const uint strideY = max(cb.debugViewResolution.y / COVERAGE_DVSCAN_GRID_ROWS, 1u);
      const uint col = threadId.x / strideX;
      const uint row = threadId.y / strideY;
      if (col < COVERAGE_DVSCAN_GRID_COLS && row < COVERAGE_DVSCAN_GRID_ROWS
          && (threadId.x % strideX) == strideX / 2u
          && (threadId.y % strideY) == strideY / 2u)
      {
        const uint idx = row * COVERAGE_DVSCAN_GRID_COLS + col;
        const uint g = base + COVERAGE_DVSCAN_GRID_BASE + idx * 3u;
        const uint3 disp = uint3(saturate(value.rgb) * 255.0f + 0.5f);
        SurfaceCoverageBuffer[g + 0u] = disp.r;
        SurfaceCoverageBuffer[g + 1u] = disp.g;
        SurfaceCoverageBuffer[g + 2u] = disp.b;
      }
    }
    const bool nanPx = any(isnan(value.rgb)) || any(value.rgb != value.rgb);
    const bool infPx = any(isinf(value.rgb));
    const bool redPx = value.r > 0.4f
                    && value.r > value.g * 1.5f + 0.05f
                    && value.r > value.b * 1.5f + 0.05f;
    if (redPx)
    {
      InterlockedAdd(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_RED], 1u);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_RED_MAXX], threadId.x);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_RED_MINX], COVERAGE_REDSCR_BIAS - threadId.x);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_RED_MAXY], threadId.y);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_RED_MINY], COVERAGE_REDSCR_BIAS - threadId.y);
    }
    if (nanPx)
    {
      InterlockedAdd(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_NAN], 1u);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_NAN_MAXX], threadId.x);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_NAN_MINX], COVERAGE_REDSCR_BIAS - threadId.x);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_NAN_MAXY], threadId.y);
      InterlockedMax(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_NAN_MINY], COVERAGE_REDSCR_BIAS - threadId.y);
    }
    if (infPx)
    {
      InterlockedAdd(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_INF], 1u);
    }
    // Same red gate on the PRE-colorization input — localizes whether the red is
    // already in the buffer (input red ~= output red) or made by colorization.
    if (inVal.r > 0.4f && inVal.r > inVal.g * 1.5f + 0.05f && inVal.r > inVal.b * 1.5f + 0.05f)
    {
      InterlockedAdd(SurfaceCoverageBuffer[base + COVERAGE_DVSCAN_SLOT_INPUTRED], 1u);
    }
    // Co-sample the primary surface index at red-LEANING pixels (looser than the
    // strict red-dominant gate above so an orange-red plane still registers) and
    // bin per-surfaceIndex → the readback maps the busiest bin to its VS. This is
    // what actually names the shader drawing the on-screen red, sidestepping the
    // opaque-path covRawAlbedo which has been disagreeing with the display.
    const bool redLean = value.r > 0.2f && value.r > value.g + 0.1f && value.r > value.b + 0.1f;
    if (redLean)
    {
      const uint surfIdx = SharedSurfaceIndex[threadId];
      if (surfIdx < uint(COVERAGE_SURFACE_SLOTS))
      {
        InterlockedAdd(SurfaceCoverageBuffer[COVERAGE_DVRED_SURFACE_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], 1u);
      }
    }
  }

  // NV-DXVK [Coverage PickRegion]: color-independent VS attribution of a
  // configurable screen rectangle (default bottom-right). Names whatever shader
  // draws there — even untinted geometry the red-gated DVRED/PureRed probes never
  // see (e.g. the TF2 first-person weapon). Runs for ANY debug view since the
  // primary SharedSurfaceIndex is valid regardless of the displayed buffer.
  {
    const vec4 rect = cb.surfaceCoveragePickRegion; // minX, minY, maxX, maxY in [0,1]
    if (rect.z > rect.x && rect.w > rect.y)
    {
      const vec2 uv = (vec2(threadId) + 0.5f) / vec2(cb.debugViewResolution);
      if (all(uv >= rect.xy) && all(uv <= rect.zw))
      {
        const uint pbase = COVERAGE_PICKREGION_SUMMARY_REGION * uint(COVERAGE_SURFACE_SLOTS);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_TOTAL], 1u);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MAXX], threadId.x);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MINX], COVERAGE_REDSCR_BIAS - threadId.x);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MAXY], threadId.y);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MINY], COVERAGE_REDSCR_BIAS - threadId.y);

        // NV-DXVK: sum the displayed pixel colour (value = this debug view's
        // output for this pixel) so the readback can report the mean colour
        // under the pick. *256 then /256/N on the CPU keeps sub-1.0 detail.
        const uint3 colQ = uint3(saturate(value.rgb) * 256.0f + 0.5f);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLR], colQ.r);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLG], colQ.g);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLB], colQ.b);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLN], 1u);

        const uint surfIdx = SharedSurfaceIndex[threadId];
        if (surfIdx < uint(COVERAGE_SURFACE_SLOTS))
        {
          InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_VALID], 1u);
          InterlockedAdd(SurfaceCoverageBuffer[COVERAGE_PICKREGION_SURFACE_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], 1u);
          // Per-surface screen bbox (biased InterlockedMax) so the readback can
          // report a location box per VS, not just one box for the whole rect.
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION_BOX_MAXX_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], threadId.x);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION_BOX_MINX_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], COVERAGE_REDSCR_BIAS - threadId.x);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION_BOX_MAXY_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], threadId.y);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION_BOX_MINY_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], COVERAGE_REDSCR_BIAS - threadId.y);
        }
        else
        {
          InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_INVALID], 1u);
        }
      }
    }
  }

  // NV-DXVK [Coverage PickRegion2]: second independent rect (same slot layout
  // in its own regions 65/66/67-70) so two screen points are attributed in one
  // frame — driven by rtx.surfaceCoveragePickRegion2.
  {
    const vec4 rect = cb.surfaceCoveragePickRegion2;
    if (rect.z > rect.x && rect.w > rect.y)
    {
      const vec2 uv = (vec2(threadId) + 0.5f) / vec2(cb.debugViewResolution);
      if (all(uv >= rect.xy) && all(uv <= rect.zw))
      {
        const uint pbase = COVERAGE_PICKREGION2_SUMMARY_REGION * uint(COVERAGE_SURFACE_SLOTS);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_TOTAL], 1u);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MAXX], threadId.x);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MINX], COVERAGE_REDSCR_BIAS - threadId.x);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MAXY], threadId.y);
        InterlockedMax(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_MINY], COVERAGE_REDSCR_BIAS - threadId.y);

        const uint3 colQ = uint3(saturate(value.rgb) * 256.0f + 0.5f);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLR], colQ.r);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLG], colQ.g);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLB], colQ.b);
        InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_COLN], 1u);

        const uint surfIdx = SharedSurfaceIndex[threadId];
        if (surfIdx < uint(COVERAGE_SURFACE_SLOTS))
        {
          InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_VALID], 1u);
          InterlockedAdd(SurfaceCoverageBuffer[COVERAGE_PICKREGION2_SURFACE_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], 1u);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION2_BOX_MAXX_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], threadId.x);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION2_BOX_MINX_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], COVERAGE_REDSCR_BIAS - threadId.x);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION2_BOX_MAXY_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], threadId.y);
          InterlockedMax(SurfaceCoverageBuffer[COVERAGE_PICKREGION2_BOX_MINY_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfIdx], COVERAGE_REDSCR_BIAS - threadId.y);
        }
        else
        {
          InterlockedAdd(SurfaceCoverageBuffer[pbase + COVERAGE_PICKREGION_SLOT_INVALID], 1u);
        }
      }
    }
  }

  Output[threadId] = value;
}
