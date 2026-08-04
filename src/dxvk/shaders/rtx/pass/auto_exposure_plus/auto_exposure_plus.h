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
#ifndef AUTO_EXPOSURE_PLUS_H
#define AUTO_EXPOSURE_PLUS_H

#include "rtx/utility/shader_types.h"

// Radius of the separable edge-aware (Kuwahara) downsample kernel. Taps run over [-R, R].
#define AUTO_EXPOSURE_PLUS_KERNEL_RADIUS 11

// Workgroup edge length. Both the downsample and the other passes dispatch 16x16.
#define AUTO_EXPOSURE_PLUS_GROUP_SIZE 16

// Length of the groupshared tile along the axis being filtered.
//
// A 16 wide group of destination texels spans 16 * 2 source texels, plus the kernel radius on
// each side, plus one for the bilinear partner of the outermost tap: 32 + 2 * 12 = 56. Rounded
// up to 64 for a power of two, which costs 64 * 16 * 8 = 8 KiB of groupshared.
//
// Levels whose source-to-destination ratio exceeds 2 (only reachable at odd, very small mip
// extents) would need a longer span; the shader detects that and falls back to direct loads.
#define AUTO_EXPOSURE_PLUS_TILE_SPAN 64

// Pyramid levels at or above this index are reconstructed with a bicubic B-spline filter rather
// than bilinear. Below it the upsampling ratio is small enough that bilinear does not produce
// visible facets, and bilinear is four times cheaper.
#define AUTO_EXPOSURE_PLUS_BSPLINE_LEVEL 4

// Middle grey. The pyramid stores log2 luminance relative to this, so a correctly exposed pixel
// sits at 0 and every value in the pyramid is in units of photographic stops.
#define AUTO_EXPOSURE_PLUS_MIDDLE_GREY 0.18f

// Luminance floor applied before the log2, ~10.5 stops below middle grey. Bounds how far a
// black pixel can pull the key down, and with it how much gain such a pixel can ask for.
#define AUTO_EXPOSURE_PLUS_MIN_LUMINANCE 1e-4f

// Fraction of each channel mixed into the other two before the key is measured. Bounds how far
// apart the channels can get, which is what keeps the geometric mean in
// autoExposurePlusLogLuminance from collapsing on a saturated or deeply shadowed pixel. Read the
// comment there before changing it - lowering this towards 0 reintroduces the collapse, and
// raising it desaturates the measurement until the key stops tracking colour at all.
#define AUTO_EXPOSURE_PLUS_CHANNEL_MIX 0.132f

#define AUTO_EXPOSURE_PLUS_INIT_COLOR_INPUT           0
#define AUTO_EXPOSURE_PLUS_INIT_PYRAMID_OUTPUT        1
#define AUTO_EXPOSURE_PLUS_INIT_EXPOSURE              2

#define AUTO_EXPOSURE_PLUS_DOWNSAMPLE_INPUT           0
#define AUTO_EXPOSURE_PLUS_DOWNSAMPLE_OUTPUT          1

#define AUTO_EXPOSURE_PLUS_FUSE_PYRAMID_INPUT         0
#define AUTO_EXPOSURE_PLUS_FUSE_COLOR_INPUT           1
#define AUTO_EXPOSURE_PLUS_FUSE_OUTPUT                2
#define AUTO_EXPOSURE_PLUS_FUSE_EXPOSURE              3

#define AUTO_EXPOSURE_PLUS_TEMPORAL_CURRENT_INPUT     0
#define AUTO_EXPOSURE_PLUS_TEMPORAL_HISTORY_INPUT     1
#define AUTO_EXPOSURE_PLUS_TEMPORAL_MOTION_INPUT      2
#define AUTO_EXPOSURE_PLUS_TEMPORAL_OUTPUT            3

#define AUTO_EXPOSURE_PLUS_APPLY_FUSED_INPUT          0
#define AUTO_EXPOSURE_PLUS_APPLY_PYRAMID_INPUT        1
#define AUTO_EXPOSURE_PLUS_APPLY_COLOR_INPUT_OUTPUT   2
#define AUTO_EXPOSURE_PLUS_APPLY_DEBUG_VIEW_OUTPUT    3
#define AUTO_EXPOSURE_PLUS_APPLY_EXPOSURE             4

// Constant buffers

struct AutoExposurePlusInitArgs
{
  uvec2 colorExtent;
  uvec2 pyramidExtent;

  float exposure;
  float intensity;
  uint enableAutoExposure;
  uint pad0;
};

struct AutoExposurePlusDownsampleArgs
{
  uvec2 srcExtent;
  uvec2 dstExtent;

  // 1 when blurring/decimating along X, 0 when along Y.
  uint horizontal;
  uint pad0;
  uint pad1;
  uint pad2;
};

struct AutoExposurePlusFuseArgs
{
  uvec2 colorExtent;
  uvec2 pyramidExtent;

  float exposure;
  float equalizationStrength;
  float equalizationPivot;
  uint lowestLevel;

  uint enableAutoExposure;
  uint pad0;
  uint pad1;
  uint pad2;
};

struct AutoExposurePlusTemporalArgs
{
  uvec2 fusedExtent;
  uvec2 motionExtent;

  // Blend weight towards the current frame. 1 means history is ignored entirely.
  float currentWeight;
  uint hasHistory;
  uint pad0;
  uint pad1;
};

struct AutoExposurePlusApplyArgs
{
  vec4 fusedTexelSize;

  uvec2 colorExtent;
  float exposure;
  float targetEV;

  float maxGainEV;
  uint enableAutoExposure;
  uint debugView;
  uint debugPyramidLevel;
};

#endif  // AUTO_EXPOSURE_PLUS_H
