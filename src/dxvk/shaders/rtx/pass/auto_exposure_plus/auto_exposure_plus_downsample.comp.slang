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

layout(binding = AUTO_EXPOSURE_PLUS_DOWNSAMPLE_INPUT)
Texture2D<vec4> InPyramidLevel;

layout(binding = AUTO_EXPOSURE_PLUS_DOWNSAMPLE_OUTPUT)
RWTexture2D<vec4> OutPyramidLevel;

layout(push_constant)
ConstantBuffer<AutoExposurePlusDownsampleArgs> cb;

// Tile indexed as [alongAxis][offAxis] so that one layout serves both orientations: for the
// horizontal pass the axis is X, for the vertical pass it is Y.
groupshared float2 g_tile[AUTO_EXPOSURE_PLUS_TILE_SPAN * AUTO_EXPOSURE_PLUS_GROUP_SIZE];

static const uint kThreadCount = AUTO_EXPOSURE_PLUS_GROUP_SIZE * AUTO_EXPOSURE_PLUS_GROUP_SIZE;
static const uint kTileCount = AUTO_EXPOSURE_PLUS_TILE_SPAN * AUTO_EXPOSURE_PLUS_GROUP_SIZE;

// Reads one source texel, clamped to the source extent, from either the tile or memory.
float2 loadSourceTexel(int axisIndex, uint localOffAxis, uint offAxisCoord, bool horizontal, uint srcExtentOnAxis, bool useTile, int tileStart)
{
  if (useTile)
  {
    // The tile was filled with the same clamping, so an index past the edge resolves to the
    // edge texel exactly as a direct clamped load would.
    const uint tileIndex = uint(axisIndex - tileStart) * AUTO_EXPOSURE_PLUS_GROUP_SIZE + localOffAxis;
    return g_tile[tileIndex];
  }

  const uint clamped = uint(clamp(axisIndex, 0, int(srcExtentOnAxis) - 1));
  const uint2 pixel = horizontal ? uint2(clamped, offAxisCoord) : uint2(offAxisCoord, clamped);
  return InPyramidLevel[pixel].xy;
}

// Edge-aware separable downsample (a Kuwahara-style variance selector).
//
// This is the single most important piece of the operator and the reason it can push far more
// local contrast than a Laplacian pyramid built on plain Gaussian mips before it halos.
//
// A normal Gaussian downsample averages straight across a silhouette, so the coarse levels
// carry a blurred mixture of the bright and dark side of every edge. When that mixture is later
// used to drive an exposure gain, the gain bleeds a bright band into the dark side and a dark
// band into the bright side - the classic local-tonemapping halo.
//
// Instead this builds two one-sided means per destination texel - one from the taps at or below
// the centre, one from the taps at or above it - along with the variance of the key on each
// side, and blends the two means weighted by inverse standard deviation. On a flat region both
// sides agree and the result is an ordinary Gaussian blur. On an edge, one side is flat and the
// other straddles the discontinuity, so the flat side dominates and the filter refuses to
// average across the boundary.
//
// Cost: 23 taps of 2 texels each per destination texel. Adjacent destination texels share
// almost all of those, so the group stages its source span into groupshared once and every
// thread reads from there - roughly an order of magnitude fewer memory operations at level 0,
// which is where nearly all of the pyramid's cost lives. Groups entirely inside the frame also
// take a variant of the tap loop with no per-tap bounds test, since only an 11 texel rim can
// ever fail it.
[shader("compute")]
[numthreads(AUTO_EXPOSURE_PLUS_GROUP_SIZE, AUTO_EXPOSURE_PLUS_GROUP_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint2 groupId : SV_GroupID, uint2 groupThreadId : SV_GroupThreadID, uint groupIndex : SV_GroupIndex)
{
  const bool horizontal = cb.horizontal != 0;

  const uint srcExtentOnAxis = horizontal ? cb.srcExtent.x : cb.srcExtent.y;
  const uint dstExtentOnAxis = horizontal ? cb.dstExtent.x : cb.dstExtent.y;
  const uint dstExtentOffAxis = horizontal ? cb.dstExtent.y : cb.dstExtent.x;

  const float srcSize = float(srcExtentOnAxis);
  const float ratio = srcSize / float(dstExtentOnAxis);

  // Bounds of this group in destination space. Every quantity derived from these is uniform
  // across the group, so the branches below never diverge.
  const uint groupAxisBase = (horizontal ? groupId.x : groupId.y) * AUTO_EXPOSURE_PLUS_GROUP_SIZE;
  const uint groupOffAxisBase = (horizontal ? groupId.y : groupId.x) * AUTO_EXPOSURE_PLUS_GROUP_SIZE;

  // Position of the group's first and last destination texel in source texel-index space.
  const float firstCenter = (float(groupAxisBase) + 0.5f) * ratio - 0.5f;
  const float lastCenter = (float(groupAxisBase + AUTO_EXPOSURE_PLUS_GROUP_SIZE - 1) + 0.5f) * ratio - 0.5f;

  // Source span the group touches, including the bilinear partner of the outermost tap.
  const int tileStart = int(floor(firstCenter)) - AUTO_EXPOSURE_PLUS_KERNEL_RADIUS;
  const int tileEnd = int(floor(lastCenter)) + AUTO_EXPOSURE_PLUS_KERNEL_RADIUS + 1;
  const bool useTile = uint(tileEnd - tileStart + 1) <= AUTO_EXPOSURE_PLUS_TILE_SPAN;

  // True when no tap of any thread in this group can fall outside the source.
  const bool interior = (firstCenter - float(AUTO_EXPOSURE_PLUS_KERNEL_RADIUS) >= -0.5f)
                     && (lastCenter + float(AUTO_EXPOSURE_PLUS_KERNEL_RADIUS) <= srcSize - 0.5f);

  const uint localOffAxis = horizontal ? groupThreadId.y : groupThreadId.x;
  const uint offAxisCoord = horizontal ? threadId.y : threadId.x;

  if (useTile)
  {
    // Cooperative fill. The off-axis extents of source and destination are always equal, so a
    // destination row/column that exists implies the matching source one exists too.
    for (uint i = groupIndex; i < kTileCount; i += kThreadCount)
    {
      const uint tileAxis = i / AUTO_EXPOSURE_PLUS_GROUP_SIZE;
      const uint tileOffAxis = i % AUTO_EXPOSURE_PLUS_GROUP_SIZE;
      const uint sourceOffAxis = groupOffAxisBase + tileOffAxis;

      float2 value = float2(0.0f);

      if (sourceOffAxis < dstExtentOffAxis)
      {
        const uint clamped = uint(clamp(tileStart + int(tileAxis), 0, int(srcExtentOnAxis) - 1));
        const uint2 pixel = horizontal ? uint2(clamped, sourceOffAxis) : uint2(sourceOffAxis, clamped);
        value = InPyramidLevel[pixel].xy;
      }

      g_tile[i] = value;
    }

    GroupMemoryBarrierWithGroupSync();
  }

  // The out-of-range early out has to come after the barrier, otherwise threads that skipped it
  // would leave the rest of the group waiting forever.
  if (any(threadId >= cb.dstExtent))
  {
    return;
  }

  const float centerCoord = (float(horizontal ? threadId.x : threadId.y) + 0.5f) * ratio - 0.5f;

  AutoExposurePlusKuwahara accum;
  accum.reset();

  if (interior)
  {
    // Fast path: no tap can be out of bounds, so the loop is a straight gather.
    [unroll]
    for (int j = -AUTO_EXPOSURE_PLUS_KERNEL_RADIUS; j <= AUTO_EXPOSURE_PLUS_KERNEL_RADIUS; ++j)
    {
      const float tapCoord = centerCoord + float(j);
      const int lowIndex = int(floor(tapCoord));
      const float fraction = tapCoord - float(lowIndex);

      const float2 low = loadSourceTexel(lowIndex, localOffAxis, offAxisCoord, horizontal, srcExtentOnAxis, useTile, tileStart);
      const float2 high = loadSourceTexel(lowIndex + 1, localOffAxis, offAxisCoord, horizontal, srcExtentOnAxis, useTile, tileStart);

      accum.accumulate(lerp(low, high, fraction), autoExposurePlusTapWeight(j), j);
    }
  }
  else
  {
    [unroll]
    for (int j = -AUTO_EXPOSURE_PLUS_KERNEL_RADIUS; j <= AUTO_EXPOSURE_PLUS_KERNEL_RADIUS; ++j)
    {
      const float tapCoord = centerCoord + float(j);

      // Drop taps whose footprint falls outside the source rather than letting clamp-to-edge
      // replicate the border, which would bias the variance estimate at the frame edges.
      if (tapCoord < -0.5f || tapCoord > srcSize - 0.5f)
      {
        continue;
      }

      const int lowIndex = int(floor(tapCoord));
      const float fraction = tapCoord - float(lowIndex);

      const float2 low = loadSourceTexel(lowIndex, localOffAxis, offAxisCoord, horizontal, srcExtentOnAxis, useTile, tileStart);
      const float2 high = loadSourceTexel(lowIndex + 1, localOffAxis, offAxisCoord, horizontal, srcExtentOnAxis, useTile, tileStart);

      accum.accumulate(lerp(low, high, fraction), autoExposurePlusTapWeight(j), j);
    }
  }

  OutPyramidLevel[threadId] = vec4(accum.resolve(), 0.0f, 0.0f);
}
