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
#include "rigid_bake_from_buffer_binding_indices.h"

layout(binding = BINDING_RBFB_CONSTANTS)
ConstantBuffer<RigidBakeFromBufferArgs> cb;

layout(binding = BINDING_RBFB_POSITION_IO)
RWStructuredBuffer<float> Positions;

layout(binding = BINDING_RBFB_NORMAL_IO)
RWStructuredBuffer<float> Normals;

// Game-owned bone-matrix buffer (e.g. engine g_boneMatrix t31), read as floats.
layout(binding = BINDING_RBFB_BONES_INPUT)
StructuredBuffer<float> Bones;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint idx : SV_DispatchThreadID)
{
  if (idx >= cb.numVertices) return;

  // Load the chosen rigid bone matrix (float3x4, row-major) from the game buffer.
  const uint b = cb.boneFloatOffset;
  const float3 r0 = float3(Bones[b + 0], Bones[b + 1], Bones[b + 2]);
  const float  tx = Bones[b + 3];
  const float3 r1 = float3(Bones[b + 4], Bones[b + 5], Bones[b + 6]);
  const float  ty = Bones[b + 7];
  const float3 r2 = float3(Bones[b + 8], Bones[b + 9], Bones[b + 10]);
  const float  tz = Bones[b + 11];

  // Position: object -> world.
  const uint basePos = (cb.positionOffset + idx * cb.positionStride) / 4;
  const float3 p = float3(Positions[basePos + 0], Positions[basePos + 1], Positions[basePos + 2]);
  const float3 wp = float3(dot(r0, p) + tx, dot(r1, p) + ty, dot(r2, p) + tz);
  Positions[basePos + 0] = wp.x;
  Positions[basePos + 1] = wp.y;
  Positions[basePos + 2] = wp.z;

  // Normal: rotate only (bones are rigid/orthonormal, so rotation == inverse-transpose).
  if (cb.normalStride != 0)
  {
    const uint baseN = (cb.normalOffset + idx * cb.normalStride) / 4;
    const float3 n = float3(Normals[baseN + 0], Normals[baseN + 1], Normals[baseN + 2]);
    float3 wn = float3(dot(r0, n), dot(r1, n), dot(r2, n));
    const float len = length(wn);
    if (len > 0.0f) wn /= len;
    Normals[baseN + 0] = wn.x;
    Normals[baseN + 1] = wn.y;
    Normals[baseN + 2] = wn.z;
  }
}
