/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "interleave_geometry_indices.h"
#include "interleave_geometry.h"

layout(binding = INTERLEAVE_GEOMETRY_BINDING_OUTPUT)
RWStructuredBuffer<float> dst;

layout(binding = INTERLEAVE_GEOMETRY_BINDING_POSITION_INPUT)
StructuredBuffer<float> srcPosition;
layout(binding = INTERLEAVE_GEOMETRY_BINDING_NORMAL_INPUT)
StructuredBuffer<float> srcNormal;
layout(binding = INTERLEAVE_GEOMETRY_BINDING_TEXCOORD_INPUT)
StructuredBuffer<float> srcTexcoord;
layout(binding = INTERLEAVE_GEOMETRY_BINDING_COLOR0_INPUT)
StructuredBuffer<uint32_t> srcColor0;

// NV-DXVK: Source Engine 2 bone matrix and bone index buffers
layout(binding = INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX)
StructuredBuffer<float> srcBoneMatrix;
layout(binding = INTERLEAVE_GEOMETRY_BINDING_BONE_INDEX)
StructuredBuffer<uint32_t> srcBoneIndex;
// NV-DXVK: per-vertex bone weight stream (TF2 skinned characters). Interpreted
// as packed UNORM16 pairs per uint32 (2 weights per uint) — the VS decodes
// the first two weights as f16 and synthesizes the third/fourth to make the
// sum 1.0. We mirror that exactly in applyWeightedBones.
layout(binding = INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT)
StructuredBuffer<uint32_t> srcBoneWeight;

layout(push_constant)
ConstantBuffer<InterleaveGeometryArgs> cb;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint idx : SV_DispatchThreadID)
{
  if (idx >= cb.vertexCount)
  {
    return;
  }

  interleaver::interleave(idx, dst, srcPosition, srcNormal, srcTexcoord, srcColor0, srcBoneMatrix, srcBoneIndex, srcBoneWeight, cb);
}
