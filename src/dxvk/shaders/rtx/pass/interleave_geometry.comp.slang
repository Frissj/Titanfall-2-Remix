// NV-DXVK: touched to force slang recompile after interleave_geometry.h
// weight-clamp + zero-bone-gap-skip fix for TF2 skinned character spikes.
// v2: tightened validity check (all 3 rows unit-len + translation bounds)
//     and bone-0 fallback for vertices where all 3 weighted bones invalid.
// v3: SENTINEL diagnostic — verts with any invalid bone output at (+1e6,0,0)
//     so we can visually confirm the validity check is firing correctly.
// v4: UNIVERSAL fix — cluster-distance check. Reject bones whose T is >400
//     units from the centroid of the vertex's 3 bones. Catches foreign-
//     character bones sharing the t30 buffer without per-character tuning.
// v5: RIGOROUS rotation matrix check — reject bones whose 3 rows aren't
//     unit-length AND mutually perpendicular. Catches uninitialized gap-
//     slot garbage without heuristics, since random bit patterns almost
//     never produce a valid orthonormal rotation.
// v6: SCALE-TOLERANT rotation check — TF2 uses scaled bones for stretchy
//     animations. Check cross(r0,r1) is parallel to r2 with consistent
//     scale across rows (allows any S in [0.1, 10]). Plus bone-0 fallback
//     so invalid-pose verts collapse to character root, not bind pose.
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

// NV-DXVK: TEXCOORD1 / lightmap UV input. Bound to the same VB as TEXCOORD0
// when the wall VS layout co-locates them (typical Source/TF2), or to a
// distinct VB. Stride/offset/format are taken from cb.texcoord1*.
layout(binding = INTERLEAVE_GEOMETRY_BINDING_TEXCOORD1_INPUT)
StructuredBuffer<float> srcTexcoord1;

// NV-DXVK: TF2 VGUI TEXCOORD3 int4 input. Carries packed glyph/style/image
// indices. Read only when (cb.flags & INTERLEAVE_GEOMETRY_FLAG_VGUI_
// LAYOUT_ENABLE) is set. Bound as uint32 since the format is
// R32G32B32A32_SINT (4×int32 per vertex).
layout(binding = INTERLEAVE_GEOMETRY_BINDING_VGUI_TEXCOORD3)
StructuredBuffer<uint32_t> srcVguiTexcoord3;

// NV-DXVK: TF2 VGUI TEXCOORD2 — glyph dims, R32G32_SFLOAT.
layout(binding = INTERLEAVE_GEOMETRY_BINDING_VGUI_GLYPH_DIMS)
StructuredBuffer<float> srcVguiGlyphDims;

// NV-DXVK: per-instance bone-base stream (COLOR1/I R16G16B16A16_UINT). Read GPU-
// side at skin time so the dynamic-buffer rename race can't corrupt the base.
layout(binding = INTERLEAVE_GEOMETRY_BINDING_BONE_BASE)
StructuredBuffer<uint32_t> srcBoneBase;

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

  interleaver::interleave(idx, dst, srcPosition, srcNormal, srcTexcoord, srcColor0, srcBoneMatrix, srcBoneIndex, srcBoneWeight, srcTexcoord1, srcVguiTexcoord3, srcVguiGlyphDims, srcBoneBase, cb);
}
