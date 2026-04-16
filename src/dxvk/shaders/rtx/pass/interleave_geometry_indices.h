/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

struct InterleaveGeometryArgs {
  uint32_t positionOffset;
  uint32_t positionStride;
  uint32_t positionFormat;

  uint32_t hasNormals;
  uint32_t normalOffset;
  uint32_t normalStride;
  uint32_t normalFormat;

  uint32_t hasTexcoord;
  uint32_t texcoordOffset;
  uint32_t texcoordStride;
  uint32_t texcoordFormat;

  uint32_t hasColor0;
  uint32_t color0Offset;
  uint32_t color0Stride;
  uint32_t color0Format;

  uint32_t minVertexIndex;
  uint32_t outputStride;
  uint32_t vertexCount;
  uint32_t forceNormals; // When set, reserve normal space in output even if hasNormals is false (writes zeros)

  // NV-DXVK: Source Engine 2 bone matrix transform
  uint32_t hasBoneTransform;   // When set, fetch bone matrix from bone slot and apply to position
  uint32_t boneIndex;          // Fallback index if bonePerVertex == 0

  // NV-DXVK (TF2 BSP): per-vertex instance index lookup for g_modelInst-style
  // batched drawing. Each vertex's COLOR1 picks its own transform.
  uint32_t bonePerVertex;      // 0 = use args.boneIndex (legacy single-bone), 1 = read srcBoneIndex[vertexIdx]
  uint32_t boneMatrixStride;   // bytes/row in bone buffer (48 for g_boneMatrix, 208 for g_modelInst)
  uint32_t boneIndexStride;    // bytes/vertex in bone-index buffer (8 for R16G16B16A16_UINT, 16 for R32G32B32A32_UINT)
  uint32_t boneIndexMask;      // 0xFFFF for 16-bit index (legacy), 0xFFFFFFFF for full 32-bit

  // NV-DXVK (TF2 skinned chars): 4-bone weighted skinning. When hasBoneWeights=1
  // the shader does Σ_i weight[i] * boneMatrix[boneIdx[i]] × position. Weight
  // stream is per-vertex and interpreted as 2×UNORM16 (weights 0..1) with the
  // other two weights synthesized so they sum to 1 (Source convention).
  uint32_t hasBoneWeights;     // 0 = single-bone skinning; 1 = 4-bone weighted
  uint32_t boneWeightOffset;   // in uint32s
  uint32_t boneWeightStride;   // in uint32s (vertex stride / 4)
  uint32_t boneIndexComponentCount; // 4 for RGBA8_UINT, else 1
  uint32_t boneIndexOffsetUints;    // byte offset of BLENDINDICES within vertex, in uint32s
};

// NV-DXVK (DX11 port): shift past the D3D11 graphics slot range (0..1151) so
// m_rc[] slots don't collide with PS CB slots. See gpu_skinning_binding_indices.h
// for the full rationale.
#define INTERLEAVE_GEOMETRY_BINDING_OUTPUT           1170
#define INTERLEAVE_GEOMETRY_BINDING_POSITION_INPUT   1171
#define INTERLEAVE_GEOMETRY_BINDING_NORMAL_INPUT     1172
#define INTERLEAVE_GEOMETRY_BINDING_TEXCOORD_INPUT   1173
#define INTERLEAVE_GEOMETRY_BINDING_COLOR0_INPUT     1174
#define INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX      1175
#define INTERLEAVE_GEOMETRY_BINDING_BONE_INDEX       1176
#define INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT      1177
