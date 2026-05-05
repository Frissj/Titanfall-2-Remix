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

  // NV-DXVK: hasNormals (was uint32) is now bit 2 of `flags` to claw back
  // push-constant bytes for the VGUI extras. See INTERLEAVE_GEOMETRY_FLAG_*.
  uint32_t normalOffset;
  uint32_t normalStride;
  uint32_t normalFormat;

  // NV-DXVK: hasTexcoord packed into `flags` bit 3.
  uint32_t texcoordOffset;
  uint32_t texcoordStride;
  uint32_t texcoordFormat;

  // NV-DXVK: TF2 / Source wall VSes pack a second TEXCOORD attribute (the
  // lightmap UV) into a separate VB stream. VS_e7abcf4e disassembly:
  //   ushr/iadd/itof/mul   v3 → o0.xy   (TEXCOORD0 albedo, tile_uv decode)
  //   utof/mul             v4 → o0.zw   (TEXCOORD1 lightmap, * 1/65535)
  // The interleaver decodes both and writes them adjacent in the output
  // stride. The lightmap stream's format and stride often differ from
  // TEXCOORD0's (e.g. TC0=R32G32_UINT, TC1=R16G16_UINT) so we can't reuse
  // texcoordStride/Format like an earlier revision did.
  //
  // PUSH-CONSTANT BUDGET: this struct is at the 128-byte Vulkan minimum.
  // To fit independent stride+format for TC1 without expanding, we pack
  // them into one uint (low 16 = stride in uint32 units, high 16 = VkFormat
  // enum value) and use `texcoord1StrideFormat == 0` as the
  // "no-lightmap-this-draw" sentinel — drops the standalone hasTexcoord1
  // bool. The slang and C++ sides both unpack via shifts/masks below.
  uint32_t texcoord1Offset;
  uint32_t texcoord1StrideFormat; // (format << 16) | stride; 0 = absent

  // NV-DXVK: hasColor0 packed into `flags` bit 4.
  uint32_t color0Offset;
  uint32_t color0Stride;
  uint32_t color0Format;

  uint32_t minVertexIndex;
  uint32_t outputStride;
  uint32_t vertexCount;
  // NV-DXVK: packed flags. Was `forceNormals` (uint32 storing one bit).
  // Now a bitfield to free push-constant space for VGUI plumbing without
  // exceeding the 128-byte Vulkan minimum push-constant size.
  //   bit 0 — forceNormals: reserve normal space in output even if
  //           hasNormals is false (writes zeros). Existing semantic.
  //   bit 1 — vguiLayoutEnable: TF2 worldspace VGUI vertex layout
  //           detected. Interleaver writes additional per-vertex data
  //           (TEXCOORD0.zw + TEXCOORD2-as-int4) to feed the slang VGUI
  //           evaluator (see opaque_surface_material_interaction.slangh).
  //           The PS-side detection lives in d3d11_rtx.cpp::FillMaterial
  //           Data and sets LegacyMaterialData::sourceIsUnlitUI; the BLAS
  //           construction path (rtx_geometry_utils dispatchInterleave
  //           Geometry) reads that flag and sets bit 1 here.
  uint32_t flags;

  // NV-DXVK: Source Engine 2 bone matrix transform
  // hasBoneTransform packed into `flags` bit 5.
  uint32_t boneIndex;          // Fallback index if bonePerVertex == 0

  // NV-DXVK (TF2 BSP): per-vertex instance index lookup for g_modelInst-style
  // batched drawing. Each vertex's COLOR1 picks its own transform.
  // bonePerVertex packed into `flags` bit 7.
  uint32_t boneMatrixStride;   // bytes/row in bone buffer (48 for g_boneMatrix, 208 for g_modelInst)
  uint32_t boneIndexStride;    // bytes/vertex in bone-index buffer (8 for R16G16B16A16_UINT, 16 for R32G32B32A32_UINT)
  uint32_t boneIndexMask;      // 0xFFFF for 16-bit index (legacy), 0xFFFFFFFF for full 32-bit

  // NV-DXVK (TF2 skinned chars): 4-bone weighted skinning. When the
  // HAS_BONE_WEIGHTS flag is set the shader does
  // Σ_i weight[i] * boneMatrix[boneIdx[i]] × position. Weight stream is
  // per-vertex and interpreted as 2×UNORM16 (weights 0..1) with the
  // other two weights synthesized so they sum to 1 (Source convention).
  // hasBoneWeights packed into `flags` bit 6.
  uint32_t boneWeightOffset;   // in uint32s
  uint32_t boneWeightStride;   // in uint32s (vertex stride / 4)
  uint32_t boneIndexComponentCount; // 4 for RGBA8_UINT, else 1
  uint32_t boneIndexOffsetUints;    // byte offset of BLENDINDICES within vertex, in uint32s

  // NV-DXVK: TF2 worldspace VGUI vertex stream — TEXCOORD3 (semantic name
  // per D3D11 input layout, an int4). Carries packed glyph/style/image
  // indices that the original PS uses to look up glyph bounds in
  // g_fontBounds and styles in g_styles. Read by the interleaver only when
  // (flags & 2u) is set; otherwise the binding is bound to a placeholder
  // and these args are ignored. Format is fixed at R32G32B32A32_SINT
  // (the only format the VGUI VS layout uses for this attribute).
  uint32_t vguiTexcoord3Offset;   // bytes into the VB
  uint32_t vguiTexcoord3Stride;   // bytes per vertex (typically 16)
  // NV-DXVK: TF2 VGUI TEXCOORD2 — glyph dimensions / scale, R32G32_SFLOAT.
  // Read by the interleaver only when the VGUI flag is set. Bound at
  // INTERLEAVE_GEOMETRY_BINDING_VGUI_GLYPH_DIMS; placeholder otherwise.
  uint32_t vguiGlyphDimsOffset;   // bytes into the VB
  uint32_t vguiGlyphDimsStride;   // bytes per vertex (typically 8)
};

// NV-DXVK: bit positions for InterleaveGeometryArgs::flags. Defined as
// macros so both slang and C++ pick them up identically. See the field's
// documentation in InterleaveGeometryArgs for semantics.
#define INTERLEAVE_GEOMETRY_FLAG_FORCE_NORMALS         (1u << 0)
#define INTERLEAVE_GEOMETRY_FLAG_VGUI_LAYOUT_ENABLE    (1u << 1)
// NV-DXVK: bools repacked into flags to fit the 128-byte push-constant limit
// (DXVK's MaxPushConstantSize is hardcoded). Each bit replaces a previously
// standalone uint32 in InterleaveGeometryArgs.
#define INTERLEAVE_GEOMETRY_FLAG_HAS_NORMALS           (1u << 2)
#define INTERLEAVE_GEOMETRY_FLAG_HAS_TEXCOORD          (1u << 3)
#define INTERLEAVE_GEOMETRY_FLAG_HAS_COLOR0            (1u << 4)
#define INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_TRANSFORM    (1u << 5)
#define INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_WEIGHTS      (1u << 6)
#define INTERLEAVE_GEOMETRY_FLAG_BONE_PER_VERTEX       (1u << 7)
// NV-DXVK: TF2 VGUI TEXCOORD3 element size discriminator. Real TF2 VGUI
// shaders use R16G16B16A16_SINT (8 bytes per vertex, 4 × int16), not the
// R32G32B32A32_SINT (16 bytes, 4 × int32) the handoff claimed. The
// interleaver branches on this flag to read 2 × uint32 and unpack 4 × int16
// (with sign extension), vs reading 4 × uint32 directly.
#define INTERLEAVE_GEOMETRY_FLAG_VGUI_TC3_IS_INT16     (1u << 8)

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
// NV-DXVK: lightmap UV (TEXCOORD1) input. Bound only when
// args.texcoord1StrideFormat is non-zero; placeholder buffer otherwise.
#define INTERLEAVE_GEOMETRY_BINDING_TEXCOORD1_INPUT  1178
// NV-DXVK: TF2 worldspace VGUI int4 stream (semantic TEXCOORD3 per the
// D3D11 input layout). Bound only when flags & VGUI_LAYOUT_ENABLE is set;
// placeholder buffer otherwise. See InterleaveGeometryArgs::vguiTexcoord3*.
#define INTERLEAVE_GEOMETRY_BINDING_VGUI_TEXCOORD3   1179
// NV-DXVK: TF2 VGUI glyph dimensions input (TEXCOORD2, R32G32_SFLOAT).
// Bound only when flags & VGUI_LAYOUT_ENABLE is set; placeholder otherwise.
#define INTERLEAVE_GEOMETRY_BINDING_VGUI_GLYPH_DIMS  1180
