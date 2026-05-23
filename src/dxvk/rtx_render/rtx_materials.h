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
#pragma once

#include <memory>
#include <atomic>
#include "rtx_texture.h"
#include "rtx_option.h"
#include "rtx_cb_types.h"
#include "../../util/util_color.h"
#include "../../util/util_macro.h"
#include "rtx/utility/shared_constants.h"
#include "rtx/concept/surface/surface_shared.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/instance_definitions.h"
#include "rtx_material_data.h"
#include "../../lssusd/mdl_helpers.h"
#include "rtx/pass/particles/particle_system_common.h"
#include "dxvk_constant_state.h"

namespace dxvk {
// Surfaces

// Todo: Compute size directly from sizeof of GPU structure (by including it), for now computed by sum of members manually
constexpr std::size_t kSurfaceGPUSize = 16 * 4 * 4;

// Note: Use caution when changing this enum, must match the values defined on the MDL side of things.

static bool isBlendTypeEmissive(const BlendType type) {
  switch (type) {
  default:
    return false;
  case BlendType::kAlphaEmissive:
  case BlendType::kReverseAlphaEmissive:
  case BlendType::kColorEmissive:
  case BlendType::kReverseColorEmissive:
  case BlendType::kEmissive:
    return true;
  }
}

static BlendType tryConvertToEmissive(const BlendType type) {
  switch (type) {
  case BlendType::kAlpha:
    return BlendType::kAlphaEmissive;
  case BlendType::kColor:
    return BlendType::kColorEmissive;
  default:
    return type;
  }
}

static_assert((int)AlphaTestType::kNever == (int)VkCompareOp::VK_COMPARE_OP_NEVER);
static_assert((int)AlphaTestType::kLess == (int)VkCompareOp::VK_COMPARE_OP_LESS);
static_assert((int)AlphaTestType::kEqual == (int)VkCompareOp::VK_COMPARE_OP_EQUAL);
static_assert((int)AlphaTestType::kLessOrEqual == (int)VkCompareOp::VK_COMPARE_OP_LESS_OR_EQUAL);
static_assert((int)AlphaTestType::kGreater == (int)VkCompareOp::VK_COMPARE_OP_GREATER);
static_assert((int)AlphaTestType::kNotEqual == (int)VkCompareOp::VK_COMPARE_OP_NOT_EQUAL);
static_assert((int)AlphaTestType::kGreaterOrEqual == (int)VkCompareOp::VK_COMPARE_OP_GREATER_OR_EQUAL);
static_assert((int)AlphaTestType::kAlways == (int)VkCompareOp::VK_COMPARE_OP_ALWAYS);

// Note: "Temporary" hacks to get RtxOptions data from this header file as we cannot include rtx_options directly
// due to cyclic includes. This should be removed once the rtx_materials implementation is moved to a source file.
bool getEnableDiffuseLayerOverrideHack();
float getEmissiveIntensity();
float getDisplacementFactor();
float getDisplacementInFactor();
float getDisplacementOutFactor();

struct RtEyeParams {
  // origin of eyeball in world space
  // used to calculate eye normals
  Vector3 eyeballOrigin = Vector3{ 0, 0, 0 };
  // right/up vectors that define an eye orientation
  // NOTE: vectors can be unnormalized, and that scale denotes an iris size 
  Vector3 eyeRightU = Vector3{ 1, 0, 0 };
  Vector3 eyeUpV = Vector3{ 0, 1, 0 };
};

struct RtSurface {
  RtSurface() {
  }

  void writeGPUData(unsigned char* data, std::size_t& offset, size_t surfaceIndex = SIZE_MAX) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;

    // Note: Position buffer and surface material index are required for proper
    // behavior of the Surface on the GPU.
    assert(positionBufferIndex != kSurfaceInvalidBufferIndex);

    writeGPUHelperExplicit<2>(data, offset, positionBufferIndex);
    writeGPUHelperExplicit<2>(data, offset, previousPositionBufferIndex);
    writeGPUHelperExplicit<2>(data, offset, normalBufferIndex);
    writeGPUHelperExplicit<2>(data, offset, texcoordBufferIndex);
    writeGPUHelperExplicit<2>(data, offset, indexBufferIndex);
    writeGPUHelperExplicit<2>(data, offset, color0BufferIndex);

    uint16_t flags0 = 0;
    flags0 |= normalFormat == VK_FORMAT_R32_UINT ? 1 : 0;
    flags0 |= isVertexColorBakedLighting ? (1 << 1) : 0;
    // NV-DXVK: lightmap UV presence (mirror of slang Surface::hasLightmap
    // at data0b.z bit 2). Bit 2 of flags0.
    flags0 |= hasLightmap ?               (1 << 2) : 0;
    // NV-DXVK: TF2 worldspace VGUI per-vertex extras present. Mirrors slang
    // Surface::isVgui at data0b.z bit 3. When set, the VGUI evaluator runs
    // at hit time and reads (vguiOffset .. vguiOffset+8) float-units from
    // the interleaved per-vertex output.
    flags0 |= isVgui ?                    (1 << 3) : 0;
    // NV-DXVK: TF2 3D-skybox cloud billboard — mirrors slang
    // Surface::isTf2SkyboxFog at data0b.z bit 4.
    flags0 |= isTf2SkyboxFog ?            (1 << 4) : 0;
    // NOTE: Spare flags bits here

    writeGPUHelper(data, offset, flags0);

    const uint16_t packedHash =
      (uint16_t) (associatedGeometryHash >> 48) ^
      (uint16_t) (associatedGeometryHash >> 32) ^
      (uint16_t) (associatedGeometryHash >> 16) ^
      (uint16_t) associatedGeometryHash;

    writeGPUHelper(data, offset, packedHash);

    writeGPUHelper(data, offset, positionOffset);
    writeGPUHelper(data, offset, normalOffset);
    writeGPUHelper(data, offset, texcoordOffset);
    writeGPUHelper(data, offset, color0Offset);
    writeGPUHelper(data, offset, objectPickingValue);

    writeGPUHelperExplicit<1>(data, offset, positionStride);
    writeGPUHelperExplicit<1>(data, offset, normalStride);
    writeGPUHelperExplicit<1>(data, offset, texcoordStride);
    writeGPUHelperExplicit<1>(data, offset, color0Stride);

    writeGPUHelperExplicit<3>(data, offset, firstIndex);
    writeGPUHelperExplicit<1>(data, offset, indexStride);

    // Note: Ensure alpha state values fit in the intended amount of bits allocated in the flags bitfield.
    assert(static_cast<uint32_t>(alphaState.alphaTestType) < (1 << 3));
    assert(static_cast<uint32_t>(alphaState.alphaTestReferenceValue) < (1 << 8));
    assert(static_cast<uint32_t>(alphaState.blendType) < (1 << 4));

    uint32_t flags1 = 0;

    flags1 |= isEmissive ? (1 << 0) : 0;
    flags1 |= alphaState.isFullyOpaque ? (1 << 1) : 0;
    flags1 |= isStatic ? (1 << 2) : 0;
    flags1 |= static_cast<uint32_t>(alphaState.alphaTestType) << 3;
    // Note: No mask needed as masking of this value to be 8 bit is done elsewhere.
    flags1 |= static_cast<uint32_t>(alphaState.alphaTestReferenceValue) << 6;
    flags1 |= static_cast<uint32_t>(alphaState.blendType) << 14;
    flags1 |= alphaState.invertedBlend ?      (1 << 18) : 0;
    flags1 |= alphaState.isBlendingDisabled ? (1 << 19) : 0;
    flags1 |= alphaState.emissiveBlend ?      (1 << 20) : 0;
    flags1 |= alphaState.isParticle ?         (1 << 21) : 0;
    flags1 |= alphaState.isDecal ?            (1 << 22) : 0;
    flags1 |= hasMaterialChanged ?            (1 << 23) : 0;
    flags1 |= isAnimatedWater ?               (1 << 24) : 0;
    flags1 |= isClipPlaneEnabled ?            (1 << 25) : 0;
    flags1 |= isMatte ?                       (1 << 26) : 0;
    flags1 |= isTextureFactorBlend ?          (1 << 27) : 0;
    flags1 |= isMotionBlurMaskOut ?           (1 << 28) : 0;
    flags1 |= skipSurfaceInteractionSpritesheetAdjustment ? (1 << 29) : 0;
    flags1 |= ignoreTransparencyLayer ?       (1 << 30) : 0;
    // Note: This flag is purely for debug view purpose. If we need to add more functional flags and running out of bits, we should move this flag to other place.
    flags1 |= isInsideFrustum ?               (1 << 31) : 0;

    writeGPUHelper(data, offset, flags1);

    // Note: Matricies are stored on the cpu side in column-major order, the same as the GPU.

    Matrix4 instanceToWorld = objectToWorld;
    Matrix4 prevInstanceToWorld = prevObjectToWorld;
    Matrix3 normalInstanceToWorld = normalObjectToWorld;

    if (instancesToObject && surfaceIndexOfFirstInstance != SIZE_MAX && surfaceIndex != SIZE_MAX) {
      const size_t instanceIndex = surfaceIndex - surfaceIndexOfFirstInstance;
      if (instanceIndex >= instancesToObject->size()) {
        // Note: This should never happen.
        assert(false);
        Logger::err("Error: invalid instance index in RtSurface::WriteGPUData.");
      } else {
        instanceToWorld = objectToWorld * (*instancesToObject)[instanceIndex];
        prevInstanceToWorld = prevObjectToWorld * (*instancesToObject)[instanceIndex];
        normalInstanceToWorld = transpose(inverse(Matrix3(instanceToWorld)));
      }
    }

    // Note: Last row of object to world matrix not needed as it does not encode any useful information
    writeGPUHelper(data, offset, prevInstanceToWorld.data[0].x);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[0].y);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[0].z);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[1].x);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[1].y);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[1].z);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[2].x);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[2].y);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[2].z);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[3].x);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[3].y);
    writeGPUHelper(data, offset, prevInstanceToWorld.data[3].z);

    writeGPUHelper(data, offset, normalInstanceToWorld.data[0]);
    writeGPUHelper(data, offset, normalInstanceToWorld.data[1]);
    writeGPUHelper(data, offset, normalInstanceToWorld.data[2].x);
    writeGPUHelper(data, offset, normalInstanceToWorld.data[2].y);

    writeGPUHelper(data, offset, instanceToWorld.data[0].x);
    writeGPUHelper(data, offset, instanceToWorld.data[0].y);
    writeGPUHelper(data, offset, instanceToWorld.data[0].z);
    writeGPUHelper(data, offset, instanceToWorld.data[1].x);
    writeGPUHelper(data, offset, instanceToWorld.data[1].y);
    writeGPUHelper(data, offset, instanceToWorld.data[1].z);
    writeGPUHelper(data, offset, instanceToWorld.data[2].x);
    writeGPUHelper(data, offset, instanceToWorld.data[2].y);
    writeGPUHelper(data, offset, instanceToWorld.data[2].z);
    writeGPUHelper(data, offset, instanceToWorld.data[3].x);
    writeGPUHelper(data, offset, instanceToWorld.data[3].y);
    writeGPUHelper(data, offset, instanceToWorld.data[3].z);

    if (eyeParams) {
      // eye vectors are aliased with texture transform
      writeGPUHelper(data, offset, eyeParams->eyeRightU[0]);
      writeGPUHelper(data, offset, eyeParams->eyeRightU[1]);
      writeGPUHelper(data, offset, eyeParams->eyeRightU[2]);
      writeGPUHelper(data, offset, uint32_t{});
      writeGPUHelper(data, offset, eyeParams->eyeUpV[0]);
      writeGPUHelper(data, offset, eyeParams->eyeUpV[1]);
      writeGPUHelper(data, offset, eyeParams->eyeUpV[2]);
      writeGPUHelper(data, offset, uint32_t{});
    } else {
      // Note: Only 2 rows of texture transform written for now due to limit of 2 element restriction.
      writeGPUHelper(data, offset, textureTransform.data[0].x);
      writeGPUHelper(data, offset, textureTransform.data[1].x);
      writeGPUHelper(data, offset, textureTransform.data[2].x);
      writeGPUHelper(data, offset, textureTransform.data[3].x);
      writeGPUHelper(data, offset, textureTransform.data[0].y);
      writeGPUHelper(data, offset, textureTransform.data[1].y);
      writeGPUHelper(data, offset, textureTransform.data[2].y);
      writeGPUHelper(data, offset, textureTransform.data[3].y);
      // NV-DXVK: final GPU-write checkpoint. If this never fires for a
      // non-identity matrix, the surface data write path is bypassing our
      // textureTransform somehow.
      {
        static uint32_t sGpuWr = 0;
        if (sGpuWr < 20) {
          const bool nonIdent =
            textureTransform.data[0].x != 1.0f
            || textureTransform.data[1].x != 0.0f
            || textureTransform.data[0].y != 0.0f
            || textureTransform.data[1].y != 1.0f
            || textureTransform.data[3].x != 0.0f
            || textureTransform.data[3].y != 0.0f;
          if (nonIdent) {
            ++sGpuWr;
            Logger::info(str::format(
              "[SurfaceGPU.UVx] wrote non-identity xform to GPU "
              "data11=(", textureTransform.data[0].x, ",",
                          textureTransform.data[1].x, ",",
                          textureTransform.data[2].x, ",",
                          textureTransform.data[3].x, ") "
              "data12=(", textureTransform.data[0].y, ",",
                          textureTransform.data[1].y, ",",
                          textureTransform.data[2].y, ",",
                          textureTransform.data[3].y, ")"));
          }
        }
      }
    }

    std::uint32_t textureSpritesheetData = 0;

    // Clamp rows and cols to at least 1, to avoid divide by 0 errors.
    textureSpritesheetData |= (static_cast<uint32_t>(std::max<uint8_t>(1, spriteSheetRows)) << 0);
    textureSpritesheetData |= (static_cast<uint32_t>(std::max<uint8_t>(1, spriteSheetCols)) << 8);
    textureSpritesheetData |= (static_cast<uint32_t>(spriteSheetFPS) << 16);
    // pack decalSortOrder into data13.x's last 8 bits.
    textureSpritesheetData |= (static_cast<uint32_t>(decalSortOrder) << 24);

    writeGPUHelper(data, offset, textureSpritesheetData);

    writeGPUHelper(data, offset, tFactor);

    std::uint32_t textureFlags = 0;

    assert((static_cast<uint32_t>(textureColorOperation) & 0x7) == static_cast<uint32_t>(textureColorOperation));
    assert((static_cast<uint32_t>(textureAlphaOperation) & 0x7) == static_cast<uint32_t>(textureAlphaOperation));
    assert(textureAlphaOperation != DxvkRtTextureOperation::Force_Modulate2x);

    textureFlags |= ((static_cast<uint32_t>(textureColorArg1Source) & 0x3));
    textureFlags |= ((static_cast<uint32_t>(textureColorArg2Source) & 0x3) << 2);
    textureFlags |= ((static_cast<uint32_t>(textureColorOperation)  & 0x7) << 4);
    textureFlags |= ((static_cast<uint32_t>(textureAlphaArg1Source) & 0x3) << 7);
    textureFlags |= ((static_cast<uint32_t>(textureAlphaArg2Source) & 0x3) << 9);
    textureFlags |= ((static_cast<uint32_t>(textureAlphaOperation)  & 0x7) << 11);

    textureFlags |= eyeParams ? (1 << 14) : 0;
    // textureFlags bits 15-16 unused

    static_assert(static_cast<uint32_t>(TexGenMode::Count) <= 4);
    textureFlags |= ((static_cast<uint32_t>(texgenMode) & 0x3) << 17);
    // NV-DXVK: bits 19-20 carry texcoordEncoding (2 bits, mirrors the
    // surface.h getter at offset (data13.z >> 19) & 0x3).
    textureFlags |= ((static_cast<uint32_t>(texcoordEncoding) & 0x3) << 19);
    // textureFlags bits 21-30 unused

    // NV-DXVK: catch the (stride=20 + encoding != Float) mismatch probe 8
    // sees on the wall surface. If this fires, the C++ surface object
    // genuinely has the mismatch (bug is upstream — instance dedup,
    // blas sharing, or a per-draw write of encoding without matching
    // stride update). If it never fires but slang still reads
    // encoding=1, the bug is in GPU bit packing or slang read.
    {
      static std::atomic<uint32_t> sMismatchCount{0};
      if (texcoordStride == 20u
          && texcoordEncoding != TexcoordEncoding::Float
          && sMismatchCount.fetch_add(1) < 30u) {
        Logger::info(str::format(
          "[SurfaceEncMismatch] surfaceIdx=", surfaceIndex,
          " texStride=", uint32_t(texcoordStride),
          " texEncoding=", uint32_t(texcoordEncoding),
          " texBufIdx=", uint32_t(texcoordBufferIndex),
          " textureFlags=0x", std::hex, textureFlags, std::dec,
          " (bits19-20=", ((textureFlags >> 19) & 0x3u), ")"));
      }
    }

    writeGPUHelper(data, offset, textureFlags);

    // Note: This element of the normal object to world matrix is encoded to minimize padding
    writeGPUHelper(data, offset, normalInstanceToWorld.data[2].z);

    writeGPUHelper(data, offset, clipPlane);

    // eye origin (data15.xyz when eyeParams set; otherwise data15.xy carry
    // the VGUI structured-buffer bindless indices and data15.z stays zero)
    if (eyeParams) {
      writeGPUHelper(data, offset, eyeParams->eyeballOrigin.x);
      writeGPUHelper(data, offset, eyeParams->eyeballOrigin.y);
      writeGPUHelper(data, offset, eyeParams->eyeballOrigin.z);
    } else {
      // NV-DXVK: data15.x — VGUI bindless indices (font in low 16 bits,
      //                     img in high 16 bits) when isVgui; else 0.
      // NV-DXVK: data15.y — VGUI styles bindless index (low 16 bits); else 0.
      // Slang Surface::vguiFontBoundsBufferIndex / vguiImgBoundsBufferIndex /
      // vguiStylesBufferIndex unpack these.
      const uint32_t packedFontImg =
          (uint32_t(vguiFontBoundsBufferIndex) & 0xFFFFu) |
          ((uint32_t(vguiImgBoundsBufferIndex) & 0xFFFFu) << 16);
      const uint32_t packedStyles =
          (uint32_t(vguiStylesBufferIndex) & 0xFFFFu);
      writeGPUHelper(data, offset, packedFontImg);
      writeGPUHelper(data, offset, packedStyles);
      writeGPUHelper(data, offset, uint32_t{});
    }
    // NV-DXVK: data15.w — repurposed from a zero pad to the VGUI per-vertex
    // offset (in float-units). Slang Surface::vguiOffset reads it. Non-VGUI
    // surfaces pass 0 here and isVgui (data0b.z bit 3) stays clear, so the
    // VGUI evaluator dispatch is gated regardless of this value.
    writeGPUHelper(data, offset, vguiOffset);

    assert(offset - oldOffset == kSurfaceGPUSize);
  }

  uint32_t positionBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t positionOffset = 0;
  uint32_t positionStride = 0;

  uint32_t normalBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t normalOffset = 0;
  uint32_t normalStride = 0;
  VkFormat normalFormat = VK_FORMAT_UNDEFINED;

  uint32_t texcoordBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t texcoordOffset = 0;
  uint32_t texcoordStride = 0;

  uint32_t indexBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t firstIndex = 0;
  uint32_t indexStride = 0;

  uint32_t color0BufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t color0Offset = 0;
  uint32_t color0Stride = 0;

  uint32_t surfaceMaterialIndex = kSurfaceInvalidSurfaceMaterialIndex;

  bool isEmissive = false;
  bool isMatte = false;
  bool isStatic = false;
  bool hasMaterialChanged = false;
  bool isAnimatedWater = false;
  // NV-DXVK: lightmap UV (TEXCOORD1) presence. Set when the source draw
  // had a second TEXCOORD attribute (Source/TF2 wall VSes); drives the
  // shader-side hasLightmap flag in flags0 bit 2 so surface_interaction
  // reads the lightmap UV from (texcoordOffset + 8 bytes) and the opaque
  // material samples lightmap textures with that UV instead of the albedo's.
  bool hasLightmap = false;
  // NV-DXVK: TF2 worldspace VGUI per-vertex extras present. Mirrored to
  // flags0 bit 3 in writeGPUData; slang Surface::isVgui reads it back.
  bool isVgui = false;
  // NV-DXVK: TF2 3D-skybox cloud billboard. Set in rtx_instance_manager.cpp
  // only when the instance is IgnoreAntiCulling (3D skybox) AND its material
  // is a fog-synthesizing premultiplied uber-shader — so it targets the
  // cloud billboards specifically, not every premultiplied fog material in
  // the playable world. Mirrored to flags0 bit 4; slang Surface::
  // isTf2SkyboxFog reads it back.
  bool isTf2SkyboxFog = false;
  // First-VGUI-float index in the interleaved per-vertex output (the 8-float
  // VGUI block sits at vguiOffset .. vguiOffset+8). In FLOAT units, not
  // bytes, since the slang surface decode reads the interleaved buffer as
  // a flat float stream. Written into data15.w (was previously a zero pad).
  uint32_t vguiOffset = 0;
  // NV-DXVK: bindless storage-buffer indices for the 3 VGUI structured-
  // buffer SRVs (g_fontBounds, g_imgBounds, g_styles). 16-bit because the
  // bindless table is capped at kMaxBindlessResources=64K. Packed into
  // data15.x (font|img) and data15.y (styles|reserved) by writeGPUData;
  // these slots are always zero-padded for non-eye non-VGUI surfaces and
  // VGUI is mutually exclusive with eyeParams (different shader paths) so
  // the aliasing is safe.
  uint16_t vguiFontBoundsBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
  uint16_t vguiImgBoundsBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
  uint16_t vguiStylesBufferIndex = uint16_t(kSurfaceInvalidBufferIndex);
  bool isClipPlaneEnabled = false;
  bool isTextureFactorBlend = false;
  bool isVertexColorBakedLighting = true;
  bool isMotionBlurMaskOut = false;
  bool skipSurfaceInteractionSpritesheetAdjustment = false;
  bool isInsideFrustum = false;
  bool ignoreTransparencyLayer = false;

  RtTextureArgSource textureColorArg1Source = RtTextureArgSource::Texture;
  RtTextureArgSource textureColorArg2Source = RtTextureArgSource::None;
  DxvkRtTextureOperation textureColorOperation = DxvkRtTextureOperation::Modulate;
  RtTextureArgSource textureAlphaArg1Source = RtTextureArgSource::Texture;
  RtTextureArgSource textureAlphaArg2Source = RtTextureArgSource::None;
  DxvkRtTextureOperation textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
  uint32_t tFactor = 0xffffffff;   // Texture blend factor; opaque white by default
  TexGenMode texgenMode = TexGenMode::None;
  // NV-DXVK: how to interpret raw bytes loaded from the texcoord buffer.
  //   Float (default) — 2x f32 read as-is.
  //   TF2BspUintPacked — 2x u32 bit-cast (the IA reads 2x f32 but the VS
  //     declares TEXCOORD0 as uint and bit-decodes it). Triggered on VSes
  //     whose ISGN reports Format=uint for TEXCOORD0/1; the discriminator
  //     is shader-agnostic so any other engine using the same packing
  //     trick gets fixed too without a per-VS hash list. Decode formula
  //     comes from disassembling TF2 BSP world VSes (e.g. 7c38fdf4,
  //     1953b6e9): TEXCOORD0 = (asint(uint(x) >> 3) + 0xf0000000) * 0.000061f,
  //     TEXCOORD1 = float(uint(x)) * 0.000015f.
  enum class TexcoordEncoding : uint8_t {
    Float            = 0,
    TF2BspUintPacked = 1,
  };
  TexcoordEncoding texcoordEncoding = TexcoordEncoding::Float;
  std::optional<RtEyeParams> eyeParams = {};

  bool doBuffersMatch(const RtSurface& surface) {
    return positionBufferIndex == surface.positionBufferIndex
        && positionOffset == surface.positionOffset
        && previousPositionBufferIndex == surface.previousPositionBufferIndex
        && normalBufferIndex == surface.normalBufferIndex
        && normalOffset == surface.normalOffset
        && texcoordBufferIndex == surface.texcoordBufferIndex
        && texcoordOffset == surface.texcoordOffset
        && color0BufferIndex == surface.color0BufferIndex
        && color0Offset == surface.color0Offset
        && firstIndex == surface.firstIndex;
  }

  void printDebugInfo(const char* name = "") const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "RtSurface ", name, "\n",
      "  address: ", this, "\n",
      "  surfaceMaterialIndex: ", surfaceMaterialIndex, "\n",
      "  associatedGeometryHash: 0x", std::hex, associatedGeometryHash, std::dec, "\n",
      "  objectPickingValue: ", objectPickingValue, "\n",
      "  decalSortOrder: ", decalSortOrder));
    
    // Print buffer info
    Logger::warn("=== Buffer Info ===");
    Logger::warn(str::format(
      "  positionBufferIndex: ", positionBufferIndex, "\n",
      "  positionOffset: ", positionOffset, "\n",
      "  positionStride: ", positionStride, "\n",
      "  previousPositionBufferIndex: ", previousPositionBufferIndex, "\n",
      "  normalBufferIndex: ", normalBufferIndex, "\n",
      "  normalOffset: ", normalOffset, "\n",
      "  normalStride: ", normalStride, "\n",
      "  normalFormat: ", static_cast<int>(normalFormat), "\n",
      "  texcoordBufferIndex: ", texcoordBufferIndex, "\n",
      "  texcoordOffset: ", texcoordOffset, "\n",
      "  texcoordStride: ", texcoordStride, "\n",
      "  indexBufferIndex: ", indexBufferIndex, "\n",
      "  firstIndex: ", firstIndex, "\n",
      "  indexStride: ", indexStride, "\n",
      "  color0BufferIndex: ", color0BufferIndex, "\n",
      "  color0Offset: ", color0Offset, "\n",
      "  color0Stride: ", color0Stride));
    
    // Print boolean flags
    Logger::warn("=== Boolean Flags ===");
    Logger::warn(str::format(
      "  isEmissive: ", isEmissive, "\n",
      "  isMatte: ", isMatte, "\n",
      "  isStatic: ", isStatic, "\n",
      "  hasMaterialChanged: ", hasMaterialChanged, "\n",
      "  isAnimatedWater: ", isAnimatedWater, "\n",
      "  isClipPlaneEnabled: ", isClipPlaneEnabled, "\n",
      "  isTextureFactorBlend: ", isTextureFactorBlend, "\n",
      "  isMotionBlurMaskOut: ", isMotionBlurMaskOut, "\n",
      "  skipSurfaceInteractionSpritesheetAdjustment: ", skipSurfaceInteractionSpritesheetAdjustment, "\n",
      "  isInsideFrustum: ", isInsideFrustum, "\n",
      "  ignoreTransparencyLayer: ", ignoreTransparencyLayer));
    
    // Print alpha state
    Logger::warn("=== Alpha State ===");
    Logger::warn(str::format(
      "  isBlendingDisabled: ", alphaState.isBlendingDisabled, "\n",
      "  isFullyOpaque: ", alphaState.isFullyOpaque, "\n",
      "  alphaTestType: ", static_cast<int>(alphaState.alphaTestType), "\n",
      "  alphaTestReferenceValue: ", static_cast<int>(alphaState.alphaTestReferenceValue), "\n",
      "  blendType: ", static_cast<int>(alphaState.blendType), "\n",
      "  invertedBlend: ", alphaState.invertedBlend, "\n",
      "  emissiveBlend: ", alphaState.emissiveBlend, "\n",
      "  isParticle: ", alphaState.isParticle, "\n",
      "  isDecal: ", alphaState.isDecal));
    
    // Print texture operations
    Logger::warn("=== Texture Operations ===");
    Logger::warn(str::format(
      "  textureColorArg1Source: ", static_cast<int>(textureColorArg1Source), "\n",
      "  textureColorArg2Source: ", static_cast<int>(textureColorArg2Source), "\n",
      "  textureColorOperation: ", static_cast<int>(textureColorOperation), "\n",
      "  textureAlphaArg1Source: ", static_cast<int>(textureAlphaArg1Source), "\n",
      "  textureAlphaArg2Source: ", static_cast<int>(textureAlphaArg2Source), "\n",
      "  textureAlphaOperation: ", static_cast<int>(textureAlphaOperation), "\n",
      "  texgenMode: ", static_cast<int>(texgenMode), "\n",
      "  tFactor: 0x", std::hex, tFactor, std::dec));
    
    // Print spritesheet info
    Logger::warn("=== Spritesheet Info ===");
    Logger::warn(str::format(
      "  spriteSheetRows: ", static_cast<int>(spriteSheetRows), "\n",
      "  spriteSheetCols: ", static_cast<int>(spriteSheetCols), "\n",
      "  spriteSheetFPS: ", static_cast<int>(spriteSheetFPS)));
    
    // Print instance info
    Logger::warn("=== Instance Info ===");
    Logger::warn(str::format(
      "  instancesToObject: ", (instancesToObject != nullptr ? "valid" : "null"), "\n",
      "  surfaceIndexOfFirstInstance: ", surfaceIndexOfFirstInstance));
#endif
  }

  // Used for calculating hashes, keep the members padded and default initialized
  struct AlphaState {
    bool isBlendingDisabled = true;
    bool isFullyOpaque = false;
    AlphaTestType alphaTestType = AlphaTestType::kAlways;
    uint8_t alphaTestReferenceValue = 0;
    BlendType blendType = BlendType::kAlpha;
    bool invertedBlend = false;
    bool emissiveBlend = false;
    bool isParticle = false;
    bool isDecal = false;
  } alphaState;

  // Original draw call state
  DxvkBlendMode blendModeState;

  // Static validation to detect any changes that require an alignment re-check
  static_assert(sizeof(AlphaState) == 9);

  Matrix4 objectToWorld;
  Matrix4 prevObjectToWorld;
  Matrix3 normalObjectToWorld;
  Matrix4 textureTransform;
  Vector4 clipPlane;

  uint8_t spriteSheetRows = 1;
  uint8_t spriteSheetCols = 1;
  uint8_t spriteSheetFPS = 0;

  XXH64_hash_t associatedGeometryHash; // NOTE: This is used for the debug view
  uint32_t objectPickingValue = 0; // NOTE: a value to fill GBUFFER_BINDING_PRIMARY_OBJECT_PICKING_OUTPUT
  uint32_t decalSortOrder = 0; // see: InstanceManager::m_decalSortOrderCounter

  // PointInstancer support - this surface may represent multiple instances, one for each transform in instancesToObject
  const std::vector<Matrix4>* instancesToObject = nullptr;
  // NV-DXVK: Lifetime owner for instancesToObject when the pointer references
  // storage whose lifetime is tied to the frame that created it (e.g. the d3d11
  // bone-fanout path). Without this, the raw pointer above could dangle once
  // the originating frame's transform buffer is recycled. Sources with external
  // ownership leave this null.
  std::shared_ptr<const std::vector<Matrix4>> instancesToObjectOwner;
  // on the GPU, multiple copies of this surface with different transforms will exist.  They will be in a continuous block, starting at surfaceIndexOfFirstInstance.
  size_t surfaceIndexOfFirstInstance = SIZE_MAX;
};

// Shared Material Defaults/Limits

struct LegacyMaterialDefaults {
  friend class ImGUI;
  RTX_OPTION("rtx.legacyMaterial", float, anisotropy, 0.f, "The default roughness anisotropy to use for non-replaced \"legacy\" materials. Should be in the range -1 to 1, where 0 is isotropic.");
  RTX_OPTION("rtx.legacyMaterial", float, emissiveIntensity, 0.f, "The default emissive intensity to use for non-replaced \"legacy\" materials.");
  RTX_OPTION("rtx.legacyMaterial", bool, useAlbedoTextureIfPresent, true, "A flag to determine if an \"albedo\" texture (a qualifying color texture) from the original application should be used if present on non-replaced \"legacy\" materials.");
  RTX_OPTION("rtx.legacyMaterial", Vector3, albedoConstant, Vector3(1.0f, 1.0f, 1.0f), "The default albedo constant to use for non-replaced \"legacy\" materials. Should be a color in sRGB colorspace with gamma encoding.");
  RTX_OPTION("rtx.legacyMaterial", float, opacityConstant, 1.f, "The default opacity constant to use for non-replaced \"legacy\" materials. Should be in the range 0 to 1.");
  RTX_OPTION_ENV("rtx.legacyMaterial", float, roughnessConstant, 0.7f, "DXVK_LEGACY_MATERIAL_DEFAULT_ROUGHNESS", "The default perceptual roughness constant to use for non-replaced \"legacy\" materials. Should be in the range 0 to 1.");
  RTX_OPTION("rtx.legacyMaterial", float, metallicConstant, 0.1f, "The default metallic constant to use for non-replaced \"legacy\" materials. Should be in the range 0 to 1.");
  RTX_OPTION("rtx.legacyMaterial", Vector3, emissiveColorConstant, Vector3(0.0f, 0.0f, 0.0f), "The default emissive color constant to use for non-replaced \"legacy\" materials. Should be a color in sRGB colorspace with gamma encoding.");
  RTX_OPTION("rtx.legacyMaterial", bool, enableEmissive, false, "A flag to determine if emission should be used on non-replaced \"legacy\" materials.");
  RTX_OPTION("rtx.legacyMaterial", bool, ignoreAlphaChannel, false, "A flag to determine if the albedo alpha channel should be ignored on non-replaced \"legacy\" materials.");
  RTX_OPTION("rtx.legacyMaterial", bool, enableThinFilm, false, "A flag to determine if a thin-film layer should be used on non-replaced \"legacy\" materials.");
  RTX_OPTION("rtx.legacyMaterial", bool, alphaIsThinFilmThickness, false, "A flag to determine if the alpha channel from the albedo source should be treated as thin film thickness on non-replaced \"legacy\" materials.");
  // Note: Should be something non-zero as 0 is an invalid thickness to have (even if this is just unused).
  RTX_OPTION("rtx.legacyMaterial", float, thinFilmThicknessConstant, 200.f,
             "The thickness (in nanometers) of the thin-film layer assuming it is enabled on non-replaced \"legacy\" materials.\n"
             "Should be any value larger than 0, typically within the wavelength of light, but must be less than or equal to OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS (" STRINGIFY(OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS) " nm).");
};

// Surface Materials

// Todo: Compute size directly from sizeof of GPU structure (by including it), for now computed by sum of members manually.
// Blocked on float16 support on the c++ side.
// NV-DXVK: grown from 64 to 80 bytes (5 × vec4 → 5 × vec4 + 1 × vec4) to
// carry the screen-space emissive override block (vec4 packed-half UV1
// matrix + vec2 packed-half translate + uint16 mask texture index + uint16
// padding). Used only when OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_
// EMISSIVE is set; default-initialised to identity for every other material
// so legacy paths render bit-identically. The trailing 16 bytes apply to
// translucent / ray-portal materials too — those types just leave them
// zero-initialised.
constexpr std::size_t kSurfaceMaterialGPUSize = 5 * 4 * 4;
// Note: 0xFFFF used for inactive texture index to indicate to the GPU that no texture is in use for a specific variable
// (as some are optional). Also used for debugging to provide wildly out of range values in case one is not set.
constexpr uint32_t kSurfaceMaterialInvalidTextureIndex = 0xFFFFu;
// Note: These defaults are used in places where no value is available for the constructor of various Surface Materials, just to
// keep things consistent across the codebase.

enum class RtSurfaceMaterialType {
  // Todo: Legacy SurfaceMaterialType in the future
  Opaque = 0,
  Translucent,
  RayPortal,

  // Extensions
  Subsurface,

  Count
};

// Todo: Legacy SurfaceMaterial in the future

struct RtOpaqueSurfaceMaterial {
  RtOpaqueSurfaceMaterial(
    uint32_t albedoOpacityTextureIndex, uint32_t normalTextureIndex,
    uint32_t tangentTextureIndex, uint32_t heightTextureIndex, uint32_t roughnessTextureIndex,
    uint32_t metallicTextureIndex, uint32_t emissiveColorTextureIndex,
    float anisotropy, float emissiveIntensity,
    const Vector4& albedoOpacityConstant,
    float roughnessConstant, float metallicConstant,
    const Vector3& emissiveColorConstant, bool enableEmission,
    bool ignoreAlphaChannel, bool enableThinFilm, bool alphaIsThinFilmThickness, float thinFilmThicknessConstant,
    uint32_t samplerIndex, float displaceIn, float displaceOut,
    uint32_t subsurfaceMaterialIndex, bool isRaytracedRenderTarget,
    uint16_t samplerFeedbackStamp,
    uint32_t secondaryTextureIndex = 0,
    uint32_t ambientOcclusionTextureIndex = kSurfaceMaterialInvalidTextureIndex,
    uint32_t lightmapTextureIndex = kSurfaceMaterialInvalidTextureIndex,
    uint32_t lightmap2TextureIndex = kSurfaceMaterialInvalidTextureIndex,
    uint32_t detailTextureIndex = kSurfaceMaterialInvalidTextureIndex,
    uint32_t cloudMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex,
    // NV-DXVK: TF2/Source `c_useAlphaModulateEmissive` semantic — when the
    // PS multiplies its emission by albedo.a per pixel, the slang shader
    // must do the same. Carried as a flag bit on the GPU material.
    bool alphaModulateEmissive = false,
    // NV-DXVK: emissiveColorConstant carries c_emissiveTint and must
    // multiply the per-pixel emissiveTexture sample (vs. the legacy
    // overwrite-as-fallback semantic). See OPAQUE_SURFACE_MATERIAL_FLAG_
    // EMISSIVE_TINT_FROM_CONSTANT.
    bool emissiveTintFromConstant = false,
    // NV-DXVK: TF2 viewmodel screen-space scrolling overlay emissive —
    // see OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE.
    bool hasScreenSpaceEmissive = false,
    const Vector2& screenSpaceEmissiveMatRow0 = Vector2(1.f, 0.f),
    const Vector2& screenSpaceEmissiveMatRow1 = Vector2(0.f, 1.f),
    const Vector2& screenSpaceEmissiveTranslate = Vector2(0.f, 0.f),
    uint32_t screenSpaceEmissiveMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex,
    // NV-DXVK: true when the albedo/opacity texture is bound with an
    // sRGB-format view (HW already did sRGB->linear). Skips the software
    // gammaToLinear() on the GPU. See OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_SRGB.
    bool albedoIsSRGB = false,
    // NV-DXVK: TF2 3D-skybox cloud billboard — the opaque surface shader
    // reconstructs the game's fog-blend synthesis. See
    // OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG.
    bool tf2SkyboxFog = false,
    // NV-DXVK: source D3D11 draw uses premultiplied alpha blending — the
    // slang shader must skip the opacity-multiply inside
    // albedoToAdjustedAlbedo / calcBaseReflectivity for this surface. See
    // OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED.
    bool albedoIsPremultiplied = false
  ) :
    m_albedoOpacityTextureIndex{ albedoOpacityTextureIndex }, m_secondaryTextureIndex{secondaryTextureIndex}, m_normalTextureIndex{ normalTextureIndex },
    m_tangentTextureIndex { tangentTextureIndex }, m_heightTextureIndex { heightTextureIndex }, m_roughnessTextureIndex{ roughnessTextureIndex },
    m_metallicTextureIndex{ metallicTextureIndex }, m_emissiveColorTextureIndex{ emissiveColorTextureIndex },
    m_ambientOcclusionTextureIndex{ ambientOcclusionTextureIndex },
    m_lightmapTextureIndex{ lightmapTextureIndex }, m_lightmap2TextureIndex{ lightmap2TextureIndex },
    m_detailTextureIndex{ detailTextureIndex }, m_cloudMaskTextureIndex{ cloudMaskTextureIndex },
    m_anisotropy{ anisotropy }, m_emissiveIntensity{ emissiveIntensity },
    m_albedoOpacityConstant{ albedoOpacityConstant },
    m_roughnessConstant{ roughnessConstant }, m_metallicConstant{ metallicConstant },
    m_emissiveColorConstant{ emissiveColorConstant }, m_enableEmission{ enableEmission },
    m_ignoreAlphaChannel { ignoreAlphaChannel }, m_enableThinFilm { enableThinFilm }, m_alphaIsThinFilmThickness { alphaIsThinFilmThickness },
    m_thinFilmThicknessConstant { thinFilmThicknessConstant }, m_samplerIndex{ samplerIndex }, m_displaceIn{ displaceIn },
    m_displaceOut{ displaceOut }, m_subsurfaceMaterialIndex(subsurfaceMaterialIndex), m_isRaytracedRenderTarget(isRaytracedRenderTarget),
    m_samplerFeedbackStamp{ samplerFeedbackStamp },
    m_alphaModulateEmissive{ alphaModulateEmissive },
    m_emissiveTintFromConstant{ emissiveTintFromConstant },
    m_hasScreenSpaceEmissive{ hasScreenSpaceEmissive },
    m_screenSpaceEmissiveMatRow0{ screenSpaceEmissiveMatRow0 },
    m_screenSpaceEmissiveMatRow1{ screenSpaceEmissiveMatRow1 },
    m_screenSpaceEmissiveTranslate{ screenSpaceEmissiveTranslate },
    m_screenSpaceEmissiveMaskTextureIndex{ screenSpaceEmissiveMaskTextureIndex },
    m_albedoIsSRGB{ albedoIsSRGB },
    m_tf2SkyboxFog{ tf2SkyboxFog },
    m_albedoIsPremultiplied{ albedoIsPremultiplied }
  {
    updateCachedData();
    updateCachedHash();
  }

  void writeGPUData(unsigned char* data, std::size_t& offset) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;
    uint16_t flags = surfaceMaterialTypeOpaque;

    // For decode process, see surface_material.h
    // this data is accessed from uint16_t data[32], so data[n] refers to a pair of bytes.

    if (m_enableThinFilm) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER;

      // Note: Only consider setting alpha as thin film thickness flag if the thin film is enabled, GPU relies on
      // this logical ordering.
      if (m_alphaIsThinFilmThickness) {
        flags |= OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_IS_THIN_FILM_THICKNESS;
      }
    }

    if (m_ignoreAlphaChannel) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL;
    }
    // NOTE: We keep the most commonly used elements in the material close together near the beginning
    //       This hopefully reduces loads for cases like opacity detection.

    if (m_isRaytracedRenderTarget) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET;
    }

    // NV-DXVK: emit alpha-modulate-emissive bit so the slang shader
    // multiplies emissiveRadiance by opacity, matching the original
    // PS's `emissive *= albedo.a` semantic when c_useAlphaModulate
    // Emissive was non-zero.
    if (m_alphaModulateEmissive) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_MODULATE_EMISSIVE;
    }
    // NV-DXVK: emit tint-from-constant bit so the slang shader treats
    // emissiveColorConstant as a per-draw c_emissiveTint that multiplies
    // the per-pixel emissiveTexture sample (vs. overwriting the constant
    // with the sample, which is the default for legacy paths whose
    // emissiveColorConstant is a no-texture fallback rather than a tint).
    if (m_emissiveTintFromConstant) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TINT_FROM_CONSTANT;
    }
    // NV-DXVK: TF2 viewmodel screen-space scrolling overlay — gates the
    // slang shader to sample emissive at SV_Position-derived UV instead of
    // mesh UV. Matrix + translate + mask texture index live in the new
    // data[32-39] block at the end of the GPU material.
    if (m_hasScreenSpaceEmissive) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE;
    }
    // NV-DXVK: albedo texture uses an sRGB-format view — tell the slang
    // shader to skip its software gammaToLinear() so we don't double-decode.
    if (m_albedoIsSRGB) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_SRGB;
    }
    // NV-DXVK: TF2 3D-skybox cloud billboard — opaque surface shader
    // reconstructs the game's fog-blend synthesis (see cb.tf2Fog*).
    if (m_tf2SkyboxFog) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG;
    }
    // NV-DXVK: premultiplied-alpha-blend source — slang shader skips
    // the opacity-multiply inside albedoToAdjustedAlbedo /
    // calcBaseReflectivity to avoid double-darkening on premult surfaces.
    if (m_albedoIsPremultiplied) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED;
    }

    float displaceIn = m_displaceIn * getDisplacementInFactor();
    float displaceOut = m_displaceOut * getDisplacementOutFactor();
    uint32_t heightTextureIndex = m_heightTextureIndex;
    if (hasValidDisplacement()) {
      flags |= OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT;
    } else {
      // If any POM attribute would disable POM, just disable all POM attributes.
      displaceIn = 0.f;
      displaceOut = 0.f;
      heightTextureIndex = BINDING_INDEX_INVALID;
    }
    assert(displaceIn <= FLOAT16_MAX);
    assert(displaceOut <= FLOAT16_MAX);

    assert(m_subsurfaceMaterialIndex <= SURFACE_INDEX_MAX_VALUE);

    // data[0 - 3]
    writeGPUHelper(data, offset, flags);
    writeGPUHelperExplicit<2>(data, offset, m_samplerIndex);
    writeGPUHelperExplicit<2>(data, offset, m_albedoOpacityTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_secondaryTextureIndex);

    // data[4 - 7]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_albedoOpacityConstant.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_albedoOpacityConstant.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_albedoOpacityConstant.z));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_albedoOpacityConstant.w));

    // data[8 - 11]
    writeGPUHelper(data, offset, glm::packHalf1x16(displaceIn));
    writeGPUHelper(data, offset, glm::packHalf1x16(displaceOut));
    writeGPUHelperExplicit<2>(data, offset, m_heightTextureIndex);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_cachedThinFilmNormalizedThicknessConstant));

    // data[12 - 15]
    writeGPUHelperExplicit<2>(data, offset, m_emissiveColorTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_roughnessTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_metallicTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_normalTextureIndex);

    // data[16 - 19]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.z));
    assert(m_cachedEmissiveIntensity <= FLOAT16_MAX);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_cachedEmissiveIntensity));

    // data[20 - 23]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_roughnessConstant));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_metallicConstant));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_anisotropy));
    writeGPUHelperExplicit<2>(data, offset, m_tangentTextureIndex);

    // data[24-25]
    writeGPUHelper(data, offset, m_subsurfaceMaterialIndex);

    // data[26]
    writeGPUHelperExplicit<2>(data, offset, m_samplerFeedbackStamp);

    // data[27] — ambientOcclusion / cavity texture, sampled at the same UV
    // as albedo and multiplied into albedo in the shader.
    writeGPUHelperExplicit<2>(data, offset, m_ambientOcclusionTextureIndex);

    // data[28-31] — TF2 aux textures: lightmap (baked static GI), lightmap2
    // (bumped / HDR range-extension), detail (fine-scale variation),
    // cloudMask (large-scale tonal banding). Exactly fills the remaining
    // 8 bytes of trailing padding.
    writeGPUHelperExplicit<2>(data, offset, m_lightmapTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_lightmap2TextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_detailTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_cloudMaskTextureIndex);

    // data[32-39] — TF2 viewmodel screen-space scrolling overlay block.
    // Layout (16 bytes):
    //   data[32] = M.row0.x  (packHalf)
    //   data[33] = M.row0.y  (packHalf)
    //   data[34] = M.row1.x  (packHalf)
    //   data[35] = M.row1.y  (packHalf)
    //   data[36] = T.x       (packHalf)
    //   data[37] = T.y       (packHalf)
    //   data[38] = screenSpaceEmissiveMaskTextureIndex (uint16)
    //   data[39] = padding   (0)
    // Slang reads the matrix + translate as float2 pairs and decodes the
    // mask texture index as a bindless uint16. Default-initialised values
    // for non-screen-space materials produce identity matrix + zero
    // translate + invalid mask index, costing nothing visually.
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveMatRow0.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveMatRow0.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveMatRow1.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveMatRow1.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveTranslate.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_screenSpaceEmissiveTranslate.y));
    writeGPUHelperExplicit<2>(data, offset, m_screenSpaceEmissiveMaskTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, uint16_t(0));  // padding to fill 16 bytes
    assert(offset - oldOffset == kSurfaceMaterialGPUSize);
  }

  bool validate() const {
    const bool hasTexture = m_albedoOpacityTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_normalTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_tangentTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_heightTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_roughnessTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_metallicTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_emissiveColorTextureIndex != kSurfaceMaterialInvalidTextureIndex;

    return !hasTexture || m_samplerIndex != kSurfaceMaterialInvalidTextureIndex;
  }

  bool hasValidDisplacement() const {
    return (m_displaceIn > 0.f || m_displaceOut > 0.f) && m_heightTextureIndex != BINDING_INDEX_INVALID;
  }

  bool operator==(const RtOpaqueSurfaceMaterial& r) const {
    return m_cachedHash == r.m_cachedHash;
  }

  XXH64_hash_t getHash() const {
    return m_cachedHash;
  }

  uint32_t getSamplerIndex() const {
    return m_samplerIndex;
  }

  uint32_t getAlbedoOpacityTextureIndex() const {
    return m_albedoOpacityTextureIndex;
  }

  uint32_t getNormalTextureIndex() const {
    return m_normalTextureIndex;
  }

  uint32_t getTangentTextureIndex() const {
    return m_tangentTextureIndex;
  }

  uint32_t getHeightTextureIndex() const {
    return m_heightTextureIndex;
  }

  uint32_t getRoughnessTextureIndex() const {
    return m_roughnessTextureIndex;
  }

  uint32_t getMetallicTextureIndex() const {
    return m_metallicTextureIndex;
  }

  uint32_t getEmissiveColorTextureIndex() const {
    return m_emissiveColorTextureIndex;
  }

  uint32_t getAmbientOcclusionTextureIndex() const {
    return m_ambientOcclusionTextureIndex;
  }

  uint32_t getLightmapTextureIndex() const { return m_lightmapTextureIndex; }
  uint32_t getLightmap2TextureIndex() const { return m_lightmap2TextureIndex; }
  uint32_t getDetailTextureIndex() const { return m_detailTextureIndex; }
  uint32_t getCloudMaskTextureIndex() const { return m_cloudMaskTextureIndex; }

  float getAnisotropy() const {
    return m_anisotropy;
  }

  float getEmissiveIntensity() const {
    return m_emissiveIntensity;
  }

  Vector4 getAlbedoOpacityConstant() const {
    return m_albedoOpacityConstant;
  }

  float getRoughnessConstant() const {
    return m_roughnessConstant;
  }

  float getMetallicConstant() const {
    return m_metallicConstant;
  }

  Vector3 getEmissiveColorConstant() const {
    return m_emissiveColorConstant;
  }

  bool getEnableEmission() const {
    return m_enableEmission;
  }

  uint32_t getSubsurfaceMaterialIndex() const {
    return m_subsurfaceMaterialIndex;
  }

  uint32_t getIsRaytracedRenderTarget() const {
    return m_isRaytracedRenderTarget;
  }

private:
  void updateCachedHash() {
    static_assert(
      sizeof(*this) == 184,
      "add new member for hashing if needed: add a MEMBER into the struct + add a VALUE into the list-init"
    );
    struct HashStruct {
      uint32_t albedoOpacityTextureIndex;
      uint32_t normalTextureIndex;
      uint32_t tangentTextureIndex;
      uint32_t heightTextureIndex;
      uint32_t roughnessTextureIndex;
      uint32_t metallicTextureIndex;
      uint32_t emissiveColorTextureIndex;
      float anisotropy;
      float emissiveIntensity;
      Vector4 albedoOpacityConstant;
      float roughnessConstant;
      float metallicConstant;
      Vector3 emissiveColorConstant;
      uint32_t enableEmission;            // NOTE: uint32_t to avoid padding
      uint32_t ignoreAlphaChannel;        // NOTE: uint32_t to avoid padding
      uint32_t enableThinFilm;            // NOTE: uint32_t to avoid padding
      uint32_t alphaIsThinFilmThickness;  // NOTE: uint32_t to avoid padding
      float thinFilmThicknessConstant;
      uint32_t samplerIndex;
      float displaceIn;
      float displaceOut;
      uint32_t subsurfaceMaterialIndex;
      uint32_t isRaytracedRenderTarget;   // NOTE: uint32_t to avoid padding
      uint32_t samplerFeedbackStamp;      // NOTE: uint32_t to avoid padding
      uint32_t secondaryTextureIndex;
      uint32_t ambientOcclusionTextureIndex;
      uint32_t lightmapTextureIndex;
      uint32_t lightmap2TextureIndex;
      uint32_t detailTextureIndex;
      uint32_t cloudMaskTextureIndex;
      // NV-DXVK: alpha-modulate-emissive bit must participate in the cache
      // key — two materials that share every other field but differ on the
      // c_useAlphaModulateEmissive semantic produce different per-pixel
      // emissive radiance and must not collide in the surface dedup map.
      uint32_t alphaModulateEmissive;     // NOTE: uint32_t to avoid padding
      uint32_t emissiveTintFromConstant;  // NOTE: uint32_t to avoid padding
      // NV-DXVK: TF2 viewmodel screen-space scrolling overlay emissive —
      // matrix + translate + mask texture index participate in the cache
      // key so two materials sharing every other field but differing in
      // the screen-space transform produce distinct surface entries.
      uint32_t hasScreenSpaceEmissive;          // NOTE: uint32_t to avoid padding
      Vector2 screenSpaceEmissiveMatRow0;
      Vector2 screenSpaceEmissiveMatRow1;
      Vector2 screenSpaceEmissiveTranslate;
      uint32_t screenSpaceEmissiveMaskTextureIndex;
      uint32_t albedoIsSRGB;              // NOTE: uint32_t to avoid padding
      uint32_t tf2SkyboxFog;              // NOTE: uint32_t to avoid padding
      uint32_t albedoIsPremultiplied;     // NOTE: uint32_t to avoid padding
      // NOTE: There must be NO padding between members, as the struct is used for hashing
    };
    static_assert(alignof(HashStruct) == 4 && sizeof(HashStruct) % 4 == 0);
    HashStruct hashData = HashStruct{
      m_albedoOpacityTextureIndex,
      m_normalTextureIndex,
      m_tangentTextureIndex,
      m_heightTextureIndex,
      m_roughnessTextureIndex,
      m_metallicTextureIndex,
      m_emissiveColorTextureIndex,
      m_anisotropy,
      m_emissiveIntensity,
      m_albedoOpacityConstant,
      m_roughnessConstant,
      m_metallicConstant,
      m_emissiveColorConstant,
      m_enableEmission,
      m_ignoreAlphaChannel,
      m_enableThinFilm,
      m_alphaIsThinFilmThickness,
      m_thinFilmThicknessConstant,
      m_samplerIndex,
      m_displaceIn,
      m_displaceOut,
      m_subsurfaceMaterialIndex,
      m_isRaytracedRenderTarget,
      m_samplerFeedbackStamp,
      m_secondaryTextureIndex,
      m_ambientOcclusionTextureIndex,
      m_lightmapTextureIndex,
      m_lightmap2TextureIndex,
      m_detailTextureIndex,
      m_cloudMaskTextureIndex,
      m_alphaModulateEmissive,
      m_emissiveTintFromConstant,
      m_hasScreenSpaceEmissive ? 1u : 0u,
      m_screenSpaceEmissiveMatRow0,
      m_screenSpaceEmissiveMatRow1,
      m_screenSpaceEmissiveTranslate,
      m_screenSpaceEmissiveMaskTextureIndex,
      m_albedoIsSRGB ? 1u : 0u,
      m_tf2SkyboxFog ? 1u : 0u,
      m_albedoIsPremultiplied ? 1u : 0u,
    };
    m_cachedHash = XXH3_64bits(&hashData, sizeof(hashData));
  }

  void updateCachedData() {
    // Note: Ensure the thin film thickness constant is within the expected range for normalization.
    assert(m_thinFilmThicknessConstant <= OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS);

    // Note: Opaque material does not take an emissive radiance directly, so zeroing out the intensity works
    // fine as a way to disable it (in case a texture is in use).
    m_cachedEmissiveIntensity = std::min(m_enableEmission ? m_emissiveIntensity : 0.0f, FLOAT16_MAX);
    // Note: Pre-normalize thickness constant so that it does not need to be done on the GPU.
    m_cachedThinFilmNormalizedThicknessConstant = m_thinFilmThicknessConstant / OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS;
  }

  uint32_t m_albedoOpacityTextureIndex;
  uint32_t m_secondaryTextureIndex;
  uint32_t m_normalTextureIndex;
  uint32_t m_tangentTextureIndex;
  uint32_t m_heightTextureIndex;
  uint32_t m_roughnessTextureIndex;
  uint32_t m_metallicTextureIndex;
  uint32_t m_emissiveColorTextureIndex;
  uint32_t m_samplerIndex;

  float m_anisotropy;
  float m_emissiveIntensity;

  Vector4 m_albedoOpacityConstant;
  float m_roughnessConstant;
  float m_metallicConstant;
  Vector3 m_emissiveColorConstant;

  bool m_enableEmission;

  bool m_ignoreAlphaChannel;
  bool m_enableThinFilm;
  bool m_alphaIsThinFilmThickness;
  float m_thinFilmThicknessConstant;

  // How far inwards a height_texture value of 0 maps to.
  float m_displaceIn;
  // How far outwards a height_texture value of 1 maps to.
  float m_displaceOut;

  uint32_t m_subsurfaceMaterialIndex;

  bool m_isRaytracedRenderTarget;

  uint32_t m_ambientOcclusionTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_lightmapTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_lightmap2TextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_detailTextureIndex = kSurfaceMaterialInvalidTextureIndex;
  uint32_t m_cloudMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex;

  uint16_t m_samplerFeedbackStamp;

  // NV-DXVK: TF2/Source `c_useAlphaModulateEmissive` — emission multiplied
  // by albedo.a per-pixel in slang when set. Encoded as a flag bit on the
  // GPU surface material; see writeGPUData() and OPAQUE_SURFACE_MATERIAL_
  // FLAG_ALPHA_MODULATE_EMISSIVE.
  bool m_alphaModulateEmissive = false;
  // NV-DXVK: emissiveColorConstant is a per-draw tint (c_emissiveTint)
  // that must MULTIPLY the per-pixel emissive texture sample. See flag
  // OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TINT_FROM_CONSTANT.
  bool m_emissiveTintFromConstant = false;

  // NV-DXVK: TF2 viewmodel screen-space scrolling overlay emissive. When
  // m_hasScreenSpaceEmissive is true (encoded as
  // OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE on the GPU),
  // the slang shader samples emissive at SV_Position-derived UV using the
  // 2x2 matrix (rows 0/1) plus the translate, then optionally multiplies
  // by the mask texture sampled at mesh UV.
  bool m_hasScreenSpaceEmissive = false;
  Vector2 m_screenSpaceEmissiveMatRow0 = Vector2(1.f, 0.f);
  Vector2 m_screenSpaceEmissiveMatRow1 = Vector2(0.f, 1.f);
  Vector2 m_screenSpaceEmissiveTranslate = Vector2(0.f, 0.f);
  uint32_t m_screenSpaceEmissiveMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex;

  // NV-DXVK: albedo texture is bound with an sRGB-format image view (HW
  // sRGB->linear on sample). Encoded as OPAQUE_SURFACE_MATERIAL_FLAG_
  // ALBEDO_IS_SRGB; the slang shader then skips its software gammaToLinear()
  // for albedo. Participates in the material hash (per the HashStruct
  // convention), though in practice it is a deterministic function of
  // m_albedoOpacityTextureIndex which is already hashed.
  bool m_albedoIsSRGB = false;

  // NV-DXVK: TF2 3D-skybox cloud billboard. Encoded as
  // OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG; the slang opaque surface
  // shader then reconstructs the game's fog-blend synthesis from the
  // captured cb.tf2Fog* constants. Set by FillMaterialData when the PS
  // reads c_fogColorFactor and the draw is premultiplied-blended.
  bool m_tf2SkyboxFog = false;

  // NV-DXVK: source D3D11 draw uses premultiplied alpha blending
  // (rt0: BlendEnable=1, SrcBlend=ONE, DestBlend=INV_SRC_ALPHA,
  // BlendOp=ADD). When true, the slang shader passes opacity=1 to
  // albedoToAdjustedAlbedo and calcBaseReflectivity so the encoded
  // albedo stays as the (already-premultiplied) sample color instead
  // of double-multiplying by alpha — fixes the soft-edge darkening
  // that produced noisy dark dots on TF2 cloud billboards. Set by
  // FillMaterialData purely from D3D state, no hash list.
  bool m_albedoIsPremultiplied = false;

  XXH64_hash_t m_cachedHash;

  // Note: Cached values are not involved in the hash as they are derived from the input data
  float m_cachedEmissiveIntensity;
  float m_cachedThinFilmNormalizedThicknessConstant;
};

struct RtTranslucentSurfaceMaterial {
  RtTranslucentSurfaceMaterial(
    uint32_t normalTextureIndex,
    uint32_t transmittanceTextureIndex,
    uint32_t emissiveColorTextureIndex,
    float refractiveIndex,
    float transmittanceMeasurementDistance, const Vector3& transmittanceColor,
    bool enableEmission, float emissiveIntensity, const Vector3& emissiveColorConstant,
    bool isThinWalled, float thinWallThickness, bool useDiffuseLayer, uint32_t samplerIndex) :
    m_normalTextureIndex(normalTextureIndex),
    m_transmittanceTextureIndex(transmittanceTextureIndex),
    m_emissiveColorTextureIndex(emissiveColorTextureIndex),
    m_refractiveIndex(refractiveIndex),
    m_transmittanceMeasurementDistance(transmittanceMeasurementDistance), m_transmittanceColor(transmittanceColor),
    m_enableEmission(enableEmission), m_emissiveIntensity(emissiveIntensity), m_emissiveColorConstant(emissiveColorConstant),
    m_isThinWalled(isThinWalled), m_thinWallThickness(thinWallThickness), m_useDiffuseLayer(useDiffuseLayer), m_samplerIndex(samplerIndex)
  {
    updateCachedData();
    updateCachedHash();
  }

  void writeGPUData(unsigned char* data, std::size_t& offset, uint32_t surfaceIndex) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;

    // For decode process, see surface_material.h
    // this data is accessed from uint16_t data[32], so data[n] refers to a pair of bytes.

    uint16_t flags = surfaceMaterialTypeTranslucent;

    // Note: Respect override flag here to let the GPU do less work in determining if the diffuse layer should be used or not.
    if (m_useDiffuseLayer || getEnableDiffuseLayerOverrideHack()) {
      flags |= TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_DIFFUSE_LAYER;
    }

    // data[0- 1]
    writeGPUHelper(data, offset, flags);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_cachedBaseReflectivity));
    // data[2 - 4]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_transmittanceColor.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_transmittanceColor.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_transmittanceColor.z));
    // data[5 - 9]
    writeGPUHelperExplicit<2>(data, offset, m_samplerIndex);
    writeGPUHelperExplicit<2>(data, offset, m_transmittanceTextureIndex);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_cachedTransmittanceMeasurementDistanceOrThickness));
    writeGPUHelperExplicit<2>(data, offset, m_normalTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_emissiveColorTextureIndex);

    // data[10]
    assert(m_cachedEmissiveIntensity <= FLOAT16_MAX);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_cachedEmissiveIntensity));

    // data[11]
    // Note: Ensure IoR falls in the range expected by the encoding/decoding logic for the GPU (this should also be
    // enforced in the MDL and relevant content pipeline to prevent this assert from being triggered).
    assert(m_refractiveIndex >= 1.0f && m_refractiveIndex <= 3.0f);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_refractiveIndex));

    // data[12-13]: sourceSurfaceMaterialIndex
    assert(surfaceIndex <= SURFACE_INDEX_MAX_VALUE && "Surface index exceeds SURFACE_INDEX_MAX_VALUE for TranslucentSurfaceMaterial");
    writeGPUHelperExplicit<4>(data, offset, static_cast<uint32_t>(surfaceIndex));

    // data[14-16]: emissiveColorConstant
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_emissiveColorConstant.z));
    
    // data[17 - 31]
    writeGPUPadding<30>(data, offset);

    assert(offset - oldOffset == kSurfaceMaterialGPUSize);
  }

  bool validate() const {
    const bool hasTexture = m_normalTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_transmittanceTextureIndex != kSurfaceMaterialInvalidTextureIndex ||
                            m_emissiveColorTextureIndex != kSurfaceMaterialInvalidTextureIndex;

    return !hasTexture || m_samplerIndex != kSurfaceMaterialInvalidTextureIndex;
  }

  bool operator==(const RtTranslucentSurfaceMaterial& r) const {
    return m_cachedHash == r.m_cachedHash;
  }

  XXH64_hash_t getHash() const {
    return m_cachedHash;
  }
private:
  void updateCachedHash() {
    static_assert(
      sizeof(*this) == 96,
      "add new member for hashing if needed: add a MEMBER into the struct + add a VALUE into the list-init"
    );
    struct HashStruct {
      uint32_t normalTextureIndex;
      uint32_t transmittanceTextureIndex;
      uint32_t emissiveColorTextureIndex;
      float refractiveIndex;
      Vector3 transmittanceColor;
      float transmittanceMeasurementDistance;
      uint32_t enableEmission;  // NOTE: uint32_t to avoid padding
      float emissiveIntensity;
      Vector3 emissiveColorConstant;
      uint32_t isThinWalled;    // NOTE: uint32_t to avoid padding
      float thinWallThickness;
      uint32_t useDiffuseLayer; // NOTE: uint32_t to avoid padding
      uint32_t samplerIndex;
      // NOTE: There must be NO padding between members, as the struct is used for hashing
    };
    static_assert(alignof(HashStruct) == 4 && sizeof(HashStruct) % 4 == 0);
    HashStruct hashData = HashStruct{
      m_normalTextureIndex,
      m_transmittanceTextureIndex,
      m_emissiveColorTextureIndex,
      m_refractiveIndex,
      m_transmittanceColor,
      m_transmittanceMeasurementDistance,
      m_enableEmission,
      m_emissiveIntensity,
      m_emissiveColorConstant,
      m_isThinWalled,
      m_thinWallThickness,
      m_useDiffuseLayer,
      m_samplerIndex,
    };
    m_cachedHash = XXH3_64bits(&hashData, sizeof(hashData));
  }

  void updateCachedData() {
    // Note: Based on the Fresnel Equations with the assumption of a vacuum (nearly air
    // as the surrounding medium always) and an IoR of always >=1 (implicitly ensured by encoding
    // logic assertions later):
    // https://en.wikipedia.org/wiki/Fresnel_equations#Special_cases
    const float x = (1.0f - m_refractiveIndex) / (1.0f + m_refractiveIndex);

    m_cachedBaseReflectivity = x * x;
    m_cachedTransmittanceMeasurementDistanceOrThickness =
      m_isThinWalled ? -m_thinWallThickness : m_transmittanceMeasurementDistance;

    // Note: Translucent material does not take an emissive radiance directly, so zeroing out the intensity works
    // fine as a way to disable it (in case a texture is in use).
    m_cachedEmissiveIntensity = std::min(m_enableEmission ? m_emissiveIntensity : 0.0f, FLOAT16_MAX);

    // Note: Ensure the transmittance measurement distance or thickness was encoded properly by ensuring
    // it is not 0. This is because we currently do not actually check the sign bit but just use a less than
    // comparison to check the sign bit as neither of these values should be 0 in valid materials.
    assert(m_cachedTransmittanceMeasurementDistanceOrThickness != 0.0f);
  }

  uint32_t m_normalTextureIndex;
  uint32_t m_transmittanceTextureIndex;
  uint32_t m_emissiveColorTextureIndex;
  uint32_t m_samplerIndex;

  float m_refractiveIndex;
  Vector3 m_transmittanceColor;
  float m_transmittanceMeasurementDistance;
  bool m_enableEmission;
  float m_emissiveIntensity;
  Vector3 m_emissiveColorConstant;
  bool m_isThinWalled;
  float m_thinWallThickness;
  bool m_useDiffuseLayer;

  XXH64_hash_t m_cachedHash;

  // Note: Cached values are not involved in the hash as they are derived from the input data
  float m_cachedBaseReflectivity;
  float m_cachedTransmittanceMeasurementDistanceOrThickness;
  float m_cachedEmissiveIntensity;
};

struct RtRayPortalSurfaceMaterial {
  RtRayPortalSurfaceMaterial(
    uint32_t maskTextureIndex, uint32_t maskTextureIndex2, uint8_t rayPortalIndex,
    float rotationSpeed, bool enableEmission, float emissiveIntensity, uint32_t samplerIndex, uint32_t samplerIndex2) :
    m_maskTextureIndex{ maskTextureIndex }, m_maskTextureIndex2 { maskTextureIndex2 }, m_rayPortalIndex{ rayPortalIndex },
    m_rotationSpeed { rotationSpeed }, m_enableEmission(enableEmission), m_emissiveIntensity(emissiveIntensity), m_samplerIndex(samplerIndex), m_samplerIndex2(samplerIndex2) {
    updateCachedHash();
  }

  void writeGPUData(unsigned char* data, std::size_t& offset) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;

    // For decode process, see surface_material.h
    // this data is accessed from uint16_t data[32], so data[n] refers to a pair of bytes.

    uint16_t flags = surfaceMaterialTypeRayPortal;
    // data[0]
    writeGPUHelper(data, offset, flags);

    // data[1]
    writeGPUHelper(data, offset, uint16_t(m_rayPortalIndex));

    // data[2 - 3]
    writeGPUHelperExplicit<2>(data, offset, m_maskTextureIndex);
    writeGPUHelperExplicit<2>(data, offset, m_maskTextureIndex2);

    // data[4 - 5]
    assert(m_rotationSpeed < FLOAT16_MAX);
    writeGPUHelper(data, offset, glm::packHalf1x16(m_rotationSpeed));
    float emissiveIntensity = m_enableEmission ? m_emissiveIntensity : 1.0f;
    writeGPUHelper(data, offset, glm::packHalf1x16(emissiveIntensity));

    // data[6 - 7]
    writeGPUHelperExplicit<2>(data, offset, m_samplerIndex);
    writeGPUHelperExplicit<2>(data, offset, m_samplerIndex2);

    // data[8 - 31]
    writeGPUPadding<48>(data, offset); // Note: Padding for unused space
    assert(offset - oldOffset == kSurfaceMaterialGPUSize);
  }

  bool validate() const {
    if (m_maskTextureIndex != kSurfaceMaterialInvalidTextureIndex && m_samplerIndex == kSurfaceMaterialInvalidTextureIndex) {
      return false;
    }

    if (m_maskTextureIndex2 != kSurfaceMaterialInvalidTextureIndex && m_samplerIndex2 == kSurfaceMaterialInvalidTextureIndex) {
      return false;
    }

    return true;
  }

  bool operator==(const RtRayPortalSurfaceMaterial& r) const {
    return m_cachedHash == r.m_cachedHash;
  }

  XXH64_hash_t getHash() const {
    return m_cachedHash;
  }

  uint32_t getMaskTextureIndex() const {
    return m_maskTextureIndex;
  }

  uint32_t getMaskTextureIndex2() const {
    return m_maskTextureIndex2;
  }

  uint32_t getSamplerIndex() const {
    return m_samplerIndex;
  }

  uint32_t getSamplerIndex2() const {
    return m_samplerIndex2;
  }

  uint8_t getRayPortalIndex() const {
    return m_rayPortalIndex;
  }

  float getRotationSpeed() const {
    return m_rotationSpeed;
  }

  bool getEnableEmission() const {
    return m_enableEmission;
  }

  float getEmissiveIntensity() const {
    return m_emissiveIntensity;
  }

private:
  void updateCachedHash() {
    static_assert(
      sizeof(*this) == 40,
      "add new member for hashing if needed: add a MEMBER into the struct + add a VALUE into the list-init"
    );
    struct HashStruct {
      uint32_t maskTextureIndex;
      uint32_t maskTextureIndex2;
      uint32_t rayPortalIndex;  // NOTE: uint32_t to avoid padding
      float rotationSpeed;
      uint32_t enableEmission;  // NOTE: uint32_t to avoid padding
      float emissiveIntensity;
      uint32_t samplerIndex;
      uint32_t samplerIndex2;
      // NOTE: There must be NO padding between members, as the struct is used for hashing
    };
    static_assert(alignof(HashStruct) == 4 && sizeof(HashStruct) % 4 == 0);
    HashStruct hashData = HashStruct{
      m_maskTextureIndex,
      m_maskTextureIndex2,
      m_rayPortalIndex,
      m_rotationSpeed,
      m_enableEmission,
      m_emissiveIntensity,
      m_samplerIndex,
      m_samplerIndex2,
    };
    m_cachedHash = XXH3_64bits(&hashData, sizeof(hashData));
  }

  uint32_t m_maskTextureIndex;
  uint32_t m_maskTextureIndex2;
  uint32_t m_samplerIndex;
  uint32_t m_samplerIndex2;

  uint8_t m_rayPortalIndex;
  float m_rotationSpeed;
  bool m_enableEmission;
  float m_emissiveIntensity;

  XXH64_hash_t m_cachedHash;
};

// Extension of the three basic types of materials.
// Don't use material types below standalone. Instead, attach them to the materials above as side load data.

// Subsurface Material
struct RtSubsurfaceMaterial {
  RtSubsurfaceMaterial(
    const uint32_t subsurfaceTransmittanceTextureIndex,
    const uint32_t subsurfaceThicknessTextureIndex,
    const uint32_t subsurfaceSingleScatteringAlbedoTextureIndex,
    const Vector3& subsurfaceTransmittanceColor,
    const float subsurfaceMeasurementDistance,
    const Vector3& subsurfaceSingleScatteringAlbedo,
    const float subsurfaceVolumetricAnisotropy,
    const float subsurfaceRadiusScale,
    const float subsurfaceMaxSampleRadius)
    :
    m_subsurfaceTransmittanceTextureIndex(subsurfaceTransmittanceTextureIndex),
    m_subsurfaceThicknessTextureIndex(subsurfaceThicknessTextureIndex),
    m_subsurfaceSingleScatteringAlbedoTextureIndex(subsurfaceSingleScatteringAlbedoTextureIndex),
    m_subsurfaceTransmittanceColor { subsurfaceTransmittanceColor },
    m_subsurfaceMeasurementDistance { subsurfaceMeasurementDistance },
    m_subsurfaceSingleScatteringAlbedo { subsurfaceSingleScatteringAlbedo },
    m_subsurfaceVolumetricAnisotropy { subsurfaceVolumetricAnisotropy },
    // Because we do log on the transmittance color when mapping to attenuation coefficient, we need to clamp to a small epsilon value to avoid NaN issue.
    m_subsurfaceVolumetricAttenuationCoefficient {
      Vector3(-log(std::max(subsurfaceTransmittanceColor.x, FLT_EPSILON)),
              -log(std::max(subsurfaceTransmittanceColor.y, FLT_EPSILON)),
              -log(std::max(subsurfaceTransmittanceColor.z, FLT_EPSILON))) / std::max(subsurfaceMeasurementDistance, FLT_EPSILON) },
    m_subsurfaceRadiusScale { subsurfaceRadiusScale },
    m_subsurfaceMaxSampleRadius { subsurfaceMaxSampleRadius }
  {
    updateCachedHash();
  }

  void writeGPUData(unsigned char* data, std::size_t& offset) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;

    // For decode process, see surface_material.h
    // this data is accessed from uint16_t data[32], so data[n] refers to a pair of bytes.

    // Write an empty flags to stay consistent with the other materials.
    uint16_t flags = 0;

    // data[0]
    writeGPUHelperExplicit<2>(data, offset, flags);

    // data[1]
    writeGPUHelperExplicit<2>(data, offset, m_subsurfaceTransmittanceTextureIndex);

    // data[2]
    writeGPUHelperExplicit<2>(data, offset, m_subsurfaceThicknessTextureIndex);

    // data[3]
    writeGPUHelperExplicit<2>(data, offset, m_subsurfaceSingleScatteringAlbedoTextureIndex);

    // data[4]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceVolumetricAnisotropy));

    // data[5-8]
    if (m_subsurfaceRadiusScale < 0.0f) { // Thin Opaque
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceVolumetricAttenuationCoefficient.x));
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceVolumetricAttenuationCoefficient.y));
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceVolumetricAttenuationCoefficient.z));

      assert(m_subsurfaceMeasurementDistance <= FLOAT16_MAX);
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceMeasurementDistance));
    } else { // SSS
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceTransmittanceColor.x));
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceTransmittanceColor.y));
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceTransmittanceColor.z));

      assert(m_subsurfaceRadiusScale <= FLOAT16_MAX);
      writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceRadiusScale));
    }

    // data[9-11]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceSingleScatteringAlbedo.x));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceSingleScatteringAlbedo.y));
    writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceSingleScatteringAlbedo.z));

    // data[12]
    writeGPUHelper(data, offset, glm::packHalf1x16(m_subsurfaceMaxSampleRadius));

    // data[13-31]
    writeGPUPadding<38>(data, offset);
  }

  bool operator==(const RtSubsurfaceMaterial& r) const {
    return m_cachedHash == r.m_cachedHash;
  }

  bool validate() const {
    return true;
  }

  XXH64_hash_t getHash() const {
    return m_cachedHash;
  }

  uint32_t getSubsurfaceTransmittanceTextureIndex() const {
    return m_subsurfaceTransmittanceTextureIndex;
  }

  uint32_t getSubsurfaceThicknessTextureIndex() const {
    return m_subsurfaceThicknessTextureIndex;
  }

  uint32_t getSubsurfaceSingleScatteringAlbedoTextureIndex() const {
    return m_subsurfaceSingleScatteringAlbedoTextureIndex;
  }

  float getSubsurfaceMeasurementDistance() const {
    return m_subsurfaceMeasurementDistance;
  }

  const Vector3& getSubsurfaceVolumetricScatteringAlbedo() const {
    return m_subsurfaceSingleScatteringAlbedo;
  }

  float getSubsurfaceVolumetricAnisotropy() const {
    return m_subsurfaceVolumetricAnisotropy;
  }

  const Vector3& getSubsurfaceVolumetricAttenuationCoefficient() const {
    return m_subsurfaceVolumetricAttenuationCoefficient;
  }

  float getSubsurfaceRadiusScale() const {
    return m_subsurfaceRadiusScale;
  }

  float getSubsurfaceMaxRadius() const {
    return m_subsurfaceMaxSampleRadius;
  }

private:

  void updateCachedHash() {
    static_assert(
      sizeof(*this) == 72,
      "add new member for hashing if needed: add a MEMBER into the struct + add a VALUE into the list-init"
    );
    struct HashStruct {
      uint32_t m_subsurfaceTransmittanceTextureIndex;
      uint32_t m_subsurfaceThicknessTextureIndex;
      uint32_t m_subsurfaceSingleScatteringAlbedoTextureIndex;
      Vector3 m_subsurfaceTransmittanceColor;
      float m_subsurfaceMeasurementDistance;
      Vector3 m_subsurfaceSingleScatteringAlbedo;
      float m_subsurfaceVolumetricAnisotropy;
      Vector3 m_subsurfaceVolumetricAttenuationCoefficient;
      float m_subsurfaceRadiusScale;
      float m_subsurfaceMaxSampleRadius;
      // NOTE: There must be NO padding between members, as the struct is used for hashing
    };
    static_assert(alignof(HashStruct) == 4 && sizeof(HashStruct) % 4 == 0);
    HashStruct hashData = HashStruct{
      m_subsurfaceTransmittanceTextureIndex,
      m_subsurfaceThicknessTextureIndex,
      m_subsurfaceSingleScatteringAlbedoTextureIndex,
      m_subsurfaceTransmittanceColor,
      m_subsurfaceMeasurementDistance,
      m_subsurfaceSingleScatteringAlbedo,
      m_subsurfaceVolumetricAnisotropy,
      m_subsurfaceVolumetricAttenuationCoefficient,
      m_subsurfaceRadiusScale,
      m_subsurfaceMaxSampleRadius,
    };
    m_cachedHash = XXH3_64bits(&hashData, sizeof(hashData));
  }

  // Thin Opaque Textures Index (Shared with SSS)
  uint32_t m_subsurfaceTransmittanceTextureIndex;
  uint32_t m_subsurfaceThicknessTextureIndex;
  uint32_t m_subsurfaceSingleScatteringAlbedoTextureIndex;

  // Thin Opaque Properties (Shared with SSS)
  Vector3 m_subsurfaceTransmittanceColor;
  float m_subsurfaceMeasurementDistance;
  Vector3 m_subsurfaceSingleScatteringAlbedo; // scatteringCoefficient / attenuationCoefficient
  float m_subsurfaceVolumetricAnisotropy;

  // Cache Volumetric Properties
  Vector3 m_subsurfaceVolumetricAttenuationCoefficient; // scatteringCoefficient + absorptionCoefficient
  // Currently no need to cache scattering and absorption coefficient for single scattering simulation

  // SSS properties using Diffusion Profile
  float m_subsurfaceRadiusScale;
  float m_subsurfaceMaxSampleRadius;

  XXH64_hash_t m_cachedHash;
};

struct RtSurfaceMaterial {
  RtSurfaceMaterial(const RtOpaqueSurfaceMaterial& opaqueSurfaceMaterial) :
    m_type{ RtSurfaceMaterialType::Opaque },
    m_opaqueSurfaceMaterial{ opaqueSurfaceMaterial } {}

  RtSurfaceMaterial(const RtTranslucentSurfaceMaterial& translucentSurfaceMaterial) :
    m_type{ RtSurfaceMaterialType::Translucent },
    m_translucentSurfaceMaterial{ translucentSurfaceMaterial } {}

  RtSurfaceMaterial(const RtRayPortalSurfaceMaterial& rayPortalSurfaceMaterial) :
    m_type{ RtSurfaceMaterialType::RayPortal },
    m_rayPortalSurfaceMaterial{ rayPortalSurfaceMaterial } {}

  RtSurfaceMaterial(const RtSubsurfaceMaterial& subsurfaceMaterial) :
    m_type { RtSurfaceMaterialType::Subsurface },
    m_subsurfaceMaterial { subsurfaceMaterial } {}

  RtSurfaceMaterial(const RtSurfaceMaterial& surfaceMaterial) :
    m_type{ surfaceMaterial.m_type } {
    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      new (&m_opaqueSurfaceMaterial) RtOpaqueSurfaceMaterial{ surfaceMaterial.m_opaqueSurfaceMaterial };
      break;
    case RtSurfaceMaterialType::Translucent:
      new (&m_translucentSurfaceMaterial) RtTranslucentSurfaceMaterial{ surfaceMaterial.m_translucentSurfaceMaterial };
      break;
    case RtSurfaceMaterialType::RayPortal:
      new (&m_rayPortalSurfaceMaterial) RtRayPortalSurfaceMaterial{ surfaceMaterial.m_rayPortalSurfaceMaterial };
      break;
    case RtSurfaceMaterialType::Subsurface:
      new (&m_subsurfaceMaterial) RtSubsurfaceMaterial { surfaceMaterial.m_subsurfaceMaterial };
      break;
    }
  }

  ~RtSurfaceMaterial() {
    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      m_opaqueSurfaceMaterial.~RtOpaqueSurfaceMaterial();
      break;
    case RtSurfaceMaterialType::Translucent:
      m_translucentSurfaceMaterial.~RtTranslucentSurfaceMaterial();
      break;
    case RtSurfaceMaterialType::RayPortal:
      m_rayPortalSurfaceMaterial.~RtRayPortalSurfaceMaterial();
      break;
    case RtSurfaceMaterialType::Subsurface:
      m_subsurfaceMaterial.~RtSubsurfaceMaterial();
      break;
    }
  }

  void writeGPUData(unsigned char* data, std::size_t& offset, uint32_t surfaceIndex) const {
    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      m_opaqueSurfaceMaterial.writeGPUData(data, offset);
      break;
    case RtSurfaceMaterialType::Translucent:
      m_translucentSurfaceMaterial.writeGPUData(data, offset, surfaceIndex);
      break;
    case RtSurfaceMaterialType::RayPortal:
      m_rayPortalSurfaceMaterial.writeGPUData(data, offset);
      break;
    case RtSurfaceMaterialType::Subsurface:
      m_subsurfaceMaterial.writeGPUData(data, offset);
      break;
    }
  }

  bool validate() const {
    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      return m_opaqueSurfaceMaterial.validate();
    case RtSurfaceMaterialType::Translucent:
      return m_translucentSurfaceMaterial.validate();
    case RtSurfaceMaterialType::RayPortal:
      return m_rayPortalSurfaceMaterial.validate();
    case RtSurfaceMaterialType::Subsurface:
      return m_subsurfaceMaterial.validate();
    }

    return false;
  }

  RtSurfaceMaterial& operator=(const RtSurfaceMaterial& rtSurfaceMaterial) {
    if (this != &rtSurfaceMaterial) {
      m_type = rtSurfaceMaterial.m_type;

      switch (rtSurfaceMaterial.m_type) {
      default:
        assert(false);

        [[fallthrough]];
      case RtSurfaceMaterialType::Opaque:
        m_opaqueSurfaceMaterial = rtSurfaceMaterial.m_opaqueSurfaceMaterial;
        break;
      case RtSurfaceMaterialType::Translucent:
        m_translucentSurfaceMaterial = rtSurfaceMaterial.m_translucentSurfaceMaterial;
        break;
      case RtSurfaceMaterialType::RayPortal:
        m_rayPortalSurfaceMaterial = rtSurfaceMaterial.m_rayPortalSurfaceMaterial;
        break;
      case RtSurfaceMaterialType::Subsurface:
        m_subsurfaceMaterial = rtSurfaceMaterial.m_subsurfaceMaterial;
        break;
      }
    }

    return *this;
  }

  bool operator==(const RtSurfaceMaterial& rhs) const {
    // Note: Different Surface Material types are not the same Surface Material so comparison can return false
    if (m_type != rhs.m_type) {
      return false;
    }

    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      return m_opaqueSurfaceMaterial == rhs.m_opaqueSurfaceMaterial;
    case RtSurfaceMaterialType::Translucent:
      return m_translucentSurfaceMaterial == rhs.m_translucentSurfaceMaterial;
    case RtSurfaceMaterialType::RayPortal:
      return m_rayPortalSurfaceMaterial == rhs.m_rayPortalSurfaceMaterial;
    case RtSurfaceMaterialType::Subsurface:
      return m_subsurfaceMaterial == rhs.m_subsurfaceMaterial;
    }
  }

  XXH64_hash_t getHash() const {
    switch (m_type) {
    default:
      assert(false);

      [[fallthrough]];
    case RtSurfaceMaterialType::Opaque:
      return m_opaqueSurfaceMaterial.getHash();
    case RtSurfaceMaterialType::Translucent:
      return m_translucentSurfaceMaterial.getHash();
    case RtSurfaceMaterialType::RayPortal:
      return m_rayPortalSurfaceMaterial.getHash();
    case RtSurfaceMaterialType::Subsurface:
      return m_subsurfaceMaterial.getHash();
    }
  }

  RtSurfaceMaterialType getType() const {
    return m_type;
  }

  const RtOpaqueSurfaceMaterial& getOpaqueSurfaceMaterial() const {
    assert(m_type == RtSurfaceMaterialType::Opaque);

    return m_opaqueSurfaceMaterial;
  }

  const RtTranslucentSurfaceMaterial& getTranslucentSurfaceMaterial() const {
    assert(m_type == RtSurfaceMaterialType::Translucent);

    return m_translucentSurfaceMaterial;
  }

  const RtRayPortalSurfaceMaterial& getRayPortalSurfaceMaterial() const {
    assert(m_type == RtSurfaceMaterialType::RayPortal);

    return m_rayPortalSurfaceMaterial;
  }
private:
  // Type-specific Surface Material Information

  RtSurfaceMaterialType m_type;
  union {
    RtOpaqueSurfaceMaterial m_opaqueSurfaceMaterial;
    RtTranslucentSurfaceMaterial m_translucentSurfaceMaterial;
    RtRayPortalSurfaceMaterial m_rayPortalSurfaceMaterial;
    RtSubsurfaceMaterial m_subsurfaceMaterial;
  };
};

// Volume Materials

// Todo: Compute size directly from sizeof of GPU structure (by including it), for now computed by sum of members manually
constexpr std::size_t kVolumeMaterialGPUSize = 4;

struct RtVolumeMaterial
{
  RtVolumeMaterial() {
    updateCachedHash();
  }

  void writeGPUData(unsigned char* data, std::size_t& offset) const {
    [[maybe_unused]] const std::size_t oldOffset = offset;

    writeGPUPadding<4>(data, offset);

    assert(offset - oldOffset == kVolumeMaterialGPUSize);
  }

  bool operator==(const RtVolumeMaterial& r) const {
    assert(false);

    return m_cachedHash == r.m_cachedHash;
  }

  XXH64_hash_t getHash() const {
    assert(false);

    return m_cachedHash;
  }
private:
  void updateCachedHash() {
    const XXH64_hash_t h = 0;

    m_cachedHash = h;
  }

  XXH64_hash_t m_cachedHash;
};

enum class MaterialDataType {
  Opaque,
  Translucent,
  RayPortal,
  Count,
  Invalid
};

// Note: For use with legacy material information
struct LegacyMaterialData {
  static OpaqueMaterialData createDefault();

  LegacyMaterialData()
  { }

  LegacyMaterialData(const TextureRef& colorTexture, const TextureRef& colorTexture2, const RtxMaterial material)
    : material{ material }
  {
    assert(!colorTexture.isImageEmpty());
    // Hash is NOT computed here — caller must invoke updateCachedHash()
    // after populating blend/alpha state, since the D3D11 hash includes
    // those fields.
  }

  const XXH64_hash_t getHash() const {
    return m_cachedHash;
  }

  const TextureRef& getColorTexture() const {
    return colorTextures[0];
  }

  const TextureRef& getColorTexture2() const {
    return colorTextures[1];
  }

  const Rc<DxvkSampler>& getSampler() const {
    return samplers[0];
  }

  const Rc<DxvkSampler>& getSampler2() const {
    return samplers[1];
  }

  const RtxMaterial& getLegacyMaterial() const {
    return material;
  }

  inline const bool usesTexture() const {
    return ((getColorTexture().isValid()  && !getColorTexture().isImageEmpty()) ||
            (getColorTexture2().isValid() && !getColorTexture2().isImageEmpty()));
  }

  // A single place to define and handle conversions between legacy and raytraced materials
  template<typename T>
  T as() const;

  const void printDebugInfo(const char* name = "") const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "LegacyMaterialData ", name, "\n",
      "  address: ", this, "\n",
      "  alphaTestEnabled: ", alphaTestEnabled, "\n",
      "  alphaTestReferenceValue: ", alphaTestReferenceValue, "\n",
      "  alphaTestCompareOp: ", alphaTestCompareOp, "\n",
      "  alphaBlendEnabled: ", blendMode.enableBlending, "\n",
      "  colorSrcFactor: ", blendMode.colorSrcFactor, "\n",
      "  colorDstFactor: ", blendMode.colorDstFactor, "\n",
      "  colorBlendOp: ", blendMode.colorBlendOp, "\n",
      "  alphaSrcFactor: ", blendMode.alphaSrcFactor, "\n",
      "  alphaDstFactor: ", blendMode.alphaDstFactor, "\n",
      "  alphaBlendOp: ", blendMode.alphaBlendOp, "\n",
      "  writeMask: ", blendMode.writeMask, "\n",
      "  textureColorArg1Source: ", static_cast<int>(textureColorArg1Source), "\n",
      "  textureColorArg2Source: ", static_cast<int>(textureColorArg2Source), "\n",
      "  textureColorOperation: ", static_cast<int>(textureColorOperation), "\n",
      "  textureAlphaArg1Source: ", static_cast<int>(textureAlphaArg1Source), "\n",
      "  textureAlphaArg2Source: ", static_cast<int>(textureAlphaArg2Source), "\n",
      "  textureAlphaOperation: ", static_cast<int>(textureAlphaOperation), "\n",
      "  tFactor: ", tFactor, "\n",
      // "  material.Diffuse: ", material.Diffuse, "\n",
      // "  material.Ambient: ", material.Ambient, "\n",
      // "  material.Specular: ", material.Specular, "\n",
      // "  material.Emissive: ", material.Emissive, "\n",
      // "  material.Power: ", material.Power, "\n",
      std::hex, "  m_colorTexture: 0x", colorTextures[0].getImageHash(), "\n",
      "  m_colorTexture2: 0x", colorTextures[1].getImageHash(), "\n",
      "  m_cachedHash: 0x", m_cachedHash, std::dec));
#endif
  }

  uint32_t getColorTextureSlot(uint32_t slot) const {
    return colorTextureSlot[slot];
  }

  bool alphaTestEnabled = false;
  uint8_t alphaTestReferenceValue = 0;
  VkCompareOp alphaTestCompareOp = VkCompareOp::VK_COMPARE_OP_ALWAYS;

  DxvkBlendMode blendMode;

  RtTextureArgSource diffuseColorSource= RtTextureArgSource::None;
  RtTextureArgSource specularColorSource = RtTextureArgSource::None;
  RtTextureArgSource textureColorArg1Source = RtTextureArgSource::Texture;
  RtTextureArgSource textureColorArg2Source = RtTextureArgSource::None;
  DxvkRtTextureOperation textureColorOperation = DxvkRtTextureOperation::Modulate;
  RtTextureArgSource textureAlphaArg1Source = RtTextureArgSource::Texture;
  RtTextureArgSource textureAlphaArg2Source = RtTextureArgSource::None;
  DxvkRtTextureOperation textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
  uint32_t tFactor = 0xffffffff;  // Default: opaque white
  RtxMaterial material = {};
  bool isTextureFactorBlend = false;
  bool isVertexColorBakedLighting = true;

  void setHashOverride(XXH64_hash_t hash) {
    m_cachedHash = hash;
  }

private:
  friend class RtxContext;
  friend class D3D11Rtx;
  friend class TerrainBaker;
  friend class SceneManager;
  friend class GameCapturer;
  // NV-DXVK: InstanceManager reads sourcePsWritesCoverageMask
  // directly off the legacy material in processSceneObject — same
  // side-channel pattern D3D11Rtx uses to set it. Added because the
  // SV_Coverage hide gate fires AFTER the OpaqueMaterialData
  // routing (no public getter exists yet); rather than wire a
  // dedicated path through OpaqueMaterialData for one bool, just
  // grant InstanceManager friend access.
  friend class InstanceManager;
  friend struct RemixAPIPrivateAccessor;

  void updateCachedHash() {
    // D3D11 material hash — incorporates both texture hashes, blend state,
    // and alpha test to uniquely identify materials.  In D3D11 different
    // draw calls can bind the same albedo SRV with different blend/alpha
    // state, producing visually distinct materials that need distinct hashes.
    struct HashData {
      XXH64_hash_t tex0Hash;
      XXH64_hash_t tex1Hash;
      // NV-DXVK: include PBR-map hashes so a draw that shares an albedo with
      // another draw but binds a different normal/rough/metallic/emissive set
      // gets a unique material identity (matters for USD replacement keys).
      XXH64_hash_t normalHash;
      XXH64_hash_t roughnessHash;
      XXH64_hash_t metallicHash;
      XXH64_hash_t emissiveHash;
      XXH64_hash_t ambientOcclusionHash;
      XXH64_hash_t lightmapHash;
      XXH64_hash_t lightmap2Hash;
      XXH64_hash_t detailHash;
      XXH64_hash_t cloudMaskHash;
      uint32_t     blendEnable;
      uint32_t     colorSrc;
      uint32_t     colorDst;
      uint32_t     colorOp;
      uint32_t     alphaSrc;
      uint32_t     alphaDst;
      uint32_t     alphaOp;
      uint32_t     writeMask;
      uint32_t     alphaTestEnabled;
      uint32_t     alphaTestRef;
      uint32_t     alphaTestCmp;
    };
    static_assert(alignof(HashData) <= 8 && sizeof(HashData) % 4 == 0);

    auto safeHash = [](const TextureRef& t) -> XXH64_hash_t {
      return t.isImageEmpty() ? XXH64_hash_t(0) : t.getImageHash();
    };
    HashData hd = {
      colorTextures[0].getImageHash(),
      safeHash(colorTextures[1]),
      safeHash(normalTexture),
      safeHash(roughnessTexture),
      safeHash(metallicTexture),
      safeHash(emissiveTexture),
      safeHash(ambientOcclusionTexture),
      safeHash(lightmapTexture),
      safeHash(lightmap2Texture),
      safeHash(detailTexture),
      safeHash(cloudMaskTexture),
      blendMode.enableBlending,
      static_cast<uint32_t>(blendMode.colorSrcFactor),
      static_cast<uint32_t>(blendMode.colorDstFactor),
      static_cast<uint32_t>(blendMode.colorBlendOp),
      static_cast<uint32_t>(blendMode.alphaSrcFactor),
      static_cast<uint32_t>(blendMode.alphaDstFactor),
      static_cast<uint32_t>(blendMode.alphaBlendOp),
      blendMode.writeMask,
      alphaTestEnabled ? 1u : 0u,
      static_cast<uint32_t>(alphaTestReferenceValue),
      static_cast<uint32_t>(alphaTestCompareOp),
    };
    m_cachedHash = XXH3_64bits(&hd, sizeof(hd));
  }

  const static uint32_t kMaxSupportedTextures = 2;
  TextureRef colorTextures[kMaxSupportedTextures] = {};
  Rc<DxvkSampler> samplers[kMaxSupportedTextures] = {};
  static_assert(kInvalidResourceSlot == 0 && "Below initialization of all array members is only valid for a value of 0.");
  uint32_t colorTextureSlot[kMaxSupportedTextures] = { kInvalidResourceSlot };

  // NV-DXVK: PBR-map slots populated by D3D11Rtx::FillMaterialData when the
  // pixel-shader RDEF names its SRVs (TF2 uses albedoTexture / normalTexture /
  // glossTexture / specTexture / emissiveTexture at t0..t4). Forwarded into
  // OpaqueMaterialData by as<OpaqueMaterialData>() so the raytracer gets the
  // full material stack rather than just the albedo.
  TextureRef      normalTexture;
  TextureRef      roughnessTexture;   // note: may hold a "gloss" map (inverse)
  TextureRef      metallicTexture;    // may hold a "spec intensity" map
  TextureRef      emissiveTexture;
  // NV-DXVK: cavity / baked-AO map. Sampled at the same UV as albedo and
  // multiplied into albedo in the shader, independent of normal. Covers the
  // TF2 BSP case (FS_ac8c6ae6 uses t12 cavityTexture as `albedo *= cavity.r`)
  // and any other Source/Respawn material that names a cavityTexture or AO.
  TextureRef      ambientOcclusionTexture;
  // NV-DXVK: TF2 auxiliary BSP / world textures — lightmap (baked static GI),
  // lightmap2 (bumped / HDR range-extension), detail (fine-scale brick
  // variation), cloudMask (large-scale tonal banding). Sampled in the
  // shader and composited into albedo/emissive.
  TextureRef      lightmapTexture;
  TextureRef      lightmap2Texture;
  TextureRef      detailTexture;
  TextureRef      cloudMaskTexture;
  Rc<DxvkSampler> normalSampler;
  Rc<DxvkSampler> roughnessSampler;
  Rc<DxvkSampler> metallicSampler;
  Rc<DxvkSampler> emissiveSampler;
  Rc<DxvkSampler> ambientOcclusionSampler;
  Rc<DxvkSampler> lightmapSampler;
  Rc<DxvkSampler> lightmap2Sampler;
  Rc<DxvkSampler> detailSampler;
  Rc<DxvkSampler> cloudMaskSampler;

  // NV-DXVK: ground-truth emissive intent sourced from the PS's CBuffer at
  // FillMaterialData time. Source/TF2 Phong-style PSes carry per-pixel
  // emission via `CBufUberStatic.c_emissiveTint` (vec3 @ off 160) tinted
  // over `emissiveTexture` (slot t4), gated by D3D_SVF_USED. Reading this
  // replaces the legacy blend-state heuristic in rtx_instance_manager.cpp
  // (which mis-promoted refract/water layers to light sources).
  //
  // Populated only when the PS's RDEF declares the field AND marks it
  // used. Empty for non-Source PSes — the consumer in
  // LegacyMaterialData::as<OpaqueMaterialData>() falls back to no emission.
  bool    sourceUsesEmission = false;            // c_emissiveTint marked used
  bool    sourceAlphaModulatesEmissive = false;  // c_useAlphaModulateEmissive marked used AND non-zero
  Vector3 sourceEmissiveTint = Vector3(0.f);     // per-draw value of c_emissiveTint
  // NV-DXVK: TF2 worldspace VGUI/HUD shader marker — set in FillMaterial
  // Data when the PS RDEF declares the VGUI atlas resource trio (font
  // Texture + g_fontBounds + g_imgBounds). The consumer in
  // LegacyMaterialData::as<OpaqueMaterialData>() forwards the picked color
  // texture as emissive AND flips OpaqueMaterialData::IsUnlitOutput so
  // the surface ultimately gets isMatte=true on the GPU side, yielding
  // the "draw the UI texture as-is, no lighting" behaviour the original PS
  // produces.
  bool    sourceIsUnlitUI = false;

  // NV-DXVK: TF2 3D-skybox cloud billboard marker. Set in FillMaterialData
  // when the PS reads CBufUberStatic.c_fogColorFactor AND the draw uses a
  // premultiplied-OVER blend (the cloud-billboard signature). Forwarded by
  // LegacyMaterialData::as<OpaqueMaterialData>() to OpaqueMaterialData's
  // Tf2SkyboxFog, then to the OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG
  // GPU flag so the opaque surface shader reconstructs the fog blend.
  bool    sourceTf2FogCapable = false;

  // NV-DXVK: premultiplied-alpha-blend marker. Set in FillMaterialData when
  // the source D3D11 draw uses (BlendEnable=1, SrcBlend=ONE,
  // DestBlend=INV_SRC_ALPHA, BlendOp=ADD) — the unambiguous signature for
  // a texture authored with premultiplied alpha (.rgb is already
  // color * alpha). Forwarded to OpaqueMaterialData::AlbedoIsPremultiplied
  // → OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED so the slang
  // shader skips the opacity-multiply inside albedoToAdjustedAlbedo /
  // calcBaseReflectivity (avoids the double-darkening that produces
  // noisy dark dots on TF2 cloud billboards).
  bool    sourceAlbedoIsPremultiplied = false;

  // NV-DXVK: "PS writes SV_Coverage" marker. Set in FillMaterialData
  // when the bound PS's OSGN declares an output with systemValueType
  // == D3D_NAME_COVERAGE (oMask). Those shaders fake smooth alpha via
  // MSAA programmable sample-masking, which is meaningless in a path
  // tracer — at ray time oMask is ignored and the full RGBA writes to
  // every pixel, producing visible BOXY hard-edged corruption (TF2
  // 3D-skybox star-noise overlay = the visible sky speckling). The
  // path tracer has no way to reconstruct the rasterizer's per-sample
  // masking, so we hide the surface. Forwarded to instance manager
  // which sets m_isHidden=true → instance mask 0 → rays pass through.
  bool    sourcePsWritesCoverageMask = false;

  // NV-DXVK: "truly opaque, alpha channel is not load-bearing" marker.
  // Set in FillMaterialData when the source D3D11 draw has blend
  // disabled (BlendEnable=0) AND no alpha test was detected (neither
  // AlphaToCoverage nor a PS clip()/discard against c_alphaTestRef).
  // For such surfaces the PS either ignores the texture's alpha channel
  // entirely (verified via SPV walker for FS_44db6ff9 = the
  // 0x2a729 mountain VS — it samples t0.xyz only, never .w, and outputs
  // o0.w = 1.0 hard-coded) OR uses alpha only for fixed-function ops
  // that don't apply here. Either way, the alpha sample is leaking
  // into Remix's `opacity` field and `albedoToAdjustedAlbedo` then
  // darkens encoded albedo by `× opacity` — the residual DriftStage
  // Adjusted drift (~33k px on 0x2a729) the previous-handoff fix
  // didn't address. Forwarded to OpaqueMaterialData::IgnoreAlphaChannel
  // → OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL so the slang
  // shader forces opacity = 1 before the encode.
  bool    sourceForceIgnoreAlphaChannel = false;

  // NV-DXVK: TF2 viewmodel "screen-space scrolling overlay" emissive marker.
  // Set in FillMaterialData when the PS RDEF declares the screen-space
  // emissive pattern signature (reads c_uv1RotScaleX/Y + c_uv1Translate AND
  // has emissiveTexture bound AND has an emissiveMultiplyTexture slot bound).
  // Phase-1: detection only. Captured params live on this struct so a
  // future LegacyMaterialData::as<OpaqueMaterialData>() patch can plumb them
  // through to the GPU material once kSurfaceMaterialGPUSize grows. Source
  // PS asm at C:/Users/Friss/Downloads/.../rtx-remix/logs/ps_7836c1dd4d5c885f.asm
  // lines 280-296.
  bool    hasScreenSpaceEmissive = false;
  Vector2 screenSpaceEmissiveUv1RotScaleX = Vector2(1.f, 0.f);
  Vector2 screenSpaceEmissiveUv1RotScaleY = Vector2(0.f, 1.f);
  Vector2 screenSpaceEmissiveUv1Translate = Vector2(0.f, 0.f);
  uint16_t screenSpaceEmissiveMaskTextureSlot = 0xffffu;
  // The actual TextureRef for the t17 emissiveMultiplyTexture mask, populated
  // by FillMaterialData when the pattern matches AND the slot binds a real
  // 2D image. Empty for the maskless variant — the slang shader treats
  // missing mask as white (no-op multiply).
  TextureRef screenSpaceEmissiveMaskTexture;

  XXH64_hash_t m_cachedHash = kEmptyHash;
};

struct MaterialData {
  bool m_ignored = false;

  using MaterialVariant = std::variant<
    OpaqueMaterialData,
    TranslucentMaterialData,
    RayPortalMaterialData
  >;

  // Using variants rather than a union here, due to the MaterialData containing nested members of Rc pointers.
  MaterialVariant m_data;

  std::optional<RtxParticleSystemDesc> m_particleSystem;

  // Verify that the variant and enum stay in sync
  static_assert(std::variant_size_v<MaterialVariant> == (size_t)MaterialDataType::Count, "Enum is out of sync, please check your change.");
  static_assert(std::is_same_v<std::variant_alternative_t<(size_t)MaterialDataType::Opaque,      MaterialVariant>, OpaqueMaterialData>,      "MaterialVariant[Opaque] must be OpaqueMaterialData, please check your change.");
  static_assert(std::is_same_v<std::variant_alternative_t<(size_t)MaterialDataType::Translucent, MaterialVariant>, TranslucentMaterialData>, "MaterialVariant[Translucent] must be TranslucentMaterialData, please check your change.");
  static_assert(std::is_same_v<std::variant_alternative_t<(size_t)MaterialDataType::RayPortal,   MaterialVariant>, RayPortalMaterialData>,   "MaterialVariant[RayPortal] must be RayPortalMaterialData, please check your change.");

  MaterialData(const OpaqueMaterialData& opaque, std::optional<RtxParticleSystemDesc> particleSystem = std::nullopt, bool ignored = false)
    : m_ignored { ignored }, m_data { opaque }, m_particleSystem { particleSystem } {}

  MaterialData(const TranslucentMaterialData& translucent, std::optional<RtxParticleSystemDesc> particleSystem = std::nullopt, bool ignored = false)
    : m_ignored { ignored }, m_data { translucent }, m_particleSystem { particleSystem } {}

  MaterialData(const RayPortalMaterialData& portal, std::optional<RtxParticleSystemDesc> particleSystem = std::nullopt)
    : m_data { portal }, m_particleSystem { particleSystem } { }

  bool getIgnored() const {
    return m_ignored;
  }

  MaterialDataType getType() const {
    // NOTE: relies on the variant index matching MaterialDataType
    return static_cast<MaterialDataType>(m_data.index());
  }

  XXH64_hash_t getHash() const {
    return std::visit([](auto const& mat) { return mat.getHash(); }, m_data);
  }

  const Rc<DxvkSampler>& getSamplerOverride() const {
    return std::visit([](auto const& mat) -> const Rc<DxvkSampler>& { return mat.getSamplerOverride(); }, m_data);
  }

  const OpaqueMaterialData& getOpaqueMaterialData() const {
    assert(std::holds_alternative<OpaqueMaterialData>(m_data));
    return std::get<OpaqueMaterialData>(m_data);
  }

  OpaqueMaterialData& getOpaqueMaterialData() {
    assert(std::holds_alternative<OpaqueMaterialData>(m_data));
    return std::get<OpaqueMaterialData>(m_data);
  }

  const TranslucentMaterialData& getTranslucentMaterialData() const {
    assert(std::holds_alternative<TranslucentMaterialData>(m_data));
    return std::get<TranslucentMaterialData>(m_data);
  }

  TranslucentMaterialData& getTranslucentMaterialData() {
    assert(std::holds_alternative<TranslucentMaterialData>(m_data));
    return std::get<TranslucentMaterialData>(m_data);
  }

  const RayPortalMaterialData& getRayPortalMaterialData() const {
    assert(std::holds_alternative<RayPortalMaterialData>(m_data));
    return std::get<RayPortalMaterialData>(m_data);
  }

  RayPortalMaterialData& getRayPortalMaterialData() {
    assert(std::holds_alternative<RayPortalMaterialData>(m_data));
    return std::get<RayPortalMaterialData>(m_data);
  }

  const RtxParticleSystemDesc* getParticleSystemDesc() const {
    return m_particleSystem.has_value() ? &m_particleSystem.value() : nullptr;
  }
  
  void getSpriteSheetData(uint8_t& spriteSheetRows, uint8_t& spriteSheetCols, uint8_t& spriteSheetFPS) const {
    // Note: Extract spritesheet information from the associated material data as it ends up stored in the Surface
    // not in the Surface Material like most material information.
    switch (getType()) {
    case MaterialDataType::Opaque:
      spriteSheetRows = getOpaqueMaterialData().getSpriteSheetRows();
      spriteSheetCols = getOpaqueMaterialData().getSpriteSheetCols();
      spriteSheetFPS = getOpaqueMaterialData().getSpriteSheetFPS();

      break;
    case MaterialDataType::Translucent:
      spriteSheetRows = getTranslucentMaterialData().getSpriteSheetRows();
      spriteSheetCols = getTranslucentMaterialData().getSpriteSheetCols();
      spriteSheetFPS = getTranslucentMaterialData().getSpriteSheetFPS();

      break;
    case MaterialDataType::RayPortal:
      spriteSheetRows = getRayPortalMaterialData().getSpriteSheetRows();
      spriteSheetCols = getRayPortalMaterialData().getSpriteSheetCols();
      spriteSheetFPS = getRayPortalMaterialData().getSpriteSheetFPS();

      break;
    case MaterialDataType::Count:
    case MaterialDataType::Invalid:
      assert(0);
      break;
    }
  }
  
  void setSpriteSheetData(uint8_t spriteSheetRows, uint8_t spriteSheetCols, uint8_t spriteSheetFPS) {
    switch (getType()) {
    case MaterialDataType::Opaque:
       getOpaqueMaterialData().setSpriteSheetRows(spriteSheetRows);
       getOpaqueMaterialData().setSpriteSheetCols(spriteSheetCols);
       getOpaqueMaterialData().setSpriteSheetFPS(spriteSheetFPS);

      break;
    case MaterialDataType::Translucent:
      getTranslucentMaterialData().setSpriteSheetRows(spriteSheetRows);
      getTranslucentMaterialData().setSpriteSheetCols(spriteSheetCols);
      getTranslucentMaterialData().setSpriteSheetFPS(spriteSheetFPS);

      break;
    case MaterialDataType::RayPortal:
      getRayPortalMaterialData().setSpriteSheetRows(spriteSheetRows);
      getRayPortalMaterialData().setSpriteSheetCols(spriteSheetCols);
      getRayPortalMaterialData().setSpriteSheetFPS(spriteSheetFPS);

      break;
    case MaterialDataType::Count:
    case MaterialDataType::Invalid:
      assert(0);
      break;
    }
  }

  void mergeLegacyMaterial(const LegacyMaterialData& input) {
    std::visit([&](auto& mat) {
      using T = std::decay_t<decltype(mat)>;
      if constexpr (std::is_same_v<T, OpaqueMaterialData>) {
        OpaqueMaterialData tmp;
        tmp.getAlbedoOpacityTexture() = input.getColorTexture();
        if (auto s = input.getSampler().ptr()) {
          tmp.getFilterMode() = lss::Mdl::Filter::vkToMdl(s->info().magFilter);
          tmp.getWrapModeU() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeU);
          tmp.getWrapModeV() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeV);
        }
        mat.merge(tmp);
      } else if constexpr (std::is_same_v<T, TranslucentMaterialData>) {
        TranslucentMaterialData tmp;
        if (auto s = input.getSampler().ptr()) {
          tmp.getFilterMode() = lss::Mdl::Filter::vkToMdl(s->info().magFilter);
          tmp.getWrapModeU() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeU);
          tmp.getWrapModeV() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeV);
        }
        mat.merge(tmp);
      } else { 
        RayPortalMaterialData tmp;
        tmp.getMaskTexture() = input.getColorTexture();
        tmp.getMaskTexture2() = input.getColorTexture2();
        if (auto s = input.getSampler().ptr()) {
          tmp.getFilterMode() = lss::Mdl::Filter::vkToMdl(s->info().magFilter);
          tmp.getWrapModeU() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeU);
          tmp.getWrapModeV() = lss::Mdl::WrapMode::vkToMdl(s->info().addressModeV);
        }
        mat.merge(tmp);
      }
    }, m_data);
  }

#define POPULATE_SAMPLER_INFO(info, material) \
  info.magFilter = \
    lss::Mdl::Filter::mdlToVk(material.getFilterMode()); \
  info.minFilter = \
    lss::Mdl::Filter::mdlToVk(material.getFilterMode()); \
  info.addressModeU = \
    lss::Mdl::WrapMode::mdlToVk(material.getWrapModeU(), &info.borderColor); \
  info.addressModeV = \
    lss::Mdl::WrapMode::mdlToVk(material.getWrapModeV(), &info.borderColor);

  void populateSamplerInfo(DxvkSamplerCreateInfo& toPopulate) const {
    std::visit([&](auto const& mat) { POPULATE_SAMPLER_INFO(toPopulate, mat); }, m_data);
  }
#undef POPULATE_SAMPLER_INFO
};

} // namespace dxvk
