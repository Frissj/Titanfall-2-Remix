/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "../utility/packing_helpers.h"

// This function can be executed on the CPU or GPU!!
#ifdef __cplusplus
#define asfloat(x) *reinterpret_cast<const float*>(&x)
#define asuint(x) *reinterpret_cast<const uint32_t*>(&x)
#define WriteBuffer(T) T*
#define ReadBuffer(T) const T*

// CPU-side half-float to float conversion (GPU uses the f16tof32 intrinsic built-in)
#include "../utility/f16_conversion.h"

#else
#define WriteBuffer(T) RWStructuredBuffer<T>
#define ReadBuffer(T) StructuredBuffer<T>
#endif

namespace interleaver {

  enum SupportedVkFormats : uint32_t {
    VK_FORMAT_R8G8B8A8_UNORM = 37,
    VK_FORMAT_A2B10G10R10_SNORM_PACK32 = 65,

    // Passthrough format mapping
    VK_FORMAT_B8G8R8A8_UNORM = 44,
    VK_FORMAT_R16G16_SFLOAT = 83,
    // NV-DXVK: R16G16B16A16_SFLOAT (97) — four half-float components.
    // Source-engine games (Titanfall 2) use this for positions in some
    // vertex layouts (half-precision world-space coordinates).  Without
    // interleaver support these draws produce garbage BLAS entries that
    // cause GPU hangs (TDR / VK_ERROR_DEVICE_LOST).
    VK_FORMAT_R16G16B16A16_SFLOAT = 97,
    // NV-DXVK: R16G16_UINT (81) — Source/TF2 lightmap UV format. Two uint16
    // values packed into one uint32; the VS does utof + *1/65535. Only used
    // by convertLightmapTexcoord (TC1 path); never by position/TC0.
    VK_FORMAT_R16G16_UINT = 81,
    // NV-DXVK: R32G32_UINT (101) — Source Engine 2 (Titanfall 2) compressed
    // vertex positions.  Two uint32 values packing four fp16 components:
    //   uint0 = [half_y(31:16) | half_x(15:0)]
    //   uint1 = [half_w(31:16) | half_z(15:0)]
    // Decoded identically to R16G16B16A16_SFLOAT but declared as UINT in the
    // input layout because the vertex shader performs manual bit-extraction.
    VK_FORMAT_R32G32_UINT = 101,
    // NV-DXVK: R32_UINT (98) — TF2 lit-pass NORMAL semantic. Single 32-bit
    // word holds an axis-dominant compressed normal. Decode lives in
    // convertNormal(); generic convert() returns the default float3(1,1,1)
    // for this format because it has no defined non-normal interpretation
    // (position uses R32G32_UINT at format 101, not 98).
    VK_FORMAT_R32_UINT = 98,
    VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32_SFLOAT = 106,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,
  };

  bool formatConversionFloatSupported(uint32_t format) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_R16G16_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R16G16_UINT:
    case SupportedVkFormats::VK_FORMAT_R16G16B16A16_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32_UINT:
    case SupportedVkFormats::VK_FORMAT_R32G32_UINT:
    case SupportedVkFormats::VK_FORMAT_R32G32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32A32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
    case SupportedVkFormats::VK_FORMAT_A2B10G10R10_SNORM_PACK32:
      return true;
    default:
      return false;
    }
  }

  bool formatConversionUintSupported(uint32_t format) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_B8G8R8A8_UNORM:
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
      return true;
    default:
      return false;
    }
  }

  // Forward declaration — convertTexcoord (below) falls back to convert()
  // for non-texcoord-specific formats. Slang accepts forward references at
  // module scope but C++ (which compiles this same header) does not.
  float3 convert(uint32_t format, ReadBuffer(float) input, uint32_t index);

  // NV-DXVK: lightmap-UV-specific conversion. Source/TF2 wall VSes (e.g.
  // VS_e7abcf4e) consume R32G32_UINT or R16G16_UINT for TEXCOORD1 and
  // decode it as `float(uint(bits)) * (1.0/65535.0)` — different from
  // TEXCOORD0's tile_uv decode and from the position 21|21|22 decode.
  // Falls back to plain convert() for SFLOAT formats so artists who hand-
  // author lightmap UVs as floats still work.
  float3 convertLightmapTexcoord(uint32_t format, ReadBuffer(float) input, uint32_t index) {
    // R16G16_UINT (Source/TF2 standard lightmap layout). Two uint16 values
    // packed into one uint32 word: [V(31:16) | U(15:0)]. The native VS
    // (VS_e7abcf4e) does `utof r0.xy, v4.xy; mul o0.zw, r0, 1/65535` — same
    // result as treating each uint16 as 0..65535 and dividing by 65535.
    if (format == SupportedVkFormats::VK_FORMAT_R16G16_UINT) {
      uint data = asuint(input[index]);
      float u = float(data & 0xFFFFu)         * (1.0f / 65535.0f);
      float v = float((data >> 16u) & 0xFFFFu) * (1.0f / 65535.0f);
      return float3(u, v, 0);
    }
    // R32G32_UINT path. Each component is its own uint32. Some Source
    // shaders use this width when the lightmap atlas exceeds 16-bit range.
    if (format == SupportedVkFormats::VK_FORMAT_R32G32_UINT) {
      uint u0 = asuint(input[index + 0]);
      uint u1 = asuint(input[index + 1]);
      return float3(float(u0) * (1.0f / 65535.0f),
                    float(u1) * (1.0f / 65535.0f),
                    0);
    }
    return convert(format, input, index);
  }

  // NV-DXVK: NORMAL-specific conversion. Titanfall 2's lit-pass character VS
  // (e.g. VS_ef94e6c7fcc3c144) declares NORMAL as a single R32_UINT and
  // unpacks it inside the shader as an axis-dominant compressed normal:
  //   bits[29..30]  = axis id (0=X dominant, 1=Y, 2=Z) — 2 bits
  //   bit [28]      = sign of dominant component
  //   bits[19..27]  = first off-axis component (9-bit signed via offset 256)
  //   bits[10..18]  = second off-axis component
  //   bits[0..9]    = tangent rotation (Remix synthesizes its own TBN, ignored)
  //   bit [31]      = bitangent sign (also ignored)
  // Pure-axis verification:
  //   +X: axis=0, D=+255, a=b=0 → (D, b, a)/L = (1, 0, 0)
  //   +Y: axis=1, D=+255, a=b=0 → (a, D, b)/L = (0, 1, 0)
  //   +Z: axis=2, D=+255, a=b=0 → (b, a, D)/L = (0, 0, 1)
  float3 convertNormal(uint32_t format, ReadBuffer(float) input, uint32_t index) {
    if (format != SupportedVkFormats::VK_FORMAT_R32_UINT) {
      return convert(format, input, index);
    }
    const uint v   = asuint(input[index]);
    const float a  = float(int((v >> 10u) & 0x1FFu) - 256);
    const float b  = float(int((v >> 19u) & 0x1FFu) - 256);
    const float d  = ((v & (1u << 28u)) != 0u) ? -255.0f : 255.0f;
    const float invLen = 1.0f / sqrt(a * a + b * b + 65025.0f);
    const uint axis = (v >> 29u) & 0x3u;
    float3 n = float3(d, b, a);                   // axis 0 default
    if (axis == 1u) n = float3(a, d, b);
    if (axis == 2u) n = float3(b, a, d);
    return n * invLen;
  }

// NV-DXVK: TEXCOORD-specific conversion. Source Engine 2 / Titanfall 2 wall
  // vertex shaders consume R32G32_UINT texcoords with a tile_uv decode formula
  // that's distinct from the 21|21|22-bit POSITION decode; the original wall
  // VS (e.g. VS_e7abcf4e) computes:
  //   float u = float(int(uint(bits.x) >> 3) + int(0xF0000000)) * (1.0/16384.0);
  //   float v = float(int(uint(bits.y) >> 3) + int(0xF0000000)) * (1.0/16384.0);
  // Without this path, the generic convert() above interpreted the same uint
  // texcoord bits as packed XYZ positions, producing UVs in [-1024, +1024]
  // world-space magnitudes and pushing the gradient pipeline to max mip on
  // every wall draw — see scene_dump probe slot 5/6 and the brick-wall
  // diagnostic from 2026-04-29 (UVs at hundreds, mean mip ≥7 on 256² walls).
  // Other formats fall through to convert() unchanged so SFLOAT etc. behavior
  // is preserved.
  float3 convertTexcoord(uint32_t format, ReadBuffer(float) input, uint32_t index) {
    if (format == SupportedVkFormats::VK_FORMAT_R32G32_UINT) {
      uint u0 = asuint(input[index + 0]);
      uint u1 = asuint(input[index + 1]);
      // Replicate the wall VS's decode exactly. The (>>3 + 0xF0000000) pair
      // wraps in int32 to recentre the encoded range around zero; the
      // /16384 scale puts a typical TF2 BSP texinfo planar-projection UV
      // back into the few-tile range the artist authored.
      int   sx = int(u0 >> 3u) + int(0xF0000000);
      int   sy = int(u1 >> 3u) + int(0xF0000000);
      float u  = float(sx) * (1.0f / 16384.0f);
      float v  = float(sy) * (1.0f / 16384.0f);
      return float3(u, v, 0);
    }
    return convert(format, input, index);
  }

  float3 convert(uint32_t format, ReadBuffer(float) input, uint32_t index) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_R16G16_SFLOAT:
    {
      // Two half-floats packed into one 32-bit word: [G(31:16) | R(15:0)] in memory
      uint data = asuint(input[index]);
      float r = f16tof32(data & 0xFFFFu);
      float g = f16tof32((data >> 16u) & 0xFFFFu);
      return float3(r, g, 0);
    }
    case SupportedVkFormats::VK_FORMAT_R16G16B16A16_SFLOAT:
    {
      // NV-DXVK: Four half-floats packed into two 32-bit words.
      // Word 0: [Y_f16(31:16) | X_f16(15:0)], Word 1: [W_f16(31:16) | Z_f16(15:0)]
      uint data0 = asuint(input[index]);
      uint data1 = asuint(input[index + 1]);
      float x = f16tof32(data0 & 0xFFFFu);
      float y = f16tof32((data0 >> 16u) & 0xFFFFu);
      float z = f16tof32(data1 & 0xFFFFu);
      return float3(x, y, z);
    }
    case SupportedVkFormats::VK_FORMAT_R32G32_UINT:
    {
      // NV-DXVK: Source Engine 2 (Titanfall 2) quantized vertex positions.
      // Two uint32 values pack X, Y, Z as 21/21/22-bit unsigned integers:
      //   v0.x bits  0-20 (21 bits) → X
      //   v0.x bits 21-31 + v0.y bits 0-9 (21 bits) → Y
      //   v0.y bits 10-31 (22 bits) → Z
      // Decoded: float(uint_val) * (1.0/1024.0) + offset
      //
      // NV-DXVK TF2 VIEWMODEL FIX: Z offset is -2048.0 (not -2080.0). DXIL
      // disassembly of VS_ef94e6c7fcc3c144 (the player-body / viewmodel
      // shader that draws the gun + hands) shows the FMad constant as
      // -2.048000e+03 = 0xC5000000. Previous code used -2080 (0xC5020000)
      // from a different shader. The 32-unit Z error was enough to push
      // the entire viewmodel mesh outside the camera frustum on every
      // vertex — the gun was positioned 32 units below where Source put
      // it, just below the lower clip plane of the viewmodel-aware
      // projection. Fixing the offset restores per-vertex math parity
      // with the native game.
      uint u0 = asuint(input[index + 0]);
      uint u1 = asuint(input[index + 1]);
      uint xi = u0 & 0x001FFFFFu;                           // 21 bits
      uint yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u); // 21 bits
      uint zi = u1 >> 10u;                                   // 22 bits
      const float kScale = 1.0f / 1024.0f;  // 0.0009765625
      float x = float(xi) * kScale - 1024.0f;
      float y = float(yi) * kScale - 1024.0f;
      float z = float(zi) * kScale - 2048.0f;
      return float3(x, y, z);
    }
    case SupportedVkFormats::VK_FORMAT_R32G32_SFLOAT:
      return float3(input[index + 0], input[index + 1], 0);
    case SupportedVkFormats::VK_FORMAT_R32G32B32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32A32_SFLOAT:
      return float3(input[index + 0], input[index + 1], input[index + 2]);
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
    {
      uint data = asuint(input[index]);
      float b = unorm8ToF32(uint8_t((data >> 16) & 0xFF));
      float g = unorm8ToF32(uint8_t((data >> 8) & 0xFF));
      float r = unorm8ToF32(uint8_t((data >> 0) & 0xFF));
      return float3(r, g, b) * 2.f - 1.f;
    }
    case SupportedVkFormats::VK_FORMAT_A2B10G10R10_SNORM_PACK32:
    {
      uint data = asuint(input[index]);
      float b = unorm10ToF32(data >> 20);
      float g = unorm10ToF32(data >> 10);
      float r = unorm10ToF32(data >> 0);
      return float3(r, g, b);
    }
    }
    return float3(1, 1, 1);
  }

  uint3 convert(uint32_t format, ReadBuffer(uint32_t) input, uint32_t index) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_B8G8R8A8_UNORM:
      // Passthrough format we support in other places
      return uint3(input[index], 0, 0);
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
    {
      // D3D11 vertex colors are RGBA; Remix needs BGRA — swap R and B.
      uint32_t data = input[index];
      uint32_t r = data & 0xFFu;
      uint32_t b = (data >> 16u) & 0xFFu;
      data = (data & 0xFF00FF00u) | (r << 16u) | b;
      return uint3(data, 0, 0);
    }
    case SupportedVkFormats::VK_FORMAT_R32G32_UINT:
      // NV-DXVK: Position-uint-read hijack (see rtx_geometry_utils.cpp around
      // line 913): when source positions are R32G32_UINT, the color0 binding is
      // repurposed as a uint read source for position decoding. Any actual
      // color0 data path for this draw is bogus. Return full-white BGRA8 so
      // that if isVertexColorBakedLighting is (incorrectly) treated as true,
      // the surface receives full baked-light = visible geometry instead of
      // near-zero (0x00000001) black. The previous fallback `uint3(1,1,1)`
      // produced BGRA=(1,0,0,0) ≈ black → invisible BSP.
      return uint3(0xFFFFFFFFu, 0, 0);
    }
    return uint3(1,1,1);
  }

  // NV-DXVK: Decode R32G32_UINT position from uint buffer (avoids NaN corruption).
  float3 convertPositionUint(ReadBuffer(uint32_t) input, uint32_t index) {
    uint32_t u0 = input[index + 0];
    uint32_t u1 = input[index + 1];

    // 21/21/22-bit decode (verified from shader bytecode for ALL 1001 VS shaders)
    uint32_t xi = u0 & 0x001FFFFFu;
    uint32_t yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u);
    uint32_t zi = u1 >> 10u;
    const float kScale = 1.0f / 1024.0f;
    float x = float(xi) * kScale - 1024.0f;
    float y = float(yi) * kScale - 1024.0f;
    // NV-DXVK: bias was -2080 (from a misread `0xC5020000` constant). Actual
    // value in VS_759738774 / VS_4798dc2d / etc. disasm is l(-2048.0f) for the
    // Z component of the unpack `mad`. Verified by dumping VBUF + CPU-decoded
    // reference and observing exact -32 offset between them.
    float z = float(zi) * kScale - 2048.0f;
    // DEBUG: output u0 raw bits to verify color0 uint buffer is reading correctly.
    // If u0_lo = low 16 bits of u0, compare with raw dump word 0.
    // e.g. raw dump v0 word0 = 0x33CFD48A → low16 = 0xD48A = 54410
    // If the shader reads the same, u0_lo should be 54410 for vertex 0.
    // DEBUG: output raw uint16 components from color0 buffer
    // v0 word0 = 0x33CFD48A → lo=0xD48A=54410, hi=0x33CF=13263
    // If these match, color0 reads are correct and decode is the issue.
    return float3(x, y, z);
  }

  // NV-DXVK: Apply bone matrix (float3x4, 12 floats per matrix) to position.
  // boneMatrix layout: [r00 r01 r02 tx | r10 r11 r12 ty | r20 r21 r22 tz]
  //
  // Previously had a scale-tolerant rotation-orthogonality / translation-
  // magnitude validity check with an outValid out-param so the caller
  // could drop "garbage" bone contributions. That was a workaround for
  // missing upper-half palette data (which made slots look uninitialized).
  // The proper fix — hooking DxvkContext::copyBuffer to capture the bulk
  // rig uploads TF2 writes via staging→t30 — eliminates the need for
  // validity heuristics entirely. Zero bones (when they do appear) just
  // contribute zero to the weighted sum, matching the native VS exactly.
  float3 applyBoneMatrix(ReadBuffer(float) boneBuffer, uint32_t boneIndex, uint32_t strideFloats, float3 pos) {
    uint32_t base = boneIndex * strideFloats;
    float3 row0 = float3(boneBuffer[base+0], boneBuffer[base+1], boneBuffer[base+2]);
    float  tx   = boneBuffer[base+3];
    float3 row1 = float3(boneBuffer[base+4], boneBuffer[base+5], boneBuffer[base+6]);
    float  ty   = boneBuffer[base+7];
    float3 row2 = float3(boneBuffer[base+8], boneBuffer[base+9], boneBuffer[base+10]);
    float  tz   = boneBuffer[base+11];
    float3 result;
#ifdef __cplusplus
    result.x = row0.x*pos.x + row0.y*pos.y + row0.z*pos.z + tx;
    result.y = row1.x*pos.x + row1.y*pos.y + row1.z*pos.z + ty;
    result.z = row2.x*pos.x + row2.y*pos.y + row2.z*pos.z + tz;
#else
    result.x = dot(row0, pos) + tx;
    result.y = dot(row1, pos) + ty;
    result.z = dot(row2, pos) + tz;
#endif
    return result;
  }
  // NV-DXVK: 3-bone weighted skinning matching TF2's VS (verified via DXBC
  // disassembly of VS_ef94e6c7fcc3c144). The VS:
  //   1. reads v2.xyz as 3 uint8 bone indices (RGBA8_UINT, 4th unused)
  //   2. reads v1.xy as 2 SIGNED int16 values (R16G16, treated as int)
  //   3. decodes weights: w0 = (int16(v1.x) + 1) / 32768,
  //                       w1 = (int16(v1.y) + 1) / 32768,
  //                       w2 = 1.0 - w0 - w1
  //   4. skinned = w0 * bone[idx.x] * pos
  //              + w1 * bone[idx.y] * pos
  //              + w2 * bone[idx.z] * pos
  float3 applyWeightedBones(ReadBuffer(float) boneBuffer,
                             ReadBuffer(uint32_t) srcBoneIndex,
                             ReadBuffer(uint32_t) srcBoneWeight,
                             uint32_t vertexIndex,
                             uint32_t matrixStrideFloats,
                             uint32_t indexStrideUints,
                             uint32_t weightStrideUints,
                             uint32_t indexOffsetUints,
                             uint32_t weightOffsetUints,
                             uint32_t boneIndexBase,
                             float3 pos) {
    // Load 3 bone indices (RGBA8_UINT packed into one uint32). Ignore .w.
    // NV-DXVK: + boneIndexBase = TF2's per-instance COLOR1.y offset, so each
    // instance skins from its own bone sub-range (palette[BLENDINDICES + base]).
    const uint32_t packedIdx = srcBoneIndex[vertexIndex * indexStrideUints + indexOffsetUints];
    uint32_t boneIdx0 = ((packedIdx >>  0) & 0xFFu) + boneIndexBase;
    uint32_t boneIdx1 = ((packedIdx >>  8) & 0xFFu) + boneIndexBase;
    uint32_t boneIdx2 = ((packedIdx >> 16) & 0xFFu) + boneIndexBase;

    // Load 2 SIGNED int16 weights packed into one uint32.
    // Sign-extend each 16-bit half by value comparison (works in both HLSL
    // and C++; avoids `asint`/`>>` on unsigned which differ across backends).
    // Then apply the Source convention (v+1)/32768.
    const uint32_t packedW = srcBoneWeight[vertexIndex * weightStrideUints + weightOffsetUints];
    const uint32_t lo = packedW & 0xFFFFu;
    const uint32_t hi = (packedW >> 16u) & 0xFFFFu;
    const float fLo = (lo < 0x8000u) ? float(lo) : (float(lo) - 65536.0f);
    const float fHi = (hi < 0x8000u) ? float(hi) : (float(hi) - 65536.0f);
    float w0 = (fLo + 1.0f) * (1.0f / 32768.0f);
    float w1 = (fHi + 1.0f) * (1.0f / 32768.0f);
    float w2 = 1.0f - w0 - w1;

    // Weights left unclamped to match the native VS exactly. Source
    // engine's int16 quantization occasionally produces weights slightly
    // outside [0,1] (especially a small negative w2 when w0+w1 > 1 due
    // to sign-flipped low bits). With valid bone matrices in all palette
    // slots via the CopyBuffer hook, those edge weights don't fling
    // vertices anymore — the sum still lands close to the correct
    // position. Previous clamping was part of the spike-suppression
    // machinery that's no longer needed.

    // Apply each bone. No validity filtering — with the DXVK copyBuffer
    // hook capturing TF2's bulk rig uploads, upper-half palette slots
    // are now populated properly and don't need heuristic rejection.
    // Zero bones (when they do appear) contribute zero to the weighted
    // sum, exactly matching the native VS.
    float3 p0raw = applyBoneMatrix(boneBuffer, boneIdx0, matrixStrideFloats, pos);
    float3 p1raw = applyBoneMatrix(boneBuffer, boneIdx1, matrixStrideFloats, pos);
    float3 p2raw = applyBoneMatrix(boneBuffer, boneIdx2, matrixStrideFloats, pos);

    // NV-DXVK TF2 SKINNING: plain weighted sum matching native VS exactly.
    // Verified via DXIL of VS_ef94e6c7fcc3c144:
    //   skinned = w0·(bone[idx0].T + bone[idx0].R·pos)
    //           + w1·(bone[idx1].T + bone[idx1].R·pos)
    //           + w2·(bone[idx2].T + bone[idx2].R·pos)
    // No renormalization, no validity/fallback heuristics. With the DXVK
    // CopyBuffer hook capturing TF2's bulk rig uploads (full 16-bone
    // palettes via staging→t30), all palette slots arrive on the GPU
    // correctly populated, so no gap-detection is needed.
    return p0raw * w0 + p1raw * w1 + p2raw * w2;
  }

  // NV-DXVK: rotation-only variant of applyBoneMatrix for transforming
  // a normal/direction vector. Same matrix layout as the position version
  // but drops the translation row (tx/ty/tz) — translations don't apply
  // to direction vectors.  Result is NOT renormalized here; the caller
  // does it after the weighted blend so non-orthogonal bones (scaled
  // skin) end up unit-length on output.
  float3 applyBoneMatrixToNormal(ReadBuffer(float) boneBuffer, uint32_t boneIndex, uint32_t strideFloats, float3 n) {
    uint32_t base = boneIndex * strideFloats;
    float3 row0 = float3(boneBuffer[base+0], boneBuffer[base+1], boneBuffer[base+2]);
    float3 row1 = float3(boneBuffer[base+4], boneBuffer[base+5], boneBuffer[base+6]);
    float3 row2 = float3(boneBuffer[base+8], boneBuffer[base+9], boneBuffer[base+10]);
    float3 r;
#ifdef __cplusplus
    r.x = row0.x*n.x + row0.y*n.y + row0.z*n.z;
    r.y = row1.x*n.x + row1.y*n.y + row1.z*n.z;
    r.z = row2.x*n.x + row2.y*n.y + row2.z*n.z;
#else
    r.x = dot(row0, n);
    r.y = dot(row1, n);
    r.z = dot(row2, n);
#endif
    return r;
  }

  // NV-DXVK: 3-bone weighted skinning of a normal. Mirrors applyWeightedBones
  // but uses the rotation-only bone helper above. Without this, skinned-mesh
  // normals stay in rest-pose object space while positions end up in world
  // space (interleaver-skinned), producing visible "stale-shading" patches
  // that track bone rotations — most obvious on the viewmodel and characters.
  // Caller is responsible for the final normalize.
  float3 applyWeightedBonesToNormal(ReadBuffer(float) boneBuffer,
                                     ReadBuffer(uint32_t) srcBoneIndex,
                                     ReadBuffer(uint32_t) srcBoneWeight,
                                     uint32_t vertexIndex,
                                     uint32_t matrixStrideFloats,
                                     uint32_t indexStrideUints,
                                     uint32_t weightStrideUints,
                                     uint32_t indexOffsetUints,
                                     uint32_t weightOffsetUints,
                                     uint32_t boneIndexBase,
                                     float3 n) {
    const uint32_t packedIdx = srcBoneIndex[vertexIndex * indexStrideUints + indexOffsetUints];
    uint32_t boneIdx0 = ((packedIdx >>  0) & 0xFFu) + boneIndexBase;
    uint32_t boneIdx1 = ((packedIdx >>  8) & 0xFFu) + boneIndexBase;
    uint32_t boneIdx2 = ((packedIdx >> 16) & 0xFFu) + boneIndexBase;
    const uint32_t packedW = srcBoneWeight[vertexIndex * weightStrideUints + weightOffsetUints];
    const uint32_t lo = packedW & 0xFFFFu;
    const uint32_t hi = (packedW >> 16u) & 0xFFFFu;
    const float fLo = (lo < 0x8000u) ? float(lo) : (float(lo) - 65536.0f);
    const float fHi = (hi < 0x8000u) ? float(hi) : (float(hi) - 65536.0f);
    float w0 = (fLo + 1.0f) * (1.0f / 32768.0f);
    float w1 = (fHi + 1.0f) * (1.0f / 32768.0f);
    float w2 = 1.0f - w0 - w1;
    float3 n0 = applyBoneMatrixToNormal(boneBuffer, boneIdx0, matrixStrideFloats, n);
    float3 n1 = applyBoneMatrixToNormal(boneBuffer, boneIdx1, matrixStrideFloats, n);
    float3 n2 = applyBoneMatrixToNormal(boneBuffer, boneIdx2, matrixStrideFloats, n);
    return n0 * w0 + n1 * w1 + n2 * w2;
  }

  void interleave(const uint32_t idx, WriteBuffer(float) dst, ReadBuffer(float) srcPosition, ReadBuffer(float) srcNormal, ReadBuffer(float) srcTexcoord, ReadBuffer(uint32_t) srcColor0, ReadBuffer(float) srcBoneMatrix, ReadBuffer(uint32_t) srcBoneIndex, ReadBuffer(uint32_t) srcBoneWeight, ReadBuffer(float) srcTexcoord1, ReadBuffer(uint32_t) srcVguiTexcoord3, ReadBuffer(float) srcVguiGlyphDims, const InterleaveGeometryArgs cb) {
    const uint32_t srcVertexIndex = idx + cb.minVertexIndex;

    uint32_t writeOffset = 0;

    // NV-DXVK: For R32G32_UINT, decode 21/21/22-bit packed positions.
    // Read from uint color0 buffer to avoid NaN canonicalization.
    // NV-DXVK: For R32G32_UINT, read from the uint color0 buffer.
    // MUST use StructuredBuffer<uint32_t> to avoid GPU FTZ (flush-to-zero)
    // which destroys denormalized float bit patterns. The packed 21/21/22
    // data often looks like denormals when interpreted as IEEE float.
    float3 position;
    if (cb.positionFormat == SupportedVkFormats::VK_FORMAT_R32G32_UINT)
      position = convertPositionUint(srcColor0, srcVertexIndex * cb.color0Stride + cb.color0Offset);
    else
      position = convert(cb.positionFormat, srcPosition, srcVertexIndex * cb.positionStride + cb.positionOffset);
    // NV-DXVK: Apply bone matrix if available (Source Engine 2 skinning/instancing).
    // The bone matrix transforms decoded object-space positions to camera-relative space.
    if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_TRANSFORM) != 0u) {
      const uint32_t matrixStrideFloats = cb.boneMatrixStride / 4u;
      if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_WEIGHTS) != 0u) {
        // 4-bone weighted skinning (TF2 skinned characters). blendIdx is
        // packed RGBA8_UINT (4 bytes); weights are 2×UNORM16 (one uint32).
        // boneIndexStride/weightStride are byte strides; convert to uint32
        // stride (/4) since StructuredBuffer<uint32_t> is indexed in uints.
        position = applyWeightedBones(
          srcBoneMatrix, srcBoneIndex, srcBoneWeight,
          srcVertexIndex, matrixStrideFloats,
          cb.boneIndexStride / 4u,
          cb.boneWeightStride,   // already in uint32 units from host
          cb.boneIndexOffsetUints,
          cb.boneWeightOffset,
          cb.boneIndexBase,
          position);
      } else {
        uint32_t boneIdx;
        if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_BONE_PER_VERTEX) != 0u) {
          // TF2 BSP / batched static props: each vertex has its own COLOR1 instance
          // index. boneIndexStride is the byte stride of the source vertex stream
          // (8 for R16G16B16A16_UINT, 16 for R32G32B32A32_UINT). Divide by 4 to get
          // the per-vertex offset in the uint32_t-typed StructuredBuffer.
          const uint32_t indexStrideFloats = cb.boneIndexStride / 4u;
          const uint32_t packed = srcBoneIndex[srcVertexIndex * indexStrideFloats];
          boneIdx = (packed & cb.boneIndexMask) + cb.boneIndexBase;
        } else {
          // Legacy single-bone-per-draw path (skinned characters).
          const uint32_t packed = srcBoneIndex[0];
          boneIdx = (packed & cb.boneIndexMask) + cb.boneIndexBase;
        }
        position = applyBoneMatrix(srcBoneMatrix, boneIdx, matrixStrideFloats, position);
      }
    }

    dst[idx * cb.outputStride + writeOffset++] = position.x;
    dst[idx * cb.outputStride + writeOffset++] = position.y;
    dst[idx * cb.outputStride + writeOffset++] = position.z;

    if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_NORMALS) != 0u) {
      float3 normals = convertNormal(cb.normalFormat, srcNormal, srcVertexIndex * cb.normalStride + cb.normalOffset);
      // NV-DXVK: apply the same bone transform that positions get above so
      // skinned-mesh normals end up in world space alongside their skinned
      // positions. Without this the viewmodel / characters show patches of
      // stale shading that track bone rotations. Mirrors the position bone
      // path exactly (same flag fan-out), with the rotation-only variants.
      if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_TRANSFORM) != 0u) {
        const uint32_t matrixStrideFloats = cb.boneMatrixStride / 4u;
        if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_BONE_WEIGHTS) != 0u) {
          normals = applyWeightedBonesToNormal(
            srcBoneMatrix, srcBoneIndex, srcBoneWeight,
            srcVertexIndex, matrixStrideFloats,
            cb.boneIndexStride / 4u,
            cb.boneWeightStride,
            cb.boneIndexOffsetUints,
            cb.boneWeightOffset,
            cb.boneIndexBase,
            normals);
        } else {
          uint32_t boneIdx;
          if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_BONE_PER_VERTEX) != 0u) {
            const uint32_t indexStrideFloats = cb.boneIndexStride / 4u;
            const uint32_t packed = srcBoneIndex[srcVertexIndex * indexStrideFloats];
            boneIdx = (packed & cb.boneIndexMask) + cb.boneIndexBase;
          } else {
            const uint32_t packed = srcBoneIndex[0];
            boneIdx = (packed & cb.boneIndexMask) + cb.boneIndexBase;
          }
          normals = applyBoneMatrixToNormal(srcBoneMatrix, boneIdx, matrixStrideFloats, normals);
        }
      }
      // Renormalize: weighted blend of differently-oriented bones, plus any
      // non-orthogonality in the matrices (scaled skin), can leave the
      // result off the unit sphere. surface_interaction tolerates non-unit
      // input but the BSDF path is happier with unit normals.
      const float nLenSq = normals.x*normals.x + normals.y*normals.y + normals.z*normals.z;
      if (nLenSq > 1e-12f) {
        const float invN = 1.0f / sqrt(nLenSq);
        normals = float3(normals.x * invN, normals.y * invN, normals.z * invN);
      }
      dst[idx * cb.outputStride + writeOffset++] = normals.x;
      dst[idx * cb.outputStride + writeOffset++] = normals.y;
      dst[idx * cb.outputStride + writeOffset++] = normals.z;
    } else if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_FORCE_NORMALS) != 0u) {
      // Reserve normal space with zeros; will be filled by smooth normals pass
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
    }

    if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_TEXCOORD) != 0u) {
      // NV-DXVK: convertTexcoord() routes R32G32_UINT to the TF2 wall-VS
      // tile_uv decode instead of the position 21|21|22-bit decode. All
      // other formats fall through to plain convert() unchanged.
      float3 texcoords = convertTexcoord(cb.texcoordFormat, srcTexcoord, srcVertexIndex * cb.texcoordStride + cb.texcoordOffset);
      dst[idx * cb.outputStride + writeOffset++] = texcoords.x;
      dst[idx * cb.outputStride + writeOffset++] = texcoords.y;
    }

    // NV-DXVK: lightmap UV (TEXCOORD1). Always written immediately after
    // TEXCOORD0 in the output stride so surface_interaction can reach it
    // via (texcoordOffset + 8). When texcoord1StrideFormat is 0 the slot is omitted
    // and outputStride accounts for that — the host computes the stride
    // identically (see computeOptimalVertexStride).
    {
      // NV-DXVK: lightmap UV. cb.texcoord1StrideFormat packs stride (low 16)
      // and VkFormat (high 16); a value of 0 means "no lightmap this draw"
      // (replaces the dropped hasTexcoord1 bool — push-constant budget).
      // TC0 and TC1 can differ in format/stride (e.g. R32G32_UINT vs
      // R16G16_UINT, or different VBs entirely) so we unpack here.
      const uint tc1StrideFormat = cb.texcoord1StrideFormat;
      if (tc1StrideFormat != 0u) {
        const uint tc1Stride = tc1StrideFormat & 0xFFFFu;
        const uint tc1Format = tc1StrideFormat >> 16;
        float3 lmuv = convertLightmapTexcoord(tc1Format, srcTexcoord1, srcVertexIndex * tc1Stride + cb.texcoord1Offset);
        dst[idx * cb.outputStride + writeOffset++] = lmuv.x;
        dst[idx * cb.outputStride + writeOffset++] = lmuv.y;
      }
    }

    if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_COLOR0) != 0u) {
      uint3 color0 = convert(cb.color0Format, srcColor0, srcVertexIndex * cb.color0Stride + cb.color0Offset);
      dst[idx * cb.outputStride + writeOffset++] = asfloat(color0.x);
    }

    // NV-DXVK: TF2 worldspace VGUI extra per-vertex data. The PS reads three
    // attributes the standard pipeline doesn't carry:
    //   v0.xyzw — TEXCOORD1 in the D3D11 input layout (semantic-named that way
    //             by Source) — 4 floats covering the primary glyph quad's
    //             pos pair (xy) and a secondary glyph pair (zw).
    //   v1.xy   — TEXCOORD2 — 2 floats, glyph dimensions / scale.
    //   v2.xyzw — TEXCOORD3 — int4, packed glyph/style/image indices.
    //
    // Standard layout already wrote position (3) + maybe normals (3) + maybe
    // TC0.xy (2) + maybe TC1.xy (2) + maybe color0 (1). For VGUI we APPEND
    // 8 additional floats at the end of the per-vertex output so the standard
    // layout's offsets aren't disturbed:
    //   [+0..3] TEXCOORD1.zw, TEXCOORD2.xy   (4 floats, the missing pos pair
    //                                         + glyph dims, sampled from
    //                                         srcTexcoord/srcTexcoord1 with
    //                                         their existing offsets — no new
    //                                         binding required, the standard
    //                                         TC0/TC1 args already point at
    //                                         the same VBs in this layout.)
    //   [+4..7] TEXCOORD3.xyzw as 4 floats (asfloat(int))   (the int4 indices
    //                                         re-interpreted as float bits so
    //                                         the BLAS storage stays a single
    //                                         float stream — surface decode
    //                                         re-asints them at the hit point.)
    // Stride math for VGUI is computed by the host (see rtx_geometry_utils);
    // we just append to whatever writeOffset is now.
    if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_VGUI_LAYOUT_ENABLE) != 0u) {
      // Secondary glyph quad pos lives at offsets +2/+3 of the same texcoord
      // buffer that holds primary quad pos at +0/+1 (D3D11Rtx::SubmitDraw
      // routes TEXCOORD1 — the 4-float xyzw quad-pos pair — into
      // texcoordBuffer when the VGUI signature is detected). cb.texcoordStride
      // is in float units (= 4 for VGUI's R32G32B32A32_SFLOAT).
      float2 secondaryQuadPos = float2(0.0, 0.0);
      if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_HAS_TEXCOORD) != 0u) {
        const uint base = srcVertexIndex * cb.texcoordStride + cb.texcoordOffset;
        secondaryQuadPos.x = srcTexcoord[base + 2u];
        secondaryQuadPos.y = srcTexcoord[base + 3u];
      }
      // Glyph dims read from the dedicated TEXCOORD2 binding (R32G32_SFLOAT).
      // vguiGlyphDimsStride is bytes; convert to float units. A zero stride
      // means "no glyph dims this draw" (shouldn't happen for genuine VGUI
      // but the placeholder binding case needs the gate).
      float2 glyphDims = float2(0.0, 0.0);
      if (cb.vguiGlyphDimsStride != 0u) {
        const uint gdStride32 = cb.vguiGlyphDimsStride / 4u;
        const uint gdOff32    = cb.vguiGlyphDimsOffset / 4u;
        const uint baseGd     = srcVertexIndex * gdStride32 + gdOff32;
        glyphDims.x = srcVguiGlyphDims[baseGd + 0u];
        glyphDims.y = srcVguiGlyphDims[baseGd + 1u];
      }
      dst[idx * cb.outputStride + writeOffset++] = secondaryQuadPos.x;
      dst[idx * cb.outputStride + writeOffset++] = secondaryQuadPos.y;
      dst[idx * cb.outputStride + writeOffset++] = glyphDims.x;
      dst[idx * cb.outputStride + writeOffset++] = glyphDims.y;

      // TEXCOORD3 → 4 packed indices, written as 4 floats (asfloat of int).
      // Two source-format paths:
      //   (a) 16-bit (R16G16B16A16_SINT/UINT) — real TF2 VGUI. 8 bytes per
      //       vertex = 2 × uint32. Unpack 4 × int16 with sign extension.
      //   (b) 32-bit (R32G32B32A32_SINT/UINT/SFLOAT bit-cast) — fallback.
      //       16 bytes = 4 × uint32 read directly.
      // The shader-side reader (vgui_evaluator.slangh::decodeIndices) treats
      // each output float as `asint(.)`, so 16-bit values written here as
      // sign-extended int32 round-trip cleanly.
      const uint t3Stride32 = cb.vguiTexcoord3Stride / 4u;
      const uint t3Off32    = cb.vguiTexcoord3Offset / 4u;
      const uint base3      = srcVertexIndex * t3Stride32 + t3Off32;
      if ((cb.flags & INTERLEAVE_GEOMETRY_FLAG_VGUI_TC3_IS_INT16) != 0u) {
        // 16-bit path: 2 uint32 reads cover all 8 bytes of TC3.
        const uint w0 = srcVguiTexcoord3[base3 + 0u];
        const uint w1 = srcVguiTexcoord3[base3 + 1u];
        // Sign-extend each 16-bit half (low and high). Pattern matches
        // the asm (`int(int(packed << 16) >> 16)` is the standard 16→32
        // sign-extension idiom; works in both HLSL and Slang).
        const int x16 = int(int(w0 << 16) >> 16);
        const int y16 = int(w0) >> 16;
        const int z16 = int(int(w1 << 16) >> 16);
        const int w16 = int(w1) >> 16;
        dst[idx * cb.outputStride + writeOffset++] = asfloat(x16);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(y16);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(z16);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(w16);
      } else {
        // 32-bit path (legacy / non-real-VGUI). Direct 4-uint read.
        const uint i0 = srcVguiTexcoord3[base3 + 0u];
        const uint i1 = srcVguiTexcoord3[base3 + 1u];
        const uint i2 = srcVguiTexcoord3[base3 + 2u];
        const uint i3 = srcVguiTexcoord3[base3 + 3u];
        dst[idx * cb.outputStride + writeOffset++] = asfloat(i0);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(i1);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(i2);
        dst[idx * cb.outputStride + writeOffset++] = asfloat(i3);
      }
    }
  }
}

#ifdef __cplusplus
#undef WriteBuffer
#undef ReadBuffer

#undef asfloat
#undef asuint
#endif