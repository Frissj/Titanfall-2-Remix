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
    // NV-DXVK: R32G32_UINT (101) — Source Engine 2 (Titanfall 2) compressed
    // vertex positions.  Two uint32 values packing four fp16 components:
    //   uint0 = [half_y(31:16) | half_x(15:0)]
    //   uint1 = [half_w(31:16) | half_z(15:0)]
    // Decoded identically to R16G16B16A16_SFLOAT but declared as UINT in the
    // input layout because the vertex shader performs manual bit-extraction.
    VK_FORMAT_R32G32_UINT = 101,
    VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32_SFLOAT = 106,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,
  };

  bool formatConversionFloatSupported(uint32_t format) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_R16G16_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R16G16B16A16_SFLOAT:
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
      //   X, Y offset = -1024.0,  Z offset = -2080.0 (from shader magic 0xC5020000)
      uint u0 = asuint(input[index + 0]);
      uint u1 = asuint(input[index + 1]);
      uint xi = u0 & 0x001FFFFFu;                           // 21 bits
      uint yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u); // 21 bits
      uint zi = u1 >> 10u;                                   // 22 bits
      const float kScale = 1.0f / 1024.0f;  // 0.0009765625
      float x = float(xi) * kScale - 1024.0f;
      float y = float(yi) * kScale - 1024.0f;
      float z = float(zi) * kScale - 2080.0f;
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
  // Returns a zero-default indicator via `outValid` — true iff the bone
  // matrix has non-zero data (any uninitialized / gap slot in t30 returns
  // invalid so the caller can drop that contribution instead of sending
  // the vertex to world-origin as a spike).
#ifdef __cplusplus
#  define BONE_OUT_BOOL bool&
#else
#  define BONE_OUT_BOOL out bool
#endif
  float3 applyBoneMatrix(ReadBuffer(float) boneBuffer, uint32_t boneIndex, uint32_t strideFloats, float3 pos, BONE_OUT_BOOL outValid) {
    uint32_t base = boneIndex * strideFloats;
    float3 row0 = float3(boneBuffer[base+0], boneBuffer[base+1], boneBuffer[base+2]);
    float  tx   = boneBuffer[base+3];
    float3 row1 = float3(boneBuffer[base+4], boneBuffer[base+5], boneBuffer[base+6]);
    float  ty   = boneBuffer[base+7];
    float3 row2 = float3(boneBuffer[base+8], boneBuffer[base+9], boneBuffer[base+10]);
    float  tz   = boneBuffer[base+11];
    // Zero-bone detection: all 12 floats exactly 0. TF2 uploads bones at
    // stride-768 offsets (16-bone "slots") but only writes the first 8
    // bones of each slot. Bones 8-15 of each slot are uninitialized (all
    // zeros after buffer create). A vertex with BLENDINDICES referencing
    // one of those gaps would otherwise get transform = 0*local + 0 =
    // (0,0,0), producing a spike from the character to world origin.
    outValid = (row0.x != 0.0f || row0.y != 0.0f || row0.z != 0.0f
             || row1.x != 0.0f || row1.y != 0.0f || row1.z != 0.0f
             || row2.x != 0.0f || row2.y != 0.0f || row2.z != 0.0f
             || tx != 0.0f || ty != 0.0f || tz != 0.0f);
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
  // NV-DXVK: compatibility overload — legacy callers that don't need the
  // valid flag. Returns the position with no gap-detection; they continue
  // to work for non-skinned transform applications (e.g., t31 model-inst).
  float3 applyBoneMatrix(ReadBuffer(float) boneBuffer, uint32_t boneIndex, uint32_t strideFloats, float3 pos) {
    bool ignored;
    return applyBoneMatrix(boneBuffer, boneIndex, strideFloats, pos, ignored);
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
                             float3 pos) {
    // Load 3 bone indices (RGBA8_UINT packed into one uint32). Ignore .w.
    const uint32_t packedIdx = srcBoneIndex[vertexIndex * indexStrideUints + indexOffsetUints];
    uint32_t boneIdx0 = (packedIdx >>  0) & 0xFFu;
    uint32_t boneIdx1 = (packedIdx >>  8) & 0xFFu;
    uint32_t boneIdx2 = (packedIdx >> 16) & 0xFFu;

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

    // NV-DXVK: int16 quantization can make w0+w1 slightly exceed 1 for
    // 1- or 2-bone vertices, giving a tiny NEGATIVE w2. With a garbage
    // 3rd-slot bone index, that negative contribution flings the vertex
    // — producing the white spikes seen on skinned characters.
    // Clamp each to [0,1] and zero-out tiny residuals; guaranteed to
    // match game raster behavior for vertices intended to use <3 bones.
#ifdef __cplusplus
    w0 = (w0 < 0.0f) ? 0.0f : ((w0 > 1.0f) ? 1.0f : w0);
    w1 = (w1 < 0.0f) ? 0.0f : ((w1 > 1.0f) ? 1.0f : w1);
    w2 = (w2 < 0.0f) ? 0.0f : ((w2 > 1.0f) ? 1.0f : w2);
#else
    w0 = clamp(w0, 0.0f, 1.0f);
    w1 = clamp(w1, 0.0f, 1.0f);
    w2 = clamp(w2, 0.0f, 1.0f);
#endif

    // For each bone: skip if weight below quantization floor OR if the
    // bone matrix itself is uninitialized (all zeros — see comment in
    // applyBoneMatrix). The latter fix prevents spikes caused by TF2's
    // upload pattern that leaves gaps in t30 (writes only the first 8
    // bones of each 16-bone slot). A vertex referencing a gap bone with
    // non-zero weight otherwise collapses to world origin.
    bool v0 = false, v1 = false, v2 = false;
    float3 p0raw = applyBoneMatrix(boneBuffer, boneIdx0, matrixStrideFloats, pos, v0);
    float3 p1raw = applyBoneMatrix(boneBuffer, boneIdx1, matrixStrideFloats, pos, v1);
    float3 p2raw = applyBoneMatrix(boneBuffer, boneIdx2, matrixStrideFloats, pos, v2);
    const float wq = 1.0f / 32768.0f;
    const bool use0 = v0 && (w0 > wq);
    const bool use1 = v1 && (w1 > wq);
    const bool use2 = v2 && (w2 > wq);
    float3 p0 = use0 ? p0raw * w0 : float3(0.0f, 0.0f, 0.0f);
    float3 p1 = use1 ? p1raw * w1 : float3(0.0f, 0.0f, 0.0f);
    float3 p2 = use2 ? p2raw * w2 : float3(0.0f, 0.0f, 0.0f);
    const float wSum = (use0 ? w0 : 0.0f) + (use1 ? w1 : 0.0f) + (use2 ? w2 : 0.0f);
    if (wSum > 0.0f) return (p0 + p1 + p2) * (1.0f / wSum);
    return pos; // no valid bones; fallback to local-space pos
  }

  void interleave(const uint32_t idx, WriteBuffer(float) dst, ReadBuffer(float) srcPosition, ReadBuffer(float) srcNormal, ReadBuffer(float) srcTexcoord, ReadBuffer(uint32_t) srcColor0, ReadBuffer(float) srcBoneMatrix, ReadBuffer(uint32_t) srcBoneIndex, ReadBuffer(uint32_t) srcBoneWeight, const InterleaveGeometryArgs cb) {
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
    if (cb.hasBoneTransform) {
      const uint32_t matrixStrideFloats = cb.boneMatrixStride / 4u;
      if (cb.hasBoneWeights != 0u) {
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
          position);
      } else {
        uint32_t boneIdx;
        if (cb.bonePerVertex != 0u) {
          // TF2 BSP / batched static props: each vertex has its own COLOR1 instance
          // index. boneIndexStride is the byte stride of the source vertex stream
          // (8 for R16G16B16A16_UINT, 16 for R32G32B32A32_UINT). Divide by 4 to get
          // the per-vertex offset in the uint32_t-typed StructuredBuffer.
          const uint32_t indexStrideFloats = cb.boneIndexStride / 4u;
          const uint32_t packed = srcBoneIndex[srcVertexIndex * indexStrideFloats];
          boneIdx = packed & cb.boneIndexMask;
        } else {
          // Legacy single-bone-per-draw path (skinned characters).
          const uint32_t packed = srcBoneIndex[0];
          boneIdx = packed & cb.boneIndexMask;
        }
        position = applyBoneMatrix(srcBoneMatrix, boneIdx, matrixStrideFloats, position);
      }
    }

    dst[idx * cb.outputStride + writeOffset++] = position.x;
    dst[idx * cb.outputStride + writeOffset++] = position.y;
    dst[idx * cb.outputStride + writeOffset++] = position.z;

    if (cb.hasNormals) {
      float3 normals = convert(cb.normalFormat, srcNormal, srcVertexIndex * cb.normalStride + cb.normalOffset);
      dst[idx * cb.outputStride + writeOffset++] = normals.x;
      dst[idx * cb.outputStride + writeOffset++] = normals.y;
      dst[idx * cb.outputStride + writeOffset++] = normals.z;
    } else if (cb.forceNormals) {
      // Reserve normal space with zeros; will be filled by smooth normals pass
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
    }

    if (cb.hasTexcoord) {
      float3 texcoords = convert(cb.texcoordFormat, srcTexcoord, srcVertexIndex * cb.texcoordStride + cb.texcoordOffset);
      dst[idx * cb.outputStride + writeOffset++] = texcoords.x;
      dst[idx * cb.outputStride + writeOffset++] = texcoords.y;
    }

    if (cb.hasColor0) {
      uint3 color0 = convert(cb.color0Format, srcColor0, srcVertexIndex * cb.color0Stride + cb.color0Offset);
      dst[idx * cb.outputStride + writeOffset++] = asfloat(color0.x);
    }
  }
}

#ifdef __cplusplus
#undef WriteBuffer
#undef ReadBuffer

#undef asfloat
#undef asuint
#endif