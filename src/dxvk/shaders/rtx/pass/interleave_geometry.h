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
      //   X, Y offset = -1024.0,  Z offset = -2048.0
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
    }
    return uint3(1,1,1);
  }

  // NV-DXVK: Decode R32G32_UINT position from uint buffer (avoids NaN corruption).
  float3 convertPositionUint(ReadBuffer(uint32_t) input, uint32_t index) {
    uint32_t u0 = input[index + 0];
    uint32_t u1 = input[index + 1];
    uint32_t xi = u0 & 0x001FFFFFu;
    uint32_t yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u);
    uint32_t zi = u1 >> 10u;
    const float kScale = 1.0f / 1024.0f;
    float x = float(xi) * kScale - 1024.0f;
    float y = float(yi) * kScale - 1024.0f;
    float z = float(zi) * kScale - 2048.0f;
    return float3(x, y, z);
  }

  void interleave(const uint32_t idx, WriteBuffer(float) dst, ReadBuffer(float) srcPosition, ReadBuffer(float) srcNormal, ReadBuffer(float) srcTexcoord, ReadBuffer(uint32_t) srcColor0, const InterleaveGeometryArgs cb) {
    const uint32_t srcVertexIndex = idx + cb.minVertexIndex;

    uint32_t writeOffset = 0;

    // NV-DXVK: For R32G32_UINT, read from the uint color0 buffer to
    // avoid NaN canonicalization when loading through float buffers.
    float3 position;
    if (cb.positionFormat == SupportedVkFormats::VK_FORMAT_R32G32_UINT)
      position = convertPositionUint(srcColor0, srcVertexIndex * cb.color0Stride + cb.color0Offset);
    else
      position = convert(cb.positionFormat, srcPosition, srcVertexIndex * cb.positionStride + cb.positionOffset);
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