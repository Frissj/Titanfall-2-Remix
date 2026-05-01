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
*/
#pragma once

#ifdef __cplusplus
#include <cstdint>
#endif

struct FilterDegenerateUvTrianglesArgs {
  uint32_t triangleCount;            // total triangles = indexCount / 3
  uint32_t vertexStrideUints;        // post-interleave stride in 32-bit units (e.g. 5 for vec3 pos + vec2 uv)
  uint32_t uvOffsetInVertexUints;    // offset of UV.x within each vertex slot in uint32 units (e.g. 3)
  uint32_t indexIs32Bit;             // 1 = R32_UINT, 0 = R16_UINT
  // Squared length threshold below which a UV-edge cross product is considered zero.
  // 1e-6f works well in practice; anything smaller is degenerate.
  float twoUvAreaThreshold;
  uint32_t pad0, pad1, pad2;
};
