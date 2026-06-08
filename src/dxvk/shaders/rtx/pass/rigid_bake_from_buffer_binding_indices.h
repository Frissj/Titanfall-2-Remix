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
#pragma once

// NV-DXVK [RigidBake]: bake a single RIGID world transform, read on the GPU from a
// game-owned bone-matrix buffer, into a Remix geometry buffer (positions + normals,
// in place). Used for Titanfall instanced GPU-skinned studio models (e.g. the ridden
// Widow dropship hull) whose per-instance bone matrices live in a DEVICE-LOCAL SRV
// that cannot be CPU-mapped. The mesh is rigid (all bones share rotation), so a single
// bone matrix (bone[0] by default) is a correct object->world transform. Reading it on
// the GPU avoids any CPU readback / frame lag — the transform is exact for the frame.
//
// The bone matrix is a float3x4 stored ROW-MAJOR as 12 consecutive floats:
//   row0 = [0,1,2]=rotation, [3]=Tx ; row1 = [4,5,6],[7]=Ty ; row2 = [8,9,10],[11]=Tz
// i.e. world = (dot(row0,p)+Tx, dot(row1,p)+Ty, dot(row2,p)+Tz). Matches the engine
// g_boneMatrix StructuredBuffer<float3x4> layout (48-byte stride).

#include "rtx/utility/shader_types.h"

#define BINDING_RBFB_CONSTANTS          0
#define BINDING_RBFB_POSITION_IO        1
#define BINDING_RBFB_NORMAL_IO          2
#define BINDING_RBFB_BONES_INPUT        3

struct RigidBakeFromBufferArgs {
  uint numVertices;
  uint positionOffset;   // bytes
  uint positionStride;   // bytes
  uint normalOffset;     // bytes
  uint normalStride;     // bytes (0 => skip normals)
  uint boneFloatOffset;  // = boneIndex * 12 (float index of the chosen bone matrix)
  uint pad0;
  uint pad1;
};
