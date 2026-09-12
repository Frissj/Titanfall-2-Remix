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

// NV-DXVK [SceneCull] slice 9: one thread per CPU-owned TLAS entry. Copies the
// entry from the delta-uploaded instance table (the CPU's truth, never written
// by the GPU) into the TLAS input buffer with its mask decided by the scene
// cull verdict. Every CPU-owned entry is written every frame, tested or not,
// so the TLAS input never holds a stale entry. The mask is the only byte that
// can differ from the source -- which keeps G5's refit legal (mask is not part
// of an instance's active status; accelerationStructureReference is copied).

#include "rtx/pass/instance_culling/scene_cull_binding_indices.h"

layout(binding = SCENE_CULL_BINDING_CONSTANTS)
ConstantBuffer<SceneCullConstants> sceneCull;

layout(binding = SCENE_CULL_BINDING_RECORDS)
StructuredBuffer<SceneCullRecord> records;

layout(binding = SCENE_CULL_BINDING_SOURCE)
ByteAddressBuffer sourceInstances;

layout(binding = SCENE_CULL_BINDING_CULLED)
RWByteAddressBuffer culledInstances;

layout(binding = SCENE_CULL_BINDING_LIGHTS)
StructuredBuffer<float4> sceneCullLights;

layout(binding = SCENE_CULL_BINDING_STATS)
RWStructuredBuffer<uint> stats;

layout(binding = SCENE_CULL_BINDING_VERDICTS)
RWStructuredBuffer<uint> verdicts;

#include "rtx/pass/instance_culling/scene_cull.slangh"

// Size of VkAccelerationStructureInstanceKHR in bytes
static const uint kVkAccelInstanceSize = 64;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const uint recordIdx = threadId.x;
  if (recordIdx >= sceneCull.entryCount) {
    return;
  }

  // Record -> instance-table element. Each TLAS type's CPU entries are
  // contiguous and followed by that type's PointInstancer slots.
  uint element;
  if (recordIdx < sceneCull.typeFirstRecord.y) {
    element = sceneCull.typeBaseElement.x + recordIdx;
  } else if (recordIdx < sceneCull.typeFirstRecord.z) {
    element = sceneCull.typeBaseElement.y + (recordIdx - sceneCull.typeFirstRecord.y);
  } else {
    element = sceneCull.typeBaseElement.z + (recordIdx - sceneCull.typeFirstRecord.z);
  }

  const uint byteOffset = element * kVkAccelInstanceSize;
  // VkAccelerationStructureInstanceKHR: 3 rows of the 3x4 transform, then
  // instanceCustomIndex:24 | mask:8, sbtOffset:24 | flags:8, the BLAS reference.
  const uint4 row0 = sourceInstances.Load4(byteOffset +  0);
  const uint4 row1 = sourceInstances.Load4(byteOffset + 16);
  const uint4 row2 = sourceInstances.Load4(byteOffset + 32);
  uint4 tail       = sourceInstances.Load4(byteOffset + 48);

  const SceneCullRecord cullRecord = records[recordIdx];
  const uint mask = (tail.x >> 24) & 0xFFu;
  const uint verdict = sceneCullVerdict(asfloat(row0), asfloat(row1), asfloat(row2),
                                        cullRecord.boxMin, cullRecord.boxMax, cullRecord.flags, mask);
  if (sceneCullVerdictCulls(verdict)) {
    tail.x &= 0x00FFFFFFu;
  }

  culledInstances.Store4(byteOffset +  0, row0);
  culledInstances.Store4(byteOffset + 16, row1);
  culledInstances.Store4(byteOffset + 32, row2);
  culledInstances.Store4(byteOffset + 48, tail);

  if (recordIdx < sceneCull.verdictCount) {
    verdicts[recordIdx] = verdict;
  }
  InterlockedAdd(stats[verdict], 1u);
}
