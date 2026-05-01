/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* NV-DXVK: per-triangle degenerate-UV filter.
*
* TF2 BSP geometry contains triangles whose 3 vertex UVs are collapsed (two
* of three vertex UVs are byte-identical, producing twoUVArea = 0). These
* triangles are authored INTO the same draw call as proper-UV walls, sharing
* rasterizer/blend state — so per-draw-call filtering can't separate them.
* In native rasterization they're hidden by depth-test ordering; in
* raytracing they compete with proper walls in the BLAS and dominate the
* depth tie at close range, producing the "uniform-grey wall" artifact (or
* magenta sentinel in our diagnostic build).
*
* This compute shader runs once per triangle. For each:
*   - read the 3 indices from src
*   - read each vertex's UV from the post-interleave vertex buffer
*   - compute twoUVArea (cross-product magnitude of two UV edges)
*   - if degenerate → write (i0, i0, i0) so the triangle collapses to a
*     point in object space and the BVH builder skips it
*   - else copy (i0, i1, i2) unchanged
*
* The src index buffer is left intact; only the dst is modified, so the
* CPU caller can substitute the dst for the BLAS build without affecting
* anything else.
*/

#include "filter_degenerate_uv_triangles_indices.h"
#include "filter_degenerate_uv_triangles.h"

// 16-bit index reads: U16 view (one StructuredBuffer<uint16_t>).
// 32-bit index reads: U32 view.
// Output: U32-typed RW view (we always write the wider type for simplicity;
// caller allocates dst at uint32_t stride when indexIs32Bit is true, otherwise
// at uint16_t stride and only the low 16 bits of each write are used).
layout(binding = FILTER_DEGEN_UV_TRIS_BINDING_DST_INDICES)
RWStructuredBuffer<uint> dstIndices;

layout(binding = FILTER_DEGEN_UV_TRIS_BINDING_SRC_INDICES_U16)
StructuredBuffer<uint16_t> srcIndicesU16;

layout(binding = FILTER_DEGEN_UV_TRIS_BINDING_SRC_INDICES_U32)
StructuredBuffer<uint> srcIndicesU32;

layout(binding = FILTER_DEGEN_UV_TRIS_BINDING_INTERLEAVED_VERTS)
StructuredBuffer<float> interleavedVerts;

layout(push_constant)
ConstantBuffer<FilterDegenerateUvTrianglesArgs> cb;

uint readSrcIndex(uint flatIdx)
{
  if (cb.indexIs32Bit != 0u) {
    return srcIndicesU32[flatIdx];
  } else {
    return uint(srcIndicesU16[flatIdx]);
  }
}

void writeDstIndex(uint flatIdx, uint val)
{
  if (cb.indexIs32Bit != 0u) {
    dstIndices[flatIdx] = val;
  } else {
    // Pack two 16-bit indices per 32-bit slot; low half = even flatIdx,
    // high half = odd flatIdx. Use atomic-or so concurrent writers in the
    // same dword don't stomp each other — at most 2 threads per slot, one
    // writing low and one writing high.
    const uint slot = flatIdx >> 1u;
    const uint shift = (flatIdx & 1u) * 16u;
    const uint mask = 0xFFFFu << shift;
    const uint payload = (val & 0xFFFFu) << shift;
    // Clear the half then set it. The "clear" needs the inverse mask; since
    // we'll overwrite both halves of every slot exactly once across the
    // dispatch, we can just OR. (If a slot's other half isn't touched in
    // this pass, this would leak the prior contents. For our use, every
    // index in [0, indexCount) is touched once, so it's safe.)
    InterlockedOr(dstIndices[slot], payload);
  }
}

float2 readUvAt(uint vertexIdx)
{
  const uint base = vertexIdx * cb.vertexStrideUints + cb.uvOffsetInVertexUints;
  return float2(interleavedVerts[base + 0], interleavedVerts[base + 1]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint triIdx : SV_DispatchThreadID)
{
  if (triIdx >= cb.triangleCount) {
    return;
  }

  const uint i0base = triIdx * 3u;
  const uint i0 = readSrcIndex(i0base + 0u);
  const uint i1 = readSrcIndex(i0base + 1u);
  const uint i2 = readSrcIndex(i0base + 2u);

  const float2 uv0 = readUvAt(i0);
  const float2 uv1 = readUvAt(i1);
  const float2 uv2 = readUvAt(i2);

  const float2 e1 = uv1 - uv0;
  const float2 e2 = uv2 - uv0;
  const float twoArea = abs(e1.x * e2.y - e1.y * e2.x);

  if (twoArea < cb.twoUvAreaThreshold) {
    // Degenerate UV — collapse triangle to a point so BVH skips it.
    writeDstIndex(i0base + 0u, i0);
    writeDstIndex(i0base + 1u, i0);
    writeDstIndex(i0base + 2u, i0);
  } else {
    writeDstIndex(i0base + 0u, i0);
    writeDstIndex(i0base + 1u, i1);
    writeDstIndex(i0base + 2u, i2);
  }
}
