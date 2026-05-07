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

// GPU compute shader for radius-based culling of USD PointInstancer instances.
//
// Every thread writes a full VkAccelerationStructureInstanceKHR entry into the
// TLAS instance buffer unconditionally.  Visible instances get the real mask;
// culled instances get mask=0 so the TLAS builder ignores them.  This avoids
// any divergent early-outs and guarantees no stale data in the buffer.
//
// Each PointInstancer instance gets its own surface AND material entry so that it
// can have a unique surface/material ID in the customInstanceIndex field.  The CPU
// reserves N contiguous surface and material slots and populates only the first
// (the "template").  This shader copies the template data to each instance's slot,
// patches per-instance objectToWorld/prevObjectToWorld/normalObjectToWorld transforms
// in the surface buffer, and duplicates the material entry (a plain copy).

#include "rtx/pass/instance_culling/point_instancer_culling_bindings.slangh"

// Size of VkAccelerationStructureInstanceKHR in bytes
static const uint kVkAccelInstanceSize = 64;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  uint instanceIdx = threadId.x;
  if (instanceIdx >= cb.totalInstanceCount) {
    return;
  }

  // NV-DXVK: layout-agnostic matrix multiply.
  // HLSL/Slang's `M[r][c]` returns the math element at row r, column c regardless of
  // storage layout. Doing the 4x4 multiply element-by-element eliminates any chance
  // of Slang's storage-layout assumption disagreeing with CPU Matrix4 column-major.
  // (Empirically Slang reads CPU's column-major Matrix4 storage with correct
  // math semantics for both push-constant cb.objectToWorld AND structured-
  // buffer inputTransforms — verified by [SpawnGeomDiag.PIdump] dumps showing
  // sensible orthogonal/rotation columns.)
  float4x4 instanceToObject = inputTransforms[instanceIdx];

  // Extract all elements via [row][col] (math convention).
  const float o00 = cb.objectToWorld[0][0], o01 = cb.objectToWorld[0][1], o02 = cb.objectToWorld[0][2], o03 = cb.objectToWorld[0][3];
  const float o10 = cb.objectToWorld[1][0], o11 = cb.objectToWorld[1][1], o12 = cb.objectToWorld[1][2], o13 = cb.objectToWorld[1][3];
  const float o20 = cb.objectToWorld[2][0], o21 = cb.objectToWorld[2][1], o22 = cb.objectToWorld[2][2], o23 = cb.objectToWorld[2][3];
  const float o30 = cb.objectToWorld[3][0], o31 = cb.objectToWorld[3][1], o32 = cb.objectToWorld[3][2], o33 = cb.objectToWorld[3][3];
  const float i00 = instanceToObject[0][0], i01 = instanceToObject[0][1], i02 = instanceToObject[0][2], i03 = instanceToObject[0][3];
  const float i10 = instanceToObject[1][0], i11 = instanceToObject[1][1], i12 = instanceToObject[1][2], i13 = instanceToObject[1][3];
  const float i20 = instanceToObject[2][0], i21 = instanceToObject[2][1], i22 = instanceToObject[2][2], i23 = instanceToObject[2][3];
  const float i30 = instanceToObject[3][0], i31 = instanceToObject[3][1], i32 = instanceToObject[3][2], i33 = instanceToObject[3][3];

  // F = O * I — full 4x4 math product. Per-instance rotation/scale from
  // i2o (t31) is preserved end-to-end. Earlier this site ran a debug
  // experiment that forced i2o → translation-only, which collapsed BSP
  // tile orientations to identity and made tiles with non-axis-aligned
  // authored rotation render edge-on (most floor instances ended up
  // invisible — see [SpawnGeomDiag.PIdump]/[SpawnGeomDiag.VisibleSurf]
  // correlation showing 0/14, 4/9, 13/81 visibility for floor batches).
  // Reverted to the proper math product.
  const float f00 = o00*i00 + o01*i10 + o02*i20 + o03*i30;
  const float f01 = o00*i01 + o01*i11 + o02*i21 + o03*i31;
  const float f02 = o00*i02 + o01*i12 + o02*i22 + o03*i32;
  const float f03 = o00*i03 + o01*i13 + o02*i23 + o03*i33;
  const float f10 = o10*i00 + o11*i10 + o12*i20 + o13*i30;
  const float f11 = o10*i01 + o11*i11 + o12*i21 + o13*i31;
  const float f12 = o10*i02 + o11*i12 + o12*i22 + o13*i32;
  const float f13 = o10*i03 + o11*i13 + o12*i23 + o13*i33;
  const float f20 = o20*i00 + o21*i10 + o22*i20 + o23*i30;
  const float f21 = o20*i01 + o21*i11 + o22*i21 + o23*i31;
  const float f22 = o20*i02 + o21*i12 + o22*i22 + o23*i32;
  const float f23 = o20*i03 + o21*i13 + o22*i23 + o23*i33;
  // Row 3 of result is (0,0,0,1) for affine; not needed for VkTransformMatrixKHR.

  // Retain fullTransform for surface/normal writes below (unchanged semantics).
  float4x4 fullTransform = float4x4(
    float4(f00, f01, f02, f03),
    float4(f10, f11, f12, f13),
    float4(f20, f21, f22, f23),
    float4(0.0f, 0.0f, 0.0f, 1.0f));
  // Note: float4x4 ctor in HLSL/Slang takes rows in order, so M[0]=(f00..f03) row 0.

  // World translation = (f03, f13, f23) for affine — column 3 of math matrix.
  float3 worldPos = float3(f03, f13, f23);

  // Squared-distance cull (avoid sqrt in the common rejection case)
  float3 delta    = worldPos - cb.cameraPosition;
  float  distSq   = dot(delta, delta);
  float  radiusSq = cb.cullingRadius * cb.cullingRadius;

  bool visible = (distSq <= radiusSq);

  // Optional density fade: stochastically reject instances in the fade region.
  if (visible && cb.fadeStartRadius > 0.0 && distSq > cb.fadeStartRadius * cb.fadeStartRadius && cb.cullingRadius > cb.fadeStartRadius) {
    float dist       = sqrt(distSq);
    float fadeRange  = cb.cullingRadius - cb.fadeStartRadius;
    float fadeFactor = (dist - cb.fadeStartRadius) / fadeRange; // 0..1

    // Deterministic hash-based rejection (stable per instance, no flickering)
    uint  hash   = instanceIdx * 2654435761u; // Knuth multiplicative hash
    float rand01 = float(hash & 0xFFFFu) / 65535.0;

    if (rand01 < fadeFactor) {
      visible = false;
    }
  }

  uint mask = visible ? cb.instanceMask : 0;

  // === Per-instance surface data: copy template and patch transforms ===

  // Each instance gets its own surface index: baseSurfaceIndex + instanceIdx.
  // Copy the template surface, then patch per-instance transforms using
  // the Surface struct properties so that any future layout changes are
  // reflected here automatically.
  Surface surface = surfaceBuffer[cb.baseSurfaceIndex];

  float4x4 prevFull = mul(cb.prevObjectToWorld, instanceToObject);
  float3x3 normalMat = transpose(inverse((float3x3)fullTransform));

  surface.objectToWorld       = (mat4x3)fullTransform;
  surface.prevObjectToWorld   = (mat4x3)prevFull;
  surface.normalObjectToWorld = normalMat;

  surfaceBuffer[cb.baseSurfaceIndex + instanceIdx] = surface;

  // === Per-instance material data: copy template ===
  // Material data is identical for all instances of a PointInstancer;
  // this is a plain struct copy from the template slot.
  materialBuffer[cb.baseSurfaceIndex + instanceIdx] = materialBuffer[cb.baseSurfaceIndex];

  // === Write VkAccelerationStructureInstanceKHR unconditionally ===

  // VkTransformMatrixKHR is a 3x4 ROW-major matrix (3 rows, 4 cols, translation in col 3).
  // In HLSL/Slang, fullTransform[i] returns row i of the math matrix.
  // For a column-major affine matrix (project convention, see rtx_materials.h:177),
  // row i = (M(i,0), M(i,1), M(i,2), M(i,3)) — exactly what VkTransformMatrixKHR row i
  // needs. Write rows directly. The CPU transpose at rtx_accel_manager.cpp:835 reorders
  // BYTES (col-major→row-major) for memcpy; in the shader, accessing fullTransform[i]
  // already gives the math row, so no transpose is needed here.
  uint byteOffset = cb.instanceBufferOffset + instanceIdx * kVkAccelInstanceSize;

  // Write VkTransformMatrixKHR rows from the explicit math elements computed above.
  // VkTransformMatrixKHR row i = (M[i][0], M[i][1], M[i][2], M[i][3]).
  // Row 0
  instanceBuffer.Store(byteOffset +  0, asuint(f00));
  instanceBuffer.Store(byteOffset +  4, asuint(f01));
  instanceBuffer.Store(byteOffset +  8, asuint(f02));
  instanceBuffer.Store(byteOffset + 12, asuint(f03));
  // Row 1
  instanceBuffer.Store(byteOffset + 16, asuint(f10));
  instanceBuffer.Store(byteOffset + 20, asuint(f11));
  instanceBuffer.Store(byteOffset + 24, asuint(f12));
  instanceBuffer.Store(byteOffset + 28, asuint(f13));
  // Row 2
  instanceBuffer.Store(byteOffset + 32, asuint(f20));
  instanceBuffer.Store(byteOffset + 36, asuint(f21));
  instanceBuffer.Store(byteOffset + 40, asuint(f22));
  instanceBuffer.Store(byteOffset + 44, asuint(f23));

  // instanceCustomIndex:24 | mask:8 — each instance gets its own surface index
  uint perInstanceSurfaceIndex = cb.baseSurfaceIndex + instanceIdx;
  uint customIndexAndMask = (cb.customIndexFlags | (perInstanceSurfaceIndex & CUSTOM_INDEX_SURFACE_MASK)) | (mask << 24);
  instanceBuffer.Store(byteOffset + 48, customIndexAndMask);

  // instanceShaderBindingTableRecordOffset:24 | flags:8
  instanceBuffer.Store(byteOffset + 52, cb.sbtOffsetAndFlags);

  // accelerationStructureReference (uint64LE)
  instanceBuffer.Store(byteOffset + 56, cb.blasRefLo);
  instanceBuffer.Store(byteOffset + 60, cb.blasRefHi);
}
