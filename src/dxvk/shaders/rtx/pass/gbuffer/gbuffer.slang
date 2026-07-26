/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

//!variant gbuffer_raygen.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE


//!variant gbuffer_raygen_ser.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING


//!variant gbuffer_rayquery.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_rayquery_raygen.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_psr_raygen.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR


//!variant gbuffer_psr_raygen_ser.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       GBUFFER_PSR


//!variant gbuffer_psr_rayquery.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_psr_rayquery_raygen.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       ATMOSPHERE_AVAILABLE


/* NRC */

//!variant gbuffer_raygen_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1


//!variant gbuffer_raygen_ser_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1


//!variant gbuffer_rayquery_nrc.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_rayquery_raygen_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_psr_raygen_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1


//!variant gbuffer_psr_raygen_ser_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1


//!variant gbuffer_psr_rayquery_nrc.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE


//!variant gbuffer_psr_rayquery_raygen_nrc.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE



// WBOIT

//!variant gbuffer_raygen_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_raygen_ser_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_rayquery_wboit.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_rayquery_raygen_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_psr_raygen_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_psr_raygen_ser_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       GBUFFER_PSR
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_psr_rayquery_wboit.comp
//!>       RAY_TRACING_PRIMARY_RAY 
//!>       GBUFFER_PSR
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_psr_rayquery_raygen_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

/* NRC */

//!variant gbuffer_raygen_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT 

//!variant gbuffer_raygen_ser_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_rayquery_nrc_wboit.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_rayquery_raygen_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_psr_raygen_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_psr_raygen_ser_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant gbuffer_psr_rayquery_nrc_wboit.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE

//!variant gbuffer_psr_rayquery_raygen_nrc_wboit.rgen
//!>       RAY_TRACING_PRIMARY_RAY
//!>       RAY_PIPELINE
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE


/* No-ray-portal (opaque + translucent only) compute variants.
 *
 * NV-DXVK [perf]: the TraceRay / RayQueryRayGen pipelines have had an
 * `includePortals` permutation since upstream — getPipelineShaders() picks
 * gbuffer_material_opaque_translucent_closesthit when no ray portal materials
 * are present, which compiles resolveVertex WITHOUT the ray portal branch. The
 * compute (RaytraceMode::RayQuery) path never got that axis, so it always
 * compiled SURFACE_MATERIAL_RESOLVE_TYPE_ALL: every gbuffer_rayquery* shader
 * carried a full ray portal resolve path for a game that has zero portals.
 *
 * Two things are removed by the narrower mask, not one:
 *   1. the ray portal branch in resolveVertex (resolve.slangh, the third
 *      material type), and
 *   2. the ENTIRE intersection-billboard branch of resolveVertexUnordered.
 *      geometryResolverVertex selects GBUFFER_RAY_FLAGS by this same mask, and
 *      the narrower mask adds RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, which makes
 *      `isTriangle` a compile-time constant true. The billboard reorientation
 *      code (quaternion inverse/transform, ray-plane intersect, UV clip, vertex
 *      colour unpack, ~115 lines) is then dead and so is useIntersectionBillboards.
 *
 * That is a pure ALLOCATION cut in both the ordered and unordered halves, which
 * is the class of change the execution ablations in HANDOFF_TF2_UNORDERED_SPLIT
 * could not price. Selection is driven by the same `includePortals` predicate
 * the rt-pipeline path already uses, so a scene that does have portals still
 * gets the full shader.
 */

//!variant gbuffer_rayquery_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_rayquery_nrc_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_rayquery_wboit_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_rayquery_nrc_wboit_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_psr_rayquery_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_psr_rayquery_nrc_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_psr_rayquery_wboit_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

//!variant gbuffer_psr_rayquery_nrc_wboit_opaque_translucent.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       GBUFFER_PSR
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT

/* [Perf.Occupancy] block-size ladder — DIAGNOSTIC.
 *
 * Same source and same defines as gbuffer_rayquery_nrc_wboit_opaque_translucent
 * (the permutation this game runs: RayQuery compute, NRC on, WBOIT on, no ray
 * portals), differing only in declared workgroup size. See the REMIX_GB_BLOCK_X
 * note above main() for what each rung forces. Selected at runtime by
 * rtx.perfGbufferBlockThreads; not built for any other permutation, because this
 * is a measurement axis rather than a shipping option.
 */

//!variant gbuffer_rayquery_nrc_wboit_opaque_translucent_occ256.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT
//!>       REMIX_GB_BLOCK_X=16
//!>       REMIX_GB_BLOCK_Y=16

//!variant gbuffer_rayquery_nrc_wboit_opaque_translucent_occ384.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT
//!>       REMIX_GB_BLOCK_X=16
//!>       REMIX_GB_BLOCK_Y=24

//!variant gbuffer_rayquery_nrc_wboit_opaque_translucent_occ512.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!>       ENABLE_NRC=1
//!>       UNORDERED_RESOLVE_WBOIT
//!>       ATMOSPHERE_AVAILABLE
//!>       SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK=SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT
//!>       REMIX_GB_BLOCK_X=32
//!>       REMIX_GB_BLOCK_Y=16


//!end-variants

// Note: Opacity lighting approximation used in the GBuffer pass due to it being most visually
// prominent on particles in primary rays and important in PSR.
#define RESOLVE_OPACITY_LIGHTING_APPROXIMATION 1
// Note: Use thin film in the GBuffer pass for its usage in PSR on mirrorlike surfaces.
#define OPAQUE_MATERIAL_USE_THIN_FILM 1
#define OPAQUE_MATERIAL_USE_POM 1

#ifdef GBUFFER_PSR
// Note: This is done in primary rays only currently so that attenuation from volumetrics can be factored in
// similar to how it was done previously when it was simply applied in compositing via the total virtual hit
// distance, and ensuring that it does not attenuate secondary rays (since they do not receive inscattering and
// attenuating them would cause large amounts of energy loss as a result).
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1
#endif

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/gbuffer/gbuffer_binding_indices.h"
#include "rtx/pass/gbuffer/gbuffer_bindings.slangh"
// Note: Needed for declaration of g-buffer ray payload structure
#include "rtx/algorithm/geometry_resolver_state.slangh"

#include "rtx/concept/camera/camera.slangh"
#include "rtx/algorithm/geometry_resolver.slangh"

// NV-DXVK [perf]: occupancy ballast.
//
// The primary-ray shader compiles to Register Count=255 (the hardware cap) with
// 144 B/thread spilled, which allows 65536/255 = 257 threads per SM, i.e. 2
// blocks of 128 = 8 warps of a possible 48. The open question is whether that
// low occupancy is WHY the material evaluation costs ~47 ms, or merely a symptom
// of a shader that is genuinely doing that much ALU work.
//
// Attempts to answer it by REDUCING registers all failed to move the count off
// 255 (WBOIT removal, SurfaceInteraction debug fields). So test from the other
// direction: reduce occupancy deliberately and see whether time scales with it.
//
// Shared memory at 16896 B/block is currently not the binding constraint
// (2 blocks x 16.9 KB = 34 KB, well under the 100 KB/SM budget). Adding ~34 KB
// pushes a block over 50 KB, so only ONE block fits per SM: 128 threads, 4 warps,
// exactly half the current occupancy. Nothing else about the shader changes.
//
//   gb_primaryRays roughly DOUBLES  -> occupancy-bound. Reducing live state
//                                      (VGUI members, megakernel split) is the
//                                      fix, and the payoff is now predictable.
//   gb_primaryRays barely moves     -> ALU-throughput-bound. The 255 registers
//                                      are a symptom, not the cause, and the fix
//                                      is doing less work in the material path.
//
// Verify it applied by reading Shared Memory Size in [Perf.Shader]: it should go
// from 16896 to roughly 51712. 0 disables and is bit-identical to before.
#ifndef REMIX_OCCUPANCY_BALLAST_KB
#define REMIX_OCCUPANCY_BALLAST_KB 0
#endif

#if REMIX_OCCUPANCY_BALLAST_KB > 0
// 256 uints per KB. Allocation is static, so the whole array is reserved per
// block regardless of how few elements are actually touched below.
groupshared uint g_occupancyBallast[REMIX_OCCUPANCY_BALLAST_KB * 256];
#endif

#ifdef RAY_PIPELINE
[shader("raygeneration")]
void main()
{
  #ifdef GBUFFER_PSR
  geometryPSRResolverPass(DispatchRaysIndex().xy);
  #else
  geometryResolver(DispatchRaysIndex().xy);
  #endif
}
#else

// NV-DXVK [Perf.Occupancy]: workgroup size as a REGISTER CAP.
//
// The shared-memory ballast above cannot answer the occupancy question on Ada.
// maxComputeSharedMemorySize is 49152 and the SM carveout is 100 KB, so two
// 48 KB blocks still fit and a 128-thread block can never be pushed below 2
// blocks/SM by shared memory alone. That experiment would report "no change"
// and be misread as refuting occupancy.
//
// The workgroup size can, because a block must be resident in one SM's 65536
// registers: warps * 32 * registersPerThread <= 65536. Declaring a bigger block
// therefore FORCES the register allocator down, spilling as needed:
//
//   threads  warps  forced cap  warps/SM (1 block)
//   128        4      512  (255 actual)   8   <- current, 2 blocks/SM
//   256        8      256  (255 actual)   8      same occupancy, control case
//   384       12      170                12
//   512       16      128                16   <- 2x occupancy
//
// 256 is the control: same 8 warps/SM as today but through one block instead of
// two, so any delta there is block-granularity noise, not occupancy. 384 and 512
// buy real warps at the cost of spill traffic, which Local Memory Size on the
// [Perf.Shader] line prices directly.
//
//   gb_primaryRays falls at 384/512  -> latency-bound at 8 warps. Register
//                                       pressure IS the cause and the megakernel
//                                       split is worth building (it reaches the
//                                       same register count with NO spill).
//   flat or worse                    -> the 255 registers are a symptom. The
//                                       split cannot pay and should be dropped.
//
// This is a measurement axis, not a shipping configuration: the variants exist
// only for the permutation this game actually runs. rtx.perfGbufferBlockThreads
// selects one; 0 keeps the stock 128-thread shader.
#ifndef REMIX_GB_BLOCK_X
#define REMIX_GB_BLOCK_X 16
#endif
#ifndef REMIX_GB_BLOCK_Y
#define REMIX_GB_BLOCK_Y 8
#endif

[shader("compute")]
[numthreads(REMIX_GB_BLOCK_X, REMIX_GB_BLOCK_Y, 1)]
void main(uint2 ipos : SV_DispatchThreadID)
{
  if (any(ipos >= cb.camera.resolution))
  {
    return;
  }

#if REMIX_OCCUPANCY_BALLAST_KB > 0
  // Touch the ballast so it cannot be optimised away, at as close to zero cost
  // as possible: one shared store and one shared load per thread, no barrier.
  //
  // The load deliberately uses a DIFFERENT index from the store (tIdx vs
  // tIdx + 128) so the compiler cannot forward the stored value and promote the
  // array into registers - which would defeat the whole experiment by removing
  // the shared-memory allocation. Both indices stay inside the first 256
  // entries, so any ballast size >= 1 KB is in range.
  //
  // Reading is racy across threads and that is fine: the value is never used for
  // anything. The comparison is against a sentinel that ipos.x can never produce,
  // so the branch is never taken and control flow is unchanged.
  const uint tIdx = (ipos.x & 15u) + (ipos.y & 7u) * 16u;   // 0..127 within the block
  g_occupancyBallast[tIdx] = ipos.x;
  if (g_occupancyBallast[tIdx + 128u] == 0xFFFFFFFFu)
  {
    return;
  }
#endif

  #ifdef GBUFFER_PSR
  geometryPSRResolverPass(ipos);
  #else
  geometryResolver(ipos);
  #endif
}
#endif