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

// These are set indices - not bindings
#define BINDING_SET_BINDLESS_RAW_BUFFER          1
#define BINDING_SET_BINDLESS_TEXTURE2D           2
#define BINDING_SET_BINDLESS_SAMPLER             3


#define BINDING_ACCELERATION_STRUCTURE           0
#define BINDING_ACCELERATION_STRUCTURE_PREVIOUS  1
#define BINDING_ACCELERATION_STRUCTURE_UNORDERED 2
#define BINDING_ACCELERATION_STRUCTURE_SSS       3
#define BINDING_SURFACE_DATA_BUFFER              4
#define BINDING_SURFACE_MAPPING_BUFFER           5
#define BINDING_SURFACE_MATERIAL_DATA_BUFFER     6
#define BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER 7
#define BINDING_VOLUME_MATERIAL_DATA_BUFFER      8
#define BINDING_LIGHT_DATA_BUFFER                9
#define BINDING_PREVIOUS_LIGHT_DATA_BUFFER       10
#define BINDING_LIGHT_MAPPING                    11
#define BINDING_BILLBOARDS_BUFFER                12
#define BINDING_BLUE_NOISE_TEXTURE               13
#define BINDING_BINDLESS_INDICES_BUFFER          14
#define BINDING_CONSTANTS                        15
#define BINDING_DEBUG_VIEW_TEXTURE               16
#define BINDING_GPU_PRINT_BUFFER                 17
#define BINDING_VALUE_NOISE_SAMPLER              18
#define BINDING_SAMPLER_READBACK_BUFFER          19

#define COMMON_MAX_BINDING                       BINDING_SAMPLER_READBACK_BUFFER

// NV-DXVK: per-pixel scene dump (one-shot, ImGui-triggered). Slot 200 is
// well above every pass-specific binding (max observed ~190 across all
// passes) so it doesn't clash with hardcoded per-pass slots that start at
// 20 (e.g. RTXDI_COMPUTE_GRADIENTS_BINDING_RTXDI_RESERVOIR=20). Declared
// in common_bindings.slangh under `#ifdef RAY_TRACING_PRIMARY_RAY` so only
// primary-ray pipelines (gbuffer raygen / closesthit) emit the binding,
// keeping non-primary shaders' descriptor layouts untouched.
#define BINDING_SCENE_DUMP_BUFFER                200

// Atmosphere LUTs use high binding slots to avoid conflicts with pass-specific bindings.
// Shifted to 201-203 because BINDING_SCENE_DUMP_BUFFER already occupies 200.
#define BINDING_ATMOSPHERE_TRANSMITTANCE_LUT     201
#define BINDING_ATMOSPHERE_MULTISCATTERING_LUT   202
#define BINDING_ATMOSPHERE_SKY_VIEW_LUT          203
// NV-DXVK [AerialPerspective]: 3D LUT (32 x 32 x 32) holding the
// in-scattered radiance + transmittance for a ray of length D in
// direction view-relative-to-sun. Sampled per-shading-point in the
// path tracer to apply atmospheric haze on geometry, decoupled from
// the visible-sky source. Works in both PhysicalAtmosphere and Hybrid
// sky modes.
#define BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT 204

// NV-DXVK [Coverage]: per-pass surface-coverage histogram. Two regions
// (region 0 = geometry-resolver primary surface, region 1 = integrate-pass
// surface), each kCoverageSurfaceSlots uints, atomically incremented once
// per pixel that resolves to a given surfaceIndex. Slot 205 keeps it clear
// of the pass-specific bindings (which start at 20); ungated in the slang
// declaration like the atmosphere LUTs above so both the gbuffer and the
// integrate pipelines can write it.
#define BINDING_SURFACE_COVERAGE_BUFFER          205

// Per-region slot count for the coverage histogram. The buffer holds
// 2 * COVERAGE_SURFACE_SLOTS uints. 262144 comfortably covers any TF2
// scene's surface count; the shaders bounds-check surfaceIndex against
// it before the atomic add.
#define COVERAGE_SURFACE_SLOTS                   262144

#define COMMON_NUM_BINDINGS                      (COMMON_MAX_BINDING + 1)

// Note: Used to represent a non-existent buffer
#define BINDING_INDEX_INVALID uint16_t(0xFFFF)

// Sentinel for an invalid surface index.  Equals the 21-bit maximum (SURFACE_INDEX_MAX_VALUE
// from instance_definitions.h) so that it fits inside the packed RayInteraction._surfaceAndFlags
// field.  The surfaceMapping buffer returns int32_t(-1) for unmapped surfaces; the 21-bit
// property setter truncates 0xFFFFFFFF to 0x1FFFFF automatically.
// This reserves the highest representable surface index as "invalid", reducing the usable
// range by one (max usable index = SURFACE_INDEX_MAX_VALUE - 1 = 2,097,150).
#define SURFACE_INDEX_INVALID 0x001FFFFFu

#define SAMPLER_FEEDBACK_INVALID           uint16_t(0xFFFF)
#define SAMPLER_FEEDBACK_MAX_TEXTURE_COUNT uint16_t(0xFFFF)

// Note: Light array may only be up to a size of 2^16-1, allowing the last index to be
// used for an invalid index similar to the max binding index for materials.
#define LIGHT_INDEX_INVALID (0xFFFF)

#ifdef __cplusplus

#define COMMON_RAYTRACING_BINDINGS \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE)            \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_UNORDERED)  \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_PREVIOUS)   \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_SSS)        \
  STRUCTURED_BUFFER(BINDING_SURFACE_DATA_BUFFER)                    \
  STRUCTURED_BUFFER(BINDING_SURFACE_MAPPING_BUFFER)                 \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_DATA_BUFFER)           \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER)       \
  STRUCTURED_BUFFER(BINDING_VOLUME_MATERIAL_DATA_BUFFER)            \
  STRUCTURED_BUFFER(BINDING_LIGHT_DATA_BUFFER)                      \
  STRUCTURED_BUFFER(BINDING_PREVIOUS_LIGHT_DATA_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_MAPPING)                          \
  STRUCTURED_BUFFER(BINDING_BILLBOARDS_BUFFER)                      \
  TEXTURE2DARRAY(BINDING_BLUE_NOISE_TEXTURE)                        \
  CONSTANT_BUFFER(BINDING_CONSTANTS)                                \
  RW_TEXTURE2D(BINDING_DEBUG_VIEW_TEXTURE)                          \
  RW_STRUCTURED_BUFFER(BINDING_GPU_PRINT_BUFFER)                    \
  SAMPLER3D(BINDING_VALUE_NOISE_SAMPLER)                            \
  RW_STRUCTURED_BUFFER(BINDING_SAMPLER_READBACK_BUFFER)             \
  RW_STRUCTURED_BUFFER(BINDING_SCENE_DUMP_BUFFER)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_MULTISCATTERING_LUT)                 \
  TEXTURE2D(BINDING_ATMOSPHERE_SKY_VIEW_LUT)                        \
  TEXTURE3D(BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT)              \
  RW_STRUCTURED_BUFFER(BINDING_SURFACE_COVERAGE_BUFFER)
// NV-DXVK: SceneDumpBuffer is in COMMON_RAYTRACING_BINDINGS but uses slot
// 200 (out-of-the-way) so the C++ descriptor layout for every RT pipeline
// includes it; the slang declaration in common_bindings.slangh is gated on
// RAY_TRACING_PRIMARY_RAY so only primary shaders actually reference it.
// Non-primary pipelines bind the placeholder buffer but don't read/write
// it — the binding is silently unused.
#endif
