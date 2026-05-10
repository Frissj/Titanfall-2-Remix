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

// [SkyTrace.probePrefill] Fills all 6 SkyProbe cube faces with the analytic
// Hillaire sky × TF2's authored skyTint. Runs once per frame at the start of
// rasterizeToSkyProbe; TF2's own sky-shader draws then overlay this background
// where they have geometry coverage, so face directions with no TF2 geometry
// (notably -Y / down, since TF2's 3D-skybox is upper-hemisphere only) keep
// the analytic atmosphere instead of going to pure black.
//
// Cube face direction convention is the Vulkan/D3D one — must match what
// SkyProbe.SampleLevel(direction) consumes downstream.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// rgba16 = R16G16B16A16_UNORM (matches the SkyProbe image format observed
// at runtime: VK_FORMAT_R16G16B16A16_UNORM = 91). Using rgba16f here would
// produce SFLOAT view-format bit-patterns that the downstream cubemap
// sampler (which sees the image as UNORM) would misinterpret. The shader-
// side float4 writes are implicitly clamped to [0,1] and converted to
// 16-bit UNORM on store, so any HDR radiance > 1 is clamped — matching
// what TF2's own raster output produces in the same texture.
//
// 2D (single-layer) storage view: caller binds one cube-face view per
// dispatch and tells us which face via args.probeFace. We tried a single
// 2D-array storage view + dispatch.z=6 first; SPIR-V looked correct but
// only layer 0 received writes in practice on this DXVK build. Six
// single-layer dispatches sidestep that.
layout(binding = 1, rgba16)
RWTexture2D<float4> skyProbeCubeOutput;

// Map a (face, uv) pair to a world-space direction the path tracer's
// SkyProbe.SampleLevel uses. Vulkan/D3D cubemap convention.
//   Face 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.
//   uv is [-1, +1] face-local (0,0) at face center.
vec3 cubeFaceDirection(uint face, vec2 st) {
  vec3 d;
  switch (face) {
    case 0: d = vec3( 1.0f, -st.y, -st.x); break;  // +X
    case 1: d = vec3(-1.0f, -st.y,  st.x); break;  // -X
    case 2: d = vec3( st.x,  1.0f,  st.y); break;  // +Y
    case 3: d = vec3( st.x, -1.0f, -st.y); break;  // -Y
    case 4: d = vec3( st.x, -st.y,  1.0f); break;  // +Z
    default:d = vec3(-st.x, -st.y, -1.0f); break;  // -Z
  }
  return normalize(d);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;

  // Caller dispatches once per face with the matching single-layer 2D
  // storage view bound. Side length comes from args.probeSide, current
  // face index from args.probeFace.
  const float side = float(args.probeSide > 0u ? args.probeSide : 256u);
  vec2 st = (vec2(threadId.xy) + 0.5f) / side;
  st = st * 2.0f - 1.0f;

  // World-space direction the path tracer would query for this texel,
  // using the cube-face direction convention that matches SkyProbe.SampleLevel
  // at the path-tracer sample sites.
  vec3 dir = cubeFaceDirection(args.probeFace, st);

  // [SkyTrace.probePrefill] Debug per-face solid colors removed —
  // confirmed via readback (frames 2995/2996) that all 6 faces receive
  // writes when using single-layer 2D dispatches with proper transfer
  // ↔ uniform barriers between iterations. Real Hillaire content below.

  // Hillaire-derived radiance. evalSkyRadiance assumes Y-up internally; if
  // the engine is Z-up, evalSkyRadiance handles the swap when given the
  // isZUp flag. We pass false here because cube directions are produced
  // in the same Y-up convention used by sampling sites in geometry_resolver
  // / integrator_indirect (which also call evalSkyRadiance with isZUp from
  // cb.isZUp; for the cube prefill the direction we feed in IS already in
  // the convention SkyProbe.SampleLevel will receive at sample-time, so
  // any axis-swap is handled symmetrically).
  vec3 radiance = evalSkyRadiance(dir, args, false);

  // Apply TF2's per-map skyTint as a post-multiply, matching the LUT
  // fallback path in integrator_indirect.slangh:494 (which does
  // evalSkyRadiance(dir) * cb.atmosphereArgs.skyTint). The cube is then
  // sampled at runtime via cb.skyBrightness * SkyProbe.SampleLevel(dir),
  // so we must NOT pre-multiply skyBrightness here — it's applied at
  // sample time. See the discussion around tint-over-physics versus
  // tint-into-physics in the rtx_atmosphere comments.
  radiance = radiance * args.skyTint;

  skyProbeCubeOutput[threadId.xy] = float4(max(radiance, vec3(0.0f)), 1.0f);
}
