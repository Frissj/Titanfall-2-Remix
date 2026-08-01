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

//!variant tlas_probe.comp
//!>       RAY_TRACING_PRIMARY_RAY
//!end-variants

// NV-DXVK [TlasProbe]: ask the BUILT acceleration structure directly.
//
// One thread per surface. Each shoots a short ray straight THROUGH that
// surface's own first triangle - from one side of the face to the other - and
// records what the TLAS returns. Two queries per surface, differing only in
// mask and ray flags:
//
//   ANY    - mask 0xFF, FORCE_OPAQUE only. Maximally permissive, and the one
//            that matters: with the ray crossing the triangle's own face,
//            nothing can occlude it and no mask can exclude it, so a miss
//            means the instance is genuinely not in the built structure at
//            that transform.
//   STRICT - OBJECT_MASK_ALL, CULL_BACK_FACING | FORCE_OPAQUE. The same mask
//            and flags geometryResolver uses. Its direction is the face normal
//            rather than a view ray, so it is NOT "what the primary ray would
//            have seen" - it isolates mask and backface rejection only. ANY hit
//            with STRICT miss means one of those two excluded it.
//
// The pair is the point. Every other probe in this investigation reads
// bookkeeping ABOUT the acceleration structure - the surface table, the
// instance entries, the BLAS records, the build/bind handles - and all of it
// reads identical on frames where the geometry renders and frames where it
// disappears. This reads the structure itself, which is the one thing that
// bookkeeping cannot vouch for.
//
// NV-DXVK [CamProbe]: two further queries, and one pure-arithmetic test.
//
// The face pair above answered its question - on frames the geometry vanishes
// it is still in the TLAS, still self-hittable, and (per ONSCREEN) still inside
// the frustum, with the user confirming it is not occluded. That leaves only
// the primary ray, so the probe now also reproduces it:
//
//   CAMDOTA/B - no ray at all. Project the centroid, turn the NDC back into a
//               screen UV, ask the camera what direction that pixel's ray
//               points, and measure the angle against the true direction to the
//               centroid. Tests ONLY whether worldToProjectionJittered agrees
//               with the viewToWorld/projectionToViewJittered pair the primary
//               ray is actually built from - three fields that are assumed to
//               be inverses and have never been checked.
//   CAMRAY    - a ray from the real camera origin straight at the centroid,
//               fired twice: once with the primary pass's mask and cull flags,
//               once with neither. Tests ONLY camera position, reachability,
//               occlusion, and backface/mask rejection along the view
//               direction. Uses no projection matrix and no NDC, so it cannot
//               be contaminated by whatever CAMDOTA/B finds, and vice versa.
//
// The split is deliberate: between them they cover the entire primary ray, and
// because they share no inputs, a failure in one localises the defect to that
// half rather than to "the camera" in general.
//
// Cost: one thread and four rays per surface (~2400 total), against 518400
// primary rays already traced each frame. Gated off by default.

// Deliberately NOT including surface.slangh / surface_interaction.slangh.
//
// surface_interaction.slangh carries debug-view probes guarded on
// RAY_TRACING_PRIMARY_RAY that call storeInDebugView, which only exists in
// passes that bind the debug-view texture. This pass does not bind it, and
// pulling in the debug-view header to satisfy the call would add a descriptor
// this pass has no use for and would then have to keep in sync with the
// pipeline layout. The triangle fetch is inlined below instead - it is a dozen
// lines, and it makes this shader depend only on bindings it actually declares.
//
// Everything needed is in common_bindings.slangh: the Surface struct, the
// surfaces[] buffer, the bindless geometries[]/indices[]/indices32[] arrays,
// topLevelAS and SurfaceCoverageBuffer.
#include "rtx/pass/common_bindings.slangh"
#include "rtx/utility/common.slangh"
// BUFFER_ARRAY, used for the index/position fetch below.
#include "rtx/utility/shader_types.slangh"
// OBJECT_MASK_ALL and CUSTOM_INDEX_SURFACE_MASK. Not reachable through
// common_bindings.slangh - that header declares the bindings but not the
// instance-mask/custom-index vocabulary, so it has to be included directly.
#include "rtx/pass/instance_definitions.h"
// NV-DXVK [CamProbe]: cameraGetWorldPosition / cameraScreenUVToDirection /
// cameraNDCToScreenUV. common_bindings.slangh brings in the Camera STRUCT
// (through raytrace_args.h) but none of the camera FUNCTIONS - it declares
// bindings, not vocabulary - so this has to be included directly. It is a light
// header: its only include is camera.h.
#include "rtx/concept/camera/camera.slangh"

// NV-DXVK [CamTris]: fetch triangle `triangleIndex` of `surface` in world space.
//
// Lifted verbatim out of main's original triangle-0 fetch so that both the face
// probe and the multi-triangle camera sampling read geometry through ONE piece
// of code. Two copies that were meant to agree would be a place for them to
// quietly stop agreeing, and every conclusion here rests on the two probes
// describing the same triangles.
//
// Returns false for a degenerate or non-finite triangle, which the callers must
// treat as "not sampled" rather than "not hit" - a degenerate source triangle is
// a different finding and must never be able to masquerade as absent geometry.
bool fetchWorldTriangle(Surface surface, uint triangleIndex, out vec3 w0, out vec3 w1, out vec3 w2)
{
  w0 = vec3(0.0f);
  w1 = vec3(0.0f);
  w2 = vec3(0.0f);

  uint idx[3];
  for (uint i = 0; i < 3; i++)
  {
    const uint indexIndex = surface.firstIndex + triangleIndex * 3u + i;

    if (surface.indexBufferIndex != BINDING_INDEX_INVALID)
    {
      const uint indexBufferIndex = surface.indexBufferIndex;
      if (surface.indexStride == 4)
      {
        idx[i] = BUFFER_ARRAY(indices32, indexBufferIndex, indexIndex);
      }
      else
      {
        idx[i] = BUFFER_ARRAY(indices, indexBufferIndex, indexIndex);
      }
    }
    else
    {
      idx[i] = indexIndex;
    }
  }

  vec3 objectPositions[3];
  for (uint i = 0; i < 3; i++)
  {
    const uint positionBufferIndex = surface.positionBufferIndex;
    const uint positionElementIndex =
      (idx[i] * uint(surface.positionStride) + surface.positionOffset) / 4;

    objectPositions[i].x = BUFFER_ARRAY(geometries, positionBufferIndex, positionElementIndex + 0);
    objectPositions[i].y = BUFFER_ARRAY(geometries, positionBufferIndex, positionElementIndex + 1);
    objectPositions[i].z = BUFFER_ARRAY(geometries, positionBufferIndex, positionElementIndex + 2);
  }

  w0 = mul(surface.objectToWorld, vec4(objectPositions[0], 1.0f));
  w1 = mul(surface.objectToWorld, vec4(objectPositions[1], 1.0f));
  w2 = mul(surface.objectToWorld, vec4(objectPositions[2], 1.0f));

  const vec3 faceCross = cross(w1 - w0, w2 - w0);
  const float faceCrossLength = length(faceCross);

  return (faceCrossLength > 0.0f) && !isnan(faceCrossLength) && !isinf(faceCrossLength)
      && !any(isnan(w0)) && !any(isinf(w0))
      && !any(isnan(w1)) && !any(isinf(w1))
      && !any(isnan(w2)) && !any(isinf(w2));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  if (cb.enableResolveCensus == 0)
  {
    return;
  }

  const uint surfaceIndex = threadId.x;
  if (surfaceIndex >= cb.surfaceCount || surfaceIndex >= uint(COVERAGE_SURFACE_SLOTS))
  {
    return;
  }

  const Surface surface = surfaces[surfaceIndex];

  if (surface.positionBufferIndex == BINDING_INDEX_INVALID)
  {
    return;
  }

  // Fetch this surface's FIRST triangle, exactly the way
  // surfaceInteractionCreate does, and probe THROUGH it.
  //
  // Not a ray from the camera toward the object origin, which is what this
  // originally did and what made it answer a weaker question than intended:
  // an object origin usually sits at the mesh's base, so a camera ray toward
  // it can hit terrain instead of the mesh whether or not the mesh is in the
  // structure. "Did not hit that surface" would then have meant "the ray did
  // not happen to cross it", not "it is absent" - ambiguous in exactly the
  // place this probe has to be sharp.
  //
  // Shooting through the triangle's own face removes every confound at once:
  // there is no camera, no distance, and nothing can be in front of it. If the
  // instance is in the TLAS with the transform the surface buffer describes,
  // this ray MUST hit it. It also tests the transform for free - an instance
  // that is present but placed elsewhere will not be where its own surface
  // data says its triangle is.
  vec3 w0, w1, w2;
  // A degenerate triangle has no face to shoot through. Returning rather than
  // recording RAN keeps "nothing is there" meaning only what it says - a
  // degenerate source triangle is a different finding and must not be able to
  // masquerade as an absent instance.
  if (!fetchWorldTriangle(surface, 0u, w0, w1, w2))
  {
    return;
  }

  const vec3 centroid = (w0 + w1 + w2) / 3.0f;
  const vec3 edge1 = w1 - w0;
  const vec3 edge2 = w2 - w0;
  const vec3 faceCross = cross(edge1, edge2);
  const float faceCrossLength = length(faceCross);
  const vec3 faceNormal = faceCross / faceCrossLength;

  // Stand off proportional to the triangle so the ray starts clear of the face
  // at any scale, with a floor for degenerate-small geometry. The segment is
  // symmetric about the centroid, so the face is crossed regardless of which
  // way the normal points.
  const float standoff = max(max(length(edge1), length(edge2)) * 0.5f, 0.01f);

  RayDesc rayDesc;
  rayDesc.Origin = centroid + faceNormal * standoff;
  rayDesc.Direction = -faceNormal;
  rayDesc.TMin = 0.0f;
  rayDesc.TMax = standoff * 2.0f;

  uint flags = COVERAGE_TLASPROBE_FLAG_RAN;
  uint anyHitSurfacePlusOne = 0u;
  uint camHitSurfacePlusOne = 0u;
  uint camDotAMilliDeg = 0u;
  uint camDotBMilliDeg = 0u;
  uint camDistUnits = 0u;

  // The camera origin the primary ray actually starts from. Read through
  // cameraGetWorldPosition rather than reconstructed here, so that if that
  // helper is itself wrong this probe is wrong in exactly the same way the
  // primary pass is - the point is to reproduce the primary ray, not to
  // second-guess it.
  const vec3 cameraOrigin = cameraGetWorldPosition(cb.camera);
  const vec3 toCentroid = centroid - cameraOrigin;
  const float centroidDistance = length(toCentroid);

  // Is this triangle even on screen this frame? The probe deliberately ignores
  // the camera so that presence can be tested without occlusion or view
  // direction interfering - but that makes a self-hit prove only "in the TLAS".
  // A surface the census calls NOTRAVERSED while the probe self-hits it is a
  // defect if it was on screen and a non-event if it was not, so the two have
  // to be separable in the same line.
  {
    // worldToProjectionJittered, not worldToProjection - Camera has no
    // unjittered world-to-projection for the current frame (only
    // prevWorldToProjection for the previous one). The jitter is a sub-pixel
    // TAA offset, so it cannot change whether a centroid is inside the frustum.
    const vec4 clipPosition = mul(cb.camera.worldToProjectionJittered, vec4(centroid, 1.0f));
    if (clipPosition.w > 0.0f)
    {
      const vec2 ndc = clipPosition.xy / clipPosition.w;
      if (all(abs(ndc) <= vec2(1.0f, 1.0f)))
      {
        flags |= COVERAGE_TLASPROBE_FLAG_ONSCREEN;
      }

      // NV-DXVK [CamProbe]: close the loop the frustum test leaves open.
      //
      // ONSCREEN says the centroid projects inside the frustum under
      // worldToProjectionJittered. It does NOT say that the primary ray built
      // for the pixel it lands on points anywhere near it - that ray comes from
      // viewToWorld and projectionToViewJittered, which are different fields.
      // Walk the projection back out through the camera's own inverse helpers
      // and measure the angle between where the pixel's ray goes and where the
      // geometry actually is.
      //
      // Both Y conventions, because which one matches
      // worldToProjectionJittered is an assumption, and an assumption baked in
      // here would produce a large error for a perfectly healthy camera - the
      // exact false positive this probe exists to avoid. Calibrate on frames
      // the geometry renders: the right convention reads ~0 there.
      if (centroidDistance > 0.0f && !isnan(centroidDistance) && !isinf(centroidDistance))
      {
        const vec3 trueDirection = toCentroid / centroidDistance;

        // cameraNDCToScreenUV is the exact inverse of the cameraScreenUVToNDC
        // that cameraScreenUVToDirection applies internally, so this round trip
        // introduces no convention of its own beyond the Y sign being probed.
        const vec3 dirA = cameraScreenUVToDirection(
          cb.camera, cameraNDCToScreenUV(cb.camera, ndc), true).worldDirection;
        const vec3 dirB = cameraScreenUVToDirection(
          cb.camera, cameraNDCToScreenUV(cb.camera, vec2(ndc.x, -ndc.y)), true).worldDirection;

        // Angle, not dot: near-perfect agreement crowds every dot product into
        // the last few bits below 1.0, where a fixed-point encoding cannot
        // resolve a sub-pixel error from an exact one. Degrees stay linear
        // right down to zero. Milli-degrees because a half-pixel at 90 deg FOV
        // over 1080 lines is ~0.04 deg - a whole-degree unit would round the
        // entire healthy range to 0 and the whole scale would be a pass/fail
        // bit wearing a number.
        const float kRadToMilliDeg = 57295.779513f;
        camDotAMilliDeg = uint(min(acos(clamp(dot(dirA, trueDirection), -1.0f, 1.0f))
                                   * kRadToMilliDeg, 180000.0f));
        camDotBMilliDeg = uint(min(acos(clamp(dot(dirB, trueDirection), -1.0f, 1.0f))
                                   * kRadToMilliDeg, 180000.0f));
        flags |= COVERAGE_TLASPROBE_FLAG_CAMPROJ_RAN;
      }
    }
  }

  // NV-DXVK [CamProbe]: can a ray from the camera reach this geometry at all?
  //
  // Aimed straight down normalize(centroid - cameraOrigin) rather than through
  // a pixel reconstructed from NDC. That is deliberate and it is what makes
  // this measurement independent of the one above: no NDC, no Y convention, no
  // projection matrix. It tests the camera POSITION, reachability through the
  // TLAS, and occlusion - nothing else. Combined with CAMDOTA/B, which test the
  // projection and nothing else, the two together cover the whole primary ray
  // without either being able to contaminate the other.
  //
  // Mask and ray flags are the primary pass's, so a hit here means the primary
  // pass's own trace would have found this geometry along this direction.
  if (centroidDistance > 0.0001f && !isnan(centroidDistance) && !isinf(centroidDistance))
  {
    RayDesc camRayDesc;
    camRayDesc.Origin = cameraOrigin;
    camRayDesc.Direction = toCentroid / centroidDistance;
    camRayDesc.TMin = 0.0f;
    // Slightly past the centroid so floating point cannot stop the ray on the
    // near side of its own target face.
    camRayDesc.TMax = centroidDistance * 1.001f + 0.01f;

    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> camRayQuery =
      RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE>();
    camRayQuery.TraceRayInline(
      topLevelAS,
      RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE,
      OBJECT_MASK_ALL,
      camRayDesc);
    camRayQuery.Proceed();

    if (camRayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      flags |= COVERAGE_TLASPROBE_FLAG_CAMRAY_HIT;
      const uint camHitSurface = uint(camRayQuery.CommittedInstanceID()) & CUSTOM_INDEX_SURFACE_MASK;
      camHitSurfacePlusOne = camHitSurface + 1u;
      if (camHitSurface == surfaceIndex)
      {
        flags |= COVERAGE_TLASPROBE_FLAG_CAMRAY_SELF;
      }
    }

    // The same ray with nothing excluded. Isolates backface culling and the
    // object mask on the CAMERA direction specifically - the face probe cannot,
    // because a ray along a face's own normal is never back-facing to it, so
    // its strict/any agreement says nothing about how the triangle faces the
    // camera.
    {
      RayQuery<RAY_FLAG_FORCE_OPAQUE> camAnyQuery = RayQuery<RAY_FLAG_FORCE_OPAQUE>();
      camAnyQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, camRayDesc);
      camAnyQuery.Proceed();

      if (camAnyQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
      {
        flags |= COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_HIT;
        const uint camAnyHitSurface = uint(camAnyQuery.CommittedInstanceID()) & CUSTOM_INDEX_SURFACE_MASK;
        if (camAnyHitSurface == surfaceIndex)
        {
          flags |= COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_SELF;
        }
      }
    }

    camDistUnits = uint(min(centroidDistance, 16777216.0f));
  }

  // NV-DXVK [CamTris]: does ANY part of this surface reach the camera?
  //
  // Everything above aims at triangle 0, and its own calibration showed that is
  // not a visibility test: on frames where geometry is winning millions of
  // primary hits, triangle 0 is the frontmost thing along its view ray only 16%
  // of the time, because it usually sits behind the rest of its own mesh. So
  // sample across the surface instead and ask whether ANY sampled triangle is
  // the committed first hit from the camera. That is what "this object is on
  // screen" actually means, and it is the precondition ordSeen > 0 requires.
  //
  // Strided rather than the first N: a surface whose leading triangles happen
  // to be interior, back-facing or degenerate would otherwise be written off on
  // the strength of exactly the triangles least likely to be visible - the same
  // sampling mistake as before, just repeated N times.
  {
    const uint primCount =
      SurfaceCoverageBuffer[COVERAGE_TLASPROBE_PRIMCOUNT_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex];
    if (primCount > 0u)
    {
      const uint sampleCount = min(primCount, COVERAGE_TLASPROBE_TRI_SAMPLES);
      const uint stride = max(primCount / sampleCount, 1u);

      uint trisTested = 0u;
      uint trisReached = 0u;

      for (uint sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx)
      {
        const uint triangleIndex = min(sampleIdx * stride, primCount - 1u);

        vec3 t0, t1, t2;
        if (!fetchWorldTriangle(surface, triangleIndex, t0, t1, t2))
        {
          continue;
        }

        const vec3 triCentroid = (t0 + t1 + t2) / 3.0f;
        const vec3 toTri = triCentroid - cameraOrigin;
        const float triDistance = length(toTri);
        if (!(triDistance > 0.0001f) || isnan(triDistance) || isinf(triDistance))
        {
          continue;
        }

        // This triangle must be inside the frustum, tested per TRIANGLE.
        //
        // Without this the probe counts geometry the renderer would never
        // shade: the ray is aimed from the camera at the triangle, so it is
        // trivially "frontmost along its own ray" even when it sits behind the
        // viewer or outside the field of view, where no primary ray is ever
        // fired. In the first capture that was 119 of 171 reported surfaces -
        // the majority of the apparent defect population was simply off screen.
        //
        // Per triangle, not per surface: the surface-level ONSCREEN bit is
        // computed from triangle 0, and the triangles sampled here are
        // different triangles. Reusing it would gate one triangle's visibility
        // on another's, which is the same aiming mistake this loop exists to
        // correct.
        const vec4 triClip = mul(cb.camera.worldToProjectionJittered, vec4(triCentroid, 1.0f));
        if (!(triClip.w > 0.0f))
        {
          continue;
        }
        const vec2 triNdc = triClip.xy / triClip.w;
        if (!all(abs(triNdc) <= vec2(1.0f, 1.0f)))
        {
          continue;
        }

        // Counted only now: trisTested means "sampled AND on screen", so
        // reached/tested reads directly as "of the parts of this surface the
        // renderer could have shaded, how many is it actually frontmost for".
        ++trisTested;

        RayDesc triRayDesc;
        triRayDesc.Origin = cameraOrigin;
        triRayDesc.Direction = toTri / triDistance;
        triRayDesc.TMin = 0.0f;
        triRayDesc.TMax = triDistance * 1.001f + 0.01f;

        // The primary pass's own mask and ray flags: the question is whether
        // the ray the renderer actually fires would find this surface here.
        RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> triQuery =
          RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE>();
        triQuery.TraceRayInline(
          topLevelAS,
          RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE,
          OBJECT_MASK_ALL,
          triRayDesc);
        triQuery.Proceed();

        if (triQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
          const uint triHitSurface = uint(triQuery.CommittedInstanceID()) & CUSTOM_INDEX_SURFACE_MASK;
          if (triHitSurface == surfaceIndex)
          {
            ++trisReached;
          }
        }
      }

      InterlockedMax(
        SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMTRISTESTED_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
        trisTested);
      InterlockedMax(
        SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMTRISREACHED_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
        trisReached);
    }
  }

  // STRICT: mask and ray flags matched to the primary ray.
  {
    // Explicit construction to match every other RayQuery in this codebase
    // (resolve.slangh, visibility.slangh, rtxcr_material.slangh).
    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> rayQuery =
      RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE>();
    rayQuery.TraceRayInline(
      topLevelAS,
      RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE,
      OBJECT_MASK_ALL,
      rayDesc);
    rayQuery.Proceed();

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      flags |= COVERAGE_TLASPROBE_FLAG_STRICT_HIT;
      const uint hitSurface = uint(rayQuery.CommittedInstanceID()) & CUSTOM_INDEX_SURFACE_MASK;
      if (hitSurface == surfaceIndex)
      {
        flags |= COVERAGE_TLASPROBE_FLAG_STRICT_SELF;
      }
    }
  }

  // ANY: everything the structure contains, regardless of mask or facing.
  {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> rayQuery = RayQuery<RAY_FLAG_FORCE_OPAQUE>();
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFFu, rayDesc);
    rayQuery.Proceed();

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      flags |= COVERAGE_TLASPROBE_FLAG_ANY_HIT;
      const uint hitSurface = uint(rayQuery.CommittedInstanceID()) & CUSTOM_INDEX_SURFACE_MASK;
      anyHitSurfacePlusOne = hitSurface + 1u;
      if (hitSurface == surfaceIndex)
      {
        flags |= COVERAGE_TLASPROBE_FLAG_ANY_SELF;
      }
    }
  }

  // One thread owns each surfaceIndex, so these are exact per-surface values.
  // Atomics anyway, for the same reason the other census sites use them: a
  // future dispatch shape that doubled up must corrupt nothing.
  InterlockedOr(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_FLAGS_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    flags);
  InterlockedMax(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_HITSURF_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    anyHitSurfacePlusOne);

  // NV-DXVK [CamProbe]. Same one-thread-per-surface ownership, same atomics for
  // the same reason.
  InterlockedMax(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMHITSURF_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    camHitSurfacePlusOne);
  InterlockedMax(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMDOTA_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    camDotAMilliDeg);
  InterlockedMax(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMDOTB_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    camDotBMilliDeg);
  InterlockedMax(
    SurfaceCoverageBuffer[COVERAGE_TLASPROBE_CAMDIST_REGION * uint(COVERAGE_SURFACE_SLOTS) + surfaceIndex],
    camDistUnits);
}
