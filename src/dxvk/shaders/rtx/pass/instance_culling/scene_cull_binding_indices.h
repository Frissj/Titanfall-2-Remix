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

// NV-DXVK [SceneCull] ARCHITECTURE_OVERHAUL.md slice 9 (sec 5.3): the scene
// cull on the GPU. Shared by the scene-cull pass (every CPU-owned TLAS entry)
// and the PointInstancer culling pass (every PI instance) so both apply ONE
// verdict function to the same constants. See rtx_scene_cull.h.

#include "rtx/pass/common_binding_indices.h"
#include "rtx/utility/shader_types.h"

// One record per CPU-owned TLAS entry, in the order the entries sit in the
// instance table. The box is in the ENTRY'S OWN object space: the shader runs
// the entry's 3x4 transform over its 8 corners, exactly as the CPU loop ran
// getTransform() over the BLAS box. A merged bucket's entry carries an
// identity transform (its geometry is baked world-space), so its record is
// the union of its members' world boxes.
struct SceneCullRecord {
  vec3 boxMin;
  uint flags;       // SCENE_CULL_RECORD_*
  vec3 boxMax;
  uint pad;
};

// The box is valid: run the verdict. Clear = copied through untested (no BLAS
// box, a billboard, a bucket with a member that has no box).
#define SCENE_CULL_RECORD_TESTED   0x1u
// The box bounds PRE-SKIN vertices, so it is not where the surface is: the
// verdict is void and the entry is never culled (the 2026-08-06 BT vanish).
#define SCENE_CULL_RECORD_SKINNED  0x2u

// SceneCullConstants::flags
#define SCENE_CULL_FLAG_ACTIVE      0x01u
#define SCENE_CULL_FLAG_FRUSTUM     0x02u
#define SCENE_CULL_FLAG_RADIUS      0x04u
#define SCENE_CULL_FLAG_LIGHT       0x08u
#define SCENE_CULL_FLAG_LIGHT_ALL   0x10u
#define SCENE_CULL_FLAG_SOLID_ANGLE 0x20u

// Per-entry verdict. The keep codes name the FIRST keep term that covered the
// entry (frustum before radius before light), as the CPU loop's counters did.
#define SCENE_CULL_VERDICT_UNTESTED      0u
#define SCENE_CULL_VERDICT_KEPT_FRUSTUM  1u
#define SCENE_CULL_VERDICT_KEPT_RADIUS   2u
#define SCENE_CULL_VERDICT_KEPT_LIGHT    3u
#define SCENE_CULL_VERDICT_KEPT_SKINNED  4u   // every keep missed; the skinned exemption kept it
#define SCENE_CULL_VERDICT_CULLED        5u   // every keep missed
#define SCENE_CULL_VERDICT_CULLED_SMALL  6u   // kept off-screen, then solid-angle rejected
#define SCENE_CULL_VERDICT_COUNT         7u

// Stats buffer: [0, COUNT) = TLAS-entry verdicts, [PI_BASE, PI_BASE + COUNT) =
// PointInstancer-instance verdicts.
#define SCENE_CULL_STATS_PI_BASE  8u
#define SCENE_CULL_STATS_COUNT    16u

struct SceneCullConstants {
  // worldToProj as explicit ROWS: clip.x = dot(row0, (p, 1)), etc. Rows rather
  // than a mat4 so no storage-layout convention stands between the CPU matrix
  // and the shader's arithmetic.
  vec4 worldToProjRow0;
  vec4 worldToProjRow1;
  vec4 worldToProjRow2;
  vec4 worldToProjRow3;
  vec3 camPos;
  float radiusSq;
  float sideScale;           // 1 + frustumMargin, widens L/R/T/B
  float solidAngleMinSq;
  float lightExemptDistSq;
  uint flags;                // SCENE_CULL_FLAG_*
  uint lightSegCount;        // lights[2*i] = (a, pad), lights[2*i+1] = (b, 0)
  uint lightPointCount;      // lights[2*lightSegCount + i] = (p, 0)
  uint entryCount;           // CPU-owned TLAS entries this frame
  uint verdictCount;         // entries + PointInstancer instances with a verdict slot
  // Record index of each TLAS type's first entry (x/y/z = Opaque/Unordered/SSS,
  // w = entryCount) and the instance-table element that entry sits at.
  uvec4 typeFirstRecord;
  uvec4 typeBaseElement;
};

#define SCENE_CULL_BINDING_CONSTANTS   60
#define SCENE_CULL_BINDING_RECORDS     61
#define SCENE_CULL_BINDING_SOURCE      62
#define SCENE_CULL_BINDING_CULLED      63
#define SCENE_CULL_BINDING_LIGHTS      64
#define SCENE_CULL_BINDING_STATS       65
#define SCENE_CULL_BINDING_VERDICTS    66

#define SCENE_CULL_MIN_BINDING  SCENE_CULL_BINDING_CONSTANTS

#if SCENE_CULL_MIN_BINDING <= COMMON_MAX_BINDING
#error "Increase the base index of scene cull bindings to avoid overlap with common bindings!"
#endif
