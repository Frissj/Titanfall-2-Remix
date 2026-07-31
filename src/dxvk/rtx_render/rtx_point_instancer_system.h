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

#include <unordered_map>
#include <vector>

#include "../dxvk_format.h"
#include "../dxvk_include.h"

#include "rtx_resources.h"
#include "rtx/pass/instance_culling/point_instancer_culling_binding_indices.h"

namespace dxvk {
  class RtxContext;
  class DxvkContext;
  class InstanceManager;
  struct PooledBlas;
  struct BlasEntry;

  /**
    * GPU-driven radius culling system for USD PointInstancer replacements.
    *
    * PointInstancers produce large numbers of identical mesh instances (e.g. foliage,
    * ground clutter) specified by per-instance transforms. This system performs
    * camera-proximity culling entirely on the GPU to limit the number of instances
    * that are visible in the TLAS, reducing BVH traversal cost.
    *
    * Per-frame flow:
    *  1. AccelManager::mergeInstancesIntoBlas pushes N placeholder entries
    *     (mask=0) for each PointInstancer into m_mergedInstances/m_vkInstanceBuffer,
    *     and records batch descriptors for the GPU work.
    *  2. AccelManager::prepareSceneData uploads those placeholders to the GPU.
    *  3. AccelManager::dispatchPointInstancerCulling calls this system's
    *     dispatchCulling() method: a GPU compute shader evaluates each transform
    *     against the camera, and overwrites visible placeholders with full
    *     VkAccelerationStructureInstanceKHR entries (proper transform + mask).
    *     Culled entries stay mask=0 and are skipped by RT hardware.
    *  4. AccelManager::buildTlas proceeds normally.
    *
    * No CPU-side transform iteration occurs.
    */

  /**
    * Describes one PointInstancer dispatch recorded during mergeInstancesIntoBlas.
    * Consumed by dispatchCulling() to drive the GPU compute.
    */
  // NV-DXVK [PISurfWatch]: one GPU surface entry in 4-byte words. Mirrors
  // kSurfaceGPUSize (rtx_materials.h) without including that header here; the
  // static_assert in rtx_point_instancer_system.cpp holds the two together.
  constexpr uint32_t kSurfaceGPUDwords = 64;

  struct PointInstancerBatch {
    const std::vector<Matrix4>* transforms;       // Source instanceToObject transforms (CPU data, uploaded per batch)
    Matrix4 objectToWorld;                         // Object-to-world for this instancer
    Matrix4 prevObjectToWorld;                     // Previous-frame object-to-world (for motion vectors in surface data)
    uint32_t instanceCount;                        // Number of input transforms
    uint32_t baseSurfaceIndex;                     // surfaceIndexOfFirstInstance
    uint32_t customIndexFlags;                     // Upper bits of instanceCustomIndex (no surface mask)
    uint32_t instanceMask;                         // 8-bit visibility mask
    uint32_t sbtOffsetAndFlags;                    // Packed SBT offset (24) | flags (8)
    uint64_t blasReference;                        // BLAS device address
    uint32_t firstIndexInType;                     // Index of first placeholder within its TLAS type array
    uint32_t tlasType;                             // Tlas::Type (Opaque, Unordered, SSS)
    uint32_t instanceBufferByteOffset;             // Absolute byte offset in m_vkInstanceBuffer (resolved before dispatch)
    // NV-DXVK debug: hold a strong ref to the BLAS so we can validate at TLAS build
    // that the captured `blasReference` still matches a live AS handle.
    Rc<PooledBlas> debugBlasRef;
    bool debugAsBuiltAtCapture;                    // was scheduled in blasToBuild before addPI?
    // NV-DXVK [SpawnGeomDiag.PIBatchInventory]: source-side fingerprint
    // captured in addPointInstancerBlas. We use these at dispatch time to
    // detect cases where batch.blasReference resolves to a BLAS whose
    // primitive/vertex counts don't match the source — i.e. a stale or
    // mis-pooled handle (suspected for the 81-instance "floor" batch
    // currently pointing to a 16-prim BLAS).
    uint64_t debugVsHash;
    uint32_t debugSourcePrimCount;
    uint32_t debugSourceVertexCount;
    // NV-DXVK [PIWatch]: the geometry this batch was BUILT FROM, captured by
    // value at addPointInstancerBlas time.
    //
    // WHY BY VALUE. These are the exact expressions the BLAS builder assigns
    // into VkAccelerationStructureGeometryTrianglesDataKHR (accel_manager:388
    // and :428), so at dispatch time they can be compared directly against
    // what the LIVE BLAS at blasReference says it was built from. If the two
    // disagree, the batch is pointing at some other mesh's BLAS — which is the
    // failure that would corrupt every instance in the batch at once. Copying
    // the addresses rather than following debugSourceBlasEntry later also
    // keeps this off the raw-BlasEntry-pointer path, which is recycled
    // aggressively and has produced use-after-free crashes in other probes.
    //
    // NOTE the old debugSourcePrimCount check (primMatch) could never work:
    // PooledBlas::primitiveCounts is only ever assigned on the merged-bucket
    // path (accel_manager:2261), and PI batches use blasEntry->dynamicBlas,
    // so it is always empty and livePrim always reads 0.
    uint64_t debugSrcVtxAddr;
    uint64_t debugSrcIdxAddr;
    uint32_t debugSrcVertexStride;
    uint32_t debugCaptureFrame;
    // NV-DXVK [PIWatch]: STABLE cross-frame identity for this batch —
    // FullGeometryHash combined with an order-independent sum over the
    // per-instance transform multiset. Already computed in
    // addPointInstancerBlas for the duplicate-batch dedup; stored here so the
    // frame-to-frame watch can key on content instead of on list position.
    //
    // The watch originally keyed on (vsHash, srcPrim, srcVtx) plus an ordinal
    // to break ties between duplicate meshes. That ordinal is assigned from
    // the batch's position in m_pointInstancerBatches, so when the batch list
    // reorders — which is the very thing being measured — the ordinal can
    // hand key K to a different physical batch and manufacture a "change"
    // that never happened. It could only ever affect the duplicate tuples (2
    // of 86 batches when checked), but a probe whose identity is corrupted by
    // the effect it is measuring cannot be trusted at any rate. This key does
    // not depend on ordering at all.
    //
    // Plain uint64_t rather than XXH64_hash_t so this header need not pull in
    // xxHash; the value is an XXH64_hash_t and the types are identical.
    //
    // *** NOT USABLE AS A CROSS-FRAME KEY. *** Measured 18:28: keying [PIWatch]
    // on it produced appeared=86 vanished=86 on EVERY frame with batches=86
    // and totalInst=889 unchanged — i.e. this value differs every frame for
    // what is demonstrably the same batch. It is still exactly right for its
    // original purpose (within-frame duplicate rejection). Kept, and now split
    // into the two components below so the watch can report WHICH half moves
    // instead of just failing to match.
    uint64_t debugBatchKey;
    // The two halves of debugBatchKey, stored separately as WATCHED VALUES.
    //   debugGeoHash  — FullGeometryHash (VertexDataHash | TopologicalHash) of
    //                   the source geometry. Should be constant for static
    //                   replacement assets; if this is what moves, the vertex
    //                   data itself is being re-hashed differently per frame.
    //   debugXformSum — order-independent sum over XXH64 of each per-instance
    //                   Matrix4. If this is what moves, the PLACEMENTS are
    //                   changing frame to frame, which would be a direct
    //                   visual mechanism rather than a bookkeeping one.
    uint64_t debugGeoHash;
    uint64_t debugXformSum;
    // THE cross-frame identity for this batch, and the only one that should be
    // used as a map key: (vsHash, srcPrim, srcVtx, objectToWorld translation
    // bits). Computed once in addPointInstancerBlas so every consumer agrees.
    //
    // Do NOT substitute debugBatchKey — it is measurably unstable frame to
    // frame (see its comment) and using it as a key produces a total match
    // failure that looks like every batch being replaced every frame.
    // objectToWorld is in here because it never changed across any of the
    // 18606 change events of the 17:51 run, and it separates duplicate meshes
    // placed at different points — which is what the old ordinal failed at.
    uint64_t debugStableKey;
    // Raw translation of the batch's FIRST instance transform, so a moving
    // xformSum can be read as jitter vs a wholesale change without another
    // build. Zero when the batch has no transforms.
    float debugFirstXform[3];
    // [SpawnGeomDiag.FloorObjDump]: raw BlasEntry pointer so the OBJ-dump
    // path in dispatchPointInstancerCulling can reach the post-interleave
    // position+index buffer without searching m_debugBlasBuildEntries.
    // Diagnostic only; lifetime tracked elsewhere (BlasEntry lives in
    // SceneManager's BLAS map, longer than a single frame).
    BlasEntry* debugSourceBlasEntry;
  };

  class RtxPointInstancerSystem : public CommonDeviceObject {
  public:
    explicit RtxPointInstancerSystem(DxvkDevice* device);
    ~RtxPointInstancerSystem() = default;

    /**
      * Dispatches the GPU culling compute shader for all recorded batches.
      * Each batch writes VkAccelerationStructureInstanceKHR entries directly
      * into the TLAS instance buffer.
      *
      * \param ctx           Render context.
      * \param instanceBuffer The TLAS instance buffer (m_vkInstanceBuffer).
      * \param batches        Batch descriptors from AccelManager.
      * \param cameraPosition World-space camera position for distance test.
      */
    void dispatchCulling(Rc<DxvkContext> ctx,
                         const Rc<DxvkBuffer>& instanceBuffer,
                         const Rc<DxvkBuffer>& surfaceBuffer,
                         const Rc<DxvkBuffer>& surfaceMaterialBuffer,
                         const std::vector<PointInstancerBatch>& batches,
                         const Vector3& cameraPosition);

    /**
      * Displays ImGui settings for the point instancer culling system.
      */
    static void showImguiSettings();

    // -- Accessors for culling parameters (used by AccelManager) --

    static bool isEnabled()   { return enable(); }
    static float getCullingRadius()    { return cullingRadius(); }
    static float getFadeStartRadius()  { return fadeStartRadius(); }

  private:
    // -- RTX Options --------------------------------------------------------

    RTX_OPTION("rtx.pointInstancer", bool, enable, true,
      "Enables radius-based culling for USD PointInstancer replacements. "
      "When disabled, all instances are submitted to the TLAS regardless of distance.");

    static void onCullingRadiusChanged(DxvkDevice*) {
      // Ensure fadeStartRadius stays below cullingRadius
      fadeStartRadiusObject().setMaxValue(cullingRadius());
    }

    static void onFadeStartRadiusChanged(DxvkDevice*) {
      // Ensure cullingRadius stays above fadeStartRadius
      cullingRadiusObject().setMinValue(fadeStartRadius());
    }

    RTX_OPTION_ARGS("rtx.pointInstancer", float, cullingRadius, 20000.f,
      "Maximum distance (in world units) from the camera beyond which "
      "PointInstancer instances are culled. Instances farther than this "
      "distance are not included in the TLAS.",
      args.minValue = 0.f;
      args.onChangeCallback = onCullingRadiusChanged);

    RTX_OPTION_ARGS("rtx.pointInstancer", float, fadeStartRadius, 18000.f,
      "Distance (in world units) from the camera at which instances begin "
      "to be stochastically removed to create a smooth density falloff. "
      "Set to 0 to disable the fade region (hard culling boundary only). "
      "Must be less than cullingRadius.",
      args.minValue = 0.f;
      args.onChangeCallback = onFadeStartRadiusChanged);

    // -- GPU resources ------------------------------------------------------

    Rc<DxvkBuffer> m_cb;            // Per-dispatch constant buffer
    Rc<DxvkBuffer> m_transformsGpu; // Reused upload buffer for input transforms

    // NV-DXVK [SpawnGeomDiag.PIReadback]: host-visible copy of the TLAS
    // instance buffer. After each PI dispatch the instance buffer is copied
    // here; on the NEXT call the staged bytes are read back and logged so we
    // can see the actual VkAccelerationStructureInstanceKHR entries the GPU
    // culling shader produced for the mountain (scale-1000 o2w) batches.
    Rc<DxvkBuffer>        m_instReadbackStaging;
    bool                  m_instReadbackPending = false;
    std::vector<uint32_t> m_instReadbackMtnOffsets; // mountain-batch byte offsets in the staged copy

    // NV-DXVK [PIGpuWatch]: WHAT THE GPU ACTUALLY WROTE, diffed frame to frame.
    //
    // Everything CPU-side about these batches has been measured stable —
    // geometry hashes, BLAS handles, placements (jitter only), batch set — and
    // the flicker survived all of it. The last remaining variable is the
    // culling shader's own output: the VkAccelerationStructureInstanceKHR it
    // writes per instance. mask is the decisive field, because mask==0 is
    // exactly "this instance is not in the TLAS this frame". If groups of
    // instances flip mask between frames, that IS the flicker, observed at the
    // last stage before the TLAS rather than inferred from upstream state.
    //
    // The existing m_instReadbackMtnOffsets dump above answers a different
    // question (it prints raw entries for a hardcoded subset of mountain
    // batches, capped at 24 lines). This covers EVERY PI instance and reports
    // only transitions, which is what a per-frame comparison needs.
    struct PIReadbackRange {
      uint64_t batchKey;   // stable cross-frame identity, from the batch
      uint32_t byteOff;    // where this batch's entries live in the staged copy
      uint32_t count;
      uint32_t baseSurf;   // first surface slot, for the surface-buffer watch
      // CPU-side truth for the invariant checks below. Every change detector
      // run so far reports these stable, but STABLE IS NOT CORRECT — a batch
      // pointing at consistently wrong data looks perfectly clean to a diff.
      // These let the readback assert what the values SHOULD be.
      uint64_t expectBlasRef;
      uint32_t expectMask;
      uint32_t expectCustomFlags;
    };
    // Recorded at stage time, consumed on the NEXT call — the staged copy is
    // one dispatch behind, and byteOff moves every frame, so the offsets have
    // to travel with the frame they were captured in.
    std::vector<PIReadbackRange> m_piReadbackRanges;
    uint32_t                     m_piReadbackFrame = 0;

    struct PIGpuSlotState {
      uint64_t blasRef;
      uint64_t xformHash;   // over the 48 transform bytes
      float    tx, ty, tz;  // world translation, for readable deltas
      uint32_t mask;
      uint32_t lastFrame;
    };
    // Keyed on (batchKey, instanceIdx) — NOT on byte offset, which churns
    // every frame and would report every slot as changed.
    std::unordered_map<uint64_t, PIGpuSlotState> m_piGpuWatch;

    // NV-DXVK [PISurfWatch]: the PER-INSTANCE SURFACE the culling shader
    // writes at surfaceBuffer[baseSurfaceIndex + instanceIdx].
    //
    // [PIGpuWatch] established that every PI instance is in the TLAS with a
    // live mask on every frame (zeroMask=0 over 1754 frames), so the flicker
    // is not instances disappearing. The remaining way to get a
    // geometry-shaped artifact out of a present, unmasked instance is a wrong
    // or stale SURFACE — it carries the material index, texture params and
    // flags, so a bad one renders the instance black or wrong while it stays
    // in the TLAS. baseSurfaceIndex is also the one value already measured
    // churning every single frame.
    //
    // Deliberately NOT decoding named fields: the GPU layout is a sequential
    // writeGPUHelper append (rtx_materials.h ~:300), so hardcoding offsets
    // would silently rot. Instead the raw 256 bytes are diffed and the watch
    // reports WHICH 4-byte words changed, as a 64-bit mask. The shader patches
    // objectToWorld / prevObjectToWorld / normalObjectToWorld, so those words
    // will identify themselves by moving every frame; any word OUTSIDE that
    // set moving is the actual finding.
    Rc<DxvkBuffer> m_surfReadbackStaging;
    struct PISurfSlotState {
      uint32_t w[kSurfaceGPUDwords];
      uint32_t lastFrame;
    };
    std::unordered_map<uint64_t, PISurfSlotState> m_piSurfWatch;
  };
}
