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
  };
}
