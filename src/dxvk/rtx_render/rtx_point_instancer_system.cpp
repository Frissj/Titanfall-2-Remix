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
#include "rtx_point_instancer_system.h"
#include "rtx_debug_probes.h"

#include "dxvk_device.h"
#include "rtx_render/rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_imgui.h"

#include <rtx_shaders/point_instancer_culling.h>

namespace dxvk {

  // Forward decl for the engine-hook main-cam capture counter (defined in
  // rtx_camera_manager.cpp at dxvk::tf2 scope). Used as the gameplay-active
  // gate for the [SpawnGeomDiag.PIReadback] full-instance-buffer readback,
  // which copies the entire VkAccelerationStructureInstanceKHR buffer back
  // to host-visible memory every PI dispatch — a measurable GPU/CPU stall.
  // The dispatch itself (game-critical culling) must still run; only the
  // diagnostic readback is gated.
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;
  }


  namespace {
    class PointInstancerCullingShader : public ManagedShader {
      SHADER_SOURCE(PointInstancerCullingShader, VK_SHADER_STAGE_COMPUTE_BIT, point_instancer_culling)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(POINT_INSTANCER_CULLING_BINDING_CONSTANTS)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER)
      END_PARAMETER()
    };
  }

  RtxPointInstancerSystem::RtxPointInstancerSystem(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  void RtxPointInstancerSystem::showImguiSettings() {
    if (RemixGui::CollapsingHeader("Point Instancer Culling")) {
      ImGui::PushID("rtx_point_instancer");
      ImGui::Dummy({ 0, 2 });
      ImGui::Indent();

      RemixGui::Checkbox("Enable Culling", &enableObject());
      ImGui::BeginDisabled(!enable());

      RemixGui::DragFloat("Culling Radius", &cullingRadiusObject(), 10.f, fadeStartRadius(), 100000.f, "%.0f");
      RemixGui::DragFloat("Fade Start Radius", &fadeStartRadiusObject(), 10.f, 0.f, cullingRadius(), "%.0f");

      ImGui::EndDisabled();
      ImGui::Unindent();
      ImGui::PopID();
    }
  }

  void RtxPointInstancerSystem::dispatchCulling(
      Rc<DxvkContext> ctx,
      const Rc<DxvkBuffer>& instanceBuffer,
      const Rc<DxvkBuffer>& surfaceBuffer,
      const Rc<DxvkBuffer>& surfaceMaterialBuffer,
      const std::vector<PointInstancerBatch>& batches,
      const Vector3& cameraPosition) {
    ScopedGpuProfileZone(ctx, "PointInstancerCulling");

    if (batches.empty()) {
      return;
    }

    // NV-DXVK [SpawnGeomDiag.PIReadback gameplay gate]: the readback block
    // below copies the whole instance buffer GPU→host every PI dispatch
    // (see the stage-copy at the bottom of this function). That's a real
    // perf hit, not just log noise. Skip it during menu/loading frames —
    // the engine-hook capture counter is the same signal used by
    // rtx_instance_manager.cpp:648 and rtx_scene_manager.cpp:207 to mean
    // "we're actually in gameplay, not the title screen".
    const bool inGameplay =
      tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;

    // NV-DXVK [SpawnGeomDiag.PIReadback]: read back the instance-buffer bytes
    // staged by the PREVIOUS dispatch and log the actual
    // VkAccelerationStructureInstanceKHR entries the GPU culling shader
    // produced for the mountain (scale-1000 o2w) batches. This is the GPU
    // shader's true output — mask, BLAS ref, and world transform — which
    // determines whether the instance is live in the TLAS.
    if (inGameplay && m_instReadbackPending && m_instReadbackStaging.ptr() != nullptr) {
      const uint8_t* rb =
        reinterpret_cast<const uint8_t*>(m_instReadbackStaging->mapPtr(0));
      if (rb != nullptr) {
        const uint64_t rbSize = m_instReadbackStaging->info().size;
        uint32_t logged = 0;
        for (uint32_t off : m_instReadbackMtnOffsets) {
          if (logged >= 24u) break;
          if (static_cast<uint64_t>(off) + 64ull > rbSize) continue;
          const uint8_t* e = rb + off;
          float t[12];
          std::memcpy(t, e, 48);
          uint32_t customIdxMask = 0, sbtFlags = 0, blasLo = 0, blasHi = 0;
          std::memcpy(&customIdxMask, e + 48, 4);
          std::memcpy(&sbtFlags,      e + 52, 4);
          std::memcpy(&blasLo,        e + 56, 4);
          std::memcpy(&blasHi,        e + 60, 4);
          const uint32_t mask = (customIdxMask >> 24) & 0xFFu;
          const uint64_t blasRef = (static_cast<uint64_t>(blasHi) << 32) | blasLo;
          Logger::info(str::format(
            "[SpawnGeomDiag.PIReadback] off=", off,
            " mask=0x", std::hex, mask, std::dec,
            " sbtFlags=0x", std::hex, sbtFlags, std::dec,
            " blasRef=0x", std::hex, blasRef, std::dec,
            " worldT=(", t[3], ",", t[7], ",", t[11], ")",
            " row0=(", t[0], ",", t[1], ",", t[2], ")"));
          ++logged;
        }
      }
      m_instReadbackPending = false;
    }
    m_instReadbackMtnOffsets.clear();

    // NV-DXVK debug: throttled per-batch dump of CB inputs + first-instance transform.
    // Helps diagnose why most PI instances are invisible — see what the shader is actually reading.
    // [SpawnGeomDiag] renamed from [PI-dump] to bypass log.cpp filter.
    // Tightened from kEnableRtxDebugProbes-gated to always-on (the probes
    // constant is true in this build, but renaming makes intent explicit).
    // NV-DXVK: dump every 8 dispatches. At ~3 fps the old %60 throttle
    // produced only one (cold-start) dump before a run ended, hiding the
    // steady-state sbt/flags of the mountain batches.
    static uint32_t s_dumpFrame = 0;
    const bool doDump = ((s_dumpFrame++ % 8u) == 0);
    if (doDump) {
      // [SpawnGeomDiag.PIdump] device frame stamp lets us cross-reference
      // each batch's surfRange with [SpawnGeomDiag.VisibleSurf] from the
      // SAME frame. PIdump's own counter (call=) is the dispatch index
      // since session start; the device frame is what VisibleSurf uses.
      const uint32_t devFid =
        (ctx->getDevice() != nullptr) ? ctx->getDevice()->getCurrentFrameId() : 0u;
      Logger::info(str::format(
        "[SpawnGeomDiag.PIdump] devFrame=", devFid,
        " call=", s_dumpFrame,
        " batches=", batches.size(),
        " camPos=(", cameraPosition.x, ",", cameraPosition.y, ",", cameraPosition.z, ")"));
      for (size_t bi = 0; bi < batches.size(); ++bi) {
        const PointInstancerBatch& b = batches[bi];
        const auto& m = b.objectToWorld;
        const Vector3 t  = m.data[3].xyz();
        // [SpawnGeomDiag] Need the full 3x3 rotation block of o2w to
        // diagnose floor-visibility issues that depend on whether O is
        // truly identity-rotation. Matrix4 stores Vector4 columns; o2w
        // rotation lives in cols 0..2, rows 0..2 (= data[c].x/y/z).
        const Vector3 oCol0 = m.data[0].xyz();
        const Vector3 oCol1 = m.data[1].xyz();
        const Vector3 oCol2 = m.data[2].xyz();
        // First-instance i2o full matrix: rotation cols 0..2 + translation col 3.
        Vector3 i0t(0.f, 0.f, 0.f);
        Vector3 i0c0(0.f, 0.f, 0.f);
        Vector3 i0c1(0.f, 0.f, 0.f);
        Vector3 i0c2(0.f, 0.f, 0.f);
        if (b.transforms && !b.transforms->empty()) {
          const Matrix4& I0 = (*b.transforms)[0];
          i0t  = I0.data[3].xyz();
          i0c0 = I0.data[0].xyz();
          i0c1 = I0.data[1].xyz();
          i0c2 = I0.data[2].xyz();
        }
        // Surface-ID range covered by this batch
        const uint32_t surfFirst = b.baseSurfaceIndex;
        const uint32_t surfLast  = b.baseSurfaceIndex + b.instanceCount - 1;
        Logger::info(str::format(
          "[SpawnGeomDiag.PIdump]  batch=", bi,
          " surfRange=[", surfFirst, "..", surfLast, "]",
          " count=", b.instanceCount,
          " firstIdxInType=", b.firstIndexInType,
          " bufByteOff=", b.instanceBufferByteOffset,
          " tlas=", uint32_t(b.tlasType),
          " mask=0x", std::hex, b.instanceMask,
          " ciFlags=0x", b.customIndexFlags,
          " sbt=0x", b.sbtOffsetAndFlags, std::dec,
          " o2w.T=(", t.x, ",", t.y, ",", t.z, ")",
          " o2w.col0=(", oCol0.x, ",", oCol0.y, ",", oCol0.z, ")",
          " o2w.col1=(", oCol1.x, ",", oCol1.y, ",", oCol1.z, ")",
          " o2w.col2=(", oCol2.x, ",", oCol2.y, ",", oCol2.z, ")",
          " i2o[0].T=(", i0t.x, ",", i0t.y, ",", i0t.z, ")",
          " i2o[0].col0=(", i0c0.x, ",", i0c0.y, ",", i0c0.z, ")",
          " i2o[0].col1=(", i0c1.x, ",", i0c1.y, ",", i0c1.z, ")",
          " i2o[0].col2=(", i0c2.x, ",", i0c2.y, ",", i0c2.z, ")",
          " blasRef=0x", std::hex, b.blasReference, std::dec));
      }
    }

    const Rc<DxvkDevice>& dev = ctx->getDevice();

    // Allocate constant buffer (once)
    if (m_cb.ptr() == nullptr) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = sizeof(PointInstancerCullingConstants);
      m_cb = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               DxvkMemoryStats::Category::RTXBuffer,
                               "RTX PointInstancer - Constant Buffer");
    }

    for (const PointInstancerBatch& batch : batches) {
      const uint32_t count = batch.instanceCount;
      if (count == 0) {
        continue;
      }

      // Upload source transforms to GPU
      const size_t transformsSize = count * sizeof(Matrix4);
      if (m_transformsGpu.ptr() == nullptr || m_transformsGpu->info().size < transformsSize) {
        DxvkBufferCreateInfo info;
        info.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
        info.size   = align(transformsSize, 256);
        m_transformsGpu = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                            DxvkMemoryStats::Category::RTXBuffer,
                                            "RTX PointInstancer - Transforms Input");
      }

      ctx->writeToBuffer(m_transformsGpu, 0, transformsSize, batch.transforms->data());

      // Fill constant buffer
      // NV-DXVK (debug): force cull-disable. The conf-driven `enable()` accessor
      // is returning its hardcoded default (true) instead of the parsed config
      // value (false), even though `Effective Combined Config` confirms the
      // value reaches the option layer. Forcing here guarantees BSP doesn't
      // get mask=0'd by distance cull while we figure out why.
      const bool cullingEnabled = false; // was: enable();
      PointInstancerCullingConstants constants {};
      memcpy(&constants.objectToWorld, &batch.objectToWorld, sizeof(mat4));
      memcpy(&constants.prevObjectToWorld, &batch.prevObjectToWorld, sizeof(mat4));
      constants.cameraPosition    = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
      constants.cullingRadius     = cullingEnabled ? cullingRadius() : FLT_MAX;
      constants.totalInstanceCount = count;
      constants.baseSurfaceIndex  = batch.baseSurfaceIndex;
      constants.fadeStartRadius   = cullingEnabled ? fadeStartRadius() : 0.f;
      constants.customIndexFlags  = batch.customIndexFlags;
      constants.instanceMask      = batch.instanceMask;
      constants.sbtOffsetAndFlags = batch.sbtOffsetAndFlags;
      constants.blasRefLo         = static_cast<uint32_t>(batch.blasReference & 0xFFFFFFFFull);
      constants.blasRefHi         = static_cast<uint32_t>(batch.blasReference >> 32);
      constants.instanceBufferOffset = batch.instanceBufferByteOffset;

      // NV-DXVK [SpawnGeomDiag.PIWrite]: for sub-view-reprojected (mountain)
      // batches — identified by a scale-1000 objectToWorld, vs identity for
      // every normal PI prop — log the exact instance-buffer write extent and
      // bounds. If a mountain batch writes past instanceBuffer->size the
      // culling shader silently corrupts/drops it. Throttled per-call.
      {
        const bool isReprojBatch = std::abs(batch.objectToWorld.data[0].x) > 100.0f;
        if (isReprojBatch) {
          // Only record offsets when gameplay is active — if we record them
          // during menu, the stage-copy block at the bottom would still fire
          // even with the consumer above gated, doing a useless GPU→host
          // copy of the entire instance buffer every dispatch.
          if (inGameplay) {
            m_instReadbackMtnOffsets.push_back(batch.instanceBufferByteOffset);
          }
          static uint32_t sPIWriteLog = 0;
          if (inGameplay && (sPIWriteLog < 64u || (sPIWriteLog % 256u) == 0u)) {
            const uint64_t writeBegin = batch.instanceBufferByteOffset;
            const uint64_t writeEnd   = writeBegin
              + static_cast<uint64_t>(count) * 64ull;
            Logger::info(str::format(
              "[SpawnGeomDiag.PIWrite] n=", sPIWriteLog,
              " count=", count,
              " o2wScale=", batch.objectToWorld.data[0].x,
              " instBufOff=", writeBegin,
              " writeEnd=", writeEnd,
              " instBufSize=", instanceBuffer->info().size,
              " inBounds=", (writeEnd <= instanceBuffer->info().size ? 1 : 0),
              " baseSurfIdx=", batch.baseSurfaceIndex,
              " lastSurfIdx=", (batch.baseSurfaceIndex + count - 1),
              " surfBufSize=", surfaceBuffer->info().size,
              " tlas=", uint32_t(batch.tlasType),
              " mask=0x", std::hex, batch.instanceMask, std::dec,
              " blasRef=0x", std::hex, batch.blasReference, std::dec));
          }
          sPIWriteLog += 1;
        }
      }

      const DxvkBufferSliceHandle cSlice = m_cb->allocSlice();
      ctx->invalidateBuffer(m_cb, cSlice);
      ctx->writeToBuffer(m_cb, 0, sizeof(PointInstancerCullingConstants), &constants);

      // Bind resources
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT, DxvkBufferSlice(m_transformsGpu));
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER, DxvkBufferSlice(instanceBuffer));
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER, DxvkBufferSlice(surfaceBuffer));
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER, DxvkBufferSlice(surfaceMaterialBuffer));

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PointInstancerCullingShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { count, 1, 1 },
        VkExtent3D { 64, 1, 1 });

      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    // NV-DXVK [SpawnGeomDiag.PIReadback]: stage a host-visible copy of the
    // instance buffer AFTER all dispatches this call, so the next call can
    // read back what the GPU culling shader actually wrote (see member docs).
    // Belt-and-braces gameplay gate — m_instReadbackMtnOffsets is also only
    // populated in gameplay (see the inGameplay check at the push_back site),
    // so this branch already short-circuits in menu via the .empty() test;
    // the explicit gate makes the intent clear if someone re-enables menu
    // pushes later.
    if (inGameplay && !m_instReadbackMtnOffsets.empty() && instanceBuffer.ptr() != nullptr) {
      const VkDeviceSize copySize = instanceBuffer->info().size;
      if (m_instReadbackStaging.ptr() == nullptr
          || m_instReadbackStaging->info().size < copySize) {
        DxvkBufferCreateInfo info;
        info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
        info.size   = copySize;
        m_instReadbackStaging = dev->createBuffer(info,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          DxvkMemoryStats::Category::RTXBuffer,
          "RTX PointInstancer - Inst Readback Staging");
      }
      ctx->copyBuffer(m_instReadbackStaging, 0, instanceBuffer, 0, copySize);
      m_instReadbackPending = true;
    }
  }
}
