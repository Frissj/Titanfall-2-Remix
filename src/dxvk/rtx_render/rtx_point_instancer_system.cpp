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
// NV-DXVK [PISurfWatch]: kSurfaceGPUSize. Reachable transitively through
// rtx_context.h -> rtx_options.h, but named explicitly because a static_assert
// here pins the surface stride and must not depend on an unrelated header
// keeping its includes.
#include "rtx_materials.h"

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
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_COVERAGE_BUFFER)
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
      const Vector3& cameraPosition,
      const Rc<DxvkBuffer>& coverageBuffer) {
    ScopedGpuProfileZone(ctx, "PointInstancerCulling");

    // NV-DXVK [PIWrite]: only record when the census that consumes it is on,
    // and only when the buffer actually exists — a null coverage buffer must
    // degrade to "no diagnostic", never to a skipped or broken dispatch.
    const bool writeCensus =
      RtxOptions::logResolveCensus() && coverageBuffer.ptr() != nullptr;

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

    // ========================================================================
    // NV-DXVK [PIGpuWatch] — diff the GPU culling shader's OWN OUTPUT.
    // ========================================================================
    // Reads the instance-buffer copy staged by the PREVIOUS dispatch and
    // compares every PI instance entry against what it was the frame before.
    // Change-only, so it is silent while the TLAS input is steady.
    //
    // Why this layer: [PIWatch] in AccelManager proved the CPU-side batch
    // descriptors stable (geoHash unchanged, mismatch=0, placementMax ~1/512
    // with the camera still, no period-2), and the flicker outlived the
    // surface-remap fix. The shader's written entry is the last thing between
    // that verified-good input and the TLAS.
    //
    // mask is the field that matters. mask==0 means the instance is skipped by
    // the RT hardware — invisible for that frame. A group of instances
    // flipping mask frame to frame is not evidence FOR the flicker, it IS the
    // flicker. zeroMask= in the summary is the raw per-frame count; watch that
    // column before reading anything else.
    if (RtxOptions::logGeomDiag()
        && !m_piReadbackRanges.empty()
        && m_instReadbackStaging.ptr() != nullptr) {
      const uint8_t* rb =
        reinterpret_cast<const uint8_t*>(m_instReadbackStaging->mapPtr(0));
      if (rb != nullptr) {
        const uint64_t rbSize = m_instReadbackStaging->info().size;

        static uint32_t sGpuWatchChangeLines = 0;
        constexpr uint32_t kGpuWatchChangeCap = 20000u;
        // Separate budget: an invariant violation must never be crowded out by
        // routine change lines, and it is the whole point of this pass.
        static uint32_t sGpuWatchBadLines = 0;
        constexpr uint32_t kGpuWatchBadCap = 8000u;

        uint32_t nSlots = 0, nMaskChg = 0, nBlasChg = 0, nXformChg = 0;
        uint32_t nZeroMask = 0, nNew = 0, nOob = 0;
        uint32_t nBadSurfIdx = 0, nBadBlas = 0, nBadMask = 0, nBadFlags = 0;
        // World-space AABB + centroid over every instance transform the shader
        // wrote this frame. See the comment at the accumulation site.
        float  wMinX =  3.0e38f, wMinY =  3.0e38f, wMinZ =  3.0e38f;
        float  wMaxX = -3.0e38f, wMaxY = -3.0e38f, wMaxZ = -3.0e38f;
        double wSumX = 0.0, wSumY = 0.0, wSumZ = 0.0;
        // Traversal-test operands + the any-hit gate. See the accumulation site.
        uint32_t maskOr = 0u, maskAnd = 0xFFu;
        uint32_t flagsOr = 0u, flagsAnd = 0xFFu;
        uint32_t sbtOffOr = 0u;

        for (const PIReadbackRange& r : m_piReadbackRanges) {
          for (uint32_t i = 0; i < r.count; ++i) {
            const uint64_t entryOff =
              static_cast<uint64_t>(r.byteOff) + static_cast<uint64_t>(i) * 64ull;
            if (entryOff + 64ull > rbSize) {
              ++nOob;
              continue;
            }
            const uint8_t* e = rb + entryOff;

            float t[12];
            std::memcpy(t, e, 48);
            uint32_t customIdxMask = 0, sbtAndFlags = 0, blasLo = 0, blasHi = 0;
            std::memcpy(&customIdxMask, e + 48, 4);
            // Bytes 52..55: instanceShaderBindingTableRecordOffset:24 | flags:8.
            // The flags byte carries VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT /
            // FORCE_NO_OPAQUE_BIT, which decide whether the ANY-HIT shader runs
            // at all — i.e. whether the alpha test can reject this instance.
            // These bytes were being read by the old mountain probe and thrown
            // away by every watch since.
            std::memcpy(&sbtAndFlags,   e + 52, 4);
            std::memcpy(&blasLo,        e + 56, 4);
            std::memcpy(&blasHi,        e + 60, 4);

            PIGpuSlotState c {};
            c.mask    = (customIdxMask >> 24) & 0xFFu;
            c.blasRef = (static_cast<uint64_t>(blasHi) << 32) | blasLo;

            // ---- INVARIANT CHECKS, not change detection ----
            // Every diff so far reported this data stable, and the flicker
            // survived all of them. Stability only proves nothing MOVES; it
            // says nothing about whether the values are RIGHT. These assert
            // the entry against the CPU-side truth for the same batch.
            //
            // surfaceIndex is the pointer from a TLAS hit to the surface the
            // hit shader reads (bits 0..20, CUSTOM_INDEX_SURFACE_MASK). It was
            // never checked — only the mask byte above it was. If it does not
            // equal baseSurfaceIndex + instanceIdx, the instance is present
            // and unmasked but reads ANOTHER object's surface, which renders
            // as the wrong thing while every presence metric stays green.
            // That is the one remaining shape that fits "geometry, not
            // shading" with a provably correct TLAS.
            const uint32_t entrySurfIdx  = customIdxMask & 0x1FFFFFu;
            const uint32_t expectSurfIdx = r.baseSurf + i;
            const uint32_t entryFlagBits = customIdxMask & 0x00E00000u;

            if (entrySurfIdx != expectSurfIdx) {
              ++nBadSurfIdx;
              if (sGpuWatchBadLines++ < kGpuWatchBadCap) {
                Logger::info(str::format(
                  "[PIGpuWatch.BADSURF] f=", m_piReadbackFrame,
                  " key=0x", std::hex, r.batchKey, std::dec,
                  " i=", i,
                  " entrySurfIdx=", entrySurfIdx,
                  " expected=", expectSurfIdx,
                  " delta=", int64_t(entrySurfIdx) - int64_t(expectSurfIdx),
                  " baseSurf=", r.baseSurf,
                  " count=", r.count,
                  " off=", entryOff));
              }
            }
            if (c.blasRef != r.expectBlasRef) {
              ++nBadBlas;
              if (sGpuWatchBadLines++ < kGpuWatchBadCap) {
                Logger::info(str::format(
                  "[PIGpuWatch.BADBLAS] f=", m_piReadbackFrame,
                  " key=0x", std::hex, r.batchKey,
                  " i=", std::dec, i,
                  " entryBlas=0x", std::hex, c.blasRef,
                  " expected=0x", r.expectBlasRef, std::dec));
              }
            }
            if (c.mask != r.expectMask) {
              ++nBadMask;
            }
            if (entryFlagBits != (r.expectCustomFlags & 0x00E00000u)) {
              ++nBadFlags;
            }
            c.tx = t[3]; c.ty = t[7]; c.tz = t[11];
            // FNV-1a over the raw transform bytes — a local hash keeps this
            // free of an xxHash include for what is only a change detector.
            uint64_t h = 1469598103934665603ull;
            for (uint32_t bIdx = 0; bIdx < 48; ++bIdx) {
              h = (h ^ static_cast<uint64_t>(e[bIdx])) * 1099511628211ull;
            }
            c.xformHash = h;
            c.lastFrame = m_piReadbackFrame;

            ++nSlots;
            if (c.mask == 0u) {
              ++nZeroMask;
            }

            // WHERE THE GEOMETRY ACTUALLY IS, per frame.
            //
            // The controlled comparison at 21:37 showed all 889 instances in
            // the TLAS, unmasked, with correct blasRef and surfaceIndex, on
            // 706 frames where the object rendered ZERO pixels — state
            // identical to frames where it rendered 116000. So presence is not
            // the variable. The only field tracked but never CHECKED is this
            // one: the world transform the shader wrote into the entry. It was
            // hashed for change detection and nothing more.
            //
            // Geometry that is present, unmasked, correctly referenced and
            // simply transformed somewhere off-camera renders 0 pixels with
            // every existing metric green — which is precisely the observation.
            // An AABB over the written translations answers it directly: if
            // these bounds jump between frames, the instances are being moved,
            // and that is the flicker. If they hold still while the pixel count
            // collapses, the geometry is in the right place and the fault is in
            // traversal or resolve, not in anything written here.
            if (c.tx < wMinX) wMinX = c.tx;
            if (c.ty < wMinY) wMinY = c.ty;
            if (c.tz < wMinZ) wMinZ = c.tz;
            if (c.tx > wMaxX) wMaxX = c.tx;
            if (c.ty > wMaxY) wMaxY = c.ty;
            if (c.tz > wMaxZ) wMaxZ = c.tz;
            wSumX += double(c.tx); wSumY += double(c.ty); wSumZ += double(c.tz);

            // THE TRAVERSAL TEST'S OPERANDS, and the alpha-test gate.
            //
            // maskOr/maskAnd: the instance mask byte actually written. It has
            // only ever been checked for AGREEING with CPU intent (badMask=0);
            // its VALUE was never read, and the value is what gets ANDed with
            // the ray's cullMask. A mask that never matches the primary ray's
            // cull mask rejects the geometry while every other metric stays
            // green. OR and AND together show both the union of bits in use and
            // whether every instance carries the same mask.
            //
            // geoFlagsOr/And: VkGeometryInstanceFlags. FORCE_OPAQUE means the
            // any-hit shader is skipped entirely, FORCE_NO_OPAQUE means it
            // always runs. For alpha-tested foliage this decides whether the
            // alpha test can reject the hit and let the ray pass through to the
            // sky behind — which is the leading explanation for trees vanishing
            // against a skybox with the geometry provably present in the AS.
            // If this byte differs between frames where the trees render and
            // frames where they do not, that is the bug.
            const uint32_t geoFlags = (sbtAndFlags >> 24) & 0xFFu;
            maskOr  |= c.mask;   maskAnd  &= c.mask;
            flagsOr |= geoFlags; flagsAnd &= geoFlags;
            sbtOffOr |= (sbtAndFlags & 0x00FFFFFFu);

            uint64_t slotKey = r.batchKey;
            slotKey = (slotKey * 0x100000001b3ull) ^ static_cast<uint64_t>(i);

            auto it = m_piGpuWatch.find(slotKey);
            if (it == m_piGpuWatch.end()) {
              ++nNew;
              m_piGpuWatch.emplace(slotKey, c);
              continue;
            }

            PIGpuSlotState& p = it->second;
            const bool maskChg  = (p.mask != c.mask);
            const bool blasChg  = (p.blasRef != c.blasRef);
            const bool xformChg = (p.xformHash != c.xformHash);

            if (maskChg)  ++nMaskChg;
            if (blasChg)  ++nBlasChg;
            if (xformChg) ++nXformChg;

            // A mask or BLAS transition is always worth a line. A transform
            // transition alone is not: the CPU-side placements were measured
            // jittering in the 7th significant digit, so those would flood.
            if ((maskChg || blasChg) && sGpuWatchChangeLines++ < kGpuWatchChangeCap) {
              Logger::info(str::format(
                "[PIGpuWatch.change] f=", m_piReadbackFrame,
                " key=0x", std::hex, r.batchKey, std::dec,
                " i=", i,
                " off=", entryOff,
                (maskChg ? str::format(" mask:", p.mask, "->", c.mask) : std::string()),
                (blasChg ? str::format(" blasRef:0x", std::hex, p.blasRef,
                                       "->0x", c.blasRef, std::dec) : std::string()),
                (xformChg ? " xformMoved" : ""),
                " t=(", c.tx, ",", c.ty, ",", c.tz, ")"));
            }

            p = c;
          }
        }

        // Every frame, no throttle — zeroMask= is the raw timeline the flicker
        // has to appear in if the shader is the source.
        Logger::info(str::format(
          "[PIGpuWatch] f=", m_piReadbackFrame,
          " ranges=", m_piReadbackRanges.size(),
          " slots=", nSlots,
          // INVARIANTS FIRST — these are pass/fail against CPU truth, unlike
          // everything after them, which only reports movement. Any nonzero
          // badSurfIdx is a hit reading the wrong object's surface.
          " badSurfIdx=", nBadSurfIdx,
          " badBlas=", nBadBlas,
          " badMask=", nBadMask,
          " badFlags=", nBadFlags,
          " zeroMask=", nZeroMask,
          // The two operands of the traversal test, plus the any-hit gate.
          // maskOr==maskAnd means every instance carries the same mask.
          // geoFlags bit0=TRIANGLE_FACING_CULL_DISABLE, bit1=FLIP_FACING,
          // bit2=FORCE_OPAQUE, bit3=FORCE_NO_OPAQUE.
          " maskOr=0x", std::hex, maskOr, std::dec,
          " maskAnd=0x", std::hex, (nSlots ? maskAnd : 0u), std::dec,
          " geoFlagsOr=0x", std::hex, flagsOr, std::dec,
          " geoFlagsAnd=0x", std::hex, (nSlots ? flagsAnd : 0u), std::dec,
          " sbtOffOr=", sbtOffOr,
          // WHERE the geometry is, in world space, as actually written to the
          // TLAS. Join this against the [Coverage] pixel count for the same
          // frame: if the box moves when the pixels vanish, the instances are
          // being relocated; if it holds still, they are in the right place and
          // the fault is downstream of the TLAS.
          " wMin=(", (nSlots ? wMinX : 0.f), ",", (nSlots ? wMinY : 0.f), ",", (nSlots ? wMinZ : 0.f), ")",
          " wMax=(", (nSlots ? wMaxX : 0.f), ",", (nSlots ? wMaxY : 0.f), ",", (nSlots ? wMaxZ : 0.f), ")",
          " wCen=(", (nSlots ? float(wSumX / nSlots) : 0.f), ",",
                     (nSlots ? float(wSumY / nSlots) : 0.f), ",",
                     (nSlots ? float(wSumZ / nSlots) : 0.f), ")",
          " maskChg=", nMaskChg,
          " blasChg=", nBlasChg,
          " xformChg=", nXformChg,
          " new=", nNew,
          " oob=", nOob,
          " tracked=", m_piGpuWatch.size(),
          (sGpuWatchChangeLines >= kGpuWatchChangeCap ? " *CHANGECAP*" : "")));
      }
    }
    // ========================================================================
    // NV-DXVK [PISurfWatch] — diff the PER-INSTANCE SURFACES the shader wrote.
    // ========================================================================
    // Same staging-and-diff technique as [PIGpuWatch] above, pointed at the
    // surface buffer instead of the instance buffer. See the member docs for
    // why this is the next target and why nothing is decoded by name.
    //
    // chgWordsOr is the whole readout: the OR, across every slot, of which
    // 4-byte words differ from last frame. The shader patches three transform
    // fields, so those words move constantly and will show up as a fixed
    // pattern. A bit appearing OUTSIDE that steady pattern means a surface's
    // non-transform content changed — material index, flags, texture params —
    // on an instance that never left the TLAS. That is the shape of the
    // remaining hypothesis.
    static_assert(kSurfaceGPUDwords * 4u == kSurfaceGPUSize,
                  "kSurfaceGPUDwords must track kSurfaceGPUSize");
    if (RtxOptions::logGeomDiag()
        && !m_piReadbackRanges.empty()
        && m_surfReadbackStaging.ptr() != nullptr) {
      const uint8_t* sb =
        reinterpret_cast<const uint8_t*>(m_surfReadbackStaging->mapPtr(0));
      if (sb != nullptr) {
        const uint64_t sbSize = m_surfReadbackStaging->info().size;

        static uint32_t sSurfWatchChangeLines = 0;
        constexpr uint32_t kSurfWatchChangeCap = 20000u;

        uint32_t nSlots = 0, nChgSlots = 0, nNew = 0, nOob = 0;
        uint64_t chgWordsOr = 0ull;

        for (const PIReadbackRange& r : m_piReadbackRanges) {
          for (uint32_t i = 0; i < r.count; ++i) {
            const uint64_t surfOff =
              (static_cast<uint64_t>(r.baseSurf) + i) * static_cast<uint64_t>(kSurfaceGPUSize);
            if (surfOff + kSurfaceGPUSize > sbSize) {
              ++nOob;
              continue;
            }

            PISurfSlotState c {};
            std::memcpy(c.w, sb + surfOff, kSurfaceGPUSize);
            c.lastFrame = m_piReadbackFrame;
            ++nSlots;

            uint64_t slotKey = r.batchKey;
            slotKey = (slotKey * 0x100000001b3ull) ^ static_cast<uint64_t>(i);

            auto it = m_piSurfWatch.find(slotKey);
            if (it == m_piSurfWatch.end()) {
              ++nNew;
              m_piSurfWatch.emplace(slotKey, c);
              continue;
            }

            PISurfSlotState& p = it->second;
            uint64_t mask = 0ull;
            for (uint32_t wIdx = 0; wIdx < kSurfaceGPUDwords; ++wIdx) {
              if (p.w[wIdx] != c.w[wIdx]) {
                mask |= (1ull << wIdx);
              }
            }

            if (mask != 0ull) {
              ++nChgSlots;
              chgWordsOr |= mask;
              if (sSurfWatchChangeLines++ < kSurfWatchChangeCap) {
                Logger::info(str::format(
                  "[PISurfWatch.change] f=", m_piReadbackFrame,
                  " key=0x", std::hex, r.batchKey, std::dec,
                  " i=", i,
                  " surf=", (r.baseSurf + i),
                  " chgWords=0x", std::hex, mask, std::dec));
              }
            }

            p = c;
          }
        }

        Logger::info(str::format(
          "[PISurfWatch] f=", m_piReadbackFrame,
          " slots=", nSlots,
          " chgSlots=", nChgSlots,
          " chgWordsOr=0x", std::hex, chgWordsOr, std::dec,
          " new=", nNew,
          " oob=", nOob,
          " tracked=", m_piSurfWatch.size(),
          (sSurfWatchChangeLines >= kSurfWatchChangeCap ? " *CHANGECAP*" : "")));
      }
    }

    m_piReadbackRanges.clear();

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
      constants.enableWriteCensus = writeCensus ? 1u : 0u;

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

      // NV-DXVK [PIGpuWatch]: record where this batch's entries will land, with
      // the batch's stable identity, so the NEXT call can attribute the staged
      // bytes to a logical (batch, instanceIdx) rather than to a byte offset —
      // byteOff churns every frame and would report every slot as changed.
      if (RtxOptions::logGeomDiag()) {
        m_piReadbackRanges.push_back(
          PIReadbackRange { batch.debugStableKey, batch.instanceBufferByteOffset,
                            count, batch.baseSurfaceIndex,
                            batch.blasReference, batch.instanceMask,
                            batch.customIndexFlags });
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
      // Bound unconditionally — the descriptor slot is part of the pipeline
      // layout whether or not the census is on, so leaving it unbound would be
      // an invalid descriptor rather than a disabled feature. When the buffer
      // is null this binds an empty slice and the shader's gate keeps it unused.
      ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_COVERAGE_BUFFER,
        coverageBuffer.ptr() != nullptr ? DxvkBufferSlice(coverageBuffer) : DxvkBufferSlice());

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
    // [PIGpuWatch] shares this one staged copy — it must be taken if EITHER
    // consumer wants it, otherwise enabling the general watch would silently
    // depend on a mountain batch happening to be present.
    if (inGameplay
        && (!m_instReadbackMtnOffsets.empty() || !m_piReadbackRanges.empty())
        && instanceBuffer.ptr() != nullptr) {
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

      // NV-DXVK [PISurfWatch]: stage the surface buffer alongside the instance
      // buffer, from the same point after all dispatches, so both views
      // describe the same frame.
      if (RtxOptions::logGeomDiag()
          && !m_piReadbackRanges.empty()
          && surfaceBuffer.ptr() != nullptr) {
        const VkDeviceSize surfCopySize = surfaceBuffer->info().size;
        if (m_surfReadbackStaging.ptr() == nullptr
            || m_surfReadbackStaging->info().size < surfCopySize) {
          DxvkBufferCreateInfo info;
          info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size   = surfCopySize;
          m_surfReadbackStaging = dev->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "RTX PointInstancer - Surface Readback Staging");
        }
        ctx->copyBuffer(m_surfReadbackStaging, 0, surfaceBuffer, 0, surfCopySize);
      }
      // Stamp the frame these ranges belong to, so next call's [PIGpuWatch]
      // lines carry the frame the GPU actually wrote them on rather than the
      // frame they were read on.
      m_piReadbackFrame = dev->getCurrentFrameId();
    } else {
      // No copy taken — the ranges recorded above describe bytes that will
      // never be staged, so drop them rather than diffing this frame's offsets
      // against a stale buffer next call.
      m_piReadbackRanges.clear();
    }
  }
}
