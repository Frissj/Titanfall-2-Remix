/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_auto_exposure_plus.h"
#include "rtx_auto_exposure.h"
#include "rtx_tone_mapping.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx/pass/auto_exposure_plus/auto_exposure_plus.h"
#include "rtx_debug_view.h"

#include <rtx_shaders/auto_exposure_plus_init.h>
#include <rtx_shaders/auto_exposure_plus_downsample.h>
#include <rtx_shaders/auto_exposure_plus_fuse.h>
#include <rtx_shaders/auto_exposure_plus_temporal.h>
#include <rtx_shaders/auto_exposure_plus_apply.h>
#include "rtx_imgui.h"
#include "rtx/utility/debug_view_indices.h"

#include "../util/log/log.h"
#include "../util/util_string.h"

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {

    class AutoExposurePlusInitShader : public ManagedShader {
      SHADER_SOURCE(AutoExposurePlusInitShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_plus_init)

      PUSH_CONSTANTS(AutoExposurePlusInitArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(AUTO_EXPOSURE_PLUS_INIT_COLOR_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_INIT_PYRAMID_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_PLUS_INIT_EXPOSURE)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AutoExposurePlusInitShader);

    class AutoExposurePlusDownsampleShader : public ManagedShader {
      SHADER_SOURCE(AutoExposurePlusDownsampleShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_plus_downsample)

      PUSH_CONSTANTS(AutoExposurePlusDownsampleArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AutoExposurePlusDownsampleShader);

    class AutoExposurePlusFuseShader : public ManagedShader {
      SHADER_SOURCE(AutoExposurePlusFuseShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_plus_fuse)

      PUSH_CONSTANTS(AutoExposurePlusFuseArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(AUTO_EXPOSURE_PLUS_FUSE_PYRAMID_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_PLUS_FUSE_COLOR_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_FUSE_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_PLUS_FUSE_EXPOSURE)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AutoExposurePlusFuseShader);

    class AutoExposurePlusTemporalShader : public ManagedShader {
      SHADER_SOURCE(AutoExposurePlusTemporalShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_plus_temporal)

      PUSH_CONSTANTS(AutoExposurePlusTemporalArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(AUTO_EXPOSURE_PLUS_TEMPORAL_CURRENT_INPUT)
        SAMPLER2D(AUTO_EXPOSURE_PLUS_TEMPORAL_HISTORY_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_PLUS_TEMPORAL_MOTION_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_TEMPORAL_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AutoExposurePlusTemporalShader);

    class AutoExposurePlusApplyShader : public ManagedShader {
      SHADER_SOURCE(AutoExposurePlusApplyShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_plus_apply)

      PUSH_CONSTANTS(AutoExposurePlusApplyArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(AUTO_EXPOSURE_PLUS_APPLY_FUSED_INPUT)
        SAMPLER2D(AUTO_EXPOSURE_PLUS_APPLY_PYRAMID_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_APPLY_COLOR_INPUT_OUTPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_PLUS_APPLY_DEBUG_VIEW_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_PLUS_APPLY_EXPOSURE)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AutoExposurePlusApplyShader);

    float safeEVLog2(float v) {
      return log2f(std::max(1e-10f, v));
    }
  }

  DxvkAutoExposurePlus::DxvkAutoExposurePlus(DxvkDevice* device)
  : RtxPass(device) {
  }

  DxvkAutoExposurePlus::~DxvkAutoExposurePlus() { }

  void DxvkAutoExposurePlus::showImguiSettings() {
    // Plus and the local tonemapper are both local dynamic range compressors. Running them in
    // series over-flattens the image, and the symptom (washed out, low contrast) reads as "Plus
    // is too strong", so it is easy to chase by lowering intensity instead of fixing the pair.
    const bool operatorSelected =
      DxvkToneMapping::tonemapOperator() != DxvkToneMapping::TonemapOperator::None;
    const bool localWouldRun =
      !operatorSelected && RtxOptions::tonemappingMode() == TonemappingMode::Local;

    RemixGui::Checkbox("Force Global Tonemapper", &forceGlobalTonemapperObject());

    if (localWouldRun && !forceGlobalTonemapper()) {
      ImGui::TextWrapped(
        "WARNING: Tonemapping Mode is Local, which is also a local operator - stacking it with "
        "Auto Exposure Plus flattens the image twice.");
    }

    RemixGui::Separator();

    RemixGui::DragFloat("Exposure Level", &exposureObject(), 0.01f, 0.f, 1000.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Intensity", &intensityObject(), 0.01f, -1.f, 1.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Equalization Strength", &equalizationStrengthObject(), 0.01f, 0.f, 1.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Equalization Pivot (EV)", &equalizationPivotObject(), 0.05f, 0.f, 20.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Target Brightness (EV)", &targetEVObject(), 0.01f, -3.f, 3.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Max Gain (EV)", &maxGainEVObject(), 0.05f, 0.f, 16.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragInt("Pyramid Levels (0 = auto)", &levelsObject(), 0.06f, 0, 12);

    if (RemixGui::CollapsingHeader("Temporal##AutoExposurePlus", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();
      RemixGui::Checkbox("Temporal Accumulation", &temporalAccumulationObject());
      RemixGui::Checkbox("Match Base Adaptation Speed", &matchBaseAdaptationSpeedObject());
      ImGui::BeginDisabled(matchBaseAdaptationSpeed());
      RemixGui::DragFloat("Adaptation Speed", &adaptationSpeedObject(), 0.01f, 0.f, 100.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      ImGui::EndDisabled();
      ImGui::Unindent();
    }

    if (RemixGui::CollapsingHeader("Diagnostics##AutoExposurePlus")) {
      ImGui::Indent();
      RemixGui::DragInt("Debug Pyramid Level", &debugPyramidLevelObject(), 0.06f, 0, 12);
      RemixGui::Checkbox("Log GPU Timings", &perfLoggingObject());
      RemixGui::DragInt("Log Interval (frames)", &perfLogIntervalFramesObject(), 1.0f, 30, 10000);
      ImGui::Unindent();
    }
  }

  bool DxvkAutoExposurePlus::forcesGlobalTonemapper() {
    return DxvkAutoExposure::mode() == DxvkAutoExposure::AutoExposureMode::Plus
        && forceGlobalTonemapper();
  }

  bool DxvkAutoExposurePlus::isEnabled() const {
    // No interaction with the tonemapper selection: this pass only rescales linear radiance, so
    // it composes with every operator rather than replacing any of them.
    return DxvkAutoExposure::mode() == DxvkAutoExposure::AutoExposureMode::Plus;
  }

  VkExtent3D DxvkAutoExposurePlus::calcPyramidExtent(const VkExtent3D& targetExtent) {
    return VkExtent3D {
      std::max(1u, (targetExtent.width + 1) / 2),
      std::max(1u, (targetExtent.height + 1) / 2),
      1u
    };
  }

  uint32_t DxvkAutoExposurePlus::calcLevelCount(const VkExtent3D& targetExtent) {
    const VkExtent3D pyramidExtent = calcPyramidExtent(targetExtent);
    const uint32_t minDimension = std::max(1u, std::min(pyramidExtent.width, pyramidExtent.height));

    // Stop once the coarsest level is around 8 texels on its short side - below that a level
    // covers the whole frame and carries no spatial information worth a dispatch pair. The -2
    // is that 8-texel floor expressed in mips (540 -> 7 levels, 1080 -> 8 levels).
    const int levelCount = static_cast<int>(std::floor(std::log2(static_cast<float>(minDimension)))) - 2;

    // Never fewer than 2 levels, otherwise there is nothing to fuse. The Vulkan mip count for
    // the image is the hard ceiling.
    const uint32_t maxLevels = static_cast<uint32_t>(
      std::floor(std::log2(static_cast<float>(std::max(pyramidExtent.width, pyramidExtent.height))))) + 1;

    return std::min(maxLevels, static_cast<uint32_t>(std::max(2, levelCount)));
  }

  void DxvkAutoExposurePlus::beginTimerFrame(Rc<RtxContext> ctx) {
    if (m_timers.pool == nullptr) {
      m_timers.pool = new DxvkDLFGTimestampQueryPool(
        ctx->getDevice().ptr(), StageTimers::kMarks * StageTimers::kFrames);
    }

    // Discard the ring on the first frame after logging is switched back on. The slots still
    // hold whatever was written the last time it ran, which would resolve to the length of the
    // gap between then and now rather than to a frame's worth of work.
    if (!m_timers.wasLogging) {
      m_timers.wasLogging = true;

      for (uint32_t i = 0; i < StageTimers::kFrames; ++i) {
        m_timers.markCount[i] = 0;
      }
      for (uint32_t i = 0; i < StageTimers::kStages; ++i) {
        m_timers.accumMs[i] = 0.0;
      }
      m_timers.accumTotalMs = 0.0;
      m_timers.samples = 0;
      m_timers.framesSinceReport = 0;
    }

    m_timers.frameSlot = (m_timers.frameSlot + 1u) % StageTimers::kFrames;
    m_timers.writeCount = 0;
    // markCount is otherwise only written by markTimer, so a frame that wrote no marks at all
    // would leave a stale count pointing at slot indices from several frames ago.
    m_timers.markCount[m_timers.frameSlot] = 0;

    // Resolve the oldest frame in the ring before this frame overwrites its own slot.
    const uint32_t readSlot = (m_timers.frameSlot + 1u) % StageTimers::kFrames;
    const uint32_t marks = m_timers.markCount[readSlot];

    if (marks < StageTimers::kMarks) {
      return;
    }

    const double nsPerTick = ctx->getDevice()->adapter()->deviceProperties().limits.timestampPeriod;

    uint64_t ts[StageTimers::kMarks] = {};
    bool ok = true;
    for (uint32_t i = 0; i < marks && ok; ++i) {
      ok = m_timers.pool->readTimestamp(&ts[i], m_timers.slotIndex[readSlot][i]);
    }

    if (!ok) {
      return;
    }

    for (uint32_t i = 1; i < marks; ++i) {
      // Timestamps are monotonic within a queue; guard anyway so a wrap cannot poison the
      // accumulator.
      const uint64_t delta = (ts[i] >= ts[i - 1]) ? (ts[i] - ts[i - 1]) : 0ull;
      m_timers.accumMs[i - 1] += double(delta) * nsPerTick / 1.0e6;
    }

    const uint64_t total = (ts[marks - 1] >= ts[0]) ? (ts[marks - 1] - ts[0]) : 0ull;
    m_timers.accumTotalMs += double(total) * nsPerTick / 1.0e6;
    ++m_timers.samples;
  }

  void DxvkAutoExposurePlus::markTimer(Rc<RtxContext> ctx) {
    if (m_timers.pool == nullptr || m_timers.writeCount >= StageTimers::kMarks) {
      return;
    }

    // Force prior work to complete before stamping, so each interval measures execution rather
    // than submission order. Without this a BOTTOM_OF_PIPE stamp lands early on compute
    // dispatches and the cost shows up in a later, often empty, interval.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

    const uint32_t index = m_timers.pool->writeTimestamp(
      ctx->getCmdBuffer(DxvkCmdBuffer::ExecBuffer), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    m_timers.slotIndex[m_timers.frameSlot][m_timers.writeCount] = index;
    ++m_timers.writeCount;
    m_timers.markCount[m_timers.frameSlot] = m_timers.writeCount;
  }

  void DxvkAutoExposurePlus::reportTimers(const VkExtent3D& pyramidExtent, uint32_t levelCount) {
    ++m_timers.framesSinceReport;

    const uint32_t interval = static_cast<uint32_t>(std::max(30, perfLogIntervalFrames()));

    if (m_timers.framesSinceReport < interval || m_timers.samples == 0) {
      return;
    }

    const double inv = 1.0 / double(m_timers.samples);

    Logger::info(str::format(
      "[Perf.AutoExpPlus] pyr=", pyramidExtent.width, "x", pyramidExtent.height,
      " levels=", levelCount,
      " total=", m_timers.accumTotalMs * inv, "ms",
      " | init=", m_timers.accumMs[0] * inv,
      " downsample=", m_timers.accumMs[1] * inv,
      " fuse=", m_timers.accumMs[2] * inv,
      " temporal=", m_timers.accumMs[3] * inv,
      " apply=", m_timers.accumMs[4] * inv,
      " (n=", m_timers.samples, ")"));

    for (uint32_t i = 0; i < StageTimers::kStages; ++i) {
      m_timers.accumMs[i] = 0.0;
    }
    m_timers.accumTotalMs = 0.0;
    m_timers.samples = 0;
    m_timers.framesSinceReport = 0;
  }

  void DxvkAutoExposurePlus::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds,
    bool resetHistory,
    bool enableAutoExposure) {

    if (m_pyramid.views.size() == 0 || m_fused.image == nullptr) {
      return;
    }

    ScopedGpuProfileZone(ctx, "Auto Exposure Plus");
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    const bool timing = perfLogging();

    if (timing) {
      beginTimerFrame(ctx);
    } else {
      m_timers.wasLogging = false;
    }

    const VkExtent3D finalResolution = rtOutput.m_finalOutputExtent;
    const VkExtent3D pyramidExtent = m_pyramid.image->mipLevelExtent(0);

    // The level count is a runtime option, but the allocated pyramid is the ceiling.
    const uint32_t requestedLevels = levels() > 0 ? static_cast<uint32_t>(levels()) : m_levelCount;
    const uint32_t levelCount = std::max(2u, std::min(requestedLevels, static_cast<uint32_t>(m_pyramid.views.size())));

    const float exposureFactor = exp2f(safeEVLog2(exposure()) + RtxOptions::calcUserEVBias());

    const VkExtent3D pyramidWorkgroups = util::computeBlockCount(pyramidExtent, VkExtent3D { 16, 16, 1 });
    const VkExtent3D finalWorkgroups = util::computeBlockCount(finalResolution, VkExtent3D { 16, 16, 1 });

    DebugView& debugView = ctx->getDevice()->getCommon()->metaDebugView();

    // Temporal accumulation needs valid motion vectors; without them fall back to the stateless
    // behaviour rather than reprojecting with garbage.
    const bool motionAvailable = rtOutput.m_primaryScreenSpaceMotionVector.image != nullptr;
    const bool accumulate = temporalAccumulation() && motionAvailable;

    // On a camera cut or an engine-level history reset the motion vectors describe a
    // relationship between two frames that no longer exists, so reprojection would drag the
    // previous scene's exposure into this one. The neighbourhood clamp would eventually pull it
    // back, but only over several frames and only where the two scenes happen to overlap in
    // brightness, so the history is dropped outright instead.
    if (resetHistory) {
      m_hasHistory = false;
    }

    const uint32_t writeIndex = m_accumIndex;
    const uint32_t readIndex = m_accumIndex ^ 1u;

    if (timing) {
      markTimer(ctx);
    }

    {
      ScopedGpuProfileZone(ctx, "Auto Exposure Plus Init");

      AutoExposurePlusInitArgs pushArgs = {};
      pushArgs.colorExtent = uvec2 { finalResolution.width, finalResolution.height };
      pushArgs.pyramidExtent = uvec2 { pyramidExtent.width, pyramidExtent.height };
      pushArgs.exposure = exposureFactor;
      pushArgs.intensity = intensity();
      pushArgs.enableAutoExposure = enableAutoExposure;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_INIT_COLOR_INPUT, rtOutput.m_finalOutput.view(Resources::AccessType::Read), nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_INIT_PYRAMID_OUTPUT, m_pyramid.views[0], nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_INIT_EXPOSURE, exposureView, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusInitShader::getShader());
      ctx->dispatch(pyramidWorkgroups.width, pyramidWorkgroups.height, pyramidWorkgroups.depth);
    }

    if (timing) {
      markTimer(ctx);
    }

    // Build the pyramid. Each level is a separable edge-aware downsample: a horizontal pass
    // decimating X into the scratch chain, then a vertical pass decimating Y into the next
    // pyramid level. The scratch level k is only partially used - its valid region is
    // (width(k+1) x height(k)), which is exactly half of mip k.
    {
      ScopedGpuProfileZone(ctx, "Auto Exposure Plus Downsample");

      for (uint32_t level = 0; level + 1 < levelCount; ++level) {
        const VkExtent3D srcExtent = m_pyramid.image->mipLevelExtent(level);
        const VkExtent3D dstExtent = m_pyramid.image->mipLevelExtent(level + 1);
        const VkExtent3D intermediateExtent = VkExtent3D { dstExtent.width, srcExtent.height, 1 };

        {
          AutoExposurePlusDownsampleArgs pushArgs = {};
          pushArgs.srcExtent = uvec2 { srcExtent.width, srcExtent.height };
          pushArgs.dstExtent = uvec2 { intermediateExtent.width, intermediateExtent.height };
          pushArgs.horizontal = 1;
          ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

          ctx->bindResourceView(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_INPUT, m_pyramid.views[level], nullptr);
          ctx->bindResourceView(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_OUTPUT, m_pyramidTmp.views[level], nullptr);
          ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusDownsampleShader::getShader());

          const VkExtent3D workgroups = util::computeBlockCount(intermediateExtent, VkExtent3D { 16, 16, 1 });
          ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
        }

        {
          AutoExposurePlusDownsampleArgs pushArgs = {};
          pushArgs.srcExtent = uvec2 { intermediateExtent.width, intermediateExtent.height };
          pushArgs.dstExtent = uvec2 { dstExtent.width, dstExtent.height };
          pushArgs.horizontal = 0;
          ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

          ctx->bindResourceView(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_INPUT, m_pyramidTmp.views[level], nullptr);
          ctx->bindResourceView(AUTO_EXPOSURE_PLUS_DOWNSAMPLE_OUTPUT, m_pyramid.views[level + 1], nullptr);
          ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusDownsampleShader::getShader());

          const VkExtent3D workgroups = util::computeBlockCount(dstExtent, VkExtent3D { 16, 16, 1 });
          ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
        }
      }
    }

    if (timing) {
      markTimer(ctx);
    }

    {
      ScopedGpuProfileZone(ctx, "Auto Exposure Plus Fuse");

      AutoExposurePlusFuseArgs pushArgs = {};
      pushArgs.colorExtent = uvec2 { finalResolution.width, finalResolution.height };
      pushArgs.pyramidExtent = uvec2 { pyramidExtent.width, pyramidExtent.height };
      pushArgs.exposure = exposureFactor;
      pushArgs.equalizationStrength = equalizationStrength();
      pushArgs.equalizationPivot = std::max(1e-3f, equalizationPivot());
      pushArgs.lowestLevel = levelCount - 1;
      pushArgs.enableAutoExposure = enableAutoExposure;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      // Bound as the whole mip chain rather than a single level, since the collapse samples
      // every level at the same full frame UV.
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_FUSE_PYRAMID_INPUT, m_pyramid.view, nullptr);
      ctx->bindResourceSampler(AUTO_EXPOSURE_PLUS_FUSE_PYRAMID_INPUT, linearSampler);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_FUSE_COLOR_INPUT, rtOutput.m_finalOutput.view(Resources::AccessType::Read), nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_FUSE_OUTPUT, m_fused.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_FUSE_EXPOSURE, exposureView, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusFuseShader::getShader());
      ctx->dispatch(pyramidWorkgroups.width, pyramidWorkgroups.height, pyramidWorkgroups.depth);
    }

    if (timing) {
      markTimer(ctx);
    }

    if (accumulate) {
      ScopedGpuProfileZone(ctx, "Auto Exposure Plus Temporal");

      const VkExtent3D motionExtent = rtOutput.m_primaryScreenSpaceMotionVector.image->info().extent;

      // Same form the base auto exposure uses, so that matching the speeds actually produces
      // matching settling behaviour rather than merely matching numbers.
      const float speed = matchBaseAdaptationSpeed() ? DxvkAutoExposure::autoExposureSpeed() : adaptationSpeed();
      const float perFrameSpeed = speed * (0.001f * frameTimeMilliseconds);

      AutoExposurePlusTemporalArgs pushArgs = {};
      pushArgs.fusedExtent = uvec2 { pyramidExtent.width, pyramidExtent.height };
      pushArgs.motionExtent = uvec2 { motionExtent.width, motionExtent.height };
      pushArgs.currentWeight = std::min(1.0f, std::max(0.0f, 1.0f - exp2f(-perFrameSpeed)));
      pushArgs.hasHistory = m_hasHistory ? 1u : 0u;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_TEMPORAL_CURRENT_INPUT, m_fused.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_TEMPORAL_HISTORY_INPUT, m_fusedAccum[readIndex].view, nullptr);
      ctx->bindResourceSampler(AUTO_EXPOSURE_PLUS_TEMPORAL_HISTORY_INPUT, linearSampler);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_TEMPORAL_MOTION_INPUT, rtOutput.m_primaryScreenSpaceMotionVector.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_TEMPORAL_OUTPUT, m_fusedAccum[writeIndex].view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusTemporalShader::getShader());
      ctx->dispatch(pyramidWorkgroups.width, pyramidWorkgroups.height, pyramidWorkgroups.depth);
    }

    if (timing) {
      markTimer(ctx);
    }

    {
      ScopedGpuProfileZone(ctx, "Auto Exposure Plus Apply");

      AutoExposurePlusApplyArgs pushArgs = {};
      pushArgs.fusedTexelSize = vec4 {
        static_cast<float>(pyramidExtent.width),
        static_cast<float>(pyramidExtent.height),
        1.0f / pyramidExtent.width,
        1.0f / pyramidExtent.height
      };
      pushArgs.colorExtent = uvec2 { finalResolution.width, finalResolution.height };
      pushArgs.exposure = exposureFactor;
      pushArgs.targetEV = targetEV();
      pushArgs.maxGainEV = std::max(1e-3f, maxGainEV());
      pushArgs.enableAutoExposure = enableAutoExposure;
      pushArgs.debugView = debugView.debugViewIdx();
      pushArgs.debugPyramidLevel = std::min(static_cast<uint32_t>(std::max(0, debugPyramidLevel())), levelCount - 1);
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      const Rc<DxvkImageView>& gainField = accumulate ? m_fusedAccum[writeIndex].view : m_fused.view;

      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_APPLY_FUSED_INPUT, gainField, nullptr);
      ctx->bindResourceSampler(AUTO_EXPOSURE_PLUS_APPLY_FUSED_INPUT, linearSampler);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_APPLY_PYRAMID_INPUT, m_pyramid.view, nullptr);
      ctx->bindResourceSampler(AUTO_EXPOSURE_PLUS_APPLY_PYRAMID_INPUT, linearSampler);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_APPLY_COLOR_INPUT_OUTPUT, rtOutput.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_APPLY_DEBUG_VIEW_OUTPUT, debugView.getDebugOutput(), nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_PLUS_APPLY_EXPOSURE, exposureView, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposurePlusApplyShader::getShader());
      ctx->dispatch(finalWorkgroups.width, finalWorkgroups.height, finalWorkgroups.depth);
    }

    if (timing) {
      markTimer(ctx);
      reportTimers(pyramidExtent, levelCount);
    }

    if (accumulate) {
      m_accumIndex = readIndex;
      m_hasHistory = true;
    } else {
      // The history that would be reprojected next frame was never written, so it must not be
      // trusted when accumulation comes back on.
      m_hasHistory = false;
    }
  }

  void DxvkAutoExposurePlus::createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) {
    const VkExtent3D pyramidExtent = calcPyramidExtent(targetExtent);
    m_levelCount = calcLevelCount(targetExtent);

    m_pyramid = RtxMipmap::createResource(ctx, "auto exposure plus pyramid", pyramidExtent,
                                          VK_FORMAT_R16G16_SFLOAT, 0, { 0.f, 0.f, 0.f, 0.f }, m_levelCount);
    m_pyramidTmp = RtxMipmap::createResource(ctx, "auto exposure plus pyramid scratch", pyramidExtent,
                                             VK_FORMAT_R16G16_SFLOAT, 0, { 0.f, 0.f, 0.f, 0.f }, m_levelCount);
    m_fused = Resources::createImageResource(ctx, "auto exposure plus gain field", pyramidExtent,
                                             VK_FORMAT_R16G16_SFLOAT);
    m_fusedAccum[0] = Resources::createImageResource(ctx, "auto exposure plus gain history 0", pyramidExtent,
                                                     VK_FORMAT_R16G16_SFLOAT);
    m_fusedAccum[1] = Resources::createImageResource(ctx, "auto exposure plus gain history 1", pyramidExtent,
                                                     VK_FORMAT_R16G16_SFLOAT);

    m_accumIndex = 0;
    m_hasHistory = false;
  }

  void DxvkAutoExposurePlus::releaseTargetResource() {
    m_pyramid.reset();
    m_pyramidTmp.reset();
    m_fused.reset();
    m_fusedAccum[0].reset();
    m_fusedAccum[1].reset();
    m_levelCount = 0;
    m_hasHistory = false;
  }
}
