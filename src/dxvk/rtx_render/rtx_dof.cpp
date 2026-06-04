/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_context.h"
#include "rtx_dof.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx_engine_post_state.h"
#include "rtx/pass/dof/dof.h"

#include <rtx_shaders/dof_gather.h>
#include "rtx_imgui.h"

namespace dxvk {
  namespace {
    class DepthOfFieldShader : public ManagedShader {
      SHADER_SOURCE(DepthOfFieldShader, VK_SHADER_STAGE_COMPUTE_BIT, dof_gather)

      PUSH_CONSTANTS(DepthOfFieldArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(DOF_COLOR_INPUT)
        SAMPLER2D(DOF_VIEWZ_INPUT)
        RW_TEXTURE2D(DOF_COLOR_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(DepthOfFieldShader);
  }

  DxvkDepthOfField::DxvkDepthOfField(DxvkDevice* device): RtxPass(device), m_vkd(device->vkd()) {
  }

  DxvkDepthOfField::~DxvkDepthOfField() {
  }

  void DxvkDepthOfField::showImguiSettings() {
    ImGui::Indent();
    RemixGui::Checkbox("DoF Enabled", &enableObject());
    RemixGui::Checkbox("Always On (ignore game trigger)", &alwaysOnObject());
    ImGui::Indent();
    RemixGui::DragFloat("Depth Scale##dof", &depthScaleObject(), 0.01f, 0.f, 8.f, "%.3f");
    RemixGui::DragFloat("Blur Kernel (small texels)##dof", &blurKernelTexelsObject(), 0.1f, 0.f, 8.f, "%.2f");
    RemixGui::DragFloat("Strength##dof", &strengthObject(), 0.02f, 0.f, 1.f, "%.2f");
    RemixGui::SliderInt("Taps##dof", &numTapsObject(), 4, 64);
    ImGui::Unindent();
    ImGui::Unindent();
  }

  void DxvkDepthOfField::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Depth of Field");

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    const Resources::Resource& finalRes = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const Resources::Resource& viewZRes = rtOutput.m_primaryLinearViewZ;
    if (finalRes.image == nullptr || viewZRes.image == nullptr || m_dofScratch.image == nullptr) {
      return;
    }

    const VkExtent3D ext = finalRes.image->info().extent;

    // Stable read copy: finalOutput -> scratch, then the gather samples scratch
    // while writing back into finalOutput (avoids a read/write hazard on one image).
    VkImageSubresourceLayers sub {};
    sub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sub.layerCount = 1u;
    ctx->copyImage(m_dofScratch.image, sub, VkOffset3D { 0, 0, 0 },
                   finalRes.image, sub, VkOffset3D { 0, 0, 0 }, ext);

    const EnginePostState& eps = EnginePostState::get();
    const bool active = alwaysOn() || eps.dofActive.load(std::memory_order_relaxed);

    DepthOfFieldArgs pushArgs = {};
    pushArgs.imageSize = { ext.width, ext.height };
    pushArgs.imageSizeInverse = { 1.0f / float(ext.width), 1.0f / float(ext.height) };
    // Feed the game's harvested c_dof (DoFParams) directly into the CoC ramp, so
    // the blur depth range matches the game and the default/off config produces no
    // blur at normal gameplay depths.
    pushArgs.dofA = { eps.dof[0], eps.dof[1], eps.dof[2], eps.dof[3] };
    pushArgs.dofB = { eps.dof[4], eps.dof[5], eps.dof[6], eps.dof[7] };
    pushArgs.depthScale = std::max(0.0f, depthScale());
    // Derive the gather radius from the game's DoFBlurSmall downsample ratio
    // (renderWidth / dofBlurSmallWidth) * the small-texture kernel radius, so blur
    // strength tracks the game instead of a fixed pixel count. Falls back to a 4x
    // ratio if the texture hasn't been observed yet.
    {
      const uint32_t blurSmallW = eps.dofBlurSmallWidth.load(std::memory_order_relaxed);
      const float ratio = (blurSmallW > 0u) ? float(ext.width) / float(blurSmallW) : 4.0f;
      pushArgs.maxBlurRadius = std::clamp(ratio * std::max(0.0f, blurKernelTexels()), 0.0f, 64.0f);
    }
    pushArgs.strength = active ? std::clamp(strength(), 0.0f, 1.0f) : 0.0f;
    pushArgs.numTaps = static_cast<uint32_t>(std::clamp(numTaps(), 4, 64));
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

    // Verification: log the c_dof actually driving the CoC + the resulting ramp.
    // Change-triggered (active flips) plus a sparse heartbeat.
    {
      static int s_lastActive = -1;
      static uint64_t s_beat = 0;
      const uint64_t fr = eps.writtenFrame.load(std::memory_order_relaxed);
      if ((active ? 1 : 0) != s_lastActive || fr - s_beat >= 30) {
        s_lastActive = active ? 1 : 0;
        s_beat = fr;
        Logger::info(str::format(
          "[DoF] f=", fr, " active=", active ? 1 : 0,
          " depthScale=", pushArgs.depthScale,
          " blurSmallW=", eps.dofBlurSmallWidth.load(std::memory_order_relaxed),
          " -> radiusPx=", pushArgs.maxBlurRadius,
          " strength=", pushArgs.strength, " taps=", pushArgs.numTaps,
          " | c_dof: nearDepthEnd=", eps.dof[0],
          " worldParams=(", eps.dof[4], ",", eps.dof[5], ",", eps.dof[6], ",", eps.dof[7], ")"));
      }
    }

    const VkExtent3D workgroups = util::computeBlockCount(ext, VkExtent3D { 16, 16, 1 });

    ctx->bindResourceView(DOF_COLOR_INPUT, m_dofScratch.view, nullptr);
    ctx->bindResourceSampler(DOF_COLOR_INPUT, linearSampler);
    ctx->bindResourceView(DOF_VIEWZ_INPUT, viewZRes.view, nullptr);
    ctx->bindResourceSampler(DOF_VIEWZ_INPUT, linearSampler);
    ctx->bindResourceView(DOF_COLOR_OUTPUT, finalRes.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, DepthOfFieldShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkDepthOfField::createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) {
    m_dofScratch = Resources::createImageResource(
      ctx, "dof scratch", targetExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
  }

  void DxvkDepthOfField::releaseTargetResource() {
    m_dofScratch.reset();
  }

  bool DxvkDepthOfField::isEnabled() const {
    // Only activate when the host game's DoF is on (cinematics) or alwaysOn. This
    // keeps the full-res copy + gather off the frame entirely when DoF is inactive.
    return enable() && (alwaysOn() || EnginePostState::get().dofActive.load(std::memory_order_relaxed));
  }
}
