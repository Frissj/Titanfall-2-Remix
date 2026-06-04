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
#pragma once

#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_resources.h"

namespace dxvk {

  class DxvkDevice;

  // NV-DXVK [engine-post DoF, Route B]: depth of field that recomputes circle of
  // confusion from Remix's path-traced depth (m_primaryLinearViewZ) and applies a
  // CoC-weighted gather blur to the final tonemapped image. Triggered on/off by the
  // host game's DoF state (EnginePostState::dofActive, harvested from CBufEnginePost)
  // so it only blurs during the game's DoF moments (cinematics), unless alwaysOn.
  class DxvkDepthOfField : public RtxPass {

  public:
    explicit DxvkDepthOfField(DxvkDevice* device);
    ~DxvkDepthOfField();

    DxvkDepthOfField(const DxvkDepthOfField&) = delete;
    DxvkDepthOfField(DxvkDepthOfField&&) noexcept = delete;
    DxvkDepthOfField& operator=(const DxvkDepthOfField&) = delete;
    DxvkDepthOfField& operator=(DxvkDepthOfField&&) noexcept = delete;

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const Resources::RaytracingOutput& rtOutput);

    void showImguiSettings();

  private:
    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) override;
    virtual void releaseTargetResource() override;
    virtual bool isEnabled() const override;

    Rc<vk::DeviceFn> m_vkd;

    // Stable read copy of the final output: the gather samples this while writing
    // back into the final output, avoiding a read/write hazard on one image.
    Resources::Resource m_dofScratch;

    RTX_OPTION("rtx.dof", bool, enable, true,
               "Depth of field recomputed from Remix's path-traced depth, blended like the host game's DoF. Only blurs when the game's DoF is active (e.g. cinematics) unless rtx.dof.alwaysOn is set.");
    RTX_OPTION("rtx.dof", bool, alwaysOn, false,
               "Apply DoF every frame regardless of the game's DoF trigger. Useful for tuning depthScale.");
    RTX_OPTION("rtx.dof", float, depthScale, 1.0f,
               "Scales path-traced |viewZ| to match the linear view depth the game's CoC ramp uses (CoC = saturate(|viewZ|*depthScale*scale + bias), scale/bias from harvested c_dof). The game feeds raw linear view-space Z, so 1.0 is correct unless Remix's viewZ units differ; lower it if near-field blur reaches too far.");
    RTX_OPTION("rtx.dof", float, blurKernelTexels, 2.0f,
               "Blur kernel radius, in DoFBlurSmall texels, that the game applies to its downsampled DoF buffer. The full-res gather radius is derived live as (renderWidth / DoFBlurSmall width) * this, so it tracks the game's blur strength (which the game encodes as the downsample resolution, not a number). This is the one residual constant; disassembling the DoFBlurSmall blur pass would pin it exactly.");
    RTX_OPTION("rtx.dof", float, strength, 1.0f,
               "Overall DoF strength multiplier [0..1]. Scales the circle of confusion; 0 disables.");
    RTX_OPTION_ARGS("rtx.dof", int, numTaps, 24,
                    "Gather tap count (quality vs cost).",
                    args.minValue = 4,
                    args.maxValue = 64);
  };

}
