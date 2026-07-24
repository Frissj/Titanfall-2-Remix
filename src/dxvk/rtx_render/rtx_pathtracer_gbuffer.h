/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_resources.h"
#include "../util/rc/util_rc_ptr.h"

namespace dxvk {

  class DxvkDevice;

  class DxvkPathtracerGbuffer : public CommonDeviceObject {
  public:
    enum class RaytraceMode {
      RayQuery = 0,
      RayQueryRayGen,
      TraceRay,
      Count
    };

    explicit DxvkPathtracerGbuffer(DxvkDevice* device);
    ~DxvkPathtracerGbuffer() = default;

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;
    void dispatch(class RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    static const char* raytraceModeToString(RaytraceMode raytraceMode);

    // NV-DXVK [perf]: DIAGNOSTIC ONLY — scales the primary-ray dispatch grid.
    //
    // The question the [Perf.GpuPass] split cannot answer on its own: once the
    // barrier wait is separated out (gb_bindWait), is the remaining primary-ray
    // time proportional to the number of rays, or is it a fixed cost? Those imply
    // completely different causes. Per-ray cost means the shader or the traversal
    // is genuinely expensive and profiling the shader is the next step; a flat
    // cost means the dispatch is blocking on something and no shader work will be
    // found. 1.2 Mpix at 130 ms is ~108 ns/ray, roughly 20-50x what an RTX 4080
    // should do for one primary ray, so "the cost is not per-ray" is a live
    // possibility that has never been tested.
    //
    // Halving this quarters the launched threads. Values below 1 render an
    // incomplete gbuffer — this is a measurement knob, not a quality setting, and
    // it is left at 1.0 unless a scaling test is actively being run. Applies to
    // the primary-ray dispatch only, so PSR and every later pass keep their normal
    // grid and stay comparable across the sweep.
    RTX_OPTION("rtx", float, perfPrimaryRayGridScale, 1.0f,
               "DIAGNOSTIC: scales the primary-ray dispatch grid. 1.0 = normal. "
               "Below 1.0 renders an incomplete image; used to test whether "
               "gb_primaryRays cost is proportional to ray count.");

  private:
    static DxvkRaytracingPipelineShaders getPipelineShaders(const bool isPSRPass, const bool useRayQuery, const bool serEnabled, const bool ommEnabled, const bool includePortals, const bool nrcEnabled, const bool wboitEnabled);
    Rc<DxvkShader> getComputeShader(const bool isPSRPass, const bool nrcEnabled, const bool wboitEnabled) const;
  };
}
