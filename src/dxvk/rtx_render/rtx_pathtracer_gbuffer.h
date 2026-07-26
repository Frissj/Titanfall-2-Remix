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

    // NV-DXVK [Perf.Occupancy]: DIAGNOSTIC — declared workgroup size of the
    // primary-ray shader, which acts as a register cap.
    //
    // A block must be resident in one SM's 65536 registers, so
    // warps * 32 * registersPerThread <= 65536: 512 threads forces <= 128
    // registers and 16 warps/SM, against the 255 registers and 8 warps/SM the
    // 128-thread build gets today. This is the only lever that moves occupancy on
    // Ada — shared memory cannot, since maxComputeSharedMemorySize is 49152 and
    // two such blocks still fit the 100 KB SM carveout.
    //
    // 256 is the CONTROL rung: still 8 warps/SM, just one block instead of two.
    // A delta there is block granularity, not occupancy, and must be subtracted
    // from whatever 384 and 512 show.
    //
    // The image is unchanged at every setting — same work, same order, only the
    // thread grouping differs. Output is bit-identical; only spill traffic and
    // scheduling change. Read Register Count and Local Memory Size off the
    // [Perf.Shader] line for each rung: the spill is the price being paid for
    // the extra warps, and if the extra warps do not pay for it the megakernel
    // split cannot either.
    //
    // Valid values are 0 (stock 128), 256, 384 and 512. Anything else falls back
    // to 0. Only built for the NRC + WBOIT + no-portals RayQuery permutation;
    // ignored otherwise, which is logged.
    RTX_OPTION("rtx", uint32_t, perfGbufferBlockThreads, 0,
               "DIAGNOSTIC: primary-ray workgroup size (0=stock 128, 256, 384, 512). "
               "Larger blocks force the register allocator down and raise occupancy, "
               "at the cost of spilling. Used to test whether the 255-register cap "
               "is the cause of gb_primaryRays cost or a symptom of it.");

  private:
    static DxvkRaytracingPipelineShaders getPipelineShaders(const bool isPSRPass, const bool useRayQuery, const bool serEnabled, const bool ommEnabled, const bool includePortals, const bool nrcEnabled, const bool wboitEnabled);
    // NV-DXVK [perf]: includePortals mirrors the axis getPipelineShaders() has
    // always had. When false the shader is compiled with
    // SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT, which removes both the
    // ray portal resolve branch and (via RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES)
    // the intersection-billboard branch of the unordered resolve. See the
    // variant block in gbuffer.slang.
    Rc<DxvkShader> getComputeShader(const bool isPSRPass, const bool nrcEnabled, const bool wboitEnabled, const bool includePortals) const;

    // NV-DXVK [Perf.Occupancy]: returns the block-size ladder variant for the
    // primary-ray dispatch, or nullptr when the request does not apply (option
    // off, unsupported value, or a permutation the ladder was not built for).
    // Writes the matching block extent to blockSize on success.
    Rc<DxvkShader> getOccupancyLadderShader(const bool nrcEnabled, const bool wboitEnabled, const bool includePortals, VkExtent3D& blockSize) const;
  };
}
