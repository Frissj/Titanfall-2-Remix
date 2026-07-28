/*
/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "dxvk_options.h"
#include "vulkan/vulkan_core.h"

namespace dxvk {

  DxvkOptions::DxvkOptions(const Config& config) {
    enableStateCache      = config.getOption<bool>    ("dxvk.enableStateCache",       true);
    numCompilerThreads    = config.getOption<int32_t> ("dxvk.numCompilerThreads",     0);
    useRawSsbo            = config.getOption<Tristate>("dxvk.useRawSsbo",             Tristate::Auto);
    shrinkNvidiaHvvHeap   = config.getOption<Tristate>("dxvk.shrinkNvidiaHvvHeap",    Tristate::Auto);
    hud                   = config.getOption<std::string>("dxvk.hud", "");

    // NV-DXVK start: Integrate Aftermath
    // Was hardcoded true "for diagnosis". It has to stay overridable, because
    // Aftermath is mutually exclusive with Nsight Graphics' GPU Trace shader
    // profiler. enableAftermath does not only arm the crash-dump callbacks - it
    // is also what chains VkDeviceDiagnosticsConfigCreateInfoNV into
    // vkCreateDevice (dxvk_adapter.cpp), with all four flags:
    //   SHADER_DEBUG_INFO | SHADER_ERROR_REPORTING
    //   AUTOMATIC_CHECKPOINTS | RESOURCE_TRACKING
    // SHADER_DEBUG_INFO makes the driver emit full debug info for EVERY pipeline
    // it compiles, and AUTOMATIC_CHECKPOINTS inserts driver checkpoints around
    // every draw/dispatch. --real-time-shader-profiler wants that same per-shader
    // driver path for source correlation, over the ~9.4k shader modules DXVK
    // creates for this game's D3D11 shaders. Default stays true so crash
    // diagnosis is unchanged; the ngfx capture script sets DXVK_ENABLE_AFTERMATH=0
    // for profiling runs only, so this never has to be rebuilt to flip.
    enableAftermath = config.getOption<bool>("dxvk.enableAftermath", true, "DXVK_ENABLE_AFTERMATH");
    enableAftermathResourceTracking = config.getOption<bool>("dxvk.enableAftermathResourceTracking", true, "DXVK_ENABLE_AFTERMATH_RESOURCE_TRACKING");
    // NV-DXVK end

    // NV-DXVK start: VK_NV_present_metering opt-out for profiling runs.
    // Default true = unchanged for normal play. Set to false (or
    // DXVK_ENABLE_PRESENT_METERING=0) when running under Nsight Graphics
    // 2025.4, which does not know the extension and puts up a blocking modal at
    // device creation that no unattended capture can dismiss. Safe to drop -
    // see the comment at the enable site in dxvk_adapter.cpp.
    enablePresentMetering = config.getOption<bool>("dxvk.enablePresentMetering", true, "DXVK_ENABLE_PRESENT_METERING");
    // NV-DXVK end

    // NV-DXVK start: early submit heuristics for memcpy work
    memcpyKickoffThreshold = config.getOption<uint32_t>("dxvk.memcpyKickoffThreshold", 16 * 1024 * 1024);
    // NV-DXVK end

    // NV-DXVK start: tell the user they cant run Remix
    float nvidiaMinDriverFloat = config.getOption<float>("dxvk.nvidiaMinDriver", 572.18f, "DXVK_REMIX_NVIDIA_MIN_DRIVER");
    float nvidiaGfnMinDriverFloat = config.getOption<float>("dxvk.nvidiaGfnMinDriver", 527.01f);
    float nvidiaLinuxMinDriverFloat = config.getOption<float>("dxvk.nvidiaLinuxMinDriver", 525.60f);

    // Convert human readable version from settings to proper version number
    float major = 0;
    long minor = 0;
    // Desktop Windows
    minor = std::lround(std::modf(nvidiaMinDriverFloat, &major) * 100);
    nvidiaMinDriver = VK_MAKE_API_VERSION(0, major, minor, 0);    

    // Desktop Linux (via Proton)
    minor = std::lround(std::modf(nvidiaLinuxMinDriverFloat, &major) * 100);
    nvidiaLinuxMinDriver = VK_MAKE_API_VERSION(0, major, minor, 0);
    // NV-DXVK end
    
    // NV-DXVK start: configurable memory allocation chunk sizes
    deviceLocalMemoryChunkSizeMB = config.getOption<uint32_t>("dxvk.deviceLocalMemoryChunkSizeMB", 320);
    otherMemoryChunkSizeMB = config.getOption<uint32_t>("dxvk.otherMemoryChunkSizeMB", 128);
    // NV-DXVK end
  }

}
