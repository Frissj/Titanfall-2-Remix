/*
* Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>

#include "../util/util_time.h"
#include "../util/thread.h"

#include "dxvk_compute.h"
#include "dxvk_device.h"
#include "dxvk_pipemanager.h"
#include "dxvk_spec_const.h"
#include "dxvk_state_cache.h"

namespace dxvk {
  
  DxvkComputePipeline::DxvkComputePipeline(
          DxvkPipelineManager*        pipeMgr,
          DxvkComputePipelineShaders  shaders)
  : m_vkd(pipeMgr->m_device->vkd()), m_pipeMgr(pipeMgr),
    m_shaders(std::move(shaders)) {
    m_shaders.cs->defineResourceSlots(m_slotMapping);
    
    m_slotMapping.makeDescriptorsDynamic(
      m_pipeMgr->m_device->options().maxNumDynamicUniformBuffers,
      m_pipeMgr->m_device->options().maxNumDynamicStorageBuffers);
    
    // NV-DXVK start: descriptor set(s)
    m_layout = new DxvkPipelineLayout(m_vkd,
      m_slotMapping, VK_PIPELINE_BIND_POINT_COMPUTE, m_shaders.cs->shaderOptions().extraLayouts);
    // NV-DXVK end
  }
  
  
  DxvkComputePipeline::~DxvkComputePipeline() {
    if (m_noSpecConstantPipelines) {
      this->destroyPipeline(m_noSpecConstantPipelines->pipeline());
    }

    for (const auto& instance : m_pipelines)
      this->destroyPipeline(instance.pipeline());
  }
  
  
  VkPipeline DxvkComputePipeline::getPipelineHandle(
    const DxvkComputePipelineStateInfo& state) {
    DxvkComputePipelineInstance* instance = nullptr;

    // NV-DXVK [Perf.PipeComp]: compute-pipeline lookup census. Same rationale
    // and same read as [Perf.PipeGfx] in dxvk_graphics.cpp - see the comment
    // there. Split by pipeline type on purpose: Remix's own passes are compute,
    // the game's rasterisation is graphics, so whichever of the two counters
    // shows the compiles tells us whose shaders are still being built during
    // gameplay without having to guess from module names in a sampling profile.
    static std::atomic<uint64_t> s_lookups { 0 };
    static std::atomic<uint64_t> s_hits { 0 };
    static std::atomic<uint64_t> s_compiles { 0 };
    static std::atomic<uint64_t> s_compileNs { 0 };
    static std::atomic<uint64_t> s_maxCompileNs { 0 };
    static std::atomic<uint32_t> s_lastCompileTid { 0 };
    static std::atomic<uint64_t> s_cumCompiles { 0 };
    static std::atomic<uint64_t> s_windowStartNs { 0 };
    static const auto s_epoch = dxvk::high_resolution_clock::now();

    s_lookups.fetch_add(1, std::memory_order_relaxed);

    { std::lock_guard<sync::Spinlock> lock(m_mutex);

      instance = this->findInstance(state);

      if (instance) {
        s_hits.fetch_add(1, std::memory_order_relaxed);
        return instance->pipeline();
      }

      // If no pipeline instance exists with the given state
      // vector, create a new one and add it to the list.
      const auto compileT0 = dxvk::high_resolution_clock::now();
      instance = this->createInstance(state);
      const uint64_t compileNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        dxvk::high_resolution_clock::now() - compileT0).count());

      s_compiles.fetch_add(1, std::memory_order_relaxed);
      s_cumCompiles.fetch_add(1, std::memory_order_relaxed);
      s_compileNs.fetch_add(compileNs, std::memory_order_relaxed);
      s_lastCompileTid.store(uint32_t(GetCurrentThreadId()), std::memory_order_relaxed);

      uint64_t prevMax = s_maxCompileNs.load(std::memory_order_relaxed);
      while (compileNs > prevMax
          && !s_maxCompileNs.compare_exchange_weak(prevMax, compileNs, std::memory_order_relaxed)) { }
    }

    // Counter-gated clock read, same reasoning as the graphics path. Placed
    // after the miss branch so a compile is always counted before it can be
    // reported, never split across two windows.
    if ((s_lookups.load(std::memory_order_relaxed) & 0x3FFull) == 0ull) {
      const uint64_t nowNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        dxvk::high_resolution_clock::now() - s_epoch).count());
      uint64_t windowStart = s_windowStartNs.load(std::memory_order_relaxed);

      if (windowStart == 0ull) {
        s_windowStartNs.compare_exchange_strong(windowStart, nowNs, std::memory_order_relaxed);
      } else if (nowNs - windowStart >= 5000000000ull
              && s_windowStartNs.compare_exchange_strong(windowStart, nowNs, std::memory_order_relaxed)) {
        const uint64_t lookups  = s_lookups.exchange(0, std::memory_order_relaxed);
        const uint64_t hits     = s_hits.exchange(0, std::memory_order_relaxed);
        const uint64_t compiles = s_compiles.exchange(0, std::memory_order_relaxed);
        const uint64_t totalNs  = s_compileNs.exchange(0, std::memory_order_relaxed);
        const uint64_t maxNs    = s_maxCompileNs.exchange(0, std::memory_order_relaxed);

        Logger::warn(str::format(
          "[Perf.PipeComp] window=", double(nowNs - windowStart) / 1.0e9, "s",
          " lookups=", lookups,
          " hits=", hits,
          " compiles=", compiles,
          " compileMsTotal=", double(totalNs) / 1.0e6,
          " compileMsMax=", double(maxNs) / 1.0e6,
          " lastCompileTid=", s_lastCompileTid.load(std::memory_order_relaxed),
          " cumulativeCompiles=", s_cumCompiles.load(std::memory_order_relaxed)));
      }
    }

    if (!instance)
      return VK_NULL_HANDLE;

    // Note: Only write pipelines with actual spec constant state to the cache as
    // without this state there is nothing to cache (and circumventing this disk caching
    // dependency is part of the point of the flag to force no spec constants anyways).
    if (!m_shaders.forceNoSpecConstants) {
      this->writePipelineStateToCache(state);
    }

    return instance->pipeline();
  }


  void DxvkComputePipeline::compilePipeline(
    const DxvkComputePipelineStateInfo& state) {
    ScopedCpuProfileZone();

    std::lock_guard<sync::Spinlock> lock(m_mutex);

    if (!this->findInstance(state))
      this->createInstance(state);
  }
  
  
  DxvkComputePipelineInstance* DxvkComputePipeline::createInstance(
    const DxvkComputePipelineStateInfo& state) {
    VkPipeline newPipelineHandle = this->createPipeline(state);

    m_pipeMgr->m_numComputePipelines += 1;

    if (m_shaders.forceNoSpecConstants) {
      return &m_noSpecConstantPipelines.emplace(state, newPipelineHandle);
    } else {
      return &m_pipelines.emplace_back(state, newPipelineHandle);
    }
  }

  
  DxvkComputePipelineInstance* DxvkComputePipeline::findInstance(
    const DxvkComputePipelineStateInfo& state) {
    // Handle forced no spec constant case

    if (m_shaders.forceNoSpecConstants) {
      if (m_noSpecConstantPipelines) {
        return &*m_noSpecConstantPipelines;
      } else {
        return nullptr;
      }
    }

    // Handle typical pipeline state case

    for (auto& instance : m_pipelines) {
      if (instance.isCompatible(state))
        return &instance;
    }
    
    return nullptr;
  }
  
  
  VkPipeline DxvkComputePipeline::createPipeline(
    const DxvkComputePipelineStateInfo& state) const {
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    if (Logger::logLevel() <= LogLevel::Debug) {
      Logger::debug("Compiling compute pipeline..."); 
      Logger::debug(str::format("  cs  : ", m_shaders.cs->debugName()));
    }
    
    DxvkSpecConstants specData;

    // Note: Only set spec constants if they are needed.
    if (!m_shaders.forceNoSpecConstants) {
      for (uint32_t i = 0; i < m_layout->bindingCount(); i++)
        specData.set(i, state.bsBindingMask.test(i), true);

      for (uint32_t i = 0; i < MaxNumSpecConstants; i++)
        specData.set(getSpecId(i), state.sc.specConstants[i], 0u);
    }

    VkSpecializationInfo specInfo = specData.getSpecInfo();
    
    DxvkShaderModuleCreateInfo moduleInfo;
    moduleInfo.fsDualSrcBlend = false;

    auto csm = m_shaders.cs->createShaderModule(m_vkd, m_slotMapping, moduleInfo);

    VkComputePipelineCreateInfo info;
    info.sType                = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.pNext                = nullptr;
    // NV-DXVK [perf]: statistics are only retrievable from a pipeline that was
    // created with CAPTURE_STATISTICS. The flag costs nothing at runtime — it
    // only asks the driver to retain the compile-time report — but it must be
    // set at creation, so it cannot be turned on after the fact.
    info.flags                = m_pipeMgr->m_device->extensions().khrPipelineExecutableProperties
                                ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
    info.stage                = csm.stageInfo(&specInfo);
    info.layout               = m_layout->pipelineLayout();
    info.basePipelineHandle   = VK_NULL_HANDLE;
    info.basePipelineIndex    = -1;
    
    // Time pipeline compilation for debugging purposes
    dxvk::high_resolution_clock::time_point t0, t1;

    if (Logger::logLevel() <= LogLevel::Debug)
      t0 = dxvk::high_resolution_clock::now();
    
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (m_vkd->vkCreateComputePipelines(m_vkd->device(),
          m_pipeMgr->m_cache->handle(), 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
      Logger::err("DxvkComputePipeline: Failed to compile pipeline");
      Logger::err(str::format("  cs  : ", m_shaders.cs->debugName()));
      return VK_NULL_HANDLE;
    }
    
    if (Logger::logLevel() <= LogLevel::Debug) {
      t1 = dxvk::high_resolution_clock::now();
      auto td = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      Logger::debug(str::format("DxvkComputePipeline: Finished in ", td.count(), " ms"));
    }

    // NV-DXVK [perf]: [Perf.Shader] — ask the DRIVER what it compiled.
    //
    // The primary-ray dispatch costs ~140 ms and every structural explanation
    // measurable from the CPU has been eliminated. What is left is how the shader
    // itself was compiled: register count per thread sets how many waves fit on
    // an SM, and a shader that spills has to round-trip registers through memory
    // on paths the compiler could not keep resident. Both are decided at compile
    // time and neither is visible from any runtime counter — this is exactly the
    // data a profiler reports, retrieved in-process because Nsight cannot attach
    // to this build.
    //
    // Fires once per pipeline at creation, so it costs nothing per frame.
    // NV-DXVK [DiagLogGate]: compile-time OFF — the [Perf.Shader] occupancy
    // dump was for the primary-ray compile investigation; flip to re-enable.
    static constexpr bool kPerfShaderStats = false;
    if (kPerfShaderStats && m_pipeMgr->m_device->extensions().khrPipelineExecutableProperties) {
      VkPipelineInfoKHR pipeInfo { VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR };
      pipeInfo.pipeline = pipeline;

      uint32_t execCount = 0;
      if (m_vkd->vkGetPipelineExecutablePropertiesKHR(
            m_vkd->device(), &pipeInfo, &execCount, nullptr) == VK_SUCCESS && execCount > 0) {
        std::vector<VkPipelineExecutablePropertiesKHR> execs(execCount);
        for (auto& e : execs)
          e = VkPipelineExecutablePropertiesKHR { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR };

        if (m_vkd->vkGetPipelineExecutablePropertiesKHR(
              m_vkd->device(), &pipeInfo, &execCount, execs.data()) == VK_SUCCESS) {
          for (uint32_t i = 0; i < execCount; ++i) {
            VkPipelineExecutableInfoKHR execInfo { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR };
            execInfo.pipeline = pipeline;
            execInfo.executableIndex = i;

            uint32_t statCount = 0;
            if (m_vkd->vkGetPipelineExecutableStatisticsKHR(
                  m_vkd->device(), &execInfo, &statCount, nullptr) != VK_SUCCESS || statCount == 0)
              continue;

            // Zero the whole array before the query, not just sType. "Local Memory
            // Size" came back as 0x10_0000_0000 / 0x10_0000_0090 — a fixed high
            // dword over a plausible low dword — which is the signature of a
            // 32-bit write landing in a 64-bit slot over uninitialized bytes.
            // Explicit zeroing means a repeat of that pattern can no longer be
            // our own stack garbage, which matters because this statistic is the
            // only spill signal we have.
            std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
            std::memset(stats.data(), 0, stats.size() * sizeof(stats[0]));
            for (auto& s : stats)
              s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;

            if (m_vkd->vkGetPipelineExecutableStatisticsKHR(
                  m_vkd->device(), &execInfo, &statCount, stats.data()) != VK_SUCCESS)
              continue;

            // Statistic names are driver-defined, so print whatever it offers
            // rather than guessing at a fixed set. On NVIDIA this includes
            // register count, spills and achievable occupancy.
            std::string line;
            for (uint32_t s = 0; s < statCount; ++s) {
              line += str::format(" ", stats[s].name, "=");
              switch (stats[s].format) {
              case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                line += str::format(stats[s].value.b32 ? 1 : 0); break;
              case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                line += str::format(stats[s].value.i64); break;
              case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                line += str::format(stats[s].value.u64); break;
              case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                line += str::format(stats[s].value.f64); break;
              default:
                line += "?"; break;
              }
              // For the 64-bit integer formats also print the raw bits and the
              // low dword. A value whose high dword is set while the low dword is
              // small is not a real byte count, and printing both makes that
              // decidable from the log instead of inferable.
              if (stats[s].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
               || stats[s].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) {
                const uint64_t raw = stats[s].value.u64;
                if ((raw >> 32) != 0ull) {
                  line += str::format("(raw=0x", std::hex, raw, std::dec,
                                      " lo32=", uint32_t(raw & 0xffffffffull), ")");
                }
              }
            }

            // NV-DXVK [perf]: the device limits needed to turn the per-shader
            // numbers above into an occupancy figure. Logged once, next to the
            // statistics they qualify, because a Register Count of 255 means
            // nothing without the register file size and a Shared Memory Size
            // means nothing without the per-block cap.
            //
            // Specifically: occupancy = min(regFile / regsPerThread,
            // sharedBudget / sharedPerBlock * blockThreads, ...). At 255
            // registers the register term is already at its floor, so shared
            // memory is the only remaining way to move occupancy - and whether
            // that is even possible depends on maxComputeSharedMemorySize.
            {
              static bool sLoggedLimits = false;
              if (!sLoggedLimits) {
                sLoggedLimits = true;
                const VkPhysicalDeviceLimits& lim =
                  m_pipeMgr->m_device->properties().core.properties.limits;
                Logger::warn(str::format(
                  "[Perf.Limits] maxComputeSharedMemorySize=", lim.maxComputeSharedMemorySize,
                  " maxComputeWorkGroupInvocations=", lim.maxComputeWorkGroupInvocations,
                  " maxComputeWorkGroupSize=", lim.maxComputeWorkGroupSize[0],
                  "x", lim.maxComputeWorkGroupSize[1],
                  "x", lim.maxComputeWorkGroupSize[2],
                  " subgroupSize=", execs[i].subgroupSize));
              }
            }

            Logger::warn(str::format(
              "[Perf.Shader] cs=", m_shaders.cs->debugName(),
              " exec=", execs[i].name,
              " subgroupSize=", execs[i].subgroupSize,
              " |", line));
          }
        }
      }
    }

    return pipeline;
  }


  void DxvkComputePipeline::destroyPipeline(VkPipeline pipeline) {
    m_vkd->vkDestroyPipeline(m_vkd->device(), pipeline, nullptr);
  }
  
  
  void DxvkComputePipeline::writePipelineStateToCache(
    const DxvkComputePipelineStateInfo& state) const {
    assert(!m_shaders.forceNoSpecConstants);

    if (m_pipeMgr->m_stateCache == nullptr)
      return;
    
    DxvkStateCacheKey key;

    if (m_shaders.cs != nullptr)
      key.cs = m_shaders.cs->getShaderKey();

    m_pipeMgr->m_stateCache->addComputePipeline(key, state);
  }
  
}
