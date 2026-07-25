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

#include "../util/util_time.h"

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

    { std::lock_guard<sync::Spinlock> lock(m_mutex);

      instance = this->findInstance(state);

      if (instance)
        return instance->pipeline();
    
      // If no pipeline instance exists with the given state
      // vector, create a new one and add it to the list.
      instance = this->createInstance(state);
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
    if (m_pipeMgr->m_device->extensions().khrPipelineExecutableProperties) {
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

            std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
            for (auto& s : stats)
              s = VkPipelineExecutableStatisticKHR { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR };

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
