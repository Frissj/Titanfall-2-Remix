/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "../dxvk_device.h"
#include "../dxvk_context.h"
#include "rtx_scene_manager.h"
#include "rtx_resources.h"
#include "rtx_bindless_resource_manager.h"

#include "../shaders/rtx/pass/common_binding_indices.h"
#include "../dxvk_descriptor.h"

namespace dxvk {

  BindlessResourceManager::BindlessResourceManager(DxvkDevice* device)
  : CommonDeviceObject(device) { 
    for (int i = 0; i < kMaxFramesInFlight; i++) {
      m_tables[Table::Textures][i].reset(new BindlessTable(this));
      m_tables[Table::Buffers][i].reset(new BindlessTable(this));
      m_tables[Table::Samplers][i].reset(new BindlessTable(this));
    }

    createGlobalBindlessDescPool();
  }

  const Rc<vk::DeviceFn> BindlessResourceManager::BindlessTable::vkd() const {
    return m_pManager->m_device->vkd();
  }

  VkDescriptorSet BindlessResourceManager::getGlobalBindlessTableSet(Table type) const {
    if (m_frameLastUpdated != m_device->getCurrentFrameId())
      throw DxvkError("Getting bindless table before it's been updated for this frame!!");

    return m_tables[type][currentIdx()]->bindlessDescSet;
  }

  template<VkDescriptorType Type, typename T, typename U>
  void BindlessResourceManager::createDescriptorSet(const Rc<DxvkContext>& ctx, const std::vector<U>& engineObjects, const T& dummyDescriptor) {
    const size_t numDescriptors = std::max((size_t) 1, engineObjects.size()); // Must always leave 1 to have a valid binding set
    assert(numDescriptors <= kMaxBindlessResources);

    std::vector<T> descriptorInfos(numDescriptors);
    descriptorInfos[0] = dummyDescriptor; // we set the first descriptor to be a dummy (size is always at least 1) and overwrite it if there are valid engine objects

    // NV-DXVK [BindlessDrop]: per-slot validity this frame (texture table only).
    // NV-DXVK [MatChurn]: and the handle itself, so a slot swapping one live
    // image for another is visible as well as a slot going dummy.
    std::vector<uint8_t> curTexSlotValid;
    std::vector<uint64_t> curTexSlotView;
    if constexpr (Type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
      curTexSlotValid.assign(engineObjects.size(), 0u);
      curTexSlotView.assign(engineObjects.size(), 0ull);
    }

    uint32_t idx = 0;
    for (auto&& engineObject : engineObjects) {
      descriptorInfos[idx] = dummyDescriptor;

      if constexpr (Type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        DxvkImageView* imageView = engineObject.getImageView();
        if (imageView != nullptr) {
          descriptorInfos[idx].sampler = nullptr;
          descriptorInfos[idx].imageView = imageView->handle();
          descriptorInfos[idx].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          ctx->getCommandList()->trackResource<DxvkAccess::Read>(imageView);
          curTexSlotValid[idx] = 1u;
          curTexSlotView[idx] = reinterpret_cast<uint64_t>(imageView->handle());
        }
      } else if constexpr (Type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
        if (engineObject.defined()) {
          descriptorInfos[idx] = engineObject.getDescriptor().buffer;
          ctx->getCommandList()->trackResource<DxvkAccess::Read>(engineObject.buffer());
        }
      } else if constexpr (Type == VK_DESCRIPTOR_TYPE_SAMPLER) {
        if (engineObject != nullptr) {
          descriptorInfos[idx].sampler = engineObject->handle();
          descriptorInfos[idx].imageView = nullptr;
        }
      } else {
        static_assert(Type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || Type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || Type != VK_DESCRIPTOR_TYPE_SAMPLER, "Support for this descriptor type has not been implemented yet.");
        return;
      }

      ++idx;
    }

    // NV-DXVK [BindlessDrop]: report texture slots that flipped valid->dummy
    // since last frame. A drop means every surface whose material references
    // that slot samples the DUMMY texture for exactly this frame — a
    // one-frame, whole-group material flicker with geometry, VS attribution
    // and surface bytes all provably intact (which is precisely the state
    // every geometry-side probe of 2026-08-02 converged on). The game swaps/
    // frees texture views asynchronously (see the getImageHash UAF finding),
    // so transient null views here are expected to be the mechanism; this
    // log makes each occurrence joinable to the on-screen flick by frame id.
    if constexpr (Type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
      uint32_t dropped = 0u, recovered = 0u, changed = 0u;
      uint32_t firstDropped[8] = {};
      uint32_t nFirst = 0u;
      const size_t nCompare = std::min(curTexSlotValid.size(), m_prevTexSlotValid.size());
      for (size_t s = 0; s < nCompare; ++s) {
        if (m_prevTexSlotValid[s] != 0u && curTexSlotValid[s] == 0u) {
          ++dropped;
          if (nFirst < 8u) { firstDropped[nFirst++] = uint32_t(s); }
        } else if (m_prevTexSlotValid[s] == 0u && curTexSlotValid[s] != 0u) {
          ++recovered;
        } else if (curTexSlotValid[s] != 0u && m_prevTexSlotView[s] != curTexSlotView[s]) {
          // NV-DXVK [MatChurn]: still valid, but pointing at a DIFFERENT image
          // than last frame. Every material holding this slot index now samples
          // something else, with nothing about the material itself having moved.
          ++changed;
        }
      }
      if (dropped != 0u || recovered != 0u) {
        std::string slots;
        for (uint32_t i = 0; i < nFirst; ++i) {
          slots += (i ? "," : "");
          slots += str::format(firstDropped[i]);
        }
        Logger::info(str::format(
          "[BindlessDrop] f=", m_device->getCurrentFrameId(),
          " slots=", curTexSlotValid.size(),
          " dropped=", dropped,
          " recovered=", recovered,
          " firstDropped=[", slots, "]"));
      }

      // NV-DXVK [MatChurn]: published for the per-frame churn line. Slots past
      // the previous table's length are new entries, not changes, so they are
      // reported separately instead of inflating changed/recovered.
      m_texTableStats.slots = uint32_t(curTexSlotValid.size());
      m_texTableStats.changed = changed;
      m_texTableStats.dropped = dropped;
      m_texTableStats.recovered = recovered;
      m_texTableStats.grew = uint32_t(curTexSlotValid.size() - nCompare);

      m_prevTexSlotValid = std::move(curTexSlotValid);
      m_prevTexSlotView = std::move(curTexSlotView);
    }

    // NV-DXVK [BindlessTail]: leave no undefined slot in the table.
    //
    // The write above covers [0, liveCount). The binding is declared for
    // kMaxBindlessResources slots with VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
    // and the set is allocated once and never cleared, so before this every
    // slot past liveCount was either uninitialised descriptor memory or a
    // stale live descriptor from an earlier, larger frame. Reading one is
    // undefined behaviour, and on NVIDIA that means the hardware follows the
    // arbitrary 64-bit address it finds -- a DMA page fault that takes the
    // device down, not a wrong pixel. Nothing bounds the index on the way in
    // (BUFFER_ARRAY is a raw geometries[NonUniformResourceIndex(idx)][elem]),
    // and the index reaches the shader through grow-only buffers, so an index
    // from a larger era stays readable after the scene shrinks.
    //
    // Two distinct windows, closed two different ways:
    //   - never-written slots: one whole-table dummy fill, once per set. This
    //     is the expensive branch and it runs exactly once per table per
    //     frame-in-flight, at startup.
    //   - slots that WERE live and are now past the count: re-dummied on the
    //     frames where the count drops. Those hold descriptors whose buffers
    //     are no longer tracked (trackResource runs only for written slots),
    //     so they can point at freed and unmapped memory.
    //
    // This does not fix a stale index -- it makes the read defined, so an
    // out-of-range index samples the dummy and gets counted instead of
    // killing the device.
    Table tableType = Table::Count;
    switch (Type) {
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:  tableType = Table::Textures; break;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: tableType = Table::Buffers;  break;
    case VK_DESCRIPTOR_TYPE_SAMPLER:        tableType = Table::Samplers; break;
    default:
      return;
    }

    BindlessTable* const table = m_tables[tableType][currentIdx()].get();

    // createLayout declares every table -- samplers included -- for
    // kMaxBindlessResources, so that is the count that has to be defined.
    // (kMaxBindlessSamplers exists in the header but is referenced nowhere.)
    const uint32_t liveCount = uint32_t(numDescriptors);
    const uint32_t writeCount = table->fullyInitialized
      ? std::max(liveCount, table->liveSlots)   // only re-cover a shrink
      : kMaxBindlessResources;                  // one-time: define everything

    // resize() fills the tail with the dummy; [0, liveCount) is already built.
    descriptorInfos.resize(writeCount, dummyDescriptor);

    VkWriteDescriptorSet descWrites;
    memset(&descWrites, 0, sizeof(descWrites));
    descWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrites.descriptorCount = writeCount;
    descWrites.descriptorType = Type;

    if constexpr (std::is_same_v<T, VkDescriptorImageInfo>) {
      descWrites.pImageInfo = &descriptorInfos[0];
    } else if constexpr (std::is_same_v<T, VkDescriptorBufferInfo>) {
      descWrites.pBufferInfo = &descriptorInfos[0];
    }

    table->updateDescriptors(descWrites);

    // Only claim the table is initialised once a write actually landed --
    // updateDescriptors returns without writing if the set allocation failed.
    const bool wrote = (table->bindlessDescSet != VK_NULL_HANDLE);
    const uint32_t reDummied = (writeCount > liveCount) ? (writeCount - liveCount) : 0u;

    if (wrote) {
      const bool wasFirstWrite = !table->fullyInitialized;
      table->fullyInitialized = true;
      table->liveSlots = liveCount;

      TableStats& stats = m_tableStats[tableType];
      stats.live = liveCount;
      stats.peakLive = std::max(stats.peakLive, liveCount);
      // The one-time fill covers the whole table and would report a
      // meaningless ~64k "shrink"; only a real shrink is interesting.
      stats.reDummied = wasFirstWrite ? 0u : reDummied;
    }
  }

  void BindlessResourceManager::prepareSceneData(const Rc<DxvkContext> ctx, const std::vector<TextureRef>& rtTextures, const std::vector<RaytraceBuffer>& rtBuffers, const std::vector<Rc<DxvkSampler>>& samplers) {
    ScopedCpuProfileZone();
    if (m_frameLastUpdated == m_device->getCurrentFrameId()) {
      Logger::debug("Updating bindless tables multiple times per frame...");
      return;
    }

    // Increment
    m_globalBindlessDescSetIdx = nextIdx();

    const VkDescriptorImageInfo dummyImage = m_device->getCommon()->dummyResources().imageViewDescriptor(VK_IMAGE_VIEW_TYPE_2D, true);
    const VkDescriptorBufferInfo dummyBuffer = m_device->getCommon()->dummyResources().bufferDescriptor();
    const VkDescriptorImageInfo dummySampler = m_device->getCommon()->dummyResources().samplerDescriptor();

    createDescriptorSet<VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE>(ctx, rtTextures, dummyImage);
    createDescriptorSet<VK_DESCRIPTOR_TYPE_STORAGE_BUFFER>(ctx, rtBuffers, dummyBuffer);
    createDescriptorSet<VK_DESCRIPTOR_TYPE_SAMPLER>(ctx, samplers, dummySampler);

    m_frameLastUpdated = m_device->getCurrentFrameId();
  }

  BindlessResourceManager::BindlessTable::~BindlessTable() {
    if (layout != VK_NULL_HANDLE) {
      vkd()->vkDestroyDescriptorSetLayout(vkd()->device(), layout, nullptr);
    }
  }

  void BindlessResourceManager::BindlessTable::createLayout(const VkDescriptorType type) {
    assert(bindlessDescSet == nullptr); // can't update the layout if we already allocated a descriptor

    static const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBinding binding;
    binding.descriptorType = type;
    binding.descriptorCount = kMaxBindlessResources;
    binding.binding = 0; // Tables always bound at 0
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT |
                          VK_SHADER_STAGE_RAYGEN_BIT_KHR | 
                          VK_SHADER_STAGE_ANY_HIT_BIT_KHR | 
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | 
                          VK_SHADER_STAGE_INTERSECTION_BIT_KHR | 
                          VK_SHADER_STAGE_CALLABLE_BIT_KHR | 
                          VK_SHADER_STAGE_MISS_BIT_KHR;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    layoutInfo.flags = 0;

    VkDescriptorSetLayoutBindingFlagsCreateInfo extendedInfo { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, nullptr };
    extendedInfo.bindingCount = 1;
    extendedInfo.pBindingFlags = &flags;

    layoutInfo.pNext = &extendedInfo;

    if (vkd()->vkCreateDescriptorSetLayout(m_pManager->m_device->vkd()->device(), &layoutInfo, nullptr, &layout) != VK_SUCCESS)
      throw DxvkError("BindlessTable: Failed to create descriptor set layout");
  }

  void BindlessResourceManager::BindlessTable::updateDescriptors(VkWriteDescriptorSet set) {
    if (bindlessDescSet == nullptr) {
      // Allocate the descriptor set
      bindlessDescSet = m_pManager->m_globalBindlessPool[m_pManager->currentIdx()]->alloc(layout, "bindless descriptor set");
      if (bindlessDescSet == nullptr) {
        Logger::err(str::format("BindlessTable: failed to allocate a descriptor set for ", set.descriptorCount, " ",
                                (set.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ? "buffers" : "textures"));
        return;
      }
    }

    // Update the write descriptor with our set
    set.dstSet = bindlessDescSet;

    // Do the write
    vkd()->vkUpdateDescriptorSets(vkd()->device(), 1, &set, 0, nullptr);
  }

  void BindlessResourceManager::createGlobalBindlessDescPool() {
    // Create bindless descriptor pool
    static std::array<VkDescriptorPoolSize, Table::Count> pools = { {
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          kMaxBindlessResources * kMaxFramesInFlight },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kMaxBindlessResources * kMaxFramesInFlight },
        { VK_DESCRIPTOR_TYPE_SAMPLER,                kMaxBindlessResources * kMaxFramesInFlight }
    } };

    VkDescriptorPoolCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.maxSets = pools.size() * kMaxFramesInFlight;
    info.poolSizeCount = pools.size();
    info.pPoolSizes = pools.data();

    // Create the global pool
    for (uint32_t i = 0; i < kMaxFramesInFlight; i++) {
      // [DescPoolDiag] tag each frame's bindless pool individually so a
      // VK validation message referencing this pool's handle resolves to
      // a specific role+frame ("bindless-frame2" etc.). The bindless pool
      // intentionally serves only SAMPLED_IMAGE / STORAGE_BUFFER /
      // SAMPLER — any UNIFORM_BUFFER allocation against one of these is
      // a layout/pool routing bug.
      const std::string role = str::format("bindless-frame", i);
      m_globalBindlessPool[i] = new DxvkDescriptorPool(m_device->instance()->vki(), m_device->vkd(), info, role.c_str());
      m_tables[Table::Textures][i]->createLayout(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
      m_tables[Table::Buffers][i]->createLayout(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      m_tables[Table::Samplers][i]->createLayout(VK_DESCRIPTOR_TYPE_SAMPLER);
    }
  }

} // namespace dxvk 