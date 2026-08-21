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
#include "rtx_utils.h"
#include "rtx_common_object.h"

namespace dxvk {
  class DxvkDevice;
  class DxvkCommandList;

  class BindlessResourceManager : public CommonDeviceObject {
  public:
    friend struct BindlessTable;

    enum Table {
      Textures = 0,
      Buffers,
      Samplers,
      Count
    };

    static const uint32_t kMaxBindlessResources = 64 * 1024; // our indices are uint16_t...
    static const uint32_t kMaxBindlessSamplers = 2048; // this is the lowest max number of samplers our device base supports for remix (VkPhysicalDeviceLimits::maxDescriptorSetSamplers)

    BindlessResourceManager() = delete;

    explicit BindlessResourceManager(DxvkDevice* device);

    void prepareSceneData(const Rc<DxvkContext> ctx, const std::vector<TextureRef>& rtTextures, const std::vector<RaytraceBuffer>& rtBuffers, const std::vector<Rc<DxvkSampler>>& samplers);

    VkDescriptorSet getGlobalBindlessTableSet(Table type) const;

    VkDescriptorSetLayout getGlobalBindlessTableLayout(Table type) const {
      return m_tables[type][currentIdx()]->layout;
    }

    // NV-DXVK [MatChurn]: what the last texture-table rebuild actually wrote.
    //
    // Every frame the whole texture table is rewritten from the texture cache's
    // object table, so "a descriptor write happened" says nothing on its own -
    // what matters is whether the CONTENT of a slot moved, because a material
    // holds a slot INDEX and samples whatever that slot points at this frame:
    //
    //   slots      table length (== texture cache total count).
    //   changed    slots whose VkImageView handle differs from last frame. A
    //              material referencing one of these samples a different image
    //              than it did last frame without anything about the material,
    //              the surface or the geometry having changed.
    //   dropped    slots that went from a real view to the DUMMY descriptor.
    //              Those surfaces sample the dummy for exactly this frame - a
    //              one-frame, whole-group albedo flicker.
    //   recovered  the reverse.
    //   grew       slots appended past last frame's table length (new textures).
    struct TextureTableStats {
      uint32_t slots = 0;
      uint32_t changed = 0;
      uint32_t dropped = 0;
      uint32_t recovered = 0;
      uint32_t grew = 0;
    };

    const TextureTableStats& getTextureTableStats() const { return m_texTableStats; }

    // NV-DXVK [BindlessTail]: the measurement the device-loss chain was missing.
    //
    // The buffer table's per-frame length was never recorded anywhere -- the
    // only bindless counter in the log ([MatChurn] blSlots) is the TEXTURE
    // table, which in TF2 only ever grows, so it cannot see the shrink that
    // opens the undefined window. These are per table, so `Buffers` shrinking
    // while `Textures` grows is visible as such.
    //
    //   live       real descriptors written this frame.
    //   peakLive   high-water of `live` since startup.
    //   reDummied  slots that held a live descriptor last cycle and are past
    //              this frame's count, so they were overwritten with the dummy.
    //              NON-ZERO IS THE SHRINK: before this fix those slots kept
    //              serving stale descriptors to any out-of-range index.
    struct TableStats {
      uint32_t live = 0;
      uint32_t peakLive = 0;
      uint32_t reDummied = 0;
    };

    const TableStats& getTableStats(Table type) const { return m_tableStats[type]; }

  private:

    struct BindlessTable {
      BindlessTable() = delete;

      BindlessTable(BindlessResourceManager* pManager)
        : m_pManager(pManager) { }
      ~BindlessTable();

      VkDescriptorSetLayout layout = VK_NULL_HANDLE;
      VkDescriptorSet bindlessDescSet = VK_NULL_HANDLE;

      // NV-DXVK [BindlessTail]: this set is declared for kMaxBindlessResources
      // slots, allocated ONCE, and never cleared -- but each frame only wrote
      // engineObjects.size() of them. Every slot past that was undefined
      // memory, and the binding carries VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
      // which makes reading one undefined behaviour rather than an error: the
      // hardware follows whatever 64-bit address the slot happens to hold.
      // That is a GPU page fault, not a wrong pixel.
      //
      // Nothing bounds the index on the way in -- BUFFER_ARRAY() is a raw
      // geometries[NonUniformResourceIndex(idx)][elem] -- and the index comes
      // from Surface data reached through a TLAS instance's customIndex, both
      // of which live in grow-only buffers. So when the scene shrinks, an
      // index from the larger era stays readable and lands in the tail.
      //
      //   fullyInitialized  every slot has been written at least once, so no
      //                     slot is undefined any more. Set by the one-time
      //                     whole-table dummy fill on the first update.
      //   liveSlots         slots this set last wrote with real descriptors.
      //                     Slots between a shrunk count and this still hold
      //                     the PREVIOUS cycle's live descriptors, which point
      //                     at buffers no longer tracked (trackResource runs
      //                     only for written slots) and therefore free to be
      //                     destroyed -- so they are re-dummied on shrink.
      bool fullyInitialized = false;
      uint32_t liveSlots = 0;

      void createLayout(const VkDescriptorType type);
      void updateDescriptors(VkWriteDescriptorSet set);

    private:
      const Rc<vk::DeviceFn> vkd() const;

      BindlessResourceManager* m_pManager = nullptr;
    };

    // Persistent desc pool, our sets can be updated after bind (should be no need to reset this pool)
    Rc<DxvkDescriptorPool> m_globalBindlessPool[kMaxFramesInFlight];
    
    std::unique_ptr<BindlessTable> m_tables[Table::Count][kMaxFramesInFlight];

    uint32_t m_globalBindlessDescSetIdx = 0;
    uint32_t m_frameLastUpdated = UINT_MAX;

    // NV-DXVK [BindlessDrop]: per-texture-slot validity from the previous
    // frame's descriptor build. A slot that was valid last frame and falls
    // back to the dummy descriptor this frame means every surface sampling
    // that texture renders with the dummy for exactly this frame — the
    // one-frame group "material flicker" shape. Tracks valid->dummy and
    // dummy->valid transitions; see createDescriptorSet.
    std::vector<uint8_t> m_prevTexSlotValid;

    // NV-DXVK [MatChurn]: the actual VkImageView handle each texture slot held
    // last frame. m_prevTexSlotValid can only see a slot fall back to the dummy;
    // this sees a slot swap one real image for another, which is the streaming
    // rename case and leaves validity untouched.
    std::vector<uint64_t> m_prevTexSlotView;

    TextureTableStats m_texTableStats;

    // NV-DXVK [BindlessTail]: see getTableStats.
    TableStats m_tableStats[Table::Count];


    uint32_t currentIdx() const {
      return m_globalBindlessDescSetIdx;
    }

    uint32_t nextIdx() const {
      return (m_globalBindlessDescSetIdx + 1) % kMaxFramesInFlight;
    }

    void createGlobalBindlessDescPool();

    template<VkDescriptorType Type, typename T, typename U>
    void createDescriptorSet(const Rc<DxvkContext>& ctx, const std::vector<U>& engineObjects, const T& dummyDescriptor);
  };
} // namespace dxvk 