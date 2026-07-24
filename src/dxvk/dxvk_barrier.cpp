#include "dxvk_barrier.h"
#include <assert.h>

#include "../util/util_string.h"

namespace dxvk {

  namespace barrierstats {

    Bucket                g_buckets[kBuckets];
    std::atomic<uint64_t> g_total    { 0ull };
    std::atomic<uint64_t> g_overflow { 0ull };

    void record(
            VkPipelineStageFlags srcStages,
            VkPipelineStageFlags dstStages,
            size_t               imgCount,
            size_t               bufCount,
            bool                 hasMemBarrier) {
      g_total.fetch_add(1, std::memory_order_relaxed);

      const uint64_t key = (uint64_t(srcStages) << 32) | uint64_t(dstStages);

      // Linear probe. Keys are claimed once with a CAS and never removed, so a
      // reader can walk the table without locking.
      for (size_t i = 0; i < kBuckets; ++i) {
        Bucket& b = g_buckets[i];
        uint64_t cur = b.key.load(std::memory_order_acquire);

        if (cur != key) {
          if (cur != 0ull)
            continue;                       // occupied by another key
          uint64_t expected = 0ull;
          if (!b.key.compare_exchange_strong(expected, key,
                                             std::memory_order_acq_rel))
            if (expected != key)
              continue;                     // lost the race to a different key
        }

        b.count.fetch_add(1, std::memory_order_relaxed);
        b.imgN.fetch_add(imgCount, std::memory_order_relaxed);
        b.bufN.fetch_add(bufCount, std::memory_order_relaxed);
        if (hasMemBarrier)
          b.memN.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      g_overflow.fetch_add(1, std::memory_order_relaxed);
    }


    std::string stageName(VkPipelineStageFlags stages) {
      if (stages == 0)
        return "-";
      if (stages & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
        return "ALL";

      struct Entry { VkPipelineStageFlags bit; const char* name; };
      static constexpr Entry kEntries[] = {
        { VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,                          "TOP"    },
        { VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,                        "INDIR"  },
        { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,                         "VIN"    },
        { VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,                        "VS"     },
        { VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,                      "GS"     },
        { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,                      "FS"     },
        { VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,                 "EARLYZ" },
        { VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,                  "LATEZ"  },
        { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,              "COLOR"  },
        { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                       "CS"     },
        { VK_PIPELINE_STAGE_TRANSFER_BIT,                             "XFER"   },
        { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,                       "BOT"    },
        { VK_PIPELINE_STAGE_HOST_BIT,                                 "HOST"   },
        { VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,                         "GFX"    },
        { VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,     "ASBUILD"},
        { VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,               "RT"     },
      };

      std::string out;
      for (const auto& e : kEntries) {
        if (stages & e.bit) {
          if (!out.empty())
            out += "|";
          out += e.name;
        }
      }

      return out.empty() ? str::format("0x", std::hex, uint32_t(stages)) : out;
    }

  }


  DxvkBarrierSet:: DxvkBarrierSet(DxvkCmdBuffer cmdBuffer)
  : m_cmdBuffer(cmdBuffer) {

  }


  DxvkBarrierSet::~DxvkBarrierSet() {

  }

  
  void DxvkBarrierSet::accessMemory(
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    m_srcStages |= srcStages;
    m_dstStages |= dstStages;
    
    m_srcAccess |= srcAccess;
    m_dstAccess |= dstAccess;
  }


  void DxvkBarrierSet::accessBuffer(
    const DxvkBufferSliceHandle&    bufSlice,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    DxvkAccessFlags access = this->getAccessTypes(srcAccess);
    
    if (srcStages == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
     || dstStages == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
      access.set(DxvkAccess::Write);
    
    m_srcStages |= srcStages;
    m_dstStages |= dstStages;
    
    m_srcAccess |= srcAccess;
    m_dstAccess |= dstAccess;

    m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
  }
  
  
  void DxvkBarrierSet::accessImage(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  subresources,
          VkImageLayout             srcLayout,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkImageLayout             dstLayout,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    DxvkAccessFlags access = this->getAccessTypes(srcAccess);

    assert(dstLayout != VK_IMAGE_LAYOUT_UNDEFINED);

    if (srcStages == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
     || dstStages == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
     || srcLayout != dstLayout)
      access.set(DxvkAccess::Write);
    
    m_srcStages |= srcStages;
    m_dstStages |= dstStages;
    
    if (srcLayout == dstLayout) {
      m_srcAccess |= srcAccess;
      m_dstAccess |= dstAccess;
    } else {
      VkImageMemoryBarrier barrier;
      barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.pNext                       = nullptr;
      barrier.srcAccessMask               = srcAccess;
      barrier.dstAccessMask               = dstAccess;
      barrier.oldLayout                   = srcLayout;
      barrier.newLayout                   = dstLayout;
      barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
      barrier.image                       = image->handle();
      barrier.subresourceRange            = subresources;
      barrier.subresourceRange.aspectMask = image->formatInfo()->aspectMask;
      m_imgBarriers.push_back(barrier);
    }

    m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
  }


  void DxvkBarrierSet::releaseBuffer(
          DxvkBarrierSet&           acquire,
    const DxvkBufferSliceHandle&    bufSlice,
          uint32_t                  srcQueue,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          uint32_t                  dstQueue,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    auto& release = *this;

    release.m_srcStages |= srcStages;
    acquire.m_dstStages |= dstStages;

    VkBufferMemoryBarrier barrier;
    barrier.sType                       = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext                       = nullptr;
    barrier.srcAccessMask               = srcAccess;
    barrier.dstAccessMask               = 0;
    barrier.srcQueueFamilyIndex         = srcQueue;
    barrier.dstQueueFamilyIndex         = dstQueue;
    barrier.buffer                      = bufSlice.handle;
    barrier.offset                      = bufSlice.offset;
    barrier.size                        = bufSlice.length;
    release.m_bufBarriers.push_back(barrier);

    barrier.srcAccessMask               = 0;
    barrier.dstAccessMask               = dstAccess;
    acquire.m_bufBarriers.push_back(barrier);

    DxvkAccessFlags access(DxvkAccess::Read, DxvkAccess::Write);
    release.m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
    acquire.m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
  }


  void DxvkBarrierSet::releaseImage(
          DxvkBarrierSet&           acquire,
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  subresources,
          uint32_t                  srcQueue,
          VkImageLayout             srcLayout,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          uint32_t                  dstQueue,
          VkImageLayout             dstLayout,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    auto& release = *this;

    release.m_srcStages |= srcStages;
    acquire.m_dstStages |= dstStages;

    VkImageMemoryBarrier barrier;
    barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext                       = nullptr;
    barrier.srcAccessMask               = srcAccess;
    barrier.dstAccessMask               = 0;
    barrier.oldLayout                   = srcLayout;
    barrier.newLayout                   = dstLayout;
    barrier.srcQueueFamilyIndex         = srcQueue;
    barrier.dstQueueFamilyIndex         = dstQueue;
    barrier.image                       = image->handle();
    barrier.subresourceRange            = subresources;
    barrier.subresourceRange.aspectMask = image->formatInfo()->aspectMask;
    release.m_imgBarriers.push_back(barrier);

    if (srcQueue == dstQueue)
      barrier.oldLayout = dstLayout;

    barrier.srcAccessMask               = 0;
    barrier.dstAccessMask               = dstAccess;
    acquire.m_imgBarriers.push_back(barrier);

    DxvkAccessFlags access(DxvkAccess::Read, DxvkAccess::Write);
    release.m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
    acquire.m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
  }


  bool DxvkBarrierSet::isBufferDirty(
    const DxvkBufferSliceHandle&    bufSlice,
          DxvkAccessFlags           bufAccess) {
    return m_bufSlices.isDirty(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, bufAccess));
  }


  bool DxvkBarrierSet::isImageDirty(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  imgSubres,
          DxvkAccessFlags           imgAccess) {
    return m_imgSlices.isDirty(image->handle(),
      DxvkBarrierImageSlice(imgSubres, imgAccess));
  }


  DxvkAccessFlags DxvkBarrierSet::getBufferAccess(
    const DxvkBufferSliceHandle&    bufSlice) {
    return m_bufSlices.getAccess(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, 0));
  }

  
  DxvkAccessFlags DxvkBarrierSet::getImageAccess(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  imgSubres) {
    return m_imgSlices.getAccess(image->handle(),
      DxvkBarrierImageSlice(imgSubres, 0));
  }


  void DxvkBarrierSet::recordCommands(const Rc<DxvkCommandList>& commandList) {
    if (m_srcStages | m_dstStages) {
      VkPipelineStageFlags srcFlags = m_srcStages;
      VkPipelineStageFlags dstFlags = m_dstStages;
      
      if (!srcFlags) srcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      if (!dstFlags) dstFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

      VkMemoryBarrier memBarrier;
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.pNext = nullptr;
      memBarrier.srcAccessMask = m_srcAccess;
      memBarrier.dstAccessMask = m_dstAccess;

      VkMemoryBarrier* pMemBarrier = nullptr;
      if (m_srcAccess | m_dstAccess)
        pMemBarrier = &memBarrier;
      
      commandList->cmdPipelineBarrier(
        m_cmdBuffer, srcFlags, dstFlags, 0,
        pMemBarrier ? 1 : 0, pMemBarrier,
        m_bufBarriers.size(),
        m_bufBarriers.data(),
        m_imgBarriers.size(),
        m_imgBarriers.data());
      
      commandList->addStatCtr(DxvkStatCounter::CmdBarrierCount, 1);

      // NV-DXVK [perf]: attribute this barrier by stage pair. See barrierstats.
      barrierstats::record(srcFlags, dstFlags,
                           m_imgBarriers.size(), m_bufBarriers.size(),
                           pMemBarrier != nullptr);

      this->reset();
    }
  }
  
  
  void DxvkBarrierSet::reset() {
    m_srcStages = 0;
    m_dstStages = 0;

    m_srcAccess = 0;
    m_dstAccess = 0;
    
    m_bufBarriers.resize(0);
    m_imgBarriers.resize(0);

    m_bufSlices.clear();
    m_imgSlices.clear();
  }
  
  
  DxvkAccessFlags DxvkBarrierSet::getAccessTypes(VkAccessFlags flags) {
    const VkAccessFlags rflags
      = VK_ACCESS_INDIRECT_COMMAND_READ_BIT
      | VK_ACCESS_INDEX_READ_BIT
      | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
      | VK_ACCESS_UNIFORM_READ_BIT
      | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT
      | VK_ACCESS_SHADER_READ_BIT
      | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
      | VK_ACCESS_TRANSFER_READ_BIT
      | VK_ACCESS_HOST_READ_BIT
      | VK_ACCESS_MEMORY_READ_BIT
      | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;
      
    const VkAccessFlags wflags
      = VK_ACCESS_SHADER_WRITE_BIT
      | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_TRANSFER_WRITE_BIT
      | VK_ACCESS_HOST_WRITE_BIT
      | VK_ACCESS_MEMORY_WRITE_BIT
      | VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT
      | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
    
    DxvkAccessFlags result;
    if (flags & rflags) result.set(DxvkAccess::Read);
    if (flags & wflags) result.set(DxvkAccess::Write);
    return result;
  }
  
}