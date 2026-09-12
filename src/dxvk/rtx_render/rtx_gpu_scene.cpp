#include "rtx_gpu_scene.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  // ==========================================================================
  // SurfaceSlotTable
  // ==========================================================================

  bool SurfaceSlotTable::beginFrame(uint32_t compactSlack) {
    // 0 is the "never acquired" epoch of a default Run, so skip it on wrap.
    if (++m_epoch == 0u) {
      m_epoch = 1u;
    }

    // COMPACTION. Holes are slots below the high-water mark that no run owns.
    // They cost nothing on the GPU path (nothing references them), but every
    // per-slot walk and every per-slot buffer is sized by the high-water mark,
    // so an unbounded hole count is O(history) work. Compact only when holes
    // exceed BOTH the slack and the live count -- the table is then more than
    // half empty -- so a steady scene never pays for it. Relocation preserves
    // base order, which keeps neighbouring runs neighbours.
    const uint32_t holes = m_highWater - m_liveSlots;
    if (holes <= compactSlack || holes <= m_liveSlots) {
      return false;
    }

    std::vector<std::pair<uint32_t, uint64_t>> order;
    order.reserve(m_runs.size());
    for (const auto& [key, run] : m_runs) {
      order.emplace_back(run.base, key);
    }
    std::sort(order.begin(), order.end());

    uint32_t next = 0u;
    for (const auto& [base, key] : order) {
      Run& run = m_runs[key];
      run.base = next;
      next += run.count;
    }
    m_highWater = next;
    m_free.clear();
    m_freedThisWalk.clear();
    ++m_stats.compactions;
    return true;
  }

  void SurfaceSlotTable::insertFree(std::vector<std::pair<uint32_t, uint32_t>>& list,
                                    uint32_t base, uint32_t count) {
    if (count == 0u) {
      return;
    }
    auto it = std::lower_bound(list.begin(), list.end(), std::make_pair(base, 0u));
    it = list.insert(it, std::make_pair(base, count));

    // Coalesce with the successor, then the predecessor.
    auto nx = it + 1;
    if (nx != list.end() && it->first + it->second == nx->first) {
      it->second += nx->second;
      list.erase(nx);
    }
    if (it != list.begin()) {
      auto pv = it - 1;
      if (pv->first + pv->second == it->first) {
        pv->second += it->second;
        list.erase(it);
      }
    }
  }

  uint32_t SurfaceSlotTable::allocRange(uint32_t count, uint32_t maxSlotValue) {
    // Best fit, lowest base on a tie, so the table stays packed at the bottom
    // and trailing space can be trimmed off the high-water mark.
    size_t best = m_free.size();
    uint32_t bestCount = ~0u;
    for (size_t i = 0; i < m_free.size(); ++i) {
      const uint32_t c = m_free[i].second;
      if (c >= count && c < bestCount) {
        best = i;
        bestCount = c;
        if (c == count) {
          break;
        }
      }
    }
    if (best != m_free.size()) {
      const uint32_t base = m_free[best].first;
      if (m_free[best].second == count) {
        m_free.erase(m_free.begin() + best);
      } else {
        m_free[best].first += count;
        m_free[best].second -= count;
      }
      return base;
    }

    if (uint64_t(m_highWater) + count - 1u > maxSlotValue) {
      return kNoSlot;
    }
    const uint32_t base = m_highWater;
    m_highWater += count;
    return base;
  }

  uint32_t SurfaceSlotTable::acquire(uint64_t key, uint32_t count, uint32_t maxSlotValue) {
    if (count == 0u) {
      return kNoSlot;
    }

    uint64_t k = key;
    for (;;) {
      auto it = m_runs.find(k);
      if (it == m_runs.end()) {
        break;
      }
      Run& run = it->second;

      if (run.epoch == m_epoch) {
        // Second acquire of one key in one walk. The old per-push path gave
        // every push its own slot, so this must too -- re-key
        // deterministically, which keeps the duplicate's run stable across
        // frames as long as the acquire order is.
        ++m_stats.rekeyed;
        k = k * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
        continue;
      }

      if (run.count == count) {
        run.epoch = m_epoch;
        ++m_stats.kept;
        return run.base;
      }

      ++m_stats.resized;

      if (count < run.count) {
        // Shrink in place; the tail becomes free after this walk.
        insertFree(m_freedThisWalk, run.base + count, run.count - count);
        m_liveSlots -= run.count - count;
        run.count = count;
        run.epoch = m_epoch;
        return run.base;
      }

      // Grow in place when the slots right after the run are free, or the run
      // ends at the high-water mark.
      const uint32_t need = count - run.count;
      const uint32_t tail = run.base + run.count;
      auto fit = std::lower_bound(m_free.begin(), m_free.end(), std::make_pair(tail, 0u));
      if (fit != m_free.end() && fit->first == tail && fit->second >= need) {
        if (fit->second == need) {
          m_free.erase(fit);
        } else {
          fit->first += need;
          fit->second -= need;
        }
        m_liveSlots += need;
        run.count = count;
        run.epoch = m_epoch;
        return run.base;
      }
      if (tail == m_highWater && uint64_t(m_highWater) + need - 1u <= maxSlotValue) {
        m_highWater += need;
        m_liveSlots += need;
        run.count = count;
        run.epoch = m_epoch;
        return run.base;
      }

      // Move.
      insertFree(m_freedThisWalk, run.base, run.count);
      m_liveSlots -= run.count;
      const uint32_t base = allocRange(count, maxSlotValue);
      if (base == kNoSlot) {
        m_runs.erase(it);
        ++m_stats.overflow;
        return kNoSlot;
      }
      run.base = base;
      run.count = count;
      run.epoch = m_epoch;
      m_liveSlots += count;
      ++m_stats.moved;
      return base;
    }

    const uint32_t base = allocRange(count, maxSlotValue);
    if (base == kNoSlot) {
      ++m_stats.overflow;
      return kNoSlot;
    }
    Run run;
    run.base = base;
    run.count = count;
    run.epoch = m_epoch;
    m_runs.emplace(k, run);
    m_liveSlots += count;
    ++m_stats.created;
    return base;
  }

  void SurfaceSlotTable::endFrame() {
    for (auto it = m_runs.begin(); it != m_runs.end(); ) {
      if (it->second.epoch != m_epoch) {
        insertFree(m_freedThisWalk, it->second.base, it->second.count);
        m_liveSlots -= it->second.count;
        ++m_stats.freed;
        it = m_runs.erase(it);
      } else {
        ++it;
      }
    }

    for (const auto& f : m_freedThisWalk) {
      insertFree(m_free, f.first, f.second);
    }
    m_freedThisWalk.clear();

    while (!m_free.empty() && m_free.back().first + m_free.back().second == m_highWater) {
      m_highWater = m_free.back().first;
      m_free.pop_back();
    }
  }

  void SurfaceSlotTable::clear() {
    m_runs.clear();
    m_free.clear();
    m_freedThisWalk.clear();
    m_highWater = 0u;
    m_liveSlots = 0u;
  }

  // ==========================================================================
  // DeltaUploadTable
  // ==========================================================================

  DeltaUploadTable::DeltaUploadTable(uint32_t elementSize, const char* name)
    : m_elementSize(elementSize)
    , m_name(name) {
    m_scratch.resize(elementSize);
  }

  void DeltaUploadTable::beginFrame(uint32_t elementCount, bool bufferReplaced) {
    m_count = elementCount;
    const size_t bytes = size_t(elementCount) * m_elementSize;
    if (m_mirror.size() < bytes) {
      m_mirror.resize(bytes);
    }
    if (m_valid.size() < elementCount) {
      m_valid.resize(elementCount, uint8_t(0));
    }
    if (bufferReplaced) {
      std::fill(m_valid.begin(), m_valid.end(), uint8_t(0));
      ++m_stats.replaced;
    }
    m_dirty.clear();
    ++m_stats.frames;
    m_stats.fullBytes += bytes;
  }

  void DeltaUploadTable::commit(uint32_t index, bool gpuRewritesAfter) {
    assert(index < m_count);
    uint8_t* dst = m_mirror.data() + size_t(index) * m_elementSize;
    ++m_stats.elements;
    if (!m_valid[index] || std::memcmp(dst, m_scratch.data(), m_elementSize) != 0) {
      std::memcpy(dst, m_scratch.data(), m_elementSize);
      m_dirty.push_back(index);
    }
    m_valid[index] = gpuRewritesAfter ? uint8_t(0) : uint8_t(1);
  }

  void DeltaUploadTable::invalidateAll() {
    std::fill(m_valid.begin(), m_valid.end(), uint8_t(0));
  }

  void DeltaUploadTable::upload(DxvkContext* ctx, const Rc<DxvkBuffer>& buffer) {
    if (m_dirty.empty() || buffer == nullptr) {
      return;
    }
    if (!std::is_sorted(m_dirty.begin(), m_dirty.end())) {
      std::sort(m_dirty.begin(), m_dirty.end());
    }

    // COALESCE. Adjacent dirty elements merge. A short gap merges too, but
    // ONLY across elements the mirror vouches for: re-sending those bytes is a
    // no-op for the device buffer, whereas re-sending a GPU-owned element
    // would overwrite what the culling shader wrote with the mirror's stale
    // copy. 8 elements is 2 KB of surfaces -- cheaper to send than a region.
    constexpr uint32_t kMaxCleanGap = 8u;
    const VkDeviceSize es = m_elementSize;

    m_regions.clear();
    size_t i = 0;
    while (i < m_dirty.size()) {
      const uint32_t start = m_dirty[i];
      uint32_t end = start + 1u;
      size_t j = i + 1;
      while (j < m_dirty.size()) {
        const uint32_t next = m_dirty[j];
        if (next < end) {
          ++j;
          continue;
        }
        if (next == end) {
          end = next + 1u;
          ++j;
          continue;
        }
        if (next - end > kMaxCleanGap) {
          break;
        }
        bool clean = true;
        for (uint32_t g = end; g < next; ++g) {
          if (!m_valid[g]) {
            clean = false;
            break;
          }
        }
        if (!clean) {
          break;
        }
        end = next + 1u;
        ++j;
      }

      VkBufferCopy region;
      region.srcOffset = VkDeviceSize(start) * es;   // into the mirror
      region.dstOffset = VkDeviceSize(start) * es;
      region.size      = VkDeviceSize(end - start) * es;
      m_regions.push_back(region);
      m_stats.uploadBytes += region.size;
      i = j;
    }

    m_stats.changed += m_dirty.size();
    m_stats.regions += m_regions.size();

    ctx->writeToBufferRegions(buffer, m_mirror.data(), m_regions.data(), uint32_t(m_regions.size()));
  }

  void DeltaUploadTable::scheduleVerify(DxvkContext* ctx, DxvkDevice* device,
                                        const Rc<DxvkBuffer>& buffer, uint32_t frameId) {
    if (m_count == 0u || buffer == nullptr) {
      return;
    }
    VerifySlot* slot = nullptr;
    for (VerifySlot& s : m_verify) {
      if (!s.pending) {
        slot = &s;
        break;
      }
    }
    if (slot == nullptr) {
      return;  // both readbacks still in flight; skip rather than stall
    }

    const VkDeviceSize bytes = VkDeviceSize(m_count) * m_elementSize;
    if (bytes > buffer->info().size) {
      return;
    }
    if (slot->staging == nullptr || slot->staging->info().size < bytes) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = bytes;
      // CACHED: the harvest reads every byte on the CPU, and an uncached
      // host mapping reads far below RAM bandwidth.
      slot->staging = device->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "GpuScene verify readback");
    }

    slot->mirror.assign(m_mirror.begin(), m_mirror.begin() + size_t(bytes));
    slot->valid.assign(m_valid.begin(), m_valid.begin() + m_count);
    slot->count = m_count;
    slot->frame = frameId;
    slot->pending = true;

    ctx->copyBuffer(slot->staging, 0, buffer, 0, bytes);
  }

  void DeltaUploadTable::harvestVerify(uint32_t frameId) {
    for (VerifySlot& slot : m_verify) {
      if (!slot.pending || slot.staging == nullptr || slot.staging->isInUse()) {
        continue;
      }
      slot.pending = false;

      const uint8_t* gpu = reinterpret_cast<const uint8_t*>(slot.staging->mapPtr(0));
      if (gpu == nullptr) {
        continue;
      }

      uint32_t checked = 0u, bad = 0u, firstBad = ~0u, lastBad = 0u;
      for (uint32_t e = 0; e < slot.count; ++e) {
        if (!slot.valid[e]) {
          continue;
        }
        ++checked;
        const size_t off = size_t(e) * m_elementSize;
        if (std::memcmp(gpu + off, slot.mirror.data() + off, m_elementSize) != 0) {
          if (bad == 0u) {
            firstBad = e;
          }
          lastBad = e;
          ++bad;
        }
      }

      ++m_stats.verifyReadbacks;
      m_stats.verifyElements += checked;
      if (bad != 0u) {
        m_stats.verifyFail += bad;
        m_stats.verifyFailFrame = slot.frame;
        // Unthrottled and ungated: a non-zero count is the defect, and it must
        // not be hidden behind a stats option on the frame it first appears.
        Logger::warn(str::format(
          "[GpuScene] VERIFY-FAIL table=", m_name,
          " f=", slot.frame, " harvested=", frameId,
          " bad=", bad, "/", checked,
          " firstBad=", firstBad, " lastBad=", lastBad,
          " elements=", slot.count,
          "  <- the device buffer does NOT hold what the delta mirror says it holds;"
          " an upload was skipped that should not have been"));
      }
    }
  }

  void DeltaUploadTable::resetStats() {
    const uint32_t readbacks = m_stats.verifyReadbacks;
    const uint64_t velements = m_stats.verifyElements;
    const uint32_t fail = m_stats.verifyFail;
    const uint32_t failFrame = m_stats.verifyFailFrame;
    m_stats = Stats();
    m_stats.verifyReadbacks = readbacks;
    m_stats.verifyElements = velements;
    m_stats.verifyFail = fail;
    m_stats.verifyFailFrame = failFrame;
  }

}
