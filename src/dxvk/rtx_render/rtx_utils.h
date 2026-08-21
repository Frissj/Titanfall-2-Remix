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
#pragma once

#include <sstream>
#include <iomanip>
#include <cassert>
#include <optional>
#include <vulkan/vulkan.h>
#include <glm/gtc/packing.hpp>
#include "dxvk_buffer.h"
#include "dxvk_image.h"
#include "dxvk_sampler.h"
#include "../util/util_vector.h"
#include "../util/util_matrix.h"
#include "../util/util_quat.h"
#include "../util/util_pack.h"
#include "../util/util_fast_cache.h"
#include "dxvk_bind_mask.h"
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dxvk 
{
// 64kb is the size of a physical GPU memory page, aligning buffers to this size will eliminate redundant allocs
static constexpr size_t kBufferAlignment = 64 * 1024;
static constexpr float kPi = 3.141592653589793f;
static constexpr float kDegreesToRadians = kPi / 180.0f;
static constexpr float kRadiansToDegrees = 180.0f / kPi;
static constexpr uint32_t kMaxFramesInFlight = 4;   // ToDo: use actual swap chain image size

template<typename T>
void writeGPUHelper(unsigned char* data, std::size_t& offset, const T& value) {
  std::memcpy(data + offset, &value, sizeof(value));

  offset += sizeof(value);
}

// Note: This variant is used for writing an explicit type by cutting off the end of integers without needing to perform a static cast
template<uint32_t Bytes, typename T>
void writeGPUHelperExplicit(unsigned char* data, std::size_t& offset, const T& value) {
  static_assert(Bytes <= sizeof(T), "Explicit size must be less than or equal to the size of the original value");

  // Note: Ensure the value can fit in the requested explicit size
  // When Bytes == sizeof(T) all values of T fit, so no range check is needed (and the shift would be UB).
  if constexpr (Bytes < sizeof(T)) {
    assert(value < (static_cast<T>(1) << (Bytes * 8)));
  }

  std::memcpy(data + offset, &value, Bytes);

  offset += Bytes;
}

template<uint32_t Bytes>
void writeGPUPadding(unsigned char* data, std::size_t& offset) {
#ifndef NDEBUG
  // Note: Debug pattern for catching incorrect reads from padding regions
  std::memset(data + offset, 0xFF, Bytes);
#endif

  offset += Bytes;
}

inline const std::string hashToString(XXH64_hash_t hash) {
  //Two Hex Digits per byte
  constexpr uint8_t kNumHexits = sizeof(hash) * 2;
  std::stringstream ss;
  ss << std::uppercase << std::setfill('0') << std::setw(kNumHexits) << std::hex << hash;
  return ss.str();
}

inline const XXH64_hash_t StringToXXH64(const std::string& str, const XXH64_hash_t seed) {
  return XXH64(str.c_str(), str.size(), seed);
}

enum BufferType {
  Raster = 0,
  Raytrace
};

// Raster and Raytrace buffers are very similar, but not the same. 
// Template enforces that inequality at compile time to avoid mistakes.
template<BufferType T>
class GeometryBuffer : public DxvkBufferSlice {
 public:
   GeometryBuffer() { }

   GeometryBuffer(const DxvkBufferSlice slice, uint32_t offsetFromSlice, uint32_t stride, VkIndexType type)
    : DxvkBufferSlice(slice)
    , m_offsetFromSlice(offsetFromSlice)
    , m_stride(stride) {
    m_format.index = type;
  }

  GeometryBuffer(const DxvkBufferSlice slice, uint32_t offsetFromSlice, uint32_t stride, VkFormat vertexFormat)
    : DxvkBufferSlice(slice)
    , m_offsetFromSlice(offsetFromSlice)
    , m_stride(stride) {
    m_format.vertex = vertexFormat;
  }

  uint32_t offsetFromSlice() const { return m_offsetFromSlice;}
  uint32_t stride() const { return m_stride; }
  VkFormat vertexFormat() const {return m_format.vertex;}
  void setVertexFormat(VkFormat fmt) { m_format.vertex = fmt; }
  VkIndexType indexType() const {return m_format.index;}

  bool operator==(GeometryBuffer const& rhs) const {
    return defined() && rhs.defined() && matches(rhs)
       && m_stride == rhs.m_stride
       && m_format.vertex == rhs.m_format.vertex;
  }

  bool operator!=(GeometryBuffer const& rhs) const {
    return !(*this == rhs);
  }

  bool isPendingGpuWrite() const {
    return buffer()->isInUse(DxvkAccess::Write);
  }

  inline void* mapPtr(VkDeviceSize offset = 0) const {
    return DxvkBufferSlice::mapPtr(offset);
  }

 private:
  uint32_t m_offsetFromSlice = 0;
  uint32_t m_stride = 0;
  union Format {
    VkFormat vertex;
    VkIndexType index;
  };
  // access as m_format.vertex for vertex types, m_format.index for index types.
  Format m_format = {VkFormat(0)};
};

// Geometry buffer reference table. Maps a buffer to a bindless table slot that
// stays valid for as long as anything can still read through it.
//
// WHY THIS IS NOT A TAPE ANY MORE. The predecessor (BufferRefTable) appended to
// a vector that SceneManager cleared at the end of every frame, and only draws
// PROCESSED THIS FRAME re-appended to it. So a slot index was a position in one
// frame's tape and meant nothing in the next one. Any surface uploaded without
// being re-derived that frame -- an instance kept alive and not redrawn, 200 to
// 1700 of them per frame in Titanfall 2 -- carried an index from an older and
// possibly longer tape. Past the end of the written region, the bindless array
// is declared for kMaxBindlessResources with
// VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, so the read is undefined and the
// hardware follows whatever 64-bit value the slot happens to hold: that is a DMA
// page fault, not a wrong pixel. Inside the tape but from an earlier generation
// it is simply the wrong buffer, which is silently wrong geometry with nothing
// to catch it. Measured 2026-08-21 before this change: 1032 stale surface
// uploads across 70 frames, worst case 691 slots past the end of the tape.
//
// SLOT LIFETIME, AND WHY IT IS OBSERVED RATHER THAN ASSUMED. track() hands out a
// slot and the table keeps a reference to the buffer until the owner lets go of
// it and calls retire(). Retiring is not freeing: the slot keeps its descriptor,
// because a surface uploaded in an earlier frame can still carry the index.
// reclaim() frees a retired slot only after no surface has referenced it for
// quietFrames consecutive frames, and markReferenced() is what supplies that
// evidence -- it is called from the surface upload, which rewrites every live
// surface every frame, so an unmarked slot is genuinely unreferenced rather than
// merely old. Recycling a slot on frame age instead would hand a live wrong
// buffer to any surface that outlived the guess, which is the failure mode with
// no FAIL to catch it. Observation also keeps the rule correct under a resident
// scene, where an instance's age is deliberately not a bound on its life.
//
// A retired slot that is tracked again is revived in place. That matters for the
// ordinary case, not an exotic one: processGeometryInfo swaps the two history
// buffers on every vertex update, so position and previous-position exchange
// allocations continuously and neither should ever churn a slot.
//
// Identity is whatever KeyEqual says it is. The intended key is the buffer slice
// alone -- the underlying buffer object plus the byte range -- because that is
// all a VkDescriptorBufferInfo carries; stride and vertex format live per
// surface, so two views of one range with different strides are one slot.
//
// NOT THREAD SAFE, like the tape it replaces and like SparseUniqueCache. Every
// caller (the per-draw bind, the retire on geometry destruction, the reclaim
// sweep) runs on the dxvk-cs thread.
template<typename BufferType, class HashFn, class KeyEqual = std::equal_to<BufferType>>
struct BufferSlotTable {
  void clear() {
    m_table.clear();
    m_retired.clear();
    m_lastRefFrame.clear();
    m_freeSlots.clear();
    m_map.clear();
    m_retiredCount = 0;
    ++m_clearCount;
  }

  // Returns the slot for this buffer, allocating one if it has none. Stable: the
  // same buffer gets the same slot on every call for as long as it is tracked.
  uint32_t track(const BufferType& b, uint32_t frameId) {
    const auto it = m_map.find(b);
    if (it != m_map.end()) {
      const uint32_t idx = it->second;
      if (m_retired[idx] != 0) {
        m_retired[idx] = 0;
        --m_retiredCount;
        ++m_reviveCount;
      }
      m_lastRefFrame[idx] = frameId;
      return idx;
    }

    uint32_t idx;
    if (!m_freeSlots.empty()) {
      idx = m_freeSlots.back();
      m_freeSlots.pop_back();
      m_table[idx] = b;
      m_retired[idx] = 0;
      m_lastRefFrame[idx] = frameId;
    } else {
      idx = static_cast<uint32_t>(m_table.size());
      m_table.push_back(b);
      m_retired.push_back(0);
      m_lastRefFrame.push_back(frameId);
    }

    m_map.insert({ b, idx });
    ++m_insertCount;
    return idx;
  }

  // Fast path for an owner that already knows its slot: confirm the slot still
  // holds exactly this buffer and re-assert ownership of it. Returns false if
  // the slot has moved on, in which case the caller must track() again.
  //
  // Asking the table rather than comparing against a remembered buffer costs the
  // same and tests the invariant that actually matters, so a slot that was
  // reclaimed and handed to something else can never be mistaken for the buffer
  // that used to live in it.
  //
  // RE-ASSERTING OWNERSHIP IS THE POINT, not a side effect. One buffer can be
  // held by several owners -- the VGUI structured buffers are shared across
  // every panel draw that samples the same font atlas -- so one owner going away
  // retires a slot that another still holds. This is where that is undone.
  bool retain(uint32_t idx, const BufferType& b, uint32_t frameId) {
    if (idx >= m_table.size() || !m_table[idx].defined() || !m_table[idx].matches(b)) {
      return false;
    }
    if (m_retired[idx] != 0) {
      m_retired[idx] = 0;
      --m_retiredCount;
      ++m_reviveCount;
    }
    m_lastRefFrame[idx] = frameId;
    return true;
  }

  // The owner of this buffer has let go of it. The slot keeps its descriptor
  // until reclaim() proves nothing reads through it.
  void retire(const BufferType& b) {
    const auto it = m_map.find(b);
    if (it == m_map.end()) {
      return;
    }
    const uint32_t idx = it->second;
    if (m_retired[idx] == 0) {
      m_retired[idx] = 1;
      ++m_retiredCount;
      ++m_retireCount;
    }
  }

  // One live surface carries this slot index on frame frameId.
  //
  // Deliberately does NOT un-retire the slot, unlike retain(). A surface holding
  // a retired slot is precisely the state the sweep is waiting out: the owner has
  // let the buffer go and an instance that was not redrawn is still carrying the
  // index. Reviving on that would mean no slot with any surviving surface could
  // ever be freed.
  void markReferenced(uint32_t idx, uint32_t frameId) {
    if (idx < m_lastRefFrame.size()) {
      m_lastRefFrame[idx] = frameId;
    }
  }

  // Free every retired slot that no surface has referenced for more than
  // quietFrames frames. Returns how many were freed.
  uint32_t reclaim(uint32_t frameId, uint32_t quietFrames) {
    if (m_retiredCount == 0) {
      return 0;
    }

    uint32_t freed = 0;
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(m_table.size()); ++idx) {
      if (m_retired[idx] == 0) {
        continue;
      }
      // Written as a forward comparison rather than a subtraction so a frame id
      // that has not advanced (or has gone backwards across a device reset)
      // cannot wrap into a very large age and free a slot that is still in use.
      if (frameId <= m_lastRefFrame[idx] || (frameId - m_lastRefFrame[idx]) <= quietFrames) {
        continue;
      }

      m_map.erase(m_table[idx]);
      m_table[idx] = BufferType();
      m_retired[idx] = 0;
      --m_retiredCount;
      m_freeSlots.push_back(idx);
      ++m_freeCount;
      ++freed;
    }
    return freed;
  }

  const std::vector<BufferType>& getObjectTable() const {
    return m_table;
  }

  // Slots holding a buffer, retired ones included -- a retired slot still serves
  // a real descriptor, so it is active as far as the bindless table is concerned.
  uint32_t getActiveCount() const {
    return static_cast<uint32_t>(m_map.size());
  }

  // Table length, which is the high-water mark of concurrently held slots. This
  // is the number the bindless table is written to and the number the buffer
  // cache overflow check is against.
  uint32_t getTotalCount() const {
    return static_cast<uint32_t>(m_table.size());
  }

  uint32_t getRetiredCount() const { return m_retiredCount; }
  uint32_t getFreeSlotCount() const { return static_cast<uint32_t>(m_freeSlots.size()); }

  // Monotonic identity-churn counters, read the same way SparseUniqueCache's are:
  // the consumer diffs against its own previous sample. In a settled scene every
  // one of these is flat, and a climbing insert count means the same real buffer
  // is being given a new slot repeatedly.
  uint64_t getInsertCount() const { return m_insertCount; }
  uint64_t getRetireCount() const { return m_retireCount; }
  uint64_t getFreeCount() const { return m_freeCount; }
  uint64_t getReviveCount() const { return m_reviveCount; }
  uint64_t getClearCount() const { return m_clearCount; }

private:
  std::vector<BufferType> m_table;
  // Parallel to m_table. Separate byte and frame arrays rather than a struct so
  // the reclaim sweep walks a byte per slot in the common case where nothing is
  // retired.
  std::vector<uint8_t> m_retired;
  std::vector<uint32_t> m_lastRefFrame;
  std::vector<uint32_t> m_freeSlots;
  std::unordered_map<BufferType, uint32_t, HashFn, KeyEqual> m_map;

  uint32_t m_retiredCount = 0;
  uint64_t m_insertCount = 0;
  uint64_t m_retireCount = 0;
  uint64_t m_freeCount = 0;
  uint64_t m_reviveCount = 0;
  uint64_t m_clearCount = 0;
};

inline uint32_t setBit(uint32_t target, bool value, uint32_t oneBitMask) {
  return (target & ~oneBitMask) | (value ? oneBitMask : 0);
}

inline uint32_t setBits(uint32_t target, uint32_t value, uint32_t bitmask) {
  return (target & ~bitmask) | (value & bitmask);
}

inline uint32_t setBits(uint32_t& target, uint32_t value, uint32_t bitmask, uint32_t lshift) {
  return setBits(target, value << lshift, bitmask << lshift);
}

// Wipes the contents of a vector and releases allocated memory.
template<typename T>
void releaseVectorMemory(std::vector<T>& v) {
  std::vector<T>().swap(v);
}

} // namespace dxvk
