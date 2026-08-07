/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

#include "xxHash/xxhash.h"

namespace dxvk {

  // NV-DXVK [perf] handoff v7 sec 4b: an OPEN-ADDRESSED map for already-hashed
  // 64-bit keys, as a replacement for fast_unordered_cache on lookup-dominated
  // hot paths.
  //
  // WHY THIS EXISTS. fast_unordered_cache is std::unordered_map with a
  // passthrough hasher, so the hashing is already free -- but the container is
  // NODE-BASED. Every find() is bucket-array load -> pointer chase into one
  // heap-scattered node. SpatialMap does ~15,500 of those per frame across maps
  // holding ~1,400 entries, which is a near-guaranteed cache miss per lookup and
  // was measured as the bulk of find's 245 ns/call. Flattening removes the
  // pointer chase: the probe walks a dense array of 8-byte keys (a 2048-slot
  // table is 16 KB of keys, L1-resident) and touches the value array exactly
  // once, on a hit.
  //
  // DESIGN NOTES, all of them load-bearing:
  //
  // - Slot index is FIBONACCI-HASHED, not `key & mask`. The keys are usually
  //   XXH64 output, whose low bits are fine -- but SpatialMap also files entries
  //   under a caller-supplied overrideHash (a stablePropId, i.e. a small engine
  //   handle) and under collision-bumped keys (hash+1, hash+2, ...). Masking
  //   would pack those into one consecutive run and turn linear probing into a
  //   linear scan. One imul plus a shift removes that failure mode outright.
  //
  // - Deletion is BACKWARD-SHIFT, not tombstones. Tombstones are simpler but
  //   accumulate under churn until the table degrades into a scan and needs a
  //   rehash to recover; SpatialMap erases continuously as instances are GC'd,
  //   so it would sit in exactly that regime. Backward shift keeps the table
  //   tombstone-free, which also means a probe terminates at the first empty
  //   slot with no second state to test.
  //
  // - Max load factor is 0.75, so an empty slot always exists and every probe
  //   loop terminates. Growth targets 0.5.
  //
  // T must be default-constructible and assignable: unused slots hold a
  // default-constructed T rather than raw storage, which keeps this free of
  // manual object lifetime management.
  //
  // NOT a drop-in for std::unordered_map -- there are no iterators, and lookup
  // returns an opaque slot index so a find/erase pair costs one probe instead of
  // two. Callers that want the std interface should keep using
  // fast_unordered_cache.
  template<class T>
  class fast_flat_cache {
  public:
    static constexpr size_t kInvalidSlot = ~size_t(0);

    fast_flat_cache() = default;

    fast_flat_cache(const fast_flat_cache&) = default;
    fast_flat_cache& operator=(const fast_flat_cache&) = default;

    // Explicit rather than defaulted: the defaulted version copies the scalars,
    // leaving the moved-from object claiming a capacity its (now empty) vectors
    // do not have. Any later probe on it would index out of bounds.
    fast_flat_cache(fast_flat_cache&& other) noexcept {
      *this = std::move(other);
    }

    fast_flat_cache& operator=(fast_flat_cache&& other) noexcept {
      if (this != &other) {
        m_keys = std::move(other.m_keys);
        m_full = std::move(other.m_full);
        m_values = std::move(other.m_values);
        m_capacity = other.m_capacity;
        m_mask = other.m_mask;
        m_shift = other.m_shift;
        m_size = other.m_size;
        other.reset();
      }
      return *this;
    }

    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

    // Slot holding `key`, or kInvalidSlot. The slot index stays valid until the
    // next insert() or eraseAt() on this container.
    size_t findSlot(XXH64_hash_t key) const {
      // Also covers m_capacity == 0, and short-circuits the many BlasEntries
      // whose SpatialMap is empty.
      if (m_size == 0) {
        return kInvalidSlot;
      }
      size_t slot = slotFor(key);
      while (m_full[slot] != 0) {
        if (m_keys[slot] == key) {
          return slot;
        }
        slot = (slot + 1) & m_mask;
      }
      return kInvalidSlot;
    }

    bool contains(XXH64_hash_t key) const {
      return findSlot(key) != kInvalidSlot;
    }

    T& valueAt(size_t slot) { return m_values[slot]; }
    const T& valueAt(size_t slot) const { return m_values[slot]; }
    XXH64_hash_t keyAt(size_t slot) const { return m_keys[slot]; }

    // Returns false and leaves the existing value untouched if `key` is already
    // present -- matching std::unordered_map::emplace's contract, which the
    // caller this replaced relied on for its duplicate assert.
    bool insert(XXH64_hash_t key, const T& value) {
      if ((m_size + 1) * 4 > m_capacity * 3) {
        growFor(m_size + 1);
      }
      size_t slot = slotFor(key);
      while (m_full[slot] != 0) {
        if (m_keys[slot] == key) {
          return false;
        }
        slot = (slot + 1) & m_mask;
      }
      m_keys[slot] = key;
      m_values[slot] = value;
      m_full[slot] = 1;
      ++m_size;
      return true;
    }

    // Backward-shift deletion: walk the probe run following `slot` and pull back
    // any element whose ideal slot is not cyclically inside the gap, so no probe
    // chain is ever broken and no tombstone is left behind. Terminates because
    // the load factor guarantees an empty slot exists.
    void eraseAt(size_t slot) {
      size_t i = slot;
      for (;;) {
        size_t j = i;
        for (;;) {
          j = (j + 1) & m_mask;
          if (m_full[j] == 0) {
            // Nothing left to pull back -- `i` becomes the hole.
            m_full[i] = 0;
            m_keys[i] = 0;
            m_values[i] = T();
            --m_size;
            return;
          }
          const size_t k = slotFor(m_keys[j]);
          // Does k lie cyclically in (i, j]? If so the element at j is still
          // reachable from its own probe start and must not move.
          const bool mustStay = (i <= j) ? (i < k && k <= j)
                                         : (i < k || k <= j);
          if (!mustStay) {
            break;
          }
        }
        m_keys[i] = m_keys[j];
        m_values[i] = std::move(m_values[j]);
        i = j;
      }
    }

    bool erase(XXH64_hash_t key) {
      const size_t slot = findSlot(key);
      if (slot == kInvalidSlot) {
        return false;
      }
      eraseAt(slot);
      return true;
    }

    void clear() {
      for (size_t i = 0; i < m_capacity; ++i) {
        if (m_full[i] != 0) {
          m_values[i] = T();
          m_full[i] = 0;
        }
        m_keys[i] = 0;
      }
      m_size = 0;
    }

    // Enumerates live entries as fn(key, value). Iteration order is the table's
    // slot order, which is NOT insertion order and changes on rehash -- callers
    // must not depend on it.
    template<typename Fn>
    void forEach(Fn&& fn) const {
      for (size_t i = 0; i < m_capacity; ++i) {
        if (m_full[i] != 0) {
          fn(m_keys[i], m_values[i]);
        }
      }
    }

  private:
    static constexpr size_t kMinCapacity = 16;
    // 2^64 / golden ratio. Multiply and take the HIGH bits: this spreads keys
    // that differ only in their low bits (sequential prop ids, bumped hashes)
    // across the whole table instead of into one run.
    static constexpr uint64_t kFibMultiplier = 0x9E3779B97F4A7C15ull;

    size_t slotFor(XXH64_hash_t key) const {
      return static_cast<size_t>((static_cast<uint64_t>(key) * kFibMultiplier) >> m_shift);
    }

    void reset() {
      m_keys.clear();
      m_full.clear();
      m_values.clear();
      m_capacity = 0;
      m_mask = 0;
      m_shift = 64;
      m_size = 0;
    }

    void growFor(size_t minEntries) {
      size_t newCapacity = kMinCapacity;
      // Target 0.5 load after a grow, so the next grow is a doubling away.
      while (newCapacity < minEntries * 2) {
        newCapacity <<= 1;
      }
      if (newCapacity <= m_capacity) {
        newCapacity = m_capacity * 2;
      }
      rehash(newCapacity);
    }

    void rehash(size_t newCapacity) {
      std::vector<XXH64_hash_t> oldKeys;
      std::vector<uint8_t> oldFull;
      std::vector<T> oldValues;
      oldKeys.swap(m_keys);
      oldFull.swap(m_full);
      oldValues.swap(m_values);

      m_capacity = newCapacity;
      m_mask = newCapacity - 1;
      m_shift = 64;
      for (size_t c = newCapacity; c > 1; c >>= 1) {
        --m_shift;
      }
      m_keys.assign(newCapacity, 0);
      m_full.assign(newCapacity, 0);
      m_values.assign(newCapacity, T());
      m_size = 0;

      for (size_t i = 0; i < oldFull.size(); ++i) {
        if (oldFull[i] == 0) {
          continue;
        }
        size_t slot = slotFor(oldKeys[i]);
        while (m_full[slot] != 0) {
          slot = (slot + 1) & m_mask;
        }
        m_keys[slot] = oldKeys[i];
        m_values[slot] = std::move(oldValues[i]);
        m_full[slot] = 1;
        ++m_size;
      }
    }

    std::vector<XXH64_hash_t> m_keys;    // valid only where m_full != 0
    std::vector<uint8_t> m_full;         // 0 = empty, 1 = occupied
    std::vector<T> m_values;             // default-constructed in empty slots
    size_t m_capacity = 0;               // always a power of two, or 0
    size_t m_mask = 0;                   // m_capacity - 1
    uint32_t m_shift = 64;               // 64 - log2(m_capacity)
    size_t m_size = 0;                   // live entries
  };

}
