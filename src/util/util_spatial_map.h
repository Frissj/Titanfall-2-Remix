/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include <unordered_map>

#include "util_matrix.h"
#include "util_vector.h"
#include "util_fast_cache.h"
#include "util_string.h"
#include "./log/log.h"

namespace dxvk {
  // A structure to allow for quickly returning data close to a specific position.
  template<class T>
  class SpatialMap {
  private:
    struct Entry {
      const T* data;
      Vector3 centroid;
      XXH64_hash_t transformHash;
      Entry() : data(nullptr), centroid(0.f), transformHash(0) { }
      Entry(const T* data, const Vector3& centroid, XXH64_hash_t transformHash) : data(data), centroid(centroid), transformHash(transformHash) { }
      Entry(const Entry& other) : data(other.data), centroid(other.centroid), transformHash(other.transformHash) { }
    };
  public:
    SpatialMap(float cellSize) : m_cellSize(cellSize) {
      if (m_cellSize <= 0) {
        ONCE(Logger::err("Invalid cell size in SpatialMap. cellSize must be greater than 0."));
        m_cellSize = 1.f;
      }
    }

    SpatialMap& operator=(SpatialMap&& other) {
      m_cellSize = other.m_cellSize;
      m_cells = std::move(other.m_cells);
      m_cache = std::move(other.m_cache);
      return *this;
    }

    // returns the data with an identical transform.
    // If overrideHash != 0, uses that as the cache key instead of XXH64
    // over the matrix bytes — enables identity-based dedup for content
    // whose per-frame matrix drift exceeds normal tolerance (e.g. sub-
    // view-reprojected mountains where the engine's input drift × scale
    // produces hundreds of units of main-world translation jitter even
    // though it's the same static prop). Caller MUST pass the same
    // overrideHash to insert/move/erase for the same prop, otherwise
    // entries leak.
    const T* getDataAtTransform(const Matrix4& transform, uint64_t overrideHash = 0) const {
      const XXH64_hash_t transformHash = (overrideHash != 0)
                                         ? static_cast<XXH64_hash_t>(overrideHash)
                                         : XXH64(&transform, sizeof(transform), 0);
      auto pair = m_cache.find(transformHash);
      if ( pair != m_cache.end()) {
        return pair->second.data;
      }
      return nullptr;
    }

    // returns the entry cosest to `centroid` that passes the `filter` and is less than `sqrt(maxDistSqr)` units from `centroid`.
    // `filter` should return true if the entry is a valid result.
    const T* getNearestData(const Vector3& centroid, float maxDistSqr, float& nearestDistSqr, std::function<bool(const T*)> filter) const {
      static const std::array kOffsets{
        Vector3i{0, 0, 0},
        Vector3i{0, 0, 1},
        Vector3i{0, 1, 0},
        Vector3i{0, 1, 1},
        Vector3i{1, 0, 0},
        Vector3i{1, 0, 1},
        Vector3i{1, 1, 0},
        Vector3i{1, 1, 1}
      };
      const Vector3 cellPosition = centroid / m_cellSize - Vector3(0.5f, 0.5f, 0.5f);
      const Vector3i floorPos(int(std::floor(cellPosition.x)), int(std::floor(cellPosition.y)), int(std::floor(cellPosition.z)));

      const T* nearestData = nullptr;
      nearestDistSqr = FLT_MAX;
      for (const Vector3i& offset : kOffsets) {
        auto cell = m_cells.find(floorPos + offset);
        if (cell == m_cells.end()) {
          continue;
        }
        for (const Entry& entry : cell->second) {
          if (!filter(entry.data)) {
            continue;
          }
          const float distSqr = lengthSqr(entry.centroid - centroid);
          if (distSqr <= maxDistSqr && distSqr < nearestDistSqr) {
              nearestDistSqr = distSqr;
            if (nearestDistSqr == 0.0f) {
              // Not going to find anything closer, so stop the iteration
              return entry.data;
            }
            nearestData = entry.data;
          }
        }
      }
      return nearestData;
    }
    
    // overrideHash semantics match getDataAtTransform: when non-zero, used
    // as the cache key in place of XXH64(matrix). Caller must thread the
    // SAME overrideHash through subsequent move/erase for this entry.
    XXH64_hash_t insert(const Vector3& centroid, const Matrix4& transform, const T* data, uint64_t overrideHash = 0) {
      XXH64_hash_t transformHash = (overrideHash != 0)
                                   ? static_cast<XXH64_hash_t>(overrideHash)
                                   : XXH64(&transform, sizeof(transform), 0);
      {
        // NV-DXVK [SpatialBump trace]: log every time we hit a collision
        // and bump the hash. With overrideHash (per-prop identity dedup),
        // this should be near-zero — collisions mean two different RtInst
        // ances tried to claim the same propId, and the second one ends
        // up at hash+N which no future lookup will ever query. That
        // orphans the bumped entry until GC. First 64 + every 1024th
        // bump fire so we get the cold-start picture without spam.
        bool bumpLogged = false;
        const XXH64_hash_t origHash = transformHash;
        while(m_cache.find(transformHash) != m_cache.end()) {
          // Note: This can happen if an instance is moved to the same position as another existing instance.
          // It can cause a single frame of NaN, but shouldn't cause any crashes.
          // TODO(REMIX-4134): Once spatial map is used on draw calls and not rtInstances, it should be safe to restore the assert() below.
          if (!bumpLogged) {
            static thread_local uint64_t sBumpProbe = 0;
            if (sBumpProbe < 64 || (sBumpProbe & 0x3FF) == 0) {
              Logger::info(str::format(
                "[SpatialBump] #", sBumpProbe,
                " origHash=0x", std::hex, origHash, std::dec,
                " overrideUsed=", (overrideHash != 0 ? 1 : 0),
                " mapSize=", m_cache.size()));
            }
            sBumpProbe += 1;
            bumpLogged = true;
          }
          // assert(false);
          transformHash++;
        }
      }
      auto [iter, success] = m_cache.emplace(std::piecewise_construct,
          std::forward_as_tuple(transformHash),
          std::forward_as_tuple(data, centroid, transformHash));
      if (!success) {
        ONCE(Logger::err("Failed to add entry in SpatialMap::insert()."));
        assert(false);
        return transformHash;
      }
      m_cells[getCellPos(centroid)].emplace_back(data, centroid, transformHash);
      return transformHash;
    }

    void erase(const XXH64_hash_t& transformHash) {
      auto pair = m_cache.find(transformHash);
      if (pair != m_cache.end()) {
        // NV-DXVK [SpatialErase]: log every erase. If a propId we expected
        // to stay alive gets erased between insertion and the next lookup,
        // it'll show up here. Rate-limited heavily (first 32 + every 4096th)
        // because erases fire frequently during normal scene churn.
        static thread_local uint64_t sEraseProbe = 0;
        if (sEraseProbe < 32 || (sEraseProbe & 0xFFF) == 0) {
          Logger::info(str::format(
            "[SpatialErase] #", sEraseProbe,
            " hash=0x", std::hex, transformHash, std::dec,
            " mapSize=", m_cache.size()));
        }
        sEraseProbe += 1;
        eraseFromCell(pair->second.centroid, transformHash);
        m_cache.erase(pair);
      } else {
        // Note: This can happen if a duplicate hash is encountered in the insert() call.
        // TODO(REMIX-4134): Once spatial map is used on draw calls and not rtInstances, it should be safe to restore the assert() below.
        ONCE(Logger::warn("Specified hash was missing in SpatialMap::erase()."));
        // assert(false);
      }
    }

    XXH64_hash_t move(const XXH64_hash_t& oldTransformHash, const Vector3& centroid, const Matrix4& newTransform, const T* data, uint64_t overrideHash = 0) {
      XXH64_hash_t transformHash = (overrideHash != 0)
                                   ? static_cast<XXH64_hash_t>(overrideHash)
                                   : XXH64(&newTransform, sizeof(newTransform), 0);

      if (oldTransformHash != transformHash) {
        // NV-DXVK [SpatialMove]: log when the erase+insert path fires. With
        // identity-based dedup the propId should be stable, so oldHash ==
        // newHash should be the common case (no-op). When this fires, the
        // instance's m_stablePropId disagreed with m_spatialCacheHash —
        // tells us identity dedup is being silently corrupted somewhere.
        // Rate-limited (first 32 + every 4096th).
        static thread_local uint64_t sMoveProbe = 0;
        if (sMoveProbe < 32 || (sMoveProbe & 0xFFF) == 0) {
          Logger::info(str::format(
            "[SpatialMove] #", sMoveProbe,
            " oldHash=0x", std::hex, oldTransformHash, std::dec,
            " newHash=0x", std::hex, transformHash, std::dec,
            " overrideUsed=", (overrideHash != 0 ? 1 : 0),
            " mapSize=", m_cache.size()));
        }
        sMoveProbe += 1;
        erase(oldTransformHash);
        insert(centroid, newTransform, data, overrideHash);
      }
      return transformHash;
    }

    void rebuild(float cellSize) {
      m_cells.clear();
      for (auto pair : m_cache) {
        m_cells[getCellPos(pair.second.centroid)].emplace_back(pair.second);
      }
    }

    // NV-DXVK DIAGNOSTIC: squared distance from `centroid` to the closest
    // CACHED entry, ignoring cell partitioning, radius and every filter.
    // Writes that entry's centroid to `outCentroid`. FLT_MAX when empty.
    //
    // Exists to separate two failure modes that look identical from the
    // outside when size() > 0 but getNearestData() returns nothing:
    //   - the entry really is far away (this returns a large distance), vs
    //   - the entry is right here but m_cells disagrees with m_cache, i.e.
    //     the cell bookkeeping is stale (this returns ~0).
    // O(size) — only call from a gated probe, never on the hot path.
    // outData receives the owning entry so the caller can identify WHO placed
    // it — the decisive question when the nearest cached entry turns out to be
    // tens of thousands of units away in a different coordinate space.
    float debugClosestCachedDistSqr(const Vector3& centroid, Vector3& outCentroid,
                                    const T** outData = nullptr) const {
      float best = FLT_MAX;
      for (const auto& kv : m_cache) {
        const float d = lengthSqr(kv.second.centroid - centroid);
        if (d < best) {
          best = d;
          outCentroid = kv.second.centroid;
          if (outData != nullptr) {
            *outData = kv.second.data;
          }
        }
      }
      return best;
    }

    // NV-DXVK DIAGNOSTIC: enumerate every live cache entry as
    // (key, centroid, data). The 2026-07-29 keepN=4 run proved the misses are
    // NOT lifetime — maps held up to 13 live entries and lookups still failed
    // — so the remaining question is whether a wanted entry is filed under a
    // key the query never forms, or sits in a cell the query never visits.
    // Pair with debugCellPosOf() to answer both from one dump.
    // O(size) — probe only, never on the hot path.
    template <typename Fn>
    void debugForEachEntry(Fn&& fn) const {
      for (const auto& kv : m_cache) {
        fn(kv.first, kv.second.centroid, kv.second.data);
      }
    }

    // The cell a position maps to — the SAME arithmetic getNearestData uses,
    // so a probe can compare an entry's cell against the query's cell rather
    // than re-deriving it (and risking a different rounding).
    Vector3i debugCellPosOf(const Vector3& position) const {
      return getCellPos(position);
    }

    float debugCellSize() const { return m_cellSize; }

    // NV-DXVK DIAGNOSTIC: total entries currently held in the cell grid.
    // Compare against size(): a mismatch is direct proof that m_cells and
    // m_cache have diverged.
    size_t debugCellEntryCount() const {
      size_t n = 0;
      for (const auto& cell : m_cells) {
        n += cell.second.size();
      }
      return n;
    }

    size_t size() const {
      return m_cache.size();
    }

  private:

    Vector3i getCellPos(const Vector3& position) const {
      const Vector3 scaledPos = position / m_cellSize;
      return Vector3i(int(std::floor(scaledPos.x)), int(std::floor(scaledPos.y)), int(std::floor(scaledPos.z))); 
    }

    void eraseFromCell(const Vector3& pos, XXH64_hash_t hash) {
      auto cellIter = m_cells.find(getCellPos(pos));
      if (cellIter == m_cells.end()) {
        ONCE(Logger::err("Specified cell was already empty in SpatialMap::erase()."));
        assert(false);
        return;
      }

      std::vector<Entry>& cell = cellIter->second;
      for (auto iter = cell.begin(); iter != cell.end(); ++iter) {
        if (iter->transformHash == hash) {
          if (cell.size() > 1) {
            // Swap & pop - faster than "erase", but doesn't preserve order, which is fine here.
            std::swap(*iter, cell.back());
            cell.pop_back();
          } else {
            m_cells.erase(cellIter);
          }
          return;
        }
      }

      Logger::err("Couldn't find matching data in SpatialMap::erase().");
    }

    float m_cellSize;
    fast_spatial_cache<std::vector<Entry>> m_cells;
    fast_unordered_cache<Entry> m_cache;
  };
}
