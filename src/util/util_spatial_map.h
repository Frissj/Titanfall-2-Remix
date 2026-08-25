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
#include <atomic>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <vector>

#include "util_matrix.h"
#include "util_vector.h"
#include "util_fast_cache.h"
#include "util_flat_cache.h"
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
      // Declaring the copy constructor above suppresses the implicit MOVE
      // assignment and deprecates the implicit copy assignment. The flat cache
      // below assigns entries when it back-shifts a probe run and when it
      // rehashes, so spell both out rather than lean on a deprecated implicit.
      Entry& operator=(const Entry& other) = default;
    };
    // NV-DXVK [perf] handoff v7 sec 4b: the exact-lookup cache is open-addressed
    // (see util_flat_cache.h for why). Deliberately scoped to SpatialMap: the
    // ~30 other fast_unordered_cache users are not lookup-bound and are left on
    // std::unordered_map.
    using Cache = fast_flat_cache<Entry>;
  public:
    // ================================================================
    // NV-DXVK [MapLedger] 2026-08-24b: the WRITE-SIDE record of what happened to
    // a key.
    //
    // WHY THIS EXISTS. The residency handoff's remaining failure is a stationary
    // prop whose exact lookup misses even though its transform has not changed.
    // Two read-side theories have already been refuted against it, and both were
    // refuted because the read side can only ever observe that a key is absent —
    // it cannot say whether the key was never written or was written and then
    // vacated, and those two prescribe opposite fixes. Only the writer knows.
    //
    // ARMED FROM BIRTH, AND THAT IS THE WHOLE SOUNDNESS ARGUMENT. A SpatialMap
    // starts empty, so a ledger that begins recording at construction has seen
    // every write that map has ever taken. `None` therefore means "never written
    // on this map", not "not written since logging was switched on" — which is
    // what a ledger armed by a runtime flag mid-session would actually mean, and
    // it would report `None` for every prop filed before the flag flipped. That
    // failure mode would look exactly like the finding we are hunting.
    //
    // Read `debugLedgerEvicted()` before trusting any `None`: an evicted record
    // reports as never-written and is the one false positive this can produce.
    enum class LedgerOp : uint8_t {
      None     = 0,  // no record: this key has never been written on this map
      Inserted = 1,  // an entry was filed here and nothing has removed it since
      Refiled  = 2,  // move() erased it from here and re-filed it at `otherKey`
      Erased   = 3,  // erase() removed it outright (unlink, destroy, migrate out)
    };

    // Callers that have no frame to report pass this and the record still lands;
    // the verdict line then shows kNoFrame and the reader knows the event is
    // real but unplaced in time, rather than being told it happened at frame 0.
    //
    // Declared ABOVE LedgerRec because that struct uses it as a default member
    // initializer, and a nested class's complete-class context is the nested
    // class itself — it does not extend to members of the enclosing class
    // declared later.
    static constexpr uint32_t kNoFrame = 0xFFFFFFFFu;

    struct LedgerRec {
      // Occupancy is `op != LedgerOp::None`, NOT `key != 0`. XXH64 is entitled
      // to return 0 and an overrideHash of 0 already means "no override", so a
      // zero key is a legal key here and using it as the empty marker would make
      // one real prop permanently invisible to the ledger.
      XXH64_hash_t key      = 0;
      XXH64_hash_t otherKey = 0;  // Refiled: the key the entry moved TO
      // The owner at the time of the event. NEVER DEREFERENCED — compared and
      // printed as an address only. A ledger record outlives the instance it
      // names by design (that is the point of a ledger), so dereferencing it
      // would be a use-after-free of exactly the kind the GC-walk crash in
      // getImageHash already cost this project a session to.
      const T*  data    = nullptr;
      uint32_t  frame   = kNoFrame;  // frame the event was recorded on
      uint32_t  writes  = 0;         // times this key has been written, ever
      LedgerOp  op      = LedgerOp::None;
      uint8_t   bumped  = 0;         // insert() had to probe past a collision
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
      // [MapLedger] travels with the entries it describes. Leaving it behind
      // would hand the moved-to map a ledger that has never seen a write, and
      // every verdict against it would read None — the finding we are hunting,
      // manufactured by the move.
      m_ledger = std::move(other.m_ledger);
      m_ledgerCount = other.m_ledgerCount;
      m_ledgerEvicted = other.m_ledgerEvicted;
      other.m_ledgerCount = 0;
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
    //
    // NV-DXVK [perf] handoff v7 sec 4a: outMatrixHash, when non-null, receives
    // the XXH64 this lookup just paid for, so the caller can hand it back to
    // move() instead of hashing the same 64 bytes a second time in the same
    // frame ([MapGate]: 15,447 move()s per frame against 15,488 lookups, and
    // for a static prop -- 97% of them -- it is the identical matrix).
    //
    // Writes 0 when an overrideHash was used, and that is not a detail: on that
    // path the key is the caller's propId, NOT a hash of the matrix bytes, so
    // feeding it to a matrix-keyed move() would file the entry under the wrong
    // key. 0 means "no reusable matrix hash", which makes every consumer fall
    // back to hashing.
    const T* getDataAtTransform(const Matrix4& transform, uint64_t overrideHash = 0,
                                XXH64_hash_t* outMatrixHash = nullptr) const {
      const bool useOverride = (overrideHash != 0);
      const XXH64_hash_t transformHash = useOverride
                                         ? static_cast<XXH64_hash_t>(overrideHash)
                                         : XXH64(&transform, sizeof(transform), 0);
      if (outMatrixHash != nullptr) {
        *outMatrixHash = useOverride ? 0 : transformHash;
      }
      const size_t slot = m_cache.findSlot(transformHash);
      if (slot != Cache::kInvalidSlot) {
        return m_cache.valueAt(slot).data;
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
    XXH64_hash_t insert(const Vector3& centroid, const Matrix4& transform, const T* data, uint64_t overrideHash = 0,
                        uint32_t frame = kNoFrame) {
      XXH64_hash_t transformHash = (overrideHash != 0)
                                   ? static_cast<XXH64_hash_t>(overrideHash)
                                   : XXH64(&transform, sizeof(transform), 0);
      // [MapLedger]: the key BEFORE any collision bump, because that is the key
      // a future lookup will form. An entry parked at origHash+N is unreachable
      // by every subsequent query, so the ledger must file it under the key that
      // was asked for, not the slot it settled in.
      const XXH64_hash_t ledgerKey = transformHash;
      bool ledgerBumped = false;
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
        while (m_cache.contains(transformHash)) {
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
          ledgerBumped = true;
        }
      }
      const bool success = m_cache.insert(transformHash, Entry(data, centroid, transformHash));
      if (!success) {
        ONCE(Logger::err("Failed to add entry in SpatialMap::insert()."));
        assert(false);
        return transformHash;
      }
      m_cells[getCellPos(centroid)].emplace_back(data, centroid, transformHash);
      ++m_dbgInserts;
      ledgerRecord(ledgerKey, LedgerOp::Inserted, frame, data, /*otherKey*/ 0, ledgerBumped);
      return transformHash;
    }

    void erase(const XXH64_hash_t& transformHash, uint32_t frame = kNoFrame) {
      eraseInternal(transformHash, frame, /*vacatedTo*/ 0, /*isRefile*/ false);
    }

  private:
    // The body of erase(), plus the one fact the public signature cannot carry:
    // whether this removal is a destruction or the first half of a move().
    //
    // move() is implemented as erase-then-insert, so without this split every
    // re-filing would be logged as a destruction and the ledger could not answer
    // the question it was built for — "was the key vacated, or was the entry
    // killed" — which are the two branches §5.5 of the residency handoff names.
    void eraseInternal(const XXH64_hash_t& transformHash, uint32_t frame,
                       XXH64_hash_t vacatedTo, bool isRefile) {
      const size_t slot = m_cache.findSlot(transformHash);
      if (slot != Cache::kInvalidSlot) {
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
        // Copied out before the erase: eraseAt() backward-shifts the probe run,
        // which can overwrite this slot's value in place.
        const Vector3 centroid = m_cache.valueAt(slot).centroid;
        // Same reason, one line later than it looks: the owner is read for the
        // ledger while the slot is still valid, and it is only ever compared and
        // printed as an address afterwards.
        const T* const owner = m_cache.valueAt(slot).data;
        eraseFromCell(centroid, transformHash);
        m_cache.eraseAt(slot);
        ++m_dbgErases;
        ledgerRecord(transformHash,
                     isRefile ? LedgerOp::Refiled : LedgerOp::Erased,
                     frame, owner, vacatedTo, /*bumped*/ false);
      } else {
        // Note: This can happen if a duplicate hash is encountered in the insert() call.
        // TODO(REMIX-4134): Once spatial map is used on draw calls and not rtInstances, it should be safe to restore the assert() below.
        ONCE(Logger::warn("Specified hash was missing in SpatialMap::erase()."));
        // assert(false);
      }
    }

  public:

    // The key an entry with this transform is (or would be) filed under.
    //
    // NV-DXVK [perf]: exposed because the caller now needs to know whether the
    // key changed BEFORE it decides how much work to do preparing the move --
    // see RtInstance::onTransformChanged, which skips computing the centroid
    // when it has not. Sharing one definition is the point: a caller that
    // reimplemented this rule and drifted from move() would silently stop
    // re-filing moved instances, which does not fault and does not log.
    //
    // Costs nothing when the caller already holds an overrideHash or a
    // precomputed matrix hash, which is why calling it and then letting move()
    // call it again is not a double hash.
    static XXH64_hash_t computeKey(const Matrix4& transform, uint64_t overrideHash,
                                   XXH64_hash_t precomputedMatrixHash) {
      return (overrideHash != 0)
             ? static_cast<XXH64_hash_t>(overrideHash)
             : (precomputedMatrixHash != 0)
               ? precomputedMatrixHash
               : XXH64(&transform, sizeof(transform), 0);
    }

    // NV-DXVK [perf] handoff v7 sec 4a: precomputedMatrixHash is XXH64 over
    // newTransform's bytes, already paid for by this frame's getDataAtTransform
    // lookup for the same object. 0 (the default) means "not available", which
    // is the original behaviour -- and because the value is exactly what this
    // function would have computed, the resulting key, the erase/insert decision
    // and every downstream lookup are bit-identical either way.
    //
    // The CALLER owns the precondition that the two matrices are the same bytes;
    // it is not checkable here (this function never sees the matrix the lookup
    // used). See RtInstance::onTransformChanged for how it is discharged.
    // Ignored entirely when overrideHash is set, since that path does not hash.
    //
    // `centroid` IS READ ONLY WHEN THE KEY CHANGES -- it is forwarded to the
    // re-insert below and touched nowhere else. Callers that can decide the key
    // ahead of time (via computeKey) are entitled to skip computing it and pass
    // a default; if that ever stops being true, fix those callers, because the
    // compiler will not.
    XXH64_hash_t move(const XXH64_hash_t& oldTransformHash, const Vector3& centroid, const Matrix4& newTransform, const T* data, uint64_t overrideHash = 0,
                      XXH64_hash_t precomputedMatrixHash = 0, uint32_t frame = kNoFrame) {
      const XXH64_hash_t transformHash = computeKey(newTransform, overrideHash, precomputedMatrixHash);

      ++m_dbgMoves;
      if (oldTransformHash != transformHash) {
        ++m_dbgRefiled;
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

        // NV-DXVK [ReFile] 2026-08-25: HOW FAR DID IT ACTUALLY MOVE?
        //
        // This branch is the churn. [SpatialMove] counts ~40 re-filings a frame
        // against a population [ReapJoin] reports as stable (removed=0) and
        // [FastPathOrder] reports as byte-identical on 2,607,998 of 2,607,998
        // fast-path placements. Every one of those re-filings mints a key that
        // is never queried again -- 0 reciprocal pairs in 51 sampled moves, and
        // one 5-entry map has cycled 1,622 distinct keys.
        //
        // A re-file is only legitimate when the object MOVED. The old entry's
        // centroid is still in the cache at this point, so the distance is free
        // to compute and it separates the two cases outright:
        //
        //   d == 0, or d below any physical scale (< 0.001 units)
        //       The prop did not move. The matrix BYTES changed -- one mantissa
        //       bit is enough, because the key is XXH64 over the raw bytes -- so
        //       the key changes, the entry re-files, and the next frame's exact
        //       lookup on the old key can never hit. That is a permanent dedup
        //       failure driven by float jitter, and the fix is at whatever
        //       recomposes the transform each frame, or in keying on a quantised
        //       value rather than raw bytes.
        //   d large
        //       Genuine motion. The re-file is correct, the churn is explained,
        //       and this is not where the instability lives.
        {
          struct ReFileAgg {
            std::atomic<uint32_t> frame { 0u };
            std::atomic<uint32_t> n     { 0 };
            std::atomic<uint32_t> dZero { 0 };  // bit-identical centroid
            std::atomic<uint32_t> dJit  { 0 };  // < 0.001  -- float jitter
            std::atomic<uint32_t> dSub  { 0 };  // < 1
            std::atomic<uint32_t> dReal { 0 };  // >= 1     -- claimed "real motion"
            // dReal split by magnitude. In a stationary scene this bucket should
            // hold only the viewmodel, so its SHAPE is the diagnosis: drift and
            // teleport are different bugs and a single count cannot tell them
            // apart.
            std::atomic<uint32_t> dR10  { 0 };  // 10 .. 100
            std::atomic<uint32_t> dR100 { 0 };  // 100 .. 1000
            std::atomic<uint32_t> dR1k  { 0 };  // >= 1000  -- teleport
            std::atomic<uint32_t> maxD  { 0 };  // largest, whole units
          };
          static ReFileAgg sReFile;

          const size_t oldSlot = m_cache.findSlot(oldTransformHash);
          if (oldSlot != Cache::kInvalidSlot) {
            const Vector3 oldC = m_cache.valueAt(oldSlot).centroid;
            const Vector3 dv = centroid - oldC;
            const float d = std::sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            sReFile.n.fetch_add(1, std::memory_order_relaxed);
            if (dv.x == 0.f && dv.y == 0.f && dv.z == 0.f) {
              sReFile.dZero.fetch_add(1, std::memory_order_relaxed);
            } else if (d < 0.001f) {
              sReFile.dJit.fetch_add(1, std::memory_order_relaxed);
              // The DETAIL for this bucket now lives at the caller
              // (RtInstance::onTransformChanged, [ReFileJit]), because naming
              // the object needs a shader hash and a prop id that a class
              // templated over T cannot ask for. debugCentroidOf hands the old
              // centroid out so the caller can run this same magnitude test with
              // that context attached.
            } else if (d < 1.f) {
              sReFile.dSub.fetch_add(1, std::memory_order_relaxed);
            } else {
              sReFile.dReal.fetch_add(1, std::memory_order_relaxed);
              if (d >= 1000.f) {
                sReFile.dR1k.fetch_add(1, std::memory_order_relaxed);
              } else if (d >= 100.f) {
                sReFile.dR100.fetch_add(1, std::memory_order_relaxed);
              } else if (d >= 10.f) {
                sReFile.dR10.fetch_add(1, std::memory_order_relaxed);
              }
              {
                const uint32_t dWhole = static_cast<uint32_t>(d);
                uint32_t prevMax = sReFile.maxD.load(std::memory_order_relaxed);
                while (dWhole > prevMax
                       && !sReFile.maxD.compare_exchange_weak(prevMax, dWhole,
                                                              std::memory_order_relaxed)) {
                }
              }
              // WHAT IS MOVING IN A STILL SCENE. dReal is ~130 per frame, and
              // the scene is stationary apart from the viewmodel (hands and
              // weapon), which is a handful of instances. So this bucket is the
              // anomaly, not the healthy case -- the earlier reading that "a
              // re-file means the object moved, so this is correct" assumed
              // motion that is not there.
              //
              // Position separates the two populations without needing camera
              // context here: the viewmodel sits within a couple of hundred
              // units of the player, world props do not. Distance travelled
              // says which kind of wrong it is -- a few units is drift, hundreds
              // or thousands is a teleport, and a teleport every frame on a
              // stationary prop is exactly the "specific objects, intermittent
              // bursts" symptom.
              {
                static std::atomic<uint32_t> sRealFrame { 0u };
                static std::atomic<uint32_t> sRealLines { 0 };
                uint32_t rs = sRealFrame.load(std::memory_order_relaxed);
                if (frame != kNoFrame && frame > rs
                    && sRealFrame.compare_exchange_strong(rs, frame, std::memory_order_relaxed)) {
                  sRealLines.store(0, std::memory_order_relaxed);
                }
                if (sRealLines.fetch_add(1, std::memory_order_relaxed) < 6u) {
                  Logger::info(str::format(
                    "[ReFileMove] f=", frame,
                    " d=", d,
                    " from=(", oldC.x, ",", oldC.y, ",", oldC.z, ")",
                    " to=(", centroid.x, ",", centroid.y, ",", centroid.z, ")",
                    " oldKey=0x", std::hex, static_cast<uint64_t>(oldTransformHash),
                    " newKey=0x", static_cast<uint64_t>(transformHash), std::dec));
                }
              }
            }

            // Monotonic rollover, so two frame numbers cannot alternate and
            // emit repeatedly. One line per frame.
            uint32_t seen = sReFile.frame.load(std::memory_order_relaxed);
            if (frame != kNoFrame && frame > seen
                && sReFile.frame.compare_exchange_strong(seen, frame, std::memory_order_relaxed)) {
              const uint32_t en = sReFile.n.exchange(0, std::memory_order_relaxed);
              const uint32_t e0 = sReFile.dZero.exchange(0, std::memory_order_relaxed);
              const uint32_t ej = sReFile.dJit.exchange(0, std::memory_order_relaxed);
              const uint32_t es = sReFile.dSub.exchange(0, std::memory_order_relaxed);
              const uint32_t er = sReFile.dReal.exchange(0, std::memory_order_relaxed);
              const uint32_t e10 = sReFile.dR10.exchange(0, std::memory_order_relaxed);
              const uint32_t e100 = sReFile.dR100.exchange(0, std::memory_order_relaxed);
              const uint32_t e1k = sReFile.dR1k.exchange(0, std::memory_order_relaxed);
              const uint32_t emx = sReFile.maxD.exchange(0, std::memory_order_relaxed);
              if (seen != 0u && en != 0u) {
                Logger::info(str::format(
                  "[ReFile] f=", seen, " n=", en,
                  " dZero=", e0, " dJit=", ej, " dSub=", es, " dReal=", er,
                  " dR10=", e10, " dR100=", e100, " dR1k=", e1k, " maxD=", emx));
              }
            }
          }
        }
        // [MapLedger]: the erase half is a VACATE, and it carries the key the
        // entry is about to land on. That pair — "key K was vacated at frame F,
        // by owner P, in favour of key K2" — is the whole answer §5.5 asks for
        // when a stationary prop's lookup on K misses.
        eraseInternal(oldTransformHash, frame, /*vacatedTo*/ transformHash, /*isRefile*/ true);
        insert(centroid, newTransform, data, overrideHash, frame);
      }
      return transformHash;
    }

    void rebuild(float cellSize) {
      m_cells.clear();
      m_cache.forEach([this](XXH64_hash_t, const Entry& entry) {
        m_cells[getCellPos(entry.centroid)].emplace_back(entry);
      });
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
      m_cache.forEach([&](XXH64_hash_t, const Entry& entry) {
        const float d = lengthSqr(entry.centroid - centroid);
        if (d < best) {
          best = d;
          outCentroid = entry.centroid;
          if (outData != nullptr) {
            *outData = entry.data;
          }
        }
      });
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
      m_cache.forEach([&](XXH64_hash_t key, const Entry& entry) {
        fn(key, entry.centroid, entry.data);
      });
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

    // NV-DXVK [MapChurn]: per-map lifetime counters, so a miss can be attributed
    // to the map LOSING an entry rather than to the cell scan failing to reach
    // it. [FindSim] read cacheNearestDistSqr=11068 on a miss -- the closest entry
    // in the WHOLE cache was 105 units away, with a 600 cell size and a 300
    // search radius, so the query's own instance was not in the map at all. What
    // removed it is the open question, and these answer it: `refiled` counts
    // move() calls that actually erased and re-inserted because the key changed,
    // which for a stationary prop should be zero.
    uint64_t debugInserts() const { return m_dbgInserts; }
    uint64_t debugErases()  const { return m_dbgErases; }
    uint64_t debugMoves()   const { return m_dbgMoves; }
    uint64_t debugRefiled() const { return m_dbgRefiled; }

    // NV-DXVK [MapLedger]: what the WRITER last did to this key on this map.
    //
    // Returns a record whose `op` is the verdict. Read it as:
    //
    //   None      the key was never written on this map. The writer files under
    //             a key the reader never forms, so the fix is at the key
    //             composition, not in the matcher. CHECK debugLedgerEvicted()
    //             FIRST — an evicted record also reads None, and that is the one
    //             false positive this structure can produce.
    //   Inserted  the ledger says an entry is filed here and nothing removed it,
    //             yet the caller's lookup on this key returned nothing. The
    //             cache and the ledger disagree; look at `bumped`, which means
    //             the entry was parked at key+N where no lookup can reach it.
    //   Refiled   move() took the entry off this key at `frame` and put it on
    //             `otherKey`. This is §5.5's candidate. Compare `data` against
    //             the instance the reader expected: the same pointer means the
    //             prop's own instance was re-filed under a different key while
    //             the reader kept querying the old one, a different pointer
    //             means another instance took the slot.
    //   Erased    the entry was destroyed at `frame` — unlinked, reaped, or
    //             migrated to another BlasEntry. Then the question moves to the
    //             lifetime, and `frame` says how long ago it went.
    //
    // O(1), read-only, allocates nothing. Safe to call from the worker read
    // phase: Phase2b makes every SpatialMap read-only there and applies writes
    // in the single-threaded ordered tail, so no write — and therefore no
    // ledger growth — can overlap this.
    LedgerRec debugLedgerLookup(XXH64_hash_t key) const {
      if (m_ledger.empty()) {
        return LedgerRec { };
      }
      const size_t mask = m_ledger.size() - 1;
      size_t slot = static_cast<size_t>(key) & mask;
      for (size_t probe = 0; probe <= mask; ++probe) {
        const LedgerRec& rec = m_ledger[slot];
        if (rec.op == LedgerOp::None) {
          break;              // open addressing: an empty slot ends the run
        }
        if (rec.key == key) {
          return rec;
        }
        slot = (slot + 1) & mask;
      }
      return LedgerRec { };
    }

    // Records displaced to make room. NON-ZERO INVALIDATES EVERY `None` VERDICT,
    // because an evicted key is indistinguishable from one never written. Print
    // it on the same line as any None count or the count cannot be read.
    uint64_t debugLedgerEvicted() const { return m_ledgerEvicted; }
    size_t   debugLedgerEntries() const { return m_ledgerCount; }

    // NV-DXVK [ReFileJit]: the centroid an entry is currently filed at.
    //
    // Exists so the CALLER can decide whether a re-file was real motion before
    // it happens. The magnitude test lives naturally here -- move() holds both
    // centroids -- but the identity does not: SpatialMap is templated over T and
    // cannot ask an RtInstance for its shader hash or its prop id. Handing the
    // old centroid out lets onTransformChanged run the same test with the full
    // context attached, which is what turns "something jitters" into "this
    // object jitters".
    bool debugCentroidOf(XXH64_hash_t key, Vector3& outCentroid) const {
      const size_t slot = m_cache.findSlot(key);
      if (slot == Cache::kInvalidSlot) {
        return false;
      }
      outCentroid = m_cache.valueAt(slot).centroid;
      return true;
    }

    // Closest entry that PASSES `filter`, ignoring cells and radius entirely.
    // debugClosestCachedDistSqr ignores the filter as well, so it cannot separate
    // "the instance is gone" from "the instance is present but was already
    // claimed this frame" -- and those two want opposite fixes. This walks the
    // cache directly: disagreement with getNearestData is a cell-patch defect,
    // agreement means the entry really is absent.
    float debugClosestPassingDistSqr(const Vector3& centroid,
                                     const std::function<bool(const T*)>& filter,
                                     Vector3& outCentroid) const {
      float best = FLT_MAX;
      m_cache.forEach([&](XXH64_hash_t, const Entry& entry) {
        if (!filter(entry.data)) {
          return;
        }
        const float d = lengthSqr(entry.centroid - centroid);
        if (d < best) {
          best = d;
          outCentroid = entry.centroid;
        }
      });
      return best;
    }

  private:

    // NV-DXVK [MapLedger]: file one write event under `key`, replacing whatever
    // that key last recorded. Only the LAST event per key is kept — the question
    // is "what is the state of this key now and what put it there", and a full
    // history per key would grow without bound for no extra answer.
    //
    // THREAD SAFETY. Called only from insert() and eraseInternal(), which
    // Phase2b confines to the single-threaded ordered tail
    // (InstanceManager::applyDeferredSpatialOp); every SpatialMap is read-only
    // during the sharded worker phase. That is what makes growing the table here
    // safe: a reallocation can never overlap a worker's debugLedgerLookup.
    // If that contract is ever relaxed, this is one of the places that breaks,
    // and it will break as a use-after-free rather than as a wrong number.
    //
    // COST. Bounded by real writes, not by calls: [SpatialErase] and
    // [SpatialMove] measure ~11.7 erases and ~11.3 re-filings per frame against
    // ~15,400 move() calls, because a stationary prop's move() is a no-op that
    // returns before reaching here.
    void ledgerRecord(XXH64_hash_t key, LedgerOp op, uint32_t frame,
                      const T* data, XXH64_hash_t otherKey, bool bumped) {
      if (m_ledger.empty()) {
        m_ledger.resize(kLedgerInitialSlots);
      } else if ((m_ledgerCount + 1) * 2 > m_ledger.size()) {
        ledgerGrow();
      }

      const size_t mask = m_ledger.size() - 1;
      size_t slot = static_cast<size_t>(key) & mask;
      // Oldest-by-frame victim within the probe run, so that when the table is
      // full it is the stale keys that go and the recent history — the only part
      // a verdict can act on — survives.
      size_t victim = slot;
      uint32_t victimFrame = 0xFFFFFFFFu;
      for (size_t probe = 0; probe <= mask; ++probe) {
        LedgerRec& rec = m_ledger[slot];
        if (rec.op == LedgerOp::None) {
          ++m_ledgerCount;
          rec.key = key; rec.otherKey = otherKey; rec.data = data;
          rec.frame = frame; rec.writes = 1; rec.op = op;
          rec.bumped = static_cast<uint8_t>(bumped ? 1 : 0);
          return;
        }
        if (rec.key == key) {
          const uint32_t writes = rec.writes;
          rec.otherKey = otherKey; rec.data = data;
          rec.frame = frame; rec.writes = writes + 1; rec.op = op;
          // Sticky: once a key has been bumped, every later lookup of it is
          // suspect, so a subsequent clean write must not clear the flag.
          rec.bumped = static_cast<uint8_t>(rec.bumped | (bumped ? 1 : 0));
          return;
        }
        if (rec.frame != kNoFrame && rec.frame < victimFrame) {
          victimFrame = rec.frame;
          victim = slot;
        }
        slot = (slot + 1) & mask;
      }

      // Table full and at the size cap. Displace the oldest of the run.
      ++m_ledgerEvicted;
      LedgerRec& rec = m_ledger[victim];
      rec.key = key; rec.otherKey = otherKey; rec.data = data;
      rec.frame = frame; rec.writes = 1; rec.op = op;
      rec.bumped = static_cast<uint8_t>(bumped ? 1 : 0);
    }

    void ledgerGrow() {
      if (m_ledger.size() >= kLedgerMaxSlots) {
        return;   // capped: ledgerRecord falls through to oldest-victim eviction
      }
      std::vector<LedgerRec> old;
      old.swap(m_ledger);
      m_ledger.resize(old.size() * 2);
      m_ledgerCount = 0;
      const size_t mask = m_ledger.size() - 1;
      for (const LedgerRec& src : old) {
        if (src.op == LedgerOp::None) {
          continue;
        }
        size_t slot = static_cast<size_t>(src.key) & mask;
        while (m_ledger[slot].op != LedgerOp::None) {
          slot = (slot + 1) & mask;
        }
        m_ledger[slot] = src;
        ++m_ledgerCount;
      }
    }

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

    // [MapChurn] counters. Plain, not atomic: SpatialMap is already written
    // single-threaded by contract (Phase2b makes every map read-only on workers
    // and records writes instead), so an atomic would buy nothing and cost the
    // hot insert/erase path.
    uint64_t m_dbgInserts = 0;
    uint64_t m_dbgErases  = 0;
    uint64_t m_dbgMoves   = 0;
    uint64_t m_dbgRefiled = 0;

    // [MapLedger] storage. Lazily allocated on the map's first write, so an
    // empty map costs one empty vector; power-of-two so the probe is a mask.
    //
    // The cap exists because the ledger holds keys over TIME, not just the live
    // ones — an erased key stays until something displaces it — so a busy map
    // would otherwise grow without bound over a long session. 65,536 records
    // against the largest map yet measured (1,398 live entries) leaves room for
    // roughly 47 generations of that map's keys before eviction starts, and
    // m_ledgerEvicted says plainly when it has.
    static constexpr size_t kLedgerInitialSlots = 256;
    static constexpr size_t kLedgerMaxSlots     = 1u << 16;
    std::vector<LedgerRec> m_ledger;
    size_t   m_ledgerCount   = 0;
    uint64_t m_ledgerEvicted = 0;

    float m_cellSize;
    // NV-DXVK [perf] handoff v7 sec 4b: m_cells is deliberately NOT flattened.
    // Only the ~370 lookups/frame that miss the exact stage ever reach it, so it
    // is not the cost, and it is keyed on Vector3i rather than a hash.
    fast_spatial_cache<std::vector<Entry>> m_cells;
    Cache m_cache;
  };
}
