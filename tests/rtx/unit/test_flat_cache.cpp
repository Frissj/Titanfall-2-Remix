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

// Differential test for fast_flat_cache, the open-addressed container behind
// SpatialMap's exact-lookup path (handoff v7 sec 4b).
//
// WHY THIS EXISTS SEPARATELY FROM test_spatial_map. The failure mode that
// matters here is a broken probe-run repair after a deletion: an entry stays in
// the table but stops being findable. It does not crash and it does not corrupt
// anything -- dedup just misses, and the symptom surfaces frames later as
// instance churn, nowhere near the code that caused it.
//
// Both suites were checked against a deliberately broken eraseAt (replaced with
// a naive "mark the slot empty"), and both catch it -- so this is not covering a
// hole, it is covering the same defect closer to its source and across inputs
// SpatialMap's test cannot reach: key distributions that force long probe runs
// (consecutive ids, hash+1 neighbours), growth and rehash, moved-from state, and
// tens of thousands of randomized operations. test_spatial_map exercises one
// realistic population; this exercises the container's edges.
//
// Every operation is mirrored against std::unordered_map and compared.

#include <map>
#include <random>
#include <unordered_map>
#include <vector>

#include "../../test_utils.h"
#include "../../../src/util/util_flat_cache.h"

namespace dxvk {
  // Note: Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_flat_cache.log");
}

namespace dxvk {
  struct Val {
    uint64_t a = 0;
    double b = 0.0;
    Val() = default;
    Val(uint64_t a, double b) : a(a), b(b) { }
    bool operator==(const Val& o) const { return a == o.a && b == o.b; }
  };

  using Cache = fast_flat_cache<Val>;
  using Reference = std::unordered_map<uint64_t, Val>;

  class TestApp {
  public:
    void expect(bool condition, const std::string& what) {
      if (!condition) {
        throw DxvkError(str::format("failed: ", what));
      }
    }

    // The whole contract in one place: the cache holds exactly the reference's
    // entries, each findable under its own key with its own value, and forEach
    // enumerates each of them once and nothing else.
    void verifyAgainst(const Cache& c, const Reference& ref, const char* where) {
      expect(c.size() == ref.size(), str::format(where, ": size ", c.size(), " vs ", ref.size()));

      for (const auto& kv : ref) {
        const size_t slot = c.findSlot(kv.first);
        expect(slot != Cache::kInvalidSlot, str::format(where, ": key ", kv.first, " unreachable"));
        expect(c.valueAt(slot) == kv.second, str::format(where, ": key ", kv.first, " wrong value"));
        expect(c.keyAt(slot) == kv.first, str::format(where, ": key ", kv.first, " slot holds another key"));
      }

      std::map<uint64_t, int> seen;
      c.forEach([&](XXH64_hash_t k, const Val& v) {
        ++seen[k];
        const auto it = ref.find(k);
        expect(it != ref.end(), str::format(where, ": forEach yielded absent key ", k));
        expect(it->second == v, str::format(where, ": forEach wrong value for ", k));
      });
      expect(seen.size() == ref.size(), str::format(where, ": forEach visited ", seen.size(), " of ", ref.size()));
      for (const auto& kv : seen) {
        expect(kv.second == 1, str::format(where, ": forEach visited ", kv.first, " twice"));
      }
    }

    void testEmpty() {
      Cache c;
      expect(c.size() == 0, "empty size");
      expect(c.empty(), "empty flag");
      // Must not probe an unallocated table.
      expect(c.findSlot(0) == Cache::kInvalidSlot, "find in empty map");
      expect(c.findSlot(0xFFFFFFFFFFFFFFFFull) == Cache::kInvalidSlot, "find in empty map (high key)");
      int visits = 0;
      c.forEach([&](XXH64_hash_t, const Val&) { ++visits; });
      expect(visits == 0, "forEach on empty map");
    }

    // 0 and ~0 are ordinary keys here, not empty/tombstone sentinels. SpatialMap
    // files entries under a caller-supplied propId as well as under XXH64, so a
    // sentinel-based design would have to reserve values the caller can produce.
    void testSentinelKeys() {
      Cache c;
      expect(c.insert(0, Val(1, 1.0)), "insert key 0");
      expect(c.insert(~uint64_t(0), Val(2, 2.0)), "insert key ~0");
      expect(c.findSlot(0) != Cache::kInvalidSlot, "find key 0");
      expect(c.findSlot(~uint64_t(0)) != Cache::kInvalidSlot, "find key ~0");
      expect(!c.insert(0, Val(9, 9.0)), "duplicate insert rejected");
      expect(c.valueAt(c.findSlot(0)) == Val(1, 1.0), "duplicate insert left the value alone");
      expect(c.size() == 2, "size with sentinel-shaped keys");
      c.eraseAt(c.findSlot(0));
      expect(c.findSlot(0) == Cache::kInvalidSlot, "key 0 erased");
      expect(c.findSlot(~uint64_t(0)) != Cache::kInvalidSlot, "key ~0 survived");
    }

    // Consecutive keys are the worst case for linear probing, and strided
    // deletion through them is the case backward-shift repair exists for.
    void testConsecutiveRunDeletion() {
      for (uint64_t stride = 2; stride <= 5; ++stride) {
        for (uint64_t phase = 0; phase < stride; ++phase) {
          Cache c;
          Reference ref;
          const uint64_t base = 0x1000;
          for (uint64_t k = 0; k < 300; ++k) {
            c.insert(base + k, Val(k, 0.0));
            ref.emplace(base + k, Val(k, 0.0));
          }
          verifyAgainst(c, ref, "consecutive insert");

          for (uint64_t k = phase; k < 300; k += stride) {
            const size_t slot = c.findSlot(base + k);
            expect(slot != Cache::kInvalidSlot, "consecutive key present before erase");
            c.eraseAt(slot);
            ref.erase(base + k);
          }
          verifyAgainst(c, ref, "consecutive strided erase");

          // Re-inserting into the repaired runs must still land correctly.
          for (uint64_t k = phase; k < 300; k += stride) {
            c.insert(base + k, Val(k + 1000, 1.0));
            ref.emplace(base + k, Val(k + 1000, 1.0));
          }
          verifyAgainst(c, ref, "consecutive re-insert");
        }
      }
    }

    void testGrowShrinkChurn() {
      Cache c;
      Reference ref;
      for (int round = 0; round < 6; ++round) {
        for (uint64_t k = 0; k < 3000; ++k) {
          const uint64_t key = k * 2654435761ull + uint64_t(round);
          c.insert(key, Val(key, double(round)));
          ref.emplace(key, Val(key, double(round)));
        }
        verifyAgainst(c, ref, "grow round");

        std::vector<uint64_t> keys;
        keys.reserve(ref.size());
        for (const auto& kv : ref) {
          keys.push_back(kv.first);
        }
        // Drop 90%, keeping every tenth, so the table is left sparse and full of
        // repaired runs before the next round grows it again.
        for (size_t i = 0; i + 10 < keys.size(); i += 10) {
          for (size_t j = i; j < i + 9 && j < keys.size(); ++j) {
            const size_t slot = c.findSlot(keys[j]);
            if (slot != Cache::kInvalidSlot) {
              c.eraseAt(slot);
            }
            ref.erase(keys[j]);
          }
        }
        verifyAgainst(c, ref, "shrink round");
      }
    }

    void testMoveSemantics() {
      Cache a;
      for (uint64_t k = 0; k < 100; ++k) {
        a.insert(k * 7919, Val(k, 0.0));
      }
      Cache b(std::move(a));
      expect(b.size() == 100, "move-constructed size");
      // A moved-from cache must be empty AND safe to probe -- if the scalars
      // survived the move it would claim a capacity its vectors no longer have.
      expect(a.size() == 0, "moved-from size");
      expect(a.findSlot(7919) == Cache::kInvalidSlot, "moved-from find is safe");
      expect(a.insert(1, Val(1, 1.0)), "moved-from is reusable");

      Cache d;
      d.insert(42, Val(42, 0.0));
      d = std::move(b);
      expect(d.size() == 100, "move-assigned size");
      expect(d.findSlot(42) == Cache::kInvalidSlot, "move-assign discarded old contents");
      expect(d.findSlot(7919) != Cache::kInvalidSlot, "move-assign took the source contents");
    }

    void testClear() {
      Cache c;
      for (uint64_t k = 0; k < 500; ++k) {
        c.insert(k, Val(k, 0.0));
      }
      c.clear();
      expect(c.size() == 0, "size after clear");
      for (uint64_t k = 0; k < 500; ++k) {
        expect(c.findSlot(k) == Cache::kInvalidSlot, "key gone after clear");
      }
      expect(c.insert(7, Val(7, 0.0)), "usable after clear");
    }

    // clustered == the shapes SpatialMap actually produces: small engine prop
    // ids, and hash+1/hash+2 neighbours from insert()'s collision bump.
    void testRandomized(uint32_t seed, int iterations, uint64_t keySpace, bool clustered) {
      std::mt19937_64 rng(seed);
      Cache c;
      Reference ref;
      std::vector<uint64_t> live;

      for (int i = 0; i < iterations; ++i) {
        const int op = int(rng() % 100);
        if (op < 55 || live.empty()) {
          uint64_t key = clustered ? (rng() % keySpace) : rng();
          if ((rng() & 3) == 0 && !live.empty()) {
            key = live[rng() % live.size()] + 1;   // adjacent to a live key
          }
          const Val v(key ^ 0xABCDEFull, double(i));
          const bool inserted = c.insert(key, v);
          const bool refInserted = ref.emplace(key, v).second;
          expect(inserted == refInserted, "insert return value matches reference");
          if (inserted) {
            live.push_back(key);
          }
        } else if (op < 90) {
          const size_t idx = size_t(rng() % live.size());
          const uint64_t key = live[idx];
          const size_t slot = c.findSlot(key);
          expect(slot != Cache::kInvalidSlot, "live key findable before erase");
          c.eraseAt(slot);
          ref.erase(key);
          live[idx] = live.back();
          live.pop_back();
        } else {
          const uint64_t key = clustered ? (keySpace + (rng() % 1000)) : rng();
          if (ref.find(key) == ref.end()) {
            expect(c.findSlot(key) == Cache::kInvalidSlot, "absent key must not be found");
          }
        }

        if ((i % 5) == 0) {
          verifyAgainst(c, ref, clustered ? "randomized clustered" : "randomized wide");
        }
      }
      verifyAgainst(c, ref, "randomized final");
    }

    void run() {
      testEmpty();
      testSentinelKeys();
      testConsecutiveRunDeletion();
      testGrowShrinkChurn();
      testMoveSemantics();
      testClear();
      for (uint32_t seed = 1; seed <= 8; ++seed) {
        testRandomized(seed, 6000, 4096, true);
        testRandomized(seed + 100, 6000, 0, false);
      }
      std::cout << "All passed\n";
    }
  };
}

int main() {
  try {
    dxvk::TestApp testApp;
    testApp.run();
  }
  catch (const dxvk::DxvkError& error) {
    std::cerr << error.message() << std::endl;
    throw;
  }

  return 0;
}
