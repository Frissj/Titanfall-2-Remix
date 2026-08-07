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
#include <set>
#include <vector>
#include "../../test_utils.h"
#include "../../../src/util/util_spatial_map.h"

namespace dxvk {
  // Note: Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_spatial_map.log");
}

namespace dxvk {
  class TestApp {
  public:
    std::string ToString(const std::set<int>& input) {
      std::ostringstream result;
      std::copy(input.begin(), input.end(), std::ostream_iterator<int>(result, ", "));
      return result.str();
    }
    std::string ToString(const Vector3& pos) {
      return str::format("{", pos.x, ", ", pos.y, ", ", pos.z, "}");
    }

    void testPoint(const SpatialMap<int>& map, const Vector3& pos, int expectedResult) {
      float nearestDistSqr = FLT_MAX;
      const int* result = map.getNearestData(pos, 1.f, nearestDistSqr, [](const int* unused) {return true; });
      if (*result != expectedResult) {
        throw DxvkError(str::format("incorrect result: for pos ", ToString(pos), " expected [", expectedResult, "] but got [", *result, "]."));
      }
    }

    void expect(bool condition, const std::string& what) {
      if (!condition) {
        throw DxvkError(str::format("failed: ", what));
      }
    }

    struct TestData {
      Vector3 pos;
      int data;
      Matrix4 transform;
      TestData(Vector3 pos, int data) : pos(pos), data(data) {
        transform = translationMatrix(pos);
      }
    };

    // NV-DXVK [perf] handoff v7 sec 4a/4b: the tests above only ever touch
    // m_cells (getNearestData). Nothing covered m_cache -- the exact-hash
    // lookup that answers 97.6% of real queries, and the half this change
    // replaced. A broken exact stage does not crash: it silently misses, dedup
    // fails, and the symptom is instance churn several layers away.
    void runExactLookupTests() {
      SpatialMap<int> map(2.0f);

      const int kCount = 400;
      std::vector<TestData> data;
      data.reserve(kCount);
      for (int i = 0; i < kCount; ++i) {
        data.emplace_back(Vector3(float(i), float(i * 2), float(-i)), i);
      }

      // Insert, and check every key is retrievable by its transform.
      std::vector<XXH64_hash_t> keys(kCount, 0);
      for (int i = 0; i < kCount; ++i) {
        keys[i] = map.insert(data[i].pos, data[i].transform, &data[i].data);
      }
      expect(map.size() == size_t(kCount), "size after inserts");
      for (int i = 0; i < kCount; ++i) {
        const int* found = map.getDataAtTransform(data[i].transform);
        expect(found != nullptr && *found == i, str::format("exact lookup of ", i));
      }

      // The out-param must be the key the entry is actually filed under, and
      // must be reusable as move()'s precomputed hash.
      for (int i = 0; i < kCount; ++i) {
        XXH64_hash_t queryHash = 0;
        const int* found = map.getDataAtTransform(data[i].transform, 0, &queryHash);
        expect(found != nullptr, str::format("hash out-param lookup of ", i));
        expect(queryHash == keys[i], str::format("out-param hash matches insert key for ", i));
        // Same transform: move must be a no-op and return the same key, whether
        // or not it is told the hash.
        const XXH64_hash_t movedWith =
          map.move(keys[i], data[i].pos, data[i].transform, &data[i].data, 0, queryHash);
        const XXH64_hash_t movedWithout =
          map.move(keys[i], data[i].pos, data[i].transform, &data[i].data, 0, 0);
        expect(movedWith == keys[i], str::format("precomputed move is a no-op for ", i));
        expect(movedWith == movedWithout, str::format("precomputed move agrees with hashed move for ", i));
      }
      expect(map.size() == size_t(kCount), "no-op moves did not change the map");

      // computeKey must agree with the key insert() actually filed under, and
      // with the key move() derives internally. RtInstance::onTransformChanged
      // calls it to decide whether to bother computing a centroid at all, so a
      // disagreement here means moved instances stop being re-filed -- silently.
      for (int i = 0; i < kCount; ++i) {
        using Map = SpatialMap<int>;
        expect(Map::computeKey(data[i].transform, 0, 0) == keys[i],
               str::format("computeKey matches filed key for ", i));
        expect(Map::computeKey(data[i].transform, 0, keys[i]) == keys[i],
               str::format("computeKey honours a precomputed hash for ", i));
        expect(Map::computeKey(data[i].transform, 0xBEEFull, 0) == 0xBEEFull,
               "computeKey honours an override hash");
        expect(Map::computeKey(data[i].transform, 0xBEEFull, keys[i]) == 0xBEEFull,
               "override hash outranks a precomputed hash");
      }

      // THE LAZY-CENTROID CONTRACT. When computeKey says the key is unchanged,
      // the caller is entitled to pass a garbage centroid because move() must
      // not read it. Feed it a deliberately wrong one and prove nothing shifts:
      // not the size, not the keys, and not the cell grid the nearest-neighbour
      // search reads. If move() ever starts consuming centroid unconditionally,
      // this is what catches it.
      for (int i = 0; i < kCount; ++i) {
        const XXH64_hash_t key = SpatialMap<int>::computeKey(data[i].transform, 0, 0);
        expect(key == keys[i], "unchanged key precondition");
        map.move(keys[i], Vector3(88888.f), data[i].transform, &data[i].data, 0, key);
      }
      expect(map.size() == size_t(kCount), "no-op move with a bogus centroid changed the map");
      expect(map.debugCellEntryCount() == map.size(), "no-op move with a bogus centroid disturbed the cells");
      for (int i = 0; i < kCount; ++i) {
        const int* found = map.getDataAtTransform(data[i].transform);
        expect(found != nullptr && *found == i, str::format("still filed correctly after bogus-centroid move: ", i));
      }
      {
        // The bogus centroid must not have leaked into the spatial grid either.
        Vector3 nearest;
        const int* owner = nullptr;
        const float distSqr = map.debugClosestCachedDistSqr(Vector3(88888.f), nearest, &owner);
        expect(distSqr > 1.f, "a bogus centroid was written into the cache");
      }

      // An overrideHash key must NOT report a reusable matrix hash.
      {
        TestData extra(Vector3(999.f, 999.f, 999.f), 999);
        const XXH64_hash_t propKey = map.insert(extra.pos, extra.transform, &extra.data, 0x1234ull);
        expect(propKey == 0x1234ull, "override hash is used verbatim as the key");
        XXH64_hash_t queryHash = 0xDEADBEEFull;
        const int* found = map.getDataAtTransform(extra.transform, 0x1234ull, &queryHash);
        expect(found != nullptr && *found == 999, "override-keyed lookup");
        expect(queryHash == 0, "override-keyed lookup reports no reusable matrix hash");
        map.erase(propKey);
      }

      // A real move to a new transform re-files the entry and retires the old key.
      {
        const Vector3 newPos(-500.f, -500.f, -500.f);
        const Matrix4 newTransform = translationMatrix(newPos);
        map.move(keys[7], newPos, newTransform, &data[7].data);
        expect(map.getDataAtTransform(data[7].transform) == nullptr, "old key retired after move");
        const int* found = map.getDataAtTransform(newTransform);
        expect(found != nullptr && *found == 7, "entry findable at new transform");
        keys[7] = XXH64(&newTransform, sizeof(newTransform), 0);
        data[7].transform = newTransform;
        data[7].pos = newPos;
      }

      // Erase every third entry; the survivors must all still be reachable.
      // This is what breaks first if the probe-run repair after a deletion is
      // wrong -- an entry stays in the table but stops being findable.
      for (int i = 0; i < kCount; i += 3) {
        map.erase(keys[i]);
      }
      for (int i = 0; i < kCount; ++i) {
        const int* found = map.getDataAtTransform(data[i].transform);
        if (i % 3 == 0) {
          expect(found == nullptr, str::format("erased entry ", i, " is gone"));
        } else {
          expect(found != nullptr && *found == i, str::format("survivor ", i, " still reachable"));
        }
      }

      // Re-insert the erased ones; the map must accept them again and hold
      // exactly the original population.
      for (int i = 0; i < kCount; i += 3) {
        keys[i] = map.insert(data[i].pos, data[i].transform, &data[i].data);
      }
      expect(map.size() == size_t(kCount), "size after erase + re-insert");
      for (int i = 0; i < kCount; ++i) {
        const int* found = map.getDataAtTransform(data[i].transform);
        expect(found != nullptr && *found == i, str::format("reachable after churn: ", i));
      }

      // m_cells must not have drifted from m_cache through all of that.
      expect(map.debugCellEntryCount() == map.size(), "cell grid agrees with cache");
    }


    void run() {
      SpatialMap<int> map(2.0f);
      Matrix4 foo;
      TestData data[5] = {
        TestData(Vector3(-1.f, -1.f, -1.f), -1),
        TestData(Vector3(0.f, 0.f, 0.f), 0),
        TestData(Vector3(1.f, 1.f, 1.f), 1),
        TestData(Vector3(2.f, 2.f, 2.f), 2),
        TestData(Vector3(3.f, 3.f, 3.f), 3)
      };

      for (int i = 0; i < 5; ++i) {
        map.insert(data[i].pos, data[i].transform, &data[i].data);
      }

      // corner of a cell
      testPoint(map, Vector3(0.f, 0.f, 0.f), 0);
      // center of a cell
      testPoint(map, Vector3(1.f, 1.f, 1.f), 1);
      
      testPoint(map, Vector3(1.5f, 1.5f, 1.51f), 2);
      // near section of next cell
      testPoint(map, Vector3(2.5f, 2.5f, 2.51f), 3);
      // far section of next cell
      testPoint(map, Vector3(3.5f, 3.5f, 3.5f), 3);

      runExactLookupTests();

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
