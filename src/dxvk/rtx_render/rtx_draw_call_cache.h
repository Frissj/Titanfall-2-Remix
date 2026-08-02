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

#include <vector>
#include <limits>
#include <unordered_map>
#include <functional>

#include "../util/util_vector.h"
#include "dxvk_scoped_annotation.h"

#include "rtx_types.h"
#include "rtx_common_object.h"

namespace dxvk 
{
class DxvkDevice;

// A cache of the BlasEntries across frames.  This maintains stable BlasEntry pointers until that BlasEntry
// is erased by sceneManager's garbage collection.
class DrawCallCache : public CommonDeviceObject {
public:
  using MultimapType = std::unordered_multimap<XXH64_hash_t, BlasEntry, XXH64_hash_passthrough>;

  enum class CacheState
  {
    kNew = 0,
    kExisted = 1,
  };

  DrawCallCache(DrawCallCache const&) = delete;
  DrawCallCache& operator=(DrawCallCache const&) = delete;

  explicit DrawCallCache(DxvkDevice* device);
  ~DrawCallCache();

  CacheState get(const DrawCallState& drawCall, BlasEntry** out);

  MultimapType& getEntries() {return m_entries;}

  void clear() {
    m_entries.clear();
    m_engineClassIndex.clear();
  }

  void rebuildSpatialMaps() {
    for (auto iter = m_entries.begin(); iter != m_entries.end(); ++iter) {
      iter->second.rebuildSpatialMap();
    }
  }

  // NV-DXVK [MatBind identity]: MUST be called before an entry is erased from
  // getEntries() (scene-manager GC is the only eraser) so the engine-class
  // index never holds a dangling BlasEntry*. No-op for entries that were
  // never registered (engineClassKey == 0).
  void removeFromEngineClassIndex(BlasEntry& entry);

  // NV-DXVK [MatBind identity]: visit every entry registered under this
  // engine-class key. Used by the instance layer for cross-entry instance
  // relink — see findSimilarInstance.
  void forEachEngineClassSibling(XXH64_hash_t engineClassKey,
                                 const std::function<void(BlasEntry&)>& fn);

private:
  MultimapType m_entries;

  // NV-DXVK [MatBind identity]: secondary index of the SAME BlasEntries,
  // keyed by engine identity (matsys IMaterial* + vertex shader hash)
  // instead of content. Exists because the primary key above —
  // TopologicalHash, a hash of INDEX BYTES — is volatile for geometry the
  // game re-batches every frame (measured 2026-08-02: vs 0x2859d250 draws
  // hop topoHash buckets frame-to-frame, so the entry holding a draw's
  // live instance is alive but invisible one bucket over, and ~10 of 17
  // draws/frame paired empty entries while ~10 instances/frame were reaped
  // at keepN=1 with the population perfectly stable). The index is
  // consulted ONLY when the content-keyed search finds no eligible entry,
  // so content-stable geometry never changes behavior. Values are pointers
  // into m_entries (node-based, stable until erase); removal is wired into
  // the scene-manager GC via removeFromEngineClassIndex().
  std::unordered_multimap<XXH64_hash_t, BlasEntry*, XXH64_hash_passthrough> m_engineClassIndex;

  BlasEntry* allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall);
};

}  // namespace nvvk
