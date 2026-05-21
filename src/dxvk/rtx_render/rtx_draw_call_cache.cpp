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
#include "rtx_draw_call_cache.h"
#include "rtx_cb_types.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include <atomic>

namespace dxvk
{

// NV-DXVK [BlasLifecycle]: defined in rtx_camera_manager.cpp; gates probes
// to gameplay so we don't log thousands of load-time BLAS allocations.
namespace tf2 {
  extern std::atomic<uint32_t> g_engineHookCaptureCount;
}

namespace {
  bool exactMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    auto isSky = [](CameraType::Enum t) {
      return t == CameraType::Sky;
    };

    if (isSky(drawCall.cameraType) != isSky(blas.input.cameraType)) {
      return false;
    }

    return drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash()
        && drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>() == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
        && drawCall.getSkinningState().boneHash == blas.input.getSkinningState().boneHash;
  }
}

DrawCallCache::DrawCallCache(DxvkDevice* device) : CommonDeviceObject(device) {
  m_entries.reserve(1024);
}
DrawCallCache::~DrawCallCache() {}

DrawCallCache::CacheState DrawCallCache::get(const DrawCallState& drawCall, BlasEntry** out) {
  // First, find the right bucket:
  const XXH64_hash_t hash = drawCall.getGeometryData().getHashForRule<rules::TopologicalHash>();
  auto range = m_entries.equal_range(hash);
  if (range.first == m_entries.end()) {
    // New bucket
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  // Handle buckets with 1 entry:
  auto iter = range.first;
  iter++;
  if (iter == range.second) {
    // Only 1 element
    BlasEntry& entry = range.first->second;

    const bool updatedThisFrame = entry.frameLastTouched == m_device->getCurrentFrameId();
    const bool vertexDataMatches = entry.input.getGeometryData().getHashForRule<rules::VertexDataHash>() == drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
    const bool boneHashesMatch = entry.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    const bool materialHashesMatch = entry.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();

    if (exactMatch(drawCall, entry) || !updatedThisFrame && (vertexDataMatches && boneHashesMatch || materialHashesMatch)) {
      // Exact vertex match that is reusable for the current draw call,
      // or something that hasn't been updated this frame and is similar enough.
      // Matching the logic in the multi-element loop below.
      *out = &entry;
      return CacheState::kExisted;
    } else {
      // First frame of having two mismatching instances, and the first instance has already 
      // been paired with the existing BlasEntry.
      *out = allocateEntry(hash, drawCall);
      return CacheState::kNew;
    }
  }

  // Bucket has multiple BlasEntries

  float bestScore = std::numeric_limits<float>::min();
  Matrix4 newTransform = drawCall.getTransformData().objectToWorld;
  const Vector3 newWorldPosition = drawCall.getGeometryData().boundingBox.getTransformedCentroid(newTransform);

  for (auto bucketIter = range.first; bucketIter != range.second; bucketIter++) {
    BlasEntry& blas  = bucketIter->second;
    if (exactMatch(drawCall, blas)) {
      *out = &blas;
      return CacheState::kExisted;
    }
    if (blas.frameLastTouched == m_device->getCurrentFrameId()) {
      continue;
    }
    // TODO these heuristics could use more refinement.
    float score = 0;
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexPosition] == drawCall.getGeometryData().hashes[HashComponents::VertexPosition] &&
        blas.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash) {
      score += 1000.f;
    }
    if (blas.modifiedGeometryData.hashes[HashComponents::VertexTexcoord] == drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord]) {
      score += 1000.f;
    }
    if (blas.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash()) {
      score += 1000.f;
    }
    // TODO this is only checking the distance to the first instance that created the BlasEntry, not to
    // each instance.  It also doesn't include the portal logic from InstanceManager.
    Matrix4 oldTransform = blas.input.getTransformData().objectToWorld;
    const Vector3 worldPosition = blas.input.getGeometryData().boundingBox.getTransformedCentroid(oldTransform);
    score -= lengthSqr(newWorldPosition - worldPosition);
    if (score > bestScore) {
      bestScore = score;
      *out = &blas;
    }
  }
  if (*out == nullptr) {
    // Failed to find similar blas, so allocate a new one
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }
  return CacheState::kExisted;

}

BlasEntry* DrawCallCache::allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall) {
  auto iter = m_entries.emplace(hash, drawCall);
  BlasEntry* result = &iter->second;
  result->frameCreated = m_device->getCurrentFrameId();

  // NV-DXVK [BlasNew]: log BlasEntry creation. The sky-mountain BLASes
  // (VS_2904d2 path-13, plus VS_1baf/VS_2094 path-10) ALWAYS log — and
  // crucially they BYPASS the gameplay gate: those BLASes are allocated
  // during map LOAD and then reused every frame, so a gate that suppresses
  // load-time spam would hide exactly the allocations we need. Two
  // [BlasNew] lines with the same topoHash mean identical geometry split
  // into duplicate BLASes; comparing matHash / vtxHash / fullGeoHash
  // across those lines shows which hash component defeated reuse. All
  // other vsHashes stay gameplay-gated + rate-limited to bound spam.
  {
    const XXH64_hash_t vsHash = drawCall.getTransformData().vertexShaderHash;
    const bool isMtnVs = (vsHash == 0x2904d2163ef31a17ull)   // path-13
                      || (vsHash == 0x29146e1dd50b0314ull)   // path-10 VS_1baf
                      || (vsHash == 0x28f7ffa90d189017ull);  // path-10 VS_2094
    const bool gameplay =
      tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
    static thread_local uint32_t sBlasNewProbe = 0;
    const bool rateLimitedAllow =
      gameplay && (sBlasNewProbe < 64 || (sBlasNewProbe & 0x3FF) == 0);
    if (isMtnVs || rateLimitedAllow) {
      const XXH64_hash_t matHash = drawCall.getMaterialData().getHash();
      const XXH64_hash_t vtxHash = drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
      const XXH64_hash_t fullHash = drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>();
      // Per-component hashes: a BLAS is built from vertex POSITIONS + indices
      // only, so posHash+topoHash is the true geometry identity. uvHash is
      // logged alongside to confirm whether a duplicate-BLAS pair differs
      // ONLY in texcoords (a false split) vs genuinely in positions.
      const XXH64_hash_t posHash = drawCall.getGeometryData().hashes[HashComponents::VertexPosition];
      const XXH64_hash_t uvHash  = drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord];
      const Matrix4& o2w = drawCall.getTransformData().objectToWorld;
      Logger::info(str::format(
        "[BlasNew] #", sBlasNewProbe,
        " f=", m_device->getCurrentFrameId(),
        " topoHash=0x", std::hex, hash, std::dec,
        " vsHash=0x", std::hex, vsHash, std::dec,
        " matHash=0x", std::hex, matHash, std::dec,
        " vtxHash=0x", std::hex, vtxHash, std::dec,
        " fullGeoHash=0x", std::hex, fullHash, std::dec,
        " posHash=0x", std::hex, posHash, std::dec,
        " uvHash=0x", std::hex, uvHash, std::dec,
        " o2w.t=(", float(o2w[3][0]), ",", float(o2w[3][1]), ",", float(o2w[3][2]), ")",
        " mapSize=", m_entries.size(),
        isMtnVs ? " SKY_MOUNTAIN" : ""));
    }
    sBlasNewProbe += 1;
  }

  return result;
}

}  // namespace nvvk
