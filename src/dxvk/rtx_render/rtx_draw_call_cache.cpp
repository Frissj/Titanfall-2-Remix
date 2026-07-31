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
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include "../../util/util_fast_cache.h"   // lookupHash
#include <atomic>

namespace dxvk
{

// NV-DXVK [BlasLifecycle]: defined in rtx_camera_manager.cpp; gates probes
// to gameplay so we don't log thousands of load-time BLAS allocations.
namespace tf2 {
  extern std::atomic<uint32_t> g_engineHookCaptureCount;
}

namespace {
  // NV-DXVK [BucketRescue] — measures the min()/lowest() fix in the same run
  // that applies it, so no separate baseline capture is needed.
  //
  //   rescued = a draw paired with a candidate whose score was <= 0. These are
  //             EXACTLY the pairings the old positive-epsilon seed rejected, so
  //             this count IS the pre-fix bug rate. Non-zero => the bug was
  //             real and firing at this rate; zero => it never fired here and
  //             the flicker is something else.
  //   paired  = score > 0; the old code would have paired these too.
  //   missed  = no eligible candidate at all -> allocateEntry. Legitimate, and
  //             should now be a small residue instead of the common case.
  //
  // Counted only on the multi-entry scoring path — the single-entry path above
  // never used bestScore and is unaffected by the fix.
  std::atomic<uint32_t> g_brFrame   { 0xFFFFFFFFu };
  std::atomic<uint32_t> g_brRescued { 0 };
  std::atomic<uint32_t> g_brPaired  { 0 };
  std::atomic<uint32_t> g_brMissed  { 0 };

  // Emit and reset when the frame id changes. Called at the top of get(), so
  // the line describes the frame that just ENDED (its f= is that frame).
  void bucketRescueFlush(uint32_t currentFrame) {
    const uint32_t prev = g_brFrame.load(std::memory_order_relaxed);
    if (prev == currentFrame) {
      return;
    }
    if (prev != 0xFFFFFFFFu) {
      const uint32_t r = g_brRescued.exchange(0, std::memory_order_relaxed);
      const uint32_t p = g_brPaired.exchange(0, std::memory_order_relaxed);
      const uint32_t m = g_brMissed.exchange(0, std::memory_order_relaxed);
      if (r || p || m) {
        Logger::info(str::format(
          "[BucketRescue] f=", prev,
          " rescued=", r,          // would have been a spurious new BlasEntry
          " paired=", p,
          " missed=", m,
          " scoredTotal=", (r + p + m)));
      }
    } else {
      g_brRescued.store(0, std::memory_order_relaxed);
      g_brPaired.store(0, std::memory_order_relaxed);
      g_brMissed.store(0, std::memory_order_relaxed);
    }
    g_brFrame.store(currentFrame, std::memory_order_relaxed);
  }

  bool isSky(CameraType::Enum t) {
    return t == CameraType::Sky;
  }

  // NV-DXVK: a draw and a BlasEntry may only be paired if they live in the SAME
  // space. Sky-camera geometry is authored in 3D-skybox coordinates (~1000x
  // scale, tens of thousands of units from the main world), so pairing across
  // the boundary hands a draw a BlasEntry whose SpatialMap is populated with
  // instances in the other space.
  //
  // This invariant already existed inside exactMatch() but was enforced NOWHERE
  // ELSE, and the two fallback paths below could both cross it: the single-entry
  // clause pairs on materialHashesMatch alone, and the multi-entry scoring loop
  // never looked at cameraType at all. A 3D skybox typically contains a scaled
  // copy of the same props as the main world, so the twins share a geometry
  // hash, land in the same multimap bucket, and tie on every scoring term --
  // leaving the pairing to fall out of bucket order and frameLastTouched.
  //
  // Deliberately NOT a distance test. Distance would be a heuristic needing a
  // threshold, and would break legitimately distant main-world geometry (large
  // maps, fast movers, anything reprojected). Space identity is exact and
  // scale-independent: far-away main-world geometry still matches its
  // main-world BlasEntry at any distance.
  bool sameSpace(const DrawCallState& drawCall, const BlasEntry& blas) {
    return isSky(drawCall.cameraType) == isSky(blas.input.cameraType);
  }

  // NV-DXVK: a draw may only be paired with a BlasEntry produced by the SAME
  // vertex shader.
  //
  // exactMatch() compares material hash, FullGeometryHash, bone hash and isSky,
  // but never the shader. A 3D skybox contains a scaled copy of the same assets
  // as the main world, drawn by a different shader against the same mesh and the
  // same material -- so those two draws satisfied every term and were declared
  // the same object. They then shared one BlasEntry, and therefore one
  // SpatialMap, and each overwrote the other's instance position.
  //
  // Measured 2026-07-29 on the tree billboards: 10 of the 17 BlasEntries the
  // main-world tree (vs 0x29382bf838fda043) wrote to were also written by the
  // skybox copy (vs 0x29d5f7de0ba76c66), at ratios up to 949:1. The map
  // therefore almost always held the skybox instance, ~29000 units from the
  // main-world query, so findSimilarInstance missed and respawned the instance
  // every frame -- the flicker.
  //
  // Shader identity is exact: no threshold, no distance term, and unaffected by
  // how far apart the two copies are.
  bool sameShader(const DrawCallState& drawCall, const BlasEntry& blas) {
    return drawCall.getTransformData().vertexShaderHash
        == blas.input.getTransformData().vertexShaderHash;
  }

  bool exactMatch(const DrawCallState& drawCall, BlasEntry& blas) {
    if (isSky(drawCall.cameraType) != isSky(blas.input.cameraType)) {
      return false;
    }

    // See sameShader(): same mesh + same material drawn by a different shader is
    // a DIFFERENT object (main-world prop vs its 3D-skybox copy), and must not
    // share a BlasEntry or the SpatialMap inside it.
    if (!sameShader(drawCall, blas)) {
      return false;
    }

    return drawCall.getMaterialData().getHash() == blas.input.getMaterialData().getHash()
        && drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>() == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>()
        && drawCall.getSkinningState().boneHash == blas.input.getSkinningState().boneHash;
  }

  // NV-DXVK [XMatch]: report WHICH exactMatch term failed, comparing the draw
  // against the candidate entry.
  //
  // Every hash logged so far has been the DRAW's, and all of them are stable:
  // geometry byte-identical across 542 frames, material single-valued since the
  // hash-padding fix. Yet exactMatch still fails, which forces the fallback
  // paths. The asymmetry is that exactMatch compares against blas.input -- the
  // DrawCallState captured when the entry was ALLOCATED -- while the scoring
  // loop compares against blas.modifiedGeometryData. Nothing has ever logged the
  // entry side, so a stable draw hash proves nothing about the comparison.
  //
  // Gated on the probe VS list and on failure only.
  void logExactMatchFailure(const DrawCallState& drawCall, BlasEntry& blas, uint32_t frameId) {
    if (!lookupHash(RtxOptions::findSimilarProbeVsHashes(),
                    drawCall.getTransformData().vertexShaderHash)) {
      return;
    }
    const XXH64_hash_t dMat  = drawCall.getMaterialData().getHash();
    const XXH64_hash_t eMat  = blas.input.getMaterialData().getHash();
    const XXH64_hash_t dGeo  = drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>();
    const XXH64_hash_t eGeo  = blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>();
    // RaytraceGeometry carries the per-component `hashes` array only -- it has
    // no getHashForRule -- so compare the components the scoring loop uses.
    const XXH64_hash_t eModPos = blas.modifiedGeometryData.hashes[HashComponents::VertexPosition];
    const XXH64_hash_t eInpPos = blas.input.getGeometryData().hashes[HashComponents::VertexPosition];
    const XXH64_hash_t dPos    = drawCall.getGeometryData().hashes[HashComponents::VertexPosition];
    const XXH64_hash_t dBone = drawCall.getSkinningState().boneHash;
    const XXH64_hash_t eBone = blas.input.getSkinningState().boneHash;
    Logger::info(str::format(
      "[XMatch] f=", frameId,
      " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
      " skyDraw=", (isSky(drawCall.cameraType) ? 1 : 0),
      " skyBlas=", (isSky(blas.input.cameraType) ? 1 : 0),
      " matOK=",  (dMat == eMat ? 1 : 0),
      " geoOK=",  (dGeo == eGeo ? 1 : 0),
      " boneOK=", (dBone == eBone ? 1 : 0),
      " posInpVsMod=", (eInpPos == eModPos ? 1 : 0),
      " posDrawVsInp=", (dPos == eInpPos ? 1 : 0),
      " dMat=0x",  std::hex, static_cast<uint64_t>(dMat),
      " eMat=0x",  static_cast<uint64_t>(eMat),
      " dGeo=0x",  static_cast<uint64_t>(dGeo),
      " eGeo=0x",  static_cast<uint64_t>(eGeo),
      " dPos=0x",  static_cast<uint64_t>(dPos),
      " eInpPos=0x", static_cast<uint64_t>(eInpPos),
      " eModPos=0x", static_cast<uint64_t>(eModPos),
      " dBone=0x", static_cast<uint64_t>(dBone),
      " eBone=0x", static_cast<uint64_t>(eBone), std::dec,
      " entryFrameCreated=", blas.frameCreated,
      " entryFrameLastTouched=", blas.frameLastTouched));
  }
}

DrawCallCache::DrawCallCache(DxvkDevice* device) : CommonDeviceObject(device) {
  m_entries.reserve(1024);
}
DrawCallCache::~DrawCallCache() {}

DrawCallCache::CacheState DrawCallCache::get(const DrawCallState& drawCall, BlasEntry** out) {
  // NV-DXVK [BucketRescue]: flush the previous frame's tallies. Cheap — one
  // relaxed load per draw on the common path.
  bucketRescueFlush(m_device->getCurrentFrameId());

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

    // sameSpace() gates the FALLBACK only; exactMatch() already checks it.
    // Without it, materialHashesMatch alone pairs a main-world draw with a
    // sky-space BlasEntry (the twins share a material).
    if (!exactMatch(drawCall, entry)) {
      logExactMatchFailure(drawCall, entry, m_device->getCurrentFrameId());
    }

    if (exactMatch(drawCall, entry)
        || sameSpace(drawCall, entry) && sameShader(drawCall, entry)
           && !updatedThisFrame && (vertexDataMatches && boneHashesMatch || materialHashesMatch)) {
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

  // NV-DXVK [BucketRescue] ROOT-CAUSE FIX 2026-07-31.
  //
  // WAS: std::numeric_limits<float>::min().
  //
  // For a floating-point type min() is the smallest POSITIVE NORMAL value
  // (~1.18e-38), not the most negative one — that is lowest(). Seeding the
  // best-so-far with a positive epsilon turns the acceptance test
  // `score > bestScore` into "score must be positive", so the ranking
  // function silently doubled as a REJECTION THRESHOLD AT ZERO.
  //
  // score = up to 3000 in hash bonuses (position+bone, texcoord, material)
  //         MINUS lengthSqr(centroid distance)
  //
  // sqrt(3000) ~= 54.8, so a candidate matching on ALL THREE hashes — i.e.
  // provably the same mesh — was rejected as soon as it sat more than ~55
  // WORLD UNITS from wherever the entry's FIRST instance happened to stand.
  // Every duplicate of a shared mesh in a real map is further away than that.
  //
  // The rejected draw fell through to allocateEntry(), got a brand-new
  // BlasEntry with an EMPTY SpatialMap, so findSimilarInstance had nothing to
  // match against and respawned the instance. Destroy + recreate, every few
  // frames, on every duplicated prop in the scene, with the camera still.
  //
  // It is also self-sustaining: the spurious allocation makes the bucket
  // multi-entry, which routes that mesh's future draws into THIS scoring path
  // rather than the single-entry path above, so the failure feeds itself.
  //
  // Measured before the fix (2026-07-31, 3896 frames): 297 of 329 distinct
  // BlasEntry ADDRESSES were recycled across different geometries, i.e. the
  // entry cache was churning continuously; 96.2% of instance reaps had the
  // entry's own draw arrive on the PREVIOUS frame and not the current one
  // ([ReapJoin] pctDrew 0.62%).
  //
  // lowest() restores the intended meaning: the continue-guards above
  // (sameSpace / sameShader / already-touched-this-frame) decide ELIGIBILITY,
  // and the score only RANKS the eligible candidates. Distance still ranks —
  // the nearest instance of a shared mesh is still preferred — it just no
  // longer vetoes an otherwise-perfect match. If no eligible candidate exists,
  // *out stays null and we allocate, which is correct and unchanged.
  float bestScore = std::numeric_limits<float>::lowest();
  Matrix4 newTransform = drawCall.getTransformData().objectToWorld;
  const Vector3 newWorldPosition = drawCall.getGeometryData().boundingBox.getTransformedCentroid(newTransform);

  for (auto bucketIter = range.first; bucketIter != range.second; bucketIter++) {
    BlasEntry& blas  = bucketIter->second;
    if (exactMatch(drawCall, blas)) {
      *out = &blas;
      return CacheState::kExisted;
    }
    logExactMatchFailure(drawCall, blas, m_device->getCurrentFrameId());
    if (blas.frameLastTouched == m_device->getCurrentFrameId()) {
      continue;
    }
    // Never score a candidate from the other space. exactMatch() above rejects
    // the crossing, but this scoring fallback did not, so a main-world draw
    // whose exact match failed could be handed its 3D-skybox twin -- which is
    // how a main-world query ended up reading a SpatialMap containing only
    // sky-space instances ~29000 units away, missing, and respawning an
    // instance every frame. Skipping is correct rather than merely penalising:
    // if no same-space candidate exists, allocateEntry() below makes one, which
    // is what should happen for geometry appearing in a space for the first time.
    if (!sameSpace(drawCall, blas)) {
      continue;
    }
    // Same reasoning as sameSpace: never score a candidate produced by a
    // different shader. This is the path that actually paired the main-world
    // tree with its 3D-skybox copy -- all three scoring bonuses tie, because the
    // mesh and material really are identical.
    if (!sameShader(drawCall, blas)) {
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
    // NV-DXVK [BucketMiss]: no candidate in the bucket was accepted, so this
    // draw gets a brand-new BlasEntry with an EMPTY SpatialMap -- which forces
    // findSimilarInstance to miss and respawn the instance. bestScore is logged
    // because acceptance is `score > bestScore` seeded with
    // numeric_limits<float>::min(), the smallest POSITIVE float rather than
    // lowest(): a candidate is only ever taken when its score is positive, and
    // score is (up to 3000 in bonuses) minus SQUARED centroid distance. A
    // correct match therefore loses once it is ~55 units away. Props sharing a
    // mesh share a bucket, and the entry's stored position is wherever the
    // FIRST instance stood, so every other instance of that mesh is far enough
    // to score negative.
    //
    // bucketN vs examined tells whether the bucket even held a same-space,
    // not-yet-touched candidate to score.
    if (lookupHash(RtxOptions::findSimilarProbeVsHashes(),
                   drawCall.getTransformData().vertexShaderHash)) {
      const size_t bucketN = m_entries.count(hash);
      Logger::info(str::format(
        "[BucketMiss] f=", m_device->getCurrentFrameId(),
        " vs=0x", std::hex,
          static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash), std::dec,
        " topoHash=0x", std::hex, static_cast<uint64_t>(hash), std::dec,
        " bucketN=", bucketN,
        " bestScore=", bestScore,
        " seededMin=", std::numeric_limits<float>::min(),
        " newPos=(", newWorldPosition.x, ",", newWorldPosition.y, ",", newWorldPosition.z, ")"));
    }
    // Failed to find similar blas, so allocate a new one
    g_brMissed.fetch_add(1, std::memory_order_relaxed);
    *out = allocateEntry(hash, drawCall);
    return CacheState::kNew;
  }

  // NV-DXVK [BucketRescue]: bestScore now holds the WINNING candidate's score.
  // score <= 0 means the old std::numeric_limits<float>::min() seed would have
  // rejected this pairing and allocated a duplicate BlasEntry instead — so
  // `rescued` is a direct, per-frame measurement of the bug's former rate.
  if (bestScore > 0.0f) {
    g_brPaired.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_brRescued.fetch_add(1, std::memory_order_relaxed);
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
