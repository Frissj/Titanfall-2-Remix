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

    const bool matMatches  = drawCall.getMaterialData().getHash()
                          == blas.input.getMaterialData().getHash();
    // geoMatches is FullGeometryHash, a hash of the vertex CONTENT, and it fails
    // on most of this game's world geometry: Titanfall batches that geometry with
    // world-space vertices which are rewritten every frame, so one unchanging
    // prop hashes differently frame to frame and the term cannot re-identify its
    // entry. [XMatch] measured geoOK=0 on 83% of comparisons, 5126 of those with
    // the shader and the material both matching. That is the entry-churn defect
    // and it is open.
    //
    // It looks like commit b1928c19 (2026-08-07, rtx_hashing.h) caused this and
    // that reverting it would cure it. Do not. Before b1928c19, precombined[] was
    // uninitialised, so on the draw path this term compared 0 against 0 and
    // silently always passed; b1928c19 zeroed the array and made the draw path
    // call precombine(), which is correct. It only started enforcing a term that
    // was already unusable here. The fix is an identity that survives per-frame
    // vertex rewrites, not a weaker geometry term.
    const bool geoMatches  = drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>()
                          == blas.input.getGeometryData().getHashForRule<rules::FullGeometryHash>();
    const bool boneMatches = drawCall.getSkinningState().boneHash
                          == blas.input.getSkinningState().boneHash;
    // The per-component position hash, on top of FullGeometryHash. Deliberately
    // redundant: it is what makes the relaxation below safe to state, because a
    // scaled or relocated COPY of a mesh cannot agree on it.
    const bool posMatches  = drawCall.getGeometryData().hashes[HashComponents::VertexPosition]
                          == blas.input.getGeometryData().hashes[HashComponents::VertexPosition];

    if (!(matMatches && geoMatches && boneMatches)) {
      return false;
    }

    // See sameShader(): same mesh + same material drawn by a different shader is
    // a DIFFERENT object (main-world prop vs its 3D-skybox copy), and must not
    // share a BlasEntry or the SpatialMap inside it.
    //
    // NARROWED, and only for the case where the shader is the ONLY thing that
    // differs. sameShader cannot tell two different objects apart from two
    // PASSES OVER ONE OBJECT, and this engine does the latter constantly --
    // [TcPair] measured 323 cases of a single index range drawn twice in one
    // frame by different shaders. When a pass is split off into its own
    // BlasEntry that entry is born with an EMPTY SpatialMap, so
    // findSimilarInstance cannot match into it, the instance respawns, and last
    // frame's instance is reaped. That is the reap storm.
    //
    // Measured with [XMatch] once shaderOK= existed to see it: 5468 comparisons
    // where the shader was the only failing term, 5264 of them the single pair
    // 0x2af9b90d63850ec3 -> 0x298e12b3d5bcd082, with material, FullGeometryHash,
    // bone hash and space all byte-identical.
    //
    // WHY THIS DOES NOT REOPEN THE TREE BUG. That case was a 3D-skybox COPY:
    // same asset, different placement and scale. It is admitted here only if it
    // agrees on FullGeometryHash AND the vertex-position component hash AND the
    // material AND the bone hash AND the space -- five exact terms with no
    // threshold. Two objects that agree on all five are not a copy of each
    // other, they are the same geometry submitted twice.
    //
    // TO REVERT: restore the early `return false` on !sameShader. The
    // [ShaderSplitMerge] line below fires on every pairing this admits, so a
    // regression names itself instead of showing up as flicker.
    if (!sameShader(drawCall, blas)) {
      if (!posMatches) {
        return false;
      }
      static std::atomic<uint32_t> sMergeLines { 0 };
      constexpr uint32_t kMaxMergeLines = 400;
      if (sMergeLines.fetch_add(1, std::memory_order_relaxed) < kMaxMergeLines) {
        Logger::info(str::format(
          "[ShaderSplitMerge] pairing across shaders:"
          " dVs=0x", std::hex,
          static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash),
          " eVs=0x",
          static_cast<uint64_t>(blas.input.getTransformData().vertexShaderHash),
          " mat=0x", static_cast<uint64_t>(drawCall.getMaterialData().getHash()),
          " geo=0x",
          static_cast<uint64_t>(drawCall.getGeometryData().getHashForRule<rules::FullGeometryHash>()),
          std::dec,
          " | material, geometry, position, bone and space all identical"));
      }
    }

    return true;
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

    // NV-DXVK [KeyStab]: the CANDIDATE identity, reported and not acted on.
    //
    // THE QUESTION THIS ANSWERS. Every term above is derived from the vertex
    // CONTENT, and this engine rewrites that content every frame, so all of them
    // churn. engineDrawKey is derived from the engine's own buffer objects
    // instead. If edkOK=1 on the population where shaderOK=1 matOK=1 geoOK=0 --
    // the 5126 comparisons that are the entry churn -- then engineDrawKey is an
    // identity that survives the rewrite, and it belongs in the geometry term.
    // If edkOK=0 there too, it does not, and it must not be wired in.
    //
    // edkOK requires dEdk != 0: zero is "no vertex buffer bound, no identity",
    // and two draws with no identity must not compare equal. That is the trap
    // the geometry term itself fell into before b1928c19, when it compared 0
    // against 0 and always passed.
    const uint64_t dEdk = drawCall.engineDrawKey;
    const uint64_t eEdk = blas.input.engineDrawKey;

    const XXH64_hash_t dBone = drawCall.getSkinningState().boneHash;
    const XXH64_hash_t eBone = blas.input.getSkinningState().boneHash;
    Logger::info(str::format(
      "[XMatch] f=", frameId,
      " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
      " skyDraw=", (isSky(drawCall.cameraType) ? 1 : 0),
      " skyBlas=", (isSky(blas.input.cameraType) ? 1 : 0),
      // NV-DXVK: THE TERM THIS PROBE WAS MISSING, and it is the one deciding.
      //
      // exactMatch tests four things -- space, shader, and the three hashes.
      // This probe reported the space check and all three hashes and never the
      // shader, because sameShader was added after it (2026-07-29, the
      // tree-versus-skybox pairing). So a rejection whose only failing term was
      // the shader showed up here as every logged field passing, which reads as
      // "no reason at all".
      //
      // Measured before this field existed: 9393 comparisons with
      // matOK=1 geoOK=1 boneOK=1, skyDraw=skyBlas=0, and every raw hash
      // byte-identical -- dMat==eMat, dGeo==eGeo, dPos==eInpPos==eModPos,
      // dBone==eBone -- yet exactMatch returned false. By elimination that is
      // sameShader, but a probe that forces its reader to reason by elimination
      // is the same defect this file already recorded once, when two VS lists
      // were compared across different hash spaces and declared disjoint.
      // Print it, and print both hashes so it can be checked rather than
      // inferred.
      " shaderOK=", (sameShader(drawCall, blas) ? 1 : 0),
      " matOK=",  (dMat == eMat ? 1 : 0),
      " geoOK=",  (dGeo == eGeo ? 1 : 0),
      " boneOK=", (dBone == eBone ? 1 : 0),
      " edkOK=", (dEdk != 0ull && dEdk == eEdk ? 1 : 0),
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
      " eBone=0x", static_cast<uint64_t>(eBone),
      " dEdk=0x", dEdk,
      " eEdk=0x", eEdk,
      // Both sides of the shader term, so shaderOK above can be checked rather
      // than trusted. Same hash space on both -- getTransformData().vertexShaderHash
      // is DxvkShader::getHash() on the draw AND on the entry, so these two are
      // directly comparable to each other and to [FindSim] vs=, but NOT to the
      // GetCurrentVsPsHashes values that [RtxTcNone] and friends print.
      " dVs=0x", static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash),
      " eVs=0x", static_cast<uint64_t>(blas.input.getTransformData().vertexShaderHash),
      // NV-DXVK [MatBind identity]: engine-truth IMaterial* on both sides.
      // engOK=1 means the engine bound the same material for the draw and
      // for the entry's allocation-time draw — the class test the matcher
      // now decides on when both are non-zero.
      " dEng=0x", static_cast<uint64_t>(drawCall.engineMaterialPtr),
      " eEng=0x", static_cast<uint64_t>(blas.input.engineMaterialPtr), std::dec,
      " engOK=", (drawCall.engineMaterialPtr != 0 && drawCall.engineMaterialPtr == blas.input.engineMaterialPtr ? 1 : 0),
      " entryFrameCreated=", blas.frameCreated,
      " entryFrameLastTouched=", blas.frameLastTouched));
  }
}

namespace {
  // NV-DXVK [MatBind identity]: key for the engine-class index — the engine's
  // IMaterial* plus the vertex shader hash. Both are session-stable for a
  // living material ([DrawName] 2026-08-02: the churn VS's two material
  // pointers recurred unchanged across the whole run), unlike every
  // content-derived hash for re-batched geometry.
  XXH64_hash_t makeEngineClassKey(uint64_t engineMaterialPtr, XXH64_hash_t vertexShaderHash) {
    const uint64_t data[2] = { engineMaterialPtr, static_cast<uint64_t>(vertexShaderHash) };
    return XXH3_64bits(data, sizeof(data));
  }
}

DrawCallCache::DrawCallCache(DxvkDevice* device) : CommonDeviceObject(device) {
  m_entries.reserve(1024);
}
DrawCallCache::~DrawCallCache() {}

// NV-DXVK [MatBind identity]: enumerate all living entries registered under
// one engine-class key. Consumed by InstanceManager::findSimilarInstance to
// search sibling entries' SpatialMaps when a re-batched draw allocated a new
// entry (topology changed => new BlasEntry by design) but the OBJECT — the
// instance — still exists on last frame's entry.
void DrawCallCache::forEachEngineClassSibling(XXH64_hash_t engineClassKey,
                                              const std::function<void(BlasEntry&)>& fn) {
  auto range = m_engineClassIndex.equal_range(engineClassKey);
  for (auto it = range.first; it != range.second; ++it) {
    fn(*it->second);
  }
}

void DrawCallCache::removeFromEngineClassIndex(BlasEntry& entry) {
  if (entry.engineClassKey == 0) {
    return;
  }
  auto range = m_engineClassIndex.equal_range(entry.engineClassKey);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second == &entry) {
      m_engineClassIndex.erase(it);
      return;
    }
  }
}

DrawCallCache::CacheState DrawCallCache::get(const DrawCallState& drawCall, BlasEntry** out) {
  // NV-DXVK [BucketRescue]: flush the previous frame's tallies. Cheap — one
  // relaxed load per draw on the common path.
  bucketRescueFlush(m_device->getCurrentFrameId());

  // First, find the right bucket:
  const XXH64_hash_t hash = drawCall.getGeometryData().getHashForRule<rules::TopologicalHash>();
  auto range = m_entries.equal_range(hash);
  if (range.first == m_entries.end()) {
    // NV-DXVK [MatBind identity]: an unknown topoHash does NOT mean an unknown
    // OBJECT — for re-batched geometry the index bytes (and so this bucket
    // key) change every frame. Entry reuse across a topology change was tried
    // here 2026-08-02 and REVERTED the same night: the whole pipeline bakes in
    // "an existing entry's topology never changes" (the KBuildBVH stomp assert
    // in processGeometryInfo, the kUpdateBVH refit path, the assert in
    // onSceneObjectUpdated), and violating it produced VK_ERROR_DEVICE_LOST on
    // gameplay start. The cross-bucket continuity lives at the INSTANCE layer
    // instead: findSimilarInstance searches engine-class sibling entries
    // (m_engineClassIndex) and relinks the live instance to the new entry.
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

    // NV-DXVK [MatBind identity]: engine-truth class test, same precedence as
    // the multi-entry loop — when both sides carry the matsys IMaterial*
    // captured at submit, IT decides similarity; derived hashes only rule
    // when engine identity is absent on either side.
    const bool haveEngineIds = drawCall.engineMaterialPtr != 0 && entry.input.engineMaterialPtr != 0;
    const bool similarIdentity = haveEngineIds
      ? (drawCall.engineMaterialPtr == entry.input.engineMaterialPtr)
      : (vertexDataMatches && boneHashesMatch || materialHashesMatch);

    if (exactMatch(drawCall, entry)
        || sameSpace(drawCall, entry) && sameShader(drawCall, entry)
           && !updatedThisFrame && similarIdentity) {
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
  // NV-DXVK [XMatch fix]: recency-major ranking — see the comment block inside
  // the loop. 0 is a safe floor: real candidates always carry a real frame id.
  uint32_t bestTouched = 0;
  Matrix4 newTransform = drawCall.getTransformData().objectToWorld;
  const Vector3 newWorldPosition = drawCall.getGeometryData().boundingBox.getTransformedCentroid(newTransform);

  // NV-DXVK [EntryReject]: per-clause rejection census for this lookup, so a
  // freshly allocated entry below can be ATTRIBUTED rather than guessed at.
  //
  // The measurement that motivated it: VS 0x2af9b90d63850ec3 burned 307 distinct
  // BlasEntry addresses over 255 frames, 153 of which existed for exactly ONE
  // frame. A one-frame entry has an empty SpatialMap, so findSimilarInstance
  // cannot match into it, the instance respawns and the previous one is reaped.
  // That is the reaping, and it originates here rather than in the matcher --
  // [FindSim] measured 1 to 2 queries against maps holding 3 to 45 instances, so
  // the matcher is not starved of candidates, it is handed the wrong map.
  uint32_t dbgSiblings = 0;
  uint32_t dbgEngReject = 0;
  uint32_t dbgEngRejectWouldHashMatch = 0;
  uint32_t dbgHashReject = 0;
  uint32_t dbgTouchedReject = 0;
  uint32_t dbgSpaceReject = 0;
  uint32_t dbgShaderReject = 0;

  for (auto bucketIter = range.first; bucketIter != range.second; bucketIter++) {
    BlasEntry& blas  = bucketIter->second;
    ++dbgSiblings;
    if (exactMatch(drawCall, blas)) {
      *out = &blas;
      return CacheState::kExisted;
    }
    logExactMatchFailure(drawCall, blas, m_device->getCurrentFrameId());
    if (blas.frameLastTouched == m_device->getCurrentFrameId()) {
      ++dbgTouchedReject;
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
      ++dbgSpaceReject;
      continue;
    }
    // Same reasoning as sameSpace: never score a candidate produced by a
    // different shader. This is the path that actually paired the main-world
    // tree with its 3D-skybox copy -- all three scoring bonuses tie, because the
    // mesh and material really are identical.
    if (!sameShader(drawCall, blas)) {
      ++dbgShaderReject;
      continue;
    }
    // NV-DXVK [XMatch fix 2026-08-02]: eligibility gate + stable assignment.
    //
    // [XMatch] finally measured WHY exactMatch fails on the churning VSes
    // (206,814 failures over 1,835 frames, vs 0x2859d250): geoOK=0 on 100% of
    // them, entirely position-driven — 27,237 distinct draw position hashes
    // and only 7 ever seen in a second frame. These draws (foliage/billboard
    // batches, identity objectToWorld, world-space vertices) rewrite their
    // vertex data EVERY frame, so FullGeometryHash is structurally incapable
    // of re-identifying an entry, and every such draw lands in this loop.
    //
    // Two defects made this loop churn instead of pair:
    //
    // 1) Bonuses were additive, never gating: a candidate matching NOTHING
    //    (different material, different content) still got scored, so a draw
    //    could take over a bucket-mate of a different material purely by
    //    being nearest ([XMatch] matOK=0 on 56% of comparisons). The
    //    single-entry path above already refuses that pairing — it requires
    //    (vertexDataMatches && boneHashesMatch || materialHashesMatch).
    //    Enforce the same eligibility here.
    //    ENGINE-TRUTH upgrade: when both the draw and the entry carry the
    //    matsys IMaterial* captured at submit ([MatBind] hook ->
    //    DrawCallState::engineMaterialPtr), that pointer DECIDES the class —
    //    equal means the engine itself bound the same material for both
    //    draws, different means they are different objects no matter what
    //    any derived hash says. The hash clauses below only rule when at
    //    least one side has no engine identity (non-matsys draw).
    const bool vertexDataMatches = blas.input.getGeometryData().getHashForRule<rules::VertexDataHash>()
                                == drawCall.getGeometryData().getHashForRule<rules::VertexDataHash>();
    const bool boneHashesMatch = blas.input.getSkinningState().boneHash == drawCall.getSkinningState().boneHash;
    const bool materialHashesMatch = blas.input.getMaterialData().getHash() == drawCall.getMaterialData().getHash();
    if (drawCall.engineMaterialPtr != 0 && blas.input.engineMaterialPtr != 0) {
      if (drawCall.engineMaterialPtr != blas.input.engineMaterialPtr) {
        // NV-DXVK [EntryReject]: this clause is the suspect, and the counter
        // beside it is what convicts or clears it.
        //
        // The comment above treats engineMaterialPtr as engine truth about
        // OBJECT identity -- "different means they are different objects no
        // matter what any derived hash says". That inference only holds one
        // way. Two distinct objects sharing one material carry the SAME
        // pointer, and one object whose material binding is re-created between
        // frames carries a DIFFERENT one. Because this is a hard `continue`,
        // the vertex/bone/material fallback that would still have paired them
        // never runs.
        //
        // engRejWouldHash counts exactly the rejections where those hash
        // clauses WOULD have accepted the pairing. If it tracks engRej, this
        // gate is manufacturing the one-frame entries.
        ++dbgEngReject;
        if ((vertexDataMatches && boneHashesMatch) || materialHashesMatch) {
          ++dbgEngRejectWouldHashMatch;
        }
        continue;
      }
    } else if (!(vertexDataMatches && boneHashesMatch || materialHashesMatch)) {
      ++dbgHashReject;
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
    // 2) Ranking was distance-major, but per-frame-rewritten batches have no
    //    stable centroid: measured median frame-to-frame nearest-neighbour
    //    drift is 507 world units ([FindSim] capture 2026-08-02), far past
    //    any bonus margin. So the draw->entry assignment ROTATED every
    //    frame; each frame a different handful of entries went unpaired,
    //    their instances were reaped (numFramesToKeepInstances) and
    //    re-created on re-pairing — ~10 fresh instance ids per frame on a
    //    population that is completely stable. Rank by recency
    //    (frameLastTouched) FIRST so last frame's working set of entries
    //    keeps being paired, and use the old score only to break ties.
    //    Content-stable meshes that land here once (e.g. a streaming
    //    rewrite) all tie on recency, so distance still picks the right
    //    twin for them. Recency is exact — no threshold, no distance term.
    //    (A brand-new entry carries frameLastTouched == kInvalidFrameIndex
    //    == UINT32_MAX, but it can never be seen here: SceneManager touches
    //    every entry immediately after DrawCallCache::get returns, so a
    //    same-frame sibling is skipped by the frameLastTouched guard above.)
    if (*out == nullptr
        || blas.frameLastTouched > bestTouched
        || (blas.frameLastTouched == bestTouched && score > bestScore)) {
      bestTouched = blas.frameLastTouched;
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
    // NV-DXVK [EntryReject]: one line per allocation, naming which clause turned
    // every sibling away. A fresh entry has an empty SpatialMap, so this line is
    // the direct cause of an instance respawn and of the reap that follows.
    //
    // Read engRej against engRejWouldHash. Equal means the engine-pointer gate
    // rejected pairings that the hash clauses would have accepted, and the gate
    // is the defect. engRejWouldHash near zero means the siblings genuinely were
    // different objects and the allocation is correct -- then the churn is
    // upstream of here, in whatever makes the bucket multi-entry.
    //
    // Gated on the probe VS list so it stays scoped to the shader being chased,
    // and hard-capped because an allocation storm would otherwise be a firehose.
    if (lookupHash(RtxOptions::findSimilarProbeVsHashes(),
                   drawCall.getTransformData().vertexShaderHash)) {
      static std::atomic<uint32_t> sEntryRejectLines { 0 };
      constexpr uint32_t kMaxEntryRejectLines = 3000;
      // Local: the [BucketMiss] block above scopes its own copy.
      const size_t entryRejBucketN = m_entries.count(hash);
      if (sEntryRejectLines.fetch_add(1, std::memory_order_relaxed) < kMaxEntryRejectLines) {
        Logger::info(str::format(
          "[EntryReject] f=", m_device->getCurrentFrameId(),
          " vs=0x", std::hex,
          static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash),
          " dEng=0x", static_cast<uint64_t>(drawCall.engineMaterialPtr), std::dec,
          " siblings=", dbgSiblings,
          " engRej=", dbgEngReject,
          " engRejWouldHash=", dbgEngRejectWouldHashMatch,
          " hashRej=", dbgHashReject,
          " touchedRej=", dbgTouchedReject,
          " spaceRej=", dbgSpaceReject,
          " shaderRej=", dbgShaderReject,
          " bucketN=", entryRejBucketN,
          " | engRej~=engRejWouldHash => the engineMaterialPtr gate is"
          " manufacturing this entry"));
      }
    }

    // Failed to find similar blas, so allocate a new one
    g_brMissed.fetch_add(1, std::memory_order_relaxed);
    *out = allocateEntry(hash, drawCall);
    // NV-DXVK [MvRaw]: stamp the pairing decision onto the entry. A fresh entry
    // has an EMPTY SpatialMap, so findSimilarInstance is guaranteed to miss and
    // the instance is re-created -> teleport() -> prev == cur -> ZERO motion.
    // That is the ghosting/lag signature, and it is the opposite failure to the
    // rescued case below, so the two must be distinguishable in the log.
    (*out)->lastPairFrame       = m_device->getCurrentFrameId();
    (*out)->lastPairPrevTouched = kInvalidFrameIndex;
    (*out)->lastPairScore       = 0.f;
    (*out)->lastPairKind        = 0u;
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
  // NV-DXVK [MvRaw]: record WHICH pairing produced this entry, for the motion
  // probe to read. bestTouched is captured before the caller touches the entry
  // again, so it is the staleness that actually decided the ranking.
  (*out)->lastPairFrame       = m_device->getCurrentFrameId();
  (*out)->lastPairPrevTouched = bestTouched;
  (*out)->lastPairScore       = bestScore;
  (*out)->lastPairKind        = (bestScore > 0.0f) ? 1u : 2u;
  return CacheState::kExisted;

}

BlasEntry* DrawCallCache::allocateEntry(XXH64_hash_t hash, const DrawCallState& drawCall) {
  auto iter = m_entries.emplace(hash, drawCall);
  BlasEntry* result = &iter->second;
  result->frameCreated = m_device->getCurrentFrameId();

  // NV-DXVK [MatBind identity]: register under engine identity so future
  // frames can find this entry even after the content-derived bucket key
  // goes stale (re-batched geometry). Key stored on the entry so GC removal
  // uses the registration key (see BlasEntry::engineClassKey).
  if (drawCall.engineMaterialPtr != 0) {
    result->engineClassKey = makeEngineClassKey(
        drawCall.engineMaterialPtr, drawCall.getTransformData().vertexShaderHash);
    m_engineClassIndex.emplace(result->engineClassKey, result);
  }

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
