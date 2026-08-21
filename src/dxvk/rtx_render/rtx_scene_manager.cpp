/*
* Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <mutex>
#include <thread>
#include <unordered_set>
// NV-DXVK [Perf.Report]: [ProcDCS], [Perf.PrepScene] and [MatChurn] all emit
// from this file and are the dxvk-cs body plus the hygiene gate.
#include "rtx_perf_report.h"
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "rtx_asset_replacer.h"
#include "rtx_scene_manager.h"
#include "../dxvk_alloc_probe.h"
#include "rtx_cpu_stall_probe.h"
#include "rtx_opacity_micromap_manager.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_buffer.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_mesh_trace.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_xess.h"
#include "rtx_asset_exporter.h"
#include "../../util/util_filesys.h"
#include "../../util/util_env.h"
#include "../../lssusd/game_exporter_paths.h"

#include <assert.h>

#include "rtx_cb_types.h"
#include "vulkan/vulkan_core.h"

#include "rtx_game_capturer.h"
#include "rtx_matrix_helpers.h"
#include "rtx_intersection_test.h"

#include "dxvk_scoped_annotation.h"
#include "rtx_lights_data.h"
#include "rtx_light_utils.h"

#include "../util/util_globaltime.h"

namespace dxvk {

  // NV-DXVK [Phase2b]: the CS-side consume pointer. Set by
  // RtxContext::commitGeometryToRT around its submitDrawState call for a
  // batched draw; null everywhere else (external draws, replacements'
  // recursion, the option-off path), which makes every consumer below fall
  // through to the unchanged legacy behavior. CS-thread-local by construction.
  thread_local ShardedDrawInfo* t_shardedConsume = nullptr;

  // NV-DXVK [MatChurn]: defined in rtx_texture_manager.cpp, incremented in the
  // D3D11 layer whenever a game image enters the material path without a hash.
  // See SceneManager::logMaterialChurn.
  extern std::atomic<uint64_t> g_newGameImageHashStamps;

  // NV-DXVK [SceneClearProbe]: defined in rtx_camera_manager.cpp; used in
  // SceneManager::clear() to gate the unconditional log on gameplay so we
  // don't emit ~1024 lines during pre-gameplay loading (every load-frame
  // invalid-scene triggers a clear with default sceneKeepAliveFrames=0).
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;

    // NV-DXVK [flicker V8 follow-up: capture feedback] — DEFINITIONS.
    // Published by SceneManager::processGeometryInfo (CS thread) whenever a
    // bake consumed an uncaptured device-local source; snapshotted once per
    // frame by D3D11Rtx::SubmitDraw (game thread) to steer the
    // SubmitDraw-ordered geometry capture at draws that are about to
    // re-bake. Keys come from RasterGeometry::captureFeedbackKey.
    //
    // V1 was a 64-slot lock-free round-robin ring. ROOT DEFECT (the reason
    // the capture recovery never converged, 2026-08-03): ~140 starving bakes
    // per frame wrote the ring unthrottled, so it wrapped WITHIN a frame and
    // the once-per-frame consumer snapshot missed almost every key —
    // capState=none on 100% of [GeoCapture.wanted] lines across entire runs.
    // The recovery then hit its 8-attempt valve and froze garbage bakes:
    // the seconds-long whole-group flicker. A keyed map dedups the ~140
    // writes down to the handful of distinct keys and loses nothing.
    // Contention is one uncontended lock per starving bake (CS) plus one
    // lock per frame (game thread) — not a hot path.
    // Entries older than the consumer's 6-frame window are pruned by the
    // consumer each frame; the publisher also prunes if the map ever grows
    // past a safety bound (dead keys from a level change, feature toggled).
    std::mutex g_geomCaptureWantedMutex;
    std::unordered_map<uint64_t, uint32_t> g_geomCaptureWantedMap;

    // NV-DXVK [capture stability contract] — DEFINITIONS. The wanted-key
    // feedback above is retrospective and therefore structurally CANNOT
    // cover the first bake of a re-batched entry: the key is minted by the
    // bake that already ran uncaptured, and the next re-batch mints a
    // different one (isNew=1 on 100% of starving lines). So the capture
    // decision is inverted: the game thread captures every eligible
    // device-local draw UNLESS the CS thread has recently proven the draw's
    // identity key stable. Stable = its BlasEntry resolved kUpdateInstance
    // with no pendingSrcBake; any bake (KBuildBVH/kUpdateBVH) taints the
    // key. Keys are RasterGeometry::captureIdentityKey (stamped at the
    // SubmitDraw capture site). Values are the frame of the verdict; the
    // consumer prunes entries older than its freshness window, so a key
    // that stops being re-confirmed automatically falls back to captured.
    std::mutex g_geomCaptureStableMutex;
    std::unordered_map<uint64_t, uint32_t> g_geomCaptureStableMap;
    std::unordered_map<uint64_t, uint32_t> g_geomCaptureTaintMap;
  }

  // NV-DXVK TF2 vanish-zone diagnostic. Tracks draws-in / draws-ignored /
  // instances-kept per frame, and dumps counts + camera state when the kept
  // count drops sharply between frames (suspected PVS/cluster boundary, frustum
  // misfire, or camera-classification flip). Gated on classifier-set Main and
  // raytracedThisFrame (skips menu/loading frames per project preference).
  struct VanishDiag {
    uint32_t lastFrameId   = UINT32_MAX;
    uint32_t drawsIn       = 0;
    uint32_t drawsIgnored  = 0;
    uint32_t drawsKept     = 0;
    uint32_t baselineKept  = 0;   // running max kept across gameplay frames
    uint32_t lastLoggedFid = UINT32_MAX;
    // Per-VS-hash submission histogram for the current frame, plus a snapshot
    // from the most recent "good" frame (kept >= 95% of baseline). On a cliff
    // frame we diff the two to identify which VS bucket(s) lost draws.
    std::unordered_map<XXH64_hash_t, uint32_t> vsHistogram;
    std::unordered_map<XXH64_hash_t, uint32_t> baselineVsHistogram;
  };
  static VanishDiag& vanishDiag() { static VanishDiag s; return s; }

  SceneManager::SceneManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    , m_instanceManager(device, this)
    , m_accelManager(device)
    , m_lightManager(device)
    , m_graphManager()
    , m_rayPortalManager(device, this)
    , m_drawCallCache(device)
    , m_bindlessResourceManager(device)
    , m_pReplacer(new AssetReplacer())
    , m_terrainBaker(new TerrainBaker())
    , m_cameraManager(device)
    , m_uniqueObjectSearchDistance(RtxOptions::uniqueObjectDistance()) {
    m_indexStashPool = std::make_shared<IndexStashPool>(device);
    InstanceEventHandler instanceEvents(this);
    instanceEvents.onInstanceAddedCallback = [this](RtInstance& instance) { onInstanceAdded(instance); };
    instanceEvents.onInstanceUpdatedCallback = [this](RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, bool hasTransformChanged, bool hasVerticesChanged, bool isFirstUpdateThisFrame) { onInstanceUpdated(instance, drawCall, material, hasTransformChanged, hasVerticesChanged, isFirstUpdateThisFrame); };
    instanceEvents.onInstanceDestroyedCallback = [this](RtInstance& instance) { onInstanceDestroyed(instance); };
    // NV-DXVK [perf] sec 4c: this handler's onInstanceUpdated resolves a surface
    // material and binds its index. m_surfaceMaterialCache is never cleared per
    // frame, so for an instance whose material inputs and the binding epoch are
    // both unchanged it re-derives the index the instance already holds. Opt in to
    // being skipped in that case -- see the gate at the bottom of updateInstance.
    // The RayPortal path (processRayPortalData) is excluded there, not here.
    instanceEvents.skippableWhenBindingUnchanged = true;
    m_instanceManager.addEventHandler(instanceEvents);
    
    if (env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME") != "") {
      m_beginUsdExportFrameNum = stoul(env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME"));
    }
  }

  SceneManager::~SceneManager() {
  }

  bool SceneManager::areAllReplacementsLoaded() const {
    return m_pReplacer->areAllReplacementsLoaded();
  }

  std::vector<Mod::State> SceneManager::getReplacementStates() const {
    return m_pReplacer->getReplacementStates();
  }

  void SceneManager::initialize(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();
    m_pReplacer->initialize(ctx);
  }

  void SceneManager::logStatistics() {
    if (m_opacityMicromapManager.get()) {
      m_opacityMicromapManager->logStatistics();
    }
  }

  Vector3 SceneManager::getSceneUp() {
    return RtxOptions::zUp() ? Vector3(0.f, 0.f, 1.f) : Vector3(0.f, 1.f, 0.f);
  }

  Vector3 SceneManager::getSceneForward() {
    return RtxOptions::zUp() ? Vector3(0.f, 1.f, 0.f) : Vector3(0.f, 0.f, 1.f);
  }

  Vector3 SceneManager::calculateSceneRight() {
    const Vector3 up = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    return RtxOptions::leftHandedCoordinateSystem() ? cross(up, forward) : cross(forward, up);
  }

  Vector3 SceneManager::worldToSceneOrientedVector(const Vector3& worldVector) {
    return RtxOptions::zUp() ? worldVector : Vector3(worldVector.x, worldVector.z, worldVector.y);
  }

  Vector3 SceneManager::sceneToWorldOrientedVector(const Vector3& sceneVector) {
    // Same transform applies to and from
    return worldToSceneOrientedVector(sceneVector);
  }

  float SceneManager::getTotalMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();

    const bool temporalUpscaling = RtxOptions::isDLSSOrRayReconstructionEnabled() || RtxOptions::isXeSSEnabled() || RtxOptions::isTAAEnabled();

    float totalUpscaleMipBias = 0.0f;

    if (temporalUpscaling) {
      if (RtxOptions::isXeSSEnabled()) {
        // XeSS uses the new formula from the XeSS developer guide
        totalUpscaleMipBias = -log2(resourceManager.getUpscaleRatio());

        // Add XeSS-specific mip bias when XeSS is active
        DxvkXeSS& xess = m_device->getCommon()->metaXeSS();
        if (xess.isActive()) {
          float xessMipBias = xess.calcRecommendedMipBias();
          totalUpscaleMipBias += xessMipBias;
        }
      } else {
        // Restore original behavior for DLSS, TAA, and other upscalers
        totalUpscaleMipBias = log2(resourceManager.getUpscaleRatio()) + RtxOptions::upscalingMipBias();
      }
    }

    return totalUpscaleMipBias + RtxOptions::nativeMipBias();
  }

  float SceneManager::getCalculatedUpscalingMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
    
    const bool temporalUpscaling = RtxOptions::isXeSSEnabled();
    if (!temporalUpscaling) {
      return 0.0f;
    }
    
    float calculatedUpscalingBias = -log2(resourceManager.getUpscaleRatio());
    return calculatedUpscalingBias;
  }

  void SceneManager::clear(Rc<DxvkContext> ctx, bool needWfi) {
    ScopedCpuProfileZone();

    // NV-DXVK [SceneClearProbe]: log every clear during gameplay so we
    // catch silent SpatialMap wipes that break sub-view identity dedup.
    // Gated behind tf2::g_engineHookCaptureCount > 16 (the same gameplay
    // threshold the InvalidSceneProbe uses) to avoid emitting ~1024 lines
    // during pre-gameplay loading, where clears are expected with the
    // default sceneKeepAliveFrames=0.
    if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      Logger::warn(str::format(
        "[SceneClearProbe] f=", m_device->getCurrentFrameId(),
        " needWfi=", (needWfi ? 1 : 0),
        " — SceneManager::clear() entered during gameplay; SpatialMaps will be destroyed"));
    }

    // NV-DXVK [SceneClearRaw]: UNGATED (logSurfaceCoverage) companion to the
    // gameplay-gated [SceneClearProbe] above, which is suppressed until
    // engineHookCaptureCount>16 and so misses the load/transition window where
    // the observed f695-style reset (light list collapse + texture reload +
    // multi-frame reconverge) actually fires. Logs EVERY clear so it can be
    // cross-referenced with [Coverage] FinalGrid + [LightProbe] by frame id.
    if (RtxOptions::logSurfaceCoverage()) {
      Logger::info(str::format(
        "[SceneClearRaw] f=", m_device->getCurrentFrameId(),
        " needWfi=", (needWfi ? 1 : 0),
        " engineHookCaptureCount=", tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed)));
    }

    // NV-DXVK [ResidentScene] slice 8: the scene this was built from is gone, so
    // whatever residency had harvested describes a level that is no longer
    // loaded. Publishing it here, at the single point every reset funnels
    // through, is what lets the frame thread arm the seed pass without needing
    // to recognise a level change itself. See g_sceneEpoch.
    g_sceneEpoch.fetch_add(1u, std::memory_order_relaxed);

    auto& textureManager = m_device->getCommon()->getTextureManager();

    // Only clear once after the scene disappears, to avoid adding a WFI on every frame through clear().
    if (needWfi) {
      if (ctx.ptr())
        ctx->flushCommandList();
      m_device->waitForIdle();
    }

    // We still need to clear caches even if the scene wasn't rendered
    m_bufferCache.clear();
    m_surfaceMaterialCache.clear();
    m_preCreationSurfaceMaterialMap.clear();
    // NV-DXVK [perf]: the memo holds an index INTO m_surfaceMaterialCache, so it
    // must die with it -- a scene reset renumbers every material. ALL slots, not
    // this thread's: a surviving worker entry would name a renumbered material.
    invalidateSurfaceMaterialMemo();
    m_legacyOpaqueConversionCache.clear();
    m_surfaceMaterialExtensionCache.clear();
    m_volumeMaterialCache.clear();
    
    // Called before instance manager's clear, so that it resets all tracked instances in Opacity Micromap manager at once
    if (m_opacityMicromapManager.get())
      m_opacityMicromapManager->clear();
    
    m_instanceManager.clear();
    m_lightManager.clear();
    m_graphManager.clear();
    m_rayPortalManager.clear();
    m_drawCallCache.clear();
    textureManager.clear();

    m_previousFrameSceneAvailable = false;
  }

  void SceneManager::garbageCollection() {
    ScopedCpuProfileZone();

    // NV-DXVK [perf, GPU index stash]: return index-stash buffers that nobody
    // has re-acquired for a long time. In steady state this frees nothing (every
    // pooled buffer is reused within a frame or two); it exists to hand back
    // memory stranded in size classes a previous scene used and this one does
    // not. Cheap: a few buckets, only touched once per frame.
    if (m_indexStashPool) {
      m_indexStashPool->reclaim(m_device->getCurrentFrameId());
    }

    // NV-DXVK [SceneGc]: per-pass diagnostic counters. Logged at exit so we
    // see how many BLASes the GC pass looked at, how many it destroyed, and
    // how many marked-inside-frustum (lifetime-eligible) instances were
    // produced by the BLAS frustum loop. If we lose mountains on level
    // load, this tells us whether BLAS GC is too aggressive (high destroy
    // count) or the frustum loop is misclassifying (high markedInside).
    uint32_t sceneGcBlasIterated  = 0;
    uint32_t sceneGcBlasDestroyed = 0;
    uint32_t sceneGcInstSeen      = 0;
    uint32_t sceneGcInstInside    = 0;
    uint32_t sceneGcInstOutside   = 0;
    uint32_t sceneGcInstIgnAC     = 0;
    uint32_t sceneGcDestroyed2904 = 0;
    const uint32_t sceneGcFrame   = m_device->getCurrentFrameId();
    const bool sceneGcInGameplay  =
      tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;

    const size_t oldestFrame = m_device->getCurrentFrameId() - RtxOptions::numFramesToKeepGeometryData();

    // NV-DXVK [ResidentScene]: THE GEOMETRY HAS TO OUTLIVE THE INSTANCE, and
    // without this clause it does not.
    //
    // rtx.numFramesToKeepBLAS is 1, so a BlasEntry is destroyed the frame after
    // its last draw. onSceneObjectDestroyed then marks every linked instance for
    // collection, and m_isMarkedForGC is a clause residency deliberately does
    // NOT override -- an instance must never outlive its geometry. So an
    // off-screen object would have its instances faithfully exempted from
    // lifetime expiry by the residency clause in InstanceManager::
    // garbageCollection, and then reaped one frame later anyway through the
    // geometry. Residency would measure as working and do nothing.
    //
    // Asking the record store per BLAS rather than stamping frameLastTouched
    // from the touch: a touch only happens when a DRAW arrives, and the whole
    // population this feature exists for is the one whose draws do not.
    //
    // Costs a walk of the linked instances, and only for entries that have
    // already aged past the bound -- i.e. the ones about to be destroyed.
    const bool residencyKeepsGeometry =
      RtxOptions::ResidentScene::enable() && !RtxOptions::ResidentScene::verify();
    const ResidentScene& residentScene = m_instanceManager.getResidentScene();

    auto blasEntryGarbageCollection = [&](auto& iter, auto& entries) -> void {
      if (iter->second.frameLastTouched < oldestFrame) {
        if (residencyKeepsGeometry) {
          bool heldResident = false;
          for (const RtInstance* instance : iter->second.getLinkedInstances()) {
            if (residentScene.holdsInstance(instance)) {
              heldResident = true;
              break;
            }
          }
          if (heldResident) {
            ++iter;
            return;
          }
        }
        // NV-DXVK [BlasDestroyed]: log every BLAS destruction during
        // gameplay, with vsHash + linkedInstance count + frame-since-
        // touch. Rate-limited to bound output. Sub-view-mountain BLASes
        // (vsHash=0x2904d2) get an extra always-on tag in the totals.
        const XXH64_hash_t vsHash = iter->second.input.getTransformData().vertexShaderHash;
        const size_t linkedCount  = iter->second.getLinkedInstances().size();
        if (sceneGcInGameplay) {
          static thread_local uint32_t sBlasDestProbe = 0;
          if (sBlasDestProbe < 64 || (sBlasDestProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[BlasDestroyed] #", sBlasDestProbe,
              " f=", sceneGcFrame,
              " vsHash=0x", std::hex, vsHash, std::dec,
              " linked=", linkedCount,
              " ageSinceTouch=", (sceneGcFrame - iter->second.frameLastTouched),
              " oldestFrame=", oldestFrame));
          }
          sBlasDestProbe += 1;
          sceneGcBlasDestroyed += 1;
          if (vsHash == 0x2904d2163ef31a17ull) {
            sceneGcDestroyed2904 += 1;
          }
        }
        onSceneObjectDestroyed(iter->second);
        // NV-DXVK [MatBind identity]: unregister from the engine-class index
        // BEFORE the erase — the index holds raw BlasEntry* into this map.
        m_drawCallCache.removeFromEngineClassIndex(iter->second);
        iter = entries.erase(iter);
      } else {
        if (sceneGcInGameplay) {
          sceneGcBlasIterated += 1;
        }
        ++iter;
      }
    };

    // Garbage collection for BLAS/Scene objects
    //
    // When anti-culling is enabled, we need to check if any instances are outside frustum. Because in such
    // case the life of the instances will be extended and we need to keep the BLAS as well.
    if (!RtxOptions::AntiCulling::isObjectAntiCullingEnabled()) {
      auto& entries = m_drawCallCache.getEntries();
      if (m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
        for (auto iter = entries.begin(); iter != entries.end(); ) {
          blasEntryGarbageCollection(iter, entries);
        }
      }
    }
    else { // Implement anti-culling BLAS/Scene object GC
      fast_unordered_cache<const RtInstance*> outsideFrustumInstancesCache;

      auto& entries = m_drawCallCache.getEntries();
      for (auto iter = entries.begin(); iter != entries.end();) {
        bool isAllInstancesInCurrentBlasInsideFrustum = true;
        for (const RtInstance* instance : iter->second.getLinkedInstances()) {
          const Matrix4 objectToView = getCamera().getWorldToView(false) * instance->getTransform();

          bool isInsideFrustum = true;
          // Check for camera cut. Anti-Culling should NOT be enabled during a camera cut.
          // In some cases, we can't reliably detect a camera cut (e.g., when the game doesn't set up the View Matrix),
          // so we must disable Anti-Culling to prevent visual corruption.
          if (!getCamera().isCameraCut() && m_isAntiCullingSupported) {
            if (RtxOptions::needsMeshBoundingBox()) {
              const AxisAlignedBoundingBox& boundingBox = instance->getBlas()->input.getGeometryData().boundingBox;
              if (RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCulling()) {
                // [InfFar] Source the isInfFrustum flag from the camera's
                // actual state, not just the user option. Reverse-Z
                // infinite-far engines (TF2) report farPlane=+Inf even
                // when the option is off; the SAT culler's finite-far
                // code path applied to an infinite-far frustum produces
                // over-culling (sky-only views). See RtCamera::
                // shouldUseInfiniteFarFrustum() for the predicate.
                const bool useInfFar = getCamera().shouldUseInfiniteFarFrustum();
                isInsideFrustum = boundingBoxIntersectsFrustumSAT(
                  getCamera(),
                  boundingBox.minPos,
                  boundingBox.maxPos,
                  objectToView,
                  useInfFar);
                // [InfFar.diag] Log SAT-culler decisions wall-clock
                // throttled. Counts inside vs outside per camera type
                // to spot flicker between "everything inside" and
                // "everything outside" (the over-cull symptom).
                {
                  static thread_local uint64_t sIn[8]  = {0,0,0,0,0,0,0,0};
                  static thread_local uint64_t sOut[8] = {0,0,0,0,0,0,0,0};
                  static thread_local std::chrono::steady_clock::time_point sLastT[8] = {
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                    std::chrono::steady_clock::time_point::min(),
                  };
                  const int idx = std::min<int>(static_cast<int>(getCamera().getCameraType()), 7);
                  if (isInsideFrustum) ++sIn[idx]; else ++sOut[idx];
                  const auto now = std::chrono::steady_clock::now();
                  const bool dueByClock =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - sLastT[idx]).count() >= 500;
                  if (dueByClock) {
                    sLastT[idx] = now;
                    Logger::warn(str::format(
                      "[InfFar.diag.SAT] camType=", static_cast<int>(getCamera().getCameraType()),
                      " useInfFar=", (useInfFar ? 1 : 0),
                      " inside=", sIn[idx],
                      " outside=", sOut[idx],
                      " thisDrawInside=", (isInsideFrustum ? 1 : 0),
                      " far=", getCamera().getFarPlane(),
                      " near=", getCamera().getNearPlane()));
                    sIn[idx] = 0;
                    sOut[idx] = 0;
                  }
                }
              } else {
                isInsideFrustum = boundingBoxIntersectsFrustum(getCamera().getFrustum(), boundingBox.minPos, boundingBox.maxPos, objectToView);
              }
            }
            else {
              // Fallback to check object center under view space
              auto getViewSpacePosition = [](const Matrix4& objectToView) -> float3 {
                return float3(objectToView[3][0], objectToView[3][1], objectToView[3][2]);
              };
              isInsideFrustum = getCamera().getFrustum().CheckSphere(getViewSpacePosition(objectToView), 0);
            }
          }

          // NV-DXVK [HullSAT]: instrument WHY the anti-frustum test drops the
          // dropship hull (VS 0x292b) on despawn frames. The populated object-
          // space box did NOT stop the despawn (ShipScreen hullPx=0 while
          // HullCensus shows the instances present + notInFrustum spikes), so
          // the failure is in the box-vs-view transform or the camera fed to
          // the SAT, NOT an empty box. For the first few hull instances per
          // frame, dump: the object-space box, the view-space AABB actually
          // tested (8 transformed corners), the objectToView / worldToView
          // translations, the camera-cut state, and the decision. Diff a
          // full-ship frame (inside=1) vs a despawn frame (inside=0):
          //   - objBox sane + viewBox lands in a crazy position  => bad camera
          //     / objectToView (the worldToView the cull uses is desynced from
          //     the render camera — matches the logged camFwd not matching the
          //     player's actual view).
          //   - objBox degenerate/empty                          => box bug.
          //   - cut=1 on the despawn frame                       => the cut gate
          //     above was skipped yet the instance still ends up outside (look
          //     at how m_isInsideFrustum is left/handled on cut frames).
          // Logged OUTSIDE the cut gate so cut frames are captured too; box is
          // re-fetched here because the gate-scoped reference is out of scope.
          if (sceneGcInGameplay) {
            const uint64_t hullVs = static_cast<uint64_t>(
              instance->getBlas()->input.getTransformData().vertexShaderHash);
            if (hullVs == 0x292b6ba0d1854f28ull) {
              static thread_local uint32_t s_hullSatFrame = 0xFFFFFFFFu;
              static thread_local uint32_t s_hullSatCount = 0u;
              const uint32_t fid = m_device->getCurrentFrameId();
              if (fid != s_hullSatFrame) { s_hullSatFrame = fid; s_hullSatCount = 0u; }
              if (s_hullSatCount < 6u) {
                s_hullSatCount++;
                const AxisAlignedBoundingBox& hb =
                  instance->getBlas()->input.getGeometryData().boundingBox;
                const Vector3 bmin = hb.minPos;
                const Vector3 bmax = hb.maxPos;
                Vector3 vMin( 1e30f,  1e30f,  1e30f);
                Vector3 vMax(-1e30f, -1e30f, -1e30f);
                for (int c = 0; c < 8; ++c) {
                  const Vector4 lh(
                    (c & 1) ? bmax.x : bmin.x,
                    (c & 2) ? bmax.y : bmin.y,
                    (c & 4) ? bmax.z : bmin.z, 1.0f);
                  const Vector4 vh = objectToView * lh;
                  vMin.x = std::min(vMin.x, vh.x); vMax.x = std::max(vMax.x, vh.x);
                  vMin.y = std::min(vMin.y, vh.y); vMax.y = std::max(vMax.y, vh.y);
                  vMin.z = std::min(vMin.z, vh.z); vMax.z = std::max(vMax.z, vh.z);
                }
                const Matrix4 w2v = getCamera().getWorldToView(false);
                Logger::warn(str::format(
                  "[HullSAT] f=", fid,
                  " inside=", (isInsideFrustum ? 1 : 0),
                  " cut=", (getCamera().isCameraCut() ? 1 : 0),
                  " boxValid=", (hb.isValid() ? 1 : 0),
                  " objBox=(", bmin.x, ",", bmin.y, ",", bmin.z,
                    ")..(", bmax.x, ",", bmax.y, ",", bmax.z, ")",
                  " viewBox=(", vMin.x, ",", vMin.y, ",", vMin.z,
                    ")..(", vMax.x, ",", vMax.y, ",", vMax.z, ")",
                  " o2vT=(", objectToView[3][0], ",", objectToView[3][1], ",",
                    objectToView[3][2], ")",
                  " w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")",
                  " near=", getCamera().getNearPlane(),
                  " far=", getCamera().getFarPlane()));
              }
            }
          }

          // Only GC the objects inside the frustum to anti-frustum culling, this could cause significant performance impact
          // For the objects which can't be handled well with this algorithm, we will need game specific hash to force keeping them
          const bool hasIgnoreAntiCulling = instance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
          // NV-DXVK [SceneGc]: per-instance counters for the GC-pass summary.
          if (sceneGcInGameplay) {
            sceneGcInstSeen += 1;
            if (hasIgnoreAntiCulling) sceneGcInstIgnAC += 1;
          }
          if (isInsideFrustum && !hasIgnoreAntiCulling) {
            if (sceneGcInGameplay) sceneGcInstInside += 1;
            instance->markAsInsideFrustum();
          } else if (hasIgnoreAntiCulling) {
            if (sceneGcInGameplay) sceneGcInstOutside += 1;
            // NV-DXVK [IgnoreAntiCulling escape]: skip the anti-culling-dedup
            // override entirely for IgnoreAntiCulling instances. Sub-view-
            // reprojected content (TF2 skybox mountains, trees) lives outside
            // the main camera's frustum honest-cull domain. Their world
            // positions drift each frame due to engine-side reproject scale
            // amplifying sub-view camera motion, but the prop is the same.
            // The override below would mark the older of any duplicate pair
            // as inside-frustum → lifetime-GC'd → spatial map entry erased →
            // next-frame propId lookup misses → new instance created → next
            // frame TWO duplicates → cascade. Skipping the override here
            // means: even if a transient duplicate appears, both stay outside-
            // frustum, lifetime-GC is gated on the other (skinning/animated/
            // playerModel) clauses which don't apply, and identity dedup
            // recovers on the next frame's lookup. The trade-off: persistent
            // true duplicates would stack indefinitely, but that's the very
            // thing identity dedup is designed to prevent.
            instance->markAsOutsideFrustum();
            isAllInstancesInCurrentBlasInsideFrustum = false;
          } else {
            if (sceneGcInGameplay) sceneGcInstOutside += 1;
            instance->markAsOutsideFrustum();
            isAllInstancesInCurrentBlasInsideFrustum = false;

            // Anti-Culling GC extension:
            // Eliminate duplicated instances that are outside of the game frustum.
            // This is used to handle cases:
            //   1. The game frustum is different to our frustum
            //   2. The game culling method is NOT frustum culling

            const XXH64_hash_t antiCullingHash = instance->calculateAntiCullingHash();

            auto it = outsideFrustumInstancesCache.find(antiCullingHash);
            if (it == outsideFrustumInstancesCache.end()) {
              // No duplication, just cache the current instance
              outsideFrustumInstancesCache[antiCullingHash] = instance;
            } else {
              const RtInstance* cachedInstance = it->second;
              if (instance->getId() != cachedInstance->getId()) {
                // Only keep the instance that is latest updated
                if (instance->getFrameLastUpdated() < cachedInstance->getFrameLastUpdated()) {
                  instance->markAsInsideFrustum();
                } else {
                  cachedInstance->markAsInsideFrustum();
                  it->second = instance;
                }
              }
            }
          }
        }

        // If all instances in current BLAS are inside the frustum, then use original GC logic to recycle BLAS Objects
        if (isAllInstancesInCurrentBlasInsideFrustum &&
            m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
          blasEntryGarbageCollection(iter, entries);
        } else { // If any instances are outside of the frustum in current BLAS, we need to keep the entity
          ++iter;
        }
      }
    }

    // NV-DXVK [SceneGcSummary]: one line per GC pass during gameplay,
    // before the instance/accel/light/portal managers run their GC. Tells
    // us at-a-glance whether the BLAS GC pass is destroying lots of
    // entries (bad — geometry will pop out), whether the BLAS frustum loop
    // is correctly tagging instances OutsideFrustum (good — they survive
    // lifetime GC), and how many sub-view-mountain (vsHash=0x2904d2)
    // BLASes got destroyed this pass (any non-zero is suspicious).
    if (sceneGcInGameplay) {
      Logger::info(str::format(
        "[SceneGcSummary] f=", sceneGcFrame,
        " blasIterated=", sceneGcBlasIterated,
        " blasDestroyed=", sceneGcBlasDestroyed,
        " blasDest2904=", sceneGcDestroyed2904,
        " instSeen=", sceneGcInstSeen,
        " instInside=", sceneGcInstInside,
        " instOutside=", sceneGcInstOutside,
        " instIgnAC=", sceneGcInstIgnAC,
        " oldestFrame=", oldestFrame,
        " keepBLAS=", RtxOptions::numFramesToKeepGeometryData()));
    }

    // Perform GC on the other managers
    m_instanceManager.garbageCollection();
    m_accelManager.garbageCollection();
    m_lightManager.garbageCollection(getCamera());
    m_rayPortalManager.garbageCollection();

    // NV-DXVK [stable buffer identity]: free the bindless slots of retired
    // buffers that nothing reads through any more.
    //
    // AFTER the instance reap above, deliberately. An entry destroyed by the
    // BLAS pass marks its linked instances and m_instanceManager.
    // garbageCollection() removes them in this same call, so by this point an
    // instance cannot be the thing keeping one of those slots alive.
    //
    // kMaxFramesInFlight is the wait, and it is a pipeline fact rather than a
    // guess about lifetimes. uploadSurfaceData rewrites the whole surface buffer
    // and marks every index a live surface carries EVERY frame, so a slot that
    // has gone unmarked for longer than the pipeline depth is one that no
    // already-submitted frame can still read through either. Recycling on age
    // instead would eventually hand a live wrong buffer to a surface that
    // outlived the guess -- silently wrong geometry, with no FAIL to catch it.
    // How many it freed is reported through [MatChurn] bufFreed, which reads the
    // table's own monotonic counter rather than this call's return.
    m_bufferCache.reclaim(m_device->getCurrentFrameId(), kMaxFramesInFlight);
  }

  void SceneManager::onDestroy() {
    m_accelManager.onDestroy();
    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onDestroy();
    }
    // NV-DXVK [perf, GPU index stash]: drop cached free buffers while the
    // device is still alive — DxvkBuffer holds a raw DxvkDevice*.
    if (m_indexStashPool) {
      m_indexStashPool->shutdown();
    }
  }

  // NV-DXVK [perf, GPU index stash]: see IndexStashPool in rtx_scene_manager.h
  // for the rationale (this replaced a per-draw createBuffer).
  VkDeviceSize IndexStashPool::sizeClassFor(VkDeviceSize bytes) {
    // Power-of-two classes so any released buffer is interchangeable with any
    // later request of the same class. Guard the shift: an absurd request
    // (corrupt indexCount) must not spin or wrap to zero — serve it exactly and
    // let it fall out of the pool on release (its class exceeds kMaxHeldBytes).
    constexpr VkDeviceSize kMaxClass = 1ull << 30;  // 1 GiB
    if (bytes > kMaxClass) {
      return bytes;
    }
    VkDeviceSize cls = kMinClassBytes;
    while (cls < bytes) {
      cls <<= 1;
    }
    return cls;
  }

  std::shared_ptr<RasterGeometry::IndexGpuStash> IndexStashPool::acquire(VkDeviceSize bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    const VkDeviceSize cls = sizeClassFor(bytes);

    Rc<DxvkBuffer> buffer;
    {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      auto it = m_free.find(cls);
      if (it != m_free.end() && !it->second.empty()) {
        // Take the most recently released buffer — it is the one least likely
        // to have aged out, which keeps the reclaim pass focused on genuine
        // surplus rather than churning the working set.
        buffer = std::move(it->second.back().buffer);
        it->second.pop_back();
        m_bytesHeld.fetch_sub(cls, std::memory_order_relaxed);
        m_reused.fetch_add(1u, std::memory_order_relaxed);
      }
    }

    if (buffer == nullptr) {
      DxvkBufferCreateInfo info;
      // NV-DXVK [flicker V8]: the pool now also serves the SubmitDraw-ordered
      // geometry capture (RasterGeometry::gpuCapture), whose buffers are
      // REBOUND as the geometry's source streams — so besides the
      // transfer-in/transfer-out the index stash needed, captures are read as
      // storage buffers by the interleave / gen-trilist compute passes and may
      // be consumed as vertex/index input (terrain baker re-rasterization).
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                 | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
                  | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                  | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT
                  | VK_ACCESS_TRANSFER_READ_BIT
                  | VK_ACCESS_SHADER_READ_BIT
                  | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
                  | VK_ACCESS_INDEX_READ_BIT;
      info.size = cls;
      buffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                      DxvkMemoryStats::Category::RTXBuffer,
                                      "Index Stash Buffer");
      if (buffer == nullptr) {
        return nullptr;
      }
      m_created.fetch_add(1u, std::memory_order_relaxed);
    }

    // The deleter checks the buffer back in. weak_ptr so a handle that outlives
    // the pool (shutdown, or device teardown ordering) just frees normally.
    std::weak_ptr<IndexStashPool> weakSelf = weak_from_this();
    auto* raw = new RasterGeometry::IndexGpuStash();
    raw->buffer = buffer;
    raw->size = bytes;
    return std::shared_ptr<RasterGeometry::IndexGpuStash>(
      raw, [weakSelf](RasterGeometry::IndexGpuStash* p) {
        if (p != nullptr) {
          if (auto pool = weakSelf.lock()) {
            pool->release(std::move(p->buffer));
          }
          delete p;
        }
      });
  }

  void IndexStashPool::release(Rc<DxvkBuffer>&& buffer) {
    if (buffer == nullptr) {
      return;
    }
    const VkDeviceSize cls = buffer->info().size;
    const uint32_t frameId = m_device->getCurrentFrameId();
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    if (m_shutdown) {
      return;  // buffer destructs here, while the device is still alive
    }
    // Always accept the return — no cap. See kIdleFramesBeforeRelease in the
    // header for why a cap here would be a perf cliff rather than a bound.
    // Surplus is handled by reclaim(), which frees what nobody re-acquires.
    m_free[cls].push_back(FreeEntry { std::move(buffer), frameId });
    m_bytesHeld.fetch_add(cls, std::memory_order_relaxed);
  }

  void IndexStashPool::reclaim(uint32_t currentFrameId) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    if (m_shutdown) {
      return;
    }
    for (auto it = m_free.begin(); it != m_free.end(); ) {
      const VkDeviceSize cls = it->first;
      auto& bucket = it->second;
      // Frame ids only advance, but guard the subtraction anyway so a wrap or
      // a reset never makes everything look infinitely old (which would free
      // the live working set and reinstate allocation churn for a frame).
      const size_t before = bucket.size();
      bucket.erase(
        std::remove_if(bucket.begin(), bucket.end(),
          [currentFrameId](const FreeEntry& e) {
            return currentFrameId >= e.lastReleasedFrameId
                && (currentFrameId - e.lastReleasedFrameId) > kIdleFramesBeforeRelease;
          }),
        bucket.end());
      const size_t freed = before - bucket.size();
      if (freed > 0) {
        m_bytesHeld.fetch_sub(cls * VkDeviceSize(freed), std::memory_order_relaxed);
        m_released.fetch_add(uint64_t(freed), std::memory_order_relaxed);
      }
      if (bucket.empty()) {
        it = m_free.erase(it);  // drop the class entirely once it is empty
      } else {
        ++it;
      }
    }
  }

  void IndexStashPool::shutdown() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    m_shutdown = true;
    m_free.clear();
    m_bytesHeld.store(0u, std::memory_order_relaxed);
  }

  // NV-DXVK [Perf.GeoChurn] — Target A instrument: how much of the per-frame
  // geometry re-cache is actually necessary?
  //
  // processGeometryInfo decides, once per BlasEntry per frame, between re-caching
  // the BLAS vertex/index inputs (KBuildBVH / kUpdateBVH — allocates buffers and
  // runs cacheVertexDataOnGPU, which is the ONLY caller of interleaveGeometry:
  // ~200 dispatches/frame, ~4.4 ms GPU in the Nsight trace) and doing nothing
  // (kUpdateInstance, free).
  //
  // [Perf.Merge] already reports the BLAS-*build* side (bBuild/bUpdate/bReuse),
  // but only for unique DYNAMIC BLAS — the ~185 instances routed to the merged
  // BLAS never reach that tally. So the geometry-cache churn that actually feeds
  // interleaveGeometry is unmeasured, and the ~200 dispatches cannot currently be
  // reconciled against the ~40 dynamic BLAS that report a change.
  //
  // Reasons are counted INDEPENDENTLY rather than attributed to a single winner:
  // several can be true for one entry, and which ones co-occur is the signal. A
  // visually static scene showing pos=~200 means the position hash itself is
  // churning (renamed dynamic VBs, or a hash rule that is not content-stable),
  // which is a different fix from bone=~200 or pend=~200.
  struct GeoChurnStats {
    uint32_t frame = UINT32_MAX;
    uint32_t entries = 0;       // processGeometryInfo calls this frame
    uint32_t build = 0;         // KBuildBVH  (new entry, or index topology changed)
    uint32_t updBvh = 0;        // kUpdateBVH (re-cache of an existing entry)
    uint32_t updInst = 0;       // kUpdateInstance (free — the state we want)
    uint32_t isNewCnt = 0;      // of `build`, how many were genuinely new entries
    // Why a re-cache was chosen. Independent; may overlap.
    uint32_t rIdx = 0;          // index hash differs from the cached geometry
    uint32_t rPos = 0;          // vertex position hash differs
    uint32_t rVs = 0;           // vertex shader hash differs
    uint32_t rBone = 0;         // bone hash differs (skinned only)
    uint32_t rPend = 0;         // forced by the pendingSrcBake recovery (FIX B)
    uint32_t rNorm = 0;         // forced by a smooth-normals state flip
    // Cost proxy: which caching path a re-cache took.
    uint32_t slowPath = 0;      // interleaveGeometry compute dispatch
    uint32_t fastPath = 0;      // straight copyBuffer of an already-interleaved VB
    uint64_t vtxRecached = 0;   // vertices pushed through a re-cache
    uint64_t vtxTotal = 0;      // vertices seen, re-cached or not
    struct VsStat { uint32_t entries = 0, recache = 0; uint64_t vtx = 0; };
    std::unordered_map<uint64_t, VsStat> byVs;
  };
  static dxvk::mutex s_geoChurnMu;
  static GeoChurnStats s_geoChurn;

  // NV-DXVK [Perf.GeoSplit] — THE_OPTIMISATION_PLAN_2 post-2b Step 0. `[ProcDCS]
  // geomMs` is one number covering four unrelated mechanisms, and which of them
  // dominates decides whether any of processGeometryInfo can follow the decision
  // onto the workers. Plan rule R9: split a timer by MECHANISM before acting on
  // it. The four:
  //
  //   alloc  m_device->createBuffer inside the bake switch. Takes the device
  //          allocator lock, which is already thread-safe — MOVABLE to the
  //          shard workers if this is where the time is.
  //   rec    the barrier + cacheIndexDataOnGPU/cacheVertexDataOnGPU copies and
  //          the interleave dispatch. Records through ctx and consumes the
  //          gpuCapture rebind's outputs — CANNOT move (plan Sec 0.1).
  //          Derived: bake - alloc.
  //   tape   updateBufferCache -> the bindless slot bind and retire. MOVABLE to
  //          the ordered tail (single-threaded there, so no lock needed and
  //          indices stay deterministic). Named `tape` for the append-only table
  //          it used to walk; slots are stable now, so the common case is nine
  //          slice compares rather than nine appends.
  //   rest   geom - bake - tape: the decision, the probes, the flag bookkeeping.
  //
  // MEASUREMENT FLOOR: geom is ~4 us/draw (5-6 ms over ~1350 draws), so these
  // buckets land near ~1 us each against a 41 ns steady_clock read — ~25x the
  // read, resolvable. That is the opposite of the [markFm] case, where the real
  // per-draw work was sub-us and NO per-draw timer could ever resolve it. The
  // ~8 extra reads/draw are still ~0.45 ms/frame, so this is gated OFF by
  // default: it rides rtx.logPrepSceneSplit, the same switch that turns on
  // [Perf.GeoChurn]'s COUNTS — the two are meant to be read together (times
  // tell you what to move, counts tell you how often the path is even taken).
  //
  // TO READ THEM TOGETHER YOU NEED BOTH OF THESE IN rtx.conf:
  //   rtx.logPrepSceneSplit = True
  //   rtx.logDenyTags = -[Perf.GeoChurn]
  // because [Perf.GeoChurn] is on the logger's DEFAULT denylist (log.cpp:626,
  // "counter, not a timing source") and would otherwise emit nothing while
  // looking like the code never ran — the exact failure that denylist comment
  // warns about. [Perf.GeoSplit] itself is not denied.
  static thread_local int64_t  s_geoAllocNs = 0, s_geoBakeNs = 0, s_geoTapeNs = 0;
  static thread_local uint64_t s_geoAllocN  = 0, s_geoBakeN  = 0, s_geoTapeN  = 0;

  // Scoped accumulator. Constructed with the gate already evaluated, so a
  // gated-off zone is one predictable branch and no clock read at all.
  struct GeoSplitZone {
    std::chrono::steady_clock::time_point t0;
    int64_t* acc;
    uint64_t* cnt;
    GeoSplitZone(bool on, int64_t* a, uint64_t* c)
      : acc(on ? a : nullptr), cnt(c) {
      if (acc != nullptr) {
        t0 = std::chrono::steady_clock::now();
      }
    }
    ~GeoSplitZone() {
      if (acc == nullptr) {
        return;
      }
      *acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t0).count();
      ++*cnt;
    }
  };

  // NV-DXVK [Phase2b]: the cache-state DECISION, extracted so the flush-side
  // shard task (which cannot record GPU work) and processGeometryInfo (the CS
  // record step) share ONE implementation and cannot disagree — both inputs are
  // stable between the two call sites by the drain contract. Includes every
  // override that feeds the final result: the hash compares, the unobservable
  // GPU-bone-base rule, the pendingSrcBake recovery, and the smooth-normals
  // promotion. Does NOT include the invalid-input checks (those keep their
  // ONCE-logged sites in processGeometryInfo; the flush side tests the same two
  // conditions itself and routes such draws to the legacy path).
  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::computeGeometryCacheState(const DrawCallState& drawCallState, const RaytraceGeometry& inOutGeometry, bool* outForcedByPendingSrcBake) {
    const RasterGeometry& input = drawCallState.getGeometryData();
    ObjectCacheState result = ObjectCacheState::KBuildBVH;
    if (outForcedByPendingSrcBake != nullptr) {
      *outForcedByPendingSrcBake = false;
    }

    if (!isNew) {
      // This is a geometry we've seen before, that requires updating
      //  'inOutGeometry' has valid historical data
      if (input.hashes[HashComponents::Indices] == inOutGeometry.hashes[HashComponents::Indices]) {
        // Check if the vertex positions have changed, requiring a BVH refit
        if (input.hashes[HashComponents::VertexPosition] == inOutGeometry.hashes[HashComponents::VertexPosition]
         && input.hashes[HashComponents::VertexShader] == inOutGeometry.hashes[HashComponents::VertexShader]
         && drawCallState.getSkinningState().boneHash == inOutGeometry.lastBoneHash) {
          result = ObjectCacheState::kUpdateInstance;
        } else {
          result = ObjectCacheState::kUpdateBVH;
        }
      }
    }

    // [BoneWindow fix, part 2]: an unobservable GPU-resident bone base must be
    // treated as dirty — see the full rationale at the (former) inline site.
    if (!isNew
        && result == ObjectCacheState::kUpdateInstance
        && input.boneBaseBuffer.defined()) {
      result = ObjectCacheState::kUpdateBVH;
    }

    // [s2s mangle/black FIX B]: a prior bake caught the source mid-upload —
    // force a re-cache until a bake lands with the source ready. The out-flag
    // marks exactly this override for the recovery-termination logic in
    // processGeometryInfo (single-retry gate at the pendingSrcBake clear).
    if (!isNew && inOutGeometry.pendingSrcBake && result == ObjectCacheState::kUpdateInstance) {
      result = ObjectCacheState::kUpdateBVH;
      if (outForcedByPendingSrcBake != nullptr) {
        *outForcedByPendingSrcBake = true;
      }
    }

    // Smooth-normals state flip (added or removed) promotes to kUpdateBVH so the
    // vertex data is re-interleaved and the dispatch runs (or originals return).
    if (drawCallState.shouldGenerateSmoothNormals() != inOutGeometry.smoothNormalsApplied
        && result == ObjectCacheState::kUpdateInstance) {
      result = ObjectCacheState::kUpdateBVH;
    }

    return result;
  }

  // NV-DXVK [Phase2b]: explicit instantiations so the flush-side driver (defined
  // later in this file) and any future caller link against both variants.
  template SceneManager::ObjectCacheState SceneManager::computeGeometryCacheState<true>(const DrawCallState&, const RaytraceGeometry&, bool*);
  template SceneManager::ObjectCacheState SceneManager::computeGeometryCacheState<false>(const DrawCallState&, const RaytraceGeometry&, bool*);

  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, RaytraceGeometry& inOutGeometry) {
    ScopedCpuProfileZone();
    const RasterGeometry& input = drawCallState.getGeometryData();

    // NV-DXVK [Perf.GeoSplit]: read the gate ONCE per call — see the block above
    // the GeoSplitZone declaration for what the buckets mean and why the floor
    // arithmetic works out here but did not for [markFm].
    const bool geoSplitOn = RtxOptions::logPrepSceneSplit();

    // NV-DXVK [Phase2b]: the decision now lives in computeGeometryCacheState —
    // ONE implementation for this (CS record) site and the flush-side shard.
    // The overrides that used to be inline below (bone-base, pendingSrcBake,
    // smooth-normals promotion) are all inside it; the blocks that follow keep
    // only their probes and side effects.
    bool forcedByPendingSrcBake = false;
    ObjectCacheState result = computeGeometryCacheState<isNew>(drawCallState, inOutGeometry, &forcedByPendingSrcBake);
    // (The [BoneWindow fix, part 2] unobservable-bone-base override — full
    // rationale preserved in git history — now lives in computeGeometryCacheState.)

    // NV-DXVK [flicker V8]: a source rebound onto a SubmitDraw-ordered capture
    // (commitGeometryToRT rebind) is stable by construction — the capture copy
    // was recorded at the draw's own position in the CS stream, after this
    // frame's engine upload and before the next frame's. It must NOT count as
    // pending: the stash's own in-flight transfer write would read as
    // isPendingGpuWrite() every frame and re-arm the FIX B recovery forever.
    const bool srcCaptured = input.sourceIsGpuCapture;
    // NV-DXVK [s2s mangle/black FIX B]: is a SOURCE buffer's engine upload still
    // in flight right now? (index→collapse/black, position→explode/mangle if read now.)
    const bool srcPending = !srcCaptured
                         && ((input.indexBuffer.defined() && input.indexBuffer.isPendingGpuWrite())
                          || (input.positionBuffer.defined() && input.positionBuffer.isPendingGpuWrite()));
    // If a PRIOR bake of this geometry caught the source mid-upload, the cached
    // BLAS input is zero/garbage and kUpdateInstance would freeze it forever.
    // Force a re-cache (kUpdateBVH) every frame until a bake lands with the source
    // ready (pendingSrcBake clears below). Fix A's barrier alone was insufficient
    // (the upload is cross-queue / recorded AFTER our read in submission order),
    // so re-caching once the source is genuinely ready is the robust fix.
    //
    // NV-DXVK [Perf.GeoChurn follow-up]: this recovery never terminated. Its exit
    // condition is "a bake lands with srcPending == false", but srcPending is
    // GeometryBuffer::isPendingGpuWrite() -> DxvkResource::isInUse(Write), which
    // is a whole-buffer refcount incremented the moment ANY command touching that
    // buffer is RECORDED and only released when that command list completes on the
    // GPU. For a buffer the engine re-uploads every frame it is therefore true
    // essentially every time we sample it, so the exit condition is unreachable by
    // construction and every affected geometry re-baked forever. Measured:
    // pend == updBvh == 140 of 352 entries on every sampled frame, re-interleaving
    // 9.2M of the scene's 9.75M vertices per frame (8.4M of it the capital-ship
    // hulls, vs 0x29566a60). That is the bulk of the ~18 ms scene rebuild.
    //
    // `forcedByPendingSrcBake` marks the bakes that exist ONLY to recover from a
    // prior racy bake, so the clear below can be made to actually terminate.
    // NV-DXVK [Phase2b]: the override itself (and the flag derivation) moved
    // into computeGeometryCacheState — `forcedByPendingSrcBake` above carries
    // its out-param, so the recovery-termination logic below is unchanged.

    // NV-DXVK [ReskinProbe]: boneHash is what decides whether a skinned mesh
    // RE-SKINS this frame (kUpdateBVH) or is left as-is (kUpdateInstance). Its
    // basis changed (now hashed from the source bone bytes rather than the
    // converted Matrix4 palette), so if skinned geometry is stale/broken the
    // fault shows here as a hash that stops changing while the mesh animates.
    //
    // Per-VS, per-window aggregate: how many draws re-skinned vs were skipped,
    // and how often the bone hash actually changed. A skinned VS with
    // hashChanged=0 over a window while the model is visibly moving IS the bug.
    //
    // NV-DXVK [Perf] 2026-08-08: rtx.logReskinProbe leads the conjunction so the
    // ACCUMULATION is gated, not just the emit. The mutex + unordered_map below
    // run on every skinned draw (~1,300/frame) to feed a line printed once per 10
    // frames -- throttling the emit never touched the cost.
    if (RtxOptions::logReskinProbe()
        && !isNew && drawCallState.getSkinningState().numBones > 0) {
      const XXH64_hash_t bh = drawCallState.getSkinningState().boneHash;
      const bool hashChanged = (bh != inOutGeometry.lastBoneHash);
      // NV-DXVK: key on transformData.vertexShaderHash, NOT
      // hashes[HashComponents::VertexShader]. The latter is 0 for these draws,
      // which is why every line this probe ever printed said vs=0x0 — the
      // aggregates were real but they were all pooled under one bogus key, so
      // "which VS stopped re-skinning" was unanswerable, which is the only
      // question the probe exists to answer.
      const uint64_t vsH = static_cast<uint64_t>(
        drawCallState.getTransformData().vertexShaderHash);
      struct ReskinStat { uint64_t draws, changed, reskinned, zeroHash; };
      static dxvk::mutex sReskinMu;
      static std::unordered_map<uint64_t, ReskinStat> sReskin;
      // NV-DXVK: report every 10 frames.
      //
      // The old gate was 120 frames. Under RTX injection TF2 runs at ~2-3 fps
      // (pinned independently in the same log: [IdxStashPool]'s fid%30==15 gate
      // fired exactly 4 times across 118 s), so 120 frames was ~48 s per line.
      // 10 frames is a few seconds — the cadence you actually want to watch a
      // skinned mesh at.
      //
      // The `last` seed is the other half of the old bug: it was 0 while fid was
      // already large, so the very first call satisfied `fid - last >= N` and
      // dumped a one-draw window (the earlier log had a single
      // `draws=1 boneHashChanged=1` line for a whole session). Seed it to the
      // first fid seen so the opening window is a real 10-frame window.
      constexpr uint32_t kReskinWindowFrames = 10;
      const uint32_t fid = m_device->getCurrentFrameId();
      static uint32_t sReskinLastFrame = UINT32_MAX;
      std::lock_guard<dxvk::mutex> lk(sReskinMu);
      if (sReskinLastFrame == UINT32_MAX) sReskinLastFrame = fid;
      ReskinStat& st = sReskin[vsH];
      ++st.draws;
      if (hashChanged) ++st.changed;
      if (result != ObjectCacheState::kUpdateInstance) ++st.reskinned;
      if (bh == 0) ++st.zeroHash;
      if (fid - sReskinLastFrame >= kReskinWindowFrames) {
        sReskinLastFrame = fid;
        for (const auto& kv : sReskin) {
          // changed=0 over a full window on a VS that is visibly animating IS
          // the bug: its boneHash window no longer covers the bones it uses, so
          // processGeometryInfo keeps choosing kUpdateInstance and the mesh
          // renders from a stale skin. Called out inline so it cannot be read
          // past.
          const bool frozen = (kv.second.changed == 0 && kv.second.draws > 8);
          Logger::warn(str::format(
            "[ReskinProbe] vs=0x", std::hex, kv.first, std::dec,
            " draws=", kv.second.draws,
            " boneHashChanged=", kv.second.changed,
            " reskinned=", kv.second.reskinned,
            " zeroBoneHash=", kv.second.zeroHash,
            (frozen ? "  <- FROZEN: hash never changed this window" : "")));
        }
        sReskin.clear();
      }
    }

    // NV-DXVK [ZigGeoState]: confirm the gun's skinned-output staleness ([ZigVB])
    // is driven by tick-rate bones. processGeometryInfo runs every frame for the
    // gun (unlike the dead dispatchSkinning legacy path). Re-skin (kUpdateBVH)
    // only happens when boneHash changes (see decision above), so logging
    // boneHashChanged per frame for the skinned viewmodel directly shows the
    // bone update cadence. Identify the gun by its (recurring) vertexCount.
    //   boneHashChanged every ~3rd frame -> bones are tick-rate stale (root)
    //   result=kUpdateInstance the other frames -> confirms no re-skin (stale verts)
    if (!isNew && drawCallState.getSkinningState().numBones > 0) {
      static uint32_t s_zgLines = 0;
      if (s_zgLines < 1500) {
        ++s_zgLines;
        const uint32_t boneChanged =
          (drawCallState.getSkinningState().boneHash != inOutGeometry.lastBoneHash) ? 1u : 0u;
        Logger::info(str::format(
          "[ZigGeoState] f=", m_device->getCurrentFrameId(),
          " camType=", static_cast<int>(drawCallState.cameraType),
          " verts=", input.vertexCount,
          " result=", static_cast<int>(result),
          " boneChanged=", boneChanged,
          " boneHash=0x", std::hex, drawCallState.getSkinningState().boneHash, std::dec));
      }
    }

    // Copy the input directly to the output as a starting point for our modified geometry data
    RaytraceGeometry output = inOutGeometry;

    output.lastBoneHash = drawCallState.getSkinningState().boneHash;

    // Update draw parameters
    output.cullMode = input.cullMode;
    output.frontFace = input.frontFace;

    // Copy the hashes over
    output.hashes = input.hashes;

    if (!input.positionBuffer.defined()) {
      ONCE(Logger::err("processGeometryInfo: no position data on input detected"));
      return ObjectCacheState::kInvalid;
    }

    if (input.vertexCount == 0) {
      ONCE(Logger::err("processGeometryInfo: input data is violating some assumptions"));
      return ObjectCacheState::kInvalid;
    }

    // Set to 1 if inspection of the GeometryData structures contents on CPU is desired
    #define DEBUG_GEOMETRY_MEMORY 0
    constexpr VkMemoryPropertyFlags memoryProperty = DEBUG_GEOMETRY_MEMORY ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Assume we won't need this, and update the value if required
    output.previousPositionBuffer = RaytraceBuffer();

    // Force the interleaved vertex layout to include space for normals when the smooth normals
    // compute pass is going to run. See DrawCallState::shouldGenerateSmoothNormals() for the rule.
    const bool needsSmoothNormals = drawCallState.shouldGenerateSmoothNormals();
    const bool forceNormals = needsSmoothNormals && !input.normalBuffer.defined();

    // NV-DXVK [Phase2b]: the smooth-normals-flip promotion moved into
    // computeGeometryCacheState (it reads the PRE-copy flag there, which is the
    // same value `output` holds at this point). Only the flag clear stays here —
    // it is an output mutation, not a decision.
    if (!needsSmoothNormals) {
      output.smoothNormalsApplied = false;
    }

    // If forceNormals is true, we can't use the fast "already interleaved" path since
    // we need to change the layout to include normal space.
    // NV-DXVK: !input.vguiLayoutEnable mirrors the gate in
    // RtxGeometryUtils::cacheVertexDataOnGPU. VGUI surfaces append 8 extra
    // floats per vertex so the BLAS storage must be sized at
    // computeOptimalVertexStride (which honours vguiLayoutEnable), not the
    // source VB stride. Without this the buffer alloc here uses the
    // source stride while the interleave dispatch writes at the larger
    // stride → assertion fires at the cacheIndexDataOnGPU /
    // historyBuffer[0]->info().size check downstream.
    const size_t vertexStride = (input.isVertexDataInterleaved() && input.areFormatsGpuFriendly()
                                && !forceNormals && !input.vguiLayoutEnable)
      ? input.positionBuffer.stride()
      : RtxGeometryUtils::computeOptimalVertexStride(input, forceNormals);

    // NV-DXVK [Perf.GeoChurn] tally — see GeoChurnStats above. This is the right
    // place for it: `result` is final here (both the pendingSrcBake and the
    // smooth-normals overrides have already run), while `inOutGeometry` still
    // holds LAST frame's cached state — everything below writes to `output`,
    // the copy — so the per-reason comparisons are genuinely
    // this-frame-input vs cached-geometry. Gated on the same option that gates
    // [Perf.PrepScene]/[Perf.Merge] so it costs one bool test when not reading
    // the split, and so all three lines describe the same frames.
    if (RtxOptions::logPrepSceneSplit()) {
      const bool recache = (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH);
      // Mirrors the fast/slow branch in RtxGeometryUtils::cacheVertexDataOnGPU:
      // the slow branch is the one that dispatches interleaveGeometry.
      const bool tookFast = input.isVertexDataInterleaved() && input.areFormatsGpuFriendly()
                         && !forceNormals && !input.vguiLayoutEnable;
      const uint64_t vsH = static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash);
      const uint32_t fid = m_device->getCurrentFrameId();

      std::lock_guard<dxvk::mutex> lk(s_geoChurnMu);
      if (s_geoChurn.frame != fid) {
        s_geoChurn = GeoChurnStats();
        s_geoChurn.frame = fid;
      }
      GeoChurnStats& g = s_geoChurn;

      ++g.entries;
      g.vtxTotal += input.vertexCount;
      if (isNew) {
        ++g.isNewCnt;
      }
      switch (result) {
        case ObjectCacheState::KBuildBVH:       ++g.build;   break;
        case ObjectCacheState::kUpdateBVH:      ++g.updBvh;  break;
        case ObjectCacheState::kUpdateInstance: ++g.updInst; break;
        default: break;
      }
      if (!isNew) {
        // Same four comparisons the decision above makes, split apart so the
        // one that is actually firing is named instead of inferred.
        if (input.hashes[HashComponents::Indices] != inOutGeometry.hashes[HashComponents::Indices]) {
          ++g.rIdx;
        }
        if (input.hashes[HashComponents::VertexPosition] != inOutGeometry.hashes[HashComponents::VertexPosition]) {
          ++g.rPos;
        }
        if (input.hashes[HashComponents::VertexShader] != inOutGeometry.hashes[HashComponents::VertexShader]) {
          ++g.rVs;
        }
        if (drawCallState.getSkinningState().boneHash != inOutGeometry.lastBoneHash) {
          ++g.rBone;
        }
        if (inOutGeometry.pendingSrcBake) {
          ++g.rPend;
        }
      }
      // Read the PRE-copy flag: `output.smoothNormalsApplied` may already have
      // been cleared a few lines up, which would hide the flip.
      if (needsSmoothNormals != inOutGeometry.smoothNormalsApplied) {
        ++g.rNorm;
      }
      if (recache) {
        g.vtxRecached += input.vertexCount;
        if (tookFast) {
          ++g.fastPath;
        } else {
          ++g.slowPath;
        }
      }
      GeoChurnStats::VsStat& vs = g.byVs[vsH];
      ++vs.entries;
      if (recache) {
        ++vs.recache;
        vs.vtx += input.vertexCount;
      }
    }

    // NV-DXVK [s2s mangle/black FIX A — producer→consumer ordering]:
    // If a source geometry buffer still has an in-flight GPU write (the engine's
    // upload of this mesh hasn't completed — proven by [TrimCache] srcPosPend=1 /
    // srcIdxPend=1 on EVERY cache-bake frame of the s2s trims), the caching below
    // races it: cacheIndexDataOnGPU's blind copyBuffer (device-local source) reads
    // not-yet-uploaded indices → caches ZERO (degenerate tris → black), and
    // cacheVertexDataOnGPU's interleave dispatch reads not-yet-uploaded positions
    // → caches GARBAGE (exploded tris → mangle). The bad cache is then frozen by
    // the steady kUpdateInstance frames for the rest of the scene.
    // copyBuffer's own hazard check (isBufferDirty against m_execBarriers) MISSES
    // this because the upload write was already flushed out of the current barrier
    // set into an earlier command buffer, so no write→read barrier is inserted.
    // Force the ordering here with a pipeline barrier — same-queue submission order
    // makes this a correct dependency against the earlier-recorded upload write.
    // General fix (not trim-specific) — gated on a real pending write so ready
    // geometry (the overwhelmingly common case) pays nothing.
    // NV-DXVK [flicker V8]: also barrier when the source is a capture — the
    // capture's transfer write was recorded at the draw's position, i.e. in an
    // earlier command buffer than this bake, so copyBuffer/dispatch hazard
    // tracking (scoped to the current barrier set) cannot see it either.
    // NV-DXVK [Perf.GeoSplit]: `bake` opens HERE, at the ordering barrier, and
    // closes after the switch — barrier + copies + interleave dispatch + the
    // allocations, i.e. everything this function records or allocates. `rec`
    // (the part that cannot leave dxvk-cs) is bake minus alloc.
    const auto tGsBake0 = geoSplitOn
      ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};

    if ((result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)
        && (srcPending || srcCaptured)) {
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    }

    switch (result) {
      case ObjectCacheState::KBuildBVH: {
        // NV-DXVK: throttle — was firing per BLAS build (~150/sec during
        // asset-heavy loading), a major contributor to the loading-screen
        // stall. TDR correlation only really needs to know the build
        // happened; sample every 64th for a coarse trace without flooding.
        static uint64_t sBvhBuildLog = 0;
        if ((sBvhBuildLog++ & 0x3F) == 0) {
          Logger::info(str::format("[BVH-BUILD]",
            " hasIdxSnap=", (input.indexDataSnapshot ? 1 : 0),
            " snapBytes=", (input.indexDataSnapshot ? input.indexDataSnapshot->size() : 0),
            " idxCount=", input.indexCount,
            " vtxCount=", input.vertexCount,
            " idxBuf=0x", std::hex,
            (uintptr_t)(input.indexBuffer.defined() && input.indexBuffer.buffer() != nullptr
              ? input.indexBuffer.buffer().ptr() : nullptr),
            " idxMF=0x",
            (input.indexBuffer.defined() && input.indexBuffer.buffer() != nullptr
              ? input.indexBuffer.buffer()->memFlags() : 0), std::dec));
        }
        // Set up the ideal vertex params, if input vertices are interleaved, it's safe to assume the positionBuffer stride is the vertex stride
        output.vertexCount = input.vertexCount;

        const size_t vertexBufferSize = output.vertexCount * vertexStride;

        // Set up the ideal index params
        output.indexCount = input.isTopologyRaytraceReady() ? input.indexCount : RtxGeometryUtils::getOptimalTriangleListSize(input);
        const VkIndexType indexBufferType = input.isTopologyRaytraceReady() ? input.indexBuffer.indexType() : RtxGeometryUtils::getOptimalIndexFormat(output.vertexCount);
        const size_t indexStride = (indexBufferType == VK_INDEX_TYPE_UINT16) ? 2 : 4;

        // Make sure we're not stomping something else...
        assert(output.indexCacheBuffer == nullptr && output.historyBuffer[0] == nullptr);

        // Create a index buffer and vertex buffer we can use for raytracing.
        DxvkBufferCreateInfo info;
        info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

        info.size = align(output.indexCount * indexStride, CACHE_LINE_SIZE);
        {
          GeoSplitZone gsz(geoSplitOn, &s_geoAllocNs, &s_geoAllocN);   // NV-DXVK [Perf.GeoSplit]
          output.indexCacheBuffer = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Index Cache Buffer");
        }

        if (!RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output)) {
          ONCE(Logger::err("processGeometryInfo: failed to cache index data on GPU"));
          return ObjectCacheState::kInvalid;
        }

        output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);

        info.size = align(vertexBufferSize, CACHE_LINE_SIZE);
        {
          GeoSplitZone gsz(geoSplitOn, &s_geoAllocNs, &s_geoAllocN);   // NV-DXVK [Perf.GeoSplit]
          output.historyBuffer[0] = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
        }

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        break;
      }
      case ObjectCacheState::kUpdateBVH: {
        // TDR-DIAG: log every UPDATE to see if stale index caches are the issue.
        Logger::info(str::format("[BVH-UPDATE]",
          " hasIdxSnap=", (input.indexDataSnapshot ? 1 : 0),
          " idxCount=", input.indexCount,
          " vtxCount=", input.vertexCount,
          " idxBuf=0x", std::hex,
          (uintptr_t)(input.indexBuffer.defined() && input.indexBuffer.buffer() != nullptr
            ? input.indexBuffer.buffer().ptr() : nullptr), std::dec));
        bool invalidateHistory = false;

        // Stride changed, so we must recreate the previous buffer and use identical data
        if (output.historyBuffer[0]->info().size != align(vertexStride * input.vertexCount, CACHE_LINE_SIZE)) {
          auto desc = output.historyBuffer[0]->info();
          desc.size = align(vertexStride * input.vertexCount, CACHE_LINE_SIZE);
          {
            GeoSplitZone gsz(geoSplitOn, &s_geoAllocNs, &s_geoAllocN);   // NV-DXVK [Perf.GeoSplit]
            output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
          }

          // Invalidate the current buffer
          output.historyBuffer[1] = nullptr;

          // Mark this object for realignment
          invalidateHistory = true;
        }

        // Use the previous updates vertex data for previous position lookup
        std::swap(output.historyBuffer[0], output.historyBuffer[1]);

        if (output.historyBuffer[0].ptr() == nullptr) {
          // First frame this object has been dynamic need to allocate a 2nd frame of data to preserve history.
          GeoSplitZone gsz(geoSplitOn, &s_geoAllocNs, &s_geoAllocN);   // NV-DXVK [Perf.GeoSplit]
          output.historyBuffer[0] = m_device->createBuffer(output.historyBuffer[1]->info(), memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
        }

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        // NV-DXVK [s2s mangle/black FIX B]: the normal kUpdateBVH path only
        // refreshes the index cache from a CPU snapshot (DYNAMIC buffers) below;
        // device-local indices (the s2s trims) are NOT re-copied here. When we're
        // re-caching to RECOVER from a bad bake (prior bake caught the source
        // mid-upload), re-copy the index from source too, else the zeroed index
        // cache (collapse → black) would never be fixed.
        if (inOutGeometry.pendingSrcBake) {
          RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output);
        }

        // NV-DXVK [perf, GPU index stash]: same refresh, GPU-side. The stash
        // holds this draw's exact index range (copied in-stream at
        // commitGeometryToRT), so refresh the cache with a GPU->GPU copy —
        // no CPU scan, no writeToBuffer. Barrier per the FIX A rationale
        // (stash write lives in an earlier command buffer).
        if (input.indexDataGpuStash != nullptr
            && input.indexDataGpuStash->buffer != nullptr
            && output.indexCacheBuffer != nullptr) {
          const VkDeviceSize stashLen =
            VkDeviceSize(input.indexCount) * input.indexBuffer.stride();
          if (stashLen > 0
              && stashLen <= input.indexDataGpuStash->size
              && stashLen <= input.indexDataGpuStash->buffer->info().size
              && stashLen <= output.indexCacheBuffer->info().size) {
            ctx->emitMemoryBarrier(0,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            ctx->copyBuffer(output.indexCacheBuffer, 0,
                            input.indexDataGpuStash->buffer, 0, stashLen);
          }
        }
        // NV-DXVK: Refresh the index cache from our CPU snapshot if the
        // source D3D11 buffer was DYNAMIC. The update path assumes identical
        // index hashes imply identical content, but two different dynamic
        // buffers can hash the same (collision or reused content). Safer to
        // re-upload from the per-draw snapshot we captured at SubmitDraw.
        if (input.indexDataSnapshot && !input.indexDataSnapshot->empty()
            && output.indexCacheBuffer != nullptr
            && input.indexDataSnapshot->size() <= output.indexCacheBuffer->info().size) {
          // TDR-DIAG: OOB index scan (same as cacheIndexDataOnGPU).
          const auto& data = *input.indexDataSnapshot;
          const uint32_t stride = input.indexBuffer.stride();
          const uint32_t vtxCount = output.vertexCount;  // The cached vtx count
          uint32_t maxIdx = 0, oobCount = 0, oobFirstIdx = 0, oobFirstVal = 0;
          if (stride == 2) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data.data());
            const size_t n = data.size() / 2;
            for (size_t i = 0; i < n; ++i) {
              if (p[i] > maxIdx) maxIdx = p[i];
              if (p[i] >= vtxCount) {
                if (oobCount == 0) { oobFirstIdx = uint32_t(i); oobFirstVal = p[i]; }
                ++oobCount;
              }
            }
          } else if (stride == 4) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(data.data());
            const size_t n = data.size() / 4;
            for (size_t i = 0; i < n; ++i) {
              if (p[i] > maxIdx) maxIdx = p[i];
              if (p[i] >= vtxCount) {
                if (oobCount == 0) { oobFirstIdx = uint32_t(i); oobFirstVal = p[i]; }
                ++oobCount;
              }
            }
          }
          if (oobCount > 0) {
            Logger::err(str::format("[IDX-OOB-UPD] vtxCount=", vtxCount,
              " maxSeenIdx=", maxIdx, " oobCount=", oobCount,
              " firstOOB[", oobFirstIdx, "]=", oobFirstVal,
              " idxCount=", input.indexCount, " stride=", stride));
          }
          ctx->writeToBuffer(output.indexCacheBuffer, 0,
                             input.indexDataSnapshot->size(),
                             input.indexDataSnapshot->data());
        }

        // Sometimes, we need to invalidate history, do that here by copying the current buffer to the previous..
        if (invalidateHistory) {
          ctx->copyBuffer(output.historyBuffer[1], 0, output.historyBuffer[0], 0, output.historyBuffer[1]->info().size);
        }

        // Assign the previous buffer using the last slice (copy most params from the position, just change buffer)
        output.previousPositionBuffer = RaytraceBuffer(DxvkBufferSlice(output.historyBuffer[1], 0, output.positionBuffer.length()), output.positionBuffer.offsetFromSlice(), output.positionBuffer.stride(), output.positionBuffer.vertexFormat());
        break;
      }
      default:
        break;
    }

    // NV-DXVK [Perf.GeoSplit]: close `bake`. Counted on EVERY call, including
    // kUpdateInstance (where the switch is a no-op) — that is deliberate: the
    // per-call floor of an empty bake is exactly the baseline the other two
    // buckets have to be read against.
    if (geoSplitOn) {
      s_geoBakeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - tGsBake0).count();
      ++s_geoBakeN;
    }

    // NV-DXVK [s2s mangle/black FIX B]: record whether THIS bake read a source
    // whose engine upload was still in flight. If so, the force-recache above
    // keeps re-caching next frame; once a bake lands with srcPending=false the
    // data is good and this clears, settling back to kUpdateInstance. Only the
    // caching states (re)write the cache; kUpdateInstance carries the clear flag.
    if (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH) {
      // NV-DXVK [flicker V8 follow-up: capture feedback]: a bake that read a
      // DEVICE_LOCAL source WITHOUT a SubmitDraw-ordered capture may have
      // raced the engine's upload — and srcPending (isPendingGpuWrite) cannot
      // tell (whole-buffer refcount, see above). So for those bakes: latch the
      // recovery unconditionally AND tell the game thread to capture this
      // draw's next submits (feedback ring); the recovery then terminates on
      // the first bake whose source was a capture — proof by construction,
      // not a timing guess. Skinned draws keep the legacy path: they re-bake
      // per boneHash anyway and are excluded from capture. Non-device-local
      // sources also keep the legacy srcPending behavior.
      const bool deviceLocalSrc =
        (input.positionBuffer.defined() && input.positionBuffer.mapPtr() == nullptr)
        || (input.indexBuffer.defined() && input.indexBuffer.mapPtr() == nullptr);
      const bool skinnedDraw = drawCallState.getSkinningState().numBones > 0;
      if (srcCaptured) {
        // Captured bake: stable by construction. Recovery (if any) ends here.
        output.pendingSrcBake = false;
        output.pendingSrcBakeAttempts = 0;
      } else if (deviceLocalSrc && !skinnedDraw && input.captureEligible
                 && RtxOptions::captureSourceGeometry()) {
        constexpr uint8_t kMaxUncapturedRecoveryAttempts = 8;
        if (output.pendingSrcBakeAttempts >= kMaxUncapturedRecoveryAttempts) {
          // Feedback isn't delivering (pool exhaustion / capture disabled
          // upstream). Freeze on what we have rather than reinstating the
          // unbounded per-frame re-bake storm.
          output.pendingSrcBake = false;
        } else {
          ++output.pendingSrcBakeAttempts;
          output.pendingSrcBake = true;
          const uint64_t vsH = static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash);
          const uint64_t fbKey = RasterGeometry::captureFeedbackKey(
            vsH, input.vertexCount, input.indexCount);
          {
            const uint32_t fidPub = m_device->getCurrentFrameId();
            std::lock_guard<std::mutex> lkFb(tf2::g_geomCaptureWantedMutex);
            auto& wanted = tf2::g_geomCaptureWantedMap;
            // Safety-bound prune only — steady-state cleanup is the
            // consumer's per-frame stale sweep. This fires only if the
            // consumer stopped snapshotting (feature off upstream).
            if (wanted.size() > 4096) {
              for (auto it = wanted.begin(); it != wanted.end();) {
                it = (fidPub > it->second && fidPub - it->second > 6u)
                  ? wanted.erase(it) : std::next(it);
              }
            }
            wanted[fbKey] = fidPub;
          }
          // [GeoCapture.wanted]: name who is starving and in what state, one
          // line per key per ~30 frames. capState is the discriminator:
          //   none    — the dcs carries no capture at all (the game thread
          //             never captured this submit: predictor/feedback miss,
          //             or the dcs reached here without passing SubmitDraw)
          //   invalid — capture attempted, pool acquire failed
          //   UNBOUND — a VALID capture is attached but commitGeometryToRT's
          //             rebind never ran on this dcs: a submit path that
          //             bypasses the commit (that path is the bug to plumb)
          {
            static std::unordered_map<uint64_t, uint32_t> sWantLogLast;
            const uint32_t fidNow = m_device->getCurrentFrameId();
            uint32_t& last = sWantLogLast[fbKey];
            if (last == 0u || fidNow - last >= 30u) {
              last = fidNow;
              const char* capState = "none";
              if (input.gpuCapture != nullptr) {
                capState = input.gpuCapture->valid ? "UNBOUND" : "invalid";
              }
              Logger::info(str::format("[GeoCapture.wanted] f=", fidNow,
                " vs=0x", std::hex, vsH, std::dec,
                " vtx=", input.vertexCount, " idx=", input.indexCount,
                " result=", static_cast<uint32_t>(result),
                " isNew=", isNew ? 1 : 0,
                " attempts=", static_cast<uint32_t>(output.pendingSrcBakeAttempts),
                " capState=", capState,
                " camType=", static_cast<uint32_t>(drawCallState.cameraType)));
            }
          }
        }
      } else if (forcedByPendingSrcBake && RtxOptions::pendingSrcBakeSingleRetry()) {
        // This bake IS the recovery, and it has converged — clear rather than
        // re-latch srcPending (which would re-arm the recovery forever, see above).
        //
        // Why one retry is sufficient and not merely a frame-count guess: the
        // upload we raced was the upload of the content the previous bake was
        // trying to read. A full frame has elapsed since — including the
        // per-frame fence wait (fenceWaitMs 73-94, i.e. we block on the GPU
        // every frame) — so that upload has demonstrably completed before this
        // bake reads the source. srcPending may well still be true here, but if
        // so it is true because of a NEWER upload; and if that newer upload
        // carries different content, the vertex-position hash differs and the
        // ordinary comparison at the top of this function re-caches on its own
        // merit. Either way this geometry cannot be left frozen on stale bytes,
        // which is the failure FIX B exists to prevent.
        output.pendingSrcBake = false;
      } else {
        output.pendingSrcBake = srcPending;
      }
    }

    // NV-DXVK [capture stability contract]: publish this entry's verdict
    // under the game-thread-stamped identity key. kUpdateInstance with no
    // pending recovery re-confirms the key stable (the consumer's freshness
    // window requires this every few frames); ANY bake taints it, because a
    // bake means the next submit of this key may create/refresh an entry
    // whose bake must be captured. Erase-on-taint rather than trusting
    // ordering: within one frame several entries can share a key and the
    // taint must win for the following frame's decision.
    if (RtxOptions::captureSourceGeometry()
        && input.captureEligible && input.captureIdentityKey != 0ull) {
      const uint32_t fidVerdict = m_device->getCurrentFrameId();
      std::lock_guard<std::mutex> lkSt(tf2::g_geomCaptureStableMutex);
      if (result == ObjectCacheState::kUpdateInstance && !output.pendingSrcBake) {
        if (tf2::g_geomCaptureStableMap.size() < 65536) {
          tf2::g_geomCaptureStableMap[input.captureIdentityKey] = fidVerdict;
        }
      } else if (result == ObjectCacheState::KBuildBVH
              || result == ObjectCacheState::kUpdateBVH) {
        if (tf2::g_geomCaptureTaintMap.size() < 65536) {
          tf2::g_geomCaptureTaintMap[input.captureIdentityKey] = fidVerdict;
        }
        tf2::g_geomCaptureStableMap.erase(input.captureIdentityKey);
      }
    }

    // Update color buffer in BVH with DrawCallState
    // The user can disable/enable color buffer for specific materials, so we manually sync the DrawCallState and BVH here to keep the color buffer in BVH updated.
    // Note, we don't setup kUpdateBVH because it's too waste to update all buffers if only the color buffer needs to be updated.
    if (output.color0Buffer.defined() && !drawCallState.geometryData.color0Buffer.defined()) {
      // Remove the color buffer in BVH if the color buffer from drawcall is removed by ignoreBakedLighting
      output.color0Buffer = RaytraceBuffer();
    } else if (!output.color0Buffer.defined() && drawCallState.geometryData.color0Buffer.defined()) {
      // Write the color buffer back to BVH if the color buffer is enabled again
      const DxvkBufferSlice slice = DxvkBufferSlice(output.historyBuffer[0]);
      const auto& colorBuffer = drawCallState.geometryData.color0Buffer;
      output.color0Buffer = RaytraceBuffer(slice, colorBuffer.offsetFromSlice(), colorBuffer.stride(), colorBuffer.vertexFormat());
    }

    // NV-DXVK: TF2 worldspace VGUI — propagate the 3 captured auxiliary
    // structured-buffer SRVs (font/img/styles) from RasterGeometry to
    // RaytraceGeometry so updateBufferCache can register them in the
    // bindless storage-buffer table. The VGUI evaluator reads them at hit
    // time via the resulting bindless indices; no interleaving is needed
    // because they're already shader-friendly StructuredBuffer<>'s.
    auto cloneVguiSb = [](const RasterBuffer& src) -> RaytraceBuffer {
      if (!src.defined()) return RaytraceBuffer();
      return RaytraceBuffer(src, src.offsetFromSlice(), src.stride(), src.vertexFormat());
    };
    output.vguiFontBoundsBuffer = cloneVguiSb(input.vguiFontBoundsBuffer);
    output.vguiImgBoundsBuffer  = cloneVguiSb(input.vguiImgBoundsBuffer);
    output.vguiStylesBuffer     = cloneVguiSb(input.vguiStylesBuffer);

    // Update buffers in the cache
    {
      // NV-DXVK [Perf.GeoSplit]: `tape` — the bindless slot bind and retire.
      // Runs on EVERY call including kUpdateInstance (this site is outside the
      // switch), so it is per-DRAW cost, not per-bake, which is exactly why it
      // is a candidate for the ordered tail.
      //
      // `inOutGeometry` is still the PRE-bake geometry here: everything above
      // writes to `output`, the copy, and the finalise at the end of this
      // function is what overwrites it. That is what lets updateBufferCache see
      // both sides and work out which buffers the geometry let go of.
      GeoSplitZone gsz(geoSplitOn, &s_geoTapeNs, &s_geoTapeN);
      updateBufferCache(inOutGeometry, output);
    }

    // NV-DXVK [TrimCache]: s2s mangle/black race probe — the CACHE-WRITE side.
    // The trim BLAS build input (modifiedGeometryData index buffer + vertexCount)
    // reads corrupt at build time (all-zero indices => collapse => black;
    // garbage vtxCount => slivers) NON-DETERMINISTICALLY, while membership is
    // clean ([TlasMember] Opaque/valid). Static trims should be kUpdateInstance
    // (result=3, NO re-cache) every frame after the first KBuildBVH (result=0).
    // Log thread + cache state + the mg object address (this is the SAME object
    // as blasEntry->modifiedGeometryData) + committed counts + cache buffer ptrs.
    // Cross-ref by mg=/f= against [TlasMember]/[SpikeRB] (the build-CONSUME side):
    //   - different tid, or outVtx here != vtx there same frame => cross-thread
    //     race on the BlasEntry (write/consume overlap);
    //   - matching+sane here but [SpikeRB] maxIdxVal=0 / huge maxEdge there =>
    //     GPU copy didn't land before the build (missing barrier);
    //   - result flips to 0/2 (re-cache) on corrupt frames => realloc window.
    {
      const uint64_t tcVs = static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash);
      if (tcVs == 0x29566a60d473af50ull || tcVs == 0x29a262d2e574b21cull) {
        // ~148 trim sub-draws/frame → throttle: one heartbeat per frame (proves
        // the steady cache state + tid + mg identity) PLUS every anomaly
        // (a re-cache = result != kUpdateInstance, or an out-of-range/zero
        // vertexCount). Cache-time garbage is rare; if [TrimCache] is sane every
        // frame but [TlasMember] shows vtxOOR>0 / a differing vtxCount, the
        // corruption is introduced AFTER this write (race/lifetime, not caching).
        static uint32_t s_tcLastFrame = 0xFFFFFFFFu;
        const uint32_t tcFrame = m_device->getCurrentFrameId();
        const bool tcAnomaly = (result != ObjectCacheState::kUpdateInstance)
                            || (output.vertexCount > 600000u) || (output.vertexCount == 0u);
        const bool tcHeartbeat = (tcFrame != s_tcLastFrame);
        if (tcAnomaly || tcHeartbeat) {
          s_tcLastFrame = tcFrame;
          // SOURCE readiness: if the trim's source index/position buffer still
          // has a pending GPU write when we cache (especially on a KBuild=2 or
          // kUpdateBVH=1 frame), the blind copyBuffer in cacheIndexDataOnGPU is
          // RACING the engine's upload → caches zero/partial → frozen by the
          // following kUpdateInstance frames = the sustained collapse/mangle.
          const bool tcIdxPend = input.indexBuffer.defined() && input.indexBuffer.isPendingGpuWrite();
          const bool tcPosPend = input.positionBuffer.defined() && input.positionBuffer.isPendingGpuWrite();
          Logger::info(str::format(
            "[TrimCache] f=", tcFrame,
            " tid=", std::this_thread::get_id(),
            " vs=0x", std::hex, tcVs, std::dec,
            " result=", static_cast<int>(result),
            (tcAnomaly ? " ANOMALY" : ""),
            " mg=", (const void*) &inOutGeometry,
            " inVtx=", input.vertexCount, " outVtx=", output.vertexCount,
            " inIdx=", input.indexCount, " outIdx=", output.indexCount,
            " srcIdxPend=", (tcIdxPend ? 1 : 0), " srcPosPend=", (tcPosPend ? 1 : 0),
            " idxCacheBuf=", (const void*) (output.indexCacheBuffer != nullptr ? output.indexCacheBuffer.ptr() : nullptr),
            " histBuf0=", (const void*) (output.historyBuffer[0] != nullptr ? output.historyBuffer[0].ptr() : nullptr)));
        }
      }
    }

    // Finalize our modified geometry data to the output
    inOutGeometry = output;

    return result;
  }


  void SceneManager::onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame) {
    ScopedCpuProfileZone();

    manageTextureVram();

    if (m_enqueueDelayedClear || m_pReplacer->checkForChanges(ctx)) {
      clear(ctx, true);
      m_enqueueDelayedClear = false;
    }

    m_cameraManager.onFrameEnd();
    m_instanceManager.onFrameEnd();
    m_previousFrameSceneAvailable = raytracedThisFrame && RtxOptions::enablePreviousTLAS();

    // NV-DXVK [Phase2b]: publish the buffer table's size for the flush-side
    // pre-pass, which reads it (drain-ordered, no race) as the overflow
    // predictor because the live count is written on this thread, after the
    // flush.
    //
    // NV-DXVK [stable buffer identity]: THE CLEAR THAT USED TO BE HERE IS GONE,
    // and that is the whole point of the change. Clearing the table every frame
    // is what made a slot index mean nothing outside the frame that produced it,
    // and it is what a surface that was uploaded without being re-derived then
    // read through. Slots are now released by retire() plus the reclaim sweep in
    // garbageCollection(), on evidence that nothing references them. The only
    // remaining whole-table clear is in SceneManager::clear(), where the scene
    // genuinely goes away.
    m_bufferCacheLastFrameCount = m_bufferCache.getTotalCount();
    m_externalGpuInstancingTransforms.clear();
    if (raytracedThisFrame){
      std::lock_guard lock { m_drawCallMeta.mutex };
      const uint8_t curTick = m_drawCallMeta.ticker;
      const uint8_t nextTick = (m_drawCallMeta.ticker + 1) % m_drawCallMeta.MaxTicks;

      m_drawCallMeta.ready[curTick] = true;

      m_drawCallMeta.infos[nextTick].clear();
      m_drawCallMeta.ready[nextTick] = false;
      m_drawCallMeta.ticker = nextTick;
    }

    m_terrainBaker->onFrameEnd(ctx);

    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onFrameEnd();
    }
    
    m_activePOMCount = 0;
    m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    m_startInMediumMaterialIndex_inCache = UINT32_MAX;

    if (m_uniqueObjectSearchDistance != RtxOptions::uniqueObjectDistance()) {
      m_uniqueObjectSearchDistance = RtxOptions::uniqueObjectDistance();
      m_drawCallCache.rebuildSpatialMaps();
    }

    // NV-DXVK [perf] sec 4c: refresh the binding epoch BEFORE the per-frame caches
    // below are cleared, so a consumer reading it this frame sees the state the
    // indices were actually left in. Summing monotonic counters means any insert,
    // free or clear on either cache moves it, and nothing has to remember to set a
    // flag -- see getBindingEpoch() in the header for why a cross-frame cache of a
    // material index is unsound without this.
    //
    // INSERTS ARE DELIBERATELY EXCLUDED. The first version of this summed insert
    // counts too, and that was wrong in a way the log made obvious: [MatChurn]
    // reports matNew=2..4 and texNew=2 EVERY frame in normal play while matClear,
    // texFree and texClear stay 0, so an insert-sensitive epoch moved every frame
    // and invalidated every cached binding -- tailSkip read 0-4%.
    //
    // An insert cannot invalidate anything. SparseUniqueCache::track either pops a
    // free-list slot or push_back()s; in both cases every existing entry keeps its
    // index. Only free() (which pushes an index back for reuse) and clear() can
    // make a previously-handed-out index refer to something else, and those are
    // exactly the two events a cross-frame binding cache must not miss.
    {
      auto& epochTextureManager = m_device->getCommon()->getTextureManager();
      m_bindingEpoch = m_surfaceMaterialCache.getFreeCount()
                     + m_surfaceMaterialCache.getClearCount()
                     + epochTextureManager.getCacheFreeCount()
                     + epochTextureManager.getCacheClearCount();
    }

    // Not currently safe to cache these across frames (due to texture indices and rtx options potentially changing)
    m_preCreationSurfaceMaterialMap.clear();
    // NV-DXVK [perf]: the createSurfaceMaterial memo answers the same question
    // over the same inputs, so it carries the same lifetime. Cleared here rather
    // than relying on its frameId alone so the two can never disagree.
    invalidateSurfaceMaterialMemo();
    // NV-DXVK [perf]: same constraint, same lifetime. as<OpaqueMaterialData>() reads
    // rtx.legacyMaterial.* and rtx.ignoreAlphaOnTextures, so an entry must not
    // outlive the frame it was built in.
    m_legacyOpaqueConversionCache.clear();

    // NV-DXVK [Perf.MatCache]: hit rate for the conversion cache above. hitPct is
    // 1 - (distinct materials / draws), so a low value means the scene really does
    // draw that many distinct materials -- read it before concluding the cache is
    // broken.
    if (RtxOptions::cacheLegacyMaterialConversionCounts()) {
      static std::chrono::steady_clock::time_point sLastLog = std::chrono::steady_clock::now();
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now - sLastLog).count() >= 3000) {
        const uint64_t total = m_legacyOpaqueConversionHits + m_legacyOpaqueConversionMisses;
        Logger::info(str::format(
          "[Perf.MatCache] calls=", total,
          " hits=", m_legacyOpaqueConversionHits,
          " misses=", m_legacyOpaqueConversionMisses,
          " hitPct=", (total ? (m_legacyOpaqueConversionHits * 100 / total) : 0)));
        m_legacyOpaqueConversionHits = 0;
        m_legacyOpaqueConversionMisses = 0;
        sLastLog = now;
      }
    }

    m_thinOpaqueMaterialExist = false;
    m_sssMaterialExist = false;

    // execute graph updates after all garbage collection is complete (to avoid updating graphs that will just be deleted)
    // RtxOptions will still be pending, so any changes to them will apply next frame.
    if (raytracedThisFrame){
      m_graphManager.update(ctx);
    }

    // Clear replacement material hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameReplacementMaterialHashes();
    
    // Clear mesh hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameMeshHashes();
    
    // Reset the fog state to get it re-discovered on the next frame
    ImGUI::SetFogStates(m_fogStates, m_fog.getHash());
    m_fog = FogState();
    m_fogStates.clear();

    // NV-DXVK TF2 vanish-zone: dump per-frame draw counts + main camera state
    // when the kept-instance count drops by >=50% vs the previous gameplay
    // frame. Only fires on raytracedThisFrame + classifier-set Main, so menus
    // / loading / cinematics don't spam the log.
    {
      VanishDiag& d = vanishDiag();
      const uint32_t fid = m_device->getCurrentFrameId();
      const bool gameplay = raytracedThisFrame && m_cameraManager.isMainSetByClassifier();
      if (gameplay) {
        const uint32_t kept = d.drawsKept;
        // Baseline = running max kept count across gameplay frames. Streaming
        // new geometry in raises the baseline; the vanish-zone is detected
        // when kept drops below 90% of baseline (>=10% deficit). Throttle to
        // one log per frame and require baseline >= 32 so early frames with
        // sparse geometry don't trip the threshold.
        if (kept > d.baselineKept) {
          d.baselineKept = kept;
        }
        const bool deficit = d.baselineKept >= 32 && kept * 10u <= d.baselineKept * 9u;
        // NV-DXVK [Perf] 2026-08-18: GATED on rtx.logGeomDiag. The baseline is a
        // running MAX, so once any peak frame lands, every ordinary frame after
        // it reads as a deficit — measured 7041 lines over 789 frames, i.e. it
        // fires permanently rather than on the vanish event it was built for.
        // The counters above still update (baseline tracking is kept intact);
        // only the per-frame formatting + write is gated.
        if (deficit && d.lastLoggedFid != fid && RtxOptions::logGeomDiag()) {
          d.lastLoggedFid = fid;
          const RtCamera& cam = m_cameraManager.getMainCamera();
          const Vector3 pos = cam.getPosition();
          const auto& latch = m_cameraManager.getMainLatchSnapshot();
          const uint32_t deficitPct = 100u - (kept * 100u) / d.baselineKept;
          Logger::warn(str::format(
            "[VanishDiag] frame=", fid,
            " kept=", kept, " baseline=", d.baselineKept, " deficit=", deficitPct, "%",
            " (in=", d.drawsIn, " ignored=", d.drawsIgnored, ")",
            " camPos=(", pos.x, ", ", pos.y, ", ", pos.z, ")",
            " camFwd=(", latch.fwd.x, ", ", latch.fwd.y, ", ", latch.fwd.z, ")",
            " fov=", latch.fovRad,
            " vp=", latch.viewportW, "x", latch.viewportH,
            " maxZ=", latch.maxZ,
            " classFrame=", m_cameraManager.getMainClassifierFrameId()));

          // Diff per-VS-hash histogram against the last good (>=95% baseline)
          // snapshot to identify which VS bucket(s) lost draws. Sort by most-
          // negative delta and emit the top 8 droppers + any new appearances.
          if (!d.baselineVsHistogram.empty()) {
            struct VsDelta { XXH64_hash_t hash; int32_t delta; uint32_t base; uint32_t cur; };
            std::vector<VsDelta> deltas;
            deltas.reserve(d.baselineVsHistogram.size() + d.vsHistogram.size());
            for (const auto& [hash, baseCount] : d.baselineVsHistogram) {
              auto it = d.vsHistogram.find(hash);
              const uint32_t curCount = (it != d.vsHistogram.end()) ? it->second : 0u;
              const int32_t delta = static_cast<int32_t>(curCount) - static_cast<int32_t>(baseCount);
              if (delta != 0) {
                deltas.push_back({ hash, delta, baseCount, curCount });
              }
            }
            for (const auto& [hash, curCount] : d.vsHistogram) {
              if (d.baselineVsHistogram.find(hash) == d.baselineVsHistogram.end()) {
                deltas.push_back({ hash, static_cast<int32_t>(curCount), 0u, curCount });
              }
            }
            std::sort(deltas.begin(), deltas.end(),
              [](const VsDelta& a, const VsDelta& b) { return a.delta < b.delta; });
            const size_t maxLines = std::min<size_t>(8, deltas.size());
            for (size_t i = 0; i < maxLines; ++i) {
              const auto& v = deltas[i];
              Logger::warn(str::format(
                "[VanishDiag]   VS=0x", std::hex, v.hash, std::dec,
                " base=", v.base, " cur=", v.cur, " delta=", v.delta));
            }
          }
        } else if (d.baselineKept > 0 && kept * 100u >= d.baselineKept * 95u) {
          // Snapshot the histogram from this "good" frame for later diffing.
          d.baselineVsHistogram = d.vsHistogram;
        }
      }
      d.lastFrameId  = fid;
      d.drawsIn      = 0;
      d.drawsIgnored = 0;
      d.drawsKept    = 0;
      d.vsHistogram.clear();
    }
  }

  std::unordered_set<XXH64_hash_t> uniqueHashes;

  // NV-DXVK [perf]: shared gate for the per-draw diagnostic blocks in this file
  // (submitDrawState + processDrawCallState). These probes only ever feed
  // Logger lines that log.cpp's tag filter drops unless RTX_D3D11_DIAG=1, yet
  // their compute (strstr, getImageHash, 8-corner AABB transforms, mutex+set
  // inserts) ran for EVERY draw on the CS thread regardless. Read the env once.
  static bool sceneDiagEnabled() {
    static const bool s_enabled = []() {
      const char* v = std::getenv("RTX_D3D11_DIAG");
      return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return s_enabled;
  }

  // [SpawnGeomDiag] file-static per-frame counters tracking the
  // submitDrawState side of the fanout pipeline. Reset in prepareSceneData
  // and emitted alongside the preTlas census. Pairs with the d3d11 layer's
  // [SpawnGeomDiag.hist] line to answer: "of the N batches we shipped down
  // from d3d11, how many actually reached scene_manager / produced an
  // RtInstance? what fraction were point-instancer (instancesToObject!=null)?"
  static std::atomic<uint32_t> s_spawnDiagSubmitTotal { 0 };
  static std::atomic<uint32_t> s_spawnDiagSubmitWithFanout { 0 };
  static std::atomic<uint32_t> s_spawnDiagSubmitFanoutTforms { 0 };

  // NV-DXVK [DropTrace]: cross-file per-frame counter of dropship (Crow/Widow)
  // draws reaching submitDrawState. Read + logged by [HullCensus] in
  // InstanceManager::garbageCollection the same frame (submitDrawState runs
  // before that GC), giving a clean per-frame fate: submits vs instances vs
  // blasPrim — which the throttled/lumped logs cannot. extern'd in
  // rtx_instance_manager.cpp.
  std::atomic<uint32_t> g_dropTraceFrame{ 0xFFFFFFFFu };
  std::atomic<uint32_t> g_dropTraceSubmits{ 0u };

  // NV-DXVK [DropTrace] RAW: same counting, but incremented at the ENTRY of
  // D3D11Rtx::SubmitDraw (in d3d11_rtx.cpp, right after the studio-model name
  // is resolved, BEFORE any BumpFilter/return). g_dropTraceSubmits above counts
  // draws that survive the whole SubmitDraw cascade and reach submitDrawState;
  // this counts draws that ENTER it. Comparing the two splits the fork:
  //   raw == submits  -> nothing dropped inside SubmitDraw; the loss is engine-
  //                      side (the game stops submitting the dropship draws).
  //   raw  > submits  -> Remix's SubmitDraw filter cascade is dropping them.
  // NOTE: like the pair above, this does NOT dedup per VS/material — it
  // fetch_add(1) on EVERY matching dropship sub-mesh draw, so a 17->2 collapse
  // shows up as raw=17 (or 2). It is not the per-(vsHash,matHash) one-line
  // dedup that [SpawnGeomDiag.DrawIn] uses.
  std::atomic<uint32_t> g_dropTraceRawFrame{ 0xFFFFFFFFu };
  std::atomic<uint32_t> g_dropTraceRawSubmits{ 0u };

  // NV-DXVK [Perf.SubmitState]: stage timer for SceneManager::submitDrawState.
  // Modelled on [CommitRT] in rtx_context.cpp so the two lines read the same
  // way and can be compared directly.
  //
  // A DESTRUCTOR guard rather than inline accumulation at the end, because
  // submitDrawState has an early `return` on the buffer-cache-overflow drop
  // path. That return happens before any stage boundary, so it must still
  // count as a draw with only `entry` charged -- inline accumulation would
  // silently drop those draws and make the per-draw average look better than
  // it is. Stages that never ran contribute 0 by construction.
  namespace {
    struct SubmitStateSplitGuard {
      using clk = std::chrono::steady_clock;

      bool            on;
      clk::time_point t0, t1, t2, t3;
      bool            has1 = false, has2 = false, has3 = false;

      explicit SubmitStateSplitGuard(bool enabled)
      : on(enabled) {
        if (on)
          t0 = clk::now();
      }

      // Stage boundaries. Called positionally from submitDrawState.
      void mark1() { if (on) { t1 = clk::now(); has1 = true; } }
      void mark2() { if (on) { t2 = clk::now(); has2 = true; } }
      void mark3() { if (on) { t3 = clk::now(); has3 = true; } }

      static int64_t ns(clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
      }

      ~SubmitStateSplitGuard() {
        if (!on)
          return;

        const auto tEnd = clk::now();

        static thread_local int64_t  sEntry = 0, sHash = 0, sMat = 0, sProc = 0;
        static thread_local uint64_t sDraws = 0;
        static thread_local clk::time_point sLastLog{};
        static thread_local bool sInit = false;

        if (!sInit) { sLastLog = t0; sInit = true; }

        sEntry += ns(t0, has1 ? t1 : tEnd);
        if (has1) sHash += ns(t1, has2 ? t2 : tEnd);
        if (has2) sMat  += ns(t2, has3 ? t3 : tEnd);
        if (has3) sProc += ns(t3, tEnd);
        ++sDraws;

        // Frame id is not reachable from here without the device, so count
        // windows in wall time and report per-draw plus per-window totals; the
        // caller's [CommitRT] line already carries the per-frame view.
        if (std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - sLastLog).count() >= 3000) {
          const int64_t d = sDraws ? int64_t(sDraws) : 1;
          const int64_t tot = sEntry + sHash + sMat + sProc;

          // NV-DXVK [Perf.Report]: determineMaterialData, nested under
          // submitDrawState. sMat is a window total in ns over DRAWS, so frames
          // come from the report's clock -- dividing by sDraws here would give
          // per-draw microseconds and silently land in a ms/frame column.
          {
            static perfreport::WindowFrames s_wf;
            perfreport::publishWindow(perfreport::Slot::MatDataMs,
              double(sMat) / 1.0e6, s_wf.step());
          }

          Logger::info(str::format(
            "[Perf.SubmitState] draws=", sDraws,
            " avgUsPerDraw=", (tot / 1000 / d),
            " | entry=", (sEntry / 1000 / d), "us",
            " hash=", (sHash / 1000 / d), "us",
            " material=", (sMat / 1000 / d), "us",
            " process=", (sProc / 1000 / d), "us",
            " | pct entry=", (tot ? (sEntry * 100 / tot) : 0),
            " hash=", (tot ? (sHash * 100 / tot) : 0),
            " material=", (tot ? (sMat * 100 / tot) : 0),
            " process=", (tot ? (sProc * 100 / tot) : 0)));
          sEntry = sHash = sMat = sProc = 0;
          sDraws = 0;
          sLastLog = tEnd;
        }
      }
    };
  }

  void SceneManager::submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData) {
    ScopedCpuProfileZone();
    s_spawnDiagSubmitTotal.fetch_add(1, std::memory_order_relaxed);

    // NV-DXVK [Perf.SubmitState] (2026-08-06): split THIS function, because it
    // is the frame.
    //
    // THE CHAIN THAT ARRIVES HERE, all of it measured on the same clean window
    // (44 min flat memory, no gap sampler, 73.6 ms frames):
    //   [Perf.Busy]     wallMs 73.6            the frame
    //   [Perf.CsSplit]  dxvk-cs exec 73.3      99.6% -- dxvk-cs IS the frame
    //   [Perf.CsCmd]    commitGeometryToRT 53.1 ms/f, 1102 calls @ 48.2us
    //   [CommitRT]      submitMs 50 of perFrameMs 52, finalizeMs 0, otherMs 1
    // So ~96% of the largest item on the critical thread is this function, and
    // finalizeMs=0 says it is COMPUTING, not blocked on worker futures. Two
    // independent instruments (a vtable-keyed command probe and CommitRT's own
    // wall timer) agree to within 2%, which is why this is worth splitting
    // rather than re-measuring.
    //
    // WHAT THE STAGES SEPARATE:
    //   entry     everything before the replacement-hash lookup -- the category
    //             /fog/transform bookkeeping and the pile of per-draw
    //             diagnostics that accumulated in this function.
    //   hash      getHash(geometryAssetHashRule) + trackMeshHash +
    //             getReplacementsForMesh, plus the two LegacyAssetHash retries
    //             which each repeat all three on a miss.
    //   material  determineMaterialData.
    //   process   drawReplacements / processDrawCallState -- the actual
    //             instance + BLAS-input build.
    // If `process` owns it, the cost is real scene work and the lever is how
    // much geometry reaches here. If `entry` or `hash` owns it, the cost is
    // bookkeeping this function accreted and can be cut directly.
    //
    // COST: 4 clock reads per draw (~205 ns), ~0.23 ms/frame at 1102 draws --
    // under 0.4%. Gated off by default; turn it on for a capture only.
    const bool ssOn = RtxOptions::perfSubmitStateSplit();
    SubmitStateSplitGuard ssGuard(ssOn);

    // NV-DXVK [MeshTrace] funnel stage 1. This is the earliest point a game
    // draw is visible to RT, so a mesh counted here WAS submitted — the fact
    // [BulkPush] could never establish, because it counts per vertex shader
    // and one shader draws many meshes.
    meshtrace::record(
      static_cast<uint64_t>(input.getTransformData().vertexShaderHash),
      input.getGeometryData().vertexCount,
      meshtrace::Stage::Submitted,
      static_cast<uint64_t>(input.getMaterialData().getHash()));

    // NV-DXVK [Phase2b]: slim path for a sharded draw — the flush-side pre-pass
    // already ran everything between here and processDrawCallState (entry
    // diagnostics, the buffer-cache overflow check, the fog block, the hash +
    // replacement lookup, determineMaterialData), so re-running it would
    // double-mutate the caches it touches. The render material travels in the
    // sidecar, post-instance-manager mutations included. Legacy routes
    // (kLegacyCS / kNone / external draws with no sidecar) fall through to the
    // unchanged full path below.
    if (t_shardedConsume != nullptr
        && t_shardedConsume->route == ShardedDrawInfo::Route::kSharded) {
      assert(t_shardedConsume->renderMaterial != nullptr);
      processDrawCallState(ctx, input, *t_shardedConsume->renderMaterial, nullptr, nullptr);
      return;
    }

    // NV-DXVK [perf]: gate the per-draw diagnostic blocks below behind the same
    // RTX_D3D11_DIAG env the rest of the codebase uses. These blocks (strstr
    // model-name probes, getImageHash, 8-corner world-AABB transforms, mutex +
    // unordered_set inserts) run on the CS thread for EVERY draw (~600/frame),
    // yet their only output is a Logger::info line that log.cpp's tag filter
    // already drops unless RTX_D3D11_DIAG=1. So with the env unset, all of this
    // compute is pure waste. The gate itself reads the env once (file-scope
    // sceneDiagEnabled()). Set RTX_D3D11_DIAG=1 to bring every probe back.
    const bool s_sceneDiagEnabled = sceneDiagEnabled();

    // NV-DXVK [DropTrace]: count this draw if it's a dropship sub-mesh.
    if (s_sceneDiagEnabled) {
      const char* dtNm = input.studioModelName;
      if (dtNm[0] != '\0'
          && (std::strstr(dtNm, "Crow_dropship") != nullptr
           || std::strstr(dtNm, "widow") != nullptr)) {
        const uint32_t dtF = m_device->getCurrentFrameId();
        if (g_dropTraceFrame.load(std::memory_order_relaxed) != dtF) {
          g_dropTraceFrame.store(dtF, std::memory_order_relaxed);
          g_dropTraceSubmits.store(0u, std::memory_order_relaxed);
        }
        g_dropTraceSubmits.fetch_add(1u, std::memory_order_relaxed);
      }
    }

    // NV-DXVK [ShipSubmit] (SESSION-O): DEFINITIVE answer to the open tension in
    // HANDOFF_DROPSHIP_2026-06-08_SESSION-N — does the FULL ship REACH submitDrawState
    // during the vanish (=> drop is Remix-side, after here), or do only ~2 submeshes
    // arrive (=> dropped BEFORE submitDrawState, engine PATH-2)?
    //
    // This is the RELIABLE counterpart to [ShipDraw] (rtx_context.cpp). [ShipDraw]
    // built its world AABB by SAMPLING the position buffer, but the hull VB is
    // device-local => positionBuffer.mapPtr() returns null => sampled=0 => its
    // worldMin/Max stay at their +/-FLT_MAX init. So [ShipDraw]'s "empty/inverted AABB"
    // is a MEASUREMENT ARTIFACT of an unmappable VB, NOT proof of a degenerate ship
    // AABB (and its o2w.t=0 is just normal for VS 0x292b = static_mesh_cb3_owns_transform,
    // whose objectToWorld is identity by design). Here we derive the world AABB from the
    // finalized object-space boundingBox transformed by objectToWorld (8 corners) — no VB
    // map needed — so it is real whether or not the VB is mappable. We also log the raw
    // object-space bb, so a genuinely degenerate bb is distinguishable from a bad transform.
    //
    // ONE aggregated line PER FRAME (not per submesh). Logger::info is a SYNCHRONOUS
    // disk write — the first cut emitted up to 64 of them per frame and dropped the game
    // to ~2fps. We accumulate this frame's Crow/widow submeshes in file-static state and
    // lazily flush a single summary when the frame id advances, so logging cost is O(1)
    // lines/frame regardless of submesh count. Gated past menu/loading. Remove when done.
    //
    //   [ShipSubmit] f=N submeshes=8 prims=26996 degenerateAABB=0 worldEnv=[(min)..(max)]
    //
    // submeshes is THE answer to the open tension: stays ~8 through the vanish => the ship
    // reaches submitDrawState (drop is Remix-side, after here); collapses to ~2 (matching
    // [DropGeo] 8->2) => dropped before here on engine PATH-2. degenerateAABB>0 with
    // submeshes high => a real per-position AABB/transform bug.
    if (s_sceneDiagEnabled) {
      const char* nm = input.studioModelName;
      const bool isShip = nm[0] != '\0'
        && (std::strstr(nm, "Crow_dropship") != nullptr
         || std::strstr(nm, "widow") != nullptr);
      // Match [FloorTrace.recv]'s gameplay gate: >16 captured main-cam matrices == past
      // menu/loading. Keeps the log clean during non-gameplay frames.
      const bool inGameplay =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      if (isShip && inGameplay) {
        const auto& o2w = input.getTransformData().objectToWorld;
        const auto& bb  = input.getGeometryData().boundingBox;
        const Vector3 lmin = bb.minPos;
        const Vector3 lmax = bb.maxPos;
        Vector3 wmin( 1e30f,  1e30f,  1e30f);
        Vector3 wmax(-1e30f, -1e30f, -1e30f);
        for (int c = 0; c < 8; ++c) {
          const Vector4 lh(
            (c & 1) ? lmax.x : lmin.x,
            (c & 2) ? lmax.y : lmin.y,
            (c & 4) ? lmax.z : lmin.z,
            1.0f);
          const Vector4 wh = o2w * lh;
          wmin.x = std::min(wmin.x, wh.x); wmax.x = std::max(wmax.x, wh.x);
          wmin.y = std::min(wmin.y, wh.y); wmax.y = std::max(wmax.y, wh.y);
          wmin.z = std::min(wmin.z, wh.z); wmax.z = std::max(wmax.z, wh.z);
        }
        // empty/inverted object box OR empty/inverted world box == degenerate.
        const bool degenerate =
          !(lmax.x > lmin.x && lmax.y > lmin.y && lmax.z > lmin.z)
          || !(wmax.x >= wmin.x && wmax.y >= wmin.y && wmax.z >= wmin.z);
        const uint32_t prim = input.getGeometryData().calculatePrimitiveCount();
        const uint32_t fid  = m_device->getCurrentFrameId();

        // NV-DXVK [ShipGlitch]: SELF-TRIGGERING catcher for the intermittent
        // "all ships go funny when rotating" glitch (wrong location AND rotation).
        // Ships are world-placed via bones (o2w identity), so per submesh:
        //   - world CENTER should move smoothly → a big one-frame jump = wrong
        //     LOCATION (geometry mislocated that frame: torn bones / bad skin).
        //   - world SPAN (bbox dims) should be stable for a rigid vehicle → a big
        //     one-frame change = wrong ROTATION/deform (a rotated AABB changes
        //     extent even when the center holds).
        // Also fire on an impossibly large span (flung vertex) or non-finite.
        // Keyed per (vsHash,prim) → each submesh tracked across frames. Logs LOUD
        // ONLY on the bad frame — grep [ShipGlitch] for your 3 events + full state.
        // DISABLED — the glitch it caught is fixed; kept gated for future use.
        static const bool kEnableShipGlitch = false;
        if (kEnableShipGlitch) {
          const Vector3 ctr(0.5f * (wmin.x + wmax.x),
                            0.5f * (wmin.y + wmax.y),
                            0.5f * (wmin.z + wmax.z));
          const Vector3 span(wmax.x - wmin.x, wmax.y - wmin.y, wmax.z - wmin.z);
          const float maxSpan = std::max(span.x, std::max(span.y, span.z));
          const bool nonFinite = !(std::isfinite(ctr.x) && std::isfinite(ctr.y)
                                && std::isfinite(ctr.z) && std::isfinite(maxSpan));
          const XXH64_hash_t vsH = input.getTransformData().vertexShaderHash;
          const uint64_t key = static_cast<uint64_t>(vsH)
                             ^ (static_cast<uint64_t>(prim) * 0x9E3779B97F4A7C15ull);
          static std::mutex sGMu;
          static std::unordered_map<uint64_t, Vector3> sLastCtr;
          static std::unordered_map<uint64_t, Vector3> sLastSpan;
          float jumpCtr = 0.f, jumpSpan = 0.f;
          bool haveLast = false;
          {
            std::lock_guard<std::mutex> lk(sGMu);
            auto ic = sLastCtr.find(key);
            auto is = sLastSpan.find(key);
            if (ic != sLastCtr.end() && is != sLastSpan.end()) {
              haveLast = true;
              const Vector3 dc(ctr.x - ic->second.x, ctr.y - ic->second.y, ctr.z - ic->second.z);
              jumpCtr = std::sqrt(dc.x*dc.x + dc.y*dc.y + dc.z*dc.z);
              jumpSpan = std::max(std::abs(span.x - is->second.x),
                          std::max(std::abs(span.y - is->second.y),
                                   std::abs(span.z - is->second.z)));
            }
            if (!nonFinite) { sLastCtr[key] = ctr; sLastSpan[key] = span; }
          }
          const bool teleport = haveLast && jumpCtr  > 3000.0f;   // wrong LOCATION
          const bool rotated  = haveLast && jumpSpan > 1500.0f;   // wrong ROTATION/deform
          const bool stretched = maxSpan > 20000.0f;              // flung vertex
          if (nonFinite || teleport || rotated || stretched) {
            Logger::warn(str::format(
              "[ShipGlitch] f=", fid, " name=", nm,
              " vs=0x", std::hex, static_cast<uint64_t>(vsH), std::dec, " prim=", prim,
              " reason=", (nonFinite ? "NONFINITE" : teleport ? "TELEPORT(loc)"
                        : rotated ? "ROTATE/DEFORM" : "STRETCH"),
              " jumpCtr=", jumpCtr, " jumpSpan=", jumpSpan, " maxSpan=", maxSpan,
              " worldC=(", ctr.x, ",", ctr.y, ",", ctr.z, ")",
              " span=(", span.x, ",", span.y, ",", span.z, ")",
              " bbox=[(", wmin.x, ",", wmin.y, ",", wmin.z, ")..(",
                          wmax.x, ",", wmax.y, ",", wmax.z, ")]"));
          }
        }

        static std::mutex sMu;
        static uint32_t sFrame = UINT32_MAX;
        static uint32_t sCount = 0u, sPrims = 0u, sDegen = 0u;
        static Vector3 sWMin( 1e30f,  1e30f,  1e30f);
        static Vector3 sWMax(-1e30f, -1e30f, -1e30f);
        bool flush = false;
        uint32_t fFrame = 0u, fCount = 0u, fPrims = 0u, fDegen = 0u;
        Vector3 fWMin, fWMax;
        {
          std::lock_guard<std::mutex> lk(sMu);
          if (fid != sFrame) {
            // Frame advanced — emit the PREVIOUS frame's summary, then reset.
            if (sFrame != UINT32_MAX && sCount > 0u) {
              flush  = true;
              fFrame = sFrame; fCount = sCount; fPrims = sPrims; fDegen = sDegen;
              fWMin  = sWMin;  fWMax  = sWMax;
            }
            sFrame = fid; sCount = 0u; sPrims = 0u; sDegen = 0u;
            sWMin = Vector3( 1e30f,  1e30f,  1e30f);
            sWMax = Vector3(-1e30f, -1e30f, -1e30f);
          }
          ++sCount; sPrims += prim; if (degenerate) ++sDegen;
          sWMin.x = std::min(sWMin.x, wmin.x); sWMin.y = std::min(sWMin.y, wmin.y); sWMin.z = std::min(sWMin.z, wmin.z);
          sWMax.x = std::max(sWMax.x, wmax.x); sWMax.y = std::max(sWMax.y, wmax.y); sWMax.z = std::max(sWMax.z, wmax.z);
        }
        if (flush) {
          Logger::info(str::format(
            "[ShipSubmit] f=", fFrame,
            " submeshes=", fCount,
            " prims=", fPrims,
            " degenerateAABB=", fDegen,
            " worldEnv=[(", fWMin.x, ",", fWMin.y, ",", fWMin.z, ")..(",
                            fWMax.x, ",", fWMax.y, ",", fWMax.z, ")]"));
        }
      }
    }

    // [SpawnGeomDiag.DrawIn] Universal draw-call census. Per unique
    // (vsHash, materialHash, primCount) tuple, log one line at
    // submitDrawState entry. Includes:
    //   - vsHash + materialHash + texHash (Remix's content hashes)
    //   - primCount + vertexCount + indexCount
    //   - hasFanout / instCount
    //   - objectToWorld translation
    //   - WORLD-SPACE AABB (8 corners of local bounding box × o2w)
    // The world-space AABB is the key: if you know roughly WHERE the
    // missing surface should be in world coords, grep DrawIn lines
    // whose AABB contains that point. For TF2 spawn floor at
    // worldspawn__008__world_vr_training_vr_marble_floor, AABB is
    // X[-9024..3165] Y[-12296..8600] Z[-364..768]. Any DrawIn whose
    // world AABB lies inside that envelope is a floor-chunk
    // candidate. Cross-reference the matching matHash against
    // scene_full OBJ's `# blas …mat=0x…` comments — if absent from
    // scene_full, that draw is being dropped between
    // submitDrawState and BLAS construction.
    //
    // [SpawnGeomDiag.FloorCandidate] additionally fires (no throttle)
    // for any draw whose world AABB overlaps the floor's AABB AND
    // is at floor height (Z within [-400, 200]). Hardcoded floor
    // bounds match the user's measurement of the BSP marble-floor
    // mesh; adjust kFloorAABBMin / kFloorAABBMax if a different map
    // is being investigated.
    //
    // NV-DXVK [perf]: this is the heaviest of the per-draw diagnostics —
    // getImageHash() (also a cross-thread race risk vs streaming/GC) plus an
    // 8-corner world-AABB transform plus mutex+set lookups, every draw. Gated.
    if (s_sceneDiagEnabled) {
      const uint64_t vsHash = static_cast<uint64_t>(
        input.getTransformData().vertexShaderHash);
      const uint64_t matHash = static_cast<uint64_t>(
        input.getMaterialData().getHash());
      const uint64_t texHash = static_cast<uint64_t>(
        input.getMaterialData().getColorTexture().getImageHash());
      const uint32_t primCount = input.getGeometryData().calculatePrimitiveCount();
      const uint32_t vertCount = input.getGeometryData().vertexCount;
      const uint32_t idxCount  = input.getGeometryData().indexCount;
      const auto& o2w = input.getTransformData().objectToWorld;
      const auto& bb  = input.getGeometryData().boundingBox;

      // Compute world-space AABB by transforming all 8 corners of the
      // local bounding box and taking min/max. This is the actual
      // physical extent of the draw in world space.
      const Vector3 lmin = bb.minPos;
      const Vector3 lmax = bb.maxPos;
      Vector3 wmin( 1e30f,  1e30f,  1e30f);
      Vector3 wmax(-1e30f, -1e30f, -1e30f);
      for (int c = 0; c < 8; ++c) {
        const Vector4 lh(
          (c & 1) ? lmax.x : lmin.x,
          (c & 2) ? lmax.y : lmin.y,
          (c & 4) ? lmax.z : lmin.z,
          1.0f);
        const Vector4 wh = o2w * lh;
        wmin.x = std::min(wmin.x, wh.x); wmax.x = std::max(wmax.x, wh.x);
        wmin.y = std::min(wmin.y, wh.y); wmax.y = std::max(wmax.y, wh.y);
        wmin.z = std::min(wmin.z, wh.z); wmax.z = std::max(wmax.z, wh.z);
      }

      const uint64_t key = vsHash
        ^ ((matHash << 1) | (matHash >> 63))
        ^ (static_cast<uint64_t>(primCount) << 17);

      static std::mutex sDrawInMu;
      static std::unordered_set<uint64_t> sDrawInSeen;
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sDrawInMu);
        first = sDrawInSeen.insert(key).second;
      }

      const bool hasFanout = (input.getTransformData().instancesToObject != nullptr);
      const uint32_t instCount = hasFanout
        ? static_cast<uint32_t>(input.getTransformData().instancesToObject->size())
        : 0;
      Vector3 fanoutT0(0, 0, 0);
      if (hasFanout && instCount > 0) {
        const Matrix4 effective =
          o2w * (*input.getTransformData().instancesToObject)[0];
        fanoutT0 = Vector3(effective[3][0], effective[3][1], effective[3][2]);
      }

      if (first) {
        Logger::info(str::format(
          "[SpawnGeomDiag.DrawIn]"
          " vsHash=0x", std::hex, vsHash, std::dec,
          " matHash=0x", std::hex, matHash, std::dec,
          " texHash=0x", std::hex, texHash, std::dec,
          " primCnt=", primCount,
          " vCnt=", vertCount,
          " iCnt=", idxCount,
          " hasFanout=", (hasFanout ? 1 : 0),
          " instCount=", instCount,
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " fanoutT0=(", fanoutT0.x, ",", fanoutT0.y, ",", fanoutT0.z, ")",
          " worldAABB=[(", wmin.x, ",", wmin.y, ",", wmin.z, ")",
          "..(", wmax.x, ",", wmax.y, ",", wmax.z, ")]"));
      }

      // [SpawnGeomDiag.FloorCandidate] — fire (throttled to one log
      // per (vsHash,matHash) tuple) when this draw's world AABB
      // overlaps the marble-floor AABB at the right height. Most
      // BSP draws will overlap on X/Y; the Z gate filters out walls
      // and ceilings, leaving floor surfaces only.
      static const Vector3 kFloorAABBMin(-9024.291f, -12295.977f, -364.000f);
      static const Vector3 kFloorAABBMax( 3165.118f,   8599.516f,  768.000f);
      const bool xOverlap = (wmax.x >= kFloorAABBMin.x) && (wmin.x <= kFloorAABBMax.x);
      const bool yOverlap = (wmax.y >= kFloorAABBMin.y) && (wmin.y <= kFloorAABBMax.y);
      // Z gate: AABB must be at-or-near floor level. Walls/ceilings
      // typically have Z above the player's eye height (~108) or
      // below pit level (~-300). Floor surfaces in TF2 VR training
      // sit between roughly -50 and 200 in Z.
      const bool zFloorish = (wmin.z <= 200.0f) && (wmax.z >= -50.0f);
      if (xOverlap && yOverlap && zFloorish) {
        const uint64_t fcKey = vsHash ^ ((matHash << 1) | (matHash >> 63));
        static std::mutex sFloorCandMu;
        static std::unordered_set<uint64_t> sFloorCandSeen;
        bool fcFirst = false;
        {
          std::lock_guard<std::mutex> lk(sFloorCandMu);
          fcFirst = sFloorCandSeen.insert(fcKey).second;
        }
        if (fcFirst) {
          Logger::info(str::format(
            "[SpawnGeomDiag.FloorCandidate]"
            " vsHash=0x", std::hex, vsHash, std::dec,
            " matHash=0x", std::hex, matHash, std::dec,
            " texHash=0x", std::hex, texHash, std::dec,
            " primCnt=", primCount,
            " vCnt=", vertCount,
            " hasFanout=", (hasFanout ? 1 : 0),
            " instCount=", instCount,
            " worldAABB=[(", wmin.x, ",", wmin.y, ",", wmin.z, ")",
            "..(", wmax.x, ",", wmax.y, ",", wmax.z, ")]"));
        }
      }
    }

    if (input.getTransformData().instancesToObject != nullptr) {
      s_spawnDiagSubmitWithFanout.fetch_add(1, std::memory_order_relaxed);
      s_spawnDiagSubmitFanoutTforms.fetch_add(
        static_cast<uint32_t>(input.getTransformData().instancesToObject->size()),
        std::memory_order_relaxed);

      // [FloorTrace.recv] Mirror of [FloorTrace.emit] from d3d11. Diff
      // the two streams by frame to find batches d3d11 emits but scene
      // manager never receives — that gap is where the floor goes.
      // Frame-throttled to first 80 fanout-bearing draws per frame so
      // we always capture the early-spawn window (typically ~37 fanout
      // batches per frame in TF2 according to [SpawnGeomDiag.hist]).
      // Cap-per-frame is needed because submitDrawState runs on the CS
      // thread and can be hit hundreds of times in fast frames.
      //
      // Gameplay gate: matches [FloorTrace.emit]'s new gate in
      // d3d11_rtx.cpp — both streams must be filtered identically or the
      // diff is meaningless. tf2::g_engineHookCaptureCount > 16 means the
      // engine-hook trampoline has captured ≥16 real main-cam matrices,
      // i.e. we're past menu/loading.
      // NV-DXVK [perf]: skip the whole probe (incl. the per-fanout-draw mutex
      // lock) unless RTX_D3D11_DIAG is set — the log is filtered out otherwise.
      const bool inGameplayFloorTraceRecv = s_sceneDiagEnabled &&
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      static std::mutex sFloorTraceMtx;
      static uint32_t sFloorTraceFrame = UINT32_MAX;
      static uint32_t sFloorTracePerFrame = 0;
      const uint32_t fid = m_device->getCurrentFrameId();
      bool emit = false;
      if (inGameplayFloorTraceRecv) {
        std::lock_guard<std::mutex> lk(sFloorTraceMtx);
        if (fid != sFloorTraceFrame) {
          sFloorTraceFrame = fid;
          sFloorTracePerFrame = 0;
        }
        if (sFloorTracePerFrame < 80) {
          ++sFloorTracePerFrame;
          emit = true;
        }
      }
      if (emit) {
        const auto* xforms = input.getTransformData().instancesToObject;
        const Matrix4& o2w = input.getTransformData().objectToWorld;
        // Compose first-instance world position so we can match T0 from
        // the [FloorTrace.emit] line. Note d3d11's T0 is the post-cam-
        // origin translation column in tforms[0]; that's the same
        // tforms->vector pointed to by instancesToObject, so simply
        // reading (*xforms)[0][3] suffices.
        const Matrix4& t0 = (*xforms)[0];
        // Print the raw 64-bit hash hex to grep-match [FloorTrace.emit].
        Logger::info(str::format(
          "[FloorTrace.recv] frame=", fid,
          " vsHash=0x", std::hex,
          static_cast<uint64_t>(input.getTransformData().vertexShaderHash),
          std::dec,
          " size=", xforms->size(),
          " ptrKey=0x", std::hex,
          reinterpret_cast<uintptr_t>(xforms),
          std::dec,
          " T0=(", t0[3][0], ",", t0[3][1], ",", t0[3][2], ")",
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
      }
    }
    // NV-DXVK: UV-transform diag — SceneManager entry.
    {
      static uint32_t sSubUVx = 0;
      if (sSubUVx < 20) {
        const auto& m = input.getTransformData().textureTransform;
        if (m != Matrix4()) {
          ++sSubUVx;
          Logger::info(str::format(
            "[SceneMgr.UVx] submitDrawState got non-identity xform "
            "col0=(", m.data[0].x, ",", m.data[0].y, ") "
            "col1=(", m.data[1].x, ",", m.data[1].y, ") "
            "col3=(", m.data[3].x, ",", m.data[3].y, ")"));
        }
      }
    }
    if (m_bufferCache.getTotalCount() >= kBufferCacheLimit && m_bufferCache.getActiveCount() >= kBufferCacheLimit) {
      ONCE(Logger::info("[RTX-Compatibility-Info] This application is pushing more unique buffers than is currently supported - some objects may not raytrace."));
      // [SpawnGeomDiag.Drop] reason=bufferCacheOverflow — submitDrawState
      // returns immediately when the buffer cache is full. Once per
      // (vsHash, matHash).
      {
        const uint64_t vsH = static_cast<uint64_t>(input.getTransformData().vertexShaderHash);
        const uint64_t mH  = static_cast<uint64_t>(input.getMaterialData().getHash());
        const uint64_t key = vsH ^ ((mH << 1) | (mH >> 63)) ^ 0xb0fc4cull;
        static std::mutex sMu; static std::unordered_set<uint64_t> sSeen;
        bool first = false; { std::lock_guard<std::mutex> lk(sMu); first = sSeen.insert(key).second; }
        if (first) Logger::info(str::format(
          "[SpawnGeomDiag.Drop] reason=bufferCacheOverflow"
          " vsHash=0x", std::hex, vsH, std::dec,
          " matHash=0x", std::hex, mH, std::dec,
          " primCnt=", input.getGeometryData().calculatePrimitiveCount()));
      }
      return;
    }

    if (input.getFogState().mode != FogMode::None) {
      XXH64_hash_t fogHash = input.getFogState().getHash();
      if (m_fogStates.find(fogHash) == m_fogStates.end()) {
        // Only do anything if we haven't seen this fog before.
        m_fogStates[fogHash] = input.getFogState();

        MaterialData* pFogReplacement = m_pReplacer->getReplacementMaterial(fogHash);
        if (pFogReplacement) {
          // Track this replacement material hash for hash checking
          trackReplacementMaterialHash(fogHash);
          // Fog has been replaced by a translucent material to start the camera in,
          // meaning that it was being used to indicate 'underwater' or something similar.
          if (pFogReplacement->getType() != MaterialDataType::Translucent) {
            Logger::warn(str::format("Fog replacement materials must be translucent.  Ignoring material for ", std::hex, m_fog.getHash()));
          } else {
            uint32_t id = UINT32_MAX;
            createSurfaceMaterial(*pFogReplacement, input, &id);
            assert(id != UINT32_MAX);
            m_startInMediumMaterialIndex_inCache = id;
          }
        } else if (m_fog.mode == FogMode::None) {
          // render the first unreplaced fog.
          m_fog = input.getFogState();
        }
      }
    }


    ssGuard.mark1();   // NV-DXVK [Perf.SubmitState]: end of `entry`

    const XXH64_hash_t activeReplacementHash = input.getHash(RtxOptions::geometryAssetHashRule());

    // Track this mesh hash for mesh hash checking
    trackMeshHash(activeReplacementHash);
    
    std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForMesh(activeReplacementHash);

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash0);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash0: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash1);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash1: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    ssGuard.mark2();   // NV-DXVK [Perf.SubmitState]: end of `hash`

    MaterialData renderMaterialData = determineMaterialData(overrideMaterialData, input);

    ssGuard.mark3();   // NV-DXVK [Perf.SubmitState]: end of `material`

    if (pReplacements != nullptr) {
      drawReplacements(ctx, &input, pReplacements, renderMaterialData);
    } else {
      processDrawCallState(ctx, input, renderMaterialData, nullptr, nullptr);
    }
  }

  // NV-DXVK [Perf.MatData]: stage timer for determineMaterialData, measured at
  // 8us/draw = ~9 ms/frame = 12% of the frame by [Perf.SubmitState].
  //
  // The function is only ~45 lines, so 8us is suspicious on its face and the
  // candidates are specific:
  //   repl     m_pReplacer->getReplacementMaterial -- a hash lookup per draw.
  //   portal   RtxOptions::getRayPortalTextureIndex -- another hash lookup.
  //   convert  input.getMaterialData().as<OpaqueMaterialData>() -- builds a
  //            whole MaterialData, and note that MaterialData is RETURNED BY
  //            VALUE from this function. It carries 18 TextureRefs plus dozens
  //            of scalars, so if `convert` owns the time the cost is object
  //            construction + copy per draw, not lookup, and the fix is to stop
  //            materialising a fresh one for every draw.
  // exitPct fields say which path is actually taken, so a stage that looks
  // cheap because it rarely runs cannot be confused with one that is fast.
  namespace {
    struct MatDataSplitGuard {
      using clk = std::chrono::steady_clock;
      bool on;
      clk::time_point t0, tRepl, tPortal;
      bool hasRepl = false, hasPortal = false;
      int  exitPath = 0;   // 0=convert 1=override 2=replacement 3=highlight 4=portal

      explicit MatDataSplitGuard(bool enabled) : on(enabled) { if (on) t0 = clk::now(); }
      void markRepl()   { if (on) { tRepl   = clk::now(); hasRepl   = true; } }
      void markPortal() { if (on) { tPortal = clk::now(); hasPortal = true; } }
      void exit(int p)  { exitPath = p; }

      static int64_t ns(clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
      }

      ~MatDataSplitGuard() {
        if (!on) return;
        const auto tEnd = clk::now();

        static thread_local int64_t  sRepl = 0, sPortal = 0, sConv = 0;
        static thread_local uint64_t sCalls = 0, sExit[5] = { 0, 0, 0, 0, 0 };
        static thread_local clk::time_point sLastLog{};
        static thread_local bool sInit = false;
        if (!sInit) { sLastLog = t0; sInit = true; }

        sRepl += ns(t0, hasRepl ? tRepl : tEnd);
        if (hasRepl)   sPortal += ns(tRepl,   hasPortal ? tPortal : tEnd);
        if (hasPortal) sConv   += ns(tPortal, tEnd);
        ++sCalls;
        sExit[exitPath < 5 ? exitPath : 0]++;

        if (std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - sLastLog).count() >= 3000) {
          const int64_t c = sCalls ? int64_t(sCalls) : 1;
          const int64_t tot = sRepl + sPortal + sConv;
          Logger::info(str::format(
            "[Perf.MatData] calls=", sCalls,
            " avgUsPerCall=", (tot / 1000 / c),
            " | repl=", (sRepl / 1000 / c), "us",
            " portal=", (sPortal / 1000 / c), "us",
            " convert=", (sConv / 1000 / c), "us",
            " | pct repl=", (tot ? (sRepl * 100 / tot) : 0),
            " portal=", (tot ? (sPortal * 100 / tot) : 0),
            " convert=", (tot ? (sConv * 100 / tot) : 0),
            " | exit convert=", sExit[0], " override=", sExit[1],
            " replacement=", sExit[2], " highlight=", sExit[3], " portal=", sExit[4]));
          sRepl = sPortal = sConv = 0;
          sCalls = 0;
          for (int i = 0; i < 5; ++i) sExit[i] = 0;
          sLastLog = tEnd;
        }
      }
    };
  }

  MaterialData SceneManager::determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input) {
    MatDataSplitGuard mdSplit(RtxOptions::perfMaterialSplit());

    // First see if we have an explicit override
    if (overrideMaterialData != nullptr) {
      mdSplit.exit(1);
      return *overrideMaterialData;
    }

    // test if any direct material replacements exist
    MaterialData* pReplacementMaterial = m_pReplacer->getReplacementMaterial(input.getMaterialData().getHash());
    mdSplit.markRepl();
    if (pReplacementMaterial != nullptr) {
      // Make a copy - dont modify the replacement data.
      MaterialData renderMaterialData = *pReplacementMaterial;
      // merge in the input material from game
      renderMaterialData.mergeLegacyMaterial(input.getMaterialData());
      mdSplit.exit(2);
      return renderMaterialData;
    }

    // Detect meshes that would have unstable hashes due to the vertex hash using vertex data from a shared vertex buffer.
    // TODO: Once the vertex hash only uses vertices referenced by the index buffer, this should be removed.
    const bool highlightUnsafeAnchor = RtxOptions::useHighlightUnsafeAnchorMode() && input.getGeometryData().indexBuffer.defined() && input.getGeometryData().vertexCount > input.getGeometryData().indexCount;
    if (highlightUnsafeAnchor) {
      const static MaterialData sHighlightMaterialData(OpaqueMaterialData(TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(),
                                                                          0.f, 1.f, Vector3(0.2f, 0.2f, 0.2f), 1.0f, 0.1f, 0.1f, Vector3(0.46f, 0.26f, 0.31f), true,
                                                                          /* AlphaModulateEmissive */ false, /* EmissiveTintFromConstant */ false,
                                                                          1, 1, 0, false, false, 200.f, true, false, BlendType::kAlpha, false, AlphaTestType::kAlways, 0, 0.0f, 0.0f, Vector3(), 0.0f, Vector3(), 0.0f, false, Vector3(), 0.0f, 0.0f,
                                                                          lss::Mdl::Filter::Nearest, lss::Mdl::WrapMode::Repeat, lss::Mdl::WrapMode::Repeat,
                                                                          /* IsUnlitOutput */ false,
                                                                          /* HasScreenSpaceEmissive */ false,
                                                                          Vector2(1.f, 0.f), Vector2(0.f, 1.f), Vector2(0.f, 0.f),
                                                                          /* Tf2SkyboxFog */ false,
                                                                          /* AlbedoIsPremultiplied */ false));
      mdSplit.exit(3);
      return sHighlightMaterialData;
    }

    // Check if a Ray Portal override is needed
    size_t rayPortalTextureIndex;
    mdSplit.markPortal();
    if (RtxOptions::getRayPortalTextureIndex(input.getMaterialData().getHash(), rayPortalTextureIndex)) {
      assert(rayPortalTextureIndex < maxRayPortalCount);
      assert(rayPortalTextureIndex < std::numeric_limits<uint8_t>::max());

      MaterialData renderMaterialData = input.getMaterialData().as<RayPortalMaterialData>();
      renderMaterialData.getRayPortalMaterialData().setRayPortalIndex(rayPortalTextureIndex);
      mdSplit.exit(4);
      return renderMaterialData;
    }

    // Standard legacy material conversion.
    //
    // NV-DXVK [perf]: this is the only exit TF2 ever takes ([Perf.MatData] exit
    // convert=100%), and it rebuilds an OpaqueMaterialData -- 18 TextureRefs plus
    // 42 constants -- once per draw call for a scene that draws far fewer distinct
    // materials than draw calls. Memoized for the frame.
    //
    // Cached at THIS level rather than around the whole function on purpose: the
    // four exits above are cheap (repl 2%, portal 2% by [Perf.MatData]) but they
    // decide WHICH material a draw gets, and caching across them would let a
    // stale key choose an exit. Here the cache can only ever substitute the
    // conversion result, never change the routing.
    //
    // Returned by value, as before: drawReplacements takes a non-const
    // MaterialData& and may mutate it, so callers must not get a handle on the
    // cache's copy.
    if (!RtxOptions::cacheLegacyMaterialConversion()) {
      return input.getMaterialData().as<OpaqueMaterialData>();
    }

    const XXH64_hash_t convKey = input.getMaterialData().getOpaqueConversionKey();
    auto iter = m_legacyOpaqueConversionCache.find(convKey);
    if (iter != m_legacyOpaqueConversionCache.end()) {
      ++m_legacyOpaqueConversionHits;
      return iter->second;
    }

    // Construct straight into the map so the miss path costs one move rather than
    // an extra full copy of the material.
    ++m_legacyOpaqueConversionMisses;
    auto inserted = m_legacyOpaqueConversionCache.emplace(convKey, input.getMaterialData().as<OpaqueMaterialData>());
    return inserted.first->second;
  }

  void SceneManager::createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance) {
    const float effectLightIntensity = RtxOptions::effectLightIntensity();
    if (effectLightIntensity <= 0.f)
      return;

    const RasterGeometry& geometryData = input.getGeometryData();

    const GeometryBufferData bufferData(geometryData);
    
    if (!bufferData.indexData && geometryData.indexCount > 0 || !bufferData.positionData)
      return;

    // Find centroid of point cloud.
    Vector3 centroid = Vector3();
    uint32_t counter = 0;
    if (geometryData.indexCount > 0) {
      for (uint32_t i = 0; i < geometryData.indexCount; i++) {
        const uint16_t index = bufferData.getIndex(i);
        centroid += bufferData.getPosition(index);
        ++counter;
      }
    } else {
      for (uint32_t i = 0; i < geometryData.vertexCount; i++) {
        centroid += bufferData.getPosition(i);
        ++counter;
      }
    }
    centroid /= (float) counter;
    
    const Vector4 renderingPos = input.getTransformData().objectToView * Vector4(centroid.x, centroid.y, centroid.z, 1.0f);
    // Note: False used in getViewToWorld since the renderingPos of the object is defined with respect to the game's object to view
    // matrix, not our freecam's, and as such we want to convert it back to world space using the matching matrix.
    const Vector4 worldPos{ getCamera().getViewToWorld(false) * Vector4d{ renderingPos } };

    RtLightShaping shaping{};

    float lightRadius = std::max(RtxOptions::effectLightRadius(), 1e-3f);
    const Vector3 lightPosition { worldPos.x, worldPos.y, worldPos.z };
    Vector3 lightRadiance;
    if (RtxOptions::effectLightPlasmaBall()) {
      // Todo: Make these options more configurable via config options.
      const double timeMilliseconds = static_cast<double>(GlobalTime::get().absoluteTimeMs());
      const double animationPhase = sin(timeMilliseconds * 0.006) * 0.5 + 0.5;
      lightRadiance = lerp(Vector3(1.f, 0.921f, 0.738f), Vector3(1.f, 0.521f, 0.238f), animationPhase);
    } else {
      const auto& diffuse = input.getMaterialData().getLegacyMaterial().Diffuse;
      lightRadiance = Vector3(diffuse[0], diffuse[1], diffuse[2]) * RtxOptions::effectLightColor();
    }
    const float surfaceArea = 4.f * kPi * lightRadius * lightRadius;
    const float radianceFactor = 1e5f * effectLightIntensity / surfaceArea;
    lightRadiance *= radianceFactor;

    RtLight rtLight(RtSphereLight(lightPosition, lightRadiance, lightRadius, shaping));
    rtLight.isDynamic = true;

    m_lightManager.addLight(rtLight, input, RtLightAntiCullingType::MeshReplacement);
  }

  void SceneManager::drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, MaterialData& renderMaterialData) {
    ScopedCpuProfileZone();
    // TODO: Ideally we should create and track `replacementInstance` based on the draw call.  It currently relies on the
    // `findSimilarInstance` function of the first RtInstance created for the draw call, which is pretty clumsy.
    // We also should be tracking and garbage collecting the entire draw call together,
    // rather than doing each instance separately.
    ReplacementInstance* replacementInstance = nullptr;

    // Detect replacements of meshes that would have unstable hashes due to the vertex hash using vertex data from a shared vertex buffer.
    // TODO: Once the vertex hash only uses vertices referenced by the index buffer, this should be removed.
    const bool highlightUnsafeReplacement = RtxOptions::useHighlightUnsafeReplacementMode() &&
        input->getGeometryData().indexBuffer.defined() && input->getGeometryData().vertexCount > input->getGeometryData().indexCount;
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto& replacement = (*pReplacements)[i];
      RtInstance* instance = nullptr;
      if (replacement.includeOriginal) {
        DrawCallState newDrawCallState(*input);
        newDrawCallState.categories = replacement.categories.applyCategoryFlags(newDrawCallState.categories);
        // NV-DXVK [fanout split]: a replacement prim must resolve to exactly one
        // RtInstance — the loop below stores it in ReplacementInstance::prims and
        // asserts that later frames return the same pointer. Clearing the flag
        // keeps a replaced fanout batch on the point-instancer path.
        newDrawCallState.transformData.isFanoutBatch = false;
        const RtxParticleSystemDesc* pParticleSystemDesc = replacement.particleSystem.has_value() ? &replacement.particleSystem.value() : nullptr;
        instance = processDrawCallState(ctx, newDrawCallState, renderMaterialData, nullptr, pParticleSystemDesc);
      } else if (replacement.type == AssetReplacement::eMesh) {
        DrawCallTransforms transforms = input->getTransformData();
        
        transforms.objectToWorld = transforms.objectToWorld * replacement.replacementToObject;
        transforms.objectToView = transforms.objectToView * replacement.replacementToObject;

        if (!replacement.instancesToObject.empty()) {
          transforms.instancesToObject = &replacement.instancesToObject;
        } else {
          transforms.instancesToObject = nullptr;
        }
        // NV-DXVK [fanout split]: as above — one RtInstance per replacement prim.
        // The transforms here are the replacement's own GeomPointInstancer set,
        // which is a stable authored batch and belongs on the PI path anyway.
        transforms.isFanoutBatch = false;
        
        // Mesh replacements dont support these.
        transforms.textureTransform = Matrix4();
        transforms.texgenMode = TexGenMode::None;

        DrawCallState newDrawCallState(*input);
        newDrawCallState.geometryData = replacement.geometry->data; // Note: Geometry Data replaced
        newDrawCallState.transformData = transforms;
        newDrawCallState.categories = replacement.categories.applyCategoryFlags(newDrawCallState.categories);

        // Note: Material Data replaced if a replacement is specified in the Mesh Replacement
        if (replacement.materialData != nullptr) {
          renderMaterialData = *replacement.materialData;
        }
        if (highlightUnsafeReplacement) {
          const static MaterialData sHighlightMaterialData(OpaqueMaterialData(TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(), TextureRef(),
              0.f, 1.f, Vector3(0.2f, 0.2f, 0.2f), 1.f, 0.1f, 0.1f, Vector3(1.f, 0.f, 0.f), true,
              /* AlphaModulateEmissive */ false, /* EmissiveTintFromConstant */ false,
              1, 1, 0, false, false, 200.f, true, false, BlendType::kAlpha, false, AlphaTestType::kAlways, 0, 0.0f, 0.0f, Vector3(), 0.0f, Vector3(), 0.0f, false, Vector3(), 0.0f, 0.0f,
              lss::Mdl::Filter::Nearest, lss::Mdl::WrapMode::Repeat, lss::Mdl::WrapMode::Repeat,
              /* IsUnlitOutput */ false,
              /* HasScreenSpaceEmissive */ false,
              Vector2(1.f, 0.f), Vector2(0.f, 1.f), Vector2(0.f, 0.f),
              /* Tf2SkyboxFog */ false,
              /* AlbedoIsPremultiplied */ false));
          if ((GlobalTime::get().absoluteTimeMs()) / 200 % 2 == 0) {
            renderMaterialData = sHighlightMaterialData;
          }
        }

        const RtxParticleSystemDesc* pParticleSystemDesc = replacement.particleSystem.has_value() ? &replacement.particleSystem.value() : nullptr;

        RtInstance* existingInstance = replacementInstance ? replacementInstance->prims[i].getInstance() : nullptr;
        // Only use findSimilarInstance if we're processing the root of a replacement - all others should just rely on the existingInstance.
        instance = processDrawCallState(ctx, newDrawCallState, renderMaterialData, existingInstance, pParticleSystemDesc);
      }
      
      if (instance != nullptr) {
        if (replacementInstance == nullptr) {
          // first mesh in this replacement, so it becomes the root.
          replacementInstance = instance->getPrimInstanceOwner().getOrCreateReplacementInstance(instance, PrimInstance::Type::Instance, i, pReplacements->size());
        }
        if (replacementInstance->prims[i].getUntyped() == nullptr) {
          // First frame, need to set the replacement instance.
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        } else if (replacementInstance->prims[i].getInstance() != instance) {
          Logger::err(str::format("ReplacementInstance: instance returned by processDrawCallState is not the same as the one stored. index: ", i,"  mesh hash: ", std::hex, input->getHash(RtxOptions::geometryAssetHashRule())));
          assert(false && "instance returned by processDrawCallState is not the same as the one stored.");
        }
      }
    }

    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eLight) {
        if (replacementInstance == nullptr) {
          // TODO(TREX-1141) if we refactor instancing to depend on the pre-replacement drawcall instead
          // of the fully processed draw call, we can remove this requirement.
          Logger::err(str::format(
              "Light prims anchored to a mesh replacement must also include actual meshes.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        if (replacement.lightData.has_value()) {
          RtLight localLight = replacement.lightData->toRtLight();
          localLight.applyTransform(input->getTransformData().objectToWorld);
          
          // Handle all non-root lights as externally tracked lights - they'll be cleaned up when the root is garbage collected.
          // For mesh replacements, the root is always a mesh, so no need to handle root lights here.
          RtLight* existingLight = replacementInstance->prims[i].getLight();
          if (existingLight != nullptr) {
            if (existingLight->getPrimInstanceOwner().getReplacementInstance() != replacementInstance) {
              ONCE(assert(false && "light in a replacementInstance believes it is owned by a different replacementInstance."));
            }
            m_lightManager.updateExternallyTrackedLight(existingLight, localLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(localLight);
            newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
          }
        }
      }
    }

    // Create graphs associated with this replacement, if they haven't already been created.
    // Graphs are cleaned up when the replacementInstance is destroyed, which happens when the 
    // root instance is destroyed.
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eGraph && replacementInstance->prims[i].getGraph() == nullptr) {
        if (!replacement.graphState.has_value()) {
          Logger::err(str::format(
              "Graph prims missing graph state in mesh replacement.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        GraphInstance* graphInstance = m_graphManager.addInstance(ctx, replacement.graphState.value());
        if (graphInstance) {
          graphInstance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, graphInstance, PrimInstance::Type::Graph);
        }
      }
    }
  }

  namespace {
    // The buffers a RaytraceGeometry can hold a bindless slot for, each paired
    // with the field that holds the slot.
    //
    // ONE list, walked by both the bind and the retire, because the two used to
    // be separate hand-written runs of if/else and a buffer added to the geometry
    // has to reach both of them. Miss the bind and the surface reads a stale
    // slot; miss the retire and the slot is never reclaimed. Order is the order
    // the tape used to append in, kept so slot numbers in an old log still read
    // sensibly against a new one.
    struct GeoBufferSlot {
      RaytraceBuffer RaytraceGeometry::* buffer;
      uint32_t RaytraceGeometry::* index;
    };

    constexpr GeoBufferSlot kGeoBufferSlots[] = {
      { &RaytraceGeometry::indexBuffer,            &RaytraceGeometry::indexBufferIndex },
      { &RaytraceGeometry::normalBuffer,           &RaytraceGeometry::normalBufferIndex },
      { &RaytraceGeometry::color0Buffer,           &RaytraceGeometry::color0BufferIndex },
      { &RaytraceGeometry::texcoordBuffer,         &RaytraceGeometry::texcoordBufferIndex },
      { &RaytraceGeometry::positionBuffer,         &RaytraceGeometry::positionBufferIndex },
      { &RaytraceGeometry::previousPositionBuffer, &RaytraceGeometry::previousPositionBufferIndex },
      // NV-DXVK: TF2 worldspace VGUI auxiliary structured buffers. These are the
      // game's own SRVs rather than buffers Remix allocated, and they are
      // re-cloned from the draw on every call, so if the game renames one the
      // geometry silently starts holding a different slice. That is the reason
      // the retire below cannot assume only a destroyed geometry orphans slots.
      { &RaytraceGeometry::vguiFontBoundsBuffer,   &RaytraceGeometry::vguiFontBoundsBufferIndex },
      { &RaytraceGeometry::vguiImgBoundsBuffer,    &RaytraceGeometry::vguiImgBoundsBufferIndex },
      { &RaytraceGeometry::vguiStylesBuffer,       &RaytraceGeometry::vguiStylesBufferIndex },
    };

    constexpr size_t kGeoBufferSlotCount = sizeof(kGeoBufferSlots) / sizeof(kGeoBufferSlots[0]);
  }

  void SceneManager::updateBufferCache(const RaytraceGeometry& oldGeoData, RaytraceGeometry& newGeoData) {
    ScopedCpuProfileZone();

    const uint32_t frameId = m_device->getCurrentFrameId();

    // BIND. newGeoData was copied from oldGeoData before the bake, so every index
    // field already holds the slot the previous bake handed it. Slots are stable
    // now, so a field whose buffer did not move needs no work at all -- the index
    // it is already carrying still names the same buffer. The tape had to
    // re-track every field of every draw every frame only because its indices did
    // not survive the frame boundary; nine slice compares replace nine appends.
    //
    // retain() asks the table whether the slot still holds this exact buffer
    // rather than comparing against oldGeoData, and re-asserts ownership when it
    // does. The re-assertion matters for the buffers several geometries share
    // (the VGUI structured buffers), where one geometry being destroyed retires a
    // slot another one still holds.
    for (const GeoBufferSlot& slot : kGeoBufferSlots) {
      const RaytraceBuffer& buffer = newGeoData.*(slot.buffer);
      uint32_t& index = newGeoData.*(slot.index);

      if (!buffer.defined()) {
        index = kSurfaceInvalidBufferIndex;
        continue;
      }

      if (index != kSurfaceInvalidBufferIndex && m_bufferCache.retain(index, buffer, frameId)) {
        continue;
      }

      index = m_bufferCache.track(buffer, frameId);
    }

    // RETIRE. A bake can replace the buffers this geometry holds: a vertex count
    // change reallocates both history buffers, and the VGUI structured buffers
    // are re-cloned from the draw on every call, so they follow the game's own
    // SRV. Whatever the geometry no longer references has to be handed back, or
    // the table grows without bound -- and it is the LIVE geometry that does
    // this, so retiring only on destruction would not be enough.
    //
    // Compare against the WHOLE new set, not field by field. The two history
    // buffers exchange roles on every vertex update -- position takes what was
    // previous-position and vice versa -- so a field-by-field comparison would
    // report both as changed and retire a buffer the geometry still holds.
    //
    // The cheap test first: when every field is unchanged there is nothing to
    // retire, and that is the overwhelmingly common case (a draw that resolved to
    // kUpdateInstance touches no buffer at all).
    bool anyBufferMoved = false;
    for (const GeoBufferSlot& slot : kGeoBufferSlots) {
      if (!(oldGeoData.*(slot.buffer)).matches(newGeoData.*(slot.buffer))) {
        anyBufferMoved = true;
        break;
      }
    }

    if (!anyBufferMoved) {
      return;
    }

    for (size_t i = 0; i < kGeoBufferSlotCount; ++i) {
      const RaytraceBuffer& released = oldGeoData.*(kGeoBufferSlots[i].buffer);
      if (!released.defined()) {
        continue;
      }

      bool stillHeld = false;
      for (size_t j = 0; j < kGeoBufferSlotCount && !stillHeld; ++j) {
        const RaytraceBuffer& kept = newGeoData.*(kGeoBufferSlots[j].buffer);
        stillHeld = kept.defined() && kept.matches(released);
      }

      if (!stillHeld) {
        m_bufferCache.retire(released);
      }
    }
  }

  void SceneManager::retireGeometryBufferSlots(const RaytraceGeometry& geoData) {
    for (const GeoBufferSlot& slot : kGeoBufferSlots) {
      const RaytraceBuffer& buffer = geoData.*(slot.buffer);
      if (buffer.defined()) {
        m_bufferCache.retire(buffer);
      }
    }
  }

  SceneManager::ObjectCacheState SceneManager::onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    // This is a new object.
    ObjectCacheState result = processGeometryInfo<true>(ctx, drawCallState, pBlas->modifiedGeometryData);
    
    assert(result == ObjectCacheState::KBuildBVH);

    pBlas->frameLastUpdated = m_device->getCurrentFrameId();

    return result;
  }
  
  SceneManager::ObjectCacheState SceneManager::onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    if (pBlas->frameLastTouched == m_device->getCurrentFrameId()) {
      pBlas->cacheMaterial(drawCallState.getMaterialData());
      return SceneManager::ObjectCacheState::kUpdateInstance;
    }

    // TODO: If mesh is static, no need to do any of the below, just use the existing modifiedGeometryData and set result to kInstanceUpdate.
    ObjectCacheState result = processGeometryInfo<false>(ctx, drawCallState, pBlas->modifiedGeometryData);

    // We dont expect to hit the rebuild path here - since this would indicate an index buffer or other topological change, and that *should* trigger a new scene object (since the hash would change)
    assert(result != ObjectCacheState::KBuildBVH);

    if (result == ObjectCacheState::kUpdateBVH)
      pBlas->frameLastUpdated = m_device->getCurrentFrameId();
    
    pBlas->clearMaterialCache();
    pBlas->input = drawCallState; // cache the draw state for the next time.
    return result;
  }
  
  bool SceneManager::touchResidentRecord(uint64_t key, uint32_t frameId) {
    return m_instanceManager.getResidentScene().touch(key, frameId);
  }

  void SceneManager::onSceneObjectDestroyed(const BlasEntry& blas) {
    // The geometry goes away with the entry, so every bindless slot it holds is
    // orphaned here. Retire, do not free: a surface uploaded in an earlier frame
    // can still carry one of these indices and the GPU can still be reading it.
    // The reclaim sweep frees the slot once the surface upload stops reporting a
    // reference to it.
    retireGeometryBufferSlots(blas.modifiedGeometryData);

    for (RtInstance* instance : blas.getLinkedInstances()) {
      instance->markForGarbageCollection();
      instance->markAsUnlinkedFromBlasEntryForGarbageCollection();
    }
  }

  void SceneManager::onInstanceAdded(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->linkInstance(&instance);
    }
  }

  void SceneManager::onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, const bool hasTransformChanged, const bool hasVerticesChanged, const bool isFirstUpdateThisFrame) {
    // NV-DXVK [perf]: DxvkObjects::capturer() returns Rc<GameCapturer> BY VALUE,
    // so binding it unconditionally costs an atomic increment on entry and an
    // atomic decrement on scope exit, per INSTANCE per frame (~15,500x) -- for a
    // pointer that only the two branches below use, and [Perf.UpdInst] measures
    // those branches at xfChg=1% and prevPos=0%. Same shape as RtInstance::move()
    // in the v5 handoff (process note 5): the test was already here, it just sat
    // downstream of the work it could have skipped. Behaviour is unchanged --
    // when neither flag is set, the old code acquired the pointer and used it for
    // nothing.
    if (hasTransformChanged || hasVerticesChanged) {
      // NV-DXVK [Phase2b]: setInstanceUpdateFlag writes an unlocked global map
      // (GameCapturer::instanceFlags) — inert unless capturing, but a capture
      // during the sharded phase would race it. Cold (xfChg~1%), so the escape
      // lock covers it.
      std::unique_lock<std::mutex> capLock;
      if (inShardedInstancePhase()) {
        capLock = std::unique_lock<std::mutex>(m_instanceManager.shardEscapeMutex());
      }
      auto capturer = m_device->getCommon()->capturer();
      if (hasTransformChanged) {
        capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
      }

      if (hasVerticesChanged) {
        capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::PositionsUpdate);
        capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::NormalsUpdate);
      }
    }

    // Create and bind the RT material.
    // NV-DXVK [perf]: take the cache index while we are here. bindMaterial below
    // used to re-derive it with a second map lookup, per instance per frame.
    uint32_t surfaceMaterialIndex = UINT32_MAX;
    const RtSurfaceMaterial& surfaceMaterial = createSurfaceMaterial(material, drawCall, &surfaceMaterialIndex);

    // [SkyDiag] One line per distinct (vsHash, albedoHash, cameraType) draw -
    // a full picture of how every surface is being classified. Used to find
    // why TF2 sky cloud billboards render as opaque rectangles instead of
    // being composited into the sky. Key columns:
    //   cam=    CameraType (Sky means the draw is in the sky pass)
    //   skyCat= InstanceCategories::Sky flag
    // A cloud-card draw showing cam!=Sky AND skyCat=0 is the bug - it is
    // being treated as ordinary world geometry. v/i (vertex/index count)
    // and o2wT/o2wScale identify the billboards: a few verts at a sky-
    // distance translation. albHash lets us cross-reference the exact
    // texture (grab the cloud's hash from the Remix developer menu).
    // Gameplay-gated on g_engineHookCaptureCount (the engine-hook main-cam
    // capture counter) > 16 - the same gate the rest of the fork uses, so
    // boot/menu/loading draws don't burn the 512-entry budget. Frame-id
    // gating is unreliable here because the game runs at low framerates.
    // Deduped so a stable scene yields a bounded log.
    //
    // NV-DXVK [perf]: SATURATION LATCH. This probe is charged to the `tail`
    // bucket of [Perf.UpdInst] (4.6 ms/frame, the largest stage of
    // updateInstance) and it runs once per INSTANCE -- ~15,500x per frame -- for
    // the entire session, because g_engineHookCaptureCount > 16 is a monotonic
    // "in gameplay" latch that never goes back down. The budget is 512 keys and
    // the log holds exactly 512 [SkyDiag] lines, so it saturated long ago and has
    // since paid, per instance per frame: a TextureRef::getImageHash() (two
    // derefs into cold image memory), the key build, and a global mutex acquire
    // -- to produce nothing.
    //
    // This is the same failure as rtx.logTlasSet in the v5 handoff (sec 0b): the
    // whole cost sits UPSTREAM of the test that discards the result. The fix is
    // the same shape -- make saturation visible to the gate instead of only to
    // the insert. A fresh run still collects its full 512 lines; a saturated run
    // costs one relaxed bool load. Nothing about what gets logged changes.
    static std::atomic<bool> s_skyDiagSaturated { false };
    if (!s_skyDiagSaturated.load(std::memory_order_relaxed) &&
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      static std::mutex s_skyDiagMu;
      static std::unordered_set<uint64_t> s_skyDiagSeen;
      const LegacyMaterialData& skyDiagMat = drawCall.getMaterialData();
      const TextureRef& skyDiagAlb = skyDiagMat.getColorTexture();
      const XXH64_hash_t albHash = (skyDiagAlb.isValid() && !skyDiagAlb.isImageEmpty())
        ? skyDiagAlb.getImageHash() : 0ull;
      const uint64_t vsH = static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
      // NV-DXVK [FogHideProbe]: blend + fogCap folded into the dedupe key —
      // the fog walls draw the same (vs, albHash) in both a premult-blended
      // pass (fogCap=1, hidden) and a blend-off pass (fogCap=0, visible);
      // without these bits only the first pass would ever log.
      const uint64_t key = vsH ^ albHash
        ^ (uint64_t(static_cast<uint32_t>(drawCall.cameraType)) << 56)
        ^ (drawCall.getMaterialData().blendMode.enableBlending ? (1ull << 55) : 0ull)
        ^ (skyDiagMat.sourceTf2FogCapable ? (1ull << 54) : 0ull);
      bool firstSkyDiag = false;
      {
        std::lock_guard<std::mutex> g(s_skyDiagMu);
        if (s_skyDiagSeen.size() < 512) {
          firstSkyDiag = s_skyDiagSeen.insert(key).second;
          // Publish saturation to the gate above. Set inside the lock so the
          // size() that decides it is the same one the insert just produced.
          if (s_skyDiagSeen.size() >= 512) {
            s_skyDiagSaturated.store(true, std::memory_order_relaxed);
          }
        }
      }
      if (firstSkyDiag) {
        const Matrix4& o2w = drawCall.getTransformData().objectToWorld;
        auto colLen = [](const Vector4& c) {
          return std::sqrt(float(c.x) * float(c.x) + float(c.y) * float(c.y) + float(c.z) * float(c.z));
        };
        Logger::info(str::format("[SkyDiag]",
          " vs=0x", std::hex, vsH, std::dec,
          // NV-DXVK [FogHideProbe]: ps + fogCap identify which stacked fog
          // draws the matTf2Fog hide misses. fogCap mirrors LegacyMaterialData
          // ::sourceTf2FogCapable (PS reads c_fogColorFactor AND premult-OVER
          // blend, set in d3d11_rtx FillMaterialData) — fogCap=0 + blend=0 on
          // a garbage albHash = the draw fails the premult half of the gate.
          " ps=0x", std::hex, uint64_t(drawCall.getTransformData().pixelShaderHash), std::dec,
          " fogCap=", skyDiagMat.sourceTf2FogCapable ? 1 : 0,
          " albHash=0x", std::hex, albHash, std::dec,
          " cam=", static_cast<int>(drawCall.cameraType),
          " skyCat=", drawCall.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0,
          " ignoreAC=", drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0,
          " v=", drawCall.getGeometryData().vertexCount,
          " i=", drawCall.getGeometryData().indexCount,
          " blend=", drawCall.getMaterialData().blendMode.enableBlending ? 1 : 0,
          " alphaOp=", static_cast<int>(drawCall.getMaterialData().alphaTestCompareOp),
          " o2wT=(", float(o2w[3][0]), ",", float(o2w[3][1]), ",", float(o2w[3][2]), ")",
          " o2wScale=(", colLen(o2w[0]), ",", colLen(o2w[1]), ",", colLen(o2w[2]), ")",
          " catRaw=0x", std::hex, drawCall.getCategoryFlags().raw(), std::dec));
      }
    }

    // NV-DXVK TF2 "white character parts" diagnostic, v2.
    // v1 gated on isFirstUpdateThisFrame+albedoEmpty hypothesis: zero hits
    // across a 28-min gameplay session, so that hypothesis is wrong — first
    // updates DO arrive with a non-empty colorTextures[0]. White must come
    // from somewhere downstream of the LegacyMaterialData -> SurfaceMaterial
    // path, not from the bindMaterial gate.
    //
    // v2: log EVERY distinct material-binding event for skinned instances so
    // we can see what actually gets bound and follow it forward. Dedupe by
    // (instance, isFirst, albedoHash) tuple so a stable scene produces a
    // bounded log instead of one line per frame.
    //
    // Each line shows the LegacyMaterialData albedo that fed createSurfaceMaterial,
    // PLUS the resulting RtSurfaceMaterial type and whether it has a valid
    // albedo/opacity texture. If a draw arrives with a non-empty
    // legacy-albedo but the surface material ends up with a null albedo,
    // that's the smoking gun and points the bug at determineMaterialData /
    // as<OpaqueMaterialData>() / createSurfaceMaterial.
    //
    // Gating: skinned-only (numBones > 0). Skips world/UI/sky entirely.
    if (drawCall.getSkinningState().numBones > 0) {
      static std::mutex sDiagMtx;
      static std::atomic<uint32_t> sDiagLines { 0 };
      // key = ((uintptr)instance ^ albedoHash) << 1 | isFirst
      static std::unordered_set<uint64_t> sLoggedKeys;

      const uint32_t kMaxLines = 1500;

      if (sDiagLines.load() < kMaxLines) {
        const LegacyMaterialData& legacy = drawCall.getMaterialData();
        const TextureRef& tex0 = legacy.getColorTexture();
        const bool legacyAlbedoEmpty = !tex0.isValid() || tex0.isImageEmpty();
        const XXH64_hash_t legacyAlbedoHash = legacyAlbedoEmpty ? 0ull : tex0.getImageHash();
        const XXH64_hash_t legacyMatHash = legacy.getHash();

        // Inspect the resulting surface material to see if conversion
        // dropped the albedo somewhere between LegacyMaterialData and
        // RtSurfaceMaterial. If legacyAlbedoEmpty=0 but
        // surfaceAlbedoEmpty=1, the conversion path is the bug surface.
        const auto surfType = surfaceMaterial.getType();
        bool surfaceAlbedoEmpty = true;
        if (surfType == RtSurfaceMaterialType::Opaque) {
          const auto& opq = surfaceMaterial.getOpaqueSurfaceMaterial();
          // getAlbedoOpacityTextureIndex returns kSurfaceMaterialInvalidTextureIndex (0xFFFFu) when unset
          surfaceAlbedoEmpty = (opq.getAlbedoOpacityTextureIndex() == kSurfaceMaterialInvalidTextureIndex);
        }

        // For surfEmpty=1 cases, capture WHICH branch of hasTextureCoordinates()
        // failed: (a) texcoordBuffer not defined, (b) texgenMode == None, or
        // (c) both. trackTexture drops the bind iff !hasTexcoords, which is
        // exactly the surfEmpty=1 / legacyEmpty=0 pattern we see in the log.
        const bool tcDefined = drawCall.getGeometryData().texcoordBuffer.defined();
        const int texgen = static_cast<int>(drawCall.getTransformData().texgenMode);

        const uint64_t key = (reinterpret_cast<uint64_t>(&instance) ^ legacyAlbedoHash ^ (uint64_t(surfaceAlbedoEmpty) << 60)) * 2ull
                             + (isFirstUpdateThisFrame ? 1u : 0u);

        std::lock_guard<std::mutex> lock(sDiagMtx);
        if (sLoggedKeys.insert(key).second) {
          sDiagLines.fetch_add(1);
          const BlasEntry* pBlas = instance.getBlas();
          Logger::info(str::format(
            "[RtxWhiteDiag2]",
            " frame=", m_device->getCurrentFrameId(),
            " inst=", static_cast<const void*>(&instance),
            " blas=", static_cast<const void*>(pBlas),
            " first=", (isFirstUpdateThisFrame ? 1 : 0),
            " legacyEmpty=", (legacyAlbedoEmpty ? 1 : 0),
            " surfEmpty=", (surfaceAlbedoEmpty ? 1 : 0),
            " surfType=", static_cast<int>(surfType),
            " tcDef=", (tcDefined ? 1 : 0),
            " texgen=", texgen,
            std::hex,
            " albHash=0x", legacyAlbedoHash,
            " legMatHash=0x", legacyMatHash,
            std::dec,
            " usesPS=", (drawCall.usesPixelShader ? 1 : 0),
            " usesVS=", (drawCall.usesVertexShader ? 1 : 0),
            " zWr=", (drawCall.zWriteEnable ? 1 : 0),
            " zEn=", (drawCall.zEnable ? 1 : 0),
            " cam=", static_cast<int>(drawCall.cameraType),
            " cat=0x", std::hex, drawCall.getCategoryFlags().raw(), std::dec,
            " v=", drawCall.getGeometryData().vertexCount,
            " i=", drawCall.getGeometryData().indexCount,
            " b=", drawCall.getSkinningState().numBones));
        }
      }
    }

    if(isFirstUpdateThisFrame) {
      m_instanceManager.bindMaterial(instance, surfaceMaterial, surfaceMaterialIndex);
    }

    // Update portal
    if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
      // NV-DXVK [Phase2b]: RayPortalManager state is global and pair-order
      // sensitive; findSimilarInstance's portal virtual-matching also READS it
      // lock-free during the parallel phase, which is only sound because no one
      // writes it then. Defer to the CS record step (ordered, pre-injectRTX),
      // which re-derives the surface material via the per-thread memo.
      if (inShardedInstancePhase()) {
        t_shardPhase.currentOps->rayPortal = true;
      } else {
        m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
      }
    }
  }

  void SceneManager::onInstanceDestroyed(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    // Some BLAS were cleared in the SceneManager::garbageCollection().
    // When a BLAS is destroyed, all instances that linked to it will be automatically unlinked. In such case we don't need to
    // call onInstanceDestroyed to double unlink the instances.
    // Note: This case often happens when BLAS are destroyed faster than instances. (e.g. numFramesToKeepGeometryData >= numFramesToKeepInstances)
    if (pBlas != nullptr && !instance.isUnlinkedForGC()) {
      pBlas->unlinkInstance(&instance);
    }
  }

  // Helper to populate the texture cache with this resource (and patch sampler if required for texture)
  void SceneManager::trackTexture(const TextureRef &inputTexture,
                                  uint32_t& textureIndex,
                                  bool hasTexcoords,
                                  bool async,
                                  uint16_t samplerFeedbackStamp) {
    // If no texcoords, no need to bind the texture
    if (!hasTexcoords) {
      // NV-DXVK: count dropped texture bindings so we can tell whether the
      // "no UVs" path is the root cause of missing textures. Log a summary
      // every 500 drops so the file doesn't explode.
      static std::atomic<uint32_t> sNoUvDropCount{0};
      const uint32_t count = ++sNoUvDropCount;
      if (count == 1 || (count % 500) == 0) {
        Logger::info(str::format(
          "[RTX-Compatibility-Info] Dropping texture bind — mesh has no UVs. "
          "Total dropped so far: ", count));
      }
      return;
    }

    auto& textureManager = m_device->getCommon()->getTextureManager();
    textureManager.addTexture(inputTexture, samplerFeedbackStamp, async, textureIndex);
  }

  // NV-DXVK: auto-dump every unique texture ref to rtx-remix/captures/textures/
  // the first frame it's seen. No hotkey required. Deduped via an
  // instance-wide hash set so we don't spam the exporter with duplicates.
  void SceneManager::autoDumpMaterialTextures(Rc<DxvkContext> ctx, const MaterialData& material) {
    // NV-DXVK: disabled by default. This auto-dump writes every unique
    // material texture to rtx-remix/captures/textures/ on first sighting
    // with no hotkey — over a session it produced 127k+ .dds files and
    // filled the disk. To re-enable, set the environment variable
    // DXVK_RTX_AUTODUMP_TEXTURES=1 before launching.
    static const bool s_autoDumpEnabled =
      !env::getEnvVar("DXVK_RTX_AUTODUMP_TEXTURES").empty();
    if (!s_autoDumpEnabled) {
      return;
    }
    if (material.getType() != MaterialDataType::Opaque) {
      return;
    }
    const auto& opaque = material.getOpaqueMaterialData();

    static const std::string kTexDir =
      util::RtxFileSys::path(util::RtxFileSys::Captures).string()
      + std::string(lss::commonDirName::texDir);
    static bool kDirMade = (env::createDirectory(kTexDir), true); (void)kDirMade;

    auto& exporter = m_device->getCommon()->metaExporter();

    auto tryDump = [&](const TextureRef& tex, const char* suffix) {
      if (!tex.isValid() || tex.isImageEmpty()) {
        return;
      }
      const XXH64_hash_t h = tex.getImageHash();
      if (h == 0) {
        return;
      }
      {
        std::lock_guard<std::mutex> lk(m_autoDumpedTexturesMutex);
        if (!m_autoDumpedTextureHashes.insert(h).second) {
          return; // already dumped
        }
      }
      auto* view = tex.getImageView();
      if (!view) {
        return;
      }
      std::string filename = str::format(std::hex, h, std::dec, "_", suffix, lss::ext::dds);
      exporter.dumpImageToFile(ctx, kTexDir, filename, view->image());
    };

    tryDump(opaque.getAlbedoOpacityTexture(),   "albedo");
    tryDump(opaque.getNormalTexture(),          "normal");
    tryDump(opaque.getRoughnessTexture(),       "roughness");
    tryDump(opaque.getMetallicTexture(),        "metallic");
    tryDump(opaque.getEmissiveColorTexture(),   "emissive");
    tryDump(opaque.getAmbientOcclusionTexture(),"ao");
    tryDump(opaque.getLightmapTexture(),        "lightmap0");
    tryDump(opaque.getLightmap2Texture(),       "lightmap1");
    tryDump(opaque.getDetailTexture(),          "detail");
    tryDump(opaque.getCloudMaskTexture(),       "cloudmask");
  }

  // NV-DXVK: targeted per-draw dump (rtx.debug.dumpVertexShaders). Writes the
  // bound game textures to captures/textures/ (deduped by image hash) and logs
  // a [DumpDraw] geometry/transform report — including a flat-vs-mesh
  // coplanarity verdict computed over the actual raytraced vertex positions —
  // once per matched VS hash. Pure diagnostic; does not alter rendering.
  void SceneManager::dumpDrawForVertexShader(Rc<DxvkContext> ctx, const DrawCallState& drawCallState) {
    const auto& dumpSet = RtxOptions::dumpVertexShaders();
    if (dumpSet.empty()) {
      return;
    }
    const XXH64_hash_t vsHash = drawCallState.getTransformData().vertexShaderHash;
    if (dumpSet.count(vsHash) == 0) {
      return;
    }

    static const std::string kTexDir =
      util::RtxFileSys::path(util::RtxFileSys::Captures).string()
      + std::string(lss::commonDirName::texDir);
    static bool kDirMade = (env::createDirectory(kTexDir), true); (void) kDirMade;
    auto& exporter = m_device->getCommon()->metaExporter();

    // --- 1) dump bound game color textures (BaseTexture etc.), once per image.
    const LegacyMaterialData& md = drawCallState.getMaterialData();
    auto tryDump = [&](const TextureRef& tex, const char* suffix) {
      if (!tex.isValid() || tex.isImageEmpty()) {
        return;
      }
      const XXH64_hash_t h = tex.getImageHash();
      if (h == 0) {
        return;
      }
      {
        std::lock_guard<std::mutex> lk(m_dumpDrawMutex);
        if (!m_dumpedDrawTextureHashes.insert(h).second) {
          return; // already written this image
        }
      }
      auto* view = tex.getImageView();
      if (!view) {
        return;
      }
      const VkExtent3D ext = view->imageInfo().extent;
      const std::string filename = str::format(
        "vs_", std::hex, vsHash, "_tex_", h, std::dec,
        "_", suffix, "_", ext.width, "x", ext.height, lss::ext::dds);
      exporter.dumpImageToFile(ctx, kTexDir, filename, view->image());
      Logger::info(str::format(
        "[DumpDraw] wrote texture ", filename, " (", ext.width, "x", ext.height,
        ", fmt=", static_cast<int>(view->info().format), ")"));
    };
    tryDump(md.getColorTexture(),  "color0");
    tryDump(md.getColorTexture2(), "color1");

    // --- 2) geometry/transform report + coplanarity verdict, once per VS.
    {
      std::lock_guard<std::mutex> lk(m_dumpDrawMutex);
      if (!m_dumpedDrawVsHashes.insert(vsHash).second) {
        return;
      }
    }

    const RasterGeometry& g = drawCallState.getGeometryData();
    const DrawCallTransforms& t = drawCallState.getTransformData();
    const Matrix4& o2w = t.objectToWorld;

    Logger::info(str::format(
      "[DumpDraw] vs=0x", std::hex, vsHash,
      " matHash=0x", md.getHash(),
      " tex0=0x", md.getColorTexture().isImageEmpty() ? XXH64_hash_t(0) : md.getColorTexture().getImageHash(),
      " tex1=0x", md.getColorTexture2().isImageEmpty() ? XXH64_hash_t(0) : md.getColorTexture2().getImageHash(),
      std::dec,
      " vCnt=", g.vertexCount, " iCnt=", g.indexCount,
      " topo=", static_cast<int>(g.topology),
      " cullMode=", static_cast<int>(g.cullMode),
      " frontFace=", static_cast<int>(g.frontFace),
      " bonesPerVtx=", g.numBonesPerVertex,
      " hasNormalBuf=", g.normalBuffer.defined() ? 1 : 0,
      " hasTexcoordBuf=", g.texcoordBuffer.defined() ? 1 : 0,
      " hasColorBuf=", g.color0Buffer.defined() ? 1 : 0,
      " hasPosBuf=", g.positionBuffer.defined() ? 1 : 0,
      " posFmt=", g.positionBuffer.defined() ? static_cast<int>(g.positionBuffer.vertexFormat()) : -1,
      " isSubView=", t.isSubView ? 1 : 0,
      " isSubViewSkybox=", t.isSubViewSkybox ? 1 : 0,
      " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));

    // Which Remix instance category this draw lands in. NOTE: if this VS is
    // also in rtx.debug.hideVertexShaders, Hidden will read 1 — remove it from
    // the hide list for one run to see the *natural* category it would get.
    Logger::info(str::format(
      "[DumpDraw] categories: Hidden=", drawCallState.testCategoryFlags(InstanceCategories::Hidden) ? 1 : 0,
      " Ignore=", drawCallState.testCategoryFlags(InstanceCategories::Ignore) ? 1 : 0,
      " Sky=", drawCallState.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0,
      " WorldUI=", drawCallState.testCategoryFlags(InstanceCategories::WorldUI) ? 1 : 0,
      " WorldMatte=", drawCallState.testCategoryFlags(InstanceCategories::WorldMatte) ? 1 : 0,
      " DecalStatic=", drawCallState.testCategoryFlags(InstanceCategories::DecalStatic) ? 1 : 0,
      " DecalDynamic=", drawCallState.testCategoryFlags(InstanceCategories::DecalDynamic) ? 1 : 0,
      " DecalNoOffset=", drawCallState.testCategoryFlags(InstanceCategories::DecalNoOffset) ? 1 : 0,
      " Particle=", drawCallState.testCategoryFlags(InstanceCategories::Particle) ? 1 : 0,
      " Beam=", drawCallState.testCategoryFlags(InstanceCategories::Beam) ? 1 : 0,
      " Terrain=", drawCallState.testCategoryFlags(InstanceCategories::Terrain) ? 1 : 0,
      " zWrite=", drawCallState.zWriteEnable ? 1 : 0,
      " zTest=", drawCallState.zEnable ? 1 : 0,
      " usesPS=", drawCallState.usesPixelShader ? 1 : 0));

    // Coplanarity test over the raytraced object-space positions. A flat
    // billboard card is coplanar in any space; a real 3D mesh is not. We read
    // the positions Remix actually feeds the BLAS, so this also surfaces a
    // mis-decoded vertex format (it shows up as garbage / huge extents).
    const VkFormat posFmt = g.positionBuffer.defined()
      ? g.positionBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
    const bool posIsFloat3 =
      posFmt == VK_FORMAT_R32G32B32_SFLOAT || posFmt == VK_FORMAT_R32G32B32A32_SFLOAT;
    const uint8_t* posBase = g.positionBuffer.defined()
      ? reinterpret_cast<const uint8_t*>(
          g.positionBuffer.mapPtr((size_t) g.positionBuffer.offsetFromSlice()))
      : nullptr;

    if (posIsFloat3 && posBase != nullptr && g.vertexCount >= 3) {
      const size_t stride = g.positionBuffer.stride();
      const uint32_t n = std::min<uint32_t>(g.vertexCount, 256u);
      auto P = [&](uint32_t i, int c) -> float {
        return reinterpret_cast<const float*>(posBase + stride * i)[c];
      };
      // AABB.
      float mn[3] = { P(0,0), P(0,1), P(0,2) };
      float mx[3] = { mn[0], mn[1], mn[2] };
      for (uint32_t i = 1; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
          const float v = P(i, c);
          mn[c] = std::min(mn[c], v);
          mx[c] = std::max(mx[c], v);
        }
      }
      const float dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
      const float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
      // Plane normal from the first non-degenerate triangle of unique verts.
      float nx = 0, ny = 0, nz = 0; bool haveNormal = false;
      const float ax = P(0,0), ay = P(0,1), az = P(0,2);
      for (uint32_t j = 1; j < n && !haveNormal; ++j) {
        const float e1x = P(j,0)-ax, e1y = P(j,1)-ay, e1z = P(j,2)-az;
        for (uint32_t k = j+1; k < n; ++k) {
          const float e2x = P(k,0)-ax, e2y = P(k,1)-ay, e2z = P(k,2)-az;
          nx = e1y*e2z - e1z*e2y; ny = e1z*e2x - e1x*e2z; nz = e1x*e2y - e1y*e2x;
          const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
          if (len > 1e-6f) { nx/=len; ny/=len; nz/=len; haveNormal = true; break; }
        }
      }
      if (haveNormal) {
        float maxDev = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
          const float d = std::fabs(nx*(P(i,0)-ax) + ny*(P(i,1)-ay) + nz*(P(i,2)-az));
          maxDev = std::max(maxDev, d);
        }
        const float ratio = (diag > 1e-6f) ? (maxDev / diag) : 0.0f;
        Logger::info(str::format(
          "[DumpDraw] coplanarity: vertsTested=", n,
          " aabbDiag=", diag, " maxPlaneDev=", maxDev,
          " ratio=", ratio,
          " verdict=", (ratio < 0.01f ? "FLAT-CARD(coplanar)" : "NON-PLANAR-MESH"),
          " planeN=(", nx, ",", ny, ",", nz, ")"));
      } else {
        Logger::info(str::format(
          "[DumpDraw] coplanarity: degenerate (all sampled verts collinear/coincident), aabbDiag=", diag));
      }
    } else {
      Logger::info(str::format(
        "[DumpDraw] coplanarity: positions not readable (posIsFloat3=", posIsFloat3 ? 1 : 0,
        " mapped=", posBase != nullptr ? 1 : 0, " vCnt=", g.vertexCount, ")"));
    }
  }

  RtInstance* SceneManager::processDrawCallState(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, MaterialData& renderMaterialData, RtInstance* existingInstance, const RtxParticleSystemDesc* pParticleSystemDesc) {
    ScopedCpuProfileZone();

    // NV-DXVK [MeshTrace] funnel stage 2: survived submitDrawState's filters.
    // Submitted > ProcessDcs means the draw was rejected before ever becoming
    // scene geometry (hidden shader, category filter, early-out).
    meshtrace::record(
      static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash),
      drawCallState.getGeometryData().vertexCount,
      meshtrace::Stage::ProcessDcs);

    // NV-DXVK [perf][ProcDCS]: split this function (the ~85% of CS-thread submitDrawState
    // cost) into geom = onSceneObject{Added,Updated} (geometry interleave / BLAS input /
    // hashing) vs inst = InstanceManager::processSceneObject (RtInstance build + material).
    // Per ~3s window. Tells us which half of the ~170us/draw to attack.
    static thread_local int64_t  s_pdcsTotalNs = 0, s_pdcsGeomNs = 0, s_pdcsInstNs = 0;
    static thread_local uint64_t s_pdcsCount  = 0;
    static thread_local uint32_t s_pdcsFrames = 0, s_pdcsLastFid = UINT32_MAX;
    static thread_local std::chrono::steady_clock::time_point s_pdcsLastLog{};
    static thread_local bool s_pdcsInit = false;
    const auto tPdcs0 = std::chrono::steady_clock::now();
    if (!s_pdcsInit) { s_pdcsLastLog = tPdcs0; s_pdcsInit = true; }
    { const uint32_t f = m_device->getCurrentFrameId();
      if (f != s_pdcsLastFid) { s_pdcsLastFid = f; ++s_pdcsFrames; } }
    if (std::chrono::duration_cast<std::chrono::milliseconds>(tPdcs0 - s_pdcsLastLog).count() >= 3000) {
      const int64_t fr = s_pdcsFrames ? int64_t(s_pdcsFrames) : 1;

      // NV-DXVK [Perf.Report]: the middle of the dxvk-cs chain. instMs is the
      // parent of the four [Perf.SceneObj] stages and the report cross-checks
      // their sum against it -- the two are the same function, so a gap means a
      // stage boundary is wrong rather than that work appeared.
      perfreport::publishWindow(perfreport::Slot::ProcDcsMs,
        double(s_pdcsTotalNs) / 1.0e6, uint64_t(fr));
      perfreport::publishWindow(perfreport::Slot::ProcDcsGeomMs,
        double(s_pdcsGeomNs) / 1.0e6, uint64_t(fr));
      perfreport::publishWindow(perfreport::Slot::ProcDcsInstMs,
        double(s_pdcsInstNs) / 1.0e6, uint64_t(fr));

      Logger::info(str::format(
        "[ProcDCS] window draws=", s_pdcsCount, " frames=", s_pdcsFrames,
        " perFrameMs=", s_pdcsTotalNs / 1000000 / fr,
        " geomMs=", s_pdcsGeomNs / 1000000 / fr,
        " instMs=", s_pdcsInstNs / 1000000 / fr,
        " otherMs=", (s_pdcsTotalNs - s_pdcsGeomNs - s_pdcsInstNs) / 1000000 / fr,
        " geomUsPerDraw=", (s_pdcsCount ? s_pdcsGeomNs / 1000 / int64_t(s_pdcsCount) : 0),
        " instUsPerDraw=", (s_pdcsCount ? s_pdcsInstNs / 1000 / int64_t(s_pdcsCount) : 0)));

      // NV-DXVK [Perf.GeoSplit]: the mechanism split of geomMs. Silent unless
      // rtx.logPrepSceneSplit is on (same switch as [Perf.GeoChurn]'s counts —
      // read them together). recUs is DERIVED (bake - alloc) and is the part
      // that cannot leave dxvk-cs; allocUs and tapeUs are the movable halves,
      // and restUs is what the decision extraction already left behind.
      if (s_geoBakeN != 0) {
        const int64_t recNs = (s_geoBakeNs > s_geoAllocNs) ? (s_geoBakeNs - s_geoAllocNs) : 0;
        const int64_t restNs = s_pdcsGeomNs - s_geoBakeNs - s_geoTapeNs;
        Logger::info(str::format(
          "[Perf.GeoSplit] calls=", s_geoBakeN, " frames=", s_pdcsFrames,
          " | allocUs=", s_geoAllocNs / 1000 / fr, " (n=", s_geoAllocN / uint64_t(fr), ")",
          " recUs=", recNs / 1000 / fr,
          " tapeUs=", s_geoTapeNs / 1000 / fr, " (n=", s_geoTapeN / uint64_t(fr), ")",
          " restUs=", restNs / 1000 / fr,
          " | geomUs=", s_pdcsGeomNs / 1000 / fr,
          "   MOVABLE=alloc+tape=", (s_geoAllocNs + s_geoTapeNs) / 1000 / fr, "us"
          " PINNED=rec=", recNs / 1000 / fr, "us"));
      }
      s_geoAllocNs = s_geoBakeNs = s_geoTapeNs = 0;
      s_geoAllocN = s_geoBakeN = s_geoTapeN = 0;

      s_pdcsLastLog = tPdcs0;
      s_pdcsTotalNs = 0; s_pdcsGeomNs = 0; s_pdcsInstNs = 0; s_pdcsCount = 0; s_pdcsFrames = 0;
    }
    struct PdcsGuard {
      std::chrono::steady_clock::time_point t0; int64_t& acc; uint64_t& cnt;
      ~PdcsGuard() {
        acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        ++cnt;
      }
    } pdcsGuard{ tPdcs0, s_pdcsTotalNs, s_pdcsCount };

    // NV-DXVK: targeted dump (rtx.debug.dumpVertexShaders) — runs before any
    // drop/ignore/hidden logic so it captures even hidden draws. No-op unless
    // the option set is non-empty and contains this draw's VS hash.
    dumpDrawForVertexShader(ctx, drawCallState);

    ++vanishDiag().drawsIn;

    if (renderMaterialData.getIgnored()) {
      ++vanishDiag().drawsIgnored;
      // [SpawnGeomDiag.Drop] reason=matIgnored — material data has its
      // ignored flag set (rtx options ignoreTextures lookup, or
      // explicit setIgnored).
      {
        const uint64_t vsH = static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash);
        const uint64_t mH  = static_cast<uint64_t>(drawCallState.getMaterialData().getHash());
        const uint64_t key = vsH ^ ((mH << 1) | (mH >> 63)) ^ 0x16e0d7eull;
        static std::mutex sMu; static std::unordered_set<uint64_t> sSeen;
        bool first = false; { std::lock_guard<std::mutex> lk(sMu); first = sSeen.insert(key).second; }
        if (first) Logger::info(str::format(
          "[SpawnGeomDiag.Drop] reason=matIgnored"
          " vsHash=0x", std::hex, vsH, std::dec,
          " matHash=0x", std::hex, mH, std::dec,
          " primCnt=", drawCallState.getGeometryData().calculatePrimitiveCount()));
      }
      return nullptr;
    }

    // NV-DXVK: auto-dump material textures on first sighting.
    autoDumpMaterialTextures(ctx, renderMaterialData);

    // NV-DXVK [Phase2b]: CS-side consume. For a sharded draw the pre-pass
    // already resolved the BlasEntry, stamped frameLastTouched/noteDraw, and ran
    // the CPU bookkeeping halves of onSceneObject{Added,Updated} (material
    // cache ops + input copy) on the flush side; the shard decided the cache
    // state. What remains HERE is the GPU-record half: processGeometryInfo
    // recomputes the identical decision (computeGeometryCacheState — same
    // inputs, unchanged since the flush by the drain contract) and records the
    // bake's copies/dispatches + updateBufferCache at this draw's CS position,
    // exactly like the legacy path.
    ShardedDrawInfo* const p2b =
      (t_shardedConsume != nullptr && t_shardedConsume->route == ShardedDrawInfo::Route::kSharded)
        ? t_shardedConsume : nullptr;

    ObjectCacheState result = ObjectCacheState::kInvalid;
    BlasEntry* pBlas = nullptr;
    const auto tPdcsGeom0 = std::chrono::steady_clock::now();
    if (p2b != nullptr) {
      pBlas = p2b->pBlas;
      if (!p2b->blasFirstDrawOfFrame) {
        // Same-frame duplicate of this entry: the legacy path's touched-fast
        // branch (cacheMaterial + kUpdateInstance); cacheMaterial ran at flush.
        result = ObjectCacheState::kUpdateInstance;
      } else if (static_cast<ObjectCacheState>(p2b->geomResult) == ObjectCacheState::KBuildBVH) {
        // onSceneObjectAdded minus the flush-side halves.
        result = processGeometryInfo<true>(ctx, drawCallState, pBlas->modifiedGeometryData);
        pBlas->frameLastUpdated = m_device->getCurrentFrameId();
      } else {
        // onSceneObjectUpdated minus the flush-side halves (clearMaterialCache +
        // input copy ran at flush; the buffer-field fixup happens below).
        result = processGeometryInfo<false>(ctx, drawCallState, pBlas->modifiedGeometryData);
        if (result == ObjectCacheState::kUpdateBVH) {
          pBlas->frameLastUpdated = m_device->getCurrentFrameId();
        }
      }
      // NV-DXVK [Phase2b]: input-copy buffer fixup. The flush-side copy
      // (pBlas->input = drawCallState) ran BEFORE the gpuCapture rebind, so its
      // geometry buffers still name the renameable dynamic slices. Re-point them
      // at the capture-rebound buffers this lambda's dcs now holds — the exact
      // bindings the legacy (post-rebind) copy stored.
      if (p2b->blasFirstDrawOfFrame) {
        RasterGeometry& cachedGeo = pBlas->input.geometryData;
        const RasterGeometry& liveGeo = drawCallState.geometryData;
        cachedGeo.positionBuffer     = liveGeo.positionBuffer;
        cachedGeo.normalBuffer       = liveGeo.normalBuffer;
        cachedGeo.texcoordBuffer     = liveGeo.texcoordBuffer;
        cachedGeo.texcoord1Buffer    = liveGeo.texcoord1Buffer;
        cachedGeo.color0Buffer       = liveGeo.color0Buffer;
        cachedGeo.indexBuffer        = liveGeo.indexBuffer;
        cachedGeo.sourceIsGpuCapture = liveGeo.sourceIsGpuCapture;
        cachedGeo.indexDataGpuStash  = liveGeo.indexDataGpuStash;
        cachedGeo.indexNeedsGpuStash = liveGeo.indexNeedsGpuStash;
      }
    } else if (m_drawCallCache.get(drawCallState, &pBlas) == DrawCallCache::CacheState::kExisted) {
      result = onSceneObjectUpdated(ctx, drawCallState, pBlas);
    } else {
      result = onSceneObjectAdded(ctx, drawCallState, pBlas);
    }
    s_pdcsGeomNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - tPdcsGeom0).count();

    assert(pBlas != nullptr);
    assert(result != ObjectCacheState::kInvalid);

    if (p2b == nullptr) {
    // Update the input state, so we always have a reference to the original draw call state
    pBlas->frameLastTouched = m_device->getCurrentFrameId();
    // NV-DXVK [ReapJoin]: same site, but a COUNT rather than a flag — see the
    // BlasEntry::noteDraw comment for why the flag cannot judge one instance of
    // a multi-copy mesh. This is the only place a draw is bound to an entry, so
    // it is the only place the count can be correct.
    pBlas->noteDraw(m_device->getCurrentFrameId());
    }  // NV-DXVK [Phase2b]: sharded draws were stamped in the ordered pre-pass.

    // NV-DXVK [ShipBake]: transforms feeding the hull (0x292b) geometry bake.
    // RESULT (don't redo): objectToWorld is IDENTITY in both visible AND vanish
    // frames; worldToView/objectToView are continuous across the vanish. So the
    // bake-transform path is NOT where the "vanish" comes from. Combined with the
    // interleaver DEAD-END note (skinning blend3 match=1) and the s_zigGunInstance
    // RE-TAG warning ([ZigGunRB] in rtx_instance_manager), the whole
    // bake/skinning/transform layer is verified clean. The "teleport" is a
    // re-tagging artifact of the ZigNDC probe and/or lives upstream in the
    // BLAS-merge / draw-call-cache. Kept as a cheap regression check only.
    {
      const uint32_t bvtx = drawCallState.getGeometryData().vertexCount;
      static uint32_t sBakeFrame = 0xffffffffu;
      static uint32_t sBakeCount = 0;
      const uint32_t bakeFid = m_device->getCurrentFrameId();
      if (bakeFid != sBakeFrame) { sBakeFrame = bakeFid; sBakeCount = 0; }
      // NV-DXVK [StudioModelHook] re-gate: fire on the actual Widow engine
      // model (precise, 1:1) instead of the shared VS hash 0x292b (~70
      // draws/frame, included sky/world/weapon). Cap raised to 16/frame to
      // cover all ~12 Widow sub-meshes. Requires rtx.tf2DetectWidow (or
      // tf2HideWidow/tf2IsolateWidow) so isWidowModel is populated.
      if (drawCallState.isWidowModel && sBakeCount < 16u) {
        ++sBakeCount;
        const auto& td = drawCallState.getTransformData();
        const Matrix4& o2w = td.objectToWorld;
        const Matrix4& w2v = td.worldToView;
        const Matrix4& o2v = td.objectToView;
        Logger::info(str::format(
          "[ShipBake] f=", bakeFid, " vtx=", bvtx,
          " numBones=", drawCallState.getSkinningState().numBones,
          " bpv=", uint32_t(drawCallState.getGeometryData().numBonesPerVertex),
          " result=", uint32_t(result), " blasUpd=", (pBlas->frameLastUpdated == pBlas->frameLastTouched ? 1 : 0),
          " wtvPathId=", td.worldToViewPathId,
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " o2vT=(", o2v[3][0], ",", o2v[3][1], ",", o2v[3][2], ")",
          " w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")"));
        Logger::info(str::format(
          "[ShipBake.o2w] f=", bakeFid,
          " r0=(", o2w[0][0], ",", o2w[0][1], ",", o2w[0][2], ")",
          " r1=(", o2w[1][0], ",", o2w[1][1], ",", o2w[1][2], ")",
          " r2=(", o2w[2][0], ",", o2w[2][1], ",", o2w[2][2], ")"));
      }
    }

    // Generate smooth normals for geometry that is flagged via the SmoothNormals texture category.
    // This is useful for older games where geometry may lack smooth normals, especially
    // when using the VertexShader Capture mechanism. The smooth normals are computed on the GPU
    // from the triangle mesh (area-weighted) and written into the normal buffer.
    // Only dispatch on BVH build/update — for static geometry, positions don't change so
    // the normals computed on the first pass remain valid for subsequent frames.
    if (drawCallState.shouldGenerateSmoothNormals() &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSmoothNormals(ctx, drawCallState.getGeometryData(), pBlas->modifiedGeometryData);
      pBlas->modifiedGeometryData.smoothNormalsApplied = true;
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
    }

    if (drawCallState.getSkinningState().numBones > 0 &&
        drawCallState.getGeometryData().numBonesPerVertex > 0 &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      // NV-DXVK TF2: dispatchSkinning is the LEGACY (D3D9/fixed-function)
      // skinning path. It asserts on the normal format being one of
      // R32G32B32_SFLOAT / R32G32B32A32_SFLOAT / R32_UINT and assumes
      // float blend weights. TF2's character meshes use packed formats
      // (normals as R10G10B10A2_UNORM = fmt 98, weights as R16G16_SINT
      // = fmt 82) that the legacy shader can't handle — it was only
      // reached previously when the bone-weight-format gate kept TF2
      // draws OUT of the skinning path entirely.
      //
      // After fixing that gate so TF2 skinned draws get numBones set
      // (required so the accel manager picks the DYNAMIC BLAS path,
      // which handles bone-driven per-frame refit correctly), this
      // line started triggering the assert for TF2 draws.
      //
      // For TF2 the INTERLEAVER (processGeometryInfo / dispatchInterleave)
      // already performs the bone-weighted skinning with the correct
      // decode for packed positions + weights, writing skinned world-
      // space verts into modifiedGeometryData. So dispatchSkinning here
      // would be a redundant second pass AND would corrupt verts by
      // re-transforming them.
      //
      // Gate on the normal format: if the game's normal buffer matches
      // one of the legacy-supported formats, dispatch (legacy skinning
      // path). Otherwise skip — the interleaver handled it.
      //
      // NV-DXVK TF2: VK_FORMAT_R32_UINT (98) was previously in this list
      // under the comment "R10G10B10A2_UNORM = fmt 98". That comment was
      // wrong — fmt 98 is R32_UINT (vulkan_core.h:1710); R10G10B10A2_UNORM
      // is fmt 64. Source-side normals are never octahedral (octahedral is
      // only the smooth-normals dispatch's OUTPUT format on the modified
      // buffer), so allowing R32_UINT here only ever matches TF2's
      // axis-dominant packed normals — which the legacy float-only
      // skinning shader corrupts. The interleaver already skins TF2 verts
      // via applyWeightedBones with the correct packed-format decode.
      const VkFormat normFmt = drawCallState.getGeometryData().normalBuffer.vertexFormat();
      const bool normalFormatSupported =
        normFmt == VK_FORMAT_R32G32B32_SFLOAT
        || normFmt == VK_FORMAT_R32G32B32A32_SFLOAT;
      if (normalFormatSupported) {
        m_device->getCommon()->metaGeometryUtils().dispatchSkinning(drawCallState, pBlas->modifiedGeometryData);
        pBlas->frameLastUpdated = pBlas->frameLastTouched;
      }
    }

    // NV-DXVK [RigidFinal]: the FINAL objectToWorld that becomes the TLAS instance,
    // AFTER all SubmitDraw transform patches — to tell whether a downstream patch
    // overrode the bone[0] transform on the instanced hull (rigidBakeBoneIndex==1)
    // vs the visible non-instanced hull (same studio name, marker 0). If the two
    // translations match the bug is geometry-space; if they diverge it's an override.
    {
      // NV-DXVK [perf] 2026-08-08e: skip the whole probe -- including the two
      // per-draw strstr scans over studioModelName, which ran for EVERY studio
      // draw on the CS thread -- once both of this frame's log slots are
      // consumed. The output (one instanced + one non-instanced line per
      // frame) is unchanged; only the per-draw name scan after the first two
      // matches of a frame is gone.
      const uint32_t rff = m_device->getCurrentFrameId();
      static uint32_t s_rfInst = UINT32_MAX, s_rfNon = UINT32_MAX;
      if (s_rfInst != rff || s_rfNon != rff) {
      const char* sn = drawCallState.studioModelName;
      const bool isHullName = sn[0] != '\0' &&
        (std::strstr(sn, "widow") != nullptr || std::strstr(sn, "Crow_dropship") != nullptr);
      const bool isInstancedHull = drawCallState.getRigidBakeBoneIndex() != 0;
      if (isHullName || isInstancedHull) {
        const auto& o = drawCallState.getTransformData().objectToWorld;
        // one instanced + one non-instanced line per frame
        uint32_t& slot = isInstancedHull ? s_rfInst : s_rfNon;
        if (slot != rff) {
          slot = rff;
          Logger::warn(str::format(
            "[RigidFinal] f=", rff, " instanced=", (isInstancedHull ? 1 : 0),
            " name=", sn,
            " finalO2W.T=(", float(o[3][0]), ",", float(o[3][1]), ",", float(o[3][2]), ")",
            " verts=", drawCallState.getGeometryData().vertexCount));
        }
      }
      }  // frame-slot fast skip
    }

    // NV-DXVK [MeshTrace] funnel stage 3: a BlasEntry exists for this draw.
    meshtrace::record(
      static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash),
      drawCallState.getGeometryData().vertexCount,
      meshtrace::Stage::BlasReady);

    // NV-DXVK [fanout split]: a game-submitted bone-fanout draw carries N
    // independent props whose membership churns every frame, so it resolves to N
    // RtInstances rather than one. Conditions, all required:
    //   - the option is on (off = the pre-split single-instance behaviour, so the
    //     ~5x CPU instance count can be A/B'd against it without a rebuild)
    //   - the draw is a game fanout batch, not a USD PointInstancer or the
    //     external-GPU-instancing path (see DrawCallTransforms::isFanoutBatch)
    //   - no existingInstance: a replacement sub-prim must resolve to exactly the
    //     RtInstance the ReplacementInstance already stored
    //   - more than one placement — a batch of one has nothing to split
    const DrawCallTransforms& pdcsTransforms = drawCallState.getTransformData();
    const bool splitFanout =
      RtxOptions::splitFanoutInstances()
      && pdcsTransforms.isFanoutBatch
      && existingInstance == nullptr
      && pdcsTransforms.instancesToObject != nullptr
      && pdcsTransforms.instancesToObject->size() > 1;

    // Reused across draws so the split costs no per-draw allocation. thread_local
    // rather than a member because processDrawCallState is reached from the
    // draw-submission path and the instance manager's own probes already assume
    // per-thread state.
    static thread_local std::vector<RtInstance*> sFanoutInstances;

    // Note: The material data can be modified in instance manager
    const auto tPdcsInst0 = std::chrono::steady_clock::now();
    RtInstance* instance = nullptr;
    if (p2b != nullptr) {
      // NV-DXVK [Phase2b]: the find/update work ran in the flush-side shard (or
      // its ordered-tail continuation). Consume the produced instances, then
      // replay the CS-domain residue the workers recorded: the surface buffer
      // rebind (only valid after processGeometryInfo's updateBufferCache above),
      // the billboard stage (positional buffer reads + m_billboards appends),
      // the deferred OMM callbacks (with the REAL billboard outcome), and
      // ray-portal registration.
      sFanoutInstances.assign(p2b->instances.begin(), p2b->instances.end());
      instance = sFanoutInstances.empty() ? nullptr : sFanoutInstances[0];
      for (ShardedDrawInfo::PendingInstanceOps& ops : p2b->pendingOps) {
        if (ops.instance == nullptr) {
          continue;
        }
        if (ops.bindBuffers) {
          m_instanceManager.bindInstanceBuffersFromBlas(*pBlas, *ops.instance);
        }
        bool billboardsGotGenerated = false;
        if (ops.billboard) {
          billboardsGotGenerated = m_instanceManager.runBillboardStage(
            *ops.instance, m_cameraManager.getMainCamera().getDirection(false));
        }
        if (ops.omm || (billboardsGotGenerated && RtxOptions::getEnableOpacityMicromap())) {
          m_instanceManager.fireDeferredOmmCallbacks(
            *ops.instance, drawCallState, renderMaterialData,
            ops.evHasTransformChanged, ops.evHasPreviousPositions, ops.evIsFirstUpdateThisFrame);
        }
        if (ops.rayPortal) {
          uint32_t portalMatIndex = UINT32_MAX;
          const RtSurfaceMaterial& portalMat =
            createSurfaceMaterial(renderMaterialData, drawCallState, &portalMatIndex);
          m_rayPortalManager.processRayPortalData(*ops.instance, portalMat);
        }
      }
    } else if (splitFanout) {
      m_instanceManager.processSceneObjectFanout(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, &m_drawCallCache, sFanoutInstances);
      // The first prop stands in for the draw where a single instance is all the
      // interface allows (the ReplacementInstance identity check, which a fanout
      // draw never reaches). Everything that can legitimately act on all of them
      // iterates the full list below instead.
      instance = sFanoutInstances.empty() ? nullptr : sFanoutInstances[0];
    } else {
      sFanoutInstances.clear();
      instance = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, existingInstance, &m_drawCallCache);
      if (instance != nullptr) {
        sFanoutInstances.push_back(instance);
      }
    }
    s_pdcsInstNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - tPdcsInst0).count();

    // NV-DXVK [MeshTrace] funnel stage 4. InstanceNull is the interesting one:
    // the geometry existed and still produced no instance.
    meshtrace::record(
      static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash),
      drawCallState.getGeometryData().vertexCount,
      instance != nullptr ? meshtrace::Stage::InstanceMade : meshtrace::Stage::InstanceNull);

    // NV-DXVK [ResidentScene] slice 6, CS half: PUBLISH WHAT THIS DRAW RESOLVED
    // TO, so a later frame's gate hit has something to keep alive.
    //
    // HERE AND NOWHERE ELSE. sFanoutInstances is the complete produced list on
    // all three routes above -- the sharded consume, the fanout split and the
    // single-instance case all leave it final at this point -- and this is the
    // first line after which that is true. Anything earlier would record a
    // partial list, and a record with a hole in it stamps some of an object's
    // instances and lets the rest retire, which is the failure with no FAIL to
    // catch it.
    //
    // SCORE BEFORE BUILD. score() compares the record as it stands against what
    // the full path just produced, so it has to run while the record still
    // describes the PREVIOUS resolution. build() overwrites it.
    if (RtxOptions::ResidentScene::enable() && drawCallState.residentKey != 0ull) {
      ResidentScene& residentScene = m_instanceManager.getResidentScene();
      if (RtxOptions::ResidentScene::verify() && drawCallState.residentPredictHit) {
        residentScene.score(drawCallState.residentKey, sFanoutInstances);
      }
      // An empty list is not a record. A draw that produced no instance has
      // nothing to keep alive, and filing it would make the gate serve an empty
      // touch forever -- which reads as a hit and does nothing, the worst of
      // both. build() rejects it below; this test only saves the call.
      if (!sFanoutInstances.empty()) {
        residentScene.build(drawCallState.residentKey,
                            drawCallState.residentGenHash,
                            drawCallState.residentSrcVertexBuffer,
                            drawCallState.residentSrcIndexBuffer,
                            m_device->getCurrentFrameId(),
                            sFanoutInstances);
      }
    }

    // Check if a light should be created for this Material
    // NV-DXVK [fanout split]: once per DRAW, not once per instance. createEffectLight
    // derives the light position from the draw call's own geometry centroid and
    // objectToView and does not read the instance transform at all, so calling it
    // per split prop would stack N coincident lights rather than distribute them.
    if (instance && RtxOptions::shouldConvertToLight(drawCallState.getMaterialData().getHash())) {
      createEffectLight(ctx, drawCallState, instance);
    }

    const bool objectPickingActive = m_device->getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();

    if (objectPickingActive && instance && g_allowMappingLegacyHashToObjectPickingValue) {
      auto meta = DrawCallMetaInfo {};
      {
        XXH64_hash_t h;
        h = drawCallState.getMaterialData().getColorTexture().getImageHash();
        if (h != kEmptyHash) {
          meta.legacyTextureHash = h;
        }
        h = drawCallState.getMaterialData().getColorTexture2().getImageHash();
        if (h != kEmptyHash) {
          meta.legacyTextureHash2 = h;
        }
      }

      {
        std::lock_guard lock { m_drawCallMeta.mutex };
        // NV-DXVK [fanout split]: every instance, not just the representative.
        // objectPickingValue is per-RtInstance, so registering only the first
        // would leave the other ~53 props of a split batch unpickable — the exact
        // silent breakage that made the per-prop-instance API worth having.
        // Non-split draws contribute a single entry here, as before.
        for (RtInstance* pickInstance : sFanoutInstances) {
          auto [iter, isNew] = m_drawCallMeta.infos[m_drawCallMeta.ticker].emplace(pickInstance->surface.objectPickingValue, meta);
          ONCE_IF_FALSE(isNew, Logger::warn(
            "Found multiple draw calls with the same \'objectPickingValue\'. "
            "Ignoring further MetaInfo-s, some objects might be not be available through object picking"));
        }
      }
    }

    // Priority ordering for particle system descriptors is: Mesh, Material, Texture.  This matches the implementation in toolkit.
    // By this point, pParticleSystemDesc will contain the information from a mesh replacement (if one exists), so we just handle
    // materials replacements, and texture taggin categories below.
    RtxParticleSystemDesc globalParticleDesc; // Storage for global desc if needed
    if (!pParticleSystemDesc) {
      pParticleSystemDesc = renderMaterialData.getParticleSystemDesc();
    }
    if (!pParticleSystemDesc && drawCallState.categories.test(InstanceCategories::ParticleEmitter)) {
      globalParticleDesc = RtxParticleSystemManager::createGlobalParticleSystemDesc();
      pParticleSystemDesc = &globalParticleDesc;
    }
    if (instance && pParticleSystemDesc) {
      RtxParticleSystemManager& particleSystem = device()->getCommon()->metaParticleSystem();
      // NV-DXVK [fanout split]: spawn once per DRAW. The particle count comes from
      // the draw call (getNumberOfParticlesToSpawn(drawCallState)) and the batch
      // emits as one emitter, so calling this per split prop would multiply the
      // emission rate by the batch size.
      particleSystem.spawnParticles(ctx.ptr(), *pParticleSystemDesc, instance->getVectorIdx(), drawCallState, renderMaterialData);

      if (pParticleSystemDesc->hideEmitter) {
        // ...but hiding IS per instance: leaving the other props visible would
        // show the emitter geometry the desc asked to hide.
        for (RtInstance* emitterInstance : sFanoutInstances) {
          emitterInstance->setHidden(true);
        }
      }
    }

    if (instance) {
      ++vanishDiag().drawsKept;
      const XXH64_hash_t vsHash = drawCallState.getTransformData().vertexShaderHash;
      ++vanishDiag().vsHistogram[vsHash];
    }

    return instance;
  }

  // ======================================================================
  // NV-DXVK [Phase2b]: THE FLUSH-SIDE DRIVER — THE_OPTIMISATION_PLAN_2.md
  // Steps 1-6. Runs on the game thread inside flushGeometryBatch, after the
  // caller drained the CS thread and joined Phase B. See the .h declaration
  // and PHASE2B_IMPLEMENTATION_SPEC.md for the architecture.
  // ======================================================================
  void SceneManager::processDeferredDrawBatch(std::vector<ShardedDrawBatchItem>& batch, const ShardScheduleFn& schedule, uint32_t maxTasks) {
    ScopedCpuProfileZone();
    const uint32_t fid = m_device->getCurrentFrameId();
    const auto t2b0 = std::chrono::steady_clock::now();

    // Per-BLAS shards: item indices in arena order. thread_local so capacity is
    // reused frame-to-frame — this runs only on the game thread.
    static thread_local std::vector<std::vector<uint32_t>> sShards;
    static thread_local std::unordered_map<BlasEntry*, uint32_t> sShardOf;
    for (auto& s : sShards) { s.clear(); }
    sShardOf.clear();
    uint32_t liveShards = 0;

    uint32_t nSharded = 0, nLegacy = 0, nIgnored = 0;

    // NV-DXVK [Shard2b]: legacy BY REASON. `legacy=` alone says the win is
    // capped without saying by what, and the reasons have completely different
    // remedies: terrain must stay on CS (it records GPU work), sky is routed on
    // a four-way over-approximation that may be shrinkable, replacements are a
    // separate sharding job, and unready/unknown-camera/invalid are draws the
    // legacy path itself treats specially. First-match attribution, because
    // that is what actually chose the route; `repl` is tested later, after the
    // material work, so it is its own bucket.
    uint32_t nLgAdmit = 0, nLgFut = 0, nLgFinal = 0, nLgSky = 0;
    uint32_t nLgTerrain = 0, nLgCam = 0, nLgGeom = 0, nLgRepl = 0;

    // NV-DXVK [Shard2b]: pre-pass STAGE split. preUs is the new serial cost on
    // the game thread and the post-2b plan's Step 1 is "move four of these six
    // into Phase B" — which four is not answerable from one aggregate (plan
    // rule R9). Same gate as [Perf.GeoSplit] / [Perf.GeoChurn]: ~7 extra clock
    // reads per draw is ~0.4 ms/frame, which is a large fraction of the very
    // thing being measured, so it stays OFF unless you are bisecting.
    const bool preSplitOn = RtxOptions::logPrepSceneSplit();
    int64_t preFinNs = 0, preCamNs = 0, preFogNs = 0, preHashNs = 0, preMatNs = 0, preGetNs = 0;
    const auto preNow = [preSplitOn]() {
      return preSplitOn ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
    };
    const auto preAdd = [preSplitOn](int64_t& acc, const std::chrono::steady_clock::time_point& t0) {
      if (preSplitOn) {
        acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
      }
    };

    // ---- ORDERED PRE-PASS (Step 1), arena order --------------------------
    const bool shardingAdmissible =
      !RtxOptions::enableInstanceDebuggingTools()
      // Conservative overflow admission: the buffer table's slot count is
      // written on the CS record step, so at flush time only last frame's final
      // size predicts the cliff. Within 1/8th of the limit, route everything
      // legacy — the legacy path keeps its exact per-draw overflow check.
      && (m_bufferCacheLastFrameCount + (m_bufferCacheLastFrameCount >> 3)) < kBufferCacheLimit;

    for (ShardedDrawBatchItem& item : batch) {
      DrawCallState& dcs = *item.dcs;
      ShardedDrawInfo& info = *item.info;

      // Camera classification runs for EVERY item in arena order — the exact
      // CameraManager op order the CS path produced — and exactly once
      // (commitGeometryToRT skips it when cameraDone is set).
      // The finalize below must precede it only in that both precede any
      // consumer; finalize first matches the CS order (finalize -> classify).
      const bool futuresPending =
        dcs.geometryData.futureGeometryHashes.valid() || dcs.futureMaterialData.valid()
        || dcs.geometryData.futureBoundingBox.valid() || dcs.futureSkinningData.valid();

      bool finalized = false;
      if (!futuresPending) {
        // Idempotent under the CS-side re-run at commitGeometryToRT:5270 — all
        // futures are invalid (batched stages filled the fields directly), so
        // the re-run recombines the same hashes and re-ORs the same categories.
        const auto tPre = preNow();
        finalized = dcs.finalizePendingFutures(nullptr);
        preAdd(preFinNs, tPre);
      }

      // Classify ONLY finalize-ready draws — the CS path classifies inside its
      // own futuresReady gate, so an unready draw never classifies today and
      // must not start doing so here. Unready draws go legacy with
      // cameraDone=false; the CS side then does exactly what it always did.
      if (finalized) {
        const auto tPre = preNow();
        dcs.cameraType = m_cameraManager.processCameraData(dcs);
        preAdd(preCamNs, tPre);
        info.cameraDone = true;
      }

      // ---- ROUTE. Everything the shard path cannot prove safe takes the
      // UNCHANGED legacy CS path, in original draw order. Over-approximations
      // are deliberate: a legacy-routed draw is always exactly correct.
      const bool skyLike =
        dcs.cameraType == CameraType::Sky
        || dcs.getTransformData().isSubViewSkybox
        || dcs.testCategoryFlags(InstanceCategories::Sky)
        || dcs.skyAutoDetected;
      const bool terrainLike = dcs.testCategoryFlags(InstanceCategories::Terrain);
      const bool unknownCamera = dcs.cameraType == CameraType::Unknown;
      const RasterGeometry& geo = dcs.getGeometryData();
      const bool invalidGeom = !geo.positionBuffer.defined() || geo.vertexCount == 0;

      if (!shardingAdmissible || futuresPending || !finalized || skyLike || terrainLike
          || unknownCamera || invalidGeom) {
        info.route = ShardedDrawInfo::Route::kLegacyCS;
        ++nLegacy;
        // First-match attribution, in the same order the test above short-
        // circuits, so the bucket names the condition that actually decided it.
        if (!shardingAdmissible)  { ++nLgAdmit; }
        else if (futuresPending)  { ++nLgFut; }
        else if (!finalized)      { ++nLgFinal; }
        else if (skyLike)         { ++nLgSky; }
        else if (terrainLike)     { ++nLgTerrain; }
        else if (unknownCamera)   { ++nLgCam; }
        else                      { ++nLgGeom; }
        continue;
      }

      // ---- submitDrawState's entry/hash/material stages, hoisted (the CS slim
      // path skips them — see the [Phase2b] block in submitDrawState).
      // Fog block first, exactly like submitDrawState:2452.
      const auto tPreFog = preNow();
      if (dcs.getFogState().mode != FogMode::None) {
        const XXH64_hash_t fogHash = dcs.getFogState().getHash();
        if (m_fogStates.find(fogHash) == m_fogStates.end()) {
          m_fogStates[fogHash] = dcs.getFogState();
          MaterialData* pFogReplacement = m_pReplacer->getReplacementMaterial(fogHash);
          if (pFogReplacement) {
            trackReplacementMaterialHash(fogHash);
            if (pFogReplacement->getType() != MaterialDataType::Translucent) {
              Logger::warn(str::format("Fog replacement materials must be translucent.  Ignoring material for ", std::hex, m_fog.getHash()));
            } else {
              uint32_t id = UINT32_MAX;
              createSurfaceMaterial(*pFogReplacement, dcs, &id);
              m_startInMediumMaterialIndex_inCache = id;
            }
          } else if (m_fog.mode == FogMode::None) {
            m_fog = dcs.getFogState();
          }
        }
      }

      preAdd(preFogNs, tPreFog);

      // Replacement lookup — the same three probes submitDrawState makes. Any
      // hit routes legacy (drawReplacements builds multi-prim instances and
      // honours existingInstance; out of shard scope by design).
      const auto tPreHash = preNow();
      const XXH64_hash_t activeReplacementHash = dcs.getHash(RtxOptions::geometryAssetHashRule());
      trackMeshHash(activeReplacementHash);
      std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForMesh(activeReplacementHash);
      if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0 && !pReplacements) {
        const XXH64_hash_t legacyHash = dcs.getHashLegacy(rules::LegacyAssetHash0);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
      }
      if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1 && !pReplacements) {
        const XXH64_hash_t legacyHash = dcs.getHashLegacy(rules::LegacyAssetHash1);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
      }
      preAdd(preHashNs, tPreHash);

      if (pReplacements != nullptr) {
        info.route = ShardedDrawInfo::Route::kLegacyCS;
        ++nLegacy;
        ++nLgRepl;
        continue;
      }

      // Material — computed ONCE, here; the sidecar copy is what both the
      // flush-side instance work and every CS-side consumer read.
      const auto tPreMat = preNow();
      info.renderMaterial = std::make_shared<MaterialData>(determineMaterialData(nullptr, dcs));
      preAdd(preMatNs, tPreMat);

      info.route = ShardedDrawInfo::Route::kSharded;

      if (info.renderMaterial->getIgnored()) {
        // Matches processDrawCallState's pre-cache early return: no cache
        // touch, no stamps, no instance work. The CS slim path re-checks the
        // sidecar material and returns the same way.
        ++nIgnored;
        continue;
      }

      // ---- Step 1 proper: resolve the BlasEntry, capture first-of-frame
      // BEFORE stamping (onSceneObjectUpdated's same-frame-dup test reads
      // pre-stamp state), stamp, and assign the shard.
      const auto tPreGet = preNow();
      BlasEntry* pBlas = nullptr;
      m_drawCallCache.get(dcs, &pBlas);
      info.pBlas = pBlas;
      info.blasFirstDrawOfFrame = (pBlas->frameLastTouched != fid);
      pBlas->frameLastTouched = fid;
      pBlas->noteDraw(fid);
      preAdd(preGetNs, tPreGet);

      const uint32_t itemIdx = static_cast<uint32_t>(&item - batch.data());
      auto shardIt = sShardOf.find(pBlas);
      if (shardIt == sShardOf.end()) {
        if (liveShards == sShards.size()) {
          sShards.emplace_back();
        }
        shardIt = sShardOf.emplace(pBlas, liveShards++).first;
      }
      sShards[shardIt->second].push_back(itemIdx);
      ++nSharded;
    }

    const auto t2bPre = std::chrono::steady_clock::now();

    // ---- PARALLEL BY SHARD (Steps 2-5) -----------------------------------
    // The shard is the OWNERSHIP unit (one BLAS, its items in arena order); the
    // BUNDLE is the scheduling unit. Scheduling one task per shard would issue
    // ~460 Schedule() calls per frame, and plan Sec 13 measured this pool
    // collapsing at 119 (frames/3s 54 -> 17) — each call is a mutex round trip
    // plus a notify_one into 30 condvar-sleeping workers, and that cost is per
    // TASK, not per unit of work. Packing whole shards into maxTasks bundles
    // keeps exclusive BLAS ownership exactly (a BLAS never spans two bundles)
    // at Phase B's proven one-task-per-worker scale.
    //
    // Bundles are contiguous runs of shards balanced by ITEM COUNT, which also
    // keeps each bundle's arena accesses roughly contiguous. Item count is the
    // only cost proxy available here — plan Sec 13 records single draws costing
    // milliseconds, so a bundle can still straggle; that is what the pool's
    // work stealing and the tail timer are for, and it is strictly better than
    // paying 460 task setups to find out.
    static thread_local std::vector<Future<void>> sShardFuts;
    static thread_local std::vector<uint32_t> sBundleStart;  // shard indices, + end sentinel
    sShardFuts.clear();
    sBundleStart.clear();

    if (liveShards > 0) {
      const uint32_t bundles = std::max(1u, std::min(liveShards, maxTasks));
      const uint32_t targetItems = (nSharded + bundles - 1u) / bundles;
      uint32_t acc = 0;
      sBundleStart.push_back(0u);
      for (uint32_t s = 0; s < liveShards; ++s) {
        acc += static_cast<uint32_t>(sShards[s].size());
        // Cut only AFTER consuming a shard and never at the very end, so no
        // bundle is ever empty; stop cutting once the budget is spent.
        if (acc >= targetItems && (s + 1u) < liveShards
            && static_cast<uint32_t>(sBundleStart.size()) < bundles) {
          sBundleStart.push_back(s + 1u);
          acc = 0;
        }
      }
      sBundleStart.push_back(liveShards);
    }

    const uint32_t nBundles = sBundleStart.empty()
      ? 0u : static_cast<uint32_t>(sBundleStart.size() - 1u);
    for (uint32_t b = 0; b < nBundles; ++b) {
      const uint32_t s0 = sBundleStart[b];
      const uint32_t s1 = sBundleStart[b + 1u];
      // sShards is a game-thread thread_local of static duration and is not
      // touched again until the join, so a pointer into it is safe on a worker.
      std::vector<std::vector<uint32_t>>* pShards = &sShards;
      std::vector<ShardedDrawBatchItem>* pBatch = &batch;
      SceneManager* self = this;
      Future<void> f = schedule([self, pShards, pBatch, s0, s1]() {
        for (uint32_t s = s0; s < s1; ++s) {
          for (const uint32_t idx : (*pShards)[s]) {
            self->runShardedDrawItem((*pBatch)[idx]);
          }
        }
      });
      if (f.valid()) {
        sShardFuts.push_back(f);
      } else {
        for (uint32_t s = s0; s < s1; ++s) {
          for (const uint32_t idx : sShards[s]) {
            runShardedDrawItem(batch[idx]);
          }
        }
      }
    }
    for (auto& f : sShardFuts) {
      f.get();
    }
    sShardFuts.clear();

    const auto t2bPar = std::chrono::steady_clock::now();

    // ---- ORDERED TAIL (Step 6), arena order ------------------------------
    uint32_t nDeferred = 0;
    for (ShardedDrawBatchItem& item : batch) {
      if (item.info->route == ShardedDrawInfo::Route::kSharded && item.info->pBlas != nullptr) {
        if (item.info->needsTailContinuation || !item.info->deferredPlacements.empty()
            || !item.info->spatialOps.empty()) {
          if (item.info->needsTailContinuation || !item.info->deferredPlacements.empty()) {
            ++nDeferred;
          }
          runShardedDrawTail(item);
        }
      }
    }

    // ---- [Shard2b] heartbeat (3s window) ---------------------------------
    {
      const auto t2b1 = std::chrono::steady_clock::now();
      static thread_local std::chrono::steady_clock::time_point sLast {};
      static thread_local bool sInit = false;
      static thread_local uint64_t sFrames = 0, sSharded = 0, sLegacy = 0, sIgnored = 0;
      static thread_local uint64_t sDeferred = 0, sShardCnt = 0, sBundleCnt = 0;
      static thread_local int64_t sPreNs = 0, sParNs = 0, sTailNs = 0;
      // NV-DXVK [Shard2b]: legacy-by-reason and pre-pass-stage accumulators.
      static thread_local uint64_t sLgAdmit = 0, sLgFut = 0, sLgFinal = 0, sLgSky = 0;
      static thread_local uint64_t sLgTerrain = 0, sLgCam = 0, sLgGeom = 0, sLgRepl = 0;
      static thread_local int64_t sPreFinNs = 0, sPreCamNs = 0, sPreFogNs = 0;
      static thread_local int64_t sPreHashNs = 0, sPreMatNs = 0, sPreGetNs = 0;
      if (!sInit) { sLast = t2b0; sInit = true; }
      ++sFrames;
      sSharded += nSharded; sLegacy += nLegacy; sIgnored += nIgnored;
      sDeferred += nDeferred; sShardCnt += liveShards; sBundleCnt += nBundles;
      sLgAdmit += nLgAdmit; sLgFut += nLgFut; sLgFinal += nLgFinal; sLgSky += nLgSky;
      sLgTerrain += nLgTerrain; sLgCam += nLgCam; sLgGeom += nLgGeom; sLgRepl += nLgRepl;
      sPreFinNs += preFinNs; sPreCamNs += preCamNs; sPreFogNs += preFogNs;
      sPreHashNs += preHashNs; sPreMatNs += preMatNs; sPreGetNs += preGetNs;
      sPreNs  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2bPre - t2b0).count();
      sParNs  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2bPar - t2bPre).count();
      sTailNs += std::chrono::duration_cast<std::chrono::nanoseconds>(t2b1 - t2bPar).count();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(t2b1 - sLast).count() >= 3000) {
        const int64_t fr = sFrames ? static_cast<int64_t>(sFrames) : 1;
        // Sec-15 reads: shards ~= uniqueBlas on [Perf.Report]; deferred ~0 in
        // steady state (addedPct=0); legacy = sky/terrain/replacement volume.
        // sharded+legacy+ignored == itemsPerFrame on [BatchSubmitDraw] (the
        // plan's "sum of shard sizes == drawsCommit" does NOT hold: legacy and
        // ignored draws are deliberately outside every shard).
        // bundles = tasks actually scheduled; it must track the worker count,
        // NOT shards — if it ever approaches shards, the Sec-13 tombstone is
        // back and the pool is paying ~460 task setups a frame.
        Logger::info(str::format(
          "[Shard2b] frames=", sFrames,
          " sharded=", sSharded / fr,
          " legacy=", sLegacy / fr,
          " ignored=", sIgnored / fr,
          " shards=", sShardCnt / fr,
          " bundles=", sBundleCnt / fr,
          " deferred=", sDeferred / fr,
          " | preUs=", sPreNs / 1000 / fr,
          " parUs=", sParNs / 1000 / fr,
          " tailUs=", sTailNs / 1000 / fr));

        // WHY the legacy draws are legacy. terrain is permanent (it records GPU
        // work); sky is the four-way over-approximation and the one worth
        // attacking; repl is a separate sharding job; admit>0 means the
        // whole-frame buffer-cache gate fired and NOTHING was sharded that
        // frame, which would make every other number here meaningless.
        if (sLegacy != 0) {
          Logger::info(str::format(
            "[Shard2b.legacy] perFrame admit=", sLgAdmit / uint64_t(fr),
            " unreadyFutures=", sLgFut / uint64_t(fr),
            " notFinalized=", sLgFinal / uint64_t(fr),
            " sky=", sLgSky / uint64_t(fr),
            " terrain=", sLgTerrain / uint64_t(fr),
            " unknownCam=", sLgCam / uint64_t(fr),
            " invalidGeom=", sLgGeom / uint64_t(fr),
            " replacement=", sLgRepl / uint64_t(fr),
            "  (total=", sLegacy / uint64_t(fr), ")"));
        }

        // Which of the six ordered stages preUs actually is. Silent unless
        // rtx.logPrepSceneSplit is on. cam+get are structurally ordered (camera
        // op order / cache mutation); fin+fog+hash+mat are the candidates for
        // Phase B, and this line says whether moving them is worth the churn.
        const int64_t preSum = sPreFinNs + sPreCamNs + sPreFogNs + sPreHashNs + sPreMatNs + sPreGetNs;
        if (preSum != 0) {
          Logger::info(str::format(
            "[Shard2b.pre] perFrameUs finalize=", sPreFinNs / 1000 / fr,
            " camera=", sPreCamNs / 1000 / fr,
            " fog=", sPreFogNs / 1000 / fr,
            " hash+repl=", sPreHashNs / 1000 / fr,
            " material=", sPreMatNs / 1000 / fr,
            " cacheGet=", sPreGetNs / 1000 / fr,
            " | sum=", preSum / 1000 / fr, " of preUs=", sPreNs / 1000 / fr,
            "  MOVABLE(fin+fog+hash+mat)=",
            (sPreFinNs + sPreFogNs + sPreHashNs + sPreMatNs) / 1000 / fr, "us"));
        }

        sLast = t2b1;
        sFrames = sSharded = sLegacy = sIgnored = sDeferred = sShardCnt = sBundleCnt = 0;
        sPreNs = sParNs = sTailNs = 0;
        sLgAdmit = sLgFut = sLgFinal = sLgSky = sLgTerrain = sLgCam = sLgGeom = sLgRepl = 0;
        sPreFinNs = sPreCamNs = sPreFogNs = sPreHashNs = sPreMatNs = sPreGetNs = 0;
      }
    }
  }

  // NV-DXVK [Phase2b]: one arena item's shard-task body — geom decide + CPU
  // bookkeeping (the flush-side halves of onSceneObject{Added,Updated}) and the
  // instance work, under exclusive BLAS ownership. Runs on a worker (or the
  // game thread via the inline fallback).
  void SceneManager::runShardedDrawItem(ShardedDrawBatchItem& item) {
    DrawCallState& dcs = *item.dcs;
    ShardedDrawInfo& info = *item.info;
    BlasEntry* pBlas = info.pBlas;
    const uint32_t fid = m_device->getCurrentFrameId();

    // ---- geom decide + apply (Step 5) ------------------------------------
    if (info.blasFirstDrawOfFrame) {
      // A blas created by this frame's pre-pass has frameCreated == fid; the
      // second-and-later draws of it are not blasFirstDrawOfFrame, so this
      // conjunction is exactly DrawCallCache::get's kNew for this draw.
      const bool wasNew = (pBlas->frameCreated == fid);
      const ObjectCacheState r = wasNew
        ? computeGeometryCacheState<true>(dcs, pBlas->modifiedGeometryData)
        : computeGeometryCacheState<false>(dcs, pBlas->modifiedGeometryData);
      info.geomResult = static_cast<int8_t>(r);
      if (!wasNew) {
        // onSceneObjectUpdated's flush-side halves. The input copy is
        // pre-rebind; the CS record step re-points its buffer fields at the
        // capture (see processDrawCallState's fixup).
        pBlas->clearMaterialCache();
        pBlas->input = dcs;
      }
    } else {
      // Same-frame duplicate: onSceneObjectUpdated's touched-fast path.
      info.geomResult = static_cast<int8_t>(ObjectCacheState::kUpdateInstance);
      pBlas->cacheMaterial(dcs.getMaterialData());
    }

    // ---- instance work (Steps 3-4) ---------------------------------------
    t_shardPhase.info = &info;
    t_shardPhase.deferredThisDraw = false;
    t_shardPhase.allowMiss = false;
    t_shardPhase.currentOps = nullptr;

    static thread_local std::vector<RtInstance*> sShardInstances;
    sShardInstances.clear();

    const DrawCallTransforms& tf = dcs.getTransformData();
    const bool splitFanout =
      RtxOptions::splitFanoutInstances()
      && tf.isFanoutBatch
      && tf.instancesToObject != nullptr
      && tf.instancesToObject->size() > 1;

    if (splitFanout) {
      m_instanceManager.processSceneObjectFanout(m_cameraManager, m_rayPortalManager,
        *pBlas, dcs, *info.renderMaterial, &m_drawCallCache, sShardInstances);
    } else {
      RtInstance* inst = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager,
        *pBlas, dcs, *info.renderMaterial, nullptr, &m_drawCallCache);
      if (inst != nullptr) {
        sShardInstances.push_back(inst);
      } else if (t_shardPhase.deferredThisDraw) {
        info.needsTailContinuation = true;
      }
    }
    info.instances.assign(sShardInstances.begin(), sShardInstances.end());

    t_shardPhase = ShardedInstancePhase {};
  }

  // NV-DXVK [Phase2b]: the ordered-tail body for one item — apply the
  // parallel-phase deferred ops, then run any miss continuation in allowMiss
  // mode (creation/migration/map writes inline; CS-domain work still recorded
  // into pendingOps for the record step), then apply what the continuation
  // recorded. Single-threaded, arena order.
  void SceneManager::runShardedDrawTail(ShardedDrawBatchItem& item) {
    DrawCallState& dcs = *item.dcs;
    ShardedDrawInfo& info = *item.info;

    for (const DeferredSpatialOp& op : info.spatialOps) {
      m_instanceManager.applyDeferredSpatialOp(op);
    }
    info.spatialOps.clear();

    if (!info.needsTailContinuation && info.deferredPlacements.empty()) {
      return;
    }

    t_shardPhase.info = &info;
    t_shardPhase.deferredThisDraw = false;
    t_shardPhase.allowMiss = true;
    t_shardPhase.currentOps = nullptr;

    if (info.needsTailContinuation) {
      info.needsTailContinuation = false;
      RtInstance* inst = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager,
        *info.pBlas, dcs, *info.renderMaterial, nullptr, &m_drawCallCache);
      if (inst != nullptr) {
        info.instances.insert(info.instances.begin(), inst);
      }
    }
    if (!info.deferredPlacements.empty()) {
      m_instanceManager.processDeferredFanoutPlacements(m_cameraManager, m_rayPortalManager,
        *info.pBlas, dcs, *info.renderMaterial, &m_drawCallCache,
        info.deferredPlacements, info.instances);
      info.deferredPlacements.clear();
    }

    t_shardPhase = ShardedInstancePhase {};

    // Ops the continuation recorded in allowMiss mode are CS-domain only
    // (buffer binds / billboards / OMM live in pendingOps, replayed on CS);
    // map writes ran inline. Anything that still landed here applies now.
    for (const DeferredSpatialOp& op : info.spatialOps) {
      m_instanceManager.applyDeferredSpatialOp(op);
    }
    info.spatialOps.clear();
  }

  // NV-DXVK [Phase2b prerequisite]: see the SurfaceMaterialMemo block in the header.
  uint32_t SceneManager::surfaceMaterialMemoSlot() {
    thread_local uint32_t sSlot = UINT32_MAX;
    if (sSlot == UINT32_MAX) {
      const uint32_t claimed = m_surfaceMaterialMemoNextSlot.fetch_add(1u, std::memory_order_relaxed);
      sSlot = std::min(claimed, kSurfaceMaterialMemoSlots - 1u);
    }
    return sSlot;
  }

  void SceneManager::invalidateSurfaceMaterialMemo() {
    for (uint32_t i = 0; i < kSurfaceMaterialMemoSlots; ++i)
      m_lastSurfaceMaterial[i].valid = false;
  }

  const RtSurfaceMaterial& SceneManager::createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                               const DrawCallState& drawCallState,
                                                               uint32_t* out_indexInCache) {
    ScopedCpuProfileZone();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();
    const auto renderMaterialDataType = renderMaterialData.getType();

    // NV-DXVK [perf]: single-entry memo. See SurfaceMaterialMemo in the header
    // for why this is exact rather than a heuristic -- in short, it keys on
    // strictly more state than the preCreationHash below does, so it can only
    // ever serve what that map would have served. Every field read here is a
    // plain load; none of them is the sampler resolution or the hashing this
    // skips.
    const uint32_t     memoFrameId         = m_device->getCurrentFrameId();
    const XXH64_hash_t memoMaterialHash    = renderMaterialData.getHash();
    const void* const  memoSamplerOverride = renderMaterialData.getSamplerOverride().ptr();
    const void* const  memoDrawSampler     = drawCallState.getMaterialData().getSampler().ptr();
    const void* const  memoDrawSampler2    = drawCallState.getMaterialData().getSampler2().ptr();
    const bool         memoIsEye           = drawCallState.isEye();
    const bool         memoIsRtRenderTarget = drawCallState.isUsingRaytracedRenderTarget;

    // This thread's own slot -- no other thread reads or writes it, so the fields
    // below cannot be a torn mix of two different materials' state.
    SurfaceMaterialMemo& memo = m_lastSurfaceMaterial[surfaceMaterialMemoSlot()];

    if (memo.valid
        && memo.frameId                 == memoFrameId
        && memo.materialHash            == memoMaterialHash
        && memo.materialType            == static_cast<uint32_t>(renderMaterialDataType)
        && memo.samplerOverride         == memoSamplerOverride
        && memo.drawSampler             == memoDrawSampler
        && memo.drawSampler2            == memoDrawSampler2
        && memo.hasTexcoords            == hasTexcoords
        && memo.isEye                   == memoIsEye
        && memo.isRaytracedRenderTarget == memoIsRtRenderTarget) {
      // Counted as a lookup so matLookup keeps meaning "calls into this
      // function" and stays comparable across the change; matMemo is the new
      // split-out. matBuild is untouched by construction -- a memo hit can only
      // occur where the preCreation map would also have hit.
      m_matLookupCount.fetch_add(1, std::memory_order_relaxed);
      m_matMemoHitCount.fetch_add(1, std::memory_order_relaxed);
      if (out_indexInCache) {
        *out_indexInCache = memo.indexInCache;
      }
      // NV-DXVK [Phase2b]: return the memo slot's COPY, never a reference into
      // m_surfaceMaterialCache.m_objects — a concurrent (locked) miss-path
      // insert can push_back-reallocate that vector under a lock-free reader.
      // Contract: the reference is valid until this THREAD's next
      // createSurfaceMaterial call; both callers consume it before then.
      return *memo.material;
    }

    // NV-DXVK [Phase2b]: everything past the memo is the Sec-6 escape — sampler
    // tracking, the preCreation map, texture tracking, and the cache insert all
    // mutate shared unlocked containers ([MatChurn]: matNew=2-4 EVERY frame, so
    // the miss path is warm, not load-time-only). One lock, taken only here,
    // never on the memo-hit path above.
    std::unique_lock<std::mutex> matEscapeLock;
    if (inShardedInstancePhase()) {
      matEscapeLock = std::unique_lock<std::mutex>(m_instanceManager.shardEscapeMutex());
    }

    auto rememberSurfaceMaterial = [&](const uint32_t indexInCache) {
      memo.frameId                 = memoFrameId;
      memo.materialHash            = memoMaterialHash;
      memo.materialType            = static_cast<uint32_t>(renderMaterialDataType);
      memo.samplerOverride         = memoSamplerOverride;
      memo.drawSampler             = memoDrawSampler;
      memo.drawSampler2            = memoDrawSampler2;
      memo.hasTexcoords            = hasTexcoords;
      memo.isEye                   = memoIsEye;
      memo.isRaytracedRenderTarget = memoIsRtRenderTarget;
      memo.indexInCache            = indexInCache;
      // Copied while the escape lock (or CS single-threading) protects the
      // cache — the memo-hit path serves this copy lock-free forever after.
      memo.material                = m_surfaceMaterialCache.at(indexInCache);
      memo.valid                   = true;
    };

    // We're going to use this to create a modified sampler for replacement textures.
    // Legacy and replacement materials should follow same filtering but due to lack of override capability per texture
    // legacy textures use original sampler to stay true to the original intent while replacements use more advanced filtering
    // for better quality by default.
    const Rc<DxvkSampler>& samplerOverride = renderMaterialData.getSamplerOverride();
    Rc<DxvkSampler> sampler = samplerOverride;

    // NV-DXVK: log which branch of the sampler-decision below we take, so we
    // can tell whether legacy BSP draws are bypassing populateSamplerInfo /
    // patchSampler. Encoded as 4 cases: 0 = override+drawSampler, 1 = override
    // only (drawSampler null), 2 = no override but drawSampler valid (patch
    // path), 3 = both null. Logged once per case.
    {
      const bool ovNull = (samplerOverride == nullptr);
      const bool dsNull = (drawCallState.getMaterialData().getSampler().ptr() == nullptr);
      const uint32_t branch = (ovNull ? 2u : 0u) + (dsNull ? 1u : 0u);
      static std::unordered_set<uint32_t> seenBranch;
      if (seenBranch.insert(branch).second) {
        Logger::info(str::format(
          "[D3D11Rtx.SamplerBranch] case=", branch,
          " (ovNull=", ovNull ? 1 : 0,
          " dsNull=", dsNull ? 1 : 0,
          ") matType=", uint32_t(renderMaterialDataType)));
      }
    }

    // If the original sampler if valid and there isnt an override sampler
    // go ahead with patching and maybe merging the sampler states
    //
    // !! STANDING DEFECT: this branch is DEAD in this fork. The
    // !! [D3D11Rtx.SamplerBranch] probe directly above reports case=0
    // !! (samplerOverride non-null) as the ONLY case that ever occurs, so
    // !! samplerOverride == nullptr never holds and patchSampler is never
    // !! reached from here. Consequence: rtx.nativeMipBias, rtx.upscalingMipBias,
    // !! rtx.useAnisotropicFiltering, rtx.maxAnisotropySamples AND the DLSS mip
    // !! bias are all inert - the final sampler is TF2's own D3D11 sampler desc
    // !! verbatim (aniso 16, lod bias 0). The DLSS one is a live visual-quality
    // !! bug whenever DLSS is enabled.
    // !!
    // !! If you are here because a mip-bias or aniso setting "did nothing":
    // !! that is this, the setting is not broken and the value is not wrong.
    // !! Confirm with the case= line in the log before spending a session on it.
    // !! rtx.perfForceSamplerMipBias / rtx.perfForceSamplerAniso exist precisely
    // !! because of this - they apply AFTER the branch, to whichever sampler won.
    if (samplerOverride == nullptr && drawCallState.getMaterialData().getSampler().ptr() != nullptr) {
      DxvkSamplerCreateInfo samplerInfo = drawCallState.getMaterialData().getSampler()->info(); // Use sampler create info struct as convenience
      renderMaterialData.populateSamplerInfo(samplerInfo);

      sampler = patchSampler(samplerInfo.magFilter,
                             samplerInfo.addressModeU, samplerInfo.addressModeV, samplerInfo.addressModeW,
                             samplerInfo.borderColor);
    }
    if (drawCallState.isEye()) {
      // force eye whites and iris to not repeat
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        {}
      );
    }
    // NV-DXVK [perf]: forced mip-bias / anisotropy on the FINAL material sampler.
    //
    // Deliberately after the whole branch above, because the branch is exactly
    // what defeated the normal options: patchSampler (the only consumer of
    // getTotalMipBias and useAnisotropicFiltering) is skipped whenever
    // samplerOverride is non-null, which here is always. Re-deriving the sampler
    // from the winner's own info means this lands no matter which path produced
    // it, and it is a diagnostic override only - both options are inert by
    // default, so the shipped sampler is unchanged when they are not set.
    if (sampler != nullptr) {
      const float forcedBias  = RtxOptions::perfForceSamplerMipBias();
      const float forcedAniso = RtxOptions::perfForceSamplerAniso();

      if (forcedBias != 0.0f || forcedAniso >= 0.0f) {
        const DxvkSamplerCreateInfo& src = sampler->info();
        DxvkSamplerCreateInfo forced = src;

        forced.mipmapLodBias = src.mipmapLodBias + forcedBias;

        if (forcedAniso >= 0.0f) {
          // <=1 tap is isotropic by definition, so express that as the feature
          // being off rather than as maxAnisotropy=1 - the two are equivalent to
          // the hardware but only the former is unambiguous in the log line.
          if (forcedAniso <= 1.0f) {
            forced.useAnisotropy = VK_FALSE;
            forced.maxAnisotropy = 1.0f;
          } else {
            const VkPhysicalDeviceLimits& limits =
              m_device->properties().core.properties.limits;
            forced.useAnisotropy = VK_TRUE;
            forced.maxAnisotropy = std::min(limits.maxSamplerAnisotropy, forcedAniso);
          }
        }

        // DxvkDevice::createSampler does not deduplicate, unlike
        // Resources::getSampler which keeps its own cache. Calling it per
        // material per frame would mint a fresh VkSampler every time and run the
        // descriptor pool out of samplers, which would present as a driver error
        // rather than as a bad diagnostic. Memoize on the same hash the resource
        // manager uses. Resources::getSampler is not reusable directly here
        // because it takes a bool for anisotropy and reads the tap count from
        // RtxOptions, and this probe needs to force the count explicitly.
        static std::mutex s_forcedSamplerMu;
        static std::unordered_map<XXH64_hash_t, Rc<DxvkSampler>> s_forcedSamplerCache;
        const XXH64_hash_t forcedKey = forced.calculateHash();
        {
          std::lock_guard<std::mutex> lk(s_forcedSamplerMu);
          auto it = s_forcedSamplerCache.find(forcedKey);
          if (it != s_forcedSamplerCache.end()) {
            sampler = it->second;
          } else {
            sampler = m_device->createSampler(forced);
            s_forcedSamplerCache.emplace(forcedKey, sampler);
          }
        }
      }
    }

    uint32_t samplerIndex = trackSampler(sampler);

    // NV-DXVK: log final sampler's address modes once per (U,V,W,filter)
    // combo, regardless of which branch above produced it (override / patched
    // / eye-clamped). Tests the wrap-mode hypothesis for BSP world-scale UVs.
    // 0=REPEAT 1=MIRRORED 2=CLAMP_EDGE 3=CLAMP_BORDER 4=MIRROR_CLAMP.
    if (sampler != nullptr) {
      const auto& si = sampler->info();
      // lodBias and anisotropy are part of the key so a forced override re-fires
      // this line instead of being hidden behind an already-seen address-mode
      // combination. Without that, a sampler test could look applied in the
      // config and unapplied in reality - which is precisely what happened with
      // rtx.nativeMipBias.
      const uint32_t key =
        (uint32_t(si.addressModeU) & 0x7u) |
        ((uint32_t(si.addressModeV) & 0x7u) << 3) |
        ((uint32_t(si.addressModeW) & 0x7u) << 6) |
        ((uint32_t(si.magFilter)   & 0x7u) << 9) |
        ((uint32_t(si.mipmapMode)  & 0x7u) << 12)
        | ((uint32_t(int32_t(si.mipmapLodBias * 4.0f)) & 0xFFu) << 15)
        | ((uint32_t(si.useAnisotropy ? uint32_t(si.maxAnisotropy) : 0u) & 0x1Fu) << 23);
      static std::unordered_set<uint32_t> seen;
      if (seen.insert(key).second) {
        Logger::info(str::format(
          "[D3D11Rtx.SamplerWrap] U=", uint32_t(si.addressModeU),
          " V=", uint32_t(si.addressModeV),
          " W=", uint32_t(si.addressModeW),
          " mag=", uint32_t(si.magFilter),
          " min=", uint32_t(si.minFilter),
          " mipMode=", uint32_t(si.mipmapMode),
          " lodMin=", si.mipmapLodMin,
          " lodMax=", si.mipmapLodMax,
          " lodBias=", si.mipmapLodBias,
          " aniso=", si.useAnisotropy ? si.maxAnisotropy : 0.0f,
          " samplerIdx=", samplerIndex));
      }
    }
    uint32_t samplerIndex2 = UINT32_MAX;
    if (renderMaterialDataType == MaterialDataType::RayPortal) {
      samplerIndex2 = trackSampler(drawCallState.getMaterialData().getSampler2());
    }

    XXH64_hash_t preCreationHash = renderMaterialData.getHash();
    preCreationHash = XXH64(&samplerIndex, sizeof(samplerIndex), preCreationHash);
    preCreationHash = XXH64(&samplerIndex2, sizeof(samplerIndex2), preCreationHash);
    preCreationHash = XXH64(&hasTexcoords, sizeof(hasTexcoords), preCreationHash);
    preCreationHash = XXH64(&drawCallState.isUsingRaytracedRenderTarget, sizeof(drawCallState.isUsingRaytracedRenderTarget), preCreationHash);

    // NV-DXVK [MatChurn]: count every material resolution and, separately, the
    // ones that fall through to a full material build. m_preCreationSurfaceMaterialMap
    // is cleared every frame, so a miss here is normal once per distinct material
    // per frame - what is NOT normal is the m_surfaceMaterialCache INSERT further
    // down, which only happens when the resulting material hashes to something the
    // cache has never held. See MaterialChurnSample in the header.
    m_matLookupCount.fetch_add(1, std::memory_order_relaxed);

    auto iter = m_preCreationSurfaceMaterialMap.find(preCreationHash);
    if (iter != m_preCreationSurfaceMaterialMap.end()) {
      rememberSurfaceMaterial(iter->second);
      if (out_indexInCache) {
        *out_indexInCache = iter->second;
      }
      // NV-DXVK [Phase2b]: memo copy, not a cache reference — see the memo-hit
      // return above. rememberSurfaceMaterial just filled it under the lock.
      return *memo.material;
    }

    m_matPreMissCount.fetch_add(1, std::memory_order_relaxed);

    std::optional<RtSurfaceMaterial> surfaceMaterial;

    if (renderMaterialDataType == MaterialDataType::Opaque || drawCallState.isUsingRaytracedRenderTarget) {
      uint32_t albedoOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t secondaryTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t tangentTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t ambientOcclusionTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t lightmapTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t lightmap2TextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t detailTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t cloudMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t screenSpaceEmissiveMaskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceMaterialIndex = SURFACE_INDEX_INVALID;
      uint32_t subsurfaceTransmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceThicknessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceSingleScatteringAlbedoTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      float anisotropy;
      float emissiveIntensity;
      Vector4 albedoOpacityConstant;
      float roughnessConstant;
      float metallicConstant;
      Vector3 emissiveColorConstant;
      bool enableEmissive;
      bool thinFilmEnable = false;
      bool alphaIsThinFilmThickness = false;
      float thinFilmThicknessConstant = 0.0f;
      float displaceIn = 0.0f;
      float displaceOut = 0.0f;
      bool isUsingRaytracedRenderTarget = drawCallState.isUsingRaytracedRenderTarget;
      uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

      Vector3 subsurfaceTransmittanceColor(0.0f, 0.0f, 0.0f);
      float subsurfaceMeasurementDistance = 0.0f;
      Vector3 subsurfaceSingleScatteringAlbedo(0.0f, 0.0f, 0.0f);
      float subsurfaceVolumetricAnisotropy = 0.0f;

      float subsurfaceRadiusScale = 0.0f;
      float subsurfaceMaxSampleRadius = 0.0f;

      bool ignoreAlphaChannel = false;
      bool albedoIsSRGB = false;
      bool metallicIsSRGB = false;
      bool emissiveIsSRGB = false;
      // NV-DXVK [SubViewPremultOverride]: shadow `getAlbedoIsPremultiplied()`
      // so we can OR-in the override below for sub-view VS hashes. The
      // material's own flag flows through by default; only the targeted
      // VS-hash override flips it on.
      bool albedoIsPremultiplied = false;
      // NV-DXVK [SubViewSkyboxEmissiveOverride]: set true for draws
      // structurally identified as the 3D-skybox painted dome (large
      // world-AABB sub-view geometry). Routes albedo → emissiveRadiance
      // in the slang shader so the dome renders its baked colour
      // instead of integrating against in-scene lights that don't
      // reach 6M+ units away. See drawCallState.getTransformData()
      // .isSubViewSkybox + d3d11_rtx.cpp [SubViewSkyboxClassify].
      bool bakedAlbedoAsEmissive = false;

      constexpr Vector4 kWhiteModeAlbedo = Vector4(0.7f, 0.7f, 0.7f, 1.0f);

      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();

      if (RtxOptions::useWhiteMaterialMode()) {
        albedoOpacityConstant = kWhiteModeAlbedo;
        metallicConstant = 0.f;
        roughnessConstant = 1.f;
      } else {
        if (opaqueMaterialData.getAlbedoOpacityTexture().getManagedTexture() != nullptr) {
          samplerFeedbackStamp = opaqueMaterialData.getAlbedoOpacityTexture().getManagedTexture()->m_samplerFeedbackStamp;
        }

        trackTexture(opaqueMaterialData.getAlbedoOpacityTexture(), albedoOpacityTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

        // NV-DXVK: detect whether the albedo texture is bound with an
        // sRGB-format image view. If so the HW sampler already converts
        // sRGB->linear, and the slang shader must NOT also run its software
        // gammaToLinear() (that double-decode crushes albedo dark - it is
        // why TF2's BC7_SRGB sky cloud textures rendered near-black).
        // Flagged onto the GPU material via OPAQUE_SURFACE_MATERIAL_FLAG_
        // ALBEDO_IS_SRGB. Derived purely from the albedo texture, so it
        // needs no separate material-hash contribution.
        {
          const TextureRef& albTexRef = opaqueMaterialData.getAlbedoOpacityTexture();
          const DxvkImageView* albView = albTexRef.getImageView();
          if (albView != nullptr) {
            const DxvkFormatInfo* albFmtInfo = imageFormatInfo(albView->info().format);
            albedoIsSRGB = albFmtInfo != nullptr
              && albFmtInfo->flags.test(DxvkFormatFlag::ColorSpaceSrgb);
          }

          // NV-DXVK: same question for the metallic/spec and emissive slots. The
          // detection is identical - is the bound view an sRGB format - but the
          // three channels want three different responses to the same answer:
          //
          //   albedo    sRGB data, decode wanted     -> skip the software decode
          //   metallic  linear F0, decode never wanted -> undo it
          //   emissive  sRGB data, decode wanted     -> skip the software decode
          //
          // so they cannot share a flag. See the flag definitions in
          // shared_constants.h for why F0 in particular cannot survive the decode.
          {
            const DxvkImageView* metView = opaqueMaterialData.getMetallicTexture().getImageView();
            if (metView != nullptr) {
              const DxvkFormatInfo* metFmtInfo = imageFormatInfo(metView->info().format);
              metallicIsSRGB = metFmtInfo != nullptr
                && metFmtInfo->flags.test(DxvkFormatFlag::ColorSpaceSrgb);
            }

            const DxvkImageView* emiView = opaqueMaterialData.getEmissiveColorTexture().getImageView();
            if (emiView != nullptr) {
              const DxvkFormatInfo* emiFmtInfo = imageFormatInfo(emiView->info().format);
              emissiveIsSRGB = emiFmtInfo != nullptr
                && emiFmtInfo->flags.test(DxvkFormatFlag::ColorSpaceSrgb);
            }
          }

          // NV-DXVK [AlbedoSrgb]: raw dump of the inputs to the decision above.
          //
          // The failure this exists to catch is not "the flag is missing" - that
          // shows up immediately as crushed-dark albedo and is hard to miss. It is
          // the opposite direction: the flag set while the sampler did not actually
          // decode, so the shader skips gammaToLinear() and albedo stays in gamma
          // space. Because sRGB->linear is a power curve it EXPANDS channel ratios,
          // so skipping it compresses them - a warm grey stone measuring 1.00:0.88:0.77
          // linear reads 1.00:0.94:0.89 instead, i.e. brighter and visibly greyer,
          // with no other symptom to give it away.
          //
          // Raw values only: the format enum, the flag bit, and the outcome. No
          // threshold, no verdict - a classifier here would just be this same
          // guess baked in, and the whole point is to find out which way it went.
          //
          // Deduplicated by image hash so this is a few lines per level rather than
          // per draw. Reuses the albView pointer resolved above rather than calling
          // TextureRef::getImageHash(), which re-resolves through
          // ManagedTexture::m_currentMipView - a pointer the streaming thread frees
          // and republishes. One resolve, no second traversal, no widened race.
          if (RtxOptions::logAlbedoSrgbProbe()) {
            const XXH64_hash_t albImageHash =
              (albView != nullptr && albView->image() != nullptr)
                ? albView->image()->getHash() : 0;

            static dxvk::mutex s_albedoSrgbMutex;
            static std::unordered_set<XXH64_hash_t> s_albedoSrgbSeen;

            bool isNew = false;
            {
              std::lock_guard<dxvk::mutex> lock(s_albedoSrgbMutex);
              isNew = s_albedoSrgbSeen.insert(albImageHash).second;
            }

            if (isNew) {
              const VkFormat albFormat =
                (albView != nullptr) ? albView->info().format : VK_FORMAT_UNDEFINED;
              const DxvkFormatInfo* albFmtInfo =
                (albFormat != VK_FORMAT_UNDEFINED) ? imageFormatInfo(albFormat) : nullptr;

              Logger::info(str::format(
                "[AlbedoSrgb] tex=0x", std::hex, albImageHash, std::dec,
                " view=", (albView != nullptr ? "yes" : "NULL"),
                " vkFormat=", static_cast<uint32_t>(albFormat),
                " fmtInfo=", (albFmtInfo != nullptr ? "yes" : "NULL"),
                " colorSpaceSrgbBit=", (albFmtInfo != nullptr
                  && albFmtInfo->flags.test(DxvkFormatFlag::ColorSpaceSrgb)) ? 1 : 0,
                " -> ALBEDO_IS_SRGB=", albedoIsSRGB ? 1 : 0,
                " managed=", (albTexRef.getManagedTexture() != nullptr) ? 1 : 0));
            }
          }

          // NV-DXVK [MatMaps]: which PBR maps actually reached this material.
          //
          // The maps arrive by matching pixel-shader RDEF SRV names
          // (albedoTexture / normalTexture / glossTexture / specTexture / ...) in
          // D3D11Rtx::FillMaterialData. Those names were read off TF2's CHARACTER
          // shaders. Nothing guarantees the world, brush and viewmodel shaders name
          // their SRVs identically, and a name that does not match is not an error -
          // the channel just falls back to the rtx.legacyMaterial.* constant. The
          // surface then renders as a plausible-looking wrong material rather than
          // an obviously broken one, which is much harder to notice.
          //
          // So the question this answers is not "is the map correct" but the prior
          // one: "did a map bind here at all". Keyed by (VS, material) so a shader
          // binding nothing sits in the log right next to one binding everything.
          // Format is included because it decides how the sample is interpreted -
          // a single-channel map cannot carry gloss in .b no matter what samples it.
          if (RtxOptions::logMaterialMapProbe()) {
            const uint64_t vsHash =
              uint64_t(drawCallState.getTransformData().vertexShaderHash);
            const uint64_t matHash = uint64_t(opaqueMaterialData.getHash());
            const uint64_t key = vsHash ^ (matHash * 0x9E3779B97F4A7C15ull);

            static dxvk::mutex s_matMapsMutex;
            static std::unordered_set<uint64_t> s_matMapsSeen;

            bool isNewPair = false;
            {
              std::lock_guard<dxvk::mutex> lock(s_matMapsMutex);
              isNewPair = s_matMapsSeen.insert(key).second;
            }

            if (isNewPair) {
              // Reports hash + format for a bound map, or "-" when the RDEF name
              // never matched and the constant fallback is in play.
              auto describe = [](const TextureRef& t) -> std::string {
                if (!t.isValid() || t.isImageEmpty()) {
                  return "-";
                }
                const DxvkImageView* v = t.getImageView();
                if (v == nullptr || v->image() == nullptr) {
                  return "novw";
                }
                return str::format("0x", std::hex, v->image()->getHash(), std::dec,
                                   "/f", static_cast<uint32_t>(v->info().format));
              };

              Logger::info(str::format(
                "[MatMaps] vs=0x", std::hex, vsHash, " mat=0x", matHash, std::dec,
                " alb=", describe(opaqueMaterialData.getAlbedoOpacityTexture()),
                " nrm=", describe(opaqueMaterialData.getNormalTexture()),
                " rgh=", describe(opaqueMaterialData.getRoughnessTexture()),
                " met=", describe(opaqueMaterialData.getMetallicTexture()),
                " emi=", describe(opaqueMaterialData.getEmissiveColorTexture()),
                // The three colour-space decisions, so each one is checkable in
                // the log rather than inferred from how the frame looks. Two
                // materials can bind the same texture and still resolve these
                // differently, which is why they sit on the per-material line.
                " | srgb alb=", albedoIsSRGB ? 1 : 0,
                " met=", metallicIsSRGB ? 1 : 0,
                " emi=", emissiveIsSRGB ? 1 : 0,
                // Which emissive branch the shader will take. emiSrc=const means
                // no texture bound, so the constant is a gamma-authored fallback
                // and DOES get converted; emiSrc=tex/tex*tint means the sample
                // is the colour and the tint (if any) rides along as a linear
                // multiplier. Confirms the emissiveColorLoaded gating without
                // having to reason backwards from a screenshot.
                " emiSrc=", (opaqueMaterialData.getEmissiveColorTexture().isValid()
                             && !opaqueMaterialData.getEmissiveColorTexture().isImageEmpty())
                            ? (opaqueMaterialData.getEmissiveTintFromConstant() ? "tex*tint" : "tex")
                            : "const",
                " emiConst=(", opaqueMaterialData.getEmissiveColorConstant().x,
                ",", opaqueMaterialData.getEmissiveColorConstant().y,
                ",", opaqueMaterialData.getEmissiveColorConstant().z, ")",
                // NV-DXVK: EVERY field the material hash is built from, named.
                //
                // The describe() fields above cover five texture slots. That was
                // not enough: on 2026-08-05 a stationary object took four
                // material hashes on four consecutive frames while alb/nrm/rgh/
                // met were byte-identical between two of them — so the moving
                // input was in a slot this line never printed (tex1, AO,
                // lightmap, lightmap2, detail, cloudMask, subsurface, or one of
                // the constants). Five of seventeen textures is a probe that can
                // acquit the wrong field.
                //
                // Emitted once per (VS, material) pair like the rest of the line.
                // When the material hash is itself unstable that is once per
                // frame per object, which is exactly the rate needed to diff
                // consecutive frames — and is why this must never move to a
                // per-draw site.
                " | ", opaqueMaterialData.debugHashInputs()));
            }
          }
        }

        // NV-DXVK [SubViewSkyboxEmissiveOverride]: for the painted
        // 3D-skybox dome sub-set of sub-view draws (structurally
        // identified by AABB diagonal > 5M units — see
        // DrawCallTransforms::isSubViewSkybox), set the
        // bakedAlbedoAsEmissive flag so the slang shader routes
        // albedo into emissiveRadiance and zeros the diffuse /
        // specular response. Reason: the dome sits ~6.75M units from
        // any in-scene light, so light × albedo = 0 and the path
        // tracer renders pure black despite the GBuffer holding the
        // correct sampled colour. The flag tells the shader to skip
        // the light integral and output the baked colour directly.
        // One-shot log per VS so we can verify in-log which shaders
        // receive the override.
        if (drawCallState.getTransformData().isSubViewSkybox && !RtxOptions::disableSubViewSkyboxEmissive()) {
          const XXH64_hash_t vsHashSkybox =
              drawCallState.getTransformData().vertexShaderHash;
          if (!bakedAlbedoAsEmissive) {
            bakedAlbedoAsEmissive = true;
            static std::mutex sLogMu;
            static std::unordered_set<XXH64_hash_t> sLogged;
            bool first = false;
            {
              std::lock_guard<std::mutex> g(sLogMu);
              first = sLogged.insert(vsHashSkybox).second;
            }
            if (first) {
              Logger::info(str::format(
                "[SubViewSkyboxEmissiveOverride] forcing BAKED_ALBEDO_AS_EMISSIVE for vsHash=0x",
                std::hex, vsHashSkybox, std::dec,
                " (isSubViewSkybox=1, world-AABB diag > 5M; routes albedo → emissive in shader)"));
            }
          }
        }

        // NV-DXVK [SubView{Srgb,Premult}Override]: encoding-pipeline
        // overrides for engine-hook sub-view reproject draws (the
        // painted 3D-skybox dome, mountains, distant ships in TF2).
        // Gated on the structural `isSubView` tag set at the path-13
        // o2w site in d3d11_rtx — replaces the previous hardcoded
        // `kSubViewVsHashes` list, which silently rotted whenever
        // game binaries / mods / shaders changed.
        //
        // Why both flags:
        //  - SRGB:    AlbedoDrift coverage showed VS_eda5e (xxh
        //             0x2a729f16017d841b) pushing 786K pixels/frame of
        //             drift through DriftStageGamma — the slang
        //             gammaToLinear() decode at opaque_surface_-
        //             material_interaction.slangh:1456 was sending
        //             the sub-view's mid-tone albedo down to ~0.07
        //             linear. The sub-view's bound albedo view is
        //             UNORM (not sRGB), so format-based detection
        //             above left albedoIsSRGB=false; force it on so
        //             gammaToLinear is skipped for pre-lit content.
        //  - Premult: After the SRGB override DriftStageGamma went to
        //             ~0 but DriftStageAdjusted jumped to 148K pixels
        //             for VS_eda5e — the opacity-multiply step inside
        //             albedoToAdjustedAlbedo at slangh:1697 was
        //             darkening the sky. The premult flag is the
        //             existing in-codebase bypass that hands back raw
        //             `albedo` directly, skipping that multiply.
        //
        // Sky still rendering black after both flags is a *downstream*
        // problem (sub-view sits ~6.75M units from any light source,
        // radiance = albedo * 0 = 0) — see HANDOFF_TF2_SUBVIEW_SKY_-
        // BLACKBOX.md for Change 2 (emissive routing) which depends
        // on the structural tag added here.
        if (drawCallState.getTransformData().isSubView) {
          const XXH64_hash_t vsHash =
              drawCallState.getTransformData().vertexShaderHash;
          if (!albedoIsSRGB) {
            albedoIsSRGB = true;
            // One-shot per VS hash so we can confirm in-log which
            // sub-view shaders actually triggered the override at
            // runtime — same dedup granularity the hash-list version
            // had, just keyed off the structural signal now.
            static std::mutex sLogMu;
            static std::unordered_set<XXH64_hash_t> sLogged;
            bool first = false;
            {
              std::lock_guard<std::mutex> g(sLogMu);
              first = sLogged.insert(vsHash).second;
            }
            if (first) {
              Logger::info(str::format(
                "[SubViewSrgbOverride] forcing ALBEDO_IS_SRGB for vsHash=0x",
                std::hex, vsHash, std::dec,
                " (isSubView=1, was format-derived false, override skips gammaToLinear)"));
            }
          }
          if (!albedoIsPremultiplied) {
            albedoIsPremultiplied = true;
            static std::mutex sLogMu;
            static std::unordered_set<XXH64_hash_t> sLogged;
            bool first = false;
            {
              std::lock_guard<std::mutex> g(sLogMu);
              first = sLogged.insert(vsHash).second;
            }
            if (first) {
              Logger::info(str::format(
                "[SubViewPremultOverride] forcing ALBEDO_IS_PREMULTIPLIED for vsHash=0x",
                std::hex, vsHash, std::dec,
                " (isSubView=1, skips opacity-multiply in albedoToAdjustedAlbedo)"));
            }
          }
        }

        // NV-DXVK [SubViewPremultOverride]: OR-in the material's own
        // premult flag last, so any opaqueMaterialData that legitimately
        // says "I'm premultiplied" still propagates even if the VS hash
        // isn't in our sub-view list.
        albedoIsPremultiplied = albedoIsPremultiplied
                              || opaqueMaterialData.getAlbedoIsPremultiplied();
        trackTexture(opaqueMaterialData.getRoughnessTexture(), roughnessTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getMetallicTexture(), metallicTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getSecondaryTexture(), secondaryTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
        // NV-DXVK: per-draw diagnostic — report the actual texture indices the
        // albedo / normal / rough / metallic / emissive ended up with. If
        // albedo shows INVALID we know the shader is sampling the constant
        // (flat colour). Gated + throttled so it doesn't spam.
        // NV-DXVK [perf]: gated behind RTX_D3D11_DIAG — runs on the CS thread and
        // calls getImageHash(), which races with texture streaming/GC (documented
        // crash). Off the normal path now; the log was filtered out anyway.
        if (sceneDiagEnabled()) {
          static std::atomic<uint32_t> sIdxLogCount{0};
          const uint32_t n = ++sIdxLogCount;
          if (n <= 30 || (n % 500) == 0) {
            const auto& albTex = opaqueMaterialData.getAlbedoOpacityTexture();
            Logger::info(str::format(
              "[RTX-TexTrack] draw#", n,
              " hasUV=", (hasTexcoords ? "1" : "0"),
              " albValid=", (albTex.isValid() && !albTex.isImageEmpty() ? "1" : "0"),
              " albHash=0x", std::hex, albTex.getImageHash(), std::dec,
              " albIdx=", albedoOpacityTextureIndex,
              " albHasMgd=", (albTex.getManagedTexture().ptr() ? "1" : "0"),
              " albView=", (albTex.getImageView() ? "1" : "0")));
          }
        }

        albedoOpacityConstant.xyz() = opaqueMaterialData.getAlbedoConstant();
        albedoOpacityConstant.w = opaqueMaterialData.getOpacityConstant();
        metallicConstant = opaqueMaterialData.getMetallicConstant();
        roughnessConstant = opaqueMaterialData.getRoughnessConstant();
      }

      trackTexture(opaqueMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getTangentTexture(), tangentTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getHeightTexture(), heightTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getAmbientOcclusionTexture(), ambientOcclusionTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getLightmapTexture(),         lightmapTextureIndex,        hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getLightmap2Texture(),        lightmap2TextureIndex,       hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getDetailTexture(),           detailTextureIndex,          hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getCloudMaskTexture(),        cloudMaskTextureIndex,       hasTexcoords, true, samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getScreenSpaceEmissiveMaskTexture(), screenSpaceEmissiveMaskTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

      emissiveIntensity = opaqueMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();
      emissiveColorConstant = opaqueMaterialData.getEmissiveColorConstant();
      enableEmissive = opaqueMaterialData.getEnableEmission();
      anisotropy = opaqueMaterialData.getAnisotropyConstant();
        
      thinFilmEnable = opaqueMaterialData.getEnableThinFilm();
      alphaIsThinFilmThickness = opaqueMaterialData.getAlphaIsThinFilmThickness();
      thinFilmThicknessConstant = opaqueMaterialData.getThinFilmThicknessConstant();
      displaceIn = opaqueMaterialData.getDisplaceIn();
      displaceOut = opaqueMaterialData.getDisplaceOut();

      ignoreAlphaChannel = opaqueMaterialData.getIgnoreAlphaChannel();

      subsurfaceMeasurementDistance = opaqueMaterialData.getSubsurfaceMeasurementDistance() * RtxOptions::SubsurfaceScattering::surfaceThicknessScale();

      const bool isSubsurfaceScatteringDiffusionProfile = opaqueMaterialData.getSubsurfaceDiffusionProfile();

      if ((RtxOptions::SubsurfaceScattering::enableThinOpaque()       && subsurfaceMeasurementDistance > 0.0f) ||
          (RtxOptions::SubsurfaceScattering::enableDiffusionProfile() && isSubsurfaceScatteringDiffusionProfile)) {

        subsurfaceTransmittanceColor = opaqueMaterialData.getSubsurfaceTransmittanceColor();
        subsurfaceVolumetricAnisotropy = opaqueMaterialData.getSubsurfaceVolumetricAnisotropy();

        if (isSubsurfaceScatteringDiffusionProfile) {
          // NOTE: reuse of the variable!
          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceRadius(); 
          subsurfaceMaxSampleRadius = std::max(0.F, opaqueMaterialData.getSubsurfaceMaxSampleRadius());
          subsurfaceRadiusScale = std::max(opaqueMaterialData.getSubsurfaceRadiusScale(), 1e-5f);
          assert(subsurfaceRadiusScale > 0);

          m_sssMaterialExist = true;
        } else /* if thin opaque */ {
          assert(subsurfaceMeasurementDistance > 0);

          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceSingleScatteringAlbedo();
          subsurfaceMaxSampleRadius = 0;
          subsurfaceRadiusScale = -1;
          assert(subsurfaceRadiusScale < 0);  // if < 0, then shaders assume that
                                              // this material is not SubsurfaceScatter, but just SingleScatter
                                              // same here, but <0.F

          m_thinOpaqueMaterialExist = true;
        }

        if (RtxOptions::SubsurfaceScattering::enableTextureMaps()) {
          trackTexture(opaqueMaterialData.getSubsurfaceTransmittanceTexture(), subsurfaceTransmittanceTextureIndex, hasTexcoords, true, samplerFeedbackStamp);

          if (isSubsurfaceScatteringDiffusionProfile) {
            // NOTE: reuse of 'subsurfaceSingleScatteringAlbedoTextureIndex' variable!
            trackTexture(opaqueMaterialData.getSubsurfaceRadiusTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
          } else {
            trackTexture(opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
            trackTexture(opaqueMaterialData.getSubsurfaceThicknessTexture(), subsurfaceThicknessTextureIndex, hasTexcoords, true, samplerFeedbackStamp);
          }
        }

        const auto subsurfaceMaterial = RtSubsurfaceMaterial{
          subsurfaceTransmittanceTextureIndex,
          subsurfaceThicknessTextureIndex,
          subsurfaceSingleScatteringAlbedoTextureIndex,
          subsurfaceTransmittanceColor,
          subsurfaceMeasurementDistance,
          subsurfaceSingleScatteringAlbedo,
          subsurfaceVolumetricAnisotropy,
          subsurfaceRadiusScale,
          subsurfaceMaxSampleRadius,
        };
        subsurfaceMaterialIndex = m_surfaceMaterialExtensionCache.track(subsurfaceMaterial);
      }

      const RtOpaqueSurfaceMaterial opaqueSurfaceMaterial{
        albedoOpacityTextureIndex, normalTextureIndex,
        tangentTextureIndex, heightTextureIndex, roughnessTextureIndex,
        metallicTextureIndex, emissiveColorTextureIndex,
        anisotropy, emissiveIntensity,
        albedoOpacityConstant,
        roughnessConstant, metallicConstant,
        emissiveColorConstant, enableEmissive,
        ignoreAlphaChannel, thinFilmEnable, alphaIsThinFilmThickness,
        thinFilmThicknessConstant, samplerIndex, displaceIn, displaceOut,
        subsurfaceMaterialIndex, isUsingRaytracedRenderTarget,
        samplerFeedbackStamp,
        secondaryTextureIndex,
        ambientOcclusionTextureIndex,
        lightmapTextureIndex,
        lightmap2TextureIndex,
        detailTextureIndex,
        cloudMaskTextureIndex,
        // NV-DXVK: Source/TF2 alpha-modulate-emissive flag — sourced from
        // OpaqueMaterialData (set in LegacyMaterialData::as<>() based on the
        // PS's c_useAlphaModulateEmissive D3D_SVF_USED), routed through to
        // the GPU surface material via a single flag bit.
        opaqueMaterialData.getAlphaModulateEmissive(),
        // NV-DXVK: tint-from-constant flag — when set, the slang shader
        // multiplies the per-pixel emissive texture sample by the
        // emissiveColorConstant tint, matching the original PS's
        // `emissive = sample * c_emissiveTint`.
        opaqueMaterialData.getEmissiveTintFromConstant(),
        // NV-DXVK: TF2 viewmodel screen-space scrolling overlay emissive —
        // when true, the slang shader samples emissive at SV_Position-derived
        // UV using the matrix+translate below, and (if maskTextureIndex is
        // valid) multiplies by the mask sampled at mesh UV.
        opaqueMaterialData.getHasScreenSpaceEmissive(),
        opaqueMaterialData.getScreenSpaceEmissiveMatRow0(),
        opaqueMaterialData.getScreenSpaceEmissiveMatRow1(),
        opaqueMaterialData.getScreenSpaceEmissiveTranslate(),
        screenSpaceEmissiveMaskTextureIndex,
        // NV-DXVK: albedo texture bound with an sRGB-format view — shader
        // skips its software gammaToLinear() to avoid a double sRGB decode.
        albedoIsSRGB,
        metallicIsSRGB,
        emissiveIsSRGB,
        // NV-DXVK: TF2 3D-skybox cloud billboard — opaque surface shader
        // reconstructs the game's fog-blend synthesis. See
        // OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG.
        opaqueMaterialData.getTf2SkyboxFog(),
        // NV-DXVK: premultiplied-alpha-blend source — the slang shader
        // skips the opacity-multiply inside albedoToAdjustedAlbedo /
        // calcBaseReflectivity for these surfaces (their .rgb is
        // already (color*alpha); multiplying again would double-darken
        // soft cloud edges into noisy dark dots). See
        // OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED.
        albedoIsPremultiplied,
        // NV-DXVK: baked-albedo-as-emissive — for sub-view sky-dome
        // content whose colour is already authored complete. Slang
        // shader routes albedo → emissiveRadiance, zeros diffuse /
        // specular. See OPAQUE_SURFACE_MATERIAL_FLAG_BAKED_ALBEDO_AS_
        // EMISSIVE. Derived from DrawCallTransforms::isSubViewSkybox
        // (AABB-diagonal-based structural classifier in d3d11_rtx).
        bakedAlbedoAsEmissive
      };

      if (opaqueSurfaceMaterial.hasValidDisplacement()) {
        ++m_activePOMCount;
      }

      surfaceMaterial.emplace(opaqueSurfaceMaterial);
    } else if (renderMaterialDataType == MaterialDataType::Translucent) {
      const auto& translucentMaterialData = renderMaterialData.getTranslucentMaterialData();

      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t transmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      trackTexture(translucentMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords);
      trackTexture(translucentMaterialData.getTransmittanceTexture(), transmittanceTextureIndex, hasTexcoords);
      trackTexture(translucentMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords);

      float refractiveIndex = translucentMaterialData.getRefractiveIndex() * std::clamp(TranslucentMaterialOptions::refractiveIndexScale(), 0.0f, 3.0f);
      Vector3 transmittanceColor = translucentMaterialData.getTransmittanceColor();
      float transmittanceMeasureDistance = translucentMaterialData.getTransmittanceMeasurementDistance();
      Vector3 emissiveColorConstant = translucentMaterialData.getEmissiveColorConstant();
      bool enableEmissive = translucentMaterialData.getEnableEmission();
      float emissiveIntensity = translucentMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();
      bool isThinWalled = translucentMaterialData.getEnableThinWalled();
      float thinWallThickness = translucentMaterialData.getThinWallThickness();
      bool useDiffuseLayer = translucentMaterialData.getEnableDiffuseLayer();

      const RtTranslucentSurfaceMaterial translucentSurfaceMaterial{
        normalTextureIndex, transmittanceTextureIndex, emissiveColorTextureIndex,
        refractiveIndex,
        transmittanceMeasureDistance, transmittanceColor,
        enableEmissive, emissiveIntensity, emissiveColorConstant,
        isThinWalled, thinWallThickness, useDiffuseLayer, samplerIndex
      };

      surfaceMaterial.emplace(translucentSurfaceMaterial);
    } else if (renderMaterialDataType == MaterialDataType::RayPortal) {
      const auto& rayPortalMaterialData = renderMaterialData.getRayPortalMaterialData();

      uint32_t maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture(), maskTextureIndex, hasTexcoords, false);
      uint32_t maskTextureIndex2 = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture2(), maskTextureIndex2, hasTexcoords, false);

      uint8_t rayPortalIndex = rayPortalMaterialData.getRayPortalIndex();
      float rotationSpeed = rayPortalMaterialData.getRotationSpeed();
      bool enableEmissive = rayPortalMaterialData.getEnableEmission();
      float emissiveIntensity = rayPortalMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();

      const RtRayPortalSurfaceMaterial rayPortalSurfaceMaterial{
        maskTextureIndex, maskTextureIndex2, rayPortalIndex,
        rotationSpeed, enableEmissive, emissiveIntensity, samplerIndex, samplerIndex2
      };

      surfaceMaterial.emplace(rayPortalSurfaceMaterial);
    }

    assert(surfaceMaterial.has_value());
    assert(surfaceMaterial->validate());

    // Cache this
    const uint32_t index = m_surfaceMaterialCache.track(*surfaceMaterial);
    m_preCreationSurfaceMaterialMap[preCreationHash] = index;
    rememberSurfaceMaterial(index);
    if (out_indexInCache) {
      *out_indexInCache = index;
    }
    // NV-DXVK [Phase2b]: memo copy, not a cache reference — see the memo-hit
    // return above.
    return *memo.material;
  }

  std::optional<XXH64_hash_t> SceneManager::findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue) {
    std::lock_guard lock { m_drawCallMeta.mutex };

    auto tryFindIn = [](const std::unordered_map<ObjectPickingValue, DrawCallMetaInfo>& table, ObjectPickingValue toFind)
      -> std::optional<XXH64_hash_t> {
      auto found = table.find(toFind);
      if (found != table.end()) {
        const DrawCallMetaInfo& meta = found->second;
        if (meta.legacyTextureHash != kEmptyHash) {
          return meta.legacyTextureHash;
        }
      }
      return std::nullopt;
    };

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        if (auto h = tryFindIn(m_drawCallMeta.infos[tick], objectPickingValue)) {
          return h;
        }
      }
    }
    return std::nullopt;
  }

  std::vector<ObjectPickingValue> SceneManager::gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash) {
    std::lock_guard lock { m_drawCallMeta.mutex };
    assert(texHash != kEmptyHash);

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };

    auto correspondingValues = std::vector<ObjectPickingValue> {};
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        for (const auto& [pickingValue, meta] : m_drawCallMeta.infos[tick]) {
          if (texHash == meta.legacyTextureHash) {
            correspondingValues.push_back(pickingValue);
          } else if (texHash == meta.legacyTextureHash2) {
            correspondingValues.push_back(pickingValue);
          }
        }
        break;
      }
    }
    return correspondingValues;
  }

  SceneManager::SamplerIndex SceneManager::trackSampler(Rc<DxvkSampler> sampler) {
    if (sampler == nullptr) {
      ONCE(Logger::info("Found a null sampler. Fallback to linear-repeat"));
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VkClearColorValue {});
    }
    return m_samplerCache.track(sampler);
  }

  Rc<DxvkSampler> SceneManager::patchSampler( const VkFilter filterMode,
                                              const VkSamplerAddressMode addressModeU,
                                              const VkSamplerAddressMode addressModeV,
                                              const VkSamplerAddressMode addressModeW,
                                              const VkClearColorValue borderColor) {
    auto& resourceManager = m_device->getCommon()->getResources();
    // Create a sampler to account for DLSS lod bias and any custom filtering overrides the user has set
    return resourceManager.getSampler(
      filterMode,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      addressModeU,
      addressModeV,
      addressModeW,
      borderColor,
      getTotalMipBias(),
      RtxOptions::useAnisotropicFiltering());
  }

  void SceneManager::addLight(const RtxLegacyLight& light) {
    ScopedCpuProfileZone();
    // Attempt to convert the legacy light to RT

    std::optional<LightData> lightData = LightData::tryCreate(light);

    // Note: Skip adding this light if it is somehow malformed such that it could not be created.
    if (!lightData.has_value()) {
      return;
    }

    const RtLight rtLight = lightData->toRtLight();
    const std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForLight(rtLight.getInitialHash());

    if (pReplacements) {
      const Matrix4 lightTransform = LightUtils::getLightTransform(light);

      ReplacementInstance* replacementInstance = nullptr;

      // TODO(TREX-1091) to implement meshes as light replacements, replace the below loop with a call to drawReplacements.
      for (size_t i = 0; i < pReplacements->size(); i++) {
        const auto& replacement = (*pReplacements)[i];
        if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          LightData replacementLight = replacement.lightData.value();

          // Merge the legacy light into replacements based on overrides
          replacementLight.merge(light);

          // Convert to runtime light
          RtLight rtReplacementLight = replacementLight.toRtLight(&rtLight);

          // Transform the replacement light by the legacy light
          if (replacementLight.relativeTransform()) {
            rtReplacementLight.applyTransform(lightTransform); // note: we dont need to consider the transform of parent replacement light in this scenario, this is detected on mod load and so absolute transform is used
          }

          if (replacementInstance == nullptr) {
            // Handle the root light as a normal light.
            RtLight* newLight;

            // Setup Light Replacement for Anti-Culling
            RtLightAntiCullingType antiCullingType = RtLightAntiCullingType::Ignore;
            if (RtxOptions::AntiCulling::isLightAntiCullingEnabled() && rtLight.getType() == RtLightType::Sphere) {
              antiCullingType = RtLightAntiCullingType::LightReplacement;
            }

            // Apply the light
            newLight = m_lightManager.addLight(rtReplacementLight, antiCullingType);

            // Setup tracking for all the lights created for this replacement.
            if (newLight != nullptr) {
              // This is the first light created, so it will be the root.
              replacementInstance = newLight->getPrimInstanceOwner().getOrCreateReplacementInstance(newLight, PrimInstance::Type::Light, i, pReplacements->size());
            }
          } else {
            // Handle all non-root lights as externally tracked lights - they'll be cleaned up when the root is garbage collected.
            RtLight* existingLight = replacementInstance->prims[i].getLight();
            if (existingLight != nullptr) {
              m_lightManager.updateExternallyTrackedLight(existingLight, rtReplacementLight);
            } else {
              RtLight* newLight = m_lightManager.createExternallyTrackedLight(rtReplacementLight);
              newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
            }
          }
        } else {
          assert(false); // We don't support meshes as children of lights yet.
        }
      }
    } else {
      // This is a light coming from the game directly, so use the appropriate API for filter rules
      m_lightManager.addGameLight(light.Type, rtLight);
    }
  }

  // NV-DXVK [MatChurn]: per-frame material/texture identity churn.
  //
  // See MaterialChurnSample in rtx_scene_manager.h for why this exists and how
  // to read it. In short: the flicker is present in raw albedo but the per-pixel
  // VS identity is stable, so the same surface resolves every frame and returns
  // a different albedo. This measures the rate at which material and texture
  // IDENTITIES are being minted for a scene that is not changing.
  //
  // One aggregate line per gameplay frame. Deliberately not throttled: the
  // dominant failure mode here is a SINGLE-frame dropout, and every fixed-stride
  // sampler this investigation has used sat at one phase forever and reported
  // the artifact clean (rtx.conf trap 9). Deliberately not per-draw either:
  // per-draw logging slows the CS thread enough to change the artifact.
  //
  // Gameplay gate is the project-standard predicate (camera valid + TLAS not
  // near-empty), so menu and loading frames - where a genuine wave of new
  // materials is expected and meaningless - do not enter the timeline.
  void SceneManager::logMaterialChurn() {
    if (!RtxOptions::logMaterialChurn()) {
      return;
    }

    const uint32_t fid = m_device->getCurrentFrameId();
    const bool cameraValid = getCamera().isValid(fid);
    const bool tlasReady = getInstanceTable().size() >= 32u;
    if (!cameraValid || !tlasReady) {
      return;
    }

    auto& textureManager = m_device->getCommon()->getTextureManager();

    MaterialChurnSample cur;
    cur.matLookups      = m_matLookupCount.load(std::memory_order_relaxed);
    cur.matMemoHits     = m_matMemoHitCount.load(std::memory_order_relaxed);
    cur.matPreMiss      = m_matPreMissCount.load(std::memory_order_relaxed);
    cur.matInserts      = m_surfaceMaterialCache.getInsertCount();
    cur.matExtInserts   = m_surfaceMaterialExtensionCache.getInsertCount();
    cur.matClears       = m_surfaceMaterialCache.getClearCount();
    cur.samplerInserts  = m_samplerCache.getInsertCount();
    cur.texInserts      = textureManager.getCacheInsertCount();
    cur.texFrees        = textureManager.getCacheFreeCount();
    cur.texClears       = textureManager.getCacheClearCount();
    cur.bufInserts      = m_bufferCache.getInsertCount();
    cur.bufRetires      = m_bufferCache.getRetireCount();
    cur.bufFrees        = m_bufferCache.getFreeCount();
    cur.bufRevives      = m_bufferCache.getReviveCount();
    cur.imgStamps       = g_newGameImageHashStamps.load(std::memory_order_relaxed);
    cur.mtQueued        = RtxTextureManager::getManagedQueuedCount();
    cur.mtDemoteReq     = RtxTextureManager::getManagedDemoteRequestCount();
    cur.mtVidMem        = RtxTextureManager::getManagedVidMemCount();
    cur.mtViewSwaps     = RtxTextureManager::getManagedMipViewSwapCount();
    cur.mtFailed        = RtxTextureManager::getManagedFailedCount();
    cur.valid           = true;

    // First gameplay frame establishes the baseline. Emitting a delta against a
    // zeroed sample would report the entire load-in as one frame of churn.
    if (!m_prevChurn.valid) {
      m_prevChurn = cur;
      return;
    }

    const MaterialChurnSample& p = m_prevChurn;
    const auto d = [](uint64_t now, uint64_t before) { return now - before; };

    const BindlessResourceManager::TextureTableStats& bl =
      m_bindlessResourceManager.getTextureTableStats();

    // NV-DXVK [BindlessTail]: per-table lengths. The buffer table is the one
    // the shader indexes with Surface.positionBufferIndex / indexBufferIndex,
    // so its shrink is what opens the undefined-slot window; the texture one
    // is carried alongside only so the two can be told apart in the same line.
    const BindlessResourceManager::TableStats& bufTable =
      m_bindlessResourceManager.getTableStats(BindlessResourceManager::Table::Buffers);
    const BindlessResourceManager::TableStats& texTable =
      m_bindlessResourceManager.getTableStats(BindlessResourceManager::Table::Textures);

    // NV-DXVK [Perf.Report]: THE hygiene gate (map section 6). matNew nonzero in
    // steady state means material identities are being minted for a scene that is
    // not changing, and every timing number in the report is then incomparable to
    // any other capture -- so the report prints a loud banner on it rather than
    // letting a reader assume the table is clean.
    perfreport::publish(perfreport::Slot::HygMatNew,
      double(d(cur.matInserts, p.matInserts)));
    perfreport::publish(perfreport::Slot::HygMatTotal,
      double(m_surfaceMaterialCache.getTotalCount()));

    Logger::info(str::format(
      "[MatChurn] f=", fid,
      // Materials. matNew is the headline: a nonzero steady-state value means
      // material identities are being minted for a scene that is not changing.
      " matLookup=", d(cur.matLookups, p.matLookups),
      // NV-DXVK [perf]: matMemo is the single-entry memo's share of matLookup --
      // the calls that returned before resolving a sampler or hashing anything.
      // matLookup - matMemo is the number that actually reaches the prologue, and
      // should track the DRAW count rather than the instance count. matBuild must
      // not move when the memo lands: a memo hit can only occur where the
      // per-frame pre-creation map would also have hit, so any change there means
      // the memo key is wrong.
      " matMemo=", d(cur.matMemoHits, p.matMemoHits),
      " matBuild=", d(cur.matPreMiss, p.matPreMiss),
      " matNew=", d(cur.matInserts, p.matInserts),
      " matExtNew=", d(cur.matExtInserts, p.matExtInserts),
      // matClear separates innocent re-creation (SceneManager::clear() wiped the
      // cache, so EVERY material must be rebuilt) from real identity churn. A
      // matNew spike on a frame with matClear=1 is a scene reset, not the bug.
      " matClear=", d(cur.matClears, p.matClears),
      " matTotal=", m_surfaceMaterialCache.getTotalCount(),
      " matActive=", m_surfaceMaterialCache.getActiveCount(),
      " sampNew=", d(cur.samplerInserts, p.samplerInserts),
      // Textures. texNew/texFree are bindless slot lifetime; imgNew is the
      // producer - a game DxvkImage the material path had never seen.
      " | texNew=", d(cur.texInserts, p.texInserts),
      " texFree=", d(cur.texFrees, p.texFrees),
      " texClear=", d(cur.texClears, p.texClears),
      " texTotal=", textureManager.getCacheTotalCount(),
      " texActive=", textureManager.getCacheActiveCount(),
      " imgNew=", d(cur.imgStamps, p.imgStamps),
      // Replacement-asset streaming. All zero in a mod-less run, which is the
      // point: it rules the ManagedTexture path out instead of assuming it.
      " | mtQueue=", d(cur.mtQueued, p.mtQueued),
      " mtDemote=", d(cur.mtDemoteReq, p.mtDemoteReq),
      " mtVid=", d(cur.mtVidMem, p.mtVidMem),
      " mtSwap=", d(cur.mtViewSwaps, p.mtViewSwaps),
      " mtFail=", d(cur.mtFailed, p.mtFailed),
      // Bindless texture table, as actually written this frame.
      " | blSlots=", bl.slots,
      " blChg=", bl.changed,
      " blDrop=", bl.dropped,
      " blRecov=", bl.recovered,
      " blGrew=", bl.grew,
      // NV-DXVK [BindlessTail]: the BUFFER table, which is the one the
      // device-loss chain runs through and the one nothing was measuring.
      // blSlots above is the TEXTURE table, and in TF2 that only ever grows
      // (blDrop=0 every frame of the 23:57 device loss), so it cannot see the
      // shrink that leaves stale descriptors behind a shrinking count.
      //
      // Read it as: bufReDummied > 0 on a frame means the buffer table SHRANK,
      // and every slot in that window was, before this fix, still serving the
      // previous cycle's descriptor to any out-of-range index. Join those
      // frames against [ReapJoin] live= and [TlasRealloc] to see the scene
      // collapse that produced them.
      //
      // NV-DXVK [stable buffer identity]: bufReDummied should now be rare rather
      // than hundreds a frame. The table no longer collapses and rebuilds every
      // frame, so it only shrinks when the reclaim sweep actually hands slots
      // back, which is what bufFreed counts.
      " | bufSlots=", bufTable.live,
      " bufPeak=", bufTable.peakLive,
      " bufReDummied=", bufTable.reDummied,
      // NV-DXVK [stable buffer identity]: the identity side of the same table.
      // bufLive is how many slots hold a buffer, bufRetired how many of those
      // their owner has released and the sweep has not yet been able to free.
      // The four deltas are per-window; in a settled scene bufNew is the one
      // that must be flat, because a slot that keeps being reissued is exactly
      // the instability this replaced.
      " | bufLive=", m_bufferCache.getActiveCount(),
      " bufTotal=", m_bufferCache.getTotalCount(),
      " bufRetired=", m_bufferCache.getRetiredCount(),
      " bufSpare=", m_bufferCache.getFreeSlotCount(),
      " bufNew=", d(cur.bufInserts, p.bufInserts),
      " bufRetire=", d(cur.bufRetires, p.bufRetires),
      " bufFreed=", d(cur.bufFrees, p.bufFrees),
      " bufRevived=", d(cur.bufRevives, p.bufRevives),
      " texTableSlots=", texTable.live,
      " texTablePeak=", texTable.peakLive,
      " texReDummied=", texTable.reDummied));

    m_prevChurn = cur;
  }

  void SceneManager::prepareSceneData(Rc<RtxContext> ctx, DxvkBarrierSet& execBarriers) {
    ScopedGpuProfileZone(ctx, "Build Scene");

    // [Perf.PrepScene] CPU wall-time sub-split. prepareSceneData is the frame's
    // single biggest cost (~99.9% of entry_tailToBranch, 18-42 ms); split it to
    // find the leaf. Per-frame values, logged every 10 frames at offset (==5),
    // which dodges the %30==0 [SpawnGeomDiag] census (multiples of 30 are
    // %10==0) so its cost doesn't pollute a bucket. Gated on
    // rtx.logPrepSceneSplit, NOT rtx.logSurfaceCoverage — see the emit site.
    // The timestamps themselves are unconditional: nine steady_clock::now()
    // calls at ~41 ns is ~0.4 us against an 18-42 ms stage, and taking them
    // always means the gate can be flipped at runtime with no rebuild.
    auto tPs = std::chrono::steady_clock::now();
    int64_t ps_lightMatch = 0, ps_gc = 0, ps_setup1 = 0, ps_instSetup = 0,
            ps_merge = 0, ps_accel = 0, ps_light = 0,
            ps_surfMat = 0, ps_cull = 0, ps_tlas = 0, ps_tail = 0;
    auto markPs = [&tPs](int64_t& sink) {
      const auto now = std::chrono::steady_clock::now();
      sink = std::chrono::duration_cast<std::chrono::microseconds>(now - tPs).count();
      tPs = now;
    };

    // NV-DXVK [perf]: [Perf.Alloc] allocation/lock-stall probe (dxvk_alloc_probe.h).
    // Armed and zeroed here, read at the emit site at the bottom of this
    // function, so the window is EXACTLY prepareSceneData — the same span the
    // ps_* buckets cover. That is deliberate: the question this answers is
    // "when merge/upload/dynBlas spikes, was it blocked in the allocator?", and
    // a window that matched the whole frame would mix in allocations made
    // during dispatch and present and make the correlation unreadable.
    // Allocations outside this function are simply not counted.
    allocProbe::g_enabled.store(RtxOptions::logPrepSceneSplit(), std::memory_order_relaxed);
    allocProbe::resetAll();

    // NV-DXVK [perf]: [Perf.Stall] blocked-vs-busy sample. Only taken on the
    // frames that actually emit, because QueryThreadCycleTime costs ~0.44 us per
    // call — negligible twice a frame, which is why it works here and was
    // refuted for per-draw buckets. Same window as [Perf.Alloc] and the ps_*
    // buckets: entry to emit of this function.
    const bool psStallSample = RtxOptions::logPrepSceneSplit()
                            && (m_device->getCurrentFrameId() % 10u) == 5u;
    cpuStall::Sample psStall0;
    if (psStallSample) {
      psStall0 = cpuStall::take();
    }

  #ifdef REMIX_DEVELOPMENT
    if (m_device->getCurrentFrameId() == RtxOptions::dumpAllInstancesOnFrame()) {
      // Print all RtInstances for debugging
      printAllRtInstances();
    }
  #endif

    // Needs to happen before garbageCollection to avoid destroying dynamic lights
    m_lightManager.dynamicLightMatching();
    markPs(ps_lightMatch);

    garbageCollection();
    markPs(ps_gc);

    m_graphManager.applySceneOverrides(ctx);

    m_terrainBaker->prepareSceneData(ctx);

    auto& textureManager = m_device->getCommon()->getTextureManager();
    m_bindlessResourceManager.prepareSceneData(ctx, textureManager.getTextureTable(), getBufferTable(), getSamplerTable());

    // NV-DXVK [MatChurn]: emitted here because everything it reports is now
    // final for this frame - all of this frame's draws have resolved their
    // materials and textures, garbageCollection has run, and the bindless table
    // has just been rebuilt, so blChg/blDrop describe the descriptors the ray
    // tracer is about to sample from.
    logMaterialChurn();

    markPs(ps_setup1);

    // NV-DXVK [SpawnGeomDiag]: per-frame TLAS-side instance census. Pairs
    // with the [SpawnGeomDiag] line emitted from D3D11Rtx::EndFrame so we
    // can correlate "how many fanouts/tforms did the d3d11 layer feed in"
    // with "how many RtInstances actually survived to TLAS submission, and
    // where in world space are they". Critical for debugging missing
    // geometry on spawn — a frame with high tforms count but tiny
    // RtInstance count means scene_manager dropped them; a frame with
    // matching counts but instances clustered far from the camera origin
    // means they're being correctly built but possibly outside the camera
    // frustum or behind something.
    {
      const uint32_t fid = m_device->getCurrentFrameId();
      // Throttle: emit on the first 16 frames after device init, then once
      // every 30 frames thereafter, plus always emit when active count is 0
      // (catches "TLAS empty on spawn" instantly).
      const uint32_t total = m_instanceManager.getActiveCount();
      // NV-DXVK [perf]: gated on rtx.logGeomDiag. The emit body walks the whole
      // instance table doing twelve category tests plus AABB math per instance.
      // The atomic snapshot+reset below stays ungated, as its comment requires.
      const bool emitNow = RtxOptions::logGeomDiag()
                        && ((fid < 16) || ((fid % 30) == 0) || (total == 0));
      // [SpawnGeomDiag] snapshot+reset the submit-side counters even on
      // non-emit frames so the per-frame totals don't accumulate across
      // gaps between emits. Reset is unconditional; values are only
      // logged on emitNow frames.
      const uint32_t submitTotal      = s_spawnDiagSubmitTotal.exchange(0, std::memory_order_relaxed);
      const uint32_t submitWithFanout = s_spawnDiagSubmitWithFanout.exchange(0, std::memory_order_relaxed);
      const uint32_t submitTforms     = s_spawnDiagSubmitFanoutTforms.exchange(0, std::memory_order_relaxed);
      if (emitNow) {
        // Categorize instances. Buckets cover the categories we care about
        // for the missing-world-geom debug; everything else falls in
        // 'other'.
        uint32_t catSky = 0, catHidden = 0, catIgnore = 0,
                 catParticle = 0, catTerrain = 0, catWorldUI = 0,
                 catWorldMatte = 0, catBeam = 0, catDecal = 0,
                 catThirdPersonPlayer = 0, other = 0;
        // World-space AABB of all instance origins so we can see the
        // physical extent of the live scene.
        float minX =  std::numeric_limits<float>::max();
        float minY =  std::numeric_limits<float>::max();
        float minZ =  std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();
        const auto& tbl = m_instanceManager.getInstanceTable();
        for (const RtInstance* inst : tbl) {
          if (inst == nullptr) continue;
          const auto cf = inst->getCategoryFlags();
          if (cf.test(InstanceCategories::Sky))                ++catSky;
          if (cf.test(InstanceCategories::Hidden))             ++catHidden;
          if (cf.test(InstanceCategories::Ignore))             ++catIgnore;
          if (cf.test(InstanceCategories::Particle))           ++catParticle;
          if (cf.test(InstanceCategories::Terrain))            ++catTerrain;
          if (cf.test(InstanceCategories::WorldUI))            ++catWorldUI;
          if (cf.test(InstanceCategories::WorldMatte))         ++catWorldMatte;
          if (cf.test(InstanceCategories::Beam))               ++catBeam;
          if (cf.test(InstanceCategories::DecalStatic) ||
              cf.test(InstanceCategories::DecalDynamic) ||
              cf.test(InstanceCategories::DecalSingleOffset) ||
              cf.test(InstanceCategories::DecalNoOffset))      ++catDecal;
          if (cf.test(InstanceCategories::ThirdPersonPlayerModel) ||
              cf.test(InstanceCategories::ThirdPersonPlayerBody))
            ++catThirdPersonPlayer;
          if (!cf.any(InstanceCategories::Sky,
                      InstanceCategories::Hidden,
                      InstanceCategories::Ignore,
                      InstanceCategories::Particle,
                      InstanceCategories::Terrain,
                      InstanceCategories::WorldUI,
                      InstanceCategories::WorldMatte,
                      InstanceCategories::Beam,
                      InstanceCategories::DecalStatic,
                      InstanceCategories::DecalDynamic,
                      InstanceCategories::DecalSingleOffset,
                      InstanceCategories::DecalNoOffset,
                      InstanceCategories::ThirdPersonPlayerModel,
                      InstanceCategories::ThirdPersonPlayerBody)) {
            ++other;
          }
          const Matrix4& xform = inst->getTransform();
          const float wx = xform[3][0];
          const float wy = xform[3][1];
          const float wz = xform[3][2];
          if (std::isfinite(wx) && std::isfinite(wy) && std::isfinite(wz)) {
            if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
            if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
            if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
          }
        }
        const RtCamera& mainCam = m_cameraManager.getCamera(CameraType::Main);
        const Vector3 camPos = mainCam.getPosition();
        const bool camValid = m_cameraManager.isCameraValid(CameraType::Main);
        const bool haveAabb = (minX <= maxX);
        const std::string aabbStr = haveAabb
          ? str::format(" worldAabb=[(", minX, ",", minY, ",", minZ, ")..(",
                                          maxX, ",", maxY, ",", maxZ, ")]")
          : std::string(" worldAabb=<empty>");
        Logger::info(str::format(
          "[SpawnGeomDiag] frame=", fid, " phase=preTlas",
          " inst=", total,
          " cat[sky=", catSky,
          " hidden=", catHidden,
          " ignore=", catIgnore,
          " particle=", catParticle,
          " terrain=", catTerrain,
          " worldUI=", catWorldUI,
          " worldMatte=", catWorldMatte,
          " beam=", catBeam,
          " decal=", catDecal,
          " 3pPlayer=", catThirdPersonPlayer,
          " other=", other, "]",
          " camMain=", camValid ? "valid" : "INVALID",
          " camPos=(", camPos.x, ",", camPos.y, ",", camPos.z, ")",
          " submitTot=", submitTotal,
          " submitFanout=", submitWithFanout,
          " submitTformsSum=", submitTforms,
          aabbStr));

        // Sample the first up-to-12 instances on early frames so we can
        // see real world-space positions in the log. Not throttled by
        // category — we want to see whatever is actually present.
        if (fid < 6 && total > 0) {
          const uint32_t kSample = std::min<uint32_t>(12u, total);
          for (uint32_t i = 0; i < kSample; ++i) {
            const RtInstance* inst = tbl[i];
            if (inst == nullptr) continue;
            const Matrix4& xform = inst->getTransform();
            const BlasEntry* pBlas = inst->getBlas();
            const uint32_t vCount = pBlas != nullptr
              ? pBlas->input.getGeometryData().vertexCount : 0;
            const uint32_t iCount = pBlas != nullptr
              ? pBlas->input.getGeometryData().indexCount : 0;
            const auto cf = inst->getCategoryFlags();
            Logger::info(str::format(
              "[SpawnGeomDiag.sample] frame=", fid, " i=", i,
              " inst=", static_cast<const void*>(inst),
              " blas=", static_cast<const void*>(pBlas),
              " v=", vCount, " idx=", iCount,
              " cat=0x", std::hex, cf.raw(), std::dec,
              " pos=(", xform[3][0], ",", xform[3][1], ",", xform[3][2], ")"));
          }
        }
      }
    }

    // If there are no instances, we should do nothing!
    if (m_instanceManager.getActiveCount() == 0) {
      // Clear the ray portal data before the next frame
      m_rayPortalManager.clear();
      return;
    }

    m_rayPortalManager.prepareSceneData(ctx);
    // Note: only main camera needs to be teleportation corrected as only that one is used for ray tracing & denoising
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::Main));
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::ViewModel));
    m_rayPortalManager.createVirtualCameras(m_cameraManager);
    const bool didTeleport = m_rayPortalManager.detectTeleportationAndCorrectCameraHistory(
      m_cameraManager.getCamera(CameraType::Main),
      m_cameraManager.isCameraValid(CameraType::ViewModel) ? &m_cameraManager.getCamera(CameraType::ViewModel) : nullptr);

    if (m_cameraManager.isCameraCutThisFrame()) {
      // Ignore camera cut events on teleportation so we don't flush the caches
      if (!didTeleport) {
        Logger::info(str::format("Camera cut detected on frame ", m_device->getCurrentFrameId()));
        m_enqueueDelayedClear = true;
      }
    }

    // Initialize/remove opacity micromap manager
    if (RtxOptions::getEnableOpacityMicromap()) {
      if (!m_opacityMicromapManager.get() || 
          // Reset the manager on camera cuts
          m_enqueueDelayedClear) {
        if (m_opacityMicromapManager.get())
          m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());

        m_opacityMicromapManager = std::make_unique<OpacityMicromapManager>(m_device);
        m_instanceManager.addEventHandler(m_opacityMicromapManager->getInstanceEventHandler());
        Logger::info("[RTX] Opacity Micromap: enabled");
      }
    } else if (m_opacityMicromapManager.get()) {
      m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());
      m_opacityMicromapManager = nullptr;
      Logger::info("[RTX] Opacity Micromap: disabled");
    }

    RtxParticleSystemManager& particles = m_device->getCommon()->metaParticleSystem();
    particles.simulate(ctx.ptr());

    m_instanceManager.findPortalForVirtualInstances(m_cameraManager, m_rayPortalManager);
    m_instanceManager.createViewModelInstances(ctx, m_cameraManager, m_rayPortalManager);
    m_instanceManager.createPlayerModelVirtualInstances(ctx, m_cameraManager, m_rayPortalManager);

    // [SpawnGeomDiag.preMerge] Confirms control flow reached this line —
    // pairs with [SpawnGeomDiag.merge] inside mergeInstancesIntoBlas to
    // detect whether the call site executes but the function body doesn't
    // (linker/build mismatch) or whether the call site is being skipped
    // (control flow issue earlier in prepareSceneData).
    {
      static uint32_t sPreMergeFrame = 0;
      if (RtxOptions::logGeomDiag() && (sPreMergeFrame++ % 30u) == 0) {
        Logger::info(str::format(
          "[SpawnGeomDiag.preMerge] frame=", m_device->getCurrentFrameId(),
          " call=", sPreMergeFrame,
          " activeInstances=", m_instanceManager.getActiveCount()));
      }
    }
    markPs(ps_instSetup);
    m_accelManager.mergeInstancesIntoBlas(ctx, execBarriers, textureManager.getTextureTable(), m_cameraManager, m_instanceManager, m_opacityMicromapManager.get());
    markPs(ps_merge);

    // Call on the other managers to prepare their GPU data for the current scene.
    //
    // NV-DXVK [perf]: these two used to share one `accelLight` bucket, which read
    // as "4.9 ms/frame on a light stage" on a map whose light table is empty
    // ([EngineLights.census] resident=0 active=0) and sent one investigation
    // looking for phantom light work. The bucket was a portmanteau of two
    // different managers. Split so the accel half and the light half are never
    // again attributed to each other.
    m_accelManager.prepareSceneData(ctx, execBarriers, m_instanceManager);
    markPs(ps_accel);
    m_lightManager.prepareSceneData(ctx, m_cameraManager);
    markPs(ps_light);

    // Upload surface material buffer BEFORE the GPU culling dispatch so the
    // compute shader can copy template material entries to per-instance slots.
    // For PointInstancer duplicate entries we skip writeGPUData and advance past
    // the gap — the GPU shader will fill those slots.
    {
      DxvkBufferCreateInfo matInfo;
      matInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      matInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
                     | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      matInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      if (m_surfaceMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterials");
        // Note: We duplicate the materials in the buffer so we don't have to do pointer chasing on the GPU
        size_t surfaceMaterialsGPUSize = m_accelManager.getSurfaceCount() * kSurfaceMaterialGPUSize;
        if (m_startInMediumMaterialIndex_inCache != UINT32_MAX) {
          surfaceMaterialsGPUSize += kSurfaceMaterialGPUSize;
        }

        matInfo.size = align(surfaceMaterialsGPUSize, kBufferAlignment);
        if (m_surfaceMaterialBuffer == nullptr || matInfo.size > m_surfaceMaterialBuffer->info().size) {
          m_surfaceMaterialBuffer = m_device->createBuffer(matInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Buffer");
        }

        std::size_t dataOffset = 0;
        uint32_t surfaceIndex = 0;
        std::vector<unsigned char> surfaceMaterialsGPUData(surfaceMaterialsGPUSize);
        for (auto&& pInstance : m_accelManager.getOrderedInstances()) {
          // For PointInstancer duplicates (entries beyond the template), skip
          // writeGPUData — the GPU culling shader copies the template material.
          const auto& surf = pInstance->surface;
          if (surf.instancesToObject != nullptr &&
              surf.surfaceIndexOfFirstInstance != SIZE_MAX &&
              surfaceIndex > surf.surfaceIndexOfFirstInstance) {
            dataOffset += kSurfaceMaterialGPUSize;
          } else {
            auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[surf.surfaceMaterialIndex];
            surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          }
          surfaceIndex++;
        }

        if (m_startInMediumMaterialIndex_inCache != UINT32_MAX) {
          auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[m_startInMediumMaterialIndex_inCache];
          surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          m_startInMediumMaterialIndex = surfaceIndex;
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialsGPUSize);
        assert(surfaceMaterialsGPUData.size() == surfaceMaterialsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialBuffer, 0, surfaceMaterialsGPUData.size(), surfaceMaterialsGPUData.data());
      }
    }
    markPs(ps_surfMat);

    // GPU-driven PointInstancer culling: overwrites visible instance placeholders
    // in m_vkInstanceBuffer with proper transforms and masks, copies per-instance
    // surface and material data from templates. Must run after prepareSceneData
    // (which uploads placeholders) and before buildTlas.
    m_accelManager.dispatchPointInstancerCulling(ctx, m_cameraManager, m_surfaceMaterialBuffer);
    markPs(ps_cull);

    // Build the TLAS
    m_accelManager.buildTlas(ctx);
    markPs(ps_tlas);

    // Todo: These updates require a lot of temporary buffer allocations and memcopies, ideally we should memcpy directly into a mapped pointer provided by Vulkan,
    // but we have to create a buffer to pass to DXVK's updateBuffer for now.
    {
      // Allocate the instance buffer and copy its contents from host to device memory
      DxvkBufferCreateInfo info;
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      // Surface Material Extension Buffer
      if (m_surfaceMaterialExtensionCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterialExtensions");
        const auto surfaceMaterialExtensionsGPUSize = m_surfaceMaterialExtensionCache.getTotalCount() * kSurfaceMaterialGPUSize;

        info.size = align(surfaceMaterialExtensionsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_surfaceMaterialExtensionBuffer == nullptr || info.size > m_surfaceMaterialExtensionBuffer->info().size) {
          m_surfaceMaterialExtensionBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Extension Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> surfaceMaterialExtensionsGPUData(surfaceMaterialExtensionsGPUSize);

        uint32_t surfaceIndex = 0;
        for (auto&& surfaceMaterialExtension : m_surfaceMaterialExtensionCache.getObjectTable()) {
          surfaceMaterialExtension.writeGPUData(surfaceMaterialExtensionsGPUData.data(), dataOffset, surfaceIndex);
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialExtensionsGPUSize);
        assert(surfaceMaterialExtensionsGPUData.size() == surfaceMaterialExtensionsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionsGPUData.size(), surfaceMaterialExtensionsGPUData.data());
      }

      // Volume Material buffer
      if (m_volumeMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateVolumeMaterials");
        const auto volumeMaterialsGPUSize = m_volumeMaterialCache.getTotalCount() * kVolumeMaterialGPUSize;

        info.size = align(volumeMaterialsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_volumeMaterialBuffer == nullptr || info.size > m_volumeMaterialBuffer->info().size) {
          m_volumeMaterialBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Volume Material Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> volumeMaterialsGPUData(volumeMaterialsGPUSize);

        for (auto&& volumeMaterial : m_volumeMaterialCache.getObjectTable()) {
          volumeMaterial.writeGPUData(volumeMaterialsGPUData.data(), dataOffset);
        }

        assert(dataOffset == volumeMaterialsGPUSize);
        assert(volumeMaterialsGPUData.size() == volumeMaterialsGPUSize);

        ctx->writeToBuffer(m_volumeMaterialBuffer, 0, volumeMaterialsGPUData.size(), volumeMaterialsGPUData.data());
      }
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // Update stats
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBlasCount, AccelManager::getBlasCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBufferCount, m_bufferCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxTextureCount, textureManager.getTextureTable().size());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxInstanceCount, m_instanceManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialCount, m_surfaceMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialExtensionCount, m_surfaceMaterialExtensionCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxVolumeMaterialCount, m_volumeMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxLightCount, m_lightManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSamplers, m_samplerCache.getActiveCount());

    auto capturer = m_device->getCommon()->capturer();
    if (m_device->getCurrentFrameId() == m_beginUsdExportFrameNum) {
      capturer->triggerNewCapture();
    }
    capturer->step(ctx, ctx->getCommonObjects()->getLastKnownWindowHandle());

    // Clear the ray portal data before the next frame
    m_rayPortalManager.clear();

    // Check Anti-Culling Support:
    // When the game doesn't set up the View Matrix, we must disable Anti-Culling to prevent visual corruption.
    m_isAntiCullingSupported = (getCamera().getViewToWorld() != Matrix4d());

    markPs(ps_tail);
    // [Perf.PrepScene] per-frame sub-split (us) of the entry_tailToBranch leaf.
    // Throttled at fid%10==5, which still dodges the %30==0 [SpawnGeomDiag]
    // census (30 % 10 == 0, so the two never collide) and is a superset of the
    // old %30==15 sample points.
    //
    // Gate: rtx.logPrepSceneSplit. This was rtx.logSurfaceCoverage, which is a
    // ~104 ms/frame switch (it arms 52 atomics per primary hit in the shader),
    // so reading the split required first quadrupling the frame — every number
    // it printed described a workload that does not exist in normal play. The
    // new gate costs one bool test plus one Logger::warn per ten frames; at
    // ~10 fps that is ~1 line/s, next to nothing beside [NrcBounds] which
    // prints every frame.
    //
    // Ten frames rather than thirty because this stage swings 2.3x (18.5 /
    // 26.4 / 42.4 ms) on a STATIC camera with flat instance counts. At ~10 fps
    // the old throttle sampled once per three seconds, and three points cannot
    // show a distribution — the variance is itself the lead, so it has to be
    // sampled densely enough to see which sub-stage carries it.
    if (RtxOptions::logPrepSceneSplit() && (m_device->getCurrentFrameId() % 10u) == 5u) {
      // NV-DXVK [Perf.Report]: the once-per-frame SERIAL pass -- the largest
      // indivisible block anywhere in the frame, and nothing overlaps it. Values
      // here are microseconds for ONE frame (this line is per-frame, not a
      // window), so they convert straight to ms without a frame divisor.
      {
        const double kUsToMs = 1.0 / 1000.0;
        perfreport::publish(perfreport::Slot::PrepSceneMs,
          double(ps_gc + ps_setup1 + ps_instSetup + ps_merge + ps_accel
               + ps_light + ps_surfMat + ps_cull + ps_tlas + ps_tail) * kUsToMs);
        perfreport::publish(perfreport::Slot::PrepMergeMs,   double(ps_merge)   * kUsToMs);
        perfreport::publish(perfreport::Slot::PrepGcMs,      double(ps_gc)      * kUsToMs);
        perfreport::publish(perfreport::Slot::PrepSurfMatMs, double(ps_surfMat) * kUsToMs);
        perfreport::publish(perfreport::Slot::PrepSetupMs,
          double(ps_setup1 + ps_instSetup) * kUsToMs);
        perfreport::publish(perfreport::Slot::PrepTlasMs,
          double(ps_tlas + ps_accel + ps_light) * kUsToMs);
      }

      Logger::warn(str::format("[Perf.PrepScene] frame=", m_device->getCurrentFrameId(),
        " lightMatch=", ps_lightMatch, " gc=", ps_gc, " setup1=", ps_setup1,
        " instSetup=", ps_instSetup, " merge=", ps_merge,
        " accel=", ps_accel, " light=", ps_light,
        " surfMat=", ps_surfMat, " cull=", ps_cull, " tlas=", ps_tlas, " tail=", ps_tail,
        " inst=", m_instanceManager.getActiveCount(),
        " surf=", m_accelManager.getSurfaceCount()));

      // NV-DXVK [Perf.GeoChurn] — this frame's geometry re-cache decisions,
      // emitted on the same frames as [Perf.PrepScene] and [Perf.Merge] so the
      // three are directly cross-readable:
      //   entries      every BlasEntry processed this frame (dynamic AND
      //                merged-routed) — the superset [Perf.Merge] cannot see
      //   slow         should equal the interleaveGeometry dispatch count in an
      //                Nsight capture of the same scene (~200)
      //   updInst      entries that cost nothing. Every one moved out of
      //                build/updBvh into this column is free frame time.
      // The `why:` counters overlap by design; read them as "which input is
      // churning", not as a partition of the re-cache count.
      {
        GeoChurnStats snapshot;
        {
          std::lock_guard<dxvk::mutex> lk(s_geoChurnMu);
          if (s_geoChurn.frame == m_device->getCurrentFrameId()) {
            snapshot = s_geoChurn;
          }
        }
        Logger::warn(str::format("[Perf.GeoChurn] frame=", m_device->getCurrentFrameId(),
          " entries=", snapshot.entries,
          " build=", snapshot.build, " updBvh=", snapshot.updBvh, " updInst=", snapshot.updInst,
          " isNew=", snapshot.isNewCnt,
          " | why: idx=", snapshot.rIdx, " pos=", snapshot.rPos, " vs=", snapshot.rVs,
          " bone=", snapshot.rBone, " pend=", snapshot.rPend, " norm=", snapshot.rNorm,
          " | path: slow=", snapshot.slowPath, " fast=", snapshot.fastPath,
          " | vtx: recached=", snapshot.vtxRecached, "/", snapshot.vtxTotal));

        // Per-VS breakdown, top 8 by re-cache count. Without this the aggregate
        // says "200 entries re-cache" but not WHICH geometry, and the fix
        // (cache it / stabilise its hash / stop re-uploading it) depends
        // entirely on which shader's draws dominate the column.
        if (!snapshot.byVs.empty()) {
          std::vector<std::pair<uint64_t, GeoChurnStats::VsStat>> ranked(snapshot.byVs.begin(), snapshot.byVs.end());
          std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.second.recache > b.second.recache; });
          const size_t cap = std::min<size_t>(ranked.size(), 8u);
          for (size_t i = 0; i < cap; ++i) {
            if (ranked[i].second.recache == 0) {
              break;  // the rest are all-free, nothing to report
            }
            Logger::warn(str::format("[Perf.GeoChurn]   vs=0x", std::hex, ranked[i].first, std::dec,
              " recache=", ranked[i].second.recache, "/", ranked[i].second.entries,
              " vtx=", ranked[i].second.vtx));
          }
        }
      }
      // NV-DXVK [perf, GPU index stash]: pool health. `created` must FLATTEN
      // after warmup — it counts real device allocations, which is exactly the
      // per-draw churn the pool exists to remove. A `created` that keeps
      // climbing with `reused` near zero means recycling is broken (handles
      // being retained, or size classes fragmenting), not merely a cold cache.
      if (m_indexStashPool) {
        Logger::warn(str::format("[IdxStashPool] created=", m_indexStashPool->created(),
          " reused=", m_indexStashPool->reused(),
          " agedOut=", m_indexStashPool->released(),
          " freeBytes=", m_indexStashPool->bytesHeld()));
      }

      // NV-DXVK [perf]: [Perf.Alloc] — allocator/lock stalls incurred INSIDE this
      // prepareSceneData call. Pair it with the [Perf.PrepScene] line printed
      // just above (same frame, same window).
      //
      // Each mechanism reports total us, worst SINGLE occurrence us, and count.
      // maxUs is the field that matters: the spikes being chased are one bad
      // acquire, and an average over hundreds of fast calls would bury it.
      //
      //   memAlloc   DxvkMemoryAllocator::alloc overall
      //   typeLock   contended waits on the per-memory-type mutex
      //   vkAlloc    vkAllocateMemory (the driver call)
      //   freeChunks freeEmptyChunks, which runs with the type mutex dropped
      //   slice      DxvkBuffer::allocSlice free-list refills (+ refills count)
      //
      // How to read it against the spiking bucket:
      //   a maxUs in the same ms range as the spike  -> that is the stall, fix it
      //   all buckets ~0 on a spiking frame          -> NOT the allocator; the
      //                                                 next suspects are the
      //                                                 texture/staging paths
      //                                                 and resource tracking
      // NV-DXVK [perf]: [Perf.Stall] — is the spiking bucket BLOCKED or BUSY?
      // See rtx_cpu_stall_probe.h. Read this FIRST when a ps_* bucket spikes:
      //   blockedUs ~= the spike  -> the thread was off-core; find what it waits
      //                              on. busyPct will be well under 100.
      //   cpuUs ~= wallUs         -> real execution. Then check faults: a large
      //                              delta means memory-stalled work (first-touch
      //                              on grown vectors, working-set trimming),
      //                              not compute, and the fix is allocation
      //                              patterns rather than the loop itself.
      // faults is process-wide, so other threads contribute; treat a big delta
      // as a pointer, not proof.
      if (psStallSample) {
        const auto d = cpuStall::diff(psStall0, cpuStall::take());
        Logger::warn(str::format("[Perf.Stall] frame=", m_device->getCurrentFrameId(),
          " wallUs=", d.wallUs,
          " cpuUs=", d.cpuUs,
          " blockedUs=", d.blockedUs,
          " busyPct=", d.busyPct,
          " faults=", d.faults,
          " wsMB=", (d.workingSet >> 20)));
      }

      {
        using namespace allocProbe;
        const uint64_t totalStallNs = g_memAlloc.totalNs.load(std::memory_order_relaxed)
                                    + g_typeLock.totalNs.load(std::memory_order_relaxed)
                                    + g_freeChunks.totalNs.load(std::memory_order_relaxed)
                                    + g_slice.totalNs.load(std::memory_order_relaxed);
        auto us = [](uint64_t ns) { return ns / 1000u; };
        Logger::warn(str::format("[Perf.Alloc] frame=", m_device->getCurrentFrameId(),
          " stallTotal=", us(totalStallNs),
          " | memAlloc=", us(g_memAlloc.totalNs.load(std::memory_order_relaxed)),
            " max=", us(g_memAlloc.maxNs.load(std::memory_order_relaxed)),
            " n=", g_memAlloc.count.load(std::memory_order_relaxed),
          " | typeLock=", us(g_typeLock.totalNs.load(std::memory_order_relaxed)),
            " max=", us(g_typeLock.maxNs.load(std::memory_order_relaxed)),
            " n=", g_typeLock.count.load(std::memory_order_relaxed),
          " | vkAlloc=", us(g_vkAlloc.totalNs.load(std::memory_order_relaxed)),
            " max=", us(g_vkAlloc.maxNs.load(std::memory_order_relaxed)),
            " n=", g_vkAlloc.count.load(std::memory_order_relaxed),
          " | freeChunks=", us(g_freeChunks.totalNs.load(std::memory_order_relaxed)),
            " max=", us(g_freeChunks.maxNs.load(std::memory_order_relaxed)),
            " n=", g_freeChunks.count.load(std::memory_order_relaxed),
          " | slice=", us(g_slice.totalNs.load(std::memory_order_relaxed)),
            " max=", us(g_slice.maxNs.load(std::memory_order_relaxed)),
            " refills=", g_sliceRefills.load(std::memory_order_relaxed)));
      }
    }
  }

  static_assert(std::is_same_v< decltype(RtSurface::objectPickingValue), ObjectPickingValue>);

  void SceneManager::submitExternalDraw(Rc<DxvkContext> ctx, ExternalDrawState&& state) {
    if (m_externalSampler == nullptr) {
      auto s = DxvkSamplerCreateInfo {};
      {
        s.magFilter = VK_FILTER_LINEAR;
        s.minFilter = VK_FILTER_LINEAR;
        s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        s.mipmapLodBias = 0.f;
        s.mipmapLodMin = 0.f;
        s.mipmapLodMax = 0.f;
        s.useAnisotropy = VK_FALSE;
        s.maxAnisotropy = 1.f;
        s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.compareToDepth = VK_FALSE;
        s.compareOp = VK_COMPARE_OP_NEVER;
        s.borderColor = VkClearColorValue {};
        s.usePixelCoord = VK_FALSE;
      }
      m_externalSampler = m_device->createSampler(s);
    }

    {
      state.drawCall.materialData.samplers[0] = m_externalSampler;
      state.drawCall.materialData.samplers[1] = m_externalSampler;
    }
    {
      const RtCamera& rtCamera = ctx->getCommonObjects()->getSceneManager().getCameraManager()
        .getCamera(state.cameraType);
      state.drawCall.transformData.worldToView = Matrix4 { rtCamera.getWorldToView() };
      state.drawCall.transformData.viewToProjection = Matrix4 { rtCamera.getViewToProjection() };
      state.drawCall.transformData.objectToView = state.drawCall.transformData.worldToView * state.drawCall.transformData.objectToWorld;
    }

    if (!state.gpuInstancingTransforms.empty()) {
      m_externalGpuInstancingTransforms.push_back(std::move(state.gpuInstancingTransforms));
      state.drawCall.transformData.instancesToObject = &m_externalGpuInstancingTransforms.back();
    }

    for (const RasterGeometry& submesh : m_pReplacer->accessExternalMesh(state.mesh)) {
      state.drawCall.geometryData = submesh;
      state.drawCall.geometryData.cullMode = state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;

      const MaterialData* material = m_pReplacer->accessExternalMaterial(submesh.externalMaterial);
      if (material != nullptr) {
        state.drawCall.materialData.setHashOverride(material->getHash());
      } 

      const RtxParticleSystemDesc* pParticles = nullptr;
      if(state.optionalParticleDesc.has_value()) {
        pParticles = &state.optionalParticleDesc.value();
      }

      processDrawCallState(ctx, state.drawCall, material != nullptr ? MaterialData(*material) : LegacyMaterialData().as<OpaqueMaterialData>(), nullptr, pParticles);
    }
  }

  namespace {
    bool ifTrue_andThenSetFalse(std::atomic_bool& atomicBool) {
      bool expected = true;
      if (atomicBool.compare_exchange_strong(expected, false)) {
        return true;
      }
      return false;
    }
  } // unnamed

  void SceneManager::requestTextureVramFree() {
    m_forceFreeTextureMemory.store(true);
  }

  void SceneManager::requestVramCompaction() {
    m_forceFreeUnusedDxvkAllocatorChunks.store(true);
  }

  void SceneManager::manageTextureVram() {
    bool freeUnused = false;
    bool freeTextures = false;
    {
      if (ifTrue_andThenSetFalse(m_forceFreeTextureMemory)) {
        freeTextures = true;
        freeUnused = true;
      }
      if (ifTrue_andThenSetFalse(m_forceFreeUnusedDxvkAllocatorChunks)) {
        freeUnused = true;
      }
    }

    if (freeTextures) {
      m_device->getCommon()->getTextureManager().clear();

      if (m_opacityMicromapManager) {
        m_opacityMicromapManager->clear();
      }
    }

    // NV-DXVK [Perf.ChunkTrim] (2026-08-06): the automatic trigger this call has
    // never had. Until now freeUnusedChunks() was reachable ONLY through the two
    // m_forceFree* atomics above, both of which are manual UI actions -- so on a
    // normal play session the allocator's high-water mark was never given back,
    // which is precisely why VRAM was observed to climb and never recover.
    //
    // MEASURED, [Perf.MemCat], BT mission steady state:
    //   heap0(VRAM) alloc=7569MB used=5313MB slack=2255MB
    // 30% of everything taken from the driver held in chunks with nothing
    // suballocated in them -- a bigger line item than the game's own textures
    // (appTex=2329MB) or the acceleration structures (rtxAccel=2229MB).
    //
    // SAFETY IS INHERITED, NOT ASSUMED: this sets the SAME freeUnused flag the
    // force path sets, so the trim still happens at the same point in the frame
    // (SceneManager::onFrameEnd), on the same thread, through the same call.
    // Nothing about the trim changes -- only when it is asked for.
    //
    // Tested against the LARGEST heap's slack rather than the sum, so a big
    // VRAM heap cannot be masked by a small tidy system heap.
    bool         autoTrim      = false;
    VkDeviceSize autoTrimSlack = 0;

    if (!freeUnused && RtxOptions::autoFreeUnusedChunks()) {
      DxvkMemoryAllocator& memoryManager = m_device->getCommon()->memoryManager();
      const uint32_t heapCount = memoryManager.getMemoryProperties().memoryHeapCount;

      VkDeviceSize worstSlack = 0;
      for (uint32_t i = 0; i < heapCount; ++i) {
        const DxvkMemoryStats stats = m_device->getMemoryStats(i);
        const VkDeviceSize allocated = stats.totalAllocated();
        const VkDeviceSize used      = stats.totalUsed();

        if (allocated > used)
          worstSlack = std::max(worstSlack, allocated - used);
      }

      const VkDeviceSize slackThreshold =
        VkDeviceSize(RtxOptions::autoFreeUnusedChunksSlackMB()) << 20;

      // BACK-OFF AFTER A FRUITLESS TRIM. Slack over the threshold does NOT mean
      // there is anything to recover: only entirely-empty chunks can be freed,
      // and slack sitting inside partially-used chunks is fragmentation that no
      // trim reaches. MEASURED 2026-08-06: after 7 productive trims recovered
      // 1360 MB, the next 50 consecutive trims freed 0 chunks each while slack
      // sat at ~1900 MB -- fifty heap walks, each taking every per-memory-type
      // lock, to accomplish nothing. Without this the policy retries forever.
      //
      // The retry condition is slack GROWTH, and that is exact rather than a
      // heuristic. A chunk can only become newly empty if either allocated rose
      // (new chunks exist) or used fell (suballocations were released), and
      // slack = allocated - used rises in both cases. Slack falling means used
      // rose, i.e. more allocation, which cannot empty a chunk. So if slack has
      // not grown since a trim that found nothing, there is provably nothing
      // new to find.
      const VkDeviceSize retryDelta =
        VkDeviceSize(RtxOptions::autoFreeUnusedChunksRetryDeltaMB()) << 20;
      const bool worthRetrying =
        (m_fruitlessTrimSlack == 0) || (worstSlack > m_fruitlessTrimSlack + retryDelta);

      if (worstSlack >= slackThreshold && worthRetrying) {
        const auto now = std::chrono::steady_clock::now();
        const bool neverTrimmed =
          m_lastChunkTrimTime == std::chrono::steady_clock::time_point::min();
        const int64_t sinceMs = neverTrimmed ? 0 :
          std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_lastChunkTrimTime).count();

        if (neverTrimmed || sinceMs >= int64_t(RtxOptions::autoFreeUnusedChunksCooldownMs())) {
          m_lastChunkTrimTime = now;
          freeUnused    = true;
          autoTrim      = true;
          autoTrimSlack = worstSlack;
        }
      }
    }

    if (freeUnused) {
      // DXVK doesnt free chunks for us by default (its high water mark) so force release some memory back to the system here.

      // NV-DXVK [Perf.ChunkTrim]: report what the trim actually RECOVERED, not
      // what triggered it. Only genuinely empty chunks can be released; slack
      // inside a partially-used chunk is fragmentation and no policy reaches it,
      // so recovered is expected to be well under the slack that fired this.
      // Tune rtx.autoFreeUnusedChunksSlackMB against this line, not a guess.
      DxvkMemoryAllocator& memoryManager = m_device->getCommon()->memoryManager();
      const uint32_t heapCount = memoryManager.getMemoryProperties().memoryHeapCount;

      VkDeviceSize beforeAllocated = 0;
      VkDeviceSize beforeUsed      = 0;
      for (uint32_t i = 0; i < heapCount; ++i) {
        const DxvkMemoryStats stats = m_device->getMemoryStats(i);
        beforeAllocated += stats.totalAllocated();
        beforeUsed      += stats.totalUsed();
      }

      // THE AUTOMATIC PATH IS BUDGETED, THE MANUAL ONE IS NOT, and the
      // difference is deliberate. A chunk is 320 MB device-local and releasing
      // it is a vkFreeMemory of the whole allocation, so freeing every empty
      // chunk at once is a multi-GB burst of driver work on one frame. That is
      // fine when the USER asked for it at a moment of their choosing, and it
      // is a visible stutter when a background policy decides mid-firefight.
      // So: forced -> 0 (unlimited, unchanged behaviour); auto -> a small
      // per-trim budget, which still drains the heap fully, just spread over
      // seconds. 2255 MB of slack is ~7 chunks, i.e. ~7 seconds at the default
      // 1 chunk/second, with one vkFreeMemory on each of those frames.
      const uint32_t trimBudget =
        autoTrim ? RtxOptions::autoFreeUnusedChunksMaxPerTrim() : 0u;

      const uint32_t chunksFreed = memoryManager.freeUnusedChunks(trimBudget);

      VkDeviceSize afterAllocated = 0;
      for (uint32_t i = 0; i < heapCount; ++i)
        afterAllocated += m_device->getMemoryStats(i).totalAllocated();

      const VkDeviceSize recovered =
        (beforeAllocated > afterAllocated) ? (beforeAllocated - afterAllocated) : 0;

      // Arm or clear the back-off described above. A trim that freed nothing
      // records the slack level it gave up at, so the policy stays quiet until
      // slack grows past it; a productive trim clears the record, because more
      // chunks may now be reachable and the next attempt should be allowed.
      // Only the automatic path participates -- an explicit force-free is a
      // user decision and must never be suppressed by a previous null result.
      if (autoTrim)
        m_fruitlessTrimSlack = (chunksFreed == 0) ? autoTrimSlack : 0;

      Logger::warn(str::format(
        "[Perf.ChunkTrim] trigger=", (autoTrim ? "auto" : "forced"),
        " budget=", trimBudget, (trimBudget == 0 ? "(unlimited)" : ""),
        " chunks=", chunksFreed,
        " allocated=", (beforeAllocated >> 20), "->", (afterAllocated >> 20), "MB",
        " recovered=", (recovered >> 20), "MB",
        " used=", (beforeUsed >> 20), "MB",
        " slack=", ((beforeAllocated > beforeUsed) ? ((beforeAllocated - beforeUsed) >> 20) : 0), "MB",
        (autoTrim && chunksFreed == 0) ? " -> backing off until slack grows" : ""));
    }
  }

  void SceneManager::printAllRtInstances() {
  #ifdef REMIX_DEVELOPMENT
    
    const auto& instances = m_instanceManager.getInstanceTable();
    Logger::info(str::format("=== Printing all RtInstances (", instances.size(), " total) ==="));
    
    for (size_t i = 0; i < instances.size(); ++i) {
      const RtInstance* instance = instances[i];
      if (instance != nullptr) {
        Logger::info(str::format("Instance ", i, ":"));
        instance->printDebugInfo();
      } else {
        Logger::warn(str::format("Instance ", i, ": nullptr"));
      }
    }
    
    Logger::info("=== End RtInstances Print ===");
  #endif
  }

  void SceneManager::trackReplacementMaterialHash(XXH64_hash_t materialHash) {
    if (materialHash != kEmptyHash) {
      m_currentFrameReplacementMaterialHashes[materialHash]++;
    }
  }

  bool SceneManager::isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const {
    return m_currentFrameReplacementMaterialHashes.find(materialHash) != m_currentFrameReplacementMaterialHashes.end();
  }

  uint32_t SceneManager::getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const {
    auto it = m_currentFrameReplacementMaterialHashes.find(materialHash);
    return (it != m_currentFrameReplacementMaterialHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameReplacementMaterialHashes() {
    m_currentFrameReplacementMaterialHashes.clear();
  }

  void SceneManager::trackMeshHash(XXH64_hash_t meshHash) {
    if (meshHash != kEmptyHash) {
      m_currentFrameMeshHashes[meshHash]++;
    }
  }

  bool SceneManager::isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const {
    return m_currentFrameMeshHashes.find(meshHash) != m_currentFrameMeshHashes.end();
  }

  uint32_t SceneManager::getMeshHashUsageCount(XXH64_hash_t meshHash) const {
    auto it = m_currentFrameMeshHashes.find(meshHash);
    return (it != m_currentFrameMeshHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameMeshHashes() {
    m_currentFrameMeshHashes.clear();
  }

}  // namespace nvvk
