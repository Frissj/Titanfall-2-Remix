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
#include <unordered_set>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstring>

#include "rtx_asset_replacer.h"
#include "rtx_scene_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_buffer.h"
#include "rtx_context.h"
#include "rtx_options.h"
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

  // NV-DXVK [SceneClearProbe]: defined in rtx_camera_manager.cpp; used in
  // SceneManager::clear() to gate the unconditional log on gameplay so we
  // don't emit ~1024 lines during pre-gameplay loading (every load-frame
  // invalid-scene triggers a clear with default sceneKeepAliveFrames=0).
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;
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
    InstanceEventHandler instanceEvents(this);
    instanceEvents.onInstanceAddedCallback = [this](RtInstance& instance) { onInstanceAdded(instance); };
    instanceEvents.onInstanceUpdatedCallback = [this](RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, bool hasTransformChanged, bool hasVerticesChanged, bool isFirstUpdateThisFrame) { onInstanceUpdated(instance, drawCall, material, hasTransformChanged, hasVerticesChanged, isFirstUpdateThisFrame); };
    instanceEvents.onInstanceDestroyedCallback = [this](RtInstance& instance) { onInstanceDestroyed(instance); };
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
    auto blasEntryGarbageCollection = [&](auto& iter, auto& entries) -> void {
      if (iter->second.frameLastTouched < oldestFrame) {
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
  }

  void SceneManager::onDestroy() {
    m_accelManager.onDestroy();
    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onDestroy();
    }
  }

  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, RaytraceGeometry& inOutGeometry) {
    ScopedCpuProfileZone();
    ObjectCacheState result = ObjectCacheState::KBuildBVH;
    const RasterGeometry& input = drawCallState.getGeometryData();

    // Determine the optimal object state for this geometry
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

    // When smooth normals state changes (added or removed), promote to kUpdateBVH so the vertex
    // data is re-interleaved and the smooth normals dispatch runs (or original normals are restored).
    if (needsSmoothNormals != output.smoothNormalsApplied && result == ObjectCacheState::kUpdateInstance) {
      result = ObjectCacheState::kUpdateBVH;
    }
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
        output.indexCacheBuffer = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Index Cache Buffer");

        if (!RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output)) {
          ONCE(Logger::err("processGeometryInfo: failed to cache index data on GPU"));
          return ObjectCacheState::kInvalid;
        }

        output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);

        info.size = align(vertexBufferSize, CACHE_LINE_SIZE);
        output.historyBuffer[0] = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

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
          output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

          // Invalidate the current buffer
          output.historyBuffer[1] = nullptr;

          // Mark this object for realignment
          invalidateHistory = true;
        }

        // Use the previous updates vertex data for previous position lookup
        std::swap(output.historyBuffer[0], output.historyBuffer[1]);

        if (output.historyBuffer[0].ptr() == nullptr) {
          // First frame this object has been dynamic need to allocate a 2nd frame of data to preserve history.
          output.historyBuffer[0] = m_device->createBuffer(output.historyBuffer[1]->info(), memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
        } 

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

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
    updateBufferCache(output);

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

    m_bufferCache.clear();
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

    // Not currently safe to cache these across frames (due to texture indices and rtx options potentially changing)
    m_preCreationSurfaceMaterialMap.clear();

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
        if (deficit && d.lastLoggedFid != fid) {
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

  void SceneManager::submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData) {
    ScopedCpuProfileZone();
    s_spawnDiagSubmitTotal.fetch_add(1, std::memory_order_relaxed);

    // NV-DXVK [DropTrace]: count this draw if it's a dropship sub-mesh.
    {
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
    {
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
    {
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
      const bool inGameplayFloorTraceRecv =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      static std::mutex sFloorTraceMtx;
      static uint32_t sFloorTraceFrame = UINT32_MAX;
      static uint32_t sFloorTracePerFrame = 0;
      const uint32_t fid = m_device->getCurrentFrameId();
      bool emit = false;
      {
        std::lock_guard<std::mutex> lk(sFloorTraceMtx);
        if (fid != sFloorTraceFrame) {
          sFloorTraceFrame = fid;
          sFloorTracePerFrame = 0;
        }
        if (inGameplayFloorTraceRecv && sFloorTracePerFrame < 80) {
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

    MaterialData renderMaterialData = determineMaterialData(overrideMaterialData, input);

    if (pReplacements != nullptr) {
      drawReplacements(ctx, &input, pReplacements, renderMaterialData);
    } else {
      processDrawCallState(ctx, input, renderMaterialData, nullptr, nullptr);
    }
  }

  MaterialData SceneManager::determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input) {
    // First see if we have an explicit override
    if (overrideMaterialData != nullptr) {
      return *overrideMaterialData;
    } 

    // test if any direct material replacements exist
    MaterialData* pReplacementMaterial = m_pReplacer->getReplacementMaterial(input.getMaterialData().getHash());
    if (pReplacementMaterial != nullptr) {
      // Make a copy - dont modify the replacement data.
      MaterialData renderMaterialData = *pReplacementMaterial;
      // merge in the input material from game
      renderMaterialData.mergeLegacyMaterial(input.getMaterialData());
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
      return sHighlightMaterialData;
    }

    // Check if a Ray Portal override is needed
    size_t rayPortalTextureIndex;
    if (RtxOptions::getRayPortalTextureIndex(input.getMaterialData().getHash(), rayPortalTextureIndex)) {
      assert(rayPortalTextureIndex < maxRayPortalCount);
      assert(rayPortalTextureIndex < std::numeric_limits<uint8_t>::max());

      MaterialData renderMaterialData = input.getMaterialData().as<RayPortalMaterialData>();
      renderMaterialData.getRayPortalMaterialData().setRayPortalIndex(rayPortalTextureIndex);
      return renderMaterialData;
    }

    // Standard legacy material conversion
    return input.getMaterialData().as<OpaqueMaterialData>();
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

  void SceneManager::updateBufferCache(RaytraceGeometry& newGeoData) {
    ScopedCpuProfileZone();
    if (newGeoData.indexBuffer.defined()) {
      newGeoData.indexBufferIndex = m_bufferCache.track(newGeoData.indexBuffer);
    } else {
      newGeoData.indexBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.normalBuffer.defined()) {
      newGeoData.normalBufferIndex = m_bufferCache.track(newGeoData.normalBuffer);
    } else {
      newGeoData.normalBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.color0Buffer.defined()) {
      newGeoData.color0BufferIndex = m_bufferCache.track(newGeoData.color0Buffer);
    } else {
      newGeoData.color0BufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.texcoordBuffer.defined()) {
      newGeoData.texcoordBufferIndex = m_bufferCache.track(newGeoData.texcoordBuffer);
    } else {
      newGeoData.texcoordBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.positionBuffer.defined()) {
      newGeoData.positionBufferIndex = m_bufferCache.track(newGeoData.positionBuffer);
    } else {
      newGeoData.positionBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.previousPositionBuffer.defined()) {
      newGeoData.previousPositionBufferIndex = m_bufferCache.track(newGeoData.previousPositionBuffer);
    } else {
      newGeoData.previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
    }

    // NV-DXVK: TF2 worldspace VGUI auxiliary structured buffers. m_bufferCache
    // dedups by DxvkBufferSlice so a single font atlas / styles table that
    // gets reused across many VGUI panel draws only consumes one bindless
    // slot per type.
    if (newGeoData.vguiFontBoundsBuffer.defined()) {
      newGeoData.vguiFontBoundsBufferIndex = m_bufferCache.track(newGeoData.vguiFontBoundsBuffer);
    } else {
      newGeoData.vguiFontBoundsBufferIndex = kSurfaceInvalidBufferIndex;
    }
    if (newGeoData.vguiImgBoundsBuffer.defined()) {
      newGeoData.vguiImgBoundsBufferIndex = m_bufferCache.track(newGeoData.vguiImgBoundsBuffer);
    } else {
      newGeoData.vguiImgBoundsBufferIndex = kSurfaceInvalidBufferIndex;
    }
    if (newGeoData.vguiStylesBuffer.defined()) {
      newGeoData.vguiStylesBufferIndex = m_bufferCache.track(newGeoData.vguiStylesBuffer);
    } else {
      newGeoData.vguiStylesBufferIndex = kSurfaceInvalidBufferIndex;
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
  
  void SceneManager::onSceneObjectDestroyed(const BlasEntry& blas) {
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
    auto capturer = m_device->getCommon()->capturer();
    if (hasTransformChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
    }

    if (hasVerticesChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::PositionsUpdate);
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::NormalsUpdate);
    }

    // Create and bind the RT material
    const RtSurfaceMaterial& surfaceMaterial = createSurfaceMaterial(material, drawCall);

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
    if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      static std::mutex s_skyDiagMu;
      static std::unordered_set<uint64_t> s_skyDiagSeen;
      const LegacyMaterialData& skyDiagMat = drawCall.getMaterialData();
      const TextureRef& skyDiagAlb = skyDiagMat.getColorTexture();
      const XXH64_hash_t albHash = (skyDiagAlb.isValid() && !skyDiagAlb.isImageEmpty())
        ? skyDiagAlb.getImageHash() : 0ull;
      const uint64_t vsH = static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
      const uint64_t key = vsH ^ albHash ^ (uint64_t(static_cast<uint32_t>(drawCall.cameraType)) << 56);
      bool firstSkyDiag = false;
      {
        std::lock_guard<std::mutex> g(s_skyDiagMu);
        if (s_skyDiagSeen.size() < 512 && s_skyDiagSeen.insert(key).second)
          firstSkyDiag = true;
      }
      if (firstSkyDiag) {
        const Matrix4& o2w = drawCall.getTransformData().objectToWorld;
        auto colLen = [](const Vector4& c) {
          return std::sqrt(float(c.x) * float(c.x) + float(c.y) * float(c.y) + float(c.z) * float(c.z));
        };
        Logger::info(str::format("[SkyDiag]",
          " vs=0x", std::hex, vsH, std::dec,
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
      m_instanceManager.bindMaterial(instance, surfaceMaterial);
    }

    // Update portal
    if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
      m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
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

    ObjectCacheState result = ObjectCacheState::kInvalid;
    BlasEntry* pBlas = nullptr;
    if (m_drawCallCache.get(drawCallState, &pBlas) == DrawCallCache::CacheState::kExisted) {
      result = onSceneObjectUpdated(ctx, drawCallState, pBlas);
    } else {
      result = onSceneObjectAdded(ctx, drawCallState, pBlas);
    }
    
    assert(pBlas != nullptr);
    assert(result != ObjectCacheState::kInvalid);

    // Update the input state, so we always have a reference to the original draw call state
    pBlas->frameLastTouched = m_device->getCurrentFrameId();

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
      const char* sn = drawCallState.studioModelName;
      const bool isHullName = sn[0] != '\0' &&
        (std::strstr(sn, "widow") != nullptr || std::strstr(sn, "Crow_dropship") != nullptr);
      const bool isInstancedHull = drawCallState.getRigidBakeBoneIndex() != 0;
      if (isHullName || isInstancedHull) {
        const auto& o = drawCallState.getTransformData().objectToWorld;
        const uint32_t rff = m_device->getCurrentFrameId();
        // one instanced + one non-instanced line per frame
        static uint32_t s_rfInst = UINT32_MAX, s_rfNon = UINT32_MAX;
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
    }

    // Note: The material data can be modified in instance manager
    RtInstance* instance = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, existingInstance);

    // Check if a light should be created for this Material
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
        auto [iter, isNew] = m_drawCallMeta.infos[m_drawCallMeta.ticker].emplace(instance->surface.objectPickingValue, meta);
        ONCE_IF_FALSE(isNew, Logger::warn(
          "Found multiple draw calls with the same \'objectPickingValue\'. "
          "Ignoring further MetaInfo-s, some objects might be not be available through object picking"));
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
      particleSystem.spawnParticles(ctx.ptr(), *pParticleSystemDesc, instance->getVectorIdx(), drawCallState, renderMaterialData);

      if (pParticleSystemDesc->hideEmitter) {
        instance->setHidden(true);
      }
    }

    if (instance) {
      ++vanishDiag().drawsKept;
      const XXH64_hash_t vsHash = drawCallState.getTransformData().vertexShaderHash;
      ++vanishDiag().vsHistogram[vsHash];
    }

    return instance;
  }

  const RtSurfaceMaterial& SceneManager::createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                               const DrawCallState& drawCallState,
                                                               uint32_t* out_indexInCache) {
    ScopedCpuProfileZone();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();
    const auto renderMaterialDataType = renderMaterialData.getType();

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
    uint32_t samplerIndex = trackSampler(sampler);

    // NV-DXVK: log final sampler's address modes once per (U,V,W,filter)
    // combo, regardless of which branch above produced it (override / patched
    // / eye-clamped). Tests the wrap-mode hypothesis for BSP world-scale UVs.
    // 0=REPEAT 1=MIRRORED 2=CLAMP_EDGE 3=CLAMP_BORDER 4=MIRROR_CLAMP.
    if (sampler != nullptr) {
      const auto& si = sampler->info();
      const uint32_t key =
        (uint32_t(si.addressModeU) & 0x7u) |
        ((uint32_t(si.addressModeV) & 0x7u) << 3) |
        ((uint32_t(si.addressModeW) & 0x7u) << 6) |
        ((uint32_t(si.magFilter)   & 0x7u) << 9) |
        ((uint32_t(si.mipmapMode)  & 0x7u) << 12);
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

    auto iter = m_preCreationSurfaceMaterialMap.find(preCreationHash);
    if (iter != m_preCreationSurfaceMaterialMap.end()) {
      if (out_indexInCache) {
        *out_indexInCache = iter->second;
      }
      return m_surfaceMaterialCache.at(iter->second);
    }

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
        {
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
    if (out_indexInCache) {
      *out_indexInCache = index;
    }
    return m_surfaceMaterialCache.at(index);
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

  void SceneManager::prepareSceneData(Rc<RtxContext> ctx, DxvkBarrierSet& execBarriers) {
    ScopedGpuProfileZone(ctx, "Build Scene");

  #ifdef REMIX_DEVELOPMENT
    if (m_device->getCurrentFrameId() == RtxOptions::dumpAllInstancesOnFrame()) {
      // Print all RtInstances for debugging
      printAllRtInstances();
    }
  #endif

    // Needs to happen before garbageCollection to avoid destroying dynamic lights
    m_lightManager.dynamicLightMatching();

    garbageCollection();

    m_graphManager.applySceneOverrides(ctx);

    m_terrainBaker->prepareSceneData(ctx);

    auto& textureManager = m_device->getCommon()->getTextureManager();
    m_bindlessResourceManager.prepareSceneData(ctx, textureManager.getTextureTable(), getBufferTable(), getSamplerTable());

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
      const bool emitNow = (fid < 16) || ((fid % 30) == 0) || (total == 0);
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
      if ((sPreMergeFrame++ % 30u) == 0) {
        Logger::info(str::format(
          "[SpawnGeomDiag.preMerge] frame=", m_device->getCurrentFrameId(),
          " call=", sPreMergeFrame,
          " activeInstances=", m_instanceManager.getActiveCount()));
      }
    }
    m_accelManager.mergeInstancesIntoBlas(ctx, execBarriers, textureManager.getTextureTable(), m_cameraManager, m_instanceManager, m_opacityMicromapManager.get());

    // Call on the other managers to prepare their GPU data for the current scene
    m_accelManager.prepareSceneData(ctx, execBarriers, m_instanceManager);
    m_lightManager.prepareSceneData(ctx, m_cameraManager);

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

    // GPU-driven PointInstancer culling: overwrites visible instance placeholders
    // in m_vkInstanceBuffer with proper transforms and masks, copies per-instance
    // surface and material data from templates. Must run after prepareSceneData
    // (which uploads placeholders) and before buildTlas.
    m_accelManager.dispatchPointInstancerCulling(ctx, m_cameraManager, m_surfaceMaterialBuffer);

    // Build the TLAS
    m_accelManager.buildTlas(ctx);

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

    if (freeUnused) {
      // DXVK doesnt free chunks for us by default (its high water mark) so force release some memory back to the system here.
      m_device->getCommon()->memoryManager().freeUnusedChunks();
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
