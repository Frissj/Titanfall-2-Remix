/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_camera_manager.h"

#include <cstdio>
#include <unordered_set>

#include "dxvk_device.h"
#include "rtx_resources.h"

namespace {
  constexpr float kFovToleranceRadians = 0.001f;
}

namespace dxvk {

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    return accessCamera(*this, cameraType).isValid(m_device->getCurrentFrameId());
  }

  void CameraManager::onFrameEnd() {
    m_lastSetCameraType = CameraType::Unknown;
    m_decompositionCache.clear();
  }

  CameraType::Enum CameraManager::processCameraData(const DrawCallState& input) {
    // If theres no real camera data here - bail
    if (isIdentityExact(input.getTransformData().viewToProjection)) {
      return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }

    switch (RtxOptions::fusedWorldViewMode()) {
    case FusedWorldViewMode::None:
      if (input.getTransformData().objectToView == input.getTransformData().objectToWorld && !isIdentityExact(input.getTransformData().objectToView)) {
        return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
      }
      break;
    case FusedWorldViewMode::View:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if World is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToWorld),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to View but World transform is not identity!"));
      }
      break;
    case FusedWorldViewMode::World:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if View is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToView),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to World but View transform is not identity!"));
      }
      break;
    }

    // Get camera params
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(input.getTransformData().viewToProjection);

    // Filter invalid cameras, extreme shearing
    static auto isFovValid = [](float fovA) {
      return fovA >= kFovToleranceRadians;
    };
    static auto areFovsClose = [](float fovA, const RtCamera& cameraB) {
      return std::abs(fovA - cameraB.getFov()) < kFovToleranceRadians;
    };

    if (std::abs(decomposeProjectionParams.shearX) > 0.01f || !isFovValid(decomposeProjectionParams.fov)) {
      ONCE(Logger::warn("[RTX] CameraManager: rejected an invalid camera"));
      return input.getCategoryFlags().test(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }


    auto isViewModel = [this](float fov, float maxZ, uint32_t frameId) {
      if (RtxOptions::ViewModel::enable()) {
        // Note: max Z check is the top-priority
        if (maxZ <= RtxOptions::ViewModel::maxZThreshold()) {
          return true;
        }
        // NV-DXVK: only trust Main's FoV for the "different-FoV → ViewModel"
        // inference if the CLASSIFIER latched Main (not the safety net). The
        // safety net populates Main from whatever ExtractTransforms produced
        // at frame end — often a UI/fallback matrix with a wrong FoV. If we
        // compared gameplay draws against that, they'd all be marked ViewModel
        // and never reach the classifier's gameplay-VS allowlist, so Main
        // would never get classifier-latched and the loop self-sustains.
        // Classifier must have latched Main within the last few frames for
        // Main's FoV to be authoritative here. If the last classifier latch is
        // older than that, Main is effectively stale (or was overwritten by
        // the safety net) and shouldn't drive ViewModel inference.
        const uint32_t lastClassifierLatchFrame = getMainClassifierFrameId();
        const bool mainClassifierRecent =
          isMainSetByClassifier()
          && (frameId <= lastClassifierLatchFrame
              || (frameId - lastClassifierLatchFrame) <= 2);
        if (mainClassifierRecent && getCamera(CameraType::Main).isValid(frameId)) {
          // FOV is different from Main camera => assume that it's a ViewModel one
          if (!areFovsClose(fov, getCamera(CameraType::Main))) {
            return true;
          }
        }
      }
      return false;
    };

    const uint32_t frameId = m_device->getCurrentFrameId();

    auto cameraType = CameraType::Main;
    if (input.isDrawingToRaytracedRenderTarget) {
      cameraType = CameraType::RenderToTexture;
    } else if (input.testCategoryFlags(InstanceCategories::Sky)) {
      cameraType = CameraType::Sky;
    } else if (isViewModel(decomposeProjectionParams.fov, input.maxZ, frameId)) {
      cameraType = CameraType::ViewModel;
    }

    // NV-DXVK: Deterministic Main-camera classifier — game-native per-draw
    // identity, no matrix-property heuristics.
    //
    // Empirically (probe I), TF2's BSP gameplay-world pass is drawn by a small
    // stable set of vertex shaders with a compressed depth range (maxZ ~ 0.05),
    // while fullscreen / HUD / post passes that happen to share a similar
    // projection shape bind maxZ=1.0 and w2vT≈identity. Shader hash + maxZ
    // band uniquely identifies the real-pose world draws. Everything else
    // falls back to Unknown and never wins the Main latch.
    //
    // Hashes are DxvkShader::getHash() values observed in remix-dxvk.log at
    // the 59.84° gameplay FoV with real player-pose worldToView translation.
    // If TF2 ships a shader update, these will need re-identification via
    // probe I (look for the cluster whose w2vT matches player world coords).
    if (cameraType == CameraType::Main) {
      // NV-DXVK: Main-camera classifier — physical-property gates only,
      // no hash allowlist. The hash allowlist was too narrow: it caught
      // TF2's gameplay-world pass (3 specific BSP shaders, maxZ=0.05) but
      // missed the cinematic camera which uses different shaders and the
      // standard depth range (maxZ=1.0). Both share the player's actual
      // world coordinate frame (w2vT magnitude ~10⁴), so the right criterion
      // is "is this draw in world space?" not "is this a specific shader?".
      //
      // Three checks (any failure → Unknown, no Main update):
      // 1. |w2vT| > 100: rejects fullscreen/UI/composite passes that share a
      //    gameplay-shaped projection but render at origin (|w2vT| < 10).
      // 2. viewport aspect != 1 (non-square): rejects shadow cascades and
      //    cubemap face renders.
      // 3. maxZ in (0, 1.5]: rejects degenerate viewport configs.
      const auto& td = input.getTransformData();
      const Matrix4& w2v = td.worldToView;
      const float w2vMagSq =
        w2v[3][0]*w2v[3][0] + w2v[3][1]*w2v[3][1] + w2v[3][2]*w2v[3][2];
      // "Real camera" check. TF2 has TWO conventions:
      //   • Cinematic / external view: worldToView translation = world-space
      //     player position (magnitude ~10⁴). Rotation = camera orientation.
      //   • Actual gameplay: camera-local vertex space. worldToView translation
      //     ≈ 0, but rotation is still the camera's view rotation (player
      //     looking around).
      // Both are real cameras. The case we want to REJECT is fullscreen/UI/
      // composite passes where worldToView is the FULL identity matrix
      // (rotation = I AND translation = 0). Detect that specifically.
      const bool transNearZero = w2vMagSq < (1.0f * 1.0f);
      const bool rotIsIdentity =
        std::abs(w2v[0][0] - 1.0f) < 0.01f && std::abs(w2v[0][1]) < 0.01f && std::abs(w2v[0][2]) < 0.01f &&
        std::abs(w2v[1][0]) < 0.01f && std::abs(w2v[1][1] - 1.0f) < 0.01f && std::abs(w2v[1][2]) < 0.01f &&
        std::abs(w2v[2][0]) < 0.01f && std::abs(w2v[2][1]) < 0.01f && std::abs(w2v[2][2] - 1.0f) < 0.01f;
      // isInWorld is true unless this is the pure-identity (UI/composite) case.
      const bool isInWorld = !(transNearZero && rotIsIdentity);
      const float vw = td.viewportWidth;
      const float vh = td.viewportHeight;
      const float vpAspect = (vh > 0.0f) ? (vw / vh) : 0.0f;
      const bool isNonSquare = (vw > 0.0f && vh > 0.0f) &&
                               std::abs(vpAspect - 1.0f) >= 0.02f;
      const bool isReasonableDepth = input.maxZ > 0.0f && input.maxZ <= 1.5f;
      // FoV sanity. Standard game cameras (gameplay, cinematic, mech cockpit)
      // are 30°–120°. TF2 also issues:
      //   • ~140°/160°/147°: cubemap / reflection / fog volume cameras (wide).
      //   • ~179.9°: degenerate fog/volume math (essentially flat projection).
      // Latching Main on any of these produces the rainbow-scanline garbage
      // because volume rendering downstream expects a sane frustum.
      const float fovDeg = decomposeProjectionParams.fov * (180.0f / 3.14159265f);
      const bool isReasonableFov = fovDeg > 30.0f && fovDeg < 120.0f;
      const bool keepAsMain =
        isInWorld && isNonSquare && isReasonableDepth && isReasonableFov;
      if (!keepAsMain) {
        static uint32_t sVpLog = 0;
        if (sVpLog < 40) {
          ++sVpLog;
          char vsHex[32];
          std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                        static_cast<unsigned long long>(td.vertexShaderHash));
          Logger::info(str::format(
            "[CamMgr] Demoted-from-Main #", sVpLog,
            " vsHash=", vsHex,
            " viewport=", int(vw), "x", int(vh),
            " maxZ=", input.maxZ,
            " fov=", fovDeg, "deg",
            " |w2vT|=", std::sqrt(w2vMagSq),
            " isInWorld=", isInWorld ? 1 : 0,
            " isNonSquare=", isNonSquare ? 1 : 0,
            " isReasonableDepth=", isReasonableDepth ? 1 : 0,
            " isReasonableFov=", isReasonableFov ? 1 : 0));
        }
        cameraType = CameraType::Unknown;
      }
    }
    
    // Check fov consistency across frames
    if (frameId > 0) {
      if (getCamera(cameraType).isValid(frameId - 1) && !areFovsClose(decomposeProjectionParams.fov, getCamera(cameraType))) {
        ONCE(Logger::info("[RTX] CameraManager: FOV of a camera changed between frames"));
      }
    }

    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    bool shouldUpdateMainCamera = cameraType == CameraType::Main && camera.getLastUpdateFrame() != frameId;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    Matrix4 worldToView = input.getTransformData().worldToView;
    Matrix4 viewToProjection = input.getTransformData().viewToProjection;

    // NV-DXVK (probe I): comprehensive per-draw camera classification log.
    // One line per UNIQUE (cameraType, viewport, Sx, Sy, vsHash, w2vT-int)
    // tuple, capped at ~120 total. The vsHash + w2vT-int additions disambiguate
    // draws that share a projection shape (e.g. gameplay-world vs. fullscreen
    // post-pass using the same 55.41° matrix) so we can identify the actual
    // gameplay VS hash for an allowlist.
    {
      struct Key {
        uint32_t type; int vw; int vh; int sxBucket; int syBucket;
        uint64_t vsHash; int tX; int tY; int tZ;
      };
      static std::vector<Key> seen;
      static uint32_t sLogCount = 0;
      const Matrix4& p = viewToProjection;
      const float Sx = p[0][0];
      const float Sy = p[1][1];
      const float vw = input.getTransformData().viewportWidth;
      const float vh = input.getTransformData().viewportHeight;
      const uint64_t vsHash =
        static_cast<uint64_t>(input.getTransformData().vertexShaderHash);
      Key k{ static_cast<uint32_t>(cameraType), int(vw), int(vh),
             int(Sx * 100.0f), int(Sy * 100.0f),
             vsHash,
             int(worldToView[3][0]), int(worldToView[3][1]), int(worldToView[3][2]) };
      bool isNew = true;
      for (const auto& s : seen) {
        if (s.type == k.type && s.vw == k.vw && s.vh == k.vh &&
            s.sxBucket == k.sxBucket && s.syBucket == k.syBucket &&
            s.vsHash == k.vsHash &&
            s.tX == k.tX && s.tY == k.tY && s.tZ == k.tZ) {
          isNew = false; break;
        }
      }
      if (isNew && sLogCount < 120) {
        seen.push_back(k);
        ++sLogCount;
        const float fovDeg = decomposeProjectionParams.fov * (180.0f / 3.14159265f);
        const float aspect = std::abs(decomposeProjectionParams.aspectRatio);
        const bool isIdentityProj =
          std::abs(p[0][0]-1.0f) < 0.01f && std::abs(p[1][1]-1.0f) < 0.01f &&
          std::abs(p[2][3]) < 0.01f && std::abs(p[3][3]-1.0f) < 0.01f;
        // Print VS hash in hex so it's trivial to paste into an allowlist.
        char vsHex[32];
        std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                      static_cast<unsigned long long>(vsHash));
        Logger::info(str::format(
          "[CamMgr.probeI] unique #", sLogCount,
          " cameraType=", static_cast<uint32_t>(k.type),
          " viewport=", k.vw, "x", k.vh,
          " Sx=", Sx, " Sy=", Sy, " aspect=", aspect,
          " fov=", fovDeg, "deg",
          " maxZ=", input.maxZ,
          " vsHash=", vsHex,
          " w2vT=(", worldToView[3][0], ",", worldToView[3][1], ",", worldToView[3][2], ")",
          " m23=", p[2][3], " m33=", p[3][3],
          " identityProj=", isIdentityProj ? 1 : 0,
          " shouldUpdateMain=", shouldUpdateMainCamera ? 1 : 0));
      }
    }

    // NV-DXVK: worldToView is LEFT AT GAME VALUES. Previously we zeroed the
    // translation to match the camera-relative TLAS frame, but that starved
    // NRC / motion vectors / denoisers of real world-space camera motion and
    // caused TDRs. The preferred fix is the other direction: shift the TLAS
    // into absolute world by adding c_cameraOrigin to every BSP per-instance
    // translation in d3d11_rtx's fanout, so camera, TLAS, NRC, and motion
    // all live in the same absolute-world coordinate system.

    if (isPlaying || isBrowsing) {
      if (shouldUpdateMainCamera) {
        RtCamera::RtCameraSetting setting;
        cameraSequence->getRecord(cameraSequence->currentFrame(), setting);
        isCameraCut = camera.updateFromSetting(frameId, setting, 0);

        if (isPlaying) {
          cameraSequence->goToNextFrame();
        }
      }
    } else if (cameraType != CameraType::Unknown) {
      // NV-DXVK: critical guard. accessCamera() ALIASES Unknown to the Main
      // camera object (it's documented at the top of CameraManager that we
      // "never update Unknown camera directly"). Without this guard, every
      // Unknown-classified draw would call .update() on the Main camera,
      // stamping its lastUpdateFrame with the current frameId. The next
      // gameplay draw that legitimately classifies as Main then sees
      // shouldUpdateMainCamera = false (because lastUpdateFrame == frameId
      // already) and never gets to latch its real player pose. Net effect:
      // Main is permanently pinned to whatever the first Unknown draw of
      // each frame happened to carry — usually a UI/fallback transform.
      // Skipping the update for Unknown is the only correct option since
      // we can't write to a "discarded" camera slot.
      isCameraCut = camera.update(
        frameId,
        worldToView,
        viewToProjection,
        decomposeProjectionParams.fov,
        decomposeProjectionParams.aspectRatio,
        decomposeProjectionParams.nearPlane,
        decomposeProjectionParams.farPlane,
        decomposeProjectionParams.isLHS
      );
    }


    if (shouldUpdateMainCamera && RtCameraSequence::mode() == RtCameraSequence::Mode::Record) {
      auto& setting = camera.getSetting();
      cameraSequence->addRecord(setting);
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;

    // NV-DXVK: log Main camera latch events with position so the TLAS-coherence
    // filter in d3d11_rtx can be correlated to camera updates frame-by-frame.
    // Also mark that this frame's Main was set by the CLASSIFIER (trusted
    // pose), not the safety net (untrusted pose). The TLAS filter gates on
    // this flag so it only rejects draws when Main's position is reliable.
    if (shouldUpdateMainCamera && cameraType == CameraType::Main) {
      noteMainSetByClassifier(frameId);
      static uint32_t sMainLatchLog = 0;
      if (sMainLatchLog < 40) {
        ++sMainLatchLog;
        const Vector3 pos = camera.getPosition(/*freecam=*/false);
        // Print the basis rows of worldToView so we can verify orientation.
        // Expected Vulkan view convention:
        //   row0 (right)  ≈ (1,0,0) when camera faces world -Z
        //   row1 (up)     ≈ (0,1,0)
        //   row2 (back)   ≈ (0,0,1) (camera looks down -Z so back = +Z view)
        // 45° roll = row0/row1 rotated around row2 axis.
        const Matrix4& w = worldToView;
        Logger::info(str::format(
          "[CamMgr.latch] #", sMainLatchLog, " frame=", frameId,
          " pos=(", pos.x, ",", pos.y, ",", pos.z, ")",
          " fov=", decomposeProjectionParams.fov * (180.0f / 3.14159265f), "deg",
          " maxZ=", input.maxZ,
          " cameraCut=", isCameraCut ? 1 : 0,
          " right=(", w[0][0], ",", w[0][1], ",", w[0][2], ")",
          " up=(",    w[1][0], ",", w[1][1], ",", w[1][2], ")",
          " fwd=(",   w[2][0], ",", w[2][1], ",", w[2][2], ")",
          " VP_m23=", viewToProjection[2][3],   // -1 = RH proj, +1 = LH proj
          " VP_diag=(", viewToProjection[0][0], ",", viewToProjection[1][1], ",", viewToProjection[2][2], ")",
          " VP_translateZ=", viewToProjection[3][2],
          " wtvPathId=", input.getTransformData().worldToViewPathId));
      }
    }

    return cameraType;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      viewToProjection,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS);
  }

    DecomposeProjectionParams CameraManager::getOrDecomposeProjection(const Matrix4& viewToProjection) {
      XXH64_hash_t projectionHash = XXH64(&viewToProjection, sizeof(viewToProjection), 0);
      auto iter = m_decompositionCache.find(projectionHash);
      if (iter != m_decompositionCache.end()) {
        return iter->second;
      }

      DecomposeProjectionParams decomposeProjectionParams;
      decomposeProjection(viewToProjection, decomposeProjectionParams);
      m_decompositionCache.emplace(projectionHash, decomposeProjectionParams);
      return decomposeProjectionParams;
    }
}  // namespace dxvk
