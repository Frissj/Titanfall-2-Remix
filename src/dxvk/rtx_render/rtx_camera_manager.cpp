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
#include <Windows.h>

#include "dxvk_device.h"
#include "rtx_resources.h"

// NV-DXVK [classify-eye-truth]: read the engine's stable eye position
// (client.dll local-player + 0x3D6C). Used by the [CamMgr.classify]
// diagnostic to compare each Main candidate's recovered camera position
// against engine ground truth, so we can identify whether a draw is the
// real player view vs. some other pass that's getting misclassified as
// Main. Returns nullptr if accessor or local-player aren't yet resolvable.
namespace {
  inline const float* GetEngineEyeCM() {
    using GetLocalPlayerFn = void* (*)();
    static GetLocalPlayerFn s_getLp = nullptr;
    if (!s_getLp) {
      HMODULE clientDll = GetModuleHandleA("client.dll");
      if (clientDll) {
        s_getLp = reinterpret_cast<GetLocalPlayerFn>(
          reinterpret_cast<uint8_t*>(clientDll) + 0x14EAE0);
      }
      if (!s_getLp) return nullptr;
    }
    void* lp = s_getLp();
    if (!lp) return nullptr;
    return reinterpret_cast<const float*>(
      reinterpret_cast<const uint8_t*>(lp) + 0x3D6C);
  }
}

namespace {
  constexpr float kFovToleranceRadians = 0.001f;

  // NV-DXVK TF2: per-frame histogram of Main-candidate reject reasons. Populated
  // by processCameraData and dumped by CameraManager::onFrameEnd so we can see
  // which specific gate blocks Main latches on frames where the camera fails
  // to update. Counters are reset when the observed frameId advances.
  struct MainRejectHistogram {
    uint32_t frameId = UINT32_MAX;
    uint32_t candidates = 0;
    uint32_t accepted = 0;
    // Physical gates.
    uint32_t rejIsInWorld = 0;
    uint32_t rejIsNonSquare = 0;
    uint32_t rejIsReasonableDepth = 0;
    uint32_t rejIsReasonableFov = 0;
    uint32_t rejIsLargeEnough = 0;
    // Hysteresis gates.
    uint32_t rejVpMatches = 0;
    uint32_t rejMaxZMatches = 0;
    uint32_t rejFovClose = 0;
    uint32_t rejBasisClose = 0;
    uint32_t rejStreakNotMet = 0;
  };
  MainRejectHistogram g_mainHist;

  inline void noteFrame(uint32_t frameId) {
    if (g_mainHist.frameId != frameId) {
      g_mainHist = MainRejectHistogram{};
      g_mainHist.frameId = frameId;
    }
  }
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
    // NV-DXVK TF2: dump the Main-candidate reject histogram for the frame
    // that just ended. One line per frame — makes it trivial to spot which
    // gate is blocking Main updates when the camera lags the player.
    if (g_mainHist.frameId != UINT32_MAX && g_mainHist.candidates > 0) {
      Logger::info(str::format(
        "[CamMgr.hist] frame=", g_mainHist.frameId,
        " cand=", g_mainHist.candidates,
        " accept=", g_mainHist.accepted,
        " phys{inWorld=", g_mainHist.rejIsInWorld,
        " nonSq=", g_mainHist.rejIsNonSquare,
        " depth=", g_mainHist.rejIsReasonableDepth,
        " fov=", g_mainHist.rejIsReasonableFov,
        " large=", g_mainHist.rejIsLargeEnough,
        "} hyst{vp=", g_mainHist.rejVpMatches,
        " maxZ=", g_mainHist.rejMaxZMatches,
        " fovClose=", g_mainHist.rejFovClose,
        " basisClose=", g_mainHist.rejBasisClose,
        " streak=", g_mainHist.rejStreakNotMet,
        "}"));
    }
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
      // NV-DXVK [VM.check]: trace every invocation so we can see why the
      // viewmodel is / isn't being classified. Throttled per-frame.
      const float vmThr = RtxOptions::ViewModel::maxZThreshold();
      const bool vmEnable = RtxOptions::ViewModel::enable();
      {
        static uint32_t sLastVMFrame = 0;
        static uint32_t sVMLogCount = 0;
        if (frameId != sLastVMFrame) { sLastVMFrame = frameId; sVMLogCount = 0; }
        if (sVMLogCount < 32) {
          ++sVMLogCount;
          Logger::info(str::format(
            "[VM.check] f=", frameId,
            " maxZ=", maxZ,
            " fov=", fov,
            " thr=", vmThr,
            " enable=", (vmEnable ? 1 : 0),
            " maxZHit=", (vmEnable && maxZ <= vmThr ? 1 : 0)));
        }
      }
      if (vmEnable) {
        // Note: max Z check is the top-priority
        if (maxZ <= vmThr) {
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
    // NV-DXVK [VM.class]: log every camera-type decision so we can see the
    // post-isViewModel result. Throttled per frame.
    {
      static uint32_t sLastVMClassFrame = 0;
      static uint32_t sVMClassLog = 0;
      if (frameId != sLastVMClassFrame) { sLastVMClassFrame = frameId; sVMClassLog = 0; }
      if (sVMClassLog < 32) {
        ++sVMClassLog;
        Logger::info(str::format(
          "[VM.class] f=", frameId,
          " maxZ=", input.maxZ,
          " fov=", decomposeProjectionParams.fov,
          " type=", static_cast<uint32_t>(cameraType),
          " isRT=", (input.isDrawingToRaytracedRenderTarget ? 1 : 0),
          " isSky=", (input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0)));
      }
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
      // NV-DXVK TF2: also reject any candidate whose world translation is
      // near zero, regardless of rotation. ExtractTransforms path 1 always
      // bakes the real camera world position into w2v[3] (= -dot(axis,camPos)),
      // so a path-1 output with |w2v[3]| ≈ 0 means the per-draw cb2 RDEF read
      // returned a stale/HUD/identity camera (e.g. (0.0004,0,0)). Without this
      // gate, such a candidate could win Main's first latch and freeze the
      // camera at origin while gameplay draws update body geometry → visible
      // body-races-ahead-of-camera lag. Threshold 10 units handles spawn
      // points near origin while reliably catching the (~0,~0,~0) garbage.
      const bool transTooSmall = w2vMagSq < (10.0f * 10.0f);
      const bool isInWorld = !(transNearZero && rotIsIdentity) && !transTooSmall;
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
      // NV-DXVK (fix 2): minimum viewport size gate. isNonSquare above already
      // rejects the 1024×1024 / 128×128 / 16×16 / 1×1 shadow & probe viewports
      // whose aspect is exactly 1, but TF2 also issues 640×360 / 1280×720 /
      // 80×360 / 160×360 viewports with ~16:9 aspect — HUD compositing,
      // thumbnails, minimaps — that were previously latching as Main and
      // causing the flick. Require a minimum pixel count; the true backbuffer
      // is always at least 720p.
      const bool isLargeEnough = vw >= 1200.0f && vh >= 600.0f;
      const bool keepAsMain =
        isInWorld && isNonSquare && isReasonableDepth && isReasonableFov && isLargeEnough;
      noteFrame(frameId);
      ++g_mainHist.candidates;
      if (!isInWorld) ++g_mainHist.rejIsInWorld;
      if (!isNonSquare) ++g_mainHist.rejIsNonSquare;
      if (!isReasonableDepth) ++g_mainHist.rejIsReasonableDepth;
      if (!isReasonableFov) ++g_mainHist.rejIsReasonableFov;
      if (!isLargeEnough) ++g_mainHist.rejIsLargeEnough;
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
            " isReasonableFov=", isReasonableFov ? 1 : 0,
            " isLargeEnough=", isLargeEnough ? 1 : 0));
        }
        cameraType = CameraType::Unknown;
      } else {
        // NV-DXVK (fixes 1 + 3): latch hysteresis. Once Main is latched by the
        // classifier, subsequent candidates that pass the physical gates must
        // also look CONSISTENT with the existing latch — same FoV (±3°), same
        // viewport (±4 px), same forward direction (dot > 0.5, i.e. within
        // ~60° — accommodates normal mouse look but rejects the 90° axis
        // twists seen in the log between wtvPathId=1 and wtvPathId=3). On
        // kCutStreakThreshold consecutive disagreements we assume a real cut
        // and allow the re-latch. Without this, every draw that passes the
        // gates overwrites Main — and multiple draws per frame pass, so Main
        // flickers between shadow/reflection/gameplay poses.
        const uint32_t curFrameId = m_device->getCurrentFrameId();
        const auto& snap = m_mainLatchSnapshot;
        // Snapshot counts as fresh if it was set within the last couple of
        // frames. Older than that, we assume the view was paused/stale and
        // allow a fresh latch unconditionally.
        const bool snapFresh =
          snap.valid
          && (curFrameId <= snap.frameId || (curFrameId - snap.frameId) <= 3);
        if (snapFresh) {
          const float fovDiff = std::abs(decomposeProjectionParams.fov - snap.fovRad);
          const bool fovClose = fovDiff < 0.052f; // ~3 degrees
          const bool vpMatches =
            std::abs(vw - snap.viewportW) < 4.0f && std::abs(vh - snap.viewportH) < 4.0f;
          // Forward from this draw's worldToView row-major convention: col 2.
          const Vector3 newFwd(w2v[0][2], w2v[1][2], w2v[2][2]);
          const float newFwdLen2 =
            newFwd.x*newFwd.x + newFwd.y*newFwd.y + newFwd.z*newFwd.z;
          const float dot =
            (newFwdLen2 > 0.001f)
              ? (newFwd.x*snap.fwd.x + newFwd.y*snap.fwd.y + newFwd.z*snap.fwd.z)
              : 0.0f;
          // NV-DXVK TF2: forward-vector check — same-hemisphere (~90° cap).
          // Allows normal fast mouse look while rejecting 180° axis flips.
          const bool fwdClose = dot > 0.0f;
          // NV-DXVK TF2: also compare the right vector so roll around the
          // forward axis is part of the basis check. Two candidate draws can
          // share a forward direction yet differ by a 90° roll (TF2 produces
          // both), and without this check either pose can win the Main latch
          // — the camera then renders sideways. 0.7 ≈ cos(45°): tolerates
          // moderate roll drift (camera tilt anims, lean), rejects 90°+ flips.
          const Vector3 newRight(w2v[0][0], w2v[1][0], w2v[2][0]);
          const float newRightLen2 =
            newRight.x*newRight.x + newRight.y*newRight.y + newRight.z*newRight.z;
          const float rightDot =
            (newRightLen2 > 0.001f)
              ? (newRight.x*snap.right.x + newRight.y*snap.right.y + newRight.z*snap.right.z)
              : 0.0f;
          const bool rightClose = rightDot > 0.7f;
          const bool basisClose = fwdClose && rightClose;
          // NV-DXVK: differentiate hard and soft rejects. A viewport mismatch
          // is a DIFFERENT RENDER PASS (HUD compositing, scope zoom, minimap
          // preview, thumbnail) — not a camera cut. Accepting it as a "cut"
          // after N tries just means the wrong render pass steals Main. So
          // viewport-wrong candidates are HARD-rejected and never contribute
          // to the cut streak. Only FoV-change + basis-change count, because
          // those are real camera cuts (level change, teleport, cinematic).
          // NV-DXVK TF2 FIX: also HARD-reject maxZ mismatches. Each TF2
          // render pass uses a distinct viewport depth range — main world
          // is 0.1, viewmodel 0.05, shadow 1.0, probe/env 1.0 at different
          // resolutions. A candidate with different maxZ is a DIFFERENT
          // render pass, not a camera cut. Without this gate, every frame
          // we alternate between passes and Main oscillates → visible flash
          // on frame 1 + subsequent-frame geometry pops.
          const bool maxZMatches = std::abs(input.maxZ - snap.maxZ) < 0.01f;
          // NV-DXVK TF2: intra-frame position-magnitude check. With multi-latch
          // enabled (last-wins per frame), a later draw whose worldToView
          // carries a wildly different translation magnitude from the snapshot
          // is almost certainly a different render pass that happens to share
          // viewport/maxZ/basis (e.g. a camera-relative HUD overlay drawn at a
          // small offset from origin). Rejecting these intra-frame protects
          // Main from being yanked to a wrong pose by the last passing draw.
          // Only enforced when the snapshot was set THIS frame (intra-frame
          // refinement); cross-frame motion goes through the wider basis/fov
          // path. Tolerance is loose (200 world units) — the player can move
          // ~150 u/s, and we just need to filter the obvious order-of-magnitude
          // mismatches like (5188 vs 50) without rejecting same-scene refinements.
          const float newW2vTMag = std::sqrt(w2vMagSq);
          const float snapPosMag = std::sqrt(
            snap.pos.x*snap.pos.x + snap.pos.y*snap.pos.y + snap.pos.z*snap.pos.z);
          const bool intraFrame = (snap.frameId == curFrameId);
          const bool posMagOk = !intraFrame
            || std::abs(newW2vTMag - snapPosMag) < 500.0f;
          if (!vpMatches) {
            ++g_mainHist.rejVpMatches;
            static uint32_t sHystLog = 0;
            if (sHystLog < 40) {
              ++sHystLog;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (wrong viewport)",
                " vp=(", int(vw), "x", int(vh), ")",
                " snapVp=(", int(snap.viewportW), "x", int(snap.viewportH), ")"));
            }
            cameraType = CameraType::Unknown;
          } else if (!maxZMatches) {
            ++g_mainHist.rejMaxZMatches;
            static uint32_t sHystLog2 = 0;
            if (sHystLog2 < 40) {
              ++sHystLog2;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (wrong maxZ)",
                " maxZ=", input.maxZ,
                " snapMaxZ=", snap.maxZ));
            }
            cameraType = CameraType::Unknown;
          } else if (!posMagOk) {
            // Same frame, vp/maxZ match, but eye-space translation magnitude
            // is far from the in-frame latched pose. Different render pass.
            static uint32_t sHystLog3 = 0;
            if (sHystLog3 < 40) {
              ++sHystLog3;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (intra-frame |w2vT| mismatch)",
                " newMag=", newW2vTMag,
                " snapMag=", snapPosMag));
            }
            cameraType = CameraType::Unknown;
          } else if (!fovClose || !basisClose) {
            if (!fovClose) ++g_mainHist.rejFovClose;
            if (!basisClose) ++g_mainHist.rejBasisClose;
            ++m_disagreeStreak;
            // NV-DXVK TF2: reduced from 8 → 3. The streak exists to suppress
            // intra-frame flicker between competing candidate draws, not to
            // suppress legitimate frame-to-frame motion. Eight frames meant
            // the camera could be 8 frames behind reality before accepting a
            // re-latch, which was a major contributor to the main-camera lag
            // observed while walking. Three is enough to filter same-frame
            // multi-candidate noise while tracking real motion promptly.
            constexpr uint32_t kCutStreakThreshold = 3;
            if (m_disagreeStreak < kCutStreakThreshold) {
              ++g_mainHist.rejStreakNotMet;
              static uint32_t sHystLog = 0;
              if (sHystLog < 40) {
                ++sHystLog;
                Logger::info(str::format(
                  "[CamMgr.hyst] reject streak=", m_disagreeStreak,
                  " fovClose=", fovClose ? 1 : 0,
                  " fwdClose=", fwdClose ? 1 : 0,
                  " rightClose=", rightClose ? 1 : 0,
                  " fwdDot=", dot,
                  " rightDot=", rightDot,
                  " fovDelta=", fovDiff * (180.0f / 3.14159265f), "deg"));
              }
              cameraType = CameraType::Unknown;
            } else {
              // Consistent disagreement for many frames — accept as cut.
              m_disagreeStreak = 0;
              Logger::info("[CamMgr.hyst] accepting re-latch (cut)");
            }
          } else {
            m_disagreeStreak = 0;
          }
        }
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
    // NV-DXVK TF2: previously this gated on `lastUpdateFrame != frameId`,
    // making the FIRST passing draw per frame win Main and locking out
    // every subsequent draw. That's the root cause of the "body races
    // ahead of camera" lag: when the first-latched draw read a stale cb2
    // (the game can submit multiple cbuffers per frame, and DX11's per-draw
    // RDEF lookup can land on one whose CBufCommonPerCamera value is older
    // than the gameplay one), Main froze on it for the whole frame even
    // though later gameplay draws had fresher data. Now Main re-latches on
    // every passing candidate within the frame; LAST-WINS semantics. The
    // strengthened isInWorld gate (|w2vT|>10) plus existing hysteresis
    // (vp/maxZ HARD reject + fwd/right basis check + position-proximity
    // below) ensure only legitimate gameplay candidates can re-latch, so
    // the last one carries the freshest pose.
    bool shouldUpdateMainCamera = cameraType == CameraType::Main;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    Matrix4 worldToView = input.getTransformData().worldToView;
    Matrix4 viewToProjection = input.getTransformData().viewToProjection;

    // NV-DXVK [classify-trace]: per-call log of camera classification +
    // worldToView translation. Used to verify whether multiple distinct
    // worldToView values get classified as Main within the same frame
    // (proves the "alternating cameras" oscillation theory). Throttled
    // to first 400 events to capture ~5-10 seconds of gameplay.
    {
      static uint32_t sClassifyLog = 0;
      static uint32_t sLastFrameId = UINT32_MAX;
      static uint32_t sMainsThisFrame = 0;
      static Vector3 sFirstMainW2vTThisFrame{0,0,0};
      static Vector3 sLastMainW2vTThisFrame{0,0,0};
      const uint32_t frameId = m_device->getCurrentFrameId();
      // Detect frame boundary; emit a per-frame summary of the spread
      // between FIRST and LAST Main candidate's w2vT.
      if (frameId != sLastFrameId && sMainsThisFrame > 1 && sClassifyLog < 400) {
        const Vector3 spread = sLastMainW2vTThisFrame - sFirstMainW2vTThisFrame;
        const float spreadMag = std::sqrt(
          spread.x*spread.x + spread.y*spread.y + spread.z*spread.z);
        Logger::info(str::format(
          "[CamMgr.classify-spread] frame=", sLastFrameId,
          " mainCandidates=", sMainsThisFrame,
          " firstMainW2vT=(", sFirstMainW2vTThisFrame.x, ",",
                              sFirstMainW2vTThisFrame.y, ",",
                              sFirstMainW2vTThisFrame.z, ")",
          " lastMainW2vT=(", sLastMainW2vTThisFrame.x, ",",
                             sLastMainW2vTThisFrame.y, ",",
                             sLastMainW2vTThisFrame.z, ")",
          " |spread|=", spreadMag));
        ++sClassifyLog;
      }
      if (frameId != sLastFrameId) {
        sLastFrameId = frameId;
        sMainsThisFrame = 0;
      }
      const Vector3 curW2vT(worldToView[3][0], worldToView[3][1], worldToView[3][2]);
      if (cameraType == CameraType::Main) {
        if (sMainsThisFrame == 0) sFirstMainW2vTThisFrame = curW2vT;
        sLastMainW2vTThisFrame = curW2vT;
        ++sMainsThisFrame;
      }
      // Per-call classify log: type, frame, w2vT + recovered camera world
      // position from full matrix decomposition + engine ground-truth eye.
      // The recovered C tells us what player position THIS matrix encodes;
      // delta vs engineEye tells us whether this candidate is the real
      // player camera (delta < 1u) or some other pass misclassified as Main.
      if (sClassifyLog < 400) {
        const float vw = input.getTransformData().viewportWidth;
        const float vh = input.getTransformData().viewportHeight;
        const Matrix4& p = viewToProjection;
        const float Sy = p[1][1];
        const float fovDeg = (std::abs(Sy) > 1e-6f)
          ? (2.f * std::atan(1.f / Sy) * (180.f / 3.14159265f)) : 0.f;
        const char* typeName =
          cameraType == CameraType::Main ? "Main" :
          cameraType == CameraType::ViewModel ? "ViewModel" :
          cameraType == CameraType::Sky ? "Sky" :
          cameraType == CameraType::Portal0 ? "Portal0" :
          cameraType == CameraType::Portal1 ? "Portal1" :
          cameraType == CameraType::Unknown ? "Unknown" : "?";
        // Recover camera world position C from worldToView (orthonormal
        // assumption). dxvk Matrix4 stores M[col][row]; math row 0 = R,
        // row 1 = U, row 2 = F, so R.x=W[0][0], R.y=W[1][0], R.z=W[2][0];
        // U.x=W[0][1], etc. C = -R_rot^T · t where t = (W[3][0..2]).
        const Matrix4& w = worldToView;
        const float Cx = -(w[0][0]*w[3][0] + w[0][1]*w[3][1] + w[0][2]*w[3][2]);
        const float Cy = -(w[1][0]*w[3][0] + w[1][1]*w[3][1] + w[1][2]*w[3][2]);
        const float Cz = -(w[2][0]*w[3][0] + w[2][1]*w[3][1] + w[2][2]*w[3][2]);
        // Engine ground truth (lp+0x3D6C). May be null early in startup.
        const float* eye = GetEngineEyeCM();
        const bool haveEye =
          eye && std::isfinite(eye[0]) && std::isfinite(eye[1]) && std::isfinite(eye[2]);
        const float eX = haveEye ? eye[0] : 0.f;
        const float eY = haveEye ? eye[1] : 0.f;
        const float eZ = haveEye ? eye[2] : 0.f;
        const float dCx = haveEye ? (Cx - eX) : 0.f;
        const float dCy = haveEye ? (Cy - eY) : 0.f;
        const float dCz = haveEye ? (Cz - eZ) : 0.f;
        Logger::info(str::format(
          "[CamMgr.classify] frame=", frameId,
          " type=", typeName,
          " w2vT=(", curW2vT.x, ",", curW2vT.y, ",", curW2vT.z, ")",
          " recovC=(", Cx, ",", Cy, ",", Cz, ")",
          " engineEye=", haveEye ? "valid" : "null",
          " eye=(", eX, ",", eY, ",", eZ, ")",
          " delta=(", dCx, ",", dCy, ",", dCz, ")",
          " R=(", w[0][0], ",", w[1][0], ",", w[2][0], ")",
          " U=(", w[0][1], ",", w[1][1], ",", w[2][1], ")",
          " F=(", w[0][2], ",", w[1][2], ",", w[2][2], ")",
          " vp=", int(vw), "x", int(vh),
          " fov=", fovDeg, "deg"));
        ++sClassifyLog;
      }
    }

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
      noteFrame(frameId);
      ++g_mainHist.accepted;
      // NV-DXVK: record the snapshot used by the hysteresis gate on future
      // candidates. Must happen AFTER camera.update so getPosition/getForward
      // reflect the new latch.
      {
        MainLatchSnapshot& snap = m_mainLatchSnapshot;
        snap.fovRad      = decomposeProjectionParams.fov;
        snap.viewportW   = input.getTransformData().viewportWidth;
        snap.viewportH   = input.getTransformData().viewportHeight;
        snap.maxZ        = input.maxZ;
        // Forward from row-major worldToView col 2, right from col 0.
        const Matrix4& w = worldToView;
        snap.fwd         = Vector3(w[0][2], w[1][2], w[2][2]);
        snap.right       = Vector3(w[0][0], w[1][0], w[2][0]);
        // Normalize (input may not be perfectly unit).
        const float fwdLen2 = snap.fwd.x*snap.fwd.x + snap.fwd.y*snap.fwd.y + snap.fwd.z*snap.fwd.z;
        if (fwdLen2 > 0.001f) {
          const float invLen = 1.0f / std::sqrt(fwdLen2);
          snap.fwd = Vector3(snap.fwd.x * invLen, snap.fwd.y * invLen, snap.fwd.z * invLen);
        }
        const float rightLen2 = snap.right.x*snap.right.x + snap.right.y*snap.right.y + snap.right.z*snap.right.z;
        if (rightLen2 > 0.001f) {
          const float invLen = 1.0f / std::sqrt(rightLen2);
          snap.right = Vector3(snap.right.x * invLen, snap.right.y * invLen, snap.right.z * invLen);
        }
        snap.pos         = camera.getPosition(/*freecam=*/false);
        snap.frameId     = frameId;
        snap.valid       = true;
      }
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
