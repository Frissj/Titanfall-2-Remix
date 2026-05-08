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

#include <atomic>
#include <cstdio>
#include <unordered_set>
#include <Windows.h>

// NV-DXVK [pilot-eye-capture]: file-scope mirror of the viewmodel-pass cb2
// c_cameraOrigin (Source's authoritative eye on this TF2 build).
//
// Producer: src/d3d11/d3d11_rtx.cpp at the cb2 RDEF fanout site, when the
//   draw is the viewmodel pass (vpMaxDepth ≤ 0.08). The viewmodel pass is
//   what Source renders weapons through; its c_cameraOrigin always carries
//   the actual pilot eye in pilot-on-foot, titan-cockpit, AND rodeo (pilot
//   on top of titan) modes. lp+0x3D6C is unreliable on this build (the log
//   shows it constant at (14158,-10801,877) over 46km of player travel —
//   it's a static script anchor, not a live eye field).
//
// Consumer: CameraManager::processCameraData snaps Main's worldToView
//   translation column to this so primary rays come from the actual pilot
//   position, not the BSP-pass cb2's titan body origin.
//
// Definitions must live in libdxvk (this TU) because libdxvk can't pull
// symbols out of the d3d11 DLL — the dependency points d3d11 → dxvk.
//
// Atomics: single-producer single-consumer, no ordering dependency, so
// memory_order_relaxed is sufficient at both ends.
namespace dxvk { namespace tf2 {
  std::atomic<float> g_pilotEyeX{ 0.0f };
  std::atomic<float> g_pilotEyeY{ 0.0f };
  std::atomic<float> g_pilotEyeZ{ 0.0f };
  std::atomic<bool>  g_pilotEyeValid{ false };
}}

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

  // NV-DXVK [viewmodel-eye-capture]: cached canonical pilot eye position,
  // captured from the viewmodel-pass cb2 (CamCatalog #11 family — maxZ≤0.08,
  // vp=backbuffer). Source's viewmodel pass renders weapons in eye-relative
  // space, so its cb2.c_cameraOrigin is wherever Source thinks the real eye
  // is — including titan-cockpit eye when piloting a Titan, which lp+0x3D6C
  // gets wrong on this build (reads ~60u above the actual eye, plausibly the
  // head-top / helmet-anchor or a stale on-foot field). The Main snap reads
  // this in preference to engineEye when fresh; falls back to engineEye if
  // never set or stale (e.g. before first frame, weapon-holstered cinematic).
  struct ViewModelEyeCache {
    float x = 0.f, y = 0.f, z = 0.f;
    uint32_t frame = UINT32_MAX;
    bool valid = false;
  };
  ViewModelEyeCache g_vmEye;

  // NV-DXVK [eye-snap-killswitch]: read once at first call. Lets the user
  // A/B disable my eye-snap + engineEye-reject without a recompile, so we
  // can isolate whether the camera fix or something else is causing the
  // initial-load DxvkMemoryAllocator: Memory allocation failed crash.
  //
  // Mechanism of concern: my snap rewrites the worldToView translation
  // column on the first Main draw to point at the pilot eye (~600u away
  // from the un-snapped BSP-pass position). RtCamera::isCameraCut() returns
  // true when ViewToWorld translation moves more than 300u between frames.
  // A camera-cut on the very first frame forces a full TLAS rebuild and
  // breaks BLAS reuse via uniqueObjectDistance proximity (every instance
  // looks "new" because its world position is 600u away from the previously
  // cached camera-relative position) → ~200 fresh BLAS allocs in a single
  // frame during initial scene upload → host-coherent transfer pool runs
  // out of contiguous space → 32 MB alloc fails.
  //
  // Set REMIX_TF2_DISABLE_EYE_SNAP=1 in env to neuter both the snap and the
  // reject. Game will revert to pre-fix camera behavior (titan-base / sky
  // teleport bugs return) but the alloc crash should also revert if my
  // theory is right. If alloc still fails with the snap disabled, the
  // crash is unrelated and I should stop blaming my code.
  inline bool EyeSnapDisabled() {
    // NV-DXVK [eye-snap-killswitch]: confirmed earlier that a fresh-activation
    // snap on frame=2 caused the DxvkMemoryAllocator: Memory allocation failed
    // crash via camera-cut → TLAS rebuild → BLAS cache invalidation cascade
    // → host-coherent pool exhaustion during initial scene upload.
    //
    // The fix below (gating + ramping) defeats the cause:
    //   1. Activation gate at kActivationFrame frames + a pilot-eye streak
    //      requirement ensures initial scene upload has fully completed and
    //      we're in steady-state gameplay, not a loading screen / intro
    //      cutscene / menu where camera state is unstable.
    //   2. Per-frame ramp limits how much the camera moves between frames
    //      to less than RtCamera::isCameraCut()'s threshold
    //      (uniqueObjectDistance²=300²), so the snap never triggers a cut
    //      regardless of the magnitude of the BSP→pilot-eye correction.
    // Master kill remains here for emergency disable. Set to true to revert
    // to BSP-pass camera (sky/titan-feet bugs return) if the gated snap
    // misbehaves.
    //
    // NV-DXVK [restore-excellent-state]: flipped to TRUE per user request
    // to reproduce the camera state that was visible at the "excellent
    // you pick the right camera but the game stays on one frame" moment
    // in conversation. At that moment, this entire eye-snap subsystem
    // didn't exist yet — the camera was whatever cachedSave/cachedConsume
    // produced from raw cb2 reads + the existing 5-unit player-cam
    // filter. Disabling the snap reverts to that pre-snap behavior.
    static const bool s_disabled = true;
    static const bool s_logged = []() {
      // str::format is in namespace dxvk; we're file-scope before
      // `namespace dxvk { ... }` opens, so we must fully qualify it.
      dxvk::Logger::info(dxvk::str::format(
        "[CamMgr.eye-snap-killswitch] HARDCODED disabled=", s_disabled ? "1" : "0",
        " — gated/ramped snap is", s_disabled ? " OFF" : " ON"));
      return true;
    }();
    (void)s_logged;
    return s_disabled;
  }

  // NV-DXVK [eye-snap-gating]: the activation gates and ramp constants. All
  // tunables are here so they're easy to find and adjust without sifting
  // through processCameraData's body.
  //
  // The user runs this at ~1 FPS, so frame counts are in real seconds.
  // A 1200-frame gate would mean 20 minutes of waiting. The actual safety
  // mechanism is the ramp (kMaxRampStepU keeps per-frame camera motion
  // strictly below the camera-cut threshold), so the activation gate is
  // really just defense-in-depth — small values are fine.
  //
  // kActivationFrame: minimum frameId before snap will engage. Set just
  //   past the very-early init phase (cb2 RDEF cache cold, swap chain
  //   not fully constructed) where draws can carry junk transforms.
  //
  // kPilotEyeStreakRequired: minimum consecutive Main-classified frames
  //   where the pilot-eye atomic was already valid. Confirms we're past
  //   any short loading transition into real gameplay. Counter resets
  //   to zero if pilot_eye becomes invalid (cinematic / menu).
  //
  // kMaxRampStepU: maximum per-frame change in the snapped camera world
  //   position. MUST stay strictly below RtCamera::isCameraCut()'s
  //   threshold of uniqueObjectDistance=300u so no cut ever fires from
  //   our snap motion. 250u leaves margin for normal player movement
  //   (~150u/s at 1 FPS) which adds linearly to the ramp step.
  constexpr uint32_t kActivationFrame = 5;
  constexpr uint32_t kPilotEyeStreakRequired = 2;
  constexpr float    kMaxRampStepU = 250.0f;
  // Vector3 is in namespace dxvk; we're file-scope before `namespace dxvk`
  // opens here, so use the qualified type.
  uint32_t       g_pilotEyeStreak = 0;
  dxvk::Vector3  g_lastSnappedCam{ 0.f, 0.f, 0.f };
  bool           g_haveLastSnapped = false;
  // NV-DXVK [eye-snap-per-frame-ramp]: track which frame we last advanced
  // the ramp on. processCameraData fires once per Main-classified DRAW,
  // and there are many draws per frame. Without this gate, a single frame
  // would ramp through the entire BSP→pilot delta (1000+ units) in one
  // frame's worth of draws — RtCamera's first-wins would latch only the
  // first draw's small step, but g_lastSnappedCam would jump fully — so
  // the NEXT frame's first draw snaps from the fully-advanced state and
  // delta-from-prev-frame becomes huge → camera-cut. Locking the ramp
  // advance to the first call per frame fixes this: g_lastSnappedCam
  // advances by ≤ kMaxRampStepU once per frame, in lockstep with what
  // RtCamera::update first-wins-latches into ViewToWorld[3].
  uint32_t       g_lastRampFrame = UINT32_MAX;
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
    // NV-DXVK [engineEye-gate]: candidate's decoded camera world position is
    // far from the engine ground-truth eye (lp+0x3D6C). Catches TF2's 3D-
    // skybox draw to backbuffer, whose w2v encodes the sky_camera entity
    // origin in skybox-scale coords (~thousands of units off from the real
    // player eye) and whose physical properties (vp/aspect/maxZ/fov) match
    // a real wide-FoV gameplay camera so the other gates can't reject it.
    uint32_t rejEngineEyeFar = 0;
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
        " engineEye=", g_mainHist.rejEngineEyeFar,
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
    // [pcdEnter] One-per-frame log proving processCameraData was called.
    // If pcdTrace later in the function NEVER fires for a frame, but
    // pcdEnter does, we know the function was called but every call
    // bailed via one of the early returns (identity v2p, fused-world-view
    // path, fov/shear validation). Then the exit-path log below will
    // confirm which return type each call ended up at.
    {
      const uint32_t fid = m_device->getCurrentFrameId();
      static std::atomic<uint32_t> sLastEnterFrame{UINT32_MAX};
      uint32_t expected = sLastEnterFrame.load(std::memory_order_relaxed);
      if (expected != fid) {
        if (sLastEnterFrame.compare_exchange_strong(expected, fid,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          const auto& w = input.getTransformData().worldToView;
          const auto& vtp = input.getTransformData().viewToProjection;
          const float tR = float(w[3][0]), tU = float(w[3][1]), tF = float(w[3][2]);
          const float camX = -(float(w[0][0])*tR + float(w[0][1])*tU + float(w[0][2])*tF);
          const float camY = -(float(w[1][0])*tR + float(w[1][1])*tU + float(w[1][2])*tF);
          const float camZ = -(float(w[2][0])*tR + float(w[2][1])*tU + float(w[2][2])*tF);
          Logger::info(str::format(
            "[pcdEnter] frame=", fid,
            " camPos=(", camX, ",", camY, ",", camZ, ")",
            " v2pIdent=", isIdentityExact(vtp) ? 1 : 0,
            " skyCat=", input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0));
        }
      }
    }

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
      // NV-DXVK [SpawnGeomDiag]: was ONCE() — flipped to a per-frame
      // throttled warn that prints actual decomposition so the missing-
      // geometry debug can correlate "frame N rejected camera with
      // shearX=…/fov=…" against the [SpawnGeomDiag] frame=N census.
      // ONCE() suppressed every rejection after the first, which made
      // spawn-window analysis blind once any earlier UI/sky pass ate the
      // single allowed message.
      const uint32_t fid = m_device->getCurrentFrameId();
      static uint32_t sLastWarnFrame = UINT32_MAX;
      if (fid != sLastWarnFrame) {
        sLastWarnFrame = fid;
        const bool isSky = input.getCategoryFlags().test(InstanceCategories::Sky);
        const bool fovBad = !isFovValid(decomposeProjectionParams.fov);
        const bool shearBad = std::abs(decomposeProjectionParams.shearX) > 0.01f;
        // [v2pReject] Dump the full viewToProjection matrix that's being
        // rejected. shearX=0.955 + nearPlane=-0 + aspect=-0.028 is suspicious
        // — looks more like the decomposer is misinterpreting TF2's matrix
        // layout (possibly transposed / Y-flipped / different handedness)
        // than it being a genuinely sheared projection. The matrix dump
        // lets us decide whether to relax the gate, fix the decomposer,
        // or transpose the input before decomposition.
        const auto& v2p = input.getTransformData().viewToProjection;
        Logger::warn(str::format(
          "[RTX] CameraManager: rejected an invalid camera",
          " frame=", fid,
          " sky=", (isSky ? 1 : 0),
          " fov=", decomposeProjectionParams.fov,
          " fovBad=", (fovBad ? 1 : 0),
          " shearX=", decomposeProjectionParams.shearX,
          " shearBad=", (shearBad ? 1 : 0),
          " nearPlane=", decomposeProjectionParams.nearPlane,
          " aspectRatio=", decomposeProjectionParams.aspectRatio,
          " => CameraType::", (isSky ? "Sky" : "Unknown")));
        Logger::warn(str::format(
          "[v2pReject] frame=", fid,
          " row0=(", v2p[0][0], ",", v2p[0][1], ",", v2p[0][2], ",", v2p[0][3], ")",
          " row1=(", v2p[1][0], ",", v2p[1][1], ",", v2p[1][2], ",", v2p[1][3], ")",
          " row2=(", v2p[2][0], ",", v2p[2][1], ",", v2p[2][2], ",", v2p[2][3], ")",
          " row3=(", v2p[3][0], ",", v2p[3][1], ",", v2p[3][2], ",", v2p[3][3], ")"));
      }
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

    // NV-DXVK [viewmodel-eye-capture]: when this draw classifies as ViewModel
    // via the maxZ ≤ vmThr branch (the canonical Source viewmodel-pass marker;
    // FoV-mismatch branch is excluded because it can fire for weird non-eye
    // passes that share Main's FoV by coincidence), record its decoded camera
    // world position. Subsequent Main draws snap their translation column to
    // this captured eye instead of lp+0x3D6C, which on Titan-piloting maps
    // (BT-7274) reads ~60u high — the viewmodel-pass cb2 is what Source uses
    // for the real eye position regardless of whether the player is on foot
    // or in a Titan, so it tracks correctly across both modes.
    if (cameraType == CameraType::ViewModel
        && RtxOptions::ViewModel::enable()
        && input.maxZ <= RtxOptions::ViewModel::maxZThreshold()) {
      const auto& td = input.getTransformData();
      const Matrix4& w = td.worldToView;
      // Sanity: only capture if rotation rows look orthonormal-ish — the
      // viewmodel pass we want has full mouse-look basis (R/U/F all
      // non-degenerate). The other viewmodel-classified draws bind a near-
      // identity matrix that just translates the weapon to eye-space and
      // would give a recovC of ~(0, 0, 0) which we don't want as snap target.
      const float r0Mag2 = w[0][0]*w[0][0] + w[1][0]*w[1][0] + w[2][0]*w[2][0];
      const float r1Mag2 = w[0][1]*w[0][1] + w[1][1]*w[1][1] + w[2][1]*w[2][1];
      const float r2Mag2 = w[0][2]*w[0][2] + w[1][2]*w[1][2] + w[2][2]*w[2][2];
      const bool basisOk =
        std::abs(r0Mag2 - 1.0f) < 0.05f &&
        std::abs(r1Mag2 - 1.0f) < 0.05f &&
        std::abs(r2Mag2 - 1.0f) < 0.05f;
      // Reject identity-rotation viewmodel matrices (the eye-relative
      // weapon-positioning ones — they'd set recovC to (0,0,0)).
      const bool rotIsIdentityVm =
        std::abs(w[0][0] - 1.0f) < 0.01f && std::abs(w[1][1] - 1.0f) < 0.01f &&
        std::abs(w[2][2] - 1.0f) < 0.01f;
      if (basisOk && !rotIsIdentityVm) {
        const float Cx = -(w[0][0]*w[3][0] + w[0][1]*w[3][1] + w[0][2]*w[3][2]);
        const float Cy = -(w[1][0]*w[3][0] + w[1][1]*w[3][1] + w[1][2]*w[3][2]);
        const float Cz = -(w[2][0]*w[3][0] + w[2][1]*w[3][1] + w[2][2]*w[3][2]);
        // |C| > 100 avoids capturing degenerate (0,0,0)-near matrices.
        if (Cx*Cx + Cy*Cy + Cz*Cz > 100.0f * 100.0f) {
          g_vmEye.x = Cx;
          g_vmEye.y = Cy;
          g_vmEye.z = Cz;
          g_vmEye.frame = frameId;
          g_vmEye.valid = true;
          static uint32_t sVmEyeLog = 0;
          if (sVmEyeLog < 40) {
            ++sVmEyeLog;
            Logger::info(str::format(
              "[CamMgr.viewmodel-eye-capture] #", sVmEyeLog,
              " frame=", frameId,
              " eye=(", Cx, ",", Cy, ",", Cz, ")",
              " maxZ=", input.maxZ));
          }
        }
      }
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
      // NV-DXVK [engineEye-gate]: when engine ground truth (client.dll
      // local-player + 0x3D6C) is available, reject any Main candidate whose
      // decoded camera world position is more than ~kEngineEyeMaxDistU units
      // from the player's actual eye. This catches TF2's 3D-skybox draw to
      // backbuffer — its worldToView encodes the sky_camera entity origin
      // in skybox-scale coords (e.g. (14651,-2017,-13) on BT-7274) which is
      // thousands of units off from the player. The physical-property gates
      // above can't reject it because vp=1920×1080 / aspect=1.78 / fov ~119°
      // / maxZ=1 all look like a legitimate wide-FoV gameplay pass. Without
      // this gate, the skybox draw wins Main on whichever frame it lands
      // first, the next frame the gameplay-world draw wins, and the next a
      // viewmodel-pass — RtCamera teleports between three positions per
      // frame and the user perceives "feet view" / "camera in wrong
      // position." Log evidence: see [rtcam-pos-trace] in remix-dxvk.log
      // showing |delta|=9264 between consecutive frames.
      //
      // The distance threshold has to tolerate a few hundred units of
      // legitimate disagreement (the engine eye is the un-bobbed eye, while
      // a per-draw cb2 may carry the bobbed eye or an off-center cinematic;
      // and dxvk-side worldToView decomposition is not exact when R/U/F are
      // not perfectly orthonormal). 1500 u is generous: real player motion
      // never leaves the gate range; the sky_camera origin sits ~9000 u
      // away, dramatically out of range.
      bool keepAsMain =
        isInWorld && isNonSquare && isReasonableDepth && isReasonableFov && isLargeEnough;
      bool engineEyeFar = false;
      // Same precedence as the snap below: pilot-eye atomic > lp+0x3D6C.
      // The pilot-eye atomic tracks the player live; lp is static on this
      // build so it'd reject Main candidates that legitimately followed the
      // player away from the static anchor.
      float refX = 0.f, refY = 0.f, refZ = 0.f;
      bool haveRef = false;
      const char* refSource = nullptr;
      if (dxvk::tf2::g_pilotEyeValid.load(std::memory_order_relaxed)) {
        refX = dxvk::tf2::g_pilotEyeX.load(std::memory_order_relaxed);
        refY = dxvk::tf2::g_pilotEyeY.load(std::memory_order_relaxed);
        refZ = dxvk::tf2::g_pilotEyeZ.load(std::memory_order_relaxed);
        haveRef = true;
        refSource = "pilot";
      } else if (const float* eye = GetEngineEyeCM()) {
        if (std::isfinite(eye[0]) && std::isfinite(eye[1]) && std::isfinite(eye[2])) {
          refX = eye[0]; refY = eye[1]; refZ = eye[2];
          haveRef = true;
          refSource = "lp";
        }
      }
      // Gate the reject by the same activation rule as the snap below.
      // Activating the reject during initial load would also force a Main
      // re-classification (a previously-Main candidate becomes Unknown,
      // a different candidate may take Main next frame) → camera-cut →
      // BLAS cache invalidation cascade → same OOM as the un-gated snap.
      // After kActivationFrame frames + kPilotEyeStreakRequired streak, we
      // can safely reject the 9000u-off skybox candidate without risk.
      const bool rejectGateOpen =
        frameId >= kActivationFrame
        && g_pilotEyeStreak >= kPilotEyeStreakRequired;
      if (keepAsMain && haveRef && rejectGateOpen && !EyeSnapDisabled()) {
        // Decode camera world position C = -R_rot^T · t from worldToView.
        // dxvk Matrix4 stores M[col][row]; math row 0 = R = first column
        // of the rotation block laid out across w2v[0..2][0]. Mirror the
        // decomposition done in [CamMgr.classify] above so reject and
        // diagnostic log agree exactly.
        const float Cx = -(w2v[0][0]*w2v[3][0] + w2v[0][1]*w2v[3][1] + w2v[0][2]*w2v[3][2]);
        const float Cy = -(w2v[1][0]*w2v[3][0] + w2v[1][1]*w2v[3][1] + w2v[1][2]*w2v[3][2]);
        const float Cz = -(w2v[2][0]*w2v[3][0] + w2v[2][1]*w2v[3][1] + w2v[2][2]*w2v[3][2]);
        const float dx = Cx - refX;
        const float dy = Cy - refY;
        const float dz = Cz - refZ;
        const float d2 = dx*dx + dy*dy + dz*dz;
        constexpr float kEngineEyeMaxDistU = 1500.0f;
        if (d2 > (kEngineEyeMaxDistU * kEngineEyeMaxDistU)) {
          engineEyeFar = true;
          keepAsMain = false;
          static uint32_t sEELog = 0;
          if (sEELog < 40) {
            ++sEELog;
            Logger::info(str::format(
              "[CamMgr.engineEye-reject] #", sEELog,
              " src=", refSource,
              " frame=", frameId,
              " recovC=(", Cx, ",", Cy, ",", Cz, ")",
              " ref=(", refX, ",", refY, ",", refZ, ")",
              " |delta|=", std::sqrt(d2),
              " threshold=", kEngineEyeMaxDistU,
              " fov=", fovDeg, "deg vp=", int(vw), "x", int(vh)));
          }
        }
      }
      noteFrame(frameId);
      ++g_mainHist.candidates;
      if (!isInWorld) ++g_mainHist.rejIsInWorld;
      if (!isNonSquare) ++g_mainHist.rejIsNonSquare;
      if (!isReasonableDepth) ++g_mainHist.rejIsReasonableDepth;
      if (!isReasonableFov) ++g_mainHist.rejIsReasonableFov;
      if (!isLargeEnough) ++g_mainHist.rejIsLargeEnough;
      if (engineEyeFar) ++g_mainHist.rejEngineEyeFar;
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
            " isLargeEnough=", isLargeEnough ? 1 : 0,
            " engineEyeFar=", engineEyeFar ? 1 : 0));
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

    // NV-DXVK [engineEye-snap]: when this draw classifies as Main and engine
    // ground truth (lp+0x3D6C) is valid, rewrite the worldToView translation
    // column so the encoded camera world position is exactly engineEye while
    // R/U/F (the mouse-look basis) is preserved.
    //
    // Why: TF2's BSP-world cb2 binds c_cameraOrigin = the player's world-
    // reference point (titan-base on BT-7274 — Z=214 with the cockpit eye at
    // Z=877, ~600u above). Vanilla TF2 composites BSP and viewmodel passes
    // separately and they each get the right per-pass cb2. Remix funnels the
    // gameplay-world worldToView into a single RtCamera and shoots primary
    // rays from there — so the camera is rendered from titan-feet position
    // and the user sees the inside-of-titan / "feet view" symptom.
    // The viewmodel cb2 (#11 in [CamCatalog]) already binds the real eye but
    // its R/U/F is identity-like (viewmodel is rendered in eye-relative
    // space) so we can't take its basis. The right answer is to keep the
    // BSP-world basis (correct mouse-look orientation) and replace its
    // translation with -basis·engineEye so the encoded camera position is
    // the actual cockpit eye. BLAS positions live in world space and are
    // unaffected — only the camera's ray-origin moves to the right place.
    //
    // Gated on |recovC - engineEye| <= kEngineEyeMaxDistU above (1500 u),
    // so frankly-wrong matrices (3D-skybox draw at 9000+ u off, F-row +Z)
    // were already rejected and never reach this point.
    if (cameraType == CameraType::Main && !EyeSnapDisabled()) {
      // Snap target precedence:
      //   1. d3d11_rtx pilot-eye atomic (g_pilotEyeX/Y/Z) — the raw cb2
      //      c_cameraOrigin field captured at the viewmodel-pass fanout
      //      site. Source's authoritative eye, correct in pilot-on-foot,
      //      titan-cockpit, AND rodeo (pilot on top of Titan). NO matrix
      //      decomposition → no per-VS float noise.
      //   2. Local matrix-decomposed g_vmEye (legacy fallback) — only if
      //      the d3d11_rtx atomic isn't valid yet.
      //   3. lp+0x3D6C (engineEye) — on this build it's a static script
      //      anchor, so it's the LAST resort.
      // Skip entirely if no source is valid (very early frames).
      float snapEyeX = 0.f, snapEyeY = 0.f, snapEyeZ = 0.f;
      const char* snapSource = nullptr;
      const uint32_t curFrame = m_device->getCurrentFrameId();
      const bool pilotEyeValid =
        dxvk::tf2::g_pilotEyeValid.load(std::memory_order_relaxed);
      if (pilotEyeValid) {
        snapEyeX = dxvk::tf2::g_pilotEyeX.load(std::memory_order_relaxed);
        snapEyeY = dxvk::tf2::g_pilotEyeY.load(std::memory_order_relaxed);
        snapEyeZ = dxvk::tf2::g_pilotEyeZ.load(std::memory_order_relaxed);
        snapSource = "pilot";
      } else if (g_vmEye.valid
          && (curFrame <= g_vmEye.frame || (curFrame - g_vmEye.frame) <= 1)) {
        snapEyeX = g_vmEye.x;
        snapEyeY = g_vmEye.y;
        snapEyeZ = g_vmEye.z;
        snapSource = "vm";
      } else if (const float* eye = GetEngineEyeCM()) {
        if (std::isfinite(eye[0]) && std::isfinite(eye[1]) && std::isfinite(eye[2])) {
          snapEyeX = eye[0];
          snapEyeY = eye[1];
          snapEyeZ = eye[2];
          snapSource = "lp";
        }
      }

      // NV-DXVK [eye-snap-gating]: defer activation past initial load and
      // intro-cinematic frames, then ramp the snap target gradually over
      // multiple frames so RtCamera::isCameraCut() never sees a step larger
      // than its threshold. See kActivationFrame / kPilotEyeStreakRequired
      // / kMaxRampStepU at the top of this file for tunable values.
      //
      // Streak: increment when pilot_eye is valid in this frame's snap
      // attempt; reset to zero otherwise. The streak only progresses while
      // we're seeing a Main-classified candidate, which means we're rendering
      // gameplay (not a paused menu / loading screen).
      if (pilotEyeValid) {
        ++g_pilotEyeStreak;
      } else {
        g_pilotEyeStreak = 0;
      }
      const bool gateOpen =
        snapSource != nullptr
        && curFrame >= kActivationFrame
        && g_pilotEyeStreak >= kPilotEyeStreakRequired;

      // Once the gate opens, ramp the snap-applied camera world position
      // toward the snap target, capping per-frame motion to kMaxRampStepU.
      // First gate-open frame: initialize g_lastSnappedCam to the matrix's
      // own recovC so the first snap is a no-op (zero motion → zero risk
      // of camera-cut). Subsequent frames step toward snapEye.
      if (gateOpen) {
        const Matrix4& w = worldToView;
        const float Cx = -(w[0][0]*w[3][0] + w[0][1]*w[3][1] + w[0][2]*w[3][2]);
        const float Cy = -(w[1][0]*w[3][0] + w[1][1]*w[3][1] + w[1][2]*w[3][2]);
        const float Cz = -(w[2][0]*w[3][0] + w[2][1]*w[3][1] + w[2][2]*w[3][2]);
        if (!g_haveLastSnapped) {
          g_lastSnappedCam = Vector3(Cx, Cy, Cz);
          g_haveLastSnapped = true;
          dxvk::Logger::info(str::format(
            "[CamMgr.engineEye-snap-activate] frame=", curFrame,
            " streak=", g_pilotEyeStreak,
            " initialCam=(", Cx, ",", Cy, ",", Cz, ")",
            " target=(", snapEyeX, ",", snapEyeY, ",", snapEyeZ, ")"));
        }
        // Advance the ramp ONCE per frame, regardless of how many Main
        // draws fire processCameraData. Subsequent draws in the same frame
        // reuse the same g_lastSnappedCam, so they all snap to the same
        // camera position — keeping multiple Main candidates within a
        // frame consistent with each other. Without this gate, the ramp
        // advances by 250u per draw within a single frame; RtCamera's
        // first-wins latches the first draw's small step but g_lastSnappedCam
        // jumps the full distance, and the NEXT frame's first call would
        // snap from a fully-advanced position → camera-cut between frames.
        if (curFrame != g_lastRampFrame) {
          g_lastRampFrame = curFrame;
          const float rdx = snapEyeX - g_lastSnappedCam.x;
          const float rdy = snapEyeY - g_lastSnappedCam.y;
          const float rdz = snapEyeZ - g_lastSnappedCam.z;
          const float rdMag = std::sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
          Vector3 effective;
          if (rdMag > kMaxRampStepU) {
            const float s = kMaxRampStepU / rdMag;
            effective.x = g_lastSnappedCam.x + rdx * s;
            effective.y = g_lastSnappedCam.y + rdy * s;
            effective.z = g_lastSnappedCam.z + rdz * s;
          } else {
            effective.x = snapEyeX;
            effective.y = snapEyeY;
            effective.z = snapEyeZ;
          }
          g_lastSnappedCam = effective;
        }
        // All draws in this frame snap to the same g_lastSnappedCam value.
        snapEyeX = g_lastSnappedCam.x;
        snapEyeY = g_lastSnappedCam.y;
        snapEyeZ = g_lastSnappedCam.z;
      } else {
        // Gate not yet open: do not snap. Leave snapSource set so the
        // log below still shows what the source matrix's recovC was vs
        // what the snap target would be, useful for verifying the gate
        // logic without yet touching worldToView.
        snapSource = nullptr;
      }

      if (snapSource) {
        const Matrix4& w = worldToView;
        const float Cx = -(w[0][0]*w[3][0] + w[0][1]*w[3][1] + w[0][2]*w[3][2]);
        const float Cy = -(w[1][0]*w[3][0] + w[1][1]*w[3][1] + w[1][2]*w[3][2]);
        const float Cz = -(w[2][0]*w[3][0] + w[2][1]*w[3][1] + w[2][2]*w[3][2]);
        const float dx = Cx - snapEyeX;
        const float dy = Cy - snapEyeY;
        const float dz = Cz - snapEyeZ;
        const float d2 = dx*dx + dy*dy + dz*dz;
        // Only snap when the matrix is in the right ballpark — the gate
        // upstream rejects > 1500 u, so this is a redundant safety.
        // Skip the rewrite if recovC is already very close (< 1 u) to
        // avoid pointless float churn on already-correct frames.
        if (d2 > 1.0f && d2 <= (1500.0f * 1500.0f)) {
          // t = -(R·C, U·C, F·C). Same row-major decomposition as recovC
          // above, just inverted.
          const float new_tx = -(w[0][0]*snapEyeX + w[1][0]*snapEyeY + w[2][0]*snapEyeZ);
          const float new_ty = -(w[0][1]*snapEyeX + w[1][1]*snapEyeY + w[2][1]*snapEyeZ);
          const float new_tz = -(w[0][2]*snapEyeX + w[1][2]*snapEyeY + w[2][2]*snapEyeZ);
          worldToView[3][0] = new_tx;
          worldToView[3][1] = new_ty;
          worldToView[3][2] = new_tz;
          // Throttled diagnostic so we can confirm the snap is firing and
          // see how big the correction was per draw.
          static uint32_t sSnapLog = 0;
          if (sSnapLog < 60) {
            ++sSnapLog;
            Logger::info(str::format(
              "[CamMgr.engineEye-snap] #", sSnapLog,
              " src=", snapSource,
              " frame=", curFrame,
              " oldC=(", Cx, ",", Cy, ",", Cz, ")",
              " newC=(", snapEyeX, ",", snapEyeY, ",", snapEyeZ, ")",
              " |delta|=", std::sqrt(d2)));
          }
        }
      }
    }

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

    // [pcdTrace] One log per (frame, cameraType-class) so we see every
    // *kind* of classification result that processCameraData produces
    // each frame. Goal: during the (0,0,0)-bootstrap freeze, find out
    // which cameraType the player-position draws are getting
    // classified as. If they're flipping to Unknown / ViewModel /
    // RenderToTexture instead of Main, that's why pcdMainTrace stopped
    // and the Main camera stops advancing.
    {
      const uint32_t fid = m_device->getCurrentFrameId();
      const uint32_t typeIdx = static_cast<uint32_t>(cameraType);
      // Pack (frame, type) into single uint64 for atomic CAS dedup.
      const uint64_t key = (uint64_t(fid) << 8) | (typeIdx & 0xff);
      static std::atomic<uint64_t> sLastKey{UINT64_MAX};
      uint64_t expected = sLastKey.load(std::memory_order_relaxed);
      if (expected != key) {
        if (sLastKey.compare_exchange_strong(expected, key,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          const auto& w = input.getTransformData().worldToView;
          const float tR = float(w[3][0]), tU = float(w[3][1]), tF = float(w[3][2]);
          const float camX = -(float(w[0][0])*tR + float(w[0][1])*tU + float(w[0][2])*tF);
          const float camY = -(float(w[1][0])*tR + float(w[1][1])*tU + float(w[1][2])*tF);
          const float camZ = -(float(w[2][0])*tR + float(w[2][1])*tU + float(w[2][2])*tF);
          const char* tname = "?";
          switch (cameraType) {
            case CameraType::Main:             tname = "Main"; break;
            case CameraType::Sky:              tname = "Sky"; break;
            case CameraType::ViewModel:        tname = "ViewModel"; break;
            case CameraType::RenderToTexture:  tname = "RenderToTexture"; break;
            case CameraType::Unknown:          tname = "Unknown"; break;
            default: break;
          }
          Logger::info(str::format(
            "[pcdTrace] frame=", fid,
            " type=", tname,
            " camPos=(", camX, ",", camY, ",", camZ, ")",
            " skyCat=", input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0));
        }
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
