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

#include "rtx_camera.h"
#include "rtx_option.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_matrix_helpers.h"

namespace dxvk {
  class DxvkDevice;
  class RayPortalManager;
  class Config;

  class CameraManager : public CommonDeviceObject {
  public:
    explicit CameraManager(DxvkDevice* device);
    ~CameraManager() override = default;

    CameraManager(const CameraManager& other) = delete;
    CameraManager(CameraManager&& other) noexcept = delete;
    CameraManager& operator=(const CameraManager& other) = delete;
    CameraManager& operator=(CameraManager&& other) noexcept = delete;

    const RtCamera& getCamera(CameraType::Enum cameraType) const { return accessCamera(*this, cameraType); }
    RtCamera& getCamera(CameraType::Enum cameraType) { return accessCamera(*this, cameraType); }

    const RtCamera& getMainCamera() const { return getCamera(CameraType::Main); }
    RtCamera& getMainCamera() { return getCamera(CameraType::Main); }

    CameraType::Enum getLastSetCameraType() const { return m_lastSetCameraType; }
    
    bool isCameraValid(CameraType::Enum cameraType) const;

    void onFrameEnd();

    // Calculates a camera type for the specified draw call.
    CameraType::Enum processCameraData(const DrawCallState& input);
    void processExternalCamera(CameraType::Enum type, const Matrix4& worldToView, const Matrix4& viewToProjection);

    uint32_t getLastCameraCutFrameId() const { return m_lastCameraCutFrameId; }
    bool isCameraCutThisFrame() const;

    // NV-DXVK: TLAS-coherence support. Distinguishes Main set by the classifier
    // (trusted pose — draw's projection matched a gameplay VS hash at depth
    // range 0.05 and |w2vT| > 100) from Main set by the safety net
    // (processExternalCamera — just whatever ExtractTransforms produced, often
    // identity during menus/cinematics). The TLAS coherence filter only trusts
    // classifier-set Main; safety-net Main passes all draws through.
    bool isMainSetByClassifier() const { return m_mainSetByClassifierFrameId != UINT32_MAX; }
    uint32_t getMainClassifierFrameId() const { return m_mainSetByClassifierFrameId; }
    void noteMainSetByClassifier(uint32_t frameId) { m_mainSetByClassifierFrameId = frameId; }

    // NV-DXVK: hysteresis state used by the Main classifier to reject
    // frame-to-frame flicker between shadow/reflection/cubemap candidates and
    // the real gameplay view. See processCameraData.
    struct MainLatchSnapshot {
      float fovRad       = 0.0f;
      float viewportW    = 0.0f;
      float viewportH    = 0.0f;
      // NV-DXVK TF2: also snapshot maxZ so subsequent latches from a
      // different render pass (same viewport+fov but different depth range,
      // e.g. shadow/env/viewmodel) get HARD-rejected by hysteresis instead
      // of stealing Main.
      float maxZ         = 0.0f;
      Vector3 fwd        = Vector3(0.0f, 0.0f, 0.0f);
      // NV-DXVK TF2: also snapshot the camera's right vector so the basis
      // consistency check can detect roll around the forward axis. Without
      // this, two candidate draws with identical forward but ~90° different
      // roll both pass the check and Main can latch onto a sideways pose.
      Vector3 right      = Vector3(0.0f, 0.0f, 0.0f);
      Vector3 pos        = Vector3(0.0f, 0.0f, 0.0f);
      bool   valid       = false;
      uint32_t frameId   = UINT32_MAX;
    };
    const MainLatchSnapshot& getMainLatchSnapshot() const { return m_mainLatchSnapshot; }

  private:
    template<
      typename T,
      std::enable_if_t<std::is_same_v<T, CameraManager> || std::is_same_v<T, const CameraManager>, bool> = true>
    static auto& accessCamera(T& cameraManager, CameraType::Enum cameraType) {
      assert(cameraType < CameraType::Count);
      // Default Unknown to Main camera object
      // since cameras can get rejected but rtx pipeline can
      // still try to retrieve a camera corresponding to a DrawCall object.
      // In that case it will read from the Main camera.
      // This is OK as we never update Unknown camera directly.
      if (cameraType == CameraType::Unknown) {
        return cameraManager.m_cameras[CameraType::Main];
      }
      return cameraManager.m_cameras[cameraType];
    }

  private:
    std::array<RtCamera, CameraType::Count> m_cameras;
    CameraType::Enum m_lastSetCameraType = CameraType::Unknown;
    uint32_t m_lastCameraCutFrameId = -1;
    // NV-DXVK: last frame where Main was latched by the classifier (vs. safety
    // net). UINT32_MAX means never. See isMainSetByClassifier() doc above.
    uint32_t m_mainSetByClassifierFrameId = UINT32_MAX;
    // NV-DXVK: snapshot of the last accepted Main latch — used for hysteresis
    // on subsequent candidates. See processCameraData.
    MainLatchSnapshot m_mainLatchSnapshot;
    // NV-DXVK: count of consecutive frames where a candidate disagreed with
    // the latched Main. Reset on agreement. Once it exceeds kCutStreakThreshold
    // the classifier accepts a re-latch (assumed camera cut).
    uint32_t m_disagreeStreak = 0;
    uint32_t m_lastCameraViewHistoryInvalidationFrameId = -1;
    fast_unordered_cache<DecomposeProjectionParams> m_decompositionCache;

    DecomposeProjectionParams getOrDecomposeProjection(const Matrix4& viewToProjection);

    RTX_OPTION("rtx", bool, rayPortalEnabled, false, "Enables ray portal support. Note this requires portal texture hashes to be set for the ray portal geometries in rtx.rayPortalModelTextureHashes.");
  };
}  // namespace dxvk
