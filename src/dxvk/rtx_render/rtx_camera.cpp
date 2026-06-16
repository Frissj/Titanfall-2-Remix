/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "../../util/util_math.h"
#include "../../util/util_vector.h"
#include "../../util/util_matrix.h"
#include "../imgui/dxvk_imgui.h"
#include "rtx_camera.h"
#include <windows.h>
#include "rtx_options.h"
#include "rtx_option_layer.h"
#include "rtx_matrix_helpers.h"
#include "rtx_imgui.h"
#include "rtx_xess.h"

/*
*             Free/Debug Camera
* 
* NOTE: Enable with the 'rtx.useFreeCamera = True' DXVK setting
* 
* W --------------------------- Move forward
* S --------------------------- Move backward
* A --------------------------- Move left
* D --------------------------- Move right
* Q --------------------------- Move down
* E --------------------------- Move up
* Left Shift ------------------ Move speed boost (hold)
* Left click (hold) + Mouse --- Look around
* Right click ----------------- Reset camera to default
* 
*/

namespace dxvk
{
  RtCameraSequence* RtCameraSequence::s_instance = nullptr;

  Vector3 RtCamera::getPosition(bool freecam) const {
    return Vector3{ getViewToWorld(freecam)[3].xyz() };
  }

  Vector3 RtCamera::getDirection(bool freecam) const {
    // Note: To get a "forward" direction in world space, the basis vector mapping world to view space's Z axis
    // is used, but unlike the up/right vectors we must consider which direction the projection matrix is treating as
    // forwards. With left handed projection matrices this is the +Z axis, but with right handed matrices this is -Z.
    if (m_context.isLHS) {
      return Vector3{ getViewToWorld(freecam)[2].xyz() };
    } else {
      return -Vector3{ getViewToWorld(freecam)[2].xyz() };
    }
  }

  Vector3 RtCamera::getUp(bool freecam) const {
    return Vector3{ getViewToWorld(freecam)[1].xyz() };
  }

  Vector3 RtCamera::getRight(bool freecam) const {
    return Vector3{ getViewToWorld(freecam)[0].xyz() };
  }

  Vector3 RtCamera::getPreviousPosition(bool freecam) const {
    return Vector3 { getPreviousViewToWorld(freecam)[3].xyz() };
  }

  std::pair<float, float> RtCamera::calculateNearFarPlanes() const {
    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatViewToProj{ getViewToProjection() };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatViewToProj));

    uint32_t flags;
    float    cameraParams[PROJ_NUM];
    DecomposeProjection(
      NDC_D3D,
      NDC_D3D,
      *reinterpret_cast<float4x4*>(&floatViewToProj),
      &flags,
      cameraParams,
      nullptr,
      nullptr,
      nullptr,
      nullptr);

    return { cameraParams[PROJ_ZNEAR], cameraParams[PROJ_ZFAR] };
  }

  bool RtCamera::isCameraCut() const {
    return lengthSqr(getViewToWorld()[3] - getPreviousViewToWorld()[3]) > RtxOptions::getUniqueObjectDistanceSqr();
  }

  bool RtCamera::isFreeCameraEnabled() {
    return enableFreeCamera();
  }

  Vector3 RtCamera::getHorizontalForwardDirection() const {
    const Vector3 forward = getDirection(false);
    const Vector3 up = getUp(false);
    const bool isZUp = RtxOptions::zUp();

    Vector3 direction = forward;

    if (isZUp)
      direction.z = 0.f;
    else
      direction.y = 0.f;

    float len = length(direction);
    if (len == 0.f) {
      // Looking straight down or up
      if (isZUp && forward.z > 0 || !isZUp && forward.y > 0)
        direction = -up;
      else
        direction = up;

      if (isZUp)
        direction.z = 0.f;
      else
        direction.y = 0.f;

      len = length(direction);
    }

    direction /= len;
    return direction;
  }

  namespace
  {
    /** Returns elements of the Halton low-discrepancy sequence.
        \param[in] index Index of the queried element, starting from 0.
        \param[in] base Base for the digit inversion. Should be the next unused prime number.
    */
    float halton(uint32_t index, uint32_t base)
    {
      // Reversing digit order in the given base in floating point.
      float result = 0.0f;
      float factor = 1.0f;

      for (; index > 0; index /= base) {
        factor /= base;
        result += factor * (index % base);
      }

      return result;
    }
  }

  HaltonSamplePattern::HaltonSamplePattern(uint32_t sampleCount)
  {
    mSampleCount = sampleCount;
    mCurSample = 0;
  }

  glm::vec2 HaltonSamplePattern::next()
  {
    glm::vec2 value = { halton(mCurSample, 2), halton(mCurSample, 3) };

    // Modular increment.
    ++mCurSample;
    if (mSampleCount != 0) {
      mCurSample = mCurSample % mSampleCount;
    }

    // Map the result so that [0, 1) maps to [-0.5, 0.5) and 0 maps to the origin.
    return glm::fract(value + 0.5f) - 0.5f;
  }

  Vector2 calculateHaltonJitter(uint32_t currentFrame, uint32_t jitterSequenceLength) {
    // Halton jitter
    Vector2 result(0.0f, 0.0f);

    uint32_t frameIndex = currentFrame % jitterSequenceLength;

    constexpr uint32_t baseX = 2;
    uint32_t Index = frameIndex + 1;
    float invBase = 1.0f / baseX;
    float fraction = invBase;
    while (Index > 0) {
      result.x += (Index % baseX) * fraction;
      Index /= baseX;
      fraction *= invBase;
    }

    constexpr uint32_t baseY = 3;
    Index = frameIndex + 1;
    invBase = 1.0f / baseY;
    fraction = invBase;
    while (Index > 0) {
      result.y += (Index % baseY) * fraction;
      Index /= baseY;
      fraction *= invBase;
    }

    result.x -= 0.5f;
    result.y -= 0.5f;
    return result;
  }

  const Matrix4d& RtCamera::getViewToWorldToFreeCamViewToWorld() const {
    assert(m_type == CameraType::Main && enableFreeCamera() && "Transform is only calculated for Main Camera when Free Camera is enabled");
    return m_matCache[MatrixType::ViewToWorldToFreeCamViewToWorld];
  }

  void RtCamera::setResolution(const uint32_t renderResolution[2], const uint32_t finalResolution[2]) {
    if (finalResolution[0] != m_finalResolution[0] ||
        finalResolution[1] != m_finalResolution[1] ||
        renderResolution[0] != m_renderResolution[0] ||
        renderResolution[1] != m_renderResolution[1]) {
      float resolutionRatio = float(finalResolution[1]) / float(renderResolution[1]);
      float basePhaseCount = 8;
      float totalPhases = basePhaseCount * resolutionRatio * resolutionRatio;
      m_halton = HaltonSamplePattern(uint32_t(totalPhases));

      m_renderResolution[0] = renderResolution[0];
      m_renderResolution[1] = renderResolution[1];
      m_finalResolution[0] = finalResolution[0];
      m_finalResolution[1] = finalResolution[1];
    }
  }

  const Matrix4d& RtCamera::getWorldToView(bool freecam) const { 
    return (freecam && isFreeCameraEnabled()) 
      ? m_matCache[MatrixType::FreeCamWorldToView] 
      : m_matCache[MatrixType::WorldToView]; 
  }

  const Matrix4d& RtCamera::getPreviousWorldToView(bool freecam) const { 
    return (freecam && isFreeCameraEnabled()) 
      ? m_matCache[MatrixType::FreeCamPreviousWorldToView] 
      : m_matCache[MatrixType::PreviousWorldToView]; 
  }

  const Matrix4d& RtCamera::getPreviousPreviousWorldToView(bool freecam) const { 
    return (freecam && isFreeCameraEnabled()) 
      ? m_matCache[MatrixType::FreeCamPreviousPreviousViewToWorld]
      : m_matCache[MatrixType::PreviousPreviousWorldToView];
  }

  const Matrix4d& RtCamera::getViewToWorld(bool freecam) const { 
    return (freecam && isFreeCameraEnabled()) 
      ? m_matCache[MatrixType::FreeCamViewToWorld] 
      : m_matCache[MatrixType::ViewToWorld]; 
  }

  const Matrix4d& RtCamera::getPreviousViewToWorld(bool freecam) const { 
    return (freecam && isFreeCameraEnabled()) 
      ? m_matCache[MatrixType::FreeCamPreviousViewToWorld] 
      : m_matCache[MatrixType::PreviousViewToWorld]; 
  }
  
  const Matrix4d& RtCamera::getPreviousPreviousViewToWorld(bool freecam) const {
    return (freecam && isFreeCameraEnabled())
      ? m_matCache[MatrixType::FreeCamPreviousPreviousViewToWorld]
      : m_matCache[MatrixType::PreviousPreviousViewToWorld];
  }

  const Matrix4d& RtCamera::getTranslatedWorldToView(bool freecam) const {
    return (freecam && isFreeCameraEnabled())
      ? m_matCache[MatrixType::FreeCamTranslatedWorldToView]
      : m_matCache[MatrixType::TranslatedWorldToView];
  }

  const Matrix4d& RtCamera::getPreviousTranslatedWorldToView(bool freecam) const {
    return (freecam && isFreeCameraEnabled())
      ? m_matCache[MatrixType::FreeCamPreviousTranslatedWorldToView]
      : m_matCache[MatrixType::PreviousTranslatedWorldToView];
  }

  const Matrix4d& RtCamera::getViewToTranslatedWorld(bool freecam) const {
    return (freecam && isFreeCameraEnabled())
      ? m_matCache[MatrixType::FreeCamViewToTranslatedWorld]
      : m_matCache[MatrixType::ViewToTranslatedWorld];
  }

  const Matrix4d& RtCamera::getPreviousViewToTranslatedWorld(bool freecam) const {
    return (freecam && isFreeCameraEnabled())
      ? m_matCache[MatrixType::FreeCamPreviousViewToTranslatedWorld]
      : m_matCache[MatrixType::PreviousViewToTranslatedWorld];
  }

  void RtCamera::setPreviousWorldToView(const Matrix4d& worldToView, bool freecam) {
    Matrix4d viewToWorld = inverse(worldToView);

    Matrix4d viewToTranslatedWorld = viewToWorld;
    viewToTranslatedWorld[3] = Vector4d(0.0, 0.0, 0.0, viewToTranslatedWorld[3].w);

    Matrix4d translatedWorldToView = inverse(viewToTranslatedWorld);

    if (freecam && isFreeCameraEnabled()) {
      m_matCache[MatrixType::FreeCamPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::FreeCamPreviousPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::FreeCamPreviousWorldToView] = worldToView;
      m_matCache[MatrixType::FreeCamPreviousPreviousWorldToView] = worldToView;

      m_matCache[MatrixType::FreeCamPreviousViewToTranslatedWorld] = viewToTranslatedWorld;
      m_matCache[MatrixType::FreeCamPreviousTranslatedWorldToView] = translatedWorldToView;
    } else {
      m_matCache[MatrixType::PreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::PreviousPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::PreviousWorldToView] = worldToView;
      m_matCache[MatrixType::PreviousPreviousWorldToView] = worldToView;

      m_matCache[MatrixType::PreviousViewToTranslatedWorld] = viewToTranslatedWorld;
      m_matCache[MatrixType::PreviousTranslatedWorldToView] = translatedWorldToView;
    }

    m_previousArtificalWorldOffset = Vector3(0.f);
  }

  void RtCamera::setPreviousViewToWorld(const Matrix4d& viewToWorld, bool freecam) {
    Matrix4d viewToTranslatedWorld = viewToWorld;
    viewToTranslatedWorld[3] = Vector4d(0.0, 0.0, 0.0, viewToTranslatedWorld[3].w);

    Matrix4d worldToView = inverse(viewToWorld);
    Matrix4d translatedWorldToView = inverse(viewToTranslatedWorld);

    if (freecam && isFreeCameraEnabled()) {
      m_matCache[MatrixType::FreeCamPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::FreeCamPreviousPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::FreeCamPreviousWorldToView] = worldToView;
      m_matCache[MatrixType::FreeCamPreviousPreviousWorldToView] = worldToView;

      m_matCache[MatrixType::FreeCamPreviousViewToTranslatedWorld] = viewToTranslatedWorld;
      m_matCache[MatrixType::FreeCamPreviousTranslatedWorldToView] = translatedWorldToView;
    } else {
      m_matCache[MatrixType::PreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::PreviousPreviousViewToWorld] = viewToWorld;
      m_matCache[MatrixType::PreviousWorldToView] = worldToView;
      m_matCache[MatrixType::PreviousPreviousWorldToView] = worldToView;

      m_matCache[MatrixType::PreviousViewToTranslatedWorld] = viewToTranslatedWorld;
      m_matCache[MatrixType::PreviousTranslatedWorldToView] = translatedWorldToView;
    }

    m_previousArtificalWorldOffset = Vector3(0.f);
  }

  void RtCamera::applyArtificialWorldOffset(const Vector3& worldOffset) {
    m_matCache[MatrixType::ViewToWorld][3].xyz() += Vector3d{ worldOffset };
    m_matCache[MatrixType::WorldToView] = inverse(m_matCache[MatrixType::ViewToWorld]);
    // Note: Translated world space matrices do not get offset here as they do not need a world offset, only
    // the current to previous frame translated world space offset needs to be updated, but this is currently
    // automatically derived from the view to world and previous view to world matrices, so no work here is needed.

    m_artificalWorldOffset += worldOffset;
  }

  Matrix4 getMatrixFromEulerAngles(float pitch, float yaw) {
    float cosPitch = cos(pitch);
    float sinPitch = sin(pitch);
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);
    Matrix4 customTransform;
    customTransform[0] = Vector4{ cosYaw, 0, -sinYaw, 0.0 };
    customTransform[1] = Vector4{ sinYaw * sinPitch, cosPitch, cosYaw * sinPitch, 0.0 };
    customTransform[2] = Vector4{ sinYaw * cosPitch, -sinPitch, cosPitch * cosYaw, 0.0 };
    return customTransform;
  }

  const RtCamera::RtCameraSetting& RtCamera::getSetting() {
    return m_context;
  }

  Matrix4d RtCamera::updateFreeCamera(uint32_t flags) {
    auto currTime = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsedSec = currTime - m_prevRunningTime;
    m_prevRunningTime = currTime;

    // Perform custom camera controls logic
    // NOTE: We track whether camera input occurred this frame so we don't overwrite
    // pending UI changes. Only call setDeferred() when there's actual movement input.

    float moveLeftRight = 0;
    float moveBackForward = 0;
    float moveDownUp = 0;
    float deltaYaw = 0;
    float deltaPitch = 0;
    bool hasMovementInput = false;
    bool hasRotationInput = false;
    bool resetCamera = false;

    // Read current values for use in matrix calculation
    float currentYaw = freeCameraYaw();
    float currentPitch = freeCameraPitch();
    Vector3 currentPosition = freeCameraPosition();

    HWND fgWin = GetForegroundWindow();
    DWORD processId = 0;
    if (fgWin) {
      GetWindowThreadProcessId(fgWin, &processId);
    }

    if ((flags & (int)RtCamera::UpdateFlag::UpdateFreeCamera) != 0) {
      float coordSystemScale = m_context.isLHS ? -1.f : 1.f;
      float pitchDirection = freeCameraInvertY() ? -1.f : 1.f;

      // Keyboard controls - works unless typing in a text field
      if (!ImGui::GetIO().WantTextInput) {
        bool isKeyAvailable =
          !ImGui::IsKeyDown(ImGuiKey_LeftCtrl) &&
          !ImGui::IsKeyDown(ImGuiKey_RightCtrl) &&
          !ImGui::IsKeyDown(ImGuiKey_LeftAlt) &&
          !ImGui::IsKeyDown(ImGuiKey_RightAlt) && !lockFreeCamera();

        if (isKeyAvailable) {
          float speed = elapsedSec.count() * RtxOptions::sceneScale() * freeCameraSpeed();
          float angularSpeed = elapsedSec.count() * M_PI * freeCameraTurningSpeed();
          // Speed booster
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveFaster(), true)) {
            speed *= 4;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveForward(), true)) {
            moveBackForward += coordSystemScale * speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveLeft(), true)) {
            moveLeftRight -= speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveBack(), true)) {
            moveBackForward -= coordSystemScale * speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveRight(), true)) {
            moveLeftRight += speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveUp(), true)) {
            moveDownUp += speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyMoveDown(), true)) {
            moveDownUp -= speed;
            hasMovementInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyPitchDown(), true)) {
            deltaPitch += coordSystemScale * pitchDirection * angularSpeed;
            hasRotationInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyPitchUp(), true)) {
            deltaPitch -= coordSystemScale * pitchDirection * angularSpeed;
            hasRotationInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyYawLeft(), true)) {
            deltaYaw += coordSystemScale * angularSpeed;
            hasRotationInput = true;
          }
          if (ImGUI::checkHotkeyState(RtxOptions::FreeCam::keyYawRight(), true)) {
            deltaYaw -= coordSystemScale * angularSpeed;
            hasRotationInput = true;
          }
        }
      }

      // Mouse controls - only when mouse is not over menu
      if (!ImGui::GetIO().WantCaptureMouse) {
        auto mouseDelta = ImGui::GetIO().MouseDelta;
        if (!lockFreeCamera() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          deltaYaw -= coordSystemScale * ImGui::GetIO().MouseDelta.x * 0.1f * elapsedSec.count();
          deltaPitch -= coordSystemScale * pitchDirection * ImGui::GetIO().MouseDelta.y * 0.2f * elapsedSec.count();
          if (mouseDelta.x != 0 || mouseDelta.y != 0) {
            hasRotationInput = true;
          }
        }

        // Reset
        if (!lockFreeCamera() && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
          resetCamera = true;
        }
      }
    }

    // Apply deltas to current values for matrix calculation
    currentYaw += deltaYaw;
    currentPitch += deltaPitch;

    // Calculate Free Camera matrix information

    dxvk::Matrix4d freeCamViewToWorld(m_matCache[MatrixType::ViewToWorld]);

    // Check if the up vector in view matrix is upside down
    const dxvk::Vector3d& up = m_matCache[MatrixType::ViewToWorld][1].xyz();
    bool isViewUpsideDown =
      (!RtxOptions::zUp() && up.y < 0.f) ||
      (RtxOptions::zUp() && up.z < 0.f);
    const float upSign = isViewUpsideDown ? -1.f : 1.f;

    freeCamViewToWorld[3] = Vector4d(0.0);
    freeCamViewToWorld *= getMatrixFromEulerAngles(upSign * currentPitch, currentYaw);

    if (m_type == CameraType::Main && (flags & (uint32_t)UpdateFlag::UpdateFreeCamera)) {
      currentPosition += moveLeftRight * Vector3(freeCamViewToWorld.data[0].xyz());
      currentPosition += upSign * moveDownUp * Vector3(freeCamViewToWorld.data[1].xyz());
      currentPosition -= moveBackForward * Vector3(freeCamViewToWorld.data[2].xyz());

      // Only update options when there's actual input - don't overwrite pending UI changes
      // Use the User edit target so these go to rtx.conf (these are UI-driven via camera controls)
      if (resetCamera) {
        RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);
        freeCameraPosition.setDeferred(Vector3(0.f));
        freeCameraYaw.setDeferred(0.0f);
        freeCameraPitch.setDeferred(0.0f);
        currentPosition = Vector3(0.f);
        currentYaw = 0.0f;
        currentPitch = 0.0f;
      } else {
        RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);
        if (hasMovementInput) {
          freeCameraPosition.setDeferred(currentPosition);
        }
        if (hasRotationInput) {
          freeCameraYaw.setDeferred(currentYaw);
          freeCameraPitch.setDeferred(currentPitch);
        }
      }
      
      // save free camera context using local values (since setDeferred doesn't update immediately)
      m_context.enableFreeCamera = enableFreeCamera();
      m_context.freeCameraPosition = currentPosition;
      m_context.freeCameraYaw = currentYaw;
      m_context.freeCameraPitch = currentPitch;
      m_context.freeCameraViewRelative = freeCameraViewRelative();
    }

    freeCamViewToWorld[3] = m_matCache[MatrixType::ViewToWorld][3] + Vector4d{ Vector3d{ currentPosition }, 0.0 };
    return freeCamViewToWorld;
  }

  bool RtCamera::shouldUseInfiniteFarFrustum() const {
    return !std::isfinite(m_context.farPlane)
        || RtxOptions::AntiCulling::Object::enableInfinityFarFrustum();
  }

  void RtCamera::updateAntiCulling(float fov, float aspectRatio, float nearPlane, float farPlane, bool isLHS) {
    // [InfFar.diag] Comprehensive entry log for the anti-culling frustum
    // setup. Wall-clock throttled per-camera-type so a low-FPS session
    // surfaces at least the first few hits for each camera type
    // (Main / ViewModel / Sky / etc.) without flooding when steady-state.
    {
      static thread_local uint64_t sN[8] = {0,0,0,0,0,0,0,0};
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
      const int idx = std::min<int>(static_cast<int>(m_type), 7);
      ++sN[idx];
      const auto now = std::chrono::steady_clock::now();
      const bool firstFew = sN[idx] <= 5;
      const bool dueByClock =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - sLastT[idx]).count() >= 500;
      if (firstFew || dueByClock) {
        sLastT[idx] = now;
        const bool farInf  = !std::isfinite(farPlane);
        const bool optOn   = RtxOptions::AntiCulling::Object::enableInfinityFarFrustum();
        const bool acEnabled = RtxOptions::AntiCulling::isObjectAntiCullingEnabled();
        Logger::warn(str::format(
          "[InfFar.diag.updateAntiCulling] camType=", static_cast<int>(m_type),
          " n=", sN[idx],
          " fov=", fov,
          " aspect=", aspectRatio,
          " near=", nearPlane,
          " far=", farPlane,
          " farIsInf=", (farInf ? 1 : 0),
          " isLHS=", (isLHS ? 1 : 0),
          " objectAntiCullingEnabled=", (acEnabled ? 1 : 0),
          " optEnableInfFar=", (optOn ? 1 : 0),
          " willUseInfFarBranch=", ((farInf || optOn) ? 1 : 0)));
      }
    }
    // Create Anti-Culling frustum
    if (RtxOptions::AntiCulling::isObjectAntiCullingEnabled()) {
      const float fovScale = RtxOptions::AntiCulling::Object::fovScale();
      const float farPlaneScale = RtxOptions::AntiCulling::Object::farPlaneScale();
      // NV-DXVK [AC frustum degeneracy guard]: SetupByHalfFovy[Inf] feeds
      // SetupByFrustum(-xmax, xmax, -ymax, ymax, ...) where
      //   ymax = nearPlane * tan(halfFovy)
      //   xmax = ymax * aspectRatio
      // SetupByFrustum asserts `left < right` — i.e. xmax > 0 and ymax > 0.
      // Reverse-Z TF2 cameras can present with nearPlane < 0 (Inf-far flag)
      // and some sub-camera draws (3D-skybox sub-cams reaching this path
      // for the first time after slot-based basis recovery) carry fov/aspect
      // values that collapse the frustum to zero/negative width. Probe the
      // exact values SetupByFrustum will receive and skip the AC setup
      // entirely when they're not strictly positive — anti-culling is an
      // optimization, the frame still renders correctly without it.
      const float halfFovyAC = (float)(fov * fovScale * 0.5);
      const float tanHalfAC = std::tan(halfFovyAC);
      const float ymaxProbeAC = nearPlane * tanHalfAC;
      const float xmaxProbeAC = ymaxProbeAC * aspectRatio;
      const bool acDegenerate =
        !std::isfinite(halfFovyAC) || !std::isfinite(aspectRatio) ||
        !std::isfinite(nearPlane)  || !std::isfinite(tanHalfAC)  ||
        !std::isfinite(ymaxProbeAC) || !std::isfinite(xmaxProbeAC) ||
        !(ymaxProbeAC > 0.0f) || !(xmaxProbeAC > 0.0f);
      if (acDegenerate) {
        static uint32_t sDegenACCount = 0;
        if (sDegenACCount < 30) {
          ++sDegenACCount;
          Logger::err(str::format(
            "[RtCamera::updateAntiCulling] DEGENERATE anti-culling frustum #", sDegenACCount,
            " camType=", static_cast<int>(m_type),
            " fov=", fov, " fovScale=", fovScale,
            " aspect=", aspectRatio, " near=", nearPlane, " far=", farPlane,
            " halfFovy=", halfFovyAC, " tan(halfFovy)=", tanHalfAC,
            " ymax=", ymaxProbeAC, " xmax=", xmaxProbeAC,
            " — skipping AC frustum setup for this update"));
        }
      } else {
        float4x4 frustumMatrix;
        // [InfFar] Use the infinite-far matrix whenever the camera input
        // itself is infinite-far (farPlane=+Inf, reverse-Z projection), in
        // addition to when the user has explicitly opted in via the
        // option. shouldUseInfiniteFarFrustum() encapsulates both cases.
        // Consumers (boundingBoxIntersectsFrustumSAT, etc.) must read the
        // same predicate so the frustum matrix and the consumer code path
        // agree on whether the far plane is finite.
        if (shouldUseInfiniteFarFrustum()) {
          frustumMatrix.SetupByHalfFovyInf(halfFovyAC, aspectRatio, nearPlane, (isLHS ? PROJ_LEFT_HANDED : 0));
        } else {
          frustumMatrix.SetupByHalfFovy(halfFovyAC, aspectRatio, nearPlane, farPlane * farPlaneScale, (isLHS ? PROJ_LEFT_HANDED : 0));
        }
        m_frustum.Setup(NDC_OGL, frustumMatrix);

        m_frustum.calculateFrustumGeometry(nearPlane, farPlane, fov, aspectRatio, isLHS);
      }
    }

    if (RtxOptions::AntiCulling::isLightAntiCullingEnabled()) {
      const float fovScale = RtxOptions::AntiCulling::Light::fovScale();
      const float scaledHalfFov = std::min(fov * fovScale * 0.5f, 1.55f); // Clamp to half fov to 89 degrees
      const float projectionMatrixFovScale = std::tan(fov * 0.5f) / std::tan(scaledHalfFov);
      Matrix4 viewToProjection = getViewToProjection();
      viewToProjection[0][0] *= projectionMatrixFovScale;
      viewToProjection[1][1] *= projectionMatrixFovScale;

      const Matrix4 worldToProj = viewToProjection * getWorldToView(false);
      m_lightAntiCullingFrustum.Setup(NDC_OGL, *reinterpret_cast<const float4x4*>(&worldToProj));
    }
  }

  Matrix4d RtCamera::overrideNearPlane(const Matrix4d& modifiedViewToProj) {
    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatModifiedViewToProj { modifiedViewToProj };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatModifiedViewToProj));

    uint32_t flags;
    float cameraParams[PROJ_NUM];
    DecomposeProjection(NDC_D3D, NDC_D3D, *reinterpret_cast<float4x4*>(&floatModifiedViewToProj), &flags, cameraParams, nullptr, nullptr, nullptr, nullptr);

    // NV-DXVK: Reverse-Z games (TF2, flags=PROJ_REVERSED_Z) can have negative
    // zNear and angles reported in swapped sign pairs. The raw angle ordering
    // is therefore not a reliable guard — what matters is whether the final
    // frustum bounds SetupByFrustum sees (xmin = tan(angleMinX) * zNear, etc.)
    // satisfy left < right. Compute those and gate on that. Only fall back to
    // the game's original projection for truly-broken matrices (NaN/Inf or
    // zero-width frustum).
    const float kMinNearOverride = std::min(RtxOptions::nearPlaneOverride(), cameraParams[PROJ_ZNEAR]);
    const float probeNear = kMinNearOverride; // SetupByAngles uses the value passed in, not PROJ_ZNEAR
    const float xminF = std::tan(cameraParams[PROJ_ANGLEMINX]) * probeNear;
    const float xmaxF = std::tan(cameraParams[PROJ_ANGLEMAXX]) * probeNear;
    const float yminF = std::tan(cameraParams[PROJ_ANGLEMINY]) * probeNear;
    const float ymaxF = std::tan(cameraParams[PROJ_ANGLEMAXY]) * probeNear;
    const bool degenerate =
      !(xminF < xmaxF) ||
      !(yminF < ymaxF) ||
      !std::isfinite(xminF) || !std::isfinite(xmaxF) ||
      !std::isfinite(yminF) || !std::isfinite(ymaxF) ||
      !std::isfinite(cameraParams[PROJ_ZNEAR]) ||
      !std::isfinite(cameraParams[PROJ_ZFAR]);
    if (degenerate) {
      static uint32_t sDegenNearCount = 0;
      if (sDegenNearCount < 20) {
        ++sDegenNearCount;
        const Matrix4& m = floatModifiedViewToProj;
        Logger::err(str::format(
          "[RtCamera::overrideNearPlane] DEGENERATE projection #", sDegenNearCount,
          " flags=0x", std::hex, flags, std::dec,
          " angles=(minX=", cameraParams[PROJ_ANGLEMINX], ", maxX=", cameraParams[PROJ_ANGLEMAXX],
          ", minY=", cameraParams[PROJ_ANGLEMINY], ", maxY=", cameraParams[PROJ_ANGLEMAXY], ")",
          " zNear=", cameraParams[PROJ_ZNEAR],
          " zFar=", cameraParams[PROJ_ZFAR],
          " matrix=[",
            "(", m[0][0], ",", m[0][1], ",", m[0][2], ",", m[0][3], ") ",
            "(", m[1][0], ",", m[1][1], ",", m[1][2], ",", m[1][3], ") ",
            "(", m[2][0], ",", m[2][1], ",", m[2][2], ",", m[2][3], ") ",
            "(", m[3][0], ",", m[3][1], ",", m[3][2], ",", m[3][3], ")]"));
      }
      return modifiedViewToProj;
    }

    // Prevent user controls exceeding the near plane distance from original projection
    // (kMinNearOverride was already computed above for the degeneracy probe.)
    const float minNearPlane = kMinNearOverride;

    float4x4 newProjection;
    newProjection.SetupByAngles(cameraParams[PROJ_ANGLEMINX], cameraParams[PROJ_ANGLEMAXX], cameraParams[PROJ_ANGLEMINY], cameraParams[PROJ_ANGLEMAXY], minNearPlane, cameraParams[PROJ_ZFAR], flags);
    memcpy(&floatModifiedViewToProj, &newProjection, sizeof(float4x4));

    return Matrix4d { floatModifiedViewToProj };
  }


  bool RtCamera::updateFromSetting(uint32_t frameIdx, const RtCameraSetting& setting, uint32_t flags) {
    // Use the User edit target so these go to rtx.conf (these are UI-driven via camera settings)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);
    
    enableFreeCamera.setDeferred(setting.enableFreeCamera);
    freeCameraPosition.setDeferred(setting.freeCameraPosition);
    freeCameraYaw.setDeferred(setting.freeCameraYaw);
    freeCameraPitch.setDeferred(setting.freeCameraPitch);
    freeCameraViewRelative.setDeferred(setting.freeCameraViewRelative);

    RtxOptions::shakeCamera.setDeferred(setting.isCameraShaking);

    m_context = setting;

    return update(frameIdx, setting.worldToView, setting.viewToProjection, setting.fov, setting.aspectRatio, setting.nearPlane, setting.farPlane, setting.isLHS, flags);
  }

  bool RtCamera::update(
    uint32_t frameIdx, const Matrix4& newWorldToView, const Matrix4& newViewToProjection,
    float fov, float aspectRatio, float nearPlane, float farPlane, bool isLHS, uint32_t flags) {
    if (m_frameLastTouched == frameIdx) {
      return false;
    }

    // [NaNGuard] RtCamera::update is the SINGLE entry point that mutates
    // m_context.{worldToView,viewToProjection}. Every downstream consumer
    // (getVolumeShaderConstants → SetupByAngles + MvpToPlanes,
    // getNearAndFar → DecomposeProjection, overrideNearPlane →
    // DecomposeProjection + SetupByAngles) reads from those member matrices
    // and feeds them directly into MathLib functions whose Sqrt template
    // (MathLib.h:1918) asserts on NaN input (`x >= 0` fails for NaN).
    //
    // IMPORTANT: reject NaN ONLY, NOT Inf. TF2 (and any modern engine using
    // reverse-Z infinite-far projection) legitimately produces far=Inf with
    // m[2][3]=-1, m[3][2]=-1 as its standard camera matrix. The previous
    // version of this gate used !isfinite (which catches both NaN and Inf)
    // and over-rejected every frame's update → camera never advanced →
    // visible "geo pops in after a while" as the stale camera fails to
    // cull the new viewpoint. Sqrt(Inf) returns Inf and Inf >= 0 is true,
    // so Inf in the input does NOT trip the assert — only NaN does.
    auto hasNaN4x4 = [](const Matrix4& m) -> bool {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          if (std::isnan(m[r][c])) return true;
      return false;
    };
    const bool w2vBad = hasNaN4x4(newWorldToView);
    const bool v2pBad = hasNaN4x4(newViewToProjection);
    const bool scalarsBad = std::isnan(fov) || std::isnan(aspectRatio)
                          || std::isnan(nearPlane) || std::isnan(farPlane);
    if (w2vBad || v2pBad || scalarsBad) {
      const bool w2vFinite = !w2vBad;
      const bool v2pFinite = !v2pBad;
      // Wall-clock throttled log; first 10 hits unconditionally so even a
      // session that crashes immediately surfaces at least one entry.
      static uint64_t sNaNN = 0;
      static std::chrono::steady_clock::time_point sLastLogT
        = std::chrono::steady_clock::time_point::min();
      ++sNaNN;
      const auto now = std::chrono::steady_clock::now();
      const bool firstFew = sNaNN <= 10;
      const bool dueByClock =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          now - sLastLogT).count() >= 500;
      if (firstFew || dueByClock) {
        sLastLogT = now;
        Logger::warn(str::format(
          "[RtCamera::update.NaNGuard] n=", sNaNN,
          " frameIdx=", frameIdx,
          " w2vFinite=", (w2vFinite ? 1 : 0),
          " v2pFinite=", (v2pFinite ? 1 : 0),
          " fov=", fov,
          " aspect=", aspectRatio,
          " near=", nearPlane,
          " far=", farPlane,
          " w2v.m0=(", newWorldToView[0][0], ",", newWorldToView[0][1], ",", newWorldToView[0][2], ",", newWorldToView[0][3], ")",
          " w2v.m3=(", newWorldToView[3][0], ",", newWorldToView[3][1], ",", newWorldToView[3][2], ",", newWorldToView[3][3], ")",
          " v2p.m0=(", newViewToProjection[0][0], ",", newViewToProjection[0][1], ",", newViewToProjection[0][2], ",", newViewToProjection[0][3], ")",
          " v2p.m2=(", newViewToProjection[2][0], ",", newViewToProjection[2][1], ",", newViewToProjection[2][2], ",", newViewToProjection[2][3], ")"));
      }
      // Reject the update entirely. Returning false signals "no change" to
      // caller (CameraManager::processCameraData line ~867), which is the
      // safe outcome: keep last frame's known-good camera, skip this draw's
      // contribution. Frame counter advances so this frame's update isn't
      // retried, matching the early-return path above for same-frame double-update.
      m_frameLastTouched = frameIdx;
      return false;
    }

    m_context.worldToView = newWorldToView;
    m_context.viewToProjection = newViewToProjection;
    m_context.fov = fov;
    m_context.aspectRatio = aspectRatio;
    m_context.nearPlane = nearPlane;
    m_context.farPlane = farPlane;
    m_context.isLHS = isLHS;
    m_context.flags = flags;

    m_previousArtificalWorldOffset = m_artificalWorldOffset;
    m_artificalWorldOffset = Vector3(0.f);

    if (!RtxOptions::shakeCamera()) {
      m_context.cameraShakeFrameCount = 0;
      m_context.cameraRotationFrameCount = 0;
    }

    // Setup World/View Matrix Data

    m_matCache[MatrixType::PreviousPreviousWorldToView] = m_matCache[MatrixType::PreviousWorldToView];
    m_matCache[MatrixType::PreviousPreviousViewToWorld] = m_matCache[MatrixType::PreviousViewToWorld];
    m_matCache[MatrixType::PreviousWorldToView] = m_matCache[MatrixType::WorldToView];
    m_matCache[MatrixType::PreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];
    m_matCache[MatrixType::UncorrectedPreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];
    m_matCache[MatrixType::WorldToView] = freeCameraViewRelative() ? Matrix4d{ newWorldToView } : Matrix4d();
    m_matCache[MatrixType::ViewToWorld] = inverse(m_matCache[MatrixType::WorldToView]);

    // Setup Translated World/View Matrix Data

    m_matCache[MatrixType::PreviousTranslatedWorldToView] = m_matCache[MatrixType::TranslatedWorldToView];
    m_matCache[MatrixType::PreviousViewToTranslatedWorld] = m_matCache[MatrixType::ViewToTranslatedWorld];
    m_matCache[MatrixType::UncorrectedPreviousTranslatedWorldToView] = m_matCache[MatrixType::TranslatedWorldToView];

    auto viewToTranslatedWorld = m_matCache[MatrixType::ViewToWorld];
    viewToTranslatedWorld[3] = Vector4d(0.0f, 0.0f, 0.0f, viewToTranslatedWorld[3].w);

    m_matCache[MatrixType::ViewToTranslatedWorld] = freeCameraViewRelative() ? viewToTranslatedWorld : Matrix4d();
    // Note: Slightly non-ideal to have to inverse an already inverted matrix when we have the original world to view matrix,
    // but this is the safest way to ensure a proper inversion when modifying the view to world transform manually.
    m_matCache[MatrixType::TranslatedWorldToView] = inverse(m_matCache[MatrixType::ViewToTranslatedWorld]);

    // Setup View/Projection Matrix Data

    m_matCache[MatrixType::PreviousViewToProjection] = m_matCache[MatrixType::ViewToProjection];
    m_matCache[MatrixType::PreviousProjectionToView] = m_matCache[MatrixType::ProjectionToView];

    auto modifiedViewToProj = Matrix4d{ newViewToProjection };

    // Correct for games that use a negative Y scale in the projection matrix
    // (e.g., certain Unity titles) which causes the ray-traced scene to render upside-down.
    if (correctProjectionYFlip()) {
      modifiedViewToProj[1][1] = -modifiedViewToProj[1][1];
    }

    updateAntiCulling(fov, aspectRatio, nearPlane, farPlane, isLHS);

    // Sometimes we want to modify the near plane for RT.  See DevSettings->Camera->Advanced
    if(RtxOptions::enableNearPlaneOverride()) {
      modifiedViewToProj = overrideNearPlane(modifiedViewToProj);
    }
    
    m_matCache[MatrixType::ViewToProjection] = modifiedViewToProj;
    m_matCache[MatrixType::ProjectionToView] = inverse(modifiedViewToProj);

    // Apply free camera shaking

    if (!enableFreeCamera() && RtxOptions::shakeCamera()) {
      auto newViewToWorld = getShakenViewToWorldMatrix(m_matCache[MatrixType::ViewToWorld], flags);
      auto newViewToTranslatedWorld = newViewToWorld;
      newViewToTranslatedWorld[3] = Vector4d(0.0, 0.0, 0.0, newViewToTranslatedWorld[3].w);

      // Note: Error added here from an extra inverse operation, but should be fine as camera shaking is only used as a debugging path.
      m_matCache[MatrixType::WorldToView] = inverse(newViewToWorld);
      m_matCache[MatrixType::ViewToWorld] = newViewToWorld;
      m_matCache[MatrixType::TranslatedWorldToView] = inverse(newViewToTranslatedWorld);
      m_matCache[MatrixType::ViewToTranslatedWorld] = newViewToTranslatedWorld;
    }

    // Apply jittering

    if (flags & (uint32_t) UpdateFlag::UpdateJitterFrame) {
      m_context.jitterFrameIdx = frameIdx;
    }

    auto newViewToProjectionJittered = modifiedViewToProj;
    applyAndGetJitter(
      newViewToProjectionJittered,
      m_jitter,
      m_context.jitterFrameIdx);

    m_context.jitter[0] = m_jitter[0];
    m_context.jitter[1] = m_jitter[1];
    m_context.isCameraShaking = RtxOptions::shakeCamera();

    m_matCache[MatrixType::PreviousViewToProjectionJittered] = m_matCache[MatrixType::ViewToProjectionJittered];
    m_matCache[MatrixType::PreviousProjectionToViewJittered] = m_matCache[MatrixType::ProjectionToViewJittered];
    m_matCache[MatrixType::ViewToProjectionJittered] = newViewToProjectionJittered;
    m_matCache[MatrixType::ProjectionToViewJittered] = inverse(newViewToProjectionJittered);

    m_frameLastTouched = frameIdx;

    // [rtcam-pos-trace]: log the actual camera position that gets baked into
    // m_matCache after this update. This is what shaders see for ray
    // origins / motion vectors / temporal accumulation. Throttled to first
    // 200 entries; logs only on Main camera updates so we can correlate
    // user-visible oscillation to actual viewToWorld translation changes.
    if (m_type == CameraType::Main) {
      static uint32_t sRtCamPosLog = 0;
      static Vector3 sLastPos{1e30f, 1e30f, 1e30f};
      const auto& vtw = m_matCache[MatrixType::ViewToWorld];
      const Vector3 pos{
        float(vtw[3][0]), float(vtw[3][1]), float(vtw[3][2])};
      const Vector3 delta{
        pos.x - sLastPos.x, pos.y - sLastPos.y, pos.z - sLastPos.z};
      const float deltaMag = std::sqrt(
        delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
      // Log if pos changed by > 0.05u (catches both sub-unit bob and
      // larger movements). First-call log fires unconditionally.
      if (sLastPos.x > 1e29f || deltaMag > 0.05f) {
        if (sRtCamPosLog < 200) {
          ++sRtCamPosLog;
          Logger::info(str::format(
            "[rtcam-pos-trace] frame=", frameIdx,
            " pos=(", pos.x, ",", pos.y, ",", pos.z, ")",
            " delta=(", delta.x, ",", delta.y, ",", delta.z, ")",
            " |delta|=", deltaMag));
        }
        sLastPos = pos;
      }
    }

    // NV-DXVK [MtnCam] one-directional skybox streak hunt. The 3D-skybox reproject
    // pins the mountains to a CONSTANT world anchor (T.t = -scale*anchor, invariant to
    // ship motion — see d3d11_rtx.cpp:18975), so [MtnMotion] shows rep=0/pml=0: the
    // geometry is stationary in world space and the ENTIRE screen motion vector comes
    // from this (Main) camera's prev-vs-cur matrices. On a moving ship the camera
    // translates every frame; the MV is correct ONLY if prev lags cur by exactly one
    // consistent ship-step. This logs, per frame (with frameIdx to line up against the
    // [MtnRadiance] ssmv and [MtnMotion] probes):
    //   curO            cur camera world origin (ViewToWorld[3])
    //   d1 = cur - prev          this frame's ship-step (the translation the MV encodes)
    //   d2 = prev - prevprev     last frame's ship-step
    //   stepRatio = |d1|/|d2|    ~1 on steady motion; oscillating/!=1 = prev/cur PHASE fault (double-step then zero-step => streak)
    //   fwdDot          dot(curForward, prevForward); <1 = rotation this frame
    // DECIDER: on a steadily moving ship, if d1 and d2 alternate (stepRatio swings ~2 then ~0)
    // the engine-hook is delivering camera origins out of phase with the frame cadence ->
    // systematically wrong MV in the ship's travel direction = the one-way streak.
    if (m_type == CameraType::Main && RtxOptions::logSurfaceCoverage() && !m_firstUpdate) {
      const auto& vtw  = m_matCache[MatrixType::ViewToWorld];
      const auto& pvtw = m_matCache[MatrixType::PreviousViewToWorld];
      const auto& ppvtw = m_matCache[MatrixType::PreviousPreviousViewToWorld];
      const Vector3 cO { float(vtw[3][0]),  float(vtw[3][1]),  float(vtw[3][2]) };
      const Vector3 pO { float(pvtw[3][0]), float(pvtw[3][1]), float(pvtw[3][2]) };
      const Vector3 ppO{ float(ppvtw[3][0]),float(ppvtw[3][1]),float(ppvtw[3][2]) };
      const Vector3 d1 { cO.x-pO.x, cO.y-pO.y, cO.z-pO.z };
      const Vector3 d2 { pO.x-ppO.x, pO.y-ppO.y, pO.z-ppO.z };
      const float m1 = std::sqrt(d1.x*d1.x + d1.y*d1.y + d1.z*d1.z);
      const float m2 = std::sqrt(d2.x*d2.x + d2.y*d2.y + d2.z*d2.z);
      const float stepRatio = (m2 > 1e-6f) ? (m1 / m2) : -1.0f;
      // camera forward = ViewToWorld * (0,0,1) view-space Z -> column 2
      const Vector3 cFwd { float(vtw[2][0]),  float(vtw[2][1]),  float(vtw[2][2]) };
      const Vector3 pFwd { float(pvtw[2][0]), float(pvtw[2][1]), float(pvtw[2][2]) };
      const float fwdDot = cFwd.x*pFwd.x + cFwd.y*pFwd.y + cFwd.z*pFwd.z;
      Logger::info(str::format(
        "[MtnCam] f=", frameIdx,
        " curO=(", cO.x, ",", cO.y, ",", cO.z, ")",
        " d1=(", d1.x, ",", d1.y, ",", d1.z, ") |d1|=", m1,
        " d2=(", d2.x, ",", d2.y, ",", d2.z, ") |d2|=", m2,
        " stepRatio=", stepRatio,
        " fwdDot=", fwdDot));
    }

    // For our first update, we should init both previous and current to the same value
    if (m_firstUpdate) {
      m_matCache[MatrixType::PreviousWorldToView] = m_matCache[MatrixType::WorldToView];
      m_matCache[MatrixType::PreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];
      m_matCache[MatrixType::PreviousPreviousWorldToView] = m_matCache[MatrixType::WorldToView];
      m_matCache[MatrixType::PreviousPreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];

      m_matCache[MatrixType::PreviousTranslatedWorldToView] = m_matCache[MatrixType::TranslatedWorldToView];
      m_matCache[MatrixType::PreviousViewToTranslatedWorld] = m_matCache[MatrixType::ViewToTranslatedWorld];

      m_matCache[MatrixType::PreviousViewToProjection] = m_matCache[MatrixType::ViewToProjection];
      m_matCache[MatrixType::PreviousProjectionToView] = m_matCache[MatrixType::ProjectionToView];

      m_matCache[MatrixType::PreviousViewToProjectionJittered] = m_matCache[MatrixType::ViewToProjectionJittered];
      m_matCache[MatrixType::PreviousProjectionToViewJittered] = m_matCache[MatrixType::ProjectionToViewJittered];

      // Never do this again
      m_firstUpdate = false;
    }

    // Only calculate free camera matrices for main camera
    if (!enableFreeCamera() || m_type != CameraType::Main) {
      return isCameraCut();
    }

    auto freeCamViewToWorld = updateFreeCamera(flags);

    if (RtxOptions::shakeCamera()) {
      freeCamViewToWorld = getShakenViewToWorldMatrix(freeCamViewToWorld, flags);
    }

    auto freeCamViewToTranslatedWorld = freeCamViewToWorld;
    freeCamViewToTranslatedWorld[3] = Vector4d(0.0f, 0.0f, 0.0f, freeCamViewToTranslatedWorld[3].w);

    m_matCache[MatrixType::FreeCamPreviousPreviousWorldToView] = m_matCache[MatrixType::FreeCamPreviousWorldToView];
    m_matCache[MatrixType::FreeCamPreviousPreviousViewToWorld] = m_matCache[MatrixType::FreeCamPreviousViewToWorld];
    m_matCache[MatrixType::FreeCamPreviousWorldToView] = m_matCache[MatrixType::FreeCamWorldToView];
    m_matCache[MatrixType::FreeCamPreviousViewToWorld] = m_matCache[MatrixType::FreeCamViewToWorld];
    m_matCache[MatrixType::FreeCamWorldToView] = inverse(freeCamViewToWorld);
    m_matCache[MatrixType::FreeCamViewToWorld] = freeCamViewToWorld;

    m_matCache[MatrixType::FreeCamPreviousTranslatedWorldToView] = m_matCache[MatrixType::FreeCamTranslatedWorldToView];
    m_matCache[MatrixType::FreeCamPreviousViewToTranslatedWorld] = m_matCache[MatrixType::FreeCamViewToTranslatedWorld];
    m_matCache[MatrixType::FreeCamTranslatedWorldToView] = inverse(freeCamViewToTranslatedWorld);
    m_matCache[MatrixType::FreeCamViewToTranslatedWorld] = freeCamViewToTranslatedWorld;

    m_matCache[MatrixType::ViewToWorldToFreeCamViewToWorld] = m_matCache[MatrixType::WorldToView] * m_matCache[MatrixType::FreeCamViewToWorld];

    return false; // If we are using the debug/free camera, never do camera cuts
  }

  void RtCamera::getJittering(float jitter[2]) const {
    jitter[0] = m_jitter[0];
    jitter[1] = m_jitter[1];
  }
  
  Vector2 RtCamera::calcPixelJitter(uint32_t jitterFrameIdx) const {
    // Only apply jittering when DLSS/XeSS/TAA is enabled, or if forced by settings
    if (!RtxOptions::isDLSSOrRayReconstructionEnabled() &&
        !RtxOptions::isXeSSEnabled() &&
        !RtxOptions::isTAAEnabled() &&
        !RtxOptions::forceCameraJitter()) {
      return Vector2{ 0, 0 };
    }

#define USE_DLSS_DEMO_JITTER_PATTERN 1
#if USE_DLSS_DEMO_JITTER_PATTERN
    uint32_t jitterSequenceLength = RtxOptions::cameraJitterSequenceLength();

    if (RtxOptions::isXeSSEnabled() && DxvkXeSS::XessOptions::useRecommendedJitterSequenceLength()) {
      float upscaleFactor = static_cast<float>(m_finalResolution[1]) / m_renderResolution[1];

      // XeSS 2.1 formula: ceil(upscale_factor^2 * 8)
      // The 8.0 multiplier ensures sufficient temporal samples for higher upscaling factors
      uint32_t xessLength = static_cast<uint32_t>(std::ceil(upscaleFactor * upscaleFactor * 8.0f));
      
      // Apply minimum jitter sequence length
      uint32_t minLength = DxvkXeSS::XessOptions::minJitterSequenceLength();
      xessLength = std::max(xessLength, minLength);
      
      jitterSequenceLength = xessLength;
    }
    
    return calculateHaltonJitter(jitterFrameIdx, jitterSequenceLength);
#else
    return m_halton.next();
#endif
  }

  Vector2 RtCamera::calcClipSpaceJitter(
    const Vector2& pixelJitter, 
    float ratioX,
    float ratioY) const {
    if (m_renderResolution[0] == 0 || m_renderResolution[1] == 0) {
      return Vector2{ 0, 0 };
    }
    return Vector2{
      pixelJitter[0] / static_cast<float>(m_renderResolution[0]) * ratioX * 2.f,
      pixelJitter[1] / static_cast<float>(m_renderResolution[1]) * ratioY * 2.f,
    };
  }

  void RtCamera::applyJitterTo(
    Matrix4& inoutProjection,
    uint32_t jitterFrameIdx) const {
    Vector2 pixelJitter = calcPixelJitter(jitterFrameIdx);
    float ratioX = Sign(inoutProjection[2][3]);
    float ratioY = -Sign(inoutProjection[2][3]);
    Vector2 clipSpaceJitter = calcClipSpaceJitter(pixelJitter,ratioX, ratioY);
    if (std::abs(clipSpaceJitter[0]) < std::numeric_limits<float>::min() &&
        std::abs(clipSpaceJitter[1]) < std::numeric_limits<float>::min()) {
      return;
    }
    inoutProjection[2][0] += clipSpaceJitter[0];
    inoutProjection[2][1] += clipSpaceJitter[1];
  }

  void RtCamera::applyAndGetJitter(
    Matrix4d& inoutProjection, 
    float (&outPixelJitter)[2], 
    uint32_t jitterFrameIdx) const {
    Vector2 pixelJitter = calcPixelJitter(jitterFrameIdx);
    float ratioX = Sign(inoutProjection[2][3]);
    float ratioY = -Sign(inoutProjection[2][3]);
    Vector2 clipSpaceJitter = calcClipSpaceJitter(pixelJitter, ratioX, ratioY);
    {
      outPixelJitter[0] = pixelJitter[0];
      outPixelJitter[1] = pixelJitter[1];
    }
    if (std::abs(clipSpaceJitter[0]) < std::numeric_limits<float>::min() &&
        std::abs(clipSpaceJitter[1]) < std::numeric_limits<float>::min()) {
      return;
    }
    inoutProjection[2][0] += clipSpaceJitter[0];
    inoutProjection[2][1] += clipSpaceJitter[1];
  }

  Camera RtCamera::getShaderConstants(bool freecam) const {
    auto& worldToView = getWorldToView(freecam);
    auto& translatedWorldToView = getTranslatedWorldToView(freecam);
    auto& viewToWorld = getViewToWorld(freecam);
    auto& viewToTranslatedWorld = getViewToTranslatedWorld(freecam);
    auto& viewToProjection = getViewToProjection();
    auto& projectionToView = getProjectionToView();
    auto& prevWorldToView = getPreviousWorldToView(freecam);
    auto& prevTranslatedWorldToView = getPreviousTranslatedWorldToView(freecam);
    auto& prevViewToWorld = getPreviousViewToWorld(freecam);
    auto& viewToProjectionJittered = getViewToProjectionJittered();
    auto& projectionToViewJittered = getProjectionToViewJittered();
    auto& prevViewToProjection = getPreviousViewToProjection();
    auto& prevViewToProjectionJittered = getPreviousViewToProjectionJittered();
    auto& prevProjectionToView = getPreviousProjectionToView();
    auto& prevProjectionToViewJittered = getPreviousProjectionToViewJittered();
    auto viewToPrevView = prevWorldToView * viewToWorld;

    Camera camera;
    camera.worldToView = worldToView;
    camera.viewToWorld = viewToWorld;
    camera.viewToProjection = viewToProjection;
    camera.projectionToView = projectionToView;
    camera.viewToProjectionJittered = viewToProjectionJittered;
    camera.projectionToViewJittered = projectionToViewJittered;
    camera.worldToProjectionJittered = viewToProjectionJittered * worldToView;
    camera.projectionToWorldJittered = viewToWorld * projectionToViewJittered;
    camera.translatedWorldToView = translatedWorldToView;
    camera.translatedWorldToProjectionJittered = viewToProjectionJittered * translatedWorldToView;
    camera.projectionToTranslatedWorld = viewToTranslatedWorld * projectionToView;

    camera.prevWorldToView = prevWorldToView;
    camera.prevViewToWorld = prevViewToWorld;
    camera.prevProjectionToView = prevProjectionToView;
    camera.prevProjectionToViewJittered = prevProjectionToViewJittered;
    camera.prevWorldToProjection = prevViewToProjection * prevWorldToView;
    camera.prevWorldToProjectionJittered = prevViewToProjectionJittered * prevWorldToView;
    camera.prevTranslatedWorldToView = prevTranslatedWorldToView;
    camera.prevTranslatedWorldToProjection = prevViewToProjection * prevTranslatedWorldToView;

    camera.projectionToPrevProjectionJittered = prevViewToProjectionJittered * viewToPrevView * projectionToViewJittered;
    
    camera.resolution = uvec2 { m_renderResolution[0], m_renderResolution[1] };
    camera.nearPlane = m_context.nearPlane;

    camera.flags = ((!m_context.isLHS) ? rightHandedFlag : 0);

    return camera;
  }

  VolumeDefinitionCamera RtCamera::getVolumeShaderConstants(const float maxDistance, const float guardBand/* = 1.f*/) const {
    auto& translatedWorldToView = getTranslatedWorldToView();
    auto& viewToTranslatedWorld = getViewToTranslatedWorld();
    auto& viewToWorld = getViewToWorld();

    auto viewToProjection = getViewToProjection();
    viewToProjection[0][0] /= guardBand;
    viewToProjection[1][1] /= guardBand;

    auto viewToProjectionJittered = getViewToProjectionJittered();
    viewToProjectionJittered[0][0] /= guardBand;
    viewToProjectionJittered[1][1] /= guardBand;

    auto prevViewToProjection = getPreviousViewToProjection();
    prevViewToProjection[0][0] /= guardBand;
    prevViewToProjection[1][1] /= guardBand;

    auto projectionToView = inverse(viewToProjection);

    bool isFreeCamera = isFreeCameraEnabled();
    auto& prevViewToWorld = isFreeCamera ? m_matCache[FreeCamPreviousViewToWorld] : m_matCache[UncorrectedPreviousViewToWorld];
    auto& prevTranslatedWorldToView = isFreeCamera ? m_matCache[FreeCamPreviousTranslatedWorldToView] : m_matCache[UncorrectedPreviousTranslatedWorldToView];
    
    VolumeDefinitionCamera camera;

    camera.viewToProjection = viewToProjection;
    camera.translatedWorldToView = translatedWorldToView;
    camera.translatedWorldToProjectionJittered = viewToProjectionJittered * translatedWorldToView;
    camera.projectionToTranslatedWorld = viewToTranslatedWorld * projectionToView;
    camera.prevTranslatedWorldToView = prevTranslatedWorldToView;
    camera.prevTranslatedWorldToProjection = prevViewToProjection * prevTranslatedWorldToView;
    
    camera.translatedWorldOffset = Vector3{ viewToWorld[3].xyz() };
    camera.previousTranslatedWorldOffset = Vector3{ prevViewToWorld[3].xyz() };
    camera.nearPlane = m_context.nearPlane;
    camera.flags = ((!m_context.isLHS) ? rightHandedFlag : 0);

    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatModifiedViewToProj { viewToProjection };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatModifiedViewToProj));

    uint32_t flags;
    float cameraParams[PROJ_NUM];
    DecomposeProjection(NDC_D3D, NDC_D3D, *reinterpret_cast<float4x4*>(&floatModifiedViewToProj), &flags, cameraParams, nullptr, nullptr, nullptr, nullptr);

    // NV-DXVK: Same degeneracy guard as overrideNearPlane. Evaluate what
    // SetupByFrustum will actually see (xmin/xmax after tan*zNear with the
    // reverse-Z sign flip) rather than raw angle ordering, so we don't bail
    // out for valid reverse-Z matrices where min/max are symmetric and zNear
    // is negative.
    const float probeNearVol = cameraParams[PROJ_ZNEAR];
    const float xminFV = std::tan(cameraParams[PROJ_ANGLEMINX]) * probeNearVol;
    const float xmaxFV = std::tan(cameraParams[PROJ_ANGLEMAXX]) * probeNearVol;
    const float yminFV = std::tan(cameraParams[PROJ_ANGLEMINY]) * probeNearVol;
    const float ymaxFV = std::tan(cameraParams[PROJ_ANGLEMAXY]) * probeNearVol;
    const bool degenerateVol =
      !(xminFV < xmaxFV) ||
      !(yminFV < ymaxFV) ||
      !std::isfinite(xminFV) || !std::isfinite(xmaxFV) ||
      !std::isfinite(yminFV) || !std::isfinite(ymaxFV) ||
      !std::isfinite(cameraParams[PROJ_ZNEAR]) ||
      !std::isfinite(cameraParams[PROJ_ZFAR]);
    if (degenerateVol) {
      static uint32_t sDegenVolCount = 0;
      if (sDegenVolCount < 20) {
        ++sDegenVolCount;
        const Matrix4& m = floatModifiedViewToProj;
        Logger::err(str::format(
          "[RtCamera::getVolumeDefinitionCamera] DEGENERATE projection #", sDegenVolCount,
          " flags=0x", std::hex, flags, std::dec,
          " angles=(minX=", cameraParams[PROJ_ANGLEMINX], ", maxX=", cameraParams[PROJ_ANGLEMAXX],
          ", minY=", cameraParams[PROJ_ANGLEMINY], ", maxY=", cameraParams[PROJ_ANGLEMAXY], ")",
          " zNear=", cameraParams[PROJ_ZNEAR],
          " zFar=", cameraParams[PROJ_ZFAR],
          " matrix=[",
            "(", m[0][0], ",", m[0][1], ",", m[0][2], ",", m[0][3], ") ",
            "(", m[1][0], ",", m[1][1], ",", m[1][2], ",", m[1][3], ") ",
            "(", m[2][0], ",", m[2][1], ",", m[2][2], ",", m[2][3], ") ",
            "(", m[3][0], ",", m[3][1], ",", m[3][2], ",", m[3][3], ")]"));
      }
    }

    // Prevent user controls exceeding the far plane distance from original projection
    const float minFarPlane = std::min(maxDistance, cameraParams[PROJ_ZFAR]);

    float4x4 newProjection;
    if (degenerateVol) {
      // Use the original projection unchanged for camera.viewToProjection below.
      memcpy(&newProjection, &floatModifiedViewToProj, sizeof(float4x4));
    } else {
      newProjection.SetupByAngles(cameraParams[PROJ_ANGLEMINX], cameraParams[PROJ_ANGLEMAXX], cameraParams[PROJ_ANGLEMINY], cameraParams[PROJ_ANGLEMAXY], cameraParams[PROJ_ZNEAR], minFarPlane, flags);
    }

    memcpy(&floatModifiedViewToProj, &newProjection, sizeof(float4x4));

    Matrix4 worldToProjection = floatModifiedViewToProj * translatedWorldToView;

    MvpToPlanes(NDC_D3D, *reinterpret_cast<const float4x4*>(&worldToProjection), &camera.worldPlanes[0]);

    return camera;
  }

  void RtCamera::showImguiSettings() {
    const static ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    if (RemixGui::CollapsingHeader("Free Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();

      RemixGui::Checkbox("Enable Free Camera", &enableFreeCameraObject());
      RemixGui::Checkbox("Lock Free Camera", &lockFreeCameraObject());
      RemixGui::Checkbox("Use Free Camera for Components", &useFreeCameraForComponentsObject());
      RemixGui::DragFloat3("Position", &freeCameraPositionObject(), 0.1f, -1e5, -1e5, "%.3f", sliderFlags);
      RemixGui::DragFloat("Yaw", &freeCameraYawObject(), 0.1f, -Pi<float>(2), Pi<float>(2), "%.3f", sliderFlags);
      RemixGui::DragFloat("Pitch", &freeCameraPitchObject(), 0.1f, -Pi<float>(2), Pi<float>(2), "%.3f", sliderFlags);
      RemixGui::DragFloat("Speed", &freeCameraSpeedObject(), 0.1f, 0.f, 5000.0f, "%.3f");
      RemixGui::DragFloat("Turning Speed", &freeCameraTurningSpeedObject(), 0.01f, 0.f, 3.0f, "%.3f");
      RemixGui::Checkbox("Invert Y", &freeCameraInvertYObject());
      RemixGui::Checkbox("View Relative", &freeCameraViewRelativeObject());

      if (RemixGui::CollapsingHeader("Show Camera Controls")) {
        ImGui::TextUnformatted("MoveFaster:");  ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveFaster()).c_str());
        ImGui::TextUnformatted("MoveForward:"); ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveForward()).c_str());
        ImGui::TextUnformatted("MoveLeft:");    ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveLeft()).c_str());
        ImGui::TextUnformatted("MoveBack:");    ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveBack()).c_str());
        ImGui::TextUnformatted("MoveRight:");   ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveRight()).c_str());
        ImGui::TextUnformatted("MoveUp:");      ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveUp()).c_str());
        ImGui::TextUnformatted("MoveDown:");    ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyMoveDown()).c_str());
        ImGui::TextUnformatted("PitchDown:");   ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyPitchDown()).c_str());
        ImGui::TextUnformatted("PitchUp:");     ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyPitchUp()).c_str());
        ImGui::TextUnformatted("YawLeft:");     ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyYawLeft()).c_str());
        ImGui::TextUnformatted("YawRight:");    ImGui::SameLine(150); ImGui::TextUnformatted(buildKeyBindDescriptorString(RtxOptions::FreeCam::keyYawRight()).c_str());
      }

      ImGui::Unindent();
    }
  }

  Matrix4d RtCamera::getShakenViewToWorldMatrix(Matrix4d& viewToWorld, uint32_t flags) {
    float moveLeftRight = 0;
    float moveBackForward = 0;
    float shakeYaw = 0;
    float shakePitch = 0;

    int period = RtxOptions::cameraShakePeriod();
    float sceneScale = RtxOptions::sceneScale();
    CameraAnimationMode animationMode = RtxOptions::cameraAnimationMode();
    float amplitude = RtxOptions::cameraAnimationAmplitude();

    float offset = sin(float(m_context.cameraShakeFrameCount) / (2 * period) * 2.0f * M_PI);
    switch (animationMode) {
    case CameraAnimationMode::CameraShake_LeftRight:
      moveLeftRight += amplitude * sceneScale * offset;
      break;
    case CameraAnimationMode::CameraShake_FrontBack:
      moveBackForward += amplitude * sceneScale * offset;
      break;
    case CameraAnimationMode::CameraShake_Yaw:
      shakeYaw = amplitude * 0.05 * offset;
      break;
    case CameraAnimationMode::CameraShake_Pitch:
      shakePitch = amplitude * 0.05 * offset;
      break;
    case CameraAnimationMode::YawRotation:
      const float constSpeed = float(m_context.cameraRotationFrameCount) / (2 * period) * 2.0f * M_PI;
      shakeYaw = amplitude * 0.05 * constSpeed;
      break;
    }

    if (flags & (uint32_t)UpdateFlag::IncrementShakingFrame) {
      ++m_context.cameraRotationFrameCount;
      m_context.cameraShakeFrameCount = (m_context.cameraShakeFrameCount + 1) % (2 * period);
    }

    Matrix4d viewRot = viewToWorld;
    viewRot[3] = Vector4d(0.0);
    viewRot *= getMatrixFromEulerAngles(shakePitch, shakeYaw);

    Vector4d shakeOffset = static_cast<double>(moveLeftRight) * viewRot.data[0];
    shakeOffset += static_cast<double>(moveBackForward) * viewRot.data[2];

    viewRot[3] = viewToWorld[3] + shakeOffset;

    return viewRot;
  }

  void RtFrustum::calculateFrustumGeometry(const float nearPlane, const float farPlane, const float fov, const float aspectRatio, const bool isLHS) {
    // Calculate frustum near and far plane extents
    const float tanHalfFov = std::tan(fov * 0.5f);
    nearPlaneUpExtent = nearPlane * tanHalfFov;
    nearPlaneRightExtent = nearPlaneUpExtent * aspectRatio;
    farPlaneUpExtent = farPlane * tanHalfFov;
    farPlaneRightExtent = farPlaneUpExtent * aspectRatio;

    const float N = isLHS ? nearPlane : -nearPlane;
    const float F = isLHS ? farPlane : -farPlane;

    // Near Plane Vertices
    nearPlaneFrustumVertices[0] = Vector3(-nearPlaneRightExtent, -nearPlaneUpExtent, N);
    nearPlaneFrustumVertices[1] = Vector3(-nearPlaneRightExtent,  nearPlaneUpExtent, N);
    nearPlaneFrustumVertices[2] = Vector3( nearPlaneRightExtent,  nearPlaneUpExtent, N);
    nearPlaneFrustumVertices[3] = Vector3( nearPlaneRightExtent, -nearPlaneUpExtent, N);

    // Far Plane Vertices
    farPlaneFrustumVertices[0] = Vector3(-farPlaneRightExtent, -farPlaneUpExtent, F);
    farPlaneFrustumVertices[1] = Vector3(-farPlaneRightExtent,  farPlaneUpExtent, F);
    farPlaneFrustumVertices[2] = Vector3( farPlaneRightExtent,  farPlaneUpExtent, F);
    farPlaneFrustumVertices[3] = Vector3( farPlaneRightExtent, -farPlaneUpExtent, F);

    // [InfFar] Edge Vectors (Normalized).
    // Original: normalize(farPlaneVtx[i] - nearPlaneVtx[i]). For infinite-
    // far (TF2's reverse-Z camera reports farPlane=+Inf):
    //   farPlaneVtx[i] = (±Inf, ±Inf, ±Inf)
    //   far - near    = (±Inf, ±Inf, ±Inf)
    //   normalize(Inf vec) = Inf vec / sqrt(Inf*Inf + ...) = Inf / Inf = NaN
    // The NaN edge vectors then feed boundingBoxIntersectsFrustumSAT and
    // contaminate every projection test downstream — that's the over-cull
    // / flicker symptom (NaN compared with anything is false → SAT rejects
    // every separating-axis test as overlapping or non-overlapping
    // depending on order, manifesting as objects randomly culling).
    //
    // Geometrically: a view-space frustum is a pyramid with apex at the
    // origin. The four edges go from the apex outward through the near
    // corners (and continue to the far corners). The direction from a
    // near corner to its far corner is the SAME as the direction from
    // the origin through that near corner — both are colinear with the
    // edge ray. So we can compute the edge direction from the near
    // corner alone, which is finite even when the far plane is at
    // infinity.
    for (int i = 0; i < 4; ++i) {
      if (std::isfinite(farPlaneFrustumVertices[i].x)
          && std::isfinite(farPlaneFrustumVertices[i].y)
          && std::isfinite(farPlaneFrustumVertices[i].z)) {
        frustumEdgeVectors[i] = normalize(farPlaneFrustumVertices[i] - nearPlaneFrustumVertices[i]);
      } else {
        // Infinite-far branch: direction from apex (origin) through near
        // corner. Equivalent to the original formula in the finite-far
        // limit; mathematically the limit as farPlane → ∞ of
        // normalize(far - near).
        frustumEdgeVectors[i] = normalize(nearPlaneFrustumVertices[i]);
      }
    }

    // [InfFar.diag] Log the resulting geometry once per camera type +
    // every 500ms to surface NaN/Inf if any slip through.
    {
      static thread_local uint64_t sN = 0;
      static thread_local std::chrono::steady_clock::time_point sLastT
        = std::chrono::steady_clock::time_point::min();
      ++sN;
      const auto now = std::chrono::steady_clock::now();
      const bool firstFew = sN <= 5;
      const bool dueByClock =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - sLastT).count() >= 500;
      const bool anyNaN =
        std::isnan(frustumEdgeVectors[0].x) || std::isnan(frustumEdgeVectors[1].x)
       || std::isnan(frustumEdgeVectors[2].x) || std::isnan(frustumEdgeVectors[3].x);
      if (firstFew || dueByClock || anyNaN) {
        sLastT = now;
        Logger::warn(str::format(
          "[InfFar.diag.calcFrustum] n=", sN,
          " near=", nearPlane, " far=", farPlane,
          " fov=", fov, " aspect=", aspectRatio, " isLHS=", (isLHS ? 1 : 0),
          " edge0=(", frustumEdgeVectors[0].x, ",", frustumEdgeVectors[0].y, ",", frustumEdgeVectors[0].z, ")",
          " edge1=(", frustumEdgeVectors[1].x, ",", frustumEdgeVectors[1].y, ",", frustumEdgeVectors[1].z, ")",
          " edge2=(", frustumEdgeVectors[2].x, ",", frustumEdgeVectors[2].y, ",", frustumEdgeVectors[2].z, ")",
          " edge3=(", frustumEdgeVectors[3].x, ",", frustumEdgeVectors[3].y, ",", frustumEdgeVectors[3].z, ")",
          " anyNaN=", (anyNaN ? 1 : 0)));
      }
    }
  }

  void RtCameraSequence::startPlay() {
    mode.setDeferred(Mode::Playback);
    m_currentFrame = 0;
  }

  void RtCameraSequence::startRecord() {
    mode.setDeferred(Mode::Record);
    m_currentFrame = 0;
    m_settings.clear();
  }

  void RtCameraSequence::addRecord(const RtCamera::RtCameraSetting& setting) {
    m_settings.push_back(setting);
    m_currentFrame = m_settings.size() - 1;
  }

  void RtCameraSequence::save() {
    std::ofstream file(filePath(), std::ios::out | std::ios::binary);

    if (file.is_open()) {
      int nElements = m_settings.size();
      Header header;
      memset(&header, 0, sizeof(header));
      header.nElements = nElements;
      file.write((char*)&header, sizeof(header));

      for (int i = 0; i < nElements; i++) {
        FrameData frameData = {};
        memset(&frameData, 0, sizeof(frameData));
        frameData.setting = m_settings[i];
        file.write((char*)&frameData, sizeof(frameData));
      }

      file.close();
    }
  }

  void RtCameraSequence::load() {
    std::ifstream file(filePath(), std::ios::in | std::ios::binary);

    if (file.is_open()) {
      Header header;
      file.read((char*) &header, sizeof(header));
      int nElements = header.nElements;

      m_settings.resize(nElements);
      for (int i = 0; i < nElements; i++) {
        FrameData frameData = {};
        file.read((char*)&frameData, sizeof(frameData));
        m_settings[i] = frameData.setting;
      }
    }
  }

  bool RtCameraSequence::getRecord(int frame, RtCamera::RtCameraSetting& setting) {
    if (autoLoad() && m_settings.size() == 0 && filePath().size() > 0) {
      load();
    }
    if (frame < 0 || frame >= m_settings.size()) {
      return false;
    }
    setting = m_settings[frame];
    return true;
  }

  void RtCameraSequence::goToNextFrame() {
    if (m_settings.size() == 0) {
      m_currentFrame = 0;
      return;
    }
    m_currentFrame = (m_currentFrame + 1) % (int) m_settings.size();
  }

  void RtCameraSequence::showImguiSettings() {
    RemixGui::InputText("File Path", &filePathObject(), ImGuiInputTextFlags_EnterReturnsTrue);

    if (ImGui::Button("Load Sequence")) {
      load();
    }

    ImGui::SameLine();

    if (ImGui::Button("Save Sequence")) {
      save();
    }

    int oldFrame = m_currentFrame;
    IMGUI_ADD_TOOLTIP(ImGui::SliderInt("Current Frame", &m_currentFrame, 0, m_settings.size() -1, "%d", ImGuiSliderFlags_AlwaysClamp), "Current Frame.");
    m_currentFrame = std::min(m_currentFrame, (int)m_settings.size());

    Mode currentMode = mode();
    if (oldFrame != m_currentFrame && currentMode == Mode::None) {
      mode.setDeferred(Mode::Browse);
      currentMode = Mode::Browse;
    }

    bool isRecording = currentMode == Mode::Record;
    bool isPlaying = currentMode == Mode::Playback;
    bool isBrowsing = currentMode == Mode::Browse;
    bool isStopping = currentMode == Mode::None;

    // Record Button
    {
      ImGui::BeginDisabled(isPlaying);

      if (isRecording) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5, 0, 0, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 0, 0, 1));
      }

      if (ImGui::Button("Record")) {
        startRecord();
      }

      if (isRecording) {
        ImGui::PopStyleColor(2);
      }
      ImGui::EndDisabled();
    }

    ImGui::SameLine();

    // Play Button
    {
      ImGui::BeginDisabled(isRecording);

      if (isPlaying) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0.5, 0, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 1.0, 0, 1));
      }

      if (ImGui::Button("Play")) {
        startPlay();
      }

      if (isPlaying) {
        ImGui::PopStyleColor(2);
      }

      ImGui::EndDisabled();
    }

    ImGui::SameLine();

    // Stop Button
    {
      if (ImGui::Button("Stop")) {
        mode.setDeferred(Mode::None);
        RtxOptions::shakeCamera.setDeferred(false);
      }
    }

    ImGui::SameLine();

    // Browse Button
    {
      ImGui::BeginDisabled(isRecording || isPlaying);

      if (isBrowsing) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5, 0.5, 0, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0, 1.0, 0, 1));
      }

      if (ImGui::Button("Browse")) {
        mode.setDeferred(Mode::Browse);
        m_currentFrame = 0;
      }

      if (isBrowsing) {
        ImGui::PopStyleColor(2);
      }

      ImGui::EndDisabled();
    }

    ImGui::Text("Total Frames: %d", m_settings.size());
  }
}