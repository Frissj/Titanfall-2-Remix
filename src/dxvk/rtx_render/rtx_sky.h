/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_context.h"

#include "dxvk_device.h"
#include "rtx_resources.h"
#include "rtx_scene_manager.h"

namespace dxvk {

  static Matrix4 makeViewMatrixForCubePlane(uint32_t plane, const Vector3& cameraPosition) {
    assert(plane >= 0 && plane < 6);

    constexpr static Vector3 targets[]{
      Vector3{+1,  0,  0},
      Vector3{-1,  0,  0},
      Vector3{ 0, +1,  0},
      Vector3{ 0, -1,  0},
      Vector3{ 0,  0, +1},
      Vector3{ 0,  0, -1},
    };
    constexpr static Vector3 ups[]{
      Vector3{ 0, +1,  0},
      Vector3{ 0, +1,  0},
      Vector3{ 0,  0, -1},
      Vector3{ 0,  0, +1},
      Vector3{ 0, +1,  0},
      Vector3{ 0, +1,  0},
    };
    const static Vector3 axisZ[]{
      targets[0],
      targets[1],
      targets[2],
      targets[3],
      targets[4],
      targets[5],
    };
    const static Vector3 axisX[]{
      cross(ups[0], axisZ[0]),
      cross(ups[1], axisZ[1]),
      cross(ups[2], axisZ[2]),
      cross(ups[3], axisZ[3]),
      cross(ups[4], axisZ[4]),
      cross(ups[5], axisZ[5]),
    };
    const static Vector3 axisY[]{
      cross(axisZ[0], axisX[0]),
      cross(axisZ[1], axisX[1]),
      cross(axisZ[2], axisX[2]),
      cross(axisZ[3], axisX[3]),
      cross(axisZ[4], axisX[4]),
      cross(axisZ[5], axisX[5]),
    };
    assert(isApproxNormalized(axisX[0], 0.0001f) && isApproxNormalized(axisY[0], 0.0001f) && isApproxNormalized(axisZ[0], 0.0001f));
    assert(isApproxNormalized(axisX[1], 0.0001f) && isApproxNormalized(axisY[1], 0.0001f) && isApproxNormalized(axisZ[1], 0.0001f));
    assert(isApproxNormalized(axisX[2], 0.0001f) && isApproxNormalized(axisY[2], 0.0001f) && isApproxNormalized(axisZ[2], 0.0001f));
    assert(isApproxNormalized(axisX[3], 0.0001f) && isApproxNormalized(axisY[3], 0.0001f) && isApproxNormalized(axisZ[3], 0.0001f));
    assert(isApproxNormalized(axisX[4], 0.0001f) && isApproxNormalized(axisY[4], 0.0001f) && isApproxNormalized(axisZ[4], 0.0001f));
    assert(isApproxNormalized(axisX[5], 0.0001f) && isApproxNormalized(axisY[5], 0.0001f) && isApproxNormalized(axisZ[5], 0.0001f));

    const Vector3 translation{
      dot(axisX[plane], -cameraPosition),
      dot(axisY[plane], -cameraPosition),
      dot(axisZ[plane], -cameraPosition),
    };

    return Matrix4{
      axisX[plane].x, axisY[plane].x, axisZ[plane].x, 0.f,
      axisX[plane].y, axisY[plane].y, axisZ[plane].y, 0.f,
      axisX[plane].z, axisY[plane].z, axisZ[plane].z, 0.f,
      translation.x,  translation.y,  translation.z,  1.f,
    };
  }

  static bool isSkyboxQuad(const DrawCallState& state) {
    if (state.getMaterialData().blendMode.enableBlending) {
      return false;
    }
    if (state.getGeometryData().indexCount == 0) {
      return state.getGeometryData().vertexCount <= 6;
    }
    return state.getGeometryData().indexCount <= 6;
  }

} // namespace dxvk

#include "MathLib/MathLib_f.h"

namespace dxvk {
  static Matrix4d overrideNearFarPlanes(const Matrix4d& modifiedViewToProj, float nearPlane, float farPlane) {
    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatModifiedViewToProj{ modifiedViewToProj };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatModifiedViewToProj));

    uint32_t flags;
    float    cameraParams[PROJ_NUM];
    DecomposeProjection(
      NDC_D3D,
      NDC_D3D,
      *reinterpret_cast<float4x4*>(&floatModifiedViewToProj),
      &flags,
      cameraParams,
      nullptr,
      nullptr,
      nullptr,
      nullptr);

    float4x4 newProjection;
    newProjection.SetupByAngles(
      cameraParams[PROJ_ANGLEMINX],
      cameraParams[PROJ_ANGLEMAXX],
      cameraParams[PROJ_ANGLEMINY],
      cameraParams[PROJ_ANGLEMAXY],
      nearPlane,
      farPlane,
      flags);
    memcpy(&floatModifiedViewToProj, &newProjection, sizeof(float4x4));

    return Matrix4d{ floatModifiedViewToProj };
  }
} // namespace dxvk

dxvk::RtxContext::TryHandleSkyResult dxvk::RtxContext::tryHandleSky(const DrawParameters* originalParams,
                                                                    DrawCallState* originalDrawCallState) {

  // Skip all sky geometry when using physical atmosphere mode
  if (originalParams && originalDrawCallState && originalDrawCallState->cameraType == CameraType::Sky &&
      RtxOptions::skyMode() == SkyMode::PhysicalAtmosphere) {
    return TryHandleSkyResult::SkipSubmit;
  }

  // NV-DXVK [SubViewSkyProbe]: TF2 3D-skybox dome populates the SkyProbe
  // cubemap.
  //
  // The dome is classified isSubViewSkybox at the d3d11 frontend and
  // its cb2 snapshot lives in originalDrawCallState->skyProbeCubeCapture
  // (captured at d3d11_rtx.cpp:~16450). The dome is NOT tagged Sky and
  // does not have CameraType::Sky — SetSkyCategoryFromCb2 deliberately
  // routes it as a regular emissive mesh in TLAS so the primary-ray
  // visible sky still has geometric detail and parallax (a Sky-tagged
  // cubemap would collapse 3D structure to a single direction).
  //
  // But the path tracer's secondary/shadow rays still need a populated
  // environment cubemap. Emissive-in-TLAS only contributes when a ray
  // happens to hit it (no NEE on emissive surfaces), so world geometry
  // facing away from the sun gets zero skylight and renders black.
  //
  // Fix: when we see the dome's draw arrive at tryHandleSky, rasterize
  // its actual VS+PS into the 6 cube faces by replaying through the
  // existing rasterizeToSkyProbe machinery (it reads the cb2 snapshot
  // and overrides the c_cameraRelativeToClip slot per face). Then fall
  // through so the dome's normal TLAS submission still proceeds for
  // primary-ray visible-sky rendering.
  //
  // Intentionally NOT calling rasterizeSky / rasterizeToSkyMatte: the
  // matte path would composite a second copy of the sky behind the
  // emissive TLAS dome, double-rendering visible-sky pixels. Probe-only
  // is the right subset for this draw.
  //
  // Gated on cameraType != Sky so a hypothetically-Sky-tagged dome
  // routes through the existing branch below instead of double-firing.
  if (originalParams && originalDrawCallState
      && originalDrawCallState->transformData.isSubViewSkybox
      && originalDrawCallState->skyProbeCubeCapture.valid
      && originalDrawCallState->cameraType != CameraType::Sky) {

    // Mirror the resource init the cameraType==Sky branch does. The
    // current draw's color RT format drives the SkyMatte / SkyProbe
    // image formats. skyForceHDR forces B10G11R11_UFLOAT regardless.
    m_skyRtColorFormat = m_state.om.renderTargets.color[0].view->image()->info().format;
    m_skyColorFormat = TextureUtils::toSRGB(m_skyRtColorFormat);
    if (RtxOptions::skyForceHDR()) {
      m_skyRtColorFormat = m_skyColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    }
    getResourceManager().getCompatibleViewForView(
      getResourceManager().getSkyMatte(this, m_skyColorFormat).view,
      m_skyRtColorFormat);
    initSkyProbe();

    // Save render-target + viewport state so the dome's subsequent TLAS
    // submission sees the same bindings the d3d11 layer set up. Inside
    // rasterizeToSkyProbe each cube face is bound as a render target
    // and viewports are resized to the cube face size; the last face
    // bound is what remains after the 6-face loop returns.
    DxvkRenderTargets curRts = m_state.om.renderTargets;
    const uint32_t curViewportCount = m_state.gp.state.rs.viewportCount();
    const DxvkViewportState curVp = m_state.vp;

    rasterizeToSkyProbe(*originalParams, *originalDrawCallState);
    m_skyClearDirty = false;

    setViewports(curViewportCount, curVp.viewports.data(), curVp.scissorRects.data());
    bindRenderTargets(curRts);

    // Intentional fall-through. The dome continues into the regular
    // non-Sky flow below: m_delayedRayTracedSky handling (no-op when
    // empty) then return Default so the caller submits the dome to
    // TLAS as emissive geometry for primary-ray visible sky.
  }

  if (originalParams && originalDrawCallState && originalDrawCallState->cameraType == CameraType::Sky) {

    // Initialize the sky render targets
    {
      // Use game render target format for sky render target views whether it is linear, HDR or sRGB -- to render into the images
      m_skyRtColorFormat = m_state.om.renderTargets.color[0].view->image()->info().format;
      // Use sRGB (or linear for HDR formats) for image and sampling views -- to use in ray tracing
      m_skyColorFormat = TextureUtils::toSRGB(m_skyRtColorFormat);
      if (RtxOptions::skyForceHDR()) {
        m_skyRtColorFormat = m_skyColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
      }

      getResourceManager().getCompatibleViewForView(
        getResourceManager().getSkyMatte(this, m_skyColorFormat).view,
        m_skyRtColorFormat);
      initSkyProbe();
    }

    auto l_forceRaster = [&]() {
      if (!RtxOptions::skyReprojectToMainCameraSpace()) {
        return true;
      }
      // When skyForceAutoDetectedToReproject is enabled, draw calls classified as sky by
      // autoDetect are always reprojected instead of rasterized to the cubemap. This fixes
      // a class of bugs where autoDetect misclassifies world geometry as sky (due to shared
      // camera positions), causing that geometry to become invisible.
      if (RtxOptions::skyForceAutoDetectedToReproject() && originalDrawCallState->skyAutoDetected) {
        return false;
      }
      // Always rasterize sky planes
      if (isSkyboxQuad(*originalDrawCallState)) {
        return true;
      }
      return false;
    };

    if (l_forceRaster()) {
      rasterizeSky(*originalParams, *originalDrawCallState);
      return TryHandleSkyResult::Default;
    }

    // But for 3D skybox (i.e. the objects that are rendered in sky camera space),
    // we would need to know the main camera to be able to reproject from sky to main camera space,
    // so delay ray traced logic until then
    // [SkyTrace.delayPush] 3D-skybox geometry is reprojected and ray-traced
    // INTO the main scene — bypasses the sky-matte path entirely. If yellow
    // bleeds in via 3D-skybox emissive surfaces (sun disc, sun-tinted clouds,
    // etc.), it shows up here regardless of what the matte contains. First
    // 3 pushes per frame logged to keep volume sane.
    {
      const uint32_t frameId = m_device->getCurrentFrameId();
      static std::atomic<uint32_t> sFrame{ UINT32_MAX };
      static std::atomic<uint32_t> sCount{ 0 };
      const uint32_t prevFrame = sFrame.load(std::memory_order_relaxed);
      if (prevFrame != frameId) {
        sFrame.store(frameId, std::memory_order_relaxed);
        sCount.store(0, std::memory_order_relaxed);
      }
      const uint32_t pushIdx = sCount.fetch_add(1, std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && pushIdx < 3u) {
        Logger::info(str::format(
          "[SkyTrace.delayPush] frame=", frameId,
          " idx=", pushIdx,
          " matHash=0x", std::hex, originalDrawCallState->getMaterialData().getHash(), std::dec,
          " vsHash=0x", std::hex, originalDrawCallState->transformData.vertexShaderHash, std::dec));
      }
    }
    m_delayedRayTracedSky.push_back(std::move(*originalDrawCallState));
    return TryHandleSkyResult::SkipSubmit;
  }

  // Received a non-sky 'originalDrawCallState'
  assert(!originalDrawCallState || originalDrawCallState->cameraType != CameraType::Sky);

  if (m_delayedRayTracedSky.empty()) {
    return TryHandleSkyResult::Default;
  }

  // 2. Submit ray traced sky geometry as a part of the main scene by reprojecting its transform
  const RtCamera& mainCam = getSceneManager().getCameraManager().getCamera(CameraType::Main);

  if (mainCam.getLastUpdateFrame() != m_device->getCurrentFrameId()) {
    // Skip, if the main camera hasn't been updated yet
    return TryHandleSkyResult::Default;
  }

  // Note: getNearPlane() / getFarPlane() do not return actual values in case if overrideNearPlane is enabled
  const auto [mainCamNearPlane, mainCamFarPlane] = mainCam.calculateNearFarPlanes();

  const float scale = RtxOptions::skyReprojectScale();
  const Matrix4d scaleMatrix{
    scale, 0,     0,     0,
    0,     scale, 0,     0,
    0,     0,     scale, 0,
    0,     0,     0,     1,
  };

  // [SkyTrace.delayReplay] How many delayed 3D-skybox draws are about to
  // be reprojected and submitted as main-scene ray-traced geometry, plus
  // the first material hash for a quick "is the sun disc in here?" check.
  // If yellow tracks with delayCount > 0, the 3D skybox is the source.
  {
    const uint32_t frameId = m_device->getCurrentFrameId();
    static std::atomic<uint32_t> sLastFrame{ UINT32_MAX };
    const uint32_t lastLogged = sLastFrame.load(std::memory_order_relaxed);
    const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
    const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
    if (cameraValid && tlasReady && lastLogged != frameId) {
      sLastFrame.store(frameId, std::memory_order_relaxed);
      const XXH64_hash_t firstMatHash = m_delayedRayTracedSky.empty()
        ? 0ull : m_delayedRayTracedSky.front().getMaterialData().getHash();
      Logger::info(str::format(
        "[SkyTrace.delayReplay] frame=", frameId,
        " delayCount=", m_delayedRayTracedSky.size(),
        " reprojScale=", scale,
        " firstMatHash=0x", std::hex, firstMatHash, std::dec));
    }
  }

  for (DrawCallState& skyGeometry : m_delayedRayTracedSky) {
    // Swap camera
    skyGeometry.cameraType = CameraType::Main;
    skyGeometry.categories.clr(InstanceCategories::Sky);

    // And reproject
    DrawCallTransforms& skyTransform = skyGeometry.transformData;

    // Near / far planes must match to prevent problems related to the mismatching Z-space
    const Matrix4d skyViewToProjection = overrideNearFarPlanes(
      skyTransform.viewToProjection,
      mainCamNearPlane,
      mainCamFarPlane);

    const Matrix4d skyViewToMainWorld =
      mainCam.getViewToWorld(false) *
      (mainCam.getProjectionToView() * skyViewToProjection) *
      scaleMatrix;

    skyTransform.objectToWorld    = skyViewToMainWorld * skyTransform.worldToView * skyTransform.objectToWorld;
    skyTransform.worldToView      = mainCam.getWorldToView();
    skyTransform.viewToProjection = mainCam.getViewToProjection();
    skyTransform.sanitize();

    getSceneManager().submitDrawState(this, skyGeometry, nullptr);
  }
  m_delayedRayTracedSky.clear();

  // since 'originalDrawCallState' is not a sky (it just triggers delayed sky submit),
  // proceed with the default path
  assert(!originalDrawCallState || originalDrawCallState->cameraType != CameraType::Sky);
  return TryHandleSkyResult::Default;
}
