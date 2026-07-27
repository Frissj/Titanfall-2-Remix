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
#pragma once

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

namespace dxvk {

class DxvkContext;
class DxvkDevice;

/**
 * \brief Hillaire Physically-Based Atmospheric Scattering
 * 
 * Manages lookup table (LUT) resources and compute shader dispatch
 * for atmospheric scattering based on Sebastien Hillaire's method.
 */
class RtxAtmosphere : public CommonDeviceObject {
public:
  explicit RtxAtmosphere(DxvkDevice* device);
  ~RtxAtmosphere();

  /**
   * \brief Initialize atmosphere resources
   */
  void initialize(Rc<DxvkContext> ctx);

  /**
   * \brief Compute atmospheric LUTs if needed
   * 
   * Checks if parameters have changed and recomputes LUTs if necessary.
   */
  void computeLuts(Rc<DxvkContext> ctx);

  /**
   * \brief Bind atmosphere resources to pipeline
   */
  void bindResources(Rc<DxvkContext> ctx, VkPipelineBindPoint pipelineBindPoint);

  /**
   * \brief Check if LUTs need recomputation
   */
  bool needsLutRecompute() const;

  /**
   * \brief Get transmittance LUT resource
   */
  Resources::Resource getTransmittanceLut() const { return m_transmittanceLut; }

  /**
   * \brief Get multiscattering LUT resource
   */
  Resources::Resource getMultiscatteringLut() const { return m_multiscatteringLut; }

  /**
   * \brief Get sky view LUT resource
   */
  Resources::Resource getSkyViewLut() const { return m_skyViewLut; }

  /**
   * \brief Get aerial perspective LUT (3D) resource
   *
   * Filled by aerial_perspective_lut.comp.slang. Sampled by the path
   * tracer to apply atmospheric in-scatter + transmittance to surface
   * shading regardless of which sky source produces the visible sky.
   */
  Resources::Resource getAerialPerspectiveLut() const { return m_aerialPerspectiveLut; }

  /**
   * \brief Get current atmosphere parameters
   */
  AtmosphereArgs getAtmosphereArgs() const;

  /**
   * \brief Fill all 6 SkyProbe cube faces with Hillaire-derived sky × skyTint.
   *
   * Called once per frame from RtxContext::rasterizeToSkyProbe before TF2's
   * own sky-shader overlays its partial-coverage content. Faces without TF2
   * geometry (e.g. -Y for upper-hemisphere skyboxes) retain the analytic
   * background instead of being literal zero.
   *
   * Uses six single-layer 2D storage views and dispatches once per face,
   * passing the face index in args. We tried a single 2D-array storage
   * view + one dispatch with z=6 first; only layer 0 received writes in
   * practice, see m_skyProbeCubePlaneStorageViews comment in rtx_context.h.
   */
  /*
   * NV-DXVK [SkyPrefillCache]: cubeSkyImage is the probe cube itself, needed so
   * the analytic result can be cached and replayed instead of recomputed.
   *
   * Measured 2026-07-27 by Nsight GPU Trace (hardware timings, not our own
   * BOTTOM_OF_PIPE stage marks, which misattributed this five times):
   *   Atmosphere Cube Sky Prefill  75.375 ms   <- this function
   *   InjectRTX                    55.359 ms   <- the ENTIRE path tracer
   * i.e. filling the sky probe cost MORE than path tracing the frame. At the
   * 1024 default that is 6 dispatches x 1024^2 = 6.3 M pixels of full analytic
   * Hillaire atmosphere, every frame.
   *
   * It ran every frame because the caller gates on m_skyClearDirty, which is set
   * whenever the GAME clears a sky render target (rtx_context.cpp:10340, :10355).
   * That is the wrong condition: the analytic sky depends only on atmosphere
   * parameters (sun direction, tint, turbidity), not on TF2 clearing a target,
   * so a byte-identical cubemap was rebuilt from scratch each frame.
   *
   * WHY A CACHE IMAGE AND NOT JUST AN EARLY RETURN: skipping the dispatch is not
   * equivalent to running it. The probe faces are re-rendered by TF2's own sky
   * draws every frame, and those draws inherit whatever blend state the game had
   * bound - nothing in the cube-face pass forces blending off. If the prefill is
   * simply skipped, the analytic background is never re-established and any
   * blended sky draw accumulates over previous frames. Copying a cached cube in
   * reproduces today's starting state exactly, so correctness does not depend on
   * assumptions about the game's blend state.
   */
  void dispatchCubeSkyPrefill(Rc<DxvkContext> ctx,
                              const Rc<DxvkImageView>* cubePlaneStorageViews,
                              uint32_t cubeFaceSize,
                              const Rc<DxvkImage>& cubeSkyImage);

private:
  void createLutResources(Rc<DxvkContext> ctx);
  void dispatchTransmittanceLut(Rc<DxvkContext> ctx);
  void dispatchMultiscatteringLut(Rc<DxvkContext> ctx);
  void dispatchSkyViewLut(Rc<DxvkContext> ctx);
  void dispatchAerialPerspectiveLut(Rc<DxvkContext> ctx);

  // LUT dimensions
  static constexpr uint32_t kTransmittanceLutWidth = 512;   // Increased from 256 for better precision
  static constexpr uint32_t kTransmittanceLutHeight = 128;  // Increased from 64 for better precision
  static constexpr uint32_t kMultiscatteringLutSize = 32;
  static constexpr uint32_t kSkyViewLutWidth = 512;   // Increased from 192 to eliminate aliasing artifacts
  static constexpr uint32_t kSkyViewLutHeight = 256;  // Increased from 108 to eliminate aliasing artifacts
  // NV-DXVK [AerialPerspective]: cubic 3D LUT. 32^3 = 32K voxels x 8B = 256 KB.
  // Cheap to recompute per frame (sun direction changes track camera/time).
  static constexpr uint32_t kAerialPerspectiveLutSize = 32;
  static constexpr float    kAerialPerspectiveMaxDistanceKm = 32.0f;

  // Scale heights for exponential density profiles (in km)
  static constexpr float kRayleighScaleHeight = 8.0f;
  static constexpr float kMieScaleHeight = 1.2f;

  // NV-DXVK [SkyPrefillCache]: last analytic cube, and the inputs that produced
  // it. m_cubeSkyCacheArgs is compared field-for-field against the current
  // AtmosphereArgs to decide recompute vs replay. getAtmosphereArgs() starts
  // from `AtmosphereArgs args = {}` so padding bytes are zeroed and a plain
  // memcmp is well-defined here.
  Resources::Resource m_cubeSkyCache;
  AtmosphereArgs      m_cubeSkyCacheArgs = {};
  uint32_t            m_cubeSkyCacheSide = 0;
  bool                m_cubeSkyCacheValid = false;

  Resources::Resource m_transmittanceLut;
  Resources::Resource m_multiscatteringLut;
  Resources::Resource m_skyViewLut;
  Resources::Resource m_aerialPerspectiveLut;
  
  Rc<DxvkBuffer> m_constantsBuffer;

  AtmosphereArgs m_cachedArgs;
  bool m_initialized = false;
  bool m_lutsNeedRecompute = true;
};

} // namespace dxvk
