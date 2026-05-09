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
#include "rtx_atmosphere.h"
#include "rtx_engine_sun.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_options.h"
#include "rtx_context.h"
#include "rtx_render/rtx_shader_manager.h"
#include <rtx_shaders/transmittance_lut.h>
#include <rtx_shaders/multiscattering_lut.h>
#include <rtx_shaders/sky_view_lut.h>
#include <rtx_shaders/aerial_perspective_lut.h>
#include <cmath>
#include <fstream>
#include <chrono>
#include <mutex>

namespace dxvk {

  // NV-DXVK [EngineSunCapture]: file-scope storage for the snapshot the
  // d3d11 producer publishes. Both publishEngineSunCapture (called from
  // d3d11_rtx.cpp via the rtx_engine_sun.h decl) and fetchEngineSunCapture
  // (called from getAtmosphereArgs below) operate on these.
  namespace {
    std::mutex        g_engineSunMutex;
    EngineSunSnapshot g_engineSun{};
  }

  void publishEngineSunCapture(const EngineSunSnapshot& snap) {
    std::lock_guard<std::mutex> lk(g_engineSunMutex);
    g_engineSun = snap;
  }

  EngineSunSnapshot fetchEngineSunCapture() {
    std::lock_guard<std::mutex> lk(g_engineSunMutex);
    return g_engineSun;
  }
  // Shader definitions for atmosphere LUT generation
  namespace {
    class TransmittanceLutShader : public ManagedShader {
      SHADER_SOURCE(TransmittanceLutShader, VK_SHADER_STAGE_COMPUTE_BIT, transmittance_lut)
      
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE2D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(TransmittanceLutShader);

    class MultiscatteringLutShader : public ManagedShader {
      SHADER_SOURCE(MultiscatteringLutShader, VK_SHADER_STAGE_COMPUTE_BIT, multiscattering_lut)
      
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        SAMPLER(2)
        RW_TEXTURE2D(3)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(MultiscatteringLutShader);

    class SkyViewLutShader : public ManagedShader {
      SHADER_SOURCE(SkyViewLutShader, VK_SHADER_STAGE_COMPUTE_BIT, sky_view_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        SAMPLER(3)
        RW_TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SkyViewLutShader);

    // NV-DXVK [AerialPerspective]: 3D LUT generation shader.
    // Bindings mirror SkyViewLutShader except the output is RW 3D.
    class AerialPerspectiveLutShader : public ManagedShader {
      SHADER_SOURCE(AerialPerspectiveLutShader, VK_SHADER_STAGE_COMPUTE_BIT, aerial_perspective_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        SAMPLER(3)
        RW_TEXTURE3D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AerialPerspectiveLutShader);
  }

RtxAtmosphere::RtxAtmosphere(DxvkDevice* device)
  : CommonDeviceObject(device) {
  // Create constant buffer for atmosphere parameters
  DxvkBufferCreateInfo info;
  info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  info.access = VK_ACCESS_UNIFORM_READ_BIT;
  info.size = sizeof(AtmosphereArgs);
  m_constantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Atmosphere constants buffer");
}

RtxAtmosphere::~RtxAtmosphere() {
}

void RtxAtmosphere::initialize(Rc<DxvkContext> ctx) {
  if (m_initialized) {
    return;
  }

  createLutResources(ctx);
  m_initialized = true;
  m_lutsNeedRecompute = true;
}

AtmosphereArgs RtxAtmosphere::getAtmosphereArgs() const {
  AtmosphereArgs args = {};

  // Convert sun angles to direction vector (in Y-up space, for LUT generation)
  constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
  float azimuthRad = RtxOptions::sunRotation() * kDegToRad; // Mapped to Rotation
  float elevationRad = RtxOptions::sunElevation() * kDegToRad;

  // Sun direction is always in Y-up space since the LUTs are generated in Y-up space
  args.sunDirection.x = std::cos(elevationRad) * std::sin(azimuthRad);
  args.sunDirection.y = std::sin(elevationRad);
  args.sunDirection.z = std::cos(elevationRad) * std::cos(azimuthRad);

  // Basic atmosphere parameters
  args.planetRadius = RtxOptions::planetRadius();
  args.atmosphereThickness = RtxOptions::atmosphereThickness();

  // Sun illuminance (Base * Intensity)
  // Allows customizing base color via options/presets, while simple UI controls intensity
  args.sunIlluminance = RtxOptions::sunIlluminance() * RtxOptions::sunIntensity();

  // NV-DXVK [EngineSunCapture]: override the slider-derived sun with the
  // game's per-frame value if a fresh snapshot exists. Producer is
  // D3D11Rtx::CaptureEngineSunFromCb (reads CBufCommonPerCamera.c_sunDir
  // off=288 and .c_sunColor off=276 — confirmed via the dump diagnostics).
  if (RtxOptions::useEngineSun()) {
    EngineSunSnapshot snap = fetchEngineSunCapture();
    if (snap.valid) {
      // Game world-space -> LUT Y-up space. TF2/Source is Z-up: world
      // (x,y,z) maps to Y-up (x, z, y). engineSunDirIsTowardsLight
      // controls the from-/to-light convention flip — TF2's c_sunDir
      // points TO the sun (z=+0.69 with sun above horizon), so default
      // is true (no flip needed).
      Vector3 d = snap.worldDirection;
      if (RtxOptions::engineSunIsZUp()) {
        d = Vector3 { d.x, d.z, d.y };
      }
      if (!RtxOptions::engineSunDirIsTowardsLight()) {
        d = Vector3 { -d.x, -d.y, -d.z };
      }
      const float len = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
      if (len > 1e-4f) {
        const float inv = 1.0f / len;
        args.sunDirection.x = d.x * inv;
        args.sunDirection.y = d.y * inv;
        args.sunDirection.z = d.z * inv;
      }

      // Colour: TF2 pushes pre-scaled values (~6.3 magnitude on a typical
      // map) so engineSunIntensityScale is just a small fudge factor on
      // top to match the slider default illuminance scale (~20).
      const float k = RtxOptions::engineSunIntensityScale();
      args.sunIlluminance = Vector3 {
        snap.colorLinear.x * k,
        snap.colorLinear.y * k,
        snap.colorLinear.z * k
      };

      // Sun disc angular size from c_sunHighlightSize (TF2 cbuffer
      // off 512). The engine ships per-map values - cinematic skies
      // have visibly larger suns than gameplay maps. We previously
      // used the static sunSize slider (0.545deg = real-world Earth
      // sun) which was wrong for any TF2 sky that artist-authored a
      // larger disc.
      // sunHighlightSize is observed in the 0.01..0.5 range which we
      // interpret as the cosine-of-half-angle (typical engine
      // encoding). Values outside that range fall back to the slider.
      if (snap.sunHighlightSize > 1e-5f && snap.sunHighlightSize < 1.0f) {
        // Convert cos(halfAngle) -> halfAngle radians.
        // Floor sunHighlightSize at cos(5deg)=0.996 to keep
        // numerically stable when TF2 encodes a tiny 'realistic' sun.
        const float cosHalf = std::min(snap.sunHighlightSize, 0.99996f);
        args.sunAngularRadius = std::acos(cosHalf);
      }
    }
  }

  // Scattering coefficients (Base * Density Multiplier)
  // Allows advanced customization of scattering colors while exposing simple density sliders
  float airDensity = RtxOptions::airDensity();
  args.rayleighScattering = RtxOptions::rayleighScattering() * airDensity;
  
  float aerosolDensity = RtxOptions::aerosolDensity();
  args.mieScattering = RtxOptions::mieScattering() * aerosolDensity;
  
  args.mieAnisotropy = RtxOptions::mieAnisotropy();
  
  // Sun Angular Radius (from Sun Size in degrees)
  // sunSize is diameter in degrees. Radius = Size / 2
  float sunSizeRad = RtxOptions::sunSize() * kDegToRad;
  args.sunAngularRadius = sunSizeRad * 0.5f;
  
  // Brightness multiplier
  args.sunRayBrightness = 1.0f; 

  // Ozone absorption (Base * Density Multiplier)
  float ozoneDensity = RtxOptions::ozoneDensity();
  args.ozoneAbsorption = RtxOptions::ozoneAbsorption() * ozoneDensity;
  
  // Internal ozone params
  args.ozoneLayerAltitude = RtxOptions::ozoneLayerAltitude();
  args.ozoneLayerWidth = RtxOptions::ozoneLayerWidth();

  // View Altitude (converted m to km)
  args.viewAltitude = RtxOptions::altitude() * 0.001f;

  // LUT dimensions
  args.transmittanceLutWidth = kTransmittanceLutWidth;
  args.transmittanceLutHeight = kTransmittanceLutHeight;
  args.multiscatteringLutSize = kMultiscatteringLutSize;
  args.skyViewLutWidth = kSkyViewLutWidth;
  args.skyViewLutHeight = kSkyViewLutHeight;

  // Derived parameters
  args.atmosphereRadius = args.planetRadius + args.atmosphereThickness;
  args.rayleighScaleHeight = kRayleighScaleHeight;
  args.mieScaleHeight = kMieScaleHeight;
  args.pad2 = 0;

  // NV-DXVK [AerialPerspective]: 3D LUT dimensions and tuning. The
  // worldToKm scale converts game units (TF2 hammer = ~52.5 units/m,
  // so 1 km = ~52,500 hammer => 1.9e-5 km/hammer). Strength is a user
  // multiplier applied at sample time so the haze can be dialed up or
  // down without rebuilding the LUT.
  args.aerialPerspectiveLutSize       = kAerialPerspectiveLutSize;
  args.aerialPerspectiveMaxDistanceKm = kAerialPerspectiveMaxDistanceKm;
  args.aerialPerspectiveStrength      = RtxOptions::aerialPerspectiveStrength();
  args.aerialPerspectiveWorldToKm     = RtxOptions::aerialPerspectiveWorldToKm();

  return args;
}

bool RtxAtmosphere::needsLutRecompute() const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  // Check if any parameters have changed
  AtmosphereArgs currentArgs = getAtmosphereArgs();
  
  // Compare with cached args (simple memcmp would work for POD types)
  return memcmp(&currentArgs, &m_cachedArgs, sizeof(AtmosphereArgs)) != 0;
}

void RtxAtmosphere::createLutResources(Rc<DxvkContext> ctx) {
  // Create transmittance LUT (stores atmospheric transmittance)
  VkExtent3D transmittanceExtent = { kTransmittanceLutWidth, kTransmittanceLutHeight, 1 };
  m_transmittanceLut = Resources::createImageResource(
    ctx,
    "Atmosphere Transmittance LUT",
    transmittanceExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Create multiscattering LUT (stores multiple scattering contribution)
  VkExtent3D multiscatteringExtent = { kMultiscatteringLutSize, kMultiscatteringLutSize, 1 };
  m_multiscatteringLut = Resources::createImageResource(
    ctx,
    "Atmosphere Multiscattering LUT",
    multiscatteringExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Create sky view LUT (main view-dependent sky color LUT)
  VkExtent3D skyViewExtent = { kSkyViewLutWidth, kSkyViewLutHeight, 1 };
  m_skyViewLut = Resources::createImageResource(
    ctx,
    "Atmosphere Sky View LUT",
    skyViewExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // NV-DXVK [AerialPerspective]: 3D LUT for in-scatter+transmittance
  // along a ray of length D. 32^3 RGBA16F = 256 KB. Re-dispatched
  // alongside the other atmosphere LUTs whenever args change.
  VkExtent3D aerialExtent = { kAerialPerspectiveLutSize,
                              kAerialPerspectiveLutSize,
                              kAerialPerspectiveLutSize };
  m_aerialPerspectiveLut = Resources::createImageResource(
    ctx,
    "Atmosphere Aerial Perspective LUT",
    aerialExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_VIEW_TYPE_3D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT,
    VkClearColorValue{},
    1 // mipLevels
  );
}

void RtxAtmosphere::computeLuts(Rc<DxvkContext> ctx) {
  if (!needsLutRecompute()) {
    return;
  }

  // Update cached args
  m_cachedArgs = getAtmosphereArgs();

  // Dispatch compute shaders to generate LUTs
  // Note: Barriers are needed between dispatches since each LUT depends on previous ones
  dispatchTransmittanceLut(ctx);
  
  // Barrier: Ensure transmittance LUT is written before reading in subsequent passes
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT);
  
  dispatchMultiscatteringLut(ctx);
  
  // Barrier: Ensure multiscattering LUT is written before reading in sky view pass
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT);
  
  dispatchSkyViewLut(ctx);

  // NV-DXVK [AerialPerspective]: 3D LUT depends on transmittance + multi-
  // scattering same as sky view, so we can reuse the prior barriers
  // (multiscattering already published before sky view ran). Add one
  // barrier after sky view in case both reads compete for the same SRV
  // backing memory on some drivers.
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_READ_BIT);

  dispatchAerialPerspectiveLut(ctx);

  // Final barrier: Ensure all LUTs are written before use in ray tracing
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_ACCESS_SHADER_READ_BIT);

  m_lutsNeedRecompute = false;
}

void RtxAtmosphere::dispatchTransmittanceLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Transmittance LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_transmittanceLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TransmittanceLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kTransmittanceLutWidth + 15) / 16;
  uint32_t groupsY = (kTransmittanceLutHeight + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchMultiscatteringLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Multiscattering LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  
  // Create and bind a linear sampler. NV-DXVK: must zero-init —
  // DxvkSamplerCreateInfo is a POD with no default ctor; the original
  // code only set the 6 fields below and left mipmapLodBias/Min/Max,
  // useAnisotropy, maxAnisotropy, compareOp, borderColor and
  // usePixelCoord holding stack garbage, which crashes vkCreateSampler
  // (TF2 first-frame AV inside DxvkSampler ctor at dxvk_sampler.cpp:51).
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.mipmapLodBias = 0.0f;
  samplerInfo.mipmapLodMin = 0.0f;
  samplerInfo.mipmapLodMax = 0.25f;
  samplerInfo.useAnisotropy = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.compareToDepth = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
  samplerInfo.usePixelCoord = VK_FALSE;
  Rc<DxvkSampler> linearSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(2, linearSampler);

  ctx->bindResourceView(3, m_multiscatteringLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_multiscatteringLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, MultiscatteringLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kMultiscatteringLutSize + 15) / 16;
  uint32_t groupsY = (kMultiscatteringLutSize + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchSkyViewLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky View LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);
  
  // Create and bind a linear sampler. See dispatchMultiscatteringLut for
  // why the zero-init + explicit fills are mandatory (POD struct, garbage
  // stack memory was crashing vkCreateSampler).
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.mipmapLodBias = 0.0f;
  samplerInfo.mipmapLodMin = 0.0f;
  samplerInfo.mipmapLodMax = 0.25f;
  samplerInfo.useAnisotropy = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.compareToDepth = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
  samplerInfo.usePixelCoord = VK_FALSE;
  Rc<DxvkSampler> linearSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(3, linearSampler);

  ctx->bindResourceView(4, m_skyViewLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyViewLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SkyViewLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kSkyViewLutWidth + 15) / 16;
  uint32_t groupsY = (kSkyViewLutHeight + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

// NV-DXVK [AerialPerspective]: dispatch the 3D LUT generator. 32^3 = 32K
// threads, organised as 8x8x8 workgroups => 4x4x4 dispatch groups.
void RtxAtmosphere::dispatchAerialPerspectiveLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Aerial Perspective LUT");

  // One-shot startup banner so the user can verify the LUT pipeline is
  // running and what tunables are in effect. Logs once per process.
  static std::atomic<bool> sLoggedStartup{ false };
  bool expected = false;
  if (sLoggedStartup.compare_exchange_strong(expected, true,
        std::memory_order_relaxed)) {
    const AtmosphereArgs a = getAtmosphereArgs();
    Logger::info(str::format(
      "[Atmosphere.startup] aerialPerspectiveLut=", kAerialPerspectiveLutSize,
      "^3 maxKm=", a.aerialPerspectiveMaxDistanceKm,
      " strength=", a.aerialPerspectiveStrength,
      " worldToKm=", a.aerialPerspectiveWorldToKm,
      " sunDir=(", a.sunDirection.x, ",", a.sunDirection.y, ",", a.sunDirection.z, ")",
      " sunIlluminance=(", a.sunIlluminance.x, ",", a.sunIlluminance.y, ",", a.sunIlluminance.z, ")",
      " sunAngularRadius=", a.sunAngularRadius));

    // [AerialPerspective.spv] Walk the embedded SPIR-V header section
    // (every OpCapability appears before the first non-capability op
    // per the SPIR-V layout rule) and log the capability list so we
    // can confirm whether StorageImageWriteWithoutFormat (id 56) made
    // it through compile_shaders.py + add_spirv_capabilities.py into
    // the running blob. If it's missing here, the validation-layer
    // 'StorageImageWriteWithoutFormat' errors at startup come from
    // this shader; if it's present, the patcher worked and the errors
    // refer to a different shader.
    constexpr uint32_t kSpirvMagic = 0x07230203u;
    constexpr uint32_t kOpCapability = 17u;
    constexpr uint32_t kCapShader = 1u;
    constexpr uint32_t kCapStorageImageReadWithoutFormat = 55u;
    constexpr uint32_t kCapStorageImageWriteWithoutFormat = 56u;

    const uint32_t* spv = aerial_perspective_lut;
    const size_t spvWords = sizeof(aerial_perspective_lut) / sizeof(uint32_t);
    bool spvOk = (spvWords >= 5) && (spv[0] == kSpirvMagic);
    bool capShader = false, capRead = false, capWrite = false;
    std::string capList;
    size_t capCount = 0;
    if (spvOk) {
      // Header is 5 words; instructions begin at word 5.
      size_t pos = 5;
      while (pos + 1 <= spvWords) {
        uint32_t word = spv[pos];
        uint32_t wc = (word >> 16) & 0xFFFFu;
        uint32_t op = word & 0xFFFFu;
        if (wc == 0 || pos + wc > spvWords) break;
        if (op != kOpCapability) {
          // Capabilities must come first; anything else means we're done.
          break;
        }
        if (wc >= 2) {
          uint32_t cap = spv[pos + 1];
          if (cap == kCapShader)                          capShader = true;
          if (cap == kCapStorageImageReadWithoutFormat)   capRead   = true;
          if (cap == kCapStorageImageWriteWithoutFormat)  capWrite  = true;
          if (capCount) capList += ",";
          capList += std::to_string(cap);
          ++capCount;
        }
        pos += wc;
      }
    }
    Logger::info(str::format(
      "[AerialPerspective.spv] words=", spvWords,
      " magicOk=", (spvOk ? "true" : "false"),
      " capCount=", capCount,
      " caps=[", capList, "]",
      " Shader=", (capShader ? "true" : "false"),
      " StorageImageReadWithoutFormat=", (capRead ? "true" : "false"),
      " StorageImageWriteWithoutFormat=", (capWrite ? "true" : "false")));
    if (!capWrite) {
      Logger::warn("[AerialPerspective.spv] StorageImageWriteWithoutFormat "
                   "capability MISSING from embedded blob — "
                   "add_spirv_capabilities.py did not run on this shader, "
                   "or its output got overwritten. This explains the "
                   "validation errors in dxgi.log; non-fatal but should "
                   "be fixed in the build pipeline.");
    }
  }

  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);

  // Linear sampler — same zero-init pattern as the other dispatches.
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.mipmapLodBias = 0.0f;
  samplerInfo.mipmapLodMin = 0.0f;
  samplerInfo.mipmapLodMax = 0.25f;
  samplerInfo.useAnisotropy = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.compareToDepth = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
  samplerInfo.usePixelCoord = VK_FALSE;
  Rc<DxvkSampler> linearSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(3, linearSampler);

  ctx->bindResourceView(4, m_aerialPerspectiveLut.view, nullptr);

  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_aerialPerspectiveLut.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AerialPerspectiveLutShader::getShader());

  // 8x8x8 thread groups
  const uint32_t groups = (kAerialPerspectiveLutSize + 7) / 8;
  ctx->dispatch(groups, groups, groups);
}

void RtxAtmosphere::bindResources(Rc<DxvkContext> ctx, VkPipelineBindPoint pipelineBindPoint) {
  // TODO: Bind atmosphere LUT resources to the pipeline
  // This will be called from RtxContext to make the LUTs available to shaders
}

} // namespace dxvk
