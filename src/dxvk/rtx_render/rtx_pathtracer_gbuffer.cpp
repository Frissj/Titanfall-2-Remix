/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_pathtracer_gbuffer.h"
#include "dxvk_device.h"
#include <algorithm>
#include "rtx_shader_manager.h"
#include "rtx_options.h"
#include "rtx_neural_radiance_cache.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/gbuffer/gbuffer_binding_indices.h"
#include "rtx/concept/surface_material/surface_material_hitgroup.h"

#include <rtx_shaders/gbuffer_variants.h>
#include <rtx_shaders/gbuffer_rayquery_variants.h>
#include <rtx_shaders/gbuffer_psr_prepare.h>
#include <rtx_shaders/gbuffer_decal_prepare_variants.h>

// NV-DXVK [perf]: opaque+translucent (no ray portal) compute variants. Same
// source, narrower SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK — see gbuffer.slang.
#include <rtx_shaders/gbuffer_rayquery_opaque_translucent.h>
#include <rtx_shaders/gbuffer_rayquery_nrc_opaque_translucent.h>
#include <rtx_shaders/gbuffer_rayquery_wboit_opaque_translucent.h>
#include <rtx_shaders/gbuffer_rayquery_nrc_wboit_opaque_translucent.h>
#include <rtx_shaders/gbuffer_psr_rayquery_opaque_translucent.h>
#include <rtx_shaders/gbuffer_psr_rayquery_nrc_opaque_translucent.h>
#include <rtx_shaders/gbuffer_psr_rayquery_wboit_opaque_translucent.h>
#include <rtx_shaders/gbuffer_psr_rayquery_nrc_wboit_opaque_translucent.h>

// NV-DXVK [Perf.Occupancy]: block-size ladder, diagnostic only.
#include <rtx_shaders/gbuffer_rayquery_nrc_wboit_opaque_translucent_occ256.h>
#include <rtx_shaders/gbuffer_rayquery_nrc_wboit_opaque_translucent_occ384.h>
#include <rtx_shaders/gbuffer_rayquery_nrc_wboit_opaque_translucent_occ512.h>

#include <rtx_shaders/gbuffer_miss.h>
#include <rtx_shaders/gbuffer_nrc_miss.h>
#include <rtx_shaders/gbuffer_psr_miss.h>
#include <rtx_shaders/gbuffer_psr_nrc_miss.h>

#include <rtx_shaders/gbuffer_material_opaque_translucent_closesthit.h>
#include <rtx_shaders/gbuffer_material_rayPortal_closesthit.h>
#include <rtx_shaders/gbuffer_nrc_material_opaque_translucent_closesthit.h>
#include <rtx_shaders/gbuffer_nrc_material_rayPortal_closesthit.h>
#include <rtx_shaders/gbuffer_psr_material_opaque_translucent_closesthit.h>
#include <rtx_shaders/gbuffer_psr_material_rayPortal_closesthit.h>
#include <rtx_shaders/gbuffer_psr_nrc_material_opaque_translucent_closesthit.h>
#include <rtx_shaders/gbuffer_psr_nrc_material_rayPortal_closesthit.h>

#include <rtx_shaders/gbuffer_miss_wboit.h>
#include <rtx_shaders/gbuffer_nrc_miss_wboit.h>
#include <rtx_shaders/gbuffer_psr_miss_wboit.h>
#include <rtx_shaders/gbuffer_psr_nrc_miss_wboit.h>

#include <rtx_shaders/gbuffer_material_opaque_translucent_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_material_rayPortal_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_nrc_material_opaque_translucent_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_nrc_material_rayPortal_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_psr_material_opaque_translucent_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_psr_material_rayPortal_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_psr_nrc_material_opaque_translucent_closesthit_wboit.h>
#include <rtx_shaders/gbuffer_psr_nrc_material_rayPortal_closesthit_wboit.h>

#include "dxvk_scoped_annotation.h"
#include "rtx_context.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_sparse_rendering.h"

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class GbufferRayGenShader : public ManagedShader {
    public:
      BINDLESS_ENABLED()

      PUSH_CONSTANTS(GbufferPushConstants)

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        SAMPLER(GBUFFER_BINDING_LINEAR_WRAP_SAMPLER)

        SAMPLER3D(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT)
        SAMPLER3D(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_FLAGS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_RADIANCE_RG_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_RADIANCE_B_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_INTEGRATION_SURFACE_PDF_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_MATERIAL_DATA0_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_MATERIAL_DATA1_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_TEXTURE_COORD_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_SURFACE_INDEX_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_SUBSURFACE_DATA_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_OUTPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_ATTENUATION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_WORLD_SHADING_NORMAL_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_LINEAR_VIEW_Z_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_ALBEDO_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_BASE_REFLECTIVITY_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_VIRTUAL_MVEC_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_SCREEN_SPACE_MOTION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_DENOISING_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_HIT_DISTANCE_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_VIEW_DIRECTION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_CONE_RADIUS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_POSITION_ERROR_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_SHADOW_TERMINATOR_FIX_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_OBJECT_PICKING_OUTPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_ATTENUATION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_WORLD_SHADING_NORMAL_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_PERCEPTUAL_ROUGHNESS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_LINEAR_VIEW_Z_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_ALBEDO_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_BASE_REFLECTIVITY_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_VIRTUAL_MVEC_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_VIRTUAL_WORLD_SHADING_NORMAL_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_VIRTUAL_WORLD_SHADING_NORMAL_DENOISING_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_HIT_DISTANCE_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_VIEW_DIRECTION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_CONE_RADIUS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_WORLD_POSITION_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SECONDARY_POSITION_ERROR_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_SURFACE_FLAGS_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_DISOCCLUSION_THRESHOLD_MIX_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_DEPTH_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_SHARED_BIAS_CURRENT_COLOR_MASK_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_ALPHA_BLEND_GBUFFER_OUTPUT)
        SAMPLER2D(GBUFFER_BINDING_SKYMATTE)
        SAMPLERCUBE(GBUFFER_BINDING_SKYPROBE)

        RW_TEXTURE2D(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_0)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_WORLD_POSITION_OUTPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_1)

        RW_TEXTURE2D(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_2)
        RW_TEXTURE2D(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_0)
        RW_TEXTURE2D(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_1)
        RW_TEXTURE2D(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_2)
        RW_TEXTURE2D(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_3)

        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_DEPTH_DLSSRR_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_NORMAL_DLSSRR_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_PRIMARY_SCREEN_SPACE_MOTION_DLSSRR_OUTPUT)

        RW_STRUCTURED_BUFFER(GBUFFER_BINDING_NRC_QUERY_PATH_INFO_OUTPUT)
        RW_STRUCTURED_BUFFER(GBUFFER_BINDING_NRC_TRAINING_PATH_INFO_OUTPUT)
        RW_STRUCTURED_BUFFER(GBUFFER_BINDING_NRC_TRAINING_PATH_VERTICES_OUTPUT)
        RW_STRUCTURED_BUFFER(GBUFFER_BINDING_NRC_QUERY_RADIANCE_PARAMS_OUTPUT)
        RW_STRUCTURED_BUFFER(GBUFFER_BINDING_NRC_COUNTERS_OUTPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_NRC_QUERY_PATH_DATA0_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_NRC_QUERY_PATH_DATA1_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_NRC_TRAINING_PATH_DATA1_OUTPUT)

        RW_TEXTURE2D(GBUFFER_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_RG_OUTPUT)
        RW_TEXTURE2D(GBUFFER_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_B_OUTPUT)

      END_PARAMETER()
    };

    class GbufferClosestHitShader : public ManagedShader {

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    class GbufferMissShader : public ManagedShader {
      
      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    constexpr uint32_t kGbufferVariantPSR = 1u << 0;
    constexpr uint32_t kGbufferVariantWBOIT = 1u << 1;
    constexpr uint32_t kGbufferVariantSER = 1u << 2;
    constexpr uint32_t kGbufferVariantPortals = 1u << 3;
    constexpr uint32_t kGbufferVariantLeanPSRFeatures = 1u << 4;
    constexpr uint32_t kGbufferVariantLeanNoPSRFeatures = 1u << 5;
    constexpr uint32_t kGbufferVariantDebugFeatures = 1u << 6;
    constexpr uint32_t kGbufferVariantNRC = 1u << 7;

    constexpr uint32_t kGbufferDecalPrepareVariantDebug = 1u << 0;

    constexpr uint32_t kGbufferRayQueryFeatureRaytracedRenderTarget = 1u << 0;
    constexpr uint32_t kGbufferRayQueryFeatureStochasticAlpha = 1u << 1;
    constexpr uint32_t kGbufferRayQueryFeatureVariantPass = 1u << 8;
    constexpr uint32_t kGbufferRayQueryFeatureVariantWBOIT = 1u << 9;
    constexpr uint32_t kGbufferRayQueryFeatureVariantNRC = 1u << 10;
    constexpr uint32_t kGbufferRayQueryFeatureVariantCount =
      RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_DEBUG + 1u;

    static_assert(RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_NO_PSR == 0u);
    static_assert(RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_INLINE == 4u);
    static_assert(RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_PREPARE == 8u);
    static_assert(RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_DEBUG == 10u);

    constexpr uint32_t getGbufferFeatureVariantKey(
      const uint32_t features) {
      switch (features) {
      case RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_DEBUG:
        return kGbufferVariantDebugFeatures;
      case RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_PSR:
        return kGbufferVariantLeanPSRFeatures;
      case RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_NO_PSR:
        return kGbufferVariantLeanNoPSRFeatures;
      default:
        return 0u;
      }
    }

    constexpr uint32_t getGbufferMatrixVariantKey(
      const uint32_t features,
      const uint32_t psr,
      const uint32_t nrc,
      const uint32_t wboit) {
      return
        getGbufferFeatureVariantKey(features) |
        (psr == RTX_SHADER_VARIANT_MATRIX_GBUFFER_PSR_PSR ? kGbufferVariantPSR : 0u) |
        (nrc == RTX_SHADER_VARIANT_MATRIX_GBUFFER_NRC_NRC ? kGbufferVariantNRC : 0u) |
        (wboit == RTX_SHADER_VARIANT_MATRIX_GBUFFER_WBOIT_WBOIT ? kGbufferVariantWBOIT : 0u);
    }

    constexpr uint32_t getGbufferVariantKey(
      const uint32_t features,
      const bool isPSRPass,
      const bool nrcEnabled,
      const bool wboitEnabled) {
      return
        getGbufferFeatureVariantKey(features) |
        (isPSRPass ? kGbufferVariantPSR : 0u) |
        (nrcEnabled ? kGbufferVariantNRC : 0u) |
        (wboitEnabled ? kGbufferVariantWBOIT : 0u);
    }

    constexpr uint32_t getGbufferPassVariantKey(
      const bool isPSRPass,
      const bool nrcEnabled,
      const bool wboitEnabled) {
      return getGbufferVariantKey(
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_NONE,
        isPSRPass,
        nrcEnabled,
        wboitEnabled);
    }

    constexpr uint32_t selectGbufferFeatureVariant(
      const bool debugViewEnabled,
      const bool objectPickingEnabled,
      const bool psrEnabled,
      const bool useDefaultFeatureVariant) {
      if (debugViewEnabled || objectPickingEnabled) {
        return RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_DEBUG;
      }

      if (useDefaultFeatureVariant) {
        return RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_NONE;
      }

      return psrEnabled
        ? RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_PSR
        : RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_NO_PSR;
    }

    constexpr uint32_t getGbufferRayQueryFeatureMask(
      const bool raytracedRenderTargetEnabled,
      const bool stochasticAlphaEnabled) {
      return
        (raytracedRenderTargetEnabled ? kGbufferRayQueryFeatureRaytracedRenderTarget : 0u) |
        (stochasticAlphaEnabled ? kGbufferRayQueryFeatureStochasticAlpha : 0u);
    }

    constexpr uint32_t getGbufferRayQueryPSRPrepareFeatureMask(
      const uint32_t featureMask) {
      return (featureMask & kGbufferRayQueryFeatureStochasticAlpha) >> 1u;
    }

    constexpr uint32_t selectGbufferRayQueryFeatureVariant(
      const bool debugViewEnabled,
      const bool objectPickingEnabled,
      const bool psrEnabled,
      const bool usePSRPrepare,
      const bool raytracedRenderTargetEnabled,
      const bool stochasticAlphaEnabled) {
      if (debugViewEnabled || objectPickingEnabled) {
        return RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_DEBUG;
      }

      const uint32_t featureMask = getGbufferRayQueryFeatureMask(
        raytracedRenderTargetEnabled,
        stochasticAlphaEnabled);

      if (!psrEnabled) {
        return RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_NO_PSR + featureMask;
      }

      if (usePSRPrepare) {
        return RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_PREPARE +
          getGbufferRayQueryPSRPrepareFeatureMask(featureMask);
      }

      return RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_INLINE + featureMask;
    }

    constexpr uint32_t getGbufferTraceRayGenVariantKey(
      const uint32_t features,
      const uint32_t psr,
      const uint32_t pipeline,
      const uint32_t nrc,
      const uint32_t wboit) {
      return getGbufferMatrixVariantKey(features, psr, nrc, wboit) |
        (pipeline == RTX_SHADER_VARIANT_MATRIX_GBUFFER_PIPELINE_RAYGEN_SER ? kGbufferVariantSER : 0u);
    }

    constexpr uint32_t getGbufferRayQueryMatrixVariantKey(
      const uint32_t features,
      const uint32_t pass,
      const uint32_t nrc,
      const uint32_t wboit) {
      return
        features |
        (pass == RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_PASS_PSR ? kGbufferRayQueryFeatureVariantPass : 0u) |
        (nrc == RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_NRC_NRC ? kGbufferRayQueryFeatureVariantNRC : 0u) |
        (wboit == RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_WBOIT_WBOIT ? kGbufferRayQueryFeatureVariantWBOIT : 0u);
    }

    constexpr uint32_t getGbufferRayQueryVariantKey(
      const uint32_t features,
      const bool isPSRPass,
      const bool nrcEnabled,
      const bool wboitEnabled) {
      return
        features |
        (isPSRPass ? kGbufferRayQueryFeatureVariantPass : 0u) |
        (nrcEnabled ? kGbufferRayQueryFeatureVariantNRC : 0u) |
        (wboitEnabled ? kGbufferRayQueryFeatureVariantWBOIT : 0u);
    }

    constexpr uint32_t getGbufferDecalPrepareMatrixVariantKey(
      const uint32_t debug) {
      return
        (debug == RTX_SHADER_VARIANT_MATRIX_GBUFFER_DECAL_PREPARE_DEBUG_DEBUG ? kGbufferDecalPrepareVariantDebug : 0u);
    }

    constexpr uint32_t getGbufferDecalPrepareVariantKey(
      const bool debugViewEnabled,
      const bool objectPickingEnabled) {
      return
        (debugViewEnabled || objectPickingEnabled ? kGbufferDecalPrepareVariantDebug : 0u);
    }

    Rc<DxvkShader> getGbufferRayQueryRayGenShader(const uint32_t key) {
      switch (key) {
#define GBUFFER_RAYQUERY_RAYGEN_CASE(features, psr, pipeline, nrc, wboit, code) \
        case getGbufferMatrixVariantKey(features, psr, nrc, wboit): \
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, GbufferRayGenShader, code);
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_PIPELINE_RAYQUERY_RAYGEN_RGEN(GBUFFER_RAYQUERY_RAYGEN_CASE)
#undef GBUFFER_RAYQUERY_RAYGEN_CASE
        default:
          assert(false && "Invalid GBuffer RayQuery raygen shader variant");
          return nullptr;
      }
    }

    Rc<DxvkShader> getGbufferTraceRayGenShader(const uint32_t key) {
      switch (key) {
#define GBUFFER_TRACE_RAYGEN_CASE(features, psr, pipeline, nrc, wboit, code) \
        case getGbufferTraceRayGenVariantKey(features, psr, pipeline, nrc, wboit): \
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, GbufferRayGenShader, code);
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_PIPELINE_RAYGEN_RGEN(GBUFFER_TRACE_RAYGEN_CASE)
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_PIPELINE_RAYGEN_SER_RGEN(GBUFFER_TRACE_RAYGEN_CASE)
#undef GBUFFER_TRACE_RAYGEN_CASE
        default:
          assert(false && "Invalid GBuffer TraceRay raygen shader variant");
          return nullptr;
      }
    }

    Rc<DxvkShader> getGbufferRayQueryComputeShader(const uint32_t key) {
      switch (key) {
#define GBUFFER_RAYQUERY_COMPUTE_CASE(features, pass, nrc, wboit, code) \
        case getGbufferRayQueryMatrixVariantKey(features, pass, nrc, wboit): \
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, code);
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_PASS_NONE_COMP(GBUFFER_RAYQUERY_COMPUTE_CASE)
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_PASS_PSR_COMP(GBUFFER_RAYQUERY_COMPUTE_CASE)
#undef GBUFFER_RAYQUERY_COMPUTE_CASE
        default:
          assert(false && "Invalid GBuffer RayQuery compute shader variant");
          return nullptr;
      }
    }

    Rc<DxvkShader> getGbufferPSRPrepareComputeShader() {
      return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_prepare);
    }

    Rc<DxvkShader> getGbufferDecalPrepareComputeShader(const uint32_t key) {
      switch (key) {
#define GBUFFER_DECAL_PREPARE_CASE(debug, code) \
        case getGbufferDecalPrepareMatrixVariantKey(debug): \
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, code);
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_DECAL_PREPARE_DEBUG_NONE_COMP(GBUFFER_DECAL_PREPARE_CASE)
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_DECAL_PREPARE_DEBUG_DEBUG_COMP(GBUFFER_DECAL_PREPARE_CASE)
#undef GBUFFER_DECAL_PREPARE_CASE
        default:
          assert(false && "Invalid GBuffer decal prepare shader variant");
          return nullptr;
      }
    }

    void prewarmGbufferDecalPrepareComputeShaders() {
      for (uint32_t key = 0; key < (1u << 1); key++) {
        getGbufferDecalPrepareComputeShader(key);
      }
    }

    void prewarmGbufferRayQueryComputeShaders(
      const bool allVariants,
      const bool nrcEnabled,
      const bool wboitEnabled) {
      if (allVariants) {
        for (uint32_t featureVariant = 0; featureVariant < kGbufferRayQueryFeatureVariantCount; featureVariant++) {
          for (int32_t isPSRPass = 1; isPSRPass >= 0; isPSRPass--) {
            for (int32_t useNRC = nrcEnabled; useNRC >= 0; useNRC--) {
              for (int32_t useWBOIT = 1; useWBOIT >= 0; useWBOIT--) {
                getGbufferRayQueryComputeShader(getGbufferRayQueryVariantKey(featureVariant, isPSRPass, useNRC, useWBOIT));
              }
            }
          }
        }

        return;
      }

      const uint32_t featureVariants[] = {
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_NO_PSR,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_NO_PSR + 3u,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_INLINE,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_INLINE + 3u,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_PREPARE,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_PSR_PREPARE + 1u,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_RAYQUERY_FEATURES_DEBUG,
      };

      for (const uint32_t featureVariant : featureVariants) {
        for (int32_t isPSRPass = 1; isPSRPass >= 0; isPSRPass--) {
          for (int32_t useNRC = nrcEnabled; useNRC >= 0; useNRC--) {
            getGbufferRayQueryComputeShader(getGbufferRayQueryVariantKey(featureVariant, isPSRPass, useNRC, wboitEnabled));
          }
        }
      }
    }

    Rc<DxvkShader> getGbufferMissShader(const uint32_t key) {
      switch (key) {
      case 0:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_miss);
      case kGbufferVariantPSR:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_psr_miss);
      case kGbufferVariantNRC:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_nrc_miss);
      case kGbufferVariantPSR | kGbufferVariantNRC:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_psr_nrc_miss);
      case kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_miss_wboit);
      case kGbufferVariantPSR | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_psr_miss_wboit);
      case kGbufferVariantNRC | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_nrc_miss_wboit);
      case kGbufferVariantPSR | kGbufferVariantNRC | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, GbufferMissShader, gbuffer_psr_nrc_miss_wboit);
      default:
        assert(false && "Invalid GBuffer miss shader variant");
        return nullptr;
      }
    }

    Rc<DxvkShader> getGbufferClosestHitShader(const uint32_t key) {
      switch (key) {
      case 0:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_material_opaque_translucent_closestHit);
      case kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_material_rayportal_closestHit);
      case kGbufferVariantNRC:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_nrc_material_opaque_translucent_closestHit);
      case kGbufferVariantNRC | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_nrc_material_rayportal_closestHit);
      case kGbufferVariantPSR:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_material_opaque_translucent_closestHit);
      case kGbufferVariantPSR | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_material_rayportal_closestHit);
      case kGbufferVariantPSR | kGbufferVariantNRC:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_nrc_material_opaque_translucent_closestHit);
      case kGbufferVariantPSR | kGbufferVariantNRC | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_nrc_material_rayportal_closestHit);
      case kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_material_opaque_translucent_closestHit_wboit);
      case kGbufferVariantWBOIT | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_material_rayportal_closestHit_wboit);
      case kGbufferVariantNRC | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_nrc_material_opaque_translucent_closestHit_wboit);
      case kGbufferVariantNRC | kGbufferVariantWBOIT | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_nrc_material_rayportal_closestHit_wboit);
      case kGbufferVariantPSR | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_material_opaque_translucent_closestHit_wboit);
      case kGbufferVariantPSR | kGbufferVariantWBOIT | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_material_rayportal_closestHit_wboit);
      case kGbufferVariantPSR | kGbufferVariantNRC | kGbufferVariantWBOIT:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_nrc_material_opaque_translucent_closestHit_wboit);
      case kGbufferVariantPSR | kGbufferVariantNRC | kGbufferVariantWBOIT | kGbufferVariantPortals:
        return GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GbufferClosestHitShader, gbuffer_psr_nrc_material_rayportal_closestHit_wboit);
      default:
        assert(false && "Invalid GBuffer closest hit shader variant");
        return nullptr;
      }
    }
  }

  DxvkPathtracerGbuffer::DxvkPathtracerGbuffer(DxvkDevice* device) : CommonDeviceObject(device) {
  }

  void DxvkPathtracerGbuffer::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    ScopedCpuProfileZoneN("Gbuffer Shader Prewarming");

    const bool isNrcSupported = NeuralRadianceCache::checkIsSupported(device());
    const bool isOpacityMicromapSupported = OpacityMicromapManager::checkIsOpacityMicromapSupported(*m_device);
    const bool isShaderExecutionReorderingSupported = 
      RtxContext::checkIsShaderExecutionReorderingSupported(*m_device) && 
      RtxOptions::isShaderExecutionReorderingInPathtracerGbufferEnabled();
    const bool portalsEnabled = RtxOptions::rayPortalModelTextureHashes().size() > 0;

    if (RtxOptions::Shader::prewarmAllVariants()) {
      const uint32_t featureVariants[] = {
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_NONE,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_DEBUG,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_PSR,
        RTX_SHADER_VARIANT_MATRIX_GBUFFER_FEATURES_LEAN_NO_PSR,
      };

      for (const uint32_t featureVariant : featureVariants) {
        for (int32_t isPSRPass = 1; isPSRPass >= 0; isPSRPass--) {
          for (int32_t nrcEnabled = isNrcSupported; nrcEnabled >= 0; nrcEnabled--) {
            for (int32_t wboitEnabled = 1; wboitEnabled >= 0; wboitEnabled--) {
              for (int32_t includePortals = portalsEnabled; includePortals >= 0; includePortals--) {
                for (int32_t useRayQuery = 1; useRayQuery >= 0; useRayQuery--) {
                  for (int32_t serEnabled = isShaderExecutionReorderingSupported; serEnabled >= 0; serEnabled--) {
                    for (int32_t ommEnabled = isOpacityMicromapSupported; ommEnabled >= 0; ommEnabled--) {
                      pipelineManager.registerRaytracingShaders(getPipelineShaders(featureVariant, isPSRPass, useRayQuery, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
                    }
                  }
                }
              }
            }

            // Both portal permutations of the compute shader, matching the
            // includePortals loop the rt-pipeline registration above walks.
            getComputeShader(isPSRPass, nrcEnabled, wboitEnabled, true);
            getComputeShader(isPSRPass, nrcEnabled, wboitEnabled, false);
          }
        }
      }

      prewarmGbufferRayQueryComputeShaders(true, isNrcSupported, false);
      getGbufferPSRPrepareComputeShader();
      prewarmGbufferDecalPrepareComputeShaders();
    } else {
      // Note: The getters for these SER/OMM enabled flags also check if SER/OMMs are supported, so we do not need to check for that manually.
      const bool serEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerGbufferEnabled();
      const bool ommEnabled = RtxOptions::getEnableOpacityMicromap();
      const bool nrcEnabled = RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::NeuralRadianceCache;
      const bool wboitEnabled = RtxOptions::wboitEnabled();

      // Need both PSR and non-PSR passes.
      for (int32_t isPSRPass = 1; isPSRPass >= 0; isPSRPass--) {
        for (int32_t includePortals = portalsEnabled; includePortals >= 0; includePortals--) {
          DxvkComputePipelineShaders shaders;
          switch (RtxOptions::renderPassGBufferRaytraceMode()) {
          case RaytraceMode::RayQuery:
            // Mirror dispatch()'s predicate exactly, or prewarming warms the
            // permutation the frame will not use and the first real dispatch
            // pays a pipeline compile.
            getComputeShader(isPSRPass, nrcEnabled, wboitEnabled,
                             includePortals != 0
                               || (RtxOptions::useIntersectionBillboardsOnPrimaryRays()
                                   && RtxOptions::enableBillboardOrientationCorrection()));
            break;
          case RaytraceMode::RayQueryRayGen:
            pipelineManager.registerRaytracingShaders(getPipelineShaders(isPSRPass, true, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
            break;
          case RaytraceMode::TraceRay:
            pipelineManager.registerRaytracingShaders(getPipelineShaders(isPSRPass, false, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
            break;
          case RaytraceMode::Count:
            assert(false && "Invalid RaytraceMode in DxvkPathtracerGbuffer::prewarmShaders");
            break;
          }
        }
        break;
      case RaytraceMode::Count:
        assert(false && "Invalid RaytraceMode in DxvkPathtracerGbuffer::prewarmShaders");
        break;
      }

      if (!portalsEnabled) {
        getGbufferPSRPrepareComputeShader();
      }

      prewarmGbufferDecalPrepareComputeShaders();
    }
  }

  void DxvkPathtracerGbuffer::dispatch(
    RtxContext* ctx, 
    const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Gbuffer Raytracing");
    ctx->setFramePassStage(RtxFramePassStage::GBufferPrimaryRays);

    // Bind resources

    ctx->bindCommonRayTracingResources(rtOutput);

    // Note: Clamp to edge used to avoid interpolation to black on the edges of the view.
    Rc<DxvkSampler> linearClampSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    Rc<DxvkSampler> linearWrapSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    ctx->bindResourceSampler(GBUFFER_BINDING_LINEAR_WRAP_SAMPLER, linearWrapSampler);

    const RtxGlobalVolumetrics& globalVolumetrics = ctx->getCommonObjects()->metaGlobalVolumetrics();
    ctx->bindResourceView(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT, globalVolumetrics.getCurrentVolumeAccumulatedRadianceY().view, nullptr);
    ctx->bindResourceSampler(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT, linearClampSampler);
    ctx->bindResourceView(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT, globalVolumetrics.getCurrentVolumeAccumulatedRadianceCoCg().view, nullptr);
    ctx->bindResourceSampler(GBUFFER_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT, linearClampSampler);

    ctx->bindResourceView(GBUFFER_BINDING_SKYMATTE, ctx->getResourceManager().getSkyMatte(ctx).view, nullptr);
    ctx->bindResourceSampler(GBUFFER_BINDING_SKYMATTE, linearClampSampler);

    // Requires the probe too for PSRR/T miss
    ctx->bindResourceView(GBUFFER_BINDING_SKYPROBE, ctx->getResourceManager().getSkyProbe(ctx).view, nullptr);
    ctx->bindResourceSampler(GBUFFER_BINDING_SKYPROBE, linearClampSampler);

    // Output resources

    ctx->bindResourceView(GBUFFER_BINDING_SHARED_FLAGS_OUTPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_RADIANCE_RG_OUTPUT, rtOutput.m_sharedRadianceRG.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_RADIANCE_B_OUTPUT, rtOutput.m_sharedRadianceB.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_INTEGRATION_SURFACE_PDF_OUTPUT, rtOutput.m_sharedIntegrationSurfacePdf.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_MATERIAL_DATA0_OUTPUT, rtOutput.m_sharedMaterialData0.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_MATERIAL_DATA1_OUTPUT, rtOutput.m_sharedMaterialData1.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_OUTPUT, rtOutput.m_sharedMediumMaterialIndex.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_BIAS_CURRENT_COLOR_MASK_OUTPUT, rtOutput.m_sharedBiasCurrentColorMask.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_TEXTURE_COORD_OUTPUT, rtOutput.m_sharedTextureCoord.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_SURFACE_INDEX_OUTPUT, rtOutput.m_sharedSurfaceIndex.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_SUBSURFACE_DATA_OUTPUT, rtOutput.m_sharedSubsurfaceData.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_OUTPUT, rtOutput.m_sharedSubsurfaceDiffusionProfileData.view, nullptr);

    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_ATTENUATION_OUTPUT, rtOutput.m_primaryAttenuation.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_WORLD_SHADING_NORMAL_OUTPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_OUTPUT, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_LINEAR_VIEW_Z_OUTPUT, rtOutput.m_primaryLinearViewZ.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_ALBEDO_OUTPUT, rtOutput.m_primaryAlbedo.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_BASE_REFLECTIVITY_OUTPUT, rtOutput.m_primaryBaseReflectivity.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_VIRTUAL_MVEC_OUTPUT, rtOutput.m_primaryVirtualMotionVector.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_SCREEN_SPACE_MOTION_OUTPUT, rtOutput.m_primaryScreenSpaceMotionVector.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_OUTPUT, rtOutput.m_primaryVirtualWorldShadingNormalPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_DENOISING_OUTPUT, rtOutput.m_primaryVirtualWorldShadingNormalPerceptualRoughnessDenoising.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_HIT_DISTANCE_OUTPUT, rtOutput.m_primaryHitDistance.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_VIEW_DIRECTION_OUTPUT, rtOutput.m_primaryViewDirection.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_CONE_RADIUS_OUTPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_POSITION_ERROR_OUTPUT, rtOutput.m_primaryPositionError.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SHARED_SHADOW_TERMINATOR_FIX_OUTPUT, rtOutput.getCurrentSharedTerminatorFix().view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_SURFACE_FLAGS_OUTPUT, rtOutput.m_primarySurfaceFlags.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_DISOCCLUSION_THRESHOLD_MIX_OUTPUT, rtOutput.m_primaryDisocclusionThresholdMix.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_DEPTH_OUTPUT, rtOutput.m_primaryDepth.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_OBJECT_PICKING_OUTPUT, rtOutput.m_primaryObjectPicking.view, nullptr);

    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_ATTENUATION_OUTPUT, rtOutput.m_secondaryAttenuation.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_WORLD_SHADING_NORMAL_OUTPUT, rtOutput.m_secondaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_PERCEPTUAL_ROUGHNESS_OUTPUT, rtOutput.m_secondaryPerceptualRoughness.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_LINEAR_VIEW_Z_OUTPUT, rtOutput.m_secondaryLinearViewZ.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_ALBEDO_OUTPUT, rtOutput.m_secondaryAlbedo.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_BASE_REFLECTIVITY_OUTPUT, rtOutput.m_secondaryBaseReflectivity.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_VIRTUAL_MVEC_OUTPUT, rtOutput.m_secondaryVirtualMotionVector.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_VIRTUAL_WORLD_SHADING_NORMAL_OUTPUT, rtOutput.m_secondaryVirtualWorldShadingNormalPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_VIRTUAL_WORLD_SHADING_NORMAL_DENOISING_OUTPUT, rtOutput.m_secondaryVirtualWorldShadingNormalPerceptualRoughnessDenoising.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_HIT_DISTANCE_OUTPUT, rtOutput.m_secondaryHitDistance.view, nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_VIEW_DIRECTION_OUTPUT, rtOutput.m_secondaryViewDirection.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_CONE_RADIUS_OUTPUT, rtOutput.m_secondaryConeRadius.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_POSITION_ERROR_OUTPUT, rtOutput.m_secondaryPositionError.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_SECONDARY_WORLD_POSITION_OUTPUT, rtOutput.m_secondaryWorldPositionWorldTriangleNormal.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_ALPHA_BLEND_GBUFFER_OUTPUT, rtOutput.m_alphaBlendGBuffer.view, nullptr);

    ctx->bindResourceView(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_0, rtOutput.m_gbufferPSRData[0].view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_WORLD_POSITION_OUTPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Write), nullptr);

    ctx->bindResourceView(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_1, rtOutput.m_gbufferPSRData[1].view(Resources::AccessType::Write), nullptr);

    // Note: m_gbufferPSRData[2..6] are aliased with various radiance textures that are used later as integrator outputs.
    ctx->bindResourceView(GBUFFER_BINDING_REFLECTION_PSR_DATA_STORAGE_2, rtOutput.m_gbufferPSRData[2].view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_0, rtOutput.m_gbufferPSRData[3].view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_1, rtOutput.m_gbufferPSRData[4].view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_2, rtOutput.m_gbufferPSRData[5].view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_TRANSMISSION_PSR_DATA_STORAGE_3, rtOutput.m_gbufferPSRData[6].view(Resources::AccessType::Write), nullptr);

    // Bind necessary buffers for DLSS-RR. 
    // Note: RR uses different PSR rules compared to other uses, and its resolves are resolved in an another shader.
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_DEPTH_DLSSRR_OUTPUT, rtOutput.m_primaryDepthDLSSRR.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_NORMAL_DLSSRR_OUTPUT, rtOutput.m_primaryWorldShadingNormalDLSSRR.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(GBUFFER_BINDING_PRIMARY_SCREEN_SPACE_MOTION_DLSSRR_OUTPUT, rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR.view, nullptr);

    // Bind necessary resources for Neural Radiance Cache
    NeuralRadianceCache& nrc = ctx->getCommonObjects()->metaNeuralRadianceCache();
    nrc.bindGBufferPathTracingResources(*ctx);    
  
    const VkExtent3D& rayDims = rtOutput.m_compositeOutputExtent;

    const bool nrcEnabled = nrc.isActive();
    const bool serEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerGbufferEnabled();
    const bool ommEnabled = RtxOptions::getEnableOpacityMicromap();
    const bool includePortals = RtxOptions::rayPortalModelTextureHashes().size() > 0 || rtOutput.m_raytraceArgs.numActiveRayPortals > 0;
    const bool wboitEnabled = RtxOptions::wboitEnabled();
    const bool psrEnabled = rtOutput.m_raytraceArgs.enablePSRR || rtOutput.m_raytraceArgs.enablePSTR;
    const bool debugViewEnabled = rtOutput.m_raytraceArgs.debugView != 0;
    const bool usePipelineDefaultFeatureVariant =
      rtOutput.m_raytraceArgs.enableDLSSRR ||
      rtOutput.m_raytraceArgs.enableRaytracedRenderTarget ||
      rtOutput.m_raytraceArgs.enableStochasticAlphaBlend;
    const uint32_t pipelineFeatureVariant = selectGbufferFeatureVariant(
      debugViewEnabled,
      rtOutput.m_raytraceArgs.enableObjectPicking,
      psrEnabled,
      usePipelineDefaultFeatureVariant);
    const bool deferUnorderedDecals =
      rtOutput.m_raytraceArgs.enableSeparateUnorderedApproximations &&
      rtOutput.m_raytraceArgs.enableDecalMaterialBlending;
    const uint32_t decalPrepareVariantKey = getGbufferDecalPrepareVariantKey(
      debugViewEnabled,
      rtOutput.m_raytraceArgs.enableObjectPicking);
    // Keep NRC on the original inline PSR sampling path. NRC updates depend
    // on knowing whether the current GBuffer hit is the final integrated
    // surface, which PSR prepare defers to a later pass.
    const bool usePSRPrepare =
      psrEnabled &&
      !nrcEnabled &&
      !rtOutput.m_raytraceArgs.enableRaytracedRenderTarget;
    const uint32_t rayQueryFeatureVariant = selectGbufferRayQueryFeatureVariant(
      debugViewEnabled,
      rtOutput.m_raytraceArgs.enableObjectPicking,
      psrEnabled,
      usePSRPrepare,
      rtOutput.m_raytraceArgs.enableRaytracedRenderTarget,
      rtOutput.m_raytraceArgs.enableStochasticAlphaBlend);
    const VkExtent3D workgroups = util::computeBlockCount(rayDims, VkExtent3D { 16, 8, 1 });

    // NV-DXVK [perf]: which resolve types the COMPUTE shader must be built with.
    //
    // The narrow (opaque+translucent) build sets RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
    // on the unordered ray and hardcodes useIntersectionBillboards to false. For a
    // portals-off scene that is already the runtime behaviour — geometry_resolver's
    // useIntersectionBillboards is `enableBillboardOrientationCorrection &&
    // (directionAltered || useIntersectionBillboardsOnPrimaryRays)`, directionAltered
    // is only ever set by the ray portal path, and the ray mask picks
    // OBJECT_MASK_UNORDERED_ALL_GEOMETRY (the triangle representation of the same
    // billboards) in that case. So nothing is dropped.
    //
    // The one exception is the dev-only "Use i-prims on primary rays" checkbox,
    // which asks for the intersection-primitive representation with no portals in
    // the scene. Keep the full build in that case rather than silently ignoring the
    // option. (getPipelineShaders() keys only off includePortals, so the rt-pipeline
    // modes do not honour that checkbox either; not changing existing behaviour
    // there, but the compute path should not acquire the same trap.)
    const bool needsIntersectionPrimitives =
      RtxOptions::useIntersectionBillboardsOnPrimaryRays() && RtxOptions::enableBillboardOrientationCorrection();
    const bool computeIncludePortals = includePortals || needsIntersectionPrimitives;

    GbufferPushConstants pushArgs = {};
    pushArgs.isTransmissionPSR = 0;
    pushArgs.usePSRPrepare = usePSRPrepare;
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

    // The main GBuffer pass now leaves some uncommon work for small follow-up
    // passes. This keeps the hot Primary Rays shader variants from carrying
    // PSR sampling or unordered decal blending when those features are inactive
    // or rare.
    auto dispatchPSRPrepare = [&]() {
      if (!usePSRPrepare) {
        return;
      }

      // Classify PSR candidate pixels and serialize the PSR payloads used by
      // the Reflection/Transmission PSR passes. This avoids running the
      // material-heavy PSR sampling path for every primary hit.
      ScopedGpuProfileZone(ctx, "PSR Prepare");
      pushArgs.isTransmissionPSR = 0;
      pushArgs.usePSRPrepare = 0;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, getGbufferPSRPrepareComputeShader());
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    };

    auto dispatchDeferredDecals = [&]() {
      if (!deferUnorderedDecals) {
        return;
      }

      // Apply unordered decal material blending after the main GBuffer pass.
      // Decals are sparse, so keeping their sampling/blending out of Primary
      // Rays reduces register pressure in the common path.
      ScopedGpuProfileZone(ctx, "Unordered Decal Prepare");
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, getGbufferDecalPrepareComputeShader(decalPrepareVariantKey));
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    };

    // NV-DXVK [perf]: the primary-ray dispatch is ~130 ms/frame ([Perf.GpuPass]
    // gb_primaryRays) and has proven invariant to every workload option — bounce
    // count, integrator, PSR, resolver interactions, BVH quality. The two inputs
    // never actually verified are how many threads it launches and which shader
    // permutation it launches, both of which set the cost directly. Logged once
    // per change, so it is silent at steady state.
    // NV-DXVK [perf]: diagnostic ray-count scale, see perfPrimaryRayGridScale.
    // Applied to the primary-ray launch only, and clamped so a mistyped value
    // cannot produce a zero-thread dispatch (which would read as "the cost
    // vanished" and be badly misleading).
    const float rayGridScale = std::max(0.01f, std::min(1.0f, perfPrimaryRayGridScale()));
    VkExtent3D primaryRayDims = rayDims;
    if (rayGridScale < 1.0f) {
      primaryRayDims.width  = std::max(1u, uint32_t(float(rayDims.width)  * rayGridScale));
      primaryRayDims.height = std::max(1u, uint32_t(float(rayDims.height) * rayGridScale));
    }

    {
      static VkExtent3D s_lastDims = { 0, 0, 0 };
      static uint32_t   s_lastPerm = UINT32_MAX;
      static float      s_lastScale = -1.0f;
      // perfGbStopAfter must be part of the change key for the same reason the
      // raytrace mode is: it silently changes what the dispatch does, so a
      // mistyped or unapplied value would otherwise be reported as whatever the
      // previous run used, and the ladder would be read off the wrong rung.
      static uint32_t   s_lastStopAfter = UINT32_MAX;

      // The raytrace mode MUST be part of the change key. It is printed on this
      // line, but it was not tracked, so the line fired once on the first
      // dispatch and never again — and RtxOptions applies config values through a
      // deferred layer, so a mode that takes effect on a later frame was reported
      // for ever after as its frame-one value. That silently invalidated a whole
      // test run.
      const uint32_t modeBits = static_cast<uint32_t>(RtxOptions::renderPassGBufferRaytraceMode()) << 5;

      const uint32_t perm = modeBits
                          | (nrcEnabled ? 1u : 0u)
                          | (serEnabled ? 2u : 0u)
                          | (ommEnabled ? 4u : 0u)
                          | (includePortals ? 8u : 0u)
                          | (wboitEnabled ? 16u : 0u)
                          // Must be in the key: it selects a different compute
                          // shader permutation, so a run that silently used the
                          // full-resolve-type build would otherwise be reported
                          // as the narrow one for the rest of the session.
                          | (computeIncludePortals ? 1024u : 0u);

      const uint32_t stopAfter = RtxOptions::perfGbStopAfter();

      // Same reasoning as stopAfter: the block-size ladder swaps the primary-ray
      // shader for one with a different register cap. A rung that changed without
      // re-logging would have its timing read off the previous rung's line.
      static uint32_t s_lastBlockThreads = UINT32_MAX;
      const uint32_t blockThreads = perfGbufferBlockThreads();

      if (rayDims.width != s_lastDims.width || rayDims.height != s_lastDims.height
       || perm != s_lastPerm || rayGridScale != s_lastScale
       || stopAfter != s_lastStopAfter || blockThreads != s_lastBlockThreads) {
        s_lastDims      = rayDims;
        s_lastPerm      = perm;
        s_lastScale     = rayGridScale;
        s_lastStopAfter = stopAfter;
        s_lastBlockThreads = blockThreads;

        const double mpix = double(rayDims.width) * double(rayDims.height) / 1.0e6;
        const VkExtent3D& downscaled = ctx->getResourceManager().getDownscaleDimensions();
        const VkExtent3D& target     = ctx->getResourceManager().getTargetDimensions();

        // Log where rayDims came from, not just what it is. rayDims is
        // m_compositeOutputExtent, which is *assigned* the downscaled extent —
        // printing the target extent and the resolution scale next to it makes a
        // dispatch that is secretly running at full res self-evident instead of
        // something to be inferred from an option value.
        Logger::warn(str::format(
          "[Perf.GbDispatch] rayDims=", rayDims.width, "x", rayDims.height,
          " (", mpix, " Mpix)",
          " primaryRayDims=", primaryRayDims.width, "x", primaryRayDims.height,
          " (", double(primaryRayDims.width) * double(primaryRayDims.height) / 1.0e6, " Mpix)",
          " gridScale=", rayGridScale,
          " downscaled=", downscaled.width, "x", downscaled.height,
          " target=", target.width, "x", target.height,
          " resScale=", RtxOptions::resolutionScale(),
          " upscaler=", static_cast<uint32_t>(RtxOptions::upscalerType()),
          " mode=", static_cast<uint32_t>(RtxOptions::renderPassGBufferRaytraceMode()),
          " nrc=", (nrcEnabled ? 1 : 0),
          " ser=", (serEnabled ? 1 : 0),
          " omm=", (ommEnabled ? 1 : 0),
          " portals=", (includePortals ? 1 : 0),
          // csPortals=0 means the compute path is running the
          // opaque+translucent build: no ray portal resolve branch and no
          // intersection-billboard branch. Cross-check against the variant name
          // on the [Perf.Shader] cs= line.
          " csPortals=", (computeIncludePortals ? 1 : 0),
          " wboit=", (wboitEnabled ? 1 : 0),
          " gbStopAfter=", stopAfter,
          // 0 = stock 128-thread shader. Cross-check against the variant name on
          // the [Perf.Shader] cs= line and the Register Count it reports.
          " blockThreads=", blockThreads));
      }
    }

    // NV-DXVK [perf]: split the primary-ray bucket into wait-vs-work. markGpuStage
    // writes at BOTTOM_OF_PIPE, so the ~130 ms currently attributed to
    // gb_primaryRays covers everything from the volumetrics mark through this
    // dispatch completing — including the execution barriers DXVK flushes inside
    // dispatch()/traceRays() immediately before issuing it. If the primary-ray
    // shader is genuinely slow, gb_primaryRays keeps the time and gb_bindWait is
    // ~0. If the dispatch is instead stalling on prior GPU work (the acceleration
    // structure build being the obvious candidate), gb_bindWait takes it. That is
    // the difference between a shader problem and a scheduling problem, and no
    // amount of option-toggling can distinguish them.
    ctx->markGpuStageBeforeNextDispatch();

    // Reports which switch branch actually ran, rather than what an option says
    // it should be. Option reads go through a deferred layer and can disagree
    // with the executed path on early frames; this cannot, because it is emitted
    // from inside the branch itself. Logs only on change, so it is silent at
    // steady state.
    auto noteBranch = [](const char* name) {
      static const char* s_last = nullptr;
      if (s_last != name) {
        s_last = name;
        Logger::warn(str::format("[Perf.GbBranch] executing=", name));
      }
    };

    switch (RtxOptions::renderPassGBufferRaytraceMode()) {
    case RaytraceMode::RayQuery: {
      noteBranch("RayQuery(compute)");
      VkExtent3D workgroups = util::computeBlockCount(rayDims, VkExtent3D { 16, 8, 1 });
      {
        ScopedGpuProfileZone(ctx, "Primary Rays");

        // NV-DXVK [Perf.Occupancy]: the block-size ladder replaces the primary-ray
        // shader AND its block extent together. Getting one without the other
        // would either miss pixels or launch redundant blocks, so they are
        // resolved in a single call and the block extent defaults to the stock
        // 16x8 when the ladder is off.
        VkExtent3D primaryBlockSize = VkExtent3D { 16, 8, 1 };
        Rc<DxvkShader> primaryShader =
          getOccupancyLadderShader(nrcEnabled, wboitEnabled, computeIncludePortals, primaryBlockSize);

        if (primaryShader == nullptr) {
          primaryShader = getComputeShader(false, nrcEnabled, wboitEnabled, computeIncludePortals);
        }

        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, primaryShader);
        // Primary rays only — PSR and every later pass keep the full grid so they
        // stay comparable across a perfPrimaryRayGridScale sweep.
        const VkExtent3D primaryWorkgroups =
          util::computeBlockCount(primaryRayDims, primaryBlockSize);
        ctx->dispatch(primaryWorkgroups.width, primaryWorkgroups.height, primaryWorkgroups.depth);
      }
      // Keeps the mark count fixed at 27/frame even if the dispatch was skipped.
      ctx->markGpuStageIfPending();
      // NV-DXVK [perf]: [Perf.GpuPass] pins ~135 ms/frame on this whole
      // function, unchanged by every workload knob tried (secondary bounces,
      // integrator, PSR enables, resolver interactions, BVH quality). It is not
      // one dispatch though — it is three at full resolution, and the two PSR
      // ones are issued even when enablePSRR/enablePSTR are false, since those
      // options only travel to the shader as constants. Split the three so the
      // cost lands on a specific one.
      ctx->markGpuStage();

      dispatchDeferredDecals();
      dispatchPSRPrepare();

      {
        // Warning: do not change the order of Reflection and Transmission PSR, that will break
        // PSR data dependencies due to resource aliasing.
        ScopedGpuProfileZone(ctx, "Reflection PSR");
        ctx->setFramePassStage(RtxFramePassStage::ReflectionPSR);
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, getComputeShader(true, nrcEnabled, wboitEnabled, computeIncludePortals));
        ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
      }
      ctx->markGpuStage();

      {
        ScopedGpuProfileZone(ctx, "Transmission PSR");
        ctx->setFramePassStage(RtxFramePassStage::TransmissionPSR);
        pushArgs.isTransmissionPSR = 1;
        ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
        ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
      }
      ctx->markGpuStage();
      break;
    }

      case RaytraceMode::RayQueryRayGen:
        noteBranch("RayQueryRayGen(rt-pipeline)");
      {
        ScopedGpuProfileZone(ctx, "Primary Rays");
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(false, true, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
        ctx->traceRays(primaryRayDims.width, primaryRayDims.height, primaryRayDims.depth);
      }
      // The mark count must not depend on RaytraceMode: [Perf.GpuPass] labels
      // stages by position, so a branch that skips marks shifts every name after
      // it. Only the RayQuery path was marked before.
      ctx->markGpuStageIfPending();
      ctx->markGpuStage();

      dispatchDeferredDecals();
      dispatchPSRPrepare();

      {
        // Warning: do not change the order of Reflection and Transmission PSR, that will break
        // PSR data dependencies due to resource aliasing.
        ScopedGpuProfileZone(ctx, "Reflection PSR");
        ctx->setFramePassStage(RtxFramePassStage::ReflectionPSR);
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(pipelineFeatureVariant, true, true, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
      }
      ctx->markGpuStage();

      {
        ScopedGpuProfileZone(ctx, "Transmission PSR");
        ctx->setFramePassStage(RtxFramePassStage::TransmissionPSR);
        pushArgs.isTransmissionPSR = 1;
        ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
      }
      ctx->markGpuStage();
      break;

      case RaytraceMode::TraceRay:
        noteBranch("TraceRay(rt-pipeline)");
      {
        ScopedGpuProfileZone(ctx, "Primary Rays");
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(false, false, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
        ctx->traceRays(primaryRayDims.width, primaryRayDims.height, primaryRayDims.depth);
      }
      ctx->markGpuStageIfPending();
      ctx->markGpuStage();

      dispatchDeferredDecals();
      dispatchPSRPrepare();

      {
        // Warning: do not change the order of Reflection and Transmission PSR, that will break
        // PSR data dependencies due to resource aliasing.
        ScopedGpuProfileZone(ctx, "Reflection PSR");
        ctx->setFramePassStage(RtxFramePassStage::ReflectionPSR);
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(pipelineFeatureVariant, true, false, serEnabled, ommEnabled, includePortals, nrcEnabled, wboitEnabled));
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
      }
      ctx->markGpuStage();

      {
        ScopedGpuProfileZone(ctx, "Transmission PSR");
        ctx->setFramePassStage(RtxFramePassStage::TransmissionPSR);
        pushArgs.isTransmissionPSR = 1;
        ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
      }
      ctx->markGpuStage();
      break;
      case RaytraceMode::Count:
        assert(false && "Invalid RaytraceMode in DxvkPathtracerGbuffer::dispatch");
      break;
    }
  }

  DxvkRaytracingPipelineShaders DxvkPathtracerGbuffer::getPipelineShaders(
    const uint32_t featureVariant,
    const bool isPSRPass,
    const bool useRayQuery,
    const bool serEnabled,
    const bool ommEnabled,
    const bool includePortals,
    const bool nrcEnabled,
    const bool wboitEnabled) {
    ScopedCpuProfileZone();

    const uint32_t rayGenBaseKey = getGbufferVariantKey(featureVariant, isPSRPass, nrcEnabled, wboitEnabled);
    const uint32_t passKey = getGbufferPassVariantKey(isPSRPass, nrcEnabled, wboitEnabled);
    DxvkRaytracingPipelineShaders shaders;

    if (useRayQuery) {
      shaders.addGeneralShader(getGbufferRayQueryRayGenShader(rayGenBaseKey));
      shaders.debugName = isPSRPass ? "GBuffer PSR RayQuery (RGS)" : "GBuffer RayQuery (RGS)";
    } else {
      const uint32_t rayGenKey = rayGenBaseKey | (serEnabled ? kGbufferVariantSER : 0u);
      const uint32_t closestHitKey = passKey | (includePortals ? kGbufferVariantPortals : 0u);

      shaders.addGeneralShader(getGbufferTraceRayGenShader(rayGenKey));
      shaders.addGeneralShader(getGbufferMissShader(passKey));
      shaders.addHitGroup(getGbufferClosestHitShader(closestHitKey), nullptr, nullptr);
      shaders.debugName = isPSRPass ? "GBuffer PSR TraceRay (RGS)" : "GBuffer TraceRay (RGS)";
    }

    if (ommEnabled) {
      shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
    }

    return shaders;
  }

  Rc<DxvkShader> DxvkPathtracerGbuffer::getComputeShader(
    const uint32_t featureVariant,
    const bool isPSRPass,
    const bool nrcEnabled,
    const bool wboitEnabled,
    const bool includePortals) const {
    // NV-DXVK [perf]: !includePortals selects the
    // SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT build of the same source.
    // This is the compute-path equivalent of the opaque_translucent vs rayPortal
    // closest-hit choice getPipelineShaders() makes, and is driven by the same
    // predicate, so the two raytrace modes now compile the same feature set.
    if (includePortals) {
      if (wboitEnabled) {
        if (nrcEnabled) {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_nrc_wboit);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_wboit);
          }
        } else {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_wboit);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_wboit);
          }
        }
      } else {
        if (nrcEnabled) {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_nrc);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc);
          }
        } else {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery);
          }
        }
      }
    } else {
      if (wboitEnabled) {
        if (nrcEnabled) {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_nrc_wboit_opaque_translucent);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_wboit_opaque_translucent);
          }
        } else {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_wboit_opaque_translucent);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_wboit_opaque_translucent);
          }
        }
      } else {
        if (nrcEnabled) {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_nrc_opaque_translucent);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_opaque_translucent);
          }
        } else {
          if (isPSRPass) {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_psr_rayquery_opaque_translucent);
          } else {
            return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_opaque_translucent);
          }
        }
      }
    }
  }

  Rc<DxvkShader> DxvkPathtracerGbuffer::getOccupancyLadderShader(
    const bool nrcEnabled,
    const bool wboitEnabled,
    const bool includePortals,
    VkExtent3D& blockSize) const {
    const uint32_t requested = perfGbufferBlockThreads();

    if (requested == 0) {
      return nullptr;
    }

    // The ladder is built for one permutation only. Rather than silently
    // rendering with the stock shader while the option reads as set — which is
    // exactly how a sweep gets attributed to the wrong rung — say so.
    // Log only on change, matching noteBranch()/[Perf.GbDispatch] in this file, so
    // this is silent at steady state but cannot go stale if the permutation moves.
    static uint32_t s_lastComplaint = UINT32_MAX;
    const uint32_t complaintKey = requested
                                | (nrcEnabled ? (1u << 24) : 0u)
                                | (wboitEnabled ? (1u << 25) : 0u)
                                | (includePortals ? (1u << 26) : 0u);

    if (!nrcEnabled || !wboitEnabled || includePortals) {
      if (s_lastComplaint != complaintKey) {
        s_lastComplaint = complaintKey;
        Logger::warn(str::format(
          "[Perf.Occupancy] rtx.perfGbufferBlockThreads=", requested,
          " IGNORED: the ladder is only built for the NRC + WBOIT + no-portals RayQuery"
          " permutation (have nrc=", (nrcEnabled ? 1 : 0),
          " wboit=", (wboitEnabled ? 1 : 0),
          " portals=", (includePortals ? 1 : 0), ")"));
      }
      return nullptr;
    }

    switch (requested) {
    case 256:
      blockSize = VkExtent3D { 16, 16, 1 };
      return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_wboit_opaque_translucent_occ256);
    case 384:
      blockSize = VkExtent3D { 16, 24, 1 };
      return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_wboit_opaque_translucent_occ384);
    case 512:
      blockSize = VkExtent3D { 32, 16, 1 };
      return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, GbufferRayGenShader, gbuffer_rayquery_nrc_wboit_opaque_translucent_occ512);
    default:
      if (s_lastComplaint != complaintKey) {
        s_lastComplaint = complaintKey;
        Logger::warn(str::format(
          "[Perf.Occupancy] rtx.perfGbufferBlockThreads=", requested,
          " is not a ladder rung (valid: 0, 256, 384, 512) - using the stock 128-thread shader"));
      }
      return nullptr;
    }
  }

  const char* DxvkPathtracerGbuffer::raytraceModeToString(RaytraceMode raytraceMode)
  {
    switch (raytraceMode)
    {
    case RaytraceMode::RayQuery:
      return "Ray Query [CS]";
    case RaytraceMode::RayQueryRayGen:
      return "Ray Query [RGS]";
    case RaytraceMode::TraceRay:
      return "Trace Ray [RGS]";
    default:
      return "Unknown";
    }
  }
}
