/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx/utility/shader_types.h"
#ifdef __cplusplus
#include "rtx/concept/camera/camera.h"
#include "rtx/concept/ray_portal/ray_portal.h"
#else
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/ray_portal/ray_portal.slangh"
#endif

#include "rtx/pass/nrd_args.h"
#include "rtx/pass/nrc_args.h"
#include "rtx/pass/volume_args.h"
#include "rtx/pass/material_args.h"
#include "rtx/pass/view_distance_args.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"
#include "rtx/concept/light/light_types.h"
#include "rtx/concept/surface/surface_shared.h"
#include "rtx/algorithm/nee_cache_data.h"

struct LightRangeInfo {
  uint offset;
  uint count;
  uint16_t rtxdiSampleCount;
  uint16_t volumeRISSampleCount;
  uint16_t risSampleCount;
  uint16_t pad;
};

// Note: ensure 16B alignment
struct TerrainArgs {
  uint2 cascadeMapSize;    // Number of cascade tiles in each dimension
  float2 rcpCascadeMapSize;

  uint maxCascadeLevel;
  float lastCascadeScale;
  float displaceIn;
  uint pad0;
};

struct NeeCacheArgs {
  uint enable;
  uint enableImportanceSampling;
  uint enableMIS;
  uint enableOnFirstBounce;

  uint enableAnalyticalLight;
  float specularFactor;
  float uniformSamplingProbability;
  float cullingThreshold;

  NeeEnableMode enableModeAfterFirstBounce;
  float ageCullingSpeed;
  float emissiveTextureSampleFootprintScale;
  uint approximateParticleLighting;

  float resolution;
  float minRange;
  float learningRate;
  uint clearCache;

  float triangleExplorationRangeRatio;
  uint  triangleExplorationMaxRange;
  float triangleExplorationProbability;
  float triangleExplorationAcceptRangeRatio;

  uint padding;
  uint enableReshuffleResilience;
  uint reshuffleMaxAge;
  uint enableSpatialReuse;
};

struct DomeLightArgs {
  mat4 worldToLightTransform;

  vec3 radiance;
  uint active;

  uint3 pad0;
  uint textureIndex;
};

struct SssArgs {
  uint enableThinOpaque;
  uint enableDiffusionProfile;
  float diffusionProfileScale;
  u16vec2 diffusionProfileDebuggingPixel;
};

struct EyeArgs {
  uint  enableEyes;
  float normalBendingEyeball;
  float normalBendingCornea;
  float whitesAlbedoScale;

  float irisRadius;
  float irisDepth;
  uint  pad0;
  uint  pad1;
};

#define OBJECT_PICKING_INVALID (cb.clearColorPicking)

// Constant buffer
struct RaytraceArgs {
  // NOTE: this class should be kept as all structs, then all non-structs.  This is because the padding rules are different between C++ and shaders.
  Camera camera;

  // Note: Primary combined variant used in place of the primary direct denoiser when seperated direct/indirect
  // lighting is not used.
  NrdArgs primaryDirectNrd;
  NrdArgs primaryIndirectNrd;
  NrdArgs secondaryCombinedNrd;

  // Note: Not tightly packed, meaning these indices will align with the Ray Portal Index in the
  // Surface Material. Do note however due to elements being potentially "empty" each Ray Portal Hit Info
  // must be checked to be empty or not before usage. Additionally both Ray Portals in a pair will match
  // in state, either being present or not.
  // The first `maxRayPortalCount` portals are for this frame, the second `maxRayPortalCount` are for the previous frame.
  RayPortalHitInfo rayPortalHitInfos[maxRayPortalCount * 2];

  VolumeArgs volumeArgs;
  OpaqueMaterialArgs opaqueMaterialArgs;
  TranslucentMaterialArgs translucentMaterialArgs;
  ViewDistanceArgs viewDistanceArgs;

  LightRangeInfo lightRanges[lightTypeCount];

  TerrainArgs terrainArgs;
  NeeCacheArgs neeCacheArgs;
  DomeLightArgs domeLightArgs;
  NrcArgs nrcArgs;
  SssArgs sssArgs;
  EyeArgs eyeArgs;
  AtmosphereArgs atmosphereArgs;

  Camera renderTargetCamera;

  // ------------------------- Structs above this line, non structs below this line -----------------------------------

  uint frameIdx;
  float ambientIntensity;
  uint16_t lightCount;
  uint16_t risTotalSampleCount;
  uint16_t volumeRISTotalSampleCount;
  uint16_t rtxdiTotalSampleCount;

  // The maximum probability of continuing a path when Russian Roulette is being used.
  RussianRouletteMode russianRouletteMode;
  float russianRouletteDistanceFactor;
  float russianRouletteDiffuseContinueProbability;
  float russianRouletteSpecularContinueProbability;

  float russianRouletteMaxContinueProbability;
  float russianRoulette1stBounceMinContinueProbability;
  float russianRoulette1stBounceMaxContinueProbability;
  float fireflyFilteringLuminanceThreshold;

  // The minimum number of indirect bounces the path must complete before Russian Roulette can be used. Must be < 16.
  uint8_t pathMinBounces;
  // The maximum number of indirect bounces the path will be allowed to complete. Must be < 16.
  uint8_t pathMaxBounces;
  // The number of samples to clamp temporal reservoirs to. Note this is not the same as RTXDI's history length as it is not scaled
  // by the number of samples the current reservoir performs (due to variability in how many actual current reservoir samples are done).
  uint16_t volumeTemporalReuseMaxSampleCount;
  // The maximum number of resolve interactions for primary (geometry resolver) rays.
  uint8_t primaryRayMaxInteractions;
  // The maximum number of resolve interactions for PSR (geometry resolver) rays.
  uint8_t psrRayMaxInteractions;
  // The maximum number of resolve interactions for secondary (integrator) rays.
  uint8_t secondaryRayMaxInteractions;
  // The number of active Ray Portals (Used for Ray Portal sampling). Always <= RAY_PORTAL_MAX_COUNT
  uint8_t numActiveRayPortals;
  float secondarySpecularFireflyFilteringThreshold;
  uint  outputParticleLayer;

  // Note: Packed as float16, uses uint16_t due to being shared on C++ side
  uint16_t emissiveBlendOverrideEmissiveIntensity;
  // The maximum number of bounces to evaluate reflection PSR over.
  uint8_t psrrMaxBounces;
  // The maximum number of bounces to evaluate transmission PSR over.
  uint8_t pstrMaxBounces;
  float viewModelRayTMax;
  uint16_t particleSoftnessFactor;
  uint16_t emissiveIntensity;
  uint8_t rtxdiSpatialSamples;
  uint8_t rtxdiDisocclusionSamples;
  uint8_t rtxdiMaxHistoryLength;
  uint8_t virtualInstancePortalIndex; // portal space for which virtual view model or player model instances were generated for

  float indirectRaySpreadAngleFactor;
  // Half the angle of the cone spawned by each pixel to use for ray cone texture filtering.
  float screenSpacePixelSpreadHalfAngle;
  uint debugView;
  float primaryDirectMissLinearViewZ;


  vec4 debugKnob;     // For temporary tuning in shaders, has a dedicated UI widget.

  // NV-DXVK: TF2 3D-skybox cloud fog reconstruction. The cloud-billboard
  // pixel shader synthesizes its colour as
  //   lerp(albedo, fogColor * c_fogColorFactor, fogFactor)
  // with fogColor = k2.xyz * sunAmount^2 + k1.xyz. k0-k3 are c_fogParams
  // from CBufCommonPerCamera; c_fogColorFactor / c_maxLightingValue are
  // captured alongside them. All are camera-global, captured per-frame from
  // the cloud draws in d3d11_rtx.cpp::FillMaterialData and stashed on
  // SceneManager; RtxContext copies them here each frame. The opaque surface
  // material shader applies the blend for surfaces flagged
  // OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG. fogFactor is taken as k0.w
  // (the fog cap) — the 3D skybox is reprojected to effective infinite
  // distance, which saturates the height/distance fog integral. Placed here
  // (vec4 after vec4) so cbuffer 16-byte alignment matches host and shader.
  // See HANDOFF_CLOUDS.md.
  vec4 tf2FogK1_K0W;   // xyz = k1 (base fog colour), w = k0.w (fog cap)
  vec4 tf2FogK2_K2W;   // xyz = k2 (sun-tint fog colour), w = k2.w (sun bias)
  vec4 tf2FogK3;       // xyz = k3 (sun direction), w = k3.w (sun scale)
  vec4 tf2FogMisc;     // x = c_fogColorFactor, y = c_maxLightingValue,
                       // z = valid (1 once captured), w = unused

  // Values to use on a ray miss
  vec3 clearColorNormal;
  float clearColorDepth;

  float2 upscaleFactor;   // Displayed(upscaled) / RT resolution
  uint32_t clearColorPicking;

  uint enableDLSSRR;
  uint setLogValueForDisocclusionMaskForDLSSRR;

  // NOTE: Variables need to be in groups of 4x32 bits above this comment.

  uint uniformRandomNumber;
  uint16_t opaqueDiffuseLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueDiffuseLobeSamplingProbability;
  uint16_t opaqueSpecularLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueSpecularLobeSamplingProbability;
  uint16_t opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueOpacityTransmissionLobeSamplingProbability;
  uint16_t opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueDiffuseTransmissionLobeSamplingProbability;

  uint16_t translucentSpecularLobeSamplingProbabilityZeroThreshold;
  uint16_t minTranslucentSpecularLobeSamplingProbability;
  uint16_t translucentTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minTranslucentTransmissionLobeSamplingProbability;
  float roughnessDemodulationOffset;
  float timeSinceStartSeconds;
  
  uint enableCalculateVirtualShadingNormals;
  uint enableDirectLighting;
  uint enableEmissiveBlendEmissiveOverride;
  uint enablePortalFadeInEffect;
  uint enableRussianRoulette;
  uint enableSecondaryBounces;
  uint enableSeparateUnorderedApproximations;
  uint enableStochasticAlphaBlend;
  uint16_t enableDirectTranslucentShadows;
  uint16_t enableDirectAlphaBlendShadows;
  uint16_t enableIndirectTranslucentShadows;
  uint16_t enableIndirectAlphaBlendShadows;
  uint enableFirstBounceLobeProbabilityDithering;
  uint enableUnorderedResolveInIndirectRays;
  uint enableProbabilisticUnorderedResolveInIndirectRays;
  uint enableUnorderedEmissiveParticlesInIndirectRays;
  uint enableTransmissionApproximationInIndirectRays;
  uint enableDecalMaterialBlending;
  uint enableBillboardOrientationCorrection;
  uint enablePlayerModelInPrimarySpace;
  uint enablePlayerModelPrimaryShadows;
  uint enablePreviousTLAS;
  uint useIntersectionBillboardsOnPrimaryRays;

  // NV-DXVK [Perf.GbStop]: ablation ladder for the primary-ray dispatch.
  //
  // !! THE ms FIGURES IN THIS BLOCK AND THE NEXT ARE HISTORICAL (pre-2026-07-25).
  // !! They were measured with the coverage-atomic block still running
  // !! unconditionally inside opaqueSurfaceMaterialInteractionCreate, which is
  // !! what made the material stage look expensive. That block is now gated on
  // !! cb.perfCoverageWrites and the pass went 131 ms -> 27 ms. Do not plan
  // !! against these numbers and do not "confirm" them - the whole point of the
  // !! retraction is that the old split was an artefact of the instrumentation.
  // !! Re-derive with rtx.perfAutoSweep, whose gb1..gb4 rungs walk this ladder
  // !! at runtime and print a fresh split every run.
  //
  // CURRENT SPLIT (2026-07-25, post-fix, reproduced across two full sweeps,
  // baseline ~24-27 ms depending on thermal state):
  //
  //   launch + G-buffer store   0.4 ms    ( 2%)
  //   traversal                 1.8-2.0   ( 8%)
  //   UNORDERED RESOLVE        13.1-13.4  (~56%)   <- dominant
  //   material                  7.4-9.3   (~32%)
  //   tail                      0.6-1.6   (under the noise floor)
  //
  // Supporting census: 1.206 unordered candidates/pixel, acceptRate 1.0,
  // against a primaryRayMaxInteractions cap of 32 - so the unordered stage is
  // NOT volume-bound, and lowering that cap buys nothing.
  //
  // Every probe aimed inside the unordered stage has measured flat (see the
  // RESULT notes on perfCheapTextureGradients and perfCoherentUnorderedFetch
  // below, and [Perf.DecalBins] in resolve.slangh). The pass is register-capped
  // at 255 (8 warps/SM) and latency-bound; no localised cut moves it.
  //
  // The pass sits inside ONE dispatch with no internal timers, and this build
  // has no VK_KHR_shader_clock support, so the only way to attribute it is to
  // cut the shader short at known points and diff the pass timer.
  //   0 = full shader (default, no behavioural change)
  //   1 = ray generation only, primary trace skipped
  //   2 = + primary TLAS traversal, hit function returns immediately
  //   3 = + unordered resolve, resolveVertex skipped
  //   4 = + resolveVertex (surface + material evaluation)
  //  >4 = full
  // Differences between consecutive steps are the per-stage costs. Every step
  // still writes the G-buffer, so step 1 is the launch+store floor rather than
  // zero, and the deltas are not contaminated by store cost appearing only once.
  uint perfGbStopAfter;

  // NV-DXVK [Perf.SubLadder]: the stage ladder above once resolved gb_primaryRays
  // to 3.0 ms traversal + 42.5 ms unordered resolve + 74.5 ms material eval +
  // ~0 ms tail/loop. HISTORICAL - see the warning above; that 74.5 ms was mostly
  // the coverage atomics, not material work. These cuts go finer than the stage
  // ladder can, which is still why they exist; only the attribution is void.
  //
  // Runtime rather than compile-time on purpose. The compile-time form existed to
  // read per-stage register counts; occupancy is now refuted (rung 4 runs at 168
  // registers vs 255 full, 12 warps vs 8, for identical time), so nothing here
  // needs dead-code elimination - only a pass timer diff. Runtime means one build
  // answers every rung.
  //
  // perfUnorderedStopAfter - cuts inside resolveVertexUnordered's candidate loop.
  // Every rung still runs the rayQuery.Proceed() loop to completion, so traversal
  // cost stays in all of them and the deltas isolate per-candidate body work.
  //   0 = full
  //   1 = traversal only (count candidates, process none)
  //   2 = + hit/ray interaction + surfaceInteractionCreate
  //   3 = + view distance and surface clip test
  //   4 = + material interaction create (this is where the texture reads are)
  //   5 = + opaque/translucent approximations and decal binning
  //   6 = + WBOIT / bin accumulation (everything but the final flush + composite)
  uint perfUnorderedStopAfter;

  // NV-DXVK [Perf.SubLadder]: independent feature skips inside
  // opaqueSurfaceMaterialInteractionCreate, which was believed to be the bulk of
  // the 74.5 ms - see the HISTORICAL warning on perfGbStopAfter above.
  // Deliberately NOT a ladder - these are not nested, and each one needs to be
  // isolatable on its own and in combination.
  //   perfSkipPom              - skip the POM raymarch, keep interpolated texcoords
  //   perfSkipMaterialTextures - skip every material texture read, use constants
  //   perfSkipThinFilm         - force the thin film layer flag off
  // All produce a wrong image by design; they are timing probes only.
  uint perfSkipPom;
  uint perfSkipMaterialTextures;
  uint perfSkipThinFilm;

  // NV-DXVK [Perf.UnorderedSteps]: raw per-pixel counters for the unordered
  // stage, so the 42.5 ms can be attributed to loop COUNT vs per-candidate COST
  // before any cut is interpreted. Accumulates into SurfaceCoverageBuffer regions
  // 54-56 (steps, interactions, pixels) and is read back on the existing coverage
  // throttle. Off by default - it adds three InterlockedAdds per pixel.
  uint perfUnorderedStepCensus;

  // NV-DXVK [Perf.MatLadder]: cuts INSIDE opaqueSurfaceMaterialInteractionCreate,
  // which the sub-stage sweep localised as ~98 ms of the ~126 ms pass (23.3 ms
  // inside the unordered stage + the ordered path's 74.5 ms), called ~2.2x per
  // pixel. rtx.perfSkipMaterialTextures removed 6 of its 7 texture reads for only
  // -11.4 ms, so the bulk is construction work and the function is 1530 lines
  // long - too large to attribute without bisecting it.
  //
  // The function returns a partially-built interaction at any nonzero rung, so
  // the image is garbage and downstream values are whatever the zero-init left.
  //   0 = full
  //   1 = texture reads only, return immediately after them
  //   2 = + emissive/gamma/material modifiers    (through ~line 1608)
  //   3 = + normal detail and modifiers          (through ~line 1709)
  //   4 = + opacity/albedo/roughness composition (through the VGUI early-out)
  //  >4 = full
  // Consecutive differences attribute the 1530 lines into four buckets.
  uint perfMaterialStopAfter;

  // NV-DXVK [Perf.CoverageGate]: enables the 52-atomic-per-primary-hit coverage
  // instrumentation block at the end of opaqueSurfaceMaterialInteractionCreate.
  // Driven by rtx.logSurfaceCoverage, which previously gated only the CPU-side
  // readback while the GPU writes ran unconditionally on every primary hit.
  //
  // COST: turning this on costs ~104 ms/frame at 1080p (gb_primaryRays 27 -> 131).
  // It is not a logging toggle. Several of the atomics target a single shared
  // address, so every thread in a 2.07 Mpix dispatch serialises on one word.
  // Consequence for measurement: rtx.logSurfaceCoverage and ANY perf run are
  // mutually exclusive. In particular rtx.perfAutoSweep's census step wants
  // coverage on, and enabling it re-arms these atomics for EVERY step of the
  // sweep, not just that one - which silently invalidates the whole table.
  // Run the census as its own pass.
  uint perfCoverageWrites;

  // NV-DXVK [perf]: bypass computeAnisotropicEllipseAxes in favour of the
  // existing kFootprintFromTextureCoordDiff path (two vector subtractions).
  //
  // The ellipse-axes function is fork-local and runs per pixel per hit: four
  // clip-space projections of the triangle plus the hit point, a screen-space
  // determinant, rank-1 UV detection, and UV/world area ratios. Upstream Remix
  // derives texture LOD from the ray cone with a few scalar ops. Since the
  // ladder put 47 ms in surface + material evaluation, and forcing a +6 mip bias
  // (a ~4096x cut in texel footprint) moved only 10%, the cost is ALU/register
  // pressure rather than texture traffic - and this is the largest fork-local
  // block of ALU in that stage.
  //
  // Substituting an already-present cheap path rather than deleting work keeps
  // the shader structurally intact, so the delta attributes cleanly. Mip
  // selection changes, so expect aliasing; this is a probe, not a setting.
  //
  // !! RESULT 2026-07-25: REFUTED. delta -0.19 ms against a 1.50 ms resolution
  // !! floor, i.e. zero. computeAnisotropicEllipseAxes is NOT the cost, despite
  // !! being the largest fork-local ALU block in the stage that owns the time.
  // !!
  // !! This is the load-bearing negative of the whole investigation: removing
  // !! ALU from the hot rung changed nothing, which is what redirected the
  // !! search from compute to memory, and eventually to register pressure.
  // !! Do not re-run it expecting a different answer.
  uint perfCheapTextureGradients;

  // NV-DXVK [Perf.CoherentFetch]: make the unordered candidate body's memory
  // accesses COHERENT instead of scattered, without changing how many there are.
  //
  // Why this shape. The unordered resolve is 13.1 ms of a ~24 ms pass and the
  // ladder puts ~10.3 ms of that in one rung (hit info, ray interaction, Surface
  // load, surfaceInteractionCreate). Removing ALU from that rung
  // (perfCheapTextureGradients) measured -0.19 ms against a 1.5 ms floor, so it
  // is not compute. That leaves memory - but "how much" and "how scattered" are
  // different problems with different fixes, and deleting loads cannot tell them
  // apart: a cut there is dead-code eliminated, and cutting earlier collapses
  // the resolve loop's trip count (see the uno1 caveat).
  //
  // So this keeps every load and every instruction and only makes the addresses
  // uniform across the wave. If the stage is bound on cache misses and
  // divergent access, the time collapses; if it is bound on the volume of data
  // moved, it does not. That is the actual open question.
  //
  //   0 = off (ship default, bit-identical)
  //   1 = Surface load coherent: surfaces[0] for every candidate
  //   2 = + vertex fetch coherent: primitiveIndex/barycentrics forced to 0
  //
  // Scoped to the UNORDERED loop only. The ordered path's opacity feeds
  // continueResolving, and perturbing it there would change trip counts and
  // reproduce exactly the confound that makes uno1 unreadable.
  //
  // Produces a garbage image by construction - it is a probe, not a setting.
  //
  // !! RESULT 2026-07-25: BOTH MODES REFUTED. Locality is not the mechanism.
  // !!
  // !!   coh=1 (Surface load coherent): -0.216 ms median, +0.565 ms min.
  // !!         Opposite signs => a hitch, not a cost.
  // !!   coh=2 (+ vertex fetch coherent): +0.195 median, +0.454 min,
  // !!         floor 0.622 => flat, and marginally slower if anything.
  // !!
  // !! Making every memory access in the hot rung uniform across the wave, at
  // !! identical instruction count, changed nothing. So the unordered stage is
  // !! not bound on cache misses or divergent addressing. Combined with the
  // !! cheapTextureGradients null above (not ALU either), this is what pointed
  // !! at the real constraint: the shader sits at Register Count=255, the
  // !! hardware cap, i.e. 8 warps/SM, so there is nothing to hide latency with
  // !! and no individual block owns the time. See the [Perf.DecalBins] note in
  // !! resolve.slangh for the follow-on experiment that also failed.
  uint perfCoherentUnorderedFetch;

  uint enableRtxdi;
  uint enableRtxdiPermutationSampling;
  uint enableRtxdiRayTracedBiasCorrection;
  uint enableRtxdiSampleStealing;
  uint enableRtxdiStealBoundaryPixelSamplesWhenOutsideOfScreen;
  uint enableRtxdiCrossPortalLight;
  uint enableRtxdiTemporalBiasCorrection;
  uint enableRtxdiInitialVisibility;
  uint enableRtxdiTemporalReuse;
  uint enableRtxdiSpatialReuse;
  uint enableRtxdiDiscardInvisibleSamples;
  uint enableRtxdiDiscardEnlargedPixels;
  uint enableDirectLightBoilingFilter;
  uint enableRtxdiBestLightSampling;
  float directLightBoilingThreshold;
  float rtxdiDisocclusionFrames;

  uint enableDemodulateRoughness;
  uint enableHitTFiltering;
  uint enableReplaceDirectSpecularHitTWithIndirectSpecularHitT;
  uint enableSeparatedDenoisers;

  uint enableViewModelVirtualInstances;

  uint enablePSRR;
  uint enablePSTR;
  uint enablePSTROutgoingSplitApproximation;
  uint enablePSTRSecondaryIncidentSplitApproximation;
  float psrrNormalDetailThreshold;
  float pstrNormalDetailThreshold;

  uint enableEnhanceBSDFDetail;
  uint enhanceBSDFIndirectMode;
  float enhanceBSDFDirectLightPower;
  float enhanceBSDFIndirectLightPower;
  float enhanceBSDFDirectLightMaxValue;
  float enhanceBSDFIndirectLightMaxValue;
  float enhanceBSDFIndirectLightMinRoughness;

  uint startInMediumMaterialIndex;
  uint enableReSTIRGI;
  uint enableReSTIRGIFinalVisibility;
  uint enableReSTIRGIReflectionReprojection;
  float restirGIReflectionMinParallax;
  uint enableReSTIRGIVirtualSample;
  float reSTIRGIVirtualSampleLuminanceThreshold;
  float reSTIRGIVirtualSampleRoughnessThreshold;
  float reSTIRGIVirtualSampleSpecularThreshold;
  float reSTIRGIVirtualSampleMaxDistanceRatio;
  uint reSTIRGIMISMode;
  float reSTIRGIMISModePairwiseMISCentralWeight;
  uint enableReSTIRGIPermutationSampling;
  uint enableReSTIRGIDLSSRRCompatibilityMode;
  float reSTIRGIDLSSRRTemporalRandomizationRadius;
  uint enableReSTIRGISampleStealing;
  float reSTIRGISampleStealingJitter;
  uint enableReSTIRGIStealBoundaryPixelSamplesWhenOutsideOfScreen;
  uint enableReSTIRGISpatialReuse;
  uint enableReSTIRGITemporalReuse;
  uint reSTIRGIBiasCorrectionMode;
  uint enableReSTIRGIBoilingFilter;
  float boilingFilterLowerThreshold;
  float boilingFilterHigherThreshold;
  float boilingFilterRemoveReservoirThreshold;
  uint temporalHistoryLength;
  uint permutationSamplingSize;
  uint enableReSTIRGITemporalBiasCorrection;
  uint enableReSTIRGIDiscardEnlargedPixels;
  float reSTIRGIHistoryDiscardStrength;
  uint enableReSTIRGITemporalJacobian;
  float reSTIRGIFireflyThreshold;
  float reSTIRGIRoughnessClamp;
  float reSTIRGIMISRoughness;
  float reSTIRGIMISParallaxAmount;
  uint enableReSTIRGIDemodulatedTargetFunction;
  uint enableReSTIRGILightingValidation;
  uint enableReSTIRGIVisibilityValidation;
  float reSTIRGISampleValidationThreshold;
  float reSTIRGIVisibilityValidationRange;

  uint surfaceCount;
  uint teleportationPortalIndex; // 0 means no teleportation, 1+ means portal 0+

  float resolveTransparencyThreshold;
  float resolveOpaquenessThreshold;
  float resolveStochasticAlphaBlendThreshold;
  float translucentDecalAlbedoFactor;

  uint enableHeuristicSingleScatteringTransmission;

  float skyBrightness;
  uint skyMode;  // 0 = skybox rasterization, 1 = physical atmosphere, 2 = hybrid
  // NV-DXVK [EngineSun]: when 1, the atmosphere sun is provided as an RTXDI Distant light,
  // so the bespoke NEE sun (evalAtmosphereSunNEE) is skipped to avoid double lighting.
  uint sunAsRtxdiLight;
  // NV-DXVK [SkyProbe.cubeRender] true once rasterizeToSkyProbe has run
  // at least once and populated the m_skyProbe cubemap with TF2's
  // authored sky from 6 cube faces. Path tracer IBL sample sites switch
  // from Hillaire-only fallback to SkyProbe sampling when this is true,
  // which surfaces TF2's painted clouds / 3D-skybox detail in indirect
  // bounces and reflections.
  uint skyProbePopulated;

  uint isLastCompositeOutputValid;
  uint isZUp; // Note: Indicates if the Z axis is the "up" axis in world space if true, otherwise the Y axis if false.
  uint enableCullingSecondaryRays;

  u16vec2 gpuPrintThreadIndex;
  uint gpuPrintElementIndex;
  uint enableObjectPicking;

  // NV-DXVK: scene dump (one-shot per-pixel capture). When sceneDumpEnabled
  // is non-zero, the opaque material shader writes one SceneDumpElement per
  // primary-ray pixel into BINDING_SCENE_DUMP_BUFFER at index
  // pixel.y * sceneDumpStride + pixel.x. Stride is the buffer's allocated
  // pitch in elements (= downscaledExtent.width).
  uint sceneDumpEnabled;
  uint sceneDumpStride;

  DisplacementMode pomMode;
  uint pomEnableDirectLighting;
  uint pomEnableIndirectLighting;
  uint pomEnableNEECache;
  uint pomEnableReSTIRGI;
  uint pomEnablePSR;
  uint pomMaxIterations;
  uint enableSssTransmission;
  uint enableSssTransmissionSingleScattering;
  uint sssTransmissionBsdfSampleCount;
  uint sssTransmissionSingleScatteringSampleCount;
  uint enableTransmissionDiffusionProfileCorrection;
  float totalMipBias;

  uint forceFirstHitInGBufferPass;

  uint enableRaytracedRenderTarget;
  // NRC enablement is controlled by global macros being defined.
  // When macros are not used (i.e. in some passes) this variable controls the NRC enablement.
  uint enableNrc;

  // Debug override to disallow NRC training when it is enabled in the first place,
  // hence why it is not named enableNrcTraining here
  uint allowNrcTraining;

  float vertexColorStrength;
  float alphaBlendSurfacePackMult; // for packing/unpacking hitT into Float16 in AlphaBlendSurface

  float wboitEnergyLossCompensation;
  float wboitDepthWeightTuning;
  uint wboitEnabled;
  // NV-DXVK: TF2 holo-character / viewmodel screen-space emissive — per-
  // frame engine `c_gameTime` (CBufCommonPerCamera offset 300), captured
  // from any draw that hits the OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE
  // pattern (see d3d11_rtx.cpp::FillMaterialData). The opaque surface
  // material slang uses this as the multiplier on c_uv1Translate so the
  // pattern scrolls each frame to match native, instead of being frozen
  // at translate × 1.0. Distinct from `timeSinceStartSeconds` because that
  // one is wall-clock from app start (keeps ticking during pause) and the
  // engine value freezes during pause — which is the visually-correct
  // behaviour for the holographic scan-line pattern.
  float screenSpaceEmissiveTime;

  // NV-DXVK [debug.disableDetailOverlay]: when non-zero, the opaque material
  // shader skips the TF2 MOD2X detail-texture albedo overlay. Diagnostic for
  // the Ark "red blot" (brown bare sample turned red by detail). uint not bool.
  uint disableDetailOverlay;

  // NOTE: Add structs to the top section of RaytraceArgs, not the bottom.
  // NOTE: bool does not work in debug builds, use uint instead.
};
