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

#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <limits>
#include <sstream>
#include <iomanip>

#include "../util/util_keybind.h"
#include "../util/config/config.h"
#include "../util/xxHash/xxhash.h"
#include "../util/util_math.h"
#include "../util/util_env.h"
#include "rtx/algorithm/accumulate.h"
#include "rtx_utils.h"
#include "rtx/concept/ray_portal/ray_portal.h"
#include "rtx_global_volumetrics.h"
#include "rtx_pathtracer_gbuffer.h"
#include "rtx_pathtracer_integrate_direct.h"
#include "rtx_pathtracer_integrate_indirect.h"
#include "rtx_dlss.h"
#include "rtx_materials.h"
#include "rtx/pass/material_args.h"
#include "rtx_option.h"
#include "rtx_option_manager.h"
#include "rtx_hashing.h"
#include "rtx_mod_manager.h"

enum _NV_GPU_ARCHITECTURE_ID;
typedef enum _NV_GPU_ARCHITECTURE_ID NV_GPU_ARCHITECTURE_ID;
enum _NV_GPU_ARCH_IMPLEMENTATION_ID;
typedef enum _NV_GPU_ARCH_IMPLEMENTATION_ID NV_GPU_ARCH_IMPLEMENTATION_ID;

// RTX specific options

namespace dxvk {
  class DxvkDevice;

  using RenderPassVolumeIntegrateRaytraceMode = RtxGlobalVolumetrics::RaytraceMode;
  using RenderPassGBufferRaytraceMode = DxvkPathtracerGbuffer::RaytraceMode;
  using RenderPassIntegrateDirectRaytraceMode = DxvkPathtracerIntegrateDirect::RaytraceMode;
  using RenderPassIntegrateIndirectRaytraceMode = DxvkPathtracerIntegrateIndirect::RaytraceMode;

  // DLSS-RR is not listed here, because it's considered as a special mode of DLSS
  enum class UpscalerType : int {
    None = 0,
    DLSS,
    NIS,
    TAAU,
    XeSS
  };

  enum class GraphicsPreset : int {
    Ultra = 0,
    High,
    Medium,
    Low,
    Custom,
    // Note: Used to automatically have the graphics preset set on initialization, not used beyond this case
    // as it should be overridden by one of the other values by the time any other code uses it.
    Auto
  };

  enum class RaytraceModePreset {
    Custom = 0,
    Auto = 1
  };

  enum class DlssPreset : int {
    Off = 0,
    On,
    Custom
  };

  enum class XeSSPreset : int {
    UltraPerf = 0,
    Performance,
    Balanced,
    Quality,
    UltraQuality,
    UltraQualityPlus,
    NativeAA,
    Custom,
    Invalid
  };

  enum class NisPreset : int {
    Performance = 0,
    Balanced,
    Quality,
    Fullscreen
  };

  enum class TaauPreset : int {
    UltraPerformance = 0,
    Performance,
    Balanced,
    Quality,
    Fullscreen
  };

  enum class CameraAnimationMode : int {
    CameraShake_LeftRight = 0,
    CameraShake_FrontBack,
    CameraShake_Yaw,
    CameraShake_Pitch,
    YawRotation
  };

  enum class TonemappingMode : int {
    Global = 0,
    Local
  };

  enum class UIType : int {
    None = 0,
    Basic,
    Advanced,
    Count
  };

  enum class ReflexMode : int {
    None = 0,
    LowLatency,
    LowLatencyBoost
  };

  enum class FusedWorldViewMode : int {
    None = 0,
    View,
    World
  };
  
  enum class SkyAutoDetectMode : int {
    None = 0,
    CameraPosition,
    CameraPositionAndDepthFlags
  };

  enum class SkyMode : int {
    SkyboxRasterization = 0,
    PhysicalAtmosphere  = 1,
    // NV-DXVK [HybridSky]: best-of-both. Visible sky + bounce light come
    // from TF2's actual rasterized skybox (artist intent preserved -
    // purple skies cast purple bounce, painted dropships visible
    // behind playable space). Direct sun NEE still runs through the
    // Hillaire LUT, so the sun's apparent colour gets atmospheric
    // transmittance (warm yellow at noon, red at horizon). Sun
    // direction comes from Tier 1 c_sunDir capture.
    Hybrid              = 2
  };

  enum class EnableVsync : int {
    Off = 0,
    On = 1,
    WaitingForImplicitSwapchain = 2   // waiting for the app to create the device + implicit swapchain, we latch the vsync setting from there
  };

  enum class IntegrateIndirectMode : int {
    ImportanceSampled = 0,   // Importance sampled integration - provides the noisiest output and used primarily for reference comparisons
    ReSTIRGI = 1,            // Importance Sampled + ReSTIR GI integrations
    NeuralRadianceCache = 2, // Implements a live trained neural network to provide a world space radiance cache and allow the pathtracer to terminate paths earlier into the cache.
  
    Count
  };

  class RtxOptions {
    friend class ImGUI;
    friend class ImGuiSplash;
    friend class ImGuiCapture;
    friend class NeuralRadianceCache;
    friend class RtxContext;
    friend class RtxInitializer;
    friend class RtxComposite;

    RTX_OPTION("rtx", fast_unordered_set, lightmapTextures, {},
                  "Textures used for lightmapping (baked static lighting on surfaces) in older games.\n"
                  "These textures will be ignored when attempting to determine the desired textures from a draw to use for ray tracing.");
    RTX_OPTION("rtx", fast_unordered_set, skyBoxTextures, {},
                  "Textures on draw calls used for the sky or are otherwise intended to be very far away from the camera at all times (no parallax).\n"
                  "Any draw calls using a texture in this list will be treated as sky and rendered as such in a manner different from typical geometry.");    
    RTX_OPTION("rtx", fast_unordered_set, skyBoxGeometries, {},
                  "Geometries from draw calls used for the sky or are otherwise intended to be very far away from the camera at all times (no parallax).\n"
                  "Any draw calls using a geometry hash in this list will be treated as sky and rendered as such in a manner different from typical geometry.\n"
                  "The geometry hash being used for sky detection is based off of the asset hash rule, see: \"rtx.geometryAssetHashRuleString\".");
    RTX_OPTION("rtx", fast_unordered_set, ignoreTextures, {},
                  "Textures on draw calls that should be ignored.\n"
                  "Any draw call using an ignore texture will be skipped and not ray traced, useful for removing undesirable rasterized effects or geometry not suitable for ray tracing.");
    RTX_OPTION("rtx", fast_unordered_set, ignoreLights, {},
                  "Lights that should be ignored.\nAny matching light will be skipped and not added to be ray traced.");
    RTX_OPTION("rtx", fast_unordered_set, uiTextures, {},
                  "Textures on draw calls that should be treated as screenspace UI elements.\n"
                  "All exclusively UI-related textures should be classified this way and doing so allows the UI to be rasterized on top of the ray traced scene like usual.\n"
                  "Note that currently the first UI texture encountered triggers RTX injection (though this may change in the future as this does cause issues with games that draw UI mid-frame).");
    // NV-DXVK: Shader-hash variants of uiTextures, for games whose HUD
    // draws only bind dynamic / zero-hash textures (e.g. Titanfall 2's
    // VGUI, which hits MaybeEarlyInjectForUITexture with hashes=[] on
    // every HUD draw). Entries are the first 16 hex chars of the
    // bound shader's SHA1 (what Remix logs as "VS_xxxxxxxxxxxxxxxx" /
    // "FS_xxxxxxxxxxxxxxxx"), written as 64-bit hex values —
    // e.g. 0xd69c3951f050e757. If either the VS or PS hash of a draw
    // matches, MaybeEarlyInjectForUITexture treats the draw like a
    // uiTextures hit and schedules injectRTX ahead of the HUD raster.
    RTX_OPTION("rtx", fast_unordered_set, uiVertexShaderHashes, {},
                  "Vertex-shader hashes (first 16 hex chars of the SHA1, written as 0x...) whose draws should trigger the screenspace-UI injection path.\n"
                  "Use when the game's HUD binds only dynamic / zero-hash textures so rtx.uiTextures can't key on them. The D3D11Rtx.UITex HUD-filter log lines print the VS hash in a copy-pasteable form.");
    RTX_OPTION("rtx", fast_unordered_set, uiPixelShaderHashes, {},
                  "Pixel-shader hashes (first 16 hex chars of the SHA1, written as 0x...) whose draws should trigger the screenspace-UI injection path.\n"
                  "Same role as rtx.uiVertexShaderHashes but keyed on the PS. Either a VS or PS match is sufficient to fire the inject.");
    RTX_OPTION("rtx", fast_unordered_set, worldSpaceUiTextures, {},
                  "Textures on draw calls that should be treated as worldspace UI elements.\n"
                  "Unlike typical UI textures this option is useful for improved rendering of UI elements which appear as part of the scene (moving around in 3D space rather than as a screenspace element).");
    RTX_OPTION("rtx", fast_unordered_set, worldSpaceUiBackgroundTextures, {}, 
                  "Hack/workaround option for dynamic world space UI textures with a coplanar background.\n"
                  "Apply to backgrounds if the foreground material is a dynamic world texture rendered in UI that is unpredictable and rapidly changing.\n"
                  "This offsets the background texture backwards.");
    RTX_OPTION("rtx", fast_unordered_set, hideInstanceTextures, {},
                  "Textures on draw calls that should be hidden from rendering, but not totally ignored.\n"
                  "This is similar to rtx.ignoreTextures but instead of completely ignoring such draw calls they are only hidden from rendering, allowing for the hidden objects to still appear in captures.\n"
                  "As such, this is mostly only a development tool to hide objects during development until they are properly replaced, otherwise the objects should be ignored with rtx.ignoreTextures instead for better performance.");
    RTX_OPTION("rtx", fast_unordered_set, playerModelTextures, {}, "");
    RTX_OPTION("rtx", fast_unordered_set, playerModelBodyTextures, {}, "");
    RTX_OPTION("rtx", fast_unordered_set, lightConverter, {}, "");
    RTX_OPTION("rtx", fast_unordered_set, particleTextures, {},
                  "Textures on draw calls that should be treated as particles.\n"
                  "When objects are marked as particles more approximate rendering methods are leveraged allowing for more effecient and typically better looking particle rendering.\n"
                  "Generally any billboard-like blended particle objects in the original application should be classified this way.");
    RTX_OPTION("rtx", fast_unordered_set, beamTextures, {},
                  "Textures on draw calls that are already particles or emissively blended and have beam-like geometry.\n"
                  "Typically objects marked as particles or objects using emissive blending will be rendered with a special method which allows re-orientation of the billboard geometry assumed to make up the draw call in indirect rays (reflections for example).\n"
                  "This method works fine for typical particles, but some (e.g. a laser beam) may not be well-represented with the typical billboard assumption of simply needing to rotate around its centroid to face the view direction.\n"
                  "To handle such cases a different beam mode is used to treat objects as more of a cylindrical beam and re-orient around its main spanning axis, allowing for better rendering of these beam-like effect objects.");
    RTX_OPTION("rtx", fast_unordered_set, ignoreTransparencyLayerTextures, {},
                  "Textures on draw calls that should not be stored in the transparency layer, when DLSS-RR is on.\n"
                  "The transparency layer stores noise-free transparent objects which bypasses DLSS-RR denoising, but it has lower anti-aliasing quality.\n"
                  "Transparent objects that have aliasing/flickering issues, like laser beams, can be added to this list to achieve better anti-aliasing quality.");
    RTX_OPTION("rtx", fast_unordered_set, decalTextures, {},
                  "Textures on draw calls used for static geometric decals or decals with complex topology.\n"
                  "These materials will be blended over the materials underneath them when decal material blending is enabled.\n"
                  "A small configurable offset is applied to each flat/co-planar part of these decals to prevent coplanar geometric cases (which poses problems for ray tracing).");
    // Deprecated decal texture options - these are migrated to decalTextures via onChange callbacks
    public: static void dynamicDecalTexturesOnChange(DxvkDevice* device);
    public: static void singleOffsetDecalTexturesOnChange(DxvkDevice* device);
    public: static void nonOffsetDecalTexturesOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", fast_unordered_set, dynamicDecalTextures, {},
                  "Warning: This option is deprecated, please use rtx.decalTextures instead.\n"
                  "Textures on draw calls used for dynamically spawned geometric decals, such as bullet holes.\n"
                  "These materials will be blended over the materials underneath them when decal material blending is enabled.\n"
                  "A small configurable offset is applied to each quad part of these decals to prevent coplanar geometric cases (which poses problems for ray tracing).",
                  args.onChangeCallback = &dynamicDecalTexturesOnChange);
    RTX_OPTION_ARGS("rtx", fast_unordered_set, singleOffsetDecalTextures, {},
                  "Warning: This option is deprecated, please use rtx.decalTextures instead.\n"
                  "Textures on draw calls used for geometric decals that don't inter-overlap for a given texture hash. Textures must be tagged as \"Decal Texture\" or \"Dynamic Decal Texture\" to apply.\n"
                  "Applies a single shared offset to all the batched decal geometry rendered in a given draw call, rather than increasing offset per decal within the batch (i.e. a quad in case of \"Dynamic Decal Texture\").\n"
                  "Note, the offset adds to the global offset among all decals drawn with different draw calls.\n"
                  "The decal textures tagged this way must not inter-overlap within a batch / single draw call since the same offset is applied to all of them.\n"
                  "Applying a single offset is useful for stabilizing decal offsets when a game dynamically batches decals together.\n"
                  "In addition, it makes the global decal offset index grow slower and thus it minimizes a chance of hitting the \"rtx.decals.maxOffsetIndex limit\".",
                  args.onChangeCallback = &singleOffsetDecalTexturesOnChange);
    RTX_OPTION_ARGS("rtx", fast_unordered_set, nonOffsetDecalTextures, {},
                  "Warning: This option is deprecated, please use rtx.decalTextures instead.\n"
                  "Textures on draw calls used for geometric decals with arbitrary topology that are already offset from the base geometry.\n"
                  "These materials will be blended over the materials underneath them when decal material blending is enabled.\n"
                  "Unlike typical decals however these decals have no offset applied to them due assuming the offset is already being done by whatever is passing data to Remix.",
                  args.onChangeCallback = &nonOffsetDecalTexturesOnChange);
    RTX_OPTION("rtx", fast_unordered_set, terrainTextures, {}, "Albedo textures that are baked blended together to form a unified terrain texture used during ray tracing.\n"
                                                                  "Put albedo textures into this category if the game renders terrain as a blend of multiple textures.");
    RTX_OPTION("rtx", fast_unordered_set, opacityMicromapIgnoreTextures, {}, "Textures to ignore when generating Opacity Micromaps. This generally does not have to be set and is only useful for black listing problematic cases for Opacity Micromap usage.");
    RTX_OPTION("rtx", fast_unordered_set, animatedWaterTextures, {},
                  "Textures on draw calls to be treated as \"animated water\".\n"
                  "Objects with this flag applied will animate their normals to fake a basic water effect based on the layered water material parameters, and only when rtx.opaqueMaterial.layeredWaterNormalEnable is set to true.\n"
                  "Should typically be used on static water planes that the original application may have relied on shaders to animate water on.");
    RTX_OPTION("rtx", fast_unordered_set, ignoreBakedLightingTextures, {},
                  "Textures for which to ignore two types of baked lighting, Texture Factors and Vertex Color.\n\n"
                  "Texture Factor disablement:\n"
                  "Using this feature on selected textures will eliminate the texture factors.\n"
                  "For instance, if a game bakes lighting information into the Texture Factor for particular textures, applying this option will remove them.\n"
                  "This becomes useful when unexpected results occur due to the Texture Factor.\n"
                  "Consider an example where the original texture contains red tints baked into the Texture Factor. If a user replaces the texture, it will blend with the red tints, resulting in an undesirable reddish outcome.\n"
                  "In such cases, users can employ this option to eliminate the unwanted tints from their replacement textures.\n"
                  "Similarly, users can tag textures if shadows are baked into the Texture Factor, causing the replacing texture to appear darker than anticipated.\n\n"
                  "Vertex Color disablement:\n"
                  "Using this feature on selected textures will eliminate the vertex colors.\n\n"
                  "Note, enabling this setting will automatically disable multiple-stage texture factor blendings for the selected textures.\n"
                  "Only use this option when necessary, as the Texture Factor and Vertex Color can be used for simulating various texture effects, tagging a texture with this option will unexpectedly eliminate these effects.");
    RTX_OPTION("rtx", fast_unordered_set, ignoreAlphaOnTextures, {}, 
                  "Textures for which to ignore the alpha channel of the legacy colormap. Textures will be rendered fully opaque as a result.");
    RTX_OPTION("rtx.antiCulling", fast_unordered_set, antiCullingTextures, {},
                  "Textures that are forced to extend life length when anti-culling is enabled.\n"
                  "Some games use different culling methods we can't fully match, use this option to manually add textures to force extend their life when anti-culling fails.");
    RTX_OPTION("rtx.postfx", fast_unordered_set, motionBlurMaskOutTextures, {}, "Disable motion blur for meshes with specific texture.");

    public: static void geometryGenerationHashRuleStringOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", std::string, geometryGenerationHashRuleString, "positions,indices,texcoords,geometrydescriptor,vertexlayout,vertexshader",
                  "Defines which asset hashes we need to generate via the geometry processing engine.",
                  args.onChangeCallback = &geometryGenerationHashRuleStringOnChange);
    public: static void geometryAssetHashRuleStringOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", std::string, geometryAssetHashRuleString, "positions,indices,geometrydescriptor",
                  "Defines which hashes we need to include when sampling from replacements and doing USD capture.",
                  args.onChangeCallback = &geometryAssetHashRuleStringOnChange);
    RTX_OPTION("rtx", fast_unordered_set, raytracedRenderTargetTextures, {}, "DescriptorHashes for Render Targets. (Screens that should display the output of another camera).");
    RTX_OPTION("rtx", fast_unordered_set, particleEmitterTextures, {}, "Objects rendered with these textures will emit particles that inherit the material of the object itself.");
    RTX_OPTION("rtx", fast_unordered_set, smoothNormalsTextures, {},
                  "Textures on draw calls whose geometry should have smooth normals generated on the GPU.\n"
                  "This is useful for older games where the geometry may be missing smooth normals, especially when using the VertexShader Capture mechanism.\n"
                  "When a draw call matches, area-weighted smooth normals will be computed from the triangle mesh and used for ray tracing.");
    
  public:
    RTX_OPTION("rtx", bool, showRaytracingOption, true, "Enables or disables the option to toggle ray tracing in the UI. When set to false the ray tracing checkbox will not appear in the Remix UI.");
    RTX_OPTION_ENV("rtx", bool, enableRaytracing, true, "DXVK_ENABLE_RAYTRACING",
                   "Globally enables or disables ray tracing. When set to false the original game should render mostly as it would in DXVK typically.\n"
                   "Some artifacts may still appear however compared to the original game either due to issues with the underlying DXVK translation or issues in Remix itself.");

    RTX_OPTION("rtx", float, sceneScale, 1, "Defines the ratio of rendering unit (1cm) to game unit, i.e. sceneScale = 1cm / GameUnit.");
    RTX_OPTION("rtx", bool, zUp, false, "Indicates that the Z axis is the \"upward\" axis in the world when true, otherwise the Y axis when false.");
    RTX_OPTION("rtx", bool, leftHandedCoordinateSystem, false, "Indicates that the world space coordinate system is left-handed when true, otherwise right-handed when false.");
    // Note: This time is in milliseconds, should be named something like millisecondDeltaBetweenFrames ideally, but keeping it as it is for now.
    RTX_OPTION_ENV("rtx", float, timeDeltaBetweenFrames, 0.f, "RTX_FRAME_TIME_DELTA_MS",
                   "Frame time delta in milliseconds to use for rendering.\n"
                   "Setting this to 0 will use actual frame time delta for a given frame. Non-zero value allows the actual time delta to be overridden and is primarily used for automation to ensure determinism run to run without variance due to frame time fluctuations.");

    RTX_OPTION_FLAG("rtx", bool, keepTexturesForTagging, false, RtxOptionFlags::NoSave, "A flag to keep all textures in video memory, which can drastically increase VRAM consumption. Intended to assist with tagging textures that are only used for a short period of time (such as loading screens). Use only when necessary!");
    RTX_OPTION_ARGS("rtx.gui", float, textureGridThumbnailScale, 1.f, 
                    "A float to set the scale of thumbnails while selecting textures.\n"
                    "This will be scaled by the default value of 120 pixels.\n"
                    "This value must always be greater than zero.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, skipDrawCallsPostRTXInjection, false, "Ignores all draw calls recorded after RTX Injection, the location of which varies but is currently based on when tagged UI textures begin to draw.");
    RTX_OPTION("rtx", bool, deferMaterialCompute, true, "When true, D3D11Rtx::SubmitDraw runs only the cheap material RESOLVE (texture/sampler binding + sourceIsUnlitUI + blendMode) synchronously and schedules the expensive FillMaterialData COMPUTE (PS-cbuffer value reads, emissive/fog/alpha flags, tail) on a geometry worker, finalized on the consumer thread. Reclaims serial SubmitDraw CPU time. Set false to run the whole material fill synchronously (original behaviour).");
    RTX_OPTION("rtx", uint32_t, geometryWorkerThreads, 0, "Number of D3D11 geometry-worker threads (geometry hashing, bounding-box + deferred material compute). 0 = auto = hardware_concurrency - 2 (leaving one core for the game thread and one for the DXVK CS/submit thread). The previous behaviour was cores/2 capped at 6, which starved high-core-count CPUs. Set explicitly to tune.");
    RTX_OPTION("rtx", bool, batchSubmitDrawStages, false, "EXPERIMENTAL. When true, D3D11Rtx (immediate context only) collects every RT geometry commit of the frame into a per-frame arena instead of scheduling a Future per draw, then finalizes ALL deferred per-draw work (material compute, and — via the batch* sub-flags below — hashing/bbox/skinning) in ONE parallel-for at frame end (D3D11Rtx::EndFrame, before injectRTX), re-emitting the commits in original draw order. Removes the per-draw make_shared/Schedule allocator contention that made per-draw material deferral break even. While on, it supersedes rtx.deferMaterialCompute's per-draw material future. Reorders geometry EmitCs to frame end (semantically safe: commitGeometryToRT only builds RT scene state consumed atomically at injectRTX). Set false for the proven per-draw path.");
    // Per-stage sub-flags for rtx.batchSubmitDrawStages. Default true so enabling the
    // parent flag gives the fully-coherent batch (all deferrable stages in the one
    // parallel-for); disable an individual stage to bisect it back to its per-draw path.
    RTX_OPTION("rtx", bool, batchHashes, true, "Only when rtx.batchSubmitDrawStages is on: route geometry hashing (ComputeGeometryHashes) through the frame-end parallel-for instead of a per-draw worker Future. The compute body is shared (identical hashes); finalizeGeometryHashes takes the pre-computed branch on the CS thread. Set false to keep hashing on its per-draw future.");
    RTX_OPTION("rtx", bool, batchBoundingBox, true, "Only when rtx.batchSubmitDrawStages is on: route the object-space bounding-box scan (dynamic position buffers) through the frame-end parallel-for instead of a per-draw worker Future. Static/immutable geometry keeps its synchronous cached path either way. Set false to keep the bbox scan on its per-draw future.");
    RTX_OPTION("rtx", bool, batchSkinning, true, "Only when rtx.batchSubmitDrawStages is on: route the skinned float3x4->Matrix4 bone-palette build through the frame-end parallel-for. The boneHash change-detector stays synchronous (computed at collect). Note the palette build only runs for legacy/single-bone/capture draws (TF2 skins GPU-side and skips it), so this is usually a no-op. Set false to build the palette inline.");
    RTX_OPTION("rtx", bool, shardInstanceProcessing, true, "PHASE 2B (THE_OPTIMISATION_PLAN_2.md). Only when rtx.batchSubmitDrawStages is on: move the scene/instance CPU pipeline (drawCallCache resolve, geometry cache-state decision, instance find/update — the 16-23 ms/frame that [ProcDCS] attributes to dxvk-cs) off the CS thread into flushGeometryBatch: an ordered pre-pass resolves the BlasEntry per draw and groups draws into per-BLAS shards, a second parallel-for runs each shard's geom-decide + instance work with exclusive BLAS ownership, and an ordered tail applies deferred spatial-map writes and dedup-miss continuations in draw order. dxvk-cs keeps only the positional stash copy + capture rebind plus the GPU-record residue (bake dispatches, buffer-table update, surface buffer binding, billboards, OMM). A full CS-thread drain (SynchronizeCsThread, overlapped with the existing Phase B parallel-for) at flush start guarantees strict alternation between CS-side scene reads (frame N) and game-thread scene writes (frame N+1). Draws the pre-pass cannot prove shard-safe (mesh replacements, terrain bake, sky handling) take the unchanged legacy CS path, in original draw order. Set false for the proven CS-side path — byte-identical behavior, no rebuild needed. VERIFY per plan Sec 15: [Shard2b] shards ~= uniqueBlas, deferred~0; [ProcDCS] instMs/geomMs collapse; [Perf.Report] HYGIENE inst/matNew/drawsCommit unchanged.");
    // NV-DXVK [Perf.PushInst] -- PHASE 2, push-not-poll. See the block comment on
    // InstanceManager::fanoutRecordFingerprint for the mechanism and the contract.
    RTX_OPTION("rtx", bool, pushInstanceRecords, false, "PHASE 2. Persistent per-batch instance records for the fanout path. A fanout draw whose placement transforms, base object-to-world, material and camera are all byte-identical to the last frame it was submitted in resolves to exactly the instances it resolved to before, so the per-placement find/update poll is replaced by a bulk frame-id stamp over the recorded list. That stamp is the ONLY thing GC requires (clauseLifetime is m_frameLastUpdated + keepN <= currentFrame), which is why keep-alive without reprocessing is expressible at all. Does nothing unless the scene is actually static; measured REDUNDANT=97% in steady-state TF2. START WITH rtx.pushInstanceRecordsVerify ON.");
    RTX_OPTION("rtx", bool, pushInstanceRecordsVerify, true, "SAFETY GATE for rtx.pushInstanceRecords, and the default. When on, the record is built and its prediction is scored but NOTHING is skipped -- the placement loop always runs, and the recorded instance list is compared pointer-for-pointer against what the loop actually produced. A divergence is logged as [Perf.PushInst.FAIL] with the vertex-shader hash and the differing index. FAIL must read 0 over a real session, including a level transition and a combat frame, before this is turned off. FAIL>0 means the fingerprint is incomplete -- some input the resolution reads is not hashed -- and the failure mode is an instance in the wrong place or a stale instance held alive. Frame time under verify is meaningless by construction: the skip is what the feature is for, and verify exists to not take it.");
    RTX_OPTION("rtx", uint32_t, pushInstanceRecordsMaxBatches, 4096, "Only when rtx.pushInstanceRecords is on: cap on live fanout batch records. Each record holds an instance-pointer vector for one (vertex shader, geometry, material) batch; the cap bounds the map against a scene that churns batch identity every frame rather than reusing it. On overflow the oldest-served record is evicted and its batch falls back to the placement loop, which is correct but unaccelerated -- watch evict= on [Perf.PushInst] and raise this rather than letting it thrash.");
    RTX_OPTION("rtx", bool, parallelInstanceFanout, false, "EXPERIMENTAL. Splits SubmitInstancedDraw's per-instance transform-build loop into a parallel DECODE followed by a serial COMPACT, dispatched on the same rtx.geometryWorkerThreads pool that rtx.batchSubmitDrawStages uses.\n"
               "WHY. [Perf.DrawEntry] deOnDraw minus [Perf.SubmitDraw] wallUs is a large block of the frame thread that no SubmitDraw stage marker reaches; [Perf.InstDraw] localises it to build_ms, and loop_ms is the bulk of that. The loop reads m_instBufCache and m_t31ReadCache -- both already bulk-copied to CPU-side arrays before it opens -- so it touches no live D3D11 state, no mapped write-combined memory, and has no ordering dependency between instances. It is the only large data-parallel block left on the pole thread.\n"
               "WHY TWO PHASES. The loop cannot be a plain parallel-for: it COMPACTS (instances failing the OOB / non-finite / zero-row checks are skipped, so the output index is not the loop index), it captures m_fanoutRawT0 from the FIRST surviving instance, and it reads tforms->back() when an instance has no previous-frame matrix. Phase 1 decodes instance i into a pre-sized slot with a status byte and touches nothing shared; phase 2 walks the slots IN ORDER and does the appends, the first-survivor capture and the four m_geomDiagFanoutInst* counters. Output is therefore identical to the serial path, element for element.\n"
               "SAFETY. Falls back to the serial loop whenever any per-instance diagnostic is live ([MtnFanoutIdx], the idx/t31 dump lines, RTX_D3D11_DIAG's [T31Scale], rtx.dumpFanoutInstanceStruct), so those paths stay byte-identical and are never run off-thread.\n"
               "VERIFY. [Perf.InstDraw] loop_ms falls while instPerCall, calls and instances are unchanged, and parInst/parCalls on the same line show how much of the work actually took the parallel path. A changed instPerCall or a changed dedup outcome in [FindStage] means the compact is wrong -- back it out.");
    RTX_OPTION("rtx", uint32_t, parallelInstanceFanoutMinInstances, 64, "Only when rtx.parallelInstanceFanout is on: smallest instanceCount that takes the parallel path. Below this the dispatch and join cost more than the decode. The mean instances-per-call is modest, so this threshold decides whether the feature touches most of the work or almost none of it -- read parInst vs instances on [Perf.InstDraw] and tune, do not guess.");
    RTX_OPTION_FLAG("rtx", bool, logMemoCeiling, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Measures the upper bound for SubmitDraw memoization: per frame, of the draws reaching commit, how many are identical to a draw from the PREVIOUS frame and could therefore be replayed from cache instead of re-running the full inject pipeline (ExtractTransforms recon, cull scan, material snapshot). Emits [MemoCeiling] with geomStable (VS + bound VB/IB identity matched last frame => geometry-dependent work is cacheable) and fullStable (that PLUS the bound VS transform-cbuffer bytes matched => the entire commit is replayable). The gap between them is the 'same object, camera moved' set. Use this number to decide whether building the memoization is worth it.");
    RTX_OPTION_FLAG("rtx", bool, logDupPass, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Characterizes WITHIN-frame geometry re-injection: many draws reaching commit share the same geometry (VB/IB) as an earlier draw THIS frame (multi-pass). Emits [DupPass] per frame with depthOnly (color write mask == 0 — z-prepass / shadow, which RT arguably does not need) and, for the duplicates, a breakdown of what distinguishes them from the first sighting of that geometry: dupDepthOnly (color write off), dupDiffCam (different cameraType — sub-view like 3D-skybox / water reflection), dupDiffPs (different pixel shader — multi-material pass), dupSame (identical pass — truly redundant). Decides which redundant passes can be filtered before commit.");
    RTX_OPTION("rtx", bool, filterDupSameDraws, false, "EXPERIMENTAL. Skips the RT commit of within-frame draws that are BYTE-REDUNDANT with an earlier draw of the same frame. [DupPass] (2026-08-07, recorded in rtx.conf) measured 14.6% of committed draws as exact duplicates (same geometry, same cameraType, same PS; depthOnly=0 so none are z-prepass), costing ~4.5-5.8 ms/frame of dxvk-cs commit work whose output the instance manager dedups anyway.\n"
               "THE KEY IS DELIBERATELY STRICTER than [DupPass]'s classifier, because that key cannot prove redundancy: it ignores transforms (the same mesh drawn at two positions is two INSTANCES, not a duplicate) and materials (same PS, different textures = multi-material pass). This filter's identity is: geometry (VS + VB/IB ptr/off/stride + counts) + cameraType + PS hash + the full objectToWorld bytes + the bound PS SRV identities + blend-state identity. v1 scope guards: instanced-fanout draws (instancesToObject) and blending-enabled draws are NEVER filtered -- fanout duplicates would need the per-instance transform arrays hashed, and coincident blended surfaces can be intentional (additive particles).\n"
               "SAFETY ARGUMENT: a draw identical in ALL of the above maps to the same RtInstance via findSimilarInstance (same geometry, same transform) with an identical material -- its commit is a redundant re-update of state that cannot have changed ([Perf.UpdInst] REDUNDANT=97%, matChg=0%). The instance is already alive and frame-touched from the first sighting, so lifetime/anti-culling is unaffected.\n"
               "VERIFY: [DupFilter] line prints skipped/frame. Regression check: drawsCommit on [Perf.Report] HYGIENE should fall by exactly the skipped count while inst stays ~equal; any visual change means the key is too weak -- report it, set this false.");
    RTX_OPTION("rtx", bool, fastInstanceUpdate, true, "Fast path through InstanceManager::updateInstance for provably-unchanged instances -- the dirty-list realisation of the [Perf.UpdInst] REDUNDANT=97% finding (recorded in rtx.conf ~1690-1950): 15,500 per-instance visits/frame load cold RtInstance state to conclude nothing changed and do nothing, ~12 ms on dxvk-cs.\n"
               "AN INSTANCE TAKES THE FAST PATH ONLY WHEN ALL of the following hold: the shared instance-state key (m_instStateKey, the same digest that authorises the surf/tail skips) matches; the per-draw fast bits (winding, proj parity, RT-target, sub-view flags) match those stored at the last full update; the incoming objectToWorld is BYTE-IDENTICAL to the stored one; the SpatialMap is provably in sync (propId-keyed and cacheHash==propId, or matrix-keyed and this frame's lookup hash == cacheHash); and the instance is not in an excluded population (ViewModel camera, player model, unordered/billboard, decal, skinned prev-positions, RayPortal, first frame after creation). Option flips are self-healing: a once-per-frame digest of every option the skipped region reads forces one full-slow frame whenever any of them changes.\n"
               "WHAT THE FAST PATH STILL DOES, every frame: buffer rebind (processInstanceBuffers -- slice renaming), frame/GC liveness, camera registration, picking value, isInsideFrustum, prev-transform advance, textureTransform/clipPlane copies, the m_isHidden promotions (fog/coverage), and the NON-skippable event handlers (OMM). Everything else is provably a rewrite of identical bytes.\n"
               "VERIFY: [Perf.FastInst] prints hit/slow-reason counts every 300 frames. Expect fast/frame ~= 12-14k of ~15.5k inst. Regression check: [Perf.Report] dxvk-cs processSceneObject falls by several ms while HYGIENE inst/drawsCommit stay ~equal. Any visual anomaly: set False (conf, no rebuild) and report which [Perf.FastInst] reason bucket changed.");
    RTX_OPTION_FLAG("rtx", bool, logMapGate, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Emits [MapGate] per frame: onTransformChanged / teleport1 / nullBlas / unsetFrame call counts and the spatial-map write balance (mapWritesExpected == mapWrMove + mapWrInsert + mapSkipInSync). Catches a spatial-map write that should have happened and did not. DEFAULT OFF, AND THE DEFAULT IS THE POINT: mapGateAccount() is called unconditionally from RtInstance::onTransformChanged, which [MapGate] itself measured at ~15,441 calls/frame, and each call did 2-4 atomic RMWs on a handful of shared static counters. That is ~30-60k contended cache-line operations per frame on dxvk-cs, the thread the frame is sized by. The same shape in the accel manager's world-extent census (an unconditional per-instance std::mutex plus atomics, 8,375x/frame) was measured on 2026-08-07 to cost far more than its arithmetic: gating it took dxvk-cs from 70.5 to ~46 ms/frame and sped up UNRELATED functions on that thread by 25-30% per call ([Perf.SceneObj] update 0.827 -> 0.568 us at an identical call count), because the cost is cache-coherence traffic, not instructions. Turn this on only to audit map writes, and expect to pay for it while it is on.");
    RTX_OPTION_FLAG("rtx", uint32_t, perfReportFrames, 0, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Emit the assembled [Perf.Report] frame breakdown every N frames. 0 = off. 50 is a reasonable value; see the cadence note below before going lower.\n"
               "WHAT IT IS. ~90 [Perf.*] instruments exist, each correct, each PARTIAL: its own window, its own cadence, its own units, on one of six concurrent timelines. This assembles them into one nested table -- frame thread, dxvk-cs, GPU -- runs cross-validation between instruments that measure the same span by different routes, and prints an automatic bottleneck verdict.\n"
               "IT MEASURES NOTHING. Every value is already computed by an existing probe for its own line; the emit sites publish() what they just calculated. Adding a row must never add a clock read. Cost when off is one option read per frame.\n"
               "THE VERDICT RULE. frame ~= max(frame thread, dxvk-cs); the GPU is a third clock and is only the pole when its IDLE is small in absolute terms. Do NOT call GPU-bound from [Perf.Gpu] (idleMs+fenceWaitMs+reapMs == frame is true by construction) or from [Perf.GpuPass] outsideRtMs (GPU-timeline time outside the RT passes, which INCLUDES idle). Both have produced a wrong GPU-bound call in this repo's history.\n"
               "THE NUMBER TO READ. 'realised' = min(size of a target, poleMs - secondMs). A saving on the pole is worth nothing past the point where the other thread becomes the pole, so an 8 ms win on a thread with 2 ms of slack is a 2 ms win. That cap is what decides what is worth working on and it was never computed by hand.\n"
               "CADENCE. Most source probes refresh on a ~5 s timer, which at 20 fps is ~100 frames. At N=50 roughly half the rows will repeat the previous report's values; each row carries no age field but the SKIP lines in the cross-validation section name any source that never published at all (usually a probe gated off). Set N >= the source cadence in frames for fully fresh tables.\n"
               "REQUIRES the probes that feed it. rtx.perfThreadCensus for the frame-thread block, rtx.logSubmitStall for [Perf.DrawEntry]/[Perf.InstDraw]/[Perf.LoopCut], rtx.perfSceneObjSplit for the updateInstance split, rtx.perfGpuStageSerialize for clean per-pass GPU attribution. Rows whose source is off print n/a rather than 0 -- a gated-off probe must never read as free.");
    RTX_OPTION_FLAG("rtx", bool, logSurfaceGeomDiag, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Enables the three first-sighting censuses inside InstanceManager::processInstanceBuffers -- [VguiSurface], [TC1Surface] and [BlasGeom]. Each logs one line per NEW (buffer index, offset, stride, flag) tuple, so they answer 'did the lightmap-UV / VGUI / texcoord-layout propagation reach the surface' and then have nothing left to say for the rest of the run.\n"
               "DEFAULT OFF, AND THE DEFAULT IS THE POINT. Two of the three were UNGATED, in a function that runs once per INSTANCE (~15,500/frame on dxvk-cs, the thread the frame is sized by). Per instance they paid: a key build, a std::lock_guard<std::mutex> on a function-static mutex, and an std::unordered_set insert ([TC1Surface]); plus a second key build and a second unordered_set insert ([BlasGeom]). That is ~15,500 mutex acquire/release pairs and ~31,000 node-based container probes per frame to re-answer a question that saturated in the first second of the run.\n"
               "This is the SAME SHAPE as the accel manager's world-extent census (unconditional per-instance std::mutex + 8-corner transform, 8,375x/frame), whose gating on 2026-08-07 took [Perf.Merge] loop from 9.0-13.8 ms to 4.3-5.9 ms. It is NOT the shape of rtx.logMapGate or tallyReorderedPush, which were gated the same day and returned zero -- those were memory_order_relaxed atomics written by one thread, 1-2 ns each, staying resident in that core's L1. A mutex plus a heap-scattered container probe is ~100-150 ns. Do not generalise one result to the other.\n"
               "GATING COVERS THE WHOLE BLOCK -- key construction, mutex and container included, not just the Logger::info. Wrapping only the emit was the exact defect the world-extent census had.\n"
               "A NOTE ON THREADING: the [BlasGeom] set had NO mutex while the [TC1Surface] set immediately above it did, so the two blocks disagreed about whether processInstanceBuffers is reached from more than one thread. It is -- InstanceManager::m_fanoutPrevHitCount is atomic because findSimilarInstance is reachable from the scene manager's draw-processing threads, and those reach here too. [BlasGeom] is locked now; before that it was an unsynchronised unordered_set::insert, i.e. container corruption waiting on two threads hitting a new tuple at once, which only the census's rarity kept quiet.\n"
               "VERIFY. This option is NoSave and runtime-toggleable on purpose: A/B it WITHIN one process (six windows per state, alternating) and watch [Perf.SceneObj] update us/call and dxvk-cs ms/frame, holding inst (~15.5k) and draws (~1063) fixed. Between-run spread at identical workload is ~45%, which is how the world-extent census came to be mis-sized by 6x -- never compare one process against another. Deliberately a separate option from rtx.logGeomDiag: that one also drives the world-extent census and tallyReorderedPush, and toggling it would move ~4 ms of unrelated work through the middle of the measurement.");
    RTX_OPTION_FLAG("rtx", bool, hoistSurfaceBufferBinding, true, RtxOptionFlags::NoSave, "PERF. Resolves the BLAS buffer binding that InstanceManager::processInstanceBuffers writes into RtSurface ONCE PER DRAW (into DrawScopedState::buffers) instead of once per INSTANCE. ~1,063 derivations per frame instead of ~15,500; the per-instance write of all ~23 surface fields is unchanged, because it has to be.\n"
               "WHY THIS IS SAFE, AND WHY THE OBVIOUS VERSION IS NOT. blas.modifiedGeometryData is written ONLY in SceneManager::processDrawCallState -- processGeometryInfo, then the smooth-normals and skinning dispatches -- and all of those complete BEFORE processSceneObject/processSceneObjectFanout is called. Nothing in rtx_instance_manager.cpp writes it. So within one draw's placement loop the binding is immutable BY CONSTRUCTION, which is what makes this need no freshness key at all.\n"
               "Do NOT 'improve' this into a per-BLAS cache keyed on BlasEntry::frameLastUpdated. Several draws resolve to one BlasEntry per frame (that is what BlasEntry::drawCount counts) and each pairing can rewrite the binding, while frameLastUpdated is only stamped on KBuildBVH/kUpdateBVH -- a Map(WRITE_DISCARD) slice rename does not stamp it. Such a cache would hand instances a freed slice, which is the failure mode that froze the game on 2026-08-07 when the state key was used to skip the rebind entirely. Per-DRAW needs no key; per-BLAS needs one that does not exist.\n"
               "Set false to restore the per-instance derivation. The paths are value-identical by construction, so this exists for A/B measurement, not as a safety valve -- if the two ever LOOK different, the immutability claim above has been broken by a new write to modifiedGeometryData and that is the thing to fix.");
    RTX_OPTION_FLAG("rtx", uint32_t, cbStageEntries, 8, RtxOptionFlags::NoSave, "PERF. Number of ring slots searched and filled by the per-thread constant-buffer staging cache (stagedCbBytes, d3d11_rtx.cpp). Clamped to [1, 32]; storage is fixed at 32 so this is a live sweep inside ONE process, which is the point -- between-run spread at identical workload is ~45% in this build and has mis-sized two measurements already.\n"
               "WHAT IT IS FOR. [Perf.WcCopy] bills ~3.4 MB/frame of write-combined reads to a single frame-thread call site at 6.31 KB per call, i.e. a cache with a ~0% hit rate re-staging whole constant buffers on every call, on the thread the frame is sized by (frame thread 43.55 ms @ 93.4% busy vs dxvk-cs 39.25 ms). Raising this only helps if the misses are CAPACITY misses.\n"
               "READ [Perf.CbStage] BEFORE TOUCHING IT, and confirm the win by MECHANISM (hitPct rising) not by a timing bucket. missCap high -> the working set exceeds the ring and raising this is the whole fix. missGen high -> the buffer is still resident but Map(WRITE_DISCARD) bumped GetContentGeneration() since it was staged, the (buffer, generation) key can never repeat, and NO ring size can help -- the fix is then to stage only the bytes each consumer reads. dupSlots high alongside missCap means the round-robin insert is holding one buffer in several slots and deduping the insert buys more than widening does.\n"
               "COST OF RAISING IT. Each slot holds its own std::vector staged copy of up to 64 KB per thread, and the hit path is a linear scan, so 32 ways is 32 pointer compares on a miss. Both are negligible next to one 6.31 KB write-combined read; the reason not to raise it blindly is that a ring change that does not move hitPct proves nothing and hides the real key defect.");
    RTX_OPTION_FLAG("rtx", bool, perfSubmitCpuCycles, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Enables the per-draw QueryThreadCycleTime pair in SubmitDraw's SubmitCpuGuard (d3d11_rtx.cpp ~22784/22789), which produces cpuCycles on [Perf.SubmitDraw.acc] and the cycle half of the [SdStall] wall-vs-cpu split.\n"
               "DEFAULT OFF BECAUSE QTCT IS EXPENSIVE AND THIS CALLED IT ON EVERY DRAW, UNGATED. Measured in this tree at ~0.44 us/call -- a per-stage QueryThreadCycleTime experiment was already REFUTED and reverted on that number (+5 ms/frame). SubmitCpuGuard kept two calls per draw, constructor and destructor: at ~1,080 draws/frame that is ~0.95 ms/frame on the FRAME THREAD, which is the pole. steady_clock::now() by contrast is ~41 ns and stays.\n"
               "WHAT YOU KEEP WITH IT OFF: everything wall-clock. wallUs, the per-draw counts, every markStg/markXt bucket and therefore the whole [Perf.SubmitDraw] breakdown and [Perf.Report] frame-thread tree are unaffected -- they are steady_clock, not QTCT. What you lose is cpuCycles and the SdStall cycle-vs-wall attribution, i.e. the 'is this stage blocking or computing' question.\n"
               "TURN IT ON when you specifically need to separate a stall from real CPU work in a stage; leave it off for any capture where the frame time itself is the measurement, because it perturbs exactly the thread being measured.");
    RTX_OPTION_FLAG("rtx", bool, logCullProbes, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Enables the per-frame [OccProbe] and [PitchProbe] lines in InstanceManager (rtx_instance_manager.cpp ~1628).\n"
               "DEFAULT OFF, AND THE DEFAULT IS THE POINT. [OccProbe] walks the ENTIRE live instance list on every gameplay frame -- ~15,500 instances on this map -- reading each instance transform and dotting its origin against the camera plane to bin front/behind/behindFar, then emits two Logger::warn lines with full str::format. Its only gate was 'is this a gameplay frame', so it ran permanently rather than for a capture. Same shape as the censuses behind rtx.logSurfaceGeomDiag: a per-instance loop re-answering a question that is only asked during an investigation.\n"
               "GATING COVERS THE WHOLE BLOCK -- camera resolve, the ~15,500-iteration classification loop, and both emits. Wrapping only the Logger::warn would leave the loop, and the loop is the cost.\n"
               "Turn it on for a capture when you need the off-screen-occluder population binned by pitch/yaw, then turn it back off.");
    RTX_OPTION_FLAG("rtx", bool, logEngineCamFrame, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Enables the per-frame [EngineCamFrame] trace of the engine-hook main camera pose, projection and latch counters (d3d11_rtx.cpp ~40394).\n"
               "One Logger::warn per gameplay frame carrying ~14 formatted floats, self-capped at 6000 lines. Cheaper than rtx.logCullProbes because there is no per-instance loop, but it is still a format-and-write on the frame thread every frame, and the transient it was written to catch is a first-10-to-20-seconds-of-level event -- it has nothing to say for the rest of a session and cannot be turned off without this option.");
    RTX_OPTION_FLAG("rtx", bool, logReskinProbe, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Enables the [ReskinProbe] per-VS re-skinning aggregate in SceneManager::processGeometryInfo (rtx_scene_manager.cpp ~950).\n"
               "DEFAULT OFF BECAUSE THE COST IS PER DRAW, NOT PER LINE. The reporting is throttled to one window every 10 frames, but the ACCUMULATION runs on EVERY skinned draw and takes a std::lock_guard on a function-static mutex plus an unordered_map probe to do it -- on the order of 1,300 mutex acquire/release pairs and hash lookups per frame to feed a line printed once per 10 frames. That is exactly the defect the rtx.logSurfaceGeomDiag censuses had, and throttling the EMIT does not touch it.\n"
               "Turn it on when a skinned mesh is visibly stale and you need to know which VS stopped re-skinning (hashChanged=0 over a window while the model is visibly animating IS the bug). NOTE: [ReskinProbe] is also listed in log.cpp's kFilteredTags, so enabling this option may additionally need rtx.logDenyTags to carry -[ReskinProbe] to un-filter the output.");
    RTX_OPTION_FLAG("rtx", bool, logSubmitStall, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Finds WHERE the SubmitDraw wall time goes. The outer OnDraw* timer (submitDrawAccUs) is ~2.5x the inner instrumented wallUs, and single-draw maxes hit 11-62ms, but the 1-in-64 [SdStall] sampler shows wall==cpu everywhere — so the cost is a few OUTLIER draws it misses. This logs every OnDraw*/OnDrawIndexed*/instanced call whose whole-call wall time exceeds rtx.submitStallUs as [SubmitStall] (frame draw-ordinal, type, VS hash, prim/instance counts, us), plus a per-frame [SubmitStall.Frame] roll-up (draws / #stalls / stallUs / totalUs / stall%). The ordinal shows whether the slow draws cluster at frame start (GPU sync) or are specific VSes / instanced fanout (map / backpressure blocks).");
    RTX_OPTION_FLAG("rtx", uint32_t, submitStallUs, 1500, RtxOptionFlags::NoSave, "Threshold in microseconds for rtx.logSubmitStall: an OnDraw* call whose total wall time is >= this is logged as a [SubmitStall] outlier. Default 1500 (1.5ms).");
    RTX_OPTION_FLAG("rtx", bool, capturePhase2, false, RtxOptionFlags::NoSave, "DIAGNOSTIC / plumbing (no behaviour change). GPU-driven-injection Phase 2 (HANDOFF_GPU_DRIVEN_INJECTION.md §6). At the RT commit point — the confirmed-injected funnel past every SubmitDraw early-return — snapshot a thin per-draw record into a per-frame arena: the draw's layoutId (indexes the Phase-1 VsLayoutTable), the raw camera cbuffer inputs the future GPU pass will read (proj + view matrix bytes, copied by value), the per-draw o2w source class, and the CPU-resolved matrices as Phase-3 ground truth. Runs alongside the existing CPU path (both execute; nothing consumes the arena yet). Emits a per-frame [Phase2] line verifying #captured == #injected and flagging any committed draw that arrived WITHOUT a layoutId (a capture gap). Default off — zero cost until enabled for verification or Phase 3.");
    RTX_OPTION_FLAG("rtx", bool, logPhase0Descriptor, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). GPU-driven-injection Phase 0 feasibility gate (HANDOFF_GPU_DRIVEN_INJECTION.md). For every injected (non-UI-fallback) draw, splits the resolved reconstruction into two tiers: TIER 1 the per-VS LAYOUT (classifier kind, projection & view cb slot/offset/stage, column-major, texcoord encoding, IlFacts) which the Phase-1 descriptor table bakes per-VS and must be stable; TIER 2 the per-DRAW objectToWorld SOURCE class (identity / t31-per-instance / bones / cbuffer-field) which the Phase-2 capture record carries per draw. Keyed by VS hash it emits [Phase0] NEW (first sighting), [Phase0] UNSTABLE-LAYOUT (a VS whose TIER-1 layout moved — the real GPU-descriptor blocker), [Phase0.MultiMode] (a VS drawing from >1 o2w source — per-draw, expected), and a throttled [Phase0.Summary] with uniqueVS / unstableLayoutVS / multiModeVS counts plus each multi-mode VS's source breakdown. Confirms the per-VS layout is bounded (~dozens) and stable before any GPU shader work begins.");
    RTX_OPTION_ARGS("rtx", DlssPreset, dlssPreset, DlssPreset::On, "Combined DLSS Preset for quickly controlling Upscaling, Frame Interpolation and Latency Reduction.",
                    args.environment = "RTX_DLSS_PRESET",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", NisPreset, nisPreset, NisPreset::Balanced, "Adjusts NIS scaling factor, trades quality for performance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", TaauPreset, taauPreset, TaauPreset::Balanced,  "Adjusts TAA-U scaling factor, trades quality for performance.",
                    args.flags = RtxOptionFlags::UserSetting);
    static void graphicsPresetOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", GraphicsPreset, graphicsPreset, GraphicsPreset::Auto, "Overall rendering preset, higher presets result in higher image quality, lower presets result in better performance.",
                    args.environment = "DXVK_GRAPHICS_PRESET_TYPE",
                    args.onChangeCallback = &graphicsPresetOnChange,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ENV("rtx", RaytraceModePreset, raytraceModePreset, RaytraceModePreset::Auto, "DXVK_RAYTRACE_MODE_PRESET_TYPE", "");
    RTX_OPTION_FLAG("rtx", bool, lowMemoryGpu, false, RtxOptionFlags::NoSave | RtxOptionFlags::UserSetting, "Enables low memory mode, where we aggressively detune caches and streaming systems to accomodate the lower memory available.");
    RTX_OPTION_ARGS("rtx", float, emissiveIntensity, 1.0f, "A general scale factor on all emissive intensity values globally. Generally per-material emissive intensities should be used, but this option may be useful for debugging without needing to author materials.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx", float, fireflyFilteringLuminanceThreshold, 1000.0f, "Maximum luminance threshold for the firefly filtering to clamp to.",
                    args.minValue = 0.0f);
    RTX_OPTION("rtx", float, secondarySpecularFireflyFilteringThreshold, 1000.0f, "Firefly luminance clamping threshold for secondary specular signal.");
    RTX_OPTION_ARGS("rtx", float, vertexColorStrength, 0.6f,
                    "A scalar to apply to how strong vertex color influence should be on materials.\n"
                    "A value of 1 indicates that it should be fully considered (though do note the texture operation and relevant parameters still control how much it should be blended with the actual albedo color), a value of 0 indicates that it should be fully ignored.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION("rtx", bool, vertexColorIsBakedLighting, true, "If true, brightness contribution will be removed from the vertex color by dividing each component by the largest component.");
    RTX_OPTION("rtx", bool, ignoreAllVertexColorBakedLighting, false, "If true, all baked lighting bound to all vertex colors will be ignored.");
    RTX_OPTION("rtx", bool, allowFSE, false,
               "A flag indicating if the application should be able to utilize exclusive full screen mode when set to true, otherwise force it to be disabled when set to false.\n"
               "Exclusive full screen may see performance benefits over other fullscreen modes at the cost of stability in some cases.\n"
               "Do note that on modern Windows full screen optimizations will likely be used regardless which in most cases results in performance similar to exclusive full screen even when it is not in use.");
    RTX_OPTION("rtx", std::string, baseGameModRegex, "", "Regex used to determine if the base game is running a mod, like a sourcemod.");
    RTX_OPTION("rtx", std::string, baseGameModPathRegex, "", "Regex used to redirect RTX Remix Runtime to another path for replacements and rtx.conf.");
    RTX_OPTION("rtx", bool, disableAMDSwitchableGraphics, true,
               "A flag indicating if Remix should attempt to disable AMD's switchable graphics Vulkan layer (VK_LAYER_AMD_swichable_graphics).\n"
               "Due to how some older AMD drivers filter devices exposed to Vulkan it is possible for Remix to see no valid GPUs on a machine when using an integerated AMD GPU with a dedicated Nvidia GPU (for instance a laptop).\n"
               "This is because on such machines both Nvidia Optimus and AMD switchable graphics attempt to filter the device list to promote their respective GPUs, but rather than leaving at least one device all end up filtered out.\n"
               "To work around this issue, Remix can attempt to disable the AMD switchable graphics layer which should eliminate this buggy filtering. As such, this option should generally remain enabled.\n"
               "If this causes an undesired GPU to be selected (e.g. if for some reason you want to force Remix to run on an integerated AMD GPU via the switchable graphics layer), then this option should be disabled.");


    // Shader Compilation
    struct Shader {
      friend class ShaderManager;

      // Note: Shader recompilation is only useful with a development setup for the most part and is disabled when REMIX_DEVELOPMENT is not defined,
      // so these options will not take effect in such builds. They are however still included rather than ifdeffed out to keep consistent options documentation
      // across builds.
      RTX_OPTION("rtx.shader", bool, asyncSpirVRecompilation, true,
                 "When set to true runtime shader recompilation will recompile shaders to SPIR-V asynchronously rather than blocking until complete.\n"
                 "Do note that despite setting this option the actual compilation of the shader from SPIR-V to the ISA will still be blocking as only the prewarming process can handle this step asynchronously for now.\n"
                 "Generally this option should remain enabled, though disabling it may be useful for CI where deterministic behavior is needed, and may be useful to maximize performance at the cost of blocking (by not having application running while compiling to SPIR-V).\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation.");
      RTX_OPTION("rtx.shader", bool, recompileOnLaunch, false,
                 "When set to true runtime shader recompilation will execute on the first frame after launch.\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation. Also see rtx.useLiveShaderEditMode for a similar option which auto-detects shader changes instead.");
      RTX_OPTION("rtx.shader", bool, useLiveEditMode, false,
                 "When set to true shaders will be automatically recompiled when any shader file is updated (saved for instance) in addition to the usual manual recompilation trigger.\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation.");

      RTX_OPTION_ENV("rtx.shader", bool, prewarmAllVariants, false, "RTX_PREWARM_ALL_VARIANTS",
                     "When set to true, all variants of shaders will be prewarmed at launch. Only takes effect when rtx.initializer.asyncShaderPrewarming is set to true.\n"
                     "By default Remix only prewarms shaders which may actually be used at runtime or are accessible by user-facing graphics menus rather than all shader variants accessible by changing options in the developer menu.\n"
                     "This has the benefit of minimizing shader compilation cost for typical users, but may cause shader compilation stalls when changing various options in the developer menu. As such, this option is useful to enable during development to minimize these stalls.\n"
                     "Do note however that enabling this option will have a significant performance impact whenever shaders are uncached (e.g. on first load) due to requiring many more shaders to be compiled. As such using the enviornment variable to set this option locally on a developer's machine is recommended over a configuration file change to ensure it is not accidently enabled for users.");
      RTX_OPTION_ENV("rtx.shader", bool, enableAsyncCompilation, true, "RTX_ENABLE_ASYNC_COMPILATION",
                 "When set to true shader compilation (especially that of prewarming) will be done asynchronously rather than blocking.\n"
                 "Typically shader prewarming with async finalization is done to attempt to compile all required shader variants before they are used, often by overlapping this work with a startup sequence (e.g. a game's loading screen). Often times however this prewarming takes longer than the time available, or an application may not have a startup sequence to begin with and immediately begin using Remix shaders.\n"
                 "To accomodate this, async shader compilation allows for this work to be done asynchronously to avoid blocking the application at the cost of being unable to render anything until the process is complete.\n"
                 "This is typically better choice than blocking however and is recommended to be enabled as on Windows Remix blocking will cause the application to stop responding, making it seem as if the application has crashed if shader compilation takes a long time. Additionally, when combined with rtx.shader.enableAsyncCompilationUI the progress of the compilation process can be shown to the user as a UI, improving user experience.\n"
                 "The main downside to this approach is that when blocking shader compilation is allowed to take up more of the CPU, whereas async shader compilation will have to compete with the application which can make compilation take slightly longer than it would otherwise (especially true if the application's framerate is uncapped).\n"
                 "To mitigate this, Remix can optionally throttle the application during async compilation via rtx.shader.asyncCompilationThrottleMilliseconds to ensure enough time is available for compilation.\n"
                 "Finally, a more minor downside is that when async shader compilation is in use Remix currently has no way of keeping the application in a startup sequence (e.g. keeping a game on its loading screen) while it waits for shaders to compile.\n"
                 "This will mean for instance a game's menu may be active but not be able to render until the compilation is complete, rather than blocking on the loading screen and transitioning to the menu only once all shaders are loaded. Not blocking the application is typically better for user experience regardless though as long as some sort of progress UI is displayed to indicate what is happening.");
      RTX_OPTION("rtx.shader", bool, enableAsyncCompilationUI, true,
                 "Enables a UI message when async shader compilation is in progress to indicate the current compilation progress. Only takes effect when rtx.shader.enableAsyncCompilation is true.\n"
                 "This should usually be enabled as providing information to the user about the current progress of compilation is useful. May be disabled however for automated testing purposes if the nondeterministic behavior of the UI's rendered text interferes with testing.");
      RTX_OPTION("rtx.shader", std::uint32_t, asyncCompilationThrottleMilliseconds, 33,
                 "Specifies a time in milliseconds to throttle each application frame when async shader compilation is in progress. Set to 0 to disable, and only takes effect when rtx.shader.enableAsyncCompilation is true.\n"
                 "This generally should be set to a value low enough to not impact the application framerate significantly (especially if non-ray traced visuals are capable of being displayed by the application while loading, e.g. an intro video), but also high enough to get the desired shader compilation performance (especially relevant if the application is fairly heavy on the CPU during async shader compilation, or on CPUs with few hardware threads).");
    } shader;

    struct RaytracedRenderTarget {
      RTX_OPTION("rtx.raytracedRenderTarget", bool, enable, true, "Enables or disables raytracing for render-to-texture effects.  The render target to be raytraced must be specified in the texture selection menu.");
    } raytracedRenderTarget;

    struct ViewModel {
      friend class ImGUI;
      RTX_OPTION("rtx.viewModel", bool, enable, true, "If true, try to resolve view models (e.g. first-person weapons). World geometry doesn't have shadows / reflections / etc from the view models.");
      RTX_OPTION("rtx.viewModel", float, rangeMeters, 1.0f, "[meters] Max distance at which to find a portal for view model virtual instances. If rtx.viewModel.separateRays is true, this is also max length of view model rays.");
      RTX_OPTION("rtx.viewModel", float, scale, 1.0f, "Scale for view models. Minimize to prevent clipping.");
      RTX_OPTION("rtx.viewModel", bool, enableVirtualInstances, true, "If true, virtual instances are created to render the view models behind a portal.");
      RTX_OPTION("rtx.viewModel", bool, perspectiveCorrection, true, "If true, apply correction to view models (e.g. different FOV is used for view models).");
      RTX_OPTION("rtx.viewModel", float, maxZThreshold, 0.08f, "If a draw call's viewport has max depth less than or equal to this threshold, then assume that it's a view model. Default 0.08 catches Source-engine style viewmodel passes (MaxDepth=0.05) without touching the main world pass (MaxDepth=0.1).");
    } viewModel;

    struct PlayerModel {
      friend class ImGUI;
      RTX_OPTION("rtx.playerModel", bool, enableVirtualInstances, true, "");
      RTX_OPTION("rtx.playerModel", bool, enableInPrimarySpace, false, "");
      RTX_OPTION("rtx.playerModel", bool, enablePrimaryShadows, true, "");
      RTX_OPTION("rtx.playerModel", float, backwardOffset, 0.f, "");
      RTX_OPTION("rtx.playerModel", float, horizontalDetectionDistance, 34.f, "");
      RTX_OPTION("rtx.playerModel", float, verticalDetectionDistance, 64.f, "");
      RTX_OPTION("rtx.playerModel", float, eyeHeight, 64.f, "");
      RTX_OPTION("rtx.playerModel", float, intersectionCapsuleRadius, 24.f, "");
      RTX_OPTION("rtx.playerModel", float, intersectionCapsuleHeight, 68.f, "");
    } playerModel;

    struct Displacement {
      friend class ImGUI;
      RTX_OPTION("rtx.displacement", DisplacementMode, mode, DisplacementMode::QuadtreePOM, "What algorithm the displacement uses.\n"
        "RaymarchPOM: advances the ray in linear steps until the ray is below the heightfield.\n"
        "QuadtreePOM: Relies on special mipmaps with maximum values instead of average values.  Uses the mipmap as a quadtree.");
      RTX_OPTION("rtx.displacement", bool, enableDirectLighting, true, "Whether direct lighting accounts for displacement mapping");
      RTX_OPTION("rtx.displacement", bool, enableIndirectLighting, true, "Whether indirect lighting accounts for displacement mapping");
      RTX_OPTION("rtx.displacement", bool, enableNEECache, true, "Whether the NEE cache accounts for displacement mapping");
      RTX_OPTION("rtx.displacement", bool, enableReSTIRGI, true, "Whether ReSTIR GI accounts for displacement mapping");
      RTX_OPTION("rtx.displacement", bool, enableIndirectHit, false, "Whether indirect ray hits account for displacement mapping (Enabling this is expensive.  Without it, non-perfect reflections of displaced objects will not show displacement.)");
      RTX_OPTION("rtx.displacement", bool, enablePSR, false, "Enable PSR (perfect reflections) for materials with displacement.  Rays that have been perfectly reflected off a POM surface will not collide correctly with other parts of that same surface.");
      RTX_OPTION("rtx.displacement", float, displacementFactor, 1.0f, "Scaling factor for all displacement maps");
      RTX_OPTION("rtx.displacement", float, displacementInFactor, 1.0f, "Scale factor for inwards displacement");
      RTX_OPTION("rtx.displacement", float, displacementOutFactor, 1.0f, "Scale factor for outwards displacement");
      RTX_OPTION("rtx.displacement", uint, maxIterations, 64, "The max number of times the POM raymarch will iterate.");
    } displacement;

    RTX_OPTION("rtx", bool, resolvePreCombinedMatrices, true, "");
    // NV-DXVK [PreCombinedGuard]: resolvePreCombinedMatrices fires on
    // isIdentityExact(worldToView) alone, reading that as "the game fused world
    // and view into objectToView, so recover world by multiplying by
    // viewToWorld". That premise only holds when objectToView actually CARRIES
    // the fused transform. When the D3D11 layer extracted nothing for a shader
    // permutation, worldToView AND objectToView both arrive exactly identity —
    // and the resolve then evaluates to viewToWorld alone, whose translation is
    // the camera position. The geometry gets planted on the camera and follows
    // it. Measured in TF2 on one 811-vtx mesh submitted under several VS
    // permutations in the same frame:
    //   vs=0x29cb608ecc5f69fa w2vT=(-10879.2,-7487.03,1856.92) -> o2wT=(0,0,0)   correct
    //   vs=0x28d6a40d7ef896e0 w2vT=(0,0,0) o2vRot0=(1,0,0)     -> o2wT=camPos    welded
    //   vs=0x28d611d92a2401ef w2vT=(0,0,0) o2vRot0=(1,0,0)     -> o2wT=camPos    welded
    // With this on, a draw whose objectToView is also exactly identity is left
    // alone: there is no fused transform to un-fuse, so fabricating one from the
    // camera is strictly worse than leaving objectToWorld as it stands.
    //
    // CAVEAT: geometry whose vertices genuinely live in VIEW space (a classic
    // viewmodel) legitimately arrives with both matrices identity, and for it
    // o2w = viewToWorld is correct. Such draws should be reaching the ViewModel
    // camera path rather than this one, but set this False to restore the old
    // behaviour if a view-space mesh regresses.
    // NV-DXVK [CamReconColorRt]: LAYER-1 fix. The path-3 camera reconstruction
    // in d3d11_rtx.cpp (projSlot==UINT32_MAX && m_hasEverFoundProj) was gated on
    // the draw having a R32G32_UINT position layout, because "fmt=106 draws that
    // fail projection detection are shadow/depth passes with light-space
    // transforms -> applying the main camera VP to them produces extreme BLAS ->
    // GPU TDR". That premise over-collects: float3-position SKINNED WORLD
    // geometry also fails projection detection and is not a shadow pass, so it
    // was excluded and left with NO transform at all — worldToView identity,
    // objectToView identity, and then the pre-combined resolver plants it on the
    // camera. Measured on one 811-vtx mesh, same frame, same albedo:
    //   vs=0x29cb608ecc5f69fa posFmt=101 (uint pos) -> wtvPath=3, placed
    //   vs=0x28d6a40d7ef896e0 posFmt=106 (float3)   -> wtvPath=0, camera-welded
    //   vs=0x28d611d92a2401ef posFmt=106 (float3)   -> wtvPath=0, camera-welded
    // ([NoProj] scanRan=1 on all three: a full scan ran and genuinely found no
    // projection, so this is not the neg-cache.)
    //
    // The distinguishing property of the draws the original guard meant to
    // exclude is that a depth-only shadow pass binds NO colour render target.
    // That is a structural test on what the draw actually binds, so it admits
    // colour-writing world geometry regardless of position format while still
    // rejecting light-space depth passes. Set False to restore the strict
    // uint-position-only gate if a shadow pass slips through and TDRs.
    // NV-DXVK [SkinDecomposeRevive]: LAYER-2 fix, two halves that MUST ship
    // together — either alone makes things worse.
    //
    // (a) finalizeSkinningData's decompose sits behind `pLastCamera != nullptr`,
    //     but pLastCamera is null on 100% of skinned draws ([SkinPlace]
    //     camSet=0, 2000/2000). lastCamera comes from getLastSetCameraType(),
    //     which onFrameEnd resets to Unknown (rtx_camera_manager.cpp:269) and
    //     only processCameraData reassigns (:1053) — and processCameraData runs
    //     AFTER finalizePendingFutures (rtx_context.cpp:5227 vs :5223; see the
    //     TODO at :5182). So the whole decompose is dead code. With
    //     tf2SkinnedUseDrawCamera on, the math reads ONLY the draw's own
    //     matrices; pLastCamera is touched solely by the fallback, so the gate
    //     blocks work that does not need it.
    //
    // (b) Under fusedMode == None the block first does
    //     `objectToView = worldToView`, discarding the objectToView that
    //     d3d11_rtx.cpp:13911-13913 correctly computed as worldToView *
    //     objectToWorld. The decompose then evaluates inverse(w2v) * w2v =
    //     IDENTITY for every skinned mesh — so simply un-gating (a) would drop
    //     all skinned geometry onto the world origin. Left alone, objectToView
    //     is already correct and the decompose recovers
    //     inverse(w2v) * (w2v * o2w) = o2w, i.e. it preserves placement instead
    //     of destroying it. Only a genuinely fused mode should rewrite it.
    //
    // Together the decompose becomes placement-preserving by construction,
    // which is what makes reviving it safe. Set False to restore the dead gate
    // and the overwrite exactly as they were.
    RTX_OPTION("rtx", bool, skinDecomposeRevive, true,
               "Run the skinned-mesh objectToWorld decompose using the draw's own camera even when no "
               "last-set camera is available, and stop overwriting objectToView with worldToView when "
               "matrices are not fused. Both halves apply together: the decompose then preserves world "
               "placement instead of collapsing every skinned mesh to the origin.");
    RTX_OPTION("rtx", bool, camReconAllowColorRtDraws, true,
               "Allow the fallback camera reconstruction to run for draws that bind a colour render target, "
               "not only for draws with a R32G32_UINT position layout. Fixes float3-position world geometry "
               "being left with no worldToView at all (which ends up welded to the camera). "
               "Set False to restore the uint-position-only gate.");
    RTX_OPTION("rtx", bool, resolvePreCombinedRequiresFusedTransform, true,
               "When resolving pre-combined world/view matrices, require objectToView to be non-identity. "
               "Blocks the resolve from fabricating an objectToWorld out of the camera transform for draws "
               "where no transform was extracted at all (both matrices identity), which welds that geometry "
               "to the camera. Set False to restore the unguarded behaviour.");

    RTX_OPTION("rtx", uint32_t, minPrimsInDynamicBLAS, 1000, "The minimum number of triangles required to promote a mesh to it's own BLAS, otherwise it lands in the merged BLAS with multiple other meshes.");
    RTX_OPTION("rtx", uint32_t, maxPrimsInMergedBLAS, 50000, "The maximum number of triangles for a mesh that can be in the merged BLAS.  ");
    RTX_OPTION_FLAG("rtx", bool, forceMergeAllMeshes, false, RtxOptionFlags::NoSave, "Force merges all meshes into as few BLAS as possible.  This is generally not desirable for performance, but can be a useful debugging tool.");
    RTX_OPTION_FLAG("rtx", bool, minimizeBlasMerging, false, RtxOptionFlags::NoSave, "Minimize BLAS merging to the minimum possible, this option tries to give all meshes their own BLAS.  This is generally not desirable forperformance, but can be a useful debugging tool.");
    RTX_OPTION("rtx", bool, persistStaticBlas, false, "EXPERIMENTAL. Route STATIC geometry to its own persistent dynamic BLAS instead of the per-frame-rebuilt merged BLAS. A mesh qualifies when it is non-skinned, not point-instancer geometry, its primitive count is >= rtx.minPrimsInDynamicBLAS, and its geometry did NOT change this frame (processGeometryInfo returned kUpdateInstance, so BlasEntry::frameLastUpdated lags the current frame). The existing dynamic-BLAS path then reuses the acceleration structure across frames (built once, then neither rebuilt nor refit while the geometry stays static) — eliminating the full merged-BLAS rebuild that dominates frames with large volumes of static world geometry. Prim-count floor keeps tiny meshes merged (avoids TLAS bloat from many small BLAS). Set false for the stock prim-count-only routing.");

    RTX_OPTION("rtx", bool, pendingSrcBakeSingleRetry, true, "Terminate the pendingSrcBake (s2s mangle/black FIX B) geometry re-cache after a single recovery bake instead of re-arming it every frame.\n"
               "FIX B force-re-caches a geometry whose BLAS inputs were baked while an engine upload to the source buffer was still in flight, and was written to stop once 'a bake lands with the source ready'. That readiness test is GeometryBuffer::isPendingGpuWrite(), i.e. DxvkResource::isInUse(Write) - a whole-buffer refcount raised as soon as any command touching the buffer is recorded and cleared only when that command list retires on the GPU. For a buffer the game re-uploads every frame it is true essentially whenever it is sampled, so the exit condition never fired and every affected geometry re-interleaved its full vertex buffer every frame (measured: 140 of 352 entries, 9.2M of 9.75M scene vertices per frame).\n"
               "With this enabled the recovery bake clears the flag instead of re-latching it: one full frame plus the per-frame fence wait has elapsed since the raced upload, so that upload has completed, and any genuinely newer content is caught by the ordinary vertex-position hash comparison.\n"
               "Set false to restore the original re-arming behaviour. If s2s geometry renders black or mangled, that is the regression this switch controls.");

    RTX_OPTION("rtx", bool, captureSourceGeometry, true, "Fix for single-frame flicker / long-term absence of world props whose engine-side vertex+index buffers are DEVICE_LOCAL and rewritten by the game (the TF2 prop-batch flicker).\n"
               "The BLAS bake is recorded at end-of-frame position in the command stream, so nothing orders it against the engine's buffer uploads; a bake that loses the race reads mid-upload bytes (zeroed indices collapse the mesh invisibly, garbage positions explode it). With this enabled, draws predicted to (re)bake get a GPU->GPU copy of their exact vertex/index windows recorded AT THE DRAW'S OWN POSITION in the command stream - the one spot guaranteed to be after this frame's upload and before the next - and the bake consumes the copy instead of the live buffer.\n"
               "Set false to fall back to baking from live source buffers (pre-fix behavior).");
    RTX_OPTION("rtx", bool, captureSourceHotVeto, true,
               "MEASUREMENT SWITCH, default = existing behaviour. Controls the `srcHot` term in the geometry-capture decision "
               "(d3d11_rtx.cpp: doCapture = srcHot || !provenStable || feedbackWantsCapture).\n"
               "srcHot is DxvkResource::isInUse(Write) on the source buffer — a WHOLE-BUFFER refcount, set the moment any command "
               "touching that buffer is recorded. TF2 packs many meshes into shared buffers it rewrites every frame, so srcHot reads "
               "true for a draw whose own bytes nobody touched, and short-circuits the provenStable veto. Measured over five 5-second "
               "windows: 6655-8932 draws captured per window (~107/frame at ~15 fps, ~4-5 GB per window, ~1 GB/s of copyBuffer) while "
               "186-231 keys were proven stable and fbDraws — the captures the feedback ring actually asked for — was under 10% of the total.\n"
               "Set false to drop the srcHot term and let provenStable/feedbackWantsCapture decide alone.\n"
               "RESULT — MEASURED 2026-08-04, A/B run, same scene: setting this false cut capture volume from 50-69 MB/frame and 94-107 "
               "draws/frame down to 10-26 MB/frame and 59-79 draws/frame (3-5x), and bought NOTHING. Frame time was 71.5/73.7 ms before "
               "and 70.6-77.5 ms after; tail_emit — the SubmitDraw stage that actually contains this code (29379->31845; tail_capture is "
               "the sky/sun/light stage at 29073->29379 and does NOT contain it) — went 1.70/1.71 ms/frame before to 1.74-2.06 ms/frame "
               "after, i.e. marginally up.\n"
               "WHY: copyBuffer is a RECORDING call. Its CPU cost is command-list writing and is near-independent of size; the bytes land "
               "on the GPU, which absorbed them (GPUIDLE ~0-1.4 ms, busyPct 87-92% on the CPU side). So srcHot costs copy BANDWIDTH, not "
               "frame time, and disabling it only surrenders the correctness margin it exists for: without it, draws whose source really "
               "is mid-rewrite stop being captured and the s2s mangle/black corruption rtx.captureSourceGeometry was written to fix can "
               "reappear.\n"
               "Do NOT flip the default to false on the reasoning that >90% of captures are this veto over-firing at whole-buffer "
               "granularity. That observation is TRUE and was the motivation for this switch — it is simply not worth any frame time. "
               "Kept as a measurement switch; if you are hunting per-draw CPU cost, the leaf is elsewhere (tail_capture ~21 ms/frame, "
               "bt_cullVtx ~7.8 ms/frame).");
    RTX_OPTION("rtx", uint32_t, captureSourceGeometryWarmFrames, 3, "How many consecutive frames a newly-seen draw keeps being captured by rtx.captureSourceGeometry.\n"
               "Engine re-batches create a new BlasEntry on TWO consecutive frames (paired dedup-misses observed via [MtnDedup]), so the window must cover at least the first 2 sightings; 3 adds one frame of slack. Larger values waste copy bandwidth on draws that will not re-bake.");

    RTX_OPTION("rtx", bool, logFanoutSplitStats, false,
               "[FanoutSplit] per-placement health probes in InstanceManager::processSceneObjectFanout. DEFAULT OFF -- this is a frame-time switch, not a logging one.\n"
               "WHAT IT COSTS. The probes run per PLACEMENT, not per draw: ~14,600/frame, measured as 15,489 processSceneObjectImpl calls against ~1,060 draws. Each one paid two XXH64s (64 B over the composed matrix, 36 B over its 3x3 basis) plus FOUR unordered_set operations -- two membership lookups and two inserts, whose nodes are freed again at the next frame's rollover. That is ~29,000 hash-set operations per frame.\n"
               "WHY IT WAS FOUND. [Perf.SceneObj]'s four buckets summed ~16.3 ms against [ProcDCS] instMs=22: the guard brackets processSceneObjectImpl and never enters the fanout wrapper, so this work was outside every probe. Same shape as the [PassCensus] and [InstCounts] blocks deleted from processSceneObjectImpl on the same day -- ungated per-instance bookkeeping feeding a tag on the logDenyTags list, formatted and thrown away.\n"
               "WHAT YOU LOSE WITH IT OFF. mirrored / mtxStable / basisStable / newProp / missProp are not accumulated and the [FanoutSplit] line is not emitted (emitting it would print zeroes and read as a collapse). Nothing the renderer does depends on any of them. Collision DETECTION is unaffected -- it is the out_instances scan, not these counters -- and the collision DUMP still prints mtxHash/firstMtxHash, recomputed from sPlacedO2w when it fires.\n"
               "TURN ON to re-run the dedup-key bakeoff (mtxStable vs basisStable vs hybStable) or to diagnose fanout churn, and remember to take [FanoutSplit] off rtx.logDenyTags or the line will be denied anyway.");

    RTX_OPTION("rtx", bool, memoExtractTransforms, false,
               "[Perf.MemoXf] THE OPTIMISATION PLAN Phase 3.2b -- memoise ExtractTransforms WITHIN a frame, keyed on the captured inputs. DEFAULT OFF. Requires rtx.useDrawSnapshot.\n"
               "WHY WITHIN A FRAME AND NOT ACROSS FRAMES. [DrawRedund] 2026-08-12 measured perFrameDistinct{cb2=10 cb3=215 wholeDraw=254} against drawsPerFrame=1367: ONE frame contains 254 distinct cbuffer-content sets and we run the derivation 1367 times. That is 5.4x redundancy inside a single frame, and unlike every cross-frame number in this plan it does NOT assume a static scene, so it survives combat -- which is exactly where REDUNDANT=97% collapses and where frame rate matters most.\n"
               "WHY THE KEY IS THE CONTENT AND NOT THE ROUTE. This is the deleted replay tier's idea with the defect removed. That tier gated on rr.ineligBits -- seven ROUTE rules ANDed together (fallback/UI, bone, votes, instanced, projSlot, viewOk, o2wPath) -- and admitted 0.06% of draws, which is why it logged 0 hits in 6.5M draws. It never asked whether the inputs were duplicates. The gate here is safeToDefer(), the three-axis DrawSnapshot purity certificate that did not exist when the tier was built: measured xfEligN=33257 vs xfIneligN=13811, i.e. 70.7% of draws, and they carry 68% of bt_extractXf at 14.3us/draw against the ineligible 16.2us -- NOT the cheap tail. Do not reuse ineligBits.\n"
               "WHAT THE KEY COVERS. The whole record: shader/layout/state identity, all bindings, viewports, cbuffer binding identity, the captured cbuffer BYTES, the t31/COLOR1 entry, the five carrier-group fingerprints (so a draw that reads a carrier another draw moved cannot hit a stale entry), and m_currentInstanceIndex. Both cheap shortcuts are measured DEAD and must not be substituted: contentGen bumps on every Map(WRITE_DISCARD) whether or not bytes changed (cheap{ok=0}), and binding identity predicts the bytes only 25% of the time (ptr{ok=6604 WRONG=20288}). The bytes must be compared.\n"
               "WHAT IT REFUSES, AND WHY THAT IS NOT OPTIONAL. DrawCallTransforms holds instancesToObject / prevInstancesToObject raw pointers plus their shared_ptr owners. Memoising a draw that carries any of them would serve one draw's per-instance array to another -- the aliasing class this project has already paid for twice (dropship COLOR1.y, getImageHash). Draws carrying any instance pointer, or isFanoutBatch, are never stored, which also mirrors filterDupSameDraws' fanout exclusion.\n"
               "STORE REQUIRES safeToDefer(), which is known only AFTER the derivation ran -- so a draw is stored on the way out, never on the way in. A hit therefore inherits a proven-pure producer, and because the key covers the carrier fingerprints the consumer would have executed identically.\n"
               "VERIFY BEFORE BELIEVING: rtx.memoExtractVerify runs the derivation anyway on every hit and bit-compares. FAIL must be 0. That is the same discipline the replay tier was held to and it is the only thing standing between this and a silent wrong-transform bug.\n"
               "TURN OFF before quoting a frame-time number taken with the verify mode on -- it derives twice by construction.");

    RTX_OPTION("rtx", bool, memoExtractVerify, false,
               "[Perf.MemoXf] Correctness mode for rtx.memoExtractTransforms. On every memo HIT, run ExtractTransforms anyway and bit-compare the result against the memoised one; report FAIL. DEFAULT OFF.\n"
               "FAIL>0 MEANS THE KEY IS INCOMPLETE -- the derivation read something the key does not cover -- and the feature is unsafe until the missing input is found and hashed. It does NOT mean 'mostly fine'. A wrong transform is a wrong object position, and the failure mode is silent.\n"
               "COMPARES THE POD FIELDS ONLY (the five matrices, clip plane, texgen/texcoord modes, viewport, VS/PS hashes, path ids, isSubView/isSubViewSkybox, stablePropId). The pointer fields are not compared because a memoised entry is never allowed to carry them -- see the refusal rule on the parent option.\n"
               "IT ALSO REPORTS FAILcarrier, AND FAIL=0 ALONE IS NOT A LICENCE TO TURN THIS OFF. Verify-on and verify-off are not the same program. bt_extractXf is the ONLY stage that moves cross-draw D3D11Rtx state and it does so on ~14% of draws (see DrawSnapshot's carrier note). Under verify the derivation still RUNS on every hit, so those writes still happen; with verify off a hit skips the derivation and any carrier write it would have made never happens, and a LATER draw then derives from cross-draw state that was never moved. The field comparison cannot see that -- this draw's transforms are correct and the damage lands elsewhere. The hit path already had to republish s_xtEligLast by hand for exactly this reason; that was the instance of it somebody noticed.\n"
               "FAILcarrier IS THE GENERAL CHECK: a stored entry's producer passed safeToDefer(), which requires !wroteCarrier(), so it moved nothing; if the key is complete, a draw hitting that entry ran on the same inputs and must move nothing either. A hit that re-derives and DOES move a carrier is proof the key is incomplete in the one dimension that goes silent the moment verify is off. It can only be measured while this flag is ON, because the derivation it checks is the one being skipped.\n"
               "COSTS A FULL SECOND DERIVATION ON EVERY HIT, so frame time under this flag is meaningless by construction. Run it to get FAIL=0 AND FAILcarrier=0 over a real session, then turn it off and measure.");

    RTX_OPTION("rtx", bool, memoExtractAblate, false,
               "[Perf.MemoXf.Ablate] Component ablation for rtx.memoExtractTransforms' key. DEFAULT OFF. Requires rtx.memoExtractTransforms, and should be run with rtx.memoExtractVerify=True.\n"
               "WHAT IT ANSWERS, AND WHY IT EXISTS. Four builds narrowed drawMemoKey by argument -- vertex/index bindings, then cbLoc/bone/route carriers, then constantOffset -- and hit went 1% -> 4% -> 5% -> 5%. Each cycle cost a build, a run and a reading, and the third and fourth bought nothing. The handoff's own MISTAKES list names the cause twice: guessing instead of measuring, and reading a FOLD as if it were a component. This probe replaces the whole loop with one run: it splits the key into its 16 components, rebuilds ~20 candidate keys per draw with one component (or set) held out, and reports for each what the hit rate WOULD have been and whether holding that component out is CORRECT.\n"
               "THE THREE NUMBERS. distinct/frame is the candidate's key count, and it is the one to read against [DrawRedund] perFrameDistinct{wholeDraw}, the cbuffer-content ceiling -- a candidate at the ceiling is a candidate that has stopped keying on object identity. hitSafe% is the hit rate that candidate would actually deliver, modelling the real store gate (an entry becomes servable only once a safeToDefer() producer has stored it), and hitAll% is the same with the gate removed, so the DIFFERENCE between them is exactly what relaxing the carrier axis is worth -- measured, not argued.\n"
               "COLLIDE IS THE CORRECTNESS TEST AND IT IS THE POINT. Two draws whose candidate key matches but whose DERIVED TRANSFORMS differ would have been served a wrong transform by that candidate. The comparison is byte-for-byte the field list rtx.memoExtractVerify's FAIL uses, so COLLIDE and FAIL mean the same thing -- COLLIDE just measures it for keys that were never built, on data, before a line of the real key is touched. A candidate with COLLIDE=0 over a real session is safe to adopt and its hitSafe% is what you will get; a candidate with COLLIDE>0 is disqualified outright and no amount of hit rate redeems it.\n"
               "COLLIDE IS ONLY INFORMATIVE WHEN dist/f DROPPED -- corrected 2026-08-12 after the first run, and the correction matters more than the original rule. A candidate that did not reduce the key count merged no draws, so it never had the opportunity to collide, and its COLLIDE=0 says nothing whatever about whether it is safe. Measured: the full key sits at ~1010 distinct against ~1091 draws -- nearly unique per draw -- so removing any single component merges almost nothing and reads COLLIDE=0 trivially. Read every row as a PAIR: did dist/f fall, and if it did, was COLLIDE still 0.\n"
               "THE ! ROWS WERE DESIGNED AS POSITIVE CONTROLS AND THREE OF THEM FAILED AS SUCH. -vsIl, -ps and -vp hold out components the derivation demonstrably consumes, so they were predicted to collide; they read 0 because they merge nothing, per the paragraph above. That is a defect in the control, not in the probe -- the machinery demonstrably fires wherever merges actually happen (-bytes!=945, -geoAll=1625, BYTES=14144). Keep them as a floor check on the COLLIDE path, not as a veto on the run.\n"
               "COLLIDE COMPARES OUTPUTS AND THEREFORE CANNOT SEE A SIDE EFFECT. If the derivation WRITES a piece of cross-draw state rather than merely reading it, two draws can produce identical transforms while differing in whether that write happened, and this probe will call the narrowing safe. That is not hypothetical: kSdepCbLoc is written by the derivation, the key had stopped hashing it, COLLIDE and FAIL both read clean, and rtx.memoExtractVerify's FAILcarrier caught it at ~10% of hits with mask=4 on 100% of failures. Use FAILcarrier, not this probe, to decide anything about a component the derivation writes.\n"
               "THE OUT ROW IS THE REAL CEILING. It keys on the derived transforms themselves, so its distinct/frame is the number of genuinely different transforms in a frame -- the floor no key can go below and the number that settles whether the 81% figure (from wholeDraw=254 of 1367) or the 40-50% downgrade (from geo=620) was right. Nothing else in this tree measures it.\n"
               "COSTS: ~16 component hashes plus ~20 small folds and ~20 hash-map probes per draw, and it holds ~20 maps of a frame's keys. That is a diagnostic budget, not a shipping one. It also SKIPS any draw whose transforms were served from the memo rather than derived (reported as served=), because an ablation of a value the derivation did not produce measures the memo, not the derivation -- which is why verify should be on.");

    // DEFAULT FLIPPED false -> true on 2026-08-13. Measured FAIL=0 across every
    // window at ~51% serve with the mask defaults below, camera moving. Safe to
    // enable only BECAUSE those defaults changed in the same commit -- at the
    // old ObjKeyMask=2047 this flag serves wrong transforms. Keep the two
    // together; do not backport this flag alone.
    RTX_OPTION("rtx", bool, splitTransformCache, true,
               "[Perf.SplitXf] THE OPTIMISATION PLAN Phase 3 proper -- cache the OBJECT half of the derivation across frames and the VIEW half per frame, then compose. DEFAULT OFF. Requires rtx.useDrawSnapshot.\n"
               "WHY THE FUSED MEMO CANNOT GET THERE, MEASURED 2026-08-12. [Perf.MemoXf.Ablate] OUT dist/f=378 against 1091 draws/frame: there are only 378 genuinely distinct DrawCallTransforms in a frame, so a key over the whole struct caps at 65% hit no matter how it is narrowed (the full key sits at 1010 distinct, and stripping every removable identity field reaches 907 -- 5% -> 13%). The remaining fragmentation is inside the cbuffer bytes and the t31/COLOR1 entry, which the derivation genuinely reads. No key change reaches it.\n"
               "WHY THE SPLIT DOES. The output is ALREADY factored -- d3d11_rtx.cpp:21532 computes objectToView = worldToView * objectToWorld -- and the two factors have completely different lifetimes. [DrawRedund] perFrameDistinct{cb2=10} says the view/projection takes TEN distinct values across 1091 draws, because a frame has ~10 views (main, 3D-skybox sub-view, shadow/cube passes, UI), not 1091. [Perf.FastInst] xformMiss=250 of calls=15476 says the OBJECT transform moves on 1.6% of instances per frame. Fused, the camera moving invalidates everything every frame (bytes=19% cross-frame identical); split, the camera invalidates only the ten view entries.\n"
               "THIS IS WHAT UE DOES. FMeshDrawCommand is cached at AddToScene and the view state is bound per view at render time, which is why UE's per-draw derivation cost is zero for a static scene. The piece everyone assumes is the hard part -- persistent identity -- this tree already has: [FindStage] exact=15148/15476 (97.9%), created=0, and stablePropId is already a field of DrawCallTransforms.\n"
               "THE CAMERA-RELATIVE PATHS ARE THE CATCH AND THEY ARE ALREADY IDENTIFIED. Some objectToWorld sites are derived FROM the view (path 4 = invView*cb3Mat, paths 5/6 rdef with a live camera origin, 8 = cb2@4 camera fallback, 11 = cached VP + live cam, 13 = camera-relative). Those object parts are NOT view-independent and must never enter a cross-frame cache. The deleted replay tier had already worked out this exact set (d3d11_rtx.cpp:15843). rtx.splitTransformO2wPathMask is the allowlist, defaulted to the paths that are camera-free by construction and widenable at runtime with verify on.\n"
               "STORE REQUIRES safeToDefer() AND a camera-free o2w path AND no instance pointers, and is decided on the way OUT, exactly as the memo does it. Serving requires BOTH halves present -- a view entry from THIS frame and an object entry from any frame.\n"
               "VERIFY BEFORE BELIEVING: rtx.splitTransformVerify derives anyway on every serve and bit-compares, same field list and same discipline as rtx.memoExtractVerify. It reports FAIL and FAILcarrier. Both must be 0.");

    RTX_OPTION("rtx", bool, splitTransformVerify, false,
               "[Perf.SplitXf] Correctness mode for rtx.splitTransformCache. On every SERVE, run ExtractTransforms anyway and bit-compare the composed result against the derived one; report FAIL. DEFAULT OFF.\n"
               "FAIL>0 MEANS THE SPLIT IS WRONG for that draw -- either a key is missing an input, or the object half is not actually view-independent on that path (in which case take its bit OUT of rtx.splitTransformO2wPathMask; do not weaken the comparison).\n"
               "FAILcarrier IS THE SECOND GATE and it is the one that caught the fused memo. bt_extractXf is the only stage that moves cross-draw state (~14% of draws). Under verify the derivation still runs on a serve so those writes still happen; with verify off a serve skips them and a LATER draw derives from state that was never moved. The transform comparison cannot see that. Both counters must be 0 before this ships.\n"
               "COSTS A FULL SECOND DERIVATION ON EVERY SERVE, so frame time under this flag is meaningless by construction.");

    // DEFAULT RAISED 0x60F -> 0x7FFFFFFE on 2026-08-13, matching the measured
    // session: every o2w path except p0 is cacheable. 0x60F admitted only a
    // subset, so a build shipping the new key defaults with the old path mask
    // would silently cache far fewer paths than the ~51% figure was taken on.
    RTX_OPTION("rtx", uint32_t, splitTransformO2wPathMask, 0x7FFFFFFEu,
               "[Perf.SplitXf] Bitmask of m_lastO2wPathId values whose objectToWorld may enter the CROSS-FRAME object cache. Bit N = path N is camera-free. DEFAULT 0x60F = paths 0,1,2,3,9,10.\n"
               "THE PATHS (d3d11_rtx.h ~:1450): 0 identity, 1 non-instanced BSP t31, 2 t30 CPU bone, 3 t30 bone-slice, 4 CB3->O2W (invView*cb3Mat), 5 RDEF, 6 trySourceFloat3x4, 7 tryWorldCb generic scan, 8 cb2@4 cameraOrigin fallback, 9 per-instance fanout override, 10 bone-instanced identity, 11 cached VP + live cam, 13 camera-relative.\n"
               "WHY THE DEFAULT IS NARROW. 4, 5, 6, 8, 11 and 13 all derive objectToWorld FROM the camera in some form, so their object half is not view-independent and caching it across a camera move is a wrong-position bug -- silent, and exactly the class this project has paid for before. 7 is a generic matrix scan whose source is not established, so it is excluded until it is.\n"
               "WIDEN IT WITH VERIFY ON, ONE BIT AT A TIME. If FAIL stays 0 for a bit over a real session including camera movement, that path is genuinely camera-free and the bit stays. If FAIL goes non-zero, the bit comes back out -- that is the measurement, and it is cheaper than the argument.");

    // DEFAULT RAISED 2047 -> 5705727 on 2026-08-13, and two of the added bits
    // are CORRECTNESS, not tuning. At 2047 the cache serves wrong transforms:
    //   bit 20 (1048576) routes o2w path 3's bone-0 read through the snapshot.
    //                    Without it the key hashes bytes captured at snapshot
    //                    time while the deferred derivation reads them live,
    //                    ~850 bone-cache merges later. Measured 133 FAIL/run.
    //   bit 22 (4194304) puts the per-VS neg-proj cache state in the object
    //                    key. Without it one shader's o2w path depends on how
    //                    many frames ago a TTL was stamped, and nothing in the
    //                    snapshot separates the two cases. Measured 12 FAIL/run.
    // Also on: 11 (drop uncaptured t31Entry), 16/17 (shipping camCorr + path
    // gate), 18 (bone0 bytes -- load-bearing, A/B off = 168 FAIL).
    // Deliberately NOT on: 19 (path-private drops) and 21 (rtv presence), both
    // measured redundant once bit 22 landed. See HANDOFF_SPLITXF_OBJKEY.md.
    //
    // RAISED 5705727 -> 14094335 on 2026-08-13, adding bit 23 (8388608), the
    // draw range. This one is not tuning either: without it the identity key
    // covers the IA BINDING but not the draw call's own
    // Start/BaseVertexLocation, so two different meshes sub-allocated out of
    // one pooled buffer -- which is how Source draws most of the world -- share
    // an object key and evict each other every frame. That collision is what
    // an N-way object cache was built to absorb (serve 68->77%, objStale
    // 3979->2350, at obj x ways x ~200 B of inline way array); splitting the
    // key removes it instead of holding N entries under one.
    RTX_OPTION("rtx", uint32_t, splitTransformObjKeyMask, 14094335u,
               "[Perf.SplitXf] Which components go into the CROSS-FRAME object key. Bitmask, default 2047 = all 11 bits. THE POINT IS TO SWEEP THIS FROM rtx.conf WITHOUT REBUILDING.\n"
               "BITS: 1 vsIl, 2 ps, 4 state(blend/depth/raster), 8 topology, 16 numCbRanges, 32 layoutResolved, 64 geoContent(t31Entry+instSem), 128 geoSel(slots/indices), 256 instanceIndex, 512 cbLoc carrier, 1024 objectSpans(cb3 bytes).\n"
               "BIT 23 (8388608) IS THE DRAW RANGE -- IndexCount/StartIndexLocation/BaseVertexLocation plus the indexed flag, folded into the IDENTITY key beside the IA binding. It is the other half of an address the key already half-covered: vbOffset/vbStride/ibOffset are in there because meshes are sub-allocated out of pooled buffers, and Start/Base is the other way D3D11 addresses that same sub-allocation. Clearing it merges distinct meshes onto one entry.\n"
               "READ IT ON: serve% and objStale should improve and instanceable on [Perf.SplitXf.Merge] should FALL -- that line's key is built from this one, so the 433/4-frame reading that ruled out instancing counted different sub-meshes as instances of one another. ident{new/f} is the tripwire in the other direction: a wider key splits entries, so if new/f climbs materially the range is churning (dynamic geometry) rather than discriminating, and that is the reading that would take the bit back out.\n"
               "WHY IT EXISTS. The object cache is measured CORRECT and COLD: whole windows of FAIL with o2w absent, while newObj/frame runs 381-646 against the ~4-13 objects [Perf.FastInst] says actually move. The churn ablation cleared instIdx (549 vs 562 baseline) and cbLoc (541 vs 562) and convicted two components of comparable weight -- geoContent and objectSpans, each roughly halving the count, with geoContent collapsing 85 -> 12 in a quiet window. 12/frame is exactly the predicted range, so the ceiling is real and the shipping key is what fails to reach it.\n"
               "HOW TO USE IT. Clear one bit, run with rtx.splitTransformVerify=True, read newObj/frame and serve%. FAIL MUST STAY 0 -- a component removed from this key is a component the derivation may still read, and the only thing standing between a narrower key and a silently wrong transform is that comparison. A bit that raises serve% with FAIL=0 over a session INCLUDING CAMERA MOVEMENT is a bit that was never an input. A bit that raises serve% and FAIL together is a bit you needed.\n"
               "DO NOT CLEAR 1024 AND 64 TOGETHER AS A FIRST MOVE. They are the two convicted churners and dropping both at once cannot tell you which one mattered -- that is the fold mistake this feature has already paid for three times.\n"
               "THIS OPTION WAS DEAD UNTIL 2026-08-12. It shipped with the bit list and the paragraph above and NO CODE READ IT, so every sweep of it from rtx.conf changed nothing and any conclusion drawn from one is void. Wired up in drawSplitKeys now. Same defect, found the same way, as rtx.splitTransformObjCacheMax -- when an option in this feature does not behave, check that something reads it before theorising.\n"
               "IT MASKS THE SHIPPING KEY ONLY. The ablation variants are built from the UNMASKED head deliberately, because they are the baseline this is measured against.");

    // NEW 2026-08-13, DEFAULT ON. The evidence is [Perf.SplitXf.CbWaste], which
    // was built specifically to decide this and came back at 49% of captured
    // bytes dead and 37% of write-combined transactions dead.
    RTX_OPTION("rtx", bool, splitTransformManifestPathFilter, true,
               "[ManifestPath] Capture a VS's LEARNED cbuffer spans only for the o2w path this draw is predicted to take, instead of every path that shader has ever taken. DEFAULT ON.\n"
               "WHAT IT FIXES. The per-VS span manifest replays every span the shader's derivation was ever observed to read. Whether a span is read is decided by the PATH, and 55% of draws sit on a VS that takes more than one -- so p13's cb3 span was being copied onto p1's draws, every draw, all frame. [Perf.SplitXf.CbWaste] priced it: 49% of ALL captured bytes are never read by any drawCbSpan call; p1 has cb3 captured on 100% of its draws and read on ZERO of them (dead{3=8184/8063}); p3 the same; 37% of every write-combined transaction the capture issues, and 65% of the cb3 transactions, are for spans nothing reads. The capture is ~95% of pfs_guard, the largest item on the frame thread.\n"
               "WHY IT IS SAFE, AND WHY IT IS NOT THE DELETED DROP MASKS. The prediction is the same (vs, il) -> last-derived-path cache [DropByPath] already ships, and it is NOT exact: o2wPath{agree=17491-22699 disagree=58-73}, ~0.35% wrong. Being wrong is FREE here. A withheld span makes find() return nullptr, the consumer takes its original live path (correct bytes, not pure), and that live read increments cbLiveReads -> clears contentPure() -> clears cbSafeToDefer() -> REFUSES THE STORE. So a mispredicted draw cannot create an entry whose validity hash is missing the span it actually read. DropExt narrowed the hash and stored anyway, which is why FAILcarrier grp{cbLoc} went 0 -> 337; the store gate closes that here with no new mechanism. The live read also teaches the manifest that path's bit, so it self-heals on the next draw.\n"
               "IT COUNTS, IT DOES NOT LATCH -- AND THE FIRST VERSION LATCHED. v1 used a boolean 'has this path ever read this span' union. Measured one run later: cb1 transactions collapsed 94% (p1 never reads cb1, so no bit was ever set) while cb3 did not move AT ALL, 0.794 -> 0.796 per draw. [Perf.SplitXf.CbRead] named it on the same line -- p1{n=14685 cb 2=14677 3=8}, eight draws in 14,685 read cb3, and those eight set the bit permanently for all of them. So each entry now keeps saturating per-path reads/draws counters and is captured only when the path reads it on the MAJORITY of its draws. That rule is the cost model, not a tuned threshold: capturing costs one write-combined transaction unconditionally, while not capturing costs the same transaction as a live read but only on the draws that actually read it, plus a refused store there. Counters halve together on overflow, so a span read heavily during loading does not hold a path hostage for the session.\n"
               "pathDraws==0 means NOT YET MEASURED and captures for everyone, so entries learned before this shipped are never silently withheld.\n"
               "SCORING COUNTS RECORD HITS, NOT JUST LIVE READS. noteCbSpanRead only runs on the live path, so a span that was captured and then served from the record teaches it nothing; DrawSnapshot::rangeToManifest maps each captured range back to its manifest entry and m_xtCbRangesRead says which ranges were consumed. Without that half, every successfully-captured span would score as unread and the manifest would withhold its own working set -- a self-reinforcing collapse.\n"
               "LEARNED ON THE ACTUAL PATH, FILTERED ON THE PREDICTED ONE. Deliberate and not a slip: attributing on the prediction would close the loop on itself, filing a wrongly-predicted draw's reads under the wrong path and keeping them there forever. Attribution happens at the END of the derivation, beside the s_o2wPathByVsIl write, because m_lastO2wPathId does not exist when the early proj/view spans are read.\n"
               "READ IT AS: mfSkip/mfCap on [DrawSnap] is what the filter withheld. wcMiss and deadWc/f on [Perf.SplitXf.CbWaste] are the win and should fall together. refuse{unsafe (cb=)} on [Perf.SplitXf] is the cost and should move by roughly the disagree rate and no more -- if it jumps, the prediction is worse than o2wPath{} claims and the filter is buying transactions with stores. [DrawPure] why{miss=} should spike per VS and decay. FAIL and FAILcarrier must stay 0.\n"
               "IT COVERS THE NAMED SPANS TOO, ON A STRICTER RULE. Updated 2026-08-14 -- the first version filtered learned spans only, and measurement said that is where the cost was NOT: after it shipped, cb1 collapsed 94% and cb2 55% while cb3 did not move at all, because p1's residual cb3 waste was 1,739,200 B over 27,175 spans = exactly 64.000 B/span, i.e. the NAMED 64 B proj/view captures, not the 48 B learned o2w block.\n"
               "PROJ/VIEW USE A STRICT RULE, NOT THE MAJORITY ONE, AND THE DIFFERENCE IS A CORRECTNESS GATE. Their consumers maintain m_projSlot/m_projOffset/m_projStage/m_columnMajor/m_viewSlot/m_viewOffset, which is EXACTLY what kSdepCbLoc hashes -- so a live read that resolves differently rewrites them, and that IS a carrier move. Filtering them on the majority rule took FAILcarrier 0 -> 99..156 with grp{cbLoc} every window. They now withhold only after 255 consecutive draws on the path without a single read, where one read re-enables for another full run; 255 is the counter's width, not a tuned number. Measured with that rule: FAILcarrier=0 grp{} empty, wcMiss/draw 0.433 (from a 1.702 baseline), pfs_guard 14.92 ms -> 0.66 ms. cameraOrigin is NOT in that hash (memoCamOriginLoc is a per-shader memo) so it keeps the majority rule.\n"
               "kMaxCbManifest WAS RAISED 6 -> 10 FOR THIS. A withheld named span that some path does read is re-learned into the manifest and coexists with the named span, so shaders filled up: satN went 750 -> 4,687 -> 18,504 with manN 4.48/6. At 10 entries satN reads 0 with manN 4.75 and ovf 0. If satN ever returns, that constant is the knob -- learned spans decline through spanFits() rather than setting `overflowed`, so the range budget is not the binding limit.\n"
               "kMfFilterNamedProjView in d3d11_rtx.cpp is the one-line A/B if grp{cbLoc} ever returns: false keeps the learned-span and cameraOrigin filtering and gives back only the proj/view portion.");

    RTX_OPTION("rtx", bool, splitTransformKeySweep, false,
               "[Perf.KeySweep] Cycle the object-key predicate/drop ladder inside ONE run, then print a summary table. DEFAULT OFF. Requires rtx.splitTransformCache and rtx.splitTransformVerify.\n"
               "OVERRIDES rtx.splitTransformKeyPredicate / KeyDropTrue / KeyDropFalse while it runs. Reuses rtx.perfAutoSweepSeconds and rtx.perfAutoSweepSettleSeconds for the hold and settle windows rather than adding its own -- same driver shape as RtxContext::updatePerfSweep (gameplay gate with warmup, settle discarded at each step head, clock paused when gameplay drops out).\n"
               "EACH RUNG STARTS COLD: the object cache is cleared on every step change, because entries keyed under the previous rung can never match the next one and leaving them in makes the new rung's miss rate a property of the old rung. The settle window covers the refill.\n"
               "***MOVE THE CAMERA THROUGHOUT.*** Churn is a camera-motion phenomenon -- newObj/frame runs 15-21 standing still against ~730 moving -- so a stationary sweep scores every rung identically and tells you nothing.\n"
               "READ THE SUMMARY AS: lowest newObj/f with FAIL=0 wins. FAIL is the gate, not serve% -- a rung that raises serve and FAIL together broke the key, which is exactly what predicate=1 dropFalse=6 did (serve 12%->32-39%, FAIL 0->96-193). Check the two baseline rows against each other first: if they disagree the scene moved and the table is a cross-scene comparison, the error this sweep exists to prevent.");

    // DEFAULTS RAISED on 2026-08-13 to the configuration FAIL=0 was measured on:
    // predicate 1 (t31Valid), dropTrue 1 (spans), dropFalse 6 (geoC+geoSel).
    // Leaving these at 0 is SAFE but slow -- no drops means a fuller hash and
    // fewer serves. They are what the ~51% figure was taken with, so a build
    // that ships the mask defaults without these will read correct and slower.
    RTX_OPTION("rtx", uint32_t, splitTransformKeyPredicate, 1u,
               "[Perf.SplitXf] Which snapshot predicate selects the object key's component set per draw. 0 = off (full key, the safe default). SWEEPABLE FROM rtx.conf WITHOUT A REBUILD -- that is the point of it.\n"
               "VALUES: 1 t31Valid, 2 geoDynSrvT31, 3 instSemValid, 4 geoDynVsSrv, 5 geoDynSrvOther, 6 geoDynIb, 7 geoDynVbMask!=0, 8 layoutResolved, 9 instSemSlot!=UINT32_MAX, 10 t31CharIdx!=0.\n"
               "All are DrawSnapshot state, i.e. readable BEFORE the derivation runs, which is what makes a key selected from them free of the blind spot a learned per-VS mask has (a narrowed key that HITS never derives, so it never observes that the o2w path changed -- and 55% of draws are on shaders that take more than one path).\n"
               "WHAT IT IS FOR. [Perf.SplitXf.Path] measured that geo is read by p1 alone and spans by p13 alone, so most draws carry a component their path never reads -- that dead weight is the churn. Predicate 1 with dropFalse=6 was built and FAILED (96-193 FAIL/window, 206 of 216 on o2wPath=0): geo is also what separates p0, whose objectToWorld is always identity, from p13, whose is not, for the SAME shader. Judge a candidate by TOTAL far, never by farS -- a cross-path collision is still a wrong serve.\n"
               "SWEEP IT WITH VERIFY ON AND REQUIRE FAIL=0. far only ever compared objectToWorld; an entry also carries textureTransform, clipPlane, stablePropId, texgenMode, texcoordEncoding and the subView flags, so FAIL is the real gate and fld{} names the field.");

    RTX_OPTION("rtx", uint32_t, splitTransformKeyDropTrue, 1u,
               "[Perf.SplitXf] Components dropped from the object key when rtx.splitTransformKeyPredicate evaluates TRUE. Bits: 1 objectSpans, 2 geoContent, 4 geoSel. Default 0 = drop nothing.");

    RTX_OPTION("rtx", uint32_t, splitTransformKeyDropFalse, 6u,
               "[Perf.SplitXf] Components dropped from the object key when rtx.splitTransformKeyPredicate evaluates FALSE. Bits: 1 objectSpans, 2 geoContent, 4 geoSel. Default 0 = drop nothing.\n"
               "NOTE geoSel IS FREE TO KEEP: the ablation measured -geoSel changing dist/f by ZERO, so it contributes no churn. Prefer dropping 2 (geoContent alone) over 6 (both) -- geoSel may be carrying the p0-vs-p13 separation that dropping 6 destroyed.");

    // DEFAULT RAISED 0 -> 512 on 2026-08-13: p3 drops objectSpans, nothing else.
    // A pure serve lever and the clean case -- p3's chgR{} is empty, so spans
    // are never real there. p1's field is deliberately 0: spans ARE real on p1
    // (chgR spans=3786/window), and the p0/p1 collision that made them look
    // load-bearing is fixed properly by ObjKeyMask bit 22.
    RTX_OPTION("rtx", uint32_t, splitTransformKeyDropByPath, 512u,
               "[Perf.SplitXf] PER-PATH object-key drops, packed 3 bits per o2w path: bits (3*p) .. (3*p+2) hold path p's mask, same meaning as KeyDropTrue (1 objectSpans, 2 geoContent, 4 geoSel). Paths 0-9 fit; 0 = disabled, fall back to the predicate.\n"
               "IT CANNOT REACH p13 AND IT CANNOT SPELL cbLoc/ps/state. Both limits are structural: 3 bits per path in a uint32_t stops at path 9, and the three bits are spent. DO NOT ADD MORE DROP MASKS TO FIX THAT -- three per-component per-path masks were built on 2026-08-13 and removed the same day. They bought ~2 points, they are heuristics by construction, and dropping cbLoc from the validity hash introduced a real carrier defect (FAILcarrier grp{cbLoc} 0 -> 238 -> 337 with REPLAYFAIL=0 throughout) because an entry stored under the all-clean rule records no cbLoc and so cannot replay what the hash stopped checking.\n"
               "THE PROPER FIX FOR THIS CLASS IS TO HASH WHAT THE SITE READS, and rtx.splitTransformCamRelObjSpans is the worked example: p13's hash covered raw cb3 bytes when the site consumes a camera-corrected world transform, so it tracked camera motion and called it object motion. Correcting that took p13's spurPct from 67% to 7% -- an order of magnitude more than every drop mask combined, with no new per-path knob. p3 and p5 want the same treatment, not a mask that says 'ignore this component here'.\n"
               "WHY THIS EXISTS, AND WHY THE PREDICATE COULD NOT DO IT. [Perf.SplitXf.Bytes] sets{} measured p3 as 100% recoverable -- EVERY set R=0, dominated by [+spans]{S=2407 R=0} -- while p13's [+geoC+spans+t31Ent] reads {S=1737 R=2897}, i.e. spans changes there are REAL 63% of the time. Both sit on t31Valid=FALSE, so any predicate that drops spans for p3 also drops them for p13 and serves wrong transforms. The paths disagree, so the mask has to be indexed by path.\n"
               "IT IS APPLIED ON A PREDICTED PATH. m_lastO2wPathId is written BY the derivation and the key is built before it, so the path is predicted per (vs, inputLayout) from that shader's last derivation -- the same mechanism proven for the w2v path, which measured agree=48047 disagree=0 in one session. A misprediction is a WRONG SERVE, not a missed hit, because the drop changes what the validity hash covers. That is why the applied mask is recorded on the entry and mask{mismatch/wrong} on the report line is the gate.\n"
               "SWEEP IT WITH rtx.splitTransformVerify ON AND REQUIRE FAIL=0. Judge by TOTAL far, never farS -- a cross-path collision is still a wrong serve. Suggested first step: p3 only, i.e. (1 << (3*3)) = 512, which drops spans for path 3 and nothing else.\n"
               "OVERRIDES the predicate for any path whose 3-bit field is non-zero; paths left at 0 keep the predicate result, so the two can coexist while one path at a time is qualified.");

    // DEFAULT RAISED 0 -> 64 on 2026-08-13: p6 fully keyed, nothing else.
    // p6 measured spur=0 with a populated chgR, so masking it could only lose.
    // Bit 0 (p0) is deliberately NOT set: it was tried and measured inert,
    // because this option keys on the PREDICTED path and those draws predict p1.
    RTX_OPTION("rtx", uint32_t, splitTransformKeyNoDropByPath, 64u,
               "[Perf.SplitXf] Bit N forces o2w path N to drop NOTHING -- full validity hash, predicate and KeyDropByPath both overridden. Default 0.\n"
               "WHY A SECOND OPTION INSTEAD OF A FIELD IN KeyDropByPath. That option's 3-bit field already spends all eight of its values on real masks (0-7 = the eight combinations of spans/geoContent/geoSel), and it reads 0 as 'inherit the predicate'. There is no bit pattern left that means 'drop nothing', so 'leave this path alone' and 'leave this path FULLY KEYED' are not expressible in the same field. This says the second one.\n"
               "IT EXISTS BECAUSE A PATH CAN BE PURE LOSS. [Perf.SplitXf.Stale] measured p6 at stale{spur=0 real=47 spurPct=0} with chgR{geoC=47 spans=47 t31Ent=37} -- ZERO spurious stales, so dropping recovers nothing, while every component it drops is genuinely read. The predicate handed p6 mask=6 anyway (geoContent+geoSel) and the result was the session's only surviving plain FAIL, twice, both at t31skip=ran. A path whose spur is 0 has nothing to win and its whole hash to lose.\n"
               "READ spur BEFORE SETTING A BIT. High spur with empty chgR is what masking is for -- set the bit in KeyDropByPath, not here. spur=0 with a populated chgR is the opposite case and belongs here. Applied last, so a path listed in both options ends up fully keyed.\n"
               "SWEEP WITH rtx.splitTransformVerify ON AND REQUIRE FAIL=0. Setting a bit can only ADD hash coverage, so it can cost serve% and can never cause a wrong serve -- the safe direction to be wrong in, and the reason this is a bitmask rather than a per-path mask.");


    // NV-DXVK [CamRelSpans] 2026-08-13. The third repair of the same defect
    // bits 20 and 22 fixed: an input the derivation reads that the key modelled
    // wrongly. See the block at d3d11_rtx.cpp ~:25765 for the derivation reading.
    // DEFAULT FLIPPED false -> true 2026-08-14. It shipped in rtx.conf and has
    // run every session since; the default was the only thing still saying OFF,
    // which is the exact shape of a "swept an option nothing read" mistake in
    // reverse. Measured: p13 spurPct 67% -> 7%, an order of magnitude more than
    // every deleted drop mask combined, with FAIL=0 and FAILcarrier=0 alongside.
    RTX_OPTION("rtx", bool, splitTransformCamRelObjSpans, true,
               "[Perf.SplitXf] For the CAMERA-RELATIVE o2w paths (13 and 5), replace the raw cb3 contribution to the object validity hash with a hash of the camera-corrected world transform those sites actually build. DEFAULT ON since 2026-08-14.\n"
               "WHY THEY STALE WHEN NOTHING MOVED. Both sites build objectToWorld from cb3 with translation (m[3],m[7],m[11]) + c_cameraOrigin -- p13 at ~:20803, p5 at ~:21463, the same expression. p13's own comment records what cb3 holds: objectToCameraRelative. Move the camera and leave the object alone, and cb3's translation moves by -delta while c_cameraOrigin moves by +delta -- THE SUM IS UNCHANGED. The validity hash was over the raw cb3 bytes, so every static object on those paths was invalidated on every camera step while its cached transform stayed exactly correct. The hash was tracking CAMERA motion and calling it OBJECT motion.\n"
               "IT IS THE BULK OF ALL RECOVERABLE STALENESS. [Perf.SplitXf.Stale] measured p13 at spur=3439 real=1646 against p1's spur=7, with p13 at 457 draws/frame serving 27% where p1 served 75%; p5 measured spurPct=100 with chgR{} EMPTY. Nothing else in this feature is this size.\n"
               "IT IS ALSO A NARROWING, FOR FREE. These sites read cb3[0,48) and nothing else, while the slot-3 accumulator hashed every captured cb3 range -- [Perf.SplitXf.Bytes] shows s0c3+0, s0c3+48 and s0c3+64 in p13's population and the site touches only the first. Re-basing onto the twelve floats it consumes drops the other two by construction.\n"
               "IT IS KEYED ON WHAT THE SITE COMPUTES, NOT ON A PER-PATH TABLE, and that distinction is the whole point. Three per-component per-path drop masks were built and removed on 2026-08-13: they were tuned from a statistics table, bought ~2 points, and one of them introduced a real carrier defect. This reads the derivation and hashes its output, so adding a path means finding another site with the same expression -- not adding a knob.\n"
               "THE FAILURE MODE IS A LOST HIT, NOT A WRONG SERVE, which is why it is worth a run. If the camera origin resolves differently from the one the site adds, the correction fails to cancel and the entry stales exactly as it does now. Two different world transforms still hash differently, because the corrected value IS the world transform. Only rtx.splitTransformCamRelSpansQuantum can cause a wrong serve.\n"
               "p3 IS NOT COVERED AND THAT IS DELIBERATE. Its site (~:20086) builds o2w from BONE 0, not cb3, so its cb3 contribution looks like dead weight -- but that is an inference from one block against 119 spurious stales a window. Read the site properly before acting on it.\n"
               "SWEEP WITH rtx.splitTransformVerify ON AND REQUIRE FAIL=0. Expect p13 and p5 out{srv=} to rise and their stale{spur} to fall. If spur does not move, read priorF (what this targets) against sameF (identity contention, which this cannot reach).");

    RTX_OPTION("rtx", uint32_t, splitTransformCamRelSpansQuantum, 0u,
               "[Perf.SplitXf] Quantise the corrected p13 translation before hashing. 0 = OFF (no rounding, default). N>0 rounds to 1/2^(N-1) units: 1 -> 1.0, 2 -> 0.5, 3 -> 0.25, 4 -> 0.125, 5 -> 0.0625.\n"
               "WHY IT MIGHT BE NEEDED. bm's translation and c_cameraOrigin are large and opposite, so their sum cancels to a small residue and the low bits are where float cancellation lands. A genuinely static object can still jitter by an ULP between frames and stale. Rounding absorbs that.\n"
               "IT IS THE ONE WRONG-SERVE MECHANISM IN THE p13 CORRECTION, so it ships off. It also absorbs a REAL move smaller than the quantum -- exactly how the equivalent attempt on p1's geoC quantum FAILed with chg{t31Ent} when 1.0 proved too coarse. Powers of two so the rounding is exact in float and adds no error of its own.\n"
               "PICK IT FROM DATA, NOT BY FEEL. Read [Perf.SplitXf.Bytes] p13 quant{maxSpur= minRealRot=} and choose a value strictly between them. If no such value exists the two populations genuinely overlap and that reading is the finding -- leave this at 0 and take the correction unquantised.\n"
               "TRY 0 FIRST. If FAIL=0 and p13's spur collapses without it, the cancellation is exact and this option is not needed at all.");


    RTX_OPTION("rtx", uint32_t, splitTransformObjCacheMax, 16384u,
               "[Perf.SplitXf] Entry cap for the cross-frame object cache. On overflow, entries are aged out at the next frame boundary by a descending ladder of age bounds (300, 60, 8, 2 frames), stopping at the first bound that gets under the cap. Default 16384.\n"
               "THIS OPTION WAS DEAD UNTIL 2026-08-12. It was documented as sweepable from the day it landed and no code read it -- the eviction used a bare 16384 literal. Any earlier conclusion drawn from sweeping it (in particular 'raising the cap does not help, so the entries are genuinely single-use') was drawn against an unchanged 16384 and means nothing. Re-take it.\n"
               "THE LADDER REPLACED A FIXED 300-FRAME BOUND THAT COULD NOT FIRE. At the measured store rate of 597-663 new entries/frame the cap is reached in ~25 frames, so nothing in the map was ever 300 frames old, the sweep freed zero every time, and it fell through to the flat wipe it was written to replace -- roughly every 25 frames, taking the stable working set with it. A fixed bound only works if the store rate is under cap/bound (~55/frame here), which is the standing-still case that was never in trouble. Read evict{} on the report line: wipes must be 0, and rung names which bound did the freeing.\n"
               "RAISING THIS trades memory for hits. Each entry is ~200 bytes, so 16384 is ~3 MB and 131072 is ~26 MB. If raising it stops moving serve%, the working set fits and the remaining cost is churn (rtx.splitTransformObjKeyMask) or the store gate (rtx.splitTransformCarrierReplay), not capacity.");

    // DEFAULT FLIPPED false -> true on 2026-08-13. This option's own doc sets
    // the gate -- "REPLAYFAIL is the count of draws where the replayed write did
    // not match the derived one ... It must be 0 before this ships" -- and it is
    // met: replay{store=1319 serve=727 FAIL=0}, every window of a camera-moving
    // session. Without it, refuse{unsafe} blocks ~30% of draws from storing at
    // all and the serve ceiling drops hard. Paired with ReplayMask 5 below.
    RTX_OPTION("rtx", bool, splitTransformCarrierReplay, true,
               "[Perf.SplitXf] Let draws that MOVED A CROSS-DRAW CARRIER store into the object cache, by recording what they wrote and reproducing that write on every serve. DEFAULT ON since 2026-08-13. Requires rtx.splitTransformCache.\n"
               "THIS WAS THE STORE GATE'S BIGGEST SINGLE COST, AND WITH THIS FLAG ON IT IS NO LONGER THE BINDING ONE. The numbers that motivated it were taken with the flag OFF: refuse{unsafe} ~30% of all draws, carrier ~86% of it (9558 of 11168 per 3 s window), ceiling (draws-refused)/draws = ~71%. MEASURED 2026-08-13 WITH CarrierReplay=True Mask=5, draws=19238/window: refuse{noPath=136 unsafe=1030 (cb=289 carrier=210 geo=672)} -- unsafe is 5.4% of draws, carrier is 20% of THAT, and the ceiling is ~94%. DO NOT QUOTE THE 30%/71% PAIR AS CURRENT; a handoff did exactly that and recorded the store gate as the thing holding serve at 51% when it is not. Read refuse{} off the live report line instead.\n"
               "WHAT THE REFUSAL STILL COSTS. A refused draw stores NEITHER half, so it also starves the per-frame view cache -- which is why miss{view} must be read with `both` counted, not from the `view=` field alone. Those draws derive correctly every frame forever and no key work reaches them.\n"
               "WHY REPLAY IS SOUND WHERE RETIRING THE GROUP WAS NOT. Two previous attempts took cbLoc OUT of the carrier set -- a seed sentinel, then a frame-scoped hint -- and both were reverted, the second breaking the replay tier with FAIL=1. Those attempts CHANGED what an unresolved draw holds, and those members are an input to the replay tier's keying. This does the opposite: it honours the dependency by writing exactly the values the derivation would have written. Downstream readers see the same state either way.\n"
               "IT IS ONLY SOUND PER GROUP, WHICH IS WHY THE MASK EXISTS. Recording a value and replaying it is correct only when that value is a pure function of THIS draw's inputs. kSdepCbLoc qualifies: the resolved proj/view slot+offset is a property of the shader, and the object key contains vs+il, so an entry serves exactly the shader that produced it. kSdepCam does NOT: m_smoothedCamPos is an accumulator whose new value depends on its old one, so replaying a value recorded N frames ago writes a stale camera position. See rtx.splitTransformCarrierReplayMask.\n"
               "VERIFY IT. With rtx.splitTransformVerify on, every served draw replays, then derives anyway, then re-hashes the carrier groups and compares. REPLAYFAIL is the count of draws where the replayed write did not match the derived one, per group. It must be 0 before this ships -- MEASURED replay{store=1319 serve=727 FAIL=0}, every window.\n"
               "FAILcarrier IS NOT A SECOND OPINION ON THIS, AND UNTIL 2026-08-13 IT READ LIKE ONE. That counter used a bare wroteCarrier() predicate, so it counted every draw the replay had just handled: FAILcarrier=727 grp{cbLoc=727} against replay{serve=727 FAIL=0}, exactly equal in 8 of 8 windows. The gate 'both FAIL and FAILcarrier must be 0' was therefore unsatisfiable by construction whenever this flag was on, and a handoff read the resulting 500-1800/window as outstanding correctness debt in the carrier axis. It now subtracts the groups that were replayed AND that REPLAYFAIL confirmed, and reports the subtracted population as fgv{} beside grp{}. grp{} is the residue and is the number the gate means; fgv{} large with grp{} empty is the replay working.");

    // DEFAULT RAISED 4 -> 5 on 2026-08-13, matching the measured session.
    RTX_OPTION("rtx", uint32_t, splitTransformCarrierReplayMask, 5u,
               "[Perf.SplitXf] Which carrier groups rtx.splitTransformCarrierReplay may reproduce. Bit N = SdepGroup N. DEFAULT 5 = kSdepCam | kSdepCbLoc, both shipped and both measured at REPLAYFAIL=0.\n"
               "BITS: 1 kSdepCam (camera origins / fanout latches / smoothing), 2 kSdepRoute (hashes nothing by construction, so it can never fire), 4 kSdepCbLoc (shared proj/view cb slot+offset), 8 kSdepBone (bone/cb3 caches, measured at ZERO moves per window), 16 kSdepStatic (m_lastGoodTransforms + m_foundRealProjThisFrame fallback latches).\n"
               "A draw may store when every group it moved is in this mask. Groups outside it still refuse the store exactly as before, so widening this mask is the only thing that changes behaviour and each bit can be measured on its own.\n"
               "THIS MASK SELECTS; IT CANNOT EXTEND. Only groups whose replay is actually implemented (kXfReplayable in d3d11_rtx.cpp = kSdepCbLoc | kSdepCam) can be forgiven -- setting any other bit here does nothing at all. That is deliberate: forgiving a store for a group with no write would store the draw and then reproduce nothing, and with verify off that is silent. Adding a group is a code change (fields in the entry, the write in the replay block, the bit in kXfReplayable), not a config change.\n"
               "EACH GROUP IS REPLAYED FROM THE HALF THAT CAN HOLD IT WITHOUT GOING STALE, and that is the whole design. kSdepCbLoc is recorded on the OBJECT entry: the resolved proj/view slot is a property of the shader, and the object key carries vs+il. kSdepCam is recorded on the VIEW entry: it is camera state, the view cache is cleared at every frame boundary, and the view key contains comp[kMcCarCam] -- so a view entry is always same-frame and always from a draw that entered with the same camera carrier.\n"
               "THIS IS WHY kSdepCam WAS PREVIOUSLY REFUSED AND IS NOW ALLOWED. The old reasoning -- m_smoothedCamPos is an accumulator and the origin latches track the camera, so a recording goes stale -- was correct about the OBJECT cache and does not apply to a per-frame one. Both objections are objections to crossing a frame. Putting these fields on the object half would reintroduce exactly that bug, which is why kXfObjReplayable and kXfViewReplayable are separate constants and the serve ANDs each entry against its own.\n"
               "5 (cam|cbLoc) IS THE DEFAULT AND IT DELIVERED THE CEILING IT PROMISED. The prediction, taken with cbLoc alone: refuse{carGrp} cam~7,500 against cbLoc~1,480 per window, cam ~86% of what was left of the store gate, ceiling 80.5%, and replaying cam should take refusals from ~20% of draws to ~6% and the ceiling to ~94%. MEASURED at mask 5 on 2026-08-13: refuse{unsafe=1030 (cb=289 carrier=210 geo=672) carGrp{ cbLoc=159 static=18 camSm=86 }} of draws=19238 -- 5.4% refused, ceiling ~94%, and `cam` no longer appears in carGrp at all. The prediction held; the store gate is no longer the binding constraint.\n"
               "SO STOP ATTACKING THE STORE GATE. At mask 5 the residual refusals are geo=672 and cb=289 -- neither has a recording that fixes it -- while serve sits at 51% against a ~94% ceiling. The gap is now miss{obj}/objStale, not refuse{}. Read the report line before spending a run here.\n"
               "BIT 16 IS PROBABLY FREE AND UNMEASURED: kSdepStatic is a last-known-good fallback latch at ~48 draws/window, order-affecting but not consumed as this draw's answer. Not worth a build on its own; turn it on only if refuse{carGrp} says it matters.\n"
               "REPLAYFAIL IS THE ARBITER FOR EVERY BIT HERE. It fingerprints the carrier groups after the replay, derives anyway, fingerprints again, and compares per group -- so a group whose recorded value is not reproducible names itself. cam is the bit most likely to fail, being the one whose members track something outside the draw; if it does, check whether the offenders are fanout draws (the CARRIER line prints instIdx= and t31=).");

    // ==================================================================
    // NV-DXVK [XfDefer] 2026-08-19 -- STEP 1 OF HANDOFF_XFDEFER Sec 8:
    // DECOUPLE THE TIER FROM cbLoc. These three ship together and the first two
    // default ON as a PAIR -- turning one off alone is a measurement, turning
    // the second off alone is a defect. See below.
    RTX_OPTION("rtx", bool, splitTransformCbLocResolved, true,
               "[Perf.SplitXf] Key the object validity hash and the view key on the RESOLVED cb location (the per-VS sVsCbLocCache entry, recorded as DrawSnapshot::cbLoc) instead of on the entry-state fingerprint carrierGrp[kSdepCbLoc]. DEFAULT ON.\n"
               "WHAT WAS WRONG WITH THE FINGERPRINT. carrierGrp[kSdepCbLoc] hashes the six m_projSlot/m_viewSlot MEMBERS as they stood at SubmitDraw entry -- i.e. whatever the PREVIOUS DRAW left behind. ExtractTransforms then OVERWRITES all seven of them from this shader's per-VS entry before anything reads them, and the older XtReplayRec tier never reads the members at all (its route proof compares vsLocGen/vsLocNeg, the same per-VS entry). So on a resolved draw the fingerprint could not influence the derivation, while re-keying that shader's entry every time a different shader ran before it. TF2 swaps VS on ~25% of draws.\n"
               "WHY IT IS NOT 'DROPPING cbLoc FROM THE KEY'. It is a SUBSTITUTION. The resolved location is a property of the shader and its cb layout, both already in the object key, so it is order-independent AND strictly more precise than what it replaces. An unresolved draw (haveLoc false) additionally folds the raw fingerprint back in, because there the members really are an input -- a first-sighting VS borrows the previous draw's location as its search seed, the bootstrap channel two reverted attempts proved load-bearing.\n"
               "IT REQUIRES rtx.splitTransformCbLocReplayAlways, and that is not optional. The old fingerprint was ACCIDENTALLY guarding a second thing: an entry stored under the all-clean rule records carGroups=0 and replays nothing, so serving it to a draw that entered with another shader's location skipped a write the derivation would have made. The fingerprint refused those serves as stale. Remove it without the always-replay and that defect comes back -- measured once already, as FAILcarrier grp{cbLoc} 0 -> 238 -> 337 with REPLAYFAIL=0 throughout.\n"
               "MEASURED BY serve% and miss{objStale} on [Perf.SplitXf], with FAIL and FAILcarrier as the gates. Sweep it False to get the before reading in the same session.");

    RTX_OPTION("rtx", bool, splitTransformCbLocReplayAlways, true,
               "[Perf.SplitXf] Replay the recorded cb-location exit state on EVERY object-cache serve, not only on entries whose producing derivation moved it. DEFAULT ON. Requires rtx.splitTransformCarrierReplay (it is ANDed with it).\n"
               "THE CORRECTNESS PARTNER OF rtx.splitTransformCbLocResolved -- read that option first. carGroups answers 'did the producer MOVE cbLoc'; the serve needs 'are the recorded values this shader's derivation exit state', and those differ on every entry stored under the all-clean rule.\n"
               "WHY ALWAYS IS SOUND. The store gate admits only cbSafeToDefer() draws, which implies layoutResolved, which implies the VS had a resolved location -- so the derivation opened by loading that location and its EXIT state is a pure function of inputs the key and the validity hash already cover, never of the borrowed seed. The six fields were already written unconditionally at the store (plain assignments, not gated on the forgiveness), so this records nothing new.\n"
               "IT ALSO WIDENS THE STORE GATE, and store and serve must agree or a draw is admitted under one rule and replayed under another: with this on, kSdepCbLoc is forgiven regardless of rtx.splitTransformCarrierReplayMask bit 4. That is what xReplayMaskEff is in the code.\n"
               "REPLAYFAIL IS STILL THE ARBITER, now over a larger population. It must stay 0.");

    RTX_OPTION("rtx", bool, xfDeferSeedCbLocFromRecord, false,
               "[XfDefer] Seed the cb-location members and the projection neg-cache verdict from DrawSnapshot::cbLoc instead of from this thread's sVsCbLocCache. DEFAULT OFF -- it is the behaviour half of a measurement whose instrument ships on.\n"
               "WHY IT EXISTS. sVsCbLocCache is thread_local, so a worker running a deferred derivation reads ITS thread's map -- cold on the first sighting of every shader. The derivation would then depend on which draws that worker happened to get, which is the same order dependence step 1 removes, moved rather than fixed. The record already holds the entry, resolved on the frame thread in this draw's own SubmitDraw, so seeding from it makes the derivation a pure function of the record on any thread.\n"
               "READ xt_locDis AND xt_negDis ON [Perf.Report] BEFORE SETTING THIS. They count draws where the record's copy and the live entry differ at the moment the derivation reads them, and they are counted with this flag OFF. Both must be 0: non-zero means something writes sVsCbLocCache between capture and the override, and that writer has to be found rather than tolerated. xt_locFromRec is the population the seed actually served and stays 0 while this is off -- so xt_locDis=0 with xt_locFromRec=0 is the CONTROL, not a result.\n"
               "IT IS NOT THE THIRD SEEDING ATTEMPT. The two reverted ones (sentinel seed, frame-scoped hint) changed what an UNRESOLVED draw holds and so cut the bootstrap borrow, taking resolved% from 79% to 41%/49% and breaking the replay tier with FAIL=1. This changes nothing for an unresolved draw -- haveLoc false in the record falls through to the identical live path -- and only substitutes one copy of a resolved location for another copy of the same location.");

    RTX_OPTION("rtx", bool, xfDeferRoute, false,
               "[XfDefer] Take the routing verdict for the transform derivation and publish it on the record (DrawSnapshot::deferRouted). DEFAULT OFF. NOTHING DISPATCHES ON IT YET -- with it on, the decision is computed and counted but every draw still derives inline, so this is a SIZING instrument first and a switch second.\n"
               "WHAT IT ANSWERS. [Perf.Report] xfRouted says how many of the ~1080 draws/frame would leave the frame thread. Read it against [Perf.EligCost] to price the extractXf -> end-of-SubmitDraw code motion (~9-10 ms, 23 downstream transformData references) BEFORE doing it. xfRefuse{rec= taint=} splits the rest: rec is the record itself (not arena-backed, layout unresolved, overflowed) and nothing can fix those; taint is the shader-level throughput hint.\n"
               "THE GATE IS NOT THE SAFETY ARGUMENT. It conjoins DrawSnapshot::deferrable -- exact, all three terms are properties of the record -- with a learned shared-carrier shader set that is a THROUGHPUT HINT. A shader not in the set is not proven clean; the set is learned, so a shader's first shared-carrier draw always passes. What makes routing safe is that every carrier a deferred derivation can write is now thread-private (kSdepCbLoc's seven members, kSdepCamSmooth's accumulator, kSdepCam's per-draw scratch) or a monotone atomic latch (m_foundRealProjThisFrame, m_hasEverFoundProj), so such a draw has corrupted nothing by the time xfDeferMustAbort() sees it -- it has wasted work, and the abort reclaims it. The residuals are m_lastGoodTransforms, which keeps its mutex and is a last-known-good fallback rather than this draw's answer, and kSdepCam's frame-state half, which the derivation only reads.\n"
               "WHY THE HINT IS WORTH HAVING. With kSdepCbLoc privatised the shader partition collapses: drawsInCarrierVs/drawsSeen 94.3% -> sharedDrawsIn/drawsSeen 14.3%, tainted shaders 37 -> 17 of 76. The earlier verdict that a shader partition is dead was taken on the first pair, which is 97.7% kSdepCbLoc -- it measures a different predicate and will keep saying 89%.\n"
               "READ xfAbort{geo= cb= carrier=} AGAINST xfRouted, NOT AGAINST draws, and do not read them as a partition -- a draw can trip more than one. geo is the only axis that could ever have corrupted anything and was measured at 6 live reads in 36,015 draws. carrier is the rate at which the hint is wrong.\n"
               "A DISPATCHER MUST CALL D3D11Rtx::xfDeferLearn FROM THE ORDERED JOIN for routed draws. The set is consulted on the frame thread and is thread_local; a worker learning into its own copy would leave the gate permanently blind to whatever that worker discovered.");

    // DEFAULT RAISED 255 -> 767 on 2026-08-13: adds bit 512, the view-half
    // serve gate. 255 alone leaves one viewKey able to serve two different
    // worldToView fixups (measured: w2vPath 1->3, fields{o2v w2v v2p w2vPath}
    // with o2w absent). Bit 256 stays OFF -- it is a wrong-serve mechanism.
    RTX_OPTION("rtx", uint32_t, splitTransformViewKeyMask, 767u,
               "[Perf.SplitXf] Which components go into the PER-FRAME view key. Bitmask, default 255 = all 8. Sweepable from rtx.conf without a rebuild.\n"
               "BITS: 1 viewport, 2 carCam, 4 carStatic, 8 cbLoc, 16 viewSpans, 32 boneDraw, 64 rtv, 128 vsIl. Bit 128 is ANDed with rtx.splitTransformViewKeyVsIl.\n"
               "BIT 256 SUBSTITUTES the PREDICTED fixup path for vsIl in the key, collapsing ~4,000 (shader, view) pairs onto (fixup, view) pairs. It is a WRONG-SERVE mechanism when the prediction misses: measured 12 FAIL/run reading fld{o2v w2v v2p w2vPath} with o2w absent and w2vPath 1->3. It also makes the vkPath census circular -- the key IS the prediction, so a lookup can only find entries whose prediction already agreed, and disagree reads 0 while draws render wrong. Leave it OFF.\n"
               "BIT 512 GATES THE SERVE on the same comparison instead of keying on it: refuse a view entry whose recorded worldToViewPathId disagrees with this draw's predicted path. This is the view half's copy of the object half's storePath gate (rtx.splitTransformObjKeyMask bit 131072) and exists because the discriminator is NOT expressible as a key component -- which fixup runs depends on cross-draw carrier state (derivePath3WorldToView succeeding, m_hasFanoutVpRows, the slot pick), not on any property of the view. A refused draw derives exactly as on a miss, so this can only cost serve%, never correctness. Priced by the census at agree=32818 disagree=4 per window. Requires bit 256 OFF to be meaningful.\n"
               "THE VIEW HALF IS NOW THE DOMINANT MISS, and the older note that miss{view} runs 'under 1%' is a misreading: it read the `view=` field alone and ignored `both=`, which is also a view miss. Measured standing still at serve=84%: miss{obj=801 view=3830 both=2579} of draws=46640 -- object 7.2%, VIEW 13.7%. With the store gate fixed the ceiling is 95%, so the view half is the larger part of what is left.\n"
               "vsIl IS THE COST AND IT CANNOT SIMPLY BE REMOVED. It took store{view} from 390-590 to ~4,000 per window, but without it FAIL reads fld{o2v w2v v2p} with o2w ABSENT -- the view matrix itself wrong, because worldToView is BUILT by ~11 tagged sites applying different fixups to the same cb2 bytes and which one runs is a function of the shader. The real discriminator is WHICH FIXUP PATH RUNS, which is coarser-grained than shader identity; this mask exists so a cheaper candidate can be compared without a rebuild each time.\n"
               "SWEEP WITH VERIFY ON. FAIL=0 is the gate. A bit that lowers store{view} and miss{view} with FAIL=0 was never an input; one that lowers them and raises FAIL was load-bearing.");

    RTX_OPTION("rtx", bool, splitTransformViewKeyVsIl, true,
               "[Perf.SplitXf] Include vertex-shader + input-layout identity in the PER-FRAME view key. Default true.\n"
               "TRUE IS CORRECT AND WAS PAID FOR. With it false the first serving run read fld{o2v=113 w2v=113 v2p=93} with o2w ABSENT -- the object half byte-correct and the view half wrong. worldToView is not read from a buffer, it is BUILT by ~11 tagged sites that apply different fixups (transpose, Y-flip, Source->D3D axis remap, fallback latches) to the same cb2 bytes, and which one runs is a function of the shader. Turning it on took FAIL to 0.\n"
               "IT COSTS: store{view} went 390-590 -> 1437-1677 per window. Exposed as an option because the RIGHT key is whatever selects the fixup path, not full shader identity, and if a cheaper discriminator is found this is how it gets compared. Setting it false is a diagnostic, not a configuration -- it reintroduces a known wrong-matrix bug.");

    RTX_OPTION("rtx", bool, perfDrawRedundancy, false,
               "[DrawRedund] -- THE OPTIMISATION PLAN, Phase 1.2. Hash each draw's DrawSnapshot frame to frame and report what fraction is IDENTICAL to the same draw last frame. DEFAULT OFF. Requires rtx.useDrawSnapshot.\n"
               "WHY IT IS BLOCKING. Phase 3 ('skip the derivation for unchanged draws') is the only line item large enough to close the frame thread, and its size is entirely unknown. INSTANCE-level redundancy is measured at 97% ([Perf.UpdInst] REDUNDANT), but that is the dxvk-cs side, counted per instance after the derivation has already run. DRAW-level redundancy -- the thing that decides whether the derivation can be skipped at all -- has only ever been INFERRED from it. This measures it.\n"
               "IT MEASURES AN UPPER BOUND, NOT THE POPULATION. The hash covers what DrawSnapshot covers: shader/layout/state identity, vertex+index+SRV BINDINGS, cbuffer bindings with their contentGen, the captured cbuffer BYTES, and the t31/COLOR1 entry. It does NOT cover bone palettes, vertex/index buffer CONTENTS, or PS state. That gap is not academic -- it is exactly the defect that shipped a visible bug once already: rtx.filterDupSameDraws v1 read the viewmodel's alternate-pose passes as byte-identical because bones are not hashed (12 KB+/draw), and the skipped commits made the gun flicker. So read `same` as a CEILING and read `sameNoSrv` as the conservatively sound floor -- a draw with no VS SRV bound has no bone or instance stream for the record to be blind to.\n"
               "THE CROSS-TAB IS THE REAL QUESTION, and it is the one the plan's arithmetic hangs on. Phase 1.1 resolved that ~95% of pfs_guard IS captureDrawSnapshot (~13 ms/frame, ~11 of it the cbuffer span copy), so a skip test that has to BUILD the snapshot first cannot go below the capture floor. The only way out is a test that runs on something cheaper than the bytes it is avoiding -- and CbBinding::contentGen is exactly that candidate: it is available from the binding, before any byte is copied, and it moves only on DiscardSlice() i.e. only on a rename. So the probe cross-tabulates the cheap test against the true one:\n"
               "  cheapOk    bindings+generations identical AND bytes identical -- the cheap test is right, and this is Phase 3's real population.\n"
               "  cheapWRONG bindings+generations identical BUT bytes differ -- THE NUMBER THAT KILLS IT. Any nonzero reading means a rename-free write path exists (Map(WRITE_NO_OVERWRITE), UpdateSubresource, or a discard that does not bump the generation) and skipping on generations alone would serve stale transforms. This must read 0 before anything is built on it.\n"
               "  cheapMiss  generations moved but the bytes did not -- pure lost opportunity, safe, and it sizes what a content hash would recover over a generation compare.\n"
               "MATCHING DRAWS ACROSS FRAMES is itself an open question and is reported both ways, because a skip mechanism needs a key and nobody has shown one exists. pos{} matches the Nth CAPTURE of this frame against the Nth of last frame, which assumes the engine emits the same draw in the same slot every frame -- note the ordinal is counted by the probe and is deliberately NOT DrawSnapshot::drawIndex, which is m_drawCallID and only advances for draws that survive the filters, so it is not a dense per-capture index. key{} matches by (identity hash, occurrence ordinal of that identity within the frame), which survives a draw being inserted or dropped earlier in the stream. If keyHit is high and pos{same} trails key{SAME}, the stream reorders and Phase 3 needs the keyed form; if the two agree, the dense capture ordinal is enough and the mechanism is far cheaper than a per-frame map.\n"
               "COST. Four XXH64s over ~1-2 KB plus one hash-map insert and one lookup per draw, and a per-frame map rollover. It is billed to NOTHING -- the call site re-stamps the stage marker afterwards, the same way the StageDep census does, so pfs_guard and every leaf around it stay comparable with prior logs. It is still real time on the pole. TURN IT OFF BEFORE TAKING ANY FRAME-TIME NUMBER (see the 55x distortion note in the plan's Appendix C.5).");

    RTX_OPTION("rtx", bool, perfSceneObjSplit, false,
               "[Perf.SceneObj] -- split InstanceManager::processSceneObjectImpl into find / mid / add / update. DEFAULT OFF.\n"
               "This is the DEEPEST LEAF the measured chain reaches: frame 73.6 ms -> dxvk-cs 73.3 (99.6%) -> commitGeometryToRT 54 -> submitDrawState 50 -> processDrawCallState 39 -> [ProcDCS] instMs 32. So processSceneObject is ~32 ms/frame at 29us/call, 44% of the whole frame.\n"
               "find = findSimilarInstance, the dedup search. If it dominates, the fix is the dedup key rather than the instance work -- and note the propId round-robin already documented under rtx.suppressStablePropIdVsHashes forces 100% of some shaders' lookups into the nearest-neighbour stage. mid = decision logic plus the per-INSTANCE diagnostics between search and instance work. add = addInstance, only on a dedup miss (addedPct reports how often). update = updateInstance, expected to dominate on a steady scene where nearly every draw dedups.\n"
               "SIZING THE GAP. [Perf.UpdInst] floor-corrects to ~9.1 ms/frame and [ProcDCS] instMs reads 25, so ~15.7 ms/frame -- 29% of a 55 ms frame -- lives in find + mid + add and has never been attributed. That is what this switch is for. Sum the four estMsPerFrame values against instMs; a large residual means a stage boundary is misplaced, not that the time vanished.\n"
               "COST -- CORRECTED 2026-08-07. This read '~0.23 ms/frame (4 clock reads per call)', budgeted against ~1,119 DRAWS while the guard is billed per INSTANCE (~15,500/frame): the true cost of timing every call was ~3.2 ms/frame, 14x the stated figure, on a probe measuring 25 ms. Durations are now sampled 1 instance in 64 (~0.05 ms/frame) and all counts stay exact, matching rtx.perfUpdateInstSplit. Turn off after the capture.\n"
               "ALSO EMITS [Perf.MidWork] -- exact per-instance counters for the five diagnostic blocks inside mid (mtn / subvk / pass / prop / vs2904), plus mapOps, strMaps, fmt and fmtDropped. These are COUNTED, NOT TIMED: each block is sub-microsecond against a ~41 ns clock read, so sub-timers would be measurement-floored and the mechanism is what decides the fix. fmtDropped is str::format work whose output the log.cpp denylist discards -- [PassCensus], [InstCounts], [VS2904Trace] and [SubViewKey.create] are all denied, so any nonzero reading there is pure waste that deleting the block recovers in full.");

    RTX_OPTION("rtx", bool, useDrawSnapshot, false,
               "[DrawSnapshot] Capture each draw's D3D11 inputs into an owned, immutable record at SubmitDraw entry. DEFAULT OFF.\n"
               "THE PROBLEM IT SOLVES. SubmitDraw's derivation reads LIVE context state -- ~460 sites across 24 distinct m_context->m_state paths -- so it can only run on the frame thread, in draw order, before the game rebinds for the next draw. That is the only reason the derivation is serial, and it is why proving two draws independent needed a whole cross-draw carrier census (rtx.perfStageDepCensus) rather than being a property of the data. A production renderer inverts this: UE's FMeshDrawCommand owns its bindings, streams and counts and references no context, which is what lets its pass setup build and sort commands on a worker with ordering carried by a sort key instead of by thread joins.\n"
               "THE RULE THAT MAKES IT CORRECT. Com<D3D11Buffer> pins the buffer OBJECT, not its CONTENTS -- a dynamic buffer the game Map(WRITE_DISCARD)es keeps the same D3D11Buffer while its backing slice is renamed underneath. That is the dropship COLOR1.y race and the getImageHash streaming race, both already paid for here. So anything whose BYTES the derivation reads is COPIED at capture time on the frame thread; a ref is permitted only for identity and lifetime.\n"
               "WHAT IT DOES TODAY. NINE content consumers are converted, all fallback-safe (find() returns nullptr and the original live path runs unchanged): freshProj, the projOk memcmp, capViewHit/isViewMatrix, cb3->objectToWorld, and five c_cameraOrigin readers (path-13 replay refresh, viewmodel, path 1, path 13, and the replay RECORD capture -- so the record's camera input now comes from the same instant as its proj/view spans).\n"
               "CAPTURED SPANS: projection 64 B and view 64 B are NAMED (derived from the per-VS location cache); everything else is LEARNED by the per-VS span manifest, which records what the derivation actually read and replays it on the next draw of that shader. The cb3 object-to-world span and its per-VS predictor were deleted 2026-08-10 -- the manifest owns slot 3 now, taught by the o2w consumer itself once that site was routed through drawCbSpan's narrowScratch overload. Do not re-add a hand-named cb3 span or a predictor for it.\n"
               "ONE c_cameraOrigin READER IS DELIBERATELY LEFT LIVE -- the VP+origin site (~:17790, camLocL). It reads the 64 B view-projection matrix at baseCam+12, so a 12 B span would serve half its data from the record and half from live state, and mixing sources inside one read is how a convention bug starts. If it is ever converted, capture a 76 B span for it; do NOT widen the shared 12 B one.\n"
               "THE HARDCODED cb2@4 FALLBACK (~:9600, src='H') is also not served from this span, by definition: it names a different location than the RDEF-resolved one. All six RDEF-resolved readers were checked to share one convention -- cb.constantOffset*16 + camLoc->offset, 12 bytes -- BEFORE the span was added, per trap 2.\n"
               "HOW THE CAPTURE IS PRICED -- CORRECTED 2026-08-10. Do NOT try to price this by A/B-ing the flag. captureDrawSnapshot is called at SubmitDraw entry, INSIDE the pfs_guard span, which also holds the MeshTrace entry hash, the SubmitCpuGuard and the batch-arena reset -- and pfs_guard is the largest row in the whole [Perf.SubmitDraw] report (636 ms per 5 s window = 24.5% of all SubmitDraw time, 6.28 us/draw, and RISING across a session: 5455 ns/draw at 03:17:06 to 6279 at 03:18:11 with wallUs/draw flat). An A/B moves that bucket without saying which part moved. Read the three ns leaves instead: drawSnap_ns (whole call), drawSnapAlloc_ns (arena slot acquisition), drawSnapId_ns (identity/stream/viewport field copies). drawSnap_ns - (alloc + id) is the named-cbuffer-span copy, the only part any consumer reads.\n"
               "WHAT THE RECORD COSTS PER DRAW. DrawSnapshot is ~2.7 KB: 128 SRV slots = 1024 B, 32 vertex-buffer bindings = 512 B, cbBytes = 512 B, 16 viewports = 384 B, plus the rest. The arena is 4096 slots, ~11 MB. clear() at frame end keeps capacity but destroys the elements, so every draw value-initialises a fresh ~2.7 KB slot -- cache-cold write-allocate traffic, ~2.5 MB/frame at 913 draws/frame. Of that record, ~1.9 KB is identity that NO consumer currently reads (grep: the only snapshot reads in the tree are four m_drawSnapCur->find() calls, and find() touches cbRanges/cbBytes exclusively). Those fields are captured for the future pure-function design, which is correct as intent -- but the drawSnapId_ns leaf now says what the intent costs, so shrinking or gating the record is a decision on a number rather than on intent.\n"
               "WHY THE BYTE CAPTURE IS NARROW. The live consumers call stagedCbBytes(), which returns the WHOLE buffer, and several SCAN it (the offset-64 projection scan, the cross-stage all-cb scan, the m_viewSlot hunt). A scan cannot be made pure by a narrow copy, and copying whole cbuffers per draw is the ~6.3 KB/call WC staging cost [Perf.WcCopy] billed at 91% of pole-thread WC bytes. So only spans that can be NAMED are captured; everything else reports find()==nullptr and stays on the live path -- the same population the replay tier already leaves serial.\n"
               "WHICH SPANS ARE CAPTURED. Driven by sVsCbLocCache, the per-VS location cache ExtractTransforms itself populates and consumes: the projection and view matrices at that VS's resolved (slot, offset), 64 B each, plus the cb3 object-to-world block (48 B at constantOffset*16) whenever cb3 is bound. Stage 0 (VERTEX) only -- not a shortcut, because the replay tier's own eligibility already requires m_projStage == 0, so stage 0 IS the replayable population. A VS with no resolved location yet captures nothing and takes the live path.\n"
               "DO NOT re-point this at VsLayoutDescriptor / m_vsLayoutTable. That was the first implementation and [DrawSnap] measured resolved=0% on every window: the table is populated only under rtx.capturePhase2 or rtx.logPhase0Descriptor, so in a production run it is empty. It is a formalisation built for the GPU-driven-injection experiment, not live state.\n"
               "READING [DrawSnap] -- MEASURED 2026-08-10: resolved=79-80%, meanRanges=2.59, ovf=0, with the replay tier reporting FAIL=0 alongside. 80% is the CORRECT ceiling, not a shortfall: it is the reachedTier share of all draws (StageDep 6505/8144 = 79.9%), and the ~20% that do not resolve are UI/fallback draws that never resolve a projection, so there is no span to name. Do NOT chase the 92% figure -- that is replayed-over-reachedTier, a different denominator, and an earlier version of this doc wrongly named it as the target. meanRanges decomposes as cb3 ~1.00 + proj ~0.79 + view ~0.80. ovf>0 means a range did not fit the caps in DrawSnapshot and MUST be fixed, not ignored -- an incomplete snapshot that gets derived from is a wrong result.\n"
               "PURITY LEDGER -- live m_context->m_state reads inside ExtractTransforms, 2026-08-10. Was 106 across 11 paths; now 34 across 6, and 2 of those 34 are deliberate. Converted via the accessors drawViewport0 / drawViewportKey / drawVertexShaderCom / drawInputLayout / drawVsSrv / drawVertexBuffer / drawRtv0: rs.viewports 10->0, rs.numViewports 5->0, ia.vertexBuffers 5->0, om.renderTargetViews 2->0 (new rtv0 capture field -- presence only, both consumers just test 'is anything bound'), vs.shader 35->0, ia.inputLayout 11->1, vs.shaderResources 6->1. The two survivors are the live-path FALLBACK branches inside the converted sites, which are supposed to read live state.\n"
               "WHAT IS LEFT: 30 reads, and 'cbuffers' is TWO problems, not one. (a) BINDING IDENTITY -- which buffer, at what offset, at what content generation. Pure, cheap, and now captured as DrawSnapshot::vsCbs; the camera-fallback cache keys were converted onto it. (b) CONTENTS -- the scans, and these are the actual blocker: the offset-64 projection scan, the cross-stage all-cb scan (that is what ps/gs/ds.constantBuffers 2-each are: pointer arrays handed to a scan, not reads of their own), and the m_viewSlot hunt. A scan is not made pure by a narrow copy -- see WHY THE BYTE CAPTURE IS NARROW above. Do not conflate the two and conclude 'cbuffers are impossible'.\n"
               "contentGen IS A DEFERRAL PREREQUISITE, NOT A CONVENIENCE. DrawSnapshot::CbBinding::contentGen is sampled at the capture instant because it moves ONLY in DiscardSlice(), i.e. only when the backing slice is RENAMED -- exactly the dropship COLOR1.y failure, where Map(WRITE_DISCARD) swapped the bytes under a pointer that still compared equal. A consumer running off the frame thread cannot ask 'are these still my draw's bytes?' by re-reading the buffer, because by then the answer has already changed. Any future deferred reader MUST compare against the captured generation, never a fresh GetContentGeneration().\n"
               "ACCESS RULE. Read a snapshot ONLY through drawSnap() and the accessors above, never m_drawSnapCur directly. m_drawSnapValid is latched per draw at the capture point, so the accessors are correct anywhere reachable from SubmitDraw and WRONG outside it (they would serve the previous draw's bindings). 10 of the 22 ia.inputLayout reads in d3d11_rtx.cpp are outside SubmitDraw, which is why this conversion was applied per site rather than by a blanket replace.\n"
               "THE SCANS ARE A COLD-START PATH -- MEASURED 2026-08-10, AND THIS CHANGES THE PLAN. The scans are what supposedly blocks deferral. Over the 03:18 windows they fired ZERO times: xt_projScan1N=0 (cross-stage projection scan), pv_rescanN=0 (rescan), against xt_vsLocHits=5849 with xt_vsLocMiss=0. The scan is guarded by `projSlot == UINT32_MAX && !skipExpensiveProjScan`, and a VS with a resolved sVsCbLocCache entry sets projSlot from that cache before the test. So in steady state nothing scans; the scans run on a VS's first sighting, after the 4096-entry cache clear, and after a neg-cache TTL expiry.\n"
               "THEREFORE the question is not 'how do we make a scan pure' -- it is 'can we tell, before deriving, that this draw will not take one'. DrawSnapshot::layoutResolved is that answer and it is free: the capture already does the sVsCbLocCache lookup to decide which spans to copy. Partition on it; do not try to purify the scan.\n"
               "layoutResolved IS NOT A PURITY CERTIFICATE. It says the draw skips the scans. It does NOT say the derivation is a pure function of the record: the CONTENT reads still go through stagedCbBytes()/GetMappedSlice(), which read live buffer bytes at consume time, and the m_last* cross-draw carriers are a separate axis that the StageDep census measures. Deferring on this flag alone repeats the COLOR1.y mistake.\n"
               "THE CONTENT HALF IS NOW A MECHANISM, NOT A LIST -- 2026-08-10. Cbuffer CONTENT reads in ExtractTransforms go through D3D11Rtx::drawCbSpan(slot, relOffset, byteCount, rebase), which serves the bytes from the record when the span was captured and otherwise reads live exactly as the site did before. Converting sites one at a time did not scale and could not: each site re-derived its own offset arithmetic, and the proj-vs-view convention split (one adds constantOffset*16, the other does not) made every new site a fresh chance to capture a different 64 bytes than the consumer reads. Here the convention is a PARAMETER used by both the lookup and the capture, so the two cannot disagree by construction. rebase is deliberately NOT defaulted -- spelling it out at every call site is the point.\n"
               "THE SPAN MANIFEST -- how the capture learns what to copy. VsCbLocations::manifest is a bounded (6-entry) per-VS list of the spans that shader's derivation was OBSERVED to read: drawCbSpan appends when it has to serve a read live, and captureDrawSnapshot replays the entries on the next draw of the same shader, after the named spans and deduped against them via find(). This is cb3Want generalised from one hand-named span to all of them, and it inherits the same property: BEING WRONG IS FREE. A span not yet learned is served live -- correct, just not pure -- and learning it costs one bounded array write. Offsets are stored RELATIVE with their convention so a cbuffer bound at a different constantOffset still resolves; storing absolutes would not be unsafe (find() would simply miss) but would never converge. Learned spans are BEST-EFFORT: they check spanFits() first, because `overflowed` invalidates the whole record and a learned span must never be able to do that. Like cb3Want, manifest entries must NEVER bump the location entry's `gen`.\n"
               "THE PURITY CERTIFICATE. DrawSnapshot::cbLiveReads / cbRecordReads are written at the END of the derivation (RAII, so every exit publishes) and contentPure() means cbLiveReads==0 && cbRecordReads>0: this draw's derivation read cbuffer content ONLY from bytes copied at capture time, under the captured generations. That is the fact layoutResolved could not assert and the fact a deferred reader needs. STILL NOT SUFFICIENT ALONE -- it covers cbuffer content; the m_last* carriers are a separate axis (rtx.perfStageDepCensus) and vertex/index/bone bytes have their own capture path. A deferral gate is the CONJUNCTION of all three, still proven by FAIL=0.\n"
               "THE CARRIER AXIS IS A WRITE TEST, NOT A READ TEST -- and the wrong turn is instructive. The obvious move is to capture the m_last* carriers into the record and serve the derivation's ~99 carrier reads from the copy. That does NOT make deferral correct: the carriers are written BY the derivation, so if draw N and N+1 are both deferred, N+1's capture runs on the frame thread BEFORE N's worker produced the write, and N+1 snapshots a stale carrier. Capturing converts a race into a deterministic divergence -- better, but not correctness. The sufficient condition is that the draw MOVES no carrier, because then there is no ordering edge into any later draw at all. DrawSnapshot::carrierGrp fingerprints EACH GROUP at capture, carrierMask is set by re-hashing and comparing group by group at the derivation's exit, and the enumeration is stageDepCarrierGroups() -- the tree's existing reviewed carrier list, reused rather than re-derived, because a field missing from that list is invisible to the census AND to this gate. Conservative by construction: the baseline is taken a few stages EARLY, so a carrier moved by some other stage reports a false positive (a lost deferral opportunity), never a false negative -- measured empty, all 47 non-derivation stages read wrote=0%.\n"
               "PER-GROUP, NOT FOLDED. The groups are not the same KIND of dependency: kSdepCam is a real draw-N-writes / draw-N+1-reads edge, while kSdepCbLoc / kSdepBone are CACHES (a worker that misses one rescans -- a cost, never a wrong result) and kSdepStatic is a last-known-good fallback latch. safeToDefer() still requires ALL of them clean; the mask exists so the decision about a per-group gate is made from [DrawPure] eligCacheOk (draws that pass axis 1 and moved nothing but caches) instead of from an argument. eligCacheOk MINUS eligible is that gate's entire yield. Per-group comparison is also strictly safer than the fold it replaced: two different group vectors could fold to the same 64 bits and report clean.\n"
               "safeToDefer() IS THE GATE. cbSafeToDefer() && !wroteCarrier() -- two of the three axes. THE THIRD IS ABSENT ON PURPOSE: vertex / index / bone bytes go through BatchSkinJob and the geometry capture, which have their own copy-vs-pin rules, and nothing in the record speaks for them. A caller that hands a draw to a worker on safeToDefer() alone has proven two thirds of what it needs. Cross-check [DrawPure]'s carrierMoved% against [Perf.StageDep] bt_extractXf{wrote=} -- that probe samples 1 draw in 8 and hashes per stage, this counts every draw and hashes twice; they measure the same thing by different means, and a large disagreement means one is wrong, which is worth knowing BEFORE anything is deferred on it.\n"
               "cbSafeToDefer() IS ONE AXIS, contentPure() ALONE IS NOT EVEN THAT. The conjunction with layoutResolved is load-bearing and easy to miss: the SCANS do not go through drawCbSpan (they iterate offsets, so there is no span to name) and therefore never increment cbLiveReads -- a scanning draw would report contentPure() while reading live bytes throughout. layoutResolved is exactly what excludes it, because a draw that scans is a draw whose VS had no resolved location. Neither flag is sufficient and the pair is not redundant.\n"
               "READING [DrawPure]. draws / touched / pure / eligible / liveReads / recReads / satN / why{} / liveSlots{}. The denominator for pure% is TOUCHED, not draws: a replay-tier hit skips the derivation and reads nothing through the accessor, so it is excluded rather than counted as trivially pure. eligible is cbSafeToDefer() over ALL draws -- compare it against [DrawSnap]'s resolved% (~80%), which is its ceiling. satN>0 with pure% stuck means a shader reads more than 6 distinct spans -- raise kMaxCbManifest (and kMaxCbRanges with it), do NOT widen an individual span.\n"
               "why{} SPLITS THE RESIDUAL INTO THREE UNRELATED CAUSES, because a percentage cannot and they want opposite fixes. noSnap = no usable record at all (capture off, arena past cap, record overflowed) -- a capture-coverage problem the manifest cannot help. miss = record present, span not in it -- the ONLY one the manifest closes, and it should decay toward 0 within seconds; persistently non-zero with satN=0 means a span whose offset MOVES per draw, so it is learned and immediately stale: find the site, do not raise the cap. nonVs = the read is on GS/DS/PS and is NOT fixable by any manifest change, because the record holds VS bindings only -- this is a hard floor on pure%, and the number that says whether capturing other stages would pay. liveSlots{} names which cbuffer, zeroes omitted.\n"
               "FIRST MEASURED RUN, 05:28-05:32: pure=96% of touched steady, satN=0, liveReads/draw=0.09, [DrawSnap] meanRanges 2.13 -> 5.11 (the manifest capturing ~3 more spans/draw, EXPECTED -- not a regression), ovf=0, cbOff=0, [Perf.Replay] FAIL=0 throughout. pure% reached 100 within one 3 s window of a fresh scene and held; the residual appears in steady state, not at start-up, which is why it is attributed rather than waited out.\n"
               "WHAT IS STILL LIVE ON PURPOSE. The SCANS (offset-64 projection scan, cross-stage all-cb scan, the m_viewSlot hunt): they iterate offsets, so there is no span to name, and they are cold-start only -- see above. Non-VS-stage reads: the record holds VS bindings only, so a projection living in GS/DS/PS stays live and is COUNTED as a live read rather than quietly skipped, which is why such a draw can never read pure. Also the SE2 cb2@96 last-resort block (three spans off one shared pointer, and only reached when no projection resolved at all) and the camLocL VP+origin site (76 B, see above).\n"
               "THE CAPTURE'S COST IS WC TRANSACTION COUNT -- MEASURED 2026-08-10, and it is the only thing that made the record expensive. [Perf.WcCopy] billed the capture's narrow copies at 222,858 calls / 12.43 MB per 5 s window: 55.7 bytes each, ~2,894 separate write-combined reads per frame. WC is LATENCY-bound (this tree's own figure is 1192-2333 ns for 64 B, i.e. ~30-50 MB/s), so ~2,894 x ~1.5 us is ~4.3 ms/frame, which was essentially the whole 5.07 ms/frame the span copy measured. One read per SPAN was the defect; the manifest simply multiplied it.\n"
               "TWO FIXES, both structural rather than tuning. (1) A 256 B COALESCING WINDOW: on a WC miss the capture reads a bounded window at the span's 16-aligned offset and keeps it for the rest of that draw, so every later span on the same cbuffer is a cached memcpy and no transaction. TF2's spans cluster in the first ~160 B (cb2 at 4/16/96, cb3 at 0..48), so one window serves a whole buffer. This is NOT trap 3 returning: trap 3 forbids stagedCbBytes() here because its miss path stages the WHOLE ~6.3 KB buffer for one narrow read, and that reasoning was written when the capture took ONE 48 B span; the window is bounded at 256 B and is one transaction per BUFFER, not per span. Watch wcMiss / wcHit / coalesced% in [DrawSnap]; cross-check wcMiss/draw against [Perf.WcCopy]'s narrow-copy count. (2) ARENA SLOT REUSE: drawSnapAlloc_ns measured 1.44 ms/frame value-initialising a ~3 KB record per draw, every byte of which the identity block then overwrote (vertexBuffers, vsSrvs and vsCbs are assigned wholesale). The arena is now sized once and slots reused, clearing only the five scalars the capture does NOT unconditionally reassign. IF YOU ADD A CONDITIONALLY-WRITTEN FIELD TO DrawSnapshot, add it to that clear -- otherwise it carries a previous draw's value, which is the exact bug class the record exists to prevent.\n"
               "DO NOT JUDGE THE ARCHITECTURE ON A COST NUMBER TAKEN BEFORE THIS. The first measured run had drawSnap_ns at 89.5% of pfs_guard, 7.12 ms/frame -- and 11 us/draw to copy ~300 bytes was never CPU work. It was ~5 WC transactions per draw in a first draft with no optimisation pass and no deferral built, i.e. all cost and no benefit by construction. Re-measure after the two fixes above before drawing any conclusion about whether the record pays.\n"
               "FOUR SITES KEEP THEIR HAND-ROLLED find() ON PURPOSE, and it is trap 3, not an oversight. freshProj, the projOk memcmp and capViewHit are REPLAY-TIER sites: they run instead of the derivation, on a population deferral does not target. The cb3->objectToWorld site is on the full path, and all four probe the staging ring HIT-ONLY and fall to a narrow memcpyFromWC -- because cb3 is Map(WRITE_DISCARD)ed per draw, so its generation moves every draw and the ring can never hit for it. drawCbSpan's fallback calls the full stagedCbBytes(), which on a miss stages the WHOLE ~6.3 KB buffer to serve 48 bytes, the exact [Perf.WcCopy] shape a previous session removed from that very site. Routing them would reintroduce it. The cb3 site DOES feed the purity tally without being routed, because not counting it opened a FALSE PURE: a draw whose cb3 read fell back to live while its other reads came from the record would report cbSafeToDefer() while holding a pointer into live bytes.\n"
               "NEXT STEP: deferral. Purity is verifiable for free -- the replay tier's 16-replays-per-VS bit-compare must still report FAIL=0. Partition on safeToDefer() AND a per-draw statement about the third axis (vertex/index/bone bytes), which does not exist yet -- that is the last thing missing. Then defer via the existing GeometryProcessor pool, with results applied in DrawSnapshot::drawIndex order (that field exists for exactly this; it is UE's sort key). Size the win from `eligible` BEFORE building the parallel path. Convert BEFORE deferring, never after: moving work off-thread on an unproven purity claim is exactly how the two races above happened.")

    RTX_OPTION("rtx", bool, perfStageDepCensus, false,
               "[Perf.StageDep] -- which SubmitDraw stages WRITE cross-draw shared state, split by whether the draw replayed. DEFAULT OFF.\n"
               "THE QUESTION. SubmitDraw is serial because its stages mutate D3D11Rtx member state, so a later draw can depend on an earlier one. The replay tier already proved 92.9% of draws are independent of that state -- but only for ExtractTransforms, which is 6.01 of the 23.64 ms/frame the two draw entry points cost. Whether that partition GENERALISES to vsAnalysis / filters / cbc_rawUv / skyClassify / tail_capture / the hashes has never been measured, and both routes to a faster frame (the whole-draw memo and a parallel replayable/non-replayable split) depend on the answer. SubmitDraw is one 13,600-line function; guessing this wrong costs a session of refactoring in the wrong direction.\n"
               "HOW. Hash the cross-draw carrier state and re-hash at every stage boundary (inside markStg, so every stage is covered with no call-site changes). A stage whose hash moved WROTE shared state on that draw. Counted per stage, and split by whether the tier replayed the draw -- because that is exactly the population a parallel path or a memo would serve.\n"
               "READING IT. wroteEligPct=0 means that stage never writes shared state on the replayable population: parallelisable for those draws, and memoisable with no state-restore step. wroteEligPct>0 means it does, so it either stays on the serial path or its writes need enumerating the way XtReplayRec enumerates ExtractTransforms'. reachPct says how many sampled draws got that far, so a stage behind an early return cannot read as falsely clean.\n"
               "IT IS A DETECTOR, NOT A PROOF. It sees writes that CHANGE a value; a stage that rewrites a carrier with the value it already held reads clean -- which is the correct answer for parallelism (no observable dependency) though not for a 'does it touch this memory' audit. m_lastGoodTransforms is read without its mutex: a torn read can only manufacture a FALSE 'wrote', never a false 'clean', which is the safe direction for this decision.\n"
               "SCOPE -- READ THIS BEFORE ACTING ON A ZERO. It watches D3D11Rtx MEMBER state and nothing else. A stage that creates an ordering dependency through state OUTSIDE this class -- the scene manager, the geometry/bone buffers, the command list -- reads wroteElig=0 here and is still not safe to reorder. tail_emit, bt_geoCopy and bonePalette are exactly that shape: their zero means 'adds no dependency through D3D11Rtx', NOT 'parallelisable'. Treat a zero as clearing the FIRST of two hurdles; the pure-derivation stages (vsAnalysis, bt_cullVtx, bt_hashes, cbc_rawUv, skyClassify, cvr_*, tail_capture, te_*) are the ones for which it is the only hurdle.\n"
               "COST. One ~30-field hash per stage boundary on sampled draws only (1 in 8), plus one predictable branch per markStg when off. Turn off before taking any frame-time number.")

    RTX_OPTION("rtx", bool, perfUpdateInstSplit, false,
               "[Perf.UpdInst] -- split InstanceManager::updateInstance into entry / surf / xform / flags / viewmodel / billboard / census / anticull / tail. DEFAULT OFF.\n"
               "WHY. This is the deepest leaf of the measured chain and the single biggest item left in the frame: [Perf.SubmitState] process=96% -> [ProcDCS] instMs 32-33 at 28-29 us/draw -> [Perf.SceneObj] update=62%, i.e. ~20 ms/frame across ~16,100 instances. HANDOFF_PERF_2026-08-06_v4 sec 4c is explicit that it must be SPLIT before anything here is optimised, because the last time this function was reasoned about without a split the answer was wrong.\n"
               "SAMPLED DURATIONS, EXACT COUNTS -- the [Perf.CsCmd] method, and it is not optional here. updateInstance runs ~16,100 times per frame, so timestamping all nine stages on every instance would cost ~6.6 ms/frame: a third of the function, which is precisely the rtx.perfGapSampler failure (v4 sec 0c) and the 11x mis-sizing of rtx.perfSceneObjSplit. Durations are therefore taken on 1 instance in 64 (~252/frame, ~0.09 ms) while the counters below are exact on every instance. estMsPerFrame is mean x true count, so a rare-but-expensive stage cannot hide behind sampling probability.\n"
               "THE COUNTERS ARE THE POINT, not the timings. first/xfChg/matChg/prevPos/static are read from values the function ALREADY computes, and REDUNDANT = !xfChg && !matChg && !prevPos is the answer to sec 4c's open question: why does a scene with addedPct=0 and bBuild=3 re-update 16,000 instances every frame? A high REDUNDANT% means most of this work is rebuilding state that did not change, and skipping or incrementalising it is worth far more than micro-optimising the stages.\n"
               "COST ~0.2 ms/frame total. Turn it off before taking any frame-time number.");

    RTX_OPTION("rtx", bool, replayExtractTransforms, true,
               "[Perf.Replay] Stage-1 of the cross-frame replay tier (HANDOFF_PERF_2026-08-08e sec 2): intra-frame route replay for ExtractTransforms. F7 toggles live for A/B.\n"
               "v6.2 overhead fix: the first warm F7 A/B measured v6.1 at NET +2-2.5 ms/frame -- bit-correct but the per-draw two-pass linear scan over fat records plus the per-frame re-capture of never-hitting streams cost more than the ~390 replayed draws saved. v6.2 adds MRU record probes (scan only on stream transitions) and capture dormancy (8 no-hit captures -> bookkeeping touch instead of a ~1 KB rebuild). RE-MEASURE with F7 at a warm heavy spot; if ON still reads slower than OFF, that number decides the default, not this comment.\n"
               "MECHANISM. The first draw of each (VS, input-layout, cb-binding-shape, cb2-content-generation) combination each frame runs the FULL discovery path and snapshots its outputs (all camera-side matrices + every member the function writes) plus its resolved routes. Subsequent draws with an IDENTICAL key skip the whole discovery tangle and re-execute ONLY the per-draw content read: cb3 -> objectToWorld via the recorded route (path 13 camera-relative or identity). cb2 is DISCARD-mapped once per camera per frame, so its content generation in the key proves the camera-side inputs are byte-identical to the record -- the reuse is exact by construction, not heuristic.\n"
               "POPULATION (v6.9 -- the pre-v6.3 text here described o2w route 13 or 0/identity only and was years of work out of date). o2w routes 1/2/3/4/5/6/13 all replay; the record supplies the ROUTE and every value is re-read live, so admitting a route costs only the branch that re-reads it. Still excluded: fallback/UI draws, unsettled axis votes, and fanout draws whose per-draw state the record cannot prove. Measured population at a warm heavy spot: ~81% of draws hit, ~17/frame ineligible.\n"
               "ROUTE PROOF (v6.9). A record stores a route, so replaying it is only sound while the inputs that CHOSE that route still hold. Three are re-checked per attempt, because none of them is provable from the cb2 witness the value tier uses: (a) per-VS location state -- VsCbLocations::gen plus the neg-cache TTL flag, two integer compares; (b) cb2 content -- classifyPerspective on cb2@16/@96, which is what picks 'projection resolved' (wtvPath 1) over 'use the cached camera' (wtvPath 3); (c) the view-matrix slot's content -- isViewMatrix at viewSlot/viewOffset, which picks wtvPath 5 over 1. A mismatch counts routeMiss and takes the full path, which re-records. Expect routeMiss ~1-4/frame; a large routeMiss with a falling hit rate means gen is churning, not that the tier is broken.\n"
               "Per-VS quarantine: any draw that fails the sampled verify (see rtx.replayExtractVerify) disqualifies its VS for the session. Tried per (VS,key) in v6.8 and REVERTED -- quarantined draws went 81 -> 189/frame and records 1716 -> 3620, because the failing population was never one bad route per shader.\n"
               "REGRESSION LINES. [Perf.Replay] every 300 frames: hit/full/keyMiss/inelig/quar/verifyFail/o2wReadFail/routeMiss. verifyFail MUST stay 0; bt_extractXf on [Perf.SubmitDraw.acc] is the payoff metric. Zero visual change is the contract -- any mismatch quarantines rather than renders.\n"
               "STILL UNMEASURED, and it is the only question that matters: whether the tier pays at all. The last clean same-session F7 A/B read +1.60 ms with it ON, at a 28.9% hit rate. Nothing has re-measured it since the rate reached ~81%. The tax (~7.6 us key+lookup) falls on EVERY draw while the payout falls only on hits, and the full path already memoises the same route searches -- so a net-negative result is a live possibility, not a formality.");
    RTX_OPTION("rtx", bool, replayExtractVerify, true,
               "Sampled self-check for rtx.replayExtractTransforms. DEFAULT ON.\n"
               "On the first 16 replays per VS and 1-in-256 thereafter, the draw runs the FULL path anyway and the replay result is compared bit-exactly (matrices memcmp + every routed flag). A mismatch quarantines the VS from replay for the session and bumps [Perf.Replay] verifyFail. The verified draw uses the FULL path's result, so a broken replay can only ever cost the time it saved, never correctness. Cost: ~2-6% of the replay savings while warm.");
    RTX_OPTION("rtx", bool, replayExtractVerifyAll, false,
               "[Perf.Replay] CENSUS MODE for rtx.replayExtractVerify. DEFAULT OFF -- this is a diagnostic run, not a setting.\n"
               "WHY IT EXISTS. Normal verify checks the first 16 replays per record and 1-in-256 after, which at a warm heavy spot is ~0.4% of all replays ([Perf.Replay] verifyCov reports the exact fraction). So verifyFail=0 has only ever meant 'none in the sampled subset'. It cannot tell a genuinely clean tier apart from a divergence rare enough that sampling keeps missing it, and every route bug fixed in v6.6-v6.9 was found by whichever failure sampling happened to catch -- never by looking for them.\n"
               "WHAT IT DOES. Forces EVERY eligible replay to also run the full path and bit-compare. verifyFail then counts the true divergence population rather than an estimate of it, and [Perf.Replay.route] names the shaders. Pair it with a fixed warm camera position so two runs are comparable.\n"
               "COST. Roughly the entire saving of the tier, since every replayed draw also runs the discovery path. Frame time will get worse while it is on -- that is expected and is not a regression. Run it 60s, read verifyFail and quarSize, turn it off. Never leave it on for an F7 A/B: it would make the ON side pay for both paths and the measurement meaningless.");
    RTX_OPTION("rtx", bool, perfMaterialSplit, false,
               "[Perf.MatData] -- split SceneManager::determineMaterialData into repl / portal / convert. DEFAULT OFF.\n"
               "Measured at 8us/draw = ~9 ms/frame = 12% of the frame ([Perf.SubmitState] material stage). The function is only ~45 lines, so 8us is suspicious on its face.\n"
               "convert = input.getMaterialData().as<OpaqueMaterialData>(). MaterialData is RETURNED BY VALUE from determineMaterialData and carries 18 TextureRefs plus dozens of scalars, so if convert dominates the cost is per-draw object construction and copying, not lookup, and the fix is to stop materialising a fresh one every draw. repl/portal are hash lookups.\n"
               "The exit* counters report which return path was actually taken, so a stage that looks cheap because it rarely runs is not mistaken for one that is fast.");

    RTX_OPTION("rtx", bool, cacheLegacyMaterialConversion, true,
               "Memoize LegacyMaterialData::as<OpaqueMaterialData>() for the duration of a frame. DEFAULT ON; set False to A/B against the uncached path without a rebuild.\n"
               "WHY. The conversion runs once per no-replacement draw ([Perf.MatData] exit convert=100% on TF2) and builds a fresh OpaqueMaterialData -- 18 TextureRefs plus 42 constants -- every time, even though a scene draws far fewer distinct materials than draw calls. Cached on rtx.cacheLegacyMaterialConversionCounts you can see the ratio directly.\n"
               "KEY. LegacyMaterialData::getOpaqueConversionKey(), a digest of every field the conversion reads. Deliberately NOT getHash(): the material hash covers 11 texture hashes plus blend and alpha-test state only, while the conversion also reads the source* emissive/fog/premultiplied markers, the screen-space emissive params and the sampler, so getHash() would collide two draws that need different materials.\n"
               "LIFETIME. Cleared every frame, alongside m_preCreationSurfaceMaterialMap and for the same reason: the conversion also reads rtx.legacyMaterial.* and rtx.ignoreAlphaOnTextures, which the user can change at any time, and texture indices move. A frame is short enough that no option change survives it.");

    RTX_OPTION("rtx", bool, cacheLegacyMaterialConversionCounts, false,
               "[Perf.MatCache] -- report hit/miss/unique-material counts for rtx.cacheLegacyMaterialConversion every 3s. DEFAULT OFF.\n"
               "hitPct is the whole story: it equals 1 - (distinct materials / draw calls), so a low number means the scene genuinely draws that many distinct materials and the cache cannot help, not that the cache is broken. Costs two counter increments per draw.");

    RTX_OPTION("rtx", bool, perfSubmitStateSplit, false,
               "[Perf.SubmitState] -- split SceneManager::submitDrawState into entry / hash / material / process. DEFAULT OFF; turn it on for a capture and back off afterwards.\n"
               "WHY HERE. The chain is measured and every link agrees: [Perf.Busy] frame 73.6 ms -> [Perf.CsSplit] dxvk-cs exec 73.3 ms (99.6%, so dxvk-cs IS the frame) -> [Perf.CsCmd] RtxContext::commitGeometryToRT 53.1 ms/frame across 1102 calls at 48.2us -> [CommitRT] submitMs 50 of perFrameMs 52 with finalizeMs 0. So ~96% of the biggest item on the critical thread is this one function, and finalizeMs=0 means it is computing rather than blocked on worker futures.\n"
               "WHAT THE STAGES DECIDE. process = drawReplacements/processDrawCallState, the real instance + BLAS-input build; if it dominates, the cost is genuine scene work and the lever is how much geometry reaches here. entry = the category/fog/transform bookkeeping and the per-draw diagnostics this function has accreted; hash = getHash + trackMeshHash + getReplacementsForMesh, plus two LegacyAssetHash retries that repeat all three on a miss. If either of those dominates, the cost is overhead and can be cut directly.\n"
               "COST. 4 clock reads per draw (~205 ns), ~0.23 ms/frame at 1102 draws/frame, under 0.4%. Uses a destructor guard so the buffer-cache-overflow early return still counts as a draw with only entry charged -- otherwise those draws vanish and the per-draw average reads better than it is.");

    RTX_OPTION("rtx", bool, perfCsCmdProbe, false,
               "[Perf.CsCmd] -- attribute dxvk-cs execution time to the EmitCs call site that produced each command. DEFAULT OFF; turn it on for a capture and back off afterwards.\n"
               "WHAT IT ANSWERS. dxvk-cs is the frame (execMs/frame 73.3 of a 73.6 ms frame, busyPct 99.7). One fat chunk per frame is 22.4 ms and is already split by [Perf.PrepScene] (merge 11.0, accelLight 4.7, gc 2.2, surfMat 1.2). The other ~50 ms, spread over ~1370 small chunks, has never been attributed -- and it is the larger half.\n"
               "WHY THE EXISTING ANSWER DOES NOT COUNT. HANDOFF_PERF_2026-08-06_v3 sec 3b called that 50 ms flat with no hot function. It was measured with rtx.perfGapSampler aimed at dxvk-cs, and that sampler was eating ~16 ms/frame of the very thread it profiled -- the <10ms CsSplit bucket fell 16.2 -> 0.64 ms/frame when it was switched off. A profiler that suspends its target 500x/s and consumes 22% of it manufactures a flat histogram. Do not carry that verdict forward.\n"
               "HOW IT DIFFERS FROM A SAMPLING PROFILER. Counts are EXACT (one increment per command, keyed on the command's vtable pointer, which is unique per EmitCs lambda and already in cache). Only the DURATION is sampled, 1-in-64 per type, and the estimate is count x mean -- so a hot site cannot hide behind sampling probability and a once-per-frame site is still reported with its true count.\n"
               "COST. ~103,000 commands/frame. Timestamping all of them would be ~8.4 ms/frame, the same mistake the gap sampler made; at 1-in-64 it is ~0.13 ms/frame, under 0.2%. Off, it is one bool load and a predicted branch per command.");

    // NV-DXVK [Perf.ChunkTrim] (2026-08-06): automatic release of EMPTY DXVK
    // allocator chunks. See SceneManager::manageTextureVram.
    RTX_OPTION("rtx", bool, autoFreeUnusedChunks, true,
               "Automatically release DXVK allocator chunks that hold no live suballocations, instead of only ever doing so when the UI's force-free button is pressed.\n"
               "WHY THIS EXISTS. DxvkMemoryAllocator is a high-water-mark allocator: it grows chunks on demand and only gives one back under two conditions, neither of which fires in normal play.\n"
               "  1. Allocation pressure. tryAllocFromType calls freeEmptyChunks, but only behind shouldFreeEmptyChunks (dxvk_memory.cpp:857), which is 'totalAllocated + thisAllocation > budget'. That is a LAST-DITCH reclaim at the edge of the heap budget -- by the time it fires you are already in the degraded regime it was supposed to prevent. It never trims a merely wasteful footprint.\n"
               "  2. An explicit user action. freeUnusedChunks is otherwise reached only through SceneManager::manageTextureVram's two m_forceFree* atomics, both set from the Remix UI.\n"
               "So on a session where nobody presses the button and the heap never quite saturates, the high-water mark is never given back. That is the mechanism behind 'VRAM climbs and never comes back down'.\n"
               "MEASURED 2026-08-06 on the BT mission, steady state, [Perf.MemCat]: heap0(VRAM) alloc=7569MB used=5313MB slack=2255MB -- 30% of every byte taken from the driver was sitting in chunks with nothing suballocated in them. That was the single largest line item in an 8 GB footprint, larger than the game's textures (appTex=2329MB) and larger than the acceleration structures (rtxAccel=2229MB).\n"
               "This does NOT add a new call site or a new failure mode: it sets the SAME freeUnused flag the force path sets, at the SAME point in the frame (SceneManager::onFrameEnd), so everything that was already true of the manual trim is still true of this one. It only changes WHEN it fires.\n"
               "Set false to restore stock behaviour (trim only on explicit request).");
    RTX_OPTION("rtx", uint32_t, autoFreeUnusedChunksSlackMB, 512,
               "Slack (allocated minus suballocated, per heap) at or above which rtx.autoFreeUnusedChunks trims, in MB. The largest single heap's slack is what is tested, not the sum.\n"
               "Raise it to trim less often and tolerate more headroom; lower it to keep the footprint tighter at the cost of more frequent trims. Note that empty chunks are the ONLY thing recoverable -- slack sitting inside partially-used chunks is fragmentation and no threshold reaches it, which is why a trim can legitimately recover far less than the slack figure that triggered it.");
    RTX_OPTION("rtx", uint32_t, autoFreeUnusedChunksMaxPerTrim, 1,
               "Maximum allocator chunks released per automatic trim. THIS IS THE KNOB THAT KEEPS THE TRIM INVISIBLE, and it is why the automatic path cannot hitch the way an unbounded free can.\n"
               "A chunk is 320 MB device-local / 128 MB otherwise (dxvk_options.cpp), and releasing one is a vkFreeMemory of that whole allocation. Freeing every empty chunk at once is therefore a multi-hundred-MB-to-multi-GB burst of driver work on a single frame -- acceptable at a load boundary, a visible stutter mid-gameplay. At 1 chunk per trim the cost per frame is one vkFreeMemory and the footprint still drains completely, just over several seconds instead of instantly: the measured 2255 MB of VRAM slack is about 7 chunks, so it clears in ~7 trims.\n"
               "The manual force-free path is unaffected and stays unbounded -- it is an explicit user action at a moment of their choosing, which is exactly when paying the whole cost at once is correct.\n"
               "Raise it to drain faster if you can absorb the cost; there is no reason to lower it below 1.");
    RTX_OPTION("rtx", uint32_t, autoFreeUnusedChunksRetryDeltaMB, 128,
               "How much slack must GROW, in MB, before retrying after an automatic trim that freed nothing.\n"
               "Slack over the threshold does not imply anything is recoverable: only entirely-empty chunks can be released, and slack inside partially-used chunks is fragmentation no trim reaches. MEASURED 2026-08-06: 7 productive trims recovered 1360 MB, then 50 consecutive trims freed 0 chunks each while slack sat at ~1900 MB -- fifty heap walks taking every per-memory-type lock to accomplish nothing.\n"
               "Growth is the exact retry condition, not a heuristic: a chunk can only become newly empty if allocated rose (new chunks) or used fell (suballocations released), and slack = allocated - used rises in both cases. Falling slack means more allocation, which cannot empty a chunk. So if slack has not grown since a null trim, there is provably nothing new to find.\n"
               "Default is one 128 MB non-device-local chunk. The explicit force-free path ignores this entirely -- a user request is never suppressed by a previous null result.");
    RTX_OPTION("rtx", uint32_t, autoFreeUnusedChunksCooldownMs, 1000,
               "Minimum milliseconds between automatic trims. Paired with rtx.autoFreeUnusedChunksMaxPerTrim, this sets the drain RATE: at the defaults, one 320 MB chunk per second.\n"
               "It also stops the policy thrashing. Once the empty chunks are gone the slack that remains is fragmentation inside partially-used chunks, which no trim can reach, so re-testing every frame would walk the heaps forever to recover nothing.\n"
               "Every trim logs [Perf.ChunkTrim] with chunks= and recovered= -- tune against that line, not against a guess.");

    RTX_OPTION_ENV("rtx", bool, enableAlwaysCalculateAABB, false, "RTX_ALWAYS_CALCULATE_AABB", "Calculate an Axis Aligned Bounding Box for every draw call.\n This may improve instance tracking across frames for skinned and vertex shaded calls.");

    // Camera
    struct FreeCam{
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveFaster,  {VirtualKey{VK_LSHIFT}}, "Move faster in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveForward = RSHIFT'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveForward, {VirtualKey{'W'}}, "Move forward in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveForward = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveLeft,    {VirtualKey{'A'}}, "Move left in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveLeft = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveBack,    {VirtualKey{'S'}}, "Move back in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveBack = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveRight,   {VirtualKey{'D'}}, "Move right in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveRight = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveUp,      {VirtualKey{'E'}}, "Move up in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveUp = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyMoveDown,    {VirtualKey{'Q'}}, "Move down in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyMoveDown = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyPitchDown,   {VirtualKey{'I'}}, "Pitch down in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyPitchDown = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyPitchUp,     {VirtualKey{'K'}}, "Pitch up in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyPitchUp = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyYawLeft,     {VirtualKey{'J'}}, "Yaw left in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyYawLeft = P'");
      RTX_OPTION("rtx.freeCam", VirtualKeys, keyYawRight,    {VirtualKey{'L'}}, "Yaw right in free camera mode.\nExample override: 'rtx.rtx.freeCam.keyYawRight = P'");
    } freeCam;
    RTX_OPTION_ENV("rtx", bool, shakeCamera, false, "RTX_FREE_CAMERA_ENABLE_ANIMATION", "Enables animation of the free camera.");
    RTX_OPTION_ENV("rtx", CameraAnimationMode, cameraAnimationMode, CameraAnimationMode::CameraShake_Pitch, "RTX_FREE_CAMERA_ANIMATION_MODE", "Free camera's animation mode.");
    RTX_OPTION_ENV("rtx", int, cameraShakePeriod, 20, "RTX_FREE_CAMERA_ANIMATION_PERIOD", "Period of the free camera's animation.");
    RTX_OPTION_ENV("rtx", float, cameraAnimationAmplitude, 2.0f, "RTX_FREE_CAMERA_ANIMATION_AMPLITUDE", "Amplitude of the free camera's animation.");
    RTX_OPTION("rtx", bool, skipObjectsWithUnknownCamera, false, "");
    RTX_OPTION("rtx", bool, enableNearPlaneOverride, false,
               "A flag to enable or disable the Camera's near plane override feature.\n"
               "Since the camera is not used directly for ray tracing the near plane the application uses typically does not matter, but for certain matrix-based operations (such as temporal reprojection or voxel grid projection) it is still relevant.\n"
               "The issue arises when geometry is ray traced that is behind where the chosen Camera's near plane is located, typically common on viewmodels especially with how they are ray traced, causing graphical artifacts and other issues.\n"
               "This option helps correct this issue by overriding the near plane value to else (usually smaller) to sit behind the objects in question (such as the view model). As such this option should usually be enabled on games with viewmodels.\n"
               "Do note that when adjusting the near plane the larger the relative magnitude gap between the near and far plane the worse the precision of matrix operations will be, so the near plane should be set as high as possible even when overriding.");
    RTX_OPTION("rtx", float, nearPlaneOverride, 0.1f,
               "The near plane value to use for the Camera when the near plane override is enabled.\n"
               "Only takes effect when rtx.enableNearPlaneOverride is enabled, see that option for more information about why this is useful.");

    RTX_OPTION("rtx", bool, useRayPortalVirtualInstanceMatching, true, "");
    RTX_OPTION("rtx", bool, enablePortalFadeInEffect, false, "");

    RTX_OPTION_ENV("rtx", bool, useRTXDI, true, "DXVK_USE_RTXDI",
                   "A flag indicating if RTXDI should be used, true enables RTXDI, false disables it and falls back on simpler light sampling methods.\n"
                   "RTXDI provides improved direct light sampling quality over traditional methods and should generally be enabled for improved direct lighting quality at the cost of some performance.");
    RTX_OPTION_ARGS("rtx", IntegrateIndirectMode, integrateIndirectMode, IntegrateIndirectMode::NeuralRadianceCache,
                   "Indirect integration mode:\n"
                   "0: Importance Sampled. Importance sampled mode uses typical GI sampling and it is not recommended for general use as it provides the noisiest output.\n"
                   "   It serves as a reference integration mode for validation of other indirect integration modes.\n"
                   "1: ReSTIR GI. ReSTIR GI provides improved indirect path sampling over \"Importance Sampled\" mode \n"
                   "   with better indirect diffuse and specular GI quality at increased performance cost.\n"
                   "2: RTX Neural Radiance Cache (NRC). NRC is an AI based world space radiance cache. It is live trained by the path tracer\n"
                   "   and allows paths to terminate early by looking up the cached value and saving performance.\n"
                   "   NRC supports infinite bounces and often provides results closer to that of reference than ReSTIR GI\n"
                   "   while improving performance in scenarios where ray paths have 2 or more bounces on average.\n",
                   args.environment = "RTX_INTEGRATE_INDIRECT_MODE",
                   args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", UpscalerType, upscalerType, UpscalerType::DLSS, "Upscaling boosts performance with varying degrees of image quality tradeoff depending on the type of upscaler and the quality mode/preset.",
                    args.environment = "DXVK_UPSCALER_TYPE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableRayReconstruction, true, "Enables DLSS ray reconstruction, an AI-based denoiser designed for real time ray tracing.",
                    args.environment = "DXVK_RAY_RECONSTRUCTION",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx", float, resolutionScale, 0.75f, "",
                    args.flags = RtxOptionFlags::UserSetting);
    // NV-DXVK [perf]: DIAGNOSTIC — see the census site in rtx_accel_manager.cpp.
    // Masks off instances whose world-space AABB diagonal exceeds this, to test
    // whether TLAS fan-out is what costs the primary-ray pass ~140 ms. Geometry
    // visibly disappears while set; 0 disables.
    // NV-DXVK [perf]: acceleration-structure build quality. See
    // accelerationStructureBuildQualityFlags() in rtx_accel_manager.cpp for why
    // these exist and what the measurements behind them were. Both default to
    // the pre-existing behaviour; switching either at runtime forces a rebuild.
    RTX_OPTION("rtx", bool, asPreferFastTrace, false,
               "Build acceleration structures with PREFER_FAST_TRACE instead of "
               "PREFER_FAST_BUILD. Slower to build, substantially faster to trace. "
               "Correct trade for a path tracer, which builds once and casts "
               "millions of rays; costs more in prepScene.");
    RTX_OPTION("rtx", bool, asDisableUpdates, false,
               "Drop ALLOW_UPDATE so acceleration structures are always fully "
               "rebuilt rather than refitted. Refits keep the topology from the "
               "original build while geometry moves, so traversal quality decays "
               "the longer a BLAS goes without a rebuild.");
    RTX_OPTION("rtx", float, perfCullInstancesLargerThan, 0.0f,
               "DIAGNOSTIC: mask off instances whose world AABB diagonal exceeds "
               "this value, so they are invisible to rays. 0 = disabled. Used to "
               "measure how much of the primary-ray cost is TLAS fan-out over "
               "map-spanning instances. Makes geometry disappear.");
    RTX_OPTION("rtx", bool, mergePersistentBuckets, true,
               "PERF ([Perf.MergeP]): persist the merged-BLAS buckets across frames. "
               "When this frame's merged-instance sequence (pointers + per-instance "
               "fingerprints of every bucket-shaping input) is identical to the one "
               "the buckets were built from, the per-frame re-derivation (geometry "
               "fill, bucket append, transform-address rewrite) is skipped for the "
               "whole merged population; only transform contents and resource "
               "tracking refresh. Any change - membership, order, geometry address, "
               "mask/flags, routing options, transform-buffer realloc - rebuilds "
               "everything exactly like the legacy path. Kill switch: set False.");
    RTX_OPTION("rtx", bool, mergePersistentBucketsVerify, true,
               "SAFETY for rtx.mergePersistentBuckets: every 64th reuse frame, run "
               "the legacy bucket derivation anyway and bit-compare it against the "
               "persistent buckets (geometries/ranges/instances/offsets/compat "
               "fields). A mismatch uses the fresh result, disables persistence for "
               "the session, and logs [Perf.MergeP] VERIFY-FAIL. Near-free (fires "
               "on 1/64 of reuse frames).");
    // NV-DXVK [perf]: the primary ray is cast with RAY_FLAG_FORCE_OPAQUE (see
    // geometryResolver in geometry_resolver.slangh), so a non-opaque instance
    // costs nothing during traversal - it costs a whole EXTRA TraceRayInline.
    // Any hit whose material resolves below resolveOpaquenessThreshold sets
    // continueResolving, and RESOLVE_RAY_QUERY re-traces the full TLAS from the
    // hit point, up to primaryRayMaxInteractions (32) times, each iteration also
    // running a second unordered-TLAS traversal (up to 128 steps). So per-pixel
    // cost scales with how many non-opaque surfaces a ray passes through, which
    // is a content property, not a shading option - which is why it stayed
    // invisible to every rtx.conf knob tried so far. This census reports how much
    // of the scene is non-opaque and which routing branch put it there.
    RTX_OPTION("rtx", bool, perfNonOpaqueCensus, false,
               "DIAGNOSTIC: log [Perf.NonOpaque] once per second, a per-branch "
               "census of how instances were classified opaque vs non-opaque, "
               "since only non-opaque hits drive the primary-ray resolve loop.\n"
               "DEFAULT FLIPPED true -> false 2026-08-06. IT REPORTS once per "
               "second but it ACCUMULATES PER INSTANCE, and it does so inside "
               "InstanceManager::updateInstance -- the ~20 ms/frame function that "
               "HANDOFF_PERF_2026-08-06_v4 sec 4c names as the single biggest item "
               "in the frame. Per qualifying instance it takes a std::mutex and "
               "does an unordered_map<vsHash> lookup; [Perf.NonOpaque] itself "
               "reports instPerFrame=8853-9304, so that is ~8,900 lock/unlock "
               "pairs and ~8,900 map probes every frame.\n"
               "TURN IT OFF BEFORE MEASURING 4c. Left on, it is counted inside the "
               "[Perf.SceneObj] `update` bucket, so it inflates the very number 4c "
               "exists to attribute. The option test is the first term of the "
               "gate, so off it costs one bool load per instance and nothing "
               "else.");
    // NV-DXVK [perf]: geometryData.boundingBox is computed over the whole vertex
    // RANGE (for vi in 0..vertCount), not over the vertices the draw's index
    // buffer actually references. Several draws sharing one vertex buffer
    // therefore all report the same box - which is why four TF2 ship-hull BLASes
    // report an identical diag=42945.3 at different primitive counts. That box is
    // what Anti-Culling consumes, and it is what the earlier TLAS fan-out test
    // measured, so:
    //   1. the fan-out refutation was computed on the wrong box, and
    //   2. anti-culling is very likely mis-culling as a standing defect.
    // This probe computes both boxes side by side over identical bytes so the
    // error is quantified rather than argued about. Opt-in: it scans the full
    // index buffer, so it is one-shot per (vs, vertCount, indexCount) rather
    // than per draw, and off by default so it cannot perturb a timing run.
    RTX_OPTION("rtx", bool, perfBlasBoundsProbe, false,
               "DIAGNOSTIC: log [Perf.BlasBounds] once per unique (vertex shader, "
               "vertex count, index count), comparing the whole-vertex-range "
               "bounding box against one restricted to indexed vertices.");
    // NV-DXVK [perf]: see the perfGbStopAfter comment in raytrace_args.h for the
    // ladder itself. Sweep 1..4 and diff gb_primaryRays in [Perf.GpuPass]; the
    // rendered image is meaningless at any nonzero value.
    RTX_OPTION("rtx", uint32_t, perfGbStopAfter, 0,
               "DIAGNOSTIC: truncate the primary-ray shader after a given stage to "
               "attribute gb_primaryRays. 0=full, 1=raygen only, 2=+traversal, "
               "3=+unordered resolve, 4=+material. Produces a garbage image.");
    // NV-DXVK [Perf.SubLadder]: see the perfUnorderedStopAfter comment in
    // raytrace_args.h. Sweep 1..6 and diff gb_primaryRays in [Perf.GpuPass].
    // The stage being subdivided is 42.5 ms of the ~126 ms pass.
    RTX_OPTION("rtx", uint32_t, perfUnorderedStopAfter, 0,
               "DIAGNOSTIC: truncate resolveVertexUnordered's candidate loop after a "
               "given step to attribute the 42.5 ms unordered stage. 0=full, "
               "1=traversal only, 2=+surface interaction, 3=+clip test, "
               "4=+material interaction, 5=+approximations/decals, 6=+blend. "
               "Traversal runs in every rung. Produces a garbage image.");
    RTX_OPTION("rtx", bool, perfSkipPom, false,
               "DIAGNOSTIC: skip the POM raymarch in opaque material evaluation. "
               "Timing probe, produces a wrong image.");
    RTX_OPTION("rtx", bool, perfSkipMaterialTextures, false,
               "DIAGNOSTIC: skip every opaque material texture read and use material "
               "constants instead. Timing probe, produces a wrong image.");
    RTX_OPTION("rtx", bool, perfSkipThinFilm, false,
               "DIAGNOSTIC: force the thin film layer off in opaque material "
               "evaluation. Timing probe, produces a wrong image.");
    // NV-DXVK [Perf.GeomFetch]: cumulative ablation of the vertex buffer reads in
    // surfaceInteractionCreate - the last per-ray axis no probe has cut. See the
    // rung table on perfSkipGeometryFetch in raytrace_args.h. Control flow is
    // preserved at every rung, so consecutive differences are attributable.
    // NV-DXVK [Perf.ShaderClock]: in-shader cycle counters. One run gives every
    // region's mean cost, replacing the one-launch-per-rung ladders. Requires
    // VK_KHR_shader_clock and the shader built with REMIX_SHADER_CLOCK_AVAILABLE;
    // without either it is inert.
    RTX_OPTION("rtx", bool, perfShaderClock, false,
               "DIAGNOSTIC: accumulate in-shader cycle counts per region and log "
               "[Perf.ShaderClock]. Instrumented runs are slower than real ones - "
               "read proportions between regions, not absolute timings.");
    // Default 5, not 120: this game runs at 2-4 fps, so a 120-frame interval is
    // 30-60 seconds between lines. Sample count is not the constraint - one frame
    // is already ~2.5M candidate executions - so a short interval costs nothing in
    // statistical quality and makes the counters usable interactively.
    RTX_OPTION("rtx", uint32_t, perfShaderClockLogInterval, 5,
               "DIAGNOSTIC: frames between [Perf.ShaderClock] lines. Counters are "
               "zeroed after each log so each line is that interval's sample.");
    // NV-DXVK [Perf.SICut]: bisects surfaceInteractionCreate, the 7.96 ms function.
    // Stack with perfGbStopAfter=3 and perfUnorderedStopAfter=2 to isolate it.
    RTX_OPTION("rtx", uint32_t, perfSurfaceInteractionStopAfter, 0,
               "DIAGNOSTIC: cut surfaceInteractionCreate short. Cumulative: "
               "1=positions, 2=+triangle normal, 3=+interpolated normal, "
               "4=+motion, 5=+texcoords, >5=full. Timing probe, wrong image.");
    RTX_OPTION("rtx", uint32_t, perfSkipGeometryFetch, 0,
               "DIAGNOSTIC: skip vertex buffer reads in surface interaction "
               "construction. Cumulative: 1=colour, 2=+normal, 3=+texcoord, "
               "4=+position (indices then die by DCE). Timing probe, produces a "
               "wrong image.");
    // NV-DXVK [Perf.UnorderedSteps]: raw counts first. Whether the unordered
    // stage is 42.5 ms because of many candidates or expensive candidates is
    // not decidable from the cut ladder alone.
    // NV-DXVK [Perf.MatLadder]: see raytrace_args.h. Bisects the 1530-line
    // opaqueSurfaceMaterialInteractionCreate, which the sweep localised as
    // ~98 ms of the ~126 ms pass. Sweep steps mat_stop1..4 drive this.
    RTX_OPTION("rtx", uint32_t, perfMaterialStopAfter, 0,
               "DIAGNOSTIC: return early from opaqueSurfaceMaterialInteractionCreate "
               "after a given block. 0=full, 1=texture reads, 2=+emissive/modifiers, "
               "3=+normal detail, 4=+opacity/albedo/roughness. Garbage image.");
    RTX_OPTION("rtx", bool, perfUnorderedStepCensus, false,
               "DIAGNOSTIC: log [Perf.UnorderedSteps] - mean rayQuery candidates "
               "stepped and interactions accepted per pixel in the unordered "
               "resolve. Adds three InterlockedAdds per pixel.");
    // NV-DXVK [ResolveCensus]: the instrument for HANDOFF_PI_FLICKER_V4 §7 -
    // "why does Class-B geometry never write m_sharedSurfaceIndex despite being
    // in the TLAS with a correct mask". [HitCensus] can only read the winner
    // after the fact; this counts, per surface, whether the ordered resolver
    // saw it at all, whether it passed through it, whether it was only ever an
    // unordered interaction, and whether it took the primary write. See the
    // region block in common_binding_indices.h for how to read the four buckets.
    //
    // Independent of rtx.logSurfaceCoverage: it reuses the coverage buffer's
    // readback but is a far cheaper dump (one line per VS, no per-region
    // histogram), so it can be left on for long captures where the full
    // coverage dump's ~104 ms/frame cannot.
    RTX_OPTION("rtx", bool, logResolveCensus, false,
               "DIAGNOSTIC: log [ResolveCensus] - per vertex shader, how many "
               "primary-resolve interactions saw each surface (ordSeen), how "
               "many took the SharedSurfaceIndex write (ordFinal), how many "
               "were unordered-only (unoSeen), and how many were resolved past "
               "(continued). Separates 'ray never hit it' from 'hit it and "
               "resolved past it'.");
    // NV-DXVK [CamProbe]: raw per-surface lines, not the per-VS rollup.
    //
    // The rollup is what let the last five hypotheses survive as long as they
    // did. A mean camDot of 0.99 across 258 surfaces hides a single surface at
    // 30 degrees, and the failure being hunted is per-object and single-frame,
    // so the aggregate is structurally the wrong shape for it. These lines are
    // emitted only for surfaces in the population that matters - VS verdict
    // NOTRAVERSED, probe self-hit, on screen - which is a handful per frame,
    // and hard-capped per frame regardless.
    RTX_OPTION("rtx", bool, logResolveCensusRaw, true,
               "DIAGNOSTIC: with rtx.logResolveCensus, also emit [CamProbe] raw "
               "per-surface lines for NOTRAVERSED surfaces the TLAS probe "
               "self-hits on screen: camera round-trip angular error in "
               "millidegrees (both NDC Y conventions), what a camera ray hits "
               "instead, and distance. Capped per frame.");
    RTX_OPTION("rtx", int, logResolveCensusRawPerFrame, 12,
               "DIAGNOSTIC: max [CamProbe] raw per-surface lines per frame.");
    // NV-DXVK [Perf.Sweep]: run the whole sub-stage probe set inside ONE run.
    //
    // Every probe above is individually a conf edit plus a restart, and the
    // resulting comparison is cross-run. Cross-run comparison is what produced
    // every wrong number this investigation has had to retract - the 360 ms
    // finalBlit (a debug view left on for three hours), the 0.3 ms unordered
    // stage, the 2.21x loop multiplier. Holding the camera still fixes one
    // variable; running all the probes inside a single process fixes the rest.
    //
    // When enabled the sweep overrides perfUnorderedStopAfter / perfSkipPom /
    // perfSkipMaterialTextures / perfSkipThinFilm / perfUnorderedStepCensus,
    // ignoring whatever those are set to individually.
    RTX_OPTION("rtx", bool, perfAutoSweep, false,
               "DIAGNOSTIC: cycle through every gb_primaryRays sub-stage probe in "
               "one run, holding each for perfAutoSweepSeconds, then log a "
               "[Perf.Sweep] summary table. Overrides the individual perf probe "
               "options while running. Produces a garbage image throughout.");
    RTX_OPTION("rtx", float, perfAutoSweepSeconds, 10.0f,
               "Seconds to hold each step of the rtx.perfAutoSweep ladder.");
    // The timestamp ring is kFrames deep, so the frames immediately after a state
    // change still resolve work recorded under the PREVIOUS step. Discarding a
    // settle window is what keeps step boundaries from smearing into each other.
    RTX_OPTION("rtx", float, perfAutoSweepSettleSeconds, 2.5f,
               "Seconds discarded at the start of each rtx.perfAutoSweep step "
               "before gb_primaryRays samples are accumulated.");
    // Terminates the process once the summary table has been written, so the
    // sweep can be started and walked away from. TerminateProcess rather than a
    // graceful quit on purpose: this fork's normal shutdown path has a known
    // crash/hang where a cached client.dll function pointer is called after the
    // engine has unloaded that module, and a diagnostic run must not be able to
    // lose its own results to it.
    RTX_OPTION("rtx", bool, perfAutoSweepExitOnFinish, true,
               "When rtx.perfAutoSweep completes, flush the log and terminate the "
               "process. Only ever fires if perfAutoSweep was explicitly enabled.");
    RTX_OPTION("rtx", bool, perfGapSampler, false,
               "DIAGNOSTIC: [GapSampler] — sample the game's present thread's RIP every ~2ms\n"
               "(suspend/read/resume) and log a per-module + per-RVA-bucket histogram every 5s.\n"
               "Built 2026-08-06 to name the ~70ms/frame of game CPU that sits AFTER Present\n"
               "and BEFORE the next D3D11 call ([Perf.Boundary] postMs), which the F8 CullOffAB\n"
               "A/B proved independent of every cull site. RVA buckets resolve directly in the\n"
               "client.dll/engine.dll IDBs (IDB addr = 0x180000000 + RVA). ~0.2% overhead on\n"
               "the sampled thread; turn off once the gap is attributed.");
    RTX_OPTION("rtx", bool, perfThreadCensus, false,
               "DIAGNOSTIC: [ThreadCensus] — per-thread CPU census for the WHOLE process,\n"
               "logged every 5s as ms-of-CPU per second of wall (100% = one core saturated).\n"
               "Built 2026-08-06 for the question rtx.perfGapSampler structurally cannot answer:\n"
               "the presenting thread is BLOCKED 66.6ms of a 102.3ms frame ([Perf.Busy]), the\n"
               "wait is NOT a DXVK condvar ([Perf.CondWait] <=0.13ms/frame) and only\n"
               "materialsystem_dx11.dll / tier0.dll import SleepConditionVariableSRW — so it is\n"
               "asleep on the ENGINE's condvar, and the sampler only ever watches g_presentTid.\n"
               "Read busySum first: with a ~102ms frame, a LOW busySum means nothing is computing\n"
               "and the frame is paced by a wait chain or a throttle (stop hunting expensive\n"
               "code); one thread near 100% means that thread is the critical path and the\n"
               "sampler should be pointed at it. Uses QueryThreadCycleTime, whose deltas are\n"
               "EXACT rather than sampled, so it polls at 250ms and never suspends a thread —\n"
               "none of the SuspendThread hazards that make perfGapSampler intrusive apply.");
    RTX_OPTION("rtx", std::string, perfGapSamplerThread, "",
               "DIAGNOSTIC: which thread rtx.perfGapSampler samples. EMPTY (default) = the\n"
               "present thread, i.e. the original behaviour. Otherwise the name set via\n"
               "SetThreadDescription, matched case-insensitively as a substring — \"dxvk-cs\"\n"
               "for the command-stream thread. Re-resolved every 2s, so it survives the\n"
               "thread being created after the sampler starts, and thread ids changing\n"
               "between runs.\n"
               "Point it here when [ThreadCensus] finds a saturated thread: on 2026-08-06\n"
               "dxvk-cs measured 96.97% of one core on a 98.8ms frame — it IS the frame time,\n"
               "with the present thread blocked 61ms and the GPU idle 76ms downstream of it.\n"
               "NOTE the histogram reads differently for a BUSY target than for a blocked one:\n"
               "the syscallCallers/APPcallers lines only populate when the target is parked in\n"
               "a syscall, so for a running thread they go sparse and the per-RVA 'top' line is\n"
               "the signal. That line names OUR code directly — resolve it against d3d11.pdb.");

    // NV-DXVK [NsysAuto]: unattended Nsight Systems capture, same shape as
    // rtx.perfAutoSweep - arm it in rtx.conf, launch, walk away.
    //
    // The point of the gameplay clock is REPRODUCIBILITY. A capture triggered by
    // hand (or by a fixed delay from process start) lands at a different point in
    // the run every time, because load times vary, so two captures are not
    // comparable and a regression cannot be told from a different moment in the
    // level. This counts seconds of ACTUAL GAMEPLAY using the same signal the
    // sweep uses - a non-empty ordered-instance list, plus a warmup - and freezes
    // while in a menu, on a loading screen or alt-tabbed.
    //
    // These options do not talk to nsys. The director advances the clock and
    // prints [NsysAuto] markers; "Capture TF2 (Nsight Systems auto).ps1" watches
    // for them and drives nsys start/stop. That split is deliberate: only
    // something outside the process can confirm the .nsys-rep finished writing
    // before the game is killed, which is the whole point of an unattended run.
    RTX_OPTION("rtx", bool, nsysAutoCapture, false,
               "Unattended Nsight Systems capture director. Waits for gameplay, counts rtx.nsysAutoCaptureSettleSeconds of it, then prints [NsysAuto] CAPTURE-BEGIN / CAPTURE-END markers rtx.nsysAutoCaptureSeconds apart.\n"
               "It does NOT start nsys itself - run the game under 'Capture TF2 (Nsight Systems auto).ps1', which watches for those markers, calls nsys start/stop, waits for the report to finish writing and then closes the game.");
    RTX_OPTION("rtx", float, nsysAutoCaptureSettleSeconds, 10.0f,
               "Seconds of gameplay to wait before triggering the capture. The clock only advances on frames that submitted world geometry to the ray tracer, so menu and loading time does not count and two runs capture the same point in the run.\n"
               "Note that texture streaming is still ramping for the first ~20 s after a level loads (textures 497 -> 997, frame time 79.8 ms vs 64.3 ms settled), so a short settle can capture upload cost that is not representative of steady play. Raise to 45-60 for a settled-scene measurement.");
    RTX_OPTION("rtx", float, nsysAutoCaptureSeconds, 10.0f,
               "Length of the capture window, in seconds of gameplay. Nsight Systems traces are large; 10 s at ~15 fps is ~150 frames, which is plenty for per-pass attribution.");
    RTX_OPTION("rtx", float, nsysAutoCaptureDrainSeconds, 30.0f,
               "Seconds to keep rendering after CAPTURE-END before rtx.nsysAutoCaptureExitOnFinish may terminate the process. Only relevant when the wrapper script is NOT driving the run - the script does not need this, because it waits for the report file itself.");
    RTX_OPTION("rtx", bool, nsysAutoCaptureExitOnFinish, false,
               "Terminate the process rtx.nsysAutoCaptureDrainSeconds after CAPTURE-END. Default false because the wrapper script owns the exit: it can see the .nsys-rep file and so can wait for the report to be written, which this cannot. Set true only when running without the script.");
    // NV-DXVK [Perf.Sweep] quick mode: baseline, probe, baseline. ~45 s instead
    // of ~6 minutes.
    //
    // The full table is a sustained GPU soak and it was being used to answer one
    // question at a time. Two back-to-back full runs overheated the machine into
    // a hard crash, and the second of those tested a single hypothesis that a
    // 45-second bracket would have refuted just as conclusively.
    //
    // The middle step is PASSTHROUGH: it runs with whatever rtx.perf* knobs are
    // set in the config, so a new hypothesis needs no code change and no new
    // table row. The two baselines force every override off, so the bracket is
    // honest regardless of what was left set. Same A-B-A scoring as the full
    // table; with two baselines the RESOLUTION FLOOR is just the gap between
    // them, which is cruder but still an honest error bar.
    // NV-DXVK [Perf.CoherentFetch]: see raytrace_args.h. Distinguishes "too much
    // data moved" from "too scattered", which deleting loads cannot do.
    RTX_OPTION("rtx", uint32_t, perfCoherentUnorderedFetch, 0,
               "DIAGNOSTIC: make the unordered candidate body's memory accesses "
               "coherent without changing their count. 0=off, 1=Surface load "
               "reads index 0 for every candidate, 2=+vertex fetch reads "
               "primitive 0. Produces a garbage image.");
    RTX_OPTION("rtx", bool, perfAutoSweepQuick, false,
               "DIAGNOSTIC: run rtx.perfAutoSweep as a 3-step baseline/probe/"
               "baseline bracket (~45s) instead of the full table (~6min). The "
               "probe step uses whatever rtx.perf* options are set in rtx.conf, "
               "so it can test any single hypothesis without a code change.");
    // NV-DXVK [perf]: forced sampler overrides for the MATERIAL samplers.
    //
    // These exist because none of rtx.nativeMipBias / rtx.upscalingMipBias /
    // rtx.useAnisotropicFiltering / rtx.maxAnisotropySamples reach a material
    // sampler in this fork. They are consumed only by SceneManager::patchSampler,
    // and patchSampler is gated on `samplerOverride == nullptr`
    // (rtx_scene_manager.cpp) - while [D3D11Rtx.SamplerBranch] reports case=0
    // (samplerOverride non-null) as the ONLY case that ever occurs here. So the
    // final sampler is TF2's own D3D11 sampler desc verbatim: aniso 16, lod bias
    // 0, and no Remix knob can touch it.
    //
    // Applied after the branch, to whatever sampler won, so a test cannot be
    // silently skipped the way rtx.nativeMipBias was.
    //   perfForceSamplerMipBias: added to mipmapLodBias. Positive = blurrier =
    //     smaller texel footprint. Separates texture bandwidth/cache cost from
    //     ALU, divergence and occupancy, all of which are untouched by it.
    //   perfForceSamplerAniso: <0 leaves anisotropy alone, <=1 disables it,
    //     otherwise clamps the tap count. Separates aniso TAP COUNT from raw
    //     texel bandwidth, which the mip bias alone cannot do.
    RTX_OPTION("rtx", float, perfForceSamplerMipBias, 0.0f,
               "DIAGNOSTIC: mip LOD bias forced onto material samplers, which the "
               "normal mip bias options do not reach. Positive values are blurrier.");
    RTX_OPTION("rtx", float, perfForceSamplerAniso, -1.0f,
               "DIAGNOSTIC: force anisotropic tap count on material samplers. "
               "-1 = leave unchanged, <=1 = disable anisotropy, N = clamp to N.");
    // NV-DXVK [perf]: see the perfCheapTextureGradients comment in
    // raytrace_args.h. Substitutes the cheap texcoord-difference footprint for
    // the per-pixel clip-space triangle reprojection.
    RTX_OPTION("rtx", bool, perfCheapTextureGradients, false,
               "DIAGNOSTIC: replace computeAnisotropicEllipseAxes with the cheap "
               "texcoord-difference gradient path. Causes texture aliasing; used "
               "to measure what the per-pixel gradient setup costs.");
    RTX_OPTION("rtx", bool, forceCameraJitter, false, "Force enables camera jitter frame to frame.");
    RTX_OPTION("rtx", uint32_t, cameraJitterSequenceLength, 64, "Sets a camera jitter sequence length [number of frames]. It will loop around once the length is reached.");
    RTX_OPTION("rtx", bool, enableDirectLighting, true, "Enables direct lighting (lighting directly from lights on to a surface) on surfaces when set to true, otherwise disables it.");
    RTX_OPTION("rtx", bool, enableSecondaryBounces, true, "Enables indirect lighting (lighting from diffuse/specular bounces to one or more other surfaces) on surfaces when set to true, otherwise disables it.");

    // NV-DXVK: master on/off for the TF2 3D-skybox cloud rendering tech.
    // When true,  the cloud billboards get the fog-blend reconstruction +
    //             unlit cloud-edge compositing (Surface::isTf2SkyboxFog set).
    // When false, the cloud billboards are hidden entirely (their texture is
    //             a near-black coverage map, so without the reconstruction
    //             they would just render solid black). Toggleable at runtime
    //             in the developer menu.
    RTX_OPTION("rtx", bool, enableTf2SkyboxCloudFog, false, "Titanfall 2 3D-skybox cloud rendering. True = fog-blend reconstruction + unlit cloud-edge compositing. False = hide the cloud billboards entirely (without the reconstruction they render solid black).");

    // Needs to be > 0
    RTX_OPTION_ARGS("rtx", float, uniqueObjectDistance, 300.f, "The distance (in game units) that an object can move in a single frame before it is no longer considered the same object.\n"
                    "If this is too low, fast moving objects may flicker and have bad lighting.  If it's too high, repeated objects may flicker.\n"
                    "This does not account for sceneScale.", args.minValue = 0.f);
    
    RTX_OPTION("rtx", bool, useNewGuiInputMethod, true, "Disables the previous method for getting mouse/keyboard input and enables a new method which should be more reliable.  If successful the old method will be deprecated.  This setting can't be changed at runtime, so it must be set in a .conf file.");

    RTX_OPTION_ARGS("rtx", UIType, showUI, UIType::None, "0 = Don't Show, 1 = Show Simple, 2 = Show Advanced.",
                    args.environment = "RTX_GUI_DISPLAY_UI",
                    args.flags = RtxOptionFlags::NoSave | RtxOptionFlags::NoReset);
    RTX_OPTION_ARGS("rtx", bool, defaultToAdvancedUI, false, "Whether to default to the Advanced UI when opening the developer menu.", 
                    args.flags = RtxOptionFlags::UserSetting | RtxOptionFlags::NoReset);

    public: static void showUICursorOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", bool, showUICursor, true, "If true, the ImGUI mouse cursor will be shown when the UI is active.\n"
                    "Can be toggled with Alt + Delete.", args.onChangeCallback = &showUICursorOnChange);
    
    public: static void blockInputToGameInUIOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", bool, blockInputToGameInUI, true,
                    "If true, input will not be passed to the game when the UI is active.\n"
                    "Can be toggled with Alt + Backspace", args.onChangeCallback = &blockInputToGameInUIOnChange, args.flags = RtxOptionFlags::NoSave);

    RTX_OPTION_ARGS("rtx", bool, restoreCursorPosition, false,
                    "If true, the game's mouse cursor position will be restored when the Remix UI is closed.\n"
                    "This should fix the issue where the game camera suddenly turns when closing the UI.\n",
                    args.flags = RtxOptionFlags::UserSetting);

    inline static const VirtualKeys kDefaultRemixMenuKeyBinds{ VirtualKey{VK_MENU},VirtualKey{'X'} };
    RTX_OPTION("rtx", VirtualKeys, remixMenuKeyBinds, kDefaultRemixMenuKeyBinds,
               "Hotkey to open the Remix menu.\n"
               "example override: 'rtx.remixMenuKeyBinds = CTRL, SHIFT, Z'.\n"
               "Full list of key names available in `src/util/util_keybind.h`.");

    RTX_OPTION_ARGS("rtx", DLSSProfile, qualityDLSS, DLSSProfile::Auto, "Adjusts internal DLSS scaling factor, trades quality for performance.",
                    args.environment = "RTX_QUALITY_DLSS",
                    args.flags = RtxOptionFlags::UserSetting);
    // Note: All ray tracing modes depend on the rtx.raytraceModePreset option as they may be overridden by automatic defaults for a specific vendor if the preset is set to Auto. Set
    // to Custom to ensure these settings are not overridden.
    //RenderPassVolumeIntegrateRaytraceMode renderPassVolumeIntegrateRaytraceMode = RenderPassVolumeIntegrateRaytraceMode::RayQuery;
    RTX_OPTION_ARGS("rtx", RenderPassGBufferRaytraceMode, renderPassGBufferRaytraceMode, RenderPassGBufferRaytraceMode::RayQuery,
                   "The ray tracing mode to use for the G-Buffer pass which resolves the initial primary and secondary surfaces to apply lighting to.",
                   args.environment = "DXVK_RENDER_PASS_GBUFFER_RAYTRACE_MODE",
                   args.maxValue = RenderPassGBufferRaytraceMode(uint32_t(RenderPassGBufferRaytraceMode::Count) - 1),
                   args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", RenderPassIntegrateDirectRaytraceMode, renderPassIntegrateDirectRaytraceMode, RenderPassIntegrateDirectRaytraceMode::RayQuery,
                   "The ray tracing mode to use for the Direct Lighting pass which applies lighting to the primary/secondary surfaces.",
                   args.environment = "DXVK_RENDER_PASS_INTEGRATE_DIRECT_RAYTRACE_MODE",
                   args.maxValue = RenderPassIntegrateDirectRaytraceMode(uint32_t(RenderPassIntegrateDirectRaytraceMode::Count) - 1),
                   args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", RenderPassIntegrateIndirectRaytraceMode, renderPassIntegrateIndirectRaytraceMode, RenderPassIntegrateIndirectRaytraceMode::TraceRay,
                   "The ray tracing mode to use for the Indirect Lighting pass which applies lighting to the primary/secondary surfaces.",
                   args.environment = "DXVK_RENDER_PASS_INTEGRATE_INDIRECT_RAYTRACE_MODE",
                   args.maxValue = RenderPassIntegrateIndirectRaytraceMode(uint32_t(RenderPassIntegrateIndirectRaytraceMode::Count) - 1),
                   args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, captureDebugImage, false, "");

    // Denoiser Options
    RTX_OPTION_ENV("rtx", bool, useDenoiser, true, "DXVK_USE_DENOISER",
                   "Enables usage of denoiser(s) when set to true, otherwise disables denoising when set to false.\n"
                   "Denoising is important for filtering the raw noisy ray traced signal into a smoother and more stable result at the cost of some potential spatial/temporal artifacts (ghosting, boiling, blurring, etc).\n"
                   "Generally should remain enabled except when debugging behavior which requires investigating the output directly, or diagnosing denoising-related issues.");
    RTX_OPTION_ENV("rtx", bool, useDenoiserReferenceMode, false, "DXVK_USE_DENOISER_REFERENCE_MODE",
                   "Enables reference \"denoiser\" (~ accumulation mode) when set to true, otherwise uses a standard denoiser.\n"
                   "The reference denoiser accumulates frames over time to generate a reference multi-sample per pixel contribution\n"
                   "which should converge slowly to the ideal result the renderer is working towards.\n"
                   "It is useful for analyzing quality differences in various denoising methods, post-processing filters,\n"
                   "or for more accurately comparing subtle effects of potentially biased rendering techniques\n"
                   "which may be hard to see through noise and filtering.\n"
                   "It is also useful for higher quality artistic renders of a scene beyond what is possible in real-time.");
    RTX_OPTION("rtx", bool, forceResetDenoiserHistory, false,
               "TF2 diagnostic. When true, forces the NRD denoiser to reset its temporal history EVERY frame.\n"
               "Keeps per-frame spatial denoising but removes all temporal accumulation. Used to confirm whether the\n"
               "black 3D-skybox mountains are caused by bad temporal history (they vanish with this on) vs a per-frame\n"
               "input. Global and noisy (no temporal denoise anywhere) — diagnostic only, not for normal use.");
    struct Accumulation {
      RTX_OPTION_ARGS("rtx.accumulation", uint32_t, numberOfFramesToAccumulate, 1024,
                 "Number of frames to accumulate render output.\n"
                 "This can be used for generating reference images smoothed over time.\n"
                 "By default the accumulation stops once the limit is reached.\n"
                 "When desired, continous accumulation can be enabled via enableContinuousAccumulation.",
                 args.environment = "RTX_ACCUMULATION_NUMBER_OF_FRAMES_TO_ACCUMULATE",
                 args.minValue = 1);
      RTX_OPTION_ENV("rtx.accumulation", AccumulationBlendMode, blendMode, AccumulationBlendMode::Average, "RTX_ACCUMULATION_BLEND_MODE",
                     "The blend mode to use for accumulating debug view output.\n"
                     "Supported modes are: 0 = Average, 1 = Min, 2 = Max.\n"
                     "Average is the default mode and is the most common mode to use for accumulation.\n"
                     "Min and Max are useful for visualizing the minimum or maximum value of a debug view output over time.");
      RTX_OPTION_ENV("rtx.accumulation", bool, resetOnCameraTransformChange, true, "RTX_ACCUMULATION_RESET_ON_CAMERA_TRANFORM_CHANGE",
                      "Resets the accumulated debug view output when the camera transform changes.");
    } accumulation;

    RTX_OPTION_ARGS("rtx", bool, denoiseDirectAndIndirectLightingSeparately, true, "Denoising quality, high uses separate denoising of direct and indirect lighting for higher quality at the cost of performance.",
                    args.environment = "DXVK_DENOISE_DIRECT_AND_INDIRECT_LIGHTING_SEPARATELY",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, replaceDirectSpecularHitTWithIndirectSpecularHitT, true, "");
    RTX_OPTION("rtx", bool, adaptiveResolutionDenoising, true, "");
    RTX_OPTION_ENV("rtx", bool, adaptiveAccumulation, true, "DXVK_USE_ADAPTIVE_ACCUMULATION", "");
    RTX_OPTION("rtx", uint32_t, numFramesToKeepInstances, 1, "");
    // NV-DXVK [SubViewKeepLong]: separate, longer keep value for instances
    // carrying InstanceCategories::IgnoreAntiCulling — applied exclusively
    // to TF2 3D-skybox sub-view content (dome + distant mountains), which
    // the engine throttles independently of frame rate (it can skip drawing
    // the sub-view fan for several frames at a time as a LOD optimization).
    // The default numFramesToKeepInstances=1 retires any instance that
    // misses even ONE frame's touch — fine for bone-anim entities which
    // are drawn every frame, but catastrophic for sub-view: when the
    // engine skips a frame, ~90 sub-view instances retire together, their
    // ordered-surface slots get reallocated, and the GBuffer pixels
    // painted from the previous frame (still referencing the retired
    // slots) render as the new occupant — visible as large black/wrong
    // rectangular blocks on the mountains and dome. Default 16 matches
    // the temporal-accumulation window over which a stale slot can still
    // be classified by the Coverage diagnostic.
    RTX_OPTION("rtx", uint32_t, numFramesToKeepSubViewInstances, 16,
               "Lifetime in frames for instances tagged IgnoreAntiCulling "
               "(TF2 3D-skybox sub-view content). Higher than "
               "numFramesToKeepInstances to absorb engine-side draw-rate "
               "throttling on distant sub-view geometry.");
    // NV-DXVK [debug.hideSubViewMountains]: diagnostic toggle to hide
    // all sub-view content EXCEPT the dome (VS_eda5e / vsHash
    // 0x2a729f16017d841b). Used to A/B test whether the residual stale-
    // surfaceIndex corruption is actually caused by the LOD-popping
    // mountain meshes (they change geometry frame-to-frame so propId is
    // fundamentally unstable for them). With this on, only the dome and
    // main-world content render — if the black-block corruption
    // disappears, mountain stale-slot churn is confirmed as the source.
    // Default false (mountains visible).
    RTX_OPTION("rtx.debug", bool, hideSubViewMountains, false,
               "Diagnostic: hide all sub-view (IgnoreAntiCulling) "
               "geometry except the dome. A/B test for whether mountain "
               "propId churn is the source of residual stale-surface "
               "corruption.");
    // NV-DXVK [debug.hideSubViewDome]: also hide the sub-view sky DOME
    // (VS_eda5e / 0x2a729f16017d841b), which hideSubViewMountains deliberately
    // exempts. Use together with hideSubViewMountains to remove ALL sub-view
    // geometry — if the box/triangle artifact disappears only when the dome is
    // also hidden, the dome shader is the culprit (it renders opaque, so
    // enableAlphaBlend=False cannot mask it). Default false.
    RTX_OPTION("rtx.debug", bool, hideSubViewDome, false,
               "Diagnostic: also hide the sub-view sky dome (the one hideSubViewMountains exempts).");
    // NV-DXVK [debug.disableDetailOverlay]: diagnostic toggle to skip the
    // TF2 MOD2X detail overlay (albedo *= detailSample.rgb*2) in
    // opaque_surface_material_interaction.slangh. The Coverage probe showed
    // the "red blot" on the boarding Ark has a BROWN bare albedo sample
    // (~75,41,14) but a RED post-modulation albedo, and detail is the only
    // hue-changing modulation stage. Flip this on to see what the surface
    // looks like WITHOUT the detail overlay: if it's correctly-brown ship
    // hull that fits the scene, the bug is purely the detail overlay (color)
    // and the geometry belongs; if a wrong brown blob floats in empty sky,
    // the geometry itself is misplaced. Default false.
    RTX_OPTION("rtx.debug", bool, disableDetailOverlay, false,
               "Diagnostic: skip the TF2 MOD2X detail-texture albedo overlay.");
    // NV-DXVK [debug.hideVertexShaders]: hide draws by VERTEX-SHADER hash
    // (not texture hash). Needed for multi-material geometry that no single
    // texture identifies — e.g. the sub-view BSP plane drawn by
    // VS 0x2af9b90d63850ec3 / 0x29aa034553107f54, which uses 12+ textures.
    // Matched against DrawCallTransforms::vertexShaderHash. Sets the Hidden
    // category (same effect as hideInstanceTextures: removed from render,
    // still present in captures). Diagnostic tool.
    RTX_OPTION("rtx.debug", fast_unordered_set, hideVertexShaders, {},
               "Diagnostic: hide draw calls whose vertex-shader hash is in this set (Hidden category).");
    // NV-DXVK [debug.dumpVertexShaders]: dump the bound game textures + a
    // geometry/transform report for draws whose VERTEX-SHADER hash is in this
    // set. Fires once per unique texture-hash (textures -> .dds in
    // rtx-remix/captures/textures/) and once per VS hash (the [DumpDraw]
    // report, including a flat-vs-mesh coplanarity verdict over the actual
    // raytraced vertex positions). Runs at the very top of processDrawCallState
    // so it works even when the same VS is also in hideVertexShaders. Pure
    // diagnostic — does not change rendering. Matched against
    // DrawCallTransforms::vertexShaderHash.
    RTX_OPTION("rtx.debug", fast_unordered_set, dumpVertexShaders, {},
               "Diagnostic: dump bound textures + a geometry report for draw calls whose vertex-shader hash is in this set.");
    // NV-DXVK [SpawnGeomDiag.AutoScene]: capture EVERY distinct mesh the TLAS
    // receives, not just the hashes listed in dumpVertexShaders, and aggregate
    // them into one continuously-appended world-space OBJ. Meant for "I do not
    // know which shader draws the thing I am looking at" — load the scene in
    // Blender, find the object, read its group name for the vs/mat hashes.
    // Independent of dumpVertexShaders: that set still works on its own and
    // both can be active at once.
    RTX_OPTION("rtx.debug", bool, dumpAllNewDraws, false,
               "Diagnostic: OBJ-dump every distinct (vertexShader, material, mesh) the TLAS receives into one aggregated world-space scene file, as each first appears.");
    // NV-DXVK [MeshTrace]: follow specific meshes frame by frame, keyed on the
    // MATERIAL hash rather than the vertex-shader hash. This is deliberate —
    // one VS draws several unrelated meshes, so a vs-keyed trace mixes them,
    // while the auto_scene OBJ group names (auto_s<n>_vs<..>_mat<..>) give a
    // material per mesh you picked out in Blender. Paste those mat hashes here
    // and every frame reports where each one is, how big it is, and whether it
    // reached the TLAS at all.
    RTX_OPTION("rtx.debug", fast_unordered_set, traceMaterials, {},
               "Diagnostic: per-frame [MeshTrace] report (position, scale, AABB, presence/absence) for instances whose material hash is in this set. NOT stable across runs - see traceVertexShaders.");
    // The run-stable way to name a mesh. A material hash is NOT usable for
    // this: LegacyMaterialData::updateCachedHash hashes texture image hashes,
    // and DxvkImage::setHash is fed the image POINTER in this fork
    // (d3d11_rtx.cpp ImgHashKey), so material hashes are regenerated on every
    // restart — measured 0 of 628 materials shared between two runs of the
    // same scene. Vertex-shader hashes and vertex counts are properties of the
    // shader and the mesh, and do survive.
    //
    // The three trace sets AND together, ignoring any that are empty, so
    // vs+vertexCount pins one mesh out of a shader that draws many.
    RTX_OPTION("rtx.debug", fast_unordered_set, traceVertexShaders, {},
               "Diagnostic: restrict [MeshTrace] to instances whose vertex-shader hash is in this set. Run-stable, unlike traceMaterials.");
    // NOTE: hash-set options are parsed with std::stoull(s, nullptr, 16) —
    // ALWAYS base 16, with or without an 0x prefix. A vertex count copied
    // straight out of a group name as decimal is therefore misread (632 would
    // become 0x632 = 1586) and silently matches nothing. Write it as hex.
    RTX_OPTION("rtx.debug", fast_unordered_set, traceVertexCounts, {},
               "Diagnostic: restrict [MeshTrace] to instances with one of these vertex counts (the v<N> field in auto_scene OBJ group names). Values are parsed as HEXADECIMAL - write 632 as 0x278. Run-stable.");
    RTX_OPTION("rtx", uint32_t, numFramesToKeepBLAS, 1, "");
    RTX_OPTION("rtx", uint32_t, numFramesToKeepLights, 100, ""); // NOTE: This was the default we've had for a while, can probably be reduced...
    RTX_OPTION("rtx", uint32_t, sceneKeepAliveFrames, 0,
               "Number of consecutive frames without valid camera or raytracing before clearing the scene."
               " Set to 0 to clear immediately (legacy behavior). Higher values prevent scene clearing during"
               " brief shader loading delays, camera cuts, etc.");

    static uint32_t numFramesToKeepGeometryData() {
      return numFramesToKeepBLAS();
    }

    static uint32_t numFramesToKeepMaterialTextures() {
      return numFramesToKeepBLAS();
    }

    static bool enablePreviousTLAS() {
      return !isRayReconstructionEnabled() || useReSTIRGI();
    }

    struct AntiCulling {
      struct Object {
        friend class ImGUI;
        friend class RtxOptions;
        // Anti-Culling Options
        RTX_OPTION_ENV("rtx.antiCulling.object", bool, enable, false, "RTX_ANTI_CULLING_OBJECTS", "Extends lifetime of objects that go outside the camera frustum (anti-culling frustum).");
        RTX_OPTION("rtx.antiCulling.object", bool, enableHighPrecisionAntiCulling, true, "Use robust intersection check with Separate Axis Theorem.\n"
                   "This method is slightly expensive but it effectively addresses object flickering issues that arise from corner cases in the fast intersection check method.\n"
                   "Typically, it's advisable to enable this option unless it results in a notable performance drop; otherwise, the presence of flickering artifacts could significantly diminish the overall image quality.");
        RTX_OPTION("rtx.antiCulling.object", bool, enableInfinityFarFrustum, false, "Enable infinity far plane frustum for anti-culling.");
        RTX_OPTION("rtx.antiCulling.object", bool, hashInstanceWithBoundingBoxHash, true, "Hash instances with bounding box hash for object duplication check.\n Disable this when the game using primitive culling which may cause flickering.");
        // TODO: This should be a threshold of memory size
        RTX_OPTION("rtx.antiCulling.object", uint32_t, numObjectsToKeep, 10000, "The maximum number of RayTracing instances to keep when Anti-Culling is enabled.");
        RTX_OPTION("rtx.antiCulling.object", float, fovScale, 1.0f, "Scale applied to the FOV of Anti-Culling Frustum for matching the culling frustum in the original game.");
        RTX_OPTION("rtx.antiCulling.object", float, farPlaneScale, 10.0f, "Scale applied to the far plane for Anti-Culling Frustum for matching the culling frustum in the original game.");
      };
      struct Light {
        friend class ImGUI;
        friend class RtxOptions;
        RTX_OPTION_ENV("rtx.antiCulling.light", bool, enable, false, "RTX_ANTI_CULLING_LIGHTS", "Enable Anti-Culling for lights.");
        RTX_OPTION("rtx.antiCulling.light", uint32_t, numLightsToKeep, 1000, "(DEPRECATED)");
        RTX_OPTION("rtx.antiCulling.light", uint32_t, numFramesToExtendLightLifetime, 1000, "Maximum number of frames to keep  when Anti-Culling is enabled. Make sure not to set this too low (then the anti-culling won't work), nor too high (which will hurt the performance).");
        RTX_OPTION("rtx.antiCulling.light", float, fovScale, 1.0f, "Scalar of the FOV of lights Anti-Culling Frustum.");
      };

      inline static bool isObjectAntiCullingEnabled() {
        return RtxOptions::AntiCulling::Object::enable() && !RtCamera::enableFreeCamera();
      }

      inline static bool isLightAntiCullingEnabled() {
        return RtxOptions::AntiCulling::Light::enable() && !RtCamera::enableFreeCamera();
      }
    };

    // NV-DXVK [CullOff]: switch the GAME's own culling off, so Remix sees the
    // whole scene and can apply its own scope limit (RtxOptions::SceneCull).
    //
    // These drive verified byte patches into client.dll and engine.dll — see
    // cullOffUpdate() in d3d11_rtx.cpp for the site table, the exact
    // instructions, and the IDA evidence for each. Every site is byte-verified
    // before it is written and restored exactly when its flag goes false, so each
    // can be A/B'd live.
    //
    // Order to enable them in: frustum first (that is the one that makes
    // off-screen objects vanish from shadows and reflections), then distanceFade,
    // then the two default-off ones only if you still see popping — visibilityMask
    // and pvs both widen what the engine submits a lot, and pvs in particular
    // hands the path tracer the entire map.
    //
    // The two staticProp* flags are off to the side of that order: they are the only
    // ones that leave client.dll, and they are what to reach for when what vanishes
    // is scenery/props rather than entities. No client.dll flag can affect static
    // props at all — the engine gathers and culls those itself, in its own module,
    // reached by vtable pointer rather than by a call the client-side scan can see.
    //   staticPropFade     distance only (bounding-radius fade)
    //   staticPropFrustum  VIEW DIRECTION — this is the "look down and the scenery
    //                      disappears" one, measured at r = -0.95 against pitch
    //
    // worldFrustum covers the third mask — world surfaces — which is separate from
    // both the renderable and static-prop ones and collapses with pitch too
    // (r = -0.66). Its producer is a subsystem none of the older flags reach; see
    // the site table for why it stayed hidden through nine byte patches.
    //
    // The three "what actually vanished" flags, by what you are looking at:
    //   frustum            entities / dynamic renderables   (client.dll)
    //   staticPropFrustum  props, scenery, .mdl models      (engine.dll, by vtable)
    //   worldFrustum       walls, terrain, ground           (client.dll, sub_1802E8DA0)
    struct CullOff {
      friend class ImGUI;
      friend class RtxOptions;
      // DEFAULT ON since 2026-08-05. The area-layer chain (sites 12 + 13 + 15)
      // closed the view-dependent culling that caused the light leak: a20 went
      // from 10.75 in the pitch/yaw well against 19-24 on the shoulders, to a
      // FLAT 15 per view across pitch -75..+9 and all 360 degrees of yaw
      // (662 of 670 frames exactly 15; the other 8 are one cold startup frame
      // at eb620=74 and five two-view frames reading 30 = 2 x 15).
      // That is the rotation-invariance criterion: flat across the whole sweep
      // => every view-dependent cull is off. allocFail 0 on all 670 frames,
      // pool peak 202/4092.
      // Turning this off restores the game's own culling and the leak with it.
      RTX_OPTION("rtx.cullOff", bool, enable, true,
                 "Master switch for the engine culling patches. Off = the game culls normally.");
      RTX_OPTION("rtx.cullOff", bool, frustum, true,
                 "Disable the per-renderable frustum cull in client.dll BuildRenderableRenderLists\n"
                 "(sub_1801A9C70 for the main view, the AABB test for shadow/sub-views). This is the\n"
                 "cull that removes off-screen shadow casters and reflection sources.");
      // DEFAULT OFF since 2026-08-05: never validated by any measurement. It was
      // added chasing a DISTANCE-fade theory; the actual bug was view-direction
      // and is now fixed by sites 12/13/15. Forcing full visibility at all
      // ranges is a permanent cost with no demonstrated benefit.
      // Turn on only if props/renderables pop or fade with RANGE, not with view
      // direction — and record the observation when you do.
      RTX_OPTION("rtx.cullOff", bool, distanceFade, false,
                 "Disable the distance-fade cull (sub_1801A90E0 / sub_1801A95B0): props stop fading\n"
                 "out and being dropped from the render list at range. Patches the reject branches,\n"
                 "not the bit-clear, so the render-list entry each renderable carries is still filled\n"
                 "in — clearing only the cull would leave those entries holding stack garbage.");
      RTX_OPTION("rtx.cullOff", bool, probeWorldVis, false,
                 "Diagnostic, not a fix. Wraps client.dll+0x1A8350 (BuildRenderableRenderLists) and\n"
                 "logs [WorldVis] popcounts of ALL FIVE bitmasks packed at the per-view context's\n"
                 "+0x54088, sized from the five-DWORD header at +0x54070:\n"
                 "  m1  world mask 1      b   leaf/PVS input (read by the call)\n"
                 "  m2  static props      t   second renderable mask\n"
                 "  r   renderables       ret render-list entry count\n"
                 "r and ret are read AFTER the call, because that function WRITES them — an\n"
                 "entry-time popcount of r reads 0 by construction, which is what made the earlier\n"
                 "version of this probe unable to resolve anything. [WorldVis.Leaf] adds the four\n"
                 "ClientLeafSystem masks that feed r, and [WorldVis.Cv] names the two ConVars that\n"
                 "gate the build (r_drawallrenderables, pvs_start_early).\n"
                 "Correlate against pitchDeg from the [PitchProbe] lines, BINNED BY PITCH — a global\n"
                 "average cannot resolve this. Check layoutOk=1 and rPre=0 before trusting a run.");
      RTX_OPTION("rtx.cullOff", bool, probeWorldJobs, false,
                 "Diagnostic, not a fix. Wraps client.dll+0x2E8DA0 (the WORLD VISIBILITY WORKER,\n"
                 "sub_1802E8DA0) and logs one [JobProbe] line per frame, on the same line cadence as\n"
                 "[PitchProbe] and carrying pitchDeg itself so it is binned by pitch by construction:\n"
                 "  calls     invocations that frame — ONE CALL PER JOB. This is the number.\n"
                 "  recCntSum sum of the count word each job's record points at (supply proxy;\n"
                 "            exact semantics NOT confirmed, do not build an argument on it)\n"
                 "  jobIdx    min/max of the a2 job index seen\n"
                 "  planes    [lo,hi] of dword_1811FC0C0, the frustum plane count (4, or 8 when the\n"
                 "            extra plane blocks are live)\n"
                 "  leafSkip  [lo,hi] of dword_1811FC110, the leaf-skip threshold\n"
                 "  split     [lo,hi] of dword_1811FC114, the JOB-SPLIT SUBTREE THRESHOLD — the worker\n"
                 "            spawns a new job via JT_GrowJobArray_Lock whenever a subtree exceeds it,\n"
                 "            so this sets the job count directly\n"
                 "  leafSkipF/splitF  the same two reinterpreted as float. IDA types both `dword_`,\n"
                 "            which is its default for an untyped global, not a proven type.\n"
                 "lo==hi is the expected reading; lo!=hi means the value moved mid-frame across job\n"
                 "threads and the field is a range, not a reading. [0,0] on all three means the\n"
                 "install-time readability check FAILED (it logs a warning) — that is the fields being\n"
                 "ABSENT, not zero. CULLING_BIBLE §4a: a diagnostic that silently reports a constant\n"
                 "is a probe defect, not a result.\n"
                 "MEASURED RESULT ACROSS PITCH (2026-08-05): calls/frame is FLAT — 192/206/207/186/\n"
                 "186/184/184/184 over 10-degree bins to 80 degrees, non-monotonic, sd=0 across the\n"
                 "top three bins — while instance count falls 23% (r=-0.77) on a camera that moved\n"
                 "9 units.\n"
                 "MEASURED RESULT ACROSS YAW (2026-08-05, later, stationary camera, camPos spread\n"
                 "0.3u, all eleven [CullOff] sites ON): calls is NOT flat — 181 / 95 / 48 over yaw\n"
                 "bins 136-140 / 142-148 / 148-152, with m1 1989 / 1078 / 372 and inst 610 / 455 / 295.\n"
                 "Fully reversible across four sweeps. So the pitch-era conclusion 'job supply is flat,\n"
                 "sub_1802EB620 excluded' was AXIS-SPECIFIC and does not hold under yaw.\n"
                 "BUT calls is NOT a supply measurement (see the node-ring note below — the worker\n"
                 "splits ITSELF, so job count is an output of the traversal too). recCntSum/calls is\n"
                 "4.0 in every yaw bin, i.e. the per-job record content is invariant and only the\n"
                 "NUMBER of jobs moves. The three threshold fields exist to settle that: they are\n"
                 "written before dispatch and read by every job, so they are a genuine input.\n"
                 "  a threshold moves with yaw => that is the mechanism, stop there.\n"
                 "  all three flat             => the input is invariant, job count is self-generated,\n"
                 "     and the next target is sub_1802ED900 (producer of the 0x1380A40 records, called\n"
                 "     from EB620 at 0x2EB8C5 / 0x2EC9FD).\n"
                 "See rtx.cullOff.probeWorldDrain for the output side.\n"
                 "USE THIS INSTEAD OF THE x64dbg HIT COUNTER in HANDOFF_PITCH_CULL_2026-08-05 §10.1.\n"
                 "That route costs one debug exception per hit — ~1600 hits/frame drove the game to\n"
                 "0.5 fps and killed it twice — and cannot be normalised per frame without\n"
                 "hand-correlating two clocks.\n"
                 "DO NOT count node-loop iterations to answer §6. Its premise that the loop bound\n"
                 "comes from the caller is false: var_12A0 is popped from a 1024-entry ring queue the\n"
                 "function fills itself, so with the reject branches NOPed the traversal EXPANDS.\n"
                 "Measured: 6.46 nodes/call at pitch 5.8 deg vs 9.98 at 89 deg — up, not down.");
      RTX_OPTION("rtx.cullOff", bool, probeDispatch, false,
                 "Diagnostic, not a fix. [DispProbe] — entry trampolines on the AREA DISPATCH path\n"
                 "in sub_1802EB620: sub_1802E8A20 (one call per area dispatched, and where\n"
                 "dword_1811FC0D8 is sampled) and sub_1802E7C70 (the record allocator; a -1 return\n"
                 "makes sub_1802E8A20 skip its ENTIRE body — no jobs, no mask bits, no log, and no\n"
                 "reject branch any other flag can patch).\n"
                 "SEPARATE FLAG ON PURPOSE, and default OFF: this probe FROZE THE GAME on\n"
                 "2026-08-05. The cause was a third hook on sub_1802ED900, which does NOT decompile\n"
                 "— IDA's one-_DWORD-arg guess was wrapped as int __fastcall(unsigned int), and that\n"
                 "clobbers the volatile xmm registers a SIMD plane-builder needs. That hook is gone\n"
                 "and must not come back as a C wrapper. Only functions that DECOMPILE are hooked\n"
                 "here, because a C wrapper commits to a calling convention and IDA's argument list\n"
                 "for a function it could not decompile is a guess, not a fact.\n"
                 "If the game hangs again with this on, set it False — no rebuild needed — and say\n"
                 "so, because that would mean one of the two REMAINING hooks is also unsafe.\n"
                 "Areas dropped by sub_1802ED900's -1 at 0x2EB8D0 need no hook: such an area never\n"
                 "reaches sub_1802E8A20, so dropped = (pend+1) - a20, both already on the line.\n"
                 "ALSO INSTALLS [AreaSeed] on sub_1802EF090 (client.dll+0x2EF090), which DECOMPILES\n"
                 "as `__int64 __fastcall(unsigned int, __int64)` — signature read from the body, not\n"
                 "guessed — and is sub_1802EB620's direct caller. It reports the layer above\n"
                 "[AreaDump]: which areas sub_1802EAD60 seeded into the order list word_1811FE920 at\n"
                 "all. The queue loop at 0x2EB864 iterates that list, so an area missing from it can\n"
                 "never dispatch whatever its selector is — which is why rtx.cullOff.areaPortal\n"
                 "raised a20 from 5 to 23 at low yaw and changed nothing at 155-167deg.\n"
                 "  org       the PRE-TRANSFORM fc000, read from EF090's a2 argument. NOT the same\n"
                 "            value as fc000 on the [DispProbe] line: that one is sampled after\n"
                 "            EB620's tail overwrites the global at 0x2ECB2F, so the two differing is\n"
                 "            expected and is the confirmation. Compare org against camPos.\n"
                 "  listLen   how many entries EAD60 wrote. Recovered by sentinel-filling\n"
                 "            word_1811FE920 with 0xFFFF before the call and scanning after, which\n"
                 "            avoids hooking EAD60 (it does NOT decompile). Writing below the queue\n"
                 "            loop's cursor is inert — it only ever reads [cursor, nAreas).\n"
                 "READ listLen BINNED BY YAW against the same frame's a20: falling means EAD60 is\n"
                 "view-dependent and its 0x2EB0F1 crossing test is the gate; flat while a20 falls\n"
                 "means EAD60 is exonerated and the loss is between the order list and the selector.\n"
                 "ef090Inst is on the line because an uninstalled hook and a never-called function\n"
                 "both report zero.\n"
                 "ALSO ADDS ed900Drop to the [DispProbe] line — areas DROPPED by sub_1802ED900\n"
                 "returning -1 at 0x2EB8D0, counted with a tail-jmp counter island on loc_1802EBE71\n"
                 "(one xref, 0x2EB8D0, so it cannot merge another path). This is the verdict, not\n"
                 "the call count: ed900 sat flat at ~1.0/frame across yaw while a20 fell 5 -> 2 and\n"
                 "that was recorded as an acquittal, but the drop happens BEFORE the dispatch at\n"
                 "0x2EB910 and before the portal loop at 0x2EB915, so one -1 on an area whose portals\n"
                 "would have opened the rest takes the whole flood with it — and a flat call count is\n"
                 "exactly what one early drop looks like. ED900 reads xmmword_1811FC030 (camera\n"
                 "FORWARD) at four sites, so its verdict is view-dependent even where its count is\n"
                 "not. Read ed900Drop against a20 on the same line; ed900DropInst guards the zero.");
      RTX_OPTION("rtx.cullOff", bool, worldPortal, true,
                 "THE EIGHTH REJECT — the residual that survived rtx.cullOff.worldFrustum.\n"
                 "client.dll+0x2E955C in sub_1802E8DA0. This is the AREA-PORTAL / occluder test,\n"
                 "NOT the frustum test: the loop at 0x2E941C builds 4-6 silhouette planes per job\n"
                 "record and accumulates acceptance in ebx, then `neg ebx / sbb eax,eax / and eax,2 /\n"
                 "jz loc_1802E9DB1` drops the whole node when nothing accepted.\n"
                 "WHY IT WAS MISSED: the exhaustiveness proof xref'd the reject TAILS\n"
                 "(loc_1802E9DB7 / loc_1802E9CA2) and found 3+4 branches. This one targets\n"
                 "loc_1802E9DB1, which falls THROUGH into loc_1802E9DB7 six bytes later, so it never\n"
                 "appears in an xref of the tail. Xref every label in the tail's basic block.\n"
                 "The patch jumps to 0x2E9567 — the engine's own portal-PASSED path (`mov eax, 1`,\n"
                 "reload planes, restore regs) — rather than NOPing the jz, because eax is a mode\n"
                 "consumed at 0x2E95BE (1=accept, 2=partial) and a NOP would fall through with eax=0,\n"
                 "a state no downstream path expects.\n"
                 "Kept SEPARATE from worldFrustum so the A/B is attributable — the staticProp sites\n"
                 "were previously left on and credited for an improvement they did not cause.\n"
                 "VERIFY with rtx.cullOff.probeWorldDrain: m1Max binned by pitch should stop falling.\n"
                 "Before this patch it went 1425/1532/1329/1047/769/791 over 10-degree bins to 60.\n"
                 "COST: this submits nodes the area-portal system would have occluded, so it is the\n"
                 "first flag to turn off if indoor/portal-heavy areas regress on draw count.");
      RTX_OPTION("rtx.cullOff", bool, areaPortal, true,
                 "THE AREA LAYER — three levels above every other flag here, and the reason\n"
                 "occluders still go missing with all of them on. client.dll+0x2EB9CF in\n"
                 "sub_1802EB620.\n"
                 "MEASURED (2026-08-05, [AreaDump], stationary camera, yaw 51-153deg): exactly five\n"
                 "areas dispatch per frame. 124 and 178 are unconditional; 127 drops at ~146deg,\n"
                 "then 149 at ~150deg, then 125 at ~152deg, and when they go their geometry never\n"
                 "reaches the TLAS at all. Fully reversible with yaw.\n"
                 "WHAT IT IS: sub_1802EB620's per-portal loop trivially rejects a portal when every\n"
                 "one of its vertices lies outside the SAME plane. r9d starts at 0x1FF and is ANDed\n"
                 "with each vertex's outcode from dword_18120092C; `test r9d,r9d / jnz loc_1802EC91C`\n"
                 "then skips to the next portal, so the neighbour area never reaches the record\n"
                 "allocation at 0x2EC6FA, keeps selector -1 in dword_1811FF91C, and is skipped by the\n"
                 "queue loop's own `jz` at 0x2EB884.\n"
                 "WHY IT IS VIEW-DEPENDENT: the outcodes are built per frame by sub_1802EE940 (job\n"
                 "type byte_1811FBD92) as movemask(dot(plane, vert - xmmword_1811FC000) < 0) against\n"
                 "the four frustum SIDE planes xmmword_1811FC040..070 in bits 0-3, plus the camera\n"
                 "FORWARD plane xmmword_1811FC030 in bit 8. Bits 4-7 are the 5-8 plane set and stay\n"
                 "clear because sub_1802EF090 sets dword_1811FC0C0 = 4 for the main view. So the live\n"
                 "test is over five view-dependent planes, and it is the ONLY yaw-dependent gate on\n"
                 "the path.\n"
                 "THE PATCH is `test r9d,r9d` -> `xor r9d,r9d` (45 85 C9 -> 45 31 C9). Same length,\n"
                 "forces ZF=1 so the jnz is never taken and control falls through to the engine's own\n"
                 "accept path with the branch left intact — nothing is NOPed. Clobbering r9d is safe:\n"
                 "the accept path re-zeroes it four instructions later at 0x2EB9E0.\n"
                 "SEPARATE FLAG from worldPortal on purpose. worldPortal is the node-level occluder\n"
                 "test INSIDE the worker sub_1802E8DA0; this is the area-level portal flood that\n"
                 "decides whether that worker is ever handed the area. Different subsystem, three\n"
                 "levels up, and only an independent toggle makes the A/B attributable.\n"
                 "VERIFY with rtx.cullOff.probeDispatch: [AreaDump] should show all five nodes at\n"
                 "every yaw and [DispProbe] a20 should stop falling (it went 5 -> 2 across the sweep).\n"
                 "COST AND WHAT TO WATCH: this is only the TRIVIAL reject. Accepted portals still run\n"
                 "the edge-clip loop at 0x2EBA50, and the worker still culls surfaces against the\n"
                 "resulting plane set, so this dispatches areas rather than guaranteeing their\n"
                 "geometry survives. It also removes the bound on the portal flood — every crossing\n"
                 "now allocates from the 4092-block pool at unk_181380A40, which peaked at 36 with\n"
                 "the reject in place. sub_1802E7C70 returning -1 at 0x2EC6FF drops an area silently,\n"
                 "so if areas still go missing check allocFail/poolHi in [JobProbe] FIRST.");
      // DEFAULT ON, and it MUST be paired with areaSkipClip — on its own it only
      // DEFERS the reject from 0x2EBCDC to 0x2EC675 (measured: 8,940 rejects,
      // 100% with 0 entries and 0 planes). See areaSkipClip for the pair.
      RTX_OPTION("rtx.cullOff", bool, areaClip, true,
                 "THE EXACT FORM OF areaPortal's TEST, and the reason that flag alone was not\n"
                 "enough. client.dll+0x2EBCDC in sub_1802EB620's per-edge clip loop.\n"
                 "MEASURED with areaPortal ON and verified: a20 rose 5 -> 23 and new areas (BSP\n"
                 "nodes 62/74/75/126/127) appeared at low yaw, but at 155-167deg a20 and alloc were\n"
                 "BYTE-IDENTICAL to the unpatched run (2.00 / 6.0). The two upstream candidates were\n"
                 "eliminated on their own measurements: [AreaSeed] listLen is 30 on every frame from\n"
                 "82 to 159deg (min == max) and contains every area that goes missing, so\n"
                 "sub_1802EAD60's order list is not the gate; [Ed900Drop] counts 68 drops at yaw 50\n"
                 "and 0 at 120-150, so sub_1802ED900's -1 tracks how many areas are QUEUED rather\n"
                 "than the collapse.\n"
                 "WHAT IT IS: the clip loop accumulates min/max of the record's planes projected onto\n"
                 "each portal edge plane (minps/maxps at 0x2EBC13/0x2EBC16), then\n"
                 "`cmpltps xmm0(0), xmm7 / movmskps / test / jz loc_1802EC8ED` abandons the portal\n"
                 "when max <= 0 in every lane — the polygon is wholly outside. That is the same\n"
                 "predicate as areaPortal's outcode test at 0x2EB9CF, at full precision. Patching the\n"
                 "cheap one let through portals it rejected and this one accepts (low yaw) and\n"
                 "changed nothing where this one rejects too (high yaw). Both have to go.\n"
                 "THE PATCH retargets the jz to loc_1802EC6B6 instead of NOPing it. Falling through\n"
                 "with max <= 0 means min <= 0 as well, so the trivial-accept at 0x2EBCEB is not\n"
                 "taken either and the clipper runs on an empty polygon, which 0x2EC675 then abandons\n"
                 "anyway — a NOP would only move the reject. loc_1802EC6B6 is the engine's own\n"
                 "trivial-accept continuation and is the exact target 0x2EBCEB jumps to, so\n"
                 "'wholly outside' is handled as 'wholly inside': the plane set is left unnarrowed\n"
                 "for that edge and the loop continues to the allocation with valid counts. Safe\n"
                 "because only xmm0/xmm6/eax differ between the two sites and all are dead there;\n"
                 "the displacement checks against the engine's own encoding, 0x2EBCE2 + 0x9D4 ==\n"
                 "0x2EBCF1 + 0x9C5 == 0x2EC6B6.\n"
                 "*** THE PARAGRAPH THAT USED TO BE HERE — 'it also keeps 0x2EC675/0x2EC67F out of\n"
                 "*** play, a wholly-outside portal no longer enters the clipper, so it cannot\n"
                 "*** produce a degenerate result' — IS REFUTED BY MEASUREMENT. [DegenPair],\n"
                 "*** 2026-08-05, 570 frames incl. 164 in the well: 8,940 rejects at 0x2EC675 and\n"
                 "*** EVERY ONE is r11=0 AND r9=0, with clipDegenB 0 throughout.\n"
                 "THE ERROR: the branch is a verdict on ONE PORTAL EDGE, not on the portal. The loop\n"
                 "at 0x2EBA50 walks every edge in turn, narrowing the volume against each edge plane.\n"
                 "Retargeting the wholly-outside verdict skips the narrowing for THAT edge only; the\n"
                 "portal's remaining edges still enter the clipper at 0x2EBCF1 and clip normally — on\n"
                 "a volume that has already been found to miss the portal. They reduce it to nothing,\n"
                 "and 0x2EC675 abandons it with 0 entries and 0 planes. So the reject is not kept out\n"
                 "of play, it is DEFERRED: from 0x2EBCDC to 0x2EC675. Exactly the failure predicted\n"
                 "two lines above for the NOP variant, arriving by a different route.\n"
                 "This is also why 0x2EC675 must not be patched (see areaDegen): with r11=0 and r9=0\n"
                 "the clip is telling the truth — there is no polygon — and forcing a rec[+2]==0\n"
                 "record through it is what crashed sub_1802ED900.\n"
                 "IF CONFIRMED BY THE abMode A/B (park in the well, PageUp to 1 and back, watch\n"
                 "clipDegenA), the coherent form of 'disable the antiportal narrowing' is to skip the\n"
                 "clip for the WHOLE portal rather than for one verdict:\n"
                 "  0x2EBCE6:  0F 50 C6 -> 31 C0 90   movmskps eax,xmm6 -> xor eax,eax ; nop\n"
                 "`test eax,eax` then always sets ZF, the jz at 0x2EBCEB always takes trivial-accept,\n"
                 "0x2EBCF1 never runs, r15d/esi keep the INCOMING record's counts, and 0x2EC6F5\n"
                 "allocates a copy of the parent volume — a 0/0 record becomes structurally\n"
                 "impossible. Same falsify-the-condition idiom as areaPortal, and it REPLACES this\n"
                 "site rather than stacking on it. eax and xmm6 are dead there (xmm6 is\n"
                 "re-initialised at 0x2EBA82 every iteration).\n"
                 "COST: removes the antiportal narrowing for every wholly-outside portal, so the\n"
                 "flood widens well beyond areaPortal. Pool peaked at 14/4092 with allocFail=0, so\n"
                 "there is headroom — watch poolHi/allocFail in [JobProbe].\n"
                 "SEPARATE FLAG from areaPortal so the A/B can attribute the exact reject on its own.\n"
                 "VERIFY with rtx.cullOff.probeDispatch: [AreaDump] should show the dropped nodes at\n"
                 "every yaw and [DispProbe] a20 should stop falling past 140deg.");
      // DEFAULT ON. Together with areaClip this is THE FIX for the light leak —
      // verified 2026-08-05: a20 flat at 15 per view across pitch -75..+9 and
      // all 360 degrees of yaw, where it had been 10.75 in the well against
      // 19-24 on the shoulders. Neither site does it alone.
      RTX_OPTION("rtx.cullOff", bool, areaSkipClip, true,
                 "SITE 15 — skip the antiportal narrowing for the WHOLE portal.\n"
                 "client.dll+0x2EBCE6. REQUIRES rtx.cullOff.areaClip = True; the code ANDs the two.\n"
                 "*** CORRECTED 2026-08-05 AFTER ITS FIRST CAPTURE. This option originally said it\n"
                 "*** SUPERSEDED areaClip and was ANDed with !areaClip. That was wrong. The byte\n"
                 "*** order settles it: 0x2EBCDC (wholly-outside -> reject, the branch areaClip\n"
                 "*** patches) is evaluated BEFORE 0x2EBCE6, and site 15 does not touch it. With\n"
                 "*** areaClip off, a wholly-outside portal is still rejected outright, so site 15\n"
                 "*** alone removes the NARROWING but not the REJECT — and it raised a20 in no bin\n"
                 "*** of its first capture, which is what that looks like.\n"
                 "The two are COMPLEMENTARY and only the pair expresses 'disable the antiportal\n"
                 "cull': 13 turns the reject into a continue, 15 stops every other edge narrowing.\n"
                 "WHAT IT IS: the loop at 0x2EBA50 walks the portal's EDGES, narrowing the volume\n"
                 "against each edge plane. 0x2EBCDC short-circuits on wholly-outside (reject) and\n"
                 "0x2EBCEB on wholly-inside (continue, no narrowing). areaClip retargets the FIRST\n"
                 "to the SECOND's destination, which skips narrowing for the one edge that reported\n"
                 "wholly-outside — but the portal's OTHER edges still enter the clipper at\n"
                 "0x2EBCF1 and clip a volume already known to miss the portal down to nothing.\n"
                 "This site falsifies the wholly-INSIDE test instead, so trivial-accept is taken\n"
                 "for every edge: movmskps eax,xmm6 -> xor eax,eax + nop, so `test eax,eax` at\n"
                 "0x2EBCE9 always sets ZF and 0x2EBCF1 becomes unreachable. r15d/esi keep the\n"
                 "INCOMING record's counts and 0x2EC6F5 allocates a copy of the parent volume.\n"
                 "MEASURED BASIS (2026-08-05, [DegenPair]): every one of 8,940 rejects at 0x2EC675\n"
                 "is cell 0/0 — r11=0 AND r9=0 — and the per-area degen rate roughly halves with\n"
                 "areaClip off (0.83 -> 0.44 overall, 1.60 -> 0.68 in the well). So areaClip DEFERS\n"
                 "the reject rather than removing it, and areaDegen existed only to suppress the\n"
                 "deferred one. Note the halving, not elimination: a residual source remains and\n"
                 "areaPortal (site 12) is the untested candidate for it.\n"
                 "CANNOT CRASH THE WAY areaDegen DID, structurally: a 0/0 record is produced by\n"
                 "the clipper and the clipper never runs, so rec[+2] >= 3 is inherited from the\n"
                 "parent and ED900's unguarded post-test loop at 0x1802EDA30 can never be handed a\n"
                 "bound of zero.\n"
                 "JUDGE IT ON a20 IN THE WELL, NOT ON clipDegen. clipDegen going to 0 is guaranteed\n"
                 "by the patch and proves nothing. PASS = well a20 (pitch -45..-50, yaw >= 130)\n"
                 "rises toward the shoulder value; baseline is 10.75 there against 19-24 on the\n"
                 "shoulders with areaClip on, 5.44 with everything in the area layer off.\n"
                 "FAIL = the well persists, which exonerates the area layer entirely and means the\n"
                 "leak is elsewhere — stop patching this layer and build the rotation-invariance\n"
                 "baseline instead.\n"
                 "COST: removes the narrowing entirely, so the flood is wider than areaClip's.\n"
                 "Watch poolHi/allocFail on [JobProbe] — areaClip alone took the pool peak from\n"
                 "14/4092 to 222/4092.");
      // DEFAULT ON. SITE 16 — the answer to HANDOFF_AREA_CLUSTER_SITE15's open
      // question: makes site 15 and sub_1802ED900 coexist instead of removing
      // either. Found 2026-08-06 from the pitch-sweep capture + ED900 disasm.
      RTX_OPTION("rtx.cullOff", bool, areaMergeSalvage, true,
                 "SITE 16 — stop sub_1802ED900's DEGENERATE merge verdict from dropping areas.\n"
                 "client.dll+0x2EDDDD, inside ED900 itself.\n"
                 "MECHANISM: ED900 is only called (0x2EB8C5) when an area's selector chain has more\n"
                 "than one record — several portals reached the area and each appended its volume.\n"
                 "It merges the chain into one record and returns -1 if fewer than 3 edges survive\n"
                 "(cmp r12d,3 at 0x2EDDD9); the caller then DROPS the area (0x2EB8D0 -> 0x2EBE71),\n"
                 "and a dropped area takes everything behind it. With areaSkipClip on, chains carry\n"
                 "full inherited parent volumes, identical planes meet in the merge, their triple\n"
                 "products read 0, and edges collapse through the degenerate path (0x2EDED5, the\n"
                 "thing clipDegen counts — 457/frame in the 2026-08-06 capture) until r12d < 3.\n"
                 "The merge is saying DEGENERATE, not INVISIBLE; rejecting the area on it is the\n"
                 "residual cull that killed the 11-area cluster.\n"
                 "MEASURED (2026-08-06 pitch sweep, fixed position): the dropped identity is purely\n"
                 "view-dependent — pitch -47..-56 drops hub area 127, which feeds 124/125/126/149,\n"
                 "and a20 collapses to 4-8; other pitch bands drop 64/65/92 instead. Same mechanism\n"
                 "as the y=-9984 step (there the victims were 124/149).\n"
                 "THE PATCH reuses the engine's own fallback: for merge OVERFLOW (r12d > 255) ED900\n"
                 "already skips the intersection and builds a conservative quad record over ALL the\n"
                 "chain's planes (0x2EDDF0/0x2EDF9E). The jb at 0x2EDDDD is retargeted from the\n"
                 "return -1 epilogue (rel32 0x7D2) to that fallback (rel32 0x0D). Register state at\n"
                 "0x2EDDF0 is identical from either entry; r12d is not read there.\n"
                 "VISUAL RISK: none by construction — the salvage record is an over-approximation\n"
                 "of the union, so it can only UNDER-cull (more geometry dispatched), never hide\n"
                 "anything. Cost is perf: wider flood, more TLAS surfaces.\n"
                 "NOT gated on areaSkipClip: the every-frame background drops (64/65, usually 92)\n"
                 "happen in every configuration and share the mechanism.\n"
                 "VERIFY on [DispProbe]: dropAreas empties, a20 stops dipping at pitch -47..-56,\n"
                 "and the 81/84/92/96/113/141/146/148/152/153/166 cluster stays resident across\n"
                 "the y=-9984 step and a full pitch sweep.");
      RTX_OPTION("rtx.cullOff", bool, areaDegen, false,
                 "*** CRASHES. DO NOT ENABLE. *** Access violation (write) at client.dll+0x2EDA45\n"
                 "inside sub_1802ED900, called from EB620 at 0x2EB8C5 on a tier0 job thread, after a\n"
                 "collapse to ~3 fps. Records with 0 entries/planes are fine for the allocator and\n"
                 "for the copy loops at 0x2EC74B — which is as far as the original check went — but\n"
                 "sub_1802ED900 consumes the record afterwards whenever rec[+4] != -1, does NOT\n"
                 "decompile, and underflows its loop bounds on zero counts.\n"
                 "REQUIREMENTS NOW READ FROM DISASSEMBLY (2026-08-05). rec[+4] is the next-record\n"
                 "link of a per-area chain; ED900 walks it, appends each record's planes to\n"
                 "unk_181E60EF0, then frees the record. Only ONE of the two counts is lethal:\n"
                 "  rec[+0] (edges,  from r11 at 0x2EC671) == 0 is GUARDED at 0x1802EDA84.\n"
                 "  rec[+2] (planes, from r9  at 0x2EC67B) == 0 walks a POST-TEST loop at\n"
                 "    0x1802EDA30 whose bound is that count, so zero never terminates and it\n"
                 "    writes 16 bytes per iteration off the end of the array at 0x1802EDA45 —\n"
                 "    the exact faulting instruction, and exactly a write.\n"
                 "So the crash is the 0x2EC67B half alone. Relaxing ONLY 0x2EC671 (cmp r11,3 ->\n"
                 "cmp r11,0) is memory-safe: the untouched cmp r9,3 guarantees rec[+2] >= 3, and\n"
                 "ED900 is skipped entirely for a chain of length 1 (0x2EB8C0), which is the\n"
                 "141/148 single-crossing case. NEVER relax 0x2EC67B, and do not 'soften' it to\n"
                 "cmp r9,1 either — 1-2 planes is an unbounded volume pushed into the job array.\n"
                 "*** ANSWERED 2026-08-05 BY [DegenPair], AND THE SALVAGE IS DEAD. 570 frames,\n"
                 "*** 164 of them in the well: every reject is cell 0/0 — r11 = 0 AND r9 = 0,\n"
                 "*** 8,940 of them, no exceptions, clipDegenB 0 throughout. The r11-only patch is\n"
                 "*** a strict no-op (relaxing cmp r11,3 hands control to cmp r9,3 with r9 = 0,\n"
                 "*** which rejects to the same 0x2EC8ED). Do not split the flag group for it.\n"
                 "The same capture confirms the crash from live data rather than from a register\n"
                 "dump: the degenerate records really are rec[+2] == 0, at 14-42 per frame in the\n"
                 "well, which is precisely the input that makes ED900's post-test loop run forever.\n"
                 "AND THERE IS NO UPSTREAM FIX EITHER — 'intervene where the clip produces the\n"
                 "degenerate polygon' is retired. r11=0 ^ r9=0 requires the plane compaction at\n"
                 "0x2EBD60 to keep no plane, the wholly-inside edge mask r8 to be empty AND the\n"
                 "straddle mask var_8A0 to be empty. That is a volume entirely outside the portal:\n"
                 "the clip is not producing a sliver, it is producing nothing, and this reject is\n"
                 "reporting that truthfully. Nothing upstream can invent a polygon.\n"
                 "The cause is one level up, in rtx.cullOff.areaClip — see that option.\n"
                 "Everything below is the original rationale, kept because the ANALYSIS that located\n"
                 "the site is sound even though the chosen fix is not.\n"
                 "THE PITCH HALF. areaPortal and areaClip fixed the yaw axis; this is the one that\n"
                 "closes the diagonal. client.dll+0x2EC671 and +0x2EC67B in sub_1802EB620.\n"
                 "REQUIRES rtx.cullOff.worldPortal — see the dependency note below. The flag is\n"
                 "ANDed with it in code, so this site simply will not apply without it.\n"
                 "HOW IT WAS FOUND: the collapse is a localised WELL in 2D, not a trend on either\n"
                 "axis, which is why pitch-only and yaw-only binning both looked flat. At yaw >= 130\n"
                 "a20 runs 23.6 (pitch -65) / 19.1 (-55) / 11.8 (-50) / 11.0 (-45) / 14.8 (-40) /\n"
                 "17.7 (-35), recovering on both shoulders, and areas 113/141/148/153 vanish as a\n"
                 "GROUP while sitting in the order list on every frame. The [ClipDegen] recorder then\n"
                 "named the abandoned crossings: at the well bottom they are into 148 and 141,\n"
                 "~0.66/frame each, absent from both shoulders. A per-frame COUNT could never have\n"
                 "shown that — one crossing dying does not move a total of 17-40, which is why the\n"
                 "earlier 'clipDegen is anti-correlated so it is not the gate' reading was wrong.\n"
                 "WHAT IT IS: after the per-edge clip, `cmp r11,3 / jb` and `cmp r9,3 / jb` abandon\n"
                 "the portal when the clipped polygon has fewer than 3 entries or planes. At grazing\n"
                 "pitch against floor- and ceiling-adjacent portal edges that is exactly what the\n"
                 "clip produces.\n"
                 "THE PATCH changes each immediate 3 -> 0. Unsigned '< 0' is impossible, so neither\n"
                 "jb can fire, r11/r9 are untouched, and the engine's own fall-through at 0x2EC685\n"
                 "publishes them and swaps buffers exactly as for a successful clip. One byte each.\n"
                 "WHY NOT DISCARD THE CLIP INSTEAD: by 0x2EC675 the previous polygon is gone. rbx and\n"
                 "rdi are destroyed in the final dedup pass (0x2EC554/0x2EC5F1/0x2EC61F/0x2EC570),\n"
                 "var_840's saved entry count is consumed at 0x2EC52F, and the pre-clip plane count\n"
                 "died at 0x2EBD55 with its only copy overwritten at 0x2EBE96. There is nothing to\n"
                 "restore, so a discard path is not writable at that site.\n"
                 "WHY NOT TOUCH xmmword_1811FC030: it looks like the clip plane and is not. Its only\n"
                 "consumer is 0x2EC2C6 `mulps xmm0,xmm7` -> horizontal sum -> andps sign mask ->\n"
                 "xorps, i.e. the camera forward supplies the SIGN that orients each silhouette plane\n"
                 "outward. Changing it leaves planes inside-out, not the volume wider.\n"
                 "SAFETY: a 0-2 plane record is an UNDER-constrained antiportal volume, so the worker\n"
                 "culls less, which is the direction every flag here wants. The allocator sizes it by\n"
                 "its own arithmetic ((4*(0+0)+71)>>6 = 1 block) and every copy loop at 0x2EC74B is\n"
                 "guarded, so zero counts copy nothing.\n"
                 "DEPENDENCY: a 0-entry record empties site 11's accumulator at 0x2E941C and its\n"
                 "`jz loc_1802E9DB1` would drop the node. rtx.cullOff.worldPortal forces that branch\n"
                 "to accept. With worldPortal off this site would LOSE nodes, hence the AND in code.\n"
                 "VERIFY with rtx.cullOff.probeDispatch: clipDegen should fall to 0 and degenTo empty\n"
                 "(the recorder's branch can no longer be taken; the two coexist because the\n"
                 "retargeted rel32 at 0x2EC677/0x2EC681 does not overlap these cmp bytes), and a20\n"
                 "should stop dipping at pitch -45..-50.");
      RTX_OPTION("rtx.cullOff", bool, probeWorldDrain, false,
                 "Diagnostic, not a fix. Wraps client.dll+0x2F04F0 (sub_1802F04F0, the DRAIN that\n"
                 "ORs accepted leaf runs into the world visibility mask) and logs one [DrainProbe]\n"
                 "line per frame carrying pitchDeg, alongside [JobProbe]/[PitchProbe].\n"
                 "The drain applies NO test on any path — its only store is `or [r9], rdx` — so a\n"
                 "popcount of the mask AFTER it returns IS the accepted-leaf count.\n"
                 "  m1/m2/r   popcounts of the world / static-prop / renderable regions\n"
                 "  m1Max     per-frame max, since the drain runs once per view and a sum alone\n"
                 "            cannot separate one big view from several small ones\n"
                 "  layoutOk  structural check (c70<c74<c78 and c7C==c78+c80). If this is 0 the\n"
                 "            run is NOT interpretable — the mask pointer was stale or unrelated.\n"
                 "READ IT, binned by pitch (magnitudes, not Pearson r):\n"
                 "  m1/m2 FALL with pitch => the §2 reject set is not exhaustive after all and the\n"
                 "    worker is still culling. Go back into sub_1802E8DA0.\n"
                 "  m1/m2 FLAT with pitch => the mask is fully populated and the loss is entirely\n"
                 "    downstream of the visibility build. Stop looking at this subsystem.\n"
                 "Pairs with probeWorldJobs, which already showed the INPUT is flat (192->184\n"
                 "calls/frame, non-monotonic) while instances fall 23% (r=-0.77) — which is what\n"
                 "excluded sub_1802EB620 and made the output the remaining question.");
      RTX_OPTION("rtx.cullOff", bool, staticPropFade, false,
                 "Disable the STATIC-PROP fade cull in engine.dll\n"
                 "(CStaticPropMgr::GatherVisibleStaticProps, sub_1801B2200). Separate module and\n"
                 "separate cull from rtx.cullOff.distanceFade, which only reaches client.dll\n"
                 "renderables — static props never pass through BuildRenderableRenderLists, so no\n"
                 "client.dll patch affects them.\n"
                 "That gather's only per-prop reject is dist^2 vs a per-prop fade distance, and the\n"
                 "fade distance is derived from the model's BOUNDING RADIUS:\n"
                 "  fadeDist = max(radius * model_defaultFadeDistScale (40), model_defaultFadeDistMin (400))\n"
                 "so a large object assembled from small instanced pieces is culled on the size of a\n"
                 "piece, not on what it looks like. This is the flag for props that vanish at range\n"
                 "while distanceFade is already on.\n"
                 "Patches the reject branch AND the partial-fade band together — the band computes a\n"
                 "negative alpha past fadeEnd if only the reject is removed.\n"
                 "Note the engine caps the gathered list at 16320 props per view and then stops\n"
                 "walking, the same way the render list truncates at 4096 renderables.");
      RTX_OPTION("rtx.cullOff", bool, staticPropFrustum, true,
                 "Disable the STATIC-PROP FRUSTUM cull in engine.dll (sub_1801B2DD0 for the main\n"
                 "view, sub_1801B2FA0 for shadow views). This is the view-direction cull for props —\n"
                 "distinct from staticPropFade, which is distance only, and from frustum, which only\n"
                 "reaches client.dll renderables.\n"
                 "This is the flag for 'I look down and the scenery disappears'. Measured with\n"
                 "[WorldVis] over 59 main-view frames binned by camera pitch: the renderable mask is\n"
                 "FLAT with pitch (r = +0.34, the client.dll patches work), while the static-prop\n"
                 "mask collapses from 906 bits at 0-10deg to 23 at 50deg+ (r = -0.95).\n"
                 "It was missed for so long because it is dispatched by pointer, not called:\n"
                 "sub_1801A8350 hands the prop slice of the visibility bitmask to\n"
                 "StaticPropMgrClient005 vtable+152 (main) / +160 (shadow) together with *(a2+24) —\n"
                 "the same frustum planes the renderable cull uses. No client.dll patch and no scan\n"
                 "for the +0x54088 displacement can see it.\n"
                 "Patches the bit-clear STORE in each function rather than a branch, so the plane\n"
                 "test still runs and only its verdict is dropped; both callers discard the return\n"
                 "value, so nothing downstream reads a partial result.\n"
                 "SCOPE: props only. World surfaces are a separate mask with a separate cull —\n"
                 "see rtx.cullOff.worldFrustum.");
      RTX_OPTION("rtx.cullOff", bool, worldFrustum, true,
                 "Disable the WORLD-GEOMETRY frustum cull in client.dll sub_1802E8DA0. This is the\n"
                 "one that removes walls/terrain/ground when you look away, and it lives in a\n"
                 "subsystem no other flag here touches.\n"
                 "sub_1802EE7D0 dispatches to FIVE visibility builders on *(a2); the main view uses\n"
                 "selector 0 -> sub_1802EF090 -> sub_1802EB620 (confirmed live in x64dbg). pvs_debug\n"
                 "is read only by sub_1802EB290/sub_1802ECFC0/sub_1802F0870, none of which are on\n"
                 "that path — which is why forcing pvs_debug|2 never changed anything.\n"
                 "sub_1802E8DA0 run-length encodes the ACCEPTED leaves into 28 per-thread buckets\n"
                 "that sub_1802F04F0 ORs into the mask, so a rejected leaf is never added rather\n"
                 "than being cleared: the reject path (loc_1802E9CA2) flushes the open run and sets\n"
                 "runStart = i+1, while the accept path (loc_1802E9CF7) does nothing at all.\n"
                 "The cull is TWO-LEVEL and both must be patched or nothing changes. The same\n"
                 "AABB-vs-plane test runs on NODES at loc_1802E8F50, and its reject tails skip to\n"
                 "the next node without ever entering the leaf loop. Patching only the leaf level\n"
                 "measured as completely inert (m1 still 499 -> 54 across pitch) because those\n"
                 "sites only run for nodes that already passed. A non-pausing hit counter on\n"
                 "sub_1802E8DA0 is what proved it: calls/frame fell only 33% (197.7 -> 132.6\n"
                 "between 19.9deg and 89deg) while m1 fell ~10x — same jobs, far less work each.\n"
                 "Removes VIEW-DIRECTION culling only. The PVS/area node bitmask and the outer loops\n"
                 "are untouched, so this is much cheaper than rtx.cullOff.pvs, which submits the\n"
                 "whole map regardless of where you are.\n"
                 "Expect world draw counts to rise substantially when looking at open sightlines —\n"
                 "everything in the current PVS set is now submitted regardless of facing.");
      RTX_OPTION("rtx.cullOff", bool, visibilityMask, false,
                 "Stop ANDing the precomputed per-renderable visibility masks (leafSys+2104 main view,\n"
                 "+3128 for 0xC views) into the render list. Default off: the producer of those masks\n"
                 "has not been identified, so what exactly they gate is unproven. Turning this on can\n"
                 "only ADD renderables, never remove them.");
      RTX_OPTION("rtx.cullOff", bool, pvs, false,
                 "Force the render-list seed bitmask to all-ones, i.e. consider every registered\n"
                 "renderable regardless of PVS/area visibility (the engine's own null check on each\n"
                 "renderable is left intact). Default off, and expensive: this is the one that stops\n"
                 "the engine hiding everything behind you and in other areas. The render list is hard\n"
                 "capped at 4096 renderables per view, so past that the engine truncates.");
      RTX_OPTION("rtx.cullOff", bool, studioLod, false,
                 "Force r_lod to 0 so studio models always use LOD0 instead of distance LOD. Distant\n"
                 "models otherwise fall to a near-empty top LOD, which reads as a vanish. Default off\n"
                 "because it multiplies triangle counts, and therefore BLAS build cost.");
      RTX_OPTION("rtx.cullOff", bool, raiseStudioDrawCap, false,
                 "Raise the materialsystem_dx11 per-view studio draw-group cap from 255 to 1023 (the\n"
                 "size its own buffers hold). Relevant here because with the culls off a busy view\n"
                 "exceeds 255 groups easily and the engine silently drops the surplus draws — the same\n"
                 "kind of vanishing this feature exists to remove.\n"
                 "KNOWN REGRESSIVE, hence default off: the existing [DrawCap] patch also NOPs the\n"
                 "convar clamp next to the immediate, and that clamp turned out to be load-bearing —\n"
                 "it caused unrelated missing geometry on views that legitimately exceed 255 groups.\n"
                 "Only reach for this if geometry starts dropping AFTER the culls are off, and expect\n"
                 "to have to split the two halves of that patch first.");
    };

    // NV-DXVK [SceneCull]: Remix-side replacement for the engine's culling.
    //
    // The engine's own culls (client.dll BuildRenderableRenderLists: PVS +
    // per-renderable frustum + distance fade; studio LOD) are wrong for a path
    // tracer, because off-screen geometry still contributes shadows, reflections
    // and GI. d3d11_rtx.cpp's [CullOff] patches turn them off; this replaces them
    // with a scope limit we control.
    //
    // WHERE THIS RUNS AND WHAT IT COSTS: the test lives in
    // AccelManager::mergeInstancesIntoBlas, immediately before the existing
    // `mask == 0` early-out, and culls by zeroing the instance mask. That point
    // is deliberate:
    //   - the BlasEntry has already been touched this frame by SceneManager, so
    //     the BLAS and its geometry buffers stay resident (scene GC keeps them
    //     for numFramesToKeepBLAS frames after the last touch) and a culled
    //     object costs nothing to bring back;
    //   - the instance never enters a BLAS bucket, so no BLAS build/refit runs
    //     for it, it gets no TLAS entry and no surface, and no ray can hit it;
    //   - it does NOT save the CPU cost of the draw call itself (SubmitDraw,
    //     geometry extraction, hashing, instance update) — that work happens
    //     upstream. Cull here to save GPU/BVH work, not draw-submission CPU.
    // Zeroing the mask (rather than skipping the instance) is also required for
    // correctness: instances merged into a shared BLAS have their geometry baked
    // into it and the bucket mask is the OR of its members, so the mask must be
    // cleared before bucketing. Same reasoning as perfCullInstancesLargerThan.
    //
    // CAVEAT worth knowing before turning frustum culling on: anything culled
    // stops casting shadows into the view and stops appearing in reflections and
    // indirect light. frustumMargin exists to buy back the near-edge cases; the
    // radius cull is the safe one, since distant geometry contributes least.
    struct SceneCull {
      friend class ImGUI;
      friend class RtxOptions;
      // RESTRUCTURED 2026-08-06 to RT_CULLING_2026-08-05.md's §1A shape: a UNION
      // OF KEEPS with no direction-keyed reject. An instance is kept if ANY
      // enabled keep covers it (frustum = directly visible, radius = near-camera
      // reflections/GI, lightInfluence = shadow casters), and culled only when
      // every keep misses. A bug in a keep term over-keeps (perf); it cannot
      // make an occluder vanish. The pre-2026-08-06 form was reject-based and
      // its frustum reject was §1-unsound (dropped off-screen shadow casters).
      RTX_OPTION("rtx.sceneCull", bool, enable, false,
                 "Master switch for Remix-side culling of ray-traced instances, structured as a\n"
                 "union of keeps: an instance is kept if the frustum keep, the radius keep, or the\n"
                 "light-influence keep covers it, and culled only when every enabled keep misses.\n"
                 "Culls by zeroing the instance mask in mergeInstancesIntoBlas, which keeps the BLAS\n"
                 "and geometry resident (no rebuild when the object returns) but removes the instance\n"
                 "from the TLAS. Intended to replace the engine culling disabled by rtx.cullOff.*.\n"
                 "If no keep term is enabled the cull is inert (nothing is removed), never cull-all.");
      RTX_OPTION("rtx.sceneCull", float, radius, 0.0f,
                 "KEEP term: instances whose world-space bounding box comes within this many world\n"
                 "units of the main camera are kept, in every direction — this is what preserves\n"
                 "behind-the-camera geometry for mirrors and GI. 0 disables the term. Measured to\n"
                 "the closest point of the box, so the camera being inside a large box keeps it.");
      RTX_OPTION("rtx.sceneCull", bool, frustumEnable, true,
                 "KEEP term: instances whose world-space bounding box touches the (margin-widened)\n"
                 "main camera frustum are kept — case #1, directly visible. No longer a reject: an\n"
                 "instance outside the frustum survives if the radius or light-influence keep covers\n"
                 "it. Boxes wholly behind the near plane do not count as touching.");
      RTX_OPTION("rtx.sceneCull", float, frustumMargin, 0.25f,
                 "Fractional widening of the left/right/top/bottom frustum planes for the frustum\n"
                 "keep, e.g. 0.25 keeps anything within 125% of the screen extents.");
      RTX_OPTION("rtx.sceneCull", bool, frustumCullBehindCamera, false,
                 "OBSOLETE since the 2026-08-06 keep-union restructure; no longer read. Behind-camera\n"
                 "geometry is kept exactly when the radius or light-influence keep covers it.");
      RTX_OPTION("rtx.sceneCull", bool, lightInfluenceEnable, true,
                 "KEEP term (RT_CULLING doc §2.3): keep off-screen instances that can intercept light\n"
                 "headed for the visible region — shadow casters, case #2, and the ONLY sound answer\n"
                 "for the sun. Per light, the visible-bounds volume is extruded toward the light (the\n"
                 "cascaded-shadow-map trick) and instances inside the extrusion are kept: for a\n"
                 "distant/sun light the sweep runs lightInfluenceSunLength along BOTH senses of the\n"
                 "light axis (sign convention deliberately unverified — one-way with the wrong sign\n"
                 "would cull real sun occluders and recreate the light leak; both ways only\n"
                 "over-keeps), for positioned lights the sweep runs camera-to-light. The visible\n"
                 "bounds are the camera-centered cube of half-extent visibleRange — a superset of\n"
                 "the capped frustum and rotation-invariant, so the keep set only moves with\n"
                 "position. Every approximation errs toward keeping; this term cannot hide geometry,\n"
                 "only spend performance. Watch keptLight on [SceneCull] logStats.");
      RTX_OPTION("rtx.sceneCull", float, visibleRange, 0.0f,
                 "Half-extent, in world units, of the camera-centered cube standing in for 'the\n"
                 "visible region' in the light-influence keep. 0 = use rtx.sceneCull.radius, or\n"
                 "50000 if that is also unset. Smaller = tighter extrusions = more culled; must\n"
                 "still cover the far geometry you can actually see.");
      RTX_OPTION("rtx.sceneCull", float, lightInfluenceSunLength, 100000.0f,
                 "Sweep length, in world units, of the visible-bounds extrusion along a distant\n"
                 "(sun) light's axis, applied in both senses. Clamped up to visibleRange. Too short\n"
                 "culls far sun occluders (mountain shadows vanish); too long only costs perf.");
      RTX_OPTION("rtx.sceneCull", bool, solidAngleCull, true,
                 "REJECT term, the only one (RT_CULLING doc §2.2, far-field occluder bound): cull\n"
                 "kept OFF-SCREEN instances whose apparent angular size — longest world-box axis\n"
                 "over distance to the box — is below solidAngleMin. On-screen (frustum-kept)\n"
                 "instances are never rejected, so nothing visible can pop. Sound because dome/sky\n"
                 "occlusion scales with the SOLID ANGLE an occluder subtends (quadratic in angular\n"
                 "size) and strong occlusion is local to the occluder, so the on-screen error of\n"
                 "culling it is bounded by a small multiple of its own angular size — this is what\n"
                 "actually culls on sky/dome-lit maps (e.g. TF2's BT mission: light table empty,\n"
                 "light keep covers everything). Skinned instances are exempt (their box does not\n"
                 "bound the drawn surface), and so is anything within solidAngleLightExemptRadius\n"
                 "of a positioned light (penumbra magnification: a small occluder near a light\n"
                 "casts a large shadow). Known limit: many sub-threshold occluders can aggregate;\n"
                 "draw-batch AABBs blunt this, and the fix for a visibly missing aggregate shadow\n"
                 "is lowering solidAngleMin.");
      RTX_OPTION("rtx.sceneCull", float, solidAngleMin, 0.01f,
                 "Apparent-size threshold for the off-screen solid-angle reject, in radians. 0.01\n"
                 "is ~5 pixels at 1080p/90FOV; as an off-screen occluder that is ~0.001% of the\n"
                 "hemisphere's light, well under perception. An object 10,000 units away is culled\n"
                 "only if its longest axis is under ~100 units. Raise for more culling, lower if\n"
                 "an aggregate shadow (forest, fence line) visibly thins; 0 disables the reject.");
      RTX_OPTION("rtx.sceneCull", float, solidAngleLightExemptRadius, 1000.0f,
                 "Solid-angle reject exemption: keep any instance whose box is within this many\n"
                 "world units of a positioned (non-distant) light, however small it looks — near a\n"
                 "light, a small occluder throws a large shadow.");
      RTX_OPTION("rtx.sceneCull", bool, logStats, false,
                 "Log one [SceneCull] line per second: instances tested, which keep term covered\n"
                 "them first (keptFrustum/keptRadius/keptLight/keptSkinned), how many the keep\n"
                 "union culled (culled), and how many the solid-angle reject removed (culledSmall).");
    };

    // Resolve Options
    // Todo: Potentially document that after a number of resolver interactions is exhausted the next interaction will be treated as a hit regardless.
    RTX_OPTION_ARGS("rtx", uint8_t, primaryRayMaxInteractions, 32,
               "The maximum number of resolver interactions to use for primary (initial G-Buffer) rays.\n"
               "This affects how many Decals, Ray Portals and potentially particles (if unordered approximations are not enabled) may be interacted with along a ray at the cost of performance for higher amounts of interactions.",
               args.minValue = static_cast<uint8_t>(1), args.maxValue = std::numeric_limits<uint8_t>::max());
    RTX_OPTION_ARGS("rtx", uint8_t, psrRayMaxInteractions, 32,
               "The maximum number of resolver interactions to use for PSR (primary surface replacement G-Buffer) rays.\n"
               "This affects how many Decals, Ray Portals and potentially particles (if unordered approximations are not enabled) may be interacted with along a ray at the cost of performance for higher amounts of interactions.",
               args.minValue = static_cast<uint8_t>(1), args.maxValue = std::numeric_limits<uint8_t>::max());
    RTX_OPTION_ARGS("rtx", uint8_t, secondaryRayMaxInteractions, 8,
               "The maximum number of resolver interactions to use for secondary (indirect) rays.\n"
               "This affects how many Decals, Ray Portals and potentially particles (if unordered approximations are not enabled) may be interacted with along a ray at the cost of performance for higher amounts of interactions.\n"
               "This value is recommended to be set lower than the primary/PSR max ray interactions as secondary ray interactions are less visually relevant relative to the performance cost of resolving them.",
               args.minValue = static_cast<uint8_t>(1), args.maxValue = std::numeric_limits<uint8_t>::max());
    RTX_OPTION("rtx", bool, enableSeparateUnorderedApproximations, true,
               "Use a separate loop during resolving for surfaces which can have lighting evaluated in an approximate unordered way on each path segment (such as particles).\n"
               "This improves performance typically in how particles or decals are rendered and should usually always be enabled.\n"
               "Do note however the unordered nature of this resolving method may result in visual artifacts with large numbers of stacked particles due to difficulty in determining the intended order.\n"
               "Additionally, unordered approximations will only be done on the first indirect ray bounce (as particles matter less in higher bounces), and only if enabled by its corresponding setting.");
    RTX_OPTION("rtx", bool, trackParticleObjects, true, "Track last frame's corresponding particle object.");
    RTX_OPTION_ENV("rtx", bool, enableDirectTranslucentShadows, false, "RTX_ENABLE_DIRECT_TRANSLUCENT_SHADOWS", "Calculate coloured shadows for translucent materials (i.e. glass, water) in direct lighting. In engineering terms: include OBJECT_MASK_TRANSLUCENT into primary visibility rays.");
    RTX_OPTION_ENV("rtx", bool, enableDirectAlphaBlendShadows, true, "RTX_ENABLE_DIRECT_ALPHABLEND_SHADOWS", "Calculate shadows for semi-transparent materials (alpha blended) in direct lighting. In engineering terms: include OBJECT_MASK_ALPHA_BLEND into primary visibility rays.");
    RTX_OPTION_ENV("rtx", bool, enableIndirectTranslucentShadows, false, "RTX_ENABLE_INDIRECT_TRANSLUCENT_SHADOWS", "Calculate coloured shadows for translucent materials (i.e. glass, water) in indirect lighting (i.e. reflections and GI). In engineering terms: include OBJECT_MASK_TRANSLUCENT into secondary visibility rays.");
    RTX_OPTION_ENV("rtx", bool, enableIndirectAlphaBlendShadows, true, "RTX_ENABLE_INDIRECT_ALPHABLEND_SHADOWS", "Calculate shadows for semi-transparent (alpha blended) objects in indirect lighting (i.e. reflections and GI). In engineering terms: include OBJECT_MASK_ALPHA_BLEND into secondary visibility rays.");

    public: static void resolveTransparencyThresholdOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", float, resolveTransparencyThreshold, 1.0f / 255.0f, "A threshold for which any opacity value below is considered totally transparent and may be safely skipped without as significant of a performance cost.",
               args.minValue = 0.0f, args.maxValue = 1.0f, args.onChangeCallback = &resolveTransparencyThresholdOnChange);
    RTX_OPTION_ARGS("rtx", float, resolveOpaquenessThreshold, 254.0f / 255.0f, "A threshold for which any opacity value above is considered totally opaque.",
               args.minValue = 0.0f, args.maxValue = 1.0f);

    // PSR Options
    RTX_OPTION("rtx", bool, enablePSRR, true,
               "A flag to enable or disable reflection PSR (Primary Surface Replacement).\n"
               "When enabled this feature allows higher quality mirror-like reflections in special cases by replacing the G-Buffer's surface with the reflected surface.\n"
               "Should usually be enabled for the sake of quality as almost all applications will utilize it in the form of glass or mirrors.");
    RTX_OPTION("rtx", bool, enablePSTR, true,
               "A flag to enable or disable transmission PSR (Primary Surface Replacement).\n"
               "When enabled this feature allows higher quality glass-like refraction in special cases by replacing the G-Buffer's surface with the refracted surface.\n"
               "Should usually be enabled for the sake of quality as almost all applications will utilize it in the form of glass.");
    RTX_OPTION_ARGS("rtx", uint8_t, psrrMaxBounces, 10,
               "The maximum number of Reflection PSR bounces to traverse. Must be 15 or less due to payload encoding.\n"
               "Should be set higher when many mirror-like reflection bounces may be needed, though more bounces may come at a higher performance cost.",
               args.minValue = static_cast<uint8_t>(1), args.maxValue = static_cast<uint8_t>(254));
    RTX_OPTION_ARGS("rtx", uint8_t, pstrMaxBounces, 10,
               "The maximum number of Transmission PSR bounces to traverse. Must be 15 or less due to payload encoding.\n"
               "Should be set higher when refraction through many layers of glass may be needed, though more bounces may come at a higher performance cost.",
               args.minValue = static_cast<uint8_t>(1), args.maxValue = static_cast<uint8_t>(254));
    RTX_OPTION("rtx", bool, enablePSTROutgoingSplitApproximation, true,
               "Enable transmission PSR on outgoing transmission events such as leaving translucent materials (rather than respecting no-split path PSR rule).\n"
               "Typically this results in better looking glass when enabled (at the cost of accuracy due to ignoring non-TIR inter-reflections within the glass itself).");
    RTX_OPTION("rtx", bool, enablePSTRSecondaryIncidentSplitApproximation, true,
               "Enable transmission PSR on secondary incident transmission events such as entering a translucent material on an already-transmitted path (rather than respecting no-split path PSR rule).\n"
               "Typically this results in better looking glass when enabled (at the cost accuracy due to ignoring reflections off of glass seen through glass for example).");
    
    // Note: In a more technical sense, any PSR reflection or transmission from a surface with "normal detail" greater than the specified value will generate a 1.0 in the
    // disocclusionThresholdMix mask, indicating that the alternate disocclusion threshold in the denoiser should be used.
    // A value of 0 is a valid setting as it means that any detail at all, no matter how small, will set that mask bit (e.g. any usage of a normal map deviating from from the
    // underlying normal).
    RTX_OPTION("rtx", float, psrrNormalDetailThreshold, 0.0f,
               "A threshold value to indicate that the denoiser's alternate disocclusion threshold should be used when normal map \"detail\" on a reflection PSR surface exceeds a desired amount.\n"
               "Normal detail is defined as 1-dot(tangent_normal, vec3(0, 0, 1)), or in other words it is 0 when no normal mapping is used, and 1 when the normal mapped normal is perpendicular to the underlying normal.\n"
               "This is typically used to reduce flickering artifacts resulting from reflection on surfaces like glass leveraging normal maps as often the denoiser is too aggressive with disocclusion checks frame to frame when DLSS or other camera jittering is in use.");
    RTX_OPTION("rtx", float, pstrNormalDetailThreshold, 0.0f,
               "A threshold value to indicate that the denoiser's alternate disocclusion threshold should be used when normal map \"detail\" on a transmission PSR surface exceeds a desired amount.\n"
               "Normal detail is defined as 1-dot(tangent_normal, vec3(0, 0, 1)), or in other words it is 0 when no normal mapping is used, and 1 when the normal mapped normal is perpendicular to the underlying normal.\n"
               "This is typically used to reduce flickering artifacts resulting from refraction on surfaces like glass leveraging normal maps as often the denoiser is too aggressive with disocclusion checks frame to frame when DLSS or other camera jittering is in use.");

    // Shader Execution Reordering Options
    RTX_OPTION_ENV("rtx", bool, isShaderExecutionReorderingSupported, true, "DXVK_IS_SHADER_EXECUTION_REORDERING_SUPPORTED", "Enables Shader Execution Reordering (SER) if it is supported by the target HW and SW."); 
    // True if `isShaderExecutionReorderingSupported` is true and the computer actually supports it.
    public: static inline bool enableShaderExecutionReordering = true;
    RTX_OPTION("rtx", bool, enableShaderExecutionReorderingInPathtracerGbuffer, false, "(Note: Hard disabled in shader code) Enables Shader Execution Reordering (SER) in GBuffer Raytrace pass if SER is supported.");
    RTX_OPTION("rtx", bool, enableShaderExecutionReorderingInPathtracerIntegrateIndirect, true, "Enables Shader Execution Reordering (SER) in Integrate Indirect pass if SER is supported.");

    // Path Options
    RTX_OPTION("rtx", bool, enableRussianRoulette, true,
               "A flag to enable or disable Russian Roulette, a rendering technique to give paths a chance of terminating randomly with each bounce based on their importance.\n"
               "This is usually useful to have enabled as it will ensure useless paths are terminated earlier while more important paths are allowed to accumulate more bounces.\n"
               "Furthermore this allows for the renderer to remain unbiased whereas a hard clamp on the number of bounces will introduce bias (though this is also done in Remix for the sake of performance).\n"
               "On the other hand, randomly terminating paths too aggressively may leave threads in GPU warps without work which may hurt thread occupancy when not used with a thread-reordering technique like SER.\n"
               "Additionally, Russian Roulette will always for the most part increase variance and will reduce the average path depth from whatever the current maximum path length is set to.\n"
               "This increase in variance will slightly impact image quality especially on scenes relying heavily on many bounces of indirect lighting, but this is usually worth it for efficiency purposes, as Russian Roulette allows each ray to reduces variance more than it would otherwise.");
    RTX_OPTION_ARGS("rtx", RussianRouletteMode, russianRouletteMode, RussianRouletteMode::ThroughputBased, "Russian Roulette Mode. Throughput Based: paths with higher throughput become longer; Specular Based: specular paths become longer.\n",
                    args.environment = "DXVK_PATH_TRACING_RR_MODE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", float, russianRouletteDiffuseContinueProbability, 0.1f, "The probability of continuing a diffuse path when Russian Roulette is being used. Only apply to specular based mode.\n");
    RTX_OPTION("rtx", float, russianRouletteSpecularContinueProbability, 0.98f, "The probability of continuing a specular path when Russian Roulette is being used. Only apply to specular based mode.\n");
    RTX_OPTION("rtx", float, russianRouletteDistanceFactor, 0.1f, "Path segments whose distance proportion are under this threshold are more likely to continue. Only apply to specular based mode.\n");
    RTX_OPTION_ARGS("rtx", float, russianRouletteMaxContinueProbability, 0.9f,
               "The maximum probability of continuing a path when Russian Roulette is being used.\n"
               "This ensures all rays have a small probability of terminating each bounce, mostly to prevent infinite paths in perfectly reflective mirror rooms (though the maximum path bounce count will also ensure this).",
               args.minValue = 0.0f, args.maxValue = 1.0f,
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", float, russianRoulette1stBounceMinContinueProbability, 0.6f,
               "The minimum probability of continuing a path when Russian Roulette is being used on the first bounce.\n"
               "This ensures that on the first bounce rays are not terminated too aggressively as it may be useful for some denoisers to have a contribution even if it is a relatively unimportant one rather than a missing indirect sample.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", float, russianRoulette1stBounceMaxContinueProbability, 1.0f,
               "The maximum probability of continuing a path when Russian Roulette is being used on the first bounce.\n"
               "This is similar to the usual max continuation probability for Russian Roulette, but specifically only for the first bounce.");
    public: static void pathMinBouncesOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", uint8_t, pathMinBounces, 1,
                   "The minimum number of indirect bounces the path must complete before Russian Roulette can be used. Must be < 16.\n"
                   "This value is recommended to stay fairly low (1 for example) as forcing longer paths when they carry little contribution quickly becomes detrimental to performance.",
                   args.environment = "DXVK_PATH_TRACING_MIN_BOUNCES",
                   args.minValue = static_cast<uint8_t>(0), args.maxValue = static_cast<uint8_t>(15),
                   args.onChangeCallback = &pathMinBouncesOnChange,
                   args.flags = RtxOptionFlags::UserSetting);
    public: static void pathMaxBouncesOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", uint8_t, pathMaxBounces, 4,
                   "The maximum number of indirect bounces the path will be allowed to complete. Must be < 16.\n"
                   "Higher values result in better indirect lighting quality due to biasing the signal less, lower values result in better performance.\n"
                   "Very high values are not recommended however as while long paths may be technically needed for unbiased rendering, in practice the contributions from higher bounces have diminishing returns.",
                   args.environment = "DXVK_PATH_TRACING_MAX_BOUNCES",
                   args.minValue = static_cast<uint8_t>(0), args.maxValue = static_cast<uint8_t>(15),
                   args.onChangeCallback = &pathMaxBouncesOnChange,
                   args.flags = RtxOptionFlags::UserSetting);
    // Note: Use caution when adjusting any zero thresholds as values too high may cause entire lobes of contribution to be missing in material edge cases. For example
    // with translucency, a zero threshold on the specular lobe of 0.05 removes the entire contribution when viewing straight on for any glass with an IoR below 1.58 or so
    // which can be paticularly noticable in some scenes. To bias sampling more in the favor of one lobe the min probability should be used instead, but be aware this will
    // end up wasting more samples in some cases versus pure importance sampling (but may help denoising if it cannot deal with super sparse signals).
    RTX_OPTION("rtx", float, opaqueDiffuseLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero opaque diffuse probability weight values.");
    RTX_OPTION_ARGS("rtx", float, minOpaqueDiffuseLobeSamplingProbability, 0.25f, "The minimum allowed non-zero value for opaque diffuse probability weights.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", float, opaqueSpecularLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero opaque specular probability weight values.");
    RTX_OPTION_ARGS("rtx", float, minOpaqueSpecularLobeSamplingProbability, 0.25f, "The minimum allowed non-zero value for opaque specular probability weights.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", float, opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero opaque opacity probability weight values.");
    RTX_OPTION("rtx", float, minOpaqueOpacityTransmissionLobeSamplingProbability, 0.25f, "The minimum allowed non-zero value for opaque opacity probability weights.");
    RTX_OPTION("rtx", float, opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero thin opaque diffuse transmission probability weight values.");
    RTX_OPTION("rtx", float, minOpaqueDiffuseTransmissionLobeSamplingProbability, 0.25f, "The minimum allowed non-zero value for thin opaque diffuse transmission probability weights.");
    // Note: 0.01 chosen as mentioned before to avoid cutting off reflection lobe on most common types of glass when looking straight on (a base reflectivity
    // of 0.01 corresponds to an IoR of 1.22 or so). Avoid changing this default without good reason to prevent glass from losing its reflection contribution.
    RTX_OPTION("rtx", float, translucentSpecularLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero translucent specular probability weight values.");
    RTX_OPTION("rtx", float, minTranslucentSpecularLobeSamplingProbability, 0.3f, "The minimum allowed non-zero value for translucent specular probability weights.");
    RTX_OPTION("rtx", float, translucentTransmissionLobeSamplingProbabilityZeroThreshold, 0.01f, "The threshold for which to zero translucent transmission probability weight values.");
    RTX_OPTION("rtx", float, minTranslucentTransmissionLobeSamplingProbability, 0.25f, "The minimum allowed non-zero value for translucent transmission probability weights.");

    RTX_OPTION("rtx", float, indirectRaySpreadAngleFactor, 0.05f,
               "A tuning factor applied to the spread angle calculated from the sampled lobe solid angle PDF. Should be 0-1.\n"
               "This scaled spread angle is used to widen a ray's cone angle after indirect lighting BRDF samples to essentially prefilter the effects of the BRDF lobe's spread which potentially may reduce noise from indirect rays (e.g. reflections).\n"
               "Prefiltering will overblur detail however compared to the ground truth of casting multiple samples especially given this calculated spread angle is a basic approximation and ray cones to begin with are a simple approximation for ray pixel footprint.\n"
               "As such rather than using the spread angle fully this spread angle factor allows it to be scaled down to something more narrow so that overblurring can be minimized. Similarly, setting this factor to 0 disables this cone angle widening feature.");
    RTX_OPTION("rtx", bool, rngSeedWithFrameIndex, true,
               "Indicates that pseudo-random number generator should be seeded with the frame number of the application every frame, otherwise seed with 0.\n"
               "This should generally always be enabled as without the frame index each frame will typically be identical in the random values that are produced which will result in incorrect rendering. Only meant as a debugging tool.");
    RTX_OPTION_ARGS("rtx", bool, enableFirstBounceLobeProbabilityDithering, true,
               "A flag to enable or disable screen-space probability dithering on the first indirect lobe sampled.\n"
               "Generally sampling a diffuse, specular or other lobe relies on a random number generated against the probability of sampling each lobe, effectively focusing more rays/paths on lobes which matter more.\n"
               "This can cause issues however with denoisers which do not handle sparse stochastic signals (like those from path tracing) well as they may be expecting a more \"complete\" signal like those used in simpler branching ray tracing setups.\n"
               "To help solve this issue this option uses a temporal screenspace dithering based on the probability rather than a purely random choice to determine which lobe to sample from on the first indirect bounce.\n"
               "This as a result helps ensure there will always be a diffuse or specular sample within the dithering pattern's area and should help the denoising resolve a more stable result.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableUnorderedResolveInIndirectRays, true,
               "A flag to enable or disable unordered resolve approximations in indirect rays.\n"
               "This allows for the presence of unordered approximations in resolving to be overridden in indirect rays and as such requires separate unordered approximations to be enabled to have any effect.\n"
               "This option should be enabled if objects which can be resolvered in an unordered way in indirect rays are expected for higher quality in reflections, but may come at a performance cost.\n"
               "Note that even with this option enabled, unordered resolve approximations are only done on the first indirect bounce for the sake of performance overall.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, enableProbabilisticUnorderedResolveInIndirectRays, true,
               "A flag to enable or disable probabilistic unordered resolve approximations in indirect rays.\n"
               "This flag speeds up the unordered resolve for indirect rays by probabilistically deciding when to perform unordered resolve or not.  Must have both unordered resolve and unordered resolve in indirect rays enabled for this to take effect.\n"
               "This option should be enabled by default as it can significantly improve performance on some hardware.  In rare cases it may come at the cost of some quality for particles and decals in reflections.\n"
               "Note that even with this option enabled, unordered resolve approximations are only done on the first indirect bounce for the sake of performance overall.");
    RTX_OPTION_ARGS("rtx", bool, enableUnorderedEmissiveParticlesInIndirectRays, false,
                   "A flag to enable or disable unordered resolve emissive particles specifically in indirect rays.\n"
                   "Should be enabled in higher quality rendering modes as emissive particles are fairly important in reflections, but may be disabled to skip such interactions which can improve performance on lower end hardware.\n"
                   "Note that rtx.enableUnorderedResolveInIndirectRays must first be enabled for this option to take any effect (as it will control if unordered resolve is used to begin with in indirect rays).",
                   args.environment = "DXVK_EMISSIVE_INDIRECT_PARTICLES",
                   args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableTransmissionApproximationInIndirectRays, false,
               "A flag to enable transmission approximations in indirect rays.\n"
               "Translucent objects hit by indirect rays will not alter ray direction, just change the ray throughput.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, enableDecalMaterialBlending, true,
               "A flag to enable or disable material blending on decals.\n"
               "This should generally always be enabled when decals are in use as this allows decals to be blended down on to the surface they sit slightly above which results in more convincing decals rendering.");

    RTX_OPTION("rtx", bool, enableBillboardOrientationCorrection, true, "");
    RTX_OPTION("rtx", bool, useIntersectionBillboardsOnPrimaryRays, false, "");
    RTX_OPTION("rtx", float, translucentDecalAlbedoFactor, 10.0f,
               "A global scale factor applied to the albedo of decals that are applied to a translucent base material, to make the decals more visible.\n"
               "This is generally needed as albedo values for decals may be fairly low when dealing with opaque surfaces, but the translucent diffuse layer requires a fairly high albedo value to result in an expected look.\n"
               "The need for this option could be avoided by simply authoring decals applied to translucent materials with a higher albedo to begin with, but sometimes applications may share decals between different material types.");

    RTX_OPTION("rtx", float, worldSpaceUiBackgroundOffset, -0.01f, "Distance along normal to offset objects rendered as worldspace UI, specifically for the background of screens.");

    // Light Selection/Sampling Options
    RTX_OPTION_ARGS("rtx", uint16_t, risLightSampleCount, 7,
               "The number of lights randomly selected from the global pool to consider when selecting a light with RIS.\n"
               "Higher values generally increases the quality of RIS light sampling, but also has diminishing returns and higher performance cost past a point.\n"
               "Note that RIS is only used when RTXDI is disabled for direct lighting, or for light sampling in indirect rays, so the impact of this effect will vary.",
               args.minValue = static_cast<uint16_t>(1), args.maxValue = std::numeric_limits<uint16_t>::max());

    // Subsurface Scattering
    struct SubsurfaceScattering {
      friend class RtxOptions;
      friend class ImGUI;

      RTX_OPTION("rtx.subsurface", bool, enableThinOpaque, true, "Enable thin opaque material. The materials withthin opaque properties will fallback to normal opaque material.");
      RTX_OPTION("rtx.subsurface", bool, enableTextureMaps, true, "Enable texture maps such as thickness map or scattering albedo map. The corresponding subsurface properties will fallback to per-material constants if this is disabled.");
      RTX_OPTION("rtx.subsurface", float, surfaceThicknessScale, 1.0f, "Scalar of the subsurface thickness.");
      RTX_OPTION("rtx.subsurface", bool, enableDiffusionProfile, true, "Enable subsurface material. Solve subsurface rendering equation with (burley/SOTO) diffusion profile.");
      RTX_OPTION("rtx.subsurface", float, diffusionProfileScale, 1.0f, "Scalar of the diffusion profile scale.");
      RTX_OPTION("rtx.subsurface", bool, enableTransmission, true, "Enable subsurface transmission. Implement single scattering transmission for thin or curved SSS surface.");
      RTX_OPTION("rtx.subsurface", bool, enableTransmissionSingleScattering, true, "Enable single scattering for subsurface transmission. If this option is disabled, then the refracted ray will not be scattered again inside of the volume.");
      RTX_OPTION("rtx.subsurface", bool, enableTransmissionDiffusionProfileCorrection, false,
        "Enable diffusion profile correction when enabling SSS Transmission.\n"
        "Both burley's diffusion profile and SSS Transmission includes the single scattering energy.\n"
        "The correction removes the single scattering part from diffusion profile to avoid double counting the single scattering energy.");
      RTX_OPTION("rtx.subsurface", bool, enableHeuristicSingleScatteringTransmission, true,
        "Heuristically checks the mean free path (MFP) of SSS materials to determine whether "
        "single scattering transmission should be enabled. Extremely large MFP values usually "
        "indicate thick volumes dominated by high-order scattering, which is already approximated "
        "by the diffusion profile and captures most of the SSS energy. In these cases, the single "
        "scattering contribution can be safely ignored.");
      RTX_OPTION("rtx.subsurface", uint8_t, transmissionBsdfSampleCount, 1, "The sample count for transmission BSDF.(1spp as default)");
      RTX_OPTION("rtx.subsurface", uint8_t, transmissionSingleScatteringSampleCount, 1, "The sample count for every single scattering on BSDF transmission (refracted) ray.(1spp as default)");
      RTX_OPTION("rtx.subsurface", Vector2i, diffusionProfileDebugPixelPosition, Vector2i(INT32_MAX, INT32_MAX), "Pixel position where we show debugging sampling positions for diffusion profile. Requires set debug view to 'SSS Diffusion Profile Sampling'.");
    };

    // Alpha Test/Blend Options
    RTX_OPTION("rtx", bool, enableAlphaBlend, true, "Enable rendering alpha blended geometry, used for partial opacity and other blending effects on various surfaces in many games.");
    RTX_OPTION("rtx", bool, enableAlphaTest, true, "Enable rendering alpha tested geometry, used for cutout style opacity in some games.");
    RTX_OPTION("rtx", bool, enableCulling, true, "Enable front/backface culling for opaque objects. Objects with alpha blend or alpha test are not culled.");
    RTX_OPTION("rtx", bool, enableCullingInSecondaryRays, false, "Enable front/backface culling for opaque objects. Objects with alpha blend or alpha test are not culled.  Only applies in secondary rays, defaults to off.  Generally helps with light bleeding from objects that aren't watertight.");
    RTX_OPTION("rtx", bool, enableEmissiveBlendModeTranslation, true, "Treat incoming semi/additive D3D blend modes as emissive.");
    RTX_OPTION("rtx", bool, enableEmissiveBlendEmissiveOverride, true, "Override typical material emissive information on draw calls with any emissive blending modes to emulate their original look more accurately.");
    RTX_OPTION_ARGS("rtx", float, emissiveBlendOverrideEmissiveIntensity, 0.2f, "The emissive intensity to use when the emissive blend override is enabled. Adjust this if particles for example look overly bright globally.",
               args.minValue = 0.0f, args.maxValue = FLOAT16_MAX);
    RTX_OPTION_ARGS("rtx", float, particleSoftnessFactor, 0.05f, "Multiplier for the view distance that is used to calculate the particle blending range.",
               args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION("rtx", float, forceCutoutAlpha, 0.5f,
               "When an object is added to the cutout textures list it will have a cutout alpha mode forced on it, using this value for the alpha test.\n"
               "This is meant to improve the look of some legacy mode materials using low-resolution textures and alpha blending instead of alpha cutout as this can cause blurry halos around edges due to the difficulty of handling this sort of blending in Remix.\n"
               "Such objects are generally better handled with actual replacement assets using fully opaque geometry replacements or alpha cutout with higher resolution textures, so this should only be relied on until proper replacements can be authored.");
    RTX_OPTION("rtx", float, wboitEnergyLossCompensation, 4.f, "Multiplier for the coverage term in the weighted blended OIT imlementation - allows for some configuration to recover energy loss from the technique.  This is non physical, be careful overtuning ");
    RTX_OPTION("rtx", float, wboitDepthWeightTuning, 2.f, "Allows for tuning the weighted blended OIT depth weight - which can be used to fine tune blending for various circumstances.  This control has a side effect, larger numbers here can adversely affect brightness of emissive blended materials.");
    RTX_OPTION("rtx", bool, wboitEnabled, true, "Enables the new rendering mode for handling alpha blended objects.  Changing this will trigger a shader recompile.  The new mode improves rendering accuracy, especially in cases where there are many layers of transparent things being rendered.");

    // Ray Portal Options
    // Note: Not a set as the ordering of the hashes is important. Keep this list small to avoid expensive O(n) searching (should only have 2 or 4 elements usually).
    // Also must always be a multiple of 2 for proper functionality as each pair of hashes defines a portal connection.
    public: static void rayPortalModelTextureHashesOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", std::vector<XXH64_hash_t>, rayPortalModelTextureHashes, {},
                    "Texture hashes identifying ray portals.\n"
                    "Entries are interpreted as pairs of hashes; the list length must be even and will be clamped to the internal max portal count.",
                    args.onChangeCallback = &rayPortalModelTextureHashesOnChange);
    // Todo: Add option for if a model to world transform matrix should be used or if PCA should be used instead to attempt to guess what the matrix should be (for games with
    // pretransformed Ray Portal vertices).
    // Note: Axes used for orienting the portal when PCA is used.
    RTX_OPTION("rtx", Vector3, rayPortalModelNormalAxis, Vector3(0.0f, 0.0f, 1.0f), "The axis in object space to map the ray portal geometry's normal axis to. Currently unused (as PCA is not implemented).");
    RTX_OPTION("rtx", Vector3, rayPortalModelWidthAxis, Vector3(1.0f, 0.0f, 0.0f), "The axis in object space to map the ray portal geometry's width axis to. Currently unused (as PCA is not implemented).");
    RTX_OPTION("rtx", Vector3, rayPortalModelHeightAxis, Vector3(0.0f, 1.0f, 0.0f), "The axis in object space to map the ray portal geometry's height axis to. Currently unused (as PCA is not implemented).");
    public: static void rayPortalSamplingWeightMinDistanceOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", float, rayPortalSamplingWeightMinDistance, 10.0f,
               "The minimum distance from a portal which the interpolation of the probability of light sampling through portals will begin (and is at its maximum value).\n"
               "Currently unimplemented, kept here for future use.",
               args.minValue = 0.0f, args.onChangeCallback = &rayPortalSamplingWeightMinDistanceOnChange);
    public: static void rayPortalSamplingWeightMaxDistanceOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", float, rayPortalSamplingWeightMaxDistance, 1000.0f,
               "The maximum distance from a portal which the interpolation of the probability of light sampling through portals will end (and is at its minimum value such that no portal light sampling will happen beyond this point).\n"
               "Currently unimplemented, kept here for future use.",
               args.minValue = 0.0f, args.onChangeCallback = &rayPortalSamplingWeightMaxDistanceOnChange);
    RTX_OPTION("rtx", bool, rayPortalCameraHistoryCorrection, false,
               "A flag to control if history correction on ray portal camera teleportation events is enabled or disabled.\n"
               "This allows for the previous camera matrix to be set to a virtual matrix to correct the large discontunity in position and view direction which happens when a camera teleports from moving through a ray portal (in games like Portal).\n"
               "As such this option should always be enabled in games utilizing ray portals the camera can pass through as it should fix artifacts from incorrectly calculated motion vectors or other deltas that rely on the current and previous camera matrix.");
    RTX_OPTION("rtx", bool, rayPortalCameraInBetweenPortalsCorrection, false,
               "A flag to contol correction when the camera is \"in-between\" a pair of ray portals.\n"
               "This is mostly relevant in applications which allow the camera to move through a ray portal (games like Portal) as often the ray portals are placed slightly off of a surface, allowing the camera to sometimes end up in this tiny gap for a frame.\n"
               "To correct this artifact (as it can mess up denoising and other temporal surface consistency checks due to the sudden frame of geometry in front of the camera) this option pushes the camera slightly backwards if this occurs when entering a ray portal.\n"
               "Similar to ray portal camera history correction this option should always be enabled in games utilizing ray portals the camera can pass through.");
    RTX_OPTION("rtx", float, rayPortalCameraInBetweenPortalsCorrectionThreshold, 0.1f,
               "The threshold to use for camera \"in-between\" ray portal detection in meters.\n"
               "When the camera is less than this distance behind the surface of a ray portal it will be pushed backwards to stay behind the ray portal.\n"
               "This value should stay small but be large enough to cover the gap between ray portals and the geometry behind them (if such a gap exists in the underlying application).\n"
               "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");

    RTX_OPTION_ENV("rtx", bool, useWhiteMaterialMode, false, "RTX_USE_WHITE_MATERIAL_MODE", "Override all objects' materials by white material");
    RTX_OPTION("rtx", bool, useHighlightLegacyMode, false, "");
    RTX_OPTION("rtx", bool, useHighlightUnsafeAnchorMode, false, "");
    RTX_OPTION("rtx", bool, useHighlightUnsafeReplacementMode, false, "");
    RTX_OPTION("rtx", float, nativeMipBias, 0.0f,
               "Specifies a mipmapping level bias to add to all material texture filtering. Stacks with the upscaling mip bias.\n"
               "Mipmaps are determined based on how far away a texture is, using this can bias the desired level in a lower quality direction (positive bias), or a higher quality direction with potentially more aliasing (negative bias).\n"
               "Note that mipmaps are also important for good spatial caching of textures, so too far negative of a mip bias may start to significantly affect performance, therefore changing this value is not recommended");
    RTX_OPTION("rtx", float, upscalingMipBias, 0.0f,
               "Specifies a mipmapping level bias to add to all material texture filtering when upscaling (such as DLSS) is used.\n"
               "Mipmaps are determined based on how far away a texture is, using this can bias the desired level in a lower quality direction (positive bias), or a higher quality direction with potentially more aliasing (negative bias).\n"
               "Note that mipmaps are also important for good spatial caching of textures, so too far negative of a mip bias may start to significantly affect performance, therefore changing this value is not recommended");
    RTX_OPTION("rtx", bool, useAnisotropicFiltering, true,
               "A flag to indicate if anisotropic filtering should be used on material textures, otherwise typical trilinear filtering will be used.\n"
               "This should generally be enabled as anisotropic filtering allows for less blurring on textures at grazing angles than typical trilinear filtering with only usually minor performance impact (depending on the max anisotropy samples).");
    // NV-DXVK: bumped from 8 → 16 to match the source D3D11 sampler's
    // aniMax. The original 8 default silently halved aniso vs native,
    // which was visible as severe directional smearing ("vertical
    // stretching") on TF2 BSP wall slivers — triangles whose authored
    // UV layout has aspect ratios up to 100:1 in screen space depend on
    // high aniso to resolve detail along the major axis. Native uses
    // the game-bound aniMax=16 ([D3D11Rtx.SampPick] log); Remix used to
    // override that down to 8 via min(limits, 8). 16 matches HW
    // maxSamplerAnisotropy on every consumer GPU since Maxwell, so the
    // min-clamp at the call site keeps it within HW limits regardless.
    RTX_OPTION("rtx", float, maxAnisotropySamples, 16.0f,
               "The maximum number of samples to use when anisotropic filtering is enabled.\n"
               "The actual max anisotropy used will be the minimum between this value and the hardware's maximum. Higher values increase quality but will likely reduce performance.");

    // Developer Options
    RTX_OPTION_FLAG_ENV("rtx", bool, enableBreakIntoDebuggerOnPressingB, false, RtxOptionFlags::NoSave, "RTX_BREAK_INTO_DEBUGGER_ON_PRESSING_B",
                    "Enables a break into a debugger at the start of InjectRTX() on a press of key \'B\'.\n"
                    "If debugger is not attached at the time, it will wait until a debugger is attached and break into it then.");
    
    // Crash Hotkey Feature (Development builds only)
    // When enabled via checkbox in the Development tab, pressing the configured hotkey will trigger a deliberate crash.
    // This is useful for testing crash handling, crash dumps, and crash reporting systems.
    // The feature is only available in REMIX_DEVELOPMENT builds and defaults to disabled.
    // The checkbox state is not saved to config files (NoSave), but can be pre-enabled by setting rtx.enableCrashHotkey = True in rtx.conf.
    RTX_OPTION_FLAG_ENV("rtx", bool, enableCrashHotkey, false, RtxOptionFlags::NoSave, "RTX_ENABLE_CRASH_HOTKEY",
                    "Arms the crash hotkey feature. When enabled, pressing the crash hotkey combination (Ctrl+Shift+Alt+K by default) will trigger a deliberate crash.\n"
                    "This option is only available in development builds and is intended for testing crash handling and crash dump generation.\n"
                    "The armed state is indicated by a red warning overlay on screen. This setting is not saved to config files but can be set manually in rtx.conf.");
    inline static const VirtualKeys kDefaultCrashHotkey{ VirtualKey{VK_CONTROL}, VirtualKey{VK_SHIFT}, VirtualKey{VK_MENU}, VirtualKey{'K'} };
    RTX_OPTION_FLAG("rtx", VirtualKeys, crashHotkey, kDefaultCrashHotkey, RtxOptionFlags::NoSave,
                    "The hotkey combination that triggers a deliberate crash when the crash hotkey feature is armed.\n"
                    "Default is Ctrl+Shift+Alt+K. Only takes effect when rtx.enableCrashHotkey is True.\n"
                    "This setting is not saved to config files but can be set manually in rtx.conf.");
    RTX_OPTION_FLAG("rtx", bool, enableInstanceDebuggingTools, false, RtxOptionFlags::NoSave, "NOTE: This will disable temporal correllation for instances, but allow the use of instance developer debug tools");
    RTX_OPTION("rtx", Vector2i, drawCallRange, Vector2i(0, INT32_MAX), "");
    RTX_OPTION("rtx", Vector3, instanceOverrideWorldOffset, Vector3(0.f, 0.f, 0.f), "");
    RTX_OPTION("rtx", uint, instanceOverrideInstanceIdx, UINT32_MAX, "");
    RTX_OPTION("rtx", uint, instanceOverrideInstanceIdxRange, 15, "");
    RTX_OPTION("rtx", bool, instanceOverrideSelectedInstancePrintMaterialHash, false, "");
    RTX_OPTION("rtx", bool, enablePresentThrottle, false,
               "A flag to enable or disable present throttling, when set to true a sleep for a time specified by the throttle delay will be inserted into the DXVK presentation thread.\n"
               "Useful to manually reduce the framerate if the application is running too fast or to reduce GPU power usage during development to keep temperatures down.\n"
               "Should not be enabled in anything other than development situations.");
    RTX_OPTION("rtx", std::uint32_t, presentThrottleDelay, 16U,
               "A time in milliseconds that the DXVK presentation thread should sleep for. Requires present throttling to be enabled to take effect.\n"
               "Note that the application may sleep for longer than the specified time as is expected with sleep functions in general.");
    RTX_OPTION_ENV("rtx", bool, validateCPUIndexData, false, "DXVK_VALIDATE_CPU_INDEX_DATA", "");
    RTX_OPTION("rtx", uint, dumpAllInstancesOnFrame, UINT32_MAX, "If set, and running in a REMIX_DEVELOPMENT build, this will dump all active instances to the log on the specified frame.");
    // Note: Use use areValidationLayersEnabled helper function rather than accessing this option directly as additional logic must be done to determine if validation layers should be used or not.
    RTX_OPTION_FLAG_ENV("rtx", bool, enableValidationLayers, false, RtxOptionFlags::NoSave, "DXVK_ENABLE_VALIDATION_LAYERS",
                        "A flag to enable validation layers in Vulkan. Note that in Debug builds validation layers will always be enabled and this flag will have no effect.\n"
                        "Enabling validation layers is useful for debugging and development to catch common issues in Vulkan, but will reduce overall performance.\n"
                        "Should only be enabled by developers during development and not put into production builds of any project.\n"
                        "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");
    RTX_OPTION_FLAG_ENV("rtx", bool, enableValidationLayerExtendedValidation, false, RtxOptionFlags::NoSave, "DXVK_ENABLE_VALIDATION_LAYER_EXTENDED_VALIDATION",
                        "A flag to enable extended validation to validation layers in Vulkan. Only takes effect if validation layers are enabled already.\n"
                        "This flag enables GPU assisted and synchronization validation along with best practices within the Vulkan validation layers which allow for greater error-checking capability at the cost of significant performance impact.\n"
                        "Much like the rtx.enableValidationLayers option, this option should only be enabled by developers during development and not be put into production builds of any project.\n"
                        "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");
    RTX_OPTION_FLAG_ENV("rtx", bool, logCallstacksOnValidationLayerErrors, true, RtxOptionFlags::NoSave, "DXVK_LOG_CALLSTACKS_ON_VALIDATION_LAYER_ERRORS",
                        "A flag to enable logging of callstacks when validation layer errors occur.\n"
                        "This is useful for debugging and development to help track down the source of validation layer errors more easily.\n"
                        "Requires pdb symbols to be present next to Remix's d3d11 dll and/or in the working directory to resolve symbols.");


    struct Aliasing {
      RTX_OPTION("rtx.aliasing", RtxFramePassStage, beginPass, RtxFramePassStage::FrameBegin, "The first render pass where the aliasing resource is bound in a frame.");
      RTX_OPTION("rtx.aliasing", RtxFramePassStage, endPass, RtxFramePassStage::FrameEnd, "The last render pass where the aliasing resource is bound in a frame.");
      RTX_OPTION("rtx.aliasing", RtxTextureFormatCompatibilityCategory, formatCategory, RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory, "Specifies the texture format compatibility category for the aliasing resource.");
      RTX_OPTION("rtx.aliasing", RtxTextureExtentType, extentType, RtxTextureExtentType::DownScaledExtent, "Specifies the resolution type for the aliasing resource. If a 3D texture is used, depth must also be set.");
      RTX_OPTION("rtx.aliasing", uint32_t, width, 1280, "The width of the aliasing resource in pixels.");
      RTX_OPTION("rtx.aliasing", uint32_t, height, 720, "The height of the aliasing resource in pixels.");
      RTX_OPTION("rtx.aliasing", uint32_t, depth, 1, "The depth of the aliasing resource. Required for 3D textures.");
      RTX_OPTION("rtx.aliasing", uint32_t, layer, 1, "The number of layers in the aliasing resource.");
      RTX_OPTION("rtx.aliasing", VkImageType, imageType, VkImageType::VK_IMAGE_TYPE_2D, "The image type of the aliasing resource (e.g., 1D, 2D, or 3D).");
      RTX_OPTION("rtx.aliasing", VkImageViewType, imageViewType, VkImageViewType::VK_IMAGE_VIEW_TYPE_2D, "The image view type of the aliasing resource (e.g., 1D, 2D, 3D, or cube).");
    } aliasing;

    struct OpacityMicromap {
      friend class RtxOptions;
      friend class ImGUI;
      bool isSupported = false;
      RTX_OPTION_ENV("rtx.opacityMicromap", bool, enable, true, "DXVK_ENABLE_OPACITY_MICROMAP", 
                     "Enables Opacity Micromaps for geometries with textures that have alpha cutouts.\n"
                     "This is generally the case for geometries such as fences, foliage, particles, etc. .\n"
                     "Opacity Micromaps greatly speed up raytracing of partially opaque triangles.\n"
                     "Examples of scenes that benefit a lot: multiple trees with a lot of foliage,\n"
                     "a ground densely covered with grass blades or steam consisting of many particles.");
    } opacityMicromap;

    RTX_OPTION_ARGS("rtx", ReflexMode, reflexMode, ReflexMode::LowLatency,
               "Reflex mode selection, enabling it helps minimize input latency, boost mode may further reduce latency by boosting GPU clocks in CPU-bound cases.\n"
               "Supported enum values are 0 = None (Disabled), 1 = LowLatency (Enabled), 2 = LowLatencyBoost (Enabled + Boost).\n"
               "Note that even when using the \"None\" Reflex mode Reflex will attempt to be initialized. Use rtx.isReflexEnabled to fully disable to skip this initialization if needed.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_FLAG("rtx", bool, isReflexEnabled, true, RtxOptionFlags::NoSave,
                    "Enables or disables Reflex globally.\n"
                    "Note that this option when set to false will prevent Reflex from even attempting to initialize, unlike setting the Reflex mode to \"None\" which simply tells an initialized Reflex not to take effect.\n"
                    "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");

    // Store the computed value separately from the user preference.  This enables changing it immediately when needed,
    // and lets us store the final value to be used by the game.
    public: inline static EnableVsync enableVsyncState = EnableVsync::WaitingForImplicitSwapchain;
    public: static void EnableVsyncOnChange(DxvkDevice* device) {
      if (enableVsync() != EnableVsync::WaitingForImplicitSwapchain) {
        enableVsyncState = enableVsync();
      }
      // If the option is changed to WaitingForImplicitSwapchain, just leave the computed state as it was.
    }
    RTX_OPTION_ARGS("rtx", EnableVsync, enableVsync, EnableVsync::WaitingForImplicitSwapchain, "Controls the game's V-Sync setting. Native game's V-Sync settings are ignored.", 
                    args.flags = RtxOptionFlags::NoSave | RtxOptionFlags::UserSetting,
                    args.onChangeCallback = &EnableVsyncOnChange);

    // Replacement options
    RTX_OPTION_ARGS("rtx", bool, enableReplacementAssets, true, "Globally enables or disables all enhanced asset replacement (materials, meshes, lights) functionality.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableReplacementLights, true,
               "Enables or disables enhanced light replacements.\n"
               "Requires replacement assets in general to be enabled to have any effect.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableReplacementMeshes, true,
               "Enables or disables enhanced mesh replacements.\n"
               "Requires replacement assets in general to be enabled to have any effect.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, enableReplacementMaterials, true,
               "Enables or disables enhanced material replacements.\n"
               "Requires replacement assets in general to be enabled to have any effect.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, enableReplacementInstancerMeshRendering, true,
               "Enables or disables rendering GeomPointInstancer meshes using an optimized path.\n"
               "Requires reloading replacement assets.");
    RTX_OPTION("rtx", uint, adaptiveResolutionReservedGPUMemoryGiB, 2,
               "The amount of GPU memory in gibibytes to reserve away from consideration for adaptive resolution replacement textures.\n"
               "This value should only be changed to reflect the estimated amount of memory Remix itself consumes on the GPU (aside from texture loading, mostly from rendering-related buffers) and should not be changed otherwise.\n"
               "Only relevant when force high resolution replacement textures is disabled and adaptive resolution replacement textures is enabled. See asset estimated size parameter for more information.\n");
    RTX_OPTION("rtx", uint, limitedBonesPerVertex, 4,
               "Limit the number of bone influences per vertex for replacement geometry.  Legacy games were limited to 4, which is the default.  In rare instances you may want to increase this based on your preference for replaced assets.  This config only takes affect when set on startup via the rtx.conf.");

    struct TextureManager {
      RTX_OPTION("rtx.texturemanager", int, budgetPercentageOfAvailableVram, 50,
                 "The percentage of available VRAM we should use for material textures.  If material textures are required beyond "
                 "this budget, then those textures will be loaded at lower quality.  Important note, it's impossible to perfectly "
                 "match the budget while maintaining reasonable quality levels, so use this as more of a guideline.  If the "
                 "replacements assets are simply too large for the target GPUs available vid mem, we may end up going overbudget "
                 "regularly.  Defaults to 50% of the available VRAM.");
      RTX_OPTION("rtx.texturemanager", bool, fixedBudgetEnable, false, "If true, rtx.texturemanager.fixedBudgetMiB is used instead of rtx.texturemanager.budgetPercentageOfAvailableVram.");
      RTX_OPTION_ARGS("rtx.texturemanager", int, fixedBudgetMiB, 2048, "Fixed-size VRAM budget for replacement textures. In mebibytes. To use, set rtx.texturemanager.fixedBudgetEnable to True.",
                      args.minValue = 256,
                      args.maxValue = 1024 * 32);
      RTX_OPTION_ENV("rtx.texturemanager", bool, samplerFeedbackEnable, true, "DXVK_TEXTURES_SAMPLER_FEEDBACK_ENABLE",
                 "Enable texture sampler feedback. If true, a texture prioritization logic considers the amount of mip-levels that was sampled by a GPU while rendering a scene."
                 "(For example, if a texture is in the distance, it will have a lower priority compared to a texture rendered just in front of the camera).");
      RTX_OPTION_FLAG_ENV("rtx.texturemanager", bool, skipSamplerFeedbackReadback, false, RtxOptionFlags::NoSave, "DXVK_TEXTURES_SKIP_SF_READBACK",
                 "Perf diagnostic. If true, RtxTextureManager::copySamplerFeedbackToHost skips the device->readback buffer copy. The "
                 "garbageCollection pass that consumes it still runs (it is CPU-only and issues nothing into the measured GPU span, so leaving "
                 "it on keeps [Perf.TexBudget] reporting during the A/B); it just reads stale feedback while this is set. Purpose is to A/B the "
                 "'postComposite' GPU stage: that stage's span contains only this "
                 "copy plus dispatchObjectPicking, yet reads ~92ms, while its CPU counter reads ~12us. If disabling this makes the ~92ms VANISH, "
                 "the copy really costs that much; if the ~92ms simply MOVES to the next stage (finalBlit) with frame time unchanged, the bucket "
                 "was a pipeline drain landing on the frame's first hard sync point and this copy is innocent (see rtx_context.h markGpuStage, "
                 "which documents the same BOTTOM_OF_PIPE attribution trap for an earlier 130ms bucket). Note this also freezes mip streaming "
                 "decisions, which is near-harmless while the forced-full-mip loop in garbageCollection is active, since that loop ignores "
                 "sampler-feedback priority anyway. Diagnostic only - do not ship enabled.");
      RTX_OPTION_FLAG_ENV("rtx.texturemanager", bool, neverDowngradeTextures, false, RtxOptionFlags::NoSave, "DXVK_TEXTURES_NEVER_DOWNGRADE",
                 "Debug option to forcibly prevent uploading lower resolution data, if the texture already has been promoted to a high resolution.");
      RTX_OPTION("rtx.texturemanager", int, stagingBufferSizeMiB, 96,
                 "Size of a pre-allocated staging (intermediate) buffer to use when sending a texture from a RAM to GPU VRAM. "
                 "If a texture size exceeds this limit, it will not be considered for the texture streaming. In mebibytes.");
      RTX_OPTION_FLAG_ENV("rtx.texturemanager", bool, hotReload, false, RtxOptionFlags::NoSave, "DXVK_TEXTURES_HOTRELOAD",
                 "While a game is running, if a texture file is modified on a disk, it will be automatically reuploaded to GPU.");
      RTX_OPTION_FLAG_ENV("rtx.texturemanager", uint, hotReloadRateMs, 100, RtxOptionFlags::NoSave, "DXVK_TEXTURES_HOTRELOAD_RATE_MS",
                 "Amount of time to wait between filesystem OS events, for texture hot-reloading. In milliseconds.");
    };
    RTX_OPTION_FLAG_ENV("rtx", bool, perfGpuStageSerialize, false, RtxOptionFlags::NoSave, "DXVK_PERF_GPU_STAGE_SERIALIZE",
               "Perf diagnostic. Emits a full ALL_COMMANDS execution+memory barrier before every markGpuStage timestamp, so each "
               "[Perf.GpuPass] stage measures work that has actually COMPLETED rather than work that has merely been submitted.\n"
               "Why this is needed: the timestamps are written at BOTTOM_OF_PIPE, which does not wait for compute/raytracing dispatches "
               "to drain in practice. The result is that a mark placed immediately after the gbuffer dispatch reads 0.05 ms while the "
               "EMPTY interval behind it reads 90.9 ms (recordVisibleSurfacesReadback compiles out entirely - kEnableRtxDebugProbes is "
               "false). The cost accumulates and is billed to whichever later mark the GPU actually reaches late, so the big bucket "
               "migrates as marks are added: postComposite -> rtxdi -> pt_visSurfReadback across three builds of the same scene. Null "
               "controls confirm the pattern - pc_null and gpuDrain are empty and read ~0, while other empty intervals carry the frame.\n"
               "This serialises the GPU, so the frame gets SLOWER and stage times stop overlapping. That is the point: totals are not "
               "comparable to a normal run, but the per-stage split is trustworthy. Use it to find the owner, then turn it off to measure. "
               "Diagnostic only - never leave enabled.");
    RTX_OPTION_FLAG_ENV("rtx", bool, skyVertsReadbackEnable, false, RtxOptionFlags::NoSave, "DXVK_SKY_VERTS_READBACK",
               "Sky-geometry diagnostic. When true, RtxContext::rasterizeToSkyMatte copies up to 1024 vertices of each of the first "
               "8 sky draws per frame to a host-visible buffer and decodes them asynchronously ([SkyTrace.matteRaster] / "
               "[SkyTrace.skyVerts]), to determine which world directions TF2's sky quads span.\n"
               "Default is false because this is NOT free. Per gameplay frame it creates up to 8 host-visible buffers, emits 16 "
               "pipeline barriers (VERTEX_INPUT->TRANSFER and TRANSFER->HOST), issues 8 buffer copies, raises 8 timeline signals and "
               "spawns 8 std::async tasks - all inside the rasterizeToSkyMatte GPU profile zone. Nsight Systems (661-frame capture, "
               "2026-07-27) measured rasterizeToSkyMatte at p50 0.011 ms but p95 91 ms / p99 132 ms, with 655 spikes over 50 ms in 661 "
               "frames - about one ~100 ms stall per frame, 65.5 s of a 203 s capture, and NONE of it inside InjectRTX. That is the "
               "long-unexplained gap between the ~55 ms InjectRTX span and the ~170 ms frame.\n"
               "It stayed invisible because log.cpp filters the \"[SkyTrace.\" prefix, which suppresses the output but not the "
               "readback itself. Enable only while actively investigating sky geometry, and read medians with care: 9 of the 10 sky "
               "draws per frame are microseconds, so a median hides the one that is not.");
    RTX_OPTION("rtx", bool, reloadTextureWhenResolutionChanged, false, "Reload texture when resolution changed.");
    RTX_OPTION_FLAG_ENV("rtx", bool, alwaysWaitForAsyncTextures, false, RtxOptionFlags::NoSave, "DXVK_WAIT_ASYNC_TEXTURES", 
               "Force CPU to wait for the texture upload. Do not use an asynchronous thread for textures. If true, a frame stutter should be expected.");
    RTX_OPTION_FLAG_ENV("rtx.initializer", bool, asyncAssetLoading, true, RtxOptionFlags::NoSave, "DXVK_ASYNC_ASSET_LOADING", "If true, a separate thread is created to load USD assets asynchronously.");
    RTX_OPTION("rtx", bool, usePartialDdsLoader, true,
               "A flag controlling if the partial DDS loader should be used, true to enable, false to disable and use GLI instead.\n"
               "Generally this should be always enabled as it allows for simple parsing of DDS header information without loading the entire texture into memory like GLI does to retrieve similar information.\n"
               "Should only be set to false for debugging purposes if the partial DDS loader's logic is suspected to be incorrect to compare against GLI's implementation.");

    RTX_OPTION("rtx", TonemappingMode, tonemappingMode, TonemappingMode::Local,
               "The tonemapping type to use, 0 for Global, 1 for Local (Default).\n"
               "Global tonemapping tonemaps the image with respect to global parameters, usually based on statistics about the rendered image as a whole.\n"
               "Local tonemapping on the other hand uses more spatially-local parameters determined by regions of the rendered image rather than the whole image.\n"
               "Local tonemapping can result in better preservation of highlights and shadows in scenes with high amounts of dynamic range whereas global tonemapping may have to comprimise between over or underexposure.\n"
               "Overridden to Global while rtx.autoExposure.mode is Plus, because Plus is itself a local dynamic range compressor and running the local tonemapper after it compresses the image twice. The value stored here is left alone and takes effect again as soon as Plus is turned off. See rtx.autoExposurePlus.forceGlobalTonemapper.");
    RTX_OPTION("rtx", bool, useLegacyACES, true,
               "Use a luminance-only approximation of ACES that over-saturates the highlights. If false, use a refined ACES transform that converts between color spaces with more precision.");
    RTX_OPTION("rtx", bool, showLegacyACESOption, false,
               "Show \'rtx.useLegacyACES\' in the developer menu. Default is OFF, as the non-legacy ACES is currently experimental and the implementation is a subject to change.");

    // Capture Options
    //   General
    RTX_OPTION("rtx", bool, captureShowMenuOnHotkey, true,
               "If true, then the capture menu will appear whenever one of the capture hotkeys are pressed. A capture MUST be started by using a button in the menu, in that case.\n"
               "If false, the hotkeys behave as expected. The user must manually open the menu in order to change any values.");
    inline static const VirtualKeys kDefaultCaptureMenuKeyBinds{VirtualKey{VK_CONTROL},VirtualKey{VK_SHIFT},VirtualKey{'Q'}};
    RTX_OPTION("rtx", VirtualKeys, captureHotKey, kDefaultCaptureMenuKeyBinds,
               "Hotkey to trigger a capture without bringing up the menu.\n"
               "example override: 'rtx.captureHotKey = CTRL, SHIFT, P'.\n"
               "Full list of key names available in `src/util/util_keybind.h`.");
    RTX_OPTION("rtx", bool, captureInstances, true,
               "If true, an instanced snapshot of the game scene will be captured and exported to a USD stage, in addition to all meshes, textures, materials, etc.\n"
               "If false, only meshes, etc will be captured.");
    RTX_OPTION("rtx", bool, captureNoInstance, false, "Same as \'rtx.captureInstances\' except inverse. This is the original/old variant, and will be deprecated, however is still functional.");
    RTX_OPTION("rtx", std::string, captureTimestampReplacement, "{timestamp}",
               "String that can be used for auto-replacing current time stamp in instance stage name.\n"
               "Note: Changing this value does not change the default value for rtx.captureInstanceStageName.");
    // Note: default values are used before configs are loaded.  Cannot use the value of `captureTimestampReplacement` to set the default value of `captureInstanceStageName`.
    RTX_OPTION("rtx", std::string, captureInstanceStageName, "capture_{timestamp}.usd",
               "Name of the \'instance\' stage (see: \'rtx.captureInstances\').");
    RTX_OPTION("rtx", bool, captureOverwriteExistingCapture, false,
               "If true, a capture with the same filename will overwrite any existing capture file instead of appending a numeric suffix to avoid collisions.");
    RTX_OPTION("rtx", bool, captureEnableMultiframe, false, "Enables multi-frame capturing. THIS HAS NOT BEEN MAINTAINED AND SHOULD BE USED WITH EXTREME CAUTION.");
    RTX_OPTION("rtx", uint32_t, captureMaxFrames, 1, "Max frames capturable when running a multi-frame capture. The capture can be toggled to completion manually.");
    RTX_OPTION("rtx", uint32_t, captureFramesPerSecond, 24,
               "Playback rate marked in the USD stage.\n"
               "Will eventually determine frequency with which game state is captured and written. Currently every frame -- even those at higher frame rates -- are recorded.");
    //   Mesh
    RTX_OPTION("rtx", float, captureMeshPositionDelta, 0.3f, "Inter-frame position min delta warrants new time sample.");
    RTX_OPTION("rtx", float, captureMeshNormalDelta, 0.3f, "Inter-frame normal min delta warrants new time sample.");
    RTX_OPTION("rtx", float, captureMeshTexcoordDelta, 0.3f, "Inter-frame texcoord min delta warrants new time sample.");
    RTX_OPTION("rtx", float, captureMeshColorDelta, 0.3f, "Inter-frame color min delta warrants new time sample.");
    RTX_OPTION("rtx", float, captureMeshBlendWeightDelta, 0.01f, "Inter-frame blend weight min delta warrants new time sample.");

    RTX_OPTION("rtx", bool, useVirtualShadingNormalsForDenoising, true,
               "A flag to enable or disable the usage of virtual shading normals for denoising passes.\n"
               "This is primairly important for anything that modifies the direction of a primary ray, so mainly PSR and ray portals as both of these will view a surface from an angle different from the \"virtual\" viewing direction perceived by the camera.\n"
               "This can cause some issues with denoising due to the normals not matching the expected perception of what the normals should be, for example normals facing away from the camera direction due to being viewed from a different angle via refraction or portal teleportation.\n"
               "To correct this, virtual normals are calculcated such that they always are oriented relative to the primary camera ray as if its direction was never altered, matching the virtual perception of the surface from the camera's point of view.\n"
               "As an aside, virtual normals themselves can cause issues with denoising due to the normals suddenly changing from virtual to \"real\" normals upon traveling through a portal, causing surface consistency failures in the denoiser, but this is accounted for via a special transform given to the denoiser on camera ray portal teleportation events.\n"
               "As such, this option should generally always be enabled when rendering with ray portals in the scene to have good denoising quality.");
    RTX_OPTION("rtx", bool, resetDenoiserHistoryOnSettingsChange, false, "");

    RTX_OPTION("rtx", bool, volumetricFogSkipSky, false,
               "When enabled, sky-tagged draw calls are excluded from volumetric fog parameter extraction.\n"
               "Useful when sky geometry introduces incorrect fog values that bleed into the scene.");

    RTX_OPTION("rtx", float, skyBrightness, 1.f, "");
    RTX_OPTION("rtx", bool, skyForceHDR, false, "By default sky will be rasterized in the color format used by the game. Set the checkbox to force sky to be rasterized in HDR intermediate format. This may be important when sky textures replaced with HDR textures.");
    RTX_OPTION("rtx", uint32_t, skyProbeSide, 1024, "Resolution of the skybox for indirect illumination (rough reflections, global illumination etc).");
    RTX_OPTION_FLAG("rtx", uint32_t, skyUiDrawcallCount, 0, RtxOptionFlags::NoSave, "");
    RTX_OPTION("rtx", uint32_t, skyDrawcallIdThreshold, 0, "It's common in games to render the skybox first, and so, this value provides a simple mechanism to identify those early draw calls that are untextured (textured draw calls can still use the Sky Textures functionality.");
    RTX_OPTION("rtx", float, skyMinZThreshold, 1.f, "If a draw call's viewport has min depth greater than or equal to this threshold, then assume that it's a sky.");
    RTX_OPTION("rtx", SkyAutoDetectMode, skyAutoDetect, SkyAutoDetectMode::None, 
               "Automatically tag sky draw calls using various heuristics.\n"
               "0 = None\n"
               "1 = CameraPosition - assume the first seen camera position is a sky camera.\n"
               "2 = CameraPositionAndDepthFlags - assume the first seen camera position is a sky camera, if its draw call's depth test is disabled. If it's enabled, assume no sky camera.\n"
               "Note: if all draw calls are marked as sky, then assume that there's no sky camera at all.");
    RTX_OPTION("rtx", float, skyAutoDetectUniqueCameraDistance, 1.0f,
               "If multiple cameras are found, this threshold distance (in game units) is used to distinguish a sky camera from a main camera. "
               "Active if sky auto-detect is set to CameraPosition / CameraPositionAndDepthFlags.")
    RTX_OPTION("rtx", bool, skyReprojectToMainCameraSpace, false,
               "Move sky geometry to the main camera space.\n"
               "Useful, if a game has a skybox that contains geometry that can be a part of the main scene (e.g. buildings, mountains). "
               "So with this option enabled, that geometry would be promoted from sky rasterization to ray tracing.");
    RTX_OPTION("rtx", float, skyReprojectScale, 16.0f, "Scaling of the sky geometry on reprojection to main camera space.");
    // NV-DXVK [TF2 reproject bisect]: when false, the engine-hook sub-view
    // reproject transform (T_reproject) is NOT applied to sky draws, while the
    // engine-hook main camera feed stays active. Diagnostic to separate "the
    // reproject geometry scaling is the artifact" from "the hook's
    // infinite-far-plane main projection is the artifact" — both otherwise
    // only toggle together via useEngineHookMainCamera. Default true = ship
    // behavior unchanged.
    RTX_OPTION("rtx", bool, tf2ApplySubViewReproject, true,
               "Titanfall 2 diagnostic. When false, skips applying the engine-hook sub-view reproject transform to sky geometry (camera hook stays on).");
    RTX_OPTION("rtx", bool, tf2SetupBonesFullVertexMask, true,
               "Titanfall 2 only. Razor-spike fix: in the client.dll SetupBones entry hook, "
               "widen any bone mask carrying vertex-LOD bits (BONE_USED_BY_VERTEX_LOD0..7 = "
               "0x3FC00) to ALL eight vertex-LOD bits. The game poses only bones the current "
               "LOD's vertices use; the rest keep bind-local cache entries, and that "
               "part-posed palette is uploaded as-is ([BoneUpLocal]). RT skins captured "
               "meshes with every blend index/weight, so verts weighted 6-11% to un-posed "
               "bones stretch ~12k units ([SkinFanW]). Widening only ADDS freshly-computed "
               "bones: SetupBones already copies the FULL bone cache to callers regardless "
               "of mask, so no buffer grows and no valid data is overwritten. False = stock.");
    // NV-DXVK [MatBatch]: see matBatchInstallHook in d3d11_rtx.cpp. Gates BOTH
    // the vtable swap and the per-batch dump, so at the default false no detour
    // is placed at all. Kept OUT of the RTX_DISABLE_ENGINE_HOOKS master switch on
    // purpose: that switch also disables the engine-hook camera, so it stays set
    // during normal play and cannot be used to arm a diagnostic. Live-toggleable
    // — flipping it true at runtime installs the hook on the next frame; flipping
    // it back false silences the dump but leaves the (harmless) wrapper in place.
    RTX_OPTION("rtx", bool, tf2MatsysBatchDump, false,
               "Titanfall 2 diagnostic. Hooks the materialsystem_dx11 studio-model batch renderer "
               "(vtable slots 0x550/0x558 -> sub_18001D780) and dumps each batch's 96-byte element "
               "array as [MatBatch]. This is the last point ABOVE that function's input-layout gate, "
               "where a null ID3D11InputLayout makes the engine skip the draw entirely and issue no "
               "D3D11 call at all. Diff the elements against the DrawIndexed calls reaching DXVK: an "
               "element present here with no matching draw was swallowed by the layout gate; an "
               "element missing entirely was culled upstream in studiorender. Per-batch logging on "
               "the render thread — leave false unless capturing.");
    // NV-DXVK [TF2 Basic/unlit family]: Source/Titanfall splits material pixel
    // shaders into two cbuffer families — "Uber" (CBufUberStatic/Dynamic) for
    // the full lit world+model path, and "Basic" (CBufBasicStatic/Dynamic) for
    // the simple UNLIT family. Basic shaders output texture*const+fog with no
    // normal/lighting math, so a normal-less Basic quad path-traced as lit
    // renders pitch black (e.g. the Ark numeric panel). When enabled, an
    // OPAQUE, DEPTH-WRITING Basic-family draw (and not Uber) with a bound color
    // texture is routed to the same unlit/matte+emissive path as VGUI
    // (sourceIsUnlitUI), forwarding its texture unlit; the glyph interleaver
    // self-skips (no TEXCOORD3).
    //
    // DEFAULT OFF. The opaque+depthWrite gate is REQUIRED: the Basic family
    // also covers high-volume alpha-blended sprites/HUD/overlays, and routing
    // those to emissive turns thousands of surfaces into emissive meshes and
    // FREEZES scene build (observed). Opaque+depth-writing restricts it to the
    // small set of solid world panels. Opt in via conf once validated.
    RTX_OPTION("rtx", bool, tf2RouteBasicShadersUnlit, false,
               "Titanfall 2 only. Route OPAQUE depth-writing draws whose pixel shader uses the unlit 'Basic' cbuffer family (not CBufUber*) to unlit/matte emissive output. Fixes normal-less solid UI panels (e.g. the Ark numeric display) rendering black. Default off; the opaque+depthWrite gate excludes high-volume blended sprites/HUD that would otherwise freeze scene build.");
    // NV-DXVK [engine-post forward]: harvest the host game's final post-process
    // composite (Source/Titanfall2 CBufEnginePost: tonemap+bloom+DoF+color-correct)
    // and forward its parameters into Remix's own post pipeline instead of letting
    // that fullscreen quad inject as grey scene geometry. Master gate; per-effect
    // sub-gates below let each effect be toggled independently while tuning.
    RTX_OPTION("rtx", bool, enginePostForward, true,
               "Titanfall 2 / Source. Detect the engine's final post-process composite draw (binds CBufEnginePost: tonemap, bloom, depth-of-field, color-correction), drop it so it stops rendering as a flat grey fullscreen surface, and forward its parameters into Remix's post pipeline. Master gate for enginePostForwardBloom/Exposure/Tonemap/ColorCorrect/Dof.");
    RTX_OPTION("rtx", bool, enginePostForwardBloom, true,
               "When enginePostForward is on, drive Remix's bloom (rtx.bloom.*) from the game's c_bloomAmount/c_wideBloomAmount/c_streakBloomAmount. Native Remix bloom is used.");
    RTX_OPTION("rtx", bool, enginePostForwardExposure, true,
               "When enginePostForward is on, drive Remix's exposure (rtx.tonemap.exposureBias / auto-exposure) from the game's exposureTexture / c_forceExposure.");
    RTX_OPTION("rtx", bool, enginePostForwardTonemap, true,
               "When enginePostForward is on, reproduce the game's filmic toe/mid/shoulder tone curve (c_debugTonemap*) inside Remix's apply-tonemapping shader. Disables Remix's own tone curve for those frames to avoid double tonemapping.");
    RTX_OPTION("rtx", bool, enginePostForwardColorCorrect, true,
               "When enginePostForward is on, sample the game's 3D color-correction LUT volume (ColorCorrectionVolumeTexture0, weighted by c_colorCorrectionVolumeWeights) on Remix's tonemapped output.");
    RTX_OPTION("rtx", bool, enginePostForwardDof, true,
               "When enginePostForward is on, apply depth-of-field on Remix's composite output, recomputing circle-of-confusion from the path-traced depth using the game's c_dof params (Route B). No-op on frames where the game's DoF is inactive.");
    // NV-DXVK [tf2StableBackfaceCull]: the RT front/back (FLIP_FACING) decision
    // in determineInstanceFlags uses the objectToWorld mirror parity ALONE.
    // For billboard/skinned draws rendered under the sub-view camera (e.g. the
    // Ark worldspace marker card VS 0x2939c0d0), objectToWorld's determinant
    // sign flips frame-to-frame as the card reorients, toggling FLIP_FACING and
    // thus which face is culled — so the dark back face (which the raster game
    // always culls) leaks in from some camera angles. The raster game culls on
    // the NET object->projection screen-space winding, which is stable
    // (o2wMirror and w2pMirror flip oppositely; their parity holds). When true,
    // use that net parity for the FLIP decision. It is mathematically identical
    // to the current rule whenever the projection is non-mirrored (w2pMirror=0)
    // — i.e. ALL normal main-camera geometry (BSP floor/walls/props) is
    // unaffected; only mirrored/sub-view projections, where the o2w-only rule is
    // wrong, change. Default off — opt in to A/B against the tuned BSP rule.
    RTX_OPTION("rtx", bool, tf2StableBackfaceCull, false,
               "Titanfall 2 only. Base the ray-traced front/back-face (cull) decision on the net object->projection winding parity instead of objectToWorld alone. Fixes single-sided sub-view billboard cards (e.g. the Ark marker) whose culled face flips with camera motion and leaks a dark back face. No effect on non-mirrored main-camera geometry.");
    // NV-DXVK [StudioModelHook]: a true BY-MODEL gate for Titanfall 2. The
    // studiorender.dll draw-site trampolines (see D3D11Rtx::EndFrame install)
    // capture the engine model/material name path for each studiorender draw;
    // SubmitDraw matches "widow" in that path. Unlike texture/VS-hash hiding,
    // this is 1:1 with the engine model, immune to shared shaders/textures.
    RTX_OPTION("rtx", bool, tf2HideWidow, false,
               "Titanfall 2 only. Hide the Widow dropship by its engine model name (studiorender hook). Drops every draw whose studiorender material/model path contains \"widow\". True by-model gate (not texture/VS-hash based).");
    RTX_OPTION("rtx", bool, tf2IsolateWidow, false,
               "Titanfall 2 only. Inverse of tf2HideWidow: hide every OTHER studiorender model draw and keep only the Widow, to isolate the ship visually (e.g. in the Diffuse Albedo debug view) and decide whether a real render bug exists. Non-studiorender draws (world BSP, UI) are unaffected.");
    RTX_OPTION("rtx", bool, tf2DetectWidow, false,
               "Titanfall 2 only. Compute the by-model Widow tag (DrawCallState::isWidowModel) on every studiorender draw WITHOUT hiding anything. Lets the deep diagnostic probes ([ShipBake], [HullWorldRB], [interleaver.skin], [ShipBox]) re-gate on the actual engine model instead of the shared VS hash. Implied by tf2HideWidow / tf2IsolateWidow.");
    RTX_OPTION("rtx", bool, tf2DumpStudioNames, false,
               "Titanfall 2 only. Log every DISTINCT studiorender model/material name path that passes through the draw-site hook, once per name ([StudioName]). Not frame-throttled, deduped by name. Reveals the full set of model paths reaching the hook (e.g. to find which material the visible ship/floor surface uses and whether it goes through the hooked draw path at all).");
    RTX_OPTION("rtx", bool, tf2DisableAlphaSurfaces, false,
               "Titanfall 2: drop EVERY alpha-blended (translucent) draw at submission — any draw whose RT0 blend is enabled is filtered out (FilterReason::AlphaSurface) and never reaches the RT scene. Removes all translucent surfaces at once (glass, fog cards, premult billboards, decals). Off by default.");
    RTX_OPTION("rtx", bool, tf2LogTranslucentDraws, false,
               "Titanfall 2 diagnostic: log once per vertex-shader every ALPHA-BLENDED (translucent) draw reaching FillMaterialData ([TransCensus]), with VS/PS hash, blend factors and depthWrite. The surface-coverage / PickRegion probes attribute only OPAQUE primary hits, so they are blind to translucent surfaces (which is why such a surface is invisible in the Diffuse Albedo view and why PickRegion reports the opaque world behind it). Use this to name a translucent artifact (e.g. a foreground fog wedge) by VS hash so it can be targeted with rtx.debug.hideVertexShaders. Off by default; deduped per VS so it can't flood.");
    RTX_OPTION("rtx", bool, tf2HeavyProbes, false,
               "Titanfall 2 diagnostics master switch for the EXPENSIVE per-frame GPU-readback probes ([MtnComposite]/[SurfTrack], [HullWorldRB], [interleaver.skin]). Default OFF: these flush+waitForIdle and dump per-pixel data every frame, which stalls the GPU and floods the log, causing the recurring Aftermath device-loss (freeze→crash). Leave OFF for normal play / the cull investigation (the cheap [ShipBox] + [HullCensus] probes stay on under logSurfaceCoverage/tf2DetectWidow). Set True only for a short targeted capture.");
    RTX_OPTION("rtx", bool, tf2SkinnedUseDrawCamera, true,
               "Titanfall 2 fix (default ON). In DrawCallState::finalizeSkinningData, un-fuse a skinned mesh's objectToWorld against the DRAW'S OWN worldToView (the camera it was actually rendered with) instead of the global last-set/engine-hook Main camera (pLastCamera). For normal titles the two cameras are identical so this is a no-op; under TF2's single-global engine-hook camera they diverge and the Main-based decompose teleports the skinned geometry's world-space BLAS (the Widow 'vanish'). Set False to restore the original pLastCamera behavior.");
    // NV-DXVK [TF2 inf-far clamp]: the engine hook feeds a Source/Titanfall
    // infinite-far reverse projection (zFar=inf). Remix's RT passes assume a
    // finite far — the inf-far matrix trips the degeneracy guards in
    // overrideNearPlane / getVolumeDefinitionCamera and yields a degenerate
    // ProjectionToView inverse (screen-space world-pos reconstruction → NaN).
    // When true, processExternalCamera rebuilds the Main projection with a
    // large finite far (still well past the reprojected skybox at ~1.5e7).
    RTX_OPTION("rtx", bool, tf2ClampEngineFarPlane, true,
               "Titanfall 2. Rebuild the engine-hook Main projection with a finite far plane (the engine supplies an infinite far plane that breaks Remix's RT passes).");
    RTX_OPTION("rtx", bool, skyForceAutoDetectedToReproject, false,
               "When enabled, draw calls classified as sky by auto-detect are always reprojected to main camera space "
               "instead of being rasterized to the sky cubemap. This fixes a class of bugs where auto-detect misclassifies "
               "world geometry as sky (due to shared camera positions), causing that geometry to become invisible. "
               "Only effective when Sky Auto-Detect and Reproject Sky to Main Camera are both enabled.");

    // NV-DXVK [SkyAutoCb2]: defaulted to PhysicalAtmosphere so the Hillaire
    // atmospheric LUT pipeline (rtx_atmosphere.cpp) runs out-of-the-box for
    // TF2 — paired with the cb2.c_cameraOrigin sky detector in
    // D3D11Rtx::SetSkyCategoryFromCb2, sky-camera draws are dropped from
    // the BLAS and the path tracer samples the LUTs for sky rays instead.
    RTX_OPTION("rtx", SkyMode, skyMode, SkyMode::Hybrid,
               "Sky rendering mode. SkyboxRasterization uses traditional skybox rasterization, "
               "PhysicalAtmosphere uses Hillaire atmospheric scattering, "
               "Hybrid uses TF2's rasterized skybox for visible sky + bounce while keeping "
               "Hillaire's sun NEE so direct sunlight gets atmospheric transmittance.");

    // Atmosphere parameters
    RTX_OPTION("rtx.atmosphere", bool, sunDisc, true, "Include the sun itself in the output.");
    RTX_OPTION("rtx.atmosphere", float, sunSize, 0.545f, "Size of sun disc in degrees.");
    RTX_OPTION("rtx.atmosphere", float, sunIntensity, 1.0f, "Strength of Sun.");
    // NV-DXVK [EngineSun]: register the atmosphere sun as a real RTXDI Distant light. The
    // bespoke NEE sun (sampleAtmosphereSunLight) is invisible to RTXDI, so sun-lit skybox
    // geometry gets rtxdiIllum=0 -> denoiser confidence floors -> NRD blacks it out (and
    // leaves streaks). Routing the sun through RTXDI populates the reservoir so confidence
    // is valid. When on, the NEE sun is disabled (gated in the integrator) to avoid double
    // lighting; tune sunRtxdiRadianceScale to match the previous brightness.
    RTX_OPTION("rtx.atmosphere", bool, sunAsRtxdiLight, false, "Register the physical-atmosphere sun as an RTXDI Distant light (fixes denoiser-blacked / streaking sun-lit skybox geometry). Disables the NEE sun to avoid double lighting.");
    RTX_OPTION("rtx.atmosphere", float, sunRtxdiRadianceScale, 0.3f, "Radiance scale applied to sunIlluminance when sunAsRtxdiLight is on. ~0.3 approximates the previous NEE-sun brightness (atmosphere applies ~0.5*mie*vis/pi); tune to taste.");
    RTX_OPTION("rtx.atmosphere", float, sunElevation, 15.0f, "Sun angle from horizon in degrees.");
    RTX_OPTION("rtx.atmosphere", float, sunRotation, 0.0f, "Rotation of sun around zenith in degrees.");
    RTX_OPTION("rtx.atmosphere", float, altitude, 100.0f, "Height from sea level in meters.");
    RTX_OPTION("rtx.atmosphere", float, airDensity, 1.0f, "Density of air molecules multiplier (1.0 = clear sky).");
    RTX_OPTION("rtx.atmosphere", float, aerosolDensity, 1.0f, "Density of aerosols/dust multiplier (1.0 = typical).");
    RTX_OPTION("rtx.atmosphere", float, ozoneDensity, 1.0f, "Density of ozone layer multiplier (1.0 = typical).");
    
    // Advanced/Internal Atmosphere Parameters
    RTX_OPTION("rtx.atmosphere", float, planetRadius, 6371.0f, "Planet radius in kilometers.");
    RTX_OPTION("rtx.atmosphere", float, atmosphereThickness, 100.0f, "Atmosphere thickness in kilometers.");
    RTX_OPTION("rtx.atmosphere", float, mieAnisotropy, 0.97f, "Mie phase function anisotropy (g parameter, -1 to 1).");
    
    // Base coefficients (can be used for non-Earth atmospheres, scaled by density sliders)
    RTX_OPTION("rtx.atmosphere", Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f), "Base Rayleigh scattering coefficients (km^-1).");
    RTX_OPTION("rtx.atmosphere", Vector3, mieScattering, Vector3(3.996e-3f, 3.996e-3f, 3.996e-3f), "Base Mie scattering coefficients (km^-1).");
    RTX_OPTION("rtx.atmosphere", Vector3, ozoneAbsorption, Vector3(2.04e-3f, 4.97e-3f, 2.14e-4f), "Base Ozone absorption coefficients (km^-1).");
    RTX_OPTION("rtx.atmosphere", float, ozoneLayerAltitude, 25.0f, "Altitude of ozone layer peak in kilometers.");
    RTX_OPTION("rtx.atmosphere", float, ozoneLayerWidth, 15.0f, "Width of the ozone layer in kilometers.");
    RTX_OPTION("rtx.atmosphere", Vector3, sunIlluminance, Vector3(20.0f, 20.0f, 20.0f), "Base Sun illuminance color/intensity.");

    // NV-DXVK [AerialPerspective]: 3D LUT-based atmospheric haze applied
    // to all geometry by the path tracer, regardless of sky source.
    // Strength is a user multiplier on the effect; worldToKm converts
    // the game's world units to km for the LUT distance axis. TF2
    // hammer = ~52,500 units/km so the default of 1.9e-5 is the natural
    // value for that engine. Bump strength toward 2-3 for stylised
    // distance haze, drop to 0 to disable entirely.
    // NV-DXVK [AerialPerspective.dedupe]: previously hard-defaulted to
    // 0.0 to mask a double-apply bug — geometry_resolver.slangh and
    // composite.comp.slang both applied AP to the same radiance,
    // producing a cyan/blue ghost. The geometry_resolver application
    // has been removed; AP now applies once in composite which is the
    // intended single-application path. Restored physics-correct 1.0
    // default. The [Atmosphere.lut] readback in rtx_context.cpp
    // confirmed the LUT values themselves are sane.
    RTX_OPTION("rtx.atmosphere", float, aerialPerspectiveStrength, 1.0f,
               "Multiplier on the aerial perspective LUT contribution. "
               "0 = effect disabled; 1 = physics-correct; 2-3 = exaggerated "
               "haze for stylised look.");
    RTX_OPTION("rtx.atmosphere", float, aerialPerspectiveWorldToKm, 1.9e-5f,
               "Multiplier from game world units to km for the aerial "
               "perspective distance axis. TF2 hammer is ~52500 units/km "
               "so the default 1.9e-5 is correct. Other games may need "
               "tuning - lower = haze pushes to longer distances.");

    // NV-DXVK [EngineSunCapture]: drive the Hillaire atmosphere from
    // TF2's per-frame sun. Field names confirmed from the dump:
    //   CBufCommonPerCamera.c_sunDir   (off=288, vec3, used by VS+PS)
    //   CBufCommonPerCamera.c_sunColor (off=276, vec3, used by VS+PS)
    // Producer is D3D11Rtx::CaptureEngineSunFromCb; consumer is
    // RtxAtmosphere::getAtmosphereArgs. Snapshot transit lives in
    // rtx_engine_sun.h (intentionally lightweight - the prior attempt
    // routed through rtx_atmosphere.h and broke engine init).
    RTX_OPTION("rtx.atmosphere", bool, useEngineSun, true,
               "Override the static slider sun with TF2's per-frame "
               "c_sunDir / c_sunColor. Default ON so every map's "
               "light_environment colour and CSM direction drives the "
               "Hillaire sky without per-map config.");
    RTX_OPTION("rtx.atmosphere", float, engineSunIntensityScale, 3.0f,
               "Multiplier on the captured c_sunColor. TF2 stores it "
               "pre-scaled (~6 magnitude on a typical map) - 3x brings it "
               "into the slider default illuminance range (~20).");
    RTX_OPTION("rtx.atmosphere", bool, engineSunIsZUp, true,
               "True if the captured world-space sun direction is in "
               "Z-up space (TF2/Source). The atmosphere LUTs are Y-up.");
    RTX_OPTION("rtx.atmosphere", bool, engineSunDirIsTowardsLight, true,
               "True if c_sunDir points FROM the surface TOWARDS the sun. "
               "TF2's c_sunDir is towards-light (z=+0.69 with sun above "
               "horizon), so leave true unless captures look mirrored.");
    RTX_OPTION("rtx.atmosphere", float, engineFogStrengthScale, 1.0f,
               "Multiplier on captured TF2 fog max density when blending "
               "Hillaire AP inscatter toward TF2's authored fog colour. "
               "0 = use pure Hillaire physics; 1 = honour TF2 fog strength; "
               ">1 = exaggerate the artist's fog. Affects only the AP "
               "consumer in composite.comp.slang.");
    // Diagnostics off by default now that field names are locked in.
    // Flip to true if a future TF2 build renames the cbuffer fields.
    RTX_OPTION("rtx.atmosphere", bool, dumpEngineSunCBFields, false,
               "Diagnostic - log every (cbufferName, fieldName, offset, size, "
               "used) tuple the d3d11 producer sees, throttled to 1/64 draws "
               "and dedup'd per-tuple.");
    RTX_OPTION("rtx.atmosphere", bool, dumpEngineSunCBValues, false,
               "Diagnostic - read every vec3/vec4-sized field, classify as "
               "DIRECTION_LIKELY / COLOR_LIKELY / OTHER, log periodically. "
               "Throttled to 1/64 draws.");
    RTX_OPTION("rtx.atmosphere", uint32_t, dumpEngineSunValuesEveryNFrames, 1,
               "[EngineSun.values] log cadence. Per-(stage,cb,field) tuple is "
               "logged at most once every N frames. 1 = every frame.");

    // NV-DXVK [EngineCam]: read TF2's authoritative main-camera matrices
    // (worldToView + viewToProjection) directly from the engine via a
    // trampoline at engine.dll!R_DrawWorldMeshes. The trampoline snapshots
    // the matrices from the first arg (client.dll's view-setup global) into
    // d3d11_rtx globals, filtered to the main world pass by (r8 & 0x400).
    // EndFrame consumes them via processExternalCamera(Main, ...), and the
    // per-draw classifier in CameraManager skips Main updates so the
    // engine-derived pose is authoritative and never overwritten.
    //
    // When OFF: the legacy per-draw classifier seeds Main from
    // dcs.transformData (cls12Recon decompositions). Decompositions are
    // inconsistent across VSes; Main alternates between unrelated cameras.
    //
    // When ON: ONE deterministic Main per frame, sourced from the same
    // matrix the engine uploads to cb2 for the main world pass.
    // NV-DXVK [EngineCam]: hardcoded ON for TF2 — the only build target.
    // The per-draw classifier path produces inconsistent worldToView
    // decompositions across VSes in the same frame (skybox cluster wins
    // canonical, main view's matrices get overwritten), causing the
    // alternating ship/feet/broken-view symptom. The engine-hook reads
    // the engine's actual main-pass matrix from R_DrawWorldMeshes' a1
    // argument once per frame and feeds it to Main authoritatively.
    // Default flipped to true so a fresh checkout works without conf
    // tweaks. Flip to false to A/B against the legacy override stack.
    RTX_OPTION("rtx", bool, useEngineHookMainCamera, true,
               "TF2/Titanfall2 only. When true, Remix's Main camera is "
               "driven by an engine-side trampoline at R_DrawWorldMeshes "
               "instead of by the per-draw classifier. Eliminates the "
               "'alternating ship/feet/broken view' jitter caused by "
               "inconsistent worldToView decompositions across vertex "
               "shaders. Requires the trampoline patch toggle "
               "kHookRDrawWorldMeshes to be active.");

    // NV-DXVK: engineHookMainCameraFrameDelay was REMOVED 2026-07-30. It fed
    // Main a capture N engine-frames behind the newest to phase-align it with
    // the geometry stream -- a compensation for the camera/geometry mismatch
    // rather than a fix for it, and therefore wrong on any frame where the
    // engine did not advance exactly once per Remix frame. Superseded by
    // useCamGeoLatch below, which removes the mismatch itself. If you find a
    // stale rtx.conf still setting it, the line is now inert and can be
    // deleted.

    // NV-DXVK [CamGeoLatch]: pair the engine-hook camera to the geometry it
    // belongs to, instead of compensating for the mismatch with a fixed delay.
    //
    // The engine.dll trampoline writes g_engineMainW2v on the engine's render
    // thread; the EndFrame consumer used to re-read that global at submit
    // time. Two independent samples of two different clocks with nothing
    // pairing them, so frame N's geometry could be rendered through a camera
    // from a later engine frame. Everything that MOVES gets displaced by the
    // gap -- confirmed on screen 2026-07-30, the viewmodel AND the moving
    // platform both zig-zag, which is why no viewmodel-side fix ever covered
    // it. engineHookMainCameraFrameDelay cancels this ON AVERAGE by reading
    // one frame stale; it cannot be right on frames where the engine does not
    // advance exactly once per Remix frame (observed at remixFrame=5971:
    // engine +2, capture +0).
    //
    // With this on, the camera is snapshotted at the frame's FIRST SubmitDraw
    // -- the moment its geometry starts being recorded -- and the consumer
    // uses that snapshot. Structural pairing, self-correcting across skipped
    // or doubled engine frames, and one less frame of camera latency than the
    // compensation it replaces.
    //
    // CONFIRMED on screen 2026-07-30, and in the log: latch=1 on 1555/1555
    // frames, and liveEf != engFrame on 1552 of them -- the old consumer was
    // reading a different engine frame's camera than the geometry almost
    // always, which is why a constant delay of 1 appeared to work. The delay
    // option and its ring were deleted in the same pass.
    //
    // Setting this False now falls all the way back to the pre-fix behaviour
    // (live read, no compensation of any kind), i.e. the zig-zag. It is kept
    // as an off-switch for isolating this mechanism, not as a tuning knob.
    RTX_OPTION("rtx", bool, useCamGeoLatch, true,
               "TF2/Titanfall2 only. Pair the engine-hook Main camera to the "
               "geometry recorded in the same Remix frame (latched at that "
               "frame's first draw) instead of re-reading the live capture at "
               "EndFrame. False reverts to the pre-fix live read, which "
               "reintroduces the moving-platform/viewmodel zig-zag. Falls back "
               "to the live read anyway on frames that recorded no draws.");

    // NV-DXVK [tf2_engine_cvars]: the five tf2StaticProp* options MOVED to
    // src/d3d11/tf2_options.h on 2026-07-30. They are read only from
    // d3d11_rtx.cpp, and rtx_options.h is pulled in by dxvk.pch.h, so every
    // edit here rebuilds ~213 translation units instead of 1. The rtx.conf
    // keys are unchanged (rtx.tf2StaticProp*) -- the config key comes from the
    // macro's category+name strings, not from which struct declares it.

    // NV-DXVK [BoneStablePropId fanout position-only] PROTOTYPE. TF2's path-10
    // PI prop-fanout (VS_2947c6 — the 3D-skybox/mountain terrain that reaches
    // Remix only via the spot-shadow pass) round-robins a pool of transient
    // vertex/index buffers. MakeBoneStablePropId keys identity on vbPtr/ibPtr,
    // so the SAME static prop gets a new propId whenever the engine cycles to a
    // different buffer (confirmed: BoneIdProbe.Bulker shows 342 distinct
    // fanout positions but 518 distinct hashes — same fanoutT, multiple
    // propIds). The unstable identity makes SpatialMap dedup miss every frame,
    // so the instance retires at numFramesToKeepInstances=1 the frame the
    // shadow source stops -> the geometry vanishes and pops back later. When
    // true, fanout draws (firstInstanceObjectToWorld != null with non-zero
    // rounded translation) derive identity from the STABLE rounded fanout
    // position + vertex stride only, dropping the rotating buffer pointers.
    // Static fanout terrain has a fixed, well-separated world position, so the
    // rounded position is both stable across the buffer rotation and distinct
    // per prop. Prereq for granting these instances the longer GC keep. Verify
    // via BoneIdProbe.Bulker distinct-hash count plateauing to ~distinct-
    // fanoutT count. Default off (legacy buffer-pointer identity).
    RTX_OPTION("rtx", bool, boneStablePropIdFanoutPositionOnly, false,
               "TF2/Titanfall2 only. PROTOTYPE: for bone fanout draws, key the "
               "stable prop identity on the rounded fanout position instead of "
               "the engine's rotating vertex/index buffer pointers, so static "
               "shadow-sourced terrain keeps one identity across the buffer "
               "arena rotation (fixes the missing/pops-in 3D-skybox geometry). "
               "Default off = legacy buffer-pointer identity.");

    // NV-DXVK [fanout instance split]. TF2's bone-instanced fanout draws
    // (d3d11_rtx path 10) submit ONE draw call carrying N per-prop transforms in
    // transformData.instancesToObject, and Remix historically made ONE RtInstance
    // for the whole batch. Dedup then has to identify a *batch*, and the batch is
    // not a stable entity: measured 2026-08-05 with a static camera, nInst
    // oscillates between 53/54 and 59/60 as ~6 props enter and leave every frame,
    // giving 89 distinct propIds and 86 distinct order-independent set-digests
    // across 89 reaps of a single object. Every one of those misses destroys the
    // instance, and because a freshly created instance goes through teleport()
    // (prevObjectToWorld := objectToWorld) the WHOLE batch is declared static for
    // that frame and its motion vectors are zeroed — the one-frame blur smear on
    // small instanced props, at 4-10 reaps/frame with ~51% of them respawns.
    //
    // When true, each element of instancesToObject gets its own RtInstance with
    // objectToWorld = drawO2W * instancesToObject[i] and instancesToObject=null,
    // keyed on a stablePropId derived from that prop's OWN rounded translation.
    // A membership change then costs one instance instead of invalidating ~54.
    //
    // GPU cost is unchanged: addPointInstancerBlas already reserved N surface
    // slots and writeGPUData already wrote a distinct transform into each, so the
    // same N surfaces with the same N transforms are produced either way. The
    // cost is CPU bookkeeping — m_instances grows from ~840 to roughly the PI
    // slot count (~3940 measured), i.e. ~5x on the GC walk, the SpatialMap and
    // instance update. Turn this off to A/B against that baseline without a
    // rebuild; it does not affect USD PointInstancers or replacement draws, which
    // never carry DrawCallTransforms::isFanoutBatch.
    RTX_OPTION("rtx", bool, splitFanoutInstances, true,
               "TF2/Titanfall2 only. Give every prop in a game-submitted bone "
               "fanout batch its own RtInstance instead of representing the whole "
               "batch with one, so a prop keeps its dedup identity and temporal "
               "history when the batch's membership changes (fixes the one-frame "
               "motion-vector blur on small instanced props). Costs ~5x CPU-side "
               "instances; GPU surface count is unchanged. Default on.");

    // NV-DXVK [T31Struct] 2026-08-05. DIAGNOSTIC, default off.
    //
    // Raw dump of the FULL 208-byte g_modelInst (t31) entry behind each fanout
    // placement. Only bytes 0..47 — the float3x4 the geometry needs — have ever
    // been read; the remaining 160 bytes per placement have never been looked at.
    //
    // This exists because every identity key tried so far was derived from the
    // TRANSFORM (rounded position, hybrid, basis) or from the SLOT INDEX
    // (charIdx), and both families are dead: transforms move, and charIdx is a
    // 256-entry per-draw window that wraps and merges props across draws. If the
    // engine puts a genuine per-prop handle anywhere dxvk can already see it, the
    // unread 160 bytes are the only place left. A per-prop lighting origin, a
    // variation seed, or — the layout worth hoping for — a PREVIOUS-FRAME matrix
    // at bytes 48..95 would each solve it, the last one exactly: prev-frame
    // matrix in frame N equals cur-frame matrix in frame N-1, which IS temporal
    // correspondence, no handle required.
    //
    // Dumps raw floats AND raw uint32 hex, because an integer handle is
    // unreadable as a float and vice versa. No thresholds, no classification —
    // the point is to see the bytes before deciding what they are.
    RTX_OPTION("rtx", bool, dumpFanoutInstanceStruct, false,
               "DIAGNOSTIC. Dump the full 208-byte per-instance g_modelInst entry "
               "for the first few placements of each fanout draw, once per frame, "
               "as both float and hex. Used to find a stable per-prop identity in "
               "the 160 bytes beyond the instance matrix. Self-limiting.");

    // NV-DXVK [keepStablePropIdInstancesLong] PROTOTYPE (step 2 of the
    // shadow-sourced-geometry fix). Geometry whose only submission route is
    // TF2's spot-shadow pass (VS_2947c6 3D-skybox/mountain terrain) vanishes
    // the frame the shadow culls it, because the instance retires at
    // numFramesToKeepInstances=1. Granting it the longer
    // numFramesToKeepSubViewInstances keep retains its BLAS/instance across
    // the shadow-off frames so it stays put. This is only safe once the prop
    // has a STABLE identity (rtx.boneStablePropIdFanoutPositionOnly) — with the
    // old rotating-pointer identity the long keep let per-frame duplicates
    // coexist and doubled the ordered-surface table. When true, any instance
    // with a non-zero stablePropId gets the long keep. Continuously-submitted
    // props (viewmodel, vehicles) are unaffected — they're updated every frame
    // so the keep window never elapses; only props that STOP being submitted
    // benefit. Pairs with boneStablePropIdFanoutPositionOnly. Default off.
    RTX_OPTION("rtx", bool, keepStablePropIdInstancesLong, false,
               "TF2/Titanfall2 only. PROTOTYPE: give any instance with a "
               "non-zero stablePropId the longer numFramesToKeepSubViewInstances "
               "GC keep, so shadow-sourced terrain survives the frames TF2's "
               "shadow pass culls it instead of vanishing. Only safe with "
               "rtx.boneStablePropIdFanoutPositionOnly=True (stable identity). "
               "Default off.");

    // NV-DXVK [EngineCam-Skybox]: short-circuit the sky classifier when
    // we want 3D-skybox geometry to flow into TLAS as regular ray-traced
    // content (mountains, distant ships, terrain). With the engine-hook
    // main-camera path active, the sky classifier latches CORRECTLY on
    // the 3D-skybox cb2 origin, which causes those VSes (~227k verts/frame
    // of mountain/terrain geo) to be sky-tagged and dropped from TLAS.
    // That's the standard "sky goes through raster + Hillaire atmosphere"
    // architecture, but Titanfall's 3D-skybox content is artistically
    // meaningful geometry (mountains visible from the dropship, distant
    // ships in formation) that the user wants ray-traced, not lost to
    // sky compositing.
    //
    // When true: SetSkyCategoryFromCb2 early-returns false (does NOT tag
    // any draw as InstanceCategories::Sky). All geometry — main world,
    // viewmodel, AND 3D-skybox — flows into TLAS for ray tracing.
    //
    // Trade-off: Hillaire atmosphere / sun NEE still runs (those don't
    // depend on sky-tagging), but the rasterized skybox path won't run
    // either (it's gated on sky-tagged draws existing). For the TF2 intro
    // that's actually what we want: the 3D-skybox geometry IS the visible
    // horizon, so rendering it as TLAS is the correct outcome.
    RTX_OPTION("rtx", bool, disableSkyTagging, false,
               "TF2/Titanfall2 only. When true, completely disables the "
               "sky-classifier draw tagging. 3D-skybox geometry (mountains, "
               "distant ships) flows into TLAS as regular ray-traced content "
               "instead of being routed through the rasterized/atmosphere "
               "sky path. Useful when the 3D-skybox carries level-meaningful "
               "geometry that the user expects to see in the ray-traced "
               "image.");

    // NV-DXVK [SkyMissMagentaProbe]: diagnostic — paint every primary-
    // miss pixel bright magenta in the final composite output. Used
    // alongside the widened [SkyTrace.primaryMiss] readback to disambig-
    // uate "black sky because no sky source" vs "black sky because far-
    // Z geometry occludes the miss shader". If you toggle this on and
    // the formerly-black sky turns magenta, the miss shader IS firing
    // and the bug is "no sky source bound" — go fix the SkyAutoCb2
    // classifier / probe population. If the sky stays black with this
    // on, the miss shader isn't even firing there → geometry occludes
    // and the bug is upstream.
    RTX_OPTION("rtx.debug", bool, visualizeMissedPixels, false,
               "Diagnostic. Override the composite output to bright magenta "
               "(1,0,1) for any pixel where primaryLinearViewZ == "
               "primaryDirectMissLinearViewZ (i.e. the primary ray missed "
               "all geometry). Used to verify whether the black-sky region "
               "of the frame is actually 'miss shader returned black' (turns "
               "magenta) or 'something is occluding and the miss shader "
               "never fired' (stays black).");

    // NV-DXVK [TF2SkyShader]: structural sky-draw tagger. Per-draw
    // detector in SetSkyCategoryFromCb2 (d3d11_rtx.cpp): tags as Sky
    // any draw whose depth-stencil state has DepthWriteMask=ZERO AND
    // PS samples a TextureCube SRV AND VS does NOT read
    // CBufModelInstance.c_modelInst. The conjunction is what makes it
    // tight: depthWrite=0 alone catches translucent surfaces;
    // TextureCube alone catches reflection-mapped meshes (metallic
    // hulls, glass); requiring all three plus the absence of a
    // per-model transform pins down fullscreen sky-quad draws.
    //
    // Ground truth (validated via the [SkyCandidate] probe in
    // FillMaterialData on a TF2 level intro):
    //   VS_ef94e6c7 + FS_62b1e6d4 (sky pass A)
    //   VS_962b9944 + FS_3bc1fc9b (sky pass B)
    // History: an earlier field-name heuristic (c_skyColor +
    // c_envMapLightScale .used) hid an interactive ship — see commit
    // 26af2ba6 regression note in SetSkyCategoryFromCb2.
    RTX_OPTION("rtx", bool, tagTF2SkyShaders, true,
               "TF2/Titanfall2 only. When true, draws matching the "
               "structural sky-pass signature (depthWrite=0 + PS samples "
               "TextureCube + VS has no per-model transform) are tagged "
               "InstanceCategories::Sky so they don't reach the TLAS as "
               "opaque primary world geometry. Independent of "
               "disableSkyTagging.");

    // NV-DXVK [SubViewSkyboxEmissiveOverride] kill-switch. When true, the
    // isSubViewSkybox -> BAKED_ALBEDO_AS_EMISSIVE promotion is SKIPPED, so the
    // distant 3D-skybox mountains render through the normal material path
    // (albedo + lighting) instead of baked-albedo-as-emissive. Diagnostic +
    // candidate fix for the "black mountain tops": with this True the albedo
    // G-buffer shows the raw texture sample (so [MtnRadiance] albedo is no
    // longer force-zeroed), and we see whether the tops light normally.
    RTX_OPTION("rtx", bool, disableSubViewSkyboxEmissive, false,
               "TF2/Titanfall2 only. Skip the isSubViewSkybox albedo->emissive "
               "override; distant 3D-skybox geometry renders via the normal "
               "albedo+lighting path instead.");

    // NV-DXVK [SubViewSkyboxProbeOnly]: the isSubViewSkybox HDRI dome (VS_eda5e,
    // a fisheye sky image on a ~25M-unit dome) is normally rasterized into the
    // SkyProbe cubemap AND submitted to the TLAS as a baked-emissive sphere for
    // primary-ray visible sky (tryHandleSky fall-through). That emissive dome is
    // what explodes into the rainbow streaks (its projected verts reach 26M
    // units). When this is true, after populating the probe we SkipSubmit the
    // dome so it never enters the TLAS — visible sky then comes from the probe/
    // matte instead. The HDRI still drives IBL/reflections (probe is populated
    // either way). Only affects isSubViewSkybox draws (AABB>5M sky dome); the
    // real 3D-skybox terrain/ships (AABB<5M) keep their TLAS reproject path.
    // DEFAULT OFF: dropping the dome from the TLAS makes the sky BLACK in this
    // build. Confirmed cause: the visible sky here is the emissive dome itself
    // — primary rays HIT it (that's why [SkyTrace.primaryMiss]=0), the matte is
    // bound only as composite sky-LIGHT, and the dome's transform is sub-view
    // (sky-camera) space so it can't serve as a main-camera primary-miss
    // background. Probe-only needs a real primary-miss sky source first
    // (env-map-on-miss sampling the SkyProbe cubemap, in main-camera space).
    // Leave this off until that path exists; flip on only to experiment.
    RTX_OPTION("rtx", bool, subViewSkyboxProbeOnly, false,
               "TF2/Titanfall2 only. Route the isSubViewSkybox HDRI sky dome to "
               "the SkyProbe cubemap only and drop it from the TLAS. WARNING: "
               "currently makes the sky black — the emissive dome IS the visible "
               "sky in this build and nothing substitutes for it on primary-ray "
               "miss yet. Off until an env-map-on-miss path samples the probe.");

    // NV-DXVK: flip the shading normal for sub-view (3D-skybox) geometry. The
    // sub-view reproject submits this geometry with inverted winding (already
    // worked around for culling by forcing double-sided), which also inverts the
    // shading normals — they face AWAY from the sun (N·L<0) so the path tracer
    // renders them black despite unshadowed sun + valid albedo (confirmed via
    // [MtnRadiance]/[SkyboxNormalProbe]). Negating normalInstanceToWorld restores
    // N·L>0 so the distant mountains light normally. Default false until verified.
    RTX_OPTION("rtx", bool, flipSubViewSkyboxNormals, false,
               "TF2/Titanfall2 only. Negate the shading normal for isSubView "
               "(3D-skybox) geometry to correct the reproject's inverted-winding "
               "normal flip, so distant skybox terrain receives direct lighting.");

    // NV-DXVK [EngineLightsCapture]: Tier 2 - dynamic point/spot lights.
    // The cbuffer dump caught a structured buffer "s_globalLights" with
    // 112-byte (= 7 x float4) elements bound as PS SRV. RDEF strips the
    // per-field names on $Element so we have to discover the layout by
    // dumping live values once. Workflow:
    //   1. Fly into a map with muzzle flashes / env_lights, set
    //      dumpEngineLightsBuffer=true, walk for ~30s.
    //   2. Grep [EngineLights.dump] in remix-dxvk.log - identify which
    //      vec4 holds position (large world coords), which holds colour
    //      (positive 0..N), which holds direction/radius, etc.
    //   3. Hardcode the layout, set useEngineLights=true, ship.
    RTX_OPTION("rtx.lights", bool, dumpEngineLightsBuffer, true,
               "Diagnostic - read s_globalLights structured buffer when bound, "
               "log first N elements as 7 float4s each. Use to identify the "
               "TF2 light struct layout. Throttled.");
    RTX_OPTION("rtx.lights", uint32_t, dumpEngineLightsCount, 16,
               "Number of populated s_globalLights entries to dump per call. "
               "Bumped to 16 so we see multiple light types (sun cascades, "
               "point, spot, projector) and can identify the per-type field "
               "layout from the variation.");
    RTX_OPTION("rtx.lights", uint32_t, dumpEngineLightsEveryNFrames, 600,
               "[EngineLights.dump] cadence in frames. 600 = once per ~10s @60fps. "
               "Bumped from 60 once layout is identified - dumps stay infrequent.");

    // NV-DXVK [EngineLightsCapture]: Tier 2 submission. Convert the
    // populated entries of TF2's mirrored s_globalLights buffer into
    // RtxLegacyLight per frame and feed them through the same path
    // legacy DX9 fixed-function lights take (RtxContext::addLights ->
    // SceneManager::addLight -> LightData::tryCreate -> LightManager).
    // Default ON so all visible map lights drive the path tracer
    // out of the box.
    // ON by default — but with engineLightSubmitMaxCount=64 cap so we
    // don't churn DxvkMemoryAllocator on the CS thread. Unlimited (cap=0)
    // crashes after ~30s with 1846 lights/frame; need anti-culling /
    // dedup before that's safe.
    RTX_OPTION("rtx.lights", bool, submitEngineLights, true,
               "Submit TF2's s_globalLights entries to the Remix scene "
               "as RtxLegacyLight every frame. Each populated buffer "
               "entry becomes a Point or Spot light depending on type.");
    RTX_OPTION("rtx.lights", float, engineLightIntensityScale, 1.0f,
               "Scale factor on captured colour (Diffuse RGB) before "
               "feeding to RtxLegacyLight. TF2 stores HDR-pre-scaled "
               "values so 1.0 is the correct default; tune if scenes "
               "are too bright/dim.");
    RTX_OPTION("rtx.lights", float, engineLightDefaultSpotInner, 0.7f,
               "cos(half inner cone) for type=2 spotlights when we "
               "haven't fully decoded v4/v5 cone params yet. 0.7 -> "
               "~45 degrees. Tune to match in-game spot footprints.");
    RTX_OPTION("rtx.lights", float, engineLightDefaultSpotOuter, 0.5f,
               "cos(half outer cone) for type=2 spotlights. 0.5 -> "
               "~60 degrees outer falloff.");
    RTX_OPTION("rtx.lights", uint32_t, engineLightSubmitLogEveryN, 256,
               "Throttle for [EngineLights.submit] confirmation logs. "
               "Once every N submit calls (one per frame). 0 disables.");
    RTX_OPTION("rtx.lights", uint32_t, engineLightSubmitMaxCount, 0,
               "Cap on number of lights submitted per frame. 0 = unlimited "
               "(NVIDIA's native behaviour - submit every game light and let "
               "LightManager dedup/anti-cull). Any non-zero value trims to "
               "the closest N by distance, which can drop bright in-range "
               "fixtures and pick out-of-range ones - only set non-zero as a "
               "stopgap if unlimited churns the allocator. Watch the "
               "[EngineLights.census] log: if resident-light count plateaus, "
               "unlimited is safe; if it climbs unbounded, dedup isn't "
               "sticking and lights need a stable identity hash.");
    RTX_OPTION("rtx.lights", bool, dumpEngineLightSamplesPerFrame, false,
               "Diagnostic - log one example RtxLegacyLight per type "
               "(t0/t1/t2/t3) every submission so we can verify that "
               "the position/colour/range/direction values look right "
               "in-world. Off by default to keep the log clean.");
    RTX_OPTION("rtx.lights", bool, dumpEngineLightFieldStats, false,
               "Diagnostic - one-shot statistical analysis of the "
               "s_globalLights buffer. Walks every populated entry, "
               "computes per-type per-vec4-component (min, max, "
               "isConstant) and logs a summary so unknown encoding "
               "fields can be identified. Logs once per session per "
               "buffer pointer (re-fires if the buffer is reallocated, "
               "i.e. on map change).");
    RTX_OPTION("rtx.lights", bool, dumpEngineLightShaderAsm, false,
               "Diagnostic - when a shader is created that declares "
               "s_globalLights, disassemble its DXBC via D3DDisassemble "
               "and write the asm to a file in the logs directory "
               "(tf2_<TYPE>_<HASH>.asm). Default OFF - layout already "
               "decoded from prior session. Flip to true if a future "
               "TF2 build changes the s_globalLights struct.");

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    RTX_OPTION("rtx", bool, logLegacyHashReplacementMatches, false, "");

    // NV-DXVK [Coverage]: when true, and while the Diffuse Albedo debug
    // view is selected, the geometry resolver bins every resolved primary
    // surface by material type and the result is logged as two [Coverage]
    // lists of vertex-shader hashes: OpaquePrimary and NonOpaquePrimary.
    // A translucent / ray-portal surface resolved as primary shows up in
    // NonOpaquePrimary — those are the shaders that render black-blotting
    // in Diffuse Albedo while staying invisible in the opaque-only Raw
    // Albedo view. Logging is throttled to one dump per 3 frames.
    // !! PERFORMANCE: this is a ~104 ms/frame switch at 1080p, not just logging.
    // It also drives cb.perfCoverageWrites, which arms a block of 52 atomics per
    // primary hit at the end of opaqueSurfaceMaterialInteractionCreate. Several
    // of those target one shared address, so a 2.07 Mpix dispatch serialises on
    // a single word: gb_primaryRays goes 27 ms -> 131 ms with this on.
    //
    // That coupling is the bug this flag already caused once. For three sessions
    // the GPU writes ran unconditionally while this option gated only the CPU
    // readback, and because atomic serialisation has no expensive instruction,
    // every hotspot hunt came back empty and the emptiness was misread as
    // "uniformly throttled execution". Do not turn this on during any timing
    // work, and do not turn it on to feed rtx.perfAutoSweep's census step - it
    // is global and would contaminate every step of the sweep.
    RTX_OPTION("rtx", bool, logSurfaceCoverage, false,
               "DIAGNOSTIC, COSTS ~104 ms/frame: while on the Diffuse Albedo "
               "debug view, logs the resolved primary surfaces split into "
               "OpaquePrimary / NonOpaquePrimary vertex-shader lists so "
               "translucent surfaces blotting the image can be identified by "
               "shader hash. Also arms the per-primary-hit coverage atomics - "
               "never leave this on for a performance measurement.");

    // NV-DXVK [MvRaw]: gate for the raw per-instance motion-vector record in
    // rtx_instance_manager.cpp. See the long note at the emit site for how to
    // read the fields. Deliberately NOT folded into logSurfaceCoverage: that
    // one arms the ~104 ms/frame coverage atomics, which would change frame
    // pacing and therefore change the very instance-update timing this probe is
    // measuring. It has to be armable on its own.
    RTX_OPTION("rtx", bool, logMotionVectorRaw, false,
               "DIAGNOSTIC, HIGH LOG VOLUME: logs [MvRaw], one raw line per instance "
               "per update, for the motion-vector flash hunt (geometry blurs as though "
               "it moved while standing still). Per row: mv = the motion the vector "
               "encodes, rep = the motion that actually happened, pml = whether "
               "prevObjectToWorld is this object's own last position or someone "
               "else's, fsl = frames since the instance was last updated. The fault is "
               "mv large while rep ~ 0; pml and fsl then separate a wrong BlasEntry "
               "pairing from a stale/skipped instance. Also logs the DrawCallCache "
               "pairing decision (pairKind/pairScore/pairAge) so the 612ff00d recency "
               "ranking can be confirmed or ruled out. Uncapped and unsampled on "
               "purpose - the artifact lasts a split second. Gameplay-gated. "
               "Leave False for normal play.");

    // NV-DXVK [perf]: gate for the [Perf.PrepScene] / [IdxStashPool] CPU
    // sub-split in SceneManager::prepareSceneData. This used to ride on
    // logSurfaceCoverage, which is the ~104 ms/frame switch documented above -
    // so the only way to read the split was to quadruple the frame first and
    // measure a completely different workload. prepareSceneData is currently
    // ~99.9% of the CPU tail (18-42 ms), i.e. the actual frame bottleneck, so
    // its instrumentation needs a gate that costs nothing to turn on. This one
    // does: nine steady_clock::now() calls (~41 ns each) and one Logger::warn
    // every ten frames.
    RTX_OPTION("rtx", bool, logPrepSceneSplit, false,
               "DIAGNOSTIC, ~free: logs the [Perf.PrepScene] CPU wall-time "
               "sub-split of SceneManager::prepareSceneData (lightMatch / gc / "
               "setup1 / instSetup / merge / accelLight / surfMat / cull / "
               "tlas / tail, in microseconds), the [Perf.Merge] second-level "
               "split of mergeInstancesIntoBlas, and [IdxStashPool] health - "
               "all on the same frames, once every ten frames. Safe to leave "
               "on during timing work, unlike rtx.logSurfaceCoverage, which "
               "these used to be gated on.");

    // NV-DXVK [perf]: master switch for the geometry-investigation probes in
    // rtx_accel_manager.cpp — [SpikeRB], the [SpawnGeomDiag.*] family and
    // [TlasCensus]. These were built for specific bugs (the s2s mangle, the
    // PointInstancer wrong-BLAS-pool hypothesis, the view1/view2 geometry
    // swap) and left switched on after those investigations closed, where
    // they cost real per-frame time: [SpikeRB] alone issues two buffer copies
    // and then walks EVERY triangle of the captured draw on the CPU, every
    // frame, and its tag is not in log.cpp's filter list so it also writes to
    // disk. Most of the [SpawnGeomDiag.*] tags ARE filtered, which saves the
    // file I/O but not the str::format that built the string.
    //
    // Runtime rather than compile-time (cf. kEnableRtxDebugProbes, which gates
    // the older probes D/E) specifically so these can be brought back by
    // editing rtx.conf instead of rebuilding — they are worth keeping. Cost
    // when off is one bool load per site.
    // NV-DXVK [FindSim probe]: vertex-shader hashes to trace through
    // InstanceManager::findSimilarInstance. The probe reports, per draw,
    // whether dedup matched at the EXACT stage (getDataAtTransform, keyed on
    // stablePropId or else the raw matrix bytes) or fell through to NEAREST
    // (getNearestData within uniqueObjectDistance), and for a nearest miss
    // which filter clause rejected each candidate (already-updated-this-frame
    // / material-hash / sub-prim).
    //
    // That is the decisive read on "why is this object re-created and reaped
    // every frame": an exact miss with propId=0 means the matrix bytes moved
    // (camera-facing billboards rewrite their rotation every frame), and the
    // nearest counters then say whether the fallback could have rescued it.
    //
    // Was hardcoded to two VS hashes from closed investigations
    // (0x2904d2163ef31a17, 0x29146e1dd50b0314); this makes it aimable from
    // rtx.conf with no rebuild. Empty = off. Cost when empty is one
    // hash-set lookup per draw.
    RTX_OPTION("rtx", fast_unordered_set, findSimilarProbeVsHashes, {},
               "DIAGNOSTIC: vertex-shader hashes to trace through "
               "findSimilarInstance. Logs [FindSim] lines showing whether "
               "instance dedup hit the exact-transform stage or the nearest-"
               "neighbour stage, and which filter clause rejected candidates. "
               "Use to diagnose per-frame instance churn (flicker). Empty = off.");

    // NV-DXVK [suppressStablePropIdVsHashes]: force stablePropId back to 0 for
    // the listed vertex shaders, so SpatialMap dedup keys on the TRANSFORM BYTES
    // instead of the engine-derived prop identity.
    //
    // WHY THIS IS NOT A HACK: 0 is already the documented contract of both
    // producers — MakeBoneStablePropId returns 0 when no identity is available
    // and the caller is specified to fall back to matrix-bytes hashing. This
    // switch only lets us reach that state deliberately, per shader, without a
    // rebuild.
    //
    // WHAT IT TESTS. Measured 2026-08-05 on VS 0x292b6ba0d1854f28: for a
    // stationary object (BlasEntry pointer and SpatialMap centroid identical
    // every frame) the propId round-robins with period EXACTLY 3 —
    // 0x9badf6.. -> 0x19f7da.. -> 0x38d6cf.. -> repeat — because
    // MakeBoneStablePropId hashes vbPtr/ibPtr/vbOffset/ibOffset and TF2 rotates a
    // three-deep transient buffer arena. The map therefore always holds the
    // PREVIOUS frame's key and getDataAtTransform misses 100% of the time. Every
    // lookup is pushed into the nearest-neighbour fallback, which is the only
    // stage that tests the material hash — so any material-hash wobble (texture
    // streaming, mip promotion) turns into a respawned instance and a reap.
    //
    // The transform for these draws IS byte-stable, so with propId suppressed the
    // exact stage should hit and the material hash should never be consulted.
    // Confirm with [FindSim] stage=exact hit=1 and a drop in [ReapJoin] respawn=.
    //
    // If it works, the FIX belongs at the propId producer (stabilise the identity
    // against the rotating arena), not here — this option stays diagnostic.
    // Empty = off; cost when empty is one hash-set lookup per bone-anim draw.
    // NV-DXVK [MvRaw aiming]: restrict rtx.logMotionVectorRaw to these vertex
    // shaders. EMPTY = every shader, i.e. exactly the pre-existing scene-wide
    // behaviour — this option can only ever narrow, never change what a row says.
    //
    // WHY IT WAS NEEDED. [MvRaw] is deliberately raw and uncapped: one row per
    // instance per update per frame, no thresholds, no sampling, because the
    // artifact lasts a single frame and any cap can drop the one row that
    // matters. That is the right design and it is kept. But scene-wide it is
    // ~840 rows/frame through the shared Logger mutex, which is not runnable
    // long enough to catch a random flash. The probe's own comment already
    // prescribes the fix — "narrow by id or vs AFTER a flash frame is
    // identified" — and the churn census of 2026-08-05 identified the
    // population: 568 objects churn, but the top 10 cause 23% of it and the top
    // 50 cause 55%, almost all on the point-instancer fanout shaders.
    //
    // Narrowing by VS is lossless in the way a cap is not: every row for the
    // selected shaders is still logged, so a one-frame flash on those objects
    // cannot be sampled away. It only blinds you to shaders you did not select,
    // which is a decision you make once and can see in the conf.
    RTX_OPTION("rtx", fast_unordered_set, motionVectorRawVsHashes, {},
               "DIAGNOSTIC: restricts rtx.logMotionVectorRaw to these vertex "
               "shader hashes. Empty = all shaders (scene-wide, very high log "
               "volume). Does not change what is logged per row, only which "
               "draws are eligible.");

    RTX_OPTION("rtx", fast_unordered_set, suppressStablePropIdVsHashes, {},
               "DIAGNOSTIC: vertex-shader hashes for which stablePropId is "
               "forced to 0, making instance dedup key on the object-to-world "
               "matrix bytes instead of engine prop identity. Use when a prop's "
               "propId is unstable across frames and the exact-match dedup stage "
               "misses every frame. Empty = off.");

    // NV-DXVK [InstUpProbe]: the flicker's prime remaining suspect (handoff V7
    // §5.2) is the per-frame upload of m_mergedInstances into the device-local
    // m_vkInstanceBuffer — a partial or late-landing staging copy would leave a
    // contiguous range of TLAS instances stale/garbage for exactly one frame,
    // which is what the aligned census measures (whole-VS ordSeen=0 for single
    // frames while CPU bookkeeping is clean). This probe closes that gap
    // byte-for-byte: prepareSceneData keeps the exact staged bytes in a ring,
    // buildTlas records a GPU copy of the same buffer regions in the same
    // command stream the TLAS build consumes them, and the two are compared
    // when the ring slot is recycled. Runtime option (not
    // kEnableRtxDebugProbes, which is compiled out) so it can be switched from
    // rtx.conf without a rebuild. Cost per frame: ~64B×instances memcpy +
    // one same-size GPU transfer + a memcmp — no CPU/GPU sync is introduced
    // (readback is 4 frames deferred).
    // NV-DXVK [FirstBakeHold]: DEFAULT OFF — the 2026-08-02 hold attempt did
    // NOT cure the visible flicker (user-confirmed identical) and can itself
    // render a garbage BLAS for the whole hold when the FIRST stash was taken
    // from an entry whose own bake was still source-pending (the stash-chain
    // guard only protects RE-stashes). Left in place, gated off, so the
    // machinery can be revisited with a validity check on the stashed bake.
    RTX_OPTION("rtx", bool, firstBakeHold, false,
               "Render the previously-linked BLAS for an instance whose relink "
               "destination entry's first bake is still source-pending, instead "
               "of the (possibly collapsed) fresh bake. EXPERIMENTAL: did not "
               "cure the TF2 flicker and can hold a garbage BLAS when the "
               "stash source was itself pending; keep off unless testing.");

    // NV-DXVK [InstUpBarrier]: the flicker fix this probe chain led to — see
    // the barrier comment in AccelManager::prepareSceneData. Runtime-gated so
    // the causality can be A/B'd from rtx.conf: barrier on + dropouts gone,
    // barrier off + dropouts back = proven. Default ON (it is a correctness
    // barrier; cost is nil except on frames where the race would have fired).
    RTX_OPTION("rtx", bool, instanceBufferWarBarrier, true,
               "Execution barrier ordering the previous frame's TLAS-build "
               "reads of the instance buffer before this frame's rewrite of "
               "it (upload copy + PI culling shader). The build's input read "
               "is a raw vkCmd call invisible to dxvk's hazard tracking, so "
               "without this nothing prevents cross-frame overlap on a "
               "GPU-backlogged frame — the single-frame whole-batch geometry "
               "dropouts. Disable only to A/B-prove the mechanism.");

    RTX_OPTION("rtx", bool, debugInstanceUploadProbe, false,
               "DIAGNOSTIC: verify the merged TLAS instance-buffer upload "
               "byte-for-byte against what the TLAS build consumes, in the "
               "build's own command stream. Logs [InstUpProbe] cmp=OK "
               "heartbeats and loud [InstUpProbe.bad] per-instance dumps on "
               "any divergence. Join mismatch frames against [ResolveCensus] "
               "ordSeen=0 dropout frames: mismatch on a dropout frame = the "
               "upload race IS the geometry flicker; OK on dropout frames "
               "exonerates the instance bytes (bug is build/driver side).\n"
               "DEFAULT FLIPPED true -> false 2026-08-06. IT IS NOT CHEAP AND IT "
               "DEFAULTED ON. Every frame it maps a host-visible staging buffer "
               "and memcmps 256 KB OUT OF IT, then resize + memcpy 256 KB back. "
               "Reads from host-visible memory run far below cached-RAM "
               "bandwidth, so that compare costs far more than its byte count "
               "suggests. Measured with [Perf.Accel]: this probe plus "
               "rtx.logTlasSet were together ~4.8 ms of a 73.6 ms frame -- the "
               "whole of what HANDOFF_PERF_2026-08-06_v4 sec 4b recorded as "
               "'accelLight 4.9 ms on a stage with zero lights'. With both off, "
               "AccelManager::prepareSceneData falls to ~50 us. Turn this back "
               "on only for an active instance-upload flicker hunt, and off "
               "again after -- a diagnostic that defaults on and eats the stage "
               "it lives in is the same trap rtx.perfGapSampler set (v4 sec 0c).");

    // NV-DXVK: per-draw sub-view reproject gate trace. The existing probes on
    // this path cannot answer "which gate rejected this draw":
    //   [SubViewGateCounts] aggregates per FRAME with no VS attribution, but
    //     the 2026-07-29 capture proved the MAIN/SKY split happens WITHIN a
    //     frame (179 of 464 multi-create frames carry both spaces), so a
    //     per-frame bucket cannot attribute it.
    //   [ReprojectGate] only reaches the !inSubViewPass case — it sits inside
    //     an `if (g_engineSkyCamOriginValid != 0u)`, so the failSkyValid case
    //     is structurally unreachable — and additionally drops anything past
    //     200u from the sky cam, at 2 lines/frame.
    //   [SubViewMiss] fired 0 times, which rules out a marginal distance
    //     failure but says nothing about the two gates upstream of it.
    // This option logs EVERY outcome for the named VSes, including both
    // failure modes, so the flip can be attributed per draw. Empty = off;
    // cost when empty is one hash-set lookup per draw.
    RTX_OPTION("rtx", fast_unordered_set, subViewGateProbeVsHashes, {},
               "DIAGNOSTIC: vertex-shader hashes to trace through the sub-view "
               "reproject gate. Logs [SubViewGate] lines carrying skyValid, "
               "inSubViewPass, r8, cb2 origin, sky-cam origin and distSq for "
               "every draw, whether it reprojected or was skipped, with the "
               "deciding clause named. Use to diagnose sub-view content that "
               "reaches the scene in raw sky space instead of main-world "
               "space. Empty = off.");

    RTX_OPTION("rtx", bool, piForceOpaque, false,
               "DIAGNOSTIC A/B: force VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT on "
               "every PointInstancer instance, which skips the any-hit shader "
               "entirely and therefore disables alpha testing for that "
               "geometry. Foliage will render as solid untextured cutouts - "
               "visually wrong on purpose. "
               "WHY: everything that can DELIVER this geometry to a ray is "
               "verified constant and correct (buffer contents, BLAS, "
               "surfaces, surfaceIndex, world transform, AS build, AS bind, "
               "instance mask 0x8, geometry flags 0x3), yet the trees render "
               "0 pixels on ~35% of frames with the camera frozen. Neither "
               "FORCE_OPAQUE nor FORCE_NO_OPAQUE is set, so any-hit runs and "
               "the alpha test is the only remaining per-frame variable. "
               "Flicker STOPS with this True => the alpha test is the "
               "mechanism; look at the foliage albedo's mips/streaming. "
               "Flicker CONTINUES => alpha testing is innocent too. "
               "Deliberately declared under 'rtx.' and NOT "
               "'rtx.pointInstancer.', because that namespace has a known "
               "plumbing defect - see rtx_point_instancer_system.cpp:256, "
               "where enable() was returning its compile-time default instead "
               "of the parsed conf value, which silently made an entire A/B "
               "meaningless.");

    RTX_OPTION("rtx", std::string, logDenyTags, "",
               "DIAGNOSTIC: per-run edit to the logger's tag denylist, so a "
               "silenced probe can be brought back without a rebuild. "
               "Comma-separated tag prefixes. A plain entry SILENCES that "
               "prefix; a leading '-' UN-SILENCES it by dropping every built-in "
               "entry that starts with it. Example: "
               "\"-[SpawnGeomDiag., [BulkPush], [InstReap]\" restores the "
               "SpawnGeomDiag family and silences two noisy tags. Matching is "
               "by prefix throughout, so '-[Zig' clears the whole [Zig* family. "
               "Empty = use the built-in list unchanged. Applies to lines "
               "logged after the conf is parsed; startup lines always use the "
               "built-in list.");

    RTX_OPTION("rtx", bool, logAlbedoSrgbProbe, false,
               "DIAGNOSTIC: logs [AlbedoSrgb] once per distinct albedo texture, "
               "recording the bound image view's raw VkFormat, whether that "
               "format carries the sRGB colour-space flag, and the resulting "
               "OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_SRGB.\n"
               "That flag decides whether the shader runs its software "
               "gammaToLinear() or leaves the hardware sampler's decode to "
               "stand. Getting it wrong in the 'set but not actually decoded' "
               "direction leaves albedo in gamma space, which roughly halves "
               "its linear channel ratios and raises its level - i.e. brighter "
               "AND greyer. Emits at material build time, deduplicated by "
               "texture hash, so the cost is a handful of lines per level.");

    RTX_OPTION("rtx", bool, logMaterialMapProbe, false,
               "DIAGNOSTIC: logs [MatMaps] once per (vertexShader, material) pair, "
               "recording which PBR maps the RDEF name-matching actually bound - "
               "normal, roughness/gloss, metallic/spec, emissive, AO - with each "
               "one's image hash and VkFormat.\n"
               "The map names were verified against TF2's character shaders; world, "
               "brush and viewmodel shaders need not name their SRVs the same way. "
               "When a name does not match, that channel silently falls back to the "
               "rtx.legacyMaterial.* constant, which looks like a working-but-wrong "
               "material rather than a missing one. Keyed by VS so a shader that "
               "binds nothing is visible next to one that binds everything.");

    RTX_OPTION("rtx", bool, logGeomDiag, false,
               "DIAGNOSTIC, COSTS REAL TIME: enables the geometry-investigation "
               "probes - [SpikeRB] per-frame BLAS position/index readback and "
               "triangle walk, the [SpawnGeomDiag.*] PointInstancer routing and "
               "batch-inventory dumps, and the [TlasCensus] per-instance "
               "inventory. Off by default; turn on only while chasing a "
               "geometry bug, never during a performance measurement.");

    // NV-DXVK [TlasSet]: WHICH geometry is in the TLAS, not how much. Left ON by
    // default for the same reason as logMaterialChurn - it is the current line of
    // investigation, and the cost is one hash per instance plus a ~938-element
    // sort per frame. Every count-based instrument reports this scene as
    // perfectly stable while meshes are visibly missing, so a count is known to
    // be the wrong shape for this artifact.
    RTX_OPTION("rtx", bool, logTlasSet, false,
               "DIAGNOSTIC, O(instances) PER FRAME: logs [TlasSet], one line per "
               "frame per TLAS giving the order-independent signature of the "
               "instance SET fed to the acceleration-structure build, plus "
               "appeared/vanished counts diffed against the previous frame, and "
               "maskZero/blasNull for entries that are present but will be "
               "skipped by the tracer. [TlasSet.gone] names the BLAS and world "
               "position of what left (capped at 8 per frame). vanished>0 with a "
               "still camera is the geometry leaving the TLAS, caught at the last "
               "point before the build.\n"
               "DEFAULT FLIPPED true -> false 2026-08-06, AND THE WORD 'cheap' "
               "REMOVED FROM THIS STRING -- it was the claim that justified the "
               "on-by-default. Per frame it builds one entry per TLAS instance, "
               "std::sorts ~9,000+ of them, linear-diffs against last frame, then "
               "copies the whole vector to keep for the next diff. Measured with "
               "[Perf.Accel]: this plus rtx.debugInstanceUploadProbe were "
               "together ~4.8 ms of a 73.6 ms frame.\n"
               "IT WAS ALSO PRODUCING NOTHING. '[TlasSet' sits in the project's "
               "rtx.logDenyTags, and log.cpp applies that filter inside emitMsg, "
               "i.e. AFTER str::format has already built the string -- so the "
               "sort, the diff, the copy and the formatting were all paid and "
               "every line was then discarded. If you enable this, REMOVE "
               "'[TlasSet' from rtx.logDenyTags in the same edit or it will cost "
               "the frame and tell you nothing.");

    // NV-DXVK [MatChurn]: the material/texture identity churn counters. Left ON
    // by default because it is the current line of investigation and the cost is
    // a handful of integer increments plus ONE Logger::info per gameplay frame -
    // roughly 2000 lines over a several-minute run, against logs that routinely
    // run to tens of megabytes. Turning it off must never be the reason a run
    // comes back without the measurement.
    RTX_OPTION("rtx", bool, logMaterialChurn, true,
               "DIAGNOSTIC, ~free: logs [MatChurn], one aggregate line per "
               "gameplay frame giving the rate at which NEW material and texture "
               "identities are created (matNew / texNew / imgNew), how the "
               "bindless texture table changed (blChg / blDrop / blRecov), and "
               "replacement-asset streaming transitions (mt*). Join f= against "
               "the on-screen flick: in a steady scene with a still camera the "
               "'New' columns should all be 0.");

    // When enabled, vkDeviceWaitIdle is called before mapping the
    // host-visible Coverage buffer in dispatchDebugView. Without this
    // sync, mapPtr returns a pointer to data still being written by
    // in-flight GPU dispatches — the CPU reads whichever frame the GPU
    // last finished (typically N-2 with triple buffering), then the
    // dump compares those stale counts against the CURRENT frame's
    // m_reorderedSurfaces.size(), producing phantom "OOB at slot X"
    // reports that the GPU's own per-callsite OOB probes never saw
    // (because at write-time, the index WAS in range for that older
    // frame's surfaceCount). Diagnostic-only; enables stalls every
    // frame the dump runs, so do NOT leave on in normal play.
    // NV-DXVK [Coverage PickRegion]: screen rectangle (minXFrac, minYFrac,
    // maxXFrac, maxYFrac) in normalized [0,1] coords (origin top-left) that the
    // color-independent surface-coverage attribution probe bins by surfaceIndex.
    // Whatever vertex shaders draw inside this rect are ranked by pixel count in
    // the [Coverage] PickRegionVS log lines, regardless of color — used to name an
    // un-tinted object (e.g. the TF2 first-person weapon) that the red-gated DVRED
    // probe can't see. Default is the bottom-right quadrant; shrink/move it at
    // runtime (no rebuild) to tighten onto a single object. Requires
    // rtx.logSurfaceCoverage and an active debug view. A degenerate rect
    // (max <= min) disables the probe.
    RTX_OPTION("rtx", Vector4, surfaceCoveragePickRegion, Vector4(0.5f, 0.5f, 1.0f, 1.0f),
               "Normalized screen rectangle (minXFrac, minYFrac, maxXFrac, maxYFrac) "
               "for the color-independent surface-coverage pick probe. Ranks the "
               "vertex shaders drawing inside the rect by pixel count in the "
               "[Coverage] PickRegionVS log. Default bottom-right; sweep to identify "
               "an object by VS hash. Needs rtx.logSurfaceCoverage + a debug view.");

    // NV-DXVK [Coverage PickRegion2]: a SECOND, independent pick rect attributed
    // in the same frame (own regions). Logs as [Coverage] PickRegion2 /
    // PickRegion2VS. Default = screen center. Use it to compare two points
    // (e.g. the streak at y=0.25 vs the center) in one capture.
    RTX_OPTION("rtx", Vector4, surfaceCoveragePickRegion2, Vector4(0.499f, 0.499f, 0.501f, 0.501f),
               "Second normalized screen rectangle for the surface-coverage pick "
               "probe, attributed alongside rtx.surfaceCoveragePickRegion. Logs as "
               "[Coverage] PickRegion2 / PickRegion2VS. Default screen center.");

    // NV-DXVK [SerializeSceneBuild] — one decisive test for GPU-side
    // nondeterminism, with a defined negative outcome.
    //
    // State of the flicker investigation that motivates it: with the camera
    // byte-identical across 514 of 532 consecutive frame pairs, a specific set
    // of vertex shaders flips drawn<->lost on 35-52% of adjacent frames while
    // two others in the same frames never flip (0.8-0.9%). On the frames the
    // geometry vanishes, every CPU-side probe reads correct — ordered-list
    // pushes, PI batch/instance counts, BLAS asLive/builtAtCap, TLAS built and
    // bound, instance mask 0x8, geometry flags 0x3, and the primary ray traces
    // OBJECT_MASK_ALL. A paired adjacent-frame diff over every per-frame
    // numeric field, controlled against non-flip pairs, finds nothing (largest
    // excess 9.2 points, and the batch-churn fields go the WRONG way).
    //
    // Identical CPU input + identical camera + different GPU output means the
    // nondeterminism is on the GPU: a missing barrier, or state carried across
    // frames in GPU memory that nothing orders. This flushes and fully idles
    // the device after the whole scene build (BLAS + TLAS + surface/instance
    // uploads + PointInstancer culling) and before any raytracing pass reads
    // it, which makes every such hazard in that span unobservable.
    //
    // Read it as a bisect, not a fix — it is far too expensive to ship:
    //   flip rate collapses  -> a scene-build/raytrace ordering hazard exists;
    //                           bisect by moving the wait earlier per pass.
    //   flip rate unchanged  -> GPU ordering across that boundary is EXONERATED
    //                           outright and the cause is in the acceleration
    //                           structure contents themselves. Not a weak
    //                           result — it removes the entire class.
    RTX_OPTION("rtx", bool, debugSerializeSceneBuild, false,
               "DIAGNOSTIC: flush the command list and vkDeviceWaitIdle after "
               "SceneManager::prepareSceneData and before any raytracing pass, "
               "so every scene-build GPU write is complete before anything "
               "reads it. Makes missing barriers across that boundary "
               "unobservable. Enormously expensive (full pipeline stall every "
               "frame) - a bisecting instrument for nondeterministic geometry "
               "loss, never a shipping setting. MEASURED: changes nothing - a "
               "full flush+waitForIdle across that boundary leaves flip rates "
               "identical, so GPU ordering there is exonerated. Kept as a "
               "bisecting instrument.");

    // NV-DXVK [ABSweep]: alternate one option BASE/TEST inside ONE session, on
    // a hotkey, instead of comparing two separate runs. Currently sweeps
    // rtx.enableSeparateUnorderedApproximations.
    //
    // This exists because across-run comparison is worthless in this title.
    // Flip rate is LOCATION-dependent for every shader: 0x2af9b90d63850ec3
    // measured 0.0% flip at one spot and 46.1% at another, while the target
    // went 43.1% -> 14.6% over the same move. Three hypotheses were graded on
    // cross-run numbers and all three were really measuring where the player
    // was standing.
    //
    // Interleaving fixes that: the player holds position, arms alternate every
    // N frames, and both see the SAME viewpoint, geometry and drift. Any
    // residual scene change is spread evenly across both arms instead of
    // landing on one. Start in BASE so the baseline is measured at that exact
    // spot first and a quiet location cannot masquerade as a fix.
    //
    // F9 toggles. Every frame logs [ABSweep] f= arm= sepUnordered= cycle= so
    // per-frame Coverage counts can be joined to the arm that produced them.
    RTX_OPTION("rtx", uint32_t, abSweepFramesPerPhase, 300,
               "Frames spent in each arm of the F9 A/B sweep before flipping. "
               "The swept variable is currently "
               "rtx.enableSeparateUnorderedApproximations. 300 at ~10 fps "
               "under the diagnostic load is ~30s per arm - ample to separate "
               "a 40% flip rate from a near-zero one, while keeping a 4-arm "
               "run to about two minutes of holding position.");
    RTX_OPTION("rtx", uint32_t, abSweepCycles, 2,
               "Number of BASE->TEST cycles the F9 A/B sweep runs before "
               "stopping itself. 2 cycles = 4 arms. More than one crossing is "
               "the point: a single BASE->TEST transition cannot be told apart "
               "from the scene going quiet on its own.");
    RTX_OPTION("rtx", bool, abSweepExitOnFinish, true,
               "Terminate the process when the F9 A/B sweep completes, the "
               "same way rtx.perfAutoSweepExitOnFinish does. On by default so "
               "the capture ends itself and the log is closed and ready to "
               "read without holding position waiting for it. Uses "
               "TerminateProcess, not exit(), because this fork's clean "
               "shutdown path calls a cached client.dll pointer after the "
               "engine has unloaded that module.");

    // NV-DXVK [HitCensus]: dense per-VS primary-hit census, logged alongside
    // [MtnRadiance] (same readback, same frame stamp).
    //
    // Exists because nothing else distinguishes "the ray never hit this
    // geometry" from "the ray hit it and the pixel ended up owned by another
    // surface". [Coverage] counts pixels after resolution; the [MtnRadiance]
    // grid samples 288 texels and is far too sparse for a shader covering a
    // few percent of screen. This histograms EVERY texel of the primary
    // surface-index buffer, so per frame you get the exact number of primary
    // rays that resolved to each vertex shader.
    //
    // Join [HitCensus] f=N against [Coverage] gpuFrame=N for the same VS:
    //   hits > 0, Coverage pixels == 0 -> hit, then lost after resolution.
    //   hits == 0                      -> never hit, despite the instance
    //                                     being provably in the TLAS with the
    //                                     right mask/transform/BLAS.
    // Those point in opposite directions, which is the point - it is a
    // measurement, not a hypothesis test, so it cannot come back "refuted".
    //
    // Costs one W*H pass on the existing async decode thread (~518k texels at
    // 960x540). Needs rtx.logSurfaceCoverage on and coveragePickRegionOnly off,
    // same as [MtnRadiance].
    RTX_OPTION("rtx", bool, logPrimaryHitCensus, false,
               "DIAGNOSTIC: log [HitCensus] - the number of primary rays that "
               "resolved to each vertex shader, counted over every pixel of "
               "the shared surface-index buffer rather than a sparse grid. "
               "Join against [Coverage] for the same frame to tell a ray that "
               "never hit geometry from one that hit it and lost it "
               "afterwards. Also reports invalidSurf and oobSurf counts.");

    // NV-DXVK [MtnRadiance] steering. The probe dumps RAW per-pixel primary-hit
    // GBuffer (viewZ, albedo, direct/indirect, normal, surfaceIndex -> VS, sv,
    // svSky, nb) on a 24x12 grid. Its sample band and distance gate used to be
    // hardcoded to the upper 60% of screen and |viewZ| > 1e4, which is correct
    // for the distant-mountain question it was written for and useless for any
    // other. Both are now options so the same instrument can be aimed at an
    // arbitrary object.
    //
    // Aim it at geometry that intermittently disappears and read, per frame:
    //   viewZ far + vs = a different (sky/backdrop) VS -> the ray never hit the
    //     object; the defect is in AS content / traversal, not shading.
    //   viewZ at the object's depth but vs != the object -> it WAS hit and the
    //     surface resolved to something else; the defect is in resolution.
    //   surfIdx = SURFACE_INDEX_INVALID -> no surface at all on that pixel.
    RTX_OPTION("rtx", Vector4, mtnRadianceRegion, Vector4(0.0f, 0.0f, 1.0f, 0.6f),
               "Normalized screen rectangle (minXFrac, minYFrac, maxXFrac, maxYFrac) "
               "the [MtnRadiance] raw per-pixel GBuffer probe samples on a 24x12 grid. "
               "Default (0,0,1,0.6) reproduces the original hardcoded upper-60% band. "
               "Aim it at an object to read what the primary ray actually got there. "
               "A degenerate rect (max <= min) falls back to the full screen.");

    RTX_OPTION("rtx", float, mtnRadianceMinAbsViewZ, 1.0e4f,
               "[MtnRadiance] only logs a sampled pixel when |viewZ| exceeds this. "
               "Default 1e4 keeps the original behaviour (distant backdrop hits only, "
               "skipping near props and no-hit pixels). SET THIS TO 0 to log every "
               "sampled pixel raw — required when the question is whether a ray hit "
               "the object at all, since a miss or a near hit is exactly what the "
               "default gate discards.");

    RTX_OPTION("rtx", bool, coverageSyncBeforeReadback, false,
               "Insert vkDeviceWaitIdle before the per-frame surface "
               "coverage buffer readback so the CPU sees this frame's "
               "GPU writes instead of a prior frame's. Diagnostic; "
               "stalls every frame.");

    // NV-DXVK [Coverage PickRegion fast path]: when true, the per-frame
    // coverage dump emits ONLY the PickRegion + PickRegion2 lines and skips
    // every other [Coverage] section (the 71-region histogram, PureRed,
    // DebugViewScan, etc.). The full dump prints ~80 Logger::info lines per
    // frame, which at gameplay drops FPS to ~0.16 — so "log every frame"
    // really means one pick line every ~5 seconds. Pick-only cuts the
    // per-frame logging to two lines, raising FPS enough to actually watch
    // the center VS change as you sweep the camera. PickRegion still runs
    // every frame; this only trims the noise around it.
    RTX_OPTION("rtx", bool, coveragePickRegionOnly, false,
               "Per-frame coverage dump logs only PickRegion / PickRegion2 "
               "and skips all other [Coverage] sections, so the dump costs "
               "two lines/frame instead of ~80. Use when you want a fast, "
               "high-frame-rate stream of the pick-region VS while moving "
               "the camera. Needs rtx.logSurfaceCoverage + a debug view.");

    RTX_OPTION("rtx", FusedWorldViewMode, fusedWorldViewMode, FusedWorldViewMode::None, "Set if game uses a fused World-View transform matrix.");

    RTX_OPTION("rtx", bool, useBuffersDirectly, true, "When enabled Remix will use the incoming vertex buffers directly where possible instead of copying data.");
    RTX_OPTION("rtx", bool, alwaysCopyDecalGeometries, true, "When set to True tells the geometry processor to always copy decals geometry. This is an optimization flag to experiment with when rtx.useBuffersDirectly is True.");

    // Vertex capture / pipeline classification
    RTX_OPTION("rtx", bool, orthographicIsUI, true, "When enabled, draw calls that are orthographic will be considered as UI.");
    RTX_OPTION("rtx", bool, allowCubemaps, false, "When enabled, cubemaps from the game are processed through Remix, but they may not render correctly.");
    RTX_OPTION("rtx", bool, useVertexCapture, true, "When enabled, injects code into the original vertex shader to capture final shaded vertex positions. Useful for games using simple vertex shaders that still set legacy transform matrices.");
    RTX_OPTION("rtx", bool, useInputAssemblerNormals, true,
               "When enabled, vertex normals from the D3D11 input layout (NORMAL semantic) are extracted and used in raytracing.\n"
               "Disable if a game provides garbage normals in its vertex buffers — Remix will regenerate normals instead.");
    RTX_OPTION("rtx", bool, useCBufferWorldMatrices, true,
               "When enabled, world/model matrices are extracted from D3D11 constant buffers for per-object transforms.\n"
               "Disable if a game's CB layout causes incorrect world matrix detection (objects appear at wrong positions).");
    RTX_OPTION("rtx", bool, ignoreSecondaryTextures, false,
               "When enabled, only the highest-scoring texture SRV is used per draw call; secondary textures (lightmaps, detail maps) are discarded.\n"
               "Enable if a game's secondary textures (e.g. lightmaps) interfere with material identification.");
    RTX_OPTION("rtx", uint32_t, maxInstanceSubmissions, 512,
               "Maximum number of instances submitted per instanced draw call.\n"
               "Each D3D11 instance with a per-instance world matrix becomes a separate Remix draw. Lower values improve performance;\n"
               "raise if distant instanced geometry (foliage, props) is visibly cut off. Set to 1 to disable instancing expansion entirely.");

    RTX_OPTION("rtx.terrain", bool, terrainAsDecalsEnabledIfNoBaker, false, "If terrain baker is disabled, attempt to blend with the decals.");
    RTX_OPTION("rtx.terrain", bool, terrainAsDecalsAllowOverModulate, false, "Set to true, if it's known that terrain layers with ModulateX2 / ModulateX4 flags do not contain a lighting info, but ModulateX2 / ModulateX4 are used only to blend layers.");

    RTX_OPTION_ARGS("rtx.userBrightness", int, userBrightness, 50, "How bright the final image should be. [0,100] range.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.userBrightnessEVRange", float, userBrightnessEVRange, 3.f, "The exposure value (EV) range for \'rtx.userBrightness\' slider, i.e. how much of EV there is between 0 and 100 slider values.");

    struct Eye {
      RTX_OPTION("rtx.eye", bool, showOptions, false, "Show eye options in the developer menu.");
      RTX_OPTION("rtx.eye", bool, enable, false, "Enable shader code for eye drawing (eyeball normals, iris blending).");
      RTX_OPTION("rtx.eye", bool, assumeViewTexgenModeAsEye, true, 
                 "Used to detect eyes and its vectors, by assuming that a draw call with camera-space texcoord generation and a specific texture transform is an eye draw call.");
      RTX_OPTION("rtx.eye", float, eyeballSphereOffset, 0.18F,
                 "How much to offset a sphere origin when calculating the eye normals on Whites. "
                 "The larger the value, the more pronounced the ambient shadowing is on an eyeball, to better ground the eyes on a face.");
      RTX_OPTION("rtx.eye", float, corneaSphereOffset, 0.1F,
                 "How much to offset a sphere origin when calculating the eye normals on Cornea. "
                 "Positive values make the eye cornea appear more spherical. Negative values - more flat.");
      RTX_OPTION("rtx.eye", float, eyeWhitesAlbedoScale, 0.5F, "Brightness multiplier for the eye whites.");
      RTX_OPTION("rtx.eye", float, irisRadius, 0.165F,
                 "Size of an iris in the iris texture. "
                 "If the iris texture is sampled outside of this radius, it's assumed that that area is a transition to the eye whites, "
                 "so iris depth gradually goes to 0.");
      RTX_OPTION("rtx.eye", float, irisDepth, 0.06F,
                 "How deep should the iris (colored part of an eye) be placed behind the cornea (eye lens). "
                 "The larger the value the more distortion there is, because of the lens.");
    };

    // Automation Options
    struct Automation {
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableBlockingDialogBoxes, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_BLOCKING_DIALOG_BOXES",
                          "Disables various blocking blocking dialog boxes (such as popup windows) requiring user interaction when set to true, otherwise uses default behavior when set to false.\n"
                          "This option is typically meant for automation-driven execution of Remix where such dialog boxes if present may cause the application to hang due to blocking waiting for user input.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableDisplayMemoryStatistics, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_DISPLAY_MEMORY_STATISTICS",
                          "Disables display of memory statistics in the Remix window.\n"
                          "This option is typically meant for automation of tests for which we don't want non-deterministic runtime memory statistics to be shown in GUI that is included as part of test image output.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableUpdateUpscaleFromDlssPreset, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_UPDATE_UPSCALER_FROM_DLSS_PRESET",
                          "Disables updating upscaler from DLSS preset.\n"
                          "This option is typically meant for automation of tests for which we don't want upscaler to be updated based on a DLSS preset.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, suppressAssetLoadingErrors, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_SUPPRESS_ASSET_LOADING_ERRORS",
                          "Suppresses asset loading errors by turning them into warnings.\n"
                          "This option is typically meant for automation of tests for which acceptable asset loading issues are known.");
    };

  public:
    LegacyMaterialDefaults legacyMaterial;
    OpaqueMaterialOptions opaqueMaterialOptions;
    TranslucentMaterialOptions translucentMaterialOptions;
    ViewDistanceOptions viewDistanceOptions;

    static const HashRule& geometryHashGenerationRule() {
      return s_geometryHashGenerationRule;
    }
    static const HashRule& geometryAssetHashRule() {
      return s_geometryAssetHashRule;
    }

  private:
    static HashRule s_geometryHashGenerationRule;
    static HashRule s_geometryAssetHashRule;

    RTX_OPTION("rtx", Vector3, effectLightColor, Vector3(1, 1, 1), "Colour of the effect light, if not using plasma ball mode.  Effect lights can be attached to materials from the remix runtime menu, using the `Add Light to Texture` texture tag in game setup.");
    RTX_OPTION("rtx", float, effectLightIntensity, 1.f, "The intensity of the effect light.  Effect lights can be attached to materials from the remix runtime menu, using the `Add Light to Texture` texture tag in game setup.");
    RTX_OPTION("rtx", float, effectLightRadius, 5.f, "The sphere radius of the effect light.  Effect lights can be attached to materials from the remix runtime menu, using the `Add Light to Texture` texture tag in game setup.");
    RTX_OPTION("rtx", bool, effectLightPlasmaBall, false, "Use plasma ball mode, in this mode the effect light color is ignored.  Effect lights can be attached to materials from the remix runtime menu, using the `Add Light to Texture` texture tag in game setup.");

    RTX_OPTION("rtx", bool, useObsoleteHashOnTextureUpload, false,
               "Whether or not to use slower XXH64 hash on texture upload.\n"
               "New projects should not enable this option as this solely exists for compatibility with older hashing schemes.");

    RTX_OPTION("rtx", uint32_t, applicationId, 102100511, "Used to uniquely identify the application to DLSS. Generally should not be changed without good reason.");

    static RtxOptions* s_instance;

  public:

    RtxOptions(const RtxOptions&) = delete;
    RtxOptions(RtxOptions&&) = delete;
    RtxOptions& operator=(const RtxOptions&) = delete;
    RtxOptions& operator=(RtxOptions&&) = delete;

  private:

    RtxOptions(bool invokeCallbacks = true) {
      Logger::info(str::format("Initializing RtxOptions... (invokeCallbacks=", invokeCallbacks, ")"));

      // Latch the per-DLL "suppress callbacks" flag for the lifetime of this DLL instance. Once
      // set true, every subsequent applyPendingValues call in this DLL's address space — including
      // the per-frame one in RtxContext — will skip onChange callbacks. Without this, after Create()
      // returns the per-frame loop would re-fire the same storm Create() avoided.
      RtxOptionManager::setSuppressCallbacksForThisDll(!invokeCallbacks);

      // Optionally write documentation (captures code-defined defaults from RTX_OPTION macros)
      if (env::getEnvVar("DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD") == "1") {
        RtxOptionManager::writeMarkdownDocumentation("RtxOptions.md");
      }

      // Initialize all system layers (creates layers from config files)
      RtxOptionLayer::initializeSystemLayers();

      // Need to set this to true after conf files are parsed, but before any options are accessed.
      RtxOptionImpl::setInitialized(true);

      // Replacement options
      if (env::getEnvVar("DXVK_DISABLE_ASSET_REPLACEMENT") == "1") {
        enableReplacementAssets.setDeferred(false);
        enableReplacementLights.setDeferred(false);
        enableReplacementMeshes.setDeferred(false);
        enableReplacementMaterials.setDeferred(false);
      }

      // Mark all options with onChange callbacks as dirty. This ensures that options with derived
      // settings (like NRC's qualityPreset which sets trainingMaxPathBounces) are properly
      // initialized even when using default values.
      // Skipped in satellite DLLs: those callbacks assume they execute inside the rendering DLL's
      // DxvkInstance context and have side effects that corrupt per-frame state if fired against
      // a half-built environment. The satellite DLL only needs m_resolvedValue populated so that
      // direct option reads (e.g. d3d11.dll's UI-texture-hash lookups, debugViewIdx) return the
      // user's conf-set value instead of the compile-time default.
      if (invokeCallbacks) {
        RtxOptionManager::markOptionsWithCallbacksDirty();
      }

      // Ensure all of the above values are promoted before the first frame starts.
      // DxvkDevice hasn't been created yet, so pass nullptr here.
      // forceOnChange is only meaningful when callbacks are being invoked.
      RtxOptionManager::applyPendingValues(nullptr,
                                           /* forceOnChange */ invokeCallbacks,
                                           /* invokeCallbacks */ invokeCallbacks);

      // NV-DXVK: hand rtx.logDenyTags to the Logger. This is the earliest point
      // the conf value is resolved, and it must run BEFORE logEffectiveValues()
      // below — that call is itself several hundred log lines, and a run that
      // set logDenyTags to silence something would otherwise still pay for all
      // of them. The Logger lives in util/ and cannot see RtxOptions, so the
      // value is pushed down rather than pulled.
      Logger::setDenyTags(logDenyTags());

      // Log effective RtxOption values after all initialization and migrations are complete
      RtxOptionManager::logEffectiveValues();
    }

  public:
    static void updateUpscalerFromDlssPreset();
    static void updateUpscalerFromNisPreset();
    static void updateUpscalerFromTaauPreset();
    static void updateUpscalerFromXeSSPreset();
    static void updatePresetFromUpscaler();
    static NV_GPU_ARCHITECTURE_ID getNvidiaArch();
    static NV_GPU_ARCH_IMPLEMENTATION_ID getNvidiaChipId();
    static void updateGraphicsPresets(DxvkDevice* device);
    static void updateLightingSetting();
    static void updatePathTracerPreset(PathTracerPreset preset);
    static void updateRaytraceModePresets(const uint32_t vendorID, const VkDriverId driverID);

    static void resetUpscaler();

    // invokeCallbacks=false suppresses every onChange callback during Create(). Use this when the
    // calling DLL does not own the rendering DxvkInstance — d3d11.dll, for example, has its own
    // per-DLL inline-static RtxOption<T> storage that needs to be populated from rtx.conf so reads
    // return the user's value, but the callbacks (which expect to mutate the rendering device's
    // state) must not fire from this context. Default true preserves dxvk.dll behavior.
    static void Create(bool invokeCallbacks = true) {
      if (s_instance == nullptr) {
        s_instance = new RtxOptions(invokeCallbacks);
      }
      // If called a second time, nothing to do - singleton already exists with all options initialized.
    }

    // Returns the merged configuration for DXKV Options. This includes all config files loaded from DXVK_CONFIG_FILE and DXVK_RTX_CONFIG_FILE.
    // This is available after Create() is called and can be used for DxvkOptions, etc.
    static const Config& getMergedConfig() {
      return RtxOptionLayer::getMergedConfig();
    }

    static bool getRayPortalTextureIndex(const XXH64_hash_t& h, std::size_t& index) {
      const auto findResult = std::find(rayPortalModelTextureHashes().begin(), rayPortalModelTextureHashes().end(), h);

      if (findResult == rayPortalModelTextureHashes().end()) {
        return false;
      }

      index = std::distance(rayPortalModelTextureHashes().begin(), findResult);

      return true;
    }

    static bool useReSTIRGI() {
      return integrateIndirectMode() == IntegrateIndirectMode::ReSTIRGI;
    }

    static bool shouldConvertToLight(const XXH64_hash_t& h) {
      return lightConverter().find(h) != lightConverter().end();
    }


    static bool isRayReconstructionEnabled() {
      return upscalerType() == UpscalerType::DLSS && enableRayReconstruction();
    }

    static bool showRayReconstructionOption() {
      return RtxOptions::upscalerType() == UpscalerType::DLSS;
    }

    static bool isDLSSEnabled() {
      // Note: DLSS-RR performs both denoising and upscaling so DLSS-SR should be disabled when it is enabled.
      return upscalerType() == UpscalerType::DLSS && !enableRayReconstruction();
    }

    static bool isDLSSOrRayReconstructionEnabled() {
      return upscalerType() == UpscalerType::DLSS;
    }
    static bool isNISEnabled() { return upscalerType() == UpscalerType::NIS; }
    static bool isTAAEnabled() { return upscalerType() == UpscalerType::TAAU; }
    static bool isXeSSEnabled() { return upscalerType() == UpscalerType::XeSS; }
    
    static float getUniqueObjectDistanceSqr() { return uniqueObjectDistance() * uniqueObjectDistance(); }
    static uint32_t getNumFramesToPutLightsToSleep() { return numFramesToKeepLights() /2; }
    static float getMeterToWorldUnitScale() { return 100.f * sceneScale(); } // RTX Remix world unit is in 1cm 

    // Returns shared enablement composed of multiple enablement inputs
    static bool needsMeshBoundingBox();
    
    static bool isShaderExecutionReorderingInPathtracerGbufferEnabled() { return enableShaderExecutionReorderingInPathtracerGbuffer() && enableShaderExecutionReordering; }
    static bool isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled() { return enableShaderExecutionReorderingInPathtracerIntegrateIndirect() && enableShaderExecutionReordering; }

    // Developer Options
    static bool areValidationLayersEnabled() {
      // Honor the flag in BOTH Debug and Release. Previously Debug builds
      // force-enabled the Khronos validation layer unconditionally, which
      // triples process memory and tracks all kMaxBindlessResources (64K)
      // descriptor slots — its mimalloc arena allocator then crashes during
      // the bindless descriptor-set allocation (VkLayer_khronos_validation.dll,
      // write to a poison page). Default off; opt in with
      // DXVK_ENABLE_VALIDATION_LAYERS=1 (or rtx.enableValidationLayers) when
      // you actually need Vulkan API validation.
      return enableValidationLayers();
    }

    static bool getIsOpacityMicromapSupported() { return s_instance && s_instance->opacityMicromap.isSupported; }
    static void setIsOpacityMicromapSupported(bool enabled) { if (s_instance) s_instance->opacityMicromap.isSupported = enabled; }
    static bool getEnableOpacityMicromap() { return s_instance && s_instance->opacityMicromap.enable() && s_instance->opacityMicromap.isSupported; }

    static bool getEnableAnyReplacements() { return enableReplacementAssets() && (enableReplacementLights() || enableReplacementMeshes() || enableReplacementMaterials()); }
    static bool getEnableReplacementLights() { return enableReplacementAssets() && enableReplacementLights(); }
    static bool getEnableReplacementMeshes() { return enableReplacementAssets() && enableReplacementMeshes(); }
    static bool getEnableReplacementMaterials() { return enableReplacementAssets() && enableReplacementMaterials(); }

    // Capture Options
    //   General
    static bool getCaptureInstances() {
      if (captureNoInstance() != captureNoInstance.getDefaultValue()) {
        Logger::warn("rtx.captureNoInstance has been deprecated, but will still be respected for the time being, unless rtx.captureInstances is set.");
        if (captureInstances() != captureInstances.getDefaultValue()) {
          return captureInstances();
        }
        return !captureNoInstance();
      }
      return captureInstances();
    }
    
    static std::string getCurrentDirectory();

    static float calcUserEVBias() {
      return (float(RtxOptions::userBrightness() - 50) / 100.f)
        * RtxOptions::userBrightnessEVRange();
    }
  };
}
