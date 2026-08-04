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
    RTX_OPTION_FLAG("rtx", bool, logMemoCeiling, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Measures the upper bound for SubmitDraw memoization: per frame, of the draws reaching commit, how many are identical to a draw from the PREVIOUS frame and could therefore be replayed from cache instead of re-running the full inject pipeline (ExtractTransforms recon, cull scan, material snapshot). Emits [MemoCeiling] with geomStable (VS + bound VB/IB identity matched last frame => geometry-dependent work is cacheable) and fullStable (that PLUS the bound VS transform-cbuffer bytes matched => the entire commit is replayable). The gap between them is the 'same object, camera moved' set. Use this number to decide whether building the memoization is worth it.");
    RTX_OPTION_FLAG("rtx", bool, logDupPass, false, RtxOptionFlags::NoSave, "DIAGNOSTIC (no behaviour change). Characterizes WITHIN-frame geometry re-injection: many draws reaching commit share the same geometry (VB/IB) as an earlier draw THIS frame (multi-pass). Emits [DupPass] per frame with depthOnly (color write mask == 0 — z-prepass / shadow, which RT arguably does not need) and, for the duplicates, a breakdown of what distinguishes them from the first sighting of that geometry: dupDepthOnly (color write off), dupDiffCam (different cameraType — sub-view like 3D-skybox / water reflection), dupDiffPs (different pixel shader — multi-material pass), dupSame (identical pass — truly redundant). Decides which redundant passes can be filtered before commit.");
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
    RTX_OPTION("rtx", bool, perfNonOpaqueCensus, true,
               "DIAGNOSTIC: log [Perf.NonOpaque] once per second, a per-branch "
               "census of how instances were classified opaque vs non-opaque, "
               "since only non-opaque hits drive the primary-ray resolve loop.");
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
    // These drive verified byte patches into client.dll — see cullOffUpdate() in
    // d3d11_rtx.cpp for the site table, the exact instructions, and the IDA
    // evidence for each. Every site is byte-verified before it is written and
    // restored exactly when its flag goes false, so each can be A/B'd live.
    //
    // Order to enable them in: frustum first (that is the one that makes
    // off-screen objects vanish from shadows and reflections), then distanceFade,
    // then the two default-off ones only if you still see popping — visibilityMask
    // and pvs both widen what the engine submits a lot, and pvs in particular
    // hands the path tracer the entire map.
    struct CullOff {
      friend class ImGUI;
      friend class RtxOptions;
      RTX_OPTION("rtx.cullOff", bool, enable, false,
                 "Master switch for the engine culling patches. Off = the game culls normally.");
      RTX_OPTION("rtx.cullOff", bool, frustum, true,
                 "Disable the per-renderable frustum cull in client.dll BuildRenderableRenderLists\n"
                 "(sub_1801A9C70 for the main view, the AABB test for shadow/sub-views). This is the\n"
                 "cull that removes off-screen shadow casters and reflection sources.");
      RTX_OPTION("rtx.cullOff", bool, distanceFade, true,
                 "Disable the distance-fade cull (sub_1801A90E0 / sub_1801A95B0): props stop fading\n"
                 "out and being dropped from the render list at range. Patches the reject branches,\n"
                 "not the bit-clear, so the render-list entry each renderable carries is still filled\n"
                 "in — clearing only the cull would leave those entries holding stack garbage.");
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
      RTX_OPTION("rtx.sceneCull", bool, enable, false,
                 "Master switch for Remix-side radius/frustum culling of ray-traced instances.\n"
                 "Culls by zeroing the instance mask in mergeInstancesIntoBlas, which keeps the BLAS\n"
                 "and geometry resident (no rebuild when the object returns) but removes the instance\n"
                 "from the TLAS. Intended to replace the engine culling disabled by rtx.cullOff.*.");
      RTX_OPTION("rtx.sceneCull", float, radius, 0.0f,
                 "Cull instances whose world-space bounding box is entirely farther than this many\n"
                 "world units from the main camera. 0 disables the radius cull. Measured from the\n"
                 "closest point of the box, so large objects survive until fully outside the sphere.");
      RTX_OPTION("rtx.sceneCull", bool, frustumEnable, false,
                 "Cull instances whose world-space bounding box lies entirely outside the main\n"
                 "camera frustum. Removes off-screen shadow casters and reflection/GI contributors —\n"
                 "widen frustumMargin, or prefer the radius cull, if that shows.");
      RTX_OPTION("rtx.sceneCull", float, frustumMargin, 0.25f,
                 "Fractional widening of the left/right/top/bottom frustum planes for the frustum\n"
                 "cull, e.g. 0.25 keeps anything within 125% of the screen extents. Costs a little\n"
                 "scope, buys back most edge-of-screen shadow and reflection contributors.");
      RTX_OPTION("rtx.sceneCull", bool, frustumCullBehindCamera, false,
                 "Also reject instances entirely behind the camera near plane. Off by default: geometry\n"
                 "behind the camera is exactly what mirrors and indirect lighting need.");
      RTX_OPTION("rtx.sceneCull", bool, logStats, false,
                 "Log one [SceneCull] line per second: instances tested, and how many each rule culled.");
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

    RTX_OPTION("rtx", bool, debugInstanceUploadProbe, true,
               "DIAGNOSTIC: verify the merged TLAS instance-buffer upload "
               "byte-for-byte against what the TLAS build consumes, in the "
               "build's own command stream. Logs [InstUpProbe] cmp=OK "
               "heartbeats and loud [InstUpProbe.bad] per-instance dumps on "
               "any divergence. Join mismatch frames against [ResolveCensus] "
               "ordSeen=0 dropout frames: mismatch on a dropout frame = the "
               "upload race IS the geometry flicker; OK on dropout frames "
               "exonerates the instance bytes (bug is build/driver side).");

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
    RTX_OPTION("rtx", bool, logTlasSet, true,
               "DIAGNOSTIC, cheap: logs [TlasSet], one line per frame per TLAS "
               "giving the order-independent signature of the instance SET fed to "
               "the acceleration-structure build, plus appeared/vanished counts "
               "diffed against the previous frame, and maskZero/blasNull for "
               "entries that are present but will be skipped by the tracer. "
               "[TlasSet.gone] names the BLAS and world position of what left "
               "(capped at 8 per frame). vanished>0 with a still camera is the "
               "geometry leaving the TLAS, caught at the last point before the "
               "build.");

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
