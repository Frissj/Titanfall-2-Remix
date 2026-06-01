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

// These are set indices - not bindings
#define BINDING_SET_BINDLESS_RAW_BUFFER          1
#define BINDING_SET_BINDLESS_TEXTURE2D           2
#define BINDING_SET_BINDLESS_SAMPLER             3


#define BINDING_ACCELERATION_STRUCTURE           0
#define BINDING_ACCELERATION_STRUCTURE_PREVIOUS  1
#define BINDING_ACCELERATION_STRUCTURE_UNORDERED 2
#define BINDING_ACCELERATION_STRUCTURE_SSS       3
#define BINDING_SURFACE_DATA_BUFFER              4
#define BINDING_SURFACE_MAPPING_BUFFER           5
#define BINDING_SURFACE_MATERIAL_DATA_BUFFER     6
#define BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER 7
#define BINDING_VOLUME_MATERIAL_DATA_BUFFER      8
#define BINDING_LIGHT_DATA_BUFFER                9
#define BINDING_PREVIOUS_LIGHT_DATA_BUFFER       10
#define BINDING_LIGHT_MAPPING                    11
#define BINDING_BILLBOARDS_BUFFER                12
#define BINDING_BLUE_NOISE_TEXTURE               13
#define BINDING_BINDLESS_INDICES_BUFFER          14
#define BINDING_CONSTANTS                        15
#define BINDING_DEBUG_VIEW_TEXTURE               16
#define BINDING_GPU_PRINT_BUFFER                 17
#define BINDING_VALUE_NOISE_SAMPLER              18
#define BINDING_SAMPLER_READBACK_BUFFER          19

#define COMMON_MAX_BINDING                       BINDING_SAMPLER_READBACK_BUFFER

// NV-DXVK: per-pixel scene dump (one-shot, ImGui-triggered). Slot 200 is
// well above every pass-specific binding (max observed ~190 across all
// passes) so it doesn't clash with hardcoded per-pass slots that start at
// 20 (e.g. RTXDI_COMPUTE_GRADIENTS_BINDING_RTXDI_RESERVOIR=20). Declared
// in common_bindings.slangh under `#ifdef RAY_TRACING_PRIMARY_RAY` so only
// primary-ray pipelines (gbuffer raygen / closesthit) emit the binding,
// keeping non-primary shaders' descriptor layouts untouched.
#define BINDING_SCENE_DUMP_BUFFER                200

// Atmosphere LUTs use high binding slots to avoid conflicts with pass-specific bindings.
// Shifted to 201-203 because BINDING_SCENE_DUMP_BUFFER already occupies 200.
#define BINDING_ATMOSPHERE_TRANSMITTANCE_LUT     201
#define BINDING_ATMOSPHERE_MULTISCATTERING_LUT   202
#define BINDING_ATMOSPHERE_SKY_VIEW_LUT          203
// NV-DXVK [AerialPerspective]: 3D LUT (32 x 32 x 32) holding the
// in-scattered radiance + transmittance for a ray of length D in
// direction view-relative-to-sun. Sampled per-shading-point in the
// path tracer to apply atmospheric haze on geometry, decoupled from
// the visible-sky source. Works in both PhysicalAtmosphere and Hybrid
// sky modes.
#define BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT 204

// NV-DXVK [Coverage]: per-pass surface-coverage histogram. Two regions
// (region 0 = geometry-resolver primary surface, region 1 = integrate-pass
// surface), each kCoverageSurfaceSlots uints, atomically incremented once
// per pixel that resolves to a given surfaceIndex. Slot 205 keeps it clear
// of the pass-specific bindings (which start at 20); ungated in the slang
// declaration like the atmosphere LUTs above so both the gbuffer and the
// integrate pipelines can write it.
#define BINDING_SURFACE_COVERAGE_BUFFER          205

// Per-region slot count for the coverage histogram. The buffer holds
// 17 * COVERAGE_SURFACE_SLOTS uints (regions: 0=EncodedNonzero,
// 1=RawNonzero, 2=OpaquePrimary, 3=TranslucentPrimary,
// 4=HighSurfaceIndexBySite, 5=AlbedoDrift, 6=DriftStageGamma,
// 7=DriftStageScaleBias, 8=DriftStageMetallic, 9=DriftStageAdjusted,
// 10=MetallicHigh, 11=OpacityLow, 12=IsMatteHits, 13=IsTf2SkyboxFogHits,
// 14=MetallicLoaded, 15=FlagPremultSet, 16=DecalPrimaryHit). 262144
// comfortably covers any TF2 scene's surface count; the shaders bounds-
// check surfaceIndex against it before the atomic add.
// Region 16 is incremented when a primary-ray traversal hits a surface
// where surfaceIsDecal(surface)==true with frontHit + opacity > 0.001
// (the same gate the resolver uses to actually register the decal). The
// CPU readback maps the per-surfaceIndex counts back to VS hashes, so we
// can see exactly which draws are entering the decal classification path
// (and whether sub-view reproject geometry is wrongly flagged).
#define COVERAGE_SURFACE_SLOTS                   262144
#define COVERAGE_NUM_REGIONS                     17u
// NV-DXVK [Coverage color]: 3 extra regions accumulate the per-surface SUM of
// quantized raw diffuse albedo (each channel saturated to [0,1] then *255),
// keyed to the same surfaceIndex + same gate as region 1 (RawNonzero). The
// CPU readback divides by the region-1 pixel count to get the average raw
// albedo per VS — used to match an on-screen colour (e.g. a bright-red
// corruption blot) back to the exact shader. These are NOT pixel-count
// regions, so the main readback loop must skip them.
#define COVERAGE_RAWALBEDO_R_REGION              17u
#define COVERAGE_RAWALBEDO_G_REGION              18u
#define COVERAGE_RAWALBEDO_B_REGION              19u
// NV-DXVK [Coverage red]: per-surface count of pixels whose RAW diffuse albedo
// is red-dominant (r high, r >> g and r >> b). The per-VS average dilutes a
// localized bright-red blot to near-black, so a dedicated red-pixel COUNT is
// what actually pins which shader(s) draw the red corruption.
#define COVERAGE_REDPIXEL_REGION                 20u
// NV-DXVK [Coverage red worldpos]: world-space AABB of red-dominant pixels,
// stored at slot 0 of each region as biased InterlockedMax (decode: max =
// val-BIAS, min = BIAS-val). World position is captured at render time so it
// sidesteps the cross-frame surfaceIndex→instance mapping entirely; matching
// the AABB to the o2wT/position fields in the CPU instance logs (FloorTrace,
// MtnDedup, SpawnGeomDiag) pins the physical object drawing the corruption.
#define COVERAGE_REDPOS_BIAS                     1000000.0f
#define COVERAGE_REDPOS_MAXX_REGION              21u
#define COVERAGE_REDPOS_MINX_REGION              22u
#define COVERAGE_REDPOS_MAXY_REGION              23u
#define COVERAGE_REDPOS_MINY_REGION              24u
#define COVERAGE_REDPOS_MAXZ_REGION              25u
#define COVERAGE_REDPOS_MINZ_REGION              26u
// NV-DXVK [Coverage red screenbox]: screen-space bbox of red pixels (slot 0 of
// each region, biased InterlockedMax). Settles whether the red is a COMPACT box
// (small w/h ⇒ one sub-mesh / localized GBuffer region) or SCATTERED speckle
// (w/h spanning the object ⇒ sporadic stale-surface reads). BIAS=4096 > any
// resolution so min-via-max stays positive under the 0-init memset.
#define COVERAGE_REDSCR_BIAS                     4096u
#define COVERAGE_REDSCR_MAXX_REGION              27u
#define COVERAGE_REDSCR_MINX_REGION              28u
#define COVERAGE_REDSCR_MAXY_REGION              29u
#define COVERAGE_REDSCR_MINY_REGION              30u
// NV-DXVK [Coverage red hitdist]: hit-distance (camera→surface) stats of red
// pixels at slot 0. Distinguishes "correct-depth hit, wrong albedo" (red pixels
// cluster at the ship's depth with a normal spread ⇒ fix is albedo/material
// resolution) from "anomalous-depth wrong hit" (red pixels span wildly varying
// depths ⇒ BVH/ray precision picking the wrong surface). min=BIAS-MIN_REGION,
// max=MAX_REGION, avg=SUM_REGION/redCount.
#define COVERAGE_REDDIST_BIAS                    1000000.0f
#define COVERAGE_REDDIST_MAX_REGION              31u
#define COVERAGE_REDDIST_MIN_REGION              32u
#define COVERAGE_REDDIST_SUM_REGION              33u
// NV-DXVK [Coverage red sample/mip]: discriminator probe for the "one ship
// sub-mesh resolves to red albedo at a correct hit" bug. The existing
// RawAlbedo regions (17-19) capture covRawAlbedo = albedo AFTER the
// AO/detail/cloud modulation block, so they can't tell whether the red is
// already in the bare texture sample or was manufactured by modulation.
// These regions, written under the SAME red-pixel gate as region 20, settle
// mip-vs-UV-vs-content:
//   34-36 = SUM of the BARE pre-modulation albedoOpacitySample.rgb (*255).
//           If avg(34-36)/redCount ≈ (255,0,0) the texture sample itself is
//           red ⇒ wrong UV / wrong mip / red texel (NOT modulation). If it's
//           the normal brown but the post-mod RawAlbedo (17-19) is red, a
//           modulation stage (detail) is the culprit.
//   37-39 = texture-gradient magnitude stats (max/min/sum) of red pixels,
//           g = max(len(textureGradientX), len(textureGradientY)) scaled by
//           COVERAGE_REDGRAD_SCALE. A blown-up (huge) or collapsed (~0)
//           gradient vs neighbouring non-red pixels = bad mip selection
//           (same ray-cone gradient pipeline as the resolved braille bug).
//           avg = (SUM/SCALE)/redCount; min = (BIAS - MIN_REGION)/SCALE.
//   40    = InterlockedOr mask of (1u << min(debugPathCode,31)) over red
//           pixels. Bit positions map to surface.h debugPathCode values:
//           bit0=perspective-ok, bit1=behind-near, bit2=subpixel-det,
//           bit3=interpInvW-degenerate, bit4=cap-fired(>64 cone-iso
//           fallback), bit5=NaN/Inf, bit10=1D-degenerate-UV-fallback,
//           bit31=clamped/unset. If red pixels light bit4/bit5/bit3/bit1 the
//           gradient pipeline mis-fired ⇒ fix in surface_interaction.slangh;
//           if only bit0, gradients are healthy ⇒ look at UV / texcoord-gen.
#define COVERAGE_REDSAMPLE_R_REGION              34u
#define COVERAGE_REDSAMPLE_G_REGION              35u
#define COVERAGE_REDSAMPLE_B_REGION              36u
#define COVERAGE_REDGRAD_SCALE                   1000.0f
#define COVERAGE_REDGRAD_BIAS                    10000000.0f
#define COVERAGE_REDGRAD_MAX_REGION              37u
#define COVERAGE_REDGRAD_MIN_REGION              38u
#define COVERAGE_REDGRAD_SUM_REGION              39u
#define COVERAGE_REDPATH_MASK_REGION             40u
// NV-DXVK [Coverage red detail]: the bare albedo sample is brown (RedBareSample
// ≈ (75,41,14)) yet covRawAlbedo (post-modulation) is red — and of the three
// modulation stages only the detail overlay (albedo *= detailSample.rgb*2) is
// per-channel (AO/cloud are grayscale .rrr and cannot change hue). These
// regions confirm the detail stage IS the source and capture what to fix:
//   41-43 = SUM of detailSample.rgb (*255) over red pixels that had
//           detailLoaded. avg = /region-47-count. ≈(>128,low,low) ⇒ the detail
//           texel is red and MOD2X amplifies brown→red.
//   44    = OR-mask of which modulation stages were active on red pixels:
//           bit0=ambientOcclusionLoaded, bit1=detailLoaded, bit2=cloudMaskLoaded.
//   45/46 = detailTextureIndex max / min (min via DETAILIDX_BIAS - idx). A
//           garbage/stale index range during the engine-hook warmup ⇒ the
//           detail slot is mis-bound to a red texture; a stable sane index ⇒
//           the bound detail texture itself (or its blend mode) is the issue.
//   47    = count of red pixels with detailLoaded (denominator for 41-43).
#define COVERAGE_REDDETAIL_R_REGION              41u
#define COVERAGE_REDDETAIL_G_REGION              42u
#define COVERAGE_REDDETAIL_B_REGION              43u
#define COVERAGE_REDMOD_MASK_REGION              44u
#define COVERAGE_DETAILIDX_BIAS                  1048576u
#define COVERAGE_REDDETAILIDX_MAX_REGION         45u
#define COVERAGE_REDDETAILIDX_MIN_REGION         46u
#define COVERAGE_REDDETAIL_COUNT_REGION          47u
// NV-DXVK [Coverage NaN]: opaque-hit pixels whose RAW diffuse albedo is NaN or
// Inf are invisible to every region above — saturate(NaN)->0 fails both the
// region-1 dot>0.001 gate and the red-dominant gate, so they never increment a
// counter — yet they are EXACTLY what the debug-view Inf/NaN visualizer paints
// red (NaN) / blue (Inf). With the engine-hook main camera + transforms both
// proven clean (sanitize.NaN=0, [EngineCam] finite), the red plane's NaN must
// enter at the per-surface albedo level; this is the only probe that can see
// it. Region 48 = per-surfaceIndex count of NaN/Inf-albedo primary hits (CPU
// readback groups by VS like RawAlbedoColor). 49-52 = screen-space bbox of
// those pixels (biased InterlockedMax, reuses COVERAGE_REDSCR_BIAS): a compact
// box ⇒ one sub-mesh; a frame-spanning box ⇒ a fullscreen plane. Region 53 =
// count of NaN/Inf-albedo hits whose surfaceIndex was OUT of slot range (so the
// per-surface map missed them) — separates "known surface, NaN albedo" from
// "the index itself is bad".
#define COVERAGE_NAN_COUNT_REGION                48u
#define COVERAGE_NANSCR_MAXX_REGION              49u
#define COVERAGE_NANSCR_MINX_REGION              50u
#define COVERAGE_NANSCR_MAXY_REGION              51u
#define COVERAGE_NANSCR_MINY_REGION              52u
#define COVERAGE_NAN_OOR_COUNT_REGION            53u
// NV-DXVK [Coverage DebugViewScan]: the surface-side probes above sample the
// albedo INSIDE the opaque material shader (covRawAlbedo). Proven insufficient:
// disabling the detail overlay collapsed the surface-side red count but left the
// on-screen red plane unchanged — so the displayed pixels are NOT produced by
// the opaque-closesthit albedo path (miss/sky/secondary surface, or NaN). This
// region scans the ACTUAL displayed DEBUG_VIEW_RAW_ALBEDO buffer in the debug
// postprocess (one thread per screen pixel) — ground truth for what's on screen.
// Single region, indexed by slot (not per-surface):
//   slot 0 = red-dominant pixel count   slot 1 = NaN pixel count
//   slot 2 = Inf pixel count            slot 3 = total pixels scanned
//   slots 4-7  = red screen bbox  (maxX, BIAS-minX, maxY, BIAS-minY) biased InterlockedMax
//   slots 8-11 = NaN screen bbox  (same encoding)
#define COVERAGE_DVSCAN_REGION                   54u
#define COVERAGE_DVSCAN_SLOT_RED                 0u
#define COVERAGE_DVSCAN_SLOT_NAN                 1u
#define COVERAGE_DVSCAN_SLOT_INF                 2u
#define COVERAGE_DVSCAN_SLOT_TOTAL               3u
#define COVERAGE_DVSCAN_SLOT_RED_MAXX            4u
#define COVERAGE_DVSCAN_SLOT_RED_MINX            5u
#define COVERAGE_DVSCAN_SLOT_RED_MAXY            6u
#define COVERAGE_DVSCAN_SLOT_RED_MINY            7u
#define COVERAGE_DVSCAN_SLOT_NAN_MAXX            8u
#define COVERAGE_DVSCAN_SLOT_NAN_MINX            9u
#define COVERAGE_DVSCAN_SLOT_NAN_MAXY            10u
#define COVERAGE_DVSCAN_SLOT_NAN_MINY            11u
// Red count in the postprocess INPUT (pre-colorization) vs the OUTPUT counter
// above (post-colorization). If INPUT red ~= OUTPUT red, the red is already in
// the debug-view buffer (a non-opaque/miss write or stale data). If INPUT red
// ~0 but OUTPUT red large, the postprocess colorization fabricates the red.
#define COVERAGE_DVSCAN_SLOT_INPUTRED            12u
// Gate-free per-pixel DISPLAYED color dump on a fixed grid. NO threshold, NO
// average: each grid point stores the exact saturate(displayed.rgb)*255 of that
// one pixel, so the readback prints the literal value at points across the
// screen (incl. the red region). Slots 16.. = GRID_COLS*GRID_ROWS*3 entries,
// laid out idx=(row*COLS+col), channel at GRID_BASE + idx*3 + {0,1,2}.
#define COVERAGE_DVSCAN_GRID_BASE                16u
#define COVERAGE_DVSCAN_GRID_COLS                32u
#define COVERAGE_DVSCAN_GRID_ROWS                18u
// NV-DXVK [Coverage DebugViewScan]: per-surfaceIndex count of DISPLAYED red
// pixels — co-sampled from SharedSurfaceIndex at red pixels in the debug
// postprocess. CPU readback groups by VS (like RawAlbedoColor), so the line
// with the most pixels names the VS actually drawing the on-screen red plane,
// independent of the opaque-path covRawAlbedo (which has been disagreeing with
// the displayed value). surfaceIndex == SURFACE_INDEX_INVALID range / miss is
// dropped (only valid in-range indices recorded).
#define COVERAGE_DVRED_SURFACE_REGION            55u
// NV-DXVK [Coverage PureRed]: VALID-STAGE attribution of the pure-red plane.
// Recorded in the opaque material shader at the RAW_ALBEDO write, where the
// surfaceIndex is guaranteed valid (unlike the aliased SharedSurfaceIndex the
// postprocess co-samples, which may be stale → VS=0). Gated on pure-red raw
// albedo (r>0.5, g<0.05, b<0.05 — the (0.8,0,0) seen on screen). Region 56 =
// per-surfaceIndex count (CPU groups by VS + maps to material/colorTexture).
// Region 57 slot 0 = total pure-red opaque hits, slot 1 = those WITH a loaded
// color texture, slot 2 = those WITHOUT (constant fallback). If most pure-red
// hits have NO loaded texture, the red is an unresolved-texture fallback; if
// region 56 is ~empty while the screen is red, the red pixels are NOT opaque
// hits at all (miss/sky path), and this layer is the wrong one.
#define COVERAGE_PURERED_SURFACE_REGION          56u
#define COVERAGE_PURERED_SUMMARY_REGION          57u
#define COVERAGE_PURERED_SLOT_TOTAL              0u
#define COVERAGE_PURERED_SLOT_TEXLOADED          1u
#define COVERAGE_PURERED_SLOT_TEXMISSING         2u
// Probe AT the RAW_ALBEDO store site (line ~1195), UNGATED by surfaceIndex, so
// it fires for opaque red hits even with an out-of-range index that the
// <SLOTS-gated PureRed above skips. slot 3 = count; slots 4/5 = inCovSurfaceIndex
// max / (UINT_MAX-min) so we can read the index RANGE of the red surface. If
// slot 3 == 0, the red is NOT written by the opaque store at all (truly non-opaque).
#define COVERAGE_PURERED_SLOT_STORETOTAL         3u
#define COVERAGE_PURERED_SLOT_STOREIDXMAX        4u
#define COVERAGE_PURERED_SLOT_STOREIDXMIN        5u
// NV-DXVK [Coverage OOBWhy]: WHY a primary-ray surfaceIndex is out of range.
// surfaceIndex = (customIndex & CUSTOM_INDEX_SURFACE_MASK) + geometryIndex, so
// the cause is either the instance's customIndex BASE or the geometryIndex
// offset. Captured at ray-decode time when surfaceIndex >= cb.surfaceCount.
// slot 0 = count; slots 1-5 = InterlockedMax of fullCustomIndex, base,
// geometryIndex, surfaceIndex, surfaceCount (one OOB example's magnitudes).
// If base >= surfaceCount → bad customIndex (TLAS instance points at a slot
// that doesn't exist). If base is fine but base+geometryIndex overflows →
// geometryIndex is too large (BLAS has more geometries than registered).
#define COVERAGE_OOBWHY_REGION                   58u
#define COVERAGE_OOBWHY_SLOT_COUNT               0u
#define COVERAGE_OOBWHY_SLOT_CUSTOMINDEX         1u
#define COVERAGE_OOBWHY_SLOT_BASE                2u
#define COVERAGE_OOBWHY_SLOT_GEOMETRYINDEX       3u
#define COVERAGE_OOBWHY_SLOT_SURFACEINDEX        4u
#define COVERAGE_OOBWHY_SLOT_SURFACECOUNT        5u
#define COVERAGE_TOTAL_REGIONS                   59u

#define COMMON_NUM_BINDINGS                      (COMMON_MAX_BINDING + 1)

// Note: Used to represent a non-existent buffer
#define BINDING_INDEX_INVALID uint16_t(0xFFFF)

// Sentinel for an invalid surface index.  Equals the 21-bit maximum (SURFACE_INDEX_MAX_VALUE
// from instance_definitions.h) so that it fits inside the packed RayInteraction._surfaceAndFlags
// field.  The surfaceMapping buffer returns int32_t(-1) for unmapped surfaces; the 21-bit
// property setter truncates 0xFFFFFFFF to 0x1FFFFF automatically.
// This reserves the highest representable surface index as "invalid", reducing the usable
// range by one (max usable index = SURFACE_INDEX_MAX_VALUE - 1 = 2,097,150).
#define SURFACE_INDEX_INVALID 0x001FFFFFu

#define SAMPLER_FEEDBACK_INVALID           uint16_t(0xFFFF)
#define SAMPLER_FEEDBACK_MAX_TEXTURE_COUNT uint16_t(0xFFFF)

// Note: Light array may only be up to a size of 2^16-1, allowing the last index to be
// used for an invalid index similar to the max binding index for materials.
#define LIGHT_INDEX_INVALID (0xFFFF)

#ifdef __cplusplus

#define COMMON_RAYTRACING_BINDINGS \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE)            \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_UNORDERED)  \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_PREVIOUS)   \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_SSS)        \
  STRUCTURED_BUFFER(BINDING_SURFACE_DATA_BUFFER)                    \
  STRUCTURED_BUFFER(BINDING_SURFACE_MAPPING_BUFFER)                 \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_DATA_BUFFER)           \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER)       \
  STRUCTURED_BUFFER(BINDING_VOLUME_MATERIAL_DATA_BUFFER)            \
  STRUCTURED_BUFFER(BINDING_LIGHT_DATA_BUFFER)                      \
  STRUCTURED_BUFFER(BINDING_PREVIOUS_LIGHT_DATA_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_MAPPING)                          \
  STRUCTURED_BUFFER(BINDING_BILLBOARDS_BUFFER)                      \
  TEXTURE2DARRAY(BINDING_BLUE_NOISE_TEXTURE)                        \
  CONSTANT_BUFFER(BINDING_CONSTANTS)                                \
  RW_TEXTURE2D(BINDING_DEBUG_VIEW_TEXTURE)                          \
  RW_STRUCTURED_BUFFER(BINDING_GPU_PRINT_BUFFER)                    \
  SAMPLER3D(BINDING_VALUE_NOISE_SAMPLER)                            \
  RW_STRUCTURED_BUFFER(BINDING_SAMPLER_READBACK_BUFFER)             \
  RW_STRUCTURED_BUFFER(BINDING_SCENE_DUMP_BUFFER)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_MULTISCATTERING_LUT)                 \
  TEXTURE2D(BINDING_ATMOSPHERE_SKY_VIEW_LUT)                        \
  TEXTURE3D(BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT)              \
  RW_STRUCTURED_BUFFER(BINDING_SURFACE_COVERAGE_BUFFER)
// NV-DXVK: SceneDumpBuffer is in COMMON_RAYTRACING_BINDINGS but uses slot
// 200 (out-of-the-way) so the C++ descriptor layout for every RT pipeline
// includes it; the slang declaration in common_bindings.slangh is gated on
// RAY_TRACING_PRIMARY_RAY so only primary shaders actually reference it.
// Non-primary pipelines bind the placeholder buffer but don't read/write
// it — the binding is silently unused.
#endif
