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

// NV-DXVK [Perf.ShaderClock]: dedicated cycle-counter accumulator. Deliberately
// its OWN buffer and its own binding rather than more SurfaceCoverageBuffer
// regions: that buffer's readback is gated on rtx.logSurfaceCoverage, which arms
// 52 atomics per primary hit and costs ~104 ms/frame. Timing counters that can
// only be read by paying 104 ms are useless, and worse, those atomics sit inside
// opaqueSurfaceMaterialInteractionCreate and would inflate one specific region,
// skewing the attribution rather than just the total.
//
// 64 uint slots, host-visible and coherent, read straight off mapPtr - no
// compaction pass, no barrier, no dependency on the coverage machinery.
#define BINDING_SHADER_CLOCK_BUFFER              206
#define SHADER_CLOCK_SLOT_COUNT                  64u

// Slot layout. Each region uses a (cycles, hits) pair so the log can print a mean
// per invocation rather than a total that moves with how many candidates ran.
// The regions form a FLAT, NON-OVERLAPPING partition of the per-candidate path,
// so share% sums to ~100% and no region is nested inside another. Nesting would
// double-count and make the shares uninterpretable - the parent's clock reads
// would also be charged the child's instrumentation. This is why there is no
// "whole surfaceInteractionCreate" region: it is partitioned into 3..7 instead.
#define SHADER_CLOCK_REGION_STRIDE               2u
#define SHADER_CLOCK_UNO_TRAVERSAL               0u   // rayQuery.Proceed()
#define SHADER_CLOCK_UNO_HITINFO                 1u   // candidate attrs + rayInteractionCreate
#define SHADER_CLOCK_UNO_SURFACE_LOAD            2u   // surfaces[] struct load
#define SHADER_CLOCK_SI_POSITIONS                3u   // indices, positions, transforms, posError
// normals + motion + texcoords + gradients + tangents + colour, as one region.
// The SICut ladder priced this whole tail at 0.19 ms of the 7.96, so subdividing
// it would only add clock reads to a region already known to be negligible.
#define SHADER_CLOCK_SI_REST                     4u
#define SHADER_CLOCK_UNO_CLIP                    5u   // view distance + surface clip
#define SHADER_CLOCK_UNO_MATERIAL                6u   // material interaction + approximations
#define SHADER_CLOCK_UNO_BLEND                   7u   // WBOIT / bin accumulation
#define SHADER_CLOCK_ORDERED_MATERIAL            8u   // resolveVertex, the ordered path
#define SHADER_CLOCK_REGION_COUNT                9u
// Cycle deltas are accumulated shifted right by this much to keep the uint32
// accumulator clear of its ceiling - see the note in shader_clock.slangh. The CPU
// readback multiplies the mean back up by (1 << SHADER_CLOCK_CYCLE_SHIFT).
#define SHADER_CLOCK_CYCLE_SHIFT                 4u

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
// NV-DXVK [Coverage PickRegion]: COLOR-INDEPENDENT per-surfaceIndex attribution
// inside a configurable screen rectangle (default bottom-right, where the TF2
// first-person weapon sits). Every other on-screen attribution probe (DVRED,
// PureRed) is gated on the pixel being RED, so an un-tinted object like the gun
// is never named. This probe bins SharedSurfaceIndex for EVERY pixel whose
// normalized coords fall inside cb.surfaceCoveragePickRegion (minXFrac, minYFrac,
// maxXFrac, maxYFrac), so the CPU readback ranks the VS hashes drawing inside the
// rect by pixel count regardless of color. Sweep / shrink the rectangle at
// runtime via rtx.debugView.surfaceCoveragePickRegion (no rebuild) to tighten
// onto a single object, then read its VS hash off [Coverage] PickRegionVS.
// Region 59 = per-surfaceIndex pixel count inside the rect.
#define COVERAGE_PICKREGION_SURFACE_REGION       59u
// Region 60 = rect summary (slot-indexed, not per-surface):
//   slot 0 = pixels scanned inside the rect
//   slot 1 = those with a valid in-range surfaceIndex (attributed to a VS)
//   slot 2 = those whose surfaceIndex was out of slot range (miss/sky/stale)
//   slots 4-7 = actual screen bbox covered inside the rect (maxX, BIAS-minX,
//               maxY, BIAS-minY) biased InterlockedMax (reuses COVERAGE_REDSCR_BIAS),
//               so the readback can confirm the rect landed where expected.
#define COVERAGE_PICKREGION_SUMMARY_REGION       60u
#define COVERAGE_PICKREGION_SLOT_TOTAL           0u
#define COVERAGE_PICKREGION_SLOT_VALID           1u
#define COVERAGE_PICKREGION_SLOT_INVALID         2u
#define COVERAGE_PICKREGION_SLOT_MAXX            4u
#define COVERAGE_PICKREGION_SLOT_MINX            5u
#define COVERAGE_PICKREGION_SLOT_MAXY            6u
#define COVERAGE_PICKREGION_SLOT_MINY            7u
// NV-DXVK [Coverage PickRegion]: summed displayed-pixel colour (value.rgb *
// 255) over the rect, plus a count, so the CPU readback can log the mean
// colour the user is aiming at. Reflects whatever debug view is active
// (raw albedo = grey; a lit view = the actual lit colour).
#define COVERAGE_PICKREGION_SLOT_COLR            8u
#define COVERAGE_PICKREGION_SLOT_COLG            9u
#define COVERAGE_PICKREGION_SLOT_COLB            10u
#define COVERAGE_PICKREGION_SLOT_COLN            11u
// NV-DXVK [Coverage PickRegion]: PER-surfaceIndex screen bbox of the pixels each
// surface covers inside the rect, so the CPU readback can report a separate
// on-screen box per VS (combining the boxes of all surfaces sharing that VS).
// A compact box in the bottom-right = a near-eye weapon; a frame-spanning box =
// world/sky. Biased InterlockedMax (decode: max = val, min = BIAS - val), reusing
// COVERAGE_REDSCR_BIAS. Indexed by surfaceIndex like region 59.
#define COVERAGE_PICKREGION_BOX_MAXX_REGION      61u
#define COVERAGE_PICKREGION_BOX_MINX_REGION      62u
#define COVERAGE_PICKREGION_BOX_MAXY_REGION      63u
#define COVERAGE_PICKREGION_BOX_MINY_REGION      64u
// NV-DXVK [Coverage PickRegion2]: a SECOND independent pick rect (same slot
// layout / decode as region 1) so two screen points can be attributed in one
// frame — e.g. the streak (y=0.25) vs screen center. Driven by
// rtx.surfaceCoveragePickRegion2 → cb.surfaceCoveragePickRegion2.
#define COVERAGE_PICKREGION2_SUMMARY_REGION      65u
#define COVERAGE_PICKREGION2_SURFACE_REGION      66u
#define COVERAGE_PICKREGION2_BOX_MAXX_REGION     67u
#define COVERAGE_PICKREGION2_BOX_MINX_REGION     68u
#define COVERAGE_PICKREGION2_BOX_MAXY_REGION     69u
#define COVERAGE_PICKREGION2_BOX_MINY_REGION     70u

// NV-DXVK [Perf.UnorderedSteps]: raw per-pixel census for the unordered resolve
// stage, which the compile-time stage ladder measured at 42.5 ms of the ~126 ms
// gb_primaryRays pass (2026-07-25, static camera).
//
// Only slot 0 of each region is used - these are three scalar accumulators, not
// per-surface histograms. They live in the coverage buffer purely to reuse its
// existing readback and throttling rather than adding another GPU->CPU path.
//
//   STEPS        - sum of rayQuery.Proceed() candidates stepped, all pixels
//   INTERACTIONS - sum of candidates that survived to be accepted as hits
//   PIXELS       - number of pixels that entered the unordered loop
//
// steps/pixels and interactions/pixels together decide whether the 42.5 ms is
// loop COUNT or per-candidate COST. kMaxUnorderedResolveSteps is 128 on primary
// rays, so a pass timer alone cannot distinguish 128 cheap candidates from 2
// expensive ones.
#define COVERAGE_UNORDERED_STEPS_REGION          71u
#define COVERAGE_UNORDERED_INTERACTIONS_REGION   72u
#define COVERAGE_UNORDERED_PIXELS_REGION         73u

// NV-DXVK [VsPix]: per-frame primary-hit pixel count per vertex shader, indexed
// by Surface::vsDebugId (slot 0 unused, ids 1..124 — see vsDebugIdToColor).
//
// This exists because DEBUG_VIEW_VERTEX_SHADER_ID alone cannot answer "which
// shader painted that one-frame flash": a flash lasting a single frame cannot
// be screenshotted, and the instance census only proves an instance EXISTED,
// not that it covered any pixels. This counts actual pixels, per frame, per
// shader. Only slots 0..124 are ever touched, so the CPU read scans 125 uints
// rather than a full 262144-slot region.
#define COVERAGE_VSPIX_REGION                    74u
#define COVERAGE_VSPIX_MAX_ID                    124u

// NV-DXVK [ResolveCensus]: per-surface census of the PRIMARY RESOLVE PATH,
// indexed by surfaceIndex exactly like regions 0..16.
//
// The question this exists to answer, from HANDOFF_PI_FLICKER_V4 §7: a set of
// vertex shaders ("Class B") is visibly rendering tens of thousands of pixels
// yet wins 0-7% of primary hits, i.e. it almost never writes
// m_sharedSurfaceIndex. [HitCensus] reads that buffer AFTER the fact, so it can
// only report the final winner - it cannot distinguish a surface the ray never
// intersected from one the ray hit and then resolved past. These four counters
// split that, per surface, per frame:
//
//   ORDSEEN   - the ordered resolver committed a hit on this surface and ran
//               resolveVertex on it (traversal reached it, material evaluated)
//   ORDFINAL  - this surface was the one written to SharedSurfaceIndex, i.e. it
//               became the resolved primary. Incremented at the write itself so
//               it can never drift from it.
//   UNOSEEN   - an unordered-resolve interaction accepted this surface. Unordered
//               geometry contributes attenuation/emissive only and is NEVER
//               eligible to be the primary surface, so this being the only
//               nonzero bucket is a by-construction explanation, not a defect.
//   CONTINUED - after resolveVertex, the resolver set continueResolving, i.e. it
//               deliberately passed THROUGH this surface and kept walking.
//
// Reading the result (no hypothesis is baked in - these four partition the space):
//   ordSeen==0 && unoSeen==0                  -> traversal never reaches it. The
//                                                defect is in the TLAS/BLAS/mask,
//                                                upstream of all shading.
//   unoSeen>0 && ordSeen==0                   -> routed to the unordered TLAS;
//                                                non-primary by design.
//   ordSeen>0 && ordFinal==0 && continued==ordSeen
//                                             -> the resolver passes through it
//                                                every time (alpha / opacity /
//                                                decal decision).
//   ordSeen>0 && ordFinal==0 && continued<ordSeen
//                                             -> it terminated resolution but
//                                                something else (PSR, secondary
//                                                surface selection) took the
//                                                primary write.
#define COVERAGE_RESOLVE_ORDSEEN_REGION          75u
#define COVERAGE_RESOLVE_ORDFINAL_REGION         76u
#define COVERAGE_RESOLVE_UNOSEEN_REGION          77u
#define COVERAGE_RESOLVE_CONTINUED_REGION        78u

// NV-DXVK [PIWrite]: what the point-instancer culling shader ACTUALLY wrote
// into the TLAS instance entry, per surfaceIndex, on the same frame — so it
// joins to [ResolveCensus] above by surfaceIndex with no inference in between.
//
// Why this exists. The census showed the flickering geometry is binary: either
// traversed and winning ~100% of its ordered hits, or NOTRAVERSED (ordSeen==0
// AND unoSeen==0) while still holding an unchanged number of slots in
// m_reorderedSurfaces. So the break is between "the CPU has the surface" and
// "the ray can reach it". Everything in that gap is written by
// point_instancer_culling.comp.slang, which is where these are taken.
//
// Deliberately RAW VALUES, not a cull classification. The obvious hypothesis
// (distance cull zeroing the mask) is already dead: rtx_point_instancer_system
// hard-forces cullingEnabled=false, so cullingRadius=FLT_MAX and
// fadeStartRadius=0, which makes `visible` true for every finite distance. The
// one surviving path to mask==0 in that shader is distSq being NaN, since
// `distSq <= radiusSq` is false for NaN even against FLT_MAX. Recording the
// values rather than a verdict means the log answers the question whichever of
// those it turns out to be — and stays readable if it is none of them.
//
// Each surfaceIndex is written by exactly ONE shader thread per frame
// (perInstanceSurfaceIndex = baseSurfaceIndex + instanceIdx is unique), so
// these are exact per-instance values, not aggregates. Atomics are used anyway
// so that overlapping batch ranges would corrupt nothing.
//
//   MASK   - mask + 1, so 0 unambiguously means "the culling shader never ran
//            for this slot this frame" and 1 means "it ran and wrote mask 0".
//            Distinguishing those two is the entire point.
//   FLAGS  - bit0 distSq non-finite, bit1 worldPos non-finite,
//            bit2 blasRef==0, bit3 visible (mask!=0)
//   DIST   - uint(distance) in world units, 0xFFFFFFFF if non-finite
//   BLASLO - low 32 bits of the BLAS device address written into the entry
#define COVERAGE_PIW_MASK_REGION                 79u
#define COVERAGE_PIW_FLAGS_REGION                80u
#define COVERAGE_PIW_DIST_REGION                 81u
#define COVERAGE_PIW_BLASLO_REGION               82u

#define COVERAGE_PIW_FLAG_DIST_NONFINITE         (1u << 0)
#define COVERAGE_PIW_FLAG_POS_NONFINITE          (1u << 1)
#define COVERAGE_PIW_FLAG_BLASREF_ZERO           (1u << 2)
#define COVERAGE_PIW_FLAG_VISIBLE                (1u << 3)

// NV-DXVK [TlasProbe]: interrogate the BUILT acceleration structure directly.
//
// This exists because the external route is unavailable: RenderDoc does not
// attach to Source games, and PIX cannot see this at all - with Remix attached
// the real rendering is Vulkan, so a D3D-side capture tool has nothing to
// inspect. But an external tool was only ever a means to one question, and we
// own the ray tracer, so we can ask the structure ourselves.
//
// Every measurement so far reads the data that FEEDS the TLAS build, and all of
// it is identical on frames where geometry renders and frames where it
// vanishes. This shoots an actual ray at the built TLAS, per surface, per
// frame, and reports what came back. It is the only probe here that reads the
// acceleration structure rather than the bookkeeping about it.
//
//   FLAGS  - bit0 probe ran for this surface
//            bit1 STRICT query hit something   (primary's mask + ray flags)
//            bit2 ANY query hit something      (mask 0xFF, no cull flags)
//            bit3 STRICT hit resolved to THIS surface
//            bit4 ANY hit resolved to THIS surface
//   HITSURF- (surfaceIndex of the ANY query's committed hit) + 1, 0 on miss.
//
// The ray is shot THROUGH the surface's own first triangle - standing off along
// the face normal and crossing back through it - not from the camera. That
// matters: a camera ray aimed at an object's origin can hit terrain instead of
// the mesh whether or not the mesh is in the structure, so a miss would have
// meant "the ray did not happen to cross it" rather than "it is absent". Firing
// through the face removes occlusion, distance and view direction as
// explanations in one step: if the instance is in the TLAS at the transform its
// own surface data describes, this ray must hit it.
//
// Reading it against a NOTRAVERSED census line for the same surface:
//   anySelf == 0           -> the instance is NOT in the built structure at the
//                             transform its surface data describes, even though
//                             every input to that build measured correct. Search
//                             moves to build flags, scratch memory and
//                             update-vs-rebuild.
//   anySelf == 1           -> it IS present and reachable. The geometry exists
//                             in the structure, so a primary-ray miss on the
//                             same frame is about the primary ray, not the TLAS.
//   anyHit 1, anySelf 0    -> something is at that location but it is not this
//                             surface; HITSURF names what is.
//   anySelf 1, strictSelf 0-> mask or backface rejection excludes it. Those are
//                             the only two differences between the queries.
// Note strictSelf is NOT "the primary ray would have found it" - the probe's
// direction is the face normal, not a view ray.
#define COVERAGE_TLASPROBE_FLAGS_REGION          83u
#define COVERAGE_TLASPROBE_HITSURF_REGION        84u

#define COVERAGE_TLASPROBE_FLAG_RAN              (1u << 0)
#define COVERAGE_TLASPROBE_FLAG_STRICT_HIT       (1u << 1)
#define COVERAGE_TLASPROBE_FLAG_ANY_HIT          (1u << 2)
#define COVERAGE_TLASPROBE_FLAG_STRICT_SELF      (1u << 3)
#define COVERAGE_TLASPROBE_FLAG_ANY_SELF         (1u << 4)
// Triangle centroid projects inside the view frustum this frame. The probe
// fires through the face and ignores the camera entirely, which is what makes
// it a clean presence test - but it therefore proves "in the TLAS", NOT
// "visible". Without this bit a NOTRAVERSED surface that the probe self-hits is
// ambiguous between "present and the primary ray wrongly missed it" and
// "present and simply off screen", and those are a defect and a non-defect.
#define COVERAGE_TLASPROBE_FLAG_ONSCREEN         (1u << 5)

// NV-DXVK [CamProbe]: does the camera agree with itself, and can a ray from the
// camera reach the geometry?
//
// prOnScreenSelf answered V5 §5: on NOTRAVERSED frames the geometry IS present,
// self-hittable and inside the frustum, and the user confirms it is not hiding
// behind anything. So the remaining gap is the ray itself - and there is a
// structural reason to suspect it, which nothing has tested yet.
//
// ONSCREEN is decided by ONE matrix:
//     cb.camera.worldToProjectionJittered
// The primary ray never touches that matrix. rayCreatePrimaryFromPixel builds
// from TWO DIFFERENT members of the same struct (camera.slangh):
//     origin    = transpose(camera.viewToWorld)[3].xyz
//     direction = mat3(camera.viewToWorld) * (camera.projectionToViewJittered * ndc)
// Those three fields are SUPPOSED to be inverses of each other. Nothing in this
// investigation has ever checked that they are. If the CPU fills them at
// different moments - and this tree has a known engine-hook camera phase
// problem - then "inside the frustum by matrix A" and "the ray built from basis
// B points at it" are independent claims, and only the second one decides
// whether the geometry renders. That is precisely the observed symptom: perfect
// bookkeeping, present in the TLAS, on screen, and never hit.
//
//   CAMDOTA/CAMDOTB - the round trip, in MILLIDEGREES of angular error:
//                     project the centroid with worldToProjectionJittered, turn
//                     the resulting NDC back into a screen UV, ask
//                     cameraScreenUVToDirection what direction that pixel's ray
//                     points, and compare against the true direction from the
//                     camera origin to that same centroid. A self-consistent
//                     camera gives 0 for every surface on every frame.
//                     Anything above ~40 (0.04 deg, a sub-pixel at 90 deg FOV
//                     over 1080 lines) is a real disagreement.
//                     TWO values because the NDC Y sign convention between
//                     worldToProjectionJittered and cameraScreenUVToNDC is an
//                     assumption, and an assumption baked into the instrument
//                     would manufacture exactly the failure it is looking for.
//                     A is the NDC as-is, B is the NDC with Y negated. The data
//                     picks the convention: on frames the geometry demonstrably
//                     renders, the correct one reads ~0. Read the OTHER one only
//                     after that calibration, and never average them.
//   CAMHITSURF      - fire a ray from the real camera origin straight at the
//                     centroid, with the primary pass's mask and ray flags, and
//                     record (committed surfaceIndex + 1), 0 on miss. Aimed
//                     along normalize(centroid - origin) rather than through a
//                     reconstructed pixel ON PURPOSE: it needs no NDC
//                     convention, so it is independent of CAMDOTA/B and tests
//                     camera POSITION, reachability and occlusion only.
//                     == self  -> a camera ray does reach it; the camera origin,
//                                 the TLAS and occlusion are all exonerated and
//                                 the defect is in which pixels get traced.
//                     == other -> that surface is the occluder, named.
//                     == 0     -> nothing along the whole segment, though the
//                                 face probe hits it. Would be a major finding.
//   CAMDIST         - camera-to-centroid distance in world units, rounded. Raw
//                     context for reading the above, and the cheapest way to see
//                     a distance correlation if one exists.
#define COVERAGE_TLASPROBE_CAMHITSURF_REGION     85u
#define COVERAGE_TLASPROBE_CAMDOTA_REGION        86u
#define COVERAGE_TLASPROBE_CAMDOTB_REGION        87u
#define COVERAGE_TLASPROBE_CAMDIST_REGION        88u

// Centroid was in front of the camera, so the projection round trip ran and
// CAMDOTA/CAMDOTB hold real values. Without this bit an unwritten slot reads 0
// millidegrees, which is indistinguishable from a perfect round trip - the
// instrument would report its own silence as a pass.
#define COVERAGE_TLASPROBE_FLAG_CAMPROJ_RAN      (1u << 6)
#define COVERAGE_TLASPROBE_FLAG_CAMRAY_HIT       (1u << 7)
#define COVERAGE_TLASPROBE_FLAG_CAMRAY_SELF      (1u << 8)
// The same camera ray fired again with no culling and mask 0xFF. Without this
// pair the strict ray is ambiguous in the one place it must not be: a
// camRaySelf of 0 would mean either "the camera cannot reach this geometry" or
// "it is back-facing to the camera and the primary flags cull it", and those
// point at completely different bugs. The face probe cannot settle it either -
// it fires along the face normal, so it can never be back-facing to itself.
//   anySelf 1, strictSelf 0 -> RAY_FLAG_CULL_BACK_FACING_TRIANGLES is removing
//                              the geometry from the primary ray. This fork has
//                              had a winding/mirroring bug of exactly that
//                              shape before, so it is a live candidate, not a
//                              formality.
//   anySelf 0, strictSelf 0 -> genuinely unreachable from the camera origin.
#define COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_HIT   (1u << 9)
#define COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_SELF  (1u << 10)

// NV-DXVK [AnyHitProbe] REMOVED 23:5x, before it ever ran. It was going to fire
// the ANY face ray without RAY_FLAG_FORCE_OPAQUE to test whether the any-hit
// shader rejects the vanishing geometry. Two reasons it was wrong:
//   1. THIS FORK HAS NO ANY-HIT SHADER. resolve.slangh:1620 handles opacity in
//      the resolve LOOP - hit, decide, re-trace from the hit point with a
//      reduced tMax - so there is no any-hit callback for FORCE_OPAQUE to skip.
//   2. An inline RayQuery cannot test it anyway. Without FORCE_OPAQUE, non-
//      opaque candidates suspend and require CommitNonOpaqueTriangleHit(); not
//      committing makes every alpha-tested surface read as a miss BY
//      CONSTRUCTION, and committing makes the query identical to FORCE_OPAQUE.
//      It could only ever have manufactured the answer it was looking for.
// Kept as a note because "rawHit is counted in the hit function, on every
// committed hit, BEFORE resolve can discard it" (geometry_resolver.slangh:1660)
// is the fact that makes rawHit=0 mean traversal truly never commits.
// ---------------------------------------------------------------------------
// (original comment follows, for the flag numbering history)
// NV-DXVK [AnyHitProbe]: the one thing every query above excludes by
// construction - the ANY-HIT SHADER.
//
// Both face queries and both camera queries pass RAY_FLAG_FORCE_OPAQUE, which
// skips any-hit entirely. The primary ray CANNOT do that: any-hit is where the
// alpha test runs, which is the whole reason rtx.primaryRayMaxInteractions and
// the resolver's interaction loop exist. So every "the geometry IS in the
// acceleration structure" result this probe has ever produced is true and stays
// true - and is completely silent on whether any-hit would ACCEPT a hit on that
// geometry.
//
// That silence is exactly the size of the remaining mystery. Measured on the
// 22:57 run, for the three largest shaders in the scene:
//     0x2af9b90d63850ec3  314 surfaces  rawHit on 0.0% of frames
//     0x29aa034553107f54  302 surfaces  rawHit on 0.0% of frames
//     0x29d5f7de0ba76c66  258 surfaces  rawHit on 4.3% of frames
// rawHit is counted the instant traversal commits, BEFORE resolve can discard
// anything (geometry_resolver.slangh:1660), so those hits are not being thrown
// away later - traversal never commits them at all. Meanwhile the FORCE_OPAQUE
// face probe self-hits the same surfaces on ~100% of frames, against the same
// TLAS, in the same command buffer, immediately before the gbuffer pass.
//
// If any-hit rejects every hit on those surfaces, traversal never commits ->
// rawHit=0 -> ordSeen=0 -> unoSeen=0 -> the ray sails through and the pixel
// shows whatever is behind. That is every measurement of this session at once,
// with nothing left over.
//
// This pair is the identical ANY ray - same origin, direction, tMax, mask 0xFF -
// with FORCE_OPAQUE REMOVED, so it is the only difference between the two.
//
//   anySelf 1, noFoSelf 0 -> ANY-HIT IS REJECTING THE GEOMETRY. The geometry is
//                            in the structure and the alpha/opacity path is
//                            throwing every hit away. Named exactly, no
//                            inference left.
//   anySelf 1, noFoSelf 1 -> any-hit accepts. The face ray reaches it with the
//                            real opacity path live, and the defect is in the
//                            CAMERA ray specifically, not in acceptance.
//   noFoHit 1, noFoSelf 0 -> any-hit accepts something else along that ray but
//                            not this surface.
//
// NOTE the conf records rtx.piForceOpaque=True as tested and negative. Treat
// that as unverified rather than as a refutation: it sets the force-opaque bit
// on PointInstancer instances only, and rtx.pointInstancer.enable = False was
// PROVEN this session not to reach its consumer (889 PI slots present with it
// set). PI option plumbing in this fork has a track record of not arriving.
// Bits 11 and 12 are FREE - the NOFO pair that was going to use them was
// removed before it ran, for the reasons above.

// NV-DXVK [CamTris]: does this surface own ANY pixel?
//
// The camera-ray probe above aims at ONE triangle - the surface's first - and
// that turned out to be the wrong question. Its own calibration proved it: on
// WINS_PRIMARY lines, where the geometry is winning millions of primary hits
// and is beyond argument visible, the camera ray reached that triangle only 16%
// of the time. A single arbitrary triangle is usually not the frontmost thing
// along its own view ray - it sits behind the rest of its own mesh - so
// "camRaySelf = 0" says almost nothing about whether the object is visible.
//
// What actually matters is whether ANY part of the surface is frontmost
// somewhere on screen, because that is what "the object is on screen" means and
// what ordSeen > 0 would require. So sample triangles ACROSS the surface and
// count how many are the committed first hit from the camera.
//
//   PRIMCOUNT     - triangles in this surface, written by the CPU before the
//                   probe dispatch. The shader cannot derive it: Surface exposes
//                   firstIndex but no count, and striding past the end without
//                   one would silently read the NEXT surface's indices and
//                   report its triangles as this one's. The count lives on the
//                   BLAS build range, which is CPU-side only.
//   CAMTRISTESTED - triangles sampled AND inside the frustum (<= the K cap,
//                   skipping degenerates and off-screen triangles). The
//                   denominator; without it "reached 0" cannot be told from
//                   "sampled nothing".
//                   The frustum test is per TRIANGLE and is not optional: a ray
//                   aimed from the camera at a triangle is trivially frontmost
//                   along its own ray even when the triangle is behind the
//                   viewer, where no primary ray is ever fired. Before this gate
//                   existed, 119 of 171 reported surfaces were simply off
//                   screen - the majority of the apparent defect population.
//   CAMTRISREACHED- of those, how many the camera ray's committed first hit
//                   resolved to THIS surface.
//
// Read reached/tested on a WINS_PRIMARY line FIRST. That is the calibration:
// geometry that is definitively rendering must score well above zero, and
// whatever it scores is the ceiling this instrument can report. Only then read
// it on NOTRAVERSED lines:
//   reached > 0, ordSeen == 0 -> the surface IS frontmost at one or more points
//                                on screen and the resolver still recorded no
//                                interaction with it anywhere. That is the
//                                defect, finally localised to the primary ray
//                                actually being traced (or not) at those pixels.
//   reached == 0              -> nothing of this surface is frontmost anywhere;
//                                it is genuinely behind other geometry, and
//                                NOTRAVERSED is correct behaviour for it.
#define COVERAGE_TLASPROBE_PRIMCOUNT_REGION      89u
#define COVERAGE_TLASPROBE_CAMTRISTESTED_REGION  90u
#define COVERAGE_TLASPROBE_CAMTRISREACHED_REGION 91u

// Triangles sampled per surface. Spread by stride across the whole index range
// rather than taken from the front, so a surface whose leading triangles happen
// to be interior or back-facing is not written off on the strength of them.
#define COVERAGE_TLASPROBE_TRI_SAMPLES           16u

// NV-DXVK [RawHit]: the layer between traversal and the census.
//
// ORDSEEN is written after rayInteractionHasHit(), which is AFTER resolveVertex
// has run on the hit. So a ray that traversal genuinely committed on a surface,
// but which resolveVertex then discarded, is indistinguishable from a ray that
// never touched the geometry at all - and every conclusion drawn from
// "ordSeen == 0" so far has quietly assumed the latter.
//
// This counter is written at the point the committed hit is constructed, before
// any resolve logic can reject it. Against the same VS on the same frame:
//   rawHit > 0, ordSeen == 0 -> traversal DID hit the geometry and resolveVertex
//                               dropped it. The defect is in resolve / surface
//                               data / material lookup, not in the ray.
//   rawHit == 0              -> traversal genuinely never hit it, even though
//                               the probe reaches it against the same TLAS with
//                               a SUBSET of the primary's ray mask. The defect
//                               is in the trace itself.
// Those are different bugs in different files, and nothing measured so far can
// tell them apart.
//
// Indexed with calculateSurfaceIndex(), i.e. (customIndex & MASK) +
// geometryIndex - the same arithmetic the resolver uses, so the counter lands
// in the slot the resolver would have attributed the hit to.
#define COVERAGE_RAWHIT_REGION                   92u

// surfaceMapping[surfaceIndex] as the shader sees it, recorded per surface by
// the probe. Read as a raw value, not a verdict: SURFACE_INDEX_INVALID is
// 0x1FFFFF (the 21-bit maximum), NOT 0xFFFF, and the buffer holds int32_t(-1)
// for unmapped surfaces which the 21-bit setter truncates to 0x1FFFFF. Logging
// the number means a wrong assumption about the sentinel cannot silently turn
// into a wrong conclusion.
#define COVERAGE_SURFMAP_REGION                  93u

// The instance mask actually built into the TLAS for this surface's instance,
// written CPU-side from RtInstance::censusMask(). A zero mask is traced by
// nothing, which would drop the geometry from the primary ray while leaving
// every other bookkeeping field correct - and the probe would still find it,
// because the probe's own queries use their own masks.
//
// CPU-owned like PRIMCOUNT, and excluded from the readback memset for the same
// reason: it is written immediately before the dispatch that consumes it.
#define COVERAGE_INSTMASK_REGION                 94u

// NV-DXVK [Occluder]: name what is in front.
//
// The flicker VS has every instance present in the TLAS, correctly masked,
// unmoved, and still self-hittable by the face probe - yet on dropout frames it
// is frontmost nowhere and traversal commits no hit on it. Something is in
// front of it, and ~5 other VS lose their hits on the same frames, so they are
// all behind the same thing.
//
// This records, for each sampled on-screen triangle the camera ray did NOT
// reach, the surface the ray committed to instead. The CPU resolves that index
// through the same reordered-instance table the census uses, so the occluder is
// reported as a vertex shader hash rather than a bare number.
//
// Stored as (surfaceIndex + 1); 0 means "no sample was occluded by anything".
// InterlockedMax rather than a first-writer-wins store: several triangles of a
// surface can be blocked by different things, and taking the max is at least a
// deterministic choice rather than a race between them. The per-surface raw
// lines carry the individual values when the distribution matters.
#define COVERAGE_TLASPROBE_OCCLUDER_REGION       95u

// NV-DXVK [NoCull]: is backface culling what removes the geometry?
//
// camTrisReached collapses from 5.11 to 0.18 on dropout frames while prAnySelf
// stays level - the face probe still finds the geometry in the TLAS, a camera
// ray no longer reaches it. The two probes differ in exactly one way that
// matters here: the face probe fires ALONG the triangle normal, so its ray can
// never be back-facing to its own target, while the camera ray carries
// RAY_FLAG_CULL_BACK_FACING_TRIANGLES like the primary pass. A winding or
// mirroring flip is therefore invisible to one probe and fatal to the other,
// and this fork has already shipped one bug of exactly that shape
// (drawClockwise != objectToWorldMirrored).
//
//   CAMTRISREACHED_NOCULL - the same camera ray, same origin, same direction,
//                           same tMax, with culling off and mask 0xFF. The ONLY
//                           difference from CAMTRISREACHED is the cull flag, so
//                           a gap between them is backface rejection and
//                           nothing else.
//   CAMTRISMISSED         - sampled on-screen triangles whose camera ray
//                           committed NO hit at all. Needed because
//                           reached/blocked do not partition the sample:
//                           occludedSurf counts SURFACES, not triangles, so
//                           "tested - reached" has never been attributable to
//                           blocked-vs-missed. With this, per triangle:
//                           tested = reached + blockedBySomethingElse + missed.
//
// Read on a dropout line: reachedNoCull > 0 while reached == 0 means the
// primary ray's cull flag is discarding geometry that is present and in front.
// Both zero, with missed high, means the ray passes through empty space where
// the surface buffer says geometry is - a different bug, in the transform.
#define COVERAGE_TLASPROBE_CAMTRISNOCULL_REGION  96u
#define COVERAGE_TLASPROBE_CAMTRISMISSED_REGION  97u

// NV-DXVK [Spike]: does the geometry itself deform?
//
// On the frames the flicker VS vanishes, two of the surfaces occluding it GAIN
// pixels (+70% and +31%) and cover more screen - while their instance count,
// TLAS membership, surface count, triangle count and transforms are all
// unchanged, and instMoveMax is 0. Geometry that expands on screen without its
// instance moving is not being placed differently; its VERTICES are moving.
// That is skinning, and this fork has a documented bug of exactly that shape:
// partially-posed bone palettes flinging weighted vertices hundreds of units.
//
// A vertex flung toward the camera occludes everything behind it for one frame,
// which is the only explanation so far that accounts for all of it at once -
// the target vanishing, ~5 VS dropping together, every instance-level field
// staying pristine, and the sporadic single-frame timing.
//
//   MAXRADIUS - the largest distance, in world units, from the surface's own
//               object origin to any sampled triangle centroid. This is a
//               deformation measure, NOT a position measure: it is taken
//               relative to the instance's own origin, so moving the object
//               cannot change it and only the vertices can. A mesh whose radius
//               jumps by hundreds of units between frames, with an unchanged
//               transform, has had its geometry deformed.
//
// Read it on the OCCLUDERS across the target's dropout vs normal frames, not on
// the target itself - the suspect here is the geometry doing the covering.
#define COVERAGE_TLASPROBE_MAXRADIUS_REGION      98u

// NV-DXVK [IdentProbe]: the IDENTITY of the surface entry the primary hit
// actually fetched, snapshotted at the [RawHit] site (geometryResolverVertex)
// under the DERIVED surface index. Value layout (single 32-bit store so the
// frame stamp and identity can never tear against the concurrent CPU read):
//   bit 31      = written flag
//   bits 27..30 = low nibble of the SUBMISSION frame id (from
//                 cb.enableResolveCensus, which carries frameId + 1)
//   bits 11..26 = Surface::hashPacked (16-bit fold of associatedGeometryHash)
//   bits 0..10  = Surface::vsDebugId
// The census readback lags GPU execution by frames-in-flight while the
// surface table reshuffles every frame, so the observation is only comparable
// against the expected identities of the frame that wrote it: the CPU keeps a
// ring of per-frame expected snapshots (rtx_context) and joins on the frame
// nibble. (v1 compared against live CPU state and read a meaningless constant
// 87% mismatch on healthy frames — readback lag, not a real desync.) A
// mismatch AFTER the frame join is direct proof the ray consumed a different
// frame's surface-table layout than the CPU submitted — the one-frame desync
// suspected of causing the single-frame group flicker on table-shift frames
// (dropouts correlate 30-40x with |dTotalSurf|, measured 2026-08-02).
#define COVERAGE_OBS_IDENT_REGION                99u

// NV-DXVK [ProbeAlign]: the frame the tlas_probe pass RAN for this slot,
// stamped by the probe shader itself so it travels with the probe's data
// through the readback lag. Same layout as OBS_IDENT's marker+nibble (bit 31
// = written, bits 27..30 = low nibble of frameId from cb.enableResolveCensus
// - 1); low bits unused. Written unconditionally for every slot the probe
// pass covers, including ones that early-out, so an unstamped slot (0) means
// the GPU never ran the probe for it this readback.
//
// Why it exists: every GPU-written TLASPROBE_*/SURFMAP field was being joined
// to a VS through the LIVE ordered-instance table at readback — the exact
// flaw [CensusAlign] fixed for the hit counts, one region over. Measured
// 2026-08-03 ([CamProbe] ringVs): the live owner and the GPU-frame owner
// agreed on 0 of 12677 lines; V10 §3.3's "prAnySelf=100%" was measured on
// other shaders' surfaces (predominantly 0x292b6ba0d1854f28's).
#define COVERAGE_TLASPROBE_FRAME_REGION          100u

// NV-DXVK [Centroid]: the world position the ONSCREEN test is actually run on,
// and the clip w it produced.
//
// WHY. The flicker VS 0x29d5f7de0ba76c66 renders on ONE frame, then is dark for
// 8-146 frames, repeatedly, and [FlickerTrack] shows the change is neither a
// SURFACE nor an AS event: it stays in the surface table and stays in the
// acceleration structure the whole time (prAnySelf 60-164 of 259 on the dark
// frames). What flips is prOnScreen, in perfect lockstep with ordSeen - 26
// frames both, 1201 frames neither, ZERO frames rendered while off screen.
//
// Nothing that should control that is moving: camPos is bit-identical across
// 1300 frames and instMoved/instMoveMax are 0 on every frame including the
// flash. But instMoved reads RtInstance::getWorldPosition(), i.e. the instance
// TRANSLATION, while the ONSCREEN test projects the triangle CENTROID through
// surface.objectToWorld. A rotation or scale change - or a sub-view
// reprojection landing on one frame in N - moves the centroid without moving
// the translation at all, and no field in this census could see it.
//
// So record the centroid itself. Float bit patterns via asuint, plain stores:
// one thread owns each surface slot (same ownership the FRAME region relies
// on), so no atomic is needed and a max over a bit pattern would be meaningless
// anyway. The census reads them back as floats and reports the per-VS centroid
// BBOX, which needs no cross-frame surface identity to compare - the slot a
// surface occupies is frame-local, so a per-slot diff would measure the table
// reshuffling rather than the geometry.
//
// CLIPW is the other half of the question and it is not optional. ONSCREEN is
// `w > 0 && all(abs(ndc) <= 1)`, so an off-screen verdict has two completely
// different causes: the geometry is BEHIND the camera (w <= 0) or it is in
// front and outside the frustum bounds. Those point at different bugs, and
// without w the census cannot tell them apart.
#define COVERAGE_TLASPROBE_CENTROIDX_REGION      101u
#define COVERAGE_TLASPROBE_CENTROIDY_REGION      102u
#define COVERAGE_TLASPROBE_CENTROIDZ_REGION      103u
#define COVERAGE_TLASPROBE_CLIPW_REGION          104u

#define COVERAGE_TOTAL_REGIONS                   105u

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
  RW_STRUCTURED_BUFFER(BINDING_SURFACE_COVERAGE_BUFFER)             \
  RW_STRUCTURED_BUFFER(BINDING_SHADER_CLOCK_BUFFER)
// NV-DXVK: SceneDumpBuffer is in COMMON_RAYTRACING_BINDINGS but uses slot
// 200 (out-of-the-way) so the C++ descriptor layout for every RT pipeline
// includes it; the slang declaration in common_bindings.slangh is gated on
// RAY_TRACING_PRIMARY_RAY so only primary shaders actually reference it.
// Non-primary pipelines bind the placeholder buffer but don't read/write
// it — the binding is silently unused.
#endif
