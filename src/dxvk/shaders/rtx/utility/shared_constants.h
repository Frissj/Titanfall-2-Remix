/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

// contains constants shared between shader and host code

static const uint8_t surfaceMaterialTypeOpaque = uint8_t(0u);
static const uint8_t surfaceMaterialTypeTranslucent = uint8_t(1u);
static const uint8_t surfaceMaterialTypeRayPortal = uint8_t(2u);
static const uint8_t surfaceMaterialTypeMask = uint8_t(0x3u);

#define COMMON_MATERIAL_FLAG_TYPE_MASK surfaceMaterialTypeMask
#define COMMON_MATERIAL_FLAG_TYPE_OFFSET(X) (2 + X)

// NOTE: Each material memory structure contains a set of flags.  The first 2 bits in that flag identify the material type (opaque, etc).
//       We must ensure all other material flags are written to byte addresses after these first two bits.  Use the COMMON_MATERIAL_FLAG_TYPE_OFFSET(x) 
//       macro, and ensure there is enough storage in the flags to represent desired bits accordingly.

// maximum value for thin film thickness in nanometers
#define OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS (1500.0f)
// bits for flags field in OpaqueSurfaceMaterial
#define OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_IS_THIN_FILM_THICKNESS (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(1))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(2))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(3))
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(4))
// NV-DXVK: Source/TF2 Phong-style PS uses `c_useAlphaModulateEmissive` to
// gate `emissive *= albedo.a` per pixel. When set, the slang emissive
// integration multiplies emissiveRadiance by opacity. Sourced from the PS
// RDEF + per-draw CBufUberStatic read in d3d11_rtx.cpp::FillMaterialData.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_MODULATE_EMISSIVE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(5))
// NV-DXVK: emissiveColorConstant is a per-draw TINT (c_emissiveTint) that
// must MULTIPLY the per-pixel emissiveTexture sample, matching the original
// PS's `emissive = sample * c_emissiveTint`. The default slang behaviour
// without this flag is to OVERWRITE the constant with the sample (legacy
// where emissiveColorConstant is a no-texture fallback, default (0,0,0)).
// Set only by LegacyMaterialData::as<OpaqueMaterialData>() when source
// UsesEmission populates the constant from the PS CB.
#define OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TINT_FROM_CONSTANT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(6))
// NV-DXVK: TF2 viewmodel "screen-space scrolling overlay" emissive pattern.
// The original PS samples the emissive texture at a UV derived from
// SV_Position (the pixel's screen coord), not from mesh UV — producing a
// camera-aligned scrolling effect that's then masked through a t17
// `emissiveMultiplyTexture` sampled at the mesh UV. Confirmed via fxc
// /dumpbin on PS 0x7836c1dd4d5c885f / 0xea2b85b0f20fddf3 — see lines
// 280-296 of ps_7836c1dd4d5c885f.asm. When this flag is set, the slang
// material reads a per-material 2x2 + translate from the screen-space
// emissive override block and samples the emissive texture at
//   screenUv = pixelCoord × M + T
// instead of at surfaceInteraction.textureCoordinates. Phase-1 detection
// only — GPU plumbing of M/T/maskTextureIndex lands in a follow-up patch.
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(7))
// NV-DXVK: the albedo/opacity texture is bound with an sRGB-format image
// view, so the hardware sampler already converts sRGB->linear on read.
// Remix's surface-material code otherwise assumes textures are raw
// (non-sRGB-format) and applies gammaToLinear() to the albedo in software;
// for an sRGB-view texture that double-decodes and crushes the albedo dark
// (e.g. TF2's BC7_SRGB sky cloud textures rendered near-black). When this
// flag is set the shader skips the software gammaToLinear() for albedo.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_SRGB (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(8))
// NV-DXVK: TF2 3D-skybox cloud billboards. Their final colour is synthesized
// by the game pixel shader's atmosphere/fog math (lerp toward a sun-tinted
// fog colour) — the bound texture is only a near-black coverage/detail map,
// so path-tracing the texture RGB as albedo renders the soft cloud edges
// black ("black smoke"). When this flag is set the opaque surface material
// shader reconstructs the fog blend from the captured CBufCommonPerCamera
// fog params in the global cb (cb.tf2Fog*). Set in rtx_scene_manager.cpp
// createSurfaceMaterial for materials whose PS reads c_fogColorFactor AND
// use a kAlpha (premultiplied OVER) blend — the cloud-billboard signature.
#define OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(9))
// NV-DXVK: source D3D11 draw uses *premultiplied alpha blending*
// (rt0.BlendEnable=1, SrcBlend=ONE, DestBlend=INV_SRC_ALPHA, BlendOp=ADD).
// For premult sources the texture's .rgb is already authored as
// (color * alpha), so multiplying by opacity again inside
// albedoToAdjustedAlbedo / calcBaseReflectivity at the encode site is a
// double-multiply that visibly darkens soft-translucent edges (TF2 cloud
// billboards rendering as a noisy speckled dark blob). When this flag is
// set the slang shader passes opacity=1 to those two helpers so the
// encoded albedo stays as the premultiplied color; the per-surface
// `opacity` field is still preserved for path-tracer transmission.
// Inferred purely from D3D blend state in d3d11_rtx.cpp::FillMaterialData
// — no per-VS hash allowlist.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(10))
// NV-DXVK: for content whose final colour is *already baked* into the
// texture by the original game pixel shader (atmospheric blend, fog
// tint, sun coloration all applied at authoring time), the path tracer
// must NOT try to light it — there's no light source to integrate
// against. Concrete case: TF2's 3D-skybox painted hemispheric dome
// sits 6.75M units from any in-scene light, so `radiance = albedo ×
// light_contribution = albedo × 0 = 0` and the sky renders pure black
// despite the GBuffer holding the correct sampled colour. When this
// flag is set the opaque-surface-material shader routes the sampled
// albedo into emissiveRadiance and zeros albedo + baseReflectivity,
// so the path tracer outputs the baked colour directly. Plumbed from
// DrawCallTransforms::isSubViewSkybox (an AABB-diagonal-derived
// structural classifier, no hash list) via rtx_scene_manager.cpp.
#define OPAQUE_SURFACE_MATERIAL_FLAG_BAKED_ALBEDO_AS_EMISSIVE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(11))

// NV-DXVK: the metallic/spec slot is bound with an sRGB-format view, so the
// hardware sampler applied an sRGB->linear decode on the way in.
//
// Unlike albedo, that decode is never wanted here. The texture routed to this
// slot (TF2's `specTexture`) holds a Schlick F0 - a physically LINEAR
// reflectance - so the stored 8-bit number is already the value the BRDF wants.
// Decoding it darkens F0, and because the metallic reprojection is
// `saturate((luminance(F0) - 0.04) / 0.96)` the darkened value falls under the
// 0.04 dielectric floor and clamps to exactly zero. Measured over all 108 spec
// maps in TF2 (9.5M BC1 endpoints): 79% of texels land under the threshold once
// decoded, versus 15% read linearly - so four fifths of every surface collapses
// to a flat 0.04 base reflectivity.
//
// That the data is linear is not inferred from the format. Read as sRGB, the
// MEDIAN texel in the game decodes to F0 = 0.0144, well below the ~0.02 floor of
// any real material - not something an artist can author.
//
// The shader undoes the sampler's decode when this is set. Re-encoding rather
// than rebinding a UNORM view because these are game-created D3D11 images
// wrapped by DXVK; a second view of a different format needs the image to carry
// VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, which nothing here guarantees.
#define OPAQUE_SURFACE_MATERIAL_FLAG_METALLIC_IS_SRGB (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(12))

// NV-DXVK: the emissive slot is bound with an sRGB-format view.
//
// The opposite case to metallic: emissive radiance IS display-referred colour,
// so the sampler's sRGB->linear decode is correct and wanted. The bug is that
// the shader then ran its own gammaToLinear() on top - resolving the upstream
// "Todo: Disable this for when a sRGB texture is the source of the emissive
// color" - double-decoding and crushing emissive dark.
//
// So this flag SKIPS the software decode, exactly like ALBEDO_IS_SRGB, rather
// than undoing anything.
#define OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_IS_SRGB (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(13))


#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_HAS_HEIGHT_TEXTURE (1 << 0)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_USE_THIN_FILM_LAYER (1 << 1)
// flags overlap with type field when in gbuffer, which occupies last 2 bits.
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_MASK 0x3F


// Note: Bits for flags field in TranslucentSurfaceMaterial and TranslucentSurfaceMaterialInteraction
// If set, then the texture bound to transmittanceOrDiffuseTextureIndex is an albedo map for the diffuse layer
#define TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_DIFFUSE_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))

#endif // ifndef SHARED_CONSTANTS_H
