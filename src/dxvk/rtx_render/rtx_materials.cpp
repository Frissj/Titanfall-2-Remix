/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_materials.h"

#include "rtx_options.h"

namespace dxvk {

bool getEnableDiffuseLayerOverrideHack() {
  return TranslucentMaterialOptions::enableDiffuseLayerOverride();
}

float getEmissiveIntensity() {
  return RtxOptions::emissiveIntensity();
}

float getDisplacementFactor() {
  return RtxOptions::Displacement::displacementFactor();
}

float getDisplacementInFactor() {
  return RtxOptions::Displacement::displacementFactor() * RtxOptions::Displacement::displacementInFactor();
}

float getDisplacementOutFactor() {
  return RtxOptions::Displacement::displacementFactor() * RtxOptions::Displacement::displacementOutFactor();
}

dxvk::OpaqueMaterialData LegacyMaterialData::createDefault() {
  OpaqueMaterialData opaqueMat;
  opaqueMat.setAnisotropyConstant(LegacyMaterialDefaults::anisotropy());
  opaqueMat.setEmissiveIntensity(LegacyMaterialDefaults::emissiveIntensity());
  opaqueMat.setAlbedoConstant(LegacyMaterialDefaults::albedoConstant());
  opaqueMat.setOpacityConstant(LegacyMaterialDefaults::opacityConstant());
  opaqueMat.setRoughnessConstant(LegacyMaterialDefaults::roughnessConstant());
  opaqueMat.setMetallicConstant(LegacyMaterialDefaults::metallicConstant());
  opaqueMat.setEmissiveColorConstant(LegacyMaterialDefaults::emissiveColorConstant());
  opaqueMat.setEnableEmission(LegacyMaterialDefaults::enableEmissive());
  opaqueMat.setEnableThinFilm(LegacyMaterialDefaults::enableThinFilm());
  opaqueMat.setAlphaIsThinFilmThickness(LegacyMaterialDefaults::alphaIsThinFilmThickness());
  opaqueMat.setThinFilmThicknessConstant(LegacyMaterialDefaults::thinFilmThicknessConstant());
  return opaqueMat;
}

template<> OpaqueMaterialData LegacyMaterialData::as() const {
  // Legacy materials have parameters that can directly carry over onto the opaque material.
  const OpaqueMaterialData defaultLegacyOpaqueMaterial = createDefault();
  // Copy off the defaults, and make dynamic adjustments for the remaining params from this legacy material
  OpaqueMaterialData opaqueMat(defaultLegacyOpaqueMaterial);
  if (LegacyMaterialDefaults::useAlbedoTextureIfPresent()) {
    opaqueMat.setAlbedoOpacityTexture(getColorTexture());
  }
  if (getColorTexture2().isValid()) {
    opaqueMat.setSecondaryTexture(getColorTexture2());
  }
  // NV-DXVK: forward PBR maps discovered from the pixel-shader RDEF.
  // Populated by D3D11Rtx::FillMaterialData for games that name their SRVs
  // (e.g. Titanfall 2: albedoTexture/normalTexture/glossTexture/specTexture/
  // emissiveTexture at t0..t4). When empty, the opaque material falls back
  // to the rtx.legacyMaterial.* constants for that channel.
  if (normalTexture.isValid() && !normalTexture.isImageEmpty()) {
    opaqueMat.setNormalTexture(normalTexture);
  }
  if (roughnessTexture.isValid() && !roughnessTexture.isImageEmpty()) {
    opaqueMat.setRoughnessTexture(roughnessTexture);
  }
  if (metallicTexture.isValid() && !metallicTexture.isImageEmpty()) {
    opaqueMat.setMetallicTexture(metallicTexture);
  }
  // NV-DXVK: emission gated on the PS's own intent — the legacy classifier
  // forwards `emissiveTexture` (slot t4) for every Source PS that declares
  // it, but TF2 PSes only actually compute per-pixel emission when their
  // RDEF marks `CBufUberStatic.c_emissiveTint` as used (D3D_SVF_USED).
  // Reading that flag at FillMaterialData time and gating here means the
  // emissive map is wired into the BRDF only for materials whose original
  // shader was authored to be emissive — water/refract/decal layers that
  // happen to carry an emissive_texture binding stay non-emissive, killing
  // the "lighting bouncing off surfaces" bug that the blend-state-driven
  // promotion in rtx_instance_manager.cpp produced. Tint is forwarded
  // verbatim from the PS CB; alpha-modulate semantic is wired through to
  // the slang shader via setAlphaModulateEmissive (see flag bit
  // OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_MODULATE_EMISSIVE).
  if (sourceUsesEmission) {
    if (emissiveTexture.isValid() && !emissiveTexture.isImageEmpty()) {
      opaqueMat.setEmissiveColorTexture(emissiveTexture);
    }
    opaqueMat.setEmissiveColorConstant(sourceEmissiveTint);
    opaqueMat.setEnableEmission(true);
    // Tint already carries the strength via its magnitude, so leave the
    // intensity scalar at unit. Game's c_emissiveTint=(1,1,1,_) for "use
    // texture as-is", smaller magnitudes for dimmed effects.
    opaqueMat.setEmissiveIntensity(1.0f);
    opaqueMat.setAlphaModulateEmissive(sourceAlphaModulatesEmissive);
    // NV-DXVK: signal the slang shader that emissiveColorConstant carries
    // a per-draw tint that must MULTIPLY the per-pixel emissive texture
    // sample (vs. the legacy overwrite-as-fallback semantic). Required for
    // exact PS fidelity on materials whose c_emissiveTint != (1,1,1).
    opaqueMat.setEmissiveTintFromConstant(true);
  }

  // NV-DXVK: TF2 worldspace VGUI/HUD shader handling. The original PS
  // composes its final color from font glyph + atlas + per-style buffers
  // and writes it to SV_Target with no lighting — so the path tracer must
  // render the surface unlit (isMatte=true via the GPU surface flag set
  // from rtx_instance_manager.cpp) AND output the picked color texture as
  // emissive radiance so the pixel is visible at all. Without this UI
  // panels otherwise appear pitch-black inside dark rooms because the only
  // light source the BRDF sees is the world's (very dim, post-fix).
  if (sourceIsUnlitUI) {
    if (getColorTexture().isValid() && !getColorTexture().isImageEmpty()) {
      opaqueMat.setEmissiveColorTexture(getColorTexture());
    }
    opaqueMat.setEmissiveColorConstant(Vector3(1.f, 1.f, 1.f));
    opaqueMat.setEnableEmission(true);
    opaqueMat.setEmissiveIntensity(1.0f);
    // Tint = (1,1,1) and EmissiveTintFromConstant=true → slang multiplies
    // emissiveSample × (1,1,1) = emissiveSample (no tint shift), exactly
    // what the original PS produces.
    opaqueMat.setEmissiveTintFromConstant(true);
    // Routed to surface.isMatte at instance build time — see
    // rtx_instance_manager.cpp:appendInstance.
    opaqueMat.setIsUnlitOutput(true);
  }

  // NV-DXVK: TF2 3D-skybox cloud billboard — forward the fog-reconstruction
  // marker. d3d11_rtx.cpp::FillMaterialData set this when the PS reads
  // c_fogColorFactor and the draw is premultiplied-blended. The opaque
  // surface shader uses the resulting OPAQUE_SURFACE_MATERIAL_FLAG_TF2_
  // SKYBOX_FOG to rebuild the game's fog-blend synthesis from cb.tf2Fog*.
  if (sourceTf2FogCapable) {
    opaqueMat.setTf2SkyboxFog(true);
  }
  // NV-DXVK: premultiplied-alpha-blend source — forward the marker so the
  // GPU material gets OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED.
  // The slang shader then skips the opacity-multiply inside albedoToAdjusted-
  // Albedo / calcBaseReflectivity for these surfaces (their .rgb is already
  // premultiplied, multiplying again would double-darken soft edges).
  if (sourceAlbedoIsPremultiplied) {
    opaqueMat.setAlbedoIsPremultiplied(true);
  }
  if (ambientOcclusionTexture.isValid() && !ambientOcclusionTexture.isImageEmpty()) {
    opaqueMat.setAmbientOcclusionTexture(ambientOcclusionTexture);
  }
  if (lightmapTexture.isValid() && !lightmapTexture.isImageEmpty()) {
    opaqueMat.setLightmapTexture(lightmapTexture);
  }
  if (lightmap2Texture.isValid() && !lightmap2Texture.isImageEmpty()) {
    opaqueMat.setLightmap2Texture(lightmap2Texture);
  }
  if (detailTexture.isValid() && !detailTexture.isImageEmpty()) {
    opaqueMat.setDetailTexture(detailTexture);
  }
  if (cloudMaskTexture.isValid() && !cloudMaskTexture.isImageEmpty()) {
    opaqueMat.setCloudMaskTexture(cloudMaskTexture);
  }

  // NV-DXVK: forward the TF2 viewmodel "screen-space scrolling overlay"
  // emissive params to the slang material when FillMaterialData detected
  // the pattern. The slang shader uses these (gated by the
  // OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE flag bit) to
  // sample the emissive texture at SV_Position-derived UV instead of mesh
  // UV, and optionally multiplies by the mask texture sampled at mesh UV.
  // c_uv1RotScale and c_uv1Translate are the raw values from CBufUberStatic;
  // the per-frame engine animation scalar (c_rcpRenderTargetSize and the
  // per-frame translate scalar) get folded in on the slang side using its
  // existing camera-state binding.
  if (hasScreenSpaceEmissive) {
    opaqueMat.setHasScreenSpaceEmissive(true);
    opaqueMat.setScreenSpaceEmissiveMatRow0(screenSpaceEmissiveUv1RotScaleX);
    opaqueMat.setScreenSpaceEmissiveMatRow1(screenSpaceEmissiveUv1RotScaleY);
    opaqueMat.setScreenSpaceEmissiveTranslate(screenSpaceEmissiveUv1Translate);
    if (screenSpaceEmissiveMaskTexture.isValid() && !screenSpaceEmissiveMaskTexture.isImageEmpty()) {
      opaqueMat.setScreenSpaceEmissiveMaskTexture(screenSpaceEmissiveMaskTexture);
    }
    // NV-DXVK: one-shot per-LegacyMaterialData log to confirm the
    // screen-space emissive flag actually flows from FillMaterialData
    // through to OpaqueMaterialData (and from there into the GPU material
    // via writeGPUData's OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_
    // EMISSIVE bit). Hash-keyed so we don't spam per draw.
    static std::unordered_set<XXH64_hash_t> sSseConvLogged;
    static std::mutex sSseConvLoggedMu;
    bool firstConv = false;
    {
      std::lock_guard<std::mutex> lk(sSseConvLoggedMu);
      firstConv = sSseConvLogged.insert(getHash()).second;
    }
    if (firstConv) {
      Logger::info(str::format(
        "[ScreenSpaceEmissive.Conv] LegacyMaterialData hash=0x", std::hex, getHash(), std::dec,
        " → OpaqueMaterialData with HasScreenSpaceEmissive=true"
        " mat=(", screenSpaceEmissiveUv1RotScaleX.x, ",", screenSpaceEmissiveUv1RotScaleX.y,
        " | ", screenSpaceEmissiveUv1RotScaleY.x, ",", screenSpaceEmissiveUv1RotScaleY.y, ")",
        " trsl=(", screenSpaceEmissiveUv1Translate.x, ",", screenSpaceEmissiveUv1Translate.y, ")",
        " maskTexValid=", (screenSpaceEmissiveMaskTexture.isValid() && !screenSpaceEmissiveMaskTexture.isImageEmpty() ? 1 : 0)));
    }
  }
  // Indicate that we have an exact sampler to use on this material, directly from game
  if (getSampler().ptr()) {
    opaqueMat.setSamplerOverride(getSampler());
  }
  // Ignore colormap alpha of legacy texture if tagged as 'ignoreAlphaOnTextures' 
  bool ignoreAlphaChannel = LegacyMaterialDefaults::ignoreAlphaChannel();
  if (!ignoreAlphaChannel) {
    ignoreAlphaChannel = lookupHash(RtxOptions::ignoreAlphaOnTextures(), getHash());
  }
  opaqueMat.setIgnoreAlphaChannel(ignoreAlphaChannel);
  return opaqueMat;
}

template<> TranslucentMaterialData LegacyMaterialData::as() const {
  TranslucentMaterialData transluscentMat;
  if (getSampler().ptr()) {
    transluscentMat.setSamplerOverride(getSampler());
  }
  return transluscentMat;
}

template<> RayPortalMaterialData LegacyMaterialData::as() const {
  RayPortalMaterialData portalMat;
  portalMat.getMaskTexture() = getColorTexture();
  portalMat.getMaskTexture2() = getColorTexture2();
  portalMat.setEnableEmission(true);
  portalMat.setEmissiveIntensity(1.f);
  portalMat.setSpriteSheetCols(1);
  portalMat.setSpriteSheetRows(1);
  if (getSampler().ptr()) {
    portalMat.setSamplerOverride(getSampler());
  }
  return portalMat;
}


} // namespace dxvk
