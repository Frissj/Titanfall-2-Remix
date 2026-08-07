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

// NV-DXVK [perf]: every set*() on a generated material ends in sanitizeData() +
// updateCachedHash(), and both walk EVERY field of the material (18
// TextureRef::getImageHash() + 42 chained XXH64 for the opaque material). Setting
// N fields with set*() therefore hashes the whole material N times to produce one
// result. This builder and as<OpaqueMaterialData>() below both use the set*Deferred()
// form -- field + dirty bit only -- and pay for exactly one finalizeEdits() at the
// end. The material produced is byte-identical; see finalizeEdits() in
// rtx_material_data.h for why. Measured at 8us/call ([Perf.MatData]), ~9 ms of a
// 73.6 ms TF2 frame, because this runs once per no-replacement draw call.
dxvk::OpaqueMaterialData LegacyMaterialData::createDefault() {
  OpaqueMaterialData opaqueMat;
  opaqueMat.setAnisotropyConstantDeferred(LegacyMaterialDefaults::anisotropy());
  opaqueMat.setEmissiveIntensityDeferred(LegacyMaterialDefaults::emissiveIntensity());
  opaqueMat.setAlbedoConstantDeferred(LegacyMaterialDefaults::albedoConstant());
  opaqueMat.setOpacityConstantDeferred(LegacyMaterialDefaults::opacityConstant());
  opaqueMat.setRoughnessConstantDeferred(LegacyMaterialDefaults::roughnessConstant());
  opaqueMat.setMetallicConstantDeferred(LegacyMaterialDefaults::metallicConstant());
  opaqueMat.setEmissiveColorConstantDeferred(LegacyMaterialDefaults::emissiveColorConstant());
  opaqueMat.setEnableEmissionDeferred(LegacyMaterialDefaults::enableEmissive());
  opaqueMat.setEnableThinFilmDeferred(LegacyMaterialDefaults::enableThinFilm());
  opaqueMat.setAlphaIsThinFilmThicknessDeferred(LegacyMaterialDefaults::alphaIsThinFilmThickness());
  opaqueMat.setThinFilmThicknessConstantDeferred(LegacyMaterialDefaults::thinFilmThicknessConstant());
  // Finalize here rather than leaving it to the caller: createDefault() is a public
  // factory and must hand back a material whose getHash() is current. as<>() below
  // then does its own single finalizeEdits() after its own run of deferred writes,
  // so a conversion costs two full hash passes instead of the ~18 it used to.
  opaqueMat.finalizeEdits();
  return opaqueMat;
}

// NV-DXVK [perf]: see the declaration in rtx_materials.h for why getHash() cannot
// be used as the conversion key. Keep this in sync with as<OpaqueMaterialData>()
// below -- a field read there but missing here is a false cache hit, i.e. one
// draw silently rendered with another draw's material.
XXH64_hash_t LegacyMaterialData::getOpaqueConversionKey() const {
  // Texture identity, not texture content hash: getUniqueKey() is a plain member
  // read, while getImageHash() chases into the image object. Same uniqueKey means
  // the same underlying view (it is derived from the VkImageView handle), which is
  // all the conversion needs -- the converted material stores the TextureRef
  // itself, so streaming that swaps a mip view stays correct through a cache hit.
  auto texKey = [](const TextureRef& t) -> uint64_t {
    // getUniqueKey() asserts on an invalid ref, so gate on isValid() first.
    return t.isValid() ? static_cast<uint64_t>(t.getUniqueKey()) : 0ull;
  };

  struct KeyData {
    XXH64_hash_t legacyHash;
    uint64_t     texKeys[12];
    uint64_t     sampler;
    uint32_t     texEmptyMask;
    uint32_t     flags;
    float        emissiveTint[3];
    float        sseRow0[2];
    float        sseRow1[2];
    float        sseTranslate[2];
  };

  // memset, do NOT use an aggregate initialiser. KeyData has trailing padding and
  // this is hashed with sizeof(), so uninitialised padding would give one material
  // a different key on different calls -- the exact failure mode documented on
  // updateCachedHash() above, where a byte-for-byte identical material produced a
  // different hash depending on the call path.
  KeyData kd;
  std::memset(&kd, 0, sizeof(kd));
  static_assert(alignof(KeyData) <= 8 && sizeof(KeyData) % 4 == 0);

  // Covers the 11 texture content hashes plus blend and alpha-test state, and is
  // itself read by the conversion via lookupHash(ignoreAlphaOnTextures, getHash()).
  kd.legacyHash = m_cachedHash;

  // Order must match the texEmptyMask bit order below.
  const TextureRef* const texes[12] = {
    &colorTextures[0], &colorTextures[1], &normalTexture, &roughnessTexture,
    &metallicTexture,  &emissiveTexture,  &ambientOcclusionTexture,
    &lightmapTexture,  &lightmap2Texture, &detailTexture, &cloudMaskTexture,
    &screenSpaceEmissiveMaskTexture
  };
  for (uint32_t i = 0; i < 12; ++i) {
    kd.texKeys[i] = texKey(*texes[i]);
    // The conversion branches on isImageEmpty(), not just validity, so the key has
    // to carry it or a texture that streams in mid-scene would keep serving the
    // cached no-texture material.
    if (texes[i]->isImageEmpty()) {
      kd.texEmptyMask |= (1u << i);
    }
  }

  // Pointer identity is sound here: a cache entry holds an Rc to its sampler, so
  // that address cannot be recycled by a different sampler while the entry lives.
  kd.sampler = reinterpret_cast<uint64_t>(getSampler().ptr());

  kd.flags = (sourceUsesEmission               ? 1u << 0 : 0u)
           | (sourceAlphaModulatesEmissive     ? 1u << 1 : 0u)
           | (sourceIsUnlitUI                  ? 1u << 2 : 0u)
           | (sourceTf2FogCapable              ? 1u << 3 : 0u)
           | (sourceAlbedoIsPremultiplied      ? 1u << 4 : 0u)
           | (sourceForceIgnoreAlphaChannel    ? 1u << 5 : 0u)
           | (hasScreenSpaceEmissive           ? 1u << 6 : 0u);

  kd.emissiveTint[0] = sourceEmissiveTint.x;
  kd.emissiveTint[1] = sourceEmissiveTint.y;
  kd.emissiveTint[2] = sourceEmissiveTint.z;
  kd.sseRow0[0]      = screenSpaceEmissiveUv1RotScaleX.x;
  kd.sseRow0[1]      = screenSpaceEmissiveUv1RotScaleX.y;
  kd.sseRow1[0]      = screenSpaceEmissiveUv1RotScaleY.x;
  kd.sseRow1[1]      = screenSpaceEmissiveUv1RotScaleY.y;
  kd.sseTranslate[0] = screenSpaceEmissiveUv1Translate.x;
  kd.sseTranslate[1] = screenSpaceEmissiveUv1Translate.y;

  return XXH3_64bits(&kd, sizeof(kd));
}

template<> OpaqueMaterialData LegacyMaterialData::as() const {
  // Legacy materials have parameters that can directly carry over onto the opaque material.
  // NV-DXVK [perf]: createDefault() already returns a fresh value-typed default;
  // move/copy-elide it straight into opaqueMat instead of materializing a separate
  // `defaultLegacyOpaqueMaterial` and copy-constructing from it. That dropped one
  // full OpaqueMaterialData copy (a large struct holding ~18 TextureRefs) per
  // conversion — and this runs once per no-replacement draw, i.e. ~every TF2 draw.
  OpaqueMaterialData opaqueMat = createDefault();
  if (LegacyMaterialDefaults::useAlbedoTextureIfPresent()) {
    opaqueMat.setAlbedoOpacityTextureDeferred(getColorTexture());
  }
  if (getColorTexture2().isValid()) {
    opaqueMat.setSecondaryTextureDeferred(getColorTexture2());
  }
  // NV-DXVK: forward PBR maps discovered from the pixel-shader RDEF.
  // Populated by D3D11Rtx::FillMaterialData for games that name their SRVs
  // (e.g. Titanfall 2: albedoTexture/normalTexture/glossTexture/specTexture/
  // emissiveTexture at t0..t4). When empty, the opaque material falls back
  // to the rtx.legacyMaterial.* constants for that channel.
  if (normalTexture.isValid() && !normalTexture.isImageEmpty()) {
    opaqueMat.setNormalTextureDeferred(normalTexture);
  }
  if (roughnessTexture.isValid() && !roughnessTexture.isImageEmpty()) {
    opaqueMat.setRoughnessTextureDeferred(roughnessTexture);
  }
  if (metallicTexture.isValid() && !metallicTexture.isImageEmpty()) {
    opaqueMat.setMetallicTextureDeferred(metallicTexture);
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
      opaqueMat.setEmissiveColorTextureDeferred(emissiveTexture);
    }
    opaqueMat.setEmissiveColorConstantDeferred(sourceEmissiveTint);
    opaqueMat.setEnableEmissionDeferred(true);
    // Tint already carries the strength via its magnitude, so leave the
    // intensity scalar at unit. Game's c_emissiveTint=(1,1,1,_) for "use
    // texture as-is", smaller magnitudes for dimmed effects.
    opaqueMat.setEmissiveIntensityDeferred(1.0f);
    opaqueMat.setAlphaModulateEmissiveDeferred(sourceAlphaModulatesEmissive);
    // NV-DXVK: signal the slang shader that emissiveColorConstant carries
    // a per-draw tint that must MULTIPLY the per-pixel emissive texture
    // sample (vs. the legacy overwrite-as-fallback semantic). Required for
    // exact PS fidelity on materials whose c_emissiveTint != (1,1,1).
    opaqueMat.setEmissiveTintFromConstantDeferred(true);
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
      opaqueMat.setEmissiveColorTextureDeferred(getColorTexture());
    }
    opaqueMat.setEmissiveColorConstantDeferred(Vector3(1.f, 1.f, 1.f));
    opaqueMat.setEnableEmissionDeferred(true);
    opaqueMat.setEmissiveIntensityDeferred(1.0f);
    // Tint = (1,1,1) and EmissiveTintFromConstant=true → slang multiplies
    // emissiveSample × (1,1,1) = emissiveSample (no tint shift), exactly
    // what the original PS produces.
    opaqueMat.setEmissiveTintFromConstantDeferred(true);
    // Routed to surface.isMatte at instance build time — see
    // rtx_instance_manager.cpp:appendInstance.
    opaqueMat.setIsUnlitOutputDeferred(true);
  }

  // NV-DXVK: TF2 3D-skybox cloud billboard — forward the fog-reconstruction
  // marker. d3d11_rtx.cpp::FillMaterialData set this when the PS reads
  // c_fogColorFactor and the draw is premultiplied-blended. The opaque
  // surface shader uses the resulting OPAQUE_SURFACE_MATERIAL_FLAG_TF2_
  // SKYBOX_FOG to rebuild the game's fog-blend synthesis from cb.tf2Fog*.
  if (sourceTf2FogCapable) {
    opaqueMat.setTf2SkyboxFogDeferred(true);
  }
  // NV-DXVK: premultiplied-alpha-blend source — forward the marker so the
  // GPU material gets OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_PREMULTIPLIED.
  // The slang shader then skips the opacity-multiply inside albedoToAdjusted-
  // Albedo / calcBaseReflectivity for these surfaces (their .rgb is already
  // premultiplied, multiplying again would double-darken soft edges).
  if (sourceAlbedoIsPremultiplied) {
    opaqueMat.setAlbedoIsPremultipliedDeferred(true);
  }
  if (ambientOcclusionTexture.isValid() && !ambientOcclusionTexture.isImageEmpty()) {
    opaqueMat.setAmbientOcclusionTextureDeferred(ambientOcclusionTexture);
  }
  if (lightmapTexture.isValid() && !lightmapTexture.isImageEmpty()) {
    opaqueMat.setLightmapTextureDeferred(lightmapTexture);
  }
  if (lightmap2Texture.isValid() && !lightmap2Texture.isImageEmpty()) {
    opaqueMat.setLightmap2TextureDeferred(lightmap2Texture);
  }
  if (detailTexture.isValid() && !detailTexture.isImageEmpty()) {
    opaqueMat.setDetailTextureDeferred(detailTexture);
  }
  if (cloudMaskTexture.isValid() && !cloudMaskTexture.isImageEmpty()) {
    opaqueMat.setCloudMaskTextureDeferred(cloudMaskTexture);
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
    opaqueMat.setHasScreenSpaceEmissiveDeferred(true);
    opaqueMat.setScreenSpaceEmissiveMatRow0Deferred(screenSpaceEmissiveUv1RotScaleX);
    opaqueMat.setScreenSpaceEmissiveMatRow1Deferred(screenSpaceEmissiveUv1RotScaleY);
    opaqueMat.setScreenSpaceEmissiveTranslateDeferred(screenSpaceEmissiveUv1Translate);
    if (screenSpaceEmissiveMaskTexture.isValid() && !screenSpaceEmissiveMaskTexture.isImageEmpty()) {
      opaqueMat.setScreenSpaceEmissiveMaskTextureDeferred(screenSpaceEmissiveMaskTexture);
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
  // NV-DXVK: structural per-draw override from D3D11 state — set in
  // FillMaterialData when blend is disabled AND no alpha test fires.
  // For those surfaces the alpha channel is non-load-bearing; without
  // this override the sampled alpha leaks into opacity and
  // albedoToAdjustedAlbedo darkens encoded albedo, producing the
  // 0x2a729-class residual drift the previous-handoff fix left behind.
  // OR'd with the existing hash-allowlist path so manually-tagged
  // textures still win when applicable.
  if (sourceForceIgnoreAlphaChannel) {
    ignoreAlphaChannel = true;
  }
  opaqueMat.setIgnoreAlphaChannel(ignoreAlphaChannel);
  // Single flush for every set*Deferred() above. setSamplerOverride /
  // setIgnoreAlphaChannel are hand-written accessors that never touched the hash,
  // so their position relative to this call does not matter.
  opaqueMat.finalizeEdits();
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
