# HANDOFF — TF2 3D-skybox cloud billboards

## Current symptom (the ONE remaining bug)

The **white cloud bodies render correctly** (white, soft). The problem is the
**"black smoke" fringe around the OUTSIDE of the clouds** — the soft wispy
low-coverage edges render **black** in the final image.

Key observation from the user:
- The black smoke is **black in the normal/final view**.
- The black smoke does **NOT appear in any debug view** (not in raw-albedo).
- The white cloud cores **do** show in raw-albedo and render fine.

=> The black smoke is a **final-composite artifact on the translucent edges**,
not an albedo problem. Inner (dense) cloud is fine; outer (wispy, low-alpha)
edge is black.

## Fog reconstruction (IN PROGRESS — built, untested)

Root cause confirmed: the cloud texture RGB is a near-black coverage/detail
map. The visible cloud colour is synthesized by the game PS's atmosphere
math — `colour = lerp(albedo, fogColor*c_fogColorFactor, fogFactor)` with
`fogColor = c_fogParams.k2.xyz*sunAmount² + k1.xyz`. `c_albedoTint` is (1,1,1)
for the main cloud PS (`[CloudFog]` log) — not the brightness carrier.

Implemented (~9 files) — reconstructs that fog blend in Remix:
- `d3d11_rtx.cpp` FillMaterialData: captures `c_fogParams` k0-k3 +
  `c_maxLightingValue` (CBufCommonPerCamera) + `c_fogColorFactor`
  (CBufUberStatic) from cloud draws; `SceneManager::setTf2CloudFog`. Gate:
  PS reads `c_fogColorFactor` AND premultiplied-OVER blend (cloud signature).
- `rtx_scene_manager.h`: `Tf2CloudFogParams` + mutex-guarded set/get.
- `rtx_context.cpp`: copies into `RaytraceArgs.tf2Fog*` each frame.
- `raytrace_args.h`: 4×vec4 `tf2Fog*` (placed after `debugKnob`).
- Material flag `OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG` (bit 9) =
  "fog-synthesizing premultiplied uber material": `shared_constants.h`,
  `rtx_material_data.h` (X-macro `Tf2SkyboxFog`), `rtx_materials.{h,cpp}`,
  `rtx_scene_manager.cpp`.
- Surface flag `Surface::isTf2SkyboxFog` (flags0 bit 4) = the material flag
  AND `InstanceCategories::IgnoreAntiCulling` (3D skybox). Set in
  `rtx_instance_manager.cpp`. THIS is what the shaders gate on — the
  material flag alone is too broad (the premultiplied fog uber-shader is
  used by playable-world smoke/effects/props too; gating on it scene-wide
  washed the whole frame with fog). The 3D-skybox category restricts it to
  the cloud billboards.
- `opaque_surface_material_interaction.slangh`: rebuilds the fog blend into
  `albedo` for flagged surfaces (fogFactor = k0.w, distance-saturated).

Unlit treatment (the lit/unlit half) — DONE:
The soft edges are stochastic-alpha-blend surfaces, composited in
`composite_alpha_blend.comp.slang` as `surfaceLight * albedo * alpha` with
`surfaceLight` borrowed from a neighbour pixel — which is ~0 for the far
skybox. With the now-bright reconstructed albedo, opaque-path frames of a
pixel showed bright while alpha-blend-path frames composited black → the
sky flickered. Fix (2 edits, no new plumbing):
- `geometry_resolver.slangh`: when recording the cloud's AlphaBlendSurface,
  set its `hasEmissive` bit (re-used as "show colour directly, don't light")
  by checking `OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG` on the raw
  `surfaceMaterials[rayInteraction.surfaceIndex].data[0].x`.
- `composite_alpha_blend.comp.slang`: for `hasEmissive` surfaces, override
  `surfaceLight = surface.color.xyz` → outputs the baked colour unlit
  (premultiplied by alpha) instead of multiplying toward black.
Residual: the opaque-g-buffer path still lights the cloud (sky IBL ≈ bright,
so ≈ matches the unlit alpha-blend path). If a subtle lit-vs-unlit shimmer
remains between the two stochastic paths, making the opaque path unlit too
is the polish follow-up.

## What is FIXED and confirmed (do not redo)

1. **Alpha-test from in-shader `clip()`** — `d3d11_rtx.cpp` `FillMaterialData`.
   Reads `c_alphaTestReference` out of the bound PS's `CBufUberStatic` cbuffer
   (RDEF `used` flag + mapped-cbuffer read) and sets
   `mat.alphaTestEnabled / alphaTestCompareOp(=GREATER_OR_EQUAL) /
   alphaTestReferenceValue`. Fixed tree/foliage cutouts rendering opaque-black.

2. **Emissive blend misclassification** — `rtx_instance_manager.cpp`
   `calculateAlphaState`, the `ONE, ONE_MINUS_SRC_ALPHA` branch. Added
   `alphaIsOverComposite`: recognizes BOTH spellings of an alpha-channel OVER
   composite (`ONE/ONE_MINUS_SRC_ALPHA` and `ONE_MINUS_DST_ALPHA/ONE`). When
   the alpha channel is a genuine OVER composite the surface is classified
   `BlendType::kAlpha` (translucent) — NOT `kReverseAlphaEmissive` (emissive).
   Before this the cloud billboards rendered as glowing rectangles.
   CONFIRMED via `[CloudRoute]` log: `route=ALPHA_BLEND blendType=0
   emissiveBlend=0 isUnordered=0`.

3. **sRGB double-decode** — new material flag
   `OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_SRGB`:
   - `src/dxvk/shaders/rtx/utility/shared_constants.h` — flag def (bit 10).
   - `rtx_materials.h` — `RtOpaqueSurfaceMaterial`: ctor param `albedoIsSRGB`,
     member `m_albedoIsSRGB`, `writeGPUData` sets the flag, added to
     `HashStruct` + list-init, `static_assert` size 176 -> 184.
   - `rtx_scene_manager.cpp` `createSurfaceMaterial` — detects sRGB via
     `imageFormatInfo(albView->info().format)->flags.test(ColorSpaceSrgb)`.
   - `opaque_surface_material_interaction.slangh` — skips the software
     `gammaToLinear(albedo)` when the flag is set (HW already sRGB-decoded
     a BC7_SRGB texture; doing it twice crushed albedo near-black).
   NOTE: emissive has the identical `Todo` (line ~1440, `gammaToLinear(
   emissiveColor)`) — left untouched, clouds aren't emissive-textured.

4. **vsync assert crash (unrelated)** — `d3d11_swapchain.cpp` `Present`:
   latches `RtxOptions::enableVsyncState` from the swapchain on first present
   so opening the developer menu doesn't trip the `showVsyncOptions` assert.

## Diagnostic logging still in the tree (REMOVE once clouds are done)

- `[AlphaTestDiag]` — `d3d11_rtx.cpp` FillMaterialData
- `[BlendDiag]` — `d3d11_rtx.cpp` FillMaterialData
- `[SkyDiag]` — `rtx_scene_manager.cpp` `onInstanceUpdated`
- `[CloudRoute]` — `rtx_instance_manager.cpp` (after object-mask decode)
- `[CloudPremultProbe]` — `opaque_surface_material_interaction.slangh`
  (writes gpuPrint ring slot, tag 20.0) + `rtx_context.cpp` slot-20 decode
- `[CloudFog]` — `d3d11_rtx.cpp` FillMaterialData. For cloud-billboard draws
  (PS reads `c_fogColorFactor` + premultiplied-OVER blend) logs the captured
  fog constants k0-k3 + `c_fogColorFactor` + `c_maxLightingValue` once per PS
  hash (gameplay-gated, capped 64). Confirms CAPTURE + material flagging.
- `[CloudFogShader]` — `opaque_surface_material_interaction.slangh` gpuPrint
  ring slot tag 23, decoded in `rtx_context.cpp`. Fires when the shader fog
  reconstruction branch runs on a flagged surface; reports the reconstructed
  colour luminance, fogFactor, sunAmount. Confirms the SHADER half end-to-end
  (material flag + global-cb fog params + shader branch all live). Needs
  gpuPrint enabled (debugView.gpuPrint).
All are gameplay-gated (`tf2::g_engineHookCaptureCount > 16u`).

- `[CloudEdgeDebug]` — three debug views (NOT gameplay-gated; only do work
  when the view is selected, so zero cost otherwise). Written from
  `composite_alpha_blend.comp.slang`; indices 182/183/184 in
  `debug_view_indices.h`; menu names in `rtx_debug_view.cpp`:
  - "Stochastic Alpha Blend Light" (182) — raw light the alpha-blend surface
    received, before albedo/alpha multiply. Black here = surface got no light.
  - "Stochastic Alpha Blend Search Result" (183) — green = light borrowed
    from a neighbour pixel, red = volumetric radiance cache, blue = neither,
    black = no alpha-blend surface at that pixel.
  - "Stochastic Alpha Blend Radiance" (184) — final contribution written to
    AlphaBlendRadiance (light * alpha).
  Requires Stochastic Alpha Blend enabled (the cloud edges route through it;
  `[CloudRoute]` shows `route=ALPHA_BLEND`). The soft low-alpha cloud edges
  are alpha-blend surfaces lit in this pass; the dense white cores get
  promoted into the primary opaque g-buffer instead (why they show in
  raw-albedo and the edges do not).

## Cloud billboard facts (verified from shader_dumps disassembly + logs)

- Game shader dumps: `Titanfall2/shader_dumps/` (DXBC + SPV per shader).
  Disassemble SPV with `C:/VulkanSDK/1.4.321.1/Bin/spirv-dis.exe`.
- Cloud billboard pixel shaders: `FS_635c09a18d267ec4`, `FS_9106f3eb90c7b6de`,
  `FS_16bb3327105ee5ca`. (`FS_44db6ff9a371156e` is a related OPAQUE shader —
  no discard, `o0.a` hardcoded 1.0.)
- Cloud billboard VS hashes: `0x290deec3935b6277`, `0x296dc3ae4947efe6`,
  `0x2a904f3dafd359f5`, `0x2904d2163ef31a17`.
- Cloud albedo textures: BC7_SRGB (VkFormat 146), 128x256 / 256x256, mips=1.
  Hashes seen: `0x61319a307ec7b721`, `0x487039acb2135258`, `0xfdc1dd6e719db570`.
- D3D11 blend state for the cloud draws (from `[CloudRoute]` / `[BlendDiag]`):
  color `src=ONE dst=ONE_MINUS_SRC_ALPHA op=ADD` (premultiplied OVER),
  alpha `src=ONE_MINUS_DST_ALPHA dst=ONE op=ADD` (also OVER), writeMask RGBA.
- Cloud PS color path (`FS_635c09a18d267ec4`, fully traced via fxc /dumpbin):
  `base = sampledTex.rgb * c_albedoTint(cb0[6].xyz) * vertexColor.rgb`
  then HDR clamp `* cb2[33].z / max(maxChannel(base), cb2[33].z)` (this step
  only ever DARKENS, never brightens), then fog lerp toward
  `fogColor * c_layerBlendRamp(cb0[5].z)`; final `o0.rgb = foggedColor * alpha`
  (premultiply). `c_albedoTint` lives in CBufUberStatic offset 96 — the SAME
  cbuffer the alpha-test fix already reads `c_alphaTestReference` from.
  => The cloud's bright look is NOT in the texture (texRGB ~0.02, near-black).
  It comes from `c_albedoTint` and/or `vertexColor.rgb`. `[CloudTintProbe]`
  reads the runtime `c_albedoTint` value to decide which.
- The cloud texture is **straight alpha**, NOT premultiplied: `[CloudPremult
  Probe]` found texels with `lum > alpha`, impossible for premultiplied.
  The PS premultiplies its OUTPUT in-shader.
- The cloud billboards live in the 3D skybox: `IgnoreAntiCulling` category
  (`catRaw=0x20`), `o2wScale ~1000`, huge translation. `cam=0` (Main).
- `[SkyDiag]` proved **TF2 has NO sky-camera classification at all** — zero
  draws ever get `cam=4` (Sky) or `InstanceCategories::Sky`. The 3D skybox is
  recognized only via the `IgnoreAntiCulling` + 1000x reproject bolt-on
  (`SetSkyCategoryFromCb2`), and renders as ordinary Main-camera geometry.

## The remaining bug — analysis & open questions

The black smoke = the translucent (`kAlpha`) low-alpha cloud edges rendering
black in the final composite. NOT in albedo (confirmed by user).

Leading hypothesis (UNVERIFIED — verify before coding):
- TF2 3D-skybox cloud billboards are **unlit, shader-synthesized, premultiplied
  sprites**. Remix now classifies them as translucent `kAlpha` surfaces, which
  get **lit by path-traced scene lights**. Reprojected far-skybox geometry
  receives ~no scene light, so the translucent edge contribution composites
  toward black. Native never lights them — the game shader bakes the look.
- The white cores survive because they're dense enough to read as bright
  regardless; the wispy edges have nothing to carry them.

Candidate causes still to disambiguate:
1. **Lighting** — translucent skybox geometry gets no light => black.
2. **Premultiplied vs straight composite** — game blend is premultiplied OVER;
   Remix's `kAlpha` is straight-alpha. Remix has no premultiplied translucent
   blend type. At low alpha this can darken edges.
3. The cloud texture's edge texels may genuinely be dark RGB (probe saw a
   `lum ~0.02` population) — but native hides that via `rgb*a` premultiply.

## Recommended next steps

1. **Decide: lit vs unlit.** A 3D-skybox cloud sprite is conceptually UNLIT —
   its color is fully baked by the game shader. The correct mapping to a path
   tracer is probably an **unlit / emissive-from-albedo** surface (emit the
   albedo, opacity from alpha) — NOT a lit translucent surface. The earlier
   emissive attempt was wrong ONLY because it used `kReverseAlphaEmissive` +
   `invertedBlend` (glowed where it should be clear). A correct unlit
   treatment for `IgnoreAntiCulling` skybox geometry has not been tried.
2. **Or** investigate proper premultiplied-alpha translucent support (Remix
   only has straight `kAlpha`; decals have premultiplied handling in
   `decal_material_interaction.slangh` — that path could be a model).
3. **Or** the intended Remix workflow: replace the cloud billboards with a
   Remix replacement asset (stable texture hashes above).
4. Longer term, "better than native": volumetric clouds via the froxel system
   (`rtx_global_volumetrics.cpp`, `froxel.slangh`) — a from-scratch feature.

## Unrelated issue noticed

Camera shakes rapidly / briefly flips to 3rd person (player model visible).
`[VM.class]` shows `type=0` stable, `shakeCamera=0` — nothing conclusive.
NOT caused by the cloud/material/shader work (none of it touches the camera
manager). Needs its own investigation in the `CamMgr` / viewmodel-latch
subsystem. Track separately.

## Log / repro

- Remix log: `Titanfall2/rtx-remix/logs/remix-dxvk.log`. Grep the markers
  above.
- The cloud diagnostics are gameplay-gated; just have the sky in view.
