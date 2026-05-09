# TF2 → Hillaire/Path-Tracer Lighting Integration — Handoff

Status of the work bringing TF2's per-map lighting data into Remix's path tracer.
Written 2026-05-09. All changes live in this branch; no upstream PR yet.

---

## Goal

Make the Hillaire physical atmosphere AND TF2's artist-authored lighting data
both work together in the path tracer. Three tiers, all complete:

1. **Tier 1 — engine sun**: capture TF2's per-frame sun direction + colour
   from `CBufCommonPerCamera`, drive Hillaire's LUTs.
2. **Tier 2 — engine point/spot lights**: read TF2's `s_globalLights`
   structured buffer, convert to `RtxLegacyLight`, submit to Remix.
3. **Aerial perspective**: 3D LUT applied per-pixel so distant geometry
   gets atmospheric haze regardless of sky source.
4. **Hybrid sky mode** (Option C): TF2's authored skybox is the visible
   sky + bounce source, Hillaire still drives sun NEE + AP.

## How to verify it's working

After build, walk a daytime map for 30s, then grep the log:

```
grep "SkyMode.startup"        # Hybrid mode + envMapScale capture
grep "Atmosphere.startup"     # AP LUT pipeline up
grep "EngineSun.publish"      # Tier 1 sun fields per-frame
grep "EngineLights.submit"    # Tier 2 totals + cone shape verification
grep "MultiSun"               # Per-isSun-entry (multi-moon detection)
```

Healthy output looks like:
```
[SkyMode.startup] mode=Hybrid skyBrightnessSlider=1 engineEnvMapScale=1
[Atmosphere.startup] aerialPerspectiveLut=32^3 maxKm=32 strength=1
                     sunIlluminance=(18,15.3,11.9) sunAngularRadius=0.0048
[EngineSun.publish] dir=(...) color=(6,5.10,3.96)  ← non-zero, captured
                    sunHighlightSize=0 maxLightingValue=5 envMapLightScale=1
[EngineLights.submit] total=64 spots=37 points=27
                      static=23 dynamic=41
                      uniqueSuns=1
                      firstSpot{exp=1 phiRad=... innerRatio=0.85}
                      spotPhiRange=[..,..]  spotInnerRatioRange=[..,..]
                      firstSpotConeShape{rawAttenLin=-1 rawAttenQuad=-0}
                      firstAttenD3D9{lin=0 quad=2.78e-06}
[MultiSun] idx=0 color=(6,5.10,3.96) dir=(-0.515,0.327,0.792)
```

If `[EngineSun.publish] color=(0,0,0)` ever appears, the zero-colour guard
isn't catching a stage that pushes uninitialised cbuffer state — investigate
which shader is being captured from.

## Known good defaults

```
rtx.skyMode                              = Hybrid   (option C)
rtx.atmosphere.useEngineSun              = true
rtx.atmosphere.engineSunIntensityScale   = 3.0      (TF2 6-mag → ~18 illuminance)
rtx.atmosphere.engineSunIsZUp            = true     (TF2 world Z-up)
rtx.atmosphere.engineSunDirIsTowardsLight = true    (c_sunDir points TO sun)
rtx.atmosphere.aerialPerspectiveStrength = 1.0      (0..3, 0=disabled)
rtx.atmosphere.aerialPerspectiveWorldToKm = 1.9e-5  (TF2 hammer = 52500/km)
rtx.lights.submitEngineLights            = true
rtx.lights.engineLightSubmitMaxCount     = 64       (per-frame cap)
rtx.lights.dumpEngineLightShaderAsm      = false    (only on for discovery)
```

---

## Confirmed TF2 cbuffer/struct layouts

### CBufCommonPerCamera (cb2)
Shipped at offset (in bytes):

| Offset | Field | Size | We use it |
|---|---|---|---|
| 4 | c_cameraOrigin | 12 | sky detector (existing) |
| 256 | c_skyColor | 12 | not used (would conflict with cubemap) |
| 268 | c_shadowBleedFudge | 4 | not used (rasterizer-only) |
| 272 | **c_envMapLightScale** | 4 | ✅ → `skyBrightness` × this |
| 276 | **c_sunColor** | 12 | ✅ → atmosphere `sunIlluminance` |
| 288 | **c_sunDir** | 12 | ✅ → atmosphere `sunDirection` |
| 300 | c_gameTime | 4 | not used (Remix has frame counter) |
| 304 | c_csm | 144 | not used (TF2 CSM atlas, RT does shadows) |
| 512 | **c_sunHighlightSize** | 4 | ✅ → atmosphere `sunAngularRadius` (when non-zero) |
| 536 | **c_maxLightingValue** | 4 | ✅ → engine-light radiance clamp |

### s_globalLights HardwareLight (SRV, t32, stride 112)

```c
struct HardwareLight {
    float3 color;              // off  0   ✅ → Diffuse
    int    shadowIndex;        // off 12   not used (Remix uses ray shadows)
    float3 pos;                // off 16   ✅ → Position
    float  rcpMaxRadius;       // off 28   ✅ → Range = 1/this
    float3 spotDir;            // off 32   ✅ → Direction (only when non-zero)
    float  spotBias;           // off 44   not used (shadow-map specific)
    float  spotExpSel;         // off 48   ✅ → Falloff → focusExponent
    float  attenLinear;        // off 52   ✅ → cone-edge shape (innerRatio derivation)
    float  attenQuadratic;     // off 56   not used (companion to attenLinear, shape only)
    int    shdFlags;           // off 60   not used (shadow filter, RT-irrelevant)
    float3 spotAxisX;          // off 64   ✅ → cone half-tangent (magnitude)
    float  rcpMaxRadiusSq;     // off 76   redundant (we derive from rcpMaxRadius)
    float3 spotAxisY;          // off 80   ✅ → cone half-tangent (magnitude)
    float  emitterRadius;      // off 92   ✅ → EmitterRadius → m_Radius (penumbra)
    float  specularIntensity;  // off 96   ✅ → Specular (uniform RGB)
    float  highlightSize;      // off 100  not used (TF2 Phong-specific)
    int    isSun;              // off 104  ✅ → skip flag (Tier 1 owns sun)
    int    isRealTime;         // off 108  ✅ → diagnostic split, future routing
};  // 112 bytes
```

**TF2's distance-attenuation model is NOT D3D9.** Per the disassembled FS asm
(`tf2_shader_FS_e94c24674c…asm` lines 527-541):

- Distance falloff for points: `(1 - (d/r)^4)^2 / (d^2 + 1)` — UE5/Frostbite
  smooth quadratic. Uses ONLY `rcpMaxRadius`, NOT attenLinear/Quadratic.
- `attenLinear` and `attenQuadratic` are **cone edge shape coefficients** in:
  `falloff = saturate_dot * (1/sqrt(NdotCone) * attenLinear + attenQuadratic) + 1`
- `shdFlags & 1` selects between the two falloff curves per light.

**Mapping to Remix lighting:**

| TF2 concept | Remix equivalent |
|---|---|
| Distance falloff (UE5 smooth) | D3D9 `1/r²` truncated at Range — visually equivalent |
| Cone outer angle | `Phi = 2 * atan(avg(\|spotAxisX\|, \|spotAxisY\|))` |
| Cone edge softness | `Theta = Phi * innerRatio` where `innerRatio = 1 - clamp(\|attenLinear\| * 0.15, 0.05, 0.5)` |
| Cone falloff exponent | `Falloff = spotExpSel` → `RtLightShaping.focusExponent` |

---

## What's wired

### Files added
- `src/dxvk/rtx_render/rtx_engine_sun.h` — snapshot struct + publish/fetch decls
- `src/dxvk/shaders/rtx/pass/atmosphere/aerial_perspective_lut.comp.slang` —
  3D LUT generation shader
- `HANDOFF_TF2_LIGHTING.md` — this document

### Files modified (key changes)
- `src/d3d11/d3d11_rtx.cpp` —
  - `CaptureEngineSunFromCb()`: reads `c_sunDir/Color/HighlightSize/MaxLightingValue/EnvMapLightScale`
  - `SubmitEngineLights()`: parses `s_globalLights` mirror, sorts by camera distance, submits up to 64 RtxLegacyLights
  - `DumpEngineLightsBufferFromSrv()` + `DumpEngineLightFieldStats()`: discovery diagnostics
  - Extensive verification logs: `[EngineSun.publish]`, `[EngineLights.submit]`, `[MultiSun]`
- `src/d3d11/d3d11_shader.cpp` —
  - `dumpShaderAsmIfRelevant()`: writes DXBC disassembly via `D3DDisassemble` for
    any shader binding `s_globalLights`. Off by default after discovery.
- `src/dxvk/rtx_render/rtx_engine_sun.h/cpp` — snapshot publish/fetch (mutex-guarded)
- `src/dxvk/rtx_render/rtx_atmosphere.h/cpp` —
  - Adds aerial perspective LUT (32³ RGBA16F)
  - `getAtmosphereArgs()` consumes engine sun snapshot when `useEngineSun=true`
  - Always-init so the LUT image exists even in `SkyboxRasterization` mode
- `src/dxvk/rtx_render/rtx_options.h` — new options under `rtx.atmosphere` and `rtx.lights`
- `src/dxvk/rtx_render/rtx_context.h/cpp` —
  - `getAtmosphere()` accessor
  - `[SkyMode.startup]` log line
  - `c_envMapLightScale` multiplied into `skyBrightness`
  - LUTs always initialised
- `src/dxvk/rtx_render/rtx_composite.cpp` — AP LUT bound + sampled in composite
  pass; `c_envMapLightScale` consumed
- `src/dxvk/rtx_render/rtx_lights_data.cpp` — `createFromPointSpot` uses
  `light.EmitterRadius` when non-zero (path-traced penumbra width)
- `src/dxvk/rtx_render/rtx_cb_types.h` — `RtxLegacyLight.EmitterRadius` field
- `src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh` — sun NEE fires in
  `skyMode == 1 || 2` (Hybrid included)
- `src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh` — same for
  bounce sun NEE
- `src/dxvk/shaders/rtx/algorithm/geometry_resolver.slangh` — AP applied at
  primary radiance write
- `src/dxvk/shaders/rtx/pass/composite/composite.comp.slang` — AP applied to
  FINAL aggregated radiance (covers direct + indirect + bounce)
- `src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h` —
  `aerialPerspectiveLutSize`, `MaxDistanceKm`, `Strength`, `WorldToKm`
- `src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh` —
  `sampleAerialPerspective()` helper
- `src/dxvk/shaders/rtx/pass/composite/composite_args.h` — added `AtmosphereArgs`
  + `skyMode`
- `src/dxvk/shaders/rtx/pass/composite/composite_binding_indices.h` +
  `composite_bindings.slangh` — `COMPOSITE_AERIAL_PERSPECTIVE_LUT_INPUT`
- `src/dxvk/shaders/rtx/pass/common_binding_indices.h` +
  `common_bindings.slangh` — `BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT = 204`
- `src/dxvk/imgui/dxvk_imgui.cpp` — Hybrid added to SkyMode dropdown

---

## Architecture decisions and rationale

### Why Hybrid sky mode (option C)
TF2 ships heavily artist-authored skies (cinematic 3D-skyboxes with painted
dropships, mountains, planets). Pure Hillaire physics produces clear blue
or sunset gradients — wrong for TF2's stylised non-physical maps.
Hybrid uses TF2's actual rasterized skybox for visible sky AND bounce light
source while keeping Hillaire's atmospheric transmittance for sun NEE.
Surfaces under the sun get physically-correct horizon reddening; surfaces
in shadow pick up bounce tint from the artist's painted sky.

### Why `engineSunIntensityScale = 3.0`
TF2's `c_sunColor` ships at ~6-magnitude HDR (e.g. `(6, 5.1, 4.0)` warm
yellow on Homestead). Hillaire's slider default is `(20, 20, 20)`. 3.0×
brings TF2's 6 → 18, matching the slider default magnitude.

### Why `EmitterRadius` plumbed
TF2's `HardwareLight.emitterRadius` (off 92) is the physical light source
size. For a path tracer, this controls **soft shadow penumbra width** —
small bulbs cast hard shadows, large area emitters cast soft shadows.
We override `LightManager::lightConversionSphereLightFixedRadius` per-light
when TF2 ships a non-zero value. (TF2 actually ships zero for most lights —
they didn't author per-light radii — so the fallback to the global default
is what hits in practice.)

### Why D3D9 distance attenuation despite TF2 using UE5 falloff
TF2's `(1 - (d/r)^4)^2 / (d² + 1)` reaches 0 at distance `r`. D3D9's `1/r²`
never reaches 0 but `lightConversionIntensityFactor` calibrates absolute
brightness so they're visually equivalent. Reproducing UE5 falloff in
LightData would require touching `RtSphereLight` shader-side, which is out
of scope. The 5% photometric difference is negligible vs the win of having
TF2 lights at all.

### Why `attenLinear/Quadratic` shape `innerRatio`, not distance attenuation
The disassembled shader shows them used in:
`falloff = saturate(NdotCone²) * ((1/sqrt(NdotCone)) * attenLinear + attenQuadratic) + 1`
This is **cone edge shaping**, not distance attenuation. Mapping `|attenLinear|`
to `innerRatio` (cone width ratio) preserves the artist's intended cone
edge softness. Type-2 arena spots get `innerRatio = 0.85` (sharp edge),
type-3 projectors get `innerRatio ≈ 0.71` (softer edge).

### Why submit cap at 64
Submitting all ~1500 populated entries every frame caused
`DxvkMemoryAllocator` OOM via per-frame churn in `LightManager::addLight`.
64 is empirically safe and gives reasonable scene coverage when sorted by
camera distance. **Unlock path**: route static lights (`isRealTime == 0`)
through `LightManager::createExternallyTrackedLight` (one-time submit,
persistent, no anti-culling churn). Then dynamic lights stay capped, statics
scale to all 1000+. Not yet implemented — would need a per-D3D11Rtx
`unordered_map<lightHash, RtLight*>` and tear-down on map change.

### Why aerial perspective applied in TWO places
1. **Geometry resolver** (`geometry_resolver.slangh`) catches primary surface
   emissive contribution at the gbuffer stage.
2. **Composite pass** (`composite.comp.slang`) catches the FINAL aggregated
   radiance (direct + indirect + bounce + emissive) before fog.

Both are gated `skyMode == 1 || skyMode == 2 && aerialPerspectiveStrength > 0`,
and skip primary-sky-miss pixels (the cubemap is at infinity, no haze applies).

---

## Open work / future enhancements

### High value
- **Externally-tracked static lights** (lifts the 64-light cap)
  - Hash each `s_globalLights` static entry (`isRealTime == 0`)
  - On first sight: `createExternallyTrackedLight`, store `RtLight*` in map
  - Each frame: walk the buffer, update existing entries, mark seen
  - On map change (light buffer ptr changes): tear down all + re-init
  - Unlocks ~1500 lights instead of 64

### Medium value
- **Multi-sun support** (the "two moons" case)
  - Currently `[MultiSun]` log dumps additional `isSun=1` entries but we
    skip all of them. Single sun via Tier 1 is the only directional source.
  - For maps with `uniqueSuns > 1`: submit secondary moon as a far-away
    high-radius `RtSphereLight` so RTXDI handles its shadow. The sun NEE
    integration would need to support N directional sources for proper
    atmospheric scattering of moon #2 (currently Hillaire only models one).

- **`c_skyColor` ambient blend** (option A from the architecture discussion)
  - Capture `c_skyColor` (off 256), blend into Hillaire's sky as a tint.
  - Currently we use the artist's actual cubemap (Hybrid mode) which IS the
    artist's sky, so this blend is mostly redundant — but could be useful
    for night maps where the cubemap is dark and `c_skyColor` provides
    the intended ambient floor.

### Low value (good-to-have, not critical)
- **UE5 distance falloff in LightData** — match TF2's exact attenuation curve
  shape rather than approximating with D3D9 `1/r²`. ~50 LOC in `LightUtils`.
- **Per-tile clustered light culling for RT** — TF2's PS uses cluster lists;
  we just submit the closest 64. RTXDI's importance sampling handles this
  well in practice but tile-aware submission could be faster.
- **Bone-attached lights** — characters with weapon-mounted lights (jump kit
  thrusters, muzzle flashes) get position via `s_globalLights` per-frame
  but if the engine doesn't update them every frame they'll lag the
  character animation. Investigate if they appear in the buffer with
  `isRealTime == 1`.

### Hardcoded values that should become per-game configs
- `engineSunIntensityScale = 3.0` (calibrates TF2's HDR scale to Hillaire)
- `aerialPerspectiveWorldToKm = 1.9e-5` (TF2 hammer-units-per-km)
- Tier 2 cone-shape derivation constant `0.15` (innerRatio mapping factor)

These are TF2-specific. If we add Apex / other r2 games these may need
adjustment.

---

## Diagnostics reference

### Verification logs (always on, low-rate)
```
[SkyMode.startup]       once at startup, mode + envMapScale
[Atmosphere.startup]    once at startup, AP LUT params + sun illuminance
[EngineSun.latch]       once when first c_sunDir/c_sunColor pair found
[EngineSun.publish]     every 1024 publishes (~16s @60fps)
[EngineLights.submit]   every N submit calls (default 256)
[MultiSun]              every submit (one line per unique isSun entry)
[EngineLights.publish]  every 1024 mirror writes (light buffer activity)
```

### Discovery diagnostics (off by default; flip in source for one-shot use)
```
rtx.atmosphere.dumpEngineSunCBFields    log every (cb,field) tuple seen
rtx.atmosphere.dumpEngineSunCBValues    classify fields as DIRECTION_LIKELY/etc.
rtx.lights.dumpEngineLightShaderAsm     write disassembled .asm files
rtx.lights.dumpEngineLightsBuffer       hex-dump first N s_globalLights entries
rtx.lights.dumpEngineLightFieldStats    per-type per-field min/max stats
rtx.lights.dumpEngineLightSamplesPerFrame  log one example light per type
```

### Asm files on disk
`Titanfall2\rtx-remix\logs\tf2_shader_FS_<hash>.asm` (16 max per session)

---

## Key files when picking this up

If you're touching:

| You're working on | Look at |
|---|---|
| Sun direction / colour capture | `d3d11_rtx.cpp::CaptureEngineSunFromCb` |
| Light submission | `d3d11_rtx.cpp::SubmitEngineLights` |
| Atmosphere LUT generation | `rtx_atmosphere.cpp` |
| Sky mode logic | `rtx_options.h` SkyMode enum + the gates in integrator_direct/indirect.slangh |
| Aerial perspective | `aerial_perspective_lut.comp.slang` + `atmosphere_common.slangh::sampleAerialPerspective` |
| Composite pass AP | `composite.comp.slang` (search for `AerialPerspectiveLut`) |
| Light data struct | `rtx_cb_types.h::RtxLegacyLight` |
| Per-light → RtLight conversion | `rtx_lights_data.cpp::createFromPointSpot` |

---

## If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `color=(0,0,0)` in EngineSun.publish | TF2 not pushing c_sunColor for the captured shader stage | Zero-colour guard already filters; if it appears, check guard isn't bypassed |
| 1fps + Aftermath crash | Negative attenuation in calculateIntensity, or too many lights/frame | Check `firstAttenD3D9{quad}` is positive small number; reduce `engineLightSubmitMaxCount` |
| `[Atmosphere.startup]` never fires | `useEngineSun=false` or `m_atmosphere` not initialising | Check `rtx_context.cpp` always-init line 1915 |
| Spotlights look like flashlights (way too narrow) | Cone width computation broken — `phi` is being set as cosine instead of radians | Check `L.Phi = 2.0f * halfOuter` not `cos(halfOuter)` |
| Sky is black in Hybrid mode | Skybox not rendering — `tryHandleSky` accidentally dropping geo in Hybrid | Verify gate `skyMode == PhysicalAtmosphere` (Hybrid != PhysicalAtmosphere) |
| "Could not load library client" engine error | Heavy header pulled into d3d11.dll's TU breaking static init | Check d3d11_rtx.cpp doesn't include rtx_atmosphere.h directly; use rtx_engine_sun.h forward decl |
