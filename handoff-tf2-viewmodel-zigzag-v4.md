# HANDOFF v4 — TF2 (Titanfall 2) / dxvk-remix-DX11 — Viewmodel + platform horizontal zig-zag

**Status:** NOT FIXED, but the gun is **positively identified** and the root is **measured and
localized** (not guessed). v4's headline results: (1) a reusable coverage tool that pins any
on-screen object to a VS hash; (2) the gun VS confirmed; (3) the zig-zag is a **real 1-frame
geometric lag in the gun's world-space position buffer**, NOT a camera lag and NOT the bone-0
transform. Read "THE BIG CORRECTION" — it overturns v3's "Main-instance transform" framing AND
my own mid-session "1-frame camera lag" dead-end.

- Repo: `C:\Users\Friss\Documents\RemixDX11\dxvk-remix-DX11`
- Log:  `...Titanfall2\rtx-remix\logs\remix-dxvk.log`  (read directly; user says "check log")
- Conf: `...Titanfall2\rtx.conf`
- Scene: campaign intro "THE ARK" (moving platform). Bug: first-person weapon **and the
  platform you stand on** jerk/zig-zag HORIZONTALLY while walking; freezes at rest. Only with
  `rtx.useEngineHookMainCamera = True`. Visible in **Raw Albedo debug view** (so NOT temporal /
  denoiser — it is geometric).

---

## GUN POSITIVELY IDENTIFIED
**Gun VS hash = `0x292b6ba0d1854f28`.** Found with the new PickRegion coverage probe (largest
stable center-bottom surface in the bottom-right rect) and CONFIRMED by a hide test:
`rtx.debug.hideVertexShaders = 0x292b6ba0d1854f28` removes the weapon on screen. It is a
**Main-classified** (`camType=0`), **rigid** (`boneXf=0`, NOT GPU-skinned) mesh placed via
**`o2wPathId=3` / `wtvPathId=3`** (bone-0 matrix → `transforms.objectToWorld`). It is NOT the
3-vtx decoy / viewmodel pipeline that v2/v3 chased.

---

## THE NEW TOOL — PickRegion coverage probe (keep; generally useful)
Color-independent, location-aware VS attribution. The old `RawAlbedoColor` tool identifies an
object by **color** (per-VS avg255 + pixel count, whole screen); PickRegion identifies by
**location** (bins the primary `SharedSurfaceIndex` over a configurable screen rect, reports
per-VS pixel count + screen bbox). That's why it could find the gun (untextured, doesn't
dominate the frame) when the color tool couldn't.

- Conf: `rtx.surfaceCoveragePickRegion = minXFrac, minYFrac, maxXFrac, maxYFrac` (default
  `0.5,0.5,1.0,1.0` = bottom-right; origin top-left). Sweepable at runtime, no rebuild.
- Needs `rtx.logSurfaceCoverage = True` + an active debug view (Raw Albedo, `debugViewIdx=32`).
- Output: `[Coverage] PickRegion scanned=… validPixels=… screenBox …` then ranked
  `[Coverage]   PickRegionVS VS=0x… pixels=… box x=[..] y=[..] w h colorTexture=… material=…`.
- Code: regions 59–64 in `common_binding_indices.h`; shader probe in
  `debug_view_postprocess.comp.slang`; CB field `surfaceCoveragePickRegion` in
  `debug_view_args.h` (filled in `rtx_debug_view.cpp`); readback in `rtx_context.cpp`.
- One-shot freshness marker (rev-bumped each build): `[PickRegion BUILD MARKER] rev=N readback
  LOADED …` (currently **rev=5**). If you see it but not `[Coverage] PickRegion`, the rect is
  degenerate; if you see neither while `DebugViewScan` lines appear, the running dll is stale.

---

## THE BIG CORRECTION — measured root (overturns v3 AND my own session detour)
Re-anchored `[ZigGun]`/`[ZigVB]`/`[ZigNDC]`/`[ZigGunD3D]` onto the real gun (select
`s_zigGunInstance` by VS `0x292b6ba0`, highest vtx). Then **measured**, not theorized:

1. **No camera lag.** `[ZigGunFix]` + timestamp-correlated `[ZigVB]`: `camMain` (CameraManager
   Main render camera) `== g_engineMainCamOrigin` (live engine camera) EXACTLY, same wall-clock.
   The "render camera is 1 frame behind" idea (and the documented EndFrame 1-frame lag at
   `d3d11_rtx.cpp ~21601`) does NOT manifest as a position lag here.
2. **Bone transform is glued.** At draw/bake time `rawBoneT.Y - live.Y = CONSTANT` (~+414, no
   sawtooth). The bone-0 `transforms.objectToWorld` (o2wPathId=3) tracks the camera perfectly.
   So the placement transform is correct.
3. **My "consumed-origin" fix was a NO-OP** (`m_renderCamOriginConsumed == live`, shift=0) and
   was **reverted**. Do not re-try a camera-origin re-glue on the bone transform — measured dead.
4. **The position BUFFER lags 1 frame (REAL).** `[ZigVB]` originally compared the readback ring's
   PREVIOUS-frame verts to the LIVE camera → a phantom sawtooth = the camera's per-frame step.
   I FIXED the probe to be **frame-aligned** (compare verts to the camera captured at the same
   copy frame, `s_zvbCamEye[]`). **dY STILL sawtooths ±120 (−260↔−380), freezing at −253 at
   rest.** So the gun's world-space **`modifiedGeometryData.positionBuffer`** (what is actually
   ray-traced, written by the skinning/correction compute pass) is **one frame stale relative to
   camMain**, even though the bone-0 transform feeding it is current. That ~120u constant
   world-space offset is invisible on far world geometry but a big screen swing on near geometry
   (gun + the platform you stand on) — pure perspective/parallax (`screen ∝ worldΔ / distance`).

**So the bug is in the vertex bake, not the camera and not the bone transform.** The thing that
renders (`positionBuffer`) is 1 frame behind the camera it's rayed with.

---

## v5 — FIX B IMPLEMENTED (phase-align Main to the geometry stream)
**The root was refined past v4's "positionBuffer written stale by a compute pass."**
The positionBuffer is NOT stale-written — it is this-frame's correct game submission
(`o2wT=(0,0,0)`, identity placement; verts are world-baked by TF2). Correlating three
traces by engine-frame (at remixFrame 636, walking):
- Gun verts `v0.y` = −15171.2  → engine frame **23**
- `camMain` (rendered)  = −15150.8  → engine frame **24**
- Live engine cam     = −15149.3  → engine frame **25**

Each is exactly **one engine-frame apart**. So Main lags the live engine camera by 1
frame (the deferred EndFrame consumer), and the gun's world-baked verts lag Main by 1
MORE frame. The bug exists ONLY with `useEngineHookMainCamera=True` because that path
makes Main the R_DrawWorldMeshes capture, which is phase-shifted from the D3D draw
stream that bakes the gun/platform verts (Source's queued renderer + DXVK EmitCs
deferral). With the hook OFF, Main = per-draw classifier camera = same draw stream as
the verts → glued. Static world never reveals it (constant verts → reads as latency);
only MOVING geometry (gun + ARK platform, both baked per-frame at live world pos) shows
it as a horizontal zig-zag that freezes at rest.

**Why v4's `m_renderCamOriginConsumed` fix was a no-op:** it used the wrong delta —
`consumedOrigin − liveOrigin` (intra-frame, ≈0) applied to the identity bone transform.
The delta that glues is the INTER-frame camera step. Proof: at f=636,
`camMain(N) − camMain(N−1) = −15150.8 − (−15171.8) = +21.0`; adding it to `v0.y=−15171.2`
→ `−15150.2`, matching Main. ✓

**Fix B (implemented):** delay the engine-hook Main camera by N engine-frames so its
pose phase-aligns with the geometry of the same engine frame. New runtime-tunable option
`rtx.engineHookMainCameraFrameDelay` (default **1** = the measured fix; 0 = legacy
zig-zag). Implemented as an 8-deep ring of distinct-engine-frame captures in the EndFrame
consumer; Main is fed the capture N frames behind the newest. Fixes gun + platform + any
moving geometry at once (single Main pose).
- Code: `rtx_options.h` (new RTX_OPTION), `d3d11_rtx.h` (ring members
  `m_engineCamRing*`, `m_engineCamDelayed*`), `d3d11_rtx.cpp` EndFrame consumer
  (advance branch ~21683 pushes+selects; `m_renderCamOriginConsumed` recovered from the
  delayed sample; no-advance branch ~22133 re-feeds the cached delayed pose).
- Conf: `rtx.engineHookMainCameraFrameDelay = 1` added to rtx.conf.

**VERIFY (one build, then runtime sweep — no rebuild needed):** walk in THE ARK with the
[ZigVB] probe on. `dY` should go (near-)CONSTANT while walking instead of sawtoothing.
If 1 doesn't fully glue, sweep `rtx.engineHookMainCameraFrameDelay = 0/2/3` at runtime.
CAVEAT: the [ZigVB] readback reads the first 4 verts of positionBuffer, whose ordering
isn't a stable point on the gun (v0.x jumps −15.88↔−199.58 across frames), so judge the
fix primarily by the VISUAL (gun + platform glued in Raw Albedo) and the trend of dY, not
a single vertex's exact value.

## NEXT STEP (concrete) — superseded by v5 fix B above; original v4 plan kept for record
Find where the gun's `modifiedGeometryData.positionBuffer` is written and why its content is 1
frame stale vs `camMain`:
- `[ZigVB]` copies `geo.positionBuffer` and notes "the skinning/correction compute pass wrote
  these" (`rtx_geometry_utils.cpp ~612`). Trace that write: `interleaveGeometry` /
  `dispatchViewModelCorrection` / the geometry-processing path in `rtx_geometry_utils.cpp`,
  `rtx_scene_manager.cpp`, `rtx_accel_manager.cpp` (grep `modifiedGeometryData.positionBuffer`,
  `dispatchViewModelCorrection`, `interleaveGeometry`).
- Determine the failure shape: either (a) the compute pass bakes with a **1-frame-stale
  transform** (cache/prev-frame matrix) while camMain is current — fix = feed it the current
  transform; or (b) a **pipeline/double-buffer latency** where this frame's `positionBuffer` is
  rebuilt but the BLAS/TLAS rayed this frame is from last frame's buffer — fix = re-anchor at
  TLAS/instance time to the current camera, or remove the latency.
- Verify with the frame-aligned `[ZigVB]` dY → must go **constant** while walking. Cross-check
  visually in Raw Albedo (gun + platform glued).

DISTANCE DISCRIMINATOR already done: jumps in Raw Albedo ⇒ geometric (not temporal/denoiser).
DON'T re-do camera-lag or bone-transform fixes — both measured clean.

---

## RULED OUT / KEY FACTS
- Camera lag: `camMain == g_engineMainCamOrigin` exactly (no position lag). EXONERATED.
- Bone-0 `objectToWorld` (o2wPathId=3): glued (`rawBoneT - camMain = const`). NOT the cause.
- `g_engineMainW2v` translation is at indices **3/7/11** (column 3 of a row-major view matrix),
  NOT 12/13/14 (that's the bottom row = 0,0,0,1). The `[ZigGunD3D]` `engW2vT` field reads the
  wrong indices and always shows (0,0,0) — ignore it; use `g_engineMainCamOrigin`.
- `[ZigVB]` PRE-FIX dY sawtooth was a measurement artifact (stale verts vs live camera). Now
  frame-aligned; the REMAINING sawtooth is real.

---

## DIAGNOSTIC PROBES (remove all when done; prefixes NOT in log.cpp filter)
- `[ZigGun]` — `rtx_instance_manager.cpp ~2530`: RE-ANCHORED to VS `0x292b6ba0`; tags
  `s_zigGunInstance` (highest vtx) and logs per-frame instance transform + projected ndcX.
  (Gameplay-gated; requires the gun NOT be in `hideVertexShaders`.)
- `[ZigGunRB]`/`[ZigVB]`/`[ZigNDC]` — `rtx_geometry_utils.cpp debugReadbackViewModelVerts`:
  readback of the gun's `positionBuffer` (now reads the real gun via the re-anchor). `[ZigVB]`
  dY is now **frame-aligned** (`s_zvbCamEye[]` added) — this is THE root signal. Keep until fixed.
- `[ZigGunD3D]` — `d3d11_rtx.cpp ~8205` (transform finalize, keyed on gun VS): logs
  o2wPathId/wtvPathId/boneXf + o2wT + camera fields. Confirmed o2wPathId=3, boneXf=0, glued.
- Viewmodel-pipeline probes (`[ZigInst]/[ZigCand]/[ZigDispatch]/[ZigBlas]`) watch the 3-vtx
  decoy — IGNORE; not the Main gun.
- PickRegion regions/marker (see tool section) — optional to keep (generally useful).

## CODE STATE — changes this session
- **PickRegion coverage tool** (new, keep): `common_binding_indices.h` (regions 59–64,
  `COVERAGE_TOTAL_REGIONS`=65), `debug_view_args.h`, `debug_view_postprocess.comp.slang`,
  `rtx_options.h` (`surfaceCoveragePickRegion`), `rtx_debug_view.cpp`, `rtx_context.cpp`
  (readback + build marker rev=5).
- **`[ZigGun]` re-anchor** on VS `0x292b6ba0` (`rtx_instance_manager.cpp`).
- **`[ZigGunD3D]` probe** (`d3d11_rtx.cpp ~8205`).
- **`[ZigVB]` frame-aligned camera** (`rtx_geometry_utils.cpp`: `s_zvbCamEye[]` stored at copy,
  used in dY). This is the key probe correction — keep.
- **REVERTED**: the bone-0 `objectToWorld` "consumed-origin" shift (o2wPathId=3 block,
  `d3d11_rtx.cpp ~6448`) — left a NOTE comment; the member `m_renderCamOriginConsumed` +
  EndFrame snapshot (`d3d11_rtx.cpp ~21713`, `d3d11_rtx.h`) are harmless leftovers (no-op).
- **Build script** `build-remixDx11.bat` hardened (see PROCESS LESSONS) — unrelated to the bug
  but fixed real recurring breakage this session.

## CONF STATE (rtx.conf, current)
`rtx.debug.hideVertexShaders = 0x292b6ba0d1854f28` (the GUN — REMOVE THIS to measure/repro;
hidden = mask 0 = excluded from the readback the probes consume).
`rtx.logSurfaceCoverage = True`, `rtx.coverageSyncBeforeReadback = True`,
`rtx.useEngineHookMainCamera = True`, `rtx.debugView.debugViewIdx = 0` (set to **32** = Raw
Albedo, or switch via the Remix menu, for PickRegion + the visible repro).
`rtx.surfaceCoveragePickRegion` defaults to bottom-right if unset.

## PROCESS LESSONS
- **Identify the object before measuring** (PickRegion → VS hash + hide test). v2/v3 measured a
  decoy for two sessions.
- **Distrust frame-delayed probes for phase analysis.** The `[ZigVB]` readback ring compares
  PREVIOUS-frame verts to the LIVE camera — that alone manufactures a 1-frame sawtooth equal to
  the camera step. Frame-align (compare to the camera captured at the verts' copy frame) before
  concluding anything about lag. I lost a cycle to this.
- **Measure, don't theorize, on camera/transform timing.** `camMain==live` and bone-glued were
  both provable in one run; the "1-frame camera lag" theory (mine and the old handoff's) was
  wrong. The real lag is in the position BUFFER, downstream.
- **Build/deploy gotchas** (now fixed in `build-remixDx11.bat`): ninja tracks the shader `.spv`
  but not the `.h` C++ includes (orphan `.h` → C1083); fast mode now self-heals orphans + only
  rebuilds `.d`-dependents of changed includes; deploy silently skipped a **locked** `d3d11.dll`
  while still copying `.spv` (stale dll, fresh shaders) — now a post-deploy size verify +
  retry + loud fail; the partial-build marker was cleared too late (after deploy) causing
  needless shader re-wipes — now cleared right after ninja produces the dll. See memory
  [[reference_deploy_stale_dll]].
