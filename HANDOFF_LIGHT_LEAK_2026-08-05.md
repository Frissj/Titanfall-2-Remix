# HANDOFF — light leaks through missing occluders (view-direction dependent)

**Date:** 2026-08-05 (late session)
**Status of the PREVIOUS bug:** the pitch-driven vanish is **FIXED** — see `CULLING_BIBLE.md` §0.2f.
Do not reopen it. This is a **different symptom** that surfaced after that fix landed.

**Confidence legend:** `[V]` verified (measured / decompiled), `[I]` inferred, `[H]` hypothesis.

---

## 0. THE SYMPTOM (user's words, and read them literally)

> *"if i look in a specific direction light floods in, and if i look another direction, it becomes
> very dark."*

**It is NOT a light being culled.** That was my first reading and it was wrong — the user corrected
it explicitly:

> *"it floods in because its not occluding the bright light outside. its not because of another
> light."*

The bright outdoor light is constant. What changes with view direction is whether the **geometry
that should block it** is present in the TLAS. When the occluder is missing, sun/sky light leaks
through the hole where a wall should be → "floods in". When it is present → correctly dark.

So the thing to hunt is **missing occluder geometry**, not lights, not the light buffer, not
`s_globalLights`. I wasted a chunk of this session on the light path; §5 records that so it is not
re-walked.

---

## 1. HARD CONSTRAINTS — do not violate

- **Never propose, mention, or reason about the object-keeping / lifetime-extension features.** The
  user has ruled these out twice, in explicit terms, including the frame-count option that feeds
  them. They are not an acceptable answer to any part of this. Do not raise them again in any form.
- **Never suggest a rebuild is needed / the binary is stale.** Assume the deployed DLL has the
  latest edit; if a fix does not work, the bug is in the code or the hypothesis.
- **Do not run builds.** The user compiles manually.
- **No hacks, heuristics or allowlists.** Proper plumb only, even across many files.
- **Do not ask the user to choose between investigation branches.** Pick one and do it.

---

## 2. WHAT IS ALREADY FIXED AND MUST NOT BE REOPENED `[V]`

The world visibility cull took **eight** reject sites in `client.dll sub_1802E8DA0`:

| flag | sites | what |
|---|---|---|
| `rtx.cullOff.worldFrustum` | `0x2E8FB0 0x2E904E 0x2E90F1` (node) + `0x2E9756 0x2E97D6 0x2E9879 0x2E9C9C` (leaf) | frustum rejects |
| `rtx.cullOff.worldPortal` | **`0x2E955C`** | area-portal/occluder reject — the eighth, found 2026-08-05 |

`0x2E955C` was missed for a day because it branches to `loc_1802E9DB1`, which **falls through** into
the `loc_1802E9DB7` tail everyone was xref'ing. Full write-up in `CULLING_BIBLE.md` §0.2f.

Also closed by measurement: **job supply is flat across pitch** (`[JobProbe]`, 192→184 calls/frame,
non-monotonic), which excluded `sub_1802EB620` as a target. Do not dig there for this bug either
without new evidence.

---

## 3. THE CRITICAL MEASUREMENT GAP, NOW FIXED `[V]`

**Nothing in the codebase logged YAW.** `[PitchProbe]` emitted `pitchDeg`/`fwdZ` only, so during a
capture where the user was sweeping the camera **horizontally the entire time**, every probe line
read `pitchDeg=10.8816` — constant. I then reported "m1Max is constant under yaw", which was
meaningless: I was binning frames I could not tell apart.

**`yawDeg` is now on `[PitchProbe]`, `[JobProbe]`, `[OccProbe]` and `[DrainProbe]`**
(`rtx_instance_manager.cpp`, computed as `atan2(v2w[2].y, v2w[2].x)` in degrees, range −180..180).

> **Any analysis of this bug that is not binned by `yawDeg` is worthless.** The symptom lives on the
> horizontal axis. Bin by `yawDeg`, compare **magnitudes** not Pearson r.

---

## 4. `[OccProbe]` — BUILT, NOT YET VALIDLY RUN

`rtx_instance_manager.cpp`, in the `[PitchProbe]` block, one line per gameplay frame:

```
[OccProbe] f= pitchDeg= yawDeg= inst= front= behind= behindFar=
```

`behind` = instances whose origin is on the far side of the camera plane
(`dot(instPos − camPos, fwd) < 0`). `behindFar` = the same beyond 256u, to exclude near-camera
artifacts. **That population is the off-screen occluder set** — what a rasteriser has no reason to
submit and a path tracer still needs, because it is what blocks the sun from behind you.

**READ IT, binned by `yawDeg`:**
- `behind`/`behindFar` healthy and steady as you sweep → occluders ARE present; the leak is not
  missing geometry, and the next suspect is the lighting/shadow path, not culling.
- `behind`/`behindFar` ≈ 0, or collapsing as you turn → geometry behind the camera never reaches the
  TLAS. That is the leak, and the fix belongs wherever it is being dropped.

### 4a. THE FIRST VERSION OF THIS PROBE WAS DEFECTIVE — the lesson `[V]`

v1 counted `RtInstance::m_isInsideFrustum` and reported `outFr=0` on **all 459 sampled frames**. That
is not a result, it is a dead field:

```
markAsOutsideFrustum()  is called ONLY from rtx_scene_manager.cpp:563 / :567,
inside the `else` branch opened at rtx_scene_manager.cpp:389,
which does not execute in this configuration.
=> m_isInsideFrustum never leaves its `= true` initialiser (rtx_instance_manager.h:263)
```

`inFr == inst` on every line, forever, regardless of camera. **Before using any engine-maintained
flag in a diagnostic, find the site that CLEARS it and prove that site runs.** This is the same
failure as `[WorldVis]` reporting `m1=0 m2=0` next to `layoutOk=1` (bible §10.2) — it happened twice
in one week. v2 computes the classification itself from `camPos`/`camFwd` and the instance
transform, so it cannot silently no-op.

---

## 5. DEAD ENDS THIS SESSION — do NOT re-walk `[V]`

- **The light path is not it.** `s_globalLights` is a TF2 structured buffer that Remix mirrors and
  walks in `D3D11Rtx::SubmitEngineLights` (`d3d11_rtx.cpp` ~35301) →
  `RtxContext::addLights` → `SceneManager::addLight`. There is a closest-first trim to
  `engineLightSubmitMaxCount`. **None of this is the symptom** — the user confirmed the light is
  constant and the occluder is what changes. Do not instrument light counts for this bug.
- **`m_isInsideFrustum` carries no information in this configuration** (§4a).
- **`inst` jitter of ±14 is not this bug.** It is submission churn, visible as `[InstReap]` lines
  with `age=1`. It is ~2% of ~650 and unrelated to a light leak.
- **`dword_1813C0940` is not a job count** — it is the `JT_GrowJobArray_Lock`/`Unlock` handle. Its
  value churns 2.4M–4.5M with the next three dwords always 0. Already removed from `[JobProbe]`.
- **`[WorldVis]` cannot answer world-cull questions.** It hooks `sub_1801A8350` (the renderable
  path), which never touches M1/M2, so it prints `m1=0 m2=0` beside a valid `layoutOk=1`.

---

## 6. UNEXAMINED CULL SITES ON THE MAIN-VIEW PATH `[I]`

If `[OccProbe]` shows the occluders ARE reaching the TLAS, this is where I would go next. Six
functions read the frustum plane count `dword_1811FC0C0` and only one has ever been examined:

| function | called from | notes |
|---|---|---|
| `sub_1802ED900` (0xd10) | **`sub_1802EB620`** at `0x2EB8C5`, `0x2EC9FD` | **on the main-view path**, right beside `JT_DoneGrowingJobArray`. Reads plane count at `0x2EDE05 0x2EDE47 0x2EDE82 0x2EDEC5 0x2EE308`. Writes the 64-byte records at `unk_181380A40` that `sub_1802E8DA0` later tests — i.e. it is the PRODUCER of the portal plane sets. |
| `sub_1802EAD60` (0x47d) | `sub_1802EB620` at `0x2EB7E1` | 4 plane-count reads |
| `sub_1802E8350` (0x65a) | fn-ptr table `0x183C54D54` only — **no code xrefs** | sibling of the worker; will not decompile |
| `sub_1802ED380`, `sub_1802ED480`, `sub_1802EE940` | — | 3–4 plane-count reads each |

Remember the standing lesson: **"no xrefs" means dispatched indirectly, never dead**, and **xref the
tail AND every label that falls into it** — that is what hid `0x2E955C`.

---

## 7. NEXT STEPS, IN ORDER

1. **Build, then sweep horizontally through the bright↔dark transition**, holding position. Bin
   `[OccProbe]` by `yawDeg`. This is the fork that decides everything (§4).
2. If `behindFar` collapses with yaw → find where geometry behind the camera is dropped. Start at the
   draw-submission side: the visibility mask is full (`[DrainProbe] m1Max` stable) yet instances are
   ~650, so confirm whether the game ISSUES DRAWS for masked-but-off-screen leaves at all.
3. If `behindFar` is stable → occluders are present, and the leak is in the lighting/shadow path.
   Re-scope; it is not a culling bug.
4. Only then consider §6.

**Verification standard for any fix here:** `[OccProbe]` and `[DrainProbe]` binned by **`yawDeg`**,
magnitudes compared, with the camera stationary and `camPos` spread reported. A run that does not
show the bright↔dark transition actually occurring proves nothing — verify the artifact is in the
capture window before drawing a conclusion.

---

## 8. FILES TOUCHED THIS SESSION

| file | change |
|---|---|
| `src/d3d11/d3d11_rtx.cpp` | site 11 (`0x2E955C`, `kCullOffWorldPortal`) + `[JobProbe]`/`[DrainProbe]` install wiring |
| `src/d3d11/tf2_decal_hook.cpp` / `.h` | `[JobProbe]` hook on `0x2E8DA0`, `[DrainProbe]` hook on `0x2F04F0` |
| `src/dxvk/rtx_render/rtx_camera_manager.cpp` | probe counters in `dxvk::tf2` (link-side reason documented there) |
| `src/dxvk/rtx_render/rtx_instance_manager.cpp` | `yawDeg` on all probes; `[OccProbe]` |
| `src/dxvk/rtx_render/rtx_options.h` | `worldPortal`, `probeWorldJobs`, `probeWorldDrain` |
| `Titanfall2/rtx.conf` | `worldPortal=True`; `distanceFade=False`, `staticPropFade=False` (both unvalidated, never measured, turned off to retire unmeasured cost) |
| `CULLING_BIBLE.md` | §STATUS, §0.2 site table, §0.2e corrections, **§0.2f new**, §9, §10.1, §10.2, §13 |
