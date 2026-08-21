# HANDOFF — RESIDENT SCENE / PULL ARCHITECTURE

**Session 2026-08-20.** Companion to `RESIDENT_SCENE_PLAN.md` (the design).
This file is state, corrections and traps. The plan is the what; this is the
where-we-actually-are.

---

## 0. STATUS IN ONE SCREEN

```
engine culling      ON   (rtx.cullOff.enable = False -- master OFF = no patches = stock engine)
                         57.99 ms -> 19.97 ms, 17.2 -> 50.1 fps, drawsIn 1338 -> 538
                         scene healthy: inst ~1670, uniqueBlas ~217

residency           BUILT, INERT.  rtx.residentScene.enable = False
                    Slices 1-5 of the plan are in the tree. Nothing is armed,
                    nothing on screen can move.

census              BUILT, LIVE, UNCONDITIONAL.  [SceneCensus] / [VsResidency]
                    Not behind an option -- it is the measurement that decides
                    the architecture. Gameplay-gated (100-draw floor), 10-frame
                    throttle.

stall watchdog      BUILT, LIVE.  [StallWatch]  <- NOT YET RUN, no data
                    Announces "watchdog armed" at startup. If that line is
                    absent it did not start.
```

**The correctness cost being paid right now:** engine culling deletes off-screen
geometry from the RT scene. Live and measurable as
`[ReapJoin] respawn=0 starved=1..22/frame`, which is this tree's own definition
of "the geometry received fewer draws than it had instances". `respawn=0` proves
it is not a dedup failure. **When residency lands, `starved` -> ~0. That is the
acceptance gate and it needs no new instrumentation.**

---

## 1. WHAT IS BUILT

| slice | files | state |
|---|---|---|
| 1 options | `rtx_options.h` (`ResidentScene` block) | done, one batched pass |
| 2 record store | `rtx_resident_scene.{h,cpp}` (new), `meson.build` | done |
| 3 lifetime contract | `rtx_instance_manager.h` (`m_residentKey`, friend) | done |
| 4 GC clause + invalidation | `rtx_instance_manager.cpp` | done |
| 5 frame-thread gate | `d3d11_rtx.{h,cpp}` | **census only; the GATE is not written** |
| 6 TouchRecord + CS stamp | — | not started |
| 7 arm it | conf | blocked on §3 |
| 8 seed pass | — | not started |

`RtInstance` grew 952 -> 960 (`m_residentKey`); the size assert and the
copy-ctor skip-list are updated. The copy ctor needed no code change (explicit
member-init list, so the field defaults to 0, which is the required policy).

**NO ANTI-CULLING.** Nothing added reads or writes `rtx.antiCulling.*`,
`AntiCulling::isObjectAntiCullingEnabled` or `InstanceCategories::IgnoreAntiCulling`.
The existing `IgnoreAntiCulling` long-keep clause in `garbageCollection` is
untouched; residency sits beside it on its own key.

---

## 2. WHAT WAS MEASURED, AND IT TOOK FOUR CORRECTIONS TO GET RIGHT

The headline result is sound. The route to it was not, and the corrections are
the most valuable thing in this file — **every one of them produced a plausible,
confident, wrong number first.**

### 2.1 SETTLED

- **Identity is stable.** `IdentHead` (vbPtr, ibPtr, vsIl, offsets, stride, draw
  range) reads `newKeys=0` on every shader, every window, camera moving. This was
  the gate on the whole architecture and it passes.
- **Geometry never moves.** `dirty{vb=0 ib=0}` on every non-fanout shader, every
  capture, regardless of camera state. BLAS/geometry residency is safe on the
  cheap `GetMapGeneration()` test.
- **The derived `objectToWorld` is camera-invariant and stable:**
  `o2w{n=1101 cleanPct=100 moved=100 nMoved=1078}` on the dominant shader —
  100% clean across 1078 camera-MOVING judgements. **No epsilon needed**; an
  exact byte compare reads 100%.
- **The fanout shaders are genuinely dynamic** (`dirty{ib=20}` / `dirty{vb=10}`)
  and are NOT residency candidates. They already have their own mechanism:
  `pushInstanceRecords` / `FanoutBatchRecord`.
- **Bootstrap set is small:** `modelsLive` 343-392 against ~520 draws/frame.
  Seeding means having drawn a few hundred geometries, not a level of objects.
- **No shadow-pass feed:** `depthPct=0`. (Caveat: TF2 may bind a colour RTV on
  shadow passes, so this could be the instrument. Unverified.)

### 2.2 THE FOUR CORRECTIONS — READ THESE BEFORE TRUSTING ANY NUMBER

**(a) `GetMapGeneration()` is useless for the constants.** The plan named it
"a proof, not a heuristic". True for vb/ib, false for cbuffers: TF2 writes
per-draw constants into shared scratch buffers renamed every frame regardless of
value, so generation reads dirty on 100% of draws. Had the gate shipped on it,
`hit` would have read ~0% and the conclusion would have been "residency does not
work for TF2". **It does. The sensor did not.**

**(b) The content hash of input bytes is a CAMERA key.** `cleanContentPct` read
86-100% across four captures. All of them were standing still. Joining
`[SceneCensus]` against `[Cam.snapshot]` showed the camera drifting **0.3 world
units over 40 frames** took content dirt from 6% to 100%, collapsing back the
instant `Main=` froze. Sub-unit camera jitter cannot correlate with real object
motion — TF2's object constants are camera-relative, so no slot mask saves the
input-byte approach. **Third time this tree has hit that wall** (see the o2w
object-key note at `d3d11_rtx.cpp:29790`).

**(c) Two counting bugs in the census itself.** `ve.instances` was tallied below
a once-per-frame early return, so every placement past the first went uncounted
and `fanout` collapsed to 1 on shaders measured at 24 the run before. And
`newKeys` counted "not seen last frame" rather than "never seen", jumping 0 -> 45
on a provably stable key. Opposite findings sharing one counter.

**(d) The contiguity filter excluded the population residency exists for.**
Requiring `gapLen == 1` produced `judged=0` on every shader that submits on
ALTERNATE frames (`gapMean=2`), ~900 draws/window. Those are precisely the
objects `numFramesToKeepInstances=1` retires. It was also wrong on its own terms:
residency asks "is this the same as when I last saw it", not "did it change in
one frame". Gap length is a dimension of the answer, not a precondition.

**Also fixed:** `class{}` buckets were lifetime-cumulative, so "static" meant
"never dirty since the session began" and decayed to zero as any session ran
(read 41 one capture, 0 the next, same scene). Now windowed.

---

## 3. OPEN — IN ORDER

1. **`o2w{n=0}` on four shaders: 0x29a8a769 (351 draws), 0x2966cb89 (240),
   0x28674497 (230), 0x29dbd7a3 (144).** ~965 draws, ~22% of the frame, that the
   gate would silently never cover. `arrived=`/`noStash=` columns were added to
   split "never reaches SubmitDrawTail" from "reaches it with an empty stash" —
   **built, not yet run.** `rtx.debug.traceVertexShaders = 0x29A8A7698EEBE180` is
   armed for this (MeshTrace; queue the other three one at a time).
   *A draw class the gate never sees is a draw class residency does not cover.*
   Settle before arming anything.

2. **The stall.** Multi-second freezes, recurring, sometimes forcing a restart.
   `[Perf.Gap] maxAfter=GetData afterGetData=12028ms/5`, `[Perf.SyncSite]
   bursts=0` through every freeze, `gpuIdleMs=12026`, scene populated. The game
   is NOT spinning in GetData — it left and did not come back, GPU idle. Every
   in-DLL probe is blind during exactly that window. `[StallWatch]` built to
   sample the stalled thread from outside; **no data yet.**
   NOT established: whether the census contributed. It is unconditional right
   now, so gating it and running once with it off isolates that in one sitting.

3. **Write the gate (slice 5 proper) + slice 6.** Gate goes at the END of
   `SubmitDrawTail` — `objectToWorld` only exists after `ExtractTransforms`, so
   it CANNOT sit ahead of the derivation as the plan originally assumed. It saves
   capture, material resolution, geometry copy, instance resolution and commit;
   not the derivation. Softened by the derivation already being ~76% cache-served
   (`xfServeN=80829` vs `xfDeriveN=25229`).

4. **sceneCull is inert and it is the GPU half.** `culled=0` every frame:
   `lights=0` -> `lightAllKeep` -> the light keep covers all ~1300 off-screen
   instances. The code names its own lever: *"the perf lever on such maps is the
   solid-angle reject"*, and `solidAngleMin` sits at its most conservative
   default rejecting 11 of ~1298. **Do not sweep it until residency puts the
   off-screen set back** — there is nothing off-screen to cull today.

5. **Off-screen movers** (plan §6). Only genuinely needs the engine side. Do not
   start until it is visible on screen; the per-class option (leave entities
   unculled, cull props/world) may remove the need entirely.

---

## 4. TRAPS

**Instrument traps, all paid for this session:**

- A number that is stable across many captures is not thereby trustworthy —
  `cleanContentPct` held 86-100% across four runs and was a standing-still
  artefact the whole time.
- **Cross-run comparison is invalid here.** Same scene + restart gives materially
  different performance. A/B inside one sitting or it does not count, and state
  the camera state next to any number.
- **Silence from a probe is not evidence until the probe is confirmed armed.**
  MeshTrace read zero; only checking the log for the parsed option line proved it
  had actually armed.
- **Check for an existing probe before writing one.** Three times this session
  the data already existed: `[Cam.snapshot]` (camera motion), `[ReapJoin]` (the
  retirement gate), `[MeshTrace]` (exit attribution). Two rebuilds were wasted.

**Code traps, from the tree's own history:**

- `setFrameLastUpdated()` CLEARS `m_seenCameraTypes`. A skip path must
  re-register the camera or portal/view-model logic silently loses its set.
- Several draws per frame share one (vs, geometry) identity and resolution is
  stateful within a frame — `pushInstanceRecords` verify read FAIL=3367 for this.
  The key needs a per-frame occurrence ordinal.
- **A long keep on an unstable key makes things WORSE.**
  `rtx_instance_manager.cpp` `[PropIdKeepLong attempt reverted]`:
  `m_reorderedSurfaces` doubled 8500 -> 17155. **Stability first, keep second.**
- Resident != refreshed. The s2s "two views" bug was live instances that draws
  failed to refresh — a silent correctness bug with no FAIL to catch it.
- Eviction: use the `[Perf.SplitXf]` age-rung ladder (`{300,60,8,2}`, sweep only
  when over cap, wipe as last resort, counters CUMULATIVE). A single fixed age
  bound cannot fire at a high fill rate — that sweep freed nothing for weeks and
  read as working because it had no counter.

---

## 5. CONFIG STATE (rtx.conf)

```
rtx.cullOff.enable              = False    engine culls normally (master off = stock)
rtx.numFramesToKeepInstances    = 1        pinned explicitly; DO NOT RAISE until §3.1 green
rtx.residentScene.enable        = False    inert
rtx.residentScene.verify        = True     scores, acts on nothing
rtx.residentScene.logStats      = False
rtx.residentScene.seedFrames    = 0
rtx.sceneCull.enable            = True     built, culling 0 (see §3.4)
rtx.debug.traceVertexShaders    = 0x29A8A7698EEBE180   armed for §3.1
rtx.logSubmitStall              = False    reverted; re-arm alone for the entry-point split
rtx.findSimilarProbeVsHashes    = disarmed (was costing ~4.5 ms via [DrawName])
```

Backups: `rtx.conf.bak-pre-pull-arch`, `rtx.conf.bak-pre-meshtrace`.

**Arming order, non-negotiable:** settle §3.1 -> write the gate -> `enable=True`
with `verify=True` -> read `[ResidentGate]`/`[ResidentScene]` across a full
pitch-and-yaw sweep -> only with `newKeys=0`, records plateaued and
`touchMiss ~0`, set `verify=False`.

## 6. THE NUMBER

Not the 8.20 ms of frame-thread slack — that measures additional headroom on top
of a frame that is currently lossy. The right comparison is between the two ways
to have a **correct** scene:

```
culling OFF              57.99 ms    correct scene
culling ON + residency   ~20 ms      correct scene
```

**~38 ms**, plus most of ~17 us/draw on the ~90% of draws the gate can take —
the gate itself priced at 1-3 us, not the ~50 ns first claimed, because the
transform test compares a derived matrix rather than a generation counter.
