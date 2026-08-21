# THE RESIDENT SCENE

**Written 2026-08-20.** Supersedes OPTIMISATION_PLAN_3 §3 (its Phase 0 is closed:
engine culling is back ON, `cullOff.enable=False`, 57.99 → 19.97 ms, 17.2 → 50.1 fps,
drawsIn 1338 → 538, scene healthy at inst≈1670 / uniqueBlas≈217).

> **IMPLEMENTED 2026-08-21. Slices 1-8 are in the tree, unbuilt and unrun.**
>
> This document keeps its original text. Where the implementation contradicted
> it, the claim is left standing with a correction block underneath, the same way
> §1.2 already handles its own two reversals — a plan that quietly rewrites its
> refuted parts teaches the next reader nothing about which kinds of claim in it
> are load-bearing and which were guesses.
>
> **What changed most, and it is the shape of the whole thing:** the gate reaches
> its verdict on the frame thread, but the SKIP IS TAKEN ON THE CS THREAD, at the
> top of `RtxContext::commitGeometryToRT` immediately ahead of `submitDrawState`.
> §2.1's flow diagram assumes the frame thread collapses the payload. It cannot:
> the evidence that makes a skip safe is whether a resident record actually
> exists, and only the CS side can see that. See the correction under §2.1.
>
> **Two death signals this document did not have, both of which would have made
> residency do nothing or do harm:** the BLAS outliving the instance (§2.3), and
> the per-frame work that hangs off an instance rather than off its geometry,
> transform or material (§2.2). Both are now closed and both are new traps (§3).
>
> **And a prerequisite this document does not mention at all:** buffer identity
> had to become stable first, or every gate hit minted a stale bindless index.
> See §0.2, added.

**NO ANTI-CULLING.** `rtx.antiCulling.*`, `AntiCulling::isObjectAntiCullingEnabled`
and `InstanceCategories::IgnoreAntiCulling` are OUT OF SCOPE and must not be read,
written, or extended by anything here. The existing long-keep gate at
rtx_instance_manager.cpp:2985 is keyed on `IgnoreAntiCulling`; this work does NOT
extend it. Residency gets its own key and its own clause.

---

## 0. THE INVERSION

Today the draw stream **is** the scene. Remix rebuilds the world from 538 draws
every frame and forgets whatever did not arrive.

Target: the scene is a **persistent database**; a draw is demoted to a
liveness-and-update signal. A draw no longer says *"here is an object"*, it says
*"an object you already have may have changed"*. **Absence of a draw says nothing.**

Three mechanisms, three different jobs, composing:

```
engine cull   object never reaches D3D11             saves CPU  (entry point + engine gap)   DONE
residency     object survives not being drawn        keeps RT correctness                    THIS
sceneCull     object resident but masked out of TLAS saves GPU  (BVH + traversal)            EXISTS, inert
```

**Status 2026-08-21.** Residency is BUILT and arms behind two switches, in this
order and no other: `rtx.residentScene.enable=True` with `verify=True` scores the
prediction and acts on nothing; `verify=False` takes the skip and the extended
keep. sceneCull is still inert and still waiting on residency to put the
off-screen set back before its `solidAngleMin` sweep means anything.

The claim: every object present as if culling were off, paying neither cost.

## 0.0 WHERE THE LIST COMES FROM — READ THIS BEFORE ANYTHING ELSE

The line above says *"engine cull → object never reaches D3D11"*, and that cuts both
ways. **You cannot raytrace geometry you have never received.** A culled object issues
no `DrawIndexed`, so Remix has no vertices, no material and no transform for it —
there is nothing to make resident. Residency does require a list of objects, and that
list can only ever be built out of draws that actually happened.

So the resident set is, by construction, **the union of everything visible since level
load**. Left to accumulate on its own that is a warm-up bug, not a scene: geometry you
have never faced casts no shadow and appears in no reflection until you look at it
once. It is also view-HISTORY-dependent, which fails §5's own invariance gate — sweep
pitch and yaw and the count grows instead of staying flat.

**SEED IT. The seeder is already built and just got proven.** `cullOff` stops being a
steady state and becomes a population pass:

```
level load ─► cullOff.enable = True for N frames     engine culls defeated, whole level submits
              └─ harvest every draw into resident records      (~58 ms/frame, a few frames)
           ─► cullOff.enable = False                 engine culls normally, ~538 draws
              └─ coast on the resident set                     (~20 ms/frame, indefinitely)
```

The 58 ms is paid for a handful of frames at load instead of every frame for the whole
session. This is the ONLY correct use of the cullOff patches once residency exists.

**Two gaps seeding does not close.** Both are real, both must be designed for, neither
is a reason to delay slices 1-6:

1. **Streaming.** Geometry that loads after the seed pass has never been drawn and is
   invisible to residency. Needs a re-seed trigger — a rolling budget (culls off for K
   objects/frame) or a hook on the streaming event. Until then, newly-streamed geometry
   behaves exactly as it does today.
2. **Off-screen movers.** An entity that walks behind you keeps updating its transform
   in the engine, but no draw arrives, so its resident record holds the pose it had
   when last seen and its shadow goes stale. Buffer generations can detect THAT it
   changed; they cannot supply the new matrix, because the matrix arrives in the draw.
   **This is the one place an engine-side hook is genuinely load-bearing** rather than a
   convenience, and it applies only to the Dynamic class — never to world or static
   props, which cannot move. Deferred to slice 9; see §6.

> **IMPLEMENTED 2026-08-21, and the trigger is a scene EPOCH rather than a
> level-load hook.** `SceneManager::clear()` is the one point every scene reset
> funnels through — a level change reaches it via the camera-cut delayed clear,
> a replacement reload reaches it directly — so it publishes `g_sceneEpoch`, and
> the frame thread arms the seed when the epoch differs from the one it last
> seeded. An epoch rather than a flag because a clear that happens while the
> frame thread is not looking cannot then be missed, and two clears in quick
> succession arm the seed exactly once.
>
> Two details this section did not anticipate:
>
> **It has to be gameplay-gated.** The first epoch is the process starting, not a
> level arriving. A seed spent on menu and loading frames harvests nothing and
> then does not re-arm until the next clear, which is worse than not seeding.
> Same floor the census and the scene-clear probes already use.
>
> **It flips the cullOff MASTER switch only.** The individual patch groups keep
> whatever the conf says, so the seed frames reproduce exactly the configuration
> `rtx.cullOff.enable=True` was measured at 57.99 ms on. Forcing every group on
> would be a wider configuration nobody has run — `pvs` in particular is
> documented as much more expensive — and the seed is not the place to find that
> out. The A/B override still wins: `abMode==2` means the user is deliberately
> restoring engine culling to compare, and a seed that ignored it would silently
> invalidate the comparison.

## 0.1 What it is worth

Not the 8.20 ms of frame-thread slack. That measures *additional* headroom on top
of today's frame, which is the wrong question. The right comparison is between the
two ways to have a **correct** scene:

```
culling OFF                57.99 ms      correct scene
culling ON + residency     ~20 ms        correct scene, and falling
```

**≈38 ms**, plus whatever the pull gate takes off the remaining 17 us/draw -- and
since 1.2 the gate is priced at 1-3 us rather than ~50 ns, because the transform
test has to compare CONTENT and not just a generation counter. Take the second
term as "most of 17 us on ~90% of draws", not "all of it on all of them". Today's
19.97 ms is not a banked baseline — it is a loan against missing geometry, and
`[ReapJoin] respawn=0 starved=1..22/frame` is the interest being paid.

> **CORRECTED 2026-08-21 BY THE IMPLEMENTATION: the second term is not the
> 17 us/draw on the frame thread.** That was the right number for a gate that
> collapses the payload on the frame thread, and the skip does not live there.
>
> Where the money is, from this tree's own measurements on one window:
>
> ```
> [Perf.Busy]     wallMs 73.6                the frame
> [Perf.CsSplit]  dxvk-cs exec 73.3          99.6% -- dxvk-cs IS the frame
> [Perf.CsCmd]    commitGeometryToRT 53.1 ms/f over ~1100 calls
> [CommitRT]      submitMs 50 of perFrameMs 52
> ```
>
> So the term to remove is `submitDrawState`, ~50 of `commitGeometryToRT`'s ~52,
> and that is exactly what a hit removes. The ~2 ms above it — camera
> classification, sky handling, transform fixups — still runs, deliberately
> (§2.1). The frame thread keeps its ~17 us/draw.
>
> **That is not a loss, it is the correct order.** dxvk-cs is 99.6% of wall time;
> the frame thread has 8.20 ms of slack that §0.1 opens by dismissing. Cutting CS
> first and measuring what becomes the ceiling afterwards is the same
> "verify first, skip second" discipline this document applies everywhere else.
> If the frame thread does become the ceiling, moving the gate earlier is a
> separate change with its own evidence — and it will need one, because a gate
> that runs before `captureDrawSnapshot` has to be able to un-skip a draw whose
> transform turns out to have moved.

## 0.2 THE PREREQUISITE THIS PLAN DOES NOT MENTION, AND IT BLOCKED EVERYTHING

**Added 2026-08-21.** Bindless BUFFER identity was not stable, and every gate hit
was by construction a buffer that had not been re-tracked that frame.

`m_bufferCache` was an append-only tape cleared at `SceneManager::onFrameEnd`
every frame and refilled only by geometry PROCESSED THIS FRAME. A slot index was
therefore a position in one frame's tape and meant nothing in the next one. An
instance kept alive and not redrawn — which is the entire population this
document exists to create — re-uploaded its surface every frame carrying indices
from an older and possibly longer tape. Measured before the fix: 1032 stale
surface uploads over 70 frames, worst case 691 slots past the end of the tape.
Past the end is an undefined descriptor in an array declared with
`VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT`, which the hardware follows as an
arbitrary address; inside the tape but from an earlier generation it is simply
the wrong buffer.

So residency would have made the device-loss chain WORSE in exact proportion to
how well it worked: the gate's hit rate and the stale-index count are the same
number counted twice.

The table is now a `BufferSlotTable` with stable slots, freed on evidence rather
than on frame age — `retire()` says the owner let go, and `reclaim()` frees only
after the surface upload has reported no reference for longer than the pipeline
depth. That last part is what makes it correct HERE specifically: recycling a
slot on frame age would eventually hand a live wrong buffer to an instance that
outlived the guess, and residency's whole premise is that an instance's age is
not a bound on its life.

`[StaleTape] staleSurfaces = 0` across a level load and a stationary capture is
the gate for this, and it is a prerequisite for reading anything else here.

---

## 1. THE FOUR PARTS, AND THREE OF THEM ALREADY EXIST

### 1.1 Identity — `IdentHead`, not `stablePropId` `[M]`

d3d11_rtx.cpp:29815. 56 bytes: `vbPtr, ibPtr, vsIl, vbOffset, vbStride, ibOffset,
drawStart, drawCount, drawBase`. Camera-invariant by construction — engine
allocations reused for the entity's lifetime. Available off live `m_context->m_state`
**before any snapshot is built**, which is what lets the gate run ahead of
`captureDrawSnapshot`. Measured:

```
46-93 new keys/frame vs 933-959 for the byte key, noProp=0
per path new/f:   p1 = 0    p3 = 0    p5 = 0    p13 = 56
```

Zero new keys per frame on world geometry, camera moving.

**`stablePropId` is the wrong field and the code already says so** (d3d11_rtx.cpp:42281):
keying on it read `noProp=100%`, 27274 of 27274 draws, 57 windows — `ExtractTransforms`
runs before any producer writes it, and its producers only ever cover bone-anim and
sky. Nothing writes one for p1 or p13, which are the bulk of the frame.

### 1.2 Dirty — TWO tests, not one `[M]`

> **CORRECTED 2026-08-20 by measurement.** An earlier version of this section
> named `GetMapGeneration()` as *the* dirty test and called it "a proof, not a
> heuristic". That is true for the vertex and index buffers and **false for the
> constants**, and the difference is the whole feature. Had the gate shipped on
> it, `[ResidentGate] hit` would have read ~0% and the obvious-looking
> conclusion would have been "residency does not work for TF2". It does. The
> sensor did not. Keep this correction: the failure is invisible from the code
> and only shows in a capture.

**What the capture says** (`[VsResidency]`, gameplay, ~522 draws/frame):

```
vs=0x2af9b90d draws=1206 judged=975  cleanPct=0  cleanObjPct=0  cleanContentPct=90
              dirty{vb=0 ib=0 cbGen=975 cbContent=975}
              cbMoved=0xe   cbContentMoved=0xc
```

- `vb=0 ib=0` — **vertex and index buffers never move.** Geometry is genuinely
  static, on the cheap test, for the dominant shaders.
- `cbGen=975` of 975 — **constant-buffer generations move on 100% of draws.**
- `cleanContentPct=90` — but the BYTES THIS DRAW READS are unchanged on 90%.

TF2 writes per-draw constants into shared dynamic scratch buffers that are
renamed every frame whether or not the values changed. `GetMapGeneration()`
answers *"was this buffer written"*, and for shared per-draw scratch the answer
is always yes — so it cannot decide whether **this object** moved. That is the
"one pooled buffer backing many objects → false dirty" fallback condition this
section used to list as a risk, now measured as the actual state.

Same trap as the o2w object key at d3d11_rtx.cpp:29790 — *"the bytes track the
camera while the matrix they produce does not… ~97% of all churn was a key
chasing something its own value did not do."* Different mechanism, identical
shape. Third time in this tree.

**So: two tests, matched to what each is good for and what each costs.**

| what | test | cost | measured |
|---|---|---|---|
| geometry — BLAS, vertex/index data | `GetMapGeneration()` | ~free, a few atomic loads | never moves |
| transform | **hash of the DERIVED `objectToWorld`** | must run after ExtractTransforms | **100% clean under camera motion** |
| material | fold over PS + its SRVs, samplers, blend/depth/raster state | ~16 cache lines, engine pointers | ADDED 2026-08-21, unmeasured |

> **A THIRD TEST, ADDED BY THE IMPLEMENTATION, AND IT IS NOT OPTIONAL.** This
> section reasons entirely about geometry and transform, and a record does not
> hold either of those — it holds RtInstances, and an instance carries a surface
> MATERIAL as well. A draw whose vertex buffers and object-to-world are both
> unchanged can still be asking for a different texture or a different blend
> mode, and a two-test gate would skip it and leave the old material resident
> with nothing to catch it. That is trap 4's shape exactly: alive, and not
> refreshed.
>
> What identifies a material here is whatever `FillMaterialData` reads: the pixel
> shader, the resources it samples, the samplers, and the blend / depth-stencil /
> rasterizer state objects. All engine objects created once and reused for the
> material's lifetime — so unlike the constant buffers they do NOT churn per
> frame, and they are the same kind of input as the IA pointers §1.1's key is
> made of, stable for the same reason.
>
> **The whole 128-slot shader-resource array, not the handful of slots TF2 is
> known to use.** This is the place the superset argument actually holds: it can
> only make the gate refuse a serve it could have made, where narrowing to "the
> slots the material path looks at" is a guess about another module's behaviour
> and being wrong there means serving a stale material. Sixteen contiguous cache
> lines against the ~48 us of CS-side commit a hit removes.
>
> `[ResidentGate] missMat` is what says whether that was the wrong trade: high
> means the game rebinds material state per draw and the fold wants narrowing to
> what `FillMaterialData` actually reads — a measurement, not a guess.

> **CORRECTED AGAIN 2026-08-20, and this is the version that survived.** The row
> above used to say "content hash of the draw's own cbuffer window, 90%
> unchanged". That number was measured standing still. Joining `[SceneCensus]`
> against the ALREADY-EXISTING `[Cam.snapshot]` showed the camera drifting 0.3
> world units over 40 frames took content dirt from 6% to 100%, and the instant
> `Main=` froze it collapsed back. Sub-unit camera jitter cannot correlate with
> real object motion: **TF2's object constants are camera-relative, so hashing
> their bytes is a camera key however the slots are masked.** Third time this
> tree has hit that wall — see the o2w object-key note.
>
> Keying on the DERIVED matrix fixes it outright, measured in one sitting with
> the camera moving for essentially the whole capture:
>
> ```
>                       input bytes        derived objectToWorld
> vs=0x2af9b90d 1362dr  cleanContent=86    o2w n=1101 clean=100 moved=100 (nMoved=1078)
> vs=0x292b6ba0  297dr  cleanContent=96    o2w n=297  clean=100 moved=100
> vs=0x29aa0345  176dr  cleanContent=89    o2w n=157  clean=100 moved=100
> vs=0x29a262d2  198dr  cleanContent=50    o2w n=18   clean=100 moved=100
> vs=0x298e12b3  659dr  cleanContent=90    o2w n=549  clean=90  moved=90
> ```
>
> **No epsilon was needed** — an exact byte compare reads 100%. The
> rtx_types.h "slightly different matrices each frame for the same static prop"
> jitter does not reach here; the derivation or the split-transform cache
> already absorbs it. Do NOT add a tolerance: leading with one would blur
> "jittering" and "moving", which is the distinction this test exists to make.

**WHERE THE GATE THEREFORE LIVES.** `objectToWorld` only exists after
`ExtractTransforms`, so the gate CANNOT sit ahead of the derivation as §0.1
originally assumed. It sits at the end of `SubmitDrawTail`, where o2w is final
for every path (9-12 assign it well after ExtractTransforms) and the frame
thread still owns the draw. It therefore saves capture, material resolution,
geometry copy, instance resolution and commit — but NOT the derivation itself.
Mitigating that: the derivation is already ~76% cache-served
(`xfServeN=80829` against `xfDeriveN=25229`), so the un-skippable remainder is
small.

**OPEN, and it is 22% of the frame.** Four shaders report `o2w{n=0}` —
0x29a8a769 (351 draws), 0x2966cb89 (240), 0x28674497 (230), 0x29dbd7a3 (144).
Their draws are judged at the head of SubmitDraw but never reach the tail, so
they are filtered before the transform is finalized. Whether they are
RT-captured at all is UNKNOWN and must be settled before the gate is armed: a
draw class the gate never sees is a draw class residency silently does not
cover.

Geometry is the expensive thing to re-cache and it is provably static on the
cheap test. Only the transform needs bytes — and only its own window, taken from
`constantOffset`/`constantBound`, which is what makes the test per-object rather
than per-buffer even out of a shared allocation.

**The camera slot is excluded, and which slot that is was measured, not assumed.**
`cbMoved=0xe` (slots 1,2,3) against `cbContentMoved=0xc` (slots 2,3): slot 1's
generation moves every frame while its content never does — pure false dirty.
Slot 2 is the camera, proven by shaders whose only moving content is slot 2
(`cbContentMoved=0x4`) reading `cleanContentPct=100`.

**What this costs, stated plainly.** The generation-only gate was ~50 ns. A
content gate reads and hashes up to 256 B/slot out of write-combined memory, and
this tree measures the unbounded version of that at ~31% of `SubmitDraw`. So the
gate is realistically 1–3 µs against the ~17 µs it replaces — still most of the
win, but §0.1's arithmetic is optimistic by that much and should be read with
this number in hand.

**What survives unchanged from the original claim:** the test still does not need
the draw for the GEOMETRY half, so a resident record whose draw never arrived can
still be proven geometrically clean. The transform half does need the draw, which
is §6's off-screen-mover problem and is unaffected either way.

IDA remains unnecessary for all of this.

### 1.3 Keep-alive without reprocessing — already solved `[M]`

rtx_instance_manager.h:768, the `pushInstanceRecords` contract:

> *"garbageCollection reaps on `m_frameLastUpdated + instanceKeepN <= currentFrame`
> and nothing else. So 'skip' must mean KEEP ALIVE WITHOUT REPROCESSING, and that is
> exactly a frame-id stamp."*

`FanoutBatchRecord` is the ResidentRecord shape already working for the fanout path,
with the lifetime contract solved: `RtInstance::m_batchRecordKey` is a back-pointer
making `removeInstance` invalidation O(1) and **total** — an instance cannot be
destroyed without its record being invalidated in the same call. We copy that pattern
exactly, under a second key.

### 1.4 Visibility — `sceneCull`, built and currently inert `[M]`

Union of keeps (frustum | radius | lightInfluence), one reject (solidAngle), culling
by zeroing the TLAS instance mask in `mergeInstancesIntoBlas` so BLAS and geometry
stay resident and nothing rebuilds when an object returns. Its doc string already
says *"Intended to replace the engine culling disabled by rtx.cullOff.*"*.

Live reading, every frame:

```
[SceneCull] tested=1625 keptFrustum=327 keptLight=1298 culled=0 culledSmall=11
            lights=0 lightAllKeep=1
```

**`culled=0`.** Not a bug and not bad code — `lights=0`, so `lightAllKeep` fires
(rtx_accel_manager.cpp:1346) and the light keep covers all 1298 off-screen instances,
because dome light arrives from every direction and no extrusion can bound its
occluders. The code names the consequence itself: *"The perf lever on such maps is the
solid-angle reject below, not this term."* And `solidAngleMin` sits at its most
conservative default, rejecting 11 of ~1298.

**Do not sweep it yet.** There is nothing off-screen worth culling today — the engine
already deleted it. The sweep only becomes meaningful once residency puts the
off-screen set back, and it is then the whole GPU-side lever.

---

## 2. ARCHITECTURE

```
ResidentRecord {
  key          uint64      // XXH64(IdentHead) + occurrence ordinal
  srcGenHash   uint64      // XXH64 over map generations of vb0 + ib + bound VS cbs
  frameLastSeen, frameLastBuilt
  valid                    // cleared by removeInstance / BLAS teardown
  instances    vector<RtInstance*>   // raw, guarded by m_residentKey back-pointer
}
```

`surfaceSlot` is not a field — it is held implicitly, and that is the point: the
instance is never retired, so its ordered-surface slot is never returned and never
reallocated. Hazard 1 of OPTIMISATION_PLAN_3 §4 dies by construction.

### 2.1 Per-frame flow

```
FRAME
 │
 ├─ FRAME THREAD, per draw (538)
 │    SubmitDraw head, off live m_context->m_state, NO snapshot built:
 │      k    = residentKey()      ~50 ns
 │      gens = residentSrcGens()  ~7 atomic loads
 │      ├─ record hit && gens match  → emit TouchRecord{k}, return   ← THE GATE
 │      └─ otherwise                 → full path (snapshot, extract, commit)
 │
 ├─ CS THREAD
 │    TouchRecord{k} → bulk-stamp m_frameLastUpdated on record.instances
 │                     + re-register the draw's camera  (see trap 1)
 │
 ├─ RESIDENT SWEEP, once per frame, over records NO draw touched
 │    Static  → nothing; the transform cannot be stale
 │    Dynamic → gens match? hold last o2w : mark for re-derive
 │    ** NO RETIREMENT HERE **
 │
 ├─ VISIBILITY — sceneCull over the FULL resident set, not the drawn set
 │
 └─ TLAS over  resident ∩ visible
```

The frame thread cannot touch `RtInstance` — instances are CS/RT-owned. So the gate
does not stamp; it collapses the payload from a full `DrawCallState` to an 8-byte
`TouchRecord` in the existing `EmitCs` stream, and the stamp happens on the CS thread
where it is already legal. Threading model unchanged.

### 2.2 Classes

| class | examples | resident | dirty test | cost/frame |
|---|---|---|---|---|
| **Static** | BSP world, static props | forever | never (gens pinned at 0) | gate hit |
| **Dynamic** | entities, animated props | forever | gen compare | gate hit, or partial re-derive |
| **Transient** | particles, UI, viewmodel | no | n/a | full path, exactly as today |

Static needs no dirty logic at all. Transient keeps today's behaviour verbatim, so
nothing can regress into it.

### 2.3 Death — the only real retirement signals

Records do not die of frame age:

- **source buffer destroyed** — the engine freeing the VB/IB *is* the object ceasing
  to exist, and DXVK observes it directly
- **level / map change** — bulk clear
- **memory pressure** — LRU by `frameLastSeen`

`numFramesToKeepInstances` stops being a lifetime and becomes a Transient-only fallback.

---

## 3. TRAPS — ALL FOUR ALREADY PAID FOR IN THIS TREE

1. **`setFrameLastUpdated()` clears `m_seenCameraTypes`** (rtx_instance_manager.h:772).
   A skip path that does not re-register the draw's camera silently loses the camera
   set and breaks portal / view-model logic.
2. **Several draws per frame share one (vs, geometry) identity, and resolution is
   stateful within a frame** — the `pushInstanceRecords` verify run read FAIL=3367 for
   exactly this. Draw 1 stores, draw 2 matches the fingerprint but resolves to
   different instances because draw 1's are already claimed. **The key needs a
   per-frame occurrence ordinal.**
3. **Long keep on an unstable key makes things WORSE, not better**
   (rtx_instance_manager.cpp:2966, `[PropIdKeepLong attempt reverted]`): dedup missed
   every frame, new instances were created and then all kept alive,
   `m_reorderedSurfaces` doubled 8500 → 17155, and every collapse brought down twice
   the stale pixels. **Stability first, keep second. Not negotiable.**
4. **Resident ≠ refreshed.** The `s2s two views` bug was intro-era instances staying
   alive while steady-state draws failed to refresh them — two different worlds on
   screen, with no FAIL to catch it. A record that a draw *should* have updated and
   did not is a silent correctness bug.

**And the order of operations, which this tree has now used three times (split-transform
cache, extract memo, pushInstanceRecords):** verify first, skip second.
`rtx.residentScene.verify` defaults ON, runs both paths, scores the prediction, and
acts on nothing. **Nothing is skipped until FAIL reads 0.**

---

## 4. BUILD ORDER

| # | slice | files | gate |
|---|---|---|---|
| 1 | options (ONE batched pass) | `rtx_options.h` | — |
| 2 | ResidentScene map + key + dirty fold | `rtx_resident_scene.{h,cpp}` (new), `meson.build` | — |
| 3 | lifetime contract: `m_residentKey` + O(1) total invalidation | `rtx_instance_manager.{h,cpp}` | — |
| 4 | GC keep clause consulting residency (NOT `IgnoreAntiCulling`) | `rtx_instance_manager.cpp` | — |
| 5 | frame-thread key + **vb/ib gen fold AND non-camera cbuffer CONTENT hash** + verify scoring (see 1.2 -- generations alone read dirty on 100% of draws) | `d3d11_rtx.{h,cpp}` | verify FAIL=0 |
| 6 | `TouchRecord` payload + CS-side bulk stamp | `d3d11_rtx.cpp`, `rtx_scene_manager.cpp` | slice 5 green |
| 7 | flip `verify` off, raise the keep | conf | slices 1-6 green |
| 8 | **seed pass** — drive `cullOff` for N frames at level load, harvest, release (§0.0) | `d3d11_rtx.cpp` | slice 7 green |
| 9 | off-screen mover transforms — engine-side (§6) | TBD | slice 8 green, and only if it shows on screen |
| 10 | sceneCull `solidAngleMin` sweep — the GPU half | conf | residency landed |

`rtx_options.h` is touched **once**, in slice 1, with every option this work will
ever need. It is included nearly everywhere and any touch is a ~20 minute rebuild.

## 5. ACCEPTANCE, AND IT IS ALREADY INSTRUMENTED

```
[ReapJoin] removed=1-22/frame  respawn=0  starved=ALL   ← today: the engine cull eating the RT scene
[ReapJoin] starved -> ~0                                 ← residency landed
```

`starved` is this tree's own definition of *"the geometry received fewer draws than it
had instances — a submission gap upstream of Remix"*, and `respawn=0` proves it is not
a dedup failure. Free gate, nothing to write.

Plus, as hard gates:

- resident instance count **flat** over a 5-minute stationary capture (a rising count
  is trap 3 recurring and must fail the build)
- `m_reorderedSurfaces` flat with it
- invariance, not an average: fix the camera, sweep all pitch and yaw, require the
  count to be **flat**. A good mean over a sweep hides the band where it drops.

---

## 6. THE OFF-SCREEN MOVER, AND WHY IT IS LAST

Slices 1-8 make world geometry and static props correct, and those cannot move —
"static" is the engine's own word for them, not an inference. The residual defect is
narrow and it is worth stating exactly so nobody widens it:

> A **Dynamic**-class entity that changes pose while the engine is culling it holds the
> pose it had when last drawn. Its shadow and its reflection go stale until it is
> visible again.

Buffer generations detect that its source changed; they cannot supply the new matrix,
because the matrix only ever arrives in a draw. So this one genuinely needs the engine
side, and it is the ONLY thing here that does.

**Do not start it until it is visible on screen.** Three reasons:

1. The visible error is bounded by how long an entity stays culled and by how fast it
   moves. A stale shadow for an object nobody is looking at may simply not be
   perceptible, and that is a measurement, not a guess — take it.
2. Every prior attempt in this tree to reason about entity transforms ahead of a
   measurement got the layer wrong: the vanishing floor was a studiorender model, not
   world geometry, and invalidated the whole world-geometry investigation stack.
3. The order this tree's rules prescribe is `RtlCaptureStackBackTrace` at a
   dxvk-controlled site FIRST to find who writes an entity's transform, IDA second.
   `CULLING_BIBLE.md` §5 already maps renderable → model name, §5.1 the
   `IClientRenderable` vtable and §5.2 the `C_BaseEntity` / `C_BaseAnimating`
   subobjects, so much of the reversing is done.

The cheap alternative, if it does show: keep Dynamic-class entities OUT of residency
entirely and let them retire at `numFramesToKeepInstances=1` exactly as today. They are
the minority of the draw stream, they are the class the engine re-submits every frame
anyway, and the whole prize lives in world + static props. Residency is not all-or-
nothing and the class table in §2.2 exists to make that a one-line change.
