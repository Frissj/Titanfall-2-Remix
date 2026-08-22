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

> **SUPERSEDED AS THE ANSWER, 2026-08-21. THE SEED CANNOT DO WHAT THIS SECTION
> ASKS OF IT, AND THE REASON IS IN THE CONF RATHER THAN THE CODE.**
>
> The section is right that residency needs a list and that the list can only be
> built out of draws that happened. It is wrong that seeding produces one.
>
> The seed flips the `cullOff` MASTER switch only — deliberately, and the block
> above gives the reason. But the individual groups then keep their conf values,
> and in the live conf the culls that actually hide the level are all OFF as
> patches, i.e. LIVE as culls: `pvs` and `visibilityMask` are commented out,
> `distanceFade`, `staticPropFade` and `studioLod` are all `False`. So a seed
> harvests one position's PVS-and-range-filtered slice of the map from all
> angles, not the map. `rtx.conf` names a sixth cull with no flag at all —
> `sub_1802EAD60`'s position-keyed area order list, 30 of 179 areas, *"THE ONLY
> ONE WITH NO FLAG"* — which no seed configuration can defeat.
>
> And the trigger cannot re-arm for it: `g_sceneEpoch` is bumped by
> `SceneManager::clear()`, so walking into a new PVS cluster does not re-seed.
> The resident set therefore still grows with view history, which is §5's
> invariance gate failing by construction — the exact defect seeding was added
> to remove.
>
> **The replacement is §7: take the object list from the engine upstream of
> culling and use THAT as the liveness signal.** Seeding is not deleted — it is
> demoted from "how the population is built" to "how the identity mapping is
> learned once", which is a much smaller job and one it can actually do.

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

> **AS BUILT: 64 bytes, not 56, and the four extra fields are not padding.**
> `residentDrawKey`'s `KeyHead` adds the VS hash, the input-layout pointer, the
> index-buffer format and an indexed flag. Each of them changes what the draw
> MEANS out of the same allocation, which is the same argument this section
> already makes for including the offsets and the stride: TF2 sub-allocates many
> meshes out of one pooled buffer, and two draws that differ only in their layout
> are two objects.
>
> **The occurrence ordinal is applied here rather than left to the caller** —
> trap 2 — so a key is (identity, which occurrence of it this frame). It is
> allocated at the ENTRY half, once per draw, which is what lets the tail half
> judge the same draw the entry half keyed. A draw that returns before the tail
> still consumes its ordinal, so the numbering stays aligned frame to frame as
> long as the draw stream is.

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

> **CORRECTED 2026-08-21. The VERDICT lives at the end of `SubmitDrawTail`; the
> SKIP does not, and the list of what it saves is wrong.**
>
> `captureDrawSnapshot` runs near the head of `SubmitDraw`, ~4,900 lines before
> the tail, so a gate at the tail was never going to save it. The geometry copy
> and the hashes are likewise already paid by then. What the tail could have
> skipped is `SubmitDrawDeferred`, and that is not where the skip went either —
> see the correction under §2.1 for why the decision has to happen on the CS
> thread instead, and §0.1 for what it actually removes.
>
> The paragraph is right about the one thing it was written to establish: the
> gate lives downstream of the derivation and does not save it, and that is
> affordable for the reason given.

**OPEN, and it is 22% of the frame.** Four shaders report `o2w{n=0}` —
0x29a8a769 (351 draws), 0x2966cb89 (240), 0x28674497 (230), 0x29dbd7a3 (144).
Their draws are judged at the head of SubmitDraw but never reach the tail, so
they are filtered before the transform is finalized. Whether they are
RT-captured at all is UNKNOWN and must be settled before the gate is armed: a
draw class the gate never sees is a draw class residency silently does not
cover.

> **NOW MEASURED FROM THE GATE ITSELF, 2026-08-21 — `[ResidentGate] noTail`.**
> The entry half stashes the key; the judge at the tail consumes it. A key still
> in the stash when the NEXT draw arrives means the previous draw never reached
> the judge, and that is counted rather than inferred from a separate probe.
>
> This is strictly better than settling it with `[VsResidency] o2w{n=}`, and the
> reason is the one this section already gives for why it matters: a gate that
> silently never fires for a fifth of the frame reads identically to one that
> fires and hits. `hitPct` is computed over draws that reached the gate, so it
> cannot see the shortfall; `noTail` is the shortfall.
>
> It does NOT answer whether those draws are RT-captured at all — only that
> residency does not cover them. That question is still open and still wants
> `arrived=`/`noStash=`.

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

> **AS BUILT, and the pattern needed one thing the fanout path does not.** The
> stamp itself is exactly as described: `ResidentScene::touch` bulk-writes
> `setFrameLastUpdated` over `record.instances` and replays the camera set
> (trap 1, and see `cameraMask` in §2). The back-pointer contract is copied
> verbatim and it is why this is O(1) and total.
>
> What is new is that keeping the INSTANCE alive turned out not to keep the
> object alive. `touch` also stamps `frameLastTouched` on each instance's BLAS,
> and the BLAS collector separately consults the record store for the population
> that gets no draw at all. §2.3.1 — it is the trap this section's confidence
> would otherwise have walked straight past, because "reaping reads ONLY that
> frame id" is true of the instance reaper and there is a second reaper.

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

**AS BUILT, 2026-08-21.** Four fields the sketch above does not have, and each one
closes a hole that would have been a silent bug rather than a missing feature:

```
ResidentScene::Record {
  srcGenHash        uint64   as above
  frameLastSeen, frameLastBuilt, valid, instances     as above

  cameraMask        uint32   the camera set the instances carried at build time
  srcVertexBuffer   uint64   raw D3D11Buffer addresses, NOT references
  srcIndexBuffer    uint64
  skipUnsafe        bool     this record's instances carry per-frame work
}
```

- **`cameraMask`** — `setFrameLastUpdated()` CLEARS `m_seenCameraTypes` on the
  first stamp of a frame, which is trap 1, and the touch has to put it back. The
  frame thread cannot supply the value: the camera is classified on the CS side
  from the DrawCallState, and the gate deliberately runs before any of that
  exists. So the record captures it off the instances themselves. Replaying the
  last known set is not an approximation of a better answer — a draw that did not
  arrive cannot say which camera it would have been classified under.
- **`srcVertexBuffer` / `srcIndexBuffer`** — the death signal, §2.3. Addresses
  rather than references because holding a reference would keep alive the very
  object whose destruction is the signal.
- **`skipUnsafe`** — §2.2. Measured off the instances at build time.

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

> **CORRECTED 2026-08-21 BY THE IMPLEMENTATION. THERE IS NO `TouchRecord`, AND
> THE FRAME THREAD DOES NOT DECIDE.**
>
> This was built as drawn — the frame thread collapsing the payload into a batch
> of keys handed to CS at the arena flush — and then moved, because the flow above
> has a hole that only shows once you ask what makes a skip SAFE rather than what
> makes it possible.
>
> **The hole: a draw that commits for a side effect rather than for geometry.** A
> sky pass, a fog registration, a terrain bake. Those draws are perfectly stable
> under all three of the gate's tests, so the gate predicts a hit on them every
> frame — and skipping one does not merely fail to help, it removes the side
> effect. The frame thread cannot tell them apart from ordinary draws, because
> the thing that distinguishes them is whether they resolved to any RtInstance,
> which is a CS-side fact.
>
> **The fix is not a rule, it is a place.** A record is filed only by a draw that
> actually resolved to instances, so THE RECORD'S EXISTENCE IS THE EVIDENCE. Move
> the decision to where that evidence already is and the whole class disappears
> by construction, with no allowlist and nothing to keep in step:
>
> ```
> FRAME
>  │
>  ├─ FRAME THREAD, per draw
>  │    SubmitDraw head, off live m_context->m_state, NO snapshot built:
>  │      k     = residentDrawKey()       identity, + occurrence ordinal
>  │      gens  = residentGeomGenFold()   vb0 + ib generations only
>  │      mat   = residentMaterialFold()  PS + SRVs + samplers + states
>  │      stash them; DECIDE NOTHING
>  │
>  │    SubmitDrawTail end, objectToWorld now final:
>  │      residentGateJudge() -> writes {key, genHash, predictHit} INTO the
>  │      DrawCallState, which is the one object every route to CS already moves
>  │
>  ├─ CS THREAD, commitGeometryToRT, ahead of submitDrawState:
>  │    predictHit && enable && !verify && touch(key) succeeded
>  │      ├─ yes → return           ← THE SKIP. ~50 of this call's ~52 ms.
>  │      └─ no  → submitDrawState, and build/score the record at the end of
>  │               processDrawCallState from the instances it resolved to
>  │
>  ├─ RESIDENT SWEEP -- ResidentScene::onFrameEnd, from InstanceManager GC
>  │    drain buffer deaths -> invalidate; erase invalid; LRU on the age ladder
>  │    ** NO RETIREMENT BY FRAME AGE **
>  │
>  ├─ VISIBILITY — sceneCull over the FULL resident set (still inert)
>  │
>  └─ TLAS over  resident ∩ visible
> ```
>
> **What survives from the original flow, unchanged and load-bearing:** the frame
> thread still never touches an RtInstance; the two maps are still one per thread
> with no lock; the join is still a single uint64 travelling through the existing
> `EmitCs` stream. It rides on the `DrawCallState` rather than in a message of its
> own, which is strictly less machinery — and it had to, because the ordered chain
> rebuilds its tail context field by field from its own slot while MOVING the
> DrawCallState wholesale, so a verdict on the context would have needed adding to
> the slot, to the seam that fills it and to the rebuild. Three places to keep in
> step, and a future field added to two of them goes missing silently.
>
> **What it costs to decide late:** the frame thread still builds and hands over
> the full DrawCallState for a draw that will be skipped. That is the ~17 us/draw
> §0.1 hoped to remove, and it is not where the frame time is — see the correction
> there. The ~2 ms of camera classification and sky handling above the cut also
> still runs, which is not a cost but the point: those are exactly the side
> effects that must not vanish.
>
> **`gens` is no longer "~7 atomic loads".** It is two — vb0 and ib. The bound VS
> constant buffers came out of the fold entirely; see §1.2's first correction for
> the measurement that forced it, and the material row for what replaced them.

### 2.2 Classes

| class | examples | resident | dirty test | cost/frame |
|---|---|---|---|---|
| **Static** | BSP world, static props | forever | never (gens pinned at 0) | gate hit |
| **Dynamic** | entities, animated props | forever | gen compare | gate hit, or partial re-derive |
| **Transient** | particles, UI, viewmodel | no | n/a | full path, exactly as today |

Static needs no dirty logic at all. Transient keeps today's behaviour verbatim, so
nothing can regress into it.

> **AS BUILT: the Transient class is DERIVED, not declared, and the criterion is
> not the one this table implies.**
>
> "particles, UI, viewmodel" is a description of what those draws look like. What
> actually decides the class is a property of the INSTANCES the draw resolved to,
> and it is not about how often they move. Three things this tree rebuilds from
> scratch every frame hang off the instance rather than off its geometry, its
> transform or its material — so the gate's three tests, however green, say
> nothing at all about them:
>
> ```
> billboards    createBillboards / createBeams append into a per-frame vector
>               that is cleared each frame. A skipped particle or beam is simply
>               not drawn.
> ray portals   processRayPortalData registers the portal every frame.
> decals        decalSortOrder is a per-frame counter approximating GPU draw
>               order. A stale one misorders the decal.
> ```
>
> `ResidentScene::Record::skipUnsafe` is set at build time when any instance in
> the record has a billboard count, is a RayPortal material, or is a decal, and
> `touch()` refuses. Opacity micromaps are the fourth and are read from the option
> instead of latched, because the micromap manager subscribes to the
> instance-update event and keeps per-instance bookkeeping on the assumption that
> a live instance is updated every frame — skipping the draw stops delivering it.
>
> **Why this is worth stating rather than filing as a detail.** The obvious way to
> exclude these classes is a list of shader hashes or material types, and this
> tree has a standing rule against exactly that. Asking the instances what they
> turned out to be needs no list, cannot drift from what the code actually does,
> and gets a class right the first time somebody adds a fifth one — the fifth one
> will show up as a visual bug rather than as a silent skip only if this stays a
> property test. `[ResidentScene] missUnsafe` counts the refusals, so a scene
> reading `touched=0` says WHY rather than merely reading zero.
>
> The **Dynamic** row's "gen compare" is also now wrong in its details: the gate
> compares the derived `objectToWorld` and the material fold, not generations.
> The row's conclusion holds — a mover re-derives every frame anyway and is not a
> residency candidate — but a Dynamic entity that happens to be standing still
> IS one, and that is the case §2.3's death signal exists for.

### 2.3 Death — the only real retirement signals

Records do not die of frame age:

- **source buffer destroyed** — the engine freeing the VB/IB *is* the object ceasing
  to exist, and DXVK observes it directly
- **level / map change** — bulk clear
- **memory pressure** — LRU by `frameLastSeen`

`numFramesToKeepInstances` stops being a lifetime and becomes a Transient-only fallback.

> **ALL THREE IMPLEMENTED 2026-08-21 — AND THE FIRST ONE IS NOT OPTIONAL, WHICH
> THIS SECTION'S PLACEMENT OF IT AS ONE BULLET OF THREE UNDERSTATES.**
>
> A resident instance is exempt from lifetime expiry. That is the entire feature,
> it is correct for world geometry and static props, and it is WRONG for anything
> that can be destroyed. Without the buffer-death signal a killed entity's
> geometry stays in the ray-traced scene — casting shadows, appearing in
> reflections — until its record happens to be evicted, which under
> `maxRecords=65536` may be never. A ghost, with nothing in the log to say so.
> That is the worst-shaped bug this feature could produce, and it is the direct
> consequence of the very sentence at the head of this section.
>
> Built as: `~D3D11Buffer` reports into a batched, locked queue; the sweep in
> `ResidentScene::onFrameEnd` drains it and invalidates every record whose
> `srcVertexBuffer` or `srcIndexBuffer` is in it, folded into the pass that
> already walks every record. Two properties the destructor forced:
>
> - **Safe during teardown.** The queue lives in a deliberately leaked
>   allocation, because `~D3D11Buffer` runs late in process shutdown — the same
>   hazard the existing comment in that destructor is a standing warning about.
> - **Free when residency is off.** One relaxed atomic load and return, until a
>   record has ever been built. This runs for every buffer the engine frees,
>   which during streaming is a hot path.
>
> `[ResidentScene] srcDied` is cumulative. Reading zero for a whole session in a
> game with entities means the signal is not arriving, which is a defect and not
> a quiet scene.

### 2.3.1 THE FOURTH LIFETIME, WHICH THIS PLAN DOES NOT HAVE

**Added 2026-08-21. The GEOMETRY has to outlive the instance, and by default it
does not.**

`rtx.numFramesToKeepBLAS` is **1**. A `BlasEntry` is destroyed the frame after its
last draw; `onSceneObjectDestroyed` then marks every linked instance for
collection; and `m_isMarkedForGC` is a clause residency deliberately does NOT
override, because an instance must never outlive its geometry.

So the off-screen object this whole document exists to keep — the one that gets
no draw at all — would have had its instances faithfully exempted from lifetime
expiry by §1.3's clause, and then reaped one frame later anyway, through the
geometry. Residency would have measured as working and done nothing.

Two places now keep geometry alive, and they are for two different populations:

```
draw arrives, gate hits     ResidentScene::touch stamps blas->frameLastTouched
                            through each instance's own m_linkedBlas
no draw arrives at all      SceneManager::garbageCollection asks the record store
                            (ResidentScene::holdsInstance) before destroying an
                            aged-out BlasEntry
```

The second is the important one and the first cannot substitute for it: a touch
only happens when a draw arrives, and the population at issue is the one whose
draws do not. Cost is a walk of the linked instances, paid only for entries that
have already aged past the bound — i.e. the ones about to be destroyed anyway.

**The general lesson, and it is the one to carry into slices 9 and 10:** an
exemption granted to an object is worth nothing while a shorter lifetime it
depends on is still running underneath. Before extending residency to anything
new, ask what else in the tree is counting frames on that object's behalf.

---

## 3. TRAPS — ALL FOUR ALREADY PAID FOR IN THIS TREE (AND FOUR MORE SINCE)

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

**FOUND BY THE IMPLEMENTATION, 2026-08-21. Traps 5-8 are all the same shape as 4
— alive, and not refreshed — arriving through four doors this document did not
have. That repetition is the finding: "resident" is a claim about ONE of an
object's dependencies, and every other dependency it has is a separate lifetime
that has to be checked by name.**

5. **The gate proves three things, and a record holds more than three.** A record
   holds RtInstances, and an instance carries a MATERIAL as well as geometry and
   a transform. Two-test gate, unchanged texture assumed, stale material served.
   Closed by §1.2's third row. The general form: enumerate what the record
   actually holds before deciding what the gate has to test.
6. **A draw can exist for something other than its geometry.** Sky, fog, terrain.
   Perfectly stable under every test the gate has, and skipping one removes the
   side effect rather than the cost. Closed structurally by deciding on the CS
   thread, where "did this draw resolve to any instance" is answerable — see the
   correction under §2.1. **A guard that has to be listed is a guard that will be
   incomplete; find the place where the question answers itself.**
7. **Per-frame work that hangs off the instance.** Billboards, ray portals,
   decals, opacity micromaps. None of them are inputs to the gate's three tests
   and all of them are rebuilt every frame. Closed by `skipUnsafe`, measured off
   the instances rather than declared per shader — see §2.2.
8. **A shorter lifetime running underneath the one you exempted.**
   `numFramesToKeepBLAS` is 1, so the geometry dies a frame after its last draw
   and takes the instances with it through `m_isMarkedForGC`, which residency
   correctly does not override. The exemption was real and worth nothing. Closed
   by §2.3.1. **This is the one that would have been hardest to diagnose from a
   log**: every counter residency owns would have read healthy.

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

### 4.1 STATE, 2026-08-21

| # | slice | state |
|---|---|---|
| 0.5 | **stable buffer identity** — NOT IN THE ORIGINAL TABLE, blocked 5-8 | LANDED. §0.2. Gate: `[StaleTape] staleSurfaces = 0` |
| 1 | options | LANDED, and touched a second time to DELETE `foldCbGenerations` — see below |
| 2 | ResidentScene map + key + dirty fold | LANDED. `build`/`touch`/`score`/`holdsInstance` are new since the first pass; the fold is now three tests, not one |
| 3 | lifetime contract | LANDED |
| 4 | GC keep clause | LANDED, and it needed a second clause for the BLAS — §2.3.1 |
| 5 | frame-thread key + folds + verify scoring | LANDED. Split into an entry half and a tail half; content hash REPLACED by the derived `objectToWorld`, plus a material fold |
| 6 | collapsed payload + CS-side stamp | LANDED, in a different place than drawn — the skip is on the CS thread. §2.1 |
| 7 | flip `verify` off, raise the keep | **YOURS, AND NOW A HARD PREREQUISITE FOR §7 SLICE D, not an independent step.** Both keep clauses read `enable() && !verify()` — rtx_instance_manager.cpp:3062 and rtx_scene_manager.cpp:388 — so with `verify=True` residency holds nothing, whatever the liveness source is. Still conf only, still on §5's evidence |
| 8 | seed pass | LANDED. Epoch-triggered, gameplay-gated, master-switch only. §0.0 |
| 9 | off-screen mover transforms | NOT STARTED, deliberately. §6 says do not, until it is visible — and §7.0 now closes it as a side effect, so it should not be built on its own |
| 10 | sceneCull `solidAngleMin` sweep | **YOURS.** Conf only, and unchanged by §7 except in WHEN it becomes meaningful: it waits on §7 slice D rather than on slice 7, because D is what actually puts the off-screen set back. Still the last thing, still the whole GPU-side lever |

**Nothing here is built or run.** The whole of it is source.

> **AND SLICE 8 DOES NOT DO WHAT THIS TABLE IMPLIES, 2026-08-21.** The seed pass
> LANDED, and it still cannot produce the population §0.0 needs — the culls that
> hide the level are conf-disabled as patches and stay live during the seed, and
> the epoch only re-arms on `SceneManager::clear()`. See the correction under
> §0.0 and the replacement in **§7**, whose build order supersedes this one for
> everything after slice 8. Slices 1-8 are not wasted: §7.0 lists what carries
> over, which is nearly all of it.

**`rtx_options.h` was touched twice, and the second touch was a deletion.**
`rtx.residentScene.foldCbGenerations` is gone. It defaulted true and folding the
bound VS constant-buffer generations makes the gate hit on ~0% of draws in this
game — §1.2's own measurement, arrived at after the option was written. Leaving
it as a switch would have been a default that quietly disables the feature, and
its stated purpose (a safe superset) is served instead by the material fold,
where the inputs do not churn per frame. One line removed rather than one line
re-documented, per the tree's own rule about flags that exist to avoid a
deletion.

### 4.2 THE INTERACTION TO KNOW ABOUT BEFORE READING ANY NUMBERS

`rtx.shardInstanceProcessing` defaults **true**. Under it the per-instance work
(drawCallCache resolve, geometry cache-state decision, instance find/update)
already ran on workers at flush, before the CS-side skip fires. So a hit still
removes the geometry bake, the material work and the CS-domain residue — but not
the worker half, which has already been paid.

Residency and Phase 2b therefore overlap, and the honest reading is that the two
of them are attacking the same 16-23 ms from opposite ends. Extending the gate to
the flush pre-pass would need the record store readable from the game thread,
which is exactly the second shared map §2.1 exists to avoid — so it is a separate
change with its own design, not a tuning step. **Take the A/B against
`rtx.shardInstanceProcessing=False` before concluding anything about how much
residency is worth.**

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

### 5.1 THE ARMING SEQUENCE AND WHAT EACH COUNTER MEANS

**Added 2026-08-21.** The gates above are still the gates. This is what the code
now prints against them, and in what order to read it.

**Step 0 — the prerequisite (§0.2).** `[StaleTape] staleSurfaces = 0` across a
level load and a stationary capture. It splits its count two ways now: `oob` is
an index past the end of the table, which nothing hands out any more, and `dead`
is an index into a slot that was reclaimed — the one to watch, because it means
the reclaim sweep freed a slot a live surface still referenced. Nothing below is
readable until this is zero.

**Step 1 — score the prediction, act on nothing:**

```
rtx.residentScene.enable   = True
rtx.residentScene.verify   = True
rtx.residentScene.logStats = True
```

`[ResidentGate]`, frame thread, the PREDICTION:

```
newKeys   ~0 in a stationary scene = the IA key is stable and safe to arm.
          Hundreds = the key churns and arming a long keep on it is trap 3.
          THIS IS THE FIRST NUMBER TO READ AND IT CAN VETO EVERYTHING ELSE.
missKey   seen before but not last frame. Contiguity refusing a stale record.
missGen   vertex or index data genuinely moved. The GOOD failure: identity is
          stable, the geometry is not, those draws are not candidates.
missO2w   geometry clean, derived transform moved. High on world geometry means
          the derivation is not reproducing itself -- a bug to find, not a class
          to exclude.
missMat   the first three clean, material state moved. High means the game
          rebinds per draw and the fold wants narrowing to what FillMaterialData
          reads. A measurement, not a guess (§1.2).
noTail    got a key, never reached the judge. The ~22% question, §1.2. hitPct is
          computed over draws that REACHED the gate, so only this sees it.
```

`[ResidentScene]`, RT side, the CONSEQUENCE:

```
records    MUST PLATEAU in a stationary scene. Rising = trap 3. Hard fail.
predicted  draws the gate said it could serve.
FAIL       ...where the record named DIFFERENT instances than the full path
           resolved to. MUST BE 0, not low, across a full pitch-and-yaw sweep.
           Every one is an object that would have been held on stale contents.
failNoRec  subset with no record at all. Two causes, and they want opposite
           responses -- read evicted= and wiped= on the same line first.
```

**Step 2 — take the skip.** Only with `newKeys ~0`, `records` plateaued and
`FAIL=0`: `rtx.residentScene.verify = False`. Now `touched` starts moving, and
the miss split says why it does not:

```
touched      draws the skip was ACTUALLY taken on. Read against [ResidentGate]
             hit= -- a large gap is the whole story of the next three columns.
missUnk      no record. Ordinary and self-correcting, and also the draws that
             have no record BY DESIGN (§2.1's side-effect class). A steady
             non-zero here is that population, not a fault.
missInval    a record's instance was destroyed. THIS is the plan's original
             "touchMiss ~0" gate, and it wants finding.
missUnsafe   refused: billboards, ray portals, decals, or micromaps on (§2.2).
             Permanent for those draws. If it reads close to hit=, residency is
             being refused nearly everywhere and the reason is one of those
             four, not the key.
srcDied      CUMULATIVE. Records retired because the engine freed their buffers
             (§2.3). READING ZERO FOR A WHOLE SESSION IN A GAME WITH ENTITIES
             MEANS THE SIGNAL IS NOT ARRIVING -- a defect, not a quiet scene.
```

And the original gate, unchanged and still the one that says it worked:
`[ReapJoin] starved -> ~0`.

**Step 3 — invariance.** `rtx.residentScene.seedFrames = N`. Without it the
resident set is view-history dependent and §5's sweep test cannot pass by
construction — §0.0. Watch for `[ResidentSeed] armed` / `complete`, and expect
those N frames to cost the unculled frame time.

**Step 4 — the GPU half.** Only now does `rtx.sceneCull.solidAngleMin` mean
anything, because only now is there an off-screen set to reject from (§1.4).

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

> **NOT STARTED, 2026-08-21, and deliberately — this section is the reason.**
> Slices 1-8 landed and this one did not, because "do not start it until it is
> visible on screen" is an instruction, not a caveat. The measurement it asks for
> has not been taken.
>
> **Two things have changed underneath it, and one of them makes the defect
> REACHABLE where before it was hypothetical.**
>
> The class table's "keep Dynamic out" is no longer a one-line change, because
> there is no Dynamic flag to key on. §2.2's class is derived from what the
> instances turned out to be, and "moves sometimes" is not one of the properties
> it tests — a Dynamic entity that happens to be STANDING STILL passes all three
> of the gate's tests and becomes resident. That is correct while it stands still
> and is exactly the case this section describes the moment it walks off-screen
> and starts moving.
>
> So if the stale pose does show, the cheap alternative has to be built rather
> than switched on. The honest version of it is a property test in the same shape
> as `skipUnsafe`: a record whose key has ever been seen with a moved
> `objectToWorld` is a mover, and movers are not residency candidates. That is
> one bit on the frame-thread gate entry, which already computes the comparison —
> `missO2w` is the count of exactly that event. Cheap, and derived rather than
> declared, which is the rule §2.2 settled on.
>
> **The other change removes the worse half of the problem.** §2.3's death signal
> means a Dynamic entity that is DESTROYED while resident is now retired properly
> rather than ghosting forever. The residual is only the stale POSE of an entity
> that is still alive — which is what this section always claimed it was, and is
> bounded by how long it stays culled and how fast it moves. That bound is the
> measurement to take before anything is built.

---

# 7. THE ENUMERATION ARCHITECTURE — WHAT REPLACES THE SEED

**Added 2026-08-21, after slices 1-8 landed and §0.0's seeding answer was
measured and found insufficient.**

## 7.0 The inversion, again, one level up

§0 demoted the draw from *"here is an object"* to *"an object you already have
may have changed"*. It could not finish the job, because it left the draw as the
only evidence an object EXISTS — and an engine-culled object produces no draw.
"Absence of a draw says nothing" is unactionable without a second source that
says something.

```
residency as built     liveness   = a draw arrived and the gate hit
                       death      = the source buffer was freed (a proxy)
                       population = whatever the player has looked at

enumeration            liveness   = the object is in the engine's pre-cull list
                       death      = it is not
                       population = the list, view-invariant by construction
```

Three things this fixes that the built version cannot:

1. **A positive liveness signal**, which is the missing piece above.
2. **A precise death signal.** `~D3D11Buffer` is a proxy: the buffer dying means
   the object died, but a pooled or cached model can die without its buffer
   being freed promptly. "Not in the list" is an observation, not a proxy.
3. **§6 closes.** The list entry is an `IClientRenderable*` and CULLING_BIBLE
   §5.1 maps its vtable, so `GetRenderOrigin` / `GetRenderAngles` supply a pose
   with NO draw. That is the one problem §6 declared permanently out of reach.

It also dissolves traps 2 and 3: an engine handle is the stable identity §1.1's
IA-pointer bundle was approximating, so the per-frame occurrence ordinal goes.

**WHAT SURVIVES FROM SLICES 1-8, WHICH IS NEARLY ALL OF IT.** The `ResidentScene`
record store, `build` / `touch` / `holdsInstance`, the `m_residentKey`
back-pointer contract, both GC keep clauses (§2.3.1), the `BufferSlotTable` fix
(§0.2), the `skipUnsafe` class test (§2.2) and the material fold (§1.2) are
unchanged. One thing changes: where liveness comes from. The frame-thread gate
stops being the decider and becomes the thing that learns the identity mapping
and detects change.

## 7.1 THE NOP IS OUT OF SCOPE UNTIL A NUMBER ASKS FOR IT

The architecture as proposed was: turn culling off so the full list arrives, then
NOP the unchanged objects so the 58 ms is not paid. Coherent, and the cheap
version gets the same correctness: **read the list, let the engine cull normally,
and use the list only to decide who stays resident.** Today's 538 draws, today's
19.97 ms, plus a complete population — with no new patch sites and no callback
into DXVK from inside a job-threaded engine traversal.

The NOP is worth building only once the engine's own per-object cost is shown to
be the ceiling, and `[Perf.CsSplit]` puts dxvk-cs at 99.6% of wall time, so it is
not. Optimising a step whose cost has not been measured is the thing
`rules/subtractive-engineering.md` exists to stop. It stays as slice G, gated on
evidence rather than on intent.

## 7.2 THERE IS NO SINGLE LIST — THERE ARE THREE, AND ONE IS NOT A LIST

| subsystem | list | shape | difficulty |
|---|---|---|---|
| Renderables (entities, animated props) | `sub_1801A8350`, output at `this+0x1D8`, `IClientRenderable*` at `+0x8000+8*slot`, count in `rax`. Hook site `client.dll+0x1A827D` (BIBLE §3.1) | clean array, per-object handle | low — the map is done |
| Static props | `CStaticPropMgr::GatherVisibleStaticProps` = `sub_1801B2200`, prop array `qword_1807D2970`, stride `0xD0`, index-addressable (BIBLE §0.2b) | clean array, and the WHOLE array is walkable, so the pre-cull set is free | low-medium — needs a per-prop draw latch |
| World geometry (BSP) | `sub_1802E8DA0`, output is run-length-encoded **accepted** leaves | no pre-cull list exists. BIBLE §0.2e: *"the subtraction is an omission"* — a culled leaf is never added | high |

> **§7.7 NOW MEASURES THIS DIRECTLY AND THE ESTIMATE HOLDS.** Every draw billed
> to the call site that queued its record: world 68%, models 14%, client 9%,
> matsys 9%. Read §7.7 for the per-site map — the shares below are right, and
> the reasoning that produced them (gate calls against draws) is not how they
> were confirmed.

**The scale check that re-orders everything.** Measured, not estimated: `[Join]`
reads gate calls 64-137 per frame against draws 403-577 per frame. Even with
perfect attribution the client renderable list accounts for at most about a fifth
of the draw stream. World geometry and static props are the other four fifths and
pass through none of it. **Slices F and H are where the population is; slice B is
the small half.** Do not read the renderable work as the main event.

World is also the case that needs this least. It is immutable, `[VsResidency]`
measured `vb=0 ib=0`, and an object that cannot change has to be seen exactly once
ever. That is the one job the seed pass can still do.

## 7.3 SLICE A — MEASURED, AND IT VETOED ITS OWN HOOK SITE

Built as `[Join]` in `d3d11_rtx.cpp`, riding the already-installed
`renderListWrapper` and `gateWrapper` hooks. Gated on `rtx.residentScene.logStats`.

**Result, 28 logged frames, unambiguous:**

```
[Join] tid{gate=37464 draw=47940 same=0} listCalls=1 listed=87..175
       gate{calls=64..137 pass=calls} draws=403..577 latched=0 noLatch=draws
       drawn=0 drawnNotListed=0
```

- **`same=0` on every frame.** Renderable dispatch and the D3D11 draw are on
  different threads, so nothing orders them and the gate cannot attribute.
- **The raw line is the proof, not the aggregate.** On the draw thread `since=`
  climbs monotonically and never resets — 24314, 24315, 24316 on frame 3199, then
  40603 and up on frame 3229. That is `t_drawsSince` never having been touched,
  so the gate's thread-local was never written on that thread at all. Not a stale
  latch and not an ordering fault: categorical thread separation.
- **`drawnNotListed=0` IS VACUOUS HERE** and must not be read as a pass, because
  `drawn=0` means nothing was ever tested.
- **`listed` = 87..175, `listCalls=1`.** BIBLE §3.3 puts Gate 1, the
  view-direction-dependent leaf-frustum test, INSIDE `sub_1801A8350`. So its
  output list is the POST-cull list, and hooking that return cannot give a
  pre-cull enumeration whatever we do about attribution. Caveat: the capture was
  ordinary play rather than the fixed-position sweep, so the variation alone does
  not isolate the cause. The code path settles it, not the number.
- `pass == calls` on all 28 frames: the per-renderable draw gate rejects nothing,
  confirming BIBLE §3A.1's existing "RULED OUT" from a second angle.

## 7.4 WHAT IDA SETTLED — materialsystem_dx11.dll

**The queue carries NO object identity.** `sub_18006FCB0` writes a 64-byte record:

```
[+0]  sub_1800708D0   the replay functor      [+8]  0
[+16] mesh            [+24] counts            [+32] .. [+48] ranges
[+56] pointer to a copied 16-byte tint blob
```

Every byte is accounted for and none of it names the object. That is not an
oversight — matsys is a primitive layer and does not know entities exist.
`sub_18001C390` confirms it at the leaf, where `+144` is `IASetVertexBuffers`,
`+152` is `IASetIndexBuffer`, `+96` is `DrawIndexed` and `+104` is `Draw`: every
input is a buffer, a range, or a global.

**What IS recoverable is the record's ADDRESS,** and that is enough. The
recording thread knows where the record lands and still holds the gate latch, and
the replay functor is handed that same address. So the address is the join key,
carried in a side table — no engine memory written and no ordering assumed.

**Why an address and not an ordinal.** `sub_180087960` picks its queue out of
thread-local storage (TEB TLS slot, index at `+4136`, descriptor array at
`qword_181BBA040`), so there is one queue PER RECORDING THREAD. "The Nth record is
the Nth replay" holds with one worker and breaks with two — intermittently, which
is the worst way for a probe to be wrong.

**Verified against the disassembly rather than assumed:** `sub_1800708D0` reads
`[rdx+4]`, `[rdx+8]`, `[rdx+0x10]`, `[rdx+0x18]`, `[rdx+0x1C]` and `[rdx+0x20]`,
and those land on written qwords only if it is handed `v14+8`. It touches `rdx`
and never `rcx`, so a two-argument forward is exact.

## 7.5 BUILD ORDER

| # | slice | where | gate |
|---|---|---|---|
| A | `[Join]` attribution probe | `d3d11_rtx.cpp` | **DONE, and it vetoed the gate site.** §7.3 |
| A2 | record/dispatch join | `d3d11_rtx.cpp` | **DONE, and it replaced its own design. See §7.7.** The two rel32 rewrites this row used to name (`matsys+0x6FD4F`, `matsys+0x6FD3C`) were both wrong and are gone: the functor is handed a STACK COPY, not the record, and that call site carried 0.3% of the frame. Built instead as two aligned atomic patches — the allocator's ENTRY at `matsys+0x87960` (300+ call sites make a call-site rewrite untenable) and the dispatcher's `call rdi` at `matsys+0x87F89`. Key is `base + head`, NOT `+8`: the allocator steps the head past its own 8-byte header before returning. Reads `hit≈98%`, `reuse=0`, `unknown=0` |
| B | pre-cull enumeration | TBD — **NOT `sub_1801A8350`'s return**, §7.3 proved that is post-cull | `listed=` FLAT across a fixed-position pitch-and-yaw sweep |
| C | re-key residency on the engine handle, and drop the occurrence ordinal for handle-keyed records | `rtx_resident_scene.*`, `d3d11_rtx.cpp` | `newKeys` to 0, `FAIL=0` with `verify=True` |
| D | **flip the liveness source** — `onFrameEnd` touches every record whose handle is listed, draw or no draw, and absent handles invalidate | `rtx_resident_scene.cpp` | `[ReapJoin] starved` to ~0 **without** `seedFrames`, and the resident count flat under the sweep |
| E | transforms without draws — `GetRenderOrigin` / `GetRenderAngles` off the listed renderable | `d3d11_rtx.cpp` | walk an entity out of view and watch its shadow track it. §6's measurement, now cheap |
| F | static props — B through D again against `sub_1801B2200` and `qword_1807D2970`. Props cannot move, so E does not apply | TBD | `rtx.cullOff.staticPropFade=False` stays and props stop leaving the RT scene |
| G | the NOP — conditional per-object skip at each subsystem's draw entry | TBD | **only** on a `[Perf.CsSplit]` reading that says the engine half is the ceiling (§7.1) |
| H | world geometry, or a decision not to | TBD | re-read after F. A static world seen once may need nothing |

**Slice B is now the open question, and it is open in a way it was not before.**
§7.3 removed the obvious hook point. The candidates, in the order the tree's own
rules prescribe — `RtlCaptureStackBackTrace` at a DXVK-controlled site first, IDA
second:

- upstream of `sub_1801A8350`, at the leaf and PVS enumeration it consumes,
  before Gate 1 runs;
- inside it, at the seed point before the frustum AND is applied. BIBLE §3.3
  names `leafSys+2104` and `leafSys+3128` as the mask source, producer unproven;
- or accept it as post-cull and defeat the culls during a bounded seed, which is
  §0.0's answer with the individual groups actually turned on — and that is a
  measurement nobody has taken, `pvs` in particular.

**Do not start C through E before B lands.** Every one of them consumes a list
whose shape and invariance are what B decides, and building against the post-cull
list would reproduce the view-history dependence this whole section exists to
remove.

## 7.6 THE ITEMS THAT ARE NOT SLICES, AND WHY EACH ONE IS STILL OPEN

§7.5 covers the enumeration work. These five came up while it was being designed,
none of them is a step on that path, and every one of them can invalidate a
reading taken from it. They are here rather than dropped because an open item
with no home is how the seed pass came to be believed for a whole session.

**1. `noTail` — a fifth of the frame the gate never sees, and enumeration does
not fix it.** §1.2 raised it and it is still unmeasured. Draws that never reach
`SubmitDrawTail` get no key, so they file no record and can never be resident —
whatever the liveness source is. This is orthogonal to §7 by construction:
changing where liveness comes from does not help a draw that never reached the
place a record is filed. `hitPct` is computed over draws that reached the gate,
so it cannot see the shortfall; `noTail` is the shortfall. **Read it before
believing any coverage number from slices C-F.** It also does not answer whether
those draws are RT-captured at all, which is a separate open question wanting
`arrived=` / `noStash=`.

**2. The seed is not deleted — it is demoted to learning the MAPPING.**
Enumeration supplies liveness; it does not supply the join from an engine handle
to the IA identity the record store is keyed on. That join can only be learned by
observing a draw, so an object still has to be drawn ONCE, ever. For world and
static props that is a one-time cost and the seed does it well. For streaming —
§0.0's gap 1, geometry that loads after the seed — it is still open, and the
re-seed trigger §0.0 asks for is still unbuilt. Under §7 the trigger is cheaper
than it was: a handle appearing in the list with no mapping IS the trigger, so it
needs no rolling budget and no streaming hook. That is a slice-C detail, noted
here so it is not rediscovered as a defect.

**3. `rtx.shardInstanceProcessing` — take the A/B before believing any residency
number.** §4.2 says it and it has not been done. It defaults true, and under it
the per-instance work already ran on workers at flush, before the CS-side skip
fires. So residency and Phase 2b attack the same 16-23 ms from opposite ends and
a measurement of one includes the other. Note the live conf pins it to **False**,
which is not the default the section was written against — so §4.2's reasoning
applies, but its arithmetic does not, and the A/B has to be run rather than
inferred either way.

**4. The occurrence ordinal survives for everything §7 does not re-key.** Slice C
drops it for handle-keyed records, which is right and is trap 2 dissolving. It
does NOT drop for world geometry or for static props before slice F, and those
are ~80% of the draw stream (§7.2). The residual risk is the one engine culling
creates: when copy 1 of a 3-copy identity is culled, copies 2 and 3 shift
ordinals and serve records built for different physical objects. `verify=True`
and `FAIL` exist to catch exactly that, and the population most likely to trip it
is the one residency is aimed at — so **`FAIL=0` has to be re-established across
a pitch-and-yaw sweep after every slice, not once at the start.**

**5. `rtx.conf` still sets `rtx.residentScene.foldCbGenerations`.** §4.1 deleted
that option from `rtx_options.h`. The line is inert and harmless, and it will
cost the next reader time. Delete it when the conf is next touched — not on its
own, because `rtx_options.h` is not involved and a conf-only edit is free.

---

# 7.7 THE PRODUCER MAP — MEASURED, AND IT REPLACES §7.2's ESTIMATE

**Added 2026-08-22, after slice A2 was rebuilt at the real chokepoint and run.**

§7.2 sized the three populations by inference. This is the same question answered
by measurement, and the technique is the point: **attribute a draw to whoever
QUEUED its record, not to whoever dispatched it.**

The allocator hook is an ENTRY detour reached by a `jmp`, so the stack is
untouched and `_ReturnAddress()` inside `qAllocWrapper` is the genuine caller of
`sub_180087960`. That address goes in the join entry beside the renderable, and
the draw thread bills every dispatched record to it. A call-site hook could not
have produced this — it would have returned the island.

**Coverage is total and stays total.** `[Join.who] draws=544 billed=544
unknown=0 sites=17`, on every logged frame. Seventeen producing call sites are
the entire frame.

### The map — one position, camera still, ~544 draws/frame

| producer | draws | share | payload | population |
|---|---|---|---|---|
| `engine.dll+0x1b42e2` | 210 | 39% | `[view+0x1d0]`, iface `client+0xC3D9B8` slot `+0x70` | world |
| `engine.dll+0xb8500` | 64 | 12% | `[view+0x1d0]`, iface `client+0xC3D988` slot `+0x40` | world |
| `engine.dll+0x1b3693` | 64 | 12% | `[view+0x1d0]`, iface `client+0xC3D9B8` slot `+0x88` | world |
| `engine.dll+0x1b3ccb` | 32 | 6% | `[rbp+0x88]`, iface `client+0xC3D9B8` slot `+0x78` | world / brush |
| `studiorender.dll+0x13f74` | 58 | 11% | path A — **`inSpan=0`** | model |
| `studiorender.dll+0x13d54` | 19 | 3% | path B — **`inSpan=19`** | model |
| `client.dll+0x6cd5bb` | 29 | 5% | queues helper `client+0x6C50B0` | client |
| `materialsystem_dx11.dll+0x6f092` / `+0x6fbf9` | 41 | 8% | — | matsys |
| six more | 15 | 3% | — | — |

**world 370 (68%) · models 77 (14%) · client 48 (9%) · matsys 47 (9%)**

That confirms §7.2's shape — world and props are the four fifths, renderables
are the fifth — but it does it with call sites instead of categories, so each
share is now a thing you can put a bracket around.

### TWO OPPOSITE ERRORS, BOTH MADE FROM MODULE NAMES

**`[JoinStack]`'s 64% `engine.dll` was attribution by DISPATCH HELPER.** The
census stands at the draw and walks up, so it names the module whose helper
drains the record on the render thread. Every one of those helpers is called
directly by the dispatcher at `matsys+0x87f9f`; none of them is the code that
decided to draw.

**Then the producer chains were read the same wrong way in the other
direction.** Every one of the 17 chains passes through `client.dll+0x372857`
(`sub_180372330..373760`) and `client.dll+0x35a9f6` (`sub_18035A7F0..35AB5A`),
which looked like proof the frame was client-driven and therefore renderable-
shaped. It is not. `sub_18036C800` reads an opaque render list out of
`[view+0x1d0]` and hands it to an engine interface wholesale:

```
18036c8b5  mov  rcx, [rip+0x8d10cc]   ; engine iface at client+0xC3D988
18036c8bc  mov  r8,  [rsi+0x1d0]      ; the world render list
18036c8c9  call [rax+0x40]            ; -> producer engine.dll+0xb8500,  64 draws
18036c8d2  mov  rcx, [rip+0x8d10df]   ; engine iface at client+0xC3D9B8
18036c8d9  mov  rdx, [rsi+0x1d0]      ; the same list
18036c8ea  call [rax+0x70]            ; -> producer engine.dll+0x1b42e2, 210 draws
```

`rsi` is the view — the function also reads `[rsi+0x188]`, `[rsi+0x14200]`,
`[rsi+0x142ac]` — and one call draws a batch. There is no per-object anything at
that level. It is world geometry reached through a client wrapper.

> **THE RULE THIS LEAVES: neither the producer's module nor the dispatcher's
> module identifies the population. Only the PAYLOAD does.** A render-list
> handle is world. A studio model handle is a model. Two of this session's
> conclusions were drawn from module names and both were wrong, in opposite
> directions, within an hour of each other.

### The two studio paths, and why the span only saw one

Both reach `studiorender`, and only one passes through a bracketed span:

```
path B  inSpan=19   client+0xf2ead -> lodWrapper -> sub_18026B516 -> studiorender+0x13d54
path A  inSpan=0    pre-pass -> sub_18036E86E -> sub_18026B750 -> sub_18026BC40 -> studiorender+0x13f74
```

`sub_18036E86E` is the 93-byte stub the visibility pre-pass at
`sub_18036E7D0` falls through into. So the batch the pre-pass gathers is drawn
through a path `lodWrapper` never sees, which is the whole reason `spans` read 4
and then 17 while the model share was 14%. **Path A is the one bracket worth
adding**; it is three times the draws of the path already covered.

---

# 7.8 RESIDENCY WITHOUT A DRAW, AND THE ONLY CORRECT EXCLUSION

§0.0 says *"you cannot raytrace geometry you have never received"*, and that is
true exactly once per object. Everything after the first receipt is a different
question, and this section is the answer to it.

**A record needs a draw ONCE, to exist. After that a draw is not evidence of
anything residency needs, and its absence is not evidence either.**

### The draw is not the signal — §1.2 already measured that

```
vs=0x2af9b90d  vb=0 ib=0  cbGen=975/975  cleanContentPct=90
```

Geometry never moves. Constant-buffer *generations* move on 100% of draws, and
the bytes the draw actually READS are unchanged on 90% of them. So a draw
arriving tells you nothing about whether anything changed; only a comparison of
content does. The gate exists because of that.

Turn it round and the same sentence is the residency argument:

- a draw that arrives with identical content produces identical output —
  reprocessing it is waste, and the gate skips it;
- a draw that never arrives, for an object whose content is identical, produces
  identical output too — **evicting it is a bug**.

Those are one statement applied to opposite cases. The gate and residency are
the same test.

### THE EXCLUSION CRITERION

**Exclude a record only when its OUTPUT would differ.** Output differs when, and
only when:

1. the geometry content changed — content, not `GetMapGeneration()`;
2. the transform changed;
3. the constants the draw actually reads changed — the content test of §1.2, not
   the generation test;
4. the object was destroyed — `~D3D11Buffer`, the only true death (§2.3).

Everything else is not a reason to exclude. Not "no draw arrived". Not "it is
off-screen". Not "the engine culled it". Not "it is in the Dynamic class".

### WHY CLASS IS THE WRONG DISCRIMINATOR, AND IT IS A TRAP

**Small props are Dynamic class even though they never move.** Source classifies
by how a thing is rendered, not by whether it moves: `prop_dynamic`, client-side
physics props, anything carrying a fade or an animation path is Dynamic-
classified. A rule keyed on class treats all of them as movers.

The measurement says they are not. `vb=0 ib=0` and `cleanContentPct=90` are the
whole draw stream, not a subset — 90% of draws produce identical output. A
class-keyed rule would exclude a large, genuinely static population, and it
would do it silently, because nothing in a class check can report that it was
wrong. **Gate on the output test, never on the class.** `skipUnsafe` (§2.2) is a
class test and it is correct as a SAFETY floor; it must not become the liveness
rule.

### WHAT THIS MEANS FOR §7

§7 wants a positive liveness signal because *"absence of a draw says nothing"*.
That is right, and the corollary is stronger than §7 states: **once the output
test exists, absence of a draw does not need to say anything.** A resident record
stays valid until something says its output changed, and three of the four
things that can say so need no object at all:

```
content changed      observed at the next draw that DOES arrive
destroyed            ~D3D11Buffer, implemented (§2.3)
map changed          bulk clear, implemented (§2.3)
transform changed    NEEDS THE OBJECT -- and only this one
```

The fourth is §0.0 gap 2 and slice E, and it applies to true movers only. So:

**Object attribution is required for the movers and for nothing else.** The
producer map prices that at 14% of the frame — the two `studiorender` paths — of
which one is already covered by the join.

The other 86% divides cleanly:

- **World, 68%.** No object, no motion. It needs to be RECEIVED once and never
  evicted, and §7.7 makes "received once" concrete: one function,
  `sub_18036C800`, one payload, `[view+0x1d0]`. This is an ENUMERATION problem
  and no amount of attribution touches it — knowing which BSP surface a draw
  belonged to does not help receive the surfaces never drawn.
- **Client and matsys, 18%.** Unclassified, small, and each one now has a call
  site to start from rather than a category.

### THE MACHINERY ALREADY EXISTS

Nothing above needs new lifetime plumbing. §1.3 keeps an instance alive with a
frame stamp and an O(1) total back-pointer; §2.3.1 keeps the GEOMETRY alive
underneath it, which is the half that was missing and would have made residency
measure as working while doing nothing; §2.3 supplies the only real death.

What is missing is not a mechanism. It is the population — §7.7's 68% — and the
output test being the thing that decides eviction instead of the draw.
