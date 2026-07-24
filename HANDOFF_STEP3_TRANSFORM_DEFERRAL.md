# HANDOFF: Step 3 (transform-extraction deferral) — plan, blocker, + batch invariants

Session date: 2026-07-24. Continues HANDOFF_BATCHED_PARALLEL_SUBMITDRAW.md.
Steps 1 (material), 2 (hashing + bbox) and 4 (skinning palette) are implemented in
the working tree behind `rtx.batchSubmitDrawStages` (+ per-stage sub-flags). This
handoff is about Step 3 (transforms) and records the **synchronous invariants** and
**cost/design decisions** from all four stages so nobody re-derives them.

Every claim below has a `file:line` you can re-verify. Where a decision was a
judgement call, the reason is recorded.

---

## 0. TL;DR / verdict on Step 3

**Deferring `ExtractTransforms()` into the frame-end parallel-for the way material /
hashing / bbox / skinning were deferred is NOT viable.** The other four stages work
because their output is consumed **only later, on the CS thread** (in
`commitGeometryToRT`). The transform result is different: `dcs.transformData` is
**read and re-written at 30+ sites inside SubmitDraw itself**, *after* the
`ExtractTransforms()` call, to drive camera classification, sky detection, stable
prop-id, anti-cull, and the native-raster / UI decision. SubmitDraw cannot proceed
without the transforms, so they cannot be produced at frame end.

So Step 3 is not "fold one more stage." It is either (a) a much narrower deferral of
only the matrices that are consumed *exclusively* by `commitGeometryToRT` (likely an
almost-empty set — see §2), or (b) a serial-cost reduction of `ExtractTransforms`
(the honest win — see §3). **Measure before refactoring.**

---

## 1. Why transforms are not a batch stage (the blocker, with evidence)

- `bt_extractXf` brackets exactly one statement: `dcs.transformData = ExtractTransforms();`
  (`d3d11_rtx.cpp:20991`). It is the dominant SubmitDraw cost (~213 ms/frame steady-state
  TF2 per the bucket comment at `d3d11_rtx.cpp:5832`).
- `ExtractTransforms()` spans `d3d11_rtx.cpp:8442`–`13394` — **~4,950 lines** (the
  in-code comment says ~3,800). It IS the classifier, camera reconstruction, and matrix
  assembly, interwoven.
- It **reads live `m_context->m_state`** (VS constant buffers, viewport) — state the
  game thread mutates on the very next draw. A worker reading it is a torn read.
- It **writes ~a dozen cross-draw member caches** that the *next* draw's classification
  reads — inherently serial state, not parallelisable per-draw:
  `m_projSlot/m_projOffset/m_projStage/m_columnMajor`, `m_viewSlot/m_viewOffset/m_viewStage`,
  `m_camFallbackCache`, `m_lastGoodTransforms` (+ its mutex), `m_foundRealProjThisFrame`,
  `m_hasEverFoundProj`, the axis-detection votes (`m_zUpVotes/m_lhVotes/m_yFlipVotes`),
  `m_smoothedCamPos`, the sky-origin latch, the fanout VP-row cache, `m_lastO2wPathId`,
  `m_lastWtvPathId`, `m_lastDrawCamOrigin`, `m_lastExtractUsedFallback`,
  `m_lastClassifierSaidUi`.
- **The killer:** `dcs.transformData` is consumed synchronously *by the rest of
  SubmitDraw*. Verified read/write sites between the call (`20991`) and the commit point
  (`~27770`) include: `21224–21236` (o2w from cached main w2v), `21309–21363` (o2w +
  stablePropId), `21382–21403`, `21440–21496` (w2v/v2p cache install + o2v build),
  `21524`, `21541/21856` (`m_lastClassifierSaidUi` gate), `21583` (inverse w2v),
  `21732` (o2w for cull), `21915` (`m_lastDrawFilteredAsUI = true`),
  `21945–21969` (cached snapshot install + o2v), and many more downstream (sky cb2
  classification, `processCameraData`, bbox o2w). These gate **native rasterisation /
  routing** decisions that must happen on the game thread, now.

Conclusion: the transform output is load-bearing mid-SubmitDraw. You cannot move its
producer to a frame-end join without also moving every one of those consumers — which
is a rewrite of the back half of SubmitDraw, not a stage fold.

---

## 2. If someone still wants to defer *something* here

The only safely-deferrable transform work is the subset of the final matrices consumed
**exclusively** by `commitGeometryToRT` and by nothing between `20991` and the commit.
From §1 that subset is close to empty (o2w, w2v, o2v, v2p, stablePropId are all read
in-function). Before attempting it:

1. Split `ExtractTransforms` conceptually into **Phase A = classify + update caches +
   resolve a "decode plan"** (slot/offset/stage, `m_columnMajor`, chosen o2w/w2v path,
   cached w2v/v2p) **+ snapshot the resolved cb2/cb3/t30/t31 bytes**, and **Phase B =
   pure matrix math** (read at the resolved offset, transpose if column-major, assemble
   o2w/w2v/o2v/v2p). Phase A MUST stay synchronous (it touches live `m_context` and the
   shared caches, and it makes the UI/native-raster decision — the invariant from the
   prior threading handoff).
2. **Instrument the A/B split and MEASURE Phase B's share of the ~213 ms** before
   writing any deferral. The sub-buckets (`d3d11_rtx.cpp:5830`–`5910`) already show the
   cost is dominated by cache-*miss* SCANS and the cls12 reconstruction, not by the pure
   decode — so Phase B is probably a small fraction, and deferring it would add threading
   risk for little gain.

Do NOT defer Phase B "blind." A wrong read of what is Phase-A state vs Phase-B decode is
a torn/stale matrix → geometry in the wrong place / vanishing — the exact artifact class
this project has spent months on.

---

## 3. The honest bt_extractXf win: reduce serial cost, don't parallelise

From the sub-buckets (`d3d11_rtx.cpp:5830`–`5910`) the cost is:
- `xt_projScan1` — first-draw / cache-miss projection scan (4 stages × 14 cbuffers ×
  up to 512 windows × classifyPerspective) — ~66 ms/frame on the miss path.
- `xt_w2vsFullScan` — view-matrix discovery full scan (paths 6/7/8/9) on cache miss.
- `pvrCls12` — the cls12 live-cb2 read + worldToView reconstruction — ~67 ms/frame.
- These are amortised by `m_projSlot`/`m_viewSlot` and the **VsLocCache** (per-VS
  slot/offset cache, `d3d11_rtx.cpp:5868`), which already cuts the ~25%-of-draws VS-swap
  thrash. Further real wins come from **extending per-VS caching / cheaper reconstruction**,
  not from moving work to threads.

Recommended next actual optimisation (separate from batching): profile with VsLocCache
hit-rate (`s_perfXtVsLocCacheHits/Misses`) and target the remaining miss population.

---

## 4. Synchronous invariants carried by the batch (do NOT move these off-thread)

These are the pieces each deferred stage had to leave on the game thread. They are the
"Phase A" of the whole SubmitDraw.

- **Material** (`hoistSyncMaterialFields`, `d3d11_rtx.cpp:~13830` in the working tree):
  `sourceIsUnlitUI` (consumed by the VGUI texcoord remap) and `blendMode` (consumed by
  decal detection) are read by the game thread *before* the commit, so they are computed
  synchronously; only the full `FillMaterialData` compute defers.
- **Native-raster / UI decision:** `m_lastDrawFilteredAsUI` / `m_lastClassifierSaidUi`
  are set by the classifier and read by `OnDraw*` to decide native rasterisation. Always
  synchronous.
- **Hashing:** `descHash` (`hashGeometryDescriptor` + bone-instance/base mixing) and
  `layoutHash` (`hashVertexLayout`), plus `hashStart`/`hashCount`, are resolved at capture
  time; only the byte-hashing of position/texcoord/index defers.
- **Skinning:** `dcs.skinningData.boneHash` — the change-detector that gates
  `kUpdateBVH` vs `kUpdateInstance` — is computed synchronously at the skin site
  (`d3d11_rtx.cpp:~24317`), from the source bytes. Only the float3x4→Matrix4 palette
  *materialisation* defers, and only when `needPalette` is true (false on TF2 → usually
  a no-op).
- **Transforms:** everything in §1 (the entire classifier + its 30+ in-SubmitDraw
  consumers).

---

## 5. Cost / design decisions worth NOT re-deriving (or reverting)

### 5a. DrawCallState made MOVABLE — this is the real "copy → move" win
`DrawCallState` declared its copy ctor + copy assignment `=default` (`rtx_types.h:989–990`),
which per C++ rules **implicitly suppressed the move ctor/assignment**. So the emit site's
`m_context->EmitCs([params, dcs = std::move(dcs)]…)` — whose comment claims a
"pointer/refcount-steal (no atomics, no 16KB bone copy)" — was **silently binding
`std::move(dcs)` to the copy** and deep-copying the whole DrawCallState (the up-to-16 KB
`skinningData.pBoneMatrices` + the Rc<DxvkBuffer> position/index/texcoord/color/bone set)
**every draw**. Fix: added `DrawCallState(DrawCallState&&) = default;` +
`operator=(DrawCallState&&) = default;` (`rtx_types.h:991`). Verified the only two
`std::move(dcs)` sites (the batch-arena collect and the EmitCs emit) are **terminal**
(dcs unused afterward), so enabling move is hazard-free, and it fixes the original emit
path too (`RasterGeometry`/`SkinningData`/`RasterBuffer` declare no special members, so
their moves genuinely pointer-steal). **This — DrawCallState — is the copy that was the
bottleneck; the answer was to enable move, not to change how anything is stored.**

### 5b. Bone bytes: still a COPY (once), then MOVE — and it is *not* a measured bottleneck
Clarifying a point that gets conflated: the skinned **bone source bytes** are still
**copied** — `pendSkin.boneBytes.assign(boneReadPtr, boneReadPtr + maxBones*48)`
(`d3d11_rtx.cpp:~24327`). You cannot "move" them: the source is a **raw pointer**
(`boneReadPtr`, into `m_fullBoneCache`), not an owning container, so there is nothing to
steal from. What we DO move is the **pending→item transfer**:
`item.skinJob = std::move(arena.pendSkin)` moves the `std::vector<uint8_t>` (no second
copy). So the flow is: **one** copy from the raw source (unavoidable), then a vector
**move** into the arena item.

Why copy at all instead of pinning + reading at flush (like hash/bbox do)? Deliberate
**safety** choice: a byte copy has **no DxvkBuffer pin to leak** if the draw is rejected
before the commit point. Combined with the fact that the palette build only runs when
`needPalette` (`legacySkinningWillRun || maxBones==1 || captureActive`) — **false on TF2**,
which skins GPU-side — this copy path usually never executes. It was NOT profiled as a
bottleneck; it is a rare, ≤12 KB, safety-first copy. Do not "optimise" it into a pinned
read without first proving it fires and matters.

### 5c. Capture-at-commit vs capture-at-site
- **Material** snapshot is captured **at the commit point** (`d3d11_rtx.cpp:~27780`),
  because PS state is provably **stable across SubmitDraw** (grep-verified: no
  `m_state.ps.{shader,constantBuffers,samplers,shaderResources}` reassignment in the
  function body). Capturing at the commit keeps the arena **1:1 with real commits** (past
  every early-return) and there is no pin to leak on abort.
- **Hash / bbox / skin** inputs are only available **mid-draw** (the hash site at
  `~20964`, the bbox site at `~26960`, the skin site at `~24325`), so they are captured
  into a **per-draw pending slot on the arena** (`GeometryBatchArena::pend{Hash,Bbox,Skin}`)
  and moved into the item at the commit. `resetPending()` runs at **SubmitDraw entry**
  and releases any pins left by a draw that was rejected before it reached the commit —
  this is the leak guard for the hash/bbox buffer pins.

### 5d. Pin lifetime is longer than the per-draw path
Hash/bbox pins (`incRef` + `acquire(Read)` on the position/texcoord/index DxvkBuffer) are
now held from capture until the **frame-end** parallel-for (vs ~immediately in the
per-draw future path). They are released in `runRange` (or `resetPending` on abort). This
keeps more dynamic-buffer slices alive per frame → higher peak pinned-slice memory,
bounded per frame and freed at flush. `acquire(Read)` is a cheap atomic, not a lock
(`dxvk_resource.h:51`) — pinning is not the cost; the count of held slices is the memory
consideration. Most geometry is immutable (no slice recycling), so this is minor.

### 5e. finalize contracts
- `finalizeGeometryHashes` (`rtx_types.cpp:369`) now takes a **pre-computed branch**:
  future valid → `.get()`; else if position hash already non-empty → use it; else
  invalid. And the matching `assert(geoData.futureGeometryHashes.valid())`
  (`rtx_context.cpp:4517`) was relaxed to `|| hashes[VertexPosition] != kEmptyHash`.
- `finalizeGeometryBoundingBox` and `finalizeSkinningData` already accepted a
  directly-populated result (no future), so no change was needed there — the batch just
  writes `boundingBox` / `pBoneMatrices` and leaves the future invalid.

### 5f. Two copies of each compute body
`runBatchHashJob` / `runBatchBboxJob` / `runBatchSkinJob` (`d3d11_rtx.cpp:~13700`) are
**byte-identical mirrors** of the per-draw worker bodies (`ComputeGeometryHashes`, the
bbox Schedule lambda, the inline palette build). The per-draw paths were left untouched
(proven), so there are now **two copies of each** — they MUST be kept in sync if either
is edited. (A future cleanup could route the per-draw path through the shared function,
but that modifies proven code and was deliberately not done here.)

### 5g. Command-stream reorder (Step 1, load-bearing)
The batch reorders all geometry commits to **frame end** (`flushGeometryBatch()` at the
top of `EndFrame`, `d3d11_rtx.cpp:~31376`). Verified safe: `commitGeometryToRT` only
*builds* RT scene state consumed atomically at `injectRTX`, and the sole `injectRTX`
trigger is the EmitCs at `d3d11_rtx.cpp:37186` (via `rtx->endFrame(...)`), which runs
**after** the flush in the same function. `MaybeEarlyInjectForUITexture` only logs +
sets a flag (it does NOT emit a mid-frame injectRTX), so the frame-end flush is the
single correct funnel. **Immediate context only** — deferred contexts never call
`EndFrame`, so they keep the per-draw EmitCs path (gated by
`m_context->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE`) and cannot orphan the arena.

### 5h. Move-not-noexcept
`DrawWorkItem` / `DrawCallState` moves are not marked `noexcept`, so a vector realloc
would copy instead of move. `m_geoBatch->items.reserve(2048)` at SubmitDraw entry avoids
mid-frame realloc in steady state (~1050 draws/frame), so this never bites. Don't add
`noexcept = default` blindly — a member with a throwing move would make it ill-formed.

---

## 6. Flags (all under the master gate; master default OFF)

- `rtx.batchSubmitDrawStages` — master (default **false**). On = collect + frame-end
  parallel-for + reorder. Supersedes `rtx.deferMaterialCompute`'s per-draw material future.
- `rtx.batchHashes` / `rtx.batchBoundingBox` / `rtx.batchSkinning` — per-stage
  (default **true**). Turn one OFF to bisect that stage back to its per-draw path at
  runtime, no rebuild.
- (No `batchTransforms` flag exists — Step 3 is not implemented; see §1.)

---

## 7. Measurement

- `[BatchSubmitDraw]` heartbeat: `itemsPerFrame`, `parallelForMsPerFrame` (Phase B
  dispatch+join wall), `workers`.
- For any Step-3 attempt: add the `ExtractTransforms` Phase-A/Phase-B split timers FIRST
  and confirm Phase B is a meaningful share before touching the classifier.
- The decisive metric is frame time, not any bucket. `finalBlit ~70ms` and
  `entry_tailToBranch` are downstream; confirm the serial SubmitDraw reduction actually
  shows at the frame level and isn't swallowed GPU-bound.

---

## 8. Do NOT re-attempt

- **Deferring `ExtractTransforms` wholesale to the batch** — its output is consumed
  synchronously across the rest of SubmitDraw (§1). Not a stage.
- **Deferring Phase-B transform decode "blind"** without measuring its share and proving
  which reads are Phase-A cache state vs pure decode (§2).
- **"Optimising" the bone-byte copy into a pinned read** — it is rare (needPalette=false
  on TF2), tiny, and copy is the safe choice (no pin to leak); prove it fires first (§5b).
- **Marking DrawCallState/DrawWorkItem moves `noexcept = default`** without checking every
  member's move is noexcept (§5h).
- **Editing one of the shared vs per-draw compute bodies without the other** (§5f).
