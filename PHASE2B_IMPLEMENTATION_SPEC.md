# PHASE 2B IMPLEMENTATION SPEC — sharded instance processing
**Written:** 2026-08-15 (night build). Implements THE_OPTIMISATION_PLAN_2.md Steps 1-6.
**Gate:** `rtx.shardInstanceProcessing` (default True; False = the proven CS-side path, byte-identical).

## 0. The seam, as actually built

The plan says "move geom+inst off dxvk-cs". Mapping (6-agent deep-read, 2026-08-15) found the
exact cut that preserves every invariant:

```
GAME THREAD, flushGeometryBatch() at EndFrame:
  Phase B     mat/hash/bbox/skin parallel-for            (existing, untouched)
  CS DRAIN    SynchronizeCsThread(SynchronizeAll)        overlapped with Phase B's workers
  PRE-PASS    ordered, arena order, per item:            camera classify -> route -> fate ->
              drawCallCache.get -> stamps -> shard build
  SHARDS      parallel by BlasEntry: geom-DECIDE + inst find/update (hit path)
  TAIL        ordered, arena order: deferred spatial-map writes, miss continuations
              (add/migrate/portal draws re-run the sequential path verbatim),
              decal sort order, ray-portal data
  Phase C     ordered emit (existing loop, lambda now carries the ShardedDrawInfo)

DXVK-CS, per emitted draw, in draw order (commitGeometryToRT):
  stash copy + gpuCapture rebind                          (untouched, positional)
  consume: skip camera re-classify, skip get(), inject precomputed ObjectCacheState,
  RECORD: the processGeometryInfo bake switch verbatim (GPU cache calls + allocations),
          updateBufferCache, surface buffer binding for every produced instance,
          smooth-normals/legacy-skin dispatches, billboards/beams, OMM callback,
          blas.input buffer fixup (post-rebind slices)
```

**Why the drain is the linchpin.** All scene state (instances, spatial maps, caches) is read by
dxvk-cs (commit records, injectRTX/prepareSceneData/GC) and now written by the game thread.
The drain at flush start creates strict alternation: CS finishes consuming frame N (including
GC) before the game thread mutates for frame N+1, and the game thread finishes all mutation
before Phase C hands frame N+1 to CS. No other synchronization is needed; pBlas/RtInstance
pointers can be used freely on the CS side of the boundary. The drain overlaps Phase B's
worker execution, so its cost is hidden behind work that already exists.

## 1. Routing (pre-pass), per item in arena order

Everything is classified BEFORE the drawCallCache is touched:

| route | condition | effect |
|---|---|---|
| kLegacyCS | mesh replacements, terrain-bake candidates, sky (tryHandleSky may SkipSubmit), external draws, **unknown camera, unready futures, invalid geometry** | ONLY camera classified at flush; the full legacy CS path runs (get/geom/inst on CS inside the drained-protected window ordering) |
| kSharded | everything else (the ~97% steady-state) | full flush-side processing |
| kSharded, no shard | ignored material | pre-pass has already mutated the fog/replacement/material caches for this draw, so the CS side must NOT re-run that prologue: it takes the slim path and stops at processDrawCallState's own pre-cache ignored return. No cache touch, no stamps, no instance work, `pBlas == nullptr` (unreachable — the ignored return precedes every p2b deref) |

**AS BUILT, correcting this table's first draft:** there is no `kDropped` route — the value was
defined and never assigned, and has been removed. Unknown-camera goes kLegacyCS (not dropped),
ignored material is the kSharded-no-shard row above, and buffer-cache overflow is **not a per-draw
route at all**: `shardingAdmissible` is a whole-frame gate read from `m_bufferCacheLastFrameCount`,
so one near-limit frame sends EVERY draw legacy. Deliberately coarse — the legacy path keeps its
exact per-draw overflow check.

Camera classification (`processCameraData`) runs in the pre-pass for **every** item in arena
order — exact CameraManager op order is preserved — and is skipped on CS via
`ShardedDrawInfo::cameraDone`. Sky/terrain/replacement decisions are made with the same
predicates the CS path uses (`getReplacementsForMesh` lookup, terrain category test, sky
category + cameraType), so kLegacyCS never diverges from what CS would have decided.

`blasFirstDrawOfFrame` is captured per item BEFORE the pre-pass stamps
`frameLastTouched`/`noteDraw` — it feeds the geom decision (onSceneObjectUpdated's
same-frame-dup fast path reads pre-stamp state today).

## 2. Geom decide/record split

`SceneManager::processGeometryInfo<isNew>` decision block (hash compares, bone-base override,
pendingSrcBake recovery, smooth-normals promotion) is extracted into
`computeGeometryCacheState<isNew>(drawCallState, inOutGeometry, firstDrawOfFrame-aware caller)`
— ONE implementation used by both sides. The flush shard calls it and stores the result in the
sidecar; the CS record step injects it (`processGeometryInfo` gains an optional precomputed
result) and runs the ENTIRE bake switch + updateBufferCache verbatim on CS, before its own
mutations, exactly as today. rtx_geometry_utils is untouched.

Flush-side inst consumers that need post-bake facts use the decided result predictively
(e.g. hasPreviousPositions := result==kUpdateBVH/KBuildBVH || previousPositionBuffer already
defined).

## 3. Instance phase on workers — divergence points

A thread_local `t_shardedInstancePhase` (+ a per-item deferral context) is set by the shard
task. When it is false (CS, tail, legacy) every path below is byte-identical to today.

When true:
1. **Spatial-map writes defer.** `RtInstance::onTransformChanged` (move) and `teleport`
   (insert) append a `DeferredSpatialOp{instance, targetBlas, centroid, transform, propId,
   precomputedMatrixHash}` to the item's op list instead of touching the map. The tail applies
   ops in arena order, reading/writing `m_spatialCacheHash` at apply time so multi-op chains
   resolve. During the parallel phase every SpatialMap is therefore read-only, which is what
   makes the migration path's cross-shard sibling reads safe.
2. **Miss defers.** `findSimilarInstance` returning null (and the portal-teleport match, and
   the cross-BLAS migration candidate) does NOT add/migrate/teleport on a worker. The impl
   returns a defer sentinel; the item is marked `needsTailContinuation`; the tail re-runs the
   full sequential impl (find again — the maps have advanced — then migrate/add/update inline).
   Steady state addedPct=0: this path is cold. Instance IDs are therefore assigned in arena
   order — deterministic, matching today.
   **CORRECTION — this claim originally covered material cache indices too, and that half is
   false.** `createSurfaceMaterial`'s miss path is NOT deferred: it runs inside the shard under
   the escape lock, so a NEW material's index in `m_surfaceMaterialCache` reflects shard
   completion order, not draw order. [MatChurn] measures matNew=2-4 every frame, so this is the
   warm path, not a load-time curiosity. Believed benign — every lookup is by hash and nothing
   depends on the numeric value surviving across frames — but it does NOT satisfy plan Sec 7's
   "new material registration, so cache indices are assigned deterministically", and a
   run-to-run index diff is expected rather than a symptom. Making it deterministic means
   hoisting material creation out of `onInstanceUpdated`; that is a separate change.
3. **Fanout**: same treatment per placement. If any placement defers, the pushInstanceRecords
   record refresh for that batch is skipped this frame (rebuilt next frame; verify-safe).
4. **Surface buffer binding defers to CS.** `processInstanceBuffers`/`applySurfaceBufferBinding`
   writes (surface.*BufferIndex etc.) are skipped on workers — the per-frame m_bufferCache tape
   is CS-domain and only valid after the CS-side updateBufferCache. The CS record step binds
   every instance in `ShardedDrawInfo::instances` after the tape is updated.
5. **Billboards/beams + OMM callback route to CS.** createBillboards reads mapped buffer
   contents (positional) and m_billboards/OMM maps are CS-domain. The worker records
   eligibility + the event-fanout args in the sidecar; the CS record step replays them.
6. **Decal sort order defers.** Worker sets `wantsDecalOrder`; tail assigns
   `m_decalSortOrderCounter++` in arena order (deterministic, matches draw order).
7. **Global vectors under the Sec-6 lock**: m_playerModelInstances / m_viewModelCandidates
   push_back (their lazy frame-clear moves to the pre-pass); capturer setInstanceUpdateFlag.
8. **RayPortal materials defer**: `processRayPortalData` runs in the tail in arena order.
9. **Escapes under ONE `std::mutex SceneManager::m_shardEscapeMutex`**: createSurfaceMaterial
   non-memo path (incl. trackSampler/trackTexture/extension-cache track), m_bufferCache.track
   is NOT locked (CS-domain now), fanout-record bookkeeping (m_fanoutRecords/m_fanoutOrdinals).
   Never taken on the memo-hit path.
10. **Material memo returns a copy.** SurfaceMaterialMemo now stores an RtSurfaceMaterial copy;
   the memo hit returns a reference to the slot's copy, never into
   m_surfaceMaterialCache.m_objects (a std::vector a concurrent locked insert may reallocate —
   [MatChurn]: matNew=2-4 EVERY frame, so this is a live hazard, not a cold one).
11. **Counters** on shared paths become relaxed atomics (m_matLookupCount etc.,
   [FindStage] census, vanishDiag increments); per-frame probe statics on worker paths get
   the same treatment or stay thread_local where fragmentation is acceptable.

## 4. What stays exactly where it was

- The gpuCapture copy lambda at SubmitDraw position; Block A/B of commitGeometryToRT.
- tryHandleSky / bakeTerrain / drawReplacements / submitExternalDraw (kLegacyCS route).
- All of prepareSceneData / injectRTX / GC on dxvk-cs.
- The Phase C emit loop and its ordering (plan Sec 10).
- The inline fallback + [BatchJoin] watchdog (plan Sec 11 / A.8).
- MeshTrace funnel stages (Submitted still recorded at submitDrawState entry on CS).

## 4b. Implementation deltas (as built, 2026-08-15 night)

- **Per-instance pending ops.** The CS-record work is recorded per updateInstance
  call (`ShardedDrawInfo::PendingInstanceOps`: bindBuffers / billboard / omm /
  rayPortal + the event args), not per draw — fanout placements keep exact
  per-placement event fidelity.
- **allowMiss mode.** The ordered tail runs with the phase machinery ON but
  `t_shardPhase.allowMiss=true`: CS-domain work still defers into pendingOps,
  while creation/migration/teleport and spatial-map writes run inline (the tail
  is single-threaded and later placements must see earlier inserts).
- **One decision function.** `computeGeometryCacheState<isNew>` is called by the
  flush shard AND by processGeometryInfo on the CS record step — the CS side
  recomputes rather than trusting the sidecar (same inputs by the drain
  contract, so they cannot disagree, and the sidecar carries the result only for
  the flush-side consumers like the hasPreviousPositions prediction).
- **Ignored materials** stay route=kSharded with no shard membership: no cache
  touch, no stamps; the CS slim path hits the same pre-cache early return.
- **Camera classify only for finalize-ready draws** — an unready draw never
  classified on the CS path either (it is inside the futuresReady gate).
- **blas.input fixup**: the flush-side `pBlas->input = dcs` copy is pre-rebind;
  the CS record re-points the six geometry buffers + capture/stash fields at the
  rebound ones, for both updated and freshly-allocated entries.
- **Memo returns a copy from ALL paths** (not just the hit): even a locked path
  cannot return a reference into the cache vector, because the caller consumes
  it after the lock releases. Contract: valid until the same thread's next
  createSurfaceMaterial call.
- **Overflow admission**: `m_bufferCacheLastFrameCount` (captured in onFrameEnd
  before the clear) within 1/8th of kBufferCacheLimit routes everything legacy.
- **[FindStage]** counters are now file-scope relaxed atomics with a CAS-elected
  per-frame emit (one line, whole-frame census, worker-safe).

## 4c. Deltas added 2026-08-18 (first-compile session)

- **SHARDS ARE PACKED INTO BUNDLES.** The first build scheduled one task per
  shard — ~460 `Schedule()` calls per frame over ~1350 draws (~2.9 draws each),
  which walks straight into plan Sec 13's tombstone: 119 tasks was measured at
  frames/3s 54 → 17, because per-task pool cost is a mutex round trip plus a
  notify_one into 30 condvar-sleeping workers. `processDeferredDrawBatch` now
  takes a `maxTasks` budget (the caller passes the worker count, matching Phase
  B's `kChunksPerWorker=1`) and packs whole shards into contiguous, item-count
  balanced bundles. **The ownership partition is unchanged** — a BLAS never
  spans two bundles — only the scheduling granularity is. `[Shard2b] bundles=`
  reports the task count; it must track workers, never shards.
- **The shard join is under the [BatchJoin] watchdog.** Same published state,
  plus `phase=shard2b|phaseB` on the STUCK line so the two joins are
  distinguishable. `futures=` is the bundle count; the itemsDone heartbeat bumps
  at bundle entry and exit, which is what separates "a bundle is running but
  slow" from "no bundle ever started" (the likely shape of an escape-lock hang).
- **`Route::kDropped` removed** — defined, never assigned. See the Sec 1 table.
- **RtInstance 944 → 952.** `m_spatialOpPendingFrame` is 4 bytes + 4 padding; the
  copy-ctor size guard now records the ruling (NOT copied — it travels with
  `m_spatialCacheHash`, which is also not copied; a clone is in no map and no
  deferred op names it).

## 5. Verification (plan Sec 15)

[Perf.Report] inst=/matNew=/uniqueBlas/drawsIn/drawsCommit, [ReapJoin], [MeshTrace] — all
unchanged mechanically. New heartbeat `[Shard2b]` (3s window): sharded/legacy/ignored counts,
shard count, bundle count, deferred count, and pre/par/tail us. Expected: shards≈uniqueBlas,
bundles≈workers, deferred≈0 steady-state, [BatchJoinSplit] tail rises (absorbing the work),
[CommitRT] submitMs collapses to the record residue.

**Two expectations in the first draft of this section were wrong; do not chase them.**

- **`[ProcDCS] geomMs` does NOT collapse.** Only the cache-state DECISION moved to the flush
  side. `processGeometryInfo` — the interleave, the bake dispatches, `updateBufferCache` — still
  runs on dxvk-cs at this draw's stream position, because that work reads mapped buffer contents
  and records GPU commands, which is exactly what plan Sec 0.1 says must stay positional. Plan
  Step 5's "5-6 ms" is therefore NOT part of this phase's win, and plan Sec 18's headline
  "16-23 ms" should be read as `inst` (11-17 ms) minus the CS-side record residue (per-instance
  buffer binds, billboards, OMM, ray portals). `instMs` is the number that must fall.
- **`matNew=0` (plan Sec 15) contradicts [MatChurn]'s measured 2-4 per frame.** The measured
  value is the real one — it is why the material memo had to start storing a copy. Read matNew
  as "unchanged versus the option-off run", not as zero.

Also: the plan Sec 3 check "sum of shard sizes == drawsCommit" does not hold as built. Legacy
and ignored draws are outside every shard by design; the identity is
`sharded + legacy + ignored == itemsPerFrame`.
