# PHASE 2B — COMPLETION PLAN
**Written:** 2026-08-18. Branch `architecture-overhaul`, all changes UNCOMMITTED.
**Implements:** THE_OPTIMISATION_PLAN_2.md Steps 1-6. Design record: PHASE2B_IMPLEMENTATION_SPEC.md.

## RULE FOR THE NEXT SESSION
**NEVER use the Agent or Workflow tools. No subagents, no fleets, ever.** Do every check
inline with grep/sed/cat. The review below is a manual checklist, not a delegation list.
Reason: two workflow runs burned ~2.5M tokens and produced ZERO findings (both died on the
session cap). The work below is small, surgical reading — it does not need parallelism.

---

# PART 1 — WHAT IS ALREADY BUILT (do not redo)

Gate: `rtx.shardInstanceProcessing` (rtx_options.h, default **True**). False = byte-identical
pre-2b path, no rebuild needed. Also requires `rtx.batchSubmitDrawStages=True` + immediate ctx.

## 1.1 The architecture as built

```
GAME THREAD, flushGeometryBatch() (d3d11_rtx.cpp:~29912):
  Phase B      mat/hash/bbox/skin parallel-for                   (pre-existing, untouched)
  CS DRAIN     SynchronizeCsThread(SynchronizeAll)               NEW - overlapped with Phase B
  PRE-PASS     ordered, arena order (SceneManager::processDeferredDrawBatch)
  SHARDS       parallel by BlasEntry (runShardedDrawItem)
  TAIL         ordered, arena order (runShardedDrawTail)
  Phase C      ordered emit, lambda now carries ShardedDrawInfo p2b
DXVK-CS, per draw (commitGeometryToRT):
  stash copy + gpuCapture rebind                                 (untouched, positional)
  slim consume: skip camera/get/stamps; RECORD bake + updateBufferCache
                + per-instance buffer rebind + billboards + OMM + portals
```

**The drain is the linchpin.** It creates strict alternation: CS finishes consuming frame N
(incl. injectRTX + GC) before the game thread mutates for frame N+1, and the game thread
finishes before Phase C hands frame N+1 to CS. That is why only ONE cold mutex is needed and
nothing on any hit path synchronizes.

## 1.2 Files changed (~1650 insertions, `git diff --stat`)

| file | what |
|---|---|
| `rtx_types.h` | `DeferredSpatialOp`, `ShardedDrawInfo` (+ nested `PendingInstanceOps`) |
| `rtx_options.h` | `rtx.shardInstanceProcessing` (1 line) |
| `rtx_instance_manager.h` | `ShardedInstancePhase` + `t_shardPhase` decl, `m_shardEscapeMutex`, `m_spatialOpPendingFrame`, 5 new method decls |
| `rtx_instance_manager.cpp` | `t_shardPhase` def; divergence points; 5 new methods |
| `rtx_scene_manager.h` | `ObjectCacheState` made public, `ShardedDrawBatchItem`/`ShardScheduleFn`, driver decls, memo `material` copy, atomic mat counters, `m_bufferCacheLastFrameCount`, `t_shardedConsume` decl |
| `rtx_scene_manager.cpp` | `t_shardedConsume` def; `computeGeometryCacheState` extraction; slim consume paths; THE DRIVER (3 functions) |
| `rtx_context.{h,cpp}` | `commitGeometryToRT` 3rd param `ShardedDrawInfo*`, RAII consume scope, camera-skip |
| `d3d11_rtx.cpp` | `DrawWorkItem::p2b`, CS drain, driver call, lambda capture, `csDrainUs` heartbeat |

## 1.3 Key mechanisms (find them all by grepping `[Phase2b]`)

- **`computeGeometryCacheState<isNew>`** (rtx_scene_manager.cpp): the cache-state DECISION
  extracted from `processGeometryInfo` — ONE implementation, called by the flush shard AND by
  `processGeometryInfo` itself on the CS record step. Carries `outForcedByPendingSrcBake`.
- **`t_shardPhase`** (thread_local): `info` non-null = a worker is in the shard phase.
  `allowMiss=true` = the ordered tail (CS-domain work still defers; creation/migration/map
  writes run inline). Null everywhere else -> byte-identical legacy behavior.
- **`t_shardedConsume`** (thread_local, CS): non-null during a batched commit; route kSharded
  makes submitDrawState/processDrawCallState take the slim path.
- **Deferred ops**: spatial-map move/insert + decal sort order recorded per draw, applied by
  the tail in arena order (`applyDeferredSpatialOp`).
- **Deferred misses**: `findSimilarInstance` returning null sets `deferredThisDraw`; the draw
  (or fanout placement) re-runs sequentially in the tail. Cold at addedPct=0.
- **Per-instance `PendingInstanceOps`**: bindBuffers / billboard / omm / rayPortal + event args,
  replayed on CS in processDrawCallState.
- **One escape lock** `InstanceManager::m_shardEscapeMutex`: new materials, fanout records,
  capturer flags, player/VM vectors, ZigGun tag. Never on the memo-hit or find-hit path.
- **Routing**: sky/terrain/replacements/unknown-camera/invalid-geom/unready-futures/
  buffer-cache-near-overflow -> `kLegacyCS` (unchanged CS path, draw order). Ignored material
  -> `kSharded` with no cache touch/stamps/shard membership.

## 1.4 Verification state (weak — this is the honest part)

- 6-agent deep-read of every touched subsystem BEFORE design (that part was worth it).
- Inline self-review found and fixed 3 real bugs: futures field paths (`dcs.geometryData.*`),
  camera classify only for finalize-ready draws, and the tail's stale-bind hazard (`allowMiss`).
- Brace-delta check vs HEAD: every edited file matches HEAD's delta exactly.
- **NOT compiled. NOT run. NOT reviewed.** Zero review findings exist — not "clean", *unchecked*.

## 1.5 SESSION 2026-08-18 — first compile attempt + architecture conformance pass

**First real compiler output.** One error, and it was the size guard doing its job:
`RtInstance` 944 → 952 (`m_spatialOpPendingFrame`, 4 bytes + 4 padding). Ruling recorded at the
static_assert: **NOT copied** — it is one value with `m_spatialCacheHash`, which is also not
copied; a clone is in no map and no deferred op names it, so inheriting the pending-frame would
make its first `onTransformChanged` see a `chained` write it never queued. Only
`rtx_instance_manager.cpp` reached the compiler — the other three files are still unbuilt.

**Conformance pass against the two design docs** (full diff read, both docs read). Findings and
what was done:

| # | finding | action |
|---|---|---|
| 1 | ~460 `Schedule()` calls/frame (one per shard) vs Sec 13's tombstone at 119 | **FIXED** — shards packed into `maxTasks` bundles (one per worker); ownership partition unchanged |
| 2 | `geom` never moved; only its decision did | **DOCS CORRECTED** — the bake is positional per Sec 0.1; geomMs will not collapse |
| 3 | the drain is an uncosted serialization point | **DOCUMENTED** in the plan's annex |
| 4 | the pre-pass carries more serial work than Sec 3 specifies | **DOCUMENTED**, measured as `preUs` |
| 5 | new-material indices are not deterministic (Sec 7 unmet) | **DOCUMENTED**, believed benign |
| 6 | `Route::kDropped` defined, never assigned | **REMOVED** |
| 7 | no watchdog on the shard join | **FIXED** — [BatchJoin] now covers it, `phase=shard2b` |
| 8 | Sec 3's "sum of shard sizes == drawsCommit" false as built | **CORRECTED** to `sharded+legacy+ignored == items` |

Checked and NOT a defect: the ignored-material draw carries `pBlas == nullptr`, but
`processDrawCallState` returns at its pre-cache ignored check (rtx_scene_manager.cpp:3561) before
the p2b block (:3591), so it cannot deref.

Still true: **not compiled past file 1 of 4, not run, Step 2 not done.**

---

# PART 2 — WHAT THE NEXT SESSION MUST DO

In order. Steps 1-2 before any build.

## STEP 1 — COMPILE-COHERENCE PASS (inline, ~30 min of reading)

Written without a compiler. Read these exact spots:

1. **Brace surgery sites** (each wraps legacy code in an `else {`): grep `[Phase2b]: end of the`
   in rtx_instance_manager.cpp — the teleport deferred/inline split and the decal-order split
   (the latter also contains an `#if !NDEBUG` block). Also the fast-path OMM loop else-wrap
   (grep `ommPendingWork`). Brace totals already match HEAD, so any error is a *placement*
   error, not a count error.
2. **Signatures vs call sites**:
   - `commitGeometryToRT` 3 args — 2 call sites (d3d11_rtx.cpp ~30234, ~51531). Checked OK.
   - `computeGeometryCacheState` — .h decl (default arg `bool* = nullptr`), definition, 2
     explicit instantiations, 2 call sites. Definition must NOT repeat the default arg.
   - `processGeometryInfo` body after extraction: confirm `input`, `result`,
     `forcedByPendingSrcBake`, `needsSmoothNormals` are still declared where used.
3. **Atomic conversions** — every read site needs `.load()`: `m_matLookupCount`,
   `m_matMemoHitCount`, `m_matPreMissCount` (fixed at the [MatChurn] sampler), `FindStageAgg`,
   `sGunF/sGunC`, `[VM.instance]` `sLastF/sCount`.
4. **Types**: `std::optional<RtSurfaceMaterial> memo.material` (RtSurfaceMaterial has no default
   ctor — that is why optional); `<optional>`/`<functional>` added to rtx_scene_manager.h;
   `PendingInstanceOps { &currentInstance }` aggregate init with NSDMIs; `Future<void>` default
   ctor exists (util_threadpool.h:241).
5. **Visibility**: `ShardedDrawInfo` reaches rtx_context.h via rtx_camera_manager.h ->
   rtx_types.h (checked OK). `SceneManager::ObjectCacheState` is now public — confirm the
   `public:`/`private:` split around it did not orphan another member.
6. **d3d11_rtx.cpp**: `d3d11_context_imm.h` include; `DxvkCsThread::SynchronizeAll`;
   `m_context->m_device->getCommon()->getSceneManager()`; the EmitCs lambda move-captures a
   move-only `ShardedDrawInfo` (explicit move ctor/assign + deleted copies — OK).

## STEP 2 — CORRECTNESS REVIEW (inline reading, six dimensions)

Read `git diff -- src/` against these. One at a time, no delegation.

**A. Races.** Contract: workers own disjoint BlasEntries; ALL SpatialMaps read-only; escapes
locked; CS drained. Walk `processSceneObjectImpl` -> `updateInstance` ->
`SceneManager::onInstanceUpdated` -> `createSurfaceMaterial` and list every write to
non-shard-owned state; each must be locked, deferred, atomic, thread_local, or cold+guarded.
Suspicious leftovers: probe statics inside `updateInstance` (grep `static ` across its body),
`acquireVsDebugId`, `bindMaterial`'s `m_pResourceCache` fallback, `computeDrawScopedState`'s
`m_fastOpt*`, OMM manager maps (should be unreachable on workers — verify).

**B. CS seam.** Does `SynchronizeCsThread` at that site take `D3D11DeviceLock`, and can
EndFrame's caller already hold it (deadlock)? Check `LockContext` around the EndFrame path.
Then verify nothing between the drain and Phase C emits to CS.

**C. Geometry equivalence.** Diff the extracted decision against the removed inline blocks in
`git diff`: override ORDER (bone-base, pendingSrcBake, smooth-normals), exact conditions, and
that `forcedByPendingSrcBake`'s consumer (the `pendingSrcBakeSingleRetry` site) still sees the
same value in every case. Does anything else in `processGeometryInfo` return `kInvalid` beyond
the two conditions the driver replicates (position defined, vertexCount)? grep `kInvalid`.

**D. blas.input fixup — HIGHEST REMAINING RISK.** The flush copies `pBlas->input = dcs`
PRE-rebind; CS re-points 6 geometry buffers + capture/stash fields. Enumerate consumers of
`blas.input`'s *buffer* fields (grep `input.getGeometryData()` and `.input.geometryData`) and
confirm each is safe with that timing.

**E. Traps (plan Secs 8-15).** Camera re-registration on every path (Sec 9), especially the
window between a deferred miss and the tail (GC runs at injectRTX, after the tail — verify the
order). Emit order untouched (Sec 10). Inline fallback in BOTH parallel-fors (Sec 11); the new
shard join has no watchdog — decide if it needs one.

**F. Driver.** Route completeness: every early-return/mutation in the legacy CS chain is either
replicated in the pre-pass, routed legacy, or unreachable. Check fog-block replication (assert +
logging dropped), replacement lookup (uniqueHashes logging dropped), `determineMaterialData`
at flush vs CS time, and `m_bufferCacheLastFrameCount` capture on a `raytracedThisFrame=false`
frame.

## STEP 3 — BUILD AND FIRST RUN (user compiles; never invoke a build)

1. User builds; fix errors at the Step 1 sites.
2. Run with defaults, then read remix-dxvk.log in this order:
   - `[Shard2b]` present. `shards` ~= `uniqueBlas`. **`bundles` ~= worker count — if it ever
     tracks `shards` instead, the Sec 13 tombstone is back.** `deferred` ~0.
     `legacy` = sky/terrain/replacement volume — this one scales the whole benefit, read it early.
   - `[BatchSubmitDraw] csDrainUs` — if large, the drain is the new cost, not a win.
   - Sec 15: `inst=` ~15.4k unchanged, `[ReapJoin]` not rising, `[MeshTrace]`
     BatchEmitted == Submitted, drawsIn/drawsCommit ratio unchanged. `matNew` **unchanged versus
     the option-off run** — NOT zero; [MatChurn] measures 2-4 every frame.
   - Then timing: `[ProcDCS] instMs` collapses; `[CommitRT] submitMs` collapses;
     `[BatchJoinSplit] tail` rises (expected). **`geomMs` stays flat — the bake never moved
     (spec Sec 5). Do not read that as a failure.**
   - `[Shard2b] preUs` is the new serial cost on the game thread; `parUs`/`tailUs` split the rest.
   - `[Shard2b.legacy]` — always on, emits whenever legacy>0. Says WHY each legacy draw is
     legacy: `terrain` is permanent, `sky` is the four-way over-approximation worth attacking,
     `replacement` is a separate job. **`admit>0` means the whole-frame buffer-cache gate fired
     and nothing was sharded that frame — every other number in that window is then meaningless.**

## STEP 3b — THE POST-2B MEASUREMENTS (Step 0 of the follow-on plan)

These are gated OFF by default. To take them, put BOTH lines in rtx.conf — the second one is
required because `[Perf.GeoChurn]` sits on the logger's default denylist (log.cpp:626) and would
otherwise emit nothing while looking like the code never ran:

```
rtx.logPrepSceneSplit = True
rtx.logDenyTags = -[Perf.GeoChurn]
```

- `[Perf.GeoSplit]` — splits `geomMs` by MECHANISM: `allocUs` (the four `createBuffer` calls,
  movable to the shard workers), `recUs` = bake−alloc (barrier + copies + interleave dispatch,
  PINNED to dxvk-cs by plan Sec 0.1), `tapeUs` (`updateBufferCache`'s nine `track()` calls,
  movable to the ordered tail *after* auditing `SparseUniqueCache::track`), `restUs` (the
  decision + probes the extraction already left). The `MOVABLE=` / `PINNED=` summary at the end
  of the line is the whole answer to "is splitting `processGeometryInfo` further worth it".
  Read it next to `[Perf.GeoChurn]`'s counts — times say what to move, counts say how often the
  path is taken at all.
- `[Shard2b.pre]` — splits `preUs` across the six ordered stages. `camera` and `cacheGet` are
  structurally ordered; `finalize`+`fog`+`hash+repl`+`material` are the Phase B candidates and
  the line totals them as `MOVABLE=`.
- Cost of having them on: ~8 clock reads/draw in geom + ~7 in the pre-pass ≈ 0.8 ms/frame, a
  large fraction of what is being measured. **Take the numbers, then turn them back off** — do
  not leave this on for a timing run.
- Floor check (this bit before): geom is ~4 µs/draw, so the buckets land ~1 µs each against a
  41 ns `steady_clock` read — resolvable, unlike `[markFm]`, whose real per-draw work was
  sub-µs and could not be resolved by any per-draw timer. If a bucket comes back at or below
  ~0.1 µs/draw, treat it as "too small to matter", not as a measurement.
   - Check `clkNs` 32-49 / `xMin=1.00` before quoting absolute ms (v1 C.3b).
3. Any anomaly: `rtx.shardInstanceProcessing = False` (no rebuild), confirm it disappears, diff
   the two runs' logs. That isolates 2b from everything else in the fork.

## STEP 4 — LIKELY FIRST FAILURES (ranked)

1. **Missing/black geometry on some draws** -> blas.input buffer fixup (D), or a route that
   should have been kLegacyCS. Check `[Shard2b] legacy=`.
2. **Instance drift / flicker** -> a skip path stamping without re-registering the camera, or
   the deferred-miss window. Watch `[ReapJoin] starved=`.
3. **Wrong materials** -> memo copy semantics, or `determineMaterialData` reading flush-time
   state that differs from CS time.
4. **Decal ordering artifacts** -> `kDecalOrder` tail application order.
5. **Hang at EndFrame** -> the drain's lock (B).

---

# PART 3 — IF IT HAS TO BE ABANDONED

`rtx.shardInstanceProcessing=False` restores exact pre-2b behavior with the code in place.
`git checkout -- src/` reverts everything (nothing is committed). THE_OPTIMISATION_PLAN_2.md and
PHASE2B_IMPLEMENTATION_SPEC.md stand on their own as the design record.
