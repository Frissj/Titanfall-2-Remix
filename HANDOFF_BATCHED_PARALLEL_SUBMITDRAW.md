# HANDOFF: Batched-parallel SubmitDraw (one sync point, all stages)

Session date: 2026-07-25. Supersedes the per-draw-future material deferral
(see §2). Everything below is in the **working tree**; the material deferral is
committed as a git checkpoint the user made before this work. Every claim has a
`file:line` or a log tag you can re-verify. Where a measurement drove a decision,
the number is recorded.

**Machine caveat (unchanged):** this box swings ~4× on identical code (thermal),
and worker-count changes shift heap contention. Trust **per-draw ratios** and
**mechanism counters** over absolute µs. Normalize any µs bucket by its `*N`
count before comparing windows.

---

## 1. The one fact that frames this whole rewrite

**Per-draw task scheduling is the wrong granularity for SubmitDraw's fine-grained
per-draw work.** The material-compute deferral proved it end-to-end:

- The compute genuinely moved off the game thread — every `fm_*`/`fmm_*` bucket in
  `[Perf.SubmitDraw.acc]` reads **0** (they now accumulate on the worker thread's
  `thread_local`, not the reporting thread).
- But `cvr_fillMat` (the serial cost at the call site) **barely moved**, because
  the *capture* cost replaced the *compute* cost. Measured `matCap_ns` (snapshot +
  heap alloc, game thread) ≈ **18 µs/draw**, i.e. ~the same as the ~18 µs compute
  it offloaded. `matSched_ns` (enqueue) ≈ 2–4 µs/draw.
- **Raising the worker count made `matCap` WORSE** (17.8 µs at 6 workers → 20–36 µs
  at ~14 workers). The only thread-contended thing in the capture is the **heap
  allocator** (`make_shared` per draw). `acquire(DxvkAccess::Read)` is a plain
  atomic increment (`dxvk_resource.h:51`), NOT a lock — pinning is cheap and is
  **not** the bottleneck.

Conclusion: when the cost of handing a work item to a thread (snapshot + `make_shared`
+ `Future` + finalize-join) rivals the work itself, per-draw futures break even at
best and lose under allocator contention. **The fix is not a cheaper per-draw
snapshot — it is to stop scheduling per draw.** Collect per-draw inputs during
SubmitDraw (O(1) append, no alloc, no future), then run **one parallel-for** across
the whole pool at a **single end-of-frame sync point**. Scheduling + allocation
overhead becomes O(threads) per frame instead of O(draws × stages).

The user's framing is correct: because `m_pGeometryWorkers` is the **same pool**
already used by hashing + bbox + material, unifying *all* per-draw deferrable
stages into one batched pass makes the net win compound.

---

## 2. What exists now (material deferral — the prototype to generalize)

All uncommitted-then-checkpointed. It works and renders correctly; keep it as the
reference for "how to run FillMaterialData off-thread," then fold it into the batch.

| Piece | File:line | State |
|---|---|---|
| `rtx.deferMaterialCompute` (default true) | `rtx_options.h` (~351) | live |
| `rtx.geometryWorkerThreads` (0=auto=cores−2) | `rtx_options.h` (~352) | live; **both** pool sites updated (`d3d11_rtx.cpp:4576`, `~17623`) |
| `DrawCallState::futureMaterialData` (`Future<shared_ptr<LegacyMaterialData>>`) + `finalizeMaterialData()` | `rtx_types.h` (~1217/1200), `rtx_types.cpp` (~272/`finalizeMaterialData`) | live; finalized FIRST in `finalizePendingFutures` (before `setupCategoriesForGeometry`, which reads the color-texture hash) |
| `MatSnapshot` + `captureMatSnapshotInto(s, defer)` + `hoistSyncMaterialFields` + `FillMaterialData(mat, snap)` | `d3d11_rtx.cpp` (~13616+) | live |
| Dispatch (hoist sync → 1×`make_shared<MatSnapshot>` → `Schedule` → aliasing `shared_ptr` return; queue-full/flag-off run inline) | `d3d11_rtx.cpp` (~21980) | live |
| Perf split `matCap_ns` / `matSched_ns` / `matDeferN` | `d3d11_rtx.cpp` (~5938 decl, ~21985 timing, reporter + reset) | live |

**Key mechanics proven here that the batch reuses:**
- `FillMaterialData` was parameterized to read **only** from an injected snapshot
  (`const auto& ps = snap.ps;`), plus a `GetCurrentVsPsHashes` shadow-lambda that
  routes the ~10 in-body hash calls to captured hashes. **Grep-verified: the body
  touches no live `m_context` on the deferred path** — every residual `m_context`
  read is a diagnostic gated by `!snap.deferred` or `s_fillMatDiagEnabled`.
- The two fields the game thread reads *before* EmitCs are produced synchronously by
  `hoistSyncMaterialFields`: `sourceIsUnlitUI` (VGUI geometry remap ~22090) and
  `blendMode` (decal detect ~22310). The audit for material found **only** those two
  + a diagnostic `colorTextures[0]` read are consumed on the game thread pre-EmitCs.
- The `getImageHash` GC race is a **ManagedTexture** hazard (`rtx_texture.h:124`);
  FillMaterialData only builds **game-texture** `TextureRef`s (Rc-pinned views), so
  it never touches it. Buffer-based work is safe; only *managed*-texture reads race.
- Dynamic D3D11 constant buffers are `Map(WRITE_DISCARD)`-recycled, so `GetMappedSlice()`
  on the worker returns a DIFFERENT slice than the draw bound. Capture the `mapPtr`
  at collect time and pin the `DxvkBuffer` (`incRef` + `acquire(Read)`), release after
  the read. Copied from `ComputeGeometryHashes` (`13486`).

**Why not just finish the material seam-split:** it would reach zero-alloc for
*material only* while transform-extraction (the biggest serial bucket, see §4) still
runs inline. The batch does all stages in one pass, so do the batch instead.

---

## 3. Target architecture — collect → one parallel-for → writeback

Three phases per frame. The hard sync point is at the END (before scene/TLAS submit).

**Phase A — SubmitDraw (game thread, per draw, CHEAP, no alloc, no future):**
- Do ONLY what must be synchronous:
  - The native-raster-vs-RT classifier decision (`m_lastDrawFilteredAsUI`) — must stay
    on the game thread on cheap inputs (classifier + staged camO + cached w2v). It must
    NOT depend on any deferred result. (Invariant from the prior threading handoff.)
  - The handful of fields read downstream before submit (material `sourceIsUnlitUI` +
    `blendMode`; any geometry-remap gate). Compute these cheaply inline.
  - Pin every buffer the batch will read (position/index/texcoord/bone/CBs): `incRef` +
    `acquire(Read)` once, here.
- Append ONE compact `DrawWorkItem` to a **per-frame arena vector** (`std::vector`
  reserved to steady-state capacity, `clear()`ed each frame — grows once, never
  per-draw-allocates). The item holds: the raw/pinned buffer pointers + strides +
  lengths, the resolved-live snapshot (shader Rc, CB mapPtrs, blend/depth state,
  scalars, hashes), and an index/pointer to the target `DrawCallState`.
- Do **not** create a `Future`, do **not** `make_shared`, do **not** `Schedule` here.

**Phase B — one sync point (before scene submit), ONE parallel-for:**
- `parallel_for(0, items.size())` chunked by `items.size()/numWorkers` so each worker
  processes a **contiguous range** (scheduling is O(workers), not O(items)). Each item
  runs, reading ONLY its snapshot (never live `m_context`):
  1. `ExtractTransforms` (the ~270 ms prize — see §4)
  2. `FillMaterialData` compute (reuse the `(mat, snap)` body verbatim)
  3. skinning (`futureSkinningData` work — bone palette build)
  4. geometry hashing (`ComputeGeometryHashes` body)
  5. bounding box (already deferred; fold in)
  Write results into the item's `DrawCallState` (each item owns a disjoint DCS — no
  cross-item sharing, so no locks).
- Join the parallel-for (the single barrier). Release all buffer pins here.

**Phase C — submit:** finalize categories/hashes and hand the now-complete
DrawCallStates to the scene manager, in original draw order.

**Overhead math to beat:** current ≈ (18 µs capture + 2 µs sched + finalize-join) ×
draws × stages. Target ≈ (cheap append) × draws + (1 parallel-for dispatch) × frame.
The append must be genuinely O(1) — a POD/trivially-movable `DrawWorkItem` into a
reserved vector. If `DrawWorkItem` holds `Com<>`/`Rc<>` (AddRef on move), keep those
to the minimum (shader + a few CBs + blend/depth); the 128-SRV array is the churn
that killed material capture — see §5 for keeping it out.

---

## 4. Stage priority (do the biggest first, but ship them together)

From a steady `[Perf.SubmitDraw.acc]` window (per-draw, ~7000 draws):

| bucket | ~µs/window | what | defer? |
|---|---|---|---|
| `bt_extractXf` | **~280–440k** | ExtractTransforms (cb2/cb3/t30/t31 matrix reads) | **YES — the prize** |
| `sf_o2w`, `w2v_world`, `xt_*` | sub-parts of the above | | with it |
| `cvr_fillMat` | ~150k | material (now = capture cost; will vanish once batched) | YES (prototype done) |
| `bonePalette`, `cbc_rawUv` | ~25k / ~80k | skinning capture + raw-UV decode | YES |
| `bt_hashes` | ~40k | geometry hashing | already deferred; fold in |
| `tail_capture`, `tail_emit`, `emitCs_ns` | ~170k / ~50k / ~50k | scene capture + EmitCs hand-off | mostly must stay; measure |

`bt_extractXf` dwarfs everything. **Transform-extraction is the real win** and it's
per-draw fine-grained too, so it has the *same* per-draw-future overhead problem —
which is exactly why it must go through the batch, not its own futures.
Gate-B (o2w finiteness guard) fires 0× on real geometry, so deferring the matrix
read is hole-free (proven in the prior threading handoff).

---

## 5. Invariants & gotchas (do not rediscover the hard way)

- **The parallel-for body must read ZERO live `m_context`.** Snapshot everything at
  collect time. Enforce with a grep after: the batched worker function must not
  reference `m_context->m_state`. A missed read is a torn/racy read (game thread
  mutates the stage state next draw), not a fixable artifact.
- **Keep the 128-SRV / 64-UAV arrays OUT of the per-draw item.** Copying a full
  `D3D11ContextStatePS` per draw was ~18 µs of `Com<>` churn. Snapshot only the bound
  slots the stage actually reads (material role slots + scoring; transforms read
  specific t30/t31/cb slots). Resolve SRVs the stage needs at collect time (hold the
  `Rc<DxvkImageView>` — game-texture views are immutable and safe off-thread).
- **Buffer pins: pair every `incRef`+`acquire(Read)` with a release** — after the
  parallel-for join, AND on any early-out/queue-full path. `acquire` is a cheap atomic
  (`dxvk_resource.h:51`); pin liberally, but never leak (VRAM/refcount).
- **Dynamic CB slice race:** capture `mapPtr` at collect time (game thread), never call
  `GetMappedSlice()` in the parallel-for. Pin keeps the physical slice valid.
- **Must-stay-sync:** `m_lastDrawFilteredAsUI` (native-raster decision) and the ~2
  downstream-read material fields. Everything else can move to Phase B.
- **Diagnostics** that read `m_context` (vs/rs/one-shot logs) must be gated so they run
  only on the sync path (`!deferred`) or be fed from the snapshot. Material already
  did this with `!snap.deferred` guards — mirror the pattern.
- **Ordering:** the batch must preserve original **draw order** for scene submit
  (decals/transparency/priority depend on it). The arena vector is naturally ordered;
  keep the submit loop sequential over it after the join.
- **Skinning boneHash:** compute the change-detector from source bytes at collect time
  (it gates `kUpdateBVH` vs `kUpdateInstance`) — do not defer that decision.
- **Managed-texture `getImageHash` GC race** (`rtx_texture.h:124`) is the ONE thing you
  must never do off-thread; game-texture views are fine. Material never hits it.

---

## 6. Suggested build order (each step renders, so bisectable)

Do NOT batch all five stages blind. The user wants it done once, but "once" means one
*correct* architecture — land it stage-by-stage behind the arena so each is testable:

1. **Arena + collect/parallel-for skeleton** carrying only material (retire the
   per-draw `futureMaterialData`; material becomes item #2 in the batch). Verify
   `matCap_ns` → ~0 (no per-draw alloc) and renders identical.
2. **Fold in geometry hashing + bbox** (already deferred; move them into the same
   parallel-for, delete their per-draw `Schedule`s). Verify `bt_hashes`/`objAabb`
   drop from the serial path.
3. **Transform-extraction** into the batch — the big one. Snapshot cb2/cb3/t30/t31 +
   the classifier inputs; keep the sync classifier decision in Phase A. Verify
   `bt_extractXf` leaves the serial path and no o2w/culling artifacts.
4. **Skinning** last (bone palette build).

Each step: grep the batched body for `m_context->m_state` (must be empty), confirm
buffer pins balance, and check `[Perf.SdThreads]` for pool saturation (bump
`rtx.geometryWorkerThreads` if the parallel-for is starved).

---

## 7. Measurement plan

- Add per-frame counters: `arenaItems` (draws collected), `batchDispatchNs` (the single
  parallel-for dispatch+join wall), `batchWorkerBusyNs` (sum of worker time). The win
  shows as: serial SubmitDraw (`wallUs` minus the barrier wait) drops toward the
  collect-only cost, while `batchWorkerBusyNs` ≈ the old serial compute spread across
  N cores.
- Kill switch: keep `rtx.deferMaterialCompute`-style flags per stage (or one
  `rtx.batchSubmitDrawStages`) so a bad stage can be turned off at runtime without a
  rebuild.
- The decisive metric is **frame time / fps**, not any single bucket. The prior
  handoff's `entry_tailToBranch ~171ms` and `finalBlit ~70ms` are downstream of this;
  confirm the serial SubmitDraw reduction actually shows at the frame level and isn't
  swallowed by a GPU-bound `finalBlit`.

---

## 8. Do NOT re-attempt (already refuted this session)

- **Per-draw `make_shared` snapshots.** Two allocs → contention; even 1 alloc (aliasing
  `shared_ptr`, current state) + in-place capture + skipping the UAV copy did not fix
  it — the cost is per-draw allocation *count* under multi-worker contention. Batching
  removes the per-draw alloc entirely. That's the point.
- **Cheaper per-draw snapshot as the fix.** The snapshot got ~3× cheaper to build and
  `matCap` still dominated — because it's contention, not copy work. Stop optimizing
  the per-draw path; delete it.
- **Deferring texture (managed) reads.** `getImageHash` GC race. Game-texture views are
  fine; managed views are not. Buffers only for the racy parts.
- **Raising worker count as a standalone fix.** It *worsened* `matCap` (more allocator
  contention). Worker count only helps once the per-draw alloc is gone (batched).
