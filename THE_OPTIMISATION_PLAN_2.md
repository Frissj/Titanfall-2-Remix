# THE OPTIMISATION PLAN 2

**Written:** 2026-08-15
**Branch:** `architecture-overhaul`, uncommitted working tree
**Supersedes:** nothing. **Extends** `THE_OPTIMISATION_PLAN.md` (v1, 2026-08-11) and closes the
Sec 6e gate that `HANDOFF_2026-08-15_IDENTITY.md` left open.
**Goal:** unchanged -- 30 fps = 33.3 ms/frame.

Confidence markers, same as v1:
`[M]` measured this session . `[I]` inferred from measured data, not itself measured . `[U]` unknown, needs a capture

---

## 0. THE ONE-LINE ANSWER, v2

v1 Sec 0 says:

> The scene is 97% static and we rebuild it every frame. **Stop doing redundant work -- do not
> parallelise it.**

That is still true and still the main axis. But this session's largest single win was **neither**
redundancy elimination nor parallelism, and v1 has no category for it:

> **A third of the frame's CPU was not computing anything. It was waiting on the wrong kind of
> memory.** 17.6 ms/frame of "geometry hashing" was 23 KB of XXH3 running at 8.7 MB/s because the
> source buffer lived in BAR (device-local, host-visible, uncached). Changing the memory type took
> it to 0.5 ms. Nothing about the loop, the algorithm, or the redundancy changed.

So the rule to add, and it is now Sec 6 rule R8:

> **Before optimising a loop, check what memory it reads.** Bandwidth three orders of magnitude
> below cached RAM is not an algorithm problem, and no amount of caching the *result*, balancing
> the *schedule*, or deleting *redundant* calls will touch it.

Both prior instances of this in the tree (index buffers 2026-08-06, t31 SRV 2026-08-07) were found
the same way and fixed the same way. This was the third. It was missed for as long as it was
because a correct-sounding comment in `d3d11_buffer.cpp` asserted vertex buffers were write-only.

## 0.1 THE THREE COST CLASSES, NAMED

v1 has two. There are three, and they need different tools:

| class | signature | tool | this session |
|---|---|---|---|
| **redundant** | the same answer recomputed | cache / gate / delete | v1 Phase 3, unchanged |
| **serial** | one thread, ordered | parallelise / shard | Phase 2b, below |
| **stalled** | throughput far below the hardware's | fix the memory, not the loop | **-20 ms `[M]`** |

Stalled work is the dangerous one because it *looks* like the other two. It shows up as a big
number next to an innocent-looking function and invites you to parallelise it -- which is exactly
what Sec 2 below tried, and it made things worse.

---

## 1. BASELINE

### 1a. Frame-end parallel-for (game thread, `flushGeometryBatch`) -- BEFORE and AFTER `[M]`

| per frame | before | after |
|---|---|---|
| `hash` | 17,600 us | **500 us** |
| `bbox` | 4,700 us | **90 us** |
| `mat` | 6,000-7,500 us | 7,500-8,800 us |
| `skin` | 55 us | 55 us |
| `sumWork` (aggregate across 30 workers) | 29,000 us | **8,500 us** |
| `tail` (critical path) | 6,500 us | **690 us** |
| `parallelForMsPerFrame` | 6 | **0** |

**~5.8 ms off the game thread's critical path, ~20 ms off total CPU.**

### 1b. dxvk-cs, current `[M]` (probe OFF, quote these)

```
[CommitRT] perFrameMs=23-32  finalizeMs=1  submitMs=20-29  otherMs=1
[ProcDCS]  perFrameMs=18-26  geomMs=5-6    instMs=11-17    otherMs=1-2
```

This is now the frame's wall. It did **not** move this session -- everything built so far was on
the game thread.

### 1c. `processSceneObjectImpl` split `[M]` 2026-08-15 (probe ON, then turned back off)

```
pct  find=32-34   mid=7-8   add=4   update=53-55        addedPct=0
ms   find=2.9-4.0 mid=0.7-0.9 add=0.4-0.5 update=4.4-6.1
callsPerFrame 6,598-11,325
```

### 1d. GPU `[M]`

```
[Perf.GpuPass] totalMs=13.2 (stable)
[Perf.Gpu]     fenceWaitMs=18-25  idleMs=43-46  outsideRtMs=47-57
```

**The frame is CPU-bound with the GPU idle ~45 ms of every ~65 ms frame.** Every trade in this
document that spends GPU time to save CPU time is being made against that headroom, and that is
why the Sec 3 memory-type change was affordable.

---

## 2. WHAT WAS BUILT, 2026-08-15

| # | change | file | result `[M]` |
|---|---|---|---|
| 1 | `[BatchJoinSplit]` -- split `parallelForMsPerFrame` into disp / self / wake / work / tail / JOINIDLE | `d3d11_rtx.cpp` | answered Sec 6e: pool tax **~280 us/pass** |
| 2 | tail chunk scheduled instead of kept for the game thread | `d3d11_rtx.cpp` | the game thread was the straggler, not a helper |
| 3 | 4x chunk oversubscription | `d3d11_rtx.cpp` | **REVERTED** -- made everything worse, see Sec 4 |
| 4 | `[BatchJoinCensus]` / `[BatchJoinPeak]` -- per-item 4-way attribution + population census | `d3d11_rtx.cpp` | found 72/1070 draws = 74% of the stage |
| 5 | `RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS` | `d3d11_buffer.cpp` | **hash 17.6 ms -> 0.5 ms**, bbox 4.7 -> 0.09 |
| 6 | per-worker `m_lastSurfaceMaterial` slots | `rtx_scene_manager.{h,cpp}` | Phase 2b prerequisite, no-op today |
| 7 | `kBjItemProbe` compile-time gate | `d3d11_rtx.cpp` | per-item probe OFF; ~220 us/frame recovered |

### 2.1 THE SEC 6e ANSWER -- the gate the handoff blocked Phase 2b on

The handoff said:

> `[BatchSubmitDraw] itemsPerFrame=1068 parallelForMsPerFrame=6 workers=30`. If this pool costs
> ~6 ms to dispatch and join 1068 items, adding two more parallel passes eats the entire 15 ms
> win. **Sec 6e is a prerequisite, not a sibling task.**

Measured, the 6 ms was **none of those things**:

```
disp=120us   JOINIDLE=1us   wake{avg=170us}   ->   taxPerPass ~280us
```

Two extra passes cost ~0.6 ms against a ~15 ms win. **The gate is passed.** The 6 ms was the
game thread's own chunk stalled on BAR reads, and after Sec 3 the whole stage is 0.69 ms.

The premise was wrong in an instructive way: a single aggregate timer was read as "pool overhead"
when it contained dispatch, wake, real work on the calling thread, and stall, in unknown
proportion. **Sec 6 rule R9.**

---

## 3. THE CENTRAL FINDING -- BAR READS `[M]`

### The chain, in the order it was actually established

1. `sumWork` 29 ms/frame across 30 workers, `tail` 6.5 ms. Cross-checked against
   `work{avg} x chunks` -- the four components **are** the stage, nothing hidden.
2. `mapPos=72/1070`. **6.7% of draws carry 74% of the work.** The other ~998 are device-local,
   take the cheap pointer-hash branch in `runBatchHashJob`, and cost ~1 us each.
3. Peak draw: `v=132 idx=78 hash=2257us`. Mesh size cannot explain it by ~3 orders of magnitude.
4. Bytes, reported by the clamping code itself: `posB=11648 tcB=11648 idxB=36` against
   `posBufLen=917504`. **The clamps are correct** -- an 11 KB window of an 896 KB buffer.
   23 KB / 2.68 ms = **8.7 MB/s**, against XXH3's 10-30 GB/s on cached RAM.
5. Independent corroboration from a second consumer: `runBatchBboxJob` memcpys before scanning
   (the standard WC workaround) and still shows 4480 bytes in 1464 us = ~3 MB/s.
6. `posMem=0x7` = `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, `HOST_CACHED` clear, unanimous
   across every sample and both buffer sizes. **BAR memory.** Every read is a PCIe transaction.

### The fix

`RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS`, on by default, in `d3d11_buffer.cpp` -- the same three
lines already applied to DYNAMIC index buffers (2026-08-06) and DYNAMIC SRV buffers (2026-08-07):
add `HOST_CACHED`, clear `DEVICE_LOCAL`, gated on `D3D11_BIND_VERTEX_BUFFER`.

### What kept this hidden

The comment at the DYNAMIC case said:

> VERTEX and CONSTANT buffers still keep write-combining: they are genuinely write-only streaming
> data from our side.

It was written when it was true, it named the one vertex-buffer reader then known (the charIdx
instance buffer, safe behind `m_instBufCache` at a 100% hit rate), and it did not know about
`runBatchHashJob`. The CONSTANT half still stands. The comment has been corrected in place.

### OUTSTANDING `[U]` -- the GPU half of this trade

**This is the one unclosed item in this document and it is load-bearing.** Vertex buffers are
fetched per-draw by the rasteriser AND copied for BLAS builds, so moving them to system RAM puts
real per-draw traffic on PCIe. That exposure is larger than either prior case.

Bounding it: only 72 of ~1070 draws use DYNAMIC vertex buffers at all, and the GPU is idle
43-46 ms per frame (Sec 1d). Circumstantially, `[Perf.GpuPass] totalMs` sits at a stable 13.2 ms
and frames/3s did not regress. **That is not the A/B.**

**To close:** set `RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS` to `false`, capture one run, compare
`[Perf.GpuPass] totalMs` **distributions across the run** -- never first window against last, this
game's per-draw cost swings ~2x in a fixed scene.

**If it is a loss:** do not simply revert. The fallback is a cached shadow captured at `Map()`
time -- the game writes those bytes through the CPU anyway, and WC *writes* are fast; only reads
are catastrophic. That keeps the buffer in VRAM and has no GPU-side exposure at all. More work,
strictly better.

---

## 4. TOMBSTONE -- CHUNK OVERSUBSCRIPTION. DO NOT RETRY. `[M]`

Reasoning at the time: `work{avg}=1000us` but `tail=6500us`, so the expensive draws must be
clustered contiguously and equal-count chunking must be grouping them. More chunks than workers
would let work stealing rebalance. `kChunksPerWorker` 1 -> 4, 30 chunks -> 119.

| | 30 chunks | 119 chunks |
|---|---|---|
| `tail` | 6500 us | **7300 us** |
| `disp` | 120 us | 370 us |
| `wake` avg | 155 us | **800 us** |
| `sum(work)` | ~30 ms | **~47 ms** |
| frames / 3 s | 54 | **17** |

Finer chunks rebalanced nothing; they multiplied the pool's per-task cost. 119 `Schedule()` calls
are 119 mutex round trips and 119 `notify_one`s into 30 condvar-sleeping workers. Aggregate worker
time rose 58%, and that CPU competes with the game and CS threads -- which is why the **frame rate**
fell, not merely this stage.

This is the failure the `GeometryProcessor` header already records from 2026-07-28: *"RAISING the
worker count made this pool slower."* **Chunk count is not a lever on this pool.**

The imbalance was real; the diagnosis was wrong. It was not clustering, it was stall (Sec 3).

### 4.1 Related, unexploited `[M]`

`Schedule()` defaults `Affinity=0xFF`, and `affinityMask = min(popcnt(0xFF)=8, m_numThread=30)`,
so **only worker queues 0-7 are ever pushed to**; the other 22 acquire work exclusively by
stealing. It currently works (`wait` ~0), so this is a note, not a defect.

---

## 5. THE REMAINING PLAN

### Phase 2b -- shard `processSceneObjectImpl`. **RE-SCOPED on 2026-08-15 numbers.**

v1/handoff decomposition, unchanged in shape:

```
1. PARALLEL   geom          per-draw, independent
2. SHARDED    claim         by BlasEntry -- instances belong to exactly one BLAS
3. PARALLEL   update        exclusive owner guaranteed by step 2
4. ORDERED    tail          migration, creation merge, new materials, SpatialMap moves
```

**What changed `[M]`:**

| | handoff 2026-08-06 | now 2026-08-15 |
|---|---|---|
| `find` (ordered) | 24-27% | **32-34%** |
| `update` (parallel) | 57-60% | **53-55%** |
| `update/find` | 2.2-2.4x | **~1.6x** |

The plan's ordering argument -- "the parallelisable half dominates the ordered half" -- is a third
weaker than written. **Step 2, the sharded claim, is now the load-bearing step**, not a supporting
one, and it is the hardest and riskiest piece.

Compare percentages, not absolutes: `callsPerFrame` is 6.6-11.3k against the handoff's 15.6k, so
this is a different scene and the ms figures are not comparable across the two.

**And one number cuts strongly the other way: `addedPct=0` in every window.** The handoff's main
safety concern about sharding was:

> Two things escape the partition, both on the **miss** path: cross-BLAS migration and
> `m_instances.push_back`.

In steady state **no instance is created at all**, so both escapes are cold. They can take a plain
lock: a lock on a path that never fires costs nothing. This makes the sharded claim far more
tractable than the handoff feared. The `add` stage still burns 0.4-0.5 ms/frame as pure branch
checking that creates nothing -- a separate, smaller target.

**Build order:**

1. ~~`m_lastSurfaceMaterial` per-worker~~ **DONE 2026-08-15.** See Sec 5.1.
2. **The sharded claim.** Partition the frame's draws by `BlasEntry`; exclusive owner per shard;
   both miss-path escapes behind a lock. Everything else depends on this.
3. Parallel `update` -- legal only once 2 guarantees an exclusive owner.
4. Parallel `geom` (5-6 ms, per-draw independent).
5. Ordered tail.

**Expected `[I]`:** `find` will not vanish, so do not size this at the handoff's "~15 ms
parallelises". With `update` fully parallel and `find` sharded, `inst` 11-17 ms -> `[U]`, gated on
how well the BLAS partition balances (~450-465 unique BLASes over ~1350 draws).

### 5.1 DONE -- the `m_lastSurfaceMaterial` prerequisite `[M]`

One slot per worker, claimed on first use, in `rtx_scene_manager.{h,cpp}`.

**It is not a `thread_local`, and the reason matters.** The memo stores an **index into
`m_surfaceMaterialCache`**, and both invalidation sites exist precisely because a cache clear
renumbers every material. A `thread_local` would leave each worker's entry alive across a reset,
still naming an index that now points at a *different* material -- silent wrong-material
corruption, not a stale miss. So: slot array, plus `invalidateSurfaceMaterialMemo()` clearing all
slots, called from both existing sites.

This also closes a failure the handoff notes but does not separate: with one shared slot a reader
could match some fields against one writer's entry and the rest against another's **and pass**,
serving the wrong material. Per-slot makes that structurally impossible.

Behaviourally identical today -- one thread claims slot 0.

### Phase 2c -- `FillMaterialData`. **DEPRIORITISED, and the reason is a trap worth naming.**

After Sec 3, `mat` is 7,500-8,800 us = ~92% of the frame-end stage. It looks like the obvious next
target. It is not:

> **`sumWork` is aggregate worker time, not wall time.** Spread across 30 workers, `mat`
> contributes only ~0.3-0.7 ms to the critical path.

Cutting it reduces total CPU -- which does matter in a CPU-bound frame (Sec 1d) -- but it will not
move the frame the way "8.5 ms" suggests. Phase 2b targets *ordered* time on dxvk-cs, which is
real serial wall time. **Do 2b first.**

Peak single call is 1314 us on one draw (`v=9058`, `mapPos=0`, device-local -- so unrelated to
Sec 3). `[U]` what that draw is.

### Phases 3-8

Unchanged from v1. Nothing this session touched them. v1 Sec 3 (within-frame redundancy) remains
the largest single item in the document and is still the main axis.

---

## 6. RULES THAT MUST NOT BE BROKEN

v1 Sec 7's rules stand. Added 2026-08-15:

**R8. Before optimising a loop, check what memory it reads.** Throughput orders of magnitude below
cached RAM is a memory-type problem. No amount of caching the result, balancing the schedule, or
deleting redundant calls will touch it. Check `memFlags()`: `DEVICE_LOCAL|HOST_VISIBLE` without
`HOST_CACHED` is BAR, and every CPU read is a PCIe transaction.

**R9. One aggregate timer is not a diagnosis.** `parallelForMsPerFrame=6` contained dispatch, wake
latency, real work on the calling thread, and stall, in unknown proportion -- and the plan built on
it blocked the wrong task for a week. Split a timer by *mechanism* before acting on it.

**R10. A probe that can only look in one place will keep confirming that place.** The tail-only
per-item probe kept reporting `@idx1045-1090` because that was the only range it timed. Widened to
all chunks, the expensive draws were at idx 4, 23, 48, 117, 132, 155, 205, 568, 1058 -- everywhere.
Size the *population* before believing a *peak*.

**R11. Aggregate worker time is not wall time.** Before targeting a big `sum(work)` number, divide
by the worker count and check it against the critical path. See Phase 2c.

**R12. In-tree comments decay silently.** Three load-bearing claims were refuted this session, all
of which were true when written: "vertex buffers are genuinely write-only", "the last range runs on
the game thread so it is not idle during the join", and the handoff's `vbPtr`-rotates note. When a
comment justifies *not* measuring something, measure it.

---

## 7. INSTRUMENTATION STATE, 2026-08-15

| switch | state | notes |
|---|---|---|
| `kBjItemProbe` (`d3d11_rtx.cpp`) | **false** | per-item 4-way probe. ~220 us/frame -- was 0.7% of a 29 ms stage, would now be 2.6% of an 8.5 ms one. Flip to `true` for `[BatchJoinPeak]`/`[BatchJoinCensus]`. |
| `[BatchJoinSplit]` | **on** | per-CHUNK only, ~60 clock reads/frame. Leave on -- this is what shows `tail`/`wait`/`disp`. |
| `RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS` | **true** | flip to `false` for the Sec 3 A/B. |
| `rtx.perfSceneObjSplit` | **False** | answered 2026-08-15 (Sec 1c). Taxes both sides; every timing taken while on is probe-taxed. |
| `rtx.logDenyTags` | `[FindStage]` **added** | one line/frame, taxes perf runs. Remove when identity evidence is wanted again. |
| `rtx.pushInstanceRecords` | True | Phase 2, shipped |
| `rtx.pushInstanceRecordsVerify` | False | |

### Known interaction, carried from the SRV case

The `HOST_CACHED` substitution depends on the `HostVisibleSmallPool` fix in `dxvk_memory.cpp`
(present), and produces a once-per-frame VRAM sawtooth against the geometry-cache trim. **That is
not a leak** -- see the long note in `d3d11_buffer.cpp`.

---

## 8. DOES IT CLOSE?

```
frame now              ~65 ms          [M]
                                       (GPU 13.2 ms of it, idle 43-46 -- CPU-bound)
game thread            -5.8 ms         [M] Sec 3, banked
dxvk-cs CommitRT       23-32 ms        [M] the wall, untouched
  Phase 2b on inst     11-17 ms  -> ?  [U] gated on BLAS partition balance
  v1 Phase 3           largest item    unchanged
target                 33.3 ms
```

**It does not close on what is banked.** Sec 3 bought 5.8 ms of critical path and 20 ms of total
CPU; the frame is still CPU-bound and `commitGeometryToRT` at 23-32 ms is still the wall. Phase 2b
plus v1 Phase 3 are what have to carry the rest, and 2b is now smaller than the handoff scoped it.

The honest read: this session removed a whole class of cost that was not in either plan, and made
the *next* measurement trustworthy. It did not move the pole.

---

## APPENDIX A -- probes added, and where

| tag | file | gate | what it answers |
|---|---|---|---|
| `[BatchJoinSplit]` | `d3d11_rtx.cpp` `flushGeometryBatch` | always on | disp / wait / tail / JOINIDLE / wake / work / taxPerPass |
| `[BatchJoinPeak]` | same | `kBjItemProbe` | worst single item, 4-way split, bytes read, `memFlags` decoded |
| `[BatchJoinCensus]` | same | `kBjItemProbe` | every item summed; `mapPos` population; `sumWork` cross-check |
| `BatchHashBytes` | `d3d11_rtx.cpp` `runBatchHashJob` | out-param, `nullptr` when off | bytes ACTUALLY read, reported by the clamping code itself |

Design notes worth keeping:

- `BatchHashBytes` is filled by `runBatchHashJob` rather than recomputed at the call site. A second
  copy of the clamp logic would be free to drift and then agree with itself while lying.
- `[BatchJoinSplit]`'s `work{max}` is a **window** max over ~50 frames, not a per-frame max.
  `avg` is the reliable half of that pair.
- The per-chunk timestamp slots are cleared before each flush; a flush with fewer chunks than the
  last would otherwise fold the previous flush's timestamps into `max()` and report a straggler
  that did not happen.
