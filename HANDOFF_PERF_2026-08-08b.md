# HANDOFF — frame thread de-poled; every remaining fps lead is measured, named, and sequenced

**Date:** 2026-08-08 (03:10–04:35 session, follows HANDOFF_PERF_2026-08-08.md)
**Confidence:** `[V]` verified by measurement · `[I]` inferred · `[H]` hypothesis

**Headline: 20.2 → 23.0 fps (49.52 → 43.54 ms), and the road to 40–50 is fully
mapped — no probe needs to run again, no bucket is unattributed. Read §2 and
start cutting.**

---

## 0. WHAT LANDED (all verified in-game unless noted)

| # | change | where | result |
|---|---|---|---|
| 1 | Sub-marker default OFF (bisect landed, file's own TODO) | `d3d11_rtx.cpp` ~8690 | ~0.5 ms `[I]`; `RTX_D3D11_SUBMARK=1` re-enables |
| 2 | **[Perf.FanoutCamCache]** — staged-CB cache keyed (buf, mapGen, cbOff, fieldOff); publish still runs per draw | `d3d11_rtx.cpp` ~9690 (struct) + ~10280 (use) | **co_cbRead 5.4 → 0.16 us/call, 99.4% hits (58.7k/342 per window)** `[V]` ≈ 2.7 ms/frame |
| 3 | mapGen bump on `UpdateSubresource` NO_OVERWRITE (contract gap: CPU write window without a bump) | `d3d11_context_imm.cpp` ~1364 | correctness for ALL mapGen-keyed caches `[V]` by inspection |
| 4 | SIMD validity checks in fanout loops (exact isfinite/==0 semantics, serial+parallel identical) | `d3d11_rtx.cpp` ~172 (helpers) + 4 sites | loop 9.3 → 8.4 us/call `[V]` ≈ 0.4 ms |
| 5 | **[MemoCeiling.Slot]/[.Diff]** probe extension — names WHICH cb slot/byte breaks fullStable | `d3d11_rtx.cpp` ~35390 | answered §3.2, OFF again |
| 6 | **[DupFilter]** — skips within-frame byte-redundant duplicate commits (strict key) | `d3d11_rtx.cpp` ~35625 + `rtx_options.h` `filterDupSameDraws` | skips ~40-49/frame, ~0.6 ms wall `[V]`; see §2.5 |
| 7 | DupFilter SCOPE FIX: bone-transformed draws never filtered | same block | fixes viewmodel flicker/translucency — **REBUILD PENDING, VERIFY FIRST** (§5) |

Frame progression this session, same scene, A/B'd across rebuilds of identical
code state: 49.52 → 44.12 (changes 1-4) → 43.54 (change 6). `[V]`

---

## 1. WHERE THE FRAME IS NOW `[V]` (43.54 ms / 23.0 fps capture)

| timeline | ms | note |
|---|---|---|
| frame thread | ~41 | pole, by ~2-3 ms |
| dxvk-cs | ~38 (pre-DupFilter) | commitGeometryToRT 21.6 (processSceneObject 11.4, geom 4.8) + fat chunk 14.2 |
| GPU busy / idle | 19.65 / ~19.8 | 45% idle — GPU is NOT the limit |

Frame thread composition: ~13.1 ms game's own code (post-Present, untouchable
floor) + ~24 ms Remix per-draw work + ~2 misc. Fat chunk split (probe-polluted
absolutes, trustworthy ratios): **merge=8.0 (loop 4.5, buildBlases 2.1, dynBlas
0.9), gc 0.9, surfMat 0.85**, rest noise.

**Target math: 40–50 fps = BOTH CPU threads ≤ 25 ms.** Frame thread needs
Remix's ~24 → ~10; dxvk-cs needs 38 → ≤25. Leaf-shaving is DEAD on both — every
remaining leaf is ≤1 ms, memoized, or load-bearing.

---

## 2. THE LADDER — do these in order, sizes are measured not guessed

### 2.1 dxvk-cs dirty-list for updateInstance — ~12 ms, THE BIGGEST `[V]`
`[Perf.UpdInst]` (recorded in rtx.conf ~line 1690-1950, with per-stage numbers):
**REDUNDANT=97%** — 15,500 per-instance VISITS per frame load cold RtInstance
state to conclude nothing changed and do nothing. "No thread count fixes a
walk; a dirty list or change-tracking structure does." Sub-facts recorded
there: `surf` pays ~4 ms deciding 15,500× that it doesn't need to run (cut the
DECISION, not the payload); `tail` = onInstanceUpdatedCallback fanout, ~4 ms,
never examined; `move()` compute-before-compare already fixed; **caveat before
touching onTransformChanged: non-frame-stable m_stablePropId keys can leave
SpatialMap on a stale key — handle that first (recorded warning).**

### 2.2 merge loop 4.5 ms + buildBlases 2.1 ms (dxvk-cs) `[V]`
`[Perf.PrepScene]`/`[Perf.Merge]` now publish (conf: logPrepSceneSplit=True).
BlasSizeCache already landed; bBuild=0, bUpdate≈25, bReuse≈420 per frame — the
loop itself (15.5k instances → buckets) is the cost, same walk-shape as 2.1.

### 2.3 Camera-derivation memo in ExtractTransforms — ~2-2.5 ms, frame thread
Full design in task notes / §3.2 below. Key contract PROVEN by FanoutCamCache
on the same buffer (99.4%). Surgery risk: 3,800-line function, 13 paths,
side-state (m_projSlot, latches, m_smoothedCamPos, path ids) — the memo span's
side effects must be replayable or invariant on hit. Do it when the frame
thread is the pole again.

### 2.4 Geometry-replay tier — the 40+ stretch `[H]`, ceiling measured `[V]`
74% of draws are geomStable (identical VS+VB/IB frame-over-frame) and for 81%
of those **only cb2 (camera) changed** (§3.2). So: cache per-geomKey the
geometry-dependent pipeline outputs (vsIdx, skinCull/maxIdx, capture decisions,
o2w recon for static objects ≈ 4-5 ms of stages) and reapply only this frame's
camera. This is the replacement for the dead "whole-commit replay" idea.

### 2.5 DupFilter — landed, modest, guard it
Strict key (geometry + ranges + camType + PS + o2w bytes + PS SRV set + blend
identity), never filters fanout/blended/**skinned** (scope fix #7 — bone
palettes aren't in the key, viewmodel re-draws same-geo-different-bones).
~30-40 skips/frame expected post-fix ≈ ~1 ms dxvk-cs. **Rule: if ANY visual
anomaly is reported, set rtx.filterDupSameDraws=False and remove it — do not
weaken the key further.** Most of DupPass's 166 dupes are fanout/blended;
filtering those would need per-instance-array hashing — not worth it.

---

## 3. SETTLED THIS SESSION — do not re-derive

### 3.1 The previous handoff's §5 lead is DEAD `[V]`
"1,080 draws die in the filters" — no: drawsIn=1333/1349, drawsCommit=1038-1066.
Only ~21% reject. Filter-reordering cannot recover the pre-filter 3.2 ms.

### 3.2 fullStable=0% is NOT a key bug — it's the camera `[V]`
`[MemoCeiling.Slot]`: slotChg=[0,0,~655,0], none=0 — **cb slot 2 (camera CB)
is the only changing input for 81% of geom-stable draws** (rest are cb2+cb3 =
genuinely moving objects). `[MemoCeiling.Diff]`: bytes[4..486] of the cb2
window move every frame (origin + VP). The camera is never bit-identical in
gameplay. Whole-commit replay can never fire; the split in 2.3/2.4 is the
correct exploitation. Answer also recorded in rtx.conf at the probe.

### 3.3 T31Cache hits=0 is genuine, not a bug `[V]`
Engine appends t31 via Map(WRITE_NO_OVERWRITE) per draw → mapGen moves every
draw. Already ranged (copies only referenced bytes, 12%). Leave it.

### 3.4 The camera CB is the opposite: 99.4% stable `[V]`
Same game, different buffer, different cadence. That asymmetry (t31 per-draw
vs cb2 per-pass) is why FanoutCamCache pays and T31Cache can't.

### 3.5 DupFilter v1 key hole: bones `[V]`
Same geometry + same o2w + same textures + DIFFERENT bone palette = visually
different draw. Filtered → viewmodel committed on alternating poses → flicker
read as translucency under temporal accumulation. Fixed by exclusion (#7).
Lesson: a redundancy key must cover EVERY input that changes appearance, or
exclude the population.

### 3.6 Probe costs — answers are recorded, re-running is pure waste `[V]`
logMemoCeiling / logDupPass: ~60-90 ms/frame ON THE FRAME THREAD (they live in
OnDraw*). Both answered, both OFF, answers in rtx.conf next to the options.
perfSceneObjSplit: the expensive one (~20 ms/frame, inflates what it splits) —
its answers (find/mid/add/update = 18/16/2/62, and the [Perf.UpdInst] split)
are recorded in rtx.conf. logPrepSceneSplit: cheap (once-per-frame marks),
currently ON — turn OFF for clean A/B captures.

### 3.7 Session rules that bit again
- **A/B inside one process; probe-polluted captures size nothing** (the 210 ms
  4.7 fps report was the MemoCeiling run — ratios fine, absolutes garbage).
- **The log ROTATES on game restart** — pull numbers you need from a clean
  capture before launching the next run, or they're gone (lost the clean 03:51
  stage table this way).
- Look for recorded answers (conf comments, this file) BEFORE re-measuring.

---

## 4. CONFIG STATE AS LEFT

```
rtx.logPrepSceneSplit   = True    # cheap CS split; turn OFF for clean A/B
rtx.filterDupSameDraws  = True    # §2.5; if visuals break: False + remove
rtx.logMemoCeiling      = False   # ANSWERED (slot 2 / camera) — recorded in conf
rtx.logDupPass          = False   # ANSWERED — recorded in conf
rtx.logSubmitStall      = True    # feeds [Perf.InstDraw]/[Perf.CamCut]/report
rtx.perfThreadCensus    = True
rtx.perfReportFrames    = 50
rtx.parallelInstanceFanout = True
rtx.batchSubmitDrawStages  = True
```
New code toggles: `rtx.filterDupSameDraws` (rtx_options.h). Env:
`RTX_D3D11_SUBMARK=1` re-enables xform/tail leaf markers without rebuild.

---

## 5. FIRST FIVE MINUTES NEXT SESSION

1. The skinned-draw scope fix (#7) may be UNVERIFIED — ask whether the gun /
   gauntlet hands look solid after the rebuild. If not: set
   rtx.filterDupSameDraws=False (conf, no rebuild) and confirm that clears it;
   then delete the filter rather than weaken the key.
2. One capture: confirm `[DupFilter] skipped` ≈ 30-40 (viewmodel share gone),
   `[Perf.CamCut] camHit` still ≈ calls, HYGIENE drawsCommit down by exactly
   the skipped count, inst ≈ 15.5k.
3. Then go straight at §2.1 (CS dirty-list). It is the largest single win left
   (~12 ms on the thread that is about to be the pole) and its groundwork —
   numbers, stage split, the propId caveat — is all recorded in rtx.conf
   ~line 1690-1950. Do NOT re-run its probes.
4. §2.3 design notes also live in this repo's task list (task #7) if the task
   system is available; the design is restated in §2.3 fully regardless.
