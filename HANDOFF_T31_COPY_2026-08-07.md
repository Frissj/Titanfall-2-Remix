# HANDOFF — the frame thread is the pole, and 7.4 ms/frame of it is one memcpy

**Date:** 2026-08-07 · **Supersedes the perf direction of** `HANDOFF_PERF_2026-08-07_v8.md`
**Confidence:** `[V]` verified by measurement · `[I]` inferred · `[H]` hypothesis

Read `PERF_INSTRUMENTATION_MAP.md` first. It lists every `[Perf.*]` tag, which
thread it belongs to, and what gates it. This session wasted most of its time
rediscovering instruments that were already built, enabled, and logging.

**The short version.** v8 aimed at dxvk-cs. dxvk-cs is not the pole and has not
been for a while. The pole is the **frame thread**, 61% of it is Remix's
`OnDraw*` hook, and the single largest item in that hook is a `memcpyFromWC` of
the t31 per-instance transform buffer: **~14.3 µs/call × ~517 calls/frame ≈ 7.4
ms/frame, ~15% of the frame.** Everything needed to act on that is measured and
in the log. One question remains open (§4) and it is a 5-line probe.

---

## 0. WHAT LANDED THIS SESSION

| # | change | files | status |
|---|---|---|---|
| 1 | `PERF_INSTRUMENTATION_MAP.md` — full instrument index by timeline | (new) | **done** |
| 2 | `rtx.parallelInstanceFanout` — two-phase parallel fanout decode | `rtx_options.h`, `d3d11_rtx.cpp` | **built, GATED OFF, do not enable — see 2.3** |
| 3 | `[T31Range]` range-limited t31 copy | `d3d11_rtx.{h,cpp}` | **landed, unconditional, ~0 win — see 3.2** |
| 4 | `[Perf.LoopCut]` — 3-way cut of `loop_ms` | `d3d11_rtx.cpp` | **landed, this is what found it** |
| 5 | `[Perf.T31Src]` — mapping-lifetime discriminator | `d3d11_rtx.cpp` | **landed, answered §3.3** |
| 6 | `parCalls` / `parInst` on `[Perf.InstDraw]` | `d3d11_rtx.cpp` | landed |

Nothing here changes rendering. 2 is off by default; 3 is a pure reduction in
bytes read; 4/5/6 are counters gated by `rtx.logSubmitStall` (already `True` in
`rtx.conf`).

---

## 1. THE FRAME `[V]`

Captures 14:32 and 14:42, ~51–54 ms/frame, `matNew=0`, `matTotal` flat.

| timeline | per frame | source |
|---|---|---|
| **frame thread (PRESENT)** | **cpu 51.0 / wall 53.2, busy 94–96%, blocked 2–3** | `[Perf.Busy]` |
| dxvk-cs | ~46 ms | `[Perf.CsSplit]` execMs / frames |
| GPU | executes 22–24, **idle 28–31** | `[Perf.Gpu]`, `[Perf.GpuPass]` |

The GPU is idle over half the frame. This is a CPU problem on one thread.

### 1.1 The frame thread, fully accounted

```
[Perf.Entry]     34.8-36.4 ms   DrawIdxInst 20.5/533   DrawIndexed 10.9/623
[Perf.Gap]       17.3 ms        afterQueryEnd ~12.3 over 6 gaps  <- TF2 engine code
[Perf.DrawEntry] deOnDraw 23.7-24.7 us/draw   deLock 0.046 us/draw
```

`[Perf.Entry]` + `[Perf.Gap]` ≈ 52 ms ≈ wall. Thread fully attributed, and
`blockedMs` 2–3 says none of it is a wait.

**DXVK is acquitted `[V]`.** `[Perf.DrawEntry]` puts 31.6 of the 31.8 ms of draw
entry time inside `OnDraw*`, i.e. inside Remix. DXVK's own per-draw cost on this
thread is ~0.17 ms/frame and the device lock is 40 ns/draw. The frame thread
records **no Vulkan commands at all** — `DrawIndexed` does
`EmitCs([=](DxvkContext* ctx){ ctx->drawIndexed(...); })`, a lambda into a CS
chunk, and for RT-captured draws the `EmitCs` is skipped entirely.

### 1.2 Inside `OnDraw*`

```
SubmitInstancedDraw           [Perf.InstDraw] total_ms 1881-1959 /window
 ├─ inner SubmitDraw          inner_ms 714-730
 └─ fanout build              build_ms 1167-1241
     ├─ loop_ms 872-927  ->   [Perf.LoopCut] per call:
     │                          t31_gather  0.15 us
     │                          t31_copy   14.3-15.1 us   <-- THE TARGET
     │                          inst_loop   3.3-4.2 us
     │                          sum_us     18.3-19.3  (vs loop_ms/calls, closes)
     └─ setupPost_ms 296-318
```

At ~517 calls/frame: **`t31_copy` ≈ 7.4–7.8 ms/frame**, 83% of `loop_ms`, and
**38% of all `DrawIndexedInstanced` time**.

---

## 2. WHAT v8 GOT WRONG, AND WHY IT MATTERS

### 2.1 §2a of v8 is unanswerable as written `[V]`

`[CommitRT] submitMs` is accumulated per-draw inside `commitGeometryToRT`, which
runs on **dxvk-cs**. `[Perf.Gpu] fenceWaitMs` is accumulated in
`DxvkSubmissionQueue::finishCmdLists`, which calls
`env::setThreadName("dxvk-queue")` (`dxvk_queue.cpp`). **Different threads.**
They cannot nest, so "timestamp both on the same thread and check whether the
intervals nest" has no same thread to run on. Both being ~22 is coincidence.

### 2.2 §2b is dead `[V]`

The 22 ms fence wait is a `condition_variable::wait` plus
`cmdList->synchronize()` on dxvk-queue — already blocking. There is no spin to
convert. The frame thread's 94–96% busy is real work (§1.1), not a wait.

Also: `[Perf.Gpu]`'s `idleMs + fenceWaitMs + reapMs == frame` is an identity by
construction — those are the three states of one thread's loop. It is not
evidence about anything.

### 2.3 v8 §1.5 was right, for a reason it did not have `[V]`

"Zeroing the entire instance path would not move the frame" — correct. `instMs`
is on dxvk-cs (~46 ms) and the frame thread is at ~51. Nothing on dxvk-cs moves
the frame until it drops below the frame thread. That is why `instMs` 26 → 14
never moved the wall clock.

**Corollary: do not enable `rtx.parallelInstanceFanout`.** It is built and
correct, but `[Perf.LoopCut]` sizes its target (`inst_loop`) at **1.8 ms/frame**,
not the ~9 that was claimed when it was commissioned. Leave it off until
`t31_copy` is gone; then re-size it.

---

## 3. THE TARGET

### 3.1 What the code does

In `D3D11Rtx::SubmitInstancedDraw` (`d3d11_rtx.cpp`), just after `tLoop0`: the
mapped t31 buffer (`g_modelInst`, 208 bytes per instance) is bulk-copied into
`m_t31ReadCache` with `memcpyFromWC`, because the per-instance loop below indexes
it by a scattered `charIdx * 208` and reading WC memory with ordinary scalar
loads is far worse. The loop then builds `tforms` / `prevTforms`, which become
the draw's `instancesToObject`.

### 3.2 The cache does not work, and ranging did not save it `[V]`

```
[Perf.T31Cache] hits=0 misses=48521 hitPct=0.0 copiedMB=303.21
                rangedMB=43.94 rangedPct=12.7 avgCopy_KB=6.4
```

`hitPct` has been **0.0 in every window measured**. The `[T31Range]` change
(item 3 in §0) narrows the copy to the byte range the draw's `charIdx` values
actually reference — it removed 12.5–13.6% of the bytes and, judging by
`t31_copy` sitting flat at 14.3–15.1 µs across every window before and after,
approximately **none of the time**. It is harmless, costs 0.15 µs/call, and
stays; it is not a fix.

The reason it did not help is §3.3.

### 3.3 The cost is per fresh PAGE, not per byte `[V]`

```
[Perf.T31Src] calls=48521 sameBufPct=45.4 samePtrPct=5.9 sameGenPct=5.9
              pages=106604 pagesPerCall=2.20 usPerPage=6.54 usPerKB=2.25
```

- **`samePtrPct` 5.9–6.5%** — the mapped slice pointer changes on ~94% of calls.
  The engine is doing **`Map(WRITE_DISCARD)` per draw**, so DXVK renames the
  slice every time (`DiscardSlice()`), and our read is always the first touch of
  freshly-allocated memory.
- `sameGenPct` tracks `samePtrPct` exactly → every generation bump is a discard.
  This is why the cache key can never hold, and 0% hits is correct behaviour,
  not a broken key.
- `sameBufPct` 45–46% → roughly two buffers alternating.
- **2.20 pages/call × 6.5 µs/page = 14.3 µs** = `t31_copy`, exactly.

So the copy costs ~6.5 µs per 4 KB page of *newly mapped* memory. That is ~630
MB/s, an order of magnitude below what `movntdqa` streaming loads from resident
write-combined system memory should give.

---

## 4. THE ONE OPEN QUESTION — BUILD THIS FIRST, IT IS 5 LINES

Two mechanisms both produce ~6.5 µs per 4 KB page and they have **different
fixes**:

- **(a) First-touch cost on a newly-renamed slice.** Fixable: warm the slice, or
  get DXVK's discard-slice pool to recycle a small ring so the pages stay
  resident.
- **(b) Reading write-combined memory simply costs this.** WC writes do not
  populate the cache, so our read is a cold uncached fetch no matter that the
  game wrote those bytes microseconds earlier. **Not fixable while we read it at
  all.**

**`pagesPerCall` is pinned at 2.19–2.20 in every window**, so pages and bytes
never move independently and no ratio on the existing lines can separate (a) from
(b). `usPerPage` vs `usPerKB` was proposed as the discriminator and **it is not
one** — it is the same measurement twice. Do not retry it.

### The probe that does separate them

On 1 call in N (N ≈ 64), run the copy **twice** into two different destinations
and record both durations:

```
t31_copy1_ns   first  memcpyFromWC of [copyBegin, copyEnd)
t31_copy2_ns   second memcpyFromWC of the SAME range, immediately after
```

Emit both as per-call µs on a new `[Perf.T31Warm]` line with the sample count.

- **copy2 ≪ copy1** → (a). The first touch paid a one-time cost and the pages are
  now warm. Go after slice reuse / warming; there is a cheap fix.
- **copy2 ≈ copy1** → (b). The memory is simply this slow to read. There is no
  CPU-side fix; go to §5.

Sample it — an unconditional second copy doubles the cost of the thing being
measured. Gate on `rtx.logSubmitStall` like the rest. Write the destination to a
scratch buffer that is never read, and make sure the compiler cannot elide it.

---

## 5. THE STRUCTURAL FIX — correct under BOTH branches

**Stop reading t31 on the CPU.** We pull ~3.1 MB/frame back from a GPU-written
buffer to rebuild transforms the GPU already has.

Precedent, same class of problem, already in this tree: the per-instance
`COLOR1.y` read was moved GPU-side (see
`project_dxvk_tf2_dropship_doubletransform` — it was a *correctness* fix for a
race on a renamed dynamic buffer, and the renaming it tripped over is the same
`WRITE_DISCARD` behaviour measured in §3.3).

What the CPU read currently produces, and therefore what a GPU-side path must
supply:
- `tforms` → the draw's `instancesToObject` (current-frame matrix per surviving
  instance, `+cameraOrigin` applied to the translation column)
- `prevTforms` → previous-frame matrix, from bytes 48..95 of the same 208-byte
  entry, absolutised with the **previous** frame's camera origin
- the compaction: instances failing OOB / non-finite / zero-row0 checks are
  dropped, so output index ≠ instance index
- `m_fanoutRawT0` — the first surviving instance's pre-camOrigin translation

Read the two-phase decode already written for `rtx.parallelInstanceFanout`
(`[ParFanout]` block in `SubmitInstancedDraw`) before designing this: it is an
exact, side-effect-free statement of the decode, with the compaction separated
out, and it is the clearest spec of the semantics that exists.

---

## 6. REFUTED — do NOT reopen

| claim | how it died |
|---|---|
| dxvk-cs is the frame | It is ~46 ms under a ~51 ms frame thread. `[ThreadCensus]` PRESENT > dxvk-cs in every window |
| `submitMs` and `fenceWaitMs` might be the same 22 ms | Different threads (§2.1). The v8 §2a probe cannot be built |
| The frame thread spins through a fence wait | The wait is on dxvk-queue and it blocks. Frame-thread time is fully accounted as work (§1.1) |
| DXVK command recording is a large slice of the frame thread | `[Perf.DrawEntry]`: 31.6 of 31.8 ms is inside `OnDraw*`. DXVK ~0.17 ms/frame, lock 40 ns/draw |
| Secondary command buffers would help | The frame thread records no Vulkan at all (§1.1) |
| The t31 buffer is appended with `WRITE_NO_OVERWRITE`, so the whole-buffer re-copy is quadratic | `[Perf.T31Src] samePtrPct=6%` — it is `WRITE_DISCARD`, renamed per draw |
| Copying fewer t31 bytes will help | `rangedPct` 12.7% removed, `t31_copy` unchanged. Cost is per fresh page (§3.3) |
| The t31 cache just needs to be N-way | `hitPct` is 0.0 because the slice is renamed every draw. No key over (buf, ptr, gen) can hit |
| `usPerPage` vs `usPerKB` separates fault cost from bandwidth | `pagesPerCall` is constant at 2.19–2.20; they are the same measurement (§4) |
| `loop_ms` is per-instance work | `[Perf.InstFit] total_perInst_ns≈18` vs `total_perCall_ns≈36000`; `[InstStall]` shows `inst=2 loopUs=1997` against `inst=256 loopUs=1521` |
| The per-instance fanout loop is worth parallelising now | `[Perf.LoopCut] inst_loop` = 1.8 ms/frame (§2.3) |

---

## 7. PROCESS NOTES EARNED HERE

1. **Bytes are not time.** The t31 copy was picked as a target off
   `[Perf.WcCopy]`'s 290 MB while `[Perf.InstFit]` was already saying the cost is
   per-call. The byte counter happened to point at the right code for the wrong
   reason; the next one may not.
2. **A bucket's name is not its contents.** `loop_ms` is 83% a `memcpy` that
   happens before the loop, because `tLoop0` opens above it.
3. **Check whether the instrument already exists.** `[Perf.InstDraw]`,
   `[Perf.MapCut]`, `[Perf.InstFit]`, `[InstStall]`, `[MemoCeiling]`,
   `[DupPass]` were all built before this session. Two were already enabled and
   emitting the answer. See `PERF_INSTRUMENTATION_MAP.md` §7.
4. **Read the falsification note the previous author left.** The `[Perf.T31Cache]`
   comment pre-registered "if the game rewrites t31 between draws the cache
   misses every time and the change is a no-op". It fired. Nobody read it for
   two sessions.
5. **This game's per-draw cost swings ~2× in a fixed scene.** Cross-capture ms
   deltas are noise. Make each window self-sufficient (that is what
   `[Perf.LoopCut]` is for) instead of asking for a cleaner capture.

---

## 8. CAPTURE HYGIENE

- `[MatChurn] matNew=0` and `matTotal` flat.
- No `rtx-asset-exporter` in `[ThreadCensus]` — it runs 13–36% of a core for
  ~45 s after spawn and tails off slowly. Check the **final** windows too.
- `[Perf.GpuPass]` says `ALIGNED`, or its stage labels are meaningless.
- Same scene and position — and even then, see §7.5.

Runtime log:
`Titanfall2\rtx-remix\logs\remix-dxvk.log`

## 9. FIRST MOVES, IN ORDER

1. **Build the double-read probe (§4).** 5 lines, sampled, settles the only open
   question. Do not skip to a fix — mechanism has been called wrong three times
   in this file's history and twice in this session.
2. If **(a)**: chase slice warming / discard-pool reuse. Verify on
   `[Perf.LoopCut] t31_copy` and `[Perf.T31Src] usPerPage`.
3. If **(b)**: build §5. Verify on `t31_copy` going to ~0 and
   `[Perf.WcCopy]` losing its largest entry, with `[Perf.InstDraw] instPerCall`
   and `[FindStage]` unchanged, and props visually in the right places.
4. **Only then** re-size `rtx.parallelInstanceFanout` against whatever
   `inst_loop` has become.
