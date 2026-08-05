# HANDOFF — light leak / missing occluders: the AREA layer

**Date:** 2026-08-05 (later session; supersedes `HANDOFF_LIGHT_LEAK_2026-08-05.md`)
**Confidence legend:** `[V]` verified (measured / decompiled) · `[I]` inferred · `[H]` hypothesis

---

## 0. STATUS IN ONE PARAGRAPH

The light leak is **missing occluder geometry**, and it is now localised to the **area layer**, three
levels above every site in `CULLING_BIBLE.md` §0.2. Exactly **five areas** are dispatched per frame.
Two are present at every yaw; three drop out progressively as the camera turns, and when they go
their geometry never reaches the TLAS at all. All fourteen `[CullOff]` patches are verified `ON` and
are irrelevant — this is upstream of every one of them.

**The structural fact that explains a whole session of failed hypotheses:**
`sub_1802E8A20` (the per-area dispatch) has **TWO call sites in `sub_1802EB620`** — `0x2EB910` and
`0x2EC937`. Only the first was ever read. It is genuinely position-only (proven, §3), and it
produces the two unconditional areas. The second was never opened, and its region reads the frustum
**side planes**. That is where the three yaw-dependent areas come from. `[AreaDump]`'s `from=` field
was added to confirm this and **has not yet been run** — that is step 1.

---

## 1. HARD CONSTRAINTS — do not violate

- **Never propose object-keeping / lifetime-extension / frame-count features.** Ruled out repeatedly
  by the user, in explicit terms. Not an acceptable answer to any part of this.
- **Never suggest a rebuild is needed or the binary is stale.** If a fix does not work the bug is in
  the code or the hypothesis.
- **Do not run builds.** The user compiles manually.
- **No hacks, heuristics or allowlists.** Proper plumb only, even across many files.
- **Do not ask the user to choose between investigation branches.** Pick one and do it.
- **ONLY HOOK FUNCTIONS THAT DECOMPILE** — see §6. This one cost a game freeze.

---

## 2. THE MEASUREMENT THAT MATTERS `[V]`

`[AreaDump]`, stationary camera (camPos spread 1.4u), sweeping yaw. Fraction of frames each area's
BSP node appears:

| yaw bin | frames | 124 | 125 | 127 | 149 | 178 |
|---|---|---|---|---|---|---|
| 130–140 | 159 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| 140–150 | 113 | 1.00 | 0.96 | **0.09** | 0.94 | 0.99 |
| 150–160 | 83 | 1.00 | **0.06** | 0.00 | **0.04** | 0.96 |
| 160–170 | 18 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |

**124 and 178 are unconditional. 127 drops first (~140°), then 125 and 149 together (~150°).**
Fully reversible. Record selectors confirm it is a real enqueue difference, not a dispatch filter:
low yaw allocates `rec = 0,4,14,6,8`; high yaw only `rec = 0,4`.

Aggregates over the same sweep, for cross-reference:

| yaw bin | eb620 | drains | a20 | ed900 | alloc | allocFail | calls | m1Max | inst |
|---|---|---|---|---|---|---|---|---|---|
| 130–135 | 1.00 | 4.00 | 5.00 | 1.00 | 17.0 | 0 | 183 | 1983 | 599 |
| 145–150 | 1.05 | 4.18 | 3.83 | 1.05 | 11.8 | 0 | 87 | 873 | 422 |
| 155–160 | 1.00 | 4.00 | 2.00 | 1.05 | 6.0 | 0 | 50 | 364 | 292 |

`m1Max` (the largest single view's mask, not a sum) falls 5.4×, so the collapse is real and not a
view-count artifact.

---

## 3. THE MAP — what is proven about the area layer `[V]`

```
sub_1802EF090  (selector 0)
  xmmword_1811FC000 = *(_OWORD*)a2      also written to ctx+0x54060
  xmmword_1811FC010/020 = a2+32 / -(a2+48)
  xmmword_1811FC030     = a2+16          <- camera FORWARD; all side planes derive from it
  xmmword_1811FC040..070 = the four frustum SIDE planes (view-dependent)
  dword_1811FC0C0 = 4
  └─ sub_1802EB620                        RUNS EXACTLY ONCE PER FRAME (measured, eb620=1.00)
       0x2EB765  memset dword_1811FF91C = -1        (per-area record selector)
       0x2EB77F  memset dword_1811FC0E0 = -1        (12 per-size-class free lists)
       0x2EB79B  dword_1811FC0DC = 0                (record pool bump pointer)
       0x2EB7B6  dword_1811FC110 = leaf-skip thresh (ConVar; measured 1)
       0x2EB7D7  dword_1811FC114 = job-split thresh (ConVar; measured 250)
       0x2EB7E1  call sub_1802EAD60                 <- AREA QUEUE PRODUCER, POSITION-ONLY
       0x2EB808  JT_WaitForJobAndOnlyHelpWithJobTypes
       0x2EB860  ┌ QUEUE LOOP  (the path that WAS read)
       0x2EB864  │  ecx = word_1811FE920[rsi + i*2]        area id
       0x2EB86F  │  r9  = dword_1811FF91C[ecx]             selector
       0x2EB884  │  jz  loc_1802EC95E                      selector -1 -> skip
       0x2EB88A  │  dword_1811FF91C[ecx] = -1              consume
       0x2EB8AF  │  dec dword_1811FC0D8
       0x2EB8B8  │  cmp rec[+4], -1
       0x2EB8C0  │  jz  loc_1802EB8EB                      SKIP ED900 (taken ~4/5 of the time)
       0x2EB8C5  │  call sub_1802ED900                     portal record builder
       0x2EB8D0  │  jz  loc_1802EBE71                      ED900 == -1 -> area dropped
       0x2EB910  └  call sub_1802E8A20                     <-- CALL SITE 1
       ...
       0x2EC937     call sub_1802E8A20                     <-- CALL SITE 2  *** NEVER READ ***
       0x2ECB74 / 0x2ECBD1   read xmmword_1811FC040        frustum side plane
       0x2ECBAD / 0x2ECC10   read xmmword_1811FC050        frustum side plane
```

### 3.1 `sub_1802EAD60` — the area queue producer, POSITION-ONLY `[V]`

Two-phase. Phase 1 (`0x2EAE20`) walks `qword_181748CF0` — entries are `{dword tag, dword area}`;
`tag == -1` means leaf → enqueue, otherwise descend. Phase 2 (`0x2EB00C`) is a portal flood from the
seeded leaf over `qword_181748D00` (12-byte portal entries: side byte, neighbour area, plane index),
writing the order list `word_1811FE920` backwards from `dword_181748D8C - 1`.

Both gates compute `dot4(plane, xmm6)` where `xmm6 = xmmword_1811FC000`:

| site | test | on failure |
|---|---|---|
| `0x2EAFA5` | `dot4(bspPlane, xmm6)` sign | descend the camera's side only |
| `0x2EB0F1` | `dot4(portalPlane, xmm6) * sideSign <= 0` (`jbe`) | **do not cross the portal** |

`0x2EB0F1` is a real, unpatched reject that drops whole areas — but it is **position-driven**.

> **`xmmword_1811FC000` MEASURED CONSTANT** at `(-13568, -13568, -14848, -1)` on **all 671 frames**,
> yaw 51°→151°. So EAD60 cannot produce the yaw dependence. NOTE: all three components are exact
> multiples of 256 and x == y — it is grid-aligned and is **not** the main camera
> (`-845.7, -9279.3, 7356.0`). `CULLING_BIBLE.md`'s label "ctx+0x54060 = camera origin" is probably
> **wrong**; it does not matter for the yaw question, but do not trust that label elsewhere.

### 3.2 `sub_1802E7C70` — the record allocator `[V]`

Bump allocator over a per-frame pool of **4092 × 64-byte blocks** at `unk_181380A40`, twelve
per-size-class free lists at `dword_1811FC0E0`:

```c
blocks = (4*(entryCount + 4*planeCount) + 71) >> 6;
if (freeList[blocks-2] empty) {
    if (dword_1811FC0DC + blocks <= 0xFFC) { bump; return record; }
    return -1;                                  // POOL EXHAUSTED
}
```

`sub_1802E8A20` wraps its **entire body** in `if (result != -1)`, so a failed allocation drops the
area with no jobs, no mask bits, no bucket push, **no log**, and no patchable branch. Real hazard —
but **not firing**: `allocFail = 0` on 732 frames, peak pool 36/4092.

### 3.3 Job types `[V]` — registered in `sub_1802ED5D0`

```
byte_1811FBD90 = sub_1802EE7D0    byte_1811FBD91 = sub_1802E8DA0   (THE WORKER)
byte_1811FBD92 = sub_1802EE940    byte_1811FBD93 = sub_1802EE8A0
byte_1811FBD94 = sub_1802EB1E0
```

`byte_1811FBD91` is the worker itself, so EB620's `JTGuts_AddJobArray` at `0x2EB6E1` creates the
worker's job array — there is **no separate queue-filling job**.

### 3.4 `sub_1802E8A20` writes the job entry array `[V]`

IDA renders it as `&xmmword_1811FC000 + 926912` = `0x11FC000 + 0x1C4980` = **`0x13C0980`**, which is
why `xrefs_to 0x13C0980` shows only the worker's own split at `0x2E9631`. Displacement scans miss
this — the bible's §0.2d lesson, hit again.

---

## 4. DEAD ENDS — do NOT re-walk `[V]`

Each killed by measurement, in the order they were tried. **Five in a row.**

| # | hypothesis | how it died |
|---|---|---|
| 1 | The traversal thresholds vary with yaw | `planes=[4,4]`, `leafSkip=[1,1]`, `split=[250,250]` — `lo==hi`, identical in every bin, 160/160 frames |
| 2 | Record pool exhaustion drops areas | `poolHi` max **36 of 4092** and *falls* with yaw; `allocFail = 0` on 732 frames. The "cull-off patches starve the allocator" story dies with it |
| 3 | `sub_1802ED900`'s `-1` at `0x2EB8D0` is the reject | `ed900` flat at **~1.0/frame** across yaw while `a20` fell 5→2. Also confirms `rec[+4] == -1` keeps `0x2EB8C0` taken ~4/5 of the time |
| 4 | `sub_1802EAD60`'s portal flood is view-dependent | `fc000` constant on 671 frames (§3.1) |
| 5 | `sub_1802ED480` dynamically enqueues the 3 areas | Hook **installed**, **zero calls**. It is fn-ptr-only (`0x183C54E44`/`0x183C54E50`) and never runs on this path |

Also dead:
- **`a20` is a per-view sum** — no. `eb620 = 1.00` flat; EB620 runs **once per frame**.
- **`drains` is a view count** — no. It counts `sub_1802F04F0`, which runs ~4× per EB620.
- **`recCntSum/calls` is pinned at 4.0** — withdrawn. At 5° bins it drifts 4.47 → 3.51.
- **The `visibilityMask` / `pvs` cullOff flags** — A/B'd 2026-08-04, regressive/inert. See `rtx.conf`.

### 4.1 Probe fields that are WRONG — ignore them

- **`queueDepth` / `areasDropped`** on `[DispProbe]` read **negative**. The derivation assumed one
  EB620 call per frame *and* that `pend` was a total.
- **`pend`** is `dword_1811FC0D8`, which **counts DOWN** (`dec` at `0x2EB8AF`) and is folded with
  **max**, so it reports maximum *remaining*, not depth. Defeated two probe designs. To read queue
  depth it must be sampled **once at EB620 entry**, not on job threads and not with max.

---

## 5. PROBES BUILT THIS SESSION

All ride `rtx.cullOff.probeWorldJobs` except `[DispProbe]`/`[AreaDump]`/`[Ed480Probe]`, which ride
**`rtx.cullOff.probeDispatch`** (separate flag, default OFF — see §6).

| tag | fields | file |
|---|---|---|
| `[JobProbe]` (extended) | `planes` `leafSkip` `split` `pool/4092` `poolPct` + `leafSkipF/splitF` (float reinterp; both are integers) | `tf2_decal_hook.cpp` + `rtx_instance_manager.cpp` |
| `[DispProbe]` | `a20` `alloc` `allocFail` `ed900` `ed900Inst` `eb620` `fc000=(x,y,z,w)` (+ two broken fields, §4.1) | same |
| `[AreaDump]` | per dispatched area: `node=` (a3, the seed BSP node = identity), `rec=`, `bucket=`, **`from=`** (caller RVA — NOT YET RUN) | same |
| `[Ed480Probe]` | `ed480` + areas, plus a one-shot `FIRST CALL` backtrace. **Zero calls so far** | same |
| `[Ed900Probe]` `[Eb620Probe]` | counter-only islands (§6) | `tf2_decal_hook.cpp` |

`ed900Inst` exists because **an uninstalled hook and a never-called function both log 0** — check the
flag before reading a zero. Same defect that had `[OccProbe]` v1 reporting `outFr=0` on 459 frames.

---

## 6. THE FREEZE, AND THE RULE IT ESTABLISHED `[V]`

Wrapping `sub_1802ED900` as `int __fastcall(unsigned int)` **froze the game**. It does not decompile;
IDA's "one `_DWORD` arg" is a *guess*; it is a 0xd10-byte SIMD plane-builder whose stack frame is all
`_OWORD` slots. A C wrapper clobbers the volatile xmm registers it needs.

> **ONLY WRAP FUNCTIONS THAT DECOMPILE.** A C wrapper commits to a calling convention, and for a
> function IDA could not decompile its argument list is a guess, not a fact.

To count entries into a non-decompiling function, use a **counter-only island** (`InstallCounterIsland`
in `tf2_decal_hook.cpp`). Its only own instruction is `lock inc`, which writes **flags** and nothing
else, and flags are dead at a function-entry boundary:

```
+0x00  F0 48 FF 05 28 00 00 00   lock inc qword ptr [rip+0x28]   -> counter at +0x30
+0x08  <stolen prologue>
+0x08+n  FF 25 00 00 00 00       jmp qword ptr [rip+0]
+0x0E+n  <qword> return address
+0x30    <qword> counter
```

This hooked ED900 successfully with no freeze and answered the question that the wrapper died on.

**Also:** a separate crash was **not** a code bug — a frozen instance still held VRAM, and the next
launch died on `vkAllocateMemory vr=-2 heapBudget=0 MiB` → unhandled `DxvkError` → `std::terminate`.
A reboot cleared it. Check for `vkAllocateMemory failed` before blaming a hook.

---

## 7. NEXT STEPS, IN ORDER

1. **Run the capture with `from=`.** Stationary camera, sweep yaw through the transition. Diff the
   `[AreaDump]` node sets. Expect `124`/`178` from `0x2eb915` and `125`/`127`/`149` from `0x2ec93c`.
   If so, both prior results were correct and the error was assuming one caller — proceed to 2.
   If all five come from one site, §3's map is wrong somewhere and stop to re-read.
2. **Read `sub_1802EB620` from ~`0x2EC6F0` to `0x2ECC20`.** Never opened. It contains a second
   `sub_1802E7C70` call (`0x2EC6FA`), its own `dword_1811FF91C` writes (`0x2EC725`, `0x2EC739`), its
   own `dword_1811FC0D8` handling (`0x2EC8E5`, `0x2EC95E`), the second dispatch (`0x2EC937`), the job
   array refs (`0x2ECA3B`, `0x2ECA47`), and **the frustum side-plane reads** (`0x2ECB74`, `0x2ECBD1`,
   `0x2ECBAD`, `0x2ECC10`). Find the gate that decides an area enters this path.
3. **Patch that gate**, following the site-11 pattern from `CULLING_BIBLE.md` §0.2f — jump to the
   engine's own accept path, **do not NOP**. Note `sub_1802E8A20` immediately does
   `64 * selector` into `unk_181380A40`, so any bypass must leave a **valid** record selector.
4. Verify with `[AreaDump]`: all five nodes present at every yaw, `m1Max` flat, and the leak gone
   on screen.

### Unexamined, if 1–3 dead-end
- `sub_1802E8350` (0x65a) — fn-ptr table `0x183C54D54` only, no code xrefs, reads
  `xmmword_1811FC040`/`050` at `0x2E8412`/`0x2E8419`/`0x2E8420`/`0x2E8439`.
- `sub_1802ED380` (0x100) — writes `unk_181380A40`, calls the allocator at `0x2ED3A9`.
- `dword_1811FC0D4 = 0x8008 - planeCount - dword_181748D94` (`0x2EB810`) — a budget nobody has
  traced to a consumer.
- `0x2E96CA`/`0x2E96D0` in the worker — leaf-skip reject (`node[+0x0D] <= dword_1811FC110`), still
  unpatched. Position-independent, so not this bug, but it is a real occluder-loss mechanism.

---

## 8. THE METHOD LESSON

Five hypotheses died because I verified **one call site of a function exhaustively** — its planes,
its pool, its thresholds, `fc000` over 671 frames — while a **second call site of the same function**
sat 4 KB further down and was never opened. `xrefs_to sub_1802E8A20` would have shown it at any
point; it was never run because one caller had been read and was assumed to be *the* caller.

`CULLING_BIBLE.md`'s standing lesson is *"the thing being hunted was reached through a pointer, a
vtable slot, or a label that was not the one being searched."* Add the sharper form:

> **Xref the function you are hooking, even when you already know a caller.** A hook at a function's
> entry merges every call site. If the counter and the code disagree, suspect the set of callers
> before suspecting the measurement.

Second: `CULLING_BIBLE.md` was **stale on three separate points** this session (the `visibilityMask`
producer, the `sub_1802EB620` exclusion, the `ctx+0x54060` "camera origin" label). It is a map, not
an oracle. Re-verify any label you are about to build an argument on.
