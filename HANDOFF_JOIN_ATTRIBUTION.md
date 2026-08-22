# HANDOFF — DRAW ATTRIBUTION FOR THE ENUMERATION ARCHITECTURE

**Written 2026-08-21.** Companion to `RESIDENT_SCENE_PLAN.md` §7, which is the
live architecture. This document covers slice A and A2 only: whether a D3D11 draw
can be attributed to an engine object, and where.

Read §7 first. This is the evidence behind §7.3 and §7.4, plus what has changed
since they were written.

---

## 1. WHERE THIS STANDS IN ONE PARAGRAPH

Residency (plan slices 1-8) is built and can keep an object alive without a draw.
It cannot put back an object the engine never submitted, because a record is only
ever filed by a draw. §7 replaces the liveness source with the engine's own
object list. Every slice of that rests on being able to attribute a draw to a
list entry, and slices A and A2 were built to find out whether that is possible.

**It is, but not at either site tried so far.** The correct site is now known,
measured, and not yet built. See §5.

---

## 2. WHAT IS IN THE TREE

One file: `src/d3d11/d3d11_rtx.cpp`. No header touched, no `rtx_options.h` touch.
Everything gates on the existing `rtx.residentScene.logStats`.

| symbol | line | what |
|---|---|---|
| `namespace joinprobe` | 2711 | latch + counters + the record/replay side table |
| `namespace joinstack` | 2983 | the bounded stack census |
| `joinProbeEndFrame()` | 3186 | `[Join]` readout, frame thread |
| `joinprobe::noteList` | 3408 | in `renderListWrapper` |
| `joinprobe::latch` | 7412 | in `gateWrapper` |
| `qAllocWrapper` | 7584 | matsys queue allocator, record side |
| `qReplayWrapper` | 7615 | **WRONG LEVEL — see §4** |
| `queuedDrawInstallHook()` | 7627 | installs the two rel32 patches |
| `joinprobe::noteDraw` / `joinstack::noteDraw` | 38927 / 38935 | in `SubmitDraw` |

**To run any of it:** `rtx.residentScene.logStats = True` in `rtx.conf`. It does
NOT need `rtx.residentScene.enable`, deliberately — these measure the engine, and
should not depend on residency being armed.

`[Join`, `[Join.raw`, `[JoinStack` are not on the `log.cpp` denylist. Check that
first if they ever stop appearing.

---

## 3. SLICE A — THE GATE IS THE WRONG SITE `[V]`

Latched the renderable at the per-renderable draw gate (`client.dll
sub_180371590`, three call sites, already hooked for `[GateAll]`) and read it at
`SubmitDraw`.

```
[Join] tid{gate=37464 draw=47940 same=0} listCalls=1 listed=87..175
       gate{calls=64..137 pass=calls} draws=403..577 latched=0 noLatch=draws
```

28 frames, `same=0` and `latched=0` on every one. Dispatch and draw are on
different threads, so nothing orders them.

**The raw line is the proof, not the aggregate.** On the draw thread `since=`
climbs monotonically and never resets — 24314, 24315, 24316 on frame 3199, then
40603 and up on frame 3229. That is `t_drawsSince` never having been written, so
the gate's thread-local was never touched on that thread at all. Not a stale
latch and not an ordering fault: categorical thread separation.

Two things fell out that are worth keeping:

- **`listed` = 87..175 with `listCalls=1`.** BIBLE §3.3 puts Gate 1, the
  view-direction-dependent leaf-frustum test, INSIDE `sub_1801A8350`. Its output
  is therefore the POST-cull list. **Hooking that return can never give a
  pre-cull enumeration**, which is what plan slice B needs. Caveat: this capture
  was ordinary play, not the fixed-position sweep, so the variation alone does
  not isolate the cause — the code path settles it, not the number.
- `pass == calls` on all 28 frames. The gate rejects nothing, confirming BIBLE
  §3A.1's "RULED OUT" from a second angle.

**`drawnNotListed=0` in that capture is VACUOUS.** `drawn=0`, so nothing was ever
tested. Do not read it as a pass.

---

## 4. SLICE A2 — RIGHT MECHANISM, WRONG LEVEL `[V]`

Two rel32 rewrites in `materialsystem_dx11.dll`, both verifying the current
target before writing, both reported installed (`call=1 lea=1`):

- `matsys+0x6FD4F` — the `call sub_180087960` inside `sub_18006FCB0`, patched at
  that call site only, to compute the record address the caller is about to use.
- `matsys+0x6FD3C` — the `lea rdi, sub_1800708D0` functor constant, repointed so
  records carry our replay wrapper.

```
rec{n=1..2 withRend=1..2 tid=36728}  replay{n=1..2 hit=0 miss=1..2 tid=44384}
```

42 frames. `rec == withRend == replay` every frame, `hit=0` every frame.

**Finding 1 — the path is minor.** 1-2 records per frame against 391-576 draws.
`sub_18006FCB0` carries about 0.3% of the frame. It is *a* queued draw path, not
*the* one.

**Finding 2 — `hit=0` is NOT an off-by-constant, and the record side was
correct.** The dispatcher does not call the functor `sub_18006FCB0` writes. It
calls the record's FIRST qword, a per-subsystem helper. `sub_18006D180` is the
matsys one:

```c
v2 = *(QWORD*)(a1 + 8) + *(uint*)(a1 + 20);   // base + read cursor = THE RECORD
v6[0] = *(OWORD*)(v2 + 24);                    // 32 bytes copied ONTO THE STACK
v6[1] = *(OWORD*)(v2 + 40);
result = (*(fn**)v2)(v4, v6);                  // functor receives &v6
```

So `sub_1800708D0`'s `a2` is a pointer to a **stack copy**, freshly located every
call. No offset correction could ever have matched it. The record-side key
`base + head + 8` was right all along — it equals the helper's `v2` exactly.
**The replay hook was one level too deep.**

That also explains why `sub_1800708D0` appears in none of the census chains.

---

## 5. THE CHOKEPOINT, AND THE NEXT BUILD `[V]`

`[JoinStack]` — bounded census, every draw for 8 gameplay frames, then stops for
the process. 4320 draws, 34 distinct chains, 0 dropped, top 12 = 84%.

**Every chain bottoms out at the same place.** `sub_180087E30` names itself:

```c
ThreadSetDebugName(0, "RenderThread");
...
180087f7a  mov edx, cs:dword_181BBA054   ; read cursor
180087f80  mov rax, cs:qword_181BBA048   ; queue base
180087f89  add edx, 8
180087f8c  mov rdi, [rcx+rax]            ; the record's first qword = the helper
180087f90  lea rcx, qword_181BBA040      ; arg1 = queue descriptor
180087f9d  call rdi                      ; <-- 100% OF THE FRAME'S DRAWS
```

**THE NEXT BUILD: move the replay hook to `matsys+0x87F9D`.** At that point
`rcx` is the descriptor and the record address is the same expression the helper
computes, `*(u64*)(desc+8) + *(u32*)(desc+20)`. Coverage goes from 0.3% to 100%
and from one subsystem to all of them.

The record side needs the matching widening: `sub_180087960` hooked for **all**
callers rather than only at `0x6FD4F`. Note the descriptor has two cursors —
**`+16` is the write head, `+20` is the read cursor** — and the record address is
`base + head` computed *after* the allocator has advanced past the 8-byte header,
which is what `qAllocWrapper` already does correctly.

**This REPLACES the two A2 patches rather than adding to them.** Net simplification.

### Subsystem split, by draws carried

Taken from the frame directly above the dispatcher:

| helper | draws | share |
|---|---|---|
| `engine.dll+0x1b11d2`, `+0x1b119e`, `+0x1b1176`, `+0xb5dad` | 2760 | ~64% |
| `studiorender.dll+0x13b6d`, `+0x13b22` | 544 | ~13% |
| `client.dll+0x6c50d2` | 168 | ~4% |
| `materialsystem_dx11.dll+0x6d470` | 144 | ~3% |

**This is the number that should steer the plan.** Engine world rendering is
about two thirds of the frame and studio models about an eighth. The client
renderable list — the thing slice B was written around — reaches roughly a
twentieth. §7.2 said world and props dominate; this confirms it from the draw
side and is harsher than the estimate there.

---

## 6. TRAPS — DO NOT REDO THESE

1. **Do not attribute at `client.dll sub_180371590`.** Different thread from the
   draw. Settled, §3.
2. **Do not attribute at `matsys sub_1800708D0`.** Its `a2` is a stack copy, not
   the record. Settled, §4. No offset correction exists.
3. **Do not treat `sub_1801A8350`'s output as a pre-cull list.** The
   view-dependent cull is inside it. §3.
4. **Do not reason about a draw path's importance from IDA alone.** That is how
   A2 happened: the `[MatBind]` comment names `sub_180072E70 → sub_18001E400 →
   sub_18001C390` as "the" draw path, and it is real — it carries 0.3%. Static
   analysis shows a path exists, never how often it runs. The tree's own rule
   applies and was not followed: `RtlCaptureStackBackTrace` at a DXVK-controlled
   site FIRST, IDA second. `[JoinStack]` is that probe and it cost far less than
   the two patches did.
5. **`drawnNotListed=0` is only meaningful when `drawn > 0`.**
6. **`noLatch` being large is not a fault.** World geometry and static props
   never pass through the client renderable list, so those draws have no latch
   by construction. On this capture that is about four fifths of the stream.

---

## 7. OPEN

- **Slice B has no hook point.** §3 removed the obvious one. Candidates are in
  plan §7.5: upstream of `sub_1801A8350`; inside it before the frustum AND
  (BIBLE §3.3 names `leafSys+2104` / `leafSys+3128`, producer unproven); or
  accept post-cull and defeat the culls during a bounded seed with the individual
  `cullOff` groups actually on — a measurement nobody has taken, `pvs` in
  particular.
- **Is a renderable even the right unit for 64% of the frame?** The engine
  helpers draw world surfaces, which have no `IClientRenderable*`. The join being
  built may be the right mechanism attached to the wrong identity for the
  majority of draws. Settle this before building slices C-E.
- **`noTail`** — plan §7.6 item 1. Still unmeasured, still orthogonal.
- The A2 patches are installed and inert-but-harmless. Remove them when the
  dispatcher hook lands rather than leaving two dead rewrites in engine code.
