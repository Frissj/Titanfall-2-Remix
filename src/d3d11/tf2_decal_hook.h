#pragma once

// NV-DXVK: std::uint64_t is used below (DrainEd900Count). This header was
// previously all-bool/void and pulled its types in transitively; make the
// dependency explicit rather than rely on an include order that a future edit
// can silently break.
#include <cstdint>

// NV-DXVK [tf2_decal_hook]: replace the AutoDecal heuristic in d3d11_rtx.cpp
// with a TF2-specific runtime hook on engine.dll's decal-render function
// (sub_1801B4330, signature-located at startup). The hook flips a thread-
// local "we are currently inside TF2's decal-render call tree" counter on
// entry and clears it on return; the AutoDecal block then becomes a single
// `if (IsInDecalRender())` check instead of the brittle blend-state pattern
// match that was misfiring on world geometry / view model / sky mountain.
//
// Why a hook and not a categoriser based on shader hash:
//   - The blend-state pattern (DepthBias<0 + DepthWrite=0 + ONE,INV_SRC_ALPHA)
//     matches many alpha-blended overlays in TF2, not just decals. Coverage
//     probe confirmed this — 21M+ pixels/frame of world geo were getting
//     flagged.
//   - The decal-emit function IS unambiguous in callstacks: only true decals
//     reach EmitCs via engine.dll's R_DrawDecals-equivalent (sub_1801B4330).
//     Bracketing that call with a TLS flag gives an exact per-thread signal
//     with zero false positives.
//
// The hook is best-effort: if the AOB scan fails (e.g. engine.dll patched
// by another mod, version mismatch), IsInDecalRender() always returns
// false and the caller falls through to whatever fallback the user
// configures (currently: do nothing, since the heuristic was the bug).

namespace tf2_decal_hook {

  // One-time install. Safe to call from any thread on any frame; the
  // first call wins via std::call_once. Returns whether the hook is now
  // active (true) or installation failed/was skipped (false). After a
  // successful first call, subsequent calls just return the cached state.
  bool EnsureInstalled();

  // NV-DXVK [WorldVis]: install the diagnostic hook on engine.dll's
  // DrawWorldMeshesDepthOnly (sub_1800B8000) to measure the per-view
  // visibility bitmask populations and capture the caller chain that leads to
  // whoever fills them. Idempotent; safe to call every frame. Returns false
  // (once, with a log line) if engine.dll is absent or the prologue does not
  // match this build. See the block comment in the .cpp for how to read it.
  bool EnsureWorldVisHookInstalled();

  // NV-DXVK [JobProbe]: per-frame counters for the WORLD VISIBILITY WORKER,
  // client.dll sub_1802E8DA0. Replaces the x64dbg zero-pause hit counter that
  // HANDOFF_PITCH_CULL_2026-08-05 §10.1 prescribes — same measurement, but
  // in-process and at ~0.5% of the cost. The debugger route costs one
  // exception per hit (~1600 hits/frame drove the game to 0.5 fps and killed
  // it twice), and it cannot be normalised per frame without hand-correlating
  // two clocks.
  //
  // WHY THESE FIELDS AND NOT A NODE COUNT. §6 asserts "the residual must be in
  // how many nodes it is given ... the loop bounds var_12A0 and arg_10 come
  // from the caller." That is wrong, and the disassembly is unambiguous:
  //
  //   1802e8f17  movzx r14d, [rbp+rcx*4+var_10C0]   ; node index
  //   1802e8f20  movzx eax,  [rbp+rcx*4+var_10BE]   ; count
  //   1802e8f2b  and   ecx, 3FFh                    ; 1024-entry RING
  //   1802e8f31  shl   r14, 5                       ; *0x20 node stride
  //   1802e8f35  mov   [var_12A0], rax              ; <-- the node loop bound
  //
  // var_12A0 is POPPED FROM A STACK QUEUE that the function fills itself as
  // the BSP walk pushes children. The caller seeds exactly one node
  // (var_10BE = r9w = 1, var_10C0 = word[r11+2]). So node iterations are an
  // OUTPUT of the traversal, not an input: with the §2 reject branches NOPed
  // nothing prunes the walk, so it EXPANDS. Measured live, that is exactly
  // what happens — 6.46 nodes/call looking forward vs 9.98 looking down.
  // Counting nodes therefore cannot answer §6; counting JOBS can.
  //
  // RESULT (2026-08-05, 429 frames, camera moved 9 units): calls/frame is
  // FLAT across pitch — 192/206/207/186/186/184/184/184 in 10-degree bins,
  // non-monotonic, sd=0 on the top three — while instance count falls 23%
  // (r = -0.77). Job supply does not carry the view dependence, so handoff
  // §6's dig into sub_1802EB620 is excluded by measurement. The question
  // moved to the worker's output: see EnsureWorldDrainHookInstalled below.
  //
  // The supply side is `calls` (one call = one job). The counters themselves
  // live in the DXVK layer, in dxvk::tf2 (defined in rtx_camera_manager.cpp),
  // not here: rtx_instance_manager.cpp drains them and it compiles into
  // libdxvk.a, which dxgi.dll links without any d3d11 sources — a call in
  // that direction resolves for d3d11.dll and fails dxgi.dll with LNK2019.
  // This hook is purely the writer.
  //
  // Install the [JobProbe] hook. Idempotent; safe to call every frame.
  bool EnsureWorldJobHookInstalled();

  // NV-DXVK [DrainProbe]: the same worker's OUTPUT, measured at the drain
  // (client.dll sub_1802F04F0) by popcounting the visibility mask it fills.
  //
  // [JobProbe] already answered the input question — job supply is flat
  // across pitch while instances fall 23% — so the open fork is whether the
  // mask itself comes out short. The drain applies no test of any kind, so
  // its post-call popcount is exactly the accepted-leaf count:
  //   m1/m2 fall with pitch => §2's reject set is not exhaustive; the worker
  //     is still culling. Back into sub_1802E8DA0.
  //   m1/m2 flat with pitch => the mask is full and the loss is downstream of
  //     the visibility build entirely.
  // Counters live in dxvk::tf2 for the same link reason as [JobProbe].
  //
  // Idempotent; safe to call every frame.
  bool EnsureWorldDrainHookInstalled();

  // NV-DXVK [DispProbe]: the AREA DISPATCH path in sub_1802EB620 — which turns
  // out to be a queue loop over areas, not a single BSP walk.
  //
  // Three entry trampolines: sub_1802E8A20 (dispatches one area's jobs, and
  // where dword_1811FC0D8 is sampled), sub_1802ED900 (builds the portal record;
  // -1 => EB620 drops the area at 0x2EB8D0), sub_1802E7C70 (the record
  // allocator; -1 => sub_1802E8A20 silently does nothing at all). Both -1 paths
  // are invisible today: no log, and no reject branch any [CullOff] site can
  // patch.
  //
  // Rides rtx.cullOff.probeWorldJobs. Counters live in dxvk::tf2 for the same
  // link reason as [JobProbe]. Idempotent; safe to call every frame.
  bool EnsureDispProbeHookInstalled();

  // NV-DXVK [Ed900Probe]: counts entries to client.dll sub_1802ED900, the
  // view-direction-dependent portal-record builder whose -1 return drops a
  // whole area at 0x2EB8D0.
  //
  // NOT a wrapper. Wrapping ED900 as a C function froze the game on
  // 2026-08-05 — it does not decompile, so its signature is a guess, and a C
  // wrapper clobbers the volatile xmm registers a SIMD plane-builder needs.
  // This installs a hand-built island whose only own instruction is
  // `lock inc`, which writes FLAGS and nothing else; flags are dead at a
  // function-entry boundary. It cannot perturb ED900 under any signature.
  //
  // Rides rtx.cullOff.probeDispatch. Idempotent; safe to call every frame.
  bool EnsureEd900ProbeInstalled();

  // Read-and-reset the island counter. Returns 0 when NOT INSTALLED, which
  // reads identically to "never called" — check for the [Ed900Probe]
  // INSTALLED line before interpreting a zero.
  std::uint64_t DrainEd900Count();

  // Entries to sub_1802EB620, the per-view area builder. a20 is summed over
  // every EB620 invocation in a frame, so a20/eb620 = areas per view — the
  // number that separates "fewer views doing work" from "fewer areas per
  // view". drains=4 counts sub_1802F04F0, not this. Same zero-vs-absent trap:
  // check the [Eb620Probe] INSTALLED line.
  std::uint64_t DrainEb620Count();

  // Areas DROPPED by sub_1802ED900 returning -1 at 0x2EB8D0 — the verdict, not
  // the call count. The drop happens before the dispatch at 0x2EB910 AND before
  // the portal loop at 0x2EB915, so one -1 removes every crossing that area
  // would have made and the whole flood behind it. That is why the flat
  // ~1.0/frame call count never acquitted this branch.
  // Installed by EnsureEd900ProbeInstalled(); check Ed900DropProbeInstalled()
  // before reading a zero, same trap as the two above.
  std::uint64_t DrainEd900DropCount();
  bool Ed900DropProbeInstalled();

  // [DropAreas] — WHICH areas the -1 dropped, read and cleared. ed900Drop has
  // been 4.00 flat in every capture and was read as an acquittal; the identity
  // of those four has never been recorded, and the drop at 0x2EB8D0 is the one
  // path that fits all five observations (enqueued, selector consumed, no
  // portal loop, not selector-less, nothing pending).
  // No count function: the counter is the same one DrainEd900DropCount()
  // reports, and two readers would race to zero it.
  std::uint32_t DrainDropAreas(std::uint32_t* out, std::uint32_t maxOut);

  // Portals abandoned because the clipped polygon came out with fewer than 3
  // vertices — 0x2EC675 (r11) and 0x2EC67F (r9), the only rejects left
  // unpatched on the area-portal path and the leading suspects for the residual
  // PITCH dependence. Instrumented by RETARGETING each branch to a counting
  // stub, so the condition, length and destination are all unchanged; the
  // shared target loc_1802EC8ED has too many predecessors for a counter placed
  // there to mean anything. Check ClipDegenProbeInstalled() before reading a
  // zero, same trap as the counters above.
  std::uint64_t DrainClipDegenACount();
  std::uint64_t DrainClipDegenBCount();
  bool ClipDegenProbeInstalled();

  // Which NEIGHBOUR AREAS the abandoned crossings would have reached, read and
  // cleared. Identity rather than volume: a per-frame count cannot see one
  // crossing failing and cascading to four areas, which is what the 2D well
  // looks like. Returns how many were written to `out`.
  std::uint32_t DrainClipDegenAreas(std::uint32_t* out, std::uint32_t maxOut);

  // [DegenPair]: the JOINT (r11, r9) distribution of the rejects at 0x2EC675,
  // read and cleared. Cell = r9*4 + r11, r11 clamped at 3, r9 at 15 — so 64
  // cells, and kDegenPairCells is the size `out` must have room for.
  //
  // The counts are not interchangeable and only one of them is dangerous:
  // r11 becomes rec[+0] (edges), which sub_1802ED900 guards for zero at
  // 0x1802EDA84, while r9 becomes rec[+2] (planes), whose copy loop at
  // 0x1802EDA30 is post-test and runs forever on a bound of zero — that loop
  // is what site 14 crashed on. A per-branch COUNT cannot separate them, and
  // the existing DrainClipDegenBCount() cannot either, because its branch sits
  // behind this one and therefore never sees the degenerate-r11 population.
  // Returns how many cells were written.
  constexpr std::uint32_t kDegenPairCells = 64;
  std::uint32_t DrainDegenPairs(std::uint64_t* out, std::uint32_t maxOut);

  // [QueueProbe] — the queue CURSOR and the selector-less skip in
  // sub_1802EB620. Added for the POSITION axis, which every prior capture
  // held fixed by construction.
  //
  // sub_1802EAD60 returns the index the queue loop starts at (0x2EB7F7
  // `mov ebx, eax`; the loop at 0x2EB860 only ever walks FORWARD from it), so
  // every order-list slot below the cursor is unreachable regardless of what
  // any portal accepts. [AreaSeed] measured listLen = 30 flat and EAD60 was
  // recorded as eliminated on that basis — but listLen and the return value
  // are different quantities and only the first was ever read.
  //
  // READ THEM AS A PAIR, they are mutually exclusive explanations:
  //   qStart falls across the step  => the cursor is the gate; the five areas
  //     (113/141/148/149/153) are never visited and no crossing is at fault.
  //   qStart flat AND those areas appear in qSkipAreas => the cursor is
  //     innocent, the areas ARE visited, and no portal crossing ever wrote
  //     them a selector — so the fault is in the crossing, upstream of
  //     0x2EC739.
  // Returns 0xFFFFFFFF from ReadQueueStartIndex when not installed; check
  // QueueProbeInstalled() first, same zero-vs-absent trap as every counter
  // above.
  bool          QueueProbeInstalled();
  std::uint32_t ReadQueueStartIndex();
  std::uint64_t DrainQStartCount();
  std::uint64_t DrainQSkipCount();
  std::uint32_t DrainQSkipAreas(std::uint32_t* out, std::uint32_t maxOut);

  // [FaceReject] — client.dll+0x2EB98F, the camera-behind-the-portal-plane
  // test, by TARGET AREA. Measured answer to "qStart is flat and the five
  // areas are selector-less, so which crossing declines to enqueue them".
  //
  // It is a signed plane-side test on the camera ORIGIN
  // (plane . xmmword_1811FC000, times a {+1,-1} sign from dword_180911100),
  // so it is the only reject on the portal path that reads camera POSITION
  // rather than orientation — and it is the FIRST test in the portal body,
  // upstream of areaPortal, areaClip/areaSkipClip, clipDegen and the
  // allocator. That ordering is why all four of those read zero while areas
  // still disappear.
  //
  // READ AGAINST qSkipAreas ON THE SAME FRAME: the same ids appearing here on
  // the LOW side and not the HIGH side closes the chain end to end.
  // Check FaceRejectProbeInstalled() before reading a zero.
  bool          FaceRejectProbeInstalled();
  std::uint64_t DrainFaceRejectCount();
  std::uint32_t DrainFaceRejectAreas(std::uint32_t* out, std::uint32_t maxOut);

  // [PortalWalk] — client.dll+0x2EB93F, the target of EVERY portal the flood
  // iterates, before any reject. Built after [FaceReject] refuted 0x2EB98F:
  // areas 113/141/148 appear in faceAreas zero times in BOTH states while
  // their qSkip goes 7/91 -> 278/281, so they are neither rejected nor
  // enqueued — which means never reached.
  //
  // Read per area against qSkipAreas on the same line:
  //   in walkAreas + selector-less => a seventh reject exists on the crossing
  //     path; the six-site enumeration is incomplete.
  //   absent from walkAreas        => confirmed cascade; the source area never
  //     ran its portal loop, and no reject anywhere is at fault.
  // Areas are DEDUPED and the no-neighbour sentinel (nAreas+1, i.e. 180 on a
  // 179-area map) is excluded from the list and counted into `oobOut` instead.
  // Without that the list was one entry per portal bit and filled its buffer
  // with duplicates and sentinels before reaching the high bits, so real areas
  // read as absent — the one verdict this probe must never fake.
  bool          PortalWalkProbeInstalled();
  std::uint64_t DrainPortalWalkCount();

  // SOURCE -> TARGET for every portal walked, deduped. Replaces the
  // target-only drain, which could not name the owner of the cluster entry.
  // The source is reverse-resolved from the recorded portal bit via the
  // per-area range table (qword_181748CF8), so the island is unchanged.
  // Source 0xFFFF means the range table did not cover that portal — listed,
  // not dropped, so a hole is visible rather than silently thinning the list.
  // NOTE this read-and-clears the same bitmap the target-only drain used;
  // only one of the two may be called per frame.
  std::uint32_t DrainPortalWalkPairs(std::uint32_t* srcOut,
                                     std::uint32_t* dstOut,
                                     std::uint32_t maxOut,
                                     std::uint32_t* oobOut);

  // [SelWrite] — client.dll+0x2EC739, the selector store itself, by target
  // area. Separates the last two live possibilities for why 149 and 124 stop
  // being dispatched while their inbound portals are still walked every frame:
  //   present in LOW => enqueued, then abandoned by the forward cursor
  //                     (ordering); rewinds= says whether 0x2EC8DC fired.
  //   absent in LOW  => a reject upstream of this store that nothing counts.
  // ReadRewindCount() is the engine's own rewind counter, cumulative since
  // process start — difference it across frames. 0xFFFFFFFF means unreadable.
  bool          SelWriteProbeInstalled();
  std::uint64_t DrainSelWriteCount();
  std::uint32_t DrainSelWriteAreas(std::uint32_t* out, std::uint32_t maxOut);
  std::uint32_t ReadRewindCount();

  // Thread-local: are we currently inside TF2's decal-render call tree?
  // Cheap (single TLS load); safe to call from any draw site.
  //
  // CAVEAT (2026-07-30, from IDA): sub_1801B4330 is NOT decal-only. It is
  // slot 14 of the vtable at engine.dll+0x604448, whose interface-name
  // string is "StaticPropMgrClient005" — i.e. it is the static prop
  // manager's MAIN-PASS draw command, which draws props AND their decals
  // merged in one sorted list (the merge is gated by the engine cvar
  // staticProp_drawDecalsInSortOrder). So this returns true for ordinary
  // static prop draws too, not just decals. See the header comment block
  // above for the original — and now known to be too strong — "zero false
  // positives" claim.
  bool IsInDecalRender();

}  // namespace tf2_decal_hook

// NV-DXVK [tf2_engine_cvars]: write TF2 engine.dll ConVar values directly.
//
// Titanfall2 retail has no usable developer console, so the engine cvars
// that gate static prop visibility cannot be swept the normal way. They
// are, however, plain ConVar objects at fixed RVAs in engine.dll's .data,
// all registered with flags = 0 (no FCVAR_CHEAT, no FCVAR_DEVELOPMENTONLY),
// so writing them is just a memory store into an already-writable section.
//
// Layout verified from engine.dll's own ConVar constructor (sub_180416A40):
//   +0x18 m_pszName   +0x28 m_nFlags   +0x38 m_pParent (self for a root
//   cvar)   +0x58 m_fValue (float)   +0x5C m_nValue (int)
//
// Every write validates m_pszName against the expected string first, so a
// different engine.dll build fails closed (logs and skips) instead of
// corrupting whatever happens to live at that RVA.
namespace tf2_engine_cvars {

  // Values the caller wants written. -1 on any float means "leave that
  // engine cvar alone". Passed in rather than read from RtxOptions here so
  // this translation unit keeps zero dependency on the rtx headers — those
  // have an include-order requirement (dxvk_device.h must precede them; see
  // d3d11_rtx.cpp:531) that a small standalone file should not have to
  // satisfy just to read five floats.
  struct Overrides {
    bool  enabled                = false;
    float earlyDepthPrepass      = -1.0f;
    float earlyDepthPrepassDist  = -1.0f;
    float includeOpaques         = -1.0f;
    float includeOpaquesDist     = -1.0f;
    float drawDecalsInSortOrder  = -1.0f;
  };

  // Apply whatever rtx.conf currently asks for. Cheap and idempotent —
  // intended to be called once per engine frame so the values can be
  // swept at runtime via rtx.conf without a rebuild. Values are re-applied
  // every call because the engine resets cvars across map loads.
  void ApplyOverrides(const Overrides& overrides);

}  // namespace tf2_engine_cvars
