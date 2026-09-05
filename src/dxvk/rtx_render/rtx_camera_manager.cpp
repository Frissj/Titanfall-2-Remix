/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx_camera_manager.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <unordered_set>
#include <Windows.h>

// NV-DXVK [pilot-eye-capture]: file-scope mirror of the viewmodel-pass cb2
// c_cameraOrigin (Source's authoritative eye on this TF2 build).
//
// Producer: src/d3d11/d3d11_rtx.cpp at the cb2 RDEF fanout site, when the
//   draw is the viewmodel pass (vpMaxDepth ≤ 0.08). The viewmodel pass is
//   what Source renders weapons through; its c_cameraOrigin always carries
//   the actual pilot eye in pilot-on-foot, titan-cockpit, AND rodeo (pilot
//   on top of titan) modes. lp+0x3D6C is unreliable on this build (the log
//   shows it constant at (14158,-10801,877) over 46km of player travel —
//   it's a static script anchor, not a live eye field).
//
// Consumer: CameraManager::processCameraData snaps Main's worldToView
//   translation column to this so primary rays come from the actual pilot
//   position, not the BSP-pass cb2's titan body origin.
//
// Definitions must live in libdxvk (this TU) because libdxvk can't pull
// symbols out of the d3d11 DLL — the dependency points d3d11 → dxvk.
//
// Atomics: single-producer single-consumer, no ordering dependency, so
// memory_order_relaxed is sufficient at both ends.
namespace dxvk { namespace tf2 {
  std::atomic<float> g_pilotEyeX{ 0.0f };
  std::atomic<float> g_pilotEyeY{ 0.0f };
  std::atomic<float> g_pilotEyeZ{ 0.0f };
  std::atomic<bool>  g_pilotEyeValid{ false };

  // NV-DXVK [SkinAABB]: center-pixel VS hash, produced by RtxContext's
  // PickRegion2 coverage readback, consumed by d3d11_rtx.cpp's [SkinAABB] gate
  // so the skin probe follows the crosshair. 0 = nothing under the pick.
  std::atomic<uint64_t> g_pickCenterVsHash{ 0 };
  // NV-DXVK [PickDraw]: the *exact* DrawCallState::drawCallID of the dominant
  // surface under the center pick — finer than the VS hash, so probes can tell
  // sub-draws of the same VS apart (deck vs tall structure). Lags the GPU
  // readback ~2 frames, so it's exact only on a steady aim (or with
  // rtx.coverageSyncBeforeReadback=True). 0 = nothing under the pick.
  std::atomic<uint32_t> g_pickCenterDrawId{ 0 };

  // NV-DXVK [EngineCam]: dxvk-side mirror of the d3d11 trampoline's main-
  // camera capture status. d3d11_rtx.cpp's EndFrame consumer bumps this
  // every time it successfully forwards an engine-derived worldToView to
  // processExternalCamera(Main, ...). The per-draw classifier gate reads
  // it (below in processCameraData) to decide whether the engine-hook
  // path is actually live — when it is, we suppress the per-draw Main
  // update so the engine matrix isn't overwritten. When it's NOT (e.g.
  // the master kEnableEnginePatches toggle had the trampoline disabled,
  // or the install failed with a prolog mismatch, or we're early in
  // session before the first main-pass fires), the suppression stays
  // OFF and the legacy classifier path keeps Main alive — preventing
  // the "black screen" failure mode where useEngineHookMainCamera is
  // requested but no matrices are flowing in.
  //
  // 0 = never captured. Bumped to non-zero on the first capture; the
  // exact value isn't read, only "is it != 0".
  std::atomic<uint32_t> g_engineHookCaptureCount{ 0 };

  // NV-DXVK [EngineCam-Skybox]: same shape as g_engineHookCaptureCount but
  // for the 3D-skybox sub-view capture. Bumped by d3d11_rtx's EndFrame
  // consumer when it successfully forwards an engine-derived skybox matrix.
  // Currently consumed only by the [EngineSky] diag log — leaves room for
  // future routing to CameraType::Sky.
  std::atomic<uint32_t> g_engineSkyHookCaptureCount{ 0 };

  // NV-DXVK [JobProbe]: counters for client.dll's world visibility worker
  // (sub_1802E8DA0). WRITTEN by the hook in d3d11/tf2_decal_hook.cpp, DRAINED
  // once per frame by InstanceManager next to [PitchProbe].
  //
  // The storage lives HERE, in the dxvk layer, for a link reason and not a
  // stylistic one: rtx_instance_manager.cpp compiles into libdxvk.a, which
  // dxgi.dll links WITHOUT any d3d11 sources. A direct call from the instance
  // manager into tf2_decal_hook resolves fine for d3d11.dll and fails
  // dxgi.dll with LNK2019. Same reason g_engineHookCaptureCount is defined in
  // this file and only declared extern on the d3d11 side.
  //
  // One call to the worker == one job, and `calls` is the measurement that
  // matters — see the [JobProbe] block comment in tf2_decal_hook.cpp for why
  // counting node-loop iterations answers the wrong question.
  std::atomic<uint64_t> g_jobProbeCalls{ 0 };
  std::atomic<uint64_t> g_jobProbeRecCntSum{ 0 };
  std::atomic<uint64_t> g_jobProbeBadReads{ 0 };
  std::atomic<uint32_t> g_jobProbeJobIdxMin{ UINT32_MAX };
  std::atomic<uint32_t> g_jobProbeJobIdxMax{ 0 };

  // NV-DXVK [JobProbe] 2026-08-05, the YAW round: the three client.dll globals
  // that govern how much of the BSP the worker walks and how it splits itself.
  // CULLING_BIBLE §13 names all three and NOTHING HAS EVER READ THEM:
  //
  //   0x11FC0C0  frustum plane count      — 4 normally, 8 enables the extra
  //              plane blocks that cull sites 10c/10f guard.
  //   0x11FC110  leaf-skip threshold      — named in §13, never examined.
  //   0x11FC114  job-split subtree threshold — the worker calls
  //              JT_GrowJobArray_Lock and spawns a NEW job whenever a subtree
  //              exceeds this. It therefore sets the job count directly.
  //
  // WHY THESE, NOW. Binned by yawDeg on a STATIONARY camera (camPos spread
  // 0.3u, so nothing position-based can be involved), with all eleven [CullOff]
  // sites verified ON:
  //
  //   yaw 136-140   calls 181   recCntSum 730   m1 1989   inst 610
  //   yaw 142-148   calls  95   recCntSum 380   m1 1078   inst 455
  //   yaw 148-152   calls  48   recCntSum 192   m1  372   inst 295
  //
  // recCntSum/calls is 4.0 in EVERY bin. So the per-job portal record content
  // is invariant under yaw and only the NUMBER of jobs moves — which is
  // exactly what 0x11FC114 controls.
  //
  // WHAT THIS CANNOT ASSUME. §0.2e's boxed note is the trap here: the node ring
  // queue is filled by the worker itself, so node counts are an OUTPUT of the
  // traversal, not an input. Job count is the same kind of quantity — a
  // traversal that visits less territory splits itself fewer times. `calls`
  // falling 3.8x is therefore NOT evidence that anything upstream supplied
  // less. These thresholds are the one part of the input that is a genuine
  // input: they are written before the jobs run and read by every job.
  //
  //   a threshold MOVES with yaw  => that is the mechanism; no IDA needed.
  //   all three FLAT with yaw     => the input really is constant, the job
  //     count is self-generated, and the next target is sub_1802ED900 (the
  //     producer of the 0x1380A40 records, called from EB620 at 0x2EB8C5 /
  //     0x2EC9FD, right beside JT_DoneGrowingJobArray).
  //
  // Min AND max per frame, not a single sample: jobs run concurrently, and a
  // lone sample from one job thread would silently hide a value that changes
  // mid-frame. lo==hi on every line is what makes the reading unambiguous.
  // Raw u32 only — CULLING_BIBLE §10.2, "log the raw value before deriving
  // anything from it" (dword_1813C0940 was logged as a job count and was
  // actually a lock handle). The float reinterpretation is emitted alongside
  // because IDA's `dword_` prefix is a default, not a proven type, and a
  // threshold on a subtree size could plausibly be either.
  std::atomic<uint32_t> g_jobProbeLeafSkipLo{ UINT32_MAX };   // 0x11FC110
  std::atomic<uint32_t> g_jobProbeLeafSkipHi{ 0 };
  std::atomic<uint32_t> g_jobProbeSplitLo{ UINT32_MAX };      // 0x11FC114
  std::atomic<uint32_t> g_jobProbeSplitHi{ 0 };
  std::atomic<uint32_t> g_jobProbePlanesLo{ UINT32_MAX };     // 0x11FC0C0
  std::atomic<uint32_t> g_jobProbePlanesHi{ 0 };

  // NV-DXVK [JobProbe] 2026-08-05, the POOL round. The three thresholds above
  // came back dead flat, which ruled them out and moved the target to what
  // sub_1802EB620 actually does — and that turned out not to be a single BSP
  // walk at all. It is a QUEUE LOOP over areas (0x2EB860..0x2EB910): pending
  // count in dword_1811FC0D8, item ids from word_1811FE920, and each surviving
  // item calls sub_1802E8A20 to dispatch ITS jobs. So `calls` counts areas that
  // got a record, not tree depth.
  //
  //   0x11FC0DC  RECORD POOL BUMP POINTER, in 64-byte blocks.
  //   0x11FC0D8  pending-item count, DECREMENTED as EB620 consumes each area.
  //
  // WHY THE POOL IS THE SUSPECT. sub_1802E7C70 is a bump allocator over a fixed
  // pool at unk_181380A40 with twelve per-size-class free lists at
  // dword_1811FC0E0 (memset -1 at 0x2EB77F) and the bump reset to 0 at
  // 0x2EB79B, i.e. PER FRAME:
  //
  //   blocks = (4*(entryCount + 4*planeCount) + 71) >> 6;
  //   if (freeList[blocks-2] empty) {
  //       if (dword_1811FC0DC + blocks <= 0xFFC) { bump; return record; }
  //       return -1;                       // <-- POOL EXHAUSTED, 4092 blocks
  //   }
  //
  // and the -1 is consumed in sub_1802E8A20 as:
  //
  //   result = sub_1802E7C70(rec[0], rec[1]);
  //   if (result != -1) { ...copy planes, write node[+12] job entries,
  //                          publish, push bucket... }
  //   return result;                       // on -1: NOTHING happens.
  //
  // The entire body is inside that `if`. On failure the area is dropped whole —
  // no jobs, no mask bits, no bucket push, NO LOG, and no reject branch that
  // any [CullOff] site could patch. That is why every one of the eleven cull
  // patches can be verified ON while geometry still disappears.
  //
  // Note this also means sub_1802E8A20 is a SECOND writer of the job entry
  // array, which the xref scan on 0x13C0980 missed because IDA renders it as
  // `&xmmword_1811FC000 + 926912` (= 0x11FC000 + 0x1C4980 = 0x13C0980). The
  // bible's own lesson: a displacement scan only finds code that addresses the
  // buffer directly.
  //
  // SUSPECTED AGGRAVATOR: with worldFrustum and worldPortal NOPed the traversal
  // accepts far more nodes, so far more areas request records, so this pool
  // exhausts SOONER. CULLING_BIBLE §0.2f already warns the portal patch
  // "submits nodes the area-portal system would have occluded". That fits the
  // symptom appearing only AFTER that fix landed.
  //
  // READ IT, binned by yawDeg:
  //   poolHi pinned near 4092 in the collapsed bins, lower in the healthy ones
  //     => EXHAUSTION IS THE BUG. Fix = enlarge the pool, or stop
  //        sub_1802E8A20 dropping silently on a failed allocation.
  //   poolHi well under 4092 everywhere => exhaustion is not firing; the drop
  //     is sub_1802ED900's own -1 return at 0x2EB8D0 instead. Instrument that.
  //   pendHi falling with yaw => fewer areas were ever QUEUED, which is upstream
  //     of both and points at sub_1802EAD60 (0x2EB7E1, seeds the queue start).
  //   pendHi flat while calls falls => areas ARE queued and then dropped. That
  //     is the discrimination this pair exists for.
  //
  // SAMPLING CAVEAT, stated because it bounds the conclusion: these are read on
  // the job threads, which interleave with EB620's loop, so poolHi is a peak
  // over the samples taken and can UNDERSTATE the true peak if the last
  // allocations land after the final job call. Pinning at 4092 is therefore
  // conclusive; a value comfortably below it is suggestive, not proof.
  std::atomic<uint32_t> g_jobProbePoolLo{ UINT32_MAX };       // 0x11FC0DC
  std::atomic<uint32_t> g_jobProbePoolHi{ 0 };

  // NV-DXVK [DispProbe] 2026-08-05 — the AREA DISPATCH path in sub_1802EB620.
  //
  // WHAT THE POOL ROUND SETTLED, so this is not re-run: poolHi maxed at 36 of
  // 4092 (0.9%) and FELL with yaw. Exhaustion refuted; the allocator is fine.
  //
  // WHAT MAKES THIS THE LAST CHEAP BRANCH. EB620's queue loop can drop an area
  // in exactly two places, neither of which is a patchable reject and neither
  // of which has ever been counted:
  //   0x2EB8D0  sub_1802ED900 returned -1  -> jz loc_1802EBE71, no dispatch
  //   inside sub_1802E8A20, sub_1802E7C70 returned -1 -> the whole body is
  //             skipped: no jobs, no mask bits, no bucket push, no log
  //
  // THE DEDUCTION THAT BOUNDS WHAT THIS CAN PROVE — worth reading before the
  // capture, so the result is not over-read. The split test at 0x2E95E0 is
  //     node[0x0E] - node[0x0D] > dword_1811FC114
  // and BOTH node fields are static BSP data, so whether a node splits is a
  // fixed property of that node, independent of the view. Therefore
  //     jobs = 1 + SUM(node[0x0C]) over visited nodes matching a STATIC test
  // which means `calls` falling 3.8x already PROVES the visited-node set
  // shrank. That is established without measuring it.
  //
  // So these counters are elimination, not explanation:
  //   a20Calls falls with yaw        => fewer areas dispatched; the loss is in
  //     EB620's queue, and ed900Fail/allocFail say which drop did it.
  //   a20Calls flat while calls falls => area dispatch is INNOCENT and the loss
  //     is entirely inside sub_1802E8DA0's own walk. That means a 12th reject,
  //     or one of the eight patched sites not doing what the table claims. The
  //     next probe is then a node-outcome census inside the worker, not here.
  // Expect the second. a20Calls is ~1-2/frame while calls is ~181, so the
  // self-split cascade dominates by two orders of magnitude. This exists to
  // make that statement measured rather than argued.
  std::atomic<uint64_t> g_dispProbeA20Calls{ 0 };
  std::atomic<uint64_t> g_dispProbeAllocCalls{ 0 };
  std::atomic<uint64_t> g_dispProbeAllocFail{ 0 };

  // pend (dword_1811FC0D8) MOVED HERE from the job-thread fold, which measured
  // the wrong thing. It counts DOWN (dec at 0x2EB8AF) and was folded with max,
  // so it reported the maximum REMAINING rather than the queue depth — and
  // since jobs only run once EB620 is already draining, it read 1-2 regardless.
  // Sampled at sub_1802E8A20 ENTRY instead: the first dispatch of a frame sees
  // the count already decremented once, so pendHi here is (queue depth - 1).
  std::atomic<uint32_t> g_dispProbePendLo{ UINT32_MAX };      // 0x11FC0D8
  std::atomic<uint32_t> g_dispProbePendHi{ 0 };

  // NV-DXVK [Ed900Probe]: entries to sub_1802ED900, accumulated here by
  // d3d11_rtx's EndFrame (which drains the hand-built island counter) and read
  // once per frame by InstanceManager. The extra hop exists for the usual link
  // reason — the island lives on the d3d11 side and this file is what libdxvk.a
  // can see. Value may skew by one frame against the [JobProbe] line; that does
  // not matter for a "is this called at all" question.
  //
  // g_ed900ProbeInstalled MUST be checked before reading a 0 as a result: an
  // uninstalled hook and a never-called function produce the identical number.
  // ([OccProbe] v1 reported outFr=0 on 459 frames for exactly this reason.)
  std::atomic<uint64_t> g_ed900ProbeCalls{ 0 };
  std::atomic<uint32_t> g_ed900ProbeInstalled{ 0 };
  // Entries to sub_1802EB620 (the per-view area builder). a20/eb620 = areas
  // per view, which is what a20 alone cannot resolve.
  // MEASURED: eb620 = 1.00 flat across yaw. EB620 runs ONCE PER FRAME, so a20
  // is not a per-view sum and drains=4 was never a view count.
  std::atomic<uint64_t> g_eb620ProbeCalls{ 0 };

  // NV-DXVK: the raw four floats at client.dll+0x11FC000, sampled once per
  // sub_1802E8A20 entry. THIS SETTLES THE LOAD-BEARING ASSUMPTION OF THE WHOLE
  // AREA-ENQUEUE ARGUMENT.
  //
  // sub_1802EAD60's two gates — the BSP descent at 0x2EAFA5 and the
  // portal-crossing test at 0x2EB0F1 — both compute dot4(plane, xmm6) where
  // xmm6 = xmmword_1811FC000. If that is the camera ORIGIN, EAD60 is
  // position-only and cannot explain areas/frame falling 5.00 -> 2.00 on a
  // camera that moved 1.4u. It was called the camera origin on the strength of
  // sub_1802EF090 writing the same value to ctx+0x54060, which CULLING_BIBLE.md
  // labels "camera origin" — but EF090 reads FOUR consecutive 16-byte fields
  // (a2+0/+16/+32/+48) and uses the last three to build frustum planes. If a2
  // is a 4x4 VIEW MATRIX then a2+0 is row 0 and is view-dependent, and the
  // bible's label is just wrong. That document has already been caught stale
  // twice in this investigation.
  //
  // The camera is stationary and rotating, so the reading is unambiguous:
  //   constant across yaw => genuinely the camera origin; EAD60 stays
  //     eliminated and the loss is somewhere not yet read.
  //   changes with yaw    => the identification was wrong, EAD60's 0x2EB0F1
  //     crossing test IS view-dependent, and that is the mechanism — with
  //     0x2EB0F1 (jbe) as the patch site, same family as the eight world
  //     rejects in CULLING_BIBLE.md 0.2e.
  // Logged RAW, no derivation — the value is the evidence.
  std::atomic<float> g_dispProbeFc000X{ 0.0f };
  std::atomic<float> g_dispProbeFc000Y{ 0.0f };
  std::atomic<float> g_dispProbeFc000Z{ 0.0f };
  std::atomic<float> g_dispProbeFc000W{ 0.0f };

  // NV-DXVK [AreaDump] — RAW per-area identity, not another count.
  //
  // WHY THE METHOD CHANGED. Six gates on the enqueue path have now each been
  // measured constant across yaw (fc000, ED900 calls, allocFail, the two
  // thresholds, the plane count), yet areas dispatched per EB620 invocation
  // falls 5.00 -> 2.00. Four hypotheses have been eliminated in a row without
  // locating the cause, which means a BELIEF about this code is wrong rather
  // than a quantity being unmeasured — and another counter cannot find that.
  // So: stop counting, record WHICH areas dispatch.
  //
  // sub_1802E8A20(a1, a2, a3): a1 = bucket index, a2 = portal record selector,
  // a3 = the seed BSP node (used as qword_181748D58 + 32*a3). a3 is the area's
  // identity. Five entries at low yaw, two at high — so the question becomes
  // "what is different about the three that vanish", answerable from data
  // instead of from another inference about the disassembly.
  //
  // Bounded by construction: at most 32 slots per frame, written by the hook,
  // drained by the frame loop. No throttle logic, no unbounded log growth, and
  // no risk of a job thread logging (which would perturb what is measured).
  // slotN is the TRUE count and may exceed 32 — compare it against a20.
  //
  // RAISED 8 -> 32 on 2026-08-05. The cap was sized when a20 was 2-5 areas. The
  // POSITION sweep (view locked at pitch -7.81836 / yaw 175.265, 365 frames,
  // only camPos moving) puts a20 at 14 on one side of y = -10000 and 9 on the
  // other, so the 8-slot window truncated exactly the frames that mattered and
  // the 5 areas that drop could not be named — the one thing this probe exists
  // to do. 32 covers the observed 14 with headroom; slotN still reports truth.
  std::atomic<uint32_t> g_dispProbeSlotN{ 0 };
  std::atomic<uint32_t> g_dispProbeSlotA1[32];
  std::atomic<uint32_t> g_dispProbeSlotA2[32];
  std::atomic<uint32_t> g_dispProbeSlotA3[32];
  // NV-DXVK: the client.dll RVA that CALLED sub_1802E8A20, per slot.
  //
  // THIS IS THE FIELD THAT EXPLAINS FIVE FAILED HYPOTHESES. sub_1802E8A20 has
  // TWO call sites in sub_1802EB620 — 0x2EB910 (the queue loop that was read)
  // and 0x2EC937 (never read). Every conclusion about "the enqueue path" was
  // built on the first one alone, so a20 has been merging two different
  // dispatch paths the entire time:
  //   0x2EB915 : the queue loop. Gated by fc000, MEASURED constant on 671
  //              frames — genuinely position-only, and it produces the two
  //              unconditional areas (BSP nodes 124, 178).
  //   0x2EC93C : the second path. Its region reads xmmword_1811FC040/050 (the
  //              frustum SIDE planes, view-dependent, unlike fc000) at
  //              0x2ECB74/0x2ECBD1/0x2ECBAD/0x2ECC10 — so this is where the
  //              three yaw-dependent areas (127, then 125/149) come from.
  // Both facts held simultaneously; the error was assuming one caller.
  // Raw RVA, no classification — CULLING_BIBLE 10.2, log the value before
  // deriving anything from it.
  std::atomic<uint32_t> g_dispProbeSlotRA[32];

  // NV-DXVK [QueueProbe] — the queue CURSOR and the selector-less skip.
  //
  // Defined HERE, in libdxvk, for the same link reason as every other counter
  // on the [DispProbe] line: the islands live in d3d11/tf2_decal_hook.cpp, but
  // the emitter is InstanceManager::garbageCollection in libdxvk.a, and
  // dxgi.dll links libdxvk.a WITHOUT the d3d11 objects. libdxvk therefore
  // cannot call into tf2_decal_hook at all — d3d11 drains the islands on its
  // own frame boundary and pushes the values down here. Calling the drain
  // functions directly from the emitter costs five LNK2019s.
  //
  // g_qStart is ASSIGNED, not accumulated: it is an order-list INDEX (the
  // value sub_1802EAD60 returns and the queue loop starts at), and summing an
  // index would be meaningless. The call counts are accumulated as usual so a
  // frame-boundary skew between producer and consumer loses nothing.
  std::atomic<uint32_t> g_qStart{ 0xFFFFFFFFu };
  std::atomic<uint64_t> g_qStartCalls{ 0 };
  std::atomic<uint64_t> g_qSkipCalls{ 0 };
  std::atomic<uint32_t> g_qSkipAreaN{ 0 };
  std::atomic<uint32_t> g_qSkipAreas[32];
  std::atomic<uint32_t> g_queueProbeInstalled{ 0 };

  // NV-DXVK [FaceReject] — client.dll+0x2EB98F, portals skipped because the
  // camera is behind the portal plane, counted and keyed by target area.
  // Same link-boundary reason as the g_q* block above.
  std::atomic<uint64_t> g_faceRejectCount{ 0 };
  std::atomic<uint32_t> g_faceRejectAreaN{ 0 };
  std::atomic<uint32_t> g_faceRejectAreas[32];
  std::atomic<uint32_t> g_faceRejectInstalled{ 0 };

  // NV-DXVK [PortalWalk] — client.dll+0x2EB93F, the target area of every
  // portal the flood iterates, before any reject. 48 slots rather than 32:
  // this is the SUPERSET of every enqueued and every rejected target, so it
  // is the widest list on the line and truncating it would silently turn a
  // "present" into an "absent" — the exact inversion the probe exists to
  // decide.
  std::atomic<uint64_t> g_portalWalkCount{ 0 };
  std::atomic<uint32_t> g_portalWalkAreaN{ 0 };
  std::atomic<uint32_t> g_portalWalkAreas[48];
  std::atomic<uint32_t> g_portalWalkInstalled{ 0 };
  // Portals whose target was the no-neighbour sentinel or did not resolve.
  // Kept visible so a short walkAreas list is never mistaken for a small walk.
  std::atomic<uint32_t> g_portalWalkOob{ 0 };

  // NV-DXVK [SelWrite] — client.dll+0x2EC739, areas that actually received a
  // selector, plus the engine's own cursor-rewind counter (dword_1811FBD98).
  // g_rewinds is a DELTA computed on the d3d11 side; the raw global is
  // cumulative since process start.
  std::atomic<uint64_t> g_selWriteCount{ 0 };
  std::atomic<uint32_t> g_selWriteAreaN{ 0 };
  std::atomic<uint32_t> g_selWriteAreas[48];
  std::atomic<uint32_t> g_selWriteInstalled{ 0 };
  std::atomic<uint32_t> g_rewinds{ 0 };

  // NV-DXVK [DropAreas] — WHICH areas sub_1802ED900 dropped at 0x2EB8D0.
  // ed900Drop (the count) has read 4.00 flat in every capture and was treated
  // as an acquittal on that basis; the identity of those four was never taken.
  std::atomic<uint32_t> g_dropAreaN{ 0 };
  std::atomic<uint32_t> g_dropAreas[48];

  // NV-DXVK [AreaSeed] — sub_1802EAD60's ORDER LIST, measured from outside it.
  //
  // WHY THIS PROBE EXISTS. rtx.cullOff.areaPortal (client.dll+0x2EB9CF) forces
  // every portal crossing accepted. Measured: a20 rises 5 -> 23 at low yaw and
  // new areas appear (BSP nodes 62/74/75/126/127), but at 155-167deg a20 and
  // alloc are BYTE-IDENTICAL to the unpatched run (2.00 / 6.0). So a second
  // gate sits UPSTREAM of the portal loop, and the only thing upstream is which
  // areas EAD60 puts in the order list at all: the queue loop at 0x2EB864 reads
  // area ids out of word_1811FE920, and an area with a selector that is not in
  // that list is never visited.
  //
  // TWO FIELDS, AND WHY EACH IS READ THE WAY IT IS.
  //
  // org = the value EF090 is about to store into xmmword_1811FC000, read from
  // the ARGUMENT (*(__m128*)a2) at EF090 entry rather than from the global. It
  // has to be read there: EB620's transformed tail OVERWRITES the global at
  // 0x2ECB2F every frame (it pushes the position and all frustum planes through
  // the matrix at [arg_8+0x50880] before the second dispatch at 0x2ECE37). Every
  // previous reading of fc000 — including the "constant on 671 frames" that
  // eliminated EAD60 — was taken after that store, so it measured the
  // post-transform leftover and could never say anything about what EAD60
  // consumed. That also explains why the logged value was grid-aligned to 256
  // and was not the main camera. This field is the first honest look at it.
  //
  // listLen/areas = the order list itself. EAD60 does not decompile, so it is
  // not hooked (the standing rule, and the wrapper that froze the game). It does
  // not need to be: it writes word_1811FE920 BACKWARDS from dword_181748D8C-1
  // (0x2EB192) and returns the final cursor+1 (0x2EB1B6). Sentinel-filling the
  // buffer with 0xFFFF before the call and scanning after it recovers the exact
  // extent from outside, with no assumption about a calling convention. Writing
  // below the cursor is inert — the queue loop only ever reads [cursor, nAreas).
  //
  // READ IT, binned by yaw, against the SAME frame's a20:
  //   listLen falls with yaw  => EAD60 IS view-dependent, its 0x2EB0F1 portal
  //     crossing test is the gate, and org will show why.
  //   listLen flat while a20 falls => EAD60 is exonerated for real this time and
  //     the loss is between the order list and the selector.
  std::atomic<float>    g_areaSeedOrgX{ 0.0f };
  std::atomic<float>    g_areaSeedOrgY{ 0.0f };
  std::atomic<float>    g_areaSeedOrgZ{ 0.0f };
  std::atomic<float>    g_areaSeedOrgW{ 0.0f };
  std::atomic<uint32_t> g_areaSeedNAreas{ 0 };   // dword_181748D8C, the buffer bound
  std::atomic<uint32_t> g_areaSeedListLen{ 0 };  // entries EAD60 actually wrote
  std::atomic<uint32_t> g_areaSeedN{ 0 };        // recorded ids; may exceed 16
  std::atomic<uint32_t> g_areaSeedAreas[48];
  std::atomic<uint64_t> g_areaSeedCalls{ 0 };    // EF090 invocations this frame
  std::atomic<uint32_t> g_areaSeedInstalled{ 0 };

  // Areas still holding a record selector when sub_1802EB620 returns: enqueued
  // by a portal crossing and NEVER VISITED by the queue loop. dword_1811FF91C
  // is memset to -1 at 0x2EB765 and consumed back to -1 at 0x2EB88A on visit,
  // so a survivor is unambiguous.
  //
  // THE QUESTION THIS ANSWERS. The collapse is a localised WELL in 2D, not a
  // trend on either axis: at yaw >= 130 a20 goes 23.6 (pitch -65) -> 19.1 (-55)
  // -> 11.8 (-50) -> 11.0 (-45) -> 14.8 (-40) -> 17.7 (-35), recovering on both
  // shoulders. Areas 113/141/148/153 vanish AS A GROUP, and all four sit in the
  // order list on every frame. Every known reject on the path is now patched
  // (0x2EB9CF, 0x2EBCDC) or measured and anti-correlated (0x2EB8D0 ed900Drop,
  // 0x2EC675 clipDegenA, 0x2EC67F never fires, allocFail = 0), so the loss is
  // upstream of all of them.
  //   live > 0 and it names 113/141/148/153 => they WERE enqueued and the queue
  //     loop never reached them; the bug is the cursor/pending interaction at
  //     0x2EC8CE-0x2EC8E5, not any reject.
  //   live == 0 => they were never enqueued at all, so the crossing INTO them
  //     failed and the cascade starts at their parent area's portal loop.
  // pending should be 0 at exit; a non-zero value means the loop abandoned work.
  std::atomic<uint32_t> g_areaSeedLive{ 0 };
  std::atomic<uint32_t> g_areaSeedLiveN{ 0 };
  std::atomic<uint32_t> g_areaSeedLiveAreas[48];
  std::atomic<uint32_t> g_areaSeedPending{ 0 };

  // NV-DXVK [Ed900Drop] — areas dropped by sub_1802ED900's -1 at 0x2EB8D0.
  //
  // THE STATISTIC THE OLD ONE WAS NOT. ed900 (the call count) sat flat at
  // ~1.0/frame across yaw while a20 fell 5 -> 2, and that was recorded as an
  // acquittal. It is not one. The drop at 0x2EB8D0 happens BEFORE the dispatch
  // at 0x2EB910 and before the portal loop at 0x2EB915, so a single -1 on an
  // area whose portals would have opened the rest removes every crossing behind
  // it — and a flat call count is exactly what one early drop looks like.
  // sub_1802ED900 reads xmmword_1811FC030 (the camera FORWARD) at four sites,
  // so its verdict is view-dependent even where its call count is not.
  //
  // Counted with a tail-jmp counter island on loc_1802EBE71, which xrefs prove
  // is reached from 0x2EB8D0 and nowhere else, so the count cannot merge
  // another path.
  //
  // READ IT binned by yaw against a20 in the same frame. Rising as a20 falls =>
  // this is the gate. Flat at 0 => ED900 is finally eliminated on evidence that
  // matches the mechanism, and the selector is being withheld somewhere else.
  //
  // NOTE IF IT IS THE GATE: 0x2EB8D0 cannot simply be forced not-taken.
  // sub_1802E8A20 does 64 * selector into unk_181380A40, and the selector is -1
  // on that path, so a bypass must first produce a valid record.
  std::atomic<uint64_t> g_ed900DropCount{ 0 };
  std::atomic<uint32_t> g_ed900DropInstalled{ 0 };

  // NV-DXVK [ClipDegen] — portals abandoned at 0x2EC675 / 0x2EC67F because the
  // clipped polygon came out with fewer than 3 vertices.
  //
  // WHAT THIS IS FOR. rtx.cullOff.areaPortal + areaClip removed the YAW
  // dependence — a20 across yaw bins is 9.4 to 27.5 with no collapse — but a
  // PITCH sweep still falls monotonically, 32.67 at +10deg to 6.42 at -50deg,
  // with allocFail = 0 and [AreaSeed] listLen pinned at 30 (min == max) in every
  // pitch bin. So the candidate set is not the cause on this axis either, and
  // the loss is still between the order list and the selector.
  //
  // WHY THESE TWO. Sites 12 and 13 both neutralise predicates that treat all
  // four frustum side planes symmetrically, and a symmetric predicate cannot
  // produce an axis-asymmetric result — so the residual must be a different
  // one. These fire only when a portal genuinely STRADDLES and the real clipper
  // at 0x2EBCF1 runs, and that clipper clips against xmmword_1811FC030, the
  // camera FORWARD. Grazing geometry against floor- and ceiling-adjacent portal
  // edges is what pitching produces.
  //
  // READ THEM binned by pitch, next to a20 on the same line:
  //   rising as a20 falls => this is the gate.
  //   flat at zero        => the loss is a cascade from somewhere else, and the
  //     next probe has to record WHICH AREA stops being visited, not which
  //     portal is rejected.
  //
  // Counted rather than patched on purpose: neither branch has a clean accept
  // target, because 0x2EC685-0x2EC6B2 is the successful-clip fixup that
  // publishes r15d/esi and restores rbx/r13/rdi/r12.
  std::atomic<uint64_t> g_clipDegenA{ 0 };
  std::atomic<uint64_t> g_clipDegenB{ 0 };
  std::atomic<uint32_t> g_clipDegenInstalled{ 0 };
  // The neighbour areas those abandoned crossings would have reached. This is
  // the field that matters — the counts above cannot distinguish "one specific
  // portal died and took four areas with it" from "there was less work to do",
  // which is why the earlier anti-correlation reading of g_clipDegenA was
  // wrong. In the well, look for the area that feeds 113/141/148/153.
  std::atomic<uint32_t> g_clipDegenAreaN{ 0 };
  std::atomic<uint32_t> g_clipDegenAreas[16];
  // Highest portal-index bit actually recorded. The stub masks the index to the
  // table width, so an index past it aliases onto the wrong entry and every
  // degenTo area becomes fiction. The first capture reported area 180 as the
  // most common target on a map with dword_181748D8C = 179 — out of range —
  // which is either an engine sentinel (0x2EC8A8 guards for exactly that) or
  // aliasing. maxBit well below 16383 means no aliasing and the areas stand.
  std::atomic<uint32_t> g_clipDegenMaxBit{ 0 };

  // NV-DXVK [DegenPair] — the joint (r11, r9) histogram of the rejects at
  // 0x2EC675. Cell = r9*4 + r11, r11 clamped at 3, r9 at 15.
  //
  // WHY A PAIR AND NOT TWO COUNTS. g_clipDegenA and g_clipDegenB are each one
  // branch's tally, and the branches are in series: 0x2EC67F is only reached
  // once 0x2EC675 has fallen through. So g_clipDegenB reading 0 forever never
  // meant "planes are never degenerate", it meant "planes are never degenerate
  // among portals that already had >=3 edges" — a different population from
  // the one that decides anything.
  //
  // It decides this: r11 becomes rec[+0] and r9 becomes rec[+2] (0x2EC6A2 /
  // 0x2EC6AC -> 0x2EC6F5-FA), and sub_1802ED900 treats the two completely
  // differently. rec[+0] == 0 is explicitly guarded at 0x1802EDA84. rec[+2]
  // == 0 walks a post-test loop at 0x1802EDA30 with a bound of zero and writes
  // off the end of unk_181E60EF0 at 0x1802EDA45 — the exact instruction and
  // exact access type of the site-14 crash. Relaxing only `cmp r11,3` cannot
  // reach that state, because `cmp r9,3` is left in place; whether it can
  // reach anything USEFUL is what these cells say.
  std::atomic<uint32_t> g_degenPair[64];
  std::atomic<uint32_t> g_degenPairInstalled{ 0 };

  // NV-DXVK [CullOffAB] — in-game A/B of the culling patches, PageUp cycles it.
  //
  // WHY A MODE AND NOT A "FULL MAP" REFERENCE. The obvious idea is to diff
  // against rtx.cullOff.pvs, but that site (client.dll+0x1A8470) is in
  // BuildRenderableRenderLists and only widens RENDERABLES. World surfaces come
  // from a different subsystem entirely — sub_1802EB620's area layer and
  // sub_1802E8DA0's leaf walk — so pvs is not an upper bound for the geometry
  // this investigation is about. There is no single "submit the whole map"
  // switch on the world side; the area layer IS the mechanism.
  //
  // What is meaningful instead is toggling the patches themselves at a fixed
  // camera angle and watching what appears and disappears:
  //   0  configured    every flag exactly as rtx.conf sets it
  //   1  area off      sites 12/13/14 restored, everything else left on — shows
  //                    precisely what the area-layer work contributes
  //   2  all off       every CullOff site restored, i.e. the engine's own
  //                    culling, which is the light leak in its original form
  //
  // cullOffUpdate() reconciles patch state against the options every frame, so
  // flipping this restores or reapplies the byte patches live with no restart.
  //
  // THE COMPLETENESS METRIC IS IN THE LOG, NOT ON SCREEN: [AreaSeed] listLen is
  // the number of candidate areas (30 on this map) and [DispProbe] a20 is how
  // many actually dispatched. a20 reaching listLen and staying there across a
  // full pitch/yaw sweep is what "the area layer is fully open" looks like.
  std::atomic<uint32_t> g_cullOffAbMode{ 0 };

  // NV-DXVK [Ed480Probe]: sub_1802ED480 is the DYNAMIC area enqueue —
  // dword_1811FF91C[a1] = rec; ++dword_1811FC0D8 — reached only through fn-ptr
  // table slots 0x183C54E44/0x183C54E50, i.e. as a job. [AreaDump] showed two
  // areas unconditional (BSP nodes 124, 178, from position-only EAD60) and
  // three view-dependent (127, then 125/149); these are the three, and this
  // counts them. Areas are the a1 argument.
  std::atomic<uint64_t> g_dispProbeEd480Calls{ 0 };
  std::atomic<uint32_t> g_dispProbeEd480N{ 0 };
  std::atomic<uint32_t> g_dispProbeEd480Area[8];

  // NV-DXVK [DrainProbe]: the world visibility worker's OUTPUT, measured at
  // the drain (client.dll sub_1802F04F0). Same writer/reader split as
  // [JobProbe] above and the same link reason for living on this side.
  //
  // WHY THE OUTPUT AND NOT THE INPUT. [JobProbe] settled the input: job
  // supply is FLAT across pitch (192 -> 184 calls/frame, non-monotonic, sd=0
  // over the last three bins) while instance count falls 23% (r = -0.77) on a
  // camera that moved 9 units. So the view dependence is not in how much work
  // is dispatched, and handoff §6's "go into sub_1802EB620" is excluded by
  // measurement.
  //
  // The drain ORs accepted leaf runs into the mask and applies NO test of any
  // kind (verified at 0x1802F0594: `or [r9], rdx`, no compare on any path),
  // so a popcount of the mask AFTER it returns IS the accepted-leaf count.
  //
  // READ IT: bin m1/m2 by pitchDeg.
  //   m1/m2 FALL with pitch  => the worker is still rejecting despite the §2
  //     patches, i.e. that reject set is not exhaustive after all. Go back
  //     into sub_1802E8DA0.
  //   m1/m2 FLAT with pitch  => the mask is fully populated and the loss is
  //     entirely downstream of it — the consumers, not the visibility build.
  //     Stop looking at the world visibility subsystem.
  std::atomic<uint64_t> g_drainProbeCalls{ 0 };
  std::atomic<uint64_t> g_drainProbeBad{ 0 };
  // Summed per frame across drain calls, plus the per-frame maximum, because
  // the drain may run once per view — a sum alone cannot tell "one big view"
  // from "several small ones", and the main view is the largest.
  std::atomic<uint64_t> g_drainProbeM1Sum{ 0 };
  std::atomic<uint64_t> g_drainProbeM2Sum{ 0 };
  std::atomic<uint64_t> g_drainProbeRSum{ 0 };
  std::atomic<uint32_t> g_drainProbeM1Max{ 0 };
  std::atomic<uint32_t> g_drainProbeM2Max{ 0 };
  // Last observed region header, and whether it passed the structural check
  // (c7C == c78 + c80). A run with layoutOk=0 is not interpretable.
  std::atomic<uint32_t> g_drainProbeC70{ 0 };
  std::atomic<uint32_t> g_drainProbeC74{ 0 };
  std::atomic<uint32_t> g_drainProbeC78{ 0 };
  std::atomic<uint32_t> g_drainProbeLayoutOk{ 0 };
}}

#include "dxvk_device.h"
#include "rtx_resources.h"
#include "rtx_engine_symbols.h"

// NV-DXVK [classify-eye-truth]: the engine's ground-truth eye position, used
// by the [CamMgr.classify] and [ZigCam] diagnostics to compare each Main
// candidate's recovered camera position against what the engine actually
// thinks the eye is.
//
// HISTORY -- why this no longer touches client.dll at all.
//
// This used to call a hardcoded `client.dll + 0x14EAE0` as a nullary
// `void*(*)()` ("GetLocalPlayer") and read `player + 0x3D6C`. Both halves
// were wrong on the shipped v2.0.11.0 build, and the first half was fatal:
//
//   * 0x14EAE0 is not a function entry on this build. It is +0x70 into an
//     unrelated float-lerp on the player camera block. Calling it entered a
//     function body having skipped the prologue that reserves stack and saves
//     xmm6, so the epilogue's `movaps xmm6,[rsp]` hit a misaligned rsp and
//     faulted -- reported by Windows as an access violation reading
//     0xFFFFFFFFFFFFFFFF, which is the SSE alignment sentinel rather than a
//     real bad pointer. Every frame, from CameraManager::onFrameEnd.
//
//     The old guards could not catch this. They proved the module was still
//     loaded and the page was committed and executable -- both true of every
//     byte inside every function. Nothing checked "is this a function ENTRY",
//     which is the question that mattered.
//
//   * +0x3D6C is not a live eye field on this build either. The
//     [classify-eye-truth] log showed it pinned at (14158,-10801,877) across
//     46 km of player travel: a static script anchor.
//
// Re-deriving those two numbers for this build would work exactly until the
// next game patch. So they are gone. The eye now comes from the viewmodel
// pass's c_cameraOrigin -- Source's authoritative eye on this build, correct
// in pilot-on-foot, titan-cockpit and rodeo -- which Remix already captures
// per draw with no module offsets whatsoever. See dxvk::getEngineEyePosition
// and rtx_engine_symbols.h for the resolution contract.
//
// Returns nullptr when no eye is available yet. It cannot crash: there is no
// game code on this path any more.
namespace {
  inline const float* GetEngineEyeCM() {
    // Thread-local so the returned pointer stays valid for the caller's
    // formatting without handing out a pointer into shared mutable state.
    // onFrameEnd (CS thread) and processCameraData both call this.
    // Qualified: this anonymous namespace sits at global scope (the dxvk::tf2
    // block above it is already closed), so the unqualified name would not
    // find the declaration in namespace dxvk.
    static thread_local float s_eye[3] = { 0.f, 0.f, 0.f };
    if (!dxvk::getEngineEyePosition(s_eye))
      return nullptr;
    return s_eye;
  }

  // (ViewModelEyeCache + eye-snap killswitch + ramp constants/globals all
  // deleted; see consolidated comment in processCameraData.)
}

namespace {
  constexpr float kFovToleranceRadians = 0.001f;

  // NV-DXVK TF2: per-frame histogram of Main-candidate reject reasons. Populated
  // by processCameraData and dumped by CameraManager::onFrameEnd so we can see
  // which specific gate blocks Main latches on frames where the camera fails
  // to update. Counters are reset when the observed frameId advances.
  struct MainRejectHistogram {
    uint32_t frameId = UINT32_MAX;
    uint32_t candidates = 0;
    uint32_t accepted = 0;
    // Physical gates.
    uint32_t rejIsInWorld = 0;
    uint32_t rejIsNonSquare = 0;
    uint32_t rejIsReasonableDepth = 0;
    uint32_t rejIsReasonableFov = 0;
    uint32_t rejIsLargeEnough = 0;
    // Hysteresis gates.
    uint32_t rejVpMatches = 0;
    uint32_t rejMaxZMatches = 0;
    uint32_t rejFovClose = 0;
    uint32_t rejBasisClose = 0;
    uint32_t rejStreakNotMet = 0;
  };
  MainRejectHistogram g_mainHist;

  inline void noteFrame(uint32_t frameId) {
    if (g_mainHist.frameId != frameId) {
      g_mainHist = MainRejectHistogram{};
      g_mainHist.frameId = frameId;
    }
  }
}

namespace dxvk {

  // NV-DXVK [VanishEdge]: latest Main-camera pose, stashed each frame by [CullCmp]
  // (onFrameEnd) and read by InstanceManager::garbageCollection's ship-vanish edge
  // detector (cross-file). Same thread, written once/frame; the frame stamp lets the
  // reader confirm freshness. These ARE the exact RtCamera values [CullCmp] logs.
  float g_veCamPx = 0.f, g_veCamPy = 0.f, g_veCamPz = 0.f;
  float g_veCamDx = 0.f, g_veCamDy = 0.f, g_veCamDz = 0.f;
  float g_veCamFov = 0.f;
  std::atomic<uint32_t> g_veCamFrame { 0xFFFFFFFFu };

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    return accessCamera(*this, cameraType).isValid(m_device->getCurrentFrameId());
  }

  void CameraManager::onFrameEnd() {
    // NV-DXVK TF2: dump the Main-candidate reject histogram for the frame
    // that just ended. One line per frame — makes it trivial to spot which
    // gate is blocking Main updates when the camera lags the player.
    if (g_mainHist.frameId != UINT32_MAX && g_mainHist.candidates > 0) {
      Logger::info(str::format(
        "[CamMgr.hist] frame=", g_mainHist.frameId,
        " cand=", g_mainHist.candidates,
        " accept=", g_mainHist.accepted,
        " phys{inWorld=", g_mainHist.rejIsInWorld,
        " nonSq=", g_mainHist.rejIsNonSquare,
        " depth=", g_mainHist.rejIsReasonableDepth,
        " fov=", g_mainHist.rejIsReasonableFov,
        " large=", g_mainHist.rejIsLargeEnough,
        "} hyst{vp=", g_mainHist.rejVpMatches,
        " maxZ=", g_mainHist.rejMaxZMatches,
        " fovClose=", g_mainHist.rejFovClose,
        " basisClose=", g_mainHist.rejBasisClose,
        " streak=", g_mainHist.rejStreakNotMet,
        "}"));
    }
    // NV-DXVK [ZigCam]: per-frame confirm for the ship/weapon zig-zag. The
    // extraction (path1/path3) is verified-fine; Main is engine-hook-locked
    // (useEngineHookMainCamera=True, engineEye stable). Hypothesis: the gun
    // wobbles because the ViewModel camera is NOT engine-suppressed (it
    // free-runs per-draw, see suppression comment ~710) while Main is stable.
    // This dumps all three positions once per frame, UN-throttled (the existing
    // classify logs cap at 400 and are exhausted at bootstrap). If Main tracks
    // engineEye steadily while ViewModel.x wobbles frame-to-frame, the fix is
    // ViewModel-side, not Main. Prefix not in log.cpp filter. Remove once fixed.
    {
      const uint32_t zcFrame = m_device->getCurrentFrameId();
      const RtCamera& zcMain = getCamera(CameraType::Main);
      const RtCamera& zcVm   = getCamera(CameraType::ViewModel);
      const bool zcMainValid = zcMain.isValid(zcFrame);
      const bool zcVmValid   = zcVm.isValid(zcFrame);
      const Vector3 zcMainP = zcMainValid ? zcMain.getPosition() : Vector3(0, 0, 0);
      const Vector3 zcVmP   = zcVmValid   ? zcVm.getPosition()   : Vector3(0, 0, 0);
      const float* zcEye = GetEngineEyeCM();
      const bool zcHaveEye =
        zcEye && std::isfinite(zcEye[0]) && std::isfinite(zcEye[1]) && std::isfinite(zcEye[2]);
      Logger::info(str::format(
        "[ZigCam] f=", zcFrame,
        " mainValid=", zcMainValid ? 1 : 0,
        " main=(", zcMainP.x, ",", zcMainP.y, ",", zcMainP.z, ")",
        " vmValid=", zcVmValid ? 1 : 0,
        " vm=(", zcVmP.x, ",", zcVmP.y, ",", zcVmP.z, ")",
        " engineEye=", zcHaveEye ? "valid" : "null",
        " eye=(", zcHaveEye ? zcEye[0] : 0.f, ",", zcHaveEye ? zcEye[1] : 0.f, ",", zcHaveEye ? zcEye[2] : 0.f, ")"));
    }
    // NV-DXVK [CullCmp]: vanishing-ship probe. The game raster-culls renderables
    // with its OWN cull frustum (client.dll, per-view buffer), but Remix path-
    // traces with this engine-hook-locked Main camera. If the two diverge in
    // forward axis or FOV, geometry the RT camera can see but the game culled is
    // simply absent from the BVH -> on-screen ship structure vanishes. This dumps
    // the RT Main camera forward+pos+fov once per frame so it can be compared
    // against the live game cull frustum (a2[3].xyz forward + apex) read from the
    // debugger in the same (held, static) geo-missing view. Prefix not in
    // log.cpp filter. Remove once the divergence is characterized.
    {
      const uint32_t ccFrame = m_device->getCurrentFrameId();
      const RtCamera& ccMain = getCamera(CameraType::Main);
      if (ccMain.isValid(ccFrame)) {
        const Vector3 ccDir = ccMain.getDirection();
        const Vector3 ccPos = ccMain.getPosition();
        Logger::info(str::format(
          "[CullCmp] f=", ccFrame,
          " renderFwd=(", ccDir.x, ",", ccDir.y, ",", ccDir.z, ")",
          " renderPos=(", ccPos.x, ",", ccPos.y, ",", ccPos.z, ")",
          " fovRad=", ccMain.getFov()));
        // [VanishEdge]: publish this pose for the GC-side edge detector.
        g_veCamPx = ccPos.x; g_veCamPy = ccPos.y; g_veCamPz = ccPos.z;
        g_veCamDx = ccDir.x; g_veCamDy = ccDir.y; g_veCamDz = ccDir.z;
        g_veCamFov = ccMain.getFov();
        g_veCamFrame.store(ccFrame, std::memory_order_release);
      }
    }
    m_lastSetCameraType = CameraType::Unknown;
    m_decompositionCache.clear();
  }

  CameraType::Enum CameraManager::processCameraData(const DrawCallState& input) {
    // [pcdEnter] One-per-frame log proving processCameraData was called.
    // If pcdTrace later in the function NEVER fires for a frame, but
    // pcdEnter does, we know the function was called but every call
    // bailed via one of the early returns (identity v2p, fused-world-view
    // path, fov/shear validation). Then the exit-path log below will
    // confirm which return type each call ended up at.
    {
      const uint32_t fid = m_device->getCurrentFrameId();
      static std::atomic<uint32_t> sLastEnterFrame{UINT32_MAX};
      uint32_t expected = sLastEnterFrame.load(std::memory_order_relaxed);
      if (expected != fid) {
        if (sLastEnterFrame.compare_exchange_strong(expected, fid,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          const auto& w = input.getTransformData().worldToView;
          const auto& vtp = input.getTransformData().viewToProjection;
          const float tR = float(w[3][0]), tU = float(w[3][1]), tF = float(w[3][2]);
          const float camX = -(float(w[0][0])*tR + float(w[0][1])*tU + float(w[0][2])*tF);
          const float camY = -(float(w[1][0])*tR + float(w[1][1])*tU + float(w[1][2])*tF);
          const float camZ = -(float(w[2][0])*tR + float(w[2][1])*tU + float(w[2][2])*tF);
          Logger::info(str::format(
            "[pcdEnter] frame=", fid,
            " camPos=(", camX, ",", camY, ",", camZ, ")",
            " v2pIdent=", isIdentityExact(vtp) ? 1 : 0,
            " skyCat=", input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0));
        }
      }
    }

    // If theres no real camera data here - bail
    if (isIdentityExact(input.getTransformData().viewToProjection)) {
      return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }

    switch (RtxOptions::fusedWorldViewMode()) {
    case FusedWorldViewMode::None:
      if (input.getTransformData().objectToView == input.getTransformData().objectToWorld && !isIdentityExact(input.getTransformData().objectToView)) {
        return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
      }
      break;
    case FusedWorldViewMode::View:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if World is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToWorld),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to View but World transform is not identity!"));
      }
      break;
    case FusedWorldViewMode::World:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if View is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToView),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to World but View transform is not identity!"));
      }
      break;
    }

    // Get camera params
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(input.getTransformData().viewToProjection);

    // Filter invalid cameras, extreme shearing
    static auto isFovValid = [](float fovA) {
      return fovA >= kFovToleranceRadians;
    };
    static auto areFovsClose = [](float fovA, const RtCamera& cameraB) {
      return std::abs(fovA - cameraB.getFov()) < kFovToleranceRadians;
    };

    if (std::abs(decomposeProjectionParams.shearX) > 0.01f || !isFovValid(decomposeProjectionParams.fov)) {
      // NV-DXVK [SpawnGeomDiag]: was ONCE() — flipped to a per-frame
      // throttled warn that prints actual decomposition so the missing-
      // geometry debug can correlate "frame N rejected camera with
      // shearX=…/fov=…" against the [SpawnGeomDiag] frame=N census.
      // ONCE() suppressed every rejection after the first, which made
      // spawn-window analysis blind once any earlier UI/sky pass ate the
      // single allowed message.
      const uint32_t fid = m_device->getCurrentFrameId();
      static uint32_t sLastWarnFrame = UINT32_MAX;
      if (fid != sLastWarnFrame) {
        sLastWarnFrame = fid;
        const bool isSky = input.getCategoryFlags().test(InstanceCategories::Sky);
        const bool fovBad = !isFovValid(decomposeProjectionParams.fov);
        const bool shearBad = std::abs(decomposeProjectionParams.shearX) > 0.01f;
        // [v2pReject] Dump the full viewToProjection matrix that's being
        // rejected. shearX=0.955 + nearPlane=-0 + aspect=-0.028 is suspicious
        // — looks more like the decomposer is misinterpreting TF2's matrix
        // layout (possibly transposed / Y-flipped / different handedness)
        // than it being a genuinely sheared projection. The matrix dump
        // lets us decide whether to relax the gate, fix the decomposer,
        // or transpose the input before decomposition.
        const auto& v2p = input.getTransformData().viewToProjection;
        Logger::warn(str::format(
          "[RTX] CameraManager: rejected an invalid camera",
          " frame=", fid,
          " sky=", (isSky ? 1 : 0),
          " fov=", decomposeProjectionParams.fov,
          " fovBad=", (fovBad ? 1 : 0),
          " shearX=", decomposeProjectionParams.shearX,
          " shearBad=", (shearBad ? 1 : 0),
          " nearPlane=", decomposeProjectionParams.nearPlane,
          " aspectRatio=", decomposeProjectionParams.aspectRatio,
          " => CameraType::", (isSky ? "Sky" : "Unknown")));
        Logger::warn(str::format(
          "[v2pReject] frame=", fid,
          " row0=(", v2p[0][0], ",", v2p[0][1], ",", v2p[0][2], ",", v2p[0][3], ")",
          " row1=(", v2p[1][0], ",", v2p[1][1], ",", v2p[1][2], ",", v2p[1][3], ")",
          " row2=(", v2p[2][0], ",", v2p[2][1], ",", v2p[2][2], ",", v2p[2][3], ")",
          " row3=(", v2p[3][0], ",", v2p[3][1], ",", v2p[3][2], ",", v2p[3][3], ")"));
      }
      return input.getCategoryFlags().test(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }


    auto isViewModel = [this](float fov, float maxZ, uint32_t frameId) {
      // NV-DXVK [VM.check]: trace every invocation so we can see why the
      // viewmodel is / isn't being classified. Throttled per-frame.
      const float vmThr = RtxOptions::ViewModel::maxZThreshold();
      const bool vmEnable = RtxOptions::ViewModel::enable();
      {
        static uint32_t sLastVMFrame = 0;
        static uint32_t sVMLogCount = 0;
        if (frameId != sLastVMFrame) { sLastVMFrame = frameId; sVMLogCount = 0; }
        if (sVMLogCount < 32) {
          ++sVMLogCount;
          Logger::info(str::format(
            "[VM.check] f=", frameId,
            " maxZ=", maxZ,
            " fov=", fov,
            " thr=", vmThr,
            " enable=", (vmEnable ? 1 : 0),
            " maxZHit=", (vmEnable && maxZ <= vmThr ? 1 : 0)));
        }
      }
      if (vmEnable) {
        // TF2's first-person viewmodel pass is identified engine-natively by a
        // compressed viewport depth range (D3D11_VIEWPORT.MaxDepth <= ~0.08, so
        // the gun never z-clips through world geometry). dcs.maxZ mirrors that
        // viewport value, so this is the authoritative ViewModel signal.
        //
        // The previous FoV-mismatch fallback ("FoV differs from Main => assume
        // ViewModel") was removed: in TF2 the viewmodel and main passes share
        // the same projection FoV, so that path never identified a real
        // viewmodel — it only fired on legitimate transient world-FoV changes
        // (stance transitions / spawn frames), mis-tagging world geometry
        // (mountains, sky) as ViewModel and dropping it from the TLAS.
        if (maxZ <= vmThr) {
          return true;
        }
      }
      return false;
    };

    const uint32_t frameId = m_device->getCurrentFrameId();

    auto cameraType = CameraType::Main;
    if (input.isDrawingToRaytracedRenderTarget) {
      cameraType = CameraType::RenderToTexture;
    } else if (input.testCategoryFlags(InstanceCategories::Sky)) {
      cameraType = CameraType::Sky;
    } else if (isViewModel(decomposeProjectionParams.fov, input.maxZ, frameId)) {
      cameraType = CameraType::ViewModel;
    }

    // (viewmodel-eye-capture deleted: only consumer was the deleted snap
    // block's g_vmEye fallback. The d3d11_rtx pilot-eye atomic — captured
    // at the viewmodel-pass fanout site — provides the same data through
    // a different path and is kept for diagnostics.)

    // NV-DXVK [VM.class]: log every camera-type decision so we can see the
    // post-isViewModel result. Throttled per frame.
    {
      static uint32_t sLastVMClassFrame = 0;
      static uint32_t sVMClassLog = 0;
      if (frameId != sLastVMClassFrame) { sLastVMClassFrame = frameId; sVMClassLog = 0; }
      if (sVMClassLog < 32) {
        ++sVMClassLog;
        Logger::info(str::format(
          "[VM.class] f=", frameId,
          " maxZ=", input.maxZ,
          " fov=", decomposeProjectionParams.fov,
          " type=", static_cast<uint32_t>(cameraType),
          " isRT=", (input.isDrawingToRaytracedRenderTarget ? 1 : 0),
          " isSky=", (input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0)));
      }
    }

    // NV-DXVK [VM.classVM]: dedicated non-throttled log for ViewModel
    // classifications only, with VS hash so we can identify which draws
    // are being mis-tagged. De-duped by (vsHash,frameId) to keep volume
    // sane when 130+ instances share one VS on the bootstrap frame.
    if (cameraType == CameraType::ViewModel) {
      const uint64_t vsHash = static_cast<uint64_t>(
        input.getTransformData().vertexShaderHash);
      static std::mutex sVMClassVMMu;
      static std::unordered_set<uint64_t> sVMClassVMSeen;
      const uint64_t key = vsHash ^ (uint64_t(frameId) * 0x9e3779b1ull);
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sVMClassVMMu);
        first = sVMClassVMSeen.insert(key).second;
      }
      if (first) {
        Logger::info(str::format(
          "[VM.classVM] f=", frameId,
          " vsHash=0x", std::hex, vsHash, std::dec,
          " maxZ=", input.maxZ,
          " fov=", decomposeProjectionParams.fov,
          " mainFov=", (getCamera(CameraType::Main).isValid(frameId)
                       ? getCamera(CameraType::Main).getFov() : -1.0f),
          " mainLatchFrame=", getMainClassifierFrameId(),
          " mainByClass=", (isMainSetByClassifier() ? 1 : 0)));
      }
    }

    // NV-DXVK: Deterministic Main-camera classifier — game-native per-draw
    // identity, no matrix-property heuristics.
    //
    // Empirically (probe I), TF2's BSP gameplay-world pass is drawn by a small
    // stable set of vertex shaders with a compressed depth range (maxZ ~ 0.05),
    // while fullscreen / HUD / post passes that happen to share a similar
    // projection shape bind maxZ=1.0 and w2vT≈identity. Shader hash + maxZ
    // band uniquely identifies the real-pose world draws. Everything else
    // falls back to Unknown and never wins the Main latch.
    //
    // Hashes are DxvkShader::getHash() values observed in remix-dxvk.log at
    // the 59.84° gameplay FoV with real player-pose worldToView translation.
    // If TF2 ships a shader update, these will need re-identification via
    // probe I (look for the cluster whose w2vT matches player world coords).
    if (cameraType == CameraType::Main) {
      // NV-DXVK: Main-camera classifier — physical-property gates only,
      // no hash allowlist. The hash allowlist was too narrow: it caught
      // TF2's gameplay-world pass (3 specific BSP shaders, maxZ=0.05) but
      // missed the cinematic camera which uses different shaders and the
      // standard depth range (maxZ=1.0). Both share the player's actual
      // world coordinate frame (w2vT magnitude ~10⁴), so the right criterion
      // is "is this draw in world space?" not "is this a specific shader?".
      //
      // Three checks (any failure → Unknown, no Main update):
      // 1. |w2vT| > 100: rejects fullscreen/UI/composite passes that share a
      //    gameplay-shaped projection but render at origin (|w2vT| < 10).
      // 2. viewport aspect != 1 (non-square): rejects shadow cascades and
      //    cubemap face renders.
      // 3. maxZ in (0, 1.5]: rejects degenerate viewport configs.
      const auto& td = input.getTransformData();
      const Matrix4& w2v = td.worldToView;
      const float w2vMagSq =
        w2v[3][0]*w2v[3][0] + w2v[3][1]*w2v[3][1] + w2v[3][2]*w2v[3][2];
      // "Real camera" check. TF2 has TWO conventions:
      //   • Cinematic / external view: worldToView translation = world-space
      //     player position (magnitude ~10⁴). Rotation = camera orientation.
      //   • Actual gameplay: camera-local vertex space. worldToView translation
      //     ≈ 0, but rotation is still the camera's view rotation (player
      //     looking around).
      // Both are real cameras. The case we want to REJECT is fullscreen/UI/
      // composite passes where worldToView is the FULL identity matrix
      // (rotation = I AND translation = 0). Detect that specifically.
      const bool transNearZero = w2vMagSq < (1.0f * 1.0f);
      const bool rotIsIdentity =
        std::abs(w2v[0][0] - 1.0f) < 0.01f && std::abs(w2v[0][1]) < 0.01f && std::abs(w2v[0][2]) < 0.01f &&
        std::abs(w2v[1][0]) < 0.01f && std::abs(w2v[1][1] - 1.0f) < 0.01f && std::abs(w2v[1][2]) < 0.01f &&
        std::abs(w2v[2][0]) < 0.01f && std::abs(w2v[2][1]) < 0.01f && std::abs(w2v[2][2] - 1.0f) < 0.01f;
      // NV-DXVK TF2: also reject any candidate whose world translation is
      // near zero, regardless of rotation. ExtractTransforms path 1 always
      // bakes the real camera world position into w2v[3] (= -dot(axis,camPos)),
      // so a path-1 output with |w2v[3]| ≈ 0 means the per-draw cb2 RDEF read
      // returned a stale/HUD/identity camera (e.g. (0.0004,0,0)). Without this
      // gate, such a candidate could win Main's first latch and freeze the
      // camera at origin while gameplay draws update body geometry → visible
      // body-races-ahead-of-camera lag. Threshold 10 units handles spawn
      // points near origin while reliably catching the (~0,~0,~0) garbage.
      const bool transTooSmall = w2vMagSq < (10.0f * 10.0f);
      const bool isInWorld = !(transNearZero && rotIsIdentity) && !transTooSmall;
      const float vw = td.viewportWidth;
      const float vh = td.viewportHeight;
      const float vpAspect = (vh > 0.0f) ? (vw / vh) : 0.0f;
      const bool isNonSquare = (vw > 0.0f && vh > 0.0f) &&
                               std::abs(vpAspect - 1.0f) >= 0.02f;
      const bool isReasonableDepth = input.maxZ > 0.0f && input.maxZ <= 1.5f;
      // FoV sanity. Standard game cameras (gameplay, cinematic, mech cockpit)
      // are 30°–120°. TF2 also issues:
      //   • ~140°/160°/147°: cubemap / reflection / fog volume cameras (wide).
      //   • ~179.9°: degenerate fog/volume math (essentially flat projection).
      // Latching Main on any of these produces the rainbow-scanline garbage
      // because volume rendering downstream expects a sane frustum.
      const float fovDeg = decomposeProjectionParams.fov * (180.0f / 3.14159265f);
      const bool isReasonableFov = fovDeg > 30.0f && fovDeg < 120.0f;
      // NV-DXVK (fix 2): minimum viewport size gate. isNonSquare above already
      // rejects the 1024×1024 / 128×128 / 16×16 / 1×1 shadow & probe viewports
      // whose aspect is exactly 1, but TF2 also issues 640×360 / 1280×720 /
      // 80×360 / 160×360 viewports with ~16:9 aspect — HUD compositing,
      // thumbnails, minimaps — that were previously latching as Main and
      // causing the flick. Require a minimum pixel count.
      //
      // NV-DXVK [pilot-on-foot half-res fix]: lowered from 1200×600 to
      // 800×400 so half-res gameplay viewports (960×540) ARE candidates.
      // For on-foot pilot, TF2 renders gameplay at half-res then composites
      // to 1920×1080 with a different cam (gun-pose origin Z≈32). The old
      // 1200×600 threshold demoted the legitimate 960×540 player-eye draws
      // and let the 1920×1080 composite (recovC Z=31, 60u below engineEye
      // Z=92) win Main → "camera in the ground". 800×400 keeps cubemap /
      // thumbnail viewports out (square at 256/512/1024 already filtered
      // by isNonSquare; 640×360 minimaps are below 400 height). Engine-eye
      // delta filters anything below 800×400 that slips through.
      const bool isLargeEnough = vw >= 800.0f && vh >= 400.0f;
      // (engineEye-reject deleted: the only working ref source on this
      // build was the pilot-eye atomic, which actually carries the gun-
      // pose Z=32 — not the player eye Z=92 — so the reject couldn't
      // distinguish a wrong-pose candidate from the legitimate one. Probe
      // showed it never fired in production; the isLargeEnough lower bound
      // (800×400) is what actually keeps the half-res player draws as Main
      // candidates and the gun-pose composite filtered out by other gates.)
      bool keepAsMain =
        isInWorld && isNonSquare && isReasonableDepth && isReasonableFov && isLargeEnough;
      noteFrame(frameId);
      ++g_mainHist.candidates;
      if (!isInWorld) ++g_mainHist.rejIsInWorld;
      if (!isNonSquare) ++g_mainHist.rejIsNonSquare;
      if (!isReasonableDepth) ++g_mainHist.rejIsReasonableDepth;
      if (!isReasonableFov) ++g_mainHist.rejIsReasonableFov;
      if (!isLargeEnough) ++g_mainHist.rejIsLargeEnough;
      if (!keepAsMain) {
        static uint32_t sVpLog = 0;
        if (sVpLog < 40) {
          ++sVpLog;
          char vsHex[32];
          std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                        static_cast<unsigned long long>(td.vertexShaderHash));
          Logger::info(str::format(
            "[CamMgr] Demoted-from-Main #", sVpLog,
            " vsHash=", vsHex,
            " viewport=", int(vw), "x", int(vh),
            " maxZ=", input.maxZ,
            " fov=", fovDeg, "deg",
            " |w2vT|=", std::sqrt(w2vMagSq),
            " isInWorld=", isInWorld ? 1 : 0,
            " isNonSquare=", isNonSquare ? 1 : 0,
            " isReasonableDepth=", isReasonableDepth ? 1 : 0,
            " isReasonableFov=", isReasonableFov ? 1 : 0,
            " isLargeEnough=", isLargeEnough ? 1 : 0));
        }
        cameraType = CameraType::Unknown;
      } else {
        // NV-DXVK (fixes 1 + 3): latch hysteresis. Once Main is latched by the
        // classifier, subsequent candidates that pass the physical gates must
        // also look CONSISTENT with the existing latch — same FoV (±3°), same
        // viewport (±4 px), same forward direction (dot > 0.5, i.e. within
        // ~60° — accommodates normal mouse look but rejects the 90° axis
        // twists seen in the log between wtvPathId=1 and wtvPathId=3). On
        // kCutStreakThreshold consecutive disagreements we assume a real cut
        // and allow the re-latch. Without this, every draw that passes the
        // gates overwrites Main — and multiple draws per frame pass, so Main
        // flickers between shadow/reflection/gameplay poses.
        const uint32_t curFrameId = m_device->getCurrentFrameId();
        const auto& snap = m_mainLatchSnapshot;
        // Snapshot counts as fresh if it was set within the last couple of
        // frames. Older than that, we assume the view was paused/stale and
        // allow a fresh latch unconditionally.
        const bool snapFresh =
          snap.valid
          && (curFrameId <= snap.frameId || (curFrameId - snap.frameId) <= 3);
        if (snapFresh) {
          const float fovDiff = std::abs(decomposeProjectionParams.fov - snap.fovRad);
          const bool fovClose = fovDiff < 0.052f; // ~3 degrees
          const bool vpMatches =
            std::abs(vw - snap.viewportW) < 4.0f && std::abs(vh - snap.viewportH) < 4.0f;
          // Forward from this draw's worldToView row-major convention: col 2.
          const Vector3 newFwd(w2v[0][2], w2v[1][2], w2v[2][2]);
          const float newFwdLen2 =
            newFwd.x*newFwd.x + newFwd.y*newFwd.y + newFwd.z*newFwd.z;
          const float dot =
            (newFwdLen2 > 0.001f)
              ? (newFwd.x*snap.fwd.x + newFwd.y*snap.fwd.y + newFwd.z*snap.fwd.z)
              : 0.0f;
          // NV-DXVK TF2: forward-vector check — same-hemisphere (~90° cap).
          // Allows normal fast mouse look while rejecting 180° axis flips.
          const bool fwdClose = dot > 0.0f;
          // NV-DXVK TF2: also compare the right vector so roll around the
          // forward axis is part of the basis check. Two candidate draws can
          // share a forward direction yet differ by a 90° roll (TF2 produces
          // both), and without this check either pose can win the Main latch
          // — the camera then renders sideways. 0.7 ≈ cos(45°): tolerates
          // moderate roll drift (camera tilt anims, lean), rejects 90°+ flips.
          const Vector3 newRight(w2v[0][0], w2v[1][0], w2v[2][0]);
          const float newRightLen2 =
            newRight.x*newRight.x + newRight.y*newRight.y + newRight.z*newRight.z;
          const float rightDot =
            (newRightLen2 > 0.001f)
              ? (newRight.x*snap.right.x + newRight.y*snap.right.y + newRight.z*snap.right.z)
              : 0.0f;
          const bool rightClose = rightDot > 0.7f;
          const bool basisClose = fwdClose && rightClose;
          // NV-DXVK: differentiate hard and soft rejects. A viewport mismatch
          // is a DIFFERENT RENDER PASS (HUD compositing, scope zoom, minimap
          // preview, thumbnail) — not a camera cut. Accepting it as a "cut"
          // after N tries just means the wrong render pass steals Main. So
          // viewport-wrong candidates are HARD-rejected and never contribute
          // to the cut streak. Only FoV-change + basis-change count, because
          // those are real camera cuts (level change, teleport, cinematic).
          // NV-DXVK TF2 FIX: also HARD-reject maxZ mismatches. Each TF2
          // render pass uses a distinct viewport depth range — main world
          // is 0.1, viewmodel 0.05, shadow 1.0, probe/env 1.0 at different
          // resolutions. A candidate with different maxZ is a DIFFERENT
          // render pass, not a camera cut. Without this gate, every frame
          // we alternate between passes and Main oscillates → visible flash
          // on frame 1 + subsequent-frame geometry pops.
          const bool maxZMatches = std::abs(input.maxZ - snap.maxZ) < 0.01f;
          // NV-DXVK TF2: intra-frame position-magnitude check. With multi-latch
          // enabled (last-wins per frame), a later draw whose worldToView
          // carries a wildly different translation magnitude from the snapshot
          // is almost certainly a different render pass that happens to share
          // viewport/maxZ/basis (e.g. a camera-relative HUD overlay drawn at a
          // small offset from origin). Rejecting these intra-frame protects
          // Main from being yanked to a wrong pose by the last passing draw.
          // Only enforced when the snapshot was set THIS frame (intra-frame
          // refinement); cross-frame motion goes through the wider basis/fov
          // path. Tolerance is loose (200 world units) — the player can move
          // ~150 u/s, and we just need to filter the obvious order-of-magnitude
          // mismatches like (5188 vs 50) without rejecting same-scene refinements.
          const float newW2vTMag = std::sqrt(w2vMagSq);
          const float snapPosMag = std::sqrt(
            snap.pos.x*snap.pos.x + snap.pos.y*snap.pos.y + snap.pos.z*snap.pos.z);
          const bool intraFrame = (snap.frameId == curFrameId);
          const bool posMagOk = !intraFrame
            || std::abs(newW2vTMag - snapPosMag) < 500.0f;
          if (!vpMatches) {
            ++g_mainHist.rejVpMatches;
            static uint32_t sHystLog = 0;
            if (sHystLog < 40) {
              ++sHystLog;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (wrong viewport)",
                " vp=(", int(vw), "x", int(vh), ")",
                " snapVp=(", int(snap.viewportW), "x", int(snap.viewportH), ")"));
            }
            cameraType = CameraType::Unknown;
          } else if (!maxZMatches) {
            ++g_mainHist.rejMaxZMatches;
            static uint32_t sHystLog2 = 0;
            if (sHystLog2 < 40) {
              ++sHystLog2;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (wrong maxZ)",
                " maxZ=", input.maxZ,
                " snapMaxZ=", snap.maxZ));
            }
            cameraType = CameraType::Unknown;
          } else if (!posMagOk) {
            // Same frame, vp/maxZ match, but eye-space translation magnitude
            // is far from the in-frame latched pose. Different render pass.
            static uint32_t sHystLog3 = 0;
            if (sHystLog3 < 40) {
              ++sHystLog3;
              Logger::info(str::format(
                "[CamMgr.hyst] HARD reject (intra-frame |w2vT| mismatch)",
                " newMag=", newW2vTMag,
                " snapMag=", snapPosMag));
            }
            cameraType = CameraType::Unknown;
          } else if (!fovClose || !basisClose) {
            if (!fovClose) ++g_mainHist.rejFovClose;
            if (!basisClose) ++g_mainHist.rejBasisClose;
            ++m_disagreeStreak;
            // NV-DXVK TF2: reduced from 8 → 3. The streak exists to suppress
            // intra-frame flicker between competing candidate draws, not to
            // suppress legitimate frame-to-frame motion. Eight frames meant
            // the camera could be 8 frames behind reality before accepting a
            // re-latch, which was a major contributor to the main-camera lag
            // observed while walking. Three is enough to filter same-frame
            // multi-candidate noise while tracking real motion promptly.
            constexpr uint32_t kCutStreakThreshold = 3;
            if (m_disagreeStreak < kCutStreakThreshold) {
              ++g_mainHist.rejStreakNotMet;
              static uint32_t sHystLog = 0;
              if (sHystLog < 40) {
                ++sHystLog;
                Logger::info(str::format(
                  "[CamMgr.hyst] reject streak=", m_disagreeStreak,
                  " fovClose=", fovClose ? 1 : 0,
                  " fwdClose=", fwdClose ? 1 : 0,
                  " rightClose=", rightClose ? 1 : 0,
                  " fwdDot=", dot,
                  " rightDot=", rightDot,
                  " fovDelta=", fovDiff * (180.0f / 3.14159265f), "deg"));
              }
              cameraType = CameraType::Unknown;
            } else {
              // Consistent disagreement for many frames — accept as cut.
              m_disagreeStreak = 0;
              Logger::info("[CamMgr.hyst] accepting re-latch (cut)");
            }
          } else {
            m_disagreeStreak = 0;
          }
        }
      }
    }
    
    // Check fov consistency across frames
    if (frameId > 0) {
      if (getCamera(cameraType).isValid(frameId - 1) && !areFovsClose(decomposeProjectionParams.fov, getCamera(cameraType))) {
        ONCE(Logger::info("[RTX] CameraManager: FOV of a camera changed between frames"));
      }
    }

    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    // NV-DXVK TF2: previously this gated on `lastUpdateFrame != frameId`,
    // making the FIRST passing draw per frame win Main and locking out
    // every subsequent draw. That's the root cause of the "body races
    // ahead of camera" lag: when the first-latched draw read a stale cb2
    // (the game can submit multiple cbuffers per frame, and DX11's per-draw
    // RDEF lookup can land on one whose CBufCommonPerCamera value is older
    // than the gameplay one), Main froze on it for the whole frame even
    // though later gameplay draws had fresher data. Now Main re-latches on
    // every passing candidate within the frame; LAST-WINS semantics. The
    // strengthened isInWorld gate (|w2vT|>10) plus existing hysteresis
    // (vp/maxZ HARD reject + fwd/right basis check + position-proximity
    // below) ensure only legitimate gameplay candidates can re-latch, so
    // the last one carries the freshest pose.
    bool shouldUpdateMainCamera = cameraType == CameraType::Main;
    // NV-DXVK [EngineCam] suppression: when the engine-hook is authoritative
    // for Main, the per-draw classifier MUST NOT update Main — otherwise the
    // last per-draw to be classified Main this frame would overwrite the
    // engine-derived pose (or stomp it on subsequent CS-thread ordering with
    // the EndFrame consumer's lambda). The hook captures the same matrix the
    // engine uploads to cb2 for the main world pass, with no per-draw
    // decomposition noise. Sky / ViewModel / RenderToTexture branches
    // are unaffected: they don't classify as Main and their per-draw update
    // is still required.
    //
    // Self-healing gate: we suppress ONLY when the engine-hook has
    // actually captured at least one main-pass matrix (g_engineHookCaptureCount
    // > 0). If the trampoline never installed (master engine-patches toggle
    // off, prolog mismatch, etc.) or hasn't fired yet (first frames of
    // session, menu/loading), we leave the per-draw classifier in charge
    // so Main always has SOME source. Without this fallback, requesting
    // useEngineHookMainCamera with a non-installed trampoline gives a
    // permanent black screen (Main never updates).
    //
    // NOTE: we only skip the update path, not the rest of the per-draw
    // classifier (FoV / viewport / hysteresis checks still run — they read
    // useful state into m_mainLatchSnapshot etc. that other code consumes).
    const bool engineCamSuppressesMainUpdate =
         RtxOptions::useEngineHookMainCamera()
      && (cameraType == CameraType::Main)
      && (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 0);
    if (engineCamSuppressesMainUpdate) {
      shouldUpdateMainCamera = false;
    }
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    Matrix4 worldToView = input.getTransformData().worldToView;
    Matrix4 viewToProjection = input.getTransformData().viewToProjection;

    // (Eye-snap apparatus deleted: hardcoded killswitch was on; shadow-snap
    // probe across a real session showed the only working ref source —
    // pilot-eye atomic — actually carries the gun-pose Z (≈32) on this
    // build, not the player eye Z (≈92). If the snap had been enabled it
    // would have actively dragged Main into the ground. The proper Bug #1
    // fix + isLargeEnough lower bound (800×400) put Main on the legitimate
    // eye-height candidate without needing any post-classify rewrite.
    // Everything that lived here is gone: snap block, ramp state, streak
    // counter, kActivationFrame / kPilotEyeStreakRequired / kMaxRampStepU,
    // EyeSnapDisabled() killswitch, g_lastSnappedCam / g_haveLastSnapped /
    // g_lastRampFrame / g_pilotEyeStreak, plus the legacy g_vmEye fallback
    // path. The pilot-eye atomic itself is kept (still useful as a debug
    // signal even if unreliable as ground truth).

    // NV-DXVK [classify-trace]: per-call log of camera classification +
    // worldToView translation. Used to verify whether multiple distinct
    // worldToView values get classified as Main within the same frame
    // (proves the "alternating cameras" oscillation theory). Throttled
    // to first 400 events to capture ~5-10 seconds of gameplay.
    {
      static uint32_t sClassifyLog = 0;
      static uint32_t sLastFrameId = UINT32_MAX;
      static uint32_t sMainsThisFrame = 0;
      static Vector3 sFirstMainW2vTThisFrame{0,0,0};
      static Vector3 sLastMainW2vTThisFrame{0,0,0};
      const uint32_t frameId = m_device->getCurrentFrameId();
      // Detect frame boundary; emit a per-frame summary of the spread
      // between FIRST and LAST Main candidate's w2vT.
      if (frameId != sLastFrameId && sMainsThisFrame > 1 && sClassifyLog < 400) {
        const Vector3 spread = sLastMainW2vTThisFrame - sFirstMainW2vTThisFrame;
        const float spreadMag = std::sqrt(
          spread.x*spread.x + spread.y*spread.y + spread.z*spread.z);
        Logger::info(str::format(
          "[CamMgr.classify-spread] frame=", sLastFrameId,
          " mainCandidates=", sMainsThisFrame,
          " firstMainW2vT=(", sFirstMainW2vTThisFrame.x, ",",
                              sFirstMainW2vTThisFrame.y, ",",
                              sFirstMainW2vTThisFrame.z, ")",
          " lastMainW2vT=(", sLastMainW2vTThisFrame.x, ",",
                             sLastMainW2vTThisFrame.y, ",",
                             sLastMainW2vTThisFrame.z, ")",
          " |spread|=", spreadMag));
        ++sClassifyLog;
      }
      if (frameId != sLastFrameId) {
        sLastFrameId = frameId;
        sMainsThisFrame = 0;
      }
      const Vector3 curW2vT(worldToView[3][0], worldToView[3][1], worldToView[3][2]);
      if (cameraType == CameraType::Main) {
        if (sMainsThisFrame == 0) sFirstMainW2vTThisFrame = curW2vT;
        sLastMainW2vTThisFrame = curW2vT;
        ++sMainsThisFrame;
      }
      // Per-call classify log: type, frame, w2vT + recovered camera world
      // position from full matrix decomposition + engine ground-truth eye.
      // The recovered C tells us what player position THIS matrix encodes;
      // delta vs engineEye tells us whether this candidate is the real
      // player camera (delta < 1u) or some other pass misclassified as Main.
      if (sClassifyLog < 400) {
        const float vw = input.getTransformData().viewportWidth;
        const float vh = input.getTransformData().viewportHeight;
        const Matrix4& p = viewToProjection;
        const float Sy = p[1][1];
        const float fovDeg = (std::abs(Sy) > 1e-6f)
          ? (2.f * std::atan(1.f / Sy) * (180.f / 3.14159265f)) : 0.f;
        const char* typeName =
          cameraType == CameraType::Main ? "Main" :
          cameraType == CameraType::ViewModel ? "ViewModel" :
          cameraType == CameraType::Sky ? "Sky" :
          cameraType == CameraType::Portal0 ? "Portal0" :
          cameraType == CameraType::Portal1 ? "Portal1" :
          cameraType == CameraType::Unknown ? "Unknown" : "?";
        // Recover camera world position C from worldToView (orthonormal
        // assumption). dxvk Matrix4 stores M[col][row]; math row 0 = R,
        // row 1 = U, row 2 = F, so R.x=W[0][0], R.y=W[1][0], R.z=W[2][0];
        // U.x=W[0][1], etc. C = -R_rot^T · t where t = (W[3][0..2]).
        const Matrix4& w = worldToView;
        const float Cx = -(w[0][0]*w[3][0] + w[0][1]*w[3][1] + w[0][2]*w[3][2]);
        const float Cy = -(w[1][0]*w[3][0] + w[1][1]*w[3][1] + w[1][2]*w[3][2]);
        const float Cz = -(w[2][0]*w[3][0] + w[2][1]*w[3][1] + w[2][2]*w[3][2]);
        // Engine ground truth (lp+0x3D6C). May be null early in startup.
        const float* eye = GetEngineEyeCM();
        const bool haveEye =
          eye && std::isfinite(eye[0]) && std::isfinite(eye[1]) && std::isfinite(eye[2]);
        const float eX = haveEye ? eye[0] : 0.f;
        const float eY = haveEye ? eye[1] : 0.f;
        const float eZ = haveEye ? eye[2] : 0.f;
        const float dCx = haveEye ? (Cx - eX) : 0.f;
        const float dCy = haveEye ? (Cy - eY) : 0.f;
        const float dCz = haveEye ? (Cz - eZ) : 0.f;
        Logger::info(str::format(
          "[CamMgr.classify] frame=", frameId,
          " type=", typeName,
          " w2vT=(", curW2vT.x, ",", curW2vT.y, ",", curW2vT.z, ")",
          " recovC=(", Cx, ",", Cy, ",", Cz, ")",
          " engineEye=", haveEye ? "valid" : "null",
          " eye=(", eX, ",", eY, ",", eZ, ")",
          " delta=(", dCx, ",", dCy, ",", dCz, ")",
          " R=(", w[0][0], ",", w[1][0], ",", w[2][0], ")",
          " U=(", w[0][1], ",", w[1][1], ",", w[2][1], ")",
          " F=(", w[0][2], ",", w[1][2], ",", w[2][2], ")",
          " vp=", int(vw), "x", int(vh),
          " fov=", fovDeg, "deg"));
        ++sClassifyLog;
      }
    }

    // NV-DXVK (probe I): comprehensive per-draw camera classification log.
    // One line per UNIQUE (cameraType, viewport, Sx, Sy, vsHash, w2vT-int)
    // tuple, capped at ~120 total. The vsHash + w2vT-int additions disambiguate
    // draws that share a projection shape (e.g. gameplay-world vs. fullscreen
    // post-pass using the same 55.41° matrix) so we can identify the actual
    // gameplay VS hash for an allowlist.
    {
      struct Key {
        uint32_t type; int vw; int vh; int sxBucket; int syBucket;
        uint64_t vsHash; int tX; int tY; int tZ;
      };
      static std::vector<Key> seen;
      static uint32_t sLogCount = 0;
      const Matrix4& p = viewToProjection;
      const float Sx = p[0][0];
      const float Sy = p[1][1];
      const float vw = input.getTransformData().viewportWidth;
      const float vh = input.getTransformData().viewportHeight;
      const uint64_t vsHash =
        static_cast<uint64_t>(input.getTransformData().vertexShaderHash);
      Key k{ static_cast<uint32_t>(cameraType), int(vw), int(vh),
             int(Sx * 100.0f), int(Sy * 100.0f),
             vsHash,
             int(worldToView[3][0]), int(worldToView[3][1]), int(worldToView[3][2]) };
      bool isNew = true;
      for (const auto& s : seen) {
        if (s.type == k.type && s.vw == k.vw && s.vh == k.vh &&
            s.sxBucket == k.sxBucket && s.syBucket == k.syBucket &&
            s.vsHash == k.vsHash &&
            s.tX == k.tX && s.tY == k.tY && s.tZ == k.tZ) {
          isNew = false; break;
        }
      }
      if (isNew && sLogCount < 120) {
        seen.push_back(k);
        ++sLogCount;
        const float fovDeg = decomposeProjectionParams.fov * (180.0f / 3.14159265f);
        const float aspect = std::abs(decomposeProjectionParams.aspectRatio);
        const bool isIdentityProj =
          std::abs(p[0][0]-1.0f) < 0.01f && std::abs(p[1][1]-1.0f) < 0.01f &&
          std::abs(p[2][3]) < 0.01f && std::abs(p[3][3]-1.0f) < 0.01f;
        // Print VS hash in hex so it's trivial to paste into an allowlist.
        char vsHex[32];
        std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                      static_cast<unsigned long long>(vsHash));
        Logger::info(str::format(
          "[CamMgr.probeI] unique #", sLogCount,
          " cameraType=", static_cast<uint32_t>(k.type),
          " viewport=", k.vw, "x", k.vh,
          " Sx=", Sx, " Sy=", Sy, " aspect=", aspect,
          " fov=", fovDeg, "deg",
          " maxZ=", input.maxZ,
          " vsHash=", vsHex,
          " w2vT=(", worldToView[3][0], ",", worldToView[3][1], ",", worldToView[3][2], ")",
          " m23=", p[2][3], " m33=", p[3][3],
          " identityProj=", isIdentityProj ? 1 : 0,
          " shouldUpdateMain=", shouldUpdateMainCamera ? 1 : 0));
      }
    }

    // NV-DXVK: worldToView is LEFT AT GAME VALUES. Previously we zeroed the
    // translation to match the camera-relative TLAS frame, but that starved
    // NRC / motion vectors / denoisers of real world-space camera motion and
    // caused TDRs. The preferred fix is the other direction: shift the TLAS
    // into absolute world by adding c_cameraOrigin to every BSP per-instance
    // translation in d3d11_rtx's fanout, so camera, TLAS, NRC, and motion
    // all live in the same absolute-world coordinate system.

    if (isPlaying || isBrowsing) {
      if (shouldUpdateMainCamera) {
        RtCamera::RtCameraSetting setting;
        cameraSequence->getRecord(cameraSequence->currentFrame(), setting);
        isCameraCut = camera.updateFromSetting(frameId, setting, 0);

        if (isPlaying) {
          cameraSequence->goToNextFrame();
        }
      }
    } else if (cameraType != CameraType::Unknown && !engineCamSuppressesMainUpdate) {
      // NV-DXVK: critical guard. accessCamera() ALIASES Unknown to the Main
      // camera object (it's documented at the top of CameraManager that we
      // "never update Unknown camera directly"). Without this guard, every
      // Unknown-classified draw would call .update() on the Main camera,
      // stamping its lastUpdateFrame with the current frameId. The next
      // gameplay draw that legitimately classifies as Main then sees
      // shouldUpdateMainCamera = false (because lastUpdateFrame == frameId
      // already) and never gets to latch its real player pose. Net effect:
      // Main is permanently pinned to whatever the first Unknown draw of
      // each frame happened to carry — usually a UI/fallback transform.
      // Skipping the update for Unknown is the only correct option since
      // we can't write to a "discarded" camera slot.
      //
      // NV-DXVK [EngineCam]: also skip when the engine-hook authoritative
      // path owns Main. Sky / ViewModel / RenderToTexture still fall
      // through to update() — they're separate camera slots, not aliased
      // to Main, and they still need per-draw classification.
      isCameraCut = camera.update(
        frameId,
        worldToView,
        viewToProjection,
        decomposeProjectionParams.fov,
        decomposeProjectionParams.aspectRatio,
        decomposeProjectionParams.nearPlane,
        decomposeProjectionParams.farPlane,
        decomposeProjectionParams.isLHS
      );
    }


    if (shouldUpdateMainCamera && RtCameraSequence::mode() == RtCameraSequence::Mode::Record) {
      auto& setting = camera.getSetting();
      cameraSequence->addRecord(setting);
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;

    // NV-DXVK: log Main camera latch events with position so the TLAS-coherence
    // filter in d3d11_rtx can be correlated to camera updates frame-by-frame.
    // Also mark that this frame's Main was set by the CLASSIFIER (trusted
    // pose), not the safety net (untrusted pose). The TLAS filter gates on
    // this flag so it only rejects draws when Main's position is reliable.
    if (shouldUpdateMainCamera && cameraType == CameraType::Main) {
      noteMainSetByClassifier(frameId);
      noteFrame(frameId);
      ++g_mainHist.accepted;
      // NV-DXVK: record the snapshot used by the hysteresis gate on future
      // candidates. Must happen AFTER camera.update so getPosition/getForward
      // reflect the new latch.
      {
        MainLatchSnapshot& snap = m_mainLatchSnapshot;
        snap.fovRad      = decomposeProjectionParams.fov;
        snap.viewportW   = input.getTransformData().viewportWidth;
        snap.viewportH   = input.getTransformData().viewportHeight;
        snap.maxZ        = input.maxZ;
        // Forward from row-major worldToView col 2, right from col 0.
        const Matrix4& w = worldToView;
        snap.fwd         = Vector3(w[0][2], w[1][2], w[2][2]);
        snap.right       = Vector3(w[0][0], w[1][0], w[2][0]);
        // Normalize (input may not be perfectly unit).
        const float fwdLen2 = snap.fwd.x*snap.fwd.x + snap.fwd.y*snap.fwd.y + snap.fwd.z*snap.fwd.z;
        if (fwdLen2 > 0.001f) {
          const float invLen = 1.0f / std::sqrt(fwdLen2);
          snap.fwd = Vector3(snap.fwd.x * invLen, snap.fwd.y * invLen, snap.fwd.z * invLen);
        }
        const float rightLen2 = snap.right.x*snap.right.x + snap.right.y*snap.right.y + snap.right.z*snap.right.z;
        if (rightLen2 > 0.001f) {
          const float invLen = 1.0f / std::sqrt(rightLen2);
          snap.right = Vector3(snap.right.x * invLen, snap.right.y * invLen, snap.right.z * invLen);
        }
        snap.pos         = camera.getPosition(/*freecam=*/false);
        snap.frameId     = frameId;
        snap.valid       = true;
      }
      static uint32_t sMainLatchLog = 0;
      if (sMainLatchLog < 40) {
        ++sMainLatchLog;
        const Vector3 pos = camera.getPosition(/*freecam=*/false);
        // Print the basis rows of worldToView so we can verify orientation.
        // Expected Vulkan view convention:
        //   row0 (right)  ≈ (1,0,0) when camera faces world -Z
        //   row1 (up)     ≈ (0,1,0)
        //   row2 (back)   ≈ (0,0,1) (camera looks down -Z so back = +Z view)
        // 45° roll = row0/row1 rotated around row2 axis.
        const Matrix4& w = worldToView;
        Logger::info(str::format(
          "[CamMgr.latch] #", sMainLatchLog, " frame=", frameId,
          " pos=(", pos.x, ",", pos.y, ",", pos.z, ")",
          " fov=", decomposeProjectionParams.fov * (180.0f / 3.14159265f), "deg",
          " maxZ=", input.maxZ,
          " cameraCut=", isCameraCut ? 1 : 0,
          " right=(", w[0][0], ",", w[0][1], ",", w[0][2], ")",
          " up=(",    w[1][0], ",", w[1][1], ",", w[1][2], ")",
          " fwd=(",   w[2][0], ",", w[2][1], ",", w[2][2], ")",
          " VP_m23=", viewToProjection[2][3],   // -1 = RH proj, +1 = LH proj
          " VP_diag=(", viewToProjection[0][0], ",", viewToProjection[1][1], ",", viewToProjection[2][2], ")",
          " VP_translateZ=", viewToProjection[3][2],
          " wtvPathId=", input.getTransformData().worldToViewPathId));
      }
    }

    // [pcdTrace] One log per (frame, cameraType-class) so we see every
    // *kind* of classification result that processCameraData produces
    // each frame. Goal: during the (0,0,0)-bootstrap freeze, find out
    // which cameraType the player-position draws are getting
    // classified as. If they're flipping to Unknown / ViewModel /
    // RenderToTexture instead of Main, that's why pcdMainTrace stopped
    // and the Main camera stops advancing.
    {
      const uint32_t fid = m_device->getCurrentFrameId();
      const uint32_t typeIdx = static_cast<uint32_t>(cameraType);
      // Pack (frame, type) into single uint64 for atomic CAS dedup.
      const uint64_t key = (uint64_t(fid) << 8) | (typeIdx & 0xff);
      static std::atomic<uint64_t> sLastKey{UINT64_MAX};
      uint64_t expected = sLastKey.load(std::memory_order_relaxed);
      if (expected != key) {
        if (sLastKey.compare_exchange_strong(expected, key,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          const auto& w = input.getTransformData().worldToView;
          const float tR = float(w[3][0]), tU = float(w[3][1]), tF = float(w[3][2]);
          const float camX = -(float(w[0][0])*tR + float(w[0][1])*tU + float(w[0][2])*tF);
          const float camY = -(float(w[1][0])*tR + float(w[1][1])*tU + float(w[1][2])*tF);
          const float camZ = -(float(w[2][0])*tR + float(w[2][1])*tU + float(w[2][2])*tF);
          const char* tname = "?";
          switch (cameraType) {
            case CameraType::Main:             tname = "Main"; break;
            case CameraType::Sky:              tname = "Sky"; break;
            case CameraType::ViewModel:        tname = "ViewModel"; break;
            case CameraType::RenderToTexture:  tname = "RenderToTexture"; break;
            case CameraType::Unknown:          tname = "Unknown"; break;
            default: break;
          }
          Logger::info(str::format(
            "[pcdTrace] frame=", fid,
            " type=", tname,
            " camPos=(", camX, ",", camY, ",", camZ, ")",
            " skyCat=", input.testCategoryFlags(InstanceCategories::Sky) ? 1 : 0));
        }
      }
    }

    return cameraType;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    // NV-DXVK [TF2 inf-far clamp]: the engine hook supplies a Source/Titanfall
    // infinite-far reverse-Z projection (zFar=inf). Several RT-side consumers
    // assume a finite far: overrideNearPlane and getVolumeDefinitionCamera both
    // bail to the raw matrix on inf-far, and the ProjectionToView inverse stored
    // by RtCamera::update goes degenerate — screen-space world-position
    // reconstruction then produces garbage. Rebuild with a large finite far,
    // reusing the same DecomposeProjection→SetupByAngles path as
    // RtCamera::overrideNearPlane so projection conventions (NDC, reverse-Z,
    // handedness flags) are preserved. The far is far past the reprojected
    // skybox (~1.5e7) so nothing legitimate is clipped. processExternalCamera is
    // only called from the engine-hook consumer, so this is scoped to TF2.
    // Far chosen so the rebuilt matrix decodes back to a FINITE far: with near
    // zn=7, M[2][2] = -F/(F-zn) must stay distinguishable from -1.0 in float32
    // (the |x+1| > ~1.2e-7 ulp limit ⇒ F < ~5.9e7), while still being past the
    // reprojected 3D-skybox extent (~2.3e7). 5e7 satisfies both. 1e8 (prior
    // value) rounded M[2][2] to exactly -1.0 ⇒ decoded back to inf ⇒ no-op.
    constexpr float kEngineHookFiniteFar = 5.0e7f;
    Matrix4 v2p = viewToProjection;
    bool farClamped = false;
    if (RtxOptions::tf2ClampEngineFarPlane()) {
      uint32_t flags;
      float p[PROJ_NUM];
      DecomposeProjection(NDC_D3D, NDC_D3D, *reinterpret_cast<float4x4*>(&v2p),
                          &flags, p, nullptr, nullptr, nullptr, nullptr);
      // Clamp purely on a non-finite far. (An earlier xmin<xmax guard never
      // fired because reverse-Z decompose returns the angle pairs sign-swapped;
      // SetupByAngles needs min<max, so normalise the pairs before rebuilding.)
      const bool farInf = !std::isfinite(p[PROJ_ZFAR]);
      if (farInf && std::isfinite(p[PROJ_ZNEAR])) {
        float aMinX = p[PROJ_ANGLEMINX], aMaxX = p[PROJ_ANGLEMAXX];
        float aMinY = p[PROJ_ANGLEMINY], aMaxY = p[PROJ_ANGLEMAXY];
        if (aMinX > aMaxX) std::swap(aMinX, aMaxX);
        if (aMinY > aMaxY) std::swap(aMinY, aMaxY);
        if (std::isfinite(aMinX) && std::isfinite(aMaxX) && (aMinX < aMaxX) &&
            std::isfinite(aMinY) && std::isfinite(aMaxY) && (aMinY < aMaxY)) {
          float4x4 rebuiltProj;
          rebuiltProj.SetupByAngles(aMinX, aMaxX, aMinY, aMaxY,
                                    p[PROJ_ZNEAR], kEngineHookFiniteFar, flags);
          memcpy(&v2p, &rebuiltProj, sizeof(v2p));
          farClamped = true;
        }
      }
      // Confirmation log (throttled). Remove once the clamp is settled.
      {
        static uint32_t sN = 0;
        if (sN < 30) {
          ++sN;
          Logger::warn(str::format(
            "[TF2FarClamp] type=", (int)type, " zNear=", p[PROJ_ZNEAR],
            " oldZFar=", p[PROJ_ZFAR], " farInf=", (farInf ? 1 : 0),
            " clamped=", (farClamped ? 1 : 0), " newFar=", (farClamped ? kEngineHookFiniteFar : p[PROJ_ZFAR])));
        }
      }
    }

    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(v2p);
    // Don't trust the round-trip far decode at large magnitudes (float precision
    // near M[2][2]=-1 can re-report inf); force the known finite far we built.
    if (farClamped) {
      decomposeProjectionParams.farPlane = kEngineHookFiniteFar;
    }

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      v2p,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS);
  }

    DecomposeProjectionParams CameraManager::getOrDecomposeProjection(const Matrix4& viewToProjection) {
      XXH64_hash_t projectionHash = XXH64(&viewToProjection, sizeof(viewToProjection), 0);
      auto iter = m_decompositionCache.find(projectionHash);
      if (iter != m_decompositionCache.end()) {
        return iter->second;
      }

      DecomposeProjectionParams decomposeProjectionParams;
      decomposeProjection(viewToProjection, decomposeProjectionParams);
      m_decompositionCache.emplace(projectionHash, decomposeProjectionParams);
      return decomposeProjectionParams;
    }
}  // namespace dxvk
