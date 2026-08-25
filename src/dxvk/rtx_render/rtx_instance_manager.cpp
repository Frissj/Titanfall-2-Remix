/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstring>
// NV-DXVK [Perf.Report]: [Perf.SceneObj]'s four estMsPerFrame values feed the
// dxvk-cs half of the assembled breakdown.
#include "rtx_perf_report.h"
#include <chrono>
#include <assert.h>

#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_draw_call_cache.h"
#include "rtx_camera_manager.h"
#include "rtx_options.h"
// NV-DXVK [VsColor][perf]: for DebugView::debugViewIdx() and
// DEBUG_VIEW_VERTEX_SHADER_ID, used by vsDebugIdIsConsumed() below.
#include "rtx_debug_view.h"
#include "rtx_materials.h"
#include "rtx_terrain_baker.h"

#include "rtx_cb_types.h"
#include "rtx_matrix_helpers.h"
#include "dxvk_scoped_annotation.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/concept/surface_material/surface_material_hitgroup.h"
#include "rtx/pass/instance_definitions.h"

namespace dxvk {

  // NV-DXVK [Phase2b]: the sharded-instance-phase context — see the declaration
  // comment in rtx_instance_manager.h and PHASE2B_IMPLEMENTATION_SPEC.md.
  thread_local ShardedInstancePhase t_shardPhase;

  // Forward decl for the engine-hook main-cam capture counter (defined in
  // rtx_camera_manager.cpp at dxvk::tf2 scope). Used as a gameplay-active
  // gate for the PropCensus probe so thread-local accumulators don't fill
  // with menu-phase noise.
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;

    // NV-DXVK [JobProbe]: written by the world-visibility-worker hook in
    // d3d11/tf2_decal_hook.cpp, drained once per frame below next to
    // [PitchProbe]. Defined in rtx_camera_manager.cpp.
    extern std::atomic<uint64_t> g_jobProbeCalls;
    extern std::atomic<uint64_t> g_jobProbeRecCntSum;
    extern std::atomic<uint64_t> g_jobProbeBadReads;
    extern std::atomic<uint32_t> g_jobProbeJobIdxMin;
    extern std::atomic<uint32_t> g_jobProbeJobIdxMax;
    extern std::atomic<uint32_t> g_jobProbeLeafSkipLo;
    extern std::atomic<uint32_t> g_jobProbeLeafSkipHi;
    extern std::atomic<uint32_t> g_jobProbeSplitLo;
    extern std::atomic<uint32_t> g_jobProbeSplitHi;
    extern std::atomic<uint32_t> g_jobProbePlanesLo;
    extern std::atomic<uint32_t> g_jobProbePlanesHi;
    extern std::atomic<uint32_t> g_jobProbePoolLo;
    extern std::atomic<uint32_t> g_jobProbePoolHi;
    extern std::atomic<uint64_t> g_dispProbeA20Calls;
    extern std::atomic<uint64_t> g_dispProbeAllocCalls;
    extern std::atomic<uint64_t> g_dispProbeAllocFail;
    extern std::atomic<uint32_t> g_dispProbePendLo;
    extern std::atomic<uint32_t> g_dispProbePendHi;
    extern std::atomic<uint64_t> g_ed900ProbeCalls;
    extern std::atomic<uint32_t> g_ed900ProbeInstalled;
    extern std::atomic<uint64_t> g_eb620ProbeCalls;
    extern std::atomic<uint64_t> g_ed900DropCount;
    extern std::atomic<uint32_t> g_ed900DropInstalled;
    extern std::atomic<uint64_t> g_clipDegenA;
    extern std::atomic<uint64_t> g_clipDegenB;
    extern std::atomic<uint32_t> g_clipDegenInstalled;
    extern std::atomic<uint32_t> g_clipDegenAreaN;
    extern std::atomic<uint32_t> g_clipDegenAreas[16];
    extern std::atomic<uint32_t> g_clipDegenMaxBit;
    extern std::atomic<uint32_t> g_degenPair[64];
    extern std::atomic<uint32_t> g_degenPairInstalled;
    // NV-DXVK [QueueProbe]. Defined in rtx_camera_manager.cpp and filled by
    // d3d11_rtx.cpp draining the islands — libdxvk.a cannot call into d3d11,
    // so these atomics are the only channel. g_qStart is an INDEX (assigned,
    // not accumulated); see the definition for why that distinction matters.
    extern std::atomic<uint32_t> g_qStart;
    extern std::atomic<uint64_t> g_qStartCalls;
    extern std::atomic<uint64_t> g_qSkipCalls;
    extern std::atomic<uint32_t> g_qSkipAreaN;
    extern std::atomic<uint32_t> g_qSkipAreas[32];
    extern std::atomic<uint32_t> g_queueProbeInstalled;
    // NV-DXVK [FaceReject]: portals skipped at client.dll+0x2EB98F because the
    // camera is behind the portal plane, keyed by target area.
    extern std::atomic<uint64_t> g_faceRejectCount;
    extern std::atomic<uint32_t> g_faceRejectAreaN;
    extern std::atomic<uint32_t> g_faceRejectAreas[32];
    extern std::atomic<uint32_t> g_faceRejectInstalled;
    // NV-DXVK [PortalWalk]: client.dll+0x2EB93F, every portal iterated.
    extern std::atomic<uint64_t> g_portalWalkCount;
    extern std::atomic<uint32_t> g_portalWalkAreaN;
    extern std::atomic<uint32_t> g_portalWalkAreas[48];
    extern std::atomic<uint32_t> g_portalWalkInstalled;
    extern std::atomic<uint32_t> g_portalWalkOob;
    // NV-DXVK [SelWrite]: the selector store at client.dll+0x2EC739.
    extern std::atomic<uint64_t> g_selWriteCount;
    extern std::atomic<uint32_t> g_selWriteAreaN;
    extern std::atomic<uint32_t> g_selWriteAreas[48];
    extern std::atomic<uint32_t> g_selWriteInstalled;
    extern std::atomic<uint32_t> g_rewinds;
    // NV-DXVK [DropAreas]: the identities behind the flat ed900Drop count.
    extern std::atomic<uint32_t> g_dropAreaN;
    extern std::atomic<uint32_t> g_dropAreas[48];
    extern std::atomic<uint32_t> g_cullOffAbMode;
    extern std::atomic<float> g_dispProbeFc000X;
    extern std::atomic<float> g_dispProbeFc000Y;
    extern std::atomic<float> g_dispProbeFc000Z;
    extern std::atomic<float> g_dispProbeFc000W;
    extern std::atomic<uint32_t> g_dispProbeSlotN;
    extern std::atomic<uint32_t> g_dispProbeSlotA1[32];
    extern std::atomic<uint32_t> g_dispProbeSlotA2[32];
    extern std::atomic<uint32_t> g_dispProbeSlotA3[32];
    extern std::atomic<uint32_t> g_dispProbeSlotRA[32];
    extern std::atomic<float>    g_areaSeedOrgX;
    extern std::atomic<float>    g_areaSeedOrgY;
    extern std::atomic<float>    g_areaSeedOrgZ;
    extern std::atomic<float>    g_areaSeedOrgW;
    extern std::atomic<uint32_t> g_areaSeedNAreas;
    extern std::atomic<uint32_t> g_areaSeedListLen;
    extern std::atomic<uint32_t> g_areaSeedN;
    extern std::atomic<uint32_t> g_areaSeedAreas[48];
    extern std::atomic<uint64_t> g_areaSeedCalls;
    extern std::atomic<uint32_t> g_areaSeedInstalled;
    extern std::atomic<uint32_t> g_areaSeedLive;
    extern std::atomic<uint32_t> g_areaSeedLiveN;
    extern std::atomic<uint32_t> g_areaSeedLiveAreas[48];
    extern std::atomic<uint32_t> g_areaSeedPending;
    extern std::atomic<uint64_t> g_dispProbeEd480Calls;
    extern std::atomic<uint32_t> g_dispProbeEd480N;
    extern std::atomic<uint32_t> g_dispProbeEd480Area[8];

    // NV-DXVK [DrainProbe]: the worker's output, written by the drain hook.
    extern std::atomic<uint64_t> g_drainProbeCalls;
    extern std::atomic<uint64_t> g_drainProbeBad;
    extern std::atomic<uint64_t> g_drainProbeM1Sum;
    extern std::atomic<uint64_t> g_drainProbeM2Sum;
    extern std::atomic<uint64_t> g_drainProbeRSum;
    extern std::atomic<uint32_t> g_drainProbeM1Max;
    extern std::atomic<uint32_t> g_drainProbeM2Max;
    extern std::atomic<uint32_t> g_drainProbeC70;
    extern std::atomic<uint32_t> g_drainProbeC74;
    extern std::atomic<uint32_t> g_drainProbeC78;
    extern std::atomic<uint32_t> g_drainProbeLayoutOk;
  }

  // NV-DXVK [VsColor]: session-stable vertex-shader -> small-id table backing
  // RtSurface::vsDebugId and DEBUG_VIEW_VERTEX_SHADER_ID.
  //
  // Why session-stable rather than per-frame: the whole point is to watch one
  // object over time, so its colour must not move between frames. A per-frame
  // renumbering would repaint the scene every frame and be exactly as useless
  // as surfaceIndex is for cross-frame attribution.
  //
  // COST -- the note here used to read "one hash lookup per draw ... being
  // complete is already cheap". CORRECTED 2026-08-06: it is not per draw, it is
  // PER INSTANCE. The only caller is updateInstance, which [Perf.UpdInst]
  // measures at instPerFrame=16,209 against ~1,110 draws -- 14.6x the assumed
  // rate. And it is not a bare hash lookup: it takes a std::mutex and does a map
  // insert-or-find, every instance, every frame, forever.
  //
  // (Third time this exact error has appeared: rtx.perfSceneObjSplit was budgeted
  // per draw and billed per instance at 11x, and rtx.perfNonOpaqueCensus said
  // "once per second" while accumulating per instance. Anything reached from
  // updateInstance or processSceneObject is per INSTANCE. Count it that way.)
  //
  // Ids start at 1; 0 is reserved for "unassigned" and paints black -- which is
  // what vsDebugIdIsConsumed() below relies on to skip the work entirely.
  static uint16_t acquireVsDebugId(XXH64_hash_t vsHash) {
    if (vsHash == 0) {
      return 0;
    }

    static std::mutex sVsIdMu;
    static std::unordered_map<XXH64_hash_t, uint16_t> sVsIds;
    static uint16_t sNextVsId = 1;

    uint16_t assigned = 0;
    bool isNew = false;
    {
      std::lock_guard<std::mutex> g(sVsIdMu);
      // try_emplace, not emplace: emplace is specified to construct the
      // value_type before it can know whether the key is already present, so on
      // the steady-state path -- which is every call after the first few frames,
      // since this table stops growing at ~31 entries -- it can build and then
      // discard a node. try_emplace constructs nothing on a hit.
      auto [it, inserted] = sVsIds.try_emplace(vsHash, sNextVsId);
      if (inserted) {
        // The colour lattice holds 124 usable codes (see vsDebugIdToColor).
        // Saturate rather than wrap: a run that somehow exceeds it degrades
        // into "everything past 124 shares id 124" instead of silently
        // aliasing onto an already-legended colour, which would make the
        // legend quietly wrong rather than obviously incomplete.
        if (sNextVsId < 124) {
          ++sNextVsId;
        }
        isNew = true;
      }
      assigned = it->second;
    }

    if (isNew) {
      // Reproduce vsDebugIdToColor() from geometry_resolver.slangh exactly.
      // The 5-level lattice quantises to these values on both sides, so the
      // logged RGB is literally what the shader paints -- and, unlike the hue
      // sweep this replaced, a channel misread by up to ~30 still decodes to
      // the right id.
      static constexpr uint32_t kLevels[5] = { 0u, 64u, 128u, 191u, 255u };
      const uint32_t r = kLevels[assigned % 5];
      const uint32_t g = kLevels[(assigned / 5) % 5];
      const uint32_t b = kLevels[(assigned / 25) % 5];

      Logger::info(str::format(
        "[VsColor] id=", assigned,
        " vs=0x", std::hex, static_cast<uint64_t>(vsHash), std::dec,
        " rgb=(", r, ",", g, ",", b, ")",
        " — DEBUG_VIEW_VERTEX_SHADER_ID (861) paints this VS this colour;"
        " decode a screenshot by rounding each channel to {0,64,128,191,255}"
        " -> levels, then id = lr + 5*lg + 25*lb"));
    }

    return assigned;
  }

  // NV-DXVK [VsColor][perf]: is anything actually going to READ vsDebugId?
  //
  // It is a pure diagnostic value and it has exactly two consumers, both
  // switchable and both off in normal play:
  //   1. geometry_resolver.slangh, the DEBUG_VIEW_VERTEX_SHADER_ID case
  //      (rtx.debugView.debugViewIdx == 861), which paints the VS id as a colour.
  //   2. identExpectedOf() in rtx_context.cpp plus geometry_resolver.slangh's
  //      COVERAGE_OBS_IDENT_REGION packing, both reached only from
  //      dispatchTlasProbe, which early-returns unless rtx.logResolveCensus.
  // Nothing functional reads it. It rides in flags0 bits 5..15, and both the
  // C++ and slang sides already define 0 as "unassigned" (the shader paints
  // black for vsId == 0), so leaving it 0 is a defined state, not corruption.
  //
  // WHY THIS GATE EXISTS: acquireVsDebugId takes a mutex and does a map lookup,
  // and updateInstance calls it once per INSTANCE -- 16,209 times per frame by
  // [Perf.UpdInst]. That is a measurable slice of the `surf` stage spent
  // producing a number that, with both switches off, no code path ever reads.
  //
  // WHEN A SWITCH IS FLIPPED ON, ids repopulate on the next update rather than
  // instantly; [Perf.UpdInst] reports first=99%, so that is one frame.
  //
  // IF YOU ADD A THIRD CONSUMER OF vsDebugId, ADD IT HERE IN THE SAME COMMIT --
  // otherwise it will silently read 0 and you will be debugging the wrong thing.
  static bool vsDebugIdIsConsumed() {
    return DebugView::debugViewIdx() == DEBUG_VIEW_VERTEX_SHADER_ID
        || RtxOptions::logResolveCensus();
  }

  // NV-DXVK [DropTrace]: per-frame dropship submit counter, defined at
  // namespace dxvk scope in rtx_scene_manager.cpp. Must be declared at
  // namespace scope (NOT block scope) so it resolves to dxvk::g_dropTrace*
  // rather than the global ::g_dropTrace* (which fails to link).
  extern std::atomic<uint32_t> g_dropTraceFrame;
  extern std::atomic<uint32_t> g_dropTraceSubmits;
  // NV-DXVK [DropTrace] RAW: dropship draws entering D3D11Rtx::SubmitDraw (set
  // in d3d11_rtx.cpp); compared against g_dropTraceSubmits (draws that survived
  // to submitDrawState) to split engine-cull vs Remix-SubmitDraw-drop.
  extern std::atomic<uint32_t> g_dropTraceRawFrame;
  extern std::atomic<uint32_t> g_dropTraceRawSubmits;

  // NV-DXVK [VanishEdge]: Main-camera pose stashed by [CullCmp] (rtx_camera_manager.cpp),
  // read by the ship-vanish edge detector below.
  extern float g_veCamPx, g_veCamPy, g_veCamPz, g_veCamDx, g_veCamDy, g_veCamDz, g_veCamFov;
  extern std::atomic<uint32_t> g_veCamFrame;

  static bool isMirrorTransform(const Matrix4& m) {
    // Note: Identify if the winding is inverted by checking if the z axis is ever flipped relative to what it's expected to be for clockwise vertices in a lefthanded space
    // (x cross y) through the series of transformations
    Vector3 x(m[0].data), y(m[1].data), z(m[2].data);
    return dot(cross(x, y), z) < 0;
  }

  static uint32_t determineInstanceFlags(const DrawCallState& drawCall, const RtSurface& surface) {
    // [SpawnGeomDiag] TF2 floor backface fix:
    // Empirically, on this dxvk-remix-DX11 branch, BLAS source vertex
    // ordering for D3D11 draws is opposite Vulkan ray tracing's default
    // front-face interpretation. With CULL_BACK active, every BSP world
    // surface (floor, walls, props) was being culled by primary rays.
    //
    // Forcing TRIANGLE_FACING_CULL_DISABLE_BIT_KHR on merged-bucket TLAS
    // instances made geometry appear (both sides hit). Forcing
    // TRIANGLE_FLIP_FACING_BIT_KHR (so CCW = front) made geometry render
    // correctly in CULL_BACK. Both confirm BLAS winding interpretation needs
    // inversion vs the OLD `drawClockwise == worldToProjectionMirrored` rule.
    //
    // Old rule's failure mode: TF2 main camera's projection determinant was
    // positive (mirror=0). drawClockwise=1, mirror=0 -> 1==0 false -> no
    // FLIP. So under the main camera, BSP draws were never flipped. Floor
    // invisible. The condition must be inverted: FLIP when drawClockwise
    // disagrees with the (post-instance-transform) winding parity.
    //
    // Switching the mirror basis to objectToWorld didn't help in isolation
    // (BSP o2w is identity). What works is the inverse comparison: FLIP when
    // `drawClockwise != objectToWorldMirrored`. For TF2 BSP main camera
    // (drawClockwise=1, o2wMirror=0): 1 != 0 -> FLIP. Floor renders.
    // For mirrored instances (drawClockwise=1, o2wMirror=1): 1 != 1 -> no
    // FLIP, which is correct because the negative-scale o2w has already
    // reversed the apparent winding.
    //
    // NV-DXVK [fanout split]: read the winding basis off the SURFACE, not the
    // draw call. updateInstance has already written surface.objectToWorld from
    // the draw's transform by the time this runs, so for every ordinary draw the
    // two are the same matrix and this is a no-op. For a split fanout placement
    // they are NOT: the draw carries the batch's identity matrix while the
    // surface carries drawO2W * instancesToObject[i].
    //
    // DO NOT read this as "the split changes backface culling". It cannot: the
    // parity computed here enters the flag as (drawClockwise != basis), and
    // addBlas / addPointInstancerBlas then XOR FLIP_FACING again for a mirrored
    // instance, so the parity cancels and the rendered flag is drawClockwise
    // (^ the projection parity under tf2StableBackfaceCull) either way. The only
    // consumer that sees it uncancelled is RtInstance::isFrontFaceFlipped, which
    // feeds the USD game capturer. This reads the surface so that value describes
    // the prop rather than the batch; the [FanoutSplit] mir= counter reports how
    // many placements it differs for (mir=0 => literally identical behaviour).
    const Matrix4& objectToWorld = surface.objectToWorld;
    const bool objectToWorldMirrored = isMirrorTransform(objectToWorld);

    // Note: Vulkan ray tracing defaults to defining the front face based on clockwise vertex order when viewed from a left-handed coordinate system. The front face
    // should therefore be flipped if a counterclockwise ordering is used in this normal case, or the inverse logic if the BLAS instance transform inverts winding.
    // See: https://www.khronos.org/registry/vulkan/specs/1.1-khr-extensions/html/chap33.html#ray-traversal-culling-face
    const bool drawClockwise = drawCall.getGeometryData().frontFace == VkFrontFace::VK_FRONT_FACE_CLOCKWISE;

    uint32_t flags = 0;

    // NV-DXVK [tf2StableBackfaceCull]: by default the winding basis is the
    // objectToWorld mirror parity alone. That is unstable for sub-view
    // billboard/skinned cards whose o2w determinant sign flips as they reorient
    // (toggling which face is culled → a dark back face leaks in). When the
    // option is on, use the NET object->projection winding parity instead,
    // which is what the raster game culls on and is stable. det(o2w*w2p) sign
    // == det(o2w) XOR det(w2p), so this is identical to the current rule when
    // the projection is non-mirrored (w2pMirror=0) — all normal main-camera
    // geometry is unchanged; only mirrored/sub-view projections differ.
    bool windingMirrorBasis = objectToWorldMirrored;
    if (RtxOptions::tf2StableBackfaceCull()) {
      const Matrix4 worldToProjection =
        drawCall.getTransformData().viewToProjection * drawCall.getTransformData().worldToView;
      windingMirrorBasis = objectToWorldMirrored != isMirrorTransform(worldToProjection);
    }

    // Note: Flip front face by setting the front face to counterclockwise, which is the opposite of Vulkan ray tracing's clockwise default.
    if (drawClockwise != windingMirrorBasis)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;

    if (!RtxOptions::enableCulling())
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    // [SpawnGeomDiag] Capture the inputs determineInstanceFlags decided on, per-draw,
    // to diagnose the TF2 BSP backface-cull issue. Logs both the OLD basis
    // (worldToProjection mirror) and the NEW basis (objectToWorld mirror) so
    // we can audit how often they disagree across all submissions and confirm
    // no unexpected regressions for non-BSP draws.
    // Throttled per (vsHash,matHash,frontFace,cullMode,o2wMirror,w2pMirror).
    {
      const Matrix4 worldToProjection = drawCall.getTransformData().viewToProjection * drawCall.getTransformData().worldToView;
      const bool worldToProjectionMirrored = isMirrorTransform(worldToProjection);
      const uint32_t vsHash = uint32_t(uint64_t(drawCall.getTransformData().vertexShaderHash) & 0xffffffffu);
      const uint32_t matHash = uint32_t(uint64_t(drawCall.getMaterialData().getHash()) & 0xffffffffu);
      const uint32_t cullMode = uint32_t(drawCall.getGeometryData().cullMode);
      const uint64_t key =
        (uint64_t(vsHash) << 32) ^ uint64_t(matHash) ^
        (uint64_t(drawClockwise ? 1 : 0) << 1) ^
        (uint64_t(objectToWorldMirrored ? 1 : 0) << 2) ^
        (uint64_t(worldToProjectionMirrored ? 1 : 0) << 3) ^
        (uint64_t(cullMode) << 8);
      // NV-DXVK [Phase2b] 2026-08-18: thread_local, and THIS ONE FROZE THE GAME.
      //
      // It was a plain function-local `static std::unordered_set`. determineInstanceFlags
      // runs per draw inside updateInstance, which Phase 2b moved onto ~29 shard
      // workers, so ~1050 concurrent insert() calls per frame hit one unsynchronized
      // MSVC hash table. A concurrent insert corrupts the bucket vector's iterator
      // pairs; a later probe then walks a broken chain and NEVER TERMINATES. Symptom:
      // one worker pinned at 98% of a core ([ThreadCensus] d3d11-geometry(16)), its
      // bundle never returns, and the game thread spins forever in the shard join —
      // caught exactly by [BatchJoin] STUCK phase=shard2b joiningFuture=0/29
      // itemsDoneDelta=0. Symbolized to `s_loggedFlagDecide` in x64dbg.
      //
      // WHY thread_local AND NOT A MUTEX: the insert runs on EVERY draw, not just on
      // the first sighting of a key, so a mutex here would be ~1050 acquisitions per
      // frame across 29 workers on the hot path — the plan's Sec 6 rule is that the
      // escape lock never touches a path taken by every draw. thread_local costs
      // nothing, cannot be corrupted by construction, and the only thing it changes
      // is fidelity: "log once per process" becomes "log once per worker thread", so
      // a given key can now produce up to N lines instead of one. For a
      // first-sighting diagnostic that is the right trade. Note the insert happens
      // BEFORE Logger::warn, so the rtx.logDenyTags filter does NOT protect you from
      // this — a silenced tag still ran the racing insert every frame.
      //
      // Nine sibling diagnostics on this same worker path got the identical
      // treatment (sCollideSeen, sFnLog, sFhSeen, sTf2CcSeen, sUiSeenWarn, sMvSeen,
      // sLastSeen, `seen`, s_cloudRouteSeen). acquireVsDebugId's sVsIds is NOT one of
      // them: it hands out a stable id consumed by the GPU surface, so per-thread
      // tables would hand the same shader different ids — it keeps its real mutex.
      static thread_local std::unordered_set<uint64_t> s_loggedFlagDecide;
      if (s_loggedFlagDecide.insert(key).second && s_loggedFlagDecide.size() <= 256) {
        Logger::warn(str::format("[SpawnGeomDiag.FlagDecide] vsHash=0x", std::hex, vsHash,
          " matHash=0x", matHash,
          " frontFace=", std::dec, uint32_t(drawCall.getGeometryData().frontFace),
          " (cw=", drawClockwise ? 1 : 0, ")",
          " o2wMirror=", objectToWorldMirrored ? 1 : 0,
          " w2pMirror=", worldToProjectionMirrored ? 1 : 0,
          " cullMode=", cullMode,
          " blend=", drawCall.getMaterialData().blendMode.enableBlending ? 1 : 0,
          " decal=", surface.alphaState.isDecal ? 1 : 0,
          " forceCullBit=", drawCall.getGeometryData().forceCullBit ? 1 : 0,
          " enableCulling=", RtxOptions::enableCulling() ? 1 : 0,
          " ignAC=", drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0,
          " catFlagsRaw=0x", std::hex, drawCall.getCategoryFlags().raw(), std::dec,
          " preFlags=0x", std::hex, flags, std::dec));
      }
    }

    // This check can be overridden by replacement assets.
    if (drawCall.getMaterialData().blendMode.enableBlending && !surface.alphaState.isDecal && !drawCall.getGeometryData().forceCullBit)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    // Disable culling for baked terrain instances when the option is enabled
    // Terrain with back face culling enabled may flicker in some circumstances.
    // Forcing the geometry to be double-sided fixes the flicker, but may be undesireable in some games.
    if (TerrainBaker::disableBackFaceCulling() && drawCall.testCategoryFlags(InstanceCategories::Terrain)) {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    // NV-DXVK [sub-view skybox cull]: 3D-skybox geometry reprojected by
    // SetSkyCategoryFromCb2 is tagged IgnoreAntiCulling. It passes through a
    // 90-degree per-instance rotation + scale-1000 reproject, but the
    // FLIP_FACING decision above is computed from objectToWorld alone and
    // does not model the per-instance transform or the reproject — so opaque
    // (blend=0, cullMode=BACK) skybox mountains end up with inverted winding
    // and get fully backface-culled (invisible). Force them double-sided: a
    // skybox backdrop being double-sided is standard and harmless, and it
    // sidesteps the winding ambiguity entirely.
    if (drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling)) {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    // NV-DXVK [flipSubViewSkyboxNormals]: the sub-view reproject submits 3D-skybox
    // geometry with INVERTED WINDING. These draws have no vertex-normal buffer
    // (confirmed via [FlipNormalDiag] hasNormalBuffer=0), so the shading normal is
    // the GEOMETRIC/face normal derived from winding — which therefore points the
    // wrong way (N·L<0 -> black despite unshadowed sun + valid albedo). Toggle the
    // FLIP_FACING bit to correct the winding so the geometric normal faces the sun.
    // (Culling is already disabled for this geometry, so this only affects the
    // shading-side front/back determination, not visibility.)
    if (RtxOptions::flipSubViewSkyboxNormals() &&
        (drawCall.getTransformData().isSubView || drawCall.getTransformData().isSubViewSkybox)) {
      flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
    }

    switch (drawCall.getGeometryData().cullMode) {
    case VkCullModeFlagBits::VK_CULL_MODE_NONE:
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_FRONT_BIT:
      // Note: Invert front face flag once more if front face culling is desired to make the current front face the backface (as we simply assume that any culling
      // desired will be backface via gl_RayFlagsCullBackFacingTrianglesEXT which helps simplify GPU-side logic).
      flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT:
      // Default in shader (gl_RayFlagsCullBackFacingTrianglesEXT)
      break;
    case VkCullModeFlagBits::VK_CULL_MODE_FRONT_AND_BACK:
      assert(0); // this should already be filtered out up stack
      break;
    }

    return flags;
  }

  RtInstance::RtInstance(const uint64_t id, uint32_t instanceVectorId)
    : m_id(id)
    , m_instanceVectorId(instanceVectorId)
    , m_surfaceIndex(SURFACE_INDEX_INVALID)
    , m_previousSurfaceIndex(SURFACE_INDEX_INVALID) { }

  // Makes a copy of an instance
  RtInstance::RtInstance(const RtInstance& src, uint64_t id, uint32_t instanceVectorId)
    : surface(src.surface)
    , m_id(id)
    , m_instanceVectorId(instanceVectorId)
    , m_seenCameraTypes(src.m_seenCameraTypes)
    , m_materialType(src.m_materialType)
    , m_albedoOpacityTextureIndex(src.m_albedoOpacityTextureIndex)
    , m_samplerIndex(src.m_samplerIndex)
    , m_secondaryOpacityTextureIndex(src.m_secondaryOpacityTextureIndex)
    , m_secondarySamplerIndex(src.m_secondarySamplerIndex)
    , m_isAnimated(src.m_isAnimated)
    , m_opacityMicromapInstanceData(src.m_opacityMicromapInstanceData)
    , m_surfaceIndex(src.m_surfaceIndex)
    , m_previousSurfaceIndex(src.m_previousSurfaceIndex)
    // NV-DXVK: must travel WITH m_previousSurfaceIndex — the two are one
    // value (base + length of the surface-slot range owned last frame). A copy
    // that inherited the base but reset the length to its default would map
    // only the first slot of a PointInstancer range for a frame, which is the
    // exact bug the range mapping in AccelManager was added to fix.
    , m_previousSurfaceCount(src.m_previousSurfaceCount)
    , m_isHidden(src.m_isHidden)
    , m_isPlayerModel(src.m_isPlayerModel)
    , m_isWorldSpaceUI(src.m_isWorldSpaceUI)
    , m_isUnordered(src.m_isUnordered)
    , m_isObjectToWorldMirrored(src.m_isObjectToWorldMirrored)
    , m_linkedBlas(src.m_linkedBlas)
    , m_materialHash(src.m_materialHash)
    , m_materialDataHash(src.m_materialDataHash)
    , m_texcoordHash(src.m_texcoordHash)
    , m_indexHash(src.m_indexHash)
    , m_vkInstance(src.m_vkInstance)
    , m_geometryFlags(src.m_geometryFlags)
    , m_firstBillboard(src.m_firstBillboard)
    , m_billboardCount(src.m_billboardCount)
    , m_categoryFlags(src.m_categoryFlags)
    // NV-DXVK: was missing. m_stablePropId is the SpatialMap dedup key (see its
    // declaration). Left at 0 the clone falls back to XXH64(matrix), which is
    // exactly the drift-prone path the stable-prop ID was added to replace, and
    // clones are sub-view-reprojected content - the case it was added FOR.
    , m_stablePropId(src.m_stablePropId)
    // NV-DXVK: was missing. This mirrors the FLIP_FACING bit of m_vkInstance.flags
    // (set from it at updateInstance), and m_vkInstance IS copied - so leaving
    // this false made the clone internally inconsistent with its own flags.
    // Currently only read by diagnostics and the game capturer, so copying it is
    // low risk and removes the contradiction.
    , isFrontFaceFlipped(src.isFrontFaceFlipped)
    // NV-DXVK [FirstBakeHold]: clones inherit the hold. m_linkedBlas is copied
    // above, so the clone renders from the same (possibly still source-pending)
    // entry as its source — without the stash it would show the collapsed
    // first bake for a frame, the exact artifact the hold exists to prevent.
    // Rc copy just addrefs; the stamp site releases it per-instance once the
    // bake lands.
    , m_prevBlasKeepAlive(src.m_prevBlasKeepAlive) {
    // NV-DXVK [2026-07-26]: m_isSubsurface is skipped DELIBERATELY, and not
    // because it is harmless. It is a genuine divergence: createInstanceCopy
    // never calls updateInstance, so a clone of a subsurface instance reports
    // isSubsurface()==false and is routed as non-subsurface.
    //
    // Copying it was tried and FROZE THE GAME (hard GPU hang during load).
    // Reason: isSubsurface() does not merely report. It
    //   - gates BLAS bucket compatibility          (rtx_accel_manager.cpp:209)
    //   - ALSO pushes the instance into m_mergedInstances[Tlas::SSS]   (:2184)
    //   - reserves an EXTRA PointInstancerBatch with tlasType=Tlas::SSS,
    //     bumping m_pointInstancerSlotsPerType[Tlas::SSS]              (:6193)
    // That last one shifts every per-type byte offset that
    // dispatchPointInstancerCulling writes into m_vkInstanceBuffer, so turning
    // it on for clones corrupts AS instance data unless the SSS slot accounting
    // and TLAS build are verified to cover them - cf. the null-AS ->
    // VK_ERROR_DEVICE_LOST warning in rtx_context.cpp:2431.
    //
    // So this is a KNOWN latent bug, not an oversight. Fixing it properly means
    // auditing the SSS region sizing in dispatchPointInstancerCulling FIRST and
    // confirming Tlas::SSS is actually built when clones land in it. Do not
    // simply add it to the init list above.
    //
    // Members for which state carry over is intentionally skipped
    /*
       m_isSubsurface  (see the note above - deliberate, and load-bearing)
       m_isMarkedForGC
       m_isUnlinkedForGC
       m_isInsideFrustum
       m_frameLastUpdated
       m_frameCreated
       m_isCreatedByRenderer
       m_spatialCacheHash
       m_spatialOpPendingFrame  (one value with m_spatialCacheHash - see the size note)
       m_batchRecordKey    (record back-pointer; a clone is in no batch)
       m_residentKey       (record back-pointer; a clone is in no resident record.
                            NOT optional and not merely a "first use runs the full
                            path" convention - see the 952 -> 960 size note for why
                            an inherited key invalidates a record the clone was
                            never in, and why that costs an UNBOUNDED skip window
                            here where the batch equivalent costs one frame.)
       m_primInstanceOwner
       buildGeometries
       buildRanges
       billboardIndices
       indexOffsets
     */
  }
  // Ensure the copy ctor copies all needed members when size changes, and update the object size check.
  // Note: The object has a different size on Debug builds. 
  //       Checking the non-Debug flavors is good enough for the sake of convenience of tracking just a single size.
 #if defined(DEBUG_OPTIMIZED) || defined(NDEBUG)
  namespace {
    template<int RtInstanceSize> struct CheckRtInstanceSize {
      // The second line of the build error should contain the new size of RtInstance in the template argument, i.e. `dxvk::CheckRtInstanceSize<newSize>`
      // 768 -> 792 on 2026-07-26, the first time a non-Debug build compiled this.
      // Audited all members against the copy ctor and the skip list below it:
      // m_isSubsurface, m_stablePropId and isFrontFaceFlipped were in NEITHER.
      // m_stablePropId and isFrontFaceFlipped are now copied. m_isSubsurface is
      // NOT - copying it froze the game; see the long note in the copy ctor.
      //
      // 792 -> 800 on 2026-07-29: RtSurface gained uint16_t vsDebugId ([VsColor],
      // DEBUG_VIEW_VERTEX_SHADER_ID). No copy-ctor change was needed - the ctor
      // takes the whole surface by value via `surface(src.surface)`, so every
      // RtSurface member including this one is already carried to clones. The
      // GPU-side surface is NOT affected: vsDebugId is OR'd into the flags0 word
      // writeGPUData already emitted, so no extra bytes are uploaded per surface.
      //
      // 800 -> 808 on 2026-07-31: added uint32_t m_previousSurfaceCount, the
      // length of the surface-slot range this instance owned last frame. Only
      // 4 bytes of payload; the other 4 are alignment padding for the members
      // that follow it.
      // COPY CTOR: DONE - it is copied, deliberately and necessarily. It forms
      // a single logical value with m_previousSurfaceIndex (base + length of
      // the range), and a clone that inherited the base while defaulting the
      // length to 1 would map only the first slot of a PointInstancer range,
      // which is the precise bug the range mapping in AccelManager::
      // uploadSurfaceData was added to fix. The two must always travel
      // together - if you ever touch one, touch the other.
      //
      // 808 -> 816 on 2026-08-02: added Rc<PooledBlas> m_prevBlasKeepAlive
      // ([FirstBakeHold] — hold the previous BLAS across a relink whose
      // destination bake is still source-pending; the geometry-flicker fix).
      // COPY CTOR: DONE - clones inherit the hold (see the note in the init
      // list); a clone rendering the collapsed first bake is the artifact the
      // member exists to prevent.
      //
      // 816 -> 824 on 2026-08-07: added XXH64_hash_t m_instStateKey (handoff v5
      // sec 4c — ONE digest covering every input read by both the `surf` block and
      // the event-fanout gate of updateInstance, so each can be skipped when none
      // of them moved. This briefly went to 832 while those were two separate
      // keys; merging them removed the second word AND the second gather, which
      // was the whole cost.
      // COPY CTOR: DELIBERATELY NOT COPIED, and this is the safe direction rather
      // than an oversight. The ctor takes `surface(src.surface)` by value, so a
      // clone already holds the source's surface CONTENTS; leaving the key at
      // kEmptyHash means the clone's first updateInstance cannot match and runs
      // both full paths, re-deriving those contents and its material binding from
      // its own draw. Copying the key would do the opposite — assert that the
      // clone's surface and binding are already correct for a draw it has never
      // seen — which is exactly the false hit this key exists to prevent. The
      // binding half is the dangerous one: the symptom would be a stale material
      // INDEX, i.e. the clone rendering with another surface's textures.
      // 824 -> 832 on 2026-08-08: added uint8_t m_fastDrawBits (rtx.
      // fastInstanceUpdate — the per-draw flags-stage inputs the instance state
      // key does not cover: winding, RT-target, sub-view flags, projection
      // parity; captured at every full update, compared by the fast path).
      // 1 byte of payload; the rest is alignment padding.
      // COPY CTOR: DELIBERATELY NOT COPIED, same direction and same reasoning
      // as m_instStateKey directly above. A clone left at the 0xFF sentinel can
      // never match any real bit set, so its first updateInstance always runs
      // the full path and re-derives flags/mask from its own draw. Copying the
      // bits would assert the clone's retained vkInstance flags are correct for
      // a draw it has never seen — the exact false hit the byte exists to
      // prevent. The two members must keep the same policy: the fast path
      // requires BOTH the key and the bits to match, so an uncopied key already
      // forces the slow path; the sentinel makes the invariant local.
      // 832 -> 936 on 2026-08-08: added CullAabbCache cullAabbCache
      // ([Perf.CullAabbCache] — cached world AABB for AccelManager's
      // SceneCull, keyed on the raw 48-byte vkInstance.transform bits + the
      // object-space box; a hit skips getTransform()'s transpose and the 8
      // corner transforms per instance per frame). 104 bytes: 48 transform
      // + 4x12 vectors + bool + padding.
      // COPY CTOR: DELIBERATELY NOT COPIED (default-initialized, valid=false).
      // The cache is purely derived state and its key is re-verified against
      // the live transform/box on every read, so copying it would also be
      // correct — but a clone defaulting to invalid simply recomputes on its
      // first cull, which is the same "first use runs the full path" policy
      // as m_instStateKey/m_fastDrawBits above and costs 8 corner transforms
      // once per clone.
      // 936 -> 944 on 2026-08-14: added uint64_t m_batchRecordKey
      // ([Perf.PushInst] PHASE 2 — which fanout batch record currently holds a
      // raw pointer to this instance, 0 for none; the back-pointer that makes
      // InstanceManager::invalidateFanoutRecordFor O(1) and TOTAL).
      // COPY CTOR: DELIBERATELY NOT COPIED, and here that is a correctness
      // requirement rather than the "first use runs the full path" convention
      // the three members above follow. A record's instance list is the exact,
      // ordered output of one placement loop, and createInstanceCopy does not
      // add the clone to it — so the clone is NOT a member of that batch. If it
      // inherited the key, removeInstance(clone) would invalidate a record the
      // clone was never in (silently costing the real batch its skip), and the
      // record's own back-pointer cleanup would skip the clone because it never
      // walks it. Left at 0, a clone simply owns no record until some placement
      // loop genuinely resolves to it, which is the truth.
      // 944 -> 952 on 2026-08-18: added uint32_t m_spatialOpPendingFrame
      // ([Phase2b] — the frame id of the last DEFERRED spatial-map op recorded
      // for this instance during the sharded instance phase. onTransformChanged
      // reads it to detect that an earlier op of the SAME frame is still
      // pending, in which case its own key-unchanged skip test is unsound:
      // m_spatialCacheHash is not written until the ordered tail applies the
      // chain. Cleared by applyDeferredSpatialOp. 4 bytes of payload; the other
      // 4 are alignment padding.
      // COPY CTOR: DELIBERATELY NOT COPIED, and it must stay that way for as
      // long as m_spatialCacheHash is also not copied (see the skip list). The
      // pair is one value: where this instance currently sits in the SpatialMap,
      // and whether a write to that position is still queued. A clone sits
      // NOWHERE — it starts at kEmptyHash, is in no map, and no deferred op
      // names it, because ops are recorded against the instance pointer that
      // existed when the worker ran. Inheriting the source's pending-frame would
      // tell the clone's first onTransformChanged that a write it never queued
      // is still in flight (`chained`), forcing an unconditional record for a
      // hash it does not own. Left at kInvalidFrameIndex the clone takes the
      // ordinary first-use path: its key is kEmptyHash, so the key test fires on
      // its own merits and the op resolves as a plain insert. If you ever make
      // the copy ctor carry m_spatialCacheHash, carry this with it.
      // 952 -> 960 on 2026-08-20: added uint64_t m_residentKey
      // ([ResidentScene] — which resident record currently holds a raw pointer
      // to this instance, 0 for none; the back-pointer that makes
      // ResidentScene::invalidateFor O(1) and TOTAL).
      // COPY CTOR: DELIBERATELY NOT COPIED, for the same correctness reason as
      // m_batchRecordKey above and MORE strongly. A resident record's instance
      // list is the exact output of one resolution pass, and createInstanceCopy
      // does not add the clone to it — so the clone is not a member of that
      // record. If it inherited the key, removeInstance(clone) would invalidate
      // a record the clone was never in, and the record's own back-pointer
      // cleanup would never reach the clone because it does not walk it.
      // WHERE THIS DIFFERS FROM m_batchRecordKey, AND WHY IT MATTERS MORE: a
      // fanout record is rebuilt every frame its batch is submitted, so a
      // spurious invalidation costs one frame's skip. A resident record is
      // designed to survive an UNBOUNDED number of frames without being
      // rebuilt — that is the entire point of residency — so a spurious
      // invalidation costs an unbounded skip window, and the instances it was
      // holding alive fall back to retiring on numFramesToKeepInstances with
      // nothing reporting why. Left at 0, a clone simply owns no record until
      // some draw genuinely resolves to it, which is the truth.
      // 960 -> 968 on 2026-08-24: added ClaimStage m_claimStage + uint32_t
      // m_claimFrame ([FindSim] blockedBy= — which stage of findSimilarInstance
      // claimed this instance, and on which frame). 5 bytes of payload, 3 of
      // alignment padding.
      //
      // DIAGNOSTIC ONLY, and the only members here that are. They exist to
      // answer whether a placement that lost its partner lost it to a stage with
      // a real right to it (exact/history) or to another distance guess
      // (nearest), which decides whether the batch resolution needs restructuring
      // or the matcher is behaving correctly.
      //
      // COPY CTOR: DELIBERATELY NOT COPIED, and here it is the trivial case
      // rather than a correctness argument. The pair means "the matcher claimed
      // me, this frame". A clone was produced by createInstanceCopy, which no
      // stage of findSimilarInstance ran for, so None/kInvalidFrameIndex is
      // simply true. Inheriting the source's stage would report the clone as
      // having blocked a search it never took part in.
      static_assert(RtInstanceSize == 968, "RtInstance size has changed.  Fix the copy constructor above this message, then update the expected size.");
    };
    CheckRtInstanceSize<sizeof(RtInstance)> _rtInstanceSizeTest;
  }
 #endif

  void RtInstance::setBlas(BlasEntry& blas) {
    m_linkedBlas = &blas;
  }

  // NV-DXVK [MapGate]: why is the SpatialMap never written?
  //
  // *** RESOLVED 2026-08-06, and the premise was FALSE. [MapWrite] logged zero
  // *** lines because the tag IS on a denylist - the HARDCODED emitMsg prefix
  // *** filter in log.cpp (~line 424, silenced 2026-07-30 for the MeshTrace
  // *** runs at 360 lines/frame), not the rtx.conf one this comment checked.
  // *** The reference_log_filter rule exists for exactly this: when a log that
  // *** must fire logs nothing, check log.cpp's built-in array FIRST.
  // *** The writes themselves DO land: [InstReap] samples mapSz=1..8+ at reap
  // *** time, and the 00:48 2026-08-06 run measures only ~49 reaps/frame
  // *** against ~19,000 live instances (0.3%/frame churn). The "58% replaced
  // *** every frame" era below described an earlier build, or a VS-local
  // *** subset, and is NOT the current state. The per-frame write counters on
  // *** the [MapGate] line (mapWrMove/mapWrInsert/mapSzMax, added with this
  // *** note) prove it from the write side without re-enabling the flood.
  // *** The [FindSim]/[MtnDedup] spatialMapSize=0 reads are per-BlasEntry:
  // *** those VSes' lookups hit freshly (re)created BlasEntries whose maps are
  // *** empty by construction - a different, narrower question than "is the
  // *** map ever written".
  //
  // Original premise, kept for the record (numbers are from that older build):
  // both write sites sit inside `if (!m_isCreatedByRenderer)`, and zero lines
  // could not distinguish (a) never called from (b) always renderer-created.
  // Consequences measured then: getNearestData zero candidates on 100% of
  // 17402 lookups, nearest matching 0.00%, 93% of reaps propId=0x0, ~31 fresh
  // instance ids per frame, 58% of a stable population replaced every frame.
  //
  // COUNTERS, not per-call lines: these functions run per instance per frame
  // (hundreds), and the log is already 127 MB. One summary line per frame
  // answers the question without adding to that. No VS gate either - this
  // file's own history records every VS gate here turning into a blind spot,
  // because it only sees writes made while the instance is linked to that VS.
  //
  // mapWritesExpected is the field to read: it counts calls that SHOULD have
  // reached the write. If it is > 0 while [MapWrite] stays 0, the write is
  // being skipped for some third reason and (a)/(b) are both wrong. If it is 0
  // with calls > 0, m_isCreatedByRenderer is the gate. If calls are 0, nothing
  // ever asks these instances to move.
  //
  // NV-DXVK [perf]: as of the sec 4b(b) early-out, a call can legitimately reach
  // the write site and decline to write, so mapWritesExpected no longer equals
  // mapWrMove + mapWrInsert on its own. mapSkipInSync carries the difference and
  // the three must still sum. Compare the SUM, not the write counts.
  //
  // Counters are atomic; the flush uses exchange, so a concurrent increment can
  // be lost across a frame boundary. That costs a unit or two on a count in the
  // hundreds and is not worth a lock on a per-instance path.
  namespace {
    std::atomic<uint32_t> s_mgFrame       { UINT32_MAX };
    std::atomic<uint32_t> s_mgOtcCalls    { 0u };
    std::atomic<uint32_t> s_mgOtcRenderer { 0u };
    std::atomic<uint32_t> s_mgTpCalls     { 0u };
    std::atomic<uint32_t> s_mgTpRenderer  { 0u };
    std::atomic<uint32_t> s_mgNullBlas    { 0u };
    std::atomic<uint32_t> s_mgUnsetFrame  { 0u };
    // Write-side proof, counters not per-call lines (the per-call form is the
    // denylisted [MapWrite] flood): how many SpatialMap move()/insert() calls
    // actually executed this frame, and the largest map they landed in.
    std::atomic<uint32_t> s_mgWrMove      { 0u };
    std::atomic<uint32_t> s_mgWrInsert    { 0u };
    std::atomic<uint32_t> s_mgWrMapSzMax  { 0u };
    // NV-DXVK [perf]: calls that reached the write site and deliberately did not
    // write, because the map is keyed on m_stablePropId and already in sync (see
    // the sec 4b(b) note in onTransformChanged). Counted so mapWritesExpected
    // keeps balancing -- WITHOUT this the skip would present as exactly the
    // "expected > 0 while writes are missing" symptom this probe exists to catch,
    // and the next reader would chase a bug that is a deliberate early-out.
    // The invariant is now:
    //     mapWritesExpected == mapWrMove + mapWrInsert + mapSkipInSync
    std::atomic<uint32_t> s_mgSkipInSync  { 0u };

    // NV-DXVK [perf] 2026-08-07: all three accounting helpers are now gated on
    // rtx.logMapGate (default OFF). They were unconditional.
    //
    // WHY, and it is not the instruction count. [MapGate]'s own output measured
    // onTransformChanged at ~15,441 calls/frame, and each call did 2-4 atomic
    // RMWs on a handful of shared static counters -- ~30-60k contended
    // cache-line operations per frame, all on dxvk-cs, the thread the frame is
    // currently sized by ([Perf.CsSplit] busyPct 93-98).
    //
    // The identical shape was measured in rtx_accel_manager's world-extent
    // census the same day: an unconditional per-instance std::mutex plus
    // atomics, 8,375x/frame. Gating it took dxvk-cs from 70.5 to ~46 ms/frame
    // -- roughly 6x the arithmetic it removed -- and sped up UNRELATED
    // functions on the same thread by 25-30% per call ([Perf.SceneObj] update
    // 0.827 -> 0.568 us at an identical 15.5k call count, `find` 0.269 ->
    // 0.203). Instructions do not do that; cache-coherence traffic does.
    //
    // A gated read of an RtxOption is a plain inlined load with no lock prefix
    // and no line ownership transfer, so the OFF path costs a predictable
    // branch instead of a cross-core round trip.
    //
    // The counters remain exact when enabled -- this changes nothing about what
    // [MapGate] reports, only whether it runs. The write-balance invariant
    // (mapWritesExpected == mapWrMove + mapWrInsert + mapSkipInSync) still
    // holds, because all three helpers gate on the SAME option and therefore
    // switch on and off together. Do not gate them individually.
    void mapSkipAccount() {
      if (!RtxOptions::logMapGate()) { return; }
      s_mgSkipInSync.fetch_add(1u, std::memory_order_relaxed);
    }

    void mapWriteAccount(bool isInsert, uint32_t mapSzAfter) {
      if (!RtxOptions::logMapGate()) { return; }
      (isInsert ? s_mgWrInsert : s_mgWrMove).fetch_add(1u, std::memory_order_relaxed);
      uint32_t seen = s_mgWrMapSzMax.load(std::memory_order_relaxed);
      while (mapSzAfter > seen &&
             !s_mgWrMapSzMax.compare_exchange_weak(seen, mapSzAfter, std::memory_order_relaxed)) {
      }
    }

    void mapGateAccount(uint32_t frame, bool isTeleport, bool isRenderer, bool blasNull) {
      if (!RtxOptions::logMapGate()) { return; }
      // kInvalidFrameIndex IS UINT32_MAX - the same value used as the "no frame
      // seen yet" seed for s_mgFrame - and a brand-new instance carries it
      // until its first update. Letting it drive a frame transition would flip
      // s_mgFrame to the seed and make the NEXT real frame discard the whole
      // bucket as unseeded. With ~31 fresh instances arriving per frame that
      // would fire constantly and quietly zero the measurement. Such calls are
      // still COUNTED (into the current bucket, and separately as unsetFrame);
      // they just never flush.
      if (frame != kInvalidFrameIndex) {
        uint32_t observed = s_mgFrame.load(std::memory_order_relaxed);
        if (observed != frame && s_mgFrame.compare_exchange_strong(observed, frame)) {
          const uint32_t otc  = s_mgOtcCalls.exchange(0u);
          const uint32_t otcR = s_mgOtcRenderer.exchange(0u);
          const uint32_t tp   = s_mgTpCalls.exchange(0u);
          const uint32_t tpR  = s_mgTpRenderer.exchange(0u);
          const uint32_t nb   = s_mgNullBlas.exchange(0u);
          const uint32_t uf   = s_mgUnsetFrame.exchange(0u);
          const uint32_t wrM  = s_mgWrMove.exchange(0u);
          const uint32_t wrI  = s_mgWrInsert.exchange(0u);
          const uint32_t wrSz = s_mgWrMapSzMax.exchange(0u);
          const uint32_t skIS = s_mgSkipInSync.exchange(0u);
          if (observed != kInvalidFrameIndex) {
            Logger::info(str::format(
              "[MapGate] f=", observed,
              " onTransformChanged=", otc,
              " otcIsRenderer=", otcR,
              " teleport1=", tp,
              " tpIsRenderer=", tpR,
              " nullBlas=", nb,
              " unsetFrame=", uf,
              " mapWritesExpected=", (otc - otcR) + (tp - tpR),
              " mapWrMove=", wrM,
              " mapWrInsert=", wrI,
              // NV-DXVK [perf]: deliberate no-write early-outs. Read the balance,
              // not the raw write count: mapWrMove + mapWrInsert + mapSkipInSync
              // should equal mapWritesExpected. A shortfall is still a real
              // missing write; mapSkipInSync alone is not.
              " mapSkipInSync=", skIS,
              " mapSzMax=", wrSz));
          }
        }
      } else {
        s_mgUnsetFrame.fetch_add(1u, std::memory_order_relaxed);
      }

      if (isTeleport) {
        s_mgTpCalls.fetch_add(1u, std::memory_order_relaxed);
        if (isRenderer) {
          s_mgTpRenderer.fetch_add(1u, std::memory_order_relaxed);
        }
      } else {
        s_mgOtcCalls.fetch_add(1u, std::memory_order_relaxed);
        if (isRenderer) {
          s_mgOtcRenderer.fetch_add(1u, std::memory_order_relaxed);
        }
      }
      if (blasNull) {
        s_mgNullBlas.fetch_add(1u, std::memory_order_relaxed);
      }
    }
  }

  void RtInstance::onTransformChanged(const bool objectToWorldChanged, const SpatialKeyHint& keyHint) {
    // NV-DXVK [MapGate]: account BEFORE the m_isCreatedByRenderer branch, which
    // is the whole point - inside it, this call would be as invisible as
    // [MapWrite] already is.
    mapGateAccount(m_frameLastUpdated, /*isTeleport*/ false,
                   m_isCreatedByRenderer, m_linkedBlas == nullptr);

    // NV-DXVK [perf] handoff v5 sec 4b(a): TEST BEFORE RECOMPUTING -- using the
    // CALLER's knowledge, because this function has none of its own.
    // m_vkInstance.transform is a pure function of surface.objectToWorld (a
    // Matrix4 transpose plus a 48-byte memcpy), and move()/moveAgain() already
    // computed the byte-compare that decides it, for their own return value.
    // [MapGate] reports onTransformChanged=15441 per frame and [Perf.UpdInst]
    // reports xfChg=1% standing / 13-18% walking, so the write is landing on
    // bytes it does not change on 82-99% of those calls.
    //
    // This half is universally sound, unlike the spatial-map block below: it
    // reads nothing but surface.objectToWorld, so "the matrix is byte-identical"
    // is the whole precondition.
    if (objectToWorldChanged) {
      // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
      // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
      const auto t = transpose(surface.objectToWorld);
      memcpy(&m_vkInstance.transform, &t, sizeof(VkTransformMatrixKHR));
    }

    if (!m_isCreatedByRenderer) {
      // NV-DXVK [perf] handoff v5 sec 4b(b): when m_stablePropId is non-zero the
      // spatial map is keyed on the PROPID, not on the transform -- SpatialMap
      // ::move takes overrideHash verbatim and never hashes the matrix in that
      // case. So m_spatialCacheHash == m_stablePropId proves both halves at once:
      // this instance is filed under its propId, and the map is already in sync.
      // move() would recompute the same key, compare equal, and return without
      // touching the map -- so everything between here and it is set up for a
      // call that does nothing: calcFirstInstanceObjectToWorld (a Matrix4
      // multiply whenever instancesToObject is set), getTransformedCentroid (an
      // AABB transform), and two accounting reads. Note this case needs no
      // transform test at all, precisely because the key does not depend on the
      // transform.
      //
      // NOT WIDENED to m_stablePropId == 0, and the blocking reason is NOT the
      // one v5 sec 4b gives (the transformHash++ collision path). It is simpler
      // and harder: there the key is XXH64(firstInstanceObjectToWorld), and
      // objectToWorldChanged == false does not prove that matrix is unchanged --
      // calcFirstInstanceObjectToWorld multiplies surface.objectToWorld by
      // (*surface.instancesToObject)[0], and those CONTENTS can change under a
      // stable pointer while objectToWorld stays byte-identical. Skipping on the
      // caller's memcmp would freeze an instanced prop at a stale key. Widening
      // needs the composed matrix compared, not the base one.
      //
      // STILL NOT WIDENED, and it never will be on this test -- but the work it
      // was trying to avoid is now avoided a different way. See the key/centroid
      // block below: rather than guess from the base matrix whether the key
      // changed, we compose the matrix, derive the key (which sec 4a's hash
      // threading made free), and compare the KEY. That is the comparison this
      // note says the widening needs, so the centroid no longer has to be
      // computed speculatively. The early return here remains worth keeping: it
      // is strictly cheaper still, skipping the compose as well.
      if (m_stablePropId != 0
          && m_spatialCacheHash == static_cast<XXH64_hash_t>(m_stablePropId)) {
        // Keep [MapGate] balanced -- see s_mgSkipInSync. An unaccounted skip
        // would read as a missing write, which is the exact failure that probe
        // was built to detect.
        mapSkipAccount();
        return;
      }

      // NOTE: This code would cache instances based on predicted position instead of current position, but in testing it fails too frequently
      // const Vector3 newPos = 2.f * surface.objectToWorld[3].xyz() - surface.prevObjectToWorld[3].xyz();

      // Cache based on current position. Pass m_stablePropId so sub-view-
      // reprojected content (where per-frame transform drift would defeat
      // matrix-bytes hashing) keeps a stable cache key tied to per-prop
      // identity. Default 0 = original matrix-hash behavior.
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();

      // NV-DXVK [perf] handoff v7 sec 4a: discharge the precondition on the
      // caller's precomputed key. The hint carries the matrix it was hashed
      // over; this is the first point where the matrix that will actually be
      // hashed exists, so this is where the two get compared. Byte-identical
      // means XXH64 would return the hint's value, so passing it through is
      // observationally equivalent -- including for a collision-bumped entry,
      // whose stored key is hash+N while move() compares against the unbumped
      // hash exactly as before. Any mismatch (a WorldMatte offset, a differently
      // composed instancesToObject) falls through to hashing, which is the old
      // path with a memcmp in front of it.
      const bool hintUsable = keyHint.isUsable();
      const bool hintMatches =
        hintUsable
        && memcmp(keyHint.matrix->data, firstInstanceObjectToWorld.data, sizeof(Matrix4)) == 0;
      const XXH64_hash_t precomputedMatrixHash = hintMatches ? keyHint.hash : 0;

      // NV-DXVK [KeyDiverge] 2026-08-25: COUNT WHAT THE MEMCMP ABOVE DECIDES.
      //
      // The comparison already existed, as a correctness guard for reusing the
      // lookup's hash. Nobody ever counted its outcome, and its outcome is the
      // whole diagnosis: `keyHint.matrix` is the matrix findSimilarInstance
      // QUERIED with this frame, and `firstInstanceObjectToWorld` is the matrix
      // this write is about to FILE under. When they differ, the entry lands on
      // a key the lookup cannot form, so that placement's exact stage misses
      // every frame from now on — permanently, not occasionally.
      //
      // WHAT THE LOG ALREADY ESTABLISHES, so this probe only has to name the
      // cause rather than prove the effect:
      //   - [MapSupply]: no non-empty map ever runs a query deficit, so there
      //     are enough instances to match.
      //   - [MapLedger]: none=95.5% of history misses, evicted=0, ins=0 — the
      //     queried key was never written on that map, and that is not a
      //     displacement artifact or a cache defect.
      //   - [FanoutPrevMiss]: sameBytes=1 and moved=0 on the sampled misses, so
      //     the READER's matrix is byte-identical frame to frame.
      //   - [SpatialMove]/[SpatialErase]: ~40 re-filings a frame with
      //     [ReapJoin] removed=0, and ledgEntries/mapSz ratios up to 324x, so
      //     the WRITER files a fresh key every frame for a stable population.
      //
      // PRE-REGISTERED READING:
      //   diff ~ the re-file rate   CONFIRMED. The two sides compose different
      //       matrices for the same instance in the same frame, and the fix is
      //       to make them compose one matrix — not to change the matcher, and
      //       not to widen the distance search that is currently papering over
      //       it. dT/dR below say which part of the transform differs.
      //   diff ~ 0 while re-files stay ~40
      //       The divergence is NOT between the lookup and the write. The
      //       writer's own matrix is drifting BETWEEN frames instead, and the
      //       question moves to what mutates surface.objectToWorld each frame.
      //   noHint dominant
      //       This comparison cannot speak for the failing population at all —
      //       split placements pass an empty hint deliberately (see the
      //       processSceneObjectFanout call site) — and the probe has to move to
      //       the write sites that serve splits before any of this is readable.
      {
        struct KeyDivergeAgg {
          // Starts at 0 rather than the usual kInvalidFrameIndex sentinel because
          // the rollover below is a MONOTONIC test — see the note on sDivFrame.
          std::atomic<uint32_t> frame  { 0u };
          std::atomic<uint32_t> same   { 0 };
          std::atomic<uint32_t> diff   { 0 };
          std::atomic<uint32_t> noHint { 0 };
        };
        static KeyDivergeAgg sKeyDiverge;
        const uint32_t nowFrame = m_frameLastUpdated;

        if (!hintUsable) {
          sKeyDiverge.noHint.fetch_add(1, std::memory_order_relaxed);
        } else if (hintMatches) {
          sKeyDiverge.same.fetch_add(1, std::memory_order_relaxed);
        } else {
          sKeyDiverge.diff.fetch_add(1, std::memory_order_relaxed);
          // Capped per frame, and computed only for logged lines. dT is the
          // translation delta and dR the largest basis-column delta: a pure dT
          // with dR ~ 0 is a positional offset applied on one side only, which
          // is what nearestToPrev clustering at a constant ~39 units across
          // props at unrelated world positions already hints at.
          // MONOTONIC, not just "different". The frame here is the INSTANCE's
          // m_frameLastUpdated, and instances reach this function carrying
          // different values, so a plain inequality test would flip the window
          // back and forth between two frame numbers and reset the cap on every
          // alternation — spending far more than four lines a frame, which is
          // trap §6.1 in a new place.
          static std::atomic<uint32_t> sDivFrame { 0u };
          static std::atomic<uint32_t> sDivLines { 0 };
          uint32_t seen = sDivFrame.load(std::memory_order_relaxed);
          if (nowFrame > seen
              && sDivFrame.compare_exchange_strong(seen, nowFrame, std::memory_order_relaxed)) {
            sDivLines.store(0, std::memory_order_relaxed);
          }
          if (sDivLines.fetch_add(1, std::memory_order_relaxed) < 4u) {
            const Matrix4& q = *keyHint.matrix;
            const Matrix4& w = firstInstanceObjectToWorld;
            const Vector3 dT = w[3].xyz() - q[3].xyz();
            float dR = 0.f;
            for (uint32_t c = 0; c < 3u; ++c) {
              for (uint32_t r = 0; r < 3u; ++r) {
                dR = std::max(dR, std::abs(w[c][r] - q[c][r]));
              }
            }
            Logger::info(str::format(
              "[KeyDiverge] f=", nowFrame,
              " vs=0x", std::hex,
                static_cast<uint64_t>(m_linkedBlas != nullptr
                  ? m_linkedBlas->input.getTransformData().vertexShaderHash : 0), std::dec,
              " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
              " instToObj=", (surface.instancesToObject
                                ? static_cast<uint32_t>(surface.instancesToObject->size()) : 0u),
              " dT=(", dT.x, ",", dT.y, ",", dT.z, ")",
              " dTlen=", std::sqrt(dT.x * dT.x + dT.y * dT.y + dT.z * dT.z),
              " dRmax=", dR,
              " qT=(", q[3][0], ",", q[3][1], ",", q[3][2], ")",
              " wT=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
          }
        }

        uint32_t seenFrame = sKeyDiverge.frame.load(std::memory_order_relaxed);
        if (nowFrame > seenFrame
            && sKeyDiverge.frame.compare_exchange_strong(seenFrame, nowFrame, std::memory_order_relaxed)) {
          const uint32_t s = sKeyDiverge.same.exchange(0, std::memory_order_relaxed);
          const uint32_t d = sKeyDiverge.diff.exchange(0, std::memory_order_relaxed);
          const uint32_t n = sKeyDiverge.noHint.exchange(0, std::memory_order_relaxed);
          if (seenFrame != 0u && (s | d | n) != 0u) {
            Logger::info(str::format(
              "[KeyDiverge] f=", seenFrame,
              " same=", s, " diff=", d, " noHint=", n));
          }
        }
      }

      // NV-DXVK [perf]: THE CENTROID IS NOW LAZY.
      //
      // move() reads `centroid` only on the re-file path -- it is forwarded to
      // the re-insert and touched nowhere else. Deriving it costs a load of
      // input.geometryData.boundingBox out of a large BlasEntry (m_linkedBlas
      // itself is hot, that field need not be) plus an AABB validity test and a
      // Matrix4 x Vector4. It was paid by every instance every frame.
      //
      // How often it is actually wanted: [SpatialMove] fires only when the key
      // changes, and its first-32-then-every-4096th throttle puts that at
      // roughly 290 of the 15,450 moves per frame -- about 2%. The other 98%
      // computed a world position that move() then ignored.
      //
      // What unblocked this is sec 4a. Deciding "did the key change?" used to
      // mean hashing the matrix, which is what the note above rejected -- paying
      // a hash to skip an AABB transform is not a trade. Now the hash is already
      // in hand from the lookup, so computeKey is a compare and a select.
      //
      // computeKey is SpatialMap's own, deliberately: this predicate must be the
      // same one move() applies internally. If it drifts, moved instances stop
      // being re-filed, and that neither faults nor logs -- it surfaces frames
      // later as dedup churn. Feeding move() the same precomputedMatrixHash then
      // makes its internal call free, so the key is derived twice but hashed at
      // most once.
      const XXH64_hash_t newKey = BlasEntry::InstanceMap::computeKey(
          firstInstanceObjectToWorld, m_stablePropId, precomputedMatrixHash);
      const Vector3 newPos = (newKey != m_spatialCacheHash)
        ? getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld)
        : Vector3();

      // NV-DXVK [ReFileJit] 2026-08-25: NAME THE OBJECT THAT JITTERS, AND SAY
      // WHICH PART OF ITS TRANSFORM MOVES.
      //
      // WHAT IS ESTABLISHED. [ReFile] counts ~9 re-filings a frame, on 75% of
      // frames, where the centroid moved less than 0.001 units. Those are not
      // animation: 3,428 events landed on only 395 distinct positions, single
      // positions recurring over 100 times each, so the value OSCILLATES back to
      // values it already held. Y and Z were bit-identical on every sample and
      // only X moved, across about four quantised values spanning 4e-4. Every
      // wobble rewrites the XXH64 over the matrix bytes, so the key changes and
      // that object's exact lookup can never hit again.
      //
      // WHY IT IS LOGGED HERE rather than in SpatialMap, where the magnitude
      // test is natural: SpatialMap is templated over T and cannot ask for a
      // shader hash or a prop id, so its version could only ever report that
      // "something" jitters. debugCentroidOf hands the old centroid out so the
      // same test runs here with the identity attached.
      //
      // READ THE TRANSFORM FIELDS, not the distance -- the distance is already
      // known and is not the question:
      //   T moves, basis bit-stable
      //       a translation is being perturbed. Combined with X-only motion,
      //       something adds a varying X offset to this object's world position
      //       every frame. Chase the producer of that offset.
      //   basis moves, T bit-stable
      //       a rotation or scale is being recomposed each frame, and the
      //       centroid shift is that recomposition leaking through the bounding
      //       box. Chase the composition, not the position.
      //   both move
      //       the whole matrix is being rebuilt from parameters rather than
      //       carried, and the fix is to carry it.
      //
      // propId is on the line because a non-zero one would make this moot: the
      // key would be the prop id and the matrix bytes would stop mattering.
      // instToObj is there because a populated instancesToObject means the write
      // composes objectToWorld * instancesToObject[0], which is a second place
      // the wobble could enter.
      if (newKey != m_spatialCacheHash && m_linkedBlas != nullptr) {
        Vector3 oldC;
        if (m_linkedBlas->getSpatialMap().debugCentroidOf(m_spatialCacheHash, oldC)) {
          const Vector3 dv = newPos - oldC;
          const float d = std::sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
          if (d > 0.f && d < 0.001f) {
            static std::atomic<uint32_t> sJitFrame { 0u };
            static std::atomic<uint32_t> sJitLines { 0 };
            uint32_t seen = sJitFrame.load(std::memory_order_relaxed);
            if (m_frameLastUpdated > seen
                && sJitFrame.compare_exchange_strong(seen, m_frameLastUpdated,
                                                     std::memory_order_relaxed)) {
              sJitLines.store(0, std::memory_order_relaxed);
            }
            if (sJitLines.fetch_add(1, std::memory_order_relaxed) < 6u) {
              const Matrix4& m = firstInstanceObjectToWorld;
              Logger::info(str::format(
                "[ReFileJit] f=", m_frameLastUpdated,
                " vs=0x", std::hex,
                  static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash),
                  std::dec,
                " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
                " instToObj=", (surface.instancesToObject
                                  ? static_cast<uint32_t>(surface.instancesToObject->size()) : 0u),
                " isRenderer=", (m_isCreatedByRenderer ? 1 : 0),
                " d=", d,
                " dv=(", dv.x, ",", dv.y, ",", dv.z, ")",
                " T=(", m[3][0], ",", m[3][1], ",", m[3][2], ")",
                " c0=(", m[0][0], ",", m[0][1], ",", m[0][2], ")",
                " c1=(", m[1][0], ",", m[1][1], ",", m[1][2], ")",
                " c2=(", m[2][0], ",", m[2][1], ",", m[2][2], ")",
                " oldKey=0x", std::hex, static_cast<uint64_t>(m_spatialCacheHash),
                " newKey=0x", static_cast<uint64_t>(newKey), std::dec));
            }
          }
        }
      }

      // NV-DXVK [Phase2b]: on a worker during the sharded instance phase, the
      // spatial-map write is RECORDED, not applied. During that phase every
      // SpatialMap is read-only by contract — that is what makes the migration
      // path's cross-shard sibling reads (findSimilarInstance) safe without a
      // lock on this, the hot path. The ordered tail applies the op in arena
      // order via InstanceManager::applyDeferredSpatialOp. In allowMiss (tail)
      // mode the write runs INLINE — single-threaded, and later work in the
      // same tail item must see it (e.g. a later placement finding this one).
      if (inShardedInstancePhase() && !t_shardPhase.allowMiss) {
        // With an earlier op of this frame still pending, m_spatialCacheHash is
        // stale and the newKey comparison above proves nothing — record
        // unconditionally and let SpatialMap::move's own key compare decide at
        // apply time (it no-ops when the key is unchanged).
        const bool chained = (m_spatialOpPendingFrame == m_frameLastUpdated);
        if (newKey != m_spatialCacheHash || chained) {
          DeferredSpatialOp op;
          op.kind = DeferredSpatialOp::Kind::kMove;
          op.instance = this;
          op.targetBlas = m_linkedBlas;
          // Eager centroid on the chained path: the lazy derivation above keyed
          // off the stale hash, so newPos may be empty even though the
          // apply-time move will re-file and read it.
          op.centroid = (newKey != m_spatialCacheHash)
            ? newPos
            : getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
          op.transform = firstInstanceObjectToWorld;
          op.stablePropId = m_stablePropId;
          op.precomputedMatrixHash = precomputedMatrixHash;
          t_shardPhase.info->spatialOps.push_back(op);
          m_spatialOpPendingFrame = m_frameLastUpdated;
        } else {
          // Key provably unchanged, no pending chain: exactly the inline no-op
          // move. Keep [MapGate]'s write balance intact via the skip counter.
          mapSkipAccount();
        }
        return;
      }

      // NV-DXVK [Otc2904]: pre-move state for VS_2904d2 instances. If we
      // see m_spatialCacheHash != m_stablePropId here, this move() will
      // erase the propId slot and re-insert at a new slot — that's how an
      // instance can "exist" yet its propId lookup miss. Log first 32 +
      // every 256th to bound output. Only fires when divergence is real.
      // NV-DXVK [perf]: CONDITION ORDER. The two cheap member compares now gate
      // the BlasEntry deref chain (m_linkedBlas->input.getTransformData()
      // .vertexShaderHash), which was being walked per instance per frame --
      // [MapGate] says 15441x -- for a probe that has logged ZERO lines across a
      // 25,700-frame run. Divergence is the rare case, so testing it first turns
      // the common path into two loads and a not-taken branch. Same predicate,
      // same output, just no longer paid by every instance that cannot match.
      // NV-DXVK 2026-08-23: the VS gate was the literal 0x2904d2163ef31a17, and
      // that shader has never diverged -- zero lines in a 25,700-frame run. The
      // probe was therefore aimed away from every shader under investigation.
      // Driven by rtx.findSimilarProbeVsHashes now, which is the option this
      // file already uses to point the other dedup probes at a shader without a
      // rebuild, so the detector follows the investigation instead of one
      // hardcoded hash. The original stays in the option's default set.
      if (m_stablePropId != 0
          && m_spatialCacheHash != static_cast<XXH64_hash_t>(m_stablePropId)
          && m_linkedBlas != nullptr
          && (m_linkedBlas->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull
              || lookupHash(RtxOptions::findSimilarProbeVsHashes(),
                            m_linkedBlas->input.getTransformData().vertexShaderHash))) {
        static thread_local uint32_t sOtcProbe = 0;
        if (sOtcProbe < 32 || (sOtcProbe & 0xFF) == 0) {
          Logger::warn(str::format(
            "[Otc2904] #", sOtcProbe,
            // The tag predates the option gate and now covers whatever
            // findSimilarProbeVsHashes names, so the shader has to be on the
            // line or the output cannot be attributed.
            " vs=0x", std::hex,
              static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash),
            " spatialCacheHash=0x", m_spatialCacheHash,
            " stablePropId=0x", m_stablePropId, std::dec,
            " DIVERGENCE — about to erase old slot + insert new"));
        }
        sOtcProbe += 1;
      }

      // NV-DXVK [MapWrite]: log every SpatialMap write for the probe VSes, from
      // the WRITE side. The tree-billboard flicker is caused by the tree's map
      // holding a SKY-SPACE entry (~29000 units from the main-world query, owner
      // = the tree's own VS, ownerFrame = curFrame-1) which the 300-unit
      // distance test then rejects, spawning a fresh instance.
      //
      // The writer could not be found from the read side: [MtnDedup] fires on
      // every draw of this VS and logged ZERO sky-space centroids across 1630
      // rows, and the m_delayedRayTracedSky reproject path -- the candidate the
      // handoff named -- is dead code here, since skyReprojectToMainCameraSpace
      // defaults false, making l_forceRaster() unconditionally true so the push
      // is never reached ([SkyTrace.delayPush] fires 0 times with the log filter
      // now un-blanketed). Whatever writes the entry therefore does not pass
      // through findSimilarInstance, so instrument insert/move themselves.
      //
      // isRenderer distinguishes an instance the renderer synthesised from one
      // built from a game draw; cam names the camera the instance was submitted
      // under. Together they identify the path without guessing.
      // Log AFTER the call so the resulting cache KEY can be printed. Joining
      // [MapWrite] to [MapDump2] by blasPtr is unsound -- BlasEntries are
      // destroyed and reallocated constantly, so one address refers to many
      // different objects over a run. The key is derived from the transform
      // bytes (propId is 0 for these props), so a sky-space transform and a
      // main-world one cannot share a key. A key in [MapDump2] with no matching
      // [MapWrite] proves a writer outside these two sites.
      const XXH64_hash_t oldKey = m_spatialCacheHash;
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().move(
          m_spatialCacheHash, newPos, firstInstanceObjectToWorld, this, m_stablePropId,
          precomputedMatrixHash, m_frameLastUpdated);
      // NV-DXVK [MapGate] write-side proof; [MapWrite] below is denylisted in
      // log.cpp so the counter is the only signal that survives to the log.
      mapWriteAccount(/*isInsert*/ false, static_cast<uint32_t>(m_linkedBlas->getSpatialMap().size()));

      // UNGATED per-frame census: one line per instance per frame. No VS gate
      // (every VS gate in this investigation was a blind spot -- it only saw
      // writes made while the instance was linked to that VS's BlasEntry) and no
      // visibility gate either, because m_surfaceIndex / mask are assigned later
      // in the frame by AccelManager and gating on them emitted nothing at all.
      // hidden/frustum/mask/surfIdx are logged as FIELDS so the on-screen subset
      // can be selected offline without dropping objects at capture time.
      //
      // NV-DXVK [perf] 2026-08-06: gated on tagDenied. [MapWrite] is in
      // log.cpp's built-in denylist (log.cpp:444, "360/frame, one line per
      // instance per frame"), so emitMsg discards this line - but str::format
      // still BUILT it first: ~300 characters, 6 floats and ~16 hex/integer
      // conversions plus the allocation, once per instance per frame, on
      // dxvk-cs, the thread [ThreadCensus] measures at ~98% of one core and
      // [Perf.CsSplit] at 98.7 ms of a ~100 ms frame.
      //
      // [Perf.FmtSite] attributed 4,249,584 of the process's 4,325,700
      // float->text conversions per 5s window to THIS ONE CALL - 98.6%. The
      // output was bounded; the work never was. Third occurrence of §5.3 in
      // the CPU handoff.
      //
      // tagDenied is called per-hit rather than cached in a static bool on
      // purpose: the tag index is built lazily on the first filtered message,
      // so an early cached read would latch "not denied" for the whole session,
      // and RTX_D3D11_DIAG=1 has to be able to disable the filter at runtime.
      // The probe is a bucketed prefix compare - orders of magnitude below the
      // string build it replaces.
      if (!Logger::tagDenied("[MapWrite]")) {
        // NV-DXVK [perf]: newPos above is only computed when the key changed, so
        // on the common path it is a default-constructed zero rather than this
        // instance's world position. Re-derive it HERE -- inside the denylist
        // gate, so it costs nothing unless the tag is actually enabled -- because
        // a census that silently logged (0,0,0) for 98% of instances would be
        // worse than no census: this field is exactly what the sky-space-entry
        // hunt reads it for.
        const Vector3 logPos =
          getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
        Logger::info(str::format(
          "[MapWrite] f=", m_frameLastUpdated,
          " hidden=", (censusHidden() ? 1 : 0),
          " inFrustum=", (censusInFrustum() ? 1 : 0),
          " mask=0x", std::hex, censusMask(), std::dec,
          " surfIdx=", censusSurfaceIndex(),
          " op=move vs=0x", std::hex,
            static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash), std::dec,
          " seenCams=0x", std::hex, static_cast<uint32_t>(m_seenCameraTypes.raw()), std::dec,
          " isRenderer=", (m_isCreatedByRenderer ? 1 : 0),
          " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
          " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(m_linkedBlas), std::dec,
          " oldKey=0x", std::hex, static_cast<uint64_t>(oldKey), std::dec,
          " key=0x", std::hex, static_cast<uint64_t>(m_spatialCacheHash), std::dec,
          " mapSz=", m_linkedBlas->getSpatialMap().size(),
          " newPos=(", logPos.x, ",", logPos.y, ",", logPos.z, ")",
          " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
            firstInstanceObjectToWorld[3][1], ",",
            firstInstanceObjectToWorld[3][2], ")"));
      }
    }
  }

  bool RtInstance::teleport(const Matrix4& objectToWorld) {
    // NV-DXVK [MapGate]: the OTHER write site, and the only one that can seed a
    // brand-new entry. Accounted before its own m_isCreatedByRenderer branch,
    // for the same reason as onTransformChanged.
    mapGateAccount(m_frameLastUpdated, /*isTeleport*/ true,
                   m_isCreatedByRenderer, m_linkedBlas == nullptr);

    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = objectToWorld;
    if (!m_isCreatedByRenderer) {
      const Matrix4 firstInstanceObjectToWorld = calcFirstInstanceObjectToWorld();
      const Vector3 centroid = getBlas()->input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
      // NV-DXVK [Phase2b]: defensive — teleport should not be reachable on a
      // worker (new instances are created in the ordered tail, in allowMiss
      // mode, where this runs inline; the portal path defers the whole draw),
      // but if a parallel-phase path does reach it, record the insert like the
      // move site does rather than mutate a map the phase declares read-only.
      if (inShardedInstancePhase() && !t_shardPhase.allowMiss) {
        DeferredSpatialOp op;
        op.kind = DeferredSpatialOp::Kind::kInsert;
        op.instance = this;
        op.targetBlas = m_linkedBlas;
        op.centroid = centroid;
        op.transform = firstInstanceObjectToWorld;
        op.stablePropId = m_stablePropId;
        t_shardPhase.info->spatialOps.push_back(op);
        m_spatialOpPendingFrame = m_frameLastUpdated;
      } else {
      // NV-DXVK [MapWrite]: see the note at the move() site in
      // onTransformChanged. teleport() is the OTHER way an entry enters the map,
      // and unlike move() it can seed a brand-new entry outright.
      // See the note at the move() site: log after the call so the resulting
      // cache key is available for an exact join against [MapDump2].
      m_spatialCacheHash = m_linkedBlas->getSpatialMap().insert(
          centroid, firstInstanceObjectToWorld, this, m_stablePropId, m_frameLastUpdated);
      // NV-DXVK [MapGate] write-side proof; see the move() site.
      mapWriteAccount(/*isInsert*/ true, static_cast<uint32_t>(m_linkedBlas->getSpatialMap().size()));

      // See the move() site: ungated census, visibility logged as fields.
      // NV-DXVK [perf] 2026-08-06: tagDenied gate, same reasoning as the move()
      // site - the tag is denylisted, so this string was built and thrown away.
      if (!Logger::tagDenied("[MapWrite]")) {
        Logger::info(str::format(
          "[MapWrite] f=", m_frameLastUpdated,
          " hidden=", (censusHidden() ? 1 : 0),
          " inFrustum=", (censusInFrustum() ? 1 : 0),
          " mask=0x", std::hex, censusMask(), std::dec,
          " surfIdx=", censusSurfaceIndex(),
          " op=teleport vs=0x", std::hex,
            static_cast<uint64_t>(m_linkedBlas->input.getTransformData().vertexShaderHash), std::dec,
          " seenCams=0x", std::hex, static_cast<uint32_t>(m_seenCameraTypes.raw()), std::dec,
          " isRenderer=", (m_isCreatedByRenderer ? 1 : 0),
          " propId=0x", std::hex, static_cast<uint64_t>(m_stablePropId), std::dec,
          " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(m_linkedBlas), std::dec,
          " key=0x", std::hex, static_cast<uint64_t>(m_spatialCacheHash), std::dec,
          " mapSz=", m_linkedBlas->getSpatialMap().size(),
          " newPos=(", centroid.x, ",", centroid.y, ",", centroid.z, ")",
          " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
            firstInstanceObjectToWorld[3][1], ",",
            firstInstanceObjectToWorld[3][2], ")"));
      }
      }  // NV-DXVK [Phase2b]: end of the deferred/inline insert split.
    }

    // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
    // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
    const auto t = transpose(surface.objectToWorld);
    memcpy(&m_vkInstance.transform, &t, sizeof(VkTransformMatrixKHR));

    return false; // freshly teleported instances are always treated as still.
  }

  bool RtInstance::teleport(const Matrix4& objectToWorld, const Matrix4& prevObjectToWorld) {
    surface.objectToWorld = objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = prevObjectToWorld;
    onTransformChanged();

    return memcmp(surface.prevObjectToWorld.data, surface.objectToWorld.data, sizeof(Matrix4)) != 0;
  }

  void RtInstance::teleportWithHistory(const Matrix4& oldToNew) {
    surface.objectToWorld = oldToNew * surface.objectToWorld;
    surface.normalObjectToWorld = transpose(inverse(Matrix3(surface.objectToWorld)));
    surface.prevObjectToWorld = oldToNew * surface.prevObjectToWorld;
    onTransformChanged();

    if (m_primInstanceOwner.isRoot(this)) {
      // this is the root of a replacement - need to update the transform history for all the instances in the replacement.
      for (size_t i = 0; i < m_primInstanceOwner.getReplacementInstance()->prims.size(); i++) {
        RtInstance* instance = m_primInstanceOwner.getReplacementInstance()->prims[i].getInstance();
        if (instance != nullptr && instance != this) {
          instance->teleportWithHistory(oldToNew);
        }
      }
    }
  }
  
  // NV-DXVK [perf]: TEST BEFORE RECOMPUTING.
  //
  // normalObjectToWorld is a pure function of objectToWorld, and the invariant
  // "normalObjectToWorld == transpose(inverse(Matrix3(objectToWorld)))" is held
  // by exactly five sites -- the three teleport overloads plus move/moveAgain --
  // each of which writes the two together. So when the incoming matrix is
  // byte-identical to the stored one, the recompute lands on the value that is
  // already there and skipping it is observationally equivalent.
  //
  // WHY IT MATTERS: inverse(Matrix3) is DOUBLE precision -- ~30 double
  // multiplies and a double divide, then 18 narrowing converts -- plus a Matrix3
  // construction and a transpose. move() runs once per INSTANCE per frame
  // (~16,100 by [Perf.UpdInst], first=99%), and that probe measures xfChg at
  // 6-18%. So on 82-94% of instances the old order computed the whole thing and
  // only afterwards discovered, via the memcmp it already had to do, that
  // nothing had moved. The comparison was always there; it was just downstream
  // of the work it could have skipped.
  //
  // The memcmp is over the full Matrix4 while the normal matrix depends only on
  // the upper 3x3. That is deliberately conservative: a translation-only change
  // recomputes unnecessarily, which is correct but not maximally lazy, and it
  // lets one comparison serve both this decision and the return value.
  //
  // NOW ALSO EXTENDED to onTransformChanged(), which takes the same flag -- see
  // the two notes at the top of that function. It splits in two: the
  // transpose+memcpy into m_vkInstance.transform is a pure function of
  // surface.objectToWorld and is skipped here on the same proof, while the
  // spatial-map block is skipped only on the m_stablePropId != 0 path, where the
  // map key does not depend on the transform at all. The m_stablePropId == 0
  // path still runs every frame: its key is XXH64(calcFirstInstanceObjectToWorld
  // ()), which composes objectToWorld with (*instancesToObject)[0], and this
  // memcmp says nothing about the latter.
  bool RtInstance::move(const Matrix4& objectToWorld, const SpatialKeyHint& keyHint) {
    // See the note below on why this comparison moved above the recompute.
    // The transform has changed even a tiny bit: the result feeds the 'isStatic'
    // surface flag, which the GPU uses to skip motion vector calculation. We need
    // nonzero motion vectors on objects moving even slightly to make RTXDI
    // temporal bias correction work. This comparison is not robust if the
    // transforms are reconstructed from baked object-to-view matrices, but it
    // works well e.g. in Portal. Even if it detects truly static objects as
    // moving, that's fine because that will only have a minor performance effect
    // of calculating extra motion vectors.
    const bool transformChanged =
      memcmp(surface.objectToWorld.data, objectToWorld.data, sizeof(Matrix4)) != 0;

    surface.prevObjectToWorld = surface.objectToWorld;
    surface.objectToWorld = objectToWorld;
    if (transformChanged) {
      surface.normalObjectToWorld = transpose(inverse(Matrix3(objectToWorld)));
    }
    onTransformChanged(transformChanged, keyHint);

    return transformChanged;
  }

  bool RtInstance::moveAgain(const Matrix4& objectToWorld, const SpatialKeyHint& keyHint) {
    // Note the two comparisons are against DIFFERENT references and both are
    // needed. The normal matrix depends on whether objectToWorld itself changed;
    // the return value is whether it differs from the frame's PREVIOUS transform,
    // which move() already advanced. They coincide in move() and do not here.
    const bool objectToWorldChanged =
      memcmp(surface.objectToWorld.data, objectToWorld.data, sizeof(Matrix4)) != 0;

    surface.objectToWorld = objectToWorld;
    if (objectToWorldChanged) {
      surface.normalObjectToWorld = transpose(inverse(Matrix3(objectToWorld)));
    }
    // Note this is objectToWorldChanged, NOT the return value below. The vk
    // transform tracks surface.objectToWorld, which is what this compare covers;
    // the return value asks a different question (against prevObjectToWorld).
    onTransformChanged(objectToWorldChanged, keyHint);

    // See comment in move()
    return memcmp(surface.prevObjectToWorld.data, surface.objectToWorld.data, sizeof(Matrix4)) != 0;
  }

  void RtInstance::setFrameCreated(const uint32_t frameIndex) {
    m_frameCreated = frameIndex;
  }

  // Sets frame id of last update, if this is the first time the frame id is set
  // instance's per frame state is reset as well
  // Returns true if this is the first update this frame
  bool RtInstance::setFrameLastUpdated(const uint32_t frameIndex) {
    if (m_frameLastUpdated != frameIndex) {
      m_seenCameraTypes.clrAll();

      m_frameLastUpdated = frameIndex;

      return true;
    }

    return false;
  }

  void RtInstance::markForGarbageCollection() const {
    m_isMarkedForGC = true;
  }

  void RtInstance::markAsUnlinkedFromBlasEntryForGarbageCollection() const {
    m_isUnlinkedForGC = true;
  }

  void RtInstance::markAsInsideFrustum() const {
    m_isInsideFrustum = true;
  }

  void RtInstance::markAsOutsideFrustum() const {
    m_isInsideFrustum = false;
  }

  bool RtInstance::registerCamera(CameraType::Enum cameraType, uint32_t frameIndex) {
    const bool settingNewCameraType = !m_seenCameraTypes.test(cameraType);

    if (settingNewCameraType)
      m_seenCameraTypes.set(cameraType);

    return settingNewCameraType;
  }

  bool RtInstance::isCameraRegistered(CameraType::Enum cameraType) const {
    return m_seenCameraTypes.test(cameraType);
  }

  void RtInstance::setCustomIndexBit(uint32_t oneBitMask, bool value) {
    m_vkInstance.instanceCustomIndex = setBit(m_vkInstance.instanceCustomIndex, value, oneBitMask);
  }

  bool RtInstance::getCustomIndexBit(uint32_t oneBitMask) const {
    return m_vkInstance.instanceCustomIndex & oneBitMask;
  }

  bool RtInstance::isOpaque() const {
    return getMaterialType() == MaterialDataType::Opaque;
  }

  bool RtInstance::isViewModel() const {
    return getCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL);
  }

  bool RtInstance::isViewModelNonReference() const {
    return m_vkInstance.mask != 0 && isViewModel();
  }

  bool RtInstance::isViewModelReference() const { 
    return m_vkInstance.mask == 0 && isViewModel();
    }

  bool RtInstance::isViewModelVirtual() const {
    return m_vkInstance.mask & OBJECT_MASK_VIEWMODEL_VIRTUAL;
  }

  void RtInstance::printDebugInfo() const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "=== RtInstance Debug Info ===\n",
      "ID: ", m_id, "\n",
      "Vector Index: ", m_instanceVectorId, "\n",
      "Frame Created: ", m_frameCreated, "\n",
      "Frame Last Updated: ", m_frameLastUpdated, "\n",
      "Frame Age: ", getFrameAge(), "\n",
      "\n",
      "=== Transform Info ===\n",
      "World Position: (", getWorldPosition().x, ", ", getWorldPosition().y, ", ", getWorldPosition().z, ")\n",
      "Transform Matrix:\n",
      "  [", getTransform()[0][0], ", ", getTransform()[0][1], ", ", getTransform()[0][2], ", ", getTransform()[0][3], "]\n",
      "  [", getTransform()[1][0], ", ", getTransform()[1][1], ", ", getTransform()[1][2], ", ", getTransform()[1][3], "]\n",
      "  [", getTransform()[2][0], ", ", getTransform()[2][1], ", ", getTransform()[2][2], ", ", getTransform()[2][3], "]\n",
      "  [", getTransform()[3][0], ", ", getTransform()[3][1], ", ", getTransform()[3][2], ", ", getTransform()[3][3], "]\n",
      "Previous World Position: (", getPrevWorldPosition().x, ", ", getPrevWorldPosition().y, ", ", getPrevWorldPosition().z, ")\n",
      "\n",
      "=== BLAS Info ===\n",
      "Linked BLAS: ", m_linkedBlas ? "Valid" : "Null"));
    
    if (m_linkedBlas) {
      Logger::warn("=== BLAS Entry Debug Info ===");
      m_linkedBlas->printDebugInfo("(from RtInstance)");
      Logger::warn("=== End BLAS Entry Debug Info ===");
      
      // Print DrawCallState info
      Logger::warn("=== DrawCallState Debug Info ===");
      m_linkedBlas->input.printDebugInfo("(from RtInstance)");
      Logger::warn("=== End DrawCallState Debug Info ===");
    }
    
    // Print RtSurface info
    Logger::warn("=== RtSurface Debug Info ===");
    surface.printDebugInfo("(from RtInstance)");
    Logger::warn("=== End RtSurface Debug Info ===");
    
    Logger::warn(str::format(
      "=== Hash Info ===\n",
      "Material Hash: 0x", std::hex, m_materialHash, std::dec, "\n",
      "Material Data Hash: 0x", std::hex, m_materialDataHash, std::dec, "\n",
      "Texcoord Hash: 0x", std::hex, m_texcoordHash, std::dec, "\n",
      "Index Hash: 0x", std::hex, m_indexHash, std::dec, "\n",
      "Spatial Cache Hash: 0x", std::hex, m_spatialCacheHash, std::dec, "\n",
      "\n",
      "=== Vulkan Instance Info ===\n",
      "VK Instance Mask: ", m_vkInstance.mask, "\n",
      "VK Instance Flags: ", m_vkInstance.flags, "\n",
      "VK Instance Custom Index: ", m_vkInstance.instanceCustomIndex, "\n",
      "VK Instance SBT Record Offset: ", m_vkInstance.instanceShaderBindingTableRecordOffset, "\n",
      "\n",
      "=== Material Info ===\n",
      "Material Type: ", static_cast<int>(m_materialType), "\n",
      "Albedo Opacity Texture Index: ", m_albedoOpacityTextureIndex, "\n",
      "Sampler Index: ", m_samplerIndex, "\n",
      "Secondary Opacity Texture Index: ", m_secondaryOpacityTextureIndex, "\n",
      "Secondary Sampler Index: ", m_secondarySamplerIndex, "\n",
      "\n",
      "=== Surface Info ===\n",
      "Surface Index: ", m_surfaceIndex, "\n",
      "Previous Surface Index: ", m_previousSurfaceIndex, "\n",
      "\n",
      "=== Billboard Info ===\n",
      "First Billboard Index: ", m_firstBillboard, "\n",
      "Billboard Count: ", m_billboardCount, "\n",
      "\n",
      "=== Geometry Info ===\n",
      "Geometry Flags: ", m_geometryFlags, "\n",
      "\n",
      "=== Boolean Flags ===\n",
      "Is Hidden: ", m_isHidden ? "true" : "false", "\n",
      "Is Player Model: ", m_isPlayerModel ? "true" : "false", "\n",
      "Is World Space UI: ", m_isWorldSpaceUI ? "true" : "false", "\n",
      "Is Unordered: ", m_isUnordered ? "true" : "false", "\n",
      "Is Object To World Mirrored: ", m_isObjectToWorldMirrored ? "true" : "false", "\n",
      "Is Created By Renderer: ", m_isCreatedByRenderer ? "true" : "false", "\n",
      "Is Animated: ", m_isAnimated ? "true" : "false", "\n",
      "Is Front Face Flipped: ", isFrontFaceFlipped ? "true" : "false", "\n",
      "\n",
      "=== Garbage Collection Flags ===\n",
      "Is Marked For GC: ", m_isMarkedForGC ? "true" : "false", "\n",
      "Is Unlinked For GC: ", m_isUnlinkedForGC ? "true" : "false", "\n",
      "Is Inside Frustum: ", m_isInsideFrustum ? "true" : "false", "\n",
      "\n",
      "=== View Model Flags ===\n",
      "Is View Model: ", isViewModel() ? "true" : "false", "\n",
      "Is View Model Non Reference: ", isViewModelNonReference() ? "true" : "false", "\n",
      "Is View Model Reference: ", isViewModelReference() ? "true" : "false", "\n",
      "Is View Model Virtual: ", isViewModelVirtual() ? "true" : "false", "\n",
      "\n",
      "=== Category Info ===\n",
      "Category Flags: ", m_categoryFlags.raw(), "\n",
      "\n",
      "=== Camera Types ===\n",
      "Seen Camera Types Mask: ", m_seenCameraTypes.raw()));

    for (uint32_t type = 0; type < CameraType::Count; ++type) {
      if (m_seenCameraTypes.test(static_cast<CameraType::Enum>(type))) {
        Logger::warn(str::format("  Camera Type: ", type));
      }
    }
    
    Logger::warn(str::format(
      "\n=== Billboard Indices ===\n",
      "Billboard Indices Count: ", billboardIndices.size()));
    
    for (size_t i = 0; i < std::min(billboardIndices.size(), size_t(5)); ++i) {
      Logger::warn(str::format("  Billboard Index ", i, ": ", billboardIndices[i]));
    }
    if (billboardIndices.size() > 5) {
      Logger::warn(str::format("  ... and ", billboardIndices.size() - 5, " more"));
    }
    
    Logger::warn(str::format(
      "\n=== Index Offsets ===\n",
      "Index Offsets Count: ", indexOffsets.size()));
    
    for (size_t i = 0; i < std::min(indexOffsets.size(), size_t(5)); ++i) {
      Logger::warn(str::format("  Index Offset ", i, ": ", indexOffsets[i]));
    }
    if (indexOffsets.size() > 5) {
      Logger::warn(str::format("  ... and ", indexOffsets.size() - 5, " more"));
    }
    
    Logger::warn("=== End RtInstance Debug Info ===");
#endif
}

  InstanceManager::InstanceManager(DxvkDevice* device, ResourceCache* pResourceCache)
    : CommonDeviceObject(device)
    , m_pResourceCache(pResourceCache) {
    m_previousViewModelState = RtxOptions::ViewModel::enable();
  }

  InstanceManager::~InstanceManager() {
  }

  void InstanceManager::removeEventHandler(void* eventHandlerOwnerAddress) {
    for (auto eventIter = m_eventHandlers.begin(); eventIter != m_eventHandlers.end(); eventIter++) {
      if (eventIter->eventHandlerOwnerAddress == eventHandlerOwnerAddress) {
        m_eventHandlers.erase(eventIter);
        break;
      }
    }
  }

  void InstanceManager::clear() {
    // NV-DXVK [InstClearProbe]: gameplay-only. SceneManager::clear() is the
    // only known external caller and has its own [SceneClearProbe], but
    // 109 instances vanished between f=1176 and f=1178 with no probe
    // firing — log here so we catch any third caller, plus a stack-trace
    // hint (the frame ID + pre-clear size) for correlation. Skip the load
    // window where clears are expected and high-volume.
    if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      const uint32_t curF = m_device->getCurrentFrameId();
      Logger::warn(str::format(
        "[InstClearProbe] f=", curF,
        " preSize=", m_instances.size(),
        " — InstanceManager::clear() entered during gameplay"));
    }

    for (RtInstance* instance : m_instances) {
      removeInstance(instance);
      delete instance;
    }

    m_instances.clear();
    // NV-DXVK [Perf.PushInst] PHASE 2: every instance the records referenced has
    // just been deleted. removeInstance above already invalidated each record it
    // reached, but drop the map wholesale rather than leaving invalidated husks
    // behind -- clear() is a level transition, so nothing here is worth keeping,
    // and an empty map cannot serve a pointer into freed memory by any route.
    m_fanoutRecords.clear();
    // NV-DXVK [ResidentScene]: same reasoning, and residency needs it MORE.
    // clear() is a level transition, and a resident record is by construction
    // the one thing in this file designed to outlive an unbounded number of
    // frames -- so it is the one thing that would otherwise carry pointers to
    // the previous level's freed instances into the next one. This is also the
    // point the seed pass (RESIDENT_SCENE_PLAN.md 0.0) is anchored to: the set
    // must be repopulated from the new level's draws, never inherited.
    m_residentScene.clear();
    m_viewModelCandidates.clear();
    m_playerModelInstances.clear();
  }

  void InstanceManager::garbageCollection() {
    // NV-DXVK [perf]: master gate for the per-frame instance-census diagnostics
    // (GcEntry / SubViewVsCensus / MtnCensus / HullCensus / DropTrace / RiddenTrace)
    // left over from the mountain / vanishing-ship / dropship investigations. Each is
    // a full O(N) walk over m_instances plus str::format+Logger EVERY frame on the
    // RT-branch thread — measured at gc=67-87ms/frame for ~558 instances in
    // [Perf.PrepScene], i.e. the single biggest chunk of prepareSceneData. The walks
    // are read-only (no effect on the real reaper below); flip to true to re-enable
    // any of these investigations.
    constexpr bool kEnableGcCensus = false;

    // Can be configured per game: 'rtx.numFramesToKeepInstances'
    const uint32_t numFramesToKeepInstances = RtxOptions::numFramesToKeepInstances();

    // NV-DXVK [GcEntry probe]: log the value GC actually uses at decision
    // time. The Rm2904 probe in removeInstance reads RtxOptions later in
    // the call stack and may see a different (post-apply) value if
    // RtxOptionManager::applyPendingValues races between the two reads.
    //
    // Also track removed-this-pass count so we can correlate spikes in
    // removeInstance calls with specific GC passes.
    uint32_t probeRemovedThisPass = 0;
    uint32_t probeKeptThisPass    = 0;
    uint32_t probeRemovedMarked   = 0;
    uint32_t probeRemovedLifetime = 0;
    uint32_t probeRemovedForce    = 0;
    const uint32_t probeFrame     = m_device->getCurrentFrameId();

    // NV-DXVK [ReapJoin] — THE PER-MESH DRAW-ARRIVAL JOIN.
    //
    // The open question after the 2026-07-31 05:00 measurement: 437 on-screen
    // meshes are destroyed and recreated on a ~5-frame cycle, continuously,
    // and joining [InstReap] against [BulkPush] showed the SHADER was submitted
    // on 100% of reap frames. But BulkPush is per-VS, and a VS draws many
    // meshes, so that could not prove THIS mesh's draw arrived.
    //
    // BlasEntry::frameLastTouched closes it exactly. It is set unconditionally
    // in SceneManager::processDrawCallState (rtx_scene_manager.cpp:2952) for
    // every arriving draw, keyed on the geometry hash — i.e. it means "a draw
    // for THIS geometry arrived this frame", per mesh, not per shader.
    //
    // ORDERING IS VERIFIED, and it is the whole reason this probe is valid:
    // garbageCollection() runs from inside prepareSceneData
    // (rtx_scene_manager.cpp:4009), which runs AFTER the frame's draws have
    // been processed. So frameLastTouched is already stamped with the current
    // frame by the time we reap. If GC ran first this would read one frame
    // stale on every line and report a false negative every time.
    //
    // THE ABOVE WAS WRONG, AND THE MEASUREMENT THAT KILLED IT (2026-08-05):
    // frameLastTouched is per-GEOMETRY, and every instance of a duplicated prop
    // shares one BlasEntry. So a single sibling's draw stamps the entry and
    // EVERY sibling reaped that frame reports drew=1, whether or not its own
    // draw arrived. Cross-tab over 693 frames / 5661 reaps: 4996 reaps with
    // drew=1 and NOT ONE of them had linked==1; 1168 of 1212 drew=0 reaps had
    // linked==1. drew= was a perfect proxy for "this mesh has siblings" and told
    // us nothing about the reaped instance. That is the same per-shader-vs-per-
    // mesh confound this block claimed to have fixed relative to [BulkPush] —
    // it moved from per-VS to per-geometry instead of being removed.
    //
    // THE FIX IS A COUNT, NOT A FLAG. BlasEntry::drawCount says how many draws
    // resolved to this geometry this frame; walking getLinkedInstances() splits
    // the siblings that were claimed this frame into ones CREATED this frame
    // (m_frameCreated == currentFrame) and ones that already existed. That makes
    // the verdict per-instance and unambiguous:
    //
    //   respawn  a sibling was CREATED this frame on this same geometry while
    //            this older instance went unclaimed. The draw existed and dedup
    //            spent it on a new id => MATCHING failure, in findSimilarInstance
    //            / the spatial map, and it is ours.
    //   starved  no sibling was created; the geometry simply received fewer
    //            draws than it had instances => the copy count genuinely fell,
    //            a submission gap upstream of Remix.
    //
    // draws= is printed raw next to freshSib=/reusedSib= so the identity
    // draws == freshSib + reusedSib can be checked rather than assumed; a
    // shortfall means a draw resolved here and its instance was then migrated
    // to another entry (the engine-class relink below), which is a third state
    // and must not be silently folded into either verdict.
    uint32_t probeReapRespawn = 0;  // reaped while a fresh sibling took its place (ours)
    uint32_t probeReapStarved = 0;  // reaped with no replacement (submission gap)
    if constexpr (kEnableGcCensus) {
      Logger::info(str::format(
        "[GcEntry] f=", probeFrame,
        " keepInstances=", numFramesToKeepInstances,
        " antiCullEnabled=", (RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1 : 0),
        " instCount=", m_instances.size()));
    }

    // NV-DXVK [InstDriftProbe]: catch silent removals between GC passes.
    // Tracking is UNCONDITIONAL so that the first gameplay GC has valid
    // history from the last pre-gameplay GC (otherwise sLastGcExitSize
    // stays UINT32_MAX and we miss the very first drift event). Only the
    // LOG is gameplay-gated. Drift events are naturally rare (one per
    // size-shrink between GCs), so no rate-limit needed.
    static uint32_t sLastGcExitSize  = UINT32_MAX;
    static uint32_t sLastGcExitFrame = UINT32_MAX;
    {
      const uint32_t currentSize = static_cast<uint32_t>(m_instances.size());
      const bool inGameplay = tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      if (inGameplay && sLastGcExitSize != UINT32_MAX && currentSize < sLastGcExitSize) {
        const int32_t drift = static_cast<int32_t>(currentSize) - static_cast<int32_t>(sLastGcExitSize);
        Logger::warn(str::format(
          "[InstDriftProbe] f=", probeFrame,
          " prevGcExitFrame=", sLastGcExitFrame,
          " prevGcExitSize=", sLastGcExitSize,
          " currentSize=", currentSize,
          " drift=", drift,
          " frameJump=", (probeFrame - sLastGcExitFrame),
          " — m_instances shrunk between GC passes; clear() or external removal path active"));
      }
      // sLastGcExit* are updated unconditionally at GcExit below.
    }

    // Remove instances past their lifetime or marked for GC explicitly
    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // NV-DXVK [HeldRaw] burst arming. One frame's worth of held instances every
    // 10 frames, capped -- see the dump site below. Armed here rather than at
    // the dump so a burst comes from ONE pass and is a snapshot of the held set
    // rather than a sample smeared across several seconds of it.
    static uint32_t s_heldRawBurstFrame = 0u;
    static uint32_t s_heldRawLastBurst  = 0u;
    static uint32_t s_heldRawPrinted    = 0u;
    if (currentFrame - s_heldRawLastBurst >= 10u) {
      s_heldRawLastBurst  = currentFrame;
      s_heldRawBurstFrame = currentFrame;
      s_heldRawPrinted    = 0u;
    }

    // NV-DXVK [PitchProbe]: per-frame instance population vs CAMERA PITCH.
    //
    // WHY THIS AND NOT [Perf.Block]: the symptom is "look down and everything
    // disappears", which is a per-frame event keyed on view direction. Every
    // number we have so far is a 5-second window aggregate, and a 25% dip in a
    // 5s bucket cannot distinguish "everything vanished for 40 frames" from
    // "slightly fewer draws throughout". Binning by the firing condition is the
    // only way to read this: pitch is the independent variable, instance count
    // is the dependent one.
    //
    // READ IT: sort the lines by pitchDeg and look at inst.
    //   inst falls off a cliff past some pitchDeg  => view-direction driven, and
    //     the cliff angle tells us which frustum/view test to go find.
    //   inst uncorrelated with pitchDeg            => NOT view-direction driven;
    //     stop looking at frustum culls entirely (again) and look at what else
    //     changes when the player looks down.
    // dInst is the frame-over-frame delta so a collapse is visible without
    // needing to diff lines by hand.
    //
    // Cost: one camera read + one size() + one log line per frame, no O(N) walk.
    // Gameplay-gated so menu/loading frames do not fill the log.
    //
    // !! THE GATE MUST NOT DEPEND ON THE ENGINE HOOKS. This read
    // !! g_engineHookCaptureCount ALONE, and that counter is incremented by an
    // !! engine-code detour (d3d11_rtx.cpp:37758), which RTX_DISABLE_ENGINE_HOOKS=1
    // !! switches off. The result was silent and total: [PitchProbe], [OccProbe],
    // !! [JobProbe], [DrainProbe], [DispProbe], [AreaSeed] and [AreaDump] ALL
    // !! emitted zero lines for an entire capture while every island cheerfully
    // !! logged INSTALLED, because installs are deliberately not gated on that
    // !! env var (d3d11_rtx.cpp:39770) but this emitter was gated on it by
    // !! accident. A probe that reports nothing is indistinguishable from a probe
    // !! that measured nothing -- the exact failure mode CULLING_BIBLE sec 10.2
    // !! is about -- and it has now cost a session twice.
    //
    // m_instances.size() is the hook-independent gameplay signal: a menu or
    // loading frame carries a handful of instances, gameplay carries thousands
    // (5,700-10,600 in every capture on this map). OR, not AND, so the probe
    // survives either signal being unavailable.
    //
    // RTX_DISABLE_ENGINE_HOOKS=1 IS THE INTENDED WORKING CONFIGURATION HERE --
    // do not "fix" it by clearing the variable. The engine detours are not
    // wanted for this investigation; only the bypass hooks are. So the probe
    // has to work with the master switch OFF, permanently, and this gate is
    // what makes that true.
    //
    // What the switch does and does not change:
    //   NOT affected -- the client.dll area-culling probes ([DispProbe] a20,
    //     qStart/qSkip, faceAreas, walkAreas). Those measure sub_1802EB620,
    //     the GAME's own culling, which runs identically either way.
    //   Affected -- provenance of the camera fields below. Without the
    //     engine-hook main camera (rtx.useEngineHookMainCamera) the Main
    //     RtCamera is whatever dxvk derives, so pitchDeg/yawDeg/camPos and the
    //     instance counts come from a different camera path than in captures
    //     taken with hooks on. Bin within a run, not across that boundary --
    //     hookN on the line below says which side a frame is on.
    {
      // NV-DXVK [Perf] 2026-08-08: rtx.logCullProbes leads the conjunction so the
      // ~15,500-iteration classification loop below, not just its two emits, is
      // what the option gates. Measured context: with the per-draw/per-instance
      // perf probes on, this frame ran 106 ms; with them off, 48 ms. This probe
      // is the same shape as the ones that cost that, and it had no switch at all.
      const bool inGameplayPitch =
        RtxOptions::logCullProbes() &&
        (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u ||
         m_instances.size() > 256u);
      if (inGameplayPitch) {
        const RtCamera& mainCam =
          m_device->getCommon()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
        const Matrix4 v2w = mainCam.getViewToWorld(false);
        // Column 2 is the camera forward axis; TF2 world space is Z-up, so the
        // Z component of forward IS the pitch sine. Straight ahead ~0, looking
        // straight down ~-1 (reported as -90 deg).
        const float fwdZ = std::max(-1.0f, std::min(1.0f, v2w[2].z));
        const float pitchDeg = std::asin(fwdZ) * (180.0f / 3.14159265358979f);
        // NV-DXVK: YAW. Added 2026-08-05 because its absence made a whole class
        // of symptom unmeasurable: the probe reported pitchDeg pinned at 10.88
        // for an entire capture in which the player was sweeping the camera
        // HORIZONTALLY the whole time. Pitch alone cannot distinguish "camera
        // still" from "camera yawing hard", so every series binned against it
        // was silently binning unrelated frames together.
        // Range -180..180, atan2 of the forward axis' X/Y (TF2 world is Z-up).
        const float yawDeg =
          std::atan2(v2w[2].y, v2w[2].x) * (180.0f / 3.14159265358979f);
        const Vector3& camPos = mainCam.getPosition(/* freecam = */ false);

        const uint32_t instNow = static_cast<uint32_t>(m_instances.size());
        static uint32_t sPrevInst = 0;
        const int32_t dInst = static_cast<int32_t>(instNow) - static_cast<int32_t>(sPrevInst);
        sPrevInst = instNow;

        // NV-DXVK [OccProbe]: are OFF-SCREEN OCCLUDERS actually present?
        //
        // THE SYMPTOM THIS ANSWERS: look one way and outdoor light floods in,
        // look another way and it goes dark. That is not a light being culled —
        // it is the GEOMETRY THAT SHOULD BLOCK the light missing from the TLAS,
        // so the sun leaks through where a wall ought to be. A path tracer needs
        // the occluder present even when it is behind the camera; a rasteriser
        // never does, which is why the engine has no reason to submit it.
        //
        // outFr is the count that matters: instances alive but OUTSIDE the view
        // frustum. Those ARE the off-screen occluders.
        //   outFr healthy and steady while yawing => occluders are present, the
        //     light leak is NOT a missing-geometry problem, look at the lighting.
        //   outFr ~0, or collapsing as you turn => off-screen geometry is being
        //     dropped before it ever reaches the TLAS, and that is the leak.
        //
        // THE TEST IS COMPUTED HERE, NOT READ FROM THE INSTANCE.
        //
        // The first version of this probe counted RtInstance::m_isInsideFrustum
        // and reported outFr=0 on all 459 sampled frames. That was a PROBE
        // DEFECT, not a result: markAsOutsideFrustum() is only reachable from
        // the branch at rtx_scene_manager.cpp:389, which does not run in this
        // configuration, so the flag never leaves its `= true` initialiser.
        // Reading it measured nothing. Never trust an engine-maintained flag
        // for a diagnostic without first finding the site that clears it.
        //
        // So: classify against the camera directly. `behind` counts instances
        // on the far side of the camera plane — dot(instPos - camPos, fwd) < 0.
        // That is exactly the population a rasteriser has no reason to submit
        // and a path tracer still needs, because it is what blocks the sun from
        // behind you. Crude (origin point, not bounds) but self-contained and
        // impossible to silently no-op.
        const Vector3 camFwd(v2w[2].x, v2w[2].y, v2w[2].z);
        uint32_t behind = 0, front = 0, behindFar = 0;
        for (RtInstance* pInst : m_instances) {
          if (pInst == nullptr)
            continue;
          const Matrix4& o2w = pInst->getTransform();
          const float dx = o2w[3][0] - camPos.x;
          const float dy = o2w[3][1] - camPos.y;
          const float dz = o2w[3][2] - camPos.z;
          const float dp = dx * camFwd.x + dy * camFwd.y + dz * camFwd.z;
          if (dp < 0.0f) {
            ++behind;
            // Beyond 256u behind: unambiguously not a near-camera artifact.
            if ((dx * dx + dy * dy + dz * dz) > (256.0f * 256.0f))
              ++behindFar;
          } else {
            ++front;
          }
        }

        Logger::warn(str::format(
          "[OccProbe] f=", currentFrame,
          " pitchDeg=", pitchDeg,
          " yawDeg=", yawDeg,
          " inst=", instNow,
          " front=", front,
          " behind=", behind,
          " behindFar=", behindFar));

        Logger::warn(str::format(
          "[PitchProbe] f=", currentFrame,
          " pitchDeg=", pitchDeg,
          " yawDeg=", yawDeg,
          " fwdZ=", fwdZ,
          " inst=", instNow,
          " dInst=", dInst,
          " camPos=(", camPos.x, ",", camPos.y, ",", camPos.z, ")",
          // hookN is the engine-hook main-cam capture count, on the line as a
          // PROVENANCE label, not a warning. hookN=0 is the expected steady
          // state here (RTX_DISABLE_ENGINE_HOOKS=1 is deliberate); hookN>16
          // means a run was made with the detours on. Both are valid captures.
          // It is here so the two are never silently mixed in one series --
          // the camera fields above come from a different path on each side.
          " hookN=", tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed)));

        // NV-DXVK [JobProbe]: drain the world-visibility-worker counters for
        // this frame. Emitted here, not on the job threads, for two reasons:
        // this is the only place with a frame boundary AND a pitch value, and
        // logging from a job thread would perturb the thing being counted.
        //
        // pitchDeg is repeated on this line deliberately. Every previous
        // round of this investigation was lost to correlating two separately
        // emitted series by hand; a line that carries both the independent
        // and the dependent variable cannot be mis-binned.
        //
        // READ IT: sort by pitchDeg and look at calls.
        //   calls falls with pitch  => the JOB ARRAY is being built smaller
        //     when looking down. The producer is sub_1802EB620's portal
        //     traversal / job construction (handoff §6) — go there.
        //   calls flat with pitch   => job supply is innocent, and the loss is
        //     downstream of the worker: the drain (sub_1802F04F0) or the mask
        //     consumers. Do NOT go into EB620 in that case.
        // Do not read recCntSum as geometry volume — see the option doc.
        const uint64_t jpCalls =
          tf2::g_jobProbeCalls.exchange(0, std::memory_order_relaxed);
        if (jpCalls != 0) {
          const uint64_t jpRecCnt =
            tf2::g_jobProbeRecCntSum.exchange(0, std::memory_order_relaxed);
          const uint64_t jpBad =
            tf2::g_jobProbeBadReads.exchange(0, std::memory_order_relaxed);
          const uint32_t jpLo =
            tf2::g_jobProbeJobIdxMin.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpHi =
            tf2::g_jobProbeJobIdxMax.exchange(0, std::memory_order_relaxed);

          // NV-DXVK: the three traversal-governing globals, as [lo,hi] over
          // every job this frame. See rtx_camera_manager.cpp for why these and
          // why now; the short version is that `calls` is an OUTPUT of the
          // traversal (§0.2e: the worker splits itself), so it cannot answer
          // "did something upstream supply less" — but these can, because they
          // are written before dispatch and read by every job.
          //
          // READ IT, binned by yawDeg, magnitudes not correlation:
          //   split[lo,hi] moves with yaw    => the job-split threshold IS the
          //     mechanism. Job count follows it directly. Stop here.
          //   leafSkip[lo,hi] moves with yaw => the worker is skipping leaves
          //     by a threshold nobody has ever looked at. Stop here.
          //   planes 4 -> 8 with yaw         => the extra plane blocks light
          //     up; cull sites 10c/10f guard those and are already patched, so
          //     this would mean an UNPATCHED 8-plane path exists.
          //   all three flat, lo==hi         => ANSWERED 2026-08-05: they are.
          //     160/160 frames, planes=4 leafSkip=1 split=250, identical in
          //     every yaw bin while calls went 181/115/68/48. Ruled out.
          //
          // lo != hi on any field means the value moved mid-frame across job
          // threads; the whole field is then a range, not a reading, and the
          // yaw binning below it is meaningless until that is explained.
          //
          // pool/pend are the follow-up, and they are the ones to read now.
          // sub_1802EB620 is a QUEUE LOOP over areas, not one BSP walk, and
          // sub_1802E7C70 hands out portal records from a per-frame pool of
          // 4092 64-byte blocks. When it runs out it returns -1, and
          // sub_1802E8A20 wraps its ENTIRE body in `if (result != -1)` — so a
          // failed allocation drops that area whole: no jobs, no mask bits, no
          // log, and no reject branch any [CullOff] site could patch.
          //   poolHi near 4092 in the collapsed yaw bins => that is the bug.
          //   poolHi low everywhere => exhaustion is not firing; the drop is
          //     sub_1802ED900's own -1 return at 0x2EB8D0 instead.
          //   pendHi falling with yaw => fewer areas ever QUEUED (upstream,
          //     sub_1802EAD60 at 0x2EB7E1). pendHi flat while calls falls =>
          //     areas are queued and then dropped. That is the discrimination.
          // poolHi is a peak over job-thread samples and can UNDERSTATE the true
          // peak if the last allocations land after the final job call, so
          // pinning at 4092 is conclusive but a low reading is only suggestive.
          const uint32_t jpSkLo = tf2::g_jobProbeLeafSkipLo.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpSkHi = tf2::g_jobProbeLeafSkipHi.exchange(0, std::memory_order_relaxed);
          const uint32_t jpSpLo = tf2::g_jobProbeSplitLo.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpSpHi = tf2::g_jobProbeSplitHi.exchange(0, std::memory_order_relaxed);
          const uint32_t jpPlLo = tf2::g_jobProbePlanesLo.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpPlHi = tf2::g_jobProbePlanesHi.exchange(0, std::memory_order_relaxed);
          const uint32_t jpPoLo = tf2::g_jobProbePoolLo.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpPoHi = tf2::g_jobProbePoolHi.exchange(0, std::memory_order_relaxed);
          const uint32_t jpPeLo = tf2::g_dispProbePendLo.exchange(UINT32_MAX, std::memory_order_relaxed);
          const uint32_t jpPeHi = tf2::g_dispProbePendHi.exchange(0, std::memory_order_relaxed);

          // IDA types both thresholds `dword_`, which is its default for an
          // untyped global and not a proven type. Emit the float
          // reinterpretation of the high sample too, so a threshold that is
          // really a distance or a size is legible on the first capture
          // instead of reading as a nine-digit integer.
          float skHiAsFloat = 0.0f, spHiAsFloat = 0.0f;
          std::memcpy(&skHiAsFloat, &jpSkHi, sizeof(float));
          std::memcpy(&spHiAsFloat, &jpSpHi, sizeof(float));

          Logger::warn(str::format(
            "[JobProbe] f=", currentFrame,
            " pitchDeg=", pitchDeg,
            " yawDeg=", yawDeg,
            " calls=", jpCalls,
            " recCntSum=", jpRecCnt,
            " jobIdx=[", (jpLo == UINT32_MAX ? 0u : jpLo), ",", jpHi, "]",
            " planes=[", (jpPlLo == UINT32_MAX ? 0u : jpPlLo), ",", jpPlHi, "]",
            " leafSkip=[", (jpSkLo == UINT32_MAX ? 0u : jpSkLo), ",", jpSkHi, "]",
            " split=[", (jpSpLo == UINT32_MAX ? 0u : jpSpLo), ",", jpSpHi, "]",
            " pool=[", (jpPoLo == UINT32_MAX ? 0u : jpPoLo), ",", jpPoHi, "]/4092",
            " poolPct=", (jpPoHi * 100u) / 4092u,
            " pend=[", (jpPeLo == UINT32_MAX ? 0u : jpPeLo), ",", jpPeHi, "]+1",
            " leafSkipF=", skHiAsFloat,
            " splitF=", spHiAsFloat,
            " bad=", jpBad));

          // NV-DXVK [DispProbe]: the AREA DISPATCH side, on its own line so it
          // stays legible next to the job counts above.
          //
          // READ IT against `calls` on the [JobProbe] line, binned by yawDeg:
          //   a20 falls with yaw   => fewer areas dispatched. ed900Fail and
          //     allocFail then say WHICH silent drop did it, and that is the
          //     bug — both are unpatchable rejects invisible to every other
          //     probe in this codebase.
          //   a20 flat while calls falls => area dispatch is INNOCENT. The loss
          //     is entirely inside sub_1802E8DA0's own walk, which means a 12th
          //     reject or a patched site not behaving as the table claims, and
          //     the next probe is a node-outcome census inside the worker.
          // EXPECT THE SECOND: a20 is ~1-2/frame against calls ~181, so the
          // self-split cascade outweighs area dispatch by ~100x. Recorded as a
          // prediction so a confirming result is not mistaken for a discovery.
          // allocFail is expected to be exactly 0 (the pool round peaked at
          // 36/4092); it is counted so that refutation rests on a direct
          // measurement instead of an inference from the bump pointer.
          const uint64_t dsA20 =
            tf2::g_dispProbeA20Calls.exchange(0, std::memory_order_relaxed);
          const uint64_t dsAlc =
            tf2::g_dispProbeAllocCalls.exchange(0, std::memory_order_relaxed);
          const uint64_t dsAlcF =
            tf2::g_dispProbeAllocFail.exchange(0, std::memory_order_relaxed);
          if ((dsA20 | dsAlc) != 0) {
            // areasDropped is DERIVED, not hooked. An area that ED900 rejects
            // at 0x2EB8D0 never reaches sub_1802E8A20 at 0x2EB910, so
            //   dropped = queueDepth - a20 = (pendHi + 1) - a20
            // Hooking sub_1802ED900 to measure this directly is what froze the
            // game on 2026-08-05: it does not decompile, and a C wrapper built
            // on IDA's guessed signature clobbers the xmm registers a SIMD
            // plane-builder needs. Deriving it costs nothing and cannot hang.
            const uint64_t queueDepth =
              (jpPeHi == 0 && jpPeLo == UINT32_MAX) ? 0ull : (uint64_t(jpPeHi) + 1ull);
            const int64_t dropped = int64_t(queueDepth) - int64_t(dsA20);
            // ed900 = entries to sub_1802ED900, from the counter-only island.
            // ed900Inst=0 means the hook did NOT install, and then ed900=0 is
            // the probe being ABSENT, not the function being uncalled — the
            // two are numerically identical, which is exactly the defect that
            // made [OccProbe] v1 report outFr=0 on 459 frames. Check the flag
            // before reading the count.
            const uint64_t dsEd9 =
              tf2::g_ed900ProbeCalls.exchange(0, std::memory_order_relaxed);
            const uint32_t dsEd9Inst =
              tf2::g_ed900ProbeInstalled.load(std::memory_order_relaxed);
            // eb620 = per-view area-builder invocations. areasPerView =
            // a20/eb620 is the number that decides the open fork: a20 is a
            // per-FRAME sum across views, so a fall in it means either fewer
            // views doing work or fewer areas in each. drains=4 counts
            // sub_1802F04F0, not EB620, so this has never been measured.
            const uint64_t dsEb6 =
              tf2::g_eb620ProbeCalls.exchange(0, std::memory_order_relaxed);
            // Which neighbour areas the abandoned crossings would have reached.
            // 0xffff means the portal table could not be resolved for that bit.
            std::string degenAreas;
            const uint32_t degenN =
              tf2::g_clipDegenAreaN.exchange(0, std::memory_order_relaxed);
            for (uint32_t i = 0; i < degenN && i < 16u; ++i)
              degenAreas += str::format((i == 0 ? " degenTo=[" : ","),
                tf2::g_clipDegenAreas[i].load(std::memory_order_relaxed));
            if (!degenAreas.empty())
              degenAreas += "]";
            // [DegenPair]: only the NON-EMPTY cells, printed raw as
            // r11/r9:count. Nothing is summed, averaged or thresholded here —
            // the whole point of the pair is that a total cannot separate the
            // safe count from the lethal one, so a total would throw away the
            // measurement on the way to the log. r9=15 is the clamp bucket
            // ("15 or more"); r11=3 must never appear, and if it does the stub
            // is reading the wrong register.
            std::string degenPair;
            uint32_t degenPairTotal = 0;
            for (uint32_t cell = 0; cell < 64u; ++cell) {
              const uint32_t c =
                tf2::g_degenPair[cell].exchange(0, std::memory_order_relaxed);
              if (c == 0)
                continue;
              degenPairTotal += c;
              degenPair += str::format((degenPair.empty() ? " degenPair=[" : ","),
                                       cell & 3u,                 // r11 bucket
                                       "/", cell >> 2,            // r9 bucket
                                       ((cell >> 2) == 15u ? "+" : ""),
                                       ":", c);
            }
            if (!degenPair.empty())
              degenPair += str::format("] degenPairN=", degenPairTotal);
            // [QueueProbe] — the two halves of "why were only 9 of the 30
            // seeded areas dispatched". qStart is sub_1802EAD60's RETURN value
            // (the order-list index the queue loop starts at); qSkipAreas are
            // the areas the loop DID reach and skipped because their selector
            // was still -1. Emitted on this line, not a new one, so it bins
            // against a20 and fc000 with no extra plumbing.
            //   qStart moves across the y~-9984 step  => cursor is the gate.
            //   qStart flat + 113/141/148/149/153 in qSkipAreas => cursor is
            //     innocent and no portal crossing wrote them a selector.
            // qStartInst guards the zero: an uninstalled island and a never-
            // called one both read 0, and qStart legitimately CAN be 0.
            // [DropAreas] — the areas sub_1802ED900 rejected with -1.
            const uint32_t dropAreaN =
              tf2::g_dropAreaN.exchange(0, std::memory_order_relaxed);
            std::string dropAreas;
            for (uint32_t i = 0; i < dropAreaN && i < 48u; ++i)
              dropAreas += str::format((i == 0 ? " dropAreas=[" : ","),
                tf2::g_dropAreas[i].load(std::memory_order_relaxed));
            if (!dropAreas.empty())
              dropAreas += "]";

            const uint32_t qInst =
              tf2::g_queueProbeInstalled.load(std::memory_order_relaxed);
            const uint32_t qStart = tf2::g_qStart.load(std::memory_order_relaxed);
            const uint64_t qStartN =
              tf2::g_qStartCalls.exchange(0, std::memory_order_relaxed);
            const uint64_t qSkipN =
              tf2::g_qSkipCalls.exchange(0, std::memory_order_relaxed);
            const uint32_t qSkipAreaN =
              tf2::g_qSkipAreaN.exchange(0, std::memory_order_relaxed);
            std::string qSkipAreas;
            for (uint32_t i = 0; i < qSkipAreaN && i < 32u; ++i)
              qSkipAreas += str::format((i == 0 ? " qSkipAreas=[" : ","),
                tf2::g_qSkipAreas[i].load(std::memory_order_relaxed));
            if (!qSkipAreas.empty())
              qSkipAreas += "]";
            // [FaceReject]: the camera-behind-the-portal skip at 0x2EB98F,
            // by target area. Read faceAreas AGAINST qSkipAreas on this same
            // line — the same ids in both, on the LOW side only, is the whole
            // chain: plane crossed -> portal skipped -> no selector -> queue
            // loop skips the area -> a20 14->9 -> m1 and m2 both halve.
            const uint64_t faceRej =
              tf2::g_faceRejectCount.exchange(0, std::memory_order_relaxed);
            const uint32_t faceAreaN =
              tf2::g_faceRejectAreaN.exchange(0, std::memory_order_relaxed);
            std::string faceAreas;
            for (uint32_t i = 0; i < faceAreaN && i < 32u; ++i)
              faceAreas += str::format((i == 0 ? " faceAreas=[" : ","),
                tf2::g_faceRejectAreas[i].load(std::memory_order_relaxed));
            if (!faceAreas.empty())
              faceAreas += "]";
            // [PortalWalk]: the superset — every portal target the flood
            // iterated. An id in qSkipAreas but ABSENT here was never reached
            // by any crossing, so no reject is at fault and the loss is a
            // cascade from whichever source area died first.
            const uint64_t walkN =
              tf2::g_portalWalkCount.exchange(0, std::memory_order_relaxed);
            const uint32_t walkAreaN =
              tf2::g_portalWalkAreaN.exchange(0, std::memory_order_relaxed);
            // src>dst, packed src<<16|dst by the drain. Printed as pairs
            // because the owner of a portal is the whole question now: the
            // cluster 81/84/92/96/113/141/146/148/152/153/166 moves as one
            // unit, so exactly one src>dst edge into it should disappear
            // between the HIGH and LOW states, and that edge names the area to
            // chase next. src=65535 means the per-area range table did not
            // cover the portal — a resolution hole, not an area.
            std::string walkAreas;
            for (uint32_t i = 0; i < walkAreaN && i < 48u; ++i) {
              const uint32_t v =
                tf2::g_portalWalkAreas[i].load(std::memory_order_relaxed);
              walkAreas += str::format((i == 0 ? " walkPairs=[" : ","),
                                       (v >> 16), ">", (v & 0xFFFFu));
            }
            if (!walkAreas.empty())
              walkAreas += "]";
            // [SelWrite]: areas that actually received a selector. THE line to
            // read against walkPairs — 149 and 124 here in LOW means enqueued
            // and then abandoned by the forward cursor (ordering, and rewinds=
            // says whether the engine's own recovery fired at all); absent
            // means a reject upstream of 0x2EC739 that nothing counts yet.
            const uint64_t selN =
              tf2::g_selWriteCount.exchange(0, std::memory_order_relaxed);
            const uint32_t selAreaN =
              tf2::g_selWriteAreaN.exchange(0, std::memory_order_relaxed);
            std::string selAreas;
            for (uint32_t i = 0; i < selAreaN && i < 48u; ++i)
              selAreas += str::format((i == 0 ? " selAreas=[" : ","),
                tf2::g_selWriteAreas[i].load(std::memory_order_relaxed));
            if (!selAreas.empty())
              selAreas += "]";
            Logger::warn(str::format(
              "[DispProbe] f=", currentFrame,
              // Labels EVERY frame with the A/B mode so the log can just be
              // grouped by it, instead of segmenting by timestamp around the
              // toggle. 0 configured / 1 area sites off / 2 all off.
              " abMode=", tf2::g_cullOffAbMode.load(std::memory_order_relaxed),
              " pitchDeg=", pitchDeg,
              " yawDeg=", yawDeg,
              " a20=", dsA20,
              " queueDepth=", queueDepth,
              " areasDropped=", dropped,
              " alloc=", dsAlc,
              " allocFail=", dsAlcF,
              " ed900=", dsEd9,
              " ed900Inst=", dsEd9Inst,
              // THE VERDICT, not the call count — areas killed by ED900's -1 at
              // 0x2EB8D0, which drops before both the dispatch (0x2EB910) and
              // the portal loop (0x2EB915). Compare against a20 on this same
              // line: rising as a20 falls means this is the gate. ed900 sitting
              // flat at ~1.0 never said anything about this.
              " ed900Drop=", tf2::g_ed900DropCount.exchange(0, std::memory_order_relaxed),
              // [DropAreas]: WHICH areas that -1 dropped. The count above has
              // read 4.00 flat on BOTH sides of the step and was treated as an
              // acquittal on exactly that basis. If these ids are
              // 124/141/148/149/153 on the LOW side and a different four on the
              // HIGH side, the constant count was concealing the whole defect —
              // the same count-for-identity substitution that already cost
              // listLen and the 16-capped seed list.
              dropAreas,
              " ed900DropInst=", tf2::g_ed900DropInstalled.load(std::memory_order_relaxed),
              // Portals abandoned because the clip produced <3 verts. The only
              // rejects still unpatched on this path, and the suspects for the
              // residual PITCH collapse (a20 32.67 at +10deg -> 6.42 at -50deg
              // while yaw is flat). Bin against pitchDeg on this same line.
              " clipDegen=", tf2::g_clipDegenA.exchange(0, std::memory_order_relaxed),
              "/", tf2::g_clipDegenB.exchange(0, std::memory_order_relaxed),
              " clipDegenInst=", tf2::g_clipDegenInstalled.load(std::memory_order_relaxed),
              // Aliasing guard for degenTo — see g_clipDegenMaxBit. Near 16383
              // means the area ids on this line are not to be believed.
              " degenMaxBit=", tf2::g_clipDegenMaxBit.load(std::memory_order_relaxed),
              degenAreas,
              // WHICH count is degenerate, r11/r9, for every reject at
              // 0x2EC675. Decides whether relaxing only `cmp r11,3` can do
              // anything: r9 >= 3 cells would clear the untouched `cmp r9,3`
              // and enqueue their neighbour; r9 < 3 cells would just die four
              // bytes further down and the fix belongs upstream instead.
              degenPair,
              " degenPairInst=",
              tf2::g_degenPairInstalled.load(std::memory_order_relaxed),
              " eb620=", dsEb6,
              // [QueueProbe]. qStart is LATCHED (last EB620 invocation of the
              // frame), not summed — it is an index, and summing an index is
              // meaningless. qStartN/qSkipN are the island call counts and
              // exist so a silent island failure is visible as 0 calls rather
              // than as a plausible-looking index.
              " qStart=", qStart,
              " qStartN=", qStartN,
              " qSkipN=", qSkipN,
              qSkipAreas,
              " qInst=", (qInst ? 1 : 0),
              // [FaceReject]. faceRej is the raw skip count; faceAreas are the
              // target areas those skipped portals led to.
              " faceRej=", faceRej,
              faceAreas,
              " faceInst=",
              tf2::g_faceRejectInstalled.load(std::memory_order_relaxed),
              // [PortalWalk]. walkN is the raw portal-iteration count and is
              // an OUTPUT of how much territory the flood covered — the same
              // trap faceRej and [JobProbe] calls fall into, so it decides
              // nothing on its own. walkAreas is the measurement.
              " walkN=", walkN,
              walkAreas,
              // Portals whose target was the no-neighbour sentinel (nAreas+1)
              // or unresolved. They are excluded from walkAreas so real ids
              // are not crowded out, but they are counted here so a short id
              // list is never read as a small walk.
              " walkOob=", tf2::g_portalWalkOob.load(std::memory_order_relaxed),
              " walkInst=",
              tf2::g_portalWalkInstalled.load(std::memory_order_relaxed),
              // [SelWrite]. selN is the raw store count; selAreas are the
              // areas enqueued. rewinds is the PER-FRAME DELTA of the
              // engine's own cursor-rewind counter (dword_1811FBD98):
              // 4294967295 means the global could not be read, which is not
              // the same as zero rewinds.
              " selN=", selN,
              selAreas,
              " rewinds=", tf2::g_rewinds.load(std::memory_order_relaxed),
              " selInst=",
              tf2::g_selWriteInstalled.load(std::memory_order_relaxed),
              // RAW, undecorated. Compare against camPos on the [PitchProbe]
              // line of the SAME frame: equal (in some frame) => camera origin
              // => sub_1802EAD60 is position-only and stays eliminated.
              // Varying with yawDeg while camPos is fixed => the identity was
              // wrong and 0x2EB0F1 is the reject to patch.
              " fc000=(", tf2::g_dispProbeFc000X.load(std::memory_order_relaxed),
              ",", tf2::g_dispProbeFc000Y.load(std::memory_order_relaxed),
              ",", tf2::g_dispProbeFc000Z.load(std::memory_order_relaxed),
              ",", tf2::g_dispProbeFc000W.load(std::memory_order_relaxed), ")"));

          // NV-DXVK [AreaSeed]: what sub_1802EAD60 SEEDED, one line per frame.
          //
          // This is the layer above [AreaDump]. AreaDump says which areas were
          // dispatched; this says which ones were ever CANDIDATES. The queue
          // loop at 0x2EB864 iterates the order list, so an area missing from
          // listLen can never dispatch no matter what its selector is — which
          // is why rtx.cullOff.areaPortal raised a20 5 -> 23 at low yaw and
          // changed nothing at all at 155-167deg.
          //
          // READ listLen BINNED BY YAW, against the same frame's a20:
          //   listLen falls with yaw  => EAD60 is view-dependent, its 0x2EB0F1
          //     crossing test is the gate, and org says why.
          //   listLen flat, a20 falls => EAD60 is genuinely exonerated and the
          //     loss is between the order list and the selector.
          //
          // org is the PRE-TRANSFORM fc000, read from EF090's a2 argument. It is
          // NOT the fc000 on the [DispProbe] line: that one is sampled after
          // EB620's tail overwrites the global at 0x2ECB2F, so the two differing
          // is expected and is itself the confirmation. Compare org against
          // camPos on the [PitchProbe] line of the same frame.
          const uint32_t asInst =
            tf2::g_areaSeedInstalled.load(std::memory_order_relaxed);
          const uint64_t asCalls =
            tf2::g_areaSeedCalls.exchange(0, std::memory_order_relaxed);
          {
            std::string seedAreas;
            const uint32_t seedN = tf2::g_areaSeedN.load(std::memory_order_relaxed);
            // 48, not 16. listLen is 30 on this map, so the old cap printed 16
            // of 30 and SILENTLY HID THE OTHER 14 — and "areas 149/124/113/141/
            // 148/153 are not in the seed list" was read off that truncated
            // line. Identical defect to the walkAreas one: a capped identity
            // list manufactures absences. Never cap an identity list below the
            // length field printed next to it.
            for (uint32_t i = 0; i < seedN && i < 48u; ++i)
              seedAreas += str::format((i == 0 ? " areas=[" : ","),
                tf2::g_areaSeedAreas[i].load(std::memory_order_relaxed));
            if (!seedAreas.empty())
              seedAreas += "]";
            std::string liveAreas;
            const uint32_t liveN = tf2::g_areaSeedLiveN.load(std::memory_order_relaxed);
            for (uint32_t i = 0; i < liveN && i < 48u; ++i)
              liveAreas += str::format((i == 0 ? " liveAreas=[" : ","),
                tf2::g_areaSeedLiveAreas[i].load(std::memory_order_relaxed));
            if (!liveAreas.empty())
              liveAreas += "]";
            Logger::warn(str::format(
              "[AreaSeed] f=", currentFrame,
              // pitch added 2026-08-05: sites 12+13 removed the yaw dependence
              // and a PITCH sweep still collapses, so the axis this has to be
              // binned on changed. Was yaw-only, which meant every pitch answer
              // had to be recovered by joining to [DispProbe] on the frame id.
              " pitchDeg=", pitchDeg,
              " yawDeg=", yawDeg,
              " ef090=", asCalls,
              " ef090Inst=", asInst,
              " nAreas=", tf2::g_areaSeedNAreas.load(std::memory_order_relaxed),
              " listLen=", tf2::g_areaSeedListLen.load(std::memory_order_relaxed),
              " org=(", tf2::g_areaSeedOrgX.load(std::memory_order_relaxed),
              ",", tf2::g_areaSeedOrgY.load(std::memory_order_relaxed),
              ",", tf2::g_areaSeedOrgZ.load(std::memory_order_relaxed),
              ",", tf2::g_areaSeedOrgW.load(std::memory_order_relaxed), ")",
              // Enqueued but never visited, and the pending count at exit.
              // live naming 113/141/148/153 in the well => the queue loop lost
              // them; live=0 => the crossing into them never happened.
              " live=", tf2::g_areaSeedLive.load(std::memory_order_relaxed),
              " pendExit=", tf2::g_areaSeedPending.load(std::memory_order_relaxed),
              liveAreas,
              seedAreas));
          }

          // NV-DXVK [AreaDump]: WHICH areas dispatched this frame, one line.
          // node= is sub_1802E8A20's a3, the seed BSP node (indexes
          // qword_181748D58 at stride 32) — the area's identity. rec= is the
          // portal record selector, bucket= is a1.
          //
          // READ IT by DIFFING the node set between a low-yaw frame (5 areas)
          // and a high-yaw frame (2 areas). The three nodes that disappear are
          // the answer; "what is different about those three" is then a
          // question about data, not about my reading of the disassembly.
          // Six gates on this path have each measured constant across yaw, so
          // a belief is wrong somewhere and counting more will not find it.
          const uint32_t slotN =
            tf2::g_dispProbeSlotN.exchange(0, std::memory_order_relaxed);
          if (slotN != 0) {
            std::string areas;
            const uint32_t shown = slotN < 32u ? slotN : 32u;
            for (uint32_t i = 0; i < shown; ++i) {
              areas += str::format(
                " [", i, "] node=", tf2::g_dispProbeSlotA3[i].load(std::memory_order_relaxed),
                " rec=", tf2::g_dispProbeSlotA2[i].load(std::memory_order_relaxed),
                " bucket=", tf2::g_dispProbeSlotA1[i].load(std::memory_order_relaxed),
                // from= is the CALLER's client.dll RVA. Expect 0x2eb915 (the
                // fc000-gated queue loop, position-only) or 0x2ec93c (the
                // second dispatch, whose region reads the frustum side planes
                // and was never read). If nodes 124/178 come from one and
                // 125/127/149 from the other, that is the whole answer.
                " from=0x", std::hex,
                tf2::g_dispProbeSlotRA[i].load(std::memory_order_relaxed), std::dec);
            }
            Logger::warn(str::format(
              "[AreaDump] f=", currentFrame,
              " pitchDeg=", pitchDeg,
              " yawDeg=", yawDeg,
              " n=", slotN,
              (slotN > 32u ? " (TRUNCATED to 32)" : ""),
              areas));
          }

          // [Ed480Probe]: the DYNAMIC enqueue. ed480 is the call count;
          // areas= are its a1 arguments, i.e. which areas were enqueued by the
          // job path rather than by position-only sub_1802EAD60.
          //   ed480 falls 3 -> 0 with yaw  => confirmed: the three
          //     view-dependent areas come from here, and the one-shot
          //     [Ed480Probe] FIRST CALL backtrace names the caller whose
          //     visibility test is the real fix site.
          //   ed480 flat or 0 => the dynamic enqueue is NOT the source either,
          //     and the three areas are entering through a path not yet found.
          const uint64_t dsEd480 =
            tf2::g_dispProbeEd480Calls.exchange(0, std::memory_order_relaxed);
          const uint32_t ed480N =
            tf2::g_dispProbeEd480N.exchange(0, std::memory_order_relaxed);
          if (dsEd480 != 0) {
            std::string eAreas;
            const uint32_t shown = ed480N < 8u ? ed480N : 8u;
            for (uint32_t i = 0; i < shown; ++i)
              eAreas += str::format(" ", tf2::g_dispProbeEd480Area[i].load(std::memory_order_relaxed));
            Logger::warn(str::format(
              "[Ed480Probe] f=", currentFrame,
              " yawDeg=", yawDeg,
              " ed480=", dsEd480,
              " areas=", eAreas));
          }
          }
        }

        // NV-DXVK [DrainProbe]: the matching OUTPUT line. Same per-frame
        // drain-and-reset, same repeated pitchDeg so it bins by itself.
        // m1Max/m2Max are carried alongside the sums because the drain runs
        // once per view — the main view is the largest, and a sum alone
        // cannot distinguish it shrinking from a sub-view disappearing.
        // Check layoutOk=1 before interpreting anything on this line.
        const uint64_t dpCalls =
          tf2::g_drainProbeCalls.exchange(0, std::memory_order_relaxed);
        if (dpCalls != 0) {
          const uint64_t dpM1 = tf2::g_drainProbeM1Sum.exchange(0, std::memory_order_relaxed);
          const uint64_t dpM2 = tf2::g_drainProbeM2Sum.exchange(0, std::memory_order_relaxed);
          const uint64_t dpR  = tf2::g_drainProbeRSum.exchange(0, std::memory_order_relaxed);
          const uint32_t dpM1Max = tf2::g_drainProbeM1Max.exchange(0, std::memory_order_relaxed);
          const uint32_t dpM2Max = tf2::g_drainProbeM2Max.exchange(0, std::memory_order_relaxed);
          const uint64_t dpBad = tf2::g_drainProbeBad.exchange(0, std::memory_order_relaxed);

          Logger::warn(str::format(
            "[DrainProbe] f=", currentFrame,
            " pitchDeg=", pitchDeg,
            " yawDeg=", yawDeg,
            " drains=", dpCalls,
            " m1=", dpM1, " m2=", dpM2, " r=", dpR,
            " m1Max=", dpM1Max, " m2Max=", dpM2Max,
            " layoutOk=", tf2::g_drainProbeLayoutOk.load(std::memory_order_relaxed),
            " c70=", tf2::g_drainProbeC70.load(std::memory_order_relaxed),
            " c74=", tf2::g_drainProbeC74.load(std::memory_order_relaxed),
            " c78=", tf2::g_drainProbeC78.load(std::memory_order_relaxed),
            " bad=", dpBad));
        }
      }
    }

    // Need to release all instances when ViewModel enablement changes
    // This is a big hammer but it's fine, it's a debugging feature
    const bool isViewModelEnabled = RtxOptions::ViewModel::enable();
    if (isViewModelEnabled != m_previousViewModelState) {
      clear();
      m_previousViewModelState = isViewModelEnabled;
    }

    // NV-DXVK [SubViewVsCensus]: per-VS bucket counts across ALL
    // sub-view RtInstances (every instance with IgnoreAntiCulling set
    // — that flag is exclusively applied by SetSkyCategoryFromCb2's
    // reproject branch, so it's a clean filter for "sub-view content").
    // The mountain investigation initially scoped to VS_2904d2, but
    // the PropIdTrace log showed 7 distinct VS hashes participating
    // in the sub-view pass:
    //   VS_1baf78e08c4e8fed  (~1933 reproject hits)
    //   VS_2094e03a7a19c026  (~1304)
    //   VS_2f543cd750faaf2d  (~512)
    //   VS_eda5efc125a8ed9c  (~163 — sky dome)
    //   VS_aa5c8f7e8788e1d2  (~17)
    //   VS_c10aa132da51c65b  (~5)
    //   VS_1ddf42076ed0b6d1  (~2)
    //   VS_2904d2163ef31a17  (the prop-mountain shader)
    // The user's "missing mountains" may be in any of these — distant
    // terrain often renders under a different VS than near props.
    // This probe counts INSTANCES per VS hash, so a sudden drop in
    // any bucket is a candidate for the missing-mountain VS.
    if constexpr (kEnableGcCensus) {
      static thread_local uint32_t sSubViewVsLastFrame = UINT32_MAX;
      if (sSubViewVsLastFrame != currentFrame) {
        sSubViewVsLastFrame = currentFrame;
        std::unordered_map<uint64_t, uint32_t> byVsTotal;
        std::unordered_map<uint64_t, uint32_t> byVsHidden;
        std::unordered_map<uint64_t, uint32_t> byVsOutFr;
        for (const RtInstance* pInst : m_instances) {
          if (pInst == nullptr) continue;
          if (!pInst->testCategoryFlags(InstanceCategories::IgnoreAntiCulling)) continue;
          // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
          // see that. Null the local so the checks below work. [ReapBlasUAF]
          const BlasEntry* pBl = pInst->isUnlinkedForGC() ? nullptr : pInst->getBlas();
          if (pBl == nullptr) continue;
          const uint64_t h = static_cast<uint64_t>(
            pBl->input.getTransformData().vertexShaderHash);
          byVsTotal[h] += 1;
          if (pInst->m_isHidden)        byVsHidden[h] += 1;
          if (!pInst->m_isInsideFrustum) byVsOutFr[h] += 1;
        }
        for (const auto& kv : byVsTotal) {
          Logger::info(str::format(
            "[SubViewVsCensus] f=", currentFrame,
            " vs=0x", std::hex, kv.first, std::dec,
            " total=", kv.second,
            " hidden=", byVsHidden[kv.first],
            " notInFr=", byVsOutFr[kv.first]));
        }
      }
    }

    // NV-DXVK [MtnCensus]: per-frame visibility census of ALL VS_2904d2
    // mountain instances. The user reports that some mountains are
    // missing while others are visible, in a way that doesn't match
    // simple frustum culling (some missing ones stay missing). The
    // existing per-instance probes are rate-limited to 5/frame, so
    // they only sample ~10% of the 48 mountain population — not enough
    // to spot a consistent missing subset.
    //
    // This probe iterates the FULL m_instances list once per frame,
    // counts every VS_2904d2 instance by visibility-state bucket, and
    // dumps a summary line. Every 60 frames it ALSO dumps each
    // instance's per-instance state so we can see which specific
    // mountains (by propId) are stuck in a non-visible state across
    // many frames. Correlate propId persistence with what the user
    // sees on screen: a propId that's `hidden=1` or `inFrustum=0` in
    // every dump but visually "should be visible" tells us which
    // pipeline stage is dropping it.
    //
    // Cost: O(N) loop over m_instances, ~3000 entries × cheap field
    // reads — negligible vs the GC pass's own loop next.
    //
    // Buckets:
    //   total           — every VS_2904d2 instance
    //   hidden          — m_isHidden==true (sky-classified or
    //                     replacement-asset hidden)
    //   notInFrustum    — m_isInsideFrustum==false (anti-culling-dedup
    //                     decided it's outside the main-cam frustum)
    //   markedGC        — m_isMarkedForGC==true (queued for removal
    //                     this pass)
    //   ignAntiCull     — has InstanceCategories::IgnoreAntiCulling
    //                     category set (sub-view sticky-preserve)
    //   staleN          — m_frameLastUpdated < currentFrame (not
    //                     touched THIS frame). N is the gap.
    if constexpr (kEnableGcCensus) {
      static thread_local uint32_t sCensusLastDumpFrame = UINT32_MAX;
      const bool dumpPerInstance =
        (currentFrame % 60u == 0u) && (sCensusLastDumpFrame != currentFrame);

      uint32_t total       = 0;
      uint32_t hidden      = 0;
      uint32_t notInFr     = 0;
      uint32_t markedGC    = 0;
      uint32_t ignAntiCull = 0;
      uint32_t staleAny    = 0;   // updated < currentFrame
      uint32_t staleMany   = 0;   // gap > 1

      uint32_t perInstIdx = 0;
      for (const RtInstance* pInst : m_instances) {
        if (pInst == nullptr) continue;
        // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
        // see that. Null the local so the checks below work. [ReapBlasUAF]
        const BlasEntry* pBlasCensus = pInst->isUnlinkedForGC() ? nullptr : pInst->getBlas();
        if (pBlasCensus == nullptr) continue;
        if (pBlasCensus->input.getTransformData().vertexShaderHash != 0x2904d2163ef31a17ull) continue;

        total += 1;
        const bool isHidden_   = pInst->m_isHidden;
        const bool inFr_       = pInst->m_isInsideFrustum;
        const bool gcMark_     = pInst->m_isMarkedForGC;
        const bool ignAC_      = pInst->testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
        const uint32_t gap     = (currentFrame > pInst->m_frameLastUpdated)
                                 ? (currentFrame - pInst->m_frameLastUpdated) : 0u;
        if (isHidden_) hidden      += 1;
        if (!inFr_)    notInFr     += 1;
        if (gcMark_)   markedGC    += 1;
        if (ignAC_)    ignAntiCull += 1;
        if (gap > 0u)  staleAny    += 1;
        if (gap > 1u)  staleMany   += 1;

        if (dumpPerInstance) {
          // Pull o2w.translation directly from the instance's current
          // worldToObject inverse — that's the TLAS-bound position.
          // If reproject worked, |t.z| should be large (>10000); if
          // the multi-shader-variant fanout clobbered it with raw
          // sub-view-local data, |t.z| stays ~15600 (the sky-anchor)
          // and the mountain renders below the visible world.
          const Matrix4& o2w = pInst->getTransform();
          const float tx = float(o2w[3][0]);
          const float ty = float(o2w[3][1]);
          const float tz = float(o2w[3][2]);
          // Reproject sanity flag: a correctly-reprojected sub-view
          // mountain ends up many thousands of units from origin in
          // main-world space (the sub-view layout × scale). Raw
          // sub-view-local coords are near the engine-sky-anchor
          // (small XY, Z near -15616). If |tz| < 20000 the o2w is
          // suspiciously close to the anchor — flag for inspection.
          const bool tzLooksRaw = (std::abs(tz) < 20000.0f);
          // NV-DXVK [renderability triad]: the user reports SOME mountain
          // instances render correctly and others are entirely absent.
          // All 22 instances exist (not hidden / GC'd), so the missing
          // ones fail BETWEEN instance and screen. Log the three gates
          // that decide it: VkInstance mask (0 => no ray can hit it),
          // BLAS primitive count (0 => BLAS not built / empty geometry),
          // and surfaceIndex (kInvalid => not bound into the surface
          // table). Whichever differs between present and absent
          // mountains is the actual cause.
          const uint32_t mtnMask = pInst->getVkInstance().mask;
          uint32_t mtnBlasPrim = 0u;
          if (pBlasCensus != nullptr && !pBlasCensus->buildRanges.empty()) {
            mtnBlasPrim = pBlasCensus->buildRanges[0].primitiveCount;
          }
          const uint32_t mtnVtx = (pBlasCensus != nullptr)
            ? pBlasCensus->modifiedGeometryData.vertexCount : 0u;
          Logger::info(str::format(
            "[MtnCensus.Inst] f=", currentFrame,
            " #", perInstIdx,
            " propId=0x", std::hex, pInst->m_stablePropId, std::dec,
            " hidden=", (isHidden_ ? 1 : 0),
            " inFrustum=", (inFr_ ? 1 : 0),
            " ignAntiCull=", (ignAC_ ? 1 : 0),
            " markedGC=", (gcMark_ ? 1 : 0),
            " lastUpdGap=", gap,
            " mask=0x", std::hex, mtnMask, std::dec,
            " blasPrim=", mtnBlasPrim,
            " vtx=", mtnVtx,
            " surfIdx=", pInst->getSurfaceIndex(),
            " category=0x", std::hex,
              static_cast<uint64_t>(pInst->getCategoryFlags().raw()),
              std::dec,
            " o2w.t=(", tx, ",", ty, ",", tz, ")",
            " tzLooksRaw=", (tzLooksRaw ? 1 : 0)));
          perInstIdx += 1;
        }
      }

      // Summary line every frame — cheap to skim.
      if (total > 0u) {
        Logger::info(str::format(
          "[MtnCensus] f=", currentFrame,
          " total=", total,
          " hidden=", hidden,
          " notInFrustum=", notInFr,
          " markedGC=", markedGC,
          " ignAntiCull=", ignAntiCull,
          " staleAny=", staleAny,
          " staleMany=", staleMany));
      }
      if (dumpPerInstance) sCensusLastDumpFrame = currentFrame;
    }

    // NV-DXVK [HullCensus]: the "vanishing ship" is a PRIMARY-VISIBILITY problem — it
    // disappears in the raw Diffuse Albedo debug view, which is written in the GBuffer pass
    // BEFORE path tracing and BEFORE the denoiser. So the denoiser/confidence/MV chase was the
    // wrong layer. A view-direction-gated, per-frame, instant G-buffer vanish means the hull is
    // either (a) dropped from the TLAS as an instance, or (b) in the TLAS but its triangles are
    // backface-culled at ray time depending on view angle. This census discriminates the two,
    // per frame, for the Widow hull's two vertex shaders. Correlate by frame with [ShipDir] f=N
    // (camera forward) to see which mechanism flips with the good/bad look direction.
    //
    //   present-but-vanished  -> mask!=0 & inFrustum & !hidden & blasPrim>0 & surfIdx valid
    //                            => ray-time BACKFACE cull (FLIP/CULLDISABLE decision wrong).
    //   instance-level cull   -> hidden=1 / inFrustum=0 / mask=0 / surfIdx invalid
    //                            => frustum/categorization/GC dropping it (wrong-camera suspect).
    //
    // VS aggregates are useless here (these VS hashes are shared with sky+floor); this iterates
    // INSTANCES, and each hull instance's world position (o2w.t) lets us track a specific piece
    // across frames regardless of the shared shader.
    if constexpr (kEnableGcCensus) {
      // NV-DXVK [StudioModelHook] re-gate: census now keys on the Widow engine
      // model (pBl->input.isWidowModel) — the shared VS-hash constants are gone.
      uint32_t hTotal = 0u, hHidden = 0u, hNotFr = 0u, hMaskZero = 0u, hSurfBad = 0u, hCullActive = 0u;
      uint32_t hIdx = 0u;
      // NV-DXVK [DropTrace]: dropship-specific per-frame aggregation, paired
      // with the submitDrawState arrival counter (g_dropTrace*, declared at
      // namespace scope above, defined in rtx_scene_manager.cpp).
      uint32_t dropInst = 0u, dropPresent = 0u, dropMaxBlas = 0u;
      // [RiddenTrace] SESSION-J: the RIDDEN ship = widow/Crow submeshes at o2w.t≈origin
      // (you're inside it; Remix recenters the world at the camera, so it lands at the
      // origin — the formation ship is 1000+ units away). One self-contained line per GC
      // walk so localization needs NO cross-line frame matching (the f= counter aliases
      // under heavy logging). It tells exactly where the ship drops:
      //   blasPrim0>0            -> geometry->BLAS dropped it (empty BLAS)
      //   hidden>0 / maskZero>0  -> excluded from the TLAS (won't be ray-traced)
      //   notInFrustum>0         -> Remix frustum-culled
      //   renderable>0 yet still invisible on screen -> it IS in the TLAS => the vanish
      //                            is shading/a pass/denoiser, NOT geometry.
      uint32_t rN = 0u, rRender = 0u, rHidden = 0u, rMaskZero = 0u, rNotFr = 0u, rBlas0 = 0u, fN = 0u;
      uint32_t rMinBlas = 0xFFFFFFFFu, rMaxBlas = 0u;
      for (const RtInstance* pInst : m_instances) {
        if (pInst == nullptr) continue;
        // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
        // see that. Null the local so the checks below work. [ReapBlasUAF]
        const BlasEntry* pBl = pInst->isUnlinkedForGC() ? nullptr : pInst->getBlas();
        if (pBl == nullptr) continue;
        const uint64_t vs = static_cast<uint64_t>(pBl->input.getTransformData().vertexShaderHash);
        // NV-DXVK [HullCensus] re-widen: the VISIBLE ship the user stands in is
        // VS=0x292b / name=(none) — WORLD geometry, NOT the studio Widow — and
        // it vanishes by dropping out of the TLAS (box rays hit the 3D skybox,
        // [ShipBox] meanViewZ jumps 1->200000 svSky=1). Census ALL 0x292b
        // instances (plus any widow-tagged) with name/mat/tex per line so a
        // good-vs-vanished diff shows which instance drops out and how
        // (hidden/mask/inFrustum/markedGC). Capped per frame to bound volume.
        if (vs != 0x292b6ba0d1854f28ull && !pBl->input.isWidowModel) continue;
        if (hIdx >= 120u) break;

        const bool isHidden_ = pInst->m_isHidden;
        const bool inFr_     = pInst->m_isInsideFrustum;
        const bool gcMark_   = pInst->m_isMarkedForGC;
        const uint32_t mask  = pInst->getVkInstance().mask;
        const uint32_t iflags= pInst->getVkInstance().flags;  // VkGeometryInstanceFlagBitsKHR
        const bool flipFace  = (iflags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR) != 0u;
        const bool cullOff   = (iflags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR) != 0u;
        const bool cullActive= !cullOff;  // if true, back-facing tris ARE culled at ray time -> directional vanish possible
        uint32_t blasPrim = 0u;
        if (!pBl->buildRanges.empty()) blasPrim = pBl->buildRanges[0].primitiveCount;
        // NV-DXVK [DropTrace]: aggregate the dropship sub-meshes specifically.
        if (pBl->input.studioModelName[0] != '\0'
            && (std::strstr(pBl->input.studioModelName, "Crow_dropship") != nullptr
             || std::strstr(pBl->input.studioModelName, "widow") != nullptr)) {
          dropInst++;
          if (blasPrim > 0u) dropPresent++;
          if (blasPrim > dropMaxBlas) dropMaxBlas = blasPrim;
        }
        const uint32_t vtx = pBl->modifiedGeometryData.vertexCount;
        // NV-DXVK [GeoChain] SESSION-E: the dropship is NOT culled — the engine
        // submits full geometry every frame ([DropGeo] ext01 count=80988 even at
        // present=0) but Remix builds a 0-primitive BLAS in the bad view
        // (maxBlasPrim 0<->26996 on identical input). Localize WHERE the count
        // collapses by logging the whole chain: engine-submitted raster counts
        // (input) -> raytrace-ready counts (modified) -> BLAS primitiveCount.
        //   inIdx>0 & modIdx==0  -> Remix's triangle-list/index gen zeroes it
        //   inIdx>0 & modIdx>0 & blasPrim==0 -> BLAS build drops all (degenerate tris)
        //   objBBvalid=0 / collapsed extent -> skinned verts collapsed (NaN/degenerate)
        const uint32_t inVtx  = pBl->input.getGeometryData().vertexCount;
        const uint32_t inIdx  = pBl->input.getGeometryData().indexCount;
        const uint32_t modIdx = pBl->modifiedGeometryData.indexCount;
        const uint32_t surfIdx = pInst->getSurfaceIndex();
        const bool surfBad = (surfIdx == SURFACE_INDEX_INVALID);
        const Matrix4& o2w = pInst->getTransform();
        const float tx = float(o2w[3][0]), ty = float(o2w[3][1]), tz = float(o2w[3][2]);
        // NV-DXVK [HullCensus] WORLD coordinates. o2w.t alone is useless for BSP-style geometry
        // (identity transform, verts carry world coords). Transform the object-space geometry
        // AABB by o2w to get the instance's actual WORLD position + extent. Diff between visible
        // and vanish frames: if the ship's world centroid/AABB MOVES when it vanishes, the
        // geometry is being teleported; if it's identical, the geometry sits still and the vanish
        // is purely a render/occlusion issue.
        const AxisAlignedBoundingBox& objBB = pBl->input.getGeometryData().boundingBox;
        const Vector3 wCen = objBB.getTransformedCentroid(o2w);
        const Vector3 wMinC = objBB.isValid() ? (o2w * Vector4(objBB.minPos, 1.0f)).xyz() : Vector3(0.f);
        const Vector3 wMaxC = objBB.isValid() ? (o2w * Vector4(objBB.maxPos, 1.0f)).xyz() : Vector3(0.f);
        const uint32_t gap = (currentFrame > pInst->m_frameLastUpdated)
                             ? (currentFrame - pInst->m_frameLastUpdated) : 0u;

        hTotal++;
        if (isHidden_) hHidden++;
        if (!inFr_)    hNotFr++;
        if (mask == 0u) hMaskZero++;
        if (surfBad)   hSurfBad++;
        if (cullActive) hCullActive++;

        // [RiddenTrace] accumulation: widow/Crow studio submesh at the recentered origin
        // is the ridden ship; far ones are the formation ship. Squared dist (no sqrt).
        {
          const bool isDropName = (pBl->input.studioModelName[0] != '\0'
              && (std::strstr(pBl->input.studioModelName, "Crow_dropship") != nullptr
               || std::strstr(pBl->input.studioModelName, "widow") != nullptr));
          if (isDropName) {
            const float o2wDist2 = tx * tx + ty * ty + tz * tz;
            if (o2wDist2 < 40000.0f) {                 // < 200u from recentered origin = ridden
              rN++;
              if (isHidden_)   rHidden++;
              if (mask == 0u)  rMaskZero++;
              if (!inFr_)      rNotFr++;
              if (blasPrim == 0u) rBlas0++;
              if (blasPrim < rMinBlas) rMinBlas = blasPrim;
              if (blasPrim > rMaxBlas) rMaxBlas = blasPrim;
              if (!isHidden_ && mask != 0u && blasPrim > 0u && inFr_) rRender++;
            } else {
              fN++;                                     // formation ship (far)
            }
          }
        }

        // Hull has few instances — log each every frame (not throttled). The per-instance
        // line is what we diff between the good and bad look direction.
        Logger::info(str::format(
          "[HullCensus.Inst] f=", currentFrame,
          " #", hIdx,
          " vs=0x", std::hex, vs, std::dec,
          " name=", (pBl->input.studioModelName[0] ? pBl->input.studioModelName : "(none)"),
          " mat=0x", std::hex, static_cast<uint64_t>(pBl->input.getMaterialData().getHash()), std::dec,
          // NV-DXVK: do NOT read getColorTexture().getImageHash() here. This
          // census runs inside InstanceManager::garbageCollection, concurrent
          // with the texture-streaming/GC thread. The isValid()+!isImageEmpty()
          // guard is NOT sufficient: streaming can free ManagedTexture::
          // m_currentMipView BETWEEN the isImageEmpty() check and getImageHash(),
          // so getImageHash() then derefs a dangling Rc<DxvkImage> → AV in
          // Rc::operator-> (util_rc_ptr.h:92). Confirmed by VS callstack:
          // getImageHash (rtx_texture.h:133) ← garbageCollection:963. A strong-
          // Rc copy does NOT help — the copy's incRef is itself the crashing
          // deref. mat=getHash() above is a plain XXH64 value (no image deref)
          // and is safe; the per-instance texture hash is simply omitted from
          // the GC-time walk. (See memory: getImageHash GC crash.)
          " tex=<skip-gc-uaf>",
          " hidden=", (isHidden_ ? 1 : 0),
          " inFrustum=", (inFr_ ? 1 : 0),
          " markedGC=", (gcMark_ ? 1 : 0),
          " lastUpdGap=", gap,
          " mask=0x", std::hex, mask, std::dec,
          " blasPrim=", blasPrim,
          " vtx=", vtx,
          // [GeoChain] engine-submitted -> raytrace-ready -> BLAS, + object-space
          // AABB validity/extent (collapsed => skinned verts degenerate).
          " inVtx=", inVtx, " inIdx=", inIdx, " modIdx=", modIdx,
          " objBBvalid=", (objBB.isValid() ? 1 : 0),
          " objBBmin=(", objBB.minPos.x, ",", objBB.minPos.y, ",", objBB.minPos.z, ")",
          " objBBmax=(", objBB.maxPos.x, ",", objBB.maxPos.y, ",", objBB.maxPos.z, ")",
          " surfIdx=", surfIdx,
          " cullActive=", (cullActive ? 1 : 0),
          " flipFace=", (flipFace ? 1 : 0),
          " cullDisable=", (cullOff ? 1 : 0),
          " category=0x", std::hex, static_cast<uint64_t>(pInst->getCategoryFlags().raw()), std::dec,
          " o2w.t=(", tx, ",", ty, ",", tz, ")",
          " worldCen=(", wCen.x, ",", wCen.y, ",", wCen.z, ")",
          " worldAABBmin=(", wMinC.x, ",", wMinC.y, ",", wMinC.z, ")",
          " worldAABBmax=(", wMaxC.x, ",", wMaxC.y, ",", wMaxC.z, ")"));
        hIdx++;
      }
      if (hTotal > 0u) {
        Logger::info(str::format(
          "[HullCensus] f=", currentFrame,
          " total=", hTotal,
          " hidden=", hHidden,
          " notInFrustum=", hNotFr,
          " maskZero=", hMaskZero,
          " surfIdxInvalid=", hSurfBad,
          " cullActive=", hCullActive));
      }
      // NV-DXVK [DropTrace]: per-frame dropship fate (logged every frame).
      //   submits     = dropship (Crow/Widow) draws that reached
      //                 submitDrawState THIS frame (runs before this GC)
      //   instances   = dropship instances currently in m_instances
      //   present     = those with blasPrim>0 (i.e. will render)
      //   maxBlasPrim = largest built prim count among them
      // Spin to sustain the despawn, then diff:
      //   submits>0 & present=0 -> draw arrives but BLAS/instance lost
      //                            downstream (scene/accel side)
      //   submits=0            -> draw stops reaching the scene under motion
      //                            (dropped in SubmitDraw / not submitted)
      {
        const uint32_t dtSubmits =
          (g_dropTraceFrame.load(std::memory_order_relaxed) == currentFrame)
            ? g_dropTraceSubmits.load(std::memory_order_relaxed) : 0u;
        // raw = dropship draws that ENTERED SubmitDraw this frame (pre-cascade).
        //   raw==submits -> engine stopped submitting (game-side cull)
        //   raw >submits -> Remix's SubmitDraw cascade dropped them
        const uint32_t dtRaw =
          (g_dropTraceRawFrame.load(std::memory_order_relaxed) == currentFrame)
            ? g_dropTraceRawSubmits.load(std::memory_order_relaxed) : 0u;
        Logger::info(str::format(
          "[DropTrace] f=", currentFrame,
          " raw=", dtRaw,
          " submits=", dtSubmits,
          " instances=", dropInst,
          " present=", dropPresent,
          " maxBlasPrim=", dropMaxBlas));

        // NV-DXVK [VanishEdge]: dump the full state at every ship appear/disappear
        // EDGE (when dropPresent crosses the visible<->gone threshold), so we can
        // characterize the TRIGGER across many transitions instead of fitting one.
        // dropPresent = ship submeshes with blasPrim>0 (actually rendering). Hysteresis
        // (>=4 present, <=2 gone) avoids flicker. Pairs the edge with the Main camera
        // pose ([CullCmp]/RtCamera, stashed in g_veCam*) + the ship counts. The player
        // RIDES the ship, so camPos is the ship's flight path; a fixed world band where
        // it vanishes shows up as the same camPos range each transition.
        {
          static int s_veState = -1;                       // -1 unknown, 0 gone, 1 present
          const int veState = (dropPresent >= 4u) ? 1
                            : (dropPresent <= 2u) ? 0
                            : s_veState;
          if (s_veState != -1 && veState != s_veState) {
            const bool poseFresh = (g_veCamFrame.load(std::memory_order_acquire) != 0xFFFFFFFFu);
            Logger::warn(str::format(
              "[VanishEdge] f=", currentFrame, " ", (veState ? "APPEAR" : "VANISH"),
              " present=", dropPresent, " submits=", dtSubmits,
              " instances=", dropInst, " maxBlasPrim=", dropMaxBlas,
              " camPos=(", g_veCamPx, ",", g_veCamPy, ",", g_veCamPz, ")",
              " camFwd=(", g_veCamDx, ",", g_veCamDy, ",", g_veCamDz, ")",
              " fov=", g_veCamFov, " camFrame=", g_veCamFrame.load(std::memory_order_relaxed),
              " poseFresh=", poseFresh ? 1 : 0));
          }
          if (veState != -1) s_veState = veState;
        }
        // [RiddenTrace]: one self-contained line — everything about the RIDDEN ship in a
        // single Logger call, so no cross-line frame matching. renderable==0 with rN>0
        // pinpoints the stage; renderable>0 while it's gone on screen => shading/pass.
        Logger::info(str::format(
          "[RiddenTrace] f=", currentFrame,
          " riddenSubmeshes=", rN,
          " renderable=", rRender,
          " blasPrim0=", rBlas0,
          " hidden=", rHidden,
          " maskZero=", rMaskZero,
          " notInFrustum=", rNotFr,
          " minBlas=", (rN ? rMinBlas : 0u),
          " maxBlas=", rMaxBlas,
          " formationSubmeshes=", fN,
          " dropPresent=", dropPresent));
      }
    }

    // NV-DXVK [SliceCollide]: does a held BLAS still own the memory it points at?
    //
    // This replaces what the builtPosHash test was supposed to do and cannot.
    // That test compares modifiedGeometryData.hashes against a copy of itself,
    // because the hash is only written by processGeometryInfo, which runs on a
    // DRAW, and a held instance is by definition one that got no draw. Measured:
    // stale=1 on 1285 of 1296 held instances, so the guard is vacuous for 99.15%
    // of the holds it decides, with touchAge running to 540 frames.
    //
    // The real question is not whether the content hash changed -- nobody
    // recomputes it -- but whether some OTHER BlasEntry now claims the same
    // vertex or index memory. Two distinct BlasEntry pointers resolving to one
    // (buffer, offset, length) means one of them is reading the other's data,
    // and if the loser is a held instance that is exactly the stretched plane:
    // geometry whose buffer was recycled underneath a hold.
    //
    // The identity triple is the one the bindless buffer cache already treats as
    // "one buffer" (RaytraceBufferHashFn in rtx_scene_manager.h), kept identical
    // on purpose so this probe and the cache cannot disagree about what sharing
    // means. Instances legitimately sharing ONE BlasEntry are ordinary
    // instancing and are not collisions; only distinct entries count.
    //
    // Positions and indices are both keyed, because a recycled index buffer
    // collapses geometry where a recycled position buffer stretches it, and the
    // two have shown up separately in this tree before.
    //
    // No buffer contents are read. These are device-local and mapPtr is null,
    // which is why the earlier attempts at a content test went through hashes in
    // the first place.
    if (RtxOptions::ResidentScene::enable() && RtxOptions::ResidentScene::logStats()) {
      struct SliceOwner {
        const BlasEntry* blas = nullptr;
        uint32_t touchAge = 0;
        uint32_t tri      = 0;
      };
      std::unordered_map<uint64_t, SliceOwner> owners;
      owners.reserve(m_instances.size() * 2u);

      uint32_t scSlices = 0, scCollide = 0, scStaleVictim = 0, scLines = 0;
      constexpr uint32_t kScMaxLines = 24;   // per call; the counts carry the rest
      // Session cap as well. If collisions are the steady state rather than an
      // event, 24 lines a frame is 80,000 lines over a short session and the
      // per-frame counts already say it is happening. The identities only need
      // saying often enough to see whether the same pair recurs.
      static uint32_t s_scLinesTotal = 0;
      constexpr uint32_t kScMaxLinesTotal = 400;

      auto sliceKey = [](const void* buf, uint64_t off, uint64_t len, uint64_t tag) {
        const uint64_t identity[4] = {
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf)), off, len, tag
        };
        return XXH3_64bits(identity, sizeof(identity));
      };

      for (RtInstance* pInst : m_instances) {
        // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
        // see that. Null the local so the checks below work. [ReapBlasUAF]
        const BlasEntry* b = (pInst != nullptr && !pInst->isUnlinkedForGC()) ? pInst->getBlas() : nullptr;
        if (b == nullptr) {
          continue;
        }
        const uint32_t bTouchAge = (b->frameLastTouched != kInvalidFrameIndex
                                    && currentFrame >= b->frameLastTouched)
                                     ? (currentFrame - b->frameLastTouched) : 0u;
        const uint32_t bTri = b->modifiedGeometryData.indexCount / 3u;

        for (uint32_t which = 0; which < 2u; ++which) {
          const auto& gb = (which == 0)
            ? b->modifiedGeometryData.positionBuffer
            : b->modifiedGeometryData.indexBuffer;
          // defined() only asks whether a buffer is attached, so a slice with a
          // non-null buffer and zero length passes it. Two of those on one
          // buffer collide trivially and are not geometry. Measured: the only
          // three collisions in two clean runs were len=0, one carrying
          // off=346902116632, on the final frame of a session that ended in an
          // unhandled exception -- dead entries at teardown, not recycling.
          if (!gb.defined() || gb.length() == 0) {
            continue;
          }
          const uint64_t key = sliceKey(gb.buffer().ptr(), gb.offset(), gb.length(), which);
          auto it = owners.find(key);
          if (it == owners.end()) {
            ++scSlices;
            owners.emplace(key, SliceOwner { b, bTouchAge, bTri });
            continue;
          }
          if (it->second.blas == b) {
            continue;   // ordinary instancing: many instances, one entry
          }
          ++scCollide;
          // The one that has not been drawn recently is the one at risk: its
          // geometry is frozen while another entry owns the same memory.
          const bool victimStale = (it->second.touchAge > 0u || bTouchAge > 0u);
          if (victimStale) {
            ++scStaleVictim;
          }
          if (scLines < kScMaxLines && s_scLinesTotal < kScMaxLinesTotal) {
            ++scLines;
            ++s_scLinesTotal;
            Logger::warn(str::format(
              "[SliceCollide] f=", currentFrame,
              " kind=", (which == 0 ? "pos" : "idx"),
              " buf=0x", std::hex,
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gb.buffer().ptr())), std::dec,
              " off=", gb.offset(),
              " len=", gb.length(),
              " incumbent{blas=0x", std::hex,
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(it->second.blas)), std::dec,
              " touchAge=", it->second.touchAge,
              " tri=", it->second.tri, "}",
              " newcomer{blas=0x", std::hex,
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b)), std::dec,
              " touchAge=", bTouchAge,
              " tri=", bTri, "}",
              " | distinct entries on one slice: one is reading the other's data"));
          }
        }
      }

      Logger::warn(str::format(
        "[SliceCollide] f=", currentFrame,
        " instances=", static_cast<uint32_t>(m_instances.size()),
        " slices=", scSlices,
        " collisions=", scCollide,
        " staleVictims=", scStaleVictim,
        " | collisions=0 means recycling is NOT the stretched-plane cause;"
        " staleVictims>0 names it"));
    }

    const bool forceGarbageCollection = (m_instances.size() >= RtxOptions::AntiCulling::Object::numObjectsToKeep());
    for (uint32_t i = 0; i < m_instances.size();) {
      // Must take a ref here since we'll be swapping
      RtInstance*& pInstance = m_instances[i];
      assert(pInstance != nullptr);

      const bool enableGarbageCollection =
        !RtxOptions::AntiCulling::isObjectAntiCullingEnabled() || // It's always True if anti-culling is disabled
        (pInstance->m_isInsideFrustum) ||
        // NV-DXVK: guarded for the reason [ReapBlasUAF] below documents -- an
        // unlinked-for-GC instance still points at the BlasEntry that
        // onSceneObjectDestroyed erased earlier in THIS SceneManager GC pass,
        // so the dereference reads freed memory. Reachable only with object
        // anti-culling ON, because the clause above it short-circuits when it
        // is off; that is why it has not fired, not because it is safe.
        // An unreadable BLAS counts as not-skinned, which is the conservative
        // side: such an instance is unlinked, so clauseMarked already removes
        // it and this term cannot change its fate.
        (!pInstance->isUnlinkedForGC() && pInstance->getBlas() != nullptr
         && pInstance->getBlas()->input.getSkinningState().numBones > 0) ||
        (pInstance->m_isAnimated) ||
        (pInstance->m_isPlayerModel);

      // NV-DXVK [GcKeep2904]: log every VS_2904d2 instance evaluated at GC,
      // not just removed ones. Rate-limit per-frame: log first 5 mountain
      // instances per GC pass so we get a representative slice without
      // 48/frame spam. Tells us whether m_isInsideFrustum is correctly
      // false (IgnoreAntiCulling working) or stuck true (BLAS frustum
      // loop didn't re-evaluate after a frame skip), and whether the
      // sticky-IgnoreAntiCulling preserve from updateInstance survived.
      {
        // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
        // see that. Null the local so the checks below work. [ReapBlasUAF]
        const BlasEntry* pBlasInspect = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
        if (pBlasInspect != nullptr
            && pBlasInspect->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
          static thread_local uint32_t sKeepProbeFrame = UINT32_MAX;
          static thread_local uint32_t sKeepProbeCount = 0;
          if (sKeepProbeFrame != currentFrame) {
            sKeepProbeFrame = currentFrame;
            sKeepProbeCount = 0;
          }
          if (sKeepProbeCount < 5) {
            Logger::info(str::format(
              "[GcKeep2904] #", sKeepProbeCount,
              " f=", currentFrame,
              " lastUpd=", pInstance->m_frameLastUpdated,
              " gap=", (currentFrame - pInstance->m_frameLastUpdated),
              " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
              " ignAntiCull=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
              " enableGC=", (enableGarbageCollection ? 1 : 0),
              // NV-DXVK: same hazard as [ReapBlasUAF] below, in the same GC
              // pass -- an unlinked-for-GC instance still points at the
              // BlasEntry that onSceneObjectDestroyed erased, so this
              // dereference reads freed memory. -1 rather than 0 because a
              // probe must not report a value it could not read.
              " skinned=", ((!pInstance->isUnlinkedForGC() && pInstance->getBlas() != nullptr)
                            ? (pInstance->getBlas()->input.getSkinningState().numBones > 0 ? 1 : 0)
                            : -1),
              " animated=", (pInstance->m_isAnimated ? 1 : 0),
              " playerMdl=", (pInstance->m_isPlayerModel ? 1 : 0),
              " markedGC=", (pInstance->m_isMarkedForGC ? 1 : 0),
              " propId=0x", std::hex, pInstance->m_stablePropId, std::dec));
            sKeepProbeCount += 1;
          }
        }
      }

      // NV-DXVK [GC clause probe]: capture WHICH side of the OR fired and
      // whether this is a sub-view mountain instance. Lets us correlate
      // [Rm2904] entries that show "all conditions false" with the actual
      // GC-time evaluation.
      //
      // NV-DXVK [SubViewKeepLong]: instances tagged IgnoreAntiCulling
      // (TF2 3D-skybox sub-view content) get the longer
      // numFramesToKeepSubViewInstances grant. TF2's engine throttles
      // sub-view rendering — on frames where it doesn't redraw the sub-
      // view fan, those instances aren't touched. With the default
      // numFramesToKeepInstances=1, ONE skipped frame retires the entire
      // sub-view fan together; their ordered-surface slots get
      // reallocated; previous-frame GBuffer pixels referencing the now-
      // retired slots render with the wrong content (large black blocks
      // on mountains / dome). Verified via [Coverage] priorOwnerFrame
      // data: an entire 800k-pixel stale event traced to one frame
      // whose SubViewGateCounts showed 90 fewer candidates than steady
      // state. The longer keep absorbs engine LOD skips up to ~16 frames.
      //
      // NV-DXVK [PropIdKeepLong attempt reverted]: a previous version of
      // this gate extended the long keep to any instance with a non-zero
      // stablePropId. That made things WORSE for the path-10 bone-
      // instanced fanout (VS_2947c6 ~7787 PI slots/frame): its
      // MakeBoneStablePropId propId isn't actually stable across frames
      // (rolls at the i2o[0].T 1u rounding boundary or whenever the
      // engine rotates the per-instance buffer arena), so dedup misses
      // were creating new RtInstances each frame. With keepN=1 the old
      // ones retired fast; with keepN=16 they all stayed alive AND every
      // touched-this-frame instance still calls addPointInstancerBlas →
      // m_reorderedSurfaces doubled to 17155, every subsequent collapse
      // brought DOWN 17000 slots' worth of stale pixels instead of 8500.
      // The right fix lives at the propId producer (stabilize
      // MakeBoneStablePropId across the rolling input), not here. Until
      // that lands, keep the gate strict to IgnoreAntiCulling.
      // NV-DXVK [keepStablePropIdInstancesLong]: shadow-sourced fanout terrain
      // (VS_2947c6) is submitted only via TF2's spot-shadow pass; when that
      // pass culls it, it stops being submitted and would retire at the
      // default keepN=1. With a STABLE identity
      // (rtx.boneStablePropIdFanoutPositionOnly) the longer keep is safe and
      // retains it across the shadow-off frames. Continuously-submitted props
      // are unaffected (their keep window never elapses).
      const bool keepLongForProp =
        RtxOptions::keepStablePropIdInstancesLong()
        && pInstance->m_stablePropId != 0ull;
      const uint32_t instanceKeepN =
        (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) || keepLongForProp)
          ? RtxOptions::numFramesToKeepSubViewInstances()
          : numFramesToKeepInstances;

      // NV-DXVK [ResidentScene]: THE RESIDENCY CLAUSE. A SEPARATE clause beside
      // the one above, deliberately not folded into it.
      //
      // WHY NOT REUSE THE IgnoreAntiCulling GATE. Because this is not
      // anti-culling and must not become entangled with it. That category is a
      // per-material/per-content tag applied to TF2 sub-view (3D-skybox)
      // geometry; residency is a statement about an instance's IDENTITY being
      // stable and its record being live, which is orthogonal. Folding them
      // would mean a change to either one silently moves the other's
      // population, and it would put residency behind an anti-culling switch
      // that is explicitly out of scope for this work.
      //
      // WHY A KEEP OVERRIDE RATHER THAN A LONGER keepN. Because the whole point
      // is to stop lifetime being a function of frame age at all. keepFrames
      // defaults to 0 = unbounded: a resident instance is reaped when its
      // record is invalidated (source buffer destroyed, BLAS torn down, LRU
      // eviction), never because a draw failed to arrive. Absence of a draw is
      // what engine culling produces, and treating it as death is exactly the
      // defect residency exists to remove.
      //
      // NOT GATED ON verify, AND IT USED TO BE. The reason is measured rather
      // than argued: with the keep gated off during verification, an instance
      // that missed one frame retired on numFramesToKeepInstances, took its
      // record with it through invalidateFor, and the next frame's prediction
      // then scored a failure for a record that verification itself had
      // destroyed. [ResidentScene] read noRecLost=481 erased=481 never=0 -- one
      // hundred percent of missing records had been filed and then erased, none
      // was ever absent. So the old gating made FAIL measure how thoroughly the
      // keep was disabled, and required FAIL=0 before enabling it. Circular.
      //
      // WHAT verify STILL MEANS, and this is the distinction the old clause
      // blurred: verify governs the SKIP, which is a prediction being acted on.
      // The keep is not a prediction -- it is driven by whether a valid record
      // exists, which is a fact the CS side already holds. Holding an instance
      // alive cannot make anything on screen wrong in the way a wrongly skipped
      // draw can; the failure mode is retaining geometry too long, which is
      // visible in records=, liveInst= and m_reorderedSurfaces.
      //
      // THE trap 3 GUARD IS NOT verify, IT IS THE CEILING BELOW plus the key
      // stability reading. [ResidentGate] newKeys reads 14-32 per ten frames
      // standing still, so the identity is stable and the
      // [PropIdKeepLong attempt reverted] precondition does not hold here.
      // AND IT DOES NOT OVERRIDE forceGarbageCollection. That path is the
      // over-the-cap reap (numObjectsToKeep), i.e. the backstop against exactly
      // the runaway trap 3 describes -- an identity that churns mints a fresh
      // instance every frame, and if residency also made every one of them
      // unreapable the count would climb until the LRU noticed, with
      // m_reorderedSurfaces climbing with it. Residency defers ordinary
      // lifetime expiry; it does not get to defeat the ceiling.
      // THROUGH holdsInstance RATHER THAN find()+valid, so this clause and the
      // BLAS clause in SceneManager::garbageCollection ask the same question
      // through the same code. They were two separate reads of the record and
      // they had already drifted -- neither excluded skipUnsafe, which only
      // stopped mattering because the touch refused those records separately.
      bool residencyHolds = false;
      if (RtxOptions::ResidentScene::enable() && !forceGarbageCollection) {
        residencyHolds = m_residentScene.holdsInstance(pInstance);
      }

      // NV-DXVK [HeldCensus]: IS THE STALE-HOLD DEFECT STILL HAPPENING, over
      // ALL geometry rather than the 24-line [HeldRaw] burst.
      //
      // The defect: a held instance whose vertex content is regenerated per
      // draw renders whatever was last written into its buffer when no draw
      // arrives. On the Titanfall viewmodel that showed as one side of the
      // weapon and hands going black -- stale normals, so a shading artefact on
      // part of a mesh rather than a missing object -- while every residency
      // counter read healthy. [HeldRaw] is capped and samples; this counts.
      //
      // bone= IS THE ACCEPTANCE COLUMN. ResidentScene::build now sets
      // skipUnsafe on any record with a non-zero builtBoneHash, so a skinned
      // instance must never be held. This must read 0. If it does not, the
      // disqualification is not reaching these instances and the bone signal is
      // wrong, which is a different finding from the fix not working.
      //
      // stale= IS THE GENERAL FORM, and it is why this counts all geometry
      // rather than skinned alone. Skinning is one way for held content to go
      // stale; any producer that rewrites a BLAS's buffers between frames is
      // another, and nothing here knows their names. An instance held with
      // frameLastTouched behind the current frame is one whose geometry nobody
      // refreshed, so stale= climbing while bone= sits at 0 says a sixth class
      // exists and names the population to go looking in.
      //
      // Counted, never classified. The counters below are raw populations with
      // no threshold and no verdict, because the last four readings of this
      // system were each a classifier confirming its own premise.
      //
      // getBlas() IS GUARDED ON isUnlinkedForGC, not on null. This runs inside
      // the same garbage-collection pass that erases BlasEntries, and an
      // unlinked instance keeps m_linkedBlas pointing at the erased entry, so
      // getBlas() returns a dangling pointer that survives a null test. That is
      // the [ReapBlasUAF] hazard the block below already documents.
      if (RtxOptions::ResidentScene::enable()) {
        static uint32_t sHcFrame = 0xFFFFFFFFu;
        static uint32_t sHcLastDump = 0xFFFFFFFFu;
        static uint32_t sHcLive = 0u, sHcHeld = 0u, sHcHeldNoDraw = 0u;
        static uint32_t sHcBone = 0u, sHcStale = 0u, sHcUnlinked = 0u, sHcNoBlas = 0u;
        constexpr uint32_t kHcDumpFrames = 300u;

        if (sHcFrame != currentFrame) {
          sHcFrame = currentFrame;
          if (sHcLastDump == 0xFFFFFFFFu) {
            sHcLastDump = currentFrame;
          } else if (currentFrame - sHcLastDump >= kHcDumpFrames) {
            sHcLastDump = currentFrame;
            Logger::warn(str::format(
              "[HeldCensus] f=", currentFrame,
              " live=", sHcLive,
              " held=", sHcHeld,
              " heldNoDraw=", sHcHeldNoDraw,
              " bone=", sHcBone,
              " stale=", sHcStale,
              " unlinked=", sHcUnlinked,
              " noBlas=", sHcNoBlas));
            sHcLive = sHcHeld = sHcHeldNoDraw = 0u;
            sHcBone = sHcStale = sHcUnlinked = sHcNoBlas = 0u;
          }
        }

        sHcLive += 1u;
        if (residencyHolds) {
          sHcHeld += 1u;
          const bool noDraw = (pInstance->m_frameLastUpdated != currentFrame);
          if (noDraw) {
            sHcHeldNoDraw += 1u;
          }
          if (pInstance->isUnlinkedForGC()) {
            sHcUnlinked += 1u;
          } else if (const BlasEntry* hcBlas = pInstance->getBlas()) {
            if (hcBlas->modifiedGeometryData.lastBoneHash != 0ull) {
              sHcBone += 1u;
            }
            if (noDraw && hcBlas->frameLastTouched != currentFrame) {
              sHcStale += 1u;
            }
          } else {
            sHcNoBlas += 1u;
          }
        }
      }

      // NV-DXVK [HeldRaw]: WHAT IS ACTUALLY BEING HELD, one raw line per
      // instance, no classification.
      //
      // skipUnsafe is a property test over three properties -- billboards, ray
      // portals, decals -- and the plan says in as many words that a fourth
      // class will announce itself as a visual bug rather than as a counter.
      // One has. Every residency counter reads healthy while the screen is
      // wrong, so the counters are the wrong instrument and a histogram or a
      // classifier here would only be the same guess with error bars.
      //
      // So this prints the facts and nothing derived: geometry size, material
      // identity, the full alpha state, the camera mask and where the thing is
      // in the world. A flat quad reads tri=2, which is what "floating plane"
      // would be, and if it is something else the line says so instead.
      //
      // STALE ONLY. An instance drawn this frame is being maintained by the full
      // path and is not what residency is doing to the scene; the population at
      // issue is the one the keep is preserving without a draw.
      if (residencyHolds
          && RtxOptions::ResidentScene::logStats()
          && pInstance->m_frameLastUpdated != currentFrame
          && s_heldRawBurstFrame == currentFrame
          && s_heldRawPrinted < 24u) {
        s_heldRawPrinted += 1;
        const auto& sf = pInstance->surface;
        const auto& as = sf.alphaState;
        const Matrix4& o2w = sf.objectToWorld;
        // The counts live on the BLAS's RaytraceGeometry, not on RtSurface,
        // which carries bindless buffer INDICES rather than sizes. Null-guarded
        // because an instance can outlive its geometry link, and only the counts
        // are read -- never the buffers, whose mapPtr is null for device-local
        // index data.
        // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
        // see that. Null the local so the checks below work. [ReapBlasUAF]
        const BlasEntry* blas = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
        const uint32_t idxCount = (blas != nullptr) ? blas->modifiedGeometryData.indexCount : 0u;
        const uint32_t vtxCount = (blas != nullptr) ? blas->modifiedGeometryData.vertexCount : 0u;
        const ResidentScene::Record* heldRec = m_residentScene.find(pInstance->m_residentKey);
        Logger::warn(str::format(
          "[HeldRaw] f=", currentFrame,
          " age=", (currentFrame - pInstance->m_frameLastUpdated),
          " tri=", (idxCount / 3u),
          " vtx=", vtxCount,
          // getMaterialDataHash, NOT getImageHash: this walk runs concurrently
          // with streaming and getImageHash on a GC walk is the documented
          // use-after-free in this tree.
          " mat=", std::hex, pInstance->getMaterialDataHash(), std::dec,
          " type=", static_cast<uint32_t>(pInstance->getMaterialType()),
          " bb=", pInstance->getBillboardCount(),
          " cam=", pInstance->getSeenCameraMask(),
          // THE CONTENT TEST, AND THE REASON IT READS ZERO.
          //
          // A held instance keeps its BLAS alive, but nothing proves the BLAS's
          // vertex content is still what the record was built against. bone!=0
          // says the geometry is skinned, so its positions are regenerated per
          // draw and a held instance with no draw renders whatever was written
          // there last. posMoved=1 says the content changed under the hold, in
          // the general form that catches unnamed producers.
          //
          // stale= is what makes posMoved readable, and it was missing.
          // modifiedGeometryData.hashes is written in exactly one place --
          // processGeometryInfo in rtx_scene_manager.cpp -- which runs when a
          // DRAW arrives. A held instance is by definition one that got no draw,
          // so its hash is frozen at the value builtPosHash was copied from, and
          // the comparison is a value against a copy of itself. It cannot report
          // a change. posMoved=0 across every held instance is that tautology,
          // not evidence, and it has already been read once as a pass.
          //
          // frameLastTouched answers "did any draw for this geometry arrive", so
          // stale=1 marks exactly the lines where posMoved knows nothing. Only
          // stale=0 lines carry a verdict. If stale=1 dominates, the hash is the
          // wrong invariant for this job rather than a broken one: what a hold
          // actually needs is that the BLAS's buffers cannot be recycled beneath
          // it, which is a different question and needs its own answer.
          " bone=", std::hex, (blas != nullptr ? blas->modifiedGeometryData.lastBoneHash : 0ull), std::dec,
          " stale=", (blas == nullptr || blas->frameLastTouched != currentFrame) ? 1 : 0,
          " touchAge=", (blas != nullptr && blas->frameLastTouched != kInvalidFrameIndex
                         && currentFrame >= blas->frameLastTouched)
                          ? (currentFrame - blas->frameLastTouched) : 0u,
          " posMoved=", (heldRec != nullptr && blas != nullptr && heldRec->builtPosHash != 0ull
                         && heldRec->builtPosHash
                              != blas->modifiedGeometryData.hashes[HashComponents::VertexPosition]) ? 1 : 0,
          " alpha{blendOff=", as.isBlendingDisabled ? 1 : 0,
          " opaque=", as.isFullyOpaque ? 1 : 0,
          " test=", static_cast<uint32_t>(as.alphaTestType),
          " blend=", static_cast<uint32_t>(as.blendType),
          " emissive=", as.emissiveBlend ? 1 : 0,
          " particle=", as.isParticle ? 1 : 0,
          " decal=", as.isDecal ? 1 : 0, "}",
          " pos=", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2],
          // NV-DXVK: THE TRANSFORM ITSELF, because a stretched quad is a
          // statement about the matrix and every field above describes the
          // MESH. tri, vtx, posMoved and the alpha state all said this geometry
          // is a healthy four-triangle alpha-tested cutout, and they would say
          // that whether it rendered correctly or smeared across the map.
          //
          // A held instance keeps the objectToWorld it last received. If that
          // matrix was captured mid-update, or if the instance is held while the
          // basis it was built against moves on, the mesh is drawn correct but
          // POSED wrong -- which is a stretch, not a content error, and no hash
          // over vertex data can see it.
          //
          // Read basis lengths against each other: a rigid placement has three
          // similar, moderate values. One axis far larger than the others is
          // literally the stretch. det near zero is a collapsed (degenerate)
          // basis, which draws as a sliver. Non-finite is a poisoned matrix.
          " basis=", length(Vector3(o2w[0][0], o2w[0][1], o2w[0][2])),
          ",", length(Vector3(o2w[1][0], o2w[1][1], o2w[1][2])),
          ",", length(Vector3(o2w[2][0], o2w[2][1], o2w[2][2])),
          " det=", (o2w[0][0] * (o2w[1][1] * o2w[2][2] - o2w[1][2] * o2w[2][1])
                  - o2w[0][1] * (o2w[1][0] * o2w[2][2] - o2w[1][2] * o2w[2][0])
                  + o2w[0][2] * (o2w[1][0] * o2w[2][1] - o2w[1][1] * o2w[2][0])),
          " finite=", (std::isfinite(float(o2w[3][0])) && std::isfinite(float(o2w[3][1]))
                    && std::isfinite(float(o2w[3][2])) && std::isfinite(float(o2w[0][0]))
                    && std::isfinite(float(o2w[1][1])) && std::isfinite(float(o2w[2][2])) ? 1 : 0)));
      }

      const bool clauseLifetime = (forceGarbageCollection || enableGarbageCollection) &&
                                  !residencyHolds &&
                                  (pInstance->m_frameLastUpdated + instanceKeepN <= currentFrame);
      const bool clauseMarked = pInstance->m_isMarkedForGC;
      const bool shouldRemove = clauseLifetime || clauseMarked;
      if (shouldRemove) {
        probeRemovedThisPass += 1;
        if (clauseMarked)   probeRemovedMarked   += 1;
        if (clauseLifetime) probeRemovedLifetime += 1;
        if (forceGarbageCollection && !enableGarbageCollection && clauseLifetime) probeRemovedForce += 1;

        // NV-DXVK [ReapJoin]: classify EVERY reap. Deliberately OUTSIDE the
        // [InstReap] detail block below, which is gated on the engine-hook
        // capture counter — the totals must describe every reap in the pass, not
        // only the ones that got a detail line. A summary that silently counts a
        // subset is how the 24-line cap misreported this same churn on
        // 2026-07-29.
        //
        // The sibling walk is O(linked), and linked is 1-3 for the churning
        // meshes and ~20 at its worst in the 2026-08-05 capture — bounded by how
        // many copies of ONE mesh exist, not by scene size.
        uint32_t reapDraws     = 0;   // draws that resolved to this geometry this frame
        uint32_t reapFreshSib  = 0;   // siblings created this frame
        uint32_t reapReusedSib = 0;   // siblings that pre-existed and were claimed this frame
        {
          const BlasEntry* pBlasJoin = pInstance->getBlas();
          // NV-DXVK [ReapBlasUAF]: this walk crashed, and the null check below
          // cannot stop it. Symbolized from a 0xc0000005 at 06:02 on 2026-08-23:
          // av=read target=0x278 with rax=0, faulting on sibling->m_frameLastUpdated
          // (+0x278) on the CS thread, frame[8] = rtx_scene_manager.cpp:700.
          //
          // WHY THE POINTER IS BAD AND WHY nullptr DOES NOT CATCH IT. Within one
          // SceneManager::garbageCollection, entries.erase() DESTROYS BlasEntries
          // at rtx_scene_manager.cpp:438, and onSceneObjectDestroyed MARKS each
          // linked instance rather than clearing its back-pointer. The instance
          // GC then runs at line 700 of that same function, so getBlas() here
          // returns a pointer to an erased entry -- non-null, so `!= nullptr`
          // passes, and getLinkedInstances() reads freed memory. The zeros it
          // read are where the null sibling came from.
          //
          // m_isUnlinkedForGC IS THE SIGNAL, and it is this class's own. It is
          // set in exactly one place -- onSceneObjectDestroyed, beside the
          // erase -- so it means "my entry is gone" and nothing else.
          // isInstanceLinkedToBlas() at rtx_instance_manager.h:149 already
          // refuses to touch the BLAS on it; this walk is a site that never got
          // that guard, and adding it here is applying an existing invariant
          // rather than inventing one.
          //
          // Testing a flag ON THE INSTANCE needs no access to the freed entry,
          // which is the whole point -- a magic number or generation stamp on
          // BlasEntry would have to read the very memory that is gone.
          //
          // NOT m_isMarkedForGC. That one is set at four sites: this one plus
          // the viewModel, clone and virtual lifecycles, where the entry is
          // alive and the walk is perfectly safe. Gating on it (as this probe
          // first did, 2026-08-23) skipped ~7 walks per frame, most of them
          // needlessly, and pushed every skipped instance into probeReapStarved
          // because reapFreshSib stayed 0.
          //
          // THIS DOES NOT FIX THE DANGLING POINTER. It refuses to dereference
          // it and says so. The invariant that wants restoring is that a
          // back-pointer must not outlive its target, and that belongs in the
          // destroy path, not here. The counter below is what says how often
          // this is reached, so the repair can be measured rather than assumed.
          //
          // A skipped walk still leaves reapFreshSib at 0, so a skipped
          // instance lands in probeReapStarved rather than probeReapRespawn.
          // That is now confined to instances whose entry really is gone, whose
          // sibling counts were never obtainable anyway, but read the # counter
          // against the starved total before trusting it -- the totals above
          // are meant to cover every reap.
          if (pInstance->isUnlinkedForGC()) {
            static thread_local uint32_t sReapUafSkips = 0;
            if (sReapUafSkips < 32 || (sReapUafSkips & 0x3FF) == 0) {
              Logger::warn(str::format(
                "[ReapBlasUAF] #", sReapUafSkips,
                " f=", currentFrame,
                " instId=", pInstance->getId(),
                " blas=0x", std::hex, reinterpret_cast<uintptr_t>(pBlasJoin), std::dec,
                " instCreated=", pInstance->m_frameCreated,
                " instLastUpdated=", pInstance->m_frameLastUpdated,
                " | marked-for-GC instance: getBlas() may be an erased entry, sibling walk SKIPPED"));
            }
            sReapUafSkips += 1;
          } else if (pBlasJoin != nullptr) {
            reapDraws = pBlasJoin->getDrawCount(currentFrame);
            for (const RtInstance* sibling : pBlasJoin->getLinkedInstances()) {
              // SECOND NET, for any route to a bad list that is not the marked
              // one above. A null element cannot come from the linking API --
              // unlinkInstance swap-and-pops and linkInstance never pushes null
              // -- so one appearing here means this list is not a live list, and
              // the walk must stop rather than read the next element out of it.
              if (sibling == nullptr) {
                static thread_local uint32_t sReapUafNull = 0;
                if (sReapUafNull < 32) {
                  Logger::warn(str::format(
                    "[ReapBlasUAF] NULL-SIBLING #", sReapUafNull,
                    " f=", currentFrame,
                    " instId=", pInstance->getId(),
                    " blas=0x", std::hex, reinterpret_cast<uintptr_t>(pBlasJoin), std::dec,
                    " linked=", static_cast<uint32_t>(pBlasJoin->getLinkedInstances().size()),
                    " | UNMARKED instance with a null in its linked list -- the marked-GC"
                    " explanation does NOT cover this one, walk ABANDONED"));
                }
                sReapUafNull += 1;
                break;
              }
              // Skip ourselves: by construction we were not updated this frame,
              // which is why we are being reaped.
              if (sibling == pInstance || sibling->m_frameLastUpdated != currentFrame) {
                continue;
              }
              if (sibling->m_frameCreated == currentFrame) {
                reapFreshSib += 1;
              } else {
                reapReusedSib += 1;
              }
            }
          }
          if (reapFreshSib > 0) {
            probeReapRespawn += 1;
          } else {
            probeReapStarved += 1;
          }
        }

        // Per-instance log for VS_2904d2: capture exactly what GC saw.
        {
          // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
          // see that. Null the local so the checks below work. [ReapBlasUAF]
          const BlasEntry* pBlasGC = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
          if (pBlasGC != nullptr
              && pBlasGC->input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
            thread_local uint32_t sGcVsProbe = 0;
            if (sGcVsProbe < 16 || (sGcVsProbe & 0xFF) == 0) {
              Logger::info(str::format(
                "[Gc2904Decide] #", sGcVsProbe,
                " f=", currentFrame,
                " lastUpd=", pInstance->m_frameLastUpdated,
                " keepN=", instanceKeepN,
                " keepNbase=", numFramesToKeepInstances,
                " force=", (forceGarbageCollection ? 1 : 0),
                " enable=", (enableGarbageCollection ? 1 : 0),
                " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
                " ignAntiCull=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
                " markedGC=", (clauseMarked ? 1 : 0),
                " clauseLifetime=", (clauseLifetime ? 1 : 0)));
            }
            sGcVsProbe += 1;
          }
        }

        // NV-DXVK [InstReap]: per-removal detail for ALL VSes — the s2s
        // "two views" flip probe. Theory: view 1's dome + s2s superstructure
        // are intro-era (f~527-531) sub-view instances coasting on the
        // 16-frame numFramesToKeepSubViewInstances grant after the steady
        // stream stopped refreshing them; the one-frame view-2 flip
        // (sky-miss 0%->45%, BlasDestroyed wave peaking x25) is their
        // collective expiry. Readout at the next flip:
        //   keepN=16, age 16-17, lastUpd in the intro band, on the
        //     big-coverage vsHashes (0x2859d250/0x292b6ba0/...) ->
        //     expiry CONFIRMED; the fix is making steady draws resolve to
        //     these instances (or their propId producer), NOT a longer keep.
        //   keepN=1, age~2 on those vsHashes -> the draw stream itself
        //     dropped/rekeyed them (churn) — look upstream of GC.
        // pos= identifies dome (|pos| huge, reprojected) vs deck (~y=-10860).
        // Gameplay-gated; 24 detail lines/frame ([GcExit] keeps the totals).
        // NV-DXVK 2026-08-06: tagDenied gate — 46 lines/frame in the 00:48
        // capture, and the gather below (map size, material hash, world pos)
        // is wasted when the denylist drops the line. The churn MAGNITUDE now
        // lives on [MapGate]'s per-frame counters ([GcExit] also keeps totals
        // but is itself denylisted by default), so denying the detail lines
        // loses identity only, not the measurement.
        // RESPAWNS ARE LOGGED UNDER THEIR OWN TAG, and the reason is volume
        // rather than taste. [InstReap] is uncapped and is denied in rtx.conf's
        // logDenyTags because it fires on EVERY reap -- ~3/frame in steady state
        // and 500+/frame during the warm-up wave, which is enough file I/O to
        // stall the game. But a respawn is the rare and interesting case: an
        // instance removed and immediately recreated for the same geometry,
        // which is what an object flickering or losing its material looks like
        // from here. Steady state is 0.1-0.4/frame.
        //
        // So the identity that already exists in the block below is emitted a
        // second time under [Respawn], which is NOT denied, and the expensive
        // per-reap firehose stays off. Same fields, one thousandth the lines.
        const bool reapIsRespawn = (reapFreshSib > 0);
        if (reapIsRespawn
            && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
          // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
          // see that. Null the local so the checks below work. [ReapBlasUAF]
          const BlasEntry* pBlasRs = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
          const Vector3 posRs = pInstance->getWorldPosition();
          Logger::warn(str::format(
            "[Respawn] f=", currentFrame,
            " vs=0x", std::hex,
            static_cast<uint64_t>(pBlasRs != nullptr
              ? pBlasRs->input.getTransformData().vertexShaderHash : 0ull),
            " mat=0x", static_cast<uint64_t>(pInstance->getMaterialDataHash()), std::dec,
            " v=", (pBlasRs != nullptr ? pBlasRs->modifiedGeometryData.vertexCount : 0u),
            " freshSib=", reapFreshSib,
            " reusedSib=", reapReusedSib,
            " draws=", reapDraws,
            // Was residency holding this one? A respawn on a HELD instance would
            // mean the keep failed to prevent the very churn it exists to stop.
            " heldKey=", (pInstance->m_residentKey != 0ull) ? 1 : 0,
            " pos=", posRs.x, ",", posRs.y, ",", posRs.z));
        }

        if (!Logger::tagDenied("[InstReap]")
            && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
          // UNCAPPED. This was 24/frame and it SATURATED on every single frame
          // of the 2026-07-29 capture — 24 reaps/frame, unbroken — so the
          // number in the log was the cap reporting itself, not the scene, and
          // the real churn magnitude was never measured. Every reap is logged
          // now. A count that is the answer must not be clipped by the probe
          // that measures it.
          {
            // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
            // see that. Null the local so the checks below work. [ReapBlasUAF]
            const BlasEntry* pBlasRp = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
            const XXH64_hash_t vsRp = (pBlasRp != nullptr)
              ? pBlasRp->input.getTransformData().vertexShaderHash : 0ull;
            const Vector3 posRp = pInstance->getWorldPosition();
            // NV-DXVK: blasPtr + that BlasEntry's SpatialMap size at reap time.
            // Two props that share a mesh share a geometry hash and therefore a
            // BlasEntry, and the 2026-07-29 capture showed such a pair — one at
            // a main-world transform, one at a 3D-skybox transform ~26000u away
            // — alternating CREATE every frame on one blasPtr while the map
            // never grows past 1. SpatialMap::insert does not evict (it bumps
            // on hash collision, and [SpatialBump] fired 0 times), so the only
            // way the loser leaves the map is this reap. blasPtr groups the
            // pair; mapSz says whether the survivor was already alone when the
            // reap ran; age vs keepN says whether this is ordinary lifetime
            // expiry (i.e. that prop's draw never arrived) or something else.
            const size_t mapSzRp = (pBlasRp != nullptr)
              ? pBlasRp->getSpatialMap().size() : 0u;
            // mat= is the only field that ties a reap back to a group in the
            // auto_scene OBJ (whose names are vs+mat). vs alone is ambiguous —
            // one VS draws several unrelated meshes.
            const uint64_t matRp = (pBlasRp != nullptr)
              ? static_cast<uint64_t>(pBlasRp->input.getMaterialData().getHash()) : 0ull;
            // v= is the [MeshTrace] identity's second half. vs alone cannot
            // name a mesh (one shader draws many) and mat is not run-stable,
            // so without this a reap cannot be joined to the mesh that
            // vanished from the TLAS.
            const uint32_t vertsRp = (pBlasRp != nullptr)
              ? pBlasRp->modifiedGeometryData.vertexCount : 0u;
            // NV-DXVK [InstReap naming]: the engine .mdl path for this geometry.
            //
            // Every identifier on this line is a hash, so a churn census can say
            // "mat=0x21edeedd7e974f08 v=32 was reaped 96 times" without anyone
            // being able to say WHAT that is. studioModelName is the engine's own
            // name for it and costs nothing here — the BlasEntry is already in
            // hand and the string is value-copied into DrawCallState at capture.
            //
            // TWO LIMITS, both real:
            //  - Empty unless rtx.tf2DumpStudioNames (or a widow flag / the pick
            //    tool) is on, and only ever populated for STUDIORENDER draws.
            //    "(unnamed)" therefore means "not a studio draw, or the gate is
            //    off" — it is NOT evidence the object has no identity.
            //  - BlasEntry::input is overwritten by every pairing, so this names
            //    the draw that most recently claimed the entry. For a stable
            //    entry that is the object; for a re-bucketing one it can lag.
            // It names the MESH, never the individual instance — a fanout batch
            // is one draw over many props, so all of them report one name.
            const char* nameRp = (pBlasRp != nullptr && pBlasRp->input.studioModelName[0] != '\0')
              ? pBlasRp->input.studioModelName : "(unnamed)";
            Logger::info(str::format(
              "[InstReap] f=", currentFrame,
              " name=", nameRp,
              " vs=0x", std::hex, static_cast<uint64_t>(vsRp),
              " mat=0x", matRp, std::dec,
              " v=", vertsRp, std::hex,
              " propId=0x", static_cast<uint64_t>(pInstance->m_stablePropId), std::dec,
              " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(pBlasRp), std::dec,
              " mapSz=", mapSzRp,
              // NV-DXVK [ReapJoin] per-line fields. verdict= is the answer:
              // respawn = a sibling instance was CREATED this frame on this same
              // geometry while this one went unclaimed (matching failure, ours),
              // starved = nothing replaced it (submission gap upstream).
              // draws/freshSib/reusedSib are the raw terms behind it — read them,
              // not the verdict alone, when draws != freshSib + reusedSib.
              //
              // drew= is RETAINED ONLY as the literal restatement of
              // (blasTouch == f). It is per-GEOMETRY and is worthless on its own
              // for a mesh with siblings — see the block at the top of
              // garbageCollection. Never aggregate it.
              " verdict=", (reapFreshSib > 0 ? "respawn" : "starved"),
              " draws=", reapDraws,
              " freshSib=", reapFreshSib,
              " reusedSib=", reapReusedSib,
              " drew=", (pBlasRp != nullptr && pBlasRp->frameLastTouched == currentFrame) ? 1 : 0,
              " blasTouch=", (pBlasRp != nullptr ? pBlasRp->frameLastTouched : 0u),
              // blasUpd advances only on a REAL geometry change (kUpdateBVH /
              // KBuildBVH), so blasTouch==f with blasUpd lagging is the normal
              // static-mesh case, not a defect. Printed so the two are never
              // confused — blasUpd is NOT a draw-arrival signal.
              " blasUpd=", (pBlasRp != nullptr ? pBlasRp->frameLastUpdated : 0u),
              // How many instances still hold this geometry. If drew=1 and a
              // sibling exists, the arriving draw went to a DIFFERENT instance
              // than the one being reaped — that is the matching failure made
              // visible, and linked= names how many candidates were in play.
              " linked=", (pBlasRp != nullptr ? pBlasRp->getLinkedInstances().size() : 0u),
              " lastUpd=", pInstance->m_frameLastUpdated,
              " age=", (currentFrame - pInstance->m_frameLastUpdated),
              " keepN=", instanceKeepN,
              " ignAC=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
              " inFrustum=", (pInstance->m_isInsideFrustum ? 1 : 0),
              " marked=", (clauseMarked ? 1 : 0),
              " pos=(", posRp.x, ",", posRp.y, ",", posRp.z, ")"));
          }
        }

        // NV-DXVK [InstReapWave]: UNGATED mass-removal detail. The view-2 flip
        // is a one-frame mass lifetime-GC (f=662: 566 removed, f=663: 322 —
        // ~60% of the scene re-keyed) but [InstReap] above is gated on the
        // engine-hook capture counter, which only crosses 16 AFTER the flip
        // (the hook feed engages AT the flip — fanoutFpAddr first capture ==
        // flip frame). This logs removals #65..96 of any pass that removes
        // >64 instances, no gate: zero noise in steady state (passes remove
        // 0-47), 32 named victims at the wave. vs/propId/pos identify the
        // population (dome? structure? camera-relative recon draws?).
        if (probeRemovedThisPass > 64u && probeRemovedThisPass <= 96u) {
          // unlinked-for-GC: getBlas() is an ERASED entry and a null check cannot
          // see that. Null the local so the checks below work. [ReapBlasUAF]
          const BlasEntry* pBlasWv = pInstance->isUnlinkedForGC() ? nullptr : pInstance->getBlas();
          const XXH64_hash_t vsWv = (pBlasWv != nullptr)
            ? pBlasWv->input.getTransformData().vertexShaderHash : 0ull;
          const uint64_t matWv = (pBlasWv != nullptr)
            ? static_cast<uint64_t>(pBlasWv->input.getMaterialData().getHash()) : 0ull;
          const uint32_t vertsWv = (pBlasWv != nullptr)
            ? pBlasWv->modifiedGeometryData.vertexCount : 0u;
          const Vector3 posWv = pInstance->getWorldPosition();
          Logger::warn(str::format(
            "[InstReapWave] f=", currentFrame,
            " n=", probeRemovedThisPass,
            " vs=0x", std::hex, static_cast<uint64_t>(vsWv),
            " mat=0x", matWv, std::dec,
            " v=", vertsWv, std::hex,
            " propId=0x", static_cast<uint64_t>(pInstance->m_stablePropId), std::dec,
            " lastUpd=", pInstance->m_frameLastUpdated,
            " keepN=", instanceKeepN,
            " ignAC=", (pInstance->testCategoryFlags(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
            " pos=(", posWv.x, ",", posWv.y, ",", posWv.z, ")"));
        }

        // Note: Pop and swap for performance, index not incremented to process swapped instance on next iteration
        removeInstance(pInstance);

        // NOTE: pInstance is now the (previously) last element
        std::swap(pInstance, m_instances.back());

        m_instances[i]->m_instanceVectorId = i;

        delete m_instances.back();

        // Remove the last element
        m_instances.pop_back();
        continue;
      }
      probeKeptThisPass += 1;
      ++i;
    }

    // [GcExit]: summarize this GC pass. If probeRemovedThisPass > 0 but
    // probeRemovedMarked == 0 && probeRemovedLifetime == 0, removeInstance
    // must be running from somewhere ELSE — telling us to look at another
    // call site.
    Logger::info(str::format(
      "[GcExit] f=", probeFrame,
      " kept=", probeKeptThisPass,
      " removed=", probeRemovedThisPass,
      " viaMarked=", probeRemovedMarked,
      " viaLifetime=", probeRemovedLifetime,
      " viaForce=", probeRemovedForce));

    // NV-DXVK [ReapJoin] — one line per frame, the decisive readout.
    //
    //   respawn=N  reaps where a sibling instance was CREATED this frame on the
    //              same geometry => the draw arrived and dedup spent it on a new
    //              id instead of this one. MATCHING BUG (ours).
    //   starved=N  reaps with no replacement created => the geometry received
    //              fewer draws than it had instances, a submission gap upstream
    //              of Remix.
    //
    // Mixed => split by vs= in [InstReap] and treat the two populations
    // separately; do not average them.
    //
    // Emitted on every GC pass including empty ones: a frame with removed=0
    // is data (it says the churn stopped), and a probe that only prints when
    // it has something to say cannot show you an absence.
    //
    // Cost is two increments per reap plus one line per frame — safe to leave
    // on. It is NOT gated on the engine-hook capture counter, unlike the
    // [InstReap] detail lines, so the totals cover every reap in the pass.
    if (probeRemovedThisPass > 0 || probeKeptThisPass > 0) {
      Logger::info(str::format(
        "[ReapJoin] f=", probeFrame,
        " removed=", probeRemovedThisPass,
        " respawn=", probeReapRespawn,
        " starved=", probeReapStarved,
        " pctRespawn=", (probeRemovedThisPass > 0
          ? (100u * probeReapRespawn) / probeRemovedThisPass : 0u),
        " kept=", probeKeptThisPass,
        " live=", static_cast<uint32_t>(m_instances.size())));
    }

    // NV-DXVK [ResidentScene]: LRU / invalidated-record sweep, and the stats
    // line. Placed AFTER the reap so the record store never holds a pointer to
    // an instance this pass has already deleted -- removeInstance invalidated
    // each one on the way out, and this is where the husks are erased.
    {
      m_residentScene.onFrameEnd(probeFrame);

      if (RtxOptions::ResidentScene::logStats()) {
        // EVERY 10 FRAMES, and the counters are reset with each line so each is
        // a per-window rate rather than a running total. The thing being read
        // here is a SHAPE over time (does `records` plateau?), and a coarse
        // throttle hides the frames where it steps.
        static uint32_t sRsLastLogFrame = 0u;
        if (probeFrame - sRsLastLogFrame >= 10u) {
          sRsLastLogFrame = probeFrame;
          const ResidentScene::Stats& rs = m_residentScene.stats();
          Logger::warn(str::format(
            "[ResidentScene] f=", probeFrame,
            // hold = the keep is armed and the skip is not, which is what
            // verify now means. It is no longer a mode in which residency does
            // nothing, so calling it "verify" would misreport what is running.
            " mode=", (RtxOptions::ResidentScene::enable()
                        ? (RtxOptions::ResidentScene::verify() ? "hold" : "live")
                        : "off"),
            " records=", static_cast<uint32_t>(m_residentScene.size()),
            " built=", rs.built,
            // touched= is the number of draws the skip was ACTUALLY TAKEN on.
            // Read it against [ResidentGate] hit= from the frame thread: a large
            // gap means the gate is predicting hits for draws that have no
            // record to serve, and missUnk below says how many.
            " touched=", rs.touched,
            " touchMiss=", rs.touchMiss,
            " missUnk=", rs.touchMissUnknown,
            " missInval=", rs.touchMissInvalid,
            // Refused because the record's instances carry per-frame work the
            // gate cannot speak for: billboards, ray portals, decals, or opacity
            // micromaps being on. Permanent for those draws, not a fault -- but
            // if this reads close to hit= then residency is being refused almost
            // everywhere and the reason is one of those four, not the key.
            " missUnsafe=", rs.touchMissUnsafe,
            " invalidated=", rs.invalidated,
            " evicted=", rs.evicted,
            // CUMULATIVE. Non-zero means maxRecords is too small for the scene,
            // which is the OPPOSITE finding from key churn and wants raising the
            // cap rather than abandoning the key.
            " wiped=", rs.wiped,
            // CUMULATIVE. Records retired because the engine freed the buffers
            // they were built from -- the signal that stops a destroyed object
            // ghosting in the ray-traced scene. Reading zero for a whole session
            // in a game with entities means the signal is not arriving, which is
            // a defect and not a quiet scene.
            // srcDied alone reads 0 for two opposite reasons; srcNotices is what
            // separates them. notices=0 means ~D3D11Buffer never runs for this
            // geometry, i.e. pooled buffers outlive the objects using them and
            // buffer death is the wrong signal for this engine. notices>0 with
            // srcDied=0 means buffers die and none belongs to a record, which is
            // a broken join. See ResidentScene::Stats.
            " srcDied=", rs.sourceDestroyed,
            " srcNotices=", rs.srcNotices,
            " srcDrained=", rs.srcDrained,
            " instStamped=", rs.instancesStamped,
            " liveInst=", static_cast<uint32_t>(m_instances.size()),
            // THE VERIFY VERDICT, and it is the gate for arming this feature.
            // predicted is how many draws the frame-thread gate said it could
            // serve from a record; FAIL is how many of those the record would
            // have got WRONG -- the full path resolved to a different instance
            // list than the record names, so the touch would have kept the
            // wrong objects alive and let the right ones retire.
            //
            // THE GATE IS realFail, NOT FAIL. noRecEmpty is the draw that
            // resolves to no instance at all: it is never filed, so the gate
            // predicts a hit on it and score() misses on it every frame for as
            // long as the level is loaded. That number is permanent, benign and
            // handled by the live path (touch finds nothing, the draw commits in
            // full), so including it means waiting for a zero that cannot
            // arrive. See ResidentScene::Stats for the full four-way split.
            //
            // realFail must read 0, not "low", across a full pitch-and-yaw
            // sweep before rtx.residentScene.verify goes off. Then read WHICH:
            // size= is the occurrence ordinal sliding under engine culling,
            // member= is the ordinal failing at its own job, and noRecLost= is a
            // record that existed and went away -- check evicted/wiped first.
            " | predicted=", rs.predicted,
            " FAIL=", rs.fail,
            " realFail=", (rs.fail - rs.failNoRecEmpty),
            " fail{noRecEmpty=", rs.failNoRecEmpty,
            " noRecLost=", rs.failNoRecLost,
            "(erased=", rs.failLostErased, " never=", rs.failLostNever, ")",
            " size=", rs.failSize,
            // over/under split the size failures by SIGN, and the sign is what
            // says whether a size failure is a defect at all: over= keeps extra
            // instances alive (the feature), under= lets instances the draw
            // resolved to retire (the defect). Both are still inside realFail
            // on purpose -- see Stats::failSizeOver.
            "(over=", rs.failSizeOver, " under=", rs.failSizeUnder, ")",
            " member=", rs.failMember, "}",
            // Outside fail{} because it is not one: same instances, different
            // order, and touch() consumes the list as a set.
            " memberPerm=", rs.memberPerm,
            // The two numbers that decide whether this is working. records must
            // PLATEAU in a stationary scene -- a rising count is trap 3 (an
            // unstable key minting a fresh record every frame) and is a hard
            // fail, not a tuning problem. touchMiss must be ~0 once the key is
            // stable; every miss is a fallback to the full path.
            " | records must plateau; missInval ~0; realFail=0 before verify goes off"));
          m_residentScene.resetStats();
        }
      }
    }

    // NV-DXVK [InstDriftProbe]: record post-GC size UNCONDITIONALLY so the
    // first gameplay GC has valid history from the last pre-gameplay GC.
    // Tracking is cheap (two uint32 writes); only the log is gated.
    sLastGcExitSize  = static_cast<uint32_t>(m_instances.size());
    sLastGcExitFrame = probeFrame;
  }

  void InstanceManager::onFrameEnd() {
    // NV-DXVK [Perf.PushInst] PHASE 2 heartbeat. Throttled to 3 s like every
    // other counter in this file.
    //
    // READ IT IN THIS ORDER, and stop at the first one that disappoints:
    //   FAIL      the only number that can veto the feature. Printed even when
    //             zero so its absence is never mistaken for the probe being off.
    //             Non-zero => fanoutRecordFingerprint is missing an input; the
    //             .FAIL line names the shader and the diverging index.
    //   predict/batches  the CEILING, and the number to read under verify.
    //             hit= counts skips actually taken, which is zero by
    //             construction while verify is on -- on the first run that made
    //             a working record and a broken one look identical. Against the
    //             measured static=97% predict/batches should be high; if it is
    //             not, the fingerprint is moving for a reason worth naming
    //             BEFORE optimising, and miss{} says which of the five causes.
    //   predictInst the instances that ceiling covers -- this is the number that
    //             maps onto [Perf.UpdInst]'s entry 2.73 + fastRet 4.15 ms,
    //             because those are per INSTANCE, not per batch.
    //   miss{sameFrame} must be 0. Non-zero means the occurrence ordinal has
    //             stopped separating sibling draws of one batch identity, which
    //             is the defect the first verify run found.
    //   evict     non-zero means the map is thrashing; raise
    //             rtx.pushInstanceRecordsMaxBatches rather than living with it.
    // NV-DXVK [Perf.PushInst] AGE SWEEP -- retirement, replacing the eviction
    // scan that used to run on insert.
    //
    // THE BOUND IS NOT ARBITRARY. A record cannot usefully outlive the instances
    // it points at, and the longest an instance can survive untouched is the
    // larger of the two keep windows -- numFramesToKeepInstances for ordinary
    // props, numFramesToKeepSubViewInstances for the IgnoreAntiCulling and
    // stable-propId classes (the conf's shadow-pass terrain note). Past that
    // point garbageCollection has certainly reaped them, every such record has
    // already been invalidated through m_batchRecordKey, and it is dead weight.
    //
    // O(records) ONCE PER FRAME, in the same frame-end pass that already walks
    // this data -- against O(records) PER INSERT for the eviction scan, paid
    // hardest during streaming when inserts are most frequent. Same asymptotic
    // work, a different multiplier, and it is the mechanism the split-transform
    // cache already uses (`[Perf.SplitXf] evict{sweeps= aged= wipes= rung=}`).
    if (RtxOptions::pushInstanceRecords() && !m_fanoutRecords.empty()) {
      const uint32_t curFrame = m_device->getCurrentFrameId();
      const uint32_t keepN = std::max(RtxOptions::numFramesToKeepInstances(),
                                      RtxOptions::numFramesToKeepSubViewInstances());
      for (auto it = m_fanoutRecords.begin(); it != m_fanoutRecords.end(); ) {
        // Guard the unsigned subtraction: frameLastServed is kInvalidFrameIndex
        // on a record that has never served, and curFrame - UINT32_MAX wraps to
        // a huge age that would retire it on its first frame alive.
        const bool aged = (it->second.frameLastServed != kInvalidFrameIndex)
                       && (curFrame > it->second.frameLastServed)
                       && (curFrame - it->second.frameLastServed > keepN);
        if (aged) {
          // Clear back-pointers before erasing: an instance whose key still
          // names an erased record would never be invalidated again, and the
          // key would collide with whatever later hashes to the same slot.
          for (RtInstance* inst : it->second.instances) {
            if (inst->m_batchRecordKey == it->first)
              inst->m_batchRecordKey = 0ull;
          }
          it = m_fanoutRecords.erase(it);
          m_piSwept += 1;
        } else {
          ++it;
        }
      }
    }

    if (RtxOptions::pushInstanceRecords()) {
      static std::chrono::steady_clock::time_point sLast {};
      const auto now = std::chrono::steady_clock::now();
      if (sLast.time_since_epoch().count() == 0) sLast = now;
      if (now - sLast >= std::chrono::seconds(3)) {
        sLast = now;
        Logger::info(str::format(
          "[Perf.PushInst] verify=", (RtxOptions::pushInstanceRecordsVerify() ? 1 : 0),
          " batches=", m_piBatches,
          " predict=", m_piPredict,
          " predictInst=", m_piPredictInst,
          " hit=", m_piHit,
          " servedInst=", m_piServedInst,
          " miss{key=", m_piMissKey,
          " input=", m_piMissInput,
          " invalid=", m_piMissInvalid,
          " sameFrame=", m_piMissSameFrame,
          " stale=", m_piMissStale,
          " guard=", m_piGuard, "}",
          " records=", m_fanoutRecords.size(),
          " swept=", m_piSwept,
          " capped=", m_piCapped,
          " FAIL=", m_piFail,
          "  | per-frame counts (batches/hit/miss/served reset each frame);"
          " records/FAIL are cumulative. FAIL must be 0 over a session including"
          " a level transition and a combat frame before"
          " rtx.pushInstanceRecordsVerify goes off."));
      }
    }

    m_viewModelCandidates.clear();
    m_playerModelInstances.clear();
    resetSurfaceIndices();
    m_billboards.clear();
    // reset decal counter
    m_decalSortOrderCounter = 0;
  }

  // NV-DXVK [fanout split]: NEVER DERIVE A PROP IDENTITY FROM ITS CURRENT
  // TRANSFORM.
  //
  // Identity here comes from the ENGINE's own per-placement HISTORY, plumbed in as
  // DrawCallTransforms::prevInstancesToObject — the previous-frame matrix the game
  // stores at bytes 48..95 of each 208-byte g_modelInst entry, alongside the
  // current one it already reads.
  //
  // The distinction that makes it work: hashing the CURRENT transform asks "what
  // is at this position", which fails the instant a prop moves. The PREVIOUS
  // transform is the engine stating where THIS prop was last frame, and an
  // instance is filed in the SpatialMap under exactly that. So a prop that moves
  // 5000 units in one frame still resolves on the exact stage, instead of falling
  // through to a nearest-neighbour search bounded by rtx.uniqueObjectDistance
  // (300 units, and a per-FRAME budget — at 12 fps a ~3750 u/s speed limit, with
  // no velocity prediction and a wrong-neighbour failure mode).
  //
  // Measured 2026-08-05: bit-exact across f13072->13073->13074 on all three
  // translation components and the full basis, and [FanoutPrev] prevHit=215-260
  // per frame against prevMiss~0 — i.e. it resolves precisely the nInst-mtxStable
  // movers it is supposed to.
  //
  // AN ENGINE HANDLE WAS TRIED HERE AND IS GONE. charIdx (the per-instance vertex
  // attribute that selects the t31 matrix) looks like a prop handle and is not:
  // [T31Struct] measured the bound t31 buffer at 2..96 entries, so charIdx is an
  // array position inside a PER-DRAW scratch buffer, and the same value in two
  // draws names two unrelated props. Live, it merged ~6000 props/frame and dropped
  // sceneInstances from ~10400 to ~8170. Nor is a handle hiding elsewhere in the
  // stream: all 208 bytes were dumped — matrix, prev matrix, tint, a constant
  // uint, baked lighting, padding. There is no id field to find.
  //
  // Composites of (per-draw base, charIdx) are also dead, and specifically CANNOT
  // BE TESTED by a distinct-count gate: charIdx is already unique within a draw,
  // so any per-draw base makes the pair unique per placement by construction and
  // scores perfectly while proving nothing about frame-to-frame persistence.
  //
  // An earlier version of this file instead derived a stablePropId from the
  // placement's world translation rounded to 1 unit. It is gone, and nothing like
  // it should come back — the 2026-08-05 key bakeoff ([FanoutSplit] mtxStable /
  // basisStable / hybStable over 300 steady-state frames) settled it three ways:
  //
  //   * Rounded position MERGED REAL PROPS. 13 distinct placements shared a
  //     rounded position while their bases differed by 90-180 degrees (maxRot
  //     1.2-2.0 on a unit basis), so one prop of each pair was silently dropped —
  //     16/frame on the dominant fanout VS plus 10/frame on the next.
  //   * Rounding BOUGHT NOTHING for it. A hybrid key of rounded translation +
  //     exact basis scored identical stability AND identical distinct-value counts
  //     to the plain full matrix on every VS. The instability it was supposed to
  //     absorb is not sub-unit jitter at all.
  //   * That instability is GENUINE MOTION. ~233 placements/frame on the dominant
  //     VS move far enough to change their rounded position too, and they are
  //     already handled correctly by findSimilarInstance's nearest-neighbour
  //     stage — this architecture's designed mechanism for moving objects, and
  //     what every non-fanout object in the scene has always relied on. Were those
  //     movers losing identity, `created` would be ~233/frame; it is 5.4, which is
  //     the genuine rate of new props entering the batch.
  //
  // Any key derived from the CURRENT transform is stuck choosing between those two
  // failures, which is why the engine's history is used instead. The current
  // transform's bytes remain the primary key for STORING — that is what keeps a
  // stationary prop's key constant, and therefore SpatialMap::move() a no-op for
  // the ~95% of the population that holds still. Chaining the stored key to the
  // history instead would re-key every instance every frame and turn ~10400
  // no-ops into ~10400 erase+inserts, which is why the history is a second LOOKUP
  // and not a new store key.
  //
  // Before reintroducing any derived id here, re-run the bakeoff. A key must win
  // on BOTH stability (stable ~= nInst - newProp) and separation (distinct ==
  // mtxDistinct), and a candidate must be scored BEFORE it goes live — the charIdx
  // regression shipped because it went live first, and its 100% idStable was
  // meaningless on a key whose value space was 0..255.
  //
  // NV-DXVK [FanoutSplit]: one line per fanout VS per frame — the readout for the
  // whole fanout-split fix.
  //
  //   nInst    placements the game submitted this frame. Churns: the handoff
  //            measured ~6 props entering and leaving per frame.
  //   live     RtInstances those placements resolved to.
  //   created  placements that had to spawn a fresh RtInstance. Raw, and on its
  //            own AMBIGUOUS — split it with the next two fields.
  //   newProp  of `created`, those whose transform was NOT submitted last frame:
  //            a prop entering the batch, or one that moved. Expected, and no
  //            identity scheme can avoid it.
  //   missProp of `created`, those whose EXACT transform WAS submitted last frame
  //            and still needed a new instance. That is a true dedup failure —
  //            the exact stage had a byte-identical key available and missed it —
  //            and it is the only part of `created` worth chasing.
  //
  //            newProp + missProp == created by construction. If they stop adding
  //            up, the membership bookkeeping is wrong, not the fix.
  //
  //   collide  two placements that resolved to one RtInstance. Under matrix-bytes
  //            keying this can ONLY happen when their transforms are byte-
  //            identical, i.e. the game listed the same placement twice, which is
  //            benign — two co-located copies of one mesh render as one. It is
  //            therefore an invariant check now rather than a defect count:
  //            [FanoutCollide] reporting sameBytes=0 would mean two DIFFERENT
  //            transforms hashed to one key, which should be impossible.
  //
  //            The identity columns that used to sit here (withEngineId /
  //            idStable / idDistinct) are gone with the charIdx plumb. The engine
  //            history is scored by [FanoutPrev] instead, which is a per-frame
  //            line rather than a per-VS one because its counters live in
  //            findSimilarInstance and have no VS context. Read the two together:
  //            prevHit should track (nInst - mtxStable) summed over the fanout
  //            VSes, since that difference IS the set of movers whose current
  //            transform cannot resolve them.
  //
  //   mtxStable / mtxDistinct
  //            placements whose composed transform was byte-identical to one this
  //            VS submitted last frame, and the distinct-transform count. Since
  //            the transform IS the dedup key, mtxStable is the exact-stage hit
  //            rate: a placement counted here is a stationary prop that will match
  //            its own instance exactly, and the shortfall is props that MOVED.
  //
  //            Measured 2026-08-05: 99.3-100% on ten of twelve fanout VSes, and
  //            95.1% on the dominant one — that shortfall is ~233 genuinely moving
  //            props per frame, which findSimilarInstance's nearest-neighbour
  //            stage matches correctly (were it not, `created` would be ~233/frame
  //            rather than 5.4). So a falling mtxStable is not automatically a
  //            defect; cross-check it against `created` before treating it as one.
  //
  //   basisStable / basisDistinct
  //            the same for the 3x3 basis alone. Kept because it localises any
  //            future regression: if mtxStable collapses, basisStable says whether
  //            the props started MOVING (basis stable, translation not) or the
  //            engine started rebuilding their ORIENTATION (basis unstable too),
  //            and those have entirely different explanations. basisDistinct is
  //            far below mtxDistinct in normal operation — many props share an
  //            orientation — so the basis alone is never an identity.
  //   mir      placements whose composed transform has NEGATIVE winding parity.
  //            Measures the one input the split changes for the backface rule:
  //            before the split every fanout instance took its parity from the
  //            batch's IDENTITY objectToWorld, i.e. always 0, so mir is exactly
  //            the number of props whose parity the split changes.
  //
  //            mir=0 => the split cannot have altered facing anywhere. (Measured:
  //            0 across all 4960 rows of the first capture.)
  //
  //            mir>0 would not mean the picture changed either: the parity enters
  //            the rendered flag twice and cancels — determineInstanceFlags sets
  //            FLIP from (drawClockwise != parity), then addBlas /
  //            addPointInstancerBlas XOR FLIP again when the instance is
  //            mirrored, and A^M^M == A. The only consumer that sees it
  //            uncancelled is RtInstance::isFrontFaceFlipped, which feeds the USD
  //            game capturer and the [TlasCensus] flip tally. So mir>0 means "N
  //            props now record their OWN winding in a capture instead of the
  //            batch's", which is the correct value, not a regression. This
  //            counter exists so that claim is measured rather than argued.
  static void logFanoutSplitFrame(uint32_t frame, uint64_t vsHash, uint32_t draws,
                                  uint32_t nInst, uint32_t live, uint32_t created,
                                  uint32_t collide, uint32_t mir,
                                  uint32_t newProp, uint32_t missProp,
                                  uint32_t mtxStable, uint32_t basisStable,
                                  uint32_t mtxDistinct, uint32_t basisDistinct,
                                  uint32_t sceneInstances) {
    Logger::info(str::format(
      "[FanoutSplit] f=", frame,
      " vs=0x", std::hex, vsHash, std::dec,
      " draws=", draws,
      " nInst=", nInst,
      " live=", live,
      " created=", created,
      " newProp=", newProp,
      " missProp=", missProp,
      " collide=", collide,
      " mir=", mir,
      " mtxStable=", mtxStable,
      " basisStable=", basisStable,
      " mtxDistinct=", mtxDistinct,
      " basisDistinct=", basisDistinct,
      " sceneInstances=", sceneInstances));
  }

  RtInstance* InstanceManager::processSceneObject(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache) {
    // One placement, so "once per draw" and "once per instance" coincide here --
    // this call is not where the 15x lives (see DrawScopedState), but the state
    // has to be built somewhere and processSceneObjectImpl must not build it.
    const DrawScopedState drawState = computeDrawScopedState(blas, drawCall, materialData);
    return processSceneObjectImpl(cameraManager, rayPortalManager, blas, drawCall, materialData,
                                  existingInstance, drawCallCache, nullptr, drawState);
  }

  // NV-DXVK [Perf.PushInst] PHASE 2. Contract and rationale on the member
  // declarations in rtx_instance_manager.h; this is the mechanics.
  //
  // THE KEY names the BATCH -- which shader, which geometry. It deliberately
  // does NOT include anything that varies frame to frame, because a key that
  // moves means the record is never found and the feature silently does
  // nothing while reporting a healthy-looking miss rate. Everything that DOES
  // vary belongs in the fingerprint instead, where a change is a miss with a
  // reason rather than a lost record.
  uint64_t InstanceManager::fanoutRecordKey(const DrawCallState& drawCall,
                                            const BlasEntry& blas) const {
    uint64_t k = static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
    k = k * 0x9E3779B97F4A7C15ull
      + static_cast<uint64_t>(blas.input.getHash(RtxOptions::geometryAssetHashRule()));
    // Never 0: 0 is the "no record" sentinel on RtInstance::m_batchRecordKey.
    return k ? k : 1ull;
  }

  // THE FINGERPRINT answers one question: would the placement loop, run now,
  // resolve to the same instances in the same order as when this record was
  // built? Everything the resolution reads has to be in here.
  //
  // WHAT IS IN, AND WHY EACH ONE:
  //   - the placement transform BYTES. This is the whole point; a prop that
  //     moved must miss. Hashed as raw bytes rather than compared per element
  //     because the SpatialMap key contract is already bit-for-bit ("reproduces
  //     last frame's composed matrix bit-for-bit"), so byte equality is exactly
  //     the right granularity -- no epsilon, deliberately.
  //   - the ENGINE HISTORY (prevInstancesToObject) bytes and its presence. The
  //     loop resolves through the history when the current transform misses, so
  //     two frames with identical current transforms but different history can
  //     land on different instances.
  //   - the base objectToWorld, which composes with every placement.
  //   - the material hash and camera type, which select the instance as surely
  //     as the transform does -- matChg reads 0% today, and hashing it is what
  //     keeps that a measurement rather than an assumption.
  //   - the placement COUNT, so a batch that grew or shrank can never match a
  //     shorter recorded list.
  uint64_t InstanceManager::fanoutRecordFingerprint(
      const DrawCallState& drawCall, const BlasEntry& blas,
      const MaterialData& materialData,
      const std::vector<Matrix4>* transforms,
      const std::vector<Matrix4>* prevTransforms) const {
    XXH64_hash_t h = 0;
    const auto& xf = drawCall.getTransformData();
    h = XXH64(&xf.objectToWorld, sizeof(xf.objectToWorld), h);
    const uint32_t n = transforms ? static_cast<uint32_t>(transforms->size()) : 0u;
    h = XXH64(&n, sizeof(n), h);
    if (n != 0u)
      h = XXH64(transforms->data(), sizeof(Matrix4) * size_t(n), h);
    const uint32_t pn = prevTransforms ? static_cast<uint32_t>(prevTransforms->size()) : 0u;
    h = XXH64(&pn, sizeof(pn), h);
    if (pn != 0u)
      h = XXH64(prevTransforms->data(), sizeof(Matrix4) * size_t(pn), h);
    const uint32_t cam = static_cast<uint32_t>(drawCall.cameraType);
    h = XXH64(&cam, sizeof(cam), h);
    const XXH64_hash_t mat = materialData.getHash();
    h = XXH64(&mat, sizeof(mat), h);
    // Category flags select instance behaviour (sub-view, anti-culling) and are
    // written onto the instance by updateInstance, so a change here is a change
    // in the output even when every matrix matches.
    const uint32_t cat = drawCall.getCategoryFlags().raw();
    h = XXH64(&cat, sizeof(cat), h);
    return static_cast<uint64_t>(h);
  }

  // O(1) and TOTAL -- see RtInstance::m_batchRecordKey. Invalidating the whole
  // record rather than erasing one element is deliberate: the recorded list is
  // only meaningful as the complete, ordered output of one placement loop, and
  // a list with a hole in it would resolve later placements onto the wrong
  // instance with full confidence. Cheap to rebuild, impossible to half-trust.
  void InstanceManager::invalidateFanoutRecordFor(const RtInstance* instance) {
    const uint64_t key = instance->m_batchRecordKey;
    if (key == 0ull)
      return;

    // Clear this instance's own back-pointer first and unconditionally: even if
    // the record is already gone it must not keep naming it, and zeroing it up
    // front is what makes the loop below skip this (dying) entry for free.
    instance->m_batchRecordKey = 0ull;

    const auto it = m_fanoutRecords.find(key);
    if (it == m_fanoutRecords.end())
      return;

    it->second.valid = false;

    // NV-DXVK [FanoutUAF, 2026-08-21]: drop the pointers NOW, not later.
    //
    // Marking the record invalid was not enough. The dying instance's pointer
    // stayed in `instances`, and TWO places walk that vector and dereference
    // every element with no validity check:
    //
    //   - the aged-record sweep in onFrameEnd, which cleared back-pointers
    //     before erasing the record;
    //   - the rebuild in processSceneObjectFanout, which clears back-pointers
    //     from the previous contents before overwriting them.
    //
    // Either one dereferences an instance destroyed since the record was built.
    // That is the AV at InstanceManager::onFrameEnd (rtx_instance_manager.cpp
    // :3465) on dxvk-cs, reading freed instance memory -- the CS thread died
    // there and the frame thread then blocked in QueryEnd, which is what the
    // freeze looked like from outside.
    //
    // Doing it here is safe in a way that doing it later is not: removeInstance
    // calls this BEFORE destroying the instance, so every pointer in the vector
    // is still live at this moment. If a sibling was already destroyed in the
    // same GC pass, its own removeInstance ran this first and emptied the
    // vector, so we return early on key == 0 and never touch it.
    //
    // Clearing the vector loses nothing: an invalid record is never served --
    // processSceneObjectFanout tests `!valid` before it looks at `instances` --
    // and the rebuild reassigns the whole list.
    for (RtInstance* other : it->second.instances) {
      if (other->m_batchRecordKey == key)
        other->m_batchRecordKey = 0ull;
    }
    it->second.instances.clear();
  }

  void InstanceManager::processSceneObjectFanout(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    DrawCallCache* drawCallCache, std::vector<RtInstance*>& out_instances) {

    out_instances.clear();

    // NV-DXVK [perf] 2026-08-07: BUILT ONCE PER DRAW, ahead of the placement loop
    // -- this is the site the 15x lives at. Every placement below used to rebuild
    // both halves of this for itself: calculateAlphaState, and a state-key gather
    // that pointer-chases five draw-scoped objects. See DrawScopedState.
    const DrawScopedState drawState = computeDrawScopedState(blas, drawCall, materialData);

    const std::vector<Matrix4>* transforms = drawCall.getTransformData().instancesToObject;
    if (transforms == nullptr || transforms->empty()) {
      // Not actually a batch — fall back to the ordinary single-instance path so
      // this entry point is safe to call unconditionally.
      RtInstance* single = processSceneObjectImpl(cameraManager, rayPortalManager, blas, drawCall,
                                                  materialData, nullptr, drawCallCache, nullptr, drawState);
      if (single != nullptr) {
        out_instances.push_back(single);
      }
      return;
    }

    const Matrix4& drawObjectToWorld = drawCall.getTransformData().objectToWorld;
    const uint32_t currentFrameIdx = m_device->getCurrentFrameId();
    const uint64_t vsHash = static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
    out_instances.reserve(transforms->size());

    // NV-DXVK [perf] 2026-08-07. Read ONCE per draw, not per placement.
    // Gates the per-placement health probes below. See the block at the mtxHash /
    // basisHash computation for why they had to be gated: they are the wrapper's
    // copy of the [PassCensus]/[InstCounts] finding, and they are the reason
    // [Perf.SceneObj]'s four buckets summed ~5.7 ms short of [ProcDCS] instMs --
    // the guard brackets processSceneObjectImpl and never enters this function.
    const bool fanoutStats = RtxOptions::logFanoutSplitStats();

    // The engine's per-placement HISTORY. Same re-check for the same reason: the
    // producer already length-gates it, and a desynced history would resolve a
    // placement onto a different prop's instance with full confidence.
    const std::vector<Matrix4>* prevTransforms = drawCall.getTransformData().prevInstancesToObject;
    if (prevTransforms != nullptr && prevTransforms->size() != transforms->size()) {
      ONCE(Logger::warn(str::format(
        "[FanoutSplit] prevInstancesToObject/instancesToObject length mismatch (",
        prevTransforms->size(), " vs ", transforms->size(),
        ") — ignoring engine history, keying on current transform only.")));
      prevTransforms = nullptr;
    }

    uint32_t created = 0;     // placements that had to spawn a fresh RtInstance
    uint32_t collide = 0;     // placements that landed on an instance already
                              // resolved earlier in THIS batch => two props share
                              // a propId and are merging into one object
    uint32_t mirrored = 0;    // placements whose composed transform has negative
                              // winding parity — see logFanoutSplitFrame
    uint32_t newProp = 0;     // created, and this transform was NOT submitted last
                              // frame: a new prop, or one that moved
    uint32_t missProp = 0;    // created even though this exact transform WAS
                              // submitted last frame — a true dedup failure
    uint32_t mtxStable = 0;   // placements whose composed transform was
                              // byte-identical to one submitted last frame
    uint32_t basisStable = 0; // ...whose 3x3 basis alone was

    // Per-VS state carried across frames. Holds both the aggregate counters and
    // the propId membership sets that separate genuine batch churn from residual
    // dedup misses (see the newProp/missProp notes on logFanoutSplitFrame).
    struct FsState {
      uint32_t frame = 0xFFFFFFFFu;
      uint32_t draws = 0, nInst = 0, live = 0, created = 0, collide = 0,
               mir = 0, newProp = 0, missProp = 0,
               mtxStable = 0, basisStable = 0;
      // Composed transforms this VS submitted in `frame`, and in the frame before
      // it. `prev` is the last frame this VS was SEEN in, not literally frame-1: a
      // VS that skips a frame entirely makes no call here, so nothing rolls over
      // and the comparison stays against its own last submission. That is the
      // right baseline for "did this placement exist a moment ago".
      std::unordered_set<uint64_t> cur;
      std::unordered_set<uint64_t> prev;
      // The same membership question asked of the 3x3 basis alone, which isolates
      // WHERE any future instability lives — see logFanoutSplitFrame.
      std::unordered_set<uint64_t> curBasis;
      std::unordered_set<uint64_t> prevBasis;
    };
    static thread_local std::unordered_map<uint64_t, FsState> sFs;
    FsState& st = sFs[vsHash];

    // Frame rollover happens HERE, before the placement loop, so the loop can test
    // each propId against the previous frame's set. (Doing it after the loop, as
    // a pure aggregate would, leaves `prev` holding this frame's own inserts.)
    if (st.frame != currentFrameIdx) {
      // NV-DXVK [perf] 2026-08-07: with fanoutStats off the counters below are not
      // populated, so emitting the line would print zeroes and read as a collapse.
      if (fanoutStats && st.frame != 0xFFFFFFFFu && st.draws > 0) {
        logFanoutSplitFrame(st.frame, vsHash, st.draws, st.nInst, st.live, st.created,
                            st.collide, st.mir, st.newProp, st.missProp,
                            st.mtxStable, st.basisStable,
                            static_cast<uint32_t>(st.cur.size()),
                            static_cast<uint32_t>(st.curBasis.size()),
                            static_cast<uint32_t>(m_instances.size()));
      }
      // NV-DXVK [fanout prev-transform identity]: one verdict line per frame.
      // Emitted from the first VS to roll over into the new frame, because the
      // counters are global across all fanout VSes rather than per-VS (they are
      // incremented inside findSimilarInstance, which has no VS context).
      {
        static uint32_t sPrevStatFrame = 0xFFFFFFFFu;
        if (sPrevStatFrame != currentFrameIdx) {
          sPrevStatFrame = currentFrameIdx;
          const uint32_t hits = m_fanoutPrevHitCount.exchange(0, std::memory_order_relaxed);
          const uint32_t misses = m_fanoutPrevMissCount.exchange(0, std::memory_order_relaxed);
          const uint32_t nulls = m_fanoutPrevNullCount.exchange(0, std::memory_order_relaxed);
          if (hits != 0u || misses != 0u || nulls != 0u) {
            // NV-DXVK [perf] 2026-08-07: the ~200-character explanation that used
            // to be concatenated into every one of these lines now lives here.
            // This emits ONCE PER FRAME (1054 lines in a 100 s capture), and the
            // fixed prose was the overwhelming majority of each line's bytes --
            // paid through the mutexed flush-per-line Logger, forever, to restate
            // something that never changes. Reference text, read it here:
            //   prevHit  the count of placements the ENGINE HISTORY resolved after
            //            the current transform missed. Compare it against
            //            nInst-mtxStable in [FanoutSplit] (rtx.logFanoutSplitStats).
            //   prevMiss falls through to the nearest search exactly as before.
            //   prevNull the exact stage missed and the draw carried NO history
            //            to retry with, so the retry never ran. High here means
            //            the plumb does not reach this draw at all, which is a
            //            different fault from a history that misses.
            Logger::info(str::format(
              "[FanoutPrev] f=", st.frame,
              " prevHit=", hits,
              " prevMiss=", misses,
              " prevNull=", nulls));
          }
          // NV-DXVK [MapLedger]: the write-side census for the SAME frame,
          // emitted here so prevHit/prevMiss sit next to it in the log.
          //
          // That adjacency is the point. 82% of this session's history misses
          // (27,319 of 33,214) fall on 124 "flip" frames where prevHit drops to
          // 0 and the entire population misses at once, against 5,895 spread
          // over 605 ordinary frames. An average over both is an average over
          // two different phenomena — §5.4's 643 stationary misses were drawn
          // from exactly that mixture — and only 9 of 162 [RsFailMember] frames
          // are flip frames, so the failure still open is mostly NOT the flip.
          // Reading prevHit on the neighbouring line separates them for free;
          // any aggregate taken without that split is trap §6.2 repeated.
          //
          // miss= must equal prevMiss above. A disagreement means a miss took a
          // path that skipped the census, and every ratio below it is then
          // measured against the wrong denominator.
          {
            const uint32_t lmiss   = m_mapLedgerAgg.miss.exchange(0, std::memory_order_relaxed);
            const uint32_t lsame   = m_mapLedgerAgg.sameBytes.exchange(0, std::memory_order_relaxed);
            const uint32_t sNone   = m_mapLedgerAgg.sameNone.exchange(0, std::memory_order_relaxed);
            const uint32_t sIns    = m_mapLedgerAgg.sameIns.exchange(0, std::memory_order_relaxed);
            const uint32_t sRefile = m_mapLedgerAgg.sameRefile.exchange(0, std::memory_order_relaxed);
            const uint32_t sErase  = m_mapLedgerAgg.sameErase.exchange(0, std::memory_order_relaxed);
            const uint32_t dNone   = m_mapLedgerAgg.diffNone.exchange(0, std::memory_order_relaxed);
            const uint32_t dIns    = m_mapLedgerAgg.diffIns.exchange(0, std::memory_order_relaxed);
            const uint32_t dRefile = m_mapLedgerAgg.diffRefile.exchange(0, std::memory_order_relaxed);
            const uint32_t dErase  = m_mapLedgerAgg.diffErase.exchange(0, std::memory_order_relaxed);
            const uint32_t levict  = m_mapLedgerAgg.evicted.exchange(0, std::memory_order_relaxed);
            if (lmiss != 0u) {
              Logger::info(str::format(
                "[MapLedger] f=", st.frame,
                " miss=", lmiss,
                " sameBytes=", lsame,
                " same{none=", sNone, " ins=", sIns, " refile=", sRefile, " erase=", sErase, "}",
                " diff{none=", dNone, " ins=", dIns, " refile=", dRefile, " erase=", dErase, "}",
                // Non-zero invalidates every none= on this line: a displaced
                // record is indistinguishable from a key never written.
                " evicted=", levict));
            }
          }
        }
      }
      st.prev = std::move(st.cur);
      st.cur.clear();
      st.prevBasis = std::move(st.curBasis);
      st.curBasis.clear();
      st.draws = 0; st.nInst = 0; st.live = 0; st.created = 0; st.collide = 0;
      st.mir = 0; st.newProp = 0; st.missProp = 0;
      st.mtxStable = 0; st.basisStable = 0;
      st.frame = currentFrameIdx;
    }

    // ================================================================
    // NV-DXVK [Perf.PushInst] PHASE 2 -- the record lookup and the skip.
    // Contract and rationale on the member declarations in the header.
    //
    // PLACED HERE, BELOW THE ROLLOVER, ON PURPOSE. The rollover must run for
    // every batch including a skipped one: it moves the per-VS propId
    // membership sets from `cur` to `prev`, and a frame that skipped the loop
    // still has to leave the next frame comparing against the right baseline.
    // Above it, this block would also reference prevTransforms and `st` before
    // either is declared.
    const bool piOn     = RtxOptions::pushInstanceRecords();
    // NV-DXVK [perf] 2026-08-14: HOISTED above the Phase 2 block. It used to be
    // declared just above the placement loop, which was fine while the skip only
    // stamped instances. The skip now composes placement matrices too (it calls
    // updateInstance), so both paths need it. Same value, still once per draw.
    const bool fanoutBaseIsIdentity = isIdentityExact(drawObjectToWorld);

    const bool piVerify = RtxOptions::pushInstanceRecordsVerify();
    uint64_t piKey = 0ull, piPrint = 0ull;
    FanoutBatchRecord* piRec = nullptr;
    bool piPredictHit = false;
    if (piOn) {
      // NV-DXVK [Phase2b]: m_fanoutRecords / m_fanoutOrdinals / the m_pi*
      // counters are shared unlocked members; under the sharded phase two
      // fanout draws in different shards reach this bookkeeping concurrently.
      // One lock per fanout DRAW (tens per frame), never per placement — the
      // placement loop below runs unlocked.
      std::unique_lock<std::mutex> piBookLock;
      if (inShardedInstancePhase()) {
        piBookLock = std::unique_lock<std::mutex>(m_shardEscapeMutex);
      }
      if (m_piFrame != currentFrameIdx) {
        m_piFrame = currentFrameIdx;
        m_piBatches = m_piHit = m_piMissKey = m_piMissInput = 0;
        m_piMissInvalid = m_piServedInst = 0;
        m_piPredict = m_piMissSameFrame = m_piPredictInst = 0;
        m_piGuard = m_piCapped = m_piMissStale = 0;  // m_piSwept cumulative, like FAIL
        // The ordinal is per FRAME: draw N of this frame must line up with draw
        // N of the last one, so the count restarts here and nowhere else.
        m_fanoutOrdinals.clear();
      }
      m_piBatches += 1;
      // Batch identity, then WHICH occurrence of it this is. See m_fanoutOrdinals.
      const uint64_t piBaseKey = fanoutRecordKey(drawCall, blas);
      const uint32_t piOrd = m_fanoutOrdinals[piBaseKey]++;
      piKey = piBaseKey * 0x9E3779B97F4A7C15ull + (static_cast<uint64_t>(piOrd) + 1ull);
      if (piKey == 0ull)
        piKey = 1ull;   // 0 is the "no record" sentinel on m_batchRecordKey
      piPrint = fanoutRecordFingerprint(drawCall, blas, materialData, transforms, prevTransforms);
      const auto it = m_fanoutRecords.find(piKey);
      if (it == m_fanoutRecords.end()) {
        m_piMissKey += 1;
      } else if (!it->second.valid) {
        m_piMissInvalid += 1;
        piRec = &it->second;
      } else if (it->second.frameLastBuilt == currentFrameIdx) {
        // INDEPENDENT BACKSTOP, and it stays even though the ordinal should make
        // it unreachable. A record is a statement about a PREVIOUS frame; one
        // built during this frame describes a sibling draw, and serving it is
        // precisely the bug the first verify run caught (FAIL=3367, every line
        // builtFrame == f). If this counter ever reads non-zero the ordinal has
        // stopped separating siblings and the reason wants finding before
        // anything is skipped -- so it is counted, not silently swallowed.
        m_piMissSameFrame += 1;
        piRec = &it->second;
      } else if (it->second.frameLastServed + 1u != currentFrameIdx) {
        // A PREDICTION HAS A SHELF LIFE OF EXACTLY ONE FRAME.
        //
        // NV-DXVK 2026-08-14: this tests frameLastSERVED, not frameLastBuilt.
        // It was frameLastBuilt, and that was a defect that only existed while
        // verify was on. frameLastBuilt is written by the REFRESH block at the
        // bottom of the FULL path; the skip returns long before it. So once
        // verify went off:
        //   f    full path, built=f, served=f
        //   f+1  built+1 == f+1  -> SKIP, built STAYS f
        //   f+2  built+1 != f+2  -> stale miss, full path, built=f+2
        // i.e. a batch submitted every single frame could be served at most
        // every OTHER frame, forever. Under verify the full path runs every
        // frame and refreshes built every frame, so the bound was always
        // satisfied and predict read 89% -- the measurement that made this look
        // healthy was the thing preventing the bug from appearing.
        //
        // frameLastServed is written by BOTH paths (here and in REFRESH), so it
        // means "the frame this record last produced the answer", which is
        // exactly what a one-frame adjacency bound wants to test. frameLastBuilt
        // keeps its original meaning -- the last frame we actually RESOLVED --
        // which is what makes it worth printing in [Perf.PushInst.FAIL].
        //
        // The 2b protection is unchanged: a batch that stops being submitted
        // stops being served, served falls behind, and the record goes stale on
        // exactly the frame it used to.
        //
        // THE MEASUREMENT THAT FORCED THIS. The second verify run left FAIL=7,
        // flat, with ALL FOUR discriminator bits zero on every one -- so the
        // inputs genuinely were identical and no instance had been reaped,
        // relinked or claimed within the frame. What every failure DID share was
        // a gap: builtFrame lagged the current frame by 3 to 11, while
        // recLastUpd sat at f-1. The batch had skipped frames, and during the
        // gap OTHER draws went on claiming and updating its instances, so the
        // resolution mapping drifted underneath a record that still looked
        // perfectly valid.
        //
        // The fingerprint cannot cover this and no amount of hashing will: what
        // moved was not an input to this batch, it was the global resolution
        // state between two of its appearances. Bounding the staleness to a
        // single frame is what makes the record's claim true again -- "these
        // inputs resolved to these instances ONE frame ago" is verifiable;
        // "...eleven frames ago" is not.
        //
        // Costs almost nothing: a batch submitted every frame always has a gap
        // of 1. Only intermittently-submitted batches lose acceleration, and
        // those are exactly the ones this refuses to trust. VS_2947c6 -- the
        // shadow-pass terrain the conf documents as submitted only when TF2's
        // spot-shadow pass runs -- is three of the seven.
        m_piMissStale += 1;
        piRec = &it->second;
      } else if (it->second.inputHash != piPrint) {
        m_piMissInput += 1;
        piRec = &it->second;
      } else if (it->second.instances.size() != transforms->size()) {
        // Belt and braces: the count is already in the fingerprint, so this can
        // only fire on a hash collision. Counted as an input miss rather than
        // trusted, because the alternative is indexing past a shorter list.
        m_piMissInput += 1;
        piRec = &it->second;
      } else {
        piRec = &it->second;
        piPredictHit = true;
        // Counted whether or not the skip is taken, so verify runs report the
        // ceiling instead of the constant zero that hit= reads under verify.
        m_piPredict += 1;
        m_piPredictInst += static_cast<uint32_t>(piRec->instances.size());
      }

      // VALIDATE THE RECORDED INSTANCES BEFORE TRUSTING THEM.
      //
      // The fingerprint answers "are the INPUTS the same". It cannot answer "are
      // the instances those inputs resolved to last frame still the ones this
      // draw would resolve to", because resolution reads state this batch does
      // not own -- the spatial map and draw-call cache are global, and an
      // earlier draw in THIS frame can legitimately claim an instance that this
      // batch used to get. No per-batch hash can see that.
      //
      // So do not try to hash it. CHECK IT. Three loads per instance against a
      // full find + update, and every failure downgrades a wrong answer into an
      // ordinary cache miss, which is always safe:
      //   - still linked to THIS geometry (an instance can be relinked)
      //   - not already stamped this frame => nobody else has claimed it
      //   - not marked for collection
      // This is the same shape as the split cache validating vkPath{agree} and
      // the memo requiring safeToDefer: prove the served answer, do not assume
      // the key was sufficient.
      if (piPredictHit) {
        for (const RtInstance* inst : piRec->instances) {
          if (inst->m_linkedBlas != &blas
              || inst->m_frameLastUpdated == currentFrameIdx
              || inst->m_isMarkedForGC) {
            piPredictHit = false;
            m_piGuard += 1;
            break;
          }
        }
      }
    }

    // THE SKIP. Only when the record predicted a hit AND verify is off.
    //
    // WHAT IT SKIPS -- AND WHAT IT DELIBERATELY DOES NOT.
    //
    // It skips processSceneObjectImpl, and the only thing in there worth
    // skipping is findSimilarInstance: the RESOLVE. That is what this phase
    // owns, it is what `entry` measures, and it is ~2.73 ms.
    //
    // It does NOT skip updateInstance. The first version of this block did --
    // a bulk setFrameLastUpdated + registerCamera + push, on the argument that
    // garbageCollection reaps on m_frameLastUpdated and nothing else, so a
    // stamp is sufficient. That argument is true and it is not the whole
    // claim. The stamp keeps the instance ALIVE. It does not keep it
    // DESCRIBED. That version shipped visible flicker.
    //
    // updateInstance's own fast path (drawState.fastPathAllowed) enumerates
    // what still has to happen every frame even when every input is
    // unchanged, and the list is not optional: the dynamic buffer rebind
    // (slice renaming -- a d3d11 dynamic VB is a DIFFERENT allocation each
    // frame, so an instance that is not rebound reads last frame's renamed
    // slice), isInsideFrustum, the prev-transform advance, the picking value,
    // textureTransform/clipPlane, the m_isHidden re-promotion, and the
    // non-skippable OMM event handlers.
    //
    // So the division of labour is:
    //     THIS record  skips the FIND.
    //     fastPathAllowed  skips the UPDATE.
    // The second one is already tuned and already validated by m_instStateKey
    // + m_fastDrawBits + an objectToWorld byte-compare + the SpatialMap sync
    // proof, and it is what `fastRet` measures. fastRet is NOT waste waiting to
    // be deleted -- it is the per-frame work that cannot be deleted. Phase 2's
    // honest ceiling is `entry`, not `entry + fastRet`, and the 6.88 ms figure
    // in the plan double-counts work that must happen.
    if (piPredictHit && !piVerify) {
      for (size_t placement = 0; placement < piRec->instances.size(); ++placement) {
        RtInstance* inst = piRec->instances[placement];

        // The SAME composition the full loop performs, and it has to be the
        // same one bit-for-bit: updateInstance byte-compares this against
        // surface.objectToWorld to decide its own fast path, so a differently
        // rounded product here would silently demote every instance to the
        // slow path and cost more than the find ever saved.
        // Safe to index transforms by the instance index: the record is only
        // stored when out_instances.size() == transforms->size() (see REFRESH),
        // and a size mismatch is rejected as an input miss above.
        FanoutSplit split;
        if (fanoutBaseIsIdentity) {
          split.objectToWorld = (*transforms)[placement];
        } else {
          split.objectToWorld = drawObjectToWorld * (*transforms)[placement];
        }
        if (prevTransforms != nullptr) {
          if (fanoutBaseIsIdentity) {
            split.prevObjectToWorld = (*prevTransforms)[placement];
          } else {
            split.prevObjectToWorld = drawObjectToWorld * (*prevTransforms)[placement];
          }
          split.hasPrevObjectToWorld = true;
        }

        // NV-DXVK [FastPathOrder] 2026-08-25: IS THIS INSTANCE THE ONE THAT
        // BELONGS AT THIS PLACEMENT?
        //
        // This line pairs piRec->instances[placement] with transforms[placement]
        // positionally. The safety note above establishes that the two lists are
        // the same LENGTH; it does not establish that index i names the same
        // object in both. Those are different claims and only the first is
        // checked.
        //
        // score() measures how often they disagree and calls it a pass:
        // m_stats.memberPerm counts "same instances, different order" and
        // returns true, on the stated grounds that "the list is consumed as a
        // SET" by touch(). That is true OF touch(). This site is the other
        // consumer, and it is positional. [ResidentScene] reads memberPerm 35.6
        // per 10-frame window, non-zero in 91% of windows, and 99 windows carry
        // it while realFail=0 -- so the acceptance test passes precisely while
        // the property this line depends on is violated.
        //
        // THE MEASUREMENT. inst->surface.objectToWorld still holds LAST frame's
        // transform here, because updateInstance has not run yet. For a
        // stationary prop -- and every sampled miss reads moved=0 with
        // sameBytes=1, so this scene's fanout props are stationary -- the
        // correctly matched instance sees a delta of exactly zero. A mismatched
        // one sees the distance to whichever placement it was given instead.
        //
        // PRE-REGISTERED READING:
        //   d0 ~ all           the pairing is correct and this whole line of
        //                      inquiry is dead. memberPerm is then genuinely
        //                      benign and the churn comes from somewhere else.
        //   dFar non-zero      instances are being handed another placement's
        //                      transform. That is a wrong position AND a wrong
        //                      motion vector -- surface.prevObjectToWorld will
        //                      name the previous placement -- so it is a visible
        //                      defect, not a bookkeeping one, and the fix is to
        //                      make the record's order authoritative or to stop
        //                      consuming it positionally.
        //   dNear non-zero     genuine small motion; compare against dFar rather
        //                      than reading either alone.
        //
        // The bucket boundaries are deliberately raw distances, not a threshold
        // on a derived score: this is the first look at this quantity and a
        // classifier here would hide the distribution that decides it.
        {
          struct FastOrderAgg {
            std::atomic<uint32_t> frame { 0u };
            std::atomic<uint32_t> n     { 0 };
            std::atomic<uint32_t> d0    { 0 };   // exactly byte-identical
            std::atomic<uint32_t> dTiny { 0 };   // < 1 unit
            std::atomic<uint32_t> dNear { 0 };   // 1 .. 100
            std::atomic<uint32_t> dFar  { 0 };   // > 100
            std::atomic<uint32_t> maxD  { 0 };   // largest delta, whole units
          };
          static FastOrderAgg sFastOrder;
          const uint32_t nowFrame = currentFrameIdx;

          const Vector3 prevT = inst->surface.objectToWorld[3].xyz();
          const Vector3 curT  = split.objectToWorld[3].xyz();
          const Vector3 dv    = curT - prevT;
          const float   d     = std::sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);

          sFastOrder.n.fetch_add(1, std::memory_order_relaxed);
          if (memcmp(inst->surface.objectToWorld.data, split.objectToWorld.data,
                     sizeof(Matrix4)) == 0) {
            sFastOrder.d0.fetch_add(1, std::memory_order_relaxed);
          } else if (d < 1.f) {
            sFastOrder.dTiny.fetch_add(1, std::memory_order_relaxed);
          } else if (d <= 100.f) {
            sFastOrder.dNear.fetch_add(1, std::memory_order_relaxed);
          } else {
            sFastOrder.dFar.fetch_add(1, std::memory_order_relaxed);
          }
          const uint32_t dWhole = static_cast<uint32_t>(d);
          uint32_t prevMax = sFastOrder.maxD.load(std::memory_order_relaxed);
          while (dWhole > prevMax
                 && !sFastOrder.maxD.compare_exchange_weak(prevMax, dWhole,
                                                           std::memory_order_relaxed)) {
          }

          // Monotonic rollover, same reason as [KeyDiverge]: the frame here is
          // shared across placements but the emitting thread is not, and a plain
          // inequality would let two frame numbers alternate.
          uint32_t seenFrame = sFastOrder.frame.load(std::memory_order_relaxed);
          if (nowFrame > seenFrame
              && sFastOrder.frame.compare_exchange_strong(seenFrame, nowFrame,
                                                          std::memory_order_relaxed)) {
            const uint32_t en = sFastOrder.n.exchange(0, std::memory_order_relaxed);
            const uint32_t e0 = sFastOrder.d0.exchange(0, std::memory_order_relaxed);
            const uint32_t et = sFastOrder.dTiny.exchange(0, std::memory_order_relaxed);
            const uint32_t ee = sFastOrder.dNear.exchange(0, std::memory_order_relaxed);
            const uint32_t ef = sFastOrder.dFar.exchange(0, std::memory_order_relaxed);
            const uint32_t em = sFastOrder.maxD.exchange(0, std::memory_order_relaxed);
            if (seenFrame != 0u && en != 0u) {
              Logger::info(str::format(
                "[FastPathOrder] f=", seenFrame,
                " n=", en, " d0=", e0, " dTiny=", et, " dNear=", ee, " dFar=", ef,
                " maxDelta=", em));
            }
          }
        }

        // Empty hint ON PURPOSE. SpatialKeyHint::isUsable() is false when
        // hash == 0, which its own definition documents as "what an instance
        // that never went through findSimilarInstance gets" -- and that is
        // precisely our situation. We did not run the find, so we have no
        // queryMatrixHash, and inventing one would file the instance in the
        // SpatialMap under a key nothing else agrees with.
        updateInstance(*inst, cameraManager, blas, drawCall, materialData,
                       drawState, &split, SpatialKeyHint{});

        out_instances.push_back(inst);
      }
      {
        // NV-DXVK [Phase2b]: same shared-member rule as the bookkeeping block.
        std::unique_lock<std::mutex> piServeLock;
        if (inShardedInstancePhase()) {
          piServeLock = std::unique_lock<std::mutex>(m_shardEscapeMutex);
        }
        piRec->frameLastServed = currentFrameIdx;
        m_piHit += 1;
        m_piServedInst += static_cast<uint32_t>(piRec->instances.size());
      }
      st.draws += 1;
      st.nInst += static_cast<uint32_t>(transforms->size());
      st.live  += static_cast<uint32_t>(out_instances.size());
      return;
    }
    // ================================================================

    // Parallel to out_instances: the composed transform and identity each accepted
    // placement was resolved with. Kept because updateInstance has already
    // overwritten surface.objectToWorld by the time a later placement is found to
    // collide with an earlier one, so the instance can no longer report where the
    // FIRST of the two stood. Without this the collision dump would compare a
    // placement against itself.
    // NV-DXVK [perf] 2026-08-07: the parallel sPlacedMtxHash array is gone. It
    // existed so the collide dump could print the FIRST placement's hash, which
    // cost an XXH64 per placement to fill an array read a few dozen times a
    // session. The dump now hashes sPlacedO2w[duplicateOf] when it actually fires.
    static thread_local std::vector<Matrix4> sPlacedO2w;
    sPlacedO2w.clear();

    // NV-DXVK [perf] 2026-08-08e (CS pole, fanout wrapper): hoist the identity
    // test the composition comment below already documents. drawObjectToWorld
    // is identity on the path-10 fanout route, yet the loop paid a full
    // Matrix4 multiply per placement -- and a SECOND one per placement when
    // the engine history is present -- ~14.6k-29k 64-float multiplies per
    // frame on the dxvk-cs thread to multiply by one. Identity => the product
    // IS the placement matrix, bit-for-bit (copy, not recompute), so the
    // SpatialMap key contract ("reproduces last frame's composed matrix
    // bit-for-bit") is preserved exactly. A non-identity base (if that path
    // ever appears) takes the original multiply unchanged.
    // (fanoutBaseIsIdentity is now declared above the Phase 2 block -- the skip
    // path composes the same matrices and must use the same rule.)

    for (size_t placement = 0; placement < transforms->size(); ++placement) {
      FanoutSplit split;
      // Same composition RtSurface::writeGPUData applies for point-instancer slot
      // i, so the prop renders exactly where it did before the split.
      if (fanoutBaseIsIdentity) {
        split.objectToWorld = (*transforms)[placement];
      } else {
        split.objectToWorld = drawObjectToWorld * (*transforms)[placement];
      }

      // Composed through the SAME drawObjectToWorld as the current transform, so
      // that for a stationary prop the product reproduces last frame's composed
      // matrix bit-for-bit and its hash is literally the key the instance was
      // filed under. (drawObjectToWorld is identity on this path — see the
      // path-10 site in d3d11_rtx — but composing it explicitly keeps the two
      // matrices in the same space if that ever changes.)
      if (prevTransforms != nullptr) {
        if (fanoutBaseIsIdentity) {
          split.prevObjectToWorld = (*prevTransforms)[placement];
        } else {
          split.prevObjectToWorld = drawObjectToWorld * (*prevTransforms)[placement];
        }
        split.hasPrevObjectToWorld = true;
      }

      // NV-DXVK [perf] 2026-08-07 -- EVERYTHING FROM HERE TO THE END OF THE HEALTH
      // PROBES IS NOW GATED, and it was the wrapper's ~5.7 ms.
      // This ran per PLACEMENT (~14,600/frame: 15,489 processSceneObjectImpl calls
      // against ~1,060 draws) and cost, every placement, every frame: two XXH64s
      // (64 B and 36 B), two unordered_set lookups and two unordered_set INSERTS --
      // ~29,000 hash-set operations per frame, whose nodes are then freed again at
      // the next frame's rollover.
      // Nothing the renderer reads depends on any of it. mirrored / mtxStable /
      // basisStable / newProp / missProp exist only to fill [FanoutSplit], which is
      // on the logDenyTags list, so the line they feed is formatted and discarded.
      // Identical shape to [PassCensus] and [InstCounts] in processSceneObjectImpl,
      // just outside the bracket [Perf.SceneObj] measures -- which is exactly why
      // that probe's buckets summed short of [ProcDCS] instMs.
      // The collide DETECTION below is functional and stays; only the collide DUMP
      // needs mtxHash, and it recomputes it from sPlacedO2w when it fires.
      bool submittedLastFrame = false;
      if (fanoutStats) {
      if (isMirrorTransform(split.objectToWorld)) {
        ++mirrored;
      }

      // Health probes. Their job is to characterise the population: how much of it
      // is stationary (mtxStable) and whether any future instability is motion or
      // re-derived orientation (basisStable). nInst - mtxStable is the mover count
      // the engine history is expected to resolve — compare it against prevHit in
      // [FanoutPrev].
      // basisHash is kept alongside because it isolates where any future
      // instability lives: if mtxStable ever collapses, basisStable says whether
      // the engine started moving the props or started rebuilding their
      // orientation, and those have completely different explanations.
      const Matrix4& mo = split.objectToWorld;
      const float basis[9] = {
        float(mo[0][0]), float(mo[0][1]), float(mo[0][2]),
        float(mo[1][0]), float(mo[1][1]), float(mo[1][2]),
        float(mo[2][0]), float(mo[2][1]), float(mo[2][2]) };

      const uint64_t mtxHash = static_cast<uint64_t>(XXH64(&mo, sizeof(Matrix4), 0));
      const uint64_t basisHash = static_cast<uint64_t>(XXH64(basis, sizeof(basis), 0));

      // Read membership BEFORE inserting, or every placement looks like a repeat.
      submittedLastFrame = st.prev.count(mtxHash) != 0;
      st.cur.insert(mtxHash);
      if (submittedLastFrame) {
        ++mtxStable;
      }
      if (st.prevBasis.count(basisHash) != 0) {
        ++basisStable;
      }
      st.curBasis.insert(basisHash);
      }   // if (fanoutStats)

      // existingInstance is always null here: a replacement draw never carries
      // isFanoutBatch (SceneManager::drawReplacements clears it), so there is no
      // pre-resolved instance to honour.
      //
      // The instance-table size is the exact "did dedup miss" test: addInstance is
      // the only thing on this path that grows m_instances, and nothing shrinks it
      // until garbage collection. isCreatedThisFrame would over-report, because a
      // batch drawn twice in one frame would re-count every instance the first
      // pass created.
      const size_t instanceCountBefore = m_instances.size();
      RtInstance* instance = processSceneObjectImpl(cameraManager, rayPortalManager, blas, drawCall,
                                                    materialData, nullptr, drawCallCache, &split, drawState);
      // NV-DXVK [Phase2b]: this placement's find missed on the worker — record
      // the index for the ordered tail (which re-runs exactly these placements
      // sequentially) and move on. The REFRESH below self-suppresses (short
      // out_instances) and the VERIFY is gated on deferredPlacements.empty(), so
      // the record machinery never sees the deliberately-incomplete list.
      if (inShardedInstancePhase() && t_shardPhase.deferredThisDraw) {
        t_shardPhase.deferredThisDraw = false;
        t_shardPhase.info->deferredPlacements.push_back(static_cast<uint32_t>(placement));
        continue;
      }
      if (m_instances.size() > instanceCountBefore) {
        ++created;
        // The decisive split of `created`. A prop the game did not submit last
        // frame HAS to be created — that is the batch churn the handoff measured
        // (~6 props entering per frame), and it is irreducible. A prop that WAS
        // submitted last frame and still needed a new instance is a dedup failure
        // and is the only part of `created` worth chasing.
        if (submittedLastFrame) {
          ++missProp;
        } else {
          ++newProp;
        }
      }
      if (instance == nullptr) {
        continue;
      }

      // Duplicate detection. findSimilarInstance's EXACT stage
      // (getDataAtTransform) returns a hit without testing m_frameLastUpdated, so
      // two placements sharing a propId WOULD both resolve to the same instance
      // and one prop would silently disappear. Scanning what this batch has
      // already produced is exact — unlike a frameLastUpdated test, it cannot be
      // confused by a legitimate second pass of the same batch in one frame.
      // Bounded by the batch size — measured at ~19-32 placements per draw, so
      // the scan is cheap even at 252 draws/frame.
      size_t duplicateOf = SIZE_MAX;
      for (size_t s = 0; s < out_instances.size(); ++s) {
        if (out_instances[s] == instance) {
          duplicateOf = s;
          break;
        }
      }
      if (duplicateOf != SIZE_MAX) {
        ++collide;

        // NV-DXVK [FanoutCollide]: the raw pair, at full float precision.
        //
        // Now an INVARIANT CHECK rather than a defect hunt. The dedup key is the
        // composed transform's bytes, so two placements can only land on one
        // RtInstance when their transforms are byte-identical — the game listed
        // the same placement twice, and collapsing them is correct. sameBytes=0
        // here would mean two different transforms resolved to one instance,
        // which should be impossible; if it ever prints, the key is not what this
        // code believes it is.
        //
        // Kept because it also caught the defect that killed the previous keying
        // scheme: under rounded-position ids, 13 of the first 46 sightings had
        // maxRot 1.2-2.0 on a unit basis — two props at one spot, 90-180 degrees
        // apart, one of them silently dropped. Read the fields in this order:
        //
        //   sameBytes=1  the two composed matrices are byte-identical => the batch
        //                genuinely lists this placement twice. Benign: two
        //                co-located copies of the same geometry render as one.
        //                DECISIVE on its own — 33 of the first 46 sightings.
        //
        //   sameBytes=0  says only that some byte differs. It is NOT decisive and
        //                must not be read as "the props differ": memcmp separates
        //                -0.0f from +0.0f and trips on a 1-ULP difference anywhere
        //                in the matrix, neither of which means anything. The first
        //                capture had 11 rows with sameBytes=0 AND d=(0,0,0), i.e.
        //                identical translations, which memcmp cannot classify.
        //                Use the two magnitudes below instead:
        //
        //   d=(...)      translation delta, unrounded, so the rounding is visible:
        //                two props at x=100.4 and x=100.6 both round to 100 and
        //                collide despite being 0.2 units apart. |d| well under 1
        //                unit means one prop with sub-unit jitter — exactly what
        //                the rounding exists to absorb, so merging is correct.
        //
        //   maxRot=      max |Δ| over the nine 3x3 basis elements. THIS is what
        //                decides the sameBytes=0 rows:
        //                  maxRot == 0 (with d ~ 0)  -> the matrices are
        //                    numerically identical and differ only in bit pattern
        //                    (signed zero). Benign, same verdict as sameBytes=1.
        //                  maxRot small vs basisScale -> same orientation, float
        //                    noise. Benign.
        //                  maxRot comparable to basisScale -> two props at ONE
        //                    position with DIFFERENT orientation. This is what
        //                    the old rounded-position key could not distinguish;
        //                    under matrix-bytes keying it cannot occur, so seeing
        //                    it means the key regressed.
        //   basisScale=  max |element| of the first matrix's 3x3, so maxRot can be
        //                read relative to the transform's own scale rather than
        //                against an absolute threshold (these props are not all
        //                unit-scale).
        //
        // firstBasis / dupBasis are dumped whenever maxRot > 0 — the raw nine
        // numbers each, so an orientation difference can be seen rather than
        // inferred from a summary statistic.
        //
        // First occurrence per (VS, propId) only — with collide constant that is a
        // few dozen lines for the whole session, not per frame.
        {
          const Matrix4& firstO2w = sPlacedO2w[duplicateOf];
          // NV-DXVK [perf] 2026-08-07: recomputed here instead of being carried
          // per placement. This branch fires on a collision AND only on the first
          // sighting of each (VS, transform) pair, capped at 400 for the session --
          // a few dozen lines a run -- so two hashes here replace ~14,600 per frame.
          // sPlacedO2w still holds the earlier placement's matrix, so the first
          // hash is recoverable without a parallel array.
          const uint64_t mtxHash =
            static_cast<uint64_t>(XXH64(&split.objectToWorld, sizeof(Matrix4), 0));
          const uint64_t firstMtxHash =
            static_cast<uint64_t>(XXH64(&firstO2w, sizeof(Matrix4), 0));
          struct CollideKey { uint64_t vs, propId; };
          const CollideKey ck { vsHash, mtxHash };
          const uint64_t ckHash = XXH64(&ck, sizeof(ck), 0xC0111DEull);
          static std::mutex sCollideMu;
          static thread_local std::unordered_set<uint64_t> sCollideSeen;
          bool firstSighting = false;
          {
            std::lock_guard<std::mutex> g(sCollideMu);
            if (sCollideSeen.size() < 400u && sCollideSeen.insert(ckHash).second) {
              firstSighting = true;
            }
          }
          if (firstSighting) {
            const bool sameBytes =
              memcmp(&firstO2w, &split.objectToWorld, sizeof(Matrix4)) == 0;
            const float dx = float(split.objectToWorld[3][0]) - float(firstO2w[3][0]);
            const float dy = float(split.objectToWorld[3][1]) - float(firstO2w[3][1]);
            const float dz = float(split.objectToWorld[3][2]) - float(firstO2w[3][2]);

            // The numeric 3x3 comparison memcmp cannot do. Columns 0..2 are the
            // basis; column 3 is the translation, already covered by d above.
            float maxRot = 0.0f;
            float basisScale = 0.0f;
            for (int bc = 0; bc < 3; ++bc) {
              for (int br = 0; br < 3; ++br) {
                const float a = float(firstO2w[bc][br]);
                const float b = float(split.objectToWorld[bc][br]);
                maxRot = std::max(maxRot, std::abs(a - b));
                basisScale = std::max(basisScale, std::abs(a));
              }
            }

            // Raw bases only when they actually differ, so the benign majority
            // stays one readable line each.
            std::string basisDump;
            if (maxRot > 0.0f) {
              basisDump = str::format(
                " firstBasis=[",
                  float(firstO2w[0][0]), ",", float(firstO2w[0][1]), ",", float(firstO2w[0][2]), " | ",
                  float(firstO2w[1][0]), ",", float(firstO2w[1][1]), ",", float(firstO2w[1][2]), " | ",
                  float(firstO2w[2][0]), ",", float(firstO2w[2][1]), ",", float(firstO2w[2][2]), "]",
                " dupBasis=[",
                  float(split.objectToWorld[0][0]), ",", float(split.objectToWorld[0][1]), ",", float(split.objectToWorld[0][2]), " | ",
                  float(split.objectToWorld[1][0]), ",", float(split.objectToWorld[1][1]), ",", float(split.objectToWorld[1][2]), " | ",
                  float(split.objectToWorld[2][0]), ",", float(split.objectToWorld[2][1]), ",", float(split.objectToWorld[2][2]), "]");
            }

            Logger::info(str::format(
              "[FanoutCollide] f=", currentFrameIdx,
              " vs=0x", std::hex, vsHash, std::dec,
              " mtxHash=0x", std::hex, mtxHash, std::dec,
              " firstMtxHash=0x", std::hex, firstMtxHash, std::dec,
              " sameBytes=", (sameBytes ? 1 : 0),
              " maxRot=", maxRot,
              " basisScale=", basisScale,
              " nInst=", static_cast<uint32_t>(transforms->size()),
              " slotFirst=", static_cast<uint32_t>(duplicateOf),
              " firstT=(", float(firstO2w[3][0]), ",",
                           float(firstO2w[3][1]), ",",
                           float(firstO2w[3][2]), ")",
              " dupT=(", float(split.objectToWorld[3][0]), ",",
                         float(split.objectToWorld[3][1]), ",",
                         float(split.objectToWorld[3][2]), ")",
              " d=(", dx, ",", dy, ",", dz, ")",
              basisDump));
          }
        }
        continue;
      }

      out_instances.push_back(instance);
      sPlacedO2w.push_back(split.objectToWorld);
    }

    // ================================================================
    // NV-DXVK [Perf.PushInst] PHASE 2 -- VERIFY, then refresh the record.
    //
    // VERIFY IS THE POINT OF THE FIRST RUN. When the record predicted a hit we
    // ran the loop anyway; compare what it produced against what the record
    // would have served. A divergence means the fingerprint is INCOMPLETE --
    // the resolution read some input that is not hashed -- and the consequence
    // of acting on it would be an instance in the wrong place or a stale
    // instance held alive by a stamp it never earned. Same failure class, and
    // the same remedy, as memoExtractVerify's FAIL.
    // NV-DXVK [Phase2b]: with deferred placements the produced list is
    // deliberately incomplete until the tail runs — a verify against it would
    // report a spurious FAIL for a fingerprint that is not at fault.
    const bool piListComplete2b =
      !inShardedInstancePhase() || t_shardPhase.info->deferredPlacements.empty();

    if (piOn && piPredictHit && piVerify && piRec != nullptr && piListComplete2b) {
      bool diverged = (piRec->instances.size() != out_instances.size());
      size_t badIdx = SIZE_MAX;
      if (!diverged) {
        for (size_t i = 0; i < out_instances.size(); ++i) {
          if (piRec->instances[i] != out_instances[i]) {
            diverged = true;
            badIdx = i;
            break;
          }
        }
      }
      if (diverged) {
        // NV-DXVK [Phase2b]: shared counter — locked under the phase (cold:
        // FAIL must read 0 in any healthy run anyway).
        if (inShardedInstancePhase()) {
          std::lock_guard<std::mutex> failLock(m_shardEscapeMutex);
          m_piFail += 1;
        } else {
        m_piFail += 1;
        }
        // Throttled: a systematic fingerprint hole would otherwise emit one
        // line per batch per frame and bury the first (most diagnosable) case.
        thread_local uint32_t sPiFailLog = 0;
        if (sPiFailLog < 32u || (sPiFailLog & 0x3FFu) == 0u) {
          // WHY IT DIVERGED, not just that it did. These four bits separate the
          // candidate causes, which a percentage cannot:
          //   actNew=1   the resolution CREATED an instance this frame => the
          //              streaming / creation path, or a dedup miss. Expected
          //              non-zero while HYGIENE reports matNew>0.
          //   recStamp=1 the recorded instance was ALREADY stamped this frame,
          //              i.e. an earlier draw claimed it => cross-batch order
          //              dependence, which no per-batch fingerprint can cover.
          //   recRelink=1 the recorded instance is now linked to a DIFFERENT
          //              geometry (the engine-class relink garbageCollection
          //              documents).
          //   recGC=1    it is marked for collection.
          // All four zero => a genuine fingerprint hole, and only then is
          // fanoutRecordFingerprint the thing to change.
          const RtInstance* recI = (badIdx != SIZE_MAX && badIdx < piRec->instances.size())
                                 ? piRec->instances[badIdx] : nullptr;
          const RtInstance* actI = (badIdx != SIZE_MAX && badIdx < out_instances.size())
                                 ? out_instances[badIdx] : nullptr;
          Logger::warn(str::format(
            "[Perf.PushInst.FAIL] #", sPiFailLog,
            " f=", currentFrameIdx,
            " vs=0x", std::hex, vsHash, std::dec,
            " key=0x", std::hex, piKey, std::dec,
            " recorded=", piRec->instances.size(),
            " actual=", out_instances.size(),
            " firstDiffIdx=", (badIdx == SIZE_MAX ? -1 : int64_t(badIdx)),
            " builtFrame=", piRec->frameLastBuilt,
            " actNew=", (actI != nullptr && actI->m_frameCreated == currentFrameIdx) ? 1 : 0,
            " recStamp=", (recI != nullptr && recI->m_frameLastUpdated == currentFrameIdx) ? 1 : 0,
            " recRelink=", (recI != nullptr && recI->m_linkedBlas != &blas) ? 1 : 0,
            " recGC=", (recI != nullptr && recI->m_isMarkedForGC) ? 1 : 0,
            " recLastUpd=", (recI != nullptr ? int64_t(recI->m_frameLastUpdated) : -1),
            "  | READ actNew/recStamp/recRelink/recGC FIRST -- all four zero is"
            " the only case that means fanoutRecordFingerprint is incomplete."
            " Do NOT turn rtx.pushInstanceRecordsVerify off until this reads 0."));
        }
        sPiFailLog += 1;
      }
    }

    // REFRESH. Store on the way out, with the fingerprint computed on the way
    // in -- the inputs cannot change mid-call (this is all one dxvk-cs call),
    // so the pair is coherent, and storing the entry the loop actually produced
    // is what makes the next frame's hit a replay rather than a guess.
    //
    // A record is only stored for a batch that resolved every placement to a
    // live instance. A short out_instances means some placement was rejected
    // (collision, non-finite transform), and those rejections are decided by
    // per-placement state this fingerprint does not claim to cover -- so such a
    // batch stays on the loop rather than being recorded and mispredicted.
    if (piOn && out_instances.size() == transforms->size() && !out_instances.empty()) {
      // NV-DXVK [Phase2b]: m_fanoutRecords insert + back-pointer rewrites are
      // shared-member writes — locked under the phase, once per fanout draw.
      std::unique_lock<std::mutex> piRefreshLock;
      if (inShardedInstancePhase()) {
        piRefreshLock = std::unique_lock<std::mutex>(m_shardEscapeMutex);
      }
      if (piRec == nullptr) {
        // THE CAP IS A BACKSTOP, NOT THE MECHANISM. Retirement is the per-frame
        // age sweep in onFrameEnd; if the map is still at the ceiling after that
        // has run, refusing the store is O(1) and merely costs this batch its
        // acceleration. The first version scanned the whole map on every insert
        // to evict the least-recently-served entry -- O(N) per insert, on
        // dxvk-cs, in the streaming phase where inserts are most frequent, i.e.
        // paid hardest exactly when the frame is already worst.
        if (m_fanoutRecords.size() >= RtxOptions::pushInstanceRecordsMaxBatches()) {
          m_piCapped += 1;
          piRec = nullptr;   // NOT `return` -- the FsState accumulation below is
                             // [FanoutSplit]'s per-frame data and must run for
                             // every batch, recorded or not.
        } else {
          piRec = &m_fanoutRecords[piKey];
        }
      }
      if (piRec != nullptr) {
        // Drop back-pointers from the previous contents before overwriting, or
        // an instance that left this batch keeps naming it and would be
        // invalidated against a record it is no longer part of.
        for (RtInstance* inst : piRec->instances) {
          if (inst->m_batchRecordKey == piKey)
            inst->m_batchRecordKey = 0ull;
        }
        piRec->instances.assign(out_instances.begin(), out_instances.end());
        piRec->inputHash       = piPrint;
        piRec->frameLastBuilt  = currentFrameIdx;
        piRec->frameLastServed = currentFrameIdx;
        piRec->valid           = true;
        for (RtInstance* inst : piRec->instances)
          inst->m_batchRecordKey = piKey;
      }
    }
    // ================================================================

    // Accumulate only — the [FanoutSplit] line for this frame is emitted by the
    // rollover at the TOP of the next frame's first call for this VS, because the
    // propId membership sets have to roll over before the placement loop reads
    // them. See logFanoutSplitFrame for what each field means.
    st.draws    += 1;
    st.nInst    += static_cast<uint32_t>(transforms->size());
    st.live     += static_cast<uint32_t>(out_instances.size());
    st.created  += created;
    st.collide  += collide;
    st.mir         += mirrored;
    st.newProp     += newProp;
    st.missProp    += missProp;
    st.mtxStable    += mtxStable;
    st.basisStable  += basisStable;
  }

  // NV-DXVK [Phase2b]: see the declaration comment. Runs in the ORDERED TAIL
  // only — the phase flag is off, so every path inside (migration, addInstance,
  // teleport, spatial-map writes) executes inline exactly like the legacy code.
  void InstanceManager::processDeferredFanoutPlacements(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData,
    DrawCallCache* drawCallCache, const std::vector<uint32_t>& placements,
    std::vector<RtInstance*>& out_instances) {
    const std::vector<Matrix4>* transforms = drawCall.getTransformData().instancesToObject;
    if (transforms == nullptr || transforms->empty()) {
      return;
    }
    const std::vector<Matrix4>* prevTransforms = drawCall.getTransformData().prevInstancesToObject;
    if (prevTransforms != nullptr && prevTransforms->size() != transforms->size()) {
      prevTransforms = nullptr;
    }
    const Matrix4& drawObjectToWorld = drawCall.getTransformData().objectToWorld;
    // MUST match processSceneObjectFanout's composition bit-for-bit — the
    // SpatialMap key is hashed from the composed matrix bytes, so a differently
    // rounded product would file the instance under a key nothing else agrees
    // with. Same identity-exact rule, same multiply, same prev handling.
    const bool fanoutBaseIsIdentity = isIdentityExact(drawObjectToWorld);
    const DrawScopedState drawState = computeDrawScopedState(blas, drawCall, materialData);

    for (const uint32_t placement : placements) {
      if (placement >= transforms->size()) {
        continue;
      }
      FanoutSplit split;
      if (fanoutBaseIsIdentity) {
        split.objectToWorld = (*transforms)[placement];
      } else {
        split.objectToWorld = drawObjectToWorld * (*transforms)[placement];
      }
      if (prevTransforms != nullptr) {
        if (fanoutBaseIsIdentity) {
          split.prevObjectToWorld = (*prevTransforms)[placement];
        } else {
          split.prevObjectToWorld = drawObjectToWorld * (*prevTransforms)[placement];
        }
        split.hasPrevObjectToWorld = true;
      }
      RtInstance* instance = processSceneObjectImpl(cameraManager, rayPortalManager, blas, drawCall,
                                                    materialData, nullptr, drawCallCache, &split, drawState);
      if (instance != nullptr) {
        // Duplicate suppression mirrors the wrapper's collide scan: two
        // placements resolving to one instance must not double-register it.
        bool duplicate = false;
        for (const RtInstance* seen : out_instances) {
          if (seen == instance) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          out_instances.push_back(instance);
        }
      }
    }
  }

  // NV-DXVK [Perf.SceneObj]: stage timer for processSceneObjectImpl.
  //
  // WHY HERE. The chain is measured end to end and every link agrees:
  //   [Perf.Busy]        frame 73.6 ms
  //   [Perf.CsSplit]     dxvk-cs exec 73.3 ms = 99.6%   -> dxvk-cs IS the frame
  //   [Perf.CsCmd]       commitGeometryToRT 54 ms/f, 1119 calls @ 48.5us
  //   [CommitRT]         submitMs 49, finalizeMs 0      -> computing, not stalled
  //   [Perf.SubmitState] process 36us/draw (80%), material 8us (18%)
  //   [ProcDCS]          instMs 32, geomMs 4            -> the instance half
  // That leaves InstanceManager::processSceneObject at ~32 ms/frame, 29us/draw
  // -- 44% of the whole frame, and the deepest leaf anything has reached.
  //
  // WHAT THE STAGES DECIDE:
  //   find    findSimilarInstance -- the dedup search (SpatialMap lookup,
  //           nearest-neighbour stage, material compare). If this dominates,
  //           the fix is the dedup key, not the instance work; note that
  //           [FindSim]/suppressStablePropIdVsHashes already established the
  //           propId round-robins with period 3 for some shaders, forcing
  //           100% of those lookups into the nearest-neighbour stage.
  //   mid     the decision logic and the pile of per-draw diagnostics between
  //           the search and the instance work ([MtnDedup] et al).
  //   add     addInstance -- only on a dedup miss. addedPct says how often.
  //   update  updateInstance -- transform/surface/material write. Expected to
  //           dominate on a steady scene where nearly every draw dedups.
  //
  // COST -- SAMPLED DURATIONS, EXACT COUNTS. CORRECTED 2026-08-07: the line here
  // used to read "4 clock reads per call (~205 ns), ~0.23 ms/frame at 1119 draws"
  // and the guard stamped the clock on EVERY call. It is billed per INSTANCE, not
  // per draw: [Perf.UpdInst] measures instPerFrame=15,500 against ~1,090 draws, so
  // 5 reads x 15,500 x ~41 ns = ~3.2 ms/frame -- 14x the budgeted figure and ~6%
  // of a 55 ms frame, on a probe whose job is to attribute 25 ms. That is the same
  // per-draw/per-instance error catalogued at the top of this file (acquireVsDebugId,
  // rtx.perfNonOpaqueCensus, and this guard once already).
  //
  // So: time 1 instance in 64 (~242/frame, ~0.05 ms) and count every instance
  // exactly, matching UpdInstSplitGuard. estMsPerFrame is the sampled mean x the
  // EXACT call count, which is directly comparable to [ProcDCS] instMs -- the two
  // measure the same function.
  //
  // Gated off by default via rtx.perfSceneObjSplit.
  namespace {
    struct SceneObjSplitGuard {
      using clk = std::chrono::steady_clock;
      static constexpr uint32_t kSampleMask = 63u;   // duration-sample 1 in 64

      bool            on;
      bool            timed   = false;
      uint32_t        frameId = 0;
      clk::time_point t0, tFind, tMid, tAdd;
      bool            hasFind = false, hasMid = false, hasAdd = false;
      bool            added   = false;

      // NV-DXVK [Perf.MidWork]: EXACT per-instance counters for the five
      // diagnostic blocks that make up `mid`. COUNTED, NEVER TIMED. Each block is
      // sub-microsecond against a ~41 ns clock read, so sub-timers here would be
      // measurement-floored and would answer "how long" with noise -- the same
      // trap as the FillMat per-draw buckets. What decides the fix is MECHANISM:
      // how many instances pay a hash-map insert-or-find, and how many str::format
      // calls are built only for the logger to throw them away. [PassCensus],
      // [InstCounts], [VS2904Trace] and [SubViewKey.create] are all on the log.cpp
      // denylist, so fmtDropped is work with no output at all.
      uint32_t nMtn = 0, nSubvk = 0, nPass = 0, nProp = 0, nVs29 = 0;
      uint32_t nMapOp = 0, nStrMap = 0, nFmt = 0, nFmtDropped = 0;

      SceneObjSplitGuard(bool enabled, uint32_t fid)
      : on(enabled), frameId(fid) {
        if (!on) {
          return;
        }
        static thread_local uint32_t s_seq = 0;
        timed = ((s_seq++ & kSampleMask) == 0u);
        if (timed) {
          t0 = clk::now();
        }
      }

      void markFind()   { if (timed) { tFind = clk::now(); hasFind = true; } }
      void markMid()    { if (timed) { tMid  = clk::now(); hasMid  = true; } }
      void markAdd()    { if (timed) { tAdd  = clk::now(); hasAdd  = true; } }
      void noteAdded()  { added = true; }
      void markUpdate() { /* end of scope; the destructor stamps it */ }

      // Called from inside each mid block, AFTER its gate, on the path that does
      // real work -- so a block reading 0 here costs nothing beyond its gate.
      // noteMapOp marks a uint64-keyed insert-or-find, noteStrMap a std::string-keyed
      // one (far more expensive: it hashes and compares 19 characters), noteFmt a
      // str::format that was actually built, and noteFmtDropped one the logger
      // then discarded.
      void noteMtn()        { if (on) ++nMtn; }
      void noteSubvk()      { if (on) ++nSubvk; }
      void notePass()       { if (on) ++nPass; }
      void noteProp()       { if (on) ++nProp; }
      void noteVs29()       { if (on) ++nVs29; }
      void noteMapOp()      { if (on) ++nMapOp; }
      void noteStrMap()     { if (on) ++nStrMap; }
      void noteFmt()        { if (on) ++nFmt; }
      void noteFmtDropped() { if (on) ++nFmtDropped; }

      static int64_t ns(clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
      }

      ~SceneObjSplitGuard() {
        if (!on)
          return;

        const auto tEnd = clk::now();

        static thread_local int64_t  sFind = 0, sMid = 0, sAdd = 0, sUpd = 0;
        static thread_local uint64_t sCalls = 0, sAdds = 0, sSamples = 0, sFrames = 0;
        static thread_local uint64_t qMtn = 0, qSubvk = 0, qPass = 0, qProp = 0, qVs29 = 0;
        static thread_local uint64_t qMapOp = 0, qStrMap = 0, qFmt = 0, qFmtDropped = 0;
        static thread_local uint32_t sLastFrame = UINT32_MAX;
        static thread_local clk::time_point sLastLog{};
        static thread_local bool sInit = false;
        if (!sInit) { sLastLog = tEnd; sInit = true; }

        if (sLastFrame != frameId) { sLastFrame = frameId; ++sFrames; }

        ++sCalls;
        if (added) ++sAdds;

        // Durations only on the sampled instances -- an unsampled call never read
        // the clock, so t0/tFind/tMid/tAdd hold nothing to accumulate.
        if (timed) {
          ++sSamples;
          sFind += ns(t0, hasFind ? tFind : tEnd);
          if (hasFind) sMid += ns(tFind, hasMid ? tMid : tEnd);
          if (hasMid)  sAdd += ns(tMid,  hasAdd ? tAdd : tEnd);
          if (hasAdd)  sUpd += ns(tAdd,  tEnd);
        }

        qMtn += nMtn; qSubvk += nSubvk; qPass += nPass; qProp += nProp; qVs29 += nVs29;
        qMapOp += nMapOp; qStrMap += nStrMap; qFmt += nFmt; qFmtDropped += nFmtDropped;

        if (std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - sLastLog).count() < 3000) {
          return;
        }

        const double  smp    = double(sSamples ? sSamples : 1);
        const double  frames = double(sFrames  ? sFrames  : 1);
        const double  perFrm = double(sCalls) / frames;
        const int64_t tot    = sFind + sMid + sAdd + sUpd;

        // usPerCall is the sampled mean; estMsPerFrame is that mean x the EXACT
        // call count, so a rare-but-expensive stage cannot hide behind sampling
        // probability. Sum the four estMsPerFrame against [ProcDCS] instMs: they
        // are the same function, and a large gap means a stage boundary is wrong.
        const double usFind = double(sFind) / 1000.0 / smp;
        const double usMid  = double(sMid)  / 1000.0 / smp;
        const double usAdd  = double(sAdd)  / 1000.0 / smp;
        const double usUpd  = double(sUpd)  / 1000.0 / smp;

        // NV-DXVK [Perf.Report]: estMsPerFrame, not usPerCall -- the report is a
        // ms/frame table and a per-call mean cannot be placed in one without the
        // call count. These four are NESTED inside [ProcDCS] instMs and the
        // report's cross-validation checks them against it.
        perfreport::publish(perfreport::Slot::SceneObjFindMs,   usFind * perFrm / 1000.0);
        perfreport::publish(perfreport::Slot::SceneObjMidMs,    usMid  * perFrm / 1000.0);
        perfreport::publish(perfreport::Slot::SceneObjAddMs,    usAdd  * perFrm / 1000.0);
        perfreport::publish(perfreport::Slot::SceneObjUpdateMs, usUpd  * perFrm / 1000.0);

        Logger::info(str::format(
          "[Perf.SceneObj] calls=", sCalls, " frames=", sFrames,
          " callsPerFrame=", perFrm,
          " samples=", sSamples,
          " addedPct=", (sCalls ? (sAdds * 100 / sCalls) : 0),
          " | usPerCall find=", usFind, " mid=", usMid, " add=", usAdd, " update=", usUpd,
          " | estMsPerFrame find=", (usFind * perFrm / 1000.0),
          " mid=", (usMid * perFrm / 1000.0),
          " add=", (usAdd * perFrm / 1000.0),
          " update=", (usUpd * perFrm / 1000.0),
          " | pct find=", (tot ? (sFind * 100 / tot) : 0),
          " mid=", (tot ? (sMid * 100 / tot) : 0),
          " add=", (tot ? (sAdd * 100 / tot) : 0),
          " update=", (tot ? (sUpd * 100 / tot) : 0)));

        // EXACT, every instance -- no sampling, no floor. `mid` has no gate of its
        // own, so these say which of its five blocks actually ran and what each one
        // cost in mechanism terms. mapOps/strMaps are hash-map insert-or-finds per
        // frame; fmtDropped is str::format work whose output the log.cpp denylist
        // discards, i.e. pure waste that deleting the block recovers in full.
        Logger::info(str::format(
          "[Perf.MidWork] frames=", sFrames, " callsPerFrame=", perFrm,
          " | perFrame mtn=", (double(qMtn) / frames),
          " subvk=", (double(qSubvk) / frames),
          " pass=", (double(qPass) / frames),
          " prop=", (double(qProp) / frames),
          " vs2904=", (double(qVs29) / frames),
          " | mapOps=", (double(qMapOp) / frames),
          " strMaps=", (double(qStrMap) / frames),
          " fmt=", (double(qFmt) / frames),
          " fmtDropped=", (double(qFmtDropped) / frames)));

        sFind = sMid = sAdd = sUpd = 0;
        sCalls = sAdds = sSamples = sFrames = 0;
        qMtn = qSubvk = qPass = qProp = qVs29 = 0;
        qMapOp = qStrMap = qFmt = qFmtDropped = 0;
        sLastLog = tEnd;
      }
    };
  }

  RtInstance* InstanceManager::processSceneObjectImpl(
    const CameraManager& cameraManager, const RayPortalManager& rayPortalManager,
    BlasEntry& blas, const DrawCallState& drawCall, MaterialData& materialData, RtInstance* existingInstance,
    DrawCallCache* drawCallCache, const FanoutSplit* split, const DrawScopedState& drawState) {

    SceneObjSplitGuard psoSplit(RtxOptions::perfSceneObjSplit(), m_device->getCurrentFrameId());

    // If the RtInstance represents multiple instances, use the full transform of the first copy for the spatial map.
    // this prevents a bad de-duplication when the same replacement asset is used in multiple GeomPointInstancer prims.
    // NV-DXVK [fanout split]: a split placement is an ordinary single-placement
    // object — its own composed transform is the dedup key, not the batch leader's.
    Matrix4 firstInstanceObjectToWorld = split != nullptr
      ? split->objectToWorld
      : drawCall.getTransformData().calcFirstInstanceObjectToWorld();

    // NV-DXVK [fanout split]: a split placement always carries 0 here — the
    // codebase's "key on the composed matrix bytes" contract, which is exact for a
    // stationary prop. Motion is handled by the engine history instead (see
    // FanoutSplit::prevObjectToWorld), not by an override key.
    //
    // The batch's own propId must never be substituted: it names whichever prop is
    // element [0] today. See the note above processSceneObjectFanout.
    const uint64_t lookupStablePropId = split != nullptr
      ? split->stablePropId
      : drawCall.getTransformData().stablePropId;

    // If we already know which instance to use, just use that.
    RtInstance* currentInstance = existingInstance;

    // NV-DXVK [perf] handoff v7 sec 4a: XXH64(firstInstanceObjectToWorld) as
    // computed by findSimilarInstance's exact stage, forwarded to updateInstance
    // so the spatial-map write does not hash the same bytes again. Stays 0 when
    // the caller already knew the instance (no lookup ran) or when the lookup
    // keyed on a stablePropId, and 0 means "recompute", i.e. the old behaviour.
    XXH64_hash_t queryMatrixHash = 0;

    // Search for an existing instance matching our input
    if (currentInstance == nullptr) {
      // NV-DXVK [fanout prev-transform identity]: hand the engine's history down
      // so the exact stage gets a second, position-independent attempt before any
      // distance-based search is considered. Only a split placement has one.
      const Matrix4* prevO2W = (split != nullptr && split->hasPrevObjectToWorld)
        ? &split->prevObjectToWorld
        : nullptr;
      currentInstance = findSimilarInstance(blas, materialData, firstInstanceObjectToWorld, drawCall.cameraType, rayPortalManager, lookupStablePropId, drawCallCache, prevO2W, &queryMatrixHash);

      // NV-DXVK [Phase2b]: findSimilarInstance raised the defer sentinel (full
      // miss, portal teleport, or migration candidate) — this draw/placement is
      // re-run by the ordered tail with the phase flag off. Skip mid/add/update
      // entirely: creating or updating anything here would double-run when the
      // tail replays it. The caller reads t_shardPhase.deferredThisDraw.
      if (inShardedInstancePhase() && t_shardPhase.deferredThisDraw) {
        psoSplit.markFind();
        return nullptr;
      }
    }

    psoSplit.markFind();   // NV-DXVK [Perf.SceneObj]: end of `find`

    // NV-DXVK [SubView phantom check]: per-VS-per-frame counters of
    // "found similar existing instance" vs "created new". If sub-view
    // VSes show new>0 in steady state after the smoothed-scale fix,
    // phantoms are still being spawned each frame and we need to look
    // at why findSimilarInstance is missing despite stable o2w.
    //
    // Note: this fires for ALL VSes, but the [SubViewVar] log already
    // identifies which hashes correspond to sub-view content — match
    // those hashes against this log's vs= field.
    const bool foundSimilar = (currentInstance != nullptr);
    {
      // NV-DXVK [MtnDedup]: for the path-10 mountain shaders, log whether
      // findSimilarInstance reused an existing RtInstance or a new one is
      // about to be spawned, with the lookup propId + transform.
      //
      // Logged EVERY mountain draw, EVERY frame (no cap, no frame sampling):
      // the game runs at a few FPS so any frame-modulo gate would almost
      // never land. Correlate per frame with the same-frame [MtnPIAdd] batch
      // census — together they show whether the TLAS over-instancing comes
      // from too many draws/RtInstances or from per-batch fanout.
      const XXH64_hash_t vsHashMd = drawCall.getTransformData().vertexShaderHash;
      // Also driven by rtx.findSimilarProbeVsHashes. This is the only probe on
      // this path that fires for EVERY draw rather than sampling, which is what
      // is needed to catch a draw that has never once appeared as a lookup:
      // [FindSim] has only ever logged main-world queries for the tree VS, yet
      // its SpatialMap keeps ending up holding a sky-coordinate entry. Whatever
      // creates that entry has to pass through here.
      const bool isOptDedupVs =
        lookupHash(RtxOptions::findSimilarProbeVsHashes(), vsHashMd);
      if (vsHashMd == 0x29146e1dd50b0314ull
          || vsHashMd == 0x28f7ffa90d189017ull
          || isOptDedupVs) {
        // UNCAPPED, for both the hardcoded VSes and the option path. The
        // option path used to carry a global 4000-line budget; aimed at the
        // high-churn VSes (0x2859d250 / 0x28d6baea / 0x292b6ba0) that draw
        // hundreds of times per frame, any budget burns out early and blinds
        // the rest of the run. Same failure as [InstReap]'s 24/frame cap.
        // NV-DXVK 2026-08-06: tagDenied gate — this site alone was 136
        // lines/frame (00:48 capture) and the whole gather+format below is
        // wasted when the denylist would drop the line anyway.
        if (!Logger::tagDenied("[MtnDedup]")) {
          // NV-DXVK [Perf.MidWork]: past the VS gate AND past the denylist, so
          // this instance pays the full gather below (a transformed centroid, six
          // geometry-hash reads, a texture hash and debugHashInputs).
          psoSplit.noteMtn();
          psoSplit.noteFmt();
          // Centroid is what the SpatialMap actually keys/cells on, and it is
          // NOT o2w.T — log both so a divergence is visible directly.
          const Vector3 mdCentroid =
            blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
          const float mdCol0 = length(Vector3(firstInstanceObjectToWorld[0][0],
                                              firstInstanceObjectToWorld[0][1],
                                              firstInstanceObjectToWorld[0][2]));
          // The DRAW's geometry, not blas.input's: the draw's hash is what
          // selected (or failed to select) the BlasEntry, so it is the input
          // to the routing decision we are diffing. blas.input is the result.
          const auto& mdGeo = drawCall.getGeometryData();
          Logger::info(str::format(
            "[MtnDedup] vsXxh=0x", std::hex, static_cast<uint64_t>(vsHashMd), std::dec,
            " frame=", m_device->getCurrentFrameId(),
            " cam=", static_cast<uint32_t>(drawCall.cameraType),
            " foundSimilar=", (foundSimilar ? 1 : 0),
            " hadExisting=", (existingInstance != nullptr ? 1 : 0),
            " propId=0x", std::hex, lookupStablePropId, std::dec,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
            " o2wCol0Len=", mdCol0,
            " centroid=(", mdCentroid.x, ",", mdCentroid.y, ",", mdCentroid.z, ")",
            " lookup.o2w.T=(", firstInstanceObjectToWorld[3][0], ",",
              firstInstanceObjectToWorld[3][1], ",",
              firstInstanceObjectToWorld[3][2], ")",
            // NV-DXVK: per-component geometry hashes. The 2026-07-29 [MapDump]
            // proved every tree miss is the FIRST SIGHTING of a new blasPtr
            // with an empty map — a forced miss, not a keying failure — while
            // the tree VS was reaped 0 times in 17078 reaps. So the defect is
            // BlasEntry churn: the same billboard hashing differently on some
            // frames. BlasEntry identity is the geometry hash, so log each
            // COMPONENT rather than the composite rules: FullGeometryHash
            // changing only says "something moved", whereas diffing two
            // consecutive frames' components names the single unstable field.
            //   VertexPosition/Texcoord -> content genuinely rewritten
            //     (camera-facing billboard verts rebaked per frame)
            //   Indices/GeometryDescriptor -> topology or draw-range changed
            //   VertexLayout/VertexShader -> binding state changed, content
            //     identical — that would be the dynamic-buffer arena rotating
            //     (d3d11_rtx.cpp:30262) and is fixable without touching content
            " hPos=0x", std::hex,
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexPosition]),
            " hUv=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexTexcoord]),
            " hIdx=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::Indices]),
            " hDesc=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::GeometryDescriptor]),
            " hLayout=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexLayout]),
            " hVs=0x",
              static_cast<uint64_t>(mdGeo.hashes[HashComponents::VertexShader]),
            " hFull=0x",
              static_cast<uint64_t>(mdGeo.getHashForRule<rules::FullGeometryHash>()),
            std::dec,
            " vtxCount=", mdGeo.vertexCount,
            " idxCount=", mdGeo.indexCount,
            // NV-DXVK: texture identity. findSimilarProbeVsHashes selects a
            // SHADER, and 0x29382bf838fda043 is not tree-specific -- it is a
            // shared prop VS (the 2026-07-29 16:59 manifest has it drawing a
            // 128x128 industrial panel atlas, 8 instances, while [SkyDiag] shows
            // 8 distinct albedos on it). Without a texture id every tree line is
            // mixed in with unrelated props and the per-component hash diff
            // above cannot be restricted to the billboard. albHash matches the
            // <hash>_albedo.dds naming in onscreen_albedo_dump/manifest.txt, so a
            // dumped texture maps straight to the draws that used it.
            " albHash=0x", std::hex,
              static_cast<uint64_t>(drawCall.getMaterialData().getColorTexture().isValid()
                ? drawCall.getMaterialData().getColorTexture().getImageHash()
                : 0ull),
            " mat=0x", static_cast<uint64_t>(drawCall.getMaterialData().getHash()), std::dec,
            // The 2026-07-29 17:21 capture showed this material hash ALTERNATING
            // every frame on a stationary tree whose geometry hashes and albedo
            // were byte-stable (10 distinct mat values at one position, e.g.
            // fc2c10c1ed54 / b64d0af14ff5 flipping f2705..f2709). Dump the hash
            // inputs so the responsible field is named rather than inferred.
            " | ", drawCall.getMaterialData().debugHashInputs()));
        }
      }
    }
    // NV-DXVK [SubViewKey]: per-frame census of sub-view-fan instance
    // keying — the second half of the two-views probe pair ([InstReap]
    // shows WHAT dies at the flip; this shows WHY the steady-state stream
    // didn't keep it alive). A draw counts if it carries a stablePropId or
    // a reprojected x1000-scale o2w (col0 length > 100; normal draws are
    // ~1). Readout, intro frame vs steady frame:
    //   steady: hit high, create=0, but the intro-era propIds never appear
    //     in any later create/hit -> the engine stopped issuing those
    //     draws; fix at keep/refresh policy or the draw source.
    //   steady: create>0 EVERY frame with new propIds -> keying churn; fix
    //     at the propId producer (cf. the reverted PropIdKeepLong note in
    //     garbageCollection — MakeBoneStablePropId instability).
    // Aggregate 1 line/frame + create detail capped 12/frame, gameplay-gated.
    {
      const bool svGameplay =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u;
      const uint64_t svPropId = lookupStablePropId;
      const float svSc0Sq =
          firstInstanceObjectToWorld[0][0] * firstInstanceObjectToWorld[0][0]
        + firstInstanceObjectToWorld[0][1] * firstInstanceObjectToWorld[0][1]
        + firstInstanceObjectToWorld[0][2] * firstInstanceObjectToWorld[0][2];
      const bool svScaled = svSc0Sq > 10000.0f;  // col0 len > 100 => reprojected
      if (svGameplay && (svPropId != 0ull || svScaled)) {
        psoSplit.noteSubvk();   // NV-DXVK [Perf.MidWork]
        struct SvKeyAgg {
          uint32_t frame = 0xFFFFFFFFu;
          uint32_t cand = 0, hit = 0, create = 0, scaled = 0, detail = 0;
        };
        static thread_local SvKeyAgg sSvKey;
        const uint32_t svFrame = m_device->getCurrentFrameId();
        if (sSvKey.frame != svFrame) {
          if (sSvKey.frame != 0xFFFFFFFFu && sSvKey.cand > 0u) {
            psoSplit.noteFmt();   // NV-DXVK [Perf.MidWork]: this one does emit
            Logger::info(str::format(
              "[SubViewKey] f=", sSvKey.frame,
              " cand=", sSvKey.cand,
              " hit=", sSvKey.hit,
              " create=", sSvKey.create,
              " scaled=", sSvKey.scaled));
          }
          sSvKey = SvKeyAgg{};
          sSvKey.frame = svFrame;
        }
        sSvKey.cand += 1;
        if (svScaled) sSvKey.scaled += 1;
        if (foundSimilar) {
          sSvKey.hit += 1;
        } else {
          sSvKey.create += 1;
          // Cap raised 12 -> 48. The 12/frame budget is shared across every
          // sub-view VS with no prioritisation, and it measurably truncated:
          // in the 2026-07-29 capture 30 of 689 frames saturated at exactly
          // 12, including f=9510 and f=9564 — the two frames carrying the
          // propId-collision reaps this probe exists to explain. Observed
          // distribution is <=5 creates on most frames, so 48 costs little
          // while making the busy frames (the interesting ones) complete.
          if (sSvKey.detail < 48u) {
            sSvKey.detail += 1;
            // NV-DXVK: blasPtr + the BlasEntry's OWN recorded VS, alongside
            // the DRAW's VS. The SpatialMap is per-BlasEntry, so [FindSim]'s
            // ownerVs — read as dbgOwner->getBlas()->input...vertexShaderHash
            // — can only ever echo the VS of the blas being queried. It is
            // structurally incapable of detecting two shaders sharing one
            // BlasEntry, which is the case it was written to test. Here the
            // draw's VS and the blas's VS are independent reads, so
            // blasVs != vs on a single line IS the shared-BlasEntry proof.
            // blasPtr cross-references directly against [MtnDedup]'s blasPtr.
            // NV-DXVK [Perf.MidWork]: [SubViewKey.create] is on the log.cpp
            // denylist, so this format is built and then discarded.
            psoSplit.noteFmt();
            psoSplit.noteFmtDropped();
            Logger::info(str::format(
              "[SubViewKey.create] f=", svFrame,
              " vs=0x", std::hex,
                static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash),
              " propId=0x", svPropId,
              " blasPtr=0x", reinterpret_cast<uintptr_t>(&blas),
              " blasVs=0x",
                static_cast<uint64_t>(blas.input.getTransformData().vertexShaderHash),
                std::dec,
              " mapSz=", blas.getSpatialMap().size(),
              " scaledCol0=", (svScaled ? 1 : 0),
              " o2wT=(", firstInstanceObjectToWorld[3][0], ",",
                         firstInstanceObjectToWorld[3][1], ",",
                         firstInstanceObjectToWorld[3][2], ")"));
          }
        }
      }
    }
    // NV-DXVK [PassCensus] DELETED 2026-08-07 -- measured, not guessed.
    // [Perf.MidWork] read pass == mapOps == callsPerFrame == 15,509.7 in all ten
    // windows of the 02:58 capture: this accumulate path had NO gate of any kind
    // -- not the gameplay gate the block above it uses, not a VS filter -- so a
    // uint64-keyed unordered_map insert-or-find plus six min/max ran on EVERY
    // instance of EVERY frame. [PassCensus] is on the log.cpp denylist, so every
    // bucket it accumulated was discarded unread.
    // Deleting beats guarding (v6 process note 2): no correctness surface, and it
    // helps the instances a guard would have missed. The view-1/view-2 per-VS diff
    // it was written for is answered and closed -- see the s2s two-views work.
    {
      const XXH64_hash_t vsHash = drawCall.getTransformData().vertexShaderHash;
      // NV-DXVK: the vsKey hex-format loop that used to sit here went with
      // [InstCounts] below -- it existed only to build that map's string key and
      // ran on every instance. [PropCensus] keys on vsHash directly.

      // NV-DXVK [PropCensus]: for VS_2904d2 sub-view mountains, track which
      // distinct world-position keys appear in this frame's submissions and
      // log the set every 30 frames. Goal: if one mountain is permanently
      // missing while the rest render, that prop's position will be
      // consistently absent from the census. Pinpoints which charIdx /
      // world-pos pair is dropping out.
      //
      // [gameplay gate] Only run during gameplay (after engine-hook main-
      // cam has fired >16 times). Without this gate the accumulating
      // thread-local maps fill with menu-phase noise before gameplay even
      // starts, polluting the "first seen / last seen" stats.
      const bool gameplayActive =
        tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16;
      if (gameplayActive && vsHash == 0x2904d2163ef31a17ull) {
        psoSplit.noteProp();   // NV-DXVK [Perf.MidWork]
        const float wpx = float(firstInstanceObjectToWorld[3][0]);
        const float wpy = float(firstInstanceObjectToWorld[3][1]);
        const float wpz = float(firstInstanceObjectToWorld[3][2]);
        const Vector3 wp{wpx, wpy, wpz};
        // Quantize to 5000u so per-frame snap drift doesn't make the
        // same prop look like multiple distinct keys.
        const int kx = int(std::round(wp.x / 5000.f));
        const int ky = int(std::round(wp.y / 5000.f));
        const int kz = int(std::round(wp.z / 5000.f));
        static thread_local std::unordered_map<uint64_t, uint32_t> sPropFirstSeen;
        static thread_local std::unordered_map<uint64_t, uint32_t> sPropLastSeen;
        const uint64_t key = (uint64_t(uint32_t(kx)) & 0x1FFFFFu)
                           | ((uint64_t(uint32_t(ky)) & 0x1FFFFFu) << 21)
                           | ((uint64_t(uint32_t(kz)) & 0x1FFFFFu) << 42);
        const uint32_t curF = m_device->getCurrentFrameId();
        auto first = sPropFirstSeen.find(key);
        if (first == sPropFirstSeen.end()) {
          sPropFirstSeen.emplace(key, curF);
          Logger::info(str::format(
            "[PropCensus] newKey f=", curF,
            " worldPos=(", wp.x, ",", wp.y, ",", wp.z, ")",
            " key=(", kx, ",", ky, ",", kz, ")"));
        }
        sPropLastSeen[key] = curF;

        // Every 60 frames, dump props whose lastSeen is 2+ frames behind
        // current frame — those are the "missing" ones.
        static thread_local uint32_t sLastReportF = UINT32_MAX;
        if (sLastReportF != curF && (curF % 60) == 0) {
          sLastReportF = curF;
          for (const auto& [k, lastF] : sPropLastSeen) {
            if (lastF + 2 <= curF && sPropFirstSeen[k] + 30 < curF) {
              // Decode position from key
              const int ux = int(k & 0x1FFFFFu);
              const int uy = int((k >> 21) & 0x1FFFFFu);
              const int uz = int((k >> 42) & 0x1FFFFFu);
              const int sx = (ux & 0x100000) ? (ux | ~int(0x1FFFFF)) : ux;
              const int sy = (uy & 0x100000) ? (uy | ~int(0x1FFFFF)) : uy;
              const int sz = (uz & 0x100000) ? (uz | ~int(0x1FFFFF)) : uz;
              Logger::warn(str::format(
                "[PropCensus] DROPPED f=", curF,
                " firstSeen=", sPropFirstSeen[k],
                " lastSeen=", lastF,
                " framesSinceSeen=", (curF - lastF),
                " posKey=(", sx*5000, ",", sy*5000, ",", sz*5000, ")"));
            }
          }
        }
      }

      // NV-DXVK [InstCounts] DELETED 2026-08-07 -- measured, not guessed.
      // [Perf.MidWork] read strMaps == callsPerFrame == 15,509.7 in all ten
      // windows of the 02:58 capture: a std::string-keyed unordered_map
      // insert-or-find on EVERY instance of EVERY frame, plus the 8-iteration
      // hex loop that built the key, plus ~77 str::format calls per frame that
      // the log.cpp denylist then threw away. [InstCounts] is denied, so the
      // entire structure produced no output at all. Deleting beats guarding
      // (v6 process note 2): there is no correctness surface here, and the
      // instances that would have missed any guard are helped too.
      // The reused/created question it answered is now carried by [MapGate]'s
      // mapWrMove/mapWrInsert and by [Perf.SceneObj] addedPct.
    }

    // NV-DXVK [VS_2904d2 trace]: this VS shows up in [InstCounts] as
    // created=48/frame (the biggest phantom source) but is invisible
    // to our d3d11-side probes [PhantomProbe] and [Cb2Track]. That
    // means it enters the instance manager via a path that bypasses
    // D3D11Rtx::SubmitDraw — likely a replacement-asset / GeomPoint
    // Instancer route — so our SetSkyCategoryFromCb2 reproject never
    // gets a shot at it.
    //
    // Goal of this probe: figure out which d3d11 VS it ACTUALLY
    // corresponds to (its raw XXH64 hash at this layer differs from
    // the d3d11 shader key), and watch the per-draw
    // firstInstanceObjectToWorld.translation so we can tell:
    //   - if it lands at far-distance main-world coords already
    //     (somebody upstream reprojected it for us), or
    //   - if it lands at sub-view-local coords < 100u from skyCam
    //     (untouched sub-view data → we need to reproject here),
    //   - whether it shifts frame-to-frame with the moving ship
    //     (drives the dedup miss → phantom trail), or
    //   - whether it's stable when the ship is stationary (the
    //     reused=48/created=0 window at frames 884-887 suggests
    //     this is the case).
    {
      const XXH64_hash_t vsHash = drawCall.getTransformData().vertexShaderHash;
      if (vsHash == 0x2904d2163ef31a17ull) {
        psoSplit.noteVs29();   // NV-DXVK [Perf.MidWork]
        thread_local uint32_t sLastFrame = UINT32_MAX;
        thread_local uint32_t sDrawIdx   = 0;
        const uint32_t curFrame = m_device->getCurrentFrameId();
        if (sLastFrame != curFrame) {
          sLastFrame = curFrame;
          sDrawIdx   = 0;
        } else {
          sDrawIdx  += 1;
        }
        // Only log first ~12 draws per frame to keep volume bounded —
        // 48 draws × every frame would spam the log.
        if (sDrawIdx < 12) {
          const Matrix4& m = firstInstanceObjectToWorld;
          const uint32_t camType = static_cast<uint32_t>(drawCall.cameraType);
          // NV-DXVK [Perf.MidWork]: denylisted -- built, then discarded.
          psoSplit.noteFmt();
          psoSplit.noteFmtDropped();
          Logger::info(str::format(
            "[VS2904Trace] frame=", curFrame, " draw=", sDrawIdx,
            " vsHash=0x", std::hex, vsHash, std::dec,
            " camType=", camType,
            " firstO2W.t=(", float(m[3][0]), ",",
                              float(m[3][1]), ",",
                              float(m[3][2]), ")",
            " foundSimilar=", (foundSimilar ? 1 : 0)));
        }
      }
    }

    psoSplit.markMid();   // NV-DXVK [Perf.SceneObj]: end of `mid`

    if (currentInstance == nullptr) {
      // NV-DXVK [Phase2b]: catch-all — no worker may ever reach addInstance
      // (m_instances.push_back + m_nextInstanceId, the Sec-6 escape). The find
      // stage's own defer sites cover its normal miss paths; this covers every
      // OTHER way currentInstance can be null here (e.g. the
      // enableInstanceDebuggingTools early return, or a future null path).
      // allowMiss (the ordered tail) creates inline, as the legacy path does.
      if (inShardedInstancePhase() && !t_shardPhase.allowMiss) {
        t_shardPhase.deferredThisDraw = true;
        return nullptr;
      }
      // No existing match - so need to create one
      currentInstance = addInstance(blas);
      psoSplit.noteAdded();
    }

    psoSplit.markAdd();   // NV-DXVK [Perf.SceneObj]: end of `add`

    // NV-DXVK [perf] handoff v7 sec 4a: firstInstanceObjectToWorld is a local of
    // this function and is never reassigned after the lookup, so the hint's
    // pointer stays valid for the whole of updateInstance.
    const SpatialKeyHint keyHint{ &firstInstanceObjectToWorld, queryMatrixHash };
    updateInstance(*currentInstance, cameraManager, blas, drawCall, materialData, drawState, split, keyHint);

    psoSplit.markUpdate();   // NV-DXVK [Perf.SceneObj]: end of `update`

    return currentInstance;
  }

  RtSurface::AlphaState InstanceManager::calculateAlphaState(const DrawCallState& drawCall, const MaterialData& materialData) {
    RtSurface::AlphaState out{};

    // Handle Alpha State for non-Opaque materials

    if (materialData.getType() == MaterialDataType::Translucent) {
      // Note: Explicitly ensure translucent materials are not considered fully opaque (even though this is the
      // default in the alpha state).
      out.isFullyOpaque = false;

      return out;
    } else if (materialData.getType() != MaterialDataType::Opaque) {
      return out;
    }

    // Determine if the Legacy Alpha State should be used based on the material data
    // Note: The Material Data may be either Legacy or Opaque here, both use the Opaque Surface Material.

    const auto& opaqueMaterialData = materialData.getOpaqueMaterialData();

    const bool useLegacyAlphaState = opaqueMaterialData.getUseLegacyAlphaState();

    // Handle Alpha Test State

    // Note: Even if the Alpha Test enable flag is set, we consider it disabled if the actual test type is set to always.
    const bool forceAlphaTest = drawCall.getCategoryFlags().test(InstanceCategories::AlphaBlendToCutout);
    const bool alphaTestEnabled = forceAlphaTest || (AlphaTestType)drawCall.getMaterialData().alphaTestCompareOp != AlphaTestType::kAlways;

    // Note: Use the Opaque Material Data's alpha test state information directly if requested,
    // otherwise derive the alpha test state from the drawcall (via its legacy material data).
    if (forceAlphaTest) {
      out.alphaTestType = AlphaTestType::kGreater;
      out.alphaTestReferenceValue = static_cast<uint8_t>(RtxOptions::forceCutoutAlpha() * 255.0);
    } else if (!useLegacyAlphaState) {
      out.alphaTestType = opaqueMaterialData.getAlphaTestType();
      out.alphaTestReferenceValue = opaqueMaterialData.getAlphaTestReferenceValue();
    } else if (alphaTestEnabled) {
      out.alphaTestType = (AlphaTestType)drawCall.getMaterialData().alphaTestCompareOp;
      out.alphaTestReferenceValue = drawCall.getMaterialData().alphaTestReferenceValue;
    }

    // Handle Alpha Blend State

    bool blendEnabled = false;
    BlendType blendType = BlendType::kColor;
    bool invertedBlend = false;

    // Note: Use the Opaque Material Data's blend state information directly if requested,
    // otherwise derive the alpha blend state from the drawcall (via its legacy material data).
    if (forceAlphaTest) {
      blendEnabled = false;
    } else if (!useLegacyAlphaState) {
      blendEnabled = opaqueMaterialData.getBlendEnabled();
      blendType = opaqueMaterialData.getBlendType();
      invertedBlend = opaqueMaterialData.getInvertedBlend();
    } else if (drawCall.getMaterialData().blendMode.enableBlending) {
      const auto srcColorBlendFactor = drawCall.getMaterialData().blendMode.colorSrcFactor;
      const auto dstColorBlendFactor = drawCall.getMaterialData().blendMode.colorDstFactor;
      const auto colorBlendOp = drawCall.getMaterialData().blendMode.colorBlendOp;

      blendEnabled = true; // Note: Set to false later for cases which don't need it

      if (colorBlendOp == VkBlendOp::VK_BLEND_OP_ADD) {
        if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) {
          // Opaque Alias
          blendEnabled = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Standard Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Inverted Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Standard Reverse Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Premultiplied Alpha vs Inverted Reverse Emissive Alpha Blending
          const auto srcAlphaBlendFactor = drawCall.getMaterialData().blendMode.alphaSrcFactor;
          const auto dstAlphaBlendFactor = drawCall.getMaterialData().blendMode.alphaDstFactor;
          const auto alphaBlendOp = drawCall.getMaterialData().blendMode.alphaBlendOp;
          const auto colorWriteMask = drawCall.getMaterialData().blendMode.writeMask;
          const bool alphaWritesDisabled = (colorWriteMask & VK_COLOR_COMPONENT_A_BIT) == 0;

          // A genuine premultiplied-alpha draw composites the alpha channel
          // with the OVER operator too (not just color). There are two
          // algebraically-identical ways to spell that alpha OVER:
          //   a = src.a*1          + dst.a*(1-src.a)   [ONE, ONE_MINUS_SRC_ALPHA]
          //   a = src.a*(1-dst.a)  + dst.a*1           [ONE_MINUS_DST_ALPHA, ONE]
          // Either one means the draw is maintaining a correct framebuffer
          // alpha channel — i.e. it is doing real translucent compositing,
          // not a fire-and-forget additive glow. TF2's sky cloud billboards
          // use the second form; the original check only knew the first, so
          // the clouds fell through to the "inverted reverse emissive"
          // branch, were tagged emissive, and rendered as glowing rectangles.
          const bool alphaIsOverComposite =
            alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
            ((srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
              dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
             (srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA &&
              dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE));

          const bool looksPremultiplied =
            (alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
             srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
             dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
            alphaWritesDisabled;

          if (alphaIsOverComposite) {
            // Genuine premultiplied-alpha translucency — OVER on both color
            // and alpha. This is compositing, never emission, so classify it
            // as kAlpha even when emissive blend translation is enabled.
            // (Real additive glows use ONE,ONE or SRC_ALPHA,ONE and are
            // caught by the emissive branches above; they do not maintain an
            // OVER-composited alpha channel, so they never reach here.)
            blendType = BlendType::kAlpha;
            invertedBlend = false;
          } else if (looksPremultiplied) {
            // Premultiplied Alpha (ONE, ONE_MINUS_SRC_ALPHA)
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
            invertedBlend = false;
          } else {
            // Inverted Reverse Emissive Alpha Blending
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
            invertedBlend = true;
          }
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Standard Color Blending
          blendType = BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Inverted Color Blending
          blendType = BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Standard Reverse Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseColorEmissive : BlendType::kReverseColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Inverted Reverse Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseColorEmissive : BlendType::kReverseColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Emissive Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) ||
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR)
          ) {
          // Standard Multiplicative Blending
          blendType = BlendType::kMultiplicative;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Double Multiplicative Blending
          blendType = BlendType::kDoubleMultiplicative;
          invertedBlend = false;
        } else {
          blendEnabled = false;
        }
      } else {
        blendEnabled = false;
      }
    }

    // Special case for the player model eyes in Portal:
    // They are rendered with blending enabled but 1.0 is added to alpha from the texture.
    // Detect this case here and turn such geometry into non-alpha-blended, otherwise
    // the eyes end up in the unordered TLAS and are not rendered correctly.
    const auto& drawMaterialData = drawCall.getMaterialData();
    if (blendEnabled && blendType == BlendType::kAlpha && !invertedBlend &&
        drawMaterialData.textureAlphaOperation == DxvkRtTextureOperation::Add &&
        drawMaterialData.textureAlphaArg1Source == RtTextureArgSource::Texture &&
        drawMaterialData.textureAlphaArg2Source == RtTextureArgSource::TFactor &&
        (drawMaterialData.tFactor >> 24) == 0xff) {
      blendEnabled = false;
    }

    if (blendEnabled) {
      out.blendType = blendType;
      out.invertedBlend = invertedBlend;
      // Note: Emissive blend flag must match which blend types are expected to use emissive override in the shader to appear emissive.
      out.emissiveBlend = isBlendTypeEmissive(blendType);

      // Handle Particle/Decal Flags
      // Note: Particles/Decals currently require blending be enabled, be it through the game's original draw call (if legacy alpha state is used),
      // or through the manually specified alpha state.

      // Note: Particles are differentiated from typical objects with opacity by labeling their source material textures as being particle textures.
      out.isParticle = drawCall.testCategoryFlags(InstanceCategories::Particle);
      out.isDecal = drawCall.testCategoryFlags(DECAL_CATEGORY_FLAGS);
    } else {
      out.invertedBlend = false;
      out.emissiveBlend = false;
    }
    
    // Set the fully opaque flag
    // Note: Fully opaque surfaces can only be signaled when no blending or alpha testing is done as well as no translucency material wise is used.
    // This is important for signaling when to not use the opacity channel in materials when it is not being used for anything.

    out.isFullyOpaque = !blendEnabled && out.alphaTestType == AlphaTestType::kAlways; // use the blend/test type from the output, rather than legacy for this so replacements can override
    out.isBlendingDisabled = !blendEnabled;

    return out;
  }

  // NV-DXVK [perf] handoff v5 sec 4c: EXACT change-detector for the `surf` block
  // of updateInstance (~0.167 us/inst floor-corrected, ~2.6 ms/frame at 15,500
  // instances). That block is ~25 straight-line writes into currentInstance
  // .surface derived entirely from (drawCall, materialData, categoryFlags,
  // alphaState); with matChg=0% every frame, it rewrites the same bytes.
  //
  // DISCIPLINE (v5 sec 0a): a field read by the block but missing from this key is
  // a FALSE HIT -- one surface silently rendered with another draw's state. Every
  // entry below is therefore justified against what the block actually reads. If
  // you add a read to that block, add it here in the same commit.
  //
  // WHAT THE TWO MATERIAL HASHES DO AND DO NOT COVER -- both were verified, not
  // assumed:
  //   materialHash = MaterialData::getHash(). The generated updateCachedHash() in
  //     rtx_material_data.h is X_TEXTURES(WRITE_TEXTURE_HASH) +
  //     X_CONSTANTS(WRITE_CONSTANT_HASH), i.e. EVERY texture and EVERY constant of
  //     the render material. That covers SpriteSheetRows/Cols/FPS,
  //     UseLegacyAlphaState, IsUnlitOutput, Tf2SkyboxFog and
  //     SubsurfaceDiffusionProfile -- all of the block's materialData reads.
  //   legacyHash = LegacyMaterialData::getHash(). Its HashData (rtx_materials.h
  //     ~2215) is 11 texture hashes + all eight blendMode fields + the three
  //     alphaTest fields, and NOTHING ELSE. So the texture arg/op sources, the
  //     texture operations, tFactor, isTextureFactorBlend and
  //     isVertexColorBakedLighting are NOT covered by it and are keyed
  //     individually below. Keying on legacyHash alone would have been the exact
  //     sec 0a defect.
  //
  // NOT IN THE KEY, ON PURPOSE -- these are hoisted out of the guarded region at
  // the call site because they legitimately change every frame:
  //   drawCall.drawCallID   -> surface.objectPickingValue (per-frame draw counter)
  //   m_isInsideFrustum     -> surface.isInsideFrustum (moves with the camera)
  // and the m_isHidden promotions, which `entry` resets each frame.
  // NV-DXVK [perf] 2026-08-07: THE DRAW-SCOPED HALF OF THE INSTANCE STATE KEY.
  //
  // This used to be the whole key and ran once per INSTANCE. Every input it
  // reads is a property of the DRAW, the material, or the frame -- drawCall
  // (const), materialData (never mutated on this path), alphaState (a pure
  // function of those two), bindingEpoch (snapshotted once per frame in
  // SceneManager::onFrameEnd) and two option reads. The only input that varied
  // between the placements of a draw was categoryFlags, which now lives in
  // mixInstStateKey below.
  //
  // WHY THAT MATTERED. [Perf.SceneObj] callsPerFrame=15,665 against [ProcDCS]
  // draws=1,060: about 15 placements per draw, each repeating this identical
  // gather. And the gather IS the cost -- see the note below: it is the
  // SCATTERED READS across LegacyMaterialData, DrawCallTransforms, the
  // MaterialData variant, GeometryHashes and three samplers, all of them
  // draw-scoped objects. Process note #2 in the CPU handoff ("a probe billed per
  // DRAW but running per INSTANCE is off by ~15x") for the fifth time in this
  // file, and the first time on code that is not a probe.
  //
  // NOT A MEMO. There is no detector, no key on the memo, and nothing to miss:
  // the caller that owns the draw computes this once and hands it down. That is
  // the difference from the geometryAssetHash memo recorded further down, which
  // was measured and reverted -- it stayed inside the per-instance call and paid
  // a TLS guard per access to skip work smaller than the guard.
  //
  // The absolute key value is NOT preserved across this change and does not need
  // to be: the key is only ever compared against the same instance's key from
  // the previous frame, so the equivalence classes are what matter, and hashing
  // the draw digest with categoryFlags preserves them exactly. One frame of
  // "everything changed" on the first frame after the switch, then steady.
  static XXH64_hash_t computeDrawStateKey(const DrawCallState& drawCall,
                                          const MaterialData& materialData,
                                          const RtSurface::AlphaState& alphaState,
                                          bool vsDebugIdConsumed,
                                          uint64_t bindingEpoch) {
    const LegacyMaterialData& lm = drawCall.getMaterialData();

    // ONE KEY, TWO DECISIONS. This digest feeds BOTH the `surf` guard and the
    // `tail` event-fanout gate.
    //
    // WHY MERGED. Measured: with a key each, surf paid ~30 ns to skip ~25 ns of
    // writes -- a net LOSS (surf 0.211 baseline -> 0.250 at v1, -> 0.233 after the
    // key shrank) -- while tail paid ~30 ns to skip ~50 ns, a win. Sharing one
    // gather makes surf's share free: the union of both input sets costs barely
    // more to hash than either alone, because the cost here is the SCATTERED READS
    // (LegacyMaterialData, DrawCallTransforms, the MaterialData variant,
    // GeometryHashes, three samplers), not the hashing.
    //
    // SIZE. XXH3_64bits tiers at <=16, <=128, <=240 and >240 bytes. The original
    // 152-byte version sat in the <=240 tier and that WAS costing -- but the
    // boundary that matters is 128, not 64 (an earlier comment here said 64; that
    // was wrong). At 88 bytes this stays in the same <=128 tier a 56-byte key
    // would, so merging the tail fields in is very nearly free.
    //
    // The 48-byte eye texture transform is deliberately NOT here: eye draws never
    // skip (see the isEye escape at the call site), which costs nothing -- there
    // are a handful of them -- and keeps every other instance out of the next tier.
    struct InstStateKeyData {
      XXH64_hash_t materialHash;       // full render-material digest   (surf + tail)
      XXH64_hash_t legacyHash;         // 11 tex hashes + blend + alpha (surf)
      XXH64_hash_t geometryAssetHash;  // drawCall.getHash(rule)        (surf)
      uint64_t     vertexShaderHash;   // vsDebugId source              (surf)
      uint64_t     samplerOverride;    // createSurfaceMaterial inputs  (tail)
      uint64_t     drawSampler;        //                               (tail)
      uint64_t     drawSampler2;       //                               (tail)
      uint64_t     bindingEpoch;       // cache-slot recycling stamp    (tail)
      uint32_t     typeAndTexgen;      // materialType | texgenMode << 16
      uint32_t     tFactor;            //                               (surf)
      uint32_t     texArgOps;          // six arg/op enums, 5 bits each (surf)
      uint32_t     alphaAndMisc;       // alpha state + misc flag bits  (surf + tail)
    };
    // NOTE: categoryFlags is deliberately absent -- it is the one per-instance
    // input, and mixing it in here is what forced the whole gather to run per
    // instance. Anything you add to this struct MUST be draw-scoped; if it can
    // differ between two placements of one draw, it belongs in mixInstStateKey.
    static_assert(sizeof(InstStateKeyData) <= 128,
                  "InstStateKeyData must stay <=128 bytes to keep XXH3 off its "
                  "long-input path; if you must add a field, pack it into an "
                  "existing word.");
    // Same reasoning as LegacyMaterialData::updateCachedHash: memset the whole
    // object so trailing/interior padding is deterministic. Do NOT switch to an
    // aggregate initialiser -- that copies a temporary's padding and produced a
    // material hash that changed with the call path (see the note there).
    static_assert(alignof(InstStateKeyData) <= 8 && sizeof(InstStateKeyData) % 4 == 0);
    InstStateKeyData kd;
    std::memset(&kd, 0, sizeof(kd));

    kd.materialHash      = materialData.getHash();
    kd.legacyHash        = lm.getHash();
    // tail-only inputs: createSurfaceMaterial resolves a sampler from these three
    // pointers, and the binding epoch says whether any previously-handed-out cache
    // index could have been recycled since. See ResourceCache::getBindingEpoch().
    kd.samplerOverride   = reinterpret_cast<uint64_t>(materialData.getSamplerOverride().ptr());
    kd.drawSampler       = reinterpret_cast<uint64_t>(lm.getSampler().ptr());
    kd.drawSampler2      = reinterpret_cast<uint64_t>(lm.getSampler2().ptr());
    kd.bindingEpoch      = bindingEpoch;
    // surface.associatedGeometryHash is assigned from this exact expression, and
    // the rule is an RtxOption that the user can change at runtime, so the rule's
    // effect has to be baked in rather than assumed constant.
    //
    // NV-DXVK [perf] 2026-08-07 -- A SINGLE-ENTRY MEMO WAS TRIED HERE AND REVERTED.
    // MEASURED, so it is not re-attempted.
    //
    // The observation that motivated it is real and still stands:
    // GeometryHashes::getHashForRule(rule) dispatches to its precombined[] cache
    // for five known rules (rtx_hashing.h ~117), but the configured asset rule --
    // rtx.geometryAssetHashRuleString = "positions,indices,geometrydescriptor",
    // i.e. VertexPosition|Indices|GeometryDescriptor -- matches NONE of them
    // (TopologicalHash has no positions, VertexDataHash no indices or descriptor,
    // FullGeometryHash carries extras). So this call falls through to
    // getHashForRuleImpl, a 9-iteration loop with a chained XXH64 per selected
    // component, once per INSTANCE (~15,500/frame) against only ~1,060 draws.
    //
    // The memo keyed on the three selected component hashes plus legacyHash, which
    // is exact by construction, and consecutive instances of a fanout batch share a
    // draw so it should have hit ~93%. It still LOST: [Perf.SceneObj] update went
    // 9.52 -> 9.96 ms between two settled, exporter-free captures, and this was one
    // of only two changes inside updateInstance between them.
    //
    // WHY, most likely: `static thread_local` with a non-trivial initialiser is
    // dynamically initialised, so every access pays a TLS-init guard check on top
    // of three field loads and four compares -- more than the loop it skipped.
    // Same lesson as the surf guard: a detector that touches more memory than the
    // work it avoids cannot win, and the stage is the only thing that decides it.
    //
    // The real fix is upstream: give the configured asset rule a precombined slot
    // so this becomes an array read like the other five. Done -- see
    // GeometryHashes::precombine() / getHashForRule(). The memo below is retained
    // on top of it.
    {
      static constexpr uint32_t kMemoMask =
          (1u << static_cast<uint32_t>(HashComponents::VertexPosition))
        | (1u << static_cast<uint32_t>(HashComponents::Indices))
        | (1u << static_cast<uint32_t>(HashComponents::GeometryDescriptor));

      const HashRule& assetRule = RtxOptions::geometryAssetHashRule();

      if (assetRule.raw() != kMemoMask) {
        kd.geometryAssetHash = drawCall.getHash(assetRule);
      } else {
        const GeometryHashes& gh = drawCall.getGeometryData().hashes;
        const XXH64_hash_t hPos  = gh[HashComponents::VertexPosition];
        const XXH64_hash_t hIdx  = gh[HashComponents::Indices];
        const XXH64_hash_t hDesc = gh[HashComponents::GeometryDescriptor];

        // NV-DXVK [perf]: PLAIN STATICS, NOT thread_local-with-initialiser. The
        // first version used `static thread_local AssetHashMemo memo` with default
        // member initialisers, which makes it DYNAMICALLY initialised -- every
        // access pays a TLS-init guard check, and update measured 9.52 -> 9.96 ms.
        // Constant-initialised thread_local scalars need no guard: they land in
        // .tbss and cost one TLS base fetch shared by all five.
        static thread_local XXH64_hash_t memoPos    = 0;
        static thread_local XXH64_hash_t memoIdx    = 0;
        static thread_local XXH64_hash_t memoDesc   = 0;
        static thread_local XXH64_hash_t memoLegacy = 0;
        static thread_local XXH64_hash_t memoOut    = 0;
        static thread_local bool         memoValid  = false;

        if (memoValid && memoPos == hPos && memoIdx == hIdx
            && memoDesc == hDesc && memoLegacy == kd.legacyHash) {
          kd.geometryAssetHash = memoOut;
        } else {
          kd.geometryAssetHash = drawCall.getHash(assetRule);
          memoPos = hPos; memoIdx = hIdx; memoDesc = hDesc;
          memoLegacy = kd.legacyHash;
          memoOut = kd.geometryAssetHash;
          memoValid = true;
        }
      }
    }
    kd.vertexShaderHash  = static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash);
    kd.tFactor           = lm.tFactor;
    kd.typeAndTexgen     = static_cast<uint32_t>(materialData.getType())
                         | (static_cast<uint32_t>(drawCall.getTransformData().texgenMode) << 16);

    // Six enums, five bits each = 30 bits. RtTextureArgSource and
    // DxvkRtTextureOperation are both small enumerations; the masks are asserted
    // rather than assumed so a new enumerator cannot silently alias two states
    // into one key and serve a surface the wrong texture stage setup.
    auto arg5 = [](auto e) -> uint32_t {
      const uint32_t v = static_cast<uint32_t>(e);
      assert(v < 32u && "texture arg/op enum outgrew its 5-bit slot in SurfKeyData");
      return v & 0x1Fu;
    };
    kd.texArgOps         = (arg5(lm.textureColorArg1Source)      )
                         | (arg5(lm.textureColorArg2Source)  << 5)
                         | (arg5(lm.textureColorOperation)   << 10)
                         | (arg5(lm.textureAlphaArg1Source)  << 15)
                         | (arg5(lm.textureAlphaArg2Source)  << 20)
                         | (arg5(lm.textureAlphaOperation)   << 25);

    // alphaState is copied wholesale into surface.alphaState, so every field of it
    // is an input. Packed explicitly rather than memcmp'd so padding can never
    // silently widen the key.
    kd.alphaAndMisc      = (static_cast<uint32_t>(alphaState.alphaTestType)  & 0xFu)
                         | ((static_cast<uint32_t>(alphaState.alphaTestReferenceValue) & 0xFFu) << 4)
                         | ((static_cast<uint32_t>(alphaState.blendType)     & 0x1Fu) << 12)
                         | (alphaState.isBlendingDisabled ? 1u << 17 : 0u)
                         | (alphaState.isFullyOpaque      ? 1u << 18 : 0u)
                         | (alphaState.invertedBlend      ? 1u << 19 : 0u)
                         | (alphaState.emissiveBlend      ? 1u << 20 : 0u)
                         | (alphaState.isParticle         ? 1u << 21 : 0u)
                         | (alphaState.isDecal            ? 1u << 22 : 0u)
                         | (lm.isTextureFactorBlend       ? 1u << 23 : 0u)
                         | (lm.isVertexColorBakedLighting ? 1u << 24 : 0u)
                         // vsDebugId is written from acquireVsDebugId only when a
                         // consumer is on, and forced to 0 otherwise -- so the
                         // consumer state itself changes the written value.
                         | (vsDebugIdConsumed             ? 1u << 25 : 0u)
                         // Anti-culling decides whether surface.isInsideFrustum
                         // tracks the instance flag or is pinned true. The flag is
                         // written outside the guard, but the MODE still selects
                         // which of the two the block would write.
                         | (RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1u << 26 : 0u)
                         // NV-DXVK [perf] fastInstanceUpdate: the camera type was
                         // never keyed -- the Sky-camera hide and the ViewModel
                         // handling read it outside the surf guard every frame.
                         // The fast path retains those outcomes, so a camera-type
                         // change must break the key. Per instance the camera is
                         // stable frame-over-frame, so this does not reduce the
                         // existing surf/tail skip rates measurably.
                         | ((static_cast<uint32_t>(drawCall.cameraType) & 0xFu) << 27);

    return XXH3_64bits(&kd, sizeof(kd));
  }

  // NV-DXVK [perf] 2026-08-07: the per-instance half. Everything expensive was
  // already folded into drawStateKey by the caller that owns the draw; all that
  // is left is the one input that genuinely varies between placements.
  //
  // categoryFlags is not simply drawCall.getCategoryFlags(): updateInstance
  // OR-preserves a sticky IgnoreAntiCulling bit onto it, so two placements of the
  // same draw CAN legitimately differ here. That is precisely why it is mixed per
  // instance rather than hoisted with the rest.
  //
  // 16 bytes lands in XXH3's smallest input tier, and both operands are already
  // in registers -- no pointer chasing at all, which was the entire cost of the
  // draw-scoped half.
  static XXH64_hash_t mixInstStateKey(XXH64_hash_t drawStateKey,
                                      const CategoryFlags& categoryFlags) {
    struct InstMixData {
      XXH64_hash_t drawStateKey;
      uint32_t     categoryFlags;
      uint32_t     reserved;
    };
    static_assert(sizeof(InstMixData) == 16,
                  "InstMixData must stay in XXH3's <=16-byte tier.");
    // Same padding discipline as InstStateKeyData: memset rather than an
    // aggregate initialiser, so `reserved` and any interior padding are
    // deterministic and the digest cannot vary by call path.
    InstMixData md;
    std::memset(&md, 0, sizeof(md));
    md.drawStateKey  = drawStateKey;
    md.categoryFlags = static_cast<uint32_t>(categoryFlags.raw());
    return XXH3_64bits(&md, sizeof(md));
  }

  // NV-DXVK [perf] 2026-08-07: everything an instance needs from its draw call
  // that does not depend on which placement it is. Built once per draw by the
  // callers below and handed down through processSceneObjectImpl.
  //
  // alphaState is in here for the same reason as the key: calculateAlphaState is
  // a pure function of (drawCall, materialData), and it was being recomputed for
  // every placement so it could be copied wholesale into surface.alphaState.
  // NV-DXVK [perf] fastInstanceUpdate: FNV digest of every runtime option that
  // the fast path's SKIPPED region reads. The fast path retains last frame's
  // derived state (vkInstance flags, mask, hidden/fog outcomes, dev options),
  // which is only sound while these options hold the values that derived it.
  // Comparing the digest across frames turns "an option changed" into "one full
  // slow frame", with zero per-instance cost. If you add an option read to any
  // region the fast path skips (the surf switch, the xform block, the flags/mask
  // block, applyDeveloperOptions), it MUST be added here or be covered by the
  // instance state key.
  static uint64_t computeFastPathOptionsDigest() {
    uint64_t d = 0xcbf29ce484222325ull;
    auto mix = [&d](uint64_t v) {
      d = (d ^ v) * 0x100000001b3ull;
    };
    mix(RtxOptions::enableCulling() ? 1u : 0u);
    mix(RtxOptions::tf2StableBackfaceCull() ? 1u : 0u);
    mix(RtxOptions::enableTf2SkyboxCloudFog() ? 1u : 0u);
    mix(RtxOptions::enableSeparateUnorderedApproximations() ? 1u : 0u);
    mix(RtxOptions::getEnableOpacityMicromap() ? 1u : 0u);
    mix(RtxOptions::enableInstanceDebuggingTools() ? 1u : 0u);
    mix(RtxOptions::flipSubViewSkyboxNormals() ? 1u : 0u);
    mix(RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1u : 0u);
    {
      const float off = RtxOptions::worldSpaceUiBackgroundOffset();
      uint32_t bits;
      std::memcpy(&bits, &off, sizeof(bits));
      mix(bits);
    }
    return d;
  }

  bool InstanceManager::fastPathOptionsStable() const {
    const uint32_t fid = m_device->getCurrentFrameId();
    if (m_fastOptFrame.load(std::memory_order_acquire) != fid) {
      std::lock_guard<std::mutex> g(m_fastOptMutex);
      if (m_fastOptFrame.load(std::memory_order_relaxed) != fid) {
        const uint64_t d = computeFastPathOptionsDigest();
        // Stable only when this frame's digest equals the previous frame's --
        // the first frame after any change (including the very first frame)
        // runs fully slow, which re-derives everything under the new values.
        m_fastOptStable = (d == m_fastOptDigest) && (m_fastOptDigest != 0);
        m_fastOptDigest = d;
        m_fastOptFrame.store(fid, std::memory_order_release);
      }
    }
    // m_fastOptStable was written before the release-store the acquire above
    // synchronised with; racing threads at the frame boundary that lose the
    // lock race simply read the freshly published value.
    return m_fastOptStable;
  }

  InstanceManager::DrawScopedState InstanceManager::computeDrawScopedState(
      const BlasEntry& blas, const DrawCallState& drawCall, const MaterialData& materialData) const {
    DrawScopedState state;
    state.alphaState = calculateAlphaState(drawCall, materialData);
    // NV-DXVK [perf] 2026-08-07: resolved HERE rather than in every placement's
    // processInstanceBuffers. Safe because blas.modifiedGeometryData is written
    // only by SceneManager::processDrawCallState, which completes every write
    // before it calls into this manager -- so the binding cannot move under the
    // placement loop and needs no freshness key. Left default when the hoist is
    // off, and processInstanceBuffers then never reads it.
    if (RtxOptions::hoistSurfaceBufferBinding()) {
      state.buffers = resolveSurfaceBufferBinding(blas);
    }
    // Eye draws never take the skip (see the isEye escape at the consumer), so
    // there is nothing for a key to authorise and no reason to build one.
    state.keyEligible = !drawCall.isEye();
    state.stateKey = state.keyEligible
      ? computeDrawStateKey(drawCall, materialData, state.alphaState,
                            vsDebugIdIsConsumed(), m_pResourceCache->getBindingEpoch())
      : kEmptyHash;

    // NV-DXVK [perf] fastInstanceUpdate: the draw-scoped inputs of the skipped
    // flags/mask stage that the state key does not cover. Compared against
    // RtInstance::m_fastDrawBits (stored at the last full update); any
    // difference forces the full path so determineInstanceFlags re-runs.
    {
      const auto& td = drawCall.getTransformData();
      uint8_t bits = 0;
      if (drawCall.getGeometryData().frontFace == VkFrontFace::VK_FRONT_FACE_CLOCKWISE) {
        bits |= 1u;
      }
      if (drawCall.isUsingRaytracedRenderTarget) {
        bits |= 2u;
      }
      if (td.isSubView) {
        bits |= 4u;
      }
      if (td.isSubViewSkybox) {
        bits |= 8u;
      }
      // Projection winding parity, only an input when tf2StableBackfaceCull is
      // on. det(v2p * w2v) sign == det(v2p) sign XOR det(w2v) sign, so the two
      // cheap 3x3 tests replace the 4x4 multiply determineInstanceFlags pays.
      // Used strictly as a change detector, so exact agreement with the flags
      // stage's own computation is not required -- only faithfulness to its
      // inputs, which sign-multiplicativity provides.
      if (RtxOptions::tf2StableBackfaceCull()
          && (isMirrorTransform(td.viewToProjection) != isMirrorTransform(td.worldToView))) {
        bits |= 16u;
      }
      state.fastDrawBits = bits;
    }
    state.fastPathAllowed = state.keyEligible
      && RtxOptions::fastInstanceUpdate()
      && !RtxOptions::enableInstanceDebuggingTools()
      && fastPathOptionsStable();
    return state;
  }

  void InstanceManager::mergeInstanceHeuristics(RtInstance& instanceToModify, const DrawCallState& drawCall, const RtSurface::AlphaState& alphaState) const {
    // "Opaqueness" takes priority!
    if (
      (alphaState.isFullyOpaque || alphaState.alphaTestType == AlphaTestType::kAlways) &&
      !(instanceToModify.surface.alphaState.isFullyOpaque || instanceToModify.surface.alphaState.alphaTestType == AlphaTestType::kAlways)
    ) {
      instanceToModify.surface.alphaState = alphaState;
    }

    // NOTE: In the future we could extend this with heuristics as needed...
  }

  RtInstance* InstanceManager::findSimilarInstance(BlasEntry& blas, const MaterialData& material, const Matrix4& firstInstanceObjectToWorld, CameraType::Enum cameraType, const RayPortalManager& rayPortalManager, uint64_t stablePropId, DrawCallCache* drawCallCache, const Matrix4* prevObjectToWorld, XXH64_hash_t* outQueryMatrixHash) {
    // NV-DXVK [perf] handoff v7 sec 4a: cleared up front so every early return
    // below leaves it defined. The exact stage overwrites it unconditionally a
    // few dozen lines down, before any of them can be taken; 0 is the safe value
    // for the paths that bypass it entirely, since 0 means "hash it yourself".
    if (outQueryMatrixHash != nullptr) {
      *outQueryMatrixHash = 0;
    }

    // Disable temporal correlation between instances so that duplicate instances are not created
    // should a developer option change instance enough for it not to match anymore
    if (RtxOptions::enableInstanceDebuggingTools()) {
      return nullptr;
    }

    RtInstance* result = nullptr;

    const uint32_t currentFrameIdx = m_device->getCurrentFrameId();
    // NV-DXVK [perf] 2026-08-07: DECLARED here, ASSIGNED past the exact stage.
    // Computing it is a pointer chase into blas.input's geometry plus a matrix
    // transform of the bounding-box centroid, and nothing before the nearest stage
    // reads it -- the exact-hash lookup returns without touching it, and so does
    // the engine-history retry. With static=97% and addedPct=0 the exact stage
    // answers almost every call, so this was computed and discarded on the
    // overwhelming majority of instances.
    //
    // It cannot simply move inside the search block: that block closes before the
    // ray-portal virtual-instance code below, which reads worldPosition too. So the
    // declaration stays at function scope and the assignment sits at the one point
    // every surviving path passes through.
    //
    // INVARIANT, and the reason this is safe: the assignment is unconditional
    // within the search block, placed after both early returns. Any path that
    // reaches this variable's later uses has executed it. If you add an early exit
    // to the search block that falls through rather than returning, move the
    // assignment above it.
    Vector3 worldPosition;

    const float uniqueObjectDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();

    // NV-DXVK [FindStage] 2026-08-07: per-frame census of how each lookup resolved.
    //
    // WHY THIS EXISTS. The standing claim about `find` is that some shaders'
    // stablePropId round-robins across frames, so the exact stage cannot match the
    // key its instance was filed under and those lookups fall through to the
    // nearest-neighbour search. That claim has never been MEASURED -- every probe
    // in this function is per-VS, throttled, and on the logDenyTags list, so none
    // of them can report a population. This can: three integer increments per call
    // and one line per frame, no timers, no sampling, nothing to floor out.
    //
    // READ IT LIKE THIS:
    //   propIdMiss ~ 0            -> the round-robin is not happening in this
    //                                scene. The lead is dead; do not force propIds
    //                                to 0 via rtx.suppressStablePropIdVsHashes,
    //                                and look elsewhere in find.
    //   propIdMiss large          -> confirmed. Every one of those pays the nearest
    //                                search AND skips the engine-history retry,
    //                                which is deliberately gated on propId == 0.
    //                                Fix belongs at the propId producer.
    //   withPropId ~ 0            -> nothing in this scene keys on prop identity at
    //                                all, so the round-robin cannot be the cost
    //                                whatever else is true.
    // NV-DXVK [Phase2b]: was `static thread_local FindStageAgg` — with find
    // sharded across up to 30 workers a per-thread census fragments into 30
    // partial lines and the Sec-15 "exact= ~97%" check stops being one number.
    // File-scope relaxed atomics keep it a single truthful census (the same
    // relaxed-counter shape rtx.logSurfaceGeomDiag documents as ~free); the
    // frame CAS elects exactly one thread to emit-and-reset per rollover.
    // Emission reads are relaxed loads racing late stragglers of the old frame
    // by at most a few counts — a diagnostic tolerance, not a correctness one.
    struct FindStageAgg {
      std::atomic<uint32_t> frame { 0xFFFFFFFFu };
      std::atomic<uint32_t> calls { 0 }, exact { 0 }, withPropId { 0 }, propIdMiss { 0 }, noPropIdMiss { 0 };
    };
    static FindStageAgg sFindStage;
    {
      uint32_t seenFrame = sFindStage.frame.load(std::memory_order_relaxed);
      if (seenFrame != currentFrameIdx
          && sFindStage.frame.compare_exchange_strong(seenFrame, currentFrameIdx, std::memory_order_relaxed)) {
        const uint32_t emitCalls = sFindStage.calls.exchange(0, std::memory_order_relaxed);
        const uint32_t emitExact = sFindStage.exact.exchange(0, std::memory_order_relaxed);
        const uint32_t emitWith  = sFindStage.withPropId.exchange(0, std::memory_order_relaxed);
        const uint32_t emitPm    = sFindStage.propIdMiss.exchange(0, std::memory_order_relaxed);
        const uint32_t emitNpm   = sFindStage.noPropIdMiss.exchange(0, std::memory_order_relaxed);
        if (seenFrame != 0xFFFFFFFFu && emitCalls > 0u) {
          Logger::info(str::format(
            "[FindStage] f=", seenFrame,
            " calls=", emitCalls,
            " exact=", emitExact,
            " withPropId=", emitWith,
            " propIdMiss=", emitPm,
            " noPropIdMiss=", emitNpm));
        }
      }
    }

    // NV-DXVK [MapSupply] 2026-08-25: PER-BLASENTRY supply, per frame.
    //
    // THE QUESTION THIS FILE HAS ASKED AND NEVER ANSWERED. The [FindSim] note
    // below records "98 queries against a map holding at most 63 instances, so
    // ~35 draws per frame CANNOT match anything no matter how good the matcher
    // is", and then says outright what it could not settle: "What that count
    // cannot say is whether the 98 hit ONE BlasEntry." Every count in this
    // function is either per-VS or per-frame-global, and a per-frame total
    // cannot be compared against a per-map size — the misses on any one frame
    // are spread across several BlasEntries, so pairing a frame's miss total
    // with one line's mapSz overstates the shortfall. This pairs them per map.
    //
    // PRE-REGISTERED READING, written before the capture:
    //   deficit <= 0 on the maps carrying the misses
    //       No pigeonhole. Every placement COULD have matched, so the failures
    //       are ordering and claim defects — §5.1's greedy first-come-first-
    //       served claim at the nearest stage — and the fix belongs in the
    //       matcher. This is the branch that justifies restructuring batch
    //       resolution, which §5.3 refused on different evidence.
    //   deficit > 0 on those maps
    //       The pigeonhole is real and the matcher is exonerated: that many
    //       placements have no instance to find, so no matching rule can serve
    //       them. The fix is upstream — either instances for this mesh are being
    //       split across sibling BlasEntries (drawCallCache routing) or they are
    //       never created — and `linked` says which, because the spatial map and
    //       the linked-instance list agree by construction when routing is sane.
    //
    // AND THE FIELD THAT SEPARATES THOSE TWO UPSTREAM CAUSES. `linked` is the
    // BlasEntry's own instance list. [FindSim] already reads spatialMapSize ==
    // linkedInst in 145 of 145 samples, so a deficit is NOT the map losing what
    // the BLAS holds; it means the BLAS itself holds too few, and then the
    // question is which BlasEntry has the rest.
    //
    // COST. One masked probe and three relaxed increments per lookup, ~351
    // lookups a frame. No string is built except the single emission line.
    struct MapSupplyAgg {
      // Sized well above the ~850 live BlasEntries so the probe run stays short;
      // power of two so the index is a mask. Overflow is counted, never silent.
      //
      // An ENUMERATOR, not a static constexpr member: this struct is local to
      // findSimilarInstance and a local class may not have static data members
      // (C2246). An enumerator with an explicit size_t underlying type gives the
      // same constant with the same type and no conversion warnings under /WX.
      enum : size_t { kSlots = 2048 };
      struct Slot {
        std::atomic<uintptr_t> blas   { 0 };
        std::atomic<uint32_t>  q      { 0 };  // lookups against this map
        std::atomic<uint32_t>  n      { 0 };  // lookups that reached the nearest search
        std::atomic<uint32_t>  sz     { 0 };  // spatial map size at query time
        std::atomic<uint32_t>  linked { 0 };  // instances linked to this BlasEntry
      };
      std::atomic<uint32_t> frame { 0xFFFFFFFFu };
      std::atomic<uint32_t> overflow { 0 };
      Slot slots[kSlots];
    };
    static MapSupplyAgg sMapSupply;

    // Frame rollover: one thread emits the worst maps and clears the table.
    // Same election shape as [FindStage] above, and the same tolerance — a late
    // straggler from the old frame lands in the new frame's counts, which moves
    // a deficit by ones and never by the tens the verdict turns on.
    {
      uint32_t seenFrame = sMapSupply.frame.load(std::memory_order_relaxed);
      if (seenFrame != currentFrameIdx
          && sMapSupply.frame.compare_exchange_strong(seenFrame, currentFrameIdx, std::memory_order_relaxed)) {
        if (seenFrame != 0xFFFFFFFFu) {
          // Rank by deficit, not by query count: a map serving 300 lookups from
          // 300 entries is healthy and a map serving 40 from 8 is not, and
          // sorting by traffic would put the healthy one at the top every frame.
          struct Worst { uint32_t q, n, sz, linked; };
          Worst worst[6] = { };
          int32_t worstDef[6] = { };
          uint32_t maps = 0, deficitMaps = 0, totalQ = 0, totalDeficit = 0;
          for (size_t i = 0; i < MapSupplyAgg::kSlots; ++i) {
            MapSupplyAgg::Slot& s = sMapSupply.slots[i];
            const uint32_t q = s.q.exchange(0, std::memory_order_relaxed);
            if (q == 0u) {
              s.blas.store(0, std::memory_order_relaxed);
              continue;
            }
            const uint32_t n  = s.n.exchange(0, std::memory_order_relaxed);
            const uint32_t sz = s.sz.load(std::memory_order_relaxed);
            const uint32_t lk = s.linked.load(std::memory_order_relaxed);
            s.blas.store(0, std::memory_order_relaxed);
            ++maps;
            totalQ += q;
            const int32_t def = static_cast<int32_t>(q) - static_cast<int32_t>(sz);
            if (def > 0) {
              ++deficitMaps;
              totalDeficit += static_cast<uint32_t>(def);
            }
            for (int32_t w = 0; w < 6; ++w) {
              if (def > worstDef[w]) {
                for (int32_t k = 5; k > w; --k) {
                  worstDef[k] = worstDef[k - 1];
                  worst[k] = worst[k - 1];
                }
                worstDef[w] = def;
                worst[w] = Worst { q, n, sz, lk };
                break;
              }
            }
          }
          const uint32_t ovf = sMapSupply.overflow.exchange(0, std::memory_order_relaxed);
          if (maps != 0u) {
            std::string worstStr;
            for (int32_t w = 0; w < 6 && worstDef[w] > 0; ++w) {
              worstStr += str::format(
                " [q=", worst[w].q, " sz=", worst[w].sz, " linked=", worst[w].linked,
                " nearest=", worst[w].n, " deficit=", worstDef[w], "]");
            }
            Logger::info(str::format(
              "[MapSupply] f=", seenFrame,
              " maps=", maps,
              " queries=", totalQ,
              " deficitMaps=", deficitMaps,
              " deficitTotal=", totalDeficit,
              // Non-zero means some map's lookups were dropped from the census
              // entirely, so deficitTotal is a lower bound on that frame.
              " overflow=", ovf,
              " worst:", worstStr.empty() ? std::string(" none") : worstStr));
          }
        }
      }
    }

    // Claim this BlasEntry's slot. Returns kSlots on overflow, which is counted
    // rather than folded into a neighbouring map's numbers — a census that
    // silently attributed one map's queries to another would invent exactly the
    // deficit it is meant to measure.
    size_t mapSupplySlot = MapSupplyAgg::kSlots;
    {
      const uintptr_t blasKey = reinterpret_cast<uintptr_t>(&blas);
      // Pointer bits 0-3 are always zero for a heap object of this size, so the
      // low bits alone would collide every allocation onto few buckets.
      size_t idx = static_cast<size_t>((blasKey >> 4) * 0x9E3779B97F4A7C15ull >> 48)
                 & (MapSupplyAgg::kSlots - 1);
      for (uint32_t probe = 0; probe < 16u; ++probe) {
        std::atomic<uintptr_t>& owner = sMapSupply.slots[idx].blas;
        uintptr_t cur = owner.load(std::memory_order_relaxed);
        if (cur == blasKey) {
          mapSupplySlot = idx;
          break;
        }
        if (cur == 0u && owner.compare_exchange_strong(cur, blasKey, std::memory_order_relaxed)) {
          mapSupplySlot = idx;
          break;
        }
        if (cur == blasKey) {           // lost the race to another worker, same map
          mapSupplySlot = idx;
          break;
        }
        idx = (idx + 1u) & (MapSupplyAgg::kSlots - 1);
      }
      if (mapSupplySlot != MapSupplyAgg::kSlots) {
        MapSupplyAgg::Slot& s = sMapSupply.slots[mapSupplySlot];
        s.q.fetch_add(1, std::memory_order_relaxed);
        // Last writer wins. Both are stable within a frame — Phase2b applies
        // every map write in the ordered tail, after this phase — so there is
        // nothing for a racing store to smear.
        s.sz.store(static_cast<uint32_t>(blas.getSpatialMap().size()), std::memory_order_relaxed);
        s.linked.store(static_cast<uint32_t>(blas.getLinkedInstances().size()), std::memory_order_relaxed);
      } else {
        sMapSupply.overflow.fetch_add(1, std::memory_order_relaxed);
      }
    }

    RtInstance* pSimilar = nullptr;
    float nearestDistSqr = FLT_MAX;

    // NV-DXVK [VS_2904d2 findSimilar probe]: diagnose why dedup fails for
    // the sub-view mountain content. After our reproject is locked (scale
    // and anchor pinned to map constants), T_reproject is bytewise stable
    // per frame, so post-reproject world positions SHOULD be byte-identical
    // across frames for static props — and getDataAtTransform's exact-hash
    // lookup should hit every time. Yet InstCounts shows created=48 each
    // frame. We need to know per-stage what's failing.
    const XXH64_hash_t vsHashProbe = blas.input.getTransformData().vertexShaderHash;
    // Hardcoded hash kept so the original 2904d2 capture still reproduces;
    // rtx.findSimilarProbeVsHashes aims the same probe at anything else
    // (e.g. the tree-billboard VS) without a rebuild.
    // NV-DXVK 2026-08-06: tagDenied gates — [FindSim] was 46 lines/frame in
    // the 00:48 capture. Gating the flag rather than each emit covers every
    // [FindSim]/[FindSimMtn] site in this function at once; the throttle
    // counters simply stop advancing while denied, which is the same state a
    // fresh run starts in.
    // NV-DXVK [perf] 2026-08-07: HOISTED INTO A STATIC. Logger::tagDenied is not a
    // flag read -- it strlen()s the tag, indexes the bucket by tag[1] and memcmps
    // every entry in that bucket. 'F' is a populated bucket here ([FindSim],
    // [FanoutSplit], [FanoutPrev]), and this ran PER INSTANCE, ~15,500 times a
    // frame, alongside the second one below. The denylist is published once by
    // Logger::setDenyTags during RtxOptions init and never changes afterwards, so
    // a function-local static is the correct lifetime -- the same fix already
    // applied to kUVxDenied further down this file.
    static const bool kFindSimDenied = Logger::tagDenied("[FindSim]");
    const bool isProbeVS = !kFindSimDenied
                        && ((vsHashProbe == 0x2904d2163ef31a17ull)
                         || lookupHash(RtxOptions::findSimilarProbeVsHashes(), vsHashProbe));
    // NV-DXVK [FindSimMtn]: the s2s "two views" black-sky root. The reprojected
    // 3D-skybox terrain VS 0x29146e1d places 14 panels every frame in both
    // views (MtnPlace=14, all reproject gates PASS), but only 13 survive to the
    // sub-view in view1 vs 9 in view2 — 4-5 panels collapse here in dedup
    // (hidden=0/notInFr=0, no reap). This probe logs, per panel, whether it
    // collapses via EXACT (getDataAtTransform on propId/transform) or NEAREST
    // (getNearestData within uniqueObjectDistanceSqr) — telling us if view2's
    // extra collapses are propId collisions (engine reused a stablePropId for
    // distinct panels) or distance merges (panels fall within the dedup radius
    // after reproject). Own counters so it captures full frames of all 14.
    // NV-DXVK [perf] 2026-08-07: hoisted for the same reason as kFindSimDenied.
    // Note this tag is NOT covered by the "[FindSim]" denylist entry even though
    // it starts the same way -- tagDenied matches whole entries, and "[FindSim]"
    // includes the closing bracket, so it cannot prefix-match "[FindSimMtn]".
    static const bool kFindSimMtnDenied = Logger::tagDenied("[FindSimMtn]");
    const bool isMtnProbe = !kFindSimMtnDenied
                         && (vsHashProbe == 0x29146e1dd50b0314ull);

    // Search the BLAS for an instance matching ours
    {
      // Search for an exact match. stablePropId (passed in from the
      // caller's drawCall.transformData) overrides the matrix-bytes
      // cache key when non-zero — anchors dedup to per-prop identity.
      // NV-DXVK [perf] handoff v7 sec 4a: this is the one place the query matrix
      // is hashed, so it is the one place that can hand the hash onward. Written
      // whether or not the lookup hits: a nearest-stage or engine-class match
      // resolves to an instance that will still be re-filed under THIS matrix by
      // onTransformChanged, so the hash is just as reusable on a miss.
      result = const_cast<RtInstance*>(blas.getSpatialMap().getDataAtTransform(firstInstanceObjectToWorld, stablePropId, outQueryMatrixHash));
      const bool exactHit = (result != nullptr);
      // See RtInstance::m_claimStage. Stamped at the claim rather than at the
      // caller, because only here is it known WHICH stage produced the match.
      if (exactHit) {
        result->m_claimStage = RtInstance::ClaimStage::Exact;
        result->m_claimFrame = currentFrameIdx;
      }

      // NV-DXVK [FindStage]: the whole census, at the one point that decides it.
      // A non-zero propId that missed here is the round-robin signature: the key
      // is position-independent, so a stationary prop can only miss if the ID
      // itself changed since the instance was filed.
      sFindStage.calls.fetch_add(1, std::memory_order_relaxed);
      if (stablePropId != 0ull) {
        sFindStage.withPropId.fetch_add(1, std::memory_order_relaxed);
      }
      if (exactHit) {
        sFindStage.exact.fetch_add(1, std::memory_order_relaxed);
      } else if (stablePropId != 0ull) {
        sFindStage.propIdMiss.fetch_add(1, std::memory_order_relaxed);
      } else {
        sFindStage.noPropIdMiss.fetch_add(1, std::memory_order_relaxed);
      }
      if (result != nullptr) {
        if (isProbeVS) {
          thread_local uint32_t sExactProbe = 0;
          if (sExactProbe < 8 || (sExactProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[FindSim] #", sExactProbe, " vs=0x", std::hex, vsHashProbe, std::dec,
              " stage=exact hit=1 propId=0x", std::hex, stablePropId, std::dec,
              " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
              " spatialMapSize=", blas.getSpatialMap().size(),
              " linkedInst=", blas.getLinkedInstances().size()));
          }
          sExactProbe += 1;
        }
        if (isMtnProbe) {
          thread_local uint32_t sMtnExact = 0;
          if (sMtnExact < 80u || (sMtnExact & 0xFFu) == 0u) {
            // NV-DXVK [perf]: computed here rather than eagerly at function entry.
            // This site is throttled AND behind isMtnProbe, so it runs a handful of
            // times per session instead of ~15,500 times per frame.
            const Vector3 mtnWorldPos =
              blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
            Logger::info(str::format(
              "[FindSimMtn] #", sMtnExact, " f=", currentFrameIdx,
              " outcome=COLLAPSE stage=exact propId=0x", std::hex, stablePropId, std::dec,
              " spatialMapSize=", blas.getSpatialMap().size(),
              " worldPos=(", mtnWorldPos.x, ",", mtnWorldPos.y, ",", mtnWorldPos.z, ")"));
          }
          sMtnExact += 1;
        }
        return result;
      }

      // NV-DXVK [fanout prev-transform identity] 2026-08-05: SECOND exact-stage
      // attempt, keyed on where the engine says this object stood LAST frame.
      //
      // The map files an instance under the hash of the transform it had when it
      // was inserted. A prop that moved therefore cannot be found by its current
      // transform — but its previous one is precisely the key it is filed under,
      // so this hits exactly, at any speed, with no distance tolerance and none
      // of the nearest stage's failure modes (the 300-unit-per-frame ceiling that
      // shrinks with framerate, no velocity prediction, and the wrong-neighbour
      // case where a fast prop steals a different prop's instance and inherits a
      // bad motion vector).
      //
      // Only attempted when the current transform missed, so a stationary prop —
      // 95% of the population, which hits above — pays nothing. Skipped when
      // keying on a stablePropId, since that key is already position-independent
      // and re-probing with the same override hash would just repeat the lookup.
      //
      // Nothing is restamped on a hit: the instance keeps being filed under its
      // CURRENT transform each frame (see RtInstance's spatial cache update), so
      // next frame's history lines up with next frame's key, and a static prop's
      // key still never changes — which keeps SpatialMap::move() a no-op for it
      // rather than forcing an erase+insert per instance per frame.
      if (prevObjectToWorld == nullptr && stablePropId == 0ull) {
        // THE STAGE COULD NOT RUN. Counted apart from prevMiss because the two
        // prescribe opposite fixes -- see m_fanoutPrevNullCount. Without this,
        // a draw carrying no history is indistinguishable from one that never
        // needed the retry, and both read as silence.
        ++m_fanoutPrevNullCount;
      }
      if (prevObjectToWorld != nullptr && stablePropId == 0ull) {
        result = const_cast<RtInstance*>(blas.getSpatialMap().getDataAtTransform(*prevObjectToWorld, 0ull));
        if (result != nullptr) {
          // Counter only — the claim this rests on is that the engine's history
          // is bit-exact, and this is the number that proves or refutes it in
          // one capture. If it stays ~0 while created/missProp stay non-zero,
          // the history is not reproducing last frame's key and I should say so
          // rather than leave a dead branch in the hot path.
          ++m_fanoutPrevHitCount;
          result->m_claimStage = RtInstance::ClaimStage::History;
          result->m_claimFrame = currentFrameIdx;
          return result;
        }
        ++m_fanoutPrevMissCount;

        // NV-DXVK [MapLedger] 2026-08-24b: THE WRITE-SIDE VERDICT, on every
        // miss rather than on the sampled few. See MapLedgerAgg in the header
        // for what each bucket means; that reading is pre-registered there and
        // is not to be revised after seeing the numbers.
        //
        // UNCAPPED ON PURPOSE, and it is the counters that make that safe: this
        // is ten integer increments and one O(1) table probe, with no string
        // built and nothing to sample. The capped detail line below carries the
        // fields that cost something. Trap §6.2 was an UNCAPPED probe aggregated
        // without binning; the fix for it is to bin, not to cap, and emitting
        // this census per frame beside [FanoutPrev] bins it by construction.
        {
          // The key the EXACT stage formed and missed on. computeKey reuses the
          // hash that stage already paid for when it is in hand, so this is a
          // compare and a select rather than a second XXH64 over 64 bytes.
          const XXH64_hash_t queryKey = BlasEntry::InstanceMap::computeKey(
              firstInstanceObjectToWorld, stablePropId,
              (outQueryMatrixHash != nullptr) ? *outQueryMatrixHash : 0);
          const auto rec = blas.getSpatialMap().debugLedgerLookup(queryKey);
          // WHOLE MATRIX, not the translation. See the header: moved= compares
          // translations only, and §5.4's premise needs byte identity.
          const bool sameBytes = (std::memcmp(&firstInstanceObjectToWorld, prevObjectToWorld,
                                              sizeof(Matrix4)) == 0);
          using LedgerOp = BlasEntry::InstanceMap::LedgerOp;
          m_mapLedgerAgg.miss.fetch_add(1, std::memory_order_relaxed);
          if (sameBytes) {
            m_mapLedgerAgg.sameBytes.fetch_add(1, std::memory_order_relaxed);
          }
          std::atomic<uint32_t>* bucket = nullptr;
          switch (rec.op) {
            case LedgerOp::Inserted:
              bucket = sameBytes ? &m_mapLedgerAgg.sameIns : &m_mapLedgerAgg.diffIns; break;
            case LedgerOp::Refiled:
              bucket = sameBytes ? &m_mapLedgerAgg.sameRefile : &m_mapLedgerAgg.diffRefile; break;
            case LedgerOp::Erased:
              bucket = sameBytes ? &m_mapLedgerAgg.sameErase : &m_mapLedgerAgg.diffErase; break;
            case LedgerOp::None:
              bucket = sameBytes ? &m_mapLedgerAgg.sameNone : &m_mapLedgerAgg.diffNone; break;
          }
          bucket->fetch_add(1, std::memory_order_relaxed);
          // Carried to the verdict line rather than read there: a none= count is
          // meaningless without it, and the two must not be able to drift apart.
          if (blas.getSpatialMap().debugLedgerEvicted() != 0ull) {
            m_mapLedgerAgg.evicted.fetch_add(1, std::memory_order_relaxed);
          }
        }

        // AND HOW FAR OFF THE HISTORY WAS, because the count alone cannot say
        // which of two opposite things went wrong and they want opposite fixes.
        //
        // getDataAtTransform is a hash of the matrix bytes, so one changed bit
        // and a completely wrong matrix miss identically. The translation
        // distance separates them:
        //
        //   dist ~ 0      the prop barely moved, so last frame's key and this
        //                 frame's history describe the same placement and the
        //                 BYTES still differ -- the transform is being
        //                 recomposed rather than carried, and the fix is at the
        //                 composition.
        //   dist large    the history genuinely describes a different placement
        //                 than the one filed last frame, so the engine's
        //                 previous-transform array is not paired with its
        //                 current one, and the fix is at the producer.
        //
        // AND WHETHER THE INSTANCE IS SITTING RIGHT THERE UNDER A DIFFERENT KEY,
        // which is the test that separates the two failure modes the verdict
        // line shows and the distance alone cannot.
        //
        // [FanoutPrev] reads prevHit=113 prevMiss=0 in steady state and then, on
        // isolated frames, prevHit=0 prevMiss=113 -- the entire population
        // flipping at once rather than a few placements drifting. An all-or-
        // nothing flip is not per-placement noise, and the two candidate causes
        // want different fixes:
        //
        //   nearest ~ 0        the instance IS in the map, a frame's worth of
        //                      motion away from where the history says it was.
        //                      The history is STALE BY A FRAME -- it points at
        //                      T(n-2) while the instance was filed under T(n-1)
        //                      -- so the producer skipped an advance and the fix
        //                      is at the phase, not at the matcher.
        //   nearest large      the instance genuinely is not there, so this is
        //                      new geometry entering view and a miss is correct.
        //
        // ownerFrame is printed beside it because "close" only means "the same
        // prop" if the entry was refreshed recently.
        //
        // PER FRAME, NOT PER SESSION. The first version of this used one session
        // cap of 200 and spent every line during load, on a cold SpatialMap
        // (mapSz=11..20 against 113 placements) where a miss is meaningless --
        // the same mistake the [RsGate] note in d3d11_rtx.cpp already records,
        // made again. A small per-frame cap samples the steady state instead.
        {
          static std::atomic<uint32_t> sPrevMissFrame { 0xFFFFFFFFu };
          static std::atomic<uint32_t> sPrevMissLines { 0 };
          constexpr uint32_t kMaxPrevMissPerFrame = 4u;
          uint32_t seen = sPrevMissFrame.load(std::memory_order_relaxed);
          if (seen != currentFrameIdx
              && sPrevMissFrame.compare_exchange_strong(seen, currentFrameIdx, std::memory_order_relaxed)) {
            sPrevMissLines.store(0, std::memory_order_relaxed);
          }
          if (sPrevMissLines.fetch_add(1, std::memory_order_relaxed) < kMaxPrevMissPerFrame) {
            const Vector3 curT = firstInstanceObjectToWorld[3].xyz();
            const Vector3 prvT = (*prevObjectToWorld)[3].xyz();
            const Vector3 d = curT - prvT;
            // The map is keyed on the transformed CENTROID, not the translation,
            // so the query has to be composed the same way the entries were or
            // the distance is measured against the wrong point. Computed after
            // the cap so only logged lines pay for it.
            const Vector3 prevCentroid =
              blas.input.getGeometryData().boundingBox.getTransformedCentroid(*prevObjectToWorld);
            Vector3 nearPos { 0.f, 0.f, 0.f };
            const RtInstance* nearOwner = nullptr;
            const float nearSqr =
              blas.getSpatialMap().debugClosestCachedDistSqr(prevCentroid, nearPos, &nearOwner);

            // NV-DXVK [MapLedger]: the sampled detail behind the census above.
            //
            // ledgSame= IS THE FIELD THAT CLOSES §5.5. §5.4 established that the
            // instance is not gone — 566 of 643 stationary misses had a live
            // neighbour 1-50 units away, clustered around 25 units across props
            // at unrelated world positions, which is systematic rather than
            // coincidence. The open question was who vacated the key. If the
            // ledger says Refiled and ledgSame=1, the instance sitting right
            // there IS the one that vacated it, and ledgOther= is the key it
            // went to: the prop's own instance was re-filed while the reader
            // kept querying the old key. ledgSame=0 on a Refiled says a
            // DIFFERENT instance took the slot, which is the mover-steals-the-
            // slot mechanism §5.5 names as its candidate.
            //
            // The pointer is compared and printed, never dereferenced: a ledger
            // record outlives the instance it names by design.
            const XXH64_hash_t queryKey = BlasEntry::InstanceMap::computeKey(
                firstInstanceObjectToWorld, stablePropId,
                (outQueryMatrixHash != nullptr) ? *outQueryMatrixHash : 0);
            const auto rec = blas.getSpatialMap().debugLedgerLookup(queryKey);
            using LedgerOp = BlasEntry::InstanceMap::LedgerOp;
            const char* ledgOpName =
                (rec.op == LedgerOp::Inserted) ? "ins"
              : (rec.op == LedgerOp::Refiled)  ? "refile"
              : (rec.op == LedgerOp::Erased)   ? "erase"
                                               : "none";

            // The neighbour's side of the comparison, hoisted because
            // calcFirstInstanceObjectToWorld returns by value and the address of
            // a temporary cannot be taken. Everything below is a read of a live
            // instance found by walking the cache this frame, not of a ledger
            // record, so dereferencing it is safe.
            const bool  nearValid  = (nearOwner != nullptr);
            const Matrix4 nearWriteMtx = nearValid ? nearOwner->calcFirstInstanceObjectToWorld()
                                                   : Matrix4();
            const XXH64_hash_t nearKey = nearValid ? nearOwner->m_spatialCacheHash : 0;
            const auto  nearRec    = nearValid ? blas.getSpatialMap().debugLedgerLookup(nearKey)
                                               : BlasEntry::InstanceMap::LedgerRec { };
            const char* nearKeyOp  = !nearValid                             ? "n/a"
              : (nearRec.op == LedgerOp::Inserted) ? "ins"
              : (nearRec.op == LedgerOp::Refiled)  ? "refile"
              : (nearRec.op == LedgerOp::Erased)   ? "erase"
                                                   : "none";
            Logger::info(str::format(
              "[FanoutPrevMiss] f=", currentFrameIdx,
              " queryKey=0x", std::hex, static_cast<uint64_t>(queryKey), std::dec,
              " ledgOp=", ledgOpName,
              " ledgFrame=", rec.frame,
              " ledgWrites=", rec.writes,
              " ledgBumped=", static_cast<uint32_t>(rec.bumped),
              " ledgOther=0x", std::hex, static_cast<uint64_t>(rec.otherKey), std::dec,
              " ledgSame=", (rec.data != nullptr && rec.data == nearOwner ? 1 : 0),
              // NV-DXVK [MapLedger] 2026-08-25: WHICH KEY THE NEIGHBOUR IS FILED
              // UNDER, which is the question `none` leaves open.
              //
              // [MapSupply] has now shown supply is adequate — in steady state
              // no non-empty map ever runs a deficit — while this census reads
              // none=95.5%. Those two are only jointly consistent if the map
              // holds the instances under keys the reader never forms. These
              // fields test that directly instead of inferring it.
              //
              //   nearKey != queryKey with nearKeyOp=ins
              //       CONFIRMED. The instance is in this map, filed and live,
              //       under a different key. The exact stage cannot reach it at
              //       any distance, so every such lookup falls to the greedy
              //       nearest search permanently, not occasionally.
              //   nearPropId != 0
              //       AND THE MECHANISM IS COMPLETE. This site is reached only
              //       when the LOOKUP passed stablePropId == 0 (the enclosing
              //       condition), so the reader keyed on the matrix hash. If the
              //       WRITE filed under a propId override, the two keyspaces can
              //       never intersect and no matrix will ever match. That is a
              //       key-composition fault at the write site, and the fix is to
              //       make the two agree rather than to touch the matcher.
              //   nearMtxSame=1 with different keys
              //       the matrices agree byte for byte and the keys still differ,
              //       which leaves the override as the only way to produce it.
              //   nearKey == queryKey
              //       the neighbour IS filed under the key we asked for, so the
              //       cache lookup itself failed — but ins=0 across every census
              //       line already argues against that, so treat a hit here as a
              //       reason to re-examine the census, not as a new lead.
              " nearKey=0x", std::hex, static_cast<uint64_t>(nearKey), std::dec,
              " nearPropId=0x", std::hex,
                static_cast<uint64_t>(nearValid ? nearOwner->m_stablePropId : 0), std::dec,
              " nearKeyOp=", nearKeyOp,
              // The matrix the WRITE side actually files this instance under,
              // against the one the READ side just queried with. These are two
              // different functions — RtInstance::calcFirstInstanceObjectToWorld
              // versus the draw call's — and the SpatialKeyHint comment in the
              // header already lists three ways they compose differently. It
              // wrote that as a safety argument for reusing the hash; the
              // consequence nobody drew is that a divergence files the entry
              // under a key the lookup can never form.
              //
              // nearInstToObj names the branch: non-zero means the write went
              // through `objectToWorld * instancesToObject[0]` while a split
              // placement's read used that placement's own transform, so the two
              // agree only for element 0 of the batch.
              " nearMtxSame=", (nearValid
                && std::memcmp(&nearWriteMtx, &firstInstanceObjectToWorld,
                               sizeof(Matrix4)) == 0 ? 1 : 0),
              " nearInstToObj=", (nearValid && nearOwner->surface.instancesToObject
                                    ? static_cast<uint32_t>(nearOwner->surface.instancesToObject->size())
                                    : 0u),
              " ledgEvicted=", blas.getSpatialMap().debugLedgerEvicted(),
              " ledgEntries=", blas.getSpatialMap().debugLedgerEntries(),
              // Whole-matrix identity, beside the translation-only moved= that
              // §5.4 read as byte identity. If these two disagree, moved= was
              // never evidence of byte identity and §5.4's premise is false.
              " sameBytes=", (std::memcmp(&firstInstanceObjectToWorld, prevObjectToWorld,
                                          sizeof(Matrix4)) == 0 ? 1 : 0),
              " vs=0x", std::hex, uint64_t(vsHashProbe), std::dec,
              " moved=", std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z),
              " nearestToPrev=", (nearSqr == FLT_MAX ? -1.f : std::sqrt(nearSqr)),
              " ownerFrame=", (nearOwner != nullptr ? nearOwner->m_frameLastUpdated : 0u),
              " cur=(", curT.x, ",", curT.y, ",", curT.z, ")",
              " prev=(", prvT.x, ",", prvT.y, ",", prvT.z, ")",
              " mapSz=", blas.getSpatialMap().size()));
          }
        }
      }

      // NV-DXVK [perf] 2026-08-07: the centroid is computed HERE, not at function
      // entry. Both exact stages have missed by this point, so every remaining path
      // -- the exact-miss probe, the nearest search, and the ray-portal code after
      // this block -- genuinely needs it. Instances that resolved exactly returned
      // above without paying for the geometry pointer chase or the transform.
      // Unconditional within this block by design; see the declaration's invariant.
      worldPosition =
        blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);

      // NV-DXVK [MapSupply]: both exact stages have missed, so this lookup is
      // now competing for an instance by distance. Counted against the SAME map
      // its query was counted against, which is the pairing the whole probe
      // exists to make — nearest= above sz= on one map is the pigeonhole, and
      // nearest= well below sz= on that map is a matcher problem instead.
      if (mapSupplySlot != MapSupplyAgg::kSlots) {
        sMapSupply.slots[mapSupplySlot].n.fetch_add(1, std::memory_order_relaxed);
      }

      // NV-DXVK [FindSim2904 exact-miss]: log propId we looked up. Pair
      // against [PropIdTrace] at the same frame to see if lookup propId
      // matches insert propId for the same physical prop. If they match
      // and the map is non-empty, the insert went into a bumped slot
      // (collision); if they differ, the rounding boundary is flipping.
      if (isProbeVS) {
        thread_local uint32_t sExactMissProbe = 0;
        if (sExactMissProbe < 32 || (sExactMissProbe & 0x3FF) == 0) {
          // NV-DXVK: include BlasEntry pointer + linked-instance count so we
          // can detect whether the lookup is going to a different BlasEntry
          // than the one holding the instance. If blasPtr changes between
          // calls for the same physical mountain, drawCallCache is routing
          // to a different BLAS — root cause of "instance alive, lookup
          // misses".
          Logger::info(str::format(
            "[FindSim] #", sExactMissProbe,
            " vs=0x", std::hex, vsHashProbe, std::dec,
            " stage=exact hit=0 propId=0x", std::hex, stablePropId, std::dec,
            " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " linkedInst=", blas.getLinkedInstances().size(),
            // WHICH GEOMETRY THIS ENTRY HOLDS, to settle what the pairing means.
            //
            // Every propId sampled for this shader appears exactly twice, once
            // against each of exactly two blasPtr values, and the two carry
            // different worldPos -- 765 units apart in one pair, 778,000 in
            // another. That is equally consistent with one prop drawn into two
            // views and with two distinct props colliding on one stablePropId,
            // and the two want opposite fixes. Same geometry hash on both means
            // one mesh in two entries; different hashes mean the ID is being
            // reused across unrelated meshes and the producer is at fault.
            " geoHash=0x", std::hex,
              static_cast<uint64_t>(blas.input.getHash(RtxOptions::geometryAssetHashRule())), std::dec,
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
        }
        sExactMissProbe += 1;
      }

      // No exact match, so find the closest match in the region
      // (need to check a 2x2x2 patch of cells to account for positions close to a border)
      // Probe: count how many instances passed/failed each filter clause.
      uint32_t probeNumCellInstances = 0;
      uint32_t probeRejFrameUpdated  = 0;
      uint32_t probeBlockExact = 0;
      uint32_t probeBlockHist  = 0;
      uint32_t probeBlockNear  = 0;
      uint32_t probeBlockNone  = 0;
      uint32_t probeRejMaterialHash  = 0;
      uint32_t probeRejSubPrim       = 0;
      uint32_t probePassedFilter     = 0;
      result = const_cast<RtInstance*>(blas.getSpatialMap().getNearestData(worldPosition, uniqueObjectDistanceSqr, nearestDistSqr,
        [&] (const RtInstance* instance) {
          // Filter out instances by returning false if the instance:
          // - has already been updated this frame
          // - doesn't use the same material
          // - is a sub prim of a replacement instance
          if (isProbeVS || isMtnProbe) {
            probeNumCellInstances += 1;
            const bool okFrame = (instance->m_frameLastUpdated != currentFrameIdx);
            const bool okMat   = (instance->m_materialHash == material.getHash());
            const bool okSub   = !instance->m_primInstanceOwner.isSubPrim();
            if (!okFrame) {
              probeRejFrameUpdated += 1;
              // WHO TOOK IT, AND BY WHAT RIGHT -- see RtInstance::m_claimStage.
              //
              // The stage counts only if it was written THIS frame. An instance
              // the caller already knew is updated without any lookup running,
              // so m_frameLastUpdated being current says nothing about whether
              // the matcher decided it; a stage from an older frame is exactly
              // that case and belongs in none=.
              const RtInstance::ClaimStage blockStage =
                (instance->m_claimFrame == currentFrameIdx)
                  ? instance->m_claimStage
                  : RtInstance::ClaimStage::None;
              switch (blockStage) {
                case RtInstance::ClaimStage::Exact:   probeBlockExact += 1; break;
                case RtInstance::ClaimStage::History: probeBlockHist  += 1; break;
                case RtInstance::ClaimStage::Nearest: probeBlockNear  += 1; break;
                default:                              probeBlockNone  += 1; break;
              }
            }
            if (!okMat)   probeRejMaterialHash += 1;
            if (!okSub)   probeRejSubPrim      += 1;
            if (okFrame && okMat && okSub) probePassedFilter += 1;
            return okFrame && okMat && okSub;
          }
          return instance->m_frameLastUpdated != currentFrameIdx && instance->m_materialHash == material.getHash() && !instance->m_primInstanceOwner.isSubPrim();
        }
      ));
      // See RtInstance::m_claimStage. A nearest claim is the only one decided by
      // submission order, so it is the only one whose loser has a grievance.
      if (result != nullptr) {
        result->m_claimStage = RtInstance::ClaimStage::Nearest;
        result->m_claimFrame = currentFrameIdx;
      }
      if (isProbeVS) {
        // Throttle raised from 8 to 256: the mapSz>0 exact-misses — the only
        // interesting case — all landed past #8 and were never logged.
        thread_local uint32_t sNearestProbe = 0;
        // UNCAPPED. The old "first 256 then every 1024th" throttle aliased the
        // exact events being hunted — a miss is rare, so sampling makes it
        // likely to be the one that is skipped.
        {
          // Distinguish "the cached entry is genuinely elsewhere" from "the
          // cell grid has lost track of an entry that is right here".
          Vector3 dbgNearestPos { 0.f, 0.f, 0.f };
          const RtInstance* dbgOwner = nullptr;
          const float dbgCacheDistSqr =
            blas.getSpatialMap().debugClosestCachedDistSqr(worldPosition, dbgNearestPos, &dbgOwner);
          const size_t dbgCellEntries = blas.getSpatialMap().debugCellEntryCount();

          // NV-DXVK: same three clauses getNearestData's filter applies, so a
          // disagreement between passDistSqr and nearestDistSqr is attributable
          // to the cell patch alone rather than to a filter difference. Kept
          // literally identical to the lambda above -- if that one changes and
          // this does not, the comparison silently stops meaning anything.
          Vector3 dbgPassPos { 0.f, 0.f, 0.f };
          const float dbgPassDistSqr =
            blas.getSpatialMap().debugClosestPassingDistSqr(worldPosition,
              [&] (const RtInstance* instance) {
                return instance->m_frameLastUpdated != currentFrameIdx
                    && instance->m_materialHash == material.getHash()
                    && !instance->m_primInstanceOwner.isSubPrim();
              }, dbgPassPos);
          // Identify who owns the far-away entry. If its VS differs from ours,
          // two different draws are sharing one BlasEntry and evicting each
          // other; if it matches, the same draw is being placed in two spaces.
          uint64_t dbgOwnerVs = 0ull;
          uint32_t dbgOwnerFrame = 0u;
          uint32_t dbgOwnerCam = 0u;
          if (dbgOwner != nullptr) {
            dbgOwnerFrame = dbgOwner->m_frameLastUpdated;
            const BlasEntry* ownerBlas = dbgOwner->getBlas();
            if (ownerBlas != nullptr) {
              dbgOwnerVs = uint64_t(ownerBlas->input.getTransformData().vertexShaderHash);
              dbgOwnerCam = uint32_t(ownerBlas->input.cameraType);
            }
          }
          // NV-DXVK [MapDump2]: on an actual MISS, dump EVERY live entry rather
          // than just the nearest one. [MapWrite] instruments BOTH SpatialMap
          // write sites (insert via teleport, move via onTransformChanged) and
          // logged ZERO sky-space centroids for this VS across 2231 writes --
          // yet debugClosestCachedDistSqr reports a sky-space nearest entry with
          // ownerVs equal to this same VS. Those cannot both be true, and a
          // single-nearest readback cannot show which observation is incomplete.
          // ownerBlas is printed alongside the queried blas: if they differ, the
          // owning instance has been relinked to another BlasEntry while its
          // cache entry stayed behind here, which no write-site probe would see.
          // Misses are rare, so this is uncapped and prints the whole map.
          if (result == nullptr) {
            uint32_t dumpIdx = 0;
            blas.getSpatialMap().debugForEachEntry(
              [&](auto key, const Vector3& c, const RtInstance* owner) {
                uint64_t  oVs = 0ull;
                uint32_t  oFrame = 0u;
                uintptr_t oBlas = 0u;
                if (owner != nullptr) {
                  oFrame = owner->m_frameLastUpdated;
                  const BlasEntry* ob = owner->getBlas();
                  if (ob != nullptr) {
                    oVs   = uint64_t(ob->input.getTransformData().vertexShaderHash);
                    oBlas = reinterpret_cast<uintptr_t>(ob);
                  }
                }
                Logger::info(str::format(
                  "[MapDump2] f=", currentFrameIdx,
                  " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
                  " #", dumpIdx++,
                  " key=0x", std::hex, static_cast<uint64_t>(key), std::dec,
                  " centroid=(", c.x, ",", c.y, ",", c.z, ")",
                  " ownerVs=0x", std::hex, oVs, std::dec,
                  " ownerBlas=0x", std::hex, oBlas, std::dec,
                  " ownerFrame=", oFrame,
                  " query=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
              });
          }

          // worldPosition is boundingBox.getTransformedCentroid(o2w), NOT
          // o2w.T. With the sub-view reproject baking scale=1000 into o2w,
          // the object-space bbox centre is amplified 1000x on its way to the
          // centroid — so a ~26-unit local offset lands ~26000 units away.
          // Log the decomposition so the arithmetic is visible rather than
          // inferred: local bbox, the o2w column lengths (the actual applied
          // scale) and o2w.T next to the resulting centroid.
          const AxisAlignedBoundingBox& dbgBox = blas.input.getGeometryData().boundingBox;
          const Vector3 dbgBoxCentre = (dbgBox.minPos + dbgBox.maxPos) * 0.5f;
          const Matrix4& dbgO2w = firstInstanceObjectToWorld;
          const float dbgScaleX = length(Vector3(dbgO2w[0][0], dbgO2w[0][1], dbgO2w[0][2]));
          const float dbgScaleY = length(Vector3(dbgO2w[1][0], dbgO2w[1][1], dbgO2w[1][2]));
          const float dbgScaleZ = length(Vector3(dbgO2w[2][0], dbgO2w[2][1], dbgO2w[2][2]));
          Logger::info(str::format(
            "[FindSim] #", sNearestProbe, " vs=0x", std::hex, vsHashProbe, std::dec,
            " boxLocalCentre=(", dbgBoxCentre.x, ",", dbgBoxCentre.y, ",", dbgBoxCentre.z, ")",
            " boxMin=(", dbgBox.minPos.x, ",", dbgBox.minPos.y, ",", dbgBox.minPos.z, ")",
            " boxMax=(", dbgBox.maxPos.x, ",", dbgBox.maxPos.y, ",", dbgBox.maxPos.z, ")",
            " o2wScale=(", dbgScaleX, ",", dbgScaleY, ",", dbgScaleZ, ")",
            " o2wT=(", float(dbgO2w[3][0]), ",", float(dbgO2w[3][1]), ",", float(dbgO2w[3][2]), ")",
            " cacheNearestDistSqr=", dbgCacheDistSqr,
            " cacheNearestPos=(", dbgNearestPos.x, ",", dbgNearestPos.y, ",", dbgNearestPos.z, ")",
            " ownerVs=0x", std::hex, dbgOwnerVs, std::dec,
            " ownerCam=", dbgOwnerCam,
            " ownerFrame=", dbgOwnerFrame,
            " curFrame=", currentFrameIdx,
            " curCam=", static_cast<uint32_t>(cameraType),
            " cellEntries=", dbgCellEntries,
            " stage=nearest exactHit=0 hit=", (result != nullptr ? 1 : 0),
            " propId=0x", std::hex, stablePropId, std::dec,
            " nearestDistSqr=", nearestDistSqr,
            " maxDistSqr=", uniqueObjectDistanceSqr,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " cellInsts=", probeNumCellInstances,
            // DID THIS PLACEMENT HAVE ENGINE HISTORY TO RETRY WITH, and it is
            // the last link in the chain rather than another column.
            //
            // The nearest stage is a greedy claim -- rejFrame counts instances
            // refused for having been updated already this frame -- so a
            // placement that reaches it can lose its own partner to whichever
            // placement resolved first, and mint a fresh instance beside it.
            // That is the [RsFailMember] event: 167 of 207 failing frames carry
            // a nearest-stage miss against ~70 expected by chance.
            //
            // The history retry above exists to resolve movers by exact hash
            // BEFORE this search is reached, and it works where it is plumbed --
            // [FanoutPrev] reads prevHit=113 prevMiss=0 in steady state. But
            // rtx_types.h states its own limit: prevInstancesToObject is "only
            // ever populated for isFanoutBatch draws", and prevNull counts
            // 13-25 placements a frame arriving here with nothing to retry.
            //
            // hadPrev=0 on the misses says those two facts are the same fact,
            // and the fix is to plumb the history rather than to rewrite the
            // matcher. hadPrev=1 says the history was present, was tried, and
            // the placement still lost the race -- and then the greedy claim
            // itself is what needs replacing.
            " hadPrev=", (prevObjectToWorld != nullptr ? 1 : 0),
            " rejFrame=", probeRejFrameUpdated,
            // THE FIX-SHAPE FIELD. near= high means the blockers were themselves
            // resolved by distance, so submission order decided an ownership
            // question neither placement had a claim to, and settling every
            // exact and history claim across the batch before any distance claim
            // removes it. exact=/hist= high means the winners owned those
            // instances outright and the loser correctly has none -- then this
            // is not a matcher defect and the residency failure wants a
            // different explanation entirely.
            " blockedBy{exact=", probeBlockExact,
            " hist=", probeBlockHist,
            " near=", probeBlockNear,
            " none=", probeBlockNone, "}",
            " rejMat=", probeRejMaterialHash,
            " rejSub=", probeRejSubPrim,
            " passed=", probePassedFilter,
            // NV-DXVK: THE SPLIT THIS PROBE COULD NOT MAKE. cacheNearest* above
            // ignores the filter, so a miss could mean either "the instance is
            // gone from the map" or "it is there and was already claimed this
            // frame", and those want opposite fixes. passDistSqr walks the cache
            // applying the SAME filter getNearestData uses, with no cell patch
            // and no radius:
            //   passDistSqr == FLT_MAX          nothing in the entire map could
            //                                   have matched. The instance is
            //                                   absent -- look at churn= below.
            //   passDistSqr <= maxDistSqr       a legal match existed and the
            //                                   CELL SCAN failed to reach it.
            //                                   That is a cell/patch defect.
            //   passDistSqr >  maxDistSqr       present but genuinely too far.
            // NV-DXVK: WHICH MAP was queried, and it is the field that decides
            // the pigeonhole. Measured f=7314: 98 queries against a map holding
            // at most 63 instances, so ~35 draws per frame CANNOT match anything
            // no matter how good the matcher is -- every entry is legitimately
            // claimed by an earlier draw and the surplus mints and is reaped.
            // What that count cannot say is whether the 98 hit ONE BlasEntry
            // (genuine surplus: more draws than objects, i.e. multi-pass, and
            // the fix is to drop the redundant pass the way CharDepthPrepass
            // and WorldNoShadeInputs already do) or SEVERAL BlasEntries (the
            // draws for one prop family are being split across entries, so the
            // instances are filed in a different map than the one queried, and
            // the fix is in entry selection in rtx_draw_call_cache). Group the
            // log by (curFrame, blas) to tell them apart.
            " blas=0x", std::hex,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&blas)), std::dec,
            " passDistSqr=", dbgPassDistSqr,
            " passPos=(", dbgPassPos.x, ",", dbgPassPos.y, ",", dbgPassPos.z, ")",
            // Lifetime churn for THIS map. refiled is the one to read: it counts
            // move() calls where the key actually changed, forcing an erase and
            // re-insert. For a stationary prop that should never happen, so
            // refiled climbing in step with the misses means the key is moving
            // under the instance and the map is losing it every frame.
            " churn{ins=", blas.getSpatialMap().debugInserts(),
            " era=", blas.getSpatialMap().debugErases(),
            " mov=", blas.getSpatialMap().debugMoves(),
            " refiled=", blas.getSpatialMap().debugRefiled(), "}",
            " cellSize=", blas.getSpatialMap().debugCellSize(),
            " cellEntryCount=", blas.getSpatialMap().debugCellEntryCount(),
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")",
            " curMatHash=0x", std::hex, material.getHash(), std::dec));
          // NV-DXVK [MapDump]: on a MISS against a NON-EMPTY map, dump every
          // live entry beside the query. The keepN=4 experiment refuted the
          // lifetime explanation (maps grew to 13 entries, misses continued),
          // so the entry the query wants may well be present — the question is
          // why the lookup cannot reach it. Two reachable answers:
          //   entryKey == queryKey but the exact stage still missed  -> the
          //     exact lookup is keyed on something else than we think
          //   entryCell != queryCell while |entryCentroid - queryCentroid| is
          //     small -> the cell grid and the distance test disagree, i.e. a
          //     border/rounding fault, not a coordinate-space fault
          //   entryCentroid genuinely far -> confirms the squatter reading
          // queryKey is formed exactly as getDataAtTransform does: propId when
          // non-zero, else XXH64 over the matrix bytes.
          if (result == nullptr && blas.getSpatialMap().size() > 0u) {
            // UNCAPPED. A miss against a non-empty map is the event this whole
            // probe exists to catch; clipping it to 4/frame risks dropping the
            // one that matters.
            {
              const uint64_t queryKey = (stablePropId != 0ull)
                ? stablePropId
                : static_cast<uint64_t>(XXH64(&firstInstanceObjectToWorld,
                                              sizeof(firstInstanceObjectToWorld), 0));
              const auto qCell = blas.getSpatialMap().debugCellPosOf(worldPosition);
              Logger::info(str::format(
                "[MapDump] f=", currentFrameIdx,
                " vs=0x", std::hex, vsHashProbe, std::dec,
                " blasPtr=0x", std::hex, reinterpret_cast<uintptr_t>(&blas), std::dec,
                " QUERY key=0x", std::hex, queryKey, std::dec,
                " propId=0x", std::hex, stablePropId, std::dec,
                " centroid=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")",
                " cell=(", qCell.x, ",", qCell.y, ",", qCell.z, ")",
                " cellSize=", blas.getSpatialMap().debugCellSize(),
                " maxDistSqr=", uniqueObjectDistanceSqr,
                " entries=", blas.getSpatialMap().size()));
              uint32_t dumped = 0;
              blas.getSpatialMap().debugForEachEntry(
                [&](XXH64_hash_t entryKey, const Vector3& entryCentroid,
                    const RtInstance* entryData) {
                  dumped += 1;   // uncapped: dump every entry in the map
                  const auto eCell = blas.getSpatialMap().debugCellPosOf(entryCentroid);
                  const Vector3 delta = entryCentroid - worldPosition;
                  const float dSq = lengthSqr(delta);
                  Logger::info(str::format(
                    "[MapDump]   entry#", dumped,
                    " key=0x", std::hex, static_cast<uint64_t>(entryKey), std::dec,
                    " keyMatchesQuery=", (static_cast<uint64_t>(entryKey) == queryKey ? 1 : 0),
                    " centroid=(", entryCentroid.x, ",", entryCentroid.y, ",", entryCentroid.z, ")",
                    " cell=(", eCell.x, ",", eCell.y, ",", eCell.z, ")",
                    " sameCell=", ((eCell.x == qCell.x && eCell.y == qCell.y && eCell.z == qCell.z) ? 1 : 0),
                    " distSq=", dSq,
                    " withinMaxDist=", (dSq < uniqueObjectDistanceSqr ? 1 : 0),
                    // The three filter clauses getNearestData applies, so a
                    // reachable-but-rejected entry is distinguishable from an
                    // unreachable one.
                    " ownerPropId=0x", std::hex,
                      (entryData != nullptr ? entryData->m_stablePropId : 0ull), std::dec,
                    " ownerLastUpd=", (entryData != nullptr ? entryData->m_frameLastUpdated : 0u),
                    " okFrame=", ((entryData != nullptr
                                   && entryData->m_frameLastUpdated != currentFrameIdx) ? 1 : 0),
                    " okMat=", ((entryData != nullptr
                                 && entryData->m_materialHash == material.getHash()) ? 1 : 0),
                    " okSub=", ((entryData != nullptr
                                 && !entryData->m_primInstanceOwner.isSubPrim()) ? 1 : 0)));
                });
            }
          }
        }
        sNearestProbe += 1;
      }
      if (isMtnProbe) {
        thread_local uint32_t sMtnNear = 0;
        if (sMtnNear < 80u || (sMtnNear & 0xFFu) == 0u) {
          Logger::info(str::format(
            "[FindSimMtn] #", sMtnNear, " f=", currentFrameIdx,
            " outcome=", (result != nullptr ? "COLLAPSE" : "KEEP-NEW"),
            " stage=nearest exactMissed=1",
            " propId=0x", std::hex, stablePropId, std::dec,
            " nearestDistSqr=", nearestDistSqr,
            " maxDistSqr=", uniqueObjectDistanceSqr,
            " cellInsts=", probeNumCellInstances,
            " passedFilter=", probePassedFilter,
            " spatialMapSize=", blas.getSpatialMap().size(),
            " worldPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
        }
        sMtnNear += 1;
      }
      if (nearestDistSqr == 0.0f && result != nullptr) {
        // Not going to find anything closer
        return result;
      }
    }

    // For portal gun and other objects that were drawn in the ViewModel, need to check the
    // virtual version of the instance from previous frame.
    if (nearestDistSqr > 0.0f &&
        cameraType == CameraType::ViewModel && 
        RtxOptions::useRayPortalVirtualInstanceMatching() ) {
      const Matrix4* teleportMatrix = nullptr;
      for (const RtInstance* instance : blas.getLinkedInstances()) {
        if (instance->m_frameLastUpdated != currentFrameIdx - 1 || 
            instance->m_materialHash != material.getHash()) {
          continue;
        }
        
        // Compare against virtual position of a predicted instance's position in the current frame
        const Vector3& prevPrevInstanceWorldPosition = instance->getPrevWorldPosition();
        const Vector3& prevInstanceWorldPosition = instance->getWorldPosition();
        const Vector3 predictedInstanceWorldPosition = prevInstanceWorldPosition +
          (prevInstanceWorldPosition - prevPrevInstanceWorldPosition);
      
        // Check all portal pairs
        for (auto& rayPortalPair : rayPortalManager.getRayPortalPairInfos()) {
          if (rayPortalPair.has_value()) {
            for (uint32_t i = 0; i < 2; i++) {
              const auto& rayPortal = rayPortalPair->pairInfos[i];

              const Vector3 virtualPredictedInstanceWorldPosition =
                rayPortalManager.getVirtualPosition(predictedInstanceWorldPosition, rayPortal.portalToOpposingPortalDirection);

              // Distance of the object from the predicted virtual position of an instance
              const float virtualDistSqr = lengthSqr(virtualPredictedInstanceWorldPosition - worldPosition);

              // Is the instance is similar, and within range?  We already know the BLAS is shared, due to the for loop
              if (virtualDistSqr <= uniqueObjectDistanceSqr && virtualDistSqr < nearestDistSqr) {
                nearestDistSqr = virtualDistSqr;
                result = const_cast<RtInstance*>(instance);
                teleportMatrix = &rayPortal.portalToOpposingPortalDirection;
                if (virtualDistSqr == 0.0f) {
                  // Not going to find anything closer.
                  break;
                }
              }
            }
          }
        }
      }
      
      // If the match was against a virtual equivalent of the instance from previous frame,
      // update the instance's transform to that of the virtual one
      if (teleportMatrix) {
        // NV-DXVK [Phase2b]: teleportWithHistory rewrites transforms, moves the
        // spatial-map entry AND recurses over replacement prims that may live in
        // other shards' BLASes. Cold path (ViewModel through a portal) — defer
        // the whole draw to the ordered tail, which re-runs this find and
        // teleports inline (allowMiss).
        if (inShardedInstancePhase() && !t_shardPhase.allowMiss) {
          t_shardPhase.deferredThisDraw = true;
          return nullptr;
        }
        result->teleportWithHistory(*teleportMatrix);
      }
    }

    // NV-DXVK [MatBind identity] cross-entry instance relink.
    //
    // For re-batched geometry (world-space vertices rewritten per frame) the
    // DrawCallCache correctly allocates a NEW BlasEntry whenever the index
    // data changes (topology is immutable on a living entry — reusing one
    // across a topology change caused VK_ERROR_DEVICE_LOST, 2026-08-02). But
    // the OBJECT still exists: last frame's instance sits in a sibling
    // entry's SpatialMap under the SAME transform key (these draws carry the
    // identity transform, and the exact key is hashed from transform bytes,
    // not position). Without this step that instance goes unpaired, dies at
    // numFramesToKeepInstances=1, and a fresh id is created here — measured
    // at ~10 of 17 draws/frame on vs 0x2859d250: the geometry flicker churn.
    //
    // So: on a full miss, search the engine-class siblings (same matsys
    // IMaterial* + shader, via DrawCallCache::m_engineClassIndex) for an
    // un-updated instance at this exact transform key, and MIGRATE it to
    // this entry. updateInstance then rebinds every surface buffer index
    // from the new entry (processInstanceBuffers runs on every first update
    // of a frame), so the instance renders the new entry's geometry while
    // keeping its id, TLAS slot continuity, and temporal history.
    // NV-DXVK [Phase2b]: a full miss on a worker is DEFERRED, not resolved. Both
    // resolutions of a miss (the engine-class migration below — which erases from
    // another shard's spatial map and relinks instance lists — and addInstance in
    // the caller — the global-vector escape) are the plan's Sec-6 escapes. The
    // ordered tail re-runs the whole draw sequentially: this find repeats against
    // the by-then-current maps, and the migration/add path runs inline with full
    // fidelity. addedPct=0 in steady state, so this trigger is cold by
    // construction, and instance IDs stay in arena order (deterministic).
    if (inShardedInstancePhase() && !t_shardPhase.allowMiss && result == nullptr) {
      t_shardPhase.deferredThisDraw = true;
      return nullptr;
    }

    if (result == nullptr && drawCallCache != nullptr && blas.engineClassKey != 0) {
      RtInstance* migrated = nullptr;
      BlasEntry* migratedFrom = nullptr;
      drawCallCache->forEachEngineClassSibling(blas.engineClassKey, [&](BlasEntry& sibling) {
        if (&sibling == &blas || migrated != nullptr) {
          return;
        }
        const RtInstance* cand =
          sibling.getSpatialMap().getDataAtTransform(firstInstanceObjectToWorld, stablePropId);
        if (cand == nullptr) {
          return;
        }
        RtInstance* c = const_cast<RtInstance*>(cand);
        // Same acceptance filters as the nearest-stage search above.
        if (c->m_frameLastUpdated == currentFrameIdx
            || c->m_materialHash != material.getHash()
            || c->m_primInstanceOwner.isSubPrim()
            || c->m_isCreatedByRenderer) {
          return;
        }
        migrated = c;
        migratedFrom = &sibling;
      });
      if (migrated != nullptr) {
        // NV-DXVK [FirstBakeHold — flicker fix]: if the DESTINATION entry's
        // first bake is still source-pending, its BLAS content is not
        // trustworthy this frame — stash the FROM-entry's built BLAS so
        // AccelManager can render the previous geometry for the handover
        // frame instead of a collapsed bake (see RtInstance member comment).
        // If the instance already carries a stash (chained re-batches on
        // consecutive frames — the observed paired dedup-miss pattern), only
        // overwrite it when the FROM entry's own bake is NOT pending:
        // otherwise we would replace a known-good BLAS with a garbage one.
        if (RtxOptions::firstBakeHold()
            && blas.modifiedGeometryData.pendingSrcBake
            && migratedFrom->dynamicBlas.ptr() != nullptr
            && migratedFrom->dynamicBlas->accelerationStructureReference != 0) {
          if (migrated->m_prevBlasKeepAlive.ptr() == nullptr
              || !migratedFrom->modifiedGeometryData.pendingSrcBake) {
            migrated->m_prevBlasKeepAlive = migratedFrom->dynamicBlas;
          }
        } else {
          migrated->m_prevBlasKeepAlive = nullptr;
        }
        // Order matters: the spatial-cache erase must run while the instance
        // still points at the OLD entry (it erases from m_linkedBlas's map).
        migrated->removeFromSpatialCache();
        migratedFrom->unlinkInstance(migrated);
        migrated->setBlas(blas);
        blas.linkInstance(migrated);
        const Vector3 newCentroid =
          blas.input.getGeometryData().boundingBox.getTransformedCentroid(firstInstanceObjectToWorld);
        migrated->m_spatialCacheHash =
          blas.getSpatialMap().insert(newCentroid, firstInstanceObjectToWorld, migrated, stablePropId,
                                      currentFrameIdx);
        if (isProbeVS) {
          static thread_local uint32_t sRelinkProbe = 0;
          if (sRelinkProbe < 64 || (sRelinkProbe & 0x3FF) == 0) {
            Logger::info(str::format(
              "[ClassRelink] #", sRelinkProbe, " f=", currentFrameIdx,
              " vs=0x", std::hex, vsHashProbe, std::dec,
              " instId=", migrated->getId(),
              " fromBlas=0x", std::hex, reinterpret_cast<uintptr_t>(migratedFrom),
              " toBlas=0x", reinterpret_cast<uintptr_t>(&blas), std::dec,
              " newPos=(", worldPosition.x, ",", worldPosition.y, ",", worldPosition.z, ")"));
          }
          sRelinkProbe += 1;
        }
        result = migrated;
      }
    }

    return result;
  }

  RtInstance* InstanceManager::addInstance(BlasEntry& blas) {
    const uint32_t currentFrameIdx = m_device->getCurrentFrameId();

    const uint32_t instanceIdx = m_instances.size();
    RtInstance* newInst = new RtInstance(m_nextInstanceId++, instanceIdx);
    m_instances.push_back(newInst);

    RtInstance* currentInstance = m_instances[instanceIdx];

    currentInstance->m_frameCreated = currentFrameIdx;
    
    // Set Instance Vulkan AS Instance information
    {
      currentInstance->m_vkInstance.mask = 0;
      currentInstance->m_vkInstance.flags = 0;
      currentInstance->m_vkInstance.instanceCustomIndex = 0;
      currentInstance->m_vkInstance.instanceShaderBindingTableRecordOffset = 0;
      currentInstance->setBlas(blas);
    }

    // Rest of the setup happens in updateInstance()

    // Notify events after instance has been added
    for (auto& event : m_eventHandlers)
      event.onInstanceAddedCallback(*currentInstance);

    // onInstanceAddedCallback will link current instance to the BLAS
    currentInstance->m_isUnlinkedForGC = false;

    return currentInstance;
  }

  // Creates a copy of an instance
  // If the copy is temporary and is not tracked via callbacks/externally, it doesn't need
  // a valid unique instance ID. In that case, set generateValidID to false to avoid overflowing the ID value
  RtInstance* InstanceManager::createInstanceCopy(const RtInstance& reference, bool generateValidID) {

    const uint32_t instanceIdx = m_instances.size();

    uint64_t id = generateValidID ? m_nextInstanceId++ : kInvalidInstanceId;
    RtInstance* newInstance = new RtInstance(reference, id, instanceIdx);
    newInstance->m_isCreatedByRenderer = true;
    m_instances.push_back(newInstance);

    return newInstance;
  }

  // NV-DXVK [perf] 2026-08-07: THE READ SIDE of processInstanceBuffers, split
  // out so it can run once per DRAW instead of once per INSTANCE.
  //
  // Every value here is a pure read of blas.modifiedGeometryData, so this is a
  // pure function of the BlasEntry. It is called from computeDrawScopedState
  // (default) or per instance when rtx.hoistSurfaceBufferBinding is off. See
  // that option for why per-draw is keyless and safe while a per-BLAS cache
  // keyed on frameLastUpdated is neither.
  InstanceManager::SurfaceBufferBinding InstanceManager::resolveSurfaceBufferBinding(const BlasEntry& blas) {
    SurfaceBufferBinding b;
    b.positionBufferIndex = blas.modifiedGeometryData.positionBufferIndex;
    b.positionOffset = blas.modifiedGeometryData.positionBuffer.offsetFromSlice();
    b.positionStride = blas.modifiedGeometryData.positionBuffer.stride();
    b.previousPositionBufferIndex = blas.modifiedGeometryData.previousPositionBufferIndex;
    b.normalBufferIndex = blas.modifiedGeometryData.normalBufferIndex;
    b.normalOffset = blas.modifiedGeometryData.normalBuffer.offsetFromSlice();
    b.normalStride = blas.modifiedGeometryData.normalBuffer.stride();
    b.normalFormat = blas.modifiedGeometryData.normalBuffer.vertexFormat();
    b.color0BufferIndex = blas.modifiedGeometryData.color0BufferIndex;
    b.color0Offset = blas.modifiedGeometryData.color0Buffer.offsetFromSlice();
    b.color0Stride = blas.modifiedGeometryData.color0Buffer.stride();
    b.texcoordBufferIndex = blas.modifiedGeometryData.texcoordBufferIndex;
    b.texcoordOffset = blas.modifiedGeometryData.texcoordBuffer.offsetFromSlice();
    b.texcoordStride = blas.modifiedGeometryData.texcoordBuffer.stride();
    b.indexBufferIndex = blas.modifiedGeometryData.indexBufferIndex;
    b.indexStride = blas.modifiedGeometryData.indexBuffer.stride();
    // NV-DXVK: derive texcoordEncoding from the BUFFER's actual vertex
    // format here — the ground truth of how the bytes are stored.
    // Previously the encoding was set later from
    // drawCall.getTransformData().texcoordEncoding (VS-ISGN-derived).
    // That broke when the same BLAS was referenced by draws of multiple
    // VS classes (geometry-hash dedup): the BLAS's texcoord buffer is
    // built from one VS's VB, but later draws with a different VS class
    // would overwrite the encoding to a value that doesn't match the
    // buffer. Probe 8 confirmed surfaces ending up with stride=20
    // (R32G32_FLOAT layout) AND encoding=1 (TF2BspUintPacked decode) —
    // an impossible combination producing 8000-range garbage UVs from
    // float-bit reinterpretation. Tying encoding to buffer.vertexFormat()
    // makes the two always consistent because they come from the same
    // source.
    const VkFormat tcFmt = blas.modifiedGeometryData.texcoordBuffer.defined()
        ? blas.modifiedGeometryData.texcoordBuffer.vertexFormat()
        : VK_FORMAT_UNDEFINED;
    b.texcoordEncoding =
        (tcFmt == VK_FORMAT_R32G32_UINT || tcFmt == VK_FORMAT_R32G32_SINT)
        ? RtSurface::TexcoordEncoding::TF2BspUintPacked
        : RtSurface::TexcoordEncoding::Float;
    // NV-DXVK: propagate lightmap-UV presence so surface_interaction can
    // read TEXCOORD1 from the second 8-byte slot the interleaver wrote
    // alongside TEXCOORD0. The flag is the single source of truth for
    // both the per-hit interpolation gate and the material code's
    // lightmap-sampler UV lookup.
    b.hasLightmap = blas.modifiedGeometryData.hasTexcoord1;
    // NV-DXVK: TF2 worldspace VGUI extras — the slang VGUI evaluator reads
    // 8 floats per vertex starting at element index vguiOffset. Both fields
    // were stamped by RtxGeometryUtils::processGeometryBuffers (slow path)
    // when input.vguiLayoutEnable was true; a non-VGUI surface sees
    // hasVgui=false and the dispatch in opaque_surface_material_interaction
    // is skipped.
    b.isVgui = blas.modifiedGeometryData.hasVgui;
    b.vguiOffset = blas.modifiedGeometryData.vguiOffset;
    // NV-DXVK: 3 VGUI structured-buffer bindless indices. Truncating to
    // uint16_t is safe because BindlessResourceManager::kMaxBindlessResources
    // is 64K (matches uint16_t range). kSurfaceInvalidBufferIndex (UINT32_MAX)
    // truncates to 0xFFFF, which the slang side compares against
    // BINDING_INDEX_INVALID(0xFFFF) — see surface_interaction.slangh:853 for
    // the same convention applied to indexBufferIndex.
    b.vguiFontBoundsBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiFontBoundsBufferIndex);
    b.vguiImgBoundsBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiImgBoundsBufferIndex);
    b.vguiStylesBufferIndex =
        uint16_t(blas.modifiedGeometryData.vguiStylesBufferIndex);
    return b;
  }

  // NV-DXVK [perf] 2026-08-07: THE WRITE SIDE. Unchanged in count and
  // destination -- every instance still gets every field written every frame.
  // Only the DERIVATION of the values moved (see resolveSurfaceBufferBinding).
  void InstanceManager::applySurfaceBufferBinding(const SurfaceBufferBinding& binding,
                                                  RtInstance& currentInstance) const {
    currentInstance.surface.positionBufferIndex = binding.positionBufferIndex;
    currentInstance.surface.positionOffset = binding.positionOffset;
    currentInstance.surface.positionStride = binding.positionStride;
    currentInstance.surface.previousPositionBufferIndex = binding.previousPositionBufferIndex;
    currentInstance.surface.normalBufferIndex = binding.normalBufferIndex;
    currentInstance.surface.normalOffset = binding.normalOffset;
    currentInstance.surface.normalStride = binding.normalStride;
    currentInstance.surface.normalFormat = binding.normalFormat;
    currentInstance.surface.color0BufferIndex = binding.color0BufferIndex;
    currentInstance.surface.color0Offset = binding.color0Offset;
    currentInstance.surface.color0Stride = binding.color0Stride;
    currentInstance.surface.texcoordBufferIndex = binding.texcoordBufferIndex;
    currentInstance.surface.texcoordOffset = binding.texcoordOffset;
    currentInstance.surface.texcoordStride = binding.texcoordStride;
    currentInstance.surface.texcoordEncoding = binding.texcoordEncoding;
    currentInstance.surface.indexBufferIndex = binding.indexBufferIndex;
    currentInstance.surface.indexStride = binding.indexStride;
    currentInstance.surface.hasLightmap = binding.hasLightmap;
    currentInstance.surface.isVgui = binding.isVgui;
    currentInstance.surface.vguiOffset = binding.vguiOffset;
    currentInstance.surface.vguiFontBoundsBufferIndex = binding.vguiFontBoundsBufferIndex;
    currentInstance.surface.vguiImgBoundsBufferIndex = binding.vguiImgBoundsBufferIndex;
    currentInstance.surface.vguiStylesBufferIndex = binding.vguiStylesBufferIndex;
  }

  // NV-DXVK [Phase2b]: ordered-tail application of a worker-recorded op. Reads
  // and writes m_spatialCacheHash HERE, at apply time, so multi-op chains on one
  // instance (move-then-move across cameras, migration erase-then-insert) resolve
  // exactly like the inline code did. Single-threaded by contract (the tail).
  void InstanceManager::applyDeferredSpatialOp(const DeferredSpatialOp& op) {
    RtInstance* inst = op.instance;
    switch (op.kind) {
      case DeferredSpatialOp::Kind::kMove:
        inst->m_spatialCacheHash = op.targetBlas->getSpatialMap().move(
            inst->m_spatialCacheHash, op.centroid, op.transform, inst, op.stablePropId,
            op.precomputedMatrixHash, inst->m_frameLastUpdated);
        mapWriteAccount(/*isInsert*/ false, static_cast<uint32_t>(op.targetBlas->getSpatialMap().size()));
        break;
      case DeferredSpatialOp::Kind::kInsert:
        inst->m_spatialCacheHash = op.targetBlas->getSpatialMap().insert(
            op.centroid, op.transform, inst, op.stablePropId, inst->m_frameLastUpdated);
        mapWriteAccount(/*isInsert*/ true, static_cast<uint32_t>(op.targetBlas->getSpatialMap().size()));
        break;
      case DeferredSpatialOp::Kind::kDecalOrder:
        assignDecalSortOrder(*inst);
        break;
    }
    inst->m_spatialOpPendingFrame = kInvalidFrameIndex;
  }

  // NV-DXVK [Phase2b]: CS-record-step surface buffer rebind — same body as the
  // non-hoisted processInstanceBuffers path, reading the post-bake,
  // post-updateBufferCache BlasEntry directly. See the declaration comment.
  void InstanceManager::bindInstanceBuffersFromBlas(const BlasEntry& blas, RtInstance& instance) const {
    applySurfaceBufferBinding(resolveSurfaceBufferBinding(blas), instance);
  }

  // NV-DXVK [Phase2b]: CS-record-step billboard stage. The eligibility gate ran
  // (and passed) on the worker — this replays only the creation, whose buffer
  // reads are positional and whose m_billboards appends are CS-domain.
  bool InstanceManager::runBillboardStage(RtInstance& instance, const Vector3& cameraDir) {
    if (instance.testCategoryFlags(InstanceCategories::Beam)) {
      createBeams(instance);
    } else if (!instance.surface.alphaState.isDecal) {
      createBillboards(instance, cameraDir);
    }
    return instance.m_billboardCount != 0;
  }

  // NV-DXVK [Phase2b]: CS-record-step replay of the deferred OMM half of the
  // updateInstance event fanout. Selects handlers by the
  // skippableWhenNoPendingOmmWork contract — the same discriminator the worker
  // used to defer them — so a future third handler routes itself by what it
  // declared, not by name.
  void InstanceManager::fireDeferredOmmCallbacks(RtInstance& instance, const DrawCallState& drawCall,
                                                 const MaterialData& materialData, bool hasTransformChanged,
                                                 bool hasPreviousPositions, bool isFirstUpdateThisFrame) {
    for (auto& event : m_eventHandlers) {
      if (!event.skippableWhenNoPendingOmmWork) {
        continue;
      }
      event.onInstanceUpdatedCallback(instance, drawCall, materialData, hasTransformChanged,
                                      hasPreviousPositions, isFirstUpdateThisFrame);
    }
  }

  void InstanceManager::processInstanceBuffers(const BlasEntry& blas, const DrawScopedState& drawState,
                                               RtInstance& currentInstance) const {
    // REBIND, NEVER SKIPPED. This is not a read the instance state key covers:
    // buffer identity is not an input to computeDrawStateKey, and
    // Map(WRITE_DISCARD) renames the slice on ~94% of draws, so geometry that
    // hashes identically lives in a different allocation every frame. See the
    // call site in updateInstance.
    if (RtxOptions::hoistSurfaceBufferBinding()) {
      applySurfaceBufferBinding(drawState.buffers, currentInstance);
    } else {
      applySurfaceBufferBinding(resolveSurfaceBufferBinding(blas), currentInstance);
    }

    // NV-DXVK [perf] 2026-08-07: THE THREE BLOCKS BELOW ARE GATED, AND TWO OF
    // THEM WERE NOT.
    //
    // They are first-sighting censuses: each keeps a static set of tuples it
    // has already logged and emits one line per new one. They saturate within
    // the first second of a run and then spend the rest of it re-proving that
    // they have nothing to say -- but the SET OPERATION runs regardless, once
    // per instance, ~15,500 times a frame on dxvk-cs. [TC1Surface] additionally
    // took a std::mutex on every one of those.
    //
    // That is the same shape as the accel manager's world-extent census
    // (unconditional per-instance std::mutex, 8,375x/frame), which cost ~4 ms.
    // It is NOT the shape of the relaxed-atomic counters gated the same day,
    // which returned zero -- see rtx.logSurfaceGeomDiag for why the two are
    // different by two orders of magnitude and must not be reasoned about
    // together.
    //
    // The gate wraps KEY CONSTRUCTION, MUTEX AND CONTAINER, not just the emit.
    // Gating only the Logger::info leaves the entire cost in place; that was
    // the defect in the world-extent census's own gate.
    if (!RtxOptions::logSurfaceGeomDiag()) {
      return;
    }

    {
      static std::unordered_set<uint64_t> sLoggedVguiSurfaceKeys;
      static std::mutex sLoggedVguiSurfaceMu;
      if (currentInstance.surface.isVgui) {
        const uint64_t key =
            (uint64_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFFu)
          | ((uint64_t(currentInstance.surface.vguiOffset) & 0xFFFFu) << 16);
        bool firstSeen = false;
        {
          std::lock_guard<std::mutex> lk(sLoggedVguiSurfaceMu);
          firstSeen = sLoggedVguiSurfaceKeys.insert(key).second;
        }
        if (firstSeen) {
          Logger::info(str::format(
            "[VguiSurface] isVgui=1",
            " texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
            " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
            " texStride=", uint32_t(currentInstance.surface.texcoordStride),
            " vguiOffset(floatUnits)=", uint32_t(currentInstance.surface.vguiOffset),
            " — VGUI evaluator reads (vguiOffset .. vguiOffset+8)"));
        }
      }
    }
    // Log unique (texBufIdx, texOffset, texStride, hasLightmap) tuples so
    // we can confirm the lightmap-UV flag actually lands on instances.
    // [TC1Detect] (IA capture) and [TC1Interleave] (interleaver dispatch)
    // are the upstream confirmations; this is the per-instance one. If
    // upstream logs fire but this one shows hasLightmap=0 across the
    // board, the propagation between RaytraceGeometry and the surface
    // metadata is broken.
    {
      static std::unordered_set<uint64_t> sLoggedTc1SurfaceKeys;
      static std::mutex sLoggedTc1SurfaceMu;
      const uint64_t key =
          (uint64_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFFu)
        | ((uint64_t(currentInstance.surface.texcoordOffset) & 0xFFu) << 16)
        | ((uint64_t(currentInstance.surface.texcoordStride) & 0xFFu) << 24)
        | ((uint64_t(currentInstance.surface.hasLightmap ? 1u : 0u)) << 32);
      bool firstSeen = false;
      {
        std::lock_guard<std::mutex> lk(sLoggedTc1SurfaceMu);
        firstSeen = sLoggedTc1SurfaceKeys.insert(key).second;
      }
      if (firstSeen) {
        Logger::info(str::format(
          "[TC1Surface] hasLightmap=", (currentInstance.surface.hasLightmap ? 1 : 0),
          " texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
          " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
          " texStride=", uint32_t(currentInstance.surface.texcoordStride),
          " — lightmap UV (when set) read at element index (texOffset/4 + 2)"));
      }
    }
    // NV-DXVK: per-instance buffer-layout dump for diagnosing the BSP-wall
    // degenerate-UV bug (probes confirmed t1==t2 in UV-space on a real-area
    // world triangle). If two adjacent triangles on the SAME instance show
    // wildly different texcoord layouts, or if texcoordStride doesn't match
    // posStride for known interleaved sources, the interleaver is feeding
    // wrong data. Logged once per (texBufIdx, texStride, texOffset, idxStride)
    // tuple to avoid spam.
    {
      const uint32_t key =
        (uint32_t(currentInstance.surface.texcoordBufferIndex) & 0xFFFu)
        | ((uint32_t(currentInstance.surface.texcoordStride) & 0xFFu) << 12)
        | ((uint32_t(currentInstance.surface.texcoordOffset) & 0xFFu) << 20)
        | ((uint32_t(currentInstance.surface.indexStride) & 0xFu) << 28);
      // NV-DXVK 2026-08-07: this set had NO lock while the two above it did,
      // even though all three run on the same call path. That path IS
      // multi-threaded -- InstanceManager::m_fanoutPrevHitCount is atomic
      // precisely because "findSimilarInstance is reachable from the scene-
      // manager's draw-processing threads", and SceneManager::
      // processDrawCallState keeps its fanout scratch thread_local for the
      // same reason. An unsynchronised unordered_set::insert from two threads
      // corrupts the container, so this was a latent crash that only the
      // census's own rarity kept quiet. Locked now; it costs nothing, because
      // the whole block is behind rtx.logSurfaceGeomDiag.
      static std::unordered_set<uint32_t> seenBlasGeom;
      static std::mutex seenBlasGeomMu;
      bool firstSeen = false;
      {
        std::lock_guard<std::mutex> lk(seenBlasGeomMu);
        firstSeen = seenBlasGeom.insert(key).second;
      }
      if (firstSeen) {
        Logger::info(str::format(
          "[BlasGeom] texBufIdx=", uint32_t(currentInstance.surface.texcoordBufferIndex),
          " texStride=", uint32_t(currentInstance.surface.texcoordStride),
          " texOffset=", uint32_t(currentInstance.surface.texcoordOffset),
          " posBufIdx=", uint32_t(currentInstance.surface.positionBufferIndex),
          " posStride=", uint32_t(currentInstance.surface.positionStride),
          " posOffset=", uint32_t(currentInstance.surface.positionOffset),
          " idxBufIdx=", uint32_t(currentInstance.surface.indexBufferIndex),
          " idxStride=", uint32_t(currentInstance.surface.indexStride),
          " normBufIdx=", uint32_t(currentInstance.surface.normalBufferIndex),
          " normStride=", uint32_t(currentInstance.surface.normalStride)));
      }
    }
  }

  // Returns true if the instance was modified
  bool InstanceManager::applyDeveloperOptions(RtInstance& currentInstance, const DrawCallState& drawCall) {
    if (!RtxOptions::enableInstanceDebuggingTools()) {
      return false;
    }

    if ((
      currentInstance.m_instanceVectorId >= RtxOptions::instanceOverrideInstanceIdx() &&
      currentInstance.m_instanceVectorId < RtxOptions::instanceOverrideInstanceIdx() + RtxOptions::instanceOverrideInstanceIdxRange())) {

      if (RtxOptions::instanceOverrideSelectedInstancePrintMaterialHash())
        Logger::info(str::format("Draw Call Material Hash: ", drawCall.getMaterialData().getHash()));

      // Apply world offset
      Vector3 worldOffset = RtxOptions::instanceOverrideWorldOffset();
      currentInstance.teleportWithHistory(translationMatrix(worldOffset));

      return true;
    }

    return false;
  }

  void InstanceManager::bindMaterial(RtInstance& instance, const RtSurfaceMaterial& material, const uint32_t indexInCache) {
    if (material.getType() == RtSurfaceMaterialType::Opaque) {
      instance.m_albedoOpacityTextureIndex = material.getOpaqueSurfaceMaterial().getAlbedoOpacityTextureIndex();
      instance.m_samplerIndex = material.getOpaqueSurfaceMaterial().getSamplerIndex();
    } else if (material.getType() == RtSurfaceMaterialType::RayPortal) {
      instance.m_albedoOpacityTextureIndex = material.getRayPortalSurfaceMaterial().getMaskTextureIndex();
      instance.m_samplerIndex = material.getRayPortalSurfaceMaterial().getSamplerIndex();
      instance.m_secondaryOpacityTextureIndex = material.getRayPortalSurfaceMaterial().getMaskTextureIndex2();
      instance.m_secondarySamplerIndex = material.getRayPortalSurfaceMaterial().getSamplerIndex2();
    }

    instance.m_vkInstance.instanceCustomIndex = (instance.m_vkInstance.instanceCustomIndex & ~(surfaceMaterialTypeMask << CUSTOM_INDEX_MATERIAL_TYPE_BIT));
    instance.m_vkInstance.instanceCustomIndex |= ((uint32_t)material.getType() << CUSTOM_INDEX_MATERIAL_TYPE_BIT);

    // Fetch the material from the cache.
    //
    // NV-DXVK [perf]: ResourceCache::find is an unordered_map lookup keyed on the
    // whole RtSurfaceMaterial, and this runs once per INSTANCE per frame
    // (~15,500x). The only caller has just come out of createSurfaceMaterial,
    // which resolved that exact index and hands it back through out_indexInCache
    // -- both refer to m_surfaceMaterialCache, so the lookup was re-deriving a
    // value the caller already held. Same shape as the other wins in the v5
    // handoff: not a faster lookup, no lookup.
    if (indexInCache != UINT32_MAX) {
      instance.surface.surfaceMaterialIndex = indexInCache;
    } else {
      m_pResourceCache->find(material, instance.surface.surfaceMaterialIndex);
    }
  }

  // Updates the state of the instance with the draw call inputs
  // It handles multiple draw calls called for a same instance within a frame
  // To be called on every draw call
  // [ZigGun] handoff: updateInstance has no DxvkContext (can't GPU-readback), so
  // we tag the near-eye high-vtx gun instance here and read it back in
  // createViewModelInstances (which has ctx). Same frame, set before the accel
  // build, so the pointer is valid when consumed.
  static RtInstance* s_zigGunInstance = nullptr;
  // NV-DXVK: frame id when s_zigGunInstance was last tagged. The tag is a RAW
  // pointer into the pooled RtInstance storage and is only safe to dereference in
  // the SAME frame it was set. A tag left over from a previous frame can dangle
  // after the instance is freed (e.g. device loss on alt+tab skips the
  // consume+null in createViewModelInstances) -> use-after-free: getBlas() reads
  // the freed (0xDD-filled) instance's m_blas as a wild pointer, then
  // ->frameLastUpdated dereferences it and AVs. Gate every deref on this == fid.
  static uint32_t    s_zigGunInstanceFrameId = UINT32_MAX;

  // NV-DXVK [Perf.UpdInst]: stage split of updateInstance -- the deepest leaf of
  // the measured chain and the biggest single item left in the frame (~20 ms of
  // ~65, [Perf.SceneObj] update=62% of a function that runs ~16,100x/frame).
  //
  // SAMPLED DURATIONS, EXACT COUNTS. This function runs once per INSTANCE, not
  // once per draw. Nine steady_clock reads on every instance would be
  // 9 x 16,100 x ~41 ns = ~6.6 ms/frame -- a third of the thing being measured.
  // That is the rtx.perfGapSampler mistake (v4 sec 0c) and the 11x mis-sizing of
  // rtx.perfSceneObjSplit, which was budgeted per DRAW and billed per INSTANCE.
  // So: time 1 instance in 64 (~252/frame, ~0.09 ms) and count every instance
  // exactly. estMsPerFrame is mean x true count, which is the [Perf.CsCmd]
  // construction -- a hot stage cannot hide behind sampling probability and a
  // rarely-taken stage is not mistaken for a cheap one.
  //
  // THE COUNTERS MATTER MORE THAN THE TIMINGS. They are read from values the
  // function already computes, and REDUNDANT (= no transform change, no material
  // change, no previous positions) answers sec 4c's open question directly: how
  // much of this 20 ms is rebuilding state that did not change?
  namespace {
    struct UpdInstSplitGuard {
      using clk = std::chrono::steady_clock;
      // NV-DXVK [Perf.UpdInst] 2026-08-09: 9 -> 10. Stage 9 (`fastRet`) closes the
      // FAST-PATH span. See the mark(9) site for why it had to exist.
      static constexpr int      kStages     = 10;
      static constexpr uint32_t kSampleMask = 63u;   // duration-sample 1 in 64

      bool on = false;
      bool timed = false;
      uint32_t frameId = 0;
      clk::time_point t;
      int64_t ns[kStages] = {};
      // NV-DXVK [Perf.UpdInst] 2026-08-09: which marks actually FIRED on this
      // instance. Needed because updateInstance has an early return (the fast
      // path, at the mark(9) site ~:6795) that
      // most instances take, so stages differ in how many instances reach them and
      // a single denominator cannot serve them all -- see the print.
      uint16_t marked = 0;

      // Exact, every instance. Set once at the end from the function's own values.
      bool cFirst = false, cXfChg = false, cMatChg = false, cPrevPos = false, cStatic = false;
      // NV-DXVK [perf] sec 4c: set when the surf-state guard skipped the block.
      // Reported as surfSkip= so the fix is verified by MECHANISM (the guard fired
      // on N% of instances) and not only by the surf timing moving -- a key that
      // is accidentally always-equal and a key that is correctly always-equal look
      // identical in the timing alone.
      bool cSurfSkip = false;
      void surfSkipped() { cSurfSkip = true; }
      // NV-DXVK [perf] sec 4c: set when the event fanout was skipped outright.
      bool cTailSkip = false;
      void tailSkipped() { cTailSkip = true; }

      UpdInstSplitGuard(bool enabled, uint32_t fid) : on(enabled), frameId(fid) {
        if (!on) {
          return;
        }
        // NV-DXVK [Perf.UpdInst] 2026-08-09: PHASE-OFFSET BY HALF A PERIOD, and
        // the reason is a measurement bug that both cross-vals caught.
        //
        // SceneObjSplitGuard (~:3868) uses the IDENTICAL construction -- same
        // mask, same increment, same thread -- and processSceneObjectImpl calls
        // updateInstance exactly once per instance. So the two s_seq counters ran
        // in LOCKSTEP and sampled the SAME instances. SceneObj's `update` span
        // brackets this call, so every instance it timed was an instance this
        // guard was also timing: its sampled mean absorbed this guard's
        // clk::now() reads (~4 of them, ~164 ns, on the 92% fast path) and was
        // then multiplied by the FULL instance count -- projecting a 1-in-64 cost
        // onto 64-in-64.
        //
        // Measured: SceneObj summed to 18.71 ms against [ProcDCS] instMs 14.64 --
        // children larger than their parent, which is impossible -- and both
        // cross-vals FAILed at a constant 21.7% across windows with different
        // absolute values. 15,266 inst x ~164 ns is ~2.5 ms/frame, the right
        // order for the 3.66 ms residual between UpdInst's 8.27 and SceneObj's
        // 11.93.
        //
        // Starting half a period out means the outer probe never samples an
        // instance this one is timing. TWO SAMPLED PROBES AT THE SAME STRIDE ARE
        // NOT TWO INDEPENDENT INSTRUMENTS -- if a third nested probe is ever
        // added here, give it its own phase too.
        static thread_local uint32_t s_seq = (kSampleMask + 1u) / 2u;
        timed = ((s_seq++ & kSampleMask) == 0u);
        if (timed) {
          t = clk::now();
        }
      }

      void mark(int s) {
        if (!timed) {
          return;
        }
        const auto now = clk::now();
        ns[s] = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
        marked |= static_cast<uint16_t>(1u << s);
      }

      void counts(bool first, bool xfChg, bool matChg, bool prevPos, bool isStatic) {
        cFirst = first; cXfChg = xfChg; cMatChg = matChg; cPrevPos = prevPos; cStatic = isStatic;
      }

      ~UpdInstSplitGuard() {
        if (!on) {
          return;
        }

        static thread_local int64_t  sNs[kStages] = {};
        // NV-DXVK [Perf.UpdInst] 2026-08-09: sampled instances that REACHED each
        // stage. Only touched on sampled instances (1 in 64), so ~10 increments
        // per 64 instances.
        static thread_local uint64_t sMarked[kStages] = {};
        static thread_local uint64_t sSamples = 0, sInst = 0, sFrames = 0;
        static thread_local uint64_t sFirst = 0, sXfChg = 0, sMatChg = 0, sPrevPos = 0,
                                     sStatic = 0, sRedundant = 0, sSurfSkip = 0, sTailSkip = 0;
        static thread_local uint32_t sLastFrame = UINT32_MAX;
        static thread_local clk::time_point sLastLog{};
        static thread_local bool sInit = false;

        const auto tEnd = clk::now();
        if (!sInit) { sLastLog = tEnd; sInit = true; }

        if (sLastFrame != frameId) { sLastFrame = frameId; ++sFrames; }

        ++sInst;
        if (timed) {
          ++sSamples;
          for (int i = 0; i < kStages; ++i) {
            sNs[i] += ns[i];
            if (marked & (1u << i)) {
              ++sMarked[i];
            }
          }
        }
        if (cFirst)   ++sFirst;
        if (cXfChg)   ++sXfChg;
        if (cMatChg)  ++sMatChg;
        if (cPrevPos) ++sPrevPos;
        if (cStatic)  ++sStatic;
        if (cSurfSkip) ++sSurfSkip;
        if (cTailSkip) ++sTailSkip;
        if (!cXfChg && !cMatChg && !cPrevPos) ++sRedundant;

        if (std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - sLastLog).count() < 3000) {
          return;
        }

        const int64_t  smp    = sSamples ? int64_t(sSamples) : 1;
        const uint64_t frames = sFrames ? sFrames : 1;
        const uint64_t inst   = sInst ? sInst : 1;
        const double   perFrm = double(sInst) / double(frames);

        // Three columns, three questions: what a stage costs on an instance that
        // RUNS it (usPerInst), what it costs the frame (estMsPerFrame), and how
        // many instances get there at all (reachPct). The first and third are new
        // 2026-08-09; before that a single denominator served every stage and the
        // fast-path early return (the mark(9) site) made the late ones unreadable.
        std::string us, est, rch;
        double estMs[kStages] = {};   // kept for the [Perf.Report] publish below
        static const char* kNames[kStages] = {
          "entry", "surf", "xform", "flags", "viewmodel", "billboard", "census", "anticull", "tail",
          "fastRet"
        };
        for (int i = 0; i < kStages; ++i) {
          // TWO DENOMINATORS, on purpose -- they answer different questions and
          // sharing one is what made this instrument unreadable.
          //  usPerInst   -> divide by the instances that ACTUALLY RAN the stage.
          //     The fast-path early return takes ~92% of instances, so stages 1-8
          //     are reached by ~8% of them; dividing those by sSamples reported a
          //     per-instance mean ~13x too low and made every stage look free.
          //  estMsPerFrame -> keep the sNs/sSamples form. The sampled zeros and
          //     the reach fraction cancel exactly, so this stays an UNBIASED
          //     estimate of the stage's total ms/frame and remains directly
          //     comparable to [ProcDCS] instMs. Do not "fix" it to match usPerInst.
          const uint64_t reach  = sMarked[i] ? sMarked[i] : 1;
          const double   meanNs = double(sNs[i]) / double(reach);
          estMs[i] = (double(sNs[i]) / double(smp)) * perFrm / 1e6;
          us  += str::format(" ", kNames[i], "=", (meanNs / 1000.0));
          est += str::format(" ", kNames[i], "=", estMs[i]);
          rch += str::format(" ", kNames[i], "=", (sMarked[i] * 100 / uint64_t(smp)));
        }

        // NV-DXVK [Perf.Report] 2026-08-09: publish updateInstance's stages as
        // children of SceneObjUpdateMs. estMsPerFrame only -- already computed
        // for the line above, so this adds no clock reads. Stage order is the
        // kNames order; `rest` folds the four ~0.1 ms stages so the nested rows
        // stay readable and the sum still closes exactly.
        {
          using perfreport::Slot;
          double total = 0.0;
          for (int i = 0; i < kStages; ++i) {
            total += estMs[i];
          }
          perfreport::publish(Slot::UpdInstEntryMs,   estMs[0]);
          perfreport::publish(Slot::UpdInstSurfMs,    estMs[1]);
          perfreport::publish(Slot::UpdInstXformMs,   estMs[2]);
          perfreport::publish(Slot::UpdInstFlagsMs,   estMs[3]);
          perfreport::publish(Slot::UpdInstRestMs,    estMs[4] + estMs[5] + estMs[6] + estMs[7]);
          perfreport::publish(Slot::UpdInstTailMs,    estMs[8]);
          perfreport::publish(Slot::UpdInstFastRetMs, estMs[9]);
          perfreport::publish(Slot::UpdInstTotalMs,   total);
        }

        Logger::info(str::format(
          "[Perf.UpdInst] inst=", sInst, " frames=", sFrames,
          " instPerFrame=", perFrm, " samples=", sSamples,
          " | usPerInst", us,
          " | estMsPerFrame", est,
          // NV-DXVK [Perf.UpdInst] 2026-08-09: % of SAMPLED instances that reached
          // each stage -- the mechanism check for the two columns above. Expected
          // shape: entry=100 (mark(0) is unconditional), fastRet~92 (matches
          // [Perf.FastInst] fast / instPerFrame), and stages 1-8 all at
          // ~(100 - fastRet) -- EXCEPT surf, which sits inside a conditional (see
          // the mark(1) site) and should read ~1 point lower. That one is checked
          // and benign. Any OTHER stage reading differently from its neighbours
          // means a further conditional exit exists between the marks -- which is
          // exactly the bug fastRet was added to close, so check here first.
          " | reachPct", rch,
          " | first=",     (sFirst     * 100 / inst),
          "% xfChg=",      (sXfChg     * 100 / inst),
          "% matChg=",     (sMatChg    * 100 / inst),
          "% prevPos=",    (sPrevPos   * 100 / inst),
          "% static=",     (sStatic    * 100 / inst),
          "% REDUNDANT=",  (sRedundant * 100 / inst),
          // NV-DXVK [perf] sec 4c: share of instances whose surf block was skipped
          // because every input it reads was unchanged. Expect this to track
          // REDUNDANT closely in a settled scene; a low value with REDUNDANT high
          // means the key is picking up something that moves per frame, which is a
          // bug in computeSurfStateKey, not a property of the scene.
          "% surfSkip=",   (sSurfSkip  * 100 / inst),
          // NV-DXVK [perf] sec 4c: share of instances whose event fanout was
          // skipped outright. Should sit just under surfSkip in a settled scene.
          // A COLLAPSE here while surfSkip stays high points at the binding epoch
          // moving every frame (streaming churn), not at the material key.
          "% tailSkip=",   (sTailSkip  * 100 / inst), "%"));

        for (int i = 0; i < kStages; ++i) { sNs[i] = 0; sMarked[i] = 0; }
        sSamples = sInst = sFrames = 0;
        sFirst = sXfChg = sMatChg = sPrevPos = sStatic = sRedundant = sSurfSkip = sTailSkip = 0;
        sLastLog = tEnd;
      }
    };
  }

  // NV-DXVK [Perf.FastInst]: exact counters for the rtx.fastInstanceUpdate path.
  // Relaxed atomics only -- the shape measured at ~zero cost, NOT the per-instance
  // mutex shape rtx.logMapGate's note documents at multiple ms. One warn line
  // every 300 frames, from whichever thread wins the CAS.
  namespace {
    enum FastInstReason : uint32_t {
      kFastNotAllowed  = 0,  // drawState.fastPathAllowed false (option off / eye / option-flip frame)
      kFastKeyMiss     = 1,  // instance state key mismatch (genuine state change or fresh instance)
      kFastCreated     = 2,  // first frame after creation (teleport path)
      kFastPop         = 3,  // excluded population (viewmodel/player/unordered/decal/skinned/portal/gun)
      kFastBitsMiss    = 4,  // per-draw fast bits changed (winding/parity/subview/rt-target)
      kFastXformMiss   = 5,  // objectToWorld bytes changed
      kFastSpatialMiss = 6,  // SpatialMap not provably in sync
      kFastMaskMiss    = 7,  // mask unreconstructable (was hidden, now visible)
      kFastReasonCount = 8
    };
    std::atomic<uint64_t> s_fastInstHit { 0 };
    std::atomic<uint64_t> s_fastInstSlow[kFastReasonCount] = {};
    // NV-DXVK (2026-08-08d §3): fast-path commits whose non-skippable handler
    // dispatch was elided via skippableWhenNoPendingOmmWork. Healthy steady
    // state tracks `fast` almost exactly; a large gap means instances carry a
    // pending OMM flag forever (a starvation bug worth investigating).
    std::atomic<uint64_t> s_fastInstOmmSkips { 0 };
    std::atomic<uint32_t> s_fastInstLastLog { 0 };

    inline void fastInstEmit(uint32_t frameId) {
      const uint32_t last = s_fastInstLastLog.load(std::memory_order_relaxed);
      if (frameId - last < 300u) {
        return;
      }
      uint32_t expected = last;
      if (!s_fastInstLastLog.compare_exchange_strong(expected, frameId, std::memory_order_relaxed)) {
        return;
      }
      const double frames = double(frameId - last);
      const double hit = double(s_fastInstHit.exchange(0, std::memory_order_relaxed)) / frames;
      double slow[kFastReasonCount];
      for (uint32_t i = 0; i < kFastReasonCount; ++i) {
        slow[i] = double(s_fastInstSlow[i].exchange(0, std::memory_order_relaxed)) / frames;
      }
      const double ommSkip = double(s_fastInstOmmSkips.exchange(0, std::memory_order_relaxed)) / frames;
      Logger::warn(str::format(
        "[Perf.FastInst] perFrame: fast=", hit,
        " ommSkip=", ommSkip,
        " | slow: notAllowed=", slow[kFastNotAllowed],
        " keyMiss=", slow[kFastKeyMiss],
        " created=", slow[kFastCreated],
        " pop=", slow[kFastPop],
        " bitsMiss=", slow[kFastBitsMiss],
        " xformMiss=", slow[kFastXformMiss],
        " spatialMiss=", slow[kFastSpatialMiss],
        " maskMiss=", slow[kFastMaskMiss]));
    }
  }

  void InstanceManager::updateInstance(RtInstance& currentInstance,
                                       const CameraManager& cameraManager,
                                       const BlasEntry& blas,
                                       const DrawCallState& drawCall,
                                       MaterialData& materialData,
                                       const DrawScopedState& drawState,
                                       const FanoutSplit* split,
                                       const SpatialKeyHint& keyHint) {
    UpdInstSplitGuard uiSplit(RtxOptions::perfUpdateInstSplit(), m_device->getCurrentFrameId());

    // NV-DXVK [Phase2b]: open this call's per-instance deferred-ops record. The
    // divergence sites below (buffer bind, billboard stage, OMM fanout) write
    // into it; the CS record step replays it after the bake. One entry per
    // updateInstance call keeps fanout placements' event args exact.
    if (inShardedInstancePhase()) {
      t_shardPhase.info->pendingOps.push_back(
          ShardedDrawInfo::PendingInstanceOps { &currentInstance });
      t_shardPhase.currentOps = &t_shardPhase.info->pendingOps.back();
    }
    // NV-DXVK [sticky IgnoreAntiCulling]: preserve IgnoreAntiCulling across
    // shader-variant draws that update the same RtInstance within a frame.
    // The mountain mesh is drawn with multiple d3d11 shader variants (e.g.
    // VS_c10aa, VS_2f543cd, VS_2904d2 SHA1 prefixes for the same logical
    // shader). Only the variants that pass SetSkyCategoryFromCb2's reproject
    // gate carry IgnoreAntiCulling on dcs; non-reproject variants arrive
    // with the flag unset, and an unconditional assignment would clobber
    // the flag that an earlier draw set this same frame. OR-preserving
    // means: once any draw this frame tagged the instance as sub-view,
    // it stays tagged for the whole frame's GC pass.
    const CategoryFlags prevFlags = currentInstance.m_categoryFlags;
    currentInstance.m_categoryFlags = drawCall.getCategoryFlags();
    if (prevFlags.test(InstanceCategories::IgnoreAntiCulling)) {
      currentInstance.m_categoryFlags.set(InstanceCategories::IgnoreAntiCulling);
    }

    // NV-DXVK [UpdInst2904 probe]: for VS_2904d2 draws, log the incoming
    // drawCall's categoryFlags and stablePropId so we know whether the
    // SetSkyCategoryFromCb2 reproject tagging is reaching updateInstance at
    // all. If incomingIgnAC=0 here despite PropIdTrace firing for the same
    // logical shader, something between d3d11_rtx and processSceneObject
    // is clearing the bit (replacement path, anonymous dcs copy, etc).
    // NV-DXVK [perf] 2026-08-08e ([Perf.UpdInst] entry=1.66 ms, the largest
    // stage left in updateInstance): option gate FIRST. The hash compare
    // itself is one deref chain into blas.input -- a cold line of a large
    // BlasEntry -- paid by all ~15.6k instances every frame for a probe whose
    // investigation (VS_2904 sky reproject tagging) is resolved. logGeomDiag
    // is a static option read; the cold deref now happens only when the
    // diagnostics are actually wanted.
    if (RtxOptions::logGeomDiag()
        && blas.input.getTransformData().vertexShaderHash == 0x2904d2163ef31a17ull) {
      thread_local uint32_t sUpdProbe = 0;
      if (sUpdProbe < 16 || (sUpdProbe & 0x3FF) == 0) {
        const auto incomingFlags = drawCall.getCategoryFlags();
        Logger::info(str::format(
          "[UpdInst2904] #", sUpdProbe,
          " f=", m_device->getCurrentFrameId(),
          " incomingIgnAC=", (incomingFlags.test(InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
          " incomingPropId=0x", std::hex, drawCall.getTransformData().stablePropId, std::dec,
          " incomingFlagsRaw=0x", std::hex, incomingFlags.raw(), std::dec,
          " prevInstFlagsRaw=0x", std::hex, prevFlags.raw(), std::dec,
          " finalInstFlagsRaw=0x", std::hex, currentInstance.m_categoryFlags.raw(), std::dec));
      }
      sUpdProbe += 1;
    }
    // NV-DXVK [fanout split]: a split placement is a plain single-placement
    // surface. Clearing instancesToObject is what makes AccelManager take the
    // ordinary addBlas path instead of addPointInstancerBlas, and what makes
    // RtSurface::writeGPUData use surface.objectToWorld directly instead of
    // deriving a transform from (surfaceIndex - surfaceIndexOfFirstInstance).
    // surfaceIndexOfFirstInstance is reset alongside it so an RtInstance that was
    // a point instancer before the option was toggled cannot keep a stale base
    // slot; writeGPUData's PI branch needs BOTH to be set.
    if (split != nullptr) {
      currentInstance.surface.instancesToObject = nullptr;
      currentInstance.surface.instancesToObjectOwner = nullptr;
      currentInstance.surface.surfaceIndexOfFirstInstance = SIZE_MAX;
    } else {
      currentInstance.surface.instancesToObject = drawCall.getTransformData().instancesToObject;
      // NV-DXVK: Couple lifetime of the transform vector to this RtInstance so the
      // raw pointer above cannot dangle once the draw-call source's own ring-buffer
      // releases its reference. Null for sources with externally-owned storage
      // (USD replacements, etc.) — that's fine, they already manage lifetime.
      currentInstance.surface.instancesToObjectOwner = drawCall.getTransformData().instancesToObjectOwner;
    }

    // NV-DXVK: flag sub-view (3D-skybox) geometry so WriteGPUData negates its
    // shading normal — corrects the reproject's inverted-winding normal flip that
    // leaves the distant mountains facing away from the sun (N·L<0 -> black).
    // Gate widened to isSubView OR isSubViewSkybox (the dome is tagged the latter
    // and may not carry isSubView at instance time).
    const bool svFlip = drawCall.getTransformData().isSubView || drawCall.getTransformData().isSubViewSkybox;
    currentInstance.surface.flipShadingNormal = RtxOptions::flipSubViewSkyboxNormals() && svFlip;
    // NV-DXVK [FlipNormalDiag]: one line per VS — confirms WHICH draws get the flip
    // and whether they even have a vertex-normal buffer (if not, the shading normal
    // is the geometric/face normal and negating normalInstanceToWorld is a no-op).
    if (svFlip) {
      // NV-DXVK [perf] 2026-08-06: this took a process-global mutex and did an
      // unordered_set insert for EVERY sub-view instance, EVERY frame, purely
      // to decide whether to emit a line that only ever fires once per VS hash.
      // The output was one-shot; the work was per-instance-per-frame, on
      // dxvk-cs - the thread [ThreadCensus] measures at ~96 ms of a ~99 ms
      // frame. Same defect as the [DrawName] probe in §5.3 of the CPU handoff:
      // bound the WORK, not just the output.
      //
      // A thread-local set fronts the shared one. Steady state (the hash has
      // been seen by this thread before) is one hash lookup and no lock at all.
      // The shared set stays the authority for "first globally", so the line is
      // still emitted exactly once across all threads rather than once per
      // thread - the thread-local set only ever suppresses work, never grants
      // permission to log.
      // NV-DXVK [perf] 2026-08-07: SINGLE-ENTRY FRONT on top of the thread-local
      // set. The note above is right about the principle and the fix it describes
      // did not finish the job: a thread_local unordered_set::insert is still a
      // hash plus a bucket probe plus a cache miss, paid by EVERY sub-view instance
      // EVERY frame, to gate a line that fires once per VS for the whole session.
      // Sub-view instances arrive in runs that share a vertex shader, so comparing
      // against the last hash this thread already resolved collapses the steady
      // state to one 8-byte compare and touches the set only on a VS change.
      // Sized honestly: this is small -- the block is already gated on svFlip, so
      // it is bounded by the sub-view population, not by all ~15,500 instances.
      // Do not expect it to move [Perf.UpdInst] entry on its own.
      const XXH64_hash_t vsFn = drawCall.getTransformData().vertexShaderHash;
      static thread_local XXH64_hash_t tLastFn = 0;
      static thread_local bool tLastFnValid = false;
      bool firstFn = false;
      if (!tLastFnValid || vsFn != tLastFn) {
        static thread_local std::unordered_set<XXH64_hash_t> tFnSeen;
        if (tFnSeen.insert(vsFn).second) {
          static std::mutex sFnMu;
          static thread_local std::unordered_set<XXH64_hash_t> sFnLog;
          std::lock_guard<std::mutex> g(sFnMu);
          firstFn = sFnLog.insert(vsFn).second;
        }
        // Only after the set has resolved this hash, so a VS is never skipped
        // before its first-sighting question has been asked.
        tLastFn = vsFn;
        tLastFnValid = true;
      }
      if (firstFn) {
        Logger::info(str::format(
          "[FlipNormalDiag] vsXxh=0x", std::hex, uint64_t(vsFn), std::dec,
          " isSubView=", drawCall.getTransformData().isSubView ? 1 : 0,
          " isSubViewSkybox=", drawCall.getTransformData().isSubViewSkybox ? 1 : 0,
          " hasNormalBuffer=", drawCall.getGeometryData().normalBuffer.defined() ? 1 : 0,
          " flipApplied=", currentInstance.surface.flipShadingNormal ? 1 : 0));
      }
    }

    // NV-DXVK [Stable prop ID]: mirror the drawCall's per-prop identity onto
    // the instance so subsequent spatial-map operations from onTransformChanged
    // and teleport (which see no drawCall reference) can use the same cache
    // key. ONLY OVERWRITE WHEN NON-ZERO — a non-reproject draw variant for
    // the same logical mountain arrives with stablePropId=0; clobbering the
    // existing non-zero value would erase the SpatialMap entry at the propId
    // slot (via move(oldHash=propIdA, overrideHash=0) → erase+insert at
    // matrix-hash slot), guaranteeing the next frame's propId lookup misses.
    //
    // NV-DXVK [fanout split]: mirror the SPLIT's identity, not the batch's.
    // Inheriting the batch id would be actively wrong — it names whichever prop
    // was element [0] when the draw was submitted, and it rolls every frame. 0
    // here means "no engine handle available", which leaves the instance on
    // matrix-bytes keying, matching what findSimilarInstance looked it up with.
    const uint64_t incomingStablePropId = split != nullptr
      ? split->stablePropId
      : drawCall.getTransformData().stablePropId;
    if (split != nullptr) {
      // Assign UNCONDITIONALLY for a split placement, including 0. The
      // non-zero-only guard above exists because a second, non-reproject draw
      // variant of the same mountain arrives with propId 0 and would erase a good
      // entry — that hazard does not exist here (a placement's id comes from one
      // source and does not vary between variants), and keeping a stale id after
      // the engine handles stopped being available would be strictly worse: the
      // instance would sit in the map under an id that findSimilarInstance is no
      // longer looking up with. m_stablePropId must always equal what the lookup
      // used, or the entry is orphaned until GC.
      currentInstance.m_stablePropId = incomingStablePropId;
    } else if (incomingStablePropId != 0) {
      currentInstance.m_stablePropId = incomingStablePropId;
    }

    // setFrameLastUpdated() must be called first as it resets instance's state on a first call in a frame
    const bool isFirstUpdateThisFrame = currentInstance.setFrameLastUpdated(m_device->getCurrentFrameId());

    // These can change in the Runtime UI so need to check during update
    currentInstance.m_isHidden = currentInstance.testCategoryFlags(InstanceCategories::Hidden);
    currentInstance.m_isPlayerModel = currentInstance.testCategoryFlags(InstanceCategories::ThirdPersonPlayerModel);
    currentInstance.m_isWorldSpaceUI = currentInstance.testCategoryFlags(InstanceCategories::WorldUI);

    // Hide the sky instance since it is not raytraced.
    // Sky mesh and material are only good for capture and replacement purposes.
    if (drawCall.cameraType == CameraType::Sky) {
      currentInstance.m_isHidden = true;
    }

    // Register camera
    bool isNewCameraSet = currentInstance.registerCamera(drawCall.cameraType, m_device->getCurrentFrameId());

    const bool overridePreviousCameraUpdate = isNewCameraSet &&
      (drawCall.cameraType == CameraType::Main ||
       // Don't overwrite transform from when the instance was seen with the main camera
       !currentInstance.isCameraRegistered(CameraType::Main));

    // NV-DXVK [perf] 2026-08-07: was calculateAlphaState(drawCall, materialData)
    // here, once per placement. It is a pure function of those two, neither of
    // which varies across a draw's placements, so it is now computed once per
    // draw in computeDrawScopedState. Bound by reference: it is copied wholesale
    // into surface.alphaState further down, and that copy is the only consumer
    // that needs its own storage.
    const RtSurface::AlphaState& alphaState = drawState.alphaState;
    bool hasTransformChanged = false;
    bool hasPreviousPositions = false;
    // NV-DXVK [perf] sec 4c: result of the single per-instance state-key compare,
    // set in the `surf` block below and read again by the event-fanout gate at the
    // bottom. Defaults false so any path that does not reach the compare (e.g.
    // !isFirstUpdateThisFrame with overridePreviousCameraUpdate) runs both blocks.
    bool instStateUnchanged = false;

    if (!isFirstUpdateThisFrame) {
      // This is probably the same instance, being drawn twice!  Merge it
      mergeInstanceHeuristics(currentInstance, drawCall, alphaState);
    }
    
    uiSplit.mark(0);   // NV-DXVK [Perf.UpdInst]: end of `entry`

    // NV-DXVK [perf] handoff v5 sec 4c -- THE `surf` SKIP WAS INVESTIGATED AND IS
    // NOT AVAILABLE. Recorded here so it is not re-attempted.
    //
    // 4c proposes an exact change-detector over this block's inputs, on the
    // grounds that it is gated on isFirstUpdateThisFrame (99%) with matChg=0%,
    // and that "its inputs are enumerable". The inputs are enumerable. The
    // block is not skippable anyway, for two structural reasons that no amount
    // of key-auditing fixes:
    //
    //   (1) IT IS NOT ASSIGNMENT-ONLY. It promotes currentInstance.m_isHidden in
    //       two places -- the matTf2Fog hide and the sourcePsWritesCoverageMask
    //       hide further down. `entry` RESETS m_isHidden every frame from the
    //       Hidden category flag, and `flags` later reads it to force the
    //       instance mask to 0. Skipping this block therefore un-hides the TF2
    //       fog-pipeline surfaces and the SV_Coverage sky overlay -- both
    //       previously-diagnosed visual bugs, silently reintroduced.
    //
    //   (2) ITS OUTPUT LEGITIMATELY CHANGES EVERY FRAME. surface.objectPickingValue
    //       is assigned drawCall.drawCallID, and that is a PER-FRAME draw counter
    //       (D3D11Rtx::m_drawCallID, reset by resetDrawCallID() and accumulated
    //       across deferred contexts each frame). So a "nothing changed" key is
    //       false by construction: include drawCallID and the detector never
    //       fires, exclude it and objectPickingValue freezes, breaking picking.
    //
    // Closing 4c for this stage means RESTRUCTURING it -- hoisting the two
    // m_isHidden promotions and the picking-value assignment out of the skipped
    // region -- not gating it. At 0.167 us/inst floor-corrected (~2.6 ms) that is
    // worth doing, but it is a redesign with its own verification, not the
    // key-audit 4c describes. The cheap wins inside the block (the two probe
    // latches below) were taken instead.
    //
    // Updates done only once a frame unless overriden due to an explicit state
    if (isFirstUpdateThisFrame || overridePreviousCameraUpdate) {

      if (isFirstUpdateThisFrame) {
        // NV-DXVK [perf] 2026-08-07 (v4): THE KEY TEST IS HOISTED ABOVE THIS
        // PREAMBLE. It used to sit ~55 lines below, so everything here ran on
        // every instance even when the guard then decided nothing had changed.
        //
        // WHY v1-v3 ALL FAILED (see the block below, kept intact): each tried to
        // make the KEY cheaper. The key is already down to a 16-byte hash of two
        // registers -- ~10 ns, ~0.15 ms/frame over 15.5k instances -- so it was
        // never the cost. [Perf.UpdInst] 18:37 measures `surf` at ~4.0 ms/frame
        // (probe-corrected) with surfSkip=99%, and that 4 ms is THIS preamble:
        // processInstanceBuffers plus four hash getters, on 99% of instances
        // (first=99%). matChg=0% in every capture ever taken, so the two
        // material hashes are recomputed 15.5k times a frame to produce a value
        // that has never once differed. The guard was simply a dozen lines too
        // late.
        //
        // SAFETY. Every read below is an input to computeDrawStateKey, so a
        // matching key proves each would reproduce its current value -- skipping
        // preserves the last-written m_materialHash / m_materialDataHash /
        // m_texcoordHash / m_indexHash rather than leaving them stale, because a
        // change in any of them changes the key and therefore cannot be skipped.
        //
        // hasMaterialChanged is the ONE exception and is NOT a pure function of
        // the key: it is an edge signal (old != new), so a frame that legitimately
        // set it true is followed by frames whose key matches. Leaving it latched
        // would report a material change every frame forever after. It is cleared
        // explicitly on the skip path. That asymmetry is why this is written as
        // if/else rather than an early-out around the block.
        const bool instKeyEligible = drawState.keyEligible;
        const XXH64_hash_t instStateKey = instKeyEligible
          ? mixInstStateKey(drawState.stateKey, currentInstance.m_categoryFlags)
          : kEmptyHash;
        instStateUnchanged = instKeyEligible && (instStateKey == currentInstance.m_instStateKey);
        currentInstance.m_instStateKey = instStateKey;

        // NV-DXVK [perf] fastInstanceUpdate -- THE DIRTY-LIST REALISATION of the
        // [Perf.UpdInst] REDUNDANT=97% finding (rtx.conf ~1690-1950). The visits
        // themselves cannot be avoided (the game re-draws everything), but a
        // visit whose EVERY input is provably unchanged can exit after the
        // handful of writes that are genuinely per-frame, instead of walking the
        // remaining ~600 lines of derivations, probe gates and flag routing to
        // rewrite identical bytes. The recorded stage baseline this attacks:
        // tail 4.0 + xform 3.7 + surf-residual + flags + entry probes.
        //
        // WHAT PROVES "UNCHANGED": the same m_instStateKey digest that already
        // authorises the surf/tail skips (now also carrying cameraType), PLUS
        // m_fastDrawBits for the unkeyed flags-stage inputs, PLUS a byte-compare
        // of the incoming objectToWorld, PLUS the SpatialMap sync proof (the
        // m_stablePropId caveat recorded in rtx.conf: a propId that moved while
        // the transform did not leaves the map on a stale key -- caught here
        // because entry's mirror already updated m_stablePropId, so the
        // cacheHash==propId test fails and the full path re-files). Runtime
        // option flips are handled by fastPathOptionsStable() forcing one full
        // frame. Populations with genuine per-frame side effects (viewmodel
        // candidate list, player-model list, billboard/beam regeneration, decal
        // sort order, skinned prev-positions, RayPortal, the [ZigGun] probe tag)
        // never take it.
        //
        // WHAT THE FAST PATH STILL DOES: everything the guards' own notes name
        // as per-frame -- the buffer rebind (slice renaming), liveness/camera
        // (already done in entry), picking value, isInsideFrustum, prev-transform
        // advance, textureTransform/clipPlane copies (present in no key), the
        // m_isHidden promotions (entry resets the flag every frame, so they must
        // re-apply), and the NON-skippable event handlers (OMM bookkeeping).
        if (drawState.fastPathAllowed) {
          const uint32_t fastFid = m_device->getCurrentFrameId();
          uint32_t fastReject = kFastReasonCount;   // sentinel: eligible so far
          if (!instStateUnchanged) {
            fastReject = kFastKeyMiss;
          } else if (currentInstance.isCreatedThisFrame(fastFid)) {
            fastReject = kFastCreated;
          } else if (drawCall.cameraType == CameraType::ViewModel
                     || currentInstance.m_isPlayerModel
                     || currentInstance.m_isUnordered
                     || currentInstance.surface.alphaState.isDecal
                     || materialData.getType() == MaterialDataType::RayPortal
                     || blas.modifiedGeometryData.previousPositionBuffer.defined()
                     // the [ZigGun] diagnostic re-tags its instance every frame
                     || drawCall.getTransformData().vertexShaderHash == 0x292b6ba0d1854f28ull) {
            fastReject = kFastPop;
          } else if (drawState.fastDrawBits != currentInstance.m_fastDrawBits) {
            fastReject = kFastBitsMiss;
          } else {
            const Matrix4& fastO2W = split != nullptr
              ? split->objectToWorld
              : drawCall.getTransformData().objectToWorld;
            if (memcmp(fastO2W.data, currentInstance.surface.objectToWorld.data, sizeof(Matrix4)) != 0) {
              // Also naturally catches the WorldMatte background-offset case:
              // the stored matrix carries the offset, the incoming one does not.
              fastReject = kFastXformMiss;
            } else {
              const bool spatialInSync = currentInstance.m_isCreatedByRenderer
                || (currentInstance.m_stablePropId != 0
                      ? currentInstance.m_spatialCacheHash == static_cast<XXH64_hash_t>(currentInstance.m_stablePropId)
                      // keyHint.hash is XXH64 of THIS frame's composed
                      // first-instance matrix (findSimilarInstance's exact
                      // stage), so equality with the stored cache key proves the
                      // map entry matches this frame's key -- including the
                      // instancesToObject[0] contents the base-matrix memcmp
                      // above cannot see. Collision-bumped entries never match
                      // and simply stay on the full path.
                      : (keyHint.isUsable() && keyHint.hash == currentInstance.m_spatialCacheHash));
              if (!spatialInSync) {
                fastReject = kFastSpatialMiss;
              }
            }
          }

          if (fastReject == kFastReasonCount) {
            // Re-derive the per-frame m_isHidden promotions. Entry already reset
            // the flag from the Hidden category and applied the Sky-camera hide;
            // the material-driven promotions live in the (skipped) switch below
            // and only ever promote to true, so re-running them here is exactly
            // the slow path's net effect. All writes in this block are values the
            // slow path would derive identically, so falling through to the full
            // path afterwards (the mask case below) is harmless.
            currentInstance.m_materialType = materialData.getType();
            bool fastHidden = currentInstance.m_isHidden;
            if (currentInstance.m_materialType == MaterialDataType::Opaque) {
              const auto& fastOmd = materialData.getOpaqueMaterialData();
              if (fastOmd.getIsUnlitOutput()) {
                currentInstance.surface.isMatte = true;
              }
              const bool fastIgnoreAC = currentInstance.testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
              const bool fastTf2Fog = fastOmd.getTf2SkyboxFog();
              const bool fastFogEnabled = RtxOptions::enableTf2SkyboxCloudFog();
              currentInstance.surface.isTf2SkyboxFog = fastIgnoreAC && fastTf2Fog && fastFogEnabled;
              if (fastTf2Fog && !fastFogEnabled) {
                fastHidden = true;
              }
              if (drawCall.getMaterialData().sourcePsWritesCoverageMask) {
                fastHidden = true;
              }
              currentInstance.m_isSubsurface = fastOmd.getSubsurfaceDiffusionProfile();
            }
            if (fastHidden) {
              currentInstance.m_isHidden = true;
              currentInstance.m_vkInstance.mask = 0;
            } else if (currentInstance.m_vkInstance.mask == 0) {
              // Was hidden (or never masked), now visible: the mask bits must be
              // re-derived by the full routing chain -- retention cannot
              // reconstruct them from 0.
              fastReject = kFastMaskMiss;
            }
          }

          if (fastReject == kFastReasonCount) {
            // COMMIT the fast update: only the genuinely per-frame writes.
            // NV-DXVK [Phase2b]: on a worker the surface buffer binding is
            // deferred to the CS record step — the bake that decides which
            // bindless slots this geometry holds only runs there (see the spec
            // Sec 3.4).
            if (inShardedInstancePhase()) {
              t_shardPhase.currentOps->bindBuffers = true;
            } else {
              processInstanceBuffers(blas, drawState, currentInstance);
            }
            currentInstance.surface.hasMaterialChanged = false;
            currentInstance.surface.isInsideFrustum =
              RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? currentInstance.m_isInsideFrustum : true;
            currentInstance.surface.objectPickingValue = drawCall.drawCallID;
            // Advance transform history: cur is byte-identical to the incoming
            // matrix, so prev := cur is what move() would have produced, and the
            // vkInstance transform / normal matrix / mirror parity all retain.
            currentInstance.surface.prevObjectToWorld = currentInstance.surface.objectToWorld;
            currentInstance.surface.isStatic = true;   // !(xfChg || prevPos), both provably false
            currentInstance.surface.textureTransform = drawCall.getTransformData().textureTransform;
            currentInstance.surface.isClipPlaneEnabled = drawCall.getTransformData().enableClipPlane;
            currentInstance.surface.clipPlane = drawCall.getTransformData().clipPlane;
            currentInstance.m_billboardCount = 0;

            // Same per-handler granularity as the tail gate below: only handlers
            // that opted in to being skippable are dropped; OMM's stays live.
            // NV-DXVK [perf] 2026-08-08 (handoff d §3, fast-path floor): OMM's
            // non-skippable handler reduced, for THIS population (binding
            // unchanged + frameAge != 0 -- kFastCreated rejected new
            // instances above), to `if (needsToCalculateNumTexelsPerMicro-
            // Triangle) calculate()`. That flag lives on the RtInstance, so
            // handlers that declared the skippableWhenNoPendingOmmWork
            // contract are skipped on a member load instead of paying the
            // std::function dispatch + profile zone ~14k times a frame.
            // The flag-set case still dispatches, so pending OMM work is
            // picked up on exactly the frame it would have been before.
            {
              const bool ommPendingWork = currentInstance
                .getOpacityMicromapInstanceData().hasPendingNumTexelsCalculation();
              // NV-DXVK [Phase2b]: the surviving handlers on this path are the
              // non-skippable ones — OMM. Its callback reads buffer contents
              // (texel calc) and writes unlocked OMM maps, both CS-domain, so on
              // a worker it is deferred to the CS record step with these args.
              if (inShardedInstancePhase()) {
                if (ommPendingWork) {
                  t_shardPhase.currentOps->omm = true;
                  t_shardPhase.currentOps->evHasTransformChanged = false;
                  t_shardPhase.currentOps->evHasPreviousPositions = false;
                  t_shardPhase.currentOps->evIsFirstUpdateThisFrame = true;
                } else {
                  s_fastInstOmmSkips.fetch_add(1, std::memory_order_relaxed);
                }
              } else {
              for (auto& event : m_eventHandlers) {
                if (event.skippableWhenBindingUnchanged) {
                  continue;
                }
                if (event.skippableWhenNoPendingOmmWork && !ommPendingWork) {
                  s_fastInstOmmSkips.fetch_add(1, std::memory_order_relaxed);
                  continue;
                }
                event.onInstanceUpdatedCallback(currentInstance, drawCall, materialData,
                                                /*hasTransformChanged*/ false,
                                                /*hasPreviousPositions*/ false,
                                                /*isFirstUpdateThisFrame*/ true);
              }
              }
            }

            // NV-DXVK [Perf.UpdInst] 2026-08-09 -- CLOSE THE FAST-PATH SPAN.
            // Until now mark(0) closed `entry` above and this path returned below
            // without ever calling mark(1), so everything in between -- the
            // fast-path eligibility checks and the OMM event-handler dispatch loop
            // immediately above -- was timed into NO STAGE AT ALL. That is not a
            // rare corner: [Perf.FastInst] measures fast=14,358/frame against
            // [Perf.UpdInst] instPerFrame=15,529, so 92.5% of every instance's
            // work was unmeasured. It is the block [ProcDCS] instMs could see and
            // the nine stages could not, and it is why the two instruments on
            // updateInstance disagreed (SceneObj update=12.86 vs the stage sum
            // 4.47). Marked BEFORE the probe bookkeeping below so the probe's own
            // atomics and emit are not billed to game work.
            uiSplit.mark(9);
            uiSplit.surfSkipped();
            uiSplit.tailSkipped();
            uiSplit.counts(/*first*/ true, /*xfChg*/ false, /*matChg*/ false,
                           /*prevPos*/ false, /*isStatic*/ true);
            s_fastInstHit.fetch_add(1, std::memory_order_relaxed);
            fastInstEmit(fastFid);
            return;
          }

          s_fastInstSlow[fastReject].fetch_add(1, std::memory_order_relaxed);
          fastInstEmit(fastFid);
        } else {
          s_fastInstSlow[kFastNotAllowed].fetch_add(1, std::memory_order_relaxed);
        }

        // NEVER SKIPPED -- 2026-08-07, this froze the game when it was inside the
        // else below. processInstanceBuffers is not a read the key covers: it
        // REBINDS the instance to blas's buffers, and buffer identity is not an
        // input to computeDrawStateKey. Geometry hashes identically while living
        // in a different allocation every frame -- Map(WRITE_DISCARD) renames the
        // slice on ~94% of draws (see HANDOFF_T31_RESOLVED sec 2). Skipping it
        // left instances bound to the previous frame's freed slices.
        // The "every skipped read is a key input" argument does not reach this
        // call, and it must stay outside the guard.
        // NV-DXVK [Phase2b]: NOT skipped — DEFERRED to the CS record step, which
        // rebinds after this frame's bake has settled which slots the geometry
        // holds. The 2026-08-07 freeze was a skip with NO later rebind; the
        // deferred rebind still happens every frame, before prepareSceneData
        // consumes the surface.
        if (inShardedInstancePhase()) {
          t_shardPhase.currentOps->bindBuffers = true;
        } else {
          processInstanceBuffers(blas, drawState, currentInstance);
        }

        currentInstance.m_materialType = materialData.getType();

        if (instStateUnchanged) {
          // Edge signal, not state -- see the note above. Must not latch.
          currentInstance.surface.hasMaterialChanged = false;
        } else {

          const XXH64_hash_t materialInstanceHash = materialData.getHash();
          currentInstance.m_materialDataHash = drawCall.getMaterialData().getHash();
          currentInstance.surface.hasMaterialChanged = currentInstance.m_materialHash != kEmptyHash && currentInstance.m_materialHash != materialInstanceHash;
          currentInstance.m_materialHash = materialInstanceHash;

          currentInstance.m_texcoordHash = drawCall.getGeometryData().hashes[HashComponents::VertexTexcoord];
          currentInstance.m_indexHash = drawCall.getGeometryData().hashes[HashComponents::Indices];
        }

        // NV-DXVK [perf] handoff v5 sec 4c: THE `surf` SKIP.
        //
        // Everything from here to the vsDebugId assignment below derives purely
        // from (drawCall, materialData, m_categoryFlags, alphaState) and is
        // rewritten byte-identically every frame -- matChg=0% in every capture ever
        // taken. computeSurfStateKey() digests exactly those inputs; see the note
        // above it for what each entry is justified against, and in particular for
        // which fields the two material hashes do NOT cover (that was the sec 0a
        // defect and it is the thing to re-check if a surface ever renders with
        // another draw's state).
        //
        // Reads deliberately left OUTSIDE the guard because they genuinely change
        // per frame, so a skip must not freeze them:
        //   surface.objectPickingValue <- drawCall.drawCallID, a per-frame counter
        //   surface.isInsideFrustum    <- m_isInsideFrustum, moves with the camera
        // and the two m_isHidden promotions in the Opaque case further down, which
        // `entry` resets from the Hidden category flag every frame and `flags`
        // consumes. Those stay outside entirely.
        //
        // isEmissive/isMatte are RESETS paired with promotions that live outside
        // the guard (the switch below), so skipping is consistent in both
        // directions: the promotion re-applies every frame, and the reset re-runs
        // whenever the material identity that drives the promotion changes.
        //
        // MEASURED HISTORY, so the shape is not re-litigated. v1 of this guard was
        // a NET LOSS: surf 0.211 baseline -> 0.250 us/inst at surfSkip=99%. v2
        // shrank the key from 152 bytes and deleted four dead variant reads from
        // the switch below, reaching 0.233 -- better, still a loss. The cause was
        // never the hashing; it is that this block's ~25 writes go SEQUENTIALLY
        // into one hot RtSurface while any key must READ from four scattered
        // objects (LegacyMaterialData, DrawCallTransforms, the MaterialData
        // variant, GeometryHashes). A detector that touches more memory than the
        // work it skips cannot win on its own.
        //
        // v3 stops trying to make it win on its own: the key is SHARED with the
        // event-fanout gate at the bottom of this function, which saves ~50 ns of
        // real work per instance and was already paying for a near-identical
        // gather of its own. One digest, one scatter, two skips -- surf's share of
        // the detector is now marginal rather than the whole cost.
        // NV-DXVK [perf] 2026-08-07: the gather that used to sit here now runs
        // once per DRAW (computeDrawScopedState), and what is left per instance
        // is a 16-byte hash of two registers. See computeDrawStateKey for why the
        // split is sound -- every input except m_categoryFlags is draw-, material-
        // or frame-scoped, and m_categoryFlags is exactly what gets mixed in here.
        // NV-DXVK [perf] 2026-08-07 (v4): the key computation and the
        // instStateUnchanged decision that used to sit HERE are now hoisted to
        // the top of this `if (isFirstUpdateThisFrame)` block, above the
        // material/geometry-hash preamble they were always meant to guard. See
        // the long note up there for why. instKeyEligible / instStateKey /
        // instStateUnchanged are all still in scope at this point -- the hoist
        // moved them within the same block, it did not narrow their scope.
        //
        // Decided ONCE above, consumed twice: by this block and by the
        // event-fanout gate at the bottom of the function. The key covers the
        // union of both blocks' inputs, so one match authorises both skips --
        // and the fanout adds its own escapes (transform change, billboards,
        // RayPortal) on top, which are only known later, after the xform stage
        // has run.
        if (instStateUnchanged) {
          uiSplit.surfSkipped();
        } else {

        // Surface meta data
        currentInstance.surface.isEmissive = false;
        currentInstance.surface.isMatte = false;
        currentInstance.surface.textureColorArg1Source = drawCall.getMaterialData().textureColorArg1Source;
        currentInstance.surface.textureColorArg2Source = drawCall.getMaterialData().textureColorArg2Source;
        currentInstance.surface.textureColorOperation = drawCall.getMaterialData().textureColorOperation;
        currentInstance.surface.textureAlphaArg1Source = drawCall.getMaterialData().textureAlphaArg1Source;
        currentInstance.surface.textureAlphaArg2Source = drawCall.getMaterialData().textureAlphaArg2Source;
        currentInstance.surface.textureAlphaOperation = drawCall.getMaterialData().textureAlphaOperation;
        currentInstance.surface.texgenMode = drawCall.getTransformData().texgenMode; // NOTE: Make it material data...
        // NV-DXVK: texcoordEncoding intentionally NOT set from drawCall
        // here. processInstanceBuffers() above derives it from the BLAS's
        // texcoord buffer.vertexFormat() — the only source consistent
        // with the actual buffer bytes. See the SurfaceEncMismatch fix
        // log in writeGPUData for the bug history. Per-draw assignment
        // here used to clobber the correct value when the same BLAS was
        // referenced by mixed-VS-class draws.
        currentInstance.surface.tFactor = drawCall.getMaterialData().tFactor;
        currentInstance.surface.alphaState = alphaState;

        // NV-DXVK [MtnAlphaState]: confirm the RESOLVED alpha state (post calculateAlphaState,
        // which can override the raw compare op) for the two SKY_MOUNTAIN VS in the black-tops
        // investigation. Prediction: culprit 0x28f7ffa9 has alphaTestType != kAlways(7) =>
        // isFullyOpaque=0 (goes through the binary cutout path -> can flip the integration
        // surface off -> demodulate-zero -> black); clean 0x29146e1d is kAlways/fullyOpaque.
        // Targeted (2 hashes) + first-update-only so it never spams.
        if (RtxOptions::logSurfaceCoverage()) {
          const uint64_t vsH = (uint64_t) drawCall.getTransformData().vertexShaderHash;
          if (vsH == 0x28f7ffa90d189017ull || vsH == 0x29146e1dd50b0314ull) {
            Logger::info(str::format(
              "[MtnAlphaState] vs=0x", std::hex, vsH, std::dec,
              " alphaTestType=", (uint32_t) alphaState.alphaTestType,
              " alphaRef=", (uint32_t) alphaState.alphaTestReferenceValue,
              " blendDisabled=", alphaState.isBlendingDisabled ? 1 : 0,
              " fullyOpaque=", alphaState.isFullyOpaque ? 1 : 0,
              " invertedBlend=", alphaState.invertedBlend ? 1 : 0,
              " emissiveBlend=", alphaState.emissiveBlend ? 1 : 0,
              " isParticle=", alphaState.isParticle ? 1 : 0,
              " isDecal=", alphaState.isDecal ? 1 : 0));
          }
        }
        currentInstance.surface.isAnimatedWater = currentInstance.testCategoryFlags(InstanceCategories::AnimatedWater);
        currentInstance.surface.associatedGeometryHash = drawCall.getHash(RtxOptions::geometryAssetHashRule());
        currentInstance.surface.isTextureFactorBlend = drawCall.getMaterialData().isTextureFactorBlend;
        currentInstance.surface.isVertexColorBakedLighting = drawCall.getMaterialData().isVertexColorBakedLighting;
        currentInstance.surface.isMotionBlurMaskOut = currentInstance.testCategoryFlags(InstanceCategories::IgnoreMotionBlur);
        currentInstance.surface.ignoreTransparencyLayer = currentInstance.testCategoryFlags(InstanceCategories::IgnoreTransparencyLayer);

        // Note: Skip the spritesheet adjustment logic in the surface interaction when using Ray Portal materials as this logic
        // is done later in the Surface Material Interaction (and doing it in both places will just double up the animation).
        currentInstance.surface.skipSurfaceInteractionSpritesheetAdjustment = (currentInstance.m_materialType == MaterialDataType::RayPortal);

        currentInstance.surface.blendModeState = drawCall.getMaterialData().blendMode;

        if (drawCall.isEye()) {
          // assume that the texture transform has eye parameters encoded
          const Matrix4& texTransform = drawCall.getTransformData().textureTransform;
          RtEyeParams eyeParams{};
          eyeParams.eyeballOrigin = Vector3{ texTransform.data[0].w, texTransform.data[1].w, texTransform.data[2].w };
          eyeParams.eyeRightU = Vector3{ texTransform.data[0].x, texTransform.data[1].x, texTransform.data[2].x };
          eyeParams.eyeUpV = Vector3{ texTransform.data[0].y, texTransform.data[1].y, texTransform.data[2].y };
          currentInstance.surface.eyeParams = eyeParams;
        }

        materialData.getSpriteSheetData(currentInstance.surface.spriteSheetRows, currentInstance.surface.spriteSheetCols, currentInstance.surface.spriteSheetFPS);
        currentInstance.m_isAnimated = currentInstance.surface.spriteSheetFPS != 0;
        // NV-DXVK [VsColor]: stamp the per-pixel vertex-shader identity. Taken
        // from the DRAW (not the linked BlasEntry): when several draws share a
        // BlasEntry the blas reports only whichever VS created it, which is the
        // exact aliasing that made every VS-gated probe in this investigation a
        // blind spot. The draw's own hash is what actually produced the pixel.
        // NV-DXVK [perf]: skipped entirely unless something reads it -- see
        // vsDebugIdIsConsumed(). This is once per INSTANCE, not once per draw.
        currentInstance.surface.vsDebugId =
          vsDebugIdIsConsumed()
            ? acquireVsDebugId(drawCall.getTransformData().vertexShaderHash)
            : uint16_t(0);

        }   // NV-DXVK [perf] sec 4c: end of the surf-state guard

        // NV-DXVK [perf] sec 4c: PER-FRAME, NEVER SKIPPED. Hoisted out of the
        // guard because their sources change every frame and a skip would freeze
        // them -- drawCallID is a per-frame draw counter (D3D11Rtx::m_drawCallID,
        // reset each frame) and m_isInsideFrustum tracks the camera. Neither is in
        // computeSurfStateKey by construction: keying on drawCallID would make the
        // key differ every frame and the guard would never fire.
        currentInstance.surface.isInsideFrustum = RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? currentInstance.m_isInsideFrustum : true;
        currentInstance.surface.objectPickingValue = drawCall.drawCallID;

        // Note: Extract spritesheet information from the associated material data as it ends up stored in the Surface
        // not in the Surface Material like most material information.
        switch (materialData.getType()) {
        case MaterialDataType::Opaque:
        {
          // NV-DXVK [perf]: FOUR DEAD VARIANT ACCESSES REMOVED HERE.
          //
          // This case used to open by reading getSpriteSheetRows/Cols/FPS into
          // locals and getUseLegacyAlphaState() into another -- none of which was
          // ever read again. The spritesheet values were already written straight
          // into the surface by materialData.getSpriteSheetData() above (which
          // runs its own switch over the same type), so the material was being
          // asked for them twice and one copy thrown away; and the only
          // useLegacyAlphaState that matters is calculateAlphaState's own local.
          //
          // Each of those was a std::get<OpaqueMaterialData> into a 184-byte
          // variant alternative plus a getter, once per Opaque instance per frame
          // (~15,500). Deleting beats guarding: no key, no detector cost, no
          // correctness surface at all.

          // NV-DXVK: TF2 worldspace VGUI/HUD shaders are inherently unlit —
          // the PS writes the final composed UI color directly. Setting
          // surface.isMatte=true makes the slang shader zero out albedo
          // and baseReflectivity, leaving only the emissive contribution
          // (the picked color texture forwarded by LegacyMaterialData::
          // as<OpaqueMaterialData>()) so the UI texture is rendered as-is
          // without world lighting. Detection happens upstream in
          // d3d11_rtx.cpp::FillMaterialData via PS RDEF resource names.
          if (materialData.getOpaqueMaterialData().getIsUnlitOutput()) {
            currentInstance.surface.isMatte = true;
          }

          // NV-DXVK: TF2 3D-skybox cloud billboard handling. A cloud
          // billboard = an instance in the 3D skybox (IgnoreAntiCulling,
          // applied by SetSkyCategoryFromCb2) whose material is the
          // fog-synthesizing premultiplied uber shader (Tf2SkyboxFog). The
          // material flag alone is too broad — that PS family is also used
          // by playable-world smoke / effects / props — so it is ANDed with
          // the 3D-skybox category.
          //
          // Switched by rtx.enableTf2SkyboxCloudFog:
          //  - enabled:  flag Surface::isTf2SkyboxFog so the opaque material
          //              interaction reconstructs the game's fog-blend
          //              colour (the alpha-blend unlit-cloud composite keys
          //              off the same flag).
          //  - disabled: the cloud texture is a near-black coverage map with
          //              no colour of its own, so leaving the billboards in
          //              renders them solid black. Hide them entirely
          //              instead — m_isHidden forces the instance mask to 0
          //              (see below), so the rays miss them and the sky
          //              shows through cleanly.
          const bool ignoreAntiCullCat =
            currentInstance.testCategoryFlags(InstanceCategories::IgnoreAntiCulling);
          const bool matTf2Fog =
            materialData.getOpaqueMaterialData().getTf2SkyboxFog();
          const bool isTf2Cloud = ignoreAntiCullCat && matTf2Fog;
          const bool tf2CloudFogEnabled = RtxOptions::enableTf2SkyboxCloudFog();
          currentInstance.surface.isTf2SkyboxFog = isTf2Cloud && tf2CloudFogEnabled;
          // [TF2 fog-pipeline hide] When the user has fog reconstruction
          // disabled, we hide every surface tagged matTf2Fog — not just
          // the sub-view ones the original isTf2Cloud check caught.
          // Bisect data: with isTf2Cloud-only the hide-list correctly
          // dropped the four sub-view cloud VSes (0x2904, 0x290deec3,
          // 0x296dc3ae, 0x2a904f3d) but three MAIN-WORLD matTf2Fog
          // surfaces (0x29a262d2, 0x29566a60, 0x28d6a5dc) were left
          // visible — they render through the premult-encode bypass
          // without the fog reconstruction the original PS expected,
          // producing the dark sky-region corruption. Same intent as
          // the cloud hide: "we can't reconstruct the fog math, so
          // don't render the fog-dependent surface". The
          // IgnoreAntiCulling restriction in the original gate was
          // incidental to the cloud-billboard case and excluded
          // legitimate fog-pipeline targets that happen to live in
          // main-world coords.
          if (matTf2Fog && !tf2CloudFogEnabled) {
            currentInstance.m_isHidden = true;
          }

          // NV-DXVK [FogHideProbe]: the visible garbage is ~29 stacked VS
          // 0x29566a60 draws at one origin, flickering. [Tf2CloudClass] is
          // one-shot-per-VS so it can't reveal whether EVERY stacked draw gets
          // matTf2Fog/hidden. Log each distinct (ignoreAntiCull,matTf2Fog,
          // m_isHidden) outcome for this VS — if any line shows matTf2Fog=0 (or
          // m_isHidden=0), those draws are NOT hidden → they render and flicker.
          {
            const uint64_t vsHfh = uint64_t(drawCall.getTransformData().vertexShaderHash);
            if (vsHfh == 0x29566a60d473af50ull) {
              const uint32_t key = (ignoreAntiCullCat ? 4u : 0u)
                                 | (matTf2Fog ? 2u : 0u)
                                 | (currentInstance.m_isHidden ? 1u : 0u);
              // NV-DXVK [perf]: the key is three bits, so eight values exhaust
              // this probe -- and the log holds two lines. Everything after that
              // was a process-global mutex acquired per instance of this VS per
              // frame to re-discover a combination already recorded. A
              // thread-local bitmask fronts the shared set: steady state is a
              // shift and a test. Same idiom as [FlipNormalDiag] above -- the
              // thread-local only ever SUPPRESSES work, and the shared set stays
              // the authority for "first globally", so the line is still emitted
              // exactly once across all threads rather than once per thread.
              static thread_local uint8_t tFhSeen = 0;
              const uint8_t fhBit = static_cast<uint8_t>(1u << key);
              bool firstFh = false;
              if ((tFhSeen & fhBit) == 0) {
                tFhSeen |= fhBit;
                static std::mutex sFhMu;
                static thread_local std::unordered_set<uint32_t> sFhSeen;
                std::lock_guard<std::mutex> g(sFhMu);
                if (sFhSeen.insert(key).second) firstFh = true;
              }
              if (firstFh)
                Logger::warn(str::format(
                  "[FogHideProbe] VS=0x29566a60 matTf2Fog=", (matTf2Fog ? 1 : 0),
                  " m_isHidden=", (currentInstance.m_isHidden ? 1 : 0),
                  " ignoreAntiCull=", (ignoreAntiCullCat ? 1 : 0),
                  " tf2CloudFogEnabled=", (tf2CloudFogEnabled ? 1 : 0)));
            }
          }

          // NV-DXVK [SV_Coverage hide]: PSes that write SV_Coverage
          // (oMask) implement smooth alpha via MSAA sub-pixel sample
          // masking. The path tracer can't honor oMask — full RGBA
          // hits every pixel, producing the visible BOXY corruption
          // in TF2's 3D-skybox (VS_95da0b01 + FS_e508ad41 = the
          // single-tri sky-noise overlay flooding the sky speckling
          // pattern the user reported). Hide unconditionally; there
          // is no path-tracer-compatible rendering of these draws.
          // Flag set in d3d11_rtx.cpp::FillMaterialData by walking
          // the PS OSGN for systemValueType == D3D_NAME_COVERAGE.
          if (drawCall.getMaterialData().sourcePsWritesCoverageMask) {
            currentInstance.m_isHidden = true;
          }

          // NV-DXVK [Tf2CloudClass]: one-shot per VS hash, log the
          // classification result so we can debug "cloud not being
          // tagged Tf2Cloud" vs "tagged but rendering wrong" without
          // recompile. Caps at 32 distinct VSes per session.
          // Gameplay-gated via captureCount > 16.
          {
            if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u
                && (ignoreAntiCullCat || matTf2Fog)) {
              const uint64_t vsHashCc = uint64_t(drawCall.getTransformData().vertexShaderHash);
              // NV-DXVK [perf]: this probe caps at 32 distinct VS hashes and the
              // log shows only 14 ever appear -- so the size() < 32 check NEVER
              // stops handing out the mutex, and every sub-view / fog instance
              // took a process-global lock every frame for a line emitted once
              // per VS long ago. Unlike a saturating budget there is nothing to
              // latch on, so front it with a thread-local set: the steady state
              // is one hash lookup and no lock at all. Same idiom as
              // [FlipNormalDiag] above -- the thread-local only ever SUPPRESSES
              // work, never grants permission to log, so the shared set remains
              // the authority for "first globally" and the line is still emitted
              // exactly once across all threads.
              static thread_local std::unordered_set<uint64_t> tTf2CcSeen;
              bool firstCc = false;
              if (tTf2CcSeen.insert(vsHashCc).second) {
                static std::mutex sTf2CcMu;
                static thread_local std::unordered_set<uint64_t> sTf2CcSeen;
                std::lock_guard<std::mutex> g(sTf2CcMu);
                if (sTf2CcSeen.size() < 32u
                    && sTf2CcSeen.insert(vsHashCc).second) {
                  firstCc = true;
                }
              }
              if (firstCc) {
                Logger::warn(str::format(
                  "[Tf2CloudClass] vsXxh=0x", std::hex, vsHashCc, std::dec,
                  " ignoreAntiCull=", (ignoreAntiCullCat ? 1 : 0),
                  " matTf2Fog=", (matTf2Fog ? 1 : 0),
                  " isTf2Cloud=", (isTf2Cloud ? 1 : 0),
                  " tf2CloudFogEnabled=", (tf2CloudFogEnabled ? 1 : 0),
                  " → surface.isTf2SkyboxFog=", (currentInstance.surface.isTf2SkyboxFog ? 1 : 0),
                  " m_isHidden=", (currentInstance.m_isHidden ? 1 : 0)));
              }
            }
          }

          // NV-DXVK: emission is now driven by the PS's own CBuffer signal
          // (CBufUberStatic.c_emissiveTint marked D3D_SVF_USED, read in
          // D3D11Rtx::FillMaterialData and forwarded through LegacyMaterial
          // Data::as<OpaqueMaterialData>()). The previous heuristics that
          // lived here — promoting on `m_isWorldSpaceUI` or on detected
          // emissive-blend Vulkan blend states — were both wrong-by-
          // construction in the absence of the ground-truth signal: the
          // blend-state path mis-classified TF2 refract/water/decal layers
          // as light sources (24/32 distinct materials in observed logs),
          // producing the "lighting bouncing off surfaces" bug.
          //
          // Worldspace UI is left unhandled here intentionally — TF2 doesn't
          // categorise any draws as InstanceCategories::WorldUI in this
          // fork (verified via [EmissivePromote.WorldUI] zero hits across
          // multiple gameplay sessions). If a future map does, the right
          // place to resurrect that promotion is alongside the PS-CB read,
          // not as a pre-material-data mutation. Log a one-shot if it
          // ever fires so we notice.
          if (currentInstance.m_isWorldSpaceUI) {
            const XXH64_hash_t matHash = drawCall.getMaterialData().getHash();
            static thread_local std::unordered_set<XXH64_hash_t> sUiSeenWarn;
            if (sUiSeenWarn.insert(matHash).second) {
              Logger::warn(str::format(
                "[EmissivePromote.WorldUI.Unhandled] matHash=0x",
                std::hex, matHash, std::dec,
                " — TF2 fork no longer auto-promotes WorldUI to emissive;"
                " if this material should glow, route via PS CBuffer signal"
                " (see d3d11_rtx.cpp::FillMaterialData)."));
            }
          }

          currentInstance.m_isSubsurface = materialData.getOpaqueMaterialData().getSubsurfaceDiffusionProfile();

          break;
        }
        case MaterialDataType::Translucent:
          // NV-DXVK [perf]: same three dead reads as the Opaque case above, same
          // deletion. Kept as an explicit empty case rather than folded away
          // because /we4062 makes an unhandled enumerator a build error, and
          // because "this type needs nothing here" is worth stating.
          break;
        case MaterialDataType::RayPortal:
          // NV-DXVK [perf]: dead reads removed, see the Opaque case above.
          break;
        case MaterialDataType::Count:
        case MaterialDataType::Invalid:
          assert(0);
          break;
        }
      }

      // NV-DXVK [Perf.UpdInst]: end of `surf`. NOTE THE SCOPE -- unlike marks 0
      // and 2-8 this one is INSIDE `if (isFirstUpdateThisFrame ||
      // overridePreviousCameraUpdate)` (~:6588), so on the ~1% of instances where
      // that is false it never fires and reachPct=surf reads ~1 point below its
      // neighbours.
      //
      // THAT IS CORRECT, NOT A LEAK, and it was checked rather than assumed:
      // between mark(0) (~:6554) and the `if` there is only comment, so when the
      // block is skipped the mark(0)->mark(2) span covers no executable code and
      // `xform` absorbs ~0. surf genuinely did not run; a zero reach is the
      // honest report. Do NOT "fix" this by hoisting the mark past the block's
      // closing brace (~:7536) -- that would move everything between this mark
      // and that brace out of `xform` and
      // into `surf` and silently redefine both stages.
      uiSplit.mark(1);

      // Update transform
      {
        // Heuristic for MS5 - motion vectors on translucent surfaces cannot be trusted.  This will help with IQ, but need a longer term solution [TREX-634]
        const bool isMotionUnstable = currentInstance.m_materialType == MaterialDataType::Translucent
                                   || currentInstance.testCategoryFlags(InstanceCategories::Particle)
                                   || currentInstance.testCategoryFlags(InstanceCategories::WorldUI);

        // NV-DXVK [Phase2b]: on a worker the bake has not run yet (it records on
        // CS after the flush), so previousPositionBuffer still holds LAST frame's
        // state. Predict this frame's value from the geometry decision the shard
        // already made: kUpdateBVH always produces a previous-position buffer,
        // KBuildBVH always resets it, kUpdateInstance leaves it as-is. This is
        // exactly what processGeometryInfo's bake switch does to the field.
        bool prevPosDefinedAfterBake = blas.modifiedGeometryData.previousPositionBuffer.defined();
        if (inShardedInstancePhase() && t_shardPhase.info->geomResult >= 0) {
          const auto r = static_cast<SceneManager::ObjectCacheState>(t_shardPhase.info->geomResult);
          if (r == SceneManager::ObjectCacheState::kUpdateBVH) {
            prevPosDefinedAfterBake = true;
          } else if (r == SceneManager::ObjectCacheState::KBuildBVH) {
            prevPosDefinedAfterBake = false;
          }
        }
        hasPreviousPositions = prevPosDefinedAfterBake && !isMotionUnstable;
        const bool isFirstUpdateAfterCreation = currentInstance.isCreatedThisFrame(m_device->getCurrentFrameId()) && isFirstUpdateThisFrame;

        // Note: objectToView is aliased on updates, since findSimilarInstance() doesn't discern it
        // NV-DXVK [fanout split]: a split placement carries its own composed
        // transform (drawO2W * instancesToObject[i]); the draw call itself only
        // holds the batch-wide identity matrix, which would place every prop of
        // the batch at the world origin.
        Matrix4 objectToWorld = split != nullptr
          ? split->objectToWorld
          : drawCall.getTransformData().objectToWorld;

        // Hack for TREX-2272. In Portal, in the GLaDOS chamber, the monitors show a countdown timer with background, and the digits and background are coplanar.
        // We cannot reliably determine the digits material because it's a dynamic texture rendered by vgui that contains all kinds of UI things.
        // So instead of offsetting the digits or making them live in unordered TLAS (either of which would solve the problem), we offset the screen background backwards.
        const float worldSpaceUiBackgroundOffset = RtxOptions::worldSpaceUiBackgroundOffset();
        if (worldSpaceUiBackgroundOffset != 0.f && currentInstance.testCategoryFlags(InstanceCategories::WorldMatte)) {
          objectToWorld[3] += objectToWorld[2] * worldSpaceUiBackgroundOffset;
        }

        // Update the transform based on what state we're in
        int mtnMovePath = -1;  // 0=teleport(prev=cur,zero motion) 1=move(prev=old,correct) 2=moveAgain(prev STALE)
        if (isFirstUpdateAfterCreation) {
          hasTransformChanged = currentInstance.teleport(objectToWorld);
          mtnMovePath = 0;
        } else if (isFirstUpdateThisFrame) {
          // NV-DXVK [perf] handoff v7 sec 4a: keyHint carries this frame's
          // already-computed spatial key. teleport() above is deliberately left
          // out -- [MapGate] puts it at mapWrInsert=4/frame, so there is nothing
          // there to save, and insert()'s collision-bump loop would need the same
          // reasoning applied a second time for no measurable return.
          hasTransformChanged = currentInstance.move(objectToWorld, keyHint);
          mtnMovePath = 1;
        } else {
          hasTransformChanged = currentInstance.moveAgain(objectToWorld, keyHint);
          mtnMovePath = 2;
        }

        // NV-DXVK [MvRaw] 2026-08-03: scene-wide RAW motion-vector record.
        //
        // SYMPTOM: a split-second blur flash where the motion vectors say the
        // geometry moved a long way, while the geometry visibly does not move.
        // Absent at cd50c7c4, present at 47c65fb5. The camera is NOT a suspect:
        // rtx_camera.cpp has no prev-matrix or frame-phase change anywhere in
        // cd50c7c4..47c65fb5, and engineHookMainCameraFrameDelay predates both.
        //
        // WHAT A MOTION VECTOR IS HERE. The GPU derives it from
        // surface.objectToWorld vs surface.prevObjectToWorld. Those are set by
        // exactly the three calls above:
        //   path 0 teleport()  prev := cur          -> zero motion
        //   path 1 move()      prev := THIS INSTANCE's previous cur -> correct
        //   path 2 moveAgain() prev untouched       -> prev is now two-or-more
        //                                              updates old
        // So a large motion vector on stationary geometry means prev does not
        // hold this object's own last position. There are only two ways that
        // happens: the instance was paired to a DIFFERENT placement (prev is
        // some other prop's position), or the instance was not updated for a
        // frame or more and prev went stale.
        //
        // WHY THE PAIRING FIELDS ARE LOGGED ALONGSIDE. 612ff00d changed
        // DrawCallCache::get to rank candidates by frameLastTouched FIRST and
        // use the old distance score only as a tie-break, and re-seeded
        // bestScore from numeric_limits<float>::min() to lowest(). Both changes
        // make a draw accept a BlasEntry it would previously have rejected -
        // including one whose instances stand somewhere else entirely (the same
        // capture measured 507 units of drift between twins in one bucket). If
        // that is the cause, the flash frames will carry pairKind=2 and/or a
        // stale pairPrevTouched, and pml will be large. If it is not, they will
        // not, and the pairing is exonerated without a rebuild.
        //
        // READ IT LIKE THIS, per row:
        //   mv   = |curT - prevT|      what the motion vector actually encodes
        //   rep  = |curT - lastCurT|   how far this id genuinely moved since
        //                              its last recorded update
        //   pml  = |prevT - lastCurT|  prev-matches-last. ~0 means prev is
        //                              correctly this object's own last
        //                              position; large means prev came from
        //                              somewhere else.
        //   fsl  = frames since this id was last seen (1 = every frame,
        //                              >1 = it missed frames, so prev is stale)
        // THE FLASH IS: mv large while rep ~ 0. Then pml decides which cause -
        // large pml with fsl==1 is a wrong pairing, large pml with fsl>1 is a
        // stale/skipped instance. If mv ~ rep on every row the motion vectors
        // are honest and the artifact is downstream of this file.
        //
        // Deliberately RAW and UNCAPPED: every instance, every update path,
        // every frame, no thresholds, no averaging, no VS allowlist. The
        // artifact lasts a split second, so any sampling or per-frame cap can
        // miss the one frame that matters, and a per-frame aggregate would
        // average a single bad instance away to nothing. Volume is the price of
        // catching it; narrow by id or vs AFTER a flash frame is identified.
        // Off by default - costs one bool load per draw when disabled.
        // NV-DXVK [MvRaw aiming]: rtx.motionVectorRawVsHashes narrows this to a
        // chosen set of shaders. Empty (the default) keeps the original
        // scene-wide behaviour exactly. Checked LAST so the common disabled case
        // still costs only the bool load, and the set lookup is skipped entirely
        // whenever the set is empty.
        if (RtxOptions::logMotionVectorRaw()
            && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u
            && (RtxOptions::motionVectorRawVsHashes().empty()
                || lookupHash(RtxOptions::motionVectorRawVsHashes(),
                              drawCall.getTransformData().vertexShaderHash))) {
          const uint32_t curFrameMv = m_device->getCurrentFrameId();
          const uint64_t idMv       = currentInstance.getId();
          const Vector4  cTv        = currentInstance.surface.objectToWorld[3];
          const Vector4  pTv        = currentInstance.surface.prevObjectToWorld[3];

          auto dist3 = [](float ax, float ay, float az,
                          float bx, float by, float bz) -> float {
            const float dx = ax - bx, dy = ay - by, dz = az - bz;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
          };

          const float mvLen = dist3(float(cTv.x), float(cTv.y), float(cTv.z),
                                    float(pTv.x), float(pTv.y), float(pTv.z));

          // Keyed by stable instance id, not by pointer: RtInstances are pooled
          // and reused, so an address can be two different objects over a run.
          // Separate from [MtnMotion]'s map on purpose - sharing it would let
          // whichever probe ran first define "last frame" for the other.
          struct MvSeen { uint32_t frame; float x, y, z; };
          static std::mutex sMvMtx;
          static thread_local std::unordered_map<uint64_t, MvSeen> sMvSeen;
          int   fslMv = -1;
          float repMv = -1.0f, pmlMv = -1.0f;
          {
            std::lock_guard<std::mutex> lk(sMvMtx);
            auto it = sMvSeen.find(idMv);
            if (it != sMvSeen.end()) {
              const MvSeen& ls = it->second;
              fslMv = int(curFrameMv) - int(ls.frame);
              repMv = dist3(float(cTv.x), float(cTv.y), float(cTv.z), ls.x, ls.y, ls.z);
              pmlMv = dist3(float(pTv.x), float(pTv.y), float(pTv.z), ls.x, ls.y, ls.z);
            }
            // Record only on the first update of a frame. moveAgain() draws are
            // additional submissions of the SAME instance within one frame; if
            // they overwrote the baseline, fsl would read 0 and rep would
            // collapse to the intra-frame delta, hiding the frame-to-frame
            // motion this probe exists to measure.
            if (isFirstUpdateThisFrame) {
              sMvSeen[idMv] = MvSeen { curFrameMv,
                                       float(cTv.x), float(cTv.y), float(cTv.z) };
            }
          }

          // Pairing decision that produced this entry, read only when it was
          // stamped THIS frame (see BlasEntry::lastPairFrame in rtx_types.h).
          const bool     pairFresh = (blas.lastPairFrame == curFrameMv);
          const uint32_t pairKind  = pairFresh ? blas.lastPairKind : 9u;
          const float    pairScore = pairFresh ? blas.lastPairScore : 0.f;
          const int64_t  pairAge   = (pairFresh && blas.lastPairPrevTouched != kInvalidFrameIndex)
                                   ? (int64_t(curFrameMv) - int64_t(blas.lastPairPrevTouched))
                                   : -1;

          Logger::info(str::format(
            "[MvRaw] f=", curFrameMv,
            " id=", idMv,
            " vs=0x", std::hex,
              static_cast<uint64_t>(drawCall.getTransformData().vertexShaderHash), std::dec,
            " path=", mtnMovePath,
            " first=", (isFirstUpdateThisFrame ? 1 : 0),
            " created=", (isFirstUpdateAfterCreation ? 1 : 0),
            " mv=", mvLen,
            " rep=", repMv,
            " pml=", pmlMv,
            " fsl=", fslMv,
            " curT=(", cTv.x, ",", cTv.y, ",", cTv.z, ")",
            " prevT=(", pTv.x, ",", pTv.y, ",", pTv.z, ")",
            " pairKind=", pairKind,
            " pairScore=", pairScore,
            " pairAge=", pairAge,
            " blasTouched=", blas.frameLastTouched,
            " xfChanged=", (hasTransformChanged ? 1 : 0),
            " motionUnstable=", (isMotionUnstable ? 1 : 0),
            " hasPrevPos=", (hasPreviousPositions ? 1 : 0)));
        }

        // NV-DXVK [MtnMotion]: for the black/streaking SKY_MOUNTAIN VS, log how motion is being
        // tracked. The streak + low denoiser confidence point at a motion-vector fault. This
        // shows, per draw: which update path ran (teleport/move/moveAgain), whether the engine
        // marked a transform change, the material type + isMotionUnstable gate (translucent kills
        // prev-positions), and the actual prev-vs-cur world translation delta. If delta~0 while
        // the camera moves, OR path=0/2 in steady state, motion is being dropped -> streak + the
        // gradient spike that collapses RTXDI confidence -> NRD black.
        if (RtxOptions::logSurfaceCoverage()) {
          const uint64_t vsHmm = (uint64_t) drawCall.getTransformData().vertexShaderHash;
          if (vsHmm == 0x28f7ffa90d189017ull || vsHmm == 0x29146e1dd50b0314ull) {
            const Vector4 cT = currentInstance.surface.objectToWorld[3];
            const Vector4 pT = currentInstance.surface.prevObjectToWorld[3];
            const float dT = std::sqrt(float((cT.x-pT.x)*(cT.x-pT.x) + (cT.y-pT.y)*(cT.y-pT.y) + (cT.z-pT.z)*(cT.z-pT.z)));
            // NV-DXVK [MtnMotion] streak hunt: the motion vector is correct ONLY if this
            // frame's prevObjectToWorld == LAST frame's reprojected objectToWorld. The
            // ssmv probe shows the screen MV pops every ~3 frames (=streak); test whether
            // that pop is a STALE PREV (prev didn't follow the reproject drift) or a
            // genuine restream gap. Keyed by stable instance id so fanout (instCount=2)
            // doesn't mix the two instances. Logs:
            //  fsl  = frames since this instance was last seen here (1=every frame; >1=restream gap / skipped frame)
            //  rep  = |curT - lastCurT|   how far the reproject moved this instance's world pos vs last frame (T_reproject change)
            //  pml  = |prevT - lastCurT|  prev-matches-last; ~0 = prev correctly == last frame's cur (good); large = STALE PREV (bad)
            // DECIDER on an ssmv-spike frame: pml large => fix prev-tracking (move()/prev not getting last reproject);
            //                                 pml ~0 but screen MV still spikes => camera-phase mismatch (GPU prev-cam != prev-o2w frame).
            const uint32_t curFrameMM = m_device->getCurrentFrameId();
            const uint64_t instId = currentInstance.getId();
            struct LastSeen { uint32_t frame; float x, y, z; };
            static std::mutex sMtnMtx;
            static thread_local std::unordered_map<uint64_t, LastSeen> sLastSeen;
            int      fsl = -1;
            float    rep = -1.0f, pml = -1.0f;
            {
              std::lock_guard<std::mutex> lk(sMtnMtx);
              auto it = sLastSeen.find(instId);
              if (it != sLastSeen.end() && isFirstUpdateThisFrame) {
                const LastSeen& ls = it->second;
                fsl = int(curFrameMM) - int(ls.frame);
                rep = std::sqrt((float(cT.x)-ls.x)*(float(cT.x)-ls.x) + (float(cT.y)-ls.y)*(float(cT.y)-ls.y) + (float(cT.z)-ls.z)*(float(cT.z)-ls.z));
                pml = std::sqrt((float(pT.x)-ls.x)*(float(pT.x)-ls.x) + (float(pT.y)-ls.y)*(float(pT.y)-ls.y) + (float(pT.z)-ls.z)*(float(pT.z)-ls.z));
              }
              if (isFirstUpdateThisFrame)
                sLastSeen[instId] = LastSeen { curFrameMM, float(cT.x), float(cT.y), float(cT.z) };
            }
            Logger::info(str::format(
              "[MtnMotion] f=", curFrameMM, " vs=0x", std::hex, vsHmm, std::dec,
              " id=", instId,
              " path=", mtnMovePath, " xfChanged=", hasTransformChanged ? 1 : 0,
              " matType=", (uint32_t) currentInstance.m_materialType,
              " motionUnstable=", isMotionUnstable ? 1 : 0,
              " hasPrevPos=", hasPreviousPositions ? 1 : 0,
              " firstThisFrame=", isFirstUpdateThisFrame ? 1 : 0,
              " fsl=", fsl, " rep=", rep, " pml=", pml,
              " curT=(", float(cT.x), ",", float(cT.y), ",", float(cT.z), ")",
              " prevDelta=", dT));
          }
        }

        currentInstance.surface.textureTransform = drawCall.getTransformData().textureTransform;
        // NV-DXVK: log ONE entry per (hash-of-matrix, vsHash) combo so we see
        // every distinct transform any draw uses, paired with the VS hash so
        // we can correlate to wall draws. Key: we want to know the scale on
        // the WALL surfaces specifically — runtime probe sees uvLen ~1000
        // post-transform; if the wall transform is identity then the runtime
        // value IS what the BSP+VS-decode produces; if it's a strong scale,
        // the source decode is at a different magnitude.
        //
        // NV-DXVK [perf]: THE WHOLE PROBE IS NOW GATED. "[RTX-InstMgr.UVx]" is in
        // log.cpp's built-in denylist (log.cpp:455) and log.cpp applies that
        // filter inside emitMsg -- AFTER str::format has built the string. So
        // this block ran on every instance every frame and threw every line away,
        // paying for: six FNV steps over the texture transform, a full Matrix4
        // identity comparison whose result is read only by the discarded line, a
        // multiply, and an unordered_set insert into a set that never saturates.
        // The log confirms it: ZERO [RTX-InstMgr.UVx] lines in the entire run.
        // Process note 6 of the v5 handoff, and the same fix already applied at
        // the [MapWrite] site in onTransformChanged.
        //
        // Latched rather than re-asked: tagDenied is an atomic acquire load, a
        // strlen and a bucket walk. That is cheap for a per-draw site and NOT
        // cheap ~15,500 times a frame, and the answer cannot change -- the
        // denylist is published by Logger::setDenyTags during RtxOptions init,
        // long before the first gameplay frame, and is never rewritten after.
        static const bool kUVxDenied = Logger::tagDenied("[RTX-InstMgr.UVx]");
        if (!kUVxDenied) {
          const auto& m = currentInstance.surface.textureTransform;
          const bool isIdent = (m == Matrix4());
          // Hash of the four 2D-relevant entries (col0.x, col0.y, col1.x, col1.y, col3.x, col3.y)
          uint64_t key = 0;
          auto hashF = [&key](float v) {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            key = key * 0x100000001b3ull ^ uint64_t(bits);
          };
          hashF(m.data[0].x); hashF(m.data[0].y);
          hashF(m.data[1].x); hashF(m.data[1].y);
          hashF(m.data[3].x); hashF(m.data[3].y);
          // Combine with the geometry's texcoord hash so per-mesh-distinct
          // transforms log separately. Texcoord hash is a stable identifier
          // for which mesh's UVs we're seeing (same mesh = same hash).
          const uint64_t txcHash = uint64_t(currentInstance.m_texcoordHash);
          const uint64_t comboKey = key ^ (txcHash * 0x9E3779B97F4A7C15ull);
          static thread_local std::unordered_set<uint64_t> seen;
          if (seen.insert(comboKey).second) {
            Logger::info(str::format(
              "[RTX-InstMgr.UVx] txcHash=0x", std::hex, txcHash, std::dec,
              " ident=", isIdent ? "1" : "0",
              " col0=(", m.data[0].x, ",", m.data[0].y, ")",
              " col1=(", m.data[1].x, ",", m.data[1].y, ")",
              " col3=(", m.data[3].x, ",", m.data[3].y, ")"));
          }
        }

        currentInstance.surface.isStatic = !(hasTransformChanged || hasPreviousPositions) || currentInstance.m_materialType == MaterialDataType::RayPortal;

        currentInstance.surface.isClipPlaneEnabled = drawCall.getTransformData().enableClipPlane;
        currentInstance.surface.clipPlane = drawCall.getTransformData().clipPlane;

        // Apply developer options
        if (isFirstUpdateThisFrame)
          applyDeveloperOptions(currentInstance, drawCall);
      }
    }

    uiSplit.mark(2);   // NV-DXVK [Perf.UpdInst]: end of `xform`

    // We only have 1 hit shader.
    currentInstance.m_vkInstance.instanceShaderBindingTableRecordOffset = 0;

    // Update instance flags.
    // Note: this should happen on instance updates and not creation because the same geometry can be drawn
    // with different flags, and the instance manager can match an old instance of a geometry to a new one with different draw mode.
    currentInstance.m_vkInstance.flags = determineInstanceFlags(drawCall, currentInstance.surface);
    currentInstance.isFrontFaceFlipped = (currentInstance.m_vkInstance.flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR) != 0;

    // Apply the decal sort index for this instance so we can approximate order correctness on the GPU in AHS
    if (currentInstance.surface.alphaState.isDecal) {
      // NV-DXVK [Phase2b]: m_decalSortOrderCounter is ORDER-SENSITIVE — its value
      // must reflect draw order, which the parallel shard phase does not have.
      // Defer to the ordered tail, which assigns in arena (= draw) order via
      // InstanceManager::assignDecalSortOrder, so the values match the
      // sequential path exactly instead of racing. Tail continuations
      // (allowMiss) assign inline — they already run in arena order.
      if (inShardedInstancePhase() && !t_shardPhase.allowMiss) {
        DeferredSpatialOp op;
        op.kind = DeferredSpatialOp::Kind::kDecalOrder;
        op.instance = &currentInstance;
        t_shardPhase.info->spatialOps.push_back(op);
      } else {
      currentInstance.surface.decalSortOrder = m_decalSortOrderCounter++;
#if !NDEBUG
      if (m_decalSortOrderCounter > 255) {
        ONCE(Logger::err("Too many decals in this scene to sort correctly, may see some decal corruption issues."));
      }
#endif
      }  // NV-DXVK [Phase2b]: end of the deferred/inline decal-order split.
    }

    // NV-DXVK [Perf.NonOpaque]: which branch below claimed this instance. Only
    // kOpaque avoids the primary-ray resolve loop; every other value means a ray
    // that hits this instance re-traces the whole TLAS at least once more.
    enum NonOpaqueReason : uint8_t {
      kRTRenderTarget = 0,  // forced opaque-pass, any-hit geometry flags
      kUnorderedBlend = 1,  // particle/decal/player-blend/emissive -> unordered TLAS + FORCE_NO_OPAQUE
      kAlphaTested    = 2,  // primary TLAS, duplicate any-hits allowed
      kAlphaBlended   = 3,  // primary TLAS, FORCE_NO_OPAQUE
      kTranslucent    = 4,
      kRayPortal      = 5,
      kClipPlane      = 6,  // FORCE_NO_OPAQUE purely to run clip planes in any-hit
      kOpaque         = 7,
      kReasonCount    = 8
    };
    uint8_t nonOpaqueReason = kOpaque;

    // Update the geometry and instance flags
    if (currentInstance.isOpaque() && drawCall.isUsingRaytracedRenderTarget) {
      nonOpaqueReason = kRTRenderTarget;
      // render target texture - need this to be in the opaque pass, even if alphaState.isFullyOpaque is false.
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    } else if (
      (!currentInstance.surface.alphaState.isFullyOpaque && currentInstance.surface.alphaState.isParticle) ||
      (currentInstance.surface.alphaState.isDecal) ||
      // Note: include alpha blended geometry on the player model into the unordered TLAS. This is hacky as there might be
      // suitable geometry outside of the player model, but we don't have a way to distinguish it from alpha blended geometry
      // that should be alpha tested instead, like some metallic stairs in Portal -- those should be resolved normally.
      (!currentInstance.surface.alphaState.isFullyOpaque && !currentInstance.surface.alphaState.isBlendingDisabled && currentInstance.m_isPlayerModel) ||
      currentInstance.surface.alphaState.emissiveBlend
    ) {
      // Alpha-blended and emissive particles go to the separate "unordered" TLAS as non-opaque geometry
      nonOpaqueReason = kUnorderedBlend;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      currentInstance.m_isUnordered = true;
      // Unordered resolve only accumulates via any-hits and ignores opaque hits, therefore force 
      // the opaque hits resolve via OMMs to be turned into any-hits.
      // Note: this has unexpected effect even with OMM off and results in minor visual changes in Portal MF A DLSS test
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque && currentInstance.surface.alphaState.isBlendingDisabled) {
      // Alpha-tested geometry goes to the primary TLAS as non-opaque geometry with potential duplicate hits.
      nonOpaqueReason = kAlphaTested;
      currentInstance.m_geometryFlags = 0;
    } else if (currentInstance.isOpaque() && !currentInstance.surface.alphaState.isFullyOpaque) {
      // Alpha-blended geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      nonOpaqueReason = kAlphaBlended;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
      // Treat all non-transparent hits as any-hits
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::Translucent) {
      // Translucent (e.g. glass) geometry goes to the primary TLAS as non-opaque geometry with no duplicate hits.
      nonOpaqueReason = kTranslucent;
      currentInstance.m_geometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    } else if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
      // Portals go to the primary TLAS as opaque.
      nonOpaqueReason = kRayPortal;
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    } else if (currentInstance.surface.isClipPlaneEnabled) {
      nonOpaqueReason = kClipPlane;
      // Use non-opaque hits to process clip planes on visibility rays.
      // To handle cases when the same *static* object is used both with and without clip planes,
      // use the force bit to avoid BLAS confusion (because the geometry flags are baked into BLAS).
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
      currentInstance.m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    } else {
      // All other fully opaques go to the primary TLAS as opaque.
      currentInstance.m_geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    }
    
    // Enable backface culling for Portals to avoid additional hits to the back of Portals
    if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
      currentInstance.m_vkInstance.flags &= ~VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    // Update mask
    {
      uint mask = isFirstUpdateThisFrame ? 0 : currentInstance.m_vkInstance.mask;

      if (currentInstance.m_isPlayerModel && drawCall.cameraType != CameraType::ViewModel) {
        mask |= OBJECT_MASK_PLAYER_MODEL;
        // Lazy-clear stale instances if onFrameEnd() was skipped last frame (e.g. device loss on alt+tab)
        // NV-DXVK [Phase2b]: m_playerModelInstances is a global vector — a Sec-6
        // escape. Player-model draws are a handful per frame, so the lock is
        // cold; taking it around the lazy clear keeps the clear-then-push atomic.
        const uint32_t currentFrameId = m_device->getCurrentFrameId();
        const auto pushPlayerModel = [&]() {
          if (m_playerModelInstancesFrameId != currentFrameId) {
            m_playerModelInstances.clear();
            m_playerModelInstancesFrameId = currentFrameId;
          }
          m_playerModelInstances.push_back(&currentInstance);
        };
        if (inShardedInstancePhase()) {
          std::lock_guard<std::mutex> lock(m_shardEscapeMutex);
          pushPlayerModel();
        } else {
          pushPlayerModel();
        }
      } else {
        currentInstance.m_isPlayerModel = false;
        if (currentInstance.m_isUnordered && RtxOptions::enableSeparateUnorderedApproximations()) {
          if (currentInstance.surface.alphaState.isDecal) {
            mask = OBJECT_MASK_UNORDERED_ALL_BLENDED;
          } else {
            // Separate set of mask bits for the unordered TLAS
            if (currentInstance.surface.alphaState.emissiveBlend)
              mask |= OBJECT_MASK_UNORDERED_ALL_EMISSIVE;
            else
              mask |= OBJECT_MASK_UNORDERED_ALL_BLENDED;
          }
        }
        else {
          if (currentInstance.m_materialType == MaterialDataType::Translucent) {
            // Translucent material
            mask |= OBJECT_MASK_TRANSLUCENT;
          } else if (currentInstance.m_materialType == MaterialDataType::RayPortal) {
            // Portal
            mask |= OBJECT_MASK_PORTAL;
          } else {
            mask |= currentInstance.surface.alphaState.isBlendingDisabled ? OBJECT_MASK_OPAQUE : OBJECT_MASK_ALPHA_BLEND;
          }
        }
      }

      if (currentInstance.m_isHidden)
        mask = 0;

      currentInstance.m_vkInstance.mask = mask;
    }
    // This flag translates to a flip of VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR when the instance
    // is a separate BLAS instance, and to nothing if it's a part of a merged BLAS.
    // The reason is in this bit of Vulkan spec:
    //     VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR indicates that the facing determination for geometry in this instance
    //     is inverted. Because the facing is determined in object space, an instance transform does not change the winding,
    //     but a geometry transform does.
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGeometryInstanceFlagBitsNV.html 
    // NV-DXVK [fanout split]: same basis as determineInstanceFlags above — the
    // instance's own transform, which equals the draw call's for every
    // non-split draw and is the per-prop composed matrix for a split placement.
    // This value cancels out of the rendered facing flag (see the note in
    // determineInstanceFlags); it survives only into isFrontFaceFlipped and the
    // USD capture, where per-prop is the correct answer.
    uiSplit.mark(3);   // NV-DXVK [Perf.UpdInst]: end of `flags` (vkInstance flags, decal, opaque/non-opaque routing)

    currentInstance.m_isObjectToWorldMirrored = isMirrorTransform(currentInstance.surface.objectToWorld);

    bool billboardsGotGenerated = false;
    currentInstance.m_billboardCount = 0;
    
    if (drawCall.cameraType == CameraType::ViewModel && !currentInstance.m_isHidden && isFirstUpdateThisFrame) {
      // Lazy-clear stale candidates if onFrameEnd() was skipped last frame (e.g. device loss on alt+tab)
      // NV-DXVK [Phase2b]: m_viewModelCandidates is a global vector — Sec-6
      // escape, same treatment as m_playerModelInstances above. ViewModel draws
      // are a handful per frame; the lock is cold.
      const uint32_t currentFrameId = m_device->getCurrentFrameId();
      const auto pushVmCandidate = [&]() {
        if (m_viewModelCandidatesFrameId != currentFrameId) {
          m_viewModelCandidates.clear();
          m_viewModelCandidatesFrameId = currentFrameId;
        }
        m_viewModelCandidates.push_back(&currentInstance);
        Logger::info(str::format(
          "[VM.candidate] f=", currentFrameId,
          " candidates=", m_viewModelCandidates.size()));
      };
      if (inShardedInstancePhase()) {
        std::lock_guard<std::mutex> lock(m_shardEscapeMutex);
        pushVmCandidate();
      } else {
        pushVmCandidate();
      }
    }
    // NV-DXVK [VM.instance]: log every draw that reaches instance update so
    // we can see whether ViewModel-classified draws arrive here (proves
    // SceneManager → InstanceManager plumbing works).
    if (drawCall.cameraType == CameraType::ViewModel) {
      // NV-DXVK [Phase2b]: atomics — this path now runs on workers. The CAS on
      // the frame id makes exactly one thread reset the count per frame; the
      // fetch_add caps total lines at ~16/frame regardless of interleaving.
      static std::atomic<uint32_t> sLastF { 0 };
      static std::atomic<uint32_t> sCount { 0 };
      const uint32_t fid = m_device->getCurrentFrameId();
      uint32_t seenF = sLastF.load(std::memory_order_relaxed);
      if (seenF != fid && sLastF.compare_exchange_strong(seenF, fid, std::memory_order_relaxed)) {
        sCount.store(0, std::memory_order_relaxed);
      }
      if (sCount.fetch_add(1, std::memory_order_relaxed) < 16) {
        Logger::info(str::format(
          "[VM.instance] f=", fid,
          " isHidden=", (currentInstance.m_isHidden ? 1 : 0),
          " firstUpdate=", (isFirstUpdateThisFrame ? 1 : 0),
          " mask=", currentInstance.getVkInstance().mask));
      }
    }

    // [ZigGun] RE-ANCHORED on the POSITIVELY IDENTIFIED gun VS hash. The old
    // near-eye/vtx>1000 heuristic tagged the WRONG object (a 3-vtx decoy / player
    // body), so every prior [Zig*] measurement was of a non-gun — the whole v2/v3
    // dead-end. The first-person weapon is VS=0x292b6ba0d1854f28, confirmed via the
    // PickRegion coverage probe (largest stable center-bottom surface) AND a hide
    // test: rtx.debug.hideVertexShaders=0x292b6ba0d1854f28 removes the weapon.
    // Selecting s_zigGunInstance by this hash makes the downstream [ZigGunRB] /
    // [ZigNDC] skinned-vertex readback finally measure the thing that zig-zags.
    //
    // The per-frame line below localizes the horizontal sawtooth. The gun BLAS is
    // LOCAL space + an objectToWorld, so its world origin is the o2w translation;
    // we project that through the Main worldToView -> viewToProjection to a screen
    // ndcX. Across a walk cycle:
    //   sawtooth in wp/o2wT.x        -> the INSTANCE TRANSFORM jitters (fix there)
    //   smooth wp but sawtooth ndcX  -> the Main worldToView/camera is the source
    //   both smooth, gun still jerks -> the SKELETON/BONES (watch [ZigNDC], which
    //                                   reads the actual skinned verts, not o2w)
    // Gameplay-gated + throttled per project logging conventions. NOTE: to measure
    // the gun you must REMOVE it from rtx.debug.hideVertexShaders (hidden -> mask=0,
    // excluded from the AS build the readback consumes).
    {
      static constexpr XXH64_hash_t kGunVsHash = 0x292b6ba0d1854f28ull;
      if (drawCall.getTransformData().vertexShaderHash == kGunVsHash
          && !currentInstance.m_isHidden
          && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
        const uint32_t vtx = currentInstance.getBlas() ? currentInstance.getBlas()->modifiedGeometryData.vertexCount : 0u;
        // NV-DXVK: drop a stale tag from a previous frame BEFORE dereferencing it
        // below. The pointed-to RtInstance may have been freed since it was tagged
        // (device loss on alt+tab), so the getBlas() comparison would be a
        // use-after-free (rax=0xDD). Same-frame retags stay valid.
        // NV-DXVK [Phase2b]: the (s_zigGunInstance, s_zigGunInstanceFrameId) PAIR
        // must stay consistent — a torn (stale ptr, fresh fid) pair is exactly
        // the frame-gate-defeating UAF this tag already crashed on once. VS-gated
        // to a handful of draws per frame, so the escape lock is cold here.
        {
          std::unique_lock<std::mutex> zigLock;
          if (inShardedInstancePhase()) {
            zigLock = std::unique_lock<std::mutex>(m_shardEscapeMutex);
          }
        const uint32_t zigFid = m_device->getCurrentFrameId();
        if (s_zigGunInstanceFrameId != zigFid) {
          s_zigGunInstance = nullptr;
          s_zigGunInstanceFrameId = zigFid;
        }
        // Tag the highest-vtx gun draw (the body, if the weapon is split across
        // sub-meshes) for the ctx-readback in createViewModelInstances.
        if (s_zigGunInstance == nullptr
            || (s_zigGunInstance->getBlas() && vtx > s_zigGunInstance->getBlas()->modifiedGeometryData.vertexCount)) {
          s_zigGunInstance = &currentInstance;
        }
        }
        static std::atomic<uint32_t> sGunF { 0 }; static std::atomic<uint32_t> sGunC { 0 };
        const uint32_t fid = m_device->getCurrentFrameId();
        uint32_t seenGunF = sGunF.load(std::memory_order_relaxed);
        if (seenGunF != fid && sGunF.compare_exchange_strong(seenGunF, fid, std::memory_order_relaxed)) {
          sGunC.store(0, std::memory_order_relaxed);
        }
        if (sGunC.fetch_add(1, std::memory_order_relaxed) < 6) {
          const auto& mainCam = cameraManager.getMainCamera();
          const Matrix4 w2v = mainCam.getWorldToView(false);
          const Matrix4 v2p = mainCam.getViewToProjection();
          const Vector3 wp = currentInstance.getWorldPosition();
          const Matrix4 o2w = currentInstance.getTransform();
          const Vector4 vpos = w2v * Vector4(wp, 1.0f);
          const Vector4 ppos = v2p * vpos;
          const float ndcX = (ppos.w != 0.0f) ? (ppos.x / ppos.w) : 0.0f;
          Logger::info(str::format(
            "[ZigGun] f=", fid,
            " camType=", (uint32_t)drawCall.cameraType,
            " vtx=", vtx,
            " wp=(", wp.x, ",", wp.y, ",", wp.z, ")",
            " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
            " viewX=", vpos.x, " viewZ=", vpos.z,
            " ndcX=", ndcX,
            " mask=0x", std::hex, (uint32_t)currentInstance.getVkInstance().mask, std::dec));
        }
      }
    }

    uiSplit.mark(4);   // NV-DXVK [Perf.UpdInst]: end of `viewmodel`

    if (RtxOptions::enableSeparateUnorderedApproximations() &&
        (drawCall.cameraType == CameraType::Main || drawCall.cameraType == CameraType::ViewModel) &&
        currentInstance.m_isUnordered &&
        !currentInstance.m_isHidden &&
        currentInstance.getVkInstance().mask != 0) {

      // NV-DXVK [Phase2b]: createBillboards reads MAPPED BUFFER CONTENTS (a
      // positional read — the physical slice a logical buffer resolves to is a
      // property of CS stream position) and both create paths append to the
      // CS-domain m_billboards vector. On a worker, record eligibility and let
      // the CS record step run InstanceManager::runBillboardStage after the
      // capture rebind, exactly where the read is well-defined today.
      // billboardsGotGenerated stays false here; the flush-side event gate below
      // substitutes the pending flag, and the CS side re-evaluates with the real
      // outcome before firing the deferred OMM callback.
      if (inShardedInstancePhase()) {
        t_shardPhase.currentOps->billboard = true;
      } else {
      if (currentInstance.testCategoryFlags(InstanceCategories::Beam)) {
        createBeams(currentInstance);
      } else if(!currentInstance.surface.alphaState.isDecal) {
        createBillboards(currentInstance, cameraManager.getMainCamera().getDirection(false));
      }

      billboardsGotGenerated = currentInstance.m_billboardCount != 0;
      }
    }

    // NV-DXVK [Perf.NonOpaque]: census of the classification above, over the
    // instances that actually end up in a TLAS (mask != 0, not hidden).
    //
    // Why this and not [Perf.Tlas]/[Perf.TlasOverlap]: those count PRIMITIVES,
    // and the primitive share of any-hit geometry (3.1%) was used to retire the
    // opacity theory. That is the wrong denominator. The primary ray runs with
    // RAY_FLAG_FORCE_OPAQUE, so non-opaque geometry never costs any-hit
    // traversal - what it costs is an extra full-TLAS re-trace per pixel from
    // the resolve loop, and that fires once per *surface crossed*, regardless of
    // how many triangles that surface has. A 2-triangle fullscreen haze quad and
    // a 100k-triangle wall cost the resolve loop exactly the same. So the
    // denominator has to be instances (and, better, screen coverage), which is
    // where the 39%-of-instances number came from and why it disagrees so hard
    // with the 3.1%-of-primitives number.
    //
    uiSplit.mark(5);   // NV-DXVK [Perf.UpdInst]: end of `billboard`

    // COST -- and the earlier note here said "reported once per second so it is
    // safe to leave on during a timing run", which was wrong in the way that
    // matters. It REPORTS once per second; it ACCUMULATES per instance, right
    // here, inside updateInstance -- the ~20 ms/frame function that is the single
    // biggest item in the frame (v4 sec 4c). Per qualifying instance this takes
    // a mutex and probes an unordered_map, and its own output says how often:
    // instPerFrame=8853-9304. Default flipped to OFF 2026-08-06.
    //
    // Two things keep it honest if you turn it back on:
    //   - the option test is the FIRST term of the gate below, so when off the
    //     whole block is one bool load per instance;
    //   - while on, it is charged to the [Perf.SceneObj] `update` bucket, i.e.
    //     it inflates the exact number 4c exists to attribute. Turn it off
    //     before splitting updateInstance.
    // The mutex is uncontended in practice (updateInstance runs on dxvk-cs) but
    // uncontended is not free, and neither is the map probe's cache miss.
    if (RtxOptions::perfNonOpaqueCensus()
        && !currentInstance.m_isHidden
        && currentInstance.getVkInstance().mask != 0) {
      struct NonOpaqueCensus {
        std::mutex mu;
        uint64_t frames = 0;
        uint64_t instances = 0;
        uint64_t byReason[kReasonCount] = {};
        uint64_t primsByReason[kReasonCount] = {};
        uint64_t forceNoOpaque = 0;
        uint64_t unorderedTlas = 0;
        uint32_t lastFrame = UINT32_MAX;
        // Which vertex shaders produce the resolve-loop drivers, tallied by
        // INSTANCE count rather than primitive count. Primitive count is the
        // wrong weight for this mechanism by the same argument as above: the
        // resolve loop pays per surface crossed, not per triangle. Naming the
        // vertex shader is what makes the number actionable, since it maps back
        // to a specific piece of TF2 content.
        struct Driver { uint64_t instances = 0; uint64_t prims = 0; uint8_t reason = kOpaque; };
        std::unordered_map<uint64_t /*vsHash*/, Driver> byVs;
        std::chrono::steady_clock::time_point lastLog {};
      };
      static NonOpaqueCensus s_census;

      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t prims = drawCall.getGeometryData().calculatePrimitiveCount();
      const uint64_t vsHash = uint64_t(drawCall.getTransformData().vertexShaderHash);
      const bool fno = (currentInstance.getVkInstance().flags
                        & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) != 0;

      std::lock_guard<std::mutex> g(s_census.mu);

      if (s_census.lastFrame != frameId) {
        s_census.lastFrame = frameId;
        ++s_census.frames;
      }
      ++s_census.instances;
      ++s_census.byReason[nonOpaqueReason];
      s_census.primsByReason[nonOpaqueReason] += prims;
      if (fno)
        ++s_census.forceNoOpaque;
      if (currentInstance.m_isUnordered)
        ++s_census.unorderedTlas;

      const bool isDriver = nonOpaqueReason == kUnorderedBlend
                         || nonOpaqueReason == kAlphaTested
                         || nonOpaqueReason == kAlphaBlended
                         || nonOpaqueReason == kTranslucent
                         || nonOpaqueReason == kClipPlane;
      if (isDriver) {
        // Cap only new keys, so an already-tracked shader keeps accumulating
        // instead of freezing at whatever count it had when the map filled.
        auto it = s_census.byVs.find(vsHash);
        if (it == s_census.byVs.end() && s_census.byVs.size() < 512)
          it = s_census.byVs.emplace(vsHash, NonOpaqueCensus::Driver {}).first;
        if (it != s_census.byVs.end()) {
          ++it->second.instances;
          it->second.prims += prims;
          it->second.reason = nonOpaqueReason;
        }
      }

      const auto now = std::chrono::steady_clock::now();
      if (s_census.lastLog.time_since_epoch().count() == 0)
        s_census.lastLog = now;
      else if (now - s_census.lastLog >= std::chrono::seconds(1) && s_census.frames > 0) {
        const double ff = double(s_census.frames);
        const double perFrameInst = double(s_census.instances) / ff;
        // Only these five branches produce a surface whose material can resolve
        // below resolveOpaquenessThreshold and therefore re-arm continueResolving.
        // kRayPortal and kRTRenderTarget are deliberately excluded: both take an
        // early-out path in resolveVertex, so counting them here would inflate the
        // headline number with instances that do not drive the loop.
        const uint64_t resolveLoopDrivers =
            s_census.byReason[kUnorderedBlend]
          + s_census.byReason[kAlphaTested]
          + s_census.byReason[kAlphaBlended]
          + s_census.byReason[kTranslucent]
          + s_census.byReason[kClipPlane];
        const double driverPct = s_census.instances
          ? 100.0 * double(resolveLoopDrivers) / double(s_census.instances) : 0.0;

        static const char* kReasonNames[kReasonCount] = {
          "rtTarget", "unorderedBlend", "alphaTest", "alphaBlend",
          "translucent", "portal", "clipPlane", "opaque"
        };

        std::string byReason;
        for (uint32_t r = 0; r < kReasonCount; ++r) {
          byReason += str::format(" ", kReasonNames[r], "=",
                                  double(s_census.byReason[r]) / ff,
                                  "(", double(s_census.primsByReason[r]) / ff, "p)");
        }

        // Top vertex shaders by driver-instance count over the window.
        std::vector<std::pair<uint64_t, NonOpaqueCensus::Driver>> ranked(
          s_census.byVs.begin(), s_census.byVs.end());
        std::partial_sort(
          ranked.begin(),
          ranked.begin() + std::min<size_t>(5, ranked.size()),
          ranked.end(),
          [](const auto& a, const auto& b) { return a.second.instances > b.second.instances; });

        std::string offenders;
        for (size_t i = 0; i < ranked.size() && i < 5; ++i) {
          offenders += str::format(" vs=0x", std::hex, ranked[i].first, std::dec,
                                   ":", kReasonNames[ranked[i].second.reason],
                                   ":", double(ranked[i].second.instances) / ff, "inst",
                                   ":", double(ranked[i].second.prims) / ff, "p");
        }

        Logger::warn(str::format(
          "[Perf.NonOpaque] f=", frameId, " frames=", s_census.frames,
          " instPerFrame=", perFrameInst,
          " resolveLoopDrivers=", double(resolveLoopDrivers) / ff, " (", driverPct, "%)",
          " forceNoOpaque=", double(s_census.forceNoOpaque) / ff,
          " unorderedTlas=", double(s_census.unorderedTlas) / ff,
          " | byReason(count(prims)):", byReason.c_str(),
          " | topDriversByInstances:", offenders.c_str()));

        // Field-wise reset: the struct holds a std::mutex (which this scope still
        // owns the lock on), so it is neither copy- nor move-assignable.
        s_census.frames = 0;
        s_census.instances = 0;
        s_census.forceNoOpaque = 0;
        s_census.unorderedTlas = 0;
        for (uint32_t r = 0; r < kReasonCount; ++r) {
          s_census.byReason[r] = 0;
          s_census.primsByReason[r] = 0;
        }
        s_census.byVs.clear();
        s_census.lastLog = now;
        // Not frameId: the rest of THIS frame still has instances to report, and
        // they would otherwise land in a window whose frame count is zero.
        s_census.lastFrame = UINT32_MAX;
      }
    }

    // [CloudRoute] How are the 3D-skybox blended billboards (TF2 sky cloud
    // cards) routed? They reach here IgnoreAntiCulling-tagged + blend-enabled.
    // Logged AFTER billboard generation so m_billboardCount is final. route=
    // decodes the final object mask into the BVH/pass the geometry traces in:
    //   OPAQUE      -> opaque BVH, drawn solid (the bug, if a cloud lands here)
    //   ALPHA_BLEND -> alpha-blend BVH (ordered translucent)
    //   TRANSLUCENT -> translucent material pass
    //   U_BLENDED/U_EMISSIVE -> unordered TLAS (m_isUnordered billboard path)
    //   <none/hidden> -> mask==0, not traced at all
    // Keyed per vsHash, gameplay-gated.
    uiSplit.mark(6);   // NV-DXVK [Perf.UpdInst]: end of `census` (~0 when rtx.perfNonOpaqueCensus is off, which is the default)

    if (drawCall.testCategoryFlags(InstanceCategories::IgnoreAntiCulling)
        && !currentInstance.surface.alphaState.isBlendingDisabled
        && tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
      const uint32_t vsHashCR = uint32_t(uint64_t(drawCall.getTransformData().vertexShaderHash) & 0xffffffffu);
      static std::mutex s_cloudRouteMu;
      static thread_local std::unordered_set<uint32_t> s_cloudRouteSeen;
      bool firstCR = false;
      {
        std::lock_guard<std::mutex> g(s_cloudRouteMu);
        if (s_cloudRouteSeen.size() < 128 && s_cloudRouteSeen.insert(vsHashCR).second)
          firstCR = true;
      }
      if (firstCR) {
        const auto& as = currentInstance.surface.alphaState;
        const auto& bm = drawCall.getMaterialData().blendMode;
        const uint32_t m = currentInstance.getVkInstance().mask;
        // The unordered TLAS uses a SEPARATE bit namespace from the standard
        // mask - decode with the right one (matches the routing if/else above).
        const bool unorderedNs = currentInstance.m_isUnordered
                              && RtxOptions::enableSeparateUnorderedApproximations();
        std::string route;
        if (m == 0) {
          route = "<none/hidden>";
        } else if (unorderedNs) {
          if (m & OBJECT_MASK_UNORDERED_EMISSIVE_GEOMETRY)               route += "U_EMISSIVE_GEO ";
          if (m & OBJECT_MASK_UNORDERED_BLENDED_GEOMETRY)                route += "U_BLENDED_GEO ";
          if (m & OBJECT_MASK_UNORDERED_EMISSIVE_INTERSECTION_PRIMITIVE) route += "U_EMISSIVE_PRIM ";
          if (m & OBJECT_MASK_UNORDERED_BLENDED_INTERSECTION_PRIMITIVE)  route += "U_BLENDED_PRIM ";
        } else {
          if (m & OBJECT_MASK_TRANSLUCENT)          route += "TRANSLUCENT ";
          if (m & OBJECT_MASK_PORTAL)               route += "PORTAL ";
          if (m & OBJECT_MASK_ALPHA_BLEND)          route += "ALPHA_BLEND ";
          if (m & OBJECT_MASK_OPAQUE)               route += "OPAQUE ";
          if (m & OBJECT_MASK_VIEWMODEL)            route += "VIEWMODEL ";
          if (m & OBJECT_MASK_VIEWMODEL_VIRTUAL)    route += "VIEWMODEL_VIRTUAL ";
          if (m & OBJECT_MASK_PLAYER_MODEL)         route += "PLAYER_MODEL ";
          if (m & OBJECT_MASK_PLAYER_MODEL_VIRTUAL) route += "PLAYER_MODEL_VIRTUAL ";
        }
        if (route.empty()) route = "<unknown>";
        Logger::warn(str::format("[CloudRoute] vs=0x", std::hex, vsHashCR, std::dec,
          " cam=", static_cast<int>(drawCall.cameraType),
          " route=", route.c_str(),
          "(mask=0x", std::hex, m, std::dec, ")",
          " isUnordered=", currentInstance.m_isUnordered ? 1 : 0,
          " billboards=", currentInstance.m_billboardCount,
          " billboardsGen=", billboardsGotGenerated ? 1 : 0,
          " isHidden=", currentInstance.m_isHidden ? 1 : 0,
          " isPlayerModel=", currentInstance.m_isPlayerModel ? 1 : 0,
          " matType=", static_cast<int>(currentInstance.m_materialType),
          " | alphaState: fullyOpaque=", as.isFullyOpaque ? 1 : 0,
          " blendingDisabled=", as.isBlendingDisabled ? 1 : 0,
          " blendType=", static_cast<int>(as.blendType),
          " invertedBlend=", as.invertedBlend ? 1 : 0,
          " emissiveBlend=", as.emissiveBlend ? 1 : 0,
          " isParticle=", as.isParticle ? 1 : 0,
          " isDecal=", as.isDecal ? 1 : 0,
          " alphaTestType=", static_cast<int>(as.alphaTestType),
          // VkBlendFactor: 0=ZERO 1=ONE 6=SRC_ALPHA 7=ONE_MINUS_SRC_ALPHA;
          // VkBlendOp: 0=ADD. Used to see why looksPremultiplied failed in
          // calculateAlphaState (the ONE,ONE_MINUS_SRC_ALPHA disambiguation).
          " | blend: colorSrc=", static_cast<int>(bm.colorSrcFactor),
          " colorDst=", static_cast<int>(bm.colorDstFactor),
          " colorOp=", static_cast<int>(bm.colorBlendOp),
          " alphaSrc=", static_cast<int>(bm.alphaSrcFactor),
          " alphaDst=", static_cast<int>(bm.alphaDstFactor),
          " alphaOp=", static_cast<int>(bm.alphaBlendOp),
          " writeMask=0x", std::hex, static_cast<uint32_t>(bm.writeMask), std::dec));
      }
    }

    uiSplit.mark(7);   // NV-DXVK [Perf.UpdInst]: end of `anticull`

    // NV-DXVK [Phase2b]: on a worker the billboard stage was deferred, so
    // billboardsGotGenerated is structurally false. Substitute the pending flag
    // — the conservative "would have generated" — so the flush-side (SceneManager)
    // half of the fanout fires in the same cases as the sequential path. The
    // deferred (OMM) half is re-gated on the CS record step with the REAL
    // outcome of runBillboardStage, so over-prediction never reaches OMM.
    const bool billboardsPredicted = billboardsGotGenerated
      || (inShardedInstancePhase() && t_shardPhase.currentOps->billboard);

    // Updates done only once a frame unless overriden due to an explicit state
    if (isFirstUpdateThisFrame || overridePreviousCameraUpdate ||
        (billboardsPredicted && RtxOptions::getEnableOpacityMicromap())) {

      // NV-DXVK [perf] handoff v5 sec 4c: SKIP THE FANOUT ITSELF.
      //
      // The earlier 4c work memoized the CALLEE (createSurfaceMaterial) but still
      // made the CALL, ~15,500 times a frame through two std::function
      // indirections. What the two handlers actually produce for a settled
      // instance is a material index it already holds: m_surfaceMaterialCache is
      // never cleared per frame (only in SceneManager::clear(); [MatChurn] reports
      // matClear=0 and matNew=0 in every settled capture), so surfaceMaterialIndex
      // is stable across frames.
      //
      // The shared m_instStateKey covers what these handlers consume, because it
      // was WIDENED to do so: createSurfaceMaterial reads three sampler pointers
      // that the surf block does not, and those are carried in the key precisely
      // so one digest can authorise both skips. Reusing a surf-only key here would
      // have been a false hit of exactly the sec 0a kind.
      //
      // The binding epoch is the part that makes this sound across frames rather
      // than merely within one: see ResourceCache::getBindingEpoch(). Caching a
      // material index across frames is exactly what m_preCreationSurfaceMaterialMap
      // documents as unsafe, and the epoch is what re-establishes safety -- it
      // moves the moment either cache frees or clears a slot (inserts cannot move
      // an existing index, and including them made the gate never fire).
      //
      // ESCAPES -- conditions under which the fanout must run in full, because a
      // handler does per-frame work that has nothing to do with the material:
      //   hasTransformChanged / hasPreviousPositions -> GameCapturer update flags
      //   billboardsGotGenerated                     -> OMM builds on first sight
      //   RayPortal material                         -> processRayPortalData
      //
      // NOTE: "OMM is enabled" is deliberately NOT an escape. It was, in the first
      // version of this gate, and that was wrong: rtx.opacityMicromap.enable is ON
      // in this configuration ("[RTX] Opacity Micromap: enabled" in the log), so a
      // blanket OMM escape made fanoutMustRun unconditionally true and the gate
      // never fired once. OMM's own handler is instead marked NOT skippable via
      // InstanceEventHandler::skippableWhenBindingUnchanged, so it keeps seeing
      // every instance while SceneManager's handler -- the expensive one -- is
      // skipped. Per-handler is the correct granularity here; a global flag can
      // only ever be as permissive as its most demanding listener.
      const bool fanoutMustRun =
        hasTransformChanged || hasPreviousPositions || billboardsPredicted ||
        materialData.getType() == MaterialDataType::RayPortal;

      // NO SECOND KEY. instStateUnchanged was decided once in the `surf` block from
      // a digest covering the union of both blocks' inputs -- including the three
      // sampler pointers and the binding epoch that only this gate needs. Hashing
      // again here was the previous shape and it was pure waste: the cost of these
      // keys is the scattered reads, not the hash, and both gates read the same
      // objects. If the fanout is forced to run, the key stays as written above,
      // which is correct: this frame DID re-derive the binding, so next frame may
      // legitimately skip on it.
      const bool skipFanout = instStateUnchanged && !fanoutMustRun;

      if (skipFanout) {
        uiSplit.tailSkipped();
      }

      // Inform the listeners. When the binding is provably unchanged only the
      // handlers that OPTED IN to being skippable are dropped; everything else
      // still sees every instance, so a listener that does per-frame work
      // unrelated to the material cannot be starved by this gate.
      // NV-DXVK [Phase2b]: on a worker the fanout splits by CONTRACT, not type:
      // handlers that declared skippableWhenNoPendingOmmWork (OMM — reads buffer
      // contents, writes unlocked CS-domain maps) are deferred to the CS record
      // step with these exact args; everything else (SceneManager — materials,
      // capturer, ray portals; internally Phase2b-aware) runs inline here.
      for (auto& event : m_eventHandlers) {
        if (skipFanout && event.skippableWhenBindingUnchanged) {
          continue;
        }
        if (inShardedInstancePhase() && event.skippableWhenNoPendingOmmWork) {
          t_shardPhase.currentOps->omm = true;
          t_shardPhase.currentOps->evHasTransformChanged = hasTransformChanged;
          t_shardPhase.currentOps->evHasPreviousPositions = hasPreviousPositions;
          t_shardPhase.currentOps->evIsFirstUpdateThisFrame = isFirstUpdateThisFrame;
          continue;
        }
        event.onInstanceUpdatedCallback(currentInstance, drawCall, materialData, hasTransformChanged, hasPreviousPositions, isFirstUpdateThisFrame);
      }
    }

    uiSplit.mark(8);   // NV-DXVK [Perf.UpdInst]: end of `tail` (the event-handler fanout)

    // NV-DXVK [perf] fastInstanceUpdate: this full update derived flags/mask
    // from THIS draw's unkeyed inputs -- record them so next frame's fast check
    // can prove they have not moved. Written on every full path so the baseline
    // can never go stale.
    currentInstance.m_fastDrawBits = drawState.fastDrawBits;

    // NV-DXVK [Perf.UpdInst]: exact per-instance counters, taken from the values
    // this function already computed. REDUNDANT (no transform change, no material
    // change, no previous positions) is the one that answers sec 4c's open
    // question -- how much of ~20 ms/frame is rebuilding unchanged state.
    uiSplit.counts(isFirstUpdateThisFrame,
                   hasTransformChanged,
                   currentInstance.surface.hasMaterialChanged,
                   hasPreviousPositions,
                   currentInstance.surface.isStatic);
  }

  void InstanceManager::removeInstance(RtInstance* instance) {
    // NV-DXVK [VS_2904d2 removeInstance probe]: log when sub-view mountain
    // instances get destroyed. There are three paths to removal:
    //   (1) Lifetime expiry: instance->m_frameLastUpdated + numFramesToKeep
    //       Instances <= currentFrame (the standard GC pass).
    //   (2) m_isMarkedForGC=true (BLAS destroyed via onSceneObjectDestroyed,
    //       or viewModel/clone/virtual instance lifecycle).
    //   (3) forceGarbageCollection (instance count over numObjectsToKeep cap).
    // Logging which path fires tells us WHY dedup misses despite our reproject
    // producing byte-identical world positions across frames.
    {
      const BlasEntry* pBlas = instance->getBlas();
      if (pBlas != nullptr) {
        const XXH64_hash_t vsHash = pBlas->input.getTransformData().vertexShaderHash;
        if (vsHash == 0x2904d2163ef31a17ull) {
          thread_local uint32_t sRemoveProbe = 0;
          if (sRemoveProbe < 32 || (sRemoveProbe & 0x3FF) == 0) {
            const uint32_t curFrame = m_device->getCurrentFrameId();
            const uint32_t lastUpdated = instance->m_frameLastUpdated;
            const uint32_t lifetimeOpt = RtxOptions::numFramesToKeepInstances();
            const bool lifetimeExpired = (lastUpdated + lifetimeOpt <= curFrame);
            Logger::info(str::format(
              "[Rm2904] #", sRemoveProbe,
              " f=", curFrame,
              " lastUpd=", lastUpdated,
              " markedGC=", (instance->m_isMarkedForGC ? 1 : 0),
              " unlinkedGC=", (instance->m_isUnlinkedForGC ? 1 : 0),
              " lifetimeExp=", (lifetimeExpired ? 1 : 0),
              " keepN=", lifetimeOpt,
              " insideFrustum=", (instance->m_isInsideFrustum ? 1 : 0),
              " isHidden=", (instance->m_isHidden ? 1 : 0),
              " createdByRenderer=", (instance->m_isCreatedByRenderer ? 1 : 0)));
          }
          sRemoveProbe += 1;
        }
      }
    }

    // NV-DXVK [Perf.PushInst] PHASE 2: drop any batch record holding this
    // pointer. FIRST, and above the m_isCreatedByRenderer early-return below,
    // because this is a lifetime invariant and not a feature: every path that
    // destroys an instance passes through here, and an instance that leaves
    // without invalidating its record leaves a dangling RtInstance* behind for
    // the next frame's bulk stamp to write through. No-op when the feature is
    // off (m_batchRecordKey stays 0), so it costs one predictable load.
    invalidateFanoutRecordFor(instance);

    // NV-DXVK [ResidentScene]: and drop any RESIDENT record holding this
    // pointer, for exactly the same reason and in exactly the same place.
    //
    // This one matters MORE than the fanout invalidation above, not less: a
    // fanout record is rebuilt every frame the batch is submitted, so a stale
    // pointer in it has a short window. A resident record is designed to
    // survive an UNBOUNDED number of frames without being rebuilt, so a
    // dangling RtInstance* left here would be written through by the bulk
    // stamp for as long as the record lives -- which is forever, by design.
    // No-op when residency is off (m_residentKey stays 0).
    m_residentScene.invalidateFor(instance);

    // Always clean up replacement instance references, even for renderer-created instances
    // to avoid use-after-free bugs in ReplacementInstance.prims
    instance->getPrimInstanceOwner().setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex, instance, PrimInstance::Type::Instance);
    instance->removeFromSpatialCache();
    
    // In these cases we skip calling onInstanceDestroyed:
    //   Some view model and player instances are created in the renderer and don't have onInstanceAdded called,
    //   so not call onInstanceDestroyed either.
    if (instance->m_isCreatedByRenderer) {
      return;
    }

    
    for (auto& event : m_eventHandlers) {
      event.onInstanceDestroyedCallback(*instance);
    }
  }

  RtInstance* InstanceManager::createViewModelInstance(Rc<DxvkContext> ctx,
                                                       const RtInstance& reference,
                                                       const Matrix4d& perspectiveCorrection,
                                                       const Matrix4d& prevPerspectiveCorrection) {

    // Create a view model instance corresponding to the reference instance, for one frame 

    // Don't pollute global instance id with View Models since they're not tracked in game capturer
    const bool needValidGlobalInstanceId = false;

    RtInstance* viewModelInstance = createInstanceCopy(reference, needValidGlobalInstanceId);

    const uint32_t frameId = m_device->getCurrentFrameId();
    viewModelInstance->setFrameCreated(frameId);
    viewModelInstance->setFrameLastUpdated(frameId);
    viewModelInstance->m_vkInstance.mask = OBJECT_MASK_VIEWMODEL;
    viewModelInstance->setCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL, true);

    // View model instances are recreated every frame
    viewModelInstance->markForGarbageCollection();

    // NV-DXVK [zig-zag ROOT FIX]: for the TF2 engine-hook viewmodel the per-draw
    // o2w (reference.getTransform()) was sampled from the camera-manager Main on
    // the DRAW thread — a frame out of step with the perspectiveCorrection built
    // on the CS thread (proven: [ZigSync] renderCam lagged [ZigPC] mV2W_T by one
    // frame, so mainW2V*o2w != I and the residual sawtoothed into the horizontal
    // zig-zag). For the Main terms to cancel (clip = vmProj*scale*v) the o2w MUST
    // be the IDENTICAL matrix perspectiveCorrection used. Re-source it here from
    // the SAME camera-manager Main — same CS thread, same frame as the pc built
    // by the caller — and teleport the instance so the geometry-correction path
    // also renders with it. Gated to the engine-hook path so non-TF2 viewmodels
    // keep their real per-draw o2w.
    // [ZigProbe] instrumentation state — what happened to THIS viewmodel instance.
    int  zigBranch = -1;            // 0=ordinary-teleport, 1=geometry-dispatched, 2=geom-gated-out
    bool zigDispatched = false;
    uint64_t zigPosAddrBefore = 0, zigPosAddrAfter = 0;
    uint32_t zigVtxCount = 0;
    const void* zigBlasPtr = (const void*)viewModelInstance->getBlas();
    if (viewModelInstance->getBlas() != nullptr) {
      const auto& pb0 = viewModelInstance->getBlas()->modifiedGeometryData.positionBuffer;
      if (pb0.defined()) { zigPosAddrBefore = (uint64_t)pb0.getDeviceAddress() + pb0.offsetFromSlice(); }
      zigVtxCount = viewModelInstance->getBlas()->modifiedGeometryData.vertexCount;
    }

    Matrix4 refXform = reference.getTransform();
    Matrix4 refPrevXform = reference.getPrevTransform();
    if (RtxOptions::useEngineHookMainCamera()) {
      const auto& mainCam =
        ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
      refXform = mainCam.getViewToWorld(false);
      refPrevXform = mainCam.getPreviousViewToWorld(false);
      viewModelInstance->teleport(refXform, refPrevXform);
    }

    if (RtxOptions::ViewModel::perspectiveCorrection()) {
      // A transform that looks "correct" only from a main camera's point of view
      const auto corrected = perspectiveCorrection * refXform;
      const auto prevCorrected = prevPerspectiveCorrection * refPrevXform;

      auto isOrdinary = [](const Matrix4d& m) {
        auto isCloseTo = [](auto a, auto b) {
          return std::abs(a - b) < 0.001;
        };
        return isCloseTo(m[0][3], 0.0)
          && isCloseTo(m[1][3], 0.0)
          && isCloseTo(m[2][3], 0.0)
          && isCloseTo(m[3][3], 1.0);
      };

      // If matrices are not convoluted, don't modify the vertex data: just set the transforms directly
      // [zig-zag FIX] GROUND TRUTH ([ZigNDC]): v0 IS the gun's world position
      // (v0-camMain=(0,0,-60)); mainW2V*v0 gives view-space pos with constant Y
      // but SAWTOOTHING X = the horizontal zig-zag. v0 lags because it's baked in
      // the ViewModel-camera frame, which phase-lags Main. The RT renders v0 as
      // world (instance transform is effectively ignored — proven: applying it
      // throws v0 14000u away yet the gun renders glued). So the fix must modify
      // v0, and we FORCE the geometry path to do so for the engine-hook viewmodel.
      if (isOrdinary(corrected) && isOrdinary(prevCorrected) && !RtxOptions::useEngineHookMainCamera()) {
        viewModelInstance->teleport(corrected, prevCorrected);
        zigBranch = 0;
      } else {
        ONCE(Logger::info("[RTX-Compatibility-Info] Unexpected values in the perspective-corrected transform of a view model. Fallback to geometry modification"));
        // Only need to run this on BVH op (maybe this could be moved to geometry processing?)
        if (viewModelInstance->getBlas()->frameLastUpdated == frameId) {
          Matrix4d instancePositionTransform;
          if (RtxOptions::useEngineHookMainCamera()) {
            // Reproject the lagging world verts so they are GLUED to the Main
            // camera: v_new = mainV2W * vmW2V * v. Then mainW2V*v_new = vmW2V*v =
            // the gun relative to its OWN (same-frame) bake camera = constant, no
            // lag. Verify: [ZigNDC] viewDirect.x must go flat (was -7..+68 sawtooth).
            auto& cm = ctx->getCommonObjects()->getSceneManager().getCameraManager();
            const Matrix4 mainV2W = cm.getCamera(CameraType::Main).getViewToWorld(false);
            const Matrix4 vmW2V   = cm.getCamera(CameraType::ViewModel).getWorldToView(false);
            instancePositionTransform = Matrix4d(mainV2W * vmW2V);
          } else {
            const auto worldToObject = inverse(refXform);
            instancePositionTransform = worldToObject * perspectiveCorrection * refXform;
          }

          ctx->getCommonObjects()->metaGeometryUtils().dispatchViewModelCorrection(ctx,
            viewModelInstance->getBlas()->modifiedGeometryData, instancePositionTransform);
          zigBranch = 1; zigDispatched = true;

          // [zig-zag FIX] the verts are now WORLD-space (reprojected onto Main).
          // The instance transform must be IDENTITY so the RT renders them as-is
          // instead of re-applying mainV2W (=refXform from line 2724), which would
          // re-displace the now-world geometry. This is the completing half of the
          // geometry-path fix.
          if (RtxOptions::useEngineHookMainCamera()) {
            viewModelInstance->teleport(Matrix4());
          }
        } else {
          zigBranch = 2;
        }
      }
    }

    // [ZigProbe] DECISIVE per-instance dump: which BLAS, which buffer, branch,
    // whether the correction ran, the buffer device-address (match this against
    // [ZigBlas] in accel_manager to see if the RT builds from THIS buffer), and
    // the final instance transform. Logged for every viewmodel instance, every
    // frame, so multi-instance / wrong-BLAS / gated-out cases are all visible.
    {
      if (viewModelInstance->getBlas() != nullptr) {
        const auto& pb1 = viewModelInstance->getBlas()->modifiedGeometryData.positionBuffer;
        if (pb1.defined()) { zigPosAddrAfter = (uint64_t)pb1.getDeviceAddress() + pb1.offsetFromSlice(); }
      }
      const auto& ft = viewModelInstance->getTransform();
      Logger::info(str::format(
        "[ZigInst] f=", frameId,
        " inst=", (const void*)viewModelInstance,
        " blas=", zigBlasPtr,
        " branch=", zigBranch, " dispatched=", (zigDispatched ? 1 : 0),
        " vtx=", zigVtxCount,
        " posAddrBefore=", zigPosAddrBefore,
        " posAddr=", zigPosAddrAfter,
        " mask=0x", std::hex, (uint32_t)viewModelInstance->m_vkInstance.mask, std::dec,
        " o2wT=(", ft[3][0], ",", ft[3][1], ",", ft[3][2], ")",
        " blasFrameUpd=", (viewModelInstance->getBlas() ? viewModelInstance->getBlas()->frameLastUpdated : 0u),
        " thisFrame=", frameId));
    }

    // NV-DXVK [ZigScreen]: measure the gun's ACTUAL on-screen position of a
    // FIXED object-space point through the exact effective chain it renders with
    // (mainCam.viewToProj * mainCam.worldToView * perspectiveCorrection * refXform).
    // A fixed point removes the bones from the equation:
    //   ndc sawtooths during smooth motion -> transform chain is the culprit
    //                                          (vmProj / cancellation not holding)
    //   ndc smooth                          -> transform OK, sawtooth is in bones v
    if (RtxOptions::useEngineHookMainCamera()) {
      const uint32_t zsFrame = m_device->getCurrentFrameId();
      static uint32_t s_zsLastFrame = UINT32_MAX;
      if (zsFrame != s_zsLastFrame) {
        s_zsLastFrame = zsFrame;
        const auto& mc =
          ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
        const Matrix4d eff =
          mc.getViewToProjection() * (mc.getWorldToView(false) * (perspectiveCorrection * refXform));
        // Object origin is the eye (view-local gun) → w≈0 there, useless. Instead
        // report where FORWARD points land: ndc.x(z) = (eff[2][0]*z + eff[3][0]) /
        // (eff[2][3]*z + eff[3][3]); as z→∞ this → eff[2][0]/eff[2][3], i.e. the
        // gun's horizontal screen anchor — independent of the unknown depth/scale.
        // If this ratio sawtooths during smooth motion, the gun sawtooths
        // horizontally and the cause is the transform chain, not the bones.
        const double c20 = eff[2][0], c21 = eff[2][1], c23 = eff[2][3]; // depth col: x,y,w
        Logger::info(str::format(
          "[ZigScreen] f=", zsFrame,
          " ndcX_fwd=", (c23 != 0.0 ? c20 / c23 : 0.0),
          " ndcY_fwd=", (c23 != 0.0 ? c21 / c23 : 0.0),
          " c20=", c20, " c21=", c21, " c23=", c23));
      }
    }

    // NV-DXVK [ZigVB]: ground-truth readback of the gun's final vertices, run
    // every frame regardless of teleport-vs-geometry path (unlike
    // dispatchViewModelCorrection which is gated by the BLAS-update). Also
    // reports whether the BLAS was rebuilt this frame.
    if (RtxOptions::useEngineHookMainCamera() && viewModelInstance->getBlas() != nullptr) {
      const uint32_t blasUpd =
        (viewModelInstance->getBlas()->frameLastUpdated == frameId) ? 1u : 0u;
      // [ZigNDC] pass the EXACT chain the ray tracer uses: the instance's final
      // objectToWorld (post-teleport) + the Main render camera's worldToView and
      // viewToProjection, so the readback can compute the gun's true on-screen NDC.
      const auto& ndcMainCam =
        ctx->getCommonObjects()->getSceneManager().getCameraManager().getCamera(CameraType::Main);
      ctx->getCommonObjects()->metaGeometryUtils().debugReadbackViewModelVerts(
        ctx, viewModelInstance->getBlas()->modifiedGeometryData, blasUpd,
        Matrix4(viewModelInstance->getTransform()),
        ndcMainCam.getWorldToView(false),
        ndcMainCam.getViewToProjection());
    }

    // ViewModel should never be considered static
    viewModelInstance->surface.isStatic = false;

    // Note this is an instance copy of a input reference. It is unknown to the source engine, so we don't call onInstanceAdded callbacks for it
    // It also results in this instance not being linked to reference instance BLAS and thus not considered in findSimilarInstances' lookups
    // This is desired as ViewModel instances are not to be linked frame to frame

    // NV-DXVK [VM.final]: log the final viewmodel instance's transform +
    // mask so we can see where it'd render and whether the BVH/TLAS upload
    // will accept it.
    {
      const auto& t = viewModelInstance->getTransform();
      // NV-DXVK [ZigGeo]: split the gun's ±1u horizontal wobble into its two
      // inputs. corrected (=T) = perspectiveCorrection * reference.getTransform().
      // Cameras are PROVEN stable ([ZigCam]: main/vm/engineEye all steady at
      // -25.60), so if T wobbles the noise is in ONE of these two:
      //   refT = reference.getTransform() = the gun's RAW per-draw objectToWorld
      //          (pure geometry — if THIS wobbles, the source is the draw's o2w).
      //   pcT  = perspectiveCorrection translation (camera-derived — if THIS
      //          wobbles despite stable cameras, the correction math is unstable).
      // Watch the .x of refT vs pcT across frames against T.x's ±1u sawtooth.
      const auto& refT = reference.getTransform();
      const auto& pcT  = perspectiveCorrection;
      Logger::info(str::format(
        "[VM.final] f=", frameId,
        " mask=0x", std::hex, (uint32_t)viewModelInstance->m_vkInstance.mask, std::dec,
        " pc=", (RtxOptions::ViewModel::perspectiveCorrection() ? 1 : 0),
        " T=(", t[3][0], ",", t[3][1], ",", t[3][2], ")",
        " refT=(", refT[3][0], ",", refT[3][1], ",", refT[3][2], ")",
        " pcT=(", pcT[3][0], ",", pcT[3][1], ",", pcT[3][2], ")",
        " diag=(", t[0][0], ",", t[1][1], ",", t[2][2], ")"));
    }

    return viewModelInstance;
  }

  void InstanceManager::createViewModelInstances(Rc<DxvkContext> ctx,
                                                 const CameraManager& cameraManager,
                                                 const RayPortalManager& rayPortalManager) {
    ScopedGpuProfileZone(ctx, "ViewModel");

    const uint32_t fid = m_device->getCurrentFrameId();
    const bool vmEnable = RtxOptions::ViewModel::enable();
    const bool vmCamValid = cameraManager.isCameraValid(CameraType::ViewModel);
    Logger::info(str::format(
      "[VM.create] f=", fid,
      " enable=", (vmEnable ? 1 : 0),
      " camValid=", (vmCamValid ? 1 : 0),
      " candidates=", m_viewModelCandidates.size()));

    // [ZigGun] readback the REAL gun (tagged in updateInstance) through the same
    // viewDirect machinery used on the 3-vtx decoy. Runs first this frame so the
    // readback ring captures the GUN, not the viewmodel triangle. Tells us
    // whether the gun's geometry lags (viewDirect.x sawtooth) and its vertex space.
    // NV-DXVK: only consume the tag if it was set THIS frame. On a device-loss
    // frame the gun isn't drawn (updateInstance never retags) yet this code still
    // runs; the leftover tag points at a freed RtInstance -> use-after-free
    // (rax=0xDD AV at getBlas()->frameLastUpdated). The frame-id gate makes the
    // raw cross-frame pointer safe.
    if (s_zigGunInstance != nullptr && s_zigGunInstanceFrameId == fid && s_zigGunInstance->getBlas() != nullptr) {
      const auto& gunMain = cameraManager.getMainCamera();
      const uint32_t gunBlasUpd =
        (s_zigGunInstance->getBlas()->frameLastUpdated == fid) ? 1u : 0u;
      // NV-DXVK [ShipXform]: full o2w + the Main worldToView it's built from. The ship's verts
      // are static (o2wT=0) but in the vanish view direction a WRONG o2w teleports them behind the
      // camera (world (-199.582,-15364,10187.6), view Z +100). Dump the full matrices so we can
      // see whether the bad o2w is the rotation block being corrupted, a camera matrix leaking in,
      // or worldToView itself being wrong for that yaw — and trace the source.
      const Matrix4 zo2w = Matrix4(s_zigGunInstance->getTransform());
      const Matrix4 zw2v = gunMain.getWorldToView(false);
      // ========================================================================
      // !!! WARNING: s_zigGunInstance RE-TAGS BETWEEN DIFFERENT INSTANCES !!!
      // ------------------------------------------------------------------------
      // Do NOT trust [ZigNDC]/[ShipXform]/[ZigGunRB] world position as "one mesh
      // moving." s_zigGunInstance is set per-frame to whichever 0x292b draw was
      // tagged last, and across the "vanish" it FLIPS between distinct instances:
      // the 15817-vtx mesh (reads near-camera) and the 25537-vtx mesh (reads at
      // the fixed -199.582 anchor). posHash changes with the flip. So the dramatic
      // "world teleports behind camera" is largely a TAG-TRACKING ARTIFACT of this
      // probe, not a proven single-mesh teleport. (Earlier handoffs built a whole
      // engine-vs-Remix-bake theory on this probe — see the DEAD-END note in
      // rtx_geometry_utils.cpp interleaveGeometry: the skinning is verified CORRECT
      // (blend3 match=1). Don't chase bones/cb3/o2w via this probe again.)
      // To measure a real teleport, pin to ONE stable instance (filter by vtx +
      // a persistent key), don't consume the last-wins s_zigGunInstance tag.
      // ========================================================================
      // [ZigGunRB] instance-identity fields: inst ptr / vtx / posHash / blasUpd —
      //   posHash SAME across the jump  -> source object verts unchanged; the teleport is a
      //     transform/bake-application bug (Remix applies the wrong space when baking world).
      //   posHash CHANGES across the jump -> the source vertex data itself changed (engine
      //     re-uploaded the packed VB in a different space / pose).
      // blasUpd=1 means the BLAS was rebuilt THIS frame (modifiedGeometryData refreshed).
      const auto& zGeo = s_zigGunInstance->getBlas()->modifiedGeometryData;
      Logger::info(str::format(
        "[ZigGunRB] f=", fid,
        " inst=", (const void*)s_zigGunInstance,
        " vtx=", zGeo.vertexCount,
        " blasUpd=", gunBlasUpd,
        std::hex,
        " posHash=0x", zGeo.hashes[HashComponents::VertexPosition],
        " idxHash=0x", zGeo.hashes[HashComponents::Indices],
        std::dec,
        " o2wT=(", zo2w[3][0], ",", zo2w[3][1], ",", zo2w[3][2], ")"));
      Logger::info(str::format(
        "[ShipXform] f=", fid,
        " o2w_r0=(", zo2w[0][0], ",", zo2w[0][1], ",", zo2w[0][2], ",", zo2w[0][3], ")",
        " o2w_r1=(", zo2w[1][0], ",", zo2w[1][1], ",", zo2w[1][2], ",", zo2w[1][3], ")",
        " o2w_r2=(", zo2w[2][0], ",", zo2w[2][1], ",", zo2w[2][2], ",", zo2w[2][3], ")",
        " o2w_r3=(", zo2w[3][0], ",", zo2w[3][1], ",", zo2w[3][2], ",", zo2w[3][3], ")"));
      Logger::info(str::format(
        "[ShipXform] f=", fid,
        " w2v_r0=(", zw2v[0][0], ",", zw2v[0][1], ",", zw2v[0][2], ",", zw2v[0][3], ")",
        " w2v_r1=(", zw2v[1][0], ",", zw2v[1][1], ",", zw2v[1][2], ",", zw2v[1][3], ")",
        " w2v_r2=(", zw2v[2][0], ",", zw2v[2][1], ",", zw2v[2][2], ",", zw2v[2][3], ")",
        " w2v_r3=(", zw2v[3][0], ",", zw2v[3][1], ",", zw2v[3][2], ",", zw2v[3][3], ")"));
      ctx->getCommonObjects()->metaGeometryUtils().debugReadbackViewModelVerts(
        ctx, s_zigGunInstance->getBlas()->modifiedGeometryData, gunBlasUpd,
        Matrix4(s_zigGunInstance->getTransform()),
        gunMain.getWorldToView(false),
        gunMain.getViewToProjection());
    }
    s_zigGunInstance = nullptr;

    // NV-DXVK [HullWorldRB]: STABLE-INSTANCE world probe — the correct successor to the
    // re-tagging s_zigGunInstance/[ZigNDC] readback (see warning above). Enumerate ALL
    // big 0x292b instances this frame and batch a GPU readback of each one's vertex-0
    // world position, keyed by stable BLAS ptr. Diff across a visible->vanish pair: a
    // single blas key teleporting to -199.582 = REAL teleport (then chase BLAS-merge /
    // draw-call-cache); a near-cam key AND a -199.582 key BOTH present every frame =
    // the vanish was the tag artifact. Throttled to once per frame (heavy: waitForIdle).
    // Gated behind tf2HeavyProbes (default OFF): this readback does a per-frame
    // flush+waitForIdle — a primary Aftermath device-loss (freeze→crash) driver.
    if (RtxOptions::tf2HeavyProbes()) {
      static uint32_t s_hullRbLastFid = UINT32_MAX;
      if (fid != s_hullRbLastFid) {
        s_hullRbLastFid = fid;
        std::vector<RtxGeometryUtils::HullReadbackItem> items;
        for (const RtInstance* pInst : m_instances) {
          if (pInst == nullptr) continue;
          const BlasEntry* pBl = pInst->getBlas();
          if (pBl == nullptr) continue;
          // NV-DXVK [StudioModelHook] re-gate: enumerate the actual Widow
          // engine model (precise) instead of the shared VS hash 0x292b
          // (which also matched sky/world/weapon). Requires rtx.tf2DetectWidow
          // (or tf2HideWidow/tf2IsolateWidow). vertexCount floor dropped — all
          // Widow sub-meshes are interesting now, and the 16-item cap below
          // bounds the heavy readback.
          if (!pBl->input.isWidowModel) continue;
          const RaytraceGeometry& g = pBl->modifiedGeometryData;
          if (!g.positionBuffer.defined()) continue;
          RtxGeometryUtils::HullReadbackItem it;
          it.key     = static_cast<const void*>(pBl);
          it.vtx     = g.vertexCount;
          it.posHash = g.hashes[HashComponents::VertexPosition];
          it.matHash = static_cast<uint64_t>(pBl->input.getMaterialData().getHash());
          // NV-DXVK: same GC/streaming UAF as the HullCensus line ~963 —
          // isImageEmpty()→getImageHash() is not atomic vs the streaming
          // thread freeing m_currentMipView, so getImageHash() can deref a
          // dangling Rc<DxvkImage> (AV in Rc::operator->). This probe is
          // gated off by default (tf2HeavyProbes), but omit the image-hash
          // read here too so enabling it can't reintroduce the crash. mat
          // hash above is a plain value and is safe.
          it.texHash = 0ull;
          it.buffer  = g.positionBuffer.buffer();
          it.offset  = g.positionBuffer.offsetFromSlice();
          it.o2w     = Matrix4(pInst->getTransform());
          items.push_back(it);
          if (items.size() >= 16u) break;
        }
        if (!items.empty()) {
          ctx->getCommonObjects()->metaGeometryUtils().debugReadbackHullWorldPositions(ctx, fid, items);
        }
      }
    }

    if (!vmEnable)
      return;

    if (!vmCamValid)
      return;

    // If the first person player model is enabled, hide the view model.
    if (RtxOptions::PlayerModel::enableInPrimarySpace()) {
      for (auto* candidateInstance : m_viewModelCandidates) {
        candidateInstance->m_vkInstance.mask = 0;
      }
      return;
    }

    const RtCamera& camera = cameraManager.getMainCamera();
    const RtCamera& viewModelCamera = cameraManager.getCamera(CameraType::ViewModel);

    // Use the FOV (XY scaling) from the view-model matrix and the near/far planes (ZW scaling) from the main matrix.
    // The view-model camera has different near/far planes, so if that projection matrix is used naively,
    // the gun ends up being scaled up by a factor of 7 or so (in Portal).
    const auto& mainProjectionMatrix = camera.getViewToProjection();
    auto viewModelProjectionMatrix = viewModelCamera.getViewToProjection();
    viewModelProjectionMatrix[2][2] = mainProjectionMatrix[2][2];
    viewModelProjectionMatrix[2][3] = mainProjectionMatrix[2][3];
    viewModelProjectionMatrix[3][2] = mainProjectionMatrix[3][2];

    const auto& mainPreviousProjectionMatrix = camera.getPreviousViewToProjection();
    auto previousViewModelProjectionMatrix = viewModelCamera.getPreviousViewToProjection();
    previousViewModelProjectionMatrix[2][2] = mainPreviousProjectionMatrix[2][2];
    previousViewModelProjectionMatrix[2][3] = mainPreviousProjectionMatrix[2][3];
    previousViewModelProjectionMatrix[3][2] = mainPreviousProjectionMatrix[3][2];

    // Apply an extra scaling matrix to the view-space positions of view model to make it less likely to interact with world geometry.
    Matrix4d scaleMatrix {};
    scaleMatrix[0][0] = scaleMatrix[1][1] = scaleMatrix[2][2] = RtxOptions::ViewModel::scale();
    scaleMatrix[3][3] = 1.0;

    // Compute the view-model perspective correction matrix.
    // This expression (read right-to-left) is a solution to the following equation:
    //   (mainProjection * mainView * objectToWorld) * transformedPosition = (viewModelProjection * viewModelView * objectToWorld) * position
    // where 'position' is the original vertex data supplied by the game, and 'transformedPosition' is what we need to compute in order to make
    // the view model project into the same screen positions using the main camera.
    // The 'objectToWorld' matrices are applied later, in createViewModelInstance, because they're different per-instance.
    // NV-DXVK [zig-zag fix pt2]: use the MAIN camera's (engine-locked, stable)
    // worldToView for the view part of the correction instead of the ViewModel
    // camera's. [ZigPC] proved vmW2V is stale (updates every ~3 frames, jumps
    // ~130u) while Main advances every frame; that fresh-vs-stale mismatch is
    // what makes pcT (and the gun) sawtooth ±60u. The viewmodel sits at the
    // SAME eye as Main (ZigCam/ZigPC: both at -25.60) so the view matrices
    // should be identical — only the projection (FOV/depth) legitimately
    // differs, and that is already handled by viewModelProjectionMatrix above.
    // Using the stable Main view makes the correction a stable similarity
    // transform (mainV2W * projRemap * mainW2V) instead of a stale-vs-fresh mix.
    const auto perspectiveCorrection = camera.getViewToWorld(false) * (camera.getProjectionToView() * viewModelProjectionMatrix * scaleMatrix) * camera.getWorldToView(false);
    const auto prevPerspectiveCorrection = camera.getPreviousViewToWorld(false) * (camera.getPreviousProjectionToView() * previousViewModelProjectionMatrix * scaleMatrix) * camera.getPreviousWorldToView(false);

    // NV-DXVK [ZigPC]: refT (gun o2w) is smooth after the engine-view fix, but
    // [VM.final] T still oscillates → the residual jerk is in perspectiveCorrection
    // itself (its translation pcT.y sawtooths ±63u). Isolate which of the 4
    // factors wobbles, per frame. Main (camera.*) is engine-locked → mV2W_T /
    // mP2V_XY expected steady. The ViewModel camera is NOT engine-suppressed, so
    // its worldToView (vmW2V_T) and projection (vmProjXY) are the suspects.
    {
      const uint32_t pcFrame = m_device->getCurrentFrameId();
      static uint32_t s_pcLastFrame = UINT32_MAX;
      if (pcFrame != s_pcLastFrame) {
        s_pcLastFrame = pcFrame;
        const auto mV2W  = camera.getViewToWorld(false);
        const auto vmW2V = viewModelCamera.getWorldToView(false);
        const auto mP2V  = camera.getProjectionToView();
        Logger::info(str::format(
          "[ZigPC] f=", pcFrame,
          " mV2W_T=(", mV2W[3][0], ",", mV2W[3][1], ",", mV2W[3][2], ")",
          " vmW2V_T=(", vmW2V[3][0], ",", vmW2V[3][1], ",", vmW2V[3][2], ")",
          " vmProjXY=(", viewModelProjectionMatrix[0][0], ",", viewModelProjectionMatrix[1][1], ")",
          " mP2V_XY=(", mP2V[0][0], ",", mP2V[1][1], ")",
          " pcT=(", perspectiveCorrection[3][0], ",", perspectiveCorrection[3][1], ",", perspectiveCorrection[3][2], ")"));
      }
    }

    // Create any valid view model instances from the list of candidates
    std::vector<RtInstance*> viewModelInstances;
    uint32_t vmSkippedMulticam = 0, vmSkippedNoCam = 0, vmCreated = 0;
    for (auto* candidateInstance : m_viewModelCandidates) {

      // Valid view model instances must be associated only with the view model camera
      // Check: exactly one bit set (power-of-two check via raw bitmask)
      const auto seenMask = candidateInstance->m_seenCameraTypes.raw();
      // [ZigCand] every candidate: its BLAS, vtx count, seenMask, current mask,
      // and whether it'll be skipped. Match blas/vtx against [ZigInst]/[ZigBlas]
      // to find whether the VISIBLE gun is a candidate we actually correct, or a
      // skipped/other instance. Also dumps the reference's pre-hide mask.
      {
        const void* candBlas = (const void*)candidateInstance->getBlas();
        const uint32_t candVtx = candidateInstance->getBlas() ? candidateInstance->getBlas()->modifiedGeometryData.vertexCount : 0u;
        Logger::info(str::format(
          "[ZigCand] f=", fid,
          " cand=", (const void*)candidateInstance,
          " blas=", candBlas, " vtx=", candVtx,
          " seenMask=0x", std::hex, (uint32_t)seenMask,
          " mask=0x", (uint32_t)candidateInstance->m_vkInstance.mask, std::dec,
          " skip=", (seenMask == 0 ? "noCam" : ((seenMask & (seenMask - 1)) != 0 ? "multiCam" : "no"))));
      }
      if (seenMask == 0) { ++vmSkippedNoCam; continue; }
      if ((seenMask & (seenMask - 1)) != 0) { ++vmSkippedMulticam; continue; }

      // Hide the reference instance since we'll create a separate instance for the view model
      candidateInstance->m_vkInstance.mask = 0;

      // Tag the instance as ViewModel so it can be checked for it being a reference view model instance
      candidateInstance->setCustomIndexBit(CUSTOM_INDEX_IS_VIEW_MODEL, true);

      viewModelInstances.push_back(createViewModelInstance(ctx, *candidateInstance, perspectiveCorrection, prevPerspectiveCorrection));
      ++vmCreated;
    }
    Logger::info(str::format(
      "[VM.created] f=", fid,
      " total=", m_viewModelCandidates.size(),
      " created=", vmCreated,
      " skipNoCam=", vmSkippedNoCam,
      " skipMultiCam=", vmSkippedMulticam));

    // Create virtual instances for the view model instances
    createRayPortalVirtualViewModelInstances(viewModelInstances, cameraManager, rayPortalManager);
  }

  static bool isInsidePlayerModel(const Vector3& playerModelPosition, const Vector3& instancePosition) {
    const Vector3 playerToInstance = instancePosition - playerModelPosition;
    const float horizontalDistance = length(Vector2(playerToInstance.x, playerToInstance.y));
    const float verticalDistance = fabs(playerToInstance.z);

    // Distance thresholds determined experimentally to match the portal gun held in player's hands
    // but not match the gun on the pedestals.
    const float maxHorizontalDistance = RtxOptions::PlayerModel::horizontalDetectionDistance();
    const float maxVerticalDistance = RtxOptions::PlayerModel::verticalDetectionDistance();

    return (horizontalDistance <= maxHorizontalDistance) && (verticalDistance <= maxVerticalDistance);
  }

  void InstanceManager::filterPlayerModelInstances(const Vector3& playerModelPosition, const RtInstance* bodyInstance) {
    for (size_t i = 0; i < m_playerModelInstances.size(); ++i) {
      RtInstance* instance = m_playerModelInstances[i];

      // Don't compare the body to itself.
      if (instance == bodyInstance)
        continue;

      if (instance->m_isUnordered) {
        // Particles don't have a valid position in the instance matrix and often combine many particles
        // in one instance. So we rely on the analysis done for billboard creation earlier and see if the billboards
        // intersect with the player model.

        // Start assuming that the instance is actually part of the player model.
        bool isPlayerModelInstance = true;

        if (instance->m_billboardCount > 0) {
          // Check if the billboards are used as intersection primitives. 
          // Note: If one billboard is used as an intersection primitive, all of them are
          if (m_billboards[instance->m_firstBillboard].allowAsIntersectionPrimitive) {
            // If there are billboards, look at their centers, and if any of them are outside of the player model
            // limits, consider the entire instance non-player-model.
            // Opposite approach is possible, too, not entirely sure what's better.
            for (uint32_t billboardIndex = 0; billboardIndex < instance->m_billboardCount; ++billboardIndex) {
              const IntersectionBillboard& billboard = m_billboards[billboardIndex + instance->m_firstBillboard];
              if (!isInsidePlayerModel(playerModelPosition, billboard.center)) {
                isPlayerModelInstance = false;
                break;
              }
            }
          }
        }

        if (isPlayerModelInstance) {
          if (instance->m_billboardCount > 0) {
            // If this instance contains particles and is part of the player model,
            // assign the PLAYER_MODEL mask to its billboards and hide the original instance.
            for (uint32_t billboardIndex = 0; billboardIndex < instance->m_billboardCount; ++billboardIndex) {
              IntersectionBillboard& billboard = m_billboards[billboardIndex + instance->m_firstBillboard];
              billboard.instanceMask = OBJECT_MASK_PLAYER_MODEL;
            }

            instance->getVkInstance().mask = 0;
          }
        } else {
          // Remove the instance from the list to avoid creating virtual instances for it.
          m_playerModelInstances.erase(m_playerModelInstances.begin() + i);
          --i;
        }
      } else {
        const Vector3 instancePosition = instance->getTransform()[3].xyz();

        if (!isInsidePlayerModel(playerModelPosition, instancePosition)) {
          // Note: just use the OPAQUE flag here, which works for Portal with current assets.
          // Might want to apply more complex logic if that is insufficient one day.
          instance->getVkInstance().mask = OBJECT_MASK_OPAQUE;

          // Remove this instance from the player model list.
          m_playerModelInstances.erase(m_playerModelInstances.begin() + i);
          --i;
        }
      }
    }
  }

  void InstanceManager::detectIfPlayerModelIsVirtual(
    const CameraManager& cameraManager,
    const RayPortalManager& rayPortalManager,
    const Vector3& playerModelPosition,
    bool* out_PlayerModelIsVirtual,
    const SingleRayPortalDirectionInfo** out_NearPortalInfo,
    const SingleRayPortalDirectionInfo** out_FarPortalInfo) const {
    auto& rayPortalPair = *rayPortalManager.getRayPortalPairInfos().begin();

    *out_PlayerModelIsVirtual = false;
    int portalIndexForVirtualInstances = -1;

    if (rayPortalPair.has_value()) {

      // Estimate the position of the player model's eyes (where the camera normally is), ignoring crouching.
      // Note that in Portal, the player model is always upright, even if the player is flying out of a floor portal upside down.
      // This makes the detection of whether the player model is virtual more robust.

      Vector3 playerModelEyePosition = playerModelPosition;
      playerModelEyePosition.z += RtxOptions::PlayerModel::eyeHeight();

      // Find the portal that is closest to the model

      float distanceOfModelPortal = FLT_MAX;
      int playerModelNearPortalIndex = 0;

      for (int portalIndex = 0; portalIndex < 2; ++portalIndex) {
        const RayPortalInfo& portalInfo = rayPortalPair->pairInfos[portalIndex].entryPortalInfo;
        const float distanceToModel = length(portalInfo.centroid - playerModelEyePosition);
        if (distanceToModel < distanceOfModelPortal) {
          distanceOfModelPortal = distanceToModel;
          playerModelNearPortalIndex = portalIndex;
        }
      }

      const Vector3& camPos = cameraManager.getCamera(CameraType::Main).getPosition(/* freecam = */ false);

      // Find the portal that the imaginary player (i.e. a blob around the camera, or camera volume) is currently intersecting

      uint32_t cameraVolumePortalIntersectionMask = 0;

      for (uint i = 0; i < 2; i++) {
        const auto& rayPortal = rayPortalPair->pairInfos[i];
        const Vector3 dirToPortalCentroid = rayPortal.entryPortalInfo.centroid - camPos;

        // Approximate the player collision model with this capsule-like shape
        const float maximumNormalDistance = lerp(RtxOptions::PlayerModel::intersectionCapsuleRadius(),
                                                 RtxOptions::PlayerModel::intersectionCapsuleHeight(),
                                                 clamp(rayPortal.entryPortalInfo.planeNormal.z, 0.f, 1.f));

        // Test if that shape intersects with the portal and if the camera is in front of it
        const float planeDistanceNormal = -dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeNormal);
        const float planeDistanceX = dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeBasis[0]);
        const float planeDistanceY = dot(dirToPortalCentroid, rayPortal.entryPortalInfo.planeBasis[1]);
        const bool cameraVolumeIntersectsPortal = 0.f < planeDistanceNormal && planeDistanceNormal < maximumNormalDistance
          && std::abs(planeDistanceX) < rayPortal.entryPortalInfo.planeHalfExtents.x
          && std::abs(planeDistanceY) < rayPortal.entryPortalInfo.planeHalfExtents.y;

        if (cameraVolumeIntersectsPortal) {
          portalIndexForVirtualInstances = i;
          cameraVolumePortalIntersectionMask |= (1 << i);
        }
      }

      // If the camera volume intersects exactly one portal, and the player model is closer to another portal,
      // that must mean the game is rendering the model at the other side of a portal (i.e. the player model is virtual/ghost).
      // This excludes the case when the camera intersects both portals.
      // De-virtualize the player model using the same portal that was used to virtualize it.
      const int playerModelFarPortalIndex = !playerModelNearPortalIndex;
      // Additional heuristic that tells if the player model eyes become closer to the camera if it's de-virtualized.
      // Fixes false virtual player model detections when there is one portal on a wall and another on the floor right next to it,
      // and you stand between these portals (see TREX-2254).
      const float playerModelEyeDistanceToCamera = length(playerModelEyePosition - camPos);
      const Vector3 devirtualizedPlayerModelEyePosition = (rayPortalPair->pairInfos[playerModelNearPortalIndex].portalToOpposingPortalDirection * Vector4(playerModelEyePosition, 1.f)).xyz();
      const float devirtualizedPlayerModelEyeDistanceToCamera = length(devirtualizedPlayerModelEyePosition - camPos);
      if (cameraVolumePortalIntersectionMask == (1 << playerModelFarPortalIndex) && devirtualizedPlayerModelEyeDistanceToCamera < playerModelEyeDistanceToCamera) {
        *out_PlayerModelIsVirtual = true;
        portalIndexForVirtualInstances = !portalIndexForVirtualInstances;
      }
      // In other (regular) situations, if the camera volume intersects at least one volume, make sure to use
      // the same portal for virtual player model as the one used for the virtual view model,
      // to avoid inconsistencies in tracing.
      else if (m_virtualInstancePortalIndex >= 0 && portalIndexForVirtualInstances >= 0) {
        portalIndexForVirtualInstances = m_virtualInstancePortalIndex;
      }
    }

    *out_NearPortalInfo = (portalIndexForVirtualInstances >= 0) ? &rayPortalPair->pairInfos[portalIndexForVirtualInstances] : nullptr;
    *out_FarPortalInfo = (portalIndexForVirtualInstances >= 0) ? &rayPortalPair->pairInfos[!portalIndexForVirtualInstances] : nullptr;
  }

  void InstanceManager::createPlayerModelVirtualInstances(Rc<DxvkContext> ctx, const CameraManager& cameraManager, const RayPortalManager& rayPortalManager) {
    if (m_playerModelInstances.empty())
      return;

    // Sometimes, the game renders the player model on the other side of the portal
    // that is closest to the camera. To detect that, we look at the model position.
    // Here, we also detect the instances of the portal gun that are rendered in the world
    // using the same mesh and texture as the held portal gun but should not be considered
    // a part of the player model. Those are detected by comparing their position to the body.
    
    // Find the instance marked with the "playerBody" material
    const RtInstance* bodyInstance = nullptr;
    for (RtInstance* instance : m_playerModelInstances) {
      if (instance->testCategoryFlags(InstanceCategories::ThirdPersonPlayerBody))
        bodyInstance = instance;
    }

    if (!bodyInstance)
      return;

    // Get the position from the transform matrix - works for Portal
    Vector3 playerModelPosition = bodyInstance->getTransform()[3].xyz();

    // Detect instances that are too far away from the body, make them regular objects.
    // This fixes the guns placed on pedestals to be picked up.
    filterPlayerModelInstances(playerModelPosition, bodyInstance);

    // Detect if the player model rendered by the game is virtual or not
    bool playerModelIsVirtual = false;
    // Near portal is where the original instance is
    const SingleRayPortalDirectionInfo* nearPortalInfo = nullptr;
    // Far portal is where the cloned instance will be
    const SingleRayPortalDirectionInfo* farPortalInfo = nullptr;
    detectIfPlayerModelIsVirtual(cameraManager, rayPortalManager, playerModelPosition, &playerModelIsVirtual, &nearPortalInfo, &farPortalInfo);
        
    const uint32_t frameId = m_device->getCurrentFrameId();

    // Set up the math to offset the player model backwards if it's to be shown in primary space
    float backwardOffset = RtxOptions::PlayerModel::backwardOffset();

    const bool createVirtualInstances = RtxOptions::PlayerModel::enableVirtualInstances() && (nearPortalInfo != nullptr);

    // The loop below creates virtual instances and applies the offset. Exit if neither is necessary.
    if (!createVirtualInstances && backwardOffset == 0.f)
      return;

    // Calculate the offset vector
    Vector3 backwardOffsetVector = cameraManager.getMainCamera().getHorizontalForwardDirection();
    backwardOffsetVector *= -backwardOffset;

    if (playerModelIsVirtual && farPortalInfo) {
      // Transform the offset vector into portal space
      backwardOffsetVector = (farPortalInfo->portalToOpposingPortalDirection * Vector4(backwardOffsetVector, 0.f)).xyz();
    }

    const Matrix4 backwardOffsetMatrix {
      Vector4{ 1.f, 0.f, 0.f, 0.f },
      Vector4{ 0.f, 1.f, 0.f, 0.f },
      Vector4{ 0.f, 0.f, 1.f, 0.f },
      Vector4(backwardOffsetVector, 1.f)
    };
    
    // Create virtual instances for player model instances that are close to portals.
    // Offset both real and virtual instances by backwardOffset units if enabled.
    for (RtInstance* originalInstance : m_playerModelInstances) {

      if (backwardOffset != 0.f) {
        // Offset the original instance
        originalInstance->teleportWithHistory(backwardOffsetMatrix);

        // Offset the original instance particles
        for (uint32_t i = 0; i < originalInstance->m_billboardCount; ++i) {
          m_billboards[originalInstance->m_firstBillboard + i].center += backwardOffsetVector;
        }
      }

      if (!createVirtualInstances)
        continue;
      
      // Don't pollute global instance id with Player Models since they're not tracked in game capturer
      const bool needValidGlobalInstanceId = false;

      RtInstance* clonedInstance = createInstanceCopy(*originalInstance, needValidGlobalInstanceId);
      
      clonedInstance->setFrameCreated(frameId);
      clonedInstance->setFrameLastUpdated(frameId);

      // Cloned player model instances are recreated every frame
      clonedInstance->markForGarbageCollection();

      // Compute the instance masks for both original and cloned instances.
      // When the original instance is real (which is the case normally), the cloned one is virtual and located on the other side of a portal.
      // When the original instance is virtual (rendered by the game on the other side of a portal), the cloned one is not.
      const uint32_t originalInstanceMask = playerModelIsVirtual ? OBJECT_MASK_PLAYER_MODEL_VIRTUAL : OBJECT_MASK_PLAYER_MODEL;
      const uint32_t clonedInstanceMask = playerModelIsVirtual ? OBJECT_MASK_PLAYER_MODEL : OBJECT_MASK_PLAYER_MODEL_VIRTUAL;

      if (originalInstance->m_billboardCount > 0) {
        // If this is a translucent instance with billboards, clone the billboards and hide the original instance.
        
        // Allocate some billboard entries first
        clonedInstance->m_firstBillboard = m_billboards.size();
        clonedInstance->m_billboardCount = originalInstance->m_billboardCount;
        m_billboards.resize(m_billboards.size() + originalInstance->m_billboardCount);

        // Copy the billboards to the new location and patch them
        for (uint32_t i = 0; i < originalInstance->m_billboardCount; ++i) {
          IntersectionBillboard* originalBillboard = &m_billboards[originalInstance->m_firstBillboard + i];
          IntersectionBillboard* clonedBillboard = &m_billboards[clonedInstance->m_firstBillboard + i];

          *clonedBillboard = *originalBillboard;
          clonedBillboard->instance = clonedInstance;

          // Update the instance masks of both instances
          originalBillboard->instanceMask = originalInstanceMask;
          clonedBillboard->instanceMask = clonedInstanceMask;

          // Update the center.
          // The orientation is irrelevant because the GPU will re-derive it for each ray.
          clonedBillboard->center = (nearPortalInfo->portalToOpposingPortalDirection * Vector4(originalBillboard->center, 1.0f)).xyz();
        }

        // Hide the geometric instances but keep them in the list so that surface data is generated for them.
        originalInstance->m_vkInstance.mask = 0;
        clonedInstance->m_vkInstance.mask = 0;
      }
      else {
        // Update the instance masks of both instances
        originalInstance->m_vkInstance.mask = originalInstanceMask;
        clonedInstance->m_vkInstance.mask = clonedInstanceMask;
      }
      
      // Update cloned instance transforms given the reference and the portal transform
      {
        clonedInstance->teleportWithHistory(nearPortalInfo->portalToOpposingPortalDirection);
      }

      // Use a clip plane to make sure that the cloned instance doesn't stick through a slab
      // that the other portal might be placed on.
      clonedInstance->surface.isClipPlaneEnabled = true;
      clonedInstance->surface.clipPlane = Vector4(farPortalInfo->entryPortalInfo.planeNormal,
        -dot(farPortalInfo->entryPortalInfo.planeNormal, farPortalInfo->entryPortalInfo.centroid));
      // Use the FORCE_NO_OPAQUE flag to enable any-hit processing in the visiblity rays for this clipped instance.
      clonedInstance->m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;

      // Same clip plane logic for the original instance, only using the near portal.
      originalInstance->surface.isClipPlaneEnabled = true;
      originalInstance->surface.clipPlane = Vector4(nearPortalInfo->entryPortalInfo.planeNormal,
        -dot(nearPortalInfo->entryPortalInfo.planeNormal, nearPortalInfo->entryPortalInfo.centroid));
      originalInstance->m_vkInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    }
  }

  void InstanceManager::findPortalForVirtualInstances(const CameraManager& cameraManager, const RayPortalManager& rayPortalManager) {
    m_virtualInstancePortalIndex = -1;

    // Virtual instances for the view model and the player model are generated for the closest portal to the camera.

    static_assert(maxRayPortalCount == 2);
    auto& rayPortalPair = *rayPortalManager.getRayPortalPairInfos().begin();

    if (!rayPortalPair.has_value())
      return;

    const Vector3& camPos = cameraManager.getCamera(CameraType::Main).getPosition(/* freecam = */ false);

    const float kMaxDistanceToPortal = RtxOptions::ViewModel::rangeMeters() * RtxOptions::getMeterToWorldUnitScale();

    // Find the closest valid portal to generate the instances for since we can generate 
    // virtual instances only for one of the portals due to instance mask bit allocation.
    // This will result in missing virtual viewModel geo for some corner cases, 
    // such as when portals are close to each other in a corner arrangement
    float minDistanceToPortal = FLT_MAX;

    for (uint i = 0; i < 2; i++) {
      const auto& rayPortal = rayPortalPair->pairInfos[i];
      const Vector3 dirToPortalCentroid = rayPortal.entryPortalInfo.centroid - camPos;
      const float distanceToPortal = length(dirToPortalCentroid);

      if (distanceToPortal <= kMaxDistanceToPortal &&
          distanceToPortal < minDistanceToPortal) {
        minDistanceToPortal = distanceToPortal;
        m_virtualInstancePortalIndex = rayPortal.entryPortalInfo.portalIndex;
      }
    }

  }

  void InstanceManager::createRayPortalVirtualViewModelInstances(const std::vector<RtInstance*>& viewModelReferenceInstances,
                                                                 const CameraManager& cameraManager,
                                                                 const RayPortalManager& rayPortalManager) {
    // Early out if there is no eligible portal
    if (m_virtualInstancePortalIndex < 0)
      return;

    if (rayPortalManager.getRayPortalPairInfos().empty()) {
      assert(!"There must be a portal pair in createRayPortalVirtualViewModelInstances if m_virtualInstancePortalIndex is defined");
      return;
    }

    if (!RtxOptions::ViewModel::enableVirtualInstances())
      return;

    const SingleRayPortalDirectionInfo& closestPortalInfo = rayPortalManager.getRayPortalPairInfos()[0]->pairInfos[m_virtualInstancePortalIndex];
    
    const uint32_t frameId = m_device->getCurrentFrameId();

    // Create virtual instances for view model instances that are close to portals
    for (RtInstance* referenceInstance : viewModelReferenceInstances) {

      // Create a view model virtual instance corresponding to the view model instance, for one frame

      // Don't pollute global instance id with View Models since they're not tracked in game capturer
      const bool needValidGlobalInstanceId = false;

      RtInstance* virtualInstance = createInstanceCopy(*referenceInstance, needValidGlobalInstanceId);

      virtualInstance->setFrameCreated(frameId);
      virtualInstance->setFrameLastUpdated(frameId);

      // Virtual view model instances are recreated every frame
      virtualInstance->markForGarbageCollection();

      // Virtual instances are to be visible only in their corresponding portal spaces
      static_assert(maxRayPortalCount == 2);
      // View model virtual instance
      virtualInstance->m_vkInstance.mask = OBJECT_MASK_VIEWMODEL_VIRTUAL;
    
      // Update virtual instance transforms given the reference and the portal transform
      {
        virtualInstance->teleportWithHistory(closestPortalInfo.portalToOpposingPortalDirection);
      }

      // Note this is an instance copy of an input reference. It is unknown to the source engine, so we don't call onInstanceAdded callbacks for it
      // It also results in this instance not being linked to reference instance BLAS and thus not considered in findSimilarInstances' lookups
      // This is desired as ViewModel instances are not to be linked frame to frame
    }
  }

  void InstanceManager::resetSurfaceIndices() {
    for (auto instance : m_instances)
      instance->m_surfaceIndex = SURFACE_INDEX_INVALID;
  }

  inline bool isFpSpecial(float x) {
    const uint32_t u = *(uint32_t*) &x;
    return (u & 0x7f800000) == 0x7f800000;
  }

  void InstanceManager::createBillboards(RtInstance& instance, const Vector3& cameraViewDirection)
  {
    const RasterGeometry& geometryData = instance.getBlas()->input.getGeometryData();

    constexpr uint32_t indicesPerQuad = 6;

    // Check if this is a supported geometry first
    if (geometryData.indexCount < indicesPerQuad || 
        (geometryData.indexCount % indicesPerQuad) != 0 ||
        geometryData.indexBuffer.indexType() != VK_INDEX_TYPE_UINT16 ||
        geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      return;
    
    const GeometryBufferData bufferData(geometryData);

    // Check if the necessary buffers exist
    // Warning: do not generate billboards for instances without indices as other code sections using billboards expect indices to be present
    if (!bufferData.indexData || !bufferData.positionData || !bufferData.texcoordData)
      return;

    const bool hasNonIdentityTextureTransform = instance.surface.textureTransform != Matrix4();
    bool bSuccess = true;
    bool areAllBillboardsValidIntersectionCandidates = true;
    uint32_t billboardCount = 0;
    instance.m_firstBillboard = m_billboards.size();

    const Matrix4 instanceTransform = instance.getTransform();

    // Go over all quads in this draw call.
    // Note: decals are often batched into a few draw calls, and we want to offset each decal separately.
    for (int indexOffset = 0; indexOffset + indicesPerQuad <= geometryData.indexCount; indexOffset += indicesPerQuad) {
      // Load indices for a quad
      uint16_t indices[indicesPerQuad];
      for (size_t idx = 0; idx < indicesPerQuad; ++idx) {
        indices[idx] = bufferData.getIndex(idx + indexOffset);
      }


      // Make sure that these indices follow a known quad pattern: A, B, C, A, C, D
      // If they don't, we can't process this "quad" - so, cancel the whole instance.
      if (indices[0] != indices[3] || indices[2] != indices[4]) {
        ONCE(Logger::info("[RTX] InstanceManager: detected unsupported quad index layout for billboard creation"));
        // This quad is incompatible altogether. Abort processing billboards for this instance and skip billboard processing for it
        bSuccess = false;
        break;
      }
      
      // Load data for a triangle
      Vector3 positions[3];
      Vector2 texcoords[4];
      uint8_t vertexOpacities8bit[4] = {};
      
      for (size_t idx = 0; idx < 3; ++idx) {
        const uint16_t currentIndex = indices[idx];

        Vector4 objectSpacePosition = Vector4(bufferData.getPosition(currentIndex), 1.0f);

        positions[idx] = (instanceTransform * objectSpacePosition).xyz();

        texcoords[idx] = bufferData.getTexCoord(currentIndex);

        if (hasNonIdentityTextureTransform)
          texcoords[idx] = (instance.surface.textureTransform * Vector4(texcoords[idx].x, texcoords[idx].y, 0.f, 1.f)).xy();

        if (bufferData.vertexColorData)
          vertexOpacities8bit[idx] = bufferData.getVertexColor(indices[idx]) >> 24;
      }

      // Load one vertex color - assuming that the entire billboard uses the same color
      uint32_t vertexColor = ~0u;
      if (bufferData.vertexColorData)
        vertexColor = bufferData.getVertexColor(indices[0]);

      // Compute the normal
      const Vector3 xVector { positions[2] - positions[1] };
      const Vector3 yVector { positions[1] - positions[0] };
      const Vector3 center { (positions[2] + positions[0]) * 0.5f };

      IntersectionBillboard billboard;

      const bool centerIsSpecial = isFpSpecial(center.x) || isFpSpecial(center.y) || isFpSpecial(center.z);
      if (centerIsSpecial) {
        areAllBillboardsValidIntersectionCandidates = false;
      }

      const float xLength = length(xVector);
      const float yLength = length(yVector);
      const float dotAxes = dot(xVector, yVector) / (xLength * yLength);
      // Note: This could probably be handled in a better way (like skipping this quad) rather than just assigning
      // a fallback normal, but this is simple enough.
      const Vector3 normal = safeNormalize(cross(xVector, yVector), Vector3(0.0f, 0.0f, 1.0f));
      const float normalDotCamera = dot(normal, cameraViewDirection);


      // Limit the set of particles that are turned into intersection primitives:
      // - Must be roughly square
      const bool isSquare = xLength <= yLength * 1.5f && yLength <= xLength * 1.5f;
      // - The original quad must have perpendicular sides
      const bool hasPerpendicularSides = std::abs(dotAxes) < 0.01f;
      // - Must be in the camera view plane, i.e. only auto-oriented particles, not world-space ones
      //   (except player model particles, which are oriented towards the camera and not in the view plane)
      const bool isInViewPlane = std::abs(normalDotCamera) > 0.99f;
      // Assume that all billboards on the player model are camera facing
      const bool isCameraFacing = instance.m_isPlayerModel;
      if (!isSquare || !hasPerpendicularSides || !isInViewPlane && !isCameraFacing) {
        areAllBillboardsValidIntersectionCandidates = false;
      }

      const Vector2 xVectorUV { texcoords[2] - texcoords[1] };
      const Vector2 yVectorUV { texcoords[1] - texcoords[0] };
      const Vector2 centerUV { (texcoords[2] + texcoords[0]) * 0.5f };

      // Fill in data for the quad's last/4th vertex
      texcoords[3] = bufferData.getTexCoord(indices[5]);
      if (bufferData.vertexColorData)
        vertexOpacities8bit[3] = bufferData.getVertexColor(indices[5]) >> 24;

      billboard.center = center;
      billboard.xAxis = xVector / xLength;
      billboard.width = xLength;
      billboard.yAxis = yVector / yLength;
      billboard.height = yLength;
      billboard.xAxisUV = xVectorUV * 0.5f;
      billboard.yAxisUV = yVectorUV * 0.5f;
      billboard.centerUV = centerUV;
      billboard.instance = &instance;
      billboard.vertexColor = vertexColor;
      billboard.instanceMask = instance.getVkInstance().mask & OBJECT_MASK_UNORDERED_ALL_INTERSECTION_PRIMITIVE;
      billboard.texCoordHash = XXH64(texcoords, sizeof(texcoords), kEmptyHash);
      billboard.vertexOpacityHash = XXH64(vertexOpacities8bit, sizeof(vertexOpacities8bit), kEmptyHash);
      billboard.allowAsIntersectionPrimitive = true;
      billboard.isBeam = false;
      billboard.isCameraFacing = isCameraFacing;
      m_billboards.push_back(billboard);
      ++billboardCount;
    }

    if (bSuccess) {
      instance.m_billboardCount = billboardCount;

      if (areAllBillboardsValidIntersectionCandidates) {
        // Update the instance mask to hide it from rays that look only for intersection billboards.
        instance.getVkInstance().mask &= OBJECT_MASK_UNORDERED_ALL_GEOMETRY;
      } else {
        // Disable the rest of the billboards as intersection primitives since only a single mask can be used
        // per instance
        for (uint32_t i = m_billboards.size() - instance.m_billboardCount; i < m_billboards.size(); i++) {
          IntersectionBillboard& billboard = m_billboards[i];
          billboard.allowAsIntersectionPrimitive = false;
        }
      }
    } else {
      // Revert the billboards that were created successfully before the first failure,
      // because one of the failed to be created
      m_billboards.erase(m_billboards.end() - billboardCount, m_billboards.end());
    }
  }

  void InstanceManager::createBeams(RtInstance& instance) {
    const RasterGeometry& geometryData = instance.getBlas()->input.getGeometryData();

    // Check if this is a supported geometry first
    if (geometryData.indexCount < 4 ||
        (geometryData.indexCount % 2) != 0 ||
        geometryData.indexBuffer.indexType() != VK_INDEX_TYPE_UINT16 ||
        geometryData.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
      return;

    const GeometryBufferData bufferData(geometryData);

    // Check if the necessary buffers exist
    if (!bufferData.indexData || !bufferData.positionData || !bufferData.texcoordData)
      return;

    // Extract the beams from the triangle strip.
    // Start by loading the first 2 indices.
    uint16_t indices[4];
    indices[0] = bufferData.getIndex(0);
    indices[1] = bufferData.getIndex(1);

    for (int index = 2; index < geometryData.indexCount - 1; index += 2) {
      // When there are multiple beams packed into one triangle strip, they are separated
      // by a pair of repeating indices, such as: (0 1 2 3) 3 4 (4 5 6 7)
      // We want to keep looking at indices until either the end of the strip is reached,
      // or until we detect such a repeating pair. In the latter case, we skip the pair
      // at the end of this loop.
      const bool endOfStrip = index >= geometryData.indexCount - 2;
      const bool restart = !endOfStrip && (bufferData.getIndex(index + 1) == bufferData.getIndex(index + 2));

      if (!endOfStrip && !restart)
        continue;

      // Load the indices of the last 2 vertices of the beam.
      indices[2] = bufferData.getIndex(index);
      indices[3] = bufferData.getIndex(index + 1);

      // Load the source data for the 4 vertices that define our beam.
      Vector3 positions[4];
      Vector2 texcoords[4];
      for (int i = 0; i < 4; ++i) {
        positions[i] = bufferData.getPosition(indices[i]);
        texcoords[i] = bufferData.getTexCoord(indices[i]);
      }

      // Load one vertex color - assuming that the entire beam uses the same color
      uint32_t vertexColor = ~0u;
      if (bufferData.vertexColorData)
        vertexColor = bufferData.getVertexColor(indices[0]);

      // Extract the beam cylinder axis, length and width from the vertices.
      // Note that the 4 vertices are not necessarily coplanar: the beam is tessellated
      // in the axial direction, and each segment is rotated separately to face the camera.
      // The vertices are laid out in a triangle strip order:
      //     0-2
      //  -- |/| --> axis
      //     1-3
      const Vector3 startPosition = (positions[0] + positions[1]) * 0.5f;
      const Vector3 endPosition = (positions[2] + positions[3]) * 0.5f;
      const float beamWidth = length(positions[1] - positions[0]);
      const float beamLength = length(endPosition - startPosition);

      // Fill out the billboard struct.
      IntersectionBillboard billboard;
      billboard.center = (startPosition + endPosition) * 0.5f;
      billboard.xAxis = normalize(positions[1] - positions[0]);
      billboard.width = beamWidth;
      billboard.yAxis = normalize(endPosition - startPosition);
      billboard.height = beamLength;
      billboard.xAxisUV = (texcoords[1] - texcoords[0]) * 0.5f;
      billboard.yAxisUV = (texcoords[2] - texcoords[0]) * 0.5f;
      billboard.centerUV = (texcoords[0] + texcoords[3]) * 0.5f;
      billboard.vertexColor = vertexColor;
      billboard.instanceMask = instance.getVkInstance().mask & OBJECT_MASK_UNORDERED_ALL_INTERSECTION_PRIMITIVE;
      billboard.instance = &instance;
      billboard.texCoordHash = 0;
      billboard.vertexOpacityHash = 0;
      billboard.allowAsIntersectionPrimitive = true;
      billboard.isBeam = true;
      billboard.isCameraFacing = false;
      m_billboards.push_back(billboard);

      // If there are enough vertices left in the strip to fit one more beam, after the separator pair,
      // skip the separator and load the first two indices of the next beam.
      if (index <= geometryData.indexCount - 8) {
        index += 4;
        indices[0] = bufferData.getIndex(index);
        indices[1] = bufferData.getIndex(index + 1);
      }
    }

    instance.getVkInstance().mask &= OBJECT_MASK_UNORDERED_ALL_GEOMETRY;

    // Note: setting the instance's billboardCount to 0 here because we don't need either of the uses of that count:
    // - Beams cannot be parts of a player model;
    // - Beams should not be split into quads for OMM reuse.
    instance.m_billboardCount = 0;
  }

  const XXH64_hash_t RtInstance::calculateAntiCullingHash() const {
    if (RtxOptions::AntiCulling::isObjectAntiCullingEnabled()) {
      const Vector3 pos = getWorldPosition();
      const XXH64_hash_t posHash = XXH3_64bits(&pos, sizeof(pos));
      XXH64_hash_t antiCullingHash = XXH3_64bits_withSeed(&m_materialDataHash, sizeof(XXH64_hash_t), posHash);

      if (RtxOptions::AntiCulling::Object::hashInstanceWithBoundingBoxHash() &&
          RtxOptions::needsMeshBoundingBox()) {
        const AxisAlignedBoundingBox& boundingBox = getBlas()->input.getGeometryData().boundingBox;
        const XXH64_hash_t bboxHash = boundingBox.calculateHash();
        antiCullingHash = XXH3_64bits_withSeed(&bboxHash, sizeof(antiCullingHash), antiCullingHash);
      }
      return antiCullingHash;
    }

    return XXH64_hash_t();
  }
}  // namespace dxvk
