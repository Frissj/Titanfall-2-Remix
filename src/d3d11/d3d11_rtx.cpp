#include "d3d11_rtx.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <string>

#include "d3d11_vanish_diag.h"
#include <filesystem>
#include <set>
#include <sstream>
#include "../util/config/config.h"
#include "../util/util_env.h"

#ifdef _WIN32
// RtlCaptureStackBackTrace — pulled in for the bone-diag stack trace below.
#  include <windows.h>
#endif
#include <chrono>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

// NV-DXVK TF2 BONE CAPTURE: global mirror populated by DxvkContext::copyBuffer
// when the game bulk-uploads rig matrices via staging→t30 copies. The
// D3D11 UpdateSubresource hook below only catches per-palette updates;
// those bulk copies are the source of upper-half bone data. Merge
// this mirror into m_fullBoneCache during skinning capture.
namespace dxvk { namespace tf2 {
  extern std::mutex g_boneCacheMirrorMutex;
  extern std::vector<uint8_t> g_boneCacheMirror;
  extern bool g_boneCacheMirrorPopulated;
}}
// All bone-diag state lives in dxvk_bone_diag.h.
#include "../dxvk/dxvk_bone_diag.h"

// NV-DXVK [pilot-eye-capture]: extern declarations of the atomic mirrors
// the d3d11 producer writes whenever the viewmodel-pass cb2 RDEF read
// produces a new c_cameraOrigin. Definitions live in rtx_camera_manager.cpp
// (libdxvk side) because the consumer (CameraManager::processCameraData)
// also lives there and the d3d11 → dxvk link direction means dxvk can't
// pull symbols out of d3d11.
namespace dxvk { namespace tf2 {
  extern std::atomic<float> g_pilotEyeX;
  extern std::atomic<float> g_pilotEyeY;
  extern std::atomic<float> g_pilotEyeZ;
  extern std::atomic<bool>  g_pilotEyeValid;
}}

// Include dxvk_device.h before any rtx headers so that dxvk_buffer.h and
// sibling headers (included bare by rtx_utils.h) are already in the TU.
#include "../dxvk/dxvk_device.h"

#include "d3d11_context.h"
#include "d3d11_buffer.h"
#include "d3d11_input_layout.h"
#include "d3d11_device.h"
#include "d3d11_vs_classifier.h"
#include "d3d11_view_srv.h"
#include "d3d11_sampler.h"
#include "d3d11_depth_stencil.h"
#include "d3d11_blend.h"
#include "d3d11_rasterizer.h"

#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_point_instancer_system.h"
#include "../dxvk/rtx_render/rtx_materials.h"
#include "../dxvk/rtx_render/rtx_debug_view.h"
#include "../dxvk/rtx_render/rtx_camera.h"
#include "../dxvk/rtx_render/rtx_camera_manager.h"
#include "../dxvk/rtx_render/rtx_scene_manager.h"
#include "../dxvk/rtx_render/rtx_light_manager.h"
#include "../dxvk/rtx_render/rtx_bloom.h"
#include "../dxvk/rtx_render/rtx_matrix_helpers.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// NV-DXVK: scene dumper. Writes per-instance world-space triangles to
// scene_dump.obj for offline inspection. Triggered automatically once the game
// has been rendering for >5 seconds. Currently focused on BSP-style
// (R32G32_UINT packed position + g_modelInst SRV) draws.
namespace SceneDump {
  static std::ofstream g_obj;
  static std::mutex    g_mutex;
  static uint32_t      g_baseVtx       = 0;
  static uint32_t      g_objectsWritten = 0;
  static bool          g_done          = false;
  static std::chrono::steady_clock::time_point g_firstFrameTime;
  static bool          g_armed         = false;

  static const char* const kOutPath =
    "C:/Users/Friss/Downloads/Compressed/Titanfall-2-Digital-Deluxe-Edition-AnkerGames/Titanfall2/scene_dump.obj";

  static void armOnFirstGameplayFrame(uint32_t rawDraws) {
    if (g_done || g_armed) return;
    if (rawDraws < 50) return;  // skip menu frames
    g_firstFrameTime = std::chrono::steady_clock::now();
    g_armed = true;
  }
  static bool shouldDumpThisFrame() {
    // NV-DXVK: disabled. Capture path kept compiled for easy re-enable.
    return false;
  }
  static void open() {
    if (!g_obj.is_open()) {
      g_obj.open(kOutPath, std::ios::out | std::ios::trunc);
      if (g_obj.is_open()) {
        g_obj << "# Titanfall 2 BSP scene dump\n";
        dxvk::Logger::info(dxvk::str::format("[SceneDump] writing to ", kOutPath));
      } else {
        dxvk::Logger::err(dxvk::str::format("[SceneDump] FAILED to open ", kOutPath));
      }
    }
  }
  // Emit a small unit-cube at (0,0,0) — that's where the camera lives in this
  // dump's coordinate frame (geometry is camera-relative). Lets you eyeball
  // distance from camera to BSP chunks in Blender/MeshLab.
  static void writeCameraMarker() {
    if (!g_obj.is_open()) return;
    g_obj << "o CAMERA\n";
    const float s = 8.0f; // unit cube edge half-size in world units
    static const float corners[8][3] = {
      {-s,-s,-s},{ s,-s,-s},{ s, s,-s},{-s, s,-s},
      {-s,-s, s},{ s,-s, s},{ s, s, s},{-s, s, s},
    };
    for (int i = 0; i < 8; ++i)
      g_obj << "v " << corners[i][0] << " " << corners[i][1] << " " << corners[i][2] << "\n";
    const uint32_t b = g_baseVtx + 1;
    // 12 triangles via 6 quads — splitting each into two
    static const int faces[12][3] = {
      {0,1,2},{0,2,3},  {4,6,5},{4,7,6},
      {0,4,5},{0,5,1},  {2,6,7},{2,7,3},
      {1,5,6},{1,6,2},  {0,3,7},{0,7,4},
    };
    for (int i = 0; i < 12; ++i)
      g_obj << "f " << (b+faces[i][0]) << " " << (b+faces[i][1]) << " " << (b+faces[i][2]) << "\n";
    g_baseVtx += 8;
    ++g_objectsWritten;
  }
  static void close() {
    if (g_obj.is_open()) {
      g_obj.close();
      g_done = true;
      dxvk::Logger::info(dxvk::str::format(
        "[SceneDump] done. ", g_objectsWritten, " objects, ", g_baseVtx, " vertices"));
    }
  }
  static inline uint32_t decodeX(uint32_t u0)              { return u0 & 0x001FFFFFu; }
  static inline uint32_t decodeY(uint32_t u0, uint32_t u1) { return ((u0 >> 21) & 0x7FFu) | ((u1 & 0x3FFu) << 11u); }
  static inline uint32_t decodeZ(uint32_t u1)              { return u1 >> 10; }
}

namespace dxvk {

  // NV-DXVK [VanishDiag-A2Hook]: trampoline-captured per-frame state
  // from engine.dll!R_DrawWorldMeshes. The struct pointer (`a2`) is
  // saved into g_vanishDiagCapturedA2, and the first 8 qwords of the
  // dynamic bucket bitmask `[a2+0x54088]` are SNAPSHOTTED to
  // g_vanishDiagBitmaskSnap right at the call site — because by
  // EndFrame time the per-frame allocation may have been freed/reused
  // and reading [a2+0x54088] returns zeros. The trampoline does
  // `rep movsq` of 8 qwords from the live bitmask into the snapshot
  // every time R_DrawWorldMeshes is invoked. The EndFrame logger then
  // reads from the snapshot.
  volatile uint64_t g_vanishDiagCapturedA2 = 0;
  volatile uint64_t g_vanishDiagBitmaskSnap[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

  // NV-DXVK [VanishDiag-A3]: third arg of R_DrawWorldMeshes (the flag word
  // built by the caller — bits select per-pass filter, e.g. main vs shadow,
  // depth-only, etc.). The per-bucket filter inside R_DrawWorldMeshes builds
  // v7 from a3 bits; if a3 differs between visible and vanish frames, the
  // filter rejects different buckets even though the WorldVis bitmask is
  // identical. Captured each call; last-fire-wins.
  volatile uint32_t g_vanishDiagCapturedA3 = 0;

  // NV-DXVK [VanishDiag-BuildBatches]: snapshots from BuildWorldMeshBatches'
  // output (same WriterStruct as R_DrawWorldMeshes' a2). Captured by the
  // R_DrawWorldMeshes trampoline since it runs AFTER BuildWorldMeshBatches.
  //
  // Per-pass end indices: a2[+0..+12] are 4 u32 values written by
  // sub_1800B6FB0 — pass 0 end, pass 1 end, pass 2 end, pass 3 end. If
  // a bucket index exceeds the current pass's range, sub_1800B6FB0 advances
  // to the next pass; if it overflows pass 2 it drops the remaining buckets
  // (LABEL_24). Visible vs vanish diff in these values would indicate the
  // pass-range overflow drop path is the cull mechanism.
  //
  // Batch count: a2[+0x8010] = final number of batch entries written by
  // sub_1800B6FB0. If the visible frame produces N batches and the vanish
  // produces N-12, that's the smoking gun.
  volatile uint32_t g_buildBatchesPassEnds[4] = { 0, 0, 0, 0 };
  volatile uint32_t g_buildBatchesBatchCount = 0;

  // NV-DXVK [VanishDiag-B84C0]: captures from sub_1800B84C0 (engine.dll
  // RVA 0xB84C0) — the per-pass draw-list submit function called from
  // R_DrawWorldMeshes. Inputs:
  //   a1 (rcx) = WriterStruct (with draw list at +16 + 16*idx)
  //   a2 (edx) = filter mask (= v7 = 0x60 typical)
  //   a3 (r8d) = pass index (0..3)
  // Pass-range values: a1[a3] = start, a1[a3+1] = end. Range size = end-start
  // = number of draw entries this pass processes. If range differs between
  // visible and vanish frames, BuildWorldMeshBatches dropped draws upstream.
  // If range is identical, sub_1800B84C0's per-draw filter is the cull.
  volatile uint64_t g_b84c0_a1 = 0;
  volatile uint32_t g_b84c0_filter_mask = 0;
  volatile uint32_t g_b84c0_pass_idx = 0;
  volatile uint32_t g_b84c0_range_start = 0;
  volatile uint32_t g_b84c0_range_end = 0;
  volatile uint32_t g_b84c0_call_count = 0;

  // NV-DXVK [VanishDiag-PropCull]: captures from sub_1801B2200 (engine.dll
  // RVA 0x1B2200) — the static-prop visibility gatherer that distance-culls
  // each prop. We capture the camera-state inputs to its cull formula:
  //   sceneScale = a1[+0x50048]   (the formula's denominator scale)
  //   cam(X,Y,Z) = a1[+0x4FFDC..+0x4FFE4]
  //
  // Hypothesis: in Remix's pipeline, sceneScale comes in based on the
  // downscaled internal render resolution rather than the upscaled output,
  // shrinking the effective cull distance. If we see sceneScale << 1.0
  // (or much smaller than expected for the rendering target), that's the
  // Remix-bug confirmation. The fix would override sceneScale at this
  // function's input (or stretch the cull-distance multiplier).
  volatile uint64_t g_propCull_a1 = 0;
  volatile float    g_propCull_sceneScale = 0.0f;
  volatile float    g_propCull_camX = 0.0f;
  volatile float    g_propCull_camY = 0.0f;
  volatile float    g_propCull_camZ = 0.0f;
  volatile uint32_t g_propCull_callCount = 0;

  // 256-slot ring buffer. With the |camX|>4000 main-view filter, only
  // ~main-view-shaped sub_1801B2200 calls write here, so 256 slots covers
  // ~4 seconds of history at 60fps even with multiple main-view passes per
  // frame. That comfortably spans the user walking through a visible→vanish
  // transition, so a single P-press at the end captures both sides.
  // Each slot is 24 bytes so we can compute slot_addr via lea*3 + shl 3.
  // Trampoline: head++; slot = ring[head & 255]; write a1 + 4 floats.
#pragma pack(push, 1)
  struct PropCullSlot {
    uint64_t a1;          // +0
    float    sceneScale;  // +8
    float    camX;        // +12
    float    camY;        // +16
    float    camZ;        // +20
  };
#pragma pack(pop)
  static_assert(sizeof(PropCullSlot) == 24, "PropCullSlot must be 24 bytes for trampoline math");
  static constexpr uint32_t kPropCullRingSize = 256;
  static constexpr uint32_t kPropCullRingMask = kPropCullRingSize - 1;
  volatile PropCullSlot g_propCullRing[kPropCullRingSize] = {};
  volatile uint32_t g_propCullRingHead = 0;

  // NV-DXVK [VanishDiag-PropCullDecision]: per-prop cull DECISION snapshot,
  // captured by a trampoline at engine.dll RVA 0x1B2476 (the `ja → loc_top`
  // after the `comiss xmm3, xmm0` distance/radius compare).
  //
  // At the hook point we have:
  //   r15d  = global prop index (= bucket*64 + bit)
  //   rbx   = pointer to prop struct (208 bytes)
  //   xmm3  = adj_dist²    = dist² × (1/sceneScale²)
  //   xmm0  = thresh       = (max(1, render[+0x28]) × prop[+0x40])²
  //                          × ((zoom+1)² × 0.996 − 0.004)
  //   EFLAGS still set from the comiss compare; CF=0 && ZF=0 → ja taken (cull)
  //
  // Slot is 32 bytes (power of two, fast lea+shl):
#pragma pack(push, 1)
  struct PropCullDecisionSlot {
    uint32_t propIdx;     // +0
    uint32_t cullFlag;    // +4   1 if culled (ja taken), 0 if kept
    float    adjDistSq;   // +8   xmm3
    float    thresh;      // +12  xmm0
    float    propX;       // +16  [rbx+0x34]
    float    propY;       // +20  [rbx+0x38]
    float    propZ;       // +24  [rbx+0x3C]
    float    propRadius;  // +28  [rbx+0x40]
  };
#pragma pack(pop)
  static_assert(sizeof(PropCullDecisionSlot) == 32,
                "PropCullDecisionSlot must be 32 bytes for trampoline math");
  static constexpr uint32_t kPropCullDecisionRingSize = 4096;
  static constexpr uint32_t kPropCullDecisionRingMask = kPropCullDecisionRingSize - 1;
  volatile PropCullDecisionSlot g_propCullDecisionRing[kPropCullDecisionRingSize] = {};
  volatile uint32_t g_propCullDecisionRingHead = 0;

  // NV-DXVK [VanishDiag-DispatchDecision]: per-entry dispatcher decision
  // captured at engine.dll RVA 0x1B32ED — the `je 0x1801B35BE` after
  // `and esi, eax` (where eax=entry[+0xD], esi=entry[+0xE]^-1, so
  // skip when (entry[+0xD] & ~entry[+0xE]) == 0). This is the per-entry
  // filter inside sub_1801B31E0's iteration over view+0x8028. Captures
  // the entry's raw 16 bytes plus the skip decision so we can identify
  // which floor entries are being filtered out.
#pragma pack(push, 1)
  struct DispatchDecisionSlot {
    uint64_t modelPtr;     // +0   entry[0..7]
    uint64_t entryHi;      // +8   entry[8..F] (packed bytes/words including masks)
    uint64_t entryAddr;    // +16  &entry (= view + 0x8028 + idx*16)
    uint32_t skipFlag;     // +24  1 if engine would skip (je taken), 0 otherwise
    float    viewCamX;     // +28  view[+0x5003C] for filter verification
  };
#pragma pack(pop)
  static_assert(sizeof(DispatchDecisionSlot) == 32,
                "DispatchDecisionSlot must be 32 bytes for trampoline math");
  static constexpr uint32_t kDispatchDecisionRingSize = 4096;
  static constexpr uint32_t kDispatchDecisionRingMask = kDispatchDecisionRingSize - 1;
  volatile DispatchDecisionSlot g_dispatchDecisionRing[kDispatchDecisionRingSize] = {};
  volatile uint32_t g_dispatchDecisionRingHead = 0;

  // [VanishDiag-HitCounters] per-trampoline hit counters. Each trampoline
  // starts with `inc dword [rip+disp32]` (6 bytes, ~1 cycle, non-atomic;
  // count slop from races is fine for telemetry). Logged in the auto-dump.
  volatile uint32_t g_hitsEntryHook        = 0;
  volatile uint32_t g_hitsCullJumpHook     = 0;
  volatile uint32_t g_hitsBitmaskLoadHook  = 0;
  volatile uint32_t g_hitsDispatchMFence   = 0;

  // [VanishDiag-DispatchCapture] when 0, the dispatch trampoline skips
  // the ring-write block. Was set OFF in v15 for perf, but turning it
  // OFF empirically broke the floor fix — the ring writes themselves
  // (the explicit `mov rax,[r14]` etc.) appear to be the load-bearing
  // mechanism, likely via cache-coherency side effects that force the
  // engine's stale entry-byte reads to be re-fetched. Default ON now;
  // the ring-write cost (~50 cy × ~2.5k hits/frame ≈ 50µs/frame) is
  // acceptable, and the writes go to a 4096-slot ring that gets
  // overwritten so memory pressure is bounded. End key still toggles.
  volatile uint32_t g_dispatchCaptureEnabled = 1;

  // NV-DXVK [VanishDiag-ForceBitmask]: when set to 1, the sub_1801B2200
  // hooks force the per-view bitmask reads to all-ones for main view.
  // Two hooks share this flag:
  //   1. Entry-time force-fill (writes 0xFF... to bitmask memory)
  //   2. Load-time OR -1 at 0x1B23D6 (overrides each `mov rdx,[rax+r8*8]`)
  // The load-time hook is the load-bearing one — it survives any code
  // between function entry and the bitmask read that might rewrite the
  // bitmask. OFF by default — when ON, force-fill exposes invalid prop
  // bits in the last (partial) word, corrupting the dispatch list and
  // breaking world geometry. The load-time hook now masks the last word
  // (skips the OR if r8 == wordCount-1) so partial-word slop is bounded
  // to that one word, but keep default OFF so normal gameplay isn't
  // affected unless the user explicitly enables via Home.
  volatile uint32_t g_forceMainViewBitmask = 1;  // v57: ON by default — body
                                                  // force-fill is the real fix.
                                                  // Home key still toggles it.

  // ================================================================
  // ENGINE PATCH TOGGLES — flip any to false to disable the patch.
  // All gating is compile-time (constexpr): zero runtime cost when on,
  // patch sites short-circuit before VirtualProtect when off. Original
  // engine.dll bytes are left untouched. Set the master kEnableEnginePatches
  // to false to turn EVERYTHING off in one go regardless of individual
  // toggles.
  // ================================================================
  namespace tf2patches {
    static constexpr bool kEnableEnginePatches      = false;  // master OFF

    // Static byte patches in engine.dll
    static constexpr bool kPatchVertexBudget        = true;  // +0xB7100  (3.1M-vert cap → 0x7FFFFFFF)
    static constexpr bool kPatchEntityMaskGate      = true;  // +0x730DA  (jz → nop, force OR)
    static constexpr bool kPatchDispatchEntryE      = true;  // +0x1B32DF (movzx → xor esi,esi)

    // Trampolines in engine.dll
    static constexpr bool kHookSubB2200             = true;  // sub_1801B2200 prologue (v57 fix)
    static constexpr bool kHookRDrawWorldMeshes     = true;  // R_DrawWorldMeshes
    static constexpr bool kHookSub1800B45D0         = true;  // OR-site hook
    static constexpr bool kHookSub18036BD30         = true;
    static constexpr bool kHookSub1802EB290         = true;
    static constexpr bool kHookSub1800B84C0         = true;

    // All-on resolution: AND with master toggle for one-shot disable.
    template <bool individual>
    static constexpr bool kPatchActive = kEnableEnginePatches && individual;
  }

  // NV-DXVK [VanishDiag-Stack]: capture call-stack at OnDraw* when a target
  // VS hash matches one of the floor's vertex shaders (per scene_dump CSV
  // diff: VS_2947 lost -29 draws, VS_29D5 lost -17, VS_28EA lost -13). The
  // stack reveals the immediate TF2 caller — which is the actual cull
  // decision point. F9 dumps and resets all three slots.
  struct VanishStackSlot {
    uint64_t vsHash;
    uint32_t frameCount;
    uint64_t frames[16];
  };
  static constexpr uint64_t k_VanishStackTargets[3] = {
    0x2947c6346103a2dbULL,
    0x29d58573f42e22fdULL,
    0x28ea29dae516dbd7ULL,
  };
  static VanishStackSlot g_vanishStack[3] = {};
  volatile uint32_t g_vanishStackTotalHits = 0;

  // NV-DXVK [VanishDiag-GlobalSnap]: parallel snapshot of engine.dll's
  // qword_192205120 (RVA 0x12205120) — the global "bucket dirty" bitmask
  // ORed by sub_1800B45D0 during BVH traversal. Comparing this against
  // g_vanishDiagBitmaskSnap tells us whether [a2+0x54088] is aliased to
  // qword_192205120 (same memory, just two views) or built independently
  // by an external caller (in which case we need to look elsewhere).
  volatile uint64_t g_vanishDiagGlobalSnap[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

  // NV-DXVK [VanishDiag-BucketHist]: histogram of bucket indices visited
  // by sub_1800B45D0's OR-into-qword_192205120 site. Trampoline at engine.dll
  // RVA 0xB4870 increments [v17] each call. Per-frame the EndFrame logger
  // reads [401] specifically (our floor bucket) and total nonzero count,
  // then resets to 0. Sized 1024 entries; trampoline bounds-checks against
  // this constant to avoid OOB if engine ever feeds a larger index.
  volatile uint32_t g_vanishDiagBucketHist[1024] = { 0 };

  // NV-DXVK [VanishDiag-B30Hook]: per-call captures from a trampoline
  // installed at client.dll!sub_18036BD30 (RVA 0x36BD30) — the wrapper
  // that takes (this, view_ctx, source_bitmask) and tail-calls
  // sub_1802EF230 to memmove source_bitmask → WriterStruct[+0x54088].
  //
  // This is the per-view bitmask copy site. Each call passes a different
  // source_bitmask address (per-item / per-view, persistent across frames
  // for that item). The hook captures `r8` (source_bitmask) and `rdx`
  // (view_ctx) plus the first 8 qwords of the source bitmask data. Across
  // frames we'll see calls for shadow views (small ~13 bits) and the main
  // world view (~67 bits) — once we identify the main view's source
  // bitmask address, that's a stable target for a HW write BP to find
  // the actual writer of the bitmask data without per-frame allocation
  // churn (the buffer pointer is per-item, not per-frame).
  //
  // Last-fire-wins single-slot capture (race conditions tolerable for
  // diagnostics — we sample many frames and grep for high-bit entries).
  volatile uint64_t g_b30_view_ctx = 0;
  volatile uint64_t g_b30_source_bm = 0;
  volatile uint64_t g_b30_snap[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  volatile uint32_t g_b30_call_count = 0;

  // NV-DXVK [VanishDiag-EB290]: per-bucket histogram of calls to
  // client.dll!sub_1802EB290 — the per-bucket visibility/frustum test
  // invoked from sub_1802EB1E0 (the JT-job that ORs bits into the
  // main view's WriterStruct bitmask). Each call passes a bucket index
  // (a2 = rdx). The function returns 0 for culled, 1 for visible.
  //
  // Cross-referenced against [VanishDiag-WorldVis]'s WorldVis bitmask:
  //   - hist[i] > 0 AND bit i SET in WorldVis  →  bucket tested, passed
  //   - hist[i] > 0 AND bit i CLEAR in WorldVis →  bucket tested, REJECTED
  //     (this identifies which buckets the visibility test culls)
  //   - hist[i] == 0                            →  bucket pre-filtered out
  //     (qword_181748DD0 had its bit clear)
  //
  // Reset each frame; sized 2048 to bound-check against game's max
  // bucket count (typical scenes have ~500).
  volatile uint32_t g_eb290_hist[2048] = { 0 };
  volatile uint32_t g_eb290_call_count = 0;

  // Map D3D11_BLEND → VkBlendFactor.  Mirrors D3D11BlendState::DecodeBlendFactor
  // but kept local to avoid exposing internal statics.
  static VkBlendFactor mapD3D11Blend(D3D11_BLEND b, bool isAlpha) {
    switch (b) {
      case D3D11_BLEND_ZERO:              return VK_BLEND_FACTOR_ZERO;
      case D3D11_BLEND_ONE:               return VK_BLEND_FACTOR_ONE;
      case D3D11_BLEND_SRC_COLOR:         return VK_BLEND_FACTOR_SRC_COLOR;
      case D3D11_BLEND_INV_SRC_COLOR:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
      case D3D11_BLEND_SRC_ALPHA:         return VK_BLEND_FACTOR_SRC_ALPHA;
      case D3D11_BLEND_INV_SRC_ALPHA:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      case D3D11_BLEND_DEST_ALPHA:        return VK_BLEND_FACTOR_DST_ALPHA;
      case D3D11_BLEND_INV_DEST_ALPHA:    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      case D3D11_BLEND_DEST_COLOR:        return VK_BLEND_FACTOR_DST_COLOR;
      case D3D11_BLEND_INV_DEST_COLOR:    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
      case D3D11_BLEND_SRC_ALPHA_SAT:     return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
      case D3D11_BLEND_BLEND_FACTOR:      return isAlpha ? VK_BLEND_FACTOR_CONSTANT_ALPHA : VK_BLEND_FACTOR_CONSTANT_COLOR;
      case D3D11_BLEND_INV_BLEND_FACTOR:  return isAlpha ? VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
      case D3D11_BLEND_SRC1_COLOR:        return VK_BLEND_FACTOR_SRC1_COLOR;
      case D3D11_BLEND_INV_SRC1_COLOR:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
      case D3D11_BLEND_SRC1_ALPHA:        return VK_BLEND_FACTOR_SRC1_ALPHA;
      case D3D11_BLEND_INV_SRC1_ALPHA:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
      default:                            return VK_BLEND_FACTOR_ONE;
    }
  }

  // Map D3D11_BLEND_OP → VkBlendOp.
  static VkBlendOp mapD3D11BlendOp(D3D11_BLEND_OP op) {
    switch (op) {
      case D3D11_BLEND_OP_ADD:          return VK_BLEND_OP_ADD;
      case D3D11_BLEND_OP_SUBTRACT:     return VK_BLEND_OP_SUBTRACT;
      case D3D11_BLEND_OP_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
      case D3D11_BLEND_OP_MIN:          return VK_BLEND_OP_MIN;
      case D3D11_BLEND_OP_MAX:          return VK_BLEND_OP_MAX;
      default:                          return VK_BLEND_OP_ADD;
    }
  }

  // NV-DXVK: static definitions (shared across all D3D11Rtx instances).
  bool D3D11Rtx::m_foundRealProjThisFrame = false;
  bool D3D11Rtx::m_hasEverFoundProj       = false;
  DrawCallTransforms D3D11Rtx::m_lastGoodTransforms = {};
  std::mutex D3D11Rtx::m_lastGoodTransformsMutex;

  D3D11Rtx::D3D11Rtx(D3D11DeviceContext* pContext)
    : m_context(pContext) {}

  Rc<DxvkSampler> D3D11Rtx::getDefaultSampler() const {
    if (m_defaultSampler == nullptr) {
      // NV-DXVK: default sampler for Remix fallback paths. The D3D11 spec
      // default (CLAMP_TO_EDGE) is wrong for world-surface sampling: Source
      // engine BSP stores UVs in world units (U=32.5, V=64.1 etc.), and
      // CLAMP collapses every sample to the edge texel → flat colour on
      // every textured wall. REPEAT is the correct default for BSP/prop
      // content; UI/decal draws that actually need CLAMP will provide their
      // own sampler via PSSetSamplers, which we prefer when present.
      DxvkSamplerCreateInfo info;
      info.magFilter      = VK_FILTER_LINEAR;
      info.minFilter      = VK_FILTER_LINEAR;
      info.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      info.mipmapLodBias  = 0.0f;
      info.mipmapLodMin   = -1000.0f;
      info.mipmapLodMax   =  1000.0f;
      info.useAnisotropy  = VK_FALSE;
      info.maxAnisotropy  = 1.0f;
      info.addressModeU   = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.addressModeV   = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.addressModeW   = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.compareToDepth = VK_FALSE;
      info.compareOp      = VK_COMPARE_OP_NEVER;
      info.borderColor    = VkClearColorValue{};
      info.usePixelCoord  = VK_FALSE;
      m_defaultSampler = m_context->m_device->createSampler(info);
    }
    return m_defaultSampler;
  }

  void D3D11Rtx::Initialize() {
    // Scale geometry workers to available cores (min 2, max 6).
    // D3D11 games typically have high draw call counts, so more workers pay off.
    const uint32_t cores = std::max(2u, std::thread::hardware_concurrency());
    const uint32_t workers = std::min(std::max(cores / 2, 2u), 6u);
    m_pGeometryWorkers = std::make_unique<GeometryProcessor>(workers, "d3d11-geometry");

    // --- D3D11 sensible defaults (Default layer = lowest priority) ---
    // Written to the Default layer so rtx.conf, user.conf, and all other
    // config layers override them naturally.  Without this, setDeferred()
    // writes to the Derived layer (priority 5) which stomps rtx.conf (priority 3)
    // and makes per-game config files useless.
    const RtxOptionLayer* defaults = RtxOptionLayer::getDefaultLayer();

    // FusedWorldViewMode::View tells Remix to treat objectToView as the full
    // local-to-view transform.  In commitGeometryToRT it sets:
    //   objectToWorld = objectToView   (fused transform)
    //   worldToView   = identity       (camera at origin)
    // This works because we bake the worldToView into objectToView via
    // objectToView = worldToView * objectToWorld (line ~1787).  The camera
    // position is encoded in worldToView's translation, so after the fuse
    // geometry is centred near origin (camera-relative) and the RT camera
    // at origin sees it correctly.
    RtxOptions::fusedWorldViewModeObject().setDeferred(FusedWorldViewMode::View, defaults);

    // Anti-culling: D3D11 engines aggressively frustum-cull objects before
    // issuing draw calls.  Without anti-culling, off-screen objects vanish
    // from reflections, shadows, and GI.
    RtxOptions::AntiCulling::Object::enableObject().setDeferred(true, defaults);
    RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCullingObject().setDeferred(true, defaults);
    RtxOptions::AntiCulling::Object::numObjectsToKeepObject().setDeferred(20000u, defaults);
    RtxOptions::AntiCulling::Object::fovScaleObject().setDeferred(2.0f, defaults);
    RtxOptions::AntiCulling::Object::farPlaneScaleObject().setDeferred(10.0f, defaults);
    RtxOptions::AntiCulling::Light::enableObject().setDeferred(true, defaults);

    // Use incoming vertex buffers directly (skip copy to staging → saves VRAM + bandwidth).
    RtxOptions::useBuffersDirectlyObject().setDeferred(true, defaults);

    // --- Fallback lighting ---
    // D3D11 has no legacy lighting API — all lighting is shader-driven,
    // so Remix never receives explicit light definitions from the application.
    // Force the fallback light to Always so the scene is lit even if there are
    // no Remix USD light assets placed yet.  Use a bright distant light that
    // produces reasonable illumination for most indoor/outdoor scenes.
    LightManager::fallbackLightModeObject().setDeferred(LightManager::FallbackLightMode::Always, defaults);
    LightManager::fallbackLightTypeObject().setDeferred(LightManager::FallbackLightType::Sphere, defaults);
    // NV-DXVK: sphere-light fallback tuned for TF2 world-unit scale (1 unit
    // ≈ 1 inch). Sphere is co-located at the camera so inverse-square
    // falloff is minimal across the player's immediate view — everywhere you
    // look gets lit. Large radius + high radiance compensates for the camera
    // being effectively inside the sphere (emission from the far side of
    // the sphere surface still reaches nearby geometry). Direction / angle
    // entries are kept for the Distant variant but ignored for Sphere.
    LightManager::fallbackLightRadianceObject().setDeferred(Vector3(500.0f, 500.0f, 500.0f), defaults);
    LightManager::fallbackLightRadiusObject().setDeferred(36.0f, defaults);
    LightManager::fallbackLightPositionOffsetObject().setDeferred(Vector3(0.0f, 0.0f, 0.0f), defaults);
    // NV-DXVK: suppress bloom halos on walls. Remix's default bloom threshold
    // (0.25 linear) was tuned for game outputs that already include bloom
    // baked in; Remix's fallback-lit pixels easily exceed it, producing
    // visible glow halos around lit walls. Raise threshold so only true
    // emissives (lightmaps, muzzle flashes, screens) trigger bloom.
    DxvkBloom::luminanceThresholdObject().setDeferred(5.0f, defaults);
    LightManager::fallbackLightDirectionObject().setDeferred(Vector3(-0.3f, -1.0f, 0.5f), defaults);
    LightManager::fallbackLightAngleObject().setDeferred(5.0f, defaults);

    // Start with the Remix developer menu visible so users can verify Remix is
    // active.  showUI has NoSave flag, so rtx.conf cannot override it through
    // the normal config layer path — setting it here on Default ensures the
    // menu is open on first frame.  Alt+X toggle still works (writes to Derived).
    RtxOptions::showUIObject().setDeferred(UIType::Advanced, defaults);
  }

  // NV-DXVK: return-value helper. True = skip D3D11 native rasterization
  // (RT already owns the output), false = emit native raster via EmitCs.
  // Once Remix is active we normally suppress ALL subsequent raster to avoid
  // the "shared RT target" write hazards documented on m_remixActiveThisFrame,
  // BUT UI/HUD draws must rasterize natively or they never appear — the RT
  // composite doesn't include them. So: if THIS draw was RT-captured, or
  // the frame already had RT activity AND this draw was NOT UI-classified,
  // return true. If this draw was filtered as UI, always return false so
  // the native raster path runs (and the HUD shows up).
  // NOTE: RT's final blit in rtx_context.cpp:725 currently copies the RT
  // output OVER the backbuffer, which can still clobber the UI pixels that
  // native raster just wrote. Full fix requires either deferring UI emits
  // past injectRTX or a masked composite; this change is the necessary-
  // but-not-sufficient first step.
  // NV-DXVK: D3D11SwapChain::PresentImage calls this every frame with its
  // m_swapImage so MaybeEarlyInjectForUITexture has a target to hand to
  // injectRTX. The DxvkImage under m_swapImage is stable until the chain
  // is recreated on resize; cheap-to-invoke every present, only logs on
  // actual change.
  // Forward decl — definition is further down (alongside the cache parser)
  // because it depends on findRtxConfPath / parseHashListInto helpers that
  // live with the cache struct definition.
  static void ensureD3D11UiHashCacheLoaded();

  void D3D11Rtx::SetSwapchainBackbuffer(const Rc<DxvkImage>& backbuffer) {
    // Lazy-load the d3d11-local UI-hash cache here — this is the first
    // point in the DLL's lifetime where both (a) the exe path is stable
    // and (b) we're definitely off the DllMain / static-init thread.
    // std::call_once inside guarantees the parse runs exactly once per
    // process even though PresentImage fires every frame.
    ensureD3D11UiHashCacheLoaded();

    const DxvkImage* oldRaw = m_cachedBackbuffer.ptr();
    const DxvkImage* newRaw = backbuffer.ptr();
    if (oldRaw == newRaw)
      return;
    m_cachedBackbuffer = backbuffer;
    Logger::info(str::format(
      "[D3D11Rtx.UITex] swap-image cached: img=0x",
      std::hex, reinterpret_cast<uintptr_t>(newRaw), std::dec,
      " extent=", newRaw ? newRaw->info().extent.width  : 0u, "x",
                  newRaw ? newRaw->info().extent.height : 0u));
  }


  // NV-DXVK: d3d11.dll-local cache for the three UI-texture option sets.
  //
  // We can't call RtxOptions::Create() here to populate d3d11.dll's own
  // copies of the RtxOption<fast_unordered_set> statics — that fires
  // applyPendingValues(forceOnChange=true) against d3d11.dll's
  // singleton, which in turn invokes every registered onChange callback.
  // Many of those callbacks expect a valid DxvkInstance context that
  // only exists in dxgi.dll; fired in d3d11.dll they set up state that
  // collapses main-menu FPS to ~3 (observed 280-300ms per frame for
  // ~1-draw/frame Titanfall 2 menu).
  //
  // So instead of going through RtxOptions at all in d3d11.dll, parse
  // just the three entries we care about out of rtx.conf directly and
  // cache them in this DLL's own static maps. MaybeEarlyInjectForUITexture
  // and LogPsHashesForHudFilter consult these instead of
  // RtxOptions::xxx(). The dxgi.dll side keeps using RtxOptions normally
  // — this cache only affects the HUD deferral decision inside d3d11.dll.
  struct D3D11UiHashCache {
    fast_unordered_set uiTextures;
    fast_unordered_set uiVertexShaderHashes;
    fast_unordered_set uiPixelShaderHashes;
    bool initialized = false;
  };
  static D3D11UiHashCache g_d3d11UiHashCache;

  // Search candidate paths match resolveConfigPaths in rtx_option_layer.cpp
  // (see that function for the rationale on <gameRoot>/rtx-remix/...).
  static std::filesystem::path findRtxConfPath() {
    std::error_code ec;
    const std::filesystem::path exeDir =
        std::filesystem::path(env::getExePath()).parent_path();
    const std::array<std::filesystem::path, 6> candidates = {
      std::filesystem::path("rtx.conf"),
      std::filesystem::path("rtx-remix") / "rtx.conf",
      exeDir / "rtx.conf",
      exeDir / "rtx-remix" / "rtx.conf",
      exeDir.parent_path() / "rtx.conf",
      exeDir.parent_path() / "rtx-remix" / "rtx.conf",
    };
    for (const auto& p : candidates) {
      if (!p.empty() && std::filesystem::exists(p, ec))
        return p;
    }
    return std::filesystem::path();
  }

  // Parse a comma-separated "0xHEX,0xHEX,..." value into the target set.
  // Malformed entries silently skip (matches HashSetLayer::parseFromStrings).
  static void parseHashListInto(const std::string& value, fast_unordered_set& outSet) {
    std::stringstream ss(value);
    std::string s;
    while (std::getline(ss, s, ',')) {
      size_t start = s.find_first_not_of(" \t\r\n");
      size_t end   = s.find_last_not_of(" \t\r\n");
      if (start == std::string::npos)
        continue;
      const std::string trimmed = s.substr(start, end - start + 1);
      try {
        const XXH64_hash_t h = std::stoull(
          trimmed[0] == '-' ? trimmed.substr(1) : trimmed, nullptr, 16);
        if (trimmed[0] != '-')
          outSet.insert(h);
      } catch (const std::exception&) {
        // malformed — skip silently
      }
    }
  }

  // Lazy one-time population of g_d3d11UiHashCache from whatever rtx.conf
  // the candidate-search finds. Called from SetSwapchainBackbuffer (first
  // primary present) so the search runs only after the exe path is stable
  // and the filesystem is accessible. Safe to call from multiple threads
  // — std::call_once guards the parse.
  static void ensureD3D11UiHashCacheLoaded() {
    static std::once_flag sOnce;
    std::call_once(sOnce, []() {
      const auto path = findRtxConfPath();
      if (path.empty()) {
        Logger::info("[D3D11Rtx.UITex] d3d11-local cache: no rtx.conf found on any candidate path");
        g_d3d11UiHashCache.initialized = true;
        return;
      }
      Logger::info(str::format(
        "[D3D11Rtx.UITex] d3d11-local cache: parsing ",
        path.generic_string()));
      const Config cfg = Config::getOptionLayerConfig(path.string());
      const auto& opts = cfg.getOptions();
      auto take = [&](const char* key, fast_unordered_set& dst) {
        auto it = opts.find(key);
        if (it == opts.end()) return;
        parseHashListInto(it->second, dst);
      };
      take("rtx.uiTextures",            g_d3d11UiHashCache.uiTextures);
      take("rtx.uiVertexShaderHashes",  g_d3d11UiHashCache.uiVertexShaderHashes);
      take("rtx.uiPixelShaderHashes",   g_d3d11UiHashCache.uiPixelShaderHashes);
      g_d3d11UiHashCache.initialized = true;
      Logger::info(str::format(
        "[D3D11Rtx.UITex] d3d11-local cache loaded:"
        " uiTextures=",           g_d3d11UiHashCache.uiTextures.size(),
        " uiVertexShaderHashes=", g_d3d11UiHashCache.uiVertexShaderHashes.size(),
        " uiPixelShaderHashes=",  g_d3d11UiHashCache.uiPixelShaderHashes.size()));
    });
  }

  // NV-DXVK: "Standard Remix way" UI handling (was missing from this
  // port). Called from SubmitDraw before the filter cascade so it runs on
  // EVERY draw including ones that will later get filtered. Scans the
  // bound PS SRVs; if any image hash appears in the d3d11-local
  // uiTextures cache, emits injectRTX into the main CS chunk. Once per
  // frame only — the idempotency guard in rtx_context.cpp:491 would turn
  // later calls into no-ops anyway, but skipping them avoids redundant
  // lambda emission.
  //
  // The subsequent native-raster EmitCs the caller will do for this (and
  // every following HUD) draw lands AFTER the injectRTX lambda in the
  // main CS chunk, so on the CS thread the order becomes:
  //   [pre-HUD captures] → [injectRTX + RT blit] → [HUD native draws]
  // and the HUD composites on top of the RT scene. Because this gates on
  // a user-declared hash set, post-process passes (tone-map, bloom, etc.)
  // are NOT pulled into the deferred region — they keep running before
  // injectRTX in the main chunk as they always did.
  // NV-DXVK: Reconstruct the first 16 hex chars of a shader's SHA1 as a
  // 64-bit value. That's the form the [D3D11Rtx.UITex] HUD-filter log
  // prints (e.g. 0xd69c3951f050e757) — so a hash the user grabs from the
  // log and drops into rtx.uiVertexShaderHashes / rtx.uiPixelShaderHashes
  // hits here as the same bitpattern.
  //
  // Sha1Hash::dword(i) reads the SHA1 bytes little-endian, but
  // Sha1Hash::toString() prints them big-endian. We bswap to match
  // toString's ordering.
  static XXH64_hash_t sha1HashPrefix64(const Sha1Hash& sha1) {
    auto bswap32 = [](uint32_t x) {
      return ((x >> 24) & 0x000000FFu)
           | ((x >>  8) & 0x0000FF00u)
           | ((x <<  8) & 0x00FF0000u)
           | ((x << 24) & 0xFF000000u);
    };
    const uint64_t high = bswap32(sha1.dword(0));
    const uint64_t low  = bswap32(sha1.dword(1));
    return (high << 32) | low;
  }

  // NV-DXVK: Fetch the (VS, PS) shader hash prefix for the currently bound
  // shaders. Returns 0 for slots with no shader bound, which will never
  // appear in a user's rtx.uiXxxShaderHashes set, so the match is safe.
  // Member of D3D11Rtx because m_state on the device context is protected
  // and only friends (D3D11Rtx is one) can reach in.
  void D3D11Rtx::GetCurrentVsPsHashes(XXH64_hash_t& outVs, XXH64_hash_t& outPs) const {
    outVs = 0;
    outPs = 0;
    if (const auto* vsPtr = m_context->m_state.vs.shader.ptr()) {
      if (const auto* cs = vsPtr->GetCommonShader()) {
        const auto& sh = cs->GetShader();
        if (sh != nullptr)
          outVs = sha1HashPrefix64(sh->getShaderKey().sha1());
      }
    }
    if (const auto* psPtr = m_context->m_state.ps.shader.ptr()) {
      if (const auto* cs = psPtr->GetCommonShader()) {
        const auto& sh = cs->GetShader();
        if (sh != nullptr)
          outPs = sha1HashPrefix64(sh->getShaderKey().sha1());
      }
    }
  }

  void D3D11Rtx::MaybeEarlyInjectForUITexture() {
    if (m_earlyInjectFiredThisFrame)
      return;
    if (m_cachedBackbuffer.ptr() == nullptr)
      return;

    // NV-DXVK: Critical gate — only fire early-inject AFTER at least one
    // gameplay draw has been captured into the RT scene this frame. Source
    // engine does HUD-shader-prep / pre-frame UI work very early in a
    // frame (log shows rawSoFar=2 drawsSoFar=0 matches on first HUD draw)
    // — if we inject at that point the CS-thread injectRTX lambda runs
    // before any commitGeometryToRT has populated the scene manager, and
    // injectRTX's camera-validity check fails. That alone would be a
    // harmless no-op, BUT injectRTX calls commitGraphicsState +
    // texture-manager work + various setup BEFORE its camera-invalid
    // early-return (rtx_context.cpp:458+), and those side effects
    // poison subsequent commitGeometryToRT calls so the camera never
    // latches valid even once gameplay draws arrive. Net result: scene
    // black, only HUD visible.
    //
    // Requiring m_remixActiveThisFrame means we only fire on HUD draws
    // that Source emits AFTER the gameplay pass (the VGUI batches at
    // end-of-frame which are the ones the user actually wants to land on
    // top of the RT image).  Frames where no gameplay was captured (menu,
    // loading) never fire early-inject at all — same behaviour as the
    // original code before this fix.
    if (!m_remixActiveThisFrame) {
      static uint64_t sSkipCount = 0;
      if ((sSkipCount++ & 0xFF) == 0) {
        Logger::info(str::format(
          "[D3D11Rtx.UITex] skipping early-inject: no captures yet this"
          " frame (rawSoFar=", m_rawDrawCount, " drawsSoFar=", m_drawCallID, ")"));
      }
      return;
    }

    // d3d11-local cache, populated from rtx.conf at first PresentImage.
    // NOT RtxOptions::xxx() — see D3D11UiHashCache comment for why we
    // can't use the normal option accessors from this DLL without
    // triggering the menu-slowdown.
    const auto& uiTexHashes = g_d3d11UiHashCache.uiTextures;
    const auto& uiVsHashes  = g_d3d11UiHashCache.uiVertexShaderHashes;
    const auto& uiPsHashes  = g_d3d11UiHashCache.uiPixelShaderHashes;

    // NV-DXVK: One-shot size probe — confirms the d3d11-local cache has
    // the user's rtx.conf entries by the time the first HUD-class draw
    // reaches us.  Pre-cache-load (race between D3D11CoreCreateDevice and
    // first HUD draw) shows zeros; after the first PresentImage call
    // lazy-loads the cache (see ensureD3D11UiHashCacheLoaded) subsequent
    // draws see the populated sets.
    {
      static bool sLoggedSizes = false;
      if (!sLoggedSizes) {
        sLoggedSizes = true;
        Logger::info(str::format(
          "[D3D11Rtx.UITex] probe (first MaybeEarlyInject, d3d11-local cache):"
          " cacheInit=",      (g_d3d11UiHashCache.initialized ? 1 : 0),
          " uiTex=",          uiTexHashes.size(),
          " uiVs=",           uiVsHashes.size(),
          " uiPs=",           uiPsHashes.size()));
        // Sample up to 3 hashes from the VS set if any exist, so we can
        // confirm content reached us (vs just seeing size=0).
        uint32_t n = 0;
        std::string sample;
        for (auto it = uiVsHashes.begin(); it != uiVsHashes.end() && n < 3; ++it, ++n) {
          char buf[24]; std::snprintf(buf, sizeof(buf), "0x%016llx",
                                       static_cast<unsigned long long>(*it));
          if (n > 0) sample += ",";
          sample += buf;
        }
        Logger::info(str::format(
          "[D3D11Rtx.UITex] uiVsHashes sample=[", sample, "]"));
      }
    }

    if (uiTexHashes.empty() && uiVsHashes.empty() && uiPsHashes.empty())
      return;

    // Record which classifier caused the match so the log is actionable.
    enum class Trigger { None, Texture, VertexShader, PixelShader };
    Trigger      triggerKind = Trigger::None;
    XXH64_hash_t triggerHash = 0;
    uint32_t     triggerSlot = UINT32_MAX;

    // Texture-hash classifier (standard Remix path).
    if (!uiTexHashes.empty()) {
      const auto& srvs = m_context->m_state.ps.shaderResources.views;
      for (uint32_t i = 0; i < srvs.size(); ++i) {
        const auto* srv = srvs[i].ptr();
        if (srv == nullptr)
          continue;
        const Rc<DxvkImageView> view = srv->GetImageView();
        if (view == nullptr || view->image() == nullptr)
          continue;
        const XXH64_hash_t h = view->image()->getHash();
        if (h != 0 && lookupHash(uiTexHashes, h)) {
          triggerKind = Trigger::Texture;
          triggerHash = h;
          triggerSlot = i;
          break;
        }
      }
    }

    // Shader-hash classifiers (our extension, for games whose HUD has no
    // hashable textures — TF2 VGUI is the motivating case).
    if (triggerKind == Trigger::None
        && (!uiVsHashes.empty() || !uiPsHashes.empty())) {
      XXH64_hash_t vsHash = 0, psHash = 0;
      GetCurrentVsPsHashes(vsHash, psHash);
      if (vsHash != 0 && lookupHash(uiVsHashes, vsHash)) {
        triggerKind = Trigger::VertexShader;
        triggerHash = vsHash;
      } else if (psHash != 0 && lookupHash(uiPsHashes, psHash)) {
        triggerKind = Trigger::PixelShader;
        triggerHash = psHash;
      }
    }

    if (triggerKind == Trigger::None)
      return;

    m_earlyInjectFiredThisFrame = true;

    const char* kindStr = "?";
    switch (triggerKind) {
      case Trigger::Texture:      kindStr = "tex"; break;
      case Trigger::VertexShader: kindStr = "vs";  break;
      case Trigger::PixelShader:  kindStr = "ps";  break;
      default: break;
    }

    const uint32_t drawsAtInject = m_drawCallID;
    const uint32_t rawAtInject   = m_rawDrawCount;
    const std::string triggerVs  = m_currentVsHashCache.empty()
        ? std::string("?")
        : m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
    const Rc<DxvkImage> bb = m_cachedBackbuffer;

    // First fire of the process → loud. Thereafter throttle to once per
    // 256 fires so steady-state gameplay logs a heartbeat without flooding.
    {
      static bool sLoggedFirst = false;
      static uint64_t sFireCount = 0;
      if (!sLoggedFirst) {
        Logger::info(str::format(
          "[D3D11Rtx.UITex] FIRST injectRTX scheduled this process:"
          " kind=",     kindStr,
          " hash=0x",   std::hex, triggerHash, std::dec,
          " psSrvSlot=", (triggerSlot == UINT32_MAX ? std::string("-") : std::to_string(triggerSlot)),
          " triggerVs=", triggerVs,
          " drawsSoFar=", drawsAtInject,
          " rawSoFar=",   rawAtInject));
        sLoggedFirst = true;
      } else if ((sFireCount & 0xFF) == 0) {
        Logger::info(str::format(
          "[D3D11Rtx.UITex] injectRTX scheduled:"
          " kind=",     kindStr,
          " hash=0x",   std::hex, triggerHash, std::dec,
          " drawsSoFar=", drawsAtInject));
      }
      ++sFireCount;
    }

    // Capture the raw pointer + hash for the CS-thread log. Don't capture
    // `this` — D3D11Rtx lives on the main thread and the CS lambda should
    // depend only on its own arguments.
    const std::string kindStrCopy(kindStr);
    m_context->EmitCs([triggerHash, drawsAtInject, kindStrCopy](DxvkContext* ctx) {
      RtxContext* rtx = static_cast<RtxContext*>(ctx);
      const uint32_t fid = rtx->getDevice()->getCurrentFrameId();
      const bool camValid = rtx->getSceneManager().getCamera().isValid(fid);
      Logger::info(str::format(
        "[D3D11Rtx.UITex] CS injectRTX: frameId=", fid,
        " kind=",          kindStrCopy,
        " hash=0x",        std::hex, triggerHash, std::dec,
        " drawsAtInject=", drawsAtInject,
        " camValid=",      camValid ? 1 : 0));
    });
  }


  // NV-DXVK: On HUD-class filter rejections, dump the bound PS SRV image
  // hashes so the user can copy them into rtx.uiTextures. Gated by a
  // per-(VS,PS,hashSet) one-shot set so repeated identical draws across
  // many frames log exactly once; each newly-seen HUD VS/PS combination
  // produces one log line with up to 8 PS texture hashes.
  void D3D11Rtx::LogPsHashesForHudFilter(const char* site) {
    const auto& psSrvs = m_context->m_state.ps.shaderResources.views;

    // Walk every PS SRV slot. Record:
    //   hashes[]/slots[] — slots whose view has a nonzero image hash
    //                      (these are the real candidates for rtx.uiTextures)
    //   zeroHashSlots[]  — slots bound to an image whose getHash()==0
    //                      (dynamic/RT/staging — the HUD may actually use
    //                      these in TF2; we still want to see they exist so
    //                      the absence of "real" hashes isn't mysterious)
    //   bufSlots[]       — slots bound to a buffer view (not an image at all)
    //   totalBound       — count of non-null SRV entries in this draw
    // We log even when there are no hashable images — that's the whole
    // point of this diagnostic: if a HUD draw has zero real texture hashes,
    // the user needs to see that too (tells them rtx.uiTextures can't
    // classify this draw, and they should look at shader-hash gating or
    // the ImGui "UI Texture" picker's captured-frame flow instead).
    std::array<XXH64_hash_t, 8> hashes{};
    std::array<uint32_t, 8>     slots{};
    uint32_t nHashes = 0;

    std::array<uint32_t, 8>     zeroHashSlots{};
    uint32_t nZeroHash = 0;

    std::array<uint32_t, 8>     bufSlots{};
    uint32_t nBufSlots = 0;

    uint32_t totalBound = 0;

    for (uint32_t i = 0; i < psSrvs.size(); ++i) {
      const auto* srv = psSrvs[i].ptr();
      if (srv == nullptr)
        continue;
      ++totalBound;

      const Rc<DxvkImageView> view = srv->GetImageView();
      if (view == nullptr || view->image() == nullptr) {
        // Buffer SRV (typed/structured buffer, or null image view) — not
        // usable as a uiTextures entry but still worth knowing the slot.
        if (nBufSlots < bufSlots.size())
          bufSlots[nBufSlots++] = i;
        continue;
      }
      const XXH64_hash_t h = view->image()->getHash();
      if (h == 0) {
        if (nZeroHash < zeroHashSlots.size())
          zeroHashSlots[nZeroHash++] = i;
        continue;
      }

      if (nHashes < hashes.size()) {
        hashes[nHashes] = h;
        slots [nHashes] = i;
        ++nHashes;
      }
    }

    // One-shot de-dupe key. Include the hashable hashes AND the count of
    // non-hashable bindings so two draw types that happen to have the same
    // real hashes but different "everything else" still log separately.
    std::string psName = "null";
    const auto* psShader = m_context->m_state.ps.shader.ptr();
    if (psShader != nullptr && psShader->GetCommonShader() != nullptr) {
      const auto& sh = psShader->GetCommonShader()->GetShader();
      if (sh != nullptr)
        psName = sh->getShaderKey().toString().substr(0, 19);
    }
    const std::string vsName = m_currentVsHashCache.empty()
        ? std::string("?")
        : m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));

    XXH64_hash_t keyXor = 0;
    for (uint32_t i = 0; i < nHashes; ++i)
      keyXor ^= hashes[i];
    const std::string key = vsName + "|" + psName
                          + "|" + std::to_string(keyXor)
                          + "|" + std::to_string(totalBound)
                          + "|" + std::to_string(nZeroHash)
                          + "|" + std::to_string(nBufSlots);

    static std::unordered_set<std::string> sSeen;
    if (!sSeen.insert(key).second)
      return;

    // Build human-readable descriptions of the three buckets.
    auto fmtHashes = [&]() {
      std::string out;
      for (uint32_t i = 0; i < nHashes; ++i) {
        if (i > 0) out += ",";
        char buf[24];
        std::snprintf(buf, sizeof(buf), "0x%016llx",
                      static_cast<unsigned long long>(hashes[i]));
        out += buf;
        out += "(s";
        out += std::to_string(slots[i]);
        out += ")";
      }
      return out;
    };
    auto fmtSlotList = [](const std::array<uint32_t, 8>& arr, uint32_t n) {
      std::string out;
      for (uint32_t i = 0; i < n; ++i) {
        if (i > 0) out += ",";
        out += "s";
        out += std::to_string(arr[i]);
      }
      return out;
    };

    // Also surface the 64-bit VS/PS hash prefixes so the user has a
    // copy-pasteable value for rtx.uiVertexShaderHashes /
    // rtx.uiPixelShaderHashes even when `hashes=[]` (TF2 HUD case: every
    // PS SRV is dynamic or a buffer view, so rtx.uiTextures is unusable).
    XXH64_hash_t vsPrefix = 0, psPrefix = 0;
    GetCurrentVsPsHashes(vsPrefix, psPrefix);
    char vsHex[24], psHex[24];
    std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                  static_cast<unsigned long long>(vsPrefix));
    std::snprintf(psHex, sizeof(psHex), "0x%016llx",
                  static_cast<unsigned long long>(psPrefix));

    Logger::info(str::format(
      "[D3D11Rtx.UITex] HUD-filter draw (", site, ")"
      " vs=",            vsName,    " vsHash=",  vsHex,
      " ps=",            psName,    " psHash=",  psHex,
      " boundPsSRVs=",   totalBound,
      " hashes=[",       fmtHashes(),                     "]"
      " zeroHash=[",     fmtSlotList(zeroHashSlots, nZeroHash),  "]"
      " bufSRVs=[",      fmtSlotList(bufSlots,      nBufSlots),  "]"
      " → paste vsHash into rtx.uiVertexShaderHashes OR psHash into rtx.uiPixelShaderHashes (hashes[] into rtx.uiTextures if non-empty)"));
  }

  // NV-DXVK [VanishDiag-Stack]: capture call-stack ONCE per F9 cycle for
  // each target VS hash. Called from OnDraw/OnDrawIndexed at function
  // entry — the stack at this point includes the TF2 caller frames above
  // DXVK's D3D11 layer.
  static inline void captureVanishStackIfTarget(uint64_t vsHash) {
    for (int i = 0; i < 3; ++i) {
      if (vsHash != k_VanishStackTargets[i]) continue;
      // Only capture if slot is empty (first hit since last F9 reset).
      if (g_vanishStack[i].frameCount != 0) return;
      void* frames[16];
      const USHORT n = RtlCaptureStackBackTrace(0, 16, frames, nullptr);
      g_vanishStack[i].vsHash = vsHash;
      g_vanishStack[i].frameCount = n;
      for (USHORT k = 0; k < n && k < 16; ++k) {
        g_vanishStack[i].frames[k] = reinterpret_cast<uint64_t>(frames[k]);
      }
      ++g_vanishStackTotalHits;
      return;
    }
  }

  bool D3D11Rtx::OnDraw(UINT vertexCount, UINT startVertex) {
    ++m_rawDrawCount;
    { if (auto* vsP = m_context->m_state.vs.shader.ptr()) { if (auto* csP = vsP->GetCommonShader()) { const auto& shP = csP->GetShader(); if (shP != nullptr) { const uint64_t vsR = static_cast<uint64_t>(shP->getHash()); ++m_rawVsHistogram[vsR]; captureVanishStackIfTarget(vsR); } } } }
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    m_lastDrawIsHudClass   = false;
    SubmitDraw(false, vertexCount, startVertex, 0);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawIndexed(UINT indexCount, UINT startIndex, INT baseVertex) {
    ++m_rawDrawCount;
    { if (auto* vsP = m_context->m_state.vs.shader.ptr()) { if (auto* csP = vsP->GetCommonShader()) { const auto& shP = csP->GetShader(); if (shP != nullptr) { const uint64_t vsR = static_cast<uint64_t>(shP->getHash()); ++m_rawVsHistogram[vsR]; captureVanishStackIfTarget(vsR); } } } }
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    m_lastDrawIsHudClass   = false;
    SubmitDraw(true, indexCount, startIndex, baseVertex);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    // NV-DXVK [HUD-Option5 v4]: rescue-override for TF2's composite-
    // chain VSes. In gameplay, Remix's classifier was capturing these
    // draws for RT (m_lastDrawCaptured=true), so OnDrawIndexed returned
    // true and the caller skipped the native drawIndexed. Result: TF2's
    // composite never wrote to the backbuffer, and the HUD scratch was
    // never composited back — no HUD on screen. Fix: force these draws
    // to native raster by setting filteredAsUI=true.
    //
    // Hash list is intentionally inline (user instruction: don't move
    // to rtx.conf). Each hash confirmed via CsDrawTrace + shader
    // decompile during the 2026-04-23 session.
    if (m_lastDrawCaptured && !m_lastDrawFilteredAsUI) {
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      const bool isCompositeChain =
        (vsH == 0xca1e169b461e81eeULL) ||   // final scene composite
        (vsH == 0x550a39e2a6910fbfULL) ||   // menu/transition HUD composite
        (vsH == 0x9da6a507fdd3f028ULL) ||   // gameplay HUD writer -> scratch
        (vsH == 0xd69c3951f050e757ULL) ||   // transition HUD writer
        (vsH == 0x3abf38dd1f5bd794ULL) ||   // transition HUD writer
        (vsH == 0x7e61f941fd9d5cacULL);     // transition secondary HUD writer
      if (isCompositeChain) {
        m_lastDrawFilteredAsUI = true;
        m_lastDrawCaptured = false;
      }
    }
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawInstanced(UINT vertexCountPerInstance, UINT instanceCount, UINT startVertex, UINT startInstance) {
    ++m_rawDrawCount;
    { if (auto* vsP = m_context->m_state.vs.shader.ptr()) { if (auto* csP = vsP->GetCommonShader()) { const auto& shP = csP->GetShader(); if (shP != nullptr) { const uint64_t vsR = static_cast<uint64_t>(shP->getHash()); ++m_rawVsHistogram[vsR]; captureVanishStackIfTarget(vsR); } } } }
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    m_lastDrawIsHudClass   = false;
    SubmitInstancedDraw(false, vertexCountPerInstance, startVertex, 0, instanceCount, startInstance);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawIndexedInstanced(UINT indexCountPerInstance, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance) {
    ++m_rawDrawCount;
    { if (auto* vsP = m_context->m_state.vs.shader.ptr()) { if (auto* csP = vsP->GetCommonShader()) { const auto& shP = csP->GetShader(); if (shP != nullptr) { const uint64_t vsR = static_cast<uint64_t>(shP->getHash()); ++m_rawVsHistogram[vsR]; captureVanishStackIfTarget(vsR); } } } }
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    m_lastDrawIsHudClass   = false;
    SubmitInstancedDraw(true, indexCountPerInstance, startIndex, baseVertex, instanceCount, startInstance);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  void D3D11Rtx::SubmitInstancedDraw(bool indexed, UINT count, UINT start, INT base,
                                      UINT instanceCount, UINT startInstance) {
    try {
    if (instanceCount <= 1) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    // Find per-instance float4 rows in the input layout that form a world matrix.
    // Engines encode this as 3 or 4 consecutive float4 elements with per-instance step rate,
    // using semantics like INSTANCETRANSFORM, WORLD, I, INST, or TEXCOORD at high indices.
    auto* layout = m_context->m_state.ia.inputLayout.ptr();
    if (!layout) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    const auto& semantics = layout->GetRtxSemantics();

    struct Float4Row {
      uint32_t inputSlot;
      uint32_t byteOffset;
    };

    std::vector<Float4Row> instRows;
    uint32_t instSlot = UINT32_MAX;

    for (const auto& s : semantics) {
      if (!s.perInstance) continue;
      if (s.format != VK_FORMAT_R32G32B32A32_SFLOAT) continue;

      // Accept any per-instance float4 — most engines use INSTANCETRANSFORM, WORLD, I, INST,
      // or repurpose TEXCOORD with high indices. The key signal is per-instance + float4.
      if (instSlot == UINT32_MAX)
        instSlot = s.inputSlot;

      // Only collect rows from the same input slot.
      if (s.inputSlot != instSlot) continue;
      instRows.push_back({s.inputSlot, s.byteOffset});
    }

    if (instRows.size() < 3) {
      // NV-DXVK: Source Engine 2 bone-index instancing.
      // Per-instance data is R16G16B16A16_UINT containing bone indices,
      // NOT float4 transform rows.  The actual transforms are in VS SRV t30
      // (g_boneMatrix, StructuredBuffer<float3x4>, stride=48).
      // Read the bone index for each instance, fetch the float3x4 from t30,
      // and use it as the instance world transform.
      bool handledAsBoneInstancing = false;
      const D3D11RtxSemantic* boneIdxSem = nullptr;
      ID3D11ShaderResourceView* boneSrv = nullptr;
      {
        // Find per-instance UINT semantic. Accepts both:
        //   R16G16B16A16_UINT — legacy skinned-character per-instance bone idx
        //   R32G32_UINT       — TF2 BSP / batched-prop per-instance modelInst idx
        for (const auto& s : semantics) {
          if (!s.perInstance) continue;
          if (s.format == VK_FORMAT_R16G16B16A16_UINT
              || s.format == VK_FORMAT_R32G32_UINT
              || s.format == VK_FORMAT_R32G32B32A32_UINT) {
            boneIdxSem = &s;
            break;
          }
        }
        // DEBUG: count fanout entries by semantic format
        if (boneIdxSem) {
          static uint32_t sFanoutR16 = 0, sFanoutR32x2 = 0, sFanoutR32x4 = 0;
          uint32_t* counter = (boneIdxSem->format == VK_FORMAT_R16G16B16A16_UINT) ? &sFanoutR16
                            : (boneIdxSem->format == VK_FORMAT_R32G32_UINT) ? &sFanoutR32x2
                            : &sFanoutR32x4;
          if ((*counter) < 5) {
            ++(*counter);
            Logger::info(str::format(
              "[D3D11Rtx] FanoutSem: fmt=", uint32_t(boneIdxSem->format),
              " perInst=", boneIdxSem->perInstance ? 1 : 0,
              " slot=", boneIdxSem->inputSlot,
              " byteOff=", boneIdxSem->byteOffset,
              " counts(R16=", sFanoutR16, " R32x2=", sFanoutR32x2,
              " R32x4=", sFanoutR32x4, ")"));
          }
        }
        // NV-DXVK: deterministic slot selection via RDEF — ask the VS itself
        // whether it reads g_modelInst (BSP) or g_boneMatrix (skinned). Falls
        // back to blind t31/t30 probing for shaders without RDEF.
        uint32_t usedSlot = 0;
        bool isModelInstFanout = false;
        {
          uint32_t modelInstSlot = UINT32_MAX, boneMatrixSlot = UINT32_MAX;
          auto vsPtr = m_context->m_state.vs.shader;
          if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
            const D3D11CommonShader* common = vsPtr->GetCommonShader();
            modelInstSlot  = common->FindResourceSlot("g_modelInst");
            boneMatrixSlot = common->FindResourceSlot("g_boneMatrix");
          }
          if (modelInstSlot != UINT32_MAX
              && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            boneSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
            if (boneSrv) { usedSlot = modelInstSlot; isModelInstFanout = true; }
          }
          if (!boneSrv && boneMatrixSlot != UINT32_MAX
              && boneMatrixSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            boneSrv = m_context->m_state.vs.shaderResources.views[boneMatrixSlot].ptr();
            if (boneSrv) usedSlot = boneMatrixSlot;
          }
          if (!boneSrv) {
            // Last-resort blind probe — RDEF didn't tell us which slot the VS
            // reads. Log loudly: every blind hit is a shader we can't classify
            // deterministically and may mis-route (BSP transforms vs bone
            // matrices). Either RDEF was stripped or the resource has a name
            // we don't recognize.
            const uint32_t kInstTransformSlot = 31, kBoneSrvSlot = 30;
            if (kInstTransformSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
              boneSrv = m_context->m_state.vs.shaderResources.views[kInstTransformSlot].ptr();
              if (boneSrv) { usedSlot = kInstTransformSlot; isModelInstFanout = true; }
            }
            if (!boneSrv && kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
              boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();
              if (boneSrv) usedSlot = kBoneSrvSlot;
            }
            if (boneSrv) {
              static std::unordered_set<uintptr_t> sBlindLogged;
              auto vsPtr = m_context->m_state.vs.shader;
              uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
              if (key && sBlindLogged.insert(key).second) {
                std::string vsHash = "?";
                if (vsPtr->GetCommonShader() != nullptr) {
                  auto& s = vsPtr->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash = s->getShaderKey().toString();
                }
                Logger::err(str::format(
                  "[D3D11Rtx] BLIND-PROBE fanout for VS=", vsHash,
                  " (RDEF lookup found neither g_modelInst nor g_boneMatrix)",
                  " — guessing slot=", usedSlot,
                  " isModelInst=", isModelInstFanout ? 1 : 0,
                  ". This shader will use heuristic routing and may be wrong."));
              }
            }
          }
        }
        static bool sLoggedSlot = false;
        if (!sLoggedSlot && boneSrv) {
          sLoggedSlot = true;
          Com<ID3D11Resource> r; boneSrv->GetResource(&r);
          auto* b = static_cast<D3D11Buffer*>(r.ptr());
          Logger::info(str::format("[D3D11Rtx] Using SRV slot ", usedSlot,
            " bufSize=", (b ? b->Desc()->ByteWidth : 0),
            " usage=", (b ? b->Desc()->Usage : 0),
            " hasImmData=", (b ? b->GetImmutableData().size() : 0)));
        }

        if (boneIdxSem && boneSrv) {
          // Get the bone matrix buffer
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          DxvkBufferSlice boneBufSlice = boneBuf ? boneBuf->GetBufferSlice() : DxvkBufferSlice();
          const uint8_t* bonePtr = boneBufSlice.defined() ?
            reinterpret_cast<const uint8_t*>(boneBufSlice.mapPtr(0)) : nullptr;
          const size_t boneBufLen = boneBufSlice.defined() ? boneBufSlice.length() : 0;

          // Get the per-instance index buffer
          const auto& instVb = m_context->m_state.ia.vertexBuffers[boneIdxSem->inputSlot];
          const uint8_t* boneReadPtr = bonePtr;
          size_t boneReadLen = boneBufLen;

          // Try multiple paths for bone buffer if direct mapPtr failed
          if (!boneReadPtr && boneBuf) {
            const auto mapped = boneBuf->GetMappedSlice();
            if (mapped.mapPtr && mapped.length >= 48) {
              boneReadPtr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              boneReadLen = mapped.length;
            }
          }
          if (!boneReadPtr && boneBuf) {
            void* p = boneBuf->GetBuffer()->mapPtr(0);
            if (p) {
              boneReadPtr = reinterpret_cast<const uint8_t*>(p);
              boneReadLen = boneBuf->GetBuffer()->info().size;
            }
          }

          // Read IMMUTABLE instance buffer data from CPU cache (set at CreateBuffer time).
          if (instVb.buffer != nullptr && m_cachedInstBufPtr != instVb.buffer.ptr()) {
            const auto& immData = instVb.buffer->GetImmutableData();
            if (!immData.empty()) {
              m_instBufCache = immData;
              m_cachedInstBufPtr = instVb.buffer.ptr();
            }
          }

          // Log VS hash on first bone-instanced draw (to find the shader)
          static bool sLoggedVsHash = false;
          if (!sLoggedVsHash) {
            sLoggedVsHash = true;
            auto vsShader = m_context->m_state.vs.shader;
            if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
              auto& shader = vsShader->GetCommonShader()->GetShader();
              if (shader != nullptr) {
                Logger::info(str::format(
                  "[D3D11Rtx] Bone-instanced VS hash: ",
                  shader->getShaderKey().toString()));
              }
            }
          }

          // 1 BLAS + N TLAS instances via instancesToObject.
          // Per-instance transforms are in the DYNAMIC t31 buffer (208 bytes/instance).
          m_boneInstBatches++;

          // Read the t31 buffer (per-instance world transforms) directly.
          // It's DYNAMIC (MAP_WRITE_DISCARD) so we use GetMappedSlice().
          const uint8_t* t31Data = nullptr;
          size_t t31Len = 0;
          {
            Com<ID3D11Resource> res;
            boneSrv->GetResource(&res);
            auto* t31Buf = static_cast<D3D11Buffer*>(res.ptr());
            if (t31Buf) {
              auto mapped = t31Buf->GetMappedSlice();
              if (mapped.mapPtr && mapped.length > 0) {
                t31Data = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                t31Len = mapped.length;
              } else {
                // Fallback: try the underlying DxvkBuffer's slice
                void* p = t31Buf->GetBuffer()->mapPtr(0);
                if (p) {
                  t31Data = reinterpret_cast<const uint8_t*>(p);
                  t31Len = t31Buf->GetBuffer()->info().size;
                }
              }
            }
          }

          // Debug logging — fires after frame 50 (when user has loaded into scene)
          static uint32_t sFrameCount = 0;
          static uint32_t sDumpedAll = 0;
          sFrameCount++;

          // Track a specific batch's t31 translation across frames to see
          // if it's view-dependent (changes when camera moves) or stable.
          static uint64_t sTrackedKey = 0;
          static uint32_t sTrackedCount = 0;
          if (t31Data && t31Len >= 48 && sTrackedCount < 30) {
            // Pick the first batch we see and track its t31[0] every frame
            uint64_t myKey = reinterpret_cast<uintptr_t>(boneSrv);
            if (sTrackedKey == 0) sTrackedKey = myKey;
            if (myKey == sTrackedKey) {
              const float* m = reinterpret_cast<const float*>(t31Data);
              ++sTrackedCount;
              Logger::info(str::format(
                "[D3D11Rtx] Track srv=", myKey, " frame=", sFrameCount,
                " t31[0].T=(", m[3], ",", m[7], ",", m[11], ")"));
            }
          }

          // #3: Dump FULL t31 matrices ONCE after 50 frames (rotation + translation)
          if (sFrameCount > 50 && sDumpedAll < 1 && t31Data && t31Len >= 48) {
            ++sDumpedAll;
            uint32_t numInst = static_cast<uint32_t>(t31Len / 208);
            // Full matrix dump for first 2 instances — check scale/rotation/translation
            for (uint32_t k = 0; k < std::min(numInst, 2u); ++k) {
              const float* m = reinterpret_cast<const float*>(t31Data + k * 208);
              // compute magnitude of each row (scale per axis)
              float mag0 = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
              float mag1 = std::sqrt(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);
              float mag2 = std::sqrt(m[8]*m[8] + m[9]*m[9] + m[10]*m[10]);
              Logger::info(str::format(
                "[D3D11Rtx] T31 mat[", k, "]:"
                " r0=(", m[0], ",", m[1], ",", m[2], ") T0=", m[3], " mag=", mag0,
                " r1=(", m[4], ",", m[5], ",", m[6], ") T1=", m[7], " mag=", mag1,
                " r2=(", m[8], ",", m[9], ",", m[10], ") T2=", m[11], " mag=", mag2));
            }
            std::string dump = str::format("inst=", numInst);
            for (uint32_t k = 0; k < numInst && k * 208 + 48 <= t31Len; ++k) {
              const float* m = reinterpret_cast<const float*>(t31Data + k * 208);
              dump += str::format(" [", k, "]=(", m[3], ",", m[7], ",", m[11], ")");
            }
            Logger::info(str::format("[D3D11Rtx] DumpAllT31: ", dump));

            // (vertex decode test removed — Z offset bug already fixed from shader decomp)

            // #5: log cb0, cb2, cb3 sizes
            const auto& vsCbs = m_context->m_state.vs.constantBuffers;
            for (uint32_t sl = 0; sl < 4; ++sl) {
              if (vsCbs[sl].buffer != nullptr) {
                Logger::info(str::format("[D3D11Rtx] VS cb[", sl, "] size=",
                  vsCbs[sl].buffer->Desc()->ByteWidth,
                  " off=", vsCbs[sl].constantOffset));
              }
            }
          }

          if (!m_instBufCache.empty() && t31Data) {
            const UINT maxInstances = instanceCount;
            constexpr uint32_t BYTES_PER_INSTANCE = 208;

            // NV-DXVK (TF2 BSP): t31 stores objectToCameraRelative transforms
            // (vertex buffers hold world - cameraOrigin). Remix's camera, NRC,
            // denoisers, and motion-vector systems live in absolute world, so
            // we ADD c_cameraOrigin to each per-instance translation at push
            // time to shift BSP from camera-relative into absolute world.
            // camOrigin is read from CBufCommonPerCamera offset 4 below and
            // applied in the tforms loop.
            float camOrigin[3] = { 0.0f, 0.0f, 0.0f };
            bool haveCamOrigin = false;
            // DEBUG: log failure reason once per unique VS
            const char* failReason = nullptr;
            {
              auto vsPtr4 = m_context->m_state.vs.shader;
              if (vsPtr4 == nullptr || vsPtr4->GetCommonShader() == nullptr) {
                failReason = "no_common_shader";
              } else {
                const D3D11CommonShader* common = vsPtr4->GetCommonShader();
                auto camLoc = common->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
                if (!camLoc) {
                  failReason = "FindCBField_returned_null";
                } else if (camLoc->size < 12) {
                  failReason = "size<12";
                } else if (camLoc->slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                  failReason = "slot_oob";
                } else {
                  const auto& vsCbs2 = m_context->m_state.vs.constantBuffers;
                  const auto& cb = vsCbs2[camLoc->slot];
                  if (cb.buffer == nullptr) {
                    failReason = "cb_buffer_null";
                  } else {
                    const auto mapped = cb.buffer->GetMappedSlice();
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + camLoc->offset;
                    if (!p) {
                      failReason = "mapPtr_null";
                    } else if (base + 12 > cb.buffer->Desc()->ByteWidth) {
                      failReason = "base+12_oob";
                    } else {
                      const float* fp = reinterpret_cast<const float*>(p + base);
                      if (!std::isfinite(fp[0]) || !std::isfinite(fp[1]) || !std::isfinite(fp[2])) {
                        failReason = "non_finite";
                      } else {
                        camOrigin[0] = fp[0]; camOrigin[1] = fp[1]; camOrigin[2] = fp[2];
                        haveCamOrigin = true;
                        // NV-DXVK [diag]: log RAW cb read on EVERY value change
                        // (also first read of session). Now includes VS hash so
                        // we can see if multiple VS passes are alternating
                        // different cam origins through the fanout gate (which
                        // would explain stationary-camera Z oscillation).
                        {
                          static uint64_t sFanoutReadN = 0;
                          static float sLastCx = 1e30f, sLastCy = 1e30f, sLastCz = 1e30f;
                          static uintptr_t sLastBuf = 0;
                          static std::string sLastVs;
                          const uintptr_t bufPtr = reinterpret_cast<uintptr_t>(cb.buffer.ptr());
                          // Resolve the bound VS hash for the read.
                          std::string vsKey = "?";
                          {
                            auto vsP = m_context->m_state.vs.shader;
                            if (vsP != nullptr && vsP->GetCommonShader() != nullptr) {
                              auto& sh = vsP->GetCommonShader()->GetShader();
                              if (sh != nullptr) {
                                vsKey = sh->getShaderKey().toString().substr(0, 19);
                              }
                            }
                          }
                          const bool moved =
                               std::abs(fp[0] - sLastCx) > 0.01f
                            || std::abs(fp[1] - sLastCy) > 0.01f
                            || std::abs(fp[2] - sLastCz) > 0.01f
                            || bufPtr != sLastBuf
                            || vsKey != sLastVs;
                          if (moved) {
                            sLastCx = fp[0]; sLastCy = fp[1]; sLastCz = fp[2];
                            sLastBuf = bufPtr;
                            sLastVs = vsKey;
                            const auto& vps = m_context->m_state.rs.viewports;
                            Logger::info(str::format(
                              "[fanoutCBRead] #", sFanoutReadN,
                              " vs=", vsKey,
                              " cam=(", fp[0], ",", fp[1], ",", fp[2], ")",
                              " buf=", bufPtr,
                              " vp=", uint32_t(vps[0].Width), "x", uint32_t(vps[0].Height),
                              " minD=", vps[0].MinDepth,
                              " maxD=", vps[0].MaxDepth,
                              " slot=", camLoc->slot,
                              " cbOff=", cb.constantOffset));
                          }
                          ++sFanoutReadN;
                        }
                        // NV-DXVK: publish to m_lastFanoutCamOrigin ONLY if
                        // this draw is from the MAIN gameplay camera pass, not
                        // a shadow cascade / reflection probe / cubemap etc.
                        // Heuristic: main pass has a non-square, target-sized
                        // viewport (e.g. 2560x1440). Shadow cascades use square
                        // viewports (1024x1024). Reflection probes use tiny
                        // off-screen RTs. Without this filter fanout publishes
                        // ~15+ different origins per frame and path 1/3 end up
                        // using whichever fanned out last → chaos.
                        bool isMainViewport = false;
                        {
                          const auto& vps = m_context->m_state.rs.viewports;
                          const float vw = vps[0].Width;
                          const float vh = vps[0].Height;
                          // NV-DXVK [viewmodel reject — REAPPLIED]:
                          // TF2's first-person viewmodel pass renders at the
                          // backbuffer aspect ratio (passes the aspect filter
                          // below) but with a compressed depth range
                          // (MaxDepth ≤ 0.05–0.08) so the gun never z-clips
                          // through world geometry. Its c_cameraOrigin is
                          // ~18-unit offset from main, and if it leaks into
                          // m_lastFanoutCamOrigin the cachedSave player-cam
                          // filter alternately accepts main and viewmodel
                          // saves into m_lastGoodTransforms — the ray
                          // tracer's main camera then flickers between the
                          // two, and the visible-frame camera position
                          // depends on whoever wrote last.
                          // The earlier-this-session new log confirms
                          // VS_ef94e6c7fcc3c144 (TF2's viewmodel VS, per
                          // user memory note) renders at maxD=0.05 with
                          // origin=(14000,-10800,800) — exactly the case
                          // this gate is for.
                          const float vpMaxDepth = vps[0].MaxDepth;
                          const bool isViewModelPass = (vpMaxDepth <= 0.08f);
                          if (isViewModelPass) {
                            // Don't publish viewmodel cb2 origin to fanout
                            // (m_lastFanoutCamOrigin must stay BSP-pass).
                            // BUT do publish to m_lastViewmodelCamOrigin —
                            // the viewmodel pass's c_cameraOrigin is Source's
                            // authoritative eye, which on rodeo (pilot on
                            // top of Titan) is the only field that carries
                            // the actual pilot eye position (BSP-pass cb2
                            // has Titan cockpit; lp+0x3D6C is static on this
                            // build). CameraManager snaps Main's worldToView
                            // translation to this. Captured here as the raw
                            // 3-float c_cameraOrigin field — no decomposition,
                            // no per-VS float noise.
                            const bool vmChanged =
                              !m_hasViewmodelCamOrigin
                              || std::abs(m_lastViewmodelCamOrigin.x - fp[0]) > 0.01f
                              || std::abs(m_lastViewmodelCamOrigin.y - fp[1]) > 0.01f
                              || std::abs(m_lastViewmodelCamOrigin.z - fp[2]) > 0.01f;
                            m_lastViewmodelCamOrigin = Vector3(fp[0], fp[1], fp[2]);
                            m_hasViewmodelCamOrigin = true;
                            // Mirror to file-scope atomics so CameraManager
                            // can read across-translation-unit. memory_order
                            // relaxed is fine — single producer, single
                            // consumer, no ordering dependency.
                            dxvk::tf2::g_pilotEyeX.store(fp[0], std::memory_order_relaxed);
                            dxvk::tf2::g_pilotEyeY.store(fp[1], std::memory_order_relaxed);
                            dxvk::tf2::g_pilotEyeZ.store(fp[2], std::memory_order_relaxed);
                            dxvk::tf2::g_pilotEyeValid.store(true, std::memory_order_relaxed);
                            if (vmChanged) {
                              static uint32_t sVmEyeWriteN = 0;
                              if (sVmEyeWriteN < 60) {
                                ++sVmEyeWriteN;
                                Logger::info(str::format(
                                  "[viewmodelCamWrite] #", sVmEyeWriteN,
                                  " cam=(", fp[0], ",", fp[1], ",", fp[2], ")",
                                  " maxD=", vpMaxDepth));
                              }
                            }
                          } else if (vw > 0.0f && vh > 0.0f) {
                            // NV-DXVK: self-calibrated main-viewport check.
                            // The "main view" is whatever viewport matches the
                            // captured composite RT extent (CompositeOut v4
                            // detects this from VS hash + format + extent ==
                            // viewport at composite-draw time, then publishes
                            // the dims here via m_compositeOutputW/H). This
                            // auto-tracks resolution changes / DLSS / render-
                            // scale toggles with no thresholds.
                            //
                            // Fallback for the very first frames before
                            // composite has been seen: use the old non-square
                            // + reasonable-size heuristic.
                            if (m_compositeOutputW != 0 && m_compositeOutputH != 0) {
                              // Match by ASPECT RATIO, not exact pixels. With
                              // DLSS / dynamic render scale, gameplay can be
                              // rendered at e.g. 960x540 while the composite
                              // is 1920x1080 — both 16:9, both player view.
                              // Probes / shadow cascades are square (asp=1),
                              // so the aspect comparison still rejects them.
                              const float vAsp = vw / vh;
                              const float cAsp = float(m_compositeOutputW)
                                               / float(m_compositeOutputH);
                              isMainViewport = std::abs(vAsp - cAsp) < 0.01f
                                            && vw >= 240.0f && vh >= 135.0f;
                            } else {
                              const float asp = vw / vh;
                              const bool nonSquare = std::abs(asp - 1.0f) > 0.02f;
                              const bool bigEnough = vw >= 480.0f && vh >= 270.0f;
                              isMainViewport = nonSquare && bigEnough;
                            }
                          }
                        }
                        if (isMainViewport) {
                          // NV-DXVK [diag] one-shot address logger.
                          // fp points into the engine-mapped CPU memory of cb2
                          // (mapPtr + cbOffset + camOriginOffset). Whatever the
                          // engine writes to fp[2] (the bobbed c_cameraOrigin.z)
                          // can be caught by setting a HW write BP on fp+8 in
                          // x64dbg. Logs the address whenever it changes
                          // (dxvk discard-map can rotate the underlying pool).
                          {
                            static uintptr_t sLastFpAddr = 0;
                            const uintptr_t fpAddr =
                              reinterpret_cast<uintptr_t>(fp);
                            if (fpAddr != sLastFpAddr) {
                              sLastFpAddr = fpAddr;
                              Logger::info(str::format(
                                "[fanoutFpAddr] fp=",
                                reinterpret_cast<void*>(fpAddr),
                                " fpZ=",
                                reinterpret_cast<void*>(fpAddr + 8),
                                " buf=", (void*)cb.buffer.ptr(),
                                " base=", base,
                                " curCam=(", fp[0], ",", fp[1], ",", fp[2], ")"));
                            }
                          }
                          const bool changed =
                            !m_hasFanoutCamOrigin
                            || std::abs(m_lastFanoutCamOrigin.x - fp[0]) > 0.01f
                            || std::abs(m_lastFanoutCamOrigin.y - fp[1]) > 0.01f
                            || std::abs(m_lastFanoutCamOrigin.z - fp[2]) > 0.01f;
                          // NV-DXVK [diag] every-write trace. Logs the Z value
                          // each time we overwrite m_lastFanoutCamOrigin so we
                          // can see how rapidly it's being clobbered between
                          // cls12Recon reads. If multiple consecutive writes
                          // within the same frame produce different Z values,
                          // the engine is updating its cbuffer mid-frame across
                          // sub-passes, and we're picking up the moving target.
                          {
                            static uint64_t sFanoutWrite = 0;
                            ++sFanoutWrite;
                            if ((sFanoutWrite % 1) == 0
                                && std::abs(m_lastFanoutCamOrigin.z - fp[2]) > 0.001f) {
                              Logger::info(str::format(
                                "[fanoutCamWrite] #", sFanoutWrite,
                                " z: ", m_lastFanoutCamOrigin.z, " -> ", fp[2],
                                " (delta=", (fp[2] - m_lastFanoutCamOrigin.z), ")"));
                            }
                          }
                          m_lastFanoutCamOrigin = Vector3(fp[0], fp[1], fp[2]);
                          m_hasFanoutCamOrigin = true;
                          // NV-DXVK: capture the VP rows at the SAME cb/offset
                          // we just pulled camOrigin from. CBufCommonPerCamera
                          // lives at camLoc->slot; c_cameraRelativeToClip is at
                          // +16 (current frame VP) and ...PrevFrame at +96. We
                          // prefer +16 but fall back to +96 when +16 is still
                          // identity (very early frames). The same cb is
                          // authoritative for "the gameplay pose" when read
                          // from the fanout VS — path 3 should reuse this
                          // rather than reading cb2@96 of whichever VS it
                          // happens to be running under.
                          {
                            const size_t vpBaseCurr =
                              static_cast<size_t>(cb.constantOffset) * 16 + 16;
                            const size_t vpBasePrev =
                              static_cast<size_t>(cb.constantOffset) * 16 + 96;
                            const size_t bsz = cb.buffer->Desc()->ByteWidth;
                            auto tryReadVP = [&](size_t b) -> bool {
                              if (b + 64 > bsz) return false;
                              const float* vp = reinterpret_cast<const float*>(p + b);
                              for (int k = 0; k < 12; ++k)
                                if (!std::isfinite(vp[k])) return false;
                              Vector3 r0(vp[0], vp[1], vp[2]);
                              Vector3 r1(vp[4], vp[5], vp[6]);
                              Vector3 r2(vp[8], vp[9], vp[10]);
                              // Reject identity — means VP not yet populated.
                              if (std::abs(r0.x - 1.0f) < 1e-4f
                                  && std::abs(r1.y - 1.0f) < 1e-4f
                                  && std::abs(r2.z - 1.0f) < 1e-4f
                                  && std::abs(r0.y) < 1e-4f && std::abs(r0.z) < 1e-4f)
                                return false;
                              // Reject zero / degenerate rows.
                              const float l0 = length(r0), l1 = length(r1), l2 = length(r2);
                              if (l0 < 0.1f || l1 < 0.1f || l2 < 0.001f) return false;
                              // VERIFIED handedness check: VP rows for a
                              // proper player view produce det = -1 when fed
                              // through cls 3/4's reconstruction (matches the
                              // saveMatrix dumps from path1). Mirror passes
                              // (water reflection etc.) produce det = +1 and
                              // would, if cached as fanout VP rows, cause
                              // subsequent cls 3/4 player draws to render
                              // mirrored. Reject by sign of det. Sign of det
                              // doesn't depend on row magnitudes, so compute
                              // on raw rows (no normalization needed).
                              // det = r0 · (r1 × r2)
                              const Vector3 cross12(
                                r1.y * r2.z - r1.z * r2.y,
                                r1.z * r2.x - r1.x * r2.z,
                                r1.x * r2.y - r1.y * r2.x);
                              const float vpDet = r0.x * cross12.x
                                                + r0.y * cross12.y
                                                + r0.z * cross12.z;
                              if (vpDet >= 0.0f) {
                                ++m_geomDiagFanoutMirrorRej;
                                static uint32_t sFanoutMirrorLog = 0;
                                if (sFanoutMirrorLog < 10) {
                                  ++sFanoutMirrorLog;
                                  Logger::info(str::format(
                                    "[fanoutVPRejectMirror] det=", vpDet,
                                    " r0=(", r0.x, ",", r0.y, ",", r0.z, ")",
                                    " r2=(", r2.x, ",", r2.y, ",", r2.z, ")",
                                    " — mirror basis, not publishing"));
                                }
                                return false;
                              }
                              m_lastFanoutVpRow0 = r0;
                              m_lastFanoutVpRow1 = r1;
                              m_lastFanoutVpRow2 = r2;
                              m_hasFanoutVpRows = true;
                              return true;
                            };
                            if (!tryReadVP(vpBaseCurr)) tryReadVP(vpBasePrev);
                          }
                          ++m_geomDiagFanoutPublishes;
                          m_geomDiagLastCamAbs[0] = fp[0];
                          m_geomDiagLastCamAbs[1] = fp[1];
                          m_geomDiagLastCamAbs[2] = fp[2];
                          m_geomDiagHaveCamAbs    = true;
                          if (changed) {
                            static uint32_t sPublishLog = 0;
                            if (sPublishLog < 30) {
                              ++sPublishLog;
                              Logger::info(str::format(
                                "[D3D11Rtx.fanoutOri] publish #", sPublishLog,
                                " draw=", m_drawCallID,
                                " cam=(", fp[0], ",", fp[1], ",", fp[2], ")",
                                " vpRows=", m_hasFanoutVpRows ? 1 : 0));
                            }
                          }
                        } else {
                          ++m_geomDiagFanoutRejects;
                          // Log rejection once per unique non-main viewport so
                          // we can see what's being correctly filtered out.
                          static uint32_t sRejectLog = 0;
                          if (sRejectLog < 10) {
                            ++sRejectLog;
                            const auto& vps = m_context->m_state.rs.viewports;
                            Logger::info(str::format(
                              "[D3D11Rtx.fanoutOri] reject #", sRejectLog,
                              " vp=", int(vps[0].Width), "x", int(vps[0].Height),
                              " cam=(", fp[0], ",", fp[1], ",", fp[2], ")"));
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            if (failReason) {
              ++m_geomDiagBspCamFail;
              static std::unordered_set<uintptr_t> sFailLogged;
              auto vsPtr6 = m_context->m_state.vs.shader;
              uintptr_t key6 = (vsPtr6 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr6.ptr()) : 0;
              if (key6 && sFailLogged.insert(key6).second && sFailLogged.size() < 12) {
                std::string vsHash6 = "?";
                if (vsPtr6->GetCommonShader() != nullptr) {
                  auto& s = vsPtr6->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash6 = s->getShaderKey().toString();
                }
                Logger::warn(str::format(
                  "[D3D11Rtx] BSP camOrigin lookup FAILED VS=", vsHash6,
                  " reason=", failReason));
              }
            }
            // DEBUG: log camOrigin once per unique VS that uses BSP fanout.
            if (haveCamOrigin) {
              static std::unordered_set<uintptr_t> sCamOriginLogged;
              auto vsPtr5 = m_context->m_state.vs.shader;
              uintptr_t key5 = (vsPtr5 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr5.ptr()) : 0;
              if (key5 && sCamOriginLogged.insert(key5).second && sCamOriginLogged.size() < 12) {
                Logger::info(str::format(
                  "[D3D11Rtx] BSP camOrigin=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")"));
              }
            }

            // Track unique position vertex buffers this frame
            {
              uint32_t posSlot = UINT32_MAX;
              for (const auto& s : semantics) {
                if (!s.perInstance && s.format == VK_FORMAT_R32G32_UINT) { posSlot = s.inputSlot; break; }
              }
              if (posSlot != UINT32_MAX) {
                const auto& pvb = m_context->m_state.ia.vertexBuffers[posSlot];
                if (pvb.buffer != nullptr)
                  m_boneInstVbPtrs.insert(reinterpret_cast<uintptr_t>(pvb.buffer.ptr()));
              }
            }

            // ONE SubmitDraw per batch = matches original game's draw count.
            // Scene manager expands to N TLAS instances via instancesToObject.
            auto tforms = std::make_shared<std::vector<Matrix4>>();
            tforms->reserve(maxInstances);

            const uint8_t* instData = m_instBufCache.data();
            const uint32_t stride = instVb.stride;
            const uint32_t boneOff = boneIdxSem->byteOffset;
            // DEBUG: per-VS, dump the first few charIdx values + raw t31 matrix
            // so we can verify the per-instance VB actually contains valid
            // indices and the t31 lookups produce sensible matrices.
            std::string idxDumpLine;
            std::string t31DumpLine;
            bool dumpThisDraw = false;
            {
              static std::unordered_set<uintptr_t> sIdxDumpVsLogged;
              auto vsPtr2 = m_context->m_state.vs.shader;
              uintptr_t key2 = (vsPtr2 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr2.ptr()) : 0;
              if (key2 && sIdxDumpVsLogged.size() < 12 && sIdxDumpVsLogged.insert(key2).second)
                dumpThisDraw = true;
            }
            for (uint32_t i = 0; i < maxInstances; ++i) {
              ++m_geomDiagFanoutInstSeen;
              size_t instOff = static_cast<size_t>(startInstance + i) * stride + boneOff;
              // Index width depends on the semantic format.
              // R16G16B16A16_UINT -> first uint16 (legacy bones)
              // R32G32_UINT / R32G32B32A32_UINT -> first uint32 (BSP)
              uint32_t charIdx = 0;
              if (boneIdxSem->format == VK_FORMAT_R16G16B16A16_UINT) {
                if (instOff + 2 <= m_instBufCache.size())
                  charIdx = *reinterpret_cast<const uint16_t*>(instData + instOff);
              } else {
                if (instOff + 4 <= m_instBufCache.size())
                  charIdx = *reinterpret_cast<const uint32_t*>(instData + instOff);
              }
              size_t t31Off = static_cast<size_t>(charIdx) * BYTES_PER_INSTANCE;
              if (dumpThisDraw && i < 6) {
                idxDumpLine += str::format(" [", i, "]=", charIdx);
                if (t31Off + 48 <= t31Len) {
                  const float* mm = reinterpret_cast<const float*>(t31Data + t31Off);
                  t31DumpLine += str::format(" T", i, "=(", mm[3], ",", mm[7], ",", mm[11], ")");
                } else {
                  t31DumpLine += str::format(" T", i, "=OOB");
                }
              }
              if (t31Off + 48 > t31Len) {
                ++m_geomDiagFanoutInstOob;
                continue;
              }

              const float* m = reinterpret_cast<const float*>(t31Data + t31Off);
              bool allFinite = true;
              for (int f = 0; f < 12; ++f) if (!std::isfinite(m[f])) { allFinite = false; break; }
              if (!allFinite) {
                ++m_geomDiagFanoutInstBadFinite;
                continue;
              }
              if (m[0] == 0.f && m[1] == 0.f && m[2] == 0.f && m[3] == 0.f) {
                ++m_geomDiagFanoutInstZeroRow0;
                continue;
              }

              // NV-DXVK (fanout+camOrigin): t31 stores
              // objectToCameraRelative — a float3x4 whose translation column
              // is (mesh_world - cameraOrigin). Applying it to BLAS (plain-
              // decoded local positions) produces (world - cam), i.e. camera-
              // relative world. Remix's worldToView (kCameraAtOrigin=false)
              // expects absolute-world input and subtracts cam itself, so if
              // we leave BSP in camera-relative space, w2v double-subtracts
              // and geometry lands at (world - 2·cam) — usually entirely
              // behind the player. Shift to absolute world by adding
              // +cameraOrigin to the translation column.
              //
              // (Verified via DXBC disassembly of VS_597b7e49…: the VS does
              // clip = cb2.c_cameraRelativeToClip × (objectToCameraRelative ×
              // local + 1), which by construction produces camera-relative
              // world pre-projection. c_cameraOrigin is [unused] in the VS
              // itself — only the CB layout declares it — so reading cb2@4
              // here is safe and always gives the current camera pose.)
              const float adjTx = haveCamOrigin ? (m[3]  + camOrigin[0]) : m[3];
              const float adjTy = haveCamOrigin ? (m[7]  + camOrigin[1]) : m[7];
              const float adjTz = haveCamOrigin ? (m[11] + camOrigin[2]) : m[11];
              tforms->push_back(Matrix4(
                Vector4(m[0], m[4], m[8],  0.0f),
                Vector4(m[1], m[5], m[9],  0.0f),
                Vector4(m[2], m[6], m[10], 0.0f),
                Vector4(adjTx, adjTy, adjTz, 1.0f)));
            }

            if (dumpThisDraw) {
              std::string vsHash3 = "?";
              auto vsPtr3 = m_context->m_state.vs.shader;
              if (vsPtr3 != nullptr && vsPtr3->GetCommonShader() != nullptr) {
                auto& s = vsPtr3->GetCommonShader()->GetShader();
                if (s != nullptr) vsHash3 = s->getShaderKey().toString();
              }
              Logger::info(str::format(
                "[D3D11Rtx] InstIdxDump VS=", vsHash3,
                " maxInst=", maxInstances, " stride=", stride, " boneOff=", boneOff,
                " t31Len=", t31Len,
                " idx:", idxDumpLine,
                " t31_T:", t31DumpLine));
            }
            // NV-DXVK: scene dump. After 5s of gameplay, dump per-instance
            // BSP geometry to OBJ. Skips skinned characters (their VBs aren't
            // in immutable storage we can read here).
            if (isModelInstFanout && !tforms->empty() && SceneDump::shouldDumpThisFrame()) {
              std::lock_guard<std::mutex> lk(SceneDump::g_mutex);
              const bool firstOpen = !SceneDump::g_obj.is_open();
              SceneDump::open();
              if (firstOpen && SceneDump::g_obj.is_open()) {
                SceneDump::writeCameraMarker();
              }
              if (SceneDump::g_obj.is_open()) {
                // Find the position semantic + its VB.
                const D3D11RtxSemantic* posS = nullptr;
                for (const auto& s : semantics) {
                  if (!s.perInstance && s.format == VK_FORMAT_R32G32_UINT) { posS = &s; break; }
                }
                // Read VB + IB via immutable cache (BSP buffers are typically IMMUTABLE).
                const uint8_t* posData = nullptr; size_t posLen = 0;
                if (posS) {
                  const auto& pvb = m_context->m_state.ia.vertexBuffers[posS->inputSlot];
                  if (pvb.buffer != nullptr) {
                    const auto& imm = pvb.buffer->GetImmutableData();
                    if (!imm.empty()) {
                      posData = imm.data() + pvb.offset + posS->byteOffset;
                      posLen  = imm.size() - (pvb.offset + posS->byteOffset);
                    }
                  }
                }
                const uint8_t* idxData = nullptr; size_t idxLen = 0;
                VkIndexType ixType = VK_INDEX_TYPE_UINT16;
                if (indexed) {
                  const auto& ib = m_context->m_state.ia.indexBuffer;
                  if (ib.buffer != nullptr) {
                    const auto& imm = ib.buffer->GetImmutableData();
                    if (!imm.empty()) {
                      idxData = imm.data() + ib.offset;
                      idxLen  = imm.size() - ib.offset;
                      // ib.format is DXGI_FORMAT, map to VkIndexType.
                      ixType  = (ib.format == DXGI_FORMAT_R16_UINT)
                                  ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                    }
                  }
                }
                if (posData && (!indexed || idxData)) {
                  const uint32_t posStride = posS ? std::max<uint32_t>(8u, m_context->m_state.ia.vertexBuffers[posS->inputSlot].stride) : 8u;
                  // Decode constants — match the VS shader: scale 1/1024, bias (-1024,-1024,-2048)
                  const float kScale  = 1.0f / 1024.0f;
                  const float kBiasZ  = -2048.0f;
                  for (uint32_t inst = 0; inst < tforms->size(); ++inst) {
                    const Matrix4& T = (*tforms)[inst];
                    SceneDump::g_obj << "o BSP_" << SceneDump::g_objectsWritten++
                                     << "_inst" << inst << "\n";
                    // Determine vertex count: use either count (non-indexed) or
                    // max index seen + 1 (indexed). For simplicity, dump the first
                    // 'count' vertices for non-indexed; for indexed, dump every
                    // referenced vertex as positions and emit triangles via faces.
                    if (!indexed) {
                      for (uint32_t v = 0; v < count; ++v) {
                        size_t off = static_cast<size_t>(v) * posStride;
                        if (off + 8 > posLen) break;
                        const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
                        uint32_t xi = SceneDump::decodeX(up[0]);
                        uint32_t yi = SceneDump::decodeY(up[0], up[1]);
                        uint32_t zi = SceneDump::decodeZ(up[1]);
                        float lx = float(xi) * kScale - 1024.0f;
                        float ly = float(yi) * kScale - 1024.0f;
                        float lz = float(zi) * kScale + kBiasZ;
                        float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
                        float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
                        float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
                        SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
                      }
                      const uint32_t triCount = count / 3;
                      for (uint32_t t = 0; t < triCount; ++t) {
                        uint32_t a = SceneDump::g_baseVtx + t * 3 + 1;
                        SceneDump::g_obj << "f " << a << " " << (a+1) << " " << (a+2) << "\n";
                      }
                      SceneDump::g_baseVtx += count;
                    } else {
                      // Indexed: scan to find max used vertex, emit those, then faces.
                      const uint32_t idxStride = (ixType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
                      uint32_t maxV = 0;
                      for (uint32_t i = 0; i < count; ++i) {
                        size_t io = static_cast<size_t>(start + i) * idxStride;
                        if (io + idxStride > idxLen) { maxV = 0; break; }
                        uint32_t idx = (idxStride == 2)
                          ? *reinterpret_cast<const uint16_t*>(idxData + io)
                          : *reinterpret_cast<const uint32_t*>(idxData + io);
                        idx += static_cast<uint32_t>(std::max(base, 0));
                        if (idx > maxV) maxV = idx;
                      }
                      const uint32_t vCount = maxV + 1;
                      for (uint32_t v = 0; v < vCount; ++v) {
                        size_t off = static_cast<size_t>(v) * posStride;
                        if (off + 8 > posLen) break;
                        const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
                        uint32_t xi = SceneDump::decodeX(up[0]);
                        uint32_t yi = SceneDump::decodeY(up[0], up[1]);
                        uint32_t zi = SceneDump::decodeZ(up[1]);
                        float lx = float(xi) * kScale - 1024.0f;
                        float ly = float(yi) * kScale - 1024.0f;
                        float lz = float(zi) * kScale + kBiasZ;
                        float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
                        float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
                        float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
                        SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
                      }
                      const uint32_t triCount = count / 3;
                      for (uint32_t t = 0; t < triCount; ++t) {
                        uint32_t i0base = (start + t * 3);
                        size_t i0o = static_cast<size_t>(i0base + 0) * idxStride;
                        size_t i1o = static_cast<size_t>(i0base + 1) * idxStride;
                        size_t i2o = static_cast<size_t>(i0base + 2) * idxStride;
                        if (i2o + idxStride > idxLen) break;
                        uint32_t i0 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i0o) : *reinterpret_cast<const uint32_t*>(idxData + i0o);
                        uint32_t i1 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i1o) : *reinterpret_cast<const uint32_t*>(idxData + i1o);
                        uint32_t i2 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i2o) : *reinterpret_cast<const uint32_t*>(idxData + i2o);
                        i0 += static_cast<uint32_t>(std::max(base, 0));
                        i1 += static_cast<uint32_t>(std::max(base, 0));
                        i2 += static_cast<uint32_t>(std::max(base, 0));
                        SceneDump::g_obj << "f " << (SceneDump::g_baseVtx + i0 + 1) << " "
                                                  << (SceneDump::g_baseVtx + i1 + 1) << " "
                                                  << (SceneDump::g_baseVtx + i2 + 1) << "\n";
                      }
                      SceneDump::g_baseVtx += vCount;
                    }
                  }
                }
              }
            }

            // DEBUG: distance to closest geometry from camera, per VS.
            // In the camera-relative world frame Remix uses, the camera sits
            // at the origin — so |T| is the distance from camera to that
            // instance. We also log the absolute-world camera origin (read
            // from cb2.c_cameraOrigin earlier) for cross-reference with the
            // GPU cull shader's `cameraPosition` (which comes from
            // CameraManager and is in absolute world).
            if (isModelInstFanout && !tforms->empty()) {
              // [SpawnGeomDiag] every-batch min/max accumulation so EndFrame
              // emits a frame-wide |T| range, not just the once-per-VS sample.
              {
                float minDistSq = std::numeric_limits<float>::max();
                float maxDistSq = 0.0f;
                for (const Matrix4& tm : *tforms) {
                  const float dx = tm[3][0], dy = tm[3][1], dz = tm[3][2];
                  const float ds = dx*dx + dy*dy + dz*dz;
                  if (ds < minDistSq) minDistSq = ds;
                  if (ds > maxDistSq) maxDistSq = ds;
                }
                const float minD = std::sqrt(minDistSq);
                const float maxD = std::sqrt(maxDistSq);
                if (!m_geomDiagFanoutHaveDist) {
                  m_geomDiagFanoutMinDist = minD;
                  m_geomDiagFanoutMaxDist = maxD;
                  m_geomDiagFanoutHaveDist = true;
                } else {
                  if (minD < m_geomDiagFanoutMinDist) m_geomDiagFanoutMinDist = minD;
                  if (maxD > m_geomDiagFanoutMaxDist) m_geomDiagFanoutMaxDist = maxD;
                }
              }
              static std::unordered_set<uintptr_t> sDistLogged;
              auto vsPtr7 = m_context->m_state.vs.shader;
              uintptr_t key7 = (vsPtr7 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr7.ptr()) : 0;
              if (key7 && sDistLogged.size() < 12 && sDistLogged.insert(key7).second) {
                ++m_geomDiagBspDistSamples;
                float minDistSq = std::numeric_limits<float>::max();
                float maxDistSq = 0.0f;
                for (const Matrix4& tm : *tforms) {
                  const float dx = tm[3][0], dy = tm[3][1], dz = tm[3][2];
                  const float ds = dx*dx + dy*dy + dz*dz;
                  if (ds < minDistSq) minDistSq = ds;
                  if (ds > maxDistSq) maxDistSq = ds;
                }
                std::string vsHash7 = "?";
                if (vsPtr7->GetCommonShader() != nullptr) {
                  auto& s = vsPtr7->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash7 = s->getShaderKey().toString();
                }
                Logger::info(str::format(
                  "[D3D11Rtx] BSP dist VS=", vsHash7,
                  " tforms=", tforms->size(),
                  " closest=", std::sqrt(minDistSq),
                  " farthest=", std::sqrt(maxDistSq),
                  " camOriginAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                  " (camera in our frame is at origin; cullingRadius default is 5000)"));
              }
            }

            // NV-DXVK [VanishDiag-T]: frame-throttled per-fanout T-vector
            // spatial extent log. Answers "are the origins really far out?"
            // by reporting per-component min/max of T plus a |T| histogram.
            // If T's are camera-relative (the assumption everywhere else in
            // the renderer), magnitudes should be roughly bounded by the
            // visible map radius. If T's are in absolute world space, bounds
            // will look like the level's world coords (often 5-digit numbers
            // shifted away from origin).
            // Gating: skip if no fanout, only emit while in gameplay
            // (m_boneInstFrameId increments only during real frames), and cap
            // emissions per frame so we don't flood the log.
            if (isModelInstFanout && !tforms->empty()) {
              static uint32_t sVdtLastFrame = UINT32_MAX;
              static uint32_t sVdtPerFrame  = 0;
              static uint32_t sVdtTotal     = 0;
              const uint32_t curFrame = m_boneInstFrameId;
              if (curFrame != sVdtLastFrame) {
                sVdtLastFrame = curFrame;
                sVdtPerFrame  = 0;
              }
              // Per-frame cap of 8 fanouts; total session cap of 2000 to avoid
              // unbounded growth if the game runs for hours.
              if (sVdtPerFrame < 8 && sVdtTotal < 2000) {
                ++sVdtPerFrame;
                ++sVdtTotal;
                float tMinX = std::numeric_limits<float>::max();
                float tMinY = std::numeric_limits<float>::max();
                float tMinZ = std::numeric_limits<float>::max();
                float tMaxX = -std::numeric_limits<float>::max();
                float tMaxY = -std::numeric_limits<float>::max();
                float tMaxZ = -std::numeric_limits<float>::max();
                float dMin = std::numeric_limits<float>::max();
                float dMax = 0.0f;
                double dSum = 0.0;
                uint32_t bLt5k = 0, bLt10k = 0, bLt20k = 0, bLt50k = 0, bGe50k = 0;
                for (const Matrix4& tm : *tforms) {
                  const float tx = tm[3][0], ty = tm[3][1], tz = tm[3][2];
                  if (tx < tMinX) tMinX = tx; if (tx > tMaxX) tMaxX = tx;
                  if (ty < tMinY) tMinY = ty; if (ty > tMaxY) tMaxY = ty;
                  if (tz < tMinZ) tMinZ = tz; if (tz > tMaxZ) tMaxZ = tz;
                  const float d = std::sqrt(tx*tx + ty*ty + tz*tz);
                  if (d < dMin) dMin = d;
                  if (d > dMax) dMax = d;
                  dSum += d;
                  if      (d <  5000.0f) ++bLt5k;
                  else if (d < 10000.0f) ++bLt10k;
                  else if (d < 20000.0f) ++bLt20k;
                  else if (d < 50000.0f) ++bLt50k;
                  else                   ++bGe50k;
                }
                const float dAvg = static_cast<float>(dSum / double(tforms->size()));
                std::string vsHashV = "?";
                auto vsPtrV = m_context->m_state.vs.shader;
                if (vsPtrV != nullptr && vsPtrV->GetCommonShader() != nullptr) {
                  auto& sV = vsPtrV->GetCommonShader()->GetShader();
                  if (sV != nullptr) vsHashV = sV->getShaderKey().toString();
                }
                Logger::info(str::format(
                  "[VanishDiag-T] frame=", curFrame,
                  " VS=", vsHashV,
                  " tforms=", tforms->size(),
                  " camOriginAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                  " T.x=[", tMinX, "..", tMaxX, "]",
                  " T.y=[", tMinY, "..", tMaxY, "]",
                  " T.z=[", tMinZ, "..", tMaxZ, "]",
                  " |T| min=", dMin, " avg=", dAvg, " max=", dMax,
                  " bins(<5k/<10k/<20k/<50k/>=50k)=",
                  bLt5k, "/", bLt10k, "/", bLt20k, "/", bLt50k, "/", bGe50k));
              }
            }
            if (!tforms->empty()) {
              // NV-DXVK: log EVERY fanout submit (cap ~40 per session) so we can
              // see which camera context each PI batch belongs to — main view vs
              // shadow cascade. camOriginAbs is the absolute-world c_cameraOrigin
              // read from the VS's CBufCommonPerCamera; its value distinguishes
              // main camera from shadow cascades in TF2.
              static uint32_t sFanoutLogCount = 0;
              auto vsPtr = m_context->m_state.vs.shader;
              if (sFanoutLogCount < 40) {
                ++sFanoutLogCount;
                std::string vsHash = "?";
                if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
                  auto& s = vsPtr->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash = s->getShaderKey().toString();
                }
                Logger::info(str::format(
                  "[D3D11Rtx] FanoutSubmit #", sFanoutLogCount,
                  " VS=", vsHash,
                  " isModelInst=", isModelInstFanout ? 1 : 0,
                  " usedSlot=", usedSlot,
                  " tforms=", tforms->size(),
                  " idxFmt=", uint32_t(boneIdxSem->format),
                  " camOriginAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                  " sample T0=(", (*tforms)[0][3][0], ",", (*tforms)[0][3][1], ",", (*tforms)[0][3][2], ")"));
              }
              // Keep alive via ring buffer
              if (m_boneTransformRing.empty()) m_boneTransformRing.resize(4);
              m_boneTransformRing[m_boneInstFrameId % 4].push_back(tforms);

              m_currentInstancesToObject = tforms.get();
              // NV-DXVK: Carry ownership alongside the raw pointer so the RtInstance
              // consuming this survives beyond the 4-frame ring buffer.
              m_currentInstancesToObjectOwner = tforms;
              m_boneInstanceCount = static_cast<uint32_t>(tforms->size());
              m_boneInstTotal += m_boneInstanceCount;
              ++m_geomDiagFanoutBatches;
              const uint32_t bsz = static_cast<uint32_t>(tforms->size());
              m_geomDiagFanoutTforms += bsz;
              if      (bsz == 0)    ++m_geomDiagFanoutBucket0;
              else if (bsz == 1)    ++m_geomDiagFanoutBucket1;
              else if (bsz <= 4)    ++m_geomDiagFanoutBucket4;
              else if (bsz <= 16)   ++m_geomDiagFanoutBucket16;
              else if (bsz <= 64)   ++m_geomDiagFanoutBucket64;
              else if (bsz <= 256)  ++m_geomDiagFanoutBucket256;
              else if (bsz <= 1024) ++m_geomDiagFanoutBucket1k;
              else                  ++m_geomDiagFanoutBucketBig;
              // [FloorTrace] Per-fanout-batch emit log. The existing
              // FanoutSubmit log caps at 40 per *session*, which means
              // after early frames it stops capturing — useless once we
              // need spawn-window data. This one caps per-frame and is
              // gated on detailedDump so it stays quiet outside debug
              // sessions. Pairs with [FloorTrace.recv] in scene_manager
              // to find which batches the engine emitted but Remix never
              // received. VS-hash + size + sampleT0 + camOriginAbs forms
              // the correlation key. Including ptrKey (the tforms
              // shared_ptr address) gives us a deterministic match if
              // sample T0 ever collides between similar batches.
              {
                static uint32_t sFloorTraceFrame = UINT32_MAX;
                static uint32_t sFloorTracePerFrame = 0;
                // Use the device frame counter so emit/recv frame IDs
                // line up with scene_manager's [FloorTrace.recv] (which
                // uses m_device->getCurrentFrameId()).
                const uint32_t curFrame =
                  (m_context != nullptr && m_context->m_device != nullptr)
                    ? m_context->m_device->getCurrentFrameId()
                    : m_boneInstFrameId;
                if (curFrame != sFloorTraceFrame) {
                  sFloorTraceFrame = curFrame;
                  sFloorTracePerFrame = 0;
                }
                if (sFloorTracePerFrame < 80) {
                  ++sFloorTracePerFrame;
                  // Print the raw 64-bit hash hex so emit/recv lines
                  // grep-match — scene_manager only has the raw hash
                  // (XXH64_hash_t), not the DxvkShaderKey toString().
                  uint64_t vsHashRaw = 0;
                  auto vsPtrFt = m_context->m_state.vs.shader;
                  if (vsPtrFt != nullptr && vsPtrFt->GetCommonShader() != nullptr) {
                    auto& s = vsPtrFt->GetCommonShader()->GetShader();
                    if (s != nullptr) vsHashRaw = static_cast<uint64_t>(s->getHash());
                  }
                  Logger::info(str::format(
                    "[FloorTrace.emit] frame=", curFrame,
                    " vsHash=0x", std::hex, vsHashRaw, std::dec,
                    " size=", bsz,
                    " ptrKey=0x", std::hex, reinterpret_cast<uintptr_t>(tforms.get()), std::dec,
                    " camAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                    " T0=(", (*tforms)[0][3][0], ",", (*tforms)[0][3][1], ",", (*tforms)[0][3][2], ")"));

                  // [FloorTrace.aabb] Decode the BLAS triangles of the
                  // FIRST instance and compute world-space AABB. This
                  // tells us *where the geometry actually is* after the
                  // BSP packed-uint decode + per-instance transform.
                  // If a batch with T0=(-5193,-52,-302) (floor-class
                  // anchor) decodes to triangles spanning some other
                  // region, the decode path is the bug. Throttled
                  // separately to first 24 batches per frame because
                  // decoding can touch hundreds of indices/vertices.
                  static uint32_t sFloorAabbPerFrame = 0;
                  static uint32_t sFloorAabbFrame   = UINT32_MAX;
                  if (curFrame != sFloorAabbFrame) {
                    sFloorAabbFrame = curFrame;
                    sFloorAabbPerFrame = 0;
                  }
                  if (sFloorAabbPerFrame < 24) {
                    ++sFloorAabbPerFrame;
                    // Find position semantic (R32G32_UINT, non-per-instance).
                    const D3D11RtxSemantic* posS_ft = nullptr;
                    for (const auto& s : semantics) {
                      if (!s.perInstance && s.format == VK_FORMAT_R32G32_UINT) { posS_ft = &s; break; }
                    }
                    const uint8_t* posData_ft = nullptr; size_t posLen_ft = 0;
                    uint32_t posStride_ft = 0;
                    if (posS_ft != nullptr && posS_ft->inputSlot < m_context->m_state.ia.vertexBuffers.size()) {
                      const auto& vb = m_context->m_state.ia.vertexBuffers[posS_ft->inputSlot];
                      if (vb.buffer != nullptr) {
                        const auto& imm = vb.buffer->GetImmutableData();
                        if (!imm.empty()) {
                          // Mirror SceneDump's offset math (line 1963): VB
                          // base + binding offset + per-semantic byteOffset.
                          // Skipping byteOffset would read TEXCOORD/COLOR
                          // bytes as positions on layouts where they share
                          // a slot.
                          const size_t baseOff = static_cast<size_t>(vb.offset) +
                                                 static_cast<size_t>(posS_ft->byteOffset);
                          if (baseOff < imm.size()) {
                            posData_ft = imm.data() + baseOff;
                            posLen_ft  = imm.size() - baseOff;
                            posStride_ft = std::max<uint32_t>(8u, vb.stride);
                          }
                        }
                      }
                    }
                    const uint8_t* idxData_ft = nullptr; size_t idxLen_ft = 0;
                    uint32_t idxStride_ft = 0;
                    if (indexed) {
                      const auto& ib = m_context->m_state.ia.indexBuffer;
                      if (ib.buffer != nullptr) {
                        const auto& imm = ib.buffer->GetImmutableData();
                        if (!imm.empty()) {
                          idxData_ft = imm.data() + ib.offset;
                          idxLen_ft  = imm.size() - ib.offset;
                          idxStride_ft = (ib.format == DXGI_FORMAT_R16_UINT) ? 2u : 4u;
                        }
                      }
                    }
                    bool aabbValid = false;
                    float aMinX =  std::numeric_limits<float>::max();
                    float aMinY =  std::numeric_limits<float>::max();
                    float aMinZ =  std::numeric_limits<float>::max();
                    float aMaxX = -std::numeric_limits<float>::max();
                    float aMaxY = -std::numeric_limits<float>::max();
                    float aMaxZ = -std::numeric_limits<float>::max();
                    float lMinX =  std::numeric_limits<float>::max();
                    float lMinY =  std::numeric_limits<float>::max();
                    float lMinZ =  std::numeric_limits<float>::max();
                    float lMaxX = -std::numeric_limits<float>::max();
                    float lMaxY = -std::numeric_limits<float>::max();
                    float lMaxZ = -std::numeric_limits<float>::max();
                    uint32_t sampledV = 0;
                    if (posData_ft != nullptr) {
                      const Matrix4& T = (*tforms)[0];
                      // Sample up to first 192 indices (= 64 triangles)
                      // for indexed; or first 192 vertices for non-
                      // indexed. Decode constants match the in-shader
                      // BSP unpacker (kScale=1/1024, kBias=-1024 for
                      // X/Y, -2048 for Z).
                      const float kScale = 1.0f / 1024.0f;
                      const uint32_t kCap = std::min<uint32_t>(192u, count);
                      auto decodeOne = [&](uint32_t v) {
                        size_t off = static_cast<size_t>(v) * posStride_ft;
                        if (off + 8 > posLen_ft) return;
                        const uint32_t* up = reinterpret_cast<const uint32_t*>(posData_ft + off);
                        const uint32_t xi = SceneDump::decodeX(up[0]);
                        const uint32_t yi = SceneDump::decodeY(up[0], up[1]);
                        const uint32_t zi = SceneDump::decodeZ(up[1]);
                        const float lx = float(xi) * kScale - 1024.0f;
                        const float ly = float(yi) * kScale - 1024.0f;
                        const float lz = float(zi) * kScale - 2048.0f;
                        if (lx < lMinX) lMinX = lx; if (lx > lMaxX) lMaxX = lx;
                        if (ly < lMinY) lMinY = ly; if (ly > lMaxY) lMaxY = ly;
                        if (lz < lMinZ) lMinZ = lz; if (lz > lMaxZ) lMaxZ = lz;
                        const float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
                        const float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
                        const float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
                        if (wx < aMinX) aMinX = wx; if (wx > aMaxX) aMaxX = wx;
                        if (wy < aMinY) aMinY = wy; if (wy > aMaxY) aMaxY = wy;
                        if (wz < aMinZ) aMinZ = wz; if (wz > aMaxZ) aMaxZ = wz;
                        ++sampledV;
                      };
                      if (indexed && idxData_ft != nullptr) {
                        for (uint32_t i = 0; i < kCap; ++i) {
                          size_t io = static_cast<size_t>(start + i) * idxStride_ft;
                          if (io + idxStride_ft > idxLen_ft) break;
                          uint32_t idx = (idxStride_ft == 2)
                            ? *reinterpret_cast<const uint16_t*>(idxData_ft + io)
                            : *reinterpret_cast<const uint32_t*>(idxData_ft + io);
                          idx += static_cast<uint32_t>(std::max(base, 0));
                          decodeOne(idx);
                        }
                      } else {
                        for (uint32_t v = 0; v < kCap; ++v) decodeOne(v);
                      }
                      aabbValid = (sampledV > 0);
                    }
                    if (aabbValid) {
                      Logger::info(str::format(
                        "[FloorTrace.aabb] frame=", curFrame,
                        " vsHash=0x", std::hex, vsHashRaw, std::dec,
                        " ptrKey=0x", std::hex, reinterpret_cast<uintptr_t>(tforms.get()), std::dec,
                        " sampled=", sampledV, "/", count,
                        " idx=", (indexed ? 1 : 0),
                        " posStride=", posStride_ft,
                        " local=[(", lMinX, ",", lMinY, ",", lMinZ, ")..(",
                                       lMaxX, ",", lMaxY, ",", lMaxZ, ")]",
                        " worldInst0=[(", aMinX, ",", aMinY, ",", aMinZ, ")..(",
                                          aMaxX, ",", aMaxY, ",", aMaxZ, ")]"));
                    } else {
                      Logger::info(str::format(
                        "[FloorTrace.aabb] frame=", curFrame,
                        " vsHash=0x", std::hex, vsHashRaw, std::dec,
                        " ptrKey=0x", std::hex, reinterpret_cast<uintptr_t>(tforms.get()), std::dec,
                        " — could not decode (posS=", (posS_ft ? 1 : 0),
                        " posData=", (posData_ft ? 1 : 0),
                        " indexed=", (indexed ? 1 : 0),
                        " idxData=", (idxData_ft ? 1 : 0), ")"));
                    }
                  }
                }
              }
              SubmitDraw(indexed, count, start, base);
              m_boneInstanceCount = 0;
              m_currentInstancesToObject = nullptr;
              m_currentInstancesToObjectOwner.reset();
            }
            handledAsBoneInstancing = true;
          } else {
            m_boneInstNoCache++;
            handledAsBoneInstancing = true;
          }
        }
      }

      // Old single-draw bone path removed — handled by async extract above.

      if (!handledAsBoneInstancing) {
        static uint32_t sNoInstXformLog = 0;
        if (sNoInstXformLog < 3) {
          ++sNoInstXformLog;
          Logger::info(str::format("[D3D11Rtx] Instanced draw (", instanceCount,
                                   " instances) has no per-instance transform (", instRows.size(),
                                   " float4 rows). Submitting single draw."));
        }
        SubmitDraw(indexed, count, start, base);
      }
      return;
    }

    // Read the instance buffer
    const auto& vb = m_context->m_state.ia.vertexBuffers[instSlot];
    if (vb.buffer == nullptr) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    DxvkBufferSlice instBufSlice = vb.buffer->GetBufferSlice(vb.offset);
    const uint32_t instStride = vb.stride;
    const size_t instBufLen = instBufSlice.length();
    if (instStride == 0) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    const UINT maxInstances = instanceCount;

    static uint32_t sInstLog = 0;
    if (sInstLog < 3) {
      ++sInstLog;
      Logger::info(str::format("[D3D11Rtx] Instanced draw: ", instanceCount,
                               " instances, ", instRows.size(), " float4 rows in slot ",
                               instSlot, ", stride=", instStride));
    }

    for (UINT i = 0; i < maxInstances; ++i) {
      UINT instIdx = startInstance + i;
      size_t instOffset = static_cast<size_t>(instIdx) * instStride;

      // Read 3 or 4 float4 rows to build a world matrix.
      // Row layout: each row is at instOffset + row.byteOffset within the instance buffer.
      float rows[4][4] = {};
      bool valid = true;

      for (size_t r = 0; r < std::min<size_t>(instRows.size(), 4); ++r) {
        size_t rowOff = instOffset + instRows[r].byteOffset;
        if (rowOff + 16 > instBufLen) { valid = false; break; }
        const void* ptr = instBufSlice.mapPtr(rowOff);
        if (!ptr) { valid = false; break; }
        std::memcpy(rows[r], ptr, 16);
        for (int c = 0; c < 4; ++c) {
          if (!std::isfinite(rows[r][c])) { valid = false; break; }
        }
        if (!valid) break;
      }

      if (!valid) continue;

      // If only 3 rows, the 4th row is (0,0,0,1) — affine transform.
      if (instRows.size() == 3) {
        rows[3][0] = 0.f; rows[3][1] = 0.f; rows[3][2] = 0.f; rows[3][3] = 1.f;
      }

      Matrix4 instMatrix(
        Vector4(rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
        Vector4(rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
        Vector4(rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
        Vector4(rows[3][0], rows[3][1], rows[3][2], rows[3][3]));

      SubmitDraw(indexed, count, start, base, &instMatrix);
    }
    } catch (const std::exception& e) {
      Logger::err(str::format("[D3D11Rtx] CRASH in SubmitInstancedDraw: ", e.what()));
    } catch (...) {
      Logger::err("[D3D11Rtx] CRASH in SubmitInstancedDraw: unknown exception");
    }
  }

  // Read a row-major float4x4 from a mapped cbuffer.  Returns identity on bounds violation
  // or if any element is NaN/Inf (corrupt GPU memory, emulator artifacts, etc.).
  static Matrix4 readCbMatrix(const uint8_t* ptr, size_t offset, size_t bufSize) {
    if (offset + 64 > bufSize)
      return Matrix4();
    float raw[4][4];
    std::memcpy(raw, ptr + offset, 64);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        if (!std::isfinite(raw[r][c]))
          return Matrix4();
    return Matrix4(
      Vector4(raw[0][0], raw[0][1], raw[0][2], raw[0][3]),
      Vector4(raw[1][0], raw[1][1], raw[1][2], raw[1][3]),
      Vector4(raw[2][0], raw[2][1], raw[2][2], raw[2][3]),
      Vector4(raw[3][0], raw[3][1], raw[3][2], raw[3][3]));
  }

  // Detect a perspective projection matrix in either memory layout.
  //
  // Row-major layout (D3D standard, CryEngine, id Tech, Source):
  //   m[0] = [±Sx, 0,   0,    0  ]
  //   m[1] = [0,  ±Sy,  0,    0  ]
  //   m[2] = [Jx,  Jy,  Q,   ±1 ]  ← perspective-divide at m[2][3]
  //   m[3] = [0,   0,   Wz,   0  ]
  //
  // Column-major read as row-major (UE4/UE5, Unity, Godot):
  //   m[0] = [±Sx, 0,   0,    0  ]
  //   m[1] = [0,  ±Sy,  0,    0  ]
  //   m[2] = [Jx,  Jy,  Q,   Wz ]  ← m[2][3] = nearPlane or 0
  //   m[3] = [0,   0,  ±1,    0  ]  ← perspective-divide at m[3][2]
  //
  // Returns: 0 = not perspective, 1 = row-major pure P, 2 = column-major-as-row pure P,
  //          3 = row-major combined View*Proj, 4 = column-major combined View*Proj.
  //
  // allowCombinedVP: when false, only cls 1/2 (pure projection) are returned.
  // This prevents false positives from degenerate matrices on splash/UI
  // frames that happen to have m[2][3]≈±1 and m[3][3]≈0 but are NOT real
  // view*projection matrices.  The caller should set this to true only when
  // the current frame has enough draws to be confident it's gameplay.
  static int classifyPerspective(const Matrix4& m, bool allowCombinedVP = true) {
    constexpr float kTol = 0.02f;
    constexpr float kJitterTol = 0.15f;

    // ---- Pure projection (diagonal rows 0-1) ----
    // These match a standalone projection matrix that the engine stores
    // separately from the view transform.  Rows 0-1 must be diagonal
    // (no off-axis terms), which is true for standard D3D perspective.
    const bool diag01 =
        std::abs(m[0][1]) <= kTol && std::abs(m[0][2]) <= kTol && std::abs(m[0][3]) <= kTol &&
        std::abs(m[1][0]) <= kTol && std::abs(m[1][2]) <= kTol && std::abs(m[1][3]) <= kTol &&
        std::abs(m[0][0]) >= 0.1f && std::abs(m[1][1]) >= 0.1f;

    if (diag01) {
      // Row-major check: m[2][3] ≈ ±1, m[3][3] ≈ 0.
      const bool r23 = std::abs(std::abs(m[2][3]) - 1.0f) < kTol;
      const bool r33z = std::abs(m[3][3]) < kTol;
      if (r23 && r33z) {
        if (std::abs(m[2][0]) <= kJitterTol && std::abs(m[2][1]) <= kJitterTol &&
            std::abs(m[3][0]) <= kTol && std::abs(m[3][1]) <= kTol)
          return 1;
      }

      // Column-major-as-row check: m[3][2] ≈ ±1, m[3][3] ≈ 0.
      const bool c32 = std::abs(std::abs(m[3][2]) - 1.0f) < kTol;
      const bool c33z = std::abs(m[3][3]) < kTol;
      if (c32 && c33z) {
        if (std::abs(m[2][0]) <= kJitterTol && std::abs(m[2][1]) <= kJitterTol &&
            std::abs(m[3][0]) <= kTol && std::abs(m[3][1]) <= kTol)
          return 2;
      }
    }

    // ---- Combined View*Projection (Source engine, Titanfall 2, etc.) ----
    //
    // Many engines (especially Source-based ones) store a pre-multiplied
    // View × Projection matrix in the VS cbuffer instead of separate V
    // and P matrices.  The view rotation is baked into ALL rows, so the
    // off-diagonal elements in rows 0-2 are large (they encode the camera
    // basis scaled by FOV and depth range).  The only invariant that
    // survives the multiplication is the perspective-divide signature:
    //
    //   Row-major VP:    m[2][3] ≈ ±1,  m[3][3] ≈ 0
    //   Col-major VP:    m[3][2] ≈ ±1,  m[3][3] ≈ 0
    //
    // We add a few lightweight sanity checks to avoid false positives:
    //   * All 16 entries must be finite (reject NaN/Inf garbage)
    //   * At least one entry in the first two rows must have magnitude
    //     > 0.01 (reject zero/near-zero matrices)
    //
    // When ExtractTransforms sees cls == 3 or 4, it treats the matrix as
    // a combined VP and uses it as viewToProjection directly, with
    // worldToView set to identity (since the view is already baked in).
    {
      // Finite-value check (reject garbage / padding / uninitialised data)
      bool allFinite = true;
      bool anySignificant = false;
      for (int r = 0; r < 4 && allFinite; ++r) {
        for (int c = 0; c < 4; ++c) {
          if (!std::isfinite(m[r][c])) { allFinite = false; break; }
          if (r < 2 && std::abs(m[r][c]) > 0.01f) anySignificant = true;
        }
      }

      if (allFinite && anySignificant && allowCombinedVP) {
        // Additional sanity: a real VP matrix has rows 0-1 with substantial
        // magnitudes (they encode camera right/up scaled by Sx/Sy ≈ 0.3-3.0
        // for typical FOVs).  Reject near-zero rows that would produce
        // degenerate Sx/Sy (≈0) and cause the decomposition to output
        // garbage (fwd=(0,0,-1), pos=(0,0,0)).  This prevents false
        // positives on orthographic/identity-like matrices that happen to
        // have the right signature in m[2][3] and m[3][3] (e.g. the 2D UI
        // ortho projection that Source stores at offset 0 of the same
        // cbuffer that holds the real VP at offset 96).
        constexpr float kMinRowMag = 0.1f;
        const float magR0 = std::sqrt(m[0][0]*m[0][0] + m[0][1]*m[0][1] + m[0][2]*m[0][2]);
        const float magR1 = std::sqrt(m[1][0]*m[1][0] + m[1][1]*m[1][1] + m[1][2]*m[1][2]);

        // Additional: at least one of rows 0-1 must have magnitude that
        // DIFFERS from 1.0 by more than 0.05.  A real VP matrix has
        // Sx = cot(fovY/2)/aspect and Sy = cot(fovY/2), which only both
        // equal 1.0 for the unlikely case of exactly 90° FOV on a 1:1
        // aspect display.  False positives from identity-like parameter
        // matrices (common in Source cbuffer slot 0) have BOTH row
        // magnitudes at exactly 1.0, which this check rejects.
        // The strongest false-positive filter: the extracted projection
        // scales Sx (= magR0) and Sy (= magR1) encode the FOV and
        // aspect ratio.  For a real VP matrix:
        //   Sx = cot(fovY/2) / viewportAspect
        //   Sy = cot(fovY/2)
        // So Sy/Sx ≈ viewportAspect (within ~20% to account for
        // non-square pixels, guard bands, etc.).
        //
        // False positives from game-parameter cbuffers have random
        // Sx/Sy ratios that almost never match the viewport.
        //
        // We can't access the viewport from this static function, so
        // we use the most common gaming aspect ratios (16:9 = 1.778,
        // 16:10 = 1.6, 21:9 = 2.333, 4:3 = 1.333) and accept if the
        // ratio is within the plausible range [1.0, 3.0].  This rejects
        // ratios like 0.25 or 4.0 that come from false positives.
        const float devFromUnit = std::max(
            std::abs(magR0 - 1.0f),
            std::abs(magR1 - 1.0f));
        const float aspectRatio = (magR0 > 0.001f) ? (magR1 / magR0) : 0.0f;

        if (magR0 >= kMinRowMag && magR1 >= kMinRowMag
            && devFromUnit > 0.05f
            && aspectRatio >= 1.0f && aspectRatio <= 3.0f) {
          // Row-major combined VP: m[2][3] ≈ ±1, m[3][3] ≈ 0
          if (std::abs(std::abs(m[2][3]) - 1.0f) < kTol && std::abs(m[3][3]) < kTol)
            return 3;

          // Column-major combined VP: m[3][2] ≈ ±1, m[3][3] ≈ 0
          if (std::abs(std::abs(m[3][2]) - 1.0f) < kTol && std::abs(m[3][3]) < kTol)
            return 4;
        }
      }
    }

    return 0;
  }

  // Return true if m looks like a camera view matrix (rigid-body: rotation + translation).
  // Expects row-major convention (or column-major already transposed by the caller).
  // The upper-left 3×3 should be approximately orthonormal and the last column [0,0,0,1].
  static bool isViewMatrix(const Matrix4& m) {
    // Row 3 must be [*, *, *, 1] (affine).
    if (std::abs(m[3][3] - 1.0f) > 0.01f) return false;
    // Columns 0-2 of rows 0-2 should have unit length (orthonormal rotation).
    for (int col = 0; col < 3; ++col) {
      float lenSq = m[0][col] * m[0][col] + m[1][col] * m[1][col] + m[2][col] * m[2][col];
      if (std::abs(lenSq - 1.0f) > 0.1f) return false;
    }
    // m[0][3], m[1][3], m[2][3] should be 0 (no perspective warp).
    if (std::abs(m[0][3]) > 0.01f || std::abs(m[1][3]) > 0.01f || std::abs(m[2][3]) > 0.01f)
      return false;
    // Reject identity — identity means "no view transform" which is not useful.
    if (isIdentityExact(m)) return false;
    return true;
  }

  DrawCallTransforms D3D11Rtx::ExtractTransforms() {
    DrawCallTransforms transforms;

    // NV-DXVK: Reset per-call.  Will be set to true below only if no real
    // perspective matrix is found in any cbuffer and the viewport fallback
    // block ends up running.  SubmitDraw reads this immediately after the
    // call returns to decide whether the draw is UI (fallback was used =>
    // skip RTX submission, let the native raster path handle it).
    m_lastExtractUsedFallback = false;
    m_lastClassifierSaidUi = false;
    m_currentDrawIsBoneTransformed = false;
    m_lastDrawCamOriginSet = false;
    m_lastWtvPathId = 0;
    m_lastO2wPathId = 0;
    m_skipViewMatrixScan = false;

    // NV-DXVK: SHADOW CLASSIFICATION — run the new pure classifier and log
    // what it says for each unique VS. This does NOT alter current behavior;
    // it only emits one log line per VS so we can A/B verify the classifier
    // matches reality before swapping the dispatcher over. Expected output:
    //   VS_6e3e6f28... → StaticWorld (rdef_cb3_CBufModelInstance)
    //   VS_ef94e6c7... → SkinnedChar (sem_blendindices_canonical_t30)
    //   VS_597b7e49... → InstancedBsp (sem_uint4+rdef_g_modelInst)
    //   VS_8027c7a1... → UI (no_signals)  [menu shaders]
    {
      auto vsPtrC = m_context->m_state.vs.shader;
      if (vsPtrC != nullptr) {
        const D3D11CommonShader* common = vsPtrC->GetCommonShader();
        static std::unordered_set<uintptr_t> sShadowLogged;
        uintptr_t key = reinterpret_cast<uintptr_t>(vsPtrC.ptr());
        if (sShadowLogged.insert(key).second) {
          const auto* il = m_context->m_state.ia.inputLayout.ptr();
          const std::vector<D3D11RtxSemantic> kEmpty;
          const auto& sems = il ? il->GetRtxSemantics() : kEmpty;
          auto cls = D3D11VsClassifier::classify(common, sems);
          std::string vsName = "?";
          if (common != nullptr) {
            auto& s = common->GetShader();
            if (s != nullptr) vsName = s->getShaderKey().toString();
          }
          Logger::info(str::format(
            "[VsClass] vs=", vsName,
            " kind=", D3D11VsClassifier::kindName(cls.kind),
            " reason=", cls.reason,
            " cb3=", cls.cb3Slot,
            " modelInst=", cls.modelInstSlot, (cls.modelInstFromRdef ? "(rdef)" : ""),
            " bonePal=",   cls.bonePaletteSlot, (cls.bonePaletteFromRdef ? "(rdef)" : "")));
        }
      }
    }

    // NV-DXVK: helper — read current bound VS hash (for per-path logging).
    // Returns truncated 16-char lowercase hex string or "<novs>".
    auto getVsHashShort = [this]() -> std::string {
      auto vsPtr = m_context->m_state.vs.shader;
      if (vsPtr == nullptr || vsPtr->GetCommonShader() == nullptr) return "<novs>";
      auto& s = vsPtr->GetCommonShader()->GetShader();
      if (s == nullptr) return "<novs>";
      std::string full = s->getShaderKey().toString();
      // Format is typically "VS_<40hexchars>". Shorten to first 19 chars
      // (VS_ + 16 hex) for log readability.
      return full.substr(0, std::min<size_t>(full.size(), 19));
    };

    // Maximum bytes to scan per cbuffer. Projection/view/world matrices are
    // always in the first few hundred bytes of a cbuffer — capping the scan
    // prevents multi-second stalls on emulators that pack all constants into
    // a single 64KB+ UBO (Xenia, Yuzu, RPCS3, Citra).
    static constexpr size_t kMaxScanBytes = 8192;  // 128 matrices

    // Compute the scannable byte range for a cbuffer binding: the intersection
    // of the bound range (constantOffset..constantOffset+constantCount) with
    // the buffer allocation, capped to kMaxScanBytes from the start of the range.
    auto cbRange = [](const D3D11ConstantBufferBinding& cb) -> std::pair<size_t, size_t> {
      const size_t bufSize = cb.buffer->Desc()->ByteWidth;
      const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
      if (base >= bufSize)
        return { 0, 0 };
      size_t end;
      if (cb.constantCount > 0)
        end = std::min(base + static_cast<size_t>(cb.constantCount) * 16, bufSize);
      else
        end = bufSize;
      if (end - base > kMaxScanBytes)
        end = base + kMaxScanBytes;
      return { base, end };
    };

    // Column-major engines (Unity, Godot) store matrices transposed in memory;
    // transposing after read normalizes them to row-major for all our checks.
    auto readMatrix = [this](const uint8_t* ptr, size_t offset, size_t bufSize) -> Matrix4 {
      Matrix4 m = readCbMatrix(ptr, offset, bufSize);
      return m_columnMajor ? transpose(m) : m;
    };

    // Viewport aspect ratio — used to score projection candidates and reject
    // shadow map / cubemap projections that don't match the screen.
    float viewportAspect = 0.0f;
    {
      const auto& vp = m_context->m_state.rs.viewports[0];
      if (vp.Height > 0.0f)
        viewportAspect = vp.Width / vp.Height;
    }

    // Score a perspective projection: higher = more likely main game camera.
    // Shadow maps have square aspect, cubemaps have 90° FOV, tool cameras
    // have extreme FOV — all score lower than a typical game camera.
    auto scorePerspective = [viewportAspect](const Matrix4& proj) -> float {
      float score = 1.0f;
      DecomposeProjectionParams dpp;
      decomposeProjection(proj, dpp);
      // Guard against degenerate decomposition (NaN/Inf from near-singular matrices).
      if (!std::isfinite(dpp.fov) || !std::isfinite(dpp.aspectRatio) || !std::isfinite(dpp.nearPlane))
        return score;
      float fovDeg = dpp.fov * (180.0f / 3.14159265f);
      if (fovDeg >= 30.0f && fovDeg <= 120.0f)
        score += 2.0f;
      else if (fovDeg >= 15.0f && fovDeg <= 150.0f)
        score += 1.0f;
      if (viewportAspect > 0.0f) {
        float diff = std::abs(std::abs(dpp.aspectRatio) - viewportAspect);
        if (diff < 0.15f)
          score += 2.0f;
        else if (diff < 0.5f)
          score += 1.0f;
      }
      if (dpp.nearPlane > 0.001f && dpp.nearPlane < 100.0f)
        score += 1.0f;
      return score;
    };

    // All shader stages to scan for camera matrices.
    // VS is most common; emulators (Dolphin, PCSX2, Xenia, Citra) and some
    // deferred renderers put camera matrices in GS, DS, or PS cbuffers.
    const D3D11ConstantBufferBindings* stageCbs[] = {
      &m_context->m_state.vs.constantBuffers,
      &m_context->m_state.gs.constantBuffers,
      &m_context->m_state.ds.constantBuffers,
      &m_context->m_state.ps.constantBuffers,
    };
    static constexpr int kNumStages = 4;
    static const char* kStageNames[] = { "VS", "GS", "DS", "PS" };

    // Scan one stage's cbuffers for the best-scoring perspective matrix.
    // classifyPerspective detects both row-major and column-major-as-row
    // layouts in a single pass, so no separate transpose pass is needed.
    auto scanStageForProj = [&](int stageIdx,
        uint32_t& outSlot, size_t& outOff, float& outScore,
        Matrix4& outMat, bool& outColMajor) -> bool
    {
      bool found = false;
      const auto& cbs = *stageCbs[stageIdx];
      for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) continue;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        auto [base, end] = cbRange(cb);
        for (size_t off = base; off + 64 <= end; off += 16) {
          Matrix4 m = readCbMatrix(ptr, off, bufSize);
          // NV-DXVK: Only allow combined VP (cls 3/4) detection once
          // we're confident this is a gameplay frame (50+ draws).
          // Splash/UI frames with 1-20 draws produce false positives
          // from degenerate ortho matrices that happen to have the
          // right m[2][3]/m[3][3] signature.
          const bool allowVP = (m_rawDrawCount > 250);
          int cls = classifyPerspective(m, allowVP);
          if (cls == 0) continue;
          // Column-major-as-row (cls==2 or 4): transpose to row-major for scoring/use.
          const bool isCol = (cls == 2 || cls == 4);
          Matrix4 normalized = isCol ? transpose(m) : m;
          float s;
          if (cls <= 2) {
            // Pure projection: use the existing FOV/aspect scorer.
            s = scorePerspective(normalized);
          } else {
            // Combined VP (cls 3/4): score by how well Sy/Sx matches
            // the viewport aspect ratio.  False positives from game
            // parameter cbuffers have random Sy/Sx ratios; the real VP
            // has Sy/Sx ≈ viewportAspect because Sx = cot(fov/2)/aspect
            // and Sy = cot(fov/2).
            const float r0mag = std::sqrt(normalized[0][0]*normalized[0][0]
                                        + normalized[0][1]*normalized[0][1]
                                        + normalized[0][2]*normalized[0][2]);
            const float r1mag = std::sqrt(normalized[1][0]*normalized[1][0]
                                        + normalized[1][1]*normalized[1][1]
                                        + normalized[1][2]*normalized[1][2]);
            const float vpAspect = (r0mag > 0.001f) ? (r1mag / r0mag) : 0.0f;
            const float diff = std::abs(vpAspect - viewportAspect);
            // Base 1.0 + up to 10.0 for perfect aspect match
            s = 1.0f + 10.0f / (1.0f + diff * 5.0f);
          }
          // NV-DXVK: Debug logging for every VP candidate found during
          // the scan.  Helps track down false positives by showing which
          // slot/offset/cls/score combinations the scanner is evaluating.
          // Gated on a one-shot latch + frame count so we don't spam on
          // every frame after the per-frame re-scan for combined VP.
          {
            static uint32_t s_scanLogCount = 0;
            if (s_scanLogCount < 20) {
              ++s_scanLogCount;
              Logger::info(str::format(
                  "[D3D11Rtx] Projection scan candidate: stage=",
                  kStageNames[stageIdx],
                  " slot=", slot, " off=", off, " cls=", cls,
                  " score=", s,
                  " diag=(", normalized[0][0], ",", normalized[1][1],
                  ",", normalized[2][2], ")",
                  " m23=", normalized[2][3], " m33=", normalized[3][3],
                  " rawDraw=", m_rawDrawCount));
            }
          }
          if (s > outScore) {
            outSlot     = slot;
            outOff      = off;
            outScore    = s;
            outMat      = normalized;
            outColMajor = isCol;
            found       = true;
          }
        }
      }
      return found;
    };

    uint32_t projSlot   = m_projSlot;
    size_t   projOffset = m_projOffset;
    int      projStage  = m_projStage;

    // --- PROJECTION: Source Engine 2 fast-path ---
    // From IDA/shader analysis: CBufCommonPerCamera (cb2) has
    // c_cameraRelativeToClipPrevFrame at offset 96 (always filled).
    // Use offset 96 (prev-frame VP) — offset 16 (current-frame VP) is
    // identity on early draws and can contain degenerate values during
    // loading/transitions that cause assertion failures in SetupByFrustum.
    // --- TF2 deterministic projection: CBufCommonPerCamera at cb2 VS.
    // Layout (from VS RDEF / shader disasm):
    //   offset  0: c_zNear
    //   offset  4: c_cameraOrigin
    //   offset 16: row_major float4x4 c_cameraRelativeToClip    ← CURRENT-FRAME VP
    //   offset 84: c_cameraOriginPrevFrame
    //   offset 96: row_major float4x4 c_cameraRelativeToClipPrevFrame ← PREV VP
    // The active VP for THIS draw is whichever the game wrote into offset 16
    // for that pass (gameplay/shadow/portal/fog/...). Remix classifies the
    // resulting camera downstream. No scoring, no multi-slot scan.
    if (projSlot == UINT32_MAX) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const uint32_t kSourceCamSlot = 2;
      const auto& srcCb = vsCbs[kSourceCamSlot];
      if (srcCb.buffer != nullptr) {
        const auto mapped = srcCb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        const size_t bufSize = srcCb.buffer->Desc()->ByteWidth;
        if (ptr && bufSize >= 160) {
          Matrix4 raw16 = readCbMatrix(ptr, 16, bufSize);
          const int cls16 = classifyPerspective(raw16, true);
          int usedCls = 0;
          if (cls16 > 0) {
            projSlot    = kSourceCamSlot;
            projOffset  = 16;
            projStage   = 0;
            m_projSlot    = kSourceCamSlot;
            m_projOffset  = 16;
            m_projStage   = 0;
            m_columnMajor = (cls16 == 2);
            usedCls = cls16;
          } else {
            Matrix4 raw96 = readCbMatrix(ptr, 96, bufSize);
            const int cls96 = classifyPerspective(raw96, true);
            if (cls96 > 0) {
              projSlot    = kSourceCamSlot;
              projOffset  = 96;
              projStage   = 0;
              m_projSlot    = kSourceCamSlot;
              m_projOffset  = 96;
              m_projStage   = 0;
              m_columnMajor = (cls96 == 2);
              usedCls = cls96;
            }
          }
          static uint32_t sFastLog = 0;
          static uint32_t sFirstPerspLog = 0;
          const bool isPersp = (usedCls > 0);
          // Log first 3 calls (including identity failures) AND the first 3
          // successful perspective picks separately, so we can see exactly
          // when real gameplay VPs start arriving at cb2@16.
          if (sFastLog < 3 || (isPersp && sFirstPerspLog < 3)) {
            if (sFastLog < 3) ++sFastLog;
            if (isPersp) ++sFirstPerspLog;
            Logger::info(str::format(
              "[D3D11Rtx] TF2 deterministic VP: offset=",
              (projSlot == kSourceCamSlot ? (int)projOffset : -1),
              " cls=", usedCls,
              " isPersp=", isPersp ? 1 : 0,
              " diag16=(", raw16[0][0], ",", raw16[1][1], ",", raw16[2][2], ")",
              " m23_16=", raw16[2][3], " m33_16=", raw16[3][3]));
          }
        }
      }
    }

    // --- PROJECTION: first-draw scan (cache miss) ---
    // Single pass across all stages — classifyPerspective handles both layouts.
    if (projSlot == UINT32_MAX) {
      float bestScore = 0.0f;
      Matrix4 bestMat;
      uint32_t bestSlot = UINT32_MAX;
      size_t bestOff = SIZE_MAX;
      int bestStage = -1;
      bool bestCol = false;

      for (int si = 0; si < kNumStages; ++si) {
        uint32_t ts = UINT32_MAX; size_t to = SIZE_MAX;
        float tsc = bestScore; Matrix4 tm; bool tc = false;
        if (scanStageForProj(si, ts, to, tsc, tm, tc) && tsc > bestScore) {
          bestScore = tsc;
          bestSlot = ts; bestOff = to; bestStage = si; bestMat = tm;
          bestCol = tc;
        }
      }

      if (bestSlot != UINT32_MAX) {
        projSlot   = bestSlot;
        projOffset = bestOff;
        projStage  = bestStage;
        m_projSlot   = bestSlot;
        m_projOffset = bestOff;
        m_projStage  = bestStage;
        m_columnMajor = bestCol;
        // NV-DXVK: Track whether the match was a combined VP so EndFrame
        // can invalidate the cache for next frame (combined VP must be
        // re-scanned because it changes with camera movement and is only
        // valid on certain draws within the frame).
        // Check: re-read the matrix and classify to determine if it's VP.
        {
          const auto& cbCheck = (*stageCbs[bestStage])[bestSlot];
          if (cbCheck.buffer != nullptr) {
            const auto mappedCheck = cbCheck.buffer->GetMappedSlice();
            const uint8_t* ptrCheck = reinterpret_cast<const uint8_t*>(mappedCheck.mapPtr);
            if (ptrCheck) {
              Matrix4 mCheck = readCbMatrix(ptrCheck, bestOff, cbCheck.buffer->Desc()->ByteWidth);
              int clsCheck = classifyPerspective(mCheck, true);
              m_projIsCombinedVP = (clsCheck >= 3);
            }
          }
        }
      }
    }

    // --- PROJECTION: validate cached location, re-scan on stale ---
    if (projSlot != UINT32_MAX && projStage >= 0 && projStage < kNumStages) {
      const auto& cbs = *stageCbs[projStage];
      const auto& cb = cbs[projSlot];
      Matrix4 proj;
      bool valid = false;
      if (cb.buffer != nullptr) {
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr) {
          Matrix4 raw = readCbMatrix(ptr, projOffset, cb.buffer->Desc()->ByteWidth);
          // NV-DXVK: Always allow VP in validation — the projection was
          // already found and classified on a previous draw.  Restricting
          // allowVP by rawDrawCount causes the cache to invalidate on
          // every early-frame draw, making all pre-250 draws fall to
          // uiFallback even though the cached projection is still valid.
          const bool allowVP = true;
          int cls = classifyPerspective(raw, allowVP);
          if (cls > 0) {
            proj = (cls == 2 || cls == 4) ? transpose(raw) : raw;
            valid = true;
            // NV-DXVK: For combined View*Proj matrices (cls 3/4), decompose
            // into a clean pure projection + a view matrix extracted from
            // the camera direction/position encoded in the VP rows.
            //
            // For row-major VP = V × P (D3D convention with P having
            // perspSign in m[2][3]):
            //
            //   Row 0 of VP = CamRight × ProjScale  (right dir scaled by Sx)
            //   Row 1 of VP = CamUp    × ProjScale  (up dir scaled by Sy)
            //   Row 2 of VP = CamFwd   × ProjScale  (fwd dir scaled by Q, + perspSign in w)
            //   Row 3 of VP = CamPos   × ProjScale  (position scaled)
            //
            // The forward direction is recoverable as normalize(VP[2][0:2]).
            // The right/up directions are recoverable as normalize(VP[0/1][0:2]).
            // The projection scales Sx, Sy are the magnitudes of those rows.
            //
            // With these we construct:
            //   P = standard perspective from Sx, Sy, conservative near/far
            //   V = rigid-body view matrix from the normalized directions + position
            if (cls == 3 || cls == 4) {
              // NV-DXVK: prefer the fanout-cached VP rows over proj[] when
              // available. The cached projection slot/offset may contain a
              // DIFFERENT VS's cb2 content than the gameplay fanout VS's, so
              // per-draw reads of proj[] can flip the basis by 90° between
              // draws. Fanout rows are captured once per frame from the
              // authoritative gameplay VS, so everyone sees the same pose.
              Vector3 vpRight = m_hasFanoutVpRows ? m_lastFanoutVpRow0
                                                  : Vector3(proj[0][0], proj[0][1], proj[0][2]);
              Vector3 vpUp    = m_hasFanoutVpRows ? m_lastFanoutVpRow1
                                                  : Vector3(proj[1][0], proj[1][1], proj[1][2]);
              Vector3 vpFwd   = m_hasFanoutVpRows ? m_lastFanoutVpRow2
                                                  : Vector3(proj[2][0], proj[2][1], proj[2][2]);

              const float magRight = length(vpRight);
              const float magUp    = length(vpUp);
              const float magFwd   = length(vpFwd);

              // Projection scales from row magnitudes.
              // Sx = |right row|, Sy = |up row|.  For the forward row
              // the magnitude encodes Q (depth scale) which we don't
              // directly need for building P -- we use conservative near/far.
              const float Sx = std::max(magRight, 0.001f);
              const float Sy = std::max(magUp,    0.001f);

              // NV-DXVK PROPER FIX (c_cameraRelativeToClip decomposition):
              // The cb2 matrix rows ALREADY encode the scaled camera basis.
              // Per TF2 VS DXIL disasm (VS_ef94e6c7fcc3c144):
              //   clip.x = dot(cam_rel, c2c.row0.xyz) + c2c.row0.w
              //   clip.y = dot(cam_rel, c2c.row1.xyz) + c2c.row1.w
              //   clip.z = dot(cam_rel, c2c.row2.xyz) + c2c.row2.w
              //   clip.w = dot(cam_rel, c2c.row3.xyz) + c2c.row3.w
              // For a standard P·V factorization:
              //   clip.x = Sx · (R · cam_rel)        → c2c.row0.xyz = Sx · R
              //   clip.y = Sy · (U · cam_rel)        → c2c.row1.xyz = Sy · U
              //   clip.z = a · (F · cam_rel) + b     → c2c.row2.xyz = a · F
              //   clip.w = F · cam_rel               → c2c.row3.xyz = F
              // So the camera's world-space axes are DIRECTLY:
              //   R = normalize(c2c.row0.xyz)
              //   U = normalize(c2c.row1.xyz)
              //   F = normalize(c2c.row3.xyz) (== row2 up to scalar `a`)
              // And the projection scales are the row magnitudes:
              //   Sx = |c2c.row0.xyz|, Sy = |c2c.row1.xyz|, a = |c2c.row2.xyz|
              //
              // The previous code threw row0/row1 away and re-derived R via
              // cross(F, worldUp). That produced a basis oriented for a
              // +Z-up world, which accidentally-worked only when the game's
              // projection happened to agree — and failed hard for TF2's
              // Source-convention X-forward cameras (gun + hands invisible
              // even after all other fixes, because Remix's reconstructed
              // view matrix had forward along +Y instead of +X).
              //
              // Direct extraction keeps the ENTIRE basis encoded in cb2
              // intact: no cross products, no worldUp assumption, no
              // re-derivation. Works for any convention the game happens
              // to use (X-fwd, Z-fwd, Y-up, Z-up, etc.).
              Vector3 fwd   = (magFwd   > 0.001f) ? vpFwd   / magFwd   : Vector3(0, 0, -1);
              Vector3 right = (magRight > 0.001f) ? vpRight / magRight : Vector3(1, 0, 0);
              Vector3 up    = (magUp    > 0.001f) ? vpUp    / magUp    : Vector3(0, 1, 0);
              {
                static uint32_t sBasisLog = 0;
                if (sBasisLog < 3) {
                  ++sBasisLog;
                  Logger::info(str::format(
                    "[D3D11Rtx.path1.basis] #", sBasisLog,
                    " right=(", right.x, ",", right.y, ",", right.z, ")",
                    " up=(", up.x, ",", up.y, ",", up.z, ")",
                    " fwd=(", fwd.x, ",", fwd.y, ",", fwd.z, ")"));
                }
              }
              const float rightLen = length(right);
              if (rightLen > 0.001f) right = right / rightLen;

              // Camera world-space position: read c_cameraOrigin directly
              // from cb2 offset 4 (float3).  This is the current-frame
              // ground truth.  Previously we read from projOffset-16
              // (offset 80) which contained c_frameNum (garbage) +
              // c_cameraOriginPrevFrame (1 frame behind).  The heuristic
              // to skip c_frameNum was fragile and always 1 frame stale.
              //
              // CBufCommonPerCamera layout:
              //   offset  0: c_zNear        (float,  4 bytes)
              //   offset  4: c_cameraOrigin (float3, 12 bytes) ← THIS
              //   offset 16: c_cameraRelativeToClip (float4x4, 64 bytes)
              //   offset 80: c_frameNum     (int,    4 bytes)
              //   offset 84: c_cameraOriginPrevFrame (float3, 12 bytes)
              //   offset 96: c_cameraRelativeToClipPrevFrame (float4x4)
              const float perspSign = (proj[2][3] < 0.0f) ? -1.0f : 1.0f;
              float Tx = 0.0f, Ty = 0.0f, Tz = 0.0f;
              {
                bool gotCamPos = false;
                char sourceP1 = '-';
                // NV-DXVK: read c_cameraOrigin FRESH from cb2 each draw via
                // RDEF. The previous code preferred m_lastFanoutCamOrigin,
                // but fanout capture only fires for specific draw types
                // (BSP instance fanout with main viewport) and publishes
                // ONCE early in gameplay. Subsequent camera movement was
                // invisible to path 1 because it returned the stale cached
                // value. Diagnosed via direct cb2 raw-byte dump: raw cb2
                // byte 4-15 tracks the player's movement correctly, but
                // m_lastFanoutCamOrigin stays at spawn pose for the whole
                // session. Always re-read cb2 first; fanout is the
                // fallback for shaders without RDEF CBufCommonPerCamera.
                bool isViewModelPass = false;
                if (m_context->m_state.rs.numViewports > 0) {
                  isViewModelPass = m_context->m_state.rs.viewports[0].MaxDepth <= 0.08f;
                }
                const auto vsPtrP1 = m_context->m_state.vs.shader;
                // NV-DXVK [pilot-eye-capture]: when this draw IS a viewmodel
                // pass, read cb2.c_cameraOrigin via the same RDEF mechanism
                // path 1 uses for non-viewmodel draws and publish it to the
                // file-scope pilot-eye atomics. CameraManager snaps Main's
                // worldToView translation to this so primary rays come from
                // Source's actual eye position (titan cockpit / rodeo /
                // pilot-on-foot — all routed through the viewmodel pass cb2).
                // Path 1's NORMAL viewmodel guard (!isViewModelPass below)
                // is preserved so we don't disturb its existing logic; this
                // is a pure-add capture, no behavior change to anything else.
                if (isViewModelPass
                    && vsPtrP1 != nullptr && vsPtrP1->GetCommonShader() != nullptr) {
                  const auto* commonVm = vsPtrP1->GetCommonShader();
                  auto camLocVm = commonVm->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
                  if (camLocVm && camLocVm->size >= 12
                      && camLocVm->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                    const auto& vsCbsVm = m_context->m_state.vs.constantBuffers;
                    const auto& camCbVm = vsCbsVm[camLocVm->slot];
                    if (camCbVm.buffer != nullptr) {
                      const auto mapVm = camCbVm.buffer->GetMappedSlice();
                      const uint8_t* pVm = reinterpret_cast<const uint8_t*>(mapVm.mapPtr);
                      const size_t baseVm =
                        static_cast<size_t>(camCbVm.constantOffset) * 16 + camLocVm->offset;
                      if (pVm && baseVm + 12 <= camCbVm.buffer->Desc()->ByteWidth) {
                        const float* fpVm = reinterpret_cast<const float*>(pVm + baseVm);
                        if (std::isfinite(fpVm[0]) && std::isfinite(fpVm[1]) && std::isfinite(fpVm[2])) {
                          dxvk::tf2::g_pilotEyeX.store(fpVm[0], std::memory_order_relaxed);
                          dxvk::tf2::g_pilotEyeY.store(fpVm[1], std::memory_order_relaxed);
                          dxvk::tf2::g_pilotEyeZ.store(fpVm[2], std::memory_order_relaxed);
                          dxvk::tf2::g_pilotEyeValid.store(true, std::memory_order_relaxed);
                          // Throttled diag — first 60 unique-XYZ writes.
                          {
                            static uint32_t sVmEyeWriteN = 0;
                            static float sLastVmX = 1e30f, sLastVmY = 1e30f, sLastVmZ = 1e30f;
                            const bool changed =
                                 std::abs(fpVm[0] - sLastVmX) > 0.01f
                              || std::abs(fpVm[1] - sLastVmY) > 0.01f
                              || std::abs(fpVm[2] - sLastVmZ) > 0.01f;
                            if (changed && sVmEyeWriteN < 60) {
                              sLastVmX = fpVm[0]; sLastVmY = fpVm[1]; sLastVmZ = fpVm[2];
                              ++sVmEyeWriteN;
                              const auto& vps = m_context->m_state.rs.viewports;
                              Logger::info(str::format(
                                "[viewmodelCamWrite] #", sVmEyeWriteN,
                                " cam=(", fpVm[0], ",", fpVm[1], ",", fpVm[2], ")",
                                " maxD=", vps[0].MaxDepth,
                                " vp=", uint32_t(vps[0].Width), "x", uint32_t(vps[0].Height)));
                            }
                          }
                        }
                      }
                    }
                  }
                }
                if (!isViewModelPass
                    && vsPtrP1 != nullptr && vsPtrP1->GetCommonShader() != nullptr) {
                  const auto* commonP1 = vsPtrP1->GetCommonShader();
                  auto camLocP1 = commonP1->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
                  if (camLocP1 && camLocP1->size >= 12
                      && camLocP1->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                    const auto& vsCbsP1 = m_context->m_state.vs.constantBuffers;
                    const auto& camCbP1 = vsCbsP1[camLocP1->slot];
                    if (camCbP1.buffer != nullptr) {
                      const auto mapP1 = camCbP1.buffer->GetMappedSlice();
                      const uint8_t* pP1 = reinterpret_cast<const uint8_t*>(mapP1.mapPtr);
                      const size_t baseP1 =
                        static_cast<size_t>(camCbP1.constantOffset) * 16 + camLocP1->offset;
                      if (pP1 && baseP1 + 12 <= camCbP1.buffer->Desc()->ByteWidth) {
                        const float* fp = reinterpret_cast<const float*>(pP1 + baseP1);
                        if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                          // Sanity-gate: cb2's c_cameraOrigin must match the
                          // fanout cache's X/Y within 5 units. Otherwise this
                          // is a non-main camera (mirror/portal/etc.) that
                          // happened to slip past the vpMaxDepth filter; use
                          // fanout instead so worldToView stays canonical.
                          const bool xyMatchesFanout =
                            !m_hasFanoutCamOrigin
                            || (std::abs(fp[0] - m_lastFanoutCamOrigin.x) < 5.0f
                                && std::abs(fp[1] - m_lastFanoutCamOrigin.y) < 5.0f);
                          if (xyMatchesFanout) {
                            Tx = fp[0]; Ty = fp[1]; Tz = fp[2];
                            gotCamPos = true;
                            sourceP1 = 'R';
                          }
                        }
                      }
                    }
                  }
                }
                // Fallback: old hardcoded cb (the one we decomposed VP from).
                // Left in place for shaders that don't expose CBufCommonPerCamera.
                if (!gotCamPos) {
                  const size_t cbBase = static_cast<size_t>(cb.constantOffset) * 16;
                  const size_t bufSize = cb.buffer->Desc()->ByteWidth;
                  if (cbBase + 16 <= bufSize) {
                    const float* cam4 = reinterpret_cast<const float*>(ptr + cbBase + 4);
                    if (std::isfinite(cam4[0]) && std::isfinite(cam4[1]) && std::isfinite(cam4[2])
                        && (std::abs(cam4[0]) > 1.0f || std::abs(cam4[1]) > 1.0f || std::abs(cam4[2]) > 1.0f)) {
                      Tx = cam4[0]; Ty = cam4[1]; Tz = cam4[2];
                      gotCamPos = true;
                      sourceP1 = 'H';
                    }
                  }
                  if (!gotCamPos && cbBase + 96 <= bufSize) {
                    const float* cam84 = reinterpret_cast<const float*>(ptr + cbBase + 84);
                    if (std::isfinite(cam84[0]) && std::isfinite(cam84[1]) && std::isfinite(cam84[2])) {
                      Tx = cam84[0]; Ty = cam84[1]; Tz = cam84[2];
                      gotCamPos = true;
                      sourceP1 = 'H';
                    }
                  }
                }
                // Last-resort fallback: fanout-cached origin. Should never
                // be needed in normal gameplay (RDEF + hardcoded cb2@4 both
                // work reliably), but preserve for shaders that don't bind
                // CBufCommonPerCamera at all.
                if (!gotCamPos && m_hasFanoutCamOrigin) {
                  Tx = m_lastFanoutCamOrigin.x;
                  Ty = m_lastFanoutCamOrigin.y;
                  Tz = m_lastFanoutCamOrigin.z;
                  gotCamPos = true;
                  sourceP1 = 'F';
                }
                // Log which source path 1 used (first ~50 draws or changes).
                {
                  static uint32_t sP1Log = 0;
                  static char sLastSource = '?';
                  static Vector3 sLastValue(-1e9f, -1e9f, -1e9f);
                  const bool changed = sourceP1 != sLastSource
                    || std::abs(sLastValue.x - Tx) > 0.5f
                    || std::abs(sLastValue.y - Ty) > 0.5f
                    || std::abs(sLastValue.z - Tz) > 0.5f;
                  if (changed && sP1Log < 30) {
                    ++sP1Log;
                    sLastSource = sourceP1;
                    sLastValue = Vector3(Tx, Ty, Tz);
                    Logger::info(str::format(
                      "[D3D11Rtx.path1Cam] #", sP1Log,
                      " src=", sourceP1,
                      " cam=(", Tx, ",", Ty, ",", Tz, ")"));
                  }
                }
              }

              // Build the D3D row-major view matrix:
              //   V = [Rx  Ry  Rz  0]
              //       [Ux  Uy  Uz  0]
              //       [Fx  Fy  Fz  0]
              //       [Tx' Ty' Tz' 1]
              //
              // where T' = -dot(dir, pos) for each axis (the "eye-space translation").
              //
              // NV-DXVK EXPERIMENT: TF2 renders camera-relative — vertex buffers
              // hold (world - cameraOrigin) and t31 (objectToCameraRelative)
              // transforms place geometry relative to camera. Our TLAS therefore
              // sits in camera-at-origin space. If we encode c_cameraOrigin into
              // the view matrix, Remix's RtCamera::position = cameraOrigin in
              // world, but our TLAS entries are at small camera-relative coords —
              // rays fire from the wrong origin and miss everything. Force the
              // view translation to zero so Remix's camera sits at origin,
              // matching the TLAS frame. Only the viewmodel/particles (already
              // at identity in view space) were rendering before; with this,
              // world geometry should also be hit.
              // NV-DXVK: world-space Main camera (NOT camera-relative). Previously
              // this was true (camera at origin + all geo in camera-relative frame),
              // but NRC's spatial cache needs STABLE world coords — camera-relative
              // makes every position shift per-frame, invalidating NRC. Motion
              // vectors / denoisers also need real world-space camera motion.
              // With false, Main gets its actual world translation and BSP's
              // translate(cameraOrigin) o2w fallback produces matching world coords.
              constexpr bool kCameraAtOrigin = false;
              const float Tx_use = kCameraAtOrigin ? 0.0f : Tx;
              const float Ty_use = kCameraAtOrigin ? 0.0f : Ty;
              const float Tz_use = kCameraAtOrigin ? 0.0f : Tz;
              // NV-DXVK PART 2b: REMOVED fwdSign negation. Previously this
              // code negated `fwd` when cb2's perspSign<0 so that the
              // rebuilt (V,P) pair produced a POSITIVE clip.w despite both
              // halves individually flipping sign. The cancellation worked
              // mathematically but left Remix's RtCamera with an inverted
              // forward axis — col[2] of worldToView pointed to -F world
              // instead of +F. RtCamera's ray generator then fired primary
              // rays in the OPPOSITE of cb2's gameplay forward direction,
              // so any geometry (gun, hands, anything) that cb2 placed in
              // +F world was never hit by rays.
              //
              // Proper fix: use `fwd` unchanged. This makes col[2] = true
              // world forward. Pair this with perspSign = +1 in the
              // rebuilt projection so clip.w = +view.z (standard D3D LH
              // convention: in-front verts have positive clip.w). See the
              // `proj = Matrix4(...)` construction further below.
              const Vector3 fwdV = fwd;
              const float dotR = -(right.x * Tx_use + right.y * Ty_use + right.z * Tz_use);
              const float dotU = -(up.x    * Tx_use + up.y    * Ty_use + up.z    * Tz_use);
              const float dotF = -(fwdV.x  * Tx_use + fwdV.y  * Ty_use + fwdV.z  * Tz_use);

              // NV-DXVK: construct the Matrix4 from COLUMNS.
              // dxvk's Matrix4 stores data[i] as column i, and its multiply
              // operator treats data[i] as column i. For a proper view matrix
              // where row 0 = right, row 1 = up, row 2 = fwd (so V*P produces
              // view-space coords via row-i · (P,1)), we must pass the
              // COLUMNS of that matrix to the constructor:
              //   column 0 = (right.x, up.x, fwd.x, 0)
              //   column 1 = (right.y, up.y, fwd.y, 0)
              //   column 2 = (right.z, up.z, fwd.z, 0)
              //   column 3 = (dotR, dotU, dotF, 1)
              // Previous code passed ROWS (right, up, fwd, translation) as
              // args, producing V^T instead of V. With camera at origin this
              // was invisible (V^T = V for translation-free rotations around
              // trivial axes), but with a real camera position inverse(V^T)[3]
              // != cameraPos, which is the bug the log shows.
              m_lastWtvPathId = 1; // path 1: generic VP-decomposition
              transforms.worldToView = Matrix4(
                Vector4(right.x, up.x, fwdV.x, 0.0f),
                Vector4(right.y, up.y, fwdV.y, 0.0f),
                Vector4(right.z, up.z, fwdV.z, 0.0f),
                Vector4(dotR,    dotU,  dotF,   1.0f));
              {
                static uint32_t sW2vLog = 0;
                if (sW2vLog < 3) {
                  ++sW2vLog;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path1.w2v] #", sW2vLog,
                    " dotR=", dotR, " dotU=", dotU, " dotF=", dotF,
                    " w2v[3][0..2]=(", w[3][0], ",", w[3][1], ",", w[3][2], ")",
                    " cam=(", Tx, ",", Ty, ",", Tz, ")"));
                }
              }

              // Build a clean pure perspective projection from the extracted scales.
              const float nearZ = 1.0f;
              const float farZ  = 20000.0f;
              const float Q     = farZ / (farZ - nearZ);
              // NV-DXVK PART 2b: use perspSign = +1 unconditionally. This
              // is standard D3D LH convention: clip.w = +view.z so in-front
              // vertices (view.z > 0 because V uses un-negated fwd) get
              // clip.w > 0 and survive rasterizer clipping. Remix's
              // RtCamera and viewToProjection-dependent downstream code
              // both assume clip.w > 0 for visible vertices, so we want
              // the REBUILT (V,P) to satisfy that unconditionally — not
              // to inherit whatever handedness cb2 happened to use.
              // Previous code passed perspSign (= -1 for TF2's RH cb2)
              // here, which required fwdSign = -1 elsewhere to cancel;
              // that cancellation hid the bug that Remix's fwd axis was
              // inverted. With fwdSign removed AND perspSign pinned to
              // +1, the view matrix has a true-forward col[2] and the
              // projection maps view.z → clip.w sanely, end-to-end.
              proj = Matrix4(
                Vector4(Sx,   0.0f, 0.0f,          0.0f),
                Vector4(0.0f, Sy,   0.0f,          0.0f),
                Vector4(0.0f, 0.0f, Q,             1.0f),
                Vector4(0.0f, 0.0f, -nearZ * Q,    0.0f));

              // Log decompositions periodically (every 100th) so we can
              // verify the camera position/direction tracks player movement
              // across frames without flooding the log.
              static uint32_t s_vpDecompLogCount = 0;
              ++s_vpDecompLogCount;
              if (s_vpDecompLogCount <= 3 || (s_vpDecompLogCount % 100) == 0) {
                // DIAG: dump the cb2 buffer pointer, mapped pointer, and
                // first 16 floats of the VP region. If camera is stuck,
                // we can tell if it's the buffer itself (same ptr same
                // data = game not writing), the mapping (same ptr different
                // data wouldn't happen), or a different buffer each frame
                // (rotating allocations, ptrs differ, new draws target a
                // buffer we're not reading).
                const auto& cbDiag = cbs[projSlot];
                uintptr_t bufAddr = reinterpret_cast<uintptr_t>(cbDiag.buffer.ptr());
                uintptr_t mapAddr = 0;
                float raw16Floats[16] = {0};
                if (cbDiag.buffer != nullptr) {
                  const auto mapped = cbDiag.buffer->GetMappedSlice();
                  mapAddr = reinterpret_cast<uintptr_t>(mapped.mapPtr);
                  const uint8_t* pDiag = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  const size_t baseDiag = static_cast<size_t>(cbDiag.constantOffset) * 16;
                  if (pDiag && baseDiag + 64 <= cbDiag.buffer->Desc()->ByteWidth) {
                    std::memcpy(raw16Floats, pDiag + baseDiag, 64);
                  }
                }
                Logger::info(str::format(
                    "[D3D11Rtx] Decomposed combined VP (cls=", cls,
                    "): Sx=", Sx, " Sy=", Sy,
                    " fwd=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
                    " pos=(", Tx, ",", Ty, ",", Tz, ")",
                    " perspSign=", perspSign,
                    " bufAddr=", bufAddr,
                    " mapAddr=", mapAddr,
                    " projOff=", projOffset,
                    " raw@4=(", raw16Floats[1], ",", raw16Floats[2], ",", raw16Floats[3], ")"));
              }
            }
          }
        }
      }

      if (!valid && projSlot == m_projSlot && projStage == m_projStage) {
        // Cached location is stale (different pass). Re-scan all stages.
        projSlot = UINT32_MAX;
        float bestScore = 0.0f;
        for (int si = 0; si < kNumStages; ++si) {
          uint32_t ts = UINT32_MAX; size_t to = SIZE_MAX;
          float tsc = bestScore; Matrix4 tm; bool tc = false;
          if (scanStageForProj(si, ts, to, tsc, tm, tc)) {
            projSlot = ts; projOffset = to; projStage = si;
            proj = tm; bestScore = tsc;
          }
        }
      }

      if (projSlot != UINT32_MAX) {
        // Strip TAA jitter — Remix does its own TAA.
        proj[2][0] = 0.0f;
        proj[2][1] = 0.0f;

        // --- AXIS AUTO-DETECTION (projection-derived) ---
        // Vote on Y-flip and LH/RH from each valid projection matrix.
        // Votes accumulate until a threshold is reached, then the setting
        // is permanently locked — no re-evaluation.  This guarantees that
        // objectToWorld transforms (and therefore geometry/spatial hashes)
        // see a consistent coordinate system for the entire session.
        {
          const bool canVote = !m_yFlipSettled || !m_lhSettled;

          if (canVote) {
            m_axisDetected = true;

            // Y-flip: negative Y scale in projection
            m_yFlipVotes += (proj[1][1] < 0.0f) ? 1 : -1;
            if (!m_yFlipSettled && std::abs(m_yFlipVotes) >= kVoteThreshold) {
              m_yFlipSettled = true;
              const bool yFlip = m_yFlipVotes > 0;
              RtCamera::correctProjectionYFlipObject().setDeferred(yFlip);
            }

            // LH/RH from projection decomposition
            DecomposeProjectionParams dpp;
            decomposeProjection(proj, dpp);
            if (std::isfinite(dpp.fov) && std::isfinite(dpp.aspectRatio)) {
              m_lhVotes += dpp.isLHS ? 1 : -1;
              if (!m_lhSettled && std::abs(m_lhVotes) >= kVoteThreshold) {
                m_lhSettled = true;
                const bool isLH = m_lhVotes > 0;
                RtxOptions::leftHandedCoordinateSystemObject().setDeferred(isLH);
              }
            }
          }

        }

        transforms.viewToProjection = proj;

        // NV-DXVK [pure-projection cls 1/2 worldToView reconstruction]:
        // For cls 1/2, the V/P decomposition above (cls 3/4 only) didn't
        // run — `transforms.worldToView` is still the default 4x4 identity
        // and the saveW2vValid guard below would reject the save, leaving
        // the cache stuck on whatever cinematic/idle camera last managed
        // to be classified as cls 3/4. Result: scene renders, but locked
        // to that stale camera; player movement isn't reflected.
        //
        // We already have everything we need to build a real worldToView
        // for these draws: fanout-published VP rows (rotation basis) and
        // m_lastFanoutCamOrigin (translation). Same construction path 3
        // uses around line ~3543. Only run when we have BOTH, otherwise
        // leave w2v as identity and let the existing guard skip the save.
        if (m_hasFanoutVpRows && m_hasFanoutCamOrigin) {
          const auto& cur = transforms.worldToView;
          const bool wIsIdentity =
               std::abs(cur[0][0] - 1.0f) + std::abs(cur[1][1] - 1.0f)
             + std::abs(cur[2][2] - 1.0f)
             + std::abs(cur[0][1]) + std::abs(cur[0][2])
             + std::abs(cur[1][0]) + std::abs(cur[1][2])
             + std::abs(cur[2][0]) + std::abs(cur[2][1])
             + std::abs(cur[3][0]) + std::abs(cur[3][1]) + std::abs(cur[3][2])
             < 0.01f;
          if (wIsIdentity) {
            // NV-DXVK [viewmodel-fix]: prefer LIVE cb2 read for THIS draw's
            // VS over the fanout cache.
            //
            // Symptom this addresses: viewmodel "gun moves back, camera
            // moves, gun wants to point to center" — i.e., the gun lags or
            // chases the camera basis. Root cause: cls 1/2 draws (which
            // include viewmodel) were using the fanout cache, last published
            // during the BSP-world sub-pass. By the time the viewmodel sub-
            // pass fires later in the frame, the engine has rebound cb2
            // with the viewmodel's own VP, but cls12Recon was still using
            // the world's snapshot — so viewmodel rays go through the
            // world-camera basis instead of its own.
            //
            // Fix: re-read CBufCommonPerCamera live from the currently-bound
            // VS at THIS draw. If that fails (no RDEF / unbound), fall back
            // to the fanout cache so we don't regress.
            //
            // Layout (verified): c_cameraOrigin at FindCBField-resolved
            // offset; c_cameraRelativeToClip is row-major float4x4 starting
            // 12 bytes after c_cameraOrigin (tight packing inside the same
            // 16-byte slot for c_cameraOrigin's float3, then matrix on the
            // next 16-byte boundary at +16 of cb if zNear+camOrigin live at
            // 0..15).
            Vector3 vpRight = m_lastFanoutVpRow0;
            Vector3 vpUp    = m_lastFanoutVpRow1;
            Vector3 vpFwd   = m_lastFanoutVpRow2;
            Vector3 liveCam(m_lastFanoutCamOrigin.x,
                            m_lastFanoutCamOrigin.y,
                            m_lastFanoutCamOrigin.z);
            bool usedLive = false;
            {
              const auto vsPtrL = m_context->m_state.vs.shader;
              if (vsPtrL != nullptr && vsPtrL->GetCommonShader() != nullptr) {
                const auto* commonL = vsPtrL->GetCommonShader();
                auto camLocL = commonL->FindCBField(
                  "CBufCommonPerCamera", "c_cameraOrigin");
                if (camLocL && camLocL->size >= 12
                    && camLocL->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                  const auto& vsCbsL = m_context->m_state.vs.constantBuffers;
                  const auto& cbL = vsCbsL[camLocL->slot];
                  if (cbL.buffer != nullptr) {
                    const auto mapL = cbL.buffer->GetMappedSlice();
                    const uint8_t* pL = reinterpret_cast<const uint8_t*>(mapL.mapPtr);
                    const size_t bszL = cbL.buffer->Desc()->ByteWidth;
                    const size_t baseCam =
                      static_cast<size_t>(cbL.constantOffset) * 16 + camLocL->offset;
                    // VP matrix at +12 from c_cameraOrigin (next 16B slot).
                    const size_t baseVP = baseCam + 12;
                    if (pL && baseCam + 12 <= bszL && baseVP + 64 <= bszL) {
                      const float* fpC = reinterpret_cast<const float*>(pL + baseCam);
                      const float* vp  = reinterpret_cast<const float*>(pL + baseVP);
                      bool finite = true;
                      for (int k = 0; k < 12 && finite; ++k)
                        if (!std::isfinite(vp[k])) finite = false;
                      if (finite && std::isfinite(fpC[0]) && std::isfinite(fpC[1])
                          && std::isfinite(fpC[2])) {
                        Vector3 r0(vp[0], vp[1], vp[2]);
                        Vector3 r1(vp[4], vp[5], vp[6]);
                        Vector3 r2(vp[8], vp[9], vp[10]);
                        const float l0 = length(r0), l1 = length(r1), l2 = length(r2);
                        // Reject identity / degenerate (matches fanout gate).
                        const bool isId =
                             std::abs(r0.x - 1.0f) < 1e-4f
                          && std::abs(r1.y - 1.0f) < 1e-4f
                          && std::abs(r2.z - 1.0f) < 1e-4f
                          && std::abs(r0.y) < 1e-4f && std::abs(r0.z) < 1e-4f;
                        const bool camAtZero =
                             std::abs(fpC[0]) < 1e-3f
                          && std::abs(fpC[1]) < 1e-3f
                          && std::abs(fpC[2]) < 1e-3f;
                        // ALWAYS log raw cb-read values (regardless of
                        // accept/reject) so we can see what the viewmodel
                        // pass actually puts in cb2 when it hits cls12Recon.
                        // Throttle by VS hash so we get one entry per shader.
                        {
                          std::string vsKeyR = "?";
                          {
                            auto vsP = m_context->m_state.vs.shader;
                            if (vsP != nullptr && vsP->GetCommonShader() != nullptr) {
                              auto& sh = vsP->GetCommonShader()->GetShader();
                              if (sh != nullptr) vsKeyR = sh->getShaderKey().toString().substr(0, 19);
                            }
                          }
                          static std::unordered_map<std::string, uint32_t> sRawLog;
                          auto& cnt = sRawLog[vsKeyR];
                          if (cnt < 2) {
                            ++cnt;
                            const char* reject =
                              (isId ? "ID" :
                              (camAtZero ? "CAMZERO" :
                              ((l0 > 0.1f && l1 > 0.1f && l2 > 0.001f) ? "ACCEPT" : "DEGEN")));
                            Logger::info(str::format(
                              "[cls12RawCB] vs=", vsKeyR,
                              " gate=", reject,
                              " rawCam=(", fpC[0], ",", fpC[1], ",", fpC[2], ")",
                              " rawR0=(", r0.x, ",", r0.y, ",", r0.z, ")",
                              " rawR1=(", r1.x, ",", r1.y, ",", r1.z, ")",
                              " rawR2=(", r2.x, ",", r2.y, ",", r2.z, ")"));
                          }
                        }
                        if (!isId && !camAtZero
                            && l0 > 0.1f && l1 > 0.1f && l2 > 0.001f) {
                          vpRight = r0; vpUp = r1; vpFwd = r2;
                          liveCam = Vector3(fpC[0], fpC[1], fpC[2]);
                          usedLive = true;
                        }
                      }
                    }
                  }
                }
              }
              {
                // Resolve the bound VS hash for the diag — lets us see
                // which shader is producing which path so we can verify
                // (e.g. usedLive=0 + cam=(0,0,0) should be a viewmodel-
                // family VS, usedLive=1 a world-family VS).
                std::string vsKeyL = "?";
                {
                  auto vsP = m_context->m_state.vs.shader;
                  if (vsP != nullptr && vsP->GetCommonShader() != nullptr) {
                    auto& sh = vsP->GetCommonShader()->GetShader();
                    if (sh != nullptr) vsKeyL = sh->getShaderKey().toString().substr(0, 19);
                  }
                }
                static std::unordered_map<std::string, uint32_t> sLiveLogPerVs;
                auto& counter = sLiveLogPerVs[vsKeyL + (usedLive ? "+" : "-")];
                if (counter < 3) {
                  ++counter;
                  Logger::info(str::format(
                    "[cls12ReconLive] vs=", vsKeyL,
                    " usedLive=", usedLive ? 1 : 0,
                    " liveCam=(", liveCam.x, ",", liveCam.y, ",", liveCam.z, ")",
                    " liveR2=(", vpFwd.x, ",", vpFwd.y, ",", vpFwd.z, ")",
                    " fanoutCam=(", m_lastFanoutCamOrigin.x, ",",
                                    m_lastFanoutCamOrigin.y, ",",
                                    m_lastFanoutCamOrigin.z, ")"));
                }
              }
            }
            const float magR = length(vpRight);
            const float magU = length(vpUp);
            const float magF = length(vpFwd);
            // Same validity check as path 3.
            if (magR > 0.1f && magU > 0.1f && magF > 0.001f
                && std::abs(magR - magU) > 0.01f) {
              // Source RH (X=fwd, Y=left, Z=up): right = fwd × worldUp.
              Vector3 fwd = vpFwd / magF;
              Vector3 right = cross(fwd, Vector3(0.0f, 0.0f, 1.0f));
              float rl = length(right);
              right = (rl > 0.001f) ? right / rl : Vector3(0.0f, -1.0f, 0.0f);
              Vector3 up = cross(right, fwd);
              float ul = length(up);
              up = (ul > 0.001f) ? up / ul : Vector3(0.0f, 0.0f, 1.0f);
              const float camX = liveCam.x;
              const float camY = liveCam.y;
              const float camZ = liveCam.z;
              const float tR = -(right.x*camX + right.y*camY + right.z*camZ);
              const float tU = -(up.x*camX    + up.y*camY    + up.z*camZ);
              const float tF = -(fwd.x*camX   + fwd.y*camY   + fwd.z*camZ);
              transforms.worldToView = Matrix4(
                Vector4(right.x, up.x, fwd.x, 0.0f),
                Vector4(right.y, up.y, fwd.y, 0.0f),
                Vector4(right.z, up.z, fwd.z, 0.0f),
                Vector4(tR,      tU,   tF,   1.0f));
              m_lastWtvPathId = 1; // path 1, but cls 1/2 reconstruction branch
              static uint32_t sCls12Recon = 0;
              static float sLastCx = 0, sLastCy = 0, sLastCz = 0;
              const float dx = camX - sLastCx, dy = camY - sLastCy, dz = camZ - sLastCz;
              // Lower threshold so fine player movements are visible at 1 fps.
              // 0.01 unit = sub-millimeter in TF2 units; only suppresses repeated
              // identical reads, shows every actual movement.
              if (sCls12Recon == 0 || (dx*dx + dy*dy + dz*dz) > 0.0001f) {
                ++sCls12Recon;
                sLastCx = camX; sLastCy = camY; sLastCz = camZ;
                Logger::info(str::format(
                  "[cls12Recon] path1 (cls 1/2 implicit) @", m_rawDrawCount,
                  " usedLive=", usedLive ? 1 : 0,
                  " cam=(", camX, ",", camY, ",", camZ, ")",
                  " R=(", right.x, ",", right.y, ",", right.z, ")",
                  " F=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
                  " recons=", sCls12Recon));
              }
            }
          }
        }

        // NV-DXVK: Mark this frame as "has real projection" so subsequent
        // draws that hit the fallback path can reuse these transforms
        // instead of being filtered as UIFallback.
        //
        // CRITICAL GUARD: only commit transforms to the shared cache if
        // worldToView has real translation. For pure-projection cases
        // (cls 1/2) path 1's inner VP-decomposition block doesn't run,
        // so transforms.worldToView stays at default-identity. Saving
        // that identity would clobber a previously-real cached w2v —
        // then deferred BSP draws reading the cache see identity and
        // get rejected as degenerate_cached_w2v. This was the bug
        // causing all gameplay BSP VSes to be filtered even with the
        // mutex fix and static sharing in place.
        const auto& saveW = transforms.worldToView;
        // NV-DXVK: previously tested only translation (saveW[3]). TF2 (and
        // other Source-engine games) renders camera-relative — vertices are
        // pre-translated by -cameraOrigin so worldToView's translation
        // column is legitimately (0,0,0). The old guard rejected every
        // real player draw, leaving only odd cinematic / view-model draws
        // in the cache.
        // The case the guard NEEDS to catch is when worldToView is left as
        // the default 4x4 identity (rotation = identity AND translation =
        // zero). So check both: a real camera with zero translation still
        // has non-identity rotation columns.
        const float trMag2 = saveW[3][0]*saveW[3][0]
                           + saveW[3][1]*saveW[3][1]
                           + saveW[3][2]*saveW[3][2];
        const float rotDiagDiff =
             std::abs(saveW[0][0] - 1.0f)
           + std::abs(saveW[1][1] - 1.0f)
           + std::abs(saveW[2][2] - 1.0f);
        const float rotOffDiag =
             std::abs(saveW[0][1]) + std::abs(saveW[0][2])
           + std::abs(saveW[1][0]) + std::abs(saveW[1][2])
           + std::abs(saveW[2][0]) + std::abs(saveW[2][1]);
        const bool rotIsIdentity = (rotDiagDiff + rotOffDiag) < 0.01f;
        const bool trIsZero      = trMag2 < 1e-4f;
        const bool saveW2vValid  = !(rotIsIdentity && trIsZero);
        // NV-DXVK [player-cam filter]: derive the camera origin encoded in
        // this worldToView and compare to the fanout-published player cam.
        // worldToView is column-major: cols 0..2 hold rotation as rows
        // (right.{x,y,z}, up.{x,y,z}, fwd.{x,y,z}), col 3 = -V_rot * camPos.
        // So camPos = -V_rot^T * t. Only save when the encoded cam matches
        // the player cam within tolerance — keeps probes/shadows/viewmodel
        // out of the cache so consume reads always return the player view.
        // Falls open (accepts anything) when fanout hasn't published yet
        // (boot frames) so the cache can still warm up.
        bool saveIsPlayerCam = true;
        if (m_hasFanoutCamOrigin) {
          const float tR = saveW[3][0];
          const float tU = saveW[3][1];
          const float tF = saveW[3][2];
          const float camX = -(saveW[0][0]*tR + saveW[0][1]*tU + saveW[0][2]*tF);
          const float camY = -(saveW[1][0]*tR + saveW[1][1]*tU + saveW[1][2]*tF);
          const float camZ = -(saveW[2][0]*tR + saveW[2][1]*tU + saveW[2][2]*tF);
          const float dxc = camX - m_lastFanoutCamOrigin.x;
          const float dyc = camY - m_lastFanoutCamOrigin.y;
          const float dzc = camZ - m_lastFanoutCamOrigin.z;
          // NV-DXVK [main-vs-viewmodel filter — REAPPLIED]: 5-unit tolerance.
          //
          // Pairs with the viewmodel-reject gate at the fanout publish site
          // (search this file for [viewmodel reject — REAPPLIED]). Together
          // they restrict m_lastGoodTransforms saves to only the actual main
          // pass — viewmodel saves alternately writing to the cache caused
          // the ray tracer's per-draw worldToView consume to flicker between
          // two camera positions, which manifested as "the camera lowers to
          // the floor" / "I only see feet" depending on which side of the
          // race won during the visible frame.
          //
          // 5 units is well above floating-point jitter (sub-cm in
          // hammer-unit scale) and well below the ~18 unit viewmodel
          // offset, so it cleanly admits only the actual main pass.
          // Real probes/shadows are thousands of units off — already
          // safely rejected by the broader probe/cubemap filters above.
          const float kCamMatchTol2 = 5.0f * 5.0f;
          const bool camMatches = (dxc*dxc + dyc*dyc + dzc*dzc) <= kCamMatchTol2;
          // VERIFIED from saveMatrix dumps: normal player saves have
          // det = -1 in this matrix convention (left-handed-as-stored).
          // Mirror passes (water/reflection) have det = +1 — exactly
          // row 2 of the matrix negated (F → -F, tF → -tF).
          // Cross-product expansion: det = R · (U × F) where
          //   R = (saveW[0][0], saveW[1][0], saveW[2][0])
          //   U = (saveW[0][1], saveW[1][1], saveW[2][1])
          //   F = (saveW[0][2], saveW[1][2], saveW[2][2])
          const float Rx = saveW[0][0], Ry = saveW[1][0], Rz = saveW[2][0];
          const float Ux = saveW[0][1], Uy = saveW[1][1], Uz = saveW[2][1];
          const float Fx = saveW[0][2], Fy = saveW[1][2], Fz = saveW[2][2];
          const float det3 =
              Rx * (Uy * Fz - Uz * Fy)
            - Ry * (Ux * Fz - Uz * Fx)
            + Rz * (Ux * Fy - Uy * Fx);
          // Reject saves with det > 0 (mirror). Slight tolerance band
          // around 0 to avoid floating-point edge cases.
          const bool isNormalHandedness = det3 < -0.5f;
          saveIsPlayerCam = camMatches && isNormalHandedness;
          // Diag: log full matrix on every accepted save so we can see
          // exactly how mirror passes encode their basis. Throttled by
          // change in the upper-left 3x3 (16 floats logged every time the
          // rotation actually differs from the previous save). Proves out
          // the actual layout before we apply any "is mirror?" heuristic.
          if (saveIsPlayerCam) {
            static float sLast[16] = {};
            const float cur[16] = {
              saveW[0][0], saveW[0][1], saveW[0][2], saveW[0][3],
              saveW[1][0], saveW[1][1], saveW[1][2], saveW[1][3],
              saveW[2][0], saveW[2][1], saveW[2][2], saveW[2][3],
              saveW[3][0], saveW[3][1], saveW[3][2], saveW[3][3]
            };
            float diff = 0.0f;
            for (int k = 0; k < 12; ++k)  // skip translation (last 3 entries already logged)
              diff += std::abs(cur[k] - sLast[k]);
            if (diff > 0.01f) {
              std::memcpy(sLast, cur, sizeof(sLast));
              Logger::info(str::format(
                "[saveMatrix] @", m_rawDrawCount,
                " col0=(", cur[0], ",", cur[1], ",", cur[2], ",", cur[3], ")",
                " col1=(", cur[4], ",", cur[5], ",", cur[6], ",", cur[7], ")",
                " col2=(", cur[8], ",", cur[9], ",", cur[10], ",", cur[11], ")",
                " col3=(", cur[12], ",", cur[13], ",", cur[14], ",", cur[15], ")"));
            }
          }
        }
        if (saveW2vValid && saveIsPlayerCam) {
          std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
          m_foundRealProjThisFrame = true;
          m_hasEverFoundProj = true;
          m_lastGoodTransforms = transforms;
        } else if (saveW2vValid && !saveIsPlayerCam) {
          // Real perspective draw, but cam is NOT the player — log so we
          // can see what we're filtering out (probes, shadows, viewmodels
          // with displaced cams, etc.). Throttled by reconstructed cam.
          static uint32_t sNonPlayerLog = 0;
          static float sLastNpX = 0, sLastNpY = 0, sLastNpZ = 0;
          const float tR = saveW[3][0], tU = saveW[3][1], tF = saveW[3][2];
          const float camX = -(saveW[0][0]*tR + saveW[0][1]*tU + saveW[0][2]*tF);
          const float camY = -(saveW[1][0]*tR + saveW[1][1]*tU + saveW[1][2]*tF);
          const float camZ = -(saveW[2][0]*tR + saveW[2][1]*tU + saveW[2][2]*tF);
          const float dx = camX - sLastNpX, dy = camY - sLastNpY, dz = camZ - sLastNpZ;
          if (sNonPlayerLog == 0 || (dx*dx + dy*dy + dz*dz) > 100.0f) {
            ++sNonPlayerLog;
            sLastNpX = camX; sLastNpY = camY; sLastNpZ = camZ;
            Logger::info(str::format(
              "[cachedSaveSkipNonPlayer] @", m_rawDrawCount,
              " saveCam=(", camX, ",", camY, ",", camZ, ")",
              " playerCam=(", m_lastFanoutCamOrigin.x, ",",
                              m_lastFanoutCamOrigin.y, ",",
                              m_lastFanoutCamOrigin.z, ")",
              " skips=", sNonPlayerLog));
          }
          // Keep marking projection-found state so consume can still
          // happen with whatever last-good (player) cache exists.
          m_foundRealProjThisFrame = true;
          m_hasEverFoundProj = true;
        } else {
          // Still mark projection found (viewToProjection IS real),
          // but don't stomp the cache with identity w2v.
          m_foundRealProjThisFrame = true;
          m_hasEverFoundProj = true;
          // Log when a real perspective draw is REJECTED for w2v-too-small.
          // If real player camera draws are landing here, we're filtering
          // them out and only the cinematic / scripted cam survives. Log
          // each unique draw index that hits this path (rate-limited per
          // index so we don't spam during a stable identity-cam stretch).
          {
            static std::unordered_map<uint32_t, uint32_t> sRejectCounts;
            uint32_t& n = sRejectCounts[m_rawDrawCount];
            if ((n % 5) == 0) {
              Logger::info(str::format(
                "[cachedRejectIdentity] path1 @", m_rawDrawCount,
                " w2v=(", saveW[3][0], ",", saveW[3][1], ",", saveW[3][2], ")",
                " hits=", n + 1,
                " (real perspective draw, but w2v translation < 0.01 — not saved)"));
            }
            ++n;
          }
        }
        {
          // Log on change. Threshold 0.01 unit (squared 0.0001) so we see
          // sub-cm movements at 1 fps debug build — needed to confirm slow
          // creep / actual movement vs frozen camera.
          static uint32_t sSaveLog = 0;
          static float    sLastX   = 0.0f;
          static float    sLastY   = 0.0f;
          static float    sLastZ   = 0.0f;
          const auto& w = m_lastGoodTransforms.worldToView;
          const float dx = w[3][0] - sLastX;
          const float dy = w[3][1] - sLastY;
          const float dz = w[3][2] - sLastZ;
          const bool moved = (dx*dx + dy*dy + dz*dz) > 0.0001f;
          if (sSaveLog == 0 || moved) {
            ++sSaveLog;
            sLastX = w[3][0]; sLastY = w[3][1]; sLastZ = w[3][2];
            // Column-major: the basis vectors are formed by reading element
            // i across columns 0..2.  right = (w[0][0], w[1][0], w[2][0]),
            // fwd = (w[0][2], w[1][2], w[2][2]).  Wild F.z swings between
            // saves = camera pitching down/up between draws.
            // Reconstruct world-space camera position so the log line is
            // directly comparable to playerCam in [cachedSaveSkipNonPlayer]:
            // camPos = -V_rot^T * t, same formula as the saveIsPlayerCam
            // filter above.
            const float reconCamX =
              -(w[0][0] * w[3][0] + w[0][1] * w[3][1] + w[0][2] * w[3][2]);
            const float reconCamY =
              -(w[1][0] * w[3][0] + w[1][1] * w[3][1] + w[1][2] * w[3][2]);
            const float reconCamZ =
              -(w[2][0] * w[3][0] + w[2][1] * w[3][1] + w[2][2] * w[3][2]);
            const float fanX = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.x : 0.f;
            const float fanY = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.y : 0.f;
            const float fanZ = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.z : 0.f;
            const float camDelta = std::sqrt(
              (reconCamX - fanX) * (reconCamX - fanX) +
              (reconCamY - fanY) * (reconCamY - fanY) +
              (reconCamZ - fanZ) * (reconCamZ - fanZ));
            Logger::info(str::format(
              "[cachedSave] path1 @", m_rawDrawCount,
              " w2v=(", w[3][0], ",", w[3][1], ",", w[3][2], ")",
              " camPos=(", reconCamX, ",", reconCamY, ",", reconCamZ, ")",
              " playerCam=(", fanX, ",", fanY, ",", fanZ, ")",
              " camDelta=", camDelta,
              " R=(", w[0][0], ",", w[1][0], ",", w[2][0], ")",
              " F=(", w[0][2], ",", w[1][2], ",", w[2][2], ")",
              " saves=", sSaveLog));
          }
        }
      }
    }

    // --- FALLBACK PROJECTION ---
    // If no perspective matrix was found in any cbuffer, synthesize one from
    // the viewport.  This gives Remix a valid camera so geometry renders at
    // roughly correct positions even when: (a) the engine packs matrices in
    // a format we don't recognize, (b) the game uses compute-based rendering,
    // or (c) all cbuffers are GPU-only / unmappable.  The fallback is
    // intentionally conservative (60° FOV, 0.1–10000 range) and is only used
    // when the real scan comes up empty.
    //
    // NV-DXVK: For Source-engine games (Titanfall 2, etc.), the main-menu
    // and HUD draws legitimately have NO perspective projection — they use
    // orthographic UI projections.  When that happens we flag the extract
    // as "used fallback" so SubmitDraw can drop the draw out of the RTX
    // pipeline (it still rasterizes natively via the EmitCs path in
    // D3D11DeviceContext::Draw*), and EndFrame's camera safety net will
    // skip firing for the frame.  That leaves injectRTX() with an invalid
    // camera, which early-returns (rtx_context.cpp:492), so the native
    // raster content in the backbuffer passes through unchanged instead of
    // being overwritten by a path-traced empty scene compressed into a
    // viewport-fallback corner.
    // NV-DXVK: Source Engine 2 last-resort — if no projection was found
    // by the generic scanner, try reading cb2@96 directly as a combined VP.
    // This is c_cameraRelativeToClipPrevFrame which is always populated
    // even on early draws where c_cameraRelativeToClip (cb2@16) is identity.
    if (projSlot == UINT32_MAX) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const auto& srcCb = vsCbs[2];
      if (srcCb.buffer != nullptr) {
        const auto mapped = srcCb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr && srcCb.buffer->Desc()->ByteWidth >= 160) {
          Matrix4 raw = readCbMatrix(ptr, 96, srcCb.buffer->Desc()->ByteWidth);
          // Check if it looks like a perspective matrix (m[2][3] == ±1)
          if (std::abs(std::abs(raw[2][3]) - 1.0f) < 0.1f) {
            projSlot = 2;
            projOffset = 96;
            projStage = 0;

            // Decompose the combined VP into projection + view
            Vector3 vpRight(raw[0][0], raw[0][1], raw[0][2]);
            Vector3 vpUp   (raw[1][0], raw[1][1], raw[1][2]);
            Vector3 vpFwd  (raw[2][0], raw[2][1], raw[2][2]);
            const float Sx = std::max(length(vpRight), 0.001f);
            const float Sy = std::max(length(vpUp),    0.001f);
            const float magFwd = length(vpFwd);
            Vector3 fwd = magFwd > 0.001f ? vpFwd / magFwd : Vector3(0, 0, -1);
            // NV-DXVK: Source RH (X=fwd, Y=left, Z=up) — right = fwd × worldUp.
            const Vector3 worldUpLS(0.0f, 0.0f, 1.0f);
            Vector3 right = cross(fwd, worldUpLS);
            float rightLen = length(right);
            if (rightLen > 0.001f) right = right / rightLen;
            else right = Vector3(0.0f, -1.0f, 0.0f);
            Vector3 up = cross(right, fwd);
            float upLen = length(up);
            if (upLen > 0.001f) up = up / upLen;
            else up = worldUpLS;

            // Camera position: try offset 4 (current frame), fall back to
            // offset 84 (prev frame) if current is zero (early draws).
            // NV-DXVK: respect cb.constantOffset — see path 1 fix comment.
            float Tx = 0, Ty = 0, Tz = 0;
            {
              const size_t cbBase = static_cast<size_t>(srcCb.constantOffset) * 16;
              const size_t bsz = srcCb.buffer->Desc()->ByteWidth;
              bool got = false;
              if (cbBase + 16 <= bsz) {
                const float* c4 = reinterpret_cast<const float*>(ptr + cbBase + 4);
                if (std::isfinite(c4[0]) && std::isfinite(c4[1]) && std::isfinite(c4[2])
                    && (std::abs(c4[0]) > 1.0f || std::abs(c4[1]) > 1.0f || std::abs(c4[2]) > 1.0f)) {
                  Tx = c4[0]; Ty = c4[1]; Tz = c4[2]; got = true;
                }
              }
              if (!got && cbBase + 96 <= bsz) {
                const float* c84 = reinterpret_cast<const float*>(ptr + cbBase + 84);
                if (std::isfinite(c84[0]) && std::isfinite(c84[1]) && std::isfinite(c84[2])) {
                  Tx = c84[0]; Ty = c84[1]; Tz = c84[2];
                }
              }
            }
            const float dotR = -(right.x*Tx + right.y*Ty + right.z*Tz);
            const float dotU = -(up.x*Tx    + up.y*Ty    + up.z*Tz);
            const float dotF = -(fwd.x*Tx   + fwd.y*Ty   + fwd.z*Tz);
            // NV-DXVK: store by columns — see path 1 fix.
            m_lastWtvPathId = 2; // path 2: TF2 cb2@96 last-resort VP-decomp
            // Apply perspSign to fwd in view matrix (same fix as path 1).
            const float perspSign2 = raw[2][3] < 0 ? -1.0f : 1.0f;
            const float fwdSign2 = (perspSign2 < 0.0f) ? -1.0f : 1.0f;
            const Vector3 fwdV2(fwdSign2 * fwd.x, fwdSign2 * fwd.y, fwdSign2 * fwd.z);
            const float dotF2 = -(fwdV2.x*Tx + fwdV2.y*Ty + fwdV2.z*Tz);
            transforms.worldToView = Matrix4(
              Vector4(right.x, up.x, fwdV2.x, 0),
              Vector4(right.y, up.y, fwdV2.y, 0),
              Vector4(right.z, up.z, fwdV2.z, 0),
              Vector4(dotR,    dotU, dotF2,   1));

            const float nearZ = 1.0f, farZ = 20000.0f;
            const float Q = farZ / (farZ - nearZ);
            // NV-DXVK: D3D-style Q, matches path 3.
            transforms.viewToProjection = Matrix4(
              Vector4(Sx,   0, 0,          0),
              Vector4(0,    Sy, 0,         0),
              Vector4(0,    0, Q,          perspSign2),
              Vector4(0,    0, -nearZ*Q,   0));

            // Only commit to cached if this path produced a real w2v.
            // cb2@96 is c_cameraRelativeToClipPrevFrame (marked [unused]
            // in every VS — the game may never write it). When it's zero
            // or junk, passes-sniff-test data can produce a w2v with
            // ~zero translation that corrupts m_lastGoodTransforms,
            // causing downstream "degenerate cached w2v" rejections to
            // filter every BSP draw as UIFallback.
            const bool path2W2vValid =
                 std::abs(dotR)  > 0.01f
              || std::abs(dotU)  > 0.01f
              || std::abs(dotF2) > 0.01f;
            if (path2W2vValid) {
              // Same player-cam filter as path1 — only save the player view.
              // Also require up ≈ +Z (rejects water-reflection / mirror passes
              // whose camera is reflected through the water plane).
              bool path2IsPlayerCam = true;
              const auto& sw = transforms.worldToView;
              if (m_hasFanoutCamOrigin) {
                const float tR = sw[3][0], tU = sw[3][1], tF = sw[3][2];
                const float camX = -(sw[0][0]*tR + sw[0][1]*tU + sw[0][2]*tF);
                const float camY = -(sw[1][0]*tR + sw[1][1]*tU + sw[1][2]*tF);
                const float camZ = -(sw[2][0]*tR + sw[2][1]*tU + sw[2][2]*tF);
                const float dxc = camX - m_lastFanoutCamOrigin.x;
                const float dyc = camY - m_lastFanoutCamOrigin.y;
                const float dzc = camZ - m_lastFanoutCamOrigin.z;
                const bool camMatches = (dxc*dxc + dyc*dyc + dzc*dzc) <= 200.0f*200.0f;
                // det < 0 = normal player handedness, det > 0 = mirror.
                const float Rx = sw[0][0], Ry = sw[1][0], Rz = sw[2][0];
                const float Ux = sw[0][1], Uy = sw[1][1], Uz = sw[2][1];
                const float Fx = sw[0][2], Fy = sw[1][2], Fz = sw[2][2];
                const float det3 =
                    Rx * (Uy * Fz - Uz * Fy)
                  - Ry * (Ux * Fz - Uz * Fx)
                  + Rz * (Ux * Fy - Uy * Fx);
                const bool isNormalHandedness = det3 < -0.5f;
                path2IsPlayerCam = camMatches && isNormalHandedness;
              }
              if (path2IsPlayerCam) {
                {
                  std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
                  m_foundRealProjThisFrame = true;
                  m_lastGoodTransforms = transforms;
                }
                {
                  static uint32_t sSave2Log = 0;
                  if (sSave2Log < 20) {
                    ++sSave2Log;
                    const auto& w = m_lastGoodTransforms.worldToView;
                    Logger::info(str::format(
                      "[cachedSave] path2 @", m_rawDrawCount,
                      " w2v=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                  }
                }
                // Full-matrix dump for path2 saves so we can see what the
                // mirror pass's basis actually looks like (path1 logger
                // missed it because path1 wasn't writing those entries).
                {
                  static float sLast2[16] = {};
                  const auto& w = m_lastGoodTransforms.worldToView;
                  const float cur[16] = {
                    w[0][0], w[0][1], w[0][2], w[0][3],
                    w[1][0], w[1][1], w[1][2], w[1][3],
                    w[2][0], w[2][1], w[2][2], w[2][3],
                    w[3][0], w[3][1], w[3][2], w[3][3]
                  };
                  float diff = 0.0f;
                  for (int k = 0; k < 12; ++k)
                    diff += std::abs(cur[k] - sLast2[k]);
                  if (diff > 0.01f) {
                    std::memcpy(sLast2, cur, sizeof(sLast2));
                    Logger::info(str::format(
                      "[saveMatrix2] @", m_rawDrawCount,
                      " col0=(", cur[0], ",", cur[1], ",", cur[2], ",", cur[3], ")",
                      " col1=(", cur[4], ",", cur[5], ",", cur[6], ",", cur[7], ")",
                      " col2=(", cur[8], ",", cur[9], ",", cur[10], ",", cur[11], ")",
                      " col3=(", cur[12], ",", cur[13], ",", cur[14], ",", cur[15], ")"));
                  }
                }
              }
            }
          }
        }
      }
    }

    // NV-DXVK: If no projection found but we've found one in a prior frame,
    // reuse the cached camera — BUT only for R32G32_UINT position draws
    // (main world geometry).  fmt=106 draws that fail projection detection
    // are shadow/depth passes with light-space transforms → applying the
    // main camera VP to them produces extreme BLAS → GPU TDR.
    if (projSlot == UINT32_MAX && m_hasEverFoundProj) {
      // Check if this draw uses R32G32_UINT position format
      bool isUintPosLayout = false;
      D3D11InputLayout* il = m_context->m_state.ia.inputLayout.ptr();
      if (il) {
        for (const auto& s : il->GetRtxSemantics()) {
          if (std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0
              && s.format == VK_FORMAT_R32G32_UINT) {
            isUintPosLayout = true;
            break;
          }
        }
      }
      if (isUintPosLayout) {
        transforms.viewToProjection = m_lastGoodTransforms.viewToProjection;
        // Use the cached worldToView from the last VP decomposition.
        // This contains the camera rotation (from VP rows) and the camera
        // position (from cb2@4).  objectToView = worldToView * objectToWorld
        // is computed at line ~1787.  With FusedWorldViewMode::View, Remix
        // then sets objectToWorld = objectToView (fusing the transforms) and
        // zeros worldToView, so geometry ends up in view space centred on
        // the camera.
        //
        // For GPU bone draws: objectToWorld = identity (interleaver applies
        // bones GPU-side → world-space output).  objectToView = worldToView.
        // After fuse: objectToWorld = worldToView, camera at origin.
        // Geometry goes world → view via the instance transform. Correct.
        // Bone matrices output CAMERA-RELATIVE positions (world pos minus
        // camera origin is baked into the bone matrix by the engine).
        // Set worldToView from the cached VP + c_cameraOrigin.
        // The view matrix scan will run later but finds nothing for early
        // draws. Setting it here ensures a valid camera for R32G32_UINT draws.
        {
          float camX = 0, camY = 0, camZ = 0;
          // NV-DXVK: read fresh from cb2 each draw (same fix as path 1).
          // m_lastFanoutCamOrigin is stale — caches spawn pose forever.
          bool gotCamP3 = false;
          char sourceP3 = '-';
          {
            const auto vsPtrP3 = m_context->m_state.vs.shader;
            if (vsPtrP3 != nullptr && vsPtrP3->GetCommonShader() != nullptr) {
              const auto* commonP3 = vsPtrP3->GetCommonShader();
              auto camLocP3 = commonP3->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
              if (camLocP3 && camLocP3->size >= 12
                  && camLocP3->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                const auto& vsCbsP3 = m_context->m_state.vs.constantBuffers;
                const auto& camCbP3 = vsCbsP3[camLocP3->slot];
                if (camCbP3.buffer != nullptr) {
                  const auto mapP3 = camCbP3.buffer->GetMappedSlice();
                  const uint8_t* pP3 = reinterpret_cast<const uint8_t*>(mapP3.mapPtr);
                  const size_t baseP3 =
                    static_cast<size_t>(camCbP3.constantOffset) * 16 + camLocP3->offset;
                  if (pP3 && baseP3 + 12 <= camCbP3.buffer->Desc()->ByteWidth) {
                    const float* fp = reinterpret_cast<const float*>(pP3 + baseP3);
                    if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                      camX = fp[0]; camY = fp[1]; camZ = fp[2];
                      gotCamP3 = true;
                      sourceP3 = 'R';
                    }
                  }
                }
              }
            }
          }
          const auto& camCb = m_context->m_state.vs.constantBuffers[2];
          if (!gotCamP3 && camCb.buffer != nullptr) {
            const auto mapped = camCb.buffer->GetMappedSlice();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            // Fallback: hardcoded cb2@4 with constantOffset.
            const size_t camCbBase = static_cast<size_t>(camCb.constantOffset) * 16;
            const size_t camBufSz  = camCb.buffer->Desc()->ByteWidth;
            if (p && camCbBase + 16 <= camBufSz) {
              const float* co = reinterpret_cast<const float*>(p + camCbBase + 4);
              camX = co[0]; camY = co[1]; camZ = co[2];
              gotCamP3 = true;
              sourceP3 = 'H';
            }
          }
          // Last-resort fallback: fanout cache.
          if (!gotCamP3 && m_hasFanoutCamOrigin) {
            camX = m_lastFanoutCamOrigin.x;
            camY = m_lastFanoutCamOrigin.y;
            camZ = m_lastFanoutCamOrigin.z;
            gotCamP3 = true;
            sourceP3 = 'F';
          }
          // Log which source path 3 used (capped, only on change).
          {
            static uint32_t sP3Log = 0;
            static char sLastSource = '?';
            static Vector3 sLastValue(-1e9f, -1e9f, -1e9f);
            const bool changed = sourceP3 != sLastSource
              || std::abs(sLastValue.x - camX) > 0.5f
              || std::abs(sLastValue.y - camY) > 0.5f
              || std::abs(sLastValue.z - camZ) > 0.5f;
            if (changed && sP3Log < 30) {
              ++sP3Log;
              sLastSource = sourceP3;
              sLastValue = Vector3(camX, camY, camZ);
              Logger::info(str::format(
                "[D3D11Rtx.path3Cam] #", sP3Log,
                " src=", sourceP3,
                " cam=(", camX, ",", camY, ",", camZ, ")"));
            }
          }
          // Read VP rotation. Prefer the cached fanout VP rows (captured at
          // the same moment as m_lastFanoutCamOrigin) so every path-3 draw
          // gets the SAME gameplay pose regardless of which VS's cb2 is
          // bound for this specific draw. Only fall back to per-draw cb2@96
          // when fanout hasn't published yet (very early boot frames).
          Vector3 right(0, -1, 0), up(0, 0, 1), fwd(1, 0, 0);  // defaults
          bool gotLiveRotation = false;
          bool usedFanoutVp = false;
          if (m_hasFanoutVpRows) {
            const Vector3& vpRight = m_lastFanoutVpRow0;
            const Vector3& vpUp    = m_lastFanoutVpRow1;
            const Vector3& vpFwd   = m_lastFanoutVpRow2;
            float magR = length(vpRight), magU = length(vpUp), magF = length(vpFwd);
            if (magR > 0.1f && magU > 0.1f && magF > 0.001f &&
                std::abs(magR - magU) > 0.01f) {
              // Source RH (X=fwd, Y=left, Z=up) — right = fwd × worldUp.
              fwd   = vpFwd   / magF;
              const Vector3 worldUpP3(0.0f, 0.0f, 1.0f);
              right = cross(fwd, worldUpP3);
              float rightLenP3 = length(right);
              if (rightLenP3 > 0.001f) right = right / rightLenP3;
              else right = Vector3(0.0f, -1.0f, 0.0f);
              up = cross(right, fwd);
              float upLenP3 = length(up);
              if (upLenP3 > 0.001f) up = up / upLenP3;
              else up = worldUpP3;
              gotLiveRotation = true;
              usedFanoutVp = true;
            }
          }
          const auto& camCb2 = m_context->m_state.vs.constantBuffers[2];
          if (!gotLiveRotation && camCb2.buffer != nullptr) {
            const auto camMapped2 = camCb2.buffer->GetMappedSlice();
            const uint8_t* camPtr = reinterpret_cast<const uint8_t*>(camMapped2.mapPtr);
            // DIAGNOSED FROM SHADER DECOMPILE (VS_d69c3951f050e757):
            // CBufCommonPerCamera.c_cameraRelativeToClip is at byte offset
            // 16 (current frame). The previous code read offset 96 which
            // is c_cameraRelativeToClipPrevFrame, marked [unused] in every
            // TF2 VS — the game never writes it, so its contents are zeros
            // or stale. That produced garbage right/up/fwd extraction.
            //
            // M = P * V_rot (column convention; shader does `clip = M*v`):
            //   row 0: (Sx*Rx, Sx*Ry, Sx*Rz, 0)
            //   row 1: (Sy*Ux, Sy*Uy, Sy*Uz, 0)
            //   row 2: (Q*Fx,  Q*Fy,  Q*Fz,  -nearZ*Q)
            // Each row's xyz is R/U/F scaled by a SINGLE scalar (Sx, Sy,
            // Q). Normalizing xyz recovers the unit basis vectors.
            if (camPtr && camCb2.buffer->Desc()->ByteWidth >= 80) {
              const float* vp = reinterpret_cast<const float*>(camPtr + 16);
              Vector3 vpRight(vp[0], vp[1], vp[2]);
              Vector3 vpUp   (vp[4], vp[5], vp[6]);
              Vector3 vpFwd  (vp[8], vp[9], vp[10]);
              float magR = length(vpRight), magU = length(vpUp), magF = length(vpFwd);
              // Check if VP is valid (not identity — identity has mag ≈ 1 for all rows
              // and diagonal-dominant structure, while a real VP has different scales)
              if (magR > 0.1f && magU > 0.1f && magF > 0.001f &&
                  std::abs(magR - magU) > 0.01f) {  // real VP has different Sx vs Sy
                fwd   = vpFwd   / magF;
                // Source RH (X=fwd, Y=left, Z=up) — right = fwd × worldUp.
                const Vector3 worldUpP3cb(0.0f, 0.0f, 1.0f);
                right = cross(fwd, worldUpP3cb);
                float rightLenP3cb = length(right);
                if (rightLenP3cb > 0.001f) right = right / rightLenP3cb;
                else right = Vector3(0.0f, -1.0f, 0.0f);
                up = cross(right, fwd);
                float upLenP3cb = length(up);
                if (upLenP3cb > 0.001f) up = up / upLenP3cb;
                else up = worldUpP3cb;
                gotLiveRotation = true;
              }
            }
          }
          // Log rotation once per frame (not per draw) to track mouse look
          static uint32_t sRotLogFrame = UINT32_MAX;
          static uint32_t sRotLogCount = 0;
          if (m_rawDrawCount < 15 && sRotLogCount < 200) {
            // Only log on first draw of each frame
            uint32_t frameApprox = sRotLogCount; // approximate
            ++sRotLogCount;
            Logger::info(str::format(
              "[D3D11Rtx] ViewRot: live=", gotLiveRotation ? 1 : 0,
              " R=(", right.x, ",", right.y, ",", right.z, ")",
              " U=(", up.x, ",", up.y, ",", up.z, ")",
              " F=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
              " cam=(", camX, ",", camY, ",", camZ, ")",
              " w2vT=(", transforms.worldToView[3][0], ",", transforms.worldToView[3][1], ",", transforms.worldToView[3][2], ")",
              " raw=", m_rawDrawCount));
          }
          // Fallback: use cached rotation from VP decomposition
          if (!gotLiveRotation) {
            const Matrix4& cachedView = m_lastGoodTransforms.worldToView;
            Vector3 cRight(cachedView[0][0], cachedView[0][1], cachedView[0][2]);
            Vector3 cUp   (cachedView[1][0], cachedView[1][1], cachedView[1][2]);
            Vector3 cFwd  (cachedView[2][0], cachedView[2][1], cachedView[2][2]);
            if (length(cRight) > 0.5f && length(cFwd) > 0.5f) {
              right = cRight; up = cUp; fwd = cFwd;
            }
          }
          // cb3 = objectToCameraRelative = objectToWorld × viewRotation.
          // We need Remix's camera to know the rotation so rays track the
          // camera direction. Set worldToView = liveRotation (from VP),
          // then objectToWorld = inverse(liveRotation) × cb3 to undo the
          // double rotation. No translation in worldToView (cb3 has it).
          //
          // Use the live VP rotation from cb2@96 as worldToView.
          // DON'T apply Y-flip — the VP rotation must match what cb3 was
          // built with so inverse(rotation) × cb3 cancels correctly.
          // The axis swap is handled by the VP rotation itself (it maps
          // Source axes to the projection's expected space).
          // Validate rotation vectors are finite
          bool rotValid = std::isfinite(right.x) && std::isfinite(right.y) && std::isfinite(right.z)
                       && std::isfinite(up.x) && std::isfinite(up.y) && std::isfinite(up.z)
                       && std::isfinite(fwd.x) && std::isfinite(fwd.y) && std::isfinite(fwd.z)
                       && length(right) > 0.5f && length(fwd) > 0.5f;
          if (rotValid) {
            // Full view matrix with rotation AND camera translation
            float tR = -(right.x*camX + right.y*camY + right.z*camZ);
            float tU = -(up.x*camX    + up.y*camY    + up.z*camZ);
            float tF = -(fwd.x*camX   + fwd.y*camY   + fwd.z*camZ);
            m_lastWtvPathId = 3; // path 3: bone-fanout primary, raw VP rotation + cb2@4 cam
            // NV-DXVK: store by columns — see path 1 fix.
            transforms.worldToView = Matrix4(
              Vector4(right.x, up.x, fwd.x, 0),
              Vector4(right.y, up.y, fwd.y, 0),
              Vector4(right.z, up.z, fwd.z, 0),
              Vector4(tR,      tU,   tF,   1));
          } else {
            // Fallback: fixed axis swap (identity-like; rows=cols for this case)
            m_lastWtvPathId = 4; // path 4: bone-fanout fallback, hardcoded axis swap
            transforms.worldToView = Matrix4(
              Vector4( 0,  0,  1, 0),
              Vector4(-1,  0,  0, 0),
              Vector4( 0,  1,  0, 0),
              Vector4( 0,  0,  0, 1));
          }
          // Skip the VIEW matrix scan only — worldToView is set.
          // Allow WORLD matrix scan to extract cb3 per-draw transforms.
          m_skipViewMatrixScan = true;
        }

        // NV-DXVK (non-instanced BSP t31 path): TF2 BSP shaders (verified
        // via DXBC disasm of VS_597b7e49…) do:
        //   clip = cb2.c_cameraRelativeToClip × (t31[v1.x].objectToCameraRelative × local + 1)
        // where t31 is g_modelInst (StructuredBuffer<ModelInstance>, stride
        // 208) and v1.x is COLOR1 (R16G16B16A16_UINT per-instance, first
        // uint16). The shader does NOT use a t30 bone matrix — that code
        // path was a wrong guess.
        //
        // For non-instanced draws (instanceCount=1), the fanout code above
        // doesn't run; we fetch t31[charIdx].objectToCameraRelative here
        // and use it as objectToWorld (plus +cameraOrigin on the translation
        // column to shift from camera-relative into absolute world, matching
        // the fanout path).
        bool gotBoneTransform = false;
        {
          // Heavy diagnostic logging: tag every step so we can see exactly
          // where/why the t31 path succeeds or fails.
          const char* t31SkipReason = nullptr;
          ID3D11ShaderResourceView* modelInstSrv = nullptr;
          uint32_t modelInstSlot = UINT32_MAX;
          bool rdefFound = false;
          {
            auto vsPtrT31 = m_context->m_state.vs.shader;
            if (vsPtrT31 != nullptr && vsPtrT31->GetCommonShader() != nullptr) {
              modelInstSlot = vsPtrT31->GetCommonShader()->FindResourceSlot("g_modelInst");
              if (modelInstSlot != UINT32_MAX) rdefFound = true;
            }
          }
          // NV-DXVK principled routing: the t31 path reads g_modelInst[idx]
          // where idx comes from a per-instance COLOR1/I:R16G16B16A16_UINT
          // semantic declared by the VS. That semantic is the authoritative
          // signal the shader is genuinely instanced against t31 — when it's
          // absent AND RDEF didn't name g_modelInst, the VS is a static mesh
          // and its transform lives in cb3.CBufModelInstance (PIX-confirmed
          // for VS_6e3e6f28). Previously we fell back to slot 31 blindly,
          // reading t31[0] from an unrelated buffer and corrupting cb3's
          // correct matrix on later frames.
          //
          // Semantic-based gate (no hardcoded hashes):
          //   - hasInstanceIdx: VS declares per-instance R16G16B16A16_UINT
          //     (COLOR1/I per the Source 2 convention) → real t31 indexing
          //   - rdefFound: shader self-declared g_modelInst → real t31
          // If neither, skip t31 path and let cb3/identity handle it.
          bool hasInstanceIdxSemantic = false;
          if (il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                hasInstanceIdxSemantic = true;
                break;
              }
            }
          }
          const bool t31PathEligible = rdefFound || hasInstanceIdxSemantic;
          // Also check if this VS has a cb3 CBufModelInstance — if so, the
          // downstream RDEF cb3 path will own the transform and we must NOT
          // let the "no bone transform → fallback" flag at line ~2806 fire.
          bool cb3OwnsTransform = false;
          {
            auto vsPtrCb3 = m_context->m_state.vs.shader;
            if (vsPtrCb3 != nullptr && vsPtrCb3->GetCommonShader() != nullptr) {
              auto cbInfo = vsPtrCb3->GetCommonShader()->FindCBuffer("CBufModelInstance");
              if (cbInfo && cbInfo->bindSlot != UINT32_MAX) cb3OwnsTransform = true;
            }
          }
          if (t31PathEligible && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            modelInstSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
          } else if (!t31PathEligible) {
            // Skip the t31 fetch. If cb3 will provide the transform, pre-claim
            // gotBoneTransform so the end-of-block "no bone matrix" check
            // (line ~2806) doesn't set m_lastExtractUsedFallback=true and
            // cause SubmitDraw to filter this as UIFallback. The cb3 RDEF
            // path further down will write the real objectToWorld.
            if (cb3OwnsTransform) {
              gotBoneTransform = true;
            }
            static std::unordered_set<std::string> sT31SkipLogged;
            const std::string vkeyMiss = getVsHashShort();
            if (sT31SkipLogged.insert(vkeyMiss).second) {
              Logger::info(str::format(
                "[D3D11Rtx.o2w.t31.skip] vs=", vkeyMiss,
                " reason=no_rdef_g_modelInst_and_no_perinstance_uint4_idx",
                " cb3OwnsTransform=", cb3OwnsTransform ? 1 : 0,
                " (cb3 CBufModelInstance RDEF path owns this draw)"));
            }
          }
          // NV-DXVK: skinned-character discrimination. TF2 has TWO shader
          // families that both bind g_modelInst on t31:
          //
          //   1. BSP / batched props: POSITION0/V + COLOR1/I. t31 index is a
          //      per-INSTANCE COLOR1 value; one matrix per draw instance
          //      applies to the whole mesh. Our t31 fix is correct here.
          //   2. Skinned characters: POSITION0/V + BLENDWEIGHT0/V:fmt82 +
          //      BLENDINDICES0/V:fmt41. t31 is used as a BONE PALETTE — the
          //      VS reads t31[blendIdx[i]] PER-VERTEX and weighted-sums.
          //      Applying t31[0] to the whole character collapses every
          //      vertex onto bone 0 (pelvis/root), which is what caused the
          //      giant-face-stuck-on-camera visual.
          //
          // Detect category 2 by presence of BLENDINDICES per-vertex and skip
          // the t31 branch entirely — let the legacy t30 / skinning machinery
          // downstream handle these (as it does for classic characters).
          bool hasBlendIndices = false;
          if (il != nullptr) {
            for (const auto& sem : il->GetRtxSemantics()) {
              if (!sem.perInstance &&
                  std::strncmp(sem.name, "BLENDINDICES", 12) == 0 &&
                  sem.index == 0) {
                hasBlendIndices = true;
                break;
              }
            }
          }
          if (hasBlendIndices) {
            t31SkipReason = "has_blendindices_skinned_character";
            Logger::warn(str::format(
              "[D3D11Rtx.o2w.t31.skip] vs=", getVsHashShort(),
              " drawID=", m_drawCallID,
              " reason=", t31SkipReason,
              " (routing to legacy skinning path)"));
            // Fall through to legacy t30 bone path below.
            modelInstSrv = nullptr;
          }
          if (!modelInstSrv) {
            if (!t31SkipReason) t31SkipReason = "no_srv_bound_at_slot";
          } else {
            Com<ID3D11Resource> t31Res;
            modelInstSrv->GetResource(&t31Res);
            auto* t31Buf = static_cast<D3D11Buffer*>(t31Res.ptr());
            const uint8_t* t31Data = nullptr;
            size_t t31Len = 0;
            if (t31Buf) {
              auto t31Map = t31Buf->GetMappedSlice();
              if (t31Map.mapPtr && t31Map.length > 0) {
                t31Data = reinterpret_cast<const uint8_t*>(t31Map.mapPtr);
                t31Len  = t31Map.length;
              } else {
                void* p = t31Buf->GetBuffer()->mapPtr(0);
                if (p) {
                  t31Data = reinterpret_cast<const uint8_t*>(p);
                  t31Len  = t31Buf->GetBuffer()->info().size;
                }
              }
            }

            // Read charIdx from COLOR1 per-instance semantic (R16G16B16A16_UINT).
            // VS disasm uses v1.x which is the first uint16 of the 8-byte entry.
            uint32_t charIdx = 0;
            const char* charIdxReason = "no_perinstance_r16g16b16a16_uint_semantic";
            if (il != nullptr) {
              for (const auto& s : il->GetRtxSemantics()) {
                if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                  const auto& instVb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                  if (instVb.buffer == nullptr) {
                    charIdxReason = "instVb_buffer_null";
                  } else {
                    DxvkBufferSlice instSlice = instVb.buffer->GetBufferSlice(instVb.offset);
                    const uint8_t* instPtr =
                      instSlice.defined() ? reinterpret_cast<const uint8_t*>(instSlice.mapPtr(0)) : nullptr;
                    const size_t instOff =
                      static_cast<size_t>(m_currentInstanceIndex) * instVb.stride + s.byteOffset;
                    if (!instPtr) {
                      charIdxReason = "instPtr_null";
                    } else if (instSlice.length() < instOff + 2) {
                      charIdxReason = "instSlice_too_small";
                    } else {
                      charIdx = reinterpret_cast<const uint16_t*>(instPtr + instOff)[0];
                      charIdxReason = "ok";
                    }
                  }
                  break;
                }
              }
            }

            // Fetch t31[charIdx].objectToCameraRelative (float3x4 at entry+0).
            constexpr uint32_t BYTES_PER_INSTANCE = 208u;
            const size_t t31Off = static_cast<size_t>(charIdx) * BYTES_PER_INSTANCE;
            if (!t31Data) {
              t31SkipReason = "t31Data_null";
            } else if (t31Off + 48 > t31Len) {
              t31SkipReason = "t31Off_oob";
            } else {
              const float* m = reinterpret_cast<const float*>(t31Data + t31Off);
              bool finite = true;
              for (int k = 0; k < 12 && finite; ++k) if (!std::isfinite(m[k])) finite = false;
              const bool r0nz = m[0] != 0.f || m[1] != 0.f || m[2] != 0.f;
              const bool r1nz = m[4] != 0.f || m[5] != 0.f || m[6] != 0.f;
              const bool r2nz = m[8] != 0.f || m[9] != 0.f || m[10] != 0.f;
              if (!finite) {
                t31SkipReason = "non_finite_matrix";
              } else if (!(r0nz && r1nz && r2nz)) {
                t31SkipReason = "zero_row_in_matrix";
              } else {
                // +cameraOrigin to shift camera-relative → absolute world.
                float camOri[3] = { 0.f, 0.f, 0.f };
                bool haveCam = false;
                if (m_hasFanoutCamOrigin) {
                  camOri[0] = m_lastFanoutCamOrigin.x;
                  camOri[1] = m_lastFanoutCamOrigin.y;
                  camOri[2] = m_lastFanoutCamOrigin.z;
                  haveCam = true;
                } else {
                  const auto& cb2 = m_context->m_state.vs.constantBuffers[2];
                  if (cb2.buffer != nullptr) {
                    const auto cb2Map = cb2.buffer->GetMappedSlice();
                    const uint8_t* p2 = reinterpret_cast<const uint8_t*>(cb2Map.mapPtr);
                    const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
                    if (p2 && base + 16 <= cb2.buffer->Desc()->ByteWidth) {
                      const float* fp = reinterpret_cast<const float*>(p2 + base + 4);
                      if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                        camOri[0] = fp[0]; camOri[1] = fp[1]; camOri[2] = fp[2];
                        haveCam = true;
                      }
                    }
                  }
                }
                const float tx = haveCam ? (m[3]  + camOri[0]) : m[3];
                const float ty = haveCam ? (m[7]  + camOri[1]) : m[7];
                const float tz = haveCam ? (m[11] + camOri[2]) : m[11];
                transforms.objectToWorld = Matrix4(
                  Vector4(m[0], m[4], m[8],  0.0f),
                  Vector4(m[1], m[5], m[9],  0.0f),
                  Vector4(m[2], m[6], m[10], 0.0f),
                  Vector4(tx,   ty,   tz,    1.0f));
                gotBoneTransform = true;
                m_lastO2wPathId = 1;

                // One-shot dump of the full t31 buffer contents the first time
                // each unique VS hits this path. Helps us see whether a shader
                // variant actually uses multiple entries or always idx 0.
                {
                  static std::unordered_set<std::string> sT31Dumped;
                  const std::string vkey = getVsHashShort();
                  if (sT31Dumped.insert(vkey).second) {
                    const uint32_t entries = std::min<uint32_t>(
                      static_cast<uint32_t>(t31Len / BYTES_PER_INSTANCE), 8u);
                    for (uint32_t e = 0; e < entries; ++e) {
                      const float* em = reinterpret_cast<const float*>(
                        t31Data + e * BYTES_PER_INSTANCE);
                      Logger::info(str::format(
                        "[D3D11Rtx.t31.dump] vs=", vkey, " entry=", e,
                        " T=(", em[3], ",", em[7], ",", em[11], ")",
                        " row0=(", em[0], ",", em[1], ",", em[2], ")",
                        " row1=(", em[4], ",", em[5], ",", em[6], ")",
                        " row2=(", em[8], ",", em[9], ",", em[10], ")"));
                    }
                  }
                }

                // Log every successful t31 draw (no cap) with full context +
                // VS hash so we can correlate which shader variants take this
                // path and disassemble representative ones.
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.t31.ok] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " rdef=", rdefFound ? 1 : 0,
                  " slot=", modelInstSlot,
                  " t31Len=", t31Len,
                  " charIdx=", charIdx,
                  " charIdxReason=", charIdxReason,
                  " haveCam=", haveCam ? 1 : 0,
                  " raw.T=(", m[3], ",", m[7], ",", m[11], ")",
                  " +cam=(", camOri[0], ",", camOri[1], ",", camOri[2], ")",
                  " final.T=(", tx, ",", ty, ",", tz, ")",
                  " row0=(", m[0], ",", m[1], ",", m[2], ")",
                  " row1=(", m[4], ",", m[5], ",", m[6], ")",
                  " row2=(", m[8], ",", m[9], ",", m[10], ")"));
              }
            }
            if (t31SkipReason) {
              Logger::warn(str::format(
                "[D3D11Rtx.o2w.t31.skip] vs=", getVsHashShort(),
                " drawID=", m_drawCallID,
                " rdef=", rdefFound ? 1 : 0,
                " slot=", modelInstSlot,
                " t31Len=", t31Len,
                " charIdx=", charIdx,
                " charIdxReason=", charIdxReason,
                " reason=", t31SkipReason));
            }
          }
          if (t31SkipReason && !strstr(t31SkipReason, "t31Data") && !modelInstSrv) {
            Logger::warn(str::format(
              "[D3D11Rtx.o2w.t31.nosrv] vs=", getVsHashShort(),
              " drawID=", m_drawCallID,
              " rdef=", rdefFound ? 1 : 0,
              " slot=", modelInstSlot,
              " reason=", t31SkipReason));
          }
        }

        // Legacy t30 bone path — only used when the t31 path above didn't
        // produce a transform (skinned characters / non-BSP draws).
        //
        // NV-DXVK principled gate: only enter this block when the VS actually
        // skins against t30. Signals (any of):
        //   - RDEF declares g_boneMatrix on the VS
        //   - VS has per-vertex BLENDINDICES semantic (skinned)
        // Without these, t30 being bound is coincidental app-state leftover;
        // reading bone[0] and using it as objectToWorld would displace static
        // meshes to whatever vestigial bone is at slot 0 (observed on
        // VS_6e3e6f28 — mesh translated to (-5223,835,32) instead of cb3's
        // correct (-5246,410,43)).
        bool t30PathEligible = false;
        {
          auto vsPtrBone = m_context->m_state.vs.shader;
          if (vsPtrBone != nullptr && vsPtrBone->GetCommonShader() != nullptr) {
            if (vsPtrBone->GetCommonShader()->FindResourceSlot("g_boneMatrix") != UINT32_MAX)
              t30PathEligible = true;
          }
          if (!t30PathEligible && il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0) {
                t30PathEligible = true;
                break;
              }
            }
          }
        }
        if (!t30PathEligible && !gotBoneTransform) {
          static std::unordered_set<std::string> sT30GateLogged;
          const std::string vkey = getVsHashShort();
          if (sT30GateLogged.insert(vkey).second) {
            Logger::info(str::format(
              "[D3D11Rtx.o2w.t30.skip] vs=", vkey,
              " reason=no_rdef_g_boneMatrix_and_no_blendindices",
              " (static mesh — cb3 CBufModelInstance path owns this draw)"));
          }
        }
        if (!gotBoneTransform && t30PathEligible) {
          const uint32_t kBoneSrvSlot = 30;
          ID3D11ShaderResourceView* boneSrv = nullptr;
          if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
            boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();
          if (boneSrv) {
            // Get bone matrix buffer
            Com<ID3D11Resource> boneRes;
            boneSrv->GetResource(&boneRes);
            auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
            DxvkBufferSlice boneBufSlice = boneBuf ? boneBuf->GetBufferSlice() : DxvkBufferSlice();
            const uint8_t* bonePtr = boneBufSlice.defined() ?
              reinterpret_cast<const uint8_t*>(boneBufSlice.mapPtr(0)) : nullptr;
            const size_t boneBufLen = boneBufSlice.defined() ? boneBufSlice.length() : 0;

            // Find the per-instance bone index from slot 1
            // (R16G16B16A16_UINT, perInstance=1, instance 0 for non-instanced draws)
            uint32_t boneIdx = 0;
            bool hasBoneIdx = false;
            for (const auto& s : il->GetRtxSemantics()) {
              if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                const auto& instVb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                if (instVb.buffer != nullptr) {
                  DxvkBufferSlice instSlice = instVb.buffer->GetBufferSlice(instVb.offset);
                  const uint8_t* instPtr = reinterpret_cast<const uint8_t*>(instSlice.mapPtr(0));
                  // Read COLOR1.y (second uint16) at the current instance index.
                  // The shader does: bone_index = BLENDINDICES(0) + COLOR1.y
                  // COLOR1 layout: [x=uint16, y=uint16, z=uint16, w=uint16]
                  // COLOR1.y = the second uint16 = byte offset +2 from semantic start
                  const size_t instOff = static_cast<size_t>(m_currentInstanceIndex) * instVb.stride + s.byteOffset;
                  if (instPtr && instSlice.length() >= instOff + 4) {
                    boneIdx = reinterpret_cast<const uint16_t*>(instPtr + instOff)[1]; // [1] = COLOR1.y
                    hasBoneIdx = true;
                  }
                }
                break;
              }
            }

            if (hasBoneIdx && bonePtr) {
              size_t boneOff = static_cast<size_t>(boneIdx) * 48;
              if (boneOff + 48 <= boneBufLen) {
                const float* m = reinterpret_cast<const float*>(bonePtr + boneOff);
                bool valid = true;
                for (int j = 0; j < 12; ++j) {
                  if (!std::isfinite(m[j])) { valid = false; break; }
                }
                if (valid) {
                  // Bone matrix is objectToWorld (float3x4, row-major).
                  transforms.objectToWorld = Matrix4(
                    Vector4(m[0], m[1], m[2],  0.0f),
                    Vector4(m[4], m[5], m[6],  0.0f),
                    Vector4(m[8], m[9], m[10], 0.0f),
                    Vector4(m[3], m[7], m[11], 1.0f));
                  gotBoneTransform = true;
                  m_lastO2wPathId = 2;
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.t30cpu] vs=", getVsHashShort(),
                    " drawID=", m_drawCallID,
                    " rawDraw=", m_rawDrawCount,
                    " inst=", m_currentInstanceIndex,
                    " boneIdx=", boneIdx,
                    " T=(", m[3], ",", m[7], ",", m[11], ")",
                    " row0=(", m[0], ",", m[1], ",", m[2], ")",
                    " row1=(", m[4], ",", m[5], ",", m[6], ")",
                    " row2=(", m[8], ",", m[9], ",", m[10], ")"));
                }
              }
            } else if (!bonePtr && boneSrv) {
              // Try multiple paths to read bone 0 from the D3D11Buffer:
              // 1. GetMappedSlice (WRITE_DISCARD mapped memory)
              // 2. DxvkBuffer direct mapPtr
              // 3. Cached from UpdateSubresource
              const float* bm = nullptr;
              // Path 1: D3D11Buffer mapped slice
              const auto mappedSlice = boneBuf->GetMappedSlice();
              if (mappedSlice.mapPtr && mappedSlice.length >= 48)
                bm = reinterpret_cast<const float*>(mappedSlice.mapPtr);
              // Path 2: DxvkBuffer direct map
              if (!bm) {
                void* p = boneBuf->GetBuffer()->mapPtr(0);
                if (p)
                  bm = reinterpret_cast<const float*>(p);
              }
              // Path 3: full-bone-cache (first bone only — bone 0 is always
              // at the start). Replaces the old m_cachedBone0 single-bone
              // fallback since we now always have the full cache.
              if (!bm && m_hasFullBoneCache && m_fullBoneCache.size() >= 48)
                bm = reinterpret_cast<const float*>(m_fullBoneCache.data());
              static uint32_t sBonePath = 0;
              if (sBonePath < 5 && bm) {
                ++sBonePath;
                Logger::info(str::format(
                  "[D3D11Rtx] Bone read: path=",
                  (bm == reinterpret_cast<const float*>(mappedSlice.mapPtr)) ? "mapped" :
                  (bm == reinterpret_cast<const float*>(m_fullBoneCache.data())) ? "cache" : "dxvkBuf",
                  " T=(", bm[3], ",", bm[7], ",", bm[11], ")",
                  " mapMode=", uint32_t(boneBuf->GetMapMode())));
              }
              if (bm) {
                bool valid = true;
                for (int j = 0; j < 12; ++j)
                  if (!std::isfinite(bm[j])) { valid = false; break; }
                if (valid) {
                  transforms.objectToWorld = Matrix4(
                    Vector4(bm[0], bm[1], bm[2],  0.0f),
                    Vector4(bm[4], bm[5], bm[6],  0.0f),
                    Vector4(bm[8], bm[9], bm[10], 0.0f),
                    Vector4(bm[3], bm[7], bm[11], 1.0f));
                  gotBoneTransform = true;
                  m_lastO2wPathId = 3;
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.t30slice] vs=", getVsHashShort(),
                    " drawID=", m_drawCallID,
                    " rawDraw=", m_rawDrawCount,
                    " T=(", bm[3], ",", bm[7], ",", bm[11], ")",
                    " row0=(", bm[0], ",", bm[1], ",", bm[2], ")"));
                  static uint32_t sBoneDiag2 = 0;
                  if (sBoneDiag2 < 10) {
                    ++sBoneDiag2;
                    Logger::info(str::format(
                      "[D3D11Rtx] Bone from MappedSlice: T=(",
                      bm[3], ",", bm[7], ",", bm[11], ")",
                      " mapPtr=", mappedSlice.mapPtr != nullptr ? 1 : 0));
                  }
                }
              }

              // Log per-draw bone info: buffer address + offset to see if it changes
              static uint32_t sGpuBoneDiag = 0;
              if (sGpuBoneDiag < 30) {
                ++sGpuBoneDiag;
                // Check the instance buffer (slot 1) pointer and offset per draw
                uintptr_t instBufAddr = 0;
                uint32_t instBufOff = 0;
                for (const auto& s3 : il->GetRtxSemantics()) {
                  if (s3.perInstance && s3.format == VK_FORMAT_R16G16B16A16_UINT) {
                    const auto& ivb3 = m_context->m_state.ia.vertexBuffers[s3.inputSlot];
                    if (ivb3.buffer != nullptr) {
                      instBufAddr = reinterpret_cast<uintptr_t>(ivb3.buffer.ptr());
                      instBufOff = ivb3.offset;
                    }
                    break;
                  }
                }
                // Also check if t30 buffer changes between draws
                uintptr_t boneBufAddr = reinterpret_cast<uintptr_t>(boneBuf);
                Logger::info(str::format(
                  "[D3D11Rtx] BoneDraw raw=", m_rawDrawCount,
                  " t30buf=", boneBufAddr,
                  " instBuf=", instBufAddr, "+", instBufOff,
                  " t30len=", boneBufLen));
              }
            }
          }
        }
        if (!gotBoneTransform) {
          // No bone matrix available — can't position this geometry
          m_lastExtractUsedFallback = true;
        }
        // NOTE: do NOT set m_foundRealProjThisFrame here — that would let
        // fmt=106 shadow draws bypass the uiFallback check in SubmitDraw.
      } else {
        // Non-R32G32_UINT draw without camera — mark as fallback so it
        // gets filtered as UI in SubmitDraw (shadow/depth pass).
        m_lastExtractUsedFallback = true;
      }
    }

    if (projSlot == UINT32_MAX && !m_hasEverFoundProj) {
      m_lastExtractUsedFallback = true;
      const auto& vp = m_context->m_state.rs.viewports[0];
      if (vp.Width > 0.0f && vp.Height > 0.0f) {
        const float aspect = vp.Width / vp.Height;
        const float fovY   = 60.0f * (3.14159265f / 180.0f);
        const float nearZ  = 0.1f;
        const float farZ   = 10000.0f;
        const float yScale = 1.0f / std::tan(fovY * 0.5f);
        const float xScale = yScale / aspect;
        const float Q      = farZ / (farZ - nearZ);
        transforms.viewToProjection = Matrix4(
          Vector4(xScale, 0.0f,   0.0f,         0.0f),
          Vector4(0.0f,   yScale, 0.0f,         0.0f),
          Vector4(0.0f,   0.0f,   Q,            1.0f),
          Vector4(0.0f,   0.0f,  -nearZ * Q,    0.0f));
        static bool s_fallbackLogged = false;
        if (!s_fallbackLogged) {
          s_fallbackLogged = true;
          Logger::info(str::format(
            "[D3D11Rtx] No projection found in cbuffers — using viewport fallback (",
            vp.Width, "x", vp.Height, " aspect=", aspect, ")"));
        }
      }
    }

    // --- VIEW MATRIX ---
    // NV-DXVK: Skip if worldToView was already set (cross-frame VP for R32G32_UINT)
    if (m_skipViewMatrixScan) goto skipViewScan;
    // Cached fast path: re-read from previously discovered location.
    // Only rescan when the cached location is invalid or doesn't contain
    // a view matrix anymore (shader change, different render pass).
    bool viewCacheHit = false;
    if (m_viewSlot != UINT32_MAX && m_viewStage >= 0 && m_viewStage < kNumStages) {
      const auto& cb = (*stageCbs[m_viewStage])[m_viewSlot];
      if (cb.buffer != nullptr) {
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr) {
          Matrix4 c = readMatrix(ptr, m_viewOffset, cb.buffer->Desc()->ByteWidth);
          if (isViewMatrix(c)) {
            m_lastWtvPathId = 5; // cached view-matrix slot
            // NV-DXVK: readCbMatrix stores rows-as-columns (passes raw[i][j]
            // to Matrix4 ctor with rows as args, which dxvk treats as cols).
            // The mathematical matrix in memory ends up stored as M^T in our
            // Matrix4. Transpose to recover the intended M, matching the
            // convention path 1/3 use after the column-storage fix.
            transforms.worldToView = transpose(c);
            viewCacheHit = true;
            // Diagnostic log: confirm path-5 latches now produce same Main.pos
            // as path 1/3. Cap to 30 to avoid spam.
            static uint32_t sPath5Log = 0;
            if (sPath5Log < 30) {
              ++sPath5Log;
              const auto& w = transforms.worldToView;
              Logger::info(str::format(
                "[D3D11Rtx.path5Cam] #", sPath5Log,
                " cam=(", w[3][0], ",", w[3][1], ",", w[3][2],
                ")  (raw t-col)"));
            }
          }
        }
      }
    }

    // Full scan fallback — same logic as before, but caches the result.
    if (!viewCacheHit && projSlot != UINT32_MAX) {
      if (projStage >= 0 && projStage < kNumStages) {
        const auto& cb = (*stageCbs[projStage])[projSlot];
        if (cb.buffer != nullptr) {
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (ptr) {
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            if (projOffset >= 64) {
              Matrix4 c = readMatrix(ptr, projOffset - 64, bufSize);
              if (isViewMatrix(c)) {
                m_lastWtvPathId = 6; // scan near projection (offset-64)
                transforms.worldToView = transpose(c); // see path 5 fix comment
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = projOffset - 64;
                static uint32_t sPath6Log = 0;
                if (sPath6Log < 30) {
                  ++sPath6Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path6Cam] #", sPath6Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
              }
            }
            if (isIdentityExact(transforms.worldToView)) {
              auto [vBase, vEnd] = cbRange(cb);
              for (size_t off = vBase; off + 64 <= vEnd; off += 16) {
                if (off >= projOffset && off < projOffset + 64) continue;
                Matrix4 c = readMatrix(ptr, off, bufSize);
                if (isViewMatrix(c)) {
                  m_lastWtvPathId = 7; // scan same-cb as projection
                  transforms.worldToView = transpose(c); // see path 5 fix comment
                  m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
                  static uint32_t sPath7Log = 0;
                  if (sPath7Log < 30) {
                    ++sPath7Log;
                    const auto& w = transforms.worldToView;
                    Logger::info(str::format(
                      "[D3D11Rtx.path7Cam] #", sPath7Log,
                      " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                  }
                  break;
                }
              }
            }
          }
        }
      }

      // Cross-stage fallback: scan all stages' cbuffers for a view matrix.
      if (isIdentityExact(transforms.worldToView)) {
        for (int si = 0; si < kNumStages && isIdentityExact(transforms.worldToView); ++si) {
          const auto& cbs = *stageCbs[si];
          for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
            if (si == projStage && slot == projSlot) continue;
            const auto& cb = cbs[slot];
            if (cb.buffer == nullptr) continue;
            const auto mapped = cb.buffer->GetMappedSlice();
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            if (!ptr) continue;
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            auto [csBase, csEnd] = cbRange(cb);
            for (size_t off = csBase; off + 64 <= csEnd; off += 16) {
              Matrix4 c = readMatrix(ptr, off, bufSize);
              if (isViewMatrix(c)) {
                m_lastWtvPathId = 8; // cross-stage all-cb scan
                transforms.worldToView = transpose(c); // see path 5 fix comment
                m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
                static uint32_t sPath8Log = 0;
                if (sPath8Log < 30) {
                  ++sPath8Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path8Cam] #", sPath8Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
                break;
              }
            }
            if (!isIdentityExact(transforms.worldToView)) break;
          }
        }
      }

      // Convention fallback: if no view matrix was found, the column-major
      // detection may be wrong (ambiguous when near plane ≈ 1). Retry with
      // the opposite convention, but only for the projection cbuffer.
      if (isIdentityExact(transforms.worldToView) && projStage >= 0 && projStage < kNumStages) {
        const auto& cb = (*stageCbs[projStage])[projSlot];
        if (cb.buffer != nullptr) {
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (ptr) {
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            auto [fbBase, fbEnd] = cbRange(cb);
            for (size_t off = fbBase; off + 64 <= fbEnd; off += 16) {
              if (off >= projOffset && off < projOffset + 64) continue;
              Matrix4 raw = readCbMatrix(ptr, off, bufSize);
              Matrix4 flipped = m_columnMajor ? raw : transpose(raw);
              if (isViewMatrix(flipped)) {
                m_lastWtvPathId = 9; // convention-flip fallback
                transforms.worldToView = transpose(flipped); // see path 5 fix comment
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
                m_columnMajor = !m_columnMajor;
                static uint32_t sPath9Log = 0;
                if (sPath9Log < 30) {
                  ++sPath9Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path9Cam] #", sPath9Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
                break;
              }
            }
          }
        }
      }
    }

    // When using fallback projection (projSlot == UINT32_MAX), still search
    // all stages for a view matrix so the camera position is correct.
    if (!viewCacheHit && projSlot == UINT32_MAX && isIdentityExact(transforms.worldToView)) {
      for (int si = 0; si < kNumStages && isIdentityExact(transforms.worldToView); ++si) {
        const auto& cbs = *stageCbs[si];
        for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
          const auto& cb = cbs[slot];
          if (cb.buffer == nullptr) continue;
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (!ptr) continue;
          const size_t bufSize = cb.buffer->Desc()->ByteWidth;
          auto [csBase, csEnd] = cbRange(cb);
          for (size_t off = csBase; off + 64 <= csEnd; off += 16) {
            Matrix4 c = readMatrix(ptr, off, bufSize);
            if (isViewMatrix(c)) {
              m_lastWtvPathId = 10; // fallback-projection branch cross-stage scan
              transforms.worldToView = transpose(c); // see path 5 fix comment
              m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
              static uint32_t sPath10Log = 0;
              if (sPath10Log < 30) {
                ++sPath10Log;
                const auto& w = transforms.worldToView;
                Logger::info(str::format(
                  "[D3D11Rtx.path10Cam] #", sPath10Log,
                  " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
              }
              break;
            }
          }
          if (!isIdentityExact(transforms.worldToView)) break;
        }
      }
    }

    skipViewScan:
    // --- Z-UP / Y-UP AUTO-DETECTION (view-matrix-derived) ---
    // In a Y-up world, the view matrix "up" column (col 1) has its largest
    // component in row 1 (Y). In a Z-up world, column 1's largest component
    // is in row 2 (Z). Vote on each valid view matrix and settle via threshold.
    if (!isIdentityExact(transforms.worldToView)) {
      if (!m_zUpSettled) {
        const float absY = std::abs(transforms.worldToView[1][1]);
        const float absZ = std::abs(transforms.worldToView[2][1]);
        // Only vote when there's a clear winner (avoid ambiguous 45° views)
        if (std::abs(absZ - absY) > 0.3f) {
          m_zUpVotes += (absZ > absY) ? 1 : -1;
          if (!m_zUpSettled && std::abs(m_zUpVotes) >= kVoteThreshold) {
            m_zUpSettled = true;
            const bool zUp = m_zUpVotes > 0;
            RtxOptions::zUpObject().setDeferred(zUp);
          }
        }
      }

      // Log settled axis conventions once.
      if (m_zUpSettled && m_yFlipSettled && m_lhSettled && !m_axisLogged) {
        m_axisLogged = true;
        Logger::info(str::format("[D3D11Rtx] Axis detection settled: ",
          m_lhVotes > 0 ? "LH" : "RH",
          m_yFlipVotes > 0 ? " Y-flipped" : "",
          m_zUpVotes > 0 ? " Z-up" : " Y-up",
          m_columnMajor ? " col-major" : " row-major",
          " (proj stage=", kStageNames[std::max(0, m_projStage)],
          " slot=", m_projSlot, " off=", m_projOffset, ")"));
      }
    }

    // --- CAMERA POSITION SMOOTHING ---
    // The view matrix encodes camera position in its translation row (row 3).
    // Floating-point rounding in cbuffer reads causes sub-pixel jitter between
    // draws/frames. Apply exponential moving average on the position to dampen
    // this without introducing visible lag. The rotation (upper 3x3) is left
    // untouched — rotation jitter is rare and smoothing it causes ghosting.
    //
    // NV-DXVK: Smoothing is ONLY applied to VP-decomposition paths (1, 2, 3)
    // where float rounding in the row-magnitude normalization + basis
    // re-derivation actually produces jitter. For the cached-slot scan paths
    // (5-10) the translation column is read verbatim from a real view matrix
    // cbuffer — no jitter — and smoothing just introduces lag. m_lastWtvPathId
    // lets us gate cleanly. Paths 0 and 11 are also excluded (bone-composite).
    //
    // D3D row-major view matrix layout:
    //   [R00 R01 R02  0]    pos = -R^T * t
    //   [R10 R11 R12  0]    where t = (V[3][0], V[3][1], V[3][2])
    //   [R20 R21 R22  0]
    //   [tx  ty  tz   1]
    const bool smoothingApplies =
      m_lastWtvPathId == 1 || m_lastWtvPathId == 2 || m_lastWtvPathId == 3;
    if (smoothingApplies && !isIdentityExact(transforms.worldToView) && !m_skipViewMatrixScan) {
      // NV-DXVK [bob-bug-fix 2026-05-04]: this block USED to EMA-smooth
      // camPos and rebuild the translation column as t' = -V·smoothedCamPos.
      // That is mathematically equivalent to "snap to a slightly-different
      // camera world position", and since V (basis) carries TF2's bob roll
      // and changes every frame, t' = -V_n · smoothedCamPos oscillates
      // frame-to-frame even when smoothedCamPos is essentially stable.
      // Downstream consumers reading raw W[3] (sky detector at
      // rtx_types.cpp:429, motion-vector compute, BLAS positioning, classifier
      // hysteresis) see this oscillation and produce visible per-frame
      // motion in whatever axis the (basis × delta-position) projection lands
      // on — observed as the "crouch+stand" Z bob that's plagued this fork.
      //
      // Empirical proof: bypassing this rebuild (or any equivalent snap in
      // rtx_camera_manager) shifts the visible motion from Z to Y or X
      // depending on what's pinned. The motion source is the rebuild itself,
      // not basis-noise amplification.
      //
      // Original intent was to dampen sub-pixel cbuffer-read jitter from VP
      // decomposition (paths 1/2/3). That jitter is much smaller than the
      // visible bob this rebuild introduced — leaving t verbatim is strictly
      // better. The diagnostic camPos compute is kept for logging continuity.
      const auto& V = transforms.worldToView;
      Vector3 t(V[3][0], V[3][1], V[3][2]);
      Vector3 camPos(
        -(V[0][0] * t.x + V[1][0] * t.y + V[2][0] * t.z),
        -(V[0][1] * t.x + V[1][1] * t.y + V[2][1] * t.z),
        -(V[0][2] * t.x + V[1][2] * t.y + V[2][2] * t.z));
      m_smoothedCamPos = camPos;
      m_hasPrevCamPos = true;
      // Diagnostic: log first ~80 events with the would-be rebuild delta so
      // we can see how much the basis-rebuild was perturbing W[3] per frame.
      {
        static uint32_t sBobBugLog = 0;
        if (sBobBugLog < 80) {
          ++sBobBugLog;
          const float wouldBeTx = -(V[0][0]*camPos.x + V[0][1]*camPos.y + V[0][2]*camPos.z);
          const float wouldBeTy = -(V[1][0]*camPos.x + V[1][1]*camPos.y + V[1][2]*camPos.z);
          const float wouldBeTz = -(V[2][0]*camPos.x + V[2][1]*camPos.y + V[2][2]*camPos.z);
          Logger::info(str::format(
            "[bob-bug] path=", m_lastWtvPathId,
            " keptT=(", t.x, ",", t.y, ",", t.z, ")",
            " wouldBeT=(", wouldBeTx, ",", wouldBeTy, ",", wouldBeTz, ")",
            " camPos=(", camPos.x, ",", camPos.y, ",", camPos.z, ")"));
        }
      }
    }

    // --- WORLD MATRIX ---
    // NV-DXVK: For R32G32_UINT draws, read cb3 directly from its mapped memory.
    // cb3 is updated via Map/WRITE_DISCARD (not UpdateSubresource).
    // GetMappedSlice() returns the CPU-mapped pointer with current data.
    if (m_skipViewMatrixScan) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const auto& cb3 = vsCbs[3];
      const float* bm = nullptr;
      if (cb3.buffer != nullptr) {
        const auto mapped = cb3.buffer->GetMappedSlice();
        if (mapped.mapPtr && mapped.length >= static_cast<size_t>(cb3.constantOffset) * 16 + 48) {
          bm = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(mapped.mapPtr) + static_cast<size_t>(cb3.constantOffset) * 16);
        }
        static uint32_t sCb3Diag = 0;
        if (sCb3Diag < 10) {
          ++sCb3Diag;
          Logger::info(str::format(
            "[D3D11Rtx] CB3 read: mapPtr=", mapped.mapPtr != nullptr ? 1 : 0,
            " mapMode=", uint32_t(cb3.buffer->GetMapMode()),
            " usage=", uint32_t(cb3.buffer->Desc()->Usage),
            " size=", cb3.buffer->Desc()->ByteWidth,
            " off=", cb3.constantOffset,
            " bm=", bm != nullptr ? 1 : 0));
        }
      }
      // NV-DXVK: only run CB3→O2W if no upstream path (t31 at line 2342,
      // or legacy t30 bone paths at 2551/2604) already set objectToWorld.
      // Without this gate, CB3→O2W clobbers the t31-derived o2w with a
      // stale cb3 read for every R32G32_UINT draw — the histogram showed
      // cb3=32 commits per frame and t31=0 despite t31 firing successfully
      // thousands of times.
      if (bm && m_lastO2wPathId == 0) {
        Matrix4 cb3Mat(
          Vector4(bm[0], bm[1], bm[2],  0.0f),
          Vector4(bm[4], bm[5], bm[6],  0.0f),
          Vector4(bm[8], bm[9], bm[10], 0.0f),
          Vector4(bm[3], bm[7], bm[11], 1.0f));
        Matrix4 invView = inverse(transforms.worldToView);
        transforms.objectToWorld = invView * cb3Mat;
        m_lastO2wPathId = 4;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.cb3] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " cb3.T=(", cb3Mat[3][0], ",", cb3Mat[3][1], ",", cb3Mat[3][2], ")",
          " o2w.T=(", transforms.objectToWorld[3][0], ",",
          transforms.objectToWorld[3][1], ",",
          transforms.objectToWorld[3][2], ")"));
      }
    }

    // Scan VS cbuffers first (model matrices live in VS for virtually all engines),
    // then fall back to other stages for emulator compatibility.
    // Gated by useCBufferWorldMatrices — disable if CB layout causes wrong detections.
    // NV-DXVK: Skip for R32G32_UINT draws — cached cb3 is already set above.
    if (RtxOptions::useCBufferWorldMatrices() && !m_currentDrawIsBoneTransformed && !m_skipViewMatrixScan) {
      // --- Source-engine float3x4 world matrix (translation in column 3) ---
      // IDA analysis of materialsystem_dx11.dll confirms:
      //   VS slot 0 = per-draw texture/viewport constants (set by materialsystem)
      //   VS slot 1 = per-draw material/skinning constants (set by materialsystem)
      //   VS slots 2+ = set by engine/game code (not materialsystem)
      //   VS slot 2 = combined VP matrix at offset 96 (camera)
      //   VS slot 3 = [objectToWorld float3x4 | worldToView float3x4] (96 bytes total)
      //
      // Source/Titanfall 2 stores objectToWorld as a float3x4 at VS slot=3 offset=0.
      // Format:
      //   Row 0: [R00 R01 R02 Tx]   ← 16 bytes, translation in COLUMN 3
      //   Row 1: [R10 R11 R12 Ty]   ← 16 bytes
      //   Row 2: [R20 R21 R22 Tz]   ← 16 bytes
      //   (offset +48: second float3x4, the worldToView, NOT another object matrix)
      //
      // This is checked FIRST before the generic 4x4 scanner to prevent the
      // scanner from picking up false positives from materialsystem's slot 0/1 data.
      auto trySourceFloat3x4 = [&](const D3D11ConstantBufferBindings& cbs,
                                    uint32_t slot, uint32_t byteOffset = 0) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t cbBase  = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t base    = cbBase + byteOffset;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        // Need at least 48 bytes (3 rows × 16 bytes).
        if (base + 48 > bufSize) return false;
        const float* f = reinterpret_cast<const float*>(ptr + base);
        // Read the 3×4 matrix.
        const float R00 = f[0],  R01 = f[1],  R02 = f[2],  Tx = f[3];
        const float R10 = f[4],  R11 = f[5],  R12 = f[6],  Ty = f[7];
        const float R20 = f[8],  R21 = f[9],  R22 = f[10], Tz = f[11];
        // Sanity: all 12 entries must be finite.
        if (!std::isfinite(R00) || !std::isfinite(R01) || !std::isfinite(R02) ||
            !std::isfinite(R10) || !std::isfinite(R11) || !std::isfinite(R12) ||
            !std::isfinite(R20) || !std::isfinite(R21) || !std::isfinite(R22) ||
            !std::isfinite(Tx)  || !std::isfinite(Ty)  || !std::isfinite(Tz))
          return false;
        // Sanity: each column of the 3×3 rotation block must have unit length
        // (approximately orthonormal).  Degenerate zero-columns are rejected.
        const float col0Sq = R00*R00 + R10*R10 + R20*R20;
        const float col1Sq = R01*R01 + R11*R11 + R21*R21;
        const float col2Sq = R02*R02 + R12*R12 + R22*R22;
        if (col0Sq < 0.25f || col0Sq > 4.0f) return false;
        if (col1Sq < 0.25f || col1Sq > 4.0f) return false;
        if (col2Sq < 0.25f || col2Sq > 4.0f) return false;
        // Reject if it looks like a perspective matrix (would have large off-diagonal
        // entries and near-zero diagonal in the third row).
        if (classifyPerspective(Matrix4(
              Vector4(R00, R01, R02, 0.0f),
              Vector4(R10, R11, R12, 0.0f),
              Vector4(R20, R21, R22, 0.0f),
              Vector4(Tx,  Ty,  Tz,  1.0f))) != 0)
          return false;
        // NV-DXVK (Titanfall 2): reject identity. TF2 packs two float3x4 blocks
        // back-to-back in VS s3 where block 0 (offset 0) is identity and block 1
        // (offset 48) is the real objectToWorld. Accepting identity here made the
        // caller stop searching and every static world draw landed at origin.
        if (std::abs(R00 - 1.0f) < 1e-6f && std::abs(R11 - 1.0f) < 1e-6f && std::abs(R22 - 1.0f) < 1e-6f
            && std::abs(R01) < 1e-6f && std::abs(R02) < 1e-6f
            && std::abs(R10) < 1e-6f && std::abs(R12) < 1e-6f
            && std::abs(R20) < 1e-6f && std::abs(R21) < 1e-6f
            && std::abs(Tx) < 1e-6f && std::abs(Ty) < 1e-6f && std::abs(Tz) < 1e-6f)
          return false;
        // Row-major Matrix4 from row-major float3x4.
        transforms.objectToWorld = Matrix4(
          Vector4(R00, R01, R02, 0.0f),
          Vector4(R10, R11, R12, 0.0f),
          Vector4(R20, R21, R22, 0.0f),
          Vector4(Tx,  Ty,  Tz,  1.0f));
        m_lastO2wPathId = 6;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.sf3x4] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " slot=", slot, " off=", byteOffset,
          " T=(", Tx, ",", Ty, ",", Tz, ")"));
        return true;
      };

      // --- Standard 4x4 world matrix scan (D3D row-major convention) ---
      // Used as fallback for non-Source engines where the model matrix is stored
      // in the standard D3D convention (translation in row 3, column 3 = zero).
      // SKIPS VS slots 0-1 which are owned by materialsystem (per-draw constants
      // that contain viewport/texture data, not world transforms).
      auto tryWorldCb = [&](const D3D11ConstantBufferBindings& cbs, uint32_t slot,
                            int skipStage, uint32_t skipSlot) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        if (base + 64 > bufSize) return false;
        Matrix4 candidate = readMatrix(ptr, base, bufSize);
        if (isIdentityExact(candidate) || classifyPerspective(candidate) != 0 || isViewMatrix(candidate))
          return false;
        if (std::abs(candidate[3][3] - 1.0f) > 0.01f) return false;
        if (std::abs(candidate[0][3]) > 0.01f || std::abs(candidate[1][3]) > 0.01f || std::abs(candidate[2][3]) > 0.01f)
          return false;
        transforms.objectToWorld = candidate;
        m_lastO2wPathId = 7;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.worldcb] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " slot=", slot,
          " T=(", candidate[3][0], ",", candidate[3][1], ",", candidate[3][2], ")"));
        return true;
      };

      // NV-DXVK: sync `found` with the path-id system so the world-matrix
      // scan below (RDEF, trySourceFloat3x4, tryWorldCb) doesn't overwrite
      // an o2w already set by an upstream path (t31=1, t30cpu=2, t30slice=3).
      bool found = (m_lastO2wPathId != 0);
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;

      // NV-DXVK: for shaders with a per-vertex BLENDINDICES semantic (skinned
      // characters), the world transform does NOT live in any cbuffer — it's
      // exclusively in the t30/t31 SRV bone/model palette. Scanning cbuffers
      // for character shaders produces garbage (typically picks up a cb3
      // region that happens to contain -cameraOrigin as a 3x4 translation,
      // which then stamps o2w=-cam and plasters the character's BLAS over
      // the camera origin). Short-circuit `found = true` when BLENDINDICES
      // is present so the entire scan block below is bypassed. If t30/t31
      // couldn't produce an o2w upstream, the draw stays at identity (and
      // gets filtered as UI-fallback downstream) which is strictly better
      // than a wrong non-identity matrix.
      if (!found) {
        auto* ilGate = m_context->m_state.ia.inputLayout.ptr();
        if (ilGate) {
          for (const auto& s : ilGate->GetRtxSemantics()) {
            if (!s.perInstance &&
                std::strncmp(s.name, "BLENDINDICES", 12) == 0 &&
                s.index == 0) {
              found = true;  // poison pill: skip the cbuffer scans
              static uint32_t sBiPoisonLog = 0;
              if (sBiPoisonLog < 20) {
                ++sBiPoisonLog;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.scan.skip] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " reason=has_blendindices_no_cbuffer_scan"));
              }
              break;
            }
          }
        }
      }

      // NV-DXVK: DETERMINISTIC EXTRACTION via DXBC RDEF reflection.
      // The VS itself declares the cbuffers it binds (names + slots) and their
      // field offsets. We look up by HLSL cbuffer/field name — no size or content
      // heuristics. Only guessing is replaced; legacy path retained below as a
      // fallback for shaders that stripped RDEF.
      const D3D11CommonShader* commonVS = nullptr;
      auto vsPtr = m_context->m_state.vs.shader;
      if (vsPtr != nullptr) commonVS = vsPtr->GetCommonShader();

      auto rdefReadFloats = [&](const D3D11ConstantBufferBindings& cbs,
                                uint32_t slot, uint32_t fieldOff,
                                uint32_t fieldSize, float* out) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + fieldOff;
        if (base + fieldSize > cb.buffer->Desc()->ByteWidth) return false;
        std::memcpy(out, ptr + base, fieldSize);
        return true;
      };

      if (commonVS != nullptr) {
        // Titanfall 2 Source Engine 2 transform chain (verified via RDEF).
        // Some shader variants expose a `CBufModelInstance` cbuffer with
        // `objectToCameraRelative` at offset 0. Most BSP shaders instead
        // use the g_modelInst SRV (t31); those are handled upstream in the
        // fanout path and the non-instanced t31 read at ~line 2342.
        auto modelCb = commonVS->FindCBuffer("CBufModelInstance");

        // NV-DXVK DIAG: PIX confirmed VS 0x298e12b3d5bcd082 (merged[Opaque][0]
        // warped-mesh VS, SHA1 6e3e6f28...) binds a valid 3x4 row-major
        // objectToCameraRelative at cb slot 3 with uniform scale 0.3. If
        // FindCBuffer("CBufModelInstance") misses for this VS, the merged
        // variant uses a different HLSL cbuffer name. Dump all declared
        // cbuffers + raw cb3 bytes so we can identify the correct name.
        {
          // One-shot per VS-match; safe to leave ungated.
          const std::string vsKeyDiag = getVsHashShort();
          const bool isWarpedMeshVs = vsKeyDiag.find("VS_6e3e6f28f2156ea2") != std::string::npos;
          if (isWarpedMeshVs) {
            static bool sLoggedOnce = false;
            if (!sLoggedOnce) {
              sLoggedOnce = true;
              auto names = commonVS->GetCBufferNamesAndSlots();
              std::string cbList;
              for (const auto& p : names) {
                cbList += p.first + "@slot" + std::to_string(p.second) + " ";
              }
              Logger::info(str::format(
                "[D3D11Rtx.o2w.warpedMesh.rdefDump] vs=", vsKeyDiag,
                " cbufferCount=", names.size(),
                " cbuffers={ ", cbList, "}",
                " modelCbFound=", (modelCb != nullptr) ? 1 : 0,
                " modelCbSlot=", modelCb ? modelCb->bindSlot : UINT32_MAX));

              // Dump raw cb3 bytes as 12 floats (the objectToCameraRelative we
              // know PIX has for this draw). Bypasses RDEF to confirm the raw
              // binding has the matrix.
              const uint32_t kRawSlot = 3;
              if (kRawSlot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                const auto& cb = vsCbs[kRawSlot];
                if (cb.buffer != nullptr) {
                  const auto mapped = cb.buffer->GetMappedSlice();
                  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  if (ptr) {
                    const size_t base = static_cast<size_t>(cb.constantOffset) * 16;
                    if (base + 48 <= cb.buffer->Desc()->ByteWidth) {
                      float raw[12];
                      std::memcpy(raw, ptr + base, 48);
                      Logger::info(str::format(
                        "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                        " bufSize=", cb.buffer->Desc()->ByteWidth,
                        " constOff=", cb.constantOffset,
                        " r0=(", raw[0], ",", raw[1], ",", raw[2], ",", raw[3], ")",
                        " r1=(", raw[4], ",", raw[5], ",", raw[6], ",", raw[7], ")",
                        " r2=(", raw[8], ",", raw[9], ",", raw[10], ",", raw[11], ")"));
                    } else {
                      Logger::info(str::format(
                        "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                        " base+48 exceeds bufSize=", cb.buffer->Desc()->ByteWidth,
                        " constOff=", cb.constantOffset));
                    }
                  } else {
                    Logger::info(str::format(
                      "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                      " slot3 buffer has no mapPtr"));
                  }
                } else {
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                    " slot3 has NO buffer bound"));
                }
              }
            }
          }
        }

        if (modelCb && modelCb->bindSlot != UINT32_MAX) {
          float m[12];
          if (rdefReadFloats(vsCbs, modelCb->bindSlot, 0, 48, m)) {
            bool ok = true;
            for (int k = 0; k < 12 && ok; ++k)
              if (!std::isfinite(m[k])) ok = false;
            if (ok) {
              // Also fetch c_cameraOrigin from CBufCommonPerCamera (offset 4, 3 floats).
              float camO[3] = { 0.f, 0.f, 0.f };
              bool haveCamO = false;
              if (auto camLoc = commonVS->FindCBField("CBufCommonPerCamera", "c_cameraOrigin")) {
                if (camLoc->size >= 12 && camLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                  if (rdefReadFloats(vsCbs, camLoc->slot, camLoc->offset, 12, camO)) {
                    if (std::isfinite(camO[0]) && std::isfinite(camO[1]) && std::isfinite(camO[2])) {
                      haveCamO = true;
                    }
                  }
                }
              }

              // Detect "cb3 is identity-with-zero-translation": the BSP world render pass
              // uses cb3 = { I | 0 } because its vertex buffer is already in cam-relative
              // world space — the VS matmul passes vertices through unchanged to cam-relative.
              // For Remix RT we need objectToWorld = translate(+cameraOrigin) so those
              // cam-relative vertices land at absolute world in the BLAS.
              constexpr float kEps = 1e-4f;
              auto near_ = [&](float a, float b) { return std::abs(a - b) < kEps; };
              const bool cb3IsIdentityZeroT =
                   near_(m[0], 1.f) && near_(m[1], 0.f) && near_(m[2],  0.f) && near_(m[3],  0.f)
                && near_(m[4], 0.f) && near_(m[5], 1.f) && near_(m[6],  0.f) && near_(m[7],  0.f)
                && near_(m[8], 0.f) && near_(m[9], 0.f) && near_(m[10], 1.f) && near_(m[11], 0.f);
              const bool cb3IsAllZero =
                   m[0]==0.f && m[1]==0.f && m[2]==0.f && m[3]==0.f
                && m[4]==0.f && m[5]==0.f && m[6]==0.f && m[7]==0.f
                && m[8]==0.f && m[9]==0.f && m[10]==0.f && m[11]==0.f;
              const bool cb3IsZero = cb3IsIdentityZeroT || cb3IsAllZero;

              // c_cameraOrigin may be zero for this specific draw's cb2 binding
              // (BSP world pass). For the zeroCb3 path only, fall back to the
              // fanout-captured authoritative gameplay camera origin.
              // IMPORTANT: do NOT apply this fallback to the non-zero-cb3 path,
              // since those draws have their own real translation and adding
              // camOrigin on top would double-shift and displace them.
              float camOforZeroCb3[3] = { camO[0], camO[1], camO[2] };
              bool haveCamOforZeroCb3 = haveCamO;
              const bool rdefCamOValid = haveCamO
                && (camO[0] != 0.f || camO[1] != 0.f || camO[2] != 0.f);
              if (!rdefCamOValid && m_hasFanoutCamOrigin) {
                camOforZeroCb3[0] = m_lastFanoutCamOrigin.x;
                camOforZeroCb3[1] = m_lastFanoutCamOrigin.y;
                camOforZeroCb3[2] = m_lastFanoutCamOrigin.z;
                haveCamOforZeroCb3 = true;
              }

              // BSP world VS (VS_bb30826b) fanout path already puts absolute-world
              // translations in i2o (via adjTx = m[3]+camOrigin in the fanout code).
              // For that specific VS + fanout, leave objectToWorld=identity to avoid
              // double-shifting by camOrigin. Every other zeroCb3 case (including
              // fanout particles) gets translate(+camOrigin) as before.
              const bool isFanoutDraw = (m_currentInstancesToObject != nullptr);
              // Short VS key = "VS_" + first 16 hex of SHA1. VS_bb30826b's SHA1 starts
              // with "bb30826b03dc9a8b". Compare by SHA1 prefix string.
              std::string vsKey = getVsHashShort();
              const bool isBspWorldVsFanout = isFanoutDraw
                && vsKey.find("VS_bb30826b03dc9a8b") != std::string::npos;
              if (cb3IsZero && isBspWorldVsFanout) {
                transforms.objectToWorld = Matrix4();  // identity
                m_lastO2wPathId = 7;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.rdef.zeroCb3.bspFanout] vs=", vsKey,
                  " drawID=", m_drawCallID,
                  " → identity (i2o already has +camOrigin from fanout)"));
              } else if (cb3IsZero && haveCamOforZeroCb3) {
                transforms.objectToWorld = Matrix4(
                  Vector4(1.f, 0.f, 0.f, 0.f),
                  Vector4(0.f, 1.f, 0.f, 0.f),
                  Vector4(0.f, 0.f, 1.f, 0.f),
                  Vector4(camOforZeroCb3[0], camOforZeroCb3[1], camOforZeroCb3[2], 1.f));
                m_lastO2wPathId = 6;
                // NV-DXVK: throttle — was firing unconditionally per-draw
                // and hitting ~470/sec on loading screens, contributing to
                // the per-present log storm. One line per unique VS is
                // sufficient for diagnosing zeroCb3 routing; detailed
                // per-draw data still reachable via other traces.
                static std::unordered_set<std::string> sZeroCb3Log;
                if (sZeroCb3Log.insert(vsKey).second) {
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.rdef.zeroCb3] vs=", vsKey,
                    " drawID=", m_drawCallID,
                    " isFanout=", isFanoutDraw ? 1 : 0,
                    " camO=(", camOforZeroCb3[0], ",", camOforZeroCb3[1], ",", camOforZeroCb3[2], ")"));
                }
              } else {
                const float tx = haveCamO ? (m[3]  + camO[0]) : m[3];
                const float ty = haveCamO ? (m[7]  + camO[1]) : m[7];
                const float tz = haveCamO ? (m[11] + camO[2]) : m[11];
                // Row-major float3x4: col c of rotation = (m[c], m[4+c], m[8+c]).
                transforms.objectToWorld = Matrix4(
                  Vector4(m[0], m[4], m[8],  0.f),
                  Vector4(m[1], m[5], m[9],  0.f),
                  Vector4(m[2], m[6], m[10], 0.f),
                  Vector4(tx,   ty,   tz,    1.f));
                m_lastO2wPathId = 5;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.rdef] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " slot=", modelCb->bindSlot,
                  " r0=(", m[0], ",", m[1], ",", m[2], ") Tx=", m[3],
                  " r1=(", m[4], ",", m[5], ",", m[6], ") Ty=", m[7],
                  " r2=(", m[8], ",", m[9], ",", m[10], ") Tz=", m[11],
                  " camO=(", camO[0], ",", camO[1], ",", camO[2], ")",
                  " T_abs=(", tx, ",", ty, ",", tz, ")",
                  " haveCamO=", haveCamO ? 1 : 0));
              }
              found = true;
            }
          }
        }
        // Otherwise objectToWorld stays identity (correct for VBs already in
        // camera-relative coords — e.g. world mesh / screen-space passes).
      }

      // ======== LEGACY HEURISTIC FALLBACK (shaders without RDEF only) ========
      // Skipped entirely when we have commonVS-derived metadata above.
      if (!found && commonVS == nullptr) {
        const uint32_t kSourceModelSlot = 3;

        // === VERBOSE SLOT-3 DIAGNOSTIC ===
        // Log slot 3 state every Nth draw (per-frame counter resets outside).
        // Helps diagnose why objectToWorld stays identity in-game.
        static uint32_t s_slot3DiagFrame = UINT32_MAX;
        static uint32_t s_slot3DiagDrawInFrame = 0;
        {
          // Track frame transitions by watching m_drawCallID reset to 0.
          static uint32_t s_lastDrawCallID = UINT32_MAX;
          if (m_drawCallID == 0 || m_drawCallID < s_lastDrawCallID) {
            s_slot3DiagFrame++;
            s_slot3DiagDrawInFrame = 0;
          }
          s_lastDrawCallID = m_drawCallID;
          s_slot3DiagDrawInFrame++;
        }
        // Log: first 3 draws of every in-game frame (camValid via draws>0 approximation),
        // and any draw where slot3 is NULL or identity (unexpected).
        const bool isEarlyDraw = (s_slot3DiagDrawInFrame <= 3);
        const bool logThisDraw = (s_slot3DiagFrame < 600) && (isEarlyDraw || (m_drawCallID % 10 == 0));
        if (logThisDraw) {
          const auto& cb3 = vsCbs[kSourceModelSlot];
          if (cb3.buffer == nullptr) {
            Logger::warn(str::format(
              "[D3D11Rtx] slot3 NULL  frame=", s_slot3DiagFrame,
              " drawInFrame=", s_slot3DiagDrawInFrame,
              " m_drawCallID=", m_drawCallID));
          } else {
            const auto mapped = cb3.buffer->GetMappedSlice();
            const float* f = reinterpret_cast<const float*>(
              static_cast<const uint8_t*>(mapped.mapPtr) + static_cast<size_t>(cb3.constantOffset) * 16);
            if (f && cb3.buffer->Desc()->ByteWidth >= static_cast<size_t>(cb3.constantOffset) * 16 + 48) {
              Logger::info(str::format(
                "[D3D11Rtx] slot3 raw  frame=", s_slot3DiagFrame,
                " draw=", s_slot3DiagDrawInFrame, "/", m_drawCallID,
                " buf=", cb3.buffer->Desc()->ByteWidth,
                " off=", cb3.constantOffset,
                " R0=(", f[0], ",", f[1], ",", f[2], ",", f[3], ")",
                " R1=(", f[4], ",", f[5], ",", f[6], ",", f[7], ")",
                " R2=(", f[8], ",", f[9], ",", f[10], ",", f[11], ")"));
            }
          }
        }

        if (!found)
          found = trySourceFloat3x4(vsCbs, kSourceModelSlot, 0);

        // NV-DXVK (Titanfall 2): camera-relative rendering fallback.
        // Most TF2 VS (VS_759738774e, VS_ef94e6c7, ...) use CBufCommonPerCamera at
        // cb2 and have NO CBufModelInstance at cb3. Vertex buffers already contain
        // (world - c_cameraOrigin); the VS only multiplies by c_cameraRelativeToClip.
        // Remix defaults o2w=identity, builds BLAS at origin, camera sees nothing.
        // Restore world placement by using c_cameraOrigin (cb2 offset 4, float3)
        // as the o2w translation: BLAS_vert + cameraOrigin = (world - cameraOrigin)
        //                                                  + cameraOrigin = world.
        //
        // NV-DXVK: capture the draw's cameraOrigin for the TLAS-coherence filter
        // that runs at the SubmitInstancedDraw call site after ExtractTransforms
        // returns (can't bare-return here — this function returns a value).
        if (!found) {
          const auto& cb2 = vsCbs[2];
          if (cb2.buffer != nullptr) {
            const auto mapped = cb2.buffer->GetMappedSlice();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
            const size_t sz = cb2.buffer->Desc()->ByteWidth;
            if (p && base + 16 <= sz) {
              const float* f = reinterpret_cast<const float*>(p + base);
              // c_cameraOrigin at offset 4 (f[1..3]); f[0] is c_zNear.
              const float camX = f[1], camY = f[2], camZ = f[3];
              // NV-DXVK: diagnostic — log cb2 cameraOrigin for the first few
              // draws per frame, including the BSP gameplay hashes. Expected:
              // all BSP draws in one frame should read the SAME cameraOrigin;
              // if they disagree, we've got stale/inconsistent cb2 state.
              // Also read prev-frame origin at offset 84 for comparison.
              {
                static uint32_t sCamOrigLog = 0;
                if (sCamOrigLog < 60) {
                  // Get current VS hash
                  XXH64_hash_t vsH = 0;
                  if (m_context->m_state.vs.shader != nullptr
                      && m_context->m_state.vs.shader->GetCommonShader() != nullptr) {
                    auto& s = m_context->m_state.vs.shader->GetCommonShader()->GetShader();
                    if (s != nullptr) vsH = static_cast<XXH64_hash_t>(s->getHash());
                  }
                  // Read prev-frame cameraOrigin at offset 84 (float3)
                  float prevX = 0, prevY = 0, prevZ = 0;
                  bool havePrev = false;
                  if (base + 96 <= sz) {
                    const float* fp = reinterpret_cast<const float*>(p + base + 84);
                    prevX = fp[0]; prevY = fp[1]; prevZ = fp[2];
                    havePrev = true;
                  }
                  ++sCamOrigLog;
                  char vsHex[32];
                  std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                                static_cast<unsigned long long>(vsH));
                  Logger::info(str::format(
                    "[D3D11Rtx.camOri] #", sCamOrigLog,
                    " draw=", m_drawCallID,
                    " vs=", vsHex,
                    " cur=(", camX, ",", camY, ",", camZ, ")",
                    " prev=", havePrev ? "yes" : "no",
                    " prevPos=(", prevX, ",", prevY, ",", prevZ, ")",
                    " cb2base=", base, " bufSize=", sz));
                }
              }
              // NV-DXVK: different VS permutations store c_cameraOrigin with
              // different signs at cb2@4 (the fanout BSP VS sees +cam for
              // gameplay camera, while other VS permutations — shadow,
              // reflection, HUD — see NEGATED cam or their own pass-camera).
              // Using the per-draw cb2@4 produces inconsistent o2w.T across
              // draws in the same frame: BSP walls land at -cam, characters
              // at their own pass-cam, etc. Prefer the authoritative
              // m_lastFanoutCamOrigin (captured once per frame from the
              // gameplay BSP fanout VS). The per-draw cb2@4 is only used as
              // a last-resort fallback when fanout hasn't published yet.
              float useCamX = camX, useCamY = camY, useCamZ = camZ;
              bool camFromFanout = false;
              if (m_hasFanoutCamOrigin) {
                useCamX = m_lastFanoutCamOrigin.x;
                useCamY = m_lastFanoutCamOrigin.y;
                useCamZ = m_lastFanoutCamOrigin.z;
                camFromFanout = true;
              }
              if (std::isfinite(useCamX) && std::isfinite(useCamY) && std::isfinite(useCamZ)
                  && (std::abs(useCamX) + std::abs(useCamY) + std::abs(useCamZ)) > 1.0f) {
                transforms.objectToWorld = Matrix4(
                  Vector4(1.0f, 0.0f, 0.0f, 0.0f),
                  Vector4(0.0f, 1.0f, 0.0f, 0.0f),
                  Vector4(0.0f, 0.0f, 1.0f, 0.0f),
                  Vector4(useCamX, useCamY, useCamZ, 1.0f));
                found = true;
                m_lastO2wPathId = 8;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.cb2cam] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " T=(", useCamX, ",", useCamY, ",", useCamZ, ")",
                  " src=", camFromFanout ? "fanout" : "cb2@4"));
                m_lastDrawCamOrigin    = Vector3(useCamX, useCamY, useCamZ);
                m_lastDrawCamOriginSet = true;
                static uint32_t sCamFbLog = 0;
                if (sCamFbLog < 20) {
                  ++sCamFbLog;
                  Logger::info(str::format(
                    "[D3D11Rtx] o2w fallback (camOri): src=",
                    camFromFanout ? "fanout" : "cb2@4",
                    " cb2Cam=(", camX, ",", camY, ",", camZ, ")",
                    " use=(", useCamX, ",", useCamY, ",", useCamZ, ")"));
                }
              }
            }
          }
        }

        // Per-draw diagnostic — now logs first 8 hits PER FRAME rather than
        // per session, so in-game transforms are visible even after menu loading.
        static uint32_t s_f3x4LogFrame = UINT32_MAX;
        static uint32_t s_f3x4LogHitsThisFrame = 0;
        {
          static uint32_t s_f3x4PrevID = UINT32_MAX;
          if (m_drawCallID == 0 || m_drawCallID < s_f3x4PrevID) {
            s_f3x4LogFrame++;
            s_f3x4LogHitsThisFrame = 0;
          }
          s_f3x4PrevID = m_drawCallID;
        }
        if (found && s_f3x4LogHitsThisFrame < 8 && s_f3x4LogFrame < 600) {
          ++s_f3x4LogHitsThisFrame;
          const auto& m = transforms.objectToWorld;
          const bool isIdentity = isIdentityExact(transforms.objectToWorld);
          Logger::info(str::format(
            "[D3D11Rtx] objectToWorld slot=", kSourceModelSlot,
            " frame=", s_f3x4LogFrame, " draw=", m_drawCallID,
            isIdentity ? " IDENTITY" : "",
            " T=(", m[3][0], ",", m[3][1], ",", m[3][2], ")"
            " R=(", m[0][0], ",", m[1][1], ",", m[2][2], ")"));
        }
      }

      // Generic 4x4 scan — fallback for non-Source engines.
      // Skips VS slots 0 and 1 (materialsystem's per-draw/material cbuffers) to
      // avoid false positives from texture/viewport constants.
      if (!found) {
        // Try projSlot+1 first (common layout: proj in slot N, world in slot N+1).
        if (projSlot != UINT32_MAX && projStage == 0
            && projSlot + 1 < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
            && projSlot + 1 > 1)   // skip slots 0 and 1 (materialsystem)
          found = tryWorldCb(vsCbs, projSlot + 1, projStage, projSlot);

        if (!found) {
          for (uint32_t s = 2; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++s) {
            if (projStage == 0 && s == projSlot) continue;
            if (tryWorldCb(vsCbs, s, projStage, projSlot)) { found = true; break; }
          }
        }
        // Scan non-VS stages for emulator compatibility.
        if (!found) {
          for (int si = 1; si < kNumStages && !found; ++si) {
            for (uint32_t s = 0; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++s) {
              if (si == projStage && s == projSlot) continue;
              if (tryWorldCb(*stageCbs[si], s, projStage, projSlot)) { found = true; break; }
            }
          }
        }
        // Last resort: try float3x4 scan over all VS slots (covers engines that
        // put the model matrix in a non-slot-3 location).
        // NV-DXVK (Titanfall 2): start at slot 1, not 2 — TF2 shaders commonly
        // leave VS s0 null and put a 48-byte float3x4 world matrix in VS s1.
        // trySourceFloat3x4 rejects identity/view/proj so materialsystem data
        // in slot 1 on other engines won't produce false positives.
        if (!found) {
          for (uint32_t s = 1; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT && !found; ++s) {
            if (projStage == 0 && s == projSlot) continue;
            if (trySourceFloat3x4(vsCbs, s)) found = true;
          }
        }

        // === LOG WHEN objectToWorld STAYS IDENTITY (search failed) ===
        if (!found) {
          static uint32_t s_noWorldCount = 0;
          if (s_noWorldCount < 200) {
            ++s_noWorldCount;
            // Dump which VS slots are actually bound so we can see what slot has what.
            std::string slotInfo = "";
            for (uint32_t s = 0; s < 8; ++s) {
              const auto& cb = vsCbs[s];
              if (cb.buffer != nullptr) {
                slotInfo += str::format(" s", s, "=", cb.buffer->Desc()->ByteWidth, "@", cb.constantOffset);
              }
            }
            Logger::warn(str::format(
              "[D3D11Rtx] objectToWorld NOT FOUND (identity) drawID=", m_drawCallID,
              " projSlot=", projSlot, " projStage=", projStage,
              " boundVS:", slotInfo));
          }
        }
      }
    }

    // NV-DXVK: For bone draws, use the worldToView from a fmt=106 draw
    // (which has the correct camera). Reset objectToWorld to identity
    // since the interleaver applies the bone matrix GPU-side.
    if (m_currentDrawIsBoneTransformed) {
      transforms.objectToWorld = Matrix4();
      // Build worldToView from c_cameraOrigin (cb2@4) with explicit
      // Source Engine coordinate system mapping:
      //   Source: X=forward, Y=left, Z=up
      //   D3D view: X=right, Y=up, Z=forward
      // View rotation: D3D_right = Source_-Y, D3D_up = Source_Z, D3D_fwd = Source_X
      float camX = 0, camY = 0, camZ = 0;
      const auto& camCb = m_context->m_state.vs.constantBuffers[2];
      if (camCb.buffer != nullptr) {
        const auto mapped = camCb.buffer->GetMappedSlice();
        const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (p && camCb.buffer->Desc()->ByteWidth >= 16) {
          const float* co = reinterpret_cast<const float*>(p + 4);
          camX = co[0]; camY = co[1]; camZ = co[2];
        }
      }
      // Source: X=forward, Y=left, Z=up
      // Use the cached VP's camera direction (fwd from decomposition)
      // but fix the up axis which the VP decomposition negates (Y-flip).
      const Matrix4& cachedView = m_lastGoodTransforms.worldToView;
      // Extract axes from cached view, fix the Y-flipped up
      Vector3 right(cachedView[0][0], cachedView[0][1], cachedView[0][2]);
      Vector3 up   (cachedView[1][0], cachedView[1][1], cachedView[1][2]);
      Vector3 fwd  (cachedView[2][0], cachedView[2][1], cachedView[2][2]);
      // Fix Y-flip: if up.z is negative, the VP decomposition flipped it
      if (up.z < 0) {
        up.x = -up.x; up.y = -up.y; up.z = -up.z;
      }
      // Check if cached rotation is valid (non-zero)
      bool hasCachedRotation = (length(right) > 0.5f && length(fwd) > 0.5f);
      if (!hasCachedRotation) {
        // Fallback: hardcoded Source→D3D axis mapping (camera looking +X)
        right = Vector3(0, -1, 0);
        up    = Vector3(0,  0, 1);
        fwd   = Vector3(1,  0, 0);
      }
      const float tR = -(right.x*camX + right.y*camY + right.z*camZ);
      const float tU = -(up.x*camX    + up.y*camY    + up.z*camZ);
      const float tF = -(fwd.x*camX   + fwd.y*camY   + fwd.z*camZ);
      m_lastWtvPathId = 11; // cached VP + live camX/Y/Z reuse path
      // NV-DXVK: store by columns — see path 1 fix.
      transforms.worldToView = Matrix4(
        Vector4(right.x, up.x, fwd.x, 0),
        Vector4(right.y, up.y, fwd.y, 0),
        Vector4(right.z, up.z, fwd.z, 0),
        Vector4(tR,      tU,   tF,   1));
      static uint32_t sViewDiag = 0;
      if (sViewDiag < 5) {
        ++sViewDiag;
        Logger::info(str::format(
          "[D3D11Rtx] Bone view: cam=(", camX, ",", camY, ",", camZ, ")",
          " R=(", right.x, ",", right.y, ",", right.z, ")",
          " U=(", up.x, ",", up.y, ",", up.z, ")",
          " F=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
          " cached=", hasCachedRotation ? 1 : 0));
      }
    }

    transforms.objectToView = transforms.objectToWorld;
    if (!isIdentityExact(transforms.worldToView))
      transforms.objectToView = transforms.worldToView * transforms.objectToWorld;

    transforms.sanitize();

    // Log camera discovery once.
    static bool s_cameraLogged = false;
    if (projSlot != UINT32_MAX && !s_cameraLogged) {
      s_cameraLogged = true;
      const auto& p = transforms.viewToProjection;
      Logger::info(str::format(
        "[D3D11Rtx] Camera found: stage=", kStageNames[projStage],
        " slot=", projSlot, " off=", projOffset,
        " proj diag=(", p[0][0], ",", p[1][1], ",", p[2][2], ")",
        " m[2][3]=", p[2][3],
        m_columnMajor ? " [column-major]" : " [row-major]"));
    }

    // DEBUG: dump info for non-bone draws returning identity o2w.
    // Per-frame counter so we don't saturate on drawID=0. Skip fallback draws
    // because those get rejected downstream — we want the REAL submissions.
    if (!m_currentDrawIsBoneTransformed && isIdentityExact(transforms.objectToWorld)
        && !m_lastExtractUsedFallback) {
      static uint32_t s_silentFrame = UINT32_MAX;
      static uint32_t s_silentPerFrame = 0;
      static uint32_t s_silentPrevID = UINT32_MAX;
      if (m_drawCallID == 0 || m_drawCallID < s_silentPrevID) {
        s_silentFrame++;
        s_silentPerFrame = 0;
      }
      s_silentPrevID = m_drawCallID;
      if (s_silentFrame < 3 && s_silentPerFrame < 3) {
        ++s_silentPerFrame;
        const auto& vsCbs = m_context->m_state.vs.constantBuffers;
        std::string vsHashStr = "?";
        auto vsShader = m_context->m_state.vs.shader;
        if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
          auto& shader = vsShader->GetCommonShader()->GetShader();
          if (shader != nullptr)
            vsHashStr = shader->getShaderKey().toString();
        }
        Logger::warn(str::format("[D3D11Rtx] === SILENT-ID drawID=", m_drawCallID, " VS=", vsHashStr, " cbuffer dump ==="));
        for (uint32_t s = 0; s < 8; ++s) {
          const auto& cb = vsCbs[s];
          if (cb.buffer == nullptr) continue;
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (!p) continue;
          const size_t base = static_cast<size_t>(cb.constantOffset) * 16;
          const size_t sz = cb.buffer->Desc()->ByteWidth;
          const size_t n = std::min<size_t>(sz - base, 256);
          const float* f = reinterpret_cast<const float*>(p + base);
          // Dump as float4 rows
          for (size_t r = 0; r * 16 + 16 <= n; ++r) {
            Logger::warn(str::format(
              "  VS s", s, " [", r, "] = (", f[r*4+0], ", ", f[r*4+1], ", ", f[r*4+2], ", ", f[r*4+3], ")"));
          }
        }
      }
    }

    // ================================================================
    // NV-DXVK: CLASSIFIER-DRIVEN OVERRIDE (V2 dispatcher, Phase 1)
    // ================================================================
    // Runs after the legacy path-selection tangle. For kinds the classifier
    // is confident about, we overwrite `transforms.objectToWorld` with a
    // deterministic RDEF-sourced value. Kinds whose legacy path already
    // works (InstancedBsp, SkinnedChar) keep whatever the legacy code set.
    //
    // This block is the single source of truth for:
    //   StaticWorld  → cb3.CBufModelInstance.objectToCameraRelative
    //                  + cb2.CBufCommonPerCamera.c_cameraOrigin (to shift to
    //                  absolute world). PIX-verified on VS_6e3e6f28f2156ea2.
    //   UI/Unknown   → mark fallback so SubmitDraw filters as UIFallback.
    //
    // Guarantee: once this block runs, no downstream code should modify
    // objectToWorld. If a legacy path set a different o2w earlier in this
    // function, the override replaces it.
    {
      auto vsPtrV2 = m_context->m_state.vs.shader;
      const D3D11CommonShader* commonV2 =
        (vsPtrV2 != nullptr) ? vsPtrV2->GetCommonShader() : nullptr;
      const auto* ilV2 = m_context->m_state.ia.inputLayout.ptr();
      const std::vector<D3D11RtxSemantic> kEmptySemsV2;
      const auto& semsV2 = ilV2 ? ilV2->GetRtxSemantics() : kEmptySemsV2;
      const auto clsV2 = D3D11VsClassifier::classify(commonV2, semsV2);

      auto readCbFloats = [&](uint32_t slot, uint32_t byteOff, uint32_t nBytes,
                              float* out) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = m_context->m_state.vs.constantBuffers[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + byteOff;
        if (base + nBytes > cb.buffer->Desc()->ByteWidth) return false;
        std::memcpy(out, ptr + base, nBytes);
        for (uint32_t i = 0; i < nBytes / 4; ++i)
          if (!std::isfinite(out[i])) return false;
        return true;
      };

      // Helper: detect "zero" matrix/vec3 (all components exactly 0.f).
      auto allZero = [](const float* p, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) if (p[i] != 0.f) return false;
        return true;
      };

      switch (clsV2.kind) {
        case D3D11VsClassification::Kind::StaticWorld: {
          // V2 StaticWorld scope: UI-demotion only.
          // Read cb3 + camO purely to detect the menu/pre-gameplay case
          // where shader declares CBufModelInstance but contents are
          // uninitialized. Do NOT override legacy objectToWorld. The
          // legacy cb3-RDEF path (o2wPath=5) already produces the right
          // o2w for real gameplay draws (PIX-verified on VS_6e3e6f28);
          // overriding it here submits draws that RT can't render into
          // usable pixels, which flipped m_remixActiveThisFrame=true and
          // caused the RT blit to clobber everything the native raster
          // had drawn.
          float m[12];
          if (readCbFloats(clsV2.cb3Slot, /*offset*/ 0, /*size*/ 48, m)) {
            // Read camera origin from CBufCommonPerCamera.c_cameraOrigin.
            // Field offset is RDEF-declared; if the shader doesn't expose it
            // explicitly, fall back to the canonical cb2 offset-4 location
            // used by every TF2 Source 2 shader observed so far.
            float camO[3] = { 0.f, 0.f, 0.f };
            bool haveCam = false;
            if (commonV2 != nullptr) {
              auto camLoc = commonV2->FindCBField(
                  "CBufCommonPerCamera", "c_cameraOrigin");
              if (camLoc && camLoc->size >= 12) {
                haveCam = readCbFloats(camLoc->slot, camLoc->offset, 12, camO);
              }
            }
            if (!haveCam) {
              haveCam = readCbFloats(/*cb2*/ 2, /*byteOff*/ 4, 12, camO);
            }
            // NV-DXVK: Align with the legacy o2w path at ~4148: treat a
            // successful read of ALL-ZERO c_cameraOrigin the same as a
            // failed read. Source's BSP world pass binds cb2 with zeros
            // in the c_cameraOrigin slot for draws where the camera is
            // expected to come from cb3 (per-model) or from the fanout
            // capture. Without this, haveCam stays true on the all-zero
            // read, the fanout fallback below is skipped, and the draw
            // gets demoted as camO_all_zero_no_real_camera → filtered as
            // UIFallback → camera data never reaches the SceneManager →
            // CamMgr never latches Main → camValid=0 forever → black
            // screen. Logged once per VS hash so we can see when this
            // hits vs. fanout-fallback hits vs. legitimately-no-camera.
            const bool camReadAllZero = haveCam
                && camO[0] == 0.f && camO[1] == 0.f && camO[2] == 0.f;
            if (camReadAllZero) {
              static std::unordered_set<std::string> sV2LogZeroCb2;
              const std::string vk = getVsHashShort();
              if (sV2LogZeroCb2.insert(vk).second) {
                Logger::info(str::format(
                  "[VsClass.v2.StaticWorld.zeroCamO] vs=", vk,
                  " drawID=", m_drawCallID,
                  " cb2/c_cameraOrigin read OK but all-zero — will try"
                  " fanout fallback (hasFanout=", m_hasFanoutCamOrigin ? 1 : 0, ")"));
              }
              haveCam = false;
            }
            // Fallback to the BSP-fanout captured camera if the earlier
            // sources either failed or yielded a degenerate all-zero
            // origin.  Fanout runs during the BSP world pass and caches
            // the real player-eye origin, which is what every downstream
            // draw in that frame should be using.
            if (!haveCam && m_hasFanoutCamOrigin) {
              camO[0] = m_lastFanoutCamOrigin.x;
              camO[1] = m_lastFanoutCamOrigin.y;
              camO[2] = m_lastFanoutCamOrigin.z;
              haveCam = true;
              static std::unordered_set<std::string> sV2LogFanout;
              const std::string vk = getVsHashShort();
              if (sV2LogFanout.insert(vk).second) {
                Logger::info(str::format(
                  "[VsClass.v2.StaticWorld.fanoutFallback] vs=", vk,
                  " using fanout cam=(", camO[0], ",", camO[1], ",", camO[2], ")"));
              }
            }
            // Menu / pre-gameplay guard.
            // TF2's UI and menu shaders re-use the StaticWorld shader
            // template (same CBufModelInstance cbuffer declared in RDEF)
            // but the cbuffer contents are never written during menu
            // frames — cb3 holds identity or zeros, and the gameplay
            // camera origin hasn't been captured yet (camO all zero).
            // If we commit these to RT the BLASes pile up at world
            // origin, render nothing, and flip m_remixActiveThisFrame=
            // true so the RT blit runs and clobbers the native-raster
            // output the UI/menu buttons were drawn into.
            //
            // Signal: cameraOrigin == 0. No valid world placement is
            // possible without a real camera. Demote to UI (native
            // raster renders it, RT stays out of this frame).
            if (allZero(camO, 3)) {
              m_lastExtractUsedFallback = true;
              m_lastClassifierSaidUi    = true;
              static std::unordered_set<std::string> sV2LogUiDemote;
              const std::string vk = getVsHashShort();
              if (sV2LogUiDemote.insert(vk).second) {
                Logger::info(str::format(
                  "[VsClass.v2.StaticWorld.demoteUI] vs=", vk,
                  " reason=camO_all_zero_no_real_camera"));
              }
              break;
            }
            // Clear the fallback flag ONLY when THIS draw's per-draw
            // worldToView already has a real translation. If the per-draw
            // w2v is identity (common — most BSP draws don't re-extract
            // projection), we DELIBERATELY leave the flag set so that
            // SubmitDraw's path 4 runs and overrides the per-draw w2v with
            // the frame's cached m_lastGoodTransforms.worldToView. That
            // gives the draw a real camera from a prior extraction this
            // frame. If cached is also identity, path 4's own degenerate
            // check rejects as UIFallback (correct — no camera to render
            // from yet). Previously (unconditional clear), draws with
            // identity per-draw w2v skipped path 4 and submitted with
            // camera-at-origin, producing huge BSP meshes piled at world
            // origin → BLAS catastrophe → GPU hang.
            const bool w2vHasRealTranslation =
                 std::abs(transforms.worldToView[3][0]) > 0.01f
              || std::abs(transforms.worldToView[3][1]) > 0.01f
              || std::abs(transforms.worldToView[3][2]) > 0.01f;
            if (haveCam && w2vHasRealTranslation) {
              m_lastExtractUsedFallback = false;
            }
            static std::unordered_set<std::string> sV2LogStatic;
            const std::string vk = getVsHashShort();
            if (sV2LogStatic.insert(vk).second) {
              Logger::info(str::format(
                "[VsClass.v2.StaticWorld.pass] vs=", vk,
                " cb3Slot=", clsV2.cb3Slot,
                " haveCam=", haveCam ? 1 : 0,
                " m[3,7,11]=(", m[3], ",", m[7], ",", m[11], ")",
                " camO=(", camO[0], ",", camO[1], ",", camO[2], ")"));
            }
          }
          break;
        }
        case D3D11VsClassification::Kind::InstancedBsp:
        case D3D11VsClassification::Kind::SkinnedChar: {
          // The classifier has definitively identified these as real
          // rendering geometry (InstancedBsp = per-instance t31; SkinnedChar
          // = BLENDINDICES bone palette). Neither is UI. The legacy o2w
          // extraction for these kinds is left intact, but we force the
          // fallback flag false so SubmitDraw's UIFallback filter does not
          // accidentally reject them when a legacy sub-branch hit a
          // per-draw edge case (e.g. t31 entry out of range, cached bone
          // slice null) and wrote `m_lastExtractUsedFallback = true` at
          // line ~2806. If legacy produced a bad o2w, the finiteness /
          // magnitude guard in SubmitDraw (line ~5340) still rejects it;
          // we're only undoing the UIFallback class of rejection.
          m_lastExtractUsedFallback = false;
          static std::unordered_set<std::string> sV2LogRecognized;
          const std::string vk = getVsHashShort();
          const std::string key =
              std::string(D3D11VsClassifier::kindName(clsV2.kind)) + "|" + vk;
          if (sV2LogRecognized.insert(key).second) {
            Logger::info(str::format(
              "[VsClass.v2.", D3D11VsClassifier::kindName(clsV2.kind), "] vs=", vk,
              " fallback_cleared legacy_o2wPath=", m_lastO2wPathId));
          }
          break;
        }
        case D3D11VsClassification::Kind::UI:
        case D3D11VsClassification::Kind::Unknown:
          // No recognized transform signals. Force UIFallback so the native
          // raster path handles the draw and RTX skips it. Setting
          // m_lastClassifierSaidUi = true forces SubmitDraw into the TRUE
          // UI branch (line ~5880) regardless of m_foundRealProjThisFrame,
          // which is what makes the menu buttons/HUD reach native raster
          // after any gameplay draw has latched a real projection.
          m_lastExtractUsedFallback = true;
          m_lastClassifierSaidUi    = true;
          break;
        default:
          // Skybox/Viewmodel/Particle/Sprite2D — not produced by the
          // classifier yet. Fall through silently.
          break;
      }
    }

    // NV-DXVK: detect packed-uint TEXCOORD encoding from the bound VS's ISGN.
    // The VS bytecode declares TEXCOORD0 as `uint xy` for shaders that pack
    // a uint-packed UV into a R32G32_FLOAT VB stream and bit-decode it in
    // the VS body (TF2 BSP world VSes — verified via fxc /dumpbin on
    // VS_e7abcf4ea24b0fa7 and VS_1953b6e9cc252e4e). When the BLAS path
    // reads those bytes as plain f32 it gets garbage UVs in the hundreds,
    // producing per-pixel gradients > 1 → mip 8+ → 1×1 sample → flat walls.
    // Promotes the encoding to RtSurface so surface_interaction.slangh can
    // apply the matching decode after the raw fetch.
    {
      const D3D11VertexShader* vsPtr = m_context->m_state.vs.shader.ptr();
      if (vsPtr != nullptr) {
        const D3D11CommonShader* cs = vsPtr->GetCommonShader();
        if (cs != nullptr) {
          const auto ct = cs->GetInputSemanticComponentType("TEXCOORD", 0);
          if (ct == D3D11CommonShader::InputCompType_Uint
           || ct == D3D11CommonShader::InputCompType_Sint) {
            transforms.texcoordEncoding = RtSurface::TexcoordEncoding::TF2BspUintPacked;
          }

          // NV-DXVK: diagnostic — per-VS, log whether ISGN reports
          // TEXCOORD0 as float / uint / sint / unknown, and what
          // texcoordEncoding value we resolved. Throttled to one log
          // line per unique VS hash so it doesn't spam.
          //
          // Goal: confirm or deny that VS=7c38fdf4 (the float pass-
          // through wall family) gets ct=Float and encoding=Float (0).
          // If we see VS=7c38fdf4 with encoding=TF2BspUintPacked (1)
          // here, the ISGN parser is misclassifying — the slang side
          // would then incorrectly apply the e7abcf4e uint decode to
          // genuine float UVs, which matches the order-of-magnitude
          // -7800 vs source-VB 0.4 we observed in UVdecode logs.
          {
            XXH64_hash_t vsH_enc = 0, psH_enc = 0;
            GetCurrentVsPsHashes(vsH_enc, psH_enc);
            static std::unordered_set<uint64_t> sLoggedEncVs;
            static std::mutex sLoggedEncMu;
            bool firstSeen = false;
            {
              std::lock_guard<std::mutex> lk(sLoggedEncMu);
              firstSeen = sLoggedEncVs.insert(uint64_t(vsH_enc)).second;
            }
            if (firstSeen && vsH_enc != 0) {
              const char* ctStr =
                  (ct == D3D11CommonShader::InputCompType_Uint)    ? "Uint"
                : (ct == D3D11CommonShader::InputCompType_Sint)    ? "Sint"
                : (ct == D3D11CommonShader::InputCompType_Float)   ? "Float"
                :                                                    "Unknown";
              const uint32_t encVal = uint32_t(transforms.texcoordEncoding);
              const char* encStr = (encVal == 0u) ? "Float"
                                 : (encVal == 1u) ? "TF2BspUintPacked"
                                 :                  "?";
              Logger::info(str::format(
                "[TexcoordEnc] VS=0x", std::hex, vsH_enc, std::dec,
                " isgnCompType=", ctStr,
                " resolvedEncoding=", encStr,
                " (expectFloat=", (ct == D3D11CommonShader::InputCompType_Float ? 1 : 0), ")"));
            }
          }
        }
      }
    }

    return transforms;
  }

  // NV-DXVK: latch set in EndFrame once real gameplay starts — drives the
  // per-draw Submit log so it prints during actual scene rendering, not boot.
  static uint32_t s_GameplayLogFrames = 0;

  // NV-DXVK: helper — bump m_filterCounts AND record the reject against the
  // current VS so EndFrame can show per-shader outcomes.
  void D3D11Rtx::BumpFilter(FilterReason r) {
    const uint32_t ri = static_cast<uint32_t>(r);
    ++m_filterCounts[ri];
    if (!m_currentVsHashCache.empty()) {
      auto& st = m_vsFrameStats[m_currentVsHashCache];
      ++st.rejects[ri];
    }
    // NV-DXVK [VMHunt.result=reject]: if the rejected draw is a suspect
    // viewmodel draw, log the reason. Otherwise we don't know if a suspect
    // made it through or got filtered.
    if (m_vmHuntIsSuspect) {
      static const char* kReason[] = {
        "Throttle","NonTri","NoPS","NoRTV","CountSmall","FsQuad","NoLayout",
        "NoSem","NoPos","Pos2D","NoPosBuf","NoIdxBuf","HashFail",
        "UIFallback","UnsupPosFmt","CharDepthPrepass"
      };
      const char* reasonStr = (ri < std::size(kReason)) ? kReason[ri] : "?";
      Logger::info(str::format(
        "[VMHunt.result] count=", m_vmHuntIndexCount,
        " vs=", m_currentVsHashCache.substr(0, 19),
        " verdict=REJECT reason=", reasonStr));
      m_vmHuntIsSuspect = false; // consumed
    }
    // NV-DXVK [Reject]: log every rejected draw with semantic fingerprint +
    // VS + PS + reason + vert count + vpMaxZ + vpMinZ. Tagged `sk=1` if the
    // draw is skinned (per-vertex BLENDINDICES). The viewmodel (gun + hands
    // in first-person) is typically a small-mesh skinned draw (hands) +
    // small rigid draw (weapon parts). It may use an unusual position or
    // BLENDINDICES format that we currently don't recognise, so we log ALL
    // rejects — not just skinned — to surface it. Throttled per frame.
    {
      // NV-DXVK: this path logs up to 128 rejects/frame → ~7700 lines/sec
      // at 60fps, which wedges the main menu (Source emits thousands of
      // per-frame UI/skinned-geom reject candidates). Gate behind
      // RTX_REJECT_LOG=1 so it's silent in normal runs; also dedupe on
      // (vs,ps,reason) so repeated identical rejects log once per process.
      static const bool kRejectLogEnabled = []() {
        const char* e = std::getenv("RTX_REJECT_LOG");
        return e && e[0] == '1';
      }();
      const uint32_t fid = m_context->m_device->getCurrentFrameId();
      static uint32_t sLastFrameRS = 0;
      static uint32_t sCountThisFrameRS = 0;
      if (fid != sLastFrameRS) { sLastFrameRS = fid; sCountThisFrameRS = 0; }
      if (kRejectLogEnabled && sCountThisFrameRS < 128) {
        ++sCountThisFrameRS;
        D3D11InputLayout* layout = m_context->m_state.ia.inputLayout.ptr();
        bool isSkinned = false;
        uint32_t posFmt = 0;
        uint32_t biFmt = 0;
        if (layout != nullptr) {
          for (const auto& s : layout->GetRtxSemantics()) {
            if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0) {
              isSkinned = true;
              biFmt = (uint32_t)s.format;
            }
            if (!s.perInstance && std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0) {
              posFmt = (uint32_t)s.format;
            }
          }
        }
        // Viewport min/max depth (the ViewModel classifier uses maxZ).
        float vpMaxZ = -1.0f, vpMinZ = -1.0f;
        const auto& vps = m_context->m_state.rs.viewports;
        if (m_context->m_state.rs.numViewports > 0) {
          vpMaxZ = vps[0].MaxDepth;
          vpMinZ = vps[0].MinDepth;
        }
        // PS hash (null if depth-only).
        std::string psName = "null";
        auto psPtr = m_context->m_state.ps.shader;
        if (psPtr != nullptr && psPtr->GetCommonShader() != nullptr) {
          auto& s = psPtr->GetCommonShader()->GetShader();
          if (s != nullptr) psName = s->getShaderKey().toString().substr(0, 19);
        }
        static const char* kReasonShort[] = {
          "Throttle","NonTri","NoPS","NoRTV","CountSmall","FsQuad","NoLayout",
          "NoSem","NoPos","Pos2D","NoPosBuf","NoIdxBuf","HashFail",
          "UIFallback","UnsupPosFmt"
        };
        const char* reasonStr = (ri < std::size(kReasonShort)) ? kReasonShort[ri] : "?";
        Logger::info(str::format(
          "[Reject] f=", fid,
          " vs=", m_currentVsHashCache.substr(0, 19),
          " ps=", psName,
          " reason=", reasonStr,
          " sk=", (isSkinned ? 1 : 0),
          " posFmt=", posFmt,
          " biFmt=", biFmt,
          " vpZ=[", vpMinZ, ",", vpMaxZ, "]"));
      }
    }
  }

  Future<GeometryHashes> D3D11Rtx::ComputeGeometryHashes(
      const RasterGeometry& geo, uint32_t vertexCount,
      uint32_t hashStartVertex, uint32_t hashVertexCount) const {

    const void* posData = geo.positionBuffer.mapPtr(geo.positionBuffer.offsetFromSlice());
    const void* tcData  = geo.texcoordBuffer.defined()
                        ? geo.texcoordBuffer.mapPtr(geo.texcoordBuffer.offsetFromSlice())
                        : nullptr;
    const void* idxData = geo.indexBuffer.defined() ? geo.indexBuffer.mapPtr(0) : nullptr;

    // D3D11 dynamic buffers can be discarded (Map WRITE_DISCARD) at any time,
    // which recycles the physical slice backing our raw pointers.  Pin each
    // buffer with incRef + acquire(Read) so the allocator won't reuse the
    // memory while the hash worker is reading it.  The lambda releases them.
    DxvkBuffer* posBuf = geo.positionBuffer.buffer().ptr();
    DxvkBuffer* tcBuf  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.buffer().ptr() : nullptr;
    DxvkBuffer* idxBuf = geo.indexBuffer.defined()    ? geo.indexBuffer.buffer().ptr()    : nullptr;

    if (posBuf) { posBuf->incRef(); posBuf->acquire(DxvkAccess::Read); }
    if (tcBuf)  { tcBuf->incRef();  tcBuf->acquire(DxvkAccess::Read);  }
    if (idxBuf) { idxBuf->incRef(); idxBuf->acquire(DxvkAccess::Read); }

    const uint32_t posStride = geo.positionBuffer.stride();
    const uint32_t tcStride  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.stride() : 0u;
    const uint32_t idxStride = geo.indexBuffer.defined()    ? geo.indexBuffer.stride()    : 0u;
    const uint32_t indexType = static_cast<uint32_t>(geo.indexBuffer.indexType());
    const uint32_t topology  = static_cast<uint32_t>(geo.topology);

    const uint32_t posOffset = geo.positionBuffer.offsetFromSlice();

    XXH64_hash_t descHash   = hashGeometryDescriptor(geo.indexCount, vertexCount, indexType, topology);
    // NV-DXVK: Mix bone instance index into hash so each instance gets a unique BLAS
    if (geo.boneInstanceIndex != 0) {
      const uint32_t bi = geo.boneInstanceIndex;
      descHash = XXH3_64bits_withSeed(&bi, sizeof(bi), descHash);
    }
    const XXH64_hash_t layoutHash = hashVertexLayout(geo);

    // Compute the safe byte range available for position and texcoord data.
    // Buffer pins guarantee the memory won't be recycled, but we must still
    // clamp to the actual buffer extent to avoid reading past the allocation.
    const size_t posLength = geo.positionBuffer.length();
    const size_t tcLength  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.length() : 0;
    const size_t idxLength = geo.indexBuffer.defined()    ? geo.indexBuffer.length()    : 0;

    auto future = m_pGeometryWorkers->Schedule([posData, tcData, idxData,
                                         posBuf, tcBuf, idxBuf,
                                         posStride, tcStride, idxStride,
                                         posLength, tcLength, idxLength,
                                         vertexCount, indexCount = geo.indexCount,
                                         posOffset,
                                         hashStartVertex, hashVertexCount,
                                         descHash, layoutHash]() -> GeometryHashes {
      GeometryHashes hashes;
      hashes[HashComponents::GeometryDescriptor] = descHash;
      hashes[HashComponents::VertexLayout]       = layoutHash;

      if (posData && posStride > 0) {
        // Hash only the drawn subrange [hashStartVertex, hashStartVertex + hashVertexCount).
        // Clamp to actual buffer length to prevent OOB reads on shared/dynamic VBs.
        const size_t startByte = static_cast<size_t>(hashStartVertex) * posStride;
        size_t posBytes = static_cast<size_t>(hashVertexCount) * posStride;
        if (startByte >= posLength) {
          posBytes = 0;
        } else if (startByte + posBytes > posLength) {
          posBytes = posLength - startByte;
        }
        if (posBytes > 0) {
          const auto* posBase = static_cast<const uint8_t*>(posData) + startByte;
          hashes[HashComponents::VertexPosition] =
            XXH3_64bits_withSeed(posBase, posBytes, static_cast<XXH64_hash_t>(hashStartVertex));
        } else {
          hashes[HashComponents::VertexPosition] =
            XXH3_64bits(&posOffset, sizeof(posOffset));
        }

        if (tcData && tcStride > 0) {
          const size_t tcStartByte = static_cast<size_t>(hashStartVertex) * tcStride;
          size_t tcBytes = static_cast<size_t>(hashVertexCount) * tcStride;
          if (tcStartByte >= tcLength) {
            tcBytes = 0;
          } else if (tcStartByte + tcBytes > tcLength) {
            tcBytes = tcLength - tcStartByte;
          }
          if (tcBytes > 0) {
            const auto* tcBase = static_cast<const uint8_t*>(tcData) + tcStartByte;
            hashes[HashComponents::VertexTexcoord] =
              XXH3_64bits_withSeed(tcBase, tcBytes, static_cast<XXH64_hash_t>(hashStartVertex));
          }
        }
        if (idxData && idxStride > 0) {
          const size_t idxBytes = static_cast<size_t>(indexCount) * idxStride;
          hashes[HashComponents::Indices] =
            hashContiguousMemory(idxData, std::min(idxBytes, idxLength));
        }
      } else {
        // GPU-only buffer: stable identity hash from buffer address and offset.
        XXH64_hash_t posHash = XXH3_64bits(&posBuf, sizeof(posBuf));
        posHash = XXH3_64bits_withSeed(&posOffset, sizeof(posOffset), posHash);
        hashes[HashComponents::VertexPosition] = posHash;
      }

      hashes.precombine();

      // Release buffer pins — allow slice recycling again.
      if (posBuf) { posBuf->release(DxvkAccess::Read); posBuf->decRef(); }
      if (tcBuf)  { tcBuf->release(DxvkAccess::Read);  tcBuf->decRef();  }
      if (idxBuf) { idxBuf->release(DxvkAccess::Read); idxBuf->decRef(); }

      return hashes;
    });

    // If the worker queue was full, the lambda never runs — release pins now
    // to prevent a VRAM leak (incRef/acquire above would never be undone).
    if (!future.valid()) {
      if (posBuf) { posBuf->release(DxvkAccess::Read); posBuf->decRef(); }
      if (tcBuf)  { tcBuf->release(DxvkAccess::Read);  tcBuf->decRef();  }
      if (idxBuf) { idxBuf->release(DxvkAccess::Read); idxBuf->decRef(); }
    }

    return future;
  }

  void D3D11Rtx::FillMaterialData(LegacyMaterialData& mat) const {
    const auto& ps = m_context->m_state.ps;
    uint32_t textureID = 0;

    // NV-DXVK: expanded diagnostic — gated on gameplay frames (raw>50 matches
    // the "first gameplay frame" threshold used in endFrame) so boot-time
    // menu draws don't consume the budget. Also logs VS/PS hash + counts of
    // SRVs rejected before candidate scoring, so we can tell the difference
    // between "game bound zero real textures" and "fork's filter rejected them".
    static uint32_t s_logCount = 0;
    const bool gameplayReady = (m_rawDrawCount > 50);
    const bool doLog = gameplayReady && (s_logCount < 500);
    std::string logMsg;
    uint32_t rejNonTex2D = 0, rejTiny = 0, rejNullView = 0;

    // NV-DXVK: PS-RDEF-driven PBR classification. The fork already parses
    // each shader's resource-definition chunk at creation (see
    // D3D11CommonShader::parseRdef). We ask the PS what names it assigns
    // to its SRV slots and route the matching bound textures into the
    // OpaqueMaterialData channels — no format heuristics required when
    // the game names its textures. Falls through to the scoring path below
    // for any slot that has no classification (UI, post-fx, debug draws).
    auto classifyFromRdef = [&]() -> bool {
      const auto* psShader = ps.shader.ptr();
      if (!psShader) return false;
      const auto* cs = psShader->GetCommonShader();
      if (!cs) return false;

      // Each entry: first matching name wins; list the most common alias
      // first to keep the hot path cheap. Source/Respawn engines prefer
      // the "*Texture" suffix (TF2 verified via fxc /dumpbin); add other
      // engines' conventions as needed.
      struct Role { const char* const names[8]; TextureRef* dst; Rc<DxvkSampler>* sampDst; uint32_t* slotDst; };
      TextureRef* albedoDst = &mat.colorTextures[0];
      uint32_t*   albedoSlot = &mat.colorTextureSlot[0];
      Rc<DxvkSampler>* albedoSamp = &mat.samplers[0];

      // NV-DXVK: TF2 shader naming — verified via fxc /dumpbin on
      // FS_7a6e4c57* (character), FS_1958793ac8a24933 (sprite card), and
      // FS_8dab3ea4d706d6a9 (refract). Source engine classic uses
      // "$basetexture"/"BaseTexture"; TF2's PBR-style shaders use
      // "albedoTexture"/"normalTexture"/"glossTexture"/"specTexture"/
      // "emissiveTexture". Sprite-card shaders and decals use
      // "BaseTexture"/"OpacityTexture". Refract uses "NormalTexture0".
      const Role roles[] = {
        { { "albedoTexture","albedoMap","diffuseTexture","diffuseMap","baseColorTexture","BaseTexture","$basetexture","fontTexture" }, albedoDst, albedoSamp, albedoSlot },
        // NV-DXVK: TF2 worldspace VGUI t1 — `materialTexture` is the icon /
        // image atlas the VGUI PS samples in mode 1 (`g_imgBounds[v2.x]`
        // remap into atlas UV). Routed to colorTextures[1] so it propagates
        // to OpaqueSurfaceMaterial::secondaryTextureIndex (the iris-texture
        // slot, which VGUI surfaces never use because they're disjoint
        // from eye materials). The slang VGUI evaluator reads it via that
        // index. Mode 0 (text) still uses albedoTexture (= fontTexture =
        // SDF font atlas).
        { { "materialTexture", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }, &mat.colorTextures[1], &mat.samplers[1], &mat.colorTextureSlot[1] },
        { { "normalTexture","normalTexture0","normalMap","bumpTexture","bumpMap","NormalTexture0","$bumpmap","$normalmap" }, &mat.normalTexture, &mat.normalSampler, nullptr },
        { { "glossTexture","glossMap","roughnessTexture","roughnessMap","GlossTexture","$phongexponenttexture", nullptr, nullptr }, &mat.roughnessTexture, &mat.roughnessSampler, nullptr },
        { { "specTexture","specMap","metallicTexture","metallicMap","SpecTexture","$envmapmask","$specmap", nullptr }, &mat.metallicTexture, &mat.metallicSampler, nullptr },
        { { "emissiveTexture","emissiveMap","selfIllumTexture","EmissiveTexture","$selfillummask", nullptr, nullptr, nullptr }, &mat.emissiveTexture, &mat.emissiveSampler, nullptr },
        // NV-DXVK: cavity / baked-AO map — grayscale multiplier applied to
        // albedo in opaque_surface_material_interaction.slangh. Covers TF2's
        // cavityTexture (FS_ac8c6ae6 at t12) and other engines' AO slot
        // conventions. Independent of normalTexture — both may coexist.
        { { "cavityTexture","cavityMap","aoTexture","ambientOcclusionTexture","occlusionTexture","$ambientoccltexture", nullptr, nullptr }, &mat.ambientOcclusionTexture, &mat.ambientOcclusionSampler, nullptr },
        // NV-DXVK: TF2 auxiliary BSP textures. lightmap{0,1} = baked static
        // GI (HDR range split across two textures); detailTexture = fine
        // brick/panel variation; cloudMaskTexture = large-scale tonal /
        // shadow banding. Each is optional; classifier only binds when
        // the PS RDEF names the slot.
        { { "lightmapTexture0","lightmapTexture","lightmap_texture","$lightmap", nullptr, nullptr, nullptr, nullptr }, &mat.lightmapTexture, &mat.lightmapSampler, nullptr },
        { { "lightmapTexture1","lightmap2Texture","lightmapBumpTexture", nullptr, nullptr, nullptr, nullptr, nullptr }, &mat.lightmap2Texture, &mat.lightmap2Sampler, nullptr },
        { { "detailTexture","detailMap","$detail","DetailTexture", nullptr, nullptr, nullptr, nullptr }, &mat.detailTexture, &mat.detailSampler, nullptr },
        { { "cloudMaskTexture","cloudMask","cloudShadowTexture","CloudMaskTexture", nullptr, nullptr, nullptr, nullptr }, &mat.cloudMaskTexture, &mat.cloudMaskSampler, nullptr },
      };

      bool anyAssigned = false;
      for (const Role& r : roles) {
        uint32_t slot = UINT32_MAX;
        for (const char* name : r.names) {
          if (!name) break;
          uint32_t s = cs->FindResourceSlot(name);
          if (s != UINT32_MAX) { slot = s; break; }
        }
        if (slot == UINT32_MAX) continue;
        if (slot >= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) continue;

        D3D11ShaderResourceView* srv = ps.shaderResources.views[slot].ptr();
        if (!srv || srv->GetResourceType() != D3D11_RESOURCE_DIMENSION_TEXTURE2D) continue;
        Rc<DxvkImageView> view = srv->GetImageView();
        if (view == nullptr) continue;

        // NV-DXVK: D3D11 runtime textures never have their hash set (only
        // the fork's white-texture stub calls DxvkImage::setHash). A zero
        // hash causes every LegacyMaterial to collide in
        // m_preCreationSurfaceMaterialMap, so all draws dedupe into one
        // cached surface material — producing a "flat single colour on
        // every wall" artifact. Stamp a stable per-image hash from the
        // Vulkan image handle + extent + format, so material dedup is
        // per-texture instead of collapsing everything to the first draw.
        DxvkImage* img = view->image().ptr();
        if (img && img->getHash() == 0) {
          struct ImgHashKey {
            uint64_t handle;
            uint32_t width, height, depth;
            uint32_t format;
            uint32_t mipLevels;
          };
          const auto& info = img->info();
          ImgHashKey k = {
            reinterpret_cast<uint64_t>(img),
            info.extent.width, info.extent.height, info.extent.depth,
            uint32_t(info.format),
            info.mipLevels,
          };
          img->setHash(XXH3_64bits(&k, sizeof(k)));
        }

        *r.dst = TextureRef(view);
        // NV-DXVK: pick the correct sampler for this SRV. D3D11 decouples
        // texture slots from sampler slots — e.g. TF2's character PS has
        // shadowmapSampler at s0 (compare) and trilinearSampler at s1, with
        // albedoTexture at t0 but Sample() calls use s1. Naively using
        // ps.samplers[albedoSlot] lands on the compare sampler and sampling
        // returns garbage. Heuristic: if the same-slot sampler exists, check
        // whether the PS declares a "normal" sampler by name and prefer it;
        // fall back to same-slot sampler otherwise.
        Rc<DxvkSampler> pickedSamp;
        const char* pickedFromName = nullptr;       // points into kPreferredSamplers (static literal) OR pickedNameStorage
        std::string pickedNameStorage;              // owns name string when picked from RDEF iteration
        uint32_t pickedFromSlot = UINT32_MAX;
        int pickStage = -1; // 0=wrap-aware-REPEAT, 1=preferred-name, 2=same-slot, 3=default
        {
          // 0) NV-DXVK: wrap-aware pick. Iterate every named sampler the PS
          // declares in its RDEF, pick the first one whose U/V are REPEAT.
          // Reason: TF2's `trilinearSampler` is named "trilinear" but is bound
          // with CLAMP_TO_EDGE wrap (verified via [D3D11Rtx.PsSamplers] dump
          // for PS=0xac8c6ae6: shadowmapSampler@s0=CLAMP, trilinearSampler@s1
          // =CLAMP, allSamplers[2]@s2=REPEAT). Static name priority always
          // grabbed `trilinearSampler` first → BSP world-scale UVs collapsed
          // to a single edge texel per face. World-geometry tiled materials
          // need REPEAT regardless of name. If no REPEAT-wrap sampler exists
          // (atlas-only shader, eye whites, etc.), fall through to the legacy
          // name-priority list, which preserves prior behavior.
          {
            const auto namedRes = cs->GetResourceNamesAndSlots();
            for (const auto& kv : namedRes) {
              if (kv.first.find("ampler") == std::string::npos) continue;
              if (kv.second >= D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) continue;
              D3D11SamplerState* samp = ps.samplers[kv.second];
              if (!samp) continue;
              Rc<DxvkSampler> dvk = samp->GetDXVKSampler();
              if (dvk == nullptr) continue;
              const auto& sInfo = dvk->info();
              if (sInfo.addressModeU == VK_SAMPLER_ADDRESS_MODE_REPEAT &&
                  sInfo.addressModeV == VK_SAMPLER_ADDRESS_MODE_REPEAT) {
                pickedSamp = dvk;
                pickedNameStorage = kv.first;
                pickedFromName = pickedNameStorage.c_str();
                pickedFromSlot = kv.second;
                pickStage = 0;
                break;
              }
            }
          }
          // 1) Legacy preferred-name list (kept for shaders where no named
          // sampler is REPEAT — covers atlas/UI/shadow-only PS variants).
          static const char* kPreferredSamplers[] = {
            "trilinearSampler","anisotropicSampler","bilinearSampler",
            "linearSampler","pointSampler","wrapSampler",
            // Source-engine array-bound sampler: RDEF names it
            // "allSamplers[0]" for the element form. Some shaders also use
            // BaseTextureSampler / NormalTextureSampler style.
            "allSamplers[0]","allSamplers",
            "BaseTextureSampler","AlbedoSampler","DiffuseSampler"
          };
          if (pickedSamp == nullptr) {
            for (const char* nm : kPreferredSamplers) {
              uint32_t s = cs->FindResourceSlot(nm);
              if (s == UINT32_MAX || s >= D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) continue;
              D3D11SamplerState* samp = ps.samplers[s];
              if (!samp) continue;
              pickedSamp = samp->GetDXVKSampler();
              pickedFromName = nm;
              pickedFromSlot = s;
              pickStage = 1;
              break;
            }
          }
          // 2) Fall back to same-slot sampler if no named match.
          if (pickedSamp == nullptr && slot < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
            D3D11SamplerState* samp = ps.samplers[slot];
            if (samp) {
              pickedSamp = samp->GetDXVKSampler();
              pickedFromSlot = slot;
              pickStage = 2;
            }
          }
          // 3) Final fallback: Remix default (REPEAT).
          if (pickedSamp == nullptr) {
            pickedSamp = getDefaultSampler();
            pickStage = 3;
          }
        }
        *r.sampDst = pickedSamp;
        if (r.slotDst) *r.slotDst = slot;
        anyAssigned = true;

        // NV-DXVK: per-PS, per-role diagnostic. Tells us WHY a given material
        // ended up with a particular sampler. Logged once per (PS hash, role,
        // pickStage, addressU/V/W) tuple. Goal: identify whether BSP draws
        // hit the same-slot fallback (stage=1) with a CLAMP_TO_EDGE sampler
        // (the suspected root cause of the per-face boxy artifact). Also
        // dumps the SRV's mip range vs underlying image's mip count, to test
        // whether the bound view restricts mip access (suspected root cause
        // of the uniform-grey-grain artifact after the wrap-fix).
        if (gameplayReady) {
          // Address modes (Vulkan codes: 0=REPEAT 2=CLAMP_EDGE).
          uint32_t aU = 0, aV = 0, aW = 0;
          // NV-DXVK: also capture anisotropy + filter modes so the SampPick
          // dump tells us whether the sampler the BSP path lands on actually
          // engages anisotropic filtering. The 1D-aware gradient path
          // (textureRead → SampleGrad) sets a 100x aniso ratio between
          // gradX/gradY; if useAnisotropy=0 or maxAnisotropy=1, the hardware
          // collapses that to a single mip and the result looks flat even
          // though the texture, UVs, and gradients are all correct.
          uint32_t aniOn = 0, aniMax = 0, magF = 0, minF = 0, mipM = 0;
          // NV-DXVK: capture mipLodBias / minLod / maxLod as well — TF2's
          // Source engine sometimes sets a negative bias on world-surface
          // samplers (mat_picmip-style) to bias toward higher-detail mips.
          // If we don't see it here, the sampler is using HW defaults
          // (bias=0, lod=[0..maxFloat]) and texture coarseness on distant
          // BSP walls is purely a function of gradient magnitude, not
          // engine-set bias. Float fields logged via str::format default.
          float lodBias = 0.0f, lodMin = 0.0f, lodMax = 0.0f;
          if (pickedSamp != nullptr) {
            const auto& si = pickedSamp->info();
            aU = uint32_t(si.addressModeU);
            aV = uint32_t(si.addressModeV);
            aW = uint32_t(si.addressModeW);
            aniOn  = si.useAnisotropy ? 1u : 0u;
            aniMax = uint32_t(si.maxAnisotropy);
            magF   = uint32_t(si.magFilter);
            minF   = uint32_t(si.minFilter);
            mipM   = uint32_t(si.mipmapMode);
            lodBias = si.mipmapLodBias;
            lodMin  = si.mipmapLodMin;
            lodMax  = si.mipmapLodMax;
          }
          // SRV mip range from the D3D11 view descriptor.
          D3D11_SHADER_RESOURCE_VIEW_DESC1 sd = {};
          srv->GetDesc1(&sd);
          uint32_t srvMostDetailed = 0, srvLevels = 0;
          if (sd.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D) {
            srvMostDetailed = sd.Texture2D.MostDetailedMip;
            srvLevels       = sd.Texture2D.MipLevels;
          }
          DxvkImage* di = view->image().ptr();
          const uint32_t imgMipCount = di ? di->info().mipLevels : 0;
          const uint32_t imgW = di ? di->info().extent.width  : 0;
          const uint32_t imgH = di ? di->info().extent.height : 0;
          const uint32_t imgFmt = di ? uint32_t(di->info().format) : 0;
          // NV-DXVK: image hash, stamped earlier in this same loop iteration
          // for D3D11 runtime images. Lets us cross-reference the bound albedo
          // against captures/textures/png/<hash>_albedo.png to confirm whether
          // BSP draws are bound to the texture we expected (the "tile-grout"
          // dump) or to a different, lower-detail one — diagnostic for the
          // residual uniform-grey-grain artifact after the wrap-aware sampler
          // pick lands the right (REPEAT) sampler.
          const uint64_t imgHash = di ? uint64_t(di->getHash()) : 0ull;

          // Dedup key: PS hash + role idx + stage + wrap + srvMip range
          // + imgHash (so two materials sharing the same shader but bound
          // to different textures still produce one log line each).
          XXH64_hash_t vsHashTmp = 0, psHashTmp = 0;
          GetCurrentVsPsHashes(vsHashTmp, psHashTmp);
          const uint64_t psHash = uint64_t(psHashTmp);
          const uint64_t key =
            psHash ^
            (uint64_t(uintptr_t(r.dst)) * 0x9E3779B97F4A7C15ull) ^
            (uint64_t(uint32_t(pickStage)) << 56) ^
            (uint64_t(aU) << 48) ^ (uint64_t(aV) << 44) ^ (uint64_t(aW) << 40) ^
            (uint64_t(srvMostDetailed) << 32) ^ uint64_t(srvLevels) ^
            (uint64_t(aniOn) << 39) ^ (uint64_t(aniMax) << 30) ^
            (imgHash * 0xC2B2AE3D27D4EB4Full);
          static std::unordered_set<uint64_t> seenSampPick;
          if (seenSampPick.insert(key).second) {
            Logger::info(str::format(
              "[D3D11Rtx.SampPick] PS=0x", std::hex, psHash, std::dec,
              " role=", r.names[0],
              " texSlot=", slot,
              " stage=", pickStage,
              (pickedFromName ? " viaName=" : " viaName=-"),
              (pickedFromName ? pickedFromName : ""),
              " sampSlot=", pickedFromSlot,
              " U=", aU, " V=", aV, " W=", aW,
              " aniOn=", aniOn, " aniMax=", aniMax,
              " magF=", magF, " minF=", minF, " mipM=", mipM,
              " lodBias=", lodBias, " lodMin=", lodMin, " lodMax=", lodMax,
              " srvMip=[", srvMostDetailed, "..+", srvLevels, "]",
              " imgMips=", imgMipCount,
              " imgWxH=", imgW, "x", imgH,
              " imgFmt=", imgFmt,
              " imgHash=0x", std::hex, imgHash, std::dec));
          }

          // Once per PS-hash, dump every RDEF resource whose name contains
          // "ampler" — gives us the BSP shader's actual sampler vocabulary so
          // we can decide whether to extend kPreferredSamplers (root-cause fix
          // for stage=1 fallthrough on legitimate "named" samplers we don't
          // know about), or whether the shader genuinely has only an
          // unnamed/anonymous CLAMP sampler (a different root cause).
          static std::unordered_set<uint64_t> seenPsRdefDump;
          if (psHash != 0 && seenPsRdefDump.insert(psHash).second) {
            const auto& names = cs->GetResourceNamesAndSlots();
            std::string sampList;
            for (const auto& kv : names) {
              if (kv.first.find("ampler") == std::string::npos) continue;
              if (!sampList.empty()) sampList += ", ";
              sampList += kv.first;
              sampList += "@s";
              sampList += std::to_string(kv.second);
              // Whether the game has actually bound a sampler at this slot
              // and what its wrap modes are.
              if (kv.second < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
                D3D11SamplerState* gSamp = ps.samplers[kv.second];
                if (gSamp) {
                  Rc<DxvkSampler> dvkSamp = gSamp->GetDXVKSampler();
                  if (dvkSamp != nullptr) {
                    const auto& gsi = dvkSamp->info();
                    sampList += "(U" + std::to_string(uint32_t(gsi.addressModeU))
                              + "V" + std::to_string(uint32_t(gsi.addressModeV))
                              + "W" + std::to_string(uint32_t(gsi.addressModeW)) + ")";
                  } else {
                    sampList += "(noDvk)";
                  }
                } else {
                  sampList += "(unbound)";
                }
              }
            }
            Logger::info(str::format(
              "[D3D11Rtx.PsSamplers] PS=0x", std::hex, psHash, std::dec,
              " count=", names.size(),
              " samplers=[", sampList, "]"));
          }
        }
      }
      return anyAssigned;
    };

    const bool rdefHit = classifyFromRdef();
    // If RDEF populated the albedo (colorTextures[0]), we can skip the
    // scoring candidate search entirely — that was only there to guess
    // which SRV is the color texture. We still scan for logging when
    // diagnostics are active.
    const bool rdefAlbedoBound = mat.colorTextures[0].isValid() && !mat.colorTextures[0].isImageEmpty();

    // NV-DXVK: per-draw read of Source/TF2 emissive intent from the PS's
    // own CBuffer, replacing the blend-state heuristic in
    // rtx_instance_manager.cpp. The PS's RDEF (parsed at compile time in
    // D3D11CommonShader::parseRdef) marks `c_emissiveTint` as used iff the
    // shader actually computes per-pixel emission; we read the per-draw
    // value of the field directly from the bound CBuffer slice. Same for
    // `c_useAlphaModulateEmissive` which gates the `emissive *= albedo.a`
    // behaviour in the original PS — wired through to slang via the
    // OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_MODULATE_EMISSIVE flag bit.
    //
    // No emission is forwarded for materials whose PS does NOT mark these
    // fields used — that's the diff between water/refract/decal layers
    // (which carry an emissiveTexture binding but never sample emission)
    // and intentional muzzle-flash / sign / panel materials.
    {
      const auto* psShader = ps.shader.ptr();
      if (psShader) {
        const auto* cs = psShader->GetCommonShader();
        if (cs) {
          const auto tintLoc = cs->FindCBField("CBufUberStatic", "c_emissiveTint");
          const bool tintUsed = cs->ReadsCBField("CBufUberStatic", "c_emissiveTint");
          if (tintLoc.has_value() && tintUsed
              && tintLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
              && tintLoc->size >= 12) {
            const auto& cbBinding = m_context->m_state.ps.constantBuffers[tintLoc->slot];
            if (cbBinding.buffer != nullptr) {
              const auto mapped = cbBinding.buffer->GetMappedSlice();
              const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              const size_t bufLen = cbBinding.buffer->Desc()->ByteWidth;
              // cb.constantOffset is in 16-byte units (D3D11 binding granularity).
              const size_t cbBaseOff = size_t(cbBinding.constantOffset) * 16;
              if (base != nullptr && cbBaseOff + tintLoc->offset + 12 <= bufLen) {
                float tint[3] = { 0.f, 0.f, 0.f };
                std::memcpy(tint, base + cbBaseOff + tintLoc->offset, 12);
                mat.sourceEmissiveTint = Vector3(tint[0], tint[1], tint[2]);
                // Used flag plus non-zero magnitude → material is genuinely
                // emissive. A `used=1, tint=(0,0,0)` material is one whose
                // PS reads the field but is currently configured dark (e.g.
                // an unlit panel state); leave emission off.
                const float tintMag2 = tint[0]*tint[0] + tint[1]*tint[1] + tint[2]*tint[2];
                if (tintMag2 > 1e-6f) {
                  mat.sourceUsesEmission = true;
                }
              }
            }
          }

          const auto modLoc = cs->FindCBField("CBufUberStatic", "c_useAlphaModulateEmissive");
          const bool modUsed = cs->ReadsCBField("CBufUberStatic", "c_useAlphaModulateEmissive");
          if (modLoc.has_value() && modUsed
              && modLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
              && modLoc->size >= 4) {
            const auto& cbBinding = m_context->m_state.ps.constantBuffers[modLoc->slot];
            if (cbBinding.buffer != nullptr) {
              const auto mapped = cbBinding.buffer->GetMappedSlice();
              const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              const size_t bufLen = cbBinding.buffer->Desc()->ByteWidth;
              const size_t cbBaseOff = size_t(cbBinding.constantOffset) * 16;
              if (base != nullptr && cbBaseOff + modLoc->offset + 4 <= bufLen) {
                float v = 0.f;
                std::memcpy(&v, base + cbBaseOff + modLoc->offset, 4);
                if (v != 0.f) {
                  mat.sourceAlphaModulatesEmissive = true;
                }
              }
            }
          }

          // NV-DXVK: TF2 worldspace VGUI / HUD shader detection. These PSes
          // write the final composited UI color directly to SV_Target with
          // no lighting math — they're inherently unlit. Identifying them
          // by the presence of all three structured-buffer resources that
          // the VGUI atlas pipeline declares (fontTexture + g_fontBounds +
          // g_imgBounds) is unambiguous: no other TF2 shader binds that
          // combination. Confirmed via fxc /dumpbin on FS_9b38c8e2a0ceec15
          // and FS_8deb2867429f3da5 (the "P2016 / Precision semi-auto
          // pistol" UI label and the gauntlet stats panel).
          //
          // Routed through to the surface as `isMatte=true` plus emissive
          // forwarding of the picked color texture, so the path tracer
          // outputs the UI color directly without lighting it.
          if (cs->FindResourceSlot("fontTexture")  != UINT32_MAX
           && cs->FindResourceSlot("g_fontBounds") != UINT32_MAX
           && cs->FindResourceSlot("g_imgBounds")  != UINT32_MAX) {
            mat.sourceIsUnlitUI = true;
            // One-shot per PS hash so we can confirm the detection fires.
            static std::unordered_set<XXH64_hash_t> sUiDumped;
            static std::mutex sUiDumpMu;
            XXH64_hash_t vsH2 = 0, psH2 = 0;
            GetCurrentVsPsHashes(vsH2, psH2);
            bool firstUi = false;
            {
              std::lock_guard<std::mutex> lk(sUiDumpMu);
              if (sUiDumped.insert(psH2).second) firstUi = true;
            }
            if (firstUi) {
              Logger::info(str::format(
                "[EmissiveSource.UnlitUI] PS=0x", std::hex, psH2, std::dec,
                " — TF2 VGUI/HUD shader, will be rendered isMatte=true with"
                " color texture forwarded as emissive (unlit output)."));
              // NV-DXVK: dump which colorTextures got captured. Mode 0
              // text uses colorTextures[0] = fontTexture SDF atlas. Mode 1
              // image needs colorTextures[1] = materialTexture (icon
              // atlas). If [1] is invalid for an image VGUI shader, the
              // role-pick at line 6288 didn't match "materialTexture" —
              // panels then render black.
              Logger::info(str::format(
                "[VguiTextures] PS=0x", std::hex, psH2, std::dec,
                " ct0.valid=", (mat.colorTextures[0].isValid() && !mat.colorTextures[0].isImageEmpty() ? 1 : 0),
                " ct0.hash=0x", std::hex, mat.colorTextures[0].getImageHash(),
                " ct0.slot=", std::dec, mat.colorTextureSlot[0],
                " | ct1.valid=", (mat.colorTextures[1].isValid() && !mat.colorTextures[1].isImageEmpty() ? 1 : 0),
                " ct1.hash=0x", std::hex, mat.colorTextures[1].getImageHash(),
                " ct1.slot=", std::dec, mat.colorTextureSlot[1]));
              // Dump every PS-declared SRV name+slot so we can see what
              // t1 actually is for this shader (might be named something
              // other than "materialTexture").
              const auto namedRes2 = cs->GetResourceNamesAndSlots();
              std::string srvDump;
              for (const auto& kv : namedRes2) {
                srvDump += " {";
                srvDump += kv.first;
                srvDump += "@";
                srvDump += std::to_string(kv.second);
                srvDump += "}";
              }
              Logger::info(str::format(
                "[VguiTextures.SRVs] PS=0x", std::hex, psH2, std::dec,
                " RDEF=", srvDump.c_str()));
            }
          }

          // One-shot per PS-hash dump so we can verify which materials end
          // up routed through the genuine emissive path. Mirrors the
          // [EmissivePromote.*] log structure for grep-compatibility.
          if (mat.sourceUsesEmission || mat.sourceAlphaModulatesEmissive) {
            static std::unordered_set<XXH64_hash_t> sEmissiveDumped;
            static std::mutex sEmissiveDumpMu;
            XXH64_hash_t vsH = 0, psH = 0;
            GetCurrentVsPsHashes(vsH, psH);
            bool firstDump = false;
            {
              std::lock_guard<std::mutex> lk(sEmissiveDumpMu);
              if (sEmissiveDumped.insert(psH).second) firstDump = true;
            }
            if (firstDump) {
              Logger::info(str::format(
                "[EmissiveSource.PsCb] PS=0x", std::hex, psH, std::dec,
                " sourceUsesEmission=", mat.sourceUsesEmission ? 1 : 0,
                " sourceAlphaModulatesEmissive=", mat.sourceAlphaModulatesEmissive ? 1 : 0,
                " tint=(", mat.sourceEmissiveTint.x, ",",
                          mat.sourceEmissiveTint.y, ",",
                          mat.sourceEmissiveTint.z, ")"));
              // NV-DXVK: post-role-pick state of the emissive routing for
              // this PS. Confirms whether mat.emissiveTexture got assigned
              // (= the slang shader will sample it) and the albedo hash that
              // drives the c_useAlphaModulateEmissive multiply downstream.
              const bool eValid = mat.emissiveTexture.isValid() && !mat.emissiveTexture.isImageEmpty();
              const bool aValid = mat.colorTextures[0].isValid() && !mat.colorTextures[0].isImageEmpty();
              Logger::info(str::format(
                "[EmissiveSource.MatState] PS=0x", std::hex, psH, std::dec,
                " emissiveTex.valid=", eValid ? 1 : 0,
                " emissiveTex.hash=0x", std::hex, mat.emissiveTexture.getImageHash(), std::dec,
                " | albedoTex.valid=", aValid ? 1 : 0,
                " albedoTex.hash=0x", std::hex, mat.colorTextures[0].getImageHash(), std::dec,
                " (the c_useAlphaModulateEmissive multiply uses albedoTex.a"
                " — if alpha=0 in the lit-up regions of the original PS,"
                " emissive will be killed downstream)"));
            }
          }

          // NV-DXVK: TF2 viewmodel "screen-space scrolling overlay" pattern
          // detection. The original PS samples the emissive texture at a UV
          // derived from SV_Position (the pixel's screen coord) transformed
          // by c_uv1*, then multiplied by t17 (emissiveMultiplyTexture)
          // sampled at the mesh UV. Confirmed via fxc /dumpbin on PS
          // 0x7836c1dd4d5c885f / 0xea2b85b0f20fddf3 (asm lines 280-296):
          //   mul r3.xyzw, v4.xyxy, cb2[29].xxxy           ; r3 = SCREEN UV
          //   dp2 r0.z, r3.xy, cb0[0].xy / cb0[0].zw       ; × c_uv1RotScale
          //   mad r4.x, cb0[1].x, cb2[18].w, r0.z          ; + c_uv1Translate × time
          //   sample r4.xyz, r4.xyxx, t4.xyzw              ; emissiveTexture
          //   mul r4.xyz, r4.xyzx, cb0[10].xyzx            ; × c_emissiveTint
          //   sample r5.xyz, v0.xyxx, t17.xyzw             ; emissiveMultiplyTexture
          //   mul r4.xyz, r4.xyzx, r5.xyzx                 ; × mask
          //
          // Pattern signature for detection:
          //   1. PS reads c_uv1RotScaleX + c_uv1RotScaleY + c_uv1Translate
          //   2. PS has emissiveTexture bound (mat.emissiveTexture valid)
          //   3. PS RDEF declares an "emissiveMultiplyTexture" slot bound
          //
          // Phase-1 (this commit): detect-and-log only. Reads c_uv1*
          // values from CBufUberStatic, captures the emissiveMultiplyTexture
          // slot+hash, and sets the
          // OPAQUE_SURFACE_MATERIAL_FLAG_HAS_SCREEN_SPACE_EMISSIVE flag bit
          // on the material. GPU-side plumbing (per-material screen-UV
          // matrix + mask texture index + slang sampling) lands in the
          // Phase-2 follow-up that grows kSurfaceMaterialGPUSize.
          {
            const auto rsxLoc = cs->FindCBField("CBufUberStatic", "c_uv1RotScaleX");
            const auto rsyLoc = cs->FindCBField("CBufUberStatic", "c_uv1RotScaleY");
            const auto trLoc  = cs->FindCBField("CBufUberStatic", "c_uv1Translate");
            const bool rsxUsed = cs->ReadsCBField("CBufUberStatic", "c_uv1RotScaleX");
            const bool rsyUsed = cs->ReadsCBField("CBufUberStatic", "c_uv1RotScaleY");
            const bool trUsed  = cs->ReadsCBField("CBufUberStatic", "c_uv1Translate");
            // Smoking-gun signal that the PS computes a screen-space-derived
            // UV: it reads c_rcpRenderTargetSize (= cb2[29].xy). Mesh-UV-only
            // shaders never need (1/width, 1/height). Confirmed in
            // ps_ea2b85b0f20fddf3.asm line 57 (CBufCommonPerCamera offset 464
            // = cb2[29]).
            const bool rcpRtSizeUsed = cs->ReadsCBField("CBufCommonPerCamera", "c_rcpRenderTargetSize");
            // Mask texture is optional — the masked variant uses t17
            // emissiveMultiplyTexture (PS 0x7836c1dd4d5c885f); the
            // maskless variant has no such slot (PS 0xea2b85b0f20fddf3).
            // Both still produce the screen-space scrolling overlay.
            const uint32_t emissiveMaskSlot = cs->FindResourceSlot("emissiveMultiplyTexture");
            const bool emissiveValid = mat.emissiveTexture.isValid() && !mat.emissiveTexture.isImageEmpty();
            const bool patternMatch =
                 emissiveValid
              && rsxLoc.has_value() && rsyLoc.has_value() && trLoc.has_value()
              && rsxUsed && rsyUsed && trUsed
              && rcpRtSizeUsed
              && rsxLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;

            if (patternMatch) {
              // Read the c_uv1* CB values (matrix row1, row2, translate).
              float uv1RotScaleX[2] = { 1.f, 0.f };
              float uv1RotScaleY[2] = { 0.f, 1.f };
              float uv1Translate[2] = { 0.f, 0.f };
              const auto& cbBinding = m_context->m_state.ps.constantBuffers[rsxLoc->slot];
              if (cbBinding.buffer != nullptr) {
                const auto mapped = cbBinding.buffer->GetMappedSlice();
                const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                const size_t bufLen = cbBinding.buffer->Desc()->ByteWidth;
                const size_t cbBaseOff = size_t(cbBinding.constantOffset) * 16;
                auto readFloat2 = [&](const auto& loc, float out[2]) {
                  if (base != nullptr && loc->size >= 8 &&
                      cbBaseOff + loc->offset + 8 <= bufLen) {
                    std::memcpy(out, base + cbBaseOff + loc->offset, 8);
                  }
                };
                readFloat2(rsxLoc, uv1RotScaleX);
                readFloat2(rsyLoc, uv1RotScaleY);
                readFloat2(trLoc,  uv1Translate);
              }

              // NV-DXVK: read c_gameTime (CBufCommonPerCamera offset 300 =
              // cb2[18].w). The asm RDEF for both screen-space emissive PSes
              // explicitly names this float — `float c_gameTime; Offset: 300
              // Size: 4` — and uses it as a multiplier on c_uv1Translate (and
              // c_uv2Translate for the t14 detail path) to scroll the UV per
              // frame:
              //
              //   mad r4.x, cb0[1].x, cb2[18].w, r0.z   ; UV.x = dot + tx*t
              //   mad r4.y, cb0[1].y, cb2[18].w, r0.z   ; UV.y = dot + ty*t
              //
              // The Phase-2 slang hardcodes this scalar to 1.0 (see the
              // `// animScalar in the original PS comes from cb2[18].w` block
              // in opaque_surface_material_interaction.slangh). With t=1 the
              // pattern is frozen at game-time-one-second instead of scrolling.
              //
              // Log the live cbuffer value here so we can:
              //   (a) confirm c_gameTime is actually populated with a sane
              //       per-frame value (vs. zero / NaN / stale)
              //   (b) confirm c_uv1Translate is non-zero (otherwise the
              //       missing time scalar is visually moot)
              //   (c) compute the actual UV offset native applies vs. ours
              //       (translate × t  vs.  translate × 1.0)
              float gameTime = 0.f;
              uint32_t gameTimeSlot   = UINT32_MAX;
              uint32_t gameTimeOffset = UINT32_MAX;
              bool     gameTimeRead   = false;
              const auto gtLoc  = cs->FindCBField("CBufCommonPerCamera", "c_gameTime");
              const bool gtUsed = cs->ReadsCBField("CBufCommonPerCamera", "c_gameTime");
              if (gtLoc.has_value() &&
                  gtLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                gameTimeSlot   = gtLoc->slot;
                gameTimeOffset = gtLoc->offset;
                const auto& gtBinding = m_context->m_state.ps.constantBuffers[gtLoc->slot];
                if (gtBinding.buffer != nullptr) {
                  const auto mapped = gtBinding.buffer->GetMappedSlice();
                  const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  const size_t bufLen = gtBinding.buffer->Desc()->ByteWidth;
                  const size_t cbBaseOff = size_t(gtBinding.constantOffset) * 16;
                  if (base != nullptr && gtLoc->size >= 4 &&
                      cbBaseOff + gtLoc->offset + 4 <= bufLen) {
                    std::memcpy(&gameTime, base + cbBaseOff + gtLoc->offset, 4);
                    gameTimeRead = true;
                  }
                }
              }

              // NV-DXVK: stash the captured c_gameTime on SceneManager so
              // RtxContext can plumb it into RaytraceArgs.screenSpaceEmissiveTime
              // each frame. The slang's screen-space emissive branch reads
              // that value and uses it as the per-frame multiplier on
              // c_uv1Translate (replacing the previous hardcoded 1.0). Last
              // writer wins — c_gameTime is per-frame-uniform on the engine
              // side, so any draw within the frame produces the same value.
              if (gameTimeRead) {
                m_context->m_device->getCommon()->getSceneManager()
                  .setEngineGameTime(gameTime);
              }

              // Capture the t17 (emissiveMultiplyTexture) bound SRV hash for
              // the log so we can verify which mask atlas the PS uses.
              // Also stash the TextureRef on mat so the slang material gets
              // a bindless texture index for it (used as the mask).
              uint64_t maskHash = 0;
              TextureRef maskTexRef;
              if (emissiveMaskSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
                D3D11ShaderResourceView* maskSrv =
                  ps.shaderResources.views[emissiveMaskSlot].ptr();
                if (maskSrv != nullptr) {
                  Rc<DxvkImageView> mview = maskSrv->GetImageView();
                  if (mview != nullptr && mview->image() != nullptr) {
                    maskHash = mview->image()->getHash();
                    maskTexRef = TextureRef(mview);
                  }
                }
              }

              // Stash the captured screen-space emissive params on the
              // LegacyMaterialData so the OpaqueMaterialData converter can
              // wire them through to the GPU material in Phase 2.
              mat.hasScreenSpaceEmissive = true;
              mat.screenSpaceEmissiveUv1RotScaleX = Vector2(uv1RotScaleX[0], uv1RotScaleX[1]);
              mat.screenSpaceEmissiveUv1RotScaleY = Vector2(uv1RotScaleY[0], uv1RotScaleY[1]);
              mat.screenSpaceEmissiveUv1Translate = Vector2(uv1Translate[0], uv1Translate[1]);
              mat.screenSpaceEmissiveMaskTextureSlot = uint16_t(emissiveMaskSlot);
              mat.screenSpaceEmissiveMaskTexture = maskTexRef;

              // One-shot log per PS hash so noisy gameplay doesn't flood.
              XXH64_hash_t vsH_sse = 0, psH_sse = 0;
              GetCurrentVsPsHashes(vsH_sse, psH_sse);
              static std::unordered_set<XXH64_hash_t> sSseDumped;
              static std::mutex sSseDumpMu;
              bool firstSse = false;
              {
                std::lock_guard<std::mutex> lk(sSseDumpMu);
                firstSse = sSseDumped.insert(psH_sse).second;
              }
              if (firstSse) {
                const std::string maskSlotStr = (emissiveMaskSlot == UINT32_MAX)
                  ? std::string("none(maskless variant)")
                  : (std::string("t") + std::to_string(emissiveMaskSlot));
                const std::string gtSlotStr = gtLoc.has_value()
                  ? (std::string("cb") + std::to_string(gameTimeSlot)
                     + "[+" + std::to_string(gameTimeOffset) + "]")
                  : std::string("<NOT REFLECTED in CBufCommonPerCamera>");
                Logger::info(str::format(
                  "[ScreenSpaceEmissive.Detected] PS=0x", std::hex, psH_sse, std::dec,
                  " c_uv1RotScaleX=(", uv1RotScaleX[0], ",", uv1RotScaleX[1], ")",
                  " c_uv1RotScaleY=(", uv1RotScaleY[0], ",", uv1RotScaleY[1], ")",
                  " c_uv1Translate=(", uv1Translate[0], ",", uv1Translate[1], ")",
                  " c_gameTime=", (gameTimeRead ? gameTime : 0.f),
                  " (loc=", gtSlotStr.c_str(),
                  " used=", (gtUsed ? 1 : 0),
                  " readOK=", (gameTimeRead ? 1 : 0), ")",
                  " emissiveMaskSlot=", maskSlotStr.c_str(),
                  " maskHash=0x", std::hex, maskHash, std::dec,
                  " emissiveTexHash=0x", std::hex, mat.emissiveTexture.getImageHash(), std::dec,
                  " — c_gameTime now plumbed through"
                  " RaytraceArgs.screenSpaceEmissiveTime; slang scrolls"
                  " UV by translate × c_gameTime each frame"));
              }

              // NV-DXVK [ScreenSpaceEmissive.GameTimeWatch]: 1 Hz per-PS
              // throttled log of the live c_gameTime, so we can confirm
              // it ticks each frame (vs. being a stuck constant the engine
              // never actually animates) AND show the divergence between
              // native (translate × c_gameTime) and Remix's slang
              // (translate × 1.0). If c_gameTime barely changes between
              // samples the user can capture without ever seeing pulsing
              // lines, the deferral isn't actually visible — and we can
              // close it without plumbing RaytraceArgs.
              //
              // Gating: the outer detection block already runs only when
              // patternMatch is true (= a real screen-space emissive PS
              // was bound for this draw), which is automatically gated
              // behind menu/loading because no SSE-emissive draw issues
              // until gameplay. No extra `gameplayReady` needed here.
              if (gameTimeRead) {
                using clk = std::chrono::steady_clock;
                struct GtSample {
                  clk::time_point lastEmitted;
                  float           lastValue;
                  bool            haveLast;
                };
                static std::unordered_map<XXH64_hash_t, GtSample> sGtWatch;
                static std::mutex sGtWatchMu;
                const auto now = clk::now();
                bool emit = false;
                float prevValue = 0.f;
                int64_t sinceMs = 0;
                {
                  std::lock_guard<std::mutex> lk(sGtWatchMu);
                  auto& s = sGtWatch[psH_sse];
                  sinceMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - s.lastEmitted).count();
                  if (!s.haveLast || sinceMs >= 1000) {
                    emit      = true;
                    prevValue = s.haveLast ? s.lastValue : gameTime;
                    s.lastEmitted = now;
                    s.lastValue   = gameTime;
                    s.haveLast    = true;
                  }
                }
                if (emit) {
                  const float dt        = gameTime - prevValue;
                  const float uvOffsetX = uv1Translate[0] * gameTime;
                  const float uvOffsetY = uv1Translate[1] * gameTime;
                  Logger::info(str::format(
                    "[ScreenSpaceEmissive.GameTimeWatch] PS=0x",
                    std::hex, psH_sse, std::dec,
                    " c_gameTime=", gameTime,
                    " dt(s)=", dt,
                    " sinceLastEmitMs=", sinceMs,
                    " | c_uv1Translate=(", uv1Translate[0], ",", uv1Translate[1], ")",
                    " | UV offset (translate × c_gameTime) = (",
                    uvOffsetX, ",", uvOffsetY, ")",
                    " — slang now uses cb.screenSpaceEmissiveTime,"
                    " so this matches the native sample location"));
                }
              }
            }
          }
        }
      }
    }

    // NV-DXVK: per-draw "all bound PS SRVs" dump. SampPick logs only the
    // role-matched winner; this logs every non-null PS SRV slot together
    // with its RDEF name (so we can see which slot the shader thinks is
    // albedo vs lightmap vs detail) and image hash/dims/format. One-shot
    // per (VS,PS) so a static wall draw and a skinned character draw each
    // produce one line — making it possible to grep and compare which
    // stages they actually bind. Diagnoses the "skinned has correct
    // textures, world doesn't" symptom: if static path RDEF has no
    // albedoTexture name, or binds the lightmap on the slot the picker
    // expects to be diffuse, that shows up directly here.
    if (gameplayReady) {
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      const uint64_t key = uint64_t(vsH) ^ (uint64_t(psH) * 0x9E3779B97F4A7C15ull);
      static std::unordered_set<uint64_t> seenAllSrvs;
      if (seenAllSrvs.insert(key).second) {
        // Build slot -> RDEF name map (PS SRV side ONLY). m_resourceSlots
        // mirrors every binding type (cbuffers/SRVs/UAVs/samplers) keyed by
        // name, with bindPt living in disjoint register namespaces (cN, tN,
        // uN, sN). Naively mapping name -> slot conflates cbuffer slot 0
        // (e.g. CBufUberStatic@b0) with SRV slot 0, mislabeling textures.
        // Filter out cbuffer names (from GetCBufferNamesAndSlots) and
        // sampler names ("ampler" substring matches the existing sampler
        // heuristic at line 5348/5490). UAVs (uN) rarely share names with
        // tN textures so we accept the residual misattribution risk.
        std::array<const char*, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT> slotName{};
        std::vector<std::string> nameStorage;
        if (const auto* psShader = ps.shader.ptr()) {
          if (const auto* cs = psShader->GetCommonShader()) {
            std::unordered_set<std::string> cbufNames;
            for (const auto& kv : cs->GetCBufferNamesAndSlots()) {
              cbufNames.insert(kv.first);
            }
            const auto names = cs->GetResourceNamesAndSlots();
            nameStorage.reserve(names.size());
            for (const auto& kv : names) {
              if (kv.second >= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) continue;
              if (cbufNames.count(kv.first)) continue;
              if (kv.first.find("ampler") != std::string::npos) continue;
              if (slotName[kv.second] != nullptr) continue; // first-wins
              nameStorage.push_back(kv.first);
              slotName[kv.second] = nameStorage.back().c_str();
            }
          }
        }

        std::string srvList;
        uint32_t totalBound = 0;
        const auto& psSrvViews = ps.shaderResources.views;
        for (uint32_t i = 0; i < psSrvViews.size(); ++i) {
          auto* srv = psSrvViews[i].ptr();
          if (!srv) continue;
          ++totalBound;
          if (!srvList.empty()) srvList += " ";
          srvList += "t" + std::to_string(i);
          srvList += "{";
          srvList += "name=";
          srvList += (slotName[i] ? slotName[i] : "-");
          // Buffer SRVs and image SRVs both possible.
          Rc<DxvkImageView> view = srv->GetImageView();
          if (view == nullptr || view->image() == nullptr) {
            srvList += " buf}";
            continue;
          }
          DxvkImage* img = view->image().ptr();
          const uint64_t h = img ? uint64_t(img->getHash()) : 0ull;
          const auto& info = img->info();
          char buf[160];
          std::snprintf(buf, sizeof(buf),
            " hash=0x%llx %ux%u fmt=%u mips=%u}",
            (unsigned long long)h,
            info.extent.width, info.extent.height,
            uint32_t(info.format), info.mipLevels);
          srvList += buf;
        }
        Logger::info(str::format(
          "[D3D11Rtx.AllSrvs] VS=0x", std::hex, uint64_t(vsH), std::dec,
          " PS=0x", std::hex, uint64_t(psH), std::dec,
          " totalBound=", totalBound,
          " rdefAlbedo=", rdefAlbedoBound ? 1 : 0,
          " srvs=[", srvList, "]"));

        // NV-DXVK: TF2 cloudmap projection capture. The wall PS samples
        // cloudMaskTexture (t8) at a UV computed from the camera-relative
        // world position via 4 vec2 constants in CBufCommonPerCamera:
        //   cloudUV = c_cloudRelConst
        //           + worldPos.x * c_cloudRelForX
        //           + worldPos.y * c_cloudRelForY
        //           + worldPos.z * c_cloudRelForZ
        // Native renders these surfaces with visible structure because the
        // cloudmap (small-range UV → low mip → visible variation) modulates
        // the otherwise-flat-mip-clamped albedo. Remix samples cloudMaskTexture
        // at the same huge planar UV as albedo, so it ALSO mip-clamps and we
        // lose the modulation. Logging the constants once per PS so we can
        // (a) verify the values look sane (small enough to produce 0..N UV
        // over typical world ranges) and (b) decide whether to plumb them
        // onto Surface for slangh-side replay.
        if (const auto* psShader = ps.shader.ptr()) {
          if (const auto* cs = psShader->GetCommonShader()) {
            auto fcConst = cs->FindCBField("CBufCommonPerCamera", "c_cloudRelConst");
            auto fcForX  = cs->FindCBField("CBufCommonPerCamera", "c_cloudRelForX");
            auto fcForY  = cs->FindCBField("CBufCommonPerCamera", "c_cloudRelForY");
            auto fcForZ  = cs->FindCBField("CBufCommonPerCamera", "c_cloudRelForZ");
            if (fcConst && fcForX && fcForY && fcForZ
                && fcConst->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
                && fcConst->slot == fcForX->slot
                && fcConst->slot == fcForY->slot
                && fcConst->slot == fcForZ->slot) {
              const auto& psCbs = ps.constantBuffers;
              const auto& cbB = psCbs[fcConst->slot];
              if (cbB.buffer != nullptr) {
                const auto mapped = cbB.buffer->GetMappedSlice();
                const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                if (p != nullptr) {
                  const size_t base = static_cast<size_t>(cbB.constantOffset) * 16;
                  const size_t bufLen = cbB.buffer->Desc()->ByteWidth;
                  auto readVec2 = [&](uint32_t off, float& a, float& b) -> bool {
                    const size_t fullOff = base + off;
                    if (fullOff + 8 > bufLen) return false;
                    std::memcpy(&a, p + fullOff + 0, 4);
                    std::memcpy(&b, p + fullOff + 4, 4);
                    return std::isfinite(a) && std::isfinite(b);
                  };
                  float cc0 = 0, cc1 = 0, cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0, cz0 = 0, cz1 = 0;
                  const bool ok = readVec2(fcConst->offset, cc0, cc1)
                               && readVec2(fcForX->offset,  cx0, cx1)
                               && readVec2(fcForY->offset,  cy0, cy1)
                               && readVec2(fcForZ->offset,  cz0, cz1);
                  if (ok) {
                    Logger::info(str::format(
                      "[D3D11Rtx.CloudProj] VS=0x", std::hex, uint64_t(vsH), std::dec,
                      " PS=0x", std::hex, uint64_t(psH), std::dec,
                      " cb=", fcConst->slot,
                      " const=(", cc0, ",", cc1, ")",
                      " forX=(", cx0, ",", cx1, ")",
                      " forY=(", cy0, ",", cy1, ")",
                      " forZ=(", cz0, ",", cz1, ")"));
                  } else {
                    Logger::info(str::format(
                      "[D3D11Rtx.CloudProj] VS=0x", std::hex, uint64_t(vsH), std::dec,
                      " PS=0x", std::hex, uint64_t(psH), std::dec,
                      " read_failed_or_non_finite"));
                  }
                }
              }
            }
          }
        }
      }
    }

    auto isBlockCompressed = [](DXGI_FORMAT fmt) -> bool {
      return (fmt >= DXGI_FORMAT_BC1_TYPELESS && fmt <= DXGI_FORMAT_BC1_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC2_TYPELESS && fmt <= DXGI_FORMAT_BC2_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC3_TYPELESS && fmt <= DXGI_FORMAT_BC3_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC4_TYPELESS && fmt <= DXGI_FORMAT_BC4_SNORM)
          || (fmt >= DXGI_FORMAT_BC5_TYPELESS && fmt <= DXGI_FORMAT_BC5_SNORM)
          || (fmt >= DXGI_FORMAT_BC6H_TYPELESS && fmt <= DXGI_FORMAT_BC7_UNORM_SRGB);
    };

    // Collect currently-bound render target images AND their dimensions.
    // Only reject SRVs that point to images actively bound as RTs.
    // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT is set on most D3D11 textures
    // (engines create them with BIND_RENDER_TARGET for mip gen, dynamic
    // updates, etc.), so the flag alone is NOT a reliable RT indicator.
    const auto& omState = m_context->m_state.om;
    std::array<DxvkImage*, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> boundRTImages = {};
    uint32_t rtWidth = 0, rtHeight = 0;
    for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
      auto* rtv = omState.renderTargetViews[rt].ptr();
      if (rtv) {
        Rc<DxvkImageView> rtvView = rtv->GetImageView();
        if (rtvView != nullptr) {
          boundRTImages[rt] = rtvView->image().ptr();
          if (rt == 0) {
            rtWidth  = rtvView->image()->info().extent.width;
            rtHeight = rtvView->image()->info().extent.height;
          }
        }
      }
    }

    // First pass: collect candidate textures with scoring.
    // Score: BC=+10, mips=+5, non-RT-sized=+3, lower slot=+1.
    // This replaces the old binary accept/reject that was too aggressive.
    struct TexCandidate {
      uint32_t slot;
      Rc<DxvkImageView> view;
      int score;
      bool isCurrentRT;
      std::string info;
    };
    std::vector<TexCandidate> candidates;

    for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; ++slot) {
      D3D11ShaderResourceView* srv = ps.shaderResources.views[slot].ptr();
      if (!srv) continue;
      if (srv->GetResourceType() != D3D11_RESOURCE_DIMENSION_TEXTURE2D) { ++rejNonTex2D; continue; }

      Rc<DxvkImageView> view = srv->GetImageView();
      if (view == nullptr) { ++rejNullView; continue; }

      // NV-DXVK: stamp a stable per-image hash on the underlying DxvkImage so
      // LegacyMaterial dedup works. See equivalent block in the RDEF path
      // above for the rationale.
      {
        DxvkImage* img = view->image().ptr();
        if (img && img->getHash() == 0) {
          struct ImgHashKey {
            uint64_t handle;
            uint32_t width, height, depth;
            uint32_t format;
            uint32_t mipLevels;
          };
          const auto& ii = img->info();
          ImgHashKey k = {
            reinterpret_cast<uint64_t>(img),
            ii.extent.width, ii.extent.height, ii.extent.depth,
            uint32_t(ii.format), ii.mipLevels,
          };
          img->setHash(XXH3_64bits(&k, sizeof(k)));
        }
      }

      const auto& imgInfo = view->image()->info();
      D3D11_SHADER_RESOURCE_VIEW_DESC1 srvDesc = {};
      srv->GetDesc1(&srvDesc);
      const DXGI_FORMAT fmt = srvDesc.Format;
      const bool bc = isBlockCompressed(fmt);
      const bool hasMips = imgInfo.mipLevels > 1;

      DxvkImage* srvImage = view->image().ptr();
      bool isCurrentRT = false;
      for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
        if (boundRTImages[rt] == srvImage) { isCurrentRT = true; break; }
      }

      // Skip tiny dummy textures (1x1 default white/black).
      if (imgInfo.extent.width <= 2 && imgInfo.extent.height <= 2) {
        ++rejTiny;
        continue;
      }

      // Check if texture dimensions match current render target (likely GBuffer/intermediate).
      const bool matchesRT = (rtWidth > 0 && rtHeight > 0
        && imgInfo.extent.width == rtWidth && imgInfo.extent.height == rtHeight);

      // NV-DXVK: classify format. Source-engine PBR material sets bind albedo
      // in slot 0 and additional data-maps (normal/spec/gloss/roughness) in
      // slots 1+. The old slot-based scoring put a normal map (BC5_UNORM,
      // fmt=83) into colorTextures[1], which Remix then passed to
      // setSecondaryTexture() — corrupting the visible color. Penalize
      // 1/2-channel "data" formats so they never outrank a real color
      // texture, and bonus SRGB formats that are almost always albedo.
      auto isDataOnlyFormat = [](DXGI_FORMAT f) -> bool {
        switch (f) {
          case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM:
          case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
          case DXGI_FORMAT_R8_UNORM:     case DXGI_FORMAT_R8_SNORM:
          case DXGI_FORMAT_R8G8_UNORM:   case DXGI_FORMAT_R8G8_SNORM:
          case DXGI_FORMAT_R16_UNORM:    case DXGI_FORMAT_R16_SNORM:
          case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM:
          case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16_FLOAT:
            return true;
          default: return false;
        }
      };
      auto isSrgbFormat = [](DXGI_FORMAT f) -> bool {
        switch (f) {
          case DXGI_FORMAT_BC1_UNORM_SRGB:
          case DXGI_FORMAT_BC2_UNORM_SRGB:
          case DXGI_FORMAT_BC3_UNORM_SRGB:
          case DXGI_FORMAT_BC7_UNORM_SRGB:
          case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
            return true;
          default: return false;
        }
      };
      const bool dataOnly = isDataOnlyFormat(fmt);
      const bool srgbColor = isSrgbFormat(fmt);

      int score = 0;
      if (bc)                       score += 10;  // Block-compressed = always content
      if (hasMips)                  score += 5;   // Mipmapped = likely content
      if (!matchesRT)               score += 3;   // Different size from RT = likely content
      if (!isCurrentRT)             score += 2;   // Not actively rendering to it
      if (srgbColor)                score += 8;   // SRGB — almost always an albedo/color texture
      if (dataOnly)                 score -= 30;  // Normal/spec/roughness must never beat color
      score += std::max(0, 16 - (int)slot);       // Prefer lower slots (albedo first)

      // Currently bound as active RT → negative score (only use as absolute last resort)
      if (isCurrentRT) score = -10;

      std::string info;
      if (doLog) {
        // NV-DXVK: SRV may expose only a sub-range of the image's mips
        // (MostDetailedMip + MipLevels) — typical for streaming systems.
        // If MostDetailedMip > 0, the SRV hides the fine mips and
        // SampleGrad clamps to the coarsest available mip → wall samples
        // an averaged "mean texture colour" no matter what gradient we pass.
        uint32_t srvMipMin = 0, srvMipCount = 0;
        if (srvDesc.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D) {
          srvMipMin   = srvDesc.Texture2D.MostDetailedMip;
          srvMipCount = srvDesc.Texture2D.MipLevels;
        } else if (srvDesc.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2DARRAY) {
          srvMipMin   = srvDesc.Texture2DArray.MostDetailedMip;
          srvMipCount = srvDesc.Texture2DArray.MipLevels;
        }
        info = str::format("  slot=", slot,
          " fmt=", (uint32_t)fmt,
          " w=", imgInfo.extent.width, " h=", imgInfo.extent.height,
          " mips=", imgInfo.mipLevels,
          " srvMip=[", srvMipMin, "..+", srvMipCount, "]",
          " score=", score,
          bc ? " [BC]" : "",
          hasMips ? " [MIPS]" : "",
          isCurrentRT ? " [BOUND-RT]" : "",
          matchesRT ? " [RT-SIZED]" : "",
          srgbColor ? " [SRGB]" : "",
          dataOnly ? " [DATA]" : "", "\n");
      }

      candidates.push_back({ slot, std::move(view), score, isCurrentRT, std::move(info) });
    }

    // Sort by score descending — best content textures first.
    std::sort(candidates.begin(), candidates.end(),
      [](const TexCandidate& a, const TexCandidate& b) { return a.score > b.score; });

    // Pick up to kMaxSupportedTextures (or 1 if ignoreSecondaryTextures is set).
    // If all have negative scores (all are currently-bound RTs), accept the
    // least-bad one rather than submitting zero textures.
    // NV-DXVK: when the RDEF classifier already bound albedo, skip the
    // scoring-driven writes so we don't overwrite the named albedo with a
    // guess. Still run the candidate list if RDEF produced nothing, so
    // UI/post-fx draws (no named SRVs) keep working as before.
    const uint32_t maxTextures = RtxOptions::ignoreSecondaryTextures()
                                ? 1u : LegacyMaterialData::kMaxSupportedTextures;
    bool pickedAny = rdefAlbedoBound;
    if (rdefAlbedoBound) {
      textureID = mat.colorTextures[1].isValid() && !mat.colorTextures[1].isImageEmpty() ? 2u : 1u;
    }
    if (!rdefAlbedoBound) {
      for (auto& c : candidates) {
        if (textureID >= maxTextures) break;
        // Skip currently-bound RTs unless we have no other option.
        if (c.isCurrentRT && !candidates.empty() && candidates[0].score > 0)
          continue;

        mat.colorTextures[textureID] = TextureRef(std::move(c.view));
        mat.colorTextureSlot[textureID] = c.slot;

        if (c.slot < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
          D3D11SamplerState* samp = ps.samplers[c.slot];
          mat.samplers[textureID] = samp ? samp->GetDXVKSampler() : getDefaultSampler();
        } else {
          mat.samplers[textureID] = getDefaultSampler();
        }

        pickedAny = true;
        ++textureID;
      }
    }

    // If nothing was picked and there are candidates, take the best one anyway.
    // Remix with a dubious texture is better than Remix with no texture at all.
    if (!pickedAny && !candidates.empty()) {
      auto& c = candidates[0];
      mat.colorTextures[0] = TextureRef(Rc<DxvkImageView>(c.view));
      mat.colorTextureSlot[0] = c.slot;
      if (c.slot < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
        D3D11SamplerState* samp = ps.samplers[c.slot];
        mat.samplers[0] = samp ? samp->GetDXVKSampler() : getDefaultSampler();
      } else {
        mat.samplers[0] = getDefaultSampler();
      }
      textureID = 1;
    }

    if (doLog) {
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      for (auto& c : candidates)
        logMsg += c.info;
      // NV-DXVK: dump the PS's declared SRV names once per unique PS — lets
      // us verify which name the classifier matched as albedo (RDEF draws)
      // and find names we haven't covered yet (SCORED draws). Reports both
      // classifier-picked-slot and full RDEF list.
      std::string psRdefDump;
      {
        static std::unordered_set<XXH64_hash_t> sLoggedPsRdefs;
        if (psH != 0 && sLoggedPsRdefs.insert(psH).second) {
          if (const auto* psP = ps.shader.ptr()) {
            if (const auto* cs = psP->GetCommonShader()) {
              auto names = cs->GetResourceNamesAndSlots();
              std::sort(names.begin(), names.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
              psRdefDump = str::format("\n  PS-RDEF albSlot=", mat.colorTextureSlot[0], " names: ");
              for (const auto& kv : names) {
                psRdefDump += str::format("[", kv.second, "]", kv.first, " ");
              }
            }
          }
        }
      }
      // NV-DXVK: full sampler-state dump for the albedo slot. If mip-LOD
      // range is clamped high (only coarse mips) the texture will sample
      // as flat colour even with correct UVs. If anisotropy is disabled,
      // grazing angles produce mip-blur. Both are common causes of
      // "walls look like one solid colour" in raytraced composites.
      const char* albAddrU = "?";
      std::string albSampDetail = " noSamp";
      if (mat.samplers[0].ptr()) {
        const auto& si = mat.samplers[0]->info();
        switch (si.addressModeU) {
          case VK_SAMPLER_ADDRESS_MODE_REPEAT:               albAddrU = "REPEAT"; break;
          case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:      albAddrU = "MIRROR"; break;
          case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:        albAddrU = "CLAMP"; break;
          case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:      albAddrU = "BORDER"; break;
          case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: albAddrU = "MIRCLAMP"; break;
          default: albAddrU = "UNK"; break;
        }
        const char* minFilt = (si.minFilter == VK_FILTER_NEAREST) ? "N" : "L";
        const char* magFilt = (si.magFilter == VK_FILTER_NEAREST) ? "N" : "L";
        const char* mipMode = (si.mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST) ? "N" : "L";
        albSampDetail = str::format(
          " filt=", minFilt, magFilt, mipMode,
          " lod=[", si.mipmapLodMin, "..", si.mipmapLodMax, "]",
          " bias=", si.mipmapLodBias,
          " aniso=", si.useAnisotropy ? "on" : "off",
          "/", si.maxAnisotropy);
      }
      // NV-DXVK: PS color-output flag. Draws whose PS writes nothing
      // (depth/alpha-cutout/shadow passes) shouldn't be Remix's source
      // for material colour — their bound albedo is incidental.
      const char* psOutTag = " [PS-?]";
      if (const auto* psP = ps.shader.ptr()) {
        if (const auto* cs = psP->GetCommonShader()) {
          psOutTag = cs->HasColorOutput() ? " [PS-OUT]" : " [PS-NO-OUT]";
        }
      }
      // NV-DXVK: per-unique BSP draw rasterizer/depth/blend state diagnostic.
      // Goal: verify whether TF2 actually uses depth bias / depth-func tweaks
      // / alpha blend on overlay draw calls (the rasterizer-state mechanics
      // that hide degenerate-UV decals behind proper-UV walls in the native
      // game). If different (VS,PS) pairs in BSP geometry have *different*
      // rasterizer/depth states — especially DepthBias != 0 or BlendEnable
      // — then we know overlays ARE separate draw calls with explicit
      // priority bits that we can replay on the raytracing side. If all BSP
      // draws share identical state, the ordering/separation must come from
      // somewhere else (engine code in IDA, pre-sorted draw streams, etc.).
      {
        XXH64_hash_t vsH_rs = 0, psH_rs = 0;
        GetCurrentVsPsHashes(vsH_rs, psH_rs);
        const bool isBspVs =
             vsH_rs == 0x7c38fdf4358d5527ull
          || vsH_rs == 0x0990ac503e694beeull
          || vsH_rs == 0x1953b6e9cc252e4eull
          || vsH_rs == 0xe7abcf4ea24b0fa7ull
          || vsH_rs == 0x448e372f6d5e78e1ull;
        if (isBspVs) {
          // Rasterizer state — for DepthBias / SlopeScaledDepthBias.
          // D3D11RasterizerState exposes these via GetDesc.
          float depthBias = 0.0f;
          float depthBiasClamp = 0.0f;
          float slopeBias = 0.0f;
          uint32_t fillMode = 0;
          uint32_t cullMode = 0;
          {
            D3D11RasterizerState* rs = m_context->m_state.rs.state;
            if (rs) {
              D3D11_RASTERIZER_DESC2 rd = {};
              rs->GetDesc(reinterpret_cast<D3D11_RASTERIZER_DESC*>(&rd));
              depthBias = float(rd.DepthBias);
              depthBiasClamp = rd.DepthBiasClamp;
              slopeBias = rd.SlopeScaledDepthBias;
              fillMode = uint32_t(rd.FillMode);
              cullMode = uint32_t(rd.CullMode);
            }
          }
          // Depth-stencil state — for DepthFunc / DepthWriteMask.
          uint32_t depthFunc = 0;
          uint32_t depthWriteMask = 0;
          uint32_t depthEnable = 0;
          {
            D3D11DepthStencilState* ds = m_context->m_state.om.dsState;
            if (ds) {
              D3D11_DEPTH_STENCIL_DESC dsd = {};
              ds->GetDesc(&dsd);
              depthEnable = dsd.DepthEnable ? 1 : 0;
              depthWriteMask = uint32_t(dsd.DepthWriteMask);
              depthFunc = uint32_t(dsd.DepthFunc);
            }
          }
          // Blend state — for alpha blend / write-mask / src,dst factors.
          uint32_t blendEnable = 0;
          uint32_t srcBlend = 0, dstBlend = 0, blendOp = 0;
          uint32_t writeMask = 0;
          {
            D3D11BlendState* bs = m_context->m_state.om.cbState;
            if (bs) {
              D3D11_BLEND_DESC1 bd = {};
              bs->GetDesc1(&bd);
              const auto& rt0 = bd.RenderTarget[0];
              blendEnable = rt0.BlendEnable ? 1 : 0;
              srcBlend = uint32_t(rt0.SrcBlend);
              dstBlend = uint32_t(rt0.DestBlend);
              blendOp = uint32_t(rt0.BlendOp);
              writeMask = uint32_t(rt0.RenderTargetWriteMask);
            }
          }
          // Dedup by (VS hash low + PS hash low + state fingerprint).
          const uint64_t stateKey =
            uint64_t(uint32_t(vsH_rs))
            ^ (uint64_t(uint32_t(psH_rs)) << 16)
            ^ (uint64_t(depthEnable) << 4)
            ^ (uint64_t(depthWriteMask) << 5)
            ^ (uint64_t(depthFunc) << 6)
            ^ (uint64_t(blendEnable) << 12)
            ^ (uint64_t(srcBlend) << 20)
            ^ (uint64_t(dstBlend) << 28)
            ^ (uint64_t(uint32_t(depthBias)) << 32)
            ^ (uint64_t(uint32_t(slopeBias * 1024.0f)) << 48);
          static std::unordered_set<uint64_t> sBspStateLogged;
          if (sBspStateLogged.insert(stateKey).second) {
            Logger::info(str::format(
              "[BspRastState] VS=0x", std::hex, vsH_rs,
              " PS=0x", psH_rs, std::dec,
              " depthBias=", depthBias,
              " slopeBias=", slopeBias,
              " biasClamp=", depthBiasClamp,
              " fillMode=", fillMode, " cullMode=", cullMode,
              " depthEnable=", depthEnable,
              " depthWrite=", depthWriteMask,
              " depthFunc=", depthFunc,
              " blendEnable=", blendEnable,
              " srcBlend=", srcBlend, " dstBlend=", dstBlend,
              " blendOp=", blendOp, " writeMask=", writeMask));
          }
        }
      }

      Logger::info(str::format("[D3D11Rtx] FillMaterialData draw #", s_logCount,
        " VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
        " picked ", textureID, " of ", candidates.size(), " cand, rej(nonTex2D=",
        rejNonTex2D, " tiny=", rejTiny, " nullView=", rejNullView, ")",
        psOutTag,
        rdefHit ? " [RDEF]" : " [SCORED]",
        mat.colorTextures[0].isValid() && !mat.colorTextures[0].isImageEmpty() ? " A"  : "",
        mat.normalTexture.isValid()    && !mat.normalTexture.isImageEmpty()    ? " N"  : "",
        mat.roughnessTexture.isValid() && !mat.roughnessTexture.isImageEmpty() ? " R"  : "",
        mat.metallicTexture.isValid()  && !mat.metallicTexture.isImageEmpty()  ? " M"  : "",
        mat.emissiveTexture.isValid()  && !mat.emissiveTexture.isImageEmpty()  ? " E"  : "",
        mat.ambientOcclusionTexture.isValid() && !mat.ambientOcclusionTexture.isImageEmpty() ? " AO" : "",
        mat.lightmapTexture.isValid()    && !mat.lightmapTexture.isImageEmpty()    ? " L0" : "",
        mat.lightmap2Texture.isValid()   && !mat.lightmap2Texture.isImageEmpty()   ? " L1" : "",
        mat.detailTexture.isValid()      && !mat.detailTexture.isImageEmpty()      ? " D"  : "",
        mat.cloudMaskTexture.isValid()   && !mat.cloudMaskTexture.isImageEmpty()   ? " C"  : "",
        " albHash=0x", std::hex, mat.colorTextures[0].getImageHash(), std::dec,
        " albWrap=", albAddrU, albSampDetail,
        candidates.empty() ? " [NO CANDIDATES]" : "",
        psRdefDump,
        "\n", logMsg));
      ++s_logCount;
    }

    // Material defaults for the Remix legacy material pipeline.
    // D3D11 bakes blending/alpha into immutable state objects — we extract
    // what we can from BlendState and DepthStencilState below.
    mat.textureColorArg1Source  = RtTextureArgSource::Texture;
    mat.textureColorArg2Source  = RtTextureArgSource::None;
    mat.textureColorOperation   = DxvkRtTextureOperation::Modulate;
    mat.textureAlphaArg1Source  = RtTextureArgSource::Texture;
    mat.textureAlphaArg2Source  = RtTextureArgSource::None;
    mat.textureAlphaOperation   = DxvkRtTextureOperation::SelectArg1;
    mat.tFactor                 = 0xFFFFFFFF;  // Opaque white
    mat.diffuseColorSource      = RtTextureArgSource::None;
    mat.specularColorSource     = RtTextureArgSource::None;

    // --- Blend state ---
    D3D11BlendState* blendState = m_context->m_state.om.cbState;
    if (blendState) {
      D3D11_BLEND_DESC1 blendDesc;
      blendState->GetDesc1(&blendDesc);
      const auto& rt0 = blendDesc.RenderTarget[0];

      mat.blendMode.enableBlending = rt0.BlendEnable;
      mat.blendMode.colorSrcFactor = mapD3D11Blend(rt0.SrcBlend, false);
      mat.blendMode.colorDstFactor = mapD3D11Blend(rt0.DestBlend, false);
      mat.blendMode.colorBlendOp   = mapD3D11BlendOp(rt0.BlendOp);
      mat.blendMode.alphaSrcFactor = mapD3D11Blend(rt0.SrcBlendAlpha, true);
      mat.blendMode.alphaDstFactor = mapD3D11Blend(rt0.DestBlendAlpha, true);
      mat.blendMode.alphaBlendOp   = mapD3D11BlendOp(rt0.BlendOpAlpha);
      mat.blendMode.writeMask      = rt0.RenderTargetWriteMask;

      // AlphaToCoverage = D3D11's cutout transparency (foliage, fences, hair).
      if (blendDesc.AlphaToCoverageEnable) {
        mat.alphaTestEnabled       = true;
        mat.alphaTestCompareOp     = VK_COMPARE_OP_GREATER;
        mat.alphaTestReferenceValue = 128;
      }
    }

    // --- Alpha test from depth-stencil state ---
    // Some engines use stencil ops to simulate alpha test; detect write-mask-zero
    // with stencil as a proxy for "discard if alpha < ref".
    D3D11DepthStencilState* dsState = m_context->m_state.om.dsState;
    if (dsState && !mat.alphaTestEnabled) {
      D3D11_DEPTH_STENCIL_DESC dsDesc;
      dsState->GetDesc(&dsDesc);
      if (dsDesc.StencilEnable && dsDesc.FrontFace.StencilFunc == D3D11_COMPARISON_LESS) {
        mat.alphaTestEnabled        = true;
        mat.alphaTestCompareOp      = VK_COMPARE_OP_GREATER;
        mat.alphaTestReferenceValue  = dsDesc.StencilReadMask;
      }
    }

    mat.updateCachedHash();
  }

  void D3D11Rtx::SubmitDraw(bool indexed,
                             UINT count,
                             UINT start,
                             INT  base,
                             const Matrix4* instanceTransform) {
    // NV-DXVK: cache VS hash at entry so BumpFilter() / submit tracking can
    // attribute stats without re-fetching it at every reject site.
    m_currentVsHashCache.clear();
    m_skinnedCharNeedsCamOffset = false;
    m_vmHuntIsSuspect = false;
    m_vmHuntIndexCount = 0;

    // NV-DXVK [HUD-Option5 v4]: flush a pending composite-output blit
    // FIRST. Previous SubmitDraw detected TF2's composite PS writing to
    // its 2048x1152 R8G8B8A8_SRGB output; queue a blit of our post-
    // tonemap RT over that image now. CS-thread order becomes:
    //   [prev composite draw] -> [our blit] -> [current HUD draw]
    // so HUD rasters layer on top of our RT, and TF2's present-time
    // copy carries (our RT + HUD) into the swap chain.
    if (m_compositeOutputPending != nullptr) {
      Rc<DxvkImage> dst = m_compositeOutputPending;
      m_compositeOutputPending = nullptr;
      m_context->EmitCs([dst](DxvkContext* ctx) {
        auto* rtx = static_cast<RtxContext*>(ctx);
        rtx->blitPostTonemapScratchToCompositeOut(dst);
        rtx->requestCompositeIntercept();
      });
    }

    // v4 detect: when TF2's composite VS (ca1e169b461e81ee) is bound
    // and its RT[0] is the 2048x1152 SRGB backbuffer, cache the image
    // for the next SubmitDraw's pending flush.
    {
      auto vsShader = m_context->m_state.vs.shader;
      if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
        auto& sh = vsShader->GetCommonShader()->GetShader();
        if (sh != nullptr &&
            sh->getShaderKey().toString().compare(0, 19, "VS_ca1e169b461e81ee") == 0) {
          auto* coRtv = m_context->m_state.om.renderTargetViews[0].ptr();
          Rc<DxvkImageView> coView = (coRtv != nullptr) ? coRtv->GetImageView() : nullptr;
          if (coView != nullptr && coView->image() != nullptr) {
            const auto& co = coView->image()->info();
            // NV-DXVK: hash + format + extent==viewport. The composite VS
            // hash + 8888-RGBA format already tightly identifies TF2's
            // final composite. Comparing the bound RT's extent against the
            // current viewport (= window/swap-chain size at this draw)
            // distinguishes the FINAL composite output from any same-format
            // intermediate ping-pong RT. Auto-tracks window resizes, render
            // scale changes, fullscreen toggles — no hardcoded dimensions.
            const bool fmtOk = co.format == VK_FORMAT_R8G8B8A8_SRGB
                            || co.format == VK_FORMAT_R8G8B8A8_UNORM
                            || co.format == VK_FORMAT_B8G8R8A8_SRGB
                            || co.format == VK_FORMAT_B8G8R8A8_UNORM;
            const auto& vps = m_context->m_state.rs.viewports;
            const float vw = vps[0].Width;
            const float vh = vps[0].Height;
            const bool extentMatchesViewport =
                 vw > 0.0f && vh > 0.0f
              && std::abs(vw - float(co.extent.width))  < 1.0f
              && std::abs(vh - float(co.extent.height)) < 1.0f;
            if (fmtOk && co.extent.depth == 1 && extentMatchesViewport) {
              m_compositeOutputPending = coView->image();
              m_compositeOutputThisFrame = coView->image();
              // Publish the composite extent as the canonical main-view
              // size (used by the fanout publish to identify which draws
              // belong to the player camera vs probes/shadows). Updated on
              // every detect so resolution changes auto-propagate.
              const bool extentChanged =
                   m_compositeOutputW != co.extent.width
                || m_compositeOutputH != co.extent.height;
              m_compositeOutputW = co.extent.width;
              m_compositeOutputH = co.extent.height;
              if (extentChanged) {
                Logger::info(str::format(
                  "[CompositeOut v4] composite RT captured ",
                  co.extent.width, "x", co.extent.height,
                  " (viewport ", uint32_t(vw), "x", uint32_t(vh), ")",
                  " fmt=", (int)co.format));
              }
            }
          }
        }
      }
    }

    // NV-DXVK: Standard Remix UI-hash insertion hook. Runs on every draw,
    // before the filter cascade, so it catches UI draws regardless of how
    // they would have otherwise been classified (NoLayout, UIFallback,
    // even a passed-through captured draw). Emits injectRTX into the main
    // CS chunk once per frame if any bound PS SRV is in rtx.uiTextures;
    // the caller's subsequent EmitCs(draw) for this + later UI draws then
    // lands AFTER injectRTX in CS-execution order. See the method for the
    // full rationale.
    MaybeEarlyInjectForUITexture();

    // NV-DXVK [CamCatalog]: per-frame catalog of every unique (camOrigin,
    // viewport) pair we see. Distinct cameras → we'll see distinct
    // (origin.x, origin.y, origin.z, maxZ, w, h) tuples. This tells us how
    // many cameras TF2 actually uses in one frame (main world, viewmodel,
    // shadow, etc.) and their exact parameters. Throttled to 16 unique
    // tuples per session.
    {
      const auto& vsCb2 = m_context->m_state.vs.constantBuffers[2];
      if (vsCb2.buffer != nullptr && vsCb2.buffer->Desc()->ByteWidth >= 96
          && m_context->m_state.rs.numViewports > 0) {
        const auto mapped = vsCb2.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr) {
          const size_t base = static_cast<size_t>(vsCb2.constantOffset) * 16;
          const float* f = reinterpret_cast<const float*>(ptr + base);
          const float ox = f[1], oy = f[2], oz = f[3];
          const auto& vp = m_context->m_state.rs.viewports[0];
          const float maxZ = vp.MaxDepth;
          const float vpW = vp.Width, vpH = vp.Height;
          // Integer key so we group very similar cams.
          struct CamKey { int ox, oy, oz, mz10k, w, h; };
          auto key = CamKey{
            (int)ox, (int)oy, (int)oz,
            (int)(maxZ * 10000.0f),
            (int)vpW, (int)vpH
          };
          static std::vector<CamKey> sSeen;
          bool seen = false;
          for (const auto& k : sSeen) {
            if (k.ox == key.ox && k.oy == key.oy && k.oz == key.oz
             && k.mz10k == key.mz10k && k.w == key.w && k.h == key.h) {
              seen = true; break;
            }
          }
          if (!seen && sSeen.size() < 16) {
            sSeen.push_back(key);
            std::string vsN = "null";
            auto vs = m_context->m_state.vs.shader;
            if (vs != nullptr && vs->GetCommonShader() != nullptr) {
              auto& s = vs->GetCommonShader()->GetShader();
              if (s != nullptr) vsN = s->getShaderKey().toString().substr(0, 19);
            }
            // Dump cb2[4] row3 (clip.w coefficients) to see each camera's
            // forward axis in projection.
            Logger::info(str::format(
              "[CamCatalog] #", sSeen.size(),
              " origin=(", ox, ",", oy, ",", oz, ")",
              " maxZ=", maxZ,
              " vp=(", (int)vpW, "x", (int)vpH, ")",
              " row3=(", f[16], ",", f[17], ",", f[18], ",", f[19], ")",
              " vs=", vsN));
          }
        }
      }
    }

    // NV-DXVK [VMHunt]: targeted log for suspect viewmodel draws identified
    // by index count in the game-side PIX capture. Dumps FULL per-draw
    // state: shaders, cbuffers, viewport, input layout, bound VBs/SRVs.
    // When user sees gun in game, one of these index counts is the gun.
    {
      const bool isSuspect =
          count == 17070 || count == 13293 || count == 819
       || count == 28089 || count == 22521 || count == 9306
       || count == 2562  || count == 40224 || count == 1161;
      m_vmHuntIsSuspect = isSuspect;
      m_vmHuntIndexCount = count;
      if (isSuspect) {
        static uint32_t sVmHuntLog = 0;
        if (sVmHuntLog < 40) {
          ++sVmHuntLog;
          const uint32_t fid = m_context->m_device->getCurrentFrameId();
          // VS hash
          std::string vsN = "null";
          auto vs = m_context->m_state.vs.shader;
          if (vs != nullptr && vs->GetCommonShader() != nullptr) {
            auto& s = vs->GetCommonShader()->GetShader();
            if (s != nullptr) vsN = s->getShaderKey().toString();
          }
          // PS hash
          std::string psN = "null";
          auto ps = m_context->m_state.ps.shader;
          if (ps != nullptr && ps->GetCommonShader() != nullptr) {
            auto& s = ps->GetCommonShader()->GetShader();
            if (s != nullptr) psN = s->getShaderKey().toString();
          }
          // Viewport
          float vpMin = -1, vpMax = -1, vpW = -1, vpH = -1;
          if (m_context->m_state.rs.numViewports > 0) {
            const auto& vp = m_context->m_state.rs.viewports[0];
            vpMin = vp.MinDepth; vpMax = vp.MaxDepth;
            vpW = vp.Width; vpH = vp.Height;
          }
          // Semantics
          std::string semLine;
          bool hasBI = false, hasBW = false;
          uint32_t posFmt = 0;
          auto il = m_context->m_state.ia.inputLayout.ptr();
          if (il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              semLine += str::format(" ", s.name, s.index,
                                     s.perInstance ? "/I" : "/V",
                                     ":fmt", (uint32_t)s.format,
                                     ":sl", (uint32_t)s.inputSlot,
                                     ":off", (uint32_t)s.byteOffset);
              if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0) hasBI = true;
              if (!s.perInstance && std::strncmp(s.name, "BLENDWEIGHT", 11) == 0)  hasBW = true;
              if (!s.perInstance && std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0)
                posFmt = (uint32_t)s.format;
            }
          }
          Logger::info(str::format(
            "[VMHunt] f=", fid, " count=", count, " indexed=", (indexed ? 1 : 0),
            " vs=", vsN, " ps=", psN,
            " vp=(", vpW, "x", vpH, ",[", vpMin, "..", vpMax, "])",
            " skin=", (hasBI && hasBW ? 1 : 0),
            " posFmt=", posFmt,
            " sem={", semLine, " }"));
          // cb2 dump (first 96 bytes: c_zNear, c_cameraOrigin, c_cameraRelativeToClip)
          const auto& vsCb2 = m_context->m_state.vs.constantBuffers[2];
          if (vsCb2.buffer != nullptr) {
            const auto mapped = vsCb2.buffer->GetMappedSlice();
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            if (ptr) {
              const size_t base = static_cast<size_t>(vsCb2.constantOffset) * 16;
              const float* f = reinterpret_cast<const float*>(ptr + base);
              Logger::info(str::format(
                "[VMHunt.cb2] zNear=", f[0],
                " camOrigin=(", f[1], ",", f[2], ",", f[3], ")",
                " c2c_row0=(", f[4], ",", f[5], ",", f[6], ",", f[7], ")",
                " c2c_row1=(", f[8], ",", f[9], ",", f[10], ",", f[11], ")",
                " c2c_row2=(", f[12], ",", f[13], ",", f[14], ",", f[15], ")",
                " c2c_row3=(", f[16], ",", f[17], ",", f[18], ",", f[19], ")"));
            }
          }
          // cb3 dump (first 48 bytes: CBufModelInstance objectToCameraRelative)
          const auto& vsCb3 = m_context->m_state.vs.constantBuffers[3];
          if (vsCb3.buffer != nullptr) {
            const auto mapped = vsCb3.buffer->GetMappedSlice();
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            if (ptr) {
              const size_t base = static_cast<size_t>(vsCb3.constantOffset) * 16;
              const float* f = reinterpret_cast<const float*>(ptr + base);
              Logger::info(str::format(
                "[VMHunt.cb3] o2cr_row0=(", f[0], ",", f[1], ",", f[2], ",", f[3], ")",
                " row1=(", f[4], ",", f[5], ",", f[6], ",", f[7], ")",
                " row2=(", f[8], ",", f[9], ",", f[10], ",", f[11], ")"));
            }
          }
          // VS SRV slots: which are bound?
          std::string srvLine;
          for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; ++slot) {
            auto srv = m_context->m_state.vs.shaderResources.views[slot].ptr();
            if (srv != nullptr) {
              Com<ID3D11Resource> res;
              srv->GetResource(&res);
              uint32_t sz = 0;
              D3D11_RESOURCE_DIMENSION dim;
              res->GetType(&dim);
              if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
                auto* b = static_cast<D3D11Buffer*>(res.ptr());
                sz = b->Desc()->ByteWidth;
              }
              srvLine += str::format(" t", slot, "=", sz);
            }
          }
          Logger::info(str::format("[VMHunt.srv]", srvLine));
        }
      }
    }

    // NV-DXVK [VMPass]: log EVERY draw (skinned or rigid) that happens
    // during the viewmodel viewport (MaxDepth <= 0.08). Reveals what
    // geometry the game actually submits for first-person rendering.
    {
      const float vpMaxZ = (m_context->m_state.rs.numViewports > 0)
          ? m_context->m_state.rs.viewports[0].MaxDepth : 1.0f;
      if (vpMaxZ <= 0.08f) {
        const uint32_t fid = m_context->m_device->getCurrentFrameId();
        static uint32_t sLastF = 0;
        static uint32_t sCount = 0;
        if (fid != sLastF) { sLastF = fid; sCount = 0; }
        if (sCount < 32) {
          ++sCount;
          std::string vsN = "null", psN = "null";
          auto vs = m_context->m_state.vs.shader;
          if (vs != nullptr && vs->GetCommonShader() != nullptr) {
            auto& s = vs->GetCommonShader()->GetShader();
            if (s != nullptr) vsN = s->getShaderKey().toString().substr(0, 19);
          }
          auto ps = m_context->m_state.ps.shader;
          if (ps != nullptr && ps->GetCommonShader() != nullptr) {
            auto& s = ps->GetCommonShader()->GetShader();
            if (s != nullptr) psN = s->getShaderKey().toString().substr(0, 19);
          }
          // Probe semantic layout briefly.
          bool hasBI = false, hasBW = false;
          auto il = m_context->m_state.ia.inputLayout.ptr();
          if (il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0) hasBI = true;
              if (!s.perInstance && std::strncmp(s.name, "BLENDWEIGHT", 11) == 0)  hasBW = true;
            }
          }
          Logger::info(str::format(
            "[VMPass] f=", fid,
            " vs=", vsN, " ps=", psN,
            " verts=", count, " idx=", (indexed ? 1 : 0),
            " skin=", (hasBI && hasBW ? 1 : 0),
            " vpMaxZ=", vpMaxZ));
        }
      }
    }
    const D3D11CommonShader* commonVsForLog = nullptr;
    {
      auto vsShader = m_context->m_state.vs.shader;
      if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
        commonVsForLog = vsShader->GetCommonShader();
        auto& s = commonVsForLog->GetShader();
        if (s != nullptr) m_currentVsHashCache = s->getShaderKey().toString();
      }
    }

    // NV-DXVK NPC SKINNING DIAG: record every draw against its VS hash with
    // classification so EndFrame can dump "vs=X seen=N submitted=N skinnedV=N
    // boneSrv=N". Lets us see, in one glance, which VS hashes represent
    // animated-character draws — and whether our remix pipeline processed
    // or skipped each. Populated here on EVERY draw entry, before any
    // reject/accept decision. Gated on the bone-diag switch so default runs
    // stay silent.
    if (::dxvk::tf2::boneDiagEnabled() && !m_currentVsHashCache.empty()) {
      auto& st = m_vsFrameStats[m_currentVsHashCache];
      ++st.seen;
      if (st.firstPsHash.empty()) {
        auto psPtr = m_context->m_state.ps.shader;
        if (psPtr != nullptr && psPtr->GetCommonShader() != nullptr) {
          auto& ps = psPtr->GetCommonShader()->GetShader();
          if (ps != nullptr) st.firstPsHash = ps->getShaderKey().toString().substr(0, 19);
        }
      }
      D3D11InputLayout* layout = m_context->m_state.ia.inputLayout.ptr();
      if (layout != nullptr) {
        for (const auto& sem : layout->GetRtxSemantics()) {
          if (std::strncmp(sem.name, "BLENDINDICES", 12) != 0 || sem.index != 0) continue;
          if (sem.perInstance) ++st.skinnedPerInst;
          else                 ++st.skinnedPerVert;
        }
      }
      // t30 (g_boneMatrix) / t31 (g_modelInst) SRV presence.
      auto srv30 = m_context->m_state.vs.shaderResources.views[30].ptr();
      auto srv31 = m_context->m_state.vs.shaderResources.views[31].ptr();
      if (srv30) ++st.boneSrvBound;
      if (srv31) ++st.modelInstBound;
    }

    // NV-DXVK: one-shot per-VS signature dump — list the cbuffers + SRVs the
    // shader binds + bound VB layout, so we know what each unique shader
    // looks like without having to run fxc /dumpbin on every hash. Dumped
    // exactly once per unique VS per session.
    if (!m_currentVsHashCache.empty() && commonVsForLog != nullptr) {
      const std::string shortKey = m_currentVsHashCache.substr(0, 19);
      if (m_vsRdefDumped.insert(shortKey).second) {
        std::string cbLine;
        for (uint32_t s = 0; s < 14; ++s) {
          const auto& cb = m_context->m_state.vs.constantBuffers[s];
          if (cb.buffer == nullptr) continue;
          cbLine += str::format(" cb", s, "=", cb.buffer->Desc()->ByteWidth,
                                 "@", cb.constantOffset);
        }
        std::string srvLine;
        for (uint32_t s = 0; s < 32; ++s) {
          if (s >= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) break;
          auto* srv = m_context->m_state.vs.shaderResources.views[s].ptr();
          if (!srv) continue;
          Com<ID3D11Resource> res; srv->GetResource(&res);
          auto* buf = static_cast<D3D11Buffer*>(res.ptr());
          size_t bsz = buf ? buf->Desc()->ByteWidth : 0;
          srvLine += str::format(" t", s, "=", bsz);
        }
        std::string semLine;
        auto* il = m_context->m_state.ia.inputLayout.ptr();
        if (il) {
          for (const auto& sem : il->GetRtxSemantics()) {
            semLine += str::format(" ", sem.name, sem.index,
                                    (sem.perInstance ? "/I" : "/V"),
                                    ":fmt", uint32_t(sem.format),
                                    ":slot", sem.inputSlot,
                                    ":off", sem.byteOffset);
          }
        }
        Logger::info(str::format(
          "[D3D11Rtx.vs.sig] vs=", shortKey, " cbuffers:", cbLine,
          " SRVs:", srvLine, " semantics:", semLine));
      }
    }

    // NV-DXVK: Diagnostic — confirm SubmitDraw is reached
    {
      static uint32_t sEntryLog = 0;
      if (sEntryLog < 3) {
        ++sEntryLog;
        Logger::info(str::format("[D3D11Rtx] SubmitDraw ENTERED count=", count,
          " indexed=", indexed ? 1 : 0, " raw=", m_rawDrawCount));
      }
    }

    // NV-DXVK: Previously this returned early on deferred contexts because
    // D3D11Rtx::Initialize() is only called on the immediate context, leaving
    // m_pGeometryWorkers null everywhere else.  That meant Source-engine
    // games like Titanfall 2 — which batch-record every material draw onto
    // deferred contexts via materialsystem_dx11's threaded queue — fed zero
    // geometry to the RTX pipeline: the main menu rendered as an empty
    // ray-traced clear color with no actual scene content.
    //
    // The deferred-context CS stream is already recorded into the
    // D3D11CommandList and replayed in order by D3D11ImmediateContext::
    // ExecuteCommandList, so any EmitCs callbacks posted here will run on the
    // CS thread at the correct point relative to the game's native draws.
    // The only thing we were missing was the worker pool.  Lazy-allocate one
    // on first use so the immediate context keeps its eagerly-allocated pool
    // while every deferred context gets its own on demand.
    if (m_pGeometryWorkers == nullptr) {
      const uint32_t cores = std::max(2u, std::thread::hardware_concurrency());
      const uint32_t workers = std::min(std::max(cores / 2, 2u), 6u);
      m_pGeometryWorkers = std::make_unique<GeometryProcessor>(workers, "d3d11-geometry-def");
    }

    // NV-DXVK: One-shot cbuffer dump to identify Source's projection matrix
    // layout.  Titanfall 2 gameplay frames show raw=500+ draws but every
    // single one is being rejected as UIFallback because
    // classifyPerspective() never matches any matrix in Source's cbuffers.
    //
    // Gating: trigger only once per session, on a draw that comes *late*
    // in a frame that has already seen many draws.  m_rawDrawCount is the
    // frame-level counter incremented at the top of OnDraw*/OnDrawIndexed*
    // BEFORE SubmitDraw is invoked, so by the time we reach here for the
    // Nth draw of the frame it's already >= N.  "> 300" safely clears the
    // UI frames (which top out around raw=27) and guarantees we're inside
    // a bona-fide 3D gameplay frame.  We also scan all 15 VS cbuffer slots
    // and dump 256 bytes each (enough for 2x 4x4 matrices + padding), and
    // cover the first 4 PS slots because deferred-renderer passes often
    // put the active camera in a PS cbuffer for lighting reconstruction.
    if (!m_gameplayCBuffersDumped && m_rawDrawCount > 300) {
      m_gameplayCBuffersDumped = true;
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      Logger::info(str::format(
          "[D3D11Rtx] First late-gameplay draw (count=", count,
          ", frameRawDraws=", m_rawDrawCount,
          ") -- dumping cbuffers to find Source's projection matrix layout:"));
      for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
        const auto& cb = vsCbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) {
          Logger::info(str::format(
              "[D3D11Rtx]   VS cb slot=", slot,
              " size=", cb.buffer->Desc()->ByteWidth,
              " mapPtr=NULL"));
          continue;
        }
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t dumpBytes = std::min<size_t>(256, bufSize > base ? bufSize - base : 0);
        Logger::info(str::format(
            "[D3D11Rtx]   VS cb slot=", slot,
            " size=", bufSize,
            " constOff=", base,
            " dumping=", dumpBytes, " bytes"));
        const float* f = reinterpret_cast<const float*>(ptr + base);
        for (size_t row = 0; row < dumpBytes / 16; ++row) {
          Logger::info(str::format(
              "[D3D11Rtx]     +", row * 16, ": ",
              f[row*4+0], ", ", f[row*4+1], ", ",
              f[row*4+2], ", ", f[row*4+3]));
        }
      }
      // Also dump the first 4 PS cbuffers — Source's deferred lighting
      // passes commonly stash the active scene camera in a PS cbuffer for
      // view-space reconstruction.
      const auto& psCbs = m_context->m_state.ps.constantBuffers;
      for (uint32_t slot = 0; slot < 4; ++slot) {
        const auto& cb = psCbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) continue;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t dumpBytes = std::min<size_t>(256, bufSize > base ? bufSize - base : 0);
        Logger::info(str::format(
            "[D3D11Rtx]   PS cb slot=", slot,
            " size=", bufSize,
            " dumping=", dumpBytes, " bytes"));
        const float* f = reinterpret_cast<const float*>(ptr + base);
        for (size_t row = 0; row < dumpBytes / 16; ++row) {
          Logger::info(str::format(
              "[D3D11Rtx]     +", row * 16, ": ",
              f[row*4+0], ", ", f[row*4+1], ", ",
              f[row*4+2], ", ", f[row*4+3]));
        }
      }
    }

    // Throttle: don't exceed the worker ring buffer capacity.
    // Beyond this point new futures would overwrite in-flight ones → corrupt hashes.
    if (m_drawCallID >= kMaxConcurrentDraws) {
      BumpFilter(FilterReason::Throttle);
      return;
    }

    // --- Cheap pre-filters: discard draws that cannot contribute to raytracing ---

    // Only triangle topologies are raytraceable. Skip points, lines, patch lists, etc.
    // This check is first: it costs a single comparison before any other state is read.
    const D3D11_PRIMITIVE_TOPOLOGY d3dTopology = m_context->m_state.ia.primitiveTopology;
    if (d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST &&
        d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP) {
      BumpFilter(FilterReason::NonTriTopology);
      return;
    }

    // Skip depth-only passes: no pixel shader means depth prepass or shadow map.
    // Most engines draw opaque geometry twice — once for depth prepass (PS == null)
    // and once for the color pass (PS != null) with the same vertices.
    if (m_context->m_state.ps.shader == nullptr) {
      BumpFilter(FilterReason::NoPixelShader);
      return;
    }

    // Skip draws with no color render target (shadow maps, depth-only, auxiliary passes).
    // NV-DXVK: Source-engine games (Titanfall 2) bind render targets to
    // non-zero slots — the old "check slot 0 only" heuristic rejected every
    // single menu draw because slot 0 was null even though slots 1–N were
    // bound.  Scan every MRT slot and keep the draw if any slot holds a
    // valid RTV, matching what FillMaterialData() below does.
    {
      bool anyRtvBound = false;
      for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
        if (m_context->m_state.om.renderTargetViews[rt].ptr() != nullptr) {
          anyRtvBound = true;
          break;
        }
      }
      if (!anyRtvBound) {
        BumpFilter(FilterReason::NoRenderTarget);
        return;
      }
    }

    // Skip trivially small draws (< 3 elements = 0 triangles).
    if (count < 3) {
      BumpFilter(FilterReason::CountTooSmall);
      return;
    }

    // Read actual depth/stencil state from the OM — don't hardcode.
    bool zEnable = true;
    bool zWriteEnable = true;
    bool stencilEnabled = false;
    D3D11DepthStencilState* dsState = m_context->m_state.om.dsState;
    if (dsState) {
      D3D11_DEPTH_STENCIL_DESC dsDesc;
      dsState->GetDesc(&dsDesc);
      zEnable         = dsDesc.DepthEnable != FALSE;
      zWriteEnable    = dsDesc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ZERO;
      stencilEnabled  = dsDesc.StencilEnable != FALSE;
    }

    // Skip fullscreen quad / postprocess draws: depth disabled + 6 or fewer
    // elements (a fullscreen triangle or quad) + no depth write.
    // Only skip if BOTH depth test and write are off — some engines do
    // "depth off, write on" for sky or "depth on, write off" for decals.
    if (!zEnable && !zWriteEnable && count <= 6) {
      BumpFilter(FilterReason::FullscreenQuad);
      // Flag for native raster: these draws ARE UI/HUD/postprocess and
      // must rasterize natively once gameplay has made Remix active on
      // the frame; otherwise the menu/HUD never reaches the backbuffer.
      m_lastDrawFilteredAsUI = true;
      return;
    }

    D3D11InputLayout* layout = m_context->m_state.ia.inputLayout.ptr();
    if (!layout) {
      BumpFilter(FilterReason::NoInputLayout);
      m_lastDrawFilteredAsUI = true;
      // NV-DXVK: VGUI/HUD draws bind no input layout — definite UI.
      m_lastDrawIsHudClass = true;
      LogPsHashesForHudFilter("NoLayout");
      return;
    }

    const auto& semantics = layout->GetRtxSemantics();

    if (semantics.empty()) {
      BumpFilter(FilterReason::NoSemantics);
      m_lastDrawFilteredAsUI = true;
      // NV-DXVK: layout present but with no semantics is the same VGUI
      // pattern (immediate-mode quad batcher) — also definite UI.
      m_lastDrawIsHudClass = true;
      LogPsHashesForHudFilter("NoSemantics");
      return;
    }

    const D3D11RtxSemantic* posSem = nullptr;
    const D3D11RtxSemantic* nrmSem = nullptr;
    const D3D11RtxSemantic* tcSem  = nullptr;
    // NV-DXVK: TEXCOORD1 — Source/TF2 wall VSes use this for the lightmap
    // atlas UV. When present, the interleaver decodes it and surface_interaction
    // interpolates it per-hit so the lightmap sampler can see real lightmap UVs
    // instead of the albedo's tiling UVs.
    const D3D11RtxSemantic* tc1Sem = nullptr;
    // NV-DXVK: TF2 worldspace VGUI int4 stream (semantic-named TEXCOORD3 in
    // the input layout, an R32G32B32A32_SINT attribute carrying packed
    // glyph/style/image indices). Captured here so the geometry build path
    // can hand it to the interleaver if FillMaterialData later flags this
    // draw as VGUI (sourceIsUnlitUI). See the VGUI plumb in
    // rtx_materials.cpp::LegacyMaterialData::as<OpaqueMaterialData>().
    const D3D11RtxSemantic* vguiTc3Sem = nullptr;
    // NV-DXVK: TF2 worldspace VGUI also has TEXCOORD1 (R32G32B32A32_SFLOAT,
    // 4 floats covering primary glyph quad pos in xy + secondary quad pos in
    // zw) and TEXCOORD2 (R32G32_SFLOAT, glyph dimensions). The standard
    // TEXCOORD0 doesn't exist on VGUI shaders. We need explicit captures so
    // the interleaver can: (a) write TC1.xy into the texcoord-0 slot of the
    // interleaved per-vertex output (so surfaceInteraction.textureCoordinates
    // = primary glyph quad pos), (b) write TC1.zw into the VGUI extras slot
    // for secondaryQuadPos, (c) write TC2.xy into the VGUI extras slot for
    // glyphDims. Captured here regardless of VGUI detection because the
    // detection lives in FillMaterialData (later); the captures are cheap.
    const D3D11RtxSemantic* vguiTc1Sem = nullptr;
    const D3D11RtxSemantic* vguiTc2Sem = nullptr;
    const D3D11RtxSemantic* colSem = nullptr;
    const D3D11RtxSemantic* bwSem  = nullptr; // BLENDWEIGHT  — per-vertex bone weights
    const D3D11RtxSemantic* biSem  = nullptr; // BLENDINDICES — per-vertex bone indices

    static auto isTexcoordFmt = [](VkFormat f) {
      return f == VK_FORMAT_R32G32_SFLOAT        // 103 — standard 2-float UVs
          || f == VK_FORMAT_R16G16_SFLOAT         // 83  — half-float UVs
          || f == VK_FORMAT_R16G16_UNORM          // 77  — normalized 16-bit UVs (UE4, Unity HDRP)
          || f == VK_FORMAT_R16G16_SNORM          // 79  — signed normalized (some console ports)
          || f == VK_FORMAT_R8G8_UNORM;           // 16  — 8-bit packed UVs (mobile ports)
    };

    for (const auto& s : semantics) {
      if (s.perInstance) continue; // Skip per-instance data — only per-vertex geometry
      // Standard D3D semantic names
      if      (!posSem && std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0)
        posSem = &s;
      else if (!nrmSem && std::strncmp(s.name, "NORMAL",   6) == 0 && s.index == 0)
        nrmSem = &s;
      else if (!tcSem  && std::strncmp(s.name, "TEXCOORD", 8) == 0 && s.index == 0)
        tcSem  = &s;
      else if (!tc1Sem && std::strncmp(s.name, "TEXCOORD", 8) == 0 && s.index == 1)
        tc1Sem = &s;
      else if (!colSem && std::strncmp(s.name, "COLOR",    5) == 0 && s.index == 0)
        colSem = &s;
      else if (!bwSem  && std::strncmp(s.name, "BLENDWEIGHT",  11) == 0 && s.index == 0)
        bwSem  = &s;
      else if (!biSem  && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0)
        biSem  = &s;
      // NV-DXVK: TF2 VGUI TEXCOORD3 — int4 packed glyph/style/image indices.
      // Match by (name, index) only. The handoff claimed format=SINT (108)
      // but the actual TF2 build appears to use a different format —
      // gating on SINT was making this matcher silently miss. Downstream
      // vguiTexcoord3Buffer use is already gated on sourceIsUnlitUI, so a
      // non-VGUI shader that happens to declare a TEXCOORD3 won't actually
      // have its data read.
      else if (!vguiTc3Sem
            && std::strncmp(s.name, "TEXCOORD", 8) == 0
            && s.index == 3)
        vguiTc3Sem = &s;
      // NV-DXVK: TF2 VGUI TC1 — primary+secondary quad pos. Match by
      // (name, index) only — the format is empirically R32G32B32A32_SFLOAT
      // per the handoff but TF2 variants might ship slightly different
      // formats. The downstream guard is vguiTc3Sem (the SINT int4 stream
      // is a hard signature for VGUI), so a false-positive on TC1 alone
      // is harmless.
      else if (!vguiTc1Sem
            && std::strncmp(s.name, "TEXCOORD", 8) == 0
            && s.index == 1)
        vguiTc1Sem = &s;
      // NV-DXVK: TF2 VGUI TC2 — glyph dimensions. Same (name, index) match
      // gated by vguiTc3Sem downstream.
      else if (!vguiTc2Sem
            && std::strncmp(s.name, "TEXCOORD", 8) == 0
            && s.index == 2)
        vguiTc2Sem = &s;
    }

    // Fallback: accept TEXCOORD at any semantic index (some engines use
    // TEXCOORD1+ for primary UVs or start UV numbering at 1).
    if (!tcSem) {
      for (const auto& s : semantics) {
        if (std::strncmp(s.name, "TEXCOORD", 8) == 0) {
          tcSem = &s;
          break;
        }
      }
    }

    // Fallback: some engines use generic ATTRIBUTE semantics instead
    // of POSITION/NORMAL/TEXCOORD.  Identify by format heuristics.
    if (!posSem) {
      static auto isPositionFmt = [](VkFormat f) {
        return f == VK_FORMAT_R32G32B32_SFLOAT     // 106
            || f == VK_FORMAT_R32G32B32A32_SFLOAT  // 109
            || f == static_cast<VkFormat>(97);     // R16G16B16A16_SFLOAT
      };
      static auto isNormalFmt = [](VkFormat f) {
        return f == VK_FORMAT_R8G8B8A8_UNORM                     // 37
            || f == static_cast<VkFormat>(65); // A2B10G10R10_SNORM_PACK32
      };
      for (const auto& s : semantics) {
        if (s.perInstance) continue;
        if (std::strncmp(s.name, "ATTRIBUTE", 9) != 0) continue;
        if      (!posSem && isPositionFmt(s.format)) posSem = &s;
        else if (!tcSem  && isTexcoordFmt(s.format)) tcSem  = &s;
        else if (!nrmSem && isNormalFmt(s.format))   nrmSem = &s;
      }
    }

    // Format-based UV fallback: position was found by name but texcoord
    // wasn't (non-standard semantic name, custom engine, emulator port).
    // Scan remaining unmatched semantics for a 2-component float format.
    if (posSem && !tcSem) {
      for (const auto& s : semantics) {
        if (s.perInstance) continue;
        if (&s == posSem || &s == nrmSem || &s == colSem) continue;
        if (std::strncmp(s.name, "SV_", 3) == 0) continue;
        if (isTexcoordFmt(s.format)) {
          tcSem = &s;
          break;
        }
      }
    }

    if (!posSem) {
      BumpFilter(FilterReason::NoPosition);
      return;
    }

    // Log vertex layout once per VS-hash when texcoord is missing — diagnose UV issues.
    // NV-DXVK TF2: prior version was capped at 3 total hits, which buried
    // character-VS misses behind 3 unrelated POSITION-only depth-prepass
    // VSes. Now deduped per-VS so every unique offending shader logs its
    // full input layout exactly once.
    //
    // For the TF2 character signature (POSITION fmt=101 + BLENDWEIGHT@8
    // fmt=82 + BLENDINDICES@12 fmt=41, stride=28), also dump the first 28
    // raw bytes of vertex 0 from the bound VB at slot 0 — interpreting them
    // as 7 uint32s, 14 int16s, 14 uint16s, and 7 floats. Bytes 16-27 contain
    // the UV (and probably normal) data the IL doesn't expose. Identifying
    // the UV byte offset + format from this dump is the last unknown for
    // synthesizing geo.texcoordBuffer.
    if (!tcSem) {
      static std::mutex sNoTcMu;
      static std::unordered_set<XXH64_hash_t> sNoTcLoggedVs;
      static std::atomic<uint32_t> sNoTcLines { 0 };
      const uint32_t kMaxNoTcLines = 200;

      if (sNoTcLines.load() < kMaxNoTcLines) {
        XXH64_hash_t vsH = 0, psH = 0;
        GetCurrentVsPsHashes(vsH, psH);
        std::lock_guard<std::mutex> lock(sNoTcMu);
        if (sNoTcLoggedVs.insert(vsH).second) {
          sNoTcLines.fetch_add(1);
          Logger::info(str::format("[D3D11Rtx] SubmitDraw: no TEXCOORD found. VS=0x", std::hex,
                                   uint64_t(vsH), " PS=0x", uint64_t(psH), std::dec,
                                   " Layout has ", semantics.size(), " semantics:"));
          for (const auto& s : semantics) {
            Logger::info(str::format("[D3D11Rtx]   name=", s.name, " idx=", s.index,
                                     " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
                                     " offset=", s.byteOffset,
                                     " perInst=", (s.perInstance ? 1 : 0)));
          }

          // TF2 character signature detection + raw VB dump.
          bool hasPos101AtSlot0Off0 = false;
          bool hasBwAtOff8 = false;
          bool hasBiAtOff12 = false;
          uint32_t uvSlot = 0;
          for (const auto& s : semantics) {
            if (std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0
                && s.inputSlot == 0 && s.byteOffset == 0
                && s.format == VK_FORMAT_R32G32_UINT) { hasPos101AtSlot0Off0 = true; uvSlot = 0; }
            if (std::strncmp(s.name, "BLENDWEIGHT", 11) == 0 && s.byteOffset == 8) hasBwAtOff8 = true;
            if (std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.byteOffset == 12) hasBiAtOff12 = true;
          }
          const bool isCharSig = hasPos101AtSlot0Off0 && hasBwAtOff8 && hasBiAtOff12;
          if (isCharSig) {
            const auto& vb = m_context->m_state.ia.vertexBuffers[uvSlot];
            const uint32_t stride = vb.stride;
            if (vb.buffer != nullptr && stride >= 28) {
              // Use the same RasterBuffer::mapPtr() path that posBuffer uses
              // for NDC-quad detection (line ~8027). That goes through dxvk's
              // persistent slice mapping, which works for both static and
              // dynamic buffers; vb.buffer->GetMappedSlice() returns null for
              // dynamic buffers between Map/Unmap and missed our window.
              DxvkBufferSlice slice = vb.buffer->GetBufferSlice(vb.offset);
              RasterBuffer probe(slice, 0, stride, VK_FORMAT_UNDEFINED);
              const uint8_t* v0 = reinterpret_cast<const uint8_t*>(
                probe.mapPtr(probe.offsetFromSlice()));
              if (v0 != nullptr) {
                // Dump 28 bytes of vertex 0 as multiple interpretations.
                uint32_t u32[7]; std::memcpy(u32, v0, 28);
                int16_t  i16[14]; std::memcpy(i16, v0, 28);
                uint16_t u16[14]; std::memcpy(u16, v0, 28);
                float    f32[7]; std::memcpy(f32, v0, 28);
                Logger::info(str::format(
                  "[CharVB.v0] VS=0x", std::hex, uint64_t(vsH), std::dec,
                  " stride=", stride,
                  std::hex,
                  " u32=[", u32[0], " ", u32[1], " ", u32[2], " ", u32[3], " ", u32[4], " ", u32[5], " ", u32[6], "]",
                  std::dec));
                Logger::info(str::format(
                  "[CharVB.v0] i16=[", i16[0], " ", i16[1], " ", i16[2], " ", i16[3], " ", i16[4], " ", i16[5], " ", i16[6],
                  " ", i16[7], " ", i16[8], " ", i16[9], " ", i16[10], " ", i16[11], " ", i16[12], " ", i16[13], "]"));
                Logger::info(str::format(
                  "[CharVB.v0] u16=[", u16[0], " ", u16[1], " ", u16[2], " ", u16[3], " ", u16[4], " ", u16[5], " ", u16[6],
                  " ", u16[7], " ", u16[8], " ", u16[9], " ", u16[10], " ", u16[11], " ", u16[12], " ", u16[13], "]"));
                Logger::info(str::format(
                  "[CharVB.v0] f32=[", f32[0], " ", f32[1], " ", f32[2], " ", f32[3], " ", f32[4], " ", f32[5], " ", f32[6], "]"));
                // Interpret bytes 16..27 specifically — the unknown 12-byte tail.
                // These are the only candidates for UV (and possibly normal).
                // Float UV would land in f32[4]/f32[5] (offsets 16/20 if R32G32_SFLOAT)
                // or f32[5]/f32[6] (offsets 20/24).
                // Half UV would land in u16[8..13] / i16[8..13] (offsets 16..27).
                // Real UVs typically fall in [0..1] for floats, or [-2^15..2^15-1]
                // mapped to [0..1] via x/32768.0 for SNORM halves.
              } else {
                Logger::info(str::format(
                  "[CharVB.v0] VS=0x", std::hex, uint64_t(vsH), std::dec,
                  " stride=", stride, " — VB mapPtr null (dynamic discard?), can't read"));
              }
            }
          }
        }
      }
    }

    // NV-DXVK TF2 character depth-prepass / VSM filter.
    //
    // Both the lit-pass and depth-pass character draws share a 28-byte
    // skinned VB and the same vertex content. The DIFFERENCE is the IL:
    //   Depth-pass IL: POSITION(R32G32_UINT)@0 + BLENDWEIGHT(R16G16_SINT)@8
    //                + BLENDINDICES(R8G8B8A8_UINT)@12 — declares 16 bytes,
    //                  ignores offsets 16..27 (no NORMAL, no TEXCOORD).
    //   Lit-pass  IL: ... + NORMAL(uint x)@3 + TEXCOORD(float xy)@4 — full 28.
    //
    // Source: fxc /dumpbin of the dumped .dxbc files.
    //   Depth-pass VSes: 3ad96dddc6600325, ae99368f58913a2e
    //   Lit-pass   VS:  ef94e6c7fcc3c144
    //
    // The depth-pass draw enters the path tracer pipeline alongside the
    // lit-pass draw at the same world position. Because its IL has no
    // TEXCOORD, hasTextureCoordinates() returns false, trackTexture drops
    // the albedo bind, surface material ends up with no albedo, BLAS
    // hit returns flat white. Result: white "ghost" character submeshes
    // overlapping the lit-pass character — the user-reported bug.
    //
    // The depth-pass produces no user-visible color; it only writes Z /
    // VSM. Filtering it out before BLAS submission removes the white
    // ghost without affecting the visible lit-pass render.
    //
    // Filter signature is the depth-pass IL exactly. The lit-pass IL has
    // NORMAL or TEXCOORD or both, so it cannot match. Non-character
    // depth draws (POSITION-only without BLEND inputs) won't match either
    // because they lack BLENDWEIGHT/BLENDINDICES.
    {
      bool hasPos101AtSlot0Off0 = false;
      bool hasBwAtOff8 = false;
      bool hasBiAtOff12 = false;
      bool hasNormal = false;
      bool hasTexcoord = (tcSem != nullptr);
      for (const auto& s : semantics) {
        if (std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0
            && s.inputSlot == 0 && s.byteOffset == 0
            && s.format == VK_FORMAT_R32G32_UINT) hasPos101AtSlot0Off0 = true;
        if (std::strncmp(s.name, "BLENDWEIGHT", 11) == 0 && s.byteOffset == 8) hasBwAtOff8 = true;
        if (std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.byteOffset == 12) hasBiAtOff12 = true;
        if (std::strncmp(s.name, "NORMAL", 6) == 0) hasNormal = true;
      }
      const bool isCharDepthPrepass =
          hasPos101AtSlot0Off0 && hasBwAtOff8 && hasBiAtOff12
          && !hasTexcoord && !hasNormal;
      if (isCharDepthPrepass) {
        // One-shot-per-VS log so we can confirm the filter triggers on the
        // expected set and not unexpectedly elsewhere.
        static std::mutex sCharFilterMu;
        static std::unordered_set<XXH64_hash_t> sLoggedFilteredVs;
        XXH64_hash_t vsH = 0, psH = 0;
        GetCurrentVsPsHashes(vsH, psH);
        bool firstSeen = false;
        {
          std::lock_guard<std::mutex> lk(sCharFilterMu);
          firstSeen = sLoggedFilteredVs.insert(vsH).second;
        }
        if (firstSeen) {
          Logger::info(str::format(
            "[CharDepthFilter] rejecting depth-prepass character draw VS=0x",
            std::hex, uint64_t(vsH), " PS=0x", uint64_t(psH), std::dec,
            " (matches POSITION fmt=101 + BLENDWEIGHT@8 + BLENDINDICES@12 + no TEXCOORD + no NORMAL)"));
        }
        ++m_filterCounts[static_cast<uint32_t>(FilterReason::CharDepthPrepass)];
        return;
      }
    }

    // Skip 2D UI/HUD draws: if position is R32G32_SFLOAT it is in screen/clip space,
    // not world space, and cannot be raytraced.
    if (posSem->format == VK_FORMAT_R32G32_SFLOAT) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::Position2D)];
      return;
    }

    // NV-DXVK: Skip draws whose position format is not supported by
    // Remix's geometry interleaver.  Unsupported formats (e.g.
    // VK_FORMAT_R32G32_UINT = 101, which Source binds for compute-
    // style vertex readback passes) produce garbage positions that
    // build degenerate BLAS entries with NaN triangles → the GPU hangs
    // forever traversing them → TDR / VK_ERROR_DEVICE_LOST.  Only
    // accept formats the interleaver can actually convert to valid
    // float3 world-space positions.
    {
      const VkFormat pf = posSem->format;
      const bool supportedPosFmt =
          pf == VK_FORMAT_R32G32B32_SFLOAT       // 106 — standard 3-float
       || pf == VK_FORMAT_R32G32B32A32_SFLOAT    // 109 — 4-float (w ignored)
       || pf == VK_FORMAT_R16G16B16A16_SFLOAT    // 97  — half-float 4-component
       || pf == VK_FORMAT_R16G16B16_SFLOAT       // 90  — half-float 3-component
       || pf == VK_FORMAT_R32G32_UINT           // 101 — Source Engine 2 quantized positions
       ;
      if (!supportedPosFmt) {
        // NV-DXVK: Dump FULL input layout + vertex buffer info for R32G32_UINT draws
        static uint32_t sUnsupDiagCount = 0;
        if (pf == VK_FORMAT_R32G32_UINT && sUnsupDiagCount < 5) {
          ++sUnsupDiagCount;
          // Log all semantics in this layout
          Logger::info(str::format(
            "[D3D11Rtx] R32G32_UINT layout diag (", semantics.size(), " semantics, ",
            count, " verts):"));
          for (const auto& s : semantics) {
            Logger::info(str::format(
              "[D3D11Rtx]   elem: name=", s.name, " idx=", s.index,
              " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
              " off=", s.byteOffset, " perInst=", s.perInstance ? 1 : 0));
          }
          // Log all bound vertex buffers for slots 0-3
          for (uint32_t sl = 0; sl < 4; ++sl) {
            const auto& vb = m_context->m_state.ia.vertexBuffers[sl];
            if (vb.buffer != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   vbuf[", sl, "]: stride=", vb.stride,
                " offset=", vb.offset,
                " size=", vb.buffer->Desc()->ByteWidth,
                " usage=", uint32_t(vb.buffer->Desc()->Usage)));
            }
          }
          // Log VS shader info if available
          if (m_context->m_state.vs.shader != nullptr) {
            Logger::info(str::format(
              "[D3D11Rtx]   VS bound: yes"));
          }
          // Log VS cbuffers for transform inspection
          const auto& vsCbs = m_context->m_state.vs.constantBuffers;
          for (uint32_t sl = 0; sl < 8; ++sl) {
            if (vsCbs[sl].buffer != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   VS cb[", sl, "]: size=", vsCbs[sl].buffer->Desc()->ByteWidth,
                " off=", vsCbs[sl].constantOffset));
            }
          }
          // Log VS SRVs (structured buffers that might contain vertex data for GPU pulling)
          for (uint32_t sl = 0; sl < 8; ++sl) {
            const auto& srv = m_context->m_state.vs.shaderResources.views[sl];
            if (srv.ptr() != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   VS srv[", sl, "]: bound"));
            }
          }
        }
        BumpFilter(FilterReason::UnsupPosFmt);
        static uint32_t sUnsupPosLog = 0;
        if (sUnsupPosLog < 3) {
          ++sUnsupPosLog;
          Logger::warn(str::format(
              "[D3D11Rtx] Skipping draw with unsupported position format ",
              static_cast<uint32_t>(pf),
              " — only R32G32B32[A32]_SFLOAT and R16G16B16[A16]_SFLOAT "
              "are supported by the interleaver."));
        }
        return;
      }
    }

    auto makeVertexBuffer = [&](const D3D11RtxSemantic* sem) -> RasterBuffer {
      if (!sem)
        return RasterBuffer();
      const auto& vb = m_context->m_state.ia.vertexBuffers[sem->inputSlot];
      if (vb.buffer == nullptr)
        return RasterBuffer();
      DxvkBufferSlice slice = vb.buffer->GetBufferSlice(vb.offset);
      return RasterBuffer(slice, sem->byteOffset, vb.stride, sem->format);
    };

    RasterBuffer posBuffer = makeVertexBuffer(posSem);
    if (!posBuffer.defined()) {
      BumpFilter(FilterReason::NoPosBuffer);
      return;
    }

    // Detect NDC-space screen quads early (but defer the actual rejection until
    // after ExtractTransforms so the VP cache gets populated from these draws).
    bool isNdcScreenQuad = false;
    if (count <= 6 && posSem->format == VK_FORMAT_R32G32B32_SFLOAT) {
      const float* p = reinterpret_cast<const float*>(
        posBuffer.mapPtr(posBuffer.offsetFromSlice()));
      if (p && std::abs(p[0]) <= 1.5f && std::abs(p[1]) <= 1.5f && std::abs(p[2]) <= 1.0f)
        isNdcScreenQuad = true;
    }

    // Normal buffer: only submit if enabled and the interleaver can convert.
    // Supported: R16G16_SFLOAT(83), R32G32_SFLOAT(103), R32G32B32_SFLOAT(106),
    // R32G32B32A32_SFLOAT(109), R8G8B8A8_UNORM(37), A2B10G10R10_SNORM(65).
    // D3D11 normals are often R16G16B16A16_SFLOAT(97) or R16G16B16A16_SNORM(98)
    // which the interleaver rejects.  Remix regenerates normals when absent.
    RasterBuffer nrmBuffer;
    if (nrmSem && RtxOptions::useInputAssemblerNormals()) {
      VkFormat nf = nrmSem->format;
      if (nf == VK_FORMAT_R8G8B8A8_UNORM
       || nf == VK_FORMAT_R32G32B32_SFLOAT
       || nf == VK_FORMAT_R32G32B32A32_SFLOAT
       || nf == VK_FORMAT_R32G32_SFLOAT
       || nf == VK_FORMAT_R16G16_SFLOAT
       || nf == static_cast<VkFormat>(65)) {  // A2B10G10R10_SNORM_PACK32
        nrmBuffer = makeVertexBuffer(nrmSem);
      }
    }
    RasterBuffer tcBuffer  = makeVertexBuffer(tcSem);
    // NV-DXVK TF2 "white character parts" diagnostic.
    // RtxWhiteDiag2 in scene_manager confirmed surfEmpty=1 / tcDef=0 for
    // skinned character draws with valid LegacyMaterialData albedo. Root
    // cause is here: tcSem is non-null (input layout DOES declare TEXCOORD)
    // but makeVertexBuffer returned an undefined RasterBuffer, which means
    // the D3D11 vertex buffer at tcSem->inputSlot is null for this draw.
    // Log the VS+slot+format combo once per (VS,slot) so we can tell exactly
    // which shader/layout has the missing VB binding. Throttled hard so a
    // stable bug produces a small, finite trail.
    if (tcSem != nullptr && !tcBuffer.defined()) {
      static std::mutex sTcDropMtx;
      static std::unordered_set<uint64_t> sTcDropLogged;
      static std::atomic<uint32_t> sTcDropLines { 0 };
      const uint32_t kMaxTcDropLines = 200;

      if (sTcDropLines.load() < kMaxTcDropLines) {
        XXH64_hash_t vsH = 0, psH = 0;
        GetCurrentVsPsHashes(vsH, psH);
        const auto& vbAtSlot = m_context->m_state.ia.vertexBuffers[tcSem->inputSlot];
        const bool vbNull = (vbAtSlot.buffer == nullptr);
        const uint64_t key = uint64_t(vsH) ^ (uint64_t(tcSem->inputSlot) << 56) ^ (uint64_t(tcSem->format) << 48);

        std::lock_guard<std::mutex> lock(sTcDropMtx);
        if (sTcDropLogged.insert(key).second) {
          sTcDropLines.fetch_add(1);
          Logger::info(str::format(
            "[RtxTcDrop] VS=0x", std::hex, uint64_t(vsH),
            " PS=0x", uint64_t(psH), std::dec,
            " tcSlot=", uint32_t(tcSem->inputSlot),
            " tcByteOff=", uint32_t(tcSem->byteOffset),
            " tcFmt=", uint32_t(tcSem->format),
            " tcSemIdx=", uint32_t(tcSem->index),
            " vbNullAtSlot=", (vbNull ? 1 : 0),
            " vbStride=", (vbNull ? 0u : vbAtSlot.stride),
            " vbOffset=", (vbNull ? 0u : vbAtSlot.offset)));
        }
      }
    }
    // NV-DXVK: TEXCOORD1 (lightmap UV). Source/TF2 wall VSes declare a second
    // TEXCOORD attribute that the interleaver decodes via `* 1/65535` (uint
    // formats) or passes through (float formats). When the layout has no
    // TEXCOORD1, tc1Buffer stays undefined and the entire lightmap path
    // is bypassed downstream.
    RasterBuffer tc1Buffer = makeVertexBuffer(tc1Sem);

    // NV-DXVK: log every distinct VS that exposes a TEXCOORD1 once so we can
    // confirm the IA capture side of the lightmap plumbing is firing on the
    // expected wall VS family. Companion logs: [TC1Interleave] (rtx_geometry_utils)
    // when the interleaver runs the lightmap decode, [TC1Surface]
    // (rtx_instance_manager) when the surface flag is set.
    if (tc1Sem != nullptr) {
      static std::unordered_set<XXH64_hash_t> sLoggedTc1VsHashes;
      static std::mutex sLoggedTc1VsMu;
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      bool firstSeen = false;
      {
        std::lock_guard<std::mutex> lk(sLoggedTc1VsMu);
        firstSeen = sLoggedTc1VsHashes.insert(vsH).second;
      }
      if (firstSeen) {
        Logger::info(str::format(
          "[TC1Detect] VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
          " sem=", tc1Sem->name, tc1Sem->index,
          " inputSlot=", tc1Sem->inputSlot,
          " byteOffset=", tc1Sem->byteOffset,
          " fmt=", uint32_t(tc1Sem->format),
          " (101=R32G32_UINT decoded as *1/65535; 103=R32G32_SFLOAT passthrough)"));
      }
    }

    // NV-DXVK: dump the first few UV values from BSP-like draws so we can
    // see whether VB UVs are normalized [0,1] (prop style) or world-scale
    // (BSP style). Gated on gameplay + throttled per-VS-hash so we don't
    // spam. Reads via whichever path works (CPU-mapped, GetMappedSlice,
    // or IMMUTABLE CPU-side cache — BSP VBs are typically IMMUTABLE).
    if (tcSem && m_rawDrawCount > 50) {
      static std::unordered_set<XXH64_hash_t> sLoggedUvVsHashes;
      static std::atomic<uint32_t> sUvDumpCount{0};
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      if (sUvDumpCount.load() < 30 && sLoggedUvVsHashes.insert(vsH).second) {
        ++sUvDumpCount;
        const auto& tcVb = m_context->m_state.ia.vertexBuffers[tcSem->inputSlot];
        const uint8_t* bytes = nullptr;
        size_t         bytesLen = 0;
        const char*    source = "none";
        if (tcVb.buffer != nullptr) {
          const auto& imm = tcVb.buffer->GetImmutableData();
          if (!imm.empty()) {
            bytes = imm.data();
            bytesLen = imm.size();
            source = "IMMUTABLE";
          } else {
            const auto mapped = tcVb.buffer->GetMappedSlice();
            if (mapped.mapPtr) {
              bytes = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              bytesLen = mapped.length;
              source = "MAPPED";
            } else {
              void* p = tcVb.buffer->GetBuffer()->mapPtr(0);
              if (p) {
                bytes = reinterpret_cast<const uint8_t*>(p);
                bytesLen = tcVb.buffer->GetBuffer()->info().size;
                source = "GETBUF";
              }
            }
          }
        }
        const size_t vbOffset = tcVb.buffer.ptr() ? tcVb.offset : 0;
        const uint32_t stride = tcVb.stride;
        const VkFormat fmt    = tcSem->format;
        if (bytes && stride > 0 && fmt == VK_FORMAT_R32G32_SFLOAT) {
          const uint32_t kNumToDump = std::min(4u, count);
          std::string dump;
          bool overran = false;
          for (uint32_t v = 0; v < kNumToDump; ++v) {
            const size_t byteAt = vbOffset + tcSem->byteOffset + size_t(v) * stride;
            if (byteAt + 8 > bytesLen) { overran = true; break; }
            float u = 0, vv = 0;
            std::memcpy(&u,  bytes + byteAt,     4);
            std::memcpy(&vv, bytes + byteAt + 4, 4);
            dump += str::format("(", u, ",", vv, ") ");
          }
          Logger::info(str::format(
            "[D3D11Rtx.UVdump] VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
            " src=", source, " stride=", stride,
            " vbOff=", vbOffset, " tcSemOff=", tcSem->byteOffset,
            " bufLen=", bytesLen,
            overran ? " [OVERRUN]" : "",
            " first4UVs: ", dump));
        } else {
          Logger::info(str::format(
            "[D3D11Rtx.UVdump] VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
            " src=", source, " stride=", stride, " fmt=", uint32_t(fmt),
            " NO-READ (buffer not CPU-readable)"));
        }
      }
    }

    // Color0: the interleaver converts BGRA and RGBA packed-byte formats.
    // Both B8G8R8A8_UNORM (D3D9 D3DCOLOR) and R8G8B8A8_UNORM (D3D11) are
    // supported — the interleaver swaps R/B for RGBA.  Float vertex color
    // formats are not supported; Remix defaults to white when color0 is absent.
    RasterBuffer colBuffer;
    if (colSem && (colSem->format == VK_FORMAT_B8G8R8A8_UNORM
                || colSem->format == VK_FORMAT_R8G8B8A8_UNORM)) {
      colBuffer = makeVertexBuffer(colSem);
    }

    // NV-DXVK start: Per-vertex skinning buffers (BLENDWEIGHT + BLENDINDICES)
    RasterBuffer bwBuffer  = makeVertexBuffer(bwSem);
    RasterBuffer biBuffer  = makeVertexBuffer(biSem);
    // NV-DXVK end

    RasterBuffer idxBuffer;
    if (indexed) {
      const auto& ib = m_context->m_state.ia.indexBuffer;
      if (ib.buffer == nullptr) {
        BumpFilter(FilterReason::NoIndexBuffer);
        return;
      }
      VkIndexType idxType = (ib.format == DXGI_FORMAT_R32_UINT)
                          ? VK_INDEX_TYPE_UINT32
                          : VK_INDEX_TYPE_UINT16;
      uint32_t idxStride = (idxType == VK_INDEX_TYPE_UINT32) ? 4 : 2;
      // NV-DXVK: Bake startIndex into the slice offset. The BLAS builder and
      // the cacheIndexDataOnGPU copy both read from `slice.offset + 0`, not
      // `slice.offset + startIndex*stride`. Without this, draws with startIndex>0
      // get the wrong index range cached → BLAS sees stale indices from the
      // top of the IB → OOB vertex reads → MMU fault.
      idxBuffer = RasterBuffer(
        ib.buffer->GetBufferSlice(ib.offset + size_t(start) * idxStride),
        0, idxStride, idxType);
    }

    VkPrimitiveTopology vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (m_context->m_state.ia.primitiveTopology) {
      case D3D11_PRIMITIVE_TOPOLOGY_POINTLIST:     vkTopology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;     break;
      case D3D11_PRIMITIVE_TOPOLOGY_LINELIST:      vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      break;
      case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP:     vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     break;
      case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST:  vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  break;
      case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
      default: break;
    }

    RasterGeometry geo;
    geo.topology       = vkTopology;
    geo.frontFace      = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    geo.positionBuffer = posBuffer;
    geo.normalBuffer   = nrmBuffer;
    geo.texcoordBuffer = tcBuffer;
    geo.texcoord1Buffer = tc1Buffer;
    geo.color0Buffer   = colBuffer;
    geo.indexBuffer    = idxBuffer;
    geo.indexCount     = indexed ? count : 0;

    // NV-DXVK: TF2 VGUI TEXCOORD3 (int4) capture. Always grabbed when the
    // input layout has a SINT TEXCOORD3 attribute, even on non-VGUI draws
    // (false positives are zero per the format+name+index match above).
    // The interleaver only reads this slot when geo.vguiLayoutEnable is
    // also set, which FillMaterialData decides after PS RDEF inspection.
    if (vguiTc3Sem) {
      geo.vguiTexcoord3Buffer = makeVertexBuffer(vguiTc3Sem);
      geo.vguiTexcoord3Offset = vguiTc3Sem->byteOffset;
      const auto& vb = m_context->m_state.ia.vertexBuffers[vguiTc3Sem->inputSlot];
      geo.vguiTexcoord3Stride = vb.stride;
    }

    // NV-DXVK: TF2 worldspace VGUI TC1 / TC2 explicit routing. VGUI shaders
    // have NO TEXCOORD0 in their input layout — only TEXCOORD1 (4-float
    // primary+secondary glyph quad pos), TEXCOORD2 (2-float glyph dims),
    // and TEXCOORD3 (int4 indices). Without this re-routing, the standard
    // tcSem fallback would either land on TC1 (treating its 4-float layout
    // as if it were 2-float UVs and wasting the zw pair) or on the wrong
    // buffer entirely. Forcing the assignment here when the VGUI signature
    // (vguiTc3Sem) is present guarantees:
    //   geo.texcoordBuffer = TC1 (primary quad pos in xy, zw lost to standard
    //                              path but recovered by the interleaver's
    //                              VGUI extras block reading the same source
    //                              at offset+2/+3)
    //   geo.vguiGlyphDimsBuffer = TC2 (glyph dimensions / scale)
    // The interleaver writes TC1.zw + TC2.xy to the VGUI extras tail and
    // TC1.xy to the standard texcoord-0 slot; surfaceInteraction picks up
    // the latter as textureCoordinates so SDF AA gradient math works.
    // NV-DXVK: VGUI IA signature — TEXCOORD1 with R32G32B32A32_SFLOAT
    // format. Used to decide whether to capture the auxiliary VGUI streams
    // (TC2 glyph dims, and to keep TC1 around for later FillMaterialData
    // routing). Capturing TC2 unconditionally here is safe because the
    // dedicated vguiGlyphDimsBuffer is only READ by the interleaver when
    // vguiLayoutEnable is set — i.e. when FillMaterialData has confirmed
    // this is a real VGUI shader (sourceIsUnlitUI). Non-VGUI shaders that
    // happen to have a TEXCOORD2 just hold an unused capture.
    //
    // texcoordBuffer override is NOT done here — it would clobber the
    // legitimate TC0 of non-VGUI shaders that also have TC1 with
    // R32G32B32A32_SFLOAT (e.g. character VSes carrying tangent space in
    // TC1.xyzw). Moved to the FillMaterialData VGUI block where
    // sourceIsUnlitUI is known.
    const bool vguiIaSignature =
        (tc1Sem != nullptr && tc1Sem->format == VK_FORMAT_R32G32B32A32_SFLOAT);
    if (vguiIaSignature && vguiTc2Sem != nullptr) {
      geo.vguiGlyphDimsBuffer = makeVertexBuffer(vguiTc2Sem);
    }

    // NV-DXVK: VGUI IA-capture diagnostic. Reuses the same vguiIaSignature
    // computed above (tc1Sem with format 109 = R32G32B32A32_SFLOAT). This
    // is the actual TF2 VGUI signature; the dedicated vguiTc1Sem matcher
    // earlier never fires due to the else-if cascade short-circuiting on
    // the tc1Sem matcher.
    if (vguiIaSignature || vguiTc3Sem != nullptr) {
      static std::unordered_set<XXH64_hash_t> sVguiIaLogged;
      static std::mutex sVguiIaMu;
      XXH64_hash_t vsH = 0, psH = 0;
      GetCurrentVsPsHashes(vsH, psH);
      bool firstSeen = false;
      {
        std::lock_guard<std::mutex> lk(sVguiIaMu);
        firstSeen = sVguiIaLogged.insert(vsH).second;
      }
      if (firstSeen) {
        // Dump EVERY semantic in the layout (not just TEXCOORDs) — if TC3
        // is named differently than expected (e.g., "VGUI_INDICES" or some
        // engine-specific tag), we want to see it.
        std::string semDump;
        for (const auto& s : semantics) {
          semDump += " {n=";
          semDump += s.name;
          semDump += " i=";
          semDump += std::to_string(s.index);
          semDump += " fmt=";
          semDump += std::to_string(uint32_t(s.format));
          semDump += " slot=";
          semDump += std::to_string(s.inputSlot);
          semDump += " off=";
          semDump += std::to_string(s.byteOffset);
          semDump += "}";
        }
        Logger::info(str::format(
          "[VguiIaCapture] VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
          " sigVia=", (vguiTc3Sem ? "TC3" : "TC1-format"),
          " allSemantics=", semDump.c_str(),
          " | vguiTc1Sem=", (vguiTc1Sem ? "MATCHED" : "NULL"),
          (vguiTc1Sem ? str::format(" (idx=", uint32_t(vguiTc1Sem->index),
                                    " fmt=", uint32_t(vguiTc1Sem->format),
                                    " slot=", vguiTc1Sem->inputSlot,
                                    " off=", vguiTc1Sem->byteOffset, ")") : ""),
          " | vguiTc2Sem=", (vguiTc2Sem ? "MATCHED" : "NULL"),
          (vguiTc2Sem ? str::format(" (idx=", uint32_t(vguiTc2Sem->index),
                                    " fmt=", uint32_t(vguiTc2Sem->format),
                                    " slot=", vguiTc2Sem->inputSlot,
                                    " off=", vguiTc2Sem->byteOffset, ")") : ""),
          " | vguiTc3Sem MATCHED (idx=", uint32_t(vguiTc3Sem->index),
                                  " fmt=", uint32_t(vguiTc3Sem->format),
                                  " slot=", vguiTc3Sem->inputSlot,
                                  " off=", vguiTc3Sem->byteOffset, ")"));
        Logger::info(str::format(
          "[VguiIaCapture.Buffers]",
          " geo.texcoordBuffer.defined=", (geo.texcoordBuffer.defined() ? 1 : 0),
          " fmt=", (geo.texcoordBuffer.defined() ? uint32_t(geo.texcoordBuffer.vertexFormat()) : 0u),
          " stride=", (geo.texcoordBuffer.defined() ? geo.texcoordBuffer.stride() : 0u),
          " ofs=", (geo.texcoordBuffer.defined() ? geo.texcoordBuffer.offsetFromSlice() : 0u),
          " | geo.texcoord1Buffer.defined=", (geo.texcoord1Buffer.defined() ? 1 : 0),
          " | geo.vguiGlyphDimsBuffer.defined=", (geo.vguiGlyphDimsBuffer.defined() ? 1 : 0),
          " fmt=", (geo.vguiGlyphDimsBuffer.defined() ? uint32_t(geo.vguiGlyphDimsBuffer.vertexFormat()) : 0u),
          " stride=", (geo.vguiGlyphDimsBuffer.defined() ? geo.vguiGlyphDimsBuffer.stride() : 0u),
          " ofs=", (geo.vguiGlyphDimsBuffer.defined() ? geo.vguiGlyphDimsBuffer.offsetFromSlice() : 0u),
          " | geo.vguiTexcoord3Buffer.defined=", (geo.vguiTexcoord3Buffer.defined() ? 1 : 0),
          " fmt=", (geo.vguiTexcoord3Buffer.defined() ? uint32_t(geo.vguiTexcoord3Buffer.vertexFormat()) : 0u),
          " stride=", (geo.vguiTexcoord3Buffer.defined() ? geo.vguiTexcoord3Buffer.stride() : 0u),
          " ofs=", (geo.vguiTexcoord3Buffer.defined() ? geo.vguiTexcoord3Buffer.offsetFromSlice() : 0u)));

        // NV-DXVK: VGUI source-buffer ground-truth dump. Reads the actual
        // bytes of TC1/TC2/TC3 for the first 4 vertices DIRECTLY from the
        // game's mapped vertex buffer (before any interleave/decode). If
        // these are non-zero on the CPU side but the slang VGUI evaluator
        // sees zeros, the bug is in the BLAS-build path. If these are
        // zero, the bug is upstream — engine/VS isn't writing the data
        // we expect into the captured slots.
        const auto dumpSrcAtSlot = [&](const D3D11RtxSemantic* sem, const char* tag) {
          if (sem == nullptr) {
            Logger::info(str::format("[VguiSrcDump.", tag, "] NULL"));
            return;
          }
          if (sem->inputSlot >= D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) {
            Logger::info(str::format("[VguiSrcDump.", tag, "] slot OOB"));
            return;
          }
          const auto& vb = m_context->m_state.ia.vertexBuffers[sem->inputSlot];
          if (vb.buffer == nullptr) {
            Logger::info(str::format("[VguiSrcDump.", tag, "] vb.buffer null"));
            return;
          }
          const auto mapped = vb.buffer->GetMappedSlice();
          if (mapped.mapPtr == nullptr) {
            Logger::info(str::format("[VguiSrcDump.", tag,
              "] mapPtr null usage=", uint32_t(vb.buffer->Desc()->Usage),
              " bw=", vb.buffer->Desc()->ByteWidth));
            return;
          }
          const uint8_t* base = static_cast<const uint8_t*>(mapped.mapPtr) +
                                vb.offset + sem->byteOffset;
          const uint32_t bw = vb.buffer->Desc()->ByteWidth;
          // Dump as both float4 (works for fmt 109/103) AND uint4 (for fmt 96).
          float fv[4][4] = {};
          uint32_t uv[4][4] = {};
          for (int v = 0; v < 4; ++v) {
            const size_t off = size_t(vb.offset) + size_t(sem->byteOffset)
                             + size_t(v) * size_t(vb.stride);
            if (off + 16 > bw) break;
            const uint8_t* p = static_cast<const uint8_t*>(mapped.mapPtr) + off;
            std::memcpy(&fv[v][0], p, 16);
            std::memcpy(&uv[v][0], p, 16);
          }
          Logger::info(str::format(
            "[VguiSrcDump.", tag, "] slot=", sem->inputSlot,
            " byteOff=", sem->byteOffset,
            " vbOff=", vb.offset, " vbStride=", vb.stride,
            " fmt=", uint32_t(sem->format),
            " v0=(", fv[0][0], ",", fv[0][1], ",", fv[0][2], ",", fv[0][3], ")",
            " v1=(", fv[1][0], ",", fv[1][1], ",", fv[1][2], ",", fv[1][3], ")",
            " v2=(", fv[2][0], ",", fv[2][1], ",", fv[2][2], ",", fv[2][3], ")",
            " v3=(", fv[3][0], ",", fv[3][1], ",", fv[3][2], ",", fv[3][3], ")",
            " | asUint v0=(0x", std::hex, uv[0][0], ",0x", uv[0][1],
                ",0x", uv[0][2], ",0x", uv[0][3], ") v1=(0x", uv[1][0],
                ",0x", uv[1][1], ",0x", uv[1][2], ",0x", uv[1][3], ")", std::dec));
          (void)base;
        };
        dumpSrcAtSlot(tc1Sem, "TC1");
        dumpSrcAtSlot(vguiTc2Sem, "TC2");
        dumpSrcAtSlot(vguiTc3Sem, "TC3");
      }
    }

    // NV-DXVK: Snapshot index bytes NOW from the game's currently mapped slice.
    // Runs on the thread that owns the D3D11 state (deferred context replay on
    // CS thread, or immediate context) before any subsequent Map/DISCARD can
    // rename the physical slice. Without this, the deferred cacheIndexDataOnGPU
    // copy (later on CS thread) reads the renamed slice → garbage indices →
    // BLAS build OOB → MMU fault → TDR.
    //
    // Only snapshot DYNAMIC buffers (the only ones subject to renaming).
    // Static/immutable buffers have stable physical addresses — zero overhead.
    if (indexed) {
      static uint32_t sIdxSnapStats[4] = {0, 0, 0, 0};  // dyn_snapped, dyn_no_mapptr, static_skipped, null_buf
      static uint32_t sIdxSnapLog = 0;
      const auto& ib2 = m_context->m_state.ia.indexBuffer;
      bool snapped = false;
      if (ib2.buffer == nullptr) {
        ++sIdxSnapStats[3];
      } else if (ib2.buffer->Desc()->Usage != D3D11_USAGE_DYNAMIC) {
        ++sIdxSnapStats[2];
      } else {
        const auto mapped = ib2.buffer->GetMappedSlice();
        if (mapped.mapPtr == nullptr) {
          ++sIdxSnapStats[1];
        } else {
          const uint32_t idxStride2 = (ib2.format == DXGI_FORMAT_R32_UINT) ? 4u : 2u;
          const size_t snapLen = size_t(count) * idxStride2;
          // Draw reads indices [start, start+count) — snapshot must start at
          // start*stride, not 0, or cacheIndexDataOnGPU uploads the wrong range.
          const size_t snapOff = size_t(ib2.offset) + size_t(start) * idxStride2;
          const size_t bufLen  = ib2.buffer->Desc()->ByteWidth;
          if (snapLen > 0 && snapOff + snapLen <= bufLen) {
            geo.indexDataSnapshot = std::make_shared<std::vector<uint8_t>>(snapLen);
            std::memcpy(geo.indexDataSnapshot->data(),
                        reinterpret_cast<const uint8_t*>(mapped.mapPtr) + snapOff,
                        snapLen);
            ++sIdxSnapStats[0];
            snapped = true;
          }
        }
      }
      // Log first 30 draws + stats every 500 draws
      if (sIdxSnapLog < 30 || (sIdxSnapLog % 500) == 0) {
        Logger::info(str::format("[IDX-SNAP] snap=", snapped ? 1 : 0,
          " count=", count,
          " usage=", (ib2.buffer != nullptr ? uint32_t(ib2.buffer->Desc()->Usage) : 0u),
          " off=", ib2.offset,
          " stats: dynSnap=", sIdxSnapStats[0],
          " dynNoMap=", sIdxSnapStats[1],
          " static=", sIdxSnapStats[2],
          " null=", sIdxSnapStats[3]));
      }
      ++sIdxSnapLog;
    }

    // NV-DXVK start: Per-vertex skinning — populate blend buffers and bone count
    if (bwBuffer.defined() && biBuffer.defined()) {
      geo.blendWeightBuffer  = bwBuffer;
      geo.blendIndicesBuffer = biBuffer;
      // Derive bones-per-vertex from the blend weight format:
      // Each explicit weight implies one bone; the last bone's weight is
      // implicit (1 - sum).  So N explicit weights → N+1 bones.
      // NV-DXVK TF2: extend coverage to Source/Respawn-engine compressed
      // weight formats (R16G16_SINT = fmt=82 in DXGI). Verified from
      // VS_ef94e6c7fcc3c144 DXIL: the shader reads BLENDWEIGHT.xy as two
      // signed int16s and decodes w0, w1 with `(v+1)/32768`, with
      // w2 = 1-w0-w1. Two explicit weights = 3 bones per vertex, same as
      // R32G32_SFLOAT. Without this case the switch fell through to
      // `default: numBonesPerVertex=0`, which zeroed
      // `dcs.skinningData.numBones` downstream, which tipped the accel
      // manager's routing check (`numBones != 0`) to FALSE and sent the
      // skinned body + gun into the STATIC merged-bucket BLAS path
      // instead of the dynamic BLAS path — so the gun's bone-skinning
      // didn't refit the BLAS correctly and the mesh was effectively
      // missing from the TLAS each frame.
      switch (bwSem->format) {
        case VK_FORMAT_R32_SFLOAT:                geo.numBonesPerVertex = 2; break;
        case VK_FORMAT_R32G32_SFLOAT:             geo.numBonesPerVertex = 3; break;
        case VK_FORMAT_R32G32B32_SFLOAT:          geo.numBonesPerVertex = 4; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:       geo.numBonesPerVertex = 4; break;
        // Source / TF2 packed int16 pairs — decoded to float in the
        // interleaver via (int16+1)/32768. Two explicit weights → 3 bones.
        case VK_FORMAT_R16G16_SINT:               geo.numBonesPerVertex = 3; break;
        case VK_FORMAT_R16G16_UINT:               geo.numBonesPerVertex = 3; break;
        // 4x int16 or uint16 would give 4 explicit → 5 bones, but this is
        // unusual; cap at 4 to match the 4-wide BLENDINDICES field TF2 uses.
        case VK_FORMAT_R16G16B16A16_SINT:         geo.numBonesPerVertex = 4; break;
        case VK_FORMAT_R16G16B16A16_UINT:         geo.numBonesPerVertex = 4; break;
        // 8-bit packed (some Source variants use fmt=42 = R8G8B8A8_UINT for
        // weights too) — 4 explicit → cap at 4.
        case VK_FORMAT_R8G8B8A8_UINT:             geo.numBonesPerVertex = 4; break;
        case VK_FORMAT_R8G8B8A8_SINT:             geo.numBonesPerVertex = 4; break;
        default:                                  geo.numBonesPerVertex = 0; break;
      }
    }
    // NV-DXVK end

    // NV-DXVK start: Diagnostic — dump first N unique input layouts
    {
      static uint32_t sLayoutLog = 0;
      static uintptr_t sLastLayout = 0;
      uintptr_t layoutAddr = reinterpret_cast<uintptr_t>(m_context->m_state.ia.inputLayout.ptr());
      if (layoutAddr != sLastLayout && sLayoutLog < 20) {
        sLastLayout = layoutAddr;
        ++sLayoutLog;
        Logger::info(str::format("[D3D11Rtx] Layout #", sLayoutLog,
                                 " (", semantics.size(), " semantics):"));
        for (const auto& s : semantics) {
          Logger::info(str::format("[D3D11Rtx]   name=", s.name,
            " idx=", s.index, " fmt=", uint32_t(s.format),
            " slot=", s.inputSlot, " off=", s.byteOffset,
            " inst=", s.perInstance ? 1 : 0));
        }
      }
    }
    // NV-DXVK end

    // NV-DXVK: Track bone buffer and attach bone data for GPU instancing.
    // For R32G32_UINT positions AND for instanced bone draws (m_attachBoneBuffers),
    // attach a SRV-backed transform buffer + per-vertex/per-instance index source.
    //
    // Two TF2 patterns share most of the plumbing:
    //   (A) Skinned characters (g_boneMatrix at t30, stride=48):
    //       - Per-instance R16G16B16A16_UINT semantic (bone indices)
    //       - One bone matrix per draw (use index from semantic)
    //   (B) BSP / batched static props (g_modelInst at t31, stride=208):
    //       - Per-vertex COLOR1 R32G32B32A32_UINT semantic (instance indices)
    //       - Each vertex picks its own transform via cb.bonePerVertex path
    // DEBUG: log posSem format + SRV slot occupancy for first N draws of each
    // unique VS, so we can see why the BSP path doesn't fire.
    {
      static std::unordered_set<uintptr_t> sPosFmtLogged;
      auto vsPtr = m_context->m_state.vs.shader;
      uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
      if (key && sPosFmtLogged.size() < 40 && sPosFmtLogged.insert(key).second) {
        const auto& srvs = m_context->m_state.vs.shaderResources.views;
        std::string vsHash = "?";
        if (vsPtr->GetCommonShader() != nullptr) {
          auto& s = vsPtr->GetCommonShader()->GetShader();
          if (s != nullptr) vsHash = s->getShaderKey().toString();
        }
        Logger::info(str::format(
          "[D3D11Rtx] PosFmtProbe VS=", vsHash,
          " posFmt=", posSem ? uint32_t(posSem->format) : 0,
          " posPerInst=", posSem ? (posSem->perInstance ? 1 : 0) : 0,
          " m_attachBoneBuffers=", m_attachBoneBuffers ? 1 : 0,
          " t30=", srvs[30].ptr() ? 1 : 0,
          " t31=", srvs[31].ptr() ? 1 : 0,
          " bspGuard=", (posSem && posSem->format == VK_FORMAT_R32G32_UINT) || m_attachBoneBuffers ? 1 : 0));
      }
    }
    if (posSem->format == VK_FORMAT_R32G32_UINT || m_attachBoneBuffers) {
      // NV-DXVK: ask the VS RDEF which resource it actually declares — both
      // t30 and t31 may be bound by app state, but each VS only reads ONE.
      // Preferring t30 by default mis-routed BSP draws (which read g_modelInst
      // at t31) into the bone-skinning path and they ended up at origin.
      uint32_t modelInstSlot = UINT32_MAX;
      uint32_t boneMatrixSlot = UINT32_MAX;
      {
        auto vsPtr = m_context->m_state.vs.shader;
        if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
          const D3D11CommonShader* common = vsPtr->GetCommonShader();
          modelInstSlot  = common->FindResourceSlot("g_modelInst");
          boneMatrixSlot = common->FindResourceSlot("g_boneMatrix");
        }
        // DEBUG: log RDEF resource lookup result per unique VS
        static std::unordered_set<uintptr_t> sRdefLookupLogged;
        uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
        if (key && sRdefLookupLogged.size() < 30 && sRdefLookupLogged.insert(key).second) {
          std::string vsHash = "?";
          if (vsPtr->GetCommonShader() != nullptr) {
            auto& s = vsPtr->GetCommonShader()->GetShader();
            if (s != nullptr) vsHash = s->getShaderKey().toString();
          }
          Logger::info(str::format(
            "[D3D11Rtx] RdefLookup VS=", vsHash,
            " g_modelInst=", modelInstSlot,
            " g_boneMatrix=", boneMatrixSlot));
        }
      }
      // BSP / batched static props use g_modelInst when present. Otherwise
      // fall back to g_boneMatrix (skinned characters). Final fallback: scan
      // both slots blindly (covers shaders without RDEF).
      ID3D11ShaderResourceView* xformSrv = nullptr;
      bool isModelInst = false;
      if (modelInstSlot != UINT32_MAX
          && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
        xformSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
        if (xformSrv) isModelInst = true;
      }
      if (!xformSrv && boneMatrixSlot != UINT32_MAX
          && boneMatrixSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
        xformSrv = m_context->m_state.vs.shaderResources.views[boneMatrixSlot].ptr();
      }
      if (!xformSrv) {
        // NV-DXVK semantic-based blind probe.
        // RDEF missed both g_modelInst and g_boneMatrix. Use the VS's declared
        // input semantics to classify the shader and attach only when the
        // semantics prove t31 is needed:
        //   - BLENDINDICES per-vertex → skinned character; t31 = bone palette.
        //     Attach as bone buffer (xformSrv path below with !isModelInst).
        //   - COLOR1/I R16G16B16A16_UINT per-instance → instanced BSP/prop;
        //     t31 = g_modelInst. Attach with isModelInst=true.
        //   - Neither → static cb3-only mesh (e.g. VS_6e3e6f28). Skip — the
        //     cb3 RDEF path upstream already wrote the correct objectToWorld.
        //     Attaching t30/t31 here would route vertices through skinning or
        //     per-instance fanout and warp the mesh around the camera.
        auto* ilProbe = m_context->m_state.ia.inputLayout.ptr();
        bool semBlendIdx = false;
        bool semPerInstIdx = false;
        if (ilProbe != nullptr) {
          for (const auto& s : ilProbe->GetRtxSemantics()) {
            if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0)
              semBlendIdx = true;
            if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT)
              semPerInstIdx = true;
          }
        }
        constexpr uint32_t kT31Slot = 31;
        if ((semBlendIdx || semPerInstIdx) && kT31Slot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
          xformSrv = m_context->m_state.vs.shaderResources.views[kT31Slot].ptr();
          if (xformSrv && semPerInstIdx) isModelInst = true;
        }
        auto vsPtr = m_context->m_state.vs.shader;
        if (vsPtr != nullptr) {
          ++m_geomDiagBlindProbes;
          static std::unordered_set<uintptr_t> sBlindClassifyLogged;
          uintptr_t key = reinterpret_cast<uintptr_t>(vsPtr.ptr());
          if (sBlindClassifyLogged.insert(key).second) {
            std::string vsHash = "?";
            if (vsPtr->GetCommonShader() != nullptr) {
              auto& s = vsPtr->GetCommonShader()->GetShader();
              if (s != nullptr) vsHash = s->getShaderKey().toString();
            }
            const char* cls = semBlendIdx ? "skinned_char_t31_bone_palette"
                            : semPerInstIdx ? "instanced_bsp_t31_modelInst"
                            : "static_mesh_cb3_owns_transform_skip_attach";
            Logger::info(str::format(
              "[D3D11Rtx] BLIND-PROBE classify VS=", vsHash,
              " class=", cls,
              " attached=", xformSrv ? 1 : 0,
              " isModelInst=", isModelInst ? 1 : 0));
          }
        }
      }
      if (xformSrv && !isModelInst) {
        // Legacy skinning path only. For BSP / batched-prop draws (isModelInst)
        // we do NOT attach a bone matrix here — the per-instance fanout above
        // already creates one TLAS instance per modelInst row with the correct
        // transform. Letting the interleave shader also bone-multiply would
        // double-apply the matrix and put geometry at sqr(transform) * raw_pos.
        Com<ID3D11Resource> xformRes;
        xformSrv->GetResource(&xformRes);
        auto* xformBuf = static_cast<D3D11Buffer*>(xformRes.ptr());
        if (xformBuf) {
          const uint32_t matrixStride = 48u;
          // NV-DXVK TF2 FIX (universal): respect the SRV's FirstElement for
          // bone palette indexing. TF2 (and its NPC variants) bind t30 with
          // a per-draw FirstElement window — applying it here fixes spike
          // artifacts on NPC characters that use non-R8G8B8A8_UINT blend
          // index formats and therefore don't enter the fmt=41 block below.
          uint32_t firstElemBones = 0;
          {
            D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
            xformSrv->GetDesc(&sd);
            if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFER)
              firstElemBones = sd.Buffer.FirstElement;
            else if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFEREX)
              firstElemBones = sd.BufferEx.FirstElement;
          }
          const uint32_t byteOffset = firstElemBones * matrixStride;
          geo.boneMatrixBuffer = RasterBuffer(
            xformBuf->GetBufferSlice(byteOffset), 0, matrixStride, VK_FORMAT_UNDEFINED);
          geo.boneMatrixStrideBytes = matrixStride;
          static uint32_t sLegacyFirstElemLog = 0;
          if (firstElemBones != 0 && sLegacyFirstElemLog < 10) {
            ++sLegacyFirstElemLog;
            Logger::info(str::format(
              "[D3D11Rtx.legacySkin.firstElem] applied byteOffset=", byteOffset,
              " (firstElemBones=", firstElemBones, ")"));
          }
        }
      } else if (xformSrv && isModelInst) {
        // BSP-path: log once per unique buffer so we know fanout activated.
        Com<ID3D11Resource> xformRes; xformSrv->GetResource(&xformRes);
        auto* xformBuf = static_cast<D3D11Buffer*>(xformRes.ptr());
        static uint32_t sBspLogCount = 0;
        if (xformBuf && sBspLogCount < 20) {
          ++sBspLogCount;
          Logger::info(str::format(
            "[D3D11Rtx] BSP-fanout-path (t31): bufSize=",
            xformBuf->Desc()->ByteWidth, " (no boneMatrixBuffer attached)"));
        }
      }
      // DEBUG: dump every semantic for the first N BSP-path draws so we can see
      // what the per-vertex/per-instance index format actually is.
      if (isModelInst) {
        static uint32_t sBspSemDump = 0;
        if (sBspSemDump < 6) {
          ++sBspSemDump;
          for (const auto& s : semantics) {
            Logger::info(str::format(
              "[D3D11Rtx] BSP semantic dump: name=", s.name, " idx=", s.index,
              " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
              " byteOff=", s.byteOffset,
              " perInst=", s.perInstance ? 1 : 0));
          }
        }
      }
      // NV-DXVK (TF2 skinned characters): detect the weighted-skinning
      // fingerprint — POSITION0/V + BLENDINDICES0/V:fmt41 (RGBA8_UINT) +
      // BLENDWEIGHT0/V:fmt82 (R16G16 UNORM) + t30 SRV (g_boneMatrix, stride
      // 48). Bind t30 as matrix buffer, BLENDINDICES VB as index buffer,
      // BLENDWEIGHT VB as weight buffer. Interleaver does Σ w_i bone[idx_i].
      bool didSkinnedChar = false;
      if (biSem != nullptr && bwSem != nullptr
          && biSem->format == VK_FORMAT_R8G8B8A8_UINT
          && !biSem->perInstance) {
        // Use t30 directly (g_boneMatrix). xformSrv above may have picked t31
        // for isModelInst=false non-instanced BSP — override to t30 here.
        ID3D11ShaderResourceView* boneSrv = nullptr;
        {
          uint32_t boneSlot = UINT32_MAX;
          auto vsPtr2 = m_context->m_state.vs.shader;
          if (vsPtr2 != nullptr && vsPtr2->GetCommonShader() != nullptr)
            boneSlot = vsPtr2->GetCommonShader()->FindResourceSlot("g_boneMatrix");
          if (boneSlot == UINT32_MAX) boneSlot = 30u;
          if (boneSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
            boneSrv = m_context->m_state.vs.shaderResources.views[boneSlot].ptr();
        }
        if (boneSrv && biBuffer.defined() && bwBuffer.defined()) {
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          if (boneBuf) {
            // NV-DXVK: the game binds t30 via an SRV with a per-draw
            // `FirstElement` window. Its VS does `t30[BLENDINDICES.x]`
            // which the D3D runtime resolves as `buffer[FirstElement +
            // idx]`. Our interleaver takes a raw buffer slice and indexes
            // from 0, so without the offset we read garbage (zero slots)
            // on every draw that has FirstElement != 0 → spikes.
            uint32_t srvFirstElemBones = 0;
            {
              D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
              boneSrv->GetDesc(&sd);
              if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFER)
                srvFirstElemBones = sd.Buffer.FirstElement;
              else if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFEREX)
                srvFirstElemBones = sd.BufferEx.FirstElement;
            }
            const uint32_t boneByteOffset = srvFirstElemBones * 48u;
            geo.boneMatrixBuffer = RasterBuffer(
              boneBuf->GetBufferSlice(boneByteOffset), 0, 48u, VK_FORMAT_UNDEFINED);
            geo.boneMatrixStrideBytes = 48u;

            // NV-DXVK [BoneSrvs]: log BOTH t30 (g_boneMatrix) AND t32
            // (g_boneMatrixPrevFrame) SRV descriptors for this draw.
            // Unthrottled — every skinned draw emits one line. Gated on
            // RTX_BONE_DIAG.
            if (::dxvk::tf2::boneDiagEnabled()) {
              {
                std::string vsN = "?";
                auto vs = m_context->m_state.vs.shader;
                if (vs != nullptr && vs->GetCommonShader() != nullptr) {
                  auto& s = vs->GetCommonShader()->GetShader();
                  if (s != nullptr) vsN = s->getShaderKey().toString().substr(0, 19);
                }
                auto srv30 = m_context->m_state.vs.shaderResources.views[30].ptr();
                auto srv32 = m_context->m_state.vs.shaderResources.views[32].ptr();
                auto describe = [](ID3D11ShaderResourceView* srv, uintptr_t& outBuf,
                                   uint32_t& outFirst, uint32_t& outNum,
                                   uint32_t& outSize) {
                  outBuf = 0; outFirst = 0; outNum = 0; outSize = 0;
                  if (!srv) return;
                  D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
                  srv->GetDesc(&sd);
                  if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFER) {
                    outFirst = sd.Buffer.FirstElement;
                    outNum   = sd.Buffer.NumElements;
                  } else if (sd.ViewDimension == D3D11_SRV_DIMENSION_BUFFEREX) {
                    outFirst = sd.BufferEx.FirstElement;
                    outNum   = sd.BufferEx.NumElements;
                  }
                  Com<ID3D11Resource> r;
                  srv->GetResource(&r);
                  D3D11_RESOURCE_DIMENSION dim;
                  r->GetType(&dim);
                  if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
                    auto* b = static_cast<D3D11Buffer*>(r.ptr());
                    outBuf = reinterpret_cast<uintptr_t>(b);
                    outSize = b->Desc()->ByteWidth;
                  }
                };
                uintptr_t buf30 = 0, buf32 = 0;
                uint32_t first30 = 0, first32 = 0, num30 = 0, num32 = 0;
                uint32_t size30 = 0, size32 = 0;
                describe(srv30, buf30, first30, num30, size30);
                describe(srv32, buf32, first32, num32, size32);
                // NV-DXVK NPC SKINNING DIAG: emit the underlying Dxvk
                // VkBuffer pointers too — same id space as the
                // [Dxvk.copyBufTo393216] / [Dxvk.updateBuf393216] /
                // [interleaver.skin.offsets] logs, so we can tell whether
                // writes that hit hooks are landing on the buffer this
                // draw is actually reading from.
                uintptr_t dxvkBuf30 = 0, dxvkBuf32 = 0;
                if (srv30) {
                  Com<ID3D11Resource> r;
                  srv30->GetResource(&r);
                  D3D11_RESOURCE_DIMENSION dim;
                  r->GetType(&dim);
                  if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
                    auto* db = static_cast<D3D11Buffer*>(r.ptr())->GetBuffer().ptr();
                    dxvkBuf30 = reinterpret_cast<uintptr_t>(db);
                  }
                }
                if (srv32) {
                  Com<ID3D11Resource> r;
                  srv32->GetResource(&r);
                  D3D11_RESOURCE_DIMENSION dim;
                  r->GetType(&dim);
                  if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
                    auto* db = static_cast<D3D11Buffer*>(r.ptr())->GetBuffer().ptr();
                    dxvkBuf32 = reinterpret_cast<uintptr_t>(db);
                  }
                }
                const bool sameBuf = (buf30 == buf32 && buf30 != 0);
                Logger::info(str::format(
                  "[BoneSrvs] vs=", vsN,
                  " t30:buf=", buf30, " dxvkBuf=", dxvkBuf30,
                  " first=", first30, " num=", num30, " sz=", size30,
                  " t32:buf=", buf32, " dxvkBuf=", dxvkBuf32,
                  " first=", first32, " num=", num32, " sz=", size32,
                  " sameBuf=", (sameBuf ? 1 : 0)));
                // NV-DXVK NPC SKINNING DIAG: record this (buf, VS, first,
                // num) for the EndFrame cross-check. Dedupe on the fly so
                // a shader that issues thousands of identical-binding
                // draws produces one row with drawCount=N.
                if (dxvkBuf30 != 0) {
                  std::lock_guard<std::mutex> lk(::dxvk::tf2::g_boneSrvsMutex);
                  auto& vec = ::dxvk::tf2::g_boneSrvsThisFrame;
                  bool merged = false;
                  for (auto& r : vec) {
                    if (r.bufPtr == dxvkBuf30
                        && r.firstElem == first30
                        && r.numElem == num30
                        && std::strncmp(r.vsShort, vsN.c_str(), sizeof(r.vsShort) - 1) == 0) {
                      ++r.drawCount;
                      merged = true;
                      break;
                    }
                  }
                  if (!merged) {
                    ::dxvk::tf2::BoneSrvRecord rec{};
                    rec.bufPtr    = dxvkBuf30;
                    rec.firstElem = first30;
                    rec.numElem   = num30;
                    rec.drawCount = 1;
                    rec.sameBuf   = sameBuf ? 1u : 0u;
                    std::strncpy(rec.vsShort, vsN.c_str(), sizeof(rec.vsShort) - 1);
                    vec.push_back(rec);
                  }
                }
                // NV-DXVK NPC SKINNING DIAG: auto-arm the per-frame
                // summary on the first sameBuf=0 (NPC-style) draw.
                // Player viewmodel has sameBuf=1 because t30 == t32;
                // NPCs use distinct curr/prev frame buffers.
                if (::dxvk::tf2::boneDiagEnabled()
                    && !sameBuf && dxvkBuf30 != 0) {
                  uintptr_t expected = 0;
                  if (::dxvk::tf2::g_autoTargetBufPtr.compare_exchange_strong(
                        expected, dxvkBuf30, std::memory_order_relaxed)) {
                    Logger::info(str::format(
                      "[BoneTargetArm] auto-targeting NPC bone buf dxvkBuf=",
                      dxvkBuf30, " (sameBuf=0). Override via env",
                      " RTX_NPC_BONE_BUF=<hexPtr>."));
                  }
                }
              }
            }
            // NV-DXVK TF2 VIEWMODEL: capture first-bone world translation
            // from the full bone cache so the o2w handler downstream can
            // shift view-model meshes (srvFirstElem >= 672) from their
            // game-side junk world pos to in-front-of-camera.
            m_vmFirstElem = srvFirstElemBones;
            m_vmBoneRootValid = false;
            if (m_hasFullBoneCache
                && (boneByteOffset + 48u) <= m_fullBoneCache.size()) {
              const float* bm = reinterpret_cast<const float*>(
                  m_fullBoneCache.data() + boneByteOffset);
              // Row-major float3x4: translation is at cols [3, 7, 11].
              m_vmBoneRoot[0] = bm[3];
              m_vmBoneRoot[1] = bm[7];
              m_vmBoneRoot[2] = bm[11];
              // Guard: only treat as valid if it's a finite, non-zero T.
              const float mag = std::fabs(m_vmBoneRoot[0])
                              + std::fabs(m_vmBoneRoot[1])
                              + std::fabs(m_vmBoneRoot[2]);
              m_vmBoneRootValid = std::isfinite(mag) && mag > 1e-3f;
            }
            geo.boneIndexBuffer = biBuffer;
            geo.boneIndexStrideBytes = biBuffer.stride();
            geo.boneIndexMask = 0xFFu;  // per-byte index within packed RGBA8
            geo.boneIndexComponentCount = 4u;
            geo.bonePerVertex = true;
            geo.boneWeightBuffer = bwBuffer;
            didSkinnedChar = true;
            // NV-DXVK: bone matrices are in camera-relative space (TF2 VS
            // does `cb2.c_cameraRelativeToClip * t30[idx] * local`). After
            // weighted skinning the interleaver produces camera-relative
            // positions, so objectToWorld must translate by +fanoutCam to
            // land them in absolute world. We can't write dcs here because
            // dcs isn't constructed yet; flip a flag and apply after dcs.
            m_skinnedCharNeedsCamOffset = true;
            static uint32_t sSkinLog = 0;
            if (sSkinLog < 20) {
              ++sSkinLog;
              Logger::info(str::format(
                "[D3D11Rtx] TF2 skinned char bound: t30buf=",
                boneBuf->Desc()->ByteWidth,
                " biStride=", biBuffer.stride(),
                " bwStride=", bwBuffer.stride(),
                " biOff=", biBuffer.offsetFromSlice(),
                " bwOff=", bwBuffer.offsetFromSlice()));
            }
            // NV-DXVK SPIKE DIAG ([DrawSkin]): per-draw log — pair VS hash
            // with PS hash, t30 pointer/size, and BI/BW pointers. Throttled
            // to 8 entries per frame so we catch multiple skinned draws
            // (e.g. color pass + a second pass using a different VS that
            // might be the real source of the grey spikes) without flooding.
            {
              const uint32_t frameId = m_context->m_device->getCurrentFrameId();
              static uint32_t sLastFrame = 0;
              static uint32_t sCountThisFrame = 0;
              if (frameId != sLastFrame) { sLastFrame = frameId; sCountThisFrame = 0; }
              if (sCountThisFrame < 8) {
                ++sCountThisFrame;
                // VS hash
                std::string vsName = "?";
                auto vsKey = m_context->m_state.vs.shader;
                if (vsKey != nullptr && vsKey->GetCommonShader() != nullptr) {
                  auto& s = vsKey->GetCommonShader()->GetShader();
                  if (s != nullptr) vsName = s->getShaderKey().toString().substr(0, 19);
                }
                // PS hash
                std::string psName = "null";
                auto psKey = m_context->m_state.ps.shader;
                if (psKey != nullptr && psKey->GetCommonShader() != nullptr) {
                  auto& s = psKey->GetCommonShader()->GetShader();
                  if (s != nullptr) psName = s->getShaderKey().toString().substr(0, 19);
                }
                // Approx numVerts from BI buffer length / stride
                uint32_t approxVerts = 0;
                if (biBuffer.defined() && biBuffer.stride() > 0) {
                  approxVerts = static_cast<uint32_t>(biBuffer.length() / biBuffer.stride());
                }
                // Bound VB pointers (helps distinguish two skinned draws that
                // happen to share VS but have different meshes).
                const auto& biVbDiag = m_context->m_state.ia.vertexBuffers[biSem->inputSlot];
                const auto& bwVbDiag = m_context->m_state.ia.vertexBuffers[bwSem->inputSlot];
                // NV-DXVK spike hunt: log the t30 SRV's FirstElement /
                // NumElements / StructureStride. The game's shader does
                // `t30[BLENDINDICES.x]` with BLENDINDICES having values
                // that should fall in zero slots (upper half of each
                // 16-bone palette). If FirstElement != 0, the shader's
                // index 0 maps to a buffer slot != 0, which would mean
                // our cache-offset assumption is wrong.
                D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
                boneSrv->GetDesc(&srvDesc);
                uint32_t srvFirstElem = 0, srvNumElem = 0, srvFlags = 0;
                const char* srvKind = "?";
                if (srvDesc.ViewDimension == D3D11_SRV_DIMENSION_BUFFER) {
                  srvKind = "Buffer";
                  srvFirstElem = srvDesc.Buffer.FirstElement;
                  srvNumElem = srvDesc.Buffer.NumElements;
                } else if (srvDesc.ViewDimension == D3D11_SRV_DIMENSION_BUFFEREX) {
                  srvKind = "BufferEx";
                  srvFirstElem = srvDesc.BufferEx.FirstElement;
                  srvNumElem = srvDesc.BufferEx.NumElements;
                  srvFlags = srvDesc.BufferEx.Flags;
                }
                Logger::info(str::format(
                  "[DrawSkin] f=", frameId,
                  " vs=", vsName,
                  " ps=", psName,
                  " t30Ptr=", reinterpret_cast<uintptr_t>(boneBuf),
                  " t30Size=", boneBuf->Desc()->ByteWidth,
                  " srv=", srvKind,
                  " srvFormat=", (uint32_t)srvDesc.Format,
                  " srvFirstElem=", srvFirstElem,
                  " srvNumElem=", srvNumElem,
                  " srvFlags=", srvFlags,
                  " biVbPtr=", reinterpret_cast<uintptr_t>(biVbDiag.buffer.ptr()),
                  " bwVbPtr=", reinterpret_cast<uintptr_t>(bwVbDiag.buffer.ptr()),
                  " verts=", approxVerts));
              }
            }
            // NV-DXVK SPIKE DIAG ([skin.histo]): per-submesh bone-index range
            // + upper-half-of-palette usage. TF2 t30 is organised as
            // 16-bone palettes where only slots 0-7 are CPU-written; slots
            // 8-15 of each palette are zero → verts that index into the
            // upper half of any palette skin to ~origin → spikes. This log
            // lets us correlate per-submesh BI range with the spike verts
            // reported by [skin.spike] and test the palette-layout theory.
            {
              const uint32_t frameId2 = m_context->m_device->getCurrentFrameId();
              static uint32_t sLastFrameH = 0;
              static uint32_t sCountThisFrameH = 0;
              if (frameId2 != sLastFrameH) { sLastFrameH = frameId2; sCountThisFrameH = 0; }
              if (sCountThisFrameH < 16) {
                ++sCountThisFrameH;
                const auto& biVbH = m_context->m_state.ia.vertexBuffers[biSem->inputSlot];
                const auto& bwVbH = m_context->m_state.ia.vertexBuffers[bwSem->inputSlot];
                const uint8_t* biPtrH = nullptr; size_t biLenH = 0;
                const uint8_t* bwPtrH = nullptr; size_t bwLenH = 0;
                auto grabH = [](D3D11Buffer* b, const uint8_t*& outP, size_t& outLen) {
                  if (!b) return;
                  const auto& imm = b->GetImmutableData();
                  if (!imm.empty()) { outP = imm.data(); outLen = imm.size(); return; }
                  const auto mapped = b->GetMappedSlice();
                  if (mapped.mapPtr) {
                    outP = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    outLen = b->Desc()->ByteWidth;
                  }
                };
                grabH(biVbH.buffer.ptr(), biPtrH, biLenH);
                grabH(bwVbH.buffer.ptr(), bwPtrH, bwLenH);
                const uint32_t biStrideH = biVbH.stride;
                const uint32_t bwStrideH = bwVbH.stride;
                if (biPtrH && bwPtrH && biStrideH > 0 && bwStrideH > 0) {
                  const uint32_t vcountH = static_cast<uint32_t>(
                      std::min(biLenH / biStrideH, bwLenH / bwStrideH));
                  uint32_t minIdx = 255, maxIdx = 0;
                  uint32_t upperHalfVerts = 0;   // any active slot has idx & 0x8
                  uint32_t paletteBits = 0;       // bit k set → palette k (idx/16) touched
                  // First 3 bone indices of first vertex, for a quick sanity check.
                  uint8_t v0i0 = 0, v0i1 = 0, v0i2 = 0;
                  if (vcountH > 0) {
                    const uint8_t* bi0 = biPtrH + biSem->byteOffset;
                    v0i0 = bi0[0]; v0i1 = bi0[1]; v0i2 = bi0[2];
                  }
                  for (uint32_t v = 0; v < vcountH; ++v) {
                    const uint8_t* bi = biPtrH + v * biStrideH + biSem->byteOffset;
                    const int16_t* bw = reinterpret_cast<const int16_t*>(
                        bwPtrH + v * bwStrideH + bwSem->byteOffset);
                    const float w0 = (float(bw[0]) + 1.0f) / 32768.0f;
                    const float w1 = (float(bw[1]) + 1.0f) / 32768.0f;
                    const float w2 = 1.0f - w0 - w1;
                    const float wA[3] = { w0, w1, w2 };
                    bool vHitsUpper = false;
                    for (int k = 0; k < 3; ++k) {
                      if (wA[k] <= 0.001f) continue;
                      const uint32_t idx = bi[k];
                      if (idx < minIdx) minIdx = idx;
                      if (idx > maxIdx) maxIdx = idx;
                      const uint32_t pal = idx / 16u;
                      if (pal < 32u) paletteBits |= (1u << pal);
                      if ((idx & 0x8u) != 0u) vHitsUpper = true;
                    }
                    if (vHitsUpper) ++upperHalfVerts;
                  }
                  std::string vsNameH = "?";
                  auto vsKeyH = m_context->m_state.vs.shader;
                  if (vsKeyH != nullptr && vsKeyH->GetCommonShader() != nullptr) {
                    auto& s = vsKeyH->GetCommonShader()->GetShader();
                    if (s != nullptr) vsNameH = s->getShaderKey().toString().substr(0, 19);
                  }
                  std::string psNameH = "null";
                  auto psKeyH = m_context->m_state.ps.shader;
                  if (psKeyH != nullptr && psKeyH->GetCommonShader() != nullptr) {
                    auto& s = psKeyH->GetCommonShader()->GetShader();
                    if (s != nullptr) psNameH = s->getShaderKey().toString().substr(0, 19);
                  }
                  Logger::info(str::format(
                    "[skin.histo] f=", frameId2,
                    " vs=", vsNameH, " ps=", psNameH,
                    " verts=", vcountH,
                    " biVbPtr=", reinterpret_cast<uintptr_t>(biVbH.buffer.ptr()),
                    " minIdx=", minIdx, " maxIdx=", maxIdx,
                    " upperHalfVerts=", upperHalfVerts,
                    " paletteBits=0x", std::hex, paletteBits, std::dec,
                    " v0idx=(", (int)v0i0, ",", (int)v0i1, ",", (int)v0i2, ")"));
                }
              }
            }
            // NV-DXVK SPIKE DIAG: dump first 10 vertex blend indices/weights
            // and first 5 bone matrices from t30 once per unique VS so we
            // can see if spikes are from bad indices, zero bone slots, or
            // weights outside [0,1].
            {
              static std::unordered_set<uintptr_t> sSkinDumpLogged;
              auto vsKey = m_context->m_state.vs.shader;
              uintptr_t kk = (vsKey != nullptr) ? reinterpret_cast<uintptr_t>(vsKey.ptr()) : 0;
              if (kk && sSkinDumpLogged.insert(kk).second) {
                std::string vsName = "?";
                if (vsKey->GetCommonShader() != nullptr) {
                  auto& s = vsKey->GetCommonShader()->GetShader();
                  if (s != nullptr) vsName = s->getShaderKey().toString().substr(0, 19);
                }
                // Read BI, BW from the respective vertex buffers (slot + byte offset).
                const auto& biVb = m_context->m_state.ia.vertexBuffers[biSem->inputSlot];
                const auto& bwVb = m_context->m_state.ia.vertexBuffers[bwSem->inputSlot];
                const uint8_t* biPtr = nullptr; size_t biLen = 0; uint32_t biStride = 0;
                const uint8_t* bwPtr = nullptr; size_t bwLen = 0; uint32_t bwStride = 0;
                auto grabCpu = [](D3D11Buffer* b, const uint8_t*& outP, size_t& outLen) {
                  if (!b) return;
                  // Try immutable first (CreateBuffer INITIAL_DATA).
                  const auto& imm = b->GetImmutableData();
                  if (!imm.empty()) { outP = imm.data(); outLen = imm.size(); return; }
                  // Fall back to mapped slice.
                  const auto mapped = b->GetMappedSlice();
                  if (mapped.mapPtr) {
                    outP = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    outLen = b->Desc()->ByteWidth;
                  }
                };
                grabCpu(biVb.buffer.ptr(), biPtr, biLen); biStride = biVb.stride;
                grabCpu(bwVb.buffer.ptr(), bwPtr, bwLen); bwStride = bwVb.stride;
                Logger::info(str::format(
                  "[skin.diag] vs=", vsName,
                  " biStride=", biStride, " bwStride=", bwStride,
                  " biOff=", biSem->byteOffset, " bwOff=", bwSem->byteOffset,
                  " biFmt=", (uint32_t)biSem->format, " bwFmt=", (uint32_t)bwSem->format,
                  " biPtr=", biPtr ? 1 : 0, " biLen=", biLen,
                  " bwPtr=", bwPtr ? 1 : 0, " bwLen=", bwLen));
                for (uint32_t v = 0; v < 10 && biPtr && bwPtr; ++v) {
                  const size_t biByte = v * biStride + biSem->byteOffset;
                  const size_t bwByte = v * bwStride + bwSem->byteOffset;
                  if (biByte + 4 > biLen || bwByte + 4 > bwLen) break;
                  const uint8_t* bi = biPtr + biByte;
                  const int16_t* bw = reinterpret_cast<const int16_t*>(bwPtr + bwByte);
                  const float w0 = (float(bw[0]) + 1.0f) / 32768.0f;
                  const float w1 = (float(bw[1]) + 1.0f) / 32768.0f;
                  const float w2 = 1.0f - w0 - w1;
                  Logger::info(str::format(
                    "[skin.vert] v=", v,
                    " idx=(", (int)bi[0], ",", (int)bi[1], ",", (int)bi[2], ",", (int)bi[3], ")",
                    " bwRaw=(", (int)bw[0], ",", (int)bw[1], ")",
                    " w0=", w0, " w1=", w1, " w2=", w2));
                }
                // NV-DXVK: scan the ENTIRE VB for suspicious data. Spikes
                // come from a small number of specific vertices, not the
                // first 10. Compute: max bone index, number of "bad"
                // weight vertices (|w0| or |w1| > 2 or w2 outside [-0.1,1.1]),
                // and number of vertices with the 4th bone slot non-zero.
                if (biPtr && bwPtr && biStride > 0 && bwStride > 0) {
                  const uint32_t vcount = static_cast<uint32_t>(
                      std::min(biLen / biStride, bwLen / bwStride));
                  uint32_t maxIdx0 = 0, maxIdx1 = 0, maxIdx2 = 0, maxIdx3 = 0;
                  uint32_t sumIdx3NonZero = 0;
                  uint32_t badWeightVerts = 0;
                  uint32_t negW2Verts = 0;
                  uint32_t firstBadVert = UINT32_MAX;
                  int firstBadIdx[4] = {0};
                  int firstBadBw[2] = {0};
                  for (uint32_t v = 0; v < vcount; ++v) {
                    const uint8_t* bi = biPtr + v * biStride + biSem->byteOffset;
                    const int16_t* bw = reinterpret_cast<const int16_t*>(
                        bwPtr + v * bwStride + bwSem->byteOffset);
                    if (bi[0] > maxIdx0) maxIdx0 = bi[0];
                    if (bi[1] > maxIdx1) maxIdx1 = bi[1];
                    if (bi[2] > maxIdx2) maxIdx2 = bi[2];
                    if (bi[3] > maxIdx3) maxIdx3 = bi[3];
                    if (bi[3] != 0) ++sumIdx3NonZero;
                    const float w0 = (float(bw[0]) + 1.0f) / 32768.0f;
                    const float w1 = (float(bw[1]) + 1.0f) / 32768.0f;
                    const float w2 = 1.0f - w0 - w1;
                    const bool bad = (w0 < -0.05f || w0 > 1.05f
                                    || w1 < -0.05f || w1 > 1.05f
                                    || w2 < -0.05f || w2 > 1.05f);
                    if (bad) {
                      ++badWeightVerts;
                      if (firstBadVert == UINT32_MAX) {
                        firstBadVert = v;
                        firstBadIdx[0] = bi[0]; firstBadIdx[1] = bi[1];
                        firstBadIdx[2] = bi[2]; firstBadIdx[3] = bi[3];
                        firstBadBw[0] = bw[0]; firstBadBw[1] = bw[1];
                      }
                    }
                    if (w2 < -0.01f) ++negW2Verts;
                  }
                  Logger::info(str::format(
                    "[skin.scan] vs=", vsName,
                    " verts=", vcount,
                    " maxIdx=(", maxIdx0, ",", maxIdx1, ",", maxIdx2, ",", maxIdx3, ")",
                    " idx3NonZeroCount=", sumIdx3NonZero,
                    " badWeightVerts=", badWeightVerts,
                    " negW2Verts=", negW2Verts,
                    " firstBadV=", firstBadVert,
                    " firstBadIdx=(", firstBadIdx[0], ",", firstBadIdx[1], ",", firstBadIdx[2], ",", firstBadIdx[3], ")",
                    " firstBadBw=(", firstBadBw[0], ",", firstBadBw[1], ")"));
                }
                // NV-DXVK: second scan — count vertices with WEIGHT on a
                // bone slot whose index > 7 (outside the first-8 uploaded
                // range). These are the potential spike-producers.
                if (biPtr && bwPtr && biStride > 0 && bwStride > 0) {
                  const uint32_t vcount = static_cast<uint32_t>(
                      std::min(biLen / biStride, bwLen / bwStride));
                  uint32_t spikeCandidates = 0;
                  uint32_t firstSpikeV = UINT32_MAX;
                  int firstSpikeIdx[4] = {0};
                  int firstSpikeBw[2] = {0};
                  for (uint32_t v = 0; v < vcount; ++v) {
                    const uint8_t* bi = biPtr + v * biStride + biSem->byteOffset;
                    const int16_t* bw = reinterpret_cast<const int16_t*>(
                        bwPtr + v * bwStride + bwSem->byteOffset);
                    const float w0 = (float(bw[0]) + 1.0f) / 32768.0f;
                    const float w1 = (float(bw[1]) + 1.0f) / 32768.0f;
                    const float w2 = 1.0f - w0 - w1;
                    // A "spike candidate" has non-zero weight on a bone
                    // slot whose index is outside the first-8 range.
                    const bool bad = (bi[0] > 7 && w0 > 0.001f)
                                  || (bi[1] > 7 && w1 > 0.001f)
                                  || (bi[2] > 7 && w2 > 0.001f);
                    if (bad) {
                      ++spikeCandidates;
                      if (firstSpikeV == UINT32_MAX) {
                        firstSpikeV = v;
                        firstSpikeIdx[0] = bi[0]; firstSpikeIdx[1] = bi[1];
                        firstSpikeIdx[2] = bi[2]; firstSpikeIdx[3] = bi[3];
                        firstSpikeBw[0] = bw[0]; firstSpikeBw[1] = bw[1];
                      }
                    }
                  }
                  Logger::info(str::format(
                    "[skin.spike] vs=", vsName,
                    " verts=", vcount,
                    " spikeCandidates=", spikeCandidates,
                    " firstSpikeV=", firstSpikeV,
                    " firstSpikeIdx=(", firstSpikeIdx[0], ",", firstSpikeIdx[1], ",", firstSpikeIdx[2], ",", firstSpikeIdx[3], ")",
                    " firstSpikeBw=(", firstSpikeBw[0], ",", firstSpikeBw[1], ")",
                    " w0=", (float(firstSpikeBw[0]) + 1.0f) / 32768.0f,
                    " w1=", (float(firstSpikeBw[1]) + 1.0f) / 32768.0f));

                  // NV-DXVK [skin.spike.bones]: for the FIRST spike
                  // vertex, dump the ACTUAL matrices at its 3 referenced
                  // bone slots, with FirstElement applied. This answers:
                  // "is the slot that causes the spike actually zero in
                  // our cache, actually zero on GPU, or actually valid?"
                  if (firstSpikeV != UINT32_MAX && m_hasFullBoneCache) {
                    const uint32_t base = srvFirstElemBones;
                    auto dumpSlot = [&](const char* label, uint32_t bIdx) {
                      const uint32_t absSlot = base + bIdx;
                      const size_t byteOff = size_t(absSlot) * 48u;
                      if (byteOff + 48u > m_fullBoneCache.size()) {
                        Logger::info(str::format(
                          "[skin.spike.bones] ", label,
                          " idx=", bIdx, " absSlot=", absSlot,
                          " OUT_OF_RANGE"));
                        return;
                      }
                      const float* m = reinterpret_cast<const float*>(
                          m_fullBoneCache.data() + byteOff);
                      Logger::info(str::format(
                        "[skin.spike.bones] ", label,
                        " idx=", bIdx, " absSlot=", absSlot,
                        " cachedT=(", m[3], ",", m[7], ",", m[11], ")",
                        " cachedR0=(", m[0], ",", m[1], ",", m[2], ")",
                        " cachedR1=(", m[4], ",", m[5], ",", m[6], ")",
                        " cachedR2=(", m[8], ",", m[9], ",", m[10], ")",
                        " mag(R0)=", std::sqrt(m[0]*m[0]+m[1]*m[1]+m[2]*m[2]),
                        " mag(R1)=", std::sqrt(m[4]*m[4]+m[5]*m[5]+m[6]*m[6]),
                        " mag(R2)=", std::sqrt(m[8]*m[8]+m[9]*m[9]+m[10]*m[10]),
                        " |T|=", (std::fabs(m[3])+std::fabs(m[7])+std::fabs(m[11]))));
                    };
                    dumpSlot("bone0", (uint32_t)firstSpikeIdx[0]);
                    dumpSlot("bone1", (uint32_t)firstSpikeIdx[1]);
                    dumpSlot("bone2", (uint32_t)firstSpikeIdx[2]);
                  }
                }
                // Dump first 5 bone matrices from t30. Prefer the cached
                // copy populated in OnUpdateSubresource (m_fullBoneCache)
                // since t30 is usually a DEFAULT buffer that has no
                // CPU-visible mapping after upload.
                const uint8_t* bonePtr = nullptr;
                if (m_hasFullBoneCache && m_fullBoneCache.size() >= 48) {
                  bonePtr = m_fullBoneCache.data();
                } else {
                  const auto mapped = boneBuf->GetMappedSlice();
                  if (mapped.mapPtr) bonePtr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                }
                for (uint32_t b = 0; b < 5 && bonePtr; ++b) {
                  const float* m = reinterpret_cast<const float*>(bonePtr + b * 48);
                  Logger::info(str::format(
                    "[skin.bone] b=", b,
                    " T=(", m[3], ",", m[7], ",", m[11], ")",
                    " r0=(", m[0], ",", m[1], ",", m[2], ")"));
                }
              }
            }
          }
        }
      }

      // Find the per-vertex/per-instance index semantic. Prefer R32G32B32A32_UINT
      // (BSP), accept R16G16B16A16_UINT (legacy bone). For BSP the semantic is
      // typically per-VERTEX (inst=0); for bone-draws it's per-instance (inst=1).
      // Slice starts at vb.offset + s.byteOffset so the interleave shader can
      // index by vertex without needing semantic-internal offset awareness.
      if (!didSkinnedChar)
      for (const auto& s : semantics) {
        if (s.format == VK_FORMAT_R32G32B32A32_UINT) {
          const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
          if (vb.buffer != nullptr) {
            geo.boneIndexBuffer = RasterBuffer(
              vb.buffer->GetBufferSlice(vb.offset + s.byteOffset),
              0, vb.stride, s.format);
            geo.bonePerVertex       = !s.perInstance;   // per-vertex for BSP
            geo.boneIndexStrideBytes = vb.stride;        // typically 16 (4x uint32)
            geo.boneIndexMask        = 0xFFFFFFFFu;      // full 32-bit index
            // DEBUG
            static uint32_t sBspIdxLog = 0;
            if (sBspIdxLog < 20) {
              ++sBspIdxLog;
              Logger::info(str::format(
                "[D3D11Rtx] BSP idx semantic: name=", s.name,
                " perInst=", s.perInstance ? 1 : 0,
                " stride=", vb.stride, " byteOff=", s.byteOffset,
                " bonePerVertex=", geo.bonePerVertex ? 1 : 0));
            }
          }
          break;
        }
        if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
          const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
          if (vb.buffer != nullptr) {
            geo.boneIndexBuffer = RasterBuffer(
              vb.buffer->GetBufferSlice(vb.offset + s.byteOffset),
              0, vb.stride, s.format);
            geo.bonePerVertex       = false;             // legacy: one bone/draw
            geo.boneIndexStrideBytes = vb.stride;        // typically 8 (4x uint16)
            geo.boneIndexMask        = 0xFFFFu;
          }
          break;
        }
      }
      geo.boneInstanceIndex = m_currentInstanceIndex;
    }

    // Read cull mode from the immutable ID3D11RasterizerState object.
    // Default: no culling (safe fallback when no state is bound).
    geo.cullMode = VK_CULL_MODE_NONE;
    D3D11RasterizerState* rsState = m_context->m_state.rs.state;
    if (rsState) {
      const auto* rsDesc = rsState->Desc();
      switch (rsDesc->CullMode) {
        case D3D11_CULL_NONE:  geo.cullMode = VK_CULL_MODE_NONE;      break;
        case D3D11_CULL_FRONT: geo.cullMode = VK_CULL_MODE_FRONT_BIT; break;
        case D3D11_CULL_BACK:  geo.cullMode = VK_CULL_MODE_BACK_BIT;  break;
      }
      geo.frontFace = rsDesc->FrontCounterClockwise
        ? VK_FRONT_FACE_COUNTER_CLOCKWISE
        : VK_FRONT_FACE_CLOCKWISE;
    }

    // Compute vertex count — must cover the highest vertex index accessed by
    // this draw so Remix doesn't read out of bounds when building BLAS.
    // geo.vertexCount is the buffer capacity; the hash uses a tighter subrange.
    const uint32_t maxVBVertices = posBuffer.stride() > 0
      ? static_cast<uint32_t>(posBuffer.length() / posBuffer.stride())
      : count;
    uint32_t drawVertexCount;
    uint32_t hashStart, hashCount;
    if (!indexed) {
      // Non-indexed Draw(count, start): vertices [start, start+count) accessed.
      drawVertexCount = std::min(start + count, maxVBVertices);
      hashStart = std::min(start, maxVBVertices);
      hashCount = std::min(count, maxVBVertices - hashStart);
    } else {
      // Indexed DrawIndexed(indexCount, startIndex, base): vertex = index + base.
      // The OLD comment said we couldn't know max(index) without scanning the IB,
      // so it fell back to base + indexCount — which is wrong. BSP/static geometry
      // shares a large vertex buffer across many draws, and draws with few indices
      // frequently reference vertices far above `base + indexCount`. That caused
      // Remix to report vertexCount far smaller than the real range, the BLAS
      // builder saw "valid" primitiveCount but indices ≥ maxVertex → MMU fault
      // reading beyond the vertex buffer → VK_ERROR_DEVICE_LOST on frame 3+.
      //
      // FIX: scan the index buffer CPU-side to find the actual max index.
      // For DYNAMIC source, read mapped slice directly. For STATIC source,
      // try mapPtr (often host-visible for small buffers). If scan isn't
      // possible, fall back to full buffer capacity (safe: over-reports).
      const uint32_t baseU = static_cast<uint32_t>(std::max(base, 0));
      uint32_t maxIdxSeen = 0;
      bool scanned = false;
      if (indexed) {
        const auto& ib = m_context->m_state.ia.indexBuffer;
        if (ib.buffer != nullptr) {
          const uint32_t idxStride = (ib.format == DXGI_FORMAT_R32_UINT) ? 4u : 2u;
          const void* src = nullptr;
          // DYNAMIC: use current mapped slice (race-safe on our thread).
          if (ib.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC) {
            const auto mapped = ib.buffer->GetMappedSlice();
            src = mapped.mapPtr;
          }
          // STATIC: some immutable buffers are host-visible for staging.
          if (src == nullptr) {
            src = ib.buffer->GetBuffer()->mapPtr(0);
          }
          if (src != nullptr) {
            const size_t bufSize = ib.buffer->Desc()->ByteWidth;
            const size_t startOff = ib.offset + size_t(start) * idxStride;
            const size_t readLen = size_t(count) * idxStride;
            if (startOff + readLen <= bufSize) {
              const uint8_t* p = reinterpret_cast<const uint8_t*>(src) + startOff;
              if (idxStride == 2) {
                const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
                for (uint32_t i = 0; i < count; ++i)
                  if (q[i] > maxIdxSeen) maxIdxSeen = q[i];
              } else {
                const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
                for (uint32_t i = 0; i < count; ++i)
                  if (q[i] > maxIdxSeen) maxIdxSeen = q[i];
              }
              scanned = true;
            }
          }
        }
      }
      if (scanned) {
        // BLAS builder needs maxVertex = highest index + 1 (inclusive range).
        // Add base offset since BLAS input vertices are [base, base + maxVtx].
        drawVertexCount = std::min(baseU + maxIdxSeen + 1u, maxVBVertices);
        hashStart = std::min(baseU, maxVBVertices);
        hashCount = std::min(maxIdxSeen + 1u, maxVBVertices - hashStart);
      } else {
        // Couldn't scan — fall back to the FULL vertex buffer capacity
        // (over-reports but safe: BLAS builder can't read past buffer).
        drawVertexCount = maxVBVertices;
        hashStart = std::min(baseU, maxVBVertices);
        hashCount = std::min(count, maxVBVertices - hashStart);
        // Log fallbacks so we know if static idx buffers aren't mappable.
        static uint32_t sFallbackLog = 0, sFallbackCount = 0;
        ++sFallbackCount;
        if (sFallbackLog < 20 || (sFallbackCount % 500) == 0) {
          ++sFallbackLog;
          const auto& ib = m_context->m_state.ia.indexBuffer;
          Logger::warn(str::format("[IDX-SCAN-FALLBACK] #", sFallbackCount,
            " idxBuf=0x", std::hex,
            (uintptr_t)(ib.buffer != nullptr ? ib.buffer.ptr() : nullptr),
            std::dec,
            " usage=", (ib.buffer != nullptr ? uint32_t(ib.buffer->Desc()->Usage) : 0u),
            " count=", count, " start=", start, " base=", base,
            " maxVBVertices=", maxVBVertices,
            " → drawVertexCount=full buffer (over-reporting)"));
        }
      }
    }
    if (drawVertexCount == 0)
      drawVertexCount = count;
    if (hashCount == 0)
      hashCount = std::min(count, maxVBVertices);
    geo.vertexCount = drawVertexCount;

    geo.futureGeometryHashes = ComputeGeometryHashes(geo, drawVertexCount,
                                                     hashStart, hashCount);
    if (!geo.futureGeometryHashes.valid()) {
      BumpFilter(FilterReason::HashFailed);
      return;
    }

    DrawCallState dcs;
    dcs.geometryData     = geo;
    dcs.transformData    = ExtractTransforms();

    // NV-DXVK (TF2 skinned chars): if the earlier RasterGeometry setup flagged
    // this draw as a skinned character (BLENDINDICES+BLENDWEIGHT+t30), set
    // objectToWorld = identity. Verified via DXBC disassembly of
    // VS_ef94e6c7fcc3c144: the bone matrices in t30 store ABSOLUTE world
    // transforms (not camera-relative). The game's VS applies skinning then
    // subtracts c_cameraOrigin explicitly to get camera-relative positions
    // before the clip transform. Since the interleaver's weighted skinning
    // already produces world-space positions, objectToWorld should be
    // identity — adding fanoutCam would double the camera offset.
    if (m_skinnedCharNeedsCamOffset) {
      // NV-DXVK: skinned character path. Per DXBC disassembly of
      // VS_ef94e6c7fcc3c144, t30 bone matrices for the PLAYER CHARACTER
      // hold ABSOLUTE WORLD transforms — interleaver output is world space,
      // o2w = identity.
      //
      // NV-DXVK TF2 VIEWMODEL FIX: TF2 draws the first-person viewmodel
      // (gun + hands) through the SAME VS_ef94e6c7 but with viewport
      // MaxDepth <= 0.05 and c_cameraOrigin=(0,0,0) in cb2. Its bone
      // matrices are VIEW-LOCAL, not world, so interleaver output is in
      // view space. With identity o2w the BLAS sits at world origin —
      // thousands of units from Main camera → invisible. Detect via
      // viewport MaxDepth and apply o2w = inverse(worldToView) so the
      // view-local geometry ends up at the camera in world space.
      float vpMaxDepth = 1.0f;
      if (m_context->m_state.rs.numViewports > 0)
        vpMaxDepth = m_context->m_state.rs.viewports[0].MaxDepth;
      // NV-DXVK: viewmodel detection — use viewport MaxDepth as the only
      // reliable signal. w2v≈0 is NOT viewmodel-specific: it also fires
      // when ExtractTransforms defaults to identity worldToView, which
      // would misroute the PLAYER CHARACTER through the viewmodel o2w
      // path and double-shift it off-screen.
      const bool isViewModelDraw = (vpMaxDepth <= 0.08f);

      if (isViewModelDraw) {
        // Use the CACHED MAIN camera's worldToView (captured from the most
        // recent valid world-space draw) instead of this draw's own
        // worldToView. The viewmodel's cb2 view-to-clip has weird XY/Z
        // scaling baked in (factor ~185 on Y/Z), so inverting it produces a
        // matrix that crushes the viewmodel mesh to near-zero thickness.
        // The main camera's worldToView is a proper orthonormal rotation +
        // translation and inverts cleanly to a usable viewToWorld.
        Matrix4 mainW2v;
        bool haveMainW2v = false;
        {
          std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
          if (!isIdentityExact(m_lastGoodTransforms.worldToView)) {
            mainW2v = m_lastGoodTransforms.worldToView;
            haveMainW2v = true;
          }
        }
        if (haveMainW2v) {
          dcs.transformData.objectToWorld = inverse(mainW2v);
        } else {
          // Fallback: inverse of this draw's worldToView (bad scale but
          // better than nothing).
          dcs.transformData.objectToWorld = inverse(dcs.transformData.worldToView);
        }
        dcs.transformData.objectToView = Matrix4(); // identity (already view-space)
        m_lastO2wPathId = 12; // viewmodel: o2w = mainViewToWorld (BLAS in view space)
        static uint32_t sVmPathLog = 0;
        if (sVmPathLog < 10) {
          ++sVmPathLog;
          const auto& o2w = dcs.transformData.objectToWorld;
          Logger::info(str::format(
            "[D3D11Rtx.o2w.viewmodel] path=12 vpMaxZ=", vpMaxDepth,
            " usedMainW2v=", (haveMainW2v ? 1 : 0),
            " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
            " o2wDiag=(", o2w[0][0], ",", o2w[1][1], ",", o2w[2][2], ")"));
        }
      } else {
        dcs.transformData.objectToWorld = Matrix4(); // identity
        // NV-DXVK TF2: w2v rescue for path 11 (skinned characters incl. gun
        // + hands). The bone interleaver bakes vertex positions in WORLD
        // space (bone.T is world-space), so the BLAS sits at real world
        // coords like (-5179, 279, 92). For the BLAS-vs-camera projection
        // to land correctly, w2v MUST carry the real camera translation.
        //
        // Path 1 of ExtractTransforms intermittently produces a w2v with
        // gameplay rotation but ZERO translation (camera-relative-style
        // matrix) for the same VS_ef94e6c7 draw across consecutive frames.
        // The existing isIdentityExact rescue at the path-10 sites doesn't
        // catch this because the rotation is real — only translation is
        // zero. Result: Remix's RtCamera lands at world origin, rays fire
        // from there, BLAS at (-5179, 279, 92) is never hit. Body has 61
        // bones spanning a wide volume so it sometimes happens to clip a
        // ray; gun has 6 bones in a tight cluster and is fully missed.
        //
        // Detect zero-translation specifically and substitute the last
        // cached good w2v. Threshold of 1.0 is generous: any plausible
        // gameplay camera origin is hundreds-to-thousands of units from
        // world origin, so |T|<1 unambiguously means broken extraction.
        const auto& w2v0 = dcs.transformData.worldToView;
        const float w2vTMag2 =
          w2v0[3][0]*w2v0[3][0] + w2v0[3][1]*w2v0[3][1] + w2v0[3][2]*w2v0[3][2];
        bool didW2vRescue = false;
        if (w2vTMag2 < 1.0f) {
          std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
          const auto& cached = m_lastGoodTransforms.worldToView;
          const float cachedTMag2 =
            cached[3][0]*cached[3][0] + cached[3][1]*cached[3][1] + cached[3][2]*cached[3][2];
          if (cachedTMag2 >= 1.0f) {
            dcs.transformData.worldToView      = cached;
            dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
            didW2vRescue = true;
          }
        }
        if (didW2vRescue) {
          static uint32_t sPath11W2vRestore = 0;
          if (sPath11W2vRestore < 20) {
            ++sPath11W2vRestore;
            const auto& w2v = dcs.transformData.worldToView;
            Logger::info(str::format(
              "[D3D11Rtx.path11.w2vRestore] drawID=", m_drawCallID,
              " vs=", m_currentVsHashCache.substr(0, 19),
              " restored cached w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")"));
          }
        }
        // NV-DXVK TF2 VIEWMODEL: direct-translate fallback for the gun +
        // hands. The native rasterizer projects these verts visible because
        // cb2's c_cameraRelativeToClip uses X as the depth axis (Source
        // engine convention: world X = forward). Path 1's reconstructed
        // worldToView matrix produces a "fwd" vector along world +Y for
        // Remix's RtCamera, which is the OPPOSITE convention. Result: the
        // bone-skinned gun verts at world (-5164, 269, 71) — visible to
        // cb2 — sit BEHIND Remix's camera in its +Y-forward frame, so
        // primary rays never hit them.
        //
        // Pragmatic fix: detect the gun draws (VS_ef94e6c7 + srvFirstElem
        // >= 672, captured into m_vmFirstElem / m_vmBoneRoot) and force
        // an o2w that translates the world-baked BLAS to a position 30
        // units along Remix's fwd direction in world space. Doesn't
        // track ADS / recoil precisely (game-side logic still updates
        // bone positions, but the offset to Remix-fwd is constant), but
        // makes the gun visible in Remix's view, which is the priority.
        // NV-DXVK TF2: vmHack direct-translate disabled. Was meant to force
        // the gun + hands BLAS into Remix's camera frustum by overriding
        // o2w, but the m_vmFirstElem state variable persists across draws
        // so the `>= 672` check sometimes tripped on body draws that
        // followed a gun submit in the same frame, dragging the player
        // body into an incorrect position (boot-overhead artifact).
        //
        // The PROPER fix (in progress, Parts 1-4 of the plan) reconstructs
        // worldToView and viewToProjection so Remix's RtCamera convention
        // matches cb2's c_cameraRelativeToClip exactly, making hacks like
        // this unnecessary. Keep the code in-place (disabled) until parts
        // 2 + 4 are verified end-to-end — easier to reinstate as a
        // temporary fallback if needed than to re-author from the log.
        const bool isVmHack = false;
        if (isVmHack) {
          // Pull camera world position + Remix-fwd from cached good
          // transforms (set by path 1 on every valid main-cam draw).
          std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
          const Matrix4& w2v = m_lastGoodTransforms.worldToView;
          const float tMag2 =
            w2v[3][0]*w2v[3][0] + w2v[3][1]*w2v[3][1] + w2v[3][2]*w2v[3][2];
          if (tMag2 >= 1.0f) {
            const Matrix4 v2w = inverse(w2v);
            const Vector3 camWorld(v2w[3][0], v2w[3][1], v2w[3][2]);
            // worldToView col 2 = Remix's "fwd" axis interpretation.
            const Vector3 fwd(w2v[2][0], w2v[2][1], w2v[2][2]);
            const float fwdLen = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
            const Vector3 fwdN = (fwdLen > 1e-3f) ? Vector3(fwd.x/fwdLen, fwd.y/fwdLen, fwd.z/fwdLen)
                                                  : Vector3(0.0f, 1.0f, 0.0f);
            const float kOffset = 30.0f;
            // Desired BLAS-anchor position: 30 units in front of camera.
            const Vector3 desired(camWorld.x + fwdN.x * kOffset,
                                  camWorld.y + fwdN.y * kOffset,
                                  camWorld.z + fwdN.z * kOffset);
            // The interleaver bakes verts at world coords near
            // m_vmBoneRoot (= bone[0] world translation, e.g.
            // (-5203, 241, 63)). Shift = desired - boneRoot.
            const float sx = desired.x - m_vmBoneRoot[0];
            const float sy = desired.y - m_vmBoneRoot[1];
            const float sz = desired.z - m_vmBoneRoot[2];
            // Build a translate-only o2w (column-major Matrix4 ctor).
            dcs.transformData.objectToWorld = Matrix4(
              Vector4(1.0f, 0.0f, 0.0f, 0.0f),
              Vector4(0.0f, 1.0f, 0.0f, 0.0f),
              Vector4(0.0f, 0.0f, 1.0f, 0.0f),
              Vector4(sx,   sy,   sz,   1.0f));
            static uint32_t sVmHackLog = 0;
            if (sVmHackLog < 20) {
              ++sVmHackLog;
              Logger::info(str::format(
                "[D3D11Rtx.vmHack] drawID=", m_drawCallID,
                " camWorld=(", camWorld.x, ",", camWorld.y, ",", camWorld.z, ")",
                " fwd=(", fwdN.x, ",", fwdN.y, ",", fwdN.z, ")",
                " boneRoot=(", m_vmBoneRoot[0], ",", m_vmBoneRoot[1], ",", m_vmBoneRoot[2], ")",
                " desired=(", desired.x, ",", desired.y, ",", desired.z, ")",
                " shift=(", sx, ",", sy, ",", sz, ")"));
            }
          }
        }
        if (!isIdentityExact(dcs.transformData.worldToView))
          dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
        else
          dcs.transformData.objectToView = dcs.transformData.objectToWorld;
        m_lastO2wPathId = 11; // skinned char: identity (BLAS in world)
        static std::unordered_set<std::string> sSkinPath11Logged;
        const std::string vk = m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
        if (sSkinPath11Logged.insert(vk).second) {
          Logger::info(str::format(
            "[D3D11Rtx.o2w.skinnedChar] vs=", vk, " path=11 identity_o2w"));
        }
      }
      // Fall through to submit, NOT filter.
    }

    // NV-DXVK: TLAS coherence filter + matrix finiteness guard.
    // Fires for BOTH non-instanced (OnDraw/OnDrawIndexed → SubmitDraw) and
    // instanced (OnDrawInstanced/OnDrawIndexedInstanced → SubmitInstancedDraw
    // → SubmitDraw) paths since everything funnels here.
    //
    // (1) TLAS coherence: reject draws whose c_cameraOrigin doesn't match the
    //     Main camera's world position within kEpsilon. Different cameras mean
    //     different BLAS placements → TLAS mixes coord spaces → pathological
    //     bounds → ray traversal can run effectively forever → GPU TDR.
    // (2) Finiteness guard: reject draws whose objectToWorld matrix has any
    //     non-finite component or absurd translation magnitude. Observed in TF2
    //     where game cbuffers occasionally contain NaN (VS s2[10]=(-nan,...)).
    {
      const auto& m = dcs.transformData.objectToWorld;
      bool badMatrix = false;
      for (int r = 0; r < 4 && !badMatrix; ++r) {
        for (int c = 0; c < 4 && !badMatrix; ++c) {
          if (!std::isfinite(m[r][c])) badMatrix = true;
        }
      }
      constexpr float kMaxComponentMagnitude = 1.0e7f; // TF2 coords are ~1e4
      for (int r = 0; r < 4 && !badMatrix; ++r) {
        if (std::abs(m[3][r]) > kMaxComponentMagnitude) badMatrix = true;
      }
      if (badMatrix) {
        static uint32_t sBadMatLog = 0;
        if (sBadMatLog < 20) {
          ++sBadMatLog;
          Logger::err(str::format(
            "[TLAS-FILTER] reject draw=", m_drawCallID,
            " non-finite/absurd o2w: T=(", m[3][0], ",", m[3][1], ",", m[3][2], ")",
            " diag=(", m[0][0], ",", m[1][1], ",", m[2][2], ")"));
        }
        BumpFilter(FilterReason::FullscreenQuad);
        return;
      }
    }

    // The filter compares EVERY draw's WORLD-SPACE camera position against
    // Main's. Per-draw position is derived as inverse(worldToView)[3].xyz()
    // — same construction RtCamera::getPosition uses internally — so both
    // sides share an identical coordinate convention. Comparing raw
    // worldToView[3] columns directly fails because RtCamera's matCache
    // sometimes overwrites WorldToView with identity (depending on
    // freeCameraViewRelative()), but getPosition() always returns a valid
    // world position derived from the original view-to-world.
    Vector3 drawCamPos;
    {
      const Matrix4 v2w = inverse(dcs.transformData.worldToView);
      drawCamPos = Vector3(v2w[3][0], v2w[3][1], v2w[3][2]);
    }

    if (m_context->m_device != nullptr) {
      auto& sceneMgr = m_context->m_device->getCommon()->getSceneManager();
      auto& camMgr = sceneMgr.getCameraManager();
      auto& mainCam = camMgr.getCamera(CameraType::Main);
      const uint32_t frameId = m_context->m_device->getCurrentFrameId();
      // Only trust Main's position if the CLASSIFIER (not safety net) latched
      // it in the last few frames. Safety-net Main is whatever ExtractTransforms
      // produced — often identity/(-1,-1,-1) during menus/cinematics — and
      // would otherwise cause us to reject every real world draw.
      const bool classifierLatched = camMgr.isMainSetByClassifier();
      const uint32_t lastClassifierFrame = camMgr.getMainClassifierFrameId();
      constexpr uint32_t kMaxStaleFrames = 5; // allow last ~5 frames after latch
      const bool mainRecentlyLatched =
        classifierLatched
        && (frameId <= lastClassifierFrame
            || (frameId - lastClassifierFrame) <= kMaxStaleFrames);
      const bool mainEverValid = mainRecentlyLatched;

      // Per-frame stats — reset when we see the drawCallID counter roll over.
      static uint32_t s_tlasFilterFrame = UINT32_MAX;
      static uint32_t s_tlasAccept = 0;
      static uint32_t s_tlasReject = 0;
      static uint32_t s_tlasNoMain = 0;
      static uint32_t s_tlasPrevID = UINT32_MAX;
      if (m_drawCallID == 0 || m_drawCallID < s_tlasPrevID) {
        if (s_tlasFilterFrame != UINT32_MAX
            && (s_tlasAccept + s_tlasReject + s_tlasNoMain) > 0
            && s_tlasFilterFrame < 600) {
          Logger::info(str::format(
            "[TLAS-FILTER] frame=", s_tlasFilterFrame,
            " accept=", s_tlasAccept,
            " reject=", s_tlasReject,
            " noMain=", s_tlasNoMain));
        }
        s_tlasFilterFrame = (s_tlasFilterFrame == UINT32_MAX) ? 0 : s_tlasFilterFrame + 1;
        s_tlasAccept = 0;
        s_tlasReject = 0;
        s_tlasNoMain = 0;
      }
      s_tlasPrevID = m_drawCallID;

      if (mainEverValid) {
        // RtCamera::getPosition returns the world-space camera position,
        // derived from inverse(worldToView). Same convention as drawCamPos
        // above.
        const Vector3 mainCamPos = mainCam.getPosition(/*freecam=*/false);
        const float dx = drawCamPos.x - mainCamPos.x;
        const float dy = drawCamPos.y - mainCamPos.y;
        const float dz = drawCamPos.z - mainCamPos.z;
        const float d2 = dx*dx + dy*dy + dz*dz;
        // Big epsilon. TF2 world coords are ~1e4; draws in Main's coord space
        // share Main's worldToView translation exactly. Draws from other
        // cameras (shadow, viewmodel, reflection, cinematic origin) differ
        // by hundreds-thousands of units. 100 cleanly separates clusters.
        constexpr float kEpsilon = 100.0f;
        const bool mismatch = d2 > (kEpsilon * kEpsilon);

        if (mismatch) {
          struct Key { int x, y, z; };
          static std::vector<Key> seenOrigins;
          Key k{ int(drawCamPos.x), int(drawCamPos.y), int(drawCamPos.z) };
          bool seen = false;
          for (const auto& s : seenOrigins) {
            if (s.x == k.x && s.y == k.y && s.z == k.z) { seen = true; break; }
          }
          if (!seen && seenOrigins.size() < 32) {
            seenOrigins.push_back(k);
            Logger::info(str::format(
              "[TLAS-FILTER] new foreign cam #", seenOrigins.size(),
              " draw=", m_drawCallID, " frame=", s_tlasFilterFrame,
              " drawCamPos=(", drawCamPos.x, ",", drawCamPos.y, ",", drawCamPos.z, ")",
              " mainCamPos=(", mainCamPos.x, ",", mainCamPos.y, ",", mainCamPos.z, ")",
              " |delta|=", std::sqrt(d2)));
          }

          // NV-DXVK: rejection DISABLED. Filter runs on the D3D11 thread
          // BEFORE classification (which happens on CS thread inside
          // processCameraData). Rejecting a draw here prevents it from
          // reaching the classifier, which prevents Main from re-latching
          // when the gameplay camera moves. Net effect: Main froze at the
          // first latch and every subsequent gameplay draw was "foreign"
          // by stale comparison. Keep counting/logging for diagnostic, but
          // let the draw through. Coord-space coherence has to be enforced
          // downstream of the classifier (e.g., in RtInstanceManager when
          // building the TLAS), not pre-classification.
          ++s_tlasReject;
          // BumpFilter(FilterReason::FullscreenQuad);
          // return;
        } else {
          ++s_tlasAccept;
        }
      } else {
        // No Main yet — permit the draw through. Log once per session so we
        // know the filter is observing but passing during the first-frame gap.
        static bool sNoMainLogged = false;
        if (!sNoMainLogged) {
          sNoMainLogged = true;
          Logger::info(str::format(
            "[TLAS-FILTER] no Main latched yet at frame ", frameId,
            " — passing draws through until Main is available"));
        }
        ++s_tlasNoMain;
      }
    }

    // NV-DXVK: scene dump for cbuffer-based BSP draws (non-fanout). The
    // bone-instance fanout dump above only catches g_modelInst-style draws.
    // Anything that uses CBufModelInstance (cbuffer-based world matrix)
    // never reaches the fanout — that's where ground/walls likely live.
    // Skip if fanout already handled this draw.
    if (m_currentInstancesToObject == nullptr
        && SceneDump::shouldDumpThisFrame()
        && posSem
        && posSem->format == VK_FORMAT_R32G32_UINT) {
      std::lock_guard<std::mutex> lk(SceneDump::g_mutex);
      const bool firstOpen = !SceneDump::g_obj.is_open();
      SceneDump::open();
      if (firstOpen && SceneDump::g_obj.is_open()) {
        SceneDump::writeCameraMarker();
      }
      if (SceneDump::g_obj.is_open()) {
        const auto& pvb = m_context->m_state.ia.vertexBuffers[posSem->inputSlot];
        const uint8_t* posData = nullptr; size_t posLen = 0;
        if (pvb.buffer != nullptr) {
          const auto& imm = pvb.buffer->GetImmutableData();
          if (!imm.empty()) {
            posData = imm.data() + pvb.offset + posSem->byteOffset;
            posLen  = imm.size() - (pvb.offset + posSem->byteOffset);
          }
        }
        const uint8_t* idxData = nullptr; size_t idxLen = 0;
        VkIndexType ixType = VK_INDEX_TYPE_UINT16;
        if (indexed) {
          const auto& ib = m_context->m_state.ia.indexBuffer;
          if (ib.buffer != nullptr) {
            const auto& imm = ib.buffer->GetImmutableData();
            if (!imm.empty()) {
              idxData = imm.data() + ib.offset;
              idxLen  = imm.size() - ib.offset;
              ixType  = (ib.format == DXGI_FORMAT_R16_UINT)
                          ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            }
          }
        }
        if (posData && (!indexed || idxData)) {
          const Matrix4& T = dcs.transformData.objectToWorld;
          const uint32_t posStride = std::max<uint32_t>(8u, pvb.stride);
          const float kScale = 1.0f / 1024.0f;
          const float kBiasZ = -2048.0f;
          SceneDump::g_obj << "o BSP_CB_" << SceneDump::g_objectsWritten++
                           << "_v" << dcs.geometryData.vertexCount << "\n";
          if (!indexed) {
            for (uint32_t v = 0; v < count; ++v) {
              size_t off = static_cast<size_t>(v) * posStride;
              if (off + 8 > posLen) break;
              const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
              uint32_t xi = SceneDump::decodeX(up[0]);
              uint32_t yi = SceneDump::decodeY(up[0], up[1]);
              uint32_t zi = SceneDump::decodeZ(up[1]);
              float lx = float(xi) * kScale - 1024.0f;
              float ly = float(yi) * kScale - 1024.0f;
              float lz = float(zi) * kScale + kBiasZ;
              float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
              float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
              float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
              SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
            }
            const uint32_t triCount = count / 3;
            for (uint32_t t = 0; t < triCount; ++t) {
              uint32_t a = SceneDump::g_baseVtx + t * 3 + 1;
              SceneDump::g_obj << "f " << a << " " << (a+1) << " " << (a+2) << "\n";
            }
            SceneDump::g_baseVtx += count;
          } else {
            const uint32_t idxStride = (ixType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
            uint32_t maxV = 0;
            for (uint32_t i = 0; i < count; ++i) {
              size_t io = static_cast<size_t>(start + i) * idxStride;
              if (io + idxStride > idxLen) { maxV = 0; break; }
              uint32_t idx = (idxStride == 2)
                ? *reinterpret_cast<const uint16_t*>(idxData + io)
                : *reinterpret_cast<const uint32_t*>(idxData + io);
              idx += static_cast<uint32_t>(std::max(base, 0));
              if (idx > maxV) maxV = idx;
            }
            const uint32_t vCount = maxV + 1;
            for (uint32_t v = 0; v < vCount; ++v) {
              size_t off = static_cast<size_t>(v) * posStride;
              if (off + 8 > posLen) break;
              const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
              uint32_t xi = SceneDump::decodeX(up[0]);
              uint32_t yi = SceneDump::decodeY(up[0], up[1]);
              uint32_t zi = SceneDump::decodeZ(up[1]);
              float lx = float(xi) * kScale - 1024.0f;
              float ly = float(yi) * kScale - 1024.0f;
              float lz = float(zi) * kScale + kBiasZ;
              float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
              float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
              float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
              SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
            }
            const uint32_t triCount = count / 3;
            for (uint32_t t = 0; t < triCount; ++t) {
              uint32_t i0base = (start + t * 3);
              size_t i0o = static_cast<size_t>(i0base + 0) * idxStride;
              size_t i1o = static_cast<size_t>(i0base + 1) * idxStride;
              size_t i2o = static_cast<size_t>(i0base + 2) * idxStride;
              if (i2o + idxStride > idxLen) break;
              uint32_t i0 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i0o) : *reinterpret_cast<const uint32_t*>(idxData + i0o);
              uint32_t i1 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i1o) : *reinterpret_cast<const uint32_t*>(idxData + i1o);
              uint32_t i2 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i2o) : *reinterpret_cast<const uint32_t*>(idxData + i2o);
              i0 += static_cast<uint32_t>(std::max(base, 0));
              i1 += static_cast<uint32_t>(std::max(base, 0));
              i2 += static_cast<uint32_t>(std::max(base, 0));
              SceneDump::g_obj << "f " << (SceneDump::g_baseVtx + i0 + 1) << " "
                                       << (SceneDump::g_baseVtx + i1 + 1) << " "
                                       << (SceneDump::g_baseVtx + i2 + 1) << "\n";
            }
            SceneDump::g_baseVtx += vCount;
          }
        }
      }
    }

    // Reject NDC-space screen quads now that ExtractTransforms has cached the VP.
    if (isNdcScreenQuad) {
      BumpFilter(FilterReason::FullscreenQuad);
      return;
    }

    // NV-DXVK: UI / overlay / pre-3D filter.  ExtractTransforms flips this
    // flag when it can't find any perspective projection in the game's
    // cbuffers and has to fall back to the viewport-derived synthetic
    // camera.  That situation means one of three things, all of which
    // should keep the draw OUT of the RTX pipeline:
    //   1. The game is rendering a 2D UI / HUD / menu (Source engine main
    //      menu — this is the Titanfall 2 case).
    //   2. The game is playing a video (Bink through a fullscreen quad or
    //      textured blit).
    //   3. The game hasn't bound any cbuffers yet (very early boot frame).
    // In every case the native DXVK D3D11 raster path — which was already
    // recorded by the EmitCs([=] (DxvkContext* ctx) { ctx->draw*(); })
    // call inside D3D11DeviceContext::Draw* BEFORE we were invoked — will
    // rasterize the draw correctly to the bound render target.  Skipping
    // RTX submission just means it won't be ray-traced.  Combined with the
    // drawCallID-gated safety net below, this lets the native backbuffer
    // content pass through injectRTX() unchanged (it no-ops when the
    // camera is invalid), matching exactly what D3D9 Remix does via
    // isRenderingUI() + RtxGeometryStatus::Rasterized.
    if (m_lastExtractUsedFallback) {
      // NV-DXVK: If a previous draw in this frame already found a real
      // VP, reuse those transforms instead of filtering this draw as UI.
      // Source only populates the VP cbuffer on draws 250+ (main opaque
      // pass), but draws 1-249 (shadows, depth prepass, etc.) are still
      // real gameplay geometry that should be ray-traced.
      //
      // EXCEPT: if the V2 classifier definitively identified this draw
      // as UI (screenspace 2D with no real transform), the cached-VP
      // reuse would put a 2D NDC quad into the world TLAS where it
      // renders as nothing. Force the TRUE UI branch so native raster
      // gets to draw the HUD/menu.
      // Use m_hasEverFoundProj (session-latched) instead of only the
      // per-frame flag. Early draws of a frame (pre-projection-extraction,
      // e.g. drawID 0-169 before the main VP cbuffer is bound) would
      // otherwise hit the "TRUE UI-class" branch and get filtered as
      // UIFallback, losing real gameplay geometry. Cached transforms from
      // the last frame's extraction are a better fallback than rejecting
      // the draw entirely — the gameplay camera doesn't teleport between
      // frames, so reusing last frame's w2v is visually indistinguishable.
      if ((m_foundRealProjThisFrame || m_hasEverFoundProj) && !m_lastClassifierSaidUi) {
        // NV-DXVK: Take a consistent snapshot of cached transforms under
        // the mutex. Writes happen on the immediate-context thread; reads
        // happen on deferred-context threads. Without the lock, deferred
        // reads could see a torn matrix (half old, half new) or a stale
        // all-zero value that was never updated from that thread's cache
        // perspective — producing spurious degenerate_cached_w2v filters
        // even when the immediate thread has populated real values.
        DrawCallTransforms cachedSnap;
        {
          std::lock_guard<std::mutex> lk(m_lastGoodTransformsMutex);
          cachedSnap = m_lastGoodTransforms;
        }
        const auto& cached = cachedSnap.worldToView;
        // NV-DXVK [diag]: log what value is actually being CONSUMED at
        // injectRTX time. Compare these to [cachedSave] entries — if save
        // shows player movement but consume keeps reading the same stale
        // value, the bug is between save and consume (probable: another
        // thread / snapshot timing). Logs only on consumed-value change
        // so output is clean (one line per actually-different read).
        {
          static float sLastCx = 1e30f, sLastCy = 1e30f, sLastCz = 1e30f;
          static uint64_t sConsumeN = 0;
          const float dx = cached[3][0] - sLastCx;
          const float dy = cached[3][1] - sLastCy;
          const float dz = cached[3][2] - sLastCz;
          if ((dx*dx + dy*dy + dz*dz) > 0.0001f) {
            sLastCx = cached[3][0]; sLastCy = cached[3][1]; sLastCz = cached[3][2];
            // Reconstruct world-space cam from cached w2v so we can see
            // exactly which world position the ray tracer is rendering
            // from. Compare to [cachedSave]'s playerCam to verify the
            // consume side isn't using a stale/wrong pose.
            const float ccX =
              -(cached[0][0]*cached[3][0] + cached[0][1]*cached[3][1] + cached[0][2]*cached[3][2]);
            const float ccY =
              -(cached[1][0]*cached[3][0] + cached[1][1]*cached[3][1] + cached[1][2]*cached[3][2]);
            const float ccZ =
              -(cached[2][0]*cached[3][0] + cached[2][1]*cached[3][1] + cached[2][2]*cached[3][2]);
            const float fanX = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.x : 0.f;
            const float fanY = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.y : 0.f;
            const float fanZ = m_hasFanoutCamOrigin ? m_lastFanoutCamOrigin.z : 0.f;
            const float ccDelta = std::sqrt(
              (ccX - fanX) * (ccX - fanX) +
              (ccY - fanY) * (ccY - fanY) +
              (ccZ - fanZ) * (ccZ - fanZ));
            Logger::info(str::format(
              "[cachedConsume] @", m_drawCallID, " (rawDraw=", m_rawDrawCount, ")",
              " vs=", m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u)),
              " w2v=(", cached[3][0], ",", cached[3][1], ",", cached[3][2], ")",
              " camPos=(", ccX, ",", ccY, ",", ccZ, ")",
              " playerCam=(", fanX, ",", fanY, ",", fanZ, ")",
              " camDelta=", ccDelta,
              " R=(", cached[0][0], ",", cached[1][0], ",", cached[2][0], ")",
              " F=(", cached[0][2], ",", cached[1][2], ",", cached[2][2], ")",
              " consumes=", ++sConsumeN));
          }
        }
        if (cached[3][0] == 0.0f && cached[3][1] == 0.0f && cached[3][2] == 0.0f) {
          BumpFilter(FilterReason::UIFallback);
          m_lastDrawFilteredAsUI = true;
          // NV-DXVK: cached camera was never populated (we've never
          // extracted a real VP) — treat as HUD-class for hash logging.
          m_lastDrawIsHudClass = true;
          LogPsHashesForHudFilter("UIFallback.degen_w2v");
          {
            static std::unordered_set<std::string> sDegenVs;
            const std::string vkd = m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
            if (sDegenVs.insert(vkd).second) {
              Logger::info(str::format(
                "[UIFallback.reason] vs=", vkd,
                " drawID=", m_drawCallID,
                " site=degenerate_cached_w2v",
                " hasEverFoundProj=", m_hasEverFoundProj ? 1 : 0,
                " foundRealProjThisFrame=", m_foundRealProjThisFrame ? 1 : 0,
                " addr=", reinterpret_cast<uintptr_t>(&m_lastGoodTransforms),
                " thisRtx=", reinterpret_cast<uintptr_t>(this)));
            }
          }
          return;
        }
        // NV-DXVK: Only reuse the CAMERA transforms (viewToProjection,
        // worldToView) — NOT objectToWorld which is per-object and was
        // already extracted for THIS draw by ExtractTransforms.  The
        // previous version copied the entire m_lastGoodTransforms
        // including objectToWorld from draw #251, which gave every
        // subsequent draw the same world transform → all objects at
        // the same position → overlapping degenerate BLAS → GPU hang.
        // Use the snapshot taken under lock above (not m_lastGoodTransforms)
        // so we don't re-read the cross-thread static here.
        dcs.transformData.viewToProjection = cachedSnap.viewToProjection;
        dcs.transformData.worldToView      = cachedSnap.worldToView;
        // Recompute objectToView from the corrected worldToView +
        // this draw's own objectToWorld.
        dcs.transformData.objectToView = dcs.transformData.objectToWorld;
        if (!isIdentityExact(dcs.transformData.worldToView))
          dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
        // NV-DXVK: Reject draws where objectToView translation is extreme.
        // Shadow/depth passes use light-space transforms that, when combined
        // with the main camera VP, produce positions far from the camera
        // → huge BLAS → GPU TDR.
        {
          const auto& o2v = dcs.transformData.objectToView;
          const float maxT = 100000.0f;
          if (std::abs(o2v[3][0]) > maxT || std::abs(o2v[3][1]) > maxT || std::abs(o2v[3][2]) > maxT ||
              !std::isfinite(o2v[3][0]) || !std::isfinite(o2v[3][1]) || !std::isfinite(o2v[3][2])) {
            BumpFilter(FilterReason::UIFallback);
            {
              static std::unordered_set<std::string> sExtremeVs;
              const std::string vke = m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
              if (sExtremeVs.insert(vke).second) {
                Logger::info(str::format(
                  "[UIFallback.reason] vs=", vke,
                  " drawID=", m_drawCallID,
                  " site=extreme_o2v",
                  " o2vT=(", o2v[3][0], ",", o2v[3][1], ",", o2v[3][2], ")"));
              }
            }
            // NOTE: do NOT set m_lastDrawFilteredAsUI — this is shadow/depth
            // rejection not actual UI. Keep native raster suppressed.
            return;
          }
        }
        // Fall through to submit to RTX.
        static uint32_t s_reusedCount = 0;
        if (s_reusedCount < 100) {
          ++s_reusedCount;
          const auto& T = dcs.transformData;
          Logger::info(str::format(
              "[D3D11Rtx] Reusing frame VP for fallback draw #",
              m_drawCallID, " (rawDraw=", m_rawDrawCount, ")",
              " o2w T=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
              " w2v T=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")"));
        }
      } else {
        // NV-DXVK: TRUE UI-class draw — no real projection has been found in
        // any prior draw of this frame. Flag for OnDraw* to allow native
        // rasterization so the menu/HUD at least enters the backbuffer.
        BumpFilter(FilterReason::UIFallback);
        m_lastDrawFilteredAsUI = true;
        // NV-DXVK: classifier or projection scan definitively says UI —
        // safe to treat as HUD-class for PS-hash logging.
        m_lastDrawIsHudClass = true;
        LogPsHashesForHudFilter("UIFallback.true_ui");

        {
          static std::unordered_set<std::string> sTrueUiVs;
          const std::string vkt = m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
          if (sTrueUiVs.insert(vkt).second) {
            Logger::info(str::format(
              "[UIFallback.reason] vs=", vkt,
              " drawID=", m_drawCallID,
              " site=true_ui",
              " foundRealProjThisFrame=", m_foundRealProjThisFrame ? 1 : 0,
              " classifierSaidUi=", m_lastClassifierSaidUi ? 1 : 0,
              " hasEverFoundProj=", m_hasEverFoundProj ? 1 : 0));
          }
        }
        return;
      }
    }

    // Apply per-instance world transform when submitting instanced draws.
    if (instanceTransform) {
      dcs.transformData.objectToWorld = *instanceTransform;
      m_lastO2wPathId = 9;  // per-instance override (fanout tforms)
      // Recompute objectToView with the per-instance world matrix.
      dcs.transformData.objectToView = dcs.transformData.objectToWorld;
      if (!isIdentityExact(dcs.transformData.worldToView))
        dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
      std::string vsH = m_currentVsHashCache.empty()
        ? std::string("<novs>") : m_currentVsHashCache.substr(0, 19);
      Logger::info(str::format(
        "[D3D11Rtx.o2w.fanout] vs=", vsH,
        " drawID=", m_drawCallID,
        " inst.T=(", (*instanceTransform)[3][0], ",",
        (*instanceTransform)[3][1], ",", (*instanceTransform)[3][2], ")"));
    }

    // NV-DXVK: For bone-instanced draws with instancesToObject.
    // t31 matrix IS the world transform (from shader decompilation).
    // BLAS = localPos (bone buffers stripped), objectToWorld = identity,
    // instancesToObject[i] = t31_mat[i] places in world directly.
    if (m_boneInstanceCount > 0 && m_currentInstancesToObject) {
      static uint32_t sSubmitLog = 0;
      if (sSubmitLog < 10) {
        ++sSubmitLog;
        const auto& w2v = dcs.transformData.worldToView;
        const auto& o2w = dcs.transformData.objectToWorld;
        const auto& i2o0 = (*m_currentInstancesToObject)[0];
        Logger::info(str::format(
          "[D3D11Rtx] SubmitBone: camPos(w2v.T)=(", -w2v[3][0], ",", -w2v[3][1], ",", -w2v[3][2], ")",
          " origO2W.T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " i2o0.T=(", i2o0[3][0], ",", i2o0[3][1], ",", i2o0[3][2], ")",
          " finalInst0=(", o2w[3][0] + i2o0[3][0], ",",
          o2w[3][1] + i2o0[3][1], ",", o2w[3][2] + i2o0[3][2], ")"));
      }
      // t31 contains VIEW-SPACE transforms (view * model). Need inv(worldToView)
      // to get world-space matrices. Pre-compute inv(w2v) once per SubmitDraw
      // and multiply with each instance's transform.
      //
      // Game shader: clipPos = cb2 * t31[i] * localPos  (cb2 = projection only)
      // We want:     worldPos = inv(worldToView) * t31[i] * localPos
      // => instancesToObject[i] = inv(worldToView) * t31[i]
      //
      // Since we can't modify the vector here (it's shared, stable pointer),
      // we apply the inverse via objectToWorld. Each i2o[i] is already t31[i],
      // so objectToWorld = inv(worldToView) gives the same math.
      dcs.transformData.instancesToObject = m_currentInstancesToObject;
      // NV-DXVK: Pass ownership too so it flows into RtInstance via instance_manager.
      dcs.transformData.instancesToObjectOwner = m_currentInstancesToObjectOwner;
      // t31 is already the full world-space model transform.
      // objectToWorld = identity, instancesToObject = t31[i], done.
      dcs.transformData.objectToWorld = Matrix4();
      m_lastO2wPathId = 10;  // bone-instanced fanout: identity o2w

      // NV-DXVK CRITICAL: If ExtractTransforms produced an identity w2v
      // for this draw (observed: COMMIT w2vT=(0,0,0) o2vT=(0,0,0)), the
      // main RT camera ends up at world origin and rays never hit the
      // real-world geometry at (-5179, 279, 92). Fall back to the last
      // good cached w2v so the fanout path always has a real camera.
      if (isIdentityExact(dcs.transformData.worldToView)
          && !isIdentityExact(m_lastGoodTransforms.worldToView)) {
        dcs.transformData.worldToView      = m_lastGoodTransforms.worldToView;
        dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
        static uint32_t sPath10W2vRestore = 0;
        if (sPath10W2vRestore < 10) {
          ++sPath10W2vRestore;
          const auto& w2v = dcs.transformData.worldToView;
          Logger::info(str::format(
            "[D3D11Rtx.path10.w2vRestore] drawID=", m_drawCallID,
            " restored cached w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")"));
        }
      }
      dcs.transformData.objectToView = dcs.transformData.worldToView;

      // (TestPos log removed — inv(w2v) not used)

      static uint32_t sPostLog = 0;
      if (sPostLog < 5) {
        ++sPostLog;
        const auto& o2w = dcs.transformData.objectToWorld;
        // Compute magnitude of each row of the 3x3 rotation part
        auto col_mag = [&](int c) {
          float s = o2w[c][0]*o2w[c][0] + o2w[c][1]*o2w[c][1] + o2w[c][2]*o2w[c][2];
          return std::sqrt(s);
        };
        Logger::info(str::format(
          "[D3D11Rtx] InvW2V: T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " col0=(", o2w[0][0], ",", o2w[0][1], ",", o2w[0][2], ") mag=", col_mag(0),
          " col1=(", o2w[1][0], ",", o2w[1][1], ",", o2w[1][2], ") mag=", col_mag(1),
          " col2=(", o2w[2][0], ",", o2w[2][1], ",", o2w[2][2], ") mag=", col_mag(2)));
      }
      dcs.geometryData.boneMatrixBuffer = RasterBuffer();
      dcs.geometryData.boneIndexBuffer = RasterBuffer();
      geo.boneMatrixBuffer = RasterBuffer();
      geo.boneIndexBuffer = RasterBuffer();

      // DEBUG: skip actual RTX submit for bone-instanced draws (isolate non-instanced)
      if (m_debugHideBoneInstanced) {
        return;
      }
    }
    // NV-DXVK: For bone-instanced draws with attached bone buffers (N-draw path),
    // the interleaver applies the bone transform. Set objectToWorld to identity.
    else if (m_attachBoneBuffers && geo.boneMatrixBuffer.defined()) {
      dcs.transformData.objectToWorld = Matrix4();
      m_lastO2wPathId = 10;  // bone-instanced (N-draw path): identity o2w
      // Same camera-rescue as the fanout branch above.
      if (isIdentityExact(dcs.transformData.worldToView)
          && !isIdentityExact(m_lastGoodTransforms.worldToView)) {
        dcs.transformData.worldToView      = m_lastGoodTransforms.worldToView;
        dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
      }
      dcs.transformData.objectToView = dcs.transformData.worldToView;
    }

    // Let processCameraData() classify the camera from the matrices.
    // Hardcoding Main would bypass Remix's sky/portal/shadow detection.
    dcs.cameraType       = CameraType::Unknown;
    dcs.usesVertexShader = (m_context->m_state.vs.shader != nullptr);
    dcs.usesPixelShader  = (m_context->m_state.ps.shader != nullptr);

    // NV-DXVK: Deterministic pass classifier — pass the current D3D11 viewport
    // to Remix so camera_manager can distinguish gameplay draws (viewport ==
    // back buffer) from shadow cascades / cubemaps / RT targets (off-size or
    // square viewports). No matrix heuristics involved.
    {
      const auto& vps = m_context->m_state.rs.viewports;
      if (vps[0].Width > 0.0f && vps[0].Height > 0.0f) {
        dcs.transformData.viewportWidth  = vps[0].Width;
        dcs.transformData.viewportHeight = vps[0].Height;
      }
    }

    // NV-DXVK: Capture bound VS hash for game-native per-draw identification.
    // Gameplay-world passes use a small, stable set of vertex shaders; fullscreen /
    // post / UI draws use different ones even when they share a projection shape.
    // Keying Main-camera classification off this hash eliminates the need for
    // matrix-property heuristics (aspect/tinyScale/maxZ/w2vT).
    dcs.transformData.worldToViewPathId = m_lastWtvPathId;
    if (m_context->m_state.vs.shader != nullptr) {
      auto* common = m_context->m_state.vs.shader->GetCommonShader();
      if (common != nullptr) {
        const auto& dxvkShader = common->GetShader();
        if (dxvkShader != nullptr) {
          dcs.transformData.vertexShaderHash =
            static_cast<XXH64_hash_t>(dxvkShader->getHash());
        }
      }
    }

    // D3D11 shaders are always SM 4.0+.
    if (dcs.usesVertexShader)
      dcs.vertexShaderInfo = ShaderProgramInfo{4, 0};
    if (dcs.usesPixelShader)
      dcs.pixelShaderInfo = ShaderProgramInfo{4, 0};
    dcs.zWriteEnable     = zWriteEnable;
    dcs.zEnable          = zEnable;
    dcs.stencilEnabled   = stencilEnabled;
    dcs.drawCallID       = m_drawCallID++;
    m_lastDrawCaptured   = true;  // Signal caller to skip D3D11 rasterization
    // NV-DXVK: record the successful submit against the current VS hash.
    if (!m_currentVsHashCache.empty())
      ++m_vsFrameStats[m_currentVsHashCache].submitted;
    // NV-DXVK [VMHunt.result=pass]: suspect draw reached COMMIT. Report
    // the o2w path id so we know what transform treatment it got.
    if (m_vmHuntIsSuspect) {
      const auto& o2w = dcs.transformData.objectToWorld;
      Logger::info(str::format(
        "[VMHunt.result] count=", m_vmHuntIndexCount,
        " vs=", m_currentVsHashCache.substr(0, 19),
        " verdict=PASS o2wPathId=", m_lastO2wPathId,
        " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
      m_vmHuntIsSuspect = false; // consumed
    }

    // NV-DXVK: orientation probe — log the world-space directions that each
    // object's LOCAL +X/+Y/+Z axes map to, plus translation. No identity
    // filter — BSP uses pure-translation objectToWorld and we want to see
    // where BSP chunks are placed too.
    //
    // Log only the FIRST occurrence per VS hash so BSP (high-count shader)
    // doesn't flood and we still see prop/foliage variety.
    {
      static std::unordered_set<XXH64_hash_t> sLoggedHashes;
      static uint32_t sOrientLog = 0;
      const XXH64_hash_t vsH = dcs.transformData.vertexShaderHash;
      if (sOrientLog < 50 && sLoggedHashes.count(vsH) == 0) {
        sLoggedHashes.insert(vsH);
        ++sOrientLog;
        const auto& o = dcs.transformData.objectToWorld;
        char vsHex[32];
        std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                      static_cast<unsigned long long>(vsH));
        Logger::info(str::format(
          "[D3D11Rtx.orient] #", sOrientLog,
          " draw=", dcs.drawCallID,
          " vs=", vsHex,
          " localX_w=(", o[0][0], ",", o[0][1], ",", o[0][2], ")",
          " localY_w=(", o[1][0], ",", o[1][1], ",", o[1][2], ")",
          " localZ_w=(", o[2][0], ",", o[2][1], ",", o[2][2], ")",
          " T_w=(", o[3][0], ",", o[3][1], ",", o[3][2], ")"));
      }
    }

    // Viewport depth range from D3D11_VIEWPORT.MinDepth / MaxDepth.
    {
      const auto& vp = m_context->m_state.rs.viewports[0];
      dcs.minZ = std::clamp(vp.MinDepth, 0.0f, 1.0f);
      dcs.maxZ = std::clamp(vp.MaxDepth, 0.0f, 1.0f);
    }

    // NV-DXVK TF2 VIEWMODEL: previously this routed gun + hands draws
    // (VS_ef94e6c7, srvFirstElem >= 672) through the ViewModel pipeline by
    // forcing dcs.maxZ to 0.05. That pipeline runs a perspective-correction
    // transform `mainViewToWorld · mainProjToView · vmProj · scale ·
    // vmCam.worldToView` designed for engines where `mainProj ≠ vmProj`. In
    // TF2 the two projections share the same FoV (74.7°) so the correction
    // collapses to ~identity and `createViewModelInstance` ends up writing
    // the BLAS instance at world origin (0,0,0) — the gun is then drawn at
    // origin while the camera looks at (-5179, 279, 92), invisible.
    //
    // After fixes elsewhere (interleaver Z-offset = -2048, dropped wSum
    // renormalization, path-11 w2v rescue), the bone interleaver bakes gun
    // vertices into the BLAS at correct world coords (e.g. (-5164, 269, 71))
    // and path 11 keeps them as identity-o2w world-space geometry. The
    // gun then renders correctly without going through the broken VM
    // pipeline. Disable vmRoute entirely; the BLAS-in-world path handles it.
    //
    // ADS / recoil tracking comes for free from the bone matrices themselves
    // — the game updates the per-vertex skinning bones each frame to encode
    // the gun's current world position relative to the eye.
    if (false && m_skinnedCharNeedsCamOffset && m_vmFirstElem >= 672u && m_vmBoneRootValid) {
      dcs.maxZ = 0.05f;
      static uint32_t sVmRouteLog = 0;
      if (sVmRouteLog < 10) {
        ++sVmRouteLog;
        Logger::info(str::format(
          "[D3D11Rtx.vmRoute] srvFirst=", m_vmFirstElem,
          " boneRoot=(", m_vmBoneRoot[0], ",", m_vmBoneRoot[1], ",", m_vmBoneRoot[2], ")",
          " forcing dcs.maxZ=0.05 → ViewModel classifier"));
      }
    }

    // D3D11 has no legacy fog — engines bake fog into shaders.
    // FogState defaults to mode=0 (none), which is correct.

    // Register this context as the active rendering context so the primary
    // swap chain routes EndFrame/OnPresent through us, not a video-playback
    // device that happened to present first.
    FillMaterialData(dcs.materialData);

    // NV-DXVK: TF2 worldspace VGUI — propagate the PS-RDEF-detected unlit
    // UI flag from the material side to the geometry side. The interleaver
    // checks geometryData.vguiLayoutEnable to decide whether to write the
    // 8 extra per-vertex floats (TEXCOORD1.zw + TEXCOORD2.xy + TEXCOORD3
    // .xyzw int4). The TEXCOORD3 buffer was already captured upstream
    // (search for vguiTc3Sem) so it's available even on non-VGUI draws —
    // we just gate the interleaver write here.
    // NV-DXVK: VGUI path guard log — fires at the gate where sourceIsUnlitUI
    // gets bridged to the BLAS-side vguiLayoutEnable. If sourceIsUnlitUI is
    // true but vguiTexcoord3Buffer is undefined, the IA capture didn't get
    // the int4 stream → the entire VGUI extras path silently no-ops.
    {
      static std::unordered_set<XXH64_hash_t> sVguiGuardLogged;
      static std::mutex sVguiGuardMu;
      if (dcs.materialData.sourceIsUnlitUI) {
        XXH64_hash_t vsH = 0, psH = 0;
        GetCurrentVsPsHashes(vsH, psH);
        bool firstSeen = false;
        {
          std::lock_guard<std::mutex> lk(sVguiGuardMu);
          firstSeen = sVguiGuardLogged.insert(vsH).second;
        }
        if (firstSeen) {
          Logger::info(str::format(
            "[VguiGuard] VS=0x", std::hex, vsH, " PS=0x", psH, std::dec,
            " sourceIsUnlitUI=1",
            " geo.vguiTexcoord3Buffer.defined=",
              (dcs.geometryData.vguiTexcoord3Buffer.defined() ? 1 : 0),
            " geo.vguiGlyphDimsBuffer.defined=",
              (dcs.geometryData.vguiGlyphDimsBuffer.defined() ? 1 : 0),
            " geo.texcoordBuffer.defined=",
              (dcs.geometryData.texcoordBuffer.defined() ? 1 : 0),
            " geo.texcoordBuffer.fmt=",
              (dcs.geometryData.texcoordBuffer.defined()
                ? uint32_t(dcs.geometryData.texcoordBuffer.vertexFormat()) : 0u),
            " geo.texcoordBuffer.stride=",
              (dcs.geometryData.texcoordBuffer.defined()
                ? dcs.geometryData.texcoordBuffer.stride() : 0u),
            " WILL_FLIP_VguiLayout=",
              ((dcs.materialData.sourceIsUnlitUI
                && dcs.geometryData.vguiTexcoord3Buffer.defined()) ? 1 : 0)));
        }
      }
    }

    if (dcs.materialData.sourceIsUnlitUI && dcs.geometryData.vguiTexcoord3Buffer.defined()) {
      dcs.geometryData.vguiLayoutEnable = true;
      // NV-DXVK: VGUI texcoord routing. VGUI's TEXCOORD1 (4-float xyzw =
      // primary glyph quad pos in xy + secondary in zw) needs to land in
      // texcoordBuffer so surfaceInteraction.textureCoordinates ends up as
      // the primary quad pos (= atlas UV) for the SDF evaluator. Doing
      // this HERE (rather than in IA capture) gates the override on
      // sourceIsUnlitUI — non-VGUI shaders that also happen to have a
      // 4-float TC1 (character/sprite VSes carrying tangent space in TC1)
      // keep their original TC0 routing untouched.
      if (dcs.geometryData.texcoord1Buffer.defined()
          && dcs.geometryData.texcoord1Buffer.vertexFormat() == VK_FORMAT_R32G32B32A32_SFLOAT) {
        dcs.geometryData.texcoordBuffer = dcs.geometryData.texcoord1Buffer;
        dcs.geometryData.texcoord1Buffer = RasterBuffer();
      }

      // NV-DXVK: capture the 3 VGUI structured-buffer SRVs (g_fontBounds,
      // g_imgBounds, g_styles) so SceneManager can track them in
      // m_bufferCache and stamp the resulting bindless indices onto
      // RtSurface::vgui*BufferIndex. SceneManager runs later on the CS
      // thread, so we hold onto the underlying DxvkBuffer via the
      // RasterBuffer's slice (which owns an Rc<DxvkBuffer>) — this keeps
      // the bytes alive across the deferred boundary even if D3D11 renames
      // the source buffer mid-frame. Slot identification is by RDEF name,
      // not by slot index, because TF2 ships VGUI shader variants with
      // different t-slot orderings (verified empirically on the gauntlet
      // weapon descriptor vs. training drone stats panel).
      const auto& ps2 = m_context->m_state.ps;
      const auto* cs2 = ps2.shader != nullptr
        ? ps2.shader->GetCommonShader() : nullptr;
      if (cs2 != nullptr) {
        struct VguiSbRole {
          const char* names[3];
          dxvk::RasterBuffer* dst;
        };
        const VguiSbRole roles[] = {
          { { "g_fontBounds", "fontBounds", nullptr }, &dcs.geometryData.vguiFontBoundsBuffer },
          { { "g_imgBounds",  "imgBounds",  nullptr }, &dcs.geometryData.vguiImgBoundsBuffer },
          { { "g_styles",     "styles",     nullptr }, &dcs.geometryData.vguiStylesBuffer },
        };
        for (const VguiSbRole& r : roles) {
          uint32_t slot = UINT32_MAX;
          for (const char* name : r.names) {
            if (!name) break;
            uint32_t s = cs2->FindResourceSlot(name);
            if (s != UINT32_MAX) { slot = s; break; }
          }
          if (slot == UINT32_MAX) continue;
          if (slot >= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) continue;
          D3D11ShaderResourceView* srv = ps2.shaderResources.views[slot].ptr();
          if (!srv) continue;
          if (srv->GetResourceType() != D3D11_RESOURCE_DIMENSION_BUFFER) continue;
          Rc<DxvkBufferView> view = srv->GetBufferView();
          if (view == nullptr) continue;
          // Get the structure stride from the underlying D3D11 buffer.
          ID3D11Resource* res = nullptr;
          srv->GetResource(&res);
          if (res == nullptr) continue;
          D3D11Buffer* d3dbuf = static_cast<D3D11Buffer*>(res);
          const uint32_t structStride = d3dbuf->Desc()->StructureByteStride;
          // GetResource AddRef'd; release immediately — view->slice() keeps
          // the underlying Rc<DxvkBuffer> alive on its own.
          res->Release();
          if (structStride == 0) continue;
          // Wrap the SRV's buffer-view slice as a RasterBuffer. Format is
          // R32_UINT to match dxvk's structured-buffer view convention
          // (see d3d11_view_srv.cpp:53-57); slang reads with byte-address
          // semantics via BUFFER_ARRAY so the format is mostly nominal.
          *r.dst = dxvk::RasterBuffer(view->slice(), 0u, structStride, VK_FORMAT_R32_UINT);
        }

        // NV-DXVK: VGUI source-buffer ground-truth dump. One-shot per VS.
        // Reads the FULL contents of the 3 structured-buffer SRVs (font
        // bounds / img bounds / styles) to verify which byte offsets
        // carry which fields. The slang evaluator's style decode depends
        // on these offsets being correctly mapped.
        static std::unordered_set<XXH64_hash_t> sVguiSbDumpLogged;
        static std::mutex sVguiSbDumpMu;
        XXH64_hash_t vsH = 0, psH = 0;
        GetCurrentVsPsHashes(vsH, psH);
        bool firstVgui = false;
        {
          std::lock_guard<std::mutex> lk(sVguiSbDumpMu);
          firstVgui = sVguiSbDumpLogged.insert(vsH).second;
        }
        if (firstVgui) {
          struct SbInfo {
            const char* tag;
            uint32_t slotName;  // slot index found
            const char* name;
            const dxvk::RasterBuffer* dst;
            uint32_t entriesToDump;  // dump first N records
          };
          // Re-find the slots so we can map each role's slot back.
          const SbInfo dumps[] = {
            { "fontBounds", 0u, "g_fontBounds", &dcs.geometryData.vguiFontBoundsBuffer, 4u },
            { "imgBounds",  0u, "g_imgBounds",  &dcs.geometryData.vguiImgBoundsBuffer,  4u },
            { "styles",     0u, "g_styles",     &dcs.geometryData.vguiStylesBuffer,     16u },
          };
          for (const SbInfo& di : dumps) {
            if (!di.dst->defined()) {
              Logger::info(str::format("[VguiSbDump.", di.tag, "] not captured"));
              continue;
            }
            const auto& rb = *di.dst;
            const uint32_t stride = rb.stride();
            const uint32_t lengthBytes = uint32_t(rb.length());
            const uint32_t totalEntries = stride > 0 ? (lengthBytes / stride) : 0;
            const uint32_t dumpEntries = std::min(di.entriesToDump, totalEntries);
            const auto* bufPtr = rb.buffer().ptr();
            const void* mapPtr = bufPtr ? bufPtr->mapPtr(rb.offsetFromSlice()) : nullptr;
            if (mapPtr == nullptr) {
              Logger::info(str::format("[VguiSbDump.", di.tag, "] mapPtr null"
                " (DEVICE_LOCAL? need staging readback) stride=", stride,
                " entries=", totalEntries));
              continue;
            }
            // Format-specific dump: fontBounds/imgBounds = float4 (16-byte
            // stride); styles = 24 floats (96-byte stride).
            const float* fp = static_cast<const float*>(mapPtr);
            for (uint32_t e = 0; e < dumpEntries; ++e) {
              const uint32_t offFloats = e * (stride / 4u);
              if (stride == 16) {
                Logger::info(str::format("[VguiSbDump.", di.tag,
                  "[", e, "]] (",
                  fp[offFloats + 0], ", ",
                  fp[offFloats + 1], ", ",
                  fp[offFloats + 2], ", ",
                  fp[offFloats + 3], ")"));
              } else if (stride == 96) {
                // 24 floats per style record — split into 6 lines of 4.
                std::string out;
                for (uint32_t i = 0; i < 24u; ++i) {
                  if (i % 4u == 0u) out += " | ";
                  out += std::to_string(fp[offFloats + i]);
                  out += " ";
                }
                Logger::info(str::format("[VguiSbDump.", di.tag,
                  "[", e, "]]", out.c_str()));
              } else {
                Logger::info(str::format("[VguiSbDump.", di.tag,
                  "[", e, "]] unhandled stride=", stride));
                break;
              }
            }
          }
        }
      }
    }

    // NV-DXVK: auto-detect decals from D3D11 rasterizer/depth/blend state.
    // Remix's existing isDecal classification (rtx_types.cpp:385-388) only
    // looks up texture hashes against curated lists in rtx.conf — useless
    // for fresh game integrations. TF2's decals are flagged unmistakably by
    // D3D11 state: DepthBias < 0 (push toward camera) + DepthWrite=0 (don't
    // poison depth) + alpha-blend (ONE, INV_SRC_ALPHA) + DepthFunc=LESS_EQUAL.
    // Confirmed via [BspRastState] log: PS=0xefdf8de6, 0x8e734d411c, 0x6ba2c535
    // and friends all match this pattern. Without this auto-detect, those
    // decal triangles compete with proper-UV walls in the opaque BVH and
    // win the depth tie up close (entire wall flips magenta in our sentinel).
    //
    // With InstanceCategories::DecalNoOffset set, downstream
    // (rtx_instance_manager.cpp:1241+) routes the geometry to the unordered
    // TLAS with FORCE_NO_OPAQUE_BIT_KHR and assigns a decalSortOrder, so
    // they alpha-blend on top of walls correctly — exactly what native does.
    {
      // Read rasterizer state.
      float rsDepthBias = 0.0f;
      float rsSlopeBias = 0.0f;
      D3D11RasterizerState* rs = m_context->m_state.rs.state;
      if (rs) {
        D3D11_RASTERIZER_DESC2 rd = {};
        rs->GetDesc(reinterpret_cast<D3D11_RASTERIZER_DESC*>(&rd));
        rsDepthBias = float(rd.DepthBias);
        rsSlopeBias = rd.SlopeScaledDepthBias;
      }
      // Read depth-stencil write mask.
      bool dsDepthWrite = true;
      D3D11DepthStencilState* dsds = m_context->m_state.om.dsState;
      if (dsds) {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsds->GetDesc(&dsd);
        dsDepthWrite = (dsd.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ZERO);
      }
      // The decal pattern: TF2 uses DepthBias=-16 (any negative is decal-ish),
      // DepthWrite=0, BlendEnable=1 with src=ONE,dst=INV_SRC_ALPHA. Use
      // (negative-bias OR slope-negative) AND (depth-write disabled) AND
      // (alpha-blend enabled) — all three together strongly indicate decal.
      const auto& bm = dcs.materialData.blendMode;
      const bool hasNegBias = (rsDepthBias < 0.0f) || (rsSlopeBias < 0.0f);
      const bool isAlphaBlend = bm.enableBlending != VK_FALSE
        && bm.colorSrcFactor == VK_BLEND_FACTOR_ONE
        && bm.colorDstFactor == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      const bool isDecalPattern = hasNegBias && !dsDepthWrite && isAlphaBlend;
      if (isDecalPattern) {
        dcs.setCategory(InstanceCategories::DecalNoOffset, true);
        // One-shot per-PS log so we can verify which PSes auto-classify.
        XXH64_hash_t vsH_d = 0, psH_d = 0;
        GetCurrentVsPsHashes(vsH_d, psH_d);
        static std::unordered_set<uint64_t> sDecalAutoLogged;
        static std::mutex sDecalAutoMu;
        bool first = false;
        {
          std::lock_guard<std::mutex> g(sDecalAutoMu);
          first = sDecalAutoLogged.insert(uint64_t(psH_d)).second;
        }
        if (first) {
          Logger::info(str::format(
            "[AutoDecal] PS=0x", std::hex, psH_d, std::dec,
            " VS=0x", std::hex, vsH_d, std::dec,
            " depthBias=", rsDepthBias, " slopeBias=", rsSlopeBias,
            " depthWrite=", dsDepthWrite ? 1 : 0,
            " blendEnable=", uint32_t(bm.enableBlending),
            " src=", uint32_t(bm.colorSrcFactor), " dst=", uint32_t(bm.colorDstFactor)));
        }
      }
    }

    // NV-DXVK: extract TF2/Source "CBufUberStatic" UV transform and plant it
    // into dcs.transformData.textureTransform. Without this the raytracer
    // samples albedo with raw mesh UVs, which for BSP are stored in WORLD
    // UNITS (U=32.5, V=64.1) so the texture tiles thousands of times per
    // surface and mip-averages to the texture's mean colour → flat wall.
    // TF2's color pass multiplies:
    //   uv' = float2(dot(c_uv1RotScaleX, uv) + c_uv1Translate.x,
    //                dot(c_uv1RotScaleY, uv) + c_uv1Translate.y)
    // We build the equivalent Matrix4 so Remix's micromap/sampling paths
    // do the same multiply before the texture fetch.
    //
    // IMPORTANT GATE: character/weapon/prop PSes also declare CBufUberStatic
    // but mark these specific fields as [unused] (verified via fxc /dumpbin
    // on FS_7a6e4c57…). The game app may not write valid values for those
    // draws — the cbuffer carries stale scales from a previous BSP draw. If
    // we apply those stale values to characters/weapons we destroy their
    // UVs. Gate on D3DReflect's D3D_SVF_USED flag per field (via
    // ReadsCBField) — ground-truth "is this variable actually sampled?",
    // populated by populateFieldUsage() at shader-compile time. This
    // replaces the HasColorOutput() workaround that mis-fired on
    // character/weapon color passes which declare CBufUberStatic but
    // never sample the UV-transform fields.
    // NV-DXVK: VS cbuffer field discovery for the 3D→2D BSP UV projection.
    // The PS-side `c_uv1RotScaleX/Y/Translate` block below is a 2D→2D
    // transform applied to already-2D VB UVs — useless when the VB UVs are
    // collapsed to a 1D line (verified for BSP wall geom: txcoords[1]==[2]
    // exactly, world tri area=347101 — game's VS computes the real UV from
    // world position via cbuffer-driven planar projection). Dump every
    // cbuffer field the BSP VS declares, once per unique VS hash, so we can
    // identify the world-projection vectors (likely two 4-vectors holding
    // U-axis and V-axis world-space coefficients, or a 4x2 matrix).
    {
      if (const auto* vsShader = m_context->m_state.vs.shader.ptr()) {
        if (const auto* vsCs = vsShader->GetCommonShader()) {
          XXH64_hash_t vsH_dump = 0, psH_dump = 0;
          GetCurrentVsPsHashes(vsH_dump, psH_dump);
          const bool isBspVs =
               vsH_dump == 0x7c38fdf4358d5527ull
            || vsH_dump == 0x0990ac503e694beeull
            || vsH_dump == 0x1953b6e9cc252e4eull
            || vsH_dump == 0xe7abcf4ea24b0fa7ull
            || vsH_dump == 0x448e372f6d5e78e1ull;
          if (isBspVs) {
            static std::unordered_set<uint64_t> sDumpedVs;
            static std::mutex sDumpedVsMu;
            bool firstTime = false;
            {
              std::lock_guard<std::mutex> g(sDumpedVsMu);
              firstTime = sDumpedVs.insert(uint64_t(vsH_dump)).second;
            }
            if (firstTime) {
              const auto cbufs = vsCs->GetCBufferNamesAndSlots();
              for (const auto& cb : cbufs) {
                Logger::info(str::format(
                  "[BspVsCBuf] VS=0x", std::hex, vsH_dump, std::dec,
                  " cbName=", cb.first, " slot=", cb.second));
              }
              const auto resNames = vsCs->GetResourceNamesAndSlots();
              for (const auto& kv : resNames) {
                Logger::info(str::format(
                  "[BspVsRes] VS=0x", std::hex, vsH_dump, std::dec,
                  " name=", kv.first, " slot=", kv.second));
              }

              // NV-DXVK: dump the first 208 bytes of g_modelInst[0] for BSP
              // VSes that use the per-instance modelToWorld lookup. The base
              // 4x3 matrix only fills 48 bytes; the remaining 160 bytes per
              // instance likely hold the per-prop UV-projection axes (Source
              // engine convention for axis-aligned BSP texturing). Reading
              // this should reveal whether the per-instance struct contains
              // float4 axis vectors that we can replay on the raytrace side.
              {
                uint32_t modelInstSlot = vsCs->FindResourceSlot("g_modelInst");
                if (modelInstSlot != UINT32_MAX
                    && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
                  auto* srv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
                  if (srv) {
                    Com<ID3D11Resource> resCom;
                    srv->GetResource(&resCom);
                    Com<ID3D11Buffer> bufCom;
                    if (resCom != nullptr && SUCCEEDED(resCom->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&bufCom)))) {
                      auto* buf = static_cast<D3D11Buffer*>(bufCom.ptr());
                      if (buf != nullptr) {
                        const auto mapped = buf->GetMappedSlice();
                        const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                        const size_t bufLen = buf->Desc()->ByteWidth;
                        if (base != nullptr && bufLen >= 208) {
                          std::string floatDump;
                          for (uint32_t i = 0; i < 52; ++i) { // 52 floats = 208 bytes
                            float f = 0.f;
                            std::memcpy(&f, base + i * 4, 4);
                            if (i > 0) floatDump += " ";
                            floatDump += str::format(f);
                          }
                          Logger::info(str::format(
                            "[BspVsModelInst0] VS=0x", std::hex, vsH_dump, std::dec,
                            " bufLen=", bufLen,
                            " floats=[", floatDump, "]"));
                        } else {
                          Logger::info(str::format(
                            "[BspVsModelInst0] VS=0x", std::hex, vsH_dump, std::dec,
                            " bufLen=", bufLen,
                            " mapPtr=", base != nullptr ? "non-null" : "NULL"));
                        }
                      }
                    }
                  }
                }
              }

              // NV-DXVK: dump the IA input layout — every attribute the VS
              // expects, plus the bound vertex buffer for each input slot.
              // If we see 2+ TEXCOORDs from different slots, Remix's
              // texcoordBuffer might be reading the wrong UV stream.
              if (auto* il = m_context->m_state.ia.inputLayout.ptr()) {
                const auto& sems = il->GetRtxSemantics();
                for (const auto& s : sems) {
                  uint64_t vbHandle = 0;
                  uint64_t vbLen = 0;
                  uint32_t vbStride = 0;
                  if (s.inputSlot < D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) {
                    const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                    if (vb.buffer != nullptr) {
                      vbHandle = reinterpret_cast<uint64_t>(vb.buffer.ptr());
                      vbLen = vb.buffer->Desc()->ByteWidth;
                    }
                    vbStride = vb.stride;
                  }
                  Logger::info(str::format(
                    "[BspVsIA] VS=0x", std::hex, vsH_dump, std::dec,
                    " sem=", s.name, s.index,
                    " slot=", s.inputSlot,
                    " off=", s.byteOffset,
                    " fmt=", uint32_t(s.format),
                    " perInst=", s.perInstance ? 1 : 0,
                    " vbHandle=0x", std::hex, vbHandle, std::dec,
                    " vbStride=", vbStride,
                    " vbLen=", vbLen));
                }

                // NV-DXVK: dump actual TEXCOORD values from the bound VB for
                // the first 6 vertices. BSP world VSes (1953b6e9, e7abcf4e)
                // declare TWO TEXCOORD channels — TEXCOORD0 (R32G32_FLOAT @
                // off=20) and TEXCOORD1 (R16G16_FLOAT @ off=28). Probe 3 of
                // the GpuPrint cycle showed Remix interp UV at huge magnitudes
                // (50, -777) producing a per-pixel gradient of 1.47 → mip ~9
                // → flat 1×1 sample. If TEXCOORD1 holds the small-range tile
                // UV (typical for the albedo) and TEXCOORD0 is the planar
                // lightmap UV, switching Remix's BLAS texcoord plumbing to
                // TEXCOORD1 should fix it. Print both so we can confirm
                // before changing the plumbing.
                auto hfToFloat = [](uint16_t h) -> float {
                  const uint32_t sign = (h >> 15) & 1u;
                  const int32_t  exp  = (h >> 10) & 0x1f;
                  const uint32_t mant = h & 0x3ffu;
                  uint32_t out;
                  if (exp == 0) {
                    if (mant == 0) {
                      out = sign << 31;
                    } else {
                      // denormal — slow path
                      const float m = float(mant) / 1024.0f;
                      const float v = (sign ? -1.0f : 1.0f) * std::ldexp(m, -14);
                      float r = v;
                      return r;
                    }
                  } else if (exp == 31) {
                    out = (sign << 31) | 0x7f800000u | (mant << 13);
                  } else {
                    out = (sign << 31) | (uint32_t(exp + 112) << 23) | (mant << 13);
                  }
                  float r;
                  std::memcpy(&r, &out, sizeof(r));
                  return r;
                };
                for (const auto& s : sems) {
                  if (std::strncmp(s.name, "TEXCOORD", 8) != 0) continue;
                  if (s.inputSlot >= D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) continue;
                  const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                  if (vb.buffer == nullptr || vb.stride == 0) continue;
                  auto* d3dBuf = vb.buffer.ptr();
                  if (d3dBuf == nullptr) continue;
                  const auto mapped = d3dBuf->GetMappedSlice();
                  const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  if (base == nullptr) continue;
                  const size_t bufLen = d3dBuf->Desc()->ByteWidth;
                  const uint32_t fmt = uint32_t(s.format);
                  // 101=R32G32_FLOAT (8B), 103=R32G32B32_FLOAT (12B but UV-as-vec3 unusual),
                  //  81=R16G16_FLOAT (4B). Skip anything else.
                  const uint32_t elemBytes = (fmt == 81u) ? 4u
                                          : (fmt == 101u) ? 8u
                                          : (fmt == 103u) ? 12u
                                          : 0u;
                  if (elemBytes == 0) continue;
                  std::string vals;
                  std::string rawHex;
                  std::string decoded;
                  for (uint32_t v = 0; v < 6u; ++v) {
                    const size_t off = size_t(v) * vb.stride + s.byteOffset + vb.offset;
                    if (off + elemBytes > bufLen) break;
                    float u = 0.f, vv = 0.f;
                    uint32_t uBits = 0, vBits = 0;
                    if (fmt == 81u) {
                      uint16_t hu = 0, hv = 0;
                      std::memcpy(&hu, base + off + 0, 2);
                      std::memcpy(&hv, base + off + 2, 2);
                      u = hfToFloat(hu);
                      vv = hfToFloat(hv);
                      uBits = hu; vBits = hv;
                    } else {
                      std::memcpy(&u,  base + off + 0, 4);
                      std::memcpy(&vv, base + off + 4, 4);
                      std::memcpy(&uBits, base + off + 0, 4);
                      std::memcpy(&vBits, base + off + 4, 4);
                    }
                    if (!vals.empty())    { vals += " "; rawHex += " "; decoded += " "; }
                    vals    += "(" + str::format(u) + "," + str::format(vv) + ")";
                    {
                      // Hex dump (raw bits, what the VS sees if it reads uint)
                      char buf[64];
                      std::snprintf(buf, sizeof(buf), "(0x%08X,0x%08X)", uBits, vBits);
                      rawHex += buf;
                    }
                    {
                      // Apply the wall-VS decode formula:
                      //   tile_uv = float(int(uint(bits) >> 3) + 0xF0000000) * (1/16384)
                      // If the source data is normal float UVs and the VS treats
                      // them as uint, we'll see the resulting amplification. If
                      // the source data is already uint-encoded (~2.2B values),
                      // decoded will land in the ~1000-magnitude range observed
                      // by the runtime probe — that's the smoking gun.
                      const int32_t uDec = int32_t(uBits >> 3) + int32_t(0xF0000000);
                      const int32_t vDec = int32_t(vBits >> 3) + int32_t(0xF0000000);
                      const float uOut = float(uDec) * (1.0f / 16384.0f);
                      const float vOut = float(vDec) * (1.0f / 16384.0f);
                      decoded += "(" + str::format(uOut) + "," + str::format(vOut) + ")";
                    }
                  }
                  Logger::info(str::format(
                    "[BspVsTexCoords] VS=0x", std::hex, vsH_dump, std::dec,
                    " sem=", s.name, s.index,
                    " fmt=", fmt,
                    " stride=", vb.stride,
                    " off=", s.byteOffset,
                    " vbOff=", vb.offset,
                    " vbLen=", bufLen,
                    " floats=[", vals, "]"
                    " hex=[", rawHex, "]"
                    " vsDecoded=[", decoded, "]"));
                }
              }
            }
          }
        }
      }
    }

    // NV-DXVK: ground-truth raw VB UV decode for the uint-packed BSP VS family.
    //
    // The doc reports VS=e7abcf4e applies the in-shader decode formula
    //   tile_uv    = (int(uint >> 3) + 0xF0000000) * (1/16384)
    //   lightmap   = uint(ushort) * (1/65535)
    // PIX disassembly of the wall draw confirmed this. The question this
    // logging answers: are the UVs Remix's BVH ends up sampling (visible
    // in slang probes 5/6 as ~-7872 magnitudes) the SAME values native's
    // VS would produce, or has Remix's interleaver corrupted them?
    //
    // For each BSP draw with a uint-format TEXCOORD0 (fmt = R32G32_UINT
    // = VkFormat 101), read first 6 verts of the active VB, replay the
    // decode in C++, and log raw uints + decoded floats side-by-side.
    // Match against probe 5/6 readings (which dump the post-interleaver
    // surface buffer) — if decoded values agree, interleaver is fine and
    // the geometry truly is what we think. If they disagree, Remix is
    // either reading the wrong VB offset or skipping the decode.
    //
    // Throttle key: (VS hash, VB handle) — gives us coverage across
    // multiple VBs of the same VS without spamming on every draw.
    {
      if (const auto* vsShader = m_context->m_state.vs.shader.ptr()) {
        if (const auto* vsCs = vsShader->GetCommonShader()) {
          XXH64_hash_t vsH_decode = 0, psH_decode = 0;
          GetCurrentVsPsHashes(vsH_decode, psH_decode);
          // Same allowlist of BSP-class VSes the existing logging uses.
          const bool isBspVsDecode =
               vsH_decode == 0x7c38fdf4358d5527ull
            || vsH_decode == 0x0990ac503e694beeull
            || vsH_decode == 0x1953b6e9cc252e4eull
            || vsH_decode == 0xe7abcf4ea24b0fa7ull
            || vsH_decode == 0x448e372f6d5e78e1ull;
          if (isBspVsDecode) {
            if (auto* il = m_context->m_state.ia.inputLayout.ptr()) {
              const auto& sems = il->GetRtxSemantics();
              // NV-DXVK: dump POSITION0 too. The interleaver decodes
              // R32G32_UINT positions via a hardcoded 21|21|22-bit layout
              // with offsets -1024,-1024,-2048 (calibrated for the viewmodel
              // VS). If wall VSes use a different layout / different offsets,
              // every wall vertex's world position is wrong, screen-space
              // projection is wrong, and the gradient pipeline produces
              // wrong slivers — even though the math itself is correct.
              for (const auto& s : sems) {
                if (std::strncmp(s.name, "POSITION", 8) != 0) continue;
                if (s.inputSlot >= D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) continue;
                const uint32_t fmt = uint32_t(s.format);
                if (fmt != 101u) continue;  // R32G32_UINT only — float positions are fine

                const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                if (vb.buffer == nullptr || vb.stride == 0) continue;
                auto* d3dBuf = vb.buffer.ptr();
                if (d3dBuf == nullptr) continue;

                // Throttle by (VS hash, VB handle).
                const uint64_t vbHandle = reinterpret_cast<uint64_t>(d3dBuf);
                const uint64_t throttleKey =
                    uint64_t(vsH_decode) ^ (vbHandle << 1) ^ (uint64_t(s.index) << 33) ^ 0xDEADBEEFull;
                static std::unordered_set<uint64_t> sPosDecodeKeys;
                static std::mutex sPosDecodeMu;
                static std::atomic<uint32_t> sPosDecodeCount{0};
                bool firstSeen = false;
                {
                  std::lock_guard<std::mutex> lk(sPosDecodeMu);
                  if (sPosDecodeCount.load() < 30
                      && sPosDecodeKeys.insert(throttleKey).second) {
                    firstSeen = true;
                    ++sPosDecodeCount;
                  }
                }
                if (!firstSeen) continue;

                const auto& imm = d3dBuf->GetImmutableData();
                const uint8_t* base = nullptr;
                size_t bufLen = 0;
                const char* readSrc = "none";
                if (!imm.empty()) {
                  base = imm.data();
                  bufLen = imm.size();
                  readSrc = "IMMUTABLE";
                } else {
                  const auto mapped = d3dBuf->GetMappedSlice();
                  if (mapped.mapPtr) {
                    base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    bufLen = mapped.length;
                    readSrc = "MAPPED";
                  } else {
                    void* p = d3dBuf->GetBuffer()->mapPtr(0);
                    if (p) {
                      base = reinterpret_cast<const uint8_t*>(p);
                      bufLen = d3dBuf->GetBuffer()->info().size;
                      readSrc = "GETBUF";
                    }
                  }
                }
                if (base == nullptr) {
                  Logger::info(str::format(
                    "[POSdecode] VS=0x", std::hex, vsH_decode,
                    " VB=0x", vbHandle, std::dec,
                    " sem=", s.name, s.index,
                    " fmt=", fmt,
                    " NO-READ"));
                  continue;
                }

                std::string dump;
                for (uint32_t v = 0; v < 6u; ++v) {
                  const size_t off = vb.offset + s.byteOffset + size_t(v) * vb.stride;
                  if (off + 8u > bufLen) {
                    dump += " [OVERRUN]";
                    break;
                  }
                  uint32_t u0 = 0, u1 = 0;
                  std::memcpy(&u0, base + off + 0, 4);
                  std::memcpy(&u1, base + off + 4, 4);
                  // Replicate interleaver formula (21|21|22 bit layout).
                  const uint32_t xi = u0 & 0x001FFFFFu;
                  const uint32_t yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u);
                  const uint32_t zi = u1 >> 10u;
                  const float kScale = 1.0f / 1024.0f;
                  // Offset variant A: viewmodel calibration (-1024, -1024, -2048)
                  const float xA = float(xi) * kScale - 1024.0f;
                  const float yA = float(yi) * kScale - 1024.0f;
                  const float zA = float(zi) * kScale - 2048.0f;
                  // Offset variant B: alternate calibration (-1024, -1024, -2080)
                  const float zB = float(zi) * kScale - 2080.0f;
                  if (!dump.empty()) dump += " ";
                  dump += str::format(
                      "v", v,
                      ":raw=(", u0, ",", u1, ")",
                      " A=(", xA, ",", yA, ",", zA, ")",
                      " zB=", zB);
                }
                Logger::info(str::format(
                  "[POSdecode] VS=0x", std::hex, vsH_decode,
                  " VB=0x", vbHandle, std::dec,
                  " sem=", s.name, s.index,
                  " fmt=", fmt,
                  " stride=", vb.stride,
                  " semOff=", s.byteOffset,
                  " vbOff=", vb.offset,
                  " bufLen=", bufLen,
                  " src=", readSrc,
                  " ", dump));
              }

              for (const auto& s : sems) {
                if (std::strncmp(s.name, "TEXCOORD", 8) != 0) continue;
                if (s.inputSlot >= D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) continue;
                const uint32_t fmt = uint32_t(s.format);
                // Handled formats:
                //   101 = R32G32_UINT     → e7abcf4e tile-UV decode
                //                           (raw_uint >> 3) + 0xF0000000) * 1/16384
                //    81 = R16G16_UINT     → lightmap-UV decode (raw * 1/65535)
                //   103 = R32G32_SFLOAT   → pass-through (no decode); used by
                //                           VS=7c38fdf4 wall family. Logging
                //                           the raw floats here lets us
                //                           cross-check probe 5/6 — if the
                //                           VB contains the huge ~8000-range
                //                           values we see in the surface
                //                           buffer, Remix isn't inflating
                //                           anything; that's the genuine VB
                //                           content.
                const bool isUintTexcoord  = (fmt == 101u) || (fmt == 81u);
                const bool isFloatTexcoord = (fmt == 103u);
                if (!isUintTexcoord && !isFloatTexcoord) continue;

                const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                if (vb.buffer == nullptr || vb.stride == 0) continue;
                auto* d3dBuf = vb.buffer.ptr();
                if (d3dBuf == nullptr) continue;

                // (VS, VB-handle, semantic-index) throttle so we get one
                // sample per unique VS+VB pair rather than once per VS.
                const uint64_t vbHandle = reinterpret_cast<uint64_t>(d3dBuf);
                const uint64_t throttleKey =
                    uint64_t(vsH_decode) ^ (vbHandle << 1) ^ (uint64_t(s.index) << 33);
                static std::unordered_set<uint64_t> sUvDecodeKeys;
                static std::mutex sUvDecodeMu;
                static std::atomic<uint32_t> sUvDecodeCount{0};
                bool firstSeen = false;
                {
                  std::lock_guard<std::mutex> lk(sUvDecodeMu);
                  if (sUvDecodeCount.load() < 60
                      && sUvDecodeKeys.insert(throttleKey).second) {
                    firstSeen = true;
                    ++sUvDecodeCount;
                  }
                }
                if (!firstSeen) continue;

                const auto& imm = d3dBuf->GetImmutableData();
                const uint8_t* base = nullptr;
                size_t bufLen = 0;
                const char* readSrc = "none";
                if (!imm.empty()) {
                  base = imm.data();
                  bufLen = imm.size();
                  readSrc = "IMMUTABLE";
                } else {
                  const auto mapped = d3dBuf->GetMappedSlice();
                  if (mapped.mapPtr) {
                    base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    bufLen = mapped.length;
                    readSrc = "MAPPED";
                  } else {
                    void* p = d3dBuf->GetBuffer()->mapPtr(0);
                    if (p) {
                      base = reinterpret_cast<const uint8_t*>(p);
                      bufLen = d3dBuf->GetBuffer()->info().size;
                      readSrc = "GETBUF";
                    }
                  }
                }
                if (base == nullptr) {
                  Logger::info(str::format(
                    "[UVdecode] VS=0x", std::hex, vsH_decode,
                    " VB=0x", vbHandle, std::dec,
                    " sem=", s.name, s.index,
                    " fmt=", fmt,
                    " NO-READ (buffer not CPU-readable)"));
                  continue;
                }

                const uint32_t elemBytes = (fmt == 81u) ? 4u : 8u;
                std::string dump;
                for (uint32_t v = 0; v < 6u; ++v) {
                  const size_t off = vb.offset + s.byteOffset + size_t(v) * vb.stride;
                  if (off + elemBytes > bufLen) {
                    dump += " [OVERRUN]";
                    break;
                  }
                  if (!dump.empty()) dump += " ";
                  if (fmt == 101u) {
                    uint32_t rawU = 0, rawV = 0;
                    std::memcpy(&rawU, base + off + 0, 4);
                    std::memcpy(&rawV, base + off + 4, 4);
                    const int32_t shiftedU = static_cast<int32_t>(rawU >> 3);
                    const int32_t shiftedV = static_cast<int32_t>(rawV >> 3);
                    const int32_t biasedU = shiftedU + INT32_C(-268435456);
                    const int32_t biasedV = shiftedV + INT32_C(-268435456);
                    const float decodedU = float(biasedU) * (1.0f / 16384.0f);
                    const float decodedV = float(biasedV) * (1.0f / 16384.0f);
                    dump += "v" + str::format(v)
                         + ":raw=(" + str::format(rawU) + "," + str::format(rawV) + ")"
                         + " dec=(" + str::format(decodedU) + "," + str::format(decodedV) + ")";
                  } else if (fmt == 81u) {
                    // R16G16_UINT — lightmap stream. Decode = ushort * 1/65535.
                    uint16_t hu = 0, hv = 0;
                    std::memcpy(&hu, base + off + 0, 2);
                    std::memcpy(&hv, base + off + 2, 2);
                    const float decodedU = float(uint32_t(hu)) * (1.0f / 65535.0f);
                    const float decodedV = float(uint32_t(hv)) * (1.0f / 65535.0f);
                    dump += "v" + str::format(v)
                         + ":raw=(" + str::format(hu) + "," + str::format(hv) + ")"
                         + " dec=(" + str::format(decodedU) + "," + str::format(decodedV) + ")";
                  } else if (fmt == 103u) {
                    // R32G32_SFLOAT — float pass-through (VS=7c38fdf4 family).
                    // No decode; raw is the value the VS forwards to the PS.
                    float fU = 0.f, fV = 0.f;
                    std::memcpy(&fU, base + off + 0, 4);
                    std::memcpy(&fV, base + off + 4, 4);
                    dump += "v" + str::format(v)
                         + ":raw=(" + str::format(fU) + "," + str::format(fV) + ")";
                  }
                }
                Logger::info(str::format(
                  "[UVdecode] VS=0x", std::hex, vsH_decode,
                  " VB=0x", vbHandle, std::dec,
                  " sem=", s.name, s.index,
                  " fmt=", fmt,
                  " stride=", vb.stride,
                  " semOff=", s.byteOffset,
                  " vbOff=", vb.offset,
                  " bufLen=", bufLen,
                  " src=", readSrc,
                  " ", dump));
              }
            }
          }
        }
      }
    }

    {
      if (const auto* psShader = m_context->m_state.ps.shader.ptr()) {
        if (const auto* cs = psShader->GetCommonShader()) {
          auto rsx = cs->FindCBField("CBufUberStatic", "c_uv1RotScaleX");
          auto rsy = cs->FindCBField("CBufUberStatic", "c_uv1RotScaleY");
          auto tr  = cs->FindCBField("CBufUberStatic", "c_uv1Translate");
          const bool rsxUsed = cs->ReadsCBField("CBufUberStatic", "c_uv1RotScaleX");
          const bool rsyUsed = cs->ReadsCBField("CBufUberStatic", "c_uv1RotScaleY");
          const bool trUsed  = cs->ReadsCBField("CBufUberStatic", "c_uv1Translate");
          // NV-DXVK DIAGNOSTIC KILL-SWITCH: flip to true to unconditionally
          // disable the UV-transform extraction regardless of D3DReflect's
          // reported used-flags. If hand/weapon rendering returns to normal
          // when this is enabled, the gate is leaking (D3DReflect fallback
          // path is treating unused fields as used, or reflection failed
          // silently and the default is "used=true"). If hand still breaks,
          // the regression is NOT from UV-transform poisoning.
          constexpr bool kKillSwitch_DisableUvTransform = false;
          // NV-DXVK: VS hash allowlist (Option A). D3DReflect's D3D_SVF_USED is
          // "referenced in any reachable code path", not "definitively consumed
          // per-draw" — TF2 PSes guard the UV transform behind a material flag
          // that's false for weapons/characters/most props, so the cbuffer
          // carries stale BSP values and we poison those draws. Restrict the
          // transform to VS hashes we've verified belong to BSP world-prop /
          // alpha-cutout paths.
          XXH64_hash_t vsH_allow = 0, psH_allow = 0;
          GetCurrentVsPsHashes(vsH_allow, psH_allow);
          const bool vsIsBspAllowed =
               vsH_allow == 0x7c38fdf4358d5527ull
            || vsH_allow == 0x0990ac503e694beeull
            || vsH_allow == 0x1953b6e9cc252e4eull
            || vsH_allow == 0xe7abcf4ea24b0fa7ull
            || vsH_allow == 0x448e372f6d5e78e1ull;
          const bool psReadsUvFields = !kKillSwitch_DisableUvTransform
            && vsIsBspAllowed
            && rsx && rsy && tr && rsxUsed && rsyUsed && trUsed;
          // NV-DXVK: per-unique-PS gate decision log. If the gate passes
          // (applied=1) for a PS you know is character/weapon, D3DReflect
          // is wrongly marking the UV-transform fields as used for that
          // shader. Cross-reference with [Reflect] usedFields=N/M for
          // raw D3DReflect numbers.
          {
            static std::unordered_set<uint64_t> sGateLoggedPs;
            static std::mutex sGateLoggedMu;
            XXH64_hash_t vsH_g = 0, psH_g = 0;
            GetCurrentVsPsHashes(vsH_g, psH_g);
            bool firstTime = false;
            {
              std::lock_guard<std::mutex> lk(sGateLoggedMu);
              firstTime = sGateLoggedPs.insert(psH_g).second;
            }
            if (firstTime && psH_g != 0) {
              Logger::info(str::format(
                "[UVx-gate] PS=0x", std::hex, psH_g, std::dec,
                " declared(rsx,rsy,tr)=(", rsx ? 1 : 0, ",", rsy ? 1 : 0, ",", tr ? 1 : 0,
                ") used=(", rsxUsed ? 1 : 0, ",", rsyUsed ? 1 : 0, ",", trUsed ? 1 : 0,
                ") applied=", psReadsUvFields ? 1 : 0));
            }
          }

          // NV-DXVK: log the gate decision once per distinct PS hash so we can
          // see exactly which shaders are being poisoned vs correctly skipped.
          // Throttled to a bounded set — if character still looks wrong after
          // the D3DReflect gate switch, search this log for its PS hash and
          // confirm applied=0 (gate skipped it).
          {
            static std::unordered_set<uint64_t> sSeenPs;
            static std::mutex sSeenMu;
            // Use the XXH64 hash that matches every other [D3D11Rtx]
            // FillMaterialData log line (GetCurrentVsPsHashes), so this
            // can be cross-referenced against e.g. PS=0x7a6e4c5725a53e07.
            XXH64_hash_t vsH_uv = 0, psH_uv = 0;
            GetCurrentVsPsHashes(vsH_uv, psH_uv);
            bool firstSight = false;
            {
              std::lock_guard<std::mutex> g(sSeenMu);
              if (sSeenPs.size() < 256 && sSeenPs.insert(uint64_t(psH_uv)).second)
                firstSight = true;
            }
            if (firstSight) {
              Logger::info(str::format(
                "[UVgate] PS=0x", std::hex, psH_uv, std::dec,
                " declares{rsx=", (rsx ? 1 : 0),
                " rsy=", (rsy ? 1 : 0),
                " tr=",  (tr  ? 1 : 0), "}",
                " used{rsx=", (rsxUsed ? 1 : 0),
                " rsy=",     (rsyUsed ? 1 : 0),
                " tr=",      (trUsed  ? 1 : 0), "}",
                " applied=", (psReadsUvFields ? 1 : 0)));
            }
          }

          // NV-DXVK: per-PS dump of EVERY field in CBufUberStatic and
          // CBufUberDynamic — fires even when the UV-transform gate skips
          // (applied=0) so we can see what fields the gate-failing PSes
          // actually carry. Goal: find any small-magnitude scale field
          // (~0.001 .. 0.5) that the wall PS reads instead of the
          // c_uv1RotScale* fields the gate looks for. If native shows
          // crisp brick where Remix shows flat tan, the divergence is
          // either (a) Remix not applying a scale that the PS does, or
          // (b) the scale lives in a different field name we never
          // consulted. This dump covers both: every field, every value,
          // every used-flag.
          //
          // Format:
          //   [PsCBfields] PS=<hash> cb=<name>@<slot> field=<name>
          //     off=<bytes> sz=<bytes> used=<0|1>
          //     val=(f0, f1, f2, f3) | scale-candidate
          //
          // "scale-candidate" tag fires for f0 if abs(f0) is in
          // (1e-4, 0.5) — a strong heuristic for "this is a per-material
          // UV multiplier." We tag the field for fast grep.
          {
            static std::unordered_set<uint64_t> sCbDumpedPs;
            static std::mutex sCbDumpedMu;
            XXH64_hash_t vsH_cb = 0, psH_cb = 0;
            GetCurrentVsPsHashes(vsH_cb, psH_cb);
            bool firstCbDump = false;
            {
              std::lock_guard<std::mutex> lk(sCbDumpedMu);
              if (sCbDumpedPs.size() < 128
                  && sCbDumpedPs.insert(uint64_t(psH_cb)).second)
                firstCbDump = true;
            }
            if (firstCbDump && psH_cb != 0) {
              const char* kCbsToDump[] = {
                "CBufUberStatic",
                "CBufUberDynamic",
              };
              for (const char* cbName : kCbsToDump) {
                const auto* cbInfo = cs->FindCBuffer(cbName);
                if (!cbInfo) continue;
                const uint32_t cbSlot = cbInfo->bindSlot;
                if (cbSlot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
                  continue;
                const auto& cbBinding = m_context->m_state.ps.constantBuffers[cbSlot];
                const uint8_t* base = nullptr;
                size_t bufLen = 0;
                if (cbBinding.buffer != nullptr) {
                  const auto mapped = cbBinding.buffer->GetMappedSlice();
                  base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  bufLen = cbBinding.buffer->Desc()->ByteWidth;
                }
                // cb.constantOffset is in 16-byte units (D3D11 binding).
                const size_t cbBaseOff = size_t(cbBinding.constantOffset) * 16;
                // Sort fields by offset for legible output.
                std::vector<std::pair<std::string, D3D11CbufferField>> sortedFields;
                sortedFields.reserve(cbInfo->fields.size());
                for (const auto& kv : cbInfo->fields) sortedFields.push_back(kv);
                std::sort(sortedFields.begin(), sortedFields.end(),
                  [](const auto& a, const auto& b) {
                    return a.second.offset < b.second.offset;
                  });
                for (const auto& kv : sortedFields) {
                  const std::string& fieldName = kv.first;
                  const D3D11CbufferField& f = kv.second;
                  // Read up to 4 floats so we cover float / float2 / float3 /
                  // float4 fields uniformly. Larger fields (matrices) would
                  // need their own pass — but UV scales are always small.
                  const uint32_t nFloats = std::min<uint32_t>(4u, f.size / 4u);
                  float vals[4] = { 0.f, 0.f, 0.f, 0.f };
                  bool readOk = false;
                  if (base != nullptr && nFloats > 0
                      && cbBaseOff + f.offset + nFloats * 4 <= bufLen) {
                    std::memcpy(vals, base + cbBaseOff + f.offset, nFloats * 4);
                    readOk = true;
                  }
                  // Heuristic: a per-material UV scale is a non-zero,
                  // non-unit, finite float of moderate magnitude. We tag
                  // anything in (1e-4, 0.5) since lightmap/world-to-tile
                  // conversions in Source typically land in that range
                  // (1/64 = 0.0156, 1/256 = 0.0039, 1/1024 = 0.001 etc).
                  bool scaleCandidate = false;
                  if (readOk) {
                    for (uint32_t i = 0; i < nFloats; ++i) {
                      const float v = vals[i];
                      const float a = std::fabs(v);
                      if (std::isfinite(v) && a > 1e-4f && a < 0.5f
                          && a != 0.0f) {
                        scaleCandidate = true;
                        break;
                      }
                    }
                  }
                  Logger::info(str::format(
                    "[PsCBfields] PS=0x", std::hex, psH_cb, std::dec,
                    " cb=", cbName, "@", cbSlot,
                    " field=", fieldName,
                    " off=", f.offset,
                    " sz=", f.size,
                    " used=", (f.used ? 1 : 0),
                    " readOk=", (readOk ? 1 : 0),
                    " val=(", vals[0], ",", vals[1], ",", vals[2], ",", vals[3], ")",
                    scaleCandidate ? " | scale-candidate" : ""));
                }
                // Trailer line so we can tell "no fields enumerated" from
                // "shader has no fields" at a glance.
                Logger::info(str::format(
                  "[PsCBfields] PS=0x", std::hex, psH_cb, std::dec,
                  " cb=", cbName, "@", cbSlot,
                  " END fieldCount=", cbInfo->fields.size(),
                  " bufLen=", bufLen,
                  " cbBaseOff=", cbBaseOff));
              }
            }
          }

          if (psReadsUvFields && rsx->slot == rsy->slot && rsx->slot == tr->slot
              && rsx->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
            const auto& cb = m_context->m_state.ps.constantBuffers[rsx->slot];
            if (cb.buffer != nullptr) {
              const auto mapped = cb.buffer->GetMappedSlice();
              const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              const size_t bufLen = cb.buffer->Desc()->ByteWidth;
              // cb.constantOffset is in 16-byte units (D3D11 cbuffer binding granularity).
              const size_t cbBase = size_t(cb.constantOffset) * 16;
              if (base
                  && cbBase + rsx->offset + 8 <= bufLen
                  && cbBase + rsy->offset + 8 <= bufLen
                  && cbBase + tr->offset  + 8 <= bufLen) {
                float rsxV[2] = {}, rsyV[2] = {}, trV[2] = {};
                std::memcpy(rsxV, base + cbBase + rsx->offset, 8);
                std::memcpy(rsyV, base + cbBase + rsy->offset, 8);
                std::memcpy(trV,  base + cbBase + tr->offset,  8);

                // NV-DXVK: unconditional cb0 dump — fires once per PS
                // regardless of identity / degenerate gate, so we can tell
                // "game wrote identity" from "we read zeros" from "offsets
                // are wrong". Also dumps the first 64 bytes of the cb raw so
                // we can eyeball whether our offsets line up with what the
                // reflection layout says.
                {
                  static std::unordered_set<uint64_t> sDumpedPs;
                  static std::mutex sDumpedMu;
                  XXH64_hash_t vsH_d = 0, psH_d = 0;
                  GetCurrentVsPsHashes(vsH_d, psH_d);
                  bool first = false;
                  {
                    std::lock_guard<std::mutex> lk(sDumpedMu);
                    first = sDumpedPs.insert(uint64_t(psH_d)).second;
                  }
                  if (first) {
                    std::string rawHex;
                    const size_t dumpBytes = std::min<size_t>(64, bufLen - cbBase);
                    for (size_t i = 0; i < dumpBytes; i += 4) {
                      float f = 0.f;
                      if (cbBase + i + 4 <= bufLen) {
                        std::memcpy(&f, base + cbBase + i, 4);
                      }
                      rawHex += str::format(f, i + 4 < dumpBytes ? " " : "");
                    }
                    Logger::info(str::format(
                      "[D3D11Rtx.UVdump2] VS=0x", std::hex, vsH_d,
                      " PS=0x", psH_d, std::dec,
                      " cbBase=", cbBase,
                      " rsx@", rsx->offset, "=(", rsxV[0], ",", rsxV[1], ")",
                      " rsy@", rsy->offset, "=(", rsyV[0], ",", rsyV[1], ")",
                      " tr@",  tr->offset,  "=(", trV[0],  ",", trV[1],  ")",
                      " rawFloats: ", rawHex));
                  }
                }

                // Only install a non-identity AND non-degenerate transform.
                // Skip identity (no-op). Skip zero matrix / NaN / infinite
                // values — those usually mean the PS leaves the field
                // uninitialised (FXC marks them [unused]) and the app never
                // writes sensible defaults.
                const bool isIdentity =
                  rsxV[0] == 1.0f && rsxV[1] == 0.0f &&
                  rsyV[0] == 0.0f && rsyV[1] == 1.0f &&
                  trV[0]  == 0.0f && trV[1]  == 0.0f;
                auto isFinite = [](float f) {
                  return std::isfinite(f);
                };
                const bool allFinite =
                  isFinite(rsxV[0]) && isFinite(rsxV[1]) &&
                  isFinite(rsyV[0]) && isFinite(rsyV[1]) &&
                  isFinite(trV[0])  && isFinite(trV[1]);
                const float det = rsxV[0] * rsyV[1] - rsxV[1] * rsyV[0];
                const bool hasArea = std::fabs(det) > 1e-12f;
                if (!isIdentity && allFinite && hasArea) {
                  // Column-major Matrix4(col0, col1, col2, col3). Applied as
                  // result = M * (u, v, 0, 1) gives
                  //   result.x = rsxV[0]*u + rsxV[1]*v + trV[0]
                  //   result.y = rsyV[0]*u + rsyV[1]*v + trV[1]
                  Matrix4 uvXform(
                    Vector4(rsxV[0], rsyV[0], 0.0f, 0.0f),   // col 0 (u contrib)
                    Vector4(rsxV[1], rsyV[1], 0.0f, 0.0f),   // col 1 (v contrib)
                    Vector4(0.0f,    0.0f,    1.0f, 0.0f),   // col 2
                    Vector4(trV[0],  trV[1],  0.0f, 1.0f));  // col 3 (translate)
                  dcs.transformData.textureTransform = uvXform;

                  // NV-DXVK: log once per unique PS hash the installed
                  // cbuffer values, so we can tell which PSes are being
                  // fed BSP-style scales vs identity vs weapon-specific
                  // atlas transforms.
                  static std::unordered_set<uint64_t> sUvInstalledPs;
                  static std::mutex sUvInstalledMu;
                  XXH64_hash_t vsH_inst = 0, psH_inst = 0;
                  GetCurrentVsPsHashes(vsH_inst, psH_inst);
                  bool firstForThisPs = false;
                  {
                    std::lock_guard<std::mutex> lk(sUvInstalledMu);
                    firstForThisPs = sUvInstalledPs.insert(psH_inst).second;
                  }
                  if (firstForThisPs) {
                    Logger::info(str::format(
                      "[D3D11Rtx.UVx] installed for PS=0x", std::hex, psH_inst, std::dec,
                      ": RSX=(", rsxV[0], ",", rsxV[1], ") "
                      "RSY=(", rsyV[0], ",", rsyV[1], ") "
                      "T=(",   trV[0],  ",", trV[1],  ")"));
                  }
                }
              }
            }
          }
        }
      }
    }

    // NV-DXVK: raw-UV range diagnostic for BSP draws. The hands-vs-walls
    // mip/aliasing asymmetry suggests BSP mesh UVs are world-unit (large),
    // while character UVs are normalised. Log the actual per-vertex UV range
    // so we stop speculating — compute min/max over the drawn range, dump
    // a few sample (u,v) values, and pair with the albedo texture size so
    // we can eyeball the UV:texture ratio and compute the true per-pixel
    // delta that the raytracer is being handed.
    //
    // Gated on: gameplay (raw>50), VS ∈ BSP allowlist, once per unique PS
    // hash, and only when the texcoord buffer is R32G32_SFLOAT (the common
    // Source/Respawn encoding). Bounded sample count (64 verts) so a large
    // BSP draw doesn't stall the pipeline.
    {
      static std::unordered_set<uint64_t> sUvRangeLoggedPs;
      static std::mutex sUvRangeLoggedMu;
      XXH64_hash_t vsH_diag = 0, psH_diag = 0;
      GetCurrentVsPsHashes(vsH_diag, psH_diag);
      const bool vsIsBspAllowed_diag =
           vsH_diag == 0x7c38fdf4358d5527ull
        || vsH_diag == 0x0990ac503e694beeull
        || vsH_diag == 0x1953b6e9cc252e4eull
        || vsH_diag == 0xe7abcf4ea24b0fa7ull
        || vsH_diag == 0x448e372f6d5e78e1ull;
      const bool gameplayReady_diag = (m_rawDrawCount > 50);
      if (vsIsBspAllowed_diag && psH_diag != 0) {
        bool firstForThisPs = false;
        {
          std::lock_guard<std::mutex> lk(sUvRangeLoggedMu);
          if (sUvRangeLoggedPs.size() < 64 && sUvRangeLoggedPs.insert(uint64_t(psH_diag)).second)
            firstForThisPs = true;
        }
        if (firstForThisPs) {
          const VkFormat tcFmt = geo.texcoordBuffer.defined()
            ? geo.texcoordBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
          const uint32_t tcStride = geo.texcoordBuffer.defined()
            ? geo.texcoordBuffer.stride() : 0;
          const size_t tcLen = geo.texcoordBuffer.defined()
            ? geo.texcoordBuffer.length() : 0;
          const bool tcDefined = geo.texcoordBuffer.defined();
          const bool gameplayReady = gameplayReady_diag;

          // Decode a handful of UV samples when we recognise the format.
          float uMin =  1e30f, uMax = -1e30f;
          float vMin =  1e30f, vMax = -1e30f;
          std::string samples;
          uint32_t decoded = 0;
          if (tcDefined && tcStride >= 4) {
            const uint8_t* tcBase = reinterpret_cast<const uint8_t*>(
              geo.texcoordBuffer.mapPtr(geo.texcoordBuffer.offsetFromSlice()));
            if (tcBase) {
              const uint32_t vCount = uint32_t(tcLen / (tcStride > 0 ? tcStride : 1));
              const uint32_t sampleN = std::min<uint32_t>(64, vCount);
              auto acceptUV = [&](uint32_t i, float u, float v) {
                uMin = std::min(uMin, u); uMax = std::max(uMax, u);
                vMin = std::min(vMin, v); vMax = std::max(vMax, v);
                if (i < 6) samples += str::format("(", u, ",", v, ") ");
                ++decoded;
              };
              for (uint32_t i = 0; i < sampleN; ++i) {
                const uint8_t* p = tcBase + size_t(i) * tcStride;
                if (tcFmt == VK_FORMAT_R32G32_SFLOAT && tcStride >= 8) {
                  float uv[2] = {};
                  std::memcpy(uv, p, 8);
                  acceptUV(i, uv[0], uv[1]);
                } else if (tcFmt == VK_FORMAT_R16G16_SFLOAT && tcStride >= 4) {
                  uint16_t h[2] = {};
                  std::memcpy(h, p, 4);
                  auto halfToFloat = [](uint16_t x) -> float {
                    uint32_t s = (x & 0x8000) << 16;
                    uint32_t e = (x & 0x7C00) >> 10;
                    uint32_t m = (x & 0x03FF);
                    uint32_t bits;
                    if (e == 0) {
                      if (m == 0) { bits = s; }
                      else {
                        e = 1;
                        while (!(m & 0x0400)) { m <<= 1; --e; }
                        m &= 0x03FF;
                        bits = s | ((e + 112) << 23) | (m << 13);
                      }
                    } else if (e == 31) {
                      bits = s | 0x7F800000 | (m << 13);
                    } else {
                      bits = s | ((e + 112) << 23) | (m << 13);
                    }
                    float f; std::memcpy(&f, &bits, 4); return f;
                  };
                  acceptUV(i, halfToFloat(h[0]), halfToFloat(h[1]));
                } else if (tcFmt == VK_FORMAT_R16G16_SNORM && tcStride >= 4) {
                  int16_t s[2] = {};
                  std::memcpy(s, p, 4);
                  acceptUV(i, std::max(-1.f, s[0] / 32767.f), std::max(-1.f, s[1] / 32767.f));
                } else if (tcFmt == VK_FORMAT_R16G16_UNORM && tcStride >= 4) {
                  uint16_t s[2] = {};
                  std::memcpy(s, p, 4);
                  acceptUV(i, s[0] / 65535.f, s[1] / 65535.f);
                } else {
                  // Unknown; emit the first two u32 words of the stride as raw hex
                  if (i < 3 && tcStride >= 8) {
                    uint32_t w[2] = {};
                    std::memcpy(w, p, 8);
                    samples += str::format("[raw ", std::hex, w[0], ",", w[1], std::dec, "] ");
                  }
                }
              }
            }
          }

          uint32_t texW = 0, texH = 0;
          if (dcs.materialData.colorTextures[0].isValid()
              && !dcs.materialData.colorTextures[0].isImageEmpty()) {
            auto view = dcs.materialData.colorTextures[0].getImageView();
            if (view != nullptr) {
              const auto& ii = view->image()->info();
              texW = ii.extent.width;
              texH = ii.extent.height;
            }
          }
          Logger::info(str::format(
            "[D3D11Rtx.UVrange] VS=0x", std::hex, vsH_diag,
            " PS=0x", psH_diag, std::dec,
            " gameplay=", gameplayReady ? 1 : 0,
            " tcDefined=", tcDefined ? 1 : 0,
            " fmt=", uint32_t(tcFmt),
            " stride=", tcStride,
            " decoded=", decoded,
            " u=[", uMin, ",", uMax, "] v=[", vMin, ",", vMax, "]",
            " du=", (uMax - uMin), " dv=", (vMax - vMin),
            " tex=", texW, "x", texH,
            " texels_per_uv_u=", (uMax > uMin ? float(texW) / (uMax - uMin) : 0.0f),
            " samples: ", samples));
        }
      }
    }

    // NV-DXVK start: Per-vertex skinning — capture bone matrices from VS SRV t30
    if (geo.numBonesPerVertex > 0) {
      bool gotBones = false;
      const uint32_t kBoneSrvSlot = 30;
      ID3D11ShaderResourceView* boneSrv = nullptr;
      if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
        boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();

      if (boneSrv) {
        Com<ID3D11Resource> boneRes;
        boneSrv->GetResource(&boneRes);
        auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());

        if (boneBuf) {
          // Try multiple paths to access bone data (buffer may be GPU-only)
          const uint8_t* bonePtr = nullptr;
          size_t boneBufLen = 0;

          // Path 1: mapped slice (WRITE_DISCARD mapped memory)
          {
            const auto mapped = boneBuf->GetMappedSlice();
            if (mapped.mapPtr && mapped.length >= 48) {
              bonePtr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              boneBufLen = mapped.length;
            }
          }

          // Path 2: DxvkBuffer direct mapPtr (host-visible buffers)
          if (!bonePtr) {
            DxvkBufferSlice boneSlice = boneBuf->GetBufferSlice();
            if (boneSlice.defined()) {
              void* p = boneSlice.buffer()->mapPtr(0);
              if (p) {
                bonePtr = reinterpret_cast<const uint8_t*>(p) + boneSlice.offset();
                boneBufLen = boneSlice.length();
              }
            }
          }

          // Path 3: cached from UpdateSubresource interception.
          // NV-DXVK TF2: merge the DXVK-level bone cache mirror (which
          // captures VkCmdCopyBuffer staging→t30 uploads TF2 uses for bulk
          // rig uploads — 61-bone or 148-bone matrices spanning upper-half
          // palette slots that UpdateSubresource alone never sees) into
          // m_fullBoneCache before use. Without this merge, upper-half
          // slots are zero in our cache, causing NPCs to render in A-pose
          // because their vertices weight to those "invalid" bones.
          {
            std::lock_guard<std::mutex> lk(::dxvk::tf2::g_boneCacheMirrorMutex);
            if (::dxvk::tf2::g_boneCacheMirrorPopulated
                && ::dxvk::tf2::g_boneCacheMirror.size() == 393216) {
              if (m_fullBoneCache.size() != 393216) {
                m_fullBoneCache.resize(393216, 0);
              }
              // Mirror takes precedence where non-zero; lower-half slots
              // from UpdateSubresource remain authoritative since the
              // game keeps writing them each frame.
              const uint8_t* mirror = ::dxvk::tf2::g_boneCacheMirror.data();
              uint8_t* dst = m_fullBoneCache.data();
              // Merge: copy any mirror byte that is non-zero over our
              // cache (preserves UpdateSubresource slots that mirror may
              // not touch, and fills upper-half slots that only mirror has).
              for (size_t i = 0; i < 393216; i += 48) {
                // Work in 48-byte bone chunks. If the mirror bone is
                // non-zero (any byte != 0), copy the whole bone matrix.
                bool mirrorNonZero = false;
                for (size_t b = 0; b < 48; ++b) {
                  if (mirror[i + b] != 0) { mirrorNonZero = true; break; }
                }
                if (mirrorNonZero) {
                  std::memcpy(dst + i, mirror + i, 48);
                }
              }
              m_hasFullBoneCache = true;
            }
          }
          if (!bonePtr && m_hasFullBoneCache && !m_fullBoneCache.empty()) {
            bonePtr = m_fullBoneCache.data();
            boneBufLen = m_fullBoneCache.size();
          }

          if (bonePtr && boneBufLen >= 48) {
            const uint32_t numBones = static_cast<uint32_t>(boneBufLen / 48);
            const uint32_t maxBones = std::min(numBones, 256u); // SkinningArgs limit

            dcs.skinningData.numBonesPerVertex = geo.numBonesPerVertex;
            dcs.skinningData.numBones = maxBones;
            dcs.skinningData.minBoneIndex = 0;
            dcs.skinningData.pBoneMatrices.resize(maxBones);

            bool allValid = true;
            for (uint32_t b = 0; b < maxBones; ++b) {
              const float* m = reinterpret_cast<const float*>(bonePtr + b * 48);
              // Validate bone data isn't garbage
              if (!std::isfinite(m[0]) || !std::isfinite(m[3])) {
                allValid = false;
                break;
              }
              // float3x4 row-major → Matrix4
              dcs.skinningData.pBoneMatrices[b] = Matrix4(
                Vector4(m[0], m[1], m[2],  0.0f),
                Vector4(m[4], m[5], m[6],  0.0f),
                Vector4(m[8], m[9], m[10], 0.0f),
                Vector4(m[3], m[7], m[11], 1.0f));
            }

            if (allValid) {
              dcs.skinningData.computeHash();
              gotBones = true;
            }
          }
        }
      }

      // If we couldn't read bone matrices, disable skinning for this draw
      // to prevent dispatchSkinning from running with empty bone data.
      if (!gotBones) {
        static uint32_t sBoneFailLog = 0;
        if (sBoneFailLog < 5) {
          ++sBoneFailLog;
          Logger::warn(str::format(
            "[D3D11Rtx] Per-vertex skinning: could not read bone matrices from t30.",
            " SRV=", boneSrv ? "bound" : "null",
            " bonesPerVert=", geo.numBonesPerVertex,
            " bwFmt=", bwSem ? uint32_t(bwSem->format) : 0,
            " biFmt=", biSem ? uint32_t(biSem->format) : 0));
        }
        geo.numBonesPerVertex = 0;
        geo.blendWeightBuffer  = RasterBuffer();
        geo.blendIndicesBuffer = RasterBuffer();
        dcs.geometryData = geo;
      } else {
        static uint32_t sBoneOkLog = 0;
        if (sBoneOkLog < 3) {
          ++sBoneOkLog;
          Logger::info(str::format(
            "[D3D11Rtx] Per-vertex skinning: captured ", dcs.skinningData.numBones,
            " bones (", geo.numBonesPerVertex, " per vertex)",
            " bwFmt=", uint32_t(bwSem->format),
            " biFmt=", uint32_t(biSem->format)));
        }
      }
    }
    // NV-DXVK end

    DrawParameters params;
    params.instanceCount = 1;
    params.vertexCount   = indexed ? 0 : count;
    params.indexCount    = indexed ? count : 0;
    params.firstIndex    = indexed ? start : 0;
    params.vertexOffset  = indexed ? static_cast<uint32_t>(std::max(base, 0)) : start;

    // NV-DXVK DEBUG: Log draw parameters for fmt=101 draws
    if (posSem->format == VK_FORMAT_R32G32_UINT) {
      static uint32_t sDrawParamLog = 0;
      if (sDrawParamLog < 20) {
        ++sDrawParamLog;
        Logger::info(str::format(
          "[D3D11Rtx] DrawParams: indexed=", indexed ? 1 : 0,
          " count=", count, " start=", start, " base=", base,
          " vertCount=", geo.vertexCount,
          " idxCount=", params.indexCount,
          " firstIdx=", params.firstIndex,
          " vtxOff=", params.vertexOffset,
          " stride=", posBuffer.stride(),
          " idxFmt=", indexed ? uint32_t(m_context->m_state.ia.indexBuffer.format) : 0,
          " idxOff=", indexed ? m_context->m_state.ia.indexBuffer.offset : 0));
      }
    }

    // === PER-DRAW TRANSFORM + VERTEX DIAGNOSTIC ===
    // Log every draw for the first 5 in-game frames (m_drawCallID-based gate).
    {
      static uint32_t s_submitLogFrame = 0;
      static uint32_t s_submitPrevID   = UINT32_MAX;
      if (dcs.drawCallID == 0 || dcs.drawCallID < s_submitPrevID)
        ++s_submitLogFrame;
      s_submitPrevID = dcs.drawCallID;

      // NV-DXVK: log during first gameplay frames (after boot-time menus).
      // Tracked via global "gameplay started" latch in EndFrame.
      if (s_GameplayLogFrames > 0) {
        const auto& T = dcs.transformData;
        const bool o2wIdentity = isIdentityExact(T.objectToWorld);
        // VS hash
        std::string vsHash = "?";
        auto vsShaderCom = m_context->m_state.vs.shader;
        if (vsShaderCom != nullptr && vsShaderCom->GetCommonShader() != nullptr) {
          auto& s = vsShaderCom->GetCommonShader()->GetShader();
          if (s != nullptr) vsHash = s->getShaderKey().toString();
        }
        Logger::info(str::format(
          "[D3D11Rtx] Submit drawID=", dcs.drawCallID,
          " frame=", s_submitLogFrame,
          " VS=", vsHash,
          " verts=", geo.vertexCount,
          " o2w:", o2wIdentity ? "IDENTITY" : "nonId",
          " T=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
          " o2vT=(", T.objectToView[3][0], ",", T.objectToView[3][1], ",", T.objectToView[3][2], ")",
          " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")"));

        // Sample first vertex position from the position buffer.
        const auto& posBuf = geo.positionBuffer;
        if (posBuf.defined()) {
          const float* p = reinterpret_cast<const float*>(
            posBuf.mapPtr(posBuf.offsetFromSlice()));
          if (p) {
            Logger::info(str::format(
              "[D3D11Rtx]   pos[0]=(", p[0], ",", p[1], ",", p[2], ")",
              " stride=", posBuf.stride(),
              " fmt=", static_cast<uint32_t>(posBuf.vertexFormat())));
          }
        }
      }
    }

    // (stale transform filter removed — worldToView now set by cross-frame VP)

    // NV-DXVK [debob-timeline]: per-frame snapshot of the actual w2v
    // translation that ends up at SubmitDraw. Throttled to first draw
    // per VS per frame so we get a clean time-series suitable for
    // detecting oscillation magnitude visible to the user.
    //
    // Implementation: hash (VS, frame) — emit one log per (VS, frame).
    // We use m_drawCallID as a per-frame-ish counter; combined with
    // a frame-tag derived from m_currentFanoutFrame or similar, we
    // get one entry per frame per VS. Simpler: just throttle by
    // globally counting every Nth frame approximated by raw count.
    {
      const auto& T = dcs.transformData;
      const std::string vsKey = m_currentVsHashCache.substr(0, std::min<size_t>(m_currentVsHashCache.size(), 19u));
      // Track per-VS prev w2v Z to detect change (not just every frame).
      static std::unordered_map<std::string, float> sPrevTz;
      auto it = sPrevTz.find(vsKey);
      const float curTz = T.worldToView[3][2];
      const float prevTz = (it != sPrevTz.end()) ? it->second : 0.f;
      const float delta = curTz - prevTz;
      // Log if Tz changed by >0.05 (real motion) AND we've seen this VS before.
      // This shows the actual oscillation magnitude as it happens, not the
      // throttled "first 3 per VS" we had.
      if (it != sPrevTz.end() && std::abs(delta) > 0.05f) {
        static uint32_t sTimelineLog = 0;
        ++sTimelineLog;
        if (sTimelineLog < 400) {
          Logger::info(str::format(
            "[debobTimeline] vs=", vsKey,
            " w2vPath=", m_lastWtvPathId,
            " o2wPath=", m_lastO2wPathId,
            " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", curTz, ")",
            " ΔTz=", delta));
        }
      }
      sPrevTz[vsKey] = curTz;
    }

    // NV-DXVK: Log every submitted draw with key info for TDR diagnosis.
    // Logger flushes to disk so the last entry before a TDR is visible.
    {
      const auto& T = dcs.transformData;
      const auto& G = dcs.geometryData;

      // NV-DXVK: Log the VS hash of non-instanced bone draws (these work
      // correctly — their shader tells us the right cbuffer layout Remix reads).
      static uint32_t sLoggedNonInstBone = 0;
      const bool isBoneInst = (m_boneInstanceCount > 0 && m_currentInstancesToObject);
      if (sLoggedNonInstBone < 5 && G.boneMatrixBuffer.defined() && !isBoneInst
          && G.positionBuffer.vertexFormat() == VK_FORMAT_R32G32_UINT
          && std::abs(T.objectToWorld[3][0]) > 100.f) {  // real world translation
        ++sLoggedNonInstBone;
        const auto& o2w = T.objectToWorld;
        Logger::info(str::format(
          "[D3D11Rtx] Non-inst bone o2w:"
          " col0=(", o2w[0][0], ",", o2w[0][1], ",", o2w[0][2], ")",
          " col1=(", o2w[1][0], ",", o2w[1][1], ",", o2w[1][2], ")",
          " col2=(", o2w[2][0], ",", o2w[2][1], ",", o2w[2][2], ")",
          " T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
      }

      // NV-DXVK: bump per-frame histogram using the path tag set by
      // whichever site most recently wrote to transforms.objectToWorld.
      {
        const uint32_t pid = (m_lastO2wPathId < 16) ? m_lastO2wPathId : 15;
        ++m_o2wPathCounts[pid];
        if (!m_currentVsHashCache.empty()) {
          const std::string vsKey = m_currentVsHashCache.substr(0, 19);
          auto& arr = m_vsO2wPathCounts[vsKey];
          ++arr[pid];
        }
      }
      // NV-DXVK: throttle — was firing per captured draw (~100-150/frame
      // at gameplay, ~15x more during shader-compilation-heavy loading),
      // contributing to the per-present log storm that stalled loading.
      // One line per unique VS is enough to verify which shaders route
      // through which o2wPath; per-draw variation within a VS is rare.
      {
        static std::unordered_set<std::string> sCommitLog;
        const std::string vsKey = m_currentVsHashCache.substr(0, 19);
        if (sCommitLog.insert(vsKey).second) {
          Logger::info(str::format(
            "[D3D11Rtx] COMMIT vs=", vsKey,
            " id=", dcs.drawCallID,
            " verts=", G.vertexCount,
            " fmt=", uint32_t(G.positionBuffer.vertexFormat()),
            " stride=", G.positionBuffer.stride(),
            " bone=", G.boneMatrixBuffer.defined() ? 1 : 0,
            " inst=", G.boneInstanceIndex,
            " o2wPath=", m_lastO2wPathId,
            " w2vPath=", m_lastWtvPathId,
            " boneTfd=", m_currentDrawIsBoneTransformed ? 1 : 0,
            " o2wT=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
            " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")",
            " o2vT=(", T.objectToView[3][0], ",", T.objectToView[3][1], ",", T.objectToView[3][2], ")",
            " raw=", m_rawDrawCount));
        }
      }
      // NV-DXVK [bone-w2v-trace]: for ALL bone=1 draws, also dump the FULL
      // worldToView rotation (not just translation) so we can see what
      // basis is being applied to the gun. Throttled per-VS to first 3.
      if (G.boneMatrixBuffer.defined()) {
        static std::unordered_map<std::string, uint32_t> sBoneW2vLog;
        const std::string vsKey = m_currentVsHashCache.substr(0, 19);
        auto& cnt = sBoneW2vLog[vsKey];
        if (cnt < 3) {
          ++cnt;
          const auto& W = T.worldToView;
          Logger::info(str::format(
            "[boneW2vTrace] vs=", vsKey,
            " w2vPath=", m_lastWtvPathId,
            " boneTfd=", m_currentDrawIsBoneTransformed ? 1 : 0,
            " row0=(", W[0][0], ",", W[0][1], ",", W[0][2], ",", W[0][3], ")",
            " row1=(", W[1][0], ",", W[1][1], ",", W[1][2], ",", W[1][3], ")",
            " row2=(", W[2][0], ",", W[2][1], ",", W[2][2], ",", W[2][3], ")",
            " row3=(", W[3][0], ",", W[3][1], ",", W[3][2], ",", W[3][3], ")"));
        }
      }
      // NV-DXVK: once-per-VS throttle — was firing per committed draw
      // (~150/sec during loading = biggest remaining log-storm source).
      {
        static std::unordered_set<std::string> sO2wRotLog;
        const std::string vsKey = m_currentVsHashCache.substr(0, 19);
        if (sO2wRotLog.insert(vsKey).second) {
          const auto& M = T.objectToWorld;
          const bool identRot =
            std::abs(M[0][0] - 1.f) < 1e-4f && std::abs(M[1][1] - 1.f) < 1e-4f && std::abs(M[2][2] - 1.f) < 1e-4f &&
            std::abs(M[0][1]) < 1e-4f && std::abs(M[0][2]) < 1e-4f &&
            std::abs(M[1][0]) < 1e-4f && std::abs(M[1][2]) < 1e-4f &&
            std::abs(M[2][0]) < 1e-4f && std::abs(M[2][1]) < 1e-4f;
          Logger::info(str::format(
            "[D3D11Rtx.o2wRot] vs=", vsKey,
            " id=", dcs.drawCallID,
            " identityRot=", identRot ? 1 : 0,
            " col0=(", M[0][0], ",", M[0][1], ",", M[0][2], ")",
            " col1=(", M[1][0], ",", M[1][1], ",", M[1][2], ")",
            " col2=(", M[2][0], ",", M[2][1], ",", M[2][2], ")"));
        }
      }
    }

    // NV-DXVK [SkyAutoCb2]: detect sky from the bound VS's cb2.c_cameraOrigin
    // and stamp InstanceCategories::Sky on dcs before the commit.
    // RtxContext::tryHandleSky reads dcs.cameraType (set later in
    // rtx_camera_manager from this category) — under SkyMode::PhysicalAtmosphere
    // the sky geometry submission is dropped and Hillaire's atmospheric LUTs
    // (rtx_atmosphere.cpp) render the sky in its place.
    // NV-DXVK [restore-excellent-state]: unconditional call, matching the
    // build state at the "excellent you pick the right camera but the
    // game stays on one frame" moment. The A/B kill-switch gating on
    // rtx.skyAutoDetect is removed per user request to reproduce that
    // exact condition.
    SetSkyCategoryFromCb2(dcs);

    m_context->EmitCs([params, dcs](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, dcs);
    });
  }

  // NV-DXVK [SkyAutoCb2]: cb2-driven sky categorization, with cross-frame
  // sky-origin LATCHING.
  //
  // Source-engine games (TF2) reuse the same VS shaders for both 3D-skybox
  // draws (sky_camera entity, distinct world origin) and main-pass draws
  // — no static bytecode signal. The runtime signal is c_cameraOrigin in
  // CBufCommonPerCamera (cb2 byte 4): sky_camera binds a different origin
  // than the main camera.
  //
  // Algorithm:
  //   1. Read c_cameraOrigin via RDEF.
  //   2. If we already classified an origin as sky earlier THIS frame and
  //      this draw matches it (within skyAutoDetectUniqueCameraDistance) →
  //      sky.
  //   3. Else if a sky origin is LATCHED from a previous frame and this
  //      draw matches it → sky; promote it to this frame's sky origin.
  //   4. Else (no match yet): bootstrap path. Only on a frame where we've
  //      never observed any origin AND there is no latched sky origin
  //      AND last frame disambiguated (saw ≥2 unique origins), we tag
  //      this first-of-frame origin as sky. Otherwise → not sky.
  //
  // Why latching matters: the previous "first observed origin = sky" rule
  // wrongly marked the WHOLE frame as sky whenever sky_camera didn't run
  // (sky occluded by ceiling/walls), because then main's origin was the
  // first observed. Latching pins the sky to the known sky_camera position
  // so non-sky frames have zero false positives — Hillaire only replaces
  // the sky when sky_camera actually contributes draws this frame.
  //
  // Per-frame state resets in EndFrame; m_skyOriginLatched persists.
  bool D3D11Rtx::SetSkyCategoryFromCb2(DrawCallState& dcs) {
    // Read c_cameraOrigin from the bound VS's cb2 by RDEF-resolved address.
    const auto vsPtr = m_context->m_state.vs.shader;
    if (vsPtr == nullptr || vsPtr->GetCommonShader() == nullptr) {
      return false;
    }
    const auto* common = vsPtr->GetCommonShader();
    auto camLoc = common->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
    if (!camLoc || camLoc->size < 12
        || camLoc->slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
      return false;
    }
    const auto& vsCbs = m_context->m_state.vs.constantBuffers;
    const auto& camCb = vsCbs[camLoc->slot];
    if (camCb.buffer == nullptr) {
      return false;
    }
    const auto map = camCb.buffer->GetMappedSlice();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(map.mapPtr);
    if (p == nullptr) return false;
    const size_t base = static_cast<size_t>(camCb.constantOffset) * 16 + camLoc->offset;
    if (base + 12 > camCb.buffer->Desc()->ByteWidth) return false;
    const float* fp = reinterpret_cast<const float*>(p + base);
    if (!std::isfinite(fp[0]) || !std::isfinite(fp[1]) || !std::isfinite(fp[2])) {
      return false;
    }
    const Vector3 origin{ fp[0], fp[1], fp[2] };

    const float threshold = RtxOptions::skyAutoDetectUniqueCameraDistance();
    const float thrSq = threshold * threshold;
    // The main-camera safety threshold uses a more generous radius
    // (skyAutoDetectUniqueCameraDistance can be 1.0 unit which is tighter
    // than gameplay sub-frame jitter). 8 units = ~20cm in TF2's hammer
    // unit scale, well below sky_camera→main separation but above any
    // realistic frame-to-frame main-camera drift inside one fanout cycle.
    const float mainGuardSq = 8.0f * 8.0f;
    auto closeTo = [&](const Vector3& a, const Vector3& b, float t) {
      const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
      return (dx*dx + dy*dy + dz*dz) < t;
    };

    // SAFETY: never tag a draw as sky if its c_cameraOrigin matches the
    // KNOWN main-camera position. m_lastFanoutCamOrigin is the
    // authoritative gameplay camera origin published by the BSP fanout
    // path (search this file for [fanoutCBRead]). Even if the latch is
    // wrong (bootstrap misclassified some non-sky origin), this guard
    // prevents the catastrophic "entire main pass tagged as sky" case
    // that produced ~560 sky draws/frame and locked the Main camera at
    // the player's feet (no Main-class draws survived to feed the latch).
    if (m_hasFanoutCamOrigin
        && closeTo(origin, m_lastFanoutCamOrigin, mainGuardSq)) {
      // Track for diagnostics, then bail out.
      bool alreadySeen = false;
      for (const Vector3& seen : m_skySeenOriginsThisFrame) {
        if (closeTo(seen, origin, thrSq)) { alreadySeen = true; break; }
      }
      if (!alreadySeen) {
        m_skySeenOriginsThisFrame.push_back(origin);
        std::string vsKey;
        if (common != nullptr) {
          const auto& sh = common->GetShader();
          if (sh != nullptr) vsKey = sh->getShaderKey().toString().substr(0, 19);
        }
        float vpW = 0.f, vpH = 0.f, vpMaxD = 0.f;
        if (m_context->m_state.rs.numViewports > 0) {
          vpW = m_context->m_state.rs.viewports[0].Width;
          vpH = m_context->m_state.rs.viewports[0].Height;
          vpMaxD = m_context->m_state.rs.viewports[0].MaxDepth;
        }
        Logger::info(str::format(
          "[SkyAutoCb2.classify] frame=", m_context->m_device->getCurrentFrameId(),
          " seenIdx=", m_skySeenOriginsThisFrame.size() - 1,
          " origin=(", origin.x, ",", origin.y, ",", origin.z, ")",
          " vs=", vsKey,
          " vp=", vpW, "x", vpH,
          " maxD=", vpMaxD,
          " fanout=(", m_lastFanoutCamOrigin.x, ",",
                       m_lastFanoutCamOrigin.y, ",",
                       m_lastFanoutCamOrigin.z, ")",
          " verdict=mainGuard"));
      }
      return false;
    }

    bool isSky = false;
    const char* reason = "none";

    // 1. Already-classified-this-frame match.
    if (m_skyOriginThisFrame && closeTo(*m_skyOriginThisFrame, origin, thrSq)) {
      isSky = true;
      reason = "thisFrame";
    }
    // 2. Cross-frame latched match — promote to this frame's sky.
    else if (m_skyOriginLatched && closeTo(*m_skyOriginLatched, origin, thrSq)) {
      m_skyOriginThisFrame = origin;
      isSky = true;
      reason = "latched";
    }
    // 3. Bootstrap — accept any non-main origin THIS frame whose value
    //    also appeared in the PREVIOUS frame. The 2-frame stability
    //    check rejects transient one-offs (single weird shadow-pass or
    //    initialisation draws) which previously latched bootstrap onto
    //    the wrong origin. The origin must also be distinct from the
    //    known main camera (fanout) — enforced by the early-return
    //    safety gate at the top of this function.
    //
    //    Stable sky_camera positions in Source-engine games appear in
    //    every single frame, so 2-frame stability is a very cheap and
    //    reliable filter. We only bootstrap once per session — once
    //    m_skyOriginLatched is set, branch 2 (latched match) handles
    //    everything thereafter.
    else if (!m_skyOriginLatched && m_hasFanoutCamOrigin) {
      // NV-DXVK [restore-excellent-state]: bootstrap as it was at the
      // "excellent you pick the right camera but the game stays on one
      // frame" moment in conversation — stability check only, no
      // magnitude floor, no viewport-cubemap-face gate, no
      // fanout-moved gate.
      //
      // The earlier (multi-condition) bootstrap that filtered on
      // magnitude / viewport-cubemap-face / fanout-moved was added to
      // prevent (0,0,0) and shadow-cascade misclassifications. With the
      // simpler single-stability-check version, those misclassifications
      // CAN happen — (0,0,0) screen-space writers tend to win bootstrap
      // because they appear stably every frame — and the resulting
      // SkipSubmit of composite/tonemap freezes the screen. That frozen
      // screen happened to show a frame whose camera the user reported
      // as correct, which they want to reproduce. Reapplying the exact
      // gate logic from that build per request.
      //
      // Safety gate (mainGuard) at the top of this function still
      // prevents tagging anything matching m_lastFanoutCamOrigin, so
      // the main pass survives. Latching (0,0,0) only kills
      // screen-space passes (composite, tonemap, etc.).
      const float kStableSq = RtxOptions::skyAutoDetectUniqueCameraDistance() *
                              RtxOptions::skyAutoDetectUniqueCameraDistance();

      bool stableAcrossFrames = false;
      for (const Vector3& prev : m_skySeenOriginsLastFrame) {
        if (closeTo(prev, origin, kStableSq)) { stableAcrossFrames = true; break; }
      }

      if (stableAcrossFrames) {
        m_skyOriginThisFrame = origin;
        isSky = true;
        reason = "bootstrap";
      }
    }

    // Track distinct origins for diagnostics + the bootstrap predicate.
    bool alreadySeen = false;
    for (const Vector3& seen : m_skySeenOriginsThisFrame) {
      if (closeTo(seen, origin, thrSq)) { alreadySeen = true; break; }
    }
    if (!alreadySeen) {
      m_skySeenOriginsThisFrame.push_back(origin);
      // [SkyAutoCb2.classify] Log every NEW distinct origin observed this
      // frame with its classification verdict. Once-per-(frame, origin)
      // so output stays bounded even at high draw counts. The 4-tuple
      // {origin, fanoutCam, latch, verdict} is what we need to debug
      // bootstrap misclassification (bootstrap latching onto main, or
      // fanout publishing sky_camera as main).
      const Vector3 fc = m_hasFanoutCamOrigin
        ? m_lastFanoutCamOrigin : Vector3{ 0.f, 0.f, 0.f };
      const Vector3 lat = m_skyOriginLatched.value_or(Vector3{ 0.f, 0.f, 0.f });
      // Compute fanout-moved-since-last-frame for the log line so we can
      // see why bootstrap is/isn't accepting a candidate.
      float fanoutDelta = 0.f;
      if (m_skyPrevFrameHadFanoutCam && m_hasFanoutCamOrigin) {
        const float dxF = m_lastFanoutCamOrigin.x - m_skyPrevFrameFanoutCam.x;
        const float dyF = m_lastFanoutCamOrigin.y - m_skyPrevFrameFanoutCam.y;
        const float dzF = m_lastFanoutCamOrigin.z - m_skyPrevFrameFanoutCam.z;
        fanoutDelta = std::sqrt(dxF*dxF + dyF*dyF + dzF*dzF);
      }
      float vpW = 0.f, vpH = 0.f, vpMaxD = 0.f;
      if (m_context->m_state.rs.numViewports > 0) {
        vpW = m_context->m_state.rs.viewports[0].Width;
        vpH = m_context->m_state.rs.viewports[0].Height;
        vpMaxD = m_context->m_state.rs.viewports[0].MaxDepth;
      }
      // VS hash for origin→shader correlation. Local string built off the
      // already-resolved vsPtr/common pointers — no static state, no
      // mutex, so safe to call from per-draw context (mainCamSurvey's
      // static unordered_set+mutex was crashing — keep per-frame state
      // only on the per-D3D11Rtx-instance vector).
      std::string vsKey;
      if (vsPtr != nullptr && common != nullptr) {
        const auto& sh = common->GetShader();
        if (sh != nullptr) vsKey = sh->getShaderKey().toString().substr(0, 19);
      }
      Logger::info(str::format(
        "[SkyAutoCb2.classify] frame=", m_context->m_device->getCurrentFrameId(),
        " seenIdx=", m_skySeenOriginsThisFrame.size() - 1,
        " origin=(", origin.x, ",", origin.y, ",", origin.z, ")",
        " vs=", vsKey,
        " vp=", vpW, "x", vpH,
        " maxD=", vpMaxD,
        " fanoutKnown=", m_hasFanoutCamOrigin ? 1 : 0,
        " fanout=(", fc.x, ",", fc.y, ",", fc.z, ")",
        " fanoutDelta=", fanoutDelta,
        " latched=", m_skyOriginLatched.has_value() ? 1 : 0,
        " latch=(", lat.x, ",", lat.y, ",", lat.z, ")",
        " verdict=", reason));
    }

    if (isSky) {
      dcs.setCategory(InstanceCategories::Sky, true);
      ++m_skyDetectedThisFrame;
    }
    return isSky;
  }

  void D3D11Rtx::OnUpdateSubresource(ID3D11Resource* pDstResource, const void* pSrcData, UINT SrcDataSize, UINT DstOffset, UINT BufSize) {
    if (!pSrcData) return;
    // NV-DXVK [diag] catch the engine's source buffer pointer for c_cameraOrigin
    // so we can HW write BP it and find the function that writes the bobbed eye.
    // Heuristic: c_cameraOrigin is at byte 4 of cb2 (per fanoutFpAddr base=4).
    // So pSrcData+4 = (cam.x, cam.y, cam.z). Match cam.x ≈ -5179, cam.y ≈ 279.
    {
      if (SrcDataSize >= 16 + DstOffset) {
        const float* p = reinterpret_cast<const float*>(
          reinterpret_cast<const uint8_t*>(pSrcData) + DstOffset);
        // p[0] = cb byte 0..3 (some other field), p[1..3] = c_cameraOrigin.xyz
        const float cx = p[1];
        const float cy = p[2];
        const float cz = p[3];
        if (std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cz)
            && std::abs(cx - (-5179.0f)) < 2.0f
            && std::abs(cy - (279.0f)) < 2.0f) {
          static uintptr_t sLastSrcZ = 0;
          static uint32_t sLogged = 0;
          const uintptr_t srcZ = reinterpret_cast<uintptr_t>(&p[3]);
          if (srcZ != sLastSrcZ && sLogged < 40) {
            sLastSrcZ = srcZ;
            ++sLogged;
            Logger::info(str::format(
              "[updSubCamSrc] dst=", (void*)pDstResource,
              " bufSize=", BufSize,
              " srcDataSize=", SrcDataSize,
              " dstOff=", DstOffset,
              " srcXYZ=", reinterpret_cast<void*>(srcZ - 8),
              " srcZ=", reinterpret_cast<void*>(srcZ),
              " cam=(", cx, ",", cy, ",", cz, ")"));
          }
        }
      }
    }
    // NV-DXVK TF2: cache the t30 bone-matrix buffer (393216 bytes =
    // 8192 bones × 48). Written by BOTH paths:
    //   • here (UpdateSubresource): fills lower 8 slots of each 16-bone
    //     palette with per-frame animated bones.
    //   • dxvk::tf2::g_boneCacheMirror (CopyBuffer, see dxvk_context.cpp):
    //     fills upper slots and bulk-rig writes (61+ bones at once).
    // The two are merged lazily in EndFrame + the skinning capture path.
    if (BufSize == 393216 && SrcDataSize >= 48) {
      if (m_fullBoneCache.size() != BufSize)
        m_fullBoneCache.resize(BufSize, 0);
      const size_t maxCopy = std::min(
        static_cast<size_t>(SrcDataSize),
        static_cast<size_t>(BufSize) - static_cast<size_t>(DstOffset));
      std::memcpy(m_fullBoneCache.data() + DstOffset, pSrcData, maxCopy);
      m_hasFullBoneCache = true;
    }
    // Log ALL buffer update sizes to find cb3
    static uint32_t sAllUpdateLog = 0;
    if (sAllUpdateLog < 50) {
      // Only log unique sizes
      static std::set<uint32_t> seenSizes;
      if (seenSizes.find(BufSize) == seenSizes.end()) {
        seenSizes.insert(BufSize);
        ++sAllUpdateLog;
        const float* fData = reinterpret_cast<const float*>(
          reinterpret_cast<const uint8_t*>(pSrcData) + DstOffset);
        Logger::info(str::format(
          "[D3D11Rtx] UpdateSub: bufSize=", BufSize,
          " off=", DstOffset, " len=", SrcDataSize,
          " f0=(", (SrcDataSize >= 16 ? fData[0] : 0), ",",
          (SrcDataSize >= 16 ? fData[1] : 0), ",",
          (SrcDataSize >= 16 ? fData[2] : 0), ",",
          (SrcDataSize >= 16 ? fData[3] : 0), ")"));
      }
    }
    // NV-DXVK: log EVERY 393216-byte (t30 bone) update with ALL bone Tx
    // values to see if the game uploads DIFFERENT matrices per slot or
    // duplicates the same matrix. If Tx values are all identical, bones
    // 0-7 in an upload are the same → my earlier skin.bone dump was
    // correct in showing them identical, and the character's pose comes
    // from some other mechanism.
    if (BufSize == 393216) {
      // Per-frame throttle — answer "are slots 8-15 of any 16-bone palette
      // ever written by UpdateSubresource?" by logging EVERY upload for a
      // single frame of gameplay. Also aggregates stats into
      // [BoneUploadFrame] at frame boundaries.
      const uint32_t fid = m_context->m_device->getCurrentFrameId();
      static uint32_t sLastFrameBU = 0;
      static uint32_t sCountThisFrameBU = 0;
      static uint32_t sStatTotal = 0;
      static uint32_t sStatBytes = 0;
      static uint32_t sStatOffZeroMod768 = 0; // off % 768 == 0 (palette-aligned)
      static uint32_t sStatOff384Mod768 = 0;  // off % 768 == 384 (upper half!)
      static uint32_t sStatOffOther = 0;      // other residues
      static uint32_t sStatLen384 = 0;        // len == 384 (8 bones)
      static uint32_t sStatLen768 = 0;        // len == 768 (16 bones)
      static uint32_t sStatLenOther = 0;
      static uint32_t sStatMinOff = UINT32_MAX;
      static uint32_t sStatMaxOff = 0;
      if (fid != sLastFrameBU) {
        // Dump previous frame's aggregate before resetting.
        if (sStatTotal > 0) {
          Logger::info(str::format(
            "[BoneUploadFrame] f=", sLastFrameBU,
            " uploads=", sStatTotal,
            " bytes=", sStatBytes,
            " minOff=", sStatMinOff, " maxOff=", sStatMaxOff,
            " off%768=0:", sStatOffZeroMod768,
            " off%768=384:", sStatOff384Mod768,
            " offOther:", sStatOffOther,
            " len=384:", sStatLen384,
            " len=768:", sStatLen768,
            " lenOther:", sStatLenOther));
        }
        sLastFrameBU = fid;
        sCountThisFrameBU = 0;
        sStatTotal = sStatBytes = 0;
        sStatOffZeroMod768 = sStatOff384Mod768 = sStatOffOther = 0;
        sStatLen384 = sStatLen768 = sStatLenOther = 0;
        sStatMinOff = UINT32_MAX; sStatMaxOff = 0;
      }
      // Update aggregates EVERY upload.
      ++sStatTotal;
      sStatBytes += SrcDataSize;
      if (DstOffset < sStatMinOff) sStatMinOff = DstOffset;
      if (DstOffset > sStatMaxOff) sStatMaxOff = DstOffset;
      const uint32_t mod = DstOffset % 768u;
      if (mod == 0u) ++sStatOffZeroMod768;
      else if (mod == 384u) ++sStatOff384Mod768;
      else ++sStatOffOther;
      if (SrcDataSize == 384u) ++sStatLen384;
      else if (SrcDataSize == 768u) ++sStatLen768;
      else ++sStatLenOther;
      // Log individual uploads (throttled to 200/frame). Gated on
      // RTX_BONE_DIAG so the build stays quiet by default.
      if (::dxvk::tf2::boneDiagEnabled() && sCountThisFrameBU < 200) {
        ++sCountThisFrameBU;
        const uint32_t nBones = SrcDataSize / 48u;
        const float* fData = reinterpret_cast<const float*>(pSrcData);
        // For gun/hands diagnostics: at offsets 32256 (palette 42, srvFirstElem=672)
        // and 33024 (palette 43, srvFirstElem=688), dump the FULL 3x4 matrix of
        // the first bone so we can see if rotation is valid (orthonormal) or
        // degenerate (zero/wrong).
        std::string allTx;
        if (DstOffset == 32256 || DstOffset == 33024) {
          // NV-DXVK TF2 viewmodel hunt: dump ALL bones in this group, not
          // just the first. Bone[0] is typically the gun ROOT/HANDLE
          // anchor (held in player's hand at chest/hip height = behind
          // the camera in eye-space). The actual visible gun mesh skins
          // primarily to bones 1..N positioned forward of bone[0] at the
          // grip / barrel / sight. Without seeing them all, we can't tell
          // whether the gun's vertices project in front of the camera.
          for (uint32_t b = 0; b < nBones; ++b) {
            const float* m = fData + b * 12;
            allTx += str::format(
              " B", b, ":r0=(", m[0], ",", m[1], ",", m[2], ") T=(", m[3],
              ",", m[7], ",", m[11], ")");
          }
        } else {
          for (uint32_t b = 0; b < nBones && b < 4; ++b) {
            const float* m = fData + b * 12;
            allTx += str::format(" b", b, ".Tx=", m[3]);
          }
        }
        Logger::info(str::format(
          "[BoneUpload] f=", fid,
          " off=", DstOffset,
          " off%768=", (DstOffset % 768u),
          " len=", SrcDataSize,
          " nBones=", nBones,
          " dstBufPtr=", reinterpret_cast<uintptr_t>(pDstResource),
          allTx));

        // NV-DXVK NPC SKINNING DIAG: dump raw SOURCE memory for the
        // first 2 bones (24 floats / 96 bytes). If bone0 and bone1
        // are byte-identical here, the GAME is uploading filler — the
        // engine itself put the same matrix in N slots before calling
        // UpdateSubresource. If they DIFFER here but our dxvk-level
        // sharedRot=1 fires anyway, something between the d3d11 layer
        // and dxvk is overwriting the upload. The whole-matrix
        // memcmp + per-component dump leaves no ambiguity.
        if (nBones >= 2u) {
          const float* b0 = fData;
          const float* b1 = fData + 12;
          const bool srcShared =
            std::memcmp(b0, b1, 48) == 0;
          Logger::info(str::format(
            "[BoneUploadSrc] f=", fid,
            " off=", DstOffset,
            " nBones=", nBones,
            " srcShared01=", (srcShared ? 1 : 0),
            " B0.R0=(", b0[0], ",", b0[1], ",", b0[2], ")",
            " B0.R1=(", b0[4], ",", b0[5], ",", b0[6], ")",
            " B0.R2=(", b0[8], ",", b0[9], ",", b0[10], ")",
            " B0.T=(", b0[3], ",", b0[7], ",", b0[11], ")",
            " B1.R0=(", b1[0], ",", b1[1], ",", b1[2], ")",
            " B1.R1=(", b1[4], ",", b1[5], ",", b1[6], ")",
            " B1.R2=(", b1[8], ",", b1[9], ",", b1[10], ")",
            " B1.T=(", b1[3], ",", b1[7], ",", b1[11], ")"));

          // NV-DXVK NPC SKINNING DIAG: log the call stack that produced
          // this UpdateSubresource on t30, bucketed by the srcShared01
          // flag so we can separately identify the call site that writes
          // "filler" (shared-R/slight-T) data vs the call site (if any)
          // that writes real per-bone rig data. Each unique stack
          // fingerprint logs once per session to avoid spam.
          // Fingerprint = hash of first 6 return addresses + shared flag.
          {
            std::array<void*, 8> fp{};
            const USHORT got = RtlCaptureStackBackTrace(
              1, static_cast<ULONG>(fp.size()), fp.data(), nullptr);
            uint64_t hash = (srcShared ? 0x9E3779B97F4A7C15ULL : 0x0);
            for (USHORT i = 0; i < got && i < 6; ++i) {
              hash ^= reinterpret_cast<uint64_t>(fp[i]);
              hash *= 0x100000001B3ULL;
            }
            if (::dxvk::tf2::registerBoneStackSiteOnce(hash)) {
              std::string st = ::dxvk::tf2::captureBoneStackTrace(1);
              Logger::info(str::format(
                "[BoneUploadStack] srcShared01=", (srcShared ? 1 : 0),
                " fp=", hash,
                " off=", DstOffset,
                " nBones=", nBones,
                st));
            }
          }
        }
      }
    }
    // Cache cb3 — try multiple common sizes (208, 224, 256, 240)
    if ((BufSize == 208 || BufSize == 224 || BufSize == 240 || BufSize == 256)
        && DstOffset == 0 && SrcDataSize >= 48) {
      std::memcpy(m_cachedCb3, pSrcData, 48);
      m_hasCachedCb3 = true;
    }
  }

  // NV-DXVK TF2 vanish-zone outer-entry-point counters. Defined in
  // d3d11_context.cpp inside namespace dxvk; declared here at namespace scope
  // so the linker resolves them as dxvk::g_d3d11Draw* (rather than global).
  extern std::atomic<uint32_t> g_d3d11DrawAny;
  extern std::atomic<uint32_t> g_d3d11DrawAuto;
  extern std::atomic<uint32_t> g_d3d11DrawIdxIndirect;
  extern std::atomic<uint32_t> g_d3d11DrawInstIndirect;

  void D3D11Rtx::EndFrame(const Rc<DxvkImage>& backbuffer) {
    const uint32_t draws = m_drawCallID;
    const uint32_t raw = m_rawDrawCount;

    // Pull and reset the outer-entry-point counters. Compare to
    // m_rawDrawCount: any > raw means draws are reaching D3D11 entry points
    // but bypassing OnDraw* (e.g. via deferred contexts whose m_rawDrawCount
    // lives on a different D3D11Rtx, or via DrawAuto / *Indirect which never
    // call OnDraw*).
    const uint32_t anyDraw         = g_d3d11DrawAny.exchange(0, std::memory_order_relaxed);
    const uint32_t drawAuto        = g_d3d11DrawAuto.exchange(0, std::memory_order_relaxed);
    const uint32_t drawIdxIndirect = g_d3d11DrawIdxIndirect.exchange(0, std::memory_order_relaxed);
    const uint32_t drawInstIndirect = g_d3d11DrawInstIndirect.exchange(0, std::memory_order_relaxed);

    // NV-DXVK TF2 vanish-zone diagnostic. Pairs with [VanishDiag] in
    // rtx_scene_manager.cpp. raw    = engine OnDraw* calls submitted to us
    //                       captured = draws that reached SubmitDraw with
    //                                  m_lastDrawCaptured=true (i.e. flowed
    //                                  to processDrawCallState).
    // Compare against running baselines (max-seen). Warn if either drops
    // >=10% below baseline so we can tell whether the upstream engine sent
    // fewer draws (raw deficit -> PVS/cluster cull) or our classifier
    // dropped them post-submit (captured deficit only).
    // Drain per-call diagnostic counters (atomic exchange-to-0). These are
    // bumped at every relevant D3D11 entry point on every context.
    uint32_t callSnap[vanish_diag::CALL_COUNT];
    vanish_diag::drain(callSnap);

    {
      static uint32_t s_baselineRaw = 0;
      static uint32_t s_baselineCaptured = 0;
      static uint32_t s_baselineAny = 0;
      static uint32_t s_baselineCalls[vanish_diag::CALL_COUNT] = {};
      static bool     s_haveBaselineCalls = false;
      if (raw > s_baselineRaw) s_baselineRaw = raw;
      if (draws > s_baselineCaptured) s_baselineCaptured = draws;
      if (anyDraw > s_baselineAny) s_baselineAny = anyDraw;
      const bool rawDeficit = s_baselineRaw >= 64 && raw * 10u <= s_baselineRaw * 9u;
      const bool capDeficit = s_baselineCaptured >= 32 && draws * 10u <= s_baselineCaptured * 9u;
      const bool anyDeficit = s_baselineAny >= 64 && anyDraw * 10u <= s_baselineAny * 9u;
      const bool isGoodFrame = s_baselineCaptured >= 32 && draws * 100u >= s_baselineCaptured * 95u;

      if (rawDeficit || capDeficit || anyDeficit) {
        const uint32_t rawDef = s_baselineRaw ? 100u - (raw * 100u) / s_baselineRaw : 0u;
        const uint32_t capDef = s_baselineCaptured ? 100u - (draws * 100u) / s_baselineCaptured : 0u;
        const uint32_t anyDef = s_baselineAny ? 100u - (anyDraw * 100u) / s_baselineAny : 0u;
        Logger::warn(str::format(
          "[D3D11RtxFrame] any=", anyDraw, " baseAny=", s_baselineAny, " anyDef=", anyDef, "%",
          " raw=", raw, " baseRaw=", s_baselineRaw, " rawDef=", rawDef, "%",
          " captured=", draws, " baseCap=", s_baselineCaptured, " capDef=", capDef, "%",
          " auto=", drawAuto, " idxIndirect=", drawIdxIndirect, " instIndirect=", drawInstIndirect));

        // Per-call deltas vs the most recent good frame's snapshot.
        if (s_haveBaselineCalls) {
          for (int i = 0; i < vanish_diag::CALL_COUNT; ++i) {
            const int32_t delta = static_cast<int32_t>(callSnap[i]) - static_cast<int32_t>(s_baselineCalls[i]);
            if (delta != 0) {
              Logger::warn(str::format(
                "[D3D11RtxFrame]   ", vanish_diag::kNames[i],
                " base=", s_baselineCalls[i], " cur=", callSnap[i], " delta=", delta));
            }
          }
        }

        // Aggregate this frame's CopySubresourceRegion events by (src,dst) so
        // we can identify resources being repeatedly copied during the
        // vanish-zone. Resource descriptions were captured at record time
        // (resources may have been Release()'d before EndFrame runs).
        std::vector<vanish_diag::CopyEvent> events;
        vanish_diag::drainCopies(events);
        if (!events.empty()) {
          struct AggKey {
            void* src; void* dst;
            bool operator==(const AggKey& o) const noexcept { return src == o.src && dst == o.dst; }
          };
          struct AggKeyHash {
            size_t operator()(const AggKey& k) const noexcept {
              return std::hash<void*>{}(k.src) ^ (std::hash<void*>{}(k.dst) << 1);
            }
          };
          struct AggValue { uint32_t count; const vanish_diag::CopyEvent* sample; };
          std::unordered_map<AggKey, AggValue, AggKeyHash> agg;
          agg.reserve(events.size());
          for (const auto& e : events) {
            auto& v = agg[{ e.src, e.dst }];
            ++v.count;
            if (v.sample == nullptr) v.sample = &e;
          }
          std::vector<std::pair<AggKey, AggValue>> ranked(agg.begin(), agg.end());
          std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.second.count > b.second.count; });

          const size_t maxLines = std::min<size_t>(16, ranked.size());
          for (size_t i = 0; i < maxLines; ++i) {
            const auto& [k, v] = ranked[i];
            Logger::warn(str::format(
              "[D3D11RtxFrame]   CopySub #", i, " count=", v.count,
              " src=0x", std::hex, reinterpret_cast<uintptr_t>(k.src), std::dec,
              " [", v.sample->srcDesc, "]",
              " dst=0x", std::hex, reinterpret_cast<uintptr_t>(k.dst), std::dec,
              " [", v.sample->dstDesc, "]"));
          }
        }
      } else {
        // Not a cliff frame: still drain so the vector doesn't grow unbounded.
        std::vector<vanish_diag::CopyEvent> tmp;
        vanish_diag::drainCopies(tmp);
      }

      // NV-DXVK [VanishDiag-Raw]: snapshot of OnDraw* VS-hash histogram.
      // Diff against last good-frame histogram so we can identify exactly
      // which VS hashes the engine submitted at peak but is no longer
      // submitting (or — if Remix is dropping somewhere later — which
      // hashes are still entering OnDraw* but not reaching
      // processDrawCallState). Compare against scene_manager's vsHistogram
      // which counts only draws that reached processDrawCallState:
      //   raw[VS] still > 0 but vsHist[VS] == 0  -> dropped INSIDE Remix
      //                                              (between OnDraw* and
      //                                              processDrawCallState)
      //   raw[VS] == 0 in cliff vs > 0 in good   -> engine stopped sending
      static std::unordered_map<uint64_t, uint32_t> s_baselineRawVs;
      static bool s_haveBaselineRawVs = false;
      if (rawDeficit || capDeficit || anyDeficit) {
        if (s_haveBaselineRawVs) {
          struct VsDelta { uint64_t hash; int32_t delta; uint32_t base; uint32_t cur; };
          std::vector<VsDelta> deltas;
          deltas.reserve(s_baselineRawVs.size() + m_rawVsHistogram.size());
          for (const auto& [h, b] : s_baselineRawVs) {
            auto it = m_rawVsHistogram.find(h);
            const uint32_t c = (it != m_rawVsHistogram.end()) ? it->second : 0u;
            const int32_t d = static_cast<int32_t>(c) - static_cast<int32_t>(b);
            if (d != 0) deltas.push_back({ h, d, b, c });
          }
          for (const auto& [h, c] : m_rawVsHistogram) {
            if (s_baselineRawVs.find(h) == s_baselineRawVs.end()) {
              deltas.push_back({ h, static_cast<int32_t>(c), 0u, c });
            }
          }
          std::sort(deltas.begin(), deltas.end(),
            [](const VsDelta& a, const VsDelta& b) { return a.delta < b.delta; });
          const size_t maxLines = std::min<size_t>(12, deltas.size());
          for (size_t i = 0; i < maxLines; ++i) {
            const auto& v = deltas[i];
            Logger::warn(str::format(
              "[VanishDiag-Raw]   VS=0x", std::hex, v.hash, std::dec,
              " base=", v.base, " cur=", v.cur, " delta=", v.delta,
              "  (raw OnDraw* histogram - if base>0 cur=0, engine stopped sending; "
              "if cur>0 but scene_manager VanishDiag VS shows cur=0, Remix dropped it)"));
          }
        }
      }
      if (isGoodFrame) {
        for (int i = 0; i < vanish_diag::CALL_COUNT; ++i) {
          s_baselineCalls[i] = callSnap[i];
        }
        s_haveBaselineCalls = true;
        s_baselineRawVs = m_rawVsHistogram;
        s_haveBaselineRawVs = true;
      }
      // Clear per-frame raw histogram for next frame (do this every frame
      // regardless of deficit so it never accumulates stale counts).
      m_rawVsHistogram.clear();

      // NV-DXVK TF2 engine probe — read materialsystem_dx11.dll's BSP
      // streaming state directly so we can correlate engine-side decisions
      // with the cliff. Offsets come from IDA analysis (image base
      // 0x180000000, so RVA = ida_addr - 0x180000000):
      //   0x1BBCBB4: int32_t  active streaming mode (0..4)
      //   0x1BBCBB8: int32_t  bias
      //   0x1BBCBBC: float    stream_bsp_bucket_bias
      //   0x1BBCBC0: float    stream_bsp_dist_scale
      //   0x1BBCBF8: uint32_t engine frame counter
      //   0x1BBCC00..C08: float[3] streaming camera position (xyz)
      //   0x1BBCC0C..C10: float[2] LOD curve scalars (a,b) -> log(a*256/(b*4096))
      //   0x1BBCBD8: uint32_t linkedTextures
      //   0x1BBCBE0..E8: streaming KB stats
      // Env-var overrides (set before launching TF2):
      //   RTX_TF2_STREAM_MODE=N             force int mode (0..4)
      //   RTX_TF2_STREAM_BUCKET_BIAS=f      override bucket bias float
      //   RTX_TF2_STREAM_DIST_SCALE=f       override dist scale float
      //   RTX_TF2_STREAM_BIAS=N             override int bias
      // Forces are applied once at first frame after the dll is found.
      {
        // NV-DXVK [VanishDiag-A2Hook]: g_vanishDiagCapturedA2 is the
        // namespace-scope global declared near the top of this file
        // — readable from rtx_context.cpp's F11 scene_dump path.
        static HMODULE s_msdx11 = nullptr;
        static bool    s_inited = false;
        if (!s_inited) {
          s_inited = true;
          s_msdx11 = GetModuleHandleA("materialsystem_dx11.dll");
          if (s_msdx11 != nullptr) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(s_msdx11);
            Logger::warn(str::format(
              "[TF2Probe] materialsystem_dx11.dll @ 0x", std::hex, base, std::dec));
            auto applyInt = [&](const char* env, uintptr_t off, const char* lbl) {
              const char* v = std::getenv(env);
              if (v == nullptr) return;
              const int32_t iv = std::atoi(v);
              *reinterpret_cast<int32_t*>(base + off) = iv;
              Logger::warn(str::format("[TF2Probe] FORCE ", lbl, " -> ", iv));
            };
            auto applyFloat = [&](const char* env, uintptr_t off, const char* lbl) {
              const char* v = std::getenv(env);
              if (v == nullptr) return;
              const float fv = std::strtof(v, nullptr);
              *reinterpret_cast<float*>(base + off) = fv;
              Logger::warn(str::format("[TF2Probe] FORCE ", lbl, " -> ", fv));
            };
            applyInt   ("RTX_TF2_STREAM_MODE",        0x1BBCBB4, "stream_mode");
            applyInt   ("RTX_TF2_STREAM_BIAS",        0x1BBCBB8, "stream_bias");
            applyFloat ("RTX_TF2_STREAM_BUCKET_BIAS", 0x1BBCBBC, "stream_bsp_bucket_bias");
            applyFloat ("RTX_TF2_STREAM_DIST_SCALE",  0x1BBCBC0, "stream_bsp_dist_scale");
          } else {
            Logger::warn("[TF2Probe] materialsystem_dx11.dll NOT FOUND");
          }
        }

        // NV-DXVK [VanishDiag-EnginePatch]: patch the per-frame world-mesh
        // batch vertex budget in engine.dll. IDA on Titanfall2's engine.dll
        // shows BuildWorldMeshBatches (sub_1800B6FB0) has a hardcoded cap:
        //   1800B70FF: cmp eax, 300000h        ; 3,145,728 vertices/call
        //   1800B7104: ja  cleanup             ; bail when accumulated > cap
        // v4 (the accumulated count) persists across calls via a static.
        // At low frame rates the engine doesn't re-invoke this job often
        // enough to drain the visibility bitmask before R_DrawWorldMeshes
        // runs — surfaces beyond the 3.1M-vert budget never make it into
        // the mesh batches and silently vanish. Symptom: per-VS-hash
        // entire-family drops in [VanishDiag] (e.g. VS=0x28f7ff base=13
        // cur=0). Patching the immediate to 0x7FFFFFFF removes the cap.
        //
        // Bytes at 0x1800B70FF: 3D 00 00 30 00 = cmp eax, 0x00300000
        // The 4-byte immediate starts at 0x1800B7100 (engine.dll RVA 0xB7100).
        {
          static HMODULE s_engine = nullptr;
          static bool    s_enginePatched = false;
          if (!tf2patches::kPatchActive<tf2patches::kPatchVertexBudget>)
            s_enginePatched = true;
          if (!s_enginePatched) {
            s_engine = GetModuleHandleA("engine.dll");
            if (s_engine != nullptr) {
              const uintptr_t base = reinterpret_cast<uintptr_t>(s_engine);
              const uintptr_t patchAddr = base + 0xB7100;
              // Verify we're patching the expected immediate before writing.
              const uint32_t observed = *reinterpret_cast<const uint32_t*>(patchAddr);
              if (observed == 0x00300000u) {
                DWORD oldProtect = 0;
                if (VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 4,
                                   PAGE_EXECUTE_READWRITE, &oldProtect)) {
                  *reinterpret_cast<uint32_t*>(patchAddr) = 0x7FFFFFFFu;
                  DWORD tmp = 0;
                  VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 4,
                                 oldProtect, &tmp);
                  FlushInstructionCache(GetCurrentProcess(),
                                        reinterpret_cast<LPVOID>(patchAddr), 4);
                  s_enginePatched = true;
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll @ 0x", std::hex, base, std::dec,
                    " — PATCHED BuildWorldMeshBatches vertex budget "
                    "0x00300000 -> 0x7FFFFFFF at 0x", std::hex, patchAddr, std::dec));
                } else {
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll: VirtualProtect FAILED at 0x",
                    std::hex, patchAddr, std::dec));
                  s_enginePatched = true;  // don't keep retrying
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] engine.dll: unexpected bytes at 0x",
                  std::hex, patchAddr, " — observed 0x", observed,
                  " (expected 0x00300000), aborting patch", std::dec));
                s_enginePatched = true;  // don't retry on wrong bytes
              }
            }
            // If engine.dll not loaded yet, retry next frame.
          }
        }

        // NV-DXVK [VanishDiag-EntityGate]: patch the per-frame visibility
        // entity-mask gate in engine.dll. x64dbg HW-write trace on the
        // bucket-bitmask traced back to engine's per-frame "active VIS
        // entity mask" at runtime addr [g+0]; bits in that mask are SET
        // by entity-registration code at engine RVA 0x730DA, but ONLY
        // when a flag at [global+0x5C] is non-zero:
        //
        //   eng+0x730D6   cmp [rcx+0x5C], r14d   ; r14 = 0
        //   eng+0x730DA   jz  +0xF               ; ← 74 0F: skip OR if gate==0
        //   eng+0x730DC   mov ecx, esi
        //   eng+0x730DE   mov eax, 0x01
        //   eng+0x730E3   shl eax, cl
        //   eng+0x730E5   or  [active_mask], eax ; SET entity's bit
        //
        // When the engine registers an entity while the gate is 0, that
        // entity's bit is never OR'd into the active mask → the per-frame
        // visibility loop never iterates that entity → its visible
        // buckets are never marked → R_DrawWorldMeshes silently skips
        // every surface that entity covers → user-visible "floor
        // disappears". Different player positions hit different gate
        // states, hence the position-dependent vanish.
        //
        // Patch: replace the JZ (74 0F) at engine RVA 0x730DA with two
        // NOPs (90 90). Every entity registration now sets its bit
        // unconditionally — matches the user's "if it would contribute,
        // we don't cull it" criterion exactly.
        {
          static bool s_entityGatePatched = false;
          if (!tf2patches::kPatchActive<tf2patches::kPatchEntityMaskGate>)
            s_entityGatePatched = true;
          if (!s_entityGatePatched) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t base = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t patchAddr = base + 0x730DA;
              const uint16_t observed = *reinterpret_cast<const uint16_t*>(patchAddr);
              if (observed == 0x0F74u) {  // little-endian: bytes 74 0F = jz +0xF
                DWORD oldProtect = 0;
                if (VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 2,
                                   PAGE_EXECUTE_READWRITE, &oldProtect)) {
                  *reinterpret_cast<uint16_t*>(patchAddr) = 0x9090u;  // nop nop
                  DWORD tmp = 0;
                  VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 2,
                                 oldProtect, &tmp);
                  FlushInstructionCache(GetCurrentProcess(),
                                        reinterpret_cast<LPVOID>(patchAddr), 2);
                  s_entityGatePatched = true;
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll: PATCHED entity-mask gate JZ "
                    "(74 0F -> 90 90) at 0x", std::hex, patchAddr, std::dec,
                    " — entity bits now set unconditionally; surfaces no "
                    "longer vanish from gate-induced cull"));
                } else {
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll: VirtualProtect FAILED for "
                    "entity-gate patch at 0x", std::hex, patchAddr, std::dec));
                  s_entityGatePatched = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] engine.dll: unexpected entity-gate bytes "
                  "at 0x", std::hex, patchAddr, " — observed 0x", observed,
                  " (expected 0x0F74), aborting patch", std::dec));
                s_entityGatePatched = true;
              }
            }
            // engine.dll not loaded yet → retry next frame.
          }
        }

        // NV-DXVK [VanishDiag-DispatchEntryE-Force0]: STATIC PATCH that
        // replaces the dispatcher's read of entry[+0xE] with `xor esi,esi`
        // (force E = 0). Eliminates the floor-vanish bug at zero runtime
        // cost.
        //
        // Discovered via IDA decompile of sub_1801B31E0 + observed v12
        // dispatch-hook fix:
        //   v14 = entry[+0xD] & (v11 ^ entry[+0xE])
        //   if (v14 == 0) skip entry
        //   Two passes with v11 = -1 (pass 1) and v11 = 0 (pass 2). Pass 1
        //   processes "kept and not in fade band". Pass 2 processes "kept
        //   and in fade band".
        //
        // Bug: entry[+0xE] is occasionally read with stale "all bits set"
        // state (race vs. sub_1801B2200 producer thread). With stale E:
        //   pass 1: D & ~E = D & ~D = 0 → SKIP (bug — floor missing)
        //   pass 2: D & E  = D & D  = D → process (but pass 2 is fade-blend
        //                                        only, doesn't cover the
        //                                        full-quality draws lost in
        //                                        pass 1)
        //
        // Forcing E = 0 makes:
        //   pass 1: D & ~0 = D → process (skip only when D == 0, the
        //                                  legitimate "no parts enabled" case)
        //   pass 2: D & 0  = 0 → ALWAYS SKIP → no fade-blend rendering
        //
        // Side effect: distant props that should fade-blend now pop in
        // hard at the cull radius. Cosmetic; not user-reported as an
        // issue, and a small price for permanent geo correctness.
        //
        // Patch: 5 bytes at engine RVA 0x1B32DF.
        //   was: 41 0F B6 76 0E   (movzx esi, byte ptr [r14+0xE])
        //   new: 33 F6 90 90 90   (xor esi, esi  +  3 nops)
        {
          static bool s_dispatchEntryEForcePatched = false;
          if (!tf2patches::kPatchActive<tf2patches::kPatchDispatchEntryE>)
            s_dispatchEntryEForcePatched = true;
          if (!s_dispatchEntryEForcePatched) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t base = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t patchAddr = base + 0x1B32DF;
              const uint8_t* p = reinterpret_cast<const uint8_t*>(patchAddr);
              if (p[0] == 0x41 && p[1] == 0x0F && p[2] == 0xB6 &&
                  p[3] == 0x76 && p[4] == 0x0E) {
                DWORD oldProtect = 0;
                if (VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 5,
                                   PAGE_EXECUTE_READWRITE, &oldProtect)) {
                  uint8_t* tp = reinterpret_cast<uint8_t*>(patchAddr);
                  tp[0] = 0x33; tp[1] = 0xF6;        // xor esi, esi
                  tp[2] = 0x90; tp[3] = 0x90; tp[4] = 0x90;  // nop * 3
                  DWORD tmp = 0;
                  VirtualProtect(reinterpret_cast<LPVOID>(patchAddr), 5,
                                 oldProtect, &tmp);
                  FlushInstructionCache(GetCurrentProcess(),
                                        reinterpret_cast<LPVOID>(patchAddr), 5);
                  s_dispatchEntryEForcePatched = true;
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll: PATCHED dispatcher entry[+0xE] "
                    "read (movzx → xor esi,esi + nops) at 0x",
                    std::hex, patchAddr, std::dec,
                    " — pass1 always processes when D != 0; pass2 is no-op; "
                    "floor stays drawn through stale-E races; ZERO per-call cost"));
                } else {
                  Logger::warn(str::format(
                    "[TF2Probe] engine.dll: VirtualProtect FAILED for "
                    "dispatch-entryE-force patch at 0x", std::hex, patchAddr,
                    std::dec));
                  s_dispatchEntryEForcePatched = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] engine.dll: unexpected dispatch-entryE bytes "
                  "at 0x", std::hex, patchAddr, " — observed ",
                  uint32_t(p[0]), " ", uint32_t(p[1]), " ", uint32_t(p[2]),
                  " ", uint32_t(p[3]), " ", uint32_t(p[4]),
                  " (expected 41 0F B6 76 0E), aborting patch", std::dec));
                s_dispatchEntryEForcePatched = true;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-ProducerMFence]: minimal mfence-only hook at
        // sub_1801B2200's prologue. Test of the hypothesis that the v12
        // entry hook's load-bearing fix mechanism was just the producer-
        // side memory barrier (drains store buffer at start of producer
        // run, ensures any pending stores from prior code become visible).
        //
        // Patch site: 8 bytes at engBase + 0x1B2200
        //   was: 48 8B C4 53 55 57 41 55  (mov rax,rsp; push rbx; push rbp;
        //                                  push rdi; push r13)
        //   new: E9 disp32 + nop*3        (jmp to trampoline, 5+3 bytes)
        //
        // Trampoline:
        //   mfence                  ; 0F AE F0
        //   mov rax, rsp            ; 48 8B C4
        //   push rbx                ; 53
        //   push rbp                ; 55
        //   push rdi                ; 57
        //   push r13                ; 41 55  (replicates 8-byte prologue)
        //   jmp rel32 → engBase + 0x1B2208  ; E9 + disp32
        //
        // Cost: 1 mfence per sub_1801B2200 call ≈ 5 calls/frame ×
        //       ~30-50 cycles = ~80 ns/frame. Free.
        {
          // v28: DISABLED. The mfence-alone hypothesis was wrong (v27
          // tested it; floor stayed missing). The entry hook (re-enabled
          // above) patches the same 0x1B2200 prologue, so leaving this
          // disabled also avoids a patch-site conflict.
          static bool s_producerMFenceHookInstalled = true;
          if (!s_producerMFenceHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B2200;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x48 && tgt[1] == 0x8B && tgt[2] == 0xC4 &&
                  tgt[3] == 0x53 && tgt[4] == 0x55 && tgt[5] == 0x57 &&
                  tgt[6] == 0x41 && tgt[7] == 0x55) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;

                  // mfence  ; 0F AE F0  (3 bytes)
                  *p++ = 0x0F; *p++ = 0xAE; *p++ = 0xF0;

                  // Replicate 8-byte prologue:
                  // mov rax, rsp; push rbx; push rbp; push rdi; push r13
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0xC4;
                  *p++ = 0x53;
                  *p++ = 0x55;
                  *p++ = 0x57;
                  *p++ = 0x41; *p++ = 0x55;

                  // jmp rel32 → engBase + 0x1B2208 (after the prologue)
                  *p++ = 0xE9;
                  {
                    const uintptr_t back = engBase + 0x1B2208;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(back) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Patch 8 bytes at target: E9 disp32 + nop*3
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 8,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90; tp[6] = 0x90; tp[7] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 8,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 8);
                    s_producerMFenceHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] producer-mfence hook installed at 0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " size=", std::dec, static_cast<uint32_t>(p - tramp), " bytes",
                      " — minimal mfence at sub_1801B2200 prologue"));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] producer-mfence hook: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_producerMFenceHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] producer-mfence hook: no trampoline alloc within ±2GB");
                  s_producerMFenceHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] producer-mfence prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — expected 48 8B C4 53 55 57 41 55, aborting patch"));
                s_producerMFenceHookInstalled = true;
              }
            }
          }
        }

        // [VanishDiag-PropDistanceCull NOP probe REMOVED — crashed at startup
        //  because it bypasses safety bounds on the visible-prop output
        //  arrays and triggers a 1.0/(radius² × zoom² - radius²) divide-by-
        //  zero when zoom=0. Need a softer probe: log sceneScale + scale
        //  values to confirm they're wrong (Remix-bug hypothesis), then
        //  patch the SCALE input rather than the cull JA.]

        // NV-DXVK [VanishDiag-A2Hook]: install a trampoline at engine.dll
        // R_DrawWorldMeshes that captures its 'a2' parameter (rdx, the
        // per-frame world-render struct) into a Remix global so we can
        // read the bitmask `[a2+0x54088]` each frame from the diag block
        // below. The previous round of fixes assumed the cull lived in
        // the entity mask — runtime sampling proved that mask is 0x01 at
        // BOTH visible and vanished positions, so the differentiator is
        // somewhere else. With a2 captured we can definitively diff the
        // bucket bitmask between visible and vanished frames and trace
        // the actual cause from there.
        //
        // Hook layout:
        //   target+0..6:  48 8B C4 44 89 40 18  ; mov rax,rsp; mov [rax+18],r8d
        // We replace those 7 bytes with:
        //   E9 NN NN NN NN 90 90                ; jmp rel32 → trampoline; nop nop
        // Trampoline:
        //   48 8B C4 44 89 40 18                ; original 7 bytes
        //   48 89 15 NN NN NN NN                ; mov [rip+disp32], rdx (save a2)
        //   E9 NN NN NN NN                       ; jmp rel32 back to target+7
        {
          static bool s_a2HookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookRDrawWorldMeshes>)
            s_a2HookInstalled = true;
          if (!s_a2HookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0xB7DD0;  // R_DrawWorldMeshes
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              // Verify the prolog matches what we IDA'd.
              if (tgt[0] == 0x48 && tgt[1] == 0x8B && tgt[2] == 0xC4 &&
                  tgt[3] == 0x44 && tgt[4] == 0x89 && tgt[5] == 0x40 &&
                  tgt[6] == 0x18) {
                // Allocate a 4KB executable trampoline near the target so
                // the rel32 jmps reach. Step in 64KB increments downward
                // first, then upward, until VirtualAlloc succeeds and the
                // returned address is within ±0x7FFF0000 of target.
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  // Try below target.
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  // Try above target.
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;

                  // Trampoline layout:
                  //   1. Save volatile regs we'll clobber.
                  //   2. Snapshot bitmask: rep movsq 8 qwords from
                  //      [rdx+0x54088] to g_vanishDiagBitmaskSnap.
                  //   3. mov [rip+disp32], rdx  — save a2 to our global.
                  //   4. Restore regs.
                  //   5. Run original 7 bytes from target's prolog.
                  //   6. jmp rel32 back to target+7.
                  //
                  // We clobber rax, rcx, rsi, rdi, r10, r11 + flags.
                  // r10/r11 are caller-clobber under Windows x64 ABI.
                  // rax/rcx/rsi/rdi are also caller-clobber. The
                  // function's first instruction was `mov rax, rsp`
                  // which is ABOUT to overwrite rax anyway. We
                  // push/pop everything to be safe across calls.

                  // 1. push rax, rcx, rsi, rdi, r10, r11
                  *p++ = 0x50;                         // push rax
                  *p++ = 0x51;                         // push rcx
                  *p++ = 0x56;                         // push rsi
                  *p++ = 0x57;                         // push rdi
                  *p++ = 0x41; *p++ = 0x52;            // push r10
                  *p++ = 0x41; *p++ = 0x53;            // push r11

                  // 2. OR-accumulate first 8 qwords of [rdx+0x54088]
                  //    into g_vanishDiagBitmaskSnap. Multiple calls per
                  //    frame to R_DrawWorldMeshes (main + shadow + portal
                  //    passes) — overwriting via rep movsq lets the last
                  //    pass's empty bitmask wipe out the main view's
                  //    bits. OR-ing accumulates all calls across the
                  //    frame; EndFrame reads then resets to 0.
                  //
                  //    Loop unrolled (8 iterations × 9 bytes each = 72 bytes):
                  //      mov rax, [rdx + 0x54088 + i*8]   ; 7 bytes
                  //      or  [rip + disp32_i], rax        ; 7 bytes
                  //    But to keep the trampoline compact we do
                  //    REX.W or [rip+disp32], rax with the imm32 disp32
                  //    targeting each snapshot word.
                  for (int i = 0; i < 8; ++i) {
                    // mov rax, [rdx + (0x54088 + i*8)]
                    //   48 8B 82 disp32
                    *p++ = 0x48; *p++ = 0x8B; *p++ = 0x82;
                    {
                      const int32_t off = 0x54088 + i * 8;
                      std::memcpy(p, &off, 4); p += 4;
                    }
                    // or [rip + disp32], rax
                    //   48 09 05 disp32
                    *p++ = 0x48; *p++ = 0x09; *p++ = 0x05;
                    {
                      const uintptr_t snapAddr =
                        reinterpret_cast<uintptr_t>(&g_vanishDiagBitmaskSnap[i]);
                      const int32_t disp = static_cast<int32_t>(
                        static_cast<intptr_t>(snapAddr) -
                        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                      std::memcpy(p, &disp, 4); p += 4;
                    }
                  }

                  // 2b. OR-accumulate first 8 qwords of qword_192205120
                  //     (engine.dll RVA 0x12205120) into g_vanishDiagGlobalSnap.
                  //     Both source and destination are within ±2GB of the
                  //     trampoline page (source is in engine.dll itself, dest
                  //     in remix dll loaded near it), so RIP-relative is safe.
                  for (int i = 0; i < 8; ++i) {
                    // mov rax, [rip + disp32]   ; src = engBase + 0x12205120 + i*8
                    //   48 8B 05 disp32
                    *p++ = 0x48; *p++ = 0x8B; *p++ = 0x05;
                    {
                      const uintptr_t srcAddr =
                        engBase + 0x12205120 + static_cast<uintptr_t>(i) * 8;
                      const int32_t disp = static_cast<int32_t>(
                        static_cast<intptr_t>(srcAddr) -
                        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                      std::memcpy(p, &disp, 4); p += 4;
                    }
                    // or [rip + disp32], rax    ; dst = &g_vanishDiagGlobalSnap[i]
                    //   48 09 05 disp32
                    *p++ = 0x48; *p++ = 0x09; *p++ = 0x05;
                    {
                      const uintptr_t snapAddr =
                        reinterpret_cast<uintptr_t>(&g_vanishDiagGlobalSnap[i]);
                      const int32_t disp = static_cast<int32_t>(
                        static_cast<intptr_t>(snapAddr) -
                        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                      std::memcpy(p, &disp, 4); p += 4;
                    }
                  }

                  // 3. mov [rip+disp32], rdx  — save a2 to our global.
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x15;
                  {
                    const uintptr_t globalAddr =
                      reinterpret_cast<uintptr_t>(&g_vanishDiagCapturedA2);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(globalAddr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4);
                    p += 4;
                  }

                  // 3b. mov [rip+disp32], r8d  — save a3 (flag word) to global.
                  //     Encoding: 44 89 05 disp32 (REX.R=1 for r8, MOV r/m32,r32, RIP-rel).
                  *p++ = 0x44; *p++ = 0x89; *p++ = 0x05;
                  {
                    const uintptr_t a3Addr =
                      reinterpret_cast<uintptr_t>(&g_vanishDiagCapturedA3);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(a3Addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4);
                    p += 4;
                  }

                  // 3c. Capture BuildWorldMeshBatches outputs from a2 (rdx):
                  //     a2[+0..+8]  → g_buildBatchesPassEnds[0..1] (qword copy)
                  //     a2[+8..+16] → g_buildBatchesPassEnds[2..3] (qword copy)
                  //     a2[+0x8010] → g_buildBatchesBatchCount (dword)

                  // mov rax, [rdx + 0]                       ; 48 8B 02
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0x02;
                  // mov [rip + disp32], rax                  ; 48 89 05 disp32
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_buildBatchesPassEnds[0]);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // mov rax, [rdx + 8]                       ; 48 8B 42 08
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0x42; *p++ = 0x08;
                  // mov [rip + disp32], rax                  ; 48 89 05 disp32
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_buildBatchesPassEnds[2]);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // mov eax, [rdx + 0x8010]                  ; 8B 82 disp32
                  *p++ = 0x8B; *p++ = 0x82;
                  {
                    const int32_t off = 0x8010;
                    std::memcpy(p, &off, 4); p += 4;
                  }
                  // mov [rip + disp32], eax                  ; 89 05 disp32
                  *p++ = 0x89; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_buildBatchesBatchCount);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // [VanishDiag-Probe-Force* probes removed — both single-bit
                  //  bucket-401 force and full all-ones force-fill failed to
                  //  restore the floor. This proves the WorldVis bitmask
                  //  `[a2+0x54088]` is NOT the cull mechanism in this scene.
                  //  The cull is downstream of R_DrawWorldMeshes' bucket loop.]

                  // 4. pop r11, r10, rdi, rsi, rcx, rax  (reverse order)
                  *p++ = 0x41; *p++ = 0x5B;            // pop r11
                  *p++ = 0x41; *p++ = 0x5A;            // pop r10
                  *p++ = 0x5F;                         // pop rdi
                  *p++ = 0x5E;                         // pop rsi
                  *p++ = 0x59;                         // pop rcx
                  *p++ = 0x58;                         // pop rax

                  // 5. Replicate target's first 7 bytes.
                  std::memcpy(p, reinterpret_cast<const void*>(target), 7);
                  p += 7;

                  // 6. jmp rel32 back to target+7.
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + 7) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // 4. Patch target's first 7 bytes:
                  //      E9 NN NN NN NN 90 90  (jmp rel32 → tramp; nop; nop)
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90;
                    tp[6] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 7);
                    s_a2HookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] R_DrawWorldMeshes hook installed: "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " a2_global=0x", reinterpret_cast<uintptr_t>(&g_vanishDiagCapturedA2),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] VirtualProtect failed at target 0x",
                      std::hex, target, std::dec, " — abort hook"));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_a2HookInstalled = true;  // don't retry
                  }
                } else {
                  Logger::warn("[TF2Probe] R_DrawWorldMeshes hook: "
                               "no trampoline alloc within ±2GB; abort");
                  s_a2HookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] R_DrawWorldMeshes prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ... — abort hook"));
                s_a2HookInstalled = true;
              }
            }
            // engine.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-B45D0Hook]: trampoline at engine.dll RVA 0xB4870
        // — the 7-byte sequence `and esi, 3Fh; shr rdi, 6` immediately preceding
        // the OR-into-qword_192205120 site at 0xB488A inside sub_1800B45D0
        // (the BVH-leaf processor that ORs bucket-dirty bits). At RVA 0xB4870
        // ESI still holds the FULL 16-bit bucket index v17; one instruction
        // later it gets masked to its low 6 bits. We snapshot ESI here, bump
        // g_vanishDiagBucketHist[esi]++, then run the displaced bytes and jump
        // back to 0xB4877 (mov eax, 1). LOCK INC is used because sub_1800B45D0
        // runs from a JT job (sub_1800B4B20 dispatches it) — multiple workers
        // may execute concurrently.
        //
        // Hook layout:
        //   0xB4870..0xB4876:  83 E6 3F 48 C1 EF 06   ; and esi,3Fh; shr rdi,6
        // We replace those 7 bytes with:
        //   E9 NN NN NN NN 90 90                     ; jmp rel32 → tramp; nops
        // Trampoline (41 bytes):
        //   50                                  ; push rax
        //   51                                  ; push rcx
        //   0F B7 CE                            ; movzx ecx, si  (v17 in ecx)
        //   81 F9 00 04 00 00                   ; cmp ecx, 1024
        //   73 0E                               ; jae +14 → skip
        //   48 B8 NN NN NN NN NN NN NN NN       ; mov rax, &histogram
        //   F0 FF 04 88                         ; lock inc dword [rax+rcx*4]
        // skip:
        //   59                                  ; pop rcx
        //   58                                  ; pop rax
        //   83 E6 3F                            ; and esi, 3Fh   (replicate)
        //   48 C1 EF 06                         ; shr rdi, 6     (replicate)
        //   E9 NN NN NN NN                      ; jmp rel32 → 0xB4877
        {
          static bool s_b45d0HookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookSub1800B45D0>)
            s_b45d0HookInstalled = true;
          if (!s_b45d0HookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0xB4870;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x83 && tgt[1] == 0xE6 && tgt[2] == 0x3F &&
                  tgt[3] == 0x48 && tgt[4] == 0xC1 && tgt[5] == 0xEF &&
                  tgt[6] == 0x06) {
                // Allocate a 4KB executable trampoline near the target.
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;

                  // push rax; push rcx
                  *p++ = 0x50;
                  *p++ = 0x51;

                  // movzx ecx, si    ; ecx = v17 (zero-extended u16)
                  *p++ = 0x0F; *p++ = 0xB7; *p++ = 0xCE;

                  // cmp ecx, 1024
                  *p++ = 0x81; *p++ = 0xF9;
                  {
                    const int32_t imm = 1024;
                    std::memcpy(p, &imm, 4); p += 4;
                  }

                  // jae +14  (skip movabs + lock inc)
                  *p++ = 0x73; *p++ = 0x0E;

                  // mov rax, &g_vanishDiagBucketHist
                  *p++ = 0x48; *p++ = 0xB8;
                  {
                    const uintptr_t histAddr =
                      reinterpret_cast<uintptr_t>(&g_vanishDiagBucketHist[0]);
                    std::memcpy(p, &histAddr, 8); p += 8;
                  }

                  // lock inc dword [rax + rcx*4]
                  *p++ = 0xF0; *p++ = 0xFF; *p++ = 0x04; *p++ = 0x88;

                  // pop rcx; pop rax
                  *p++ = 0x59;
                  *p++ = 0x58;

                  // Replicated displaced bytes: and esi, 3Fh; shr rdi, 6
                  *p++ = 0x83; *p++ = 0xE6; *p++ = 0x3F;
                  *p++ = 0x48; *p++ = 0xC1; *p++ = 0xEF; *p++ = 0x06;

                  // jmp rel32 → target + 7 (= 0xB4877)
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + 7) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // Patch target's 7 bytes: E9 NN NN NN NN 90 90
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90;
                    tp[6] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 7);
                    s_b45d0HookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1800B45D0 OR-site hook installed: "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " hist=0x", reinterpret_cast<uintptr_t>(&g_vanishDiagBucketHist[0]),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1800B45D0: VirtualProtect failed at 0x",
                      std::hex, target, std::dec, " — abort hook"));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_b45d0HookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] sub_1800B45D0 hook: "
                               "no trampoline alloc within ±2GB; abort");
                  s_b45d0HookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] sub_1800B45D0 prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]), " — abort hook"));
                s_b45d0HookInstalled = true;
              }
            }
            // engine.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-B30Hook]: trampoline at client.dll
        // sub_18036BD30 (RVA 0x36BD30). This wrapper takes
        //   sub_18036BD30(this, view_ctx, source_bitmask) {
        //     view_ctx[0x1D0] = sub_1802EF230(source_bitmask);  // memmoves
        //     jmp sub_1801A8170(view_ctx, 0);
        //   }
        // The third arg `r8 = source_bitmask` points to a per-view 64-byte
        // bitmask buffer that gets memmoved into a fresh WriterStruct. The
        // hook captures r8, rdx (view_ctx), and the first 8 qwords of [r8]
        // for per-frame logging. Last-fire-wins single-slot, no thread
        // safety — diagnostic only.
        //
        // Original prolog: 53 48 83 EC 20  (push rbx; sub rsp, 0x20)  = 5 bytes
        // We replace those 5 bytes with E9 NN NN NN NN (jmp rel32). Clean
        // 5-byte fit, no nops needed.
        //
        // Trampoline layout (143 bytes):
        //   48 89 15 disp32                  ; mov [rip+disp32], rdx (view_ctx)
        //   4C 89 05 disp32                  ; mov [rip+disp32], r8  (source_bm)
        //   8 × {
        //     49 8B 80 disp32                ; mov rax, [r8 + i*8]
        //     48 89 05 disp32                ; mov [rip+disp32], rax (snap[i])
        //   }
        //   F0 FF 05 disp32                  ; lock inc dword [rip+disp32] (count)
        //   53                                ; push rbx     (replicate)
        //   48 83 EC 20                       ; sub rsp, 0x20 (replicate)
        //   E9 NN NN NN NN                    ; jmp rel32 → target+5
        //
        // We clobber rax. rax is volatile and the function's prolog
        // doesn't read it before clobbering, so no save needed.
        {
          static bool s_b30HookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookSub18036BD30>)
            s_b30HookInstalled = true;
          if (!s_b30HookInstalled) {
            HMODULE cli = GetModuleHandleA("client.dll");
            if (cli != nullptr) {
              const uintptr_t cliBase = reinterpret_cast<uintptr_t>(cli);
              const uintptr_t target  = cliBase + 0x36BD30;  // sub_18036BD30
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              // Two prolog encodings observed across client.dll builds:
              //   short:  53 48 83 EC 20            (5 bytes — IDA database)
              //   long:   40 53 48 83 EC 20         (6 bytes — actual runtime, redundant REX)
              // We auto-detect and adjust patch size accordingly.
              const bool shortProlog = (tgt[0] == 0x53 && tgt[1] == 0x48 &&
                                        tgt[2] == 0x83 && tgt[3] == 0xEC &&
                                        tgt[4] == 0x20);
              const bool longProlog  = (tgt[0] == 0x40 && tgt[1] == 0x53 &&
                                        tgt[2] == 0x48 && tgt[3] == 0x83 &&
                                        tgt[4] == 0xEC && tgt[5] == 0x20);
              const uint32_t prologLen = shortProlog ? 5 : (longProlog ? 6 : 0);
              if (prologLen != 0) {
                // Allocate trampoline within ±2GB of target.
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;

                  // mov [rip+disp32], rdx  — capture view_ctx
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x15;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_b30_view_ctx);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // mov [rip+disp32], r8  — capture source_bitmask
                  *p++ = 0x4C; *p++ = 0x89; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_b30_source_bm);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // 8 × { mov rax, [r8 + i*8]; mov [rip+disp32_i], rax }
                  for (int i = 0; i < 8; ++i) {
                    // mov rax, [r8 + (i*8)]: 49 8B 80 disp32
                    *p++ = 0x49; *p++ = 0x8B; *p++ = 0x80;
                    {
                      const int32_t off = i * 8;
                      std::memcpy(p, &off, 4); p += 4;
                    }
                    // mov [rip+disp32], rax: 48 89 05 disp32
                    *p++ = 0x48; *p++ = 0x89; *p++ = 0x05;
                    {
                      const uintptr_t snapAddr =
                        reinterpret_cast<uintptr_t>(&g_b30_snap[i]);
                      const int32_t disp = static_cast<int32_t>(
                        static_cast<intptr_t>(snapAddr) -
                        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                      std::memcpy(p, &disp, 4); p += 4;
                    }
                  }

                  // lock inc dword [rip+disp32] — atomic call counter
                  *p++ = 0xF0; *p++ = 0xFF; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_b30_call_count);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Replicated displaced bytes (copy actual runtime prolog
                  // bytes verbatim — handles both short and long encodings).
                  std::memcpy(p, reinterpret_cast<const void*>(target), prologLen);
                  p += prologLen;

                  // jmp rel32 → target + prologLen
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + prologLen) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // Patch prologLen bytes at target: E9 NN NN NN NN [+ 90 nop]
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), prologLen,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    // For long prolog (6 bytes), pad the trailing byte with NOP.
                    if (prologLen == 6) tp[5] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), prologLen,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), prologLen);
                    s_b30HookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] sub_18036BD30 hook installed: "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " src_global=0x", reinterpret_cast<uintptr_t>(&g_b30_source_bm),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] sub_18036BD30: VirtualProtect failed at 0x",
                      std::hex, target, std::dec, " — abort hook"));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_b30HookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] sub_18036BD30 hook: "
                               "no trampoline alloc within ±2GB; abort");
                  s_b30HookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] sub_18036BD30 prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]),
                  " — abort hook (expected short '53 48 83 EC 20' or "
                  "long '40 53 48 83 EC 20')"));
                s_b30HookInstalled = true;
              }
            }
            // client.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-EB290Hook]: trampoline at client.dll
        // sub_1802EB290 (RVA 0x2EB290) — the per-bucket visibility test
        // invoked from sub_1802EB1E0 (the JT job that ORs bits into the
        // main view's WriterStruct bitmask). At entry, edx = a2 = bucket
        // index. We bump g_eb290_hist[a2] and a global call counter.
        //
        // Original prolog: 48 8B C4 48 89 58 08  (mov rax, rsp; mov [rax+8], rbx) = 7 bytes
        // We replace those 7 bytes with E9 NN NN NN NN 90 90 (jmp + 2 nops).
        //
        // Trampoline (~41 bytes):
        //   50                                ; push rax
        //   51                                ; push rcx
        //   8B CA                             ; mov ecx, edx       (ecx = a2)
        //   81 F9 00 08 00 00                  ; cmp ecx, 2048
        //   73 0E                              ; jae +14 → skip
        //   48 B8 NN NN NN NN NN NN NN NN      ; mov rax, &g_eb290_hist
        //   F0 FF 04 88                        ; lock inc dword [rax+rcx*4]
        // skip:
        //   59                                 ; pop rcx
        //   58                                 ; pop rax
        //   48 8B C4                           ; mov rax, rsp        (replicate)
        //   48 89 58 08                        ; mov [rax+8], rbx   (replicate)
        //   E9 NN NN NN NN                     ; jmp rel32 → target+7
        //
        // Plus a small atomic call-counter bump (separate disp32) for
        // total-call diagnostic.
        {
          static bool s_eb290HookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookSub1802EB290>)
            s_eb290HookInstalled = true;
          if (!s_eb290HookInstalled) {
            HMODULE cli = GetModuleHandleA("client.dll");
            if (cli != nullptr) {
              const uintptr_t cliBase = reinterpret_cast<uintptr_t>(cli);
              const uintptr_t target  = cliBase + 0x2EB290;  // sub_1802EB290
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x48 && tgt[1] == 0x8B && tgt[2] == 0xC4 &&
                  tgt[3] == 0x48 && tgt[4] == 0x89 && tgt[5] == 0x58 &&
                  tgt[6] == 0x08) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;

                  // push rax; push rcx
                  *p++ = 0x50;
                  *p++ = 0x51;

                  // mov ecx, edx     ; ecx = a2 (bucket index)
                  *p++ = 0x8B; *p++ = 0xCA;

                  // cmp ecx, 2048
                  *p++ = 0x81; *p++ = 0xF9;
                  {
                    const int32_t imm = 2048;
                    std::memcpy(p, &imm, 4); p += 4;
                  }

                  // jae +14
                  *p++ = 0x73; *p++ = 0x0E;

                  // mov rax, &g_eb290_hist[0]
                  *p++ = 0x48; *p++ = 0xB8;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_eb290_hist[0]);
                    std::memcpy(p, &addr, 8); p += 8;
                  }

                  // lock inc dword [rax + rcx*4]
                  *p++ = 0xF0; *p++ = 0xFF; *p++ = 0x04; *p++ = 0x88;

                  // pop rcx; pop rax
                  *p++ = 0x59;
                  *p++ = 0x58;

                  // Bump unconditional call counter (no bounds check):
                  // lock inc dword [rip+disp32_count]
                  *p++ = 0xF0; *p++ = 0xFF; *p++ = 0x05;
                  {
                    const uintptr_t addr =
                      reinterpret_cast<uintptr_t>(&g_eb290_call_count);
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Replicated displaced 7 bytes:
                  //   48 8B C4         (mov rax, rsp)
                  //   48 89 58 08      (mov [rax+8], rbx)
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0xC4;
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x58; *p++ = 0x08;

                  // jmp rel32 → target + 7
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + 7) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // Patch 7 bytes: E9 NN NN NN NN 90 90
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90;
                    tp[6] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 7);
                    s_eb290HookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1802EB290 hook installed: "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " hist=0x", reinterpret_cast<uintptr_t>(&g_eb290_hist[0]),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1802EB290: VirtualProtect failed at 0x",
                      std::hex, target, std::dec, " — abort hook"));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_eb290HookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] sub_1802EB290 hook: "
                               "no trampoline alloc within ±2GB; abort");
                  s_eb290HookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] sub_1802EB290 prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]),
                  " — abort (expected 48 8B C4 48 89 58 08)"));
                s_eb290HookInstalled = true;
              }
            }
            // client.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-B84C0Hook]: trampoline at engine.dll
        // sub_1800B84C0 (RVA 0xB84C0) — the per-pass draw-list submit
        // called from R_DrawWorldMeshes. Capture inputs:
        //   rcx = a1 (WriterStruct), edx = a2 (filter mask), r8d = a3 (pass)
        //   plus a1[a3], a1[a3+1] (the range start/end indices).
        //
        // Original prolog (9 bytes):
        //   53 56 41 56 B8 30 80 00 00
        //   = push rbx; push rsi; push r14; mov eax, 0x8030
        // Replace with E9 NN NN NN NN 90 90 90 90 (jmp + 4 nops).
        //
        // Trampoline (~62 bytes) — no register save needed; all captures
        // use mov [rip+disp32], reg which doesn't clobber any reg:
        //   48 89 0D disp32                      ; mov [rip+disp32], rcx (a1)
        //   89 15 disp32                         ; mov [rip+disp32], edx (mask)
        //   44 89 05 disp32                      ; mov [rip+disp32], r8d (pass)
        //   42 8B 04 81                          ; mov eax, [rcx + r8*4]    (range_start)
        //   89 05 disp32                         ; mov [rip+disp32], eax
        //   42 8B 44 81 04                       ; mov eax, [rcx + r8*4 + 4] (range_end)
        //   89 05 disp32                         ; mov [rip+disp32], eax
        //   F0 FF 05 disp32                      ; lock inc dword [rip+disp32]
        //   53 56 41 56 B8 30 80 00 00           ; replicated prolog
        //   E9 NN NN NN NN                       ; jmp rel32 → target+9
        {
          static bool s_b84c0HookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookSub1800B84C0>)
            s_b84c0HookInstalled = true;
          if (!s_b84c0HookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0xB84C0;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x53 && tgt[1] == 0x56 && tgt[2] == 0x41 &&
                  tgt[3] == 0x56 && tgt[4] == 0xB8 && tgt[5] == 0x30 &&
                  tgt[6] == 0x80 && tgt[7] == 0x00 && tgt[8] == 0x00) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDispTo = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // mov [rip+disp32], rcx
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x0D;
                  emitRipDispTo((void*)&g_b84c0_a1);

                  // mov [rip+disp32], edx
                  *p++ = 0x89; *p++ = 0x15;
                  emitRipDispTo((void*)&g_b84c0_filter_mask);

                  // mov [rip+disp32], r8d
                  *p++ = 0x44; *p++ = 0x89; *p++ = 0x05;
                  emitRipDispTo((void*)&g_b84c0_pass_idx);

                  // mov eax, [rcx + r8*4]   (range_start = a1[a3])
                  *p++ = 0x42; *p++ = 0x8B; *p++ = 0x04; *p++ = 0x81;

                  // mov [rip+disp32], eax
                  *p++ = 0x89; *p++ = 0x05;
                  emitRipDispTo((void*)&g_b84c0_range_start);

                  // mov eax, [rcx + r8*4 + 4]  (range_end = a1[a3+1])
                  *p++ = 0x42; *p++ = 0x8B; *p++ = 0x44; *p++ = 0x81; *p++ = 0x04;

                  // mov [rip+disp32], eax
                  *p++ = 0x89; *p++ = 0x05;
                  emitRipDispTo((void*)&g_b84c0_range_end);

                  // lock inc dword [rip+disp32]
                  *p++ = 0xF0; *p++ = 0xFF; *p++ = 0x05;
                  emitRipDispTo((void*)&g_b84c0_call_count);

                  // Replicated 9 bytes: push rbx; push rsi; push r14; mov eax, 0x8030
                  *p++ = 0x53;
                  *p++ = 0x56;
                  *p++ = 0x41; *p++ = 0x56;
                  *p++ = 0xB8; *p++ = 0x30; *p++ = 0x80; *p++ = 0x00; *p++ = 0x00;

                  // jmp rel32 → target + 9
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + 9) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // Patch 9 bytes at target: E9 NN NN NN NN + 4 nops
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 9,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90; tp[6] = 0x90; tp[7] = 0x90; tp[8] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 9,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 9);
                    s_b84c0HookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1800B84C0 hook installed: "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " a1_global=0x", reinterpret_cast<uintptr_t>(&g_b84c0_a1),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1800B84C0: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_b84c0HookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] sub_1800B84C0 hook: no trampoline alloc within ±2GB");
                  s_b84c0HookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] sub_1800B84C0 prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]), " ", uint32_t(tgt[7]), " ",
                  uint32_t(tgt[8]),
                  " — abort (expected 53 56 41 56 B8 30 80 00 00)"));
                s_b84c0HookInstalled = true;
              }
            }
            // engine.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-PropCullHook]: trampoline at sub_1801B2200
        // (engine.dll RVA 0x1B2200) — static-prop visibility gatherer.
        // Captures sceneScale (a1+0x50048) and camera (a1+0x4FFDC..+E4)
        // at function entry. These feed the prop distance-cull formula.
        //
        // Original prolog (8 bytes):
        //   48 8B C4 53 55 57 41 55
        //   = mov rax,rsp; push rbx; push rbp; push rdi; push r13
        // Replace with E9 NN NN NN NN 90 90 90 (jmp + 3 nops).
        //
        // Trampoline:
        //   48 89 0D disp32              ; mov [rip+disp32], rcx (a1)
        //   8B 81 disp32                 ; mov eax, [rcx+0x50048]   (sceneScale)
        //   89 05 disp32                 ; mov [rip+disp32], eax
        //   8B 81 disp32                 ; mov eax, [rcx+0x4FFDC]   (camX)
        //   89 05 disp32                 ; mov [rip+disp32], eax
        //   8B 81 disp32                 ; mov eax, [rcx+0x4FFE0]   (camY)
        //   89 05 disp32                 ; mov [rip+disp32], eax
        //   8B 81 disp32                 ; mov eax, [rcx+0x4FFE4]   (camZ)
        //   89 05 disp32                 ; mov [rip+disp32], eax
        //   F0 FF 05 disp32              ; lock inc dword [rip+disp32]
        //   48 8B C4 53 55 57 41 55      ; replicated prolog (8 bytes)
        //   E9 NN NN NN NN               ; jmp rel32 → target+8
        //
        // rax is volatile; the function's first instruction was `mov rax, rsp`
        // which we replicate AFTER our captures, so rax-clobber is safe.
        {
          // RE-ENABLED with plain 32-bit `mov` instead of `movss` for the
          // float reads. The previous SSE-based version crashed pre-gameplay;
          // the floats are stored as their u32 bit patterns, decoded back to
          // float at log time via memcpy.
          // v28: RE-ENABLED. v26 (entry hook on, dispatch off) was the
          // last known-working config. v27 (mfence-only at producer
          // prologue) did NOT fix the floor — proves the load-bearing
          // mechanism is more specific than a memory barrier. Likely
          // the entry hook's xadd RMW + ring stores trigger cache
          // coherency that resolves whatever race is happening. Cost
          // ~5 hits/frame × ~80 cycles = ~0.4µs/frame.
          static bool s_propCullHookInstalled = false;
          if (!tf2patches::kPatchActive<tf2patches::kHookSubB2200>)
            s_propCullHookInstalled = true;
          if (!s_propCullHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B2200;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x48 && tgt[1] == 0x8B && tgt[2] == 0xC4 &&
                  tgt[3] == 0x53 && tgt[4] == 0x55 && tgt[5] == 0x57 &&
                  tgt[6] == 0x41 && tgt[7] == 0x55) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDisp = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // Trampoline writes to ring[(head++) & 15]. Slot is 24 bytes.
                  // Plan:
                  //   push rax / rcx / rdx
                  //   atomic head++ → eax (post-inc)
                  //   slot_offset = (eax & 15) * 24  via lea+shl trick (×3 then ×8)
                  //   rax = ring_base + slot_offset
                  //   rcx (caller) is preserved (saved as 2nd push, [rsp+8])
                  //   read original rcx into rdx  (= a1 of sub_1801B2200)
                  //   *(uint64_t*)(rax + 0)  = rdx
                  //   *(float*)(rax + 8..20) = movss from [rdx + offsets]
                  //   pop rdx / rcx / rax
                  //   replicate prolog
                  //   jmp back

                  // v33 BISECT: removed `inc dword [rip+g_hitsEntryHook]`
                  // (the v17 hit counter, 6 bytes RMW). Tests whether
                  // the RMW on the counter was the load-bearing piece.

                  // v46: starting from v33-WORKING baseline, removing ONLY
                  // the `xor eax, eax + 10 nops` (12 bytes total).
                  // Everything else from v33 is preserved.
                  //
                  // v47 attempt: removed second `mov rdx, [rsp+8]` →
                  // CRASH before gameplay even starts. Toggle was
                  // logged at 0 throughout, so the gated body never
                  // ran. Either the [rsp+8] read itself has a side
                  // effect (store-forward interaction with whatever
                  // engine wrote to that stack slot) or trampoline
                  // length/alignment matters.
                  //
                  // v48 = v46 with `and eax, 0xFF` (5 bytes, reg-only)
                  // REPLACED BY 5 NOPs. Same length, same alignment,
                  // no memory access removed. If v48 crashes too →
                  // any disturbance to the post-filter block is fatal
                  // (uninvestigable by ablation). If v48 works → the
                  // and-eax is dead and we can next try replacing the
                  // second [rsp+8] read with 5 NOPs (preserving length)
                  // to isolate the read's side effect from alignment.

                  // push rax; push rcx; push rdx
                  *p++ = 0x50;
                  *p++ = 0x51;
                  *p++ = 0x52;

                  // [Filter] mov rax, [rsp+8]; mov ecx, [rax+0x5003C];
                  // and; cmp; jb skip
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0x44; *p++ = 0x24; *p++ = 0x08;
                  *p++ = 0x8B; *p++ = 0x88;
                  { int32_t off = 0x5003C; std::memcpy(p, &off, 4); p += 4; }
                  *p++ = 0x81; *p++ = 0xE1;
                  *p++ = 0xFF; *p++ = 0xFF; *p++ = 0xFF; *p++ = 0x7F;
                  *p++ = 0x81; *p++ = 0xF9;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x7A; *p++ = 0x45;
                  *p++ = 0x0F; *p++ = 0x82;
                  uint8_t* jbDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // v46: REMOVED `xor eax, eax + 10 nops` (12 bytes)

                  // v48 confirmed `and eax, 0xFF` is dead (5 NOPs in its place
                  // worked). v49 keeps that NOP-out and ALSO swaps the second
                  // `mov rdx, [rsp+8]` for 5 NOPs (preserving length). If v49
                  // works, length is what mattered for v47, not the read.
                  // If v49 crashes, the read itself has a side effect.
                  *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90;
                  // lea edx, [rax+2*rax]; shl edx, 3
                  *p++ = 0x8D; *p++ = 0x14; *p++ = 0x40;
                  *p++ = 0xC1; *p++ = 0xE2; *p++ = 0x03;
                  // v51: REPLACED `movabs rax, &g_propCullRing[0]` (10 bytes)
                  // with 10 NOPs (same length). Tests whether the 64-bit
                  // immediate load itself is load-bearing — same family as
                  // the second [rsp+8] read.
                  *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90;
                  *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90; *p++ = 0x90;
                  // add rax, rdx
                  *p++ = 0x48; *p++ = 0x01; *p++ = 0xD0;
                  // v49 confirmed: NOP-replacing the second `mov rdx,[rsp+8]`
                  // crashes before gameplay (same length, same alignment as v46).
                  // Therefore the read itself is load-bearing — likely via a
                  // store-forwarding interaction with the prior `push rcx`.
                  // Restored to keep v48-equivalent (only `and eax,0xFF` NOPed).
                  // mov rdx, [rsp+8]
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0x54; *p++ = 0x24; *p++ = 0x08;

                  // v57: emitRipDisp computes disp = addr - (p+4) which is
                  // correct only when disp32 is the last field. For
                  // `cmp [rip+disp32], imm8`, RIP-at-decode = p+5 (after the
                  // imm8 byte), so the correct disp = addr - (p+5). The
                  // bugged form read &g_forceMainViewBitmask+1 instead, and
                  // the v46 fix relied on that off-by-one address happening
                  // to be non-zero so the body would run. Now corrected;
                  // toggle init flipped to 1 so body still runs by default.
                  *p++ = 0x83; *p++ = 0x3D;
                  {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(&g_forceMainViewBitmask) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4 + 1)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }
                  *p++ = 0x00;
                  *p++ = 0x0F; *p++ = 0x84;
                  uint8_t* jzForceDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // body (gated off, never executes when toggle=0)
                  *p++ = 0x8B; *p++ = 0x8A;
                  { int32_t off = 0x54070; std::memcpy(p, &off, 4); p += 4; }
                  *p++ = 0x48; *p++ = 0x81; *p++ = 0xC1;
                  { int32_t off = 0xA811; std::memcpy(p, &off, 4); p += 4; }
                  *p++ = 0x48; *p++ = 0x8D; *p++ = 0x0C; *p++ = 0xCA;
                  *p++ = 0x48; *p++ = 0x83; *p++ = 0xCA; *p++ = 0xFF;
                  *p++ = 0x8B; *p++ = 0x05;             // mov eax, [rip+globalCount]
                  emitRipDisp((void*)(engBase + 0x7D2988));
                  *p++ = 0x83; *p++ = 0xC0; *p++ = 0x3F; // add eax, 0x3F
                  *p++ = 0xC1; *p++ = 0xE8; *p++ = 0x06; // shr eax, 6  → wordCount
                  // v57: extra `dec eax` so we skip the LAST word. The
                  // last word covers prop indices wordCount*64-63..globalCount-1
                  // plus phantom slots above globalCount. Force-filling it
                  // makes the engine try to render those non-existent props
                  // → flickering material-less geo. Skipping it costs at
                  // most the up-to-63 real props in that word, which the
                  // engine's normal logic can still handle.
                  *p++ = 0xFF; *p++ = 0xC8;             // dec eax (skip last word)
                  *p++ = 0x85; *p++ = 0xC0;             // test eax, eax
                  *p++ = 0x74; *p++ = 0x0A;             // jz done (+10)
                  *p++ = 0xFF; *p++ = 0xC8;             // loop: dec eax
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x14; *p++ = 0xC1; // mov [rcx+rax*8], rdx
                  *p++ = 0x85; *p++ = 0xC0;             // test eax, eax
                  *p++ = 0x75; *p++ = 0xF6;             // jne loop (-10)

                  // skip_force back-patch
                  {
                    const int32_t jzDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jzForceDispAddr + 4)));
                    std::memcpy(jzForceDispAddr, &jzDisp, 4);
                  }

                  // skip back-patch (filter jb)
                  {
                    const int32_t jbDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jbDispAddr + 4)));
                    std::memcpy(jbDispAddr, &jbDisp, 4);
                  }

                  // pop rdx; pop rcx; pop rax
                  *p++ = 0x5A;
                  *p++ = 0x59;
                  *p++ = 0x58;

                  // Replicated 8 bytes: mov rax,rsp; push rbx; push rbp; push rdi; push r13
                  *p++ = 0x48; *p++ = 0x8B; *p++ = 0xC4;
                  *p++ = 0x53;
                  *p++ = 0x55;
                  *p++ = 0x57;
                  *p++ = 0x41; *p++ = 0x55;

                  // jmp rel32 → target + 8
                  *p++ = 0xE9;
                  {
                    const int32_t backDisp = static_cast<int32_t>(
                      static_cast<intptr_t>(target + 8) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &backDisp, 4);
                    p += 4;
                  }

                  // Patch 8 bytes at target: E9 NN NN NN NN + 3 nops
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 8,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90; tp[6] = 0x90; tp[7] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 8,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 8);
                    s_propCullHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1801B2200 hook installed (v57: cmp off-by-one fixed, toggle=1 default, body skips last word): "
                      "target=0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " sceneScale_global=0x", reinterpret_cast<uintptr_t>(&g_propCull_sceneScale),
                      std::dec));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] sub_1801B2200: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_propCullHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] sub_1801B2200 hook: no trampoline alloc within ±2GB");
                  s_propCullHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] sub_1801B2200 prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]), " ", uint32_t(tgt[7]),
                  " — abort (expected 48 8B C4 53 55 57 41 55)"));
                s_propCullHookInstalled = true;
              }
            }
            // engine.dll not loaded yet → retry next frame
          }
        }

        // NV-DXVK [VanishDiag-PropCullDecisionHook]: trampoline at the
        // cull-jump itself (engine.dll RVA 0x1B2476, the `ja loc_1801B23E4`
        // immediately following `comiss xmm3, xmm0`). Captures per-prop cull
        // decisions filtered to main view, into a 4096-slot ring. With this
        // we can identify the floor's propIdx, its cullRadius, and the exact
        // (adj_dist², thresh) values without a visible/vanish baseline.
        //
        // Patch site: 6 bytes (`0F 87 68 FF FF FF` = `ja rel32`) → `E9 disp32
        // + 0x90` (5+1). Trampoline:
        //   pushfq; push rax/rcx/rdx
        //   seta al                 ; capture cull decision from live flags
        //   filter |[rdi+0x5003C]| > 4000.0f
        //   atomic head++
        //   write 32-byte slot
        //   pop rdx/rcx/rax; popfq
        //   ja  rel32 → 0x1B23E4    ; original cull target
        //   jmp rel32 → 0x1B247C    ; original fall-through target
        {
          // v18+: DISABLED. Was diagnostic — per-prop cull decisions into
          // a 4096-slot ring (the cull-jump trampoline at 0x1B2476). Was
          // the dominant cost: 70k hits/frame × ~80 cycles = 1.4ms/frame.
          // Confirmed sub_1801B2200's distance cull is innocent. Flip to
          // false to re-enable.
          static bool s_propCullDecisionHookInstalled = true;
          if (!s_propCullDecisionHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B2476;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x0F && tgt[1] == 0x87 &&
                  tgt[2] == 0x68 && tgt[3] == 0xFF &&
                  tgt[4] == 0xFF && tgt[5] == 0xFF) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDisp = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // [Hit counter]
                  *p++ = 0xFF; *p++ = 0x05;
                  emitRipDisp((void*)&g_hitsCullJumpHook);

                  // pushfq; push rax; push rcx; push rdx
                  *p++ = 0x9C;
                  *p++ = 0x50;
                  *p++ = 0x51;
                  *p++ = 0x52;

                  // seta al  — capture cullFlag from LIVE flags before our
                  // own cmps overwrite them.
                  *p++ = 0x0F; *p++ = 0x97; *p++ = 0xC0;

                  // Filter: |[rdi + 0x5003C]| (cam.x bits with sign cleared)
                  // must exceed 0x457A0000 (4000.0f). Skips shadow/portal
                  // dispatchers that don't have player-magnitude cam.x.
                  // mov ecx, [rdi+0x5003C]   ; 8B 8F 3C 00 05 00
                  *p++ = 0x8B; *p++ = 0x8F;
                  { int32_t off = 0x5003C; std::memcpy(p, &off, 4); p += 4; }
                  // and ecx, 0x7FFFFFFF
                  *p++ = 0x81; *p++ = 0xE1;
                  *p++ = 0xFF; *p++ = 0xFF; *p++ = 0xFF; *p++ = 0x7F;
                  // cmp ecx, 0x457A0000        (= 4000.0f)
                  *p++ = 0x81; *p++ = 0xF9;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x7A; *p++ = 0x45;
                  // jb rel32  → skip_capture (back-patched after writes)
                  *p++ = 0x0F; *p++ = 0x82;
                  uint8_t* jbDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // movzx edx, al  — preserve cullFlag in edx across the
                  // remaining flag-clobbering arithmetic.
                  *p++ = 0x0F; *p++ = 0xB6; *p++ = 0xD0;

                  // mov eax, 1
                  *p++ = 0xB8; *p++ = 0x01; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;
                  // lock xadd dword [rip+disp32], eax
                  *p++ = 0xF0; *p++ = 0x0F; *p++ = 0xC1; *p++ = 0x05;
                  emitRipDisp((void*)&g_propCullDecisionRingHead);

                  // and eax, 0xFFF  (slot index in 0..4095)
                  *p++ = 0x25; *p++ = 0xFF; *p++ = 0x0F; *p++ = 0x00; *p++ = 0x00;
                  // shl eax, 5      (×32 = slot byte offset)
                  *p++ = 0xC1; *p++ = 0xE0; *p++ = 0x05;

                  // mov rcx, &g_propCullDecisionRing[0]   (movabs imm64)
                  *p++ = 0x48; *p++ = 0xB9;
                  {
                    const uintptr_t base =
                      reinterpret_cast<uintptr_t>(&g_propCullDecisionRing[0]);
                    std::memcpy(p, &base, 8); p += 8;
                  }
                  // add rcx, rax
                  *p++ = 0x48; *p++ = 0x01; *p++ = 0xC1;

                  // mov [rcx+0], r15d   (44 89 39  -- propIdx, no disp)
                  *p++ = 0x44; *p++ = 0x89; *p++ = 0x39;

                  // mov [rcx+4], edx    (cullFlag preserved in edx)
                  *p++ = 0x89; *p++ = 0x51; *p++ = 0x04;

                  // movss [rcx+8], xmm3  (adj_dist²)
                  *p++ = 0xF3; *p++ = 0x0F; *p++ = 0x11; *p++ = 0x59; *p++ = 0x08;
                  // movss [rcx+0xC], xmm0 (thresh)
                  *p++ = 0xF3; *p++ = 0x0F; *p++ = 0x11; *p++ = 0x41; *p++ = 0x0C;

                  // Plain int copies of prop fields (no SSE state churn).
                  // edx is now free to use as scratch.
                  auto emitU32CopyRbx = [&p](int8_t srcOff, int8_t dstOff) {
                    // mov edx, [rbx+srcOff8]  ; 8B 53 disp8
                    *p++ = 0x8B; *p++ = 0x53;
                    *p++ = static_cast<uint8_t>(srcOff);
                    // mov [rcx+dstOff8], edx  ; 89 51 disp8
                    *p++ = 0x89; *p++ = 0x51;
                    *p++ = static_cast<uint8_t>(dstOff);
                  };
                  emitU32CopyRbx(0x34, 0x10);  // prop.x
                  emitU32CopyRbx(0x38, 0x14);  // prop.y
                  emitU32CopyRbx(0x3C, 0x18);  // prop.z
                  emitU32CopyRbx(0x40, 0x1C);  // prop.radius

                  // skip_capture: back-patch jb → here
                  {
                    const int32_t jbDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jbDispAddr + 4)));
                    std::memcpy(jbDispAddr, &jbDisp, 4);
                  }

                  // pop rdx; pop rcx; pop rax; popfq
                  *p++ = 0x5A;
                  *p++ = 0x59;
                  *p++ = 0x58;
                  *p++ = 0x9D;

                  // Recreate original control flow after popfq:
                  //   ja  rel32 → engBase + 0x1B23E4   (cull target)
                  //   jmp rel32 → engBase + 0x1B247C   (keep target)
                  // ja rel32: 0F 87 disp32 (6 bytes)
                  *p++ = 0x0F; *p++ = 0x87;
                  {
                    const uintptr_t cullTarget = engBase + 0x1B23E4;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(cullTarget) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }
                  // jmp rel32: E9 disp32 (5 bytes)
                  *p++ = 0xE9;
                  {
                    const uintptr_t keepTarget = engBase + 0x1B247C;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(keepTarget) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Patch 6 bytes at target: E9 disp32 + 0x90
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 6,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 6,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 6);
                    s_propCullDecisionHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] cull-jump hook installed at 0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " size=", std::dec, static_cast<uint32_t>(p - tramp), " bytes",
                      " ringSize=", kPropCullDecisionRingSize));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] cull-jump hook: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_propCullDecisionHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] cull-jump hook: no trampoline alloc within ±2GB");
                  s_propCullDecisionHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] cull-jump prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]),
                  " — abort (expected 0F 87 68 FF FF FF)"));
                s_propCullDecisionHookInstalled = true;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-BitmaskLoadHook]: hook the bitmask LOAD at
        // engine.dll RVA 0x1B23D6 (`mov rdx, [rax + r8*8]`). When the
        // global flag g_forceMainViewBitmask is set AND filter passes
        // (|cam.x| > 4000 → main view), OR rdx with -1 so every word the
        // engine reads has all 64 bits set. Bypasses any rebuild between
        // function entry and the read. Patches 9 bytes (the 4-byte load +
        // the next 5-byte `mov r10d,[rsp+0x24]`) — both replicated in the
        // trampoline.
        {
          // v25: bisect — DISABLED to test if this hook is the load-bearing
          // fix. If floor still draws → this wasn't needed.
          static bool s_bitmaskLoadHookInstalled = true;
          if (!s_bitmaskLoadHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B23D6;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              // Expected: 4A 8B 14 C0 (mov rdx,[rax+r8*8]) +
              //           44 8B 54 24 24 (mov r10d,[rsp+0x24])
              if (tgt[0] == 0x4A && tgt[1] == 0x8B && tgt[2] == 0x14 && tgt[3] == 0xC0 &&
                  tgt[4] == 0x44 && tgt[5] == 0x8B && tgt[6] == 0x54 &&
                  tgt[7] == 0x24 && tgt[8] == 0x24) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDisp = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // [Hit counter]
                  *p++ = 0xFF; *p++ = 0x05;
                  emitRipDisp((void*)&g_hitsBitmaskLoadHook);

                  // pushfq; push rax  (rcx is dead at this point per
                  // disasm — bsf rcx,rdx writes it next, no read first)
                  *p++ = 0x9C;
                  *p++ = 0x50;

                  // Replicate original load: mov rdx, [rax + r8*8]  (4 bytes)
                  *p++ = 0x4A; *p++ = 0x8B; *p++ = 0x14; *p++ = 0xC0;

                  // Filter: |[rdi+0x5003C]| > 4000.0f
                  // mov eax, [rdi+0x5003C]   ; 8B 87 + disp32 (6 bytes)
                  *p++ = 0x8B; *p++ = 0x87;
                  { int32_t off = 0x5003C; std::memcpy(p, &off, 4); p += 4; }
                  // and eax, 0x7FFFFFFF      ; 25 + imm32 (5 bytes)
                  *p++ = 0x25;
                  *p++ = 0xFF; *p++ = 0xFF; *p++ = 0xFF; *p++ = 0x7F;
                  // cmp eax, 0x457A0000      ; 3D + imm32 (5 bytes)
                  *p++ = 0x3D;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x7A; *p++ = 0x45;
                  // jb rel32 → skip_or
                  *p++ = 0x0F; *p++ = 0x82;
                  uint8_t* jbDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // cmp dword [rip+toggle_addr], 0
                  *p++ = 0x83; *p++ = 0x3D;
                  emitRipDisp((void*)&g_forceMainViewBitmask);
                  *p++ = 0x00;
                  // jz rel32 → skip_or
                  *p++ = 0x0F; *p++ = 0x84;
                  uint8_t* jzDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // Skip the OR if this is the LAST word: bits beyond the
                  // actual prop count would map to non-existent prop
                  // slots, corrupting the dispatch list. Compute
                  // last_word_idx = (count + 63) / 64 - 1 and compare to r8d.
                  // mov eax, [rip + count_addr]
                  *p++ = 0x8B; *p++ = 0x05;
                  emitRipDisp((void*)(engBase + 0x7D2988));
                  // add eax, 63        ; eax = count+63
                  *p++ = 0x83; *p++ = 0xC0; *p++ = 0x3F;
                  // shr eax, 6         ; eax = wordCount
                  *p++ = 0xC1; *p++ = 0xE8; *p++ = 0x06;
                  // dec eax            ; eax = last_word_idx
                  *p++ = 0xFF; *p++ = 0xC8;
                  // cmp eax, r8d       ; 41 39 C0
                  *p++ = 0x41; *p++ = 0x39; *p++ = 0xC0;
                  // je skip_or         ; skip on last word
                  *p++ = 0x0F; *p++ = 0x84;
                  uint8_t* jeDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // or rdx, -1   (4 bytes; rdx becomes all-ones)
                  *p++ = 0x48; *p++ = 0x83; *p++ = 0xCA; *p++ = 0xFF;

                  // skip_or: all three skip-jumps (jb filter, jz toggle,
                  // je last-word) land here.
                  {
                    const int32_t jbDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jbDispAddr + 4)));
                    std::memcpy(jbDispAddr, &jbDisp, 4);
                    const int32_t jzDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jzDispAddr + 4)));
                    std::memcpy(jzDispAddr, &jzDisp, 4);
                    const int32_t jeDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jeDispAddr + 4)));
                    std::memcpy(jeDispAddr, &jeDisp, 4);
                  }

                  // pop rax; popfq
                  *p++ = 0x58;
                  *p++ = 0x9D;

                  // Replicate eaten instruction: mov r10d, [rsp+0x24]
                  // (5 bytes; original engine rsp at this point — same
                  // since we balanced our pushes/pops)
                  *p++ = 0x44; *p++ = 0x8B; *p++ = 0x54; *p++ = 0x24; *p++ = 0x24;

                  // jmp rel32 → engBase + 0x1B23DF (after the eaten mov)
                  *p++ = 0xE9;
                  {
                    const uintptr_t back = engBase + 0x1B23DF;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(back) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Patch 9 bytes at target: E9 disp32 + nop*4
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 9,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90; tp[6] = 0x90; tp[7] = 0x90; tp[8] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 9,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 9);
                    s_bitmaskLoadHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] bitmask-load hook installed at 0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " size=", std::dec, static_cast<uint32_t>(p - tramp), " bytes",
                      " (forces rdx=-1 when main-view filter passes & toggle ON)"));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] bitmask-load hook: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_bitmaskLoadHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] bitmask-load hook: no trampoline alloc within ±2GB");
                  s_bitmaskLoadHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] bitmask-load prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]), " ", uint32_t(tgt[7]), " ", uint32_t(tgt[8]),
                  " — abort (expected 4A 8B 14 C0 44 8B 54 24 24)"));
                s_bitmaskLoadHookInstalled = true;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-DispatchHook]: DISABLED in v16 — replaced by
        // the proper memory-barrier fix at 0x1B320B (see below). The
        // per-entry hook here was a side-effect fix (latency drained the
        // store buffer) costing ~10000-50000 hits/frame. The mfence at
        // dispatcher entry runs once per call (4-8 fences/frame) and is
        // the correct long-term fix for the producer-store-visibility race.
        //
        // Code preserved (gated by an always-false static) so we can
        // re-enable for diagnostics by flipping s_dispatchHookInstalled
        // initialisation if ever needed.
        {
          // v26: bisect — DISABLED to test if entry hook alone fixes the
          // floor. If yes, dispatch hook is unnecessary.
          static bool s_dispatchHookInstalled = true;
          if (!s_dispatchHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B32ED;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x0F && tgt[1] == 0x84 &&
                  tgt[2] == 0xCB && tgt[3] == 0x02 &&
                  tgt[4] == 0x00 && tgt[5] == 0x00) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDisp = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // === v23: REVERTED to v12-exact trampoline ===
                  // The v15 lahf/sahf optimization had subtle bugs that
                  // caused crashes when capture was forced ON. v12's full
                  // pushfq/popfq + 3 register pushes + always-on ring
                  // write is the proven-working version. Cost ~80 cycles
                  // per call × ~2.5k dispatch hits/frame = 200µs/frame.
                  // Acceptable.

                  // pushfq; push rax; push rcx; push rdx
                  *p++ = 0x9C;
                  *p++ = 0x50;
                  *p++ = 0x51;
                  *p++ = 0x52;

                  // setz al; movzx edx, al — capture skipFlag (engine's ZF
                  // from `and esi,eax` immediately before the patched je).
                  *p++ = 0x0F; *p++ = 0x94; *p++ = 0xC0;
                  *p++ = 0x0F; *p++ = 0xB6; *p++ = 0xD0;

                  // Filter: |[rdi+0x5003C]| > 4000.0f
                  *p++ = 0x8B; *p++ = 0x87;
                  { int32_t off = 0x5003C; std::memcpy(p, &off, 4); p += 4; }
                  *p++ = 0x25;
                  *p++ = 0xFF; *p++ = 0xFF; *p++ = 0xFF; *p++ = 0x7F;
                  *p++ = 0x3D;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x7A; *p++ = 0x45;
                  *p++ = 0x0F; *p++ = 0x82;
                  uint8_t* jbDispAddr = p;
                  *p++ = 0x00; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;

                  // head++ (non-atomic xadd; lock dropped — race-induced
                  // overwrites in a logging ring are tolerable).
                  *p++ = 0xB8; *p++ = 0x01; *p++ = 0x00; *p++ = 0x00; *p++ = 0x00;
                  *p++ = 0x0F; *p++ = 0xC1; *p++ = 0x05;
                  emitRipDisp((void*)&g_dispatchDecisionRingHead);

                  // and eax, 0xFFF; shl eax, 5
                  *p++ = 0x25; *p++ = 0xFF; *p++ = 0x0F; *p++ = 0x00; *p++ = 0x00;
                  *p++ = 0xC1; *p++ = 0xE0; *p++ = 0x05;

                  // mov rcx, &g_dispatchDecisionRing[0]; add rcx, rax
                  *p++ = 0x48; *p++ = 0xB9;
                  {
                    const uintptr_t base =
                      reinterpret_cast<uintptr_t>(&g_dispatchDecisionRing[0]);
                    std::memcpy(p, &base, 8); p += 8;
                  }
                  *p++ = 0x48; *p++ = 0x01; *p++ = 0xC1;

                  // Slot writes: modelPtr, entryHi, entryAddr, skipFlag, camX
                  *p++ = 0x49; *p++ = 0x8B; *p++ = 0x06;          // mov rax, [r14]
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x01;          // mov [rcx], rax
                  *p++ = 0x49; *p++ = 0x8B; *p++ = 0x46; *p++ = 0x08;   // mov rax, [r14+8]
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x41; *p++ = 0x08;   // mov [rcx+8], rax
                  *p++ = 0x4C; *p++ = 0x89; *p++ = 0xF0;          // mov rax, r14
                  *p++ = 0x48; *p++ = 0x89; *p++ = 0x41; *p++ = 0x10;   // mov [rcx+16], rax
                  *p++ = 0x89; *p++ = 0x51; *p++ = 0x18;           // mov [rcx+24], edx
                  *p++ = 0x8B; *p++ = 0x87;
                  { int32_t off = 0x5003C; std::memcpy(p, &off, 4); p += 4; }
                  *p++ = 0x89; *p++ = 0x41; *p++ = 0x1C;           // mov [rcx+28], eax

                  // skip:
                  {
                    const int32_t jbDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(p) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(jbDispAddr + 4)));
                    std::memcpy(jbDispAddr, &jbDisp, 4);
                  }

                  // pop rdx; pop rcx; pop rax; popfq
                  *p++ = 0x5A;
                  *p++ = 0x59;
                  *p++ = 0x58;
                  *p++ = 0x9D;

                  // Replicate original control flow:
                  //   je rel32 → engBase + 0x1B35BE   (skip-entry path)
                  //   jmp rel32 → engBase + 0x1B32F3  (process-entry path)
                  *p++ = 0x0F; *p++ = 0x84;
                  {
                    const uintptr_t skipTarget = engBase + 0x1B35BE;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(skipTarget) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }
                  *p++ = 0xE9;
                  {
                    const uintptr_t processTarget = engBase + 0x1B32F3;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(processTarget) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Patch 6 bytes at target: E9 disp32 + 0x90
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 6,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 6,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 6);
                    s_dispatchHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] dispatch-filter hook installed at 0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " size=", std::dec, static_cast<uint32_t>(p - tramp), " bytes",
                      " ringSize=", kDispatchDecisionRingSize));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] dispatch-filter hook: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_dispatchHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] dispatch-filter hook: no trampoline alloc within ±2GB");
                  s_dispatchHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] dispatch-filter prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]),
                  " — abort (expected 0F 84 CB 02 00 00)"));
                s_dispatchHookInstalled = true;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-DispatchMFence]: PROPER LONG-TERM FIX for the
        // floor-vanish bug. Per IDA decompile of sub_1801B31E0, the
        // dispatcher syncs via JT_WaitForJobAndOnlyHelpWithJobTypes on the
        // sub_1801B2200 producer job (handle at view+0x5004C). But the JT
        // wait apparently doesn't issue a memory barrier — sub_1801B2200's
        // stores to entry[+0xD]/entry[+0xE] can still be in the producer
        // core's store buffer when the dispatcher reads them. x86's
        // StoreLoad relaxation lets the dispatcher see stale entry[+0xE],
        // causing the floor's bit to appear in the fade mask, so pass 1
        // (D & ~E) skips the floor and the entry never reaches pass 2's
        // fade-blend draws either (because v82=0 → bit was never put in E
        // by the producer). Net: floor invisible.
        //
        // mfence right after JT_WaitForJob returns drains the store
        // buffer + invalidates speculative loads on the dispatcher core,
        // forcing entry[+0xE] reads to see the producer's actual stores.
        //
        // Patch site: `83 BF 30 00 05 00 00` at 0x1B320B = `cmp dword
        // ptr [rdi+0x50030], 0` (7 bytes) → `jmp rel32 + nop*2`. The
        // trampoline does:
        //   mfence
        //   cmp dword ptr [rdi+0x50030], 0   (replicate eaten cmp)
        //   jmp 0x1B3212                       (continue to original je)
        {
          // v20+: DISABLED. Theory was that JT_WaitForJob didn't issue a
          // memory barrier and stale entry[+E] was a producer-store race.
          // Test result: mfence after the wait did NOT fix the floor.
          // The race theory was wrong — or the race is wider than just
          // post-wait visibility. Static patch at 0x1B32DF (force E = 0)
          // is the actual fix and replaces this hook.
          static bool s_dispatchMFenceHookInstalled = true;
          if (!s_dispatchMFenceHookInstalled) {
            HMODULE eng = GetModuleHandleA("engine.dll");
            if (eng != nullptr) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              const uintptr_t target  = engBase + 0x1B320B;
              const uint8_t* tgt      = reinterpret_cast<const uint8_t*>(target);
              if (tgt[0] == 0x83 && tgt[1] == 0xBF &&
                  tgt[2] == 0x30 && tgt[3] == 0x00 &&
                  tgt[4] == 0x05 && tgt[5] == 0x00 &&
                  tgt[6] == 0x00) {
                uint8_t* tramp = nullptr;
                for (intptr_t step = 0x10000; step <= 0x40000000 && tramp == nullptr;
                     step += 0x10000) {
                  void* hint = reinterpret_cast<void*>(target - step);
                  void* alloc = VirtualAlloc(hint, 4096,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                  hint = reinterpret_cast<void*>(target + step);
                  alloc = VirtualAlloc(hint, 4096,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_EXECUTE_READWRITE);
                  if (alloc != nullptr) {
                    intptr_t d = reinterpret_cast<intptr_t>(alloc) -
                                 static_cast<intptr_t>(target);
                    if (d > -0x7FFF0000 && d < 0x7FFF0000) {
                      tramp = static_cast<uint8_t*>(alloc);
                      break;
                    }
                    VirtualFree(alloc, 0, MEM_RELEASE);
                  }
                }

                if (tramp != nullptr) {
                  uint8_t* p = tramp;
                  auto emitRipDisp = [&p](void* addr) {
                    const int32_t disp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(addr) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  };

                  // [Hit counter]
                  *p++ = 0xFF; *p++ = 0x05;
                  emitRipDisp((void*)&g_hitsDispatchMFence);

                  // mfence  ; 0F AE F0  (3 bytes)
                  *p++ = 0x0F; *p++ = 0xAE; *p++ = 0xF0;

                  // Replicate eaten cmp dword ptr [rdi+0x50030], 0
                  // (83 BF 30 00 05 00 00, 7 bytes)
                  *p++ = 0x83; *p++ = 0xBF;
                  *p++ = 0x30; *p++ = 0x00; *p++ = 0x05; *p++ = 0x00;
                  *p++ = 0x00;

                  // jmp rel32 → engBase + 0x1B3212 (the original `je`)
                  *p++ = 0xE9;
                  {
                    const uintptr_t back = engBase + 0x1B3212;
                    const int32_t disp = static_cast<int32_t>(
                      static_cast<intptr_t>(back) -
                      static_cast<intptr_t>(reinterpret_cast<uintptr_t>(p + 4)));
                    std::memcpy(p, &disp, 4); p += 4;
                  }

                  // Patch 7 bytes at target: E9 disp32 + nop*2
                  DWORD oldProtect = 0;
                  if (VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                     PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    uint8_t* tp = reinterpret_cast<uint8_t*>(target);
                    tp[0] = 0xE9;
                    const int32_t fwdDisp = static_cast<int32_t>(
                      reinterpret_cast<intptr_t>(tramp) -
                      static_cast<intptr_t>(target + 5));
                    std::memcpy(tp + 1, &fwdDisp, 4);
                    tp[5] = 0x90; tp[6] = 0x90;
                    DWORD tmp = 0;
                    VirtualProtect(reinterpret_cast<LPVOID>(target), 7,
                                   oldProtect, &tmp);
                    FlushInstructionCache(GetCurrentProcess(),
                                          reinterpret_cast<LPVOID>(target), 7);
                    s_dispatchMFenceHookInstalled = true;
                    Logger::warn(str::format(
                      "[TF2Probe] dispatch-mfence hook installed at 0x", std::hex, target,
                      " trampoline=0x", reinterpret_cast<uintptr_t>(tramp),
                      " size=", std::dec, static_cast<uint32_t>(p - tramp), " bytes",
                      " — closes the JT-wait store-visibility race (1 mfence/dispatcher call)"));
                  } else {
                    Logger::warn(str::format(
                      "[TF2Probe] dispatch-mfence hook: VirtualProtect failed at 0x",
                      std::hex, target, std::dec));
                    VirtualFree(tramp, 0, MEM_RELEASE);
                    s_dispatchMFenceHookInstalled = true;
                  }
                } else {
                  Logger::warn("[TF2Probe] dispatch-mfence hook: no trampoline alloc within ±2GB");
                  s_dispatchMFenceHookInstalled = true;
                }
              } else {
                Logger::warn(str::format(
                  "[TF2Probe] dispatch-mfence prolog mismatch at 0x",
                  std::hex, target, std::dec,
                  " — bytes ", uint32_t(tgt[0]), " ", uint32_t(tgt[1]), " ",
                  uint32_t(tgt[2]), " ", uint32_t(tgt[3]), " ",
                  uint32_t(tgt[4]), " ", uint32_t(tgt[5]), " ",
                  uint32_t(tgt[6]),
                  " — abort (expected 83 BF 30 00 05 00 00)"));
                s_dispatchMFenceHookInstalled = true;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-F9Gate]: edge-trigger on F9 to dump VanishDiag
        // logs ONLY for the frame F9 was pressed. The OR-accumulators and
        // histograms always reset each frame (so they reflect the current
        // frame's state); only the Logger::warn emissions are gated. This
        // keeps the diagnostic noise out of the log unless explicitly
        // requested. Use F9 alongside F11 (scene_dump) for paired captures.
        // [VanishDiag] Trigger key changed F9 -> P. TF2 captures F9 (default
        // screenshot bind in Source) and the keypress never reached our
        // GetAsyncKeyState in the previous run. 'P' has no default Titanfall
        // bind. Variable names left as F9-* for minimal diff; semantics are
        // "vanish-diag trigger key down/edge".
        static bool s_vanishDiagF9Latch = false;
        const bool vanishDiagF9Down =
          (GetAsyncKeyState('P') & 0x8000) != 0;
        const bool vanishDiagF9Edge =
          vanishDiagF9Down && !s_vanishDiagF9Latch;
        s_vanishDiagF9Latch = vanishDiagF9Down;

        // [VanishDiag-ForceBitmask] Home toggles g_forceMainViewBitmask.
        // O and Insert were both grabbed somewhere (no edge reached us);
        // Home/End-cluster keys per user direction. Edge-triggered same as P.
        {
          static bool s_forceBitmaskKeyLatch = false;
          const bool keyDown = (GetAsyncKeyState(VK_HOME) & 0x8000) != 0;
          const bool keyEdge = keyDown && !s_forceBitmaskKeyLatch;
          s_forceBitmaskKeyLatch = keyDown;
          if (keyEdge) {
            g_forceMainViewBitmask = (g_forceMainViewBitmask == 0) ? 1u : 0u;
            // Sample engine.dll's global static-prop count so we know how
            // many words the trampoline will fill on the next call.
            uint32_t globalCount = 0;
            uint32_t wordCount = 0;
            if (HMODULE eng = GetModuleHandleA("engine.dll")) {
              const uintptr_t engBase = reinterpret_cast<uintptr_t>(eng);
              globalCount = *reinterpret_cast<const uint32_t*>(engBase + 0x7D2988);
              wordCount = (globalCount + 63) >> 6;
              // No cap — trust the engine's globalCount.
            }
            Logger::warn(str::format(
              "[VanishDiag-ForceBitmask] toggled to ",
              (g_forceMainViewBitmask ? "ON" : "OFF"),
              " — main-view bitmask force-fill is now ",
              (g_forceMainViewBitmask ? "active" : "inactive"),
              " (engine globalCount=", globalCount,
              ", wordCount=", wordCount, " — no cap)"));
          }
        }

        // [VanishDiag-ForceBitmaskWatch] 1 Hz heartbeat of the toggle's
        // current value. v47 attempt crashed when we removed the second
        // `mov rdx, [rsp+8]`, which is only reachable past the cmp/jz
        // gate IF g_forceMainViewBitmask != 0. Log it so we can confirm
        // the toggle isn't sneakily on (or being flipped by something).
        {
          using clk = std::chrono::steady_clock;
          static auto s_lastForceWatch = clk::now();
          const auto now = clk::now();
          const auto sinceMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - s_lastForceWatch).count();
          if (sinceMs >= 1000) {
            s_lastForceWatch = now;
            Logger::warn(str::format(
              "[VanishDiag-ForceBitmaskWatch] g_forceMainViewBitmask=",
              g_forceMainViewBitmask,
              " (gated body executes when != 0)"));
          }
        }

        // [VanishDiag-DispatchCapture] End toggles g_dispatchCaptureEnabled.
        // OFF by default — the dispatch trampoline's ring-write path is
        // bypassed, leaving only the timing-fix latency. Press End to
        // enable capture for the next auto-dump cycle.
        {
          static bool s_dispatchCaptureKeyLatch = false;
          const bool keyDown = (GetAsyncKeyState(VK_END) & 0x8000) != 0;
          const bool keyEdge = keyDown && !s_dispatchCaptureKeyLatch;
          s_dispatchCaptureKeyLatch = keyDown;
          if (keyEdge) {
            g_dispatchCaptureEnabled = (g_dispatchCaptureEnabled == 0) ? 1u : 0u;
            Logger::warn(str::format(
              "[VanishDiag-DispatchCapture] toggled to ",
              (g_dispatchCaptureEnabled ? "ON" : "OFF"),
              " — dispatch ring-write block is now ",
              (g_dispatchCaptureEnabled ? "active" : "bypassed")));
          }
        }

        // [VanishDiag-AutoDump] auto-trigger the heartbeat dump every 300
        // frames. Also tracks frame-time stats and hit-counter deltas so we
        // can diagnose perf complaints — logs shown in the auto-dump path.
        using clk = std::chrono::steady_clock;
        static auto s_lastEndFrameTime = clk::now();
        static uint64_t s_frameTimeSumNs = 0;
        static uint64_t s_frameTimeMaxNs = 0;
        static uint32_t s_frameTimeSamples = 0;
        static uint32_t s_hitsEntryPrev   = 0;
        static uint32_t s_hitsCullPrev    = 0;
        static uint32_t s_hitsBitmaskPrev = 0;
        static uint32_t s_hitsMFencePrev  = 0;
        {
          const auto now = clk::now();
          const auto deltaNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - s_lastEndFrameTime).count();
          s_lastEndFrameTime = now;
          s_frameTimeSumNs += static_cast<uint64_t>(deltaNs);
          if (static_cast<uint64_t>(deltaNs) > s_frameTimeMaxNs)
            s_frameTimeMaxNs = static_cast<uint64_t>(deltaNs);
          ++s_frameTimeSamples;
        }
        // v18+: only the LIGHT [Perf] line auto-fires (one log line every
        // 5 sec, ~50µs disk I/O). Full ring dumps only fire on P press
        // (manual diagnostic). Removes the dump-induced 1-2s stalls that
        // were dragging the 1 fps frametime even further.
        static auto s_lastPerfLineTime = clk::now();
        bool perfLineThisFrame = false;
        {
          const auto now = clk::now();
          const auto sincePerf = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - s_lastPerfLineTime).count();
          if (sincePerf >= 5000) {
            s_lastPerfLineTime = now;
            perfLineThisFrame = true;
          }
        }
        const bool dumpTrigger = vanishDiagF9Edge;

        // NV-DXVK [VanishDiag-F9Heartbeat]: log a one-line marker on EVERY P
        // edge OR auto-dump tick. Counts presses, dumps the PropCull ring
        // unconditionally so we get a snapshot every press (no dependence on
        // g_vanishDiagCapturedA2 etc.).
        // Light [Perf] line — fires every 5 sec independent of full dump.
        // One log line, ~50µs. Doesn't reset frame timer if a fullDump
        // happens in the same frame (the dump path resets it).
        if (perfLineThisFrame) {
          const uint32_t hitsEntryNow   = g_hitsEntryHook;
          const uint32_t hitsCullNow    = g_hitsCullJumpHook;
          const uint32_t hitsBitmaskNow = g_hitsBitmaskLoadHook;
          const uint32_t hitsMFenceNow  = g_hitsDispatchMFence;
          const uint32_t dEntry   = hitsEntryNow   - s_hitsEntryPrev;
          const uint32_t dCull    = hitsCullNow    - s_hitsCullPrev;
          const uint32_t dBitmask = hitsBitmaskNow - s_hitsBitmaskPrev;
          const uint32_t dMFence  = hitsMFenceNow  - s_hitsMFencePrev;
          s_hitsEntryPrev   = hitsEntryNow;
          s_hitsCullPrev    = hitsCullNow;
          s_hitsBitmaskPrev = hitsBitmaskNow;
          s_hitsMFencePrev  = hitsMFenceNow;
          const uint64_t avgUs = s_frameTimeSamples
            ? (s_frameTimeSumNs / s_frameTimeSamples / 1000u)
            : 0u;
          const uint64_t maxUs = s_frameTimeMaxNs / 1000u;
          const uint32_t fps   = (avgUs > 0) ? uint32_t(1000000ULL / avgUs) : 0u;
          Logger::warn(str::format(
            "[Perf] window samples=", s_frameTimeSamples,
            " avg=", avgUs, "us (~", fps, " fps)",
            " max=", maxUs, "us",
            " | hooks_per_window: entry=", dEntry,
            " cullJump=", dCull,
            " bitmaskLoad=", dBitmask,
            " dispatchMFence=", dMFence,
            " | per_frame: entry=",
            (s_frameTimeSamples ? dEntry / s_frameTimeSamples : 0u),
            " cull=", (s_frameTimeSamples ? dCull / s_frameTimeSamples : 0u),
            " bitmask=", (s_frameTimeSamples ? dBitmask / s_frameTimeSamples : 0u),
            " mfence=", (s_frameTimeSamples ? dMFence / s_frameTimeSamples : 0u)));
          s_frameTimeSumNs   = 0;
          s_frameTimeMaxNs   = 0;
          s_frameTimeSamples = 0;
        }

        if (dumpTrigger) {
          static uint32_t s_f9PressCount = 0;
          ++s_f9PressCount;
          const uint32_t devFrameForF9 =
            (m_context != nullptr && m_context->m_device != nullptr)
              ? m_context->m_device->getCurrentFrameId()
              : 0u;
          Logger::warn(str::format(
            "[VanishDiag-F9Heartbeat] press#", s_f9PressCount,
            " frame=", devFrameForF9,
            " trigger=P — edge detected, dumping PropCull ring unconditionally below"));

          // Always-on PropCull ring dump (mirrors the gated dump below; runs
          // even when other VanishDiag scaffolding is unprimed).
          const uint32_t headSnapHB = g_propCullRingHead;
          Logger::warn(str::format(
            "[VanishDiag-PropCullRing-HB] press#", s_f9PressCount,
            " frame=", devFrameForF9,
            " head=", headSnapHB,
            " (next slot is head&", kPropCullRingMask, "=",
            (headSnapHB & kPropCullRingMask), ")"));
          // Skip empty slots so the log isn't drowned in zero rows when the
          // ring hasn't filled. Empty = a1==0 AND sceneScale-bits==0.
          for (uint32_t i = 0; i < kPropCullRingSize; ++i) {
            const uint64_t a1HB = g_propCullRing[i].a1;
            uint32_t ssBitsHB;
            std::memcpy(&ssBitsHB, const_cast<float*>(&g_propCullRing[i].sceneScale), 4);
            if (a1HB == 0 && ssBitsHB == 0) continue;
            float ssHB, cxHB, cyHB, czHB;
            std::memcpy(&ssHB, const_cast<float*>(&g_propCullRing[i].sceneScale), 4);
            std::memcpy(&cxHB, const_cast<float*>(&g_propCullRing[i].camX), 4);
            std::memcpy(&cyHB, const_cast<float*>(&g_propCullRing[i].camY), 4);
            std::memcpy(&czHB, const_cast<float*>(&g_propCullRing[i].camZ), 4);
            Logger::warn(str::format(
              "[VanishDiag-PropCullRing-HB]   slot=", i,
              " a1=0x", std::hex, uint64_t(a1HB),
              " sceneScale=", std::dec, ssHB,
              " cam=(", cxHB, ", ", cyHB, ", ", czHB, ")"));
          }

          // [VanishDiag-PropCullDecision-HB] dump unique-propIdx decisions
          // captured by the cull-jump trampoline. Dedups by propIdx so we
          // get one line per prop, with its most-recent (cullFlag, dist,
          // thresh, pos, radius) data. Floor candidates per v3 handoff:
          // prop.y ∈ [598, 877], prop.z ≈ 32. Slots with propY in roughly
          // that band are flagged in the log so they're easy to grep.
          {
            const uint32_t headDS = g_propCullDecisionRingHead;
            uint32_t totalNonEmpty = 0;
            uint32_t totalCulled   = 0;
            // Two-pass: first pass counts; second pass dedups + emits.
            // Dedup uses a tiny linear-probe hash over a 1024-bucket array
            // of propIdx → ring index. Since unique propIdx count is bounded
            // by main-view's prop list (~hundreds), 1024 buckets is enough.
            constexpr uint32_t kHashBuckets = 1024;
            uint32_t bucketIdx[kHashBuckets];
            uint32_t bucketSlot[kHashBuckets];
            for (uint32_t b = 0; b < kHashBuckets; ++b) {
              bucketIdx[b] = 0xFFFFFFFFu;
              bucketSlot[b] = 0xFFFFFFFFu;
            }
            for (uint32_t i = 0; i < kPropCullDecisionRingSize; ++i) {
              const uint32_t pidx = g_propCullDecisionRing[i].propIdx;
              const uint32_t cf   = g_propCullDecisionRing[i].cullFlag;
              uint32_t prxBits;
              std::memcpy(&prxBits, const_cast<float*>(&g_propCullDecisionRing[i].propX), 4);
              if (pidx == 0 && cf == 0 && prxBits == 0) continue;
              ++totalNonEmpty;
              if (cf) ++totalCulled;
              uint32_t b = (pidx * 2654435761u) & (kHashBuckets - 1);
              for (uint32_t probe = 0; probe < kHashBuckets; ++probe) {
                const uint32_t bb = (b + probe) & (kHashBuckets - 1);
                if (bucketIdx[bb] == 0xFFFFFFFFu) {
                  bucketIdx[bb]  = pidx;
                  bucketSlot[bb] = i;   // first-seen wins; we'll overwrite below
                  break;
                }
                if (bucketIdx[bb] == pidx) {
                  bucketSlot[bb] = i;   // most-recent wins
                  break;
                }
              }
            }
            Logger::warn(str::format(
              "[VanishDiag-PropCullDecision-HB] press#", s_f9PressCount,
              " frame=", devFrameForF9,
              " head=", headDS,
              " nonEmpty=", totalNonEmpty,
              " culled=", totalCulled,
              " (",  (totalNonEmpty ? (100u * totalCulled / totalNonEmpty) : 0u),
              "%) ringSize=", kPropCullDecisionRingSize));
            uint32_t emitted = 0;
            for (uint32_t b = 0; b < kHashBuckets; ++b) {
              if (bucketIdx[b] == 0xFFFFFFFFu) continue;
              const uint32_t i = bucketSlot[b];
              const uint32_t pidx = g_propCullDecisionRing[i].propIdx;
              const uint32_t cf   = g_propCullDecisionRing[i].cullFlag;
              float adsq, thr, px, py, pz, pr;
              std::memcpy(&adsq, const_cast<float*>(&g_propCullDecisionRing[i].adjDistSq), 4);
              std::memcpy(&thr,  const_cast<float*>(&g_propCullDecisionRing[i].thresh),    4);
              std::memcpy(&px,   const_cast<float*>(&g_propCullDecisionRing[i].propX),     4);
              std::memcpy(&py,   const_cast<float*>(&g_propCullDecisionRing[i].propY),     4);
              std::memcpy(&pz,   const_cast<float*>(&g_propCullDecisionRing[i].propZ),     4);
              std::memcpy(&pr,   const_cast<float*>(&g_propCullDecisionRing[i].propRadius),4);
              const bool nearFloor = (py >= 400.0f && py <= 1000.0f &&
                                      pz >= -100.0f && pz <= 200.0f);
              const float distApprox = (adsq > 0.0f) ? std::sqrt(adsq) : 0.0f;
              const float threshApprox = (thr  > 0.0f) ? std::sqrt(thr)  : 0.0f;
              Logger::warn(str::format(
                "[VanishDiag-PropCullDecision-HB]   ", (nearFloor ? "FLOOR? " : "       "),
                "propIdx=", pidx,
                " cull=", cf,
                " pos=(", px, ",", py, ",", pz, ")",
                " radius=", pr,
                " ~dist=", distApprox,
                " ~thresh=", threshApprox,
                " adj_dist2=", adsq,
                " thresh=", thr));
              if (++emitted >= 512) {
                Logger::warn("[VanishDiag-PropCullDecision-HB]   ... (truncated at 512 unique propIdx)");
                break;
              }
            }
          }

          // [VanishDiag-DispatchDecision-HB] dump dispatcher per-entry
          // decisions, deduped by entry's modelPtr (entry+0). One line
          // per unique modelPtr with most-recent skipFlag, raw bytes,
          // and entryAddr so we can correlate to view+0x8028 offsets.
          {
            const uint32_t headDD = g_dispatchDecisionRingHead;
            uint32_t totalNonEmptyDD = 0;
            uint32_t totalSkipped    = 0;
            constexpr uint32_t kBuckets = 1024;
            uint64_t bucketKey[kBuckets];
            uint32_t bucketSlot[kBuckets];
            for (uint32_t b = 0; b < kBuckets; ++b) {
              bucketKey[b] = 0ULL;
              bucketSlot[b] = 0xFFFFFFFFu;
            }
            for (uint32_t i = 0; i < kDispatchDecisionRingSize; ++i) {
              const uint64_t mp   = g_dispatchDecisionRing[i].modelPtr;
              const uint64_t hi   = g_dispatchDecisionRing[i].entryHi;
              const uint64_t addr = g_dispatchDecisionRing[i].entryAddr;
              const uint32_t sk   = g_dispatchDecisionRing[i].skipFlag;
              if (mp == 0 && hi == 0 && addr == 0) continue;
              ++totalNonEmptyDD;
              if (sk) ++totalSkipped;
              const uint64_t key = mp ? mp : addr;
              uint32_t b = uint32_t((key * 11400714819323198485ULL) >> 32) &
                           (kBuckets - 1);
              for (uint32_t probe = 0; probe < kBuckets; ++probe) {
                const uint32_t bb = (b + probe) & (kBuckets - 1);
                if (bucketKey[bb] == 0ULL && bucketSlot[bb] == 0xFFFFFFFFu) {
                  bucketKey[bb]  = key;
                  bucketSlot[bb] = i;
                  break;
                }
                if (bucketKey[bb] == key) {
                  bucketSlot[bb] = i;
                  break;
                }
              }
            }
            Logger::warn(str::format(
              "[VanishDiag-DispatchDecision-HB] press#", s_f9PressCount,
              " frame=", devFrameForF9,
              " head=", headDD,
              " nonEmpty=", totalNonEmptyDD,
              " skipped=", totalSkipped,
              " (", (totalNonEmptyDD ? (100u * totalSkipped / totalNonEmptyDD) : 0u),
              "%) ringSize=", kDispatchDecisionRingSize));
            uint32_t emittedDD = 0;
            for (uint32_t b = 0; b < kBuckets; ++b) {
              if (bucketSlot[b] == 0xFFFFFFFFu) continue;
              const uint32_t i = bucketSlot[b];
              const auto& s = g_dispatchDecisionRing[i];
              // Decode entry bytes 8..F as 4 bytes + 2 words (matches
              // disassembled access pattern in sub_1801B31E0).
              const uint8_t bC = uint8_t((s.entryHi >> 32) & 0xFF);
              const uint8_t bD = uint8_t((s.entryHi >> 40) & 0xFF);
              const uint8_t bE = uint8_t((s.entryHi >> 48) & 0xFF);
              const uint8_t bF = uint8_t((s.entryHi >> 56) & 0xFF);
              const uint16_t w8 = uint16_t(s.entryHi & 0xFFFF);
              const uint16_t wA = uint16_t((s.entryHi >> 16) & 0xFFFF);
              float camX;
              std::memcpy(&camX, const_cast<float*>(&s.viewCamX), 4);
              Logger::warn(str::format(
                "[VanishDiag-DispatchDecision-HB]   ",
                (s.skipFlag ? "SKIP " : "draw "),
                "modelPtr=0x", std::hex, s.modelPtr,
                " entryAddr=0x", s.entryAddr,
                " w8=0x", w8, " wA=0x", wA,
                " bC=0x", uint32_t(bC),
                " bD=0x", uint32_t(bD),
                " bE=0x", uint32_t(bE),
                " bF=0x", uint32_t(bF),
                std::dec, " camX=", camX));
              if (++emittedDD >= 256) {
                Logger::warn("[VanishDiag-DispatchDecision-HB]   ... (truncated at 256 unique entries)");
                break;
              }
            }
          }
        }

        // NV-DXVK [VanishDiag-WorldVis]: per-frame snapshot of the bucket
        // bitmask captured from R_DrawWorldMeshes. Logs only when F9 edge
        // is detected this frame; resets snapshots each frame regardless.
        if (g_vanishDiagCapturedA2 != 0) {
          const uintptr_t a2 = static_cast<uintptr_t>(g_vanishDiagCapturedA2);
          // Bitmask data comes from the trampoline-captured SNAPSHOT
          // (g_vanishDiagBitmaskSnap), NOT from `[a2+0x54088]` directly:
          // a2's allocation may be freed/reused between R_DrawWorldMeshes
          // returning and EndFrame logging — reading [a2+0x54088] then
          // sees zeros (verified by previous test). The trampoline's
          // rep movsq snapshots the live bitmask at the call site so we
          // always have valid data here.
          uint32_t totalBits = 0;
          for (int i = 0; i < 8; ++i) {
            totalBits += static_cast<uint32_t>(__popcnt64(g_vanishDiagBitmaskSnap[i]));
          }
          const uint32_t devFrame =
            (m_context != nullptr && m_context->m_device != nullptr)
              ? m_context->m_device->getCurrentFrameId()
              : 0u;
          if (vanishDiagF9Edge) {
            Logger::warn(str::format(
              "[VanishDiag-WorldVis] frame=", devFrame,
              " a2=0x", std::hex, a2,
              " a3=0x", uint32_t(g_vanishDiagCapturedA3),
              " bitsSet=", std::dec, totalBits,
              " w[0..7]=0x", std::hex, g_vanishDiagBitmaskSnap[0],
              " 0x", g_vanishDiagBitmaskSnap[1],
              " 0x", g_vanishDiagBitmaskSnap[2],
              " 0x", g_vanishDiagBitmaskSnap[3],
              " 0x", g_vanishDiagBitmaskSnap[4],
              " 0x", g_vanishDiagBitmaskSnap[5],
              " 0x", g_vanishDiagBitmaskSnap[6],
              " 0x", g_vanishDiagBitmaskSnap[7], std::dec));
            // BuildWorldMeshBatches output snapshot — same WriterStruct,
            // captured by the same trampoline. If passEnds or batchCount
            // differ between visible and vanish, sub_1800B6FB0 dropped
            // buckets even with the bitmask the same.
            Logger::warn(str::format(
              "[VanishDiag-BuildBatches] frame=", devFrame,
              " passEnds=[", g_buildBatchesPassEnds[0],
              ",", g_buildBatchesPassEnds[1],
              ",", g_buildBatchesPassEnds[2],
              ",", g_buildBatchesPassEnds[3],
              "] batchCount=", g_buildBatchesBatchCount));

            // sub_1800B84C0 input snapshot — last call's args + range
            const uint32_t b84RangeStart = g_b84c0_range_start;
            const uint32_t b84RangeEnd   = g_b84c0_range_end;
            const int32_t  b84RangeSize  = static_cast<int32_t>(b84RangeEnd - b84RangeStart);
            Logger::warn(str::format(
              "[VanishDiag-B84C0] frame=", devFrame,
              " calls=", uint32_t(g_b84c0_call_count),
              " a1=0x", std::hex, uint64_t(g_b84c0_a1),
              " mask=0x", uint32_t(g_b84c0_filter_mask),
              " pass=", std::dec, uint32_t(g_b84c0_pass_idx),
              " start=", b84RangeStart,
              " end=", b84RangeEnd,
              " size=", b84RangeSize));
            g_b84c0_call_count = 0;

            // PropCull ring buffer (256 slots, last 256 main-view-filtered
            // sub_1801B2200 calls). The slot whose cam=(x,y,z) matches the
            // player position is the main-view dispatcher; sceneScale,
            // camX/Y/Z in that slot are the actual cull-formula inputs.
            const uint32_t headSnap = g_propCullRingHead;
            Logger::warn(str::format(
              "[VanishDiag-PropCullRing] frame=", devFrame,
              " head=", headSnap,
              " (next slot is head&", kPropCullRingMask, "=",
              (headSnap & kPropCullRingMask), ")"));
            for (uint32_t i = 0; i < kPropCullRingSize; ++i) {
              const uint64_t a1Slot = g_propCullRing[i].a1;
              uint32_t ssBits;
              std::memcpy(&ssBits, const_cast<float*>(&g_propCullRing[i].sceneScale), 4);
              if (a1Slot == 0 && ssBits == 0) continue;
              float ss, cx, cy, cz;
              std::memcpy(&ss, const_cast<float*>(&g_propCullRing[i].sceneScale), 4);
              std::memcpy(&cx, const_cast<float*>(&g_propCullRing[i].camX), 4);
              std::memcpy(&cy, const_cast<float*>(&g_propCullRing[i].camY), 4);
              std::memcpy(&cz, const_cast<float*>(&g_propCullRing[i].camZ), 4);
              Logger::warn(str::format(
                "[VanishDiag-PropCullRing]   slot=", i,
                " a1=0x", std::hex, uint64_t(a1Slot),
                " sceneScale=", std::dec, ss,
                " cam=(", cx, ", ", cy, ", ", cz, ")"));
            }

            // [VanishDiag-Stack]: dump captured stack frames for the
            // floor's three target VS hashes. Each stack-trace frame is
            // an absolute address; we resolve to module+RVA so the user
            // can paste each RVA directly into IDA. After dump, reset
            // the slots so the NEXT F9 cycle captures fresh data from
            // that frame.
            HMODULE eng = GetModuleHandleA("engine.dll");
            HMODULE cli = GetModuleHandleA("client.dll");
            const uint64_t engBase = eng ? reinterpret_cast<uint64_t>(eng) : 0;
            const uint64_t cliBase = cli ? reinterpret_cast<uint64_t>(cli) : 0;
            // Conservative module-size assumption (engine.dll is ~300MB,
            // client.dll ~50MB). 0x40000000 = 1GB upper bound covers both.
            constexpr uint64_t kModRangeMax = 0x40000000ULL;
            Logger::warn(str::format(
              "[VanishDiag-Stack] frame=", devFrame,
              " totalHits=", uint32_t(g_vanishStackTotalHits),
              " (engine.dll@0x", std::hex, engBase,
              " client.dll@0x", cliBase, std::dec, ")"));
            for (int i = 0; i < 3; ++i) {
              if (g_vanishStack[i].frameCount == 0) {
                Logger::warn(str::format(
                  "[VanishDiag-Stack]   slot=", i,
                  " VS=0x", std::hex, k_VanishStackTargets[i], std::dec,
                  " — NOT CAPTURED THIS CYCLE (engine didn't draw with this VS)"));
                continue;
              }
              std::string framesStr;
              framesStr.reserve(512);
              for (uint32_t k = 0; k < g_vanishStack[i].frameCount; ++k) {
                const uint64_t addr = g_vanishStack[i].frames[k];
                const char* mod = "?";
                uint64_t rva = addr;
                if (engBase && addr >= engBase && addr < engBase + kModRangeMax) {
                  mod = "eng"; rva = addr - engBase;
                } else if (cliBase && addr >= cliBase && addr < cliBase + kModRangeMax) {
                  mod = "cli"; rva = addr - cliBase;
                }
                if (!framesStr.empty()) framesStr += " | ";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%s+0x%llx", mod,
                              static_cast<unsigned long long>(rva));
                framesStr += buf;
              }
              Logger::warn(str::format(
                "[VanishDiag-Stack]   slot=", i,
                " VS=0x", std::hex, g_vanishStack[i].vsHash, std::dec,
                " frames=", g_vanishStack[i].frameCount,
                " stack: ", framesStr));
              // Reset slot for next F9 cycle.
              g_vanishStack[i].vsHash = 0;
              g_vanishStack[i].frameCount = 0;
              for (int k = 0; k < 16; ++k) g_vanishStack[i].frames[k] = 0;
            }
          }

          // Parallel: log qword_192205120 (global "bucket dirty" bitmask
          // ORed by sub_1800B45D0). Compare g[6] vs w[6] — if equal, the
          // per-view bitmask is aliased to the global and sub_1800B45D0 is
          // the writer. If they differ on bit 17 of word 6 (bucket 401),
          // the cull path is independent.
          uint32_t globalBits = 0;
          for (int i = 0; i < 8; ++i) {
            globalBits += static_cast<uint32_t>(__popcnt64(g_vanishDiagGlobalSnap[i]));
          }
          if (vanishDiagF9Edge) {
            Logger::warn(str::format(
              "[VanishDiag-GlobalSnap] frame=", devFrame,
              " bitsSet=", std::dec, globalBits,
              " g[0..7]=0x", std::hex, g_vanishDiagGlobalSnap[0],
              " 0x", g_vanishDiagGlobalSnap[1],
              " 0x", g_vanishDiagGlobalSnap[2],
              " 0x", g_vanishDiagGlobalSnap[3],
              " 0x", g_vanishDiagGlobalSnap[4],
              " 0x", g_vanishDiagGlobalSnap[5],
              " 0x", g_vanishDiagGlobalSnap[6],
              " 0x", g_vanishDiagGlobalSnap[7], std::dec));
          }

          // Bucket histogram: report hits for index 401 (the floor bucket
          // identified by the bit-17 toggle in word 6), total OR calls
          // across all bucket indices this frame, and number of distinct
          // bucket indices touched. Diff visible vs vanish frames:
          //   - bucket 401 hit on visible but not vanish → sub_1800B45D0
          //     is the gating writer (its conditional skipped this index).
          //   - bucket 401 hit on BOTH → cull is downstream of this OR.
          //   - bucket 401 NEVER hit → BVH traversal in sub_1800B48E0
          //     never reached this leaf; gating is upstream.
          uint32_t hist401   = g_vanishDiagBucketHist[401];
          uint32_t totalCalls = 0;
          uint32_t nonzeroBuckets = 0;
          for (int i = 0; i < 1024; ++i) {
            const uint32_t v = g_vanishDiagBucketHist[i];
            totalCalls += v;
            if (v) ++nonzeroBuckets;
          }
          if (vanishDiagF9Edge) {
            Logger::warn(str::format(
              "[VanishDiag-BucketHist] frame=", devFrame,
              " hits[401]=", hist401,
              " totalCalls=", totalCalls,
              " nonzeroBuckets=", nonzeroBuckets));
          }

          // Reset all snapshots so next frame's OR-accumulators start clean.
          for (int i = 0; i < 8; ++i) {
            g_vanishDiagBitmaskSnap[i] = 0;
            g_vanishDiagGlobalSnap[i] = 0;
          }
          for (int i = 0; i < 1024; ++i) {
            g_vanishDiagBucketHist[i] = 0;
          }
        }

        // NV-DXVK [VanishDiag-B30Hook]: per-frame log of the LAST sub_18036BD30
        // call's source_bitmask. Last-fire-wins single slot; across many frames
        // we'll see different views (shadow=small ~13 bits, main view ~67 bits).
        // Grep the log for high bitsSet entries to find the main view's source.
        if (g_b30_source_bm != 0) {
          uint64_t snap[8];
          const uint64_t srcAddr  = g_b30_source_bm;
          const uint64_t viewCtx  = g_b30_view_ctx;
          const uint32_t calls    = g_b30_call_count;
          for (int i = 0; i < 8; ++i) snap[i] = g_b30_snap[i];
          uint32_t totalBits = 0;
          for (int i = 0; i < 8; ++i) {
            totalBits += static_cast<uint32_t>(__popcnt64(snap[i]));
          }
          const uint32_t devFrameB30 =
            (m_context != nullptr && m_context->m_device != nullptr)
              ? m_context->m_device->getCurrentFrameId()
              : 0u;
          if (vanishDiagF9Edge) {
            Logger::warn(str::format(
              "[VanishDiag-B30Hook] frame=", devFrameB30,
              " calls=", calls,
              " src=0x", std::hex, srcAddr,
              " view=0x", viewCtx,
              " bitsSet=", std::dec, totalBits,
              " w[0..7]=0x", std::hex, snap[0],
              " 0x", snap[1],
              " 0x", snap[2],
              " 0x", snap[3],
              " 0x", snap[4],
              " 0x", snap[5],
              " 0x", snap[6],
              " 0x", snap[7], std::dec));
          }
          // Reset call counter each frame (snap left as-is so we still see
          // the last capture even on idle frames where no call fired).
          g_b30_call_count = 0;
        }

        // NV-DXVK [VanishDiag-EB290]: per-frame report of sub_1802EB290
        // results, cross-correlated with the WorldVis bitmask:
        //   - For each bucket where hist[i] > 0 AND WorldVis bit i is CLEAR,
        //     sub_1802EB290 was called and returned 0 → REJECTED bucket.
        //   - These are the buckets the per-bucket visibility test culled.
        // We dump up to MAX_REJ rejected indices per frame to keep log
        // volume bounded.
        if (g_eb290_call_count > 0) {
          uint32_t total = g_eb290_call_count;
          uint32_t nonzero = 0;
          uint32_t rejected = 0;
          uint32_t passed = 0;
          // Snapshot WorldVis bitmask for the inBitmask check. (Already
          // captured by the R_DrawWorldMeshes trampoline; may be stale
          // by a frame relative to eb290 calls, but for diff diagnostic
          // that's tolerable.)
          uint64_t worldVis[8];
          for (int i = 0; i < 8; ++i) worldVis[i] = g_vanishDiagBitmaskSnap[i];

          // Build a comma-separated list of up to MAX_REJ rejected indices.
          constexpr int MAX_REJ = 20;
          std::string rejList;
          rejList.reserve(256);
          for (int i = 0; i < 2048; ++i) {
            const uint32_t v = g_eb290_hist[i];
            if (!v) continue;
            ++nonzero;
            // Is bit i set in the WorldVis bitmask?
            const int word = i >> 6;
            const int bit  = i & 63;
            const bool inMask =
              (word < 8) && ((worldVis[word] & (1ULL << bit)) != 0);
            if (inMask) {
              ++passed;
            } else {
              ++rejected;
              if (rejected <= MAX_REJ) {
                if (!rejList.empty()) rejList += ",";
                rejList += std::to_string(i);
                rejList += "@";
                rejList += std::to_string(v);
              }
            }
          }
          const uint32_t devFrameEb =
            (m_context != nullptr && m_context->m_device != nullptr)
              ? m_context->m_device->getCurrentFrameId()
              : 0u;
          if (vanishDiagF9Edge) {
            Logger::warn(str::format(
              "[VanishDiag-EB290] frame=", devFrameEb,
              " totalCalls=", total,
              " distinct=", nonzero,
              " passed=", passed,
              " rejected=", rejected,
              " rej[0..", MAX_REJ, "]=[", rejList, "]"));
          }

          // Reset for next frame.
          for (int i = 0; i < 2048; ++i) g_eb290_hist[i] = 0;
          g_eb290_call_count = 0;
        }

        // Per-frame snapshot. Logs EVERY frame (game runs at ~1 FPS in this
        // build so throttling drops good-period samples we need to compare
        // against). Tag GOOD vs CLIFF vs TRANSITION so the user can grep.
        static bool s_wasCliffLastFrame = false;
        const bool isCliff    = (rawDeficit || capDeficit || anyDeficit);
        const bool transition = (isCliff != s_wasCliffLastFrame);
        s_wasCliffLastFrame = isCliff;
        if (s_msdx11 != nullptr) {
          const uintptr_t base = reinterpret_cast<uintptr_t>(s_msdx11);
          const int32_t  mode       = *reinterpret_cast<const int32_t*>  (base + 0x1BBCBB4);
          const int32_t  bias       = *reinterpret_cast<const int32_t*>  (base + 0x1BBCBB8);
          const float    bucketBias = *reinterpret_cast<const float*>    (base + 0x1BBCBBC);
          const float    distScale  = *reinterpret_cast<const float*>    (base + 0x1BBCBC0);
          const float    camX       = *reinterpret_cast<const float*>    (base + 0x1BBCC00);
          const float    camY       = *reinterpret_cast<const float*>    (base + 0x1BBCC04);
          const float    camZ       = *reinterpret_cast<const float*>    (base + 0x1BBCC08);
          const float    scaleA     = *reinterpret_cast<const float*>    (base + 0x1BBCC0C);
          const float    scaleB     = *reinterpret_cast<const float*>    (base + 0x1BBCC10);
          const uint32_t engineFid  = *reinterpret_cast<const uint32_t*> (base + 0x1BBCBF8);
          const uint32_t linked     = *reinterpret_cast<const uint32_t*> (base + 0x1BBCBD8);
          const uint32_t loadedMips = *reinterpret_cast<const uint32_t*> (base + 0x1BBCBDC);
          const uint32_t totalMips  = *reinterpret_cast<const uint32_t*> (base + 0x1BBCBE0);
          const uint64_t loadedKB   = (*reinterpret_cast<const uint64_t*>(base + 0x1BBCBF0)) >> 10;
          const uint64_t totalKB    = (*reinterpret_cast<const uint64_t*>(base + 0x1BBCBE8)) >> 10;
          const char* tag = transition ? "ENTER" : (isCliff ? "CLIFF" : "GOOD");
          Logger::warn(str::format(
            "[TF2Probe.", tag, "] mode=", mode, " bias=", bias,
            " bucketBias=", bucketBias, " distScale=", distScale,
            " cam=(", camX, ",", camY, ",", camZ, ")",
            " scaleA=", scaleA, " scaleB=", scaleB,
            " engineFid=", engineFid,
            " linkedTex=", linked,
            " mipLoaded=", loadedMips, "/", totalMips,
            " kbLoaded=", loadedKB, "/", totalKB));
        }
      }

      // NV-DXVK [ClientProbe]: client.dll convar pokes for the floor-vanish
      // investigation. Same pattern as [TF2Probe] but targets client.dll.
      // Source-style ConVars store the parsed value at struct+0x58 (m_fValue,
      // float) and struct+0x5C (m_nValue, int). GetInt() and GetBool() read
      // m_nValue, GetFloat() reads m_fValue, so we write BOTH on every poke.
      //
      // RVAs from IDA on client.dll (image base 0x180000000):
      //   pvs_extraCull              struct=0x1748BB0  (default "1")
      //   pvs_drawPortals            struct=0x1748A90  (default "0")
      //   r_farz                     struct=0x216FAA0  (default "-1" = use fog ctrl)
      //   model_defaultFadeDistMin   struct=0xFB3270   (default "400" world units)
      //   model_defaultFadeDistScale struct=0xFB31B0   (default "40")
      //
      // Env-var overrides (set before launching TF2):
      //   RTX_TF2_PVS_EXTRACULL=0           force off the client-side PVS cull
      //   RTX_TF2_PVS_DRAWPORTALS=1         draw portal vis debug
      //   RTX_TF2_FARZ=99999                push far clip far away
      //   RTX_TF2_FADE_DIST_MIN=99999       disable static-prop fade min
      //   RTX_TF2_FADE_DIST_SCALE=9999      disable static-prop fade scale
      // Forces are applied once at first frame after the dll is found, then
      // re-asserted every frame (in case the engine re-parses the convar
      // string for any reason).
      {
        static HMODULE s_client = nullptr;
        static bool    s_clientInited = false;
        struct CvOverride {
          const char* env;
          uintptr_t   structRva;
          const char* lbl;
          bool        haveValue;
          float       fv;
          int32_t     iv;
        };
        // Static so values latch once at first read of the env var.
        static CvOverride s_overrides[] = {
          // REVERTED to env-var-only. The earlier hardcodes were chasing a
          // cull that turned out not to exist — the floor "vanish" is a
          // shading bug (pathCode=1 cone-iso fallback in surface_interaction
          // .slangh), confirmed by scene_dump showing 0 pathCode=255 pixels
          // and 155k pathCode=1 pixels concentrated on a single texIdx.
          // Convar overrides left available via env var so we can re-enable
          // selectively if needed for future investigation.
          { "RTX_TF2_PVS_EXTRACULL",       0x1748BB0, "pvs_extraCull",              false, 0.f, 0 },
          { "RTX_TF2_PVS_YIELD",           0x11FBE30, "pvs_yield",                  false, 0.f, 0 },
          { "RTX_TF2_PVS_WORK_THRESHOLD",  0x1748C40, "pvs_addWorkItemsThreshold",  false, 0.f, 0 },
          { "RTX_TF2_PVS_DEBUG",           0x1748A00, "pvs_debug",                  false, 0.f, 0 },
          { "RTX_TF2_PVS_DRAWPORTALS",     0x1748A90, "pvs_drawPortals",            false, 0.f, 0 },
          { "RTX_TF2_FARZ",                0x216FAA0, "r_farz",                     false, 0.f, 0 },
          { "RTX_TF2_FADE_DIST_MIN",       0x0FB3270, "model_defaultFadeDistMin",   false, 0.f, 0 },
          { "RTX_TF2_FADE_DIST_SCALE",     0x0FB31B0, "model_defaultFadeDistScale", false, 0.f, 0 },
          { "RTX_TF2_DRAW_ALL_RENDERABLES", 0xEA9D40, "r_drawallrenderables",       false, 0.f, 0 },
        };
        constexpr uintptr_t kFValueOff = 0x58;
        constexpr uintptr_t kNValueOff = 0x5C;

        if (!s_clientInited) {
          s_clientInited = true;
          s_client = GetModuleHandleA("client.dll");
          if (s_client != nullptr) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(s_client);
            Logger::warn(str::format(
              "[ClientProbe] client.dll @ 0x", std::hex, base, std::dec));
            // Read env vars once, latch parsed values, log what we'll force.
            for (auto& ov : s_overrides) {
              const char* v = std::getenv(ov.env);
              if (v == nullptr) continue;
              ov.fv = std::strtof(v, nullptr);
              ov.iv = static_cast<int32_t>(ov.fv);
              ov.haveValue = true;
              Logger::warn(str::format(
                "[ClientProbe] FORCE ", ov.lbl, " -> ", ov.fv, " (int ", ov.iv, ")"));
            }
          } else {
            Logger::warn("[ClientProbe] client.dll NOT FOUND (will retry next frame)");
            s_clientInited = false;  // retry: client.dll may load late
          }
        }

        // Re-assert every frame. Cheap and defends against any engine code
        // that might re-parse a ConVar's string value back into m_nValue /
        // m_fValue (e.g. via a Set("...") call).
        if (s_client != nullptr) {
          const uintptr_t base = reinterpret_cast<uintptr_t>(s_client);
          for (const auto& ov : s_overrides) {
            if (!ov.haveValue) continue;
            *reinterpret_cast<float*  >(base + ov.structRva + kFValueOff) = ov.fv;
            *reinterpret_cast<int32_t*>(base + ov.structRva + kNValueOff) = ov.iv;
          }

          // Per-frame readout of the live values so we can confirm the
          // overrides are sticking (and see what defaults look like when
          // not overridden).
          static bool sLoggedClientValues = false;
          static uint32_t sFrameSinceProbe = 0;
          ++sFrameSinceProbe;
          // Log once at startup, then every 240 frames so the file doesn't
          // grow without bound.
          if (!sLoggedClientValues || (sFrameSinceProbe % 240u) == 0u) {
            sLoggedClientValues = true;
            for (const auto& ov : s_overrides) {
              const float curF = *reinterpret_cast<const float*  >(base + ov.structRva + kFValueOff);
              const int32_t curI = *reinterpret_cast<const int32_t*>(base + ov.structRva + kNValueOff);
              Logger::warn(str::format(
                "[ClientProbe] ", ov.lbl, " m_fValue=", curF, " m_nValue=", curI,
                ov.haveValue ? " (FORCED)" : ""));
            }
          }
        }
      }
    }

    // NV-DXVK: arm/finalize the scene dumper.
    SceneDump::armOnFirstGameplayFrame(raw);

    // NV-DXVK NPC SKINNING DIAG: per-frame summary for the targeted
    // bone-matrix buffer (selected via RTX_NPC_BONE_BUF env var). Counts
    // writes (copyBuffer + updateBuffer) and interleaver-read dispatches
    // observed during the previous frame, then resets. Only emits when
    // the env var is set, so output stays quiet by default. Use this to
    // see whether the NPC bone buffer is actually getting per-frame
    // animation updates.
    if (::dxvk::tf2::boneDiagEnabled()) {
      static const uintptr_t sEnvTargetBufPtr = []() -> uintptr_t {
        const char* env = std::getenv("RTX_NPC_BONE_BUF");
        if (env == nullptr) return 0u;
        return static_cast<uintptr_t>(std::strtoull(env, nullptr, 16));
      }();
      const uintptr_t sTargetBufPtr = sEnvTargetBufPtr != 0u
        ? sEnvTargetBufPtr
        : ::dxvk::tf2::g_autoTargetBufPtr.load(std::memory_order_relaxed);
      const uint32_t fnum = ::dxvk::tf2::g_frameCounterForTarget.fetch_add(
        1, std::memory_order_relaxed);
      if (sTargetBufPtr != 0u) {
        const uint32_t cb = ::dxvk::tf2::g_targetCopyBufThisFrame.exchange(
          0, std::memory_order_relaxed);
        const uint32_t ub = ::dxvk::tf2::g_targetUpdateBufThisFrame.exchange(
          0, std::memory_order_relaxed);
        const uint32_t cbBytes = ::dxvk::tf2::g_targetCopyBufBytesThisFrame.exchange(
          0, std::memory_order_relaxed);
        const uint32_t ubBytes = ::dxvk::tf2::g_targetUpdateBufBytesThisFrame.exchange(
          0, std::memory_order_relaxed);
        const uint32_t reads = ::dxvk::tf2::g_targetReadDispatchesThisFrame.exchange(
          0, std::memory_order_relaxed);
        Logger::info(str::format(
          "[BoneTargetFrame] f=", fnum,
          " bufPtr=", sTargetBufPtr,
          " copyBufWrites=", cb, " bytes=", cbBytes,
          " updateBufWrites=", ub, " bytes=", ubBytes,
          " bonesUpdated~=", ((cbBytes + ubBytes) / 48u),
          " interleaverReads=", reads));
      }
      // Per-buffer per-frame summary across ALL bone-matrix buffers
      // touched this frame. One [BonePerFrameByBuf] line per buffer.
      // Snapshot under lock then clear, so frames are independent.
      std::unordered_map<uintptr_t, ::dxvk::tf2::PerBufFrameStats> snapshot;
      {
        std::lock_guard<std::mutex> lk(::dxvk::tf2::g_perBufStatsMutex);
        snapshot.swap(::dxvk::tf2::g_perBufStats);
      }
      for (const auto& kv : snapshot) {
        const auto& s = kv.second;
        const uint32_t totalBytes = s.copyBytes + s.updateBytes;
        const uint32_t totalWrites = s.copyWrites + s.updateWrites;
        const uint32_t sharedTotal = s.copySharedRot + s.updateSharedRot;
        Logger::info(str::format(
          "[BonePerFrameByBuf] f=", fnum,
          " bufPtr=", kv.first,
          " writes=", totalWrites,
          " (cb=", s.copyWrites, ",ub=", s.updateWrites, ")",
          " bytes=", totalBytes,
          " bones~=", (totalBytes / 48u),
          " sharedRot=", sharedTotal, "/", totalWrites,
          " reads=", s.reads));
      }

      // NV-DXVK NPC SKINNING DIAG: chronological timeline of writes and
      // reads against the targeted buffer for this frame. Shows, in order,
      // every copyBuffer / updateBuffer write and every interleaver
      // dispatch that sampled the buffer. Lets us confirm whether the
      // actual bone palette a VS reads contained real-rig or filler data
      // at the moment of the draw. Example of what to look for:
      //    seq=.. W off=29184 size=2928 bones=61 sharedRot=0 src=cb
      //    seq=.. W off=29184 size=336  bones=7  sharedRot=1 src=ub
      //    seq=.. R off=29184 len=7584                       src=disp
      // -> filler updateBuf LANDED AFTER the real-rig copyBuf and BEFORE
      //    the dispatch read. That's the stomp.
      std::vector<::dxvk::tf2::BoneOp> timeline;
      {
        std::lock_guard<std::mutex> lk(::dxvk::tf2::g_boneTimelineMutex);
        timeline.swap(::dxvk::tf2::g_boneTimeline);
      }
      if (!timeline.empty()) {
        Logger::info(str::format(
          "[BoneTimeline] f=", fnum,
          " bufPtr=", sTargetBufPtr,
          " events=", timeline.size()));
        static const char* kSrc[] = {"?","cb","ub","disp"};
        for (const auto& op : timeline) {
          const char* src = (op.source < 4) ? kSrc[op.source] : "?";
          Logger::info(str::format(
            "[BoneTimeline]   seq=", op.seq,
            " ", op.op,
            " off=", op.offset,
            " palette=", (op.offset / 768u),
            " size=", op.size,
            " bones=", op.bones,
            " sharedRot=", static_cast<uint32_t>(op.sharedRot),
            " src=", src));
        }
      }

      // NV-DXVK NPC SKINNING DIAG: per-frame SRV cross-check. For each
      // unique (buf, VS, FirstElement, NumElements) observed by
      // [BoneSrvs], ask: does the SRV's bone-range contain ANY real-rig
      // write (sharedRot=0 src=cb) this frame? If yes → the draw sees
      // animated bones for at least part of its palette window. If no →
      // the entire range the shader will sample is filler → T-pose.
      // The answer is only authoritative for the TARGETED buffer because
      // the timeline is collected only for that buffer; other buffers
      // emit drawCount and firstPalette for manual correlation.
      std::vector<::dxvk::tf2::BoneSrvRecord> srvs;
      {
        std::lock_guard<std::mutex> lk(::dxvk::tf2::g_boneSrvsMutex);
        srvs.swap(::dxvk::tf2::g_boneSrvsThisFrame);
      }
      if (!srvs.empty()) {
        std::sort(srvs.begin(), srvs.end(),
          [](const ::dxvk::tf2::BoneSrvRecord& a,
             const ::dxvk::tf2::BoneSrvRecord& b) {
            if (a.bufPtr != b.bufPtr) return a.bufPtr < b.bufPtr;
            return a.drawCount > b.drawCount;
          });
        Logger::info(str::format(
          "[BoneSrvCheck] f=", fnum, " unique=", srvs.size(),
          " target=", sTargetBufPtr));
        for (const auto& r : srvs) {
          const uint32_t rangeLo = r.firstElem * 48u;
          const uint32_t rangeHi = (r.firstElem + r.numElem) * 48u;
          const uint32_t palette = r.firstElem / 16u;
          const bool isTarget    = (r.bufPtr == sTargetBufPtr);
          uint32_t cbHits = 0;     // real-rig writes overlapping SRV range
          uint32_t cbFillerHits = 0;
          uint32_t ubHits = 0;     // per-palette filler updates in range
          int32_t  firstCbPalette = -1;
          if (isTarget) {
            for (const auto& op : timeline) {
              if (op.op != 'W') continue;
              const uint32_t opLo = op.offset;
              const uint32_t opHi = op.offset + op.size;
              if (opHi <= rangeLo || opLo >= rangeHi) continue; // no overlap
              if (op.source == 1u) { // copyBuf
                if (op.sharedRot == 0u) {
                  ++cbHits;
                  if (firstCbPalette < 0)
                    firstCbPalette = static_cast<int32_t>(opLo / 768u);
                } else {
                  ++cbFillerHits;
                }
              } else if (op.source == 2u) { // updateBuf
                ++ubHits;
              }
            }
          }
          Logger::info(str::format(
            "[BoneSrvCheck]   vs=", r.vsShort,
            " buf=", r.bufPtr,
            " first=", r.firstElem, " palette=", palette, " num=", r.numElem,
            " sameBuf=", static_cast<uint32_t>(r.sameBuf),
            " draws=", r.drawCount,
            " target=", (isTarget ? 1 : 0),
            " realRigCB=", cbHits,
            " fillerCB=", cbFillerHits,
            " fillerUB=", ubHits,
            " firstRealPalette=", firstCbPalette,
            " verdict=",
            (!isTarget ? "?(untargeted)" :
             (cbHits > 0 ? "REAL_RIG" : "FILLER_ONLY"))));
        }
      }
    }

    // NV-DXVK ([BoneCacheSweep]): once per frame, scan m_fullBoneCache
    // (populated from all UpdateSubresource writes to t30) and count how
    // many of the 8192 bone slots are zero. Critical: separately count
    // zeros in "lower half" (idx & 0x8 == 0) vs "upper half" (idx & 0x8 != 0)
    // of every 16-bone palette. If upperZeros >> lowerZeros → game only
    // writes lower halves via UpdateSubresource and upper halves are filled
    // by some other path (e.g. CopyResource) we're not seeing here.
    // Gated on gameplay (raw > 50) + throttled to once per ~60 frames.
    // NV-DXVK TF2: merge the DXVK-level CopyBuffer bone mirror into
    // m_fullBoneCache so the sweep below reflects the TRUE state seen
    // by the GPU (both UpdateSubresource-written lower slots AND
    // CopyBuffer-written full-rig palettes).
    {
      std::lock_guard<std::mutex> lk(::dxvk::tf2::g_boneCacheMirrorMutex);
      if (::dxvk::tf2::g_boneCacheMirrorPopulated
          && ::dxvk::tf2::g_boneCacheMirror.size() == 393216) {
        if (m_fullBoneCache.size() != 393216) {
          m_fullBoneCache.resize(393216, 0);
        }
        const uint8_t* mirror = ::dxvk::tf2::g_boneCacheMirror.data();
        uint8_t* dst = m_fullBoneCache.data();
        for (size_t i = 0; i < 393216; i += 48) {
          bool mirrorNonZero = false;
          for (size_t b = 0; b < 48; ++b) {
            if (mirror[i + b] != 0) { mirrorNonZero = true; break; }
          }
          if (mirrorNonZero) {
            std::memcpy(dst + i, mirror + i, 48);
          }
        }
        m_hasFullBoneCache = true;
      }
    }
    if (::dxvk::tf2::boneDiagEnabled()
        && m_hasFullBoneCache && m_fullBoneCache.size() >= 48 && raw > 50) {
      static uint32_t sLastSweepFrame = 0;
      const uint32_t fid = m_context->m_device->getCurrentFrameId();
      if (fid - sLastSweepFrame >= 60u) {
        sLastSweepFrame = fid;
        const uint32_t nBones = static_cast<uint32_t>(m_fullBoneCache.size() / 48u);
        uint32_t zeroLower = 0, zeroUpper = 0;
        uint32_t nonZeroLower = 0, nonZeroUpper = 0;
        // Sample the first 10 zero-slot indices we hit.
        uint32_t firstZeros[10] = {};
        uint32_t firstZerosCount = 0;
        for (uint32_t b = 0; b < nBones; ++b) {
          const float* m = reinterpret_cast<const float*>(
              m_fullBoneCache.data() + b * 48);
          // Treat zero matrix as: |r0.xyz| + |T.xyz| < 1e-6.
          const float mag = std::fabs(m[0]) + std::fabs(m[1]) + std::fabs(m[2])
                          + std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]);
          const bool isZero = mag < 1e-6f;
          const bool isUpper = (b & 0x8u) != 0u;
          if (isZero) {
            if (isUpper) ++zeroUpper; else ++zeroLower;
            if (firstZerosCount < 10) firstZeros[firstZerosCount++] = b;
          } else {
            if (isUpper) ++nonZeroUpper; else ++nonZeroLower;
          }
        }
        std::string firstZerosStr;
        for (uint32_t i = 0; i < firstZerosCount; ++i)
          firstZerosStr += str::format(i ? "," : "", firstZeros[i]);
        Logger::info(str::format(
          "[BoneCacheSweep] f=", fid,
          " nBones=", nBones,
          " zeroLower=", zeroLower, " zeroUpper=", zeroUpper,
          " nonZeroLower=", nonZeroLower, " nonZeroUpper=", nonZeroUpper,
          " firstZeros=[", firstZerosStr, "]"));
      }
    }

    // NV-DXVK: dump key rtx.conf options once we hit a real gameplay frame so
    // we can verify the config file is actually being read.
    {
      static bool sCfgLogged = false;
      if (!sCfgLogged && raw > 50) {
        sCfgLogged = true;
        Logger::info(str::format("[D3D11Rtx] rtx.conf state at first gameplay frame:"));
        Logger::info(str::format("  rtx.pointInstancer.enable = ",
          RtxPointInstancerSystem::enable() ? "True" : "False"));
        Logger::info(str::format("  rtx.pointInstancer.cullingRadius = ",
          RtxPointInstancerSystem::cullingRadius()));
        Logger::info(str::format("  rtx.legacyMaterial.albedoConstant = (",
          LegacyMaterialDefaults::albedoConstant().x, ",",
          LegacyMaterialDefaults::albedoConstant().y, ",",
          LegacyMaterialDefaults::albedoConstant().z, ")"));
        Logger::info(str::format("  rtx.legacyMaterial.useAlbedoTextureIfPresent = ",
          LegacyMaterialDefaults::useAlbedoTextureIfPresent() ? "True" : "False"));
        Logger::info(str::format("  rtx.legacyMaterial.emissiveIntensity = ",
          LegacyMaterialDefaults::emissiveIntensity()));
        // NV-DXVK: gloss/spec workflow is hardcoded ON in slang for this
        // fork (rtx.conf bypassed). Reported here just so the dump still
        // documents material-pipe behavior at first gameplay frame.
        Logger::info(str::format("  [hardcoded] gloss->roughness inversion = ON"));
        Logger::info(str::format("  [hardcoded] spec->metallic+albedo reprojection = ON"));
        Logger::info(str::format("  rtx.debugView.debugViewIdx = ",
          DebugView::debugViewIdx()));
      }
    }
    if (SceneDump::g_obj.is_open() && !SceneDump::g_done) {
      // Closes after the dump frame ends.
      SceneDump::close();
    }
    // Latch: once we see a gameplay-scale frame, log details for N frames.
    {
      static bool s_latched = false;
      if (!s_latched && raw > 50) {
        s_latched = true;
        s_GameplayLogFrames = 5;
      } else if (s_GameplayLogFrames > 0) {
        --s_GameplayLogFrames;
      }
    }
    // NV-DXVK: Per-frame dump throttle. On loading screens running at
    // 400+ FPS the EndFrame stats block (this line + per-filter + per-o2w
    // path + per-VS breakdown below) was generating ~15 log lines per
    // present → ~6000 log lines per second on the CS thread. Every line
    // is a mutex-guarded file write; the aggregate I/O stall visibly
    // locked the loading screen.  Dump the block every 64 presents in
    // steady state; always dump on the first 8 gameplay frames and
    // whenever the "gameplay latch" (s_GameplayLogFrames > 0) fires so
    // we don't lose the "first real frame" diagnostic. The actual
    // per-frame state reset + CS-thread injectRTX emission happen
    // unconditionally below — only the Logger::info calls are gated.
    static uint64_t sEndFrameDumpCount = 0;
    const uint64_t currentDump = sEndFrameDumpCount++;
    const bool detailedDump =
         s_GameplayLogFrames > 0
      || currentDump < 8
      || (currentDump & 0x3F) == 0;

    if (detailedDump) Logger::info(str::format("[D3D11Rtx] EndFrame: draws=", draws,
      " raw=", raw,
      " backbuffer=", backbuffer != nullptr ? 1 : 0));

    // NV-DXVK: diagnostic — if draws were issued but all filtered out,
    // dump the per-filter rejection counts so we know exactly which
    // SubmitDraw pre-filter is killing the game's main-menu draws.
    // Whole block is gated on detailedDump — see per-frame throttle note.
    if (detailedDump && raw > 0 && draws < raw) {
      Logger::info(str::format("[D3D11Rtx]   filters:",
        " throttle=",       m_filterCounts[static_cast<uint32_t>(FilterReason::Throttle)],
        " nonTriTopo=",     m_filterCounts[static_cast<uint32_t>(FilterReason::NonTriTopology)],
        " noPS=",           m_filterCounts[static_cast<uint32_t>(FilterReason::NoPixelShader)],
        " noRTV=",          m_filterCounts[static_cast<uint32_t>(FilterReason::NoRenderTarget)],
        " count<3=",        m_filterCounts[static_cast<uint32_t>(FilterReason::CountTooSmall)],
        " fsQuad=",         m_filterCounts[static_cast<uint32_t>(FilterReason::FullscreenQuad)],
        " noLayout=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoInputLayout)],
        " noSemantics=",    m_filterCounts[static_cast<uint32_t>(FilterReason::NoSemantics)],
        " noPos=",          m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosition)],
        " pos2D=",          m_filterCounts[static_cast<uint32_t>(FilterReason::Position2D)],
        " noPosBuf=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosBuffer)],
        " noIdxBuf=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoIndexBuffer)],
        " hashFail=",       m_filterCounts[static_cast<uint32_t>(FilterReason::HashFailed)],
        " uiFallback=",     m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)],
        " unsupFmt=",       m_filterCounts[static_cast<uint32_t>(FilterReason::UnsupPosFmt)]));

      // NV-DXVK: o2w path histogram (which code path set objectToWorld per
      // committed draw). 0 = never set (identity), 1 = non-inst BSP t31,
      // 2 = t30 CPU Bone, 3 = t30 Bone-slice, 4 = CB3→O2W, 5 = RDEF,
      // 6 = trySourceFloat3x4, 7 = tryWorldCb, 8 = cb2@4 fallback,
      // 9 = per-instance (fanout), 10 = bone-instanced identity.
      Logger::info(str::format("[D3D11Rtx]   o2wPaths:",
        " identity=", m_o2wPathCounts[0],
        " t31=",      m_o2wPathCounts[1],
        " t30cpu=",   m_o2wPathCounts[2],
        " t30slice=", m_o2wPathCounts[3],
        " cb3=",      m_o2wPathCounts[4],
        " rdef=",     m_o2wPathCounts[5],
        " sf3x4=",    m_o2wPathCounts[6],
        " worldcb=",  m_o2wPathCounts[7],
        " cb2cam=",   m_o2wPathCounts[8],
        " fanout=",   m_o2wPathCounts[9],
        " boneInst=", m_o2wPathCounts[10],
        " skinnedChar=", m_o2wPathCounts[11]));

      // Per-VS o2w path breakdown — which shader took which path.
      // Sort by total draws desc so the noisiest shaders appear first.
      if (!m_vsO2wPathCounts.empty()) {
        std::vector<std::pair<std::string, std::array<uint32_t, 16>>> sv;
        sv.reserve(m_vsO2wPathCounts.size());
        for (auto& kv : m_vsO2wPathCounts) sv.push_back(kv);
        std::sort(sv.begin(), sv.end(), [](const auto& a, const auto& b) {
          uint32_t at = 0, bt = 0;
          for (uint32_t v : a.second) at += v;
          for (uint32_t v : b.second) bt += v;
          return at > bt;
        });
        Logger::info(str::format("[D3D11Rtx]   o2wPathsByVS (", sv.size(), " unique):"));
        for (const auto& kv : sv) {
          const auto& a = kv.second;
          uint32_t tot = 0; for (uint32_t v : a) tot += v;
          if (tot == 0) continue;
          std::string line = str::format("    ", kv.first, " n=", tot);
          static const char* kName[12] = {
            "id", "t31", "t30cpu", "t30slice", "cb3", "rdef", "sf3x4",
            "worldcb", "cb2cam", "fanout", "boneInst", "skinnedChar"
          };
          for (uint32_t p = 0; p < 12; ++p) {
            if (a[p] > 0) line += str::format(" ", kName[p], "=", a[p]);
          }
          Logger::info(line);
        }
      }
    }
    for (int i = 0; i < 16; ++i) m_o2wPathCounts[i] = 0;
    m_vsO2wPathCounts.clear();
    // NV-DXVK: per-VS outcome dump — each VS hash, #submits, #rejects per filter.
    // NV-DXVK NPC SKINNING DIAG: per-frame per-VS outcome dump, expanded.
    // For each VS hash seen this frame, log: total draws observed, how many
    // remix submitted through its full pipeline, how many were rejected
    // (by reason), and our new skinning/bone-SRV classification counters.
    // Gated on RTX_BONE_DIAG so the build stays quiet by default; uncapped
    // so you can diff successive frames while NPCs are on-screen.
    if (detailedDump
        && ::dxvk::tf2::boneDiagEnabled()
        && raw > 20 && !m_vsFrameStats.empty()) {
      static const char* kReasonName[] = {
        "Throttle","NonTriTopo","NoPS","NoRTV","TooSmall","FsQuad","NoLayout",
        "NoSem","NoPos","Pos2D","NoPosBuf","NoIdxBuf","HashFail","UIFallback","UnsupFmt"
      };
      const uint32_t fid = m_context->m_device->getCurrentFrameId();
      Logger::info(str::format("[VSHashFrame] f=", fid,
        " unique=", m_vsFrameStats.size()));
      std::vector<std::pair<std::string, VsFrameStats>> sorted;
      sorted.reserve(m_vsFrameStats.size());
      for (const auto& kv : m_vsFrameStats) sorted.push_back(kv);
      std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) {
          return a.second.seen > b.second.seen;
        });
      for (const auto& kv : sorted) {
        const auto& st = kv.second;
        uint32_t rejectTotal = 0;
        for (uint32_t r : st.rejects) rejectTotal += r;
        std::string line = str::format(
          "[VSHashFrame]   vs=", kv.first.substr(0, 19),
          " ps=", (st.firstPsHash.empty() ? "-" : st.firstPsHash),
          " seen=", st.seen,
          " subm=", st.submitted,
          " rej=", rejectTotal,
          " skV=", st.skinnedPerVert,
          " skI=", st.skinnedPerInst,
          " t30=", st.boneSrvBound,
          " t31=", st.modelInstBound);
        for (uint32_t r = 0; r < static_cast<uint32_t>(FilterReason::Count); ++r) {
          if (st.rejects[r] > 0)
            line += str::format(" ", kReasonName[r], "=", st.rejects[r]);
        }
        Logger::info(line);
      }
    }
    m_vsFrameStats.clear();

    for (uint32_t i = 0; i < static_cast<uint32_t>(FilterReason::Count); ++i)
      m_filterCounts[i] = 0;

    // NV-DXVK: Per-frame bone instancing summary
    // Rotate ring buffer: clear the slot we're about to reuse next frame.
    // Each slot holds transforms from 4 frames ago. Scene manager has
    // definitely finished with them.
    ++m_boneInstFrameId;
    if (!m_boneTransformRing.empty()) {
      uint32_t nextSlot = m_boneInstFrameId % 4;
      m_boneTransformRing[nextSlot].clear();
    }

    if (detailedDump && m_boneInstBatches > 0) {
      uint32_t ringSize = 0;
      for (const auto& slot : m_boneTransformRing) ringSize += static_cast<uint32_t>(slot.size());
      Logger::info(str::format(
        "[D3D11Rtx] BoneInst: batches=", m_boneInstBatches,
        " instances=", m_boneInstTotal,
        " uniqueVB=", m_boneInstVbPtrs.size(),
        " ringEntries=", ringSize));
    }
    m_boneInstVbPtrs.clear();
    m_boneInstBatches = 0;
    m_boneInstTotal = 0;
    m_boneInstSkipped = 0;
    m_boneInstNoCache = 0;
    m_boneInstCacheHits = 0;
    m_boneInstCacheMisses = 0;

    // NV-DXVK [SpawnGeomDiag]: per-frame BSP/world-geometry census. One line
    // per frame whenever we either took *any* fanout-related action OR while
    // the gameplay-log latch is open. Lets us spot frames where:
    //   - publishes happened but batches=0 (every fanout's tforms list was
    //     dropped on the floor)
    //   - rejects > 0 with publishes=0 (camera frame never latched)
    //   - blindProbes > 0 with batches=0 (BSP draws fell into the
    //     static-mesh skip-attach branch instead of the t31 fanout branch)
    //   - mirrorRej > 0 (water/reflection passes are stomping the VP cache)
    //   - bspCamFail > 0 (cb2.c_cameraOrigin lookup broke — every fanout
    //     after that lands at -cam, behind the player)
    //   - the |T| range (closest..farthest) sits well above the GPU
    //     PointInstancer cullingRadius (default 5000, currently force-
    //     disabled in rtx_point_instancer_system.cpp:162 — but if that
    //     ever flips back on, this is where we'd see the cull threshold
    //     vs actual geometry distance)
    const bool emitGeomDiag = detailedDump
      && (m_geomDiagFanoutPublishes
        | m_geomDiagFanoutRejects
        | m_geomDiagFanoutBatches
        | m_geomDiagBlindProbes
        | m_geomDiagBspDistSamples
        | m_geomDiagFanoutMirrorRej
        | m_geomDiagBspCamFail) != 0;
    if (emitGeomDiag) {
      const float minD = m_geomDiagFanoutHaveDist ? m_geomDiagFanoutMinDist : 0.0f;
      const float maxD = m_geomDiagFanoutHaveDist ? m_geomDiagFanoutMaxDist : 0.0f;
      const std::string camAbsStr = m_geomDiagHaveCamAbs
        ? str::format("(", m_geomDiagLastCamAbs[0], ",",
                            m_geomDiagLastCamAbs[1], ",",
                            m_geomDiagLastCamAbs[2], ")")
        : std::string("<none>");
      Logger::info(str::format(
        "[SpawnGeomDiag] frame=", m_context->m_device->getCurrentFrameId(),
        " fanoutPub=", m_geomDiagFanoutPublishes,
        " fanoutRej=", m_geomDiagFanoutRejects,
        " mirrorRej=", m_geomDiagFanoutMirrorRej,
        " batches=", m_geomDiagFanoutBatches,
        " tforms=", m_geomDiagFanoutTforms,
        " blindProbes=", m_geomDiagBlindProbes,
        " bspDistSamp=", m_geomDiagBspDistSamples,
        " bspCamFail=", m_geomDiagBspCamFail,
        " |T|=[", minD, "..", maxD, "]",
        " camAbs=", camAbsStr));
      // [SpawnGeomDiag] paired histogram + per-instance build outcome
      // line. Splitting into a second line keeps the primary line at a
      // grep-friendly width and makes histogram diffing easier.
      Logger::info(str::format(
        "[SpawnGeomDiag.hist] frame=", m_context->m_device->getCurrentFrameId(),
        " batchTforms[0/1/2-4/5-16/17-64/65-256/257-1024/1025+]=",
        m_geomDiagFanoutBucket0, "/",
        m_geomDiagFanoutBucket1, "/",
        m_geomDiagFanoutBucket4, "/",
        m_geomDiagFanoutBucket16, "/",
        m_geomDiagFanoutBucket64, "/",
        m_geomDiagFanoutBucket256, "/",
        m_geomDiagFanoutBucket1k, "/",
        m_geomDiagFanoutBucketBig,
        " instSeen=", m_geomDiagFanoutInstSeen,
        " dropped[oob/nonFinite/zeroRow0]=",
        m_geomDiagFanoutInstOob, "/",
        m_geomDiagFanoutInstBadFinite, "/",
        m_geomDiagFanoutInstZeroRow0));
    }
    m_geomDiagFanoutPublishes  = 0;
    m_geomDiagFanoutRejects    = 0;
    m_geomDiagFanoutBatches    = 0;
    m_geomDiagFanoutTforms     = 0;
    m_geomDiagBlindProbes      = 0;
    m_geomDiagBspDistSamples   = 0;
    m_geomDiagFanoutMirrorRej  = 0;
    m_geomDiagBspCamFail       = 0;
    m_geomDiagFanoutMinDist    = 0.0f;
    m_geomDiagFanoutMaxDist    = 0.0f;
    m_geomDiagFanoutHaveDist   = false;
    m_geomDiagHaveCamAbs       = false;
    m_geomDiagFanoutBucket0    = 0;
    m_geomDiagFanoutBucket1    = 0;
    m_geomDiagFanoutBucket4    = 0;
    m_geomDiagFanoutBucket16   = 0;
    m_geomDiagFanoutBucket64   = 0;
    m_geomDiagFanoutBucket256  = 0;
    m_geomDiagFanoutBucket1k   = 0;
    m_geomDiagFanoutBucketBig  = 0;
    m_geomDiagFanoutInstSeen      = 0;
    m_geomDiagFanoutInstOob       = 0;
    m_geomDiagFanoutInstBadFinite = 0;
    m_geomDiagFanoutInstZeroRow0  = 0;

    // NV-DXVK: removed dead safety net. It was preempting the classifier:
    // EndFrame's EmitCs lambda ran on the CS thread the NEXT frame, calling
    // processExternalCamera with frame N's last-extracted transforms (often
    // a UI/fallback matrix) stamped as frame N+1's frameId. The next frame's
    // gameplay draws then saw Main valid for frame N+1 already →
    // shouldUpdateMainCamera=false → classifier never re-latched. The
    // current classifier + hysteresis gate leaves Main invalid on frames
    // where no gameplay draw is found, which is correct — injectRTX
    // early-returns and the native raster content passes through unchanged.

    // NV-DXVK [SkyAutoCb2]: per-frame summary + cross-frame latch update.
    // If we identified a sky origin this frame, that becomes the latched
    // sky origin for subsequent frames (sticky — sky_camera position is
    // typically static within a level). Frames where sky_camera doesn't
    // render (sky occluded) leave the latch unchanged so the next frame
    // that DOES render sky still recognizes it.
    if (m_skyDetectedThisFrame > 0 || m_skySeenOriginsThisFrame.size() >= 2) {
      const Vector3 lat = m_skyOriginLatched.value_or(Vector3{ 0.f, 0.f, 0.f });
      Logger::info(str::format(
        "[SkyAutoCb2] frame=", m_context->m_device->getCurrentFrameId(),
        " uniqueOrigins=", m_skySeenOriginsThisFrame.size(),
        " skyDraws=", m_skyDetectedThisFrame,
        " latched=", m_skyOriginLatched.has_value() ? 1 : 0,
        " latchPos=(", lat.x, ",", lat.y, ",", lat.z, ")",
        " thisFrameSky=", m_skyOriginThisFrame.has_value() ? 1 : 0));
    }
    if (m_skyOriginThisFrame) {
      m_skyOriginLatched = m_skyOriginThisFrame;
    }
    m_skyPrevFrameSeenCount = static_cast<uint32_t>(m_skySeenOriginsThisFrame.size());
    // Snapshot this frame's seen origins for the next frame's stability
    // check. swap is O(1) and we'd be discarding the list anyway.
    m_skySeenOriginsLastFrame.swap(m_skySeenOriginsThisFrame);
    m_skySeenOriginsThisFrame.clear();
    // Snapshot fanout cam for next frame's "fanout moved" bootstrap test.
    m_skyPrevFrameFanoutCam = m_lastFanoutCamOrigin;
    m_skyPrevFrameHadFanoutCam = m_hasFanoutCamOrigin;
    m_skyOriginThisFrame.reset();
    m_skyDetectedThisFrame = 0;

    m_drawCallID = 0;
    m_rawDrawCount = 0;
    m_remixActiveThisFrame = false;
    m_foundRealProjThisFrame = false;
    // Projection cache (m_projSlot, m_projOffset, m_projStage, m_columnMajor)
    // is NOT reset for pure projections (cls 1/2) — the validation path at
    // the start of ExtractTransforms re-reads and re-scans only when the
    // cached location becomes stale.  Resetting every frame would force an
    // O(stages × slots × bufferBytes) scan on the first draw, which hangs
    // emulators with 64KB+ UBOs.
    //
    // NV-DXVK: Combined VP (cls 3/4) MUST be re-scanned each frame because
    // (a) the VP content changes with camera movement, and (b) Source only
    // binds the correct VP cbuffer during the main opaque pass (draws 200+),
    // not during early shadow/depth-prepass draws.  If we cached a false
    // positive from an early draw on the previous frame, resetting here gives
    // the next frame's late-draw scan a chance to find the real VP.
    if (m_projIsCombinedVP) {
      m_projSlot   = UINT32_MAX;
      m_projOffset = SIZE_MAX;
      m_projStage  = -1;
    }
    ++m_axisDetectFrame;

    // NV-DXVK: Snapshot whether MaybeEarlyInjectForUITexture already fired
    // this frame, so the CS-thread log below can report whether this tail
    // endFrame call is doing the real injectRTX or just hitting the
    // m_frameLastInjected no-op guard.
    const bool earlyFired = m_earlyInjectFiredThisFrame;
    const bool logThis = detailedDump;

    // NV-DXVK [HUD-Option4]: snapshot the HUD target for this frame and
    m_context->EmitCs([backbuffer, draws, earlyFired, logThis](DxvkContext* ctx) {
      RtxContext* rtx = static_cast<RtxContext*>(ctx);
      if (logThis) {
        const uint32_t fid = rtx->getDevice()->getCurrentFrameId();
        const bool camValid = rtx->getSceneManager().getCamera().isValid(fid);
        Logger::info(str::format("[D3D11Rtx] CS endFrame: frameId=", fid,
          " draws=", draws, " camValid=", camValid ? 1 : 0,
          " earlyInjected=", earlyFired ? 1 : 0));
      }
      rtx->endFrame(0, backbuffer, true);
    });

    // Reset per-frame state. Values captured into the lambda above
    // so it's safe to clear before the CS thread gets to it.
    m_earlyInjectFiredThisFrame = false;
    // v4: drop any composite-output target that didn't get consumed
    // (no post-composite draw happened this frame). Avoids dangling
    // ref across frames.
    m_compositeOutputPending = nullptr;
    m_compositeOutputThisFrame = nullptr;
  }

  void D3D11Rtx::OnPresent(const Rc<DxvkImage>& swapchainImage) {
    m_context->EmitCs([swapchainImage](DxvkContext* ctx) {
      RtxContext* rtx = static_cast<RtxContext*>(ctx);
      rtx->onPresent(swapchainImage);
    });
  }

}
