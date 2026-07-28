#pragma once

#include "d3d11_include.h"
#include <array>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../dxvk/rtx_render/rtx_types.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../dxvk/rtx_render/rtx_materials.h"
#include "../dxvk/dxvk_buffer.h"
// NV-DXVK [perf]: kBoneMirrorRegions + the mirror dirty-stamp declarations
// that m_boneMirrorRegionMergedGen below is sized from.
#include "../dxvk/dxvk_bone_diag.h"
#include "../util/util_matrix.h"
#include "../util/util_threadpool.h"

namespace dxvk {

  class D3D11DeviceContext;

  // NV-DXVK [MatDefer]: self-contained snapshot of the D3D11 PS pipeline state that
  // FillMaterialData reads. Defined in d3d11_rtx.cpp (needs D3D11ContextStatePS).
  // Built synchronously on the game thread; when deferred it is captured into a
  // geometry-worker task so the material compute runs off the serial SubmitDraw path.
  struct MatSnapshot;

  // NV-DXVK [BatchSubmitDraw]: per-frame collect arena for rtx.batchSubmitDrawStages.
  // Holds one DrawWorkItem per RT commit of the frame (params + moved DrawCallState +
  // captured MatSnapshot). Defined in d3d11_rtx.cpp because DrawWorkItem embeds the
  // .cpp-local MatSnapshot; the D3D11Rtx member is an owning pointer (pImpl) which is
  // why this class needs an out-of-line destructor.
  struct GeometryBatchArena;

  // NV-DXVK [Phase1] GPU-driven-injection formal layout descriptor.
  // See HANDOFF_GPU_DRIVEN_INJECTION.md §5. This is the FORMALIZED, uploadable
  // consolidation of the per-VS caches ExtractTransforms already maintains
  // (the vsLocCache projection/view cbuffer locations + the IlFacts pure
  // input-layout facts + the VS classifier kind) into ONE record per distinct
  // vertex shader. A future GPU compute pass indexes an uploaded array of
  // these by a small dense layoutId to reconstruct the fixed camera/projection
  // part of each draw's transforms without the CPU heuristic tangle.
  //
  // Phase-0 correction to §5: the handoff's draft descriptor also listed the
  // objectToWorld source (o2wSource / o2wCbSlot). The [Phase0] measurement
  // proved o2w SOURCE is a per-DRAW property — the same VS draws instanced
  // (t31 SRV) on some draws and single-instance (cbuffer block) on others
  // within a single frame — so it is deliberately NOT here; it belongs in the
  // Phase-2 per-draw capture record. Every field below was verified STABLE
  // per VS by the [Phase0] pass. All fields are 32-bit so the struct is a
  // tight, GPU-upload-friendly POD (no float alignment / padding surprises).
  struct VsLayoutDescriptor {
    uint32_t pathClass         = 0;          // D3D11VsClassification::Kind (as uint)
    int32_t  projStage         = -1;         // projection cbuffer: pipeline stage
    uint32_t projSlot          = UINT32_MAX; //   ... cb slot
    uint32_t projByteOffset    = 0;          //   ... byte offset within the bound range
    int32_t  viewStage         = -1;         // view cbuffer: pipeline stage
    uint32_t viewSlot          = UINT32_MAX; //   ... cb slot
    uint32_t viewByteOffset    = 0;          //   ... byte offset within the bound range
    uint32_t columnMajor       = 0;          // bool: engine stores matrices column-major
    uint32_t texcoordEncoding  = 0;          // RtSurface::TexcoordEncoding (Float / packed-uint)
    uint32_t hasUintPos        = 0;          // IlFacts: POSITION0 == R32G32_UINT (BSP world-pack)
    uint32_t hasInstIdxSem     = 0;          // IlFacts: per-instance R16G16B16A16_UINT index present
    uint32_t instSemSlot       = UINT32_MAX; //   ... its input slot
    uint32_t instSemByteOffset = 0;          //   ... its byte offset
    uint32_t hasBlendIndices   = 0;          // IlFacts: per-vertex BLENDINDICES0 (skinned)

    bool equals(const VsLayoutDescriptor& o) const {
      return pathClass == o.pathClass
          && projStage == o.projStage && projSlot == o.projSlot && projByteOffset == o.projByteOffset
          && viewStage == o.viewStage && viewSlot == o.viewSlot && viewByteOffset == o.viewByteOffset
          && columnMajor == o.columnMajor && texcoordEncoding == o.texcoordEncoding
          && hasUintPos == o.hasUintPos && hasInstIdxSem == o.hasInstIdxSem
          && instSemSlot == o.instSemSlot && instSemByteOffset == o.instSemByteOffset
          && hasBlendIndices == o.hasBlendIndices;
    }
  };

  // NV-DXVK [Phase1]: owns the dense layoutId <-> descriptor mapping, keyed by
  // the VS common-shader pointer (the same identity vsLocCache uses).
  struct VsLayoutTable {
    struct Entry {
      VsLayoutDescriptor desc;                 // the uploadable per-VS descriptor
      bool               complete      = false; // proj/view were validly resolved at least once
      uint32_t           mismatchCount = 0;    // times a COMPLETE entry's layout later changed
    };
    std::unordered_map<uintptr_t, uint32_t> idByVs;  // VS common-shader ptr -> layoutId
    std::vector<Entry>                      entries; // indexed by layoutId

    // Get-or-allocate the dense layoutId for this VS and reconcile its
    // descriptor. `projValid` = the draw resolved a real projection (proj slot
    // != UINT32_MAX); only then is the descriptor trusted as COMPLETE. A cold
    // first draw whose projection wasn't found yet is stored but left
    // INCOMPLETE, and the first later valid resolution silently completes it
    // (NOT a mismatch) — mirrors the vsLocCache write guard, so the table never
    // bakes the transient. `changed` is set true only when a COMPLETE entry's
    // layout genuinely differed and was updated.
    uint32_t getOrAdd(uintptr_t vsKey, const VsLayoutDescriptor& d,
                      bool projValid, bool& changed) {
      changed = false;
      auto it = idByVs.find(vsKey);
      if (it == idByVs.end()) {
        // Bound growth across level changes (VS pointers churn + can be reused):
        // cap and reset like the neighbouring vsLocCache. layoutIds are rebuilt
        // fresh each session/level and never persisted across this clear.
        if (entries.size() >= 8192u) { idByVs.clear(); entries.clear(); }
        const uint32_t id = static_cast<uint32_t>(entries.size());
        Entry e;
        e.desc     = d;
        e.complete = projValid;
        entries.push_back(e);
        idByVs.emplace(vsKey, id);
        return id;
      }
      const uint32_t id = it->second;
      Entry& e = entries[id];
      if (!e.complete) {
        e.desc = d;                       // still filling — accept, complete on first valid proj
        if (projValid) e.complete = true;
      } else if (projValid && !e.desc.equals(d)) {
        ++e.mismatchCount;
        e.desc  = d;
        changed = true;
      }
      return id;
    }
  };

  // NV-DXVK [Phase2] GPU-driven-injection per-draw capture record.
  // One per INJECTED draw (past every SubmitDraw filter / early-return),
  // collected into a per-frame arena. Holds what a future GPU transform pass
  // needs — the draw's layoutId (indexes VsLayoutTable) + the raw camera
  // cbuffer inputs — plus the CPU-resolved matrices as ground truth for the
  // Phase-3 GPU-vs-CPU verification. Behavior-neutral: nothing consumes it yet.
  struct DrawCaptureRecord {
    uint32_t     layoutId    = UINT32_MAX; // -> D3D11Rtx::m_vsLayoutTable.entries[layoutId]
    uint32_t     drawCallID  = 0;          // correlation with the per-draw logs
    uint32_t     o2wPathId   = 0;          // per-draw objectToWorld source (Phase-0 tier 2)
    uint32_t     o2wSrcClass = 0;          // coarse class: 0 id / 1 t31 / 2 bones / 3 cbField / 4 other
    bool         usedFallback = false;     // ExtractTransforms found no real projection (UI/ortho path)
    XXH64_hash_t vsHash      = 0;          // the draw's VS hash (correlation; 0 = no VS bound)
    // Raw camera cbuffer snapshots — the GPU pass INPUTS — copied by value at
    // capture time from the descriptor-pointed slot/offset (never referenced by
    // address across the frame — §7 landmine). The read can fail (device-local /
    // unmapped / out of range), hence the validity flags.
    float   projCb[16] = {};   bool projCbValid = false;
    float   viewCb[16] = {};   bool viewCbValid = false;
    // CPU-resolved matrices — Phase-3 verifies GPU-recomputed output against these.
    Matrix4 objectToWorld;
    Matrix4 worldToView;
    Matrix4 viewToProjection;
  };

  class D3D11Rtx {
  public:
    explicit D3D11Rtx(D3D11DeviceContext* pContext);
    // NV-DXVK [BatchSubmitDraw]: out-of-line so std::unique_ptr<GeometryBatchArena>
    // (incomplete here) can be destroyed where the type is complete.
    ~D3D11Rtx();

    void Initialize();
    // Returns true if the draw was captured for RT (caller should skip rasterization).
    bool OnDraw(UINT vertexCount, UINT startVertex);
    bool OnDrawIndexed(UINT indexCount, UINT startIndex, INT baseVertex);
    bool OnDrawInstanced(UINT vertexCountPerInstance, UINT instanceCount, UINT startVertex, UINT startInstance);
    bool OnDrawIndexedInstanced(UINT indexCountPerInstance, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance);

    // NV-DXVK: Intercept UpdateSubresource to cache bone matrix data from t30.
    // Called from D3D11DeviceContext::UpdateSubresource before the data goes to GPU.
    void OnUpdateSubresource(ID3D11Resource* pDstResource, const void* pSrcData, UINT SrcDataSize, UINT DstOffset = 0, UINT BufSize = 0);

    // NV-DXVK [SubmitStall]: called from each OnDraw* after the SubmitDraw call,
    // with that call's whole-wall time. When rtx.logSubmitStall is on, logs the
    // outlier (slow) draws + a per-frame roll-up so we can see where the ~2.5x
    // gap between submitDrawAccUs and the instrumented wallUs actually is.
    void recordSubmitStall(const char* type, int64_t dUs, uint32_t primCount, uint32_t instCount);

    // Must be called with the context lock held.
    // EndFrame runs the RT pipeline writing output into backbuffer (called BEFORE recording the blit).
    void EndFrame(const Rc<DxvkImage>& backbuffer);
    // OnPresent registers the swapchain present image (called AFTER recording the blit).
    void OnPresent(const Rc<DxvkImage>& swapchainImage);

    uint32_t getDrawCallID() const { return m_drawCallID; }

    // NV-DXVK: Cross-context draw-count transfer. Deferred contexts record
    // draws onto their own D3D11Rtx instance, so their m_drawCallID is
    // independent from the immediate context's.  FinishCommandList snapshots
    // the deferred counter into the D3D11CommandList and resets it (so the
    // next recording on that deferred context starts from zero); at
    // ExecuteCommandList time the immediate context accumulates the stored
    // count so D3D11Rtx::EndFrame reports the true total for the frame and
    // the kMaxConcurrentDraws throttle remains meaningful.
    void resetDrawCallID() {
      m_drawCallID = 0;
      // [SkyAutoCb2] Reset deferred-context per-recording sky state. Carry
      // m_skyOriginLatched across recordings (it's the cross-frame latch
      // and should persist) and snapshot this recording's seen origins
      // for the next recording's stability check. Per-recording fields
      // (current sky origin, detection counter) are cleared.
      m_skyPrevFrameSeenCount = static_cast<uint32_t>(m_skySeenOriginsThisFrame.size());
      m_skySeenOriginsLastFrame.swap(m_skySeenOriginsThisFrame);
      m_skySeenOriginsThisFrame.clear();
      m_skyPrevFrameFanoutCam = m_lastFanoutCamOrigin;
      m_skyPrevFrameHadFanoutCam = m_hasFanoutCamOrigin;
      if (m_skyOriginThisFrame) {
        m_skyOriginLatched = m_skyOriginThisFrame;
      }
      m_skyOriginThisFrame.reset();
      m_skyDetectedThisFrame = 0;
    }
    void addDrawCallID(uint32_t count) { m_drawCallID += count; }

    // NV-DXVK: Cache the swap-chain backbuffer image so the uiTextures
    // insertion hook (MaybeEarlyInjectForUITexture) has a target to pass
    // to injectRTX without reaching into the swap chain mid-draw. Called
    // once per present from D3D11SwapChain::PresentImage. Only actually
    // re-binds + logs when the underlying DxvkImage changes (resize),
    // otherwise a cheap no-op.
    void SetSwapchainBackbuffer(const Rc<DxvkImage>& backbuffer);

  private:
    // NV-DXVK: Implements the "standard Remix way" UI path that was
    // declared in rtx_options.h (rtx.uiTextures) but never actually wired
    // up in this DX11 port. On entry to SubmitDraw we scan the currently
    // bound PS SRVs; if any image hash matches RtxOptions::uiTextures()
    // and we haven't already fired this frame, we emit injectRTX into the
    // main CS chunk so the RT render+blit happens BEFORE the game's
    // subsequent UI native-raster EmitCs's (this draw and all following
    // UI draws). D3D11Rtx::EndFrame's usual tail injectRTX then hits the
    // m_frameLastInjected guard (rtx_context.cpp:491) and no-ops on the
    // CS thread, so we don't double-inject and we don't touch CS-chunk
    // ordering.
    //
    // Why hash-gated instead of heuristic: my earlier FullscreenQuad /
    // NoLayout heuristic kept tripping on post-process fullscreen quads,
    // deferring the game's own scene composition past injectRTX and
    // ending up with a black scene. User-declared texture hashes don't
    // have that ambiguity — post-process passes don't use HUD textures.
    void MaybeEarlyInjectForUITexture();

    // NV-DXVK: When a draw gets rejected down one of the HUD-class
    // filter branches (NoLayout / NoSemantics / true-UI UIFallback), log
    // the bound PS SRV image hashes so the user can copy them into
    // rtx.uiTextures to actually wire up MaybeEarlyInjectForUITexture.
    // Throttled by unique (VS,PS,hashSet) tuple so we don't drown the log.
    void LogPsHashesForHudFilter(const char* site);

    // NV-DXVK: 64-bit prefix of the currently bound VS / PS SHA1 — same
    // bitpattern the HUD-filter log prints as vsHash/psHash, and the
    // comparison key used against rtx.uiVertexShaderHashes /
    // rtx.uiPixelShaderHashes. Member (not free function) because
    // D3D11DeviceContext::m_state is protected; D3D11Rtx is a friend.
    void GetCurrentVsPsHashes(XXH64_hash_t& outVs, XXH64_hash_t& outPs) const;

    // NV-DXVK [BoneStablePropId]: derive a stable per-DCS prop identity
    // for bone-animated draws (skinned characters, viewmodel, fanout
    // with bone palette). Hashes engine-side stable buffer pointers
    // — vertex buffer, index buffer, and bone palette SRV's underlying
    // D3D11Buffer — so the resulting propId survives the per-frame
    // matrix churn that defeats matrix-bytes-based SpatialMap dedup.
    //
    // For fanout (path-10): pass firstInstanceObjectToWorld so the
    // rounded translation is folded into the hash, disambiguating two
    // distinct fanout groups that happen to share the same VB/IB/t30
    // (e.g., two ship formations using the same character mesh).
    // For non-fanout paths (11, 12, path-10 N-draw): pass nullptr.
    //
    // Returns a non-zero 64-bit hash, or 0 when no IA buffers and no
    // t30 are bound (caller should leave stablePropId at its existing
    // default in that case so spatial-map dedup falls back to matrix
    // bytes).
    uint64_t MakeBoneStablePropId(const Matrix4* firstInstanceObjectToWorld) const;

    static constexpr uint32_t kMaxConcurrentDraws = 6 * 1024;
    // NV-DXVK [BatchSubmitDraw perf]: LowLatency=FALSE so idle workers SLEEP on a
    // condition variable instead of spinning. The default (LowLatency=true) makes
    // every idle worker busy-loop the work-stealing scan, and each steal attempt
    // grabs the single global m_threadMutex spinlock (util_threadpool.h) — so N
    // idle workers storm one lock at 100% CPU. This pool runs its parallel-for only
    // once per frame (flushGeometryBatch) / a burst of per-draw schedules, then sits
    // idle the rest of the frame, so spinning stole ~N cores from the game + CS
    // threads for ~95% of every frame (uniform ~40% inflation of ALL serial work).
    // Matches the other two pools in the tree (dxvk_raytracing, rtx_asset_exporter),
    // which both use LowLatency=false for the same reason. WorkStealing stays true.
    using GeometryProcessor = WorkerThreadPool<kMaxConcurrentDraws, /*WorkStealing*/ true, /*LowLatency*/ false>;

    D3D11DeviceContext*                  m_context;
    std::unique_ptr<GeometryProcessor>   m_pGeometryWorkers;
    // NV-DXVK [BatchSubmitDraw]: per-frame arena of collected RT commits, drained in
    // one parallel-for by flushGeometryBatch() at frame end (rtx.batchSubmitDrawStages).
    // Only populated on the immediate context (deferred contexts never call EndFrame,
    // so they keep the per-draw EmitCs path and cannot orphan the arena). Accessed only
    // on the owning (game) thread, so no lock is needed.
    std::unique_ptr<GeometryBatchArena>  m_geoBatch;
    // Runs the frame-end batch: parallel-for over m_geoBatch finalizing each draw's
    // deferred compute, JOIN, then re-emit commitGeometryToRT in original draw order.
    // No-op when the arena is empty. Called at the top of EndFrame (before its own
    // camera/inject EmitCs work) so all geometry is committed before injectRTX.
    void flushGeometryBatch();
    uint32_t                             m_drawCallID = 0;
    // True when SubmitDraw successfully committed a draw to the RT pipeline.
    // Checked by OnDraw* return value to suppress redundant D3D11 rasterization.
    bool                                 m_lastDrawCaptured = false;
    // NV-DXVK: set by SubmitDraw when the draw was filtered as UI. OnDraw*
    // uses this to force native rasterization for UI draws even after Remix
    // is active on the frame, so the HUD/menu stays visible. Without this
    // flag, once m_remixActiveThisFrame flips true for a gameplay draw, every
    // subsequent UI draw has its native raster suppressed as well and the UI
    // never appears on screen.
    bool                                 m_lastDrawFilteredAsUI = false;
    // NV-DXVK: Strict subset of m_lastDrawFilteredAsUI — set only when the
    // rejection reason is unambiguously HUD/VGUI (NoInputLayout,
    // NoSemantics, UIFallback "true_ui" / degenerate_cached_w2v) and NOT
    // when it's just FullscreenQuad (post-process / tone map / bloom).
    // Used by LogPsHashesForHudFilter to decide when to dump bound PS SRV
    // image hashes — those are the hashes the user should add to
    // rtx.uiTextures to make MaybeEarlyInjectForUITexture actually fire.
    bool                                 m_lastDrawIsHudClass = false;
    // NV-DXVK: V2 classifier flag. True when ExtractTransforms' classifier
    // definitively identified this draw as UI (screenspace 2D, no real
    // transform). Forces SubmitDraw into the TRUE UI branch even when
    // m_foundRealProjThisFrame=true from prior gameplay draws, so UI
    // buttons/HUD always hit native rasterization.
    bool                                 m_lastClassifierSaidUi = false;
    // True once ANY draw in the current frame was captured for RT.
    // Once Remix is active, ALL D3D11 rasterization is suppressed (including
    // filtered draws) because the game's native rasterization shares render
    // targets with Remix output → write hazards → corruption → TDR.
    // Reset to false each EndFrame. During menus (no RT captures), this stays
    // false and all draws rasterize normally.
    bool                                 m_remixActiveThisFrame = false;
    // NV-DXVK [VanishDiag-Raw]: per-VS-hash histogram of OnDraw* entries
    // this frame. Compared at EndFrame against scene_manager's vsHistogram
    // (which counts only draws that reached processDrawCallState) to
    // identify which VS families are submitted by the engine but dropped
    // by Remix's classifier between OnDraw* and processDrawCallState.
    // Cleared in EndFrame.
    std::unordered_map<uint64_t, uint32_t> m_rawVsHistogram;
    // NV-DXVK: Per-frame gate for MaybeEarlyInjectForUITexture so the
    // injectRTX lambda is emitted at most once per frame even though
    // many HUD draws will match a uiTextures entry. Reset in EndFrame.
    bool                                 m_earlyInjectFiredThisFrame = false;
    // NV-DXVK [EngineCam]: last value of g_engineMainFrame that EndFrame
    // forwarded to camera_manager via processExternalCamera. Compared each
    // EndFrame against the live counter; if equal, the trampoline didn't
    // fire this frame (e.g. menu/loading/no-world-pass) so we skip the
    // update and leave Main on its previous valid pose. Initialised to
    // UINT32_MAX so the first real capture (counter == 1) always fires.
    uint32_t                             m_lastConsumedEngineMainFrame = UINT32_MAX;
    // NV-DXVK [zigzag fix A]: camera ORIGIN of the engine pose most recently
    // CONSUMED into the Main render camera (recovered -R^T*t from g_engineMainW2v
    // at consume time) — i.e. the camera the scene is actually ray-traced from.
    // Camera-relative geometry (gun, platform) is baked against the LIVE per-draw
    // c_cameraOrigin, which runs one engine-tick AHEAD of Main (documented 1-frame
    // lag in EndFrame ~21601). The path-3 worldToView reconstruction uses THIS
    // instead, so geometry is placed against the camera it's viewed from -> no
    // horizontal zig-zag. Written in the EndFrame consumer and read on the same
    // (calling) thread during the next frame's draws, so a plain member is
    // race-free.
    Vector3                              m_renderCamOriginConsumed{ 0.0f, 0.0f, 0.0f };
    bool                                 m_hasRenderCamOriginConsumed = false;
    // NV-DXVK [zigzag fix B]: delay ring for the engine-hook Main camera.
    // The R_DrawWorldMeshes trampoline captures the world camera for engine
    // frame G, but the D3D draws of frame G (world, moving platforms, the
    // viewmodel) reach the ray tracer one engine-frame out of phase with where
    // the consumer lands Main (Source's threaded/queued renderer + DXVK's
    // EmitCs deferral). On static world this only reads as latency; on MOVING
    // geometry it is the measured horizontal zig-zag (v0.y(N) == camMain.y(N-1)
    // in the [ZigVB] trace). We hold the last few DISTINCT-engine-frame
    // captures (row-major engine floats, exactly as written into the EmitCs
    // Matrix4 below) and feed Main the capture that is
    // RtxOptions::engineHookMainCameraFrameDelay() frames behind the newest, so
    // Main phase-aligns with the geometry of the same engine frame. Pushed once
    // per distinct curEngineFrame in the EndFrame consumer (advance branch);
    // re-fed unchanged on no-advance presents. Same (calling) thread as the
    // rest of the consumer, so a plain member is race-free.
    static constexpr uint32_t            kEngineCamDelayRing = 8;
    float                                m_engineCamRingW2v[kEngineCamDelayRing][16] = {};
    float                                m_engineCamRingV2p[kEngineCamDelayRing][16] = {};
    uint32_t                             m_engineCamRingCount = 0; // total distinct-frame pushes
    bool                                 m_engineCamDelayedValid = false;
    float                                m_engineCamDelayedW2v[16] = {}; // last selected (for no-advance re-feed)
    float                                m_engineCamDelayedV2p[16] = {};
    // NV-DXVK [EngineCam-Skybox]: parallel to m_lastConsumedEngineMainFrame
    // but for the 3D-skybox sub-view trampoline capture. Used by the
    // [EngineSky] diagnostic logger in EndFrame to deduplicate the
    // capture-frame counter.
    uint32_t                             m_lastConsumedEngineSkyFrame  = UINT32_MAX;
    // NV-DXVK [HUD-Option5 v4]: TF2's composite PS (1d403438f8cee21c)
    // writes its tonemapped output to the 2048x1152 R8G8B8A8_SRGB
    // backbuffer. We blit our post-tonemap RT over that image between
    // composite and the subsequent HUD rasters, so HUD layers on top
    // of our RT. `Pending` is set when the composite draw is seen and
    // consumed on the next SubmitDraw (which queues the blit lambda
    // AFTER the composite draw and BEFORE the first HUD draw on CS).
    // `ThisFrame` is a sticky copy reset in EndFrame.
    Rc<DxvkImage>                        m_compositeOutputPending;
    Rc<DxvkImage>                        m_compositeOutputThisFrame;
    // Extent of the composite RT (set when CompositeOut v4 captures it).
    // Used as the "main viewport" reference for the fanout publish so we
    // don't have to guess with arbitrary pixel thresholds — the main view
    // is whatever viewport matches the final composite output extent.
    // Auto-tracks render-scale / fullscreen / resolution changes. Zero
    // until the first composite RT detection of the session.
    uint32_t                             m_compositeOutputW = 0;
    uint32_t                             m_compositeOutputH = 0;
    // NV-DXVK: Latest primary-swap-chain backbuffer — captured in
    // D3D11SwapChain::PresentImage on every present. Stable across frames
    // unless the swap chain is recreated (resize), so the refresh is
    // free in steady state. MaybeEarlyInjectForUITexture hands this to
    // injectRTX as its targetImage.
    Rc<DxvkImage>                        m_cachedBackbuffer;
    // NV-DXVK: Raw draw counter incremented on every OnDraw* call BEFORE
    // any filtering.  Used purely for diagnostics so the EndFrame log can
    // distinguish "game issued no draws" from "game issued N draws but all
    // of them were rejected by SubmitDraw's pre-filters".
    uint32_t                             m_rawDrawCount = 0;

    // NV-DXVK: communication channel from ExtractTransforms() back to the
    // caller (SubmitDraw) for the TLAS-coherence filter. ExtractTransforms
    // captures the draw's c_cameraOrigin (cb2 offset 4) into these; SubmitDraw
    // compares against the latched Main camera world position post-extract
    // and rejects draws whose coord space disagrees with Main.
    Vector3                              m_lastDrawCamOrigin{ 0.0f, 0.0f, 0.0f };
    bool                                 m_lastDrawCamOriginSet = false;

    // NV-DXVK: which worldToView assignment path fired for this draw. Set by
    // each `transforms.worldToView = ...` site to a unique small integer.
    // Dumped at Main-camera latch time so we can identify which path is
    // producing the (wrong) latched pose. 0 = not set this draw.
    uint32_t                             m_lastWtvPathId = 0;
    // NV-DXVK: which objectToWorld assignment path fired for this draw.
    //   0 = unset (stayed identity)
    //   1 = non-inst BSP t31 read (new)
    //   2 = legacy t30 CPU Bone (hasBoneIdx + bonePtr)
    //   3 = legacy t30 Bone-from-MappedSlice (bonePtr null, cached)
    //   4 = CB3 read (invView * cb3Mat, m_skipViewMatrixScan)
    //   5 = RDEF CBufModelInstance
    //   6 = trySourceFloat3x4 legacy heuristic
    //   7 = tryWorldCb generic 4x4 scan
    //   8 = cb2@4 cameraOrigin fallback
    //   9 = per-instance override in SubmitDraw(instanceTransform)
    //  10 = bone-instanced: o2w=identity (instancesToObject handles it)
    uint32_t                             m_lastO2wPathId = 0;

    // NV-DXVK: the canonical gameplay camera origin, populated by the
    // bone-fanout RDEF lookup at line ~593. Different VS permutations have
    // different c_cameraOrigin values bound to their cb2 (reflection probes,
    // shadow maps, mech cockpit, etc.), so path 1 and path 3 can't trust the
    // c_cameraOrigin of whatever VS happens to trigger Main latch. Instead
    // they use THIS value — the one the actual gameplay BSP fanout shader
    // reports — which is authoritative for "the camera we want to raytrace
    // from". Valid once any bone-fanout draw fires in the session.
    Vector3                              m_lastFanoutCamOrigin{ 0.0f, 0.0f, 0.0f };
    bool                                 m_hasFanoutCamOrigin = false;
    // NV-DXVK [pilot-eye-capture]: cb2 c_cameraOrigin captured from the
    // viewmodel-pass (vpMaxDepth ≤ 0.08) at the same fanout site that
    // populates m_lastFanoutCamOrigin. The viewmodel pass binds Source's
    // canonical eye position — important on rodeo (pilot riding on top of
    // a Titan) where the BSP-pass cb2 carries the Titan's cockpit origin
    // (~600u below pilot eye) and lp+0x3D6C is a static script anchor on
    // this build. Captured exactly as bound (no matrix decomposition →
    // no float decomposition noise that the rtx_camera_manager-side
    // recovC reconstruction was suffering). CameraManager reads this
    // through GetD3D11RtxPilotEye() to snap Main's worldToView translation.
    Vector3                              m_lastViewmodelCamOrigin{ 0.0f, 0.0f, 0.0f };
    bool                                 m_hasViewmodelCamOrigin = false;
    // NV-DXVK: VP rotation rows captured at the SAME fanout moment as
    // m_lastFanoutCamOrigin. Different VS permutations bind different cb2
    // contents (reflection, shadow, cubemap, mech cockpit …) with different
    // VP rotations; reading cb2@96 per-draw picks up whatever rotation was
    // bound for that particular draw, causing path 3 to produce 90°-flipped
    // bases between frames. Caching the row vectors from the authoritative
    // gameplay fanout VS gives every subsequent path-3 draw the same
    // orientation and stops the latch flicker. Each row is the raw float3
    // from cb2@96 rows 0/1/2 (right/up/fwd × projection scale) — normalize
    // and re-orthogonalize at use site, same as path 1.
    Vector3                              m_lastFanoutVpRow0{ 0.0f, 0.0f, 0.0f };
    Vector3                              m_lastFanoutVpRow1{ 0.0f, 0.0f, 0.0f };
    Vector3                              m_lastFanoutVpRow2{ 0.0f, 0.0f, 0.0f };
    bool                                 m_hasFanoutVpRows = false;
    // NV-DXVK [secondary fanout slots — per-sub-camera basis cache]:
    // The single m_lastFanout* cache only remembers whichever fanout
    // publish landed most recently. Source-engine 3D-skybox draws come
    // from multiple sub-cameras (player + sky_camera sub-cams for
    // distant geometry clusters); they all publish through the same
    // fanout path with different origins and different VP rotations.
    // cls12Recon path 3 reads the current draw's own cb2.c_cameraOrigin
    // for the camera position but pairs it with the cached single
    // m_lastFanoutVpRow* for the basis — so a ship draw whose cb2
    // origin lives in its sub-camera ends up projected through whichever
    // basis was published last (usually the player's). The transform
    // doesn't fit the geometry → vertices land off-frustum → invisible.
    //
    // Slots remember up to kFanoutSkySlotCount distinct (origin, VP-rows)
    // pairs. On a fanout publish we match by origin within 500u; if no
    // existing slot matches and a free slot is available, seed it.
    // Origins are FIXED at seed time (no EMA) so the basis we hand to
    // path 3 is deterministic across launches. In path 3 we pick the
    // slot whose origin is closest to the current draw's cb2 origin
    // within the same 500u radius; on miss we fall back to the cached
    // single m_lastFanoutVpRow* exactly like baseline.
    static constexpr uint32_t kFanoutSkySlotCount = 4;
    struct FanoutSkySlot {
      Vector3 origin{ 0.0f, 0.0f, 0.0f };
      Vector3 vpRow0{ 0.0f, 0.0f, 0.0f };
      Vector3 vpRow1{ 0.0f, 0.0f, 0.0f };
      Vector3 vpRow2{ 0.0f, 0.0f, 0.0f };
      bool    hasOrigin = false;
      bool    hasVpRows = false;
    };
    FanoutSkySlot                        m_fanoutSkySlots[kFanoutSkySlotCount];

    // NV-DXVK [SkyAutoCb2]: cb2.c_cameraOrigin-driven sky categorization.
    //
    // Source-engine-derived games (Titanfall 2) reuse the same VS shaders
    // for both the 3D-skybox draws (rendered through sky_camera) and the
    // main world pass — there is no static bytecode signal to distinguish
    // them. The reliable runtime signal is c_cameraOrigin in
    // CBufCommonPerCamera (cb2 byte 4): the sky_camera entity binds a
    // different origin than the main camera.
    //
    // Detection LATCHES the sky origin across frames. Once we've identified
    // the sky_camera origin, every future draw whose c_cameraOrigin matches
    // it (within RtxOptions::skyAutoDetectUniqueCameraDistance()) is sky.
    // The earlier "first observed origin = sky" rule was only safe when the
    // sky pass actually ran first; if sky_camera was occluded that frame,
    // the main camera's origin became "first" and the entire frame got
    // dropped to Hillaire (the user's "nothing but sky" symptom).
    // Bootstrap (no prior sky known): fall back to the old "first new origin
    // this frame is sky if last frame disambiguated" rule.
    std::vector<Vector3>                 m_skySeenOriginsThisFrame;
    // Snapshot of last frame's seen-origins list, used for the
    // stability-gated bootstrap: bootstrap only latches an origin that
    // ALSO appeared in the previous frame (within tight threshold) AND
    // that the fanout main-camera moved since then. Sky_camera positions
    // are fixed in world space — they DON'T move when the player moves.
    // Viewmodel / eye-bob / aux origins DO move with the player, so they
    // never satisfy "stable while fanout moved" and never bootstrap.
    std::vector<Vector3>                 m_skySeenOriginsLastFrame;
    // Fanout main-camera origin recorded at the END of the previous
    // frame, used to verify the player moved since then before
    // accepting a candidate as sky_camera.
    Vector3                              m_skyPrevFrameFanoutCam{ 0.f, 0.f, 0.f };
    bool                                 m_skyPrevFrameHadFanoutCam = false;
    uint32_t                             m_skyPrevFrameSeenCount = 0;
    // The c_cameraOrigin we identified as the sky_camera's, latched across
    // frames. nullopt until the first frame where bootstrap classifies a
    // first origin as sky. Once set, it sticks (sky_camera position is
    // typically static within a level) and gives subsequent frames a
    // ground-truth comparison instead of having to re-bootstrap.
    std::optional<Vector3>               m_skyOriginLatched;
    // The origin chosen as sky in the CURRENT frame. Reset each frame; if
    // it's set when EndFrame runs, m_skyOriginLatched is updated to it.
    std::optional<Vector3>               m_skyOriginThisFrame;
    // True when SetSkyCategoryFromCb2 set Sky on the most recent dcs —
    // used by the [SkyAutoCb2] log line to count detection events.
    uint32_t                             m_skyDetectedThisFrame = 0;

    // NV-DXVK: Per-frame bone instancing stats
    uint32_t                             m_boneInstBatches = 0;
    uint32_t                             m_boneInstTotal = 0;
    uint32_t                             m_boneInstSkipped = 0;
    uint32_t                             m_boneInstNoCache = 0;
    uint32_t                             m_boneInstCacheHits = 0;
    uint32_t                             m_boneInstCacheMisses = 0;
    std::unordered_set<uintptr_t>        m_boneInstVbPtrs;  // unique VB ptrs this frame

    // NV-DXVK [SpawnGeomDiag]: Per-frame BSP/world-geometry diagnostic
    // counters. Gives a one-line census per frame so we can answer "why is
    // there missing world geometry on spawn" without reading thousands of
    // capped one-shot logs. Emit happens in EndFrame; counters reset there.
    uint32_t m_geomDiagFanoutPublishes = 0; // [D3D11Rtx.fanoutOri] publish events
    uint32_t m_geomDiagFanoutRejects   = 0; // [D3D11Rtx.fanoutOri] reject events
    uint32_t m_geomDiagFanoutBatches   = 0; // SubmitDraw fired with non-empty tforms
    uint32_t m_geomDiagFanoutTforms    = 0; // sum of tforms.size() across batches
    uint32_t m_geomDiagBlindProbes     = 0; // BLIND-PROBE classify hits
    uint32_t m_geomDiagBspDistSamples  = 0; // unique-VS BSP-dist samples taken
    uint32_t m_geomDiagFanoutMirrorRej = 0; // VP rows rejected as mirror (det>=0)
    uint32_t m_geomDiagBspCamFail      = 0; // BSP camOrigin lookup failures
    // [SpawnGeomDiag] Per-frame histogram of fanout-batch tform counts.
    // Each bucket counts how many SubmitDraw calls produced an
    // instancesToObject vector of that size range. Lets us tell apart
    // "few large fanouts" (e.g. 1×400 tforms) from "many small fanouts"
    // (e.g. 100×4 tforms) — both can yield the same total but route very
    // differently through scene_manager / accel_manager. Bucket edges
    // chosen to match the typical TF2 prop-cluster sizes.
    uint32_t m_geomDiagFanoutBucket0  = 0;  // tforms == 0 (empty after build, dropped — never reaches SubmitDraw via the gated path, but a few branches accept empty)
    uint32_t m_geomDiagFanoutBucket1  = 0;  // 1
    uint32_t m_geomDiagFanoutBucket4  = 0;  // 2..4
    uint32_t m_geomDiagFanoutBucket16 = 0;  // 5..16
    uint32_t m_geomDiagFanoutBucket64 = 0;  // 17..64
    uint32_t m_geomDiagFanoutBucket256 = 0; // 65..256
    uint32_t m_geomDiagFanoutBucket1k  = 0; // 257..1024
    uint32_t m_geomDiagFanoutBucketBig = 0; // 1025+
    // Per-fanout-batch construction stats so we can see whether tforms
    // got dropped DURING build (OOB t31, non-finite m, zero matrix). The
    // existing build loop already silently `continue`s on these — these
    // counters expose how often.
    uint32_t m_geomDiagFanoutInstSeen      = 0; // total inst slots iterated
    uint32_t m_geomDiagFanoutInstOob       = 0; // t31Off OOB
    uint32_t m_geomDiagFanoutInstBadFinite = 0; // matrix had non-finite element
    uint32_t m_geomDiagFanoutInstZeroRow0  = 0; // m[0..3] all zero (degenerate)
    float    m_geomDiagFanoutMinDist   = 0.0f; // |T| min across all fanouts
    float    m_geomDiagFanoutMaxDist   = 0.0f; // |T| max across all fanouts
    bool     m_geomDiagFanoutHaveDist  = false;
    float    m_geomDiagLastCamAbs[3]   = { 0.0f, 0.0f, 0.0f };
    bool     m_geomDiagHaveCamAbs      = false;

  public:
    // Per-filter rejection reasons tracked for one frame at a time.  Kept
    // public so SubmitDraw can bump them without a friend declaration. The
    // order MUST match the labels in D3D11Rtx::EndFrame below.
    enum class FilterReason : uint32_t {
      Throttle        = 0,
      NonTriTopology  = 1,
      NoPixelShader   = 2,
      NoRenderTarget  = 3,
      CountTooSmall   = 4,
      FullscreenQuad  = 5,
      NoInputLayout   = 6,
      NoSemantics     = 7,
      NoPosition      = 8,
      Position2D      = 9,
      NoPosBuffer     = 10,
      NoIndexBuffer   = 11,
      HashFailed      = 12,
      // NV-DXVK: ExtractTransforms had to use its viewport fallback because
      // no perspective matrix was found in any cbuffer — this is the signal
      // that the draw is 2D UI / overlay / video content (matches D3D9
      // Remix's isRenderingUI() which uses the same "orthographic == UI"
      // heuristic).  Such draws must NOT go through the RTX pipeline: the
      // native DXVK D3D11 rasterizer (which runs unconditionally via EmitCs
      // before m_rtx.OnDraw* in D3D11DeviceContext::Draw*) handles them.
      UIFallback      = 13,
      UnsupPosFmt     = 14,
      // NV-DXVK TF2: character depth-prepass / VSM draws — same skinned VB
      // as the lit pass but the IL omits NORMAL+TEXCOORD (offsets 16..27),
      // so the draw produces no UV stream. Path tracer hits the resulting
      // BLAS instances and renders them flat white because surface material
      // has no albedo. Confirmed by comparing fxc /dumpbin of depth-pass
      // VSes (3ad96dddc6600325, ae99368f58913a2e) vs lit-pass VS
      // (ef94e6c7fcc3c144) — see d3d11_device.cpp dump-target list.
      // Filter signature: POSITION(R32G32_UINT)@0 + BLENDWEIGHT@8 +
      // BLENDINDICES@12 + no TEXCOORD + no NORMAL. Lit-pass adds NORMAL+
      // TEXCOORD so it's distinguishable.
      CharDepthPrepass = 15,
      // NV-DXVK: dropped because rtx.tf2DisableAlphaSurfaces is set and the
      // draw's RT0 blend is enabled (a translucent / alpha-blended surface).
      AlphaSurface    = 16,
      Count           = 17
    };
  private:
    uint32_t m_filterCounts[static_cast<uint32_t>(FilterReason::Count)] = {};
    // NV-DXVK: per-frame o2w path histogram (index = m_lastO2wPathId 0..10).
    // Bumped at COMMIT, dumped + reset in EndFrame.
    uint32_t m_o2wPathCounts[16] = {};
    // NV-DXVK: per-VS-hash o2w path breakdown. Key = VS hash short string.
    // Value[path] = how many COMMITs of that VS used that o2w path this frame.
    // Lets us see e.g. "VS_597b7e49 took t31 32 times, VS_1bcb12cd took cb3
    // 32 times" so we know which hash to disassemble next.
    std::unordered_map<std::string, std::array<uint32_t, 16>> m_vsO2wPathCounts;
    // NV-DXVK: one-shot per-VS RDEF signature dump set (populated as unique
    // VS hashes are seen so we can log cbuffer+SRV layout exactly once each).
    std::unordered_set<std::string> m_vsRdefDumped;

    // NV-DXVK: per-frame VS-hash bookkeeping so EndFrame can dump "this VS was
    // rejected as noPS 42 times, submitted 0 times" — lets us pinpoint which
    // shader category is getting nuked by which filter, no guessing.
    // Extended with skinned/bone classification so we can see which VS hashes
    // are animated-character draws vs static ones, and whether remix processed
    // them. Populated in SubmitDraw and at bone-SRV binding.
    struct VsFrameStats {
      uint32_t submitted = 0;
      uint32_t rejects[static_cast<uint32_t>(FilterReason::Count)] = {};
      uint32_t seen = 0;               // total draw calls observed (all outcomes)
      uint32_t skinnedPerVert = 0;     // has BLENDINDICES0/V (per-vertex bone idx)
      uint32_t skinnedPerInst = 0;     // has BLENDINDICES0/I (per-instance; BSP batched)
      uint32_t boneSrvBound = 0;       // t30 g_boneMatrix SRV was bound
      uint32_t modelInstBound = 0;     // t31 g_modelInst SRV was bound
      std::string firstPsHash;         // first PS hash seen for this VS
    };
    std::unordered_map<std::string, VsFrameStats> m_vsFrameStats;
    // Called instead of ++m_filterCounts[X] — records the current VS hash too.
    void BumpFilter(FilterReason r);
    // Current VS hash cache (set per SubmitDraw entry; empty if no VS).
    std::string m_currentVsHashCache;
    // NV-DXVK [VMHunt]: sticky per-draw flag set by SubmitDraw when count
    // matches a suspect viewmodel index count from PIX. Read by BumpFilter
    // and by COMMIT to emit reject/pass verdict with [VMHunt.result].
    bool m_vmHuntIsSuspect = false;
    uint32_t m_vmHuntIndexCount = 0;

    // NV-DXVK [StudioModelHook]: per-draw BY-MODEL Widow tag. Reset + computed
    // at SubmitDraw entry (from the studiorender draw-site capture slot) when
    // any of tf2HideWidow/tf2IsolateWidow/tf2DetectWidow is enabled; stamped
    // onto DrawCallState::isWidowModel at dcs construction so it reaches the
    // BlasEntry/instance probes.
    bool m_curDrawIsWidow = false;
    // NV-DXVK [StudioModelHook]: name path of the current studiorender draw
    // (NUL-terminated, <=63 chars; empty for non-studio draws). Copied into
    // DrawCallState::studioModelName at dcs construction.
    char m_curStudioName[64] = {};
    // NV-DXVK [SkinName diag]: WHY m_curStudioName is empty for a draw, so the
    // razor probe can report it. 0=resolved, 1=gate off (no name flag on),
    // 2=slot ptr null, 3=*slot==0 (matsys deferred replay = untagged),
    // 4=material name read failed (matPtr live but name offset wrong/null).
    int m_curStudioNameWhy = 1;
    // The live material pointer at resolution time (for the why=4 case so we
    // can fix the name offset for the ship's material type).
    uint64_t m_curStudioMatPtr = 0;

    // NV-DXVK: Set by ExtractTransforms to report whether it had to fall
    // back to a viewport-derived perspective instead of finding a real
    // perspective matrix in a cbuffer.  SubmitDraw uses this as a "this
    // draw is 2D UI / overlay content" signal and skips RTX submission,
    // matching what D3D9 Remix does via isRenderingUI() + orthographicIsUI().
    // Initialized to true so that the EndFrame safety net (which calls
    // ExtractTransforms before any draw on the first frame of a session)
    // correctly treats a never-invoked extract as "no real projection".
    bool                                 m_lastExtractUsedFallback = true;

    // NV-DXVK: When the scanner locks onto a combined VP (cls 3/4), the
    // cached slot/offset must be re-scanned every frame because (a) the VP
    // changes with camera movement, and (b) Source only binds the correct
    // VP cbuffer during the main opaque pass — early draws in the frame
    // (shadow/depth prepass) may have different content in the same slot.
    // This flag is set when the scanner finds a cls 3/4 match and causes
    // m_projSlot to be reset to UINT32_MAX at the top of each EndFrame
    // so the next frame re-scans instead of re-validating the stale location.
    bool                                 m_projIsCombinedVP = false;

    // NV-DXVK: Per-frame flag that becomes true once ANY draw in the
    // current frame successfully finds a real perspective projection
    // (cls 1-4) instead of the viewport fallback.  Once set, ALL
    // remaining draws in the frame bypass the UIFallback filter and
    // reuse the last-found projection — even if THEIR specific
    // ExtractTransforms call would have hit the fallback (because the
    // VP cbuffer isn't populated on early draws like shadow/depth passes).
    //
    // Without this, only draws 250+ in the frame (where the VP cbuffer
    // is bound) pass the filter, and draws 1-249 (real gameplay geometry)
    // are incorrectly rejected as "UI".  With this flag, a single late-
    // frame VP detection unlocks the entire frame.
    //
    // NV-DXVK: Static — shared across all D3D11Rtx instances (immediate +
    // deferred contexts). TF2's materialsystem_dx11 records most BSP
    // draws on deferred contexts that never run the projection-extraction
    // path themselves; they must read the cached w2v saved by the
    // immediate context. Previously these were per-instance, causing
    // every deferred-context draw to hit degenerate_cached_w2v.
    //
    // Access pattern: rare writes (once per successful projection scan),
    // frequent reads (every draw). The rejection check only tests for
    // all-zero translation, so a torn read during a concurrent write
    // either sees the old value, the new value, or a partial update —
    // all of which have non-zero translation once any real proj is
    // latched, so the rejection stays correct. No mutex required.
    static bool                          m_foundRealProjThisFrame;
    static bool                          m_hasEverFoundProj;
    static DrawCallTransforms            m_lastGoodTransforms;
    // Mutex for the three static members above. Deferred-context threads
    // (materialsystem_dx11 records most BSP/prop draws on secondary
    // threads) read m_lastGoodTransforms every draw; the immediate
    // context writes it once per successful projection extraction.
    // Without synchronization, deferred threads can see stale all-zero
    // values indefinitely (CPU cache coherence is eventual, not instant),
    // causing persistent degenerate_cached_w2v rejections.
    static std::mutex                    m_lastGoodTransformsMutex;

    // NV-DXVK [3D-skybox sub-pass tracking via cb2 update sequence]:
    // Every cb2 (CBufCommonPerCamera, BufSize=576) UpdateSubresource is a
    // sub-pass boundary. First valid update of the frame = main pass;
    // subsequent updates with a different origin are non-main sub-passes
    // (3D-skybox composite, shadow probe, etc.). Pure observation — feeds
    // the [subPassUpd]/[subPassDropProbe] logs only.
    uint32_t                             m_subPassFrameId       = UINT32_MAX;
    uint32_t                             m_subPassIndex         = 0;     // 0=first sub-pass this frame
    Vector3                              m_subPassMainOrigin    {0.f,0.f,0.f};
    Vector3                              m_subPassCurrentOrigin {0.f,0.f,0.f};
    bool                                 m_subPassMainOriginValid    = false;
    bool                                 m_subPassCurrentOriginValid = false;

    // NV-DXVK: Current instance index for GPU bone instancing. For the
    // per-instance skinning fanout this is the ABSOLUTE buffer slot
    // (startInstance + instance), so per-instance reads address the right object.
    uint32_t                             m_currentInstanceIndex = 0;
    // NV-DXVK: Set by SubmitInstancedDraw to tell SubmitDraw to attach bone buffers
    bool                                 m_attachBoneBuffers = false;
    uint32_t                             m_boneInstanceCount = 0;

    // NV-DXVK: Async bone transform extraction for 1 BLAS + N TLAS instances.
    // Frame N: compute shader extracts transforms to host-visible buffer.
    // Frame N+1: CPU reads buffer, sets instancesToObject on the draw.
    // Keyed per instanced draw batch (startInstance + instanceCount).
    // Per-draw allocated transforms. Kept alive in a ring buffer by frame
    // so scene manager's instancesToObject pointers stay valid.
    // We keep the last N frames of allocations.
    std::vector<std::vector<std::shared_ptr<std::vector<Matrix4>>>> m_boneTransformRing;
    uint32_t                             m_boneInstFrameId = 0;
    const std::vector<Matrix4>*          m_currentInstancesToObject = nullptr;
    // NV-DXVK: Companion shared_ptr carrying ownership of the storage that
    // m_currentInstancesToObject points at, so the RtInstance consuming it
    // can hold it alive beyond the 4-frame ring buffer's lifetime.
    std::shared_ptr<const std::vector<Matrix4>> m_currentInstancesToObjectOwner;
    // NV-DXVK: Set true during ExtractTransforms for bone draws to skip world matrix scan
    bool                                 m_currentDrawIsBoneTransformed = false;
    // NV-DXVK (TF2 skinned chars): flipped in the skinned-char detection
    // block inside SubmitDraw (RasterGeometry setup), consumed later when
    // `dcs` has been constructed so we can write objectToWorld there.
    // Tells us to override o2w with translate(+fanoutCameraOrigin) so the
    // interleaver's camera-relative skinned positions end up in world space.
    bool                                 m_skinnedCharNeedsCamOffset = false;
    // NV-DXVK: Skip view matrix scan but allow world matrix scan
    bool                                 m_skipViewMatrixScan = false;
    // NV-DXVK TF2: full bone-matrix cache (393216 bytes, 8192 bones × 48).
    // Populated from both D3D11 UpdateSubresource (lower-half palette
    // slots, via OnUpdateSubresource) and DXVK CopyBuffer (full rigs,
    // via the dxvk::tf2::g_boneCacheMirror merge). This replaced the
    // legacy single-bone `m_cachedBone0` / `m_lastBoneBuffer` members
    // which only kept bone 0 — insufficient for any skinned character
    // since TF2 rigs have 60+ bones.
    std::vector<uint8_t>                 m_fullBoneCache;
    bool                                 m_hasFullBoneCache = false;
    // NV-DXVK [perf]: last g_boneCacheMirrorGen this context merged into
    // m_fullBoneCache. Cheap "did ANYTHING change?" gate read atomically so the
    // common (unchanged) path never takes g_boneCacheMirrorMutex at all.
    uint64_t                             m_boneMirrorMergedGen = UINT64_MAX;
    // NV-DXVK [perf]: per-region snapshot of tf2::g_boneCacheMirrorRegionGen at
    // the last merge. A region whose stamp still matches has not been written
    // since, so the merge skips it — see the block comment on kBoneMirrorRegions
    // in dxvk_bone_diag.h. UINT64_MAX seeds a full first merge (stamps start 0).
    uint64_t                             m_boneMirrorRegionMergedGen[::dxvk::tf2::kBoneMirrorRegions];
    // NV-DXVK [perf]: bumped on EVERY write to m_fullBoneCache (mirror merge,
    // UpdateSubresource interception, end-of-frame sweep). Diagnostic write
    // counter only — nothing keys cache validity off it.
    uint64_t                             m_fullBoneCacheGen = 0;

    // Merge the dirty regions of tf2::g_boneCacheMirror into m_fullBoneCache.
    // Takes g_boneCacheMirrorMutex itself; call only when the mirror generation
    // has actually advanced.
    void MergeBoneCacheMirror();

    // NV-DXVK TOMBSTONE [perf]: a per-region generation array on THIS side
    // (m_boneCacheRegionGen + BumpBoneCacheRegions/BoneCacheWindowGen) used to
    // live here to validate an 8-entry cache of CONVERTED Matrix4 bone
    // palettes. It has been deleted, and should not be reinvented: on TF2 the
    // converted palette is never built at all (the interleaver skins GPU-side
    // from the game's own bone buffer; the CPU palette only fed boneHash, which
    // is now taken from the source bytes). The share cache measured builds=0
    // for an entire session — it was ~80 lines of dormant code whose only live
    // effect was the footgun that every new writer of m_fullBoneCache had to
    // remember to bump two different generation counters or serve stale
    // palettes. If a future title DOES need the converted palette, rebuild the
    // cache then, against a measurement showing builds > 0.
    bool                                 m_boneCacheFullNoted = false;
    uint32_t                             m_bonesPerChar = 0; // auto-detected stride

    // NV-DXVK DEBUG: if true, bone-instanced draws run through all the
    // transform math/logging but don't actually submit geometry to RTX.
    // Useful for isolating bone-instanced draws from non-instanced ones.
    bool                                 m_debugHideBoneInstanced = false;

    // NV-DXVK: Cached IMMUTABLE instance buffer data (bone indices).
    // Read once via D3D11 staging copy, reused every frame.
    std::vector<uint8_t>                 m_instBufCache;
    ID3D11Buffer*                        m_cachedInstBufPtr = nullptr; // raw ptr for identity check
    // NV-DXVK [InstStall]: scratch for the sequential bulk-copy of the mapped
    // (write-combined / possibly device-local) t31 per-instance transform buffer.
    // The fanout loop indexes t31 by a scattered charIdx, cold-faulting a page
    // per instance (~35us/instance measured). Copying the buffer once in a
    // sequential, prefetcher-friendly stream lets the per-instance reads hit
    // cached memory. Reused (capacity retained) across draws.
    std::vector<uint8_t>                 m_t31ReadCache;
    // NV-DXVK [CbStage]: staging buffer for the mapped (write-combined)
    // CBufCommonPerCamera read in the BSP fanout path. c_cameraOrigin and the
    // c_cameraRelativeToClip VP rows are read a scalar at a time out of WC
    // memory, which costs an uncached transaction per load; one streaming copy
    // into this makes them cached reads. Capacity retained across draws.
    // Sized by the CB's ByteWidth, which D3D11 caps at 64KB.
    std::vector<uint8_t>                 m_camCbStage;
    // NV-DXVK SESSION-Q: m_pendingRigidBakeO2W/Valid removed. The instanced skinned hull
    // no longer needs a transform override — it is recognised as a skinned char in
    // SubmitDraw and placed via path 11 (objectToWorld=identity), like the non-instanced hull.
    // NV-DXVK: Cached cb3 (CBufModelInstance) objectToCameraRelative float3x4
    // Updated per-draw via UpdateSubresource interception.
    float                                m_cachedCb3[12] = {};
    bool                                 m_hasCachedCb3 = false;

    // NV-DXVK: Cached bone matrix data from t30 (g_boneMatrix).
    // Copied from GPU at end of frame for use on next frame's early draws.
    std::vector<float>                   m_boneMatrixCache;
    bool                                 m_hasBoneMatrixCache = false;
    DxvkBufferSlice                      m_lastBoneSrvSlice;

    // NV-DXVK: One-shot latch for the "dump VS cbuffers on first gameplay
    // frame" diagnostic.  classifyPerspective() isn't recognizing Source's
    // projection matrix layout, so every Titanfall 2 gameplay draw gets
    // rejected as UIFallback.  Dumping the raw first 128 bytes of every
    // bound VS constant buffer once on the first gameplay-sized draw gives
    // us actual evidence of what Source's cbuffer layout looks like so we
    // can extend classifyPerspective to match it.
    bool                                 m_gameplayCBuffersDumped = false;

    // NV-DXVK [CamCache]: per-context cache of the reconstructed camera for the
    // "projection-not-found, R32G32_UINT world geometry" fallback path in
    // ExtractTransforms (d3d11_rtx.cpp ~9639). That path runs for the bulk of
    // TF2 world draws (whose projection the generic scanner can't classify) and
    // re-reads c_cameraOrigin + re-assembles worldToView from the frame-constant
    // fanout VP rows on EVERY draw — ~16ms/frame measured. The result is identical
    // for every draw sharing the same camera cbuffer, so cache it keyed on the
    // cb2 binding identity (buffer ptr + content generation + bound offset) plus
    // the frame id (fanout VP rows are frame state). Non-static = per-context, so
    // deferred recording threads each own their cache (no cross-thread races); the
    // generation read is atomic. A miss just re-derives, so a stale key is at worst
    // a one-draw recompute, never a correctness hazard.
    // Only worldToView is cached. The projection (viewToProjection, which carries
    // FOV) is read fresh from m_lastGoodTransforms on every draw — it is a single
    // matrix copy, not the expensive part — so an FOV change is always picked up
    // immediately, with no assumption about which cbuffer the projection lives in.
    struct CamFallbackCache {
      const void* cb2Buffer  = nullptr;   // D3D11Buffer* identity
      uint64_t    cb2Gen     = UINT64_MAX;
      uint32_t    cb2Offset  = UINT32_MAX; // constantOffset (16-byte units)
      uint32_t    frameId    = UINT32_MAX;
      bool        valid      = false;
      Matrix4     worldToView;
    };
    CamFallbackCache                     m_camFallbackCache;

    // Cached projection cbuffer location — found on first draw with a perspective
    // matrix and reused for the rest of the frame. Reset to invalid in EndFrame.
    uint32_t                             m_projSlot   = UINT32_MAX;
    size_t                               m_projOffset = SIZE_MAX;
    int                                  m_projStage  = -1;
    // true when the engine stores matrices in column-major order (Unity, Godot).
    // Detected during the projection scan — all subsequent reads are transposed.
    bool                                 m_columnMajor = false;

    // Cached view matrix cbuffer location — mirrors projection caching.
    // Once a valid view matrix is found at (stage, slot, offset), subsequent
    // draws re-read from the same location instead of rescanning.
    uint32_t                             m_viewSlot   = UINT32_MAX;
    size_t                               m_viewOffset = SIZE_MAX;
    int                                  m_viewStage  = -1;

    // NV-DXVK [Phase1]: formal per-VS layout table (GPU-driven injection).
    // Populated at the tail of ExtractTransforms from the same resolved state
    // the legacy caches produce, then consumed by the Phase-2 capture record
    // via m_currentLayoutId. Behavior-neutral in Phase 1 (write + verify only;
    // nothing reads the table to alter rendering yet).
    VsLayoutTable                        m_vsLayoutTable;
    // layoutId of the draw ExtractTransforms just resolved, or UINT32_MAX when
    // the draw is UI-fallback / not injected. Reset per draw; read downstream.
    uint32_t                             m_currentLayoutId = UINT32_MAX;

    // NV-DXVK [Phase2]: per-frame capture arena (GPU-driven injection). Filled
    // at the RT commit point when rtx.capturePhase2 is on; consumed by Phase 3.
    // Cleared (capacity retained) at each frame boundary. Behavior-neutral.
    std::vector<DrawCaptureRecord>       m_captureArena;
    uint32_t                             m_captureFrame = UINT32_MAX; // frame the arena holds

    // Smoothed camera position — exponential moving average dampens
    // micro-jitter from floating-point rounding in cbuffer matrix extraction.
    Vector3                              m_smoothedCamPos = Vector3(0.0f);
    bool                                 m_hasPrevCamPos  = false;

    // Axis convention auto-detection — voting system accumulates evidence
    // from projection and view matrices, then settles once confident.
    // Re-checks during warmup to correct boot/loading screen misdetections.
    bool                                 m_axisDetected = false;
    bool                                 m_axisLogged   = false;
    uint32_t                             m_axisDetectFrame = 0;

    // Voting counters for Z-up vs Y-up and LH vs RH.
    // Accumulate votes over multiple frames, settle when |votes| >= threshold.
    int                                  m_zUpVotes     = 0;  // positive = Z-up, negative = Y-up
    int                                  m_lhVotes      = 0;  // positive = LH, negative = RH
    int                                  m_yFlipVotes   = 0;  // positive = flipped, negative = normal
    bool                                 m_zUpSettled    = false;
    bool                                 m_lhSettled     = false;
    bool                                 m_yFlipSettled  = false;
    static constexpr int kVoteThreshold  = 5; // votes needed to settle
    mutable Rc<DxvkSampler>              m_defaultSampler;

    Rc<DxvkSampler> getDefaultSampler() const;
    // NV-DXVK [SkyAutoCb2]: read c_cameraOrigin from the bound VS's cb2
    // (CBufCommonPerCamera) and, if the value matches the per-frame
    // first-observed origin, set InstanceCategories::Sky on the draw.
    // Routes the draw to RtxContext::tryHandleSky, which under
    // SkyMode::PhysicalAtmosphere drops the geometry submission and
    // lets the Hillaire atmospheric LUT pipeline render the sky instead.
    // Returns true if sky was set on dcs.
    bool SetSkyCategoryFromCb2(DrawCallState& dcs);
    // NV-DXVK [ShipHunt v2]: one-shot discovery probe that logs the first
    // appearance of every distinct (VS hash, viewport width) tuple seen
    // in this session.
    //
    // v2 fix vs v1: called from the VERY TOP of SubmitDraw, BEFORE any
    // filter cascade can early-return. v1 was called after
    // SetSkyCategoryFromCb2 (~line 14968), past the shadow-pass / no-RT /
    // count<3 / HUD-class filters — which silently dropped every
    // shadow-cascade draw before logging. The known ship VS_597b7e49 at
    // vp=2048×2048 was visible in the upstream [fanoutCBRead] log but
    // entirely absent from [ShipHunt.firstSeen] for exactly this reason.
    // The new placement catches every draw that reaches SubmitDraw, full
    // stop.
    //
    // No dcs parameter: DrawCallState isn't constructed until ~line 11492.
    // Sky classification (sky=0/1) was dropped from this probe — it's
    // already covered by the existing per-distinct-origin
    // [SkyAutoCb2.classify] log with the verdict field, which is the
    // authoritative source for that signal.
    //
    // Cheap on the per-draw hot path: pulls VS/PS pointers + viewport
    // (already in m_context->m_state, no FindCBField, no cb mapping),
    // hashes into a (vs_hash, vp_width) key, takes a single short
    // mutex to check a session-lifetime unordered_set. Once a key has
    // been seen the set lookup is O(1) and the function returns false
    // without logging — steady-state cost is ~100ns/draw.
    //
    // Returns true if this call produced a log line (first sighting).
    bool LogShipHuntDiscovery();
    // NV-DXVK [EngineSunCapture]: probe the bound VS/PS cbuffers for fields
    // that hold the engine's per-frame sun direction and colour, and
    // publish them via publishEngineSunCapture() so RtxAtmosphere can
    // override the slider-driven Hillaire sun. Cheap to call per-draw:
    // once a (cb, field) pair is latched on first match, future calls
    // resolve straight via FindCBField with no string compares.
    // Returns true if a snapshot was published this call.
    bool CaptureEngineSunFromCb(DrawCallState& dcs);
    // NV-DXVK [SkyProbe.cubeRender]: snapshots cb2 (CBufCommonPerCamera)
    // contents + matrix/origin offsets into dcs.skyProbeCubeCapture, so
    // RtxContext::rasterizeToSkyProbe can later run TF2's sky shader 6
    // times with cube-face View×Projection overrides. Resolves dxvk-side
    // cb slot via computeConstantBufferBinding, field offsets via the
    // same FindCBField path as the sun/sky-tint capture. Returns true on
    // successful snapshot.
    bool CaptureSkyProbeCubeFromCb(DrawCallState& dcs);
    // NV-DXVK [EngineLightsCapture]: Tier 2 discovery dump for the
    // dynamic light array. Reads s_globalLights structured buffer
    // when bound on the active PS, logs the first few elements as
    // float4s. Returns true if a dump fired this call.
    bool DumpEngineLightsBufferFromSrv();
    // NV-DXVK [EngineLightsCapture]: one-shot per-type field statistics
    // (min/max per component, constancy flags) so the user can see
    // exactly what each unknown vec4 slot encodes.
    void DumpEngineLightFieldStats();
    // NV-DXVK [EngineLightsCapture]: convert mirrored s_globalLights
    // entries to RtxLegacyLight and submit via RtxContext::addLights.
    // Throttled to once per frame internally - safe to call from the
    // per-draw fanout point.
    void SubmitEngineLights();
    void SubmitDraw(bool indexed, UINT count, UINT start, INT base,
                    const Matrix4* instanceTransform = nullptr);
    // NV-DXVK [engine-post forward]: if the current draw is the host game's
    // final post-process composite (binds CBufEnginePost), harvest its
    // parameters into Remix's post pipeline (bloom/exposure via setDeferred,
    // tonemap/CC/DoF via EnginePostState) and return true so the caller drops
    // the draw instead of injecting it as scene geometry. Returns false (and
    // does nothing) when the gate is off or the draw is not the post pass.
    bool HarvestEnginePostAndForward();
    void SubmitInstancedDraw(bool indexed, UINT count, UINT start, INT base,
                             UINT instanceCount, UINT startInstance);
    DrawCallTransforms ExtractTransforms();
    Future<GeometryHashes> ComputeGeometryHashes(const RasterGeometry& geo,
                                                 uint32_t vertexCount,
                                                 uint32_t hashStartVertex,
                                                 uint32_t hashVertexCount) const;
    // NV-DXVK [MatDefer]: FillMaterialData now reads ALL live D3D11 state through
    // an injected MatSnapshot instead of m_context, so the identical body runs
    // either synchronously (snapshot references/copies live state) or on a geometry
    // worker (snapshot owns pinned copies). captureMatSnapshot builds it; pass
    // deferForWorker=true to pin dynamic constant buffers so their mapPtrs survive.
    void captureMatSnapshotInto(MatSnapshot& s, bool deferForWorker) const;
    void FillMaterialData(LegacyMaterialData& mat, const MatSnapshot& snap) const;
    // Cheap synchronous computation of the two material fields the game thread reads
    // before EmitCs (sourceIsUnlitUI, blendMode). Called at the SubmitDraw call site
    // so those are valid even when the full FillMaterialData defers to a worker.
    void hoistSyncMaterialFields(LegacyMaterialData& mat) const;
  };

}
