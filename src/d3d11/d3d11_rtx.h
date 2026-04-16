#pragma once

#include "d3d11_include.h"
#include <array>
#include <unordered_map>
#include <unordered_set>

#include "../dxvk/rtx_render/rtx_types.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../dxvk/rtx_render/rtx_materials.h"
#include "../dxvk/dxvk_buffer.h"
#include "../util/util_matrix.h"
#include "../util/util_threadpool.h"

namespace dxvk {

  class D3D11DeviceContext;

  class D3D11Rtx {
  public:
    explicit D3D11Rtx(D3D11DeviceContext* pContext);

    void Initialize();
    // Returns true if the draw was captured for RT (caller should skip rasterization).
    bool OnDraw(UINT vertexCount, UINT startVertex);
    bool OnDrawIndexed(UINT indexCount, UINT startIndex, INT baseVertex);
    bool OnDrawInstanced(UINT vertexCountPerInstance, UINT instanceCount, UINT startVertex, UINT startInstance);
    bool OnDrawIndexedInstanced(UINT indexCountPerInstance, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance);

    // NV-DXVK: Intercept UpdateSubresource to cache bone matrix data from t30.
    // Called from D3D11DeviceContext::UpdateSubresource before the data goes to GPU.
    void OnUpdateSubresource(ID3D11Resource* pDstResource, const void* pSrcData, UINT SrcDataSize, UINT DstOffset = 0, UINT BufSize = 0);

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
    void resetDrawCallID() { m_drawCallID = 0; }
    void addDrawCallID(uint32_t count) { m_drawCallID += count; }

  private:
    static constexpr uint32_t kMaxConcurrentDraws = 6 * 1024;
    using GeometryProcessor = WorkerThreadPool<kMaxConcurrentDraws>;

    D3D11DeviceContext*                  m_context;
    std::unique_ptr<GeometryProcessor>   m_pGeometryWorkers;
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
    // True once ANY draw in the current frame was captured for RT.
    // Once Remix is active, ALL D3D11 rasterization is suppressed (including
    // filtered draws) because the game's native rasterization shares render
    // targets with Remix output → write hazards → corruption → TDR.
    // Reset to false each EndFrame. During menus (no RT captures), this stays
    // false and all draws rasterize normally.
    bool                                 m_remixActiveThisFrame = false;
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

    // NV-DXVK: Per-frame bone instancing stats
    uint32_t                             m_boneInstBatches = 0;
    uint32_t                             m_boneInstTotal = 0;
    uint32_t                             m_boneInstSkipped = 0;
    uint32_t                             m_boneInstNoCache = 0;
    uint32_t                             m_boneInstCacheHits = 0;
    uint32_t                             m_boneInstCacheMisses = 0;
    std::unordered_set<uintptr_t>        m_boneInstVbPtrs;  // unique VB ptrs this frame

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
      Count           = 15
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
    struct VsFrameStats {
      uint32_t submitted = 0;
      uint32_t rejects[static_cast<uint32_t>(FilterReason::Count)] = {};
    };
    std::unordered_map<std::string, VsFrameStats> m_vsFrameStats;
    // Called instead of ++m_filterCounts[X] — records the current VS hash too.
    void BumpFilter(FilterReason r);
    // Current VS hash cache (set per SubmitDraw entry; empty if no VS).
    std::string m_currentVsHashCache;

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
    // Reset to false at the top of each EndFrame.
    bool                                 m_foundRealProjThisFrame = false;
    // Latched true forever once ANY frame finds a real projection.
    // Allows early draws in subsequent frames to reuse the cached VP
    // even before the current frame's late-draw detection fires.
    bool                                 m_hasEverFoundProj = false;
    // Cached transforms from the last draw that found a real projection,
    // reused for draws that would otherwise hit the UIFallback.
    DrawCallTransforms                   m_lastGoodTransforms;

    // NV-DXVK: Current instance index for GPU bone instancing
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
    // NV-DXVK: Cached bone 0 matrix from t30, updated on every UpdateSubresource to t30
    float                                m_cachedBone0[12] = {};
    bool                                 m_hasCachedBone0 = false;
    ID3D11Buffer*                        m_lastBoneBuffer = nullptr;
    // NV-DXVK: Full bone matrix cache from t30 — intercepted in OnUpdateSubresource
    std::vector<uint8_t>                 m_fullBoneCache;
    bool                                 m_hasFullBoneCache = false;
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
    void SubmitDraw(bool indexed, UINT count, UINT start, INT base,
                    const Matrix4* instanceTransform = nullptr);
    void SubmitInstancedDraw(bool indexed, UINT count, UINT start, INT base,
                             UINT instanceCount, UINT startInstance);
    DrawCallTransforms ExtractTransforms();
    Future<GeometryHashes> ComputeGeometryHashes(const RasterGeometry& geo,
                                                 uint32_t vertexCount,
                                                 uint32_t hashStartVertex,
                                                 uint32_t hashVertexCount) const;
    void FillMaterialData(LegacyMaterialData& mat) const;
  };

}
