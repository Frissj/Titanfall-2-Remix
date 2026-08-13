#pragma once

#include "d3d11_include.h"
// NV-DXVK [DrawSnapshot]: DrawSnapshot stores D3D11VertexBufferBinding /
// D3D11IndexBufferBinding BY VALUE and holds Com<> refs to the shader, input
// layout and SRV types, so it needs their full definitions -- forward
// declarations are not enough. Safe to include here: d3d11_context_state.h
// pulls in buffer/shader/state/view headers and references nothing in this
// file, so there is no cycle.
#include "d3d11_context_state.h"
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

  // NV-DXVK [Perf.StageDep]: the carrier GROUPS the stage-dependency census
  // hashes separately. Namespace scope, not a class member: the census's free
  // helpers in d3d11_rtx.cpp need it, and the grouping is a property of the
  // probe rather than of D3D11Rtx. See rtx.perfStageDepCensus.
  // MOVED HERE 2026-08-10 from just above class D3D11Rtx: DrawSnapshot now
  // stores one hash PER GROUP (see DrawSnapshot::carrierGrp), so
  // kSdepGroupCount has to be visible before the struct. Nothing else changed.
  enum SdepGroup : uint32_t {
    kSdepCam = 0,   // camera origins / fanout latches / smoothing
    kSdepRoute,     // the sticky path ids
    kSdepCbLoc,     // shared proj/view cb slot+offset locations
    kSdepBone,      // bone + cb3 caches a later draw reuses
    kSdepStatic,    // m_foundRealProjThisFrame + m_lastGoodTransforms
    // SPLIT OUT OF kSdepCam 2026-08-12, and APPENDED rather than inserted so
    // every existing group index -- and therefore every replay-mask bit value
    // already written into rtx.conf -- stays what it was.
    //
    // WHY IT IS ITS OWN GROUP. kSdepCam was made replayable from the VIEW entry
    // on the argument that the view cache is per-frame, so a recording cannot
    // go stale. That is true of the LATCHES and false of m_smoothedCamPos,
    // which is an ACCUMULATOR: its exit value is a function of its own previous
    // value, so it depends on the SEQUENCE of draws and not merely on the state
    // the draw entered with. Being same-frame is not enough. Measured:
    // REPLAYFAIL grp{cam=1..29} per window against ~7-10k replay serves, plus
    // residual FAIL fld{o2v w2v v2p} where the replayed camera perturbed the
    // view derivation.
    //
    // Splitting it lets the latches stay replayable while the accumulator goes
    // back to refusing the store, which is what it always needed.
    kSdepCamSmooth, // m_smoothedCamPos + m_hasPrevCamPos (accumulator)
    kSdepGroupCount
  };

  // ==================================================================
  // NV-DXVK [DrawSnapshot] 2026-08-10 -- an owned, immutable per-draw record of
  // every D3D11 pipeline input the RT derivation reads.
  //
  // WHY THIS EXISTS. SubmitDraw's derivation stages read LIVE context state --
  // ~460 read sites across 24 distinct m_context->m_state paths -- so they can
  // only run on the frame thread, in draw order, before the game rebinds for
  // the next draw. That is the ONLY reason the derivation is serial. It is also
  // why the cross-draw carrier census (rtx.perfStageDepCensus) had to exist:
  // once per-draw work reads shared mutable state, proving any two draws
  // independent becomes an analysis problem instead of a property of the data.
  //
  // A production renderer does not have this problem, because the dependency is
  // inverted. UE's FMeshDrawCommand (Engine/Source/Runtime/Renderer/Public/
  // MeshPassProcessor.h:1281) owns its shader bindings, vertex streams, index
  // buffer, PSO id and counts, and references no context at all -- which is
  // precisely what lets FMeshDrawCommandPassSetupTask build, sort and merge
  // commands on a worker, with ordering carried by a sort key plus an explicit
  // index array rather than by thread joins. DrawSnapshot is that record for
  // this translation layer: capture once at the draw entry point, then every
  // derivation stage becomes a pure function of it.
  //
  // THE RULE THAT MAKES IT CORRECT -- READ BEFORE ADDING A FIELD.
  // Com<D3D11Buffer> pins the buffer OBJECT, not its CONTENTS. A dynamic buffer
  // that the game Map(WRITE_DISCARD)es keeps the same D3D11Buffer while its
  // backing slice is renamed underneath, so a pinned ref read later yields
  // whatever the game wrote NEXT. That is the dropship COLOR1.y race and the
  // getImageHash streaming race, both already paid for in this project.
  // Therefore:
  //   - if the derivation reads a resource's BYTES, those bytes are COPIED here
  //     at capture time, on the frame thread, while the binding is still live;
  //   - a Com<> ref is permitted ONLY for identity (map keys, hashes, pointer
  //     compares) and lifetime, never as a promise about contents.
  // Immutable-after-creation objects (shader blobs, input layouts, and the
  // reflection data behind GetCommonShader) are safe to hold by ref: their
  // contents cannot change while the ref is held.
  //
  // STAGING. Snapshots live in a per-frame arena that is cleared, never freed,
  // so capture costs no allocation -- the same reason DrawWorkItem is a
  // reserved vector (per-draw heap traffic is what made the per-draw material
  // futures break even). The byte capture is bounded and reports overflow
  // rather than silently truncating, because a missed range is a WRONG RESULT,
  // not a slow one.
  // ==================================================================
  struct DrawSnapshot {
    // Bounded inline byte capture. 512 B covers the ranges the derivation
    // actually reads today (cb2@16 / cb2@96 projection-view blocks at 64 B,
    // the cb3 object-to-world block at 48 B, and headroom). Bone palettes are
    // deliberately NOT here -- they already have their own deferred capture
    // (BatchSkinJob), which copies rather than pins for this same reason.
    // RAISED 2026-08-10 with the per-VS span manifest. The four NAMED spans
    // (proj 64 / view 64 / cameraOrigin 12 / cb3 48) are joined by up to
    // kMaxCbManifest LEARNED spans -- the offsets a VS's derivation was
    // observed to read live -- so the ceiling has to clear 4 + 6 with margin.
    // Sizing it tight would be actively harmful, not merely wasteful:
    // `overflowed` invalidates the WHOLE record (drawSnap() returns nullptr),
    // so one span too many sends the draw fully live, identity accessors
    // included. Bytes: 192 named + up to 6 x 64 learned = 576, under 768.
    static constexpr uint32_t kMaxCbRanges = 12u;
    static constexpr uint32_t kCbBytesCap  = 768u;

    struct CbRange {
      uint32_t stage      = 0;  // 0 = VS, 1 = PS, 2 = GS, 3 = DS
      uint32_t slot       = 0;  // cbuffer slot within that stage
      uint32_t byteOffset = 0;  // offset within the BOUND range, matching the
                                //   live-path reads (constantOffset * 16 + n)
      uint32_t byteCount  = 0;
      uint32_t dataOffset = 0;  // where the bytes start in `cbBytes`
    };

    // -- ordering. drawIndex IS the sort key: it is the draw's position in the
    // frame's submission order, so results can be applied in order after any
    // amount of out-of-order derivation. UE carries this as
    // FMeshDrawCommandSortKey + the VisibleMeshDrawCommands index array.
    uint32_t frameId   = 0;
    uint32_t drawIndex = 0;

    // -- identity / immutable-by-construction. Safe as refs (see the rule).
    Com<D3D11VertexShader> vs          = nullptr;
    Com<D3D11PixelShader>  ps          = nullptr;
    Com<D3D11InputLayout>  inputLayout = nullptr;

    // -- raw state-object pointers, used ONLY for identity and for decoding
    // blend/depth/raster descriptions. Not owned: these live as long as the
    // device and the existing code already reads them bare.
    D3D11BlendState*        blendState = nullptr;
    D3D11DepthStencilState* depthState = nullptr;
    D3D11RasterizerState*   rasterState = nullptr;

    // -- geometry stream bindings. Held by ref for IDENTITY and LIFETIME only:
    // the vertex/index CONTENTS are consumed downstream on the CS thread via
    // the existing capture path, not by the derivation.
    D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    std::array<D3D11VertexBufferBinding,
               D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT> vertexBuffers = { };
    D3D11IndexBufferBinding                               indexBuffer   = { };

    // -- SRVs. Identity only (t30/t31 bone + instance streams are resolved by
    // slot, and their bytes go through the staging path).
    std::array<Com<D3D11ShaderResourceView>,
               D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT> vsSrvs = { };

    // NV-DXVK [perf] 2026-08-11: which vsSrvs slots THIS RECORD currently holds
    // a ref in. 128 bits, one per slot.
    //
    // NOT an optimisation hint -- it is what makes the narrow update CORRECT.
    // Arena slots are reused by index and are NOT reset() (see the acquisition
    // comment in captureDrawSnapshot): reuse is safe only because every field
    // is unconditionally reassigned. The old `vsSrvs = views` wholesale assign
    // satisfied that by overwriting all 128 slots, including nulling the ones
    // this draw does not bind. Copying only the bound slots would leave the
    // PREVIOUS occupant's refs behind in the rest, and drawVsSrv(30) would hand
    // a consumer another draw's SRV -- the silent-carry class that comment
    // exists to prevent.
    //
    // So the capture walks the union of (slots this record holds) and (slots
    // the draw binds): assign the bound ones, clear the stale ones, skip the
    // ~126 that are null in both. Measured motivation: dynSrvSlots{t4 t31} --
    // exactly two slots of 128 ever carry a dynamic buffer.
    uint32_t vsSrvMask[4] = { };

    // -- VS constant-buffer BINDINGS. Identity and generation, never contents.
    //
    // This is the distinction the "cbuffers are the barrier" summary hides.
    // The 32 remaining live cbuffer reads are not one problem, they are two:
    //   (a) BINDING identity -- which buffer, at what offset. Several sites
    //       read only this, as cache keys (the camera-fallback cache compares
    //       cb2Buffer / cb2Gen / cb2Off). Pure, cheap, and captured here.
    //   (b) CONTENTS -- the scans. Those cannot be made pure by a narrow copy
    //       and are still the real blocker; see the option doc.
    //
    // contentGen IS LOAD-BEARING FOR DEFERRAL, not a convenience. It moves only
    // in DiscardSlice(), i.e. only when the backing slice is RENAMED -- exactly
    // the dropship COLOR1.y failure, where a Map(WRITE_DISCARD) swapped the
    // bytes under a pointer that still compared equal. A consumer running off
    // the frame thread cannot ask "are these still my draw's bytes?" by reading
    // the buffer, because by then the answer has already changed. Captured
    // here, at the instant the binding was this draw's, it can.
    struct CbBinding {
      D3D11Buffer* buffer         = nullptr;  // IDENTITY ONLY -- never read bytes through this
      uint32_t     constantOffset = 0;
      uint32_t     constantBound  = 0;
      uint64_t     contentGen     = UINT64_MAX;  // sentinel = nothing bound
    };
    std::array<CbBinding,
               D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT> vsCbs = { };

    // -- render target 0. PRESENCE ONLY, and only slot 0: the two derivation
    // sites that read it both just ask "is anything bound here" -- one sets a
    // route bit in the replay key, the other is a depth-only-pass test. Stored
    // as a bare pointer because the state itself holds these as
    // Com<D3D11RenderTargetView, false>, i.e. already non-owning, and because
    // nothing here ever dereferences it.
    D3D11RenderTargetView* rtv0 = nullptr;

    // -- viewport state. Plain POD, copied whole: the viewport fallback path
    // reads it to synthesise a projection when no real one is found.
    uint32_t numViewports = 0;
    std::array<D3D11_VIEWPORT,
               D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE>
                                                        viewports = { };

    // -- THE COPIED BYTES. Everything above is identity; this is content.
    uint32_t                                 numCbRanges = 0;
    std::array<CbRange, kMaxCbRanges>        cbRanges    = { };
    // alignas(16) is LOAD-BEARING, not tidiness: consumers reinterpret_cast the
    // returned pointer to const float* / Matrix4 exactly as they do with the
    // staging-ring pointer they replace. std::array<uint8_t,N> is 1-byte
    // aligned by default, which would make those casts unaligned reads. The
    // capture also rounds each range's dataOffset up to 16 so every span inside
    // stays aligned regardless of the preceding span's size.
    alignas(16) std::array<uint8_t, kCbBytesCap> cbBytes = { };

    // THE DEFERRAL PARTITION PREDICATE. True when this draw's VS had a resolved
    // projection/view location in sVsCbLocCache at capture time.
    //
    // WHY THIS IS THE PREDICATE, MEASURED 2026-08-10. The stated blocker on
    // deferring the derivation is that it SCANS live cbuffers -- the offset-64
    // projection scan, the cross-stage all-cb scan, the m_viewSlot hunt -- and a
    // scan cannot be made pure by a narrow copy. That is true of the code and
    // empirically vacuous in steady state: over the 03:18 windows the scans
    // fired ZERO times (xt_projScan1N=0, pv_rescanN=0) because the per-VS
    // location cache never missed (xt_vsLocMiss=0 against xt_vsLocHits=5849).
    // The scan is guarded by `projSlot == UINT32_MAX && !skipExpensiveProjScan`,
    // and a resolved VS sets projSlot from the cache before that test.
    //
    // So the scans are a COLD-START path, and the question for deferral is not
    // "how do we make a scan pure" but "can we tell, before deriving, that this
    // draw will not take one". This flag is that answer, and it is free: the
    // capture already does the sVsCbLocCache lookup to decide which spans to
    // copy. It WILL be false for the cases that genuinely scan -- a VS's first
    // sighting, after the 4096-entry cache clear, after a neg-cache TTL expiry.
    //
    // NOT A PURITY CERTIFICATE. It says this draw skips the scans. It does NOT
    // say the derivation is a pure function of the snapshot: the content reads
    // still go through stagedCbBytes()/GetMappedSlice(), which read live buffer
    // bytes at consume time, and the m_last* carriers are a separate axis. Use
    // it to PARTITION, then prove purity the way the tier already proves it
    // (FAIL=0). Deferring on this flag alone repeats the COLOR1.y mistake.
    bool layoutResolved = false;

    // THE PURITY CERTIFICATE. Written at the END of the derivation, not at
    // capture, because it is a statement about what the derivation DID.
    //
    // layoutResolved says the draw skips the cold-start scans. It deliberately
    // does NOT say the derivation is a pure function of this record -- the
    // handoff calls that out and it was the remaining barrier: the content
    // reads still went through stagedCbBytes()/GetMappedSlice(), which read
    // LIVE buffer bytes at consume time. Deferring on layoutResolved alone
    // repeats the dropship COLOR1.y mistake.
    //
    // Every content read now goes through D3D11Rtx::drawCbSpan(), which counts
    // itself into exactly one of these two. So:
    //   cbLiveReads == 0 && cbRecordReads > 0
    // means this draw's derivation read cbuffer CONTENT only from bytes copied
    // at capture time, under the captured generations. That is the fact a
    // deferred reader needs and the fact no flag could assert before.
    //
    // NOT SUFFICIENT ON ITS OWN, and the gap is named rather than papered
    // over: this covers cbuffer CONTENT. The m_last* cross-draw carriers are a
    // separate axis (the StageDep census measures those), and vertex/index/
    // bone bytes go through their own capture path. A deferral gate is the
    // CONJUNCTION of all three, and it must still be proven the way the tier
    // proves everything else -- FAIL=0.
    uint16_t cbLiveReads   = 0;
    uint16_t cbRecordReads = 0;

    // THE SECOND AXIS: CROSS-DRAW CARRIERS.
    //
    // WHY THIS IS A WRITE TEST AND NOT A READ TEST -- the thing that took a
    // wrong turn before it took the right one. The obvious move is to capture
    // the carriers into this record the way the cbuffer bindings are captured,
    // and serve the derivation's ~99 carrier reads from the copy. That does
    // NOT make deferral correct, and the reason is worth writing down: the
    // carriers are written BY the derivation. Draw N writes
    // m_lastFanoutCamOrigin; draw N+1 reads it. If both are deferred, N+1's
    // capture runs on the frame thread at its own SubmitDraw entry, BEFORE N's
    // worker has produced that write -- so N+1 captures a stale carrier and the
    // deferred result differs from the serial one. Capturing converts a race
    // into a deterministic divergence, which is an improvement but is not
    // correctness.
    //
    // The sufficient condition is simply that the draw MOVES no carrier. Then
    // there is no ordering edge into any later draw, and reading them live or
    // from a copy is the same answer. [Perf.StageDep] already measured this:
    // bt_extractXf is the ONLY stage that moves cross-draw D3D11Rtx state, at
    // wrote=14% / wroteElig=13%, and all 47 others read 0%. So ~86% of draws
    // are already carrier-free and the question is only which ones.
    //
    // carrierGrp holds the per-group fingerprints at capture; carrierMask is
    // set at the end of the derivation by re-hashing and comparing group by
    // group. The enumeration is stageDepCarrierGroups(), which is the tree's
    // existing, reviewed carrier list -- deliberately reused rather than
    // re-derived, because a carrier missing from that list is invisible to
    // both this and the census.
    //
    // CONSERVATIVE BY CONSTRUCTION: the capture fingerprint is taken at
    // SubmitDraw entry, a few stages EARLIER than the derivation. Those stages
    // measure wrote=0%, but if one ever did move a carrier this would report
    // a carrier move on a draw whose derivation was innocent -- a false
    // positive, which costs a deferral opportunity. It can never produce a
    // false negative, which is the only direction that would be unsafe.
    // MEASURED 2026-08-10 and the concern is empty: [Perf.StageDep] reads
    // wrote=0% on all 47 stages other than bt_extractXf, so the wide baseline
    // costs nothing today. Left wide anyway -- it is the safe direction, and
    // narrowing it buys zero draws.
    //
    // ONE HASH PER GROUP, NOT ONE FOLDED HASH -- changed 2026-08-10.
    // The fold could only ever say "something moved", and the number it
    // produced (carrierMoved=13% of draws, 19% of the draws that pass axis 1)
    // is not actionable, because the five groups are not the same KIND of
    // dependency:
    //
    //   kSdepCam    a real draw-N -> draw-N+1 data flow: N writes
    //               m_lastFanoutCamOrigin, N+1 reads it. A true ordering edge,
    //               and the case the paragraph above is about. ~780 draws per
    //               3 s window, and after the cbLoc work below it is now
    //               essentially the whole carrier axis.
    //   kSdepBone   bone/cb3 caches a later draw reuses. Measured at ZERO
    //               moves per window, so it costs nothing either way.
    //   kSdepStatic m_lastGoodTransforms + m_foundRealProjThisFrame: a
    //               last-known-good FALLBACK latch, order-affecting but not
    //               consumed as this draw's answer. ~65 draws/window.
    //   kSdepCbLoc  88% of all carrier moves (~5,600/window vs cam's 780) and
    //               the single largest eligibility cost at ~3,750 draws per
    //               3 s window. STILL HASHED: it is a real dependency, and two
    //               attempts to retire it (sentinel seed; frame-scoped hint)
    //               were measured and reverted -- the second broke the replay
    //               tier with FAIL=1. See stageDepCarrierGroups.
    //   kSdepRoute  hashes nothing by construction; see stageDepCarrierGroups.
    //
    // safeToDefer() still requires ALL of them clean, byte for byte the same
    // gate as before this change. The mask exists so the NEXT decision --
    // whether a per-group gate that trusts the caches is worth building -- is
    // made from a counter rather than from this comment. DO NOT relax the gate
    // on the strength of the reasoning above; relax it on the strength of
    // [DrawPure] eligCacheOk, which is exactly the population such a gate
    // would add.
    //
    // STRICTLY SAFER THAN THE FOLD, not merely equivalent: two different group
    // vectors could fold to the same 64 bits and report clean. Comparing per
    // group removes that false negative and cannot introduce one.
    uint64_t carrierGrp[kSdepGroupCount] = {};
    // Bit g set == group g's hash differs between capture and derivation exit.
    uint8_t  carrierMask = 0;
    static_assert(kSdepGroupCount <= 8, "carrierMask is 8 bits wide");

    bool wroteCarrier() const { return carrierMask != 0; }

    bool contentPure() const {
      return cbLiveReads == 0 && cbRecordReads > 0;
    }

    // ==================================================================
    // AXIS 3 -- VERTEX / INDEX / BONE BYTES. The third of the three things
    // deferral needs, and until now the one nothing in this record spoke for.
    //
    // WHY IT IS A DIFFERENT QUESTION FROM AXIS 1. Axis 1 is about cbuffer
    // CONTENT and is answered by counting reads through drawCbSpan. These bytes
    // never go through that accessor: they reach the RT side as Rc<DxvkBuffer>
    // slots on RasterGeometry, and they are consumed by jobs that ALREADY run
    // on a worker -- runBatchHashJob, runBatchBboxJob, runBatchSkinJob in
    // flushGeometryBatch. So the bytes are read off-thread today; the question
    // is only whether what they read can change underneath them.
    //
    // THE ONE THING THAT MAKES IT UNSAFE is the rule at the top of this struct:
    // a Com<>/Rc<> pins the buffer OBJECT, not its CONTENTS. A buffer created
    // D3D11_USAGE_DYNAMIC is Map(WRITE_DISCARD)ed by the game, which RENAMES
    // its backing slice while the object identity stays put -- so a pin taken
    // at SubmitDraw entry and read at frame end yields whatever the game wrote
    // NEXT, not what this draw drew. That is the dropship COLOR1.y race
    // exactly, and it is why this is a per-draw fact and not a global claim.
    // IMMUTABLE / DEFAULT / STAGING buffers cannot be renamed that way, so a
    // pin on one of them IS a promise about contents.
    //
    // WHAT THESE FIELDS SAY. Which of this draw's geometry inputs are backed by
    // a DYNAMIC buffer. They do NOT say the draw is unsafe: a dynamic buffer
    // whose bytes were COPIED (the index snapshot, the t31 read cache, the bone
    // palette materialisation) is fine, and those copies are exactly what the
    // existing jobs' copy-vs-pin rules are for. What they give is the
    // POPULATION -- the draws where the copy-vs-pin question has to be asked at
    // all -- measured rather than assumed.
    //
    // DELIBERATELY NOT WIRED INTO safeToDefer() YET. Publishing the fact and
    // reporting it comes first, the same order axis 2 went in: the carrier mask
    // was measured for a run before anything acted on it, and that discipline
    // is what caught the cbLoc dependency instead of shipping it. Wire it in
    // once [DrawPure] shows what fraction of eligible draws also pass here.
    // MEASURED: geoDynVsSrv fires on ~100% of draws because t31 is bound and
    // DYNAMIC essentially always. That is NOT a bug to be excluded away -- it
    // was tried, on evidence that turned out to be about other consumers of
    // the same buffer, and reverted. The derivation's own o2w_t31 path reads
    // t31 LIVE off GetMappedSlice().mapPtr, so a deferred draw really would
    // race it. See the capture loop for the full account. geoStatic therefore
    // reads ~0% until that read is converted onto the record.
    uint32_t geoDynVbMask = 0;      // bit i = vertex buffer slot i is DYNAMIC
    bool     geoDynIb     = false;  // index buffer is DYNAMIC
    bool     geoDynVsSrv  = false;  // any DYNAMIC-backed VS SRV is bound
    // Split of geoDynVsSrv by whether the dynamic slot is the model-instance
    // (t31) stream the derivation reads, or some OTHER slot. The whole slot is
    // resolved from RDEF at capture (memoModelInstSlot), never hardcoded to 31.
    bool     geoDynSrvT31   = false;  // the model-inst SRV slot is DYNAMIC
    bool     geoDynSrvOther = false;  // some other bound VS SRV is DYNAMIC

    // ==================================================================
    // THE t31 / COLOR1 CONVERSION (handoff 5b) -- the bytes the DERIVATION
    // reads, moved from consume time to capture time.
    //
    // WHAT HELD AXIS 3 AT ZERO. The o2w_t31 path reads 48 bytes at
    // charIdx*208 off a DYNAMIC structured buffer, LIVE, at derivation time,
    // and t31 is bound-and-DYNAMIC on essentially every draw. The
    // binding-property test therefore vetoed everything. Excluding the slot was
    // tried and was wrong -- the evidence cited was about two OTHER consumers of
    // the same buffer (the instanced fanout's m_t31ReadCache and BatchSkinJob's
    // bone palette), and this third consumer inside the derivation still raced.
    // The fix is the conversion, not an exclusion: copy at capture, serve from
    // the record, fall back live. Same discipline that already worked twice for
    // cb3.
    //
    // COLOR1 IS PART OF THE CAPTURE, not just the t31 entry. charIdx comes from
    // a live read of the per-instance vertex buffer at
    // m_currentInstanceIndex * stride + instSemByteOffset, which is itself a
    // rename-capable source. Capturing the entry but not the index it was
    // selected by would leave half the race in place. Capturing the whole
    // 8-byte COLOR1 (uint16 x4) covers BOTH index reads the derivation does:
    // .x -> charIdx for the t31 path, .y -> boneIdx for the t30 bone path.
    //
    // ONE ENTRY IS ENOUGH because the instanced fanout issues one SubmitDraw per
    // instance -- it sets m_currentInstanceIndex and calls SubmitDraw in a loop
    // -- so a snapshot covers exactly one instance and exactly one charIdx.
    bool     instSemValid = false;  // COLOR1 below was captured this draw
    uint32_t instSemSlot  = UINT32_MAX;  // the vertex-buffer slot it came from
    uint16_t instSem[4]   = { };    // COLOR1: .x=charIdx  .y=boneIdx
    bool     t31Valid     = false;  // the 48 bytes below were captured this draw
    uint32_t t31CharIdx   = 0;      // the entry index they were read at
    // alignas(16): consumers reinterpret this as a float3x4 exactly as they do
    // the write-combined pointer it replaces. Same reason cbBytes is aligned.
    alignas(16) float t31Entry[12] = { };

    // ==================================================================
    // AXIS 3 AS A READ CERTIFICATE. The masks above are properties of the
    // BINDINGS -- "is anything bound here rename-capable". That is the wrong
    // question and it is why the axis answered ~0%: bound is not the same as
    // read, the identical defect the deleted cb3 predictor existed to fix.
    //
    // The question that actually gates deferral is the one axis 1 asks about
    // cbuffers: DID THIS DERIVATION READ RENAME-CAPABLE BYTES LIVE? A live read
    // of an IMMUTABLE or DEFAULT buffer is safe -- neither can be renamed by
    // Map(WRITE_DISCARD) -- and a buffer that is never read cannot race at all.
    // So only a live read FROM A DYNAMIC SOURCE increments geoLiveReads.
    //
    // NOT YET SUFFICIENT ON ITS OWN, and this is the trap to not walk into: the
    // certificate is only as good as the routing behind it. Every cbuffer
    // content read goes through drawCbSpan, which is what makes cbLiveReads
    // trustworthy. The geometry reads are routed at the four sites that exist
    // today (two t31, two COLOR1), but the derivation is ~3,800 lines and an
    // unrouted read would be INVISIBLE here. So geoBytesStatic() below keeps the
    // conservative binding vetoes as a backstop and uses the certificate only to
    // excuse the inputs whose reads are provably all routed. Retire a veto when
    // the numbers say the input is unread, not on the strength of this comment.
    uint16_t geoLiveReads   = 0;
    uint16_t geoRecordReads = 0;

    // No geometry input the DERIVATION reads can be renamed under a worker.
    //
    // THE BINDING VETOES ARE GONE, 2026-08-10, and this is the change that
    // finally moved the axis off zero. They were: veto if any bound VB / IB /
    // VS SRV is DYNAMIC. Measured with the certificate alongside them, they
    // rejected 100% of draws while geoLiveReads counted 6 live reads in 36,015
    // draws. They were not measuring the race; they were measuring what happened
    // to be bound. That is the deleted cb3 predictor's defect for the third time
    // in this file: BOUND IS NOT READ.
    //
    // WHAT MAKES THE CERTIFICATE SUFFICIENT is an audit, not optimism. Every
    // live read inside ExtractTransforms is one of:
    //   - a CBUFFER read (stagedCbBytes / GetMappedSlice on a cb slot). Those
    //     are axis 1's business and are counted by cbLiveReads; safeToDefer()
    //     already requires that to be zero.
    //   - the t31 model-instance entry -- CONVERTED, served from this record.
    //   - the COLOR1 per-instance entry -- CONVERTED (all four read sites).
    //   - the t30 bone palette -- SCORED via noteGeoLiveRead, not converted.
    // Nothing in it reads the index buffer, which is why geoDynIb's ~21% was
    // pure cost: the max-index scan lives in bt_cullVtx, a different stage,
    // outside anything being deferred.
    //
    // THE CONTRACT THIS PLACES ON FUTURE EDITS: a new live read of geometry
    // bytes added to the derivation MUST go through drawInstSem / drawT31Entry,
    // or call noteGeoLiveRead. An unrouted read is invisible here and would make
    // this certificate lie. That is the same contract drawCbSpan carries for
    // axis 1, and it is the price of asking about reads instead of bindings.
    // geoDynVbMask / geoDynIb / geoDynVsSrv are retained as DIAGNOSTICS -- they
    // size the population where the question could arise, which is how [DrawPure]
    // reports them and all they were ever good for.
    bool geoBytesStatic() const {
      return geoLiveReads == 0;
    }

    // THE TWO AXES THIS RECORD CAN ANSWER FOR, CONJOINED. Call this, never
    // contentPure() alone -- the conjunction is load-bearing in a way that is
    // easy to miss:
    //
    //   contentPure() counts reads that went through drawCbSpan. The SCANS do
    //   not go through it (they iterate offsets, so there is no span to name)
    //   and therefore do not increment cbLiveReads. A scanning draw would
    //   report contentPure() while reading live buffer bytes the whole time.
    //   layoutResolved is exactly what excludes it: a draw that scans is a
    //   draw whose VS had no resolved location, so layoutResolved is false.
    //
    // So neither flag is sufficient and the pair is not redundant. That is
    // also why this is a named method rather than an && at the call site: the
    // next person to add a deferral path should not have to rediscover it.
    //
    // WHAT IT STILL DOES NOT COVER: VERTEX / INDEX / BONE bytes, which have
    // their own capture path (BatchSkinJob and the geometry capture) and no
    // per-draw statement anywhere. That is the third axis and it is not
    // expressed here -- see safeToDefer().
    bool cbSafeToDefer() const {
      return layoutResolved && contentPure() && !overflowed;
    }

    // THE GATE. Two of the three axes, conjoined, per draw:
    //   cbSafeToDefer() -- the derivation read cbuffer CONTENT only from this
    //                      record, and did not take a cold-start scan.
    //   !wroteCarrier() -- it moved no cross-draw D3D11Rtx state in ANY group,
    //                      so it creates no ordering edge into any later draw.
    //                      Deliberately all-groups: see carrierGrp for which
    //                      of them are true dependencies and which are caches,
    //                      and for why that distinction must be measured
    //                      before it is acted on.
    //
    // THE THIRD AXIS IS STILL ABSENT FROM THIS CONJUNCTION, but it is no longer
    // unmeasured: geoBytesStatic() above answers it per draw, and [DrawPure]
    // reports the intersection as elig3. It is deliberately not ANDed in here
    // yet -- the carrier axis was published and read for a run before anything
    // acted on it, and that order is what surfaced the cbLoc dependency
    // instead of shipping it. AND it in once elig3 has been read.
    //
    // Until then, a caller that hands a draw to a worker on this flag alone has
    // proven two thirds of what it needs. Partition on
    // safeToDefer() && geoBytesStatic().
    //
    // Named as a gate rather than left as an && at the call site so the axis
    // list lives in one place and the missing third is impossible to forget.
    // THE THIRD AXIS IS NOW IN. It was held out deliberately while it was a
    // binding property that read 0% -- publishing a fact and reading it for a
    // run before acting on it is what caught the cbLoc dependency instead of
    // shipping it. That run happened: geoLive=6 against 36,015 draws with
    // t31Cap=15,198 and geoRec=15,258, i.e. the conversion fires and the
    // derivation reads essentially nothing renameable live. So the conjunction
    // is the real gate now, and callers no longer have to remember to AND
    // geoBytesStatic() at the call site -- which the old comment here asked them
    // to do, and which is exactly the kind of instruction that gets missed once.
    bool safeToDefer() const {
      return cbSafeToDefer() && !wroteCarrier() && geoBytesStatic();
    }

    // Set when a range did not fit in kMaxCbRanges / kCbBytesCap. A snapshot
    // with this flag is INCOMPLETE and must fall back to the live path -- it
    // must never be silently derived from. If this ever fires in normal play
    // the capture set is wrong; raise the caps and find out which consumer.
    bool overflowed = false;

    // Look up a captured range. Returns nullptr when this draw did not capture
    // it, which is the signal to take the live path rather than to assume zero.
    const uint8_t* find(uint32_t stage, uint32_t slot,
                        uint32_t byteOffset, uint32_t byteCount) const {
      for (uint32_t i = 0; i < numCbRanges; ++i) {
        const CbRange& r = cbRanges[i];
        if (r.stage == stage && r.slot == slot
            && byteOffset >= r.byteOffset
            && byteOffset + byteCount <= r.byteOffset + r.byteCount) {
          return cbBytes.data() + r.dataOffset + (byteOffset - r.byteOffset);
        }
      }
      return nullptr;
    }

    void reset() {
      numCbRanges = 0;
      overflowed  = false;
      layoutResolved = false;
      cbLiveReads = 0; cbRecordReads = 0;
      carrierMask = 0;
      geoDynVbMask = 0; geoDynIb = false; geoDynVsSrv = false;
      geoDynSrvT31 = false; geoDynSrvOther = false;
      instSemValid = false; instSemSlot = UINT32_MAX;
      t31Valid = false; t31CharIdx = 0;
      geoLiveReads = 0; geoRecordReads = 0;
      vs = nullptr; ps = nullptr; inputLayout = nullptr;
      blendState = nullptr; depthState = nullptr; rasterState = nullptr;
      rtv0 = nullptr;
      for (auto& c : vsCbs) c = { };
      indexBuffer = { };
      for (auto& v : vertexBuffers) v = { };
      for (auto& s : vsSrvs)        s = nullptr;
      vsSrvMask[0] = vsSrvMask[1] = vsSrvMask[2] = vsSrvMask[3] = 0;
      numViewports = 0;
    }
  };

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

  // (SdepGroup moved above struct DrawSnapshot -- see the note there.)

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

    // NV-DXVK [suppressStablePropIdVsHashes]: true when the currently bound
    // vertex shader is on rtx.suppressStablePropIdVsHashes, i.e. every propId
    // producer must leave dcs.transformData.stablePropId at 0 so SpatialMap
    // dedup keys on the object-to-world matrix bytes instead.
    //
    // Shared by ALL THREE producers (MakeBoneStablePropId, the sky/cubemap
    // buffer-id site, and the sub-view buffer-id site) on purpose. They key on
    // the same rotating IA buffer pointers and so share the same failure; a
    // switch that silenced only some of them would read in the log as
    // "suppression made no difference", which is the one wrong conclusion this
    // diagnostic must not produce.
    bool IsStablePropIdSuppressed() const;

    // NV-DXVK [Perf.StageDep]: the CROSS-DRAW carriers this class keeps -- the
    // state that makes SubmitDraw serial -- hashed in GROUPS rather than as one
    // value. Capture 2 showed why: `filters` moved a carrier on 43% of eligible
    // draws and a single hash could only say "something", which is not enough to
    // act on. One hash per group names the culprit directly.
    // Member function because the fields are private and the probe lives in
    // markStg. Const: it must never perturb what it measures. The SdepGroup
    // enum is at namespace scope above -- the census's free helpers need it.
    void stageDepCarrierGroups(uint64_t out[kSdepGroupCount]) const;

    // NV-DXVK [DrawSnapshot] 2026-08-10: this is ALSO the per-draw
    // did-this-draw-move-a-carrier test. Called twice per draw (capture
    // baseline + derivation exit) versus the census's 48 (one per stage
    // boundary), so the deferral gate does not need that probe's 1-in-8
    // sampling. The folded carrierFingerprint() that used to sit here was
    // deleted when DrawSnapshot moved to per-group hashes -- see the note at
    // the definition in d3d11_rtx.cpp.

    static constexpr uint32_t kMaxConcurrentDraws = 6 * 1024;
    // NV-DXVK [BatchSubmitDraw perf]: LowLatency=FALSE so idle workers SLEEP on a
    // condition variable instead of spinning. The default (LowLatency=true) makes
    // every idle worker busy-loop the work-stealing scan at 100% CPU. This pool
    // runs its parallel-for only once per frame (flushGeometryBatch) / a burst of
    // per-draw schedules, then sits idle the rest of the frame, so spinning stole
    // ~N cores from the game + CS threads for ~95% of every frame (uniform ~40%
    // inflation of ALL serial work). Matches the other two pools in the tree
    // (dxvk_raytracing, rtx_asset_exporter), which both use LowLatency=false for
    // the same reason. WorkStealing stays true.
    //
    // UPDATE 2026-07-28: LowLatency=false alone did not make this pool scale.
    // nsys sampling (32-core box, auto = 30 workers, 208 items/frame) caught it
    // inside a ~25 ms/frame window where the GPU had no work at all: Schedule()
    // blocked in the kernel, workers asleep in the condvar. Three defects in
    // util_threadpool.h, all of which got WORSE with more threads, are fixed
    // there now - one pool-wide steal spinlock (now per-queue and cache-line
    // padded), notify-under-lock in Schedule (now notified after release), and
    // a failed steal scan that span instead of yielding whenever any queue held
    // work. Until those landed, RAISING the worker count made this pool slower,
    // which is why rtx.geometryWorkerThreads exists as an escape hatch.
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

    // NV-DXVK [DrawSnapshot]: the current draw's captured inputs. See the
    // DrawSnapshot struct at the top of this header for the design and, more
    // importantly, the copy-vs-pin rule that keeps it race-free.
    //
    // STAGING PLAN -- this is deliberately ONE reusable snapshot, not an arena.
    // Step 1 (this change) captures at SubmitDraw entry and converts derivation
    // stages to read from it, WITHOUT moving any work off-thread. That is
    // verifiable for free: the replay tier's 16-replays-per-VS bit-compare must
    // still report FAIL=0, which proves the converted stages are pure functions
    // of the snapshot. Only once that holds does deferral become safe, and only
    // then does this become a per-frame vector<DrawSnapshot> indexed by
    // drawIndex -- the equivalent of UE's MeshDrawCommandStorage.
    // Doing it in the other order would move work off-thread on the strength of
    // an unproven purity claim, which is how the two confirmed races here
    // (COLOR1.y, getImageHash) happened.
    // Per-frame arena, cleared not freed, so capture costs no allocation. This
    // is the MeshDrawCommandStorage equivalent: a snapshot must outlive its
    // draw for any off-thread consumer to read it, which a single reusable
    // slot cannot provide. Reserved once; never reallocated mid-frame, because
    // m_drawSnapCur points into it.
    std::vector<DrawSnapshot> m_drawSnaps;
    // Next free slot in the arena. Reset (not cleared) at frame rollover, so
    // slots are REUSED rather than destroyed and reconstructed -- see the note
    // in captureDrawSnapshot for why that is safe and what it costs when it is
    // not. Rolls over with m_drawCallID, which is the ordering key.
    size_t                    m_drawSnapNext = 0;
    // The draw currently being derived. Null when capture is off or the arena
    // is full -- every consumer treats null as "take the live path".
    DrawSnapshot*             m_drawSnapCur = nullptr;
    // Used when the arena is at cap, so m_drawSnapCur is ALWAYS valid once
    // capture has run. That keeps the consumer guard a single flag test rather
    // than a flag test plus a null test on every read.
    DrawSnapshot              m_drawSnapScratch;

    // NV-DXVK [DrawSnapshot] 2026-08-10: LATCHED per draw at the capture point.
    //
    // Consumers used to guard on RtxOptions::useDrawSnapshot() re-read at the
    // consumer, which is sound only if the option cannot change between capture
    // and consume. It can -- rtx options are runtime-tweakable from the panel.
    // Flip it on mid-draw and SubmitDraw's entry did NOT capture, so a consumer
    // reading true then either dereferences m_drawSnapCur while it is still the
    // frame-reset nullptr, or -- worse, because it is silent -- reads the
    // PREVIOUS draw's snapshot and serves its cb2 bytes as this draw's
    // transforms. Latching turns "is this draw snapshotted?" into a property of
    // the draw rather than a question re-asked of global state, which is the
    // same rule sec 1c applies to the record as a whole.
    //
    // Costs the consumer exactly what the old guard did: one bool test.
    bool                      m_drawSnapValid = false;

    // Fallback-safe accessor. Returns nullptr whenever this draw has no usable
    // snapshot -- capture off, capture skipped, or the record incomplete -- and
    // every consumer treats nullptr as "run the live path". Use this rather
    // than touching m_drawSnapCur directly; the overflow rule in particular is
    // stated on DrawSnapshot::overflowed and is easy to forget at a call site.
    const DrawSnapshot* drawSnap() const {
      return (m_drawSnapValid && m_drawSnapCur != nullptr
              && !m_drawSnapCur->overflowed) ? m_drawSnapCur : nullptr;
    }

    // Captures m_context->m_state into the arena and points m_drawSnapCur at
    // it. Frame thread only, called at SubmitDraw entry while every binding is
    // still the one this draw uses.
    void captureDrawSnapshot(uint32_t drawIndex);

    // NV-DXVK [DrawRedund] 2026-08-11 -- THE OPTIMISATION PLAN, Phase 1.2.
    //
    // Hashes the just-captured record into four components and compares them
    // against the same draw's components last frame, so the plan's single
    // largest unknown -- what fraction of draws are UNCHANGED frame to frame --
    // is a number rather than an inference from the instance-side REDUNDANT=97%.
    //
    // DELIBERATELY NOT INSIDE captureDrawSnapshot, for two reasons that are
    // both about not corrupting the thing being measured:
    //   - captureDrawSnapshot is timed by drawSnap_ns, which is 95.4% of
    //     pfs_guard, which is the largest row in the whole report. Folding a
    //     probe into it would inflate exactly the leaf Phase 1.1 just resolved.
    //   - the draw PARAMETERS (indexed/count/start/base) are not in the record
    //     and are only in scope at the SubmitDraw call site. Two draws sharing
    //     every binding but drawing a different index range are different
    //     draws, and a redundancy figure that conflated them would be wrong in
    //     the optimistic direction.
    // The call site re-stamps the stage marker afterwards so the probe is
    // billed to no bucket at all -- same discipline as the StageDep census
    // inside markStg.
    //
    // Frame thread only (it is called from SubmitDraw, where the record is
    // valid); all state is function-local static thread_local, so a deferred
    // context gets its own tallies rather than racing these.
    void noteDrawRedundancy(const DrawSnapshot& s, bool indexed,
                            uint32_t count, uint32_t start, int32_t base);

    // NV-DXVK [Perf.MemoXf] 2026-08-12 -- THE OPTIMISATION PLAN Phase 3.2b.
    //
    // The memo key for ExtractTransforms: a hash over EVERY input the
    // derivation is certified to read. Not a similarity key and not a route
    // key -- a completeness claim. If the derivation reads something this does
    // not cover, two draws that differ in that thing collide and the second
    // gets the first's transforms, silently and wrongly.
    //
    // COVERS: shader / input-layout / state-object identity, topology, rtv0,
    // all vertex+index+SRV bindings, viewports, cbuffer BINDING identity, the
    // captured cbuffer BYTES, the t31/COLOR1 entry, the five carrier-group
    // fingerprints, and m_currentInstanceIndex.
    //
    // THE CARRIER FINGERPRINTS ARE THE NON-OBVIOUS PART and they are what make
    // the memo sound rather than merely plausible. ExtractTransforms READS
    // cross-draw state (m_last*, the per-VS cb-location cache). Two draws with
    // identical D3D11 inputs still derive differently if another draw moved a
    // carrier between them. DrawSnapshot::carrierGrp is sampled at capture, so
    // folding it into the key makes any such move a key change -- the entry is
    // not invalidated, it simply stops being found, which is the failure
    // direction that costs a hit rather than correctness.
    //
    // DELIBERATELY WITHIN-FRAME. The map is cleared at frame rollover, so a
    // per-frame-rotating input (the bone SRV pointer, which [DrawRedund] showed
    // takes a new value every frame and is bound on ~100% of draws) is a
    // CONSTANT inside the window the memo lives in. Extending this across
    // frames would need that input excluded and re-proven, and the measured
    // within-frame redundancy (254 distinct of 1367) is the larger prize
    // anyway.
    uint64_t drawMemoKey(const DrawSnapshot& s) const;

    // NV-DXVK [Perf.MemoXf.Ablate] 2026-08-12 -- drawMemoKey, split.
    //
    // WHY A SECOND FUNCTION AND NOT A REFACTOR OF THE FIRST. drawMemoKey's
    // exact bits are the thing FAIL=0 was measured against; rebuilding it out
    // of these components would change those bits and silently invalidate that
    // history. So this MIRRORS it instead, and the mirror is checked rather
    // than asserted: the FULL candidate's distinct count is reported next to
    // the production store count, and a drift between them means this function
    // and drawMemoKey have diverged. Re-read both when they disagree.
    //
    // THE SPLIT IS THE WHOLE VALUE. [DrawRedund] already learned this lesson
    // twice -- once when a single `ident` hash read 0% everywhere because one
    // per-frame nonce swamped it, once when `bytes` folded all spans and hid
    // the object/view split -- and the handoff then repeated it a third time by
    // reading the geo FOLD as a single irreducible input. A fold can only say
    // "something moved". These sixteen can say which.
    //
    // kMcGeoContent / kMcGeoSel is the split that motivated the probe. The geo
    // component hashes six fields, and three of them are SELECTORS rather than
    // content: instSemSlot is "the vertex-buffer slot it came from",
    // t31CharIdx is "the entry index they were read at", and instSem[0] IS that
    // charIdx (COLOR1 .x), i.e. the selector is in the key twice over,
    // alongside the entry it selects. Content is instSem[1..3] (.y = boneIdx,
    // which really does pick the bone base and must stay) and t31Entry[12].
    enum MemoComp : uint32_t {
      kMcVsIl = 0,    // vertex shader + input layout
      kMcPs,          // pixel shader
      kMcState,       // blend / depth / raster state objects
      kMcRtv,         // rtv0 PRESENCE (not identity -- see drawMemoKey)
      kMcTopo,        // primitive topology
      kMcVp,          // viewport count + contents
      kMcInstIdx,     // m_currentInstanceIndex
      kMcNumCb,       // numCbRanges
      kMcLayout,      // layoutResolved
      kMcSrv,         // vsSrvMask + every bound VS SRV pointer
      kMcCbPtr,       // every bound cbuffer's buffer pointer + slot
      kMcBytes,       // captured cb span descriptors + their BYTES
      kMcGeoContent,  // instSem[1..3] + t31Entry[12] + both valid flags
      kMcGeoSel,      // instSemSlot + t31CharIdx + instSem[0] (charIdx)
      // ALL FIVE CARRIER GROUPS, one component each -- widened 2026-08-12
      // after [Perf.MemoXf] FAILcarrier read mask=4 (kSdepCbLoc) on 100% of
      // failures. The component set has to be able to express every group, or
      // a key built from it inherits the same blind spot by construction.
      kMcCarCam,      // carrierGrp[kSdepCam]
      kMcCarRoute,    // carrierGrp[kSdepRoute]
      kMcCarCbLoc,    // carrierGrp[kSdepCbLoc]
      kMcCarBone,     // carrierGrp[kSdepBone]
      kMcCarStatic,   // carrierGrp[kSdepStatic]
      kMcCount
    };
    void drawMemoComponents(const DrawSnapshot& s, uint64_t out[kMcCount]) const;

    // ==================================================================
    // NV-DXVK [Perf.SplitXf.Stale] 2026-08-12 -- THE VALIDITY-HASH COMPONENTS,
    // named individually so a stale miss can say WHICH input changed.
    //
    // WHY THIS EXISTS. Every masking decision in this feature so far has been
    // argued from the churn ablation's far columns, and two of them were wrong
    // in a way that cost a run each: geo-on-p1 (refuted, 368 of 368 sampled
    // FAILs were p1) and the p0/p13 merge before it. The far columns cannot
    // settle those questions because they measure a DIFFERENT cache -- variant
    // maps keyed the old byte way, with no identity and no validity hash.
    //
    // These do measure the shipping thing. The first eleven are ObjHead field
    // for field, i.e. exactly what the validity hash is built from. The last
    // four decompose geo, because geo is the component that has cost the most
    // and "geo" is not one thing: kMcGeoContent hashes t31Entry[12] -- twelve
    // floats that ARE a 3x4 transform -- together with instSem[1..3] and two
    // valid flags, and kMcGeoSel hashes three selectors. A path whose
    // objectToWorld is BUILT from t31Entry can never have geo dropped, and a
    // path that merely has t31 bound can. Nothing in the current instrument
    // tells those two apart, and that distinction is the whole p1 question.
    enum XfComp : uint32_t {
      kXcVsIl = 0, kXcPs, kXcState, kXcTopo, kXcNumCb, kXcLayout,
      kXcGeoC, kXcGeoS, kXcInstIdx, kXcCbLoc, kXcSpans,
      // ---- geo, decomposed. Not in the hash; measured alongside it.
      kXcT31Entry,     // t31Entry[12] alone -- the per-instance transform
      kXcInstSem123,   // instSem[1..3] (.y = boneIdx)
      kXcGeoFlags,     // instSemValid + t31Valid
      kXcGeoSelParts,  // instSemSlot + t31CharIdx + instSem[0]
      kXcCount
    };

    // ==================================================================
    // NV-DXVK [Perf.SplitXf] 2026-08-12 -- THE OPTIMISATION PLAN Phase 3,
    // the version the measurements actually support.
    //
    // THE FUSED MEMO IS CAPPED AT 65% AND THAT IS NOT A KEY PROBLEM.
    // [Perf.MemoXf.Ablate] measured OUT dist/f=378 against 1091 draws: a frame
    // contains only 378 genuinely distinct DrawCallTransforms, so ANY key over
    // the whole struct tops out at (1091-378)/1091. Narrowing the key from
    // 1010 distinct to 907 moved hit 5% -> 13% and there is nothing left to
    // take out -- the residue is cbuffer bytes and the t31 entry, both of which
    // the derivation reads for real.
    //
    // THE OUTPUT IS ALREADY FACTORED. The derivation's own tail computes
    //     objectToView = worldToView * objectToWorld            (~:21532)
    // and the two factors have completely different lifetimes:
    //   VIEW   perFrameDistinct{cb2=10} -- ten distinct view/projections in a
    //          frame of 1091 draws, because a frame has ~10 VIEWS.
    //   OBJECT [Perf.FastInst] xformMiss=250 of 15476 -- the object transform
    //          moves on 1.6% of instances per frame.
    // Fused, a camera move invalidates every entry every frame, which is why
    // cross-frame draw redundancy measures bytes=19% while its cb3 half
    // measures 88%. Split, a camera move invalidates ten entries.
    //
    // WHAT MAKES IT SOUND, AND IT IS NOT THIS COMMENT. Two gates, both already
    // proven mechanisms in this tree rather than new ones:
    //   1. safeToDefer(), the same three-axis purity certificate the memo
    //      stores on, decided on the way OUT because it is a statement about
    //      what the derivation DID.
    //   2. the o2w path allowlist. Several objectToWorld sites derive from the
    //      CAMERA (path 4 invView*cb3Mat, 5/6 rdef-with-live-cam, 8 cb2@4
    //      fallback, 11 cached-VP, 13 camera-relative). Their object half is
    //      not view-independent and must never cross a frame. The deleted
    //      replay tier had already enumerated this set (~:15843); the mask is
    //      rtx.splitTransformO2wPathMask so it is widened by measurement.
    // and rtx.splitTransformVerify derives anyway on every serve and
    // bit-compares, reporting FAIL and FAILcarrier.
    //
    // THE VIEW KEY DELIBERATELY OMITS SHADER IDENTITY. worldToView is the
    // camera; it should not depend on which VS observed it. If two shaders in
    // one view derive different view matrices, that is a real finding and FAIL
    // will name it -- at which point vs/il goes into the view key and the ten
    // entries become a few dozen, which still works. Starting with the
    // principled key and letting verify arbitrate is the same order this tree
    // used for every other narrowing.
    // ==================================================================
    struct XfObjectPart {
      Matrix4  objectToWorld;
      Matrix4  textureTransform;
      Vector4  clipPlane;
      uint64_t stablePropId;
      // Pure identity rather than derived, but carried HERE because vs and ps
      // are both in the object key -- so an object entry serves exactly the
      // shader pair that produced it, and nothing has to re-read them.
      uint64_t vertexShaderHash;
      uint64_t pixelShaderHash;
      uint32_t texgenMode;
      uint32_t texcoordEncoding;
      bool     enableClipPlane;
      bool     isSubView;
      bool     isSubViewSkybox;
      // Frame this entry was last USED, for age eviction. See the cache block:
      // a flat wipe at the cap throws away the stable working set together
      // with the churn, which is a self-inflicted loss on top of the churn
      // itself.
      uint32_t lastFrame;

      // ---- THE VALIDITY HASH. This is what makes identity keying safe. ------
      // 2026-08-12: the map is keyed on OBJECT IDENTITY (IA buffer pointers +
      // shader pair), not on the derivation's input bytes. Identity is stable
      // across camera motion -- measured, [Perf.SplitXf] ident{}: 46-93 new
      // keys/frame while moving against 933-959 for the byte key, and 0/frame
      // on p1 -- but stability is not correctness. A stable key alone would
      // serve a moving prop its first-ever matrix forever.
      //
      // So the key finds the entry and THIS decides whether it may be served:
      // the hash of the derivation's input bytes, i.e. byte for byte the value
      // the whole map used to be keyed on. Serve only when it matches.
      //
      // The property that follows is the point of the design: THE IDENTITY KEY
      // CANNOT CAUSE A WRONG SERVE. Two distinct objects colliding on one
      // identity, or an identity too coarse to separate them, costs a hit and
      // nothing else -- the validity hash still has to match, and it is the
      // same test that gated every serve before this change. Identity buys
      // cache STABILITY (entries ~= objects instead of ~= byte patterns, which
      // is what filled a 16384-entry cap in ~25 frames and made aged run to
      // ~39,000 a window); the validity hash keeps CORRECTNESS.
      //
      // It is also what makes the per-path key mask checkable at last. A
      // narrowed mask now narrows THIS, not the key, so a draw the mask
      // wrongly forgives still lands on its own entry and verify still gets to
      // compare -- where a narrowed KEY produced a hit on a different entry
      // and the wrong serve was invisible (handoff 2.5's blind spot).
      uint64_t valHash;

      // ---- PROBE STATE. [Perf.SplitXf.Stale] / entry provenance. ------------
      // Written on every store, read on every stale miss and every FAIL. Kept
      // on the entry rather than in a side map so it cannot go out of sync with
      // the payload it describes -- a side map keyed on the same identity would
      // be updated by a different code path and this feature has already paid
      // twice for two structures that were supposed to agree.
      //
      // ~15 uint64 on a ~16k-entry cache is ~2 MB. That is affordable and the
      // alternative is another run spent guessing which component moved.
      uint64_t comps[kXcCount];
      // Which drop mask was in force when this entry was stored. THE MASK IS
      // PER DRAW AND THE ENTRY IS SHARED: a p13 draw stores under one mask and
      // a p1 draw on the same identity compares under another, against this one
      // stored hash. Measured 18:44-18:45: p1 failed 10 times with DropTrue=1
      // the moment DropFalse went non-zero, having been clean across 37 of 38
      // windows with DropFalse=0. Storing it makes the mismatch countable, and
      // is also the field the eventual fix compares.
      uint8_t  dropMask;
      // o2w path of the draw that stored this entry, and the frame it did so.
      // storeFrame is NOT lastFrame: lastFrame is refreshed on use and drives
      // eviction, this one never moves. Same-frame vs prior-frame replacement
      // is the difference between "two live instances share one identity, so
      // N-way associativity fixes it" and "the object changed over time, so it
      // does not".
      uint8_t  storePath;

      // ---- [Perf.SplitXf.Bytes] SUB-COMPONENT STATE. -----------------------
      // The component cross-tab settled that neither spans nor geo is
      // droppable on p1 or p13: both appear in chgS AND chgR, so whatever the
      // derivation reads is INSIDE one of them, not equal to one of them. That
      // is as far as component granularity can go, and going further is the
      // whole remaining prize -- 55% of draws are spurious misses.
      //
      // So record the pieces. Per-range hashes with their identity, and
      // t31Entry element by element, both compared on a stale miss and both
      // split by the spur/real verdict. A range or an element that moves on
      // spurious misses and never on real ones is a byte the validity hash is
      // covering and should not be; the reverse is one it must keep.
      //
      // Raw identity, not a bucket: rngId packs (stage, slot, byteOffset) so
      // the report names the actual cbuffer location rather than an index into
      // a table nobody can map back.
      static constexpr uint32_t kXfRngMax = 12u;   // == DrawSnapshot::kMaxCbRanges
      uint8_t  numRng;
      uint32_t rngId  [kXfRngMax];
      uint64_t rngHash[kXfRngMax];
      float    t31[12];

      // 16-BYTE SUB-BLOCKS OF THE OBJECT-SIDE RANGES. Range granularity was
      // enough for t31Entry, where the element split named the translation
      // column outright, and NOT enough for the cb3 spans: p1 reads s0c3+48
      // with S=5433 R=1679 and p13 reads s0c3+0/48/64 with R in the thousands,
      // so at range granularity both look load-bearing and neither can be
      // narrowed. The question is which 16 bytes inside them the derivation
      // reads, and one register is the natural unit -- cbuffers are laid out
      // in float4 registers and a derivation reads whole registers.
      //
      // OBJECT-SIDE ONLY (slot not 0/1/2). Slots 0/1 are the per-frame globals
      // and slot 2 is the view buffer; none are in the object validity hash,
      // so recording them would spend the budget on ranges whose changes this
      // cache does not care about. That is also why the c2 columns in the
      // range report should be read as context and not as targets.
      static constexpr uint32_t kXfBlkMax = 16u;
      uint8_t  numBlk;
      uint32_t blkId  [kXfBlkMax];   // (slot << 24) | byteOffset of the block
      uint64_t blkHash[kXfBlkMax];
      // How many times THIS IDENTITY has been overwritten, carried across the
      // replacement. Feeds the thrash histogram: an identity restored once is a
      // moving object, one restored eight times a window is several live
      // instances fighting over a single slot, and only the second is fixed by
      // making the entry N-way.
      uint16_t restoreCount;
      uint32_t storeFrame;

      // ---- CARRIER REPLAY. rtx.splitTransformCarrierReplay. ----------------
      // A serve skips the derivation, so any cross-draw state the derivation
      // would have WRITTEN never moves, and a later draw reads state that was
      // never updated. The store gate's answer has always been to refuse the
      // draw outright, and that refusal is ~86% of all refusals and the single
      // largest cost in the feature -- it is what holds the standing-still
      // ceiling at ~71% and it starves the view cache too, because a refused
      // draw stores NEITHER half.
      //
      // The alternative is to reproduce the write instead of skipping it.
      // These fields are the recorded exit state of the groups named in
      // rtx.splitTransformCarrierReplayMask, written back on every serve.
      //
      // ONLY kSdepCbLoc IS STORED HERE, and that is the whole safety argument:
      // recording a value and replaying it later is correct only if the value
      // is a pure function of THIS draw's inputs. The resolved proj/view
      // location is a property of the shader and its cb layout, both of which
      // are in the object key, so an entry replays into exactly the shader
      // that produced it. kSdepCam's m_smoothedCamPos is an accumulator -- new
      // value from old value -- and is deliberately NOT here; if that group
      // ever needs replaying, the accumulator has to come out of the group
      // first. Adding a field to a replayed group means adding it here too, or
      // the replay silently writes a subset -- REPLAYFAIL is what tells you.
      uint32_t carProjSlot;
      size_t   carProjOffset;
      int      carProjStage;
      uint32_t carViewSlot;
      size_t   carViewOffset;
      bool     carColumnMajor;
      // Which groups this entry's producing derivation actually moved. Zero
      // for an entry stored under the all-clean rule, so an entry captured
      // before the flag was turned on can never replay anything.
      uint8_t  carGroups;
    };
    struct XfViewPart {
      Matrix4  worldToView;
      Matrix4  viewToProjection;
      float    viewportWidth;
      float    viewportHeight;
      uint32_t worldToViewPathId;

      // ---- kSdepCam CARRIER REPLAY, AND WHY IT LIVES ON THE VIEW HALF -----
      // kSdepCam is ~86% of what is left of the store gate: refuse{carGrp}
      // reads cam~7,500 per window against cbLoc~1,480 once cbLoc is replayed,
      // and the gate as a whole is what holds the ceiling at 80%. Replaying it
      // takes the ceiling to ~94%, which is the only route to a 90% serve.
      //
      // IT WAS PREVIOUSLY WRITTEN OFF AS UNREPLAYABLE, and that judgement was
      // right about the OBJECT cache and wrong in general. The reasoning was:
      // m_smoothedCamPos is an accumulator (new value from old) and the origin
      // latches track the camera, so a value recorded N frames ago is stale.
      // Both objections are objections to CROSSING A FRAME -- and the view
      // cache does not. It is cleared at every frame boundary and it is keyed
      // on comp[kMcCarCam], i.e. on the very state being recorded. A view entry
      // is therefore always same-frame and always from a draw that entered with
      // the same camera carrier, so replaying it is not a stale write, it is
      // the write the skipped derivation would have made.
      //
      // That is the object/view split's own argument applied to the carrier
      // rather than to the matrix, and it is the reason this belongs here and
      // not in XfObjectPart. Putting these fields on the object half would
      // reintroduce exactly the staleness the original objection described.
      //
      // FIELD LIST IS stageDepCarrierGroups' kSdepCam BLOCK, IN ORDER. It must
      // stay that way: a field hashed there and missing here means the replay
      // writes a subset, the group still reads clean to carrierMask, and
      // REPLAYFAIL is the only thing that would catch it.
      Vector3  camLastDrawOrigin;      bool camLastDrawOriginSet;
      Vector3  camO2wOrigin;           bool camO2wValid;
      bool     camO2wFromFanout;       uint32_t camO2wOff;
      Vector3  camFanoutOrigin;        bool camHasFanoutOrigin;
      Vector3  camViewmodelOrigin;     bool camHasViewmodelOrigin;
      Vector3  camFanoutVpRow0;
      Vector3  camFanoutVpRow1;
      Vector3  camFanoutVpRow2;        bool camHasFanoutVpRows;
      // NO SMOOTHER HERE. m_smoothedCamPos/m_hasPrevCamPos are kSdepCamSmooth
      // now, which is deliberately not replayable -- recording an accumulator
      // and writing it back reproduces a value that depends on the draw
      // ORDER, not on this draw. Adding them back here is the bug that read
      // REPLAYFAIL grp{cam=1..29}.
      // Which groups this entry's producing derivation moved, same contract as
      // XfObjectPart::carGroups. Zero for an entry stored under the all-clean
      // rule, so a pre-flag entry replays nothing.
      uint8_t  carGroups;

      // ---- THE VIEW CACHE IS NO LONGER CLEARED EVERY FRAME. --------------
      // It was, on the argument that "the camera changes every frame, and ten
      // entries a frame is the measured population, so clearing is free".
      // Clearing WAS free at ten entries. vsIl took store{view} from 390-590 to
      // ~4,000 per window (~100/frame), and a per-frame cache costs one
      // derivation per distinct key per frame -- so the clear became a hard
      // ~9%-of-draws floor that no key tuning can remove. Measured: standing
      // still at serve=83%, miss{obj=865 view=3520 both=2376}, i.e. view 14.8%
      // against object 8.1%.
      //
      // The clear is also REDUNDANT with the key: the view key already carries
      // the cb2 spans and carCam, so a camera move invalidates entries by key.
      // Standing still those keys repeat and the entries simply hit.
      //
      // frameStored IS THE SAFETY VALVE, and it is what keeps the cam replay
      // honest. That replay was justified by "the view cache is per-frame, so a
      // recording cannot go stale" -- which stops being true here. So an entry
      // whose producer MOVED kSdepCam is only usable in the frame that stored
      // it; older ones are treated as a miss. Entries that moved no cam are
      // free to cross frames, because there is nothing camera-shaped to go
      // stale.
      uint32_t frameStored;
      // SEPARATE STAMP FOR EVICTION, and the two must not be merged.
      // frameStored is IMMUTABLE -- it answers "was this recorded against the
      // current camera", so refreshing it on use would make a cam-carrying
      // entry look same-frame forever and reinstate exactly the stale write
      // this design exists to prevent. lastUsed is refreshed on every serve and
      // is what the age sweep reads, so a stationary camera's entries survive
      // being hit for hundreds of frames.
      uint32_t lastUsed;
    };

    // Splits the captured cbuffer spans by SLOT: slot 2 is the view/projection
    // buffer ([DrawRedund] bytesSlot cb2=18% cross-frame, perFrameDistinct
    // cb2=10 -- it is the camera), everything else is object/pass state.
    // cb0 and cb1 go into BOTH keys: they take 4 and 2 distinct values a frame,
    // so they cost neither key anything, and putting them in both means neither
    // key can be blind to a dependency that turns out to live there.
    // CHURN ABLATION. The object cache is measured CORRECT (whole windows of
    // FAIL with o2w absent) and COLD -- newObj/frame runs in the hundreds
    // against the ~4-13 objects per frame that [Perf.FastInst] says actually
    // move. So the object VALUE is stable and the object KEY is not, and the
    // question is which component moves. Three candidates and no way to pick
    // between them by argument: accObj (cb3 is model x VIEW on the path-4
    // sites, so its bytes follow the camera while the matrix does not),
    // geoContent (the t31 per-instance entry), and instIdx (TF2 adds and drops
    // ~6 props per fanout batch per frame, so instance indices shift under a
    // stable object set -- rtx_types.h documents this).
    //
    // Same instrument as [Perf.MemoXf.Ablate], pointed at a cross-frame key
    // instead of a within-frame one: build the key with one component held
    // out, count how many NEW keys each variant takes on per frame, and the
    // variant whose count collapses names the churner. Costs one run instead
    // of one build per guess.
    enum ObjVar : uint32_t {
      kOvFull = 0,   // the shipping object key
      kOvNoSpans,    // minus accObj (the cb3 / object cbuffer bytes)
      kOvNoGeo,      // minus geoContent + geoSel
      kOvNoInstIdx,  // minus m_currentInstanceIndex
      kOvNoCbLoc,    // minus the cbLoc carrier fingerprint
      kOvCount
    };
    // outObjKey is the object cache's IDENTITY key and outObjVal the validity
    // hash that decides whether an entry found under it may be served. They are
    // two different questions -- "which object is this" and "is what I recorded
    // for it still current" -- and they were one value until 2026-08-12, which
    // is why turning the camera invalidated a static prop. See XfObjectPart::
    // valHash for why the split is safe by construction.
    // outObjComps receives the kXcCount validity components UNMASKED, and
    // outDropMask the mask that was applied to produce outObjVal. Both are
    // probe outputs and both are optional; pass nullptr on the shipping path.
    // Unmasked deliberately: the question a stale miss has to answer is which
    // input MOVED, and a masked copy has already thrown that away.
    void drawSplitKeys(const DrawSnapshot& s,
                       uint64_t& outObjKey, uint64_t& outObjVal,
                       uint64_t& outViewKey,
                       uint64_t* outObjVariants = nullptr,
                       uint64_t* outObjComps = nullptr,
                       uint32_t* outDropMask = nullptr) const;

    // NV-DXVK [DrawSnapshot] viewport accessor -- the FIRST derivation input
    // moved off live state and onto the record.
    //
    // HANDOFF sec 3 lists "the viewport fallback" among the live reads that
    // keep ExtractTransforms impure. The viewport is already captured, so this
    // is wiring, not new capture: it reads the snapshot when this draw has one
    // and falls back to m_context->m_state otherwise, which keeps every caller
    // correct whether or not rtx.useDrawSnapshot is on.
    //
    // Returns false when no viewport is bound, so callers stop indexing
    // viewports[0] unconditionally -- several sites did, and read a
    // default-constructed viewport when numViewports was 0.
    // Defined in the .cpp: D3D11DeviceContext is incomplete in this header.
    bool drawViewport0(D3D11_VIEWPORT& out) const;

    // The replay tier's packed viewport key, (Width << 32) | Height. Three
    // sites computed this independently -- the entry key build, the keyMiss
    // attribution, and the record capture at exit -- and they MUST agree or the
    // tier mis-attributes a miss or admits a record keyed on a viewport that
    // was never hashed. One helper so they cannot drift, and it routes through
    // the snapshot like every other converted read.
    //
    // Those sites carried the comment "bindings cannot change mid-draw" as an
    // assumption. Reading them from the record makes it true by construction
    // rather than by assertion, which is the whole point of the record.
    //
    // Returns PRESENCE, not just the value, because the three sites treated
    // "no viewport bound" differently and the distinction is load-bearing: the
    // key build skipped the xtMix entirely (so the key differs from one that
    // mixed a zero), while the other two left their field at 0. outKey is set
    // to 0 when this returns false.
    bool drawViewportKey(uint64_t& outKey) const;

    // The remaining identity accessors, same contract as drawViewport0: read
    // the record when this draw has one, else live state. These are the fields
    // the capture already held and nothing read -- wiring them up is what turns
    // the identity half of DrawSnapshot from dead weight into the thing sec 1c
    // describes.
    //
    // ONLY VALID INSIDE THE DRAW. m_drawSnapValid is latched at SubmitDraw
    // entry and cleared at frame reset, so these are correct anywhere reachable
    // from SubmitDraw (ExtractTransforms included) and MUST NOT be used from
    // EndFrame or any other out-of-draw context -- there they would serve the
    // last draw's bindings. That is why the conversion was applied per site
    // rather than by a blanket replace: 10 of the 22 inputLayout reads in this
    // file are outside SubmitDraw.
    // Returned BY CONST REFERENCE, deliberately: the 34 call sites this
    // replaced bind it in every shape there is -- `auto x = ...` (Com copy),
    // `const auto& x = ...` (reference), `... != nullptr`, `...->GetCommonShader()`,
    // `....ptr()`. Handing back the same Com<> lvalue the live read produced
    // keeps all of them type-identical and refcount-identical, so the
    // conversion cannot change behaviour at any of them. Both storages outlive
    // the call: the snapshot's copy lives in the arena slot for the whole draw,
    // the live one in context state.
    const Com<D3D11VertexShader>& drawVertexShaderCom() const;

    D3D11InputLayout*        drawInputLayout() const;
    D3D11ShaderResourceView* drawVsSrv(uint32_t slot) const;
    D3D11RenderTargetView*   drawRtv0() const;
    // Binding identity for a VS constant-buffer slot. Out-of-range yields an
    // empty binding (null buffer, UINT64_MAX generation), matching the
    // "nothing bound" answer every caller already handles.
    const DrawSnapshot::CbBinding& drawVsCb(uint32_t slot) const;

    // ================================================================
    // NV-DXVK [DrawSnapshot] 2026-08-10: THE CONTENT-READ ACCESSOR.
    //
    // This is the piece that finishes the conversion. The identity half was
    // done by the accessors above; the barrier that remained was cbuffer
    // CONTENT -- ~32 sites in ExtractTransforms, each hand-rolling the same
    // twelve lines: stagedCbBytes(), fall back to GetMappedSlice(), bounds-
    // check, cast at base+offset. Every one of them reads LIVE bytes at
    // consume time, which is precisely what a deferred reader cannot do.
    //
    // WHY ONE ACCESSOR RATHER THAN 32 CONVERSIONS. Converting the sites
    // individually is what the previous two sessions did for four of them, and
    // it does not converge: each site re-derives its own offset arithmetic, and
    // the proj-vs-view convention divergence (one adds constantOffset*16, the
    // other does not) means every new site is a fresh chance to capture a
    // different 64 bytes than the consumer reads. Here the convention is a
    // PARAMETER -- `rebase` -- supplied once per site and used by BOTH the
    // lookup and the capture, so the two cannot disagree by construction.
    //
    // WHAT IT DOES, in order:
    //   1. resolve the binding (from the record when this draw has one);
    //   2. abs = (rebase ? constantOffset*16 : 0) + relOffset;
    //   3. return the captured bytes if the record holds that span;
    //   4. otherwise read live exactly as the site used to -- AND record
    //      (slot, relOffset, byteCount, rebase) into the VS's span manifest,
    //      so the NEXT draw of this shader captures it and step 3 hits.
    //
    // THE LEARNING IS THE POINT, and it is the same self-healing shape the cb3
    // predictor already proved: being wrong is free. A span the manifest has
    // not learned yet is served live -- correct, just not pure -- and learning
    // it costs one bounded array write. It converges per VS within a draw or
    // two of that shader's first sighting, and it needs no advance list of
    // which offsets the derivation reads, which is exactly the list nobody
    // could write by hand.
    //
    // Returns nullptr when the slot is unbound or the span does not fit the
    // buffer -- the same answer the live code produced, and every caller
    // already handles it.
    //
    // `rebase` is NOT a default argument on purpose. The two conventions are
    // the trap this file has already been bitten by; making every call site
    // spell out which one it uses is the point.
    //
    // narrowScratch -- THE HIT-ONLY LIVE FALLBACK, added 2026-08-10, and the
    // thing that let the last un-routed content consumer come in.
    //
    // The default live fallback calls stagedCbBytes(), whose MISS path stages
    // the WHOLE buffer (~6.3 KB) to serve one narrow read. For a cbuffer the
    // game Map(WRITE_DISCARD)es every draw -- slot 3 is the case -- the content
    // generation moves every draw, so the staging ring can NEVER hit and that
    // miss path is the only path. That is the [Perf.WcCopy] shape (~3.4 MB/frame,
    // 91% of pole-thread WC bytes) a previous session removed from the cb3->o2w
    // site by hand, and it is why that site kept a private find() instead of
    // routing here (handoff 4d).
    //
    // Pass a caller-owned buffer of at least byteCount bytes and the live
    // fallback becomes: staging ring HIT-ONLY (free when another site already
    // staged this generation), else GetMappedSlice + a narrow memcpyFromWC of
    // exactly byteCount into that buffer. stagedCbBytes() is never called. The
    // returned pointer is then into the scratch, so it is valid for as long as
    // the caller's storage is -- which is why this is a parameter and not an
    // internal static: the site owns the lifetime, as it already did.
    //
    // WHY ROUTE AT ALL, given the site's hand-rolled version was already
    // correct: the private copy could not reach noteCbSpanRead(), so the span
    // manifest never learned slot 3 from the consumer that reads it most. That
    // kept the cb3 PREDICTOR alive as a second, overlapping mechanism. Routed,
    // the manifest learns it like every other span and the predictor was
    // deleted (see captureDrawSnapshot).
    const uint8_t* drawCbSpan(uint32_t slot, uint32_t relOffset,
                              uint32_t byteCount, bool rebase,
                              uint8_t* narrowScratch = nullptr);

    // [5b] THE GEOMETRY-BYTE ANALOGUES OF drawCbSpan. Same contract: serve from
    // the record when it was captured, fall back to the live read otherwise, and
    // COUNT which one happened so the record can certify itself.
    //
    // The counting rule is the one thing here that is not obvious. A live
    // fallback increments geoLiveReads ONLY when the source buffer is
    // D3D11_USAGE_DYNAMIC, because only DYNAMIC can be renamed by
    // Map(WRITE_DISCARD) under a worker. Reading an IMMUTABLE vertex buffer live
    // is perfectly safe and must not cost the draw its eligibility -- scoring it
    // as unsafe is what made the binding-property version of this axis useless.
    //
    // Both return nullptr when the value is unavailable from either source,
    // which is the signal to take the site's existing "skip" branch. Neither
    // applies the sites' finite/zero-row validity checks: those are decisions
    // about the VALUE and stay with the consumer.

    // The draw's per-instance COLOR1 entry (uint16 x4): .x = charIdx for the
    // t31 model-instance path, .y = boneIdx for the t30 bone path.
    const uint16_t* drawInstSem(uint32_t slot, uint32_t byteOffset,
                                uint16_t* scratch4);

    // t31[charIdx].objectToCameraRelative -- the float3x4 at entry+0, 48 bytes.
    // Serves from the record only when the record captured THIS charIdx; a
    // mismatch means the consumer chose a different instance than the capture
    // saw, so the read goes live and is scored live.
    const float* drawT31Entry(uint32_t srvSlot, uint32_t charIdx,
                              float* scratch12);

    // SCORING ONLY, for a derivation read that is deliberately NOT converted
    // onto the record. The t30 bone palette is the case: it is read live at four
    // points across the full and replay paths, and it is not DYNAMIC in any
    // scene measured (dynSrvSlots names only t4 and t31), so converting it would
    // buy nothing and cost a 48-byte capture on every skinned draw. Noting it
    // costs one Desc() read and keeps geoLiveReads HONEST -- the certificate
    // stops depending on t30 happening to be immutable, and correctly vetoes the
    // draw the day a scene ships a dynamic bone buffer.
    void noteGeoLiveRead(D3D11Buffer* src);
    // TOTAL: an out-of-range slot yields a binding whose buffer is null rather
    // than reading past the array. The call sites this replaced indexed
    // vertexBuffers[slot] unchecked and every one of them already handles a
    // null buffer, so this is a drop-in that removes an out-of-bounds read.
    const D3D11VertexBufferBinding& drawVertexBuffer(uint32_t slot) const;

    // NV-DXVK [flicker V8: SubmitDraw-ordered geometry capture]: predictor for
    // "will this draw (re)bake its BLAS inputs?". The real decision
    // (DrawCallCache hash comparison) happens later on the CS thread, but the
    // capture copy must be recorded NOW, at the draw's position in the CS
    // stream — so gate on when a bake is plausible: a (buffers, counts, VS)
    // key never seen before, seen within its first warm frames (engine
    // re-batches dedup-miss on TWO consecutive frames), or returning after a
    // gap (buffer address reuse across a free/realloc). Over-capture is
    // harmless (unused copy); under-capture means a bake reads the live
    // source and can flicker. Accessed only on the owning (game) thread.
    struct GeomCapturePredictorEntry {
      uint32_t firstSeenFrame = 0;
      uint32_t lastSeenFrame = 0;
      // One capture per key per frame: instanced fanout submits the same
      // geometry once per instance, but they share one BlasEntry and EmitCs
      // FIFO makes the first submit the one whose commit creates it — so the
      // first capture covers the bake and the rest would be dead copies.
      uint32_t lastCaptureFrame = 0;
    };
    std::unordered_map<uint64_t, GeomCapturePredictorEntry> m_geomCapturePredictor;
    uint32_t                             m_geomCapturePredictorSweepCounter = 0;
    // NV-DXVK [flicker V8 follow-up: capture feedback]: per-frame snapshot of
    // the CS-published capture-wanted ring (dxvk::tf2::g_geomCaptureWanted*).
    // Rebuilt on the first SubmitDraw of each frame so lookups during the
    // frame are O(1) with no atomics. Game-thread only.
    std::unordered_set<uint64_t>         m_geomCaptureWantedSnapshot;
    uint32_t                             m_geomCaptureWantedSnapshotFrame = UINT32_MAX;
    // NV-DXVK [capture stability contract]: per-frame snapshot of the
    // CS-published PROVEN-STABLE identity keys (g_geomCaptureStableMap minus
    // fresh taints). The capture decision is inverted from the wanted-key
    // feedback: capture UNLESS the key is in this set. Fail-safe direction is
    // correctness — a key the CS thread hasn't (re)confirmed stable gets
    // captured, which can only cost bandwidth, never a torn bake. This is
    // what covers the FIRST bake of a re-batched entry, whose (vs,vtx,idx)
    // feedback key does not exist anywhere until after that bake already ran
    // uncaptured (capState=none, isNew=1 on 100% of starving lines,
    // 2026-08-03 01:57 run).
    std::unordered_set<uint64_t>         m_geomCaptureStableSnapshot;
    uint32_t                             m_geomCaptureStableSnapshotFrame = UINT32_MAX;
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
    // NV-DXVK [CamGeoLatch]: the pose the EndFrame consumer last actually fed
    // to Main, kept so a present with NO new world render (the no-advance
    // branch) re-feeds exactly that instead of a fresher live capture. Without
    // it, advance and no-advance presents would land Main at different poses
    // and reintroduce a phase jump.
    //
    // This replaced [zigzag fix B], an 8-deep delay ring that fed Main the
    // capture N engine-frames behind the newest (rtx.engineHookMainCameraFrame-
    // Delay). That compensated for the camera/geometry mismatch instead of
    // removing it. The mismatch itself is now gone: the camera is latched at
    // the frame's first draw (see g_latchPending in d3d11_rtx.cpp) so it
    // travels with the geometry it belongs to. Ring, option and delay cache
    // deleted 2026-07-30 after the fix was confirmed on screen and in the log
    // (latch=1 on 1555/1555 frames; liveEf != engFrame on 1552 of them, so the
    // old consumer was reading the wrong engine frame's camera almost always).
    //
    // Written only by the EndFrame consumer, on the same (calling) thread as
    // the rest of it, so a plain member is race-free.
    bool                                 m_engineCamLastConsumedValid = false;
    float                                m_engineCamLastConsumedW2v[16] = {};
    float                                m_engineCamLastConsumedV2p[16] = {};
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
    // NV-DXVK [Perf.Replay] v6 path-5/6 widening: the RDEF CBufModelInstance
    // bind slot the rdef o2w site read this draw (paths 5/6/7). Captured into
    // the replay record so the replay can re-read the same 48 bytes and
    // re-evaluate the cb3IsZero mode per draw. UINT32_MAX = not an rdef draw.
    uint32_t                             m_lastModelCbSlot = UINT32_MAX;
    // NV-DXVK [Perf.Replay] v6.3 path-1/2/3 widening: the camera origin the
    // t31 o2w site (path 1) actually added to its translation this draw, and
    // WHERE it came from. Path 1 prefers the session-latched fanout origin
    // (m_lastFanoutCamOrigin -- a member, so the replay reads the same live
    // value the full path would) and falls back to cb2@+4, which IS cb2
    // content and therefore has to be witness/carry proven like path 13's.
    // Recording the source lets the replay reproduce the site's choice
    // bit-exactly instead of guessing which branch ran.
    //   m_lastO2wCamValid   -- the site's haveCam (false => no camO added)
    //   m_lastO2wCamFromFanout -- true: fanout latch; false: cb2@m_lastO2wCamOff
    // NV-DXVK [Perf.Replay] v6.4 genMiss fix: set when the wtvPath-3 view
    // matrix was built from the cb2@16 VP-rotation branch (plus cb2@4
    // camXYZ) -- the ONE path-3 rotation source that is pure cb2 content and
    // therefore byte-provable. The fanout-VP branch and the
    // !gotLiveRotation fallback both read mutable cross-draw state
    // (m_lastGoodTransforms), so they can never carry and are excluded.
    // v6.5: the shared path-3 camera derivation. Called by the full
    // ExtractTransforms run AND by the replay refresh, so there is exactly
    // one implementation of this math in the build.
    bool resolvePath3CamOrigin(Vector3& camOut, uint32_t& slotOut,
                               uint32_t& offOut, char& srcOut);
    // requireLiveRot: replay callers pass true so the non-reproducible
    // cached-rotation fallback returns false instead of composing from
    // mutable cross-draw state. The site passes false (it needs the fallback).
    bool derivePath3WorldToView(const Vector3& cam, Matrix4& outW2v,
                                bool& outFromCb2, int& outSlotPicked,
                                float& outSlotD2, bool requireLiveRot);
    // (m_lastWtvP3FromCb2 / m_lastWtvP3CamSlot / m_lastWtvP3CamOff deleted with
    //  the v6.4 wtvPath-3 carry proof. They recorded where path 3 read camXYZ
    //  so the proof could decide whether it was provable cb2 content; with the
    //  proof gone they were written every draw and never read.)
    Vector3                              m_lastO2wCamOrigin{ 0.0f, 0.0f, 0.0f };
    bool                                 m_lastO2wCamValid = false;
    bool                                 m_lastO2wCamFromFanout = false;
    uint32_t                             m_lastO2wCamOff = 0;

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

    // NV-DXVK [MeshTrace]: identity of the draw currently in SubmitDraw, set
    // at the entry reset so every one of the function's 33 return sites can be
    // attributed. m_meshTraceReported keeps a draw that bumps more than one
    // filter from logging twice.
    uint64_t m_meshTraceVs = 0;
    uint32_t m_meshTraceIdx = 0;
    bool     m_meshTraceActive = false;
    bool     m_meshTraceReported = false;
    // __LINE__ of the last function-scope early return that was passed. The
    // no-filter exits are 95% of all rejects, so "somewhere in SubmitDraw" is
    // not an answer; this names the site.
    uint32_t m_meshTraceCp = 0;
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
    // the [subPassUpd] and [subPassSky] logs only. ([subPassDropProbe], named
    // here previously, has no emitter anywhere in the tree.)
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

    // NV-DXVK [fanout prev-transform identity] 2026-08-05: index-parallel to
    // m_currentInstancesToObject — the PREVIOUS FRAME's transform for the same
    // placement, taken from the engine's own per-instance struct (g_modelInst
    // words 12..23, the 48 bytes immediately after the current matrix).
    //
    // This replaces charIdx as the identity source. charIdx was an index into a
    // PER-DRAW scratch buffer of 2..96 entries ([T31Struct] measured the bound
    // t31 length directly), i.e. a loop counter, so the same value in two draws
    // named two unrelated props and merged them.
    //
    // The prev transform is not another transform-derived key. Hashing the
    // CURRENT transform asks "what is at this position", which fails the moment a
    // prop moves. The previous transform is the engine stating where THIS prop
    // was last frame, so it resolves to last frame's instance exactly, however
    // far the prop travelled. Verified bit-exact across f13072→13073→13074 on all
    // three translation components and the full basis.
    //
    // Stored in ABSOLUTE world space (camOrigin already applied), because that is
    // the space the SpatialMap is keyed in. See m_fanoutPrevCamOrigin for why the
    // camOrigin used here must be the PREVIOUS frame's.
    //
    // Empty when the engine supplied no history (its prev block is all-zero on a
    // prop's first frame); the consumer then behaves exactly as it does today.
    const std::vector<Matrix4>*          m_currentPrevInstancesToObject = nullptr;
    std::shared_ptr<const std::vector<Matrix4>> m_currentPrevInstancesToObjectOwner;

    // NV-DXVK [fanout prev-transform identity]: per-VS record of the camOrigin
    // used to absolutise the fanout matrices, so the NEXT frame can absolutise
    // its prev block with the SAME value.
    //
    // Required for bit-exactness, which is the whole point. Last frame stored
    // curAbs = curRaw + camOrigin(N-1). This frame's prev block is bit-identical
    // to last frame's curRaw, so prevAbs reproduces curAbs exactly only if it is
    // offset by camOrigin(N-1). Using the current frame's camOrigin would be off
    // by one frame of camera motion and no hash would ever match.
    //
    // `ambiguous` guards the case where draws of one VS in one frame disagree on
    // camOrigin (sub-views such as the 3D skybox reconstruct around a different
    // origin). A frame flagged ambiguous supplies no prev transforms to the next
    // one, which falls back to today's behaviour rather than matching wrongly.
    // TWO slots, not one. A VS issues hundreds of draws per frame (248 on the
    // dominant fanout VS), and every one of them both READS the previous frame's
    // origin and RECORDS this frame's. With a single slot the first draw of the
    // frame reads correctly and then overwrites it, so draws 2..N find the
    // current frame in the record and get no history — measured as prevHit=11
    // against prevMiss=250, 11 being the size of one draw.
    struct FanoutCamOriginRecord {
      uint32_t curFrameId    = 0xFFFFFFFFu;
      float    curOrigin[3]  = { 0.0f, 0.0f, 0.0f };
      bool     curValid      = false;
      bool     curAmbiguous  = false;
      uint32_t prevFrameId   = 0xFFFFFFFFu;
      float    prevOrigin[3] = { 0.0f, 0.0f, 0.0f };
      bool     prevValid     = false;
    };
    std::unordered_map<uint64_t, FanoutCamOriginRecord> m_fanoutPrevCamOrigin;

    // NV-DXVK [CamOrig]: provenance of the camOrigin that path-10 adds to the
    // camera-relative t31 translation (d3d11_rtx.cpp:8188). Stashed at the
    // fanout build site and consumed by [SubmitBone] in SubmitDraw, which runs
    // later in the same call, on the same thread, for the same draw - so the
    // association is 1:1 and needs no keying.
    //
    // WHY. t31 is camera-relative; adjT = m[3] + camOrigin lifts it to world.
    // There are TWO sources for camOrigin: the draw's own c_cameraOrigin read
    // from cb2 (:7361), and an unconditional override to the 3D-skybox origin
    // g_engineSkyCamOrigin (:7893) whenever the engine sub-view flag
    // (g_vanishDiagCapturedA3 & 0x10) is set. Substituting the wrong one shifts
    // every instance of the draw by a CONSTANT - which is the one mechanism
    // that explains a wrong placement that is bit-stable across minutes, as
    // both the far steady state and each near signature measurably are.
    //
    // Measured: on flash frames instancesToObject[0] equals the draw's own
    // objectToWorld exactly (34 occurrences). Since adjT0 = rawT0 + camOrigin,
    // the origin that WOULD place this draw correctly is
    // camNeeded = origO2W.T - rawT0, computable on EVERY frame. Comparing it
    // against both candidates says which source is right and which is applied,
    // without waiting for the artifact to reproduce.
    float    m_fanoutCamOriginCb[3]   = { 0.f, 0.f, 0.f };  // cb2 c_cameraOrigin
    float    m_fanoutCamOriginUsed[3] = { 0.f, 0.f, 0.f };  // after the :7893 override
    float    m_fanoutRawT0[3]         = { 0.f, 0.f, 0.f };  // t31[0] translation, pre-add
    bool     m_fanoutHaveCamOrigin    = false;
    bool     m_fanoutCamOriginOverridden = false;
    // Set when the r8 latch said "sub-view" but the draw's own origin disagreed
    // with the sky camera by more than a refresh lag, so the substitution was
    // declined. rejN in [SubmitBone] is the fix firing: it should be 17/17 on
    // the frames that used to place this geometry 26,225u off screen.
    bool     m_fanoutCamOriginRejected = false;
    bool     m_fanoutRawT0Valid       = false;
    uint32_t m_fanoutSkyValid         = 0u;   // g_engineSkyCamOriginValid
    uint32_t m_fanoutInSubView        = 0u;   // g_vanishDiagCapturedA3 & 0x10
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
    // NV-DXVK [T31Cache]: validity key for m_t31ReadCache. The copy above was
    // unconditional -- it re-read the WHOLE mapped t31 buffer on every
    // instanced draw. Measured: 236 MB/window over 27711 calls (8.5 KB each,
    // 554 calls/frame, 4.72 MB/frame), which is 91% of all frame-thread
    // write-combined traffic and the single hottest leaf in our code
    // ([Perf.WcCopy] d3d11+0x33707b -> SubmitInstancedDraw).
    //
    // All three fields must match or we re-copy:
    //   SrcBuf  the D3D11Buffer identity. Compared ONLY -- never dereferenced,
    //           because the Com<ID3D11Resource> that produced it is released at
    //           the end of the resolve block (same contract as
    //           m_cachedInstBufPtr).
    //   MapPtr  the slice address, which changes on Map(WRITE_DISCARD).
    //   MapGen  D3D11Buffer::GetMapGeneration(), which ALSO changes on
    //           Map(WRITE_NO_OVERWRITE) -- the in-place rewrite that MapPtr
    //           alone cannot see. Without this the cache would serve stale
    //           per-instance transforms; see the note on GetMapGeneration().
    // SrcBuf == nullptr is the empty state and can never match a real fill.
    //
    // The generation is read BEFORE the bytes, never after: stamping a newer
    // generation onto older bytes is the one ordering that could serve stale
    // data, and reading it first can only ever cause a redundant re-copy.
    //
    // SCOPE: this covers the immediate context, which is the only thread that
    // reaches SubmitInstancedDraw here ([Perf.SdThreads] count=1). A DEFERRED
    // context maps buffers via AllocSlice() without touching m_mapped, so a
    // deferred writer would leave GetMappedSlice() -- and therefore t31Data
    // itself, cache or no cache -- pointing at the previous slice. That is a
    // pre-existing property of this read path, not something the key adds, and
    // bumping the generation there would re-copy the same stale slice rather
    // than fix it. If t31 ever moves to a deferred context, the read at the
    // resolve site is what has to change.
    const void*                          m_t31CacheSrcBuf = nullptr;
    const void*                          m_t31CacheMapPtr = nullptr;
    uint64_t                             m_t31CacheMapGen = 0ull;
    // NV-DXVK [T31Range]: which byte range of the source buffer m_t31ReadCache
    // actually HOLDS. The vector is still sized to the full buffer so every
    // reader can keep indexing it by ABSOLUTE offset (charIdx*208) with the
    // bounds checks unchanged -- only the FILL is narrowed.
    //
    // WHY. GetMapGeneration() bumps on Map(WRITE_NO_OVERWRITE), which is the
    // APPEND pattern: the engine writes draw N's block into a shared dynamic
    // buffer and bumps the generation, so the key can never hold across draws
    // ([Perf.T31Cache] measured 5.8% hits). Re-copying the WHOLE buffer on each
    // of those misses makes the per-frame cost quadratic in draws -- draw N
    // re-reads every earlier draw's block as well as its own. Copying only the
    // entries this draw's charIdx values actually reference makes it linear,
    // and it is a win whether or not the key ever hits again.
    //
    // Half-open [begin, end). begin == end is the empty state. A hit requires
    // the key to match AND this range to CONTAIN the range the draw needs.
    size_t                               m_t31CacheFillBegin = 0u;
    size_t                               m_t31CacheFillEnd   = 0u;
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

    // NV-DXVK [DrawSnapshot] 2026-08-10: the seven members above DO carry
    // across draws, deliberately, and two attempts to stop them were measured
    // and reverted -- see stageDepCarrierGroups' kSdepCbLoc block and the
    // VsLocCache site in ExtractTransforms. The carry is the bootstrap channel
    // by which a first-sighting VS acquires a location, and these fields are
    // an input to the replay tier's record keying. Retiring them from the
    // carrier set is worth ~3,750 eligible draws per 3 s window but needs the
    // tier decoupled from them first.

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
    // NV-DXVK [SubViewSkyTexDump]: dump the bound PS textures of a TF2
    // 3D-skybox sub-view draw (SubmitDraw, "[subPassSky]"). Deduped by image
    // hash and capped, so the GPU readback runs at most once per unique
    // texture. Writes ./rtx-remix/subview_sky/ (override with env
    // DXVK_SUBVIEW_SKY_TEX_PATH) using the SAME <imageHash>_albedo.dds
    // naming as the on-screen albedo dump, so the two sets diff by filename.
    //
    // Was DumpDroppedSubPassTextures, writing ./rtx-remix/dropped/, back when
    // this pass was deleted outright. The drop is gone (see the long note at
    // its old site); the dump stays because it is still the clearest view of
    // what the sub-view pass carries.
    void DumpSubViewSkyTextures(const DrawCallState& dcs);
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
