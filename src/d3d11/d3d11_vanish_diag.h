/*
 * NV-DXVK TF2 vanish-zone diagnostic.
 *
 * Per-frame counters for D3D11 entry points most likely to gate world geometry
 * submission (Predication, queries, RT/OM/RS state, VS/PS bindings, cbuffer
 * updates, copies, etc.). D3D11Rtx::EndFrame drains them, snapshots a baseline
 * on "good" frames (kept >= 95% of max), and on cliff frames (>=10% draw drop)
 * logs every counter whose value differs from the baseline.
 *
 * Use to identify which engine subsystem went quiet between a good frame and
 * the cliff frame at the same camPos.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

#define DXVK_VANISH_DIAG_CALLS(X) \
  X(SetPredication) X(QueryBegin) X(QueryEnd) X(GetData) \
  X(Map) X(Unmap) X(UpdateSub) \
  X(ClearRTV) X(ClearDSV) \
  X(OMSetRT) X(OMSetRTUAV) X(OMSetBlend) X(OMSetDS) \
  X(RSSetState) X(RSSetVP) X(RSSetSR) \
  X(IASetIL) X(IASetPT) X(IASetVB) X(IASetIB) \
  X(VSSetShader) X(VSSetCB) X(VSSetSRV) \
  X(PSSetShader) X(PSSetCB) X(PSSetSRV) \
  X(Dispatch) X(Discard) X(CopySub) X(CopyRes) X(ResolveSub) \
  X(CreateTex2D) X(DestroyTex2D) X(CreateBuf) X(DestroyBuf) \
  X(ExecCmdList) X(FinishCmdList) X(Flush) \
  X(Draw) X(DrawIndexed) X(DrawInstanced) X(DrawIdxInst) \
  X(DrawAuto) X(DrawIndirect) X(DrawIdxIndirect)

namespace dxvk { namespace vanish_diag {

  enum CallId : int {
#define DXVK_VANISH_DIAG_ENUM(name) name,
    DXVK_VANISH_DIAG_CALLS(DXVK_VANISH_DIAG_ENUM)
#undef DXVK_VANISH_DIAG_ENUM
    CALL_COUNT
  };

  inline constexpr const char* kNames[CALL_COUNT] = {
#define DXVK_VANISH_DIAG_NAME(name) #name,
    DXVK_VANISH_DIAG_CALLS(DXVK_VANISH_DIAG_NAME)
#undef DXVK_VANISH_DIAG_NAME
  };

  // Defined in d3d11_context.cpp.
  extern std::atomic<uint32_t> g_counts[CALL_COUNT];

  // NV-DXVK [StallWatch]: the frame thread heartbeat, written by ScopedCall on
  // a clock read it already takes. Read by the watchdog thread in
  // d3d11_context_imm.cpp. See the store site for why this exists.
  extern std::atomic<uint64_t> g_stallHeartbeatNs;
  extern std::atomic<uint32_t> g_stallHeartbeatCall;
  extern std::atomic<uint32_t> g_stallThreadId;
  // Called once from the D3D11 device; starts the watchdog on first use.
  void stallWatchStart();

  inline void bump(CallId id) {
    g_counts[id].fetch_add(1, std::memory_order_relaxed);
  }

  // Drain all counters atomically (each exchanged to 0) into out[CALL_COUNT].
  inline void drain(uint32_t out[CALL_COUNT]) {
    for (int i = 0; i < CALL_COUNT; ++i) {
      out[i] = g_counts[i].exchange(0, std::memory_order_relaxed);
    }
  }

  // NV-DXVK [perf]: wall time spent INSIDE the wrapped entry points.
  //
  // [Perf.Block] showed the frame-owning thread never blocks on the CS thread or
  // the GPU, and OnDraw* + EndFrame only cover about a third of the wall time
  // between EndFrames. The draw hooks are the only D3D11 entry points that were
  // ever timed — Map / Unmap / UpdateSubresource / ExecuteCommandList are not,
  // and a Source-engine title drives all of them hard (dynamic VB/CB uploads and
  // the materialsystem deferred-context replay). These say whether the missing
  // time is inside the wrapper at all.
  //
  // Separate accumulators from g_counts on purpose: EndFrame's vanish-diag
  // consumer drains g_counts every frame, while these drain on the 5 s [Perf]
  // cadence. Sharing them would make the two inconsistent.
  extern std::atomic<uint64_t> g_timeNs[CALL_COUNT];
  extern std::atomic<uint64_t> g_timeCalls[CALL_COUNT];

  // Most of these entry points live in the templated common context, so they are
  // reached from the game's deferred-context worker threads as well. Only the
  // thread that owns the frame is interesting here — that is the one [Perf.Busy]
  // measures — so attribute to it alone rather than blending in worker time.
  // Set by D3D11Rtx::EndFrame; a TLS read is cheaper than GetCurrentThreadId and
  // needs no windows.h in this header.
  extern thread_local bool t_isFrameThread;

  // NV-DXVK [perf] 2026-08-06: per-call accumulation is THREAD-LOCAL and
  // non-atomic; the atomics are touched once per 5 s window instead of once
  // per call.
  //
  // WHY. Everything below the `if (!timed) return` in ScopedCall runs only on
  // the frame thread, so the atomics were never actually being used for
  // cross-thread accumulation there - they were paying `lock xadd` (~18 ns
  // each, more when the line is shared with the worker threads that bump
  // g_counts) purely out of habit. Five of them per call, against ~302,000
  // GetData polls per frame ([Perf.Query]: 12.69 M polls per 5 s window,
  // 99.998% not-ready), is ~27 ms per frame of lock-prefixed traffic to
  // maintain counters that are read three times a minute.
  //
  // Plain thread-local adds are ~1 ns and need no synchronisation because a
  // thread cannot race with itself. The globals stay atomic because the
  // DRAIN still crosses threads and g_counts is still bumped by every thread.
  //
  // KNOWN LIMIT: if the frame thread changes identity, whatever the previous
  // thread had accumulated but not yet flushed is lost. That is one window of
  // one thread's diagnostic counters at a swap that happens ~never, and the
  // alternative (a registry of per-thread blocks, as [Perf.SdThreads] uses)
  // buys nothing here.
  inline thread_local uint64_t t_accTimeNs[CALL_COUNT]    = {};
  inline thread_local uint64_t t_accTimeCalls[CALL_COUNT] = {};
  inline thread_local uint64_t t_accGapNs[CALL_COUNT]     = {};
  inline thread_local uint64_t t_accGapCount[CALL_COUNT]  = {};
  inline thread_local uint64_t t_accGapMaxNs              = 0;
  inline thread_local int      t_accGapMaxCall            = 0;

  // NV-DXVK [Perf.EntryCensus] — TIER 1(b): the VOLUME axis.
  //
  // WHY THIS EXISTS. [Perf.Entry] answers "how many ms are inside the D3D11
  // entry points" and nothing else, and on a machine that downclocks 2.4x
  // (HANDOFF_PERF_2026-08-18 §0.2) an ms is not a usable unit. Half the frame
  // thread lives in here and the only split that has ever existed is the three
  // draw entry points vs one undifferentiated "state-setting" bucket. Counts
  // and bytes are clock-independent, so they read correctly on a throttled
  // window, and they are what forks the problem:
  //
  //   one expensive call     -> high share, low `n`   -> fix the call
  //   a thousand cheap ones  -> high share, high `n`  -> fix the call RATE
  //   Map traffic            -> high `KB`             -> fix the upload path
  //
  // WHAT A "UNIT" IS, per call id. Deliberately one array rather than five
  // differently-named ones: every consumer wants "the natural size of this
  // call", and the census prints the name of the unit next to it.
  //   Draw / DrawIndexed / DrawInstanced / DrawIdxInst  -> vertices or indices
  //   Map / UpdateSub / CopySub / CopyRes               -> unused (see g_bytes)
  //   *SetCB / *SetSRV / IASetVB / OMSetRT              -> slots bound (NumX)
  // g_bytes is the payload in BYTES and is only populated where a byte count
  // is knowable at the call site.
  //
  // COST: one non-atomic thread-local add per call, on paths that already pay
  // two steady_clock reads (~70-80 ns) for ScopedCall. Nothing is added to
  // GetData, which is ~302,000 calls/frame and has no volume to report.
  extern std::atomic<uint64_t> g_bytes[CALL_COUNT];
  extern std::atomic<uint64_t> g_units[CALL_COUNT];
  inline thread_local uint64_t t_accBytes[CALL_COUNT] = {};
  inline thread_local uint64_t t_accUnits[CALL_COUNT] = {};

  // Frame-thread only, exactly like the time accumulators: these entry points
  // are on the templated common context and are also reached from the game's
  // deferred-context workers, whose volume would not belong to the pole.
  inline void addBytes(const CallId id, const uint64_t n) {
    if (t_isFrameThread)
      t_accBytes[id] += n;
  }

  inline void addUnits(const CallId id, const uint64_t n) {
    if (t_isFrameThread)
      t_accUnits[id] += n;
  }

  // NV-DXVK [Perf.EntryMap]: Map split by D3D11_MAP, because the four kinds are
  // four different costs and lumping them hides the only actionable one.
  // WRITE_DISCARD renames the allocation (that is the dropship COLOR1.y
  // rename, seen from the producing side); WRITE_NO_OVERWRITE does not; READ
  // can stall on the GPU. Slot 0 is unused so the D3D11_MAP value (1..5)
  // indexes directly with no translation table to get wrong.
  constexpr uint32_t kMapKindSlots = 6;
  extern std::atomic<uint64_t> g_mapCalls[kMapKindSlots];
  extern std::atomic<uint64_t> g_mapBytes[kMapKindSlots];
  extern std::atomic<uint64_t> g_mapImageCalls;   // non-buffer Map, bytes unknowable here
  inline thread_local uint64_t t_accMapCalls[kMapKindSlots] = {};
  inline thread_local uint64_t t_accMapBytes[kMapKindSlots] = {};
  inline thread_local uint64_t t_accMapImageCalls           = 0;

  // NV-DXVK [Perf.EntryMap] 2026-08-19: the same Map census split by BIND FLAG,
  // and this one decides an architecture question rather than describing a cost.
  //
  // THE QUESTION. Moving a per-draw consumer (ExtractTransforms) onto a worker
  // requires the draw to own its inputs. The cheap way to own them is by
  // REFERENCE -- record (buffer, offset, mapGen) at draw time, read the bytes
  // later -- instead of the memcpy captureDrawSnapshot pays today. That is only
  // sound if the slice the draw read is still the slice holding those bytes
  // when the deferred reader runs.
  //
  // Map(WRITE_DISCARD) RENAMES: the game gets a fresh slice and the old bytes
  // are whatever the allocator does next. So a by-reference capture is sound
  // exactly when the buffers the derivation reads are NOT being renamed between
  // the draw and the read. The flat census already says WRITE_DISCARD runs
  // ~1227 times a frame against ~1341 draws -- about one rename per draw -- but
  // that aggregate cannot say WHICH buffers, and the answer is opposite for the
  // two populations:
  //   renames are mostly VERTEX/INDEX  -> the cbuffers the derivation reads are
  //                                       stable, by-reference capture is viable
  //   renames include CONSTANT_BUFFER  -> a reference captured at draw time
  //                                       reads the LAST draw's bytes at frame
  //                                       end. Silently wrong transforms, and
  //                                       the same failure the m_t31ReadCache
  //                                       note in HANDOFF sec 0.1 already paid
  //                                       for once on the sibling buffer.
  // Three buckets, because a buffer can carry several bind flags and the
  // interesting question is per-role: index 0 = has CONSTANT_BUFFER, 1 = has
  // VERTEX or INDEX, 2 = everything else (SRV / UAV / staging).
  constexpr uint32_t kMapBindSlots = 3;
  extern std::atomic<uint64_t> g_mapBindCalls[kMapKindSlots][kMapBindSlots];
  extern std::atomic<uint64_t> g_mapBindBytes[kMapKindSlots][kMapBindSlots];
  inline thread_local uint64_t t_accMapBindCalls[kMapKindSlots][kMapBindSlots] = {};
  inline thread_local uint64_t t_accMapBindBytes[kMapKindSlots][kMapBindSlots] = {};

  inline constexpr const char* kMapBindNames[kMapBindSlots] = { "cb", "vbib", "other" };

  // mapType is the raw D3D11_MAP; anything out of range folds into slot 0 so a
  // bad value shows up as an unnamed bucket instead of corrupting a real one.
  // bindRole is an index into kMapBindNames, classified at the call site where
  // the D3D11_BIND_* constants are in scope.
  inline void noteMapBuffer(const uint32_t mapType, const uint64_t bytes,
                            const uint32_t bindRole) {
    if (!t_isFrameThread)
      return;
    const uint32_t k = (mapType < kMapKindSlots) ? mapType : 0u;
    t_accMapCalls[k] += 1;
    t_accMapBytes[k] += bytes;
    const uint32_t r = (bindRole < kMapBindSlots) ? bindRole : (kMapBindSlots - 1u);
    t_accMapBindCalls[k][r] += 1;
    t_accMapBindBytes[k][r] += bytes;
  }

  inline void noteMapImage() {
    if (t_isFrameThread)
      t_accMapImageCalls += 1;
  }

  // Fold this thread's accumulators into the shared atomics. Called at the top
  // of both drains; the second call is a no-op because the locals are zeroed.
  // Defined below, once the [Perf.Gap] globals it touches have been declared.
  inline void flushDiagLocals();

  inline void drainTimes(uint64_t outNs[CALL_COUNT], uint64_t outCalls[CALL_COUNT]) {
    flushDiagLocals();
    for (int i = 0; i < CALL_COUNT; ++i) {
      outNs[i]    = g_timeNs[i].exchange(0, std::memory_order_relaxed);
      outCalls[i] = g_timeCalls[i].exchange(0, std::memory_order_relaxed);
    }
  }

  // NV-DXVK [Perf.Gap]: the complement of g_timeNs - time the frame thread
  // spends BETWEEN D3D11 entry points, attributed to whichever call it was last
  // seen leaving.
  //
  // Measured 2026-07-27, steady state:
  //   frame ([Perf.PresentWall] frameMsAvg)          129-139 ms
  //   inside Present                                   0.002 ms
  //   inside all immediate-ctx entry points            1.6  ms  <- g_timeNs
  //   between submissions ([Perf.SubmitGap])         ~128    ms
  // and every DXVK block counter is flat zero in the same run: [Perf.GameBlock]
  // waitResNs=0 syncCsNs=0, [Perf.Block] csSyncMs=0 gpuSyncMs=0 gpuIdleMs=0,
  // [Perf.SyncSite] polls=0. Pipeline compilation is also excluded - 3 compiles
  // in a whole session ([Perf.PipeGfx]).
  //
  // So ~127 ms per frame is unaccounted for: the game is neither inside our
  // entry points nor blocked on anything we own. This measures that residue
  // directly instead of inferring it by subtraction.
  //
  // Reading it: if one call name owns the gap, the game is stuck immediately
  // after that call and the name identifies what it is waiting on. If the gap
  // is spread evenly over thousands of calls, the engine is simply executing
  // slowly between D3D11 calls and the bottleneck is inside Titanfall, not
  // reachable from here. maxMs isolates a single long stall from an even smear.
  //
  // Note the largest single gap will usually be the frame boundary itself (last
  // call of one frame -> first call of the next, which spans the game's whole
  // simulation). That is expected and is why the per-call breakdown, not just
  // the max, is what to read.
  extern std::atomic<uint64_t> g_gapNsByCall[CALL_COUNT];
  extern std::atomic<uint64_t> g_gapCountByCall[CALL_COUNT];
  extern std::atomic<uint64_t> g_gapMaxNs;
  extern std::atomic<int>      g_gapMaxCall;

  // NV-DXVK [Perf.Gap]: QueryEnd split by D3D11_QUERY type.
  //
  // Needed because the flat afterQueryEnd bucket is an average over 6 gaps per
  // frame that are NOT the same event. [Perf.QEvent] showed only ~0.9 of those
  // 6 are EVENT queries (the rest are other types), while [Perf.Gap] maxMs runs
  // 197-249 ms against a 110-124 ms bucket total - which is only possible if one
  // gap is enormous and the others are near zero. Averaging them together hid
  // that, and it is the difference between "the engine waits on a GPU fence"
  // and "the engine waits after some unrelated query".
  //
  // 16 slots covers the whole D3D11_QUERY enum.
  constexpr uint32_t kQueryTypeSlots = 16;
  extern std::atomic<uint64_t> g_gapNsByQueryType[kQueryTypeSlots];
  extern std::atomic<uint64_t> g_gapCountByQueryType[kQueryTypeSlots];
  extern std::atomic<uint64_t> g_gapMaxNsByQueryType[kQueryTypeSlots];

  // Set by D3D11ImmediateContext::End() once the query type is known, read when
  // the NEXT entry point closes the gap. Written before ScopedCall's destructor
  // records t_lastCallId=QueryEnd, so it always describes the End() that opened
  // the interval being attributed.
  extern thread_local uint32_t t_lastQueryEndType;

  // NV-DXVK [Perf.Boundary]: split the frame-boundary gap at Present.
  //
  // [Perf.GapQ] found the whole ~110 ms sits in ONE gap per frame, the one
  // following the TIMESTAMP_DISJOINT End - while EVENT (0.003 ms) and TIMESTAMP
  // (0.008 ms) are nothing. TS_DISJOINT is frame-scoped, so its End is the last
  // timed D3D11 call of the frame and that gap is the frame boundary: it spans
  // end-of-frame, Present, and the next frame's setup all at once.
  //
  // Present itself is already known to cost 0.002 ms ([Perf.PresentWall]), so
  // the 110 ms is on one side of it or the other, and that distinction picks the
  // next move:
  //   preMs large  -> the engine blocks BEFORE presenting: finishing its frame,
  //                   draining its render queue, waiting on its own workers
  //   postMs large -> it blocks AFTER presenting, at the start of the next
  //                   frame: waiting for a frame to become available, i.e. a
  //                   swapchain/pacing wait rather than an engine one
  // Timestamps are steady_clock nanoseconds since the process epoch so the
  // swapchain and the entry-point hooks share one time base.
  extern std::atomic<uint64_t> g_presentEnterNs;
  extern std::atomic<uint64_t> g_presentExitNs;
  // NV-DXVK [GapSampler]: the thread that calls Present — the one whose
  // post-present CPU gap ([Perf.Boundary] postMs) needs naming. Written by
  // the swapchain on every Present, read by the sampler thread.
  extern std::atomic<uint32_t> g_presentTid;
  extern std::atomic<uint64_t> g_boundaryPreNs;
  extern std::atomic<uint64_t> g_boundaryPostNs;
  extern std::atomic<uint64_t> g_boundaryPreMaxNs;
  extern std::atomic<uint64_t> g_boundaryPostMaxNs;
  extern std::atomic<uint64_t> g_boundaryCount;
  extern std::atomic<uint64_t> g_boundaryNoPresent;

  inline uint64_t steadyNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  // Frame-thread only, so no synchronisation is needed on these.
  extern thread_local std::chrono::steady_clock::time_point t_lastExit;
  extern thread_local bool t_haveLastExit;
  extern thread_local int  t_lastCallId;
  extern thread_local int  t_depth;

  // Definition of the flush forward-declared above, now that g_gapNsByCall /
  // g_gapCountByCall / g_gapMaxNs / g_gapMaxCall are in scope.
  inline void flushDiagLocals() {
    for (int i = 0; i < CALL_COUNT; ++i) {
      if (t_accTimeNs[i] || t_accTimeCalls[i]) {
        g_timeNs[i].fetch_add(t_accTimeNs[i], std::memory_order_relaxed);
        g_timeCalls[i].fetch_add(t_accTimeCalls[i], std::memory_order_relaxed);
        t_accTimeNs[i] = t_accTimeCalls[i] = 0;
      }
      if (t_accGapNs[i] || t_accGapCount[i]) {
        g_gapNsByCall[i].fetch_add(t_accGapNs[i], std::memory_order_relaxed);
        g_gapCountByCall[i].fetch_add(t_accGapCount[i], std::memory_order_relaxed);
        t_accGapNs[i] = t_accGapCount[i] = 0;
      }
      // [Perf.EntryCensus]: same fold, same reason - per-call atomics on a path
      // this hot are what the 2026-08-06 note above measured at ~27 ms/frame.
      if (t_accBytes[i] || t_accUnits[i]) {
        g_bytes[i].fetch_add(t_accBytes[i], std::memory_order_relaxed);
        g_units[i].fetch_add(t_accUnits[i], std::memory_order_relaxed);
        t_accBytes[i] = t_accUnits[i] = 0;
      }
    }
    for (uint32_t k = 0; k < kMapKindSlots; ++k) {
      if (t_accMapCalls[k] || t_accMapBytes[k]) {
        g_mapCalls[k].fetch_add(t_accMapCalls[k], std::memory_order_relaxed);
        g_mapBytes[k].fetch_add(t_accMapBytes[k], std::memory_order_relaxed);
        t_accMapCalls[k] = t_accMapBytes[k] = 0;
      }
      for (uint32_t r = 0; r < kMapBindSlots; ++r) {
        if (t_accMapBindCalls[k][r] || t_accMapBindBytes[k][r]) {
          g_mapBindCalls[k][r].fetch_add(t_accMapBindCalls[k][r], std::memory_order_relaxed);
          g_mapBindBytes[k][r].fetch_add(t_accMapBindBytes[k][r], std::memory_order_relaxed);
          t_accMapBindCalls[k][r] = t_accMapBindBytes[k][r] = 0;
        }
      }
    }
    if (t_accMapImageCalls) {
      g_mapImageCalls.fetch_add(t_accMapImageCalls, std::memory_order_relaxed);
      t_accMapImageCalls = 0;
    }
    if (t_accGapMaxNs != 0) {
      uint64_t prev = g_gapMaxNs.load(std::memory_order_relaxed);
      while (t_accGapMaxNs > prev
          && !g_gapMaxNs.compare_exchange_weak(prev, t_accGapMaxNs,
               std::memory_order_relaxed)) { }
      if (t_accGapMaxNs >= prev)
        g_gapMaxCall.store(t_accGapMaxCall, std::memory_order_relaxed);
      t_accGapMaxNs = 0;
    }
  }

  inline void drainGaps(uint64_t outNs[CALL_COUNT], uint64_t outCounts[CALL_COUNT],
                        uint64_t& outMaxNs, int& outMaxCall) {
    flushDiagLocals();
    for (int i = 0; i < CALL_COUNT; ++i) {
      outNs[i]     = g_gapNsByCall[i].exchange(0, std::memory_order_relaxed);
      outCounts[i] = g_gapCountByCall[i].exchange(0, std::memory_order_relaxed);
    }
    outMaxNs   = g_gapMaxNs.exchange(0, std::memory_order_relaxed);
    outMaxCall = g_gapMaxCall.load(std::memory_order_relaxed);
  }

  // NV-DXVK [Perf.EntryCensus]: drains the volume axis. MUST be called in the
  // same emit block as drainTimes and AFTER it - drainTimes runs the flush, and
  // draining volumes on their own would publish a window of counts against a
  // window of ns that had already been zeroed.
  inline void drainVolumes(uint64_t outBytes[CALL_COUNT], uint64_t outUnits[CALL_COUNT]) {
    flushDiagLocals();
    for (int i = 0; i < CALL_COUNT; ++i) {
      outBytes[i] = g_bytes[i].exchange(0, std::memory_order_relaxed);
      outUnits[i] = g_units[i].exchange(0, std::memory_order_relaxed);
    }
  }

  inline void drainMapKinds(uint64_t outCalls[kMapKindSlots],
                            uint64_t outBytes[kMapKindSlots],
                            uint64_t& outImageCalls,
                            uint64_t outBindCalls[kMapKindSlots][kMapBindSlots],
                            uint64_t outBindBytes[kMapKindSlots][kMapBindSlots]) {
    flushDiagLocals();
    for (uint32_t k = 0; k < kMapKindSlots; ++k) {
      outCalls[k] = g_mapCalls[k].exchange(0, std::memory_order_relaxed);
      outBytes[k] = g_mapBytes[k].exchange(0, std::memory_order_relaxed);
      for (uint32_t r = 0; r < kMapBindSlots; ++r) {
        outBindCalls[k][r] = g_mapBindCalls[k][r].exchange(0, std::memory_order_relaxed);
        outBindBytes[k][r] = g_mapBindBytes[k][r].exchange(0, std::memory_order_relaxed);
      }
    }
    outImageCalls = g_mapImageCalls.exchange(0, std::memory_order_relaxed);
  }

  // NV-DXVK [Perf.EntryCensus]: the FAMILY a call id rolls up into. The plan's
  // fork is per-family, not per-call - "Map traffic" is Map+Unmap+UpdateSub
  // together, and a binder is only interesting against its siblings.
  enum Family : int {
    kFamDraw, kFamMap, kFamCbSet, kFamSrvSet, kFamState,
    kFamQuery, kFamCopy, kFamCreate, kFamOther, kFamCount
  };

  inline constexpr const char* kFamilyNames[kFamCount] = {
    "draws", "maps", "cbSet", "srvSet", "state",
    "query", "copy", "create", "other"
  };

  inline constexpr Family familyOf(const int id) {
    switch (id) {
      case Draw: case DrawIndexed: case DrawInstanced: case DrawIdxInst:
      case DrawAuto: case DrawIndirect: case DrawIdxIndirect: case Dispatch:
        return kFamDraw;
      case Map: case Unmap: case UpdateSub:
        return kFamMap;
      case VSSetCB: case PSSetCB:
        return kFamCbSet;
      case VSSetSRV: case PSSetSRV:
        return kFamSrvSet;
      case SetPredication:
      case OMSetRT: case OMSetRTUAV: case OMSetBlend: case OMSetDS:
      case RSSetState: case RSSetVP: case RSSetSR:
      case IASetIL: case IASetPT: case IASetVB: case IASetIB:
      case VSSetShader: case PSSetShader:
        return kFamState;
      case QueryBegin: case QueryEnd: case GetData:
        return kFamQuery;
      case CopySub: case CopyRes: case ResolveSub:
      case ClearRTV: case ClearDSV: case Discard:
        return kFamCopy;
      case CreateTex2D: case DestroyTex2D: case CreateBuf: case DestroyBuf:
        return kFamCreate;
      default:
        return kFamOther;   // ExecCmdList, FinishCmdList, Flush
    }
  }

  // The unit name printed next to `u=` for this call id, or nullptr when the
  // call has no meaningful volume. Keeping this next to familyOf means a new
  // call id gets a family and a unit in one edit or neither.
  inline constexpr const char* unitNameOf(const int id) {
    switch (id) {
      case Draw: case DrawInstanced:
        return "vtx";
      case DrawIndexed: case DrawIdxInst:
        return "idx";
      case VSSetCB: case PSSetCB: case VSSetSRV: case PSSetSRV:
      case IASetVB: case OMSetRT:
        return "slots";
      default:
        return nullptr;
    }
  }

  // Replaces bump() at the entry points worth timing. Still bumps the original
  // counter so the vanish-diag baseline logic is unaffected.
  struct ScopedCall {
    CallId                                id;
    bool                                  timed;
    std::chrono::steady_clock::time_point t0;

    explicit ScopedCall(CallId i)
    : id(i), timed(t_isFrameThread) {
      g_counts[i].fetch_add(1, std::memory_order_relaxed);

      if (!timed)
        return;

      // ONE clock read serves BOTH the gap close and the body start.
      //
      // NV-DXVK [perf] 2026-08-06: this deliberately reverses the earlier
      // ordering decision, and the reason is that the call rate changed by two
      // orders of magnitude. That decision took a SECOND clock read here so the
      // gap atomics below would not be charged to the entry point's body time,
      // and it was correct when the note was written: ~6000 timed calls/frame,
      // where an extra read is invisible and the misattribution was measurable.
      //
      // It is now wrong. [Perf.Query] measures 12.69 MILLION GetData polls per
      // 5 s window - ~302,000 per frame on ~6 query objects, 99.998% not-ready,
      // i.e. the engine spinning on one EVENT query while the GPU works. Every
      // one of those pays this constructor. steady_clock::now() is QPC, ~41 ns
      // on this machine, so the extra read alone is ~12 ms PER FRAME to avoid
      // charging ~20 ns of relaxed atomics to the body. [GapSampler] confirms
      // it from the other side: ntdll!RtlQueryPerformanceCounter is 620+37+30
      // of 1891 frame-thread samples, ~36% of the thread, and this header is
      // the caller.
      //
      // The trade, stated plainly: we stop paying 41 ns of real frame time and
      // accept that the gap bookkeeping now lands inside the measured body, so
      // [Perf.Entry] reads slightly HIGH and [Perf.Gap] slightly LOW against
      // older logs. That error used to be the cost of five atomic RMWs; the
      // same change that made this worth doing also moved those to thread-local
      // adds (see flushDiagLocals), so what actually leaks into the body now is
      // a handful of plain increments - single-digit ns, well under the ~41 ns
      // this saves. The original objection is answered, not merely accepted.
      //
      // Two reads per call is the floor for measuring body and gap at all;
      // going below it needs a cheaper clock (rdtsc) or sampling, not a
      // reordering.
      const auto tEnter = std::chrono::steady_clock::now();

      // NV-DXVK [StallWatch] HEARTBEAT. One relaxed store on a clock read this
      // constructor already took, on the frame thread only.
      //
      // WHY IT LIVES HERE AND NOT IN A NEW HOOK: this is the one point every
      // D3D11 entry point already passes through with a timestamp in hand, so
      // the heartbeat is free. Anywhere else would need its own clock read, and
      // this header's own comment above explains what an extra read costs at
      // TF2's call rate.
      //
      // WHAT IT IS FOR: [Perf.Gap] reports afterGetData=12028ms AFTER the fact,
      // and [Perf.SyncSite] reads bursts=0 through every freeze -- so the game
      // is NOT spinning inside D3D11, it has left and not come back. Nothing
      // inside this DLL can see where it went, because by definition it is not
      // calling us. A watchdog thread reading this heartbeat can catch the
      // thread WHILE it is away and sample where it actually is.
      g_stallHeartbeatNs.store(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          tEnter.time_since_epoch()).count()),
        std::memory_order_relaxed);
      g_stallHeartbeatCall.store(uint32_t(i), std::memory_order_relaxed);
      // NOTE: the thread id is published where t_isFrameThread is SET, not here.
      // This header deliberately does not include windows.h (see the
      // t_isFrameThread comment above) and GetCurrentThreadId would drag it in.

      // [Perf.Gap]: close the interval opened by the previous call's exit. Only
      // at depth 0 - a nested entry point would otherwise report the enclosing
      // call's body as a gap.
      if (t_depth == 0 && t_haveLastExit) {
        const uint64_t gapNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          tEnter - t_lastExit).count());

        // Thread-local: see flushDiagLocals. These were `lock xadd` on every
        // one of ~302,000 polls per frame.
        t_accGapNs[t_lastCallId]    += gapNs;
        t_accGapCount[t_lastCallId] += 1;

        // Split the QueryEnd bucket by query type - see the comment on
        // g_gapNsByQueryType. Only QueryEnd needs this; the other call ids are
        // already one thing each.
        if (t_lastCallId == QueryEnd) {
          const uint32_t qt = t_lastQueryEndType & (kQueryTypeSlots - 1u);
          g_gapNsByQueryType[qt].fetch_add(gapNs, std::memory_order_relaxed);
          g_gapCountByQueryType[qt].fetch_add(1, std::memory_order_relaxed);

          uint64_t prevQMax = g_gapMaxNsByQueryType[qt].load(std::memory_order_relaxed);
          while (gapNs > prevQMax
              && !g_gapMaxNsByQueryType[qt].compare_exchange_weak(
                   prevQMax, gapNs, std::memory_order_relaxed)) { }

          // [Perf.Boundary]: for the TS_DISJOINT gap only - the frame boundary -
          // work out how much of it fell before Present and how much after.
          if (qt == 3u) {
            const uint64_t gapEndNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
              tEnter.time_since_epoch()).count());
            const uint64_t gapStartNs = gapEndNs - gapNs;
            const uint64_t pEnter = g_presentEnterNs.load(std::memory_order_relaxed);
            const uint64_t pExit  = g_presentExitNs.load(std::memory_order_relaxed);

            // Only attribute when a complete Present actually lies inside this
            // gap. If it does not, the boundary assumption is wrong for this
            // frame and counting it would invent a split that never happened -
            // g_boundaryNoPresent records how often that occurs so the result
            // can be trusted or discarded on its own evidence.
            if (pEnter >= gapStartNs && pExit <= gapEndNs && pExit >= pEnter) {
              const uint64_t preNs  = pEnter - gapStartNs;
              const uint64_t postNs = gapEndNs - pExit;

              g_boundaryPreNs.fetch_add(preNs, std::memory_order_relaxed);
              g_boundaryPostNs.fetch_add(postNs, std::memory_order_relaxed);
              g_boundaryCount.fetch_add(1, std::memory_order_relaxed);

              uint64_t pm = g_boundaryPreMaxNs.load(std::memory_order_relaxed);
              while (preNs > pm
                  && !g_boundaryPreMaxNs.compare_exchange_weak(pm, preNs, std::memory_order_relaxed)) { }
              uint64_t qm = g_boundaryPostMaxNs.load(std::memory_order_relaxed);
              while (postNs > qm
                  && !g_boundaryPostMaxNs.compare_exchange_weak(qm, postNs, std::memory_order_relaxed)) { }
            } else {
              g_boundaryNoPresent.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }

        // Thread-local max, folded into the shared one at drain time. Was an
        // atomic load plus a CAS loop per call.
        if (gapNs > t_accGapMaxNs) {
          t_accGapMaxNs   = gapNs;
          t_accGapMaxCall = t_lastCallId;
        }
      }

      ++t_depth;
      t0 = tEnter;   // reused - see the note above
    }

    ~ScopedCall() {
      if (!timed)
        return;

      const auto t1 = std::chrono::steady_clock::now();
      const auto dNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      // Thread-local: see flushDiagLocals.
      t_accTimeNs[id]    += uint64_t(dNs);
      t_accTimeCalls[id] += 1;

      // Open the next interval. Again depth 0 only, and reusing t1 rather than
      // taking a second reading - this path runs ~6000 times per frame and the
      // cost of measuring must stay far below what it measures.
      if (--t_depth == 0) {
        t_lastExit     = t1;
        t_haveLastExit = true;
        t_lastCallId   = id;
      }
    }

    ScopedCall(const ScopedCall&) = delete;
    ScopedCall& operator=(const ScopedCall&) = delete;
  };

  // Per-frame CopySubresourceRegion event log. Recorded at every CopySub
  // call regardless of frame, drained at EndFrame. Description strings are
  // captured at record time because resources may be Release()'d before the
  // frame ends; calling ID3D11Resource virtual methods on a dangling pointer
  // at drain time crashes.
  struct CopyEvent {
    void*    src;
    void*    dst;
    uint32_t srcSub;
    uint32_t dstSub;
    char     srcDesc[96];
    char     dstDesc[96];
  };
  // Defined in d3d11_context.cpp.
  extern std::mutex              g_copyMutex;
  extern std::vector<CopyEvent>  g_copyEvents;

  // Implemented in d3d11_context.cpp where d3d11.h types are in scope.
  void recordCopy(void* src, uint32_t srcSub, void* dst, uint32_t dstSub);

  inline void drainCopies(std::vector<CopyEvent>& out) {
    std::lock_guard<std::mutex> lk(g_copyMutex);
    out.swap(g_copyEvents);
  }

}}  // namespace dxvk::vanish_diag
