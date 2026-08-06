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

  inline void drainTimes(uint64_t outNs[CALL_COUNT], uint64_t outCalls[CALL_COUNT]) {
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

  inline void drainGaps(uint64_t outNs[CALL_COUNT], uint64_t outCounts[CALL_COUNT],
                        uint64_t& outMaxNs, int& outMaxCall) {
    for (int i = 0; i < CALL_COUNT; ++i) {
      outNs[i]     = g_gapNsByCall[i].exchange(0, std::memory_order_relaxed);
      outCounts[i] = g_gapCountByCall[i].exchange(0, std::memory_order_relaxed);
    }
    outMaxNs   = g_gapMaxNs.exchange(0, std::memory_order_relaxed);
    outMaxCall = g_gapMaxCall.load(std::memory_order_relaxed);
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

      // [Perf.Gap]: close the interval opened by the previous call's exit. Only
      // at depth 0 - a nested entry point would otherwise report the enclosing
      // call's body as a gap.
      //
      // Ordering is load-bearing: this bookkeeping runs BEFORE t0 is taken, so
      // the atomics below are not charged to the entry point's body time. The
      // first version of this took t0 first and inflated [Perf.Entry] from
      // 1.6 ms to 7.8 ms per frame - ~6000 gap computations charged to the
      // calls they were measuring. The extra clock read is far cheaper than the
      // atomics it excludes, and is only paid at depth 0.
      if (t_depth == 0 && t_haveLastExit) {
        const auto tEnter = std::chrono::steady_clock::now();
        const uint64_t gapNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          tEnter - t_lastExit).count());

        g_gapNsByCall[t_lastCallId].fetch_add(gapNs, std::memory_order_relaxed);
        g_gapCountByCall[t_lastCallId].fetch_add(1, std::memory_order_relaxed);

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

        uint64_t prevMax = g_gapMaxNs.load(std::memory_order_relaxed);
        while (gapNs > prevMax
            && !g_gapMaxNs.compare_exchange_weak(prevMax, gapNs, std::memory_order_relaxed)) { }
        if (gapNs >= prevMax)
          g_gapMaxCall.store(t_lastCallId, std::memory_order_relaxed);
      }

      ++t_depth;
      t0 = std::chrono::steady_clock::now();
    }

    ~ScopedCall() {
      if (!timed)
        return;

      const auto t1 = std::chrono::steady_clock::now();
      const auto dNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      g_timeNs[id].fetch_add(uint64_t(dNs), std::memory_order_relaxed);
      g_timeCalls[id].fetch_add(1, std::memory_order_relaxed);

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
