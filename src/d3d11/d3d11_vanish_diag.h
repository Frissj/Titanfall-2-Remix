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
  X(ExecCmdList) X(FinishCmdList) X(Flush)

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

  // Replaces bump() at the entry points worth timing. Still bumps the original
  // counter so the vanish-diag baseline logic is unaffected.
  struct ScopedCall {
    CallId                                id;
    bool                                  timed;
    std::chrono::steady_clock::time_point t0;

    explicit ScopedCall(CallId i)
    : id(i), timed(t_isFrameThread) {
      g_counts[i].fetch_add(1, std::memory_order_relaxed);
      if (timed)
        t0 = std::chrono::steady_clock::now();
    }

    ~ScopedCall() {
      if (!timed)
        return;

      const auto dNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
      g_timeNs[id].fetch_add(uint64_t(dNs), std::memory_order_relaxed);
      g_timeCalls[id].fetch_add(1, std::memory_order_relaxed);
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
