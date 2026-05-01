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
  X(CreateTex2D) X(DestroyTex2D) X(CreateBuf) X(DestroyBuf)

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
