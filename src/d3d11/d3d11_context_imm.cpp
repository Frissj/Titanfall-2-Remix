#include "d3d11_cmdlist.h"
#include "d3d11_context_imm.h"
#include "d3d11_device.h"
#include "d3d11_texture.h"
#include "d3d11_vanish_diag.h"

#include <atomic>
#include <chrono>
#include <cstdlib>

// NV-DXVK [perf]: the implicit-flush policy, made runtime-tunable.
//
// [Perf.Gpu] accounts for the whole frame exactly (idle + fenceWait + reap ==
// frame time, every window), and GPU idle tracks the number of command-list
// submissions almost linearly:
//
//   cmdLists/frame  13.0  15.3  21.7  22.8  23.2
//   gpuIdleMs        36    91   217   259   253
//
// So the frame is being chopped into many small submissions with the GPU
// bubbling between them. What that correlation cannot say is which way the
// causality runs: more submissions could be starving the GPU, or a frame that
// is already long could simply give the 750us flush timer more chances to fire.
// These knobs break the tie — raising them forces fewer, larger submissions
// without touching anything else, so if idle falls the fragmentation was the
// cause, and if it does not the submissions were a symptom.
//
// Env-read once so a sweep needs no rebuild. Defaults are the stock DXVK values.
static uint32_t GetFlushTuning(const char* name, uint32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0')
    return fallback;
  const long parsed = std::strtol(v, nullptr, 10);
  return (parsed > 0) ? uint32_t(parsed) : fallback;
}

static const uint32_t MinFlushIntervalUs = GetFlushTuning("RTX_FLUSH_MIN_US",      750);
static const uint32_t IncFlushIntervalUs = GetFlushTuning("RTX_FLUSH_INC_US",      250);
static const uint32_t MaxPendingSubmits  = GetFlushTuning("RTX_FLUSH_MAX_PENDING",   6);

constexpr static VkDeviceSize MaxImplicitDiscardSize = 256ull << 10;

#include "../dxvk/dxvk_bone_diag.h"

namespace dxvk {

  // NV-DXVK [Stage0 pipeline probe]: game-thread frame-boundary block accounting.
  // The draw phase (~191ms, game thread) and the RTX inject (~102ms, CS thread)
  // currently run serially (~370ms/frame, ~2.7fps). Before redesigning the frame
  // pipeline we must know WHERE the game thread actually stalls at the boundary.
  // These count the two primitive waits the game thread can block on. Written only
  // on the (relatively rare) blocking paths, read+reset once per window by the
  // D3D11Rtx perf logger as [Perf.GameBlock].
  std::atomic<int64_t>  g_gtWaitResNs   { 0 };  // ns blocked in device->waitForResource (GPU/Map contention)
  std::atomic<uint32_t> g_gtWaitResN    { 0 };  // # of blocking resource waits
  std::atomic<uint32_t> g_gtWaitResDiscN{ 0 };  // ...of those, DISCARD maps (discard-rename SHOULD have avoided the wait)
  std::atomic<int64_t>  g_gtSyncCsNs    { 0 };  // ns blocked in m_csThread.synchronize (CS-thread catch-up)
  std::atomic<uint32_t> g_gtSyncCsN     { 0 };  // # of CS syncs that actually waited (>10us)

  D3D11ImmediateContext::D3D11ImmediateContext(
          D3D11Device*    pParent,
    const Rc<DxvkDevice>& Device)
  : D3D11DeviceContext(pParent, Device, DxvkCsChunkFlag::SingleUse),
    m_csThread(Device, Device->createRtxContext()),
    m_videoContext(this, Device) {
    EmitCs([
      cDevice                 = m_device,
      cRelaxedBarriers        = pParent->GetOptions()->relaxedBarriers,
      cIgnoreGraphicsBarriers = pParent->GetOptions()->ignoreGraphicsBarriers
    ] (DxvkContext* ctx) {
      ctx->beginRecording(cDevice->createCommandList());

      DxvkBarrierControlFlags barrierControl;

      if (cRelaxedBarriers)
        barrierControl.set(DxvkBarrierControl::IgnoreWriteAfterWrite);

      if (cIgnoreGraphicsBarriers)
        barrierControl.set(DxvkBarrierControl::IgnoreGraphicsBarriers);

      ctx->setBarrierControl(barrierControl);
    });
    
    ClearState();

    m_rtx.Initialize();
  }
  
  
  D3D11ImmediateContext::~D3D11ImmediateContext() {
    Flush();
    SynchronizeCsThread(DxvkCsThread::SynchronizeAll);
    SynchronizeDevice();
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::QueryInterface(REFIID riid, void** ppvObject) {
    if (riid == __uuidof(ID3D11VideoContext)) {
      *ppvObject = ref(&m_videoContext);
      return S_OK;
    }

    return D3D11DeviceContext::QueryInterface(riid, ppvObject);
  }


  D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE D3D11ImmediateContext::GetType() {
    return D3D11_DEVICE_CONTEXT_IMMEDIATE;
  }
  
  
  UINT STDMETHODCALLTYPE D3D11ImmediateContext::GetContextFlags() {
    return 0;
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::GetData(
          ID3D11Asynchronous*               pAsync,
          void*                             pData,
          UINT                              DataSize,
          UINT                              GetDataFlags) {
    vanish_diag::ScopedCall vdScope_GetData(vanish_diag::GetData);
    if (!pAsync || (DataSize && !pData))
      return E_INVALIDARG;
    
    // Check whether the data size is actually correct
    if (DataSize && DataSize != pAsync->GetDataSize())
      return E_INVALIDARG;
    
    // Passing a non-null pData is actually allowed if
    // DataSize is 0, but we should ignore that pointer
    pData = DataSize ? pData : nullptr;

    // Get query status directly from the query object
    auto query = static_cast<D3D11Query*>(pAsync);
    HRESULT hr = query->GetData(pData, GetDataFlags);

    // NV-DXVK [perf]: [Perf.Entry] measured ~1e6 GetData calls per frame costing
    // 80-130 ms — the largest single item in the frame, bigger than all of
    // OnDraw* or EndFrame. That is the game spinning on a query that is not
    // becoming ready. Attribute it: which query type, how many distinct query
    // objects, what fraction of polls return S_FALSE (not-ready), and whether the
    // game is passing DONOTFLUSH (which suppresses our stall handling). One line
    // per 5 s, on the frame thread only.
    {
      constexpr uint32_t kQueryTypes = 16;  // covers the whole D3D11_QUERY enum

      static thread_local uint64_t s_pollTotal = 0, s_pollNotReady = 0, s_pollDoNotFlush = 0;
      static thread_local uint64_t s_byType[kQueryTypes] = {};
      static thread_local const void* s_lastQuery = nullptr;
      static thread_local uint32_t s_lastQueryType = 0;
      static thread_local uint64_t s_queryChanges = 0;
      static thread_local auto s_lastLog = dxvk::high_resolution_clock::now();

      // GetDesc1 is a virtual call plus a struct copy; at ~1e6 calls/frame that
      // would itself distort what we are measuring. The type is a property of the
      // query object, and a spin loop hammers the same object, so memoise on the
      // pointer — GetDesc1 then runs once per switch, not once per poll.
      if (pAsync != s_lastQuery) {
        s_lastQuery = pAsync;
        ++s_queryChanges;

        D3D11_QUERY_DESC1 qd = {};
        query->GetDesc1(&qd);
        s_lastQueryType = (uint32_t(qd.Query) < kQueryTypes) ? uint32_t(qd.Query) : 0u;
      }

      ++s_pollTotal;
      ++s_byType[s_lastQueryType];
      if (hr == S_FALSE)                                  ++s_pollNotReady;
      if (GetDataFlags & D3D11_ASYNC_GETDATA_DONOTFLUSH)  ++s_pollDoNotFlush;

      // Reading the clock on every poll would cost more than everything else
      // here combined (~25 ns x 1e6 calls = ~25 ms/frame). Check it once every
      // 4096 polls instead; at these rates that is still many checks per frame.
      if ((s_pollTotal & 0xFFFu) == 0u
       && dxvk::high_resolution_clock::now() - s_lastLog >= std::chrono::seconds(5)) {
        s_lastLog = dxvk::high_resolution_clock::now();

        std::string types;
        for (uint32_t t = 0; t < kQueryTypes; ++t) {
          if (s_byType[t] != 0)
            types += str::format(" type", t, "=", s_byType[t]);
        }

        Logger::warn(str::format(
          "[Perf.Query] flushTuning(min=", MinFlushIntervalUs,
          " inc=", IncFlushIntervalUs,
          " maxPending=", MaxPendingSubmits, ")",
          " polls=", s_pollTotal,
          " notReady=", s_pollNotReady,
          " notReadyPct=", (s_pollTotal ? double(s_pollNotReady) * 100.0 / double(s_pollTotal) : 0.0),
          " doNotFlush=", s_pollDoNotFlush,
          " objSwitches=", s_queryChanges,
          " |", types,
          "  (type0=EVENT 1=OCCLUSION 2=TIMESTAMP 5=OCCLUSION_PREDICATE)"));

        s_pollTotal = s_pollNotReady = s_pollDoNotFlush = s_queryChanges = 0;
        for (auto& c : s_byType) c = 0;
      }
    }

    // If we're likely going to spin on the asynchronous object,
    // flush the context so that we're keeping the GPU busy.
    if (hr == S_FALSE) {
      // Don't mark the event query as stalling if the app does
      // not intend to spin on it. This reduces flushes on End.
      if (!(GetDataFlags & D3D11_ASYNC_GETDATA_DONOTFLUSH))
        query->NotifyStall();

      // Ignore the DONOTFLUSH flag here as some games will spin
      // on queries without ever flushing the context otherwise.
      FlushImplicit(FALSE);
    }
    
    return hr;
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::Begin(ID3D11Asynchronous* pAsync) {
    vanish_diag::ScopedCall vdScope_QueryBegin(vanish_diag::QueryBegin);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pAsync))
      return;

    auto query = static_cast<D3D11Query*>(pAsync);

    if (unlikely(!query->DoBegin()))
      return;

    EmitCs([cQuery = Com<D3D11Query, false>(query)]
    (DxvkContext* ctx) {
      cQuery->Begin(ctx);
    });
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::End(ID3D11Asynchronous* pAsync) {
    vanish_diag::ScopedCall vdScope_QueryEnd(vanish_diag::QueryEnd);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pAsync))
      return;
    
    auto query = static_cast<D3D11Query*>(pAsync);

    if (unlikely(!query->DoEnd())) {
      EmitCs([cQuery = Com<D3D11Query, false>(query)]
      (DxvkContext* ctx) {
        cQuery->Begin(ctx);
      });
    }

    EmitCs([cQuery = Com<D3D11Query, false>(query)]
    (DxvkContext* ctx) {
      cQuery->End(ctx);
    });

    if (unlikely(query->IsEvent())) {
      query->NotifyEnd();
      query->IsStalling()
        ? Flush()
        : FlushImplicit(TRUE);
    }
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::Flush() {
    Flush1(D3D11_CONTEXT_TYPE_ALL, nullptr);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::Flush1(
          D3D11_CONTEXT_TYPE          ContextType,
          HANDLE                      hEvent) {
    m_parent->FlushInitContext();

    if (hEvent)
      SignalEvent(hEvent);
    
    D3D11DeviceLock lock = LockContext();
    
    if (m_csIsBusy || !m_csChunk->empty()) {
      // Add commands to flush the threaded
      // context, then flush the command list
      EmitCs([] (DxvkContext* ctx) {
        ctx->flushCommandList();
      });
      
      FlushCsChunk();
      
      // Reset flush timer used for implicit flushes
      m_lastFlush = dxvk::high_resolution_clock::now();
      m_csIsBusy  = false;
    }
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Signal(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    Logger::err("D3D11ImmediateContext::Signal: Not implemented");
    return E_NOTIMPL;
  }


  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Wait(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    Logger::err("D3D11ImmediateContext::Wait: Not implemented");
    return E_NOTIMPL;
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::ExecuteCommandList(
          ID3D11CommandList*  pCommandList,
          BOOL                RestoreContextState) {
    vanish_diag::ScopedCall vdScope(vanish_diag::ExecCmdList);
    D3D11DeviceLock lock = LockContext();

    auto commandList = static_cast<D3D11CommandList*>(pCommandList);

    // Flush any outstanding commands so that
    // we don't mess up the execution order
    FlushCsChunk();

    // As an optimization, flush everything if the
    // number of pending draw calls is high enough.
    FlushImplicit(FALSE);

    // NV-DXVK: Fold the deferred-context RTX draw count recorded into the
    // command list (see D3D11DeferredContext::FinishCommandList) into our
    // own D3D11Rtx counter BEFORE replaying the chunks.  The chunks carry
    // EmitCs lambdas that call commitGeometryToRT on the CS thread, so the
    // geometry itself reaches the RTX pipeline via the normal CS path; the
    // only thing the immediate context needs to know is the total draw
    // count so that D3D11Rtx::EndFrame sees a non-zero value and the
    // kMaxConcurrentDraws throttle stays accurate.  This is what makes
    // Source-engine games (Titanfall 2, etc.) that record all of their
    // material draws onto worker-thread deferred contexts actually show
    // up in Remix's raytraced composite.
    const uint32_t clRtxDraws = commandList->GetRtxDrawCount();
    m_rtx.addDrawCallID(clRtxDraws);

    // NV-DXVK: diagnostic — first few ExecuteCommandList calls log the
    // recorded RTX draw count so we can see whether deferred contexts are
    // actually producing draws that pass SubmitDraw's pre-filters.
    static uint32_t s_execCount = 0;
    const uint32_t n = ++s_execCount;
    if (n <= 8) {
      Logger::info(str::format(
          "[D3D11ImmediateContext] ExecuteCommandList #", n,
          " rtxDraws=", clRtxDraws));
    }

    // NV-DXVK TF2 viewmodel diagnostic: snapshot the immediate context's
    // current VS cb2 contents at ExecuteCommandList time, BEFORE the
    // recorded deferred chunks fire. Compare against [VMHunt.cb2] which
    // captures cb2 at the deferred context's SubmitDraw recording time.
    //
    // Hypothesis: TF2 records gun + hands draws on a deferred context. Our
    // [VMHunt.cb2] capture happens on the deferred-context thread when the
    // draw is recorded. By the time ExecuteCommandList replays those chunks
    // on the immediate context, the game may have updated cb2 with the
    // CORRECT viewmodel camera (different origin, smaller maxZ frustum) via
    // the immediate context — and the GPU sees that update because cb
    // bindings carry through D3D11's versioning. If the immediate context's
    // cb2 origin here differs from what [VMHunt.cb2] logged for nearby
    // draws, we've been computing the viewmodel transform on stale data.
    //
    // Only logs when cb2 appears to hold a CBufCommonPerCamera-shaped
    // payload (offset 0 = zNear small, offset 4 = float3 cam origin
    // plausible). Throttled to ~120 calls per session and to changes (skip
    // if cb2 looks identical to the previous logged value) so this doesn't
    // flood the log.
    {
      static uint32_t sCb2LogCount = 0;
      static float sLastCamX = 1e30f, sLastCamY = 1e30f, sLastCamZ = 1e30f;
      const auto& vsCbs = m_state.vs.constantBuffers;
      // cb slot 2 is where Source engine binds CBufCommonPerCamera in TF2.
      if (sCb2LogCount < 120) {
        const auto& cb2 = vsCbs[2];
        if (cb2.buffer != nullptr) {
          const auto map = cb2.buffer->GetMappedSlice();
          const uint8_t* p = reinterpret_cast<const uint8_t*>(map.mapPtr);
          const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
          const size_t bufSize = cb2.buffer->Desc()->ByteWidth;
          // We need at least 16 bytes (zNear + camOrigin float3) to be
          // worth logging. Read 80 bytes if available so we can also dump
          // the c_cameraRelativeToClip first row for cross-checking.
          if (p && base + 16 <= bufSize) {
            const float* fp = reinterpret_cast<const float*>(p + base);
            const float zNear = fp[0];
            const float cx = fp[1], cy = fp[2], cz = fp[3];
            const bool plausible =
              std::isfinite(zNear) && std::isfinite(cx)
              && std::isfinite(cy) && std::isfinite(cz)
              && (std::abs(cx) > 1.0f || std::abs(cy) > 1.0f || std::abs(cz) > 1.0f);
            const bool changed =
              std::abs(sLastCamX - cx) > 0.5f
              || std::abs(sLastCamY - cy) > 0.5f
              || std::abs(sLastCamZ - cz) > 0.5f;
            if (plausible && changed) {
              ++sCb2LogCount;
              sLastCamX = cx; sLastCamY = cy; sLastCamZ = cz;
              // Also dump c2c row0 if buffer is large enough — helps confirm
              // whether projection matches the gameplay or viewmodel cam.
              float r0x = 0, r0y = 0, r0z = 0, r0w = 0;
              if (base + 32 <= bufSize) {
                const float* r0 = reinterpret_cast<const float*>(p + base + 16);
                r0x = r0[0]; r0y = r0[1]; r0z = r0[2]; r0w = r0[3];
              }
              Logger::info(str::format(
                "[ExecCL.cb2] #", sCb2LogCount,
                " execId=", n,
                " rtxDraws=", clRtxDraws,
                " cb2.constOff=", cb2.constantOffset,
                " cb2.bufSize=", bufSize,
                " zNear=", zNear,
                " camOrigin=(", cx, ",", cy, ",", cz, ")",
                " c2c_row0=(", r0x, ",", r0y, ",", r0z, ",", r0w, ")"));
            }
          }
        }
      }
    }

    // Dispatch command list to the CS thread and
    // restore the immediate context's state
    uint64_t csSeqNum = commandList->EmitToCsThread(&m_csThread);
    m_csSeqNum = std::max(m_csSeqNum, csSeqNum);

    if (RestoreContextState)
      RestoreState();
    else
      ClearState();

    // Mark CS thread as busy so that subsequent
    // flush operations get executed correctly.
    m_csIsBusy = true;
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::FinishCommandList(
          BOOL                RestoreDeferredContextState,
          ID3D11CommandList   **ppCommandList) {
    InitReturnPtr(ppCommandList);
    
    Logger::err("D3D11: FinishCommandList called on immediate context");
    return DXGI_ERROR_INVALID_CALL;
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Map(
          ID3D11Resource*             pResource,
          UINT                        Subresource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    vanish_diag::ScopedCall vdScope(vanish_diag::Map);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pResource))
      return E_INVALIDARG;
    
    D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resourceDim);

    HRESULT hr;
    
    if (likely(resourceDim == D3D11_RESOURCE_DIMENSION_BUFFER)) {
      hr = MapBuffer(
        static_cast<D3D11Buffer*>(pResource),
        MapType, MapFlags, pMappedResource);
      // NV-DXVK: log Map calls on t30-sized buffers to find the bone
      // upload pattern (TF2's skinned-character bone matrices).
      auto* b = static_cast<D3D11Buffer*>(pResource);
      const uint32_t sz = b->Desc()->ByteWidth;
      if (tf2::boneDiagEnabled() && sz == 393216) {
        static uint32_t sMapBoneLog = 0;
        if (sMapBoneLog < 20) {
          ++sMapBoneLog;
          Logger::info(str::format(
            "[BoneMap.diag] Map t30-sized buffer",
            " ptr=", reinterpret_cast<uintptr_t>(b),
            " mapType=", uint32_t(MapType),
            " usage=", uint32_t(b->Desc()->Usage),
            " bindFlags=", uint32_t(b->Desc()->BindFlags),
            " mapPtrResult=", reinterpret_cast<uintptr_t>(pMappedResource ? pMappedResource->pData : nullptr)));
        }
      }
    } else {
      hr = MapImage(
        GetCommonTexture(pResource),
        Subresource, MapType, MapFlags,
        pMappedResource);
    }

    if (unlikely(FAILED(hr)))
      *pMappedResource = D3D11_MAPPED_SUBRESOURCE();

    return hr;
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::Unmap(
          ID3D11Resource*             pResource,
          UINT                        Subresource) {
    vanish_diag::ScopedCall vdScope(vanish_diag::Unmap);
    // Since it is very uncommon for images to be mapped compared
    // to buffers, we count the currently mapped images in order
    // to avoid a virtual method call in the common case.
    if (unlikely(m_mappedImageCount > 0)) {
      D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      pResource->GetType(&resourceDim);

      if (resourceDim != D3D11_RESOURCE_DIMENSION_BUFFER) {
        D3D11DeviceLock lock = LockContext();
        UnmapImage(GetCommonTexture(pResource), Subresource);
      }
    }
  }

  void STDMETHODCALLTYPE D3D11ImmediateContext::UpdateSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch) {
    vanish_diag::ScopedCall vdScope(vanish_diag::UpdateSub);
    UpdateResource<D3D11ImmediateContext>(this, pDstResource,
      DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, 0);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::UpdateSubresource1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    vanish_diag::ScopedCall vdScope_UpdateSub(vanish_diag::UpdateSub);
    UpdateResource<D3D11ImmediateContext>(this, pDstResource,
      DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::OMSetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView) {
    FlushImplicit(TRUE);
    
    D3D11DeviceContext::OMSetRenderTargets(
      NumViews, ppRenderTargetViews, pDepthStencilView);
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::OMSetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    FlushImplicit(TRUE);

    D3D11DeviceContext::OMSetRenderTargetsAndUnorderedAccessViews(
      NumRTVs, ppRenderTargetViews, pDepthStencilView,
      UAVStartSlot, NumUAVs, ppUnorderedAccessViews,
      pUAVInitialCounts);
  }
  
  
  HRESULT D3D11ImmediateContext::MapBuffer(
          D3D11Buffer*                pResource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    if (unlikely(!pMappedResource))
      return E_INVALIDARG;

    if (unlikely(pResource->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_NONE)) {
      Logger::err("D3D11: Cannot map a device-local buffer");
      return E_INVALIDARG;
    }

    VkDeviceSize bufferSize = pResource->Desc()->ByteWidth;

    if (likely(MapType == D3D11_MAP_WRITE_DISCARD)) {
      // Allocate a new backing slice for the buffer and set
      // it as the 'new' mapped slice. This assumes that the
      // only way to invalidate a buffer is by mapping it.
      auto physSlice = pResource->DiscardSlice();
      pMappedResource->pData      = physSlice.mapPtr;
      pMappedResource->RowPitch   = bufferSize;
      pMappedResource->DepthPitch = bufferSize;
      
      EmitCs([
        cBuffer      = pResource->GetBuffer(),
        cBufferSlice = physSlice
      ] (DxvkContext* ctx) {
        ctx->invalidateBuffer(cBuffer, cBufferSlice);
      });

      return S_OK;
    } else if (likely(MapType == D3D11_MAP_WRITE_NO_OVERWRITE)) {
      // Put this on a fast path without any extra checks since it's
      // a somewhat desired method to partially update large buffers
      DxvkBufferSliceHandle physSlice = pResource->GetMappedSlice();
      pMappedResource->pData      = physSlice.mapPtr;
      pMappedResource->RowPitch   = bufferSize;
      pMappedResource->DepthPitch = bufferSize;
      return S_OK;
    } else {
      // Quantum Break likes using MAP_WRITE on resources which would force
      // us to synchronize with the GPU multiple times per frame. In those
      // situations, if there are no pending GPU writes to the resource, we
      // can promote it to MAP_WRITE_DISCARD, but preserve the data by doing
      // a CPU copy from the previous buffer slice, to avoid the sync point.
      bool doInvalidatePreserve = false;

      auto buffer = pResource->GetBuffer();
      auto sequenceNumber = pResource->GetSequenceNumber();

      if (MapType != D3D11_MAP_READ && !MapFlags && bufferSize <= MaxImplicitDiscardSize) {
        SynchronizeCsThread(sequenceNumber);

        bool hasWoAccess = buffer->isInUse(DxvkAccess::Write);
        bool hasRwAccess = buffer->isInUse(DxvkAccess::Read);

        if (hasRwAccess && !hasWoAccess) {
          // Uncached reads can be so slow that a GPU sync may actually be faster
          doInvalidatePreserve = buffer->memFlags() & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
      }

      if (doInvalidatePreserve) {
        FlushImplicit(TRUE);

        auto prevSlice = pResource->GetMappedSlice();
        auto physSlice = pResource->DiscardSlice();

        EmitCs([
          cBuffer      = std::move(buffer),
          cBufferSlice = physSlice
        ] (DxvkContext* ctx) {
          ctx->invalidateBuffer(cBuffer, cBufferSlice);
        });

        std::memcpy(physSlice.mapPtr, prevSlice.mapPtr, physSlice.length);
        pMappedResource->pData      = physSlice.mapPtr;
        pMappedResource->RowPitch   = bufferSize;
        pMappedResource->DepthPitch = bufferSize;
        return S_OK;
      } else {
        if (!WaitForResource(buffer, sequenceNumber, MapType, MapFlags))
          return DXGI_ERROR_WAS_STILL_DRAWING;

        DxvkBufferSliceHandle physSlice = pResource->GetMappedSlice();
        pMappedResource->pData      = physSlice.mapPtr;
        pMappedResource->RowPitch   = bufferSize;
        pMappedResource->DepthPitch = bufferSize;
        return S_OK;
      }
    }
  }
  
  
  HRESULT D3D11ImmediateContext::MapImage(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    const Rc<DxvkImage>  mappedImage  = pResource->GetImage();
    const Rc<DxvkBuffer> mappedBuffer = pResource->GetMappedBuffer(Subresource);

    auto mapMode = pResource->GetMapMode();
    
    if (unlikely(mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_NONE)) {
      Logger::err("D3D11: Cannot map a device-local image");
      return E_INVALIDARG;
    }

    if (unlikely(Subresource >= pResource->CountSubresources()))
      return E_INVALIDARG;
    
    if (likely(pMappedResource != nullptr)) {
      // Resources with an unknown memory layout cannot return a pointer
      if (pResource->Desc()->Usage         == D3D11_USAGE_DEFAULT
       && pResource->Desc()->TextureLayout == D3D11_TEXTURE_LAYOUT_UNDEFINED)
        return E_INVALIDARG;
    } else {
      if (pResource->Desc()->Usage != D3D11_USAGE_DEFAULT)
        return E_INVALIDARG;
    }

    VkFormat packedFormat = m_parent->LookupPackedFormat(
      pResource->Desc()->Format, pResource->GetFormatMode()).Format;
    
    uint64_t sequenceNumber = pResource->GetSequenceNumber(Subresource);

    auto formatInfo = imageFormatInfo(packedFormat);
    void* mapPtr;

    if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_DIRECT) {
      // Wait for the resource to become available. We do not
      // support image renaming, so stall on DISCARD instead.
      if (MapType == D3D11_MAP_WRITE_DISCARD)
        MapFlags &= ~D3D11_MAP_FLAG_DO_NOT_WAIT;

      if (MapType != D3D11_MAP_WRITE_NO_OVERWRITE) {
        if (!WaitForResource(mappedImage, sequenceNumber, MapType, MapFlags))
          return DXGI_ERROR_WAS_STILL_DRAWING;
      }
      
      // Query the subresource's memory layout and hope that
      // the application respects the returned pitch values.
      mapPtr = mappedImage->mapPtr(0);
    } else {
      constexpr uint32_t DoInvalidate = (1u << 0);
      constexpr uint32_t DoPreserve   = (1u << 1);
      constexpr uint32_t DoWait       = (1u << 2);
      uint32_t doFlags;

      if (MapType == D3D11_MAP_READ) {
        // Reads will not change the image content, so we only need
        // to wait for the GPU to finish writing to the mapped buffer.
        doFlags = DoWait;
      } else if (MapType == D3D11_MAP_WRITE_DISCARD) {
        doFlags = DoInvalidate;

        // If we know for sure that the mapped buffer is currently not
        // in use by the GPU, we don't have to allocate a new slice.
        if (m_csThread.lastSequenceNumber() >= sequenceNumber && !mappedBuffer->isInUse(DxvkAccess::Read))
          doFlags = 0;
      } else if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_STAGING && (MapFlags & D3D11_MAP_FLAG_DO_NOT_WAIT)) {
        // Always respect DO_NOT_WAIT for mapped staging images
        doFlags = DoWait;
      } else if (MapType != D3D11_MAP_WRITE_NO_OVERWRITE || mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER) {
        // Need to synchronize thread to determine pending GPU accesses
        SynchronizeCsThread(sequenceNumber);

        // Don't implicitly discard large buffers or buffers of images with
        // multiple subresources, as that is likely to cause memory issues.
        VkDeviceSize bufferSize = pResource->CountSubresources() == 1
          ? pResource->GetMappedSlice(Subresource).length
          : MaxImplicitDiscardSize;

        if (bufferSize >= MaxImplicitDiscardSize) {
          // Don't check access flags, WaitForResource will return
          // early anyway if the resource is currently in use
          doFlags = DoWait;
        } else if (mappedBuffer->isInUse(DxvkAccess::Write)) {
          // There are pending GPU writes, need to wait for those
          doFlags = DoWait;
        } else if (mappedBuffer->isInUse(DxvkAccess::Read)) {
          // All pending GPU accesses are reads, so the buffer data
          // is still current, and we can prevent GPU synchronization
          // by creating a new slice with an exact copy of the data.
          doFlags = DoInvalidate | DoPreserve;
        } else {
          // There are no pending accesses, so we don't need to wait
          doFlags = 0;
        }
      } else {
        // No need to synchronize staging resources with NO_OVERWRITE
        // since the buffer will be used directly.
        doFlags = 0;
      }

      if (doFlags & DoInvalidate) {
        FlushImplicit(TRUE);

        DxvkBufferSliceHandle prevSlice = pResource->GetMappedSlice(Subresource);
        DxvkBufferSliceHandle physSlice = pResource->DiscardSlice(Subresource);

        EmitCs([
          cImageBuffer = mappedBuffer,
          cBufferSlice = physSlice
        ] (DxvkContext* ctx) {
          ctx->invalidateBuffer(cImageBuffer, cBufferSlice);
        });

        if (doFlags & DoPreserve)
          std::memcpy(physSlice.mapPtr, prevSlice.mapPtr, physSlice.length);

        mapPtr = physSlice.mapPtr;
      } else {
        if (doFlags & DoWait) {
          // We cannot respect DO_NOT_WAIT for buffer-mapped resources since
          // our internal copies need to be transparent to the application.
          if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER)
            MapFlags &= ~D3D11_MAP_FLAG_DO_NOT_WAIT;

          // Wait for mapped buffer to become available
          if (!WaitForResource(mappedBuffer, sequenceNumber, MapType, MapFlags))
            return DXGI_ERROR_WAS_STILL_DRAWING;
        }

        mapPtr = pResource->GetMappedSlice(Subresource).mapPtr;
      }
    }

    // Mark the given subresource as mapped
    pResource->SetMapType(Subresource, MapType);

    if (pMappedResource) {
      auto layout = pResource->GetSubresourceLayout(formatInfo->aspectMask, Subresource);
      pMappedResource->pData      = reinterpret_cast<char*>(mapPtr) + layout.Offset;
      pMappedResource->RowPitch   = layout.RowPitch;
      pMappedResource->DepthPitch = layout.DepthPitch;
    }

    m_mappedImageCount += 1;
    return S_OK;
  }
  
  
  void D3D11ImmediateContext::UnmapImage(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource) {
    D3D11_MAP mapType = pResource->GetMapType(Subresource);
    pResource->SetMapType(Subresource, D3D11_MAP(~0u));

    if (mapType == D3D11_MAP(~0u))
      return;

    // Decrement mapped image counter only after making sure
    // the given subresource is actually mapped right now
    m_mappedImageCount -= 1;

    if ((mapType != D3D11_MAP_READ) &&
        (pResource->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER)) {
      // Now that data has been written into the buffer,
      // we need to copy its contents into the image
      VkImageAspectFlags aspectMask = imageFormatInfo(pResource->GetPackedFormat())->aspectMask;
      VkImageSubresource subresource = pResource->GetSubresourceFromIndex(aspectMask, Subresource);

      UpdateImage(pResource, &subresource, VkOffset3D { 0, 0, 0 },
        pResource->MipLevelExtent(subresource.mipLevel),
        DxvkBufferSlice(pResource->GetMappedBuffer(Subresource)));
    }
  }
  
  
  void D3D11ImmediateContext::UpdateMappedBuffer(
          D3D11Buffer*                  pDstBuffer,
          UINT                          Offset,
          UINT                          Length,
    const void*                         pSrcData,
          UINT                          CopyFlags) {
    DxvkBufferSliceHandle slice;

    if (likely(CopyFlags != D3D11_COPY_NO_OVERWRITE)) {
      slice = pDstBuffer->DiscardSlice();

      EmitCs([
        cBuffer      = pDstBuffer->GetBuffer(),
        cBufferSlice = slice
      ] (DxvkContext* ctx) {
        ctx->invalidateBuffer(cBuffer, cBufferSlice);
      });
    } else {
      slice = pDstBuffer->GetMappedSlice();
    }

    std::memcpy(reinterpret_cast<char*>(slice.mapPtr) + Offset, pSrcData, Length);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::SwapDeviceContextState(
          ID3DDeviceContextState*           pState,
          ID3DDeviceContextState**          ppPreviousState) {
    InitReturnPtr(ppPreviousState);

    if (!pState)
      return;
    
    Com<D3D11DeviceContextState> oldState = std::move(m_stateObject);
    Com<D3D11DeviceContextState> newState = static_cast<D3D11DeviceContextState*>(pState);

    if (oldState == nullptr)
      oldState = new D3D11DeviceContextState(m_parent);
    
    if (ppPreviousState)
      *ppPreviousState = oldState.ref();
    
    m_stateObject = newState;

    oldState->SetState(m_state);
    newState->GetState(m_state);

    RestoreState();
  }


  void D3D11ImmediateContext::SynchronizeCsThread(uint64_t SequenceNumber) {
    D3D11DeviceLock lock = LockContext();

    // Dispatch current chunk so that all commands
    // recorded prior to this function will be run
    if (SequenceNumber > m_csSeqNum)
      FlushCsChunk();

    // NV-DXVK [Stage0 probe]: time the CS-thread catch-up wait. This is the
    // game thread waiting for the CS thread (which runs injectRTX) to reach a
    // sequence number — a candidate serialization point at the frame boundary.
    const auto tSync0 = std::chrono::steady_clock::now();
    m_csThread.synchronize(SequenceNumber);
    const auto tSync1 = std::chrono::steady_clock::now();
    const int64_t syncNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tSync1 - tSync0).count();
    g_gtSyncCsNs.fetch_add(syncNs, std::memory_order_relaxed);
    if (syncNs > 10000)
      g_gtSyncCsN.fetch_add(1u, std::memory_order_relaxed);
  }
  
  
  void D3D11ImmediateContext::SynchronizeDevice() {
    m_device->waitForIdle();
  }
  
  
  bool D3D11ImmediateContext::WaitForResource(
    const Rc<DxvkResource>&                 Resource,
          uint64_t                          SequenceNumber,
          D3D11_MAP                         MapType,
          UINT                              MapFlags) {
    // Determine access type to wait for based on map mode
    DxvkAccess access = MapType == D3D11_MAP_READ
      ? DxvkAccess::Write
      : DxvkAccess::Read;
    
    // Wait for any CS chunk using the resource to execute, since
    // otherwise we cannot accurately determine if the resource is
    // actually being used by the GPU right now.
    bool isInUse = Resource->isInUse(access);

    if (!isInUse) {
      SynchronizeCsThread(SequenceNumber);
      isInUse = Resource->isInUse(access);
    }

    if (MapFlags & D3D11_MAP_FLAG_DO_NOT_WAIT) {
      if (isInUse) {
        // We don't have to wait, but misbehaving games may
        // still try to spin on `Map` until the resource is
        // idle, so we should flush pending commands
        FlushImplicit(FALSE);
        return false;
      }
    } else {
      if (isInUse) {
        // Make sure pending commands using the resource get
        // executed on the the GPU if we have to wait for it
        Flush();
        SynchronizeCsThread(SequenceNumber);

        // NV-DXVK [Stage0 probe]: time the GPU resource wait. This is the block
        // that serializes a next-frame Map behind the CS-thread RTX inject still
        // reading the resource. A DISCARD map reaching here means discard-rename
        // was defeated (e.g. a pinned slice) — the prime suspect for the draw/
        // inject serialization.
        const auto tW0 = std::chrono::steady_clock::now();
        m_device->waitForResource(Resource, access);
        const auto tW1 = std::chrono::steady_clock::now();
        g_gtWaitResNs.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(tW1 - tW0).count(),
          std::memory_order_relaxed);
        g_gtWaitResN.fetch_add(1u, std::memory_order_relaxed);
        if (MapType == D3D11_MAP_WRITE_DISCARD)
          g_gtWaitResDiscN.fetch_add(1u, std::memory_order_relaxed);
      }
    }

    return true;
  }
  
  
  void D3D11ImmediateContext::EmitCsChunk(DxvkCsChunkRef&& chunk) {
    m_csSeqNum = m_csThread.dispatchChunk(std::move(chunk));
    m_csIsBusy = true;
  }


  void D3D11ImmediateContext::TrackTextureSequenceNumber(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource) {
    pResource->TrackSequenceNumber(Subresource, m_csSeqNum + 1);
  }


  void D3D11ImmediateContext::TrackBufferSequenceNumber(
          D3D11Buffer*                pResource) {
    pResource->TrackSequenceNumber(m_csSeqNum + 1);
  }


  void D3D11ImmediateContext::FlushImplicit(BOOL StrongHint) {
    // Flush only if the GPU is about to go idle, in
    // order to keep the number of submissions low.
    uint32_t pending = m_device->pendingSubmissions();

    if (StrongHint || pending <= MaxPendingSubmits) {
      auto now = dxvk::high_resolution_clock::now();

      uint32_t delay = MinFlushIntervalUs
                     + IncFlushIntervalUs * pending;

      // Prevent flushing too often in short intervals.
      if (now - m_lastFlush >= std::chrono::microseconds(delay))
        Flush();
    }
  }


  void D3D11ImmediateContext::SignalEvent(HANDLE hEvent) {
    uint64_t value = ++m_eventCount;

    if (m_eventSignal == nullptr)
      m_eventSignal = new sync::CallbackFence();

    m_eventSignal->setCallback(value, [hEvent] {
      SetEvent(hEvent);
    });

    EmitCs([
      cSignal = m_eventSignal,
      cValue  = value
    ] (DxvkContext* ctx) {
      ctx->signal(cSignal, cValue);
    });
  }
  
}
