#pragma once

#include "d3d11_context.h"

namespace dxvk {
  
  class D3D11CommandList : public D3D11DeviceChild<ID3D11CommandList> {
    
  public:
    
    D3D11CommandList(
            D3D11Device*  pDevice,
            UINT          ContextFlags);
    
    ~D3D11CommandList();
    
    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject) final;
    
    UINT STDMETHODCALLTYPE GetContextFlags() final;
    
    void AddChunk(
            DxvkCsChunkRef&&    Chunk);

    void AddQuery(
            D3D11Query*         pQuery);
    
    void EmitToCommandList(
            ID3D11CommandList*  pCommandList);
    
    uint64_t EmitToCsThread(
            DxvkCsThread*       CsThread);

    void TrackResourceUsage(
            ID3D11Resource*     pResource,
            D3D11_RESOURCE_DIMENSION ResourceType,
            UINT                Subresource);

    // NV-DXVK: Carry the RTX draw count that the deferred context
    // accumulated into its D3D11Rtx while recording this command list, so
    // that the immediate context can fold it into its own counter on
    // ExecuteCommandList and D3D11Rtx::EndFrame reports the true per-frame
    // total.  Without this transfer, draws recorded on deferred contexts
    // (which is where Source's materialsystem_dx11 does most of its work)
    // never contribute to the immediate context's draw counter, the
    // kMaxConcurrentDraws throttle is stuck at 0, and EndFrame prints
    // draws=0 every frame.
    void SetRtxDrawCount(uint32_t count) { m_rtxDrawCount = count; }
    uint32_t GetRtxDrawCount() const     { return m_rtxDrawCount; }

  private:

    UINT         const m_contextFlags;

    std::vector<DxvkCsChunkRef>         m_chunks;
    std::vector<Com<D3D11Query, false>> m_queries;
    std::vector<D3D11ResourceRef>       m_resources;

    // NV-DXVK: RTX draw count captured by D3D11DeferredContext::
    // FinishCommandList from the recording context's D3D11Rtx.
    uint32_t          m_rtxDrawCount = 0;

    std::atomic<bool> m_submitted = { false };
    std::atomic<bool> m_warned    = { false };

    void TrackResourceSequenceNumber(
      const D3D11ResourceRef&   Resource,
            uint64_t            Seq);

    void MarkSubmitted();
    
  };
  
}
