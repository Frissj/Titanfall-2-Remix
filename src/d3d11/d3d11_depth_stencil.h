#pragma once

#include "../dxvk/dxvk_device.h"

#include "d3d11_device_child.h"
#include "d3d11_util.h"

namespace dxvk {
  
  class D3D11Device;
  
  class D3D11DepthStencilState : public D3D11StateObject<ID3D11DepthStencilState> {
    
  public:
    
    using DescType = D3D11_DEPTH_STENCIL_DESC;
    
    D3D11DepthStencilState(
            D3D11Device*              device,
      const D3D11_DEPTH_STENCIL_DESC& desc);
    ~D3D11DepthStencilState();

    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject) final;
    
    void STDMETHODCALLTYPE GetDesc(
            D3D11_DEPTH_STENCIL_DESC* pDesc) final;

    // NV-DXVK [perf]: direct const access to the immutable desc. State
    // objects never change after creation, so hot paths (SubmitDraw runs
    // per draw) read this instead of paying a struct copy via GetDesc.
    const D3D11_DEPTH_STENCIL_DESC* Desc() const {
      return &m_desc;
    }

    void BindToContext(
      const Rc<DxvkContext>&  ctx);
    static HRESULT NormalizeDesc(
            D3D11_DEPTH_STENCIL_DESC* pDesc);
    
  private:
    
    D3D11_DEPTH_STENCIL_DESC  m_desc;
    DxvkDepthStencilState     m_state;
    VkStencilOpState DecodeStencilOpState(
      const D3D11_DEPTH_STENCILOP_DESC& StencilDesc,
      const D3D11_DEPTH_STENCIL_DESC&   Desc) const;
    
    VkStencilOp DecodeStencilOp(
            D3D11_STENCIL_OP            Op) const;
    
    static bool ValidateDepthFunc(
            D3D11_COMPARISON_FUNC  Comparison);

    static bool ValidateStencilFunc(
            D3D11_COMPARISON_FUNC  Comparison);

    static bool ValidateStencilOp(
            D3D11_STENCIL_OP       StencilOp);

    static bool ValidateDepthWriteMask(
            D3D11_DEPTH_WRITE_MASK Mask);
  };
  
}
