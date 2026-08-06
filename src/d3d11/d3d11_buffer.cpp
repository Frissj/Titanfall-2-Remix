#include "d3d11_buffer.h"
#include "d3d11_context.h"
#include "d3d11_device.h"
#include "d3d11_vanish_diag.h"

#include "../dxvk/dxvk_data.h"

#ifndef RTX_D3D11_CACHED_DYNAMIC_INDEX_BUFFERS
// NV-DXVK [perf/CachedDynamicIB] 2026-08-06: allocate DYNAMIC INDEX buffers in
// HOST_CACHED (system) memory instead of write-combined BAR memory, because
// Remix reads their contents back on the CPU every frame. On by default.
//
// SET THIS TO false to restore stock DXVK behaviour (DEVICE_LOCAL|HOST_VISIBLE,
// write-combined). That is the A/B for this change — see the long rationale at
// the DYNAMIC case in GetMemoryFlags(), and measure with [Perf.CullVtx]
// reduce_MBps (CPU read side) against [Perf.GpuPass] (GPU side), since this
// trades GPU-side locality for CPU-side read speed.
#define RTX_D3D11_CACHED_DYNAMIC_INDEX_BUFFERS true
#endif

namespace dxvk {

  D3D11Buffer::D3D11Buffer(
          D3D11Device*                pDevice,
    const D3D11_BUFFER_DESC*          pDesc)
  : D3D11DeviceChild<ID3D11Buffer>(pDevice),
    m_desc        (*pDesc),
    m_resource    (this) {
    DxvkBufferCreateInfo  info;
    info.size   = pDesc->ByteWidth;
    info.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT
                | VK_ACCESS_TRANSFER_WRITE_BIT;
    
    if (pDesc->BindFlags & D3D11_BIND_VERTEX_BUFFER) {
      info.usage  |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      info.stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      info.access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
    
    if (pDesc->BindFlags & D3D11_BIND_INDEX_BUFFER) {
      info.usage  |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      info.stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      info.access |= VK_ACCESS_INDEX_READ_BIT;
    }
    
    if (pDesc->BindFlags & D3D11_BIND_CONSTANT_BUFFER) {
      info.usage  |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      info.stages |= m_parent->GetEnabledShaderStages();
      info.access |= VK_ACCESS_UNIFORM_READ_BIT;

      if (m_parent->GetOptions()->constantBufferRangeCheck)
        info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    
    if (pDesc->BindFlags & D3D11_BIND_SHADER_RESOURCE) {
      info.usage  |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT
                  |  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages |= m_parent->GetEnabledShaderStages();
      info.access |= VK_ACCESS_SHADER_READ_BIT;
    }
    
    if (pDesc->BindFlags & D3D11_BIND_STREAM_OUTPUT) {
      info.usage  |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
      info.stages |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
      info.access |= VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
    }
    
    if (pDesc->BindFlags & D3D11_BIND_UNORDERED_ACCESS) {
      info.usage  |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT
                  |  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages |= m_parent->GetEnabledShaderStages();
      info.access |= VK_ACCESS_SHADER_READ_BIT
                  |  VK_ACCESS_SHADER_WRITE_BIT;
    }
    
    if (pDesc->MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) {
      info.usage  |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      info.stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      info.access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }

    // Create the buffer and set the entire buffer slice as mapped,
    // so that we only have to update it when invalidating th buffer
    m_buffer = m_parent->GetDXVKDevice()->createBuffer(info, GetMemoryFlags(), DxvkMemoryStats::Category::AppBuffer, "d3d11 buffer");
    m_mapped = m_buffer->getSliceHandle();

    m_mapMode = DetermineMapMode();

    // For Stream Output buffers we need a counter
    if (pDesc->BindFlags & D3D11_BIND_STREAM_OUTPUT)
      m_soCounter = CreateSoCounterBuffer();
  }
  
  
  D3D11Buffer::~D3D11Buffer() {
    // Destructor counter intentionally NOT bumped here — destructors can run
    // late in process shutdown, after the vanish_diag g_counts static array
    // has been destroyed, leading to UB. CreateBuf alone answers the
    // realloc-loop question.
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11Buffer::QueryInterface(REFIID riid, void** ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;
    
    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D11DeviceChild)
     || riid == __uuidof(ID3D11Resource)
     || riid == __uuidof(ID3D11Buffer)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(IDXGIObject)
     || riid == __uuidof(IDXGIDeviceSubObject)
     || riid == __uuidof(IDXGIResource)
     || riid == __uuidof(IDXGIResource1)) {
       *ppvObject = ref(&m_resource);
       return S_OK;
    }
    
    Logger::warn("D3D11Buffer::QueryInterface: Unknown interface query");
    Logger::warn(str::format(riid));
    return E_NOINTERFACE;
  }
  
  
  UINT STDMETHODCALLTYPE D3D11Buffer::GetEvictionPriority() {
    return m_evictionPriority.load(std::memory_order_relaxed);
  }


  void STDMETHODCALLTYPE D3D11Buffer::SetEvictionPriority(UINT EvictionPriority) {
    m_evictionPriority.store(EvictionPriority, std::memory_order_relaxed);
  }
  
  
  void STDMETHODCALLTYPE D3D11Buffer::GetType(D3D11_RESOURCE_DIMENSION* pResourceDimension) {
    *pResourceDimension = D3D11_RESOURCE_DIMENSION_BUFFER;
  }
  
  
  void STDMETHODCALLTYPE D3D11Buffer::GetDesc(D3D11_BUFFER_DESC* pDesc) {
    *pDesc = m_desc;
  }
  
  
  bool D3D11Buffer::CheckViewCompatibility(
          UINT                BindFlags,
          DXGI_FORMAT         Format) const {
    // Check whether the given bind flags are supported
    if ((m_desc.BindFlags & BindFlags) != BindFlags)
      return false;

    // Structured buffer views use no format
    if (Format == DXGI_FORMAT_UNKNOWN)
      return (m_desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) != 0;

    // Check whether the given combination of buffer view
    // type and view format is supported by the device
    DXGI_VK_FORMAT_INFO viewFormat = m_parent->LookupFormat(Format, DXGI_VK_FORMAT_MODE_ANY);
    VkFormatFeatureFlags features = GetBufferFormatFeatures(BindFlags);

    return CheckFormatFeatureSupport(viewFormat.Format, features);
  }


  HRESULT D3D11Buffer::NormalizeBufferProperties(D3D11_BUFFER_DESC* pDesc) {
    // Zero-sized buffers are illegal
    if (!pDesc->ByteWidth)
      return E_INVALIDARG;

    // We don't support tiled resources
    if (pDesc->MiscFlags & (D3D11_RESOURCE_MISC_TILE_POOL | D3D11_RESOURCE_MISC_TILED))
      return E_INVALIDARG;
    
    // Constant buffer size must be a multiple of 16
    if ((pDesc->BindFlags & D3D11_BIND_CONSTANT_BUFFER)
     && (pDesc->ByteWidth & 0xF))
      return E_INVALIDARG;

    // Basic validation for structured buffers
    if ((pDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
     && ((pDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
      || (pDesc->StructureByteStride == 0)
      || (pDesc->StructureByteStride & 0x3)))
      return E_INVALIDARG;
    
    // Basic validation for raw buffers
    if ((pDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
     && (!(pDesc->BindFlags & (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS))))
      return E_INVALIDARG;

    // Mip generation obviously doesn't work for buffers
    if (pDesc->MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS)
      return E_INVALIDARG;
    
    if (!(pDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED))
      pDesc->StructureByteStride = 0;
    
    return S_OK;
  }


  BOOL D3D11Buffer::CheckFormatFeatureSupport(
          VkFormat              Format,
          VkFormatFeatureFlags  Features) const {
    VkFormatProperties properties = m_parent->GetDXVKDevice()->adapter()->formatProperties(Format);
    return (properties.bufferFeatures & Features) == Features;
  }


  VkMemoryPropertyFlags D3D11Buffer::GetMemoryFlags() const {
    VkMemoryPropertyFlags memoryFlags = 0;
    
    switch (m_desc.Usage) {
      case D3D11_USAGE_IMMUTABLE:
        memoryFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;

      case D3D11_USAGE_DEFAULT:
        memoryFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        if ((m_desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) || m_desc.CPUAccessFlags) {
          memoryFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                      |  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }

        if (m_desc.CPUAccessFlags & D3D11_CPU_ACCESS_READ) {
          memoryFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
          memoryFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
      
      case D3D11_USAGE_DYNAMIC:
        memoryFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    |  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        if (m_desc.BindFlags)
          memoryFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        // NV-DXVK [perf/CachedDynamicIB] 2026-08-06: DYNAMIC INDEX buffers are
        // read back by the CPU every frame, so give them CACHED memory.
        //
        // Remix is not a plain translation layer here: SubmitDraw scans the
        // index range of every draw to compute a tight max-index for the
        // object-space AABB producer (see the needsMeshBoundingBox tombstone in
        // d3d11_rtx.cpp). That is ~600k indices / ~1.2 MB per frame READ BACK
        // from a buffer D3D11 assumes is write-only.
        //
        // With DEVICE_LOCAL|HOST_VISIBLE those reads come from BAR memory, which
        // is write-combined: uncached, ~300 MB/s. [Perf.CullVtx] measured
        // exactly that — reduce_MBps 280-365 with acquire_ms ~1.5, i.e. the cost
        // is the reads themselves and the scan is already at the hardware limit.
        // Tuning the loop cannot beat the memory type; a 2->8 cache-line unroll
        // was tried and came back 21% WORSE (tombstoned in maxIndexFromWC).
        //
        // This is the same substitution DXVK already makes one case up: for
        // D3D11_CPU_ACCESS_READ on DEFAULT buffers it adds HOST_CACHED and
        // clears DEVICE_LOCAL. The situation is the same — CPU reads back — the
        // difference is only that D3D11 never told us, because the readback is
        // Remix's, not the application's.
        //
        // THE TRADE, explicitly: clearing DEVICE_LOCAL moves the buffer to
        // system RAM, so the GPU reaches it over PCIe. In Remix that is weaker
        // than it looks — index data is copied into RT geometry buffers for BLAS
        // builds rather than fetched per-draw by a rasterizer — but it is not
        // free, and the game's Map(WRITE_DISCARD) writes also become cached
        // rather than write-combining. Both are GPU/write-side costs traded for
        // a CPU read-side win, so it must be MEASURED, not assumed:
        //   [Perf.CullVtx] reduce_MBps  should leave ~300 for multiple GB/s
        //   [Perf.SubmitDraw] bt_cullVtx should fall from ~4.3 ms/frame
        //   [Perf.GpuPass]              watch for any rise in the RT passes
        // Set RTX_D3D11_CACHED_DYNAMIC_INDEX_BUFFERS to false to A/B it.
        //
        // Scoped to INDEX buffers alone on purpose: vertex and constant buffers
        // are not scanned per draw, so they keep write-combining, which is the
        // right choice for a write-only streaming buffer.
        if (RTX_D3D11_CACHED_DYNAMIC_INDEX_BUFFERS
         && (m_desc.BindFlags & D3D11_BIND_INDEX_BUFFER)) {
          memoryFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
          memoryFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
      
      case D3D11_USAGE_STAGING:
        memoryFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    |  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    |  VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    }
    
    if (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT && m_parent->GetOptions()->apitraceMode) {
      memoryFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                  |  VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }

    return memoryFlags;
  }


  Rc<DxvkBuffer> D3D11Buffer::CreateSoCounterBuffer() {
    Rc<DxvkDevice> device = m_parent->GetDXVKDevice();

    DxvkBufferCreateInfo info;
    info.size   = sizeof(D3D11SOCounter);
    info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
                | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
                | VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT
                | VK_ACCESS_TRANSFER_WRITE_BIT
                | VK_ACCESS_INDIRECT_COMMAND_READ_BIT
                | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT
                | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
    return device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppBuffer, "d3d11 counter buffer");
  }


  D3D11_COMMON_BUFFER_MAP_MODE D3D11Buffer::DetermineMapMode() {
    return (m_buffer->memFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      ? D3D11_COMMON_BUFFER_MAP_MODE_DIRECT
      : D3D11_COMMON_BUFFER_MAP_MODE_NONE;
  }
  

  D3D11Buffer* GetCommonBuffer(ID3D11Resource* pResource) {
    D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&dimension);

    return dimension == D3D11_RESOURCE_DIMENSION_BUFFER
      ? static_cast<D3D11Buffer*>(pResource)
      : nullptr;
  }

}
