#include "d3d11_buffer.h"
#include "d3d11_context.h"
#include "d3d11_device.h"
#include "d3d11_vanish_diag.h"

#include "../dxvk/dxvk_data.h"
#include "../dxvk/rtx_render/rtx_resident_scene.h"

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

#ifndef RTX_D3D11_CACHED_DYNAMIC_SRV_BUFFERS
// NV-DXVK [perf/CachedDynamicSRV] 2026-08-07: the same substitution as the index
// buffers above, extended to DYNAMIC SHADER_RESOURCE buffers, because the t31
// per-instance transform buffer is one and Remix reads it back on the CPU on
// every instanced draw.
//
// This is measured, not inferred. [Perf.LoopCut] t31_copy is 13-15 us/call at
// ~517 calls/frame = 6-7 ms/frame, ~15% of a 51 ms frame and the single largest
// item on the frame thread. [Perf.T31TailEvt] pinned the rate with raw pairs --
// 52.0 KB -> 38.5 us, 34.3 KB -> 25.3 us, 49.8 KB -> 37.1 us, agreeing to under
// 1% at 1.38 GB/s. That is write-combined BAR read speed, and no amount of
// copying less or copying smarter moves it:
//   [Perf.T31Cache]  hitPct 0.0     -- Map(WRITE_DISCARD) renames the slice per
//                                      draw, so no key can ever hold
//   [Perf.T31Span]   usedPct 92-95, runsPerDraw 1.0
//                                   -- the span copied is already one contiguous
//                                      run and almost entirely wanted bytes
//   [Perf.T31Warm]   c2 ~= c1       -- re-reading the same bytes costs the same,
//                                      which is what uncached memory does
// The bytes are all wanted, the layout is already optimal, and the memory type
// is the whole cost. So change the memory type.
//
// THE TRADE, same shape as the index-buffer case and equally not free: clearing
// DEVICE_LOCAL moves these buffers to system RAM, so the GPU reaches them over
// PCIe, and the game's Map(WRITE_DISCARD) writes become cached rather than
// write-combining. t31 is read by the game's own vertex shaders during the
// raster pass, not only by our extraction, so the GPU-side exposure here is
// REAL and larger than it was for index data. It must be measured:
//   [Perf.LoopCut] t31_copy   should fall from ~14 us/call toward ~2
//   [Perf.T31Size] GBs        should rise from ~1.38 toward cached bandwidth
//   [Perf.WcCopy]             should lose its largest entry
//   [Perf.Busy]   cpuMs       should fall by ~5-6 ms/frame
//   [Perf.Gpu] / [Perf.GpuPass]  WATCH THIS -- any rise in GPU pass time is the
//                             PCIe cost landing, and if it exceeds the CPU win
//                             this change is a loss even though t31_copy improved
// Set to false to A/B it against stock behaviour.
//
// ============ KNOWN INTERACTION: VRAM SAWTOOTH WITH THE TRIM ============
// 2026-08-07. Observed live with this ON: VRAM oscillates high/low EVERY FRAME
// through the automatic geometry-cache trim. It does not crash, and it is not a
// leak -- it is a free/realloc cycle running once a frame.
//
// The earlier reading of this was wrong and is recorded so it is not repeated:
// two runs showed vkAllocateMemory vr=-2 with heapAllocated climbing
// 7217 -> 7377 -> 7427 MiB and 75 failures in a second, which was taken for
// progressive exhaustion from this change. It is the top of the sawtooth, not a
// climb, and both of those runs also had [DumpDraw] texture dumping and an armed
// [OnScreenAlbedoDump] inflating the baseline (see PERF_INSTRUMENTATION_MAP s8
// on rtx-asset-exporter). A single OOM warning near the peak is the trim and
// this change fighting, not this change leaking.
//
// WHY THE TWO INTERACT. Dropping DEVICE_LOCAL here is a change to the memory
// TYPE these buffers are requested from, and the flags are a request rather than
// a restriction -- so which pool a dynamic SRV buffer lands in moves, and with it
// the slice-renaming that Map(WRITE_DISCARD) does on every draw. The trim then
// reclaims against a pool whose occupancy it was tuned for, and the two chase
// each other once a frame.
// =======================================================================
//
// A/B RESULT, 2026-08-07 -- ON, and it is a clean win.
//
//                       OFF              ON
//   wallMs range        51.5-67.8        44.9-55.5
//   t31_copy            13.3-14.5 us     0.41-0.53 us
//   t31_copy /window    630-696 ms       23-25 ms
//   fenceWaitMs         ~21.5            ~22        (unchanged)
//   OOM failures        0                0
//
// ~6.5-7 ms/frame off the whole distribution, and the arithmetic closes: 13.65 us
// x ~51k calls = 696 ms/window over ~98 frames = 7.1 ms/frame, against 0.25 with
// it on. The GPU never notices, which is what acquits the PCIe concern above.
//
// A "25-second decay" was read into the ON run and it was NOT REAL -- an apparent
// 44.9 -> 55.5 ramp, and a matching inner_ms 697 -> 791, were scene-driven
// oscillation. The OFF run swings the same way (66.2 67.8 52.9 51.7 51.5 60.2,
// inner_ms 801 797 711 695 694 768) and inner_ms occupies the same 694-800 band
// in both. HANDOFF s7.5 already says it: this game's per-draw cost swings ~2x in
// a fixed scene, so cross-window ms deltas are noise. Compare DISTRIBUTIONS
// across a run, never first-window against last.
//
// Depends on the host-visible Small-pool fix in dxvk_memory.cpp
// (HostVisibleSmallPool). Without it this change fragments the allocator until a
// forced unlimited trim recovers nothing: 8673 MB allocated, 2250 MB unreclaimable
// slack, 75 allocation failures. With it: 7425 MB stable, zero failures. Do not
// enable one without the other.
#define RTX_D3D11_CACHED_DYNAMIC_SRV_BUFFERS true
#endif

#ifndef RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS
// NV-DXVK [perf/CachedDynamicVB] 2026-08-15: the same substitution a third time,
// for DYNAMIC VERTEX buffers, because the GEOMETRY HASH reads them back on the CPU
// every frame. On by default. Set to false to A/B against stock behaviour.
//
// THIS REVERSES A CLAIM MADE AT THE DYNAMIC CASE BELOW. That comment said "VERTEX
// and CONSTANT buffers still keep write-combining: they are genuinely write-only
// streaming data from our side", and listed the one vertex-buffer reader it knew
// about (the charIdx instance buffer, safe behind m_instBufCache at a 100% hit
// rate). It missed runBatchHashJob, which reads position AND texcoord bytes out of
// these buffers to compute the mesh identity hash, once per draw, forever.
//
// MEASURED, not inferred ([BatchJoinCensus] / [BatchJoinPeak] in d3d11_rtx.cpp):
//   sumWork 29 ms/frame in the frame-end parallel-for, of which hash = 17.6 ms
//   mapPos  72 of 1070 draws are CPU-readable -- 6.7% of draws, 74% of the stage
//   peak    posB=11648 tcB=11648 hashed in 2680 us  =>  ~8.7 MB/s
//   posMem  0x7 = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, HOST_CACHED clear
// XXH3 runs at 10-30 GB/s on cached RAM, so ~8.7 MB/s is three orders of magnitude
// off and is not computation -- it is the BAR read. The byte counts also prove the
// range clamps are correct (an 11 KB window of a 917504-byte buffer), so there is
// nothing left to tune on the read side. Same conclusion the t31 work reached:
// the bytes are all wanted, the layout is already right, the memory type is the cost.
//
// Corroborated by a second, independent reader of the same buffers: runBatchBboxJob
// memcpys before scanning (the standard WC-read workaround) and still shows 4480
// bytes taking 1464 us = ~3 MB/s. Two different consumers, same ceiling.
//
// THE TRADE, and it is BIGGER HERE THAN IN EITHER PRIOR CASE. Index data is copied
// into RT geometry buffers rather than fetched per-draw, and even the t31 SRV case
// only exposed the game's own vertex shaders. Vertex buffers are fetched per-draw by
// the rasterizer AND copied for BLAS builds, so moving them to system RAM puts real
// per-draw GPU traffic on PCIe. Two things bound that exposure: only 72 of ~1070
// draws use DYNAMIC vertex buffers at all (the other ~998 are device-local immutable
// and untouched by this), and the t31 A/B -- which had the same shape of risk -- came
// back with the GPU never noticing. Precedent is favourable; it is not proof.
//
// VERIFY, and do not skip the GPU half:
//   [BatchJoinCensus] hash=17600us   should collapse toward ~100 us
//   [BatchJoinCensus] bbox=4700us    should fall too (same buffers, same ceiling)
//   [BatchJoinSplit]  tail=6500us    should fall; sumWork should drop by ~20 ms
//   [Perf.GpuPass]                   WATCH THIS. Any rise is the PCIe cost landing,
//                                    and if it exceeds the CPU win this is a LOSS
//                                    even though the hash numbers improved.
// Compare DISTRIBUTIONS across a run, never first window against last -- this game's
// per-draw cost swings ~2x in a fixed scene (see the A/B note above).
//
// Same dependency as the SRV case: the HostVisibleSmallPool fix in dxvk_memory.cpp,
// and the same known VRAM-sawtooth interaction with the geometry-cache trim.
#define RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS true
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

    // NV-DXVK [ResidentScene]: THE DEATH SIGNAL. The engine freeing an object's
    // vertex or index buffer is that object ceasing to exist, and this is the
    // one point at which DXVK observes it directly. A resident instance is
    // exempt from lifetime expiry, so without this a destroyed object would stay
    // in the ray-traced scene until its record happened to be evicted.
    //
    // SAFE IN A DESTRUCTOR, which the note above is a standing warning about:
    // the callee holds its state in a deliberately leaked allocation for exactly
    // that reason, and returns on a single relaxed atomic load when residency
    // has never built a record — which is every run with the feature off, and
    // matters because this runs for every buffer the engine frees.
    dxvk::noteResidentSourceBufferDestroyed(reinterpret_cast<uint64_t>(this));
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
        // Scope note, revised 2026-08-07: this used to say "INDEX buffers alone
        // on purpose", on the reasoning that nothing else was scanned per draw.
        // That was true of what had been measured at the time. It is not true of
        // SHADER_RESOURCE buffers -- the t31 per-instance transform buffer is
        // read back on every instanced draw for the fanout, at 6-7 ms/frame, and
        // that read was subsequently measured at 1.38 GB/s, i.e. the same
        // write-combined ceiling this carve-out exists to escape. See
        // RTX_D3D11_CACHED_DYNAMIC_SRV_BUFFERS at the top of this file.
        //
        // Scope note, revised again 2026-08-15: this used to end "VERTEX and
        // CONSTANT buffers still keep write-combining: they are genuinely
        // write-only streaming data from our side", qualified only by the charIdx
        // instance buffer being safe behind m_instBufCache. The VERTEX half of that
        // is now REFUTED -- runBatchHashJob reads position and texcoord bytes out of
        // DYNAMIC vertex buffers to build the mesh identity hash, 72 draws/frame at
        // ~8.7 MB/s for 17.6 ms/frame. See RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS
        // at the top of this file for the measurement and the (larger) trade.
        //
        // CONSTANT buffers keep write-combining and that half still stands: the
        // cbuffer readers copy on the frame thread at capture time rather than
        // reading back per draw.
        if (RTX_D3D11_CACHED_DYNAMIC_INDEX_BUFFERS
         && (m_desc.BindFlags & D3D11_BIND_INDEX_BUFFER)) {
          memoryFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
          memoryFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }

        if (RTX_D3D11_CACHED_DYNAMIC_SRV_BUFFERS
         && (m_desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
          memoryFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
          memoryFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }

        if (RTX_D3D11_CACHED_DYNAMIC_VERTEX_BUFFERS
         && (m_desc.BindFlags & D3D11_BIND_VERTEX_BUFFER)) {
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
