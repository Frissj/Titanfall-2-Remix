#pragma once

#include <atomic>

#include "../dxvk/dxvk_cs.h"
#include "../dxvk/dxvk_device.h"

#include "d3d11_device_child.h"
#include "d3d11_interfaces.h"
#include "d3d11_resource.h"

namespace dxvk {
  
  class D3D11Device;
  class D3D11DeviceContext;


  /**
   * \brief Buffer map mode
   */
  enum D3D11_COMMON_BUFFER_MAP_MODE {
    D3D11_COMMON_BUFFER_MAP_MODE_NONE,
    D3D11_COMMON_BUFFER_MAP_MODE_DIRECT,
  };


  /**
   * \brief Stream output buffer offset
   *
   * A byte offset into the buffer that
   * stores the byte offset where new
   * data will be written to.
   */
  struct D3D11SOCounter {
    uint32_t byteOffset;
  };
  
  
  class D3D11Buffer : public D3D11DeviceChild<ID3D11Buffer> {
    static constexpr VkDeviceSize BufferSliceAlignment = 64;
  public:
    
    D3D11Buffer(
            D3D11Device*                pDevice,
      const D3D11_BUFFER_DESC*          pDesc);
    ~D3D11Buffer();
    
    HRESULT STDMETHODCALLTYPE QueryInterface(
            REFIID  riid,
            void**  ppvObject) final;
    
    void STDMETHODCALLTYPE GetType(
            D3D11_RESOURCE_DIMENSION *pResourceDimension) final;
    
    UINT STDMETHODCALLTYPE GetEvictionPriority() final;
    
    void STDMETHODCALLTYPE SetEvictionPriority(UINT EvictionPriority) final;
    
    void STDMETHODCALLTYPE GetDesc(
            D3D11_BUFFER_DESC *pDesc) final;
    
    bool CheckViewCompatibility(
            UINT                BindFlags,
            DXGI_FORMAT         Format) const;

    const D3D11_BUFFER_DESC* Desc() const {
      return &m_desc;
    }

    D3D11_COMMON_BUFFER_MAP_MODE GetMapMode() const {
      return m_mapMode;
    }

    Rc<DxvkBuffer> GetBuffer() const {
      return m_buffer;
    }
    
    DxvkBufferSlice GetBufferSlice() const {
      return DxvkBufferSlice(m_buffer, 0, m_desc.ByteWidth);
    }
    
    DxvkBufferSlice GetBufferSlice(VkDeviceSize offset) const {
      VkDeviceSize size = m_desc.ByteWidth;

      return likely(offset < size)
        ? DxvkBufferSlice(m_buffer, offset, size - offset)
        : DxvkBufferSlice();
    }
    
    DxvkBufferSlice GetBufferSlice(VkDeviceSize offset, VkDeviceSize length) const {
      VkDeviceSize size = m_desc.ByteWidth;

      return likely(offset < size)
        ? DxvkBufferSlice(m_buffer, offset, std::min(length, size - offset))
        : DxvkBufferSlice();
    }

    DxvkBufferSlice GetSOCounter() {
      return m_soCounter != nullptr
        ? DxvkBufferSlice(m_soCounter)
        : DxvkBufferSlice();
    }
    
    DxvkBufferSliceHandle AllocSlice() {
      return m_buffer->allocSlice();
    }
    
    DxvkBufferSliceHandle DiscardSlice() {
      m_mapped = m_buffer->allocSlice();
      // NV-DXVK [MaxIdxCache]: invalidate the per-buffer max-index scan
      // cache when the buffer is remapped (DISCARD). Content has changed,
      // so previously-cached scan results no longer correspond to the
      // bytes the next draw will read.
      InvalidateMaxIdxCache();
      // NV-DXVK [CamCache]: bump a monotonic content generation on every
      // Map(WRITE_DISCARD). ExtractTransforms() keys its per-frame camera
      // cache (worldToView/viewToProjection) on (buffer, generation): while
      // cb2 is unchanged across consecutive draws the camera is re-used
      // instead of re-reconstructed per draw. Monotonic (never reused), so
      // it is immune to the slice-address recycling that mapPtr comparison
      // would suffer when allocSlice() hands back a freed allocation.
      m_contentGen.fetch_add(1, std::memory_order_release);
      return m_mapped;
    }

    // NV-DXVK [CamCache]: see DiscardSlice(). Strictly increases on each
    // content rewrite; stable (0) for IMMUTABLE/STATIC buffers that never
    // discard, which is also a valid cache key.
    uint64_t GetContentGeneration() const {
      return m_contentGen.load(std::memory_order_acquire);
    }

    DxvkBufferSliceHandle GetMappedSlice() const {
      return m_mapped;
    }
    bool HasSequenceNumber() const {
      return m_mapMode != D3D11_COMMON_BUFFER_MAP_MODE_NONE
          && !(m_desc.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS)
          && !(m_desc.BindFlags);
    }

    void TrackSequenceNumber(uint64_t Seq) {
      m_seq = Seq;
    }

    uint64_t GetSequenceNumber() {
      return HasSequenceNumber() ? m_seq
        : DxvkCsThread::SynchronizeAll;
    }

    /**
     * \brief Normalizes buffer description
     * 
     * \param [in] pDesc Buffer description
     * \returns \c S_OK if the parameters are valid
     */
    static HRESULT NormalizeBufferProperties(
            D3D11_BUFFER_DESC*      pDesc);

  private:

    D3D11_BUFFER_DESC             m_desc;
    D3D11_COMMON_BUFFER_MAP_MODE  m_mapMode;

    Rc<DxvkBuffer>                m_buffer;
    Rc<DxvkBuffer>                m_soCounter;
    DxvkBufferSliceHandle         m_mapped;
    uint64_t                      m_seq = 0ull;
    // NV-DXVK [CamCache]: monotonic content generation, bumped in DiscardSlice().
    // Atomic because the immediate context writes it (on Map(DISCARD)) while
    // deferred-context threads recording draws read it for cache keying.
    std::atomic<uint64_t>         m_contentGen { 0ull };

    // NV-DXVK TF2: persistent eviction priority. Source/Titanfall sets HIGH
    // on streaming targets, then later checks Get to verify residency. The
    // upstream stub returned NORMAL unconditionally, making the engine think
    // the resource was evicted and reallocate it every frame -> texture
    // streaming targets never persisted long enough for mip 0 to land.
    std::atomic<UINT>             m_evictionPriority { DXGI_RESOURCE_PRIORITY_NORMAL };

    // NV-DXVK: CPU copy of IMMUTABLE buffer data for RTX bone instancing readback.
    std::vector<uint8_t>          m_immutableData;
  public:
    const std::vector<uint8_t>& GetImmutableData() const { return m_immutableData; }
    void SetImmutableData(const void* data, size_t size) {
      m_immutableData.resize(size);
      std::memcpy(m_immutableData.data(), data, size);
    }

    // NV-DXVK [MaxIdxCache]: per-buffer cache for DrawIndexed max-index
    // scans. Stored as a linear vector of (offset, start, count, maxIdx)
    // tuples — N is typically <32 per buffer for TF2 BSPs, so linear scan
    // is faster than hash + bucket walk and avoids allocator traffic.
    //
    // STATIC/IMMUTABLE: populated once per submesh, hit forever.
    // DYNAMIC: cleared in DiscardSlice() when content changes.
    //
    // We compare the three fields directly rather than packing into a
    // 64-bit key — earlier bitpack-XOR approach silently collided when
    // `start` exceeded 24 bits (common on large TF2 BSP IBs), so two
    // unrelated draws shared a cached maxIdx → catastrophic geometry
    // corruption (visible as the whole scene collapsing onto 2-3 unique
    // surface hashes in the geometry-hash debug view).
    struct MaxIdxEntry {
      uint64_t offset;
      uint32_t start;
      uint32_t count;
      uint32_t maxIdx;
    };
    static constexpr size_t kMaxIdxCacheEntries = 128;

    // NV-DXVK [perf]: walk MOST-RECENT-FIRST. Inserts appended to the back, so
    // a forward scan found the newest entry LAST — on a buffer holding the full
    // 128 entries every lookup paid ~128 x 3 field compares before hitting.
    // Draws revisit recently-scanned ranges far more often than old ones (the
    // same submesh across consecutive draws / frames), so reverse order turns
    // the common case into a 1-2 iteration walk.
    //
    // Duplicate keys can exist once the ring wraps, but they always carry the
    // SAME maxIdx — it is a pure function of the buffer's content over a fixed
    // range, and any content change clears the whole cache via DiscardSlice —
    // so scan order can never change the value returned.
    bool LookupMaxIdx(uint64_t offset, uint32_t start, uint32_t count, uint32_t& outMaxIdx) const {
      const size_t n = m_maxIdxCache.size();
      if (n == 0)
        return false;
      // Newest entry sits immediately behind the write cursor.
      size_t i = (m_maxIdxNext == 0 ? n : m_maxIdxNext) - 1;
      for (size_t k = 0; k < n; ++k) {
        const auto& e = m_maxIdxCache[i];
        if (e.offset == offset && e.start == start && e.count == count) {
          outMaxIdx = e.maxIdx;
          return true;
        }
        i = (i == 0 ? n : i) - 1;
      }
      return false;
    }
    void InsertMaxIdx(uint64_t offset, uint32_t start, uint32_t count, uint32_t maxIdx) {
      // Cap at 128 entries per buffer. Pathological text-glyph buffers with
      // thousands of unique runs spill by overwriting the oldest.
      // NV-DXVK [perf]: was erase(begin()) + push_back once full — an O(n)
      // memmove of 128 x 24 bytes on EVERY insert past the cap, on the cold
      // path that is already the expensive one. Ring-overwrite is O(1).
      if (m_maxIdxCache.size() < kMaxIdxCacheEntries) {
        m_maxIdxCache.push_back({ offset, start, count, maxIdx });
        m_maxIdxNext = m_maxIdxCache.size() % kMaxIdxCacheEntries;
      } else {
        m_maxIdxCache[m_maxIdxNext] = { offset, start, count, maxIdx };
        m_maxIdxNext = (m_maxIdxNext + 1u) % kMaxIdxCacheEntries;
      }
    }
    void InvalidateMaxIdxCache() { m_maxIdxCache.clear(); m_maxIdxNext = 0; }

  private:
    std::vector<MaxIdxEntry>      m_maxIdxCache;
    size_t                        m_maxIdxNext = 0;  // ring write cursor

    D3D11DXGIResource             m_resource;
    BOOL CheckFormatFeatureSupport(
            VkFormat              Format,
            VkFormatFeatureFlags  Features) const;
    
    VkMemoryPropertyFlags GetMemoryFlags() const;

    Rc<DxvkBuffer> CreateSoCounterBuffer();

    D3D11_COMMON_BUFFER_MAP_MODE DetermineMapMode();

  };


  /**
   * \brief Retrieves buffer from resource pointer
   * 
   * \param [in] pResource The resource to query
   * \returns Pointer to buffer, or \c nullptr
   */
  D3D11Buffer* GetCommonBuffer(
          ID3D11Resource*       pResource);
  
}
