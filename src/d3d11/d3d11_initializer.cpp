#include <cstring>
#include <algorithm>

#include "d3d11_device.h"
#include "d3d11_initializer.h"

namespace dxvk {

  D3D11Initializer::D3D11Initializer(
          D3D11Device*                pParent)
  : m_parent(pParent),
    m_device(pParent->GetDXVKDevice()),
    m_context(m_device->createContext()) {
    m_context->beginRecording(
      m_device->createCommandList());
  }

  
  D3D11Initializer::~D3D11Initializer() {

  }


  void D3D11Initializer::Flush() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (m_transferCommands != 0)
      FlushInternal();
  }

  void D3D11Initializer::InitBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    VkMemoryPropertyFlags memFlags = pBuffer->GetBuffer()->memFlags();

    (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      ? InitHostVisibleBuffer(pBuffer, pInitialData)
      : InitDeviceLocalBuffer(pBuffer, pInitialData);
  }
  

  void D3D11Initializer::InitTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    StampContentHash(pTexture, pInitialData);

    (pTexture->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_DIRECT)
      ? InitHostVisibleTexture(pTexture, pInitialData)
      : InitDeviceLocalTexture(pTexture, pInitialData);
  }


  // NV-DXVK [texture identity]: STAMP A CONTENT HASH AT CREATION.
  //
  // THE DEFECT. d3d11_rtx.cpp's FillMaterialData stamps a DxvkImage hash only
  // when getHash()==0, and seeds it with img->cookie() -- a PER-OBJECT cookie.
  // The game recreates D3D11 texture objects for textures already on screen, so
  // every recreation mints a fresh image hash, which propagates to a fresh
  // material hash and a fresh surface material index. Measured over 1,819
  // frames of ordinary play: imgNew 1,899 (1.04/frame), matNew 1.5/frame,
  // texNew 2.6/frame, texFree 0, texTotal 363 -> 2,039 and still climbing --
  // against rtx_scene_manager.h's own acceptance line, "in a steady scene with a
  // still camera, matNew / texNew / imgNew should all be 0. Any of them running
  // hot continuously IS the churn". Remix's managed-texture streaming is idle
  // throughout (mtQueue/mtDemote/mtVid/mtSwap all 0), so the recreation is
  // game-side, not self-inflicted.
  //
  // WHY NOT THE TWO OBVIOUS KEYS, both already closed off at that site. The
  // object ADDRESS was tried and reverted ("flicker fix A"): the allocator
  // reuses addresses, so a new texture inherited a dead one's material whenever
  // the two shared extent, format and mip count. Extent/format/mips ALONE
  // collide across same-sized textures, which is exactly why the cookie was
  // added. Only the content is both stable across recreation and distinct
  // between textures.
  //
  // WHAT IS HASHED, and it is bounded on purpose. Up to kHashRows rows sampled
  // evenly across mip 0, at most kHashRowBytes each, folded with the descriptor.
  // Striding across the image rather than taking a prefix matters: art assets
  // routinely share a uniform first row (sky, UI, anything with a border), so a
  // prefix would collide where a stride does not. The cap holds the cost near
  // 64 KB per texture creation, about one creation per frame here.
  //
  // Textures with no initial data -- render targets, dynamic and staging
  // textures -- are left alone. They have no stable content to hash, they are
  // not the churning population, and FillMaterialData's cookie stamp still
  // covers them because this only fires when there is data to read.
  void D3D11Initializer::StampContentHash(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    if (pInitialData == nullptr || pInitialData[0].pSysMem == nullptr) {
      return;
    }

    // GetImage() returns by value; taken as a value rather than a const
    // reference so nothing depends on temporary lifetime extension here.
    const Rc<DxvkImage> image = pTexture->GetImage();
    if (image == nullptr) {
      return;
    }

    constexpr uint32_t kHashRows = 32u;
    constexpr size_t   kHashRowBytes = 2048u;

    const D3D11_COMMON_TEXTURE_DESC* desc = pTexture->Desc();
    const D3D11_SUBRESOURCE_DATA& d0 = pInitialData[0];

    const size_t pitch = static_cast<size_t>(d0.SysMemPitch);
    const size_t slice = static_cast<size_t>(d0.SysMemSlicePitch);
    // rows is derived rather than taken from the height, because the pitch is in
    // BLOCK rows for compressed formats and the height is in texels. Dividing
    // the slice pitch by the row pitch gives block rows without needing the
    // format's block size. A game that leaves SysMemSlicePitch at 0 for a 2D
    // texture falls back to a single row, which still hashes real content.
    const size_t rows = (pitch != 0u && slice >= pitch) ? (slice / pitch) : 1u;

    // The descriptor first, so two textures with identical sampled bytes but
    // different dimensions or formats cannot collide.
    struct DescKey {
      uint32_t width, height, mipLevels, arraySize;
      uint32_t format, sampleCount;
      uint32_t pitch, rows;
    };
    const DescKey dk = {
      desc->Width, desc->Height, desc->MipLevels, desc->ArraySize,
      static_cast<uint32_t>(desc->Format), desc->SampleDesc.Count,
      static_cast<uint32_t>(pitch), static_cast<uint32_t>(rows),
    };
    XXH64_hash_t h = XXH3_64bits(&dk, sizeof(dk));

    const uint8_t* base = reinterpret_cast<const uint8_t*>(d0.pSysMem);
    const size_t rowBytes = std::min<size_t>(pitch != 0u ? pitch : kHashRowBytes,
                                             kHashRowBytes);
    const size_t stride = (rows > kHashRows) ? (rows / kHashRows) : 1u;

    for (size_t r = 0; r < rows; r += stride) {
      h = XXH3_64bits_withSeed(base + r * pitch, rowBytes, h);
    }

    image->setHash(h);
  }


  void D3D11Initializer::InitUavCounter(
          D3D11UnorderedAccessView*   pUav) {
    auto counterBuffer = pUav->GetCounterSlice();

    if (!counterBuffer.defined())
      return;

    std::lock_guard<dxvk::mutex> lock(m_mutex);
    m_transferCommands += 1;

    const uint32_t zero = 0;
    m_context->updateBuffer(
      counterBuffer.buffer(),
      0, sizeof(zero), &zero);

    FlushImplicit();
  }


  void D3D11Initializer::InitDeviceLocalBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    DxvkBufferSlice bufferSlice = pBuffer->GetBufferSlice();

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      m_transferMemory   += bufferSlice.length();
      m_transferCommands += 1;
      
      m_context->uploadBuffer(
        bufferSlice.buffer(),
        pInitialData->pSysMem);
    } else {
      m_transferCommands += 1;

      m_context->clearBuffer(
        bufferSlice.buffer(),
        bufferSlice.offset(),
        bufferSlice.length(),
        0u);
    }

    FlushImplicit();
  }


  void D3D11Initializer::InitHostVisibleBuffer(
          D3D11Buffer*                pBuffer,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    // If the buffer is mapped, we can write data directly
    // to the mapped memory region instead of doing it on
    // the GPU. Same goes for zero-initialization.
    DxvkBufferSlice bufferSlice = pBuffer->GetBufferSlice();

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      std::memcpy(
        bufferSlice.mapPtr(0),
        pInitialData->pSysMem,
        bufferSlice.length());
    } else {
      std::memset(
        bufferSlice.mapPtr(0), 0,
        bufferSlice.length());
    }
  }


  void D3D11Initializer::InitDeviceLocalTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    
    Rc<DxvkImage> image = pTexture->GetImage();

    auto mapMode = pTexture->GetMapMode();
    auto desc = pTexture->Desc();

    VkFormat packedFormat = m_parent->LookupPackedFormat(desc->Format, pTexture->GetFormatMode()).Format;
    auto formatInfo = imageFormatInfo(packedFormat);

    if (pInitialData != nullptr && pInitialData->pSysMem != nullptr) {
      // pInitialData is an array that stores an entry for
      // every single subresource. Since we will define all
      // subresources, this counts as initialization.
      for (uint32_t layer = 0; layer < desc->ArraySize; layer++) {
        for (uint32_t level = 0; level < desc->MipLevels; level++) {
          const uint32_t id = D3D11CalcSubresource(
            level, layer, desc->MipLevels);

          VkOffset3D mipLevelOffset = { 0, 0, 0 };
          VkExtent3D mipLevelExtent = pTexture->MipLevelExtent(level);

          if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_STAGING) {
            m_transferCommands += 1;
            m_transferMemory   += pTexture->GetSubresourceLayout(formatInfo->aspectMask, id).Size;
            
            VkImageSubresourceLayers subresourceLayers;
            subresourceLayers.aspectMask     = formatInfo->aspectMask;
            subresourceLayers.mipLevel       = level;
            subresourceLayers.baseArrayLayer = layer;
            subresourceLayers.layerCount     = 1;
            
            if (formatInfo->aspectMask != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
              m_context->uploadImage(
                image, subresourceLayers,
                pInitialData[id].pSysMem,
                pInitialData[id].SysMemPitch,
                pInitialData[id].SysMemSlicePitch);
            } else {
              m_context->updateDepthStencilImage(
                image, subresourceLayers,
                VkOffset2D { mipLevelOffset.x,     mipLevelOffset.y      },
                VkExtent2D { mipLevelExtent.width, mipLevelExtent.height },
                pInitialData[id].pSysMem,
                pInitialData[id].SysMemPitch,
                pInitialData[id].SysMemSlicePitch,
                packedFormat);
            }
          }

          if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
            util::packImageData(pTexture->GetMappedBuffer(id)->mapPtr(0),
              pInitialData[id].pSysMem, pInitialData[id].SysMemPitch, pInitialData[id].SysMemSlicePitch,
              0, 0, pTexture->GetVkImageType(), mipLevelExtent, 1, formatInfo, formatInfo->aspectMask);
          }
        }
      }
    } else {
      if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_STAGING) {
        m_transferCommands += 1;
        
        // While the Microsoft docs state that resource contents are
        // undefined if no initial data is provided, some applications
        // expect a resource to be pre-cleared. We can only do that
        // for non-compressed images, but that should be fine.
        VkImageSubresourceRange subresources;
        subresources.aspectMask     = formatInfo->aspectMask;
        subresources.baseMipLevel   = 0;
        subresources.levelCount     = desc->MipLevels;
        subresources.baseArrayLayer = 0;
        subresources.layerCount     = desc->ArraySize;

        if (formatInfo->flags.any(DxvkFormatFlag::BlockCompressed, DxvkFormatFlag::MultiPlane)) {
          m_context->clearCompressedColorImage(image, subresources);
        } else {
          if (subresources.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT) {
            VkClearColorValue value = { };

            m_context->clearColorImage(
              image, value, subresources);
          } else {
            VkClearDepthStencilValue value;
            value.depth   = 0.0f;
            value.stencil = 0;
            
            m_context->clearDepthStencilImage(
              image, value, subresources);
          }
        }
      }

      if (mapMode != D3D11_COMMON_TEXTURE_MAP_MODE_NONE) {
        for (uint32_t i = 0; i < pTexture->CountSubresources(); i++) {
          auto buffer = pTexture->GetMappedBuffer(i);
          std::memset(buffer->mapPtr(0), 0, buffer->info().size);
        }
      }
    }

    FlushImplicit();
  }


  void D3D11Initializer::InitHostVisibleTexture(
          D3D11CommonTexture*         pTexture,
    const D3D11_SUBRESOURCE_DATA*     pInitialData) {
    Rc<DxvkImage> image = pTexture->GetImage();

    for (uint32_t layer = 0; layer < image->info().numLayers; layer++) {
      for (uint32_t level = 0; level < image->info().mipLevels; level++) {
        VkImageSubresource subresource;
        subresource.aspectMask = image->formatInfo()->aspectMask;
        subresource.mipLevel   = level;
        subresource.arrayLayer = layer;

        VkExtent3D blockCount = util::computeBlockCount(
          image->mipLevelExtent(level),
          image->formatInfo()->blockSize);

        VkSubresourceLayout layout = image->querySubresourceLayout(subresource);

        auto initialData = pInitialData
          ? &pInitialData[D3D11CalcSubresource(level, layer, image->info().mipLevels)]
          : nullptr;

        for (uint32_t z = 0; z < blockCount.depth; z++) {
          for (uint32_t y = 0; y < blockCount.height; y++) {
            auto size = blockCount.width * image->formatInfo()->elementSize;
            auto dst = image->mapPtr(layout.offset + y * layout.rowPitch + z * layout.depthPitch);

            if (initialData) {
              auto src = reinterpret_cast<const char*>(initialData->pSysMem)
                       + y * initialData->SysMemPitch
                       + z * initialData->SysMemSlicePitch;
              std::memcpy(dst, src, size);
            } else {
              std::memset(dst, 0, size);
            }
          }
        }
      }
    }

    // Initialize the image on the GPU
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    VkImageSubresourceRange subresources = image->getAvailableSubresources();
    
    m_context->initImage(image, subresources, VK_IMAGE_LAYOUT_PREINITIALIZED);

    m_transferCommands += 1;
    FlushImplicit();
  }


  void D3D11Initializer::FlushImplicit() {
    if (m_transferCommands > MaxTransferCommands
     || m_transferMemory   > MaxTransferMemory)
      FlushInternal();
  }


  void D3D11Initializer::FlushInternal() {
    m_context->flushCommandList();
    
    m_transferCommands = 0;
    m_transferMemory   = 0;
  }

}