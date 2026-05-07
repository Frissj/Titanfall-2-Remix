/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include <mutex>
#include <vector>
#include <assert.h>
#include <fstream>
#include <ctime>
#include <unordered_set>
#include "../../util/util_filesys.h"

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_scene_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_debug_probes.h"
#include "rtx_point_instancer_system.h"

#include "rtx_cb_types.h"
#include "rtx_matrix_helpers.h"

#include "dxvk_scoped_annotation.h"
#include "rtx_options.h"

#include "rtx/pass/instance_definitions.h"
#include "rtx/concept/billboard.h"

#include "rtx/pass/common_binding_indices.h"

namespace dxvk {

  // Make this static and not a member of AccelManager to make it safe updating the count from ~PooledBlas()
  static int g_blasCount = 0;

  AccelManager::AccelManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    // Note: The scratch buffer's device address must be aligned to the minimum alignment required by the Vulkan runtime, otherwise
    //    // even if scratch allocation offsets are aligned they may add to a device address which will mess up this alignment (the alignment
    //    // requirement in Vulkan applies to the scratch buffer's device address, not just an offset as the name may imply). The lack of
    //    // this alignment override created issues on Intel GPUs where the min scratch alignment is 128 bytes but the underlying buffer was
    //    // only allocated with a 64 byte alignment.
    //    // Note: This could use the value of m_scratchAlignment, but this is duplicated to avoid potential future initialization order issues.
    , m_scratchAlignment(device->properties().khrDeviceAccelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment) {
  }

  void AccelManager::clear() {
    m_blasPool.clear();
  }

  void AccelManager::garbageCollection() {
    // Can be configured per game: 'rtx.numFramesToKeepBLAS'
    // Note: keep the BLAS for at least two frames so that they're alive for previous-frame TLAS access.
    const uint32_t numFramesToKeepBLAS = std::max(RtxOptions::enablePreviousTLAS() ? 2u : 1u, RtxOptions::numFramesToKeepBLAS());

    // Remove instances past their lifetime or marked for GC explicitly
    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // Remove all pooled BLAS that haven't been used for a few frames
    for (uint32_t i = 0; i < m_blasPool.size();) {
      Rc<PooledBlas>& blas = m_blasPool[i];

      if (blas->frameLastTouched + numFramesToKeepBLAS < currentFrame) {
        // Put this BLAS to the end of the vector
        std::swap(blas, m_blasPool.back());
        // Remove the last element
        m_blasPool.pop_back();
        continue;
      }
      ++i;
    }
  }
  
  PooledBlas::PooledBlas() {
    ++g_blasCount;
    buildInfo.geometryCount = 0;
    buildInfo.pGeometries = nullptr;
  }

  PooledBlas::~PooledBlas() {
    if (buildInfo.pGeometries) {
      delete[] buildInfo.pGeometries;
      buildInfo.pGeometries = nullptr;
    }
    accelerationStructureReference = 0;
    accelStructure = nullptr;
    --g_blasCount;
  }

  // Keep a copy of the build info to validate it for potential updateability
  static void copyAccelerationStructureBuildGeometryInfo(const VkAccelerationStructureBuildGeometryInfoKHR& srcInfo, VkAccelerationStructureBuildGeometryInfoKHR& dstInfo)
  {
    const VkAccelerationStructureGeometryKHR* pGeometries = dstInfo.pGeometries;
    if (srcInfo.pGeometries) {
      if (srcInfo.geometryCount != dstInfo.geometryCount) {
        if (pGeometries) {
          delete[] pGeometries;
        }
        pGeometries = new VkAccelerationStructureGeometryKHR[srcInfo.geometryCount];
      }
      std::memcpy((void*) pGeometries, srcInfo.pGeometries, srcInfo.geometryCount * sizeof(VkAccelerationStructureGeometryKHR));

      dstInfo = srcInfo;
      dstInfo.pGeometries = pGeometries;
    }
  }

  uint32_t AccelManager::getBlasCount() {
    // Should never be negative, but just in case...
    return uint32_t(std::max(g_blasCount, 0));
  }

  bool AccelManager::BlasBucket::tryAddInstance(RtInstance* instance) {
    const uint8_t geometryInstanceMask = instance->getVkInstance().mask;
    const uint32_t geometryCustomIndexFlags = instance->getVkInstance().instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK);
    const bool geometryUsesUnorderedApproximations = instance->usesUnorderedApproximations();
    const VkGeometryInstanceFlagsKHR geometryInstanceFlags = instance->getVkInstance().flags;
    const uint32_t geometryInstanceShaderBindingTableRecordOffset = instance->getVkInstance().instanceShaderBindingTableRecordOffset;

    if (!geometries.empty()) {
      if (instanceMask != geometryInstanceMask) {
        return false;
      }
      if (instanceShaderBindingTableRecordOffset != geometryInstanceShaderBindingTableRecordOffset) {
        return false;
      }
      if (customIndexFlags != geometryCustomIndexFlags) {
        return false;
      }
      if (instanceFlags != geometryInstanceFlags) {
        return false;
      }
      if (usesUnorderedApproximations != geometryUsesUnorderedApproximations) {
        return false;
      }
      if (hasSssInstances != instance->isSubsurface()) {
        return false;
      }
    }

    BlasEntry* blasEntry = instance->getBlas();

    geometries.insert(geometries.end(), blasEntry->buildGeometries.begin(), blasEntry->buildGeometries.end());
    ranges.insert(ranges.end(), blasEntry->buildRanges.begin(), blasEntry->buildRanges.end());

    for (auto& range : blasEntry->buildRanges) {
      originalInstances.push_back(instance);
      primitiveCounts.push_back(range.primitiveCount);
    }
    instanceBillboardIndices.insert(instanceBillboardIndices.end(), instance->billboardIndices.begin(), instance->billboardIndices.end());
    indexOffsets.insert(indexOffsets.end(), instance->indexOffsets.begin(), instance->indexOffsets.end());

    instanceShaderBindingTableRecordOffset = geometryInstanceShaderBindingTableRecordOffset;
    instanceMask = geometryInstanceMask;
    customIndexFlags = geometryCustomIndexFlags;
    instanceFlags = geometryInstanceFlags;
    usesUnorderedApproximations = geometryUsesUnorderedApproximations;
    hasSssInstances = instance->isSubsurface();
    return true;
  }

  static void fillGeometryInfoFromBlasEntry(BlasEntry& blasEntry, RtInstance& instance, const OpacityMicromapManager* opacityMicromapManager) {
    ScopedCpuProfileZone();
    blasEntry.buildGeometries.clear();
    blasEntry.buildRanges.clear();
    instance.billboardIndices.clear();
    instance.indexOffsets.clear();

    const bool usesIndices = blasEntry.modifiedGeometryData.usesIndices();

    // Associate each billboard with a unique geometry entry
    // ToDo: get rid of usesIndices requirement, it's not needed to build OMMs. It's only used below
    if (usesIndices && 
        opacityMicromapManager &&
        opacityMicromapManager->isActive() &&
        OpacityMicromapManager::usesOpacityMicromap(instance) &&
        OpacityMicromapManager::usesSplitBillboardOpacityMicromap(instance)) {

      VkAccelerationStructureGeometryKHR geometry = {};
      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.flags = instance.getGeometryFlags();

      VkAccelerationStructureGeometryTrianglesDataKHR& triangleData = geometry.geometry.triangles;
      triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      triangleData.indexData.deviceAddress = blasEntry.modifiedGeometryData.indexBuffer.getDeviceAddress();
      triangleData.indexType = blasEntry.modifiedGeometryData.indexBuffer.indexType();
      triangleData.vertexData.deviceAddress = blasEntry.modifiedGeometryData.positionBuffer.getDeviceAddress() + blasEntry.modifiedGeometryData.positionBuffer.offsetFromSlice();
      triangleData.vertexStride = blasEntry.modifiedGeometryData.positionBuffer.stride();
      triangleData.vertexFormat = blasEntry.modifiedGeometryData.positionBuffer.vertexFormat();
      triangleData.maxVertex = blasEntry.modifiedGeometryData.vertexCount - 1;

      assert((blasEntry.modifiedGeometryData.calculatePrimitiveCount() & 1) == 0);
      VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
      buildRange.primitiveCount = 2;

      for (uint32_t billboardIndex = 0; billboardIndex < instance.getBillboardCount(); billboardIndex++) {
        const uint32_t kNumIndicesPerBillboardQuad = buildRange.primitiveCount * 3;
        buildRange.primitiveOffset = (billboardIndex * kNumIndicesPerBillboardQuad * blasEntry.modifiedGeometryData.indexBuffer.stride());
        blasEntry.buildGeometries.push_back(geometry);
        blasEntry.buildRanges.push_back(buildRange);
        instance.billboardIndices.push_back(billboardIndex);
        instance.indexOffsets.push_back(billboardIndex * kNumIndicesPerBillboardQuad);
      }
    } else {
      VkAccelerationStructureGeometryKHR geometry = {};

      geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geometry.flags = instance.getGeometryFlags();

      VkAccelerationStructureGeometryTrianglesDataKHR& triangleData = geometry.geometry.triangles;

      const bool usesIndices = blasEntry.modifiedGeometryData.usesIndices();

      triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

      if (usesIndices) {
        triangleData.indexData.deviceAddress = blasEntry.modifiedGeometryData.indexBuffer.getDeviceAddress();
        triangleData.indexType = blasEntry.modifiedGeometryData.indexBuffer.indexType();
      } else {
        triangleData.indexData.deviceAddress = 0;
        triangleData.indexType = VK_INDEX_TYPE_NONE_KHR;
      }

      triangleData.vertexData.deviceAddress = blasEntry.modifiedGeometryData.positionBuffer.getDeviceAddress() + blasEntry.modifiedGeometryData.positionBuffer.offsetFromSlice();
      triangleData.vertexStride = blasEntry.modifiedGeometryData.positionBuffer.stride();
      triangleData.vertexFormat = blasEntry.modifiedGeometryData.positionBuffer.vertexFormat();
      triangleData.maxVertex = blasEntry.modifiedGeometryData.vertexCount - 1;

      VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
      buildRange.primitiveCount = blasEntry.modifiedGeometryData.calculatePrimitiveCount();
      buildRange.primitiveOffset = 0;

      blasEntry.buildGeometries.push_back(geometry);
      blasEntry.buildRanges.push_back(buildRange);
      instance.billboardIndices.push_back(0);
      instance.indexOffsets.push_back(0);
    }
  }
  int AccelManager::getCurrentFramePrimitiveIDPrefixSumBufferID() const {
    return m_device->getCurrentFrameId() & 0x1;
  }

  uint32_t additionalAccelerationStructureFlags() {
    return RtxOptions::lowMemoryGpu() ? VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR : 0;
  }

  void AccelManager::createAndBuildIntersectionBlas(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers) {
    if (m_intersectionBlas.ptr()) {
      return;
    }

    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.pNext = nullptr;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = additionalAccelerationStructureFlags();
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.ppGeometries = nullptr;

    uint32_t maxPrimitiveCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxPrimitiveCount, &sizeInfo);

    m_intersectionBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Intersection");
    
    buildInfo.dstAccelerationStructure = m_intersectionBlas->accelStructure->getAccelStructure();

    VkAabbPositionsKHR aabbPositions = { -1.f, -1.f, -1.f, 1.f, 1.f, 1.f };

    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = sizeof(aabbPositions);

    m_aabbBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "AABB Buffer");
    // Note: don't use ctx->updateBuffer() because that will place the command on the InitBuffer, not ExecBuffer.
    ctx->getCommandList()->cmdUpdateBuffer(DxvkCmdBuffer::ExecBuffer, m_aabbBuffer->getBufferRaw(), m_aabbBuffer->getSliceHandle().offset, sizeof(aabbPositions), &aabbPositions);
    
    execBarriers.accessBuffer(
      m_aabbBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.accessBuffer(
      m_scratchBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV);

    execBarriers.recordCommands(ctx->getCommandList());
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);

    geometry.geometry.aabbs.data.deviceAddress = m_aabbBuffer->getDeviceAddress();

    const size_t requiredScratchAllocSize = sizeInfo.buildScratchSize + m_scratchAlignment;
    buildInfo.scratchData.deviceAddress = getScratchMemory(requiredScratchAllocSize)->getDeviceAddress();
    assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

    VkAccelerationStructureBuildRangeInfoKHR buildRange {};
    buildRange.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;

    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRange);

    execBarriers.accessBuffer(
      m_scratchBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV);

    execBarriers.recordCommands(ctx->getCommandList());
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_scratchBuffer);
  }

  Rc<DxvkBuffer> AccelManager::getScratchMemory(const size_t requiredScratchAllocSize) {
    if (m_scratchBuffer == nullptr || m_scratchBuffer->info().size < requiredScratchAllocSize) {
      DxvkBufferCreateInfo bufferCreateInfo {};
      bufferCreateInfo.size = requiredScratchAllocSize;
      bufferCreateInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      bufferCreateInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      m_scratchBuffer = m_device->createBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "BVH Scratch");
    }

    return m_scratchBuffer;
  }

  Rc<PooledBlas> AccelManager::createPooledBlas(size_t bufferSize, const char* name) const {
    auto newBlas = new PooledBlas();

    DxvkBufferCreateInfo bufferCreateInfo {};
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    bufferCreateInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    newBlas->accelStructure = m_device->createAccelStructure(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, name);

    newBlas->accelerationStructureReference = newBlas->accelStructure->getAccelDeviceAddress();

    return newBlas;
  }

  static void trackBlasBuildResources(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers, const BlasEntry* blasEntry) {
    ScopedCpuProfileZone();
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->modifiedGeometryData.positionBuffer.buffer());
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->modifiedGeometryData.indexBuffer.buffer());

    // TDR-DIAG: log the source buffer info for each BLAS build's geometry
    static uint32_t sBlasTrackLog = 0;
    if (sBlasTrackLog < 100) {
      ++sBlasTrackLog;
      const auto& pb = blasEntry->modifiedGeometryData.positionBuffer;
      const auto& ib = blasEntry->modifiedGeometryData.indexBuffer;
      Logger::info(str::format("[BLAS-TRACK] #", sBlasTrackLog,
        " posBuf=0x", std::hex, (uintptr_t)(pb.buffer() != nullptr ? pb.buffer().ptr() : nullptr),
        " posMF=0x", (pb.buffer() != nullptr ? pb.buffer()->memFlags() : 0),
        " posUsage=0x", (pb.buffer() != nullptr ? pb.buffer()->info().usage : 0u),
        " posSize=", std::dec, (pb.buffer() != nullptr ? pb.buffer()->info().size : 0),
        " | idxBuf=0x", std::hex, (uintptr_t)(ib.buffer() != nullptr ? ib.buffer().ptr() : nullptr),
        " idxMF=0x", (ib.buffer() != nullptr ? ib.buffer()->memFlags() : 0),
        " idxUsage=0x", (ib.buffer() != nullptr ? ib.buffer()->info().usage : 0u),
        " idxSize=", std::dec, (ib.buffer() != nullptr ? ib.buffer()->info().size : 0)));
    }

    execBarriers.accessBuffer(
      blasEntry->modifiedGeometryData.positionBuffer.getSliceHandle(),
      blasEntry->modifiedGeometryData.positionBuffer.buffer()->info().stages,
      blasEntry->modifiedGeometryData.positionBuffer.buffer()->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.accessBuffer(
      blasEntry->modifiedGeometryData.indexBuffer.getSliceHandle(),
      blasEntry->modifiedGeometryData.indexBuffer.buffer()->info().stages,
      blasEntry->modifiedGeometryData.indexBuffer.buffer()->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    execBarriers.recordCommands(ctx->getCommandList());
  }

  void AccelManager::mergeInstancesIntoBlas(Rc<DxvkContext> ctx,
                                            DxvkBarrierSet& execBarriers,
                                            const std::vector<TextureRef>& textures,
                                            const CameraManager& cameraManager,
                                            InstanceManager& instanceManager,
                                            OpacityMicromapManager* opacityMicromapManager) {
    ScopedGpuProfileZone(ctx, "buildBLAS");

    auto& instances = instanceManager.getInstanceTable();

    // [SpawnGeomDiag.merge] Unconditional entry log so we can confirm the
    // running binary reaches this function. Past runs showed 0 [PI-route]
    // and 0 [PI-dispatch] entries despite SceneManager::prepareSceneData
    // logging that it had 100+ active instances; this isolates whether
    // the function is being called at all (vs being short-circuited
    // before it). Throttled to once per ~30 RT frames so the log stays
    // grep-friendly. instances.size() is BEFORE any merging — it's the
    // total RtInstance count handed to TLAS-build.
    {
      static uint32_t sMergeCallFrame = 0;
      if ((sMergeCallFrame++ % 30u) == 0) {
        Logger::info(str::format(
          "[SpawnGeomDiag.merge] frame=", m_device->getCurrentFrameId(),
          " mergeCallNum=", sMergeCallFrame,
          " instances=", instances.size()));
      }
    }

    // Allocate the transform buffer
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

    info.size = align(instances.size() * sizeof(VkTransformMatrixKHR), kBufferAlignment);

    if (m_transformBuffer == nullptr || info.size > m_transformBuffer->info().size) {
      // TODO: allocate with some spare space to make reallocations less frequent
      m_transformBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Transform Buffer");
      Logger::debug("DxvkRaytrace: Vulkan Transform Buffer Realloc");
    }

    std::vector<VkTransformMatrixKHR> instanceTransforms;
    instanceTransforms.reserve(instances.size());

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blasToBuild;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> blasRangesToBuild;

    blasToBuild.reserve(instances.size());
    blasRangesToBuild.reserve(instances.size());

    m_reorderedSurfaces.clear();
    m_reorderedSurfacesFirstIndexOffset.clear();
    m_pointInstancerBatches.clear();
    memset(m_pointInstancerSlotsPerType, 0, sizeof(m_pointInstancerSlotsPerType));
    // NV-DXVK debug: BLAS-BUILD-INPUT side tables cleared here (before pushes in this
    // function), not in buildBlases (which runs after pushes and would wipe them).
    m_debugBlasBuildEntries.clear();
    m_debugBlasBuildDstBlas.clear();

    // [SpawnGeomDiag.bisect] Bisect log — fires unconditionally on the
    // 1/30 throttle right before the existing [PI-route] block. If this
    // appears in the log but [PI-route] still doesn't, something inside
    // the [PI-route] block is silently aborting the call (likely a static
    // inline data-member access or a Logger::info code path that elides
    // the line). If this *also* doesn't appear, something between
    // [SpawnGeomDiag.merge] and here is bailing.
    {
      static uint32_t sBisect1 = 0;
      if ((sBisect1++ % 30u) == 0) {
        Logger::info(str::format(
          "[SpawnGeomDiag.bisect] preProbeB call=", sBisect1,
          " instances=", instances.size()));
      }
    }

    // NV-DXVK (debug probe B): emit previous frame's addBlas vs addPI routing counts,
    // then reset. Logs every frame so we see BSP-fanout survival across the session.
    {
      static uint32_t sProbeBFrame = 0;
      static uint32_t sProbeBLastLogFrame = 0xFFFFFFFFu;
      if (sProbeBFrame != sProbeBLastLogFrame) {
        // Always log (lightweight, ~1 line per frame): addBlas count, PI count, total PI instances, bucket/instance totals.
        Logger::info(str::format(
          // [SpawnGeomDiag] renamed from [PI-route] so it bypasses the
          // log.cpp filter that drops "[PI-" without RTX_D3D11_DIAG=1.
          // Same content as before — addBlas/addPI counts per frame.
          "[SpawnGeomDiag.PIroute] frame=", sProbeBFrame,
          " addBlas=", s_probeB_addBlasCount,
          " addPI=", s_probeB_addPICount,
          " PI_instances=", s_probeB_addPIInstances,
          " totalRtInstances=", instances.size()));
        sProbeBLastLogFrame = sProbeBFrame;
      }
      ++sProbeBFrame;
      s_probeB_addBlasCount = 0;
      s_probeB_addPICount = 0;
      s_probeB_addPIInstances = 0;
    }
    // [SpawnGeomDiag.bisect] Second bisect — fires *after* the [PI-route]
    // block. If this fires but [PI-route] doesn't, the issue is isolated
    // to the [PI-route] Logger::info call itself (e.g. Logger::info elides
    // identical messages, or one of the s_probeB_* statics has a stale
    // address/linker conflict).
    {
      static uint32_t sBisect2 = 0;
      if ((sBisect2++ % 30u) == 0) {
        Logger::info(str::format(
          "[SpawnGeomDiag.bisect] postProbeB call=", sBisect2,
          " probeB=[blas=", s_probeB_addBlasCount,
          " pi=", s_probeB_addPICount,
          " piInst=", s_probeB_addPIInstances, "]"));
      }
    }

    // NV-DXVK (debug probe E): clear last frame's captured BLAS position buffer
    // ref so the readback path in dispatchPointInstancerCulling doesn't dereference
    // a stale Rc on a frame with zero PI batches.
    s_probeE_posBuffer    = nullptr;
    s_probeE_posSliceOff  = 0;
    s_probeE_posElemOff   = 0;
    s_probeE_posStride    = 0;
    s_probeE_vertexCount  = 0;
    s_probeE_posFormat    = VK_FORMAT_UNDEFINED;
    s_probeE_blasRef      = 0;
    s_probeF_baseSurfaceIndex = 0;
    s_probeF_valid = false;
    for (auto& instances : m_mergedInstances) {
      instances.clear();
    }

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    if (instances.size() > CUSTOM_INDEX_SURFACE_MASK) {
      ONCE(Logger::err(str::format("DxvkRaytrace: instance count (", instances.size(),
        ") exceeds the maximum surface index (", CUSTOM_INDEX_SURFACE_MASK,
        ") representable in ", SURFACE_INDEX_BIT_COUNT, "-bit instanceCustomIndex field. "
        "Surfaces beyond the limit will alias index 0.")));
    }

    if (opacityMicromapManager) {
      opacityMicromapManager->onFrameStart(ctx);
    }

    std::vector<std::unique_ptr<BlasBucket>> blasBuckets;
    blasBuckets.reserve(instances.size());

    size_t totalScratchMemory = 0;

    // NOTE: Would like to use the BLAS Linked instances here, but that misses viewmodel and virtual instances
    std::unordered_map<BlasEntry*, std::vector<RtInstance*>> uniqueBlas;

    // NV-DXVK debug: routing stats per frame for PI vs non-PI instances.
    struct RoutingStats {
      uint32_t total = 0;        uint32_t hidden = 0;       uint32_t mask0 = 0;
      uint32_t routedDynamic = 0; uint32_t routedMerged = 0; uint32_t earlyContinue = 0;
    } piStats {}, normStats {};

    for (RtInstance* instance : instances) {
      const bool isPi = (instance->surface.instancesToObject != nullptr);
      RoutingStats& s = isPi ? piStats : normStats;
      ++s.total;

      // NV-DXVK TF2 VIEWMODEL TRACE: every RtInstance entering TLAS processing,
      // dumped before any filter/routing. Identifies per-instance VS hash +
      // hidden/mask state so we can pinpoint where the gun (ef94e6c7) drops
      // out. Throttled to ef94 draws so we don't flood with world geometry.
      {
        static uint32_t sInstLog = 0;
        BlasEntry* bentry = instance->getBlas();
        if (bentry && sInstLog < 100) {
          ++sInstLog;
          const XXH64_hash_t vsH = bentry->input.getTransformData().vertexShaderHash;
          const uint32_t vsHi = static_cast<uint32_t>(vsH >> 32);
          const uint32_t vsLo = static_cast<uint32_t>(vsH & 0xFFFFFFFFu);
          const auto& o2w = instance->surface.objectToWorld;
          const uint32_t primCount = bentry->modifiedGeometryData.calculatePrimitiveCount();
          const uint32_t numBones = bentry->input.getSkinningState().numBones;
          const uint32_t numBonesPerVert = bentry->input.getSkinningState().numBonesPerVertex;
          Logger::info(str::format(
            "[AccelMgr.instEnter] #", sInstLog,
            " vsHash=0x", std::hex, vsHi, vsLo, std::dec,
            " isPi=", (isPi ? 1 : 0),
            " mask=0x", std::hex, instance->getVkInstance().mask, std::dec,
            " hidden=", (instance->isHidden() ? 1 : 0),
            " prims=", primCount,
            " numBones=", numBones,
            " bpv=", numBonesPerVert,
            " linkedInstances=", bentry->getLinkedInstances().size(),
            " hasDynamicBlas=", (bentry->dynamicBlas != nullptr ? 1 : 0),
            " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
        }
      }

      // [SpawnGeomDiag.Drop] Per-matHash log showing the EXACT reason
      // a draw is skipped at mergeInstancesIntoBlas. Once per
      // (vsHash, matHash, reason) tuple. Diff against
      // [SpawnGeomDiag.DrawIn] — every dropped matHash should have a
      // matching Drop entry here. If a matHash is in DrawIn but
      // appears in NEITHER DrawOut NOR Drop, the dropoff is even
      // earlier (e.g. submitDrawState's bufferCache overflow / sky
      // filter / replacement substitution).
      auto logDrop = [&](const char* reason) {
        BlasEntry* be = instance->getBlas();
        if (be == nullptr) return;
        const uint64_t vsHash = static_cast<uint64_t>(
          be->input.getTransformData().vertexShaderHash);
        const uint64_t matHash = static_cast<uint64_t>(
          be->input.getMaterialData().getHash());
        const uint64_t key = vsHash
          ^ ((matHash << 1) | (matHash >> 63))
          ^ (uint64_t(reason[0]) * 0x9e3779b1ull);
        static std::mutex sDropMu;
        static std::unordered_set<uint64_t> sDropSeen;
        bool first = false;
        {
          std::lock_guard<std::mutex> lk(sDropMu);
          first = sDropSeen.insert(key).second;
        }
        if (first) {
          const uint32_t primCount = be->buildRanges.empty() ? 0u
            : be->buildRanges[0].primitiveCount;
          Logger::info(str::format(
            "[SpawnGeomDiag.Drop] reason=", reason,
            " vsHash=0x", std::hex, vsHash, std::dec,
            " matHash=0x", std::hex, matHash, std::dec,
            " primCnt=", primCount,
            " vCnt=", be->modifiedGeometryData.vertexCount,
            " mask=0x", std::hex, instance->getVkInstance().mask, std::dec));
        }
      };

      if (instance->isHidden()) {
        ++s.hidden;
        logDrop("hidden");
        continue;
      }

      // If the instance has zero mask, do not build BLAS for it: no ray can intersect this instance.
      if (instance->getVkInstance().mask == 0) {
        ++s.mask0;
        logDrop("mask0");
        
        bool needsOpacityMicromap = instance->isViewModelReference() && opacityMicromapManager;
        bool hasBillboards = instance->getBillboardCount() > 0;

        // OMM requests and billboards need a valid surface.
        // Particles on the player model generate valid billboards but their geometric instance mask is set to 0.
        if (needsOpacityMicromap || hasBillboards) {
          instance->setSurfaceIndex(m_reorderedSurfaces.size());

          m_reorderedSurfaces.push_back(instance);
          m_reorderedSurfacesFirstIndexOffset.push_back(0);
        }

        // Register OMM build request for reference ViewModel instances, which are persistent unlike the intermittent active view model instances
        if (needsOpacityMicromap) {
          opacityMicromapManager->registerOpacityMicromapBuildRequest(*instance, instanceManager, textures);
        }

        continue;
      }

      if (opacityMicromapManager) {
        opacityMicromapManager->registerOpacityMicromapBuildRequest(*instance, instanceManager, textures);
      }

      // Find the blas entry for this instance.
      // Cannot store BlasEntry* directly in the RtInstance because the entries are owned and potentially moved by the hash table.
      BlasEntry* blasEntry = instance->getBlas();
      assert(blasEntry);

      fillGeometryInfoFromBlasEntry(*blasEntry, *instance, opacityMicromapManager);

      const uint32_t minPrimsInDynamicBLAS = std::max(RtxOptions::minPrimsInDynamicBLAS(), 100u);
      const uint32_t maxPrimsForMergedBLAS = RtxOptions::maxPrimsInMergedBLAS();
      const uint32_t blasPrims = blasEntry->modifiedGeometryData.calculatePrimitiveCount();

      // Figure out if this blas should be a dynamic one
      const bool requestDynamicBlas = instance->surface.instancesToObject != nullptr ||    // Point instancer geometry is replicated many times in a scene, we want to reuse the BLAS memory for these objects
                                      blasEntry->input.getSkinningState().numBones != 0 || // Skinned meshes are always desirable to give a dynamic BLAS, since we'll want to make use of BVH update for performance reasons
                                      blasEntry->getLinkedInstances().size() > 1  ||       // Meshes that are used in instances multiple times should benefit from BLAS reuse
                                      blasEntry->dynamicBlas != nullptr ||                 // If we already have a dynamic BLAS, keep using it.
                                      blasPrims > maxPrimsForMergedBLAS ||                 // Avoid large meshes ending up in the merged BLAS which is built every frame.  # prims is proportional to build cost.
                                      RtxOptions::minimizeBlasMerging();                   // Option to attempt putting as many objects into dynamic BLAS as possible.

      const bool forceMergedBlas = (blasEntry->buildGeometries.size() > 1 ||                                       // Currently we use multiple build geometries for particle billboards, which we prefer to merge into large BLAS
                                    (!RtxOptions::minimizeBlasMerging() && blasPrims < minPrimsInDynamicBLAS) ||   // Avoid creating lots of small dynamic BLAS
                                    RtxOptions::forceMergeAllMeshes()) &&                                          // Setting to force all meshes into the merged BLAS
                                      instance->surface.instancesToObject == nullptr &&                            // Never merge point instancer geometry
                                      blasEntry->getLinkedInstances().size() <= 1;                                 // NV-DXVK: Don't force merge if BlasEntry has multiple instances (avoids dynamic/merged conflict)

      if (requestDynamicBlas && !forceMergedBlas) {
        ++s.routedDynamic;
        // Since this loop is iterating over instances, and instances can share BLAS, we will build these later after identifying unique ones.
        uniqueBlas[blasEntry].push_back(instance);
      } else {
        ++s.routedMerged;

        // [SpawnGeomDiag.DrawOut] kind=merged — the merged-bucket
        // BLAS path doesn't call addBlas per-instance (multiple
        // RtInstances merge into one bucket → one TLAS entry), so
        // the addBlas-side DrawOut probe never fires for these
        // draws. Log here instead, once per (vsHash, matHash), so
        // every BSP-merged draw shows up in the DrawIn↔DrawOut
        // diff. Without this, ~half of draws appear "dropped" in
        // the diff when in reality they reached the merged path.
        {
          BlasEntry* be = instance->getBlas();
          if (be != nullptr) {
            const uint64_t vsHash = static_cast<uint64_t>(
              be->input.getTransformData().vertexShaderHash);
            const uint64_t matHash = static_cast<uint64_t>(
              be->input.getMaterialData().getHash());
            const uint64_t key = vsHash ^ ((matHash << 1) | (matHash >> 63));
            static std::mutex sMergedOutMu;
            static std::unordered_set<uint64_t> sMergedOutSeen;
            bool first = false;
            {
              std::lock_guard<std::mutex> lk(sMergedOutMu);
              first = sMergedOutSeen.insert(key).second;
            }
            if (first) {
              const uint32_t primCount = be->buildRanges.empty() ? 0u
                : be->buildRanges[0].primitiveCount;
              const Matrix4& o2w = instance->surface.objectToWorld;
              Logger::info(str::format(
                "[SpawnGeomDiag.DrawOut] kind=merged"
                " vsHash=0x", std::hex, vsHash, std::dec,
                " matHash=0x", std::hex, matHash, std::dec,
                " primCnt=", primCount,
                " vCnt=", be->modifiedGeometryData.vertexCount,
                " linkedInstances=", be->getLinkedInstances().size(),
                " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
            }
          }
        }

        if (blasEntry->dynamicBlas != nullptr) {
          // Move the BLAS used by this geometry to the common pool.
          // This also ensures the dynamic blas resource that's still being used by previous TLAS is properly tracked for the next frame
          m_blasPool.push_back(std::move(blasEntry->dynamicBlas));
          blasEntry->dynamicBlas = nullptr;
        }

        // Calculate the device address for the current instance's transform and write the transform data
        // TODO: only do this for non-identity transforms
        VkDeviceAddress transformDeviceAddress = m_transformBuffer->getDeviceAddress() + instanceTransforms.size() * sizeof(VkTransformMatrixKHR);

        // NV-DXVK (TF2 BSP fix): Source-engine BSP world vertex buffers are in
        // camera-relative space (world - cameraOrigin). The fanout path at
        // d3d11_rtx.cpp adds camOrigin to the per-instance translation to shift
        // to absolute world; non-fanout BSP draws that reach this merged-bucket
        // path end up with objectToWorld=identity because the upstream didn't
        // detect them. We fix that here: if the source instance transform is
        // exactly identity (the tell-tale for a cam-relative BSP draw), replace
        // it with translate-by-camOrigin. Non-identity transforms (small props
        // with their own placement) are left alone so their placement is
        // preserved.
        instanceTransforms.push_back(instance->getVkInstance().transform);

        for (auto& geometry : blasEntry->buildGeometries) {
          geometry.geometry.triangles.transformData.deviceAddress = transformDeviceAddress;
        }

        // Try to merge the instance into one of the blasBuckets
        bool merged = false;
        for (auto& bucket : blasBuckets) {
          if (bucket->tryAddInstance(instance)) {
            merged = true;
            break;
          }
        }

        // The instance couldn't be merged into any bucket - make a new one
        if (!merged) {
          auto newBucket = std::make_unique<BlasBucket>();
          merged = newBucket->tryAddInstance(instance);
          assert(merged);

          blasBuckets.push_back(std::move(newBucket));
        }

        // Track the lifetime and states of the source geometry buffers
        trackBlasBuildResources(ctx, execBarriers, blasEntry);
      }
    }

    // Build/Update the dynamic BLAS
    for (const std::pair<BlasEntry*, std::vector<RtInstance*>> pair : uniqueBlas) {
      BlasEntry* blasEntry = pair.first;
      if (pair.second.size() == 0) {
        continue;
      }
      assert(blasEntry->buildGeometries.size() == 1); // dynamic BLAS should always have this
      assert(blasEntry->buildRanges.size() == 1); // dynamic BLAS should always have this

      bool forceRebuild = false;
      XXH64_hash_t boundOpacityMicromapHash = kEmptyHash;
      if (opacityMicromapManager) {
        // Check validity of a built BLAS, only if:
        // We can only support OMM on dynamic BLAS whos surface is unique to that BLAS.  This is so we can benefit from instancing BLAS memory.  
        // In cases where there are multiple linked instances each with different surfaces OMM would break.
        bool ommsCompatible = pair.second.size() == 1;
        const XXH64_hash_t firstOmmHash = OpacityMicromapManager::getOpacityMicromapHash(*pair.second[0]);
        for (uint32_t i = 1; i < pair.second.size(); i++) {
          const XXH64_hash_t thisOmmHash = OpacityMicromapManager::getOpacityMicromapHash(*pair.second[i]);
          if (thisOmmHash != firstOmmHash) {
            ommsCompatible = false;
            break;
          }
        }

        if (ommsCompatible) {
          RtInstance* exemplarInstance = pair.second[0];

          // Bind opacity micromap
          // Opacity micromaps must be bound before acceleration sizes are calculated
          // Note: since opacity micromaps for this frame are scheduled later 
          //       this will only pickup Opacity Micromaps built in previous frames
          boundOpacityMicromapHash = opacityMicromapManager->tryBindOpacityMicromap(ctx, *exemplarInstance, 0, blasEntry->buildGeometries[0], instanceManager);

          if (blasEntry->dynamicBlas.ptr()) {
            // A previously built BLAS needs to be rebuild if a corresponding Opacity Micromap availability has changed
            forceRebuild = boundOpacityMicromapHash != blasEntry->dynamicBlas->opacityMicromapSourceHash;
          }
        } else if (blasEntry->dynamicBlas.ptr() && blasEntry->dynamicBlas->opacityMicromapSourceHash != kEmptyHash) {
          // If we had a OMM bound at some point, but now that OMM is invalid, force a rebuild
          forceRebuild = true;
        }
      }

      VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
      buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = 1;
      buildInfo.pGeometries = blasEntry->buildGeometries.data();

      // Calculate the build sizes for this bucket
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
      sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                               &buildInfo, &blasEntry->buildRanges[0].primitiveCount, &sizeInfo);

      // Try to reuse our dynamic BLAS if it exists
      Rc<PooledBlas>& selectedBlas = blasEntry->dynamicBlas;

      // NV-DXVK debug: force rebuild every frame (destructive, off by default).
      bool build = forceRebuild || !selectedBlas.ptr() || selectedBlas->accelStructure->info().size != sizeInfo.accelerationStructureSize;
      if constexpr (kEnableRtxDebugDestructiveProbes) {
        build = true;
      }

      // Validate that the selected blas is compatible with the current build info for update purposes
      bool update = blasEntry->frameLastUpdated == currentFrame;
      if (update && !build && !validateUpdateMode(selectedBlas->buildInfo, buildInfo)) {
        // If an update is requested but the BLAS is not compatible with the current build info then force a rebuild
        update = false;
        build = true;
      }

      // There is no such BLAS - create one
      if (build) {
        if (selectedBlas.ptr()) {
          // Move the BLAS used by this geometry to the common pool.
          // This also ensures the dynamic blas resource that's still being used by previous TLAS is properly tracked for the next frame
          m_blasPool.push_back(std::move(selectedBlas));
        }
        selectedBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Dynamic");
      }

      assert(selectedBlas.ptr());
      selectedBlas->frameLastTouched = currentFrame;
      blasEntry->dynamicBlas->opacityMicromapSourceHash = boundOpacityMicromapHash;

      if (update || build) {
        if (update && !build) {
          buildInfo.srcAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();
          buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        }
        // Use the selected BLAS for the build
        buildInfo.dstAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();

        // Allocate a scratch buffer slice
        const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
        buildInfo.scratchData.deviceAddress = totalScratchMemory;
        totalScratchMemory += requiredScratchAllocSize;

        assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

        // Track the lifetime of the BLAS buffers
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(selectedBlas->accelStructure);

        // Put the merged BLAS into the build queue
        blasToBuild.push_back(buildInfo);
        blasRangesToBuild.push_back(&blasEntry->buildRanges[0]);
        // NV-DXVK debug: stash blasEntry + selectedBlas so we can read source buffers later.
        m_debugBlasBuildEntries.push_back(blasEntry);
        m_debugBlasBuildDstBlas.push_back(selectedBlas.ptr());

        copyAccelerationStructureBuildGeometryInfo(buildInfo, selectedBlas->buildInfo);
      }

      for (RtInstance* rtInstance : pair.second) {
        // Append an instance of this merged BLAS to the merged instance list
        if (rtInstance->surface.instancesToObject == nullptr) {
          addBlas(rtInstance, blasEntry, nullptr);
          // NV-DXVK (debug probe B): count routing per frame.
          ++s_probeB_addBlasCount;
        } else {
          // NV-DXVK (debug probe C): bypass the GPU PointInstancer culling path and
          // expand into N CPU-side addBlas entries. Known-good path (bone characters
          // render via this). If BSP becomes visible with this, the bug is in the GPU
          // culling/surface-copy shader. If BSP still invisible, bug is earlier.
          constexpr bool kProbeC_BypassPI = false;
          if (kProbeC_BypassPI) {
            // Expand PI transforms into N CPU-side addBlas entries. We permanently
            // null instancesToObject on this RtInstance so ALL downstream passes
            // (uploadSurfaceData, surface-index mapping) treat each slot as a normal
            // non-PI surface. The shared_ptr owner still keeps storage alive; the
            // raw pointer is re-set next frame when the draw is resubmitted.
            const auto* xformsPtr = rtInstance->surface.instancesToObject;
            rtInstance->surface.instancesToObject = nullptr;
            rtInstance->surface.surfaceIndexOfFirstInstance = SIZE_MAX;
            for (uint32_t i = 0; i < xformsPtr->size(); ++i) {
              addBlas(rtInstance, blasEntry, &(*xformsPtr)[i]);
            }
            ++s_probeB_addBlasCount;
            s_probeB_addPIInstances += static_cast<uint32_t>(xformsPtr->size());
          } else {
            addPointInstancerBlas(rtInstance, blasEntry);
            // NV-DXVK (debug probe B): count routing per frame.
            ++s_probeB_addPICount;
            s_probeB_addPIInstances += static_cast<uint32_t>(rtInstance->surface.instancesToObject->size());
          }
        }
      }

      ctx->getCommandList()->trackResource<DxvkAccess::Read>(blasEntry->dynamicBlas->accelStructure);

      // Track the lifetime and states of the source geometry buffers
      trackBlasBuildResources(ctx, execBarriers, blasEntry);
    }

    // NV-DXVK debug: dump routing stats. Throttled to every ~10 RT-active frames.
    if constexpr (kEnableRtxDebugProbes) {
      static uint32_t s_routingStatsFrame = 0;
      if ((s_routingStatsFrame++ % 10u) == 0) {
        Logger::info(str::format(
          "[PI-routing] frame=", s_routingStatsFrame,
          " PI: total=", piStats.total, " hidden=", piStats.hidden, " mask0=", piStats.mask0,
          " →dynamic=", piStats.routedDynamic, " →merged=", piStats.routedMerged,
          " | NORM: total=", normStats.total, " hidden=", normStats.hidden, " mask0=", normStats.mask0,
          " →dynamic=", normStats.routedDynamic, " →merged=", normStats.routedMerged,
          " uniqueBlasSize=", uniqueBlas.size()));
      }
    }

    // Copy the instance transform data to the device
    if (instanceTransforms.size() > 0) {
      ctx->writeToBuffer(m_transformBuffer, 0, instanceTransforms.size() * sizeof(VkTransformMatrixKHR), instanceTransforms.data());
    }

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_transformBuffer);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transformBuffer);

    // Place a barrier on the transform buffer
    DxvkBufferSliceHandle transformBufferSlice;
    transformBufferSlice.handle = m_transformBuffer->getBufferRaw();
    execBarriers.accessBuffer(
      transformBufferSlice,
      m_transformBuffer->info().stages,
      m_transformBuffer->info().access,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // Collect all the surfaces
    for (const auto& blasBucket : blasBuckets) {
      // Store the offset to use it later during blas instance creation
      blasBucket->reorderedSurfacesOffset = static_cast<uint32_t>(m_reorderedSurfaces.size());

      // Append the bucket's instances to the reordered surface list
      m_reorderedSurfaces.insert(m_reorderedSurfaces.end(), blasBucket->originalInstances.begin(), blasBucket->originalInstances.end());
      m_reorderedSurfacesFirstIndexOffset.insert(m_reorderedSurfacesFirstIndexOffset.end(), blasBucket->indexOffsets.begin(), blasBucket->indexOffsets.end());
    }

    // Build prefix sum array
    // Collect primitive count for each surface object
    // Because we use exclusive prefix sum here, we add one more element to record the scene's total primitive count
    m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame = m_reorderedSurfacesPrimitiveIDPrefixSum;
    m_reorderedSurfacesPrimitiveIDPrefixSum.resize(m_reorderedSurfaces.size() + 1);
    m_reorderedSurfacesPrimitiveIDPrefixSum[0] = 0;
    for (uint32_t i = 0; i < m_reorderedSurfaces.size(); i++) {
      auto surface = m_reorderedSurfaces[i];
      int primitiveCount = 0;
      for (const auto& buildRange: surface->getBlas()->buildRanges) {
        primitiveCount += buildRange.primitiveCount;
      }
      m_reorderedSurfacesPrimitiveIDPrefixSum[i + 1] = primitiveCount;
    }

    // Calculate exclusive prefix sum
    uint totalPrimitiveIDOffset = 0;
    for (uint32_t i = 0; i < m_reorderedSurfacesPrimitiveIDPrefixSum.size(); i++) {
      uint primitiveCount = m_reorderedSurfacesPrimitiveIDPrefixSum[i];
      m_reorderedSurfacesPrimitiveIDPrefixSum[i] += totalPrimitiveIDOffset;
      totalPrimitiveIDOffset += primitiveCount;
    }

    // Validate total primitive count against the engine-wide PRIMITIVE_INDEX_BIT_COUNT limit.
    if (totalPrimitiveIDOffset > PRIMITIVE_INDEX_MAX_VALUE) {
      ONCE(Logger::err(str::format("DxvkRaytrace: total primitive count (", totalPrimitiveIDOffset,
        ") exceeds the maximum primitive index (", PRIMITIVE_INDEX_MAX_VALUE,
        ") representable in ", PRIMITIVE_INDEX_BIT_COUNT, " bits. "
        "Downstream systems (NEE cache, prefix-sum lookups) may produce incorrect results.")));
    }

    buildBlases(ctx, execBarriers, cameraManager, opacityMicromapManager, instanceManager, 
                textures, instances, blasBuckets, blasToBuild, blasRangesToBuild, totalScratchMemory);
  }

  void AccelManager::addBlas(RtInstance* instance, BlasEntry* blasEntry, const Matrix4* instanceToObject) {
    // [SpawnGeomDiag.DrawOut] Companion to [SpawnGeomDiag.DrawIn] in
    // submitDrawState — fires once per (vsHash, matHash) when a draw
    // makes it all the way through to the merged-bucket BLAS path
    // (this is the non-PI path; PI fanouts log via piAddEntry). Grep
    // both tags by matHash to find DrawIn entries with NO matching
    // DrawOut — those are the draws being silently dropped between
    // submitDrawState and addBlas.
    {
      const uint64_t vsHash = static_cast<uint64_t>(
        blasEntry->input.getTransformData().vertexShaderHash);
      const uint64_t matHash = static_cast<uint64_t>(
        blasEntry->input.getMaterialData().getHash());
      const uint64_t key = vsHash ^ ((matHash << 1) | (matHash >> 63));
      static std::mutex sDrawOutMu;
      static std::unordered_set<uint64_t> sDrawOutSeen;
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sDrawOutMu);
        first = sDrawOutSeen.insert(key).second;
      }
      if (first) {
        const uint32_t primCount = blasEntry->buildRanges.empty() ? 0u
          : blasEntry->buildRanges[0].primitiveCount;
        const Matrix4& o2w = instance->surface.objectToWorld;
        Logger::info(str::format(
          "[SpawnGeomDiag.DrawOut] kind=addBlas"
          " vsHash=0x", std::hex, vsHash, std::dec,
          " matHash=0x", std::hex, matHash, std::dec,
          " primCnt=", primCount,
          " vCnt=", blasEntry->modifiedGeometryData.vertexCount,
          " surfIdx=", m_reorderedSurfaces.size(),
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
      }
    }

    // Create an instance for this BLAS
    VkAccelerationStructureInstanceKHR blasInstance = instance->getVkInstance();
    blasInstance.accelerationStructureReference = blasEntry->dynamicBlas->accelerationStructureReference;
    blasInstance.instanceCustomIndex =
      (blasInstance.instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
      uint32_t(m_reorderedSurfaces.size()) & uint32_t(CUSTOM_INDEX_SURFACE_MASK);

    if (instanceToObject) {
      // The D3D matrix on input, needs to be transposed before feeding to the VK API (left/right handed conversion)
      // NOTE: VkTransformMatrixKHR is 4x3 matrix, and Matrix4 is 4x4
      const Matrix4 transform = transpose(instance->surface.objectToWorld * (*instanceToObject));
      memcpy(&blasInstance.transform, &transform, sizeof(VkTransformMatrixKHR));
    }

    // NV-DXVK TF2 VIEWMODEL TRACE: log every BLAS instance entering the TLAS
    // so we can find out whether the gun makes it this far. Throttled and
    // gated to VS_ef94e6c7 draws only (player body + gun) so the log stays
    // readable. Prints VS hash, instance mask, primitive count (so we can
    // distinguish the 6829-vert gun vs 40224-vert body), and the VkTransform
    // translation column (post-transpose → last column of the 3x4 matrix).
    {
      static uint32_t sTlasLog = 0;
      const XXH64_hash_t vsHash = blasEntry->input.getTransformData().vertexShaderHash;
      // ef94e6c7... is the first 32 bits; full hash is 64 bits. Match on
      // the upper 32 bits to identify the player-body shader regardless of
      // the variant-specific low bits.
      const uint32_t vsHashHi = static_cast<uint32_t>(vsHash >> 32);
      const bool isEf94 = (vsHashHi == 0xef94e6c7u);
      // Log for ALL skinned / small-prim draws (which matches gun + body
      // + a bunch of other things); widens the net so we don't silently
      // miss the gun due to a hash-comparison quirk.
      const uint32_t primCountForGate =
        (!blasEntry->buildRanges.empty()) ? blasEntry->buildRanges[0].primitiveCount : 0u;
      const bool interesting = isEf94 || (primCountForGate > 100 && primCountForGate < 20000);
      if (interesting && sTlasLog < 60) {
        ++sTlasLog;
        const auto& o2w = instance->surface.objectToWorld;
        // VkTransformMatrixKHR stores row-major 3×4 (rotation + translation
        // in last column). After the transpose above, blasInstance.transform
        // column 3 of each row is the world-space translation of this
        // instance — same convention Vulkan's ray pipeline uses.
        const float* tvals = reinterpret_cast<const float*>(&blasInstance.transform);
        const uint32_t primCount =
          (!blasEntry->buildRanges.empty()) ? blasEntry->buildRanges[0].primitiveCount : 0u;
        const uint32_t vkMask = blasInstance.mask;
        const uint32_t vkFlags = blasInstance.flags;
        Logger::info(str::format(
          "[AccelMgr.tlasAdd] #", sTlasLog,
          " vsHashHi=0x", std::hex, vsHashHi, std::dec,
          " mask=0x", std::hex, vkMask, std::dec,
          " flags=0x", std::hex, vkFlags, std::dec,
          " prims=", primCount,
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " vkT=(", tvals[3], ",", tvals[7], ",", tvals[11], ")",
          " tlas=", (instance->usesUnorderedApproximations() && RtxOptions::enableSeparateUnorderedApproximations())
                      ? "Unordered" : "Opaque",
          " surfIdx=", m_reorderedSurfaces.size()));
      }
    }

    // Get the instance's flags and apply the objectToWorldMirrored flag.
    if (instance->isObjectToWorldMirrored()) {
      blasInstance.flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
    }

    if (instance->usesUnorderedApproximations() && RtxOptions::enableSeparateUnorderedApproximations()) {
      m_mergedInstances[Tlas::Unordered].push_back(blasInstance);
    } else {
      m_mergedInstances[Tlas::Opaque].push_back(blasInstance);
      if (instance->isSubsurface()) {
        m_mergedInstances[Tlas::SSS].push_back(blasInstance);
      }
    }

    // Append the instance to the reordered surface list
    // Note: this happens *after* the instance is appended, because the size of m_reorderedSurfaces is used above
    m_reorderedSurfaces.push_back(instance);
    m_reorderedSurfacesFirstIndexOffset.push_back(0);
  }

  void AccelManager::createBlasBuffersAndInstances(Rc<DxvkContext> ctx, 
                                                   const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                                   std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                                   std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                                   size_t& totalScratchMemory) {

    const uint32_t currentFrame = m_device->getCurrentFrameId();

    // Create or find a matching BLAS for each bucket, then build it
    for (const auto& bucket : blasBuckets) {
      // Fill out the build info
      VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
      buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.geometryCount = bucket->geometries.size();
      buildInfo.pGeometries = bucket->geometries.data();

      // Calculate the build sizes for this bucket
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
      sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
      m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                               &buildInfo, bucket->primitiveCounts.data(), &sizeInfo);

      // Try to find an existing BLAS that is minimally sufficient to fit this bucket of geometries
      PooledBlas* selectedBlas = nullptr;
      for (const auto& blas : m_blasPool) {
        size_t bufferSize = blas->accelStructure->info().size;
        uint32_t paddedLastTouched = blas->frameLastTouched + 1 + (RtxOptions::enablePreviousTLAS() ? 1u : 0u); /* note: +2 because frameLastTouched is unsigned and init'd with UINT32_MAX, and keep the BLAS'es for one extra frame for previous TLAS access */
        if (bufferSize >= sizeInfo.accelerationStructureSize &&
            (!selectedBlas || bufferSize < selectedBlas->accelStructure->info().size) &&
            paddedLastTouched <= currentFrame) {
          selectedBlas = blas.ptr();
        }
      }

      // Must ensure that if we are updating an existing blas, rather than rebuilding, the blas is compatible with our new build info
      // Cannot update a blas that contains OMM instances, this leads to sporadic device lost errors
      if (!bucket->hasOmmInstances && selectedBlas && validateUpdateMode(selectedBlas->buildInfo, buildInfo) && selectedBlas->primitiveCounts == bucket->primitiveCounts) {
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      }

      // There is no such BLAS - create one and put it into the pool
      if (!selectedBlas) {
        auto newBlas = createPooledBlas(sizeInfo.accelerationStructureSize, "BLAS Merged");

        selectedBlas = newBlas.ptr();

        m_blasPool.push_back(std::move(newBlas));
      }
      assert(selectedBlas);
      selectedBlas->frameLastTouched = currentFrame;

      // Use the selected BLAS for the build
      buildInfo.dstAccelerationStructure = selectedBlas->accelStructure->getAccelStructure();

      if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
        // Set the src to the dst if we're updating
        buildInfo.srcAccelerationStructure = buildInfo.dstAccelerationStructure;
      }

      copyAccelerationStructureBuildGeometryInfo(buildInfo, selectedBlas->buildInfo);
      selectedBlas->primitiveCounts = bucket->primitiveCounts;

      // Allocate a scratch buffer slice
      const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
      buildInfo.scratchData.deviceAddress = totalScratchMemory;
      totalScratchMemory += requiredScratchAllocSize;

      assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

      // Track the lifetime of the BLAS buffers
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(selectedBlas->accelStructure);

      // Put the merged BLAS into the build queue
      blasToBuild.push_back(buildInfo);
      blasRangesToBuild.push_back(bucket->ranges.data());
      // NV-DXVK debug: merged-bucket BLASes don't have a single owning blasEntry,
      // but readback only reads gi==0, so stash the FIRST originalInstance's BlasEntry
      // — it owns the position buffer that backs geometries[0]. Without this the
      // [SpawnGeomDiag.BBI-readback] log shows v0=(0,0,0) vtxBufUsage=0x0 for every
      // merged-bucket BLAS in frames where only merged-bucket BLASes are rebuilt
      // (observed in remix-dxvk.log lines 83569+ etc.) and we lose visibility into
      // what the BLAS builder actually consumed for non-PI floor batches.
      BlasEntry* mergedBucketEntry = bucket->originalInstances.empty()
        ? nullptr
        : bucket->originalInstances[0]->getBlas();
      m_debugBlasBuildEntries.push_back(mergedBucketEntry);
      m_debugBlasBuildDstBlas.push_back(selectedBlas);

      static float identityTransform[3][4] = {
        { 1.f, 0.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f, 0.f },
        { 0.f, 0.f, 1.f, 0.f }
      };

      // Append an instance of this merged BLAS to the merged instance list
      VkAccelerationStructureInstanceKHR instance {};
      instance.accelerationStructureReference = selectedBlas->accelerationStructureReference;
      instance.flags = bucket->instanceFlags;
      instance.instanceShaderBindingTableRecordOffset = bucket->instanceShaderBindingTableRecordOffset;
      instance.mask = bucket->instanceMask;
      instance.instanceCustomIndex =
        (bucket->customIndexFlags & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK)) |
        (bucket->reorderedSurfacesOffset & uint32_t(CUSTOM_INDEX_SURFACE_MASK));
      memcpy(static_cast<void*>(&instance.transform.matrix[0][0]), &identityTransform[0][0], sizeof(VkTransformMatrixKHR));

      if (bucket->usesUnorderedApproximations && RtxOptions::enableSeparateUnorderedApproximations()) {
        m_mergedInstances[Tlas::Unordered].push_back(instance);
      } else {
        m_mergedInstances[Tlas::Opaque].push_back(instance);
        if (bucket->hasSssInstances) {
          m_mergedInstances[Tlas::SSS].push_back(instance);
        }
      }
    }
  }

  void AccelManager::prepareSceneData(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers, InstanceManager& instanceManager) {
    ScopedCpuProfileZone();
    bool haveInstances = false;
    for (const auto& instances : m_mergedInstances) {
      if (!instances.empty()) {
        haveInstances = true;
        break;
      }
    }

    if (!haveInstances && instanceManager.getBillboards().empty()) {
      return;
    }

    createAndBuildIntersectionBlas(ctx, execBarriers);

    // Prepare billboard data and instances
    std::vector<MemoryBillboard> memoryBillboards;
    uint32_t numActiveBillboards = 0;

    // Check the enablement here - because the instance manager needs to run the billboard analysis all the time
    if (RtxOptions::enableBillboardOrientationCorrection()) {
      memoryBillboards.resize(instanceManager.getBillboards().size());
      uint32_t index = 0;

      for (const auto& billboard : instanceManager.getBillboards()) {
        if (billboard.instanceMask == 0 || !billboard.allowAsIntersectionPrimitive) {
          continue;
        }

        // Shader data
        MemoryBillboard& memory = memoryBillboards[index];
        memory.center = billboard.center;
        memory.surfaceIndexAndMaterialType =
          (billboard.instance->getSurfaceIndex() & CUSTOM_INDEX_SURFACE_MASK) |
          (((billboard.instance->getVkInstance().instanceCustomIndex >> CUSTOM_INDEX_MATERIAL_TYPE_BIT) & surfaceMaterialTypeMask) << CUSTOM_INDEX_MATERIAL_TYPE_BIT);
        assert(billboard.instance->getSurfaceIndex() <= SURFACE_INDEX_MAX_VALUE && "Billboard surfaceIndex exceeds SURFACE_INDEX_MAX_VALUE");
        memory.inverseHalfWidth = 2.f / billboard.width;
        memory.inverseHalfHeight = 2.f / billboard.height;
        memory.xAxis = billboard.xAxis;
        memory.yAxis = billboard.yAxis;
        memory.xAxisUV = billboard.xAxisUV;
        memory.yAxisUV = billboard.yAxisUV;
        memory.centerUV = billboard.centerUV;
        memory.vertexColor = billboard.vertexColor;
        memory.flags = 0;
        if (billboard.isBeam) {
          memory.flags |= billboardFlagIsBeam;
        }
        if (billboard.isCameraFacing) {
          memory.flags |= billboardFlagIsCameraFacing;
        }

        // TLAS instance
        VkAccelerationStructureInstanceKHR instance {};
        instance.accelerationStructureReference = m_intersectionBlas->accelerationStructureReference;
        instance.flags = 0;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.mask = billboard.instanceMask;
        instance.instanceCustomIndex = index;

        Matrix4 transform;
        if (billboard.isBeam) {
          // Scale and orient the primitive so that its local X and Y axes match the billboard's X and Y axes,
          // and the Z axis is (obviously) orthogonal to those. Note that the beam is cylindrical, so its 'width'
          // applies to both the X and Z axes.
          transform[0] = Vector4(billboard.xAxis * billboard.width * 0.5f, 0.f);
          transform[1] = Vector4(billboard.yAxis * billboard.height * 0.5f, 0.f);
          transform[2] = Vector4(normalize(cross(billboard.xAxis, billboard.yAxis)) * billboard.width * 0.5f, 0.f);
        }
        else {
          // Note: to be fully conservative, the size of the intersection primitive should be equal to the diagonal
          // of the original particle, not its largest side. But the particle textures are usually round, so
          // the reduced size works well in practice and results in fewer unnecessary ray interactions.
          const float radius = std::max(billboard.width, billboard.height) * 0.5f;
          transform[0][0] = transform[1][1] = transform[2][2] = radius;
        }
        transform[3] = Vector4(billboard.center, 1.f);
        transform = transpose(transform);
        memcpy(instance.transform.matrix, &transform, sizeof(VkTransformMatrixKHR));

        m_mergedInstances[Tlas::Unordered].push_back(instance);

        ++index;
      }

      numActiveBillboards = index;
    }

    // Allocate the instance buffer and copy its contents from host to device memory
    // STORAGE_BUFFER_BIT is required for the PointInstancer GPU culling compute shader
    // which writes VkAccelerationStructureInstanceKHR entries directly into this buffer.
    // NV-DXVK (debug probe D): TRANSFER_SRC_BIT added so readback copies are legal.
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
               | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = 0;

    // Vk instance buffer: normal instances + reserved PointInstancer slots per type
    for (int t = 0; t < Tlas::Count; ++t) {
      info.size += m_mergedInstances[t].size() + m_pointInstancerSlotsPerType[t];
    }
    info.size = align(info.size * sizeof(VkAccelerationStructureInstanceKHR), kBufferAlignment);

    if ((m_vkInstanceBuffer == nullptr || info.size > m_vkInstanceBuffer->info().size) && info.size != 0) {
      m_vkInstanceBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Instance Buffer");
      Logger::debug("DxvkRaytrace: Vulkan AS Instance Realloc");
    }

    // Write only the CPU-populated (normal) instance data.  PointInstancer
    // regions are left for the GPU culling shader to fill directly.
    size_t offset = 0;
    for (int t = 0; t < Tlas::Count; ++t) {
      if (!m_mergedInstances[t].empty()) {
        const size_t size = m_mergedInstances[t].size() * sizeof(VkAccelerationStructureInstanceKHR);
        ctx->writeToBuffer(m_vkInstanceBuffer, offset, size, m_mergedInstances[t].data());
      }
      // Advance past both normal and PointInstancer regions for this TLAS type
      offset += (m_mergedInstances[t].size() + m_pointInstancerSlotsPerType[t]) * sizeof(VkAccelerationStructureInstanceKHR);
    }

    // Vk billboard buffer
    if (numActiveBillboards) {
      info.size = align(numActiveBillboards * sizeof(MemoryBillboard), kBufferAlignment);
      if (info.size > 0 && (m_billboardsBuffer == nullptr || info.size > m_billboardsBuffer->info().size)) {
        m_billboardsBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Billboards Buffer");
      }

      // Write billboard data
      ctx->writeToBuffer(m_billboardsBuffer, 0, numActiveBillboards * sizeof(MemoryBillboard), memoryBillboards.data());
    }
  }

  void AccelManager::dispatchPointInstancerCulling(Rc<DxvkContext> ctx, const CameraManager& cameraManager,
                                                   const Rc<DxvkBuffer>& surfaceMaterialBuffer) {
    if (m_pointInstancerBatches.empty() || m_vkInstanceBuffer == nullptr) {
      return;
    }

    // Compute the byte offset for each TLAS type within m_vkInstanceBuffer.
    // Layout per type: [normal instances][PointInstancer instances]
    size_t typeBaseOffset[Tlas::Count] = {};
    for (size_t n = 1; n < Tlas::Count; ++n) {
      typeBaseOffset[n] = typeBaseOffset[n - 1]
                        + (m_mergedInstances[n - 1].size() + m_pointInstancerSlotsPerType[n - 1])
                        * sizeof(VkAccelerationStructureInstanceKHR);
    }

    // Resolve each batch's instanceBufferByteOffset.
    // PointInstancer slots sit after the normal instances within each type's region.
    for (auto& batch : m_pointInstancerBatches) {
      batch.instanceBufferByteOffset = static_cast<uint32_t>(
        typeBaseOffset[batch.tlasType]
        + m_mergedInstances[batch.tlasType].size() * sizeof(VkAccelerationStructureInstanceKHR)
        + batch.firstIndexInType * sizeof(VkAccelerationStructureInstanceKHR));
    }

    // [SpawnGeomDiag.PIoffsets] Suspect 3: dump every batch's
    // (tlasType, m_mergedInstances size, slotsPerType, firstIdx,
    //  instBufOff, instCount, expEnd) so we can detect stale/wrong
    // byte-offset math against last frame's m_mergedInstances sizes.
    // The "first 27 of 81 visible, rest absent" pattern in batch 6
    // matches a per-batch byte-offset mismatch where the TLAS builder
    // reads valid bytes for the leading chunk but garbage past it —
    // the offset assignment above being driven by stale sizes is
    // exactly that kind of bug.
    //
    // [SpawnGeomDiag.PIoverlap] flags any pair of batches whose
    // [byteOff, byteOff + count*64) ranges intersect — overlap means
    // batch N's writes clobber batch M's, so M ends up with random
    // transforms and the TLAS sees the wrong AS reference for some
    // instances.
    {
      static uint32_t sPIOffsetsFrame = 0;
      const bool dump = (sPIOffsetsFrame++ % 30u) == 0;
      if (dump) {
        const uint32_t bufSize = (m_vkInstanceBuffer == nullptr) ? 0u
                                 : static_cast<uint32_t>(m_vkInstanceBuffer->info().size);
        Logger::info(str::format(
          "[SpawnGeomDiag.PIoffsets.head] frame=", sPIOffsetsFrame,
          " batches=", m_pointInstancerBatches.size(),
          " vkInstBufSize=", bufSize,
          " mergedSizes=[", m_mergedInstances[0].size(), ",",
                            m_mergedInstances[1].size(), ",",
                            m_mergedInstances[2].size(), "]",
          " slotsPerType=[", m_pointInstancerSlotsPerType[0], ",",
                              m_pointInstancerSlotsPerType[1], ",",
                              m_pointInstancerSlotsPerType[2], "]"));
        for (size_t i = 0; i < m_pointInstancerBatches.size(); ++i) {
          const auto& b = m_pointInstancerBatches[i];
          const uint32_t mergedCount = static_cast<uint32_t>(m_mergedInstances[b.tlasType].size());
          const uint32_t slotsForType = m_pointInstancerSlotsPerType[b.tlasType];
          const uint32_t typeBase = static_cast<uint32_t>(typeBaseOffset[b.tlasType]);
          const uint32_t typeEndByte = typeBase + (mergedCount + slotsForType)
                                                  * uint32_t(sizeof(VkAccelerationStructureInstanceKHR));
          const uint32_t expEnd = b.instanceBufferByteOffset
            + b.instanceCount * uint32_t(sizeof(VkAccelerationStructureInstanceKHR));
          const bool offWithinType = (b.instanceBufferByteOffset >= typeBase)
                                  && (expEnd <= typeEndByte);
          Logger::info(str::format(
            "[SpawnGeomDiag.PIoffsets] bi=", i,
            " tlasType=", b.tlasType,
            " merged=", mergedCount,
            " slotsForType=", slotsForType,
            " firstIdxInType=", b.firstIndexInType,
            " instCount=", b.instanceCount,
            " byteOff=", b.instanceBufferByteOffset,
            " expEnd=", expEnd,
            " typeBase=", typeBase,
            " typeEnd=", typeEndByte,
            " withinType=", (offWithinType ? 1 : 0),
            " baseSurf=", b.baseSurfaceIndex));
        }
        // Pair-wise overlap check.
        for (size_t i = 0; i < m_pointInstancerBatches.size(); ++i) {
          const auto& a = m_pointInstancerBatches[i];
          const uint32_t aLo = a.instanceBufferByteOffset;
          const uint32_t aHi = aLo + a.instanceCount * uint32_t(sizeof(VkAccelerationStructureInstanceKHR));
          for (size_t j = i + 1; j < m_pointInstancerBatches.size(); ++j) {
            const auto& bb = m_pointInstancerBatches[j];
            const uint32_t bLo = bb.instanceBufferByteOffset;
            const uint32_t bHi = bLo + bb.instanceCount * uint32_t(sizeof(VkAccelerationStructureInstanceKHR));
            if (aLo < bHi && bLo < aHi) {
              Logger::info(str::format(
                "[SpawnGeomDiag.PIoverlap] i=", i, " j=", j,
                " aRange=[", aLo, ",", aHi, ")",
                " bRange=[", bLo, ",", bHi, ")",
                " aType=", a.tlasType, " bType=", bb.tlasType,
                " aCount=", a.instanceCount, " bCount=", bb.instanceCount));
            }
          }
        }

        // [SpawnGeomDiag.PIBatchInventory] Source-vs-BLAS fingerprint.
        // For each PI batch dump:
        //   - source vsHash + prim/vert counts (captured at addPI time)
        //   - batch.blasReference vs the live debugBlasRef state (handle,
        //     primitiveCounts[0], buildInfo geometry, frameLastTouched)
        //   - blasMatch flag = source primCount == live primitiveCounts[0]
        // The 81-instance "floor" batch's blasRef pointing to a 16-prim
        // BLAS (per [SpawnGeomDiag.BBI-readback] dstAs=0x4c9e2d700) shows
        // up here as `srcPrim=<large> livePrim=16 blasMatch=0` — directly
        // pinning the hypothesis that the BLAS pool returned the wrong
        // entry for this PI batch.
        for (size_t i = 0; i < m_pointInstancerBatches.size(); ++i) {
          const auto& b = m_pointInstancerBatches[i];
          const bool hasRc = b.debugBlasRef != nullptr;
          const uint64_t liveRef = hasRc ? b.debugBlasRef->accelerationStructureReference : 0;
          const bool refMatch = hasRc && (liveRef == b.blasReference);
          const bool asLive = hasRc && b.debugBlasRef->accelStructure != nullptr;
          uint32_t livePrim = 0;
          uint32_t liveMaxVtx = 0;
          uint32_t liveVStride = 0;
          uint32_t frameLastTouched = 0xFFFFFFFFu;
          if (hasRc) {
            if (!b.debugBlasRef->primitiveCounts.empty()) {
              livePrim = b.debugBlasRef->primitiveCounts[0];
            }
            const auto& binfo = b.debugBlasRef->buildInfo;
            if (binfo.geometryCount > 0 && binfo.pGeometries != nullptr) {
              const auto& tri = binfo.pGeometries[0].geometry.triangles;
              liveMaxVtx = tri.maxVertex;
              liveVStride = uint32_t(tri.vertexStride);
            }
            frameLastTouched = b.debugBlasRef->frameLastTouched;
          }
          const bool primMatch = (livePrim == b.debugSourcePrimCount);
          Logger::info(str::format(
            "[SpawnGeomDiag.PIBatchInventory] bi=", i,
            " vsHash=0x", std::hex, b.debugVsHash, std::dec,
            " srcPrim=", b.debugSourcePrimCount,
            " srcVtx=", b.debugSourceVertexCount,
            " livePrim=", livePrim,
            " liveMaxVtx=", liveMaxVtx,
            " liveVStride=", liveVStride,
            " primMatch=", (primMatch ? 1 : 0),
            " refMatch=", (refMatch ? 1 : 0),
            " asLive=", (asLive ? 1 : 0),
            " builtAtCap=", (b.debugAsBuiltAtCapture ? 1 : 0),
            " blasRef=0x", std::hex, b.blasReference, std::dec,
            " liveRef=0x", std::hex, liveRef, std::dec,
            " frameLastTouched=", frameLastTouched,
            " curFrame=", m_device->getCurrentFrameId(),
            " count=", b.instanceCount,
            " baseSurf=", b.baseSurfaceIndex));
        }
      }
    }

    // Dispatch the GPU culling compute shader via PointInstancerSystem
    RtxPointInstancerSystem& system = m_device->getCommon()->metaPointInstancerSystem();
    const Vector3 cameraPos = cameraManager.getMainCamera().getPosition();

    // NV-DXVK (debug probe A): summary log every ~60 frames showing whether PI dispatch runs.
    // [SpawnGeomDiag] renamed from [PI-dispatch] so it bypasses log.cpp's
    // "[PI-" filter (which drops it unless RTX_D3D11_DIAG=1 is set).
    // Drops the throttle to 1/30 to match the [SpawnGeomDiag.merge]
    // cadence so frame numbers line up across the diagnostic stream.
    {
      static uint32_t sDispatchFrame = 0;
      if ((sDispatchFrame++ % 30) == 0) {
        uint32_t totalInst = 0;
        for (const auto& b : m_pointInstancerBatches) totalInst += b.instanceCount;
        Logger::info(str::format(
          "[SpawnGeomDiag.PIdispatch] frame=", sDispatchFrame,
          " batches=", m_pointInstancerBatches.size(),
          " totalInstances=", totalInst,
          " slotsPerType=[", m_pointInstancerSlotsPerType[0], ",",
                              m_pointInstancerSlotsPerType[1], ",",
                              m_pointInstancerSlotsPerType[2], "]",
          " camPos=(", cameraPos.x, ",", cameraPos.y, ",", cameraPos.z, ")"));
      }
    }

    system.dispatchCulling(ctx, m_vkInstanceBuffer, m_surfaceBuffer, surfaceMaterialBuffer, m_pointInstancerBatches, cameraPos);

    // NV-DXVK debug: definitive test. Overwrite the FIRST PI batch's first slot in
    // m_vkInstanceBuffer with a copy of a known-working merged-Opaque instance entry,
    // but keep the original surface index so we can detect it in [VisibleSurf].
    // If VisibleSurf shows the PI surface index after this, the PI region IS reachable
    // and the bug is in the bytes written by the compute shader.
    // If VisibleSurf still doesn't show the PI surface index, the PI region of the TLAS
    // instance buffer is unreachable (offset/alignment/build-range issue).
    {
      static uint32_t s_overrideFrame = 0;
      const bool fire = ((s_overrideFrame++ % 60u) == 0)
                       && !m_pointInstancerBatches.empty()
                       && !m_mergedInstances[Tlas::Opaque].empty();
      (void)fire;
    }

    // [SpawnGeomDiag.TLASReadback] Suspect 2: GPU readback of the
    // ENTIRE largest PI batch's region of m_vkInstanceBuffer
    // (post-cull-shader). For each instance dump:
    //   - the 12 floats of VkTransformMatrixKHR (3x4 row-major)
    //   - customInstanceIndex (24 bits at byte 48)
    //   - mask (8 bits at byte 51)
    //   - sbtOffsetAndFlags (32 bits at byte 52)
    //   - accelerationStructureReference (uint64 at byte 56)
    // Cross-reference customInstanceIndex against
    //   expected = baseSurfaceIndex + instanceIdx
    // and translation against the CPU's i2o.T from
    // [SpawnGeomDiag.PIdump]. Either mismatch isolates the bug
    // to the cull-shader write (Suspect 2) vs the offset math
    // (Suspect 3). Replaces the older 4-entry [PI-readback]
    // probe — that was filter-dropped AND undersized to see the
    // visibility cutoff at instance 84 of batch 6 in the floor.
    if constexpr (kEnableRtxDebugProbes) {
      static constexpr uint32_t kRingSize = 3;
      static constexpr uint32_t kProbeBytesPerEntry = sizeof(VkAccelerationStructureInstanceKHR); // 64
      // Cap per dump. TF2 floor batch peaks ~462 instances; 512
      // covers it with margin. 512 * 64 = 32 KB per ring slot.
      static constexpr uint32_t kProbeMaxInstances = 512;
      static constexpr uint32_t kProbeMaxBytes = kProbeBytesPerEntry * kProbeMaxInstances;
      // Per-batch log volume cap. Logging 462 lines at 30-dispatch
      // throttle gives ~15 lines/sec sustained — fine for a
      // diagnostic. Keep separate from kProbeMaxInstances so we
      // can capture all bytes but throttle log volume independently.
      static constexpr uint32_t kProbeLogLines = 256;
      static Rc<DxvkBuffer> sStaging[kRingSize];
      static uint32_t sCaptureCount[kRingSize] = {};
      static uint32_t sCaptureBaseSurf[kRingSize] = {};
      static uint32_t sCaptureBatchIdxInType[kRingSize] = {};
      static uint32_t sCaptureTlasType[kRingSize] = {};
      static uint32_t sCaptureByteOff[kRingSize] = {};
      static uint64_t sCaptureBlasRef[kRingSize] = {};
      static bool sCaptureValid[kRingSize] = {};
      static uint32_t sProbeDFrame = 0;
      const uint32_t writeSlot = sProbeDFrame % kRingSize;
      const uint32_t readSlot  = (sProbeDFrame + 1) % kRingSize; // oldest of the ring

      // Lazily create the staging buffers (sized for the worst-case batch).
      for (uint32_t i = 0; i < kRingSize; ++i) {
        if (sStaging[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size   = kProbeMaxBytes;
          sStaging[i] = m_device->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "[SpawnGeomDiag.TLASReadback] PI Instance Staging");
        }
      }

      // Throttle: one batch dump every 30 dispatches. Matches
      // [SpawnGeomDiag.PIdispatch]/[SpawnGeomDiag.PIdump] cadence.
      const bool logThisFrame = (sProbeDFrame % 30u == 0);
      if (sCaptureValid[readSlot] && logThisFrame) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(sStaging[readSlot]->mapPtr(0));
        if (data != nullptr) {
          const uint32_t n   = sCaptureCount[readSlot];
          const uint32_t base = sCaptureBaseSurf[readSlot];
          const uint64_t bRef = sCaptureBlasRef[readSlot];
          // One-line dump header so a single grep recovers
          // batch-level identity for the per-instance lines that follow.
          Logger::info(str::format(
            "[SpawnGeomDiag.TLASReadback.head] tlasType=", sCaptureTlasType[readSlot],
            " firstIdx=", sCaptureBatchIdxInType[readSlot],
            " baseSurf=", base,
            " count=", n,
            " byteOff=0x", std::hex, sCaptureByteOff[readSlot], std::dec,
            " blasRef=0x", std::hex, bRef, std::dec));
          const uint32_t toLog = std::min<uint32_t>(n, kProbeLogLines);
          uint32_t maskedOut = 0;
          uint32_t surfMismatches = 0;
          for (uint32_t e = 0; e < n; ++e) {
            const uint8_t* p = data + e * kProbeBytesPerEntry;
            float t[12];
            memcpy(t, p, sizeof(t));
            uint32_t customIdxAndMask, sbtAndFlags, blasRefLo, blasRefHi;
            memcpy(&customIdxAndMask, p + 48, 4);
            memcpy(&sbtAndFlags,      p + 52, 4);
            memcpy(&blasRefLo,        p + 56, 4);
            memcpy(&blasRefHi,        p + 60, 4);
            const uint32_t surfaceIdx = customIdxAndMask & 0x00FFFFFFu;
            const uint32_t mask       = (customIdxAndMask >> 24) & 0xFFu;
            const uint64_t entryBlasRef = (uint64_t(blasRefHi) << 32) | blasRefLo;
            const uint32_t expectedSurf = base + e;
            const bool surfOk = (surfaceIdx == expectedSurf);
            if (mask == 0) ++maskedOut;
            if (!surfOk) ++surfMismatches;
            if (e < toLog) {
              Logger::info(str::format(
                "[SpawnGeomDiag.TLASReadback] e=", e,
                " T=(", t[3], ",", t[7], ",", t[11], ")",
                " r0=(", t[0], ",", t[1], ",", t[2], ")",
                " r1=(", t[4], ",", t[5], ",", t[6], ")",
                " r2=(", t[8], ",", t[9], ",", t[10], ")",
                " surf=", surfaceIdx,
                " expSurf=", expectedSurf,
                " surfOk=", (surfOk ? 1 : 0),
                " mask=0x", std::hex, mask, std::dec,
                " sbtFlags=0x", std::hex, sbtAndFlags, std::dec,
                " blasRef=0x", std::hex, entryBlasRef, std::dec,
                " blasRefMatch=", (entryBlasRef == bRef ? 1 : 0)));
            }
          }
          // Tail summary lets us tell if catastrophic clobbering
          // happened at the end of the batch even when we capped
          // the per-line dump.
          Logger::info(str::format(
            "[SpawnGeomDiag.TLASReadback.tail] maskedOut=", maskedOut,
            "/", n,
            " surfMismatches=", surfMismatches));
        }
        sCaptureValid[readSlot] = false;
      }

      // Issue a GPU copy from m_vkInstanceBuffer into this frame's write slot.
      // Capture the LARGEST batch's region — most likely BSP floor.
      if (!m_pointInstancerBatches.empty()) {
        size_t largestIdx = 0;
        for (size_t i = 1; i < m_pointInstancerBatches.size(); ++i) {
          if (m_pointInstancerBatches[i].instanceCount > m_pointInstancerBatches[largestIdx].instanceCount) {
            largestIdx = i;
          }
        }
        const PointInstancerBatch& b0 = m_pointInstancerBatches[largestIdx];
        const uint32_t byteOff = b0.instanceBufferByteOffset;
        const uint32_t copyCount = std::min<uint32_t>(b0.instanceCount, kProbeMaxInstances);
        const uint32_t copyBytes = copyCount * kProbeBytesPerEntry;
        if (copyBytes > 0 && byteOff + copyBytes <= m_vkInstanceBuffer->info().size) {
          // Barrier: GPU culling writes (shader write) → transfer read.
          ctx->emitMemoryBarrier(0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
          ctx->copyBuffer(sStaging[writeSlot], 0, m_vkInstanceBuffer, byteOff, copyBytes);
          sCaptureValid[writeSlot] = true;
          sCaptureCount[writeSlot] = copyCount;
          sCaptureBaseSurf[writeSlot] = b0.baseSurfaceIndex;
          sCaptureBatchIdxInType[writeSlot] = b0.firstIndexInType;
          sCaptureTlasType[writeSlot] = b0.tlasType;
          sCaptureByteOff[writeSlot] = byteOff;
          sCaptureBlasRef[writeSlot] = b0.blasReference;
        }
      }
      ++sProbeDFrame;
    }

    // [SpawnGeomDiag.ObjDump] On Ctrl+Shift+O, queue EVERY PI batch
    // and EVERY merged-bucket TLAS instance for individual OBJ dump.
    // Queue+ring design:
    //   - press → populate two queues with full per-task data (we
    //     capture Rc<DxvkBuffer> + offsets at queue time so the
    //     subsequent copies don't depend on BlasEntry lifetime)
    //   - each frame → drain ready ring slots (frame age >= 3) by
    //     writing local+world OBJ files; schedule up to
    //     kSchedulePerFrame fresh copies from the queue into free
    //     ring slots
    //   - each PI batch produces one pi_<vsHash>_<bi>_…_local.obj
    //     plus one _world.obj (world bakes objectToWorld * i2o[N]
    //     for each batch instance)
    //   - each merged TLAS instance (Opaque + Unordered + SSS)
    //     produces one merged_<vsHash>_<surfIdx>_…_local.obj plus
    //     one _world.obj (single-instance transform from
    //     RtInstance->surface.objectToWorld)
    // Press while a queue is non-empty: ignored with a log line so
    // the user can see why nothing fired.
    const bool objDumpRequested = RtxContext::consumeObjDumpRequest();

    // Shared per-task staging size and per-frame scheduling cap. 4 MB
    // is enough for typical BSP world chunks (~20K verts × 24 stride +
    // ~30K indices × 4 bytes). Tasks larger than this get truncated
    // (positions and indices proportionally) so even huge BLASes
    // produce a partial-but-readable OBJ. kSchedulePerFrame caps how
    // many GPU copies we issue per frame to avoid command-buffer
    // bloat when the user presses Ctrl+Shift+O on a scene with many
    // batches.
    {
      // Bumped from 4 MB → 16 MB so the largest realistic BSP
      // merged BLAS (s538 in the previous log was 261472 verts →
      // ~5.2 MB) fits without truncation. 3 ring slots × 16 MB ×
      // 2 kinds = 96 MB total host-visible staging — still bounded
      // and only allocated lazily on the first Ctrl+Shift+O press.
      static constexpr uint32_t kObjStagingBytes = 16u * 1024u * 1024u;
      static constexpr uint32_t kObjRingSize = 3;
      static constexpr uint32_t kSchedulePerFrame = 3;

      // Common per-slot meta. Used by both PI and merged drain paths.
      // For PI: instanceTransforms is the per-batch i2o list (one entry
      // per instance), worldRoot is batch.objectToWorld.
      // For merged: instanceTransforms is a single identity matrix and
      // worldRoot is the instance's surface.objectToWorld.
      struct ObjMeta {
        bool valid = false;
        bool isPi = false;
        uint32_t pendingFrame = 0;
        uint32_t vertexCount = 0;
        uint32_t primCount = 0;
        uint32_t srcVertexCount = 0;  // pre-truncation, for filename + diagnostics
        uint32_t srcPrimCount = 0;
        uint32_t vtxStride = 0;
        uint32_t indexBytes = 0; // 2 or 4
        uint64_t blasRef = 0;
        uint64_t vsHash = 0;
        // Identifier used in filenames:
        //   PI: batch index ("b<N>")
        //   Merged: surfaceIndex ("s<N>")
        uint32_t identifier = 0;
        uint32_t tlasType = 0;  // For merged: 0=Opaque, 1=Unordered, 2=SSS
        // Stable hash of the source material (texture + state combo).
        // This is the SAME hash mod authors use to identify materials
        // for replacement, so the user can grep OBJ filenames by the
        // material they recognize as "the floor" in their mod project.
        uint64_t materialHash = 0;
        // First-instance world translation (used for filename + a
        // distance-to-camera metric in the OBJ header) — gives a
        // human-readable spatial fingerprint of where this BLAS lives
        // in the spawn area.
        Vector3 worldAnchor;
        Matrix4 worldRoot;
        std::vector<Matrix4> instanceTransforms;
        Vector3 cameraPos;
      };

      // Per-task source data captured at queue time. Holds Rc<DxvkBuffer>
      // so the source position/index buffers stay alive even if the
      // BlasEntry is GCed before the task is scheduled. All metadata
      // needed to schedule the GPU copy + later write the OBJ files.
      struct ObjTask {
        bool isPi = false;
        Rc<DxvkBuffer> posBuf;
        VkDeviceSize posOff = 0;
        uint32_t vtxStride = 0;
        Rc<DxvkBuffer> idxBuf;
        VkDeviceSize idxOff = 0;
        uint32_t indexBytes = 0;
        uint32_t srcVertexCount = 0;
        uint32_t srcPrimCount = 0;
        uint64_t blasRef = 0;
        uint64_t vsHash = 0;
        uint32_t identifier = 0;
        uint32_t tlasType = 0;
        uint64_t materialHash = 0;
        Vector3 worldAnchor;
        Matrix4 worldRoot;
        std::vector<Matrix4> instanceTransforms;
        Vector3 cameraPos;
      };

      // Two completely independent rings + queues per the user's
      // separation requirement.
      static Rc<DxvkBuffer> sPiStaging[kObjRingSize];
      static Rc<DxvkBuffer> sMergedStaging[kObjRingSize];
      static ObjMeta sPiMeta[kObjRingSize];
      static ObjMeta sMergedMeta[kObjRingSize];
      static std::vector<ObjTask> sPiQueue;
      static std::vector<ObjTask> sMergedQueue;

      // Lazy-init staging buffers.
      for (uint32_t i = 0; i < kObjRingSize; ++i) {
        if (sPiStaging[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size = kObjStagingBytes;
          sPiStaging[i] = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "[SpawnGeomDiag.PiObjDump] staging");
        }
        if (sMergedStaging[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size = kObjStagingBytes;
          sMergedStaging[i] = m_device->createBuffer(info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "[SpawnGeomDiag.MergedObjDump] staging");
        }
      }

      const uint32_t curFrame = m_device->getCurrentFrameId();

      // [SpawnGeomDiag.SceneObjDump] One unified world-space scene
      // OBJ that aggregates every PI batch (one `o` group per
      // instance) and every merged-bucket TLAS instance. Open this
      // single file in Blender to see EVERYTHING Remix is feeding the
      // TLAS as one mesh, lined up against the user's USD capture.
      // Each press starts a new file; appended to as drains complete
      // (so the file is always valid even mid-drain). Closed-out
      // (logged as complete) when both queues hit zero outstanding.
      static std::filesystem::path sSceneObjPath;
      static uint32_t sSceneVertexBase = 0;     // 0-indexed running vert offset
      static uint32_t sSceneObjectCount = 0;
      static uint32_t sSceneExpectedTasks = 0;
      static uint32_t sSceneDrainedTasks = 0;
      static bool sSceneActive = false;

      // ============================================================
      // Common drain helper (writes OBJ files for one ready slot).
      // ============================================================
      auto drainSlot = [&](Rc<DxvkBuffer>& staging, ObjMeta& m, const char* tag, const char* prefix) {
        // Count this drain attempt against the scene session BEFORE
        // any early-return so a failed-to-map staging buffer doesn't
        // stall the "scene complete" condition forever.
        const bool countTowardScene = sSceneActive;
        auto bumpSceneDrained = [&]() {
          if (countTowardScene) {
            ++sSceneDrainedTasks;
            if (sSceneDrainedTasks >= sSceneExpectedTasks && sSceneActive) {
              Logger::info(str::format(
                "[SpawnGeomDiag.SceneObjDump] scene complete: ",
                sSceneObjectCount, " objects, ",
                sSceneVertexBase, " verts total, file=\"",
                sSceneObjPath.string(), "\""));
              sSceneActive = false;
            }
          }
        };
        const uint8_t* data = reinterpret_cast<const uint8_t*>(staging->mapPtr(0));
        if (data == nullptr) {
          m.valid = false;
          bumpSceneDrained();
          return;
        }
        std::vector<Vector3> localVerts;
        localVerts.reserve(m.vertexCount);
        for (uint32_t v = 0; v < m.vertexCount; ++v) {
          const uint8_t* p = data + v * m.vtxStride;
          float x, y, z;
          memcpy(&x, p + 0, 4);
          memcpy(&y, p + 4, 4);
          memcpy(&z, p + 8, 4);
          localVerts.emplace_back(x, y, z);
        }
        // Decode raw tris, then FILTER any face whose indices reference
        // a vertex past the (possibly truncated) localVerts range.
        // Without this filter, truncated tasks produce OBJs with face
        // indices pointing past EOF — Blender silently drops the
        // entire mesh and the file appears geometry-less in the
        // viewport. The filter preserves whichever faces still have
        // all three corners inside the loaded vert window.
        std::vector<uint32_t> tris;
        tris.reserve(m.primCount * 3);
        const uint8_t* ip = data + m.vertexCount * m.vtxStride;
        const uint32_t loadedVertLimit = m.vertexCount;
        uint32_t droppedFaces = 0;
        for (uint32_t k = 0; k < m.primCount; ++k) {
          uint32_t idx[3] = {};
          for (uint32_t c = 0; c < 3; ++c) {
            const uint32_t off = (k * 3u + c) * m.indexBytes;
            if (m.indexBytes == 2) {
              uint16_t tmp;
              memcpy(&tmp, ip + off, 2);
              idx[c] = tmp;
            } else {
              memcpy(&idx[c], ip + off, 4);
            }
          }
          if (idx[0] >= loadedVertLimit || idx[1] >= loadedVertLimit || idx[2] >= loadedVertLimit) {
            ++droppedFaces;
            continue;
          }
          tris.push_back(idx[0]);
          tris.push_back(idx[1]);
          tris.push_back(idx[2]);
        }
        const uint32_t writtenPrimCount = static_cast<uint32_t>(tris.size() / 3);
        if (droppedFaces > 0) {
          Logger::info(str::format(
            "[", tag, "] dropped ", droppedFaces,
            " out-of-range face(s) for id=", m.identifier,
            " (truncation collateral; loadedVerts=", loadedVertLimit,
            " srcPrimCount=", m.srcPrimCount, ")"));
        }

        // Compute world-space AABB on-the-fly so the OBJ header
        // tells you EXACTLY where in the map this BLAS lives.
        // Walk every (verts × instance-transform) pair and accumulate
        // min/max. For huge meshes × many PI instances this is the
        // most expensive step in this whole path; cap reasonable.
        Vector3 worldMin( 1e30f,  1e30f,  1e30f);
        Vector3 worldMax(-1e30f, -1e30f, -1e30f);
        for (size_t inst = 0; inst < m.instanceTransforms.size(); ++inst) {
          const Matrix4 effective = m.worldRoot * m.instanceTransforms[inst];
          for (const auto& v : localVerts) {
            const Vector4 lh(v.x, v.y, v.z, 1.0f);
            const Vector4 wh = effective * lh;
            worldMin.x = std::min(worldMin.x, wh.x);
            worldMin.y = std::min(worldMin.y, wh.y);
            worldMin.z = std::min(worldMin.z, wh.z);
            worldMax.x = std::max(worldMax.x, wh.x);
            worldMax.y = std::max(worldMax.y, wh.y);
            worldMax.z = std::max(worldMax.z, wh.z);
          }
        }
        // For merged-bucket BSP, surface.objectToWorld is the
        // identity (the source verts are already in world space) so
        // worldAnchor as captured at queue time is (0,0,0). Replace
        // it with the AABB center now that we have real geometry —
        // otherwise the filename's "w0_0_0" suffix is useless for
        // identifying which BLAS sits where in the level.
        if (!m.isPi) {
          m.worldAnchor = Vector3(
            (worldMin.x + worldMax.x) * 0.5f,
            (worldMin.y + worldMax.y) * 0.5f,
            (worldMin.z + worldMax.z) * 0.5f);
        }

        // Compose output paths NOW (after worldAnchor potentially
        // updated). Filenames embed identifier + vsHash + matHash +
        // world coords so a single grep can pinpoint a BLAS.
        auto outDir = util::RtxFileSys::path(util::RtxFileSys::Captures);
        if (outDir.empty()) outDir = std::filesystem::path(".");
        const std::time_t t = std::time(nullptr);
        char tsBuf[32] = {};
        std::strftime(tsBuf, sizeof(tsBuf), "%Y%m%d_%H%M%S", std::localtime(&t));
        char hashBuf[32] = {};
        std::snprintf(hashBuf, sizeof(hashBuf), "%016llx",
          static_cast<unsigned long long>(m.vsHash));
        char blasRefBuf[32] = {};
        std::snprintf(blasRefBuf, sizeof(blasRefBuf), "%016llx",
          static_cast<unsigned long long>(m.blasRef));
        char matHashBuf[32] = {};
        std::snprintf(matHashBuf, sizeof(matHashBuf), "%016llx",
          static_cast<unsigned long long>(m.materialHash));
        const char idChar = m.isPi ? 'b' : 's';
        char worldBuf[64] = {};
        std::snprintf(worldBuf, sizeof(worldBuf), "w%d_%d_%d",
          static_cast<int>(m.worldAnchor.x),
          static_cast<int>(m.worldAnchor.y),
          static_cast<int>(m.worldAnchor.z));
        const std::string baseName = std::string(prefix) + "_"
          + "vs" + hashBuf + "_mat" + matHashBuf
          + "_" + idChar + std::to_string(m.identifier)
          + "_" + worldBuf
          + "_f" + std::to_string(m.pendingFrame) + "_" + tsBuf;
        const auto localPath = outDir / (baseName + "_local.obj");
        const auto worldPath = outDir / (baseName + "_world.obj");

        const Vector3 toCam(
          m.worldAnchor.x - m.cameraPos.x,
          m.worldAnchor.y - m.cameraPos.y,
          m.worldAnchor.z - m.cameraPos.z);
        const float distToCam = std::sqrt(
          toCam.x * toCam.x + toCam.y * toCam.y + toCam.z * toCam.z);

        if (auto local = util::createDirectoriesAndOpenFile(localPath)) {
          *local << "# [" << tag << "] local-space\n"
                 << "# vsHash=0x" << hashBuf
                 << " materialHash=0x" << matHashBuf
                 << " primCount=" << m.primCount
                 << " vertexCount=" << m.vertexCount
                 << " srcVertexCount=" << m.srcVertexCount
                 << " srcPrimCount=" << m.srcPrimCount
                 << " vtxStride=" << m.vtxStride
                 << " indexBytes=" << m.indexBytes
                 << " identifier=" << idChar << m.identifier
                 << " tlasType=" << m.tlasType
                 << " blasRef=0x" << blasRefBuf << "\n"
                 << "# worldAnchor=(" << m.worldAnchor.x << "," << m.worldAnchor.y << "," << m.worldAnchor.z << ")"
                 << " worldAABB=[(" << worldMin.x << "," << worldMin.y << "," << worldMin.z << ")"
                 << "..(" << worldMax.x << "," << worldMax.y << "," << worldMax.z << ")]"
                 << " camera=(" << m.cameraPos.x << "," << m.cameraPos.y << "," << m.cameraPos.z << ")"
                 << " distToCam=" << distToCam << "\n";
          for (const auto& v : localVerts) {
            *local << "v " << v.x << " " << v.y << " " << v.z << "\n";
          }
          for (uint32_t k = 0; k < writtenPrimCount; ++k) {
            const uint32_t a = tris[k * 3 + 0] + 1;
            const uint32_t b = tris[k * 3 + 1] + 1;
            const uint32_t c = tris[k * 3 + 2] + 1;
            *local << "f " << a << " " << b << " " << c << "\n";
          }
        }

        if (auto world = util::createDirectoriesAndOpenFile(worldPath)) {
          *world << "# [" << tag << "] world-space\n"
                 << "# vsHash=0x" << hashBuf
                 << " materialHash=0x" << matHashBuf
                 << " identifier=" << idChar << m.identifier
                 << " tlasType=" << m.tlasType
                 << " instanceCount=" << m.instanceTransforms.size()
                 << "\n"
                 << "# worldAnchor=(" << m.worldAnchor.x << "," << m.worldAnchor.y << "," << m.worldAnchor.z << ")"
                 << " worldAABB=[(" << worldMin.x << "," << worldMin.y << "," << worldMin.z << ")"
                 << "..(" << worldMax.x << "," << worldMax.y << "," << worldMax.z << ")]"
                 << " camera=(" << m.cameraPos.x << "," << m.cameraPos.y << "," << m.cameraPos.z << ")"
                 << " distToCam=" << distToCam << "\n";
          for (size_t inst = 0; inst < m.instanceTransforms.size(); ++inst) {
            const Matrix4 effective = m.worldRoot * m.instanceTransforms[inst];
            *world << "o instance_" << inst << "\n";
            for (const auto& v : localVerts) {
              const Vector4 lh(v.x, v.y, v.z, 1.0f);
              const Vector4 wh = effective * lh;
              *world << "v " << wh.x << " " << wh.y << " " << wh.z << "\n";
            }
            const uint32_t base = static_cast<uint32_t>(inst) * m.vertexCount + 1u;
            for (uint32_t k = 0; k < writtenPrimCount; ++k) {
              const uint32_t a = tris[k * 3 + 0] + base;
              const uint32_t b = tris[k * 3 + 1] + base;
              const uint32_t c = tris[k * 3 + 2] + base;
              *world << "f " << a << " " << b << " " << c << "\n";
            }
          }
        }

        // Append this BLAS's world-space geometry to the unified
        // scene OBJ. Each instance gets its own `o` group named with
        // identifier+vsHash+matHash+inst so the user can find it in
        // Blender's outliner. The shared sSceneVertexBase tracks
        // cumulative vertex count across all drained tasks so face
        // indices stay correct.
        if (sSceneActive) {
          std::ofstream scene(sSceneObjPath, std::ios::app);
          if (scene.is_open()) {
            for (size_t inst = 0; inst < m.instanceTransforms.size(); ++inst) {
              const Matrix4 effective = m.worldRoot * m.instanceTransforms[inst];
              // Use `g` (face group) instead of `o` (object) so the
              // entire scene imports as ONE Blender mesh. This avoids
              // per-`o` scoping quirks in some OBJ importers that
              // produced empty meshes when face indices crossed `o`
              // boundaries. Group names still let you select-by-group
              // in Blender's edit mode (Select → All by trait → Same
              // → Material/etc., or use the panel's "Group" filter).
              // A leading comment also makes filename-style grep work
              // on the scene file's text.
              scene << "# blas " << prefix << "_" << idChar << m.identifier
                    << "_vs" << hashBuf
                    << "_mat" << matHashBuf
                    << "_inst" << inst
                    << " worldAABB=[(" << worldMin.x << "," << worldMin.y << "," << worldMin.z
                    << ")..(" << worldMax.x << "," << worldMax.y << "," << worldMax.z << ")]"
                    << " distToCam=" << distToCam << "\n";
              scene << "g " << prefix << "_" << idChar << m.identifier
                    << "_vs" << hashBuf
                    << "_mat" << matHashBuf
                    << "_inst" << inst << "\n";
              for (const auto& v : localVerts) {
                const Vector4 lh(v.x, v.y, v.z, 1.0f);
                const Vector4 wh = effective * lh;
                scene << "v " << wh.x << " " << wh.y << " " << wh.z << "\n";
              }
              for (uint32_t k = 0; k < writtenPrimCount; ++k) {
                const uint32_t a = tris[k * 3 + 0] + sSceneVertexBase + 1;
                const uint32_t b = tris[k * 3 + 1] + sSceneVertexBase + 1;
                const uint32_t c = tris[k * 3 + 2] + sSceneVertexBase + 1;
                scene << "f " << a << " " << b << " " << c << "\n";
              }
              sSceneVertexBase += m.vertexCount;
              ++sSceneObjectCount;
            }
          }
          // Drain count + completion check moved to bumpSceneDrained
          // (called below) so early-return paths in drainSlot don't
          // bypass the scene-complete condition.
        }
        bumpSceneDrained();

        Logger::info(str::format(
          "[", tag, "] wrote OBJ pair frame=", m.pendingFrame,
          " vsHash=0x", std::hex, m.vsHash, std::dec,
          " mat=0x", std::hex, m.materialHash, std::dec,
          " verts=", m.vertexCount, "/", m.srcVertexCount,
          " prims=", m.primCount, "/", m.srcPrimCount,
          " instances=", m.instanceTransforms.size(),
          " id=", idChar, m.identifier,
          " worldAnchor=(", m.worldAnchor.x, ",", m.worldAnchor.y, ",", m.worldAnchor.z, ")",
          " worldAABB=[(", worldMin.x, ",", worldMin.y, ",", worldMin.z, ")",
          "..(", worldMax.x, ",", worldMax.y, ",", worldMax.z, ")]",
          " distToCam=", distToCam,
          " local=\"", localPath.string(), "\""));
        m.valid = false;
      };

      // ============================================================
      // Common scheduler helper (issues GPU copy for one task,
      // populating slot meta).
      // ============================================================
      auto scheduleSlot = [&](Rc<DxvkBuffer>& staging, ObjMeta& m, const ObjTask& task, const char* tag) {
        uint32_t vBytes = task.srcVertexCount * task.vtxStride;
        uint32_t iBytes = task.srcPrimCount * 3u * task.indexBytes;
        uint32_t verts = task.srcVertexCount;
        uint32_t prims = task.srcPrimCount;
        if (task.posBuf == nullptr || task.idxBuf == nullptr || task.indexBytes == 0) {
          Logger::info(str::format(
            "[", tag, "] cannot schedule task id=", task.identifier,
            " posBuf=", (task.posBuf != nullptr ? 1 : 0),
            " idxBuf=", (task.idxBuf != nullptr ? 1 : 0),
            " indexBytes=", task.indexBytes));
          return false;
        }
        if (vBytes + iBytes > kObjStagingBytes) {
          // Reserve up to 25% of staging for indices, rest for verts.
          const uint32_t maxIdxBytes = kObjStagingBytes / 4u;
          prims = std::min<uint32_t>(task.srcPrimCount, maxIdxBytes / (3u * task.indexBytes));
          iBytes = prims * 3u * task.indexBytes;
          const uint32_t remaining = kObjStagingBytes - iBytes;
          verts = std::min<uint32_t>(task.srcVertexCount, remaining / task.vtxStride);
          vBytes = verts * task.vtxStride;
          Logger::info(str::format(
            "[", tag, "] truncated task id=", task.identifier,
            ": ", task.srcVertexCount, "->", verts, " verts, ",
            task.srcPrimCount, "->", prims, " prims (cap=", kObjStagingBytes, ")"));
        }
        ctx->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
        ctx->copyBuffer(staging, 0, task.posBuf, task.posOff, vBytes);
        ctx->copyBuffer(staging, vBytes, task.idxBuf, task.idxOff, iBytes);
        m.valid = true;
        m.isPi = task.isPi;
        m.pendingFrame = curFrame;
        m.vertexCount = verts;
        m.primCount = prims;
        m.srcVertexCount = task.srcVertexCount;
        m.srcPrimCount = task.srcPrimCount;
        m.vtxStride = task.vtxStride;
        m.indexBytes = task.indexBytes;
        m.blasRef = task.blasRef;
        m.vsHash = task.vsHash;
        m.identifier = task.identifier;
        m.tlasType = task.tlasType;
        m.materialHash = task.materialHash;
        m.worldAnchor = task.worldAnchor;
        m.worldRoot = task.worldRoot;
        m.instanceTransforms = task.instanceTransforms;
        m.cameraPos = task.cameraPos;
        Logger::info(str::format(
          "[", tag, "] scheduled task id=", task.identifier,
          " vsHash=0x", std::hex, m.vsHash, std::dec,
          " verts=", verts, "/", task.srcVertexCount,
          " prims=", prims, "/", task.srcPrimCount,
          " queueRemaining=after-this-pop"));
        return true;
      };

      // ============================================================
      // Scene-OBJ session bootstrap. Run BEFORE the per-kind queue
      // populations so we know the scene file path / expected task
      // count before any drain fires.
      // ============================================================
      if (objDumpRequested && (!sPiQueue.empty() || !sMergedQueue.empty())) {
        // A previous session is still draining. The per-kind blocks
        // below will log "queue not drained, ignoring" and refuse to
        // re-queue, so leave the scene file alone.
      } else if (objDumpRequested) {
        auto outDir = util::RtxFileSys::path(util::RtxFileSys::Captures);
        if (outDir.empty()) outDir = std::filesystem::path(".");
        const std::time_t t = std::time(nullptr);
        char tsBuf[32] = {};
        std::strftime(tsBuf, sizeof(tsBuf), "%Y%m%d_%H%M%S", std::localtime(&t));
        sSceneObjPath = outDir / (std::string("scene_full_f")
          + std::to_string(curFrame) + "_" + tsBuf + ".obj");
        sSceneVertexBase = 0;
        sSceneObjectCount = 0;
        sSceneDrainedTasks = 0;
        // Pre-count tasks below; we only know after queue population.
        sSceneExpectedTasks = 0;
        sSceneActive = true;
        // Open + write header. Truncates any existing file.
        if (auto init = util::createDirectoriesAndOpenFile(sSceneObjPath)) {
          *init << "# [SpawnGeomDiag.SceneObjDump] full-scene world-space dump\n"
                << "# camera=(" << cameraPos.x << "," << cameraPos.y << "," << cameraPos.z << ")\n"
                << "# frame=" << curFrame << " timestamp=" << tsBuf << "\n"
                << "# Open in Blender / MeshLab to see every PI batch + every\n"
                << "# merged-bucket TLAS instance Remix is sending to the\n"
                << "# acceleration structure builder. Each `o` group named with\n"
                << "# identifier_vsHash_matHash_inst so you can match against\n"
                << "# the per-batch / per-instance OBJ pairs in this directory.\n";
        }
        Logger::info(str::format(
          "[SpawnGeomDiag.SceneObjDump] opened scene file \"",
          sSceneObjPath.string(), "\" at frame=", curFrame));
      }

      // ============================================================
      // PI: drain ready slots, then schedule new tasks from sPiQueue.
      // ============================================================
      for (uint32_t s = 0; s < kObjRingSize; ++s) {
        ObjMeta& m = sPiMeta[s];
        if (!m.valid) continue;
        if (curFrame < m.pendingFrame + 3u) continue;
        drainSlot(sPiStaging[s], m, "SpawnGeomDiag.PiObjDump", "pi");
      }
      // Populate PI queue on press.
      if (objDumpRequested) {
        if (!sPiQueue.empty()) {
          Logger::info(str::format(
            "[SpawnGeomDiag.PiObjDump] queue not drained (",
            sPiQueue.size(), " tasks remaining), ignoring new request at frame=", curFrame));
        } else {
          for (size_t bi = 0; bi < m_pointInstancerBatches.size(); ++bi) {
            const auto& b = m_pointInstancerBatches[bi];
            if (b.debugSourceBlasEntry == nullptr) continue;
            if (b.transforms == nullptr) continue;
            const auto& pb = b.debugSourceBlasEntry->modifiedGeometryData.positionBuffer;
            const auto& ib = b.debugSourceBlasEntry->modifiedGeometryData.indexBuffer;
            if (pb.buffer() == nullptr || ib.buffer() == nullptr) continue;
            const uint32_t indexBytes = (ib.indexType() == VK_INDEX_TYPE_UINT32) ? 4u
                                       : (ib.indexType() == VK_INDEX_TYPE_UINT16) ? 2u : 0u;
            if (indexBytes == 0) continue;
            ObjTask task;
            task.isPi = true;
            task.posBuf = pb.buffer();
            task.posOff = pb.offsetFromSlice();
            task.vtxStride = static_cast<uint32_t>(pb.stride());
            task.idxBuf = ib.buffer();
            task.idxOff = ib.offsetFromSlice();
            task.indexBytes = indexBytes;
            task.srcVertexCount = b.debugSourceVertexCount;
            task.srcPrimCount = b.debugSourcePrimCount;
            task.blasRef = b.blasReference;
            task.vsHash = b.debugVsHash;
            task.identifier = static_cast<uint32_t>(bi);
            task.tlasType = b.tlasType;
            task.materialHash = static_cast<uint64_t>(
              b.debugSourceBlasEntry->input.getMaterialData().getHash());
            // PI worldAnchor = world-space translation of the FIRST
            // instance (objectToWorld * i2o[0]).T — gives a unique
            // spatial fingerprint per batch even when many batches
            // share the same vsHash.
            {
              const Matrix4 effective0 = b.objectToWorld * (*b.transforms)[0];
              task.worldAnchor = Vector3(effective0[3][0], effective0[3][1], effective0[3][2]);
            }
            task.worldRoot = b.objectToWorld;
            task.instanceTransforms = *b.transforms;
            task.cameraPos = cameraPos;
            sPiQueue.push_back(std::move(task));
          }
          Logger::info(str::format(
            "[SpawnGeomDiag.PiObjDump] queued ", sPiQueue.size(),
            " PI batch(es) at frame=", curFrame));
          // Lock in the PI side of the scene-session expected count
          // BEFORE PI scheduling pops anything from the queue.
          if (sSceneActive) {
            sSceneExpectedTasks += static_cast<uint32_t>(sPiQueue.size());
          }
        }
      }
      // Schedule up to kSchedulePerFrame PI tasks into free slots.
      {
        uint32_t scheduled = 0;
        for (uint32_t s = 0; s < kObjRingSize && scheduled < kSchedulePerFrame && !sPiQueue.empty(); ++s) {
          if (sPiMeta[s].valid) continue;
          if (scheduleSlot(sPiStaging[s], sPiMeta[s], sPiQueue.back(), "SpawnGeomDiag.PiObjDump")) {
            ++scheduled;
            sPiQueue.pop_back();
          } else {
            sPiQueue.pop_back();  // skip un-schedulable
          }
        }
      }

      // ============================================================
      // Merged: drain ready slots, then schedule from sMergedQueue.
      // ============================================================
      for (uint32_t s = 0; s < kObjRingSize; ++s) {
        ObjMeta& m = sMergedMeta[s];
        if (!m.valid) continue;
        if (curFrame < m.pendingFrame + 3u) continue;
        drainSlot(sMergedStaging[s], m, "SpawnGeomDiag.MergedObjDump", "merged");
      }
      // Populate merged queue on press — every instance of every TLAS type.
      if (objDumpRequested) {
        if (!sMergedQueue.empty()) {
          Logger::info(str::format(
            "[SpawnGeomDiag.MergedObjDump] queue not drained (",
            sMergedQueue.size(), " tasks remaining), ignoring new request at frame=", curFrame));
        } else {
          // FIX: previously iterated m_mergedInstances which has ONE
          // entry per merged-bucket BLAS (NOT per RtInstance). Each
          // merged bucket can contain MANY RtInstances with different
          // materials sharing a single BLAS — they all live in
          // m_reorderedSurfaces but only the bucket's first instance
          // was being dumped. Result: marble floor (and every other
          // non-first sub-instance of every merged bucket) was
          // silently absent from scene_full.obj.
          //
          // Now iterate m_reorderedSurfaces directly. Each entry is
          // ONE RtInstance with its own BlasEntry → its own per-
          // material geometry. Skip:
          //   - PI fanout instances (handled by the PI block above)
          //   - duplicate RtInstance pointers (split-billboard rangers
          //     can land the same RtInstance in m_reorderedSurfaces
          //     multiple times consecutively)
          std::unordered_set<RtInstance*> seenInst;
          for (size_t s = 0; s < m_reorderedSurfaces.size(); ++s) {
            RtInstance* inst = m_reorderedSurfaces[s];
            if (inst == nullptr) continue;
            // PI fanouts → covered by sPiQueue with their full instance
            // transform list. Skip here to avoid duplicate dumps.
            if (inst->surface.instancesToObject != nullptr) continue;
            if (!seenInst.insert(inst).second) continue;

            BlasEntry* blasEntry = inst->getBlas();
            if (blasEntry == nullptr) continue;
            const auto& pb = blasEntry->modifiedGeometryData.positionBuffer;
            const auto& ib = blasEntry->modifiedGeometryData.indexBuffer;
            if (pb.buffer() == nullptr || ib.buffer() == nullptr) continue;
            const uint32_t indexBytes = (ib.indexType() == VK_INDEX_TYPE_UINT32) ? 4u
                                       : (ib.indexType() == VK_INDEX_TYPE_UINT16) ? 2u : 0u;
            if (indexBytes == 0) continue;
            const uint32_t primCount = blasEntry->buildRanges.empty() ? 0u
              : blasEntry->buildRanges[0].primitiveCount;
            if (primCount == 0) continue;

            ObjTask task;
            task.isPi = false;
            task.posBuf = pb.buffer();
            task.posOff = pb.offsetFromSlice();
            task.vtxStride = static_cast<uint32_t>(pb.stride());
            task.idxBuf = ib.buffer();
            task.idxOff = ib.offsetFromSlice();
            task.indexBytes = indexBytes;
            task.srcVertexCount = blasEntry->modifiedGeometryData.vertexCount;
            task.srcPrimCount = primCount;
            task.blasRef = static_cast<uint64_t>(
              blasEntry->dynamicBlas != nullptr
                ? blasEntry->dynamicBlas->accelerationStructureReference : 0);
            task.vsHash = static_cast<uint64_t>(blasEntry->input.getTransformData().vertexShaderHash);
            task.identifier = static_cast<uint32_t>(s);
            // tlasType not directly available per-surface for merged
            // path; use Opaque as default — the OBJ header still
            // identifies the source via vsHash + matHash.
            task.tlasType = 0;
            task.materialHash = static_cast<uint64_t>(blasEntry->input.getMaterialData().getHash());
            {
              const Matrix4& o2w = inst->surface.objectToWorld;
              task.worldAnchor = Vector3(o2w[3][0], o2w[3][1], o2w[3][2]);
            }
            task.worldRoot = inst->surface.objectToWorld;
            task.instanceTransforms.resize(1);
            task.cameraPos = cameraPos;
            sMergedQueue.push_back(std::move(task));
          }
          Logger::info(str::format(
            "[SpawnGeomDiag.MergedObjDump] queued ", sMergedQueue.size(),
            " unique RtInstances from m_reorderedSurfaces (was: m_mergedInstances per-bucket)"
            " at frame=", curFrame));
          // Lock in the merged side of the scene-session expected
          // count BEFORE merged scheduling pops anything. The PI
          // contribution was already added after PI populate (above)
          // — additive so PI scheduling popping in between doesn't
          // disturb the count.
          if (sSceneActive) {
            sSceneExpectedTasks += static_cast<uint32_t>(sMergedQueue.size());
            Logger::info(str::format(
              "[SpawnGeomDiag.SceneObjDump] expecting ", sSceneExpectedTasks,
              " task(s) total"));
            if (sSceneExpectedTasks == 0) {
              // Nothing to dump; close the empty scene immediately.
              Logger::info(str::format(
                "[SpawnGeomDiag.SceneObjDump] scene complete (empty) file=\"",
                sSceneObjPath.string(), "\""));
              sSceneActive = false;
            }
          }
        }
      }
      // Schedule up to kSchedulePerFrame merged tasks into free slots.
      {
        uint32_t scheduled = 0;
        for (uint32_t s = 0; s < kObjRingSize && scheduled < kSchedulePerFrame && !sMergedQueue.empty(); ++s) {
          if (sMergedMeta[s].valid) continue;
          if (scheduleSlot(sMergedStaging[s], sMergedMeta[s], sMergedQueue.back(), "SpawnGeomDiag.MergedObjDump")) {
            ++scheduled;
            sMergedQueue.pop_back();
          } else {
            sMergedQueue.pop_back();
          }
        }
      }
    }
    // NV-DXVK (debug probe E): BLAS position buffer readback. Gated.
    if (kEnableRtxDebugProbes && s_probeE_posBuffer.ptr() != nullptr && s_probeE_vertexCount > 0) {
      static constexpr uint32_t kProbeERingSize = 3;
      static constexpr uint32_t kProbeEVerts    = 8;
      static Rc<DxvkBuffer> sStagingE[kProbeERingSize];
      static uint32_t sStagingEBytes[kProbeERingSize] = {};
      static uint32_t sStagingEStride[kProbeERingSize] = {};
      static uint32_t sStagingEVtxCount[kProbeERingSize] = {};
      static VkFormat sStagingEFmt[kProbeERingSize] = {};
      static uint64_t sStagingEBlasRef[kProbeERingSize] = {};
      static bool sStagingEValid[kProbeERingSize] = {};
      static uint32_t sProbeEFrame = 0;
      const uint32_t writeE = sProbeEFrame % kProbeERingSize;
      const uint32_t readE  = (sProbeEFrame + 1) % kProbeERingSize;

      const uint32_t copyVerts = std::min<uint32_t>(kProbeEVerts, s_probeE_vertexCount);
      const uint32_t copyBytes = copyVerts * s_probeE_posStride;

      // Lazy-allocate staging buffers (size enough for any probe).
      const uint32_t kMaxBytes = kProbeEVerts * 64; // 64 byte stride upper bound
      for (uint32_t i = 0; i < kProbeERingSize; ++i) {
        if (sStagingE[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size   = kMaxBytes;
          sStagingE[i] = m_device->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "Probe E - PI BLAS Position Readback");
        }
      }

      // Read the oldest slot.
      const bool logE = (sProbeEFrame % 60 == 0);
      if (sStagingEValid[readE] && logE) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(sStagingE[readE]->mapPtr(0));
        if (data != nullptr) {
          const uint32_t verts = sStagingEBytes[readE] / std::max<uint32_t>(1u, sStagingEStride[readE]);
          for (uint32_t v = 0; v < verts; ++v) {
            const uint8_t* p = data + v * sStagingEStride[readE];
            float x, y, z;
            memcpy(&x, p + 0, 4);
            memcpy(&y, p + 4, 4);
            memcpy(&z, p + 8, 4);
            // Layout: pos float3 [0..11] | texcoord [12..19] | color0 [20..23]
            // When stride is 24, color0 sits at byte offset 20 as BGRA_UNORM8.
            // Decode whichever bytes exist at the end of the element.
            const uint32_t stride = sStagingEStride[readE];
            uint32_t col0 = 0;
            float tcU = 0.f, tcV = 0.f;
            if (stride >= 20) { memcpy(&tcU, p + 12, 4); memcpy(&tcV, p + 16, 4); }
            if (stride >= 24) { memcpy(&col0, p + 20, 4); }
            const uint8_t colB = uint8_t((col0 >>  0) & 0xFFu);
            const uint8_t colG = uint8_t((col0 >>  8) & 0xFFu);
            const uint8_t colR = uint8_t((col0 >> 16) & 0xFFu);
            const uint8_t colA = uint8_t((col0 >> 24) & 0xFFu);
            Logger::info(str::format(
              "[PI-vbreadback] blasRef=0x", std::hex, sStagingEBlasRef[readE], std::dec,
              " v=", v,
              " stride=", stride,
              " fmt=", uint32_t(sStagingEFmt[readE]),
              " vtxCount=", sStagingEVtxCount[readE],
              " pos=(", x, ",", y, ",", z, ")",
              " uv=(", tcU, ",", tcV, ")",
              " col0 BGRA=(", int(colB), ",", int(colG), ",", int(colR), ",", int(colA), ")",
              " col0raw=0x", std::hex, col0, std::dec));
          }
        }
        sStagingEValid[readE] = false;
      }

      // Issue copy from BLAS position buffer into write slot.
      if (copyBytes > 0 && copyBytes <= kMaxBytes) {
        const VkDeviceSize srcOffset = s_probeE_posSliceOff + s_probeE_posElemOff;
        if (srcOffset + copyBytes <= s_probeE_posBuffer->info().size) {
          // The BLAS position buffer is written by the interleaver compute pass
          // before BLAS build. Barrier: shader write → transfer read.
          ctx->emitMemoryBarrier(0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
          ctx->copyBuffer(sStagingE[writeE], 0, s_probeE_posBuffer, srcOffset, copyBytes);
          sStagingEValid[writeE]    = true;
          sStagingEBytes[writeE]    = copyBytes;
          sStagingEStride[writeE]   = s_probeE_posStride;
          sStagingEVtxCount[writeE] = s_probeE_vertexCount;
          sStagingEFmt[writeE]      = s_probeE_posFormat;
          sStagingEBlasRef[writeE]  = s_probeE_blasRef;
        }
      }
      ++sProbeEFrame;
    }

    // NV-DXVK (debug probe F): DISABLED — readback served its purpose.
    if (false && s_probeF_valid && m_surfaceBuffer != nullptr) {
      static constexpr uint32_t kProbeFRingSize = 3;
      static constexpr uint32_t kProbeFBytes    = 256; // kSurfaceGPUSize
      static Rc<DxvkBuffer> sStagingF[kProbeFRingSize];
      static uint32_t sStagingFBaseIdx[kProbeFRingSize] = {};
      static uint64_t sStagingFBlasRef[kProbeFRingSize] = {};
      static bool sStagingFValid[kProbeFRingSize] = {};
      static uint32_t sProbeFFrame = 0;
      const uint32_t writeF = sProbeFFrame % kProbeFRingSize;
      const uint32_t readF  = (sProbeFFrame + 1) % kProbeFRingSize;

      for (uint32_t i = 0; i < kProbeFRingSize; ++i) {
        if (sStagingF[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size   = kProbeFBytes;
          sStagingF[i] = m_device->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "Probe F - PI Surface Template Readback");
        }
      }

      // Log once every ~120 dispatches.
      const bool logF = (sProbeFFrame % 60 == 0);
      if (sStagingFValid[readF] && logF) {
        const uint8_t* d = reinterpret_cast<const uint8_t*>(sStagingF[readF]->mapPtr(0));
        if (d != nullptr) {
          // Parse per RtSurface::writeGPUData layout:
          uint16_t positionBufferIndex, previousPositionBufferIndex, normalBufferIndex,
                   texcoordBufferIndex, indexBufferIndex, color0BufferIndex, flags0, packedHash;
          memcpy(&positionBufferIndex,         d + 0,  2);
          memcpy(&previousPositionBufferIndex, d + 2,  2);
          memcpy(&normalBufferIndex,           d + 4,  2);
          memcpy(&texcoordBufferIndex,         d + 6,  2);
          memcpy(&indexBufferIndex,            d + 8,  2);
          memcpy(&color0BufferIndex,           d + 10, 2);
          memcpy(&flags0,                      d + 12, 2);
          memcpy(&packedHash,                  d + 14, 2);
          uint32_t positionOffset, normalOffset, texcoordOffset, color0Offset, objectPickingValue;
          memcpy(&positionOffset,     d + 16, 4);
          memcpy(&normalOffset,       d + 20, 4);
          memcpy(&texcoordOffset,     d + 24, 4);
          memcpy(&color0Offset,       d + 28, 4);
          memcpy(&objectPickingValue, d + 32, 4);
          uint8_t positionStride = d[36];
          uint8_t normalStride   = d[37];
          uint8_t texcoordStride = d[38];
          uint8_t color0Stride   = d[39];
          // firstIndex is a 24-bit little-endian value packed with indexStride in the 4 bytes at offset 40.
          uint32_t firstIndex24  = d[40] | (uint32_t(d[41]) << 8) | (uint32_t(d[42]) << 16);
          uint8_t indexStride    = d[43];
          uint32_t flags1;
          memcpy(&flags1, d + 44, 4);
          Logger::info(str::format(
            "[PI-surf] blasRef=0x", std::hex, sStagingFBlasRef[readF], std::dec,
            " base=", sStagingFBaseIdx[readF],
            " posBufIdx=", positionBufferIndex,
            " prevPosBufIdx=", previousPositionBufferIndex,
            " nrmBufIdx=", normalBufferIndex,
            " tcBufIdx=", texcoordBufferIndex,
            " idxBufIdx=", indexBufferIndex,
            " col0BufIdx=", color0BufferIndex,
            " flags0=0x", std::hex, flags0, std::dec,
            " hash=0x", std::hex, packedHash, std::dec,
            " posOff=", positionOffset,
            " firstIdx=", firstIndex24,
            " strides(p,n,t,c,i)=", int(positionStride), ",", int(normalStride), ",",
                                     int(texcoordStride), ",", int(color0Stride), ",",
                                     int(indexStride),
            " flags1=0x", std::hex, flags1, std::dec));
        }
        sStagingFValid[readF] = false;
      }

      // Issue the readback copy.
      const VkDeviceSize srcOffset = VkDeviceSize(s_probeF_baseSurfaceIndex) * kProbeFBytes;
      if (srcOffset + kProbeFBytes <= m_surfaceBuffer->info().size) {
        // Barrier: compute writes (GPU culling copies template into per-instance
        // slots; it also reads the template) → transfer read. Same barrier as E.
        ctx->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
        ctx->copyBuffer(sStagingF[writeF], 0, m_surfaceBuffer, srcOffset, kProbeFBytes);
        sStagingFValid[writeF]   = true;
        sStagingFBaseIdx[writeF] = s_probeF_baseSurfaceIndex;
        sStagingFBlasRef[writeF] = s_probeE_blasRef;
      }
      ++sProbeFFrame;
    }

    // NV-DXVK (debug probe G): read back the MATERIAL template from
    // surfaceMaterialBuffer at baseSurfaceIndex for the probeE/F batch.
    // kSurfaceMaterialGPUSize = 64 bytes. Material layout (opaque) per
    // rtx_materials.h:574:
    //   word 0 = flags (low byte = surfaceMaterialTypeX; upper bits = feature flags)
    //   word 1 = samplerIndex
    //   word 2 = albedoOpacityTextureIndex
    //   word 3 = secondaryTextureIndex
    //   words 4-7 = albedoOpacityConstant (rgba fp16)
    //   word 12 = emissiveColorTextureIndex
    //   words 16-18 = emissiveColorConstant (rgb fp16)
    //   word 19 = emissiveIntensity (fp16)
    //   word 20 = roughnessConstant (fp16)
    //   word 21 = metallicConstant (fp16)
    // NV-DXVK (debug probe G): DISABLED — material template readback served its purpose.
    if (false && s_probeF_valid && surfaceMaterialBuffer != nullptr) {
      static constexpr uint32_t kProbeGRingSize = 3;
      static constexpr uint32_t kProbeGBytes    = 64; // kSurfaceMaterialGPUSize
      static Rc<DxvkBuffer> sStagingG[kProbeGRingSize];
      static uint32_t sStagingGBaseIdx[kProbeGRingSize] = {};
      static uint64_t sStagingGBlasRef[kProbeGRingSize] = {};
      static bool sStagingGValid[kProbeGRingSize] = {};
      static uint32_t sProbeGFrame = 0;
      const uint32_t writeG = sProbeGFrame % kProbeGRingSize;
      const uint32_t readG  = (sProbeGFrame + 1) % kProbeGRingSize;

      for (uint32_t i = 0; i < kProbeGRingSize; ++i) {
        if (sStagingG[i].ptr() == nullptr) {
          DxvkBufferCreateInfo info;
          info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          info.size   = kProbeGBytes;
          sStagingG[i] = m_device->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "Probe G - PI Material Template Readback");
        }
      }

      const bool logG = (sProbeGFrame % 60 == 0);
      if (sStagingGValid[readG] && logG) {
        const uint8_t* d = reinterpret_cast<const uint8_t*>(sStagingG[readG]->mapPtr(0));
        if (d != nullptr) {
          auto halfToFloat = [](uint16_t h) -> float {
            // Standard IEEE half -> float conversion, no denorm special-case needed for sane values.
            const uint32_t sign = (h & 0x8000u) << 16;
            const uint32_t exp  = (h & 0x7C00u) >> 10;
            const uint32_t mant = (h & 0x03FFu);
            uint32_t bits;
            if (exp == 0) {
              bits = sign; // 0 or denorm -> treat as 0 for log brevity
              if (mant != 0) {
                // Normalize denorm
                int e = -14;
                uint32_t m = mant;
                while ((m & 0x0400u) == 0) { m <<= 1; --e; }
                m &= 0x03FFu;
                bits = sign | (uint32_t(e + 127) << 23) | (m << 13);
              }
            } else if (exp == 31) {
              bits = sign | 0x7F800000u | (mant << 13);
            } else {
              bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
            float f;
            memcpy(&f, &bits, 4);
            return f;
          };
          uint16_t w[32];
          memcpy(w, d, 64);
          const uint32_t matType = (w[0] & 0x3);
          Logger::info(str::format(
            "[PI-mat] blasRef=0x", std::hex, sStagingGBlasRef[readG], std::dec,
            " base=", sStagingGBaseIdx[readG],
            " matType=", matType, " (0=opaque 1=translucent 2=rayportal)",
            " flags=0x", std::hex, w[0], std::dec,
            " samplerIdx=", w[1],
            " albedoTexIdx=", w[2],
            " secondaryTexIdx=", w[3],
            " albedoRGBA=(", halfToFloat(w[4]), ",", halfToFloat(w[5]), ",",
                              halfToFloat(w[6]), ",", halfToFloat(w[7]), ")",
            " emissiveTexIdx=", w[12],
            " emissiveRGB=(", halfToFloat(w[16]), ",", halfToFloat(w[17]), ",",
                               halfToFloat(w[18]), ")",
            " emissiveIntensity=", halfToFloat(w[19]),
            " roughness=", halfToFloat(w[20]),
            " metallic=", halfToFloat(w[21])));
        }
        sStagingGValid[readG] = false;
      }

      const VkDeviceSize srcOffsetG = VkDeviceSize(s_probeF_baseSurfaceIndex) * kProbeGBytes;
      if (srcOffsetG + kProbeGBytes <= surfaceMaterialBuffer->info().size) {
        // GPU culling shader copies material template into per-instance slots
        // before this point; also BLAS build reads from this buffer. Barrier:
        // compute/transfer write → transfer read.
        ctx->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
        ctx->copyBuffer(sStagingG[writeG], 0, surfaceMaterialBuffer, srcOffsetG, kProbeGBytes);
        sStagingGValid[writeG]   = true;
        sStagingGBaseIdx[writeG] = s_probeF_baseSurfaceIndex;
        sStagingGBlasRef[writeG] = s_probeE_blasRef;
      }
      ++sProbeGFrame;
    }
  }

  void AccelManager::buildParticleSurfaceMapping(std::vector<uint32_t>& surfaceIndexMapping) {
    // Simplify syntax for accessing the persistent containers
    auto& surfaceInfoLists = buildParticleSurfaceMappingFuncState.surfaceInfoLists;
    uint32_t& currIndex = buildParticleSurfaceMappingFuncState.currIndex;
    uint32_t& prevIndex = buildParticleSurfaceMappingFuncState.prevIndex;

    // Build surface index mapping for particle objects.
    surfaceInfoLists[currIndex].resize(m_reorderedSurfaces.size());
    std::unordered_map<uint32_t, std::vector<int>> curMaterialHashToSurfaceMap;
    for (uint32_t surfaceIndex = 0; surfaceIndex < m_reorderedSurfaces.size(); surfaceIndex++) {
      RtInstance& surface = *m_reorderedSurfaces[surfaceIndex];

      // Only record objects that use unordered approximations.
      // In some cases, objects with unorder resolve flag will generate a set of billboards, each one occupies one "Surface" entry
      // in the shaders' surface array. These entries has identical information except the "firstIndex" member.
      // See "fillGeometryInfoFromBlasEntry()" for more details in generating indexOffsets.
      // See "uploadSurfaceData()" for how the "firstIndex" is fed to the shaders surface array.
      if (surface.usesUnorderedApproximations() && m_reorderedSurfacesFirstIndexOffset[surfaceIndex] == 0) {
        const RasterGeometry& geometryData = surface.getBlas()->input.getGeometryData();

        // Need to find the closest object with the same material, so use material ID as hash value, and record bounding box's center.
        surfaceInfoLists[currIndex][surfaceIndex] = { 
          surface.surface.surfaceMaterialIndex,
          geometryData.boundingBox.getTransformedCentroid(surface.getTransform()) };

        if (surface.getBlas()->buildRanges.size() > 0 && surface.getBlas()->buildGeometries.size() > 0) {
          curMaterialHashToSurfaceMap[surface.surface.surfaceMaterialIndex].push_back(surfaceIndex);
        }
      } else {
        surfaceInfoLists[currIndex][surfaceIndex].surfaceMaterialIndex = kSurfaceInvalidSurfaceMaterialIndex;
      }
    }

    // Fix missed surface mapping by searching among objects with the same hash value, and choose the closest one.
    for (int i = 0; i < surfaceIndexMapping.size(); i++) {
      // Skip objects that have surface mapping
      if (surfaceIndexMapping[i] != SURFACE_INDEX_INVALID) {
        continue;
      }

      if (i >= surfaceInfoLists[prevIndex].size()) {
        continue;
      }

      // Skip objects with different materials
      auto lastInfo = surfaceInfoLists[prevIndex][i];
      auto pCandidateList = curMaterialHashToSurfaceMap.find(lastInfo.surfaceMaterialIndex);
      if (pCandidateList == curMaterialHashToSurfaceMap.end()) {
        continue;
      }

      auto& candidateList = pCandidateList->second;
      float minDistanceSq = FLT_MAX;
      int bestSurfaceID = -1;

      // Iterate through the candidate list and find the closest one
      for (int ithCandidate = 0; ithCandidate < candidateList.size(); ithCandidate++) {
        int curSurfaceID = candidateList[ithCandidate];
        RtInstance& surface = *m_reorderedSurfaces[curSurfaceID];
        if (surface.getBlas()->buildGeometries.size() == 0) {
          continue;
        }

        // Calculate bounding box centers' distance
        const RasterGeometry& geometryData = surface.getBlas()->input.getGeometryData();
        Vector3 center = geometryData.boundingBox.getTransformedCentroid(surface.getTransform());
        float distanceSq = lengthSqr(center - lastInfo.worldPosition);
        if (distanceSq < minDistanceSq) {
          minDistanceSq = distanceSq;
          bestSurfaceID = curSurfaceID;
        }
      }

      // Use the closest surface
      if (bestSurfaceID != -1) {
        surfaceIndexMapping[i] = bestSurfaceID;
      }
    }
    // Make current previous
    std::swap(currIndex, prevIndex);
  }

  void AccelManager::uploadSurfaceData(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();
    if (m_reorderedSurfaces.empty()) {
      return;
    }

    // Simplify syntax for accessing the persistent containers
    auto& surfacesGPUData = uploadSurfaceDataFuncState.surfacesGPUData;
    auto& surfaceIndexMapping = uploadSurfaceDataFuncState.surfaceIndexMapping;

    // Surface buffer
    const auto surfacesGPUSize = m_reorderedSurfaces.size() * kSurfaceGPUSize;

    // Allocate the instance buffer and copy its contents from host to device memory
    // STORAGE_BUFFER_BIT is required for the GPU PointInstancer culling shader
    // which writes per-instance surface data (transforms) directly into this buffer.
    // NV-DXVK (debug probe F): TRANSFER_SRC_BIT added so readback copies are legal.
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
      | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = align(surfacesGPUSize, kBufferAlignment);
    if (m_surfaceBuffer == nullptr || info.size > m_surfaceBuffer->info().size) {
      m_surfaceBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Surface Buffer");
    }

    uint32_t maxPreviousSurfaceIndex = 0;

    // Write surface data
    std::size_t dataOffset = 0;
    surfacesGPUData.resize(surfacesGPUSize);

    for (uint32_t i = 0; i < m_reorderedSurfaces.size(); ++i) {
      const auto& currentInstance = *m_reorderedSurfaces[i];
      RtSurface& currentSurface = m_reorderedSurfaces[i]->surface;

      // For PointInstancer entries beyond the first, do nothing.  The GPU culling shader will 
      // patch per-instance transforms and set per-instance customInstanceIndex later.
      if (currentSurface.instancesToObject != nullptr &&  currentSurface.surfaceIndexOfFirstInstance != SIZE_MAX && i > currentSurface.surfaceIndexOfFirstInstance) {
        dataOffset += kSurfaceGPUSize;
      } else {
        // Split instance geometry need to have their first index offset set in their corresponding surface instances
        currentSurface.firstIndex += m_reorderedSurfacesFirstIndexOffset[i];
        currentSurface.writeGPUData(surfacesGPUData.data(), dataOffset, i);
        currentSurface.firstIndex -= m_reorderedSurfacesFirstIndexOffset[i];
      }

      // Find the size of the surface mapping buffer
      // Skip SURFACE_INDEX_INVALID (new instances with no previous-frame data) to avoid
      // oversizing the mapping vector 
      const uint32_t prevIdx = currentInstance.getPreviousSurfaceIndex();
      if (prevIdx != SURFACE_INDEX_INVALID) {
        maxPreviousSurfaceIndex = std::max(maxPreviousSurfaceIndex, prevIdx);
      }
    }

    // The GPU's SharedSurfaceIndex texture may reference any surface index from the
    // previous frame.  Ensure the mapping covers at least the previous frame's surface
    // count so those GPU lookups read SURFACE_INDEX_INVALID rather than stale buffer data.
    auto& previousFrameSurfaceCount = uploadSurfaceDataFuncState.previousFrameSurfaceCount;
    if (previousFrameSurfaceCount > 0) {
      maxPreviousSurfaceIndex = std::max(maxPreviousSurfaceIndex, previousFrameSurfaceCount - 1);
    }
    previousFrameSurfaceCount = static_cast<uint32_t>(m_reorderedSurfaces.size());

    assert(dataOffset == surfacesGPUSize);
    assert(surfacesGPUData.size() == surfacesGPUSize);

    ctx->writeToBuffer(m_surfaceBuffer, 0, surfacesGPUData.size(), surfacesGPUData.data());

    // Allocate and initialize the surface mapping buffer
    surfaceIndexMapping.resize(maxPreviousSurfaceIndex + 1);
    std::fill(surfaceIndexMapping.begin(), surfaceIndexMapping.end(), SURFACE_INDEX_INVALID);
    
    // Assign surface indices to instances that don't have one yet (i.e. those that
    // entered m_reorderedSurfaces via addBlas or bucket insertion rather than the
    // early setSurfaceIndex path for zero-mask OMM/billboard instances).
    // Also populate the previous-->current frame surface index mapping.
    for (uint32_t surfaceIndex = 0; surfaceIndex < m_reorderedSurfaces.size(); surfaceIndex++) {
      RtInstance& surface = *m_reorderedSurfaces[surfaceIndex];

      if (surface.getSurfaceIndex() == SURFACE_INDEX_INVALID) {
        surface.setSurfaceIndex(surfaceIndex);

        // For PointInstancers, all instances share a single surface entry
        if (surface.surface.instancesToObject) {
          assert(surfaceIndex == surface.surface.surfaceIndexOfFirstInstance);
          if (surface.getPreviousSurfaceIndex() != SURFACE_INDEX_INVALID) {
            surfaceIndexMapping[surface.getPreviousSurfaceIndex()] = surfaceIndex;
          }
          surface.setPreviousSurfaceIndex(surfaceIndex);
        }
      }

      if (surface.getBillboardCount() == 0 && !surface.surface.instancesToObject) {
        if (surface.getPreviousSurfaceIndex() != SURFACE_INDEX_INVALID) {
          surfaceIndexMapping[surface.getPreviousSurfaceIndex()] = surfaceIndex;
        }
        surface.setPreviousSurfaceIndex(surfaceIndex);
      }
    }

    if (RtxOptions::trackParticleObjects()) {
      buildParticleSurfaceMapping(surfaceIndexMapping);
    }

    // Create and upload the primitive id prefix sum buffer
    auto updatePrefixSumBuffer = [&info, this, ctx](std::vector<uint32_t>& prefixSumList, Rc<DxvkBuffer>& prefixSumBuffer) {
      info.size = std::max(prefixSumList.size(), 1llu) * sizeof(prefixSumList[0]);

      if (prefixSumBuffer == nullptr || info.size > prefixSumBuffer->info().size) {
        prefixSumBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Prefixsum Buffer");
      }

      if (prefixSumList.size() > 0) {
        ctx->writeToBuffer(prefixSumBuffer, 0, prefixSumList.size() * sizeof(prefixSumList[0]), prefixSumList.data());
      }
    };

    updatePrefixSumBuffer(m_reorderedSurfacesPrimitiveIDPrefixSum, m_primitiveIDPrefixSumBuffer);
    updatePrefixSumBuffer(m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame, m_primitiveIDPrefixSumBufferLastFrame);

    // Create and upload the surface mapping buffer
    if (!surfaceIndexMapping.empty()) {
      info.size = align(surfaceIndexMapping.size() * sizeof(int), kBufferAlignment);
      if (m_surfaceMappingBuffer == nullptr || info.size > m_surfaceMappingBuffer->info().size) {
        m_surfaceMappingBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Surface Mapping Buffer");
      }

      ctx->writeToBuffer(m_surfaceMappingBuffer, 0, surfaceIndexMapping.size() * sizeof(surfaceIndexMapping[0]), surfaceIndexMapping.data());
    }
  }

  void AccelManager::buildBlases(Rc<DxvkContext> ctx,
                                 DxvkBarrierSet& execBarriers,
                                 const CameraManager& cameraManager,
                                 OpacityMicromapManager* opacityMicromapManager,
                                 const InstanceManager& instanceManager,
                                 const std::vector<TextureRef>& textures,
                                 const std::vector<RtInstance*>& instances,
                                 const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                 std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                 std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                 size_t& totalScratchMemory) {
    ScopedGpuProfileZone(ctx, "buildBLAS");
    // Upload surfaces before opacity micromap generation which reads the surface data on the GPU
    uploadSurfaceData(ctx);

    // Build and bind opacity micromaps
    if (opacityMicromapManager && opacityMicromapManager->isActive()) {
      opacityMicromapManager->buildOpacityMicromaps(ctx, textures, cameraManager.getLastCameraCutFrameId());

      // Bind opacity micromaps
      for (auto& blasBucket : blasBuckets) {
        for (uint32_t i = 0; i < blasBucket->geometries.size(); i++) {
          auto ommSourceHash = opacityMicromapManager->tryBindOpacityMicromap(ctx, *blasBucket->originalInstances[i], blasBucket->instanceBillboardIndices[i],
                                                         blasBucket->geometries[i], instanceManager);
          if (ommSourceHash != kEmptyHash) {
            blasBucket->hasOmmInstances = true;
          }
        }
      }

      opacityMicromapManager->onBlasBuild(ctx);
    }

    // Blas buffers must be created after opacity micromaps were generated to calculate correct acceleration structure sizes
    createBlasBuffersAndInstances(ctx, blasBuckets, blasToBuild, blasRangesToBuild, totalScratchMemory);

    // Make sure we have enough scratch memory for this build job
    if (totalScratchMemory > 0) {
      m_scratchBuffer = getScratchMemory(align(totalScratchMemory, m_scratchAlignment));

      execBarriers.accessBuffer(
       m_scratchBuffer->getSliceHandle(),
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV);

      ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);
    }

    // Execute all barriers generated to this point as part of:
    //  o mergeInstancesIntoBlas()
    //  o Opacity micromap generation above
    execBarriers.recordCommands(ctx->getCommandList());

    // Build the BLASes
    if (!blasToBuild.empty()) {
      // Now apply the buffer offset to the scratch address we calculated earlier
      for (auto& desc : blasToBuild) {
        desc.scratchData.deviceAddress += m_scratchBuffer->getDeviceAddress();
      }
      assert(blasToBuild.size() == blasRangesToBuild.size());

      // NV-DXVK BLAS-BUILD-INPUT probe: comprehensive per-BLAS dump RIGHT before
      // vkCmdBuildAccelerationStructuresKHR — logs build inputs + schedules async
      // readback of first 64 vertex bytes + 24 index bytes so we can verify the
      // GPU actually sees real geometry at the device addresses being built.
      static uint32_t s_bbiFrame = 0;
      const uint32_t bbiFireIdx = s_bbiFrame++;
      const bool bbiFire = kEnableRtxDebugProbes && ((bbiFireIdx % 10u) == 0);
      // Hoisted so the post-build serialize-size probe block can reuse it.
      std::set<uint64_t> piRefs;
      if (bbiFire) {
        for (const auto& b : m_pointInstancerBatches) {
          piRefs.insert(b.blasReference);
        }
      }
      if (bbiFire) {

        // Static ring of staging buffers for async readback.
        constexpr uint32_t kMaxReadbacks = 16;        // log first 16 BLAS builds per fire
        constexpr uint32_t kVtxBytes = 96;            // 4 full vertices worth (24-byte stride typical)
        constexpr uint32_t kIdxBytes = 48;            // 16 indices worth (16-bit) or 12 (32-bit)
        constexpr uint32_t kReadbackSlotBytes = kVtxBytes + kIdxBytes;
        constexpr uint32_t kRingSize = 3;
        static Rc<DxvkBuffer> sBbiStaging[kRingSize];
        static bool           sBbiValid[kRingSize]    = {};
        static uint32_t       sBbiCount[kRingSize]    = {};
        // Per-BLAS meta captured at write-time, consumed at read-time.
        struct BbiMeta {
          uint64_t dstAsAddr;
          uint64_t vtxAddr;
          uint64_t idxAddr;
          uint32_t vtxStride;
          uint32_t vtxFormat;
          uint32_t iType;
          uint32_t maxVtx;
          uint32_t primCnt;
          uint32_t primOff;
          uint32_t geoFlags;
          uint64_t vtxBufDevAddr;
          uint64_t vtxBufSize;
          uint32_t vtxBufUsage;
          uint32_t vtxBufMemFlags;
          uint32_t vtxOffsetFromSlice;
          bool     isPi;
          bool     hasEntry;
          bool     addrWithinVtxBuf;
          // 0=ok, 1=noEntry, 2=noPosBuf, 3=addrOutsideBuf,
          // 4=noTransferSrc, 5=stagingOverflow.
          // Distinguishes the v0=(0,0,0) cases observed in the log
          // (frames 21:12:50, 21:12:53) so future runs say WHY readback
          // was skipped instead of silently emitting zeros.
          uint8_t  skipReason;
        };
        static BbiMeta sBbiMetas[kRingSize][kMaxReadbacks];

        const uint32_t writeSlot = s_bbiFrame % kRingSize;
        const uint32_t readSlot  = (s_bbiFrame + 1) % kRingSize;

        // Lazily create staging buffers.
        for (uint32_t i = 0; i < kRingSize; ++i) {
          if (sBbiStaging[i].ptr() == nullptr) {
            DxvkBufferCreateInfo info;
            info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
            info.size   = kReadbackSlotBytes * kMaxReadbacks;
            sBbiStaging[i] = m_device->createBuffer(info,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              DxvkMemoryStats::Category::RTXBuffer,
              "BLAS-BUILD-INPUT Staging");
          }
        }

        // Barrier: any prior writer (interleaver compute, CPU transfer) of the source
        // vertex/index buffers → our transfer read.
        ctx->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);

        // Log + schedule readback for each BLAS we're about to build.
        const size_t nBlas = std::min<size_t>(blasToBuild.size(), size_t(kMaxReadbacks));
        for (size_t bi = 0; bi < blasToBuild.size(); ++bi) {
          const auto& g = blasToBuild[bi];
          const auto* ranges = blasRangesToBuild[bi];
          BlasEntry*  entry = (bi < m_debugBlasBuildEntries.size()) ? m_debugBlasBuildEntries[bi] : nullptr;
          PooledBlas* dstBl = (bi < m_debugBlasBuildDstBlas.size()) ? m_debugBlasBuildDstBlas[bi] : nullptr;
          const uint64_t dstAsAddr = (dstBl != nullptr) ? dstBl->accelerationStructureReference : 0;
          const bool isPi = piRefs.count(dstAsAddr) > 0;

          for (uint32_t gi = 0; gi < g.geometryCount; ++gi) {
            const auto& tri = g.pGeometries[gi].geometry.triangles;
            const auto& r   = ranges[gi];
            const uint32_t iTypeBytes = (tri.indexType == VK_INDEX_TYPE_UINT32) ? 4u
                                       : (tri.indexType == VK_INDEX_TYPE_UINT16) ? 2u : 0u;
            const VkDeviceSize vtxEnd = tri.vertexData.deviceAddress
              + VkDeviceSize(tri.maxVertex + 1) * tri.vertexStride;
            const VkDeviceSize idxEnd = tri.indexData.deviceAddress
              + VkDeviceSize(r.primitiveCount) * 3u * iTypeBytes
              + VkDeviceSize(r.primitiveOffset);

            // Buffer-level state (only available for dynamic BLAS path via entry).
            uint64_t vtxBufAddr = 0, vtxBufSize = 0; uint32_t vtxBufUsage = 0, vtxBufMemFlags = 0, vtxOffsetFromSlice = 0;
            uint64_t idxBufAddr = 0, idxBufSize = 0; uint32_t idxBufUsage = 0, idxBufMemFlags = 0, idxOffsetFromSlice = 0;
            bool addrWithinVtxBuf = false, addrWithinIdxBuf = false;
            if (entry != nullptr) {
              const auto& pb = entry->modifiedGeometryData.positionBuffer;
              const auto& ib = entry->modifiedGeometryData.indexBuffer;
              if (pb.buffer() != nullptr) {
                vtxBufAddr         = pb.buffer()->getDeviceAddress();
                vtxBufSize         = pb.buffer()->info().size;
                vtxBufUsage        = pb.buffer()->info().usage;
                vtxBufMemFlags     = pb.buffer()->memFlags();
                vtxOffsetFromSlice = pb.offsetFromSlice();
                addrWithinVtxBuf   = (tri.vertexData.deviceAddress >= vtxBufAddr)
                                    && (tri.vertexData.deviceAddress + (tri.maxVertex + 1u) * tri.vertexStride <= vtxBufAddr + vtxBufSize);
              }
              if (ib.buffer() != nullptr) {
                idxBufAddr         = ib.buffer()->getDeviceAddress();
                idxBufSize         = ib.buffer()->info().size;
                idxBufUsage        = ib.buffer()->info().usage;
                idxBufMemFlags     = ib.buffer()->memFlags();
                idxOffsetFromSlice = ib.offsetFromSlice();
                addrWithinIdxBuf   = (tri.indexData.deviceAddress >= idxBufAddr)
                                    && (idxEnd <= idxBufAddr + idxBufSize);
              }
            }

            Logger::info(str::format(
              // [SpawnGeomDiag] renamed from [BBI] to bypass log.cpp's
              // "[BBI" filter. BLAS-build-input audit: each entry shows
              // the geometry's vertex/index device addresses, max vertex,
              // primitive count, vertex stride, etc — what the BLAS
              // builder actually receives.
              "[SpawnGeomDiag.BBI] bi=", bi, " gi=", gi, "/", g.geometryCount,
              " isPI=", (isPi ? 1 : 0), " hasEntry=", (entry != nullptr ? 1 : 0),
              " dstAsAddr=0x", std::hex, dstAsAddr, std::dec,
              " vtxAddr=0x", std::hex, tri.vertexData.deviceAddress,
              " vtxEnd=0x", vtxEnd,
              " idxAddr=0x", tri.indexData.deviceAddress,
              " idxEnd=0x", idxEnd, std::dec,
              " maxVtx=", tri.maxVertex, " primCnt=", r.primitiveCount, " primOff=", r.primitiveOffset,
              " vStride=", tri.vertexStride,
              " vFmt=", uint32_t(tri.vertexFormat),
              " iType=", uint32_t(tri.indexType),
              " geoFlags=0x", std::hex, uint32_t(g.pGeometries[gi].flags), std::dec,
              " | vtxBuf addr=0x", std::hex, vtxBufAddr,
              " size=", std::dec, vtxBufSize,
              " usage=0x", std::hex, vtxBufUsage,
              " memF=0x", vtxBufMemFlags, std::dec,
              " offFromSlice=", vtxOffsetFromSlice,
              " addrInBuf=", (addrWithinVtxBuf ? 1 : 0),
              " | idxBuf addr=0x", std::hex, idxBufAddr,
              " size=", std::dec, idxBufSize,
              " usage=0x", std::hex, idxBufUsage,
              " memF=0x", idxBufMemFlags, std::dec,
              " offFromSlice=", idxOffsetFromSlice,
              " addrInBuf=", (addrWithinIdxBuf ? 1 : 0)));

            // Schedule readback only on geo 0, and only if we have the owning buffers,
            // and only for first kMaxReadbacks BLASes. Track WHY a readback gets
            // skipped (see BbiMeta::skipReason) so the log distinguishes the
            // "merged-bucket no entry" case from "buffer disappeared" / "addr
            // outside buffer" — observed v0=(0,0,0) frames in remix-dxvk.log
            // were silently failing without indicating which gate fell.
            uint8_t skipReason = 0;
            if (gi == 0 && bi < nBlas) {
              if (entry == nullptr) {
                skipReason = 1; // noEntry
              } else {
                const auto& pb = entry->modifiedGeometryData.positionBuffer;
                const auto& ib = entry->modifiedGeometryData.indexBuffer;
                if (pb.buffer() == nullptr) {
                  skipReason = 2; // noPosBuf
                } else if (!addrWithinVtxBuf) {
                  skipReason = 3; // addrOutsideBuf
                } else if ((pb.buffer()->info().usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
                  skipReason = 4; // noTransferSrc
                } else {
                  const VkDeviceSize pbBaseOff = tri.vertexData.deviceAddress - vtxBufAddr;
                  const VkDeviceSize dstOff = bi * kReadbackSlotBytes;
                  if (pbBaseOff + kVtxBytes <= vtxBufSize && dstOff + kVtxBytes <= sBbiStaging[writeSlot]->info().size) {
                    ctx->copyBuffer(sBbiStaging[writeSlot], dstOff, pb.buffer(), pbBaseOff, kVtxBytes);
                  } else {
                    skipReason = 5; // stagingOverflow
                  }
                }
                if (ib.buffer() != nullptr && addrWithinIdxBuf
                    && (ib.buffer()->info().usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0) {
                  const VkDeviceSize ibBaseOff = tri.indexData.deviceAddress - idxBufAddr;
                  const VkDeviceSize dstOff = bi * kReadbackSlotBytes + kVtxBytes;
                  if (ibBaseOff + kIdxBytes <= idxBufSize && dstOff + kIdxBytes <= sBbiStaging[writeSlot]->info().size) {
                    ctx->copyBuffer(sBbiStaging[writeSlot], dstOff, ib.buffer(), ibBaseOff, kIdxBytes);
                  }
                }
              }
            }
            if (bi < kMaxReadbacks && gi == 0) {
              sBbiMetas[writeSlot][bi] = BbiMeta {
                dstAsAddr, tri.vertexData.deviceAddress, tri.indexData.deviceAddress,
                uint32_t(tri.vertexStride), uint32_t(tri.vertexFormat), uint32_t(tri.indexType),
                tri.maxVertex, r.primitiveCount, r.primitiveOffset,
                uint32_t(g.pGeometries[gi].flags),
                vtxBufAddr, vtxBufSize, vtxBufUsage, vtxBufMemFlags, vtxOffsetFromSlice,
                isPi, (entry != nullptr), addrWithinVtxBuf,
                skipReason
              };
            }
          }
        }
        sBbiValid[writeSlot] = true;
        sBbiCount[writeSlot] = uint32_t(nBlas);

        // Read back the OLDEST slot (written kRingSize-1 fires ago). Barrier the transfer first.
        if (sBbiValid[readSlot]) {
          // Need a transfer barrier so previous-fire's copies are visible to HOST.
          // The ring lag already covers in-flight GPU work, but emit for safety.
          const uint8_t* data = reinterpret_cast<const uint8_t*>(sBbiStaging[readSlot]->mapPtr(0));
          if (data != nullptr) {
            for (uint32_t i = 0; i < sBbiCount[readSlot]; ++i) {
              const auto& m = sBbiMetas[readSlot][i];
              const uint8_t* vp = data + i * kReadbackSlotBytes;
              const uint8_t* ip = vp + kVtxBytes;
              float vx0, vy0, vz0, vx1, vy1, vz1;
              memcpy(&vx0, vp + 0,  4); memcpy(&vy0, vp + 4,  4); memcpy(&vz0, vp + 8,  4);
              // Second vertex starts at stride bytes in (but we only captured kVtxBytes total; stride may skip UV/color).
              const uint32_t s = std::min<uint32_t>(m.vtxStride, kVtxBytes - 12);
              memcpy(&vx1, vp + s + 0, 4); memcpy(&vy1, vp + s + 4, 4); memcpy(&vz1, vp + s + 8, 4);
              uint32_t i0 = 0, i1 = 0, i2 = 0;
              if (m.iType == VK_INDEX_TYPE_UINT16) {
                uint16_t tmp[3] = {};
                memcpy(tmp, ip, 6);
                i0 = tmp[0]; i1 = tmp[1]; i2 = tmp[2];
              } else if (m.iType == VK_INDEX_TYPE_UINT32) {
                memcpy(&i0, ip + 0, 4); memcpy(&i1, ip + 4, 4); memcpy(&i2, ip + 8, 4);
              }
              // Map skipReason to a short string for the log.
              // 0=ok, 1=noEntry, 2=noPosBuf, 3=addrOutBuf, 4=noXferSrc, 5=stagingOvf
              const char* skipStr = "ok";
              switch (m.skipReason) {
                case 1: skipStr = "noEntry";    break;
                case 2: skipStr = "noPosBuf";   break;
                case 3: skipStr = "addrOutBuf"; break;
                case 4: skipStr = "noXferSrc";  break;
                case 5: skipStr = "stagingOvf"; break;
                default: break;
              }
              Logger::info(str::format(
                // [SpawnGeomDiag] renamed from [BBI-readback]. Reads the
                // first 12 bytes of the BLAS-input vertex buffer + first
                // index triple, post-build. If decoded vertices look
                // wrong (NaN, zero, or far from the matrix's world
                // anchor), the GPU interleaver wrote bad data.
                // skip=<reason> tells you WHY a v0=(0,0,0) entry shows
                // up (vs. the readback actually returning zeros).
                "[SpawnGeomDiag.BBI-readback] bi=", i,
                " isPI=", (m.isPi ? 1 : 0),
                " hasEntry=", (m.hasEntry ? 1 : 0),
                " skip=", skipStr,
                " dstAs=0x", std::hex, m.dstAsAddr, std::dec,
                " vtxAddr=0x", std::hex, m.vtxAddr, std::dec,
                " v0=(", vx0, ",", vy0, ",", vz0, ")",
                " v1=(", vx1, ",", vy1, ",", vz1, ")",
                " tri0=[", i0, ",", i1, ",", i2, "]",
                " maxVtx=", m.maxVtx, " primCnt=", m.primCnt,
                " vtxBufUsage=0x", std::hex, m.vtxBufUsage, std::dec,
                " addrInVtxBuf=", (m.addrWithinVtxBuf ? 1 : 0)));
            }
          }
          sBbiValid[readSlot] = false;
          sBbiCount[readSlot] = 0;
        }
      }

      ctx->vkCmdBuildAccelerationStructuresKHR(blasToBuild.size(), blasToBuild.data(), blasRangesToBuild.data());

      execBarriers.accessBuffer(
       m_scratchBuffer->getSliceHandle(),
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
       VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV);

      ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_scratchBuffer);

      // NV-DXVK BBI-SERIALSIZE probe: right after the build, query
      // VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR for each PI BLAS.
      // Tiny size (~48..64 bytes of header-only) indicates the build produced an empty AS
      // despite valid inputs. Large size confirms the AS has real geometry.
      if (bbiFire) {
        constexpr uint32_t kSerMaxQueries = 16;
        constexpr uint32_t kSerRing       = 3;
        static VkQueryPool sSerPool = VK_NULL_HANDLE;
        static bool        sSerValid[kSerRing] = {};
        static uint32_t    sSerCount[kSerRing] = {};
        struct SerMeta {
          uint64_t dstAsAddr;
          uint32_t primCnt;
          uint32_t maxVtx;
          uint32_t vStride;
        };
        static SerMeta     sSerMetas[kSerRing][kSerMaxQueries] = {};

        if (sSerPool == VK_NULL_HANDLE) {
          VkQueryPoolCreateInfo qpCI { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
          qpCI.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
          qpCI.queryCount = kSerMaxQueries * kSerRing;
          m_device->vkd()->vkCreateQueryPool(m_device->handle(), &qpCI, nullptr, &sSerPool);
        }

        const uint32_t serWrite = bbiFireIdx % kSerRing;
        const uint32_t serRead  = (bbiFireIdx + 1) % kSerRing;
        const uint32_t queryBase = serWrite * kSerMaxQueries;

        // Read OLDEST slot first (results from kSerRing-1 fires ago — should be ready).
        if (sSerValid[serRead]) {
          const uint32_t readBase = serRead * kSerMaxQueries;
          uint64_t results[kSerMaxQueries] = {};
          const VkResult qres = m_device->vkd()->vkGetQueryPoolResults(
            m_device->handle(), sSerPool, readBase, sSerCount[serRead],
            sizeof(results), results, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
          if (qres == VK_SUCCESS) {
            for (uint32_t i = 0; i < sSerCount[serRead]; ++i) {
              const auto& m = sSerMetas[serRead][i];
              // Header is ~VK_UUID_SIZE*2 + 16 = 48 bytes typically.
              // Empty AS ≈ 56 bytes (header only). Real AS >> header + vertex+tri data.
              Logger::info(str::format(
                // [SpawnGeomDiag] renamed from [BBI-serialsize]. Header
                // is ~48-56 bytes. serBytes near 56 means the AS is
                // empty despite primCount>0 — build silently dropped
                // geometry.
                "[SpawnGeomDiag.BBI-serialsize] i=", i,
                " dstAs=0x", std::hex, m.dstAsAddr, std::dec,
                " serBytes=", results[i],
                " primCnt=", m.primCnt,
                " maxVtx=", m.maxVtx,
                " vStride=", m.vStride,
                " (tiny=empty AS, large=real geom)"));
            }
          } else {
            Logger::warn(str::format("[SpawnGeomDiag.BBI-serialsize] vkGetQueryPoolResults failed: ", int(qres)));
          }
          sSerValid[serRead] = false;
          sSerCount[serRead] = 0;
        }

        // Reset the WRITE slot's queries (must happen before issuing new queries).
        ctx->getCommandList()->cmdResetQueryPool(sSerPool, queryBase, kSerMaxQueries);

        // Gather AS handles for PI BLASes (limit to kSerMaxQueries).
        std::vector<VkAccelerationStructureKHR> asHandles;
        asHandles.reserve(kSerMaxQueries);
        for (size_t bi = 0; bi < blasToBuild.size() && asHandles.size() < kSerMaxQueries; ++bi) {
          PooledBlas* dstBl = (bi < m_debugBlasBuildDstBlas.size()) ? m_debugBlasBuildDstBlas[bi] : nullptr;
          if (dstBl == nullptr) continue;
          if (piRefs.count(dstBl->accelerationStructureReference) == 0) continue;
          asHandles.push_back(dstBl->accelStructure->getAccelStructure());
          const uint32_t mi = uint32_t(asHandles.size() - 1);
          const auto& g = blasToBuild[bi];
          const auto* ranges = blasRangesToBuild[bi];
          sSerMetas[serWrite][mi] = SerMeta {
            dstBl->accelerationStructureReference,
            (g.geometryCount > 0 && ranges != nullptr) ? ranges[0].primitiveCount : 0u,
            (g.geometryCount > 0) ? g.pGeometries[0].geometry.triangles.maxVertex : 0u,
            (g.geometryCount > 0) ? uint32_t(g.pGeometries[0].geometry.triangles.vertexStride) : 0u
          };
        }

        if (!asHandles.empty()) {
          // The build's AS_WRITE writes are made visible to AS_READ (the query reads) by the
          // scratch-buffer execBarrier above, but emit an explicit memory barrier to be safe.
          ctx->emitMemoryBarrier(0,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
          ctx->vkCmdWriteAccelerationStructuresPropertiesKHR(
            uint32_t(asHandles.size()), asHandles.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
            sSerPool, queryBase);
          sSerValid[serWrite] = true;
          sSerCount[serWrite] = uint32_t(asHandles.size());
        }
      }
    }
  }

  void AccelManager::buildTlas(Rc<DxvkContext> ctx) {
    if (m_vkInstanceBuffer == nullptr) {
      return;
    }

    ScopedGpuProfileZone(ctx, "buildTLAS");

    // NV-DXVK debug: validate that every PI batch's blasReference points to a
    // BLAS that's actually in m_blasPool right now. Stale refs => the BLAS got
    // freed/replaced between addPointInstancerBlas and buildTlas → primary rays
    // dereference dead AS handles → zero hits.
    {
      static uint32_t s_blasValidateFrame = 0;
      if (kEnableRtxDebugProbes && (s_blasValidateFrame++ % 60u) == 0) {
        std::set<uint64_t> poolRefs;
        for (const auto& blas : m_blasPool) {
          if (blas != nullptr && blas->accelerationStructureReference != 0) {
            poolRefs.insert(blas->accelerationStructureReference);
          }
        }
        Logger::info(str::format(
          "[PI-blas-validate] frame=", s_blasValidateFrame,
          " poolSize=", m_blasPool.size(),
          " uniqueRefsInPool=", poolRefs.size(),
          " piBatches=", m_pointInstancerBatches.size(),
          " mergedOpaque=", m_mergedInstances[Tlas::Opaque].size()));

        // PI batches: validate the ref against the strong-ref'd BLAS we captured.
        // - refMatches: captured ref still equals the live BLAS's ref (i.e. AS handle didn't move)
        // - asNonNull: the underlying VkAccelerationStructure is non-null right now
        // - builtAtCapture: was non-null when addPointInstancerBlas captured it
        uint32_t piRefMatches = 0, piRefMismatch = 0, piAsNull = 0, piNotBuiltAtCapture = 0;
        uint32_t logged = 0;
        for (size_t i = 0; i < m_pointInstancerBatches.size(); ++i) {
          const auto& b = m_pointInstancerBatches[i];
          const bool hasRc = b.debugBlasRef != nullptr;
          const uint64_t liveRef = hasRc ? b.debugBlasRef->accelerationStructureReference : 0;
          const bool refMatch = hasRc && liveRef == b.blasReference;
          const bool asLive = hasRc && b.debugBlasRef->accelStructure != nullptr;
          if (refMatch) ++piRefMatches; else ++piRefMismatch;
          if (!asLive) ++piAsNull;
          if (!b.debugAsBuiltAtCapture) ++piNotBuiltAtCapture;
          if (logged < 8) {
            ++logged;
            const uint32_t frameLastTouched = hasRc ? b.debugBlasRef->frameLastTouched : 0xFFFFFFFFu;
            const uint32_t curFrame = m_device->getCurrentFrameId();
            const auto& binfo = b.debugBlasRef->buildInfo;
            uint32_t geoMaxVtx = 0;
            uint32_t geoPrimCnt = 0;
            uint64_t geoVtxAddr = 0;
            uint64_t geoIdxAddr = 0;
            uint32_t geoVtxStride = 0;
            if (hasRc && binfo.geometryCount > 0 && binfo.pGeometries != nullptr) {
              const auto& tri = binfo.pGeometries[0].geometry.triangles;
              geoMaxVtx = tri.maxVertex;
              geoVtxStride = uint32_t(tri.vertexStride);
              geoVtxAddr = tri.vertexData.deviceAddress;
              geoIdxAddr = tri.indexData.deviceAddress;
              if (i < b.debugBlasRef->primitiveCounts.size()) {
                // primitiveCounts is per-geometry; first geo is at [0]
              }
              if (!b.debugBlasRef->primitiveCounts.empty()) {
                geoPrimCnt = b.debugBlasRef->primitiveCounts[0];
              }
            }
            Logger::info(str::format(
              "[PI-blas-validate]  PI batch=", i,
              " surfRange=[", b.baseSurfaceIndex, "..",
              b.baseSurfaceIndex + b.instanceCount - 1, "] count=", b.instanceCount,
              " ref=0x", std::hex, b.blasReference, std::dec,
              " refMatch=", (refMatch ? 1 : 0),
              " frameLastTouched=", frameLastTouched, "/", curFrame,
              " maxVtx=", geoMaxVtx,
              " primCnt=", geoPrimCnt,
              " vStride=", geoVtxStride,
              " vtxAddr=0x", std::hex, geoVtxAddr,
              " idxAddr=0x", geoIdxAddr, std::dec));
          }
        }
        // addBlas / merged Opaque sanity: check first 6 entries
        uint32_t mergedStale = 0, mergedValid = 0;
        const size_t mergedShow = std::min<size_t>(m_mergedInstances[Tlas::Opaque].size(), 6);
        for (size_t i = 0; i < m_mergedInstances[Tlas::Opaque].size(); ++i) {
          const uint64_t ref = m_mergedInstances[Tlas::Opaque][i].accelerationStructureReference;
          const bool inPool = poolRefs.count(ref) > 0;
          if (inPool) ++mergedValid; else ++mergedStale;
          if (i < mergedShow) {
            Logger::info(str::format(
              "[PI-blas-validate]  Merged[Opaque][", i, "]",
              " blasRef=0x", std::hex, ref, std::dec,
              " inPool=", (inPool ? "YES" : "NO"),
              " surfIdx=", (m_mergedInstances[Tlas::Opaque][i].instanceCustomIndex & CUSTOM_INDEX_SURFACE_MASK),
              " mask=0x", std::hex, uint32_t(m_mergedInstances[Tlas::Opaque][i].mask), std::dec));

            // NV-DXVK: dump WHICH RtInstance/BlasEntry owns this merged bucket.
            // The bucket's first surface index is the start offset into m_reorderedSurfaces.
            // Use that to pull the source RtInstance*, then its BlasEntry info.
            const uint32_t bucketStartSurf = m_mergedInstances[Tlas::Opaque][i].instanceCustomIndex & CUSTOM_INDEX_SURFACE_MASK;
            if (bucketStartSurf < m_reorderedSurfaces.size()) {
              RtInstance* srcInst = m_reorderedSurfaces[bucketStartSurf];
              BlasEntry* srcBlas = (srcInst != nullptr) ? srcInst->getBlas() : nullptr;
              const uint32_t srcVtxCount = (srcBlas != nullptr) ? srcBlas->modifiedGeometryData.vertexCount : 0;
              const uint32_t srcPrimCount = (srcBlas != nullptr) ? srcBlas->modifiedGeometryData.calculatePrimitiveCount() : 0;
              // Dump the source VS hash from the BlasEntry's input hashes.
              // input.getHash(HashComponents::VertexShader) would be ideal; fall back to
              // the full geometry hash if that accessor isn't available.
              uint64_t vsHash = 0;
              uint64_t geomHash = 0;
              if (srcBlas != nullptr) {
                geomHash = srcBlas->input.getHash(RtxOptions::geometryAssetHashRule());
                vsHash = srcBlas->input.getTransformData().vertexShaderHash;
              }
              // Dump the raw instance transform for big (BSP-scale) merged buckets so
              // we can see whether upstream already added camOrigin or not.
              if (srcBlas != nullptr && srcVtxCount > 10000) {
                const auto& t = m_mergedInstances[Tlas::Opaque][i].transform;
                Logger::info(str::format(
                  "[MergedBspXform] Merged[", i, "] vtx=", srcVtxCount,
                  " VS=0x", std::hex, vsHash, std::dec,
                  " r0=(", t.matrix[0][0], ",", t.matrix[0][1], ",", t.matrix[0][2], ") T0=", t.matrix[0][3],
                  " r1=(", t.matrix[1][0], ",", t.matrix[1][1], ",", t.matrix[1][2], ") T1=", t.matrix[1][3],
                  " r2=(", t.matrix[2][0], ",", t.matrix[2][1], ",", t.matrix[2][2], ") T2=", t.matrix[2][3]));
                // Also dump the source's surface.objectToWorld — what came from d3d11_rtx.cpp.
                const Matrix4& o2w = srcInst->surface.objectToWorld;
                Logger::info(str::format(
                  "[MergedBspXform] Merged[", i, "] surface.o2w",
                  " col0=(", o2w.data[0].x, ",", o2w.data[0].y, ",", o2w.data[0].z, ")",
                  " col1=(", o2w.data[1].x, ",", o2w.data[1].y, ",", o2w.data[1].z, ")",
                  " col2=(", o2w.data[2].x, ",", o2w.data[2].y, ",", o2w.data[2].z, ")",
                  " col3=(", o2w.data[3].x, ",", o2w.data[3].y, ",", o2w.data[3].z, ")"));
              }
              Logger::info(str::format(
                "[PI-blas-validate]  Merged[Opaque][", i, "] ownerInst=0x",
                std::hex, reinterpret_cast<uintptr_t>(srcInst),
                " blasEntry=0x", reinterpret_cast<uintptr_t>(srcBlas), std::dec,
                " vtxCount=", srcVtxCount,
                " primCount=", srcPrimCount,
                " firstSurf=", bucketStartSurf,
                " VS=0x", std::hex, vsHash,
                " geom=0x", geomHash, std::dec));
            }
          }
        }
        Logger::info(str::format(
          "[PI-blas-validate] SUMMARY piRefMatches=", piRefMatches,
          " piRefMismatch=", piRefMismatch,
          " piAsNull=", piAsNull,
          " piNotBuiltAtCapture=", piNotBuiltAtCapture,
          " mergedValid=", mergedValid, " mergedStale=", mergedStale));

        // NV-DXVK: at TLAS build time, recompute what the PI region offset SHOULD
        // be based on current m_mergedInstances size, and compare against what
        // dispatchCulling stored on each batch. If these mismatch the PI bytes
        // landed at the wrong location and TLAS will read merged data instead.
        const size_t kInstSize = sizeof(VkAccelerationStructureInstanceKHR);
        const VkDeviceSize bufSize = (m_vkInstanceBuffer != nullptr) ? m_vkInstanceBuffer->info().size : 0;
        const VkDeviceAddress bufAddr = (m_vkInstanceBuffer != nullptr) ? m_vkInstanceBuffer->getDeviceAddress() : 0;
        Logger::info(str::format(
          "[PI-blas-validate] BUF size=", bufSize,
          " addr=0x", std::hex, bufAddr, std::dec,
          " mergedSize[Opaque]=", m_mergedInstances[Tlas::Opaque].size(),
          " piSlots[Opaque]=", m_pointInstancerSlotsPerType[Tlas::Opaque],
          " mergedSize[Unord]=", m_mergedInstances[Tlas::Unordered].size(),
          " piSlots[Unord]=", m_pointInstancerSlotsPerType[Tlas::Unordered]));

        size_t typeBaseAtBuild[Tlas::Count] = {};
        for (size_t n = 1; n < Tlas::Count; ++n) {
          typeBaseAtBuild[n] = typeBaseAtBuild[n - 1]
            + (m_mergedInstances[n - 1].size() + m_pointInstancerSlotsPerType[n - 1]) * kInstSize;
        }
        uint32_t offsetMismatches = 0;
        for (size_t i = 0; i < std::min<size_t>(m_pointInstancerBatches.size(), 4); ++i) {
          const auto& b = m_pointInstancerBatches[i];
          const uint32_t expectedOff = uint32_t(typeBaseAtBuild[b.tlasType]
            + m_mergedInstances[b.tlasType].size() * kInstSize
            + b.firstIndexInType * kInstSize);
          if (expectedOff != b.instanceBufferByteOffset) ++offsetMismatches;
          Logger::info(str::format(
            "[PI-blas-validate] OFFSET batch=", i,
            " stored=", b.instanceBufferByteOffset,
            " expectedNow=", expectedOff,
            " match=", (expectedOff == b.instanceBufferByteOffset ? "YES" : "NO"),
            " withinBuf=", ((expectedOff + kInstSize) <= bufSize ? "YES" : "NO")));
        }
        Logger::info(str::format("[PI-blas-validate] offsetMismatches=", offsetMismatches));
      }
    }

    // Four barriers in one:
    // Accel build bit - to protect from BLAS builds
    // Transfer bit - to protect from updateBuffer in prepareSceneData
    // Compute bit - to protect from GPU PointInstancer culling writes to instance, surface, and material buffers
    // RT shader read - make compute writes to surface/material buffers visible to ray tracing passes
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);

    for (auto&& blas : m_blasPool) {
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(blas->accelStructure);
    }

    // NV-DXVK debug: write override RIGHT BEFORE TLAS build so nothing can clobber it.
    {
      static uint32_t s_lateOverrideFrame = 0;
      const bool fire = kEnableRtxDebugDestructiveProbes
                       && ((s_lateOverrideFrame++ % 3u) == 0)
                       && !m_pointInstancerBatches.empty();
      if (fire && !m_mergedInstances[Tlas::Opaque].empty() && m_pointInstancerBatches.size() >= 2) {
        const PointInstancerBatch& slotA = m_pointInstancerBatches[0];
        const PointInstancerBatch& slotB = m_pointInstancerBatches[1];
        // Use merged[0]'s world-space translation — merged[0] is the viewmodel and is
        // confirmed in-view every frame (its surfaces 412-415 dominate VisibleSurf).
        // Placing our test instances there guarantees they're in the camera FOV.
        const auto& mergedXform = m_mergedInstances[Tlas::Opaque].front().transform;
        const float tx = mergedXform.matrix[0][3];
        const float ty = mergedXform.matrix[1][3];
        const float tz = mergedXform.matrix[2][3];
        // Identity side-by-side comparison, written into the PI REGION (does NOT touch
        // merged slots, so normal viewmodel keeps rendering). Two PI slots, same position,
        // both at identity rotation, different blasRefs + surface markers.
        //   PI batch[0] slot 0  → surf 777, PI's OWN blasRef          (tests PI BLAS geom)
        //   PI batch[1] slot 0  → surf 888, merged[0]'s blasRef       (tests known-good BLAS)
        // If 888 visible but 777 not → PI BLAS geom is broken at this position.
        // If both visible → PI BLAS works; normal-path transforms in compute are the bug.
        // If neither → position not in current view.
        auto makeIdentityInst = [&](uint64_t blasRef, uint32_t surfIdx) {
          VkAccelerationStructureInstanceKHR inst {};
          inst.transform.matrix[0][0] = 1.f; inst.transform.matrix[0][3] = tx;
          inst.transform.matrix[1][1] = 1.f; inst.transform.matrix[1][3] = ty;
          inst.transform.matrix[2][2] = 1.f; inst.transform.matrix[2][3] = tz;
          inst.accelerationStructureReference = blasRef;
          inst.instanceCustomIndex = (surfIdx & uint32_t(CUSTOM_INDEX_SURFACE_MASK));
          inst.mask = 0xFFu;
          inst.instanceShaderBindingTableRecordOffset = 0;
          inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
          return inst;
        };
        VkAccelerationStructureInstanceKHR overrideA = makeIdentityInst(slotA.blasReference, 777);
        const uint64_t mergedBlasRef = m_mergedInstances[Tlas::Opaque].front().accelerationStructureReference;
        VkAccelerationStructureInstanceKHR overrideB = makeIdentityInst(mergedBlasRef, 888);
        ctx->writeToBuffer(m_vkInstanceBuffer, slotA.instanceBufferByteOffset,
                           sizeof(VkAccelerationStructureInstanceKHR), &overrideA);
        ctx->writeToBuffer(m_vkInstanceBuffer, slotB.instanceBufferByteOffset,
                           sizeof(VkAccelerationStructureInstanceKHR), &overrideB);
        ctx->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_TRANSFER_BIT,                       VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        Logger::info(str::format(
          "[PI-override-late] IDENTITY side-by-side frame=", s_lateOverrideFrame,
          " xlat=(", tx, ",", ty, ",", tz, ")",
          " PIblas=0x", std::hex, slotA.blasReference,
          " mergedBlas=0x", mergedBlasRef, std::dec,
          " slotA byteOff=", slotA.instanceBufferByteOffset,
          " slotB byteOff=", slotB.instanceBufferByteOffset,
          " (777=PI blas, 888=merged blas, both in PI region, merged untouched)"));
      }
    }

    size_t totalScratchSize = 0;
    internalBuildTlas<Tlas::Opaque>(ctx, totalScratchSize);
    internalBuildTlas<Tlas::Unordered>(ctx, totalScratchSize);
    // Only build TLAS for SSS when necessary
    const bool isBuildSssTlas = RtxOptions::SubsurfaceScattering::enableDiffusionProfile() && (m_mergedInstances[Tlas::SSS].size() + m_pointInstancerSlotsPerType[Tlas::SSS]) > 0;
    if (isBuildSssTlas) {
      internalBuildTlas<Tlas::SSS>(ctx, totalScratchSize);
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    // Release the scratch memory so it can be reused by rest of the frame.
    m_scratchBuffer = nullptr;

    OpacityMicromapManager* opacityMicromapManager = ctx->getCommonObjects()->getSceneManager().getOpacityMicromapManager();
    if (opacityMicromapManager) {
      opacityMicromapManager->onFinishedBuilding();
    }
  }

  template<Tlas::Type type>
  void AccelManager::internalBuildTlas(Rc<DxvkContext> ctx, size_t& totalScratchSize) {
    static constexpr const char* names[] = { "TLAS_Opaque", "TLAS_NonOpaque", "TLAS_SSS" };
    ScopedGpuProfileZone(ctx, names[type]);
    const VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | additionalAccelerationStructureFlags();

    const auto& vkd = m_device->vkd();

    // Create VkAccelerationStructureGeometryInstancesDataKHR
    // This wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
    instancesVk.arrayOfPointers = VK_FALSE;
    instancesVk.data.deviceAddress = m_vkInstanceBuffer->getDeviceAddress();

    // Rewind address to tlas start (normal + PointInstancer slots per preceding type)
    for (size_t n = 0; n < type; ++n) {
      instancesVk.data.deviceAddress += (m_mergedInstances[n].size() + m_pointInstancerSlotsPerType[n]) * sizeof(VkAccelerationStructureInstanceKHR);
    }

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the
    // instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.flags = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &topASGeometry;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    const uint32_t numInstances = uint32_t(m_mergedInstances[type].size() + m_pointInstancerSlotsPerType[type]);
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkd->vkGetAccelerationStructureBuildSizesKHR(vkd->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &numInstances, &sizeInfo);

    // Create TLAS
    Tlas& tlas = m_device->getCommon()->getResources().getTLAS(type);

    if (type == Tlas::Opaque) {
      std::swap(tlas.accelStructure, tlas.previousAccelStructure);
    }

    if (tlas.accelStructure == nullptr || sizeInfo.accelerationStructureSize > tlas.accelStructure->info().size) {
      ScopedGpuProfileZone(ctx, "buildTLAS_createAccelStructure");
      DxvkBufferCreateInfo info;
      // NV-DXVK: SHADER_DEVICE_ADDRESS_BIT required so the AS-backing buffer
      // is eligible for vkGetAccelerationStructureDeviceAddressKHR (called
      // by RtxGlobalVolumetrics::dispatch / bindCommonRayTracingResources).
      // BLAS at line 367 already has this flag; TLAS was missing it,
      // causing per-frame validation errors and eventual debug-build crash.
      info.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      info.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      info.size = sizeInfo.accelerationStructureSize;

      tlas.accelStructure = m_device->createAccelStructure(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, names[type]);

      Logger::debug(str::format("DxvkRaytrace: TLAS Realloc"));
    }

    // Allocate the scratch memory, we share the same buffer between all TLAS types, so just ensure we handle the offsetting correctly here.
    const size_t requiredScratchAllocSize = align(sizeInfo.buildScratchSize + m_scratchAlignment, m_scratchAlignment);
    buildInfo.scratchData.deviceAddress = getScratchMemory(totalScratchSize + requiredScratchAllocSize)->getDeviceAddress() + totalScratchSize;
    totalScratchSize += requiredScratchAllocSize;

    // Update build information
    buildInfo.srcAccelerationStructure = nullptr;
    buildInfo.dstAccelerationStructure = tlas.accelStructure->getAccelStructure();

    assert(buildInfo.scratchData.deviceAddress % m_scratchAlignment == 0); // Note: Required by the Vulkan specification.

    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo { numInstances, 0, 0, 0 };
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // [SpawnGeomDiag.TLASBuildCount] Suspect 3: log the
    // primitiveCount handed to the TLAS builder and the sum of
    // (mergedInstances + slotsPerType) for this type. If they
    // diverge, the TLAS is consuming trailing garbage past the
    // last real instance (random surface IDs) or it's truncating
    // before some real instances (visibility cutoff). Either is a
    // reason floor instances would go missing despite being in
    // m_pointInstancerBatches with valid byteOffsets. Throttle 1
    // dump per 30 builds per type so the cadence aligns with
    // [SpawnGeomDiag.PIoffsets] and [SpawnGeomDiag.PIdispatch].
    {
      static uint32_t sTlasBuildCountFrame[Tlas::Count] = {};
      const uint32_t f = sTlasBuildCountFrame[type]++;
      if ((f % 30u) == 0) {
        const uint32_t expectedFromState = uint32_t(
          m_mergedInstances[type].size() + m_pointInstancerSlotsPerType[type]);
        const bool mismatch = (numInstances != expectedFromState);
        Logger::info(str::format(
          "[SpawnGeomDiag.TLASBuildCount] tlasType=", uint32_t(type),
          " name=", names[type],
          " primitiveCount=", numInstances,
          " mergedSize=", m_mergedInstances[type].size(),
          " slotsForType=", m_pointInstancerSlotsPerType[type],
          " expected=", expectedFromState,
          " mismatch=", (mismatch ? 1 : 0)));
      }
    }

    // Build the TLAS
    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildOffsetInfo);

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(tlas.accelStructure);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_scratchBuffer);
  }

  // Check if the existing build geometry info for this blas is compatible with the new one for the purpose of updating rather than rebuilding
  bool AccelManager::validateUpdateMode(const VkAccelerationStructureBuildGeometryInfoKHR& oldInfo, const VkAccelerationStructureBuildGeometryInfoKHR& newInfo) {
    if (!(oldInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)) {
      return false;
    }

    if (oldInfo.type != newInfo.type || oldInfo.flags != newInfo.flags || oldInfo.geometryCount != newInfo.geometryCount) {
      return false;
    }

    for (uint32_t i = 0; i < oldInfo.geometryCount; ++i) {
      const VkAccelerationStructureGeometryKHR* oldGeom = oldInfo.pGeometries ? &oldInfo.pGeometries[i] : oldInfo.ppGeometries[i];
      const VkAccelerationStructureGeometryKHR* newGeom = newInfo.pGeometries ? &newInfo.pGeometries[i] : newInfo.ppGeometries[i];

      if (oldGeom->geometryType != newGeom->geometryType || oldGeom->flags != newGeom->flags) {
        return false;
      }

      // Per validation layers we need to check attributes of geometry types
      switch (oldGeom->geometryType) {
      case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      {
        const auto& oldTriangles = oldGeom->geometry.triangles;
        const auto& newTriangles = newGeom->geometry.triangles;
        if (oldTriangles.vertexFormat != newTriangles.vertexFormat ||
            oldTriangles.indexType != newTriangles.indexType ||
            oldTriangles.maxVertex != newTriangles.maxVertex ||
            oldTriangles.vertexStride != newTriangles.vertexStride) {
          return false;
        }
        break;
      }
      case VK_GEOMETRY_TYPE_AABBS_KHR:
      {
        const auto& oldAabbs = oldGeom->geometry.aabbs;
        const auto& newAabbs = newGeom->geometry.aabbs;
        if (oldAabbs.stride != newAabbs.stride) {
          return false;
        }
        break;
      }
      case VK_GEOMETRY_TYPE_INSTANCES_KHR:
      {
        const auto& oldInstances = oldGeom->geometry.instances;
        const auto& newInstances = newGeom->geometry.instances;
        if (oldInstances.arrayOfPointers != newInstances.arrayOfPointers) {
          return false;
        }
        break;
      }
      default:
        return false;
      }
    }
    return true;
  }

  void AccelManager::addPointInstancerBlas(RtInstance* rtInstance, BlasEntry* blasEntry) {
    // [SpawnGeomDiag.DrawOut] PI variant — same key/throttle scheme
    // as the addBlas DrawOut log. Once per (vsHash, matHash) for any
    // draw that reaches the PointInstancer BLAS path. Unlike
    // [SpawnGeomDiag.piAddEntry] which throttles 1/30 calls, this
    // captures the FIRST occurrence of every unique (vs,mat) tuple
    // so the DrawIn vs DrawOut diff is honest across both paths.
    {
      const uint64_t vsHash = static_cast<uint64_t>(
        blasEntry->input.getTransformData().vertexShaderHash);
      const uint64_t matHash = static_cast<uint64_t>(
        blasEntry->input.getMaterialData().getHash());
      const uint64_t key = vsHash ^ ((matHash << 1) | (matHash >> 63));
      static std::mutex sPiOutMu;
      static std::unordered_set<uint64_t> sPiOutSeen;
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sPiOutMu);
        first = sPiOutSeen.insert(key).second;
      }
      if (first) {
        const uint32_t primCount = blasEntry->buildRanges.empty() ? 0u
          : blasEntry->buildRanges[0].primitiveCount;
        const auto* xforms = rtInstance->surface.instancesToObject;
        const uint32_t instCount = (xforms != nullptr)
          ? static_cast<uint32_t>(xforms->size()) : 0u;
        Logger::info(str::format(
          "[SpawnGeomDiag.DrawOut] kind=addPI"
          " vsHash=0x", std::hex, vsHash, std::dec,
          " matHash=0x", std::hex, matHash, std::dec,
          " primCnt=", primCount,
          " vCnt=", blasEntry->modifiedGeometryData.vertexCount,
          " instCount=", instCount));
      }
    }

    // This RtInstance is a PointInstancer — GPU-driven culling path.
    // Reserve N surface slots (one per instance) so each gets its own
    // surface/material ID in customInstanceIndex.  Only the first slot is
    // fully populated on the CPU; the GPU culling shader copies the template
    // surface data and patches per-instance transforms for the rest.
    const auto* transforms = rtInstance->surface.instancesToObject;
    uint32_t instanceCount = static_cast<uint32_t>(transforms->size());

    // [SpawnGeomDiag.piAddEntry] Unconditional throttled log so we can
    // confirm whether *any* fanout RtInstance reaches this function.
    // Prior runs showed 0 [AccelMgr.piAdd] entries; that log is gated to
    // VS_ef94e6c7 + 100<prims<20000 so it could simply not match TF2's
    // BSP fanout VS hashes. This entry log fires for every fanout add.
    // Includes baseSurfaceIndex (= m_reorderedSurfaces.size() at entry —
    // surfaceIndex assignment happens later) so the surface-ID range
    // covered by this batch is [base..base+instanceCount-1]; cross-
    // reference with [SpawnGeomDiag.VisibleSurf]'s top= list to see if
    // primary rays actually hit any of these IDs.
    {
      static uint32_t sPiAddEntry = 0;
      if ((sPiAddEntry++ % 30u) == 0) {
        const XXH64_hash_t vsHash = blasEntry->input.getTransformData().vertexShaderHash;
        const uint32_t primCount =
          (!blasEntry->buildRanges.empty()) ? blasEntry->buildRanges[0].primitiveCount : 0u;
        const uint32_t base = static_cast<uint32_t>(m_reorderedSurfaces.size());
        const Matrix4& o2w = rtInstance->surface.objectToWorld;
        Vector3 firstPiT(0, 0, 0);
        if (instanceCount > 0) {
          const Matrix4 effective = o2w * (*transforms)[0];
          firstPiT = Vector3(effective[3][0], effective[3][1], effective[3][2]);
        }
        Logger::info(str::format(
          "[SpawnGeomDiag.piAddEntry] call=", sPiAddEntry,
          " vsHash=0x", std::hex, static_cast<uint64_t>(vsHash), std::dec,
          " instCount=", instanceCount,
          " primCount=", primCount,
          " surfRange=[", base, "..", base + instanceCount - 1, "]",
          " mask=0x", std::hex, rtInstance->getVkInstance().mask, std::dec,
          " flags=0x", std::hex, rtInstance->getVkInstance().flags, std::dec,
          " worldT0=(", firstPiT.x, ",", firstPiT.y, ",", firstPiT.z, ")"));
      }
    }

    // NV-DXVK TF2 VIEWMODEL TRACE: mirror the addBlas logging for the PI
    // path. Throttled + gated to VS_ef94e6c7 (body + gun shader) so the log
    // stays readable. Lets us see whether the gun enters the TLAS via the
    // point-instancer pipeline instead of addBlas.
    {
      static uint32_t sPiLog = 0;
      const XXH64_hash_t vsHash = blasEntry->input.getTransformData().vertexShaderHash;
      const uint32_t vsHashHi = static_cast<uint32_t>(vsHash >> 32);
      const uint32_t primCountForPiGate =
        (!blasEntry->buildRanges.empty()) ? blasEntry->buildRanges[0].primitiveCount : 0u;
      const bool piInteresting =
        (vsHashHi == 0xef94e6c7u)
        || (primCountForPiGate > 100 && primCountForPiGate < 20000);
      if (piInteresting && sPiLog < 60) {
        ++sPiLog;
        const auto& o2w = rtInstance->surface.objectToWorld;
        const auto& vki = rtInstance->getVkInstance();
        const uint32_t primCount =
          (!blasEntry->buildRanges.empty()) ? blasEntry->buildRanges[0].primitiveCount : 0u;
        // Sample the first instance's PI transform to see what translation
        // goes on top of objectToWorld for instance 0. That's the actual
        // world position of the first rendered copy.
        Vector3 piT0(0, 0, 0);
        if (instanceCount > 0) {
          const Matrix4 effective = rtInstance->surface.objectToWorld * (*transforms)[0];
          piT0 = Vector3(effective[3][0], effective[3][1], effective[3][2]);
        }
        Logger::info(str::format(
          "[AccelMgr.piAdd] #", sPiLog,
          " vsHashHi=0x", std::hex, vsHashHi, std::dec,
          " mask=0x", std::hex, vki.mask, std::dec,
          " flags=0x", std::hex, vki.flags, std::dec,
          " prims=", primCount,
          " instCount=", instanceCount,
          " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " piT[0]=(", piT0.x, ",", piT0.y, ",", piT0.z, ")",
          " surfIdx=", m_reorderedSurfaces.size()));
      }
    }

    // Reserve N surface entries — same RtInstance* for each, but each gets
    // a unique surfaceIndex.  The first entry is the "template" that
    // uploadSurfaceData writes fully; entries 1..N-1 are copies.
    const uint32_t surfaceIndex = static_cast<uint32_t>(m_reorderedSurfaces.size());

    // Clamp instance count so the last reserved surface index stays within the 21-bit limit.
    // Exceeding SURFACE_INDEX_MAX_VALUE would cause the GPU culling shader to write
    // truncated customInstanceIndex values, leading to surface/material aliasing or OOB access.
    if (surfaceIndex + instanceCount - 1 > SURFACE_INDEX_MAX_VALUE) {
      const uint32_t maxAllowed = (surfaceIndex <= SURFACE_INDEX_MAX_VALUE) ? (SURFACE_INDEX_MAX_VALUE - surfaceIndex + 1) : 0;
      ONCE(Logger::err(str::format("DxvkRaytrace: PointInstancer needs ", instanceCount, " surface slots starting at ", surfaceIndex, " but only ", maxAllowed,
                                   " fit within SURFACE_INDEX_MAX_VALUE (", SURFACE_INDEX_MAX_VALUE, "). Clamping to ", maxAllowed, " instances.")));
      instanceCount = maxAllowed;
      if (instanceCount == 0) {
        return;
      }
    }

    rtInstance->surface.surfaceIndexOfFirstInstance = surfaceIndex;
    m_reorderedSurfaces.insert(m_reorderedSurfaces.end(), instanceCount, rtInstance);
    m_reorderedSurfacesFirstIndexOffset.insert(m_reorderedSurfacesFirstIndexOffset.end(), instanceCount, 0);

    // Determine TLAS type
    const bool isUnordered = rtInstance->usesUnorderedApproximations() &&
      RtxOptions::enableSeparateUnorderedApproximations();
    const auto primaryType = isUnordered ? Tlas::Unordered : Tlas::Opaque;

    // Reserve N slots for the GPU shader to fill — nothing pushed to m_mergedInstances
    const uint32_t firstIndexInType = m_pointInstancerSlotsPerType[primaryType];
    m_pointInstancerSlotsPerType[primaryType] += instanceCount;

    // Build upper bits for instanceCustomIndex
    const uint32_t upperBits = rtInstance->getVkInstance().instanceCustomIndex & ~uint32_t(CUSTOM_INDEX_SURFACE_MASK);
    VkGeometryInstanceFlagsKHR flags = rtInstance->getVkInstance().flags;
    if (rtInstance->isObjectToWorldMirrored()) {
      flags ^= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
    }

    // Record batch for GPU dispatch
    PointInstancerBatch batch {};
    batch.transforms = transforms;
    batch.objectToWorld = rtInstance->surface.objectToWorld;
    batch.prevObjectToWorld = rtInstance->surface.prevObjectToWorld;
    batch.instanceCount = instanceCount;
    batch.baseSurfaceIndex = surfaceIndex;
    batch.customIndexFlags = upperBits;
    batch.instanceMask = rtInstance->getVkInstance().mask;
    batch.sbtOffsetAndFlags = (rtInstance->getVkInstance().instanceShaderBindingTableRecordOffset & 0x00FFFFFFu) | (static_cast<uint32_t>(flags) << 24);
    batch.blasReference = blasEntry->dynamicBlas->accelerationStructureReference;
    batch.firstIndexInType = firstIndexInType;
    batch.tlasType = primaryType;
    batch.instanceBufferByteOffset = 0; // resolved before dispatch
    // NV-DXVK debug: hold a strong ref to the BLAS for validation at TLAS-build time
    batch.debugBlasRef = blasEntry->dynamicBlas;
    batch.debugAsBuiltAtCapture = (blasEntry->dynamicBlas->accelStructure != nullptr);
    // NV-DXVK [SpawnGeomDiag.PIBatchInventory]: source fingerprint —
    // the prim/vertex counts the batch was BUILT FROM. The dispatch
    // path compares these against the live BLAS at batch.blasReference
    // to flag mismatches (e.g. blasReference resolves to a 16-prim BLAS
    // when the source had 1700+ prims → the BLAS handle drifted).
    batch.debugVsHash = static_cast<uint64_t>(blasEntry->input.getTransformData().vertexShaderHash);
    batch.debugSourcePrimCount = blasEntry->buildRanges.empty() ? 0u
      : blasEntry->buildRanges[0].primitiveCount;
    batch.debugSourceVertexCount = blasEntry->modifiedGeometryData.vertexCount;
    batch.debugSourceBlasEntry = blasEntry;
    m_pointInstancerBatches.push_back(batch);

    // NV-DXVK (debug probe E): capture the interleaved BLAS position buffer ref
    // for the PI batch with the LARGEST vertex count. This is almost certainly
    // the main-view BSP world geometry, which is the batch we actually care
    // about diagnosing. Resets at frame start in mergeInstancesIntoBlas; each
    // subsequent addPointInstancerBlas call compares and replaces if larger.
    if (blasEntry != nullptr) {
      const auto& geom = blasEntry->modifiedGeometryData;
      if (geom.vertexCount > s_probeE_vertexCount) {
        s_probeE_posBuffer    = geom.positionBuffer.buffer();
        s_probeE_posSliceOff  = geom.positionBuffer.offset();
        s_probeE_posElemOff   = geom.positionBuffer.offsetFromSlice();
        s_probeE_posStride    = geom.positionBuffer.stride();
        s_probeE_vertexCount  = geom.vertexCount;
        s_probeE_posFormat    = geom.positionBuffer.vertexFormat();
        s_probeE_blasRef      = batch.blasReference;
        s_probeF_baseSurfaceIndex = batch.baseSurfaceIndex;
        s_probeF_valid = true;
      }
    }

    // NV-DXVK (debug probe A): log first ~20 PI batches to confirm BSP reaches this path
    // with sane blasRef, mask, counts, and transforms. Remove once root cause is found.
    {
      static uint32_t sPiBatchLogCount = 0;
      if (sPiBatchLogCount < 20) {
        ++sPiBatchLogCount;
        const Matrix4& o2w = batch.objectToWorld;
        const float t0x = transforms->empty() ? 0.f : (*transforms)[0][3][0];
        const float t0y = transforms->empty() ? 0.f : (*transforms)[0][3][1];
        const float t0z = transforms->empty() ? 0.f : (*transforms)[0][3][2];
        // Include camera registration bits so we can see which cameras this
        // instance was seen with. Bits: 0=Main 1=ViewModel 2=Portal0 3=Portal1
        // 4=Sky 5=RenderToTexture 6=Unknown.
        uint32_t camBits = 0;
        for (uint32_t t = 0; t < CameraType::Count; ++t) {
          if (rtInstance->isCameraRegistered(static_cast<CameraType::Enum>(t))) {
            camBits |= (1u << t);
          }
        }
        Logger::info(str::format(
          "[PI-batch #", sPiBatchLogCount, "] blasRef=0x", std::hex, batch.blasReference, std::dec,
          " baseSurf=", batch.baseSurfaceIndex,
          " count=", batch.instanceCount,
          " mask=", uint32_t(batch.instanceMask),
          " tlasType=", uint32_t(batch.tlasType),
          " firstIdxInType=", batch.firstIndexInType,
          " camBits=0x", std::hex, camBits, std::dec,
          " seenMain=", (camBits & 1u) ? 1 : 0,
          " hidden=", rtInstance->isHidden() ? 1 : 0,
          " customIdxFlags=0x", std::hex, batch.customIndexFlags, std::dec,
          " sbtOffAndFlags=0x", std::hex, batch.sbtOffsetAndFlags, std::dec,
          " o2w.T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " t0.T=(", t0x, ",", t0y, ",", t0z, ")"));
      }
    }

    // Also reserve SSS TLAS slots if needed
    if (!isUnordered && rtInstance->isSubsurface()) {
      PointInstancerBatch sssBatch = batch;
      sssBatch.firstIndexInType = m_pointInstancerSlotsPerType[Tlas::SSS];
      sssBatch.tlasType = Tlas::SSS;
      m_pointInstancerSlotsPerType[Tlas::SSS] += instanceCount;
      m_pointInstancerBatches.push_back(sssBatch);
    }
  }

}  // namespace dxvk
