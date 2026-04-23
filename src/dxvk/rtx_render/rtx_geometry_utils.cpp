/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_geometry_utils.h"
#include "dxvk_device.h"
#include "rtx_render/rtx_shader_manager.h"

// NV-DXVK NPC SKINNING DIAG: targeted per-buffer logging.
// All shared state and the master switch live in dxvk_bone_diag.h.
#include "../dxvk_bone_diag.h"
#include <cstdlib>
#include <atomic>

#include <rtx_shaders/gen_tri_list_index_buffer.h>
#include <rtx_shaders/gpu_skinning.h>
#include <rtx_shaders/view_model_correction.h>
#include <rtx_shaders/bake_opacity_micromap.h>
#include <rtx_shaders/decode_and_add_opacity.h>
#include <rtx_shaders/interleave_geometry.h>
#include <rtx_shaders/smooth_normals.h>
#include "dxvk_scoped_annotation.h"

#include "rtx_context.h"
#include "rtx_options.h"

#include "rtx/pass/view_model/view_model_correction_binding_indices.h"
#include "rtx/pass/opacity_micromap/bake_opacity_micromap_binding_indices.h"
#include "rtx/pass/terrain_baking/decode_and_add_opacity_binding_indices.h"
#include "rtx/pass/gpu_skinning_binding_indices.h"
#include "rtx/pass/skinning.h"
#include "rtx/pass/smooth_normals_binding_indices.h"
#include "rtx/pass/smooth_normals.h"
#include "rtx/pass/gen_tri_list_index_buffer.h"
#include "rtx/pass/interleave_geometry_indices.h"
#include "rtx/pass/interleave_geometry.h"

namespace dxvk {
  static constexpr uint32_t kMaxInterleavedComponents = 3 + 3 + 2 + 1;

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class GenTriListIndicesShader : public ManagedShader {
      SHADER_SOURCE(GenTriListIndicesShader, VK_SHADER_STAGE_COMPUTE_BIT, gen_tri_list_index_buffer)

      PUSH_CONSTANTS(GenTriListArgs)

      BEGIN_PARAMETER()
        RW_STRUCTURED_BUFFER(GEN_TRILIST_BINDING_OUTPUT)
        STRUCTURED_BUFFER(GEN_TRILIST_BINDING_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(GenTriListIndicesShader);

    class SkinningShader : public ManagedShader {
      SHADER_SOURCE(SkinningShader, VK_SHADER_STAGE_COMPUTE_BIT, gpu_skinning)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BINDING_SKINNING_CONSTANTS)
        RW_STRUCTURED_BUFFER(BINDING_POSITION_OUTPUT)
        STRUCTURED_BUFFER(BINDING_POSITION_INPUT)
        STRUCTURED_BUFFER(BINDING_BLEND_WEIGHT_INPUT)
        STRUCTURED_BUFFER(BINDING_BLEND_INDICES_INPUT)
        RW_STRUCTURED_BUFFER(BINDING_NORMAL_OUTPUT)
        STRUCTURED_BUFFER(BINDING_NORMAL_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(SkinningShader);

    class ViewModelCorrectionShader : public ManagedShader {
      SHADER_SOURCE(ViewModelCorrectionShader, VK_SHADER_STAGE_COMPUTE_BIT, view_model_correction)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BINDING_VMC_CONSTANTS)
        RW_STRUCTURED_BUFFER(BINDING_VMC_POSITION_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(BINDING_VMC_NORMAL_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ViewModelCorrectionShader);

    class BakeOpacityMicromapShader : public ManagedShader {
      SHADER_SOURCE(BakeOpacityMicromapShader, VK_SHADER_STAGE_COMPUTE_BIT, bake_opacity_micromap)
      
      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(BINDING_BAKE_OPACITY_MICROMAP_TEXCOORD_INPUT) 
        SAMPLER2D(BINDING_BAKE_OPACITY_MICROMAP_OPACITY_INPUT)
        SAMPLER2D(BINDING_BAKE_OPACITY_MICROMAP_SECONDARY_OPACITY_INPUT)
        STRUCTURED_BUFFER(BINDING_BAKE_OPACITY_MICROMAP_BINDING_SURFACE_DATA_INPUT)
        CONSTANT_BUFFER(BINDING_BAKE_OPACITY_MICROMAP_CONSTANTS)
        RW_STRUCTURED_BUFFER(BINDING_BAKE_OPACITY_MICROMAP_ARRAY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BakeOpacityMicromapShader);

    class DecodeAndAddOpacityShader : public ManagedShader {
      SHADER_SOURCE(DecodeAndAddOpacityShader, VK_SHADER_STAGE_COMPUTE_BIT, decode_and_add_opacity)

      PUSH_CONSTANTS(DecodeAndAddOpacityArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(DECODE_AND_ADD_OPACITY_BINDING_TEXTURE_INPUT)
        TEXTURE2D(DECODE_AND_ADD_OPACITY_BINDING_ALBEDO_OPACITY_TEXTURE_INPUT)
        RW_TEXTURE2D(DECODE_AND_ADD_OPACITY_BINDING_TEXTURE_OUTPUT)
        SAMPLER(DECODE_AND_ADD_OPACITY_BINDING_LINEAR_SAMPLER)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(DecodeAndAddOpacityShader);

    class InterleaveGeometryShader : public ManagedShader {
      SHADER_SOURCE(InterleaveGeometryShader, VK_SHADER_STAGE_COMPUTE_BIT, interleave_geometry)

      PUSH_CONSTANTS(InterleaveGeometryArgs)

      BEGIN_PARAMETER()
      RW_STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_OUTPUT)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_POSITION_INPUT)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_NORMAL_INPUT)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_TEXCOORD_INPUT)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_COLOR0_INPUT)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX)
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_BONE_INDEX)
      // NV-DXVK: BONE_WEIGHT was missing from the layout — the shader
      // declares srcBoneWeight at INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT
      // but the pipeline had no matching descriptor slot, producing a
      // Vulkan "binding 4294967295 ... invalid" error at dispatch time
      // (srcBoneWeight's binding was UINT32_MAX because DXVK couldn't
      // find a layout entry for it). Left unfixed, this was the root
      // GPU-hang trigger observed with Aftermath crash dumps when extra
      // CS work (e.g. HUD-compositor blits) interacted with it.
      STRUCTURED_BUFFER(INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(InterleaveGeometryShader);


    class SmoothNormalsShader : public ManagedShader {
      SHADER_SOURCE(SmoothNormalsShader, VK_SHADER_STAGE_COMPUTE_BIT, smooth_normals)

      PUSH_CONSTANTS(SmoothNormalsArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(SMOOTH_NORMALS_BINDING_POSITION_RO)
        RW_STRUCTURED_BUFFER(SMOOTH_NORMALS_BINDING_NORMAL_RW)
        STRUCTURED_BUFFER(SMOOTH_NORMALS_BINDING_INDEX_INPUT)
        RW_STRUCTURED_BUFFER(SMOOTH_NORMALS_BINDING_HASH_TABLE)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(SmoothNormalsShader);

    float calcUVTileSizeSqr(const Matrix4& objectToWorld, const uint8_t* pVertex, size_t vertexStride, const uint8_t* pTexcoord, size_t texcoordStride, uint32_t vertex1, uint32_t vertex2, uint32_t vertex3) {
      const Vector4 p1 = objectToWorld * Vector4(*reinterpret_cast<const Vector3* const>(pVertex + vertexStride * vertex1), 1.f);
      const Vector4 p2 = objectToWorld * Vector4(*reinterpret_cast<const Vector3* const>(pVertex + vertexStride * vertex2), 1.f);
      const Vector4 p3 = objectToWorld * Vector4(*reinterpret_cast<const Vector3* const>(pVertex + vertexStride * vertex3), 1.f);

      const Vector2& t1 = *reinterpret_cast<const Vector2* const>(pTexcoord + texcoordStride * vertex1);
      const Vector2& t2 = *reinterpret_cast<const Vector2* const>(pTexcoord + texcoordStride * vertex2);
      const Vector2& t3 = *reinterpret_cast<const Vector2* const>(pTexcoord + texcoordStride * vertex3);
      // UV tile size (squared)
      float len1Sqr = p1 != p2 ? lengthSqr(p1 - p2) / lengthSqr(t1 - t2) : 0.f;
      float len2Sqr = p1 != p3 ? lengthSqr(p1 - p3) / lengthSqr(t1 - t3) : 0.f;
      float len3Sqr = p2 != p3 ? lengthSqr(p2 - p3) / lengthSqr(t2 - t3) : 0.f;

      return std::max(len1Sqr, std::max(len2Sqr, len3Sqr));
    }


    float calcMaxUvTileSizeSqrIndexed(uint32_t indexCount, const Matrix4& objectToWorld, const uint8_t* pVertex, size_t vertexStride, const uint8_t* pTexcoord, size_t texcoordStride, const void* pIndexData, size_t indexStride) {
      float result = 0.f;
      if (indexStride == 2) {
        // 16 bit indices
        const uint16_t* pIndex = static_cast<const uint16_t*>(pIndexData);
        for (uint32_t i = 0; i < indexCount; i += 3) {
          uint32_t vertex1 = pIndex[i];
          uint32_t vertex2 = pIndex[i+1];
          uint32_t vertex3 = pIndex[i+2];

          result = std::max(result, calcUVTileSizeSqr(objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, vertex1, vertex2, vertex3));
        }
      } else if (indexStride == 4) {
        // 32 bit indices
        const uint32_t* pIndex = static_cast<const uint32_t*>(pIndexData);
        for (uint32_t i = 0; i + 2 < indexCount; i += 3) {
          uint32_t vertex1 = pIndex[i];
          uint32_t vertex2 = pIndex[i+1];
          uint32_t vertex3 = pIndex[i+2];

          result = std::max(result, calcUVTileSizeSqr(objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, vertex1, vertex2, vertex3));
        }
      } else {
        ONCE(Logger::err("calcMaxUvTileSizeSqrIndexed: invalid index stride"));
      }
      return result;
    }

    float calcMaxUvTileSizeSqrTriangles(uint32_t vertexCount, const Matrix4& objectToWorld, const uint8_t* pVertex, size_t vertexStride, const uint8_t* pTexcoord, size_t texcoordStride) {
      float result = 0.f;
      for (uint32_t i = 0; i < vertexCount; i += 3) {
        uint32_t vertex1 = i;
        uint32_t vertex2 = i+1;
        uint32_t vertex3 = i+2;

        result = std::max(result, calcUVTileSizeSqr(objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, vertex1, vertex2, vertex3));
      }
      return result;
    }

    float calcMaxUvTileSizeSqrTriangleStrip(uint32_t vertexCount, const Matrix4& objectToWorld, const uint8_t* pVertex, size_t vertexStride, const uint8_t* pTexcoord, size_t texcoordStride) {
      float result = 0.f;
      for (uint32_t i = 0; i + 2 < vertexCount; ++i) {
        uint32_t vertex1 = i;
        uint32_t vertex2 = i+1;
        uint32_t vertex3 = i+2;

        result = std::max(result, calcUVTileSizeSqr(objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, vertex1, vertex2, vertex3));
      }
      return result;
    }

    float calcMaxUvTileSizeSqrTriangleFan(uint32_t vertexCount, const Matrix4& objectToWorld, const uint8_t* pVertex, size_t vertexStride, const uint8_t* pTexcoord, size_t texcoordStride) {
      float result = 0.f;
      uint32_t vertex1 = 0;
      for (uint32_t i = 1; i + 1 < vertexCount; ++i) {
        uint32_t vertex2 = i+1;
        uint32_t vertex3 = i+2;

        result = std::max(result, calcUVTileSizeSqr(objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, vertex1, vertex2, vertex3));
      }
      return result;
    }
  }

  RtxGeometryUtils::RtxGeometryUtils(DxvkDevice* device) : CommonDeviceObject(device) {
    m_pCbData = std::make_unique<RtxStagingDataAlloc>(
      device,
      "RtxStagingDataAlloc: Geometry Utils CB",
      (VkMemoryPropertyFlagBits) (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    m_pSmoothNormalsHashData = std::make_unique<RtxStagingDataAlloc>(
      device,
      "RtxStagingDataAlloc: Smooth Normals Hash Table",
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      (VkBufferUsageFlags) (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);

    m_skinningContext = device->createContext();
  }

  RtxGeometryUtils::~RtxGeometryUtils() { }

  void RtxGeometryUtils::onDestroy() {
    m_pCbData = nullptr;
    m_pSmoothNormalsHashData = nullptr;
    m_skinningContext = nullptr;
  }

  void RtxGeometryUtils::dispatchSkinning(const DrawCallState& drawCallState,
                                          const RaytraceGeometry& geo) {
    const Rc<DxvkContext>& ctx = m_skinningContext;
    // Create command list for the initial skinning dispatch (e.g. The first frame we get skinning mesh draw calls)
    if (ctx->getCommandList() == nullptr) {
      ctx->beginRecording(ctx->getDevice()->createCommandList());
    }

    ScopedGpuProfileZone(ctx, "performSkinning");

    const auto normalVertexFormat = drawCallState.getGeometryData().normalBuffer.vertexFormat();

    SkinningArgs params {};

    // Note: VK_FORMAT_R32_UINT assumed to be 32 bit spherical octahedral normals.
    assert(normalVertexFormat == VK_FORMAT_R32G32B32_SFLOAT || normalVertexFormat == VK_FORMAT_R32G32B32A32_SFLOAT || normalVertexFormat == VK_FORMAT_R32_UINT);
    assert(drawCallState.getGeometryData().blendWeightBuffer.defined());

    memcpy(&params.bones[0], &drawCallState.getSkinningState().pBoneMatrices[0], sizeof(Matrix4) * drawCallState.getSkinningState().numBones);

    params.dstPositionStride = geo.positionBuffer.stride();
    params.dstPositionOffset = geo.positionBuffer.offsetFromSlice();
    params.dstNormalStride = geo.normalBuffer.stride();
    params.dstNormalOffset = geo.normalBuffer.offsetFromSlice();
    
    params.srcPositionStride = drawCallState.getGeometryData().positionBuffer.stride();
    params.srcPositionOffset = drawCallState.getGeometryData().positionBuffer.offsetFromSlice();
    params.srcNormalStride = drawCallState.getGeometryData().normalBuffer.stride();
    params.srcNormalOffset = drawCallState.getGeometryData().normalBuffer.offsetFromSlice();

    params.blendWeightStride = drawCallState.getGeometryData().blendWeightBuffer.stride();
    params.blendWeightOffset = drawCallState.getGeometryData().blendWeightBuffer.offsetFromSlice();
    params.blendIndicesStride = drawCallState.getGeometryData().blendIndicesBuffer.stride();
    params.blendIndicesOffset = drawCallState.getGeometryData().blendIndicesBuffer.offsetFromSlice();

    params.numVertices = geo.vertexCount;
    params.useIndices = drawCallState.getGeometryData().blendIndicesBuffer.defined() ? 1 : 0;
    params.numBones = drawCallState.getGeometryData().numBonesPerVertex;
    params.useOctahedralNormals = normalVertexFormat == VK_FORMAT_R32_UINT ? 1 : 0;

    // If we don't have a mappable vertex buffer then we need to do this on the GPU
    bool mustUseGPU = drawCallState.getGeometryData().positionBuffer.mapPtr() == nullptr;

    // At some point, its more efficient to do these calculations on the GPU, this limit is somewhat arbitrary however, and might require better tuning...
    const uint32_t kNumVerticesToProcessOnCPU = 256;

    // Check we have appropriate CPU access
    const bool pendingGpuWrite = drawCallState.getGeometryData().positionBuffer.isPendingGpuWrite() ||
                                 drawCallState.getGeometryData().normalBuffer.isPendingGpuWrite() ||
                                 drawCallState.getGeometryData().blendWeightBuffer.isPendingGpuWrite() ||
                                 (drawCallState.getGeometryData().blendIndicesBuffer.defined() && drawCallState.getGeometryData().blendIndicesBuffer.isPendingGpuWrite());

    const bool useCPU = params.numVertices <= kNumVerticesToProcessOnCPU && !pendingGpuWrite && !mustUseGPU;

    if (!useCPU) {
      // Setting alignment to device limit minUniformBufferOffsetAlignment because the offset value should be its multiple.
      // See https://vulkan.lunarg.com/doc/view/1.2.189.2/windows/1.2-extensions/vkspec.html#VUID-VkWriteDescriptorSet-descriptorType-00327
      const auto& devInfo = ctx->getDevice()->properties().core.properties;
      VkDeviceSize alignment = devInfo.limits.minUniformBufferOffsetAlignment;

      DxvkBufferSlice cb = m_pCbData->alloc(alignment, sizeof(SkinningArgs));
      memcpy(cb.mapPtr(0), &params, sizeof(SkinningArgs));
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(cb.buffer());

      ctx->bindResourceBuffer(BINDING_SKINNING_CONSTANTS, cb);
      ctx->bindResourceBuffer(BINDING_POSITION_OUTPUT, geo.positionBuffer);
      ctx->bindResourceBuffer(BINDING_POSITION_INPUT, drawCallState.getGeometryData().positionBuffer);
      ctx->bindResourceBuffer(BINDING_NORMAL_OUTPUT, geo.normalBuffer);
      ctx->bindResourceBuffer(BINDING_NORMAL_INPUT, drawCallState.getGeometryData().normalBuffer);
      ctx->bindResourceBuffer(BINDING_BLEND_WEIGHT_INPUT, drawCallState.getGeometryData().blendWeightBuffer);

      if (drawCallState.getGeometryData().blendIndicesBuffer.defined())
        ctx->bindResourceBuffer(BINDING_BLEND_INDICES_INPUT, drawCallState.getGeometryData().blendIndicesBuffer);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SkinningShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { params.numVertices, 1, 1 }, VkExtent3D { 128, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(cb.buffer());
    } else {
      const float* srcPosition = reinterpret_cast<float*>(drawCallState.getGeometryData().positionBuffer.mapPtr(0));
      const float* srcNormal = reinterpret_cast<float*>(drawCallState.getGeometryData().normalBuffer.mapPtr(0));
      const float* srcBlendWeight = reinterpret_cast<float*>(drawCallState.getGeometryData().blendWeightBuffer.mapPtr(0));
      const uint8_t* srcBlendIndices = reinterpret_cast<uint8_t*>(drawCallState.getGeometryData().blendIndicesBuffer.mapPtr(0));

      // For CPU we are going to update a single entry at a time...
      params.dstPositionStride = 0;
      params.dstPositionOffset = 0;
      params.dstNormalStride = 0;
      params.dstNormalOffset = 0;

      float dstPosition[3];
      float dstNormal[3];

      for (uint32_t idx = 0; idx < params.numVertices; idx++) {
        skinning(idx, &dstPosition[0], &dstNormal[0], srcPosition, srcBlendWeight, srcBlendIndices, srcNormal, params);

        ctx->writeToBuffer(geo.positionBuffer.buffer(), geo.positionBuffer.offsetFromSlice() + idx * geo.positionBuffer.stride(), sizeof(dstPosition), &dstPosition[0]);
        ctx->writeToBuffer(geo.normalBuffer.buffer(), geo.normalBuffer.offsetFromSlice() + idx * geo.normalBuffer.stride(), sizeof(dstNormal), &dstNormal[0]);
      }
    }
    ++m_skinningCommands;
  }


  void RtxGeometryUtils::dispatchViewModelCorrection(
    Rc<DxvkContext> ctx,
    const RaytraceGeometry& geo,
    const Matrix4& positionTransform) const {

    // Fill out the arguments
    ViewModelCorrectionArgs args {};
    args.positionTransform = positionTransform;
    args.vectorTransform = transpose(inverse(positionTransform));
    args.positionStride = geo.positionBuffer.stride();
    args.positionOffset = geo.positionBuffer.offsetFromSlice();
    args.normalStride = geo.normalBuffer.defined() ? geo.normalBuffer.stride() : 0;
    args.normalOffset = geo.normalBuffer.defined() ? geo.normalBuffer.offsetFromSlice() : 0;
    args.numVertices = geo.vertexCount;

    // Upload the arguments into a buffer slice
    const auto& devInfo = ctx->getDevice()->properties().core.properties;
    VkDeviceSize alignment = devInfo.limits.minUniformBufferOffsetAlignment;

    DxvkBufferSlice cb = m_pCbData->alloc(alignment, sizeof(ViewModelCorrectionArgs));
    memcpy(cb.mapPtr(0), &args, sizeof(ViewModelCorrectionArgs));
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(cb.buffer());

    // Bind other resources
    ctx->bindResourceBuffer(BINDING_VMC_CONSTANTS, cb);
    ctx->bindResourceBuffer(BINDING_VMC_POSITION_INPUT_OUTPUT, geo.positionBuffer);
    ctx->bindResourceBuffer(BINDING_VMC_NORMAL_INPUT_OUTPUT, geo.normalBuffer.defined() ? geo.normalBuffer : geo.positionBuffer);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ViewModelCorrectionShader::getShader());

    // Run the shader
    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { args.numVertices, 1, 1 }, VkExtent3D { 128, 1, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // Make sure the geom buffers are tracked for liveness
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(geo.positionBuffer.buffer());
    if (geo.normalBuffer.defined())
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(geo.normalBuffer.buffer());
  }

  // Calculates number of uTriangles to bake considering their triangle specific cost and an available budget.
  // Expects a bakeState with non-zero remaining micro triangles to be baked.
  // Returns values 1 or greater
  uint32_t RtxGeometryUtils::calculateNumMicroTrianglesToBake(
    const BakeOpacityMicromapState& bakeState,
    const BakeOpacityMicromapDesc& desc,
    // Alignment to which budget can be extended to if there are any remaining uTriangles to be baked in the last considered triangle
    const uint32_t allowedNumMicroTriangleAlignment,
    const float bakingWeightScale,
    // This budget is decreased by budget used up by the returned number of micro triangles to bake.
    // Expects a value 1 or greater
    uint32_t& availableBakingBudget) {

    uint32_t numMicroTrianglesToBake = 0;
    const uint32_t startTriangleIndex = bakeState.numMicroTrianglesBaked / desc.numMicroTrianglesPerTriangle;

    // Add uTriangles to bake from the remaining triangles in the geometry or until the baking budget limit is hit
    for (uint32_t triangleIndex = startTriangleIndex; triangleIndex < desc.numTriangles; triangleIndex++) {

      // Find number of uTriangles to bake for this triangle
      uint32_t numActiveMicroTriangles = desc.numMicroTrianglesPerTriangle;
      if (triangleIndex == startTriangleIndex && bakeState.numMicroTrianglesBaked > 0) {
        // Subtract previously baked uTriangles for this triangle
        numActiveMicroTriangles -= bakeState.numMicroTrianglesBaked 
                                 - startTriangleIndex * desc.numMicroTrianglesPerTriangle;
      }

      // Note: using floats below will result in some imprecisions, but the error should not 
      // make noticeable difference in the big picture and the floats are floor/ceil-ed such 
      // so as to not overshoot the budget

      // Calculate baking cost of a uTriangle for this triangle
      const float microTriangleCost = bakingWeightScale * 
        (1 +
         desc.numTexelsPerMicrotriangle[desc.triangleOffset + triangleIndex] * desc.costPerTexelTapPerMicroTriangleBudget);

      // Calculate baking cost of this triangle (i.e. including all of its remaining uTriangles that still need to be baked).
      // Note: take a ceil to overestimate rather than underestimate the cost
      const uint32_t weightedTriangleCost =
        static_cast<uint32_t>(
          std::min(
            ceil(numActiveMicroTriangles * microTriangleCost),
            static_cast<float>(UINT32_MAX)));

      // We have enough budget to bake uTriangles for (the rest of) the triangle
      if (weightedTriangleCost <= availableBakingBudget) {
        availableBakingBudget -= weightedTriangleCost;
        numMicroTrianglesToBake += numActiveMicroTriangles;
        continue;

      } else { // Not enough budget to bake all the uTriangles
        // Calculate how many uTriangles fit into the budget considering the alignment
        // Note: take a floor to underestimate number of uTriangles that fit
        const uint32_t maxNumMicroTrianglesWithinBakingBudgetAligned = 
          // Ensure aligning of values 1 or higher since 0 aligns with all values and thus would align to 0 which is undesired
          // as the current function's returned value is expected to be non 0
          align_safe(std::max(1u, static_cast<uint32_t>(floor(availableBakingBudget / microTriangleCost))), 
                     allowedNumMicroTriangleAlignment, 
                     UINT32_MAX);

        numMicroTrianglesToBake += std::min(numActiveMicroTriangles, maxNumMicroTrianglesWithinBakingBudgetAligned);

        // Simply nullify the budget, since it is too small for any other baking dispatch to be efficient
        availableBakingBudget = 0;

        break;
      }
    }

    return numMicroTrianglesToBake;
  }

  void RtxGeometryUtils::dispatchBakeOpacityMicromap(
    Rc<DxvkContext> ctx,
    const RtInstance& instance,
    const RaytraceGeometry& geo,
    const std::vector<TextureRef>& textures,
    const std::vector<Rc<DxvkSampler>>& samplers,
    const uint32_t albedoOpacityTextureIndex,
    const uint32_t samplerIndex,
    const uint32_t secondaryAlbedoOpacityTextureIndex,
    const uint32_t secondarySamplerIndex,
    const BakeOpacityMicromapDesc& desc,
    BakeOpacityMicromapState& bakeState,
    uint32_t& availableBakingBudget,
    Rc<DxvkBuffer> opacityMicromapBuffer) const {

    // Init textures
    const TextureRef& opacityTexture = textures[albedoOpacityTextureIndex];
    const TextureRef* secondaryOpacityTexture = nullptr;
    if (secondaryAlbedoOpacityTextureIndex != kSurfaceMaterialInvalidTextureIndex)
      secondaryOpacityTexture = &textures[secondaryAlbedoOpacityTextureIndex];

    VkExtent3D opacityTextureResolution = opacityTexture.getImageView()->imageInfo().extent;

    // Fill out the arguments
    BakeOpacityMicromapArgs args {};
    size_t surfaceWriteOffset = 0;
    instance.surface.writeGPUData(&args.surface[0], surfaceWriteOffset);
    args.numTriangles = desc.numTriangles;
    args.numMicroTrianglesPerTriangle = desc.numMicroTrianglesPerTriangle;
    args.is2StateOMMFormat = desc.ommFormat == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT;
    args.subdivisionLevel = desc.subdivisionLevel;
    args.texcoordOffset = geo.texcoordBuffer.offsetFromSlice();
    args.texcoordStride = geo.texcoordBuffer.stride();
    args.resolveTransparencyThreshold = desc.resolveTransparencyThreshold;
    args.resolveOpaquenessThreshold = desc.resolveOpaquenessThreshold;
    args.useConservativeEstimation = desc.useConservativeEstimation;
    args.isOpaqueMaterial = desc.materialType == MaterialDataType::Opaque;
    args.isRayPortalMaterial = desc.materialType == MaterialDataType::RayPortal;
    args.applyVertexAndTextureOperations = desc.applyVertexAndTextureOperations;
    args.numMicroTrianglesPerThread = args.is2StateOMMFormat ? 8 : 4;
    args.textureResolution = vec2 { static_cast<float>(opacityTextureResolution.width), static_cast<float>(opacityTextureResolution.height) };
    args.rcpTextureResolution = vec2 { 1.f / opacityTextureResolution.width, 1.f / opacityTextureResolution.height };
    args.conservativeEstimationMaxTexelTapsPerMicroTriangle = desc.conservativeEstimationMaxTexelTapsPerMicroTriangle;
    args.triangleOffset = desc.triangleOffset;

    // Init samplers
    Rc<DxvkSampler> opacitySampler;
    Rc<DxvkSampler> secondaryOpacitySampler;
    {
      const DxvkSamplerCreateInfo& samplerInfo = samplers[samplerIndex]->info();

      opacitySampler = device()->getCommon()->getResources().getSampler(
        VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        samplerInfo.addressModeU, samplerInfo.addressModeV, samplerInfo.addressModeW,
        samplerInfo.borderColor);

      if (secondaryOpacityTexture) {
        const DxvkSamplerCreateInfo& secondarySamplerInfo = samplers[secondarySamplerIndex]->info();
          secondaryOpacitySampler = device()->getCommon()->getResources().getSampler(
          VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
          secondarySamplerInfo.addressModeU, secondarySamplerInfo.addressModeV, secondarySamplerInfo.addressModeW,
          secondarySamplerInfo.borderColor);
      }
      else {
        secondaryOpacitySampler = opacitySampler;
      }
    }

    // Bind other resources
    ctx->bindResourceBuffer(BINDING_BAKE_OPACITY_MICROMAP_TEXCOORD_INPUT, geo.texcoordBuffer);
    ctx->bindResourceView(BINDING_BAKE_OPACITY_MICROMAP_OPACITY_INPUT, opacityTexture.getImageView(), nullptr);
    ctx->bindResourceSampler(BINDING_BAKE_OPACITY_MICROMAP_OPACITY_INPUT, opacitySampler);
    ctx->bindResourceView(BINDING_BAKE_OPACITY_MICROMAP_SECONDARY_OPACITY_INPUT,
                          secondaryOpacityTexture ? secondaryOpacityTexture->getImageView() : opacityTexture.getImageView(), nullptr);
    ctx->bindResourceSampler(BINDING_BAKE_OPACITY_MICROMAP_SECONDARY_OPACITY_INPUT, secondaryOpacitySampler);
    ctx->bindResourceBuffer(BINDING_BAKE_OPACITY_MICROMAP_BINDING_SURFACE_DATA_INPUT,
                            DxvkBufferSlice(device()->getCommon()->getSceneManager().getSurfaceBuffer()));
    ctx->bindResourceBuffer(BINDING_BAKE_OPACITY_MICROMAP_ARRAY_OUTPUT,
                            DxvkBufferSlice(opacityMicromapBuffer, 0, opacityMicromapBuffer->info().size));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BakeOpacityMicromapShader::getShader());

    if (!bakeState.initialized) {
      bakeState.numMicroTrianglesToBake = args.numTriangles * args.numMicroTrianglesPerTriangle;
      bakeState.numMicroTrianglesBaked = 0;
      bakeState.initialized = true;
    }

    const uint32_t numMicroTrianglesPerWord = args.is2StateOMMFormat ? 32 : 16;
    const uint32_t kNumMicroTrianglesPerComputeBlock = BAKE_OPACITY_MICROMAP_NUM_THREAD_PER_COMPUTE_BLOCK * args.numMicroTrianglesPerThread;
    const VkPhysicalDeviceLimits& limits = device()->properties().core.properties.limits;
    // Workgroup count limit can be high (i.e. 2 Billion), so avoid overflowing uint32_t limit 
    const uint32_t maxThreadsPerDispatch = std::min(limits.maxComputeWorkGroupCount[0], UINT32_MAX / kNumMicroTrianglesPerComputeBlock) *
                                           kNumMicroTrianglesPerComputeBlock;
    const uint32_t maxThreadsPerDispatchAligned = alignDown(maxThreadsPerDispatch, numMicroTrianglesPerWord); // Align down so as not to overshoot the limits

    // Baking cost increases with opacity texture resolution, so scale up the baking cost accordingly
    const float kResolutionWeight = 0.05f;  // Selected empirically
    const float minResolutionToScale = 128; // Selected empirically
    const float avgTextureResolution = 0.5f * (args.textureResolution.x + args.textureResolution.y);
    const float bakingWeightScale = 
      avgTextureResolution > minResolutionToScale 
      ? 1 + kResolutionWeight * avgTextureResolution / minResolutionToScale
      : 1;

    // Align number of microtriangles to bake up to how many are packed into a single word
    const uint32_t numMicroTrianglesAlignment = numMicroTrianglesPerWord;
    const uint32_t numMicroTrianglesToBake =
      calculateNumMicroTrianglesToBake(bakeState, desc, numMicroTrianglesAlignment, bakingWeightScale, availableBakingBudget);

    // Calculate per dispatch counts
    const uint32_t numThreads = numMicroTrianglesToBake / args.numMicroTrianglesPerThread;
    const uint32_t numThreadsPerDispatch = std::min(numThreads, maxThreadsPerDispatchAligned);
    const uint32_t numDispatches = dxvk::util::ceilDivide(numThreads, numThreadsPerDispatch);
    const uint32_t baseThreadIndexOffset = bakeState.numMicroTrianglesBaked / args.numMicroTrianglesPerThread;

    args.numActiveThreads = numThreadsPerDispatch;

    for (uint32_t i = 0; i < numDispatches; i++) {
      args.threadIndexOffset = i * numThreadsPerDispatch + baseThreadIndexOffset;

      // Upload the arguments into a buffer slice
      const auto& devInfo = ctx->getDevice()->properties().core.properties;
      DxvkBufferSlice cb = m_pCbData->alloc(devInfo.limits.minUniformBufferOffsetAlignment, sizeof(BakeOpacityMicromapArgs));
      memcpy(cb.mapPtr(0), &args, sizeof(BakeOpacityMicromapArgs));
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(cb.buffer());

      // Bind other resources
      ctx->bindResourceBuffer(BINDING_BAKE_OPACITY_MICROMAP_CONSTANTS, cb);

      // Run the shader
      const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { numThreadsPerDispatch, 1, 1 }, VkExtent3D { BAKE_OPACITY_MICROMAP_NUM_THREAD_PER_COMPUTE_BLOCK, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    bakeState.numMicroTrianglesBaked += numMicroTrianglesToBake;
    bakeState.numMicroTrianglesBakedInLastBake = numMicroTrianglesToBake;

    // Make sure the geom buffers are tracked for liveness
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(opacityMicromapBuffer);
  }

  void RtxGeometryUtils::decodeAndAddOpacity(
      Rc<DxvkContext> ctx,
      const TextureRef& albedoOpacityTexture,
      const std::vector<TextureConversionInfo>& conversionInfos) {

    ScopedGpuProfileZone(ctx, "Decode And Add Opacity");

    Resources& resourceManager = ctx->getCommonObjects()->getResources();
    Rc<DxvkSampler> linearSampler = resourceManager.getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Bind resources
    ctx->bindResourceView(DECODE_AND_ADD_OPACITY_BINDING_ALBEDO_OPACITY_TEXTURE_INPUT, albedoOpacityTexture.getImageView(), nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, DecodeAndAddOpacityShader::getShader());
    ctx->bindResourceSampler(DECODE_AND_ADD_OPACITY_BINDING_LINEAR_SAMPLER, linearSampler);
    
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    for (uint32_t i = 0; i < conversionInfos.size(); i++) {
      const TextureConversionInfo& conversionInfo = conversionInfos[i];

      // Bind resources
      ctx->bindResourceView(DECODE_AND_ADD_OPACITY_BINDING_TEXTURE_INPUT, conversionInfo.sourceTexture->getImageView(), nullptr);
      ctx->bindResourceView(DECODE_AND_ADD_OPACITY_BINDING_TEXTURE_OUTPUT, conversionInfo.targetTexture.getImageView(), nullptr);

      // Fill out args
      DecodeAndAddOpacityArgs args {};
      args.textureType = conversionInfo.type;
      const VkExtent3D& extent = conversionInfo.targetTexture.getImageView()->imageInfo().extent;
      args.resolution = uint2(extent.width, extent.height);
      args.rcpResolution = float2(1.f / extent.width, 1.f / extent.height);
      args.normalIntensity = OpaqueMaterialOptions::normalIntensity();
      args.scale = conversionInfo.scale;
      args.offset = conversionInfo.offset;

      ctx->pushConstants(0, sizeof(args), &args);

      // Run the shader
      const VkExtent3D workgroups = util::computeBlockCount(extent, VkExtent3D { DECODE_AND_ADD_OPACITY_CS_DIMENSIONS });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
  }

  uint32_t RtxGeometryUtils::getOptimalTriangleListSize(const RasterGeometry& input) {
    const uint32_t primCount = (input.indexCount > 0) ? input.indexCount : input.vertexCount;
    assert(primCount > 0);
    switch (input.topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return primCount;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return (primCount - 2) * 3; // Conservative, assume no degenerates, no restart.  Actual returned in indexCountOut
    default:
      Logger::err("getTriangleListSize: unsupported topology");
      return 0;
    }
  }

  VkIndexType RtxGeometryUtils::getOptimalIndexFormat(uint32_t vertexCount) {
    assert(vertexCount > 0);
    // Generated triangle lists always use 32-bit indices — no vertex count
    // limitation on modern APIs.  The BLAS builder accepts both formats.
    return VK_INDEX_TYPE_UINT32;
  }

  bool RtxGeometryUtils::cacheIndexDataOnGPU(const Rc<DxvkContext>& ctx, const RasterGeometry& input, RaytraceGeometry& output) {
    ScopedCpuProfileZone();
    // NV-DXVK: If a CPU snapshot of the index data was captured at SubmitDraw
    // time (for DYNAMIC D3D11 buffers — see d3d11_rtx.cpp), upload from that
    // instead of a racy GPU→GPU copy from the potentially-renamed source.
    // The snapshot is race-safe: captured on the thread owning D3D11 state.
    if (input.indexDataSnapshot && !input.indexDataSnapshot->empty()
        && input.isTopologyRaytraceReady()) {
      // TDR-DIAG: scan the snapshotted indices for OOB values that would
      // cause the BLAS builder to read past the vertex buffer → MMU fault.
      // If indexValue >= vertexCount, the AS build will read garbage memory.
      const auto& data = *input.indexDataSnapshot;
      const uint32_t stride = input.indexBuffer.stride();
      const uint32_t vtxCount = input.vertexCount;
      uint32_t maxIdx = 0;
      uint32_t oobCount = 0;
      uint32_t oobFirstIdx = 0;
      uint32_t oobFirstVal = 0;
      if (stride == 2) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data.data());
        const size_t n = data.size() / 2;
        for (size_t i = 0; i < n; ++i) {
          if (p[i] > maxIdx) maxIdx = p[i];
          if (p[i] >= vtxCount) {
            if (oobCount == 0) { oobFirstIdx = uint32_t(i); oobFirstVal = p[i]; }
            ++oobCount;
          }
        }
      } else if (stride == 4) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(data.data());
        const size_t n = data.size() / 4;
        for (size_t i = 0; i < n; ++i) {
          if (p[i] > maxIdx) maxIdx = p[i];
          if (p[i] >= vtxCount) {
            if (oobCount == 0) { oobFirstIdx = uint32_t(i); oobFirstVal = p[i]; }
            ++oobCount;
          }
        }
      }
      if (oobCount > 0) {
        Logger::err(str::format("[IDX-OOB] OOB indices in snapshot! vtxCount=",
          vtxCount, " maxSeenIdx=", maxIdx, " oobCount=", oobCount,
          " firstOOB[", oobFirstIdx, "]=", oobFirstVal,
          " idxCount=", input.indexCount, " stride=", stride,
          " snapBytes=", data.size()));
      } else {
        static uint32_t sOkLog = 0;
        if (sOkLog < 5) {
          ++sOkLog;
          Logger::info(str::format("[IDX-OK] vtxCount=", vtxCount,
            " maxSeenIdx=", maxIdx, " idxCount=", input.indexCount, " stride=", stride));
        }
      }
      ctx->writeToBuffer(output.indexCacheBuffer, 0,
                         input.indexDataSnapshot->size(),
                         input.indexDataSnapshot->data());
      return true;
    }
    // Handle index buffer replacement - since the BVH builder does not support legacy primitive topology
    if (input.isTopologyRaytraceReady()) {
      // TDR-DIAG: try to scan static source index buffer via host mapPtr if available.
      // Many DXVK immutable buffers are host-visible for staging.
      if (input.indexBuffer.buffer() != nullptr) {
        const void* mp = input.indexBuffer.buffer()->mapPtr(0);
        if (mp != nullptr) {
          const uint32_t stride = input.indexBuffer.stride();
          const uint32_t vtxCount = input.vertexCount;
          const size_t baseOff = input.indexBuffer.offset() + input.indexBuffer.offsetFromSlice();
          const size_t byteCount = size_t(input.indexCount) * stride;
          const size_t bufSize = input.indexBuffer.buffer()->info().size;
          if (baseOff + byteCount <= bufSize) {
            const uint8_t* base = reinterpret_cast<const uint8_t*>(mp) + baseOff;
            uint32_t maxIdx = 0, oobCount = 0, oobFirstVal = 0, oobFirstIdx = 0;
            if (stride == 2) {
              const uint16_t* p = reinterpret_cast<const uint16_t*>(base);
              for (uint32_t i = 0; i < input.indexCount; ++i) {
                if (p[i] > maxIdx) maxIdx = p[i];
                if (p[i] >= vtxCount) {
                  if (oobCount == 0) { oobFirstIdx = i; oobFirstVal = p[i]; }
                  ++oobCount;
                }
              }
            } else if (stride == 4) {
              const uint32_t* p = reinterpret_cast<const uint32_t*>(base);
              for (uint32_t i = 0; i < input.indexCount; ++i) {
                if (p[i] > maxIdx) maxIdx = p[i];
                if (p[i] >= vtxCount) {
                  if (oobCount == 0) { oobFirstIdx = i; oobFirstVal = p[i]; }
                  ++oobCount;
                }
              }
            }
            if (oobCount > 0) {
              Logger::err(str::format("[IDX-OOB-STATIC] vtxCount=", vtxCount,
                " maxSeenIdx=", maxIdx, " oobCount=", oobCount,
                " firstOOB[", oobFirstIdx, "]=", oobFirstVal,
                " idxCount=", input.indexCount, " stride=", stride,
                " memFlags=0x", std::hex, input.indexBuffer.buffer()->memFlags(), std::dec));
            }
          }
        }
      }
      ctx->copyBuffer(output.indexCacheBuffer, 0, input.indexBuffer.buffer(), input.indexBuffer.offset() + input.indexBuffer.offsetFromSlice(), input.indexCount * input.indexBuffer.stride());
    } else {
      return RtxGeometryUtils::generateTriangleList(ctx, input, output.indexCacheBuffer);
    }
    return true;
  }

  bool RtxGeometryUtils::generateTriangleList(const Rc<DxvkContext>& ctx, const RasterGeometry& input, Rc<DxvkBuffer> output) {
    ScopedCpuProfileZone();

    const uint32_t indexCount = getOptimalTriangleListSize(input);
    const uint32_t primIterCount = indexCount / 3;

    // Always generate 32-bit indices — no vertex count limitation at
    // Vulkan/D3D11 feature level 10.0+.  The BVH builder accepts both
    // VK_INDEX_TYPE_UINT16 and VK_INDEX_TYPE_UINT32.
    const uint32_t indexStride = 4;

    assert(output->info().size == align(indexCount * indexStride, CACHE_LINE_SIZE));

    // Prepare shader arguments
    GenTriListArgs pushArgs = { };
    pushArgs.firstIndex = 0;
    pushArgs.primCount = primIterCount;
    pushArgs.topology = (uint32_t) input.topology;
    pushArgs.useIndexBuffer = (input.indexBuffer.defined() && input.indexCount > 0) ? 1 : 0;
    pushArgs.minVertex = 0;
    pushArgs.maxVertex = input.vertexCount - 1;
    pushArgs.inputIsU16 = (pushArgs.useIndexBuffer && input.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) ? 1 : 0;
    pushArgs._pad = 0;

    ctx->getCommonObjects()->metaGeometryUtils().dispatchGenTriList(ctx, pushArgs, DxvkBufferSlice(output), pushArgs.useIndexBuffer ? &input.indexBuffer : nullptr);

    if (indexCount % 3 != 0) {
      ONCE(Logger::err(str::format("Generating indices for a mesh which has non triangle topology: (indices%3) != 0, geometry hash = 0x", std::hex, input.getHashForRule(RtxOptions::geometryAssetHashRule()))));
      return false;
    }

    return true;
  }

  void RtxGeometryUtils::dispatchGenTriList(const Rc<DxvkContext>& ctx, const GenTriListArgs& cb, const DxvkBufferSlice& dstSlice, const RasterBuffer* srcBuffer) const {
    ScopedGpuProfileZone(ctx, "generateTriangleList");
    const uint32_t kNumTrianglesToProcessOnCPU = 512;
    const bool useGPU = ((srcBuffer != nullptr) && (srcBuffer->isPendingGpuWrite())) || cb.primCount > kNumTrianglesToProcessOnCPU;

    if (useGPU) {
      ctx->bindResourceBuffer(GEN_TRILIST_BINDING_OUTPUT, dstSlice);

      // NV-DXVK: source index buffer may have an unaligned offset
      // (e.g. 32766 bytes in TF2), which violates Vulkan's SSBO
      // minStorageBufferOffsetAlignment (16). Align down to 16 bytes
      // and absorb the delta into firstIndex so the shader reads the
      // same data. Only safe for inputIsU16=1 (16-bit indices) because
      // the delta is in uint16 elements; uint32 indices would require
      // the delta to be a multiple of 4 which may not hold.
      GenTriListArgs cbAdjusted = cb;
      if (srcBuffer != nullptr) {
        const VkDeviceSize rawOff = srcBuffer->offset();
        constexpr VkDeviceSize kAlign = 16;
        const VkDeviceSize alignedOff = rawOff & ~(kAlign - 1);
        const VkDeviceSize deltaBytes = rawOff - alignedOff;
        if (deltaBytes == 0) {
          ctx->bindResourceBuffer(GEN_TRILIST_BINDING_INPUT, *srcBuffer);
        } else if (cb.inputIsU16 != 0 && (deltaBytes % 2) == 0) {
          // 16-bit indices: shift firstIndex by deltaBytes/2 elements.
          cbAdjusted.firstIndex += static_cast<uint32_t>(deltaBytes / 2);
          ctx->bindResourceBuffer(GEN_TRILIST_BINDING_INPUT,
            DxvkBufferSlice(srcBuffer->buffer(), alignedOff,
                            srcBuffer->length() + deltaBytes));
        } else if (cb.inputIsU16 == 0 && (deltaBytes % 4) == 0) {
          // 32-bit indices: shift firstIndex by deltaBytes/4 elements.
          cbAdjusted.firstIndex += static_cast<uint32_t>(deltaBytes / 4);
          ctx->bindResourceBuffer(GEN_TRILIST_BINDING_INPUT,
            DxvkBufferSlice(srcBuffer->buffer(), alignedOff,
                            srcBuffer->length() + deltaBytes));
        } else {
          // Misaligned by an odd byte count we can't absorb. Still
          // bind but expect the driver to complain — lesser evil
          // than a crash.
          ctx->bindResourceBuffer(GEN_TRILIST_BINDING_INPUT, *srcBuffer);
        }
      }

      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

      ctx->pushConstants(0, sizeof(GenTriListArgs), &cbAdjusted);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, GenTriListIndicesShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { cb.primCount, 1, 1 }, VkExtent3D { 128, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    } else {
      uint32_t dst[kNumTrianglesToProcessOnCPU * 3];

      // CPU path pre-widens 16-bit inputs so generateIndices sees uint32_t.
      uint32_t srcWidened[kNumTrianglesToProcessOnCPU * 3];
      const uint32_t* src = nullptr;

      GenTriListArgs cpuArgs = cb;
      cpuArgs.inputIsU16 = 0;  // Already widened on CPU side.

      if (cb.useIndexBuffer != 0) {
        const void* rawPtr = srcBuffer->mapPtr();
        if (srcBuffer->indexType() == VK_INDEX_TYPE_UINT32) {
          src = reinterpret_cast<const uint32_t*>(rawPtr);
        } else {
          const uint16_t* src16 = reinterpret_cast<const uint16_t*>(rawPtr);
          const uint32_t maxIdx = cb.primCount * 3 + cb.firstIndex;
          for (uint32_t i = 0; i < maxIdx; i++)
            srcWidened[i] = src16[i];
          src = srcWidened;
        }
      }

      for (uint32_t idx = 0; idx < cpuArgs.primCount; idx++) {
        generateIndices(idx, dst, src, cpuArgs);
      }

      ctx->writeToBuffer(dstSlice.buffer(), 0, cb.primCount * 3 * sizeof(uint32_t), dst);
    }
  }

  void RtxGeometryUtils::processGeometryBuffers(const InterleavedGeometryDescriptor& desc, RaytraceGeometry& output) {
    const DxvkBufferSlice targetSlice = DxvkBufferSlice(desc.buffer);

    output.positionBuffer = RaytraceBuffer(targetSlice, desc.positionOffset, desc.stride, VK_FORMAT_R32G32B32_SFLOAT);

    if (desc.hasNormals)
      output.normalBuffer = RaytraceBuffer(targetSlice, desc.normalOffset, desc.stride, VK_FORMAT_R32G32B32_SFLOAT);

    if (desc.hasTexcoord)
      output.texcoordBuffer = RaytraceBuffer(targetSlice, desc.texcoordOffset, desc.stride, VK_FORMAT_R32G32_SFLOAT);

    if (desc.hasColor0) 
      output.color0Buffer = RaytraceBuffer(targetSlice, desc.color0Offset, desc.stride, VK_FORMAT_B8G8R8A8_UNORM);
  }

  void RtxGeometryUtils::processGeometryBuffers(const RasterGeometry& input, RaytraceGeometry& output) {
    const DxvkBufferSlice slice = DxvkBufferSlice(output.historyBuffer[0]);

    output.positionBuffer = RaytraceBuffer(slice, input.positionBuffer.offsetFromSlice(), input.positionBuffer.stride(), input.positionBuffer.vertexFormat());

    if (input.normalBuffer.defined())
      output.normalBuffer = RaytraceBuffer(slice, input.normalBuffer.offsetFromSlice(), input.normalBuffer.stride(), input.normalBuffer.vertexFormat());

    if (input.texcoordBuffer.defined())
      output.texcoordBuffer = RaytraceBuffer(slice, input.texcoordBuffer.offsetFromSlice(), input.texcoordBuffer.stride(), input.texcoordBuffer.vertexFormat());

    if (input.color0Buffer.defined())
      output.color0Buffer = RaytraceBuffer(slice, input.color0Buffer.offsetFromSlice(), input.color0Buffer.stride(), input.color0Buffer.vertexFormat());
  }

  size_t RtxGeometryUtils::computeOptimalVertexStride(const RasterGeometry& input, bool forceNormals) {
    // Calculate stride
    size_t stride = sizeof(float) * 3; // position is the minimum

    if (input.normalBuffer.defined() || forceNormals) {
      stride += sizeof(float) * 3;
    }

    if (input.texcoordBuffer.defined()) {
      stride += sizeof(float) * 2;
    }

    if (input.color0Buffer.defined()) {
      stride += sizeof(uint32_t);
    }

    assert(stride <= kMaxInterleavedComponents * sizeof(float) && "Maximum number of interleaved components needs update.");

    return stride;
  }

  void RtxGeometryUtils::cacheVertexDataOnGPU(const Rc<DxvkContext>& ctx, const RasterGeometry& input, RaytraceGeometry& output, bool forceNormals) {
    ScopedCpuProfileZone();
    // When forceNormals is set, we can't use the fast interleaved copy path because
    // we need to change the vertex layout to include normal space.
    if (input.isVertexDataInterleaved() && input.areFormatsGpuFriendly() && !forceNormals) {
      const size_t vertexBufferSize = input.vertexCount * input.positionBuffer.stride();
      ctx->copyBuffer(output.historyBuffer[0], 0, input.positionBuffer.buffer(), input.positionBuffer.offset(), vertexBufferSize);

      processGeometryBuffers(input, output);
    } else {
      RtxGeometryUtils::InterleavedGeometryDescriptor interleaveResult;
      interleaveResult.buffer = output.historyBuffer[0];

      ctx->getCommonObjects()->metaGeometryUtils().interleaveGeometry(ctx, input, interleaveResult, forceNormals);

      processGeometryBuffers(interleaveResult, output);
    }
  }

  void RtxGeometryUtils::interleaveGeometry(
    const Rc<DxvkContext>& ctx,
    const RasterGeometry& input,
    InterleavedGeometryDescriptor& output,
    bool forceNormals) const {
    ScopedGpuProfileZone(ctx, "interleaveGeometry");
    // Required
    assert(input.positionBuffer.defined());

    // Calculate stride - when forceNormals is true, reserve space for normals even if input has none
    output.stride = computeOptimalVertexStride(input, forceNormals);
    
    assert(output.buffer->info().size == align(output.stride * input.vertexCount, CACHE_LINE_SIZE));

    bool mustUseGPU = input.positionBuffer.isPendingGpuWrite() || input.positionBuffer.mapPtr() == nullptr;

    // Interleave vertex data
    InterleaveGeometryArgs args;
    assert(input.positionBuffer.offsetFromSlice() % 4 == 0);
    args.positionOffset = input.positionBuffer.offsetFromSlice() / 4;
    args.positionStride = input.positionBuffer.stride() / 4;
    args.positionFormat = input.positionBuffer.vertexFormat();
    if (!interleaver::formatConversionFloatSupported(args.positionFormat)) {
      ONCE(Logger::err(str::format("[rtx-interleaver] Unsupported position buffer format (", args.positionFormat, ")")));
      return;
    }
    args.hasNormals = input.normalBuffer.defined();
    if (args.hasNormals) {
      mustUseGPU |= input.normalBuffer.isPendingGpuWrite() || input.normalBuffer.mapPtr() == nullptr;
      assert(input.normalBuffer.offsetFromSlice() % 4 == 0);
      args.normalOffset = input.normalBuffer.offsetFromSlice() / 4;
      args.normalStride = input.normalBuffer.stride() / 4;
      args.normalFormat = input.normalBuffer.vertexFormat();
      if (!interleaver::formatConversionFloatSupported(args.normalFormat)) {
        ONCE(Logger::info(str::format("[rtx-interleaver] Unsupported normal buffer format (", args.normalFormat, "), skipping normals")));
      }
    }
    args.hasTexcoord = input.texcoordBuffer.defined();
    if (args.hasTexcoord) {
      mustUseGPU |= input.texcoordBuffer.isPendingGpuWrite() || input.texcoordBuffer.mapPtr() == nullptr;
      assert(input.texcoordBuffer.offsetFromSlice() % 4 == 0);
      args.texcoordOffset = input.texcoordBuffer.offsetFromSlice() / 4;
      args.texcoordStride = input.texcoordBuffer.stride() / 4;
      args.texcoordFormat = input.texcoordBuffer.vertexFormat();
      if (!interleaver::formatConversionFloatSupported(args.texcoordFormat)) {
        ONCE(Logger::info(str::format("[rtx-interleaver] Unsupported texcoord buffer format (", args.texcoordFormat, "), skipping texcoord")));
      }
    }
    // NV-DXVK: track per-draw tcBuffer state so we can diagnose "textures
    // bind but sample as flat color" issues. Report first 20 + every 500
    // draws so the file doesn't explode.
    {
      static std::atomic<uint32_t> sWithTc{0}, sNoTc{0};
      if (args.hasTexcoord) ++sWithTc; else ++sNoTc;
      const uint32_t total = sWithTc.load() + sNoTc.load();
      if (total <= 20 || (total % 500) == 0) {
        Logger::info(str::format(
          "[rtx-interleaver.uv] draws total=", total,
          " withUV=", sWithTc.load(), " noUV=", sNoTc.load(),
          " this=", (args.hasTexcoord ? "YES" : "NO"),
          " tcFmt=", uint32_t(args.texcoordFormat),
          " tcOff=", args.texcoordOffset,
          " tcStride=", args.texcoordStride,
          " posFmt=", uint32_t(args.positionFormat),
          " posOff=", args.positionOffset,
          " posStride=", args.positionStride,
          " verts=", input.vertexCount));
      }
    }
    args.hasColor0 = input.color0Buffer.defined();
    // NV-DXVK: For R32G32_UINT positions, hijack the color0 slot to provide
    // uint-typed access to the position buffer.  Reading packed uint position
    // data through StructuredBuffer<float> corrupts NaN bit patterns (GPU
    // canonicalizes NaN on float load), breaking the 21/21/22-bit decode.
    // The color0 slot is StructuredBuffer<uint32_t> which preserves all bits.
    const bool posNeedsUintRead = (args.positionFormat == interleaver::SupportedVkFormats::VK_FORMAT_R32G32_UINT);
    if (posNeedsUintRead) {
      // Set color0 params for the position read (uint buffer redirect),
      // but DON'T set hasColor0 = true — that would make the interleaver
      // write an extra color float per vertex, misaligning the output stride.
      // The color0 buffer is only used as a READ source for position decode.
      args.color0Offset = args.positionOffset;
      args.color0Stride = args.positionStride;
      args.color0Format = args.positionFormat;
      mustUseGPU = true;
    } else if (args.hasColor0) {
      mustUseGPU |= input.color0Buffer.isPendingGpuWrite() || input.color0Buffer.mapPtr() == nullptr;
      assert(input.color0Buffer.offsetFromSlice() % 4 == 0);
      args.color0Offset = input.color0Buffer.offsetFromSlice() / 4;
      args.color0Stride = input.color0Buffer.stride() / 4;
      args.color0Format = input.color0Buffer.vertexFormat();
      if (!interleaver::formatConversionUintSupported(args.color0Format)) {
        ONCE(Logger::info(str::format("[rtx-interleaver] Unsupported color0 buffer format (", args.color0Format, "), skipping color0")));
      }
    }

    args.minVertexIndex = 0;
    assert(output.stride % 4 == 0);
    args.outputStride = output.stride / 4;
    args.vertexCount = input.vertexCount;
    args.forceNormals = (forceNormals && !input.normalBuffer.defined()) ? 1 : 0;
    args.hasBoneTransform = (input.boneMatrixBuffer.defined() && input.boneIndexBuffer.defined()) ? 1 : 0;
    args.boneIndex = input.boneInstanceIndex;  // instance index for bone lookup

    // NV-DXVK: defaults preserve legacy single-bone-per-draw skinning behavior.
    // For TF2 BSP / batched static props, RasterGeometry overrides these.
    args.bonePerVertex     = input.bonePerVertex ? 1u : 0u;
    args.boneMatrixStride  = input.boneMatrixStrideBytes != 0 ? input.boneMatrixStrideBytes : 48u;
    args.boneIndexStride   = input.boneIndexStrideBytes  != 0 ? input.boneIndexStrideBytes  : 8u;
    args.boneIndexMask     = input.boneIndexMask         != 0 ? input.boneIndexMask         : 0xFFFFu;

    // NV-DXVK (TF2 skinned chars): weighted skinning params. When the input
    // provides a boneWeightBuffer and the bone index buffer is RGBA8_UINT
    // (fmt=41), the interleaver does Σ w_i × bone[idx_i] × pos.
    args.hasBoneWeights   = input.boneWeightBuffer.defined() ? 1u : 0u;
    args.boneWeightOffset = input.boneWeightBuffer.defined()
      ? input.boneWeightBuffer.offsetFromSlice() / 4u : 0u;
    args.boneWeightStride = input.boneWeightBuffer.defined()
      ? input.boneWeightBuffer.stride() / 4u : 0u;
    args.boneIndexComponentCount = input.boneIndexComponentCount != 0
      ? input.boneIndexComponentCount : 1u;
    args.boneIndexOffsetUints = input.boneIndexBuffer.defined()
      ? input.boneIndexBuffer.offsetFromSlice() / 4u : 0u;
    if (args.hasBoneWeights && !args.hasBoneTransform) {
      // Safety: weighted skinning requires both index + weight streams AND
      // the matrix buffer. If any missing, fall back to plain.
      args.hasBoneWeights = 0u;
    }

    // NV-DXVK DEBUG: Log interleaver dispatch info for bone draws
    if (args.hasBoneTransform) {
      static uint32_t sInterleaveDiag = 0;
      if (sInterleaveDiag < 10) {
        ++sInterleaveDiag;
        Logger::info(str::format(
          "[rtx-interleaver] Bone dispatch: fmt=", args.positionFormat,
          " stride=", args.positionStride, " off=", args.positionOffset,
          " verts=", input.vertexCount, " inst=", input.boneInstanceIndex,
          " color0=", args.hasColor0, " c0fmt=", args.color0Format,
          " c0stride=", args.color0Stride, " c0off=", args.color0Offset,
          " mustGPU=", mustUseGPU));
      }
    }

    // NV-DXVK: always use GPU path. The CPU optimization for small N has
    // multiple latent bugs (no bone transforms, no R32G32_UINT remap, depends
    // on CPU-mappable buffers) that silently produce zero/garbage geometry.
    // Per-dispatch overhead is small relative to debugging cost of two paths.
    // The CPU else-branch below still references kNumVerticesToProcessOnCPU as
    // a stack-array size — keep the constant defined even though that branch
    // is now dead.
    constexpr uint32_t kNumVerticesToProcessOnCPU = 1024;
    (void)kNumVerticesToProcessOnCPU;
    const bool useGPU = true;

    if (useGPU) {
      // DEBUG: log first N GPU interleave dispatches per session, broken down
      // by whether bone transform is being applied (skinning) or not (BSP).
      static uint32_t sInterleaveGpuBone = 0;
      static uint32_t sInterleaveGpuPlain = 0;
      uint32_t* counter = args.hasBoneTransform ? &sInterleaveGpuBone : &sInterleaveGpuPlain;
      if (*counter < 8) {
        ++(*counter);
        Logger::info(str::format(
          "[rtx-interleaver] GPU dispatch (",
          args.hasBoneTransform ? "bone" : "plain",
          "): posFmt=", args.positionFormat,
          " verts=", input.vertexCount,
          " posNeedsUintRead=", (args.positionFormat == interleaver::SupportedVkFormats::VK_FORMAT_R32G32_UINT) ? 1 : 0,
          " counts(bone=", sInterleaveGpuBone, " plain=", sInterleaveGpuPlain, ")"));
      }
      ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_OUTPUT, DxvkBufferSlice(output.buffer));

      ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_POSITION_INPUT, input.positionBuffer);
      if (args.hasNormals)
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_NORMAL_INPUT, input.normalBuffer);
      if (args.hasTexcoord)
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_TEXCOORD_INPUT, input.texcoordBuffer);
      if (posNeedsUintRead) {
        // NV-DXVK: Bind position buffer to color0 slot for uint-typed access
        static uint32_t sBindLog = 0;
        if (sBindLog < 5) {
          ++sBindLog;
          Logger::info(str::format(
            "[interleaver] Color0 bind: posOff=", input.positionBuffer.offset(),
            " posLen=", input.positionBuffer.length(),
            " c0off=", args.color0Offset, " c0stride=", args.color0Stride,
            " posStride=", args.positionStride, " posOff=", args.positionOffset));
        }
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_COLOR0_INPUT, input.positionBuffer);
      } else if (args.hasColor0) {
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_COLOR0_INPUT, input.color0Buffer);
      }

      // NV-DXVK: Always bind bone slots (Vulkan requires all declared bindings bound).
      // For non-bone draws, bind the position buffer as a dummy placeholder.
      if (args.hasBoneTransform) {
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX, input.boneMatrixBuffer);
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_INDEX, input.boneIndexBuffer);
      } else {
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX, input.positionBuffer);
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_INDEX, DxvkBufferSlice(input.positionBuffer.buffer(), input.positionBuffer.offset(), std::min<VkDeviceSize>(input.positionBuffer.length(), 16)));
      }
      // NV-DXVK: bind bone-weight slot (required by shader layout even when
      // unused). Real stream for weighted skinning, position buffer as dummy
      // otherwise.
      if (args.hasBoneWeights && input.boneWeightBuffer.defined()) {
        // NV-DXVK: Vulkan SSBO bindings require pBufferInfo[i].offset to
        // be a multiple of minStorageBufferOffsetAlignment (typically
        // 16). TF2's bone-weight slice can have any byte offset (e.g.
        // 32766 observed in logs, triggering Aftermath GPU crash).
        // Align the binding offset DOWN to 16 bytes, then add the
        // delta to boneWeightOffset so the shader reads the same
        // element.
        const auto& bwBuf = input.boneWeightBuffer;
        const VkDeviceSize rawOff = bwBuf.offset();
        constexpr VkDeviceSize kAlign = 16; // minStorageBufferOffsetAlignment on most GPUs
        const VkDeviceSize alignedOff = rawOff & ~(kAlign - 1);
        const VkDeviceSize deltaBytes = rawOff - alignedOff;
        // Extend length by deltaBytes so the aligned slice still covers
        // the original range.
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT,
          DxvkBufferSlice(bwBuf.buffer(), alignedOff, bwBuf.length() + deltaBytes));
        // Shift the shader-side offset by deltaBytes/4 (uints) so the
        // math lands on the same data the original offset pointed to.
        args.boneWeightOffset += static_cast<uint32_t>(deltaBytes / 4u);
      } else {
        // Position buffer's offset may also be unaligned. Apply the
        // same alignment treatment; safe because this is a placeholder
        // — the shader never reads boneWeight when hasBoneWeights=0.
        const auto& pBuf = input.positionBuffer;
        const VkDeviceSize rawOff = pBuf.offset();
        constexpr VkDeviceSize kAlign = 16;
        const VkDeviceSize alignedOff = rawOff & ~(kAlign - 1);
        ctx->bindResourceBuffer(INTERLEAVE_GEOMETRY_BINDING_BONE_WEIGHT,
          DxvkBufferSlice(pBuf.buffer(), alignedOff,
                          std::min<VkDeviceSize>(pBuf.length() + (rawOff - alignedOff), 16)));
      }

      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

      ctx->pushConstants(0, sizeof(InterleaveGeometryArgs), &args);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, InterleaveGeometryShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { input.vertexCount, 1, 1 }, VkExtent3D { 128, 1, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

      // NV-DXVK: Dump raw INPUT vertex buffer bytes for R32G32_UINT draws
      // NV-DXVK: enable raw VBUF + interleaver output dump for first N R32G32_UINT
      // dispatches so we can verify the unpack actually produces sensible
      // float positions for BSP geometry.
      // NV-DXVK TF2 VIEWMODEL DEBUG: dump skinned R32G32_UINT dispatches
      // (e.g. the gun/body draws). Plain (non-skinned) dispatches were
      // exhausting the throttle before any skinned draw fired, so the
      // gun verts never produced a [interleaver.skin] log. Now skinned
      // and plain are tracked separately.
      // NV-DXVK NPC SKINNING DIAG: gated on RTX_BONE_DIAG env var via
      // tf2::boneDiagEnabled(). When enabled, every skinned dispatch
      // bumps the per-buffer reads counter consumed by EndFrame and
      // emits the verbose `[interleaver.skin*]` diagnostics.
      const bool sDiagEnabled = ::dxvk::tf2::boneDiagEnabled();
      static const uintptr_t sEnvTargetBufPtr = []() -> uintptr_t {
        const char* env = std::getenv("RTX_NPC_BONE_BUF");
        if (env == nullptr) return 0u;
        return static_cast<uintptr_t>(std::strtoull(env, nullptr, 16));
      }();
      const uintptr_t sTargetBufPtr = sEnvTargetBufPtr != 0u
        ? sEnvTargetBufPtr
        : ::dxvk::tf2::g_autoTargetBufPtr.load(std::memory_order_relaxed);
      static uint32_t sPlainDumpCount = 0;
      const bool isSkinned = args.hasBoneTransform != 0;
      const bool isUintPos = args.positionFormat == interleaver::SupportedVkFormats::VK_FORMAT_R32G32_UINT;
      const uintptr_t skinBufKey = (isSkinned && input.boneMatrixBuffer.defined())
        ? reinterpret_cast<uintptr_t>(input.boneMatrixBuffer.buffer().ptr())
        : 0u;
      const bool isTargetedSkin =
        isSkinned && sTargetBufPtr != 0u && skinBufKey == sTargetBufPtr;
      if (sDiagEnabled && isSkinned) {
        if (isTargetedSkin) {
          ::dxvk::tf2::g_targetReadDispatchesThisFrame.fetch_add(
            1, std::memory_order_relaxed);
          // NV-DXVK NPC SKINNING DIAG: append to the per-frame timeline so
          // EndFrame can show write/read ordering for one chosen buffer.
          // Records the slice offset+length the dispatch is about to read
          // from the targeted bone-matrix buffer.
          ::dxvk::tf2::BoneOp op{};
          op.seq    = ::dxvk::tf2::g_boneTimelineSeq.fetch_add(
                        1, std::memory_order_relaxed);
          op.op     = 'R';
          op.offset = static_cast<uint32_t>(input.boneMatrixBuffer.offset());
          op.size   = static_cast<uint32_t>(input.boneMatrixBuffer.length());
          op.bones  = op.size / 48u;
          op.sharedRot = 0;
          op.source    = 3u;
          std::lock_guard<std::mutex> lkT(::dxvk::tf2::g_boneTimelineMutex);
          ::dxvk::tf2::g_boneTimeline.push_back(op);
        }
        std::lock_guard<std::mutex> lk(::dxvk::tf2::g_perBufStatsMutex);
        ++::dxvk::tf2::g_perBufStats[skinBufKey].reads;
      }
      const bool wantDump = sDiagEnabled && isUintPos
        && ((isSkinned)
            || (!isSkinned && sPlainDumpCount < 4));
      if (wantDump) {
        if (!isSkinned) ++sPlainDumpCount;
      }
      // Keep the original variable name so the existing block below stays valid.
      static uint32_t sUintDumpCount = 0;
      if (wantDump) {
        ++sUintDumpCount;
        // NV-DXVK TF2: raised cap (was 3) so all bone-skinned dispatches
        // we let through actually emit RAW VBUF / GPU OUTPUT / interleaver.skin.
        static uint32_t sRawDumpCount = 0;
        if (sRawDumpCount < 30) {
          ++sRawDumpCount;
          // Read 2 full vertices from the INPUT position buffer (not the output)
          const uint32_t stride = input.positionBuffer.stride();
          const VkDeviceSize dumpSize = static_cast<VkDeviceSize>(stride) * 2;
          DxvkBufferCreateInfo dumpInfo;
          dumpInfo.size = std::max<VkDeviceSize>(dumpSize, 64);
          dumpInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          dumpInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          dumpInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          auto dumpBuf = m_device->createBuffer(dumpInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "vbuf-dump");
          // Barrier: ensure vertex buffer is readable
          VkBufferMemoryBarrier bar = {};
          bar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
          bar.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
          bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          bar.buffer = input.positionBuffer.buffer()->getSliceHandle().handle;
          bar.offset = input.positionBuffer.offset();
          bar.size = dumpSize;
          bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          ctx->getCommandList()->cmdPipelineBarrier(
            DxvkCmdBuffer::ExecBuffer,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &bar, 0, nullptr);
          ctx->copyBuffer(dumpBuf, 0,
            input.positionBuffer.buffer(), input.positionBuffer.offset(), dumpSize);
          VkBufferMemoryBarrier bar2 = {};
          bar2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
          bar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          bar2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          bar2.buffer = dumpBuf->getSliceHandle().handle;
          bar2.offset = 0;
          bar2.size = dumpSize;
          bar2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          bar2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          ctx->getCommandList()->cmdPipelineBarrier(
            DxvkCmdBuffer::ExecBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &bar2, 0, nullptr);
          ctx->flushCommandList();
          m_device->waitForIdle();
          const uint32_t* raw = reinterpret_cast<const uint32_t*>(dumpBuf->mapPtr(0));
          if (raw) {
            // Log raw bytes as hex for 2 vertices
            const uint32_t wordsPerVertex = stride / 4;
            char hex[512] = {};
            int pos = 0;
            for (uint32_t v = 0; v < 2 && pos < 480; ++v) {
              pos += snprintf(hex + pos, sizeof(hex) - pos, "v%u=[", v);
              for (uint32_t w = 0; w < wordsPerVertex && pos < 480; ++w) {
                pos += snprintf(hex + pos, sizeof(hex) - pos, "%s%08X",
                  w > 0 ? " " : "", raw[v * wordsPerVertex + w]);
              }
              pos += snprintf(hex + pos, sizeof(hex) - pos, "] ");
            }
            // Also CPU-decode vertex 0 and 1 to compare with GPU output
            uint32_t u0_v0 = raw[0], u1_v0 = raw[1];
            uint32_t u0_v1 = raw[wordsPerVertex], u1_v1 = raw[wordsPerVertex + 1];
            auto cpuDecode = [](uint32_t u0, uint32_t u1) -> std::array<float,3> {
              uint32_t xi = u0 & 0x001FFFFFu;
              uint32_t yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u);
              uint32_t zi = u1 >> 10u;
              float s = 1.0f / 1024.0f;
              return {float(xi)*s - 1024.f, float(yi)*s - 1024.f, float(zi)*s - 2048.f};
            };
            auto d0 = cpuDecode(u0_v0, u1_v0);
            auto d1 = cpuDecode(u0_v1, u1_v1);
            Logger::info(str::format(
              "[interleaver] RAW VBUF DUMP stride=", stride,
              " verts=", input.vertexCount, " ", hex,
              " CPU_v0=(", d0[0], ",", d0[1], ",", d0[2], ")",
              " CPU_v1=(", d1[0], ",", d1[1], ",", d1[2], ")"));

            // Now readback the GPU OUTPUT (first 2 vertices after interleaver)
            const uint32_t outStride = args.outputStride;
            const VkDeviceSize outDumpSize = static_cast<VkDeviceSize>(outStride) * 2 * 4;
            DxvkBufferCreateInfo outDumpInfo;
            outDumpInfo.size = std::max<VkDeviceSize>(outDumpSize, 64);
            outDumpInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            outDumpInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            outDumpInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
            auto outDumpBuf = m_device->createBuffer(outDumpInfo,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              DxvkMemoryStats::Category::RTXBuffer, "output-dump");
            VkBufferMemoryBarrier outBar = {};
            outBar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            outBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            outBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            outBar.buffer = output.buffer->getSliceHandle().handle;
            outBar.offset = 0;
            outBar.size = outDumpSize;
            outBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            outBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 1, &outBar, 0, nullptr);
            ctx->copyBuffer(outDumpBuf, 0, output.buffer, 0, outDumpSize);
            VkBufferMemoryBarrier outBar2 = {};
            outBar2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            outBar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            outBar2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            outBar2.buffer = outDumpBuf->getSliceHandle().handle;
            outBar2.offset = 0;
            outBar2.size = outDumpSize;
            outBar2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            outBar2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_HOST_BIT,
              0, 0, nullptr, 1, &outBar2, 0, nullptr);
            ctx->flushCommandList();
            m_device->waitForIdle();
            const float* outData = reinterpret_cast<const float*>(outDumpBuf->mapPtr(0));
            if (outData) {
              Logger::info(str::format(
                "[interleaver] GPU OUTPUT: outStride=", outStride,
                " GPU_v0=(", outData[0], ",", outData[1], ",", outData[2], ")",
                " GPU_v1=(", outData[outStride], ",", outData[outStride+1], ",", outData[outStride+2], ")"));

              // NV-DXVK TF2 VIEWMODEL DEBUG: when this dispatch is bone-skinned,
              // also CPU-replay the SHADER's exact math (decode local_pos,
              // load bones at BLENDINDICES, blend by BLENDWEIGHT) and compare
              // against the GPU output. Mismatch = interleaver still diverges
              // from the native VS in some way. Keeping this gated to the
              // same throttle as the rest of the dump (sUintDumpCount<30).
              if (args.hasBoneTransform && args.hasBoneWeights
                  && input.boneIndexBuffer.defined()
                  && input.boneWeightBuffer.defined()
                  && input.boneMatrixBuffer.defined()) {
                // Read raw BI + BW for vertex 0 (the same vertex outData[0..2] reflects).
                const VkDeviceSize biSize = 4;
                const VkDeviceSize bwSize = 4;
                DxvkBufferCreateInfo biInfo;
                biInfo.size = 64;
                biInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                biInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
                biInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
                auto biDumpBuf = m_device->createBuffer(biInfo,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  DxvkMemoryStats::Category::RTXBuffer, "bi-dump");
                auto bwDumpBuf = m_device->createBuffer(biInfo,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  DxvkMemoryStats::Category::RTXBuffer, "bw-dump");
                // NV-DXVK NPC SKINNING DIAG: read BI/BW from EXACTLY where the
                // shader reads them. Shader does
                //   srcBoneIndex[vertexIndex * (boneIndexStride/4) + boneIndexOffsetUints]
                // For vertex 0 this collapses to byte
                //   slice.offset() + boneIndexOffsetUints*4
                // = slice.offset() + offsetFromSlice() (the BLENDINDICES byte
                // offset within the per-vertex stride). Reading at slice.offset()
                // alone reads the FIRST byte of vertex 0's stride, which for
                // interleaved streams isn't the BI bytes — produces bogus values.
                const VkDeviceSize biReadOff =
                  input.boneIndexBuffer.offset()
                    + VkDeviceSize(input.boneIndexBuffer.offsetFromSlice());
                const VkDeviceSize bwReadOff =
                  input.boneWeightBuffer.offset()
                    + VkDeviceSize(input.boneWeightBuffer.offsetFromSlice());
                ctx->copyBuffer(biDumpBuf, 0,
                  input.boneIndexBuffer.buffer(), biReadOff, biSize);
                ctx->copyBuffer(bwDumpBuf, 0,
                  input.boneWeightBuffer.buffer(), bwReadOff, bwSize);
                ctx->flushCommandList();
                m_device->waitForIdle();
                const uint32_t* biPtr = reinterpret_cast<const uint32_t*>(biDumpBuf->mapPtr(0));
                const uint32_t* bwPtr = reinterpret_cast<const uint32_t*>(bwDumpBuf->mapPtr(0));
                if (biPtr && bwPtr) {
                  const uint32_t bi = biPtr[0];
                  const uint32_t bw = bwPtr[0];
                  const uint32_t idx0 = (bi >> 0) & 0xFF;
                  const uint32_t idx1 = (bi >> 8) & 0xFF;
                  const uint32_t idx2 = (bi >> 16) & 0xFF;
                  // Decode signed-int16 weights with shader's (v+1)/32768 normalization.
                  auto signExt16 = [](uint32_t u) -> float {
                    return (u < 0x8000u) ? float(int(u)) : (float(int(u)) - 65536.0f);
                  };
                  const float w0 = (signExt16(bw & 0xFFFFu) + 1.0f) * (1.0f / 32768.0f);
                  const float w1 = (signExt16((bw >> 16u) & 0xFFFFu) + 1.0f) * (1.0f / 32768.0f);
                  const float w2 = 1.0f - w0 - w1;
                  Logger::info(str::format(
                    "[interleaver.skin] vertCount=", input.vertexCount,
                    " biStride=", input.boneIndexStrideBytes,
                    " bwStride=", input.boneWeightBuffer.stride(),
                    " v0_BI=(", idx0, ",", idx1, ",", idx2, ")",
                    " v0_BW=(", w0, ",", w1, ",", w2, ")",
                    " v0_LOCAL=(", d0[0], ",", d0[1], ",", d0[2], ")",
                    " v0_GPU_WORLD=(", outData[0], ",", outData[1], ",", outData[2], ")"));
                  // Read the bone matrix at idx0 (12 floats = 48 bytes).
                  // boneMatrixBuffer is float-typed; stride is in floats.
                  // NV-DXVK NPC SKINNING DIAG: read ALL THREE bones vertex 0
                  // weights to, so we can see if the GPU output reflects the
                  // full blended skinning and whether any of the bones read
                  // as zero matrices (diagnostic only, no behavior change).
                  // Reads 3 × 48 bytes at each bone's offset in the matrix
                  // buffer, performs CPU-side 3-bone weighted blend, compares
                  // against the GPU output. `bZ0/bZ1/bZ2` flag zero matrices.
                  const VkDeviceSize boneOff0 =
                    input.boneMatrixBuffer.offset() + idx0 * input.boneMatrixStrideBytes;
                  const VkDeviceSize boneOff1 =
                    input.boneMatrixBuffer.offset() + idx1 * input.boneMatrixStrideBytes;
                  const VkDeviceSize boneOff2 =
                    input.boneMatrixBuffer.offset() + idx2 * input.boneMatrixStrideBytes;
                  DxvkBufferCreateInfo bInfo;
                  bInfo.size = 64;
                  bInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                  bInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
                  bInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
                  auto allocBone = [&](const char* name) {
                    return m_device->createBuffer(bInfo,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      DxvkMemoryStats::Category::RTXBuffer, name);
                  };
                  auto bDumpBuf0 = allocBone("bone-dump-0");
                  auto bDumpBuf1 = allocBone("bone-dump-1");
                  auto bDumpBuf2 = allocBone("bone-dump-2");
                  ctx->copyBuffer(bDumpBuf0, 0,
                    input.boneMatrixBuffer.buffer(), boneOff0, 48);
                  ctx->copyBuffer(bDumpBuf1, 0,
                    input.boneMatrixBuffer.buffer(), boneOff1, 48);
                  ctx->copyBuffer(bDumpBuf2, 0,
                    input.boneMatrixBuffer.buffer(), boneOff2, 48);
                  ctx->flushCommandList();
                  m_device->waitForIdle();
                  const float* bm0 = reinterpret_cast<const float*>(bDumpBuf0->mapPtr(0));
                  const float* bm1 = reinterpret_cast<const float*>(bDumpBuf1->mapPtr(0));
                  const float* bm2 = reinterpret_cast<const float*>(bDumpBuf2->mapPtr(0));
                  auto isZeroMat = [](const float* m) -> bool {
                    if (!m) return true;
                    // "Zero" means the WHOLE matrix is ~0 — all rotation
                    // components AND translation. Partially-zero matrices
                    // (identity with no translation, etc.) are NOT zero.
                    const float mag =
                      std::abs(m[0]) + std::abs(m[1]) + std::abs(m[2])  + std::abs(m[3])
                    + std::abs(m[4]) + std::abs(m[5]) + std::abs(m[6])  + std::abs(m[7])
                    + std::abs(m[8]) + std::abs(m[9]) + std::abs(m[10]) + std::abs(m[11]);
                    return mag < 1e-6f;
                  };
                  auto applyMat = [](const float* m, const std::array<float, 3>& p) -> std::array<float, 3> {
                    if (!m) return {0.0f, 0.0f, 0.0f};
                    return {
                      m[0]*p[0] + m[1]*p[1] + m[2]*p[2]  + m[3],
                      m[4]*p[0] + m[5]*p[1] + m[6]*p[2]  + m[7],
                      m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11]};
                  };
                  const bool bZ0 = isZeroMat(bm0);
                  const bool bZ1 = isZeroMat(bm1);
                  const bool bZ2 = isZeroMat(bm2);
                  // Dump each bone's T + R0 so you can see what the GPU
                  // read. Key signal: if bone idx1 (typically the A-pose
                  // culprit slot 8-15 of a palette) shows T=(0,0,0)
                  // R0=(0,0,0), the staging→t30 copy never wrote it for
                  // this NPC — and we need to find where the data SHOULD
                  // come from.
                  Logger::info(str::format(
                    "[interleaver.skin.bones]",
                    " idx=(", idx0, ",", idx1, ",", idx2, ")",
                    " w=(", w0, ",", w1, ",", w2, ")",
                    " isZero=(", (bZ0?1:0), ",", (bZ1?1:0), ",", (bZ2?1:0), ")",
                    " b0.T=(", (bm0?bm0[3]:0), ",", (bm0?bm0[7]:0), ",", (bm0?bm0[11]:0), ")",
                    " b1.T=(", (bm1?bm1[3]:0), ",", (bm1?bm1[7]:0), ",", (bm1?bm1[11]:0), ")",
                    " b2.T=(", (bm2?bm2[3]:0), ",", (bm2?bm2[7]:0), ",", (bm2?bm2[11]:0), ")"));
                  // Full 3-bone CPU blend — compare against GPU output.
                  const auto p0 = applyMat(bm0, d0);
                  const auto p1 = applyMat(bm1, d0);
                  const auto p2 = applyMat(bm2, d0);
                  const float cx = p0[0]*w0 + p1[0]*w1 + p2[0]*w2;
                  const float cy = p0[1]*w0 + p1[1]*w1 + p2[1]*w2;
                  const float cz = p0[2]*w0 + p1[2]*w1 + p2[2]*w2;
                  const bool match3 =
                    std::abs(cx - outData[0]) < 0.5f
                    && std::abs(cy - outData[1]) < 0.5f
                    && std::abs(cz - outData[2]) < 0.5f;
                  Logger::info(str::format(
                    "[interleaver.skin.blend3]",
                    " CPU_blend3=(", cx, ",", cy, ",", cz, ")",
                    " GPU_v0=(", outData[0], ",", outData[1], ",", outData[2], ")",
                    " match=", (match3 ? 1 : 0)));

                  // NV-DXVK NPC SKINNING DIAG (suspect #1): are we reading
                  // bones at the same VkBuffer offset the shader is? Print
                  // slice metadata, then independently read 48 bytes from
                  // THREE locations in the raw underlying VkBuffer:
                  //  A) absolute offset 0 (start of the 393216-byte t30)
                  //  B) slice.offset()        (what we *think* is bone 0
                  //                            of the bound view)
                  //  C) idx0 * 48 (no slice offset added)
                  //                           (where the data would live
                  //                            if the shader ignores
                  //                            FirstElement/sub-slice)
                  // If (C).T matches GPU_v0 while (B) is zeros, the shader
                  // is reading from absolute offset 0 — the bone-matrix
                  // sub-slice base isn't being honored.
                  {
                    const VkDeviceSize sliceOff = input.boneMatrixBuffer.offset();
                    const VkDeviceSize sliceLen = input.boneMatrixBuffer.length();
                    const VkDeviceSize bufSize  =
                      input.boneMatrixBuffer.buffer() != nullptr
                        ? input.boneMatrixBuffer.buffer()->info().size : 0;
                    const VkDeviceSize biOff    = input.boneIndexBuffer.offset();
                    const VkDeviceSize bwOff    = input.boneWeightBuffer.defined()
                      ? input.boneWeightBuffer.offset() : VkDeviceSize(0);
                    Logger::info(str::format(
                      "[interleaver.skin.offsets]",
                      " boneMat.bufPtr=",
                      reinterpret_cast<uintptr_t>(input.boneMatrixBuffer.buffer().ptr()),
                      " boneMat.off=", sliceOff,
                      " boneMat.ofs=", input.boneMatrixBuffer.offsetFromSlice(),
                      " boneMat.len=", sliceLen,
                      " boneMat.bufSize=", bufSize,
                      " boneMatStride=", input.boneMatrixStrideBytes,
                      " bi.off=", biOff,
                      " bi.ofs=", input.boneIndexBuffer.offsetFromSlice(),
                      " bi.stride=", input.boneIndexStrideBytes,
                      " bi.idxOffUints=", args.boneIndexOffsetUints,
                      " bw.off=", bwOff,
                      " bw.ofs=", input.boneWeightBuffer.defined()
                        ? input.boneWeightBuffer.offsetFromSlice() : 0u,
                      " bw.stride=", input.boneWeightBuffer.defined()
                        ? input.boneWeightBuffer.stride() : 0u,
                      " bw.wOffUints=", args.boneWeightOffset,
                      " hasBoneWeights=", (args.hasBoneWeights ? 1 : 0),
                      " hasBoneTransform=", (args.hasBoneTransform ? 1 : 0),
                      " bonePerVertex=", (args.bonePerVertex ? 1 : 0),
                      " biReadOff=", biReadOff,
                      " bwReadOff=", bwReadOff));

                    auto readBoneAt = [&](VkDeviceSize absOff,
                                          const char* tag) {
                      if (input.boneMatrixBuffer.buffer() == nullptr) return;
                      if (absOff + 48 > bufSize) {
                        Logger::info(str::format(
                          "[interleaver.skin.absbone] tag=", tag,
                          " absOff=", absOff,
                          " OUT_OF_RANGE bufSize=", bufSize));
                        return;
                      }
                      DxvkBufferCreateInfo bi;
                      bi.size   = 64;
                      bi.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                      bi.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
                      bi.access = VK_ACCESS_TRANSFER_WRITE_BIT;
                      auto db = m_device->createBuffer(bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        DxvkMemoryStats::Category::RTXBuffer, "abs-bone-dump");
                      ctx->copyBuffer(db, 0,
                        input.boneMatrixBuffer.buffer(), absOff, 48);
                      ctx->flushCommandList();
                      m_device->waitForIdle();
                      const float* m = reinterpret_cast<const float*>(db->mapPtr(0));
                      if (m == nullptr) return;
                      const float magR =
                        std::abs(m[0])+std::abs(m[1])+std::abs(m[2])
                       +std::abs(m[4])+std::abs(m[5])+std::abs(m[6])
                       +std::abs(m[8])+std::abs(m[9])+std::abs(m[10]);
                      Logger::info(str::format(
                        "[interleaver.skin.absbone] tag=", tag,
                        " absOff=", absOff,
                        " T=(", m[3], ",", m[7], ",", m[11], ")",
                        " R0=(", m[0], ",", m[1], ",", m[2], ")",
                        " rotMag=", magR));
                    };
                    readBoneAt(0,
                               "raw0");
                    readBoneAt(sliceOff,
                               "sliceStart");
                    readBoneAt(VkDeviceSize(idx0) * VkDeviceSize(48),
                               "idx0_noSlice");
                    readBoneAt(VkDeviceSize(idx1) * VkDeviceSize(48),
                               "idx1_noSlice");
                    readBoneAt(VkDeviceSize(idx2) * VkDeviceSize(48),
                               "idx2_noSlice");
                  }
                }
              }
            }

            // Dump first 6 index buffer entries (first 2 triangles)
            if (input.indexBuffer.defined()) {
              const VkDeviceSize idxDumpSize = 12; // 6 × uint16
              DxvkBufferCreateInfo idxDumpInfo;
              idxDumpInfo.size = 16;
              idxDumpInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
              idxDumpInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
              idxDumpInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
              auto idxDumpBuf = m_device->createBuffer(idxDumpInfo,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "idx-dump");
              VkBufferMemoryBarrier idxBar = {};
              idxBar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
              idxBar.srcAccessMask = VK_ACCESS_INDEX_READ_BIT;
              idxBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
              idxBar.buffer = input.indexBuffer.buffer()->getSliceHandle().handle;
              idxBar.offset = input.indexBuffer.offset();
              idxBar.size = idxDumpSize;
              idxBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              idxBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              ctx->getCommandList()->cmdPipelineBarrier(
                DxvkCmdBuffer::ExecBuffer,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 1, &idxBar, 0, nullptr);
              ctx->copyBuffer(idxDumpBuf, 0,
                input.indexBuffer.buffer(), input.indexBuffer.offset(), idxDumpSize);
              VkBufferMemoryBarrier idxBar2 = {};
              idxBar2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
              idxBar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
              idxBar2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
              idxBar2.buffer = idxDumpBuf->getSliceHandle().handle;
              idxBar2.offset = 0;
              idxBar2.size = idxDumpSize;
              idxBar2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              idxBar2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              ctx->getCommandList()->cmdPipelineBarrier(
                DxvkCmdBuffer::ExecBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &idxBar2, 0, nullptr);
              ctx->flushCommandList();
              m_device->waitForIdle();
              const uint16_t* idx = reinterpret_cast<const uint16_t*>(idxDumpBuf->mapPtr(0));
              if (idx) {
                Logger::info(str::format(
                  "[interleaver] INDEX DUMP: tri0=[", idx[0], ",", idx[1], ",", idx[2],
                  "] tri1=[", idx[3], ",", idx[4], ",", idx[5], "]",
                  " maxVerts=", input.vertexCount));
              }
            }
          }
        }
      }
      // (legacy readback code removed)
      if (false) {
        static Rc<DxvkBuffer> s_readbackBuf;
        static uint32_t sReadbackCount = 0;
        // Read back first 3 vertices (3 positions × 3 floats × 4 bytes = 36 bytes)
        // But need to account for output stride (pos + normals + tc = 8 floats)
        const uint32_t outStride = args.outputStride; // in floats
        const VkDeviceSize readSize = static_cast<VkDeviceSize>(outStride) * 3 * 4;
        if (s_readbackBuf == nullptr || s_readbackBuf->info().size < readSize) {
          DxvkBufferCreateInfo readbackInfo;
          readbackInfo.size = std::max<VkDeviceSize>(readSize, 256);
          readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          readbackInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
          readbackInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
          s_readbackBuf = m_device->createBuffer(readbackInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "bone-readback");
        }
        if (sReadbackCount < 20) {
          ++sReadbackCount;
          // Barrier: wait for compute shader write to finish before transfer read
          VkBufferMemoryBarrier barrier = {};
          barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
          barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          barrier.buffer = output.buffer->getSliceHandle().handle;
          barrier.offset = 0;
          barrier.size = readSize;
          barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          ctx->getCommandList()->cmdPipelineBarrier(
            DxvkCmdBuffer::ExecBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
          // Copy first 3 vertices from output
          VkDeviceSize copySize = std::min<VkDeviceSize>(readSize, output.buffer->info().size);
          ctx->copyBuffer(s_readbackBuf, 0, output.buffer, 0, copySize);
          // Barrier: wait for transfer to finish before host read
          VkBufferMemoryBarrier barrier2 = {};
          barrier2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
          barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          barrier2.buffer = s_readbackBuf->getSliceHandle().handle;
          barrier2.offset = 0;
          barrier2.size = readSize;
          barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          ctx->getCommandList()->cmdPipelineBarrier(
            DxvkCmdBuffer::ExecBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &barrier2, 0, nullptr);
          // Force submit and wait
          ctx->flushCommandList();
          m_device->waitForIdle();
          const float* rb = reinterpret_cast<const float*>(s_readbackBuf->mapPtr(0));
          if (rb) {
            Logger::info(str::format(
              "[interleaver] GPU READBACK",
              " v0=(", rb[0], ",", rb[1], ",", rb[2], ")",
              " v1=(", rb[outStride], ",", rb[outStride+1], ",", rb[outStride+2], ")",
              " v2=(", rb[outStride*2], ",", rb[outStride*2+1], ",", rb[outStride*2+2], ")",
              " outStride=", outStride,
              " verts=", input.vertexCount));
          }
          // Also readback the bone matrix translation (bone 0, floats 3,7,11)
          static Rc<DxvkBuffer> s_boneReadback;
          static bool s_boneReadbackDone = false;
          if (!s_boneReadbackDone && input.boneMatrixBuffer.defined()) {
            s_boneReadbackDone = (sReadbackCount >= 15);
            DxvkBufferCreateInfo brInfo;
            brInfo.size = 48;
            brInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            brInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            brInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
            if (s_boneReadback == nullptr) {
              s_boneReadback = m_device->createBuffer(brInfo,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                DxvkMemoryStats::Category::RTXBuffer, "bonematrix-readback");
            }
            VkBufferMemoryBarrier bmBarrier = {};
            bmBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bmBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bmBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bmBarrier.buffer = input.boneMatrixBuffer.buffer()->getSliceHandle().handle;
            bmBarrier.offset = input.boneMatrixBuffer.offset();
            bmBarrier.size = 48;
            bmBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 1, &bmBarrier, 0, nullptr);
            ctx->copyBuffer(s_boneReadback, 0,
              input.boneMatrixBuffer.buffer(), input.boneMatrixBuffer.offset(), 48);
            VkBufferMemoryBarrier bmBarrier2 = {};
            bmBarrier2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bmBarrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bmBarrier2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            bmBarrier2.buffer = s_boneReadback->getSliceHandle().handle;
            bmBarrier2.offset = 0;
            bmBarrier2.size = 48;
            bmBarrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmBarrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_HOST_BIT,
              0, 0, nullptr, 1, &bmBarrier2, 0, nullptr);
            ctx->flushCommandList();
            m_device->waitForIdle();
            const float* bm = reinterpret_cast<const float*>(s_boneReadback->mapPtr(0));
            if (bm) {
              Logger::info(str::format(
                "[interleaver] BONE MATRIX readback: T=(", bm[3], ",", bm[7], ",", bm[11], ")",
                " R0=(", bm[0], ",", bm[1], ",", bm[2], ")",
                " R1=(", bm[4], ",", bm[5], ",", bm[6], ")",
                " R2=(", bm[8], ",", bm[9], ",", bm[10], ")",
                " draw=", sReadbackCount));
            }
          }
          // Also readback first 32 bytes of bone INDEX buffer (4 instances × 8 bytes)
          static bool s_boneIdxReadback = false;
          if (!s_boneIdxReadback && input.boneIndexBuffer.defined()) {
            s_boneIdxReadback = true;
            DxvkBufferCreateInfo biInfo;
            biInfo.size = 32;
            biInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            biInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            biInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
            auto biBuf = m_device->createBuffer(biInfo,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              DxvkMemoryStats::Category::RTXBuffer, "boneidx-readback");
            VkBufferMemoryBarrier biBarrier = {};
            biBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            biBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            biBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            biBarrier.buffer = input.boneIndexBuffer.buffer()->getSliceHandle().handle;
            biBarrier.offset = input.boneIndexBuffer.offset();
            biBarrier.size = 32;
            biBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            biBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 1, &biBarrier, 0, nullptr);
            ctx->copyBuffer(biBuf, 0, input.boneIndexBuffer.buffer(),
              input.boneIndexBuffer.offset(), 32);
            VkBufferMemoryBarrier biBarrier2 = {};
            biBarrier2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            biBarrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            biBarrier2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            biBarrier2.buffer = biBuf->getSliceHandle().handle;
            biBarrier2.offset = 0;
            biBarrier2.size = 32;
            biBarrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            biBarrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ctx->getCommandList()->cmdPipelineBarrier(
              DxvkCmdBuffer::ExecBuffer,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_HOST_BIT,
              0, 0, nullptr, 1, &biBarrier2, 0, nullptr);
            ctx->flushCommandList();
            m_device->waitForIdle();
            const uint16_t* bi = reinterpret_cast<const uint16_t*>(biBuf->mapPtr(0));
            if (bi) {
              // 4 instances × 4 uint16 each = 16 uint16 values
              Logger::info(str::format(
                "[interleaver] BONE INDEX BUFFER: ",
                "inst0=[", bi[0], ",", bi[1], ",", bi[2], ",", bi[3], "] ",
                "inst1=[", bi[4], ",", bi[5], ",", bi[6], ",", bi[7], "] ",
                "inst2=[", bi[8], ",", bi[9], ",", bi[10], ",", bi[11], "] ",
                "inst3=[", bi[12], ",", bi[13], ",", bi[14], ",", bi[15], "]"));
            }
          }
        }
      }
    } else {
      float dst[kNumVerticesToProcessOnCPU * kMaxInterleavedComponents];

      GeometryBufferData inputData(input);

      // Don't need these in CPU path as GeometryBufferData handles the offset
      args.positionOffset = 0;
      args.normalOffset = 0;
      args.texcoordOffset = 0;
      args.color0Offset = 0;

      // CPU path doesn't support bone transforms (GPU-only buffers).
      // Pass nullptr for bone buffers.
      const float* nullBoneMatrix = nullptr;
      const uint32_t* nullBoneIndex = nullptr;
      const uint32_t* nullBoneWeight = nullptr;
      for (uint32_t i = 0; i < input.vertexCount; i++) {
        interleaver::interleave(i, dst, inputData.positionData, inputData.normalData, inputData.texcoordData, inputData.vertexColorData, nullBoneMatrix, nullBoneIndex, nullBoneWeight, args);
      }

      ctx->writeToBuffer(output.buffer, 0, input.vertexCount * output.stride, dst);
    }

    uint32_t offset = 0;

    output.positionOffset = offset;
    offset += sizeof(float) * 3;

    if (input.normalBuffer.defined() || forceNormals) {
      output.hasNormals = true;
      output.normalOffset = offset;
      offset += sizeof(float) * 3;
    }

    if (input.texcoordBuffer.defined()) {
      output.hasTexcoord = true;
      output.texcoordOffset = offset;
      offset += sizeof(float) * 2;
    }

    if (input.color0Buffer.defined()) {
      output.hasColor0 = true;
      output.color0Offset = offset;
      offset += sizeof(uint32_t);
    }
  }

  float RtxGeometryUtils::computeMaxUVTileSize(const RasterGeometry& input, const Matrix4& objectToWorld) {
    ScopedCpuProfileZone();

    const void* pVertexData = input.positionBuffer.mapPtr((size_t)input.positionBuffer.offsetFromSlice());
    const uint32_t vertexCount = input.vertexCount;
    const size_t vertexStride = input.positionBuffer.stride();

    // R16G16_SFLOAT and other non-float32 texcoord formats cannot be safely read as Vector2 on the CPU.
    // The interleaver converts them to R32G32_SFLOAT on the GPU, but this function operates on raw
    // pre-interleave input geometry, so skip computation for unsupported formats.
    const VkFormat texFmt = input.texcoordBuffer.vertexFormat();
    if (texFmt != VK_FORMAT_R32G32_SFLOAT && texFmt != VK_FORMAT_R32G32B32_SFLOAT && texFmt != VK_FORMAT_R32G32B32A32_SFLOAT) {
      return NAN;
    }

    const void* pTexcoordData = input.texcoordBuffer.mapPtr((size_t)input.texcoordBuffer.offsetFromSlice());
    const size_t texcoordStride = input.texcoordBuffer.stride();

    const void* pIndexData = input.indexBuffer.mapPtr((size_t)input.indexBuffer.offsetFromSlice());
    const uint32_t indexCount = input.indexCount;
    const size_t indexStride = input.indexBuffer.stride();

    if (pVertexData == nullptr || pTexcoordData == nullptr) {
      return NAN;
    }

    float maxUvTileSizeSqr = 0.f;
    const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
    const uint8_t* pTexcoord = static_cast<const uint8_t*>(pTexcoordData);
    switch (input.topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      if (input.indexCount > 0 && pIndexData != nullptr) {
        maxUvTileSizeSqr = calcMaxUvTileSizeSqrIndexed(indexCount, objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride, pIndexData, indexStride);
      } else {
        maxUvTileSizeSqr = calcMaxUvTileSizeSqrTriangles(vertexCount, objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride);
      }
      break;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      maxUvTileSizeSqr = calcMaxUvTileSizeSqrTriangleStrip(vertexCount, objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride);
      break;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      maxUvTileSizeSqr = calcMaxUvTileSizeSqrTriangleFan(vertexCount, objectToWorld, pVertex, vertexStride, pTexcoord, texcoordStride);
      break;
    default:
      ONCE(Logger::err("computeMaxUVTileSize: unsupported topology"));
      return 0;
    }

    return std::sqrtf(maxUvTileSizeSqr);
  }

  void RtxGeometryUtils::dispatchSmoothNormals(const Rc<DxvkContext>& ctx, const RasterGeometry& input, RaytraceGeometry& geo) {
    ScopedGpuProfileZone(ctx, "smoothNormals");

    if (!geo.positionBuffer.defined() || !geo.indexBuffer.defined() || !geo.normalBuffer.defined()) {
      ONCE(Logger::warn("dispatchSmoothNormals: geometry missing required buffers (position, index, or normal)"));
      return;
    }

    const uint32_t numTriangles = geo.calculatePrimitiveCount();
    if (numTriangles == 0 || geo.vertexCount == 0) {
      return;
    }

    // Compute hash table size as next power-of-two >= numVertices * 4 (load factor < 0.25)
    uint32_t hashTableSize = std::max(geo.vertexCount * 4u, 256u);
    hashTableSize--;
    hashTableSize |= hashTableSize >> 1;
    hashTableSize |= hashTableSize >> 2;
    hashTableSize |= hashTableSize >> 4;
    hashTableSize |= hashTableSize >> 8;
    hashTableSize |= hashTableSize >> 16;
    hashTableSize++;

    SmoothNormalsArgs params {};
    params.indexStride = (geo.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) ? 2 : 4;
    params.numTriangles = numTriangles;
    params.numVertices = geo.vertexCount;
    params.useShortIndices = (geo.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16) ? 1 : 0;
    params.phase = 0;
    params.hashTableSize = hashTableSize;

    assert(geo.vertexCount == input.vertexCount);

    // Decide whether to use the CPU path.  The input raster buffers must be host-visible
    // and not pending GPU write.  For small meshes the CPU path avoids GPU dispatch overhead.
    const bool mustUseGPU = input.indexBuffer.mapPtr() == nullptr || input.positionBuffer.mapPtr() == nullptr;
    const bool pendingGpuWrite = input.positionBuffer.isPendingGpuWrite() || input.indexBuffer.isPendingGpuWrite();

    const uint32_t kMaxTrianglesForCPU = 512;
    const bool useCPU = numTriangles <= kMaxTrianglesForCPU && !pendingGpuWrite && !mustUseGPU;

    if (useCPU) {
      // --- CPU path: uses the same shared functions as the GPU shader ---
      const float* srcPosition = reinterpret_cast<const float*>(input.positionBuffer.mapPtr(0));
      const uint32_t* srcIndex = reinterpret_cast<const uint32_t*>(input.indexBuffer.mapPtr(0));

      // Remap offsets to the raster input buffers
      params.positionOffset = input.positionBuffer.offsetFromSlice();
      params.positionStride = input.positionBuffer.stride();
      params.normalOffset = 0; // CPU output writes per-vertex via writeToBuffer, so offset/stride handled externally
      params.normalStride = 0; 
      params.indexOffset = input.indexBuffer.offsetFromSlice();

      // Allocate and zero the hash table on the CPU
      std::vector<int32_t> hashTable(hashTableSize * 4, 0);

      // Phase 1: Accumulate face normals (shared function)
      for (uint32_t tri = 0; tri < numTriangles; tri++) {
        smoothNormalsAccumulate(tri, srcPosition, srcIndex, hashTable.data(), params);
      }

      // Phase 2: Scatter & normalize — writes encoded normals to the GPU buffer
      float dstNormal;
      for (uint32_t v = 0; v < geo.vertexCount; v++) {
        smoothNormalsScatter(v, srcPosition, &dstNormal, hashTable.data(), params);
        ctx->writeToBuffer(geo.normalBuffer.buffer(), geo.normalBuffer.offsetFromSlice() + v * geo.normalBuffer.stride(), sizeof(dstNormal),  &dstNormal);
      }
    } else {
      // --- GPU path ---
      params.positionOffset = geo.positionBuffer.offsetFromSlice();
      params.positionStride = geo.positionBuffer.stride();
      params.normalOffset = geo.normalBuffer.offsetFromSlice();
      params.normalStride = geo.normalBuffer.stride();
      params.indexOffset = geo.indexBuffer.offsetFromSlice();

      // Sub-allocate from a pooled device-local buffer for the hash table (4 ints per entry: tag + 3 normal components)
      const VkDeviceSize hashBufSize = hashTableSize * 4 * sizeof(int);
      DxvkBufferSlice hashTableSlice = m_pSmoothNormalsHashData->alloc(16, hashBufSize);

      ctx->bindResourceBuffer(SMOOTH_NORMALS_BINDING_POSITION_RO, DxvkBufferSlice(geo.positionBuffer.buffer()));
      ctx->bindResourceBuffer(SMOOTH_NORMALS_BINDING_NORMAL_RW, DxvkBufferSlice(geo.normalBuffer.buffer()));
      ctx->bindResourceBuffer(SMOOTH_NORMALS_BINDING_INDEX_INPUT, DxvkBufferSlice(geo.indexBuffer.buffer()));
      ctx->bindResourceBuffer(SMOOTH_NORMALS_BINDING_HASH_TABLE, hashTableSlice);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SmoothNormalsShader::getShader());
      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

      const VkExtent3D vertexWorkgroups = util::computeBlockCount(VkExtent3D { params.numVertices, 1, 1 }, VkExtent3D { 128, 1, 1 });
      const VkExtent3D triangleWorkgroups = util::computeBlockCount(VkExtent3D { params.numTriangles, 1, 1 }, VkExtent3D { 128, 1, 1 });

      // Clear the hash table slice to zero using vkCmdFillBuffer
      ctx->clearBuffer(hashTableSlice.buffer(), hashTableSlice.offset(), hashBufSize, 0);

      // Phase 1: Accumulate area-weighted face normals into hash table by position
      params.phase = 1;
      ctx->pushConstants(0, sizeof(SmoothNormalsArgs), &params);
      ctx->dispatch(triangleWorkgroups.width, triangleWorkgroups.height, triangleWorkgroups.depth);

      // Phase 2: Each vertex reads its smoothed normal from hash table, normalizes, writes encoded output
      params.phase = 2;
      ctx->pushConstants(0, sizeof(SmoothNormalsArgs), &params);
      ctx->dispatch(vertexWorkgroups.width, vertexWorkgroups.height, vertexWorkgroups.depth);
    }

    // Smooth normals always outputs octahedral-encoded normals (single R32_UINT per vertex).
    // Update the buffer format metadata so downstream readers (surface_interaction, skinning)
    // correctly decode the normals.
    geo.normalBuffer.setVertexFormat(VK_FORMAT_R32_UINT);
  }
}
