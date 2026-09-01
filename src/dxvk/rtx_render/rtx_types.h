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
#pragma once

#include "rtx_constants.h"
#include "rtx_utils.h"
#include "rtx_materials.h"
#include "rtx_hashing.h"
#include "rtx_camera.h"
#include "vulkan/vulkan_core.h"
#include "../../util/util_bounding_box.h"
#include "../../util/util_threadpool.h"
#include "../../util/util_spatial_map.h"

#include <inttypes.h>
#include <vector>
#include <future>
#include <chrono>
#include <cmath>

using remixapi_MaterialHandle = struct remixapi_MaterialHandle_T*;
using remixapi_MeshHandle = struct remixapi_MeshHandle_T*;

namespace dxvk 
{
class RtCamera;
class RtInstance;
struct RtLight;
class GraphInstance;
struct RtxVSConstants;
struct RtxPSConstants;
struct ReplacementInstance;

struct ShaderProgramInfo {
  uint32_t m_majorVersion = 0;
  uint32_t m_minorVersion = 0;

  ShaderProgramInfo() = default;
  ShaderProgramInfo(uint32_t major, uint32_t minor)
    : m_majorVersion{major}, m_minorVersion{minor} {}

  uint32_t majorVersion() const { return m_majorVersion; }
  uint32_t minorVersion() const { return m_minorVersion; }
};

using RasterBuffer = GeometryBuffer<Raster>;
using RaytraceBuffer = GeometryBuffer<Raytrace>;

// DLFG async compute overlap: max of 2 frames in flight
// (set to 1 to serialize graphics and async compute queues)
constexpr uint32_t kDLFGMaxGPUFramesInFlight = 2;

// A container for the runtime instance that maps to a prim in a replacement heirarchy.
class PrimInstance {
public:
  enum class Type : uint8_t {
    Instance,
    Light,
    Graph,
    None
  };
  // Use `Entity()` to create a nullptr Entity.
  PrimInstance() {}

  // Default copy/move/destructors are fine - this just contains typed weak pointers.
  PrimInstance(const PrimInstance&) = default;
  PrimInstance(PrimInstance&&) noexcept = default;
  PrimInstance& operator=(const PrimInstance&) = default;
  PrimInstance& operator=(PrimInstance&&) noexcept = default;
  ~PrimInstance() = default;

  // Instance constructor, getter
  explicit PrimInstance(RtInstance* instance);
  RtInstance* getInstance() const;

  // Light constructor, getter
  explicit PrimInstance(RtLight* light);
  RtLight* getLight() const;

  // Graph constructor, getter
  explicit PrimInstance(GraphInstance* graph);
  GraphInstance* getGraph() const;

  // Untyped utilities.
  PrimInstance(void* owner, Type type);

  Type getType() const;
  void* getUntyped() const;
  void setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex);

private:
  union EntityPtr {
    void* untyped = nullptr;
    RtInstance* instance;
    RtLight* light;
    GraphInstance* graph;
  } m_ptr;
  Type m_type = Type::None;
};
std::ostream& operator << (std::ostream& os, PrimInstance::Type type);

struct ReplacementInstance {
  // Lifecycle note:
  // Currently, ReplacementInstances are created the first time a given replaced draw call
  // is rendered.  A single entity (a light or instance) is designated as the 'root'.
  // When that entity is destroyed, the ReplacementInstance is destroyed.
  // Unfortunately, lights and instances aren't always destroyed at the same time, or
  // in the same order they were created.  To accomodate that, when non-root entities
  // are deleted, they remove themselves from the `entities` vector.  Similarly, when
  // the root is deleted, all entities remaining in the vector will have their pointer
  // to the ReplacementInstance set to nullptr.
  // TODO(REMIX-4226): In the future, draw calls should be tracked and destroyed based
  // on the pre-replacement draw call, so that everything in a ReplacementInstance gets
  // destroyed at the same time.  When that change is made, the original tracked draw
  // call should own this ReplacementInstance.

  static constexpr uint32_t kInvalidReplacementIndex = UINT32_MAX;

  ~ReplacementInstance();

  void clear();

  std::vector<PrimInstance> prims;
  PrimInstance root;

  void setup(PrimInstance newRoot, size_t numPrims);
};

// Wrapper utility to share the code for handling replacementInstance ownership.
class PrimInstanceOwner {
public:
  PrimInstanceOwner() = default;
  // NOTE: primInstanceOwner is not safe to copy - the RtInstance, RtLight, etc that holds the PrimInstanceOwner
  //       would have a different address after copying, so the PrimInstanceOwner would point to the wrong object.
  PrimInstanceOwner(const PrimInstanceOwner& other) = delete;
  PrimInstanceOwner& operator=(const PrimInstanceOwner& other) = delete;

  ~PrimInstanceOwner() {
    // m_replacementInstance should always be properly cleaned up before the PrimInstanceOwner 
    // is destroyed. If this is hit, then whatever deleted the object holding the
    // primInstanceOwner needs to call setReplacementInstance(nullptr...) before doing that 
    // deletion.  If not, there will probably be use-after-free bugs later on.
    assert(m_replacementInstance == nullptr);
  }

  bool isRoot(const void* owner) const;
  void setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex, void* owner, PrimInstance::Type type);
  ReplacementInstance* getOrCreateReplacementInstance(void* owner, PrimInstance::Type type, size_t index, size_t numPrims);
  ReplacementInstance* getReplacementInstance() const { return m_replacementInstance; }
  size_t getReplacementIndex() const { return m_replacementIndex; }
  bool isSubPrim() const {
    if (m_replacementInstance == nullptr) {
      return false;
    } else {
      return m_replacementIndex != ReplacementInstance::kInvalidReplacementIndex &&
        m_replacementInstance->root.getUntyped() != m_replacementInstance->prims[m_replacementIndex].getUntyped();
    }
  }
private:
  ReplacementInstance* m_replacementInstance = nullptr;
  size_t m_replacementIndex = ReplacementInstance::kInvalidReplacementIndex;
};

// NOTE: Needed to move this here in order to avoid
// circular includes.  This probably requires a 
// general cleanup.
// NV-DXVK [perf]: copy-on-write handle around the bone matrix palette.
//
// Every skinned draw used to build its own std::vector<Matrix4> — a fresh
// ~16 KB allocation plus 256 float3x4 -> Matrix4 conversions — even though a
// multi-submesh model submits the SAME palette many times per frame (the TF2
// dropship alone is ~14 submeshes). Holding the payload behind a shared_ptr
// lets those draws share one converted palette: the per-draw cost collapses to
// a refcount bump, and DrawCallState copies get cheaper as a side effect.
//
// Sharing is only safe if nobody mutates a palette another draw is holding, so
// every mutating entry point detaches first (classic COW). Read access is
// unchanged, which is why the existing consumers compile untouched: operator[],
// size(), empty() all behave exactly like the vector did.
// Counts reads of an unmaterialised / out-of-range bone palette. See
// BonePalette::operator[]. Defined in rtx_types.cpp.
extern std::atomic<uint64_t> g_bonePaletteOobReads;

class BonePalette {
public:
  bool empty() const { return m_data == nullptr || m_data->empty(); }
  size_t size() const { return m_data != nullptr ? m_data->size() : 0u; }

  // Element access is const-ONLY, deliberately. A non-const operator[] would
  // call detach() and deep-copy the palette on every ordinary read — including
  // the per-draw diagnostics, which take a non-const DrawCallState — silently
  // undoing the sharing and adding a 16 KB copy per draw. Every reader in the
  // codebase only reads, so const-only keeps them all on the shared path; a
  // caller that genuinely needs to mutate must say so via mutableVec().
  // NV-DXVK [BonePalOob probe]: the palette is no longer materialised on every
  // skinned draw, so ANY reader that indexes it without checking size() is now
  // reading out of bounds. Rather than guess which one, count it here — every
  // read in the codebase funnels through this operator — and report identity
  // instead of dereferencing null. g_bonePaletteOobReads is logged by
  // [BonePaletteShare]; a non-zero count names this as the fault.
  const Matrix4& operator[](size_t i) const {
    if (m_data == nullptr || i >= m_data->size()) {
      g_bonePaletteOobReads.fetch_add(1u, std::memory_order_relaxed);
      static const Matrix4 kIdentity;
      return kIdentity;
    }
    return (*m_data)[i];
  }

  // Underlying vector, for APIs that take a std::vector<Matrix4> directly.
  const std::vector<Matrix4>& vec() const {
    static const std::vector<Matrix4> kEmpty;
    return m_data != nullptr ? *m_data : kEmpty;
  }

  // Explicit mutation: detaches from any shared payload first, so writes can
  // never be observed by another draw still holding the old palette.
  std::vector<Matrix4>& mutableVec() { detach(); return *m_data; }

  // Adopt an already-built palette — the sharing fast path. Callers must treat
  // the payload as immutable from here on (mutableVec() enforces that by COW).
  void adopt(const std::shared_ptr<std::vector<Matrix4>>& shared) { m_data = shared; }
  const std::shared_ptr<std::vector<Matrix4>>& shared() const { return m_data; }

private:
  // Materialise a private copy when the payload is shared (or absent), so a
  // mutation can never be observed by another draw still holding the old one.
  void detach() {
    if (m_data == nullptr) {
      m_data = std::make_shared<std::vector<Matrix4>>();
    } else if (m_data.use_count() > 1) {
      m_data = std::make_shared<std::vector<Matrix4>>(*m_data);
    }
  }

  std::shared_ptr<std::vector<Matrix4>> m_data;
};

struct SkinningData {
  BonePalette pBoneMatrices;
  uint32_t numBones = 0;
  uint32_t numBonesPerVertex = 0;
  XXH64_hash_t boneHash = 0;
  uint32_t minBoneIndex = 0; // This is the smallest index of all bones actually used by vertex data

  void computeHash() {
    if (numBones > 0) {
      assert(minBoneIndex >= 0);
      const Matrix4* firstBone = &pBoneMatrices[minBoneIndex];
      assert(numBones > minBoneIndex);
      boneHash = XXH3_64bits(firstBone, (numBones - minBoneIndex) * sizeof(Matrix4));
    } else {
      boneHash = 0;
    }
  }
};


// Stores the geometry data representing a raytracable object
// Valid until the object is destroyed.
struct RaytraceGeometry {
  // Cached hashes from draw call on last update
  GeometryHashes hashes;

  XXH64_hash_t lastBoneHash = 0;

  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  VkCullModeFlags cullMode = VkCullModeFlags(0);
  VkFrontFace frontFace = VkFrontFace(0);

  RaytraceBuffer positionBuffer;
  RaytraceBuffer previousPositionBuffer;
  RaytraceBuffer normalBuffer;
  RaytraceBuffer texcoordBuffer;
  RaytraceBuffer color0Buffer;
  RaytraceBuffer indexBuffer;
  // NV-DXVK: TF2 worldspace VGUI auxiliary structured buffers. Held as
  // RaytraceBuffer for the bindless tracking pass in
  // SceneManager::updateBufferCache; not part of the interleaved per-vertex
  // output (the slang VGUI evaluator reads from these directly via
  // BUFFER_ARRAY at hit time, indexed by the int4 packed indices in the
  // BLAS-side VGUI extras).
  RaytraceBuffer vguiFontBoundsBuffer;
  RaytraceBuffer vguiImgBoundsBuffer;
  RaytraceBuffer vguiStylesBuffer;
  // NV-DXVK: present when the source RasterGeometry had texcoord1Buffer
  // defined and the interleaver wrote a second UV slot. Drives Surface's
  // hasLightmap flag so the shader knows to read the lightmap UV from
  // (texcoordOffset + 8 bytes) of the same interleaved output.
  bool hasTexcoord1 = false;
  // NV-DXVK: TF2 worldspace VGUI per-vertex extras present (8 floats appended
  // at the end of the interleaved buffer when source RasterGeometry had
  // vguiLayoutEnable). Drives Surface::isVgui so the slang VGUI evaluator
  // knows to fetch from (vguiOffset .. vguiOffset+8) at hit time.
  bool hasVgui = false;
  // Element index (interleaved-buffer offset / sizeof(float)) of the first
  // VGUI extra float; surface decode reads (vguiOffset+0..3) as the
  // TC1.zw + TC2.xy block, and (vguiOffset+4..7) as asfloat-encoded int4
  // glyph/style/image indices.
  uint32_t vguiOffset = 0;

  uint32_t positionBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t normalBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t texcoordBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t color0BufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t indexBufferIndex = kSurfaceInvalidBufferIndex;
  // NV-DXVK: bindless storage-buffer indices for the 3 VGUI structured-
  // buffer SRVs captured at draw-capture time. Populated by SceneManager
  // via m_bufferCache.track(); slang VGUI evaluator reads them via
  // BUFFER_ARRAY at hit time.
  uint32_t vguiFontBoundsBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t vguiImgBoundsBufferIndex = kSurfaceInvalidBufferIndex;
  uint32_t vguiStylesBufferIndex = kSurfaceInvalidBufferIndex;

  Rc<DxvkBuffer> historyBuffer[2] = {nullptr};
  Rc<DxvkBuffer> indexCacheBuffer = nullptr;

  // Set to true after the smooth normals compute pass has been applied to this geometry.
  // Used to avoid redundant recomputation on subsequent frames for static geometry.
  bool smoothNormalsApplied = false;

  // NV-DXVK [s2s mangle/black FIX B]: true if this geometry's last BLAS-input
  // cache (index/vertex copy) was taken while a SOURCE buffer still had an
  // in-flight GPU write (engine upload not complete). Such a bake reads
  // zero/garbage (collapse->black / explode->mangle). processGeometryInfo forces
  // a re-cache (kUpdateBVH) every frame while this is set, instead of letting
  // kUpdateInstance freeze the bad bake, until a bake lands with the source
  // ready (then it clears). See [TrimCache] srcIdxPend/srcPosPend.
  bool pendingSrcBake = false;
  // NV-DXVK [flicker V8 follow-up: capture feedback]: how many consecutive
  // recovery re-bakes ran without a captured source landing. Safety valve —
  // if the capture feedback loop cannot deliver (pool exhaustion, feature
  // off mid-run), give up after a few attempts instead of reinstating the
  // unbounded per-frame re-bake storm FIX B's termination fix removed.
  // Reset to 0 by the first captured bake.
  uint8_t pendingSrcBakeAttempts = 0;

  bool usesIndices() const {
    return indexBuffer.defined();
  }

  uint32_t calculatePrimitiveCount() const {
    return (usesIndices() ? indexCount : vertexCount) / 3;
  }
};

// Stores a snapshot of the geometry state for a draw call.
// WARNING: Usage is undefined after the drawcall this was 
//          generated from has finished executing on the GPU
struct RasterGeometry {
  GeometryHashes hashes;
  Future<GeometryHashes> futureGeometryHashes;

  // Actual vertex/index count (when applicable) as calculated by geo-engine
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;

  // Copy of the bones per vertex from SkinningState.
  // This allows replacements to have different values from the original.
  uint32_t numBonesPerVertex = 0;

  // Hashed values
  VkPrimitiveTopology topology = VkPrimitiveTopology(0);
  VkCullModeFlags cullMode = VkCullModeFlags(0);
  VkFrontFace frontFace = VkFrontFace(0);

  // Used by replacements mostly, to force the cull bit to that set by the geometry data
  bool forceCullBit = false;

  RasterBuffer positionBuffer;
  RasterBuffer normalBuffer;
  RasterBuffer texcoordBuffer;
  // NV-DXVK: TEXCOORD1 / lightmap UV input. Captured by the D3D11 IA shim
  // when the wall VS layout declares a second TEXCOORD attribute (e.g.
  // VS_e7abcf4e in TF2). When defined, the interleaver decodes it (uint
  // → float * 1/65535 for Source-style packed lightmap UV) and writes it
  // adjacent to the primary texcoord in the output buffer.
  RasterBuffer texcoord1Buffer;

  // NV-DXVK: single source of truth for whether the lightmap UV path is
  // safe to plumb. The interleaver carries TC1's own stride+format
  // (packed into texcoord1StrideFormat in InterleaveGeometryArgs), so
  // TC0 and TC1 are allowed to differ — they often do in TF2 (e.g.
  // TC0=R32G32_UINT, TC1=R16G16_UINT). We only require the lightmap
  // format to be one convertLightmapTexcoord can decode. When this
  // returns false the lightmap UV slot is omitted from the interleaved
  // output entirely — Surface::hasLightmap stays 0 and the shader
  // doesn't read any garbage bytes at (texcoordOffset + 8).
  bool canPlumbLightmapUv() const {
    if (!texcoord1Buffer.defined()) return false;
    if (!texcoordBuffer.defined()) return false;
    const VkFormat fmt = texcoord1Buffer.vertexFormat();
    return fmt == VK_FORMAT_R16G16_UINT
        || fmt == VK_FORMAT_R32G32_UINT
        || fmt == VK_FORMAT_R32G32_SFLOAT;
  }
  RasterBuffer color0Buffer;
  RasterBuffer indexBuffer;
  RasterBuffer blendWeightBuffer;
  RasterBuffer blendIndicesBuffer;

  // NV-DXVK: Source Engine 2 bone matrix buffer (VS SRV t30, stride=48)
  // and per-instance bone index buffer (slot 1, R16G16B16A16_UINT)
  RasterBuffer boneMatrixBuffer;
  RasterBuffer boneIndexBuffer;
  // NV-DXVK (TF2 skinned chars): per-vertex bone weight stream (e.g. R16G16
  // UNORM or R8G8B8A8 UNORM). When defined, the interleaver does 4-bone
  // weighted skinning instead of single-bone-index lookup.
  RasterBuffer boneWeightBuffer;
  // Number of bone indices per vertex in boneIndexBuffer (4 for RGBA8_UINT
  // skinned characters; 1 for single-index BSP batches).
  uint32_t boneIndexComponentCount = 0;
  uint32_t boneInstanceIndex = 0;  // instance index for bone lookup (BLAS cache-key differentiator)
  // NV-DXVK (TF2 per-instance skinning): base added to every per-vertex bone
  // index before palette lookup, i.e. palette[BLENDINDICES + boneIndexBase].
  // TF2's instanced skinned draws (the widow dropships) pack a per-instance
  // COLOR1.y here (e.g. 288 / 352) so two instances skin from different bone
  // sub-ranges. 0 = no offset (all non-instanced skinned draws unchanged).
  uint32_t boneIndexBase = 0;
  // NV-DXVK [GPU per-instance bone base]: when defined, the per-vertex bone base is
  // read GPU-side in the interleaver from this buffer (the per-instance COLOR1/I
  // R16G16B16A16_UINT stream) at byte boneBaseByteOffset (= COLOR1.y), instead of
  // the CPU-read boneIndexBase. The per-instance VB is DYNAMIC and the game renames
  // it under us, so a CPU read in SubmitDraw races the rename and returns another
  // draw's bytes; the GPU reads the slice bound to THIS draw, eliminating the race.
  RasterBuffer boneBaseBuffer;
  uint32_t boneBaseByteOffset = 0;  // byte offset of this instance's COLOR1.y in boneBaseBuffer's slice
  // NV-DXVK (TF2 BSP / batched props): per-vertex instance index lookup.
  // bonePerVertex=true: each vertex's COLOR1 picks its own transform from
  // boneMatrixBuffer (which is g_modelInst SRV t31, stride=208).
  bool     bonePerVertex        = false;
  uint32_t boneMatrixStrideBytes = 0;  // 0 = use shader default 48 (= float3x4)
  uint32_t boneIndexStrideBytes  = 0;  // 0 = use shader default 8 (R16G16B16A16_UINT)
  uint32_t boneIndexMask         = 0;  // 0 = use shader default 0xFFFF (16-bit)

  // NV-DXVK: TF2 worldspace VGUI vertex layout. Set true by D3D11Rtx
  // ::SubmitDraw when the bound PS is identified as a VGUI shader (font
  // Texture + g_fontBounds + g_imgBounds RDEF resources — see
  // FillMaterialData::sourceIsUnlitUI). When set, the interleaver writes
  // 8 extra per-vertex floats covering TEXCOORD1.zw + TEXCOORD2.xy +
  // TEXCOORD3.xyzw (int4 → bit-cast float). Used by the slang VGUI
  // evaluator to compose the original PS's per-pixel output.
  bool     vguiLayoutEnable      = false;
  // TEXCOORD3 (semantic-named per the D3D11 input layout) source — int4
  // packed glyph/style/image indices. Bound to the VGUI int4 stream when
  // vguiLayoutEnable is true.
  RasterBuffer vguiTexcoord3Buffer;
  uint32_t vguiTexcoord3Offset   = 0;  // bytes
  uint32_t vguiTexcoord3Stride   = 0;  // bytes
  // NV-DXVK: TF2 VGUI TEXCOORD2 — R32G32_SFLOAT glyph dimensions / scale.
  // Captured by D3D11Rtx::SubmitDraw alongside vguiTexcoord3Buffer when
  // the VGUI signature is detected. The interleaver writes TC2.xy into
  // the VGUI extras slot at offset +2/+3 (after secondaryQuadPos at
  // offset +0/+1).
  RasterBuffer vguiGlyphDimsBuffer;
  // NV-DXVK: TF2 worldspace VGUI auxiliary structured-buffer SRVs captured
  // at FillMaterialData time (PS slots t2/t3/t4 per the RDEF triple-match
  // detection):
  //   g_fontBounds  (t2, stride 16)  — float4{ mins.xy, maxs.xy } per glyph
  //   g_imgBounds   (t3, stride 16)  — float4{ mins.xy, size.xy } per image
  //   g_styles      (t4, stride 96)  — per-style color/blend/etc record
  // SceneManager tracks these in m_bufferCache to obtain bindless storage-
  // buffer indices that get stamped onto RtSurface's vgui*BufferIndex
  // fields. The slang VGUI evaluator reads them via BUFFER_ARRAY at hit
  // time, indexed by the int4 packed indices held in the BLAS-side
  // VGUI extras (vguiTexcoord3Buffer).
  RasterBuffer vguiFontBoundsBuffer;
  RasterBuffer vguiImgBoundsBuffer;
  RasterBuffer vguiStylesBuffer;

  AxisAlignedBoundingBox boundingBox;
  Future<AxisAlignedBoundingBox> futureBoundingBox;

  remixapi_MaterialHandle externalMaterial = nullptr;

  // NV-DXVK: CPU snapshot of index data captured on the main/deferred context
  // thread at SubmitDraw time (before any subsequent Map/DISCARD can rename
  // the source D3D11 buffer's physical slice). When populated,
  // cacheIndexDataOnGPU uploads from this instead of doing a racy GPU→GPU
  // copy from the (possibly renamed) source.
  // shared_ptr so copying RasterGeometry into EmitCs lambda captures is cheap.
  std::shared_ptr<std::vector<uint8_t>> indexDataSnapshot;

  // NV-DXVK [perf, GPU index stash]: replacement for the CPU snapshot above on
  // the default path. The CPU snapshot streaming-read every dynamic IB range
  // out of write-combined memory on the game thread (~20 ms/frame in TF2),
  // then the CS side scanned + re-uploaded the same bytes. Instead, SubmitDraw
  // sets indexNeedsGpuStash and commitGeometryToRT records a GPU->GPU copy of
  // the draw's index range into this per-draw stash buffer. That record point
  // is IN-ORDER on the CS stream relative to the draw and any later
  // Map(DISCARD) rename replays, so the logical->physical slice resolution is
  // exactly the one the rasterized draw used — the same correctness argument
  // as the draw itself. (The prepScene-time copy the snapshot guarded against
  // was racy precisely because it ran at END of frame, after later rename
  // replays.) Consumers (cacheIndexDataOnGPU, the kUpdateBVH index refresh)
  // copy stash -> indexCacheBuffer with a transfer barrier instead of
  // uploading CPU bytes. The handle keeps the stash alive as long as the
  // BlasEntry input references this geometry (pendingSrcBake re-caches
  // included). RTX_IDX_CPU_SNAPSHOT=1 restores the old CPU snapshot path.
  //
  // POOLED (see IndexStashPool in rtx_scene_manager.h). The first version of
  // this allocated a fresh device-local VkBuffer per stashed draw — ~600
  // createBuffer calls per frame, each taking the memory-allocator lock on the
  // CS thread, each kept alive by the BlasEntry that referenced it. That is a
  // VRAM-churn engine, not a cache. Buffers now come from a size-classed free
  // list and return to it when the last RasterGeometry copy referencing them
  // dies, so steady state performs ZERO allocations.
  //
  // Why shared_ptr rather than a bare Rc: the pool needs to know when a buffer
  // is no longer referenced so it can hand it out again, and Rc/RcObject
  // exposes no refcount accessor. The shared_ptr's deleter is the check-in
  // hook. Reuse is safe because every stash write and every stash read is a
  // TRANSFER-stage copy on the one CS stream, ordered by the barriers the
  // consumers already emit — a recycled buffer cannot be rewritten before an
  // earlier reader has consumed it, and a buffer is only recycled once no
  // geometry references it at all (so nothing can still be waiting to read).
  struct IndexGpuStash {
    Rc<DxvkBuffer> buffer;    // pooled; capacity >= size (rounded to a class)
    VkDeviceSize   size = 0;  // valid bytes actually stashed for this draw
  };
  std::shared_ptr<IndexGpuStash> indexDataGpuStash;
  bool indexNeedsGpuStash = false;

  // NV-DXVK [flicker V8: SubmitDraw-ordered geometry capture].
  //
  // Fixes the world-prop flicker (VS 0x29d5f7de class): the BLAS bake runs at
  // END-OF-FRAME position in the CS stream (commitGeometryToRT is emitted by
  // flushGeometryBatch at EndFrame), while the engine's upload of a
  // DEVICE_LOCAL source buffer (UpdateSubresource staging copy, or the
  // initializer upload of a freshly re-batched buffer) has no ordering
  // guarantee against that read — a bake that loses reads mid-upload bytes:
  // zeroed indices collapse the mesh (invisible), garbage positions explode it.
  //
  // The ONLY stream position where "after this frame's upload, before next
  // frame's" is guaranteed is the draw's own position in the CS stream — the
  // engine recorded its upload immediately before its draw. So SubmitDraw
  // EmitCs's a lambda AT THAT POSITION that copies the draw's exact VB/IB
  // windows into pooled RTX-owned buffers (same IndexStashPool as the dynamic
  // index stash above — the struct holds any byte range, not just indices).
  // commitGeometryToRT later REBINDS this geometry's RasterBuffers onto the
  // filled captures, so every bake consumer (cacheIndexDataOnGPU fast copy,
  // interleaveGeometry dispatch, generateTriangleList) reads stable bytes with
  // no per-consumer changes. Same idea as the bone-palette CS-time copy that
  // keeps characters flicker-free, extended to VB/IB ranges.
  //
  // Lifetime mirrors indexDataGpuStash: shared_ptr'd stash handles return to
  // the pool when the last RasterGeometry copy referencing them dies.
  // Thread contract: the game thread creates the set and fills the plan
  // fields; the CS thread fills the stash handles (capture lambda) and reads
  // them (commitGeometryToRT) — strictly ordered by EmitCs FIFO, no locking.
  struct GeometryCapture {
    // Vertex streams captured per distinct (buffer, bind offset, stride)
    // group; streams sharing a D3D11 slot share one capture window of
    // vertexCount * stride bytes starting at the slice offset, so each
    // stream's offsetFromSlice stays valid inside the capture.
    //
    // Sized to the STREAM COUNT (5) on purpose: with groups < streams, a
    // mesh binding every stream from a distinct slot overflowed the group
    // table, the overflow stream was left on the LIVE racy source, and the
    // rebind still stamped sourceIsGpuCapture=true — suppressing both
    // srcPending detection and the recovery latch for a stream that could
    // tear. 5 streams can never need more than 5 groups, so the overflow
    // branch at the SubmitDraw plan site is now structurally unreachable
    // (and defends itself anyway — see captureIncomplete there).
    static constexpr uint32_t kMaxVertexGroups = 5;
    static constexpr uint8_t  kStreamNotCaptured = 0xFF;
    // Stream index -> vertex group: 0=position 1=normal 2=texcoord
    // 3=texcoord1 4=color0.
    uint8_t streamGroup[5] = { kStreamNotCaptured, kStreamNotCaptured,
                               kStreamNotCaptured, kStreamNotCaptured,
                               kStreamNotCaptured };
    bool wantIndex = false;   // plan: capture the index window too
    // Filled on the CS thread by the capture lambda:
    std::shared_ptr<IndexGpuStash> index;
    std::shared_ptr<IndexGpuStash> vertex[kMaxVertexGroups];
    // False if any pool acquire failed — commitGeometryToRT then leaves the
    // geometry on the live source (pre-capture behavior) instead of mixing
    // stable and racy ranges.
    bool valid = false;
  };
  std::shared_ptr<GeometryCapture> gpuCapture;
  // Set by commitGeometryToRT after a successful rebind. processGeometryInfo
  // treats a captured source as never-pending: the capture is stable by
  // construction, and the stash's own in-flight transfer write would otherwise
  // read as srcPending and re-arm the FIX B recovery forever.
  bool sourceIsGpuCapture = false;
  // NV-DXVK [capture stability contract]: source-identity key computed ONCE
  // at the SubmitDraw capture site (fold of every captured range's buffer
  // pointer/offset/len/stride + vs hash + vertexCount — the predictor key)
  // and carried on the geometry so the CS thread can publish stability
  // verdicts under the SAME key the game thread will check. Computing it on
  // the CS side instead would diverge after the capture rebind replaces the
  // buffers. 0 = draw never passed the capture site.
  uint64_t captureIdentityKey = 0;
  // Set by D3D11Rtx::SubmitDraw when this geometry has device-local source
  // ranges the capture machinery COULD stash (whether or not it chose to this
  // frame). The capture-feedback latch requires it: geometry that never passes
  // the SubmitDraw capture site (particle-system quads, external API meshes —
  // RTX-generated in-stream, no engine upload to race) would otherwise latch
  // pendingSrcBake and publish wanted-keys forever, pinning the feedback ring
  // and burning capture bandwidth on draws that can never converge.
  bool captureEligible = false;

  // NV-DXVK [flicker V8 follow-up: capture feedback]. The 2026-08-02 20:28 run
  // proved the SubmitDraw-side warm-window predictor alone cannot cover the
  // flicker's bakes: steady-state re-batches create a new BlasEntry every few
  // frames from the SAME source buffers ([MtnDedup] foundSimilar=0 creates
  // with unchanged hashes), so the predictor key is long past its warm window
  // and those bakes run uncaptured — then foundSimilar=1 reuse serves the one
  // torn bake to the whole prop group indefinitely. The game thread cannot
  // know the CS-side cache decision in advance, so close the loop from the
  // other side: when a bake consumes an uncaptured device-local source,
  // processGeometryInfo latches pendingSrcBake (forcing a re-bake) AND
  // publishes the draw's coarse identity to a lock-free ring the game thread
  // snapshots each frame — the next submits of that draw get captured, and the
  // recovery terminates only when a CAPTURED bake lands.
  //
  // Both sides must derive the identical key from what each has at hand:
  // vsHash + vertexCount + indexCount. Coarse on purpose — over-capture is a
  // wasted copy, under-capture is the bug.
  static inline uint64_t captureFeedbackKey(uint64_t vsHash, uint32_t vertexCount, uint32_t indexCount) {
    uint64_t k = vsHash;
    k = (k * 0x100000001b3ull) ^ vertexCount;
    k = (k * 0x100000001b3ull) ^ indexCount;
    return k != 0ull ? k : 1ull;  // 0 = empty ring slot
  }
  static constexpr uint32_t kCaptureFeedbackSlots = 64;

  template<uint32_t rule>
  const XXH64_hash_t getHashForRule() const {
    return hashes.getHashForRule<rule>();
  }

  const XXH64_hash_t getHashForRule(const HashRule& rule) const {
    return hashes.getHashForRule(rule);
  }

  const XXH64_hash_t getHashForRuleLegacy(const HashRule& rule) const {
    // Note: Only information relating to how the geometry is structured should be included here.
    XXH64_hash_t h = getHashForRule(rule);
    h = XXH64(&indexCount, sizeof(indexCount), h);
    h = XXH64(&vertexCount, sizeof(vertexCount), h);
    h = XXH64(&topology, sizeof(topology), h);
    const uint32_t vertexStride = positionBuffer.stride();
    h = XXH64(&vertexStride, sizeof(vertexStride), h);
    const VkIndexType indexType = indexBuffer.indexType();
    h = XXH64(&indexType, sizeof(indexType), h);
    return h;
  }
  
  uint32_t calculatePrimitiveCount() const;

  bool usesIndices() const {
    return indexBuffer.defined();
  }

  bool isVertexDataInterleaved() const {
    if (normalBuffer.defined() && (!positionBuffer.matches(normalBuffer) || positionBuffer.stride() != normalBuffer.stride()))
      return false;

    if (texcoordBuffer.defined() && (!positionBuffer.matches(texcoordBuffer) || positionBuffer.stride() != texcoordBuffer.stride()))
      return false;

    // NV-DXVK: any presence of texcoord1 (lightmap UV) forces the interleaver
    // path because it always needs decoding (uint → float / 65535).
    if (texcoord1Buffer.defined())
      return false;

    if (color0Buffer.defined() && (!positionBuffer.matches(color0Buffer) || positionBuffer.stride() != color0Buffer.stride()))
      return false;

    return true;
  }

  bool areFormatsGpuFriendly() const {
    assert(positionBuffer.defined());

    if (positionBuffer.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT && positionBuffer.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT)
      return false;

    if (normalBuffer.defined() && (normalBuffer.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT && normalBuffer.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT && normalBuffer.vertexFormat() != VK_FORMAT_R32_UINT))
      return false;

    if (texcoordBuffer.defined() && (texcoordBuffer.vertexFormat() != VK_FORMAT_R32G32_SFLOAT && texcoordBuffer.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT && texcoordBuffer.vertexFormat() != VK_FORMAT_R32G32B32A32_SFLOAT))
      return false;

    if (color0Buffer.defined() && (color0Buffer.vertexFormat() != VK_FORMAT_B8G8R8A8_UNORM))
      return false;

    return true;
  }

  bool isTopologyRaytraceReady() const {
    // Unsupported BVH builder topology
    if (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
      return false;

    // Unsupported BVH builder topology
    if (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)
      return false;

    // No index buffer so must create one (BVH builder does support this mode, our RT code does not)
    if (indexCount == 0)
      return false;

    return true;
  }

  const void printDebugInfo(const char* name = "", uint32_t numTrisToPrint = 0) const {
    Logger::warn(str::format(
      "GeometryData ", name, " address: ", this,
      " vertexCount: ", vertexCount,
      " indexCount: ", indexCount,
      " topology: ", topology,
      " cullMode: ", cullMode,
      " frontFace: ", frontFace,
      " currentVertexHash: 0x", hashes[HashComponents::VertexPosition],
      " drawIndexHash: 0x", hashes[HashComponents::Indices], std::dec));

    // Print Triangles:
    if (numTrisToPrint > 0) {
      uint16_t* indexPtr = (uint16_t*) indexBuffer.mapPtr();
      for (uint32_t i = 0; i < indexCount && i < numTrisToPrint * 3; i += 3) {
        Vector3* position1 = (Vector3*) ((uint8_t*) positionBuffer.mapPtr() + positionBuffer.stride() * indexPtr[i]);
        Vector3* position2 = (Vector3*) ((uint8_t*) positionBuffer.mapPtr() + positionBuffer.stride() * indexPtr[i + 1]);
        Vector3* position3 = (Vector3*) ((uint8_t*) positionBuffer.mapPtr() + positionBuffer.stride() * indexPtr[i + 2]);
        Logger::warn(str::format(
          "[", std::setw(5), indexPtr[i], ", ", std::setw(5), indexPtr[i + 1], ", ", std::setw(5), indexPtr[i + 2], "] : ", std::setprecision(6),
          "(", std::setw(9), position1->x, ", ", std::setw(9), position1->y, ", ", std::setw(9), position1->z, "),   ",
          "(", std::setw(9), position2->x, ", ", std::setw(9), position2->y, ", ", std::setw(9), position2->z, "),   ",
          "(", std::setw(9), position3->x, ", ", std::setw(9), position3->y, ", ", std::setw(9), position3->z, "), "));
      }
    }
  }
};

struct GeometryBufferData {
  uint16_t* indexData;
  size_t indexStride;

  float* positionData;
  size_t positionStride;

  float* texcoordData;
  size_t texcoordStride;

  float* normalData;
  size_t normalStride;

  uint32_t* vertexColorData;
  size_t vertexColorStride;

  GeometryBufferData(const RasterGeometry& geometryData) {
    if (geometryData.indexBuffer.defined()) {
      constexpr size_t indexSize = sizeof(uint16_t);
      indexStride = geometryData.indexBuffer.stride() / indexSize;
      indexData = (uint16_t*) geometryData.indexBuffer.mapPtr();
    } else {
      indexStride = 0;
      indexData = nullptr;
    }

    if (geometryData.positionBuffer.defined()) {
      constexpr size_t positionSubElementSize = sizeof(float);
      positionStride = geometryData.positionBuffer.stride() / positionSubElementSize;
      positionData = (float*) geometryData.positionBuffer.mapPtr((size_t) geometryData.positionBuffer.offsetFromSlice());
    } else {
      positionStride = 0;
      positionData = nullptr;
    }

    texcoordStride = 0;
    texcoordData = nullptr;
    // Only float32 texcoord formats can be safely read as Vector2 on the CPU.
    // R16G16_SFLOAT and other non-float32 formats are converted to R32G32_SFLOAT by the GPU interleaver;
    // treat them as absent here to avoid mis-reading packed half-float data as float2.
    if (geometryData.texcoordBuffer.defined()) {
      const VkFormat texFmt = geometryData.texcoordBuffer.vertexFormat();
      if (texFmt == VK_FORMAT_R32G32_SFLOAT || texFmt == VK_FORMAT_R32G32B32_SFLOAT || texFmt == VK_FORMAT_R32G32B32A32_SFLOAT) {
        constexpr size_t texcoordSubElementSize = sizeof(float);
        texcoordStride = geometryData.texcoordBuffer.stride() / texcoordSubElementSize;
        texcoordData = (float*) geometryData.texcoordBuffer.mapPtr((size_t) geometryData.texcoordBuffer.offsetFromSlice());
      }
    }

    if (geometryData.normalBuffer.defined()) {
      constexpr size_t normalSubElementSize = sizeof(std::uint32_t);
      normalStride = geometryData.normalBuffer.stride() / normalSubElementSize;
      normalData = (float*) geometryData.normalBuffer.mapPtr((size_t) geometryData.normalBuffer.offsetFromSlice());
    } else {
      normalStride = 0;
      normalData = nullptr;
    }

    if (geometryData.color0Buffer.defined()) {
      constexpr size_t colorSubElementSize = sizeof(std::uint32_t);
      vertexColorStride = geometryData.color0Buffer.stride() / colorSubElementSize;
      vertexColorData = (uint32_t*) geometryData.color0Buffer.mapPtr((size_t) geometryData.color0Buffer.offsetFromSlice());
    } else {
      vertexColorStride = 0;
      vertexColorData = nullptr;
    }
  }

  uint16_t getIndex(uint32_t i) const {
    return indexData[i * indexStride];
  }

  uint32_t getIndex32(uint32_t i) const {
    return (uint32_t)indexData[i * indexStride];
  }

  Vector3& getPosition(uint32_t index) const {
    return *(Vector3*) (positionData + index * positionStride);
  }

  Vector2& getTexCoord(uint32_t index) const {
    return *(Vector2*) (texcoordData + index * texcoordStride);
  }

  uint32_t& getVertexColor(uint32_t index) const {
    return vertexColorData[index * vertexColorStride];
  }
};


struct DrawCallTransforms {
  Matrix4 objectToWorld = Matrix4();
  Matrix4 objectToView = Matrix4();
  Matrix4 worldToView = Matrix4();
  Matrix4 viewToProjection = Matrix4();
  Matrix4 textureTransform = Matrix4();
  bool enableClipPlane = false;
  Vector4 clipPlane{ 0.f };
  TexGenMode texgenMode = TexGenMode::None;
  // NV-DXVK: matches RtSurface::TexcoordEncoding. Indicates whether the
  // texcoord buffer holds plain f32 UVs (the default) or a packed-uint
  // encoding the VS bit-decodes (TF2 BSP world VSes — discovered from
  // ISGN componentType=uint on TEXCOORD0). Set in d3d11_rtx.cpp at
  // SubmitDraw time from D3D11CommonShader::GetInputSemanticComponentType.
  RtSurface::TexcoordEncoding texcoordEncoding = RtSurface::TexcoordEncoding::Float;
  const std::vector<Matrix4>* instancesToObject = nullptr;
  // NV-DXVK: Optional lifetime owner for instancesToObject. When set, keeps the
  // backing storage alive as long as this DrawCallState / the RtInstance it feeds
  // exists. Sources whose storage has external lifetime (e.g. USD replacements)
  // can leave this null and only fill instancesToObject.
  std::shared_ptr<const std::vector<Matrix4>> instancesToObjectOwner;

  // NV-DXVK [fanout split]: true when instancesToObject holds one transform per
  // GAME-submitted prop of a bone-instanced fanout batch (d3d11_rtx path 10),
  // as opposed to a USD PointInstancer or the external-GPU-instancing path.
  //
  // Why the distinction matters: for a fanout batch the *batch* is not a stable
  // entity — TF2 adds and drops ~6 props per frame — so one RtInstance per batch
  // can never be deduped reliably, and every membership change destroys the
  // temporal history of all ~54 props in it (measured 2026-08-05: 89 distinct
  // propIds / 86 distinct set-digests across 89 reaps of one object). Only these
  // draws may be split into one RtInstance per placement by
  // InstanceManager::processSceneObjectFanout. A USD PointInstancer is a
  // genuinely stable batch authored as one prim and must keep the PI expansion,
  // and replacement draws must keep exactly one instance per replacement prim
  // (SceneManager::drawReplacements asserts on that identity), so both clear it.
  bool isFanoutBatch = false;

  // NV-DXVK [fanout prev-transform identity] 2026-08-05: index-parallel to
  // instancesToObject — where each placement stood LAST frame, in the same
  // absolute world space, as reported by the engine's own per-instance struct
  // rather than inferred by us.
  //
  // InstanceManager uses it as a second exact-stage SpatialMap probe: an instance
  // was filed last frame under the hash of its transform THEN, which is exactly
  // this matrix now, so a prop that moved still resolves by exact hash instead of
  // falling through to the nearest-neighbour search and its 300-unit-per-frame
  // ceiling, wrong-neighbour risk and filter rejections.
  //
  // Null or short means no history was available (a prop's first frame, or a
  // frame whose camOrigin could not be reproduced); the consumer then keys purely
  // on the current transform, which is the pre-existing behaviour. Only ever
  // populated for isFanoutBatch draws, and only when the length matches
  // instancesToObject — a mismatch would pair a placement with another
  // placement's history, which is worse than having none.
  const std::vector<Matrix4>* prevInstancesToObject = nullptr;
  std::shared_ptr<const std::vector<Matrix4>> prevInstancesToObjectOwner;

  // NV-DXVK: Deterministic camera-pass classifier. Populated by d3d11_rtx
  // from the currently-bound D3D11 viewport at draw submission time. Used by
  // camera_manager to distinguish gameplay draws (viewport matches back
  // buffer / non-square aspect) from shadow cascade / cubemap / RT draws
  // (square viewport, often 1024x1024, 2048x2048, 256x256 for cubes).
  // Zero means "not populated" — camera_manager will fall back to its
  // existing matrix-based heuristics.
  float viewportWidth  = 0.0f;
  float viewportHeight = 0.0f;

  // NV-DXVK: VS hash of the draw that produced this transform. Populated by
  // d3d11_rtx from m_state.gp.shaders.vs->getHash() at draw submission time.
  // Used by camera_manager to allowlist gameplay-world draws (game-native,
  // deterministic). Zero means "not populated / no VS bound".
  XXH64_hash_t vertexShaderHash = 0;

  // NV-DXVK: PS hash of the draw that produced this transform (mirror of
  // vertexShaderHash, captured from m_state.ps.shader->getHash() at submission).
  // Used by the Coverage red-pixel readback to report the exact FS_*.dxbc to
  // decompile for a flagged corruption surface. Zero = not populated.
  XXH64_hash_t pixelShaderHash = 0;

  // NV-DXVK: which code path in d3d11_rtx set worldToView. Small integer
  // tagged at each `transforms.worldToView = ...` site for diagnostic
  // correlation with the latched Main camera. 0 = not set (identity default).
  uint32_t worldToViewPathId = 0;

  // NV-DXVK [SubView]: structural tag for genuine 3D-skybox sub-view
  // draws. Replaces the previous hardcoded VS-hash list `kSubViewVsHashes`
  // in rtx_scene_manager — hash lists silently rot when binaries / mods /
  // shaders change.
  //
  // Derivation: TRUE iff (a) this draw goes through the path-13 camera-
  // relative o2w branch in d3d11_rtx (VS reflects c_cameraOrigin) AND
  // (b) the draw's c_cameraOrigin matches g_engineSkyCamOrigin (the
  // 3D-skybox camera origin latched by the engine.dll trampoline on
  // r8==0x013 sub-view-pass calls) within 4 units. Path 13 alone is
  // INSUFFICIENT — TF2's player / weapon / FX shaders also use path 13
  // with the MAIN camera, and the SRGB / Premult overrides downstream
  // must NOT apply to those.
  //
  // Consumed by SceneManager::createSurfaceMaterial to bypass the
  // encoding pipeline's gammaToLinear + opacity multiply for pre-lit
  // sub-view content (painted 3D-skybox dome / mountains / distant
  // ships). False until the engine hook has captured the sub-view
  // camera at least once (g_engineSkyCamOriginValid != 0) — during
  // menus / loading, no sub-view exists, no override needed.
  bool isSubView = false;

  // NV-DXVK [SubViewSkybox]: refinement of isSubView for the *painted
  // sky-dome* sub-set — sub-view geometry whose world-space AABB
  // diagonal exceeds 5'000'000 units. In TF2's 3D-skybox set, the
  // painted hemispheric dome enclosure has a diagonal of ~25M units,
  // while every other sub-view prop (mountains, ships, terrain) is
  // under 1M. The dome's defining property is that it ENCLOSES the
  // sub-view world from the inside; that's exactly what the size
  // threshold captures structurally.
  //
  // Set by the per-VS classification map in d3d11_rtx — first time
  // a sub-view VS's accumulated worldVert AABB sample reaches the
  // threshold, the map records true; all subsequent draws of that
  // VS read isSubViewSkybox=true at ExtractTransforms time.
  //
  // Consumed by SceneManager::createSurfaceMaterial to set
  // OPAQUE_SURFACE_MATERIAL_FLAG_BAKED_ALBEDO_AS_EMISSIVE. The slang
  // opaque-surface-material then routes the sampled albedo into
  // emissiveRadiance and zeros albedo + baseReflectivity, so the
  // path tracer outputs the painted colour directly — bypassing
  // light × albedo, which is the reason the dome rendered pure
  // black before (it sits 6.75M units from any light source).
  bool isSubViewSkybox = false;

  // NV-DXVK [Stable prop ID]: per-prop identifier that survives transform
  // drift. For content where the engine submits slightly different
  // matrices each frame for the same static prop (TF2 3D-skybox sub-view
  // content drifts ~700u/frame in main-world after our reproject ×
  // scale=1000 magnifies the engine's sub-view-local drift), XXH64 of the
  // matrix bytes misses dedup every frame even though it's the same prop.
  //
  // When non-zero, SpatialMap insert/lookup uses this as the cache key
  // INSTEAD OF XXH64(matrix). The rendered transform stays exact —
  // only the dedup identity is anchored to this ID. So the same prop
  // always dedupes to the same cached instance regardless of inter-frame
  // jitter, and distinct props with different IDs never collide.
  //
  // For sub-view-reprojected draws, populated in SetSkyCategoryFromCb2
  // from the PRE-reproject sub-view-local translation (which is stable
  // per static prop at sub-1u jitter). Default 0 = use matrix-bytes
  // hash, preserving existing behavior for all other content.
  uint64_t stablePropId = 0;

  void sanitize() {
    if (objectToWorld[3][3] == 0.f) objectToWorld[3][3] = 1.f;
    if (objectToView[3][3] == 0.f) objectToView[3][3] = 1.f;
    if (worldToView[3][3] == 0.f) worldToView[3][3] = 1.f;

    // [NaNGuard] Replace any NaN-poisoned matrix with identity. Every
    // ExtractTransforms exit + sky reproject calls sanitize(), so this is
    // the one chokepoint that guarantees downstream consumers
    // (CameraManager::getOrDecomposeProjection, RtCamera::getNearAndFar,
    // MvpToPlanes worldPlanes update) never see NaN. Previously a
    // draw could leak NaN into viewToProjection from a partially-
    // initialised cb read or a degenerate cls 3/4 rebuild, which then
    // tripped MathLib Sqrt's `x >= 0` assert (MathLib.h:1918) deep in
    // MvpToPlanes.
    //
    // IMPORTANT: reject NaN ONLY, NOT Inf. Reverse-Z infinite-far
    // projections (used by TF2 and many modern engines) legitimately have
    // far=Inf as a deliberate design choice; the projection matrix is
    // still well-formed and downstream MathLib Sqrt(Dot33(self)) of any
    // finite-or-Inf vector returns >=0 (never asserts). The previous
    // version used !isfinite which over-rejected valid infinite-far
    // matrices and caused RtCamera::update to skip valid frames.
    // Throttled log carries the field name so the leak site can still
    // be identified in the next session.
    auto hasNaN4x4 = [](const Matrix4& m) -> bool {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          if (std::isnan(m[r][c])) return true;
      return false;
    };
    auto maybeNuke = [&](Matrix4& m, const char* fieldName) {
      if (!hasNaN4x4(m)) return;
      static thread_local uint64_t sNaNGuardN[4]        = {0,0,0,0};
      static thread_local std::chrono::steady_clock::time_point sLastLogT[4] = {
        std::chrono::steady_clock::time_point::min(),
        std::chrono::steady_clock::time_point::min(),
        std::chrono::steady_clock::time_point::min(),
        std::chrono::steady_clock::time_point::min(),
      };
      // 0=o2w, 1=o2v, 2=w2v, 3=v2p. Encoded by first char of fieldName for
      // a 4-bucket slot — keeps each field's throttle independent so we
      // can see all four if they're all leaking.
      int idx = 0;
      switch (fieldName[0]) {
        case 'o': idx = (fieldName[6] == 'V' ? 1 : 0); break;  // objectToView vs objectToWorld
        case 'w': idx = 2; break;                              // worldToView
        case 'v': idx = 3; break;                              // viewToProjection
      }
      ++sNaNGuardN[idx];
      const auto now = std::chrono::steady_clock::now();
      const bool firstFew = sNaNGuardN[idx] <= 10;
      const bool dueByClock =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          now - sLastLogT[idx]).count() >= 500;
      if (firstFew || dueByClock) {
        sLastLogT[idx] = now;
        dxvk::Logger::warn(dxvk::str::format(
          "[DrawCallTransforms.sanitize.NaN] field=", fieldName,
          " n=", sNaNGuardN[idx],
          " worldToViewPathId=", worldToViewPathId,
          " m0=(", m[0][0], ",", m[0][1], ",", m[0][2], ",", m[0][3], ")",
          " m1=(", m[1][0], ",", m[1][1], ",", m[1][2], ",", m[1][3], ")",
          " m2=(", m[2][0], ",", m[2][1], ",", m[2][2], ",", m[2][3], ")",
          " m3=(", m[3][0], ",", m[3][1], ",", m[3][2], ",", m[3][3], ")"));
      }
      m = Matrix4();  // identity
    };
    maybeNuke(objectToWorld,    "objectToWorld");
    maybeNuke(objectToView,     "objectToView");
    maybeNuke(worldToView,      "worldToView");
    maybeNuke(viewToProjection, "viewToProjection");
  }

  Matrix4 calcFirstInstanceObjectToWorld() const {
    if (instancesToObject && !instancesToObject->empty()) {
      return objectToWorld * (*instancesToObject)[0];
    } else {
      return objectToWorld;
    }
  }
};

// API-agnostic fog mode — D3D11 has no fog API; this is used by the
// shared RTX backend to represent legacy fog state passed via constant buffers.
enum class FogMode : uint32_t {
  None   = 0,
  Exp    = 1,
  Exp2   = 2,
  Linear = 3,
};
std::ostream& operator << (std::ostream& os, FogMode mode);

struct FogState {
  FogMode mode = FogMode::None;
  Vector3 color = Vector3();
  float scale = 0.f;
  float end = 0.f;
  float density = 0.f;

  XXH64_hash_t getHash() const {
    return XXH3_64bits(this, sizeof(FogState));
  }
};

enum class InstanceCategories : uint32_t {
  WorldUI,
  WorldMatte,
  Sky,
  Ignore,
  IgnoreLights,
  IgnoreAntiCulling,
  IgnoreMotionBlur,
  IgnoreOpacityMicromap,
  IgnoreAlphaChannel,
  Hidden,
  Particle,
  Beam,
  DecalStatic,
  DecalDynamic,
  DecalSingleOffset,
  DecalNoOffset,
  AlphaBlendToCutout,
  Terrain,
  AnimatedWater,
  ThirdPersonPlayerModel,
  ThirdPersonPlayerBody,
  IgnoreBakedLighting,
  IgnoreTransparencyLayer,
  ParticleEmitter,
  SmoothNormals,

  Count,
};

using CategoryFlags = Flags<InstanceCategories>;

#define DECAL_CATEGORY_FLAGS InstanceCategories::DecalStatic, InstanceCategories::DecalDynamic, InstanceCategories::DecalSingleOffset, InstanceCategories::DecalNoOffset

// NV-DXVK [Phase2b sharded instance processing]: per-draw sidecar produced by
// SceneManager::processDeferredDrawBatch on the game thread at EndFrame and
// consumed by RtxContext::commitGeometryToRT on dxvk-cs. It travels through the
// Phase C EmitCs lambda ALONGSIDE the DrawCallState (not inside it, so the
// pBlas->input copy in onSceneObjectUpdated never drags it along). See
// PHASE2B_IMPLEMENTATION_SPEC.md for the full architecture.
class RtInstance;
struct BlasEntry;

// A spatial-map write recorded by a worker during the sharded instance phase and
// applied by the ordered tail. targetBlas is captured EXPLICITLY at record time:
// the migration path erases from the instance's OLD entry, so resolving through
// m_linkedBlas at apply time would target the wrong map (the relink has already
// happened by then). The op reads/writes instance->m_spatialCacheHash at APPLY
// time, so multi-op chains on one instance resolve exactly like the inline code.
struct DeferredSpatialOp {
  // kDecalOrder rides the same per-item op list: it is not a map write, but it
  // is the same shape of problem — an order-sensitive assignment that must
  // happen in arena order in the tail (decal sort order approximates draw order
  // on the GPU). Only `instance` is meaningful for it.
  enum class Kind : uint8_t { kMove, kInsert, kDecalOrder };
  Kind kind = Kind::kMove;
  RtInstance* instance = nullptr;
  BlasEntry* targetBlas = nullptr;
  Vector3 centroid = Vector3(0.f, 0.f, 0.f);
  Matrix4 transform;
  uint64_t stablePropId = 0;
  XXH64_hash_t precomputedMatrixHash = 0;
};

struct ShardedDrawInfo {
  // TWO live routes, not the four the spec's first draft listed. A `kDropped`
  // value existed for ignored materials and was removed as dead: an ignored draw
  // is routed kSharded with NO shard membership (no cache touch, no stamps, no
  // instance work), which lets the CS slim path reach processDrawCallState's own
  // pre-cache ignored return instead of re-running the whole legacy prologue —
  // the pre-pass has already mutated the fog/replacement/material caches for it.
  // The buffer-cache overflow case is likewise not a per-draw route: it is a
  // whole-frame admission gate that sends EVERY draw legacy (see
  // shardingAdmissible in processDeferredDrawBatch).
  enum class Route : uint8_t {
    kNone = 0,     // not preprocessed: CS runs the full legacy path (option off)
    kLegacyCS,     // camera classified at flush only; scene+instance work stays on CS
                   // (replacements, terrain, sky, and anything else the pre-pass
                   // routes away from the shard path)
    kSharded,      // geom decision + instance work precomputed at flush; CS consumes
                   // the results and records the GPU work
  };
  Route route = Route::kNone;
  bool cameraDone = false;           // dcs.cameraType classified in the pre-pass; CS must not re-run
  bool blasFirstDrawOfFrame = false; // frameLastTouched != fid BEFORE the pre-pass stamped it
  bool needsTailContinuation = false;// set on a dedup miss by a worker; consumed by the ordered tail
  int8_t geomResult = -1;            // SceneManager::ObjectCacheState decided at flush; <0 = none
  BlasEntry* pBlas = nullptr;        // stable node pointer (see rtx_draw_call_cache.h:39) until GC

  // NV-DXVK [Phase2b]: per-INSTANCE deferred work recorded by updateInstance on
  // a worker and replayed by the CS record step. One entry per updateInstance
  // call (fanout: one per placement), so the event args keep per-placement
  // fidelity — a per-draw flag would smear one placement's hasTransformChanged
  // over the whole batch and feed OMM wrong rebuild signals.
  struct PendingInstanceOps {
    RtInstance* instance = nullptr;
    bool bindBuffers = false;   // surface buffer rebind (post-updateBufferCache)
    bool billboard = false;     // createBillboards/createBeams eligibility passed
    bool omm = false;           // the non-skippable OMM callback was due
    bool rayPortal = false;     // processRayPortalData was due (RayPortal material)
    bool evHasTransformChanged = false;
    bool evHasPreviousPositions = false;
    bool evIsFirstUpdateThisFrame = false;
  };
  std::vector<PendingInstanceOps> pendingOps;
  // Instances produced/updated for this draw (fanout: all placements; single: one entry).
  std::vector<RtInstance*> instances;
  // Fanout placements whose find missed on the worker (deferred, cold at
  // addedPct=0). The ordered tail re-runs exactly these placement indices
  // sequentially. Non-empty also suppresses the pushInstanceRecords verify for
  // this draw (the produced list is deliberately incomplete until the tail runs)
  // — the record REFRESH already self-suppresses on a short out_instances.
  std::vector<uint32_t> deferredPlacements;
  // Spatial-map writes recorded on workers, applied by the ordered tail in arena
  // order. Never carried to CS (drained before Phase C).
  std::vector<DeferredSpatialOp> spatialOps;
  // Render material after replacement/override resolution AND after the instance
  // manager's mutations — CS-side consumers (particles, OMM callback, effect
  // lights) read this instead of re-deriving it.
  std::shared_ptr<MaterialData> renderMaterial;

  ShardedDrawInfo() = default;
  ShardedDrawInfo(ShardedDrawInfo&&) = default;
  ShardedDrawInfo& operator=(ShardedDrawInfo&&) = default;
  ShardedDrawInfo(const ShardedDrawInfo&) = delete;
  ShardedDrawInfo& operator=(const ShardedDrawInfo&) = delete;
};

struct DrawCallState {
  DrawCallState() = default;
  DrawCallState(const DrawCallState& _input) = default;
  DrawCallState& operator=(const DrawCallState& drawCallState) = default;
  // NV-DXVK [BatchSubmitDraw]: enable MOVE so the commit hand-off transfers the
  // Rc<> buffer set + bone matrices by pointer-steal instead of deep-copying them
  // per draw. The two std::move(dcs) sites (d3d11_rtx.cpp: the batch-arena collect
  // and the EmitCs emit) are both terminal — dcs is unused afterward — so this is
  // hazard-free. Declaring the copy ops above (=default) had implicitly SUPPRESSED
  // the move ctor/assign, so std::move(dcs) was silently binding to the copy (the
  // "no 16KB bone copy" the emit site's comment intended never actually happened).
  DrawCallState(DrawCallState&&) = default;
  DrawCallState& operator=(DrawCallState&&) = default;

  // Note: This uses the original material for the hash, not the replaced material
  const XXH64_hash_t getHash(const HashRule& rule) const {
    return geometryData.getHashForRule(rule) ^ materialData.getHash();
  }

  [[deprecated("(REMIX-656): Remove this once we can transition content to new hash")]]
  const XXH64_hash_t getHashLegacy(const HashRule& rule) const {
    return geometryData.getHashForRuleLegacy(rule) ^ materialData.getHash();
  }

  const RasterGeometry& getGeometryData() const {
    return geometryData;
  }

  const LegacyMaterialData& getMaterialData() const {
    return materialData;
  }

  const DrawCallTransforms& getTransformData() const {
    return transformData;
  }

  const SkinningData& getSkinningState() const {
    return skinningData;
  }

  // NV-DXVK [RigidFinal]: marker (set to 1 by D3D11Rtx::SubmitDraw) tagging the instanced
  // GPU-skinned hull draw so the [RigidFinal] probe can distinguish it from the non-instanced
  // widow parts. Repurposed from the (removed) GPU rigid-bake path.
  uint32_t getRigidBakeBoneIndex() const {
    return rigidBakeBoneIndex;
  }

  const FogState& getFogState() const {
    return fogState;
  }

  const CategoryFlags getCategoryFlags() const {
    return categories;
  }

  bool finalizePendingFutures(const RtCamera* pLastCamera);

  bool hasTextureCoordinates() const {
    return getGeometryData().texcoordBuffer.defined() || getTransformData().texgenMode != TexGenMode::None;
  }
  bool isEye() const;

  // Whether the smooth-normals compute pass should run for this draw.
  // Why: TF2 surfaces every draw call without authored normals (verified in
  // [SpawnGeomDiag.Interleaver] logs — 256/256 had hasNormals=0). With no
  // normals the surface interaction shader picks up garbage and shading
  // collapses to flat-faceted. Generating area-weighted smooth normals from
  // the triangle mesh is the only correct fix when the source has none.
  // The exclusion list covers categories where a per-vertex surface normal
  // is meaningless (sky, UI overlays, decals, particles, beams) or the
  // draw is filtered out (Hidden/Ignore).
  bool shouldGenerateSmoothNormals() const {
    if (testCategoryFlags(InstanceCategories::SmoothNormals)) {
      return true; // explicit Remix-API override
    }
    if (geometryData.normalBuffer.defined()) {
      return false; // artist authored normals; trust them
    }
    return !testCategoryFlags(
      InstanceCategories::Sky,
      InstanceCategories::Hidden,
      InstanceCategories::Ignore,
      InstanceCategories::WorldUI,
      InstanceCategories::WorldMatte,
      InstanceCategories::Particle,
      InstanceCategories::ParticleEmitter,
      InstanceCategories::Beam,
      InstanceCategories::DecalStatic,
      InstanceCategories::DecalDynamic,
      InstanceCategories::DecalSingleOffset,
      InstanceCategories::DecalNoOffset);
  }

  bool stencilEnabled = false;

  // Camera type associated with the draw call
  CameraType::Enum cameraType = CameraType::Unknown;

  // Uses programmable VS/PS
  bool usesVertexShader = false, usesPixelShader = false;

  // Shader model version — valid only when usesVertex/PixelShader is set.
  // D3D11 shaders are SM 4.0+; legacy games via Remix API may report SM 1-3.
  ShaderProgramInfo vertexShaderInfo;
  ShaderProgramInfo pixelShaderInfo;

  float minZ = 0.0f;
  float maxZ = 1.0f;

  bool zWriteEnable = false;
  bool zEnable = false;

  uint32_t drawCallID = 0;

  bool isDrawingToRaytracedRenderTarget = false;
  bool isUsingRaytracedRenderTarget = false;

  // Set when the Sky category was assigned by skyAutoDetect heuristic
  // (as opposed to explicit methods like skyBoxTextures/skyBoxGeometries/skyMinZThreshold).
  // Used by tryHandleSky to optionally bypass cubemap rasterization for autoDetected sky,
  // since it may be world geometry that should go through reprojection instead.
  bool skyAutoDetected = false;

  // NV-DXVK [StudioModelHook]: BY-MODEL Widow tag. Set in D3D11Rtx::SubmitDraw
  // when this draw came from a studiorender model whose engine name path
  // contains "widow" (via the studiorender draw-site hook). The flag is copied
  // into the cached BlasEntry.input, so deep probes that hold a DrawCallState
  // (scene_manager processDrawCallState) OR reach one via the instance
  // (instance_manager: pInst->getBlas()->input.isWidowModel) can re-gate on
  // the actual engine model instead of the shared VS hash 0x292b (~70
  // draws/frame) or non-1:1 texture hashes. Requires rtx.tf2HideWidow,
  // rtx.tf2IsolateWidow, or rtx.tf2DetectWidow to be enabled (otherwise the
  // per-draw name resolution is skipped and this stays false).
  bool isWidowModel = false;

  // NV-DXVK [StudioModelHook]: the studiorender model/material name path
  // (e.g. "models\\vehicles_r2\\aircraft\\widow\\veh_air_widow") captured at
  // SubmitDraw, NUL-terminated, truncated to 63 chars. Empty for non-
  // studiorender draws. Value-copied (not a pointer) so it survives the
  // deferred [ShipBox] readback — lets that probe report the engine model
  // name of the surface under the pick box, not just its Remix hash. Only
  // filled when a tf2*Widow / tf2DumpStudioNames option is active.
  char studioModelName[64] = {};

  // NV-DXVK [MatBind identity]: the engine's matsys IMaterial* bound right
  // before this draw (thread-local from the [MatBind] slot-0x80 hook,
  // captured in D3D11Rtx::SubmitDraw on the same thread that issued the
  // draw). This is ENGINE-TRUTH material identity — stable for the
  // material's lifetime, independent of every Remix-derived hash. Used by
  // DrawCallCache::get as the primary same-object-class test for draws
  // whose vertex data is rewritten per frame (re-batched world-space
  // geometry), where FullGeometryHash can never match and derived material
  // hashes are one padding bug away from lying. 0 = no matsys bind seen on
  // this thread (non-matsys draw) — callers must then fall back to hashes.
  uint64_t engineMaterialPtr = 0;

  // NV-DXVK [ResidentScene]: the frame thread's record key for this draw, and
  // the dirty fold it was judged against. See RESIDENT_SCENE_PLAN.md.
  //
  // WHY THE KEY TRAVELS ON THE DRAW. The gate's two halves live on different
  // threads by necessity: the frame thread owns the D3D11 state the key is made
  // of and must decide before any of the work it exists to skip, while only the
  // CS side may touch an RtInstance. The key is the entire join between them --
  // eight bytes through the channel the draw already uses, so no map is ever
  // reached by two threads and there is nothing to lock.
  //
  // 0 means "this draw has no usable identity", which is the do-not-make-
  // resident answer rather than an error: 0 is also the no-record sentinel on
  // RtInstance::m_residentKey and ResidentScene::build rejects it.
  uint64_t residentKey = 0ull;
  // The occurrence ordinal this draw got WITHIN the narrowed identity, so a
  // residency failure can be attributed. 0 means material and placement gave
  // this draw its own key; >0 means it is still separated only by submission
  // order, which is the population that shifts under culling. Reported by
  // [RsFailSize] / [RsFailMember] -- without it those lines cannot say whether
  // a failure came from the residual or from the separated majority.
  uint32_t residentOrdinal = 0u;
  // NV-DXVK [RenderObject] slice 1: THE IDENTITY THE ORDINAL IS AN ORDINAL
  // WITHIN -- residentGateJudge's `narrowed`, i.e. the IA identity folded with
  // the material fold, BEFORE the occurrence is folded on top to make
  // residentKey.
  //
  // Carried separately because residentKey cannot be decomposed back into its
  // two parts, and the object resolver needs them apart rather than together:
  // the identity is what names the OBJECT and is camera- and animation-
  // invariant by construction, while the ordinal names WHICH COPY and is the
  // known-weak half (ARCHITECTURE_OVERHAUL sec 1.2 rung 4 -- engine culling
  // removing one copy of a multi-copy identity renumbers the survivors).
  // Feeding the composite in as if it were the identity would make those two
  // failure modes indistinguishable, which is the one thing slice 1's gate has
  // to be able to tell apart.
  //
  // 0 for the same reason residentKey is 0: no usable identity.
  uint64_t residentIdentity = 0ull;
  uint64_t residentGenHash = 0ull;
  // The engine buffers this draw was made of, as raw addresses. The record keeps
  // them so that ~D3D11Buffer freeing one can retire it -- a resident instance
  // is exempt from lifetime expiry, so without a death signal a destroyed object
  // would stay in the ray-traced scene indefinitely. Addresses rather than
  // references on purpose: holding a reference would keep alive the very object
  // whose destruction is the signal.
  uint64_t residentSrcVertexBuffer = 0ull;
  uint64_t residentSrcIndexBuffer = 0ull;
  // The frame thread's verdict, carried for SCORING ONLY. While
  // rtx.residentScene.verify is on the draw runs the full path regardless and
  // this says "the gate would have skipped me" -- which is what makes the FAIL
  // count in [ResidentScene] a measurement of the prediction rather than of the
  // consequence of having acted on it.
  bool residentPredictHit = false;

  void setupCategoriesForTexture();
  void setupCategoriesForGeometry();
  void setupCategoriesForHeuristics(uint32_t prevFrameSeenCamerasCount,
                                    std::vector<Vector3>& seenCameraPositions);

  template<typename... InstanceCategories>
  bool testCategoryFlags(InstanceCategories... cat) const { return categories.any(cat...); }

  void printDebugInfo(const char* name = "") const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "DrawCallState ", name, "\n",
      "  address: ", this, "\n",
      "  drawCallID: ", drawCallID, "\n",
      "  cameraType: ", static_cast<int>(cameraType), "\n",
      "  usesVertexShader: ", usesVertexShader, "\n",
      "  usesPixelShader: ", usesPixelShader, "\n",
      "  stencilEnabled: ", stencilEnabled, "\n",
      "  zWriteEnable: ", zWriteEnable, "\n",
      "  zEnable: ", zEnable, "\n",
      "  minZ: ", minZ, "\n",
      "  maxZ: ", maxZ, "\n",
      "  isDrawingToRaytracedRenderTarget: ", isDrawingToRaytracedRenderTarget, "\n",
      "  isUsingRaytracedRenderTarget: ", isUsingRaytracedRenderTarget, "\n",
      "  categoryFlags: ", categories.raw(), "\n",
      "  hasTextureCoordinates: ", hasTextureCoordinates(), "\n",
      "  materialHash: 0x", std::hex, materialData.getHash(), std::dec));
    
    // Print geometry info
    Logger::warn("=== Geometry Info ===");
    Logger::warn(str::format(
      "  vertexCount: ", geometryData.vertexCount, "\n",
      "  indexCount: ", geometryData.indexCount, "\n",
      "  numBonesPerVertex: ", geometryData.numBonesPerVertex, "\n",
      "  topology: ", static_cast<int>(geometryData.topology), "\n",
      "  cullMode: ", static_cast<int>(geometryData.cullMode), "\n",
      "  frontFace: ", static_cast<int>(geometryData.frontFace), "\n",
      "  forceCullBit: ", geometryData.forceCullBit, "\n",
      "  externalMaterial: ", (geometryData.externalMaterial != nullptr ? "valid" : "null")));
    
    // Print transform info
    Logger::warn("=== Transform Info ===");
    Logger::warn(str::format(
      "  enableClipPlane: ", transformData.enableClipPlane, "\n",
      "  clipPlane: (", transformData.clipPlane.x, ", ", transformData.clipPlane.y, ", ", transformData.clipPlane.z, ", ", transformData.clipPlane.w, ")"));
    
    // Print skinning info
    Logger::warn("=== Skinning Info ===");
    Logger::warn(str::format(
      "  numBones: ", skinningData.numBones, "\n",
      "  numBonesPerVertex: ", skinningData.numBonesPerVertex, "\n",
      "  minBoneIndex: ", skinningData.minBoneIndex, "\n",
      "  boneHash: 0x", std::hex, skinningData.boneHash, std::dec));
    
    // Print fog info
    Logger::warn("=== Fog Info ===");
    Logger::warn(str::format(
      "  fogMode: ", fogState.mode, "\n",
      "  fogColor: (", fogState.color.x, ", ", fogState.color.y, ", ", fogState.color.z, ")\n",
      "  fogScale: ", fogState.scale, "\n",
      "  fogEnd: ", fogState.end, "\n",
      "  fogDensity: ", fogState.density));
    
    // Print material info
    Logger::warn("=== Material Info ===");
    materialData.printDebugInfo("(from DrawCallState)");
#endif
  }

private:
  friend class RtxContext;
  friend class SceneManager;
  friend class D3D11Rtx;
  friend class TerrainBaker;
  friend struct RemixAPIPrivateAccessor;
  friend class RtxParticleSystemManager;

  bool finalizeGeometryHashes();
  void finalizeGeometryBoundingBox();
  void finalizeSkinningData(const RtCamera* pLastCamera);
  // NV-DXVK [MatDefer]: resolve the deferred material-compute future into
  // materialData. No-op when the future is invalid (synchronous fill).
  void finalizeMaterialData();

  // NOTE: 'setCategory' can only add a category, it will not unset a bit
  void setCategory(InstanceCategories category, bool set);
  void removeCategory(InstanceCategories category);

  RasterGeometry geometryData;

  // Note: This represents the original material from the game frontend, which will always be a LegacyMaterialData
  // whereas the replacement material data used for rendering will be a full MaterialData.
  LegacyMaterialData materialData;

  DrawCallTransforms transformData;

  // Note: Set these pointers to nullptr when not used
  SkinningData skinningData;
  Future<SkinningData> futureSkinningData;

  // NV-DXVK [MatDefer]: async material-compute future. When rtx.deferMaterialCompute
  // is set, D3D11Rtx::SubmitDraw runs the cheap material RESOLVE (texture/sampler
  // binding + sourceIsUnlitUI + blendMode — the only material fields the game thread
  // reads before EmitCs) synchronously into materialData, then schedules the
  // expensive COMPUTE (PS-cbuffer value reads, emissive/fog/alpha flags, tail) on a
  // geometry worker, which returns a fully-populated LegacyMaterialData here. It is
  // finalized on the consumer thread in finalizeMaterialData() (before
  // setupCategoriesForGeometry, which reads the color-texture hash). An invalid
  // future means materialData was filled synchronously (deferral off, or the worker
  // queue was full and the compute ran inline). Held via shared_ptr because a
  // LegacyMaterialData is far larger than the worker Task's inline result storage.
  Future<std::shared_ptr<LegacyMaterialData>> futureMaterialData;

  // NV-DXVK [RigidFinal]: 1 = this is the instanced GPU-skinned hull draw (marker for the
  // [RigidFinal] probe). Set in D3D11Rtx::SubmitDraw.
  uint32_t rigidBakeBoneIndex = 0;

  FogState fogState;

  CategoryFlags categories = 0;

  // NV-DXVK [SkyProbe.cubeRender] Snapshot taken at the d3d11 frontend when
  // a sky-classified draw is detected. The frontend reads c_cameraRelativeToClip
  // (slot+offset via FindCBField) plus the full cb2 byte contents at the moment
  // the draw is captured. RtxContext::rasterizeToSkyProbe then dispatches the
  // same TF2 sky shader 6 times — once per cube face — overriding cb2's
  // matrix slot with cube-face View×Projection matrices to capture TF2's
  // authored sky into a real cubemap. Without this snapshot the cubemap
  // stays empty and the path tracer falls back to Hillaire-only IBL.
  struct SkyProbeCubeCapture {
    bool      valid          = false;
    uint32_t  vsCb2DxvkSlot  = 0;     // dxvk-side bind slot for VS cb2
    uint32_t  cb2MatrixOffset = 0;    // byte offset of c_cameraRelativeToClip
    uint32_t  cb2OriginOffset = 0;    // byte offset of c_cameraOrigin
    uint32_t  cb2ByteSize    = 0;     // total cb2 size for memcpy snapshot
    // Snapshot is allocated on the frontend at capture time; 1KB cap is
    // generous for CBufCommonPerCamera which is observed in the ~600B range.
    static constexpr uint32_t kSnapshotMax = 1024u;
    uint8_t   cb2Snapshot[kSnapshotMax] = {};
    // The four matrix rows captured at off=cb2MatrixOffset. Cached as a
    // typed Matrix4 so cube-face-derive code can read it without a
    // memcpy + reinterpret dance.
    Matrix4   capturedViewProj {};
    // Camera origin in TF2 world space — used to recenter cube faces.
    Vector3   capturedCameraOrigin { 0.0f, 0.0f, 0.0f };
  } skyProbeCubeCapture {};
};

 // A BLAS and its data buffer that can be pooled and used for various geometries
struct PooledBlas : public RcObject {
  Rc<DxvkAccelStructure> accelStructure;
  uint64_t accelerationStructureReference = 0;

  // Frame when this BLAS was last used in a TLAS
  uint32_t frameLastTouched = kInvalidFrameIndex;

  // Hash of a bound opacity micromap
  // Note: only used for tracking of OMMs for static BLASes
  XXH64_hash_t opacityMicromapSourceHash = kEmptyHash;

  // Keep a copy of the build info so we can validate BLAS update compatibility
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
  std::vector<uint32_t> primitiveCounts {};

  explicit PooledBlas();
  ~PooledBlas();
};

// Information about a geometry, such as vertex buffers, and possibly a static BLAS for that geometry
struct BlasEntry {
  // input contains legacy or replacements (the data can be on CPU or GPU)
  //  - Data on CPU is guaranteed to be alive during draw call's submission.
  //  - Data can be made alive on CPU for longer with an explicit ref hold on it
  //  - For shader based games the data may contain various unsupported formats a game might deliver the data in. 
  //    That is converted and optimized in RtxGeometryUtils::interleaveGeometry. 
  //    Legacy pipeline games always use supported buffer formats/encodings etc...
  DrawCallState input; 
  // modifiedGeometryData contains the same geometry as "input" but it (may) have been transformed (i.e.interleaved vertex data, 
  // converted to optimal vertex formats [we prefer float32], will always be a triangle list and could be skinned)
  // - Data is on GPU 
  // - Data is not directly mappable on CPU
  RaytraceGeometry modifiedGeometryData;

  // Frame when this geometry was seen for the first time
  uint32_t frameCreated = kInvalidFrameIndex;

  // Frame when this geometry was last used in a TLAS
  uint32_t frameLastTouched = kInvalidFrameIndex;

  // Frame when the vertex data of this geometry was last updated, used to detect static geometries
  uint32_t frameLastUpdated = kInvalidFrameIndex;

  // NV-DXVK [ReapJoin]: HOW MANY draws resolved to this entry this frame.
  //
  // frameLastTouched answers "did ANY draw for this geometry arrive", which is
  // useless for judging a single instance once a mesh has more than one copy on
  // screen: one sibling's draw stamps the entry and every sibling then looks
  // like it drew. Measured 2026-08-05 (693 frames, 5661 reaps): of 4996 reaps
  // reporting drew=1, ZERO had linked==1 — drew=1 was exactly equivalent to
  // "this mesh has siblings", so the field carried no information about the
  // reaped instance at all. A count is what distinguishes "this geometry lost a
  // copy" from "the draw arrived and dedup put it somewhere else".
  //
  // Reset lazily on the first draw of a new frame rather than in an onFrameEnd
  // sweep — entries are created and destroyed constantly and a sweep would have
  // to walk the whole cache to keep a counter honest.
  uint32_t drawCountFrame = kInvalidFrameIndex;
  uint32_t drawCount = 0;

  // Called once per arriving draw, from the single site that stamps
  // frameLastTouched (SceneManager::processDrawCallState). Plain (non-atomic)
  // like every other field here: the draw-call cache is single-threaded.
  void noteDraw(const uint32_t frameId) {
    if (drawCountFrame != frameId) {
      drawCountFrame = frameId;
      drawCount = 0;
    }
    ++drawCount;
  }

  // 0 when this entry saw no draw in frameId, so callers never read a count
  // left over from an earlier frame.
  uint32_t getDrawCount(const uint32_t frameId) const {
    return (drawCountFrame == frameId) ? drawCount : 0u;
  }

  // NV-DXVK [MvRaw]: the pairing decision DrawCallCache::get made when it last
  // handed this entry to a draw. Recorded here rather than joined by blasPtr
  // because BlasEntries are destroyed and reallocated constantly, so one
  // address refers to many different objects over a run — an address join
  // silently mixes them. These fields travel WITH the entry, so the motion
  // probe in rtx_instance_manager reads the decision that actually produced
  // the instance it is measuring.
  //
  // lastPairFrame is the guard: read these only when it equals the current
  // frame, otherwise they describe an older pairing.
  uint32_t lastPairFrame       = kInvalidFrameIndex;
  // frameLastTouched of the winning candidate AT THE TIME it won, i.e. how
  // stale the entry the draw was paired with actually was. 612ff00d made
  // recency the PRIMARY ranking term, so this is the value that decided it.
  uint32_t lastPairPrevTouched = kInvalidFrameIndex;
  // Winning candidate's score. <= 0 means the pre-612ff00d seed
  // (numeric_limits<float>::min()) would have REJECTED this pairing and
  // allocated a fresh entry instead — so this flags exactly the draws whose
  // behaviour changed in the suspect window.
  float    lastPairScore       = 0.f;
  // How the entry was obtained: 0 = kNew (fresh entry, empty SpatialMap),
  // 1 = kExisted via positive score, 2 = kExisted only because the seed was
  // fixed (the "rescued" case).
  uint32_t lastPairKind        = 0u;

  using InstanceMap = SpatialMap<RtInstance>;

  Rc<PooledBlas> dynamicBlas = nullptr;

  std::vector<VkAccelerationStructureGeometryKHR> buildGeometries;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;

  // NV-DXVK [BlasSizeCache]: memoized result of
  // vkGetAccelerationStructureBuildSizesKHR for this entry. The driver call
  // ran for EVERY unique dynamic BLAS EVERY frame in
  // AccelManager::mergeInstancesIntoBlas (~137/frame, [Perf.Merge]
  // dynBlas ~38ms/frame in heavy scenes) even though ~85% of entries are
  // reused unchanged. Per the Vulkan spec the size query depends ONLY on
  // formats/counts/flags — all address/pointer members are ignored — so the
  // result is a pure function of the key below and can be reused until the
  // geometry shape changes. Key = hash of (primitiveCount, geometry
  // triangle formats/strides/maxVertex, geometry flags, build flags);
  // 0 = not yet cached.
  uint64_t blasSizeCacheKey = 0;
  VkAccelerationStructureBuildSizesInfoKHR blasSizeCacheInfo {};

  // NV-DXVK [MatBind identity]: key under which this entry was registered in
  // DrawCallCache::m_engineClassIndex at allocation (0 = never registered —
  // the allocating draw carried no engine material). Stored so removal at GC
  // uses the REGISTRATION key, not a recomputation from `input` — `input` is
  // overwritten on every pairing and an exactMatch pairing does not gate on
  // engineMaterialPtr, so a recomputed key could differ and orphan the index
  // entry (dangling pointer).
  XXH64_hash_t engineClassKey = 0;

  BlasEntry() = default;

  BlasEntry(const DrawCallState& input_);

  void cacheMaterial(const LegacyMaterialData& newMaterial) {
    if (input.getMaterialData().getHash() != newMaterial.getHash()) {
      m_materials.emplace(newMaterial.getHash(), newMaterial);
    }
  }

  const LegacyMaterialData& getMaterialData(XXH64_hash_t matHash) const {
    if (input.getMaterialData().getHash() == matHash) {
      return input.getMaterialData();
    }
    auto iter = m_materials.find(matHash);
    if (iter != m_materials.end()) {
      return iter->second;
    }
    assert(false); // tried to get a material that the BlasEntry doesn't know about.
    return input.getMaterialData();
  }

  void clearMaterialCache() {
    m_materials.clear();
  }

  void linkInstance(RtInstance* instance) {
    m_linkedInstances.push_back(instance);
  }

  void unlinkInstance(RtInstance* instance);

  const std::vector<RtInstance*>& getLinkedInstances() const { return m_linkedInstances; }
  InstanceMap& getSpatialMap() { return m_spatialMap; }
  const InstanceMap& getSpatialMap() const { return m_spatialMap; }

  void rebuildSpatialMap();

  void printDebugInfo(const char* name = "") const {
#ifdef REMIX_DEVELOPMENT
    Logger::warn(str::format(
      "BlasEntry ", name, "\n",
      "  address: ", this, "\n",
      "  frameCreated: ", frameCreated, "\n",
      "  frameLastTouched: ", frameLastTouched, "\n",
      "  frameLastUpdated: ", frameLastUpdated, "\n",
      "  vertexCount: ", modifiedGeometryData.vertexCount, "\n",
      "  indexCount: ", modifiedGeometryData.indexCount, "\n",
      "  linkedInstances: ", m_linkedInstances.size(), "\n",
      "  cachedMaterials: ", m_materials.size(), "\n",
      "  buildGeometries: ", buildGeometries.size(), "\n",
      "  buildRanges: ", buildRanges.size(), "\n",
      "  dynamicBlas: ", (dynamicBlas != nullptr ? "valid" : "null")));
    
    // Print main material info
    Logger::warn("=== Main Material Info ===");
    input.getMaterialData().printDebugInfo("(main)");
    
    // Print cached materials info
    if (!m_materials.empty()) {
      Logger::warn("=== Cached Materials Info ===");
      for (const auto& [hash, material] : m_materials) {
        Logger::warn(str::format("Cached Material Hash: 0x", std::hex, hash, std::dec));
        material.printDebugInfo("(cached)");
      }
    }
#endif
  }

private:
  std::vector<RtInstance*> m_linkedInstances;
  InstanceMap m_spatialMap;
  std::unordered_map<XXH64_hash_t, LegacyMaterialData> m_materials;
};

// Top-level acceleration structure
struct Tlas {
  enum Type : size_t {
    Opaque,
    Unordered,
    SSS,

    Count
  };

  VkBuildAccelerationStructureFlagsKHR flags = 0;
  Rc<DxvkAccelStructure> accelStructure = nullptr;
  Rc<DxvkAccelStructure> previousAccelStructure = nullptr;

  // NV-DXVK [TlasOrphans]: the instance count last built into each AS backing
  // buffer. Swapped together with the accelStructure swap (Opaque alternates
  // between two buffers), so each count always describes the buffer it rides
  // with. AccelManager::internalBuildTlas forces a FRESH AS object whenever
  // the new build has FEWER instances than the last build into that same
  // buffer — reusing it leaves the prior build's instance metadata physically
  // present past the new build's extent on NVIDIA drivers, and primary rays
  // can hit those orphans and return a STALE customIndex (documented at the
  // [AS-Shrink-Realloc] note; previously only guarded at >2x shrink).
  uint32_t builtInstanceCount = 0u;
  uint32_t previousBuiltInstanceCount = 0u;
};

enum class RtxGeometryStatus {
  Ignored,
  RayTraced,
  Rasterized
};

struct DxvkRaytracingInstanceState {
  Rc<DxvkBuffer> vsConstantsCB;
  Rc<DxvkBuffer> psSharedStateCB;
  Rc<DxvkBuffer> vertexCaptureCB;
};

enum class RtxFramePassStage {
  FrameBegin,
  Volumetrics,
  VolumeIntegrateRestirInitial,
  VolumeIntegrateRestirVisible,
  VolumeIntegrateRestirTemporal,
  VolumeIntegrateRestirSpatialResampling,
  VolumeIntegrateRaytracing,
  GBufferPrimaryRays,
  ReflectionPSR,
  TransmissionPSR,
  RTXDI_InitialTemporalReuse,
  RTXDI_SpatialReuse,
  NEE_Cache,
  DirectIntegration,
  RTXDI_ComputeGradients,
  IndirectIntegration,
  NEE_Integration,
  NRC,
  RTXDI_FilterGradients,
  RTXDI_ComputeConfidence,
  ReSTIR_GI_TemporalReuse,
  ReSTIR_GI_SpatialReuse,
  ReSTIR_GI_FinalShading,
  Demodulate,
  NRD,
  CompositionAlphaBlend,
  Composition,
  DLSS,
  DLSSRR,
  NIS,
  XeSS,
  TAA,
  DustParticles,
  Bloom,
  PostFX,
  AutoExposure_Histogram,
  AutoExposure_Exposure,
  ToneMapping,
  FrameEnd
};

enum class RtxTextureExtentType {
  DownScaledExtent,
  TargetExtent,
  Custom
};

// Category of texture format base on the doc: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap46.html#formats-compatibility-classes
// Note: We currently only categorize the uncompressed color textures
enum class RtxTextureFormatCompatibilityCategory : uint32_t {
  Color_Format_8_Bits,
  Color_Format_16_Bits,
  Color_Format_32_Bits,
  Color_Format_64_Bits,
  Color_Format_128_Bits,
  Color_Format_256_Bits,

  Count,
  InvalidFormatCompatibilityCategory = UINT32_MAX
};

} // namespace dxvk
