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

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <variant>

#include "../dxvk_buffer.h"
#include "../dxvk_image.h"
#include "../dxvk_staging.h"
#include "../dxvk_bind_mask.h"
#include "../util/util_hashtable.h"

#include "rtx_globals.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_camera_manager.h"
#include "rtx_draw_call_cache.h"
#include "rtx_sparse_unique_cache.h"
#include "rtx_light_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_ray_portal_manager.h"
#include "rtx_bindless_resource_manager.h"
#include "rtx_objectpicking.h"
#include "rtx_mod_manager.h"
#include "graph/rtx_graph_manager.h"
#include "rtx_particle_system.h"
#include "rtx_cb_types.h"

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
struct AssetReplacement;
struct AssetReplacer;
class OpacityMicromapManager;
class TerrainBaker;

// The resource cache can be *searched* by other users
class ResourceCache {
public:
  bool find(const RtSurfaceMaterial& surf, uint32_t& outIdx) const { return m_surfaceMaterialCache.find(surf, outIdx); }
  const RtSurfaceMaterial& get(const uint32_t index) const { return m_surfaceMaterialCache.getObjectTable()[index]; }

  // NV-DXVK [perf] handoff v5 sec 4c: BINDING EPOCH.
  //
  // A monotonic stamp that changes whenever a surface-material index or a
  // bindless texture slot COULD have moved -- i.e. whenever either cache took an
  // insert, a free or a clear. It exists so a consumer can cache "this instance
  // is already bound to material index N" across frames without re-deriving it.
  //
  // WHY IT IS REQUIRED, not belt-and-braces: m_preCreationSurfaceMaterialMap is
  // cleared every frame with the stated reason "not currently safe to cache these
  // across frames (due to texture indices and rtx options potentially changing)".
  // Any cross-frame cache of a material index inherits that hazard exactly. In a
  // settled scene [MatChurn] reports matNew=0 texNew=0 texFree=0 texClear=0 so the
  // epoch is flat and the cache holds; during streaming it moves and every cached
  // binding is invalidated in the same frame the slot moved. Without it, a texture
  // that streamed into a recycled bindless slot would render on the wrong surface.
  uint64_t getBindingEpoch() const { return m_bindingEpoch; }

protected:
  BufferRefTable<RaytraceBuffer> m_bufferCache;
  BufferRefTable<Rc<DxvkSampler>> m_materialSamplerCache;

  struct SurfaceMaterialHashFn {
    size_t operator() (const RtSurfaceMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialCache;
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialExtensionCache;
  fast_unordered_cache<uint32_t> m_preCreationSurfaceMaterialMap;

  // NV-DXVK [perf]: memoizes LegacyMaterialData::as<OpaqueMaterialData>() for one
  // frame, keyed on LegacyMaterialData::getOpaqueConversionKey(). Cleared every
  // frame next to m_preCreationSurfaceMaterialMap above and for the same stated
  // reason -- the conversion reads rtx options and texture indices that can move
  // between frames. Touched only from submitDrawState on the cs thread, like the
  // uniqueHashes set in the same function, so it carries no lock.
  fast_unordered_cache<OpaqueMaterialData> m_legacyOpaqueConversionCache;
  uint64_t m_legacyOpaqueConversionHits = 0;
  uint64_t m_legacyOpaqueConversionMisses = 0;

  struct VolumeMaterialHashFn {
    size_t operator() (const RtVolumeMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtVolumeMaterial, VolumeMaterialHashFn> m_volumeMaterialCache;

  struct SamplerHashFn {
    size_t operator() (const Rc<DxvkSampler>& sampler) const {
      return (size_t) sampler->hash();
    }
  };

  struct SamplerKeyEqual {
    bool operator()(const Rc<DxvkSampler>& lhs, const Rc<DxvkSampler>& rhs) const {
      return lhs->info() == rhs->info();
    }
  };

  SparseUniqueCache<Rc<DxvkSampler>, SamplerHashFn, SamplerKeyEqual> m_samplerCache;

  // NV-DXVK [perf] sec 4c: see getBindingEpoch(). Recomputed once per frame in
  // SceneManager from the caches' own monotonic insert/free/clear counters, so it
  // cannot drift out of sync with them the way a hand-maintained dirty flag would.
  uint64_t m_bindingEpoch = 0;
};

struct ExternalDrawState {
  DrawCallState drawCall {};
  remixapi_MeshHandle mesh {};
  CameraType::Enum cameraType {};
  CategoryFlags categories {};
  bool doubleSided {};
  const std::optional<RtxParticleSystemDesc> optionalParticleDesc {};
  std::vector<Matrix4> gpuInstancingTransforms {};
};

// NV-DXVK [perf, GPU index stash]: recycling allocator for the per-draw index
// stash buffers (RasterGeometry::indexDataGpuStash).
//
// The stash exists so the CS thread can capture a dynamic index buffer's exact
// contents with a GPU->GPU copy instead of the game thread streaming those
// write-combined bytes into system memory. Allocating one buffer per stashed
// draw made that a bad trade: ~600 createBuffer calls per frame, all on the CS
// thread, all serialised on the memory-allocator lock, and all retained by the
// BlasEntry holding the geometry.
//
// Buffers are handed out rounded up to a power-of-two size class so releases
// are interchangeable, and are checked back in by the handle's deleter when the
// last referencing RasterGeometry copy dies. Steady state allocates nothing:
// each BlasEntry's next-frame stash reuses the buffer its previous stash freed.
//
// Thread safety: acquire() runs on the CS thread; the deleter runs on whichever
// thread drops the final reference (CS thread, or the render thread when a
// BlasEntry is garbage-collected), so the free list is mutex-guarded.
//
// Lifetime: owned by SceneManager (a CommonDeviceObject), because DxvkBuffer
// holds a raw DxvkDevice* and must not outlive the device. Handles keep only a
// weak_ptr, so a stash outliving the pool simply frees its buffer normally.
class IndexStashPool : public std::enable_shared_from_this<IndexStashPool> {
public:
  explicit IndexStashPool(DxvkDevice* device) : m_device(device) { }

  // Returns a handle owning a device-local buffer of at least `bytes`.
  // Returns nullptr only if `bytes` is 0 or buffer creation fails.
  std::shared_ptr<RasterGeometry::IndexGpuStash> acquire(VkDeviceSize bytes);

  // Releases buffers that have sat idle for kIdleFramesBeforeRelease frames.
  // Called once per frame from SceneManager::garbageCollection.
  void reclaim(uint32_t currentFrameId);

  // Drops every cached free buffer. Must be called while the device is still
  // alive (SceneManager::onDestroy).
  void shutdown();

  // Diagnostics for [IdxStashPool]: buffers created vs reused vs aged-out.
  uint64_t created() const { return m_created.load(std::memory_order_relaxed); }
  uint64_t reused() const { return m_reused.load(std::memory_order_relaxed); }
  uint64_t released() const { return m_released.load(std::memory_order_relaxed); }
  VkDeviceSize bytesHeld() const { return m_bytesHeld.load(std::memory_order_relaxed); }

private:
  void release(Rc<DxvkBuffer>&& buffer);
  static VkDeviceSize sizeClassFor(VkDeviceSize bytes);

  struct FreeEntry {
    Rc<DxvkBuffer> buffer;
    uint32_t lastReleasedFrameId;
  };

  // How the pool is bounded — deliberately NOT by a size or count cap.
  //
  // A cap on the free list would be a performance cliff, not a memory fix: it
  // does not limit live buffers (those are held by BlasEntries and are as large
  // as the scene demands), it only stops recycling once the cap is hit. Past
  // that point every release is dropped and every acquire calls createBuffer
  // again — reinstating the exact per-draw allocation churn this pool exists to
  // remove, in the heaviest scenes, silently.
  //
  // The pool is instead self-bounding by DEMAND: a buffer can only reach the
  // free list if it was checked out first, so buffers-per-class never exceeds
  // that class's peak concurrent use. The only way idle memory accumulates is
  // size-class fragmentation across scene changes (a 512 KB buffer cannot serve
  // a 4 KB request), so the fix targets exactly that: buffers idle for
  // kIdleFramesBeforeRelease consecutive frames are returned to the driver.
  //
  // Getting this constant wrong is safe in a way a cap is not — it changes only
  // WHEN surplus memory is returned, never whether the hot path allocates. In
  // steady state every pooled buffer is reused within a frame or two, so
  // nothing ages out and the reclaim is a no-op.
  static constexpr uint32_t kIdleFramesBeforeRelease = 256;
  static constexpr VkDeviceSize kMinClassBytes = 4 * 1024;

  DxvkDevice* m_device;
  dxvk::mutex m_mutex;
  std::unordered_map<VkDeviceSize, std::vector<FreeEntry>> m_free;
  std::atomic<uint64_t> m_created = { 0u };
  std::atomic<uint64_t> m_reused = { 0u };
  std::atomic<uint64_t> m_released = { 0u };
  std::atomic<VkDeviceSize> m_bytesHeld = { 0u };
  bool m_shutdown = false;
};

// Scene manager is a super manager, it's the interface between rendering and world state
// along with managing the operation of other caches, scene manager also manages the cache
// directly for "SceneObject"'s - which are "unique meshes/geometry", which map 1-to-1 with
// BLAS entries in raytracing terminology.
class SceneManager : public CommonDeviceObject, public ResourceCache {
public:
  SceneManager(SceneManager const&) = delete;
  SceneManager& operator=(SceneManager const&) = delete;

  explicit SceneManager(DxvkDevice* device);
  ~SceneManager();

  void initialize(Rc<DxvkContext> ctx);
  void logStatistics();

  void onDestroy();

  // NV-DXVK [perf, GPU index stash]: recycling allocator for per-draw index
  // stash buffers. Created in the SceneManager ctor so it is always valid.
  IndexStashPool& getIndexStashPool() { return *m_indexStashPool; }

  void submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData);
  void submitExternalDraw(Rc<DxvkContext> ctx, ExternalDrawState&& state);
  
  bool areAllReplacementsLoaded() const;
  std::vector<Mod::State> getReplacementStates() const;

  RtxGlobals& getGlobals() { return m_globals; }

  Rc<DxvkBuffer> getSurfaceMaterialBuffer() { return m_surfaceMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceMaterialExtensionBuffer() { return m_surfaceMaterialExtensionBuffer; }
  Rc<DxvkBuffer> getVolumeMaterialBuffer() { return m_volumeMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceBuffer() const { return m_accelManager.getSurfaceBuffer(); }
  Rc<DxvkBuffer> getSurfaceMappingBuffer() const { return m_accelManager.getSurfaceMappingBuffer(); }
  Rc<DxvkBuffer> getCurrentFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getCurrentFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getLastFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getLastFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getBillboardsBuffer() const { return m_accelManager.getBillboardsBuffer(); }
  bool isPreviousFrameSceneAvailable() const { return m_previousFrameSceneAvailable && getSurfaceMappingBuffer().ptr() != nullptr; }

  const std::vector<Rc<DxvkSampler>>& getSamplerTable() const { return m_samplerCache.getObjectTable(); }
  const std::vector<RaytraceBuffer>& getBufferTable() const { return m_bufferCache.getObjectTable(); }
  const std::vector<RtInstance*>& getInstanceTable() const { return m_instanceManager.getInstanceTable(); }
  
  const InstanceManager& getInstanceManager() const { return m_instanceManager; }
  const AccelManager& getAccelManager() const { return m_accelManager; }
  const LightManager& getLightManager() const { return m_lightManager; }
  const GraphManager& getGraphManager() const { return m_graphManager; }
  const RayPortalManager& getRayPortalManager() const { return m_rayPortalManager; }
  const BindlessResourceManager& getBindlessResourceManager() const { return m_bindlessResourceManager; }
  OpacityMicromapManager* getOpacityMicromapManager() const { return m_opacityMicromapManager.get(); }
  LightManager& getLightManager() { return m_lightManager; }
  GraphManager& getGraphManager() { return m_graphManager; }
  std::unique_ptr<AssetReplacer>& getAssetReplacer() { return m_pReplacer; }
  TerrainBaker& getTerrainBaker() { return *m_terrainBaker.get(); }

  // Scene utility functions
  static Vector3 getSceneUp();
  static Vector3 getSceneForward();
  static Vector3 calculateSceneRight();

  // Reswizzles input vector to an output that has xy coordinates on scene's horizontal axes and z coordinate to be on the scene's vertical axis
  static Vector3 worldToSceneOrientedVector(const Vector3& worldVector); 

  static Vector3 sceneToWorldOrientedVector(const Vector3& sceneVector);

  void addLight(const RtxLegacyLight& light);

  const CameraManager& getCameraManager() const { return m_cameraManager; }
  CameraManager& getCameraManager() { return m_cameraManager; }
  const RtCamera& getCamera() const { return m_cameraManager.getMainCamera(); }
  RtCamera& getCamera() { return m_cameraManager.getMainCamera(); }

  const FogState& getFogState() const { return m_fog; }
  FogState& getFogState() { return m_fog; }
  const fast_unordered_cache<FogState>& getFogStates() const { return m_fogStates; }

  uint32_t getStartInMediumMaterialIndex() { return m_startInMediumMaterialIndex; }

  // NV-DXVK: TF2 holo-character / viewmodel screen-space emissive needs
  // the engine's `c_gameTime` (CBufCommonPerCamera offset 300) as the
  // multiplier on c_uv1Translate, so the scrolling pattern matches native.
  // The d3d11 layer captures it from any draw that uses the screen-space
  // emissive pattern (see [ScreenSpaceEmissive.GameTimeWatch] in
  // d3d11_rtx.cpp::FillMaterialData) and stashes it here. RtxContext reads
  // the latest value when populating RaytraceArgs.screenSpaceEmissiveTime
  // each frame. Atomic because FillMaterialData runs on the cs thread
  // while raytrace-args fill runs from the main render path.
  //
  // Why not RtxOptions::timeSinceStartSeconds: that's wall-clock from app
  // start, which keeps ticking during pause. The native PS uses gameplay
  // time which freezes during pause — so the lines should also freeze.
  void setEngineGameTime(float t) { m_engineGameTime.store(t, std::memory_order_relaxed); }
  float getEngineGameTime() const { return m_engineGameTime.load(std::memory_order_relaxed); }

  // NV-DXVK: TF2 3D-skybox cloud fog reconstruction. Camera-global fog
  // params (CBufCommonPerCamera c_fogParams k0-k3 + c_maxLightingValue) and
  // the cloud material's c_fogColorFactor, captured per-frame from the cloud
  // draws in d3d11_rtx.cpp::FillMaterialData. RtxContext copies these into
  // RaytraceArgs.tf2Fog* each frame. Mutex-guarded because FillMaterialData
  // runs on the cs thread while raytrace-args fill runs from the render
  // path, and the four vectors must be read as a consistent set.
  struct Tf2CloudFogParams {
    Vector4 k1_k0w  = Vector4(0.f, 0.f, 0.f, 0.f); // xyz = k1, w = k0.w
    Vector4 k2_k2w  = Vector4(0.f, 0.f, 0.f, 0.f); // xyz = k2, w = k2.w
    Vector4 k3      = Vector4(0.f, 0.f, 0.f, 0.f); // xyz = sun dir, w = sun scale
    Vector4 misc    = Vector4(0.f, 0.f, 0.f, 0.f); // x = c_fogColorFactor, y = c_maxLightingValue, z = valid
  };
  void setTf2CloudFog(const Tf2CloudFogParams& p) {
    std::lock_guard<std::mutex> lk(m_tf2CloudFogMutex);
    m_tf2CloudFog = p;
  }
  Tf2CloudFogParams getTf2CloudFog() const {
    std::lock_guard<std::mutex> lk(m_tf2CloudFogMutex);
    return m_tf2CloudFog;
  }
  
  uint32_t getActivePOMCount() {return m_activePOMCount;}

  float getTotalMipBias();
  float getCalculatedUpscalingMipBias();

  // ISceneManager but not really
  void clear(Rc<DxvkContext> ctx, bool needWfi);
  void garbageCollection();
  void prepareSceneData(Rc<RtxContext> ctx, class DxvkBarrierSet& execBarriers);

  void onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame);

  // GameCapturer
  void triggerUsdCapture() const;
  bool isGameCapturerIdle() const;

  using SamplerIndex = uint32_t;

  void trackTexture(const TextureRef& inputTexture,
                    uint32_t& textureIndex,
                    bool hasTexcoords,
                    bool async = true,
                    uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID);
  [[nodiscard]] SamplerIndex trackSampler(Rc<DxvkSampler> sampler);

  std::optional<XXH64_hash_t> findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue);
  std::vector<ObjectPickingValue> gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash);

  // Replacement material hash tracking
  void trackReplacementMaterialHash(XXH64_hash_t materialHash);
  bool isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const;
  uint32_t getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const;
  void clearFrameReplacementMaterialHashes();

  // Mesh hash tracking
  void trackMeshHash(XXH64_hash_t meshHash);
  bool isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const;
  uint32_t getMeshHashUsageCount(XXH64_hash_t meshHash) const;
  void clearFrameMeshHashes();

  Rc<DxvkSampler> patchSampler( const VkFilter filterMode,
                                const VkSamplerAddressMode addressModeU,
                                const VkSamplerAddressMode addressModeV,
                                const VkSamplerAddressMode addressModeW,
                                const VkClearColorValue borderColor);

  void requestTextureVramFree();
  void requestVramCompaction();
  void manageTextureVram();

  bool isThinOpaqueMaterialExist() const { return m_thinOpaqueMaterialExist; }
  bool isSssMaterialExist() const { return m_sssMaterialExist; }

  bool isAntiCullingSupported() const { return m_isAntiCullingSupported; }

private:
  enum class ObjectCacheState
  {
    kUpdateInstance = 0,
    kUpdateBVH = 1,
    KBuildBVH = 2,
    kInvalid = -1
  };
  // Handles conversion of geometry data coming from a draw call, to the data used by the raytracing backend
  template<bool isNew>
  ObjectCacheState processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, RaytraceGeometry& modifiedGeometryData);

  // Consumes a draw call state and updates the scene state accordingly
  RtInstance* processDrawCallState(Rc<DxvkContext> ctx, 
                                   const DrawCallState& blasInput, 
                                   MaterialData& materialData, 
                                   RtInstance* existingInstance = nullptr,
                                   const RtxParticleSystemDesc* pParticleSystemDesc = nullptr);

  const RtSurfaceMaterial& createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                 const DrawCallState& drawCallState,
                                                 uint32_t* out_indexInCache = nullptr);

  // Updates ref counts for new buffers
  void updateBufferCache(RaytraceGeometry& newGeoData);

  // NV-DXVK: auto-dump every unique texture we see (albedo + normal + rough
  // + metallic + emissive + AO + lightmaps + detail + cloudMask) to
  // rtx-remix/captures/textures/ the first time that texture's hash is
  // encountered. Bypasses the capture hotkey; texture export is triggered
  // inside processDrawCallState per bound TextureRef. Deduped by
  // m_autoDumpedTextureHashes.
  void autoDumpMaterialTextures(Rc<DxvkContext> ctx, const MaterialData& material);
  std::unordered_set<XXH64_hash_t> m_autoDumpedTextureHashes;
  std::mutex                       m_autoDumpedTexturesMutex;

  // NV-DXVK: targeted per-draw dump, gated on rtx.debug.dumpVertexShaders.
  // Dumps the bound game textures (deduped by image hash) and logs a
  // geometry/transform report once per matched VS hash. See option doc.
  void dumpDrawForVertexShader(Rc<DxvkContext> ctx, const DrawCallState& drawCallState);
  std::unordered_set<XXH64_hash_t> m_dumpedDrawVsHashes;       // metadata logged once per VS
  std::unordered_set<XXH64_hash_t> m_dumpedDrawTextureHashes;  // textures written once per image
  std::mutex                       m_dumpDrawMutex;

  // Called whenever a new BLAS scene object is added to the cache
  ObjectCacheState onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);
  // Called whenever a BLAS scene object is updated
  ObjectCacheState onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);
  // Called whenever a BLAS scene object is destroyed
  void onSceneObjectDestroyed(const BlasEntry& pBlas);

  // Called whenever a new instance has been added to the database
  void onInstanceAdded(RtInstance& instance);
  // Called whenever instance metadata is updated
  void onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData& material, const bool hasTransformChanged, const bool hasVerticesdChanged, const bool isFirstUpdateThisFrame);
  // Called whenever an instance has been removed from the database
  void onInstanceDestroyed(RtInstance& instance);

  // Called to destroy a ReplacementInstance.
  // This is used to clear up all references to the ReplacementInstance.
  // Also responsible for removing any graphs from graphManager.
  void destroyReplacementInstance(ReplacementInstance* replacementInstance);

  void drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, MaterialData& renderMaterialData);

  void createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance);

  // Print all RtInstances for debugging
  void printAllRtInstances();
  
  MaterialData determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input);
  
  uint32_t m_beginUsdExportFrameNum = -1;
  bool m_enqueueDelayedClear = false;
  bool m_previousFrameSceneAvailable = false;

  RtxGlobals m_globals;

  // Hash/Cache's
  InstanceManager m_instanceManager;
  AccelManager m_accelManager;
  LightManager m_lightManager;
  GraphManager m_graphManager;
  RayPortalManager m_rayPortalManager;
  BindlessResourceManager m_bindlessResourceManager;
  // NV-DXVK [perf, GPU index stash]: shared_ptr (not unique_ptr) because stash
  // handles hold a weak_ptr back to it for check-in.
  std::shared_ptr<IndexStashPool> m_indexStashPool;

  std::unique_ptr<OpacityMicromapManager> m_opacityMicromapManager;

  DrawCallCache m_drawCallCache;

  CameraManager m_cameraManager;

  std::unique_ptr<AssetReplacer> m_pReplacer;

  std::unique_ptr<TerrainBaker> m_terrainBaker;

  FogState m_fog;
  fast_unordered_cache<FogState> m_fogStates;
  uint32_t m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
  uint32_t m_startInMediumMaterialIndex_inCache = UINT32_MAX;

  // TODO: Move the following resources and getters to RtResources class
  Rc<DxvkBuffer> m_surfaceMaterialBuffer;
  Rc<DxvkBuffer> m_surfaceMaterialExtensionBuffer;
  Rc<DxvkBuffer> m_volumeMaterialBuffer;

  uint32_t m_activePOMCount = 0;
  
  float m_uniqueObjectSearchDistance = 1.f;

  struct DrawCallMetaInfo {
    XXH64_hash_t legacyTextureHash { kEmptyHash };
    XXH64_hash_t legacyTextureHash2 { kEmptyHash };
  };
  struct DrawCallMeta {
    constexpr static inline uint8_t MaxTicks = 2;
    std::unordered_map<ObjectPickingValue, DrawCallMetaInfo> infos[MaxTicks] {};
    bool ready[MaxTicks] {};
    uint8_t ticker {};
    dxvk::mutex mutex {};
  } m_drawCallMeta {};

  // TODO: expand to many different
  Rc<DxvkSampler> m_externalSampler = nullptr;

  std::atomic_bool m_forceFreeTextureMemory = false;
  std::atomic_bool m_forceFreeUnusedDxvkAllocatorChunks = false;

  // NV-DXVK [Perf.ChunkTrim] (2026-08-06): when the automatic empty-chunk trim
  // last ran, for rtx.autoFreeUnusedChunksCooldownMs. Only ever touched from
  // SceneManager::manageTextureVram on the frame-end thread, so it needs no
  // synchronisation -- unlike the two force atomics above, which are set from
  // the UI thread. time_point::min() means "never trimmed", which makes the
  // first qualifying frame fire immediately rather than waiting out a cooldown
  // measured from process start.
  std::chrono::steady_clock::time_point m_lastChunkTrimTime =
    std::chrono::steady_clock::time_point::min();

  // NV-DXVK [Perf.ChunkTrim]: slack level at which an automatic trim last found
  // NOTHING to free, or 0 when the last one was productive. Suppresses further
  // attempts until slack grows past it -- see the back-off reasoning in
  // SceneManager::manageTextureVram. Same thread as m_lastChunkTrimTime.
  VkDeviceSize m_fruitlessTrimSlack = 0;

  // NV-DXVK: latest captured engine c_gameTime — see setEngineGameTime
  // comment above. Default 0 → before any draw of the screen-space
  // emissive pattern fires, the slang samples at the constant baseline
  // (translate × 0 = no offset), which matches native at game-time-zero.
  std::atomic<float> m_engineGameTime { 0.0f };

  // NV-DXVK: latest captured TF2 cloud fog params — see setTf2CloudFog.
  mutable std::mutex m_tf2CloudFogMutex;
  Tf2CloudFogParams m_tf2CloudFog;

  bool m_thinOpaqueMaterialExist = false;
  bool m_sssMaterialExist = false;

  bool m_isAntiCullingSupported = true;

  // Replacement material hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameReplacementMaterialHashes;

  // Mesh hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameMeshHashes;

  // Using std::deque for pointer stability: push_back doesn't invalidate existing pointers
  std::deque<std::vector<Matrix4>> m_externalGpuInstancingTransforms;

  // NV-DXVK [MatChurn]: material/texture identity churn, measured per frame.
  //
  // WHY THIS EXISTS. The flicker survives every geometry-side elimination and
  // is present in RAW ALBEDO (debugViewIdx=32) while the per-pixel vertex-shader
  // view (861) stays rock steady: the same surface is hit and resolved every
  // frame but returns a different albedo. The only remaining chain is
  // surface -> material -> texture, and material identity in this fork is
  // derived from the DxvkImage POINTER, which the game's texture streaming
  // recreates while the object is still on screen. That predicts a continuous,
  // camera-independent stream of NEW material and texture identities for
  // objects that never move - which is what these counters measure.
  //
  // Everything is a monotonic counter sampled once per frame and diffed against
  // the previous sample, so a per-frame rate needs no reset and no coordination
  // with frame boundaries. Aggregate only, one line per frame: per-draw logging
  // has repeatedly heisen-masked this artifact by slowing the CS thread.
  //
  // Read it as: in a steady scene with a still camera, matNew / texNew / imgNew
  // should all be 0. Any of them running hot continuously IS the churn, and the
  // frames where they spike are joinable by f= against the on-screen flick.
  struct MaterialChurnSample {
    uint64_t matLookups = 0;   // createSurfaceMaterial calls
    uint64_t matMemoHits = 0;  // ...answered by the single-entry memo, before any work
    uint64_t matPreMiss = 0;   // ...that missed the per-frame pre-creation map
    uint64_t matInserts = 0;   // ...that produced a brand new material index
    uint64_t matExtInserts = 0;// new subsurface-extension material indices
    uint64_t matClears = 0;    // SceneManager::clear() wiped the material cache
    uint64_t samplerInserts = 0;
    uint64_t texInserts = 0;   // new bindless texture slots
    uint64_t texFrees = 0;
    uint64_t texClears = 0;
    uint64_t imgStamps = 0;    // new game DxvkImages entering the material path
    uint64_t mtQueued = 0;
    uint64_t mtDemoteReq = 0;
    uint64_t mtVidMem = 0;
    uint64_t mtViewSwaps = 0;
    uint64_t mtFailed = 0;
    bool valid = false;        // false until the first sample is taken
  };
  MaterialChurnSample m_prevChurn;

  // The two counters above that have no home in a cache object: how many times
  // createSurfaceMaterial ran this session, and how many of those had to build a
  // material from scratch because m_preCreationSurfaceMaterialMap (cleared every
  // frame) had no entry. The ratio is the per-frame dedup rate; the INSERT count
  // into m_surfaceMaterialCache is the part that should be zero in a steady scene.
  uint64_t m_matLookupCount = 0;
  uint64_t m_matPreMissCount = 0;
  uint64_t m_matMemoHitCount = 0;

  // NV-DXVK [perf]: single-entry memo in front of createSurfaceMaterial.
  //
  // WHY. createSurfaceMaterial reads nothing from the instance -- only
  // (renderMaterialData, drawCallState) -- yet it is called from
  // SceneManager::onInstanceUpdated, which runs once per INSTANCE. [MatChurn]
  // measures the consequence directly: matLookup=15456 per frame against
  // matBuild=351, i.e. ~97.7% of calls cannot produce a distinct answer. The
  // existing m_preCreationSurfaceMaterialMap already dedups them, but only
  // AFTER the prologue has resolved the sampler (Rc refcount churn plus a
  // SparseUniqueCache map lookup), touched two static log-once sets, and folded
  // four XXH64s into preCreationHash. Instances of one draw arrive consecutively
  // (processSceneObjectFanout loops placements of a single draw call), so one
  // entry collapses that run to its first element.
  //
  // EXACTNESS. preCreationHash -- the key the existing per-frame map uses -- is
  // a digest of exactly five things: renderMaterialData.getHash(), samplerIndex,
  // samplerIndex2, hasTexcoords and isUsingRaytracedRenderTarget. The first,
  // fourth and fifth are stored below verbatim. The two sampler indices are a
  // deterministic function of the three sampler pointers plus isEye(), which are
  // stored below as well, so equal keys here imply an equal preCreationHash and
  // therefore the identical cache index. This memo is not a widening of the
  // existing contract -- it keys on strictly more state than preCreationHash
  // does, so anything it serves the old map would have served too.
  //
  // LIFETIME. Same as m_preCreationSurfaceMaterialMap, and invalidated at the
  // same two sites, for the same stated reason: the build path reads rtx options
  // and texture indices that can move between frames. frameId is belt-and-braces
  // on top of that.
  struct SurfaceMaterialMemo {
    uint32_t     frameId                 = UINT32_MAX;
    XXH64_hash_t materialHash            = 0;
    const void*  samplerOverride         = nullptr;
    const void*  drawSampler             = nullptr;
    const void*  drawSampler2            = nullptr;
    uint32_t     materialType            = UINT32_MAX;
    uint32_t     indexInCache            = UINT32_MAX;
    bool         hasTexcoords            = false;
    bool         isEye                   = false;
    bool         isRaytracedRenderTarget = false;
    bool         valid                   = false;
  };
  // NV-DXVK [Phase2b prerequisite] 2026-08-15: ONE SLOT PER WORKER, not one slot.
  //
  // This was a single memo written on every instance update. Under N workers that is
  // two separate failures at once, and the call site advertises neither:
  //   1. a data race -- concurrent writers tearing a multi-field struct, and a reader
  //      matching some fields against one entry and the rest against another, which
  //      can serve the WRONG material rather than merely missing;
  //   2. a hit-rate collapse -- one slot shared by N workers thrashes, and the memo
  //      is currently a 67-68% hit ([Perf.MatCache]) precisely because consecutive
  //      draws on ONE thread repeat their material. Split that stream across N
  //      workers and the locality it depends on is gone.
  //
  // Per-slot rather than plain thread_local because the memo holds an INDEX INTO
  // m_surfaceMaterialCache: both invalidation sites exist because a cache clear
  // renumbers every material, so invalidation MUST reach every slot. A thread_local
  // would leave each worker's entry alive across a reset, still pointing at an index
  // that now names a different material -- silent corruption, not a stale miss.
  //
  // Behaviourally identical while everything still runs on the CS thread: one thread
  // claims slot 0 and nothing else is ever touched.
  static constexpr uint32_t kSurfaceMaterialMemoSlots = 64;
  SurfaceMaterialMemo m_lastSurfaceMaterial[kSurfaceMaterialMemoSlots];
  std::atomic<uint32_t> m_surfaceMaterialMemoNextSlot = { 0 };
  // Stable per-thread slot id, handed out on first use. Overflow parks every extra
  // thread on the last slot: degraded hit rate, never a wrong answer, never OOB.
  uint32_t surfaceMaterialMemoSlot();
  // Clears ALL slots. Every existing `m_lastSurfaceMaterial.valid = false` site must
  // call this instead -- see the index-renumbering argument above.
  void invalidateSurfaceMaterialMemo();

  void logMaterialChurn();
};

}  // namespace nvvk
