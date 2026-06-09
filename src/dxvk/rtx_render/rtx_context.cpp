/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>
#include <cmath>
#include <cassert>
#include <array>
#include <chrono>
#include <future>
#include <set>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_shader_manager.h"
#include "dxvk_adapter.h"
#include "rtx_context.h"
#include "rtx_asset_exporter.h"
#include "rtx_options.h"
#include "rtx_engine_sun.h"
#include "rtx_bindless_resource_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_texture.h"
#include "rtx_neural_radiance_cache.h"
#include "rtx_ray_reconstruction.h"
#include "rtx_xess.h"
#include "rtx_rtxdi_rayquery.h"
#include "rtx_restir_gi_rayquery.h"
#include "rtx_composite.h"
#include "rtx_debug_view.h"
#include "rtx_debug_probes.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/volume_args.h"
#include "rtx/utility/debug_view_indices.h"
#include "rtx/utility/gpu_printing.h"
#include "rtx/utility/scene_dump.h"
#include "rtx_nrd_settings.h"
#include "rtx_scene_manager.h"

#include "rtx_cb_types.h"
#include "rtx_spec_constants.h"

#include "../util/log/metrics.h"
#include "../util/util_defer.h"
#include "../util/util_globaltime.h"

#include "rtx_imgui.h"
#include "dxvk_scoped_annotation.h"
#include "imgui/dxvk_imgui.h"

#include <ctime>
#include <fstream>
#include <nvapi.h>

#include <NvLowLatencyVk.h>
#include <pclstats.h>

#include "rtx_matrix_helpers.h"
#include "../util/util_fastops.h"

// Destructor requires the struct definitions
#include "rtx_sky.h"

namespace dxvk {

  // NV-DXVK [SkinAABB]: center-pixel VS hash sink — defined in
  // rtx_camera_manager.cpp, written by PickRegion2 below, read by d3d11_rtx.cpp's
  // [SkinAABB] gate so the skin probe follows the crosshair.
  namespace tf2 { extern std::atomic<uint64_t> g_pickCenterVsHash; }
  namespace tf2 { extern std::atomic<uint32_t> g_pickCenterDrawId; }

  // NV-DXVK [InvalidSceneProbe]: defined in rtx_camera_manager.cpp; used
  // below to detect whether we've entered gameplay so we can lift the
  // probe rate-limit for that window without spamming during pre-gameplay
  // loading.
  namespace tf2 {
    extern std::atomic<uint32_t> g_engineHookCaptureCount;
  }

  Metrics Metrics::s_instance;

  bool g_allowSrgbConversionForOutput = true;
  bool g_forceKeepObjectPickingImage = false;

  // [SkySrvSanitize] Shared RAII guard for both sky paths
  // (rasterizeToSkyMatte and rasterizeToSkyProbe). Both replay the captured
  // TF2 sky draw with its original PS shader resource bindings — including
  // stale R32_UINT SRVs that TF2 left bound from a previous non-sky draw
  // (e.g. an object-picking pass). The sky pipeline layout declares those
  // same PS slots as float sampled images with LINEAR samplers, which trips
  // two Vulkan validation errors:
  //   * sampled image FLOAT component vs bound VK_FORMAT_R32_UINT
  //   * VK_FILTER_LINEAR sampler vs format missing
  //     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
  // Both errors describe the SAME root: integer-format view bound to a slot
  // the sky shader reads as float. Real GPU result: uint bits reinterpreted
  // as float through a LINEAR sampler — produces NaN/Inf → visible
  // corruption. Fix: null out integer-format views before the sky draw,
  // restore on scope exit so subsequent passes (RT path, post) see the
  // original bindings intact.
  //
  // PS SRV slot range in m_rc, derived from dxbc_util.h:
  //   DxbcStageBindingCount = 16 cb + 16 samp + 128 srv = 160
  //   PixelShader stage index = 0 → stageOffset = 0
  //   SRV offset within stage = 32 (after cb + samp)
  //   → PS SRV slot range: m_rc[32 .. 159]
  class PsSrvIntegerSanitizeGuard {
  public:
    explicit PsSrvIntegerSanitizeGuard(DxvkContext* ctx) : m_ctx(ctx) {
      constexpr uint32_t kPsSrvSlotBase  = 32;
      constexpr uint32_t kPsSrvSlotCount = 128;
      m_saved.reserve(8);
      for (uint32_t i = 0; i < kPsSrvSlotCount; ++i) {
        const uint32_t slot = kPsSrvSlotBase + i;
        auto& s = m_ctx->getShaderResourceSlot(slot);
        if (s.imageView == nullptr) continue;
        if (!isIntegerSampledFormat(s.imageView->info().format)) continue;
        m_saved.emplace_back(slot, s.imageView);
        m_ctx->bindResourceView(slot, nullptr, nullptr);
      }
      if (!m_saved.empty()) {
        ONCE(Logger::info(str::format(
          "[SkySrvSanitize] unbound ", m_saved.size(),
          " PS SRV slot(s) with integer formats around sky draw")));
      }
    }
    ~PsSrvIntegerSanitizeGuard() {
      for (const auto& kv : m_saved) {
        m_ctx->bindResourceView(kv.first, kv.second, nullptr);
      }
    }
    PsSrvIntegerSanitizeGuard(const PsSrvIntegerSanitizeGuard&) = delete;
    PsSrvIntegerSanitizeGuard& operator=(const PsSrvIntegerSanitizeGuard&) = delete;
  private:
    static bool isIntegerSampledFormat(VkFormat f) {
      switch (f) {
        case VK_FORMAT_R8_UINT:           case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8G8_UINT:         case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8B8A8_UINT:     case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_B8G8R8A8_UINT:     case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32:
        case VK_FORMAT_R16_UINT:          case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16G16_UINT:       case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32_UINT:          case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32G32_UINT:       case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32B32_UINT:    case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R64_UINT:          case VK_FORMAT_R64_SINT:
          return true;
        default:
          return false;
      }
    }
    DxvkContext* m_ctx;
    std::vector<std::pair<uint32_t, Rc<DxvkImageView>>> m_saved;
  };

  // NV-DXVK: per-pixel scene dump state. One-shot, ImGui/F11 triggered.
  // The opaque material shader writes a SceneDumpElement per primary-ray
  // pixel for the single frame after the trigger fires; the CPU then waits
  // a few frames (kMaxFramesInFlight + 1) for the GPU to finish, vkDeviceWait
  // for insurance, maps the host-visible buffer, and writes a CSV alongside
  // the .exe. State machine kept here (file-static) because the trigger,
  // CB fill, and readback all live in different RtxContext methods.
  namespace {
    enum class SceneDumpState : uint32_t { Idle, AwaitingReadback };
    SceneDumpState g_sceneDumpState = SceneDumpState::Idle;
    uint32_t g_sceneDumpTriggerFrame = 0;
    uint32_t g_sceneDumpReadbackAtFrame = 0;
    VkExtent2D g_sceneDumpExtent { 0, 0 };
    bool g_sceneDumpHotkeyLatch = false;
    bool g_sceneDumpRequestThisFrame = false;
  }

  // External entry point for the ImGui "Capture Scene Dump" button (lives
  // in rtx_debug_view.cpp). Flips the request flag; the next render-loop
  // pass through updateRaytraceArgsConstantBuffer arms the GPU capture.
  void requestSceneDump() {
    g_sceneDumpRequestThisFrame = true;
  }

  void RtxContext::takeScreenshot(std::string imageName, Rc<DxvkImage> image) {
    // NOTE: Improve this, I'd like all textures from the same frame to have the same time code...  Currently sampling the time on each "dump op" results in different timecodes.
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::string path = env::getEnvVar("DXVK_SCREENSHOT_PATH");

    if (path.empty()) {
      path = "./Screenshots/";
    } else if (*path.rbegin() != '/') {
      path += '/';
    }

    auto& exporter = getCommonObjects()->metaExporter();
    exporter.dumpImageToFile(this, path, str::format(imageName, "_", tm.tm_mday, tm.tm_mon, tm.tm_year, "-", tm.tm_hour, tm.tm_min, tm.tm_sec, ".dds"), image);
  }

  // NV-DXVK [OnScreenAlbedoDump]: one-shot dump of the albedo (colorTextures[0])
  // of every on-screen instance, ~10s after gameplay starts, to a SEPARATE
  // folder (NOT the 50GB full capture). Deduped by image hash. Lets the user
  // eyeball the scene's albedos and visually pick out an object (e.g. the
  // dropship hull) that no single VS/material hash can isolate. Files land as
  // <hash>_albedo.dds so the chosen image's hash is its filename. Override the
  // target dir with env DXVK_ONSCREEN_ALBEDO_PATH (default below).
  void RtxContext::dumpOnScreenAlbedosOnce() {
    static bool s_done = false;
    if (s_done) {
      return;
    }

    const auto& instances = getSceneManager().getAccelManager().getOrderedInstances();
    // Gameplay gate: menus/loading submit very few instances. 200 sits well
    // above menu/HUD draw counts and well below a real gameplay frame. Also
    // require a valid main camera so we never fire on a black/identity frame.
    if (instances.size() < 200
        || !getSceneManager().getCamera().isValid(m_device->getCurrentFrameId())) {
      return;
    }

    static bool s_armed = false;
    static std::chrono::steady_clock::time_point s_start {};
    const auto now = std::chrono::steady_clock::now();
    if (!s_armed) {
      s_armed = true;
      s_start = now;
      Logger::info("[OnScreenAlbedoDump] gameplay detected; dumping on-screen albedos in ~10s");
      return;
    }
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - s_start).count() < 10000) {
      return;
    }

    // Fire exactly once.
    s_done = true;

    std::string dir = env::getEnvVar("DXVK_ONSCREEN_ALBEDO_PATH");
    if (dir.empty()) {
      dir = "./rtx-remix/onscreen_albedo_dump/";
    } else if (*dir.rbegin() != '/') {
      dir += '/';
    }

    env::createDirectory(dir);
    auto& exporter = getCommonObjects()->metaExporter();

    // Per-albedo accumulation: the .dds is dumped once, but we tally every
    // on-screen instance using it and record its VS + material hash so the
    // sidecar gives all three identifiers + a draw count per image.
    struct AlbedoInfo {
      uint64_t vsHash = 0ull;
      uint64_t matHash = 0ull;
      uint32_t instances = 0u;
      // NV-DXVK: largest world-space AABB diagonal across all instances using
      // this albedo. The exploded/sky surface is the one with the biggest
      // extent, so sorting by this puts it at the top — no eyeballing.
      // When a new max is found we also capture WHERE the size comes from:
      //   objExtent  = object-space bbox diagonal (huge => the verts are bad)
      //   scaleMax   = largest objectToWorld column length (huge => transform
      //                blow-up, e.g. a bad sub-view / 3D-skybox scale)
      // So objExtent≈normal + scaleMax huge  => transform bug, not verts;
      //    objExtent huge                     => the verts themselves.
      float maxExtent = 0.f;
      float objExtent = 0.f;
      float scaleMax = 0.f;
      Vector3 o2wT{ 0.f, 0.f, 0.f };
      bool dumped = false;
    };
    std::unordered_map<uint64_t, AlbedoInfo> byAlbedo;
    uint32_t dumpedCount = 0;

    for (const RtInstance* inst : instances) {
      if (inst == nullptr) {
        continue;
      }
      const BlasEntry* blas = inst->getBlas();
      if (blas == nullptr) {
        continue;
      }
      const auto& md = blas->input.getMaterialData();
      const TextureRef& tex = md.getColorTexture();
      if (!tex.isValid() || tex.isImageEmpty()) {
        continue;
      }
      const uint64_t h = uint64_t(tex.getImageHash());
      if (h == 0ull) {
        continue;
      }
      AlbedoInfo& info = byAlbedo[h];
      ++info.instances;
      info.vsHash = uint64_t(blas->input.getTransformData().vertexShaderHash);
      info.matHash = uint64_t(md.getHash());

      // NV-DXVK: world-space extent of this instance — transform the
      // geometry's object-space AABB by objectToWorld (8 corners) and take
      // the diagonal. A skybox/backdrop wall reports tens of thousands of
      // units; normal brushes are a few hundred.
      const RasterGeometry& geo = blas->input.getGeometryData();
      if (geo.boundingBox.isValid()) {
        const Matrix4& o2w = blas->input.getTransformData().objectToWorld;
        const Vector3& bmin = geo.boundingBox.minPos;
        const Vector3& bmax = geo.boundingBox.maxPos;
        Vector3 wMin(FLT_MAX, FLT_MAX, FLT_MAX), wMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int c = 0; c < 8; ++c) {
          const Vector4 corner((c & 1) ? bmax.x : bmin.x,
                               (c & 2) ? bmax.y : bmin.y,
                               (c & 4) ? bmax.z : bmin.z, 1.0f);
          const Vector4 w = o2w * corner;
          wMin.x = std::min(wMin.x, w.x); wMin.y = std::min(wMin.y, w.y); wMin.z = std::min(wMin.z, w.z);
          wMax.x = std::max(wMax.x, w.x); wMax.y = std::max(wMax.y, w.y); wMax.z = std::max(wMax.z, w.z);
        }
        const float ext = length(wMax - wMin);
        if (ext > info.maxExtent) {
          info.maxExtent = ext;
          info.objExtent = length(bmax - bmin);
          info.scaleMax = std::max({
            length(Vector3(o2w[0][0], o2w[0][1], o2w[0][2])),
            length(Vector3(o2w[1][0], o2w[1][1], o2w[1][2])),
            length(Vector3(o2w[2][0], o2w[2][1], o2w[2][2])) });
          info.o2wT = Vector3(o2w[3][0], o2w[3][1], o2w[3][2]);
        }
      }
      if (!info.dumped) {
        DxvkImageView* view = tex.getImageView();
        if (view != nullptr) {
          Rc<DxvkImage> image = view->image();
          if (image != nullptr) {
            char name[64];
            snprintf(name, sizeof(name), "%016llx_albedo.dds", (unsigned long long) h);
            exporter.dumpImageToFile(this, dir, name, image);
            info.dumped = true;
            ++dumpedCount;
          }
        }
      }
    }

    // Sidecar manifest: one row per albedo, sorted by on-screen instance count
    // (the dropship hull should be near the top). Find the image whose look
    // matches the ship, read its row -> you have albedoHash + vsHash + matHash.
    {
      std::vector<std::pair<uint64_t, AlbedoInfo>> rows(byAlbedo.begin(), byAlbedo.end());
      // Sort by world extent desc — the sky-spanning surface lands at row 1.
      std::sort(rows.begin(), rows.end(),
        [](const std::pair<uint64_t, AlbedoInfo>& a, const std::pair<uint64_t, AlbedoInfo>& b) {
          return a.second.maxExtent > b.second.maxExtent;
        });
      std::ofstream manifest(dir + "manifest.txt", std::ios::out | std::ios::trunc);
      if (manifest.is_open()) {
        manifest << "# on-screen albedo manifest (sorted by world extent desc)\n";
        manifest << "# the top row is the largest surface on screen — the sky-spanning wall\n";
        manifest << "# file  worldExtent  vsHash  matHash  onScreenInstances\n";
        char line[224];
        for (const auto& kv : rows) {
          snprintf(line, sizeof(line),
            "%016llx_albedo.dds  ext=%.0f  vs=0x%016llx  mat=0x%016llx  instances=%u\n",
            (unsigned long long) kv.first,
            kv.second.maxExtent,
            (unsigned long long) kv.second.vsHash,
            (unsigned long long) kv.second.matHash,
            kv.second.instances);
          manifest << line;
        }
      }
      // Name the winner directly in the log so no eyeballing is needed.
      if (!rows.empty()) {
        const auto& top = rows.front();
        const auto& ti = top.second;
        Logger::info(str::format(
          "[OnScreenAlbedoDump] LARGEST on-screen surface: ",
          dir, std::hex, top.first, "_albedo.dds", std::dec,
          " worldExtent=", ti.maxExtent,
          " objExtent=", ti.objExtent,
          " scaleMax=", ti.scaleMax,
          " o2wT=(", ti.o2wT.x, ",", ti.o2wT.y, ",", ti.o2wT.z, ")",
          std::hex, " vs=0x", ti.vsHash, " mat=0x", ti.matHash, std::dec,
          " | objExtent~normal + scaleMax huge => transform blow-up;"
          " objExtent huge => bad verts"));
      }
    }

    Logger::info(str::format(
      "[OnScreenAlbedoDump] dumped ", dumpedCount, " unique on-screen albedos + manifest.txt to ",
      dir, " (scanned ", instances.size(), " instances, ", byAlbedo.size(), " unique albedos)"));
  }

  void RtxContext::blitImageHelper(Rc<DxvkContext> ctx, const Rc<DxvkImage>& srcImage, const Rc<DxvkImage>& dstImage, VkFilter filter) {
    const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(dstImage->info().format);
    const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(srcImage->info().format);

    const VkImageSubresource dstSubresource = { dstFormatInfo->aspectMask, 0, 0 };
    const VkImageSubresource srcSubresource = { srcFormatInfo->aspectMask, 0, 0 };

    VkExtent3D srcExtent = srcImage->mipLevelExtent(srcSubresource.mipLevel);
    VkExtent3D dstExtent = dstImage->mipLevelExtent(dstSubresource.mipLevel);

    VkImageSubresourceLayers dstSubresourceLayers = {
      dstSubresource.aspectMask,
      dstSubresource.mipLevel,
      dstSubresource.arrayLayer, 1 };

    VkImageSubresourceLayers srcSubresourceLayers = {
      srcSubresource.aspectMask,
      srcSubresource.mipLevel,
      srcSubresource.arrayLayer, 1 };

    VkImageBlit blitInfo;

    blitInfo.dstSubresource = dstSubresourceLayers;
    blitInfo.srcSubresource = srcSubresourceLayers;

    blitInfo.dstOffsets[0] = VkOffset3D{ 0,                        0,                       0 };
    blitInfo.dstOffsets[1] = VkOffset3D{ int32_t(dstExtent.width),  int32_t(dstExtent.height),  1 };

    blitInfo.srcOffsets[0] = VkOffset3D{ 0,                          0,                         0 };
    blitInfo.srcOffsets[1] = VkOffset3D{ int32_t(srcExtent.width),    int32_t(srcExtent.height),    1 };

    VkComponentMapping swizzle = {
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    ctx->blitImage(dstImage, swizzle, srcImage, swizzle, blitInfo, filter);
  }

  RtxContext::RtxContext(const Rc<DxvkDevice>& device)
    : DxvkContext(device) {
    // Note: This may not be the best place to check for these features/properties, they ideally would be specified as
    // required upfront, but there's no good place to do that for this RTX extension (the D3D11 frontend does it before device
    // creation), so instead we just check for what is needed.
    // Note: When adding new extensions update DxvkAdapter::createDevice as it is what brings these features over.
    m_rayTracingSupported = (m_device->features().core.features.shaderInt16 &&
                             m_device->features().vulkan11Features.storageBuffer16BitAccess &&
                             m_device->features().vulkan11Features.uniformAndStorageBuffer16BitAccess &&
                             m_device->features().vulkan12Features.bufferDeviceAddress &&
                             m_device->features().vulkan12Features.descriptorIndexing &&
                             m_device->features().vulkan12Features.runtimeDescriptorArray &&
                             m_device->features().vulkan12Features.descriptorBindingPartiallyBound &&
                             m_device->features().vulkan12Features.shaderStorageBufferArrayNonUniformIndexing &&
                             m_device->features().vulkan12Features.shaderSampledImageArrayNonUniformIndexing &&
                             m_device->features().vulkan12Features.descriptorBindingVariableDescriptorCount &&
                             m_device->features().vulkan12Features.shaderInt8 &&
                             m_device->features().vulkan12Features.shaderFloat16 &&
                             m_device->features().vulkan12Features.uniformAndStorageBuffer8BitAccess &&
                             m_device->features().khrAccelerationStructureFeatures.accelerationStructure &&
                             m_device->features().khrRayQueryFeatures.rayQuery &&
                             m_device->features().khrDeviceRayTracingPipelineFeatures.rayTracingPipeline &&
                             m_device->extensions().khrShaderInt8Float16Types &&
                             m_device->properties().coreSubgroup.subgroupSize >= 1 &&
                             m_device->properties().coreSubgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT &&
                             m_device->properties().coreSubgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);

    m_dlssSupported = (m_device->extensions().nvxBinaryImport &&
                       m_device->extensions().nvxImageViewHandle &&
                       m_device->extensions().khrPushDescriptor);


    if (env::getEnvVar("DXVK_DUMP_SCREENSHOT_FRAME") != "") {
      m_screenshotFrameNum = stoul(env::getEnvVar("DXVK_DUMP_SCREENSHOT_FRAME"));
      m_screenshotFrameEnabled = true;
    }

    if (env::getEnvVar("DXVK_TERMINATE_APP_FRAME") != "") {
      m_terminateAppFrameNum = stoul(env::getEnvVar("DXVK_TERMINATE_APP_FRAME"));
      m_triggerDelayedTerminate = true;
    }

    m_prevRunningTime = std::chrono::steady_clock::now();

    checkOpacityMicromapSupport();
    checkShaderExecutionReorderingSupport();
    checkNeuralRadianceCacheSupport();
    reportCpuSimdSupport();

    GlobalTime::get().init(RtxOptions::timeDeltaBetweenFrames());

    // Initialize atmosphere system
    m_atmosphere = std::make_unique<RtxAtmosphere>(m_device.ptr());
  }

  RtxContext::~RtxContext() {
    getCommonObjects()->metaExporter().waitForAllExportsToComplete();

    if (m_screenshotFrameNum != -1 || m_terminateAppFrameNum != -1) {
      Metrics::serialize();
    }
  }

  SceneManager& RtxContext::getSceneManager() {
    return getCommonObjects()->getSceneManager();
  }
  Resources& RtxContext::getResourceManager() {
    return getCommonObjects()->getResources();
  }

  // Returns GPU idle time between calls to this in milliseconds
  float RtxContext::getGpuIdleTimeSinceLastCall() {
    uint64_t currGpuIdleTicks = m_device->getStatCounters().getCtr(DxvkStatCounter::GpuIdleTicks);
    uint64_t delta = currGpuIdleTicks - m_prevGpuIdleTicks;
    m_prevGpuIdleTicks = currGpuIdleTicks;

    return static_cast<float>(delta) * 0.001f; // to milliseconds
  }

  VkExtent3D RtxContext::setDownscaleExtent(const VkExtent3D& upscaleExtent) {
    ScopedCpuProfileZone();
    VkExtent3D downscaleExtent;
    if (shouldUseDLSS()) {
      DxvkDLSS& dlss = m_common->metaDLSS();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      dlss.setSetting(displaySize, RtxOptions::qualityDLSS(), renderSize);
      downscaleExtent.width = renderSize[0];
      downscaleExtent.height = renderSize[1];
      downscaleExtent.depth = 1;
    } else if (shouldUseRayReconstruction()) {
      DxvkRayReconstruction& rayReconstruction = m_common->metaRayReconstruction();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      rayReconstruction.setSettings(displaySize, RtxOptions::qualityDLSS(), renderSize);
      downscaleExtent.width = renderSize[0];
      downscaleExtent.height = renderSize[1];
      downscaleExtent.depth = 1;
    } else if (shouldUseXeSS()) {
      DxvkXeSS& xess = m_common->metaXeSS();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      xess.setSetting(displaySize, DxvkXeSS::XessOptions::preset(), renderSize);
      downscaleExtent.width = renderSize[0];
      downscaleExtent.height = renderSize[1];
      downscaleExtent.depth = 1;
      
      // XeSS: Apply recommended jitter sequence length if enabled
      if (DxvkXeSS::XessOptions::useRecommendedJitterSequenceLength() && xess.isActive()) {
        uint32_t recommendedJitterLength = xess.calcRecommendedJitterSequenceLength();
        uint32_t currentJitterLength = RtxOptions::cameraJitterSequenceLength();
      }
    } else if (shouldUseNIS() || shouldUseTAA()) {
      auto resolutionScale = RtxOptions::resolutionScale();
      downscaleExtent.width = uint32_t(std::roundf(upscaleExtent.width * resolutionScale));
      downscaleExtent.height = uint32_t(std::roundf(upscaleExtent.height * resolutionScale));
      downscaleExtent.depth = 1;
    } else {
      downscaleExtent = upscaleExtent;
    }
    downscaleExtent.width = std::max(downscaleExtent.width, 1u);
    downscaleExtent.height = std::max(downscaleExtent.height, 1u);

    return downscaleExtent;
  }

  void RtxContext::resetScreenResolution(const VkExtent3D& upscaleExtent) {
    // Calculate extents based on if DLSS is enabled or not
    const VkExtent3D downscaleExtent = setDownscaleExtent(upscaleExtent);

    // Resize the RT screen dependant buffers (if needed)
    getResourceManager().onResize(this, downscaleExtent, upscaleExtent);

    uint32_t renderSize[] = { downscaleExtent.width, downscaleExtent.height };
    uint32_t displaySize[] = { upscaleExtent.width, upscaleExtent.height };

    // Set resolution to cameras for jittering
    for (int i = 0; i < CameraType::Count; i++) {
      if (i == CameraType::Unknown) {
        continue;
      }
      RtCamera& camera = getSceneManager().getCameraManager().getCamera(static_cast<CameraType::Enum>(i));
      camera.setResolution(renderSize, displaySize);
    }

    // Note: Ensure the rendering resolution is not more than 2^14 - 1. This is due to assuming only
    // 14 of the 16 bits of an integer will be used for these pixel coordinates to pack additional data
    // into the free bits in memory payload structures on the GPU.
    assert((renderSize[0] < (1 << 14)) && (renderSize[1] < (1 << 14)));

    // With reloadTextureWhenResolutionChanged ON, textures will get reloaded when resolution is changed,
    // which may cause long wait when changing DLSS-RR or other upscalers' settings.
    // Therefore reloadTextureWhenResolutionChanged is set to OFF by default to improve performance. 
    if (RtxOptions::reloadTextureWhenResolutionChanged()) {
      getSceneManager().requestTextureVramFree();
    }
  }

  bool RtxContext::useRayReconstruction() const {
    return m_common->metaRayReconstruction().useRayReconstruction();
  }

  RtxContext::InternalUpscaler RtxContext::getCurrentFrameUpscaler() {
    if (shouldUseDLSS() && m_common->metaDLSS().isActive()) {
      return InternalUpscaler::DLSS;
    } else if (shouldUseRayReconstruction() && m_common->metaRayReconstruction().isActive()) {
      return InternalUpscaler::DLSS_RR;
    } else if (shouldUseXeSS() && m_common->metaXeSS().isActive()) {
      return InternalUpscaler::XeSS;
    } else if (shouldUseNIS()) {
      return InternalUpscaler::NIS;
    } else if (shouldUseTAA()) {
      return InternalUpscaler::TAAU;
    } else {
      return InternalUpscaler::None;
    }
  }

  // NV-DXVK [MtnRadiance]: async GPU->CPU readback of the primary-hit albedo / direct
  // diffuse radiance / indirect diffuse radiance / linear viewZ, sampled on a coarse grid
  // over the UPPER screen band and logged ONLY for distant-hit pixels (|viewZ| > 1e6 — the
  // ~6.5e6-unit 3D-skybox mountains). This tells us WHICH term is zero on the black tops:
  //   albedo ~0      -> material/albedo problem
  //   directLum ~0   -> direct light dead (shadow-ray precision at extreme ray origin)
  //   indirectLum ~0 -> NRC indirect dead (geometry far outside the 20000-unit NRC bounds)
  // Called pre-demodulate so radiance values are raw (not divided by albedo). Fires EVERY
  // frame (un-throttled) so it also captures the 1-frame flash; async decode mirrors the
  // [Coverage] FinalGrid readback. f= aligns with FinalGrid f=N.
  void RtxContext::captureMountainRadianceProbe(const Resources::RaytracingOutput& rtOutput) {
    if (!RtxOptions::logSurfaceCoverage())
      return;
    // NV-DXVK [Coverage PickRegion fast path]: this probe does 8 GPU->CPU
    // buffer readbacks + grid logging every frame. When the user only wants
    // a fast PickRegion stream, skip it — leaving it on would keep FPS pinned
    // even though the Coverage dump itself is trimmed to two lines.
    if (RtxOptions::coveragePickRegionOnly())
      return;

    Rc<DxvkImage> viewZImg    = rtOutput.m_primaryLinearViewZ.image;                                            // R32_SFLOAT
    Rc<DxvkImage> albedoImg   = rtOutput.m_primaryAlbedo.image;                                                 // A2B10G10R10_UNORM_PACK32
    // isAccessedByGPU=false on the aliased reads: with rtx.useDenoiser=False these buffers can be
    // re-aliased (e.g. "RTXDI Confidence 0" handed to "Secondary Cone Radius") before this read,
    // tripping the dev-build WAR-hazard assert (rtx_resources.cpp:325). false skips that bookkeeping
    // for our read-only diagnostic copy; data is valid in the normal (denoiser-on) path.
    Rc<DxvkImage> directImg   = rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read, false);     // R16G16B16A16_SFLOAT
    Rc<DxvkImage> indirectImg = rtOutput.m_primaryIndirectDiffuseRadiance.image(Resources::AccessType::Read, false);   // R16G16B16A16_SFLOAT
    Rc<DxvkImage> normalImg   = rtOutput.m_primaryWorldShadingNormal.image;                                            // R32_UINT (snorm2x16 signed-octahedral)
    Rc<DxvkImage> illumImg    = rtOutput.m_primaryRtxdiIlluminance[0].image(Resources::AccessType::Read, false);       // R16_SFLOAT (direct illuminance from RTXDI)
    Rc<DxvkImage> surfIdxImg  = rtOutput.m_sharedSurfaceIndex.image(Resources::AccessType::Read, false);               // R32_UINT (per-pixel surfaceIndex -> RtInstance)
    // NV-DXVK [MtnConf]: the RTXDI denoiser confidence (R16_SFLOAT) fed to NRD. Captured HERE
    // (after dispatchConfidence, before dispatchDenoise consumes+aliases it) — the composite
    // probe is too late, this buffer gets re-aliased. Theory: confidence~0 on the backdrop
    // mountain (lit only by sun/sky, no analytic light -> [LightContrib] nContributing=0) makes
    // NRD distrust and collapse the direct signal to black despite a valid lit input.
    Rc<DxvkImage> confImg     = rtOutput.getCurrentRtxdiConfidence().image(Resources::AccessType::Read, false);        // R16_SFLOAT
    // NV-DXVK [MtnMV]: the primary SCREEN-SPACE motion vector (R16G16_SFLOAT) — the value
    // NRD/RTXDI reproject with. If this reads ~0 on the streaking/black mountain while it
    // visibly sweeps across screen, the MV generation for isSubView reprojected 3D-skybox
    // geo is the root (gradient floors confidence -> black; same broken reproject = streak).
    // If it reads large/correct, the gradient is high for another reason (temporal reservoir
    // rejecting the per-frame world-pos drift), not the MV.
    Rc<DxvkImage> ssmvImg     = rtOutput.m_primaryScreenSpaceMotionVector.image;                                // R16G16_SFLOAT
    if (viewZImg == nullptr || albedoImg == nullptr || directImg == nullptr || indirectImg == nullptr ||
        normalImg == nullptr || illumImg == nullptr || surfIdxImg == nullptr || confImg == nullptr ||
        ssmvImg == nullptr)
      return;

    const VkExtent3D ext = viewZImg->info().extent;
    const uint32_t W = ext.width, H = ext.height;
    if (W == 0u || H == 0u)
      return;

    const VkDeviceSize size4  = VkDeviceSize(W) * H * 4u;  // viewZ, albedo, normal(R32_UINT)
    const VkDeviceSize size8  = VkDeviceSize(W) * H * 8u;  // RGBA16F radiance
    const VkDeviceSize size2  = VkDeviceSize(W) * H * 2u;  // R16F illuminance

    static Rc<DxvkBuffer> sBufViewZ, sBufAlbedo, sBufDirect, sBufIndirect, sBufNormal, sBufIllum, sBufSurfIdx, sBufConf, sBufSSMV;
    static Rc<sync::Fence> sFence;
    static uint64_t sFenceVal = 0;
    static std::vector<std::future<void>> sTasks;

    auto ensureBuf = [&](Rc<DxvkBuffer>& b, VkDeviceSize sz, const char* name) {
      if (b == nullptr || b->info().size < sz) {
        DxvkBufferCreateInfo ci {};
        ci.size   = sz;
        ci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
        ci.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        b = m_device->createBuffer(ci,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          DxvkMemoryStats::Category::RTXBuffer, name);
      }
    };
    ensureBuf(sBufViewZ,    size4, "MtnRadiance ViewZ");
    ensureBuf(sBufAlbedo,   size4, "MtnRadiance Albedo");
    ensureBuf(sBufDirect,   size8, "MtnRadiance Direct");
    ensureBuf(sBufIndirect, size8, "MtnRadiance Indirect");
    ensureBuf(sBufNormal,   size4, "MtnRadiance Normal");
    ensureBuf(sBufIllum,    size2, "MtnRadiance Illum");
    ensureBuf(sBufSurfIdx,  size4, "MtnRadiance SurfIdx");
    ensureBuf(sBufConf,     size2, "MtnRadiance Confidence");
    ensureBuf(sBufSSMV,     size4, "MtnRadiance SSMotionVec");  // R16G16_SFLOAT = 4 bytes/texel
    if (sFence == nullptr)
      sFence = new sync::Fence();

    VkImageSubresourceLayers subres {};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.layerCount = 1u;

    copyImageToBuffer(sBufViewZ,    0, 4u, 4u, viewZImg,    subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufAlbedo,   0, 4u, 4u, albedoImg,   subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufDirect,   0, 4u, 4u, directImg,   subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufIndirect, 0, 4u, 4u, indirectImg, subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufNormal,   0, 4u, 4u, normalImg,   subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufIllum,    0, 4u, 4u, illumImg,    subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufSurfIdx,  0, 4u, 4u, surfIdxImg,  subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufConf,     0, 2u, 2u, confImg,     subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufSSMV,     0, 4u, 4u, ssmvImg,     subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });

    emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t fv = ++sFenceVal;
    signal(sFence, fv);

    Rc<DxvkBuffer> bz = sBufViewZ, ba = sBufAlbedo, bd = sBufDirect, bi = sBufIndirect, bn = sBufNormal, bl = sBufIllum, bs = sBufSurfIdx, bcf = sBufConf, bmv = sBufSSMV;
    Rc<sync::Fence> fence = sFence;
    const uint32_t frameId = m_device->getCurrentFrameId();

    // NV-DXVK [ShipDir]: capture the MAIN camera orientation for THIS frame on the main
    // thread, so every async per-VS line below can be correlated with WHICH WAY the player
    // is looking. The "vanishing ship" is DIRECTIONAL (shows looking one way, gone the other,
    // at the same flight speed) — so the camera forward vector is the missing independent
    // variable. Grep [ShipDir] f=N alongside [SurfAlbedo]/[SurfTrack] f=N: if the hull VS's
    // flooredPct flips with camFwd sign while flight speed is constant, the denoiser reproject
    // is keyed on view direction (e.g. screen-space MV pointing off-screen one way), not speed.
    // Raw forward components are logged un-derived (no yaw math / axis assumptions).
    const RtCamera& shipDirCam = getSceneManager().getCamera();
    const bool shipDirValid = shipDirCam.isValid(frameId);
    const Vector3 camFwd = shipDirCam.getDirection(false);
    const Vector3 camPos = shipDirCam.getPosition(false);
    const float   camFov = shipDirCam.getFov();
    const float camFwdX = camFwd.x, camFwdY = camFwd.y, camFwdZ = camFwd.z;
    const float camPosX = camPos.x, camPosY = camPos.y, camPosZ = camPos.z;
    // NV-DXVK [ShipBox]: a screen rectangle (reuse rtx.surfaceCoveragePickRegion, normalized
    // [0,1] minX,minY,maxX,maxY) the user AIMS at the vanishing ship. Per frame we report, for
    // whatever surface is hit inside that box, its VS + mean hit distance (viewZ) + mean albedo.
    // This needs NO ship-by-VS/hash isolation: aim the box at the ship, look good then bad.
    //   vanished frame shows FAR meanViewZ (+ different VS) => ship geometry absent, world behind
    //     is the nearest hit (matches invalidPixels=0 / rays-always-hit).
    //   vanished frame shows SAME near meanViewZ but ~0 albedo => ship present but not shaded.
    const Vector4 shipBoxV = RtxOptions::surfaceCoveragePickRegion();
    const float shipBoxMinX = shipBoxV.x, shipBoxMinY = shipBoxV.y, shipBoxMaxX = shipBoxV.z, shipBoxMaxY = shipBoxV.w;

    // NV-DXVK [MtnRadiance] VS attribution. Snapshot the surfaceIndex -> VS
    // map on THIS (main) thread — getOrderedInstances() and the instance
    // table are not safe to touch from the async worker, and they change
    // every frame. Per slot we record the VS hash plus the three bits that
    // decide the backdrop's fate: isSubView (reprojected 3D-skybox geo),
    // isSubViewSkybox (got promoted to emissive), hasNormalBuffer (carries
    // vertex normals -> can be lit, vs no-normal painted backdrop). The
    // async task then attributes each sampled pixel to a VS so we can prove,
    // per black-ridge pixel, exactly which geometry it is and whether it can
    // actually be lit — instead of inferring it.
    // NV-DXVK: matHash = per-model Remix material hash. The shared VS 0x292b spans ~70
    // draws (hull/sky/world/weapon), so it CANNOT isolate the Widow. The material hash
    // is per-model — aim the pick box at the Widow and [ShipBox] reports its matHash,
    // then gate probes on that instead of the VS. See HullWorldRB matHash too.
    // matHash = per-model Remix material hash; texHash = albedo TEXTURE content hash
    // (getColorTexture().getImageHash()) — the canonical rtx.conf key, stable across
    // sessions. Either isolates the Widow far better than the shared VS 0x292b.
    // NV-DXVK [StudioModelHook]: isWidow carries the by-model Widow tag from
    // the cached DrawCallState (blas->input.isWidowModel) into the per-surface
    // snapshot — lets [ShipBox] report how many box pixels are the actual
    // Widow model (precise) instead of inferring from the shared VS 0x292b.
    // name = engine studiorender model path (StudioModelHook), value-copied so
    // it survives this deferred readback. Empty for non-studiorender surfaces
    // → tells us directly whether the box surface is a studio model at all.
    struct MtnSurfInfo { uint64_t vs; uint64_t matHash; uint64_t texHash; uint8_t isSubView; uint8_t isSubViewSkybox; uint8_t hasNormal; uint8_t isWidow; char name[64]; };
    std::vector<MtnSurfInfo> surfSnap;
    {
      const auto& reordered = getSceneManager().getAccelManager().getOrderedInstances();
      surfSnap.resize(reordered.size());
      for (size_t s = 0; s < reordered.size(); ++s) {
        const RtInstance* inst = reordered[s];
        const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
        if (blas != nullptr) {
          const auto& td = blas->input.getTransformData();
          const auto& md = blas->input.getMaterialData();
          const auto& ct = md.getColorTexture();
          surfSnap[s] = MtnSurfInfo {
            uint64_t(td.vertexShaderHash),
            uint64_t(md.getHash()),
            uint64_t(ct.isImageEmpty() ? XXH64_hash_t(0) : ct.getImageHash()),
            uint8_t(td.isSubView ? 1u : 0u),
            uint8_t(td.isSubViewSkybox ? 1u : 0u),
            uint8_t(blas->input.getGeometryData().normalBuffer.defined() ? 1u : 0u),
            uint8_t(blas->input.isWidowModel ? 1u : 0u) };  // name[] zero-init
          std::memcpy(surfSnap[s].name, blas->input.studioModelName, sizeof(surfSnap[s].name));
          surfSnap[s].name[sizeof(surfSnap[s].name) - 1] = '\0';
        } else {
          surfSnap[s] = MtnSurfInfo { 0ull, 0ull, 0ull, 0u, 0u, 0u, 0u };  // name[] zero-init
        }
      }
    }

    for (auto it = sTasks.begin(); it != sTasks.end();) {
      if (it->valid() && it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = sTasks.erase(it);
      else
        ++it;
    }

    sTasks.push_back(std::async(std::launch::async,
      [bz, ba, bd, bi, bn, bl, bs, bcf, bmv, fence, fv, W, H, frameId, surf = std::move(surfSnap),
       shipDirValid, camFwdX, camFwdY, camFwdZ, camPosX, camPosY, camPosZ, camFov,
       shipBoxMinX, shipBoxMinY, shipBoxMaxX, shipBoxMaxY]() {
        fence->wait(fv);
        const uint8_t* pz = reinterpret_cast<const uint8_t*>(bz->mapPtr(0));
        const uint8_t* pa = reinterpret_cast<const uint8_t*>(ba->mapPtr(0));
        const uint8_t* pd = reinterpret_cast<const uint8_t*>(bd->mapPtr(0));
        const uint8_t* pi = reinterpret_cast<const uint8_t*>(bi->mapPtr(0));
        const uint8_t* pn = reinterpret_cast<const uint8_t*>(bn->mapPtr(0));
        const uint8_t* pl = reinterpret_cast<const uint8_t*>(bl->mapPtr(0));
        const uint8_t* ps = reinterpret_cast<const uint8_t*>(bs->mapPtr(0));
        const uint8_t* pcf = reinterpret_cast<const uint8_t*>(bcf->mapPtr(0));
        const uint8_t* pmv = reinterpret_cast<const uint8_t*>(bmv->mapPtr(0));
        if (pz == nullptr || pa == nullptr || pd == nullptr || pi == nullptr || pn == nullptr || pl == nullptr || ps == nullptr || pcf == nullptr || pmv == nullptr)
          return;

        auto halfToFloat = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t expo = (h >> 10) & 0x1Fu;
          const uint32_t mant = h & 0x3FFu;
          uint32_t bits;
          if (expo == 0u) {
            if (mant == 0u) { bits = sign; }
            else {
              uint32_t e = 127u - 15u + 1u, m = mant;
              while ((m & 0x400u) == 0u) { m <<= 1; --e; }
              m &= 0x3FFu;
              bits = sign | (e << 23) | (m << 13);
            }
          } else if (expo == 0x1Fu) {
            bits = sign | 0x7F800000u | (mant << 13);
          } else {
            bits = sign | ((expo + (127u - 15u)) << 23) | (mant << 13);
          }
          float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        };

        // Decode R32_UINT primary world shading normal: snorm2x16 -> signed-octahedral
        // -> unit sphere direction (matches packing.slangh signedOctahedralToSphereDirection).
        // Lets us see if direct=0 is because the normal is degenerate / faces away (N·L<=0).
        auto decodeNormal = [](uint32_t packed, float& nx, float& ny, float& nz) {
          auto sn16 = [](uint16_t u) { return std::max(int16_t(u) / 32767.0f, -1.0f); };
          float px = sn16(uint16_t(packed & 0xFFFFu));
          float py = sn16(uint16_t((packed >> 16) & 0xFFFFu));
          float vx = px, vy = py, vz = 1.0f - std::fabs(px) - std::fabs(py);
          const float t = std::max(-vz, 0.0f);
          vx += (vx >= 0.0f) ? -t : t;
          vy += (vy >= 0.0f) ? -t : t;
          const float len = std::sqrt(vx*vx + vy*vy + vz*vz);
          if (len > 0.0f) { nx = vx/len; ny = vy/len; nz = vz/len; } else { nx = ny = nz = 0.0f; }
        };

        // [SurfAlbedo] per-VS aggregate over EVERY pixel (no viewZ gate, so it catches the
        // near-ish black peak the gated grid below misses). Reads GBuffer ALBEDO + DIRECT +
        // INDIRECT per pixel, bins by VS. For a black surface this distinguishes the two
        // root causes:
        //   meanAlbedo ~= 0            -> albedo texture not resolving in RT (texture/material)
        //   meanAlbedo > 0, direct ~= 0 -> albedo fine, surface is unlit (lighting/normal)
        {
          // NV-DXVK [SurfAlbedo] conf extension: also bin nrdConf (PRE-denoise confidence)
          // and pre-denoise direct/albedo over the WHOLE surface (not the sparse 8-pt grid,
          // which only lands on lit pixels and misses the black blob). Cross-referenced with
          // [SurfTrack] blackPct (POST-denoise): if a VS has meanDirect>0 (lit input) + high
          // flooredPct while blackPct is high, the black is the denoiser-confidence kill ACROSS
          // the surface. DECISIVE on STILL frames: if flooredPct drops as the camera stops but
          // blackPct stays frozen, the black is NOT the confidence floor -> structural (texture/
          // material), exactly matching "the smear never goes away". floored = conf < 0.11.
          // NV-DXVK [ShipDir] MV-direction extension: accumulate the SIGNED screen-space motion
          // vector (in render-res pixels) per VS, split into floored (conf<0.11) vs converged
          // pixels. Magnitude alone (mvBlk/mvLit in [SurfTrack]) cannot tell a directional bug
          // apart — a left-sweeping and a right-sweeping MV have identical magnitude. The SIGNED
          // mean reveals which way the reproject thinks the surface moved; offRight/offLeft/offUp/
          // offDown count pixels whose MV points the sample off the edge of the previous frame
          // (a reproject that lands outside last frame -> no history -> confidence floored ->
          // dark). If the hull's floored pixels show MV pointing off-screen ONLY when looking one
          // way, that is the directional vanish mechanism.
          struct SurfAlb { uint32_t pixels=0u; double sumAlb=0.0; double sumDir=0.0; double sumInd=0.0; double sumConf=0.0; uint32_t floored=0u;
                           double mvSx=0.0, mvSy=0.0, mvSmag=0.0;        // signed MV sums over ALL px of this VS
                           double mvSxF=0.0, mvSyF=0.0, mvSmagF=0.0;     // signed MV sums over FLOORED px only
                           uint32_t offEdge=0u, offEdgeF=0u;             // px whose MV points the reproject off-screen (all / floored)
                           int sv=-1; int svSky=-1; int nb=-1; };
          std::map<uint64_t, SurfAlb> albAgg;
          for (uint32_t yy = 0u; yy < H; ++yy) {
            for (uint32_t xx = 0u; xx < W; ++xx) {
              const VkDeviceSize idx = VkDeviceSize(yy) * W + xx;
              const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + idx * 4u);
              if (surfIdx == SURFACE_INDEX_INVALID || surfIdx >= surf.size())
                continue;
              const uint32_t apx = *reinterpret_cast<const uint32_t*>(pa + idx * 4u);
              const float ar = (apx & 0x3FFu) / 1023.0f;
              const float ag = ((apx >> 10) & 0x3FFu) / 1023.0f;
              const float ab = ((apx >> 20) & 0x3FFu) / 1023.0f;
              const float albLum = 0.2126f * ar + 0.7152f * ag + 0.0722f * ab;
              const uint16_t* dpx = reinterpret_cast<const uint16_t*>(pd + idx * 8u);
              const float dirLum = 0.2126f * halfToFloat(dpx[0]) + 0.7152f * halfToFloat(dpx[1]) + 0.0722f * halfToFloat(dpx[2]);
              const uint16_t* ipx = reinterpret_cast<const uint16_t*>(pi + idx * 8u);
              const float indLum = 0.2126f * halfToFloat(ipx[0]) + 0.7152f * halfToFloat(ipx[1]) + 0.0722f * halfToFloat(ipx[2]);
              const float conf = halfToFloat(*reinterpret_cast<const uint16_t*>(pcf + idx * 2u));
              // [ShipDir] signed screen-space MV at this pixel (R16G16_SFLOAT, render-res px).
              const uint16_t* mvp = reinterpret_cast<const uint16_t*>(pmv + idx * 4u);
              const float mvx = halfToFloat(mvp[0]);
              const float mvy = halfToFloat(mvp[1]);
              const float mvmag = std::sqrt(mvx * mvx + mvy * mvy);
              // Where this pixel reprojects FROM in the previous frame. If that lands outside the
              // frame, there is no history to reproject -> NRD floors confidence -> dark.
              const float prevX = float(xx) + mvx;
              const float prevY = float(yy) + mvy;
              const bool offEdge = (prevX < 0.f) || (prevX >= float(W)) || (prevY < 0.f) || (prevY >= float(H));
              const bool floored = (conf < 0.11f);
              SurfAlb& a = albAgg[surf[surfIdx].vs];
              if (a.pixels == 0u) { a.sv = surf[surfIdx].isSubView; a.svSky = surf[surfIdx].isSubViewSkybox; a.nb = surf[surfIdx].hasNormal; }
              a.pixels++;
              a.sumAlb += albLum;
              a.sumDir += dirLum;
              a.sumInd += indLum;
              a.sumConf += conf;
              a.mvSx += mvx; a.mvSy += mvy; a.mvSmag += mvmag;
              if (offEdge) a.offEdge++;
              if (floored) {
                a.floored++;
                a.mvSxF += mvx; a.mvSyF += mvy; a.mvSmagF += mvmag;
                if (offEdge) a.offEdgeF++;
              }
            }
          }
          // NV-DXVK [ShipDir]: one line per frame — which way the player is looking. Correlate
          // with the per-VS [SurfAlbedo] lines below (same f=). camValid=0 means the camera was
          // not valid this frame (menu/transition) so the direction is meaningless.
          Logger::info(str::format(
            "[ShipDir] f=", frameId,
            " camValid=", (shipDirValid ? 1 : 0),
            " camFwd=(", camFwdX, ",", camFwdY, ",", camFwdZ, ")",
            " camPos=(", camPosX, ",", camPosY, ",", camPosZ, ")",
            " fov=", camFov));
          for (const auto& kv : albAgg) {
            const SurfAlb& a = kv.second;
            if (a.pixels < 64u) continue;  // skip tiny/noisy surfaces
            const uint32_t flooredPct = (a.pixels > 0u) ? (100u * a.floored / a.pixels) : 0u;
            // [ShipDir] signed-MV means. mvX/mvY = mean reproject vector over the whole VS;
            // mvXfloor/mvYfloor = the same but only over floored (would-vanish) pixels. offEdge%
            // = fraction of pixels whose reproject lands off the previous frame (history miss).
            const uint32_t nFl = a.floored;
            const float mvMeanX  = static_cast<float>(a.mvSx / a.pixels);
            const float mvMeanY  = static_cast<float>(a.mvSy / a.pixels);
            const float mvMeanMag= static_cast<float>(a.mvSmag / a.pixels);
            const float mvMeanXF = (nFl > 0u) ? static_cast<float>(a.mvSxF / nFl) : 0.f;
            const float mvMeanYF = (nFl > 0u) ? static_cast<float>(a.mvSyF / nFl) : 0.f;
            const float mvMeanMagF=(nFl > 0u) ? static_cast<float>(a.mvSmagF / nFl) : 0.f;
            const uint32_t offEdgePct  = (a.pixels > 0u) ? (100u * a.offEdge / a.pixels) : 0u;
            const uint32_t offEdgeFPct = (nFl > 0u) ? (100u * a.offEdgeF / nFl) : 0u;
            Logger::info(str::format(
              "[SurfAlbedo] f=", frameId,
              " vs=0x", std::hex, kv.first, std::dec,
              " pixels=", a.pixels,
              " meanAlbedo=", static_cast<float>(a.sumAlb / a.pixels),
              " meanDirect=", static_cast<float>(a.sumDir / a.pixels),
              " meanIndirect=", static_cast<float>(a.sumInd / a.pixels),
              " meanConf=", static_cast<float>(a.sumConf / a.pixels),
              " flooredPct=", flooredPct,
              " mvX=", mvMeanX, " mvY=", mvMeanY, " mvMag=", mvMeanMag,
              " mvXfloor=", mvMeanXF, " mvYfloor=", mvMeanYF, " mvMagFloor=", mvMeanMagF,
              " offEdge=", offEdgePct, "%", " offEdgeFloor=", offEdgeFPct, "%",
              " sv=", a.sv, " svSky=", a.svSky, " nb=", a.nb));
          }
        }

        // NV-DXVK [ShipBox]: per-frame attribution of the user-aimed screen box. For every
        // pixel inside the normalized rect, bin by VS and accumulate hit distance (viewZ) and
        // albedo. Top VS by pixel count = what occupies the box this frame. Compare good vs bad
        // look direction: a jump in meanViewZ (and/or a different top VS) when the ship vanishes
        // means the ship geometry is absent and the world BEHIND it is the nearest hit.
        {
          const uint32_t bx0 = uint32_t(std::max(0.0f, std::min(1.0f, shipBoxMinX)) * (W - 1u));
          const uint32_t by0 = uint32_t(std::max(0.0f, std::min(1.0f, shipBoxMinY)) * (H - 1u));
          const uint32_t bx1 = uint32_t(std::max(0.0f, std::min(1.0f, shipBoxMaxX)) * (W - 1u));
          const uint32_t by1 = uint32_t(std::max(0.0f, std::min(1.0f, shipBoxMaxY)) * (H - 1u));
          struct BoxAgg { uint32_t pixels=0u; double sumVz=0.0; double sumAlb=0.0; double sumDir=0.0; int sv=-1; int svSky=-1; int nb=-1; uint64_t matHash=0ull; uint64_t texHash=0ull; int isWidow=-1; char name[64]={}; };
          std::map<uint64_t, BoxAgg> boxAgg;
          uint32_t boxMiss = 0u, boxTotal = 0u, boxWidowPx = 0u;
          for (uint32_t yy = by0; yy <= by1 && yy < H; ++yy) {
            for (uint32_t xx = bx0; xx <= bx1 && xx < W; ++xx) {
              const VkDeviceSize idx = VkDeviceSize(yy) * W + xx;
              boxTotal++;
              const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + idx * 4u);
              if (surfIdx == SURFACE_INDEX_INVALID || surfIdx >= surf.size()) { boxMiss++; continue; }
              const float vz = *reinterpret_cast<const float*>(pz + idx * 4u);
              const uint32_t apx = *reinterpret_cast<const uint32_t*>(pa + idx * 4u);
              const float albLum = 0.2126f * ((apx & 0x3FFu) / 1023.0f) + 0.7152f * (((apx >> 10) & 0x3FFu) / 1023.0f) + 0.0722f * (((apx >> 20) & 0x3FFu) / 1023.0f);
              const uint16_t* dpx = reinterpret_cast<const uint16_t*>(pd + idx * 8u);
              const float dirLum = 0.2126f * halfToFloat(dpx[0]) + 0.7152f * halfToFloat(dpx[1]) + 0.0722f * halfToFloat(dpx[2]);
              BoxAgg& a = boxAgg[surf[surfIdx].vs];
              if (a.pixels == 0u) { a.sv = surf[surfIdx].isSubView; a.svSky = surf[surfIdx].isSubViewSkybox; a.nb = surf[surfIdx].hasNormal; a.matHash = surf[surfIdx].matHash; a.texHash = surf[surfIdx].texHash; a.isWidow = surf[surfIdx].isWidow; std::memcpy(a.name, surf[surfIdx].name, sizeof(a.name)); a.name[sizeof(a.name)-1] = '\0'; }
              a.pixels++; a.sumVz += std::fabs(vz); a.sumAlb += albLum; a.sumDir += dirLum;
              if (surf[surfIdx].isWidow) boxWidowPx++;
            }
          }
          // Sort VS by pixel count, log the top 4 + the box header (camera dir + miss%).
          std::vector<std::pair<uint64_t, BoxAgg>> srt(boxAgg.begin(), boxAgg.end());
          std::sort(srt.begin(), srt.end(), [](const std::pair<uint64_t,BoxAgg>& a, const std::pair<uint64_t,BoxAgg>& b){ return a.second.pixels > b.second.pixels; });
          const uint32_t missPct = (boxTotal > 0u) ? (100u * boxMiss / boxTotal) : 0u;
          // widowPx = how many box pixels are the actual Widow engine model
          // (StudioModelHook tag). The direct "did the ship leave the box"
          // signal: aim the box at the ship, look good vs bad — widowPx
          // dropping to 0 (while meanViewZ jumps to the world behind) = the
          // Widow's own geometry is absent from those pixels.
          const uint32_t widowPct = (boxTotal > 0u) ? (100u * boxWidowPx / boxTotal) : 0u;
          Logger::info(str::format(
            "[ShipBox] f=", frameId, " camValid=", (shipDirValid ? 1 : 0),
            " camFwd=(", camFwdX, ",", camFwdY, ",", camFwdZ, ")",
            " boxPx=", boxTotal, " missPct=", missPct,
            " widowPx=", boxWidowPx, " widowPct=", widowPct,
            " vsCount=", uint32_t(srt.size())));
          for (uint32_t i = 0u; i < srt.size() && i < 4u; ++i) {
            const BoxAgg& a = srt[i].second;
            Logger::info(str::format(
              "[ShipBox]   #", i, " vs=0x", std::hex, srt[i].first,
              " mat=0x", a.matHash, " tex=0x", a.texHash, std::dec,
              " widow=", a.isWidow,
              " name=", (a.name[0] ? a.name : "(none)"),
              " pixels=", a.pixels,
              " meanViewZ=", static_cast<float>(a.sumVz / a.pixels),
              " meanAlbedo=", static_cast<float>(a.sumAlb / a.pixels),
              " meanDirect=", static_cast<float>(a.sumDir / a.pixels),
              " sv=", a.sv, " svSky=", a.svSky, " nb=", a.nb));
          }
        }

        // NV-DXVK [ShipScreen]: full-screen, camera-direction-INDEPENDENT census
        // of the dropship hull vs what replaces it when it "vanishes".
        //
        // Why this and not [ShipBox]: ShipBox is a FIXED screen rectangle, so a
        // yaw/pitch can slide it off the hull onto a legit open-bay sky region —
        // it can't tell "hull missing" from "box moved". This probe scans EVERY
        // pixel and buckets by what the primary ray actually resolved to, so the
        // hull-pixel count is meaningful regardless of where the camera points.
        //
        // Buckets (hull VS 0x292b is SHARED — it draws BOTH the world/hull AND
        // sky quads; the discriminator is isSubViewSkybox, per the live-frame
        // svSky check, so we split it):
        //   hullPx   : surf.vs==0x292b && isSubViewSkybox==0  (the actual hull/world)
        //   sky292Px : surf.vs==0x292b && isSubViewSkybox==1  (0x292b used as sky)
        //   skyboxPx : surf.vs==0x2a729f16                    (the far 3D-skybox VS)
        //   missPx   : surfIdx == INVALID                     (primary ray MISS → background)
        //   otherPx  : any other valid surface
        //
        // Decisive reading when the user reports the vanish:
        //   - hullPx COLLAPSES while missPx rises  => a real HOLE: no TLAS
        //     surface along those rays (the instance is in the scene per
        //     HullCensus but is NOT a hittable surface — look downstream of the
        //     instance list: BLAS build / surface assignment / ray mask).
        //   - hullPx collapses while skyboxPx/sky292Px rises => the sky is being
        //     drawn OVER the hull (overdraw / depth / sky-routing), not a cull.
        //   - hullPx stays high => the hull is actually on-screen and the
        //     "vanish" the user sees is elsewhere (re-aim the investigation).
        // hullBBox (normalized) localises where the remaining hull pixels are.
        {
          constexpr uint64_t kHullVs   = 0x292b6ba0d1854f28ull; // shared world/hull VS
          constexpr uint64_t kSkyboxVs = 0x2a729f16017d841bull; // far 3D-skybox VS
          uint32_t hullPx = 0u, sky292Px = 0u, skyboxPx = 0u, missPx = 0u, otherPx = 0u;
          uint32_t hMinX = W, hMinY = H, hMaxX = 0u, hMaxY = 0u; // hull-pixel screen bbox
          for (uint32_t yy = 0u; yy < H; ++yy) {
            for (uint32_t xx = 0u; xx < W; ++xx) {
              const VkDeviceSize idx = VkDeviceSize(yy) * W + xx;
              const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + idx * 4u);
              if (surfIdx == SURFACE_INDEX_INVALID || surfIdx >= surf.size()) {
                missPx++;
                continue;
              }
              const uint64_t vs = surf[surfIdx].vs;
              const bool svSky = (surf[surfIdx].isSubViewSkybox != 0);
              if (vs == kHullVs) {
                if (svSky) {
                  sky292Px++;
                } else {
                  hullPx++;
                  if (xx < hMinX) hMinX = xx;
                  if (yy < hMinY) hMinY = yy;
                  if (xx > hMaxX) hMaxX = xx;
                  if (yy > hMaxY) hMaxY = yy;
                }
              } else if (vs == kSkyboxVs) {
                skyboxPx++;
              } else {
                otherPx++;
              }
            }
          }
          const float invW = (W > 0u) ? 1.0f / float(W) : 0.f;
          const float invH = (H > 0u) ? 1.0f / float(H) : 0.f;
          const bool haveHull = (hullPx > 0u);
          Logger::info(str::format(
            "[ShipScreen] f=", frameId,
            " camValid=", (shipDirValid ? 1 : 0),
            " camFwd=(", camFwdX, ",", camFwdY, ",", camFwdZ, ")",
            " W=", W, " H=", H,
            " hullPx=", hullPx,
            " sky292Px=", sky292Px,
            " skyboxPx=", skyboxPx,
            " missPx=", missPx,
            " otherPx=", otherPx,
            " hullBBox=(",
              (haveHull ? float(hMinX) * invW : 0.f), ",",
              (haveHull ? float(hMinY) * invH : 0.f), ")-(",
              (haveHull ? float(hMaxX) * invW : 0.f), ",",
              (haveHull ? float(hMaxY) * invH : 0.f), ")"));
        }

        // Coarse grid over the UPPER ~60% of screen (horizon + distant mountains live here).
        constexpr uint32_t COLS = 24u, ROWS = 12u;
        const uint32_t bandH = (H * 60u) / 100u;
        for (uint32_t row = 0u; row < ROWS; ++row) {
          const uint32_t y = (row * bandH) / ROWS;
          for (uint32_t col = 0u; col < COLS; ++col) {
            const uint32_t x = ((col * 2u + 1u) * W) / (COLS * 2u);
            const float viewZ = *reinterpret_cast<const float*>(pz + (VkDeviceSize(y) * W + x) * 4u);
            // Distant-ish hits (gate lowered 1e6 -> 1e4): we must catch the backdrop
            // mountains BEFORE the flash (when they may sit at a normal/nearer distance
            // and are still lit) as well as after (pushed to ~6.5e6 + black), so we can
            // see whether the flash jumps the distance, zeroes albedo, or both. Excludes
            // only near props / no-hit (|viewZ| <= 1e4).
            if (!(std::fabs(viewZ) > 1.0e4f))
              continue;
            const uint32_t apx = *reinterpret_cast<const uint32_t*>(pa + (VkDeviceSize(y) * W + x) * 4u);
            const float ar = (apx & 0x3FFu) / 1023.0f;
            const float ag = ((apx >> 10) & 0x3FFu) / 1023.0f;
            const float ab = ((apx >> 20) & 0x3FFu) / 1023.0f;
            const uint16_t* dpx = reinterpret_cast<const uint16_t*>(pd + (VkDeviceSize(y) * W + x) * 8u);
            const uint16_t* ipx = reinterpret_cast<const uint16_t*>(pi + (VkDeviceSize(y) * W + x) * 8u);
            const float dr = halfToFloat(dpx[0]), dg = halfToFloat(dpx[1]), db = halfToFloat(dpx[2]);
            const float ir = halfToFloat(ipx[0]), ig = halfToFloat(ipx[1]), ib = halfToFloat(ipx[2]);
            const float directLum   = 0.2126f * dr + 0.7152f * dg + 0.0722f * db;
            const float indirectLum = 0.2126f * ir + 0.7152f * ig + 0.0722f * ib;
            const uint32_t npx = *reinterpret_cast<const uint32_t*>(pn + (VkDeviceSize(y) * W + x) * 4u);
            float nx, ny, nz; decodeNormal(npx, nx, ny, nz);
            const float rtxdiIllum = halfToFloat(*reinterpret_cast<const uint16_t*>(pl + (VkDeviceSize(y) * W + x) * 2u));
            const float nrdConf    = halfToFloat(*reinterpret_cast<const uint16_t*>(pcf + (VkDeviceSize(y) * W + x) * 2u));
            // Screen-space motion vector (R16G16_SFLOAT): .x,.y in (downscaled) pixels.
            const uint16_t* mvpx   = reinterpret_cast<const uint16_t*>(pmv + (VkDeviceSize(y) * W + x) * 4u);
            const float ssmvX = halfToFloat(mvpx[0]);
            const float ssmvY = halfToFloat(mvpx[1]);
            const float ssmvMag = std::sqrt(ssmvX * ssmvX + ssmvY * ssmvY);
            // VS attribution for THIS pixel: resolve surfaceIndex -> snapshot.
            const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + (VkDeviceSize(y) * W + x) * 4u);
            uint64_t vsHash = 0ull;
            int isSubView = -1, isSubViewSkybox = -1, hasNormal = -1;
            if (surfIdx != SURFACE_INDEX_INVALID && surfIdx < surf.size()) {
              vsHash          = surf[surfIdx].vs;
              isSubView       = surf[surfIdx].isSubView;
              isSubViewSkybox = surf[surfIdx].isSubViewSkybox;
              hasNormal       = surf[surfIdx].hasNormal;
            }
            Logger::info(str::format(
              "[MtnRadiance] f=", frameId, " x=", x, " y=", y,
              " viewZ=", viewZ,
              " albedo=(", ar, ",", ag, ",", ab, ")",
              " directLum=", directLum,
              " indirectLum=", indirectLum,
              " rtxdiIllum=", rtxdiIllum,
              " nrdConf=", nrdConf,
              " ssmv=(", ssmvX, ",", ssmvY, ") ssmvMag=", ssmvMag,
              " normal=(", nx, ",", ny, ",", nz, ")",
              " surfIdx=", surfIdx,
              " vs=0x", std::hex, vsHash, std::dec,
              " sv=", isSubView, " svSky=", isSubViewSkybox, " nb=", hasNormal));
          }
        }
      }));
  }

  // NV-DXVK [TonemapProbe]: read the tonemap INPUT (m_compositeOutput, HDR radiance)
  // and OUTPUT (m_finalOutput, post-operator display colour) at a sparse pixel grid
  // and log the in->out pairs. This makes the tonemap operator's effect directly
  // observable: HDR input values > 1 mapped into [0,1] proves the curve ran, and
  // the SAME input yielding DIFFERENT output when rtx.tonemap.operator changes
  // proves the selected operator is what transforms the pixels. Both buffers are
  // RGBA16F but may differ in resolution (upscaling), so sample by normalized pos.
  // Async readback (fence + worker) so it never stalls the frame.
  void RtxContext::captureTonemapProbe(const Resources::RaytracingOutput& rtOutput) {
    if (!RtxOptions::logSurfaceCoverage())
      return;

    Rc<DxvkImage> inImg  = rtOutput.m_compositeOutput.image(Resources::AccessType::Read); // R16G16B16A16_SFLOAT (HDR)
    Rc<DxvkImage> outImg = rtOutput.m_finalOutput.image(Resources::AccessType::Read);     // R16G16B16A16_SFLOAT ([0,1])
    if (inImg == nullptr || outImg == nullptr)
      return;

    const VkExtent3D inExt = inImg->info().extent, outExt = outImg->info().extent;
    const uint32_t Wi = inExt.width, Hi = inExt.height, Wo = outExt.width, Ho = outExt.height;
    if (Wi == 0u || Hi == 0u || Wo == 0u || Ho == 0u)
      return;

    const VkDeviceSize inSize  = VkDeviceSize(Wi) * Hi * 8u;
    const VkDeviceSize outSize = VkDeviceSize(Wo) * Ho * 8u;

    static Rc<DxvkBuffer> sBufIn, sBufOut;
    static Rc<sync::Fence> sFence;
    static uint64_t sFenceVal = 0;
    static std::vector<std::future<void>> sTasks;

    auto ensureBuf = [&](Rc<DxvkBuffer>& b, VkDeviceSize sz, const char* name) {
      if (b == nullptr || b->info().size < sz) {
        DxvkBufferCreateInfo ci {};
        ci.size   = sz;
        ci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
        ci.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        b = m_device->createBuffer(ci,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          DxvkMemoryStats::Category::RTXBuffer, name);
      }
    };
    ensureBuf(sBufIn,  inSize,  "TonemapProbe In");
    ensureBuf(sBufOut, outSize, "TonemapProbe Out");
    if (sFence == nullptr)
      sFence = new sync::Fence();

    VkImageSubresourceLayers subres {};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.layerCount = 1u;
    copyImageToBuffer(sBufIn,  0, 4u, 4u, inImg,  subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { Wi, Hi, 1u });
    copyImageToBuffer(sBufOut, 0, 4u, 4u, outImg, subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { Wo, Ho, 1u });

    emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t fv = ++sFenceVal;
    signal(sFence, fv);

    Rc<DxvkBuffer> bin = sBufIn, bout = sBufOut;
    Rc<sync::Fence> fence = sFence;
    const uint32_t frameId = m_device->getCurrentFrameId();
    const uint32_t op = static_cast<uint32_t>(DxvkToneMapping::tonemapOperator());

    for (auto it = sTasks.begin(); it != sTasks.end();) {
      if (it->valid() && it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = sTasks.erase(it);
      else
        ++it;
    }

    sTasks.push_back(std::async(std::launch::async,
      [bin, bout, fence, fv, Wi, Hi, Wo, Ho, frameId, op]() {
        fence->wait(fv);
        const uint8_t* pin  = reinterpret_cast<const uint8_t*>(bin->mapPtr(0));
        const uint8_t* pout = reinterpret_cast<const uint8_t*>(bout->mapPtr(0));
        if (pin == nullptr || pout == nullptr)
          return;

        auto halfToFloat = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t expo = (h >> 10) & 0x1Fu;
          const uint32_t mant = h & 0x3FFu;
          uint32_t bits;
          if (expo == 0u) {
            if (mant == 0u) { bits = sign; }
            else {
              uint32_t e = 127u - 15u + 1u, m = mant;
              while ((m & 0x400u) == 0u) { m <<= 1; --e; }
              m &= 0x3FFu;
              bits = sign | (e << 23) | (m << 13);
            }
          } else if (expo == 0x1Fu) {
            bits = sign | 0x7F800000u | (mant << 13);
          } else {
            bits = sign | ((expo + (127u - 15u)) << 23) | (mant << 13);
          }
          float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        };
        auto lum = [](float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; };

        // Sparse normalized grid so input and output (possibly different res) sample
        // the same screen points. Center band where the scene (not HUD) lives.
        const float us[] = { 0.25f, 0.50f, 0.75f };
        const float vs[] = { 0.35f, 0.55f };
        for (float v : vs) {
          for (float u : us) {
            const uint32_t xi = uint32_t(u * (Wi - 1u)), yi = uint32_t(v * (Hi - 1u));
            const uint32_t xo = uint32_t(u * (Wo - 1u)), yo = uint32_t(v * (Ho - 1u));
            const uint16_t* ip = reinterpret_cast<const uint16_t*>(pin  + (VkDeviceSize(yi) * Wi + xi) * 8u);
            const uint16_t* op16 = reinterpret_cast<const uint16_t*>(pout + (VkDeviceSize(yo) * Wo + xo) * 8u);
            const float ir = halfToFloat(ip[0]),  ig = halfToFloat(ip[1]),  ib = halfToFloat(ip[2]);
            const float orr = halfToFloat(op16[0]), og = halfToFloat(op16[1]), ob = halfToFloat(op16[2]);
            Logger::info(str::format(
              "[TonemapProbe] f=", frameId, " op=", op,
              " uv=(", u, ",", v, ")",
              " in=(", ir, ",", ig, ",", ib, ") lumIn=", lum(ir, ig, ib),
              " -> out=(", orr, ",", og, ",", ob, ") lumOut=", lum(orr, og, ob)));
          }
        }
      }));
  }

  // NV-DXVK [MtnComposite]: read the resolved on-screen radiance (m_compositeOutput,
  // RGBA16F) + per-pixel surfaceIndex on the same coarse grid as [MtnRadiance], and
  // attribute each pixel to a VS. Unlike the radiance probe (blind to emissive), this
  // sees the ACTUAL colour, so a backdrop VS whose final=(~0,~0,~0) is the black band.
  // Correlate with [MtnRadiance] lines by matching (f, x, y).
  void RtxContext::captureMountainCompositeProbe(const Resources::RaytracingOutput& rtOutput) {
    if (!RtxOptions::logSurfaceCoverage())
      return;

    Rc<DxvkImage> compImg    = rtOutput.m_compositeOutput.image(Resources::AccessType::Read, false);   // R16G16B16A16_SFLOAT
    Rc<DxvkImage> surfIdxImg = rtOutput.m_sharedSurfaceIndex.image(Resources::AccessType::Read, false); // R32_UINT
    // NV-DXVK [MtnFlags]: also read the per-pixel GeometryFlags (R16_UINT) so we can prove,
    // per black mountain pixel, WHY it is black — specifically whether the primary surface was
    // NOT selected for integration (bit0=primarySelectedIntegrationSurface). Alpha-tested
    // (not-fully-opaque) surfaces are the ones that can flip this bit off, which makes the
    // demodulate pass write zero primary direct radiance -> black after composite.
    Rc<DxvkImage> flagsImg   = rtOutput.m_sharedFlags.image;                                     // R16_UINT (GeometryFlags) — plain Resource, .image is a field
    // NV-DXVK [MtnAtten]: composite multiplies remodulated radiance by primaryAttenuation
    // (R32_UINT packed R11G11B10 UNORM). If a black mtn pixel has atten~0, that multiply is
    // what zeroes it; if atten~1, the kill is the denoiser (demodulate already ran, primSel=1,
    // albedo!=0). This is the elimination probe for the remaining two suspects.
    Rc<DxvkImage> attenImg   = rtOutput.m_primaryAttenuation.image;                              // R32_UINT (R11G11B10 UNORM)
    // NV-DXVK [MtnDenoise]: m_primaryDirectDiffuseRadiance is denoised IN PLACE (denoise writes
    // back into the same buffer the radiance probe read pre-denoise). This probe runs AFTER
    // dispatchDenoise, so reading it HERE gives the DENOISED direct-diffuse output. Cross-ref:
    // [SurfAlbedo] meanDirect (pre-denoise, lit ~0.17) vs this meanDenoised (post). If a black VS
    // has meanDenoised~0 / high denoZeroPct while its pre-denoise input was lit, the DENOISER
    // zeroed it (history reset / disocclusion) — NOT confidence, NOT albedo/atten (both ruled
    // out). If meanDenoised stays lit but composite is black, the kill is the composite multiply.
    // isAccessedByGPU=false on these THREE aliased reads: by composite-probe time their memory has
    // been re-aliased, so the default GPU-access path trips the dev-build WAR-hazard assert
    // (rtx_resources.cpp:325) — newly exposed when rtx.useDenoiser=False changes the alias ownership.
    // Passing false skips that bookkeeping for our read-only diagnostic copy. Data is valid while the
    // denoiser runs (matches blackPct); meaningless-but-harmless when the denoiser is off.
    Rc<DxvkImage> denoImg    = rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read, false);  // R16G16B16A16_SFLOAT (DENOISED here)
    // NV-DXVK [MtnNrdIn]: the three NRD inputs we have NOT yet ruled out, to discover WHY the
    // denoiser zeroes these lit pixels. Split per-VS into BLACK vs LIT pixels and compare:
    //   viewZ   (R32F)            — extreme/discontinuous viewZ breaks NRD edge-stopping
    //   normal  (A2B10G10R10)     — degenerate/missing normals (this VS has nb=0) collapse NRD weights
    //   disoccMix (R16F)          — NRD's own disocclusion signal; high on black px = NRD disoccluding
    // If BLACK px differ from LIT px of the SAME surface on any of these, that input is the kill driver.
    Rc<DxvkImage> vzImg      = rtOutput.m_primaryLinearViewZ.image;                                                                        // R32_SFLOAT
    Rc<DxvkImage> nrImg      = rtOutput.m_primaryVirtualWorldShadingNormalPerceptualRoughnessDenoising.image(Resources::AccessType::Read, false);  // A2B10G10R10 (NRD normal input)
    Rc<DxvkImage> dmImg      = rtOutput.m_primaryDisocclusionThresholdMix.image;                                                            // R16_SFLOAT
    // NV-DXVK [MtnNrdIn] the motion vector NRD ACTUALLY reprojects its temporal history with
    // (denoiseInput.motionVector = m_primaryVirtualMotionVector, NOT the screen-space MV the
    // earlier ssmv probe read). The "black clears in BLOCKS when something passes in front"
    // symptom = NRD accumulating bad history via wrong reprojection, reset on disocclusion. If
    // the VIRTUAL MV is large/wrong on the black px vs lit px, that is the reprojection driving
    // the accumulate-to-black. R16G16B16A16_SFLOAT.
    Rc<DxvkImage> mvImg      = rtOutput.m_primaryVirtualMotionVector.image(Resources::AccessType::Read, false);                             // R16G16B16A16_SFLOAT
    if (compImg == nullptr || surfIdxImg == nullptr || flagsImg == nullptr || attenImg == nullptr || denoImg == nullptr ||
        vzImg == nullptr || nrImg == nullptr || dmImg == nullptr || mvImg == nullptr)
      return;

    const VkExtent3D ext = compImg->info().extent;
    const uint32_t W = ext.width, H = ext.height;
    if (W == 0u || H == 0u)
      return;

    const VkDeviceSize size8 = VkDeviceSize(W) * H * 8u;  // RGBA16F
    const VkDeviceSize size4 = VkDeviceSize(W) * H * 4u;  // R32_UINT
    const VkDeviceSize size2 = VkDeviceSize(W) * H * 2u;  // R16_UINT (GeometryFlags)

    static Rc<DxvkBuffer> sBufComp, sBufSurf, sBufFlags, sBufAtten, sBufDeno, sBufVz, sBufNr, sBufDm, sBufMv;
    static Rc<sync::Fence> sFence;
    static uint64_t sFenceVal = 0;
    static std::vector<std::future<void>> sTasks;

    auto ensureBuf = [&](Rc<DxvkBuffer>& b, VkDeviceSize sz, const char* name) {
      if (b == nullptr || b->info().size < sz) {
        DxvkBufferCreateInfo ci {};
        ci.size   = sz;
        ci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
        ci.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        b = m_device->createBuffer(ci,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          DxvkMemoryStats::Category::RTXBuffer, name);
      }
    };
    ensureBuf(sBufComp, size8, "MtnComposite Color");
    ensureBuf(sBufSurf, size4, "MtnComposite SurfIdx");
    ensureBuf(sBufFlags, size2, "MtnComposite Flags");
    ensureBuf(sBufAtten, size4, "MtnComposite Atten");
    ensureBuf(sBufDeno, size8, "MtnComposite DenoisedDirect");  // RGBA16F
    ensureBuf(sBufVz,   size4, "MtnComposite ViewZ");           // R32F
    ensureBuf(sBufNr,   size4, "MtnComposite NormalRough");     // A2B10G10R10
    ensureBuf(sBufDm,   size2, "MtnComposite DisoccMix");       // R16F
    ensureBuf(sBufMv,   size8, "MtnComposite VirtualMV");       // RGBA16F (NRD reproject MV)
    if (sFence == nullptr)
      sFence = new sync::Fence();

    VkImageSubresourceLayers subres {};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.layerCount = 1u;
    copyImageToBuffer(sBufComp,  0, 4u, 4u, compImg,    subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufSurf,  0, 4u, 4u, surfIdxImg, subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufFlags, 0, 2u, 2u, flagsImg,   subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufAtten, 0, 4u, 4u, attenImg,   subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufDeno,  0, 4u, 4u, denoImg,    subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufVz,    0, 4u, 4u, vzImg,      subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufNr,    0, 4u, 4u, nrImg,      subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufDm,    0, 2u, 2u, dmImg,      subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });
    copyImageToBuffer(sBufMv,    0, 4u, 4u, mvImg,      subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });

    emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t fv = ++sFenceVal;
    signal(sFence, fv);

    Rc<DxvkBuffer> bc = sBufComp, bs = sBufSurf, bf = sBufFlags, ba = sBufAtten, bdn = sBufDeno, bvz = sBufVz, bnr = sBufNr, bdm = sBufDm, bmv = sBufMv;
    Rc<sync::Fence> fence = sFence;
    const uint32_t frameId = m_device->getCurrentFrameId();

    struct MtnSurfInfo2 { uint64_t vs; uint8_t isSubView; uint8_t isSubViewSkybox; uint8_t hasNormal; };
    std::vector<MtnSurfInfo2> surfSnap;
    {
      const auto& reordered = getSceneManager().getAccelManager().getOrderedInstances();
      surfSnap.resize(reordered.size());
      for (size_t s = 0; s < reordered.size(); ++s) {
        const RtInstance* inst = reordered[s];
        const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
        if (blas != nullptr) {
          const auto& td = blas->input.getTransformData();
          surfSnap[s] = MtnSurfInfo2 {
            uint64_t(td.vertexShaderHash),
            uint8_t(td.isSubView ? 1u : 0u),
            uint8_t(td.isSubViewSkybox ? 1u : 0u),
            uint8_t(blas->input.getGeometryData().normalBuffer.defined() ? 1u : 0u) };
        } else {
          surfSnap[s] = MtnSurfInfo2 { 0ull, 0u, 0u, 0u };
        }
      }
    }

    for (auto it = sTasks.begin(); it != sTasks.end();) {
      if (it->valid() && it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = sTasks.erase(it);
      else
        ++it;
    }

    sTasks.push_back(std::async(std::launch::async,
      [bc, bs, bf, ba, bdn, bvz, bnr, bdm, bmv, fence, fv, W, H, frameId, surf = std::move(surfSnap)]() {
        fence->wait(fv);
        const uint8_t* pc = reinterpret_cast<const uint8_t*>(bc->mapPtr(0));
        const uint8_t* ps = reinterpret_cast<const uint8_t*>(bs->mapPtr(0));
        const uint8_t* pf = reinterpret_cast<const uint8_t*>(bf->mapPtr(0));
        const uint8_t* pa = reinterpret_cast<const uint8_t*>(ba->mapPtr(0));
        const uint8_t* pdn = reinterpret_cast<const uint8_t*>(bdn->mapPtr(0));
        const uint8_t* pvz = reinterpret_cast<const uint8_t*>(bvz->mapPtr(0));
        const uint8_t* pnr = reinterpret_cast<const uint8_t*>(bnr->mapPtr(0));
        const uint8_t* pdm = reinterpret_cast<const uint8_t*>(bdm->mapPtr(0));
        const uint8_t* pmv = reinterpret_cast<const uint8_t*>(bmv->mapPtr(0));
        if (pc == nullptr || ps == nullptr || pf == nullptr || pa == nullptr || pdn == nullptr ||
            pvz == nullptr || pnr == nullptr || pdm == nullptr || pmv == nullptr)
          return;

        // Decode R11G11B10 UNORM (matches r11g11b10ToColor in packing.slangh).
        auto unpackAtten = [](uint32_t p) -> float {
          const float r = float(p & 0x7FFu) / 2047.0f;
          const float g = float((p >> 11) & 0x7FFu) / 2047.0f;
          const float b = float((p >> 22) & 0x3FFu) / 1023.0f;
          return 0.2126f * r + 0.7152f * g + 0.0722f * b;  // luminance of the throughput
        };

        auto halfToFloat = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t expo = (h >> 10) & 0x1Fu;
          const uint32_t mant = h & 0x3FFu;
          uint32_t bits;
          if (expo == 0u) {
            if (mant == 0u) { bits = sign; }
            else {
              uint32_t e = 127u - 15u + 1u, m = mant;
              while ((m & 0x400u) == 0u) { m <<= 1; --e; }
              m &= 0x3FFu;
              bits = sign | (e << 23) | (m << 13);
            }
          } else if (expo == 0x1Fu) {
            bits = sign | 0x7F800000u | (mant << 13);
          } else {
            bits = sign | ((expo + (127u - 15u)) << 23) | (mant << 13);
          }
          float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        };

        // [SurfTrack] per-VS aggregate over EVERY pixel of the composite this frame.
        // Full-resolution scan (not a sparse grid) so even a thin ridge band is fully
        // counted — and aggregated by SURFACE (vs), so camera motion swapping which geometry
        // is under a screen pixel can't confound it. Tracking one vs across frames: meanLum
        // trending to 0 / blackPct rising = that surface really going dark; steady = not.
        // NV-DXVK [MtnFlags]: per-VS, also break down the BLACK pixels by GeometryFlags so we
        // can prove the mechanism. primNotSel = primary surface NOT selected for integration
        // (bit0 off) -> demodulate writes 0 primary direct -> black. If a VS's black pixels are
        // ~all primNotSel, the alpha-test->not-fully-opaque->integration-flip->demodulate-zero
        // chain is confirmed (vs e.g. blackVM = viewmodel-in-shadow, a benign cause).
        struct SurfAgg { uint32_t pixels=0u; double sumLum=0.0; float minLum=0.f; float maxLum=0.f; uint32_t black=0u;
                         uint32_t primNotSel=0u; uint32_t blackPrimNotSel=0u; uint32_t blackSecMask=0u; uint32_t blackVM=0u;
                         uint32_t blackAtten0=0u; double sumAttenBlack=0.0;  // of black px: how many have atten~0, and mean atten
                         // NV-DXVK [MtnGhost] spatial footprint — to catch a DISPLACED / DUPLICATE draw
                         // (e.g. the translucent skybox-terrain "card" ghost). A normal compact surface
                         // fills most of its screen bbox (fill~1) and sits where its geometry belongs; a
                         // displaced ghost makes the SAME vs span two separated screen clusters -> huge
                         // bbox + low fill, and a card floating in the sky shows an anomalously HIGH
                         // centroid (cenY small). secMaskTot = how many of its px are secondary/translucent
                         // (the ghost panel reads see-through). bbox in render-res pixels.
                         uint32_t minX=0xFFFFFFFFu, maxX=0u, minY=0xFFFFFFFFu, maxY=0u; double sumX=0.0, sumY=0.0; uint32_t secMaskTot=0u;
                         // NV-DXVK [MtnDenoise] post-denoise direct-diffuse: mean + how many px the
                         // denoiser drove to ~0. denoZero high while pre-denoise input was lit =
                         // denoiser kill (history/disocclusion), the real black mechanism.
                         double sumDenoised=0.0; uint32_t denoZero=0u;
                         // NV-DXVK [MtnNrdIn] NRD inputs split BLACK vs LIT (lit = !black). viewZ,
                         // normal-component-sum (nsum: ~1.5 = encoded-zero/centered, ~0 = degenerate),
                         // disoccMix. nDegenBlack = black px with a fully-zero packed normal.
                         double vzB=0.0, vzL=0.0; double nlB=0.0, nlL=0.0; double dmB=0.0, dmL=0.0; uint32_t nDegenBlack=0u;
                         // virtual MV magnitude (NRD reproject input) split black vs lit
                         double mvB=0.0, mvL=0.0;
                         // NV-DXVK [ShipDir]: SIGNED virtual-MV components (x,y) split black vs lit.
                         // Magnitude alone hides a directional reproject bug; the signed mean shows
                         // which way NRD thinks the surface moved. Compare across the two viewing
                         // directions (grep with [ShipDir] f=N) at constant flight speed.
                         double mvBx=0.0, mvBy=0.0, mvLx=0.0, mvLy=0.0;
                         // [MtnViewZGrad] max |viewZ - neighbor viewZ| over the 4-neighbourhood — the
                         // exact quantity NRD edge-stopping compares. If gradBlk >> gradLit, the black
                         // pixels sit on viewZ discontinuities that reject all spatial neighbours ->
                         // no denoise support -> zeroed. Confirms the extreme-viewZ edge-stop break.
                         double gradB=0.0, gradL=0.0;
                         int sv=-1; int svSky=-1; int nb=-1; };
        std::map<uint64_t, SurfAgg> agg;
        for (uint32_t y = 0u; y < H; ++y) {
          for (uint32_t x = 0u; x < W; ++x) {
            const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + (VkDeviceSize(y) * W + x) * 4u);
            if (surfIdx == SURFACE_INDEX_INVALID || surfIdx >= surf.size())
              continue;
            const uint16_t* cpx = reinterpret_cast<const uint16_t*>(pc + (VkDeviceSize(y) * W + x) * 8u);
            const float lum = 0.2126f * halfToFloat(cpx[0]) + 0.7152f * halfToFloat(cpx[1]) + 0.0722f * halfToFloat(cpx[2]);
            SurfAgg& a = agg[surf[surfIdx].vs];
            if (a.pixels == 0u) { a.minLum = lum; a.maxLum = lum; a.sv = surf[surfIdx].isSubView; a.svSky = surf[surfIdx].isSubViewSkybox; a.nb = surf[surfIdx].hasNormal; }
            const uint16_t gf = *reinterpret_cast<const uint16_t*>(pf + (VkDeviceSize(y) * W + x) * 2u);
            const bool primSel = (gf & (1u << 0)) != 0u;  // primarySelectedIntegrationSurface
            const bool secMask = (gf & (1u << 1)) != 0u;  // secondarySurfaceMask
            const bool isVM    = (gf & (1u << 2)) != 0u;  // isViewModel
            const uint32_t attenPacked = *reinterpret_cast<const uint32_t*>(pa + (VkDeviceSize(y) * W + x) * 4u);
            const float attenLum = unpackAtten(attenPacked);
            // [MtnDenoise] post-denoise direct-diffuse luminance at this pixel
            const uint16_t* ddpx = reinterpret_cast<const uint16_t*>(pdn + (VkDeviceSize(y) * W + x) * 8u);
            const float denoLum = 0.2126f * halfToFloat(ddpx[0]) + 0.7152f * halfToFloat(ddpx[1]) + 0.0722f * halfToFloat(ddpx[2]);
            a.sumDenoised += denoLum;
            if (denoLum < 1.0e-3f) a.denoZero++;
            // [MtnNrdIn] NRD inputs at this pixel, split by black-vs-lit below
            const float vz = *reinterpret_cast<const float*>(pvz + (VkDeviceSize(y) * W + x) * 4u);
            const uint32_t nrp = *reinterpret_cast<const uint32_t*>(pnr + (VkDeviceSize(y) * W + x) * 4u);
            const float nsum = (nrp & 0x3FFu) / 1023.0f + ((nrp >> 10) & 0x3FFu) / 1023.0f + ((nrp >> 20) & 0x3FFu) / 1023.0f;
            const float dm = halfToFloat(*reinterpret_cast<const uint16_t*>(pdm + (VkDeviceSize(y) * W + x) * 2u));
            const uint16_t* mvpx = reinterpret_cast<const uint16_t*>(pmv + (VkDeviceSize(y) * W + x) * 8u);
            const float mvx = halfToFloat(mvpx[0]), mvy = halfToFloat(mvpx[1]), mvz = halfToFloat(mvpx[2]);
            const float mvMag = std::sqrt(mvx * mvx + mvy * mvy + mvz * mvz);
            // [MtnViewZGrad] max abs viewZ delta to the 4 neighbours (edge-stop discontinuity)
            const VkDeviceSize cIdx = VkDeviceSize(y) * W + x;
            float vzGrad = 0.f;
            if (x > 0u)     vzGrad = std::max(vzGrad, std::fabs(vz - *reinterpret_cast<const float*>(pvz + (cIdx - 1u) * 4u)));
            if (x + 1u < W) vzGrad = std::max(vzGrad, std::fabs(vz - *reinterpret_cast<const float*>(pvz + (cIdx + 1u) * 4u)));
            if (y > 0u)     vzGrad = std::max(vzGrad, std::fabs(vz - *reinterpret_cast<const float*>(pvz + (cIdx - W) * 4u)));
            if (y + 1u < H) vzGrad = std::max(vzGrad, std::fabs(vz - *reinterpret_cast<const float*>(pvz + (cIdx + W) * 4u)));
            const bool isBlackPx = (lum < 1.0e-3f);
            if (isBlackPx) { a.vzB += vz; a.nlB += nsum; a.dmB += dm; a.mvB += mvMag; a.mvBx += mvx; a.mvBy += mvy; a.gradB += vzGrad; if (nrp == 0u) a.nDegenBlack++; }
            else           { a.vzL += vz; a.nlL += nsum; a.dmL += dm; a.mvL += mvMag; a.mvLx += mvx; a.mvLy += mvy; a.gradL += vzGrad; }
            a.pixels++;
            a.sumLum += lum;
            a.minLum = std::min(a.minLum, lum);
            a.maxLum = std::max(a.maxLum, lum);
            // [MtnGhost] spatial footprint accumulation
            if (x < a.minX) a.minX = x;
            if (x > a.maxX) a.maxX = x;
            if (y < a.minY) a.minY = y;
            if (y > a.maxY) a.maxY = y;
            a.sumX += x; a.sumY += y;
            if (secMask) a.secMaskTot++;
            if (!primSel) a.primNotSel++;
            if (lum < 1.0e-3f) {
              a.black++;
              if (!primSel) a.blackPrimNotSel++;
              if (secMask)  a.blackSecMask++;
              if (isVM)     a.blackVM++;
              if (attenLum < 0.01f) a.blackAtten0++;
              a.sumAttenBlack += attenLum;
            }
          }
        }

        // Sparse [MtnComposite] grid — a few per-pixel spatial-reference lines (which screen
        // region a vs occupies) without spamming a line per pixel.
        constexpr uint32_t COLS = 24u, ROWS = 12u;
        const uint32_t bandH = (H * 60u) / 100u;
        for (uint32_t row = 0u; row < ROWS; ++row) {
          const uint32_t y = (row * bandH) / ROWS;
          for (uint32_t col = 0u; col < COLS; ++col) {
            const uint32_t x = ((col * 2u + 1u) * W) / (COLS * 2u);
            const uint16_t* cpx = reinterpret_cast<const uint16_t*>(pc + (VkDeviceSize(y) * W + x) * 8u);
            const float cr = halfToFloat(cpx[0]), cg = halfToFloat(cpx[1]), cb = halfToFloat(cpx[2]);
            const float lum = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;
            const uint32_t surfIdx = *reinterpret_cast<const uint32_t*>(ps + (VkDeviceSize(y) * W + x) * 4u);
            uint64_t vsHash = 0ull;
            int isSubView = -1, isSubViewSkybox = -1, hasNormal = -1;
            if (surfIdx != SURFACE_INDEX_INVALID && surfIdx < surf.size()) {
              vsHash          = surf[surfIdx].vs;
              isSubView       = surf[surfIdx].isSubView;
              isSubViewSkybox = surf[surfIdx].isSubViewSkybox;
              hasNormal       = surf[surfIdx].hasNormal;
            }
            // Only log surfaces (skip pure no-hit sky-clear pixels: surfIdx invalid AND near-black)
            if (surfIdx == SURFACE_INDEX_INVALID && lum < 1.0e-4f)
              continue;
            const uint16_t gf = *reinterpret_cast<const uint16_t*>(pf + (VkDeviceSize(y) * W + x) * 2u);
            const int primSel = (gf & (1u << 0)) ? 1 : 0;  // primarySelectedIntegrationSurface
            const int secMask = (gf & (1u << 1)) ? 1 : 0;  // secondarySurfaceMask
            const int isVM    = (gf & (1u << 2)) ? 1 : 0;  // isViewModel
            const int pstr    = (gf & (1u << 7)) ? 1 : 0;  // performPSTR
            const int psrr    = (gf & (1u << 8)) ? 1 : 0;  // performPSRR
            const uint32_t attenPacked = *reinterpret_cast<const uint32_t*>(pa + (VkDeviceSize(y) * W + x) * 4u);
            const float attenLum = unpackAtten(attenPacked);
            Logger::info(str::format(
              "[MtnComposite] f=", frameId, " x=", x, " y=", y,
              " final=(", cr, ",", cg, ",", cb, ")",
              " lum=", lum,
              (lum < 1.0e-3f ? " BLACK" : ""),
              " surfIdx=", surfIdx,
              " vs=0x", std::hex, vsHash, std::dec,
              " primSel=", primSel, " secMask=", secMask, " vm=", isVM, " pstr=", pstr, " psrr=", psrr,
              " atten=", attenLum,
              " sv=", isSubView, " svSky=", isSubViewSkybox, " nb=", hasNormal));
          }
        }

        // [SurfTrack] one line per VS this frame: mean/min/max luminance + % of its sampled
        // pixels that are black. Grep a single vs across frames to see if it's actually
        // going dark (meanLum trending to 0 / blackFrac rising) vs steady.
        for (const auto& kv : agg) {
          const SurfAgg& a = kv.second;
          const float mean = (a.pixels > 0u) ? static_cast<float>(a.sumLum / a.pixels) : 0.f;
          const uint32_t blackPct = (a.pixels > 0u) ? (100u * a.black / a.pixels) : 0u;
          // Of this VS's black pixels, what fraction is explained by primary-surface-not-selected
          // (the demodulate-zero mechanism)? ~100% => confirmed root cause for this surface.
          const uint32_t blackPrimNotSelPct = (a.black > 0u) ? (100u * a.blackPrimNotSel / a.black) : 0u;
          // blkAtten0% = fraction of black px whose primaryAttenuation~0 (=> composite multiply
          // is the killer). meanAttenBlk = mean throughput of black px. If ~1 with blkAtten0~0,
          // attenuation is innocent and the denoiser is zeroing the demodulated radiance.
          const uint32_t blackAtten0Pct = (a.black > 0u) ? (100u * a.blackAtten0 / a.black) : 0u;
          const float meanAttenBlk = (a.black > 0u) ? static_cast<float>(a.sumAttenBlack / a.black) : 0.f;
          // [MtnGhost] spatial footprint readout. bbox in render-res px; fill = pixels/bboxArea
          // (~1 compact blob, <<1 spread/two-cluster = displaced ghost); cen = centroid (a card
          // floating in the sky has small cenY); secMask = translucent/secondary px count.
          const uint32_t bw = (a.maxX >= a.minX) ? (a.maxX - a.minX + 1u) : 0u;
          const uint32_t bh = (a.maxY >= a.minY) ? (a.maxY - a.minY + 1u) : 0u;
          const float fill = (bw > 0u && bh > 0u) ? (float(a.pixels) / (float(bw) * float(bh))) : 0.f;
          const float cenX = (a.pixels > 0u) ? static_cast<float>(a.sumX / a.pixels) : 0.f;
          const float cenY = (a.pixels > 0u) ? static_cast<float>(a.sumY / a.pixels) : 0.f;
          // [MtnDenoise] post-denoise direct-diffuse: mean + % the denoiser zeroed
          const float meanDenoised = (a.pixels > 0u) ? static_cast<float>(a.sumDenoised / a.pixels) : 0.f;
          const uint32_t denoZeroPct = (a.pixels > 0u) ? (100u * a.denoZero / a.pixels) : 0u;
          // [MtnNrdIn] black-vs-lit NRD-input means. If black px differ from lit px of the SAME
          // surface, that input drives the denoiser kill. nb=number of black px, nl=number lit.
          const uint32_t nBlk = a.black;
          const uint32_t nLit = (a.pixels >= a.black) ? (a.pixels - a.black) : 0u;
          const float vzBlk  = (nBlk > 0u) ? static_cast<float>(a.vzB / nBlk) : 0.f;
          const float vzLit  = (nLit > 0u) ? static_cast<float>(a.vzL / nLit) : 0.f;
          const float nlBlk  = (nBlk > 0u) ? static_cast<float>(a.nlB / nBlk) : 0.f;
          const float nlLit  = (nLit > 0u) ? static_cast<float>(a.nlL / nLit) : 0.f;
          const float dmBlk  = (nBlk > 0u) ? static_cast<float>(a.dmB / nBlk) : 0.f;
          const float dmLit  = (nLit > 0u) ? static_cast<float>(a.dmL / nLit) : 0.f;
          const uint32_t degenBlkPct = (nBlk > 0u) ? (100u * a.nDegenBlack / nBlk) : 0u;
          const float mvBlk  = (nBlk > 0u) ? static_cast<float>(a.mvB / nBlk) : 0.f;
          const float mvLit  = (nLit > 0u) ? static_cast<float>(a.mvL / nLit) : 0.f;
          const float mvBlkX = (nBlk > 0u) ? static_cast<float>(a.mvBx / nBlk) : 0.f;
          const float mvBlkY = (nBlk > 0u) ? static_cast<float>(a.mvBy / nBlk) : 0.f;
          const float mvLitX = (nLit > 0u) ? static_cast<float>(a.mvLx / nLit) : 0.f;
          const float mvLitY = (nLit > 0u) ? static_cast<float>(a.mvLy / nLit) : 0.f;
          const float gradBlk = (nBlk > 0u) ? static_cast<float>(a.gradB / nBlk) : 0.f;
          const float gradLit = (nLit > 0u) ? static_cast<float>(a.gradL / nLit) : 0.f;
          Logger::info(str::format(
            "[SurfTrack] f=", frameId,
            " vs=0x", std::hex, kv.first, std::dec,
            " pixels=", a.pixels,
            " meanLum=", mean,
            " minLum=", a.minLum,
            " maxLum=", a.maxLum,
            " blackPct=", blackPct,
            " primNotSel=", a.primNotSel,
            " black=", a.black,
            " blkPrimNotSel=", a.blackPrimNotSel, "(", blackPrimNotSelPct, "%)",
            " blkSecMask=", a.blackSecMask,
            " blkVM=", a.blackVM,
            " blkAtten0=", a.blackAtten0, "(", blackAtten0Pct, "%)",
            " meanAttenBlk=", meanAttenBlk,
            " bbox=(", a.minX, ",", a.minY, ",", a.maxX, ",", a.maxY, ")",
            " cen=(", cenX, ",", cenY, ")",
            " fill=", fill,
            " secMask=", a.secMaskTot,
            " meanDenoised=", meanDenoised,
            " denoZeroPct=", denoZeroPct,
            " vzBlk=", vzBlk, " vzLit=", vzLit,
            " nlBlk=", nlBlk, " nlLit=", nlLit,
            " dmBlk=", dmBlk, " dmLit=", dmLit,
            " mvBlk=", mvBlk, " mvLit=", mvLit,
            " mvBlkXY=(", mvBlkX, ",", mvBlkY, ")", " mvLitXY=(", mvLitX, ",", mvLitY, ")",
            " gradBlk=", gradBlk, " gradLit=", gradLit,
            " degenBlk=", degenBlkPct, "%",
            " sv=", a.sv, " svSky=", a.svSky, " nb=", a.nb));
        }
      }));
  }

  // NV-DXVK: measure the average scene radiance (mean composite luminance) and feed it to
  // NRC's dynamicMaxExpectedRadiance, so NRC self-tunes its radiance scale to the current
  // scene instead of a hardcoded value. Throttled (brightness changes slowly), async
  // readback (never stalls the frame), EMA-smoothed. Must run AFTER dispatchComposite so
  // m_compositeOutput holds this frame's resolved radiance; NRC consumes the value at the
  // start of the next frame (1-frame lag, harmless).
  void RtxContext::updateNrcDynamicRadiance(const Resources::RaytracingOutput& rtOutput) {
    if (!NeuralRadianceCache::NrcOptions::dynamicMaxExpectedRadiance())
      return;

    constexpr uint32_t kPeriod = 8u;  // measure every Nth frame
    const uint32_t frameId = m_device->getCurrentFrameId();
    if ((frameId % kPeriod) != 0u)
      return;

    Rc<DxvkImage> compImg = rtOutput.m_compositeOutput.image(Resources::AccessType::Read);  // R16G16B16A16_SFLOAT
    if (compImg == nullptr)
      return;
    const VkExtent3D ext = compImg->info().extent;
    const uint32_t W = ext.width, H = ext.height;
    if (W == 0u || H == 0u)
      return;

    const VkDeviceSize size8 = VkDeviceSize(W) * H * 8u;

    static Rc<DxvkBuffer> sBuf;
    static Rc<sync::Fence> sFence;
    static uint64_t sFenceVal = 0;
    static std::vector<std::future<void>> sTasks;

    if (sBuf == nullptr || sBuf->info().size < size8) {
      DxvkBufferCreateInfo ci {};
      ci.size   = size8;
      ci.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      ci.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
      ci.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
      sBuf = m_device->createBuffer(ci,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "NRC DynRadiance");
    }
    if (sFence == nullptr)
      sFence = new sync::Fence();

    VkImageSubresourceLayers subres {};
    subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.layerCount = 1u;
    copyImageToBuffer(sBuf, 0, 4u, 4u, compImg, subres, VkOffset3D { 0, 0, 0 }, VkExtent3D { W, H, 1u });

    emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t fv = ++sFenceVal;
    signal(sFence, fv);

    Rc<DxvkBuffer> bc = sBuf;
    Rc<sync::Fence> fence = sFence;
    NeuralRadianceCache* nrc = &m_common->metaNeuralRadianceCache();

    for (auto it = sTasks.begin(); it != sTasks.end();) {
      if (it->valid() && it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = sTasks.erase(it);
      else
        ++it;
    }

    sTasks.push_back(std::async(std::launch::async, [bc, fence, fv, W, H, nrc]() {
      fence->wait(fv);
      const uint8_t* pc = reinterpret_cast<const uint8_t*>(bc->mapPtr(0));
      if (pc == nullptr)
        return;

      auto halfToFloat = [](uint16_t h) -> float {
        const uint32_t sign = (h & 0x8000u) << 16;
        const uint32_t expo = (h >> 10) & 0x1Fu;
        const uint32_t mant = h & 0x3FFu;
        uint32_t bits;
        if (expo == 0u) {
          if (mant == 0u) { bits = sign; }
          else {
            uint32_t e = 127u - 15u + 1u, m = mant;
            while ((m & 0x400u) == 0u) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
          }
        } else if (expo == 0x1Fu) {
          bits = sign | 0x7F800000u | (mant << 13);
        } else {
          bits = sign | ((expo + (127u - 15u)) << 23) | (mant << 13);
        }
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
      };

      // Sparse grid mean of nonzero luminance — matches how the static 250 was derived
      // ([MtnComposite] full-frame mean ~253). Bright sky is included on purpose: NRC wants
      // the average radiance of the bright daylight scene.
      constexpr uint32_t COLS = 96u, ROWS = 54u;
      double sum = 0.0;
      uint32_t n = 0u;
      for (uint32_t r = 0u; r < ROWS; ++r) {
        const uint32_t y = (r * H) / ROWS;
        for (uint32_t c = 0u; c < COLS; ++c) {
          const uint32_t x = (c * W) / COLS;
          const uint16_t* px = reinterpret_cast<const uint16_t*>(pc + (VkDeviceSize(y) * W + x) * 8u);
          const float lum = 0.2126f * halfToFloat(px[0]) + 0.7152f * halfToFloat(px[1]) + 0.0722f * halfToFloat(px[2]);
          if (std::isfinite(lum) && lum > 1.0e-4f) { sum += lum; ++n; }
        }
      }
      if (n == 0u)
        return;

      float mean = static_cast<float>(sum / n);
      mean = std::min(std::max(mean, 1.0f), 100000.0f);  // guard against a glitched frame

      // EMA so NRC sees a stable, slowly-tracking scale (first measurement adopted directly).
      const float prev = nrc->getMeasuredSceneAvgRadiance();
      constexpr float alpha = 0.25f;
      const float smoothed = (prev > 0.f) ? (prev + (mean - prev) * alpha) : mean;
      nrc->setMeasuredSceneAvgRadiance(smoothed);
    }));
  }

  VkExtent3D RtxContext::onFrameBegin(const VkExtent3D& upscaledExtent) {
    auto logRenderPassRaytraceModeRayQuery = [=](const char* renderPassName, auto mode) {
      switch (mode) {
      case decltype(mode)::RayQuery:
        Logger::info(str::format("RenderPass ", renderPassName, " Raytrace Mode: Ray Query (CS)"));
        break;
      case decltype(mode)::RayQueryRayGen:
        Logger::info(str::format("RenderPass ", renderPassName, " Raytrace Mode: Ray Query (RGS)"));
        break;
      default: 
        assert(false && "invalid RaytraceMode in logRenderPassRaytraceModeRayQuery");
        break;
      }
    };

    auto logRenderPassRaytraceMode = [=](const char* renderPassName, auto mode) {
      switch (mode) {
      case decltype(mode)::RayQuery:
      case decltype(mode)::RayQueryRayGen:
        logRenderPassRaytraceModeRayQuery(renderPassName, mode);
        break;
      case decltype(mode)::TraceRay:
        Logger::info(str::format("RenderPass ", renderPassName, " Raytrace Mode: Trace Ray (RGS)"));
        break;
      case decltype(mode)::Count:
        assert(false && "invalid RaytraceMode in logRenderPassRaytraceMode");
        break;
      }
    };

    // Log used raytracing mode
    static RenderPassGBufferRaytraceMode sPrevRenderPassGBufferRaytraceMode = RenderPassGBufferRaytraceMode::Count;
    static RenderPassIntegrateDirectRaytraceMode sPrevRenderPassIntegrateDirectRaytraceMode = RenderPassIntegrateDirectRaytraceMode::Count;
    static RenderPassIntegrateIndirectRaytraceMode sPrevRenderPassIntegrateIndirectRaytraceMode = RenderPassIntegrateIndirectRaytraceMode::Count;
    static UpscalerType sPrevUpscalerType = UpscalerType::None;

    if (sPrevRenderPassGBufferRaytraceMode != RtxOptions::renderPassGBufferRaytraceMode() ||
        sPrevRenderPassIntegrateDirectRaytraceMode != RtxOptions::renderPassIntegrateDirectRaytraceMode() ||
        sPrevRenderPassIntegrateIndirectRaytraceMode != RtxOptions::renderPassIntegrateIndirectRaytraceMode() ||
        sPrevUpscalerType != RtxOptions::upscalerType()) {

      sPrevRenderPassGBufferRaytraceMode = RtxOptions::renderPassGBufferRaytraceMode();
      sPrevRenderPassIntegrateDirectRaytraceMode = RtxOptions::renderPassIntegrateDirectRaytraceMode();
      sPrevRenderPassIntegrateIndirectRaytraceMode = RtxOptions::renderPassIntegrateIndirectRaytraceMode();
      sPrevUpscalerType = RtxOptions::upscalerType();

      logRenderPassRaytraceMode("GBuffer", RtxOptions::renderPassGBufferRaytraceMode());
      logRenderPassRaytraceModeRayQuery("Integrate Direct", RtxOptions::renderPassIntegrateDirectRaytraceMode());
      logRenderPassRaytraceMode("Integrate Indirect", RtxOptions::renderPassIntegrateIndirectRaytraceMode());

      m_resetHistory = true;
    }

    // Calculate extents based on if DLSS is enabled or not
    VkExtent3D downscaledExtent = setDownscaleExtent(upscaledExtent);

    if (!getResourceManager().validateRaytracingOutput(downscaledExtent, upscaledExtent)) {
      Logger::debug("Raytracing output resources were not available to use this frame, so we must re-create inline.");

      resetScreenResolution(upscaledExtent);
    }

    const RtCamera& mainCamera = getSceneManager().getCamera();

    // Call onFrameBegin callbacks for RtxPases
    // Note: this needs to be called after resetScreenResolution() call in a frame
    // since an RtxPass may alias some of its resources with the ones created in createRaytracingOutput()
    getResourceManager().onFrameBegin(this, getCommonObjects()->getTextureManager(), getSceneManager(), downscaledExtent,
                                      upscaledExtent, m_resetHistory, mainCamera.isCameraCut());

    // NV-DXVK [ResetHistoryTrigger]: m_resetHistory is what actually blanks the
    // denoiser/integrate history -> path-traced GEOMETRY goes near-black for one
    // frame while the rasterized sky survives (the "scene darkens, sky fine" flash).
    // Snapshot the contributing sources so a capture names the cause. Event-driven
    // (logs only on a reset), ~zero volume, survives with the per-draw firehose off.
    // f= aligns with [Coverage] FinalGrid f=N and [NrcResetTrigger] f=N.
    const bool rhEntry      = m_resetHistory;                                              // set upstream (raytrace-mode change @onFrameBegin top, etc.)
    const bool rhModeChange = (RtxOptions::integrateIndirectMode() != m_prevIntegrateIndirectMode);
    const bool rhCameraCut  = mainCamera.isCameraCut();

    // Force history reset on integrate indirect mode change to discard incompatible history
    if (RtxOptions::integrateIndirectMode() != m_prevIntegrateIndirectMode) {
      m_resetHistory = true;
      m_prevIntegrateIndirectMode = RtxOptions::integrateIndirectMode();
    }

    const bool rhNrcReset = (RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::NeuralRadianceCache &&
                             m_common->metaNeuralRadianceCache().isResettingHistory());
    if (rhNrcReset) {
      m_resetHistory = true;
    }

    // NV-DXVK [diag] rtx.forceResetDenoiserHistory: blank NRD temporal history EVERY frame.
    // This keeps the per-frame SPATIAL denoise but kills TEMPORAL accumulation. Diagnostic for
    // the black 3D-skybox mountains: every per-frame NRD input (MV, normal, viewZ, disoccMix,
    // viewZ-gradient) was proven equal black-vs-lit, yet the denoiser zeroes a lit input and the
    // black grows-while-held / clears-on-occlusion — all signatures of bad TEMPORAL history, not a
    // per-frame input. If the blob VANISHES with this on, temporal accumulation IS the black
    // (and the fix is to reset/relax temporal for the extreme-viewZ subview skybox). Global +
    // noisy (no temporal denoise anywhere) — diagnostic only.
    const bool rhForceDiag = RtxOptions::forceResetDenoiserHistory();
    if (rhForceDiag) {
      m_resetHistory = true;
    }

    if (m_resetHistory) {
      Logger::info(str::format(
        "[ResetHistoryTrigger] f=", m_device->getCurrentFrameId(),
        " entryAlreadySet=", (rhEntry ? 1 : 0),
        " indirectModeChange=", (rhModeChange ? 1 : 0),
        " nrcResettingHistory=", (rhNrcReset ? 1 : 0),
        " forceResetDiag=", (rhForceDiag ? 1 : 0),
        " cameraCut=", (rhCameraCut ? 1 : 0)));
    }

    // Release resources when switching upscalers
    m_currentUpscaler = getCurrentFrameUpscaler();
    if (m_currentUpscaler != m_previousUpscaler) {
      // Need to wait before the previous frame is executed.
      getDevice()->waitForIdle();

      // Release resources
      if (m_previousUpscaler == InternalUpscaler::DLSS_RR) {
        DxvkRayReconstruction& rayReconstruction = m_common->metaRayReconstruction();
        rayReconstruction.release();
      } else if (m_previousUpscaler == InternalUpscaler::DLSS) {
        DxvkDLSS& dlss = m_common->metaDLSS();
        dlss.release();
      }
    }

    return downscaledExtent;
  }

  void RtxContext::blitPostTonemapScratchToCompositeOut(Rc<DxvkImage> compositeOut) {
    static uint64_t sBlitCount = 0;
    static uint64_t sNoContent = 0;
    ++sBlitCount;
    if (!m_rtFinalPostTonemapScratchHasContent || m_rtFinalPostTonemapScratch == nullptr || compositeOut == nullptr) {
      ++sNoContent;
      if (sBlitCount <= 3 || (sBlitCount & 0xFF) == 0) {
        Logger::info(str::format(
          "[CompositeOut v4] blit skipped count=", sBlitCount,
          " noContent=", sNoContent,
          " hasContent=", (m_rtFinalPostTonemapScratchHasContent ? 1 : 0),
          " scratchNull=", (m_rtFinalPostTonemapScratch == nullptr ? 1 : 0),
          " dstNull=", (compositeOut == nullptr ? 1 : 0)));
      }
      return;
    }
    blitImageHelper(this, m_rtFinalPostTonemapScratch, compositeOut, VK_FILTER_LINEAR);
    if (sBlitCount <= 3 || (sBlitCount & 0xFF) == 0) {
      const auto& si = m_rtFinalPostTonemapScratch->info();
      const auto& di = compositeOut->info();
      Logger::info(str::format(
        "[CompositeOut v4] blit #", sBlitCount,
        " src=", si.extent.width, "x", si.extent.height, "/fmt=", (int)si.format,
        " dst=", di.extent.width, "x", di.extent.height, "/fmt=", (int)di.format));
    }
  }

  // Hooked into presentImage (same place HUD rendering is)
  void RtxContext::injectRTX(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage, bool skipBackbufferBlit) {
    ScopedCpuProfileZone();

    // [SkyTrace.matteClear] removed — per-frame white-clear of the matte
    // was a diagnostic to test whether the matte content was the yellow-sky
    // source. The [SkyTrace.matteContent] readback combined with
    // [SkyTrace.primaryMiss]=0 proved the matte is essentially never sampled
    // for visible-sky pixels in our test scenes (those pixels go through the
    // world-hit IBL path which samples the SkyProbe cubemap, not the matte).
    // Force-clearing the matte was nuking TF2's actual sky output for the
    // few frames/pixels that DO take the primary-miss branch, so it's gone.
    // [mainCamSnapshot] Log the RtCamera::Main world position the path
    // tracer is about to render from. This is the camera *as the path
    // tracer sees it*, after all upstream extraction / cache / snap
    // logic. The frozen-screen-with-correct-camera state shows a
    // specific value here on the frame the freeze starts; comparing it
    // to live-render values from later frames is how we find what's
    // changing.
    {
      auto& cm = getSceneManager().getCameraManager();
      const auto& mainCam = cm.getCamera(CameraType::Main);
      const uint32_t fid = m_device->getCurrentFrameId();
      const bool isValid = mainCam.isValid(fid);
      const Vector3 pos = isValid ? mainCam.getPosition(false /*non-freecam*/)
                                  : Vector3{ 0.f, 0.f, 0.f };
      // Throttled: log only on Z-change > 1 unit to avoid spamming, plus
      // every 60th frame as a heartbeat so we always have an anchor.
      static float sLastZ = 1e30f;
      static uint32_t sLastFid = 0;
      const bool zChanged = std::abs(pos.z - sLastZ) > 1.0f;
      const bool heartbeat = (fid != sLastFid) && ((fid % 60) == 0);
      if (isValid && (zChanged || heartbeat)) {
        sLastZ = pos.z;
        sLastFid = fid;
        Logger::info(str::format(
          "[mainCamSnapshot] frame=", fid,
          " pos=(", pos.x, ",", pos.y, ",", pos.z, ")",
          " valid=1"));
      }
    }
    // NV-DXVK [BlitDiag]: total injectRTX wall time. Gated off now that
    // the skip-blit stall is understood. Set RTX_BLIT_DIAG=1 to re-enable.
    static const bool s_blitDiagEnabledOuter = []() {
      const char* v = std::getenv("RTX_BLIT_DIAG");
      return v != nullptr && v[0] == '1';
    }();
    const auto tInjectStart = s_blitDiagEnabledOuter ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // [Perf.Frame] wall-time chrono — runs UNGATED so menu frames are captured too. The per-stage
    // breakdown only fills when the RT branch fires (set inside the tlasReady block); when the menu
    // skips RT we still log totalInjectUs so the menu vs. gameplay cost is comparable. Throttled
    // to once every 5 seconds wallclock per DLL.
    using PerfClock = std::chrono::steady_clock;
    const auto tPerfFrameStart = PerfClock::now();

    // [OptionSnapshot] One-shot per-DLL dump of the RtxOption values that gate matrix extraction
    // and camera classification. The BAD-camera regression came from these values silently
    // diverging between d3d11.dll and dxvk.dll because d3d11.dll's RtxOptionImpl::isInitialized()
    // was false → setDeferred() no-opped → option value differed from what dxvk.dll saw. When
    // re-attempting any kind of d3d11.dll-side option population, capture this log in BOTH a known-
    // good run (camera works) and a known-bad run (camera invalid) and diff. Any option that
    // changes between runs is a candidate for the matrix-extraction code path that implicitly
    // assumes the no-op default.
    {
      static bool sLogged = false;
      if (!sLogged) {
        sLogged = true;
        Logger::info(str::format("[OptionSnapshot] one-shot per-DLL gating-option dump",
                                 " fusedWorldViewMode=", (int)RtxOptions::fusedWorldViewMode(),
                                 " leftHandedCoordinateSystem=", RtxOptions::leftHandedCoordinateSystem() ? 1 : 0,
                                 " zUp=", RtxOptions::zUp() ? 1 : 0,
                                 " useCBufferWorldMatrices=", RtxOptions::useCBufferWorldMatrices() ? 1 : 0,
                                 " useBuffersDirectly=", RtxOptions::useBuffersDirectly() ? 1 : 0,
                                 " enableRaytracing=", RtxOptions::enableRaytracing() ? 1 : 0,
                                 " enableNearPlaneOverride=", RtxOptions::enableNearPlaneOverride() ? 1 : 0,
                                 " shakeCamera=", RtxOptions::shakeCamera() ? 1 : 0));
      }
    }
    struct PerfFrameStageTimes {
      bool   rtBranchRan = false;
      // Pre-RT-branch: from injectRTX entry to onFrameBegin call
      int64_t entryToOnFrameBeginUs = 0;
      // Subdivided buckets within entryToOnFrameBegin (sum should ≈ that total)
      int64_t entry_preCommitGfxUs    = 0; // entry → commitGraphicsState
      int64_t entry_commitGfxUs       = 0; // commitGraphicsState<true,false>
      int64_t entry_postCommitGfxUs   = 0; // options + DLSS fallback + ShaderManager update
      int64_t entry_hotReloadUs       = 0; // processAllHotReloadRequests (TextureManager)
      int64_t entry_tailToBranchUs    = 0; // rest up to the RT-branch tlasReady fork
      int64_t onFrameBeginUs = 0;
      int64_t prepUs = 0;
      int64_t volumetricsUs = 0;
      int64_t pathTracingUs = 0;
      int64_t nrcUs = 0;
      int64_t rtxdiUs = 0;
      int64_t restirUs = 0;
      int64_t demodulateUs = 0;
      int64_t denoiseUs = 0;
      int64_t compositeUs = 0;
      int64_t debugViewUs = 0;
      // Post-debug-view to end of injectRTX: object picking, upscaler, blit, etc.
      int64_t postCompositeUs = 0;
      int64_t upscalerUs = 0;
      int64_t finalBlitUs = 0;
      int64_t endFrameUs = 0;
    } perfFrame;
    // Capture the sticky flag as it was on ENTRY so the final diag log
    // can report it (the per-frame reset inside the blit block will
    // have cleared it by the time we log).
    const bool stickySkipAtEntry = m_skipBackbufferBlitThisFrame;
#ifdef REMIX_DEVELOPMENT
    m_currentPassStage = RtxFramePassStage::FrameBegin;
#endif

    if (RtxOptions::enableBreakIntoDebuggerOnPressingB() && ImGUI::checkHotkeyState({VirtualKey{ 'B' }}, true)) {
      while (!::IsDebuggerPresent()) {
        ::Sleep(100);
      }
      __debugbreak();
    }

#ifdef REMIX_DEVELOPMENT
    // Crash Hotkey Feature: When armed via the Development tab checkbox, pressing the crash hotkey
    // triggers a deliberate null pointer dereference crash. This is useful for testing crash handling,
    // crash dumps, and crash reporting systems.
    {
      static bool crashHotkeyStartupLogged = false;
      if (!crashHotkeyStartupLogged && RtxOptions::enableCrashHotkey()) {
        const auto crashHotkeyStr = buildKeyBindDescriptorString(RtxOptions::crashHotkey());
        Logger::warn(str::format("Crash hotkey is ARMED at startup (via config/environment) - press ", crashHotkeyStr, " to trigger crash"));
        crashHotkeyStartupLogged = true;
      }
      
      if (RtxOptions::enableCrashHotkey() && ImGUI::checkHotkeyState(RtxOptions::crashHotkey(), false)) {
        const auto crashHotkeyStr = buildKeyBindDescriptorString(RtxOptions::crashHotkey());
        Logger::err(str::format("Deliberate crash triggered via crash hotkey (", crashHotkeyStr, ")"));
        // Trigger a null pointer dereference to cause a crash
        volatile int* nullPtr = nullptr;
        *nullPtr = 0xDEAD;
      }
    }
#endif

    // [Perf.Frame] entry-subdivision: time commitGraphicsState alone — it
    // can be heavy when state has changed substantially.
    const auto tEntryBeforeCommitGfx = PerfClock::now();
    perfFrame.entry_preCommitGfxUs = std::chrono::duration_cast<std::chrono::microseconds>(tEntryBeforeCommitGfx - tPerfFrameStart).count();
    commitGraphicsState<true, false>();
    const auto tEntryAfterCommitGfx = PerfClock::now();
    perfFrame.entry_commitGfxUs = std::chrono::duration_cast<std::chrono::microseconds>(tEntryAfterCommitGfx - tEntryBeforeCommitGfx).count();

    auto common = getCommonObjects();
    const auto isRaytracingEnabled = RtxOptions::enableRaytracing();
    const auto asyncShaderCompilationActive = RtxOptions::Shader::enableAsyncCompilation() && common->pipelineManager().remixShaderCompilationCount() > 0;

    // Determine and set present throttle delay
    // Note: This must be done before the early out returns below which is why some logic here is redundant (e.g. checking if ray tracing is supported again)
    // just to ensure the present throttle delay is always being set properly.

    const auto requestedPresentThrottleDelay = RtxOptions::enablePresentThrottle() ? RtxOptions::presentThrottleDelay() : 0;
    std::uint32_t requestedAsyncShaderCompilationDelay = 0U;

    // Note: Only use the async shader compilation throttle delay when rendering which uses Remix shaders would actually take place. As such this delay is not
    // needed when ray tracing is not supported or enabled as Remix shaders will not be used in that case.
    if (m_rayTracingSupported && isRaytracingEnabled && asyncShaderCompilationActive) {
      requestedAsyncShaderCompilationDelay = RtxOptions::Shader::asyncCompilationThrottleMilliseconds();
    }

    // Note: Determine the throttle delay to use based on the larger of the two requested delay values as the larger should satisfy the requests of both.
    // A sum is also potentially a valid way of going about this, but a maximum makes more sense in that these delays aren't expected to stack but rather
    // are just requests for some minimum amount of time to spend waiting per frame.
    const auto computedPresentThrottleDelay = std::max(requestedPresentThrottleDelay, requestedAsyncShaderCompilationDelay);

    m_device->setPresentThrottleDelay(computedPresentThrottleDelay);

    // Early out if ray tracing is not supported or if Remix has already been injected

    if (!m_rayTracingSupported) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing doesn't appear to be supported on this HW.")));
      return;
    }

    if (m_frameLastInjected == m_device->getCurrentFrameId()) {
      return;
    }

    const bool isCameraValid = getSceneManager().getCamera().isValid(m_device->getCurrentFrameId());
    if (!isCameraValid) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace but not detecting a valid camera.")));
    }

    // Update frame counter only after actual rendering
    if (isCameraValid) {
      m_frameLastInjected = m_device->getCurrentFrameId();
    }

    if (RtxOptions::upscalerType() == UpscalerType::DLSS && !common->metaDLSS().supportsDLSS()) {
      RtxOptions::upscalerType.setDeferred(UpscalerType::TAAU);
    }

    if (DxvkDLFG::enable() && !common->metaDLFG().supportsDLFG()) {
      DxvkDLFG::enable.setDeferred(false);
    }
    
#ifdef REMIX_DEVELOPMENT
    // Update the Shader Manager

    ShaderManager::getInstance()->update();
#endif

    // [Perf.Frame] entry-subdivision: time the hot-reload pump separately
    // (TextureManager may pick up file-system / replacement events here,
    // which can produce the occasional 75ms spike observed in totals).
    const auto tEntryBeforeHotReload = PerfClock::now();
    perfFrame.entry_postCommitGfxUs = std::chrono::duration_cast<std::chrono::microseconds>(tEntryBeforeHotReload - tEntryAfterCommitGfx).count();
    common->getTextureManager().processAllHotReloadRequests();
    const auto tEntryAfterHotReload = PerfClock::now();
    perfFrame.entry_hotReloadUs = std::chrono::duration_cast<std::chrono::microseconds>(tEntryAfterHotReload - tEntryBeforeHotReload).count();

    const float gpuIdleTimeMilliseconds = getGpuIdleTimeSinceLastCall();

    bool raytracedThisFrame = false;

    // Note: Only engage ray tracing when it is enabled, the camera is valid and when no shaders are currently being compiled asynchronously (as
    // trying to render before shaders are done compiling will cause Remix to block).
    if (isRaytracingEnabled && isCameraValid && !asyncShaderCompilationActive) {
      if (targetImage == nullptr) {
        targetImage = m_state.om.renderTargets.color[0].view->image();  
      }

      const bool captureTestScreenshot = (m_screenshotFrameEnabled && m_device->getCurrentFrameId() == m_screenshotFrameNum);
      const bool captureScreenImage = s_triggerScreenshot || (captureTestScreenshot && !s_capturePrePresentTestScreenshot);
      const bool captureDebugImage = RtxOptions::captureDebugImage();
      
      if (s_triggerUsdCapture) {
        s_triggerUsdCapture = false;
        m_common->capturer()->triggerNewCapture();
      }

      if (captureTestScreenshot) {
        Logger::info(str::format("RTX: Test screenshot capture triggered"));
        Logger::info(str::format("RTX: Use separate denoiser ", RtxOptions::denoiseDirectAndIndirectLightingSeparately()));
        Logger::info(str::format("RTX: Use rtxdi ", RtxOptions::useRTXDI()));
        Logger::info(str::format("RTX: Use dlss ", RtxOptions::isDLSSOrRayReconstructionEnabled()));
        Logger::info(str::format("RTX: Use ray reconstruction ", RtxOptions::isRayReconstructionEnabled()));
        Logger::info(str::format("RTX: Use nis ", RtxOptions::isNISEnabled()));
        if (!s_capturePrePresentTestScreenshot) {
          m_screenshotFrameEnabled = false;
        }
      }

      if (captureScreenImage && captureDebugImage) {
        takeScreenshot("orgImage", targetImage);
      }

      // NV-DXVK [OnScreenAlbedoDump]: fire-and-forget; internally gated to dump
      // the on-screen albedos exactly once ~10s into gameplay. Safe here — same
      // point takeScreenshot()/dumpImageToFile() is already used.
      dumpOnScreenAlbedosOnce();

      RtxParticleSystemManager& particles = m_device->getCommon()->metaParticleSystem();
      particles.submitDrawState(this);

      this->spillRenderPass(false);

      getCommonObjects()->getTextureManager().submitTexturesToDeviceLocal(this, m_execBarriers, m_execAcquires);

      m_execBarriers.recordCommands(m_cmd);

      ScopedGpuProfileZone(this, "InjectRTX");

      // Signal Reflex rendering start

      RtxReflex& reflex = m_common->metaReflex();

      // Note: Update the Reflex mode in case the option has changed.
      reflex.updateMode();

      m_submitContainsInjectRtx = true;
      m_cachedReflexFrameId = cachedReflexFrameId;

      // NV-DXVK [EngineSun]: drive the atmosphere-sun-as-RTXDI-Distant-light BEFORE
      // prepareSceneData so it lands in this frame's light buffer. The bespoke NEE sun is
      // invisible to RTXDI (rtxdiIllum=0 on sun-lit skybox -> denoiser confidence floors ->
      // NRD blacks/streaks the mountains); a real Distant light populates the reservoir so
      // confidence is valid. Uses the current atmosphere args (1-frame lag on a static sun is
      // imperceptible). The NEE sun is disabled via cb.sunAsRtxdiLight to avoid double light.
      {
        LightManager& lightMgr = getSceneManager().getLightManager();
        const SkyMode skyModeNow = RtxOptions::skyMode();
        const bool sunRtxdiOn = RtxOptions::sunAsRtxdiLight()
          && (skyModeNow == SkyMode::PhysicalAtmosphere || skyModeNow == SkyMode::Hybrid)
          && m_atmosphere != nullptr;
        if (sunRtxdiOn) {
          const AtmosphereArgs& a = m_atmosphere->getAtmosphereArgs();
          // args.sunDirection points TOWARD the sun, in Y-up LUT space. Match the path tracer's
          // world space exactly like sampleAtmosphereSunLight does: swizzle (x,z,y) iff zUp.
          Vector3 towardSun = RtxOptions::zUp()
            ? Vector3(a.sunDirection.x, a.sunDirection.z, a.sunDirection.y)
            : Vector3(a.sunDirection.x, a.sunDirection.y, a.sunDirection.z);
          // Distant light direction is the propagation direction (sun -> scene) = -towardSun.
          const Vector3 propagation = Vector3(-towardSun.x, -towardSun.y, -towardSun.z);
          const Vector3 radiance = Vector3(
            a.sunIlluminance.x, a.sunIlluminance.y, a.sunIlluminance.z) * RtxOptions::sunRtxdiRadianceScale();
          lightMgr.setEngineSunLight(true, propagation, radiance, a.sunAngularRadius);

          // NV-DXVK [EngineSun]: confirm the exact RTXDI sun being injected — direction
          // (propagation), radiance, half angle, and the raw game inputs it was derived from.
          // Lets us verify the two things that can't be checked blind: direction SIGN (if the
          // scene is lit from the wrong side, flip the negation above) and brightness (tune
          // sunRtxdiRadianceScale). Throttled; gated on logSurfaceCoverage.
          if (RtxOptions::logSurfaceCoverage() && (m_device->getCurrentFrameId() % 30u == 0u)) {
            Logger::info(str::format(
              "[EngineSun] active=1 zUp=", RtxOptions::zUp() ? 1 : 0,
              " towardSun=(", towardSun.x, ",", towardSun.y, ",", towardSun.z, ")",
              " propagation=(", propagation.x, ",", propagation.y, ",", propagation.z, ")",
              " radiance=(", radiance.x, ",", radiance.y, ",", radiance.z, ")",
              " halfAngleRad=", a.sunAngularRadius,
              " | rawSunIllum=(", a.sunIlluminance.x, ",", a.sunIlluminance.y, ",", a.sunIlluminance.z, ")",
              " scale=", RtxOptions::sunRtxdiRadianceScale(),
              " skyMode=", static_cast<uint32_t>(skyModeNow)));
          }
        } else {
          lightMgr.setEngineSunLight(false, Vector3(0.0f), Vector3(0.0f), 0.0f);
          if (RtxOptions::logSurfaceCoverage() && (m_device->getCurrentFrameId() % 120u == 0u)) {
            Logger::info(str::format(
              "[EngineSun] active=0 optOn=", RtxOptions::sunAsRtxdiLight() ? 1 : 0,
              " atmospherePresent=", (m_atmosphere != nullptr) ? 1 : 0,
              " skyMode=", static_cast<uint32_t>(RtxOptions::skyMode())));
          }
        }
      }

      // Update all the GPU buffers needed to describe the scene
      getSceneManager().prepareSceneData(this, m_execBarriers);

      // If we really don't have any RT to do, just bail early (could be UI/menus rendering)
      // Also guard against a null Opaque TLAS: this can happen on the first frame(s) with geometry
      // if BLAS build hasn't completed yet, or when objectToWorld extraction was wrong and produced
      // degenerate geometry.  Binding a null AS handle to the RT descriptor set triggers
      // VK_ERROR_DEVICE_LOST during ray traversal (seen in TF2: pAccelerationStructures[0] = 0x0).
      const bool tlasReady = getResourceManager().getTLAS(Tlas::Opaque).accelStructure != nullptr;
      if (!tlasReady) {
        ONCE(Logger::warn("[RTX] Skipping RT dispatch this frame: Opaque TLAS is not yet built (no valid geometry submitted, or BLAS build pending)"));
      }
      if (getSceneManager().getSurfaceBuffer() != nullptr && tlasReady) {

        // [Perf.Frame] per-stage CPU wall time. Fills the outer perfFrame; the actual log line is
        // emitted at end-of-injectRTX (ungated, 5s wallclock throttle).
        perfFrame.rtBranchRan = true;
        auto markStage = [](PerfClock::time_point& last, int64_t& sink) {
          const auto now = PerfClock::now();
          sink = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
          last = now;
        };

        auto tStage = PerfClock::now();
        // Entry-to-here time: everything in injectRTX before this point (commitGraphicsState,
        // option reads, tlasReady check, etc.) — closes the largest unaccounted-for gap.
        // Subdivided into preCommitGfx/commitGfx/postCommitGfx/hotReload/tailToBranch fields
        // so the 75ms spikes can be attributed (see [Perf.Frame] log).
        perfFrame.entryToOnFrameBeginUs = std::chrono::duration_cast<std::chrono::microseconds>(tStage - tPerfFrameStart).count();
        perfFrame.entry_tailToBranchUs = std::chrono::duration_cast<std::chrono::microseconds>(tStage - tEntryAfterHotReload).count();
        VkExtent3D downscaledExtent = onFrameBegin(targetImage->info().extent);
        markStage(tStage, perfFrame.onFrameBeginUs);

        Resources::RaytracingOutput& rtOutput = getResourceManager().getRaytracingOutput();

        if (common->metaNGXContext().supportsDLFG()) {
          rtOutput.m_primaryDepthQueue.next();
          rtOutput.m_primaryScreenSpaceMotionVectorQueue.next();
        }

        rtOutput.m_primaryDepth = rtOutput.m_primaryDepthQueue.get();
        rtOutput.m_primaryScreenSpaceMotionVector = rtOutput.m_primaryScreenSpaceMotionVectorQueue.get();

        getCommonObjects()->getTextureManager().prepareSamplerFeedback(this);

        // Generate ray tracing constant buffer
        updateRaytraceArgsConstantBuffer(rtOutput, downscaledExtent, targetImage->info().extent);
        markStage(tStage, perfFrame.prepUs);

        // Volumetric Lighting
        dispatchVolumetrics(rtOutput);
        markStage(tStage, perfFrame.volumetricsUs);

        // Path Tracing
        dispatchPathTracing(rtOutput);
        markStage(tStage, perfFrame.pathTracingUs);

        // Neural Radiance Cache
        m_common->metaNeuralRadianceCache().dispatchTrainingAndResolve(*this, rtOutput);
        markStage(tStage, perfFrame.nrcUs);

        // RTXDI confidence
        m_common->metaRtxdiRayQuery().dispatchConfidence(this, rtOutput);
        markStage(tStage, perfFrame.rtxdiUs);

        // ReSTIR GI
        m_common->metaReSTIRGIRayQuery().dispatch(this, rtOutput);
        markStage(tStage, perfFrame.restirUs);

        // NV-DXVK [MtnRadiance]: sample albedo/direct/indirect for distant-hit pixels
        // HERE — before demodulate (radiance still raw, not divided by albedo).
        captureMountainRadianceProbe(rtOutput);

        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("baseReflectivity", rtOutput.m_primaryBaseReflectivity.image(Resources::AccessType::Read));
          takeScreenshot("sharedSubsurfaceData", rtOutput.m_sharedSubsurfaceData.image);
          takeScreenshot("sharedSubsurfaceDiffusionProfileData", rtOutput.m_sharedSubsurfaceDiffusionProfileData.image);
        }

        // Demodulation
        dispatchDemodulate(rtOutput);
        markStage(tStage, perfFrame.demodulateUs);

        // Note: Primary direct diffuse/specular radiance textures noisy and in a demodulated state after demodulation step.
        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("noisyDiffuse", rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read));
          takeScreenshot("noisySpecular", rtOutput.m_primaryDirectSpecularRadiance.image(Resources::AccessType::Read));
        }

        // Denoising
        dispatchDenoise(rtOutput);
        markStage(tStage, perfFrame.denoiseUs);

        // Note: Primary direct diffuse/specular radiance textures denoised but in a still demodulated state after denoising step.
        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("denoisedDiffuse", rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read));
          takeScreenshot("denoisedSpecular", rtOutput.m_primaryDirectSpecularRadiance.image(Resources::AccessType::Read));
        }

        // Composition
        dispatchComposite(rtOutput);
        markStage(tStage, perfFrame.compositeUs);

        // NV-DXVK [MtnComposite]: read the resolved on-screen colour HERE — after
        // composite (so m_compositeOutput is this frame's) but before the debug-view
        // pass below can overwrite it. Names which emissive backdrop VS renders black.
        // Gated behind tf2HeavyProbes (default OFF): this does 9 image readbacks +
        // per-pixel logging every frame — the main Aftermath device-loss driver.
        if (RtxOptions::tf2HeavyProbes())
          captureMountainCompositeProbe(rtOutput);

        // NV-DXVK: measure average scene radiance from this frame's composite and feed
        // NRC's dynamic maxExpectedAverageRadianceValue (self-tuning radiance scale).
        updateNrcDynamicRadiance(rtOutput);

        // Post composite Debug View that may overwrite Composite output
        dispatchReplaceCompositeWithDebugView(rtOutput);
        markStage(tStage, perfFrame.debugViewUs);

        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("rtxImagePostComposite", rtOutput.m_compositeOutput.resource(Resources::AccessType::Read).image);
        }

        getCommonObjects()->getTextureManager().copySamplerFeedbackToHost(this);
        dispatchObjectPicking(rtOutput, downscaledExtent, targetImage->info().extent);
        markStage(tStage, perfFrame.postCompositeUs);

        // Upscaling if DLSS/NIS enabled, or the Composition Pass will do upscaling
        if (m_currentUpscaler == InternalUpscaler::DLSS) {
          // xxxnsubtil: the DLSS indicator reads our exposure texture even with DLSS autoexposure on
          // make sure it has been created, otherwise we run into trouble on the first frame
          m_common->metaAutoExposure().createResources(this);
          dispatchDLSS(rtOutput);
        } else if (m_currentUpscaler == InternalUpscaler::DLSS_RR) {
          m_common->metaAutoExposure().createResources(this);
          dispatchRayReconstruction(rtOutput);
        } else if (m_currentUpscaler == InternalUpscaler::XeSS) {
          m_common->metaAutoExposure().createResources(this);
          dispatchXeSS(rtOutput);
        } else if (m_currentUpscaler == InternalUpscaler::NIS) {
          dispatchNIS(rtOutput);
        } else if (m_currentUpscaler == InternalUpscaler::TAAU){
          dispatchTemporalAA(rtOutput);
        } else {
          copyImage(
            rtOutput.m_finalOutput.resource(Resources::AccessType::Write).image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            { 0, 0, 0 },
            rtOutput.m_compositeOutput.image(Resources::AccessType::Read),
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            { 0, 0, 0 },
            rtOutput.m_compositeOutputExtent);
        }
        m_previousUpscaler = m_currentUpscaler;
        markStage(tStage, perfFrame.upscalerUs);

        RtxDustParticles& dust = m_common->metaDustParticles();
        dust.simulateAndDraw(this, m_state, rtOutput);

        dispatchBloom(rtOutput);
        dispatchPostFx(rtOutput);

        // Tone mapping
        // WAR for TREX-553 - disable sRGB conversion as NVTT implicitly applies it during dds->png
        // conversion for 16bit float formats
        const bool performSRGBConversion = !captureScreenImage && g_allowSrgbConversionForOutput;
        dispatchToneMapping(rtOutput, performSRGBConversion);

        // NV-DXVK [engine-post DoF, Route B]: depth of field on the tonemapped
        // image (matches the host game, whose DoF runs on the post-tonemap frame),
        // recomputing circle-of-confusion from the path-traced depth.
        dispatchDepthOfField(rtOutput);

        if (captureScreenImage) {
          if (m_common->metaDebugView().debugViewIdx() == DEBUG_VIEW_DISABLED) {
            takeScreenshot("rtxImagePostTonemapping", rtOutput.m_finalOutput.resource(Resources::AccessType::Read).image);
          }
          
          if (captureDebugImage) {
            takeScreenshot("albedo", rtOutput.m_primaryAlbedo.image);
            takeScreenshot("worldNormals", rtOutput.m_primaryWorldShadingNormal.image);
            takeScreenshot("worldMotion", rtOutput.m_primaryVirtualMotionVector.image(Resources::AccessType::Read));
            takeScreenshot("linearZ", rtOutput.m_primaryLinearViewZ.image);
          }
        }

        // Set up output src
        Rc<DxvkImage> srcImage = rtOutput.m_finalOutput.resource(Resources::AccessType::Read).image;

        // Debug view
        dispatchDebugView(srcImage, rtOutput, captureScreenImage);

        dispatchDLFG();
        markStage(tStage, perfFrame.finalBlitUs);

        // Blit to the game target
        // NV-DXVK: m_skipBackbufferBlitThisFrame gate — the D3D11
        // HUD-deferral path sets this sticky flag (via
        // requestSkipBackbufferBlit) when the early-inject hook fires
        // before the camera has latched. We still need the full RT
        // pipeline to run so scene state doesn't leak, but the final
        // blit would overwrite the HUD that D3D11 rastered to the
        // backbuffer — so skip it for this frame only. The param form
        // (skipBackbufferBlit) honors either source.
        const bool skipBlit = skipBackbufferBlit || m_skipBackbufferBlitThisFrame;
        using clk = std::chrono::steady_clock;
        const auto tBlitStart = clk::now();

        // NV-DXVK [HUD-Option5 v4]: save the POST-tonemap srcImage
        // (= m_finalOutput, linear [0,1] SFLOAT16) into a persistent
        // scratch so the d3d11-side SubmitDraw blit can read it next
        // frame. The blit runs mid-frame between TF2's composite and
        // its HUD rasters, so last-frame's RT is blitted over this
        // frame's composite output, then HUD draws layer on top. One-
        // frame RT lag is unavoidable here but is barely perceptible.
        if (!skipBlit) {
          ScopedGpuProfileZone(this, "Blit to Game");

          {
            const auto& postInfo = srcImage->info();
            const bool needsAlloc =
              m_rtFinalPostTonemapScratch == nullptr ||
              m_rtFinalPostTonemapScratchExtent.width  != postInfo.extent.width ||
              m_rtFinalPostTonemapScratchExtent.height != postInfo.extent.height ||
              m_rtFinalPostTonemapScratchFormat != postInfo.format;
            if (needsAlloc) {
              DxvkImageCreateInfo info;
              info.type = VK_IMAGE_TYPE_2D;
              info.format = postInfo.format;
              info.flags = 0;
              info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
              info.extent = { postInfo.extent.width, postInfo.extent.height, 1 };
              info.numLayers = 1;
              info.mipLevels = 1;
              info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
              info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
              info.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
              info.tiling = VK_IMAGE_TILING_OPTIMAL;
              info.layout = VK_IMAGE_LAYOUT_GENERAL;
              info.shared = VK_FALSE;
              m_rtFinalPostTonemapScratch = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "rt-final-post-tonemap-scratch");
              m_rtFinalPostTonemapScratchExtent = postInfo.extent;
              m_rtFinalPostTonemapScratchFormat = postInfo.format;
              m_rtFinalPostTonemapScratchHasContent = false;
            }
            VkImageCopy region = {};
            region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.extent = postInfo.extent;
            copyImage(m_rtFinalPostTonemapScratch, region.dstSubresource, VkOffset3D{ 0, 0, 0 },
                      srcImage,                    region.srcSubresource, VkOffset3D{ 0, 0, 0 },
                      region.extent);
            m_rtFinalPostTonemapScratchHasContent = true;
          }

          if (m_compositeInterceptedThisFrame) {
            // v4 already blitted our RT onto the game's composite output
            // mid-frame; TF2's HUD rasters landed on top, and TF2's
            // present-time copy carries the full (RT + HUD) frame into
            // the swap chain. Skip our own backbuffer blit — it would
            // clobber the HUD.
            static uint64_t sInterceptCount = 0;
            ++sInterceptCount;
            if (sInterceptCount <= 3 || (sInterceptCount & 0xFF) == 0) {
              Logger::info(str::format("[CompositeIntercept] injectRTX skipped backbuffer blit #", sInterceptCount));
            }
          } else {
            assert(srcImage->info().extent == targetImage->info().extent);
            blitImageHelper(this, srcImage, targetImage, VkFilter::VK_FILTER_NEAREST);
          }
        }
        // Reset v4 intercept flag for next frame.
        m_compositeInterceptedThisFrame = false;

        const auto tBlitEnd = clk::now();
        // Reset sticky per-frame flag whether we blitted or not.
        m_skipBackbufferBlitThisFrame = false;

        // Log stats when an image is taken
        if (captureScreenImage) {
          getSceneManager().logStatistics();
        }

        m_common->metaNeuralRadianceCache().onFrameEnd(rtOutput);

        rtOutput.onFrameEnd();
        raytracedThisFrame = true;
        markStage(tStage, perfFrame.endFrameUs);
      }

      m_framesWithoutValidScene = 0;
    } else {
      // If raytracing is only disabled because we don't have shaders available, we don't want to clear the scene.
      // This frequently happens for a single frame when a cached shader is being fetched, and causes the Logic 
      // graph state to be reset - which is problematic since Logic graphs often trigger shader fetches.
      // It might be safe to remove this clear entirely - it was added before we had any garbage collection
      // in the scene manager, so it may not be needed anymore.
      if (!isRaytracingEnabled || !isCameraValid) {
        m_framesWithoutValidScene++;
        // NV-DXVK [SceneInvalidRaw]: UNGATED (logSurfaceCoverage) companion to
        // the gameplay-gated [InvalidSceneProbe] below, which is suppressed
        // until engineHookCaptureCount>16 and so misses the load/transition
        // window where the f695-style reset fires. Logs the EXACT reason the
        // scene is considered invalid (which of the two flags) + whether this
        // frame will trigger the clear, every time the invalid branch runs.
        // Cross-reference by frame id with [SceneClearRaw]/[LightProbe]/FinalGrid.
        if (RtxOptions::logSurfaceCoverage()) {
          Logger::info(str::format(
            "[SceneInvalidRaw] f=", m_device->getCurrentFrameId(),
            " streak=", m_framesWithoutValidScene,
            " isRaytracingEnabled=", (isRaytracingEnabled ? 1 : 0),
            " isCameraValid=", (isCameraValid ? 1 : 0),
            " keepAlive=", RtxOptions::sceneKeepAliveFrames(),
            " willClear=", (m_framesWithoutValidScene > RtxOptions::sceneKeepAliveFrames() ? 1 : 0)));
        }
        // NV-DXVK [InvalidSceneProbe]: log every bad-frame transition so we
        // can see WHY the scene was about to be cleared. Continuous gameplay
        // shouldn't produce invalid frames; if it does, one of these flags
        // is wrong and we need to fix the check, not absorb it with a config.
        {
          // NV-DXVK [InvalidSceneProbe]: gameplay-only. During loading the
          // scene is invalid every frame (1000+ events) and the resulting
          // clears are expected; logging them is noise. During gameplay
          // ANY invalidation is suspect because it silently nukes the
          // SpatialMap entries that sub-view identity dedup relies on.
          // Gate matches [SceneClearProbe] in rtx_scene_manager.cpp.
          if (tf2::g_engineHookCaptureCount.load(std::memory_order_relaxed) > 16u) {
            const auto& mainCam = getSceneManager().getCamera();
            const uint32_t curF = m_device->getCurrentFrameId();
            static uint64_t sBadN = 0;
            ++sBadN;
            Logger::warn(str::format(
              "[InvalidSceneProbe] n=", sBadN,
              " f=", curF,
              " streak=", m_framesWithoutValidScene,
              " isRaytracingEnabled=", (isRaytracingEnabled ? 1 : 0),
              " isCameraValid=", (isCameraValid ? 1 : 0),
              " mainCam.lastTouched=", mainCam.getLastUpdateFrame(),
              " willClear=", (m_framesWithoutValidScene > RtxOptions::sceneKeepAliveFrames() ? 1 : 0)));
          }
        }
        // Some games may have invalid cameras for a brief period during camera cuts, but clearing the scene
        // during these cuts causes all textures to need to be reloaded, which is slow.
        if (m_framesWithoutValidScene > RtxOptions::sceneKeepAliveFrames()) {
          // Only perform Wait For Idle on the first clear to avoid expensive GPU sync on every frame
          const bool needWfi = (m_framesWithoutValidScene == RtxOptions::sceneKeepAliveFrames() + 1);
          getSceneManager().clear(this, needWfi);
        }
      } else {
        m_framesWithoutValidScene = 0;
      }
    }

    getSceneManager().onFrameEnd(this, raytracedThisFrame);

    // apply changes to RtxOptions after the frame has ended
    RtxOptionManager::applyPendingValues(m_device.ptr(), /* forceOnChange */ false);

    // Update stats
    updateMetrics(gpuIdleTimeMilliseconds);

    m_resetHistory = false;

    // NV-DXVK [BlitDiag]: end-of-function summary. Report total wall time
    // + which skip-blit inputs were active + the gpu idle time from the
    // (pre-existing) metric collector above. Compare across frames with
    // different skipBlit states to locate the stall.
    if (s_blitDiagEnabledOuter) {
      const auto tInjectEnd = std::chrono::steady_clock::now();
      const auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tInjectEnd - tInjectStart).count();
      const uint32_t fid = m_device->getCurrentFrameId();
      Logger::info(str::format(
        "[BlitDiag] frame=", fid,
        " total_us=", totalUs,
        " gpuIdle_ms=", gpuIdleTimeMilliseconds,
        " paramSkip=", (skipBackbufferBlit ? 1 : 0),
        " stickySkipAtEntry=", (stickySkipAtEntry ? 1 : 0),
        " camValid=", (isCameraValid ? 1 : 0),
        " raytraced=", (raytracedThisFrame ? 1 : 0)));
    }

    // [Perf.Frame] end-of-injectRTX log. UNGATED: covers menu frames (rtBranchRan=0) AND gameplay
    // frames (rtBranchRan=1). Throttled to once per 5 seconds wallclock so steady-state log volume
    // is constant regardless of FPS. Per-DLL via static; each DLL prints its own timeline.
    // First call always logs (sFirstCall flag) — we DON'T use time_point::min() as a sentinel
    // because subtracting it from now() overflows int64 nanoseconds and wraps to a negative
    // duration, which never satisfies the >=5000 threshold.
    {
      const auto tInjectEnd = PerfClock::now();
      const int64_t totalInjectUs = std::chrono::duration_cast<std::chrono::microseconds>(tInjectEnd - tPerfFrameStart).count();
      static bool sFirstCall = true;
      static PerfClock::time_point sLastPerfLog;
      static uint32_t sFramesSinceLastLog = 0;
      sFramesSinceLastLog++;
      int64_t sinceLast = 5000;
      if (!sFirstCall) {
        sinceLast = std::chrono::duration_cast<std::chrono::milliseconds>(tInjectEnd - sLastPerfLog).count();
      }
      if (sFirstCall || sinceLast >= 5000) {
        sFirstCall = false;
        const uint32_t fid = m_device->getCurrentFrameId();
        Logger::info(str::format("[Perf.Frame] fid=", fid,
                                 " framesSinceLastLog=", sFramesSinceLastLog,
                                 " rtBranchRan=", (perfFrame.rtBranchRan ? 1 : 0),
                                 " totalInjectUs=", totalInjectUs,
                                 " entryToOnFrameBegin=", perfFrame.entryToOnFrameBeginUs,
                                 " entry_preCommitGfx=", perfFrame.entry_preCommitGfxUs,
                                 " entry_commitGfx=", perfFrame.entry_commitGfxUs,
                                 " entry_postCommitGfx=", perfFrame.entry_postCommitGfxUs,
                                 " entry_hotReload=", perfFrame.entry_hotReloadUs,
                                 " entry_tailToBranch=", perfFrame.entry_tailToBranchUs,
                                 " onFrameBegin=", perfFrame.onFrameBeginUs,
                                 " prep=", perfFrame.prepUs,
                                 " volumetrics=", perfFrame.volumetricsUs,
                                 " pathTracing=", perfFrame.pathTracingUs,
                                 " nrc=", perfFrame.nrcUs,
                                 " rtxdi=", perfFrame.rtxdiUs,
                                 " restir=", perfFrame.restirUs,
                                 " demodulate=", perfFrame.demodulateUs,
                                 " denoise=", perfFrame.denoiseUs,
                                 " composite=", perfFrame.compositeUs,
                                 " debugView=", perfFrame.debugViewUs,
                                 " postComposite=", perfFrame.postCompositeUs,
                                 " upscaler=", perfFrame.upscalerUs,
                                 " finalBlit=", perfFrame.finalBlitUs,
                                 " endFrame=", perfFrame.endFrameUs));
        sLastPerfLog = tInjectEnd;
        sFramesSinceLastLog = 0;
      }
    }
  }

  void RtxContext::recordVisibleSurfacesReadback(const Resources::RaytracingOutput& rtOutput) {
    if constexpr (!kEnableRtxDebugProbes) {
      return;
    }
    const uint32_t curFrame = m_device->getCurrentFrameId();

    // Gate: only run during actual gameplay frames.
    //  - camera must be valid
    //  - throttle to one readback per kReadbackPeriod frames (avoids 14 MB/frame staging churn)
    constexpr uint32_t kReadbackPeriod = 3;
    if ((curFrame % kReadbackPeriod) != 0) {
      return;
    }
    if (!getSceneManager().getCamera().isValid(curFrame)) {
      // Diag: surface why we skipped, throttled to once per ~17s
      static uint32_t s_lastDiag = 0;
      if (curFrame - s_lastDiag > 1000 || curFrame < s_lastDiag) {
        s_lastDiag = curFrame;
        // [SpawnGeomDiag] renamed from [VisibleSurf] to bypass log.cpp's
        // "[VisibleSurf]" filter.
        Logger::info(str::format("[SpawnGeomDiag.VisibleSurf] frame=", curFrame, " skipped: camera invalid"));
      }
      return;
    }

    Rc<DxvkImage> srcImage = rtOutput.m_sharedSurfaceIndex.image(Resources::AccessType::Read);
    if (srcImage == nullptr) {
      return;
    }

    // Reap completed async tasks
    {
      auto& tasks = m_visibleSurfacesReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(), [](std::future<void>& f) {
        return !f.valid() || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      }), tasks.end());
    }

    const VkExtent3D extent = srcImage->info().extent;
    constexpr VkDeviceSize kBytesPerPixel = sizeof(uint32_t);
    const VkDeviceSize totalBytes = kBytesPerPixel * extent.width * extent.height;

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size    = totalBytes;
    bufInfo.usage   = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.stages  = VK_PIPELINE_STAGE_TRANSFER_BIT  | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access  = VK_ACCESS_TRANSFER_WRITE_BIT    | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer, "Visible Surfaces Readback");

    VkImageSubresourceLayers subres {};
    subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel       = 0;
    subres.baseArrayLayer = 0;
    subres.layerCount     = 1;

    copyImageToBuffer(
      readbackDst, 0, kBytesPerPixel, kBytesPerPixel * extent.width,
      srcImage, subres, VkOffset3D { 0, 0, 0 }, extent);

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_visibleSurfacesReadback.signalValue;
    this->signal(m_visibleSurfacesReadback.signal, syncValue);

    const uint32_t frameIdx   = m_device->getCurrentFrameId();
    const uint32_t pixelCount = extent.width * extent.height;
    Rc<sync::Fence> signalRef = m_visibleSurfacesReadback.signal;

    m_visibleSurfacesReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef, frameIdx, pixelCount,
       w = extent.width, h = extent.height]() mutable {
        signalRef->wait(syncValue);
        const uint32_t* p = reinterpret_cast<const uint32_t*>(cReadbackDst->mapPtr(0));
        if (p == nullptr) {
          return;
        }
        std::map<uint32_t, uint32_t> counts;
        uint32_t invalidPixels = 0;
        for (uint32_t i = 0; i < pixelCount; ++i) {
          const uint32_t v = p[i];
          if (v == SURFACE_INDEX_INVALID) { ++invalidPixels; continue; }
          counts[v]++;
        }
        // Build "id:count" pairs sorted by count desc, top 32
        std::vector<std::pair<uint32_t, uint32_t>> sorted(counts.begin(), counts.end());
        std::sort(sorted.begin(), sorted.end(),
          [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string ids;
        ids.reserve(sorted.size() * 12);
        size_t shown = 0;
        for (auto& kv : sorted) {
          if (shown++ >= 32) { ids += "..."; break; }
          ids += std::to_string(kv.first);
          ids += ':';
          ids += std::to_string(kv.second);
          ids += ' ';
        }
        // [SpawnGeomDiag.VisibleSurf] also dump the FULL set of visible
        // surface IDs as a sorted comma-separated list. The "top=" field
        // only covers the 32 highest-count IDs; without the full list we
        // can't tell whether a low-count surface (e.g. distant floor
        // tile) is actually visible or completely missed. This is the
        // diagnostic that resolves the floor question definitively: if
        // the PI batch's surfRange is fully absent from this list, those
        // instances are not being intersected by any primary ray.
        std::string allIds;
        allIds.reserve(sorted.size() * 6);
        std::vector<uint32_t> sortedById;
        sortedById.reserve(counts.size());
        for (auto& kv : counts) sortedById.push_back(kv.first);
        std::sort(sortedById.begin(), sortedById.end());
        for (uint32_t id : sortedById) {
          allIds += std::to_string(id);
          allIds += ',';
        }
        if (!allIds.empty() && allIds.back() == ',') allIds.pop_back();
        // [SpawnGeomDiag] renamed from [VisibleSurf] so log.cpp's filter
        // doesn't drop it. This is the diagnostic that answers "which
        // surface IDs do primary rays actually hit?" — for the missing-
        // floor debug, if the floor's surface IDs (from PI batch
        // baseSurfaceIndex + offset) don't appear here, the floor BLAS
        // is in TLAS but rays don't intersect it (mask=0, transform
        // wrong, or back-face culled). If they DO appear, the bug is in
        // material/shading after intersection.
        Logger::info(str::format(
          "[SpawnGeomDiag.VisibleSurf] frame=", frameIdx,
          " res=", w, "x", h,
          " unique=", counts.size(),
          " invalidPx=", invalidPixels,
          " top=[", ids, "]",
          " all=[", allIds, "]"));
      }));
  }

  // NV-DXVK [Atmosphere.lut readback]: copy three Z-slices of the
  // AerialPerspective LUT (near-camera, mid, far) to a HOST_VISIBLE
  // staging buffer, decode RGBA16F on a worker, and log min/max/avg
  // of the inscatter (.rgb) and mono transmittance (.a) per slice.
  // Layout per slice: 32x32 RGBA16F = 8 KB. Three slices => 24 KB.
  // Caller must invoke this AFTER computeLuts has dispatched the
  // generator (and emitted its compute->RT barrier). We add a
  // compute->transfer barrier internally before the copy.
  void RtxContext::recordAerialPerspectiveLutReadback() {
    if (m_atmosphere == nullptr) {
      return;
    }

    Resources::Resource lut = m_atmosphere->getAerialPerspectiveLut();
    if (lut.image == nullptr) {
      return;
    }

    // Reap completed async tasks so the vector doesn't grow forever.
    {
      auto& tasks = m_aerialPerspectiveLutReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
        [](std::future<void>& f) {
          return !f.valid()
              || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), tasks.end());
    }

    constexpr uint32_t kLutSize       = 32;     // matches kAerialPerspectiveLutSize
    constexpr uint32_t kBytesPerTexel = 8;      // RGBA16F
    constexpr uint32_t kSlicesToCopy  = 3;      // near / mid / far
    constexpr VkDeviceSize kSliceBytes = kBytesPerTexel * kLutSize * kLutSize;
    constexpr VkDeviceSize kTotalBytes = kSliceBytes * kSlicesToCopy;

    // tZ indices: distance maps as distanceKm = (tZ/(kLutSize-1))^2 * maxKm
    // For maxKm=32:
    //   z=2  -> distance ~= 0.13 km  (very near)
    //   z=8  -> distance ~= 2.05 km  (mid)
    //   z=24 -> distance ~= 18.6 km  (far)
    constexpr uint32_t kSliceZ[kSlicesToCopy] = { 2u, 8u, 24u };

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size   = kTotalBytes;
    bufInfo.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT   | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer,
      "Aerial Perspective LUT Readback");

    // Compute writes the LUT, transfer reads it. Atmosphere already
    // emitted compute->compute and compute->RT barriers; we need our
    // own compute->transfer barrier.
    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,       VK_ACCESS_TRANSFER_READ_BIT);

    VkImageSubresourceLayers subres {};
    subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel       = 0;
    subres.baseArrayLayer = 0;
    subres.layerCount     = 1;

    for (uint32_t i = 0; i < kSlicesToCopy; ++i) {
      copyImageToBuffer(
        readbackDst, kSliceBytes * i,
        kBytesPerTexel,                // dstRowAlignment
        kBytesPerTexel * kLutSize,     // dstSliceAlignment
        lut.image, subres,
        VkOffset3D { 0, 0, static_cast<int32_t>(kSliceZ[i]) },
        VkExtent3D { kLutSize, kLutSize, 1u });
    }

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_aerialPerspectiveLutReadback.signalValue;
    this->signal(m_aerialPerspectiveLutReadback.signal, syncValue);

    const uint32_t frameIdx = m_device->getCurrentFrameId();
    Rc<sync::Fence> signalRef = m_aerialPerspectiveLutReadback.signal;
    const float strength = m_atmosphere->getAtmosphereArgs().aerialPerspectiveStrength;

    m_aerialPerspectiveLutReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef,
       frameIdx, strength,
       sliceZ0 = kSliceZ[0], sliceZ1 = kSliceZ[1], sliceZ2 = kSliceZ[2]]() mutable {
        signalRef->wait(syncValue);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }
        // IEEE half->float, copied from rtx_accel_manager.cpp:2735.
        auto h2f = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t exp  = (h & 0x7C00u) >> 10;
          const uint32_t mant = (h & 0x03FFu);
          uint32_t bits;
          if (exp == 0) {
            bits = sign;
            if (mant != 0) {
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

        const uint32_t sliceZArr[3] = { sliceZ0, sliceZ1, sliceZ2 };
        constexpr uint32_t lutSize  = 32;
        constexpr uint32_t kSliceBytes = 8u * lutSize * lutSize;
        for (uint32_t s = 0; s < 3; ++s) {
          const uint16_t* h = reinterpret_cast<const uint16_t*>(base + kSliceBytes * s);
          // h is 32x32 RGBA16F, row-major. tX = column, tY = row.
          float minR = +INFINITY, maxR = -INFINITY, sumR = 0.0f;
          float minG = +INFINITY, maxG = -INFINITY, sumG = 0.0f;
          float minB = +INFINITY, maxB = -INFINITY, sumB = 0.0f;
          float minA = +INFINITY, maxA = -INFINITY, sumA = 0.0f;
          const uint32_t numTexels = lutSize * lutSize;
          for (uint32_t i = 0; i < numTexels; ++i) {
            const float r = h2f(h[i * 4 + 0]);
            const float g = h2f(h[i * 4 + 1]);
            const float b = h2f(h[i * 4 + 2]);
            const float a = h2f(h[i * 4 + 3]);
            if (r < minR) minR = r;  if (r > maxR) maxR = r;  sumR += r;
            if (g < minG) minG = g;  if (g > maxG) maxG = g;  sumG += g;
            if (b < minB) minB = b;  if (b > maxB) maxB = b;  sumB += b;
            if (a < minA) minA = a;  if (a > maxA) maxA = a;  sumA += a;
          }
          const float invN = 1.0f / static_cast<float>(numTexels);
          // Sample texels: tX=tY=16 (center), tX=0,tY=16 (sun-opposite),
          // tX=31,tY=16 (sun-aligned), tX=16,tY=31 (zenith).
          auto sampleAt = [&](uint32_t tx, uint32_t ty) {
            const uint32_t i = (ty * lutSize + tx) * 4;
            return std::array<float, 4> {
              h2f(h[i + 0]), h2f(h[i + 1]), h2f(h[i + 2]), h2f(h[i + 3])
            };
          };
          const auto cen = sampleAt(16, 16);
          const auto opp = sampleAt(0,  16);
          const auto sun = sampleAt(31, 16);
          const auto zen = sampleAt(16, 31);

          // distanceKm = (tZ/31)^2 * 32  for kLutSize=32
          const float tZNorm = static_cast<float>(sliceZArr[s]) / 31.0f;
          const float distKm = tZNorm * tZNorm * 32.0f;

          Logger::info(str::format(
            "[Atmosphere.lut] frame=", frameIdx,
            " strength=", strength,
            " tZidx=", sliceZArr[s], " (~", distKm, " km)",
            " inscatter avg=(", sumR * invN, ",", sumG * invN, ",", sumB * invN, ")",
            " min=(", minR, ",", minG, ",", minB, ")",
            " max=(", maxR, ",", maxG, ",", maxB, ")",
            " transmittance avg=", sumA * invN, " min=", minA, " max=", maxA,
            " center(tX=16,tY=16) rgba=(", cen[0], ",", cen[1], ",", cen[2], ",", cen[3], ")",
            " sunOpp(tX=0,tY=16) rgba=(", opp[0], ",", opp[1], ",", opp[2], ",", opp[3], ")",
            " sunAlg(tX=31,tY=16) rgba=(", sun[0], ",", sun[1], ",", sun[2], ",", sun[3], ")",
            " zen(tX=16,tY=31) rgba=(", zen[0], ",", zen[1], ",", zen[2], ",", zen[3], ")"));
        }
      }));
  }

  // NV-DXVK [SkyTint.diag] removed — replaced by direct cbuffer
  // capture of c_skyColor + c_envMapLightScale via the existing
  // EngineSunSnapshot pipeline. The tint is now plumbed to
  // atmosphereArgs.skyTint and applied at the IBL sample sites
  // in geometry_resolver.slangh and integrator_indirect.slangh.
  // The original GPU-side readback (32×32 sky-matte + 32×32 SkyView
  // LUT comparison) is preserved in git history if a future
  // diagnostic regression needs it.

  // [SkyTrace.primaryMiss] async readback of a 128x128 center tile of
  // PrimaryLinearViewZ. The composite shader classifies a pixel as
  // "primary miss" (sky) when its linearViewZ exactly matches a sentinel
  // value (cb.primaryDirectMissLinearViewZ). This readback counts how
  // many pixels in the screen-center tile match — if 0, the visible sky
  // pixels are NOT taking the matte-sample branch in composite, which
  // means the matte being white doesn't matter and the yellow has to
  // come from the AP+fog block (composite.comp.slang:533-577) which
  // applies cb.atmosphereArgs.fogColor.
  void RtxContext::recordPrimaryMissCountReadback(
      const Resources::RaytracingOutput& rtOutput,
      float missLinearViewZ) {
    if (rtOutput.m_primaryLinearViewZ.image == nullptr) {
      return;
    }

    // Reap completed tasks.
    {
      auto& tasks = m_primaryMissCountReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
        [](std::future<void>& f) {
          return !f.valid()
              || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), tasks.end());
    }

    const VkExtent3D pvzExtent = rtOutput.m_primaryLinearViewZ.image->info().extent;
    const VkFormat pvzFormat = rtOutput.m_primaryLinearViewZ.image->info().format;
    if (pvzFormat != VK_FORMAT_R32_SFLOAT) {
      ONCE(Logger::warn(str::format(
        "[SkyTrace.primaryMiss] expected R32_SFLOAT (",
        static_cast<uint32_t>(VK_FORMAT_R32_SFLOAT),
        "), got ", static_cast<uint32_t>(pvzFormat), " — readback skipped.")));
      return;
    }

    // NV-DXVK [SkyTrace.primaryMiss]: widened from 128x128 centered tile to
    // 512x512 so the sample covers a meaningful chunk of both sky (upper
    // band) and ground (lower band). The async readback callback splits
    // the tile into 3 horizontal bands (top/mid/bot) and reports miss
    // counts per band, so we can tell whether the rays that target the
    // sky region actually miss into the sky-trace miss shader, or hit
    // some far-out geometry (sub-view extent / dome geo / pak content)
    // that's intercepting them. 512x512 R32_SFLOAT = 1 MB readback per
    // frame, comparable to the existing GpuPrint buffer — fine.
    constexpr uint32_t kTile = 512;
    const uint32_t tileW = std::min(kTile, pvzExtent.width);
    const uint32_t tileH = std::min(kTile, pvzExtent.height);
    if (tileW == 0 || tileH == 0) {
      return;
    }
    const int32_t offX = static_cast<int32_t>((pvzExtent.width  - tileW) / 2);
    const int32_t offY = static_cast<int32_t>((pvzExtent.height - tileH) / 2);

    constexpr uint32_t kBytesPerTexel = 4;  // R32_SFLOAT
    const VkDeviceSize tileBytes = VkDeviceSize(kBytesPerTexel) * tileW * tileH;

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size   = tileBytes;
    bufInfo.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT   | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer,
      "Primary Miss Count Readback");

    // PrimaryLinearViewZ is written by gbuffer/ray-gen as a SHADER_WRITE
    // through a storage image. Guard transfer-read with a barrier.
    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,       VK_ACCESS_TRANSFER_READ_BIT);

    VkImageSubresourceLayers subres {};
    subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel       = 0;
    subres.baseArrayLayer = 0;
    subres.layerCount     = 1;

    copyImageToBuffer(
      readbackDst, /*dstOffset=*/ 0,
      kBytesPerTexel,                 // dstRowAlignment
      kBytesPerTexel * tileW,         // dstSliceAlignment
      rtOutput.m_primaryLinearViewZ.image, subres,
      VkOffset3D { offX, offY, 0 },
      VkExtent3D { tileW, tileH, 1u });

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_primaryMissCountReadback.signalValue;
    this->signal(m_primaryMissCountReadback.signal, syncValue);

    const uint32_t frameIdx = m_device->getCurrentFrameId();
    Rc<sync::Fence> signalRef = m_primaryMissCountReadback.signal;
    const uint32_t cTileW = tileW;
    const uint32_t cTileH = tileH;
    const float cMissZ = missLinearViewZ;
    const uint32_t cExtentW = pvzExtent.width;
    const uint32_t cExtentH = pvzExtent.height;

    m_primaryMissCountReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef,
       frameIdx, cTileW, cTileH, cMissZ, cExtentW, cExtentH]() mutable {
        signalRef->wait(syncValue);
        const float* base = reinterpret_cast<const float*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }
        const uint32_t numTexels = cTileW * cTileH;
        uint32_t missCount = 0;
        float minZ = +INFINITY, maxZ = -INFINITY, sumZ = 0.0f;
        uint32_t finiteCount = 0;

        // Per-band breakdown (3 horizontal bands). Rows [0,bandH) =
        // top (sky), [bandH, 2*bandH) = middle, [2*bandH, tileH) = bottom.
        // missCount per band tells us where the sky-trace miss shader
        // actually fires. If topMiss > 0 but the screenshot is still
        // black at the top, the bug is "miss shader returns black"
        // (no sky source). If topMiss == 0, the bug is "geometry
        // occludes top region" and we look at finiteZ of the top band.
        const uint32_t bandH = cTileH / 3u;
        uint32_t topMiss = 0, midMiss = 0, botMiss = 0;
        uint32_t topPx  = 0, midPx  = 0, botPx  = 0;
        float topZmin = +INFINITY, topZmax = -INFINITY, topZsum = 0.0f;
        uint32_t topZcount = 0;

        for (uint32_t y = 0; y < cTileH; ++y) {
          const uint32_t band =
              (y < bandH)        ? 0u :
              (y < 2u * bandH)   ? 1u : 2u;
          for (uint32_t x = 0; x < cTileW; ++x) {
            const float z = base[y * cTileW + x];
            const bool isMiss = (z == cMissZ);
            if (isMiss) ++missCount;
            if (std::isfinite(z)) {
              if (z < minZ) minZ = z;
              if (z > maxZ) maxZ = z;
              sumZ += z;
              ++finiteCount;
            }
            if (band == 0) {
              ++topPx;
              if (isMiss) ++topMiss;
              if (std::isfinite(z)) {
                if (z < topZmin) topZmin = z;
                if (z > topZmax) topZmax = z;
                topZsum += z;
                ++topZcount;
              }
            } else if (band == 1) {
              ++midPx;
              if (isMiss) ++midMiss;
            } else {
              ++botPx;
              if (isMiss) ++botMiss;
            }
          }
        }

        const float pct    = 100.0f * float(missCount) / float(numTexels);
        const float topPct = 100.0f * float(topMiss) / float(std::max(topPx, 1u));
        const float midPct = 100.0f * float(midMiss) / float(std::max(midPx, 1u));
        const float botPct = 100.0f * float(botMiss) / float(std::max(botPx, 1u));
        const float avgZ = finiteCount > 0
          ? (sumZ / float(finiteCount))
          : std::numeric_limits<float>::quiet_NaN();
        const float topAvgZ = topZcount > 0
          ? (topZsum / float(topZcount))
          : std::numeric_limits<float>::quiet_NaN();
        Logger::info(str::format(
          "[SkyTrace.primaryMiss] frame=", frameIdx,
          " missZSentinel=", cMissZ,
          " tile=", cTileW, "x", cTileH,
          " (centered in ", cExtentW, "x", cExtentH, ")",
          " missCount=", missCount, "/", numTexels,
          " (", pct, "%)",
          " finiteZ_min=", minZ,
          " finiteZ_max=", maxZ,
          " finiteZ_avg=", avgZ,
          " | bandMiss top=", topMiss, "/", topPx, " (", topPct, "%)",
          " mid=", midMiss, "/", midPx, " (", midPct, "%)",
          " bot=", botMiss, "/", botPx, " (", botPct, "%)",
          " | topBand finiteZ_min=", topZmin,
          " max=", topZmax,
          " avg=", topAvgZ));
      }));
  }

  // [SkyTrace.skyVerts] copies a tile of a sky draw's position buffer to
  // host memory, async-decodes the 21/21/22-bit packed format, applies the
  // captured objectToWorld matrix, and logs the world-space AABB across
  // sampled vertices. Caller throttles which draws fire this.
  void RtxContext::recordSkyDrawPositionsReadback(
      const DrawCallState& drawCallState,
      uint32_t frameId, uint32_t drawIdx) {
    const RasterGeometry& g = drawCallState.getGeometryData();
    if (!g.positionBuffer.defined() || g.vertexCount == 0u) {
      return;
    }
    if (g.positionBuffer.vertexFormat() != VK_FORMAT_R32G32_UINT) {
      // Other formats already worked CPU-side via mapPtr or aren't TF2's
      // packed sky path — nothing to do here.
      return;
    }

    // Reap completed tasks.
    {
      auto& tasks = m_skyVertsReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
        [](std::future<void>& f) {
          return !f.valid()
              || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), tasks.end());
    }

    Rc<DxvkBuffer> srcBuf = g.positionBuffer.buffer();
    if (srcBuf == nullptr) {
      return;
    }
    const uint32_t stride = g.positionBuffer.stride();
    if (stride < 8u) {
      return;
    }
    const VkDeviceSize sliceOffset = g.positionBuffer.offset();
    const VkDeviceSize attrOffset  = g.positionBuffer.offsetFromSlice();
    const VkDeviceSize srcOffsetBytes = sliceOffset + attrOffset;

    // Cap to keep per-frame readback bounded. 1024 verts × 28 B = 28 KB.
    constexpr uint32_t kMaxVertsToCopy = 1024u;
    const uint32_t nVerts = std::min<uint32_t>(g.vertexCount, kMaxVertsToCopy);
    const VkDeviceSize copyBytes = VkDeviceSize(nVerts) * stride;

    if (srcOffsetBytes + copyBytes > srcBuf->info().size) {
      return;
    }

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size   = copyBytes;
    bufInfo.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT   | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer,
      "Sky Verts Readback");

    // The position buffer was just used as a vertex input by the prior
    // sky raster; barrier vertex-input-read → transfer-read so the copy
    // observes the latest contents.
    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,     VK_ACCESS_TRANSFER_READ_BIT);

    copyBuffer(readbackDst, /*dstOffset=*/ 0,
               srcBuf, srcOffsetBytes, copyBytes);

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_skyVertsReadback.signalValue;
    this->signal(m_skyVertsReadback.signal, syncValue);

    Rc<sync::Fence> signalRef = m_skyVertsReadback.signal;
    const Matrix4 objectToWorld = drawCallState.transformData.objectToWorld;
    const uint32_t cStride = stride;
    const uint32_t cVertCount = nVerts;
    const XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();

    m_skyVertsReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef,
       frameId, drawIdx, cStride, cVertCount, objectToWorld, matHash]() mutable {
        signalRef->wait(syncValue);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }

        // 21/21/22-bit decode (mirrors interleave_geometry.h:303-327).
        // u0 lower 21 bits = X int, u0 upper 11 + u1 lower 10 = Y int (21
        // total), u1 upper 22 = Z int. Scale = 1/1024, biases = -1024, -1024,
        // -2048.
        auto decode = [](uint32_t u0, uint32_t u1) -> std::array<float, 3> {
          const uint32_t xi = u0 & 0x001FFFFFu;
          const uint32_t yi = ((u0 >> 21u) & 0x7FFu) | ((u1 & 0x3FFu) << 11u);
          const uint32_t zi = u1 >> 10u;
          const float kScale = 1.0f / 1024.0f;
          return {
            float(xi) * kScale - 1024.0f,
            float(yi) * kScale - 1024.0f,
            float(zi) * kScale - 2048.0f
          };
        };

        float minObj[3] = { +INFINITY, +INFINITY, +INFINITY };
        float maxObj[3] = { -INFINITY, -INFINITY, -INFINITY };
        float minWorld[3] = { +INFINITY, +INFINITY, +INFINITY };
        float maxWorld[3] = { -INFINITY, -INFINITY, -INFINITY };

        for (uint32_t v = 0; v < cVertCount; ++v) {
          uint32_t u0, u1;
          memcpy(&u0, base + cStride * v + 0, 4);
          memcpy(&u1, base + cStride * v + 4, 4);
          const auto p = decode(u0, u1);
          for (int i = 0; i < 3; ++i) {
            if (p[i] < minObj[i]) minObj[i] = p[i];
            if (p[i] > maxObj[i]) maxObj[i] = p[i];
          }
          // World position = objectToWorld * (px, py, pz, 1). Matrix4 is
          // column-major; access as m[col][row].
          const float wx = objectToWorld[0][0]*p[0] + objectToWorld[1][0]*p[1] + objectToWorld[2][0]*p[2] + objectToWorld[3][0];
          const float wy = objectToWorld[0][1]*p[0] + objectToWorld[1][1]*p[1] + objectToWorld[2][1]*p[2] + objectToWorld[3][1];
          const float wz = objectToWorld[0][2]*p[0] + objectToWorld[1][2]*p[1] + objectToWorld[2][2]*p[2] + objectToWorld[3][2];
          if (wx < minWorld[0]) minWorld[0] = wx;
          if (wx > maxWorld[0]) maxWorld[0] = wx;
          if (wy < minWorld[1]) minWorld[1] = wy;
          if (wy > maxWorld[1]) maxWorld[1] = wy;
          if (wz < minWorld[2]) minWorld[2] = wz;
          if (wz > maxWorld[2]) maxWorld[2] = wz;
        }

        Logger::info(str::format(
          "[SkyTrace.skyVerts] frame=", frameId,
          " drawIdx=", drawIdx,
          " matHash=0x", std::hex, matHash, std::dec,
          " sampledVerts=", cVertCount,
          " objAABB=(", minObj[0], ",", minObj[1], ",", minObj[2], ")->(",
                       maxObj[0], ",", maxObj[1], ",", maxObj[2], ")",
          " worldAABB=(", minWorld[0], ",", minWorld[1], ",", minWorld[2], ")->(",
                         maxWorld[0], ",", maxWorld[1], ",", maxWorld[2], ")"));
      }));
  }

  // [SkyTrace.probeContent] async readback of all 6 SkyProbe cube faces.
  // Copies a 32x32 center tile from each face into a single host buffer
  // (192x32 layout: face 0..5 stacked horizontally). Worker decodes each
  // face and logs avg/min/max RGB + center texel. The cube is sampled by
  // IBL/PSR for world surfaces in Hybrid mode when skyProbePopulated==1,
  // so this directly answers "what color is the cube tinting world
  // geometry with?" — the prime suspect for the altitude-correlated
  // yellow/dim-blue regression.
  // Cube face order: +X, -X, +Y, -Y, +Z, -Z.
  void RtxContext::recordSkyProbeReadback() {
    Resources::Resource probe = getResourceManager().getSkyProbe(this);
    if (probe.image == nullptr) {
      return;
    }

    // Reap completed tasks.
    {
      auto& tasks = m_skyProbeReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
        [](std::future<void>& f) {
          return !f.valid()
              || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), tasks.end());
    }

    const VkExtent3D probeExtent = probe.image->info().extent;
    const VkFormat probeFormat = probe.image->info().format;

    uint32_t bytesPerTexel = 0;
    if (probeFormat == VK_FORMAT_R8G8B8A8_SRGB
     || probeFormat == VK_FORMAT_R8G8B8A8_UNORM) {
      bytesPerTexel = 4;
    } else if (probeFormat == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
      bytesPerTexel = 4;
    } else if (probeFormat == VK_FORMAT_R16G16B16A16_SFLOAT
            || probeFormat == VK_FORMAT_R16G16B16A16_UNORM) {
      bytesPerTexel = 8;
    } else {
      ONCE(Logger::warn(str::format(
        "[SkyTrace.probeContent] unsupported probe format=",
        static_cast<uint32_t>(probeFormat),
        " — readback skipped. Add a decoder to recordSkyProbeReadback.")));
      return;
    }

    constexpr uint32_t kFaceTile = 32;
    constexpr uint32_t kNumFaces = 6;
    const uint32_t tileW = std::min(kFaceTile, probeExtent.width);
    const uint32_t tileH = std::min(kFaceTile, probeExtent.height);
    if (tileW == 0 || tileH == 0) {
      return;
    }
    const int32_t offX = static_cast<int32_t>((probeExtent.width  - tileW) / 2);
    const int32_t offY = static_cast<int32_t>((probeExtent.height - tileH) / 2);

    const VkDeviceSize faceTileBytes = VkDeviceSize(bytesPerTexel) * tileW * tileH;
    const VkDeviceSize totalBytes    = faceTileBytes * kNumFaces;

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size   = totalBytes;
    bufInfo.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT   | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer,
      "Sky Probe Readback");

    // Probe is written by both: (a) rasterizeToSkyProbe color-attachment
    // writes from TF2's per-face draws, AND (b) the cube prefill compute
    // shader. Guard transfer-read with a barrier covering BOTH source
    // stages — using only COLOR_ATTACHMENT_OUTPUT was missing the
    // compute-shader writes on faces (3, 4, 5) where TF2 doesn't draw,
    // so the readback was reading stale memory and reporting zero.
    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    for (uint32_t face = 0; face < kNumFaces; ++face) {
      VkImageSubresourceLayers subres {};
      subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subres.mipLevel       = 0;
      subres.baseArrayLayer = face;
      subres.layerCount     = 1;

      copyImageToBuffer(
        readbackDst, /*dstOffset=*/ faceTileBytes * face,
        bytesPerTexel,                 // dstRowAlignment
        bytesPerTexel * tileW,         // dstSliceAlignment
        probe.image, subres,
        VkOffset3D { offX, offY, 0 },
        VkExtent3D { tileW, tileH, 1u });
    }

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_skyProbeReadback.signalValue;
    this->signal(m_skyProbeReadback.signal, syncValue);

    const uint32_t frameIdx = m_device->getCurrentFrameId();
    Rc<sync::Fence> signalRef = m_skyProbeReadback.signal;
    const uint32_t fmtTag = static_cast<uint32_t>(probeFormat);
    const uint32_t cTileW = tileW;
    const uint32_t cTileH = tileH;

    m_skyProbeReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef,
       frameIdx, fmtTag, cTileW, cTileH]() mutable {
        signalRef->wait(syncValue);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }

        auto srgbToLinear = [](float c) -> float {
          if (c <= 0.04045f) return c / 12.92f;
          return std::pow((c + 0.055f) / 1.055f, 2.4f);
        };

        auto unpackR11G11B10 = [](uint32_t bits) -> std::array<float, 3> {
          auto unpack11 = [](uint32_t v) -> float {
            const uint32_t exp  = (v >> 6) & 0x1Fu;
            const uint32_t mant = v & 0x3Fu;
            if (exp == 0u) {
              return mant == 0u ? 0.0f : (float(mant) / 64.0f) * std::pow(2.0f, -14.0f);
            }
            if (exp == 31u) {
              return mant == 0u ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
            }
            return (1.0f + float(mant) / 64.0f) * std::pow(2.0f, float(exp) - 15.0f);
          };
          auto unpack10 = [](uint32_t v) -> float {
            const uint32_t exp  = (v >> 5) & 0x1Fu;
            const uint32_t mant = v & 0x1Fu;
            if (exp == 0u) {
              return mant == 0u ? 0.0f : (float(mant) / 32.0f) * std::pow(2.0f, -14.0f);
            }
            if (exp == 31u) {
              return mant == 0u ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
            }
            return (1.0f + float(mant) / 32.0f) * std::pow(2.0f, float(exp) - 15.0f);
          };
          const uint32_t r = bits & 0x7FFu;
          const uint32_t g = (bits >> 11) & 0x7FFu;
          const uint32_t b = (bits >> 22) & 0x3FFu;
          return { unpack11(r), unpack11(g), unpack10(b) };
        };

        // IEEE half->float, copied from recordAerialPerspectiveLutReadback.
        auto h2f = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t exp  = (h & 0x7C00u) >> 10;
          const uint32_t mant = (h & 0x03FFu);
          uint32_t bits;
          if (exp == 0) {
            bits = sign;
            if (mant != 0) {
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

        const bool isSrgb = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_SRGB));
        const bool isUnorm = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
        const bool is11_11_10 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_B10G11R11_UFLOAT_PACK32));
        const bool isHalf4 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT));
        const bool isUnorm16x4 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_UNORM));
        const uint32_t bppFmt = (isHalf4 || isUnorm16x4) ? 8u : 4u;

        constexpr uint32_t kNumFaces = 6;
        const char* kFaceNames[kNumFaces] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };
        const uint32_t numTexels = cTileW * cTileH;
        const uint32_t bytesPerFace = bppFmt * cTileW * cTileH;

        for (uint32_t face = 0; face < kNumFaces; ++face) {
          const uint8_t* faceBase = base + face * bytesPerFace;
          float minR = +INFINITY, maxR = -INFINITY, sumR = 0.0f;
          float minG = +INFINITY, maxG = -INFINITY, sumG = 0.0f;
          float minB = +INFINITY, maxB = -INFINITY, sumB = 0.0f;
          std::array<float, 3> centerLinear{ 0.0f, 0.0f, 0.0f };
          const uint32_t cx = cTileW / 2;
          const uint32_t cy = cTileH / 2;

          for (uint32_t y = 0; y < cTileH; ++y) {
            for (uint32_t x = 0; x < cTileW; ++x) {
              const uint32_t i = y * cTileW + x;
              float r = 0.0f, g = 0.0f, b = 0.0f;
              if (isSrgb || isUnorm) {
                const uint8_t* p = faceBase + i * 4u;
                const float rRaw = float(p[0]) / 255.0f;
                const float gRaw = float(p[1]) / 255.0f;
                const float bRaw = float(p[2]) / 255.0f;
                r = isSrgb ? srgbToLinear(rRaw) : rRaw;
                g = isSrgb ? srgbToLinear(gRaw) : gRaw;
                b = isSrgb ? srgbToLinear(bRaw) : bRaw;
              } else if (is11_11_10) {
                uint32_t bits;
                memcpy(&bits, faceBase + i * 4u, 4);
                const auto rgb = unpackR11G11B10(bits);
                r = rgb[0]; g = rgb[1]; b = rgb[2];
              } else if (isHalf4) {
                const uint16_t* h = reinterpret_cast<const uint16_t*>(faceBase + i * 8u);
                r = h2f(h[0]);
                g = h2f(h[1]);
                b = h2f(h[2]);
              } else if (isUnorm16x4) {
                // 16-bit UNORM: integer value in [0, 65535] -> [0, 1] linear.
                const uint16_t* h = reinterpret_cast<const uint16_t*>(faceBase + i * 8u);
                r = float(h[0]) / 65535.0f;
                g = float(h[1]) / 65535.0f;
                b = float(h[2]) / 65535.0f;
              }
              if (r < minR) minR = r; if (r > maxR) maxR = r; sumR += r;
              if (g < minG) minG = g; if (g > maxG) maxG = g; sumG += g;
              if (b < minB) minB = b; if (b > maxB) maxB = b; sumB += b;
              if (x == cx && y == cy) {
                centerLinear = { r, g, b };
              }
            }
          }

          const float invN = 1.0f / static_cast<float>(numTexels);
          Logger::info(str::format(
            "[SkyTrace.probeContent] frame=", frameIdx,
            " face=", face, " (", kFaceNames[face], ")",
            " fmt=", fmtTag,
            " tile=", cTileW, "x", cTileH,
            " avgLinear=(", sumR * invN, ",", sumG * invN, ",", sumB * invN, ")",
            " minLinear=(", minR, ",", minG, ",", minB, ")",
            " maxLinear=(", maxR, ",", maxG, ",", maxB, ")",
            " centerLinear=(", centerLinear[0], ",", centerLinear[1], ",", centerLinear[2], ")"));
        }
      }));
  }

  // [SkyTrace.matteContent] async readback of the sky-matte image at
  // composite-bind time. Copies a 64x64 center tile, decodes on a
  // worker thread, logs min/avg/max RGB + the center texel. Used to
  // confirm GPU-visible pixel content matches our white-clear test.
  // Throttled by the caller (rtx_composite.cpp).
  void RtxContext::recordSkyMatteReadback() {
    Resources::Resource matte = getResourceManager().getSkyMatte(this);
    if (matte.image == nullptr) {
      return;
    }

    // Reap completed tasks.
    {
      auto& tasks = m_skyMatteReadback.asyncTasks;
      tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
        [](std::future<void>& f) {
          return !f.valid()
              || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), tasks.end());
    }

    const VkExtent3D matteExtent = matte.image->info().extent;
    const VkFormat matteFormat = matte.image->info().format;

    // Decode supported formats only. Matte is one of: R8G8B8A8_SRGB (43)
    // / R8G8B8A8_UNORM (37) / B10G11R11_UFLOAT_PACK32 (122) /
    // R16G16B16A16_UNORM (91) / R16G16B16A16_SFLOAT (97), depending on
    // skyForceHDR() and the per-game format choice.
    uint32_t bytesPerTexel = 0;
    if (matteFormat == VK_FORMAT_R8G8B8A8_SRGB
     || matteFormat == VK_FORMAT_R8G8B8A8_UNORM) {
      bytesPerTexel = 4;
    } else if (matteFormat == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
      bytesPerTexel = 4;
    } else if (matteFormat == VK_FORMAT_R16G16B16A16_UNORM
            || matteFormat == VK_FORMAT_R16G16B16A16_SFLOAT) {
      bytesPerTexel = 8;
    } else {
      ONCE(Logger::warn(str::format(
        "[SkyTrace.matteContent] unsupported matte format=",
        static_cast<uint32_t>(matteFormat),
        " — readback skipped. Add a decoder to recordSkyMatteReadback.")));
      return;
    }

    constexpr uint32_t kTile = 64;
    const uint32_t tileW = std::min(kTile, matteExtent.width);
    const uint32_t tileH = std::min(kTile, matteExtent.height);
    if (tileW == 0 || tileH == 0) {
      return;
    }
    // Center the tile in the matte.
    const int32_t offX = static_cast<int32_t>((matteExtent.width  - tileW) / 2);
    const int32_t offY = static_cast<int32_t>((matteExtent.height - tileH) / 2);

    const VkDeviceSize tileBytes = VkDeviceSize(bytesPerTexel) * tileW * tileH;

    DxvkBufferCreateInfo bufInfo {};
    bufInfo.size   = tileBytes;
    bufInfo.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    bufInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT   | VK_ACCESS_HOST_READ_BIT;
    const VkMemoryPropertyFlags memType =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    Rc<DxvkBuffer> readbackDst = m_device->createBuffer(
      bufInfo, memType, DxvkMemoryStats::Category::RTXBuffer,
      "Sky Matte Readback");

    // The matte is written either by TF2's sky raster (color-attachment
    // write) or by our injectRTX clearRenderTarget. Both end as
    // COLOR_ATTACHMENT_OUTPUT writes, so guard transfer-read with a
    // matching barrier.
    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,                VK_ACCESS_TRANSFER_READ_BIT);

    VkImageSubresourceLayers subres {};
    subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel       = 0;
    subres.baseArrayLayer = 0;
    subres.layerCount     = 1;

    copyImageToBuffer(
      readbackDst, /*dstOffset=*/ 0,
      bytesPerTexel,                 // dstRowAlignment
      bytesPerTexel * tileW,         // dstSliceAlignment
      matte.image, subres,
      VkOffset3D { offX, offY, 0 },
      VkExtent3D { tileW, tileH, 1u });

    this->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,     VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_skyMatteReadback.signalValue;
    this->signal(m_skyMatteReadback.signal, syncValue);

    const uint32_t frameIdx = m_device->getCurrentFrameId();
    Rc<sync::Fence> signalRef = m_skyMatteReadback.signal;
    const uint32_t fmtTag = static_cast<uint32_t>(matteFormat);
    const uint32_t cTileW = tileW;
    const uint32_t cTileH = tileH;

    m_skyMatteReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst), syncValue, signalRef,
       frameIdx, fmtTag, cTileW, cTileH]() mutable {
        signalRef->wait(syncValue);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }

        // sRGB -> linear (IEC 61966-2-1).
        auto srgbToLinear = [](float c) -> float {
          if (c <= 0.04045f) return c / 12.92f;
          return std::pow((c + 0.055f) / 1.055f, 2.4f);
        };

        // Decode B10G11R11_UFLOAT_PACK32 → 3 floats. Bits: R 11 (mantissa 6, exp 5),
        // G 11 (same), B 10 (mantissa 5, exp 5). Unsigned, no sign bit.
        auto unpackR11G11B10 = [](uint32_t bits) -> std::array<float, 3> {
          auto unpack11 = [](uint32_t v) -> float {
            const uint32_t exp  = (v >> 6) & 0x1Fu;
            const uint32_t mant = v & 0x3Fu;
            if (exp == 0u) {
              return mant == 0u ? 0.0f : (float(mant) / 64.0f) * std::pow(2.0f, -14.0f);
            }
            if (exp == 31u) {
              return mant == 0u ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
            }
            return (1.0f + float(mant) / 64.0f) * std::pow(2.0f, float(exp) - 15.0f);
          };
          auto unpack10 = [](uint32_t v) -> float {
            const uint32_t exp  = (v >> 5) & 0x1Fu;
            const uint32_t mant = v & 0x1Fu;
            if (exp == 0u) {
              return mant == 0u ? 0.0f : (float(mant) / 32.0f) * std::pow(2.0f, -14.0f);
            }
            if (exp == 31u) {
              return mant == 0u ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
            }
            return (1.0f + float(mant) / 32.0f) * std::pow(2.0f, float(exp) - 15.0f);
          };
          const uint32_t r = bits & 0x7FFu;
          const uint32_t g = (bits >> 11) & 0x7FFu;
          const uint32_t b = (bits >> 22) & 0x3FFu;
          return { unpack11(r), unpack11(g), unpack10(b) };
        };

        float minR = +INFINITY, maxR = -INFINITY, sumR = 0.0f;
        float minG = +INFINITY, maxG = -INFINITY, sumG = 0.0f;
        float minB = +INFINITY, maxB = -INFINITY, sumB = 0.0f;
        const uint32_t numTexels = cTileW * cTileH;
        std::array<float, 3> centerLinear{ 0.0f, 0.0f, 0.0f };
        std::array<float, 3> centerRaw{ 0.0f, 0.0f, 0.0f };
        const uint32_t cx = cTileW / 2;
        const uint32_t cy = cTileH / 2;

        // IEEE half->float for R16G16B16A16_SFLOAT path.
        auto h2f = [](uint16_t h) -> float {
          const uint32_t sign = (h & 0x8000u) << 16;
          const uint32_t exp  = (h & 0x7C00u) >> 10;
          const uint32_t mant = (h & 0x03FFu);
          uint32_t bits;
          if (exp == 0) {
            bits = sign;
            if (mant != 0) {
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

        const bool isSrgb = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_SRGB));
        const bool isUnorm = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
        const bool is11_11_10 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_B10G11R11_UFLOAT_PACK32));
        const bool isUnorm16x4 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_UNORM));
        const bool isHalf4 = (fmtTag == static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT));
        const uint32_t bppLocal = (isUnorm16x4 || isHalf4) ? 8u : 4u;

        for (uint32_t y = 0; y < cTileH; ++y) {
          for (uint32_t x = 0; x < cTileW; ++x) {
            const uint32_t i = y * cTileW + x;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            float rRaw = 0.0f, gRaw = 0.0f, bRaw = 0.0f;
            if (isSrgb || isUnorm) {
              const uint8_t* p = base + i * 4u;
              rRaw = float(p[0]) / 255.0f;
              gRaw = float(p[1]) / 255.0f;
              bRaw = float(p[2]) / 255.0f;
              r = isSrgb ? srgbToLinear(rRaw) : rRaw;
              g = isSrgb ? srgbToLinear(gRaw) : gRaw;
              b = isSrgb ? srgbToLinear(bRaw) : bRaw;
            } else if (is11_11_10) {
              uint32_t bits;
              memcpy(&bits, base + i * 4u, 4);
              const auto rgb = unpackR11G11B10(bits);
              r = rgb[0]; g = rgb[1]; b = rgb[2];
              rRaw = r; gRaw = g; bRaw = b;
            } else if (isUnorm16x4) {
              const uint16_t* h = reinterpret_cast<const uint16_t*>(base + i * 8u);
              r = float(h[0]) / 65535.0f;
              g = float(h[1]) / 65535.0f;
              b = float(h[2]) / 65535.0f;
              rRaw = r; gRaw = g; bRaw = b;
            } else if (isHalf4) {
              const uint16_t* h = reinterpret_cast<const uint16_t*>(base + i * 8u);
              r = h2f(h[0]);
              g = h2f(h[1]);
              b = h2f(h[2]);
              rRaw = r; gRaw = g; bRaw = b;
            }
            if (r < minR) minR = r; if (r > maxR) maxR = r; sumR += r;
            if (g < minG) minG = g; if (g > maxG) maxG = g; sumG += g;
            if (b < minB) minB = b; if (b > maxB) maxB = b; sumB += b;
            if (x == cx && y == cy) {
              centerLinear = { r, g, b };
              centerRaw    = { rRaw, gRaw, bRaw };
            }
          }
        }

        const float invN = 1.0f / static_cast<float>(numTexels);
        Logger::info(str::format(
          "[SkyTrace.matteContent] frame=", frameIdx,
          " fmt=", fmtTag,
          " tile=", cTileW, "x", cTileH,
          " avgLinear=(", sumR * invN, ",", sumG * invN, ",", sumB * invN, ")",
          " minLinear=(", minR, ",", minG, ",", minB, ")",
          " maxLinear=(", maxR, ",", maxG, ",", maxB, ")",
          " centerLinear=(", centerLinear[0], ",", centerLinear[1], ",", centerLinear[2], ")",
          " centerRaw=(", centerRaw[0], ",", centerRaw[1], ",", centerRaw[2], ")"));
      }));
  }

  void RtxContext::endFrame(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage, bool callInjectRtx) {

    if (callInjectRtx) {
      // Fallback inject (is a no-op if already injected this frame, or no valid RT scene)
      injectRTX(cachedReflexFrameId, targetImage);
    }

#ifdef REMIX_DEVELOPMENT
    queryAvailableResourceAliasing();
    analyzeResourceAliasing();
    clearResourceAliasingCache();
#endif

    // Update time on the frame end so all other systems can benefit from a global time
    GlobalTime::get().update();
  }

  // Called right before present
  void RtxContext::onPresent(Rc<DxvkImage> targetImage) {
    // If injectRTX couldn't screenshot a final image or a pre-present screenshot is requested,
    // take a screenshot of a present image (with UI and others)
    {
      const bool isRaytracingEnabled = RtxOptions::enableRaytracing();
      const bool isCameraValid = getSceneManager().getCamera().isValid(m_device->getCurrentFrameId());

      if (!isRaytracingEnabled || !isCameraValid || s_capturePrePresentTestScreenshot) {
        const bool captureTestScreenshot = (m_screenshotFrameEnabled && m_device->getCurrentFrameId() == m_screenshotFrameNum);
        const bool captureDxvkScreenImage = s_triggerScreenshot || captureTestScreenshot;
        if (captureDxvkScreenImage) {
          if (targetImage == nullptr) {
            targetImage = m_state.om.renderTargets.color[0].view->image();
          }
          takeScreenshot("rtxImageDxvkView", targetImage);
        }
      }
    }
    s_triggerScreenshot = false;

    // Some time in the future kill process
    if (m_triggerDelayedTerminate &&
        (m_device->getCurrentFrameId() > m_terminateAppFrameNum) &&
        m_common->capturer()->isIdle()) {
      Logger::info(str::format("RTX: Terminating application"));
      Metrics::serialize();
      getCommonObjects()->metaExporter().waitForAllExportsToComplete();

      env::killProcess();
    }

    // This needs to happen at the end of frame, after ImGUI rendering
    GpuMemoryTracker::onFrameEnd();
  }

  void RtxContext::updateMetrics(const float gpuIdleTimeMilliseconds) const {
    ScopedCpuProfileZone();
    Metrics::logRollingAverage(Metric::dxvk_average_frame_time_ms, GlobalTime::get().deltaTimeMs()); // In milliseconds
    Metrics::logRollingAverage(Metric::dxvk_gpu_idle_time_ms, gpuIdleTimeMilliseconds); // In milliseconds
    uint64_t vidUsageMib = 0;
    uint64_t sysUsageMib = 0;
    const VkPhysicalDeviceMemoryProperties memprops = m_device->adapter()->memoryProperties();
    // Calc memory usage
    for (uint32_t i = 0; i < memprops.memoryHeapCount; i++) {
      bool isDeviceLocal = memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

      if (isDeviceLocal) {
        vidUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
      else {
        sysUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
    }
    Metrics::logRollingAverage(Metric::dxvk_vid_memory_usage_mb, static_cast<float>(vidUsageMib)); // In MB
    Metrics::logRollingAverage(Metric::dxvk_sys_memory_usage_mb, static_cast<float>(sysUsageMib)); // In MB
    Metrics::logFloat(Metric::dxvk_total_time_ms, static_cast<float>(GlobalTime::get().realTimeSinceStartMs()));
    Metrics::logFloat(Metric::dxvk_frame_count, static_cast<float>(m_device->getCurrentFrameId()));
  }

  void RtxContext::addLights(const RtxLegacyLight* pLights, const uint32_t numLights) {
    for (uint32_t i = 0; i < numLights; i++) {
      getSceneManager().addLight(pLights[i]);
    }
  }

  void RtxContext::commitGeometryToRT(const DrawParameters& params, DrawCallState& drawCallState){
    ScopedCpuProfileZone();

    RasterGeometry& geoData = drawCallState.geometryData;
    DrawCallTransforms& transformData = drawCallState.transformData;

    // NV-DXVK [FlagTrace.P3]: point 3 of three — at the entry to
    // commitGeometryToRT, inside the EmitCs lambda on the CS thread.
    // This is the LAST observation point before updateInstance runs.
    // If P2 (pre-EmitCs in SubmitDraw) shows ignAC=1 but P3 shows
    // ignAC=0, the flag is being lost in the lambda capture or the
    // DxvkContext command-stream replay. Scoped to VS_2904d2, first
    // 4/frame — matching P1/P2 so the three lines can be diffed.
    {
      const auto vsHashP3 = transformData.vertexShaderHash;
      if (vsHashP3 == 0x2904d2163ef31a17ull) {
        thread_local uint32_t sP3Frame = UINT32_MAX;
        thread_local uint32_t sP3Count = 0;
        const uint32_t curFP3 = m_device->getCurrentFrameId();
        if (sP3Frame != curFP3) { sP3Frame = curFP3; sP3Count = 0; }
        if (sP3Count < 4u) {
          sP3Count += 1;
          Logger::info(str::format(
            "[FlagTrace.P3.CommitEntry] vs=0x", std::hex, vsHashP3, std::dec,
            " f=", curFP3,
            " ignAC=", (drawCallState.getCategoryFlags().test(
                          InstanceCategories::IgnoreAntiCulling) ? 1 : 0),
            " propId=0x", std::hex, transformData.stablePropId, std::dec,
            " catRaw=0x", std::hex,
              static_cast<uint64_t>(drawCallState.getCategoryFlags().raw()), std::dec));
        }
      }
    }

    // NV-DXVK [CommitVsTrace]: log EVERY distinct VS hash that reaches
    // commitGeometryToRT once per session. Goal: identify which sub-
    // view VS hashes (visible in d3d11_rtx [PropIdTrace] log but absent
    // from instance manager's [InstCounts] log) actually arrive here
    // vs. which get dropped earlier inside d3d11_rtx::SubmitDraw.
    //
    // Each VS hash logs ONCE per session. Pair this with the existing
    // [PropIdTrace] (3340 hits for VS_1baf78e08c4e8fed) and [InstCounts]
    // (0 hits for the same) — if VS_1baf78 appears in [CommitVsTrace],
    // the rejection is downstream in commitGeometryToRT or
    // processDrawCallState. If it doesn't appear, the rejection is
    // upstream in d3d11_rtx::SubmitDraw (the EmitCs/lambda block at
    // d3d11_rtx.cpp:15208 may be conditionally skipped, OR the draw
    // gets routed through a non-RT path entirely).
    //
    // Also captures the IgnoreAntiCulling flag and the categoryFlags
    // bitset, so we can verify the reproject-branch's setCategory
    // applied to this drawcall.
    {
      static std::mutex sCommitVsMu;
      static std::unordered_set<uint64_t> sCommitVsSeen;
      const uint64_t vsH = static_cast<uint64_t>(transformData.vertexShaderHash);
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sCommitVsMu);
        first = sCommitVsSeen.insert(vsH).second;
      }
      if (first) {
        const bool hasIAC = drawCallState.getCategoryFlags().test(
          InstanceCategories::IgnoreAntiCulling);
        Logger::info(str::format(
          "[CommitVsTrace] vs=0x", std::hex, vsH, std::dec,
          " camType=", static_cast<uint32_t>(drawCallState.cameraType),
          " ignAntiCull=", (hasIAC ? 1 : 0),
          " catFlags=0x", std::hex,
            static_cast<uint64_t>(drawCallState.getCategoryFlags().raw()),
            std::dec,
          " stablePropId=0x", std::hex, transformData.stablePropId, std::dec,
          " verts=", geoData.vertexCount,
          " — first sighting at commitGeometryToRT"));
      }
    }

    // NV-DXVK: UV-transform pipeline diag — did the non-identity matrix set
    // by D3D11Rtx survive the EmitCs lambda-capture boundary?
    {
      static uint32_t sCommitUVx = 0;
      if (sCommitUVx < 20) {
        const auto& m = transformData.textureTransform;
        if (m != Matrix4()) {
          ++sCommitUVx;
          Logger::info(str::format(
            "[RtxCtx.UVx] commitGeometryToRT received non-identity xform "
            "col0=(", m.data[0].x, ",", m.data[0].y, ") ",
            "col1=(", m.data[1].x, ",", m.data[1].y, ") ",
            "col3=(", m.data[3].x, ",", m.data[3].y, ")"));
        }
      }
    }

    assert(geoData.futureGeometryHashes.valid());
    assert(geoData.positionBuffer.defined());


    const auto fusedMode = RtxOptions::fusedWorldViewMode();
    if (unlikely(fusedMode != FusedWorldViewMode::None)) {
      if (fusedMode == FusedWorldViewMode::View) {
        // Set World from WorldView transform
        transformData.objectToWorld = transformData.objectToView;
        // Set camera to identity
        transformData.worldToView = Matrix4();
      } else if (fusedMode == FusedWorldViewMode::World) {
        // Nothing to do...
      }
    }

    auto& cameraManager = getSceneManager().getCameraManager();

    // TODO: a last camera is used to finalize skinning...
    // processCameraData can be called only after finalizePendingFutures,
    // as we need geometry hash to check sky geometries
    const RtCamera* lastCamera =
      cameraManager.isCameraValid(cameraManager.getLastSetCameraType())
        ? &cameraManager.getCamera(cameraManager.getLastSetCameraType())
        : nullptr;

    // NV-DXVK [WidowCam]: WHY pLastCamera is the wrong camera. finalizeSkinning
    // (in finalizePendingFutures below) rebuilds the Widow's o2w from
    // `lastCamera` = the LAST-SET camera type — a global last-wins state (see
    // the TODO above), NOT the draw's own camera. [WidowBake] showed that
    // camera's w2v=(7107,-6387,2797) teleports the ship while the draw's own
    // w2v=(-15189,...) is correct. Dump: which type is last-set, the draw's own
    // w2v (captured BEFORE finalize overwrites it), and every valid camera's
    // w2v — so we can see which camera type actually matches the draw (= the
    // one the Widow SHOULD be finalized against). Raw, capped, by-model gated.
    if (drawCallState.isWidowModel) {
      static uint32_t s_widowCamN = 0;
      if (s_widowCamN < 160u) {
        ++s_widowCamN;
        const Matrix4& drawW2v = drawCallState.getTransformData().worldToView;
        const CameraType::Enum lastType = cameraManager.getLastSetCameraType();
        std::string camDump;
        for (uint32_t ci = 0; ci < CameraType::Count; ++ci) {
          const CameraType::Enum ct = static_cast<CameraType::Enum>(ci);
          if (!cameraManager.isCameraValid(ct)) continue;
          const Matrix4 w2v = cameraManager.getCamera(ct).getWorldToView(false);
          camDump += str::format(" [", ci, "]=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")");
        }
        Logger::info(str::format(
          "[WidowCam] n=", s_widowCamN,
          " lastSetType=", uint32_t(lastType),
          " lastCamValid=", (lastCamera != nullptr ? 1 : 0),
          " drawW2vT=(", drawW2v[3][0], ",", drawW2v[3][1], ",", drawW2v[3][2], ")",
          " validCamsW2vT:", camDump));
      }
    }

    // Sync any pending work with geometry processing threads
    if (drawCallState.finalizePendingFutures(lastCamera)) {
      drawCallState.cameraType = cameraManager.processCameraData(drawCallState);

      // NV-DXVK [ShipDraw]: per-DRAW world AABB + camera classification for the hull shaders,
      // logged BEFORE the Unknown-skip / sky / merge so it reflects each individual submission
      // (the instance census only sees the post-merge batch AABB, which is useless for one ship).
      // boundingBox here is the tight per-draw object-space box (finalized just above); transform
      // it by this draw's objectToWorld for true world coords. cameraType is the classification
      // this draw just got — if the hull draw flips to Sky/Unknown in the vanish direction, that
      // reclassification IS the vanish. Diff visible vs vanished:
      //   worldCen moves       -> ship geometry teleported.
      //   worldCen same, camType flips (Main->Sky/Unknown) -> the draw is being rerouted/dropped.
      //   worldCen same, camType Main, but still vanishes -> occlusion (skybox wins) downstream.
      {
        // NV-DXVK: these two [ShipDraw]/[ShipDraw.w2v] lines fire PER ship draw (~8-17x
        // per frame each) via synchronous Logger::info disk writes — a major framerate
        // sink. Their world AABB is also unreliable (samples a device-local VB whose
        // mapPtr() is null => sampled=0 => +/-FLT_MAX). [ShipSubmit] (one aggregated line
        // per frame in rtx_scene_manager.cpp) supersedes them for the vanish question, so
        // they are gated OFF by default. Flip to true only if you need the per-draw w2v.
        static const bool kEnableShipDrawPerDraw = false;
        const uint64_t vsH = static_cast<uint64_t>(drawCallState.getTransformData().vertexShaderHash);
        if (kEnableShipDrawPerDraw && (vsH == 0x292b6ba0d1854f28ull || vsH == 0x29146e1dd50b0314ull)) {
          const Matrix4& o2wD = drawCallState.getTransformData().objectToWorld;
          // SYNCHRONOUS per-draw world AABB from the actual vertex data (no async future, no
          // batch merge). Sample up to 64 vertices, transform each by this draw's o2w, accumulate
          // world min/max. This is the ground truth for "where is this draw in the world".
          const RasterGeometry& geoD = drawCallState.getGeometryData();
          Vector3 vMin( FLT_MAX,  FLT_MAX,  FLT_MAX);
          Vector3 vMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
          uint32_t sampled = 0u;
          const uint8_t* posBase = geoD.positionBuffer.defined()
            ? reinterpret_cast<const uint8_t*>(geoD.positionBuffer.mapPtr()) : nullptr;
          const uint32_t posStride = geoD.positionBuffer.defined() ? geoD.positionBuffer.stride() : 0u;
          const uint32_t vCount = geoD.vertexCount;
          if (posBase != nullptr && posStride >= sizeof(Vector3) && vCount > 0u) {
            const uint32_t step = (vCount > 64u) ? (vCount / 64u) : 1u;
            for (uint32_t vi = 0u; vi < vCount; vi += step) {
              const Vector3 p = *reinterpret_cast<const Vector3*>(posBase + size_t(posStride) * vi);
              const Vector3 pw = (o2wD * Vector4(p, 1.0f)).xyz();
              vMin.x = std::min(vMin.x, pw.x); vMin.y = std::min(vMin.y, pw.y); vMin.z = std::min(vMin.z, pw.z);
              vMax.x = std::max(vMax.x, pw.x); vMax.y = std::max(vMax.y, pw.y); vMax.z = std::max(vMax.z, pw.z);
              ++sampled;
            }
          }
          const Vector3 vCen = (sampled > 0u) ? ((vMin + vMax) * 0.5f) : Vector3(0.f);
          Logger::info(str::format(
            "[ShipDraw] f=", m_device->getCurrentFrameId(),
            " vs=0x", std::hex, vsH, std::dec,
            " camType=", uint32_t(drawCallState.cameraType),
            " sampled=", sampled, " vtx=", vCount,
            " worldCen=(", vCen.x, ",", vCen.y, ",", vCen.z, ")",
            " worldMin=(", vMin.x, ",", vMin.y, ",", vMin.z, ")",
            " worldMax=(", vMax.x, ",", vMax.y, ",", vMax.z, ")",
            " o2w.t=(", float(o2wD[3][0]), ",", float(o2wD[3][1]), ",", float(o2wD[3][2]), ")"));
          // NV-DXVK [ShipDraw.w2v]: 0x292b is class static_mesh_cb3_owns_transform — its placement
          // comes from THIS draw's cb3-decomposed worldToView (not the bone palette, not the main
          // camera). o2w is identity, so verts*worldToView*proj is what positions the ship. If the
          // ship's draw worldToView is wrong/stale at the vanish yaw, the geometry projects behind
          // the camera. Diff visible vs vanish: the translation row (r3 = view-space origin pos /
          // implied camera) jumping is the cb3-decomposition bug. Compare to [ShipXform] w2v (the
          // real Main camera) — if THIS differs from that during the vanish, cb3 decomposed wrong.
          const Matrix4& w2vD = drawCallState.getTransformData().worldToView;
          Logger::info(str::format(
            "[ShipDraw.w2v] f=", m_device->getCurrentFrameId(),
            " camType=", uint32_t(drawCallState.cameraType),
            " r0=(", float(w2vD[0][0]), ",", float(w2vD[0][1]), ",", float(w2vD[0][2]), ")",
            " r1=(", float(w2vD[1][0]), ",", float(w2vD[1][1]), ",", float(w2vD[1][2]), ")",
            " r2=(", float(w2vD[2][0]), ",", float(w2vD[2][1]), ",", float(w2vD[2][2]), ")",
            " t=(", float(w2vD[3][0]), ",", float(w2vD[3][1]), ",", float(w2vD[3][2]), ")"));
        }
      }

      if (drawCallState.cameraType == CameraType::Unknown) {
        if (RtxOptions::skipObjectsWithUnknownCamera()) {
          return;
        }
        // fallback
        drawCallState.cameraType = CameraType::Enum::Main;
      }

      // NV-DXVK (fix 4): TLAS coord-space coherence gate — DIAGNOSTIC ONLY.
      // The classifier hysteresis in CameraManager::processCameraData is the
      // primary defense against the per-frame Main flicker. Once that's
      // holding, a secondary rewrite here would only be useful for draws
      // that the classifier demoted (Unknown) but the user has opted in to
      // keeping via !skipObjectsWithUnknownCamera. Rewriting worldToView on
      // legitimate Sky / ViewModel / RenderToTexture / Portal draws would
      // break them, so we only LOG suspicious coord-space disagreements
      // rather than mutating transforms. If the log shows persistent
      // off-Main world draws coming through, the classifier's gates need to
      // be tightened further upstream — not patched here.
      if (drawCallState.cameraType != CameraType::Main
          && drawCallState.cameraType != CameraType::Unknown
          && cameraManager.isMainSetByClassifier()) {
        const uint32_t frameIdNow = m_device->getCurrentFrameId();
        const uint32_t latchFrame = cameraManager.getMainClassifierFrameId();
        const bool mainFresh =
          frameIdNow <= latchFrame || (frameIdNow - latchFrame) <= 5;
        if (mainFresh) {
          const RtCamera& mainCam = cameraManager.getMainCamera();
          const Matrix4 v2w = inverse(transformData.worldToView);
          const Vector3 drawPos(v2w[3][0], v2w[3][1], v2w[3][2]);
          const Vector3 mainPos = mainCam.getPosition(/*freecam=*/false);
          const float dx = drawPos.x - mainPos.x;
          const float dy = drawPos.y - mainPos.y;
          const float dz = drawPos.z - mainPos.z;
          const float d2 = dx*dx + dy*dy + dz*dz;
          constexpr float kEpsilon = 100.0f;
          if (d2 < (kEpsilon * kEpsilon)) {
            static uint32_t sSnapLog = 0;
            if (sSnapLog < 20) {
              ++sSnapLog;
              Logger::info(str::format(
                "[TLAS-coh] suspicious non-Main draw close to Main: cameraType=",
                uint32_t(drawCallState.cameraType),
                " |delta|=", std::sqrt(d2)));
            }
          }
        }
      }

      if (tryHandleSky(&params, &drawCallState) == TryHandleSkyResult::SkipSubmit) {
        return;
      }

      // Bake the terrain
      const MaterialData* overrideMaterialData = nullptr;
      bakeTerrain(params, drawCallState, &overrideMaterialData);

    
      // An attempt to resolve cases where games pre-combine view and world matrices
      if (RtxOptions::resolvePreCombinedMatrices() &&
        isIdentityExact(drawCallState.getTransformData().worldToView)) {
        const auto* referenceCamera = &cameraManager.getCamera(drawCallState.cameraType);
        // Note: we may accept a data even from a prev frame, as we need any information to restore;
        // but if camera data is stale, it introduces an scene object transform's lag
        if (!referenceCamera->isValid(m_device->getCurrentFrameId()) &&
          !referenceCamera->isValid(m_device->getCurrentFrameId() - 1)) {
          referenceCamera = &cameraManager.getCamera(CameraType::Main);
        }
        transformData.objectToWorld = referenceCamera->getViewToWorld(false) * drawCallState.getTransformData().objectToView;
        transformData.worldToView = referenceCamera->getWorldToView(false);
      }
      
      // Apply free camera transform when view space texGenMode is used.
      // Note: TerrainBaking already applies this transform for TexGenMode::CascadedViewPositions 
      if ((transformData.texgenMode == TexGenMode::ViewPositions
           || transformData.texgenMode == TexGenMode::ViewNormals)
          && RtCamera::enableFreeCamera()) {
        if (cameraManager.isCameraValid(CameraType::Main)) {
          const RtCamera& camera = cameraManager.getMainCamera();
          // Revert the main camera's viewToWorld transform and then apply the free camera's one
          transformData.textureTransform *= camera.getViewToWorldToFreeCamViewToWorld();
        } else {
          ONCE(Logger::warn(str::format("[RTX] Tried to update surface transform with Free Camera's transform "
                                        "but main camera has not been processed this frame yet. Skipping the transform update")));
        }
      }

      getSceneManager().submitDrawState(this, drawCallState, overrideMaterialData);
    }
  }

  void RtxContext::commitExternalGeometryToRT(ExternalDrawState&& state) {
    getSceneManager().submitExternalDraw(this, std::move(state));
  }

  static uint32_t jenkinsHash(uint32_t a) {
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  void RtxContext::getDenoiseArgs(NrdArgs& outPrimaryDirectNrdArgs, NrdArgs& outPrimaryIndirectNrdArgs, NrdArgs& outSecondaryNrdArgs) {
    const bool realtimeDenoiserEnabled = RtxOptions::useDenoiser() && !RtxOptions::useDenoiserReferenceMode();
    const bool separateDenoiserEnabled = RtxOptions::denoiseDirectAndIndirectLightingSeparately();

    auto& denoiser0 = (separateDenoiserEnabled ? m_common->metaPrimaryDirectLightDenoiser() : m_common->metaPrimaryCombinedLightDenoiser());
    auto& denoiser1 = (separateDenoiserEnabled ? m_common->metaPrimaryIndirectLightDenoiser() : m_common->metaPrimaryCombinedLightDenoiser());
    auto& denoiser2 = m_common->metaSecondaryCombinedLightDenoiser();

    outPrimaryDirectNrdArgs = denoiser0.getNrdArgs();
    outPrimaryIndirectNrdArgs = denoiser1.getNrdArgs();
    outSecondaryNrdArgs = denoiser2.getNrdArgs();

    // Disable ReBLUR when RR is on because ReBLUR uses a different buffer encoding
    bool useRR = useRayReconstruction();
    if (useRR) {
      outPrimaryDirectNrdArgs.isReblurEnabled = false;
      outPrimaryIndirectNrdArgs.isReblurEnabled = false;
    }
  }

  void RtxContext::updateRaytraceArgsConstantBuffer(Resources::RaytracingOutput& rtOutput,
                                                    const VkExtent3D& downscaledExtent, const VkExtent3D& targetExtent) {
    ScopedCpuProfileZone();
    // Prepare shader arguments
    RaytraceArgs &constants = rtOutput.m_raytraceArgs;
    constants = {}; 

    auto const& camera{ getSceneManager().getCamera() };
    const uint32_t frameIdx = m_device->getCurrentFrameId();

    constants.camera = camera.getShaderConstants();

    // Set the Raytraced Render Target camera matrices
    auto const& renderTargetCamera { getSceneManager().getCameraManager().getCamera(CameraType::RenderToTexture) };
    constants.renderTargetCamera = renderTargetCamera.getShaderConstants(/*freecam =*/ false);
    constants.enableRaytracedRenderTarget = renderTargetCamera.isValid(m_device->getCurrentFrameId()) && RtxOptions::RaytracedRenderTarget::enable();
    const CameraManager& cameraManager = getSceneManager().getCameraManager();

    const bool enablePortalVolumes = RtxGlobalVolumetrics::enableInPortals() &&
      cameraManager.isCameraValid(CameraType::Portal0) &&
      cameraManager.isCameraValid(CameraType::Portal1);
    
    // Note: Ensure the number of lights can fit into the ray tracing args.
    assert(getSceneManager().getLightManager().getActiveCount() <= std::numeric_limits<uint16_t>::max());
    bool useRR = shouldUseRayReconstruction();

    constants.frameIdx = RtxOptions::rngSeedWithFrameIndex() ? m_device->getCurrentFrameId() : 0;
    constants.lightCount = static_cast<uint16_t>(getSceneManager().getLightManager().getActiveCount());

    constants.fireflyFilteringLuminanceThreshold = RtxOptions::fireflyFilteringLuminanceThreshold();
    constants.secondarySpecularFireflyFilteringThreshold = RtxOptions::secondarySpecularFireflyFilteringThreshold();
    constants.primaryRayMaxInteractions = RtxOptions::primaryRayMaxInteractions();
    constants.psrRayMaxInteractions = RtxOptions::psrRayMaxInteractions();
    constants.secondaryRayMaxInteractions = RtxOptions::secondaryRayMaxInteractions();

    // Todo: Potentially move this to the volume manager in the future to be more organized.
    constants.volumeTemporalReuseMaxSampleCount = RtxGlobalVolumetrics::temporalReuseMaxSampleCount();
    
    constants.russianRouletteMode = RtxOptions::russianRouletteMode();
    constants.russianRouletteDiffuseContinueProbability = RtxOptions::russianRouletteDiffuseContinueProbability();
    constants.russianRouletteSpecularContinueProbability = RtxOptions::russianRouletteSpecularContinueProbability();
    constants.russianRouletteDistanceFactor = RtxOptions::russianRouletteDistanceFactor();
    constants.russianRouletteMaxContinueProbability = RtxOptions::russianRouletteMaxContinueProbability();
    constants.russianRoulette1stBounceMinContinueProbability = RtxOptions::russianRoulette1stBounceMinContinueProbability();
    constants.russianRoulette1stBounceMaxContinueProbability = RtxOptions::russianRoulette1stBounceMaxContinueProbability();
    constants.pathMinBounces = RtxOptions::pathMinBounces();
    constants.pathMaxBounces = RtxOptions::pathMaxBounces();
    // Note: Probability adjustments always in the 0-1 range and therefore less than FLOAT16_MAX.
    constants.opaqueDiffuseLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueDiffuseLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueDiffuseLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueDiffuseLobeSamplingProbability());
    constants.opaqueSpecularLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueSpecularLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueSpecularLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueSpecularLobeSamplingProbability());
    constants.opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueOpacityTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueOpacityTransmissionLobeSamplingProbability());
    constants.opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueDiffuseTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueDiffuseTransmissionLobeSamplingProbability());
    constants.translucentSpecularLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::translucentSpecularLobeSamplingProbabilityZeroThreshold());
    constants.minTranslucentSpecularLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minTranslucentSpecularLobeSamplingProbability());
    constants.translucentTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::translucentTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minTranslucentTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minTranslucentTransmissionLobeSamplingProbability());
    constants.indirectRaySpreadAngleFactor = RtxOptions::indirectRaySpreadAngleFactor();

    // Note: Emissibe blend override emissive intensity always clamped to FLOAT16_MAX, so this packing is fine.
    constants.emissiveBlendOverrideEmissiveIntensity = glm::packHalf1x16(RtxOptions::emissiveBlendOverrideEmissiveIntensity());
    constants.emissiveIntensity = glm::packHalf1x16(RtxOptions::emissiveIntensity());
    constants.particleSoftnessFactor = glm::packHalf1x16(RtxOptions::particleSoftnessFactor());

    constants.psrrMaxBounces = RtxOptions::psrrMaxBounces();
    constants.pstrMaxBounces = RtxOptions::pstrMaxBounces();

    auto& rayReconstruction = m_common->metaRayReconstruction();
    constants.outputParticleLayer = useRR && rayReconstruction.useParticleBuffer();

    auto& rtxdi = m_common->metaRtxdiRayQuery();
    constants.enableEmissiveBlendEmissiveOverride = RtxOptions::enableEmissiveBlendEmissiveOverride();
    constants.enableRtxdi = RtxOptions::useRTXDI();
    constants.enableSecondaryBounces = RtxOptions::enableSecondaryBounces();
    constants.enableSeparatedDenoisers = RtxOptions::denoiseDirectAndIndirectLightingSeparately();
    constants.enableCalculateVirtualShadingNormals = RtxOptions::useVirtualShadingNormalsForDenoising();
    constants.enableViewModelVirtualInstances = RtxOptions::ViewModel::enableVirtualInstances();
    constants.enablePSRR = RtxOptions::enablePSRR();
    constants.enablePSTR = RtxOptions::enablePSTR();
    constants.enablePSTROutgoingSplitApproximation = RtxOptions::enablePSTROutgoingSplitApproximation();
    constants.enablePSTRSecondaryIncidentSplitApproximation = RtxOptions::enablePSTRSecondaryIncidentSplitApproximation();
    constants.psrrNormalDetailThreshold = RtxOptions::psrrNormalDetailThreshold();
    constants.pstrNormalDetailThreshold = RtxOptions::pstrNormalDetailThreshold();
    constants.enableDirectLighting = RtxOptions::enableDirectLighting();
    constants.enableStochasticAlphaBlend = m_common->metaComposite().enableStochasticAlphaBlend();
    constants.enableSeparateUnorderedApproximations = RtxOptions::enableSeparateUnorderedApproximations() && getResourceManager().getTLAS(Tlas::Unordered).accelStructure != nullptr;
    constants.enableDirectTranslucentShadows = RtxOptions::enableDirectTranslucentShadows();
    constants.enableDirectAlphaBlendShadows = RtxOptions::enableDirectAlphaBlendShadows();
    constants.enableIndirectTranslucentShadows = RtxOptions::enableIndirectTranslucentShadows();
    constants.enableIndirectAlphaBlendShadows = RtxOptions::enableIndirectAlphaBlendShadows();
    constants.enableRussianRoulette = RtxOptions::enableRussianRoulette();
    constants.enableDemodulateRoughness = m_common->metaDemodulate().demodulateRoughness();
    constants.enableReplaceDirectSpecularHitTWithIndirectSpecularHitT = RtxOptions::replaceDirectSpecularHitTWithIndirectSpecularHitT();
    constants.enablePortalFadeInEffect = RtxOptions::enablePortalFadeInEffect();
    constants.enableEnhanceBSDFDetail = (shouldUseDLSS() || useRR || shouldUseTAA()) && m_common->metaComposite().enableDLSSEnhancement();
    constants.enhanceBSDFIndirectMode = (uint32_t)m_common->metaComposite().dlssEnhancementMode();
    constants.enhanceBSDFDirectLightPower = useRR ? 0.0 : m_common->metaComposite().dlssEnhancementDirectLightPower();
    constants.enhanceBSDFIndirectLightPower = m_common->metaComposite().dlssEnhancementIndirectLightPower();
    constants.enhanceBSDFDirectLightMaxValue = m_common->metaComposite().dlssEnhancementDirectLightMaxValue();
    constants.enhanceBSDFIndirectLightMaxValue = m_common->metaComposite().dlssEnhancementIndirectLightMaxValue();
    constants.enhanceBSDFIndirectLightMinRoughness = m_common->metaComposite().dlssEnhancementIndirectLightMinRoughness();
    constants.enableFirstBounceLobeProbabilityDithering = RtxOptions::enableFirstBounceLobeProbabilityDithering();
    constants.enableUnorderedResolveInIndirectRays = RtxOptions::enableUnorderedResolveInIndirectRays();
    constants.enableProbabilisticUnorderedResolveInIndirectRays = RtxOptions::enableProbabilisticUnorderedResolveInIndirectRays();
    constants.enableTransmissionApproximationInIndirectRays = RtxOptions::enableTransmissionApproximationInIndirectRays();
    constants.enableUnorderedEmissiveParticlesInIndirectRays = RtxOptions::enableUnorderedEmissiveParticlesInIndirectRays();
    constants.enableDecalMaterialBlending = RtxOptions::enableDecalMaterialBlending();
    constants.enableBillboardOrientationCorrection = RtxOptions::enableBillboardOrientationCorrection() && RtxOptions::enableSeparateUnorderedApproximations();
    constants.useIntersectionBillboardsOnPrimaryRays = RtxOptions::useIntersectionBillboardsOnPrimaryRays() && constants.enableBillboardOrientationCorrection;
    constants.enableDirectLightBoilingFilter = m_common->metaDemodulate().enableDirectLightBoilingFilter() && RtxOptions::useRTXDI();
    constants.directLightBoilingThreshold = m_common->metaDemodulate().directLightBoilingThreshold();
    constants.translucentDecalAlbedoFactor = RtxOptions::translucentDecalAlbedoFactor();
    constants.enablePlayerModelInPrimarySpace = RtxOptions::PlayerModel::enableInPrimarySpace();
    constants.enablePlayerModelPrimaryShadows = RtxOptions::PlayerModel::enablePrimaryShadows();
    constants.enablePreviousTLAS = RtxOptions::enablePreviousTLAS() && m_common->getSceneManager().isPreviousFrameSceneAvailable();

    constants.pomMode = getSceneManager().getActivePOMCount() > 0 ? RtxOptions::Displacement::mode() : DisplacementMode::Off;
    if (constants.pomMode == DisplacementMode::Off) {
      constants.pomEnableDirectLighting = false;
      constants.pomEnableIndirectLighting = false;
      constants.pomEnableNEECache = false;
      constants.pomEnableReSTIRGI = false;
      constants.pomEnablePSR = true; // enable PSR for materials with heightmaps if POM is completely disabled.
    } else {
      constants.pomEnableDirectLighting = RtxOptions::Displacement::enableDirectLighting();
      constants.pomEnableIndirectLighting = RtxOptions::Displacement::enableIndirectLighting();
      constants.pomEnableNEECache = RtxOptions::Displacement::enableNEECache();
      constants.pomEnableReSTIRGI = RtxOptions::Displacement::enableReSTIRGI();
      constants.pomEnablePSR = RtxOptions::Displacement::enablePSR();
    }
    constants.pomMaxIterations = RtxOptions::Displacement::maxIterations();

    constants.totalMipBias = getSceneManager().getTotalMipBias(); 

    constants.upscaleFactor = float2 {
      rtOutput.m_compositeOutputExtent.width / static_cast<float>(rtOutput.m_finalOutputExtent.width),
      rtOutput.m_compositeOutputExtent.height / static_cast<float>(rtOutput.m_finalOutputExtent.height) };

    constants.terrainArgs = getSceneManager().getTerrainBaker().getTerrainArgs();

    constants.sssArgs.enableThinOpaque = RtxOptions::SubsurfaceScattering::enableThinOpaque();
    constants.sssArgs.enableDiffusionProfile = RtxOptions::SubsurfaceScattering::enableDiffusionProfile();
    constants.sssArgs.diffusionProfileScale = std::max(RtxOptions::SubsurfaceScattering::diffusionProfileScale(), 0.001f);
    constants.enableSssTransmission = RtxOptions::SubsurfaceScattering::enableTransmission();
    constants.enableSssTransmissionSingleScattering = RtxOptions::SubsurfaceScattering::enableTransmissionSingleScattering();
    constants.sssTransmissionBsdfSampleCount = RtxOptions::SubsurfaceScattering::transmissionBsdfSampleCount();
    constants.sssTransmissionSingleScatteringSampleCount = RtxOptions::SubsurfaceScattering::transmissionSingleScatteringSampleCount();
    constants.enableTransmissionDiffusionProfileCorrection = RtxOptions::SubsurfaceScattering::enableTransmissionDiffusionProfileCorrection();
    constants.enableHeuristicSingleScatteringTransmission = RtxOptions::SubsurfaceScattering::enableHeuristicSingleScatteringTransmission();
    constants.sssArgs.diffusionProfileDebuggingPixel = u16vec2 {
      static_cast<uint16_t>(RtxOptions::SubsurfaceScattering::diffusionProfileDebugPixelPosition().x),
      static_cast<uint16_t>(RtxOptions::SubsurfaceScattering::diffusionProfileDebugPixelPosition().y) };

    auto& restirGI = m_common->metaReSTIRGIRayQuery();
    ReSTIRGISampleStealing restirGISampleStealingMode = restirGI.useSampleStealing();
    // Stealing pixels requires indirect light stored in separated buffers instead of combined with direct light,
    // steal samples if separated denoiser is disabled.
    if (restirGISampleStealingMode == ReSTIRGISampleStealing::StealPixel 
        && !RtxOptions::denoiseDirectAndIndirectLightingSeparately()) {
      restirGISampleStealingMode = ReSTIRGISampleStealing::StealSample;
    }
    constants.enableReSTIRGI = restirGI.isActive();
    constants.enableReSTIRGITemporalReuse = restirGI.useTemporalReuse();
    constants.enableReSTIRGISpatialReuse = restirGI.useSpatialReuse();
    constants.reSTIRGIMISMode = (uint32_t)restirGI.misMode();
    constants.enableReSTIRGIFinalVisibility = restirGI.useFinalVisibility();
    constants.enableReSTIRGIReflectionReprojection = restirGI.useReflectionReprojection();
    constants.restirGIReflectionMinParallax = restirGI.reflectionMinParallax();
    constants.enableReSTIRGIVirtualSample = restirGI.useVirtualSample();
    constants.reSTIRGIMISModePairwiseMISCentralWeight = restirGI.pairwiseMISCentralWeight();
    constants.reSTIRGIVirtualSampleLuminanceThreshold = restirGI.virtualSampleLuminanceThreshold();
    constants.reSTIRGIVirtualSampleRoughnessThreshold = restirGI.virtualSampleRoughnessThreshold();
    constants.reSTIRGIVirtualSampleSpecularThreshold = restirGI.virtualSampleSpecularThreshold();
    constants.reSTIRGIVirtualSampleMaxDistanceRatio = restirGI.virtualSampleMaxDistanceRatio();
    constants.reSTIRGIBiasCorrectionMode = (uint32_t) restirGI.biasCorrectionMode();
    constants.enableReSTIRGIPermutationSampling = restirGI.usePermutationSampling();
    constants.enableReSTIRGISampleStealing = (uint32_t)restirGISampleStealingMode;
    constants.reSTIRGISampleStealingJitter = restirGI.sampleStealingJitter();
    constants.enableReSTIRGIStealBoundaryPixelSamplesWhenOutsideOfScreen = (uint32_t)restirGI.stealBoundaryPixelSamplesWhenOutsideOfScreen();
    constants.enableReSTIRGIBoilingFilter = restirGI.useBoilingFilter();
    constants.boilingFilterLowerThreshold = restirGI.boilingFilterMinThreshold();
    constants.boilingFilterHigherThreshold = restirGI.boilingFilterMaxThreshold();
    constants.boilingFilterRemoveReservoirThreshold = restirGI.boilingFilterRemoveReservoirThreshold();
    constants.temporalHistoryLength = restirGI.getTemporalHistoryLength(GlobalTime::get().deltaTimeMs());
    constants.permutationSamplingSize = restirGI.permutationSamplingSize();
    constants.enableReSTIRGIDLSSRRCompatibilityMode = useRR ? restirGI.useDLSSRRCompatibilityMode() : 0;
    constants.reSTIRGIDLSSRRTemporalRandomizationRadius = constants.camera.resolution.x / 960.0f * restirGI.DLSSRRTemporalRandomizationRadius();
    constants.enableReSTIRGITemporalBiasCorrection = restirGI.useTemporalBiasCorrection();
    constants.enableReSTIRGIDiscardEnlargedPixels = restirGI.useDiscardEnlargedPixels();
    constants.reSTIRGIHistoryDiscardStrength = restirGI.historyDiscardStrength();
    constants.enableReSTIRGITemporalJacobian = restirGI.useTemporalJacobian();
    constants.reSTIRGIFireflyThreshold = restirGI.fireflyThreshold();
    constants.reSTIRGIRoughnessClamp = restirGI.roughnessClamp();
    constants.reSTIRGIMISRoughness = restirGI.misRoughness();
    constants.reSTIRGIMISParallaxAmount = restirGI.parallaxAmount();
    constants.enableReSTIRGIDemodulatedTargetFunction = restirGI.useDemodulatedTargetFunction();
    constants.enableReSTIRGILightingValidation = RtxOptions::useRTXDI() && rtxdi.enableDenoiserGradient() && restirGI.validateLightingChange();
    constants.reSTIRGISampleValidationThreshold = restirGI.lightingValidationThreshold();
    constants.enableReSTIRGIVisibilityValidation = restirGI.validateVisibilityChange();
    constants.reSTIRGIVisibilityValidationRange = 1.0f + restirGI.visibilityValidationRange();

    // Neural Radiance Cache
    NeuralRadianceCache& nrc = m_common->metaNeuralRadianceCache();
    constants.enableNrc = nrc.isActive();
    constants.allowNrcTraining = NeuralRadianceCache::NrcOptions::trainCache();
    nrc.setRaytraceArgs(constants);

    m_common->metaNeeCache().setRaytraceArgs(constants, m_resetHistory);
    constants.surfaceCount = getSceneManager().getAccelManager().getSurfaceCount();
    // NV-DXVK [Coverage]: log cb.surfaceCount being uploaded each frame so
    // we can detect when the GPU sees a stale (large) value vs what the
    // coverage readback sees later in the same frame. If these diverge,
    // the threshold check in opaque_surface_material_interaction.slangh
    // is using the wrong frame's value — explaining how primary rays can
    // hit surfaceIndex >> orderedSize without firing the per-site counter.
    {
      static uint32_t s_cbLogN = 0;
      const uint32_t n = s_cbLogN++;
      if (n < 200u || (n % 60u) == 0u) {
        Logger::info(str::format(
          "[SpawnGeomDiag.CbSurfaceCount] frame=", m_device->getCurrentFrameId(),
          " cbUploadedSurfaceCount=", constants.surfaceCount,
          " (set from getSurfaceCount() = m_reorderedSurfaces.size())"));
      }
    }

    auto* cameraTeleportDirectionInfo = getSceneManager().getRayPortalManager().getCameraTeleportationRayPortalDirectionInfo();
    constants.teleportationPortalIndex = cameraTeleportDirectionInfo ? cameraTeleportDirectionInfo->entryPortalInfo.portalIndex + 1 : 0;

    // Note: Use half of the vertical FoV for the main camera in radians divided by the vertical resolution to get the effective half angle of a single pixel.
    constants.screenSpacePixelSpreadHalfAngle = getSceneManager().getCamera().getFov() / 2.0f / constants.camera.resolution.y;

    // NV-DXVK: the engine-hook Main camera can transiently carry an invalid
    // FoV — an all-zero viewToProjection during a load/transition decodes to
    // NaN, and RtCameraSetting::fov is uninitialized until Main's first valid
    // update — which makes this NaN or non-positive. signbit(NaN) is true, so
    // it both trips the assert below and (in release) corrupts Ray Interaction
    // encoding for the whole frame. Clamp to a safe positive spread so one bad
    // camera frame doesn't poison ray cones. (Root cause is the transient bad
    // engine camera, tracked separately; this is the defensive consumption guard.)
    if (!std::isfinite(constants.screenSpacePixelSpreadHalfAngle)
        || constants.screenSpacePixelSpreadHalfAngle <= 0.0f) {
      // Surface bad-camera-at-render events so camera-stability regressions
      // are visible. Throttled: first 30 unconditionally, then 1 per 120
      // frames, so a persistent bad camera trickles instead of spamming.
      static uint64_t sBadMainCamN = 0;
      const uint64_t bn = sBadMainCamN++;
      if (bn < 30u || (bn % 120u) == 0u) {
        Logger::warn(str::format(
          "[RtxContext.BadMainCam] n=", bn,
          " frame=", m_device->getCurrentFrameId(),
          " getFov()=", getSceneManager().getCamera().getFov(),
          " resY=", constants.camera.resolution.y,
          " rawSpread=", constants.screenSpacePixelSpreadHalfAngle,
          " — Main camera FoV invalid at render (NaN/<=0); clamped to 1e-3."
          " Engine viewToProjection likely degenerate/all-zero this frame."));
      }
      constants.screenSpacePixelSpreadHalfAngle = 1.0e-3f; // ~ 80deg / 1080px
    }

    // Note: This value is assumed to be positive (specifically not have the sign bit set) as otherwise it will break Ray Interaction encoding.
    assert(std::signbit(constants.screenSpacePixelSpreadHalfAngle) == false);

    // Enable object picking only when resource was created
    // TODO: should be a spec.const
    constants.enableObjectPicking = bool { rtOutput.m_primaryObjectPicking.isValid() };

    // Debug View
    {
      const DebugView& debugView = m_common->metaDebugView();
      constants.debugView = debugView.debugViewIdx();
      constants.debugKnob = debugView.debugKnob();
      constants.forceFirstHitInGBufferPass = debugView.showFirstGBufferHit();
      
      constants.gpuPrintThreadIndex = u16vec2 { kInvalidThreadIndex, kInvalidThreadIndex };
      constants.gpuPrintElementIndex = frameIdx % kMaxFramesInFlight;

      // NV-DXVK: dropped ImGui::IsKeyDown(ModCtrl) so gpuPrint fires every
      // frame while enabled. When useMousePosition is on and no CTRL is held,
      // anchor to screen centre (since the mouse isn't captured during
      // gameplay). When pixelIndex is unset (INT_MAX), also use centre. Used
      // by geometry_resolver.slangh's gradient-magnitude readback.
      //
      // NV-DXVK DIAG: when the user is on the path-checker view, force
      // gpuPrint on and anchored to screen centre. The slang side writes a
      // packed float4 every frame at that pixel: (pathCode, hitWasOpaque,
      // overrodeColor, frameIdx). Surfaced via [PathCheckerProbe] logs so
      // the user can read what's happening on the central wall pixel
      // without needing to move the cursor or toggle anything in ImGui.
      // Force gpuPrint on for two debug views:
      //   52 (DEBUG_VIEW_GRADIENT_PATH_CHECKER) — our PathCheckerProbe
      //   32 (DEBUG_VIEW_RAW_ALBEDO)            — existing slot 5/6/7 UV-dump
      //                                           probes in surface_interaction.slangh
      // Lets the user switch to view 32 briefly to pull per-triangle UVs at the
      // centre pixel without first having to toggle gpuPrint in ImGui.
      const bool pathCheckerForceLog =
        debugView.debugViewIdx() == DEBUG_VIEW_GRADIENT_PATH_CHECKER
        || debugView.debugViewIdx() == DEBUG_VIEW_RAW_ALBEDO;
      if (debugView.gpuPrint.enable() || pathCheckerForceLog) {
        const Vector2i configured = debugView.gpuPrint.pixelIndex();
        const bool useMouse = debugView.gpuPrint.useMousePosition()
          && ImGui::IsKeyDown(ImGuiKey_ModCtrl);
        if (useMouse) {
          Vector2 toDownscaledExtentScale{
            downscaledExtent.width / static_cast<float>(targetExtent.width),
            downscaledExtent.height / static_cast<float>(targetExtent.height)
          };
          const ImVec2 mousePos = ImGui::GetMousePos();
          constants.gpuPrintThreadIndex = u16vec2 {
            static_cast<uint16_t>(mousePos.x * toDownscaledExtentScale.x),
            static_cast<uint16_t>(mousePos.y * toDownscaledExtentScale.y)
          };
        } else if (configured.x != INT32_MAX && configured.y != INT32_MAX) {
          constants.gpuPrintThreadIndex = u16vec2 {
            static_cast<uint16_t>(configured.x),
            static_cast<uint16_t>(configured.y)
          };
        } else {
          constants.gpuPrintThreadIndex = u16vec2 {
            static_cast<uint16_t>(downscaledExtent.width  / 2u),
            static_cast<uint16_t>(downscaledExtent.height / 2u)
          };
        }
      }
    }

    // NV-DXVK: scene dump trigger + buffer arming. The actual disk write
    // happens later in dispatchDebugView once the GPU has finished the
    // captured frame. F11 is the hotkey; an ImGui button could trigger the
    // same path by setting g_sceneDumpRequestThisFrame externally.
    {
      constants.sceneDumpStride = downscaledExtent.width;
      constants.sceneDumpEnabled = 0u;

      const bool keyDown = ImGui::IsKeyDown(ImGuiKey_F11);
      const bool edge = keyDown && !g_sceneDumpHotkeyLatch;
      g_sceneDumpHotkeyLatch = keyDown;
      const bool externalRequest = g_sceneDumpRequestThisFrame;
      g_sceneDumpRequestThisFrame = false;

      if ((edge || externalRequest) && g_sceneDumpState == SceneDumpState::Idle) {
        const VkDeviceSize bytes = VkDeviceSize(downscaledExtent.width)
          * VkDeviceSize(downscaledExtent.height) * sizeof(SceneDumpElement);
        const bool needAlloc = !rtOutput.m_sceneDumpBuffer.ptr()
          || rtOutput.m_sceneDumpExtent.width != downscaledExtent.width
          || rtOutput.m_sceneDumpExtent.height != downscaledExtent.height;
        if (needAlloc) {
          DxvkBufferCreateInfo info;
          info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
          info.stages = VK_PIPELINE_STAGE_HOST_BIT
            | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
          info.access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
          info.size = bytes;
          rtOutput.m_sceneDumpBuffer = m_device->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer,
            "Scene Dump Buffer");
          rtOutput.m_sceneDumpExtent = { downscaledExtent.width, downscaledExtent.height };
        }
        // Zero-init flags so missing pixels are distinguishable from real
        // hits in the CSV (flags bit 0 = hit valid, set by the shader).
        if (rtOutput.m_sceneDumpBuffer.ptr()) {
          void* mapped = rtOutput.m_sceneDumpBuffer->mapPtr(0);
          if (mapped) {
            std::memset(mapped, 0, bytes);
          }
        }
        constants.sceneDumpEnabled = 1u;
        g_sceneDumpState = SceneDumpState::AwaitingReadback;
        g_sceneDumpTriggerFrame = frameIdx;
        g_sceneDumpReadbackAtFrame = frameIdx + uint32_t(kMaxFramesInFlight) + 1u;
        g_sceneDumpExtent = { downscaledExtent.width, downscaledExtent.height };
        Logger::info(str::format(
          "[SceneDump] capture armed at frame ", frameIdx,
          " (", downscaledExtent.width, "x", downscaledExtent.height,
          ", ", double(bytes) / (1024.0 * 1024.0), " MB)"));
      }
    }

    getDenoiseArgs(constants.primaryDirectNrd, constants.primaryIndirectNrd, constants.secondaryCombinedNrd);

    RayPortalManager::SceneData portalData = getSceneManager().getRayPortalManager().getRayPortalInfoSceneData();
    constants.numActiveRayPortals = portalData.numActiveRayPortals;
    constants.virtualInstancePortalIndex = getSceneManager().getInstanceManager().getVirtualInstancePortalIndex() & 0xff;

    memcpy(&constants.rayPortalHitInfos[0], &portalData.rayPortalHitInfos, sizeof(portalData.rayPortalHitInfos));
    memcpy(&constants.rayPortalHitInfos[maxRayPortalCount], &portalData.previousRayPortalHitInfos, sizeof(portalData.previousRayPortalHitInfos));

    constants.uniformRandomNumber = jenkinsHash(constants.frameIdx);
    constants.vertexColorStrength = RtxOptions::vertexColorStrength();
    constants.viewModelRayTMax = RtxOptions::ViewModel::rangeMeters() * RtxOptions::getMeterToWorldUnitScale();
    constants.roughnessDemodulationOffset = m_common->metaDemodulate().demodulateRoughnessOffset();
    
    const RtxGlobalVolumetrics& globalVolumetrics = getCommonObjects()->metaGlobalVolumetrics();
    constants.volumeArgs = globalVolumetrics.getVolumeArgs(cameraManager, getSceneManager().getFogState(), enablePortalVolumes);
    constants.startInMediumMaterialIndex = getSceneManager().getStartInMediumMaterialIndex();
    OpaqueMaterialOptions::fillShaderParams(constants.opaqueMaterialArgs);
    TranslucentMaterialOptions::fillShaderParams(constants.translucentMaterialArgs);
    ViewDistanceOptions::fillShaderParams(constants.viewDistanceArgs, RtxOptions::getMeterToWorldUnitScale());
    constants.alphaBlendSurfacePackMult = RtxOptions::getMeterToWorldUnitScale();

    // We are going to use this value to perform some animations on GPU, to mitigate precision related issues loop time
    // at the 24 bit boundary (as we use a 8 bit scalar on top of this time which we want to fit into 32 bits without issues,
    // plus we also convert this value to a floating point value at some point as well which has 23 bits of precision).
    // Bitwise and used rather than modulus as well for slightly better performance.
    constants.timeSinceStartSeconds = (static_cast<uint32_t>(GlobalTime::get().absoluteTimeMs()) & ((1U << 24U) - 1U)) / 1000.f;

    // NV-DXVK: engine c_gameTime captured by d3d11_rtx.cpp::FillMaterialData
    // when a screen-space-emissive PS draw fires. Used as the per-frame
    // multiplier on c_uv1Translate in the opaque material slang so the
    // holo-character / viewmodel scrolling pattern matches native rather
    // than freezing at translate × 1.0. Default 0 before any such draw
    // (matches native at game-time-zero — pattern at constant baseline).
    constants.screenSpaceEmissiveTime = getSceneManager().getEngineGameTime();

    // NV-DXVK [debug.disableDetailOverlay]: diagnostic — skip the TF2 MOD2X
    // detail-texture albedo overlay in the opaque material shader.
    constants.disableDetailOverlay = RtxOptions::disableDetailOverlay() ? 1u : 0u;

    // NV-DXVK: TF2 3D-skybox cloud fog params — captured from cloud draws in
    // d3d11_rtx.cpp::FillMaterialData, consumed by the opaque surface
    // material shader for OPAQUE_SURFACE_MATERIAL_FLAG_TF2_SKYBOX_FOG.
    {
      const SceneManager::Tf2CloudFogParams fog = getSceneManager().getTf2CloudFog();
      constants.tf2FogK1_K0W = fog.k1_k0w;
      constants.tf2FogK2_K2W = fog.k2_k2w;
      constants.tf2FogK3     = fog.k3;
      constants.tf2FogMisc   = fog.misc;
    }

    m_common->metaRtxdiRayQuery().setRaytraceArgs(rtOutput);
    getSceneManager().getLightManager().setRaytraceArgs(
      constants,
      m_common->metaRtxdiRayQuery().initialSampleCount(),
      RtxGlobalVolumetrics::initialRISSampleCount(),
      RtxOptions::risLightSampleCount());

    constants.resolveTransparencyThreshold = RtxOptions::resolveTransparencyThreshold();
    constants.resolveOpaquenessThreshold = RtxOptions::resolveOpaquenessThreshold();
    constants.resolveStochasticAlphaBlendThreshold = m_common->metaComposite().stochasticAlphaBlendOpacityThreshold();

    // NV-DXVK [c_envMapLightScale]: TF2 ships a per-map env-map brightness
    // scalar at CBufCommonPerCamera off 272. Multiply it into skyBrightness
    // so cubemap bounce intensity matches the artist's intended exposure
    // for each map. Falls back to slider value when no engine snapshot
    // is fresh (menu, first frame).
    {
      const EngineSunSnapshot snap = fetchEngineSunCapture();
      const float envScale = (snap.valid && snap.envMapLightScale > 0.0f)
        ? snap.envMapLightScale : 1.0f;
      constants.skyBrightness = RtxOptions::skyBrightness() * envScale;

      // One-shot sky mode + env-scale verification log so the user can
      // grep [SkyMode.startup] to confirm the default is what they want
      // and that the per-map env scale is being captured.
      static std::atomic<bool> sLoggedSkyMode{ false };
      bool expected = false;
      if (sLoggedSkyMode.compare_exchange_strong(expected, true,
            std::memory_order_relaxed)) {
        const char* modeStr = "unknown";
        switch (RtxOptions::skyMode()) {
          case SkyMode::SkyboxRasterization: modeStr = "SkyboxRasterization"; break;
          case SkyMode::PhysicalAtmosphere:  modeStr = "PhysicalAtmosphere"; break;
          case SkyMode::Hybrid:              modeStr = "Hybrid"; break;
        }
        Logger::info(str::format(
          "[SkyMode.startup] mode=", modeStr,
          " skyBrightnessSlider=", RtxOptions::skyBrightness(),
          " engineEnvMapScale=", envScale,
          " effectiveSkyBrightness=", constants.skyBrightness,
          " snapValid=", (snap.valid ? 1 : 0)));
      }
    }
    constants.skyMode = static_cast<uint32_t>(RtxOptions::skyMode());
    // NV-DXVK [EngineSun]: when the sun is provided as an RTXDI Distant light, tell the
    // integrator to skip the bespoke NEE sun so it isn't double-counted. Only meaningful
    // in atmosphere/hybrid sky modes (matches the setEngineSunLight gate above).
    {
      const SkyMode skyModeNow = RtxOptions::skyMode();
      constants.sunAsRtxdiLight = (RtxOptions::sunAsRtxdiLight()
        && (skyModeNow == SkyMode::PhysicalAtmosphere || skyModeNow == SkyMode::Hybrid)) ? 1u : 0u;
    }
    constants.skyProbePopulated = m_skyProbeCubemapPopulated ? 1u : 0u;
    // [SkyTrace.constants] Per-gameplay-frame snapshot of the sky-pipeline
    // constants the path tracer will consume. If yellow tracks with
    // skyProbePopulated=1, the cube-render path is poisoning the LUT/PSR
    // sample sites. If skyMode flipped unexpectedly (e.g. dxvk.conf not
    // loading), this catches it without needing the dxgi.log effective-
    // config dump. Throttled once per gameplay frame.
    {
      const uint32_t frameId = m_device->getCurrentFrameId();
      static std::atomic<uint32_t> sLastFrame{ UINT32_MAX };
      const uint32_t lastLogged = sLastFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && lastLogged != frameId) {
        sLastFrame.store(frameId, std::memory_order_relaxed);
        Logger::info(str::format(
          "[SkyTrace.constants] frame=", frameId,
          " skyMode=", constants.skyMode,
          " skyBrightness=", constants.skyBrightness,
          " skyProbePopulated=", constants.skyProbePopulated));
      }
    }
    
    // Detect sky mode change and clear sky buffers when switching to Physical Atmosphere
    SkyMode currentSkyMode = RtxOptions::skyMode();
    if (currentSkyMode != m_lastSkyMode) {
      if (currentSkyMode == SkyMode::PhysicalAtmosphere) {
        // Clear the rasterized skybox buffers when switching to physical atmosphere
        auto skyProbe = getResourceManager().getSkyProbe(this, m_skyColorFormat);
        auto skyMatte = getResourceManager().getSkyMatte(this, m_skyRtColorFormat);
        
        VkClearValue clearValue = {};
        clearValue.color.float32[0] = 0.0f;
        clearValue.color.float32[1] = 0.0f;
        clearValue.color.float32[2] = 0.0f;
        clearValue.color.float32[3] = 0.0f;
        
        if (skyProbe.view != nullptr) {
          DxvkContext::clearRenderTarget(skyProbe.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);
        }
        if (skyMatte.view != nullptr) {
          DxvkContext::clearRenderTarget(skyMatte.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);
        }
      }
      m_lastSkyMode = currentSkyMode;
    }
    
    // Update atmosphere parameters. Hybrid mode also computes LUTs
    // because the sun NEE evaluator (sampleAtmosphereSunLight) reads
    // transmittance from them - we just don't sample them for the
    // visible sky in Hybrid (rasterized skybox handles that).
    const SkyMode skyModeForLut = RtxOptions::skyMode();
    if (skyModeForLut == SkyMode::PhysicalAtmosphere
        || skyModeForLut == SkyMode::Hybrid) {
      if (!m_atmosphere) {
        m_atmosphere = std::make_unique<RtxAtmosphere>(m_device.ptr());
      }
      m_atmosphere->initialize(this);
      m_atmosphere->computeLuts(this);
      constants.atmosphereArgs = m_atmosphere->getAtmosphereArgs();

      // [SkyTune.viewAlt] Override the static viewAltitude slider with
      // the camera's actual world-space altitude. Hillaire's transmittance
      // and scattering integrate over altitude, so a sniper on a tall
      // tower or a Titan pilot in a high cockpit should see thinner
      // atmosphere than someone on the ground. Camera position is in
      // the Remix-internal world frame (Y-up regardless of isZUp), so
      // .y is altitude in game units; convert to km via worldToKm.
      // Clamp to [0, atmosphereThickness] so a player buried below
      // the world or floating in vacuum-of-space cinematic doesn't
      // produce silly LUT lookups.
      {
        const Vector3 camPos = getSceneManager().getCamera().getPosition(false);
        const float worldToKm = constants.atmosphereArgs.aerialPerspectiveWorldToKm;
        const float altitudeKm = camPos.y * worldToKm;
        const float maxAltKm   = constants.atmosphereArgs.atmosphereThickness;
        if (std::isfinite(altitudeKm) && altitudeKm >= 0.0f) {
          constants.atmosphereArgs.viewAltitude =
            std::min(altitudeKm, maxAltKm);
        }
      }

      // [Atmosphere.live] Gameplay-gated dump of the args being
      // uploaded to the GPU, to diagnose the cyan/blue ghost on
      // geometry. Suspect axes:
      //   - sunDirection is in LUT Y-up space (good for LUT gen) but
      //     the path-tracer consumer at geometry_resolver.slangh:2017
      //     passes ray.direction in world space (Z-up for TF2). The
      //     dot-product against the hardcoded vec3(0,1,0) zenith and
      //     against sunDirection then samples the wrong LUT slice.
      //   - rayleighScattering / mieScattering / ozoneAbsorption drive
      //     the analytic-RGB transmittance in sampleAerialPerspective;
      //     unreasonable values manifest as a uniform tint.
      // Logs every gameplay frame (game is running at ~1 fps so a
      // 60-frame throttle would mean one log per minute). Skipped on
      // menu/loading frames via the camera+TLAS predicate.
      {
        const uint32_t frameId = m_device->getCurrentFrameId();
        const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
        const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
        if (cameraValid && tlasReady) {
          const AtmosphereArgs& a = constants.atmosphereArgs;
          // Camera forward in world space (whatever the engine uses —
          // Z-up for TF2). Consumer at geometry_resolver passes
          // ray.direction in this same world frame, then dots it
          // against sunDirection (LUT Y-up frame) and the hardcoded
          // (0,1,0) zenith. If world is Z-up, the zenith dot reads
          // the WRONG component → tY is constant across the scene
          // and every shaded pixel samples the same LUT row. That
          // would manifest as a uniform tint over geometry.
          const Vector3 camFwd = getSceneManager().getCamera().getDirection(false);
          const float dotForwardSun = camFwd.x * a.sunDirection.x
                                    + camFwd.y * a.sunDirection.y
                                    + camFwd.z * a.sunDirection.z;
          const float dotForwardYupZenith = camFwd.y; // == dot(camFwd, (0,1,0))
          const float dotForwardZupZenith = camFwd.z; // == dot(camFwd, (0,0,1))
          Logger::info(str::format(
            "[Atmosphere.live] frame=", frameId,
            " skyMode=", static_cast<uint32_t>(skyModeForLut),
            " isZUp=", (RtxOptions::zUp() ? "true" : "false"),
            " sunDir=(", a.sunDirection.x, ",", a.sunDirection.y, ",", a.sunDirection.z, ")",
            " camFwd=(", camFwd.x, ",", camFwd.y, ",", camFwd.z, ")",
            " dot(camFwd,sunDir)=", dotForwardSun,
            " dot(camFwd,Yup)=", dotForwardYupZenith,
            " dot(camFwd,Zup)=", dotForwardZupZenith,
            " sunIllum=(", a.sunIlluminance.x, ",", a.sunIlluminance.y, ",", a.sunIlluminance.z, ")",
            " viewAltKm=", a.viewAltitude,
            " apStrength=", a.aerialPerspectiveStrength,
            " apWorldToKm=", a.aerialPerspectiveWorldToKm,
            " rayleigh=(", a.rayleighScattering.x, ",", a.rayleighScattering.y, ",", a.rayleighScattering.z, ")",
            " mie=(", a.mieScattering.x, ",", a.mieScattering.y, ",", a.mieScattering.z, ")",
            " ozone=(", a.ozoneAbsorption.x, ",", a.ozoneAbsorption.y, ",", a.ozoneAbsorption.z, ")",
            " mieAniso=", a.mieAnisotropy,
            // [SkyTrace.fogColor] TF2's authored fog color, captured from cb2.
            // composite.comp.slang L573 lerps Hillaire AP inscatter toward this
            // when fogStrength > 0. If fogColor is yellow at low altitude and
            // dim/zero at high altitude, the AP+fog block could be the source
            // of the altitude-dependent yellow tint on world geometry.
            " skyTint=(", a.skyTint.x, ",", a.skyTint.y, ",", a.skyTint.z, ")",
            " fogColor=(", a.fogColor.x, ",", a.fogColor.y, ",", a.fogColor.z, ")",
            " fogStrength=", a.fogStrength));

          // [Atmosphere.lut] Same frame's GPU-side LUT readback.
          // Heavy enough (24 KB copy + async decode) that we throttle
          // to once per 4 gameplay frames at 1 fps that's ~once per
          // 4 sec which is enough to track LUT evolution.
          if ((frameId % 4u) == 0u) {
            recordAerialPerspectiveLutReadback();
          }
          // [SkyTint.diag] removed — superseded by direct cbuffer
          // capture of c_skyColor * c_envMapLightScale via
          // EngineSunCapture (see rtx_engine_sun.h, d3d11_rtx.cpp).
          // The captured tint is plumbed to atmosphereArgs.skyTint and
          // applied at the IBL sample sites.
        }
      }
    }

    constants.isLastCompositeOutputValid = restirGI.isActive() && restirGI.getLastCompositeOutput().matchesWriteFrameIdx(frameIdx - 1);
    constants.isZUp = RtxOptions::zUp();
    constants.enableCullingSecondaryRays = RtxOptions::enableCullingInSecondaryRays();

    constants.domeLightArgs = getSceneManager().getLightManager().getDomeLightArgs();
    // [SkyTrace.domeArgs] Dome-light is the OTHER consumer of SkyLight in the
    // composite shader (composite.comp.slang:513). If domeLightArgs.active
    // got flipped on, the matte we cleared is being sampled as a panoramic
    // texture via worldToLightTransform — that wraps the white-clear in a
    // way that can produce tinted output. Logged on-change of (active,
    // textureIndex, radiance), plus once per 240 frames as a heartbeat.
    {
      const DomeLightArgs& d = constants.domeLightArgs;
      static std::atomic<uint32_t> sActive{ UINT32_MAX };
      static std::atomic<uint32_t> sTexIdx{ UINT32_MAX };
      static std::atomic<float> sRadX{ -1.0f };
      static std::atomic<uint32_t> sLastFrame{ UINT32_MAX };
      const uint32_t frameId = m_device->getCurrentFrameId();
      const bool changed =
        sActive.load(std::memory_order_relaxed) != d.active ||
        sTexIdx.load(std::memory_order_relaxed) != d.textureIndex ||
        std::abs(sRadX.load(std::memory_order_relaxed) - d.radiance.x) > 1e-4f;
      const uint32_t lastLogged = sLastFrame.load(std::memory_order_relaxed);
      const bool heartbeat = (frameId != lastLogged) && (frameId % 240u == 0u);
      if (changed || heartbeat) {
        sActive.store(d.active, std::memory_order_relaxed);
        sTexIdx.store(d.textureIndex, std::memory_order_relaxed);
        sRadX.store(d.radiance.x, std::memory_order_relaxed);
        sLastFrame.store(frameId, std::memory_order_relaxed);
        Logger::info(str::format(
          "[SkyTrace.domeArgs] frame=", frameId,
          " active=", d.active,
          " textureIndex=", d.textureIndex,
          " radiance=(", d.radiance.x, ",", d.radiance.y, ",", d.radiance.z, ")",
          " trigger=", (changed ? "changed" : "heartbeat")));
      }
    }

    // Ray miss value handling
    constants.clearColorDepth = getSceneManager().getGlobals().clearColorDepth;
    constants.clearColorPicking = getSceneManager().getGlobals().clearColorPicking;
    constants.clearColorNormal = getSceneManager().getGlobals().clearColorNormal;

    // DLSS-RR
    constants.enableDLSSRR = useRR;
    constants.setLogValueForDisocclusionMaskForDLSSRR = DxvkRayReconstruction::enableDisocclusionMaskBlur();

    NrdArgs primaryDirectNrdArgs;
    NrdArgs primaryIndirectNrdArgs;
    NrdArgs secondaryNrdArgs;
    getDenoiseArgs(primaryDirectNrdArgs, primaryIndirectNrdArgs, secondaryNrdArgs);

    constants.primaryDirectMissLinearViewZ = primaryDirectNrdArgs.missLinearViewZ;

    constants.wboitEnergyLossCompensation = RtxOptions::wboitEnergyLossCompensation();
    constants.wboitDepthWeightTuning = RtxOptions::wboitDepthWeightTuning();
    constants.wboitEnabled = RtxOptions::wboitEnabled();

    constants.eyeArgs.enableEyes = RtxOptions::Eye::enable();
    constants.eyeArgs.normalBendingEyeball = RtxOptions::Eye::eyeballSphereOffset();
    constants.eyeArgs.normalBendingCornea = RtxOptions::Eye::corneaSphereOffset();
    constants.eyeArgs.whitesAlbedoScale = RtxOptions::Eye::eyeWhitesAlbedoScale();
    constants.eyeArgs.irisRadius = RtxOptions::Eye::irisRadius();
    constants.eyeArgs.irisDepth = RtxOptions::Eye::irisDepth();

    // Upload the constants to the GPU
    {
      Rc<DxvkBuffer> cb = getResourceManager().getConstantsBuffer();

      writeToBuffer(cb, 0, sizeof(constants), &constants);

      m_cmd->trackResource<DxvkAccess::Write>(cb);
    }
  }

  void RtxContext::bindCommonRayTracingResources(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();

    Rc<DxvkBuffer> constantsBuffer = getResourceManager().getConstantsBuffer();
    Rc<DxvkBuffer> surfaceBuffer = getSceneManager().getSurfaceBuffer();
    Rc<DxvkBuffer> surfaceMappingBuffer = getSceneManager().getSurfaceMappingBuffer();
    Rc<DxvkBuffer> billboardsBuffer = getSceneManager().getBillboardsBuffer();
    Rc<DxvkBuffer> surfaceMaterialBuffer = getSceneManager().getSurfaceMaterialBuffer();
    Rc<DxvkBuffer> surfaceMaterialExtensionBuffer = getSceneManager().getSurfaceMaterialExtensionBuffer();
    Rc<DxvkBuffer> volumeMaterialBuffer = getSceneManager().getVolumeMaterialBuffer();
    Rc<DxvkBuffer> lightBuffer = getSceneManager().getLightManager().getLightBuffer();
    Rc<DxvkBuffer> previousLightBuffer = getSceneManager().getLightManager().getPreviousLightBuffer();
    Rc<DxvkBuffer> lightMappingBuffer = getSceneManager().getLightManager().getLightMappingBuffer();
    Rc<DxvkBuffer> gpuPrintBuffer = getResourceManager().getRaytracingOutput().m_gpuPrintBuffer;
    Rc<DxvkBuffer> surfaceCoverageBuffer = getResourceManager().getRaytracingOutput().m_surfaceCoverageBuffer;
    Rc<DxvkImageView> valueNoiseLut = getResourceManager().getValueNoiseLut(this);
    Rc<DxvkSampler> linearSampler = getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    Rc<DxvkBuffer> samplerFeedbackBuffer = getResourceManager().getRaytracingOutput().m_samplerFeedbackDevice;
    // NV-DXVK: scene dump buffer — falls back to the 1-element placeholder
    // when no capture is in flight so the descriptor slot stays valid.
    Rc<DxvkBuffer> sceneDumpBuffer = getResourceManager().getRaytracingOutput().m_sceneDumpBuffer.ptr()
      ? getResourceManager().getRaytracingOutput().m_sceneDumpBuffer
      : getResourceManager().getRaytracingOutput().m_sceneDumpPlaceholder;

    DebugView& debugView = getCommonObjects()->metaDebugView();

    // NOTE: Opaque TLAS must be non-null here (callers are guarded by the tlasReady check in injectRTX).
    // Unordered and SSS TLAS may legitimately be null in some frames; fall back to Opaque to keep
    // descriptors valid and avoid VK_ERROR_DEVICE_LOST from null AS handles in the update template.
    const Rc<DxvkAccelStructure>& opaqueTlas   = getResourceManager().getTLAS(Tlas::Opaque).accelStructure;
    const Rc<DxvkAccelStructure>& prevTlas      = getResourceManager().getTLAS(Tlas::Opaque).previousAccelStructure;
    const Rc<DxvkAccelStructure>& unorderedTlas = getResourceManager().getTLAS(Tlas::Unordered).accelStructure;
    const Rc<DxvkAccelStructure>& sssTlas       = getResourceManager().getTLAS(Tlas::SSS).accelStructure;

    // NV-DXVK [SpawnGeomDiag.TlasBindAtFrame]: log the AS handles being
    // bound this frame so they can be cross-referenced against the build
    // calls logged from rtx_accel_manager (TlasBuildCall). If a ray hits an
    // instance with customInstanceIndex 8000+ while orderedSize=119, the AS
    // it hit must have been built with 8000+ instances — that means either
    // the bound opaqueTlas pointer matches a stale build, or the prevTlas
    // pointer holds a build that's older than expected. Log them all here.
    {
      static uint32_t s_tlasBindN = 0;
      const uint32_t n = s_tlasBindN++;
      if (n < 50u || (n % 30u) == 0u) {
        Logger::info(str::format(
          "[SpawnGeomDiag.TlasBindAtFrame] bind#", n,
          " opaqueTlas=0x", std::hex, reinterpret_cast<uintptr_t>(opaqueTlas.ptr()), std::dec,
          " prevTlas=0x", std::hex, reinterpret_cast<uintptr_t>(prevTlas.ptr()), std::dec,
          " unorderedTlas=0x", std::hex, reinterpret_cast<uintptr_t>(unorderedTlas.ptr()), std::dec,
          " sssTlas=0x", std::hex, reinterpret_cast<uintptr_t>(sssTlas.ptr()), std::dec,
          " prevEqualsCurrent=", (prevTlas.ptr() == opaqueTlas.ptr() ? 1 : 0),
          " unorderedFellBackToOpaque=", (unorderedTlas.ptr() == nullptr ? 1 : 0),
          " sssFellBackToOpaque=", (sssTlas.ptr() == nullptr ? 1 : 0)));
      }
    }

    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE,          opaqueTlas);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_PREVIOUS, prevTlas.ptr()      ? prevTlas      : opaqueTlas);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_UNORDERED, unorderedTlas.ptr() ? unorderedTlas : opaqueTlas);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_SSS,       sssTlas.ptr()       ? sssTlas       : opaqueTlas);
    bindResourceBuffer(BINDING_SURFACE_DATA_BUFFER, DxvkBufferSlice(surfaceBuffer, 0, surfaceBuffer->info().size));
    bindResourceBuffer(BINDING_SURFACE_MAPPING_BUFFER, DxvkBufferSlice(surfaceMappingBuffer, 0, surfaceMappingBuffer.ptr() ? surfaceMappingBuffer->info().size : 0));
    bindResourceBuffer(BINDING_SURFACE_MATERIAL_DATA_BUFFER, DxvkBufferSlice(surfaceMaterialBuffer, 0, surfaceMaterialBuffer->info().size));
    bindResourceBuffer(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER, surfaceMaterialExtensionBuffer.ptr() ? DxvkBufferSlice(surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionBuffer->info().size) : DxvkBufferSlice());
    bindResourceBuffer(BINDING_VOLUME_MATERIAL_DATA_BUFFER, volumeMaterialBuffer.ptr() ? DxvkBufferSlice(volumeMaterialBuffer, 0, volumeMaterialBuffer->info().size) : DxvkBufferSlice());
    bindResourceBuffer(BINDING_LIGHT_DATA_BUFFER, DxvkBufferSlice(lightBuffer, 0, lightBuffer.ptr() ? lightBuffer->info().size : 0));
    bindResourceBuffer(BINDING_PREVIOUS_LIGHT_DATA_BUFFER, DxvkBufferSlice(previousLightBuffer, 0, previousLightBuffer.ptr() ? previousLightBuffer->info().size : 0));
    bindResourceBuffer(BINDING_LIGHT_MAPPING, DxvkBufferSlice(lightMappingBuffer, 0, lightMappingBuffer.ptr() ? lightMappingBuffer->info().size : 0));
    bindResourceBuffer(BINDING_BILLBOARDS_BUFFER, DxvkBufferSlice(billboardsBuffer, 0, billboardsBuffer.ptr() ? billboardsBuffer->info().size : 0));
    bindResourceView(BINDING_BLUE_NOISE_TEXTURE, getResourceManager().getBlueNoiseTexture(this), nullptr);
    bindResourceBuffer(BINDING_CONSTANTS, DxvkBufferSlice(constantsBuffer, 0, constantsBuffer->info().size));
    bindResourceView(BINDING_DEBUG_VIEW_TEXTURE, debugView.getDebugOutput(), nullptr);
    bindResourceBuffer(BINDING_GPU_PRINT_BUFFER, DxvkBufferSlice(gpuPrintBuffer, 0, gpuPrintBuffer.ptr() ? gpuPrintBuffer->info().size : 0));
    bindResourceView(BINDING_VALUE_NOISE_SAMPLER, valueNoiseLut, nullptr);
    bindResourceSampler(BINDING_VALUE_NOISE_SAMPLER, linearSampler);
    bindResourceBuffer(BINDING_SAMPLER_READBACK_BUFFER, DxvkBufferSlice(samplerFeedbackBuffer, 0, samplerFeedbackBuffer.ptr() ? samplerFeedbackBuffer->info().size : 0));
    bindResourceBuffer(BINDING_SCENE_DUMP_BUFFER, DxvkBufferSlice(sceneDumpBuffer, 0, sceneDumpBuffer.ptr() ? sceneDumpBuffer->info().size : 0));
    bindResourceBuffer(BINDING_SURFACE_COVERAGE_BUFFER, DxvkBufferSlice(surfaceCoverageBuffer, 0, surfaceCoverageBuffer.ptr() ? surfaceCoverageBuffer->info().size : 0));

    // Bind atmosphere LUTs - must always bind since they're declared in common_bindings.slangh
    // Initialize atmosphere if not already done (needed for dummy resources)
    if (!m_atmosphere) {
      m_atmosphere = std::make_unique<RtxAtmosphere>(m_device.ptr());
    }
    // Always call initialize - it's idempotent (has internal m_initialized check)
    m_atmosphere->initialize(this);

    auto transmittanceLut = m_atmosphere->getTransmittanceLut();
    auto multiscatteringLut = m_atmosphere->getMultiscatteringLut();
    auto skyViewLut = m_atmosphere->getSkyViewLut();
    auto aerialPerspectiveLut = m_atmosphere->getAerialPerspectiveLut();

    // Always bind the LUTs (they're declared in shaders unconditionally)
    if (transmittanceLut.isValid()) {
      bindResourceView(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT, transmittanceLut.view, nullptr);
    }
    if (multiscatteringLut.isValid()) {
      bindResourceView(BINDING_ATMOSPHERE_MULTISCATTERING_LUT, multiscatteringLut.view, nullptr);
    }
    if (skyViewLut.isValid()) {
      bindResourceView(BINDING_ATMOSPHERE_SKY_VIEW_LUT, skyViewLut.view, nullptr);
    }
    if (aerialPerspectiveLut.isValid()) {
      bindResourceView(BINDING_ATMOSPHERE_AERIAL_PERSPECTIVE_LUT, aerialPerspectiveLut.view, nullptr);
    }
  }

  void RtxContext::bindResourceView(const uint32_t slot, const Rc<DxvkImageView>& imageView, const Rc<DxvkBufferView>& bufferView)
  {
    DxvkContext::bindResourceView(slot, imageView, bufferView);

#ifdef REMIX_DEVELOPMENT
    // Cache resources for aliasing
    cacheResourceAliasingImageView(imageView);
#endif
  }

  void RtxContext::checkOpacityMicromapSupport() {
    bool isOpacityMicromapSupported = OpacityMicromapManager::checkIsOpacityMicromapSupported(*m_device);
    RtxOptions::setIsOpacityMicromapSupported(isOpacityMicromapSupported);
    Logger::info(str::format("[RTX info] Opacity Micromap: ", isOpacityMicromapSupported ? "supported" : "not supported"));
  }

  bool RtxContext::checkIsShaderExecutionReorderingSupported(DxvkDevice& device) {
    if (!RtxOptions::isShaderExecutionReorderingSupported()) {
      return false;
    }

    // SER Extension support check
    const bool isSERExtensionSupported = device.extensions().nvRayTracingInvocationReorder;
    const bool isSERReorderingEnabled =
      VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV == device.properties().nvRayTracingInvocationReorderProperties.rayTracingInvocationReorderReorderingHint;
      
    return isSERExtensionSupported && isSERReorderingEnabled;
  }

  void RtxContext::checkShaderExecutionReorderingSupport() {
    const bool isSERSupported = checkIsShaderExecutionReorderingSupported(*m_device);
    
    RtxOptions::enableShaderExecutionReordering = isSERSupported;

    const VkPhysicalDeviceProperties& props = m_device->adapter()->deviceProperties();
    const NV_GPU_ARCHITECTURE_ID archId = RtxOptions::getNvidiaArch();

    Logger::info(str::format("[RTX info] Shader Execution Reordering: ", isSERSupported ? "supported" : "not supported"));

    bool isShaderExecutionReorderingEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerGbufferEnabled() ||
      RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled();

    Logger::info(str::format("[RTX info] Shader Execution Reordering: ", isShaderExecutionReorderingEnabled ? "enabled" : "disabled"));
  }

  void RtxContext::checkNeuralRadianceCacheSupport() {
    // Update RtxOption selection if Neural Radiance Cache was selected but it's not supported
    if (RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::NeuralRadianceCache &&
        !NeuralRadianceCache::checkIsSupported(m_device.ptr())) {

      // Fallback to ReSTIRGI
      Logger::warn(str::format("[RTX] Neural Radiance Cache is not supported. Switching indirect illumination mode to ReSTIR GI."));
      // TODO[REMIX-4105] trying to use NRC for a frame when it isn't supported will cause a crash, so this needs to be setImmediately.
      // Should refactor this to use a separate global for the final state, and indicate user preference with the option.
      // Use Quality layer to ensure this overrides the Environment layer (where env vars are stored).
      RtxOptions::integrateIndirectMode.setImmediately(IntegrateIndirectMode::ReSTIRGI, RtxOptionLayer::getQualityLayer());
    }
  }

  void RtxContext::dispatchVolumetrics(const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(this, "Volumetrics");
    setFramePassStage(RtxFramePassStage::Volumetrics);

    // Volume Raytracing
    {
      m_common->metaGlobalVolumetrics().dispatch(this, rtOutput, rtOutput.m_raytraceArgs.volumeArgs.numActiveFroxelVolumes);
    }
  }

  void RtxContext::dispatchIntegrate(const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(this, "Integrate Raytracing");

    // Integrate direct
    m_common->metaPathtracerIntegrateDirect().dispatch(this, rtOutput);

    // RTXDI Gradient pass
    m_common->metaRtxdiRayQuery().dispatchGradient(this, rtOutput);

    // Integrate indirect
    {
      ScopedGpuProfileZone(this, "Integrate Indirect Raytracing");
      setFramePassStage(RtxFramePassStage::IndirectIntegration);
      
      m_common->metaPathtracerIntegrateIndirect().dispatch(this, rtOutput);
    }

    // Integrate indirect - NEE Cache pass
    m_common->metaPathtracerIntegrateIndirect().dispatchNEE(this, rtOutput);
  }

  void RtxContext::dispatchPathTracing(const Resources::RaytracingOutput& rtOutput) {

    // Gbuffer Raytracing
    m_common->metaPathtracerGbuffer().dispatch(this, rtOutput);

    // NV-DXVK: visible-surface readback. Must run before any later pass aliases
    // m_sharedSurfaceIndex storage (e.g. m_primaryDisocclusionMaskForRR).
    recordVisibleSurfacesReadback(rtOutput);

    // RTXDI
    m_common->metaRtxdiRayQuery().dispatch(this, rtOutput);

    // NEE Cache
    dispatchNeeCache(rtOutput);

    // Integration Raytracing
    dispatchIntegrate(rtOutput);
  }
  
  void RtxContext::dispatchDemodulate(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();
    DemodulatePass& demodulate = m_common->metaDemodulate();
    demodulate.dispatch(this, rtOutput);
  }

  void RtxContext::dispatchNeeCache(const Resources::RaytracingOutput& rtOutput) {
    NeeCachePass& neeCache = m_common->metaNeeCache();
    neeCache.dispatch(this, rtOutput);
  }

  void RtxContext::dispatchDenoise(const Resources::RaytracingOutput& rtOutput) {
    auto& rayReconstruction = getCommonObjects()->metaRayReconstruction();

    // Primary direct denoiser used for primary direct lighting when separated, otherwise a special combined direct+indirect denoiser is used when both direct and indirect signals are combined.
    DxvkDenoise& denoiser0 = RtxOptions::denoiseDirectAndIndirectLightingSeparately() ? m_common->metaPrimaryDirectLightDenoiser() : m_common->metaPrimaryCombinedLightDenoiser();
    DxvkDenoise& referenceDenoiserSecondLobe0 = m_common->metaReferenceDenoiserSecondLobe0();
    // Primary Indirect denoiser used for primary indirect lighting when separated.
    DxvkDenoise& denoiser1 = m_common->metaPrimaryIndirectLightDenoiser();
    DxvkDenoise& referenceDenoiserSecondLobe1 = m_common->metaReferenceDenoiserSecondLobe1();
    // Secondary combined denoiser always used for secondary lighting.
    DxvkDenoise& denoiser2 = m_common->metaSecondaryCombinedLightDenoiser();
    DxvkDenoise& referenceDenoiserSecondLobe2 = m_common->metaReferenceDenoiserSecondLobe2();

    bool shouldDenoise = false;
    if (useRayReconstruction()) {
      shouldDenoise = (rayReconstruction.enableNRDForTraining() && !RtxOptions::useDenoiserReferenceMode()) || rayReconstruction.preprocessSecondarySignal();
    } else {
      shouldDenoise = RtxOptions::useDenoiser() && !RtxOptions::useDenoiserReferenceMode();
    }

    if (!shouldDenoise) {
      denoiser0.releaseResources();
      denoiser1.releaseResources();
      denoiser2.releaseResources();
      referenceDenoiserSecondLobe0.releaseResources();
      referenceDenoiserSecondLobe1.releaseResources();
      referenceDenoiserSecondLobe2.releaseResources();
      return;
    }

    ScopedGpuProfileZone(this, "Denoising");
    setFramePassStage(RtxFramePassStage::NRD);

    auto runDenoising = [&](DxvkDenoise& denoiser, DxvkDenoise& secondLobeReferenceDenoiser, DxvkDenoise::Input& denoiseInput, DxvkDenoise::Output& denoiseOutput) {
      // Since NRDContext doesn't use DxvkContext abstraction
      // but its using Compute, mark its DxvkContext's Cp pipelines as dirty
      {
        this->spillRenderPass(false);
        m_flags.set(
          DxvkContextFlag::CpDirtyPipeline,
          DxvkContextFlag::CpDirtyPipelineState,
          DxvkContextFlag::CpDirtyResources,
          DxvkContextFlag::CpDirtyDescriptorBinding);
      }

      // Need to run the denoiser twice for diffuse and specular when reference denoising is enabled on non-combined inputs
      if (denoiser.isReferenceDenoiserEnabled()) {
        denoiseInput.reference = denoiseInput.diffuse_hitT;
        denoiseOutput.reference = denoiseOutput.diffuse_hitT;
        denoiser.dispatch(this, m_execBarriers, rtOutput, denoiseInput, denoiseOutput);

        // Reference denoiser accumulates internally, so the second signal has to be denoised through a separate reference denoiser
        secondLobeReferenceDenoiser.copyNrdSettingsFrom(denoiser);
        denoiseInput.reference = denoiseInput.specular_hitT;
        denoiseOutput.reference = denoiseOutput.specular_hitT;
        secondLobeReferenceDenoiser.dispatch(this, m_execBarriers, rtOutput, denoiseInput, denoiseOutput);
      } else
        denoiser.dispatch(this, m_execBarriers, rtOutput, denoiseInput, denoiseOutput);
    };

    bool isSecondaryOnly = useRayReconstruction() && !rayReconstruction.enableNRDForTraining() && rayReconstruction.preprocessSecondarySignal();

    // Primary Direct light denoiser
    if (!isSecondaryOnly)
    {
      ScopedGpuProfileZone(this, "Primary Direct Denoising");
      
      DxvkDenoise::Input denoiseInput = {};
      denoiseInput.diffuse_hitT = &rtOutput.m_primaryDirectDiffuseRadiance.resource(Resources::AccessType::Read);
      denoiseInput.specular_hitT = &rtOutput.m_primaryDirectSpecularRadiance.resource(Resources::AccessType::Read);
      denoiseInput.normal_roughness = &rtOutput.m_primaryVirtualWorldShadingNormalPerceptualRoughnessDenoising.resource(Resources::AccessType::Read);
      denoiseInput.linearViewZ = &rtOutput.m_primaryLinearViewZ;
      denoiseInput.motionVector = &rtOutput.m_primaryVirtualMotionVector.resource(Resources::AccessType::Read);
      denoiseInput.disocclusionThresholdMix = &rtOutput.m_primaryDisocclusionThresholdMix;
      denoiseInput.reset = m_resetHistory;

      if (RtxOptions::useRTXDI() && m_common->metaRtxdiRayQuery().getEnableDenoiserConfidence(*this)) {
        denoiseInput.confidence = &rtOutput.getCurrentRtxdiConfidence().resource(Resources::AccessType::Read);
      }

      DxvkDenoise::Output denoiseOutput;
      denoiseOutput.diffuse_hitT = &rtOutput.m_primaryDirectDiffuseRadiance.resource(Resources::AccessType::Write);
      denoiseOutput.specular_hitT = &rtOutput.m_primaryDirectSpecularRadiance.resource(Resources::AccessType::Write);

      runDenoising(denoiser0, referenceDenoiserSecondLobe0, denoiseInput, denoiseOutput);
    } else {
      denoiser0.releaseResources();
      referenceDenoiserSecondLobe0.releaseResources();
    }

    // Primary Indirect light denoiser, if separate denoiser is used.
    if (RtxOptions::denoiseDirectAndIndirectLightingSeparately() && !isSecondaryOnly)
    {
      ScopedGpuProfileZone(this, "Primary Indirect Denoising");

      DxvkDenoise::Input denoiseInput = {};
      denoiseInput.diffuse_hitT = &rtOutput.m_primaryIndirectDiffuseRadiance.resource(Resources::AccessType::Read);
      denoiseInput.specular_hitT = &rtOutput.m_primaryIndirectSpecularRadiance.resource(Resources::AccessType::Read);
      denoiseInput.normal_roughness = &rtOutput.m_primaryVirtualWorldShadingNormalPerceptualRoughnessDenoising.resource(Resources::AccessType::Read);
      denoiseInput.linearViewZ = &rtOutput.m_primaryLinearViewZ;
      denoiseInput.motionVector = &rtOutput.m_primaryVirtualMotionVector.resource(Resources::AccessType::Read);
      denoiseInput.disocclusionThresholdMix = &rtOutput.m_primaryDisocclusionThresholdMix;
      denoiseInput.reset = m_resetHistory;

      DxvkDenoise::Output denoiseOutput;
      denoiseOutput.diffuse_hitT = &rtOutput.m_primaryIndirectDiffuseRadiance.resource(Resources::AccessType::Write);
      denoiseOutput.specular_hitT = &rtOutput.m_primaryIndirectSpecularRadiance.resource(Resources::AccessType::Write);

      runDenoising(denoiser1, referenceDenoiserSecondLobe1, denoiseInput, denoiseOutput);
    } else {
      denoiser1.releaseResources();
      referenceDenoiserSecondLobe1.releaseResources();
    }

    // Secondary Combined light denoiser
    {
      ScopedGpuProfileZone(this, "Secondary Combined Denoising");

      DxvkDenoise::Input denoiseInput = {};
      denoiseInput.diffuse_hitT = &rtOutput.m_secondaryCombinedDiffuseRadiance.resource(Resources::AccessType::Read);
      denoiseInput.specular_hitT = &rtOutput.m_secondaryCombinedSpecularRadiance.resource(Resources::AccessType::Read);
      denoiseInput.normal_roughness = &rtOutput.m_secondaryVirtualWorldShadingNormalPerceptualRoughnessDenoising;
      denoiseInput.linearViewZ = &rtOutput.m_secondaryLinearViewZ;
      denoiseInput.motionVector = &rtOutput.m_secondaryVirtualMotionVector.resource(Resources::AccessType::Read);
      denoiseInput.reset = m_resetHistory;

      DxvkDenoise::Output denoiseOutput;
      denoiseOutput.diffuse_hitT = &rtOutput.m_secondaryCombinedDiffuseRadiance.resource(Resources::AccessType::Write);
      denoiseOutput.specular_hitT = &rtOutput.m_secondaryCombinedSpecularRadiance.resource(Resources::AccessType::Write);

      runDenoising(denoiser2, referenceDenoiserSecondLobe2, denoiseInput, denoiseOutput);
    }
  }

  void RtxContext::dispatchDLSS(const Resources::RaytracingOutput& rtOutput) {
    DxvkDLSS& dlss = m_common->metaDLSS();
    dlss.dispatch(this, m_execBarriers, rtOutput, m_resetHistory);
  }

  void RtxContext::dispatchRayReconstruction(const Resources::RaytracingOutput& rtOutput) {
    DxvkRayReconstruction& rayReconstruction = m_common->metaRayReconstruction();
    rayReconstruction.dispatch(this, m_execBarriers, rtOutput, m_resetHistory, GlobalTime::get().deltaTimeMs());
  }

  void RtxContext::dispatchNIS(const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(this, "NIS");
    setFramePassStage(RtxFramePassStage::NIS);
    m_common->metaNIS().dispatch(this, rtOutput);
  }

  void RtxContext::dispatchXeSS(const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(this, "XeSS");
    setFramePassStage(RtxFramePassStage::XeSS);
    DxvkXeSS& xess = m_common->metaXeSS();
    xess.dispatch(this, m_execBarriers, rtOutput, m_resetHistory);
  }

  void RtxContext::dispatchTemporalAA(const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(this, "TAA");
    setFramePassStage(RtxFramePassStage::TAA);

    DxvkTemporalAA& taa = m_common->metaTAA();
    RtCamera& mainCamera = getSceneManager().getCamera();

    if (taa.isActive() && !mainCamera.isCameraCut()) {
      float jitterOffset[2];
      mainCamera.getJittering(jitterOffset);

      taa.dispatch(this,
        getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
        mainCamera.getShaderConstants().resolution,
        jitterOffset,
        rtOutput.m_compositeOutput.resource(Resources::AccessType::Read),
        rtOutput.m_primaryScreenSpaceMotionVector,
        rtOutput.m_finalOutput.resource(Resources::AccessType::Write),
        true);
    }
  }

  void RtxContext::dispatchComposite(const Resources::RaytracingOutput& rtOutput) {
    if (getSceneManager().getSurfaceBuffer() == nullptr) {
      return;
    }

    ScopedGpuProfileZone(this, "Composite");
    setFramePassStage(RtxFramePassStage::Composition);

    bool isNRDPreCompositionDenoiserEnabled = RtxOptions::useDenoiser() && !RtxOptions::useDenoiserReferenceMode();

    CompositePass::Settings settings;
    settings.fog = getSceneManager().getFogState();
    settings.isNRDPreCompositionDenoiserEnabled = isNRDPreCompositionDenoiserEnabled;
    settings.useUpscaler = shouldUseUpscaler();
    settings.useDLSS = shouldUseDLSS();
    settings.demodulateRoughness = m_common->metaDemodulate().demodulateRoughness();
    settings.roughnessDemodulationOffset = m_common->metaDemodulate().demodulateRoughnessOffset();
    m_common->metaComposite().dispatch(this,
      getSceneManager(),
      rtOutput, settings);
  }

  void RtxContext::dispatchToneMapping(const Resources::RaytracingOutput& rtOutput, bool performSRGBConversion) {
    ScopedCpuProfileZone();

    if (m_common->metaDebugView().debugViewIdx() == DEBUG_VIEW_PRE_TONEMAP_OUTPUT) {
      return;
    }

    // TODO: I think these are unnecessary, and/or should be automatically done within DXVK 
    this->spillRenderPass(false);
    this->unbindComputePipeline();

    DxvkAutoExposure& autoExposure = m_common->metaAutoExposure();    
    autoExposure.dispatch(this, 
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER),
      rtOutput, GlobalTime::get().deltaTimeMs(), performSRGBConversion);

    // We don't reset history for tonemapper on m_resetHistory for easier comparison when toggling raytracing modes.
    // The tone curve shouldn't be too different between raytracing modes, 
    // but the reset of denoised buffers causes wide tone curve differences
    // until it converges and thus making comparison of raytracing mode outputs more difficult
    setFramePassStage(RtxFramePassStage::ToneMapping);
    // NV-DXVK [tonemap operators]: the fork operators (Psycho17/GT7/Hable) live in
    // the GLOBAL tonemapper's apply shader. The default tonemappingMode is Local,
    // which bypasses the global path entirely — so when an operator is selected,
    // force the global tonemapper to run and skip the local one. Otherwise the
    // operator selection silently does nothing (the local tonemapper owns output).
    const bool operatorSelected =
      DxvkToneMapping::tonemapOperator() != DxvkToneMapping::TonemapOperator::None;
    if (RtxOptions::tonemappingMode() == TonemappingMode::Global || operatorSelected) {
      DxvkToneMapping& toneMapper = m_common->metaToneMapping();
      toneMapper.dispatch(this,
        getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER),
        autoExposure.getExposureTexture().view,
        rtOutput, GlobalTime::get().deltaTimeMs(), performSRGBConversion, autoExposure.enabled());
    }
    DxvkLocalToneMapping& localTonemapper = m_common->metaLocalToneMapping();
    if (localTonemapper.isActive() && !operatorSelected) {
      localTonemapper.dispatch(this,
        getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
        autoExposure.getExposureTexture().view,
        rtOutput, GlobalTime::get().deltaTimeMs(), performSRGBConversion, autoExposure.enabled());
    }

    // NV-DXVK [TonemapProbe]: capture tonemap in->out now, before bloom/post-fx
    // touch m_finalOutput, so the logged output is purely the tonemap operator's.
    captureTonemapProbe(rtOutput);
  }

  void RtxContext::dispatchDepthOfField(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();
    DxvkDepthOfField& dof = m_common->metaDepthOfField();
    if (!dof.isActive()) {
      return;
    }

    this->spillRenderPass(false);
    this->unbindComputePipeline();

    dof.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      rtOutput);
  }

  void RtxContext::dispatchBloom(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();
    DxvkBloom& bloom = m_common->metaBloom();
    if (!bloom.isActive()) {
      return;
    }

    // TODO: just in case, because tonemapping does the same
    this->spillRenderPass(false);
    this->unbindComputePipeline();

    bloom.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite));
  }

  void RtxContext::dispatchPostFx(Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();
    DxvkPostFx& postFx = m_common->metaPostFx();
    RtCamera& mainCamera = getSceneManager().getCamera();
    if (!postFx.enable()) {
      return;
    }

    postFx.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      mainCamera.getShaderConstants().resolution,
      RtxOptions::rngSeedWithFrameIndex() ? m_device->getCurrentFrameId() : 0,
      rtOutput,
      mainCamera.isCameraCut());
  }

  void RtxContext::dispatchDebugView(Rc<DxvkImage>& srcImage, const Resources::RaytracingOutput& rtOutput, bool captureScreenImage)  {
    ScopedCpuProfileZone();

    DebugView& debugView = m_common->metaDebugView();
    const uint32_t frameIdx = m_device->getCurrentFrameId();

    // NV-DXVK [ScreenSpaceEmissive.SlangProbe]: read back the dedicated
    // tail slot (kMaxFramesInFlight) of the GpuPrintBuffer that the opaque
    // material slang writes when its screen-space-emissive branch executes.
    // Confirms the cb.screenSpaceEmissiveTime value the slang actually
    // observed matches what C++ uploaded — verifies the slang side of the
    // c_gameTime plumb (the `[ScreenSpaceEmissive.GameTimeWatch]` log in
    // d3d11_rtx.cpp only shows the C++ capture, not what reached the GPU).
    // Throttled to ~1 Hz, gated on (a) screen-space-emissive ever having
    // fired this run (engineGameTime != 0) so we don't spam before the
    // holo character renders, and (b) the probe sentinel matching 999.0.
    if (rtOutput.m_gpuPrintBuffer.ptr() != nullptr) {
      const float currentEngineGt = getSceneManager().getEngineGameTime();
      if (currentEngineGt != 0.0f) {
        using clk = std::chrono::steady_clock;
        static auto sLastSseProbeLog = clk::now() - std::chrono::seconds(2);
        const auto now = clk::now();
        const auto sinceMs = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - sLastSseProbeLog).count();
        if (sinceMs >= 1000) {
          const VkDeviceSize sseProbeOffset =
            kMaxFramesInFlight * sizeof(GpuPrintBufferElement);
          GpuPrintBufferElement* sseProbe = reinterpret_cast<GpuPrintBufferElement*>(
            rtOutput.m_gpuPrintBuffer->mapPtr(sseProbeOffset));
          if (sseProbe != nullptr) {
            const Vector4& d = reinterpret_cast<Vector4&>(sseProbe->writtenData);
            const bool tagOK = (d.w == 999.0f);
            if (tagOK) {
              sLastSseProbeLog = now;
              const float slangSawTime = d.x;
              const uint32_t slangFrameIdx = static_cast<uint32_t>(d.y);
              const uint32_t packedThread  = static_cast<uint32_t>(d.z);
              const uint32_t threadX = packedThread & 0xFFFFu;
              const uint32_t threadY = packedThread >> 16;
              const float deltaVsCpu = slangSawTime - currentEngineGt;
              Logger::info(str::format(
                "[ScreenSpaceEmissive.SlangProbe] slang.cb.screenSpaceEmissiveTime=",
                slangSawTime,
                " cpu.engineGameTime=", currentEngineGt,
                " delta=", deltaVsCpu,
                " (slang frameIdx=", slangFrameIdx,
                " cpu frameIdx=", frameIdx,
                " lastWriterPixel=(", threadX, ",", threadY, "))",
                (std::abs(deltaVsCpu) < 0.4f
                  ? " — MATCH (within 1 frame of capture)"
                  : " — MISMATCH: slang sees a different value than CPU uploaded")));
            }
          }
        }
      }
    }

    const bool pathCheckerForceLogReadback =
      debugView.debugViewIdx() == DEBUG_VIEW_GRADIENT_PATH_CHECKER
      || debugView.debugViewIdx() == DEBUG_VIEW_RAW_ALBEDO;
    // NV-DXVK: null-check m_gpuPrintBuffer. The path-checker force-on
    // wakes this readback up on the very first frame view 52 is selected,
    // which can be earlier than RT resource allocation — dereferencing a
    // null Rc<DxvkBuffer> AVs in d3d11.dll. The original gpuPrint.enable()
    // path was protected by users having to manually enable from ImGui
    // (after RT resources were already up), so this guard wasn't needed
    // before.
    if ((debugView.gpuPrint.enable() || pathCheckerForceLogReadback)
        && rtOutput.m_gpuPrintBuffer.ptr() != nullptr) {
      // Read from the oldest element as it is guaranteed to be written on the GPU by now
      VkDeviceSize offset = ((frameIdx + 1) % kMaxFramesInFlight) * sizeof(GpuPrintBufferElement);
      GpuPrintBufferElement* gpuPrintElement = reinterpret_cast<GpuPrintBufferElement*>(rtOutput.m_gpuPrintBuffer->mapPtr(offset));

      if (gpuPrintElement && gpuPrintElement->isValid()) {
        static std::string previousString = "";
        const Vector4& d = reinterpret_cast<Vector4&>(gpuPrintElement->writtenData);
        std::string newString;
        const bool onPathCheckerView =
          debugView.debugViewIdx() == DEBUG_VIEW_GRADIENT_PATH_CHECKER;
        // Slot probes 5/6/7/11/12/13 fire on BOTH view 32 and view 52, so
        // disambiguate by `d.w` tag first regardless of view. PathChecker
        // packet writes are now suppressed (kEmitPathCheckerProbe=false in
        // opaque_surface_material_interaction.slangh) so any incoming
        // packet on view 52 should be a slot probe.
        const int slotTag = static_cast<int>(d.w);
        const bool isSlotProbe =
          (slotTag == 5 || slotTag == 6 || slotTag == 7
           || slotTag == 11 || slotTag == 12 || slotTag == 13);
        if (false /* see kEmitPathCheckerProbe */
            && pathCheckerForceLogReadback && onPathCheckerView && !isSlotProbe) {
          if (static_cast<int>(d.x) == 254) {
            // Alt packet: (sentinel, texIdx, boundDimsPacked = W*4096+H, totalMipBias)
            const int packed = static_cast<int>(d.z);
            const int boundW = packed / 4096;
            const int boundH = packed % 4096;
            newString = str::format(
              "[PathCheckerProbe] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") texIdx=", static_cast<int>(d.y),
              " boundAlbedo=", boundW, "x", boundH,
              " totalMipBias=", d.w);
          } else {
            // Per-frame physics packet: (pathCode, anisoMip, hitDistance, mipCount)
            newString = str::format(
              "[PathCheckerProbe] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") pathCode=", static_cast<int>(d.x),
              " anisoMip=", d.y,
              " distance=", d.z,
              " mipCount=", static_cast<int>(d.w));
          }
        } else {
          // RAW_ALBEDO view (32) cycles probe slots 5/6/7/11/12/13 in
          // surface_interaction.slangh. Each packet carries the slot tag in
          // d.w (5.0, 6.0, 7.0, ...). Pretty-print per-slot so the user can
          // read txcoords / positions / pathCode / clip.w / UV magnitudes
          // directly without translating raw float dumps.
          const int slot = static_cast<int>(d.w);
          if (slot == 5) {
            newString = str::format(
              "[GpuProbe slot5] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") txcoords[0]=(", d.x, ", ", d.y,
              ") txcoords[1].x=", d.z);
          } else if (slot == 6) {
            newString = str::format(
              "[GpuProbe slot6] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") txcoords[1].y=", d.x,
              " txcoords[2]=(", d.y, ", ", d.z, ")");
          } else if (slot == 7) {
            newString = str::format(
              "[GpuProbe slot7] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") positions[1].x=", d.x,
              " positions[2].x=", d.y,
              " |p1-p2|=", d.z);
          } else if (slot == 11) {
            newString = str::format(
              "[GpuProbe slot11] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") pathCode=", static_cast<int>(d.x),
              " maxComp=", d.y,
              " twoUVArea=", d.z);
          } else if (slot == 12) {
            newString = str::format(
              "[GpuProbe slot12] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") minClipW=", d.x,
              " maxClipW=", d.y,
              " ratio=", d.z);
          } else if (slot == 13) {
            newString = str::format(
              "[GpuProbe slot13] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") maxAbsUV=", d.x,
              " |uvEdge1|=", d.y,
              " |uvEdge2|=", d.z);
          } else if (slot == 14) {
            // True ellipse axes from the 2x2 Jacobian's singular values.
            // mip = log2(minor × texDim) when aniso ≤ aniMax.
            newString = str::format(
              "[GpuProbe slot14] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") major=", d.x,
              " minor=", d.y,
              " aniso=", d.z);
          } else if (slot == 20) {
            // [CloudPremultProbe] raw sampled albedo luminance / alpha / ratio
            // at the mouse pixel (translucent texels only). Point at a soft
            // cloud edge: lum/alpha ~<=1 => texture is premultiplied alpha
            // (rgb tracks coverage); lum/alpha >>1 => straight alpha
            // (full-bright rgb at low coverage).
            const char* verdict = (d.y > 0.02f && d.y < 0.85f)
              ? (d.z <= 1.2f ? "PREMULTIPLIED" : "STRAIGHT")
              : "(aim at a soft edge: alpha 0.02-0.85)";
            newString = str::format(
              "[CloudPremultProbe] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") albedoLum=", d.x,
              " alpha=", d.y,
              " lum/alpha=", d.z,
              " -> ", verdict);
          } else if (slot == 23) {
            // [CloudFogShader] confirms the TF2 cloud fog reconstruction
            // branch in opaque_surface_material_interaction.slangh actually
            // executed on the GPU. Its presence = the material flag, the
            // global-cb fog params, and the shader branch are all live.
            // colourLum = luminance of the reconstructed (fogged) albedo;
            // fogFactor = k0.w applied; sunAmount = the sun-tint term.
            newString = str::format(
              "[CloudFogShader] pixel(", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y,
              ") reconstructed colourLum=", d.x,
              " fogFactor=", d.y,
              " sunAmount=", d.z,
              " -> fog reconstruction RAN");
          } else {
            newString = str::format("GPU print value [", gpuPrintElement->threadIndex.x, ", ", gpuPrintElement->threadIndex.y, "]: ", Config::generateOptionString(reinterpret_cast<Vector4&>(gpuPrintElement->writtenData)));
          }
        }

        // Avoid spamming the console with the same output
        if (newString != previousString) {
          previousString = newString;

          // Add additional info on which we don't want to differentiate when printing out
          const std::string fullInfoString = str::format("Frame: ", gpuPrintElement->frameIndex, " - ", newString);
          Logger::info(fullInfoString);
        }

        // Invalidate the element so that it's not reused
        gpuPrintElement->invalidate();
      }
    }

    // NV-DXVK [Coverage]: surface-coverage histogram readback. The opaque
    // closesthit (opaque_surface_material_interaction.slangh) writes both
    // regions per primary hit, unconditionally regardless of the active
    // debug view:
    //   region 0 = encoded interaction.albedo > threshold (what Diffuse
    //              Albedo view displays — the polymorphic-decoded value)
    //   region 1 = raw-sampled albedo > threshold (what Diffuse Raw
    //              Albedo view displays — the texture/cavity/detail value
    //              before gamma / fog / scale-bias / metallic-F0 lerp)
    // A VS in region 0 but absent from region 1 means the transforms
    // between the raw sample and the final encoded value are fabricating
    // colour out of a zero source — that's the GBuffer-encode mismatch
    // behind the blot that appears in Diffuse Albedo + composite but not
    // in Diffuse Raw Albedo. We fold the per-surfaceIndex pixel counts
    // into per-vertex-shader-hash totals and log each region. Read is
    // deliberately loose (no fencing) — torn counts cost precision but
    // not correctness for a paused diagnostic. Throttled to one dump / 3
    // frames.
    //
    // Gameplay gate: skip menu / loading frames AND the first ~30 frames
    // after gameplay begins. Why both:
    //   1) Menu / loading frames have no world instances, so a readback there
    //      would only map the buffer and log an empty list.
    //   2) On the *first* frame where ordered-instances becomes non-empty,
    //      the AccelManager vector and the per-instance BlasEntry pointers
    //      are not yet stably wired — a TLAS rebuild + bucket reorder + BLAS
    //      attach all land on the same frame, and reading a half-wired
    //      RtInstance*/BlasEntry* here crashes (loading-into-gameplay
    //      crash was traced to this deref). A short warmup gives those
    //      structures time to settle into the steady-state layout we index
    //      with `reordered[surfaceIndex]` below. Kept small (5 frames ≈ 80 ms
    //      at 60 fps) so corruption that only manifests in the first second
    //      of gameplay still shows up in the dump.
    // NV-DXVK [Coverage]: gate. Only skip frames where getOrderedInstances()
    // is empty (loading screens / menus) — these can't produce useful dumps
    // anyway. The original half-wired-pointer crash that motivated the
    // multi-frame warmup is moot now: with per-frame TLAS rebuilds and the
    // AS-shrink realloc fix, the surfaces and BLAS entries are consistent
    // within a single frame by readback time. User explicitly wants every
    // gameplay frame logged.
    //
    // Probe: count how many times dispatchDebugView reached this code path
    // vs how many actually dumped. If they diverge significantly it means
    // dispatchDebugView itself is being called less often than render frames.
    static uint32_t s_coverageGateCallsTotal = 0;
    static uint32_t s_coverageGateCallsDumped = 0;
    static uint32_t s_coverageGateLastSummaryFrame = 0;
    static uint32_t s_coverageStableFrames = 0;
    ++s_coverageGateCallsTotal;
    const bool sceneHasInstances = !getSceneManager().getAccelManager().getOrderedInstances().empty();
    if (sceneHasInstances) {
      ++s_coverageStableFrames;
    } else {
      s_coverageStableFrames = 0;
    }

    // NV-DXVK [Coverage]: per-frame orderedSize ring buffer. Snapshot
    // UNCONDITIONALLY (outside the warmup/dump gate) so it captures every
    // frame from process start — including pre-warmup frames where
    // orderedSize was huge (13000+) but no dump fired. Without this, a
    // dump that happens after warmup sees the giant indices from earlier
    // GPU writes and misclassifies them as IMPOSSIBLE because the snapshot
    // ring is empty for the frames they actually belong to.
    struct FrameOrderedSnap { uint32_t frameId; uint32_t orderedSize; };
    constexpr size_t kSnapHistory = 64u;
    static std::array<FrameOrderedSnap, kSnapHistory> s_snapRing = {};
    static size_t s_snapWriteIdx = 0u;
    {
      const uint32_t curOrderedSize = static_cast<uint32_t>(
        getSceneManager().getAccelManager().getOrderedInstances().size());
      s_snapRing[s_snapWriteIdx] = {
        m_device->getCurrentFrameId(), curOrderedSize };
      s_snapWriteIdx = (s_snapWriteIdx + 1u) % kSnapHistory;
    }
    uint32_t recentMaxOrderedSize = 0u;
    for (const auto& s : s_snapRing) {
      if (s.orderedSize > recentMaxOrderedSize) recentMaxOrderedSize = s.orderedSize;
    }

    // NV-DXVK [Coverage]: parallel ring of (firstSlot -> vsHash) snapshots
    // for the most recent kOwnerHistory frames. Used by the UNMAPPED-
    // surfaceIndex fallback below to identify the PRIOR-FRAME owner of
    // a stale slot. The existing current-frame heuristic
    // (candidateOwnerVS = nearest current instance whose firstSlot <=
    // target) is misleading when the actual owner has already retired
    // — the prior-frame snapshot reveals who actually painted the
    // GBuffer pixel. Sized 16 to cover the temporal-accumulation window
    // over which a slot can still be classified STALE.
    struct FrameOwnerSnap {
      uint32_t frameId = UINT32_MAX;
      // Sorted ascending by firstSlot.
      std::vector<std::pair<uint32_t, uint64_t>> firstSlotToVs;
    };
    constexpr size_t kOwnerHistory = 16u;
    static std::array<FrameOwnerSnap, kOwnerHistory> s_ownerRing = {};
    static size_t s_ownerWriteIdx = 0u;
    {
      const uint32_t curFrameOw = m_device->getCurrentFrameId();
      // Skip if the most-recent entry is already this frame (defensive
      // — Coverage runs once per gameplay frame, but if dispatchDebugView
      // is re-entered the skip prevents double-pushing).
      const size_t mostRecentIdx =
        (s_ownerWriteIdx + kOwnerHistory - 1u) % kOwnerHistory;
      if (s_ownerRing[mostRecentIdx].frameId != curFrameOw) {
        FrameOwnerSnap& slot = s_ownerRing[s_ownerWriteIdx];
        slot.frameId = curFrameOw;
        slot.firstSlotToVs.clear();
        const auto& instTableOw =
          getSceneManager().getInstanceManager().getInstanceTable();
        slot.firstSlotToVs.reserve(instTableOw.size());
        for (const RtInstance* inst : instTableOw) {
          if (inst != nullptr) {
            const BlasEntry* blas = inst->getBlas();
            uint64_t vs = 0u;
            if (blas != nullptr) {
              vs = uint64_t(blas->input.getTransformData().vertexShaderHash);
            }
            slot.firstSlotToVs.emplace_back(inst->getSurfaceIndex(), vs);
          }
        }
        std::sort(slot.firstSlotToVs.begin(), slot.firstSlotToVs.end(),
                  [](const std::pair<uint32_t, uint64_t>& a,
                     const std::pair<uint32_t, uint64_t>& b) {
                    return a.first < b.first;
                  });
        s_ownerWriteIdx = (s_ownerWriteIdx + 1u) % kOwnerHistory;
      }
    }
    // Minimum 2-frame warmup: required to avoid the loading-into-gameplay
    // crash where the first non-empty frame's BlasEntry/DrawCallState
    // pointers are still half-wired and reading reordered[s]->getBlas()->
    // input.getTransformData() dereferences uninitialized memory. Without
    // this gate the game crashes during loading every time, before any
    // coverage data is even emitted. 2 frames is the minimum that gives
    // the TLAS-build + bucket-reorder + BLAS-attach a full frame to settle.
    constexpr uint32_t kCoverageWarmupFrames = 2u;
    const uint32_t curFrameId = m_device->getCurrentFrameId();
    if ((curFrameId - s_coverageGateLastSummaryFrame) >= 120u && curFrameId != s_coverageGateLastSummaryFrame) {
      Logger::info(str::format(
        "[SpawnGeomDiag.CoverageGate] frame=", curFrameId,
        " dispatchDebugView_calls_since_last_summary=", s_coverageGateCallsTotal,
        " dumps_actually_fired=", s_coverageGateCallsDumped,
        " sceneHasInstances_now=", (sceneHasInstances ? 1 : 0),
        " stableFrames=", s_coverageStableFrames));
      s_coverageGateCallsTotal = 0;
      s_coverageGateCallsDumped = 0;
      s_coverageGateLastSummaryFrame = curFrameId;
    }
    if (RtxOptions::logSurfaceCoverage()
        && rtOutput.m_surfaceCoverageBuffer.ptr() != nullptr
        && s_coverageStableFrames >= kCoverageWarmupFrames) {
      ++s_coverageGateCallsDumped;
      // NV-DXVK [PickRegion BUILD MARKER]: one-shot, unconditional, loud. The
      // ONLY purpose is to settle "is the freshly built d3d11.dll actually the
      // one the game loaded" without timestamp/process forensics. If you see
      // this line in remix-dxvk.log, the new readback IS running and a
      // [Coverage] PickRegion line MUST follow this frame. If you do NOT see it
      // while [Coverage] DebugViewScan lines appear, the game loaded a STALE
      // dll — redeploy. BUMP THE rev STRING every rebuild you want to verify.
      {
        static bool s_pickRegionBuildMarkerLogged = false;
        if (!s_pickRegionBuildMarkerLogged) {
          s_pickRegionBuildMarkerLogged = true;
          const Vector4 pr = RtxOptions::surfaceCoveragePickRegion();
          Logger::warn(str::format(
            "[PickRegion BUILD MARKER] rev=5 readback LOADED — frame=",
            m_device->getCurrentFrameId(),
            " logSurfaceCoverage=", (RtxOptions::logSurfaceCoverage() ? 1 : 0),
            " pickRegion=(", pr.x, ",", pr.y, ",", pr.z, ",", pr.w, ")"
            " — a [Coverage] PickRegion line should appear below each dump"));
        }
      }
      // NV-DXVK [Coverage]: dump every frame, no accumulation. Surface
      // indices in m_reorderedSurfaces are FRAME-LOCAL (cleared on every
      // TLAS build in rtx_accel_manager.cpp:475 and reassigned), so any
      // accumulation across frames mixes write-counts from one frame's
      // surfaceIndex layout against the lookup table of a later frame —
      // earlier-frame writes land at indices that don't exist in this
      // frame's `getOrderedInstances()`, showing up as huge "unmapped"
      // counts that are diagnostic noise rather than real bugs. Clearing
      // immediately after each read keeps every dump self-consistent: the
      // counts and the lookup table both belong to the same frame.
      {
        // NV-DXVK [Coverage]: optional GPU sync before readback. The
        // Coverage buffer is host-visible but the GPU's writes for THIS
        // frame's dispatchPathTracing are still in flight when we hit
        // mapPtr here (dispatchDebugView is recorded later in the same
        // command buffer). Without a wait-idle, the CPU sees whichever
        // frame the GPU last finished — typically N-2 with triple
        // buffering — so high-slot writes that were perfectly in-range
        // when the OLDER frame wrote them appear "OOB" only because the
        // dump compares them against the CURRENT frame's smaller
        // m_reorderedSurfaces.size(). That mismatch fully explains why
        // Region 0/1/15 show 1.46M "stale" hits at collapse frames while
        // the GPU's per-callsite OOB probe (Region 4 slot 0..4 / new
        // slot 7) reports zero — at write-time, no OOB ever happened.
        // Opt-in because vkDeviceWaitIdle every frame the dump runs is
        // not free.
        if (RtxOptions::coverageSyncBeforeReadback()) {
          m_device->waitForIdle();
        }
        uint32_t* cov = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
        if (cov != nullptr) {
          // NV-DXVK [Coverage]: which GPU frame actually produced this
          // data? The shader stamps cb.frameIdx into Region 4 slot 10
          // via InterlockedMax in ray_interaction.slangh. After
          // waitForIdle this slot holds the frameIdx of the LAST-
          // SUBMITTED frame — not the CPU's current recording frame.
          // Use it everywhere in this dump so the labels are honest
          // and align with [SpawnGeomDiag.CbSurfaceCount] frames.
          // priorOwner tracking and the "stale" classification below
          // are then meaningful: they compare against m_reordered
          // -Surfaces / instance state from the CPU's CURRENT frame,
          // so by definition they're cross-frame readings — useful
          // only if you remember that's what they are.
          const uint32_t gpuFrame = cov[4u * uint32_t(COVERAGE_SURFACE_SLOTS) + 10u];
          const uint32_t cpuRecordingFrame = m_device->getCurrentFrameId();
          // surfaceIndex -> RtInstance lookup. AccelManager::getOrderedInstances
          // is the flat array the GPU's surfaceIndex indexes into — multi-
          // surface instances (billboards, particles, viewmodel doubles) hold
          // multiple slots there, all addressable. Iterating instances and
          // taking only inst->getSurfaceIndex() (the FIRST slot per instance)
          // misses every other slot, which is why early dumps showed huge
          // unmappedPixels.
          const auto& reordered = getSceneManager().getAccelManager().getOrderedInstances();

          // NV-DXVK [Coverage PickRegion fast path]: rtx.coveragePickRegionOnly
          // skips the entire per-region histogram + color/red/NaN/DebugViewScan
          // logging below (the ~80 Logger lines/frame that crush FPS to ~0.16,
          // i.e. one pick sample every ~5s) and jumps straight to the
          // PickRegion / PickRegion2 blocks. Those need only `cov`, `reordered`
          // and `gpuFrame`, all already resolved above. The coverage buffer is
          // still cleared every frame (memset at the end, outside this guard).
          const bool coveragePickOnly = RtxOptions::coveragePickRegionOnly();
          if (!coveragePickOnly) {
          // Iterate the per-VS dump regions: 0..N-1 skipping 4. Region 4 is
          // the per-call-site orphan counter (16 fixed slots) and is dumped
          // separately below.
          for (uint32_t region = 0; region < COVERAGE_NUM_REGIONS; ++region) {
            if (region == 4u) continue;
            std::map<uint64_t, uint64_t> vsPixels; // vsHash -> summed pixel count
            // surfaceIndex -> pixel count for surfaces NOT in the live instance
            // table. NonOpaquePrimary turned up empty in TF2, so the blot is an
            // opaque surface whose surfaceIndex doesn't resolve to any RtInstance
            // (stale TLAS / sky-pass surfaces / billboards / decals — i.e.
            // anything outside the normal instance-buffer path). Dumping the
            // top-N surfaceIndex values here gives a concrete list to chase.
            std::vector<std::pair<uint32_t, uint64_t>> unmappedTop;
            uint64_t unmappedPixels = 0;
            // Split unmapped pixels into "likely in-flight stale" (the
            // surfaceIndex was valid in some recent frame's layout, so a
            // GPU write completing now against that prior frame's layout
            // is expected without a per-frame GPU sync) vs "impossible"
            // (the index exceeds *every* recent frame's orderedSize, so
            // there's no plausible frame in the snapshot window where it
            // was valid — that's a genuine bug).
            uint64_t unmappedPixels_stale = 0;
            uint64_t unmappedPixels_impossible = 0;
            const uint32_t base = region * uint32_t(COVERAGE_SURFACE_SLOTS);
            for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
              const uint32_t c = cov[base + s];
              if (c == 0u) {
                continue;
              }
              const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
              const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
              if (blas != nullptr) {
                const uint64_t vs = uint64_t(blas->input.getTransformData().vertexShaderHash);
                vsPixels[vs] += c;
              } else {
                unmappedPixels += c;
                unmappedTop.emplace_back(s, uint64_t(c));
                if (s < recentMaxOrderedSize) {
                  unmappedPixels_stale += c;
                } else {
                  unmappedPixels_impossible += c;
                }
              }
            }
            // Region naming. Regions 5..9 isolate the encode-pipeline stage
            // that introduces drift between raw-sampled and final-encoded
            // albedo (the blot signature). Regions 10..14 are co-occurrence
            // flags — comparing a flag's top-VS list against the AlbedoDrift
            // top-VS list reveals which auxiliary branch coincides with the
            // blot. Per-VS lines sorted desc so the worst offender is first.
            const char* regionName =
                (region == 0u)  ? "EncodedNonzero" :
                (region == 1u)  ? "RawNonzero" :
                (region == 2u)  ? "OpaquePrimary" :
                (region == 3u)  ? "TranslucentPrimary" :
                (region == 5u)  ? "AlbedoDrift" :
                (region == 6u)  ? "DriftStageGamma" :
                (region == 7u)  ? "DriftStageScaleBias" :
                (region == 8u)  ? "DriftStageMetallic" :
                (region == 9u)  ? "DriftStageAdjusted" :
                (region == 10u) ? "MetallicHigh" :
                (region == 11u) ? "OpacityLow" :
                (region == 12u) ? "IsMatteHits" :
                (region == 13u) ? "IsTf2SkyboxFogHits" :
                (region == 14u) ? "MetallicLoaded" :
                (region == 15u) ? "FlagPremultSet" :
                                  "DecalPrimaryHit";
            // NV-DXVK [Coverage]: container sizes in the dump header so we
            // can immediately distinguish "out-of-bounds surfaceIndex" (stale
            // GBuffer / index > orderedSize) from "in-bounds but nullptr"
            // (instance pointer cleared this frame). The instance-table size
            // is also reported as a sanity check — the CPU might have many
            // instances total but only a subset wired into the ordered vector.
            const size_t instanceTableSize = getSceneManager().getInstanceManager().getInstanceTable().size();
            Logger::info(str::format(
              "[Coverage] === ", regionName, " === gpuFrame=", gpuFrame,
              " cpuFrame=", cpuRecordingFrame,
              " distinctVS=", vsPixels.size(),
              " unmappedPixels=", unmappedPixels,
              " (stale=", unmappedPixels_stale,
              " impossible=", unmappedPixels_impossible,
              ")",
              " unmappedSurfaces=", unmappedTop.size(),
              " orderedSize=", reordered.size(),
              " recentMaxOrderedSize=", recentMaxOrderedSize,
              " instanceTableSize=", instanceTableSize,
              " (stale=valid-in-last-16-frames, impossible=never-recently-valid)"));
            // Sort per-VS counts by pixel count descending so the worst-
            // offender VS is the first line printed for each region. The
            // std::map iteration order is by VS hash, which is meaningless
            // for spotting the dominant surface — desc-by-pixels lets the
            // user scan one line per region to find the candidate.
            std::vector<std::pair<uint64_t, uint64_t>> vsSorted(vsPixels.begin(), vsPixels.end());
            std::sort(vsSorted.begin(), vsSorted.end(),
                      [](const std::pair<uint64_t, uint64_t>& a,
                         const std::pair<uint64_t, uint64_t>& b) {
                        return a.second > b.second;
                      });
            for (const auto& kv : vsSorted) {
              char vsHex[24];
              snprintf(vsHex, sizeof(vsHex), "0x%016llx", static_cast<unsigned long long>(kv.first));
              Logger::info(str::format(
                "[Coverage]   ", regionName, " VS=", vsHex, " pixels=", kv.second));
            }

            // Sort unmapped surfaces by pixel count (desc) and log the top 16.
            // 16 is enough to surface the dominant blot without flooding the
            // log when a frame happens to touch many small unmapped surfaces.
            std::sort(unmappedTop.begin(), unmappedTop.end(),
                      [](const std::pair<uint32_t, uint64_t>& a,
                         const std::pair<uint32_t, uint64_t>& b) {
                        return a.second > b.second;
                      });
            const size_t kTopUnmapped = 16;
            const size_t shown = std::min(kTopUnmapped, unmappedTop.size());

            // NV-DXVK [Coverage]: fallback lookup for unmapped surfaceIndex.
            // When reordered[s] is nullptr or s >= reordered.size(), try to
            // identify the owning RtInstance via the full instance table.
            // We build a sorted list of (firstSurfaceIndex, RtInstance*) and
            // find the *last* instance whose firstSurfaceIndex <= target;
            // multi-surface bucket inserts allocate consecutive slots so
            // that's most likely the owning instance.
            //
            // Built lazily here (per region) — only run if there's anything
            // unmapped to look up.
            if (shown > 0u) {
              const auto& instTable = getSceneManager().getInstanceManager().getInstanceTable();
              std::vector<std::pair<uint32_t, const RtInstance*>> firstSlotIndex;
              firstSlotIndex.reserve(instTable.size());
              for (const RtInstance* inst : instTable) {
                if (inst != nullptr) {
                  firstSlotIndex.emplace_back(inst->getSurfaceIndex(), inst);
                }
              }
              std::sort(firstSlotIndex.begin(), firstSlotIndex.end(),
                        [](const std::pair<uint32_t, const RtInstance*>& a,
                           const std::pair<uint32_t, const RtInstance*>& b) {
                          return a.first < b.first;
                        });

              for (size_t i = 0; i < shown; ++i) {
                const uint32_t target = unmappedTop[i].first;
                // upper_bound -> first entry with firstSlot > target; step back to
                // get the last entry with firstSlot <= target (the candidate owner).
                auto upper = std::upper_bound(
                  firstSlotIndex.begin(), firstSlotIndex.end(),
                  std::make_pair(target, static_cast<const RtInstance*>(nullptr)),
                  [](const std::pair<uint32_t, const RtInstance*>& a,
                     const std::pair<uint32_t, const RtInstance*>& b) {
                    return a.first < b.first;
                  });
                const RtInstance* candidate = (upper != firstSlotIndex.begin())
                  ? (upper - 1)->second
                  : nullptr;
                const BlasEntry* candBlas = (candidate != nullptr) ? candidate->getBlas() : nullptr;
                uint64_t candVs = 0u;
                uint32_t candFirstSlot = 0u;
                uint32_t candBillboards = 0u;
                if (candidate != nullptr) {
                  candFirstSlot = candidate->getSurfaceIndex();
                  candBillboards = candidate->getBillboardCount();
                }
                if (candBlas != nullptr) {
                  candVs = uint64_t(candBlas->input.getTransformData().vertexShaderHash);
                }
                char candVsHex[24];
                snprintf(candVsHex, sizeof(candVsHex), "0x%016llx",
                         static_cast<unsigned long long>(candVs));
                const uint32_t candDelta = (candidate != nullptr && target >= candFirstSlot)
                  ? (target - candFirstSlot) : 0u;
                const char* classification = (target < recentMaxOrderedSize)
                  ? "STALE" : "IMPOSSIBLE";

                // NV-DXVK [Coverage]: prior-owner lookup. Walk the
                // per-frame snapshot ring NEWEST-FIRST (skipping current
                // frame). For each snapshot, find the largest firstSlot
                // <= target. Report the FIRST hit (most recent prior
                // frame containing the slot). This is the ACTUAL owner
                // of the stale GBuffer pixel — the current-frame
                // candidateOwnerVS is just the nearest still-alive
                // neighbor, which can be misleading.
                uint64_t priorOwnerVs = 0ull;
                uint32_t priorOwnerFrame = 0u;
                uint32_t priorOwnerFirstSlot = 0u;
                uint32_t priorOwnerDelta = 0u;
                {
                  const uint32_t curFrameNow = m_device->getCurrentFrameId();
                  for (size_t step = 1u; step <= kOwnerHistory; ++step) {
                    const size_t idx =
                      (s_ownerWriteIdx + kOwnerHistory - step) % kOwnerHistory;
                    const auto& snap = s_ownerRing[idx];
                    if (snap.frameId == UINT32_MAX
                        || snap.frameId == curFrameNow) continue;
                    if (snap.firstSlotToVs.empty()) continue;
                    auto upper = std::upper_bound(
                      snap.firstSlotToVs.begin(), snap.firstSlotToVs.end(),
                      std::make_pair(target, uint64_t(0)),
                      [](const std::pair<uint32_t, uint64_t>& a,
                         const std::pair<uint32_t, uint64_t>& b) {
                        return a.first < b.first;
                      });
                    if (upper != snap.firstSlotToVs.begin()) {
                      auto entry = upper - 1;
                      // Defensive cap: require the slot to fall within
                      // 256 of the candidate firstSlot. Slots are
                      // typically allocated in small consecutive bursts
                      // per instance — a delta of thousands means we're
                      // looking at the wrong neighbor.
                      const uint32_t snapDelta = target - entry->first;
                      if (snapDelta < 256u) {
                        priorOwnerVs = entry->second;
                        priorOwnerFrame = snap.frameId;
                        priorOwnerFirstSlot = entry->first;
                        priorOwnerDelta = snapDelta;
                        break;
                      }
                    }
                  }
                }
                char priorOwnerVsHex[24];
                snprintf(priorOwnerVsHex, sizeof(priorOwnerVsHex),
                         "0x%016llx",
                         static_cast<unsigned long long>(priorOwnerVs));

                Logger::info(str::format(
                  "[Coverage]   ", regionName,
                  " UNMAPPED surfaceIndex=", target,
                  " class=", classification,
                  " pixels=", unmappedTop[i].second,
                  " candidateOwnerVS=", candVsHex,
                  " ownerFirstSlot=", candFirstSlot,
                  " ownerBillboards=", candBillboards,
                  " delta=", candDelta,
                  " priorOwnerVS=", priorOwnerVsHex,
                  " priorOwnerFrame=", priorOwnerFrame,
                  " priorOwnerFirstSlot=", priorOwnerFirstSlot,
                  " priorOwnerDelta=", priorOwnerDelta));
              }
            }
          }

          // NV-DXVK [Coverage color]: per-VS AVERAGE raw diffuse albedo.
          // The shader accumulates saturated*255 raw-albedo RGB sums (regions
          // 17/18/19) under the same gate/count as region 1 (RawNonzero), so
          // sum/count recovers the mean raw albedo per surface. Grouped by VS
          // and printed desc by pixel count — the line whose avg255≈(255,0,0)
          // is the bright-red corruption shader.
          {
            struct ColorAccum { uint64_t px = 0, r = 0, g = 0, b = 0, redPx = 0; };
            std::map<uint64_t, ColorAccum> vsColor;
            const uint32_t base1 = 1u * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t baseR = COVERAGE_RAWALBEDO_R_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t baseG = COVERAGE_RAWALBEDO_G_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t baseB = COVERAGE_RAWALBEDO_B_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t baseRed = COVERAGE_REDPIXEL_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
              const uint32_t px = cov[base1 + s];
              if (px == 0u) {
                continue;
              }
              const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
              const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
              const uint64_t vs = (blas != nullptr)
                ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
              ColorAccum& a = vsColor[vs];
              a.px += px;
              a.r  += cov[baseR + s];
              a.g  += cov[baseG + s];
              a.b  += cov[baseB + s];
              a.redPx += cov[baseRed + s];
            }
            std::vector<std::pair<uint64_t, ColorAccum>> sorted(vsColor.begin(), vsColor.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const std::pair<uint64_t, ColorAccum>& a,
                         const std::pair<uint64_t, ColorAccum>& b) {
                        return a.second.px > b.second.px;
                      });
            for (const auto& kv : sorted) {
              const ColorAccum& a = kv.second;
              const double inv = (a.px > 0) ? 1.0 / double(a.px) : 0.0;
              char vsHex[24];
              snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                       static_cast<unsigned long long>(kv.first));
              Logger::info(str::format(
                "[Coverage]   RawAlbedoColor VS=", vsHex,
                " avg255=(", uint32_t(a.r * inv + 0.5), ",",
                             uint32_t(a.g * inv + 0.5), ",",
                             uint32_t(a.b * inv + 0.5), ")",
                " pixels=", a.px));
            }
            // Red-dominant pixel count per VS (the averaging above hides a small
            // bright-red blot; this counts it directly). Sorted desc, only VS
            // that actually contribute red pixels — every such line is a shader
            // drawing the red corruption.
            std::vector<std::pair<uint64_t, ColorAccum>> redSorted(vsColor.begin(), vsColor.end());
            std::sort(redSorted.begin(), redSorted.end(),
                      [](const std::pair<uint64_t, ColorAccum>& a,
                         const std::pair<uint64_t, ColorAccum>& b) {
                        return a.second.redPx > b.second.redPx;
                      });
            for (const auto& kv : redSorted) {
              if (kv.second.redPx == 0u) {
                break;
              }
              char vsHex[24];
              snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                       static_cast<unsigned long long>(kv.first));
              Logger::info(str::format(
                "[Coverage]   RedPixels VS=", vsHex,
                " redPixels=", kv.second.redPx,
                " ofTotal=", kv.second.px));
            }

            // NV-DXVK [Coverage red]: also group red pixels by the COLOR-TEXTURE
            // IMAGE HASH (what rtx.hideInstanceTextures actually matches) and by
            // surfaceIndex. The dominant VS is a generic world shader (many
            // textures), so the VS grouping can't be hidden — but the specific
            // red material/texture can. Per-surface so we also expose the
            // (unstable) surfaceIndex slots feeding the blot.
            {
              std::map<uint64_t, uint64_t> texRed;   // colorTexImageHash -> red px
              std::map<uint64_t, uint64_t> matRed;   // material hash      -> red px
              std::map<uint64_t, uint64_t> psRed;    // pixelShaderHash    -> red px
              std::vector<std::pair<uint32_t, uint32_t>> surfRed; // (surfaceIndex, red px)
              for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
                const uint32_t red = cov[baseRed + s];
                if (red == 0u) {
                  continue;
                }
                surfRed.emplace_back(s, red);
                const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
                const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
                if (blas != nullptr) {
                  const auto& md = blas->input.getMaterialData();
                  texRed[uint64_t(md.getColorTexture().getImageHash())] += red;
                  matRed[uint64_t(md.getHash())] += red;
                  psRed[uint64_t(blas->input.getTransformData().pixelShaderHash)] += red;
                }
              }
              for (const auto& kv : texRed) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   RedTexture colorImageHash=", h, " redPixels=", kv.second));
              }
              for (const auto& kv : matRed) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   RedMaterial matHash=", h, " redPixels=", kv.second));
              }
              // PS hash → match against the FS_*.dxbc in shader_dumps/ to decompile.
              for (const auto& kv : psRed) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   RedPS pixelShaderHash=", h, " redPixels=", kv.second));
              }
              std::sort(surfRed.begin(), surfRed.end(),
                        [](const std::pair<uint32_t,uint32_t>& a, const std::pair<uint32_t,uint32_t>& b){ return a.second > b.second; });
              const size_t nShow = std::min<size_t>(8, surfRed.size());
              for (size_t i = 0; i < nShow; ++i) {
                Logger::info(str::format("[Coverage]   RedSurface surfaceIndex=", surfRed[i].first, " redPixels=", surfRed[i].second));
              }
              // World-space AABB of red pixels (decode biased InterlockedMax at
              // slot 0 of each pos region). Match against the o2wT/position
              // fields in the CPU instance logs to pin the physical object.
              const uint32_t mx = cov[COVERAGE_REDPOS_MAXX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t nx = cov[COVERAGE_REDPOS_MINX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t my = cov[COVERAGE_REDPOS_MAXY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t ny = cov[COVERAGE_REDPOS_MINY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t mz = cov[COVERAGE_REDPOS_MAXZ_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t nz = cov[COVERAGE_REDPOS_MINZ_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              if (mx != 0u || my != 0u || mz != 0u) { // any red pixel captured
                const double B = double(COVERAGE_REDPOS_BIAS);
                Logger::info(str::format(
                  "[Coverage]   RedWorldAABB min=(", (B - double(nx)), ",", (B - double(ny)), ",", (B - double(nz)), ")",
                  " max=(", (double(mx) - B), ",", (double(my) - B), ",", (double(mz) - B), ")"));
              }
              // Screen-space bbox: compact box vs scattered speckle.
              const uint32_t sMaxX = cov[COVERAGE_REDSCR_MAXX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMinXv = cov[COVERAGE_REDSCR_MINX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMaxY = cov[COVERAGE_REDSCR_MAXY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMinYv = cov[COVERAGE_REDSCR_MINY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              if (sMinXv != 0u || sMinYv != 0u) { // red pixels captured
                const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(sMinXv);
                const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(sMinYv);
                Logger::info(str::format(
                  "[Coverage]   RedScreenBox x=[", minX, ",", sMaxX, "] y=[", minY, ",", sMaxY, "]",
                  " w=", (int32_t(sMaxX) - minX), " h=", (int32_t(sMaxY) - minY)));
              }
              // Hit-distance stats of red pixels: clustered-at-ship-depth (wrong
              // albedo) vs wide/anomalous (wrong-hit precision).
              uint64_t totalRed = 0;
              for (const auto& sr : surfRed) { totalRed += sr.second; }
              const uint32_t dMax = cov[COVERAGE_REDDIST_MAX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t dMinV = cov[COVERAGE_REDDIST_MIN_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t dSum = cov[COVERAGE_REDDIST_SUM_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              if (totalRed > 0 && dMinV != 0u) {
                const double dMin = double(COVERAGE_REDDIST_BIAS) - double(dMinV);
                Logger::info(str::format(
                  "[Coverage]   RedHitDist min=", dMin, " max=", dMax,
                  " avg=", (double(dSum) / double(totalRed)), " count=", totalRed));
              }
              // NV-DXVK [Coverage red sample/mip]: discriminator readout.
              if (totalRed > 0) {
                // Bare pre-modulation albedo sample (avg255). ≈(255,0,0) ⇒ the
                // texture sample itself is red (UV/mip/texel); normal-brown ⇒
                // modulation fabricated the red.
                const uint32_t sr = cov[COVERAGE_REDSAMPLE_R_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const uint32_t sg = cov[COVERAGE_REDSAMPLE_G_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const uint32_t sb = cov[COVERAGE_REDSAMPLE_B_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                Logger::info(str::format(
                  "[Coverage]   RedBareSample avg255=(", (sr / totalRed), ",", (sg / totalRed), ",", (sb / totalRed), ")"));
                // Texture-gradient magnitude stats. Blown-up/collapsed vs a
                // healthy spread = bad mip selection.
                const uint32_t gMaxV = cov[COVERAGE_REDGRAD_MAX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const uint32_t gMinV = cov[COVERAGE_REDGRAD_MIN_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const uint32_t gSum = cov[COVERAGE_REDGRAD_SUM_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const double gMin = (double(COVERAGE_REDGRAD_BIAS) - double(gMinV)) / double(COVERAGE_REDGRAD_SCALE);
                Logger::info(str::format(
                  "[Coverage]   RedTexGrad min=", gMin,
                  " max=", (double(gMaxV) / double(COVERAGE_REDGRAD_SCALE)),
                  " avg=", ((double(gSum) / double(totalRed)) / double(COVERAGE_REDGRAD_SCALE))));
                // Gradient-pipeline branch mask. bit4=cap-fired(cone-iso),
                // bit5=NaN/Inf, bit1=behind-near, bit3=interpInvW-degen,
                // bit0=perspective-ok, bit10=1D-fallback.
                const uint32_t pathMask = cov[COVERAGE_REDPATH_MASK_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                Logger::info(str::format(
                  "[Coverage]   RedPathMask 0x", std::hex, pathMask, std::dec,
                  " (bit0=ok bit1=behindNear bit2=subpxDet bit3=invW bit4=capFired bit5=NaN bit10=1D)"));
                // Detail-stage confirmation: detail sample colour, which
                // modulation stages ran, and the detail texture index range.
                const uint32_t modMask = cov[COVERAGE_REDMOD_MASK_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                const uint32_t detCount = cov[COVERAGE_REDDETAIL_COUNT_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                Logger::info(str::format(
                  "[Coverage]   RedModMask 0x", std::hex, modMask, std::dec,
                  " (bit0=ao bit1=detail bit2=cloud) redWithDetail=", detCount, " ofRed=", totalRed));
                if (detCount > 0) {
                  const uint32_t dr = cov[COVERAGE_REDDETAIL_R_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                  const uint32_t dg = cov[COVERAGE_REDDETAIL_G_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                  const uint32_t db = cov[COVERAGE_REDDETAIL_B_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                  const uint32_t diMax = cov[COVERAGE_REDDETAILIDX_MAX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                  const uint32_t diMinV = cov[COVERAGE_REDDETAILIDX_MIN_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
                  const int64_t diMin = int64_t(COVERAGE_DETAILIDX_BIAS) - int64_t(diMinV);
                  Logger::info(str::format(
                    "[Coverage]   RedDetailSample avg255=(", (dr / detCount), ",", (dg / detCount), ",", (db / detCount), ")",
                    " detailTexIdx=[", diMin, ",", diMax, "]"));
                }
              }
            }
          }

          // NV-DXVK [Coverage NaN]: per-VS count of NaN/Inf-albedo primary hits
          // (region 48), the screen bbox of those pixels (49-52), and the
          // out-of-range count (53). This is the ONLY probe that sees the
          // pixels the debug-view Inf/NaN visualizer paints red — they're
          // invisible to every region above (saturate(NaN)->0). If NanPixels
          // lines appear, the matching VS is the red plane's source shader; if
          // the count is ~0 but the screen is still red, the NaN is NOT in an
          // opaque-hit albedo (look at the miss/sky path or the debug-view
          // clear value instead).
          {
            const uint32_t baseNan = COVERAGE_NAN_COUNT_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            std::map<uint64_t, uint64_t> vsNan;     // VS hash -> NaN px
            uint64_t nanTotal = 0;
            for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
              const uint32_t px = cov[baseNan + s];
              if (px == 0u) {
                continue;
              }
              nanTotal += px;
              const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
              const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
              const uint64_t vs = (blas != nullptr)
                ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
              vsNan[vs] += px;
            }
            const uint32_t nanOor = cov[COVERAGE_NAN_OOR_COUNT_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
            if (nanTotal > 0 || nanOor > 0) {
              std::vector<std::pair<uint64_t, uint64_t>> nanSorted(vsNan.begin(), vsNan.end());
              std::sort(nanSorted.begin(), nanSorted.end(),
                        [](const std::pair<uint64_t, uint64_t>& a,
                           const std::pair<uint64_t, uint64_t>& b) { return a.second > b.second; });
              for (const auto& kv : nanSorted) {
                char vsHex[24];
                snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                         static_cast<unsigned long long>(kv.first));
                Logger::info(str::format(
                  "[Coverage]   NanPixels VS=", vsHex, " nanPixels=", kv.second));
              }
              // Screen-space bbox of the NaN pixels: frame-spanning ⇒ fullscreen
              // plane (the engine-hook red plane); compact ⇒ one sub-mesh.
              const uint32_t sMaxX = cov[COVERAGE_NANSCR_MAXX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMinXv = cov[COVERAGE_NANSCR_MINX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMaxY = cov[COVERAGE_NANSCR_MAXY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              const uint32_t sMinYv = cov[COVERAGE_NANSCR_MINY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS)];
              if (sMinXv != 0u || sMinYv != 0u) {
                const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(sMinXv);
                const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(sMinYv);
                Logger::info(str::format(
                  "[Coverage]   NanScreenBox x=[", minX, ",", sMaxX, "] y=[", minY, ",", sMaxY, "]",
                  " w=", (int32_t(sMaxX) - minX), " h=", (int32_t(sMaxY) - minY),
                  " nanTotal=", nanTotal, " nanOutOfRange=", nanOor));
              } else {
                Logger::info(str::format(
                  "[Coverage]   NanSummary nanTotal=", nanTotal, " nanOutOfRange=", nanOor));
              }
            }
          }

          // NV-DXVK [Coverage DebugViewScan]: ground-truth scan of the ACTUAL
          // displayed DEBUG_VIEW_RAW_ALBEDO buffer (written by the debug
          // postprocess, one thread per screen pixel). Unlike the surface-side
          // probes above this sees the pixels the visible red plane is made of
          // even when they don't come through the opaque-closesthit albedo path
          // (the red plane survived disabling the detail overlay, proving it
          // isn't opaque albedo). redBox vs nanBox tells red-content from NaN.
          {
            const uint32_t baseDv = COVERAGE_DVSCAN_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t dvTotal = cov[baseDv + COVERAGE_DVSCAN_SLOT_TOTAL];
            const uint32_t dvRed = cov[baseDv + COVERAGE_DVSCAN_SLOT_RED];
            const uint32_t dvNan = cov[baseDv + COVERAGE_DVSCAN_SLOT_NAN];
            const uint32_t dvInf = cov[baseDv + COVERAGE_DVSCAN_SLOT_INF];
            if (dvTotal > 0u) {
              Logger::info(str::format(
                "[Coverage] DebugViewScan scanned=", dvTotal,
                " redPixels=", dvRed, " inputRedPixels=", cov[baseDv + COVERAGE_DVSCAN_SLOT_INPUTRED],
                " nanPixels=", dvNan, " infPixels=", dvInf));
              // Gate-free per-pixel grid dump of the DISPLAYED color: one line per
              // grid row, each cell = exact "r,g,b" (0-255) of that screen pixel.
              // No threshold, no averaging — read the literal value at the red.
              {
                const uint32_t gBase = baseDv + COVERAGE_DVSCAN_GRID_BASE;
                for (uint32_t row = 0; row < COVERAGE_DVSCAN_GRID_ROWS; ++row) {
                  std::string line;
                  for (uint32_t col = 0; col < COVERAGE_DVSCAN_GRID_COLS; ++col) {
                    const uint32_t idx = row * COVERAGE_DVSCAN_GRID_COLS + col;
                    const uint32_t g = gBase + idx * 3u;
                    line += str::format(" ", cov[g + 0u], ",", cov[g + 1u], ",", cov[g + 2u]);
                  }
                  Logger::info(str::format("[Coverage]   DebugViewGrid row=", row, line));
                }
              }
              if (dvRed > 0u) {
                const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_RED_MINX]);
                const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_RED_MINY]);
                const int32_t maxX = int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_RED_MAXX]);
                const int32_t maxY = int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_RED_MAXY]);
                Logger::info(str::format(
                  "[Coverage]   DebugViewRedBox x=[", minX, ",", maxX, "] y=[", minY, ",", maxY, "]",
                  " w=", (maxX - minX), " h=", (maxY - minY)));
              }
              if (dvNan > 0u) {
                const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_NAN_MINX]);
                const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_NAN_MINY]);
                const int32_t maxX = int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_NAN_MAXX]);
                const int32_t maxY = int32_t(cov[baseDv + COVERAGE_DVSCAN_SLOT_NAN_MAXY]);
                Logger::info(str::format(
                  "[Coverage]   DebugViewNanBox x=[", minX, ",", maxX, "] y=[", minY, ",", maxY, "]",
                  " w=", (maxX - minX), " h=", (maxY - minY)));
              }
            }
            // Per-VS attribution of the DISPLAYED red, via SharedSurfaceIndex
            // co-sampled at red pixels. The top line names the VS actually
            // drawing the on-screen red plane. If this is empty while redPixels>0,
            // the red pixels carry no valid primary surface index (miss/sky path
            // or the aliased surface-index buffer was stale by postprocess time).
            {
              const uint32_t baseRedSurf = COVERAGE_DVRED_SURFACE_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              std::map<uint64_t, uint64_t> vsRedDisplayed;
              uint64_t attributed = 0;
              for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
                const uint32_t px = cov[baseRedSurf + s];
                if (px == 0u) {
                  continue;
                }
                attributed += px;
                const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
                const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
                const uint64_t vs = (blas != nullptr)
                  ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
                vsRedDisplayed[vs] += px;
              }
              if (attributed > 0) {
                std::vector<std::pair<uint64_t, uint64_t>> srt(vsRedDisplayed.begin(), vsRedDisplayed.end());
                std::sort(srt.begin(), srt.end(),
                          [](const std::pair<uint64_t, uint64_t>& a,
                             const std::pair<uint64_t, uint64_t>& b) { return a.second > b.second; });
                for (const auto& kv : srt) {
                  char vsHex[24];
                  snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                           static_cast<unsigned long long>(kv.first));
                  Logger::info(str::format(
                    "[Coverage]   DebugViewRedSurface VS=", vsHex, " displayedRedPixels=", kv.second));
                }
              }
            }
          }
          } // end if(!coveragePickOnly) — skipped histogram/color/red/NaN/scan

          // NV-DXVK [Coverage PickRegion]: COLOR-INDEPENDENT attribution of a
          // configurable screen rectangle (default bottom-right). Unlike the red-
          // gated DVRED/PureRed probes above, this names whatever VS draws inside
          // the rect by pixel count — the tool for identifying an un-tinted object
          // like the TF2 first-person weapon. Sweep rtx.surfaceCoveragePickRegion
          // to tighten onto one object, then read PickRegionVS. The screen bbox
          // and valid/invalid split confirm the rect landed on real geometry (not
          // sky/miss, which carry no valid primary surfaceIndex → invalidPixels).
          {
            const uint32_t pbase = COVERAGE_PICKREGION_SUMMARY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t pkTotal = cov[pbase + COVERAGE_PICKREGION_SLOT_TOTAL];
            const uint32_t pkValid = cov[pbase + COVERAGE_PICKREGION_SLOT_VALID];
            const uint32_t pkInvalid = cov[pbase + COVERAGE_PICKREGION_SLOT_INVALID];
            // UNCONDITIONAL summary so the line appears even when scanned==0. A
            // scanned==0 line means the C++ readback IS built but the shader probe
            // wrote nothing (stale cached postprocess shader, or a degenerate
            // rtx.surfaceCoveragePickRegion). NO PickRegion line at all means the
            // C++ readback itself isn't in the running binary. This distinction is
            // why the gate was removed.
            {
              const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MINX]);
              const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MINY]);
              const int32_t maxX = int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MAXX]);
              const int32_t maxY = int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MAXY]);
              // Mean displayed colour under the pick (sum*256 / 256 / N).
              const uint32_t colN = cov[pbase + COVERAGE_PICKREGION_SLOT_COLN];
              const float colR = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLR]) / (256.0f * float(colN)) : 0.f;
              const float colG = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLG]) / (256.0f * float(colN)) : 0.f;
              const float colB = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLB]) / (256.0f * float(colN)) : 0.f;
              Logger::info(str::format(
                "[Coverage] PickRegion scanned=", pkTotal,
                " validPixels=", pkValid, " invalidPixels=", pkInvalid,
                " screenBox x=[", minX, ",", maxX, "] y=[", minY, ",", maxY, "]",
                " meanColor=(", colR, ",", colG, ",", colB, ") n=", colN));

              // Group the per-surfaceIndex rect counts by VS hash and map each VS to
              // its material + color-texture hash (so the weapon is recognizable by
              // its texture, not just an opaque hash). Sorted by pixel count desc:
              // the top line is the object filling the rect.
              const uint32_t baseSurf = COVERAGE_PICKREGION_SURFACE_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMaxX = COVERAGE_PICKREGION_BOX_MAXX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMinX = COVERAGE_PICKREGION_BOX_MINX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMaxY = COVERAGE_PICKREGION_BOX_MAXY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMinY = COVERAGE_PICKREGION_BOX_MINY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              // Per-VS aggregate: pixels, dominant texture/material, and the union of
              // its surfaces' screen boxes (so one VS spanning several surfaces gets a
              // single location box).
              struct VsBox { int32_t minX = 0x7fffffff, minY = 0x7fffffff, maxX = -1, maxY = -1; };
              std::map<uint64_t, uint64_t> vsPx;   // VS hash -> pixels
              std::map<uint64_t, uint64_t> vsTex;  // VS hash -> dominant colorTexture hash
              std::map<uint64_t, uint64_t> vsMat;  // VS hash -> dominant material hash
              std::map<uint64_t, VsBox> vsBox;     // VS hash -> screen bbox
              uint64_t attributed = 0;
              for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
                const uint32_t px = cov[baseSurf + s];
                if (px == 0u) {
                  continue;
                }
                attributed += px;
                const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
                const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
                const uint64_t vs = (blas != nullptr)
                  ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
                vsPx[vs] += px;
                // Merge this surface's screen bbox (written under the same valid-index
                // gate, so it exists whenever px>0) into the VS's union box.
                const int32_t sMaxX = int32_t(cov[baseBoxMaxX + s]);
                const int32_t sMinX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseBoxMinX + s]);
                const int32_t sMaxY = int32_t(cov[baseBoxMaxY + s]);
                const int32_t sMinY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseBoxMinY + s]);
                VsBox& bx = vsBox[vs];
                bx.minX = std::min(bx.minX, sMinX); bx.maxX = std::max(bx.maxX, sMaxX);
                bx.minY = std::min(bx.minY, sMinY); bx.maxY = std::max(bx.maxY, sMaxY);
                if (blas != nullptr) {
                  // Record this VS's color-texture / material hash so the weapon is
                  // recognizable by texture. Multiple surfaces can share a VS; the
                  // last one wins, which is fine for a single-material object.
                  const auto& md = blas->input.getMaterialData();
                  vsTex[vs] = uint64_t(md.getColorTexture().getImageHash());
                  vsMat[vs] = uint64_t(md.getHash());
                }
              }
              if (attributed > 0) {
                std::vector<std::pair<uint64_t, uint64_t>> srt(vsPx.begin(), vsPx.end());
                std::sort(srt.begin(), srt.end(),
                          [](const std::pair<uint64_t, uint64_t>& a,
                             const std::pair<uint64_t, uint64_t>& b) { return a.second > b.second; });
                for (const auto& kv : srt) {
                  char vsHex[24]; snprintf(vsHex, sizeof(vsHex), "0x%016llx", (unsigned long long)kv.first);
                  char texHex[24]; snprintf(texHex, sizeof(texHex), "0x%016llx", (unsigned long long)vsTex[kv.first]);
                  char matHex[24]; snprintf(matHex, sizeof(matHex), "0x%016llx", (unsigned long long)vsMat[kv.first]);
                  const VsBox& bx = vsBox[kv.first];
                  Logger::info(str::format(
                    "[Coverage]   PickRegionVS VS=", vsHex, " pixels=", kv.second,
                    " box x=[", bx.minX, ",", bx.maxX, "] y=[", bx.minY, ",", bx.maxY, "]",
                    " w=", (bx.maxX - bx.minX), " h=", (bx.maxY - bx.minY),
                    " colorTexture=", texHex, " material=", matHex));
                }
              } else {
                Logger::info(str::format(
                  "[Coverage]   PickRegionVS (none) — rect pixels carry no valid "
                  "primary surfaceIndex (sky/miss, or surfaceIndex >= SLOTS)"));
              }
            }
          }

          // NV-DXVK [Coverage PickRegion2]: identical readback for the second
          // independent rect (rtx.surfaceCoveragePickRegion2 → regions 65/66/67-70).
          {
            const uint32_t pbase = COVERAGE_PICKREGION2_SUMMARY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t pkTotal = cov[pbase + COVERAGE_PICKREGION_SLOT_TOTAL];
            const uint32_t pkValid = cov[pbase + COVERAGE_PICKREGION_SLOT_VALID];
            const uint32_t pkInvalid = cov[pbase + COVERAGE_PICKREGION_SLOT_INVALID];
            {
              const int32_t minX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MINX]);
              const int32_t minY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MINY]);
              const int32_t maxX = int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MAXX]);
              const int32_t maxY = int32_t(cov[pbase + COVERAGE_PICKREGION_SLOT_MAXY]);
              const uint32_t colN = cov[pbase + COVERAGE_PICKREGION_SLOT_COLN];
              const float colR = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLR]) / (256.0f * float(colN)) : 0.f;
              const float colG = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLG]) / (256.0f * float(colN)) : 0.f;
              const float colB = colN ? float(cov[pbase + COVERAGE_PICKREGION_SLOT_COLB]) / (256.0f * float(colN)) : 0.f;
              Logger::info(str::format(
                "[Coverage] PickRegion2 scanned=", pkTotal,
                " validPixels=", pkValid, " invalidPixels=", pkInvalid,
                " screenBox x=[", minX, ",", maxX, "] y=[", minY, ",", maxY, "]",
                " meanColor=(", colR, ",", colG, ",", colB, ") n=", colN));
              const uint32_t baseSurf = COVERAGE_PICKREGION2_SURFACE_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMaxX = COVERAGE_PICKREGION2_BOX_MAXX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMinX = COVERAGE_PICKREGION2_BOX_MINX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMaxY = COVERAGE_PICKREGION2_BOX_MAXY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              const uint32_t baseBoxMinY = COVERAGE_PICKREGION2_BOX_MINY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
              struct VsBox2 { int32_t minX = 0x7fffffff, minY = 0x7fffffff, maxX = -1, maxY = -1; };
              std::map<uint64_t, uint64_t> vsPx;
              std::map<uint64_t, uint64_t> vsTex;
              std::map<uint64_t, uint64_t> vsMat;
              std::map<uint64_t, VsBox2> vsBox;
              uint64_t attributed = 0;
              // NV-DXVK [PickDraw]: track the single surface with the most pixels
              // at center — its drawCallID is the EXACT sub-draw under the
              // crosshair, finer than the VS hash.
              uint32_t bestPx = 0; uint32_t bestDrawId = 0;
              for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
                const uint32_t px = cov[baseSurf + s];
                if (px == 0u) { continue; }
                attributed += px;
                const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
                const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
                const uint64_t vs = (blas != nullptr)
                  ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
                if (px > bestPx && blas != nullptr) { bestPx = px; bestDrawId = blas->input.drawCallID; }
                vsPx[vs] += px;
                const int32_t sMaxX = int32_t(cov[baseBoxMaxX + s]);
                const int32_t sMinX = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseBoxMinX + s]);
                const int32_t sMaxY = int32_t(cov[baseBoxMaxY + s]);
                const int32_t sMinY = int32_t(COVERAGE_REDSCR_BIAS) - int32_t(cov[baseBoxMinY + s]);
                VsBox2& bx = vsBox[vs];
                bx.minX = std::min(bx.minX, sMinX); bx.maxX = std::max(bx.maxX, sMaxX);
                bx.minY = std::min(bx.minY, sMinY); bx.maxY = std::max(bx.maxY, sMaxY);
                if (blas != nullptr) {
                  const auto& md = blas->input.getMaterialData();
                  vsTex[vs] = uint64_t(md.getColorTexture().getImageHash());
                  vsMat[vs] = uint64_t(md.getHash());
                }
              }
              if (attributed > 0) {
                std::vector<std::pair<uint64_t, uint64_t>> srt(vsPx.begin(), vsPx.end());
                std::sort(srt.begin(), srt.end(),
                          [](const std::pair<uint64_t, uint64_t>& a,
                             const std::pair<uint64_t, uint64_t>& b) { return a.second > b.second; });
                // NV-DXVK [SkinAABB]: publish the dominant (most-pixels) center VS
                // so d3d11_rtx's skin probe fires only on the hovered mesh.
                tf2::g_pickCenterVsHash.store(srt[0].first, std::memory_order_relaxed);
                // NV-DXVK [PickDraw]: publish the exact center sub-draw id too.
                tf2::g_pickCenterDrawId.store(bestDrawId, std::memory_order_relaxed);
                for (const auto& kv : srt) {
                  char vsHex[24]; snprintf(vsHex, sizeof(vsHex), "0x%016llx", (unsigned long long)kv.first);
                  char texHex[24]; snprintf(texHex, sizeof(texHex), "0x%016llx", (unsigned long long)vsTex[kv.first]);
                  char matHex[24]; snprintf(matHex, sizeof(matHex), "0x%016llx", (unsigned long long)vsMat[kv.first]);
                  const VsBox2& bx = vsBox[kv.first];
                  Logger::info(str::format(
                    "[Coverage]   PickRegion2VS VS=", vsHex, " pixels=", kv.second,
                    " box x=[", bx.minX, ",", bx.maxX, "] y=[", bx.minY, ",", bx.maxY, "]",
                    " w=", (bx.maxX - bx.minX), " h=", (bx.maxY - bx.minY),
                    " colorTexture=", texHex, " material=", matHex));
                }
              } else {
                // NV-DXVK [SkinAABB]: nothing under the crosshair → clear the
                // published VS + draw id so the probes go quiet instead of sticking.
                tf2::g_pickCenterVsHash.store(0, std::memory_order_relaxed);
                tf2::g_pickCenterDrawId.store(0, std::memory_order_relaxed);
                Logger::info(str::format(
                  "[Coverage]   PickRegion2VS (none) — rect pixels carry no valid "
                  "primary surfaceIndex (sky/miss, or surfaceIndex >= SLOTS)"));
              }
            }
          }

          if (!coveragePickOnly) {
          // NV-DXVK [Coverage PureRed]: VALID-STAGE attribution of the pure-red
          // plane (recorded in the opaque RAW_ALBEDO write with a guaranteed-valid
          // surfaceIndex). Groups by VS and maps to material + color-texture hash.
          // If PureRedSummary total is ~0 while DebugViewScan redPixels is large,
          // the red pixels are NOT opaque hits → look at the miss/sky path.
          {
            const uint32_t basePR = COVERAGE_PURERED_SURFACE_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t baseSum = COVERAGE_PURERED_SUMMARY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t prTotal = cov[baseSum + COVERAGE_PURERED_SLOT_TOTAL];
            const uint32_t prTexLoaded = cov[baseSum + COVERAGE_PURERED_SLOT_TEXLOADED];
            const uint32_t prTexMissing = cov[baseSum + COVERAGE_PURERED_SLOT_TEXMISSING];
            const uint32_t prStoreTotal = cov[baseSum + COVERAGE_PURERED_SLOT_STORETOTAL];
            const uint32_t prIdxMax = cov[baseSum + COVERAGE_PURERED_SLOT_STOREIDXMAX];
            const uint32_t prIdxMinEnc = cov[baseSum + COVERAGE_PURERED_SLOT_STOREIDXMIN];
            // Store-site probe prints even when the <SLOTS-gated total is 0.
            if (prStoreTotal > 0u) {
              Logger::info(str::format(
                "[Coverage] PureRedStoreSite count=", prStoreTotal,
                " surfaceIndexRange=[", (prIdxMinEnc ? (0xFFFFFFFFu - prIdxMinEnc) : 0u), ",", prIdxMax, "]",
                " (SLOTS=", uint32_t(COVERAGE_SURFACE_SLOTS), ")"));
            }
            if (prTotal > 0u) {
              Logger::info(str::format(
                "[Coverage] PureRedSummary total=", prTotal,
                " withColorTexture=", prTexLoaded, " withoutColorTexture=", prTexMissing));
              std::map<uint64_t, uint64_t> vsPR;     // VS -> px
              std::map<uint64_t, uint64_t> texPR;    // colorTexImageHash -> px
              std::map<uint64_t, uint64_t> matPR;    // material hash -> px
              for (uint32_t s = 0; s < uint32_t(COVERAGE_SURFACE_SLOTS); ++s) {
                const uint32_t px = cov[basePR + s];
                if (px == 0u) {
                  continue;
                }
                const RtInstance* inst = (s < reordered.size()) ? reordered[s] : nullptr;
                const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
                const uint64_t vs = (blas != nullptr)
                  ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
                vsPR[vs] += px;
                if (blas != nullptr) {
                  const auto& md = blas->input.getMaterialData();
                  texPR[uint64_t(md.getColorTexture().getImageHash())] += px;
                  matPR[uint64_t(md.getHash())] += px;
                }
              }
              std::vector<std::pair<uint64_t, uint64_t>> vsSorted(vsPR.begin(), vsPR.end());
              std::sort(vsSorted.begin(), vsSorted.end(),
                        [](const std::pair<uint64_t, uint64_t>& a,
                           const std::pair<uint64_t, uint64_t>& b) { return a.second > b.second; });
              for (const auto& kv : vsSorted) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   PureRedVS VS=", h, " pixels=", kv.second));
              }
              for (const auto& kv : texPR) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   PureRedColorTexture imageHash=", h, " pixels=", kv.second));
              }
              for (const auto& kv : matPR) {
                char h[24]; snprintf(h, sizeof(h), "0x%016llx", (unsigned long long)kv.first);
                Logger::info(str::format("[Coverage]   PureRedMaterial matHash=", h, " pixels=", kv.second));
              }
            }
          }

          // NV-DXVK [Coverage OOBWhy]: decomposition of an out-of-range
          // primary-ray surfaceIndex. surfaceIndex = base + geometryIndex.
          //   base >= surfaceCount  -> bad customIndex (TLAS instance slot
          //                            doesn't exist in m_reorderedSurfaces).
          //   base < surfaceCount but base+geometryIndex overflows -> the
          //                            geometryIndex offset is too large.
          {
            const uint32_t oobBase = COVERAGE_OOBWHY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
            const uint32_t oobCount = cov[oobBase + COVERAGE_OOBWHY_SLOT_COUNT];
            if (oobCount > 0u) {
              Logger::info(str::format(
                "[Coverage] OOBWhy count=", oobCount,
                " maxCustomIndex=", cov[oobBase + COVERAGE_OOBWHY_SLOT_CUSTOMINDEX],
                " maxBase=", cov[oobBase + COVERAGE_OOBWHY_SLOT_BASE],
                " maxGeometryIndex=", cov[oobBase + COVERAGE_OOBWHY_SLOT_GEOMETRYINDEX],
                " maxSurfaceIndex=", cov[oobBase + COVERAGE_OOBWHY_SLOT_SURFACEINDEX],
                " surfaceCount=", cov[oobBase + COVERAGE_OOBWHY_SLOT_SURFACECOUNT]));
            }
          }

          // Loose clear for the next accumulation window.
          // NV-DXVK [Coverage]: dump the per-call-site high-surfaceIndex
          // counts (region 4 slots 0..15). Site IDs:
          //   0 = resolve.slangh:406    (resolveVertex primary)
          //   1 = resolve.slangh:1035   (resolveVertex secondary)
          //   2 = visibility.slangh:106 (visibility ray hit — usePrevTLAS path)
          //   3 = nee_cache_light:200   (NEE cache light eval)
          //   4 = nee_cache_light:327   (NEE cache light intensity)
          //   7 = rayInteractionCreate  (UPSTREAM ray-decode probe — fires
          //                              when customIndex itself encodes
          //                              an out-of-range surfaceIndex,
          //                              i.e. BEFORE any masking/clamping
          //                              the material-interaction layer
          //                              might do. Slots 5/6 reserved.)
          // The site that has the largest count is the one feeding stale
          // prev-frame surfaceIndex values into the InterlockedAdd — i.e.
          // the root code path producing the corruption. Site 7 firing
          // with sites 0-4 silent means the OOB happens at ray-hit time
          // but is somehow lost / masked by the time material-interaction
          // gates run — that's an entirely different bug shape than the
          // "prev-frame stale read" hypothesis sites 0-4 were measuring.
          const uint32_t* siteCounts = cov + 4u * uint32_t(COVERAGE_SURFACE_SLOTS);
          // NV-DXVK [Coverage]: slot 8 = max(cb.surfaceCount) observed in
          // any primary-ray rayInteractionCreate this frame. Slot 9 =
          // sample count. Print unconditionally (not gated on siteTotal)
          // because the whole point of the probe is to compare against
          // [SpawnGeomDiag.CbSurfaceCount]. If shaderMax != cbUploaded,
          // the GPU is reading a different (stale) constant buffer value
          // than what writeToBuffer was told to upload — the OOB checks
          // at material-interaction time silently pass for indices that
          // ARE out-of-range from the CPU's m_reorderedSurfaces.size().
          // NV-DXVK [Coverage]: cross-check the CPU upload vs the
          // shader-observed value FOR THE SAME GPU FRAME. gpuFrame
          // was read at the top of this dump from Region 4 slot 10.
          // Cross-correlate via gpuFrame against the matching
          // [SpawnGeomDiag.CbSurfaceCount] log line (which also
          // records frameIdx). If they match, the cb pipeline is fine.
          const uint32_t shaderMax = siteCounts[8];
          const uint32_t shaderSamples = siteCounts[9];
          Logger::info(str::format(
            "[Coverage] CbSurfaceCountInShader gpuFrame=", gpuFrame,
            " shaderObservedMax=", shaderMax,
            " samples=", shaderSamples));
          uint64_t siteTotal = 0u;
          for (uint32_t i = 0; i < 16u; ++i) {
            // Slots 8/9/10 are the cb.surfaceCount/samples/frameIdx
            // probes, not per-site OOB counters — exclude.
            if (i == 8u || i == 9u || i == 10u) continue;
            siteTotal += siteCounts[i];
          }
          if (siteTotal > 0u) {
            Logger::info(str::format(
              "[Coverage] === HighSurfaceIndexBySite === totalHits=", siteTotal,
              " (threshold=cb.surfaceCount, sites: 0=resolvePrimary,"
              " 1=resolveSecondary, 2=visibility, 3=neeCacheLight,"
              " 4=neeCacheIntensity, 7=rayInteractionCreate)"));
            const char* siteNames[16] = {
              "resolvePrimary", "resolveSecondary", "visibility",
              "neeCacheLight", "neeCacheIntensity",
              "(reserved)", "(reserved)", "rayInteractionCreate",
              "cbSurfaceCountMax", "cbSurfaceCountSamples",
              "(reserved)", "(reserved)", "(reserved)", "(reserved)",
              "(reserved)", "(reserved)"
            };
            for (uint32_t i = 0; i < 16u; ++i) {
              if (i == 8u || i == 9u) continue;
              if (siteCounts[i] > 0u) {
                Logger::info(str::format(
                  "[Coverage]   HighSurfaceIndex site=", i, " name=", siteNames[i],
                  " hits=", siteCounts[i]));
              }
            }
          }

          } // end if(!coveragePickOnly) — skipped PureRed/OOBWhy/HighSurfaceIndex

          // Clear ALL regions (0..COVERAGE_TOTAL_REGIONS-1) so the new
          // raw-albedo colour-sum regions (17/18/19) — and region 16 — don't
          // accumulate across frames and skew the per-VS average. ALWAYS runs
          // (outside the pick-only guard) so the buffer never accumulates.
          memset(cov, 0, size_t(COVERAGE_TOTAL_REGIONS) * COVERAGE_SURFACE_SLOTS * sizeof(uint32_t));
        }
      }
    }

    // NV-DXVK: scene dump readback. Triggered N frames earlier via F11 (see
    // updateRaytraceArgsConstantBuffer). vkDeviceWaitIdle is overkill but
    // cheap insurance for a one-shot dev op — guarantees the captured frame
    // has fully retired before we map the host-visible buffer. CSV is
    // written next to the .exe; one row per pixel where flags bit 0 is set
    // (i.e. an opaque hit was recorded).
    if (g_sceneDumpState == SceneDumpState::AwaitingReadback
        && frameIdx >= g_sceneDumpReadbackAtFrame) {
      auto& dumpBuffer = rtOutput.m_sceneDumpBuffer;
      if (dumpBuffer.ptr()) {
        m_device->waitForIdle();
        const uint32_t W = g_sceneDumpExtent.width;
        const uint32_t H = g_sceneDumpExtent.height;
        const SceneDumpElement* data = reinterpret_cast<const SceneDumpElement*>(dumpBuffer->mapPtr(0));
        if (data) {
          auto t = std::time(nullptr);
          auto tm = *std::localtime(&t);
          char stamp[64];
          std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
          const std::string filename = str::format("scene_dump_", stamp, "_f", g_sceneDumpTriggerFrame, "_", W, "x", H, ".csv");
          std::ofstream out(filename, std::ios::binary);
          if (out.is_open()) {
            // NV-DXVK: lightmapU/V columns plus a hasLightmap (flag bit 1)
            // column let the user verify the TEXCOORD1 plumbing end-to-end.
            // hasLightmap=1 means surface_interaction interpolated a real
            // lightmap UV from the second 8 bytes the interleaver wrote;
            // hasLightmap=0 means the fallback (mirror albedo UV) ran.
            // NV-DXVK [VanishDiag-Shader]: extra columns minClipW, maxClipW,
            // hitWorldX/Y/Z let the user filter pathCode=1 rows by world
            // position and confirm whether the failing pixels really are
            // the floor surface (vs sky / walls / props).
            out << "pixelX,pixelY,u,v,mip,aniso,texIdx,texW,texH,pathCode,flags,lightmapU,lightmapV,hasLightmap,minClipW,maxClipW,hitWorldX,hitWorldY,hitWorldZ\n";
            uint64_t hitPixels = 0;
            uint64_t lightmapPixels = 0;
            for (uint32_t y = 0; y < H; ++y) {
              for (uint32_t x = 0; x < W; ++x) {
                const SceneDumpElement& e = data[y * W + x];
                if ((e.flags & 1u) == 0u) continue;
                const uint32_t texW = (e.texDimsPacked >> 16) & 0xFFFFu;
                const uint32_t texH = e.texDimsPacked & 0xFFFFu;
                const uint32_t lmFlag = (e.flags & 2u) ? 1u : 0u;
                out << x << ',' << y << ','
                    << e.u << ',' << e.v << ','
                    << e.mip << ',' << e.aniso << ','
                    << e.texIdx << ',' << texW << ',' << texH << ','
                    << e.pathCode << ',' << e.flags << ','
                    << e.lightmapU << ',' << e.lightmapV << ',' << lmFlag << ','
                    << e.minClipW << ',' << e.maxClipW << ','
                    << e.hitWorldX << ',' << e.hitWorldY << ',' << e.hitWorldZ << '\n';
                ++hitPixels;
                if (lmFlag) ++lightmapPixels;
              }
            }
            out.close();
            Logger::info(str::format(
              "[SceneDump] wrote ", filename, " (",
              hitPixels, " hit pixels of ", uint64_t(W) * uint64_t(H),
              "; ", lightmapPixels, " with hasLightmap=1)"));

            // NV-DXVK: sidecar texmap. The auto-dump path
            // (scene_manager.cpp:autoDumpMaterialTextures) names every
            // unique texture on disk by its 64-bit imageHash, but the
            // shader writes texIdx (bindless slot) into the dump. Walk
            // the bindless texture table and emit
            // scene_dump_*_texmap.csv mapping texIdx -> imageHash so the
            // user can find the .dds for any pixel's bound albedo at
            // rtx-remix/captures/textures/<hash>_albedo.dds.
            const std::string texmapName = str::format("scene_dump_", stamp, "_f", g_sceneDumpTriggerFrame, "_", W, "x", H, "_texmap.csv");
            std::ofstream texmapOut(texmapName, std::ios::binary);
            if (texmapOut.is_open()) {
              texmapOut << "texIdx,imageHashHex,W,H,mipCount\n";
              const auto& table = getCommonObjects()->getTextureManager().getTextureTable();
              for (uint32_t i = 0; i < table.size(); ++i) {
                const TextureRef& ref = table[i];
                if (!ref.isValid() || ref.isImageEmpty()) continue;
                const DxvkImageView* view = ref.getImageView();
                if (!view || !view->image().ptr()) continue;
                const auto& info = view->image()->info();
                char hashStr[32];
                std::snprintf(hashStr, sizeof(hashStr), "%016llx", (unsigned long long)ref.getImageHash());
                texmapOut << i << ',' << hashStr << ','
                          << info.extent.width << ',' << info.extent.height << ','
                          << info.mipLevels << '\n';
              }
              texmapOut.close();
              Logger::info(str::format("[SceneDump] wrote ", texmapName));
            } else {
              Logger::err(str::format("[SceneDump] failed to open ", texmapName));
            }

            // NV-DXVK [VanishDiag-WorldVis pair note]: the F11 frame ID is
            // embedded in this dump's filename (g_sceneDumpTriggerFrame).
            // The bucket-bitmask snapshot for that exact frame lives in
            // the [VanishDiag-WorldVis] log line in remix-dxvk.log emitted
            // by d3d11_rtx.cpp's EndFrame block (sidecar logic moved there
            // because g_vanishDiagCapturedA2 lives in d3d11.dll's address
            // space, not dxvk.dll's). Grep the log for
            // "[VanishDiag-WorldVis] frame=N" where N matches the f<frame>
            // segment of this dump filename to pair them.
          } else {
            Logger::err(str::format("[SceneDump] failed to open ", filename, " for writing"));
          }
        } else {
          Logger::err("[SceneDump] mapPtr returned null on host-visible dump buffer");
        }
      }
      g_sceneDumpState = SceneDumpState::Idle;
    }

    if (!debugView.isActive()) {
      return;
    }

    debugView.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      srcImage, rtOutput, *m_common);

    if (captureScreenImage) {
      // For overlayed debug views, we preserve the post tonemapping naming since the post tonemapped image is a base image.
      // The benefit is retention of most of the existing testing pipeline.
      if (debugView.getOverlayOnTopOfRenderOutput()) {
        takeScreenshot("rtxImagePostTonemapping", srcImage);
      } else {
        takeScreenshot("rtxImageDebugView", srcImage);
      }
    }
  }

  void RtxContext::dispatchReplaceCompositeWithDebugView(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();

    DebugView& debugView = m_common->metaDebugView();

    if (!debugView.isActive()) {
      return;
    }

    debugView.dispatchAfterCompositionPass(this,
      getResourceManager().getSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      rtOutput, *m_common);
  }

  namespace
  {
    template<typename T>
    T mapAs(const Rc<DxvkBuffer>& buf) {
      if (buf == nullptr) {
        return nullptr;
      }
      return static_cast<T>(buf->mapPtr(0));
    }

    Vector2i rescale(const float(&scale)[2], const Vector2i& pix) {
      return Vector2i {
        static_cast<int>(static_cast<float>(pix.x) * scale[0]),
        static_cast<int>(static_cast<float>(pix.y) * scale[1]),
      };
    }

    struct PixRegion {
      Vector2i from {};
      Vector2i to {};
    };

    PixRegion rescale(const float(&scale)[2], const PixRegion& original) {
      Vector2i from = rescale(scale, original.from);
      Vector2i to = rescale(scale, original.to);
      // if was at least 1 pixel, then rescaled should also contain at least 1 pixel
      if (original.to.x - original.from.x > 0) {
        to.x = std::max(to.x, from.x + 1);
      }
      if (original.to.y - original.from.y > 0) {
        to.y = std::max(to.y, from.y + 1);
      }
      return PixRegion { from, to };
    }

    PixRegion clamp(const VkExtent3D& extent, const PixRegion& original) {
      return PixRegion {
        Vector2i{
          std::clamp<int>(original.from.x, 0, std::min<uint32_t>(INT32_MAX, extent.width)),
          std::clamp<int>(original.from.y, 0, std::min<uint32_t>(INT32_MAX, extent.height)),
        },
        Vector2i{
          std::clamp<int>(original.to.x, 0, std::min<uint32_t>(INT32_MAX, extent.width)),
          std::clamp<int>(original.to.y, 0, std::min<uint32_t>(INT32_MAX, extent.height)),
        },
      };
    }

    std::optional<PixRegion> nonzero(const PixRegion& original) {
      if (original.to.x - original.from.x > 0 &&
          original.to.y - original.from.y > 0) {
        return original;
      }
      return std::nullopt;
    }

    VkOffset3D vkoffset(const PixRegion& r) {
      return VkOffset3D { r.from.x, r.from.y, 0 };
    }

    VkExtent3D vkextent(const PixRegion& request) {
      assert(request.to.x - request.from.x > 0 && request.to.y - request.from.y > 0);
      return VkExtent3D {
        static_cast<uint32_t>(std::max(0, request.to.x - request.from.x)),
        static_cast<uint32_t>(std::max(0, request.to.y - request.from.y)),
        1 };
    }

    // In C++20
    template<typename Func >
    void erase_if(std::vector<std::future<void>>& vec, Func&& predicate) {
      auto newEnd = std::remove_if(vec.begin(), vec.end(), predicate);
      vec.erase(newEnd, vec.end());
    }
  }

  void RtxContext::dispatchObjectPicking(Resources::RaytracingOutput& rtOutput,
                                         const VkExtent3D& srcExtent,
                                         const VkExtent3D& targetExtent) {
    ScopedCpuProfileZone();
    DebugView& debugView = m_common->metaDebugView();
    SceneManager& sceneManager = m_common->getSceneManager();
    const uint32_t frameIdx = m_device->getCurrentFrameId();


    auto enoughTimeHasPassedToDestroy = [&]() {
      if (g_forceKeepObjectPickingImage) {
        return false;
      }
      // there are object picking / highlighting requests, so don't destroy
      if (debugView.ObjectPicking.containsRequests() || 
          debugView.Highlighting.active(frameIdx) || 
          !m_objectPickingReadback.asyncTasks.empty()) {
        return false;
      }
      return true;
    };

    if (rtOutput.m_primaryObjectPicking.isValid()) {
      if (enoughTimeHasPassedToDestroy()) {
        rtOutput.m_primaryObjectPicking = {};
        Logger::debug("Object picking image was destroyed");
        return;
      }
    } else {
      // if object picking image exist
      // and it should be alive
      if (!enoughTimeHasPassedToDestroy()) {
        // create and schedule picking to the next frame
        auto thisRef = Rc<DxvkContext> { this };
        rtOutput.m_primaryObjectPicking =
          Resources::createImageResource(thisRef, "primary object picking", srcExtent, VK_FORMAT_R32_UINT);
        Logger::debug("Object picking image was created");
      }
      return;
    }

    erase_if(m_objectPickingReadback.asyncTasks, [](std::future<void>& f) {
      if (!f.valid()) {
        return true;
      }
      // check status with minimal wait; safe to delete, if it has completed
      return f.wait_for(std::chrono::duration<int>{0}) == std::future_status::ready;
    });


    const Resources::Resource& objectPickingSrc = rtOutput.m_primaryObjectPicking;
    assert(srcExtent == objectPickingSrc.image->info().extent);
    const float downscale[] = {
      srcExtent.width / static_cast<float>(targetExtent.width),
      srcExtent.height / static_cast<float>(targetExtent.height)
    };
    constexpr static VkDeviceSize onePixelInBytes = sizeof(ObjectPickingValue);


    // process one request per frame, to readback in the future
    if (auto request = debugView.ObjectPicking.popRequest()) {
      if (auto pixRegion = nonzero(clamp(srcExtent, rescale(downscale,
                                                            PixRegion { request->pixelFrom, request->pixelTo })))) {
        assert(objectPickingSrc.isValid());
        assert(objectPickingSrc.image->formatInfo()->elementSize == onePixelInBytes);
        assert(getSceneManager().getGlobals().clearColorPicking <= (1ull << (8 * onePixelInBytes)) - 1);

        const VkExtent3D copyExtent = vkextent(*pixRegion);

        auto info = DxvkBufferCreateInfo {};
        {
          info.size = onePixelInBytes * copyExtent.width * copyExtent.height;
          info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_HOST_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT |
            VK_ACCESS_HOST_READ_BIT;
        }

        const VkMemoryPropertyFlags memType =
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        Rc<DxvkBuffer> readbackDst = m_device->createBuffer(info, memType, DxvkMemoryStats::Category::RTXBuffer, "Picking Readback Buffer");

        auto subres = VkImageSubresourceLayers {};
        {
          subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          subres.mipLevel = 0;
          subres.baseArrayLayer = 0;
          subres.layerCount = 1;
        }
        copyImageToBuffer(
          readbackDst, 0, onePixelInBytes, onePixelInBytes,
          objectPickingSrc.image, subres, vkoffset(*pixRegion), copyExtent);


        this->emitMemoryBarrier(0,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_HOST_READ_BIT);

        const uint64_t syncValue = ++m_objectPickingReadback.signalValue;
        this->signal(m_objectPickingReadback.signal, syncValue);

        m_objectPickingReadback.asyncTasks.push_back(std::async(
          std::launch::async,
          [this,
          cReadbackDst = std::move(readbackDst),
          cSyncValueToWait = syncValue,
          cCallback = std::move(request->callback)]() {
            // async wait
            this->m_objectPickingReadback.signal->wait(cSyncValueToWait);

            const uint32_t* readback = mapAs<const uint32_t*>(cReadbackDst);
            if (!readback || cReadbackDst->info().size < onePixelInBytes) {
              assert(0);
              cCallback(std::vector<ObjectPickingValue>{}, std::nullopt);
              return;
            }

            auto values = std::vector<ObjectPickingValue> {};
            auto primaryValue = ObjectPickingValue { 0 };
            {
              size_t count = cReadbackDst->info().size / onePixelInBytes;
              values.resize(count);

              memcpy(values.data(), readback, count * onePixelInBytes);
              primaryValue = values[0];

              // sort
              std::sort(values.begin(), values.end());
              // remove consecutive duplicates
              auto endNew = std::unique(values.begin(), values.end());
              values.erase(endNew, values.end());
            }

            auto legacyHashForPrimaryValue = g_allowMappingLegacyHashToObjectPickingValue ?
              m_common->getSceneManager().findLegacyTextureHashByObjectPickingValue(primaryValue) :
              std::optional<XXH64_hash_t>{};

            cCallback(std::move(values), legacyHashForPrimaryValue);
          }
        ));
      } else {
        request->callback(std::vector<ObjectPickingValue>{}, std::nullopt);
      }
    }

    if (auto pixelAndColor = debugView.Highlighting.accessPixelToHighlight(frameIdx)) {
      pixelAndColor->first = rescale(downscale, pixelAndColor->first);
      m_common->metaPostFx().dispatchHighlighting(this,
        rtOutput,
        {},
        pixelAndColor->first,
        pixelAndColor->second);
      return;
    }

    auto [objectPickingValues, color] = debugView.Highlighting.accessObjectPickingValueToHighlight(sceneManager,frameIdx);
    m_common->metaPostFx().dispatchHighlighting(
      this,
      rtOutput,
      std::move(objectPickingValues),
      {},
      color);
  }

  void RtxContext::dispatchDLFG() {
    if (!isDLFGEnabled()) {
      return;
    }

    // force vsync off if DLFG is enabled, as we don't properly support FG + vsync
    if (RtxOptions::enableVsyncState != EnableVsync::Off) {
      RtxOptions::enableVsync.setDeferred(EnableVsync::Off);
      RtxOptions::enableVsyncState = EnableVsync::Off;
    }

    Resources::RaytracingOutput& rtOutput = getResourceManager().getRaytracingOutput();

    DxvkFrameInterpolationInfo dlfgInfo = {
      m_device->getCurrentFrameId(),
      m_device->getCommon()->getSceneManager().getCamera(),
      rtOutput.m_primaryScreenSpaceMotionVector.view,
      rtOutput.m_primaryScreenSpaceMotionVector.image->info().layout,
      rtOutput.m_primaryDepth.view,
      rtOutput.m_primaryDepth.image->info().layout,
      false,
      m_common->metaDLFG().getInterpolatedFrameCount(),
    };
    m_device->setupFrameInterpolation(dlfgInfo);
  }

  void RtxContext::flushCommandList() {
    ScopedCpuProfileZone();

    // flush the residue
    tryHandleSky(nullptr, nullptr);

    m_device->submitCommandList(
      this->endRecording(),
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
      m_submitContainsInjectRtx,
      m_cachedReflexFrameId);
    
    // Reset this now that we've completed the submission
    m_submitContainsInjectRtx = false;
    
    this->beginRecording(
      m_device->createCommandList());

    getCommonObjects()->metaGeometryUtils().flushCommandList();
  }

  void RtxContext::updateComputeShaderResources() {
    ScopedCpuProfileZone();
    DxvkContext::updateComputeShaderResources();

    auto&& layout = m_state.cp.pipeline->layout();
    if (layout->requiresExtraDescriptorSet()) {
      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Textures),
                                  BINDING_SET_BINDLESS_TEXTURE2D);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Buffers),
                                  BINDING_SET_BINDLESS_RAW_BUFFER);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(),
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Samplers),
                                  BINDING_SET_BINDLESS_SAMPLER);
    }
  }

  void RtxContext::updateRaytracingShaderResources() {
    ScopedCpuProfileZone();
    DxvkContext::updateRaytracingShaderResources();

    auto&& layout = m_state.rp.pipeline->layout();
    if (layout->requiresExtraDescriptorSet()) {
      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Textures),
                                  BINDING_SET_BINDLESS_TEXTURE2D);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Buffers),
                                  BINDING_SET_BINDLESS_RAW_BUFFER);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Samplers),
                                  BINDING_SET_BINDLESS_SAMPLER);
    }
  }

  bool RtxContext::shouldUseDLSS() const {
    // Note: m_dlssSupported only checks for the presence of some basic extensions, the actual DLSS context needs to be queried to see
    // if a given platform supports DLSS (as this will depend on if it was actually initialized successfully or not). Cases where m_dlssSupported
    // is true but supportsDLSS() is not are for example when the DLSS DLL is missing.
    return RtxOptions::isDLSSEnabled() && m_dlssSupported && m_common->metaDLSS().supportsDLSS();
  }

  bool RtxContext::shouldUseRayReconstruction() const {
    return useRayReconstruction();
  }

  bool RtxContext::shouldUseNIS() const {
    return RtxOptions::isNISEnabled();
  }

  bool RtxContext::shouldUseTAA() const {
    return RtxOptions::isTAAEnabled();
  }

  bool RtxContext::shouldUseXeSS() const {
    return RtxOptions::upscalerType() == UpscalerType::XeSS;
  }

  RtxVertexCaptureData& RtxContext::allocAndMapVertexCaptureConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.vertexCaptureCB->allocSlice();
    invalidateBuffer(m_rtState.vertexCaptureCB, slice);

    return *static_cast<RtxVertexCaptureData*>(slice.mapPtr);
  }

  RtxVSConstants& RtxContext::allocAndMapVSConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.vsConstantsCB->allocSlice();
    invalidateBuffer(m_rtState.vsConstantsCB, slice);

    return *static_cast<RtxVSConstants*>(slice.mapPtr);
  }
  RtxSharedPS& RtxContext::allocAndMapPSSharedStateConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.psSharedStateCB->allocSlice();
    invalidateBuffer(m_rtState.psSharedStateCB, slice);

    return *static_cast<RtxSharedPS*>(slice.mapPtr);
  }

  void RtxContext::rasterizeToSkyMatte(const DrawParameters& params, const DrawCallState& drawCallState) {
    ScopedGpuProfileZone(this, "rasterizeToSkyMatte");

    // [SkyTrace.matteRaster] Per-draw entry log. Logs up to 32 sky draws
    // per frame so we can see how many distinct sky-quad meshes TF2
    // submits — needed to test whether the cube's 3-blank-faces issue is
    // architectural (TF2 only emits ~3 sky quads, all roughly screen-
    // aligned, can't fill 6 cube faces) or just a "we're missing draws"
    // bug. Also dumps first 3 vertex positions per draw so we can see
    // WHERE in object space those sky vertices sit. If TF2 emits 6 quads
    // at 6 different world-axis positions, the cube SHOULD populate.
    {
      const uint32_t frameId = m_device->getCurrentFrameId();
      static std::atomic<uint32_t> sFrame{ UINT32_MAX };
      static std::atomic<uint32_t> sCount{ 0 };
      const uint32_t prevFrame = sFrame.load(std::memory_order_relaxed);
      if (prevFrame != frameId) {
        sFrame.store(frameId, std::memory_order_relaxed);
        sCount.store(0, std::memory_order_relaxed);
      }
      const uint32_t drawIdx = sCount.fetch_add(1, std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && drawIdx < 32u) {
        // Diagnose why positionBuffer is unreadable. Print buffer state +
        // format + stride + counts on every draw, then position values
        // when format permits.
        const RasterGeometry& g = drawCallState.getGeometryData();
        const bool bufDefined = g.positionBuffer.defined();
        const VkFormat vfmt = bufDefined ? g.positionBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
        const uint32_t stride = bufDefined ? g.positionBuffer.stride() : 0u;
        const void* mapPtr = bufDefined ? g.positionBuffer.mapPtr() : nullptr;
        std::stringstream ss;
        ss << " bufDefined=" << (bufDefined ? "1" : "0")
           << " vFmt=" << static_cast<uint32_t>(vfmt)
           << " stride=" << stride
           << " mapPtr=" << (mapPtr != nullptr ? "non-null" : "NULL")
           << " vCount=" << g.vertexCount
           << " iCount=" << g.indexCount;
        if (bufDefined && mapPtr != nullptr && stride > 0u && g.vertexCount > 0u
            && (vfmt == VK_FORMAT_R32G32B32_SFLOAT
             || vfmt == VK_FORMAT_R32G32B32A32_SFLOAT)) {
          const uint8_t* base = static_cast<const uint8_t*>(mapPtr);
          const uint32_t n = std::min<uint32_t>(3u, g.vertexCount);
          ss << " pos=[";
          for (uint32_t v = 0; v < n; ++v) {
            const float* p = reinterpret_cast<const float*>(base + stride * v);
            ss << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
            if (v + 1 < n) ss << ",";
          }
          ss << "]";
        }
        const std::string posStr = ss.str();
        Logger::info(str::format(
          "[SkyTrace.matteRaster] frame=", frameId,
          " drawIdx=", drawIdx,
          " matHash=0x", std::hex, drawCallState.getMaterialData().getHash(), std::dec,
          " vsHash=0x", std::hex, drawCallState.transformData.vertexShaderHash, std::dec,
          " usesVS=", (drawCallState.usesVertexShader ? "1" : "0"),
          posStr));
        // [SkyTrace.skyVerts] Fire the GPU-side position readback for the
        // first 8 sky draws per frame. mapPtr is NULL on TF2's R32G32_UINT
        // sky position buffers (GPU-only after upload), so we copy the
        // first ~1024 verts to host and decode async. AABB tells us
        // whether the sky meshes span all 6 cube directions or only some.
        if (drawIdx < 8u) {
          recordSkyDrawPositionsReadback(drawCallState, frameId, drawIdx);
        }
      }
    }

    const RtCamera& camera = getSceneManager().getCamera();
    const uint32_t* renderResolution = camera.m_renderResolution;

    union UnifiedCB {
      RtxVertexCaptureData programmablePipeline;
      RtxVSConstants vsConstants;

      UnifiedCB() { }
    };

    UnifiedCB prevCB;

    // [SkyMatte.diag.cb] Gameplay-gated probe of the constant-buffer
    // pointers right before the mapPtr(0) deref. The CB save/modify/
    // restore around the sky raster is purely a DLSS-jitter pattern on
    // a buffer that the upstream frontend was supposed to plumb in via
    // setConstantBuffers(). The d3d11 frontend never wires that, so
    // these Rc<>s stay null in this fork. We track whether we have a
    // valid CB and skip just the jitter pattern when we don't — the
    // sky still rasterizes, it just won't carry per-frame DLSS jitter
    // (acceptable: sky is far-field, jitter contribution is negligible).
    const bool haveJitterCB = drawCallState.usesVertexShader
      ? (m_rtState.vertexCaptureCB != nullptr)
      : (m_rtState.vsConstantsCB != nullptr);
    {
      static std::atomic<uint32_t> sLastLoggedFrame{ UINT32_MAX };
      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t lastLogged = sLastLoggedFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && lastLogged != frameId) {
        sLastLoggedFrame.store(frameId, std::memory_order_relaxed);
        const Rc<DxvkBuffer>& vcCB = m_rtState.vertexCaptureCB;
        const Rc<DxvkBuffer>& vsCB = m_rtState.vsConstantsCB;
        Logger::info(str::format(
          "[SkyMatte.diag.cb] frame=", frameId,
          " usesVertexShader=", (drawCallState.usesVertexShader ? "true" : "false"),
          " vertexCaptureCB=0x", std::hex,
          reinterpret_cast<uint64_t>(vcCB.ptr()), std::dec,
          " vsConstantsCB=0x", std::hex,
          reinterpret_cast<uint64_t>(vsCB.ptr()), std::dec,
          " haveJitterCB=", (haveJitterCB ? "true" : "false")));
      }
    }

    if (haveJitterCB) {
      if (drawCallState.usesVertexShader) {
        prevCB.programmablePipeline = *static_cast<RtxVertexCaptureData*>(m_rtState.vertexCaptureCB->mapPtr(0));
      } else {
        prevCB.vsConstants = *static_cast<RtxVSConstants*>(m_rtState.vsConstantsCB->mapPtr(0));
      }
    }

    auto skyMatte = getResourceManager().getSkyMatte(this, m_skyColorFormat);
    auto skyMatteView = skyMatte.view;
    // [SkyMatte.diag] Gameplay-gated and frame-coalesced: log exactly
    // once per gameplay frame, where "gameplay" means camera valid +
    // TLAS not near-empty (the project-standard predicate — menu and
    // loading frames have an invalid camera or near-empty instance
    // table even when sky draws happen). Multiple sky draws per frame
    // produce a single log line. Falls through to the original deref —
    // if skyMatteView is null the next line still crashes, but the
    // log line immediately before pinpoints which pointer was null.
    // We intentionally do NOT bypass the deref: silently skipping the
    // raster would mask the bug as broken rendering rather than a
    // clean crash.
    {
      static std::atomic<uint32_t> sLastLoggedFrame{ UINT32_MAX };
      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t lastLogged = sLastLoggedFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      const bool isGameplay = cameraValid && tlasReady;
      if (isGameplay && lastLogged != frameId) {
        sLastLoggedFrame.store(frameId, std::memory_order_relaxed);
        const Rc<DxvkImage> img = (skyMatteView != nullptr)
          ? skyMatteView->image() : Rc<DxvkImage>{};
        const VkImage imgHandle = (img != nullptr) ? img->handle() : VK_NULL_HANDLE;
        const VkFormat actualFormat = (img != nullptr)
          ? img->info().format : VK_FORMAT_UNDEFINED;
        const VkExtent3D actualExtent = (img != nullptr)
          ? img->info().extent : VkExtent3D{ 0, 0, 0 };
        Logger::info(str::format(
          "[SkyMatte.diag] frame=", frameId,
          " viewPtr=0x", std::hex,
          reinterpret_cast<uint64_t>(skyMatteView.ptr()), std::dec,
          " imagePtr=0x", std::hex,
          reinterpret_cast<uint64_t>(img.ptr()), std::dec,
          " vkImage=0x", std::hex,
          reinterpret_cast<uint64_t>(imgHandle), std::dec,
          " requestedFormat=", static_cast<uint32_t>(m_skyColorFormat),
          " actualFormat=", static_cast<uint32_t>(actualFormat),
          " extent=", actualExtent.width, "x", actualExtent.height,
          " skyMatteIsValid=", (skyMatte.isValid() ? "true" : "false")));
      }
    }
    const auto skyMatteExt = skyMatteView->mipLevelExtent(0);

    // Update spec constants
    int prevClipSpaceJitterEnabled = -1;
    {
      // Only enable clipSpaceJitter if we have a valid vertexCaptureCB
      // to write the jitter values into. Without it the shader still
      // tries to read jitterX/jitterY from a uniform-buffer slot that
      // holds the game's CB (which doesn't have those fields at the
      // expected offsets), getting garbage values that translate the
      // sky verts off-screen. The visible result is a sky-matte that
      // never gets written, falling through to whatever the underlying
      // sky path produces — the bright-yellow ghost we just observed.
      if (drawCallState.usesVertexShader && haveJitterCB) {
        prevClipSpaceJitterEnabled = getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS)
          .specConstants[RtxSpecConstantId::ClipSpaceJitterEnabled]
            ? 1
            : 0;
        // Enable, to use clipSpaceJitter, see notes below
        setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, RtxSpecConstantId::ClipSpaceJitterEnabled, true);
      }
    }

    {
      VkViewport viewport { 0.5f, static_cast<float>(skyMatteExt.height) + 0.5f,
       static_cast<float>(skyMatteExt.width),
       -static_cast<float>(skyMatteExt.height),
       drawCallState.minZ,
       drawCallState.maxZ
      };
      VkRect2D scissor {
        { 0, 0 },
        { skyMatteExt.width, skyMatteExt.height }
      };
      setViewports(1, &viewport, &scissor);
    }

    if (haveJitterCB) {
      if (drawCallState.usesVertexShader) {
        RtxVertexCaptureData modified = prevCB.programmablePipeline;
        {
          // Jittered clip space for DLSS
          // Note: we can't jitter the projection matrix, as a game might calculate
          // its gl_Position by different methods (e.g. without projection matrix at all);
          // so apply jitter directly on gl_Position
          float ratioX = Sign(drawCallState.getTransformData().viewToProjection[2][3]);
          float ratioY = -Sign(drawCallState.getTransformData().viewToProjection[2][3]);
          Vector2 clipSpaceJitter = camera.calcClipSpaceJitter(camera.calcPixelJitter(m_device->getCurrentFrameId()), ratioX, ratioY);
          modified.jitterX = clipSpaceJitter.x;
          modified.jitterY = clipSpaceJitter.y;
        }

        // Ensure that memcpy can be used for fewer memory interactions
        static_assert(std::is_trivially_copyable_v<RtxVertexCaptureData>);
        allocAndMapVertexCaptureConstantBuffer() = modified;
      } else {
        RtxVSConstants modified = prevCB.vsConstants;
        {
          // Jittered projection for DLSS
          camera.applyJitterTo(modified.Projection,
                               m_device->getCurrentFrameId());
        }
        // Ensure that memcpy can be used for fewer memory interactions
        static_assert(std::is_trivially_copyable_v<RtxVSConstants>);
        allocAndMapVSConstantBuffer() = modified;
      }
    }

    DxvkRenderTargets skyRt;
    skyRt.color[0].view = getResourceManager().getCompatibleViewForView(skyMatteView, m_skyRtColorFormat);
    skyRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
    bindRenderTargets(skyRt);

    if (m_skyClearDirty) {
      DxvkContext::clearRenderTarget(skyMatteView, VK_IMAGE_ASPECT_COLOR_BIT, m_skyClearValue);
    }

    {
      // [SkySrvSanitize] Null-out integer-format PS SRV bindings around the
      // sky-matte draw. Same root cause as rasterizeToSkyProbe — TF2 leaves
      // a stale R32_UINT SRV bound at PS t19 (carried over from a non-sky
      // draw) which trips Vulkan's FLOAT-vs-UINT and LINEAR-filter validators
      // and produces NaN/Inf color output → on-screen pink/blue noise on
      // the entire framebuffer once the sky leaks into composite.
      // See PsSrvIntegerSanitizeGuard at top of this file.
      PsSrvIntegerSanitizeGuard srvGuard(this);

      if (params.indexCount == 0) {
        DxvkContext::draw(params.vertexCount, params.instanceCount, params.vertexOffset, 0);
      } else {
        DxvkContext::drawIndexed(params.indexCount, params.instanceCount, params.firstIndex, params.vertexOffset, 0);
      }
    }

    // Restore state
    if (prevClipSpaceJitterEnabled >= 0) {
      assert(prevClipSpaceJitterEnabled == 0 || prevClipSpaceJitterEnabled == 1);
      setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, RtxSpecConstantId::ClipSpaceJitterEnabled, prevClipSpaceJitterEnabled);
    }
    if (haveJitterCB) {
      if (drawCallState.usesVertexShader) {
        allocAndMapVertexCaptureConstantBuffer() = prevCB.programmablePipeline;
      } else {
        allocAndMapVSConstantBuffer() = prevCB.vsConstants;
      }
    }
  }

  void RtxContext::initSkyProbe() {
    auto skyProbeImage = getResourceManager().getSkyProbe(this, m_skyColorFormat).image;

    if (m_skyProbeImage == skyProbeImage)
      return;

    m_skyProbeImage = skyProbeImage;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_skyRtColorFormat;
    viewInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;

    for (uint32_t n = 0; n < 6; n++) {
      viewInfo.minLayer = n;
      m_skyProbeCubePlanes[n] = m_device->createImageView(m_skyProbeImage, viewInfo);
    }

    // [SkyTrace.probePrefill] Per-face single-layer STORAGE views. See
    // m_skyProbeCubePlaneStorageViews comment in rtx_context.h for why
    // we don't use a single 2D-array storage view (multi-layer dispatch
    // writes only landed on layer 0 in practice).
    DxvkImageViewCreateInfo storageViewInfo;
    storageViewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    storageViewInfo.format = m_skyRtColorFormat;
    storageViewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    storageViewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    storageViewInfo.minLevel = 0;
    storageViewInfo.numLevels = 1;
    storageViewInfo.numLayers = 1;
    for (uint32_t face = 0; face < 6; ++face) {
      storageViewInfo.minLayer = face;
      m_skyProbeCubePlaneStorageViews[face] =
        m_device->createImageView(m_skyProbeImage, storageViewInfo);
    }
  }

  void RtxContext::rasterizeToSkyProbe(const DrawParameters& params, const DrawCallState& drawCallState) {
    ScopedGpuProfileZone(this, "rasterizeToSkyProbe");

    // Lazy init
    initSkyProbe();

    // [SkyProbe.cubeRender] C-1 implementation. Replaces the broken d9-Remix
    // CustomVertexTransformEnabled spec-constant + vertexCaptureCB flow
    // (the DXBC compiler in this fork doesn't inject the consumer code).
    //
    // Approach: the d3d11 frontend captured cb2 (CBufCommonPerCamera) full
    // contents + the matrix offset for c_cameraRelativeToClip into
    // dcs.skyProbeCubeCapture. Per cube face we discard-allocate a fresh
    // slice on TF2's bound cb2 buffer, memcpy the snapshot back in, then
    // overwrite the matrix slot with a cube-face View×Projection. TF2's
    // own sky shader runs unmodified — it just sees a different
    // c_cameraRelativeToClip per face.
    //
    // No restore needed at the end: TF2 reissues Map(WRITE_DISCARD) +
    // memcpy on cb2 before its next non-sky draw, which discard-allocates
    // another fresh slice and writes new contents. Our last cube-face
    // override only persists in cb2's slice ring until then.
    // [SkyProbe.cubeRender.entry] Per-call entry diag — gameplay-gated
    // and frame-coalesced. Logs WHY this call is taking each branch
    // (skyMode, capture validity, cb2 presence) so we can see why the
    // cube-render only fires for the first 1-2 frames of a session and
    // then silently no-ops on every subsequent sky draw. Three distinct
    // failure modes pre-existed without log visibility; this section
    // exposes them by name.
    const SkyMode currentSkyMode = RtxOptions::skyMode();
    const auto& cap = drawCallState.skyProbeCubeCapture;
    Rc<DxvkBuffer> cb2;
    bool cb2InRange = false;
    if (cap.vsCb2DxvkSlot < m_rc.size()) {
      cb2InRange = true;
      cb2 = m_rc[cap.vsCb2DxvkSlot].bufferSlice.buffer();
    }
    {
      static std::atomic<uint32_t> sLastFrame { UINT32_MAX };
      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t lastLogged = sLastFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && lastLogged != frameId) {
        sLastFrame.store(frameId, std::memory_order_relaxed);
        Logger::info(str::format(
          "[SkyProbe.cubeRender.entry] frame=", frameId,
          " skyMode=", static_cast<uint32_t>(currentSkyMode),
          " capValid=", (cap.valid ? "true" : "false"),
          " capSlot=", cap.vsCb2DxvkSlot,
          " slotInRange=", (cb2InRange ? "true" : "false"),
          " cb2Buffer=", (cb2 != nullptr ? "non-null" : "NULL"),
          " cb2ByteSize=", cap.cb2ByteSize,
          " matOffset=", cap.cb2MatrixOffset));
      }
    }

    // [SkyProbe.cubeRender.skyModeGate] Cube render only makes sense in
    // Hybrid / PhysicalAtmosphere where the path tracer actually consumes
    // the cubemap. In SkyboxRasterization (mode 0) the cube is unused
    // and running the render still corrupts cb2 — leaking the last-face
    // matrix into subsequent sky-matte draws if TF2 reuses the cb. Skip
    // entirely when the mode doesn't need it.
    if (currentSkyMode != SkyMode::Hybrid
        && currentSkyMode != SkyMode::PhysicalAtmosphere) {
      return;
    }

    if (!cap.valid) {
      ONCE(Logger::warn("[SkyProbe.cubeRender] cb2 capture not present on "
                        "sky draw — sky cubemap stays empty. Verify the "
                        "frontend's CaptureSkyProbeCubeFromCb is firing on "
                        "Sky-classified draws."));
      return;
    }
    if (!cb2InRange) {
      ONCE(Logger::err(str::format("[SkyProbe.cubeRender] captured cb2 slot ",
                                    cap.vsCb2DxvkSlot, " out of m_rc range ",
                                    m_rc.size())));
      return;
    }
    if (cb2 == nullptr) {
      ONCE(Logger::warn("[SkyProbe.cubeRender] cb2 buffer not bound at "
                        "captured slot — skipping cubemap render."));
      return;
    }

    // [SkyTrace.probePrefill] On the first sky draw of the frame, fill all
    // 6 cube faces with the analytic Hillaire sky × skyTint via a compute
    // dispatch. TF2's sky-shader draws then overlay this background where
    // they have geometry coverage. Faces with no TF2 coverage (notably -Y
    // for upper-hemisphere skyboxes) keep the analytic content instead of
    // the previous fixed-color clear, so IBL/PSR sample a coherent 360°
    // sky regardless of what TF2 emits. The per-face clearRenderTarget
    // below is now gated off when prefill ran — it would otherwise nuke
    // the prefill content immediately.
    bool prefillRan = false;
    {
      // [SkyTrace.probePrefill.gate] Log every gameplay-frame entry into
      // rasterizeToSkyProbe so we can see why prefill does/doesn't fire.
      static std::atomic<uint32_t> sLastFrame{ UINT32_MAX };
      const uint32_t fid = m_device->getCurrentFrameId();
      const uint32_t lastLogged = sLastFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(fid);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      if (cameraValid && tlasReady && lastLogged != fid) {
        sLastFrame.store(fid, std::memory_order_relaxed);
        Logger::info(str::format(
          "[SkyTrace.probePrefill.gate] frame=", fid,
          " skyClearDirty=", (m_skyClearDirty ? 1 : 0),
          " atmosphereNonNull=", (m_atmosphere != nullptr ? 1 : 0),
          " plane0NonNull=", (m_skyProbeCubePlaneStorageViews[0] != nullptr ? 1 : 0),
          " probeImageNonNull=", (m_skyProbeImage != nullptr ? 1 : 0),
          " probeImageExtent=",
            (m_skyProbeImage != nullptr ? m_skyProbeImage->info().extent.width : 0u),
          " probeImageUsage=0x", std::hex,
            (m_skyProbeImage != nullptr ? m_skyProbeImage->info().usage : 0u),
          std::dec));
      }
    }
    if (m_skyClearDirty && m_atmosphere != nullptr
        && m_skyProbeCubePlaneStorageViews[0] != nullptr) {
      // NV-DXVK [sky-probe compute-in-renderpass crash fix]: the atmosphere prefill below does
      // COMPUTE work (computeLuts + dispatchCubeSkyPrefill) and emits compute-stage memory
      // barriers (rtx_atmosphere.cpp:604). When rasterizeToSkyProbe is reached with a graphics
      // render pass still active (from the sky raster path), those barriers/dispatches execute
      // inside the render pass instance — illegal in Vulkan (dstStageMask must be graphics-only
      // inside a render pass) → driver fault → VK_ERROR_DEVICE_LOST (Aftermath crash, the freeze
      // seen in dxgi.log). dxvk's own dispatch() spills before compute, but the explicit barrier
      // in the prefill fires before that. End the render pass here first; the per-face cube draws
      // below re-open their own pass. No functional change — same compute, just legal ordering.
      this->spillRenderPass(true);
      m_atmosphere->initialize(this);
      m_atmosphere->computeLuts(this);
      m_atmosphere->dispatchCubeSkyPrefill(this, m_skyProbeCubePlaneStorageViews,
                                           m_skyProbeImage->info().extent.width);
      Logger::info(str::format(
        "[SkyTrace.probePrefill.dispatch] frame=", m_device->getCurrentFrameId(),
        " size=", m_skyProbeImage->info().extent.width));
      // Compute-write -> color-attachment-read barrier. The per-face draws
      // below blend into the cube as color attachments; we need the prefill
      // writes visible to the rasterizer.
      this->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
        | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
      prefillRan = true;
    }

    // Save rasterizer state — cube faces are rendered with cull=NONE so
    // any sky geometry winding-flips at the equator are tolerated.
    const DxvkRsInfo &ri = m_state.gp.state.rs;
    DxvkRasterizerState prevRasterizerState {};
    {
      DxvkRasterizerState newRs;
      newRs.depthClipEnable  = ri.depthClipEnable();
      newRs.depthBiasEnable  = ri.depthBiasEnable();
      newRs.polygonMode      = ri.polygonMode();
      newRs.cullMode         = ri.cullMode();
      newRs.frontFace        = ri.frontFace();
      newRs.sampleCount      = ri.sampleCount();
      newRs.conservativeMode = ri.conservativeMode();
      prevRasterizerState    = newRs;
      newRs.cullMode = VK_CULL_MODE_NONE;
      setRasterizerState(newRs);
    }

    // Cube-face viewport / scissor. Square (cube faces are square).
    const auto& skyProbeExt = m_skyProbeImage->info().extent;
    VkViewport viewport { 0,
      static_cast<float>(skyProbeExt.height),
      static_cast<float>(skyProbeExt.width),
      -static_cast<float>(skyProbeExt.height),
      0.f, 1.f
    };
    VkRect2D scissor { { 0, 0 }, { skyProbeExt.width, skyProbeExt.height } };
    setViewports(1, &viewport, &scissor);

    // 90° FOV unit-aspect projection. TF2's sky shader pre-subtracts
    // c_cameraOrigin from worldPos before multiplying by
    // c_cameraRelativeToClip, so our override matrix is just
    // (proj * cubeFaceRotation) — no translation.
    Matrix4 proj;
    // Column-major Matrix4: m[col][row]
    proj[0] = Vector4(1.0f, 0.0f, 0.0f, 0.0f);
    proj[1] = Vector4(0.0f, 1.0f, 0.0f, 0.0f);
    proj[2] = Vector4(0.0f, 0.0f, 1.0f, 1.0f);  // depth -> w
    proj[3] = Vector4(0.0f, 0.0f, 0.0f, 0.0f);

    // [SkyProbe.cubeRender.run] Track per-face success. mapPtr can come
    // back null on certain memory pressures or first-frame races; we
    // skip the face but want to see if it happens. Reported once per
    // gameplay frame.
    uint32_t facesRendered = 0;
    uint32_t facesSkippedNullMap = 0;

    // [SkySrvSanitize] Null-out integer-format PS SRV bindings around the
    // 6-face draw. See PsSrvIntegerSanitizeGuard at top of this file.
    PsSrvIntegerSanitizeGuard srvGuard(this);

    for (uint32_t face = 0; face < 6; ++face) {
      // View matrix: rotation only (camera-at-origin) — TF2 shader
      // already subtracts c_cameraOrigin before this.
      const Matrix4 view = makeViewMatrixForCubePlane(face, Vector3(0.0f, 0.0f, 0.0f));
      const Matrix4 vpMatrix = proj * view;

      // TF2 cb2 is row-major. Our Matrix4 is column-major. Transpose
      // back to row-major for the cb2 write so TF2's HLSL reads the
      // expected layout.
      float vpRowMajor[16];
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          vpRowMajor[row * 4 + col] = vpMatrix[col][row];
        }
      }

      // Discard-allocate fresh cb2 slice. invalidateBuffer renames the
      // buffer's m_physSlice so the bound slot picks up the new memory.
      DxvkBufferSliceHandle slice = cb2->allocSlice();
      invalidateBuffer(cb2, slice);
      uint8_t* dst = static_cast<uint8_t*>(slice.mapPtr);
      if (dst == nullptr) {
        ++facesSkippedNullMap;
        continue;
      }
      // Restore the rest of cb2 around our matrix override so TF2's
      // sky shader sees its other fields (c_cameraOrigin, c_skyColor,
      // c_sunDir, etc.) at the values TF2 had when the draw was queued.
      std::memcpy(dst, cap.cb2Snapshot, cap.cb2ByteSize);
      // Write the cube-face matrix on top.
      std::memcpy(dst + cap.cb2MatrixOffset, vpRowMajor, 64);
      ++facesRendered;

      // Bind the cube face as the render target.
      DxvkRenderTargets skyRt;
      skyRt.color[0].view   = m_skyProbeCubePlanes[face];
      skyRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
      bindRenderTargets(skyRt);

      // Skip the per-face clear when the prefill compute already wrote
      // analytic-sky content into this face — clearing now would replace
      // the Hillaire background with the m_skyClearValue (typically black)
      // and reintroduce the empty-face problem.
      if (m_skyClearDirty && !prefillRan) {
        DxvkContext::clearRenderTarget(m_skyProbeCubePlanes[face],
                                       VK_IMAGE_ASPECT_COLOR_BIT,
                                       m_skyClearValue);
      }

      if (params.indexCount > 0) {
        DxvkContext::drawIndexed(params.indexCount, params.instanceCount,
                                  params.firstIndex, params.vertexOffset, 0);
      } else {
        DxvkContext::draw(params.vertexCount, params.instanceCount,
                          params.vertexOffset, 0);
      }
    }

    // [SkyProbe.cubeRender.cbRestore] After the 6-face loop, cb2 holds
    // the last cube-face matrix. If TF2 emits multiple sky draws per
    // frame (e.g. main sky + 3D skybox) without re-Map'ing cb2 between
    // them, the next sky-matte draw would render with our wrong matrix
    // — exactly what produced the yellow regression. Restore cb2 to
    // the original snapshot via the same discard-allocate pattern.
    {
      DxvkBufferSliceHandle restoreSlice = cb2->allocSlice();
      invalidateBuffer(cb2, restoreSlice);
      uint8_t* dst = static_cast<uint8_t*>(restoreSlice.mapPtr);
      if (dst != nullptr) {
        std::memcpy(dst, cap.cb2Snapshot, cap.cb2ByteSize);
      }
    }

    // Mark probe populated so the path tracer can switch to SkyProbe
    // sampling for IBL in Hybrid mode (otherwise it falls back to
    // Hillaire-only). Require all 6 faces to have rendered — partial
    // populations leave 5 faces with stale/cleared content which
    // would cause hard tinting bands in indirect IBL.
    if (facesRendered == 6) {
      m_skyProbeCubemapPopulated = true;
    }

    // [SkyProbe.cubeRender.run] Frame-coalesced gameplay-gated success
    // log. Fires the first time the probe is populated, then once per
    // gameplay frame. Helps verify the cube-face render is firing 6×
    // and that no faces are being silently skipped.
    {
      static std::atomic<uint32_t> sLastLoggedFrame { UINT32_MAX };
      static std::atomic<bool>     sLoggedFirstSuccess { false };
      const uint32_t frameId = m_device->getCurrentFrameId();
      const uint32_t lastLogged = sLastLoggedFrame.load(std::memory_order_relaxed);
      const bool cameraValid = getSceneManager().getCamera().isValid(frameId);
      const bool tlasReady = getSceneManager().getInstanceTable().size() >= 32u;
      const bool firstSuccess = (facesRendered == 6) &&
        !sLoggedFirstSuccess.load(std::memory_order_relaxed);
      const bool gameplayFrame = cameraValid && tlasReady && lastLogged != frameId;
      if (firstSuccess || gameplayFrame) {
        sLastLoggedFrame.store(frameId, std::memory_order_relaxed);
        if (facesRendered == 6) {
          sLoggedFirstSuccess.store(true, std::memory_order_relaxed);
        }
        Logger::info(str::format(
          "[SkyProbe.cubeRender.run] frame=", frameId,
          " facesRendered=", facesRendered,
          " facesSkippedNullMap=", facesSkippedNullMap,
          " populated=", (m_skyProbeCubemapPopulated ? "true" : "false"),
          firstSuccess ? " (first full population)" : ""));
      }
    }

    // Restore rasterizer state. cb2 is left with the last cube face's
    // matrix; TF2 will overwrite cb2 via Map(WRITE_DISCARD) before its
    // next draw, so the override doesn't leak.
    setRasterizerState(prevRasterizerState);
    return;
    // Dead-code preservation: original implementation continued below
    // for the d9-style spec-constant override. Kept commented out so
    // the diff is reviewable but compiles to zero.
#if 0
    // Restore prev VS jitter CBs in d9-Remix lineage.
    if (drawCallState.usesVertexShader) {
      allocAndMapVertexCaptureConstantBuffer() = prevCB.programmablePipeline;
    } else {
      allocAndMapVSConstantBuffer() = prevCB.vsConstants;
    }
#endif
  }

  void RtxContext::bakeTerrain(const DrawParameters& params, DrawCallState& drawCallState, const MaterialData** outOverrideMaterialData) {
    if (!getSceneManager().getTerrainBaker().enableBaking() ||
        !drawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return;
    }

    DrawCallTransforms& transformData = drawCallState.transformData;

    // Terrain Baker (may) update bound color textures, so preserve the views
    Rc<DxvkImageView> previousColorView;
    Rc<DxvkImageView> previousSecondaryColorView;

    OpaqueMaterialData* opaqueReplacementMaterial = nullptr;
    TerrainBaker& terrainBaker = getSceneManager().getTerrainBaker();

    if (!TerrainBaker::debugDisableBaking()) {

      // Retrieve the replacement material
      MaterialData* replacementMaterial = getSceneManager().getAssetReplacer()->getReplacementMaterial(drawCallState.getMaterialData().getHash());

      if (replacementMaterial) {
        if (replacementMaterial->getType() == MaterialDataType::Opaque) {
          opaqueReplacementMaterial = &replacementMaterial->getOpaqueMaterialData();

          // Original 0th colour texture slot
          const uint32_t colorTextureSlot = drawCallState.materialData.colorTextureSlot[0];

          // Save current color texture first
          if (colorTextureSlot < m_rc.size() && m_rc[colorTextureSlot].imageView != nullptr) {
            previousColorView = m_rc[colorTextureSlot].imageView;
          }          
          
        } else {
          ONCE(Logger::warn(str::format("[RTX Texture Baker] Only opaque replacement materials are supported for terrain baking. Texture hash ",
                                        drawCallState.getMaterialData().getHash(),
                                        " has a non-opaque replacement material set. Baking the texture with legacy material instead.")));
        }
      }
    }

    // Bake the material
    const bool isBaked = terrainBaker.bakeDrawCall(this, m_state, m_rtState, params, drawCallState, opaqueReplacementMaterial, transformData.textureTransform);

    if (isBaked) {
      // Bind the baked terrain texture to the mesh
      if (!TerrainBaker::debugDisableBinding()) {

        // Set the terrain's baked material data
        *outOverrideMaterialData = terrainBaker.getMaterialData();

        // Generate texcoords in the RT shader
        transformData.texgenMode = TexGenMode::CascadedViewPositions;

        // Update the legacy material data with legacy value defaults as well as set the color textur since some of its data 
        // is still used through the Rt pipeline even though overrideMaterialData is specifide. 
        // Also SceneManager uses sampler associated with the color texture to patch samplers for the textures in the opaque material.
        LegacyMaterialData overrideMaterial;
        overrideMaterial.colorTextures[0] = (*outOverrideMaterialData)->getOpaqueMaterialData().getAlbedoOpacityTexture();
        overrideMaterial.samplers[0] = terrainBaker.getTerrainSampler();
        overrideMaterial.updateCachedHash();
        drawCallState.materialData = overrideMaterial;
      }

      // Restore state modified during baking
      if (!TerrainBaker::debugDisableBaking()) {

        // Restore bound color texture views
        if (previousColorView != nullptr) {
          bindResourceView(drawCallState.materialData.colorTextureSlot[0], previousColorView, nullptr);
        }
      }
    }
  }

  void RtxContext::rasterizeSky(const DrawParameters& params, const DrawCallState& drawCallState) {
    // Skip rasterization when using physical atmosphere mode
    if (RtxOptions::skyMode() == SkyMode::PhysicalAtmosphere) {
      return;
    }
    
    // Grab and apply replacement texture if any
    // NOTE: only the original color texture will be replaced with albedo-opacity texture
    MaterialData* replacementMaterial = getSceneManager().getAssetReplacer()->getReplacementMaterial(drawCallState.getMaterialData().getHash());
    bool replacemenIsLDR = false;
    Rc<DxvkImageView> replacementTexture = {};
    uint32_t replacementTextureSlot = UINT32_MAX;

    if (replacementMaterial && replacementMaterial->getType() == MaterialDataType::Opaque) {
      // Must pull a ref because we will modify it for loading purposes below.
      TextureRef& albedoOpacity = replacementMaterial->getOpaqueMaterialData().getAlbedoOpacityTexture();

      if (albedoOpacity.isValid()) {
        uint32_t textureIndex;
        getSceneManager().trackTexture(albedoOpacity, textureIndex, true, false);

        if (!albedoOpacity.isImageEmpty()) {
          replacementTextureSlot = drawCallState.materialData.colorTextureSlot[0];
          replacementTexture = albedoOpacity.getImageView();
          replacemenIsLDR = TextureUtils::isLDR(albedoOpacity.getImageView()->info().format);
        } else {
          ONCE(Logger::warn("A replacement texture for sky was specified, but it could not be loaded."));
        }
      }
    }
    
    Rc<DxvkImageView> curColorView = {};
    if (replacementTextureSlot < m_rc.size())
    {
      if (m_rc[replacementTextureSlot].imageView != nullptr && replacementTexture != nullptr) {
        // Save currently bound texture to restore later
        curColorView = m_rc[replacementTextureSlot].imageView;
        // Bind a replacement texture
        bindResourceView(replacementTextureSlot, replacementTexture, nullptr);
      }
    }

    // Save current RTs
    DxvkRenderTargets curRts = m_state.om.renderTargets;

    if (!TextureUtils::isLDR(m_skyRtColorFormat) && (!replacementMaterial || replacemenIsLDR)) {
      ONCE(Logger::warn("Sky may not appear correct: sky intermediate format has been forced to HDR "
                        "while the original sky is LDR and no HDR sky replacement has been found!"));
    }

    // Save viewports
    const uint32_t curViewportCount = m_state.gp.state.rs.viewportCount();
    const DxvkViewportState curVp = m_state.vp;

    rasterizeToSkyMatte(params, drawCallState);
    rasterizeToSkyProbe(params, drawCallState);

    m_skyClearDirty = false;

    // Restore VPs
    setViewports(curViewportCount, curVp.viewports.data(), curVp.scissorRects.data());

    // Restore RTs
    bindRenderTargets(curRts);

    // Restore color texture
    if (curColorView != nullptr) {
      bindResourceView(drawCallState.materialData.colorTextureSlot[0], curColorView, nullptr);
    }
  }

  void RtxContext::clearRenderTarget(const Rc<DxvkImageView>& imageView,
                                     VkImageAspectFlags clearAspects, VkClearValue clearValue) {
    // Capture color for skybox clear
    if (clearAspects & VK_IMAGE_ASPECT_COLOR_BIT) {
      m_skyClearValue = clearValue;

      // Set dirty flag so that next skyprobe rasterize will clear the views.
      // We assume that skybox drawcalls will immediately follow the clear. The logic would
      // need to be revisited if this is not true for some game.
      m_skyClearDirty = true;
    }

    DxvkContext::clearRenderTarget(imageView, clearAspects, clearValue);
  }

  void RtxContext::clearImageView(const Rc<DxvkImageView>& imageView, VkOffset3D offset,
                                  VkExtent3D extent, VkImageAspectFlags aspect, VkClearValue value) {
    // Capture color for skybox clear
    if (aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
      m_skyClearValue = value;

      // Set dirty flag so that next skyprobe rasterize will clear the views.
      // We assume that skybox drawcalls will immediately follow the clear. The logic would
      // need to be revisited if this is not true for some game.
      m_skyClearDirty = true;
    }

    DxvkContext::clearImageView(imageView, offset, extent, aspect, value);
  }

  void RtxContext::reportCpuSimdSupport() {
    switch (fast::getSimdSupportLevel()) {
    case fast::AVX512:
      dxvk::Logger::info("CPU supports SIMD: AVX512");
      break;
    case fast::AVX2:
      dxvk::Logger::info("CPU supports SIMD: AVX2");
      break;
    case fast::SSE4_1:
      dxvk::Logger::info("CPU supports SIMD: SSE 4.1");
      break;
    case fast::SSE3:
      dxvk::Logger::info("CPU supports SIMD: SSE 3");
      break;
    case fast::SSE2:
      dxvk::Logger::info("CPU supports SIMD: SSE 2");
      break;
    case fast::None:
      dxvk::Logger::info("CPU doesn't support SIMD");
      break;
    default:
      Logger::err("Invalid SIMD state");
      break;
    }
  }

  const DxvkScInfo& RtxContext::getSpecConstantsInfo(VkPipelineBindPoint pipeline) const {
    return
      pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
      ? m_state.gp.state.sc
      : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
      ? m_state.cp.state.sc
      : m_state.rp.state.sc;
  }

  void RtxContext::setSpecConstantsInfo(
    VkPipelineBindPoint pipeline,
    const DxvkScInfo& newSpecConstantInfo) {
    DxvkScInfo& specConstantInfo =
      pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
      ? m_state.gp.state.sc
      : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
      ? m_state.cp.state.sc
      : m_state.rp.state.sc;

    if (specConstantInfo != newSpecConstantInfo) {
      specConstantInfo = newSpecConstantInfo;

      m_flags.set(
        pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
        ? DxvkContextFlag::GpDirtyPipelineState
        : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
          ? DxvkContextFlag::CpDirtyPipelineState
          : DxvkContextFlag::RpDirtyPipelineState);
    }
  }

#ifdef REMIX_DEVELOPMENT
  void RtxContext::cacheResourceAliasingImageView(const Rc<DxvkImageView>& imageView) {
    if (imageView.ptr()) {
      // Determine the format compatibility category for the image view
      const auto formatCategory = Resources::getFormatCompatibilityCategory(imageView->info().format);
      const auto categoryIndex = static_cast<uint32_t>(formatCategory);
      const auto& underlyingImage = imageView->image();

      // Proceed only if the category is valid and the image view is tracked in the resource view map
      if (formatCategory != RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory &&
          Resources::s_resourcesViewMap.find(imageView.ptr()) != Resources::s_resourcesViewMap.end()) {
        bool aliasingMatchFound = false;
        // Search the cache for an existing aliased resource with the same underlying image
        for (auto& compatibleResource : m_resourceCacheTable[categoryIndex]) {
          if (compatibleResource.view->image() == underlyingImage) {
            // Match found: update the begin and end pass stages to expand their range
            compatibleResource.beginPassStage = std::min(compatibleResource.beginPassStage, m_currentPassStage);
            compatibleResource.endPassStage = std::max(compatibleResource.endPassStage, m_currentPassStage);

            // Add the current resource name to the set of names for this aliased group
            compatibleResource.names.insert(Resources::s_resourcesViewMap[imageView.ptr()]);
            aliasingMatchFound = true;
            break;
          }
        }

        if (!aliasingMatchFound) {
          // No match found: cache this as a new aliased resource entry
          m_resourceCacheTable[categoryIndex].push_back({ imageView, m_currentPassStage, m_currentPassStage, { Resources::s_resourcesViewMap[imageView.ptr()] } });
        }
      }
    }
  }

  void RtxContext::queryAvailableResourceAliasing() {
    // Check if aliasing query is enabled through user settings
    if (!Resources::s_queryAliasing) {
      return;
    }

    // Set the aliasing resource dimensions based on user options
    const VkExtent3D extent = { RtxOptions::Aliasing::width(), RtxOptions::Aliasing::height(), RtxOptions::Aliasing::depth() };
    // Get the start and end frame pass stages from user settings
    const RtxFramePassStage beginPass = RtxOptions::Aliasing::beginPass();
    const RtxFramePassStage endPass = RtxOptions::Aliasing::endPass();

    std::string newResourceAliasingQueryResult;
    // Check if the begin pass is before the end pass
    if (beginPass > endPass) {
      // Set an error message if the begin pass is invalid
      Resources::s_resourceAliasingQueryText = "Begin Pass must be before the End Pass";
      return;
    }

    // Lambda function to check if a resource matches the aliasing criteria
    auto isResourceMatches = [&](const Rc<DxvkImageView>& view) {
      const auto& imageInfo = view->image()->info();
      const auto& viewInfo = view->info();
      uint32_t aliasingWidth = RtxOptions::Aliasing::width();
      uint32_t aliasingHeight = RtxOptions::Aliasing::height();

      // Adjust dimensions if the aliasing extent type is DownScaledExtent or TargetExtent
      if (RtxOptions::Aliasing::extentType() == RtxTextureExtentType::DownScaledExtent) {
        aliasingWidth = getResourceManager().getDownscaleDimensions().width;
        aliasingHeight = getResourceManager().getDownscaleDimensions().height;
      } else if (RtxOptions::Aliasing::extentType() == RtxTextureExtentType::TargetExtent) {
        aliasingWidth = getResourceManager().getTargetDimensions().width;
        aliasingHeight = getResourceManager().getTargetDimensions().height;
      }

      // Check if the resource's dimensions and format match the aliasing query settings
      return imageInfo.extent.width == aliasingWidth &&
              imageInfo.extent.height == aliasingHeight &&
              imageInfo.extent.depth == RtxOptions::Aliasing::depth() &&
              imageInfo.numLayers == RtxOptions::Aliasing::layer() &&
              imageInfo.type == RtxOptions::Aliasing::imageType() &&
              viewInfo.type == RtxOptions::Aliasing::imageViewType();
    };

    std::string manualSolveResources;
    uint32_t matchedIndex = 0;

    bool aliasingMatchFound = false;
    const auto category = RtxOptions::Aliasing::formatCategory();
    if (category == RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory) {
      // If the format category is invalid, no aliasing can be done
      Resources::s_resourceAliasingQueryText = "Please select aliasing compatible texture format.";
      return;
    }

    // Map category to index for cache lookup
    const uint32_t index = static_cast<uint32_t>(category);

    // Loop through the resource cache table for the corresponding format category
    for (auto& compatibleResource : m_resourceCacheTable[index]) {
      // Check if the resource is compatible with the aliasing query (based on pass stages and matching criteria)
      if ((endPass < compatibleResource.beginPassStage || beginPass > compatibleResource.endPassStage ||
            (beginPass != endPass && compatibleResource.beginPassStage != compatibleResource.endPassStage &&
            (endPass == compatibleResource.beginPassStage || beginPass == compatibleResource.endPassStage))) &&
          (isResourceMatches(compatibleResource.view))) {

        // Loop through names of matching resources and prepare result string
        for (const auto& name : compatibleResource.names) {
          if (Resources::s_dynamicAliasingResourcesSet.find(compatibleResource.view.ptr()) == Resources::s_dynamicAliasingResourcesSet.end()) {
            ++matchedIndex;
            newResourceAliasingQueryResult += std::to_string(matchedIndex) + ". " + name + "\n";
            if (matchedIndex > 10) {
              break; // Limit to 10 results
            }
          } else {
            if (manualSolveResources.empty()) {
              // Give notification for users who want to do aliasing for dynamic resources
              manualSolveResources = "[WARNING] Use caution when aliasing dynamic resources. Ensure aliasing is handled every frame in Resources::onFrameBegin.\n";
            }
            manualSolveResources += name + "\n";
          }
        }
        aliasingMatchFound = true;
      }
    }

    // Set the result of the aliasing query, either showing available resources or a no-match message
    Resources::s_resourceAliasingQueryText =
      aliasingMatchFound ? (newResourceAliasingQueryResult + manualSolveResources) : "No available resources that can be aliased, please create a new resource.";
  }

  void RtxContext::clearResourceAliasingCache() {
    // Clean up caches
    for (auto& resourceCaches : m_resourceCacheTable) {
      resourceCaches.clear();
    }
    m_currentPassStage = RtxFramePassStage::FrameEnd;
  }

  void RtxContext::analyzeResourceAliasing() {
    // Early exit if the aliasing analyzer option is not enabled
    if (!Resources::s_startAliasingAnalyzer) {
      return;
    }

    // Lambda to check if two image views are compatible for aliasing
    auto isResourceCompatible = [](const Rc<DxvkImageView>& view, const Rc<DxvkImageView>& matchedView) {
      const auto& imageInfo = view->image()->info();
      const auto& matchedImageInfo = matchedView->image()->info();
      const auto& viewInfo = view->info();
      return imageInfo.extent == matchedImageInfo.extent &&
             imageInfo.numLayers == matchedImageInfo.numLayers &&
             imageInfo.type == matchedImageInfo.type;
    };

    std::string availableAliasingText;

    // Iterate over all format compatibility categories
    for (uint32_t index = 0; index < static_cast<uint32_t>(RtxTextureFormatCompatibilityCategory::Count); ++index) {
      const std::vector<ResourceCache>& cacheList = m_resourceCacheTable[index];

      // Compare each pair of resources within the same format category
      for (size_t i = 0; i < cacheList.size(); ++i) {
        for (size_t j = i + 1; j < cacheList.size(); ++j) {
          // Check for non-overlapping lifetimes (safe for aliasing)
          if ((cacheList[i].endPassStage < cacheList[j].beginPassStage || cacheList[i].beginPassStage > cacheList[j].endPassStage ||
               (cacheList[i].beginPassStage != cacheList[i].endPassStage && cacheList[j].beginPassStage != cacheList[j].endPassStage &&
                (cacheList[i].endPassStage == cacheList[j].beginPassStage || cacheList[i].beginPassStage == cacheList[j].endPassStage))) &&
              isResourceCompatible(cacheList[i].view, cacheList[j].view)) {
            // Add the resource names to the output text
            availableAliasingText += *cacheList[i].names.begin() + " <-> " + *cacheList[j].names.begin() + "\n";
          }
        }
      }
    }

    // Output the results to the GUI field
    Resources::s_aliasingAnalyzerResultText = !availableAliasingText.empty() ? availableAliasingText : "Can't find any resources that can be aliased.\n";
  }
#endif
} // namespace dxvk
