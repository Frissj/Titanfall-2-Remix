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
#include <algorithm>
#include <cassert>
#include <array>
#include <chrono>
#include <future>
// NV-DXVK [Perf.Report]: [Perf.GpuPass] and [CommitRT] both emit from this file
// and are the GPU and dxvk-cs anchors of the assembled breakdown.
#include "rtx_perf_report.h"
// NV-DXVK [Perf.Sweep]: std::this_thread::sleep_for, used to let the log drain
// before TerminateProcess. <future> tends to pull this in, but not guaranteed.
#include <thread>
#include <set>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <functional>   // std::hash, for the [NsysAuto] push/pop thread check

// NV-DXVK [NsysAuto]: NVTX, vendored at src/nvtx3 (Apache-2.0, header-only, no
// link dependency). Used only to bracket the capture window so Nsight Systems
// can be driven with '--capture-range=nvtx', i.e. the GAME opens and closes
// collection from inside the process.
//
// This replaced an interactive 'nsys launch' + 'nsys start' wrapper, which
// failed on this game every time: 'nsys start' threw
// 'ETWSession::Start ... SystemException' after ~13 s, three runs in a row,
// with cpuctxsw / sample / wddm all disabled. One-shot 'nsys profile' on the
// same game with the same trace options produced a 7.4 MB report first try, so
// the fault was the mid-run attach, not the options. NVTX keeps the
// gameplay-accurate trigger while never calling 'nsys start'.
//
// When not running under nsys the injection library is absent and every NVTX
// call is a cheap no-op, so these are safe to leave compiled in.
#include "../../nvtx3/nvToolsExt.h"
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
#include "rtx/pass/coverage/coverage_compact.h"
#include <rtx_shaders/coverage_compact.h>
#include <rtx_shaders/tlas_probe.h>
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
    // NV-DXVK [Coverage compact]: compute pass that folds the 74 MB
    // surface-coverage buffer down to its nonzero (index, value) pairs on
    // the GPU, so the per-frame [Coverage] dump never scans uncached
    // host-visible memory on the CPU (that scan measured ~1 s/frame).
    class CoverageCompactShader : public ManagedShader {
      SHADER_SOURCE(CoverageCompactShader, VK_SHADER_STAGE_COMPUTE_BIT, coverage_compact)

      PUSH_CONSTANTS(CoverageCompactArgs)

      BEGIN_PARAMETER()
        RW_STRUCTURED_BUFFER(COVERAGE_COMPACT_INPUT)
        RW_STRUCTURED_BUFFER(COVERAGE_COMPACT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(CoverageCompactShader);

    // NV-DXVK [TlasProbe]: shoots rays at the BUILT acceleration structure, one
    // thread per surface. Uses COMMON_RAYTRACING_BINDINGS and nothing else -
    // that set already carries the TLAS, the surface buffer, the camera in cb
    // and the coverage buffer it writes, so bindCommonRayTracingResources is
    // the entire binding step and this pass adds no descriptors of its own.
    class TlasProbeShader : public ManagedShader {
      SHADER_SOURCE(TlasProbeShader, VK_SHADER_STAGE_COMPUTE_BIT, tlas_probe)

      // Required: the probe reads each surface's own index and position buffers
      // through the bindless arrays to find the triangle it shoots through.
      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(TlasProbeShader);

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

    // NV-DXVK [Perf] 2026-08-14: gate the WORK on the log denylist, same lever
    // as D3D11Rtx::DumpSubViewSkyTextures.
    //
    // This had no off switch at all -- only s_done and the instance-count gate
    // below -- so every run paid it. Measured 02:29: 277 .dds files, each a GPU
    // readback plus a DDS encode and a disk write on rtx-asset-exporter, all
    // landing ~10s after gameplay is detected. Being one-shot it does not touch
    // steady state, but it does mean the first ~10-20s of EVERY capture is dirty,
    // which is exactly the window a short measurement run lives in.
    //
    // Hoisted into a function-local static rather than called per invocation:
    // tagDenied strlen()s the tag and memcmps its bucket (rtx_instance_manager.cpp
    // :5177 records that cost being real at per-instance rates). The denylist is
    // published once by Logger::setDenyTags during RtxOptions init and never
    // changes, so a static is the correct lifetime. It resolves on the first call
    // -- after the tag index exists -- and s_done is left alone so RTX_D3D11_DIAG=1
    // or removing the entry from rtx.logDenyTags restores dump and log together.
    static const bool kOnScreenAlbedoDenied =
      Logger::tagDenied("[OnScreenAlbedoDump]");
    if (kOnScreenAlbedoDenied) {
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
    // NV-DXVK [perf]: let DxvkContext::dispatch write a post-barrier timestamp on
    // request. See markGpuStageBeforeNextDispatch.
    m_gpuStageMarkFn = [](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->markGpuStage();
    };

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

    // NV-DXVK [MtnRadiance] steering — read the options HERE, on the main thread.
    // RtxOptions is not safe to touch from the async decode worker below, and the
    // values must be the ones in force for THIS frame's readback.
    // A degenerate rect falls back to the full screen rather than sampling nothing,
    // so a typo in rtx.conf can't silently kill the probe.
    const Vector4 mtnRegionV = RtxOptions::mtnRadianceRegion();
    float mtnMinXf = mtnRegionV.x, mtnMinYf = mtnRegionV.y, mtnMaxXf = mtnRegionV.z, mtnMaxYf = mtnRegionV.w;
    if (!(mtnMaxXf > mtnMinXf) || !(mtnMaxYf > mtnMinYf)) {
      mtnMinXf = 0.0f; mtnMinYf = 0.0f; mtnMaxXf = 1.0f; mtnMaxYf = 1.0f;
    }
    mtnMinXf = std::clamp(mtnMinXf, 0.0f, 1.0f); mtnMaxXf = std::clamp(mtnMaxXf, 0.0f, 1.0f);
    mtnMinYf = std::clamp(mtnMinYf, 0.0f, 1.0f); mtnMaxYf = std::clamp(mtnMaxYf, 0.0f, 1.0f);
    const float mtnMinAbsViewZ = RtxOptions::mtnRadianceMinAbsViewZ();

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
       shipBoxMinX, shipBoxMinY, shipBoxMaxX, shipBoxMaxY,
       mtnMinXf, mtnMinYf, mtnMaxXf, mtnMaxYf, mtnMinAbsViewZ]() {
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

        // Coarse grid over rtx.mtnRadianceRegion (default = the original upper-60% band).
        constexpr uint32_t COLS = 24u, ROWS = 12u;
        const uint32_t regX0 = uint32_t(mtnMinXf * float(W));
        const uint32_t regY0 = uint32_t(mtnMinYf * float(H));
        const uint32_t regW  = (uint32_t(mtnMaxXf * float(W)) > regX0) ? (uint32_t(mtnMaxXf * float(W)) - regX0) : 1u;
        const uint32_t regH  = (uint32_t(mtnMaxYf * float(H)) > regY0) ? (uint32_t(mtnMaxYf * float(H)) - regY0) : 1u;
        for (uint32_t row = 0u; row < ROWS; ++row) {
          const uint32_t y = std::min(regY0 + (row * regH) / ROWS, H - 1u);
          for (uint32_t col = 0u; col < COLS; ++col) {
            const uint32_t x = std::min(regX0 + ((col * 2u + 1u) * regW) / (COLS * 2u), W - 1u);
            const float viewZ = *reinterpret_cast<const float*>(pz + (VkDeviceSize(y) * W + x) * 4u);
            // Distance gate, now rtx.mtnRadianceMinAbsViewZ (default 1e4 = original
            // behaviour: distant backdrop hits only). Set it to 0 to log every sampled
            // pixel raw. That matters when the question is "did the ray hit the object
            // at all" — a miss or a near hit is precisely what a nonzero gate throws
            // away, so leaving the default on would answer the wrong question.
            if (mtnMinAbsViewZ > 0.0f && !(std::fabs(viewZ) > mtnMinAbsViewZ))
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

        // NV-DXVK [HitCensus]: DENSE per-VS primary-hit census.
        //
        // The question no probe in this codebase answers: on a frame where a
        // shader's geometry is gone, did a primary ray HIT it at all?
        // [Coverage] counts pixels AFTER resolution, so it cannot separate
        // "never hit" from "hit and resolved to another surface". The
        // [MtnRadiance] grid above samples 288 fixed texels, which caught the
        // shader under investigation 27 times in 325152 samples - far too
        // sparse to condition on.
        //
        // m_sharedSurfaceIndex is ALREADY copied in full to host memory above
        // for that grid, so histogramming every texel costs one pass over
        // W*H on a worker thread and needs no GPU change, no new buffer and no
        // shader hook. This is deliberately a measurement with no hypothesis
        // baked into it: it reports what the primary rays resolved to, and the
        // comparison against [Coverage]'s pixel count for the same VS on the
        // same frame is what generates the lead.
        //
        //   hits > 0  while [Coverage] pixels == 0 -> ray hit it, lost after
        //                                             resolution (shading /
        //                                             attribution).
        //   hits == 0                              -> the ray never touched
        //                                             geometry that is
        //                                             provably in the TLAS
        //                                             with correct mask,
        //                                             transform and BLAS, so
        //                                             the defect is in
        //                                             traversal itself.
        if (RtxOptions::logPrimaryHitCensus()) {
          std::unordered_map<uint64_t, uint32_t> hitsByVs;
          uint32_t invalidSurf = 0u;
          uint32_t oobSurf     = 0u;
          const uint32_t total = W * H;
          for (uint32_t i = 0u; i < total; ++i) {
            const uint32_t s = *reinterpret_cast<const uint32_t*>(ps + VkDeviceSize(i) * 4u);
            if (s == SURFACE_INDEX_INVALID) {
              ++invalidSurf;
              continue;
            }
            if (s >= surf.size()) {
              // Resolved index outside THIS frame's surface table. Counted
              // separately rather than dropped: a nonzero value here is itself
              // a finding (the GBuffer referencing surfaces that no longer
              // exist), and folding it into a VS bucket would hide it.
              ++oobSurf;
              continue;
            }
            ++hitsByVs[surf[s].vs];
          }

          // NV-DXVK [HitIdent]: DRAW-LEVEL identity per vertex shader.
          //
          // Every Remix-side property compared so far (mask, geometry flags,
          // categories, frontFace, mirror, cull, blend, decal, TLAS routing,
          // BLAS liveness) is IDENTICAL between shaders that flicker 40%+ and
          // shaders that never flicker. So the difference is likely on the GAME
          // side, in what these draws actually are.
          //
          // studioModelName is filled by the studiorender draw-site hook
          // (d3d11_rtx.cpp:22768) and carried on the BLAS input, and matHash /
          // texHash are the per-model Remix material and albedo-texture hashes.
          // The standalone name dumps (tf2DumpStudioNames) dedupe by name with
          // NO vertex-shader attribution, which is exactly what is needed here,
          // so this walks the surface table instead and reports identity keyed
          // by VS.
          //
          // Emitted for the whole surface table, not just hit surfaces: a
          // flickering shader contributes ~0 primary hits, so keying this off
          // hits would omit precisely the geometry under investigation.
          // Throttled 1-in-10 - identity is near-static, unlike the hit counts,
          // but 10 keeps the first line ~1.7s in at this load rather than 20s.
          {
            static uint32_t s_identN = 0u;
            if ((s_identN++ % 10u) == 0u) {
              struct VsIdent {
                uint32_t surfaces = 0u;
                uint64_t matHash = 0ull, texHash = 0ull;
                uint32_t sv = 0u, svSky = 0u, nb = 0u, widow = 0u;
                const char* name = nullptr;
              };
              std::unordered_map<uint64_t, VsIdent> ident;
              for (size_t s = 0; s < surf.size(); ++s) {
                VsIdent& e = ident[surf[s].vs];
                ++e.surfaces;
                if (e.matHash == 0ull) { e.matHash = surf[s].matHash; }
                if (e.texHash == 0ull) { e.texHash = surf[s].texHash; }
                e.sv    += surf[s].isSubView;
                e.svSky += surf[s].isSubViewSkybox;
                e.nb    += surf[s].hasNormal;
                e.widow += surf[s].isWidow;
                if (e.name == nullptr && surf[s].name[0] != '\0') {
                  e.name = surf[s].name;
                }
              }
              for (const auto& e : ident) {
                Logger::info(str::format(
                  "[HitIdent] f=", frameId,
                  " vs=0x", std::hex, e.first, std::dec,
                  " surfaces=", e.second.surfaces,
                  " hits=", (hitsByVs.count(e.first) ? hitsByVs[e.first] : 0u),
                  " mat=0x", std::hex, e.second.matHash, std::dec,
                  " tex=0x", std::hex, e.second.texHash, std::dec,
                  " svN=", e.second.sv,
                  " svSkyN=", e.second.svSky,
                  " nbN=", e.second.nb,
                  " widowN=", e.second.widow,
                  " name=", (e.second.name ? e.second.name : "(none)")));
              }
            }
          }
          Logger::info(str::format(
            "[HitCensus] === f=", frameId,
            " pixels=", total,
            " distinctVS=", hitsByVs.size(),
            " invalidSurf=", invalidSurf,
            " oobSurf=", oobSurf,
            " surfTableSize=", surf.size(), " ==="));
          // Sorted so the log is diffable frame to frame.
          std::vector<std::pair<uint64_t, uint32_t>> ranked(hitsByVs.begin(), hitsByVs.end());
          std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<uint64_t, uint32_t>& a, const std::pair<uint64_t, uint32_t>& b) {
              return a.second > b.second;
            });
          for (const auto& e : ranked) {
            Logger::info(str::format(
              "[HitCensus]   f=", frameId,
              " vs=0x", std::hex, e.first, std::dec,
              " hits=", e.second));
          }
        }
      }));
  }

  // NV-DXVK [TonemapProbe]: read the tonemap INPUT (m_compositeOutput, HDR radiance)
  // and OUTPUT (m_finalOutput, post-operator display colour) at a sparse pixel grid
  // and log what the operator did to them. Both buffers are RGBA16F but may differ
  // in resolution (upscaling), so sample by normalized pos. Async readback
  // (fence + worker) so it never stalls the frame.
  //
  // What it measures, and why it changed
  // ------------------------------------
  // The original version logged RGB and luma, which answers "did the curve run"
  // and nothing else. That is the question you ask once. The questions an
  // operator comparison actually turns on are how much chroma survived, whether
  // the hue moved, and how much of the frame arrived at a limit - and none of
  // those are visible in a luma number. So each sample now carries luminance,
  // ICtCp chroma and hue on both sides, and the frame emits an aggregate line.
  //
  // The aggregate line has a fixed field order on purpose. Comparing operators
  // properly means running the same scene under each configuration and diffing:
  //
  //     rtx.tonemap.operator      0 / 3 / 6 / 7 / 8 / 9 / 10
  //     rtx.autoExposurePlus      on / off
  //     rtx.tonemap.colorGrading  off, for any comparison to mean anything
  //     rtx.bloom                 off, same reason
  //
  // `op=`, `plus=` and `grade=` are recorded on every line so a log can be
  // attributed to a configuration rather than to a memory of what was set.
  // Set rtx.tf2HeavyProbes for the per-sample [TonemapProbe.px] lines too.
  //
  // The probe reads the final image, so it sees what the operator produced
  // after the operator's own internal clamp. Interior pre-clamp values are not
  // observable from here by construction; for PSDT that is what
  // rtx.tonemap.psdt.debugView 10 exists for.
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
    // Recorded alongside every sample so a log can be attributed to a
    // configuration rather than to a guess about what was enabled at the time.
    // The four-way operator x Plus comparison is four runs and a diff of these
    // lines; without them it is four runs and a memory.
    const uint32_t plusActive = m_common->metaAutoExposurePlus().isActive() ? 1u : 0u;
    const uint32_t gradingOn = DxvkToneMapping::colorGradingEnabled() ? 1u : 0u;
    const uint32_t perPixel = RtxOptions::tf2HeavyProbes() ? 1u : 0u;

    for (auto it = sTasks.begin(); it != sTasks.end();) {
      if (it->valid() && it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        it = sTasks.erase(it);
      else
        ++it;
    }

    sTasks.push_back(std::async(std::launch::async,
      [bin, bout, fence, fv, Wi, Hi, Wo, Ho, frameId, op,
       plusActive, gradingOn, perPixel]() {
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
        // Luminance, chroma and hue - not RGB and luma.
        //
        // "Did the curve run" is answerable from luma alone and was all this
        // probe ever answered. The questions an operator comparison actually
        // turns on are not: how much chroma survived, whether the hue moved,
        // and how much of the frame arrived at a limit. Those need a
        // perceptual space, so the probe carries a small ICtCp of its own -
        // the same one psdt_perceptual_space.slangh uses, at the same 100-nit
        // reference, so a number logged here is comparable with a number from
        // tools/psdt/ rather than merely similar to one.
        auto lum = [](float r, float g, float b) {
          return 0.2126390059f * r + 0.7151686788f * g + 0.0721923154f * b;
        };
        auto pqEncode = [](float v) {
          const float y = std::max(v, 0.0f) * 100.0f / 10000.0f;
          const float ym = std::pow(y, 2610.0f / 16384.0f);
          return std::pow((0.8359375f + 18.8515625f * ym) / (1.0f + 18.6875f * ym),
                          2523.0f / 4096.0f * 128.0f);
        };
        // Linear Rec.709 -> ICtCp, returning (I, chroma, hue in radians).
        auto ictcp = [&](float r, float g, float b, float& chroma, float& hue) {
          const float r2 = 0.6274039f * r + 0.3292830f * g + 0.0433131f * b;
          const float g2 = 0.0690973f * r + 0.9195404f * g + 0.0113623f * b;
          const float b2 = 0.0163914f * r + 0.0880132f * g + 0.8955953f * b;
          const float l = pqEncode((1688.f * r2 + 2146.f * g2 + 262.f * b2) / 4096.f);
          const float m = pqEncode((683.f * r2 + 2951.f * g2 + 462.f * b2) / 4096.f);
          const float s = pqEncode((99.f * r2 + 309.f * g2 + 3688.f * b2) / 4096.f);
          const float ct = (6610.f * l - 13613.f * m + 7003.f * s) / 4096.f;
          const float cp = (17933.f * l - 17390.f * m - 543.f * s) / 4096.f;
          chroma = std::sqrt(ct * ct + cp * cp);
          hue = std::atan2(cp, ct);
          return (2048.f * l + 2048.f * m) / 4096.f;
        };

        // Sparse normalized grid so input and output (possibly different res)
        // sample the same screen points. Wider than the original six: the
        // aggregate below is only as good as its sample count, and a readback
        // this size costs the same whatever is read out of it.
        const float us[] = { 0.15f, 0.30f, 0.45f, 0.60f, 0.75f, 0.90f };
        const float vs[] = { 0.25f, 0.40f, 0.55f, 0.70f };

        uint32_t n = 0, clipHi = 0, clipLo = 0, negIn = 0;
        float dHueSum = 0.0f, dHueMax = 0.0f, chromaRatioSum = 0.0f, chromaRatioN = 0.0f;
        float inYMin = 1e30f, inYMax = -1e30f, outYMin = 1e30f, outYMax = -1e30f;

        for (float v : vs) {
          for (float u : us) {
            const uint32_t xi = uint32_t(u * (Wi - 1u)), yi = uint32_t(v * (Hi - 1u));
            const uint32_t xo = uint32_t(u * (Wo - 1u)), yo = uint32_t(v * (Ho - 1u));
            const uint16_t* ip = reinterpret_cast<const uint16_t*>(pin  + (VkDeviceSize(yi) * Wi + xi) * 8u);
            const uint16_t* op16 = reinterpret_cast<const uint16_t*>(pout + (VkDeviceSize(yo) * Wo + xo) * 8u);
            const float ir = halfToFloat(ip[0]),  ig = halfToFloat(ip[1]),  ib = halfToFloat(ip[2]);
            const float orr = halfToFloat(op16[0]), og = halfToFloat(op16[1]), ob = halfToFloat(op16[2]);

            float cIn = 0.0f, hIn = 0.0f, cOut = 0.0f, hOut = 0.0f;
            ictcp(ir, ig, ib, cIn, hIn);
            ictcp(orr, og, ob, cOut, hOut);
            const float yIn = lum(ir, ig, ib), yOut = lum(orr, og, ob);

            // Hue is undefined at zero chroma, so a sample that arrived at
            // white contributes to the clipping count and not to the hue
            // statistic. Averaging atan2 noise in would make a transform that
            // correctly went white look like one with a hue problem.
            float dHue = 0.0f;
            const bool hueMeaningful = cIn > 0.02f && cOut > 0.02f;
            if (hueMeaningful) {
              dHue = hOut - hIn;
              dHue -= 6.283185307f * std::floor(dHue / 6.283185307f + 0.5f);
              dHue = std::abs(dHue) * 57.29577951f;
              dHueSum += dHue;
              dHueMax = std::max(dHueMax, dHue);
              chromaRatioSum += cOut / cIn;
              chromaRatioN += 1.0f;
            }

            // Item 10, the half of it that is observable from here: the
            // renderer's own output can be negative (denoisers undershoot),
            // and the operator's can sit exactly at a limit. Interior
            // pre-clamp values are not visible from a readback of the final
            // image at all - that needs shader instrumentation, which for PSDT
            // is rtx.tonemap.psdt.debugView 10.
            const bool anyNegIn = (ir < 0.0f) || (ig < 0.0f) || (ib < 0.0f);
            const bool anyHi = (orr >= 0.999f) || (og >= 0.999f) || (ob >= 0.999f);
            const bool anyLo = (orr <= 0.0f) || (og <= 0.0f) || (ob <= 0.0f);
            negIn += anyNegIn ? 1u : 0u;
            clipHi += anyHi ? 1u : 0u;
            clipLo += anyLo ? 1u : 0u;

            inYMin = std::min(inYMin, yIn);   inYMax = std::max(inYMax, yIn);
            outYMin = std::min(outYMin, yOut); outYMax = std::max(outYMax, yOut);
            ++n;

            if (perPixel != 0u) {
              Logger::info(str::format(
                "[TonemapProbe.px] f=", frameId, " op=", op,
                " uv=(", u, ",", v, ")",
                " in=(", ir, ",", ig, ",", ib, ") Yin=", yIn, " Cin=", cIn,
                " -> out=(", orr, ",", og, ",", ob, ") Yout=", yOut, " Cout=", cOut,
                " dHue=", dHue, (hueMeaningful ? "" : " (achromatic)")));
            }
          }
        }

        if (n != 0u) {
          // One line per frame, fixed field order. This is the line to diff
          // across the operator x Plus matrix.
          Logger::info(str::format(
            "[TonemapProbe] f=", frameId, " op=", op, " plus=", plusActive,
            " grade=", gradingOn, " n=", n,
            " Yin=[", inYMin, "..", inYMax, "]",
            " Yout=[", outYMin, "..", outYMax, "]",
            " clipHi=", clipHi, "/", n, " clipLo=", clipLo, "/", n,
            " negIn=", negIn, "/", n,
            " dHueMean=", (chromaRatioN > 0.0f ? dHueSum / chromaRatioN : 0.0f),
            " dHueMax=", dHueMax,
            " chromaOutIn=", (chromaRatioN > 0.0f ? chromaRatioSum / chromaRatioN : 0.0f),
            " chromaSamples=", uint32_t(chromaRatioN)));
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
      // [Perf.Frame] entry_tailToBranch sub-split — find the real leaf of the ~171ms.
      int64_t tail_gpuIdleUs   = 0; // CPU cost of the getGpuIdleTimeSinceLastCall() call
      float   gpuIdleMs        = 0.f; // ACTUAL GPU idle time this frame. ~0 => GPU saturated
                                      // (GPU-bound); large fraction of the frame => GPU starved
                                      // (CPU-bound). The decisive CPU-vs-GPU-bound discriminator.
      int64_t tail_preTexUs    = 0; // screenshot checks + particles + spillRenderPass
      int64_t tail_texUploadUs = 0; // submitTexturesToDeviceLocal (streaming uploads)
      int64_t tail_preSceneUs  = 0; // barriers + reflex + EngineSun
      int64_t tail_prepSceneUs = 0; // prepareSceneData (rebuild all GPU scene buffers)
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

    // [Perf.Frame] entry_tailToBranch sub-split: find the real leaf of the ~171ms
    // span (tEntryAfterHotReload -> RT-branch fork). markTail mirrors markStage
    // (per-frame assignment). Buckets only fill on frames that reach the RT branch.
    auto tTail = tEntryAfterHotReload;
    auto markTail = [](PerfClock::time_point& last, int64_t& sink) {
      const auto now = PerfClock::now();
      sink = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
      last = now;
    };

    const float gpuIdleTimeMilliseconds = getGpuIdleTimeSinceLastCall();
    perfFrame.gpuIdleMs = gpuIdleTimeMilliseconds;
    markTail(tTail, perfFrame.tail_gpuIdleUs);

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
      markTail(tTail, perfFrame.tail_preTexUs);

      // NV-DXVK [perf]: open the GPU timestamp frame HERE, not at the top of the
      // RT branch. The old t0 sat after prepareSceneData, which meant the BLAS and
      // TLAS builds — the largest GPU work outside the path tracer — were never
      // measured at all, and neither was texture upload. Both are candidates for
      // the ~150 ms/frame that [Perf.Gpu] fenceWaitMs sees but [Perf.GpuPass]
      // totalMs did not account for. Running before the branch also means menu and
      // TLAS-not-ready frames still get sampled instead of vanishing.
      beginGpuStageFrame();
      markGpuStage();  // t0

      getCommonObjects()->getTextureManager().submitTexturesToDeviceLocal(this, m_execBarriers, m_execAcquires);
      markTail(tTail, perfFrame.tail_texUploadUs);
      markGpuStage();  // texUpload

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

      markTail(tTail, perfFrame.tail_preSceneUs);

      // NV-DXVK [ABSweep]: F9 alternates rtx.enableSeparateUnorderedApproximations
      // BASE/TEST within one session so both arms see the same viewpoint.
      //
      // Placed BEFORE prepareSceneData on purpose. That option decides TLAS
      // routing (Tlas::Unordered vs Tlas::Opaque, rtx_accel_manager.cpp:8123),
      // so it has to be settled before the scene is built. Flipping it after
      // the build — where the old serialize sweep sat — would apply it to a
      // frame that had already been routed under the previous value, and every
      // arm boundary would be measuring a mixture.
      //
      // Why this variable: [HitCensus] showed the flickering shaders never win
      // a primary hit anywhere on screen (id29 has primary hits on 7% of the
      // frames Coverage reports it drawn; id34/id21/id41/id16 on ~1%) while
      // shaders that DO win primary hits agree with Coverage ~100% of the time
      // and are stable. Hit counts sum to exactly 518400 = every pixel, so
      // that is not a measurement gap: the flickering geometry is composited
      // through the unordered/non-primary path.
      //
      // setImmediately, not setDeferred: deferred resolves at end of frame, so
      // the value would land one frame late and smear across arm boundaries.
      {
        static bool     s_sweepActive     = false;
        static bool     s_sweepTestArm    = false;
        static uint32_t s_sweepPhaseFrame = 0u;
        static uint32_t s_sweepCycle      = 0u;
        static bool     s_sweepOrigValue  = true;

        // allowContinuousPress=false: this is a TOGGLE, so it must fire once
        // per press. Passing true (as the 'B' debugger-break above does) would
        // flip the arm every frame F9 is held and shred the sweep.
        if (ImGUI::checkHotkeyState({ VirtualKey{ VK_F9 } }, false)) {
          s_sweepActive = !s_sweepActive;
          if (s_sweepActive) {
            // Remember what the conf asked for, so stopping restores it rather
            // than leaving the game stuck in whichever arm it happened to end
            // on.
            s_sweepOrigValue = RtxOptions::enableSeparateUnorderedApproximations();
          } else {
            RtxOptions::enableSeparateUnorderedApproximationsObject().setImmediately(s_sweepOrigValue);
          }
          // Always (re)start in the BASE arm: the baseline has to be measured
          // at this exact spot first, or a location that simply is not
          // flickering reads as a successful fix.
          s_sweepTestArm    = false;
          s_sweepPhaseFrame = 0u;
          s_sweepCycle      = 0u;
          Logger::info(str::format(
            "[ABSweep] ", (s_sweepActive ? "STARTED" : "STOPPED"),
            " var=enableSeparateUnorderedApproximations",
            " frame=", m_device->getCurrentFrameId(),
            " framesPerPhase=", RtxOptions::abSweepFramesPerPhase(),
            " cycles=", RtxOptions::abSweepCycles(),
            " — hold position; arms alternate automatically"));
        }

        if (s_sweepActive) {
          const uint32_t perPhase = std::max(1u, RtxOptions::abSweepFramesPerPhase());
          if (s_sweepPhaseFrame >= perPhase) {
            s_sweepPhaseFrame = 0u;
            s_sweepTestArm = !s_sweepTestArm;
            // A cycle completes on the fall back to BASE, i.e. after a full
            // BASE->TEST pair, so `cycles` counts matched pairs not half-arms.
            if (!s_sweepTestArm) {
              ++s_sweepCycle;
              if (s_sweepCycle >= RtxOptions::abSweepCycles()) {
                s_sweepActive = false;
                RtxOptions::enableSeparateUnorderedApproximationsObject().setImmediately(s_sweepOrigValue);
                Logger::info(str::format(
                  "[ABSweep] FINISHED frame=", m_device->getCurrentFrameId(),
                  " completedCycles=", s_sweepCycle));

                if (RtxOptions::abSweepExitOnFinish()) {
                  Logger::warn("[ABSweep] EXIT-ON-FINISH: terminating process "
                               "(rtx.abSweepExitOnFinish).");
                  // Let the log thread drain — TerminateProcess skips the
                  // static teardown that would otherwise flush the file
                  // stream, and the final arm's markers are the ones joined.
                  std::this_thread::sleep_for(std::chrono::milliseconds(750));
                  // TerminateProcess, not exit(): this fork's shutdown path
                  // calls a cached client.dll pointer after the engine has
                  // unloaded that module, so a clean exit crashes on the way
                  // out. Same reason as the perf sweep and nsys capture.
                  ::TerminateProcess(::GetCurrentProcess(), 0u);
                }
              }
            }
          }
          ++s_sweepPhaseFrame;

          // BASE = whatever the conf asked for; TEST = the opposite. Applied
          // every frame, not just on transitions, so a value written elsewhere
          // cannot silently drift the arm mid-stretch.
          RtxOptions::enableSeparateUnorderedApproximationsObject()
            .setImmediately(s_sweepTestArm ? !s_sweepOrigValue : s_sweepOrigValue);

          // Logged EVERY frame on both arms, un-throttled, so each Coverage
          // frame can be attributed to the arm that produced it. A
          // transition-only marker would leave the join guessing exactly at
          // the arm boundaries, which is where the interesting frames are.
          Logger::info(str::format(
            "[ABSweep] f=", m_device->getCurrentFrameId(),
            " arm=", (s_sweepTestArm ? "TEST" : "BASE"),
            " sepUnordered=", (RtxOptions::enableSeparateUnorderedApproximations() ? 1 : 0),
            " cycle=", s_sweepCycle,
            " phaseFrame=", s_sweepPhaseFrame));
        }
      }

      // Update all the GPU buffers needed to describe the scene
      getSceneManager().prepareSceneData(this, m_execBarriers);

      // NV-DXVK [SerializeSceneBuild]: hard barrier between the scene build and
      // everything that reads it. See rtx.debugSerializeSceneBuild for why.
      //
      // Placed HERE, after prepareSceneData returns, because that call owns the
      // whole build — BLAS, TLAS, the surface/instance uploads and the
      // PointInstancer culling dispatch — so a wait at this point covers every
      // ordering hazard between producing scene data and consuming it, without
      // having to guess which pass is at fault. If the flip rate collapses, the
      // wait then gets walked backwards pass by pass to find the one that
      // matters.
      //
      // flushCommandList() first: waitForIdle only waits on work already
      // SUBMITTED, and the build is still sitting in the open command list at
      // this point. Idling without flushing would wait on the previous frame
      // and prove nothing — a false negative that would look like a clean
      // exoneration.
      if (RtxOptions::debugSerializeSceneBuild()) {
        // flushCommandList() FIRST: waitForIdle only waits on work already
        // submitted, and the scene build is still in the open command list
        // here. Idling without flushing would wait on the previous frame and
        // silently produce a clean-looking null result.
        // (Measured: this changes nothing. Kept as a bisecting instrument.)
        flushCommandList();
        m_device->waitForIdle();
      }

      markTail(tTail, perfFrame.tail_prepSceneUs);
      markGpuStage();  // prepScene — contains the BLAS + TLAS builds

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

        // NV-DXVK [perf]: "gpuDrain" mark. Splits what used to be one
        // onFrameBegin stage measuring 91 ms of a 156 ms frame - 61% of all GPU
        // time, four times the path tracer - which is not credible for what
        // onFrameBegin actually records. The companion tell is that prepScene
        // reads 0.057 ms even though its comment says it contains the BLAS and
        // TLAS builds, over 9.77M verts/frame. So the heavy AS and geometry work
        // is almost certainly NOT inside the span it is being charged to.
        //
        // Between the prepScene mark and THIS one, essentially no GPU commands
        // are recorded - only the tlasReady/getSurfaceBuffer CPU checks. So:
        //   gpuDrain large, onFrameBegin ~0  -> the GPU is draining work recorded
        //     earlier (AS builds, geometry interleave) or stalled at a submit
        //     boundary. The 91 ms was misattributed and the real cost is upstream.
        //   gpuDrain ~0, onFrameBegin large  -> onFrameBegin genuinely records
        //     91 ms of GPU work, and the next split goes inside it.
        markGpuStage();  // gpuDrain — GPU catch-up before onFrameBegin records anything

        VkExtent3D downscaledExtent = onFrameBegin(targetImage->info().extent);
        markStage(tStage, perfFrame.onFrameBeginUs);
        markGpuStage();

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
        markGpuStage();

        // Volumetric Lighting
        dispatchVolumetrics(rtOutput);
        markStage(tStage, perfFrame.volumetricsUs);
        markGpuStage();

        // Path Tracing
        dispatchPathTracing(rtOutput);
        markStage(tStage, perfFrame.pathTracingUs);
        markGpuStage();

        // Neural Radiance Cache
        m_common->metaNeuralRadianceCache().dispatchTrainingAndResolve(*this, rtOutput);
        markStage(tStage, perfFrame.nrcUs);
        markGpuStage();

        // RTXDI confidence
        //
        // NV-DXVK [perf]: this interval owns the frame -- 94.3 / 111.9 / 107.1 ms
        // of a 168 ms frame on a clean run (cpuSlowX=1.00), against a CPU counter
        // of 115 us, so the cost is GPU-side. But the passes immediately upstream
        // read implausibly low in those same windows (pt_gbuffer 0.67, nrc 0.06,
        // pathTracing 1.0, against 13.1 / 5.4 / 0.33 in earlier baselines at the
        // same fps), which is what a pipeline drain looks like when it lands on
        // the first pass with a hard dependency on the path-tracing output.
        //
        // This mark separates the two. It fires AFTER the pre-dispatch barrier
        // flush and BEFORE the first dispatch, so:
        //   rtxdi_barrierWait = time spent waiting on work queued earlier
        //   rtxdi             = the confidence pass's own dispatches
        //
        // Chosen over ablating the pass: setting rtx.di.enableDenoiserConfidence
        // False FROZE the game during load (2026-07-27 00:22, no fault logged).
        // This is behaviour-neutral -- nothing is disabled, the image does not
        // change, and it cannot hang.
        //
        // Trap 2: "rtxdi_barrierWait" is inserted into kStageNames before "rtxdi",
        // which is after gb_primaryRays, so kGbPrimaryRaysSlot is unaffected.
        // marks goes 29 -> 30. The mark is conditional (it fires on the next
        // dispatch), so markGpuStageIfPending below makes it unconditional -
        // otherwise a frame where confidence does not dispatch would drop a mark
        // and shift every later label.
        markGpuStageBeforeNextDispatch();
        m_common->metaRtxdiRayQuery().dispatchConfidence(this, rtOutput);
        markGpuStageIfPending();
        markStage(tStage, perfFrame.rtxdiUs);
        markGpuStage();

        // ReSTIR GI
        m_common->metaReSTIRGIRayQuery().dispatch(this, rtOutput);
        markStage(tStage, perfFrame.restirUs);
        markGpuStage();

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
        markGpuStage();

        // Note: Primary direct diffuse/specular radiance textures noisy and in a demodulated state after demodulation step.
        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("noisyDiffuse", rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read));
          takeScreenshot("noisySpecular", rtOutput.m_primaryDirectSpecularRadiance.image(Resources::AccessType::Read));
        }

        // Denoising
        dispatchDenoise(rtOutput);
        markStage(tStage, perfFrame.denoiseUs);
        markGpuStage();

        // Note: Primary direct diffuse/specular radiance textures denoised but in a still demodulated state after denoising step.
        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("denoisedDiffuse", rtOutput.m_primaryDirectDiffuseRadiance.image(Resources::AccessType::Read));
          takeScreenshot("denoisedSpecular", rtOutput.m_primaryDirectSpecularRadiance.image(Resources::AccessType::Read));
        }

        // Composition
        dispatchComposite(rtOutput);
        markStage(tStage, perfFrame.compositeUs);
        markGpuStage();

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
        markGpuStage();

        if (captureScreenImage && captureDebugImage) {
          takeScreenshot("rtxImagePostComposite", rtOutput.m_compositeOutput.resource(Resources::AccessType::Read).image);
        }

        getCommonObjects()->getTextureManager().copySamplerFeedbackToHost(this);
        dispatchObjectPicking(rtOutput, downscaledExtent, targetImage->info().extent);
        markGpuStage();

        // NV-DXVK [perf]: NULL-INTERVAL CONTROL, deliberately at this exact point.
        //
        // The mark above closes 'postComposite'. The mark below closes 'pc_null',
        // whose interval contains NOTHING - not a call, not a barrier, nothing but
        // the markStage CPU counter. Any nonzero reading on pc_null is time the
        // instrument attributes to zero commands.
        //
        // Why here specifically: postComposite reads 84-115 ms while its span is
        // already provably empty - copySamplerFeedbackToHost's copyBuffer is guarded
        // by bytesToCopy != 0 and [Perf.TexBudget] reports sfCount=0 so it never
        // runs, and dispatchObjectPicking does no GPU work without a pick request.
        // Every candidate cause has now been eliminated by measurement: the copy
        // (vacuous A/B, but sfCount=0 settles it), object picking (all its GPU work
        // sits behind popRequest()), and a submit boundary landing inside the span
        // (markCmdBuf comparison reports no *SUBMIT on any stage, any window).
        // gpuDrain proves an empty span CAN read <0.01 ms, but it sits early in the
        // frame, so it does not control for position. This one does.
        //
        //   pc_null ~= 0, postComposite still huge -> the time accrues across those
        //     two calls after all, and the next question is what the GPU is retiring
        //     there, not whether the number is real.
        //   pc_null huge -> the instrument bills zero commands at this point in the
        //     stream, and no per-stage number from here on can be trusted.
        //
        // Trap 2: inserted AFTER gb_primaryRays, so kGbPrimaryRaysSlot is unaffected.
        // kStageNames gains "pc_null" right after "postComposite" and marks goes
        // 28 -> 29; the ALIGNED check should read 29..29/29.
        markStage(tStage, perfFrame.postCompositeUs);
        markGpuStage();

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
        markGpuStage();

        RtxDustParticles& dust = m_common->metaDustParticles();
        dust.simulateAndDraw(this, m_state, rtOutput);

        // NV-DXVK [auto exposure plus]: local exposure correction, before bloom on purpose.
        // Bloom smears bright regions outwards, so a pyramid built after it would read an
        // inflated key around every highlight and darken a wider area than the highlight
        // actually covers - a dark halo, the exact artifact the edge-aware pyramid exists to
        // avoid. Running first also means bloom responds to the corrected exposure, which is
        // the right order for what is physically a lens effect.
        dispatchAutoExposurePlus(rtOutput);

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
        markGpuStage();

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
        markGpuStage();
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

    // NV-DXVK [Perf.ShaderClock]: read the cycle counters and log them.
    //
    // Gated ONLY on rtx.perfShaderClock. It deliberately shares nothing with the
    // coverage readback: that path is gated on rtx.logSurfaceCoverage, which arms
    // 52 atomics per primary hit for ~104 ms/frame, and reading timing counters
    // through a 104 ms switch would inflate the thing being timed - and inflate
    // one region more than the others, since those atomics live inside
    // opaqueSurfaceMaterialInteractionCreate.
    //
    // The buffer is HOST_VISIBLE|HOST_COHERENT, so this is a plain memory read of
    // 256 bytes with no compaction pass, no barrier and no fence. It is one frame
    // stale at worst, which is irrelevant for an accumulating counter.
    if (RtxOptions::perfShaderClock()) {
      const Rc<DxvkBuffer>& clockBuf = getResourceManager().getRaytracingOutput().m_shaderClockBuffer;

      if (clockBuf.ptr() != nullptr) {
        static uint32_t s_framesSinceLog = 0;
        const uint32_t interval = std::max(1u, RtxOptions::perfShaderClockLogInterval());

        if (++s_framesSinceLog >= interval) {
          s_framesSinceLog = 0;

          uint32_t* clk = reinterpret_cast<uint32_t*>(clockBuf->mapPtr(0));

          if (clk != nullptr) {
            // Region names must stay in step with the SHADER_CLOCK_* slot defines
            // in common_binding_indices.h.
            static const char* kRegionNames[SHADER_CLOCK_REGION_COUNT] = {
              "unoTraversal",
              "unoHitInfo",
              "unoSurfaceLoad",
              "siPositions",
              "siRest",
              "unoClip",
              "unoMaterial",
              "unoBlend",
              "orderedMaterial",
            };

            // DELTAS against the previous read, never a clear.
            //
            // MEASURED 2026-07-26: memsetting this buffer from the CPU at frame end
            // races the GPU, which is still accumulating into it - there is no
            // fence here and adding one would stall the frame. The clear wiped
            // in-flight counters and the read caught partial state, which showed up
            // as one interval reporting 27.8M hits against ~12.9M either side and
            // means swinging by 20x.
            //
            // Unsigned subtraction is wrap-safe across exactly one wrap, so this
            // also removes the overflow failure mode that the >>4 shift was added
            // for - the shift now only buys margin against wrapping TWICE inside
            // one interval, which at 1/16 sampling cannot happen.
            //
            // Reading cycles and hits a few nanoseconds apart is harmless: both are
            // monotonic, and over ~800K samples the ratio is stable.
            static uint32_t s_prev[SHADER_CLOCK_REGION_COUNT * SHADER_CLOCK_REGION_STRIDE] = {};

            std::string line = "[Perf.ShaderClock] frames=" + std::to_string(interval);
            uint64_t totalCycles = 0;

            uint32_t deltaCycles[SHADER_CLOCK_REGION_COUNT] = {};
            uint32_t deltaHits[SHADER_CLOCK_REGION_COUNT] = {};

            for (uint32_t r = 0; r < SHADER_CLOCK_REGION_COUNT; ++r) {
              const uint32_t cSlot = r * SHADER_CLOCK_REGION_STRIDE + 0u;
              const uint32_t hSlot = r * SHADER_CLOCK_REGION_STRIDE + 1u;
              const uint32_t cNow = clk[cSlot];
              const uint32_t hNow = clk[hSlot];

              deltaCycles[r] = cNow - s_prev[cSlot];
              deltaHits[r]   = hNow - s_prev[hSlot];

              s_prev[cSlot] = cNow;
              s_prev[hSlot] = hNow;

              totalCycles += uint64_t(deltaCycles[r]);
            }

            for (uint32_t r = 0; r < SHADER_CLOCK_REGION_COUNT; ++r) {
              const uint32_t cycles = deltaCycles[r];
              const uint32_t hits   = deltaHits[r];

              // Mean per execution, not the total: a total conflates "expensive"
              // with "frequent", which is the exact ambiguity the unordered census
              // had to be added to resolve. Share of total is printed alongside so
              // both readings are available.
              // Shifted back up: the shader accumulates cycles >> CYCLE_SHIFT so
              // the uint32 accumulator cannot wrap. Granularity is 16 cycles.
              const double mean = (hits > 0u)
                ? (double(cycles) * double(1u << SHADER_CLOCK_CYCLE_SHIFT) / double(hits))
                : 0.0;
              const double share = (totalCycles > 0u) ? (100.0 * double(cycles) / double(totalCycles)) : 0.0;

              line += str::format(" | ", kRegionNames[r],
                                  " hits=", hits,
                                  " meanCycles=", mean,
                                  " share=", share, "%");
            }

            Logger::warn(line);

            // Deliberately NOT cleared - see the delta note above. Clearing from
            // the CPU while the GPU accumulates is the race that corrupted the
            // first instrumented run.
          }
        }
      }
    }

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

        // NV-DXVK [perf]: effective-CPU-speed probe. Runs a FIXED amount of CPU
        // work and times it, so the result depends only on how fast this machine
        // is currently executing code — not on anything the renderer is doing.
        //
        // Why this exists: on 2026-07-26 every CPU bucket in the frame
        // ([Perf.PrepScene] merge/cull/upload, and even a bare PSSetSRV in
        // [Perf.Entry]) inflated ~3x over ~20 s while instance and surface counts
        // stayed flat and GPU pass times — which come from GPU timestamps — did
        // not move at all. A uniform multiplier across unrelated CPU work with
        // constant input is not something renderer code can cause. This field
        // tells you whether a "regression" is the machine slowing down or the
        // code doing more, which is otherwise unanswerable from the log and
        // silently invalidates any before/after comparison spanning minutes.
        //
        // Why a latency chain and not the OS's reported MHz: CallNtPowerInformation
        // returns an averaged, often stale CurrentMhz, and would drag in
        // powrprof.dll. A dependent multiply-add chain is latency-bound, so it
        // tracks real core clock x IPC, which is what actually determines how
        // long the frame's CPU work takes. It costs ~20 us and runs once per
        // 5 s log, i.e. ~0.0004% of wall time.
        //
        // Reading it: cpuCalNs is the raw time for the fixed work. cpuSlowX is
        // that divided by the FASTEST sample seen this session, so 1.00 means
        // "as fast as this machine has been" and 3.00 means every CPU operation
        // is taking three times as long as it did at its best. Compare code
        // changes only between samples at similar cpuSlowX.
        int64_t cpuCalNs = 0;
        double cpuSlowX = 1.0;
        {
          volatile uint64_t sink = 0;
          uint64_t x = 1;
          const auto tCal0 = std::chrono::steady_clock::now();
          // Dependent LCG step: each iteration needs the previous result, so the
          // chain cannot be vectorised or run out-of-order into parallel lanes.
          for (uint32_t i = 0; i < 20000u; ++i) {
            x = x * 6364136223846793005ull + 1442695040888963407ull;
          }
          const auto tCal1 = std::chrono::steady_clock::now();
          sink = x;              // keeps the chain from being optimised away
          (void)sink;
          cpuCalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tCal1 - tCal0).count();

          static int64_t sCpuCalBestNs = 0;
          if (cpuCalNs > 0 && (sCpuCalBestNs == 0 || cpuCalNs < sCpuCalBestNs)) {
            sCpuCalBestNs = cpuCalNs;
          }
          if (sCpuCalBestNs > 0) {
            cpuSlowX = double(cpuCalNs) / double(sCpuCalBestNs);
          }
        }

        Logger::info(str::format("[Perf.Frame] fid=", fid,
                                 " framesSinceLastLog=", sFramesSinceLastLog,
                                 " cpuCalNs=", cpuCalNs,
                                 " cpuSlowX=", cpuSlowX,
                                 " rtBranchRan=", (perfFrame.rtBranchRan ? 1 : 0),
                                 " totalInjectUs=", totalInjectUs,
                                 " entryToOnFrameBegin=", perfFrame.entryToOnFrameBeginUs,
                                 " entry_preCommitGfx=", perfFrame.entry_preCommitGfxUs,
                                 " entry_commitGfx=", perfFrame.entry_commitGfxUs,
                                 " entry_postCommitGfx=", perfFrame.entry_postCommitGfxUs,
                                 " entry_hotReload=", perfFrame.entry_hotReloadUs,
                                 " entry_tailToBranch=", perfFrame.entry_tailToBranchUs,
                                 " tail_gpuIdle=", perfFrame.tail_gpuIdleUs,
                                 " GPUIDLEms=", perfFrame.gpuIdleMs,
                                 " tail_preTex=", perfFrame.tail_preTexUs,
                                 " tail_texUpload=", perfFrame.tail_texUploadUs,
                                 " tail_preScene=", perfFrame.tail_preSceneUs,
                                 " tail_prepScene=", perfFrame.tail_prepSceneUs,
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

        // NV-DXVK [SkyTrace.topHitSurf]: top-band (upward-ray) surface-index
        // split, read PRE-ALIAS. This readback runs at line ~5510, BEFORE the
        // disocclusion pass reuses m_sharedSurfaceIndex storage — so it sees the
        // real primary-gbuffer surface indices. (The composite-time copy in
        // recordPrimaryMissCountReadback reads the ALIASED buffer and is
        // meaningless — see [SkyTrace.topHitSurfAliased] for the broken control.)
        // Replicates primaryMiss's 512x512 center tile; TOP band = first third
        // of the rows (topPx ~= 512*170 = 87040). RAW dump, no single-sentinel
        // guess: the genuine miss value is SURFACE_INDEX_INVALID=0x1FFFFF (NOT
        // 0xFFFF). In a BLACK frame (primaryMiss top=100%): band ~all 0x1FFFFF =>
        // TRUE geometric miss (trims absent from / mis-placed in the traversed
        // TLAS — see [TlasMember]); small valid indices => rays HIT a surface
        // that shades black (shading/material). rawSamples[] are for eyeballing.
        {
          constexpr uint32_t kTile = 512u;
          const uint32_t tileW = std::min(kTile, w);
          const uint32_t tileH = std::min(kTile, h);
          if (tileW > 0u && tileH > 0u) {
            const uint32_t offX  = (w - tileW) / 2u;
            const uint32_t offY  = (h - tileH) / 2u;
            const uint32_t bandH = std::max(1u, tileH / 3u);  // top band rows [offY, offY+bandH)
            uint32_t topPx = 0u;
            uint32_t nInvalidReal = 0u;   // == SURFACE_INDEX_INVALID (0x1FFFFF)
            uint32_t nInvalidFF = 0u;     // == 0xFFFF or 0xFFFFFFFF
            uint32_t nValid = 0u;         // anything else (a real hit surface)
            uint32_t vMin = 0xFFFFFFFFu, vMax = 0u, vSample = 0xFFFFFFFFu;
            for (uint32_t yy = 0u; yy < bandH; ++yy) {
              const uint32_t y = offY + yy;
              for (uint32_t xx = 0u; xx < tileW; ++xx) {
                const uint32_t si = p[y * w + (offX + xx)];
                ++topPx;
                if (si == SURFACE_INDEX_INVALID) {
                  ++nInvalidReal;
                } else if (si == 0xFFFFu || si == 0xFFFFFFFFu) {
                  ++nInvalidFF;
                } else {
                  ++nValid;
                  if (si < vMin) vMin = si;
                  if (si > vMax) vMax = si;
                  vSample = si;
                }
              }
            }
            // 4 raw samples at fixed band positions (corners + center) to eyeball.
            const uint32_t s0 = p[(offY)              * w + (offX)];
            const uint32_t s1 = p[(offY)              * w + (offX + tileW - 1u)];
            const uint32_t s2 = p[(offY + bandH / 2u) * w + (offX + tileW / 2u)];
            const uint32_t s3 = p[(offY + bandH - 1u) * w + (offX + tileW - 1u)];
            Logger::info(str::format(
              "[SkyTrace.topHitSurf] frame=", frameIdx,
              " topPx=", topPx,
              " invReal(0x1FFFFF)=", nInvalidReal,
              " invFF=", nInvalidFF,
              " valid=", nValid,
              " validRange=[", (nValid ? vMin : 0u), ",", vMax, "]",
              " validSample=", (nValid ? vSample : 0xFFFFFFFFu),
              " rawSamples=[", s0, ",", s1, ",", s2, ",", s3, "]"));
          }
        }
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

    // NV-DXVK [SkyTrace.topHitWorld]: also read back the per-pixel WORLD hit
    // position (R32G32B32A32_SFLOAT, xyz = world pos) for the SAME tile. For the
    // non-miss TOP-band pixels this reports WHERE the upward rays actually land
    // in world space. primaryMiss tells us *how many* top-band rays hit; this
    // tells us *what* they hit. In a frame where the top band hits geometry
    // (e.g. the "two views" VIEW1 at 0% top miss) it names the world AABB of the
    // painted sky the player sees; cross-referenced against [TlasCensus] per-VS
    // world AABBs it identifies the VS — without dereferencing any GPU surface
    // index/material (which races streaming/GC, see getImageHash UAF). The
    // world-position gbuffer is a stable read at composite time.
    Rc<DxvkImage> worldImg =
      rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().image(
        Resources::AccessType::Read, /*isAccessedByGPU=*/ false);
    Rc<DxvkBuffer> worldReadbackDst;
    bool haveWorld = false;
    if (worldImg != nullptr
        && worldImg->info().format == VK_FORMAT_R32G32B32A32_SFLOAT
        && worldImg->info().extent.width  == pvzExtent.width
        && worldImg->info().extent.height == pvzExtent.height) {
      DxvkBufferCreateInfo wbi = bufInfo;
      wbi.size = VkDeviceSize(16) * tileW * tileH;  // RGBA32F = 16 bytes/texel
      worldReadbackDst = m_device->createBuffer(
        wbi, memType, DxvkMemoryStats::Category::RTXBuffer,
        "Primary Top Hit World Readback");
      haveWorld = (worldReadbackDst != nullptr);
    }

    // NV-DXVK [SkyTrace.topHitSurf]: also read the per-pixel surface index
    // (R32_UINT, BINDING_INDEX_INVALID=0xFFFF on miss). In a BLACK frame this
    // distinguishes a TRUE geometric miss (top band all 0xFFFF — the trim isn't
    // in the traversed TLAS / positioned wrong) from a valid-surface-shaded-black
    // (valid small indices — a shading/material problem, not geometry). In a SKY
    // frame it names the dominant hit surface index (cross-ref to a VS).
    Rc<DxvkImage> surfImg =
      rtOutput.m_sharedSurfaceIndex.image(Resources::AccessType::Read, /*isAccessedByGPU=*/ false);
    Rc<DxvkBuffer> surfReadbackDst;
    bool haveSurf = false;
    if (surfImg != nullptr
        && surfImg->info().format == VK_FORMAT_R32_UINT
        && surfImg->info().extent.width  == pvzExtent.width
        && surfImg->info().extent.height == pvzExtent.height) {
      DxvkBufferCreateInfo sbi = bufInfo;
      sbi.size = VkDeviceSize(4) * tileW * tileH;  // R32_UINT = 4 bytes/texel
      surfReadbackDst = m_device->createBuffer(
        sbi, memType, DxvkMemoryStats::Category::RTXBuffer,
        "Primary Top Hit Surface Readback");
      haveSurf = (surfReadbackDst != nullptr);
    }

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

    // Same tile, world-position gbuffer (covered by the same pre-copy global
    // memory barrier above — it flushes all COMPUTE_SHADER SHADER_WRITEs).
    if (haveWorld) {
      copyImageToBuffer(
        worldReadbackDst, /*dstOffset=*/ 0,
        16u,                          // dstRowAlignment (RGBA32F texel)
        16u * tileW,                  // dstSliceAlignment
        worldImg, subres,
        VkOffset3D { offX, offY, 0 },
        VkExtent3D { tileW, tileH, 1u });
    }

    // Same tile, surface-index gbuffer (same pre-copy barrier covers it).
    if (haveSurf) {
      copyImageToBuffer(
        surfReadbackDst, /*dstOffset=*/ 0,
        4u,                           // dstRowAlignment (R32_UINT texel)
        4u * tileW,                   // dstSliceAlignment
        surfImg, subres,
        VkOffset3D { offX, offY, 0 },
        VkExtent3D { tileW, tileH, 1u });
    }

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

    // Primary-ray ORIGIN: the main camera world position this frame is the
    // origin the gbuffer pass shot primary rays from. Logging it next to the
    // top-band hit lets us correlate ray-origin vs hit/miss directly — the
    // "two views" flip is a jump in this origin's z (in-scene sub-view camera
    // ~ -1.5k vs the far engine main camera ~ -15.6k) with NO change in look
    // direction or geometry. Captured on the owning thread at composite time.
    const Vector3 cCamPos = getSceneManager().getCamera().getPosition(false);
    const float cCamX = cCamPos.x, cCamY = cCamPos.y, cCamZ = cCamPos.z;

    m_primaryMissCountReadback.asyncTasks.push_back(std::async(std::launch::async,
      [cReadbackDst = std::move(readbackDst),
       cWorldDst = std::move(worldReadbackDst), cHaveWorld = haveWorld,
       cSurfDst = std::move(surfReadbackDst), cHaveSurf = haveSurf,
       syncValue, signalRef,
       frameIdx, cTileW, cTileH, cMissZ, cExtentW, cExtentH,
       cCamX, cCamY, cCamZ]() mutable {
        signalRef->wait(syncValue);
        const float* base = reinterpret_cast<const float*>(cReadbackDst->mapPtr(0));
        if (base == nullptr) {
          return;
        }
        const float* wbase = (cHaveWorld && cWorldDst != nullptr)
          ? reinterpret_cast<const float*>(cWorldDst->mapPtr(0))
          : nullptr;
        const uint32_t* sbase = (cHaveSurf && cSurfDst != nullptr)
          ? reinterpret_cast<const uint32_t*>(cSurfDst->mapPtr(0))
          : nullptr;
        // Top-band surface-index accumulators: count invalid (miss) vs valid,
        // track the valid range + a representative (modal-ish via last-seen).
        uint32_t tsInvalid = 0, tsValid = 0;
        uint32_t tsMin = 0xFFFFFFFFu, tsMax = 0, tsSample = 0xFFFFFFFFu;
        // Top-band world-hit accumulators (non-miss, finite pixels only).
        float twXmin = +INFINITY, twXmax = -INFINITY;
        float twYmin = +INFINITY, twYmax = -INFINITY;
        float twZmin = +INFINITY, twZmax = -INFINITY;
        double twXsum = 0.0, twYsum = 0.0, twZsum = 0.0;
        uint32_t twCount = 0;
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
              // World position of what this top-band ray landed on (non-miss).
              if (wbase != nullptr && !isMiss && std::isfinite(z)) {
                const uint32_t wi = (y * cTileW + x) * 4u;
                const float wx = wbase[wi + 0];
                const float wy = wbase[wi + 1];
                const float wz = wbase[wi + 2];
                if (std::isfinite(wx) && std::isfinite(wy) && std::isfinite(wz)) {
                  if (wx < twXmin) twXmin = wx;
                  if (wx > twXmax) twXmax = wx;
                  if (wy < twYmin) twYmin = wy;
                  if (wy > twYmax) twYmax = wy;
                  if (wz < twZmin) twZmin = wz;
                  if (wz > twZmax) twZmax = wz;
                  twXsum += wx; twYsum += wy; twZsum += wz;
                  ++twCount;
                }
              }
              // Surface index of this top-band pixel. Genuine miss =
              // SURFACE_INDEX_INVALID (0x1FFFFF); 0xFFFF/0xFFFFFFFF kept too.
              // NOTE: this read is POST-ALIAS garbage (see emit below) — the
              // sentinel set is matched to the pre-alias probe only so the A/B
              // isolates the timing bug, not the classification.
              if (sbase != nullptr) {
                const uint32_t si = sbase[y * cTileW + x];
                if (si == SURFACE_INDEX_INVALID || si == 0xFFFFu || si == 0xFFFFFFFFu) {
                  ++tsInvalid;
                } else {
                  ++tsValid;
                  if (si < tsMin) tsMin = si;
                  if (si > tsMax) tsMax = si;
                  tsSample = si;
                }
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

        // [SkyTrace.topHitWorld]: world AABB of what the TOP band's non-miss
        // rays hit. Compare VIEW1 (sky present) vs VIEW2 (black) at matched
        // camera pitch: if VIEW1 reports a world region here and VIEW2 reports
        // "0 hits", the geometry VIEW1's upward rays land on is what VIEW2 is
        // missing — map the wX/wY/wZ box to a VS via [TlasCensus] world AABBs.
        if (wbase != nullptr) {
          if (twCount > 0) {
            Logger::info(str::format(
              "[SkyTrace.topHitWorld] frame=", frameIdx,
              " rayOrigin=(", cCamX, ",", cCamY, ",", cCamZ, ")",
              " topHits=", twCount, "/", topPx,
              " wX=[", twXmin, ",", twXmax, "]",
              " wY=[", twYmin, ",", twYmax, "]",
              " wZ=[", twZmin, ",", twZmax, "]",
              " wAvg=(", float(twXsum / double(twCount)),
              ",", float(twYsum / double(twCount)),
              ",", float(twZsum / double(twCount)), ")"));
          } else {
            Logger::info(str::format(
              "[SkyTrace.topHitWorld] frame=", frameIdx,
              " rayOrigin=(", cCamX, ",", cCamY, ",", cCamZ, ")",
              " topHits=0/", topPx,
              " (top band all miss — nothing for upward rays to land on)"));
          }
        }

        // [SkyTrace.topHitSurfAliased]: KNOWN-BROKEN control. This reads
        // m_sharedSurfaceIndex at COMPOSITE time, AFTER the disocclusion pass
        // (m_primaryDisocclusionMaskForRR) has aliased its storage — so it does
        // NOT contain primary surface indices and reports validSurf~=all even in
        // pure-miss black frames (confirmed: validSurf=87040 with primaryMiss
        // top=100%). Kept ONLY as the A/B control against the correct pre-alias
        // [SkyTrace.topHitSurf] (emitted from recordVisibleSurfacesReadback). If
        // the two disagree in a black frame, the aliasing is proven and THIS
        // line should be deleted. Do NOT trust this for the §7 decision tree.
        if (sbase != nullptr) {
          Logger::info(str::format(
            "[SkyTrace.topHitSurfAliased] frame=", frameIdx,
            " validSurf=", tsValid, " invalidSurf=", tsInvalid, "/", topPx,
            " validRange=[", (tsValid ? tsMin : 0u), ",", tsMax, "]",
            " sample=", (tsValid ? tsSample : 0xFFFFFFFFu)));
        }
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

  void RtxContext::commitGeometryToRT(const DrawParameters& params, DrawCallState& drawCallState, ShardedDrawInfo* shardInfo){
    ScopedCpuProfileZone();

    // NV-DXVK [Phase2b]: publish the sidecar for the scene-manager consumers
    // (submitDrawState / processDrawCallState read t_shardedConsume). RAII so
    // every early return below (unknown camera, sky SkipSubmit) clears it.
    struct ShardedConsumeScope {
      explicit ShardedConsumeScope(ShardedDrawInfo* info) { t_shardedConsume = info; }
      ~ShardedConsumeScope() { t_shardedConsume = nullptr; }
    } shardedConsumeScope(shardInfo);

    // NV-DXVK [perf, GPU index stash]: record the dynamic-IB stash copy HERE —
    // this lambda replays IN-ORDER on the CS stream, after this draw's bindings
    // and before any later Map(DISCARD) rename replay, so the logical buffer
    // still resolves to the physical slice the rasterized draw consumed. A
    // GPU->GPU copy at this point captures exactly the bytes the draw used,
    // with zero CPU reads of write-combined memory (replaces the per-draw CPU
    // index snapshot — see rtx_types.h indexDataGpuStash). The stash is read
    // back out at prepScene time by cacheIndexDataOnGPU / the kUpdateBVH
    // refresh, behind a transfer->transfer barrier.
    {
      auto& g = drawCallState.geometryData;
      if (g.indexNeedsGpuStash) {
        g.indexNeedsGpuStash = false;
        if (g.indexBuffer.defined() && g.indexBuffer.buffer() != nullptr && g.indexCount > 0) {
          const VkDeviceSize stashLen =
            VkDeviceSize(g.indexCount) * g.indexBuffer.stride();
          // POOLED: this used to createBuffer() per draw — ~600 device-local
          // allocations per frame on the CS thread, each holding the memory
          // allocator lock, each retained for the referencing BlasEntry's
          // lifetime. acquire() serves them from a size-classed free list and
          // the handle's deleter checks the buffer back in, so steady state
          // allocates nothing. See IndexStashPool in rtx_scene_manager.h.
          g.indexDataGpuStash = getSceneManager().getIndexStashPool().acquire(stashLen);
          if (g.indexDataGpuStash != nullptr) {
            copyBuffer(g.indexDataGpuStash->buffer, 0,
                       g.indexBuffer.buffer(),
                       g.indexBuffer.offset() + g.indexBuffer.offsetFromSlice(),
                       stashLen);
          }
        }
      }
    }

    // NV-DXVK [flicker V8: SubmitDraw-ordered geometry capture — consume side].
    // The capture lambda (emitted by SubmitDraw at the draw's own position in
    // the CS stream, so it ran strictly BEFORE this tail-emitted commit) filled
    // gpuCapture with pooled copies of the draw's exact VB/IB windows. Rebind
    // the geometry's source RasterBuffers onto those copies so every bake
    // consumer downstream (cacheIndexDataOnGPU, interleaveGeometry,
    // generateTriangleList) reads bytes that cannot be torn by the engine's
    // next upload. Strides, formats and offsetFromSlice are preserved, and all
    // streams of a group share one DxvkBufferSlice, so the interleaved-layout
    // detection and the fast/slow bake path choice are unchanged.
    // See RasterGeometry::GeometryCapture in rtx_types.h for the full story.
    {
      auto& g = drawCallState.geometryData;
      const auto& cap = g.gpuCapture;
      if (cap != nullptr && cap->valid) {
        if (cap->wantIndex && cap->index != nullptr && g.indexBuffer.defined()) {
          g.indexBuffer = RasterBuffer(
            DxvkBufferSlice(cap->index->buffer), 0,
            g.indexBuffer.stride(), g.indexBuffer.indexType());
        }
        // One shared slice per vertex group keeps matches() true between
        // streams that shared a source slice (isVertexDataInterleaved).
        DxvkBufferSlice groupSlice[RasterGeometry::GeometryCapture::kMaxVertexGroups];
        for (uint32_t i = 0; i < RasterGeometry::GeometryCapture::kMaxVertexGroups; ++i) {
          if (cap->vertex[i] != nullptr) {
            groupSlice[i] = DxvkBufferSlice(cap->vertex[i]->buffer);
          }
        }
        const auto rebindStream = [&](RasterBuffer& rb, uint32_t streamIdx) {
          const uint8_t grp = cap->streamGroup[streamIdx];
          if (grp == RasterGeometry::GeometryCapture::kStreamNotCaptured
              || grp >= RasterGeometry::GeometryCapture::kMaxVertexGroups
              || cap->vertex[grp] == nullptr
              || !rb.defined()) {
            return;
          }
          rb = RasterBuffer(groupSlice[grp], rb.offsetFromSlice(), rb.stride(), rb.vertexFormat());
        };
        rebindStream(g.positionBuffer,  0);
        rebindStream(g.normalBuffer,    1);
        rebindStream(g.texcoordBuffer,  2);
        rebindStream(g.texcoord1Buffer, 3);
        rebindStream(g.color0Buffer,    4);
        g.sourceIsGpuCapture = true;
      }
    }

    // NV-DXVK [perf][CommitRT]: CS-thread cost probe. This runs per-draw on the dxvk
    // CS thread — the consumer that SubmitDraw's EmitCs feeds. The game (d3d11) thread
    // spends ~half its wall time STALLED (wall >> cpuCycles) on a system with CPU
    // headroom, which points to it blocking on THIS thread via command-stream
    // backpressure. Accumulate wall time + draw count + distinct frames, emit per ~3s
    // window. If totalMs/frames ≈ the frame time, the CS thread is the real bottleneck
    // and commitGeometryToRT (not the game-thread injection) is what to optimize.
    static thread_local int64_t  s_commitAccNs   = 0;
    static thread_local int64_t  s_commitFinalizeNs = 0; // finalizePendingFutures (future sync / stall)
    static thread_local int64_t  s_commitSubmitNs   = 0; // submitDrawState (scene geometry build)
    static thread_local uint64_t s_commitCount   = 0;
    static thread_local uint32_t s_commitFrames  = 0;
    static thread_local uint32_t s_commitLastFid = UINT32_MAX;
    static thread_local std::chrono::steady_clock::time_point s_commitLastLog{};
    static thread_local bool s_commitInit = false;
    const auto tCommit0 = std::chrono::steady_clock::now();
    if (!s_commitInit) { s_commitLastLog = tCommit0; s_commitInit = true; }
    {
      const uint32_t fidNow = m_device->getCurrentFrameId();
      if (fidNow != s_commitLastFid) { s_commitLastFid = fidNow; ++s_commitFrames; }
    }
    if (std::chrono::duration_cast<std::chrono::milliseconds>(tCommit0 - s_commitLastLog).count() >= 3000) {
      // perFrameMs split: finalize (waiting on geometry-hash/skinning worker futures)
      // vs submit (SceneManager::submitDrawState — instance + BLAS-input build) vs the
      // remainder (entry diagnostics + camera classification). Tells us whether the CS
      // thread is COMPUTING geometry or STALLED waiting on worker threads.
      const int64_t fr = s_commitFrames ? int64_t(s_commitFrames) : 1;

      // NV-DXVK [Perf.Report]: top of the dxvk-cs per-draw chain. Everything
      // under it ([ProcDCS], [Perf.SceneObj]) is NESTED inside this number and
      // must never be added to it -- see PERF_INSTRUMENTATION_MAP section 2.
      perfreport::publishWindow(perfreport::Slot::CommitRtMs,
        double(s_commitAccNs) / 1.0e6, uint64_t(fr));
      perfreport::publishWindow(perfreport::Slot::CommitSubmitMs,
        double(s_commitSubmitNs) / 1.0e6, uint64_t(fr));
      perfreport::publishWindow(perfreport::Slot::CommitFinalizeMs,
        double(s_commitFinalizeNs) / 1.0e6, uint64_t(fr));
      perfreport::publishWindow(perfreport::Slot::HygDrawsCommitted,
        double(s_commitCount), uint64_t(fr));

      Logger::info(str::format(
        "[CommitRT] window draws=", s_commitCount,
        " frames=", s_commitFrames,
        " perFrameMs=", s_commitAccNs / 1000000 / fr,
        " finalizeMs=", s_commitFinalizeNs / 1000000 / fr,
        " submitMs=", s_commitSubmitNs / 1000000 / fr,
        " otherMs=", (s_commitAccNs - s_commitFinalizeNs - s_commitSubmitNs) / 1000000 / fr,
        " avgUsPerDraw=", (s_commitCount ? (s_commitAccNs / 1000 / int64_t(s_commitCount)) : 0)));
      s_commitLastLog = tCommit0;
      s_commitAccNs = 0; s_commitFinalizeNs = 0; s_commitSubmitNs = 0;
      s_commitCount = 0; s_commitFrames = 0;
    }
    struct CommitTimerGuard {
      std::chrono::steady_clock::time_point t0;
      int64_t& acc; uint64_t& cnt;
      ~CommitTimerGuard() {
        acc += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        ++cnt;
      }
    } commitGuard{ tCommit0, s_commitAccNs, s_commitCount };

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

    // NV-DXVK [BatchSubmitDraw]: with rtx.batchHashes the hashes are pre-computed in the
    // frame-end parallel-for (no future). Accept either a pending future OR already-filled
    // hashes; finalizeGeometryHashes handles both.
    assert(geoData.futureGeometryHashes.valid()
           || geoData.hashes[HashComponents::VertexPosition] != kEmptyHash);
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
    const auto tCommitFin0 = std::chrono::steady_clock::now();
    const bool futuresReady = drawCallState.finalizePendingFutures(lastCamera);
    s_commitFinalizeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - tCommitFin0).count();
    if (futuresReady) {
      // NV-DXVK [Phase2b]: the flush-side pre-pass classified EVERY batched
      // draw's camera in arena order (the exact op order this call would have
      // produced) — re-running it here would double-apply the CameraManager's
      // per-draw state updates. Legacy/unbatched draws still classify here.
      if (shardInfo == nullptr || !shardInfo->cameraDone) {
        drawCallState.cameraType = cameraManager.processCameraData(drawCallState);
      }

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
      //
      // NV-DXVK [PreCombinedGuard]: identity worldToView is NOT sufficient on its
      // own. It means either "world was fused into objectToView" (resolvable) or
      // "nothing was extracted for this draw" (not resolvable). In the second
      // case objectToView is identity too, the product below degenerates to
      // viewToWorld, and its translation is the camera position — so the mesh
      // gets welded to the camera. Require an actual fused transform to un-fuse.
      // See resolvePreCombinedRequiresFusedTransform in rtx_options.h for the
      // measured TF2 case and the viewmodel caveat.
      const bool preCombinedHasFusedTransform =
        !RtxOptions::resolvePreCombinedRequiresFusedTransform()
        || !isIdentityExact(drawCallState.getTransformData().objectToView);
      if (RtxOptions::resolvePreCombinedMatrices() &&
        isIdentityExact(drawCallState.getTransformData().worldToView) &&
        preCombinedHasFusedTransform) {
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

      // NV-DXVK [ResidentScene] slice 6: THE SKIP.
      //
      // WHY IT IS HERE AND NOT ON THE FRAME THREAD, which is where the verdict
      // was reached. Two reasons, and the second is the one that decided it.
      //
      // 1. THIS IS WHERE THE MONEY IS. dxvk-cs is 99.6% of the frame, and of
      //    commitGeometryToRT's ~52 ms/frame, submitDrawState is ~50 -- the
      //    geometry decision, the BLAS input, the material resolution and the
      //    instance resolve. Everything above this line is camera
      //    classification, sky handling and transform fixups, and it is ~2 ms.
      //    Cutting exactly here takes the 50 and keeps the 2.
      //
      // 2. THE RECORD'S EXISTENCE IS THE EVIDENCE, AND ONLY THIS THREAD HAS IT.
      //    A record is built only by a draw that actually RESOLVED TO
      //    INSTANCES. So a draw that commits for a side effect and produces
      //    none -- a sky pass, a fog registration, a terrain bake -- has no
      //    record, cannot be served, and falls through to the full path. The
      //    frame thread cannot make that distinction: it would have to be told,
      //    and telling it means a second shared map between the two threads,
      //    which is the one thing this design does not have. Deciding where the
      //    evidence already is costs nothing and removes the whole class of
      //    "the draw existed for something other than its geometry".
      //
      // So the touch is not a hopeful stamp: it either finds a valid, non-empty
      // record and keeps exactly those instances alive, or it reports a miss and
      // this draw commits in full, which is the behaviour without residency.
      // There is no third outcome and nothing is silently dropped.
      //
      // VERIFY FIRST, SKIP SECOND. While rtx.residentScene.verify is on the
      // prediction is carried, scored in processDrawCallState and DISCARDED --
      // the draw commits in full and NO DRAW IS SKIPPED.
      //
      // THAT IS NOT THE SAME AS "changes nothing on screen", which is what this
      // comment used to claim, and the difference cost a long afternoon. verify
      // disarms THE SKIP, here. It does not disarm THE KEEP: build() runs
      // unconditionally and stamps RtInstance::m_residentKey, and the retention
      // clause in rtx_instance_manager.cpp is guarded by
      // ResidentScene::enable() ALONE -- see the note there, which chose that
      // deliberately. So with enable=True and verify=True, instance LIFETIME is
      // already a function of the residency key, and changing how that key is
      // composed changes which instances are held alive and therefore what is
      // on screen. Every key experiment run under verify is a live change, not
      // a dry run. Read holdsInstance() before assuming otherwise.
      //
      // Nothing is skipped until [ResidentScene] reads FAIL=0 with the record
      // count plateaued across a pitch-and-yaw sweep. That ordering is what the
      // [PropIdKeepLong attempt reverted] note exists to enforce: a long keep on
      // an unstable identity made things measurably WORSE, not merely no better.
      if (drawCallState.residentPredictHit
          && drawCallState.residentKey != 0ull
          && RtxOptions::ResidentScene::enable()
          && !RtxOptions::ResidentScene::verify()
          && getSceneManager().touchResidentRecord(drawCallState.residentKey,
                                                   m_device->getCurrentFrameId())) {
        return;
      }

      const auto tCommitSub0 = std::chrono::steady_clock::now();
      getSceneManager().submitDrawState(this, drawCallState, overrideMaterialData);
      s_commitSubmitNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - tCommitSub0).count();
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
    // NV-DXVK [Perf.GbStop]: ablation ladder, see raytrace_args.h. Zero at
    // default, so the shipped path is bit-identical when unused.
    constants.perfGbStopAfter = RtxOptions::perfGbStopAfter();
    // NV-DXVK [Perf.SubLadder]: sub-stage cuts inside the unordered resolve and
    // the opaque material evaluation. All zero at default.
    constants.perfUnorderedStopAfter = RtxOptions::perfUnorderedStopAfter();
    constants.perfSkipPom = RtxOptions::perfSkipPom() ? 1u : 0u;
    constants.perfSkipMaterialTextures = RtxOptions::perfSkipMaterialTextures() ? 1u : 0u;
    constants.perfSkipThinFilm = RtxOptions::perfSkipThinFilm() ? 1u : 0u;
    constants.perfUnorderedStepCensus = RtxOptions::perfUnorderedStepCensus() ? 1u : 0u;
    // NV-DXVK [ResolveCensus]: not part of updatePerfSweep - the sweep zeroes
    // every knob it owns on its baseline steps, and this is a correctness probe
    // rather than a timing rung, so it must survive a sweep unchanged.
    // NV-DXVK [IdentProbe]: this flag doubles as the SUBMISSION FRAME ID
    // (frameId + 1) so GPU-side probes can stamp which frame's tables they
    // consumed. Every consumer only tests it against zero, so "nonzero =
    // enabled" is preserved. cb.frameIdx cannot serve here — it is forced to 0
    // when rtx.rngSeedWithFrameIndex is off.
    constants.enableResolveCensus = RtxOptions::logResolveCensus()
      ? (m_device->getCurrentFrameId() + 1u) : 0u;
    constants.perfMaterialStopAfter = RtxOptions::perfMaterialStopAfter();
    // NV-DXVK [Perf.GeomFetch]: deliberately NOT driven by updatePerfSweep. The
    // sweep table forces every knob it owns to zero on its baseline steps, and
    // adding a rung there would need a table revision; leaving it out means all
    // three sweep steps run with whatever rung rtx.conf sets, so the SUMMARY
    // "baselines: mean=" is that rung's number with a proper bracket around it.
    constants.perfSkipGeometryFetch = RtxOptions::perfSkipGeometryFetch();
    constants.perfSurfaceInteractionStopAfter = RtxOptions::perfSurfaceInteractionStopAfter();
    constants.perfShaderClock = RtxOptions::perfShaderClock() ? 1u : 0u;
    // NV-DXVK [Perf.CoverageGate]: the coverage atomics now run only when the
    // readback that consumes them is enabled. Previously unconditional, which
    // cost ~104 ms/frame of atomic serialisation on every primary hit and was
    // the entire reason gb_primaryRays sat at 131 ms. See perfCoverageWrites in
    // raytrace_args.h. This is a ~104 ms switch, not a logging switch: never
    // leave it True for a perf measurement, and never enable it to satisfy the
    // sweep's census step (that contaminates every other step in the run).
    constants.perfCoverageWrites = RtxOptions::logSurfaceCoverage() ? 1u : 0u;
    // NV-DXVK [Perf.Sweep]: when the auto-sweep is running it OVERRIDES the six
    // values just assigned plus perfGbStopAfter, which is set further up in this
    // same function. Done here, at the point of upload, so the values the
    // sweep reports are provably the values the shader ran with that frame -
    // there is no second path that could disagree. No-op when disabled.
    // Assigned BEFORE updatePerfSweep, not after. It used to sit below the call;
    // once the sweep started driving it, leaving it there would have overwritten
    // the sweep's value with the option every frame and the cheapTexGradients
    // step would have silently measured baseline. Anything the sweep drives has
    // to be initialised above this call.
    constants.perfCheapTextureGradients = RtxOptions::perfCheapTextureGradients() ? 1u : 0u;
    constants.perfCoherentUnorderedFetch = RtxOptions::perfCoherentUnorderedFetch();
    updatePerfSweep(constants.perfUnorderedStopAfter,
                    constants.perfSkipPom,
                    constants.perfSkipMaterialTextures,
                    constants.perfSkipThinFilm,
                    constants.perfUnorderedStepCensus,
                    constants.perfMaterialStopAfter,
                    constants.perfGbStopAfter,
                    constants.perfCheapTextureGradients,
                    constants.perfCoherentUnorderedFetch);
    // NV-DXVK [NsysAuto]: same call site as the sweep - once per frame, on the
    // frame-constants path, so its gameplay clock advances in lockstep with the
    // frames a capture would actually cover. Drives nothing in `constants`; it
    // only prints the markers the wrapper script triggers on. No-op when
    // rtx.nsysAutoCapture is False.
    updateNsysAutoCapture();
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
    //
    // NV-DXVK 2026-07-31: UN-THROTTLED and now carries frame=.
    //
    // It logged only "bind#N", which cannot be joined against anything —
    // not the build calls, not [Coverage]'s per-frame pixel counts. And the
    // 1-in-30 sampling is blind to a 2-frame alternation, which is what the
    // Opaque swap at rtx_accel_manager.cpp:7609 produces by construction.
    //
    // This is the last untested handoff. With the camera frozen for 784
    // frames the geometry alternates between ~55000 pixels and ZERO while the
    // instance buffer feeding the TLAS build is verified correct and complete
    // on every frame. Everything measured so far reads that buffer; nothing
    // has checked whether the AS actually traced is the one just built from
    // it. Pair this line with [TlasBuildCall] on the same frame: if the
    // opaqueTlas bound here is not the dstAS built this frame, the ray tracer
    // is traversing a stale acceleration structure and that is the flicker.
    {
      static uint32_t s_tlasBindN = 0;
      const uint32_t n = s_tlasBindN++;
      if (RtxOptions::logGeomDiag()) {
        Logger::info(str::format(
          "[SpawnGeomDiag.TlasBindAtFrame] frame=", m_device->getCurrentFrameId(),
          " bind#", n,
          " opaqueTlas=0x", std::hex, reinterpret_cast<uintptr_t>(opaqueTlas.ptr()), std::dec,
          " prevTlas=0x", std::hex, reinterpret_cast<uintptr_t>(prevTlas.ptr()), std::dec,
          " unorderedTlas=0x", std::hex, reinterpret_cast<uintptr_t>(unorderedTlas.ptr()), std::dec,
          " sssTlas=0x", std::hex, reinterpret_cast<uintptr_t>(sssTlas.ptr()), std::dec,
          " prevEqualsCurrent=", (prevTlas.ptr() == opaqueTlas.ptr() ? 1 : 0),
          " unorderedFellBackToOpaque=", (unorderedTlas.ptr() == nullptr ? 1 : 0),
          " sssFellBackToOpaque=", (sssTlas.ptr() == nullptr ? 1 : 0)));
      }
    }

    // NV-DXVK [TlasBind]: does the ray tracer bind the acceleration structure
    // that was built THIS frame?
    //
    // This is the one layer the census chain never reached. [ResolveCensus] +
    // [PIWrite] + the BLAS fields proved every input to the TLAS build is
    // identical between frames where geometry renders and frames where it
    // vanishes: same surface count, same live instance masks, same positions,
    // same non-empty BLASes, all touched this frame. If the inputs are equal
    // and the output differs, the remaining possibility is that the structure
    // being traversed is not the one those inputs produced.
    //
    // Emitted with frame= and gated on the same option as the census so the two
    // lines land on the same frame and need no hand-joining — unlike
    // [SpawnGeomDiag.TlasBindAtFrame]/[TlasBuildCall] above, which are two tags
    // under a different option that a reader has to correlate manually.
    //
    // Reading it:
    //   builtFrame == frame && boundObj == builtObj  -> bind is correct; the
    //       flicker is not stale-TLAS and a frame capture is the honest next
    //       step, since every CPU- and shader-visible layer now reads clean.
    //   builtFrame <  frame                          -> no TLAS build happened
    //       this frame; the ray traverses the previous structure.
    //   boundObj  != builtObj                        -> the bound object is not
    //       the one built into. For Opaque that is exactly the swap at
    //       rtx_accel_manager buildTlas landing out of phase with the bind.
    if (RtxOptions::logResolveCensus()) {
      const auto& rec = getSceneManager().getAccelManager().getTlasBuildRecord(Tlas::Opaque);
      const uint32_t curFrame = m_device->getCurrentFrameId();
      const uint64_t boundObj = reinterpret_cast<uint64_t>(opaqueTlas.ptr());
      // This function runs once per RT pass, so ~8 identical lines per frame
      // without a filter. Emit the first bind of each frame plus EVERY bind
      // that disagrees with the build record.
      //
      // Content-based, not stride-based: a mismatch can never be filtered out,
      // and a pass that binds differently from its predecessors in the same
      // frame still logs. That distinction matters here - handoff §2 is that an
      // even-stride sampler sits at one phase of a single-frame alternation and
      // reports it clean, which is exactly what a "1 in N" filter would do to
      // this signal.
      const bool mismatch =
        (boundObj != rec.tlasObj) || (rec.builtFrame != curFrame);
      static uint32_t s_tlasBindLastFrame = UINT32_MAX;
      const bool firstOfFrame = (s_tlasBindLastFrame != curFrame);
      s_tlasBindLastFrame = curFrame;
      if (firstOfFrame || mismatch) {
        Logger::info(str::format(
          "[TlasBind] f=", curFrame,
          " builtFrame=", rec.builtFrame,
          " buildAge=", (rec.builtFrame == kInvalidFrameIndex ? -1 : int32_t(curFrame - rec.builtFrame)),
          " boundObj=", str::format("0x", std::hex, boundObj, std::dec),
          " builtObj=", str::format("0x", std::hex, rec.tlasObj, std::dec),
          " builtDstAS=", str::format("0x", std::hex, rec.dstHandle, std::dec),
          " objMatch=", (boundObj == rec.tlasObj ? 1 : 0),
          " builtInstances=", rec.numInstances,
          " prevObj=", str::format("0x", std::hex, reinterpret_cast<uint64_t>(prevTlas.ptr()), std::dec),
          " boundIsPrev=", (opaqueTlas.ptr() == prevTlas.ptr() ? 1 : 0),
          " firstOfFrame=", (firstOfFrame ? 1 : 0),
          " mismatch=", (mismatch ? 1 : 0)));
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

    // NV-DXVK [Perf.ShaderClock]: bound unconditionally like the coverage buffer,
    // so the descriptor layout is the same whether the counters are armed or not.
    // Writes are gated in-shader by cb.perfShaderClock.
    Rc<DxvkBuffer> shaderClockBuffer = getResourceManager().getRaytracingOutput().m_shaderClockBuffer;
    bindResourceBuffer(BINDING_SHADER_CLOCK_BUFFER, DxvkBufferSlice(shaderClockBuffer, 0, shaderClockBuffer.ptr() ? shaderClockBuffer->info().size : 0));

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

  // NV-DXVK [perf]: see markGpuStage in rtx_context.h. The sub-marks below split
  // this dispatch into its five stages so [Perf.GpuPass] can say which one owns
  // the ~150 ms; the outer pathTracing entry then reads near zero by design,
  // since these cover it.
  void RtxContext::markGpuStageBeforeNextDispatch() {
    m_gpuMarkNextDispatch = true;
  }

  void RtxContext::markGpuStageIfPending() {
    if (m_gpuMarkNextDispatch) {
      m_gpuMarkNextDispatch = false;
      markGpuStage();
    }
  }

  // NV-DXVK [perf]: GPU time per RT pass.
  //
  // The [Perf.Frame] stage timers are CPU wall time around a *dispatch call* —
  // they say how long it took to RECORD the work, not to execute it, and they are
  // all ~1 ms while [Perf.Gpu] puts the GPU at ~350 ms/frame with idle near zero.
  // So the frame is GPU-bound and none of the CPU counters can name the owner.
  // These timestamps close that gap: written into the command stream at the same
  // boundaries as markStage, so [Perf.GpuPass] reads field-for-field against
  // [Perf.Frame].
  //
  // Deliberately reuses DxvkDLFGTimestampQueryPool rather than adding a second
  // pool implementation — it already does reset-then-write per slot and a
  // non-blocking read, and is proven in the DLFG path. Results are read back
  // kFrames later so the GPU has certainly passed them; a not-ready slot just
  // skips the sample rather than stalling.
  void RtxContext::beginGpuStageFrame() {
    auto& gpuStages = m_gpuStageTimers;

    if (gpuStages.pool == nullptr) {
      gpuStages.pool = new DxvkDLFGTimestampQueryPool(
        m_device.ptr(), GpuStageTimers::kSlots * GpuStageTimers::kFrames);
    }

    gpuStages.frameSlot  = (gpuStages.frameSlot + 1u) % GpuStageTimers::kFrames;
    gpuStages.writeCount = 0;
    // stageCount is otherwise only ever written by markGpuStage, so a frame that
    // wrote no marks at all would leave a stale count pointing at slot indices
    // from three frames ago and resolve them as if they were this frame's.
    gpuStages.stageCount[gpuStages.frameSlot] = 0;

    // Resolve the oldest frame in the ring before this frame overwrites its own
    // slot. kFrames back is far enough that the GPU has long passed it; if it
    // somehow has not, readTimestamp fails and the sample is dropped.
    const uint32_t readSlot = (gpuStages.frameSlot + 1u) % GpuStageTimers::kFrames;
    const uint32_t n = gpuStages.stageCount[readSlot];

    bool resolvedThisFrame = false;

    if (n >= 2u) {
      const double nsPerTick = m_device->adapter()->deviceProperties().limits.timestampPeriod;

      uint64_t ts[GpuStageTimers::kSlots] = {};
      bool ok = true;
      for (uint32_t i = 0; i < n && ok; ++i)
        ok = gpuStages.pool->readTimestamp(&ts[i], gpuStages.slotIndex[readSlot][i]);

      if (ok) {
        // NV-DXVK [perf]: record the mark count for the alignment check reported
        // on the [Perf.GpuPass] line. Only for frames that actually resolved, so
        // a failed readback cannot masquerade as a short frame.
        if (n < gpuStages.marksMin) gpuStages.marksMin = n;
        if (n > gpuStages.marksMax) gpuStages.marksMax = n;

        for (uint32_t i = 1; i < n; ++i) {
          // Timestamps are monotonic within a queue; guard anyway so a wrap or a
          // reset race can't poison the accumulator.
          const uint64_t d = (ts[i] >= ts[i - 1]) ? (ts[i] - ts[i - 1]) : 0ull;
          const double stageMs = double(d) * nsPerTick / 1.0e6;
          gpuStages.accumMs[i] += stageMs;
          ++gpuStages.samplesAt[i];

          // NV-DXVK [perf]: did a submit land inside this stage's interval?
          if (gpuStages.markCmdBuf[readSlot][i] != gpuStages.markCmdBuf[readSlot][i - 1])
            ++gpuStages.accumSubmitSplit[i];

          // NV-DXVK [perf]: the emission site that actually closed this interval.
          gpuStages.lastMarkLine[i] = gpuStages.markLine[readSlot][i];

          // NV-DXVK [Perf.Sweep]: feed gb_primaryRays into the auto-sweep. Slot 8
          // because accumMs[i] is named by kStageNames[i - 1], and gb_primaryRays
          // is kStageNames[7]. Asserted against the name below rather than left as
          // a bare literal, since reordering kStageNames would otherwise silently
          // make the sweep measure a different pass.
          // Was 7; became 8 when "gpuDrain" was inserted after "prepScene".
          constexpr uint32_t kGbPrimaryRaysSlot = 8u;
          if (i == kGbPrimaryRaysSlot)
            perfSweepAddSample(stageMs);
        }
        const uint64_t total = (ts[n - 1] >= ts[0]) ? (ts[n - 1] - ts[0]) : 0ull;
        gpuStages.accumTotalMs += double(total) * nsPerTick / 1.0e6;
        ++gpuStages.samples;

        // Everything the GPU did between the previous frame's last mark and this
        // frame's first one. Resolutions happen once per injectRTX in frame order,
        // so two consecutive successful resolves are two consecutive frames and
        // the gap between them is exactly the un-instrumented GPU work: the game's
        // own raster, present, and any driver-side work between submissions. All
        // stamps are on the same queue, so they are directly comparable.
        if (gpuStages.prevLastValid && ts[0] >= gpuStages.prevLastTs) {
          gpuStages.accumOutsideMs +=
            double(ts[0] - gpuStages.prevLastTs) * nsPerTick / 1.0e6;
          ++gpuStages.outsideSamples;
        }
        gpuStages.prevLastTs    = ts[n - 1];
        gpuStages.prevLastValid = true;
        resolvedThisFrame       = true;
      }
    }

    // A dropped sample breaks the frame-to-frame adjacency the outside-RT gap
    // depends on, so do not carry a stale endpoint across it.
    if (!resolvedThisFrame)
      gpuStages.prevLastValid = false;

    const auto nowLog = std::chrono::steady_clock::now();
    if (gpuStages.samples > 0
     && nowLog - gpuStages.lastLog >= std::chrono::seconds(5)) {
      gpuStages.lastLog = nowLog;

      // Order must match the markGpuStage call sequence exactly.
      //  - texUpload/prepScene are the two pre-RT-branch marks. prepScene is
      //    where the BLAS and TLAS builds are recorded, and it sat outside the
      //    old t0 entirely.
      //  - gb_bindWait is the post-barrier mark inside DxvkContext::dispatch, so
      //    the gb_primaryRays entry that follows it is the dispatch ALONE and
      //    gb_bindWait is whatever it had to wait for.
      //  - the five pt_* entries are sub-marks inside dispatchPathTracing, so the
      //    outer "pathTracing" that follows them reads ~0 by construction. Same
      //    for pt_gbuffer against the gb_* entries. Both are kept to hold the
      //    ordering, not because they carry time.
      // NV-DXVK [Perf.Sweep]: kGbPrimaryRaysSlot in the resolve loop above indexes
      // this table. Keep them in step - the name check below is the guard.
      static constexpr const char* kStageNames[] = {
        "texUpload", "prepScene",
        // NV-DXVK: gpuDrain inserted here — the span between the prepScene mark
        // and the mark just before the onFrameBegin() call, in which no GPU
        // commands are recorded. Inserting it shifted every slot after it, so
        // kGbPrimaryRaysSlot below went 7 -> 8 and the guard's index 6 -> 7.
        "gpuDrain",
        "onFrameBegin", "prep", "volumetrics",
        "gb_bindWait", "gb_primaryRays", "gb_reflectionPSR", "gb_transmissionPSR",
        "pt_gbuffer", "pt_visSurfReadback", "pt_rtxdi", "pt_neeCache", "pt_integrate",
        "pathTracing", "nrc",
        // NV-DXVK: rtxdi_barrierWait splits the ~100 ms interval that owns the
        // frame into "waiting on prior work" vs "the confidence pass's own
        // dispatches". See the insertion point at the dispatchConfidence call.
        "rtxdi_barrierWait",
        "rtxdi", "restir", "demodulate", "denoise", "composite",
        // NV-DXVK: pc_null is a NULL-INTERVAL CONTROL - the span between the two
        // marks at the end of the postComposite block contains no commands at all.
        // See the note at that insertion point. It is after gb_primaryRays, so
        // kGbPrimaryRaysSlot and the strcmp guard's index are both unaffected.
        "debugView", "postComposite", "pc_null", "upscaler", "finalBlit", "endFrame",
      };
      constexpr uint32_t kNumNames = sizeof(kStageNames) / sizeof(kStageNames[0]);

      // NV-DXVK [Perf.Sweep]: the sweep hardcodes accumMs slot 8 as gb_primaryRays
      // (accumMs[i] is named by kStageNames[i - 1]). If this table is ever
      // reordered the sweep would silently start reporting a different pass, which
      // is exactly the class of error this whole investigation kept hitting. Once
      // per log interval, so the cost is nothing.
      // Index was 6; became 7 when "gpuDrain" was inserted after "prepScene".
      // This guard is what makes that edit safe — it fires loudly if the table
      // and kGbPrimaryRaysSlot ever drift apart again.
      if (std::strcmp(kStageNames[7], "gb_primaryRays") != 0) {
        Logger::err(str::format(
          "[Perf.Sweep] STAGE TABLE REORDERED: slot 8 is now '", kStageNames[7],
          "', not 'gb_primaryRays'. Sweep results are measuring the wrong pass - "
          "update kGbPrimaryRaysSlot."));
      }

      std::string line;
      for (uint32_t i = 1; i < GpuStageTimers::kSlots; ++i) {
        if (gpuStages.samplesAt[i] == 0 || (i - 1) >= kNumNames)
          continue;
        // Per-stage divisor: see samplesAt. Print the count alongside any stage
        // that did not run on every sampled frame, so a stage that fires rarely
        // is never mistaken for a stage that is cheap.
        const double ms = gpuStages.accumMs[i] / double(gpuStages.samplesAt[i]);
        const uint32_t splits = gpuStages.accumSubmitSplit[i];
        // A stage that crossed a submit boundary is printed even when it is under
        // the 0.01 ms floor: a boundary on a near-zero stage is exactly as
        // diagnostic as one on a large stage, and hiding it would leave the
        // boundary map with holes.
        if (ms < 0.01 && splits == 0)
          continue;
        // name=ms@line -- 'line' is where the mark closing this interval was
        // actually emitted. If it does not match where kStageNames[i-1] lives in
        // this file, the label is misassigned and the number belongs to another
        // pass. Read the @line, not the name.
        line += str::format(" ", kStageNames[i - 1], "=", ms, "@", gpuStages.lastMarkLine[i]);

        // NV-DXVK [Perf.Report]: matched by NAME, not by slot index. kStageNames
        // maps positionally and that mapping has shifted before -- a slot-index
        // table here would silently relabel every GPU row the next time a mark is
        // added. The report also carries the ALIGNED flag and refuses to trust
        // any of these when it is false.
        {
          const char* nm = kStageNames[i - 1];
          const auto is = [nm](const char* s) { return std::strcmp(nm, s) == 0; };
          if      (is("pt_integrate"))   perfreport::publish(perfreport::Slot::GpuPtIntegrateMs, ms);
          else if (is("gb_primaryRays")) perfreport::publish(perfreport::Slot::GpuPrimaryRaysMs, ms);
          else if (is("prepScene"))      perfreport::publish(perfreport::Slot::GpuPrepSceneMs, ms);
          else if (is("upscaler"))       perfreport::publish(perfreport::Slot::GpuUpscalerMs, ms);
          else if (is("finalBlit"))      perfreport::publish(perfreport::Slot::GpuFinalBlitMs, ms);
          else if (is("pt_neeCache"))    perfreport::publish(perfreport::Slot::GpuNeeCacheMs, ms);
          else if (is("pt_rtxdi"))       perfreport::publish(perfreport::Slot::GpuRtxdiMs, ms);
        }
        if (gpuStages.samplesAt[i] != gpuStages.samples)
          line += str::format("(n=", gpuStages.samplesAt[i], ")");
        // *SUBMIT(k/m): a submit landed inside this interval on k of m resolved
        // frames, so this stage's number is a submission gap, not its own work.
        if (splits > 0)
          line += str::format("*SUBMIT(", splits, "/", gpuStages.samplesAt[i], ")");
      }

      const double outsideMs = gpuStages.outsideSamples > 0
        ? gpuStages.accumOutsideMs / double(gpuStages.outsideSamples)
        : -1.0;

      // NV-DXVK [perf]: expected mark count. accumMs[i] is named kStageNames[i - 1],
      // so covering all kNumNames entries needs i up to kNumNames, i.e. n marks
      // where n = kNumNames + 1 (the +1 is t0, which starts the first delta).
      constexpr uint32_t kExpectedMarks = kNumNames + 1u;
      const bool marksAligned = (gpuStages.marksMin == kExpectedMarks)
                             && (gpuStages.marksMax == kExpectedMarks);

      // NV-DXVK [Perf.Report]: the GPU is the third pole candidate. Only
      // totalMs and the ALIGNED flag go across -- deliberately NOT outsideRtMs,
      // which is GPU-timeline time outside the RT passes and INCLUDES the GPU
      // sitting idle. Adding it to totalMs and matching the frame wall is an
      // identity, and reading that identity as "GPU-bound" is a mistake this
      // repo has made more than once.
      perfreport::publish(perfreport::Slot::GpuPassTotalMs,
        gpuStages.accumTotalMs / double(gpuStages.samples));
      perfreport::publish(perfreport::Slot::GpuAligned, marksAligned ? 1.0 : 0.0);

      Logger::warn(str::format(
        "[Perf.GpuPass] perFrame totalMs=", gpuStages.accumTotalMs / double(gpuStages.samples),
        " outsideRtMs=", outsideMs, "(n=", gpuStages.outsideSamples, ")",
        " samples=", gpuStages.samples,
        " marks=", gpuStages.marksMin, "..", gpuStages.marksMax,
        "/", kExpectedMarks, (marksAligned ? " ALIGNED" : " *SHIFTED*"),
        " |", line));

      // A short or varying mark count means the positional name mapping is wrong
      // at and after the missing mark, so the per-stage labels on the line above
      // cannot be trusted - which is exactly how a heavy pass ends up reported
      // under a neighbour's name. Loud, because reading a shifted line as if it
      // were aligned is how a whole optimisation pass gets aimed at the wrong
      // stage. totalMs is unaffected: it is last-minus-first, not positional.
      if (!marksAligned) {
        Logger::err(str::format(
          "[Perf.GpuPass] STAGE NAMES SHIFTED: marks=", gpuStages.marksMin, "..",
          gpuStages.marksMax, " but kStageNames needs exactly ", kExpectedMarks,
          ". Stage LABELS at/after the missing mark are wrong (each reports its "
          "neighbour's time). totalMs is still valid. Cause is a conditional mark: "
          "the gbuffer sub-marks use markGpuStageIfPending, and the PSR dispatches "
          "that consume it are optional."));
      }

      for (auto& a : gpuStages.accumMs)
        a = 0.0;
      for (auto& s : gpuStages.samplesAt)
        s = 0;
      for (auto& sp : gpuStages.accumSubmitSplit)
        sp = 0;
      gpuStages.accumTotalMs   = 0.0;
      gpuStages.samples        = 0;
      gpuStages.accumOutsideMs = 0.0;
      gpuStages.outsideSamples = 0;
      gpuStages.marksMin       = ~0u;
      gpuStages.marksMax       = 0;
    } else if (gpuStages.lastLog.time_since_epoch().count() == 0) {
      gpuStages.lastLog = nowLog;
    }
  }

  // NV-DXVK [Perf.Sweep]: the step table. Order matters in one respect only —
  // the census must be last, because its per-pixel InterlockedAdds inflate the
  // pass timer and would contaminate every step that followed it.
  //
  // Baseline is first AND the ladder is measured against it within the same run,
  // so a step's number never has to be compared to a figure from another process.
  namespace {
    struct PerfSweepStep {
      const char* name;
      uint32_t    unorderedStopAfter;
      uint32_t    skipPom;
      uint32_t    skipMaterialTextures;
      uint32_t    skipThinFilm;
      uint32_t    stepCensus;
      uint32_t    materialStopAfter;
      // Runtime rung of the top-level stage ladder (rtx.perfGbStopAfter):
      // 0=full, 1=raygen, 2=+traversal, 3=+unordered, 4=+material. This is the
      // coarsest and most important decomposition of the pass and it had never
      // been swept, because the ladder was documented as needing one build per
      // rung. That is true only of the REGISTER counts (REMIX_GBSTOP); the
      // TIMINGS come from this constant-buffer value and cost nothing to walk.
      uint32_t    gbStopAfter;
      // Substitutes the cheap ray-cone texture-gradient path for
      // computeAnisotropicEllipseAxes (rtx.perfCheapTextureGradients).
      //
      // Added for the v3 table. That function is fork-local, runs per pixel per
      // hit, and does four clip-space projections plus a screen-space
      // determinant, rank-1 UV detection and UV/world area ratios - and it is
      // called from surfaceInteractionCreate, which the v2 sweep localised as
      // the +12.99 ms rung1->rung2 step of the unordered stage. It is the
      // largest fork-local ALU block in that stage and it had never been swept.
      //
      // It SUBSTITUTES a path rather than deleting one, which is why it is
      // trustworthy where the uno rungs are not: it cannot collapse the resolve
      // loop's trip count the way a cut that changes `opacity` does, and there
      // is nothing for the compiler to dead-code away.
      //
      // NOT a candidate setting. The expensive path exists because the rank-1
      // fallback fixes real aliasing ("grey grain walls" on TF2 BSP trim). If
      // this measures large the fix is to make that function cheaper while
      // keeping the fallback, not to ship the bypass.
      uint32_t    cheapTextureGradients;
      // Marks a row as a reference measurement rather than a probe. Probes are
      // scored against the mean of the baselines that bracket them, so the
      // summary reports work removed rather than work removed plus drift.
      uint32_t    isBaseline;
      // Quick-mode only: leave every override untouched so the step runs with
      // whatever rtx.perf* values are set in rtx.conf. This is what lets the
      // 3-step table probe ANY hypothesis without a code change - the probe row
      // is literally "whatever the user configured", bracketed by two rows that
      // force everything off.
      //
      // Deliberately the LAST field: the 38 rows of the full table predate it
      // and zero-initialise it to 0 (= override normally), so adding it did not
      // require touching a single one of them.
      uint32_t    passthrough;
      // NV-DXVK [Perf.CoherentFetch]. Appended last for the same reason
      // passthrough was: every existing row zero-initialises it to "off", so
      // adding it required touching none of them. Baseline rows therefore force
      // it off, which is what makes the quick bracket honest when the probe
      // value arrives via passthrough from rtx.conf.
      uint32_t    coherentUnorderedFetch;
    };

    // A-B-A. Every probe is preceded and followed by a baseline row.
    //
    // The old table measured 15 probes against a SINGLE baseline taken at the
    // top of a 3-minute run. That was defensible when the pass was 131 ms and
    // the probes moved it by tens of ms. It is not defensible now. With the
    // coverage atomics gone the pass is ~27 ms, the surviving features are
    // plausibly 1-3 ms each, and the observed frame-to-frame wander is +-3 ms
    // and moves EVERY pass together (pt_integrate swings 16.5 -> 46.4 in the
    // same samples), so it is global - clock, thermal, or scene - not something
    // a per-pass probe can be blamed for. Against one fixed baseline that
    // wander is indistinguishable from the effect being measured, and it is
    // larger than every effect being measured.
    //
    // Bracketing each probe cancels the linear part of that drift: a probe is
    // scored against the mean of its two neighbouring baselines, both taken
    // within ~20 s of it. The cost is roughly double the rows and double the
    // wall time, which is the cheaper half of the trade when the alternative
    // is 19 numbers that cannot be trusted.
    //
    // Order matters in one respect only - the census must be last, because its
    // per-pixel InterlockedAdds inflate the pass timer and would contaminate
    // every step that followed it.
    // v3, 2026-07-25. Retargeted onto the unordered resolve after the v2 run.
    //
    // v2 result (baseline 26.80 ms, floor 2.85 ms):
    //   launch+store 0.44 | traversal 2.02 | UNORDERED 13.43 | material 9.32
    // and the census: 1.206 candidates/pixel, acceptRate 1.0, against a cap of
    // 32. So the unordered stage is not volume-bound - ~2.5M interactions
    // costing 13.4 ms is per-candidate cost, and the uno ladder put +12.99 ms of
    // it in the single rung1->rung2 step (hit info, ray interaction, Surface
    // load, surfaceInteractionCreate).
    //
    // DROPPED from v2: mat_skipPom, mat_skipTextures, mat_skipThinFilm and all
    // four mat_stop* rungs. Every one came back under the noise floor - POM even
    // measured SLOWER at +1.37 - so the material function is ~9.3 ms spread over
    // ~1500 lines with no hotspot. Re-running them would spend six minutes
    // reconfirming a null. mat_skipAllThree is kept as a single canary: if it
    // ever leaves the noise, something changed and the rest can come back.
    //
    // Removing seven probes is what pays for the longer hold below, and step
    // length is the only lever left on the 2.85 ms resolution floor.
    static constexpr PerfSweepStep kPerfSweepSteps[] = {
      // name                     uno  pom  tex  film census mat  gb  grad base
      { "baseline_00",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },

      // Top-level stage ladder. THIS IS THE ONE TO READ FIRST: it splits the
      // pass into raygen / traversal / unordered resolve / material / tail, and
      // every finer probe below is only worth interpreting once you know which
      // of those four owns the time. Cumulative - per-stage cost is the
      // difference between consecutive rungs. Rung 1 is the launch+store floor
      // rather than zero, because every rung still writes the G-buffer.
      //
      // The pre-fix split was 3.0 traversal / 42.5 unordered / 74.5 material of
      // a ~126 ms pass. That is HISTORICAL - it was measured with the coverage
      // atomics still running inside the material function, which is exactly
      // what inflated the material number. Do not carry it forward; this ladder
      // is here to replace it.
      { "gb1_raygen",               0u,  0u,  0u,  0u,  0u,  0u,  1u,  0u,  0u },
      { "baseline_01",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "gb2_traversal",            0u,  0u,  0u,  0u,  0u,  0u,  2u,  0u,  0u },
      { "baseline_02",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "gb3_unordered",            0u,  0u,  0u,  0u,  0u,  0u,  3u,  0u,  0u },
      { "baseline_03",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "gb4_material",             0u,  0u,  0u,  0u,  0u,  0u,  4u,  0u,  0u },
      { "baseline_04",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },

      // THE PROBE THIS TABLE EXISTS FOR. Substitutes the cheap ray-cone
      // gradient path for computeAnisotropicEllipseAxes inside
      // surfaceInteractionCreate - the largest fork-local ALU block in the rung
      // that owns +12.99 ms. Placed early, while the machine is coldest and the
      // bracketing baselines are tightest.
      { "cheapTexGradients",        0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u,  0u },
      { "baseline_05",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },

      // Unordered resolve sub-ladder: which part of the candidate body owns the
      // unordered stage. Traversal runs in all of them, so rung 1 is the floor.
      // Cumulative.
      //
      // CAVEAT carried from v2, do not lose it: uno1 measured 10.69 ms, which is
      // LOWER than gb3_unordered at 15.89 despite having strictly more stages
      // enabled. Cutting the candidate body changes `opacity`, which drives
      // `continueResolving`, which collapses the ORDERED resolve loop's trip
      // count too. So uno1 is not "traversal costs 10.7" and the uno deltas
      // include relocated work. Consecutive differences are still the right
      // read; absolute values are not.
      { "uno1_traversalOnly",       1u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_06",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "uno2_surfaceInteraction",  2u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_07",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "uno3_clipTest",            3u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_08",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "uno4_materialInteraction", 4u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_09",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "uno5_approxAndDecals",     5u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_10",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },
      { "uno6_blend",               6u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_11",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },

      // Material canary. The only survivor of v2's seven material probes, all of
      // which came back under the floor. If this one ever moves, the material
      // function changed and the dropped rungs are worth restoring.
      { "mat_skipAllThree",         0u,  1u,  1u,  1u,  0u,  0u,  0u,  0u,  0u },
      { "baseline_12",              0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  1u },

      // Raw counts. Timing for this step is meaningless by construction; read
      // [Perf.UnorderedSteps] from it, not the ms column. Last, deliberately.
      //
      // This step is INERT during a normal sweep and that is on purpose. It
      // needs rtx.logSurfaceCoverage=True to print anything, but that option is
      // now the gate on the 52-atomic coverage block (~104 ms/frame), and it is
      // a global switch - enabling it for this one step re-arms those atomics
      // for all 38 steps and invalidates every number in the table. The census
      // and the timing sweep cannot share a run. If you want the counts, do a
      // separate pass with perfAutoSweep off, coverage on, and read
      // [Perf.UnorderedSteps] directly. Do NOT "fix" this by turning coverage on
      // in rtx.conf and re-running the sweep.
      { "census_rawCounts",         0u,  0u,  0u,  0u,  1u,  0u,  0u,  0u,  0u },
    };

    static constexpr uint32_t kPerfSweepStepCount =
      sizeof(kPerfSweepSteps) / sizeof(kPerfSweepSteps[0]);

    // NV-DXVK [Perf.Sweep] QUICK table: baseline, probe, baseline. ~45 s.
    //
    // Exists because the full table is a six-minute GPU soak and it was being
    // run to answer ONE question. Two of those runs overheated the machine into
    // a hard crash, and the second of them tested a single hypothesis
    // (cheapTexGradients) that a 45-second bracket would have refuted just as
    // conclusively. Thirteen probes is the right shape for "where is the time";
    // it is the wrong shape for "is it this one thing", which is every question
    // from here on.
    //
    // The probe row is passthrough, so it runs with whatever rtx.perf* knobs are
    // set in rtx.conf. That means testing a new hypothesis needs no code change
    // and no new table row - set the knob, run, read the delta. The two baseline
    // rows force every override off, so the bracket is honest regardless of what
    // was left set in the config.
    //
    // Same A-B-A scoring and the same RESOLUTION FLOOR line as the full table;
    // with only two baselines the floor is the gap between them, which is a
    // cruder but still honest error bar.
    static constexpr PerfSweepStep kPerfSweepQuickSteps[] = {
      // name                  uno pom tex film cen mat gb grad base pass
      { "quick_baseline_pre",   0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,  1u,  0u },
      { "quick_probe",          0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,  0u,  1u },
      { "quick_baseline_post",  0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,  1u,  0u },
    };

    static constexpr uint32_t kPerfSweepQuickStepCount =
      sizeof(kPerfSweepQuickSteps) / sizeof(kPerfSweepQuickSteps[0]);
  }

  // NV-DXVK [NsysAuto]: the NVTX range name that triggers the capture.
  //
  // MUST match the -p / --nvtx-capture argument in
  // "Capture TF2 (Nsight Systems auto).ps1". If these two ever disagree, nsys
  // waits forever for a range that never arrives and the run produces no report
  // at all - so change them together.
  static constexpr const char* kNsysCaptureRangeName = "RemixCapture";

  // NV-DXVK [NsysAuto]: unattended Nsight Systems capture director.
  //
  // Owns the TIMING of the capture: it opens an NVTX range after
  // nsysAutoCaptureSettleSeconds of real gameplay and closes it
  // nsysAutoCaptureSeconds later. With '--capture-range=nvtx' nsys collects for
  // exactly that window. The log markers are for progress reporting only - they
  // no longer drive anything.
  void RtxContext::updateNsysAutoCapture() {
    NsysAutoCapture& cap = m_nsysAuto;

    if (!RtxOptions::nsysAutoCapture()) {
      // Toggling the option off re-arms the director, so a second capture can be
      // taken in the same process without a restart - same behaviour as the sweep.
      if (cap.active) {
        cap = NsysAutoCapture();
      }
      return;
    }

    if (cap.phase == NsysAutoCapture::Phase::Done) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();

    // Gameplay gate, identical to [Perf.Sweep]: a non-empty ordered-instance list
    // means a world is loaded, plus a warmup because the frame it first goes
    // non-empty has a TLAS rebuild, a bucket reorder and a BLAS attach all landing
    // together - capturing that would profile level load, not gameplay.
    constexpr uint32_t kGameplayWarmupFrames = 30u;

    const bool inGameplay =
      !getSceneManager().getAccelManager().getOrderedInstances().empty();

    if (inGameplay) {
      if (cap.gameplayFrames < kGameplayWarmupFrames) {
        ++cap.gameplayFrames;
      }
    } else {
      cap.gameplayFrames = 0;
    }

    cap.gameplayReady = (cap.gameplayFrames >= kGameplayWarmupFrames);

    if (!cap.active) {
      cap.active = true;
      cap.lastTick = now;
      return;  // no usable delta on the first tick
    }

    // Clamped so one hitch cannot swallow a phase. A shader-compile stall or an
    // alt-tab can produce a multi-second frame; without the clamp a single such
    // frame could step straight past CAPTURE-BEGIN and CAPTURE-END, and the run
    // would produce a report covering nothing.
    const double rawDelta = std::chrono::duration<double>(now - cap.lastTick).count();
    const double delta = std::min(rawDelta, 0.5);
    cap.lastTick = now;

    // The clock FREEZES outside gameplay rather than resetting, so pausing or
    // alt-tabbing mid-settle costs wall time but does not restart the count.
    if (!cap.gameplayReady) {
      if (!cap.armedLogged) {
        cap.armedLogged = true;
        Logger::warn(str::format(
          "[NsysAuto] ARMED - waiting for gameplay (need ", kGameplayWarmupFrames,
          " consecutive frames with world instances)."
          " settleSeconds=", RtxOptions::nsysAutoCaptureSettleSeconds(),
          " captureSeconds=", RtxOptions::nsysAutoCaptureSeconds(),
          " - load into a level and the capture triggers on its own."));
      }
      return;
    }

    switch (cap.phase) {
      case NsysAutoCapture::Phase::Settle: {
        cap.settleSeconds += delta;

        // Heartbeat so a run that looks idle is distinguishable from one that is
        // correctly counting down.
        if (cap.settleSeconds - cap.lastHeartbeat >= 5.0) {
          cap.lastHeartbeat = cap.settleSeconds;
          Logger::warn(str::format(
            "[NsysAuto] SETTLE t=", cap.settleSeconds,
            "/", RtxOptions::nsysAutoCaptureSettleSeconds(), " gameplay seconds"));
        }

        if (cap.settleSeconds >= double(RtxOptions::nsysAutoCaptureSettleSeconds())) {
          cap.phase = NsysAutoCapture::Phase::Capture;

          // THIS is what starts the Nsight Systems collection, via
          // '--capture-range=nvtx --nvtx-capture=RemixCapture'.
          //
          // BOTH NVTX range forms are emitted, deliberately. A start/end pair
          // alone did NOT trigger nsys: two runs confirmed the range reached the
          // injection (nvtxRangeStartA returned a live id) while collection
          // never started and no report was written, with both the '@*' and the
          // plain-message forms of -p. The User Guide says only that profiling
          // starts when a matching range is "opened", never which NVTX range
          // API counts, so the remaining candidate is that only the push/pop
          // form is accepted as a trigger.
          //
          // Push/pop is a THREAD-LOCAL nested stack, so it is only correct if
          // the frame that ends the capture runs on the thread that began it.
          // That is why the start/end pair is kept alongside rather than
          // replaced - it is thread-agnostic and cannot silently fail to close.
          // The owning thread is recorded so a mismatch is visible in the log
          // instead of showing up as a range that never ends.
          cap.nvtxThreadId = uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
          nvtxRangePushA(kNsysCaptureRangeName);
          cap.nvtxRange = nvtxRangeStartA(kNsysCaptureRangeName);

          // The script also regexes this tag for progress. Keep it on one line.
          Logger::warn(str::format(
            "[NsysAuto] CAPTURE-BEGIN settled=", cap.settleSeconds,
            " captureSeconds=", RtxOptions::nsysAutoCaptureSeconds(),
            " nvtxRange=", uint64_t(cap.nvtxRange),
            " frame=", m_device->getCurrentFrameId()));
        }
        break;
      }

      case NsysAutoCapture::Phase::Capture: {
        cap.captureSeconds += delta;
        ++cap.capturedFrames;

        if (cap.captureSeconds >= double(RtxOptions::nsysAutoCaptureSeconds())) {
          cap.phase = NsysAutoCapture::Phase::Drain;

          // Closes both range forms, which under
          // '--capture-range-end=stop-shutdown' stops collection AND shuts the
          // session down, so nsys writes the report and (with --kill=true)
          // closes the game itself.
          const uint64_t endThreadId =
            uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
          const bool sameThread = (endThreadId == cap.nvtxThreadId);

          // Only pop on the thread that pushed. Popping on another thread would
          // pop that thread's stack (or nothing at all) rather than closing this
          // range - a silent corruption, and worse than leaving it to the
          // start/end pair below.
          if (sameThread) {
            nvtxRangePop();
          }
          if (cap.nvtxRange != 0) {
            nvtxRangeEnd(cap.nvtxRange);
            cap.nvtxRange = 0;
          }

          Logger::warn(str::format(
            "[NsysAuto] CAPTURE-END captured=", cap.captureSeconds,
            " frames=", cap.capturedFrames,
            " sameThread=", (sameThread ? 1 : 0),
            (sameThread ? "" : "  <- push/pop NOT closed, capture spans threads"),
            " frame=", m_device->getCurrentFrameId()));
        }
        break;
      }

      case NsysAutoCapture::Phase::Drain: {
        cap.drainSeconds += delta;

        if (cap.drainSeconds >= double(RtxOptions::nsysAutoCaptureDrainSeconds())) {
          cap.phase = NsysAutoCapture::Phase::Done;

          if (RtxOptions::nsysAutoCaptureExitOnFinish()) {
            Logger::warn("[NsysAuto] EXIT-ON-FINISH: terminating process "
                         "(rtx.nsysAutoCaptureExitOnFinish). NOTE: this cannot "
                         "confirm the .nsys-rep finished writing - the wrapper "
                         "script can, and is the supported way to run this.");
            // Give the log thread a moment to drain; TerminateProcess skips the
            // static teardown that would otherwise flush the file stream.
            std::this_thread::sleep_for(std::chrono::milliseconds(750));
            // TerminateProcess rather than exit(), for the same reason as the
            // sweep: this fork's shutdown path calls a cached client.dll pointer
            // after the engine unloaded that module.
            ::TerminateProcess(::GetCurrentProcess(), 0u);
          } else {
            Logger::warn("[NsysAuto] DONE - drain elapsed, still running "
                         "(rtx.nsysAutoCaptureExitOnFinish is False; the wrapper "
                         "script closes the game once the report is on disk).");
          }
        }
        break;
      }

      case NsysAutoCapture::Phase::Done:
      default:
        break;
    }
  }

  void RtxContext::updatePerfSweep(uint32_t& outUnorderedStopAfter,
                                   uint32_t& outSkipPom,
                                   uint32_t& outSkipMaterialTextures,
                                   uint32_t& outSkipThinFilm,
                                   uint32_t& outStepCensus,
                                   uint32_t& outMaterialStopAfter,
                                   uint32_t& outGbStopAfter,
                                   uint32_t& outCheapTextureGradients,
                                   uint32_t& outCoherentUnorderedFetch) {
    auto& sweep = m_perfSweep;

    // Table accessors. Lambdas rather than locals because sweep.quick is latched
    // partway through this function (at START), so a local captured at the top
    // would be stale for the rest of the frame that starts the sweep.
    const auto tableOf = [&sweep]() -> const PerfSweepStep* {
      return sweep.quick ? kPerfSweepQuickSteps : kPerfSweepSteps;
    };
    const auto countOf = [&sweep]() -> uint32_t {
      return sweep.quick ? kPerfSweepQuickStepCount : kPerfSweepStepCount;
    };

    // Applies one row's overrides. Passthrough rows (quick mode's probe step)
    // leave every out-param exactly as the caller set it, which is how the probe
    // ends up running with the config's own rtx.perf* values.
    const auto applyStep = [&](const PerfSweepStep& s) {
      if (!s.passthrough) {
        outUnorderedStopAfter    = s.unorderedStopAfter;
        outSkipPom               = s.skipPom;
        outSkipMaterialTextures  = s.skipMaterialTextures;
        outSkipThinFilm          = s.skipThinFilm;
        outStepCensus            = s.stepCensus;
        outMaterialStopAfter     = s.materialStopAfter;
        outGbStopAfter           = s.gbStopAfter;
        outCheapTextureGradients = s.cheapTextureGradients;
        outCoherentUnorderedFetch = s.coherentUnorderedFetch;
      }
      // Capture AFTER the branch so it covers both cases: table values for a
      // normal row, the caller's config values for a passthrough row. This is
      // the only point at which "what the shader will run with" is known.
      sweep.appliedUno    = outUnorderedStopAfter;
      sweep.appliedPom    = outSkipPom;
      sweep.appliedTex    = outSkipMaterialTextures;
      sweep.appliedFilm   = outSkipThinFilm;
      sweep.appliedCensus = outStepCensus;
      sweep.appliedMat    = outMaterialStopAfter;
      sweep.appliedGb     = outGbStopAfter;
      sweep.appliedGrad   = outCheapTextureGradients;
      sweep.appliedCoh    = outCoherentUnorderedFetch;
    };

    // NV-DXVK [Perf.Sweep]: one-shot presence marker, deliberately BEFORE the
    // enable check and unconditional. If [Perf.Sweep] START never appears, this
    // line separates the two possible causes without guesswork: marker present
    // and no START means the option read is returning false; no marker at all
    // means this function is not being reached. Fires exactly once per process.
    {
      static bool s_sweepMarkerLogged = false;
      if (!s_sweepMarkerLogged) {
        s_sweepMarkerLogged = true;
        Logger::warn(str::format(
          "[Perf.Sweep] PRESENT rev=3 (unordered-targeted: +cheapTexGradients, "
          "material rungs dropped) - updatePerfSweep reached, perfAutoSweep=",
          (RtxOptions::perfAutoSweep() ? "True" : "False"),
          " exitOnFinish=", (RtxOptions::perfAutoSweepExitOnFinish() ? "True" : "False")));
      }
    }

    if (!RtxOptions::perfAutoSweep()) {
      // Toggling the sweep off mid-run rearms it, so it can be run more than once
      // per process without a restart.
      if (sweep.active) {
        sweep.active        = false;
        sweep.finished      = false;
        sweep.step          = 0;
        sweep.censusActive  = false;
        sweep.gameplayReady = false;
        sweep.gameplayFrames = 0;
        sweep.waitingLogged = false;
      }
      return;
    }

    const auto now = std::chrono::steady_clock::now();

    // NV-DXVK [Perf.Sweep]: gameplay gate. Identical signal to the coverage
    // readback - a non-empty ordered-instance list means a world is loaded - with
    // a warmup, because the frame it first goes non-empty has a TLAS rebuild, a
    // bucket reorder and a BLAS attach all landing at once.
    //
    // Without this the sweep starts at process init and spends most of its steps
    // on menu frames, where gb_primaryRays is near zero for reasons that have
    // nothing to do with the probe being measured. That would not read as an
    // error - it would read as "every cut is free", which is exactly the kind of
    // wrong-but-plausible result this investigation has had to retract before.
    constexpr uint32_t kSweepGameplayWarmupFrames = 30u;

    const bool inGameplay =
      !getSceneManager().getAccelManager().getOrderedInstances().empty();

    if (inGameplay) {
      if (sweep.gameplayFrames < kSweepGameplayWarmupFrames)
        ++sweep.gameplayFrames;
    } else {
      sweep.gameplayFrames = 0;
    }

    sweep.gameplayReady = (sweep.gameplayFrames >= kSweepGameplayWarmupFrames);

    if (!sweep.active) {
      if (!sweep.gameplayReady) {
        // Say so once, so a sweep that appears to do nothing is distinguishable
        // from a sweep that is correctly waiting for a level to load.
        if (!sweep.waitingLogged) {
          sweep.waitingLogged = true;
          Logger::warn(str::format(
            "[Perf.Sweep] ARMED - waiting for gameplay (need ",
            kSweepGameplayWarmupFrames,
            " consecutive frames with world instances). Nothing is being measured "
            "yet; load into a level and the sweep starts on its own."));
        }
        sweep.lastTick          = now;
        outUnorderedStopAfter   = 0u;
        outSkipPom              = 0u;
        outSkipMaterialTextures = 0u;
        outSkipThinFilm         = 0u;
        outStepCensus           = 0u;
        outMaterialStopAfter    = 0u;
        outGbStopAfter          = 0u;
        outCheapTextureGradients = 0u;
        outCoherentUnorderedFetch = 0u;
        sweep.censusActive      = false;
        return;
      }
    }

    if (!sweep.active) {
      sweep.active    = true;
      sweep.finished  = false;
      sweep.step      = 0;
      sweep.stepStart = now;
      // Latch the table choice for the whole run. Read once here rather than
      // per frame so that toggling rtx.perfAutoSweepQuick mid-sweep cannot swap
      // the table under the accumulated results and silently mis-attribute
      // every row after the toggle.
      sweep.quick     = RtxOptions::perfAutoSweepQuick();
      // Freeze arithmetic below differences against lastTick; make sure it is a
      // real timestamp and not the default-constructed epoch on the first frame.
      sweep.lastTick  = now;
      sweep.minMs     = 0.0;
      sweep.maxMs     = 0.0;
      sweep.samples   = 0;
      const uint32_t startCount =
        sweep.quick ? kPerfSweepQuickStepCount : kPerfSweepStepCount;
      for (uint32_t i = 0; i < startCount && i < PerfSweep::kMaxSteps; ++i) {
        sweep.resultMs[i]      = 0.0;
        sweep.resultMinMs[i]   = 0.0;
        sweep.resultSamples[i] = 0;
      }
      const double totalSeconds =
        double(startCount) * double(RtxOptions::perfAutoSweepSeconds());
      Logger::warn(str::format(
        "[Perf.Sweep] START", (sweep.quick ? " [QUICK 3-step A-B-A]" : ""),
        " steps=", startCount,
        " holdSeconds=", RtxOptions::perfAutoSweepSeconds(),
        " settleSeconds=", RtxOptions::perfAutoSweepSettleSeconds(),
        " totalSeconds=", totalSeconds,
        " - image is garbage until this finishes; do not move the camera"));
      if (sweep.quick) {
        Logger::warn(
          "[Perf.Sweep] QUICK mode: the middle step is PASSTHROUGH - it runs "
          "with whatever rtx.perf* knobs are set in rtx.conf, bracketed by two "
          "steps that force every override off. Whatever you set is what is "
          "being measured; check the 'applied' fields on the step lines.");
      }
      // The census step needs the coverage readback path, which has its own
      // enable. Say so up front rather than letting the last step silently
      // produce no [Perf.UnorderedSteps] line 100+ seconds from now.
      if (!RtxOptions::logSurfaceCoverage()) {
        Logger::warn(
          "[Perf.Sweep] NOTE: rtx.logSurfaceCoverage is False, so the final "
          "census step will collect counters but print nothing. For the counts, "
          "set rtx.logSurfaceCoverage=True AND rtx.coveragePickRegionOnly=True "
          "(the latter suppresses the per-VS histogram spam that would otherwise "
          "dominate the frame). Timing steps are unaffected.");
      }
    }

    if (sweep.finished) {
      // Hold the last state rather than snapping back, so the log's final lines
      // are not interleaved with a state change nobody asked for.
      outUnorderedStopAfter   = 0u;
      outSkipPom              = 0u;
      outSkipMaterialTextures = 0u;
      outSkipThinFilm         = 0u;
      outStepCensus           = 0u;
      outMaterialStopAfter    = 0u;
      sweep.censusActive      = false;
      return;
    }

    // NV-DXVK [Perf.Sweep]: gameplay dropped out mid-sweep (load screen, menu,
    // alt-tab). Push the step's start forward by the elapsed wall time so the
    // hold clock effectively pauses. Resetting it instead would restart the step
    // and discard good samples; letting it run would end the step early on
    // frames that measured nothing.
    if (!sweep.gameplayReady) {
      sweep.stepStart += (now - sweep.lastTick);
      sweep.lastTick   = now;

      const PerfSweepStep& held = tableOf()[sweep.step];
      applyStep(held);
      sweep.censusActive      = (outStepCensus != 0u);
      return;
    }

    sweep.lastTick = now;

    const double heldSeconds =
      std::chrono::duration<double>(now - sweep.stepStart).count();

    if (heldSeconds >= double(RtxOptions::perfAutoSweepSeconds())) {
      // Close out the step that just ended.
      const uint32_t s = sweep.step;
      const double medianMs = perfSweepStepMedian();

      if (s < PerfSweep::kMaxSteps) {
        sweep.resultMs[s]      = medianMs;
        sweep.resultMinMs[s]   = sweep.minMs;
        sweep.resultSamples[s] = sweep.samples;
      }

      // The applied knobs are echoed on every step line. "Verify a knob reached
      // the code before believing any A/B" is the standing rule here, and three
      // tests in the previous round silently never ran; a step that reports a
      // delta of zero is ambiguous between "free" and "never applied" unless
      // the values it ran with are printed next to the number they produced.
      const PerfSweepStep& done = tableOf()[s];
      Logger::warn(str::format(
        "[Perf.Sweep] step=", s, "/", countOf() - 1u,
        " name=", done.name,
        " gb_primaryRays median=", medianMs,
        " min=", sweep.minMs,
        " max=", sweep.maxMs,
        " samples=", sweep.samples,
        // Recorded at apply time (see PerfSweep::applied*). NOT the table row -
        // that misreports passthrough steps. NOT the out-params either - this
        // log runs before this frame's applyStep, so they still hold whatever
        // the caller assigned from rtx.conf, which made every baseline echo the
        // config's knobs instead of the zeros it actually ran with.
        " | applied gb=", sweep.appliedGb,
        " uno=", sweep.appliedUno,
        " pom=", sweep.appliedPom,
        " tex=", sweep.appliedTex,
        " film=", sweep.appliedFilm,
        " mat=", sweep.appliedMat,
        " grad=", sweep.appliedGrad,
        " coh=", sweep.appliedCoh,
        " census=", sweep.appliedCensus,
        done.isBaseline ? " [BASELINE]" : "",
        done.passthrough ? " [PASSTHROUGH]" : "",
        (sweep.samples == 0u)
          ? "  *** NO SAMPLES - step shorter than settle window? ***" : ""));

      ++sweep.step;
      sweep.stepStart = now;
      sweep.minMs     = 0.0;
      sweep.maxMs     = 0.0;
      sweep.samples   = 0;

      if (sweep.step >= countOf()) {
        sweep.finished = true;

        // Final table. Each probe is scored against the MEAN OF THE BASELINES
        // THAT BRACKET IT, not against a single baseline from the top of the
        // run, so a probe's delta is work removed rather than work removed plus
        // whatever the machine drifted by in the intervening minutes.
        const PerfSweepStep* const tbl = tableOf();
        const uint32_t lastRow = (countOf() < PerfSweep::kMaxSteps)
                               ? countOf() : PerfSweep::kMaxSteps;

        // Nearest baseline row on each side; mean when both exist, the single
        // one when the probe sits at an end of the table. Reads the result
        // arrays rather than the step table so a baseline that produced no
        // samples is visible as a zero rather than silently averaged in.
        auto bracketingBaseline = [&](uint32_t i, const double* results) -> double {
          double before = 0.0, after = 0.0;
          bool haveBefore = false, haveAfter = false;

          for (int32_t j = int32_t(i) - 1; j >= 0; --j) {
            if (tbl[j].isBaseline && sweep.resultSamples[j] > 0u) {
              before = results[j];
              haveBefore = true;
              break;
            }
          }
          for (uint32_t j = i + 1u; j < lastRow; ++j) {
            if (tbl[j].isBaseline && sweep.resultSamples[j] > 0u) {
              after = results[j];
              haveAfter = true;
              break;
            }
          }

          if (haveBefore && haveAfter) return 0.5 * (before + after);
          if (haveBefore)              return before;
          if (haveAfter)               return after;
          return 0.0;
        };

        Logger::warn("[Perf.Sweep] ================ SUMMARY ================");

        // The baselines are printed as a group first. Their spread IS the
        // resolution limit of the run: no probe delta smaller than the gap
        // between the highest and lowest baseline means anything, however
        // confident the individual number looks.
        double baseLo = 0.0, baseHi = 0.0, baseSum = 0.0;
        uint32_t baseCount = 0u;
        for (uint32_t i = 0; i < lastRow; ++i) {
          if (!tbl[i].isBaseline || sweep.resultSamples[i] == 0u)
            continue;
          const double ms = sweep.resultMs[i];
          if (baseCount == 0u || ms < baseLo) baseLo = ms;
          if (baseCount == 0u || ms > baseHi) baseHi = ms;
          baseSum += ms;
          ++baseCount;
        }
        const double baseMean = (baseCount > 0u) ? (baseSum / double(baseCount)) : 0.0;

        Logger::warn(str::format(
          "[Perf.Sweep] baselines: n=", baseCount,
          " mean=", baseMean, " ms  min=", baseLo, " ms  max=", baseHi,
          " ms  spread=", baseHi - baseLo, " ms"));
        Logger::warn(str::format(
          "[Perf.Sweep] RESOLUTION FLOOR = ", baseHi - baseLo,
          " ms. Treat any |delta| below this as unmeasured, not as zero."));

        // Both columns are printed because they answer different questions and
        // disagreeing is itself information: median is the honest central value,
        // min is the least-disturbed frame in the step. When a step's median and
        // min deltas point the same way the effect is real; when only the median
        // moves, the step caught a hitch rather than a cost.
        for (uint32_t i = 0; i < lastRow; ++i) {
          if (tbl[i].isBaseline)
            continue;

          const double ms       = sweep.resultMs[i];
          const double minMs    = sweep.resultMinMs[i];
          const double localBase    = bracketingBaseline(i, sweep.resultMs);
          const double localBaseMin = bracketingBaseline(i, sweep.resultMinMs);
          const double delta    = ms - localBase;

          Logger::warn(str::format(
            "[Perf.Sweep]   ", tbl[i].name,
            " median=", ms, " delta=", delta,
            " | min=", minMs, " deltaMin=", minMs - localBaseMin,
            " | localBaseline=", localBase,
            " | pctOfBaseline=", (localBase > 0.0) ? (100.0 * ms / localBase) : 0.0,
            " (n=", sweep.resultSamples[i], ")",
            (std::abs(delta) < (baseHi - baseLo)) ? "  <-- BELOW NOISE FLOOR" : ""));
        }
        Logger::warn("[Perf.Sweep] gb* rungs are the top-level stage ladder and are "
                     "cumulative: 1=raygen 2=+traversal 3=+unordered 4=+material. "
                     "Per-stage cost is the difference between consecutive rungs; "
                     "rung 1 is the launch+store floor, not zero. READ THESE FIRST.");
        Logger::warn("[Perf.Sweep] uno* and mat_stop* rungs are cumulative too: each "
                     "includes every earlier rung, so per-part cost is the "
                     "difference between consecutive rungs, not the value itself.");
        Logger::warn("[Perf.Sweep] census_rawCounts ms is meaningless (its "
                     "InterlockedAdds inflate the pass) - read [Perf.UnorderedSteps].");
        Logger::warn("[Perf.Sweep] ========================================");

        if (RtxOptions::perfAutoSweepExitOnFinish()) {
          // Everything above must be on disk before we go. Logger::flush is not
          // exposed, but an error-level line is written through the same buffer
          // and the file stream is flushed on close during static teardown -
          // which TerminateProcess skips. So emit the marker, then give the log
          // thread a moment to drain rather than racing it.
          Logger::warn("[Perf.Sweep] EXIT-ON-FINISH: terminating process now "
                       "(rtx.perfAutoSweepExitOnFinish). Set it False to keep "
                       "playing after a sweep.");
          std::this_thread::sleep_for(std::chrono::milliseconds(750));

          // TerminateProcess rather than exit(): this fork's shutdown path has a
          // known hang/crash where a cached client.dll function pointer is
          // called after the engine unloaded that module. Running destructors
          // here risks losing the results the sweep just spent two minutes
          // gathering, and none of them matter to a process that is ending.
          ::TerminateProcess(::GetCurrentProcess(), 0u);
        }

        outUnorderedStopAfter   = 0u;
        outSkipPom              = 0u;
        outSkipMaterialTextures = 0u;
        outSkipThinFilm         = 0u;
        outStepCensus           = 0u;
        outMaterialStopAfter    = 0u;
        outGbStopAfter          = 0u;
        outCheapTextureGradients = 0u;
        outCoherentUnorderedFetch = 0u;
        sweep.censusActive      = false;
        return;
      }
    }

    const PerfSweepStep& cur = tableOf()[sweep.step];
    applyStep(cur);
    // Passthrough rows keep the config's own census setting, so read the flag
    // back from the out-param rather than from the table row.
    sweep.censusActive      = (outStepCensus != 0u);
  }

  void RtxContext::perfSweepAddSample(double gbPrimaryRaysMs) {
    auto& sweep = m_perfSweep;

    if (!sweep.active || sweep.finished)
      return;

    // Frames recorded while out of gameplay describe an empty scene. The step
    // clock is frozen for those, but the timestamp ring still resolves them, so
    // they have to be rejected here too or they would drag every mean down.
    if (!sweep.gameplayReady)
      return;

    // Discard the settle window. The timestamp ring is GpuStageTimers::kFrames
    // deep, so samples resolved just after a state change describe the previous
    // step's shader state, not this one's.
    const double heldSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sweep.stepStart).count();

    if (heldSeconds < double(RtxOptions::perfAutoSweepSettleSeconds()))
      return;

    if (sweep.samples == 0u) {
      sweep.minMs = gbPrimaryRaysMs;
      sweep.maxMs = gbPrimaryRaysMs;
    } else {
      sweep.minMs = std::min(sweep.minMs, gbPrimaryRaysMs);
      sweep.maxMs = std::max(sweep.maxMs, gbPrimaryRaysMs);
    }

    if (sweep.samples < PerfSweep::kMaxStepSamples)
      sweep.sampleMs[sweep.samples] = gbPrimaryRaysMs;

    ++sweep.samples;
  }

  // NV-DXVK [Perf.Sweep]: median of the step's samples. Deliberately a median
  // rather than a mean - see the sampleMs comment in rtx_context.h for what the
  // mean did to the first run's results.
  double RtxContext::perfSweepStepMedian() const {
    const auto& sweep = m_perfSweep;

    const uint32_t n = std::min(sweep.samples, PerfSweep::kMaxStepSamples);
    if (n == 0u)
      return 0.0;

    // Copy: nth_element reorders, and the sample buffer is reused by the next
    // step. Small and once per step, so the copy costs nothing.
    std::vector<double> sorted(sweep.sampleMs, sweep.sampleMs + n);
    std::sort(sorted.begin(), sorted.end());

    return (n % 2u == 1u)
      ? sorted[n / 2u]
      : 0.5 * (sorted[n / 2u - 1u] + sorted[n / 2u]);
  }

  void RtxContext::markGpuStage(uint32_t line) {
    auto& gpuStages = m_gpuStageTimers;

    if (gpuStages.pool == nullptr
     || gpuStages.writeCount >= GpuStageTimers::kSlots) {
      return;
    }

    // writeTimestamp resets the slot, writes it, and advances the pool's own
    // ring — sized kSlots*kFrames so a slot is not reused until kFrames later,
    // which is exactly the readback lag in injectRTX.
    // NV-DXVK [perf]: force prior work to COMPLETE before stamping, so the stage
    // measures execution rather than submission order. See perfGpuStageSerialize.
    // Without this, BOTTOM_OF_PIPE stamps early on compute/RT dispatches and the
    // cost lands on a later, often empty, interval.
    if (RtxOptions::perfGpuStageSerialize()) {
      emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }

    const VkCommandBuffer cmdBuf = getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    const uint32_t idx = gpuStages.pool->writeTimestamp(
      cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    // NV-DXVK [perf]: see markCmdBuf. A change in this handle between two
    // consecutive marks means the command buffer was submitted between them, so
    // the interval spans a submission gap and not just the commands in it.
    gpuStages.markCmdBuf[gpuStages.frameSlot][gpuStages.writeCount] = cmdBuf;
    gpuStages.markLine[gpuStages.frameSlot][gpuStages.writeCount] = line;

    gpuStages.slotIndex[gpuStages.frameSlot][gpuStages.writeCount] = idx;
    ++gpuStages.writeCount;
    gpuStages.stageCount[gpuStages.frameSlot] = gpuStages.writeCount;
  }

  // NV-DXVK [TlasProbe]: see the region block in common_binding_indices.h.
  //
  // Exists because the external route is closed - RenderDoc does not attach to
  // Source games, and PIX cannot see a Remix frame at all since the real
  // rendering is Vulkan. That only ever mattered as a way to ask one question,
  // and we own the ray tracer, so the question is asked here instead: shoot a
  // ray at the structure and see what comes back.
  // NV-DXVK [IdentProbe]: per-frame snapshots of the expected surface identity
  // (bit 31 | hashPacked << 11 | vsDebugId — the exact value writeGPUData puts
  // in the GPU surface entry) for every reordered-surface slot, ringed by
  // frame id. The census readback lags GPU execution by frames-in-flight while
  // the surface table reshuffles every frame, so observations (region 99) are
  // only comparable against the snapshot of the frame stamped in region 100.
  // CS-thread only — no locking.
  static constexpr uint32_t kIdentRingSize = 8u;
  static std::vector<uint32_t> s_identExpectedRing[kIdentRingSize];
  static uint32_t s_identExpectedRingFramePlus1[kIdentRingSize] = {};
  // NV-DXVK [CensusAlign]: per-frame slot->vsHash snapshots, same ring, same
  // fill site. The census's per-VS attribution of GPU-written counts (ordSeen,
  // rawHit, ...) used the LIVE table — the same readback-lag flaw [IdentProbe]
  // v1 had — so on table-shift frames a VS's hits were attributed to whoever
  // owns its slots at readback time and it read ordSeen=0: a FAKE dropout.
  // Every shift-correlated "dropout" statistic of 2026-08-02 (and the V6
  // handoff) is suspect for exactly this reason; the VS-id debug view shows
  // stable colors while the census reported whole-VS dropouts. Counts are now
  // attributed via the frame nibble in COVERAGE_OBS_IDENT_REGION joined to
  // this ring.
  static std::vector<uint64_t> s_identVsRing[kIdentRingSize];
  // NV-DXVK [CenSplit]: per-slot STABLE SURFACE IDENTITY, snapshotted into the
  // same ring as the VS hash and for the same reason.
  //
  // The first version of this read RtInstance::getId() from the LIVE table in
  // the raw dump, and on the 16:09 run it printed instId=1308 under
  // probeVs=0x29d5f7de0ba76c66 with vs=0x2947c6346103a2db - i.e. the identity
  // of whoever holds the slot NOW, next to probe data written when someone else
  // held it. Live owner and probe owner disagreed on 2 of 3 lines, so the pair
  // was wrong on exactly the lines it existed to explain. Mixing two frames'
  // state on one line is the [ProbeAlign] defect (V12 section 3.1), and this is
  // the third time this investigation has walked into it.
  //
  // Snapshotting here fixes it by construction: this walk runs at record time
  // over exactly the table the GPU is about to consume, so the identity travels
  // with the probe data instead of being re-derived later against a table that
  // has since reshuffled.
  static std::vector<uint64_t> s_identInstIdRing[kIdentRingSize];
  // NV-DXVK [CamProbe prevSurf]: the previous-frame slot of each slot, copied
  // from AccelManager::getProbePrevSurfaceSlots(). Rides the same ring row as
  // the VS hash and instance id so a reader gets all of them or none.
  static std::vector<uint32_t> s_identPrevSlotRing[kIdentRingSize];

  static uint32_t identExpectedOf(const RtInstance* inst) {
    if (inst == nullptr) {
      return 0u;
    }
    const RtSurface& rs = inst->surface;
    const uint16_t packedHash =
      (uint16_t) (rs.associatedGeometryHash >> 48) ^
      (uint16_t) (rs.associatedGeometryHash >> 32) ^
      (uint16_t) (rs.associatedGeometryHash >> 16) ^
      (uint16_t) rs.associatedGeometryHash;
    return 0x80000000u
      | ((uint32_t(packedHash) & 0xFFFFu) << 11)
      | (uint32_t(rs.vsDebugId) & 0x7FFu);
  }

  void RtxContext::dispatchTlasProbe(const Resources::RaytracingOutput& rtOutput) {
    if (!RtxOptions::logResolveCensus()) {
      return;
    }

    const uint32_t surfaceCount =
      getSceneManager().getAccelManager().getSurfaceCount();
    if (surfaceCount == 0) {
      return;
    }

    ScopedGpuProfileZone(this, "TlasProbe");

    // NV-DXVK [CamTris]: publish each surface's triangle count for the probe.
    //
    // The shader cannot derive this. Surface carries firstIndex but no count,
    // and striding past a surface's end without one would read the NEXT
    // surface's indices and silently report its triangles as this one's - a
    // wrong answer that looks like a real one, which is the only kind this
    // investigation cannot afford. The count lives on the BLAS build range,
    // CPU-side only, so it has to be handed over.
    //
    // This region is owned end to end HERE - written and tail-zeroed on every
    // frame immediately before the dispatch that consumes it, and deliberately
    // excluded from the census readback's memset.
    //
    // The first version let the readback clear it, and that was a race the
    // readback always won: the census runs while this dispatch has only been
    // RECORDED, so the zeroing landed before the GPU ever read the counts. The
    // sampling then did nothing at all on 2571 of 2580 frames, all-or-nothing
    // per frame. Keeping the region's lifetime inside this function makes that
    // failure impossible rather than unlikely - no other code touches it, so
    // there is no ordering left to get wrong.
    if (rtOutput.m_surfaceCoverageBuffer.ptr() != nullptr) {
      uint32_t* cov = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
      if (cov != nullptr) {
        const auto& ordered = getSceneManager().getAccelManager().getOrderedInstances();
        const uint32_t basePrim = COVERAGE_TLASPROBE_PRIMCOUNT_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseMask = COVERAGE_INSTMASK_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t scan = std::min(uint32_t(ordered.size()), uint32_t(COVERAGE_SURFACE_SLOTS));
        // NV-DXVK [IdentProbe]: snapshot THIS frame's expected identities into
        // the ring — this walk happens at record time over exactly the table
        // the GPU is about to consume, which is the alignment the probe needs.
        const uint32_t identFrame = m_device->getCurrentFrameId();
        const uint32_t identRingSlot = identFrame % kIdentRingSize;
        s_identExpectedRingFramePlus1[identRingSlot] = identFrame + 1u;
        s_identExpectedRing[identRingSlot].assign(scan, 0u);
        s_identVsRing[identRingSlot].assign(scan, 0ull);
        s_identInstIdRing[identRingSlot].assign(scan, 0ull);
        // NV-DXVK [CamProbe prevSurf]: filled from the accel manager's snapshot,
        // which uploadSurfaceData wrote earlier this frame. Default INVALID, so
        // a slot the snapshot does not cover reads as "no predecessor" rather
        // than as slot 0.
        const auto& prevSlots = getSceneManager().getAccelManager().getProbePrevSurfaceSlots();
        s_identPrevSlotRing[identRingSlot].assign(scan, uint32_t(SURFACE_INDEX_INVALID));
        for (uint32_t s = 0; s < scan; ++s) {
          const RtInstance* inst = ordered[s];
          const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
          s_identExpectedRing[identRingSlot][s] = identExpectedOf(inst);
          // NV-DXVK [CenSplit]: identity captured at the same instant as the VS
          // hash, so a later reader gets both or neither.
          s_identInstIdRing[identRingSlot][s] = (inst != nullptr) ? inst->getId() : 0ull;
          if (s < prevSlots.size()) {
            s_identPrevSlotRing[identRingSlot][s] = prevSlots[s];
          }
          s_identVsRing[identRingSlot][s] = (blas != nullptr)
            ? uint64_t(blas->input.getTransformData().vertexShaderHash) : 0ull;
          // modifiedGeometryData.calculatePrimitiveCount(), NOT
          // buildRanges[0].primitiveCount. The two are not interchangeable
          // here: buildRanges describes the acceleration-structure build, and a
          // merged BLAS covers SEVERAL surfaces, so its count would send the
          // sampler striding out of this surface's index range and into its
          // neighbour's triangles. Those hits resolve to a different
          // surfaceIndex and are counted as "not reached", which quietly
          // understates exactly the number this probe exists to report.
          //
          // calculatePrimitiveCount() is indexCount / 3 for the geometry this
          // entry actually describes, which is precisely the range the shader
          // addresses as firstIndex + triangleIndex * 3.
          //
          // 0 means "unknown", and the shader skips the sampling entirely on 0.
          // Better to measure nothing for a surface than to walk an index range
          // whose length is a guess.
          cov[basePrim + s] = (blas != nullptr)
            ? blas->modifiedGeometryData.calculatePrimitiveCount()
            : 0u;
          // NV-DXVK [InstMask]: the mask actually built into the TLAS instance.
          // A zero mask is traced by nothing - the geometry would drop out of
          // the primary ray while every other bookkeeping field stayed correct,
          // and the probe would still find it because the probe's queries carry
          // their own masks. +1 so an unwritten slot (0) is distinguishable
          // from a genuine mask of 0, which is the value that matters most.
          cov[baseMask + s] = (inst != nullptr) ? (inst->censusMask() + 1u) : 0u;
        }
        // Zero the tail rather than leaving last frame's counts there. Surface
        // slots are frame-local and get reassigned on every TLAS build, so a
        // stale count would send the sampler striding through an index range
        // belonging to geometry that no longer occupies that slot.
        if (scan < uint32_t(COVERAGE_SURFACE_SLOTS)) {
          const size_t tailBytes =
            size_t(uint32_t(COVERAGE_SURFACE_SLOTS) - scan) * sizeof(uint32_t);
          std::memset(&cov[basePrim + scan], 0, tailBytes);
          std::memset(&cov[baseMask + scan], 0, tailBytes);
        }
      }
    }

    bindCommonRayTracingResources(rtOutput);
    bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TlasProbeShader::getShader());

    // 64 threads per group, matching [numthreads(64,1,1)]. The shader also
    // bounds-checks against cb.surfaceCount, so the tail group is safe.
    dispatch((surfaceCount + 63u) / 64u, 1, 1);
  }

  void RtxContext::dispatchPathTracing(const Resources::RaytracingOutput& rtOutput) {

    // NV-DXVK [TlasProbe]: run BEFORE the gbuffer pass, against the same TLAS
    // the gbuffer is about to traverse and in the same command buffer, so the
    // probe result and the primary-ray result on that frame describe one
    // acceleration structure. Running it afterwards would leave open the
    // objection that something between the two passes changed the structure -
    // which is precisely the class of explanation this probe exists to test.
    dispatchTlasProbe(rtOutput);

    // Gbuffer Raytracing
    m_common->metaPathtracerGbuffer().dispatch(this, rtOutput);
    markGpuStage();

    // NV-DXVK: visible-surface readback. Must run before any later pass aliases
    // m_sharedSurfaceIndex storage (e.g. m_primaryDisocclusionMaskForRR).
    recordVisibleSurfacesReadback(rtOutput);
    markGpuStage();

    // RTXDI
    m_common->metaRtxdiRayQuery().dispatch(this, rtOutput);
    markGpuStage();

    // NEE Cache
    dispatchNeeCache(rtOutput);
    markGpuStage();

    // Integration Raytracing
    dispatchIntegrate(rtOutput);
    markGpuStage();
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
    // NV-DXVK [tonemap operators]: the fork operators (Psycho17/GT7/Hable/PSDT) live in
    // the GLOBAL tonemapper's apply shader. The default tonemappingMode is Local,
    // which bypasses the global path entirely — so when an operator is selected,
    // force the global tonemapper to run and skip the local one. Otherwise the
    // operator selection silently does nothing (the local tonemapper owns output).
    const bool operatorSelected =
      DxvkToneMapping::tonemapOperator() != DxvkToneMapping::TonemapOperator::None;
    // NV-DXVK [auto exposure plus]: same override as above, for the same reason. Plus already
    // did the local dynamic range compression back before bloom, so the LOCAL tonemapper would
    // compress it a second time and wash the image out. Force the global path instead. Like the
    // operator case this only redirects which tonemapper runs - rtx.tonemappingMode itself is
    // left exactly as the user set it, so turning Plus off restores their choice.
    const bool plusForcesGlobal =
      m_common->metaAutoExposurePlus().isActive() && DxvkAutoExposurePlus::forcesGlobalTonemapper();
    if (RtxOptions::tonemappingMode() == TonemappingMode::Global || operatorSelected || plusForcesGlobal) {
      DxvkToneMapping& toneMapper = m_common->metaToneMapping();
      // NV-DXVK [auto exposure plus]: carry the local tonemapper's S-curve across the override.
      // Only the native curve needs it - the fork operators (Hable/Psycho17/GT7/PSDT) bring their own
      // toe and shoulder and bypass finalizeWithACES entirely, so forcing it there would be a
      // no-op at best and misleading in the log at worst.
      const bool forceACES = plusForcesGlobal && !operatorSelected;
      toneMapper.dispatch(this,
        getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER),
        autoExposure.getExposureTexture().view,
        rtOutput, GlobalTime::get().deltaTimeMs(), performSRGBConversion,
        // NV-DXVK [auto exposure plus]: `autoExposure.enabled()` used to be passed here, where
        // it lands in `resetHistory`, not `autoExposureEnabled`. With auto exposure on (the
        // default) that set m_resetState every single frame, so the tone curve pass took its
        // `needsReset` branch every frame and skipped its own temporal filter - the adaptive
        // curve was rebuilt from one frame's histogram instead of easing towards it, which
        // renders flat and unstable. Harmless while the default tonemappingMode kept the global
        // path dormant; not harmless now that Plus routes through it. `false` is what the
        // comment above this block already asks for, and the enable flag now reaches the
        // parameter it was meant for.
        false, autoExposure.enabled(), forceACES);
    }
    DxvkLocalToneMapping& localTonemapper = m_common->metaLocalToneMapping();
    if (localTonemapper.isActive() && !operatorSelected && !plusForcesGlobal) {
      localTonemapper.dispatch(this,
        getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
        autoExposure.getExposureTexture().view,
        // NV-DXVK [auto exposure plus]: same misplaced argument as the global path above. This
        // one is inert - DxvkLocalToneMapping accepts `resetHistory` and never reads it - so
        // straightening it out cannot change how Base looks, which keeps it valid as the
        // reference to compare Plus against. Corrected so the two call sites read alike.
        rtOutput, GlobalTime::get().deltaTimeMs(), performSRGBConversion, false, autoExposure.enabled());
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

  void RtxContext::dispatchAutoExposurePlus(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();

    DxvkAutoExposurePlus& autoExposurePlus = m_common->metaAutoExposurePlus();
    if (!autoExposurePlus.isActive()) {
      return;
    }

    this->spillRenderPass(false);
    this->unbindComputePipeline();

    DxvkAutoExposure& autoExposure = m_common->metaAutoExposure();

    // Note this runs before dispatchToneMapping, where the histogram pass updates the exposure
    // texture, so the value read here is one frame old. That is harmless: it is used only to
    // locate middle grey while measuring, it is already heavily temporally smoothed by
    // rtx.autoExposure.autoExposureSpeed, and this pass smooths its own result on top.
    //
    // Less obviously, the base histogram meters m_finalOutput *after* this pass has written to
    // it, so the two form a closed loop: base exposure sets where Plus thinks middle grey is,
    // and Plus's output sets what base exposure measures. That is stable only while Plus's mean
    // gain over the frame is near zero, which it is by construction - base exposure anchors the
    // mean key to middle grey, and the gain is proportional to the key. Any bias in the key
    // measurement therefore does not just shift the image, it gets fought over: base exposure
    // pulls the other way, which moves the key, which moves the gain. Keep
    // autoExposurePlusLogLuminance unbiased.
    autoExposurePlus.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      autoExposure.getExposureTexture().view,
      rtOutput, GlobalTime::get().deltaTimeMs(),
      m_resetHistory || getSceneManager().getCamera().isCameraCut(),
      autoExposure.enabled());
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

    // NV-DXVK [Coverage compact]: record the compute pass that folds the
    // 74 MB coverage buffer's NONZERO slots into the small host-cached
    // compact buffer (see coverage_compact.h — replaces the ~1 s/frame
    // uncached CPU scan). It MUST be recorded AFTER every GPU coverage
    // writer of the frame: the RT passes ran before dispatchDebugView, but
    // debugView.dispatch()'s postprocess pass writes the PickRegion slots
    // ([PickHash] / center-VS feed) — so this is invoked at each exit of
    // this function, never before debugView.dispatch().
    auto recordCoverageCompactDispatch = [this, &rtOutput]() {
      if (!RtxOptions::logSurfaceCoverage()
          || rtOutput.m_surfaceCoverageBuffer.ptr() == nullptr
          || rtOutput.m_surfaceCoverageCompactBuffer.ptr() == nullptr) {
        return;
      }
      const Rc<DxvkBuffer>& covCompactBuf = rtOutput.m_surfaceCoverageCompactBuffer;

      ScopedGpuProfileZone(this, "Coverage Compact");
      this->spillRenderPass(false);
      this->unbindComputePipeline();

      // Zero the append cursor so this frame's pass starts fresh.
      clearBuffer(covCompactBuf, 0, COVERAGE_COMPACT_HEADER_UINTS * sizeof(uint32_t), 0);

      // 1M threads grid-striding over 18.6M elements keeps reads coalesced
      // and stays well under the 65535 group-count limit.
      constexpr uint32_t kThreadsPerGroup = 256u;
      constexpr uint32_t kGroups = 4096u;

      CoverageCompactArgs pushArgs = {};
      pushArgs.totalElements = uint32_t(size_t(COVERAGE_TOTAL_REGIONS) * COVERAGE_SURFACE_SLOTS);
      pushArgs.threadCount = kThreadsPerGroup * kGroups;
      pushArgs.entryCapacity = COVERAGE_COMPACT_MAX_ENTRIES;
      setPushConstantBank(DxvkPushConstantBank::RTX);
      pushConstants(0, sizeof(pushArgs), &pushArgs);

      bindResourceBuffer(COVERAGE_COMPACT_INPUT,
        DxvkBufferSlice(rtOutput.m_surfaceCoverageBuffer, 0, rtOutput.m_surfaceCoverageBuffer->info().size));
      bindResourceBuffer(COVERAGE_COMPACT_OUTPUT,
        DxvkBufferSlice(covCompactBuf, 0, covCompactBuf->info().size));
      bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CoverageCompactShader::getShader());
      dispatch(kGroups, 1, 1);
    };

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
    // NV-DXVK [VsPix]: per-frame pixel count per vertex shader.
    //
    // Deliberately NOT gated on rtx.logSurfaceCoverage: that option arms the
    // full-screen grid readbacks and the per-frame GPU sync (~104 ms/frame),
    // and needing it here would make the cheap probe cost as much as the
    // expensive one. This reads 125 uints from one region and nothing else.
    //
    // No sync either. Reading without one can catch a frame mid-write, so an
    // individual count is approximate — but the question this answers is
    // "WHICH shader covered pixels this frame", and presence/absence survives
    // that slop. Anything relying on the exact count should say so.
    //
    // Slots are zeroed immediately after reading so each line is one frame's
    // worth rather than a running total.
    if (m_common->metaDebugView().debugViewIdx() == DEBUG_VIEW_VERTEX_SHADER_ID
        && rtOutput.m_surfaceCoverageBuffer.ptr() != nullptr) {
      uint32_t* covVs = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
      if (covVs != nullptr) {
        const uint32_t baseVs = COVERAGE_VSPIX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t frameVs = m_device->getCurrentFrameId();

        uint32_t totalVs = 0;
        for (uint32_t i = 1; i <= COVERAGE_VSPIX_MAX_ID; ++i) {
          totalVs += covVs[baseVs + i];
        }

        if (totalVs > 0) {
          for (uint32_t i = 1; i <= COVERAGE_VSPIX_MAX_ID; ++i) {
            const uint32_t px = covVs[baseVs + i];
            if (px == 0) {
              continue;
            }
            // id joins to [VsColor], which already carries id -> VS hash and is
            // emitted once per shader per run. Keeping the hash out of here
            // avoids reaching across into the InstanceManager's id table.
            Logger::info(str::format(
              "[VsPix] f=", frameVs,
              " id=", i,
              " px=", px,
              " pctOfCovered=", (100.0f * float(px) / float(totalVs)),
              " (join id -> vs via [VsColor])"));
          }
          Logger::info(str::format(
            "[VsPix] f=", frameVs, " TOTAL coveredPixels=", totalVs));
        }

        // Clear only the slots actually used.
        std::memset(&covVs[baseVs], 0,
                    (COVERAGE_VSPIX_MAX_ID + 1) * sizeof(uint32_t));
      }
    }

    // NV-DXVK [ResolveCensus]: per-VS census of the primary resolve path.
    //
    // The open question from HANDOFF_PI_FLICKER_V4 §6/§7: a set of shaders
    // ("Class B") renders tens of thousands of visible pixels but wins 0-7% of
    // primary hits, i.e. it almost never writes m_sharedSurfaceIndex, and
    // nothing measured so far can say whether the primary ray misses that
    // geometry entirely or hits it and resolves past it. [HitCensus] reads
    // SharedSurfaceIndex, which only ever holds the winner, so it structurally
    // cannot answer that. These four counters are written by the resolver
    // itself, at the moment each decision is made.
    //
    // Standalone rather than folded into the logSurfaceCoverage block below,
    // because that dump costs ~104 ms/frame and emits ~80 lines/frame. The
    // flicker is a single-frame event (handoff §2: an even-stride sampler sits
    // at one phase and reports it clean), so the instrument that hunts it has to
    // be cheap enough to run on EVERY frame. This one reads 4 * orderedSize
    // uints and emits one line per VS.
    //
    // Deliberately NOT throttled by a frame modulo, for the same reason.
    if (RtxOptions::logResolveCensus()
        && rtOutput.m_surfaceCoverageBuffer.ptr() != nullptr
        && s_coverageStableFrames >= kCoverageWarmupFrames) {
      // Same sync knob the main dump uses (handoff §9 trap 2). Without it the
      // CPU reads whichever frame the GPU last finished while the current
      // frame's atomics are still in flight, and the memset below can then
      // clear counts mid-accumulation - which would under-report exactly the
      // single-frame dropout being hunted. If the main coverage dump also runs
      // this frame it waits again, but nothing is submitted in between so the
      // second wait costs nothing.
      if (RtxOptions::coverageSyncBeforeReadback()) {
        m_device->waitForIdle();
      }

      uint32_t* covRc = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
      if (covRc != nullptr) {
        const auto& reorderedRc = getSceneManager().getAccelManager().getOrderedInstances();
        const uint32_t frameRc = m_device->getCurrentFrameId();
        const uint32_t baseOrdSeen   = COVERAGE_RESOLVE_ORDSEEN_REGION   * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseOrdFinal  = COVERAGE_RESOLVE_ORDFINAL_REGION  * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseUnoSeen   = COVERAGE_RESOLVE_UNOSEEN_REGION   * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseContinued = COVERAGE_RESOLVE_CONTINUED_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);

        // Only slots that map to a live instance are READ. The mapping is what
        // makes a count meaningful (surfaceIndex -> VS), and reading the whole
        // 262144-slot region over this write-combined mapping would cost ~55 ms
        // in uncached reads, which would defeat the point of a cheap probe. The
        // full region is still CLEARED below - writes to write-combined memory
        // are cheap, so nothing accumulates across frames. Out-of-range /
        // stale-slot attribution is deliberately left to the existing
        // [Coverage] dump, which already reports unmapped/stale/impossible.
        const uint32_t scanRc = std::min(uint32_t(reorderedRc.size()), uint32_t(COVERAGE_SURFACE_SLOTS));

        // NV-DXVK [PIWrite]: same index space, written by the PI culling pass
        // earlier in the frame. Read alongside the census so one line carries
        // both "did the ray reach it" and "what was in its TLAS entry".
        const uint32_t basePiwMask   = COVERAGE_PIW_MASK_REGION   * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t basePiwFlags  = COVERAGE_PIW_FLAGS_REGION  * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t basePiwDist   = COVERAGE_PIW_DIST_REGION   * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t basePiwBlasLo = COVERAGE_PIW_BLASLO_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseProbeFlags = COVERAGE_TLASPROBE_FLAGS_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [CamProbe]
        const uint32_t baseCamHitSurf = COVERAGE_TLASPROBE_CAMHITSURF_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseCamDotA    = COVERAGE_TLASPROBE_CAMDOTA_REGION    * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseCamDotB    = COVERAGE_TLASPROBE_CAMDOTB_REGION    * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseCamDist    = COVERAGE_TLASPROBE_CAMDIST_REGION    * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [CamTris]
        const uint32_t basePrimCount  = COVERAGE_TLASPROBE_PRIMCOUNT_REGION      * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseTriTested  = COVERAGE_TLASPROBE_CAMTRISTESTED_REGION  * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseTriReached = COVERAGE_TLASPROBE_CAMTRISREACHED_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [RawHit] / [SurfMap] / [InstMask]
        const uint32_t baseRawHit     = COVERAGE_RAWHIT_REGION   * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [IdentProbe]
        const uint32_t baseObsIdent   = COVERAGE_OBS_IDENT_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [ProbeAlign]
        const uint32_t baseProbeFrame = COVERAGE_TLASPROBE_FRAME_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseSurfMap    = COVERAGE_SURFMAP_REGION  * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseInstMask   = COVERAGE_INSTMASK_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseOccluder   = COVERAGE_TLASPROBE_OCCLUDER_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseTriNoCull  = COVERAGE_TLASPROBE_CAMTRISNOCULL_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseTriMissed  = COVERAGE_TLASPROBE_CAMTRISMISSED_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseMaxRadius  = COVERAGE_TLASPROBE_MAXRADIUS_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [Centroid]
        const uint32_t baseCenX       = COVERAGE_TLASPROBE_CENTROIDX_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseCenY       = COVERAGE_TLASPROBE_CENTROIDY_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseCenZ       = COVERAGE_TLASPROBE_CENTROIDZ_REGION * uint32_t(COVERAGE_SURFACE_SLOTS);
        const uint32_t baseClipW      = COVERAGE_TLASPROBE_CLIPW_REGION     * uint32_t(COVERAGE_SURFACE_SLOTS);
        // NV-DXVK [Centroid]: the shader stores these regions as float bit
        // patterns. memcpy, not bit_cast - this target still builds at C++17
        // when the compiler does not offer 'latest' (meson.build:123), and
        // rtx_nrd_context.cpp:966 already records bit_cast as unavailable here.
        // Declared once at this scope so the per-surface accumulation and the
        // raw per-surface dump decode identically; two copies of a conversion
        // that are meant to agree is a place for them to stop agreeing.
        auto u2fRaw = [](uint32_t u) {
          float f;
          std::memcpy(&f, &u, sizeof(f));
          return f;
        };

        struct RcVs {
          uint64_t ordSeen = 0ull, ordFinal = 0ull, unoSeen = 0ull, continued = 0ull;
          uint32_t surfaces = 0u;
          // [PIWrite] rollup. piNotWritten counts slots the culling shader did
          // not touch this frame (raw region value 0) — for a NOTRAVERSED VS
          // that is the single most decisive number in the line, because those
          // entries keep the CPU placeholder's mask=0 and are invisible to the
          // ray without anything having gone wrong on the GPU at all.
          uint32_t piNotWritten = 0u, piMaskZero = 0u, piMaskLive = 0u;
          uint32_t piDistNonFinite = 0u, piPosNonFinite = 0u, piBlasRefZero = 0u;
          uint32_t piDistMin = UINT32_MAX, piDistMax = 0u;
          uint32_t piBlasLoAny = 0u;
          // NV-DXVK [BlasState]: the last layer between a correct instance entry
          // and a ray that misses. [PIWrite] proved the entries are equivalent
          // on render and vanish frames - same count, same live masks, same
          // positions, same nonzero BLAS reference - so what remains is what
          // that reference POINTS AT, and whether the TLAS build covered it.
          //
          // All read CPU-side from reordered[s]->getBlas(); no GPU plumbing is
          // needed because the BLAS bookkeeping lives on BlasEntry/PooledBlas.
          //   blasRefMismatch - the GPU entry's blasRefLo != the low 32 bits of
          //                     the reference the CPU currently holds for that
          //                     geometry. Nonzero means the entry addresses a
          //                     DIFFERENT acceleration structure than the one
          //                     the CPU thinks is live: a stale AS, which is
          //                     invisible while looking perfectly valid.
          //   blasNull        - no dynamicBlas / no accelStructure at all.
          //   blasPrims       - primitives in the build range. A live reference
          //                     to a zero-primitive BLAS is exactly "entry is
          //                     correct, ray hits nothing".
          //   blasTouchAge    - currentFrame - PooledBlas::frameLastTouched
          //                     ("last used in a TLAS"). Large on a frame where
          //                     the geometry vanished says the build skipped it.
          //   blasUpdateAge   - currentFrame - BlasEntry::frameLastUpdated.
          uint32_t blasNull = 0u, blasRefMismatch = 0u, blasRefZeroCpu = 0u;
          uint32_t blasPrimsMin = UINT32_MAX, blasPrimsMax = 0u, blasPrimsZero = 0u;
          uint32_t blasTouchAgeMax = 0u, blasUpdateAgeMax = 0u;
          uint64_t blasRefAny = 0ull;
          // NV-DXVK [TlasProbe]: results of the per-surface ray query against
          // the built TLAS. prAnyMiss is the decisive one - a surface the
          // maximally permissive query cannot find is absent from the structure
          // the driver built, whatever the inputs to that build said.
          uint32_t prRan = 0u, prStrictHit = 0u, prAnyHit = 0u;
          uint32_t prStrictSelf = 0u, prAnySelf = 0u, prAnyMiss = 0u;
          // On-screen counts. prOnScreenSelf is the one that decides whether a
          // NOTRAVERSED verdict is a defect: present in the TLAS, self-hittable,
          // AND inside the frustum, yet the primary ray recorded nothing.
          uint32_t prOnScreen = 0u, prOnScreenSelf = 0u;
          // NV-DXVK [CamProbe]: prOnScreenSelf came back > 0 on NOTRAVERSED
          // frames and the geometry is not occluded, so the primary ray itself
          // is the only remaining link. These split it in two.
          //
          // camDot* are MAXIMA, not means, and only over surfaces in the
          // population that matters (on screen and self-hittable). A mean over
          // all 258 surfaces of a VS would average a single 30-degree outlier
          // into invisibility, and one wrong surface is the whole bug - the
          // failure is per-object. camProjRan is the denominator; without it a
          // max of 0 cannot be told from nothing having been measured.
          uint32_t camProjRan = 0u;
          uint32_t camDotAMax = 0u, camDotBMax = 0u;
          uint32_t camDotAMaxOnScreenSelf = 0u, camDotBMaxOnScreenSelf = 0u;
          // Camera-ray outcomes over the same on-screen self-hittable
          // population: reached it, hit something else (occluder), or found
          // nothing at all along the segment.
          uint32_t camRaySelf = 0u, camRayOther = 0u, camRayMiss = 0u;
          // camRayCulled: the permissive camera ray reached this surface but
          // the primary pass's flags did not. That difference is exactly
          // backface culling plus the object mask, on the camera direction.
          uint32_t camRayAnySelf = 0u, camRayCulled = 0u;
          uint32_t camDistMin = UINT32_MAX, camDistMax = 0u;
          // NV-DXVK [CamTris]: the visibility test the single-triangle probe
          // could not be. Summed over ALL surfaces of the VS, not just on-screen
          // ones - the question "does this object own any pixel" is answered by
          // the object as a whole, and restricting to a per-surface on-screen
          // flag derived from triangle 0 would reintroduce the same bad aim.
          // camTrisSurfaces counts surfaces with a nonzero denominator, so
          // reached==0 can be told from nothing having been sampled.
          uint32_t camTrisTested = 0u, camTrisReached = 0u, camTrisSurfaces = 0u;
          // NV-DXVK [RawHit]: hits traversal committed, counted before
          // resolveVertex could discard them. rawHit > 0 with ordSeen == 0 is
          // the whole point - it says the ray DID find this geometry and the
          // resolve stage threw it away, which no measurement so far could see.
          uint64_t rawHit = 0ull;
          // NV-DXVK [IdentProbe]: GPU-observed surface identity vs what the CPU
          // wrote into that slot THIS frame. identSeen counts slots the probe
          // wrote (bit 31); identBad counts disagreements — each one is a hit
          // that consumed a different frame's surface-table layout than the CPU
          // submitted. First offender kept raw for the log.
          uint32_t identSeen = 0u, identBad = 0u;
          uint32_t identBadFirstSlot = UINT32_MAX;
          uint32_t identBadExpected = 0u, identBadObserved = 0u;
          // NV-DXVK [SurfMap] / [InstMask]: the two indirections between a
          // committed hit and a resolved surface. Counted as populations rather
          // than averaged - one surface with a broken mapping is the bug, and a
          // mean over 32 surfaces would bury it.
          uint32_t surfMapUnwritten = 0u, surfMapInvalid = 0u, surfMapMismatch = 0u;
          uint32_t instMaskZero = 0u, instMaskUnwritten = 0u, instMaskAny = 0u;
          // NV-DXVK [InstDiag]: the layer ABOVE the ray.
          //
          // rawHit == 0 on every NOTRAVERSED row proved traversal never commits
          // a hit, and the probe agrees the geometry is frontmost nowhere - so
          // the rays, the TLAS and the resolve stage are all behaving. What has
          // never been checked is whether the instances are still THERE and
          // still WHERE they were: dropouts hit ~5 VS at once and lose more
          // pixels than the geometry accounts for, which is what a group of
          // instances leaving the scene for one frame looks like.
          //
          //   instLive     - instances the instance manager holds for this VS.
          //   instInTlas   - how many of those the accel manager actually built
          //                  into this frame's TLAS. instLive > instInTlas is an
          //                  instance that exists and is not traceable.
          //   instMaskLive0- live instances carrying mask 0 (traced by nothing).
          //   instMoved    - instances whose world position differs from the
          //                  IMMEDIATELY previous frame, and the largest such
          //                  jump. Compared frame-to-frame rather than against a
          //                  threshold, so a teleport and a slow drift are
          //                  distinguishable instead of both being "moved".
          uint32_t instLive = 0u, instInTlas = 0u, instMaskLive0 = 0u, instMoved = 0u;
          float instMoveMax = 0.0f;
          // NV-DXVK [Occluder]: what blocked this VS's geometry, resolved from
          // a surface index to the vertex shader that drew it. Kept as a small
          // tally rather than a single winner - if one thing is responsible it
          // will dominate, and if several are, that is itself the answer.
          std::unordered_map<uint64_t, uint32_t> occluderVs;
          uint32_t occludedSurfaces = 0u;
          // NV-DXVK [NoCull]: the cull-flag A/B, and the per-triangle miss
          // count that finally makes the sample add up
          // (tested = reached + missed + blocked-by-something-else).
          uint32_t camTrisReachedNoCull = 0u, camTrisMissed = 0u;
          // NV-DXVK [Spike]: per-surface geometric extent about its own origin.
          // Max, not mean - one deformed surface out of 300 is the bug, and a
          // mean over the VS would divide it away to nothing.
          uint32_t maxRadius = 0u;
          // NV-DXVK [FlickerTrack]: this VS's history, copied in from the
          // persistent tracker below so piSuffix can print it without the
          // tracker having to be threaded through the lambda.
          //
          // Every other field in this struct describes ONE frame. That shape is
          // what let a whole session be spent on the wrong population: a VS that
          // is absent on 100% of frames and a VS that is absent on 40% of them
          // produce IDENTICAL census lines, and only the second one is a
          // flicker. A count of state CHANGES separates them in one number, and
          // it is the only number on the line that can.
          // NV-DXVK [Centroid]: the world position the ONSCREEN verdict is
          // computed from, as a per-VS bbox over its surfaces' triangle-0
          // centroids, plus the clip w that verdict branches on.
          //
          // A BBOX rather than a per-surface list or a per-slot frame-to-frame
          // diff: surface slots are frame-local and get reassigned on every
          // TLAS build, so diffing slot s against slot s last frame measures
          // the table reshuffling, not the geometry. A bbox over the same VS's
          // surfaces is identity-free and directly comparable across frames.
          //
          // cenSeen is the denominator - without it, a bbox of (0,0,0) from
          // "no surface was sampled" reads exactly like geometry sitting at
          // the world origin.
          uint32_t cenSeen = 0u;
          float cenMn[3] = { 1e30f, 1e30f, 1e30f };
          float cenMx[3] = { -1e30f, -1e30f, -1e30f };
          // NV-DXVK [CenSplit]: per-surface camera-distance DISTRIBUTION, and
          // how much of each part of it is on screen.
          //
          // WHY THE BBOX WAS NOT ENOUGH, and this is the correction it exists
          // to prevent repeating: the bbox CENTRE moved 9356 units between
          // frames where this VS rendered and frames where it did not, and that
          // was read as the geometry teleporting. It was not. cenMin is
          // identical to six figures on both - the far group never moves - and
          // only cenMax changes, because on rendering frames the VS ALSO draws
          // surfaces at the camera. A centre moves when the SET changes, and a
          // per-VS aggregate cannot tell that from motion. This VS is already
          // recorded as drawing both far 3D-skybox and near-camera content
          // (handoff section 4: "classify per-surface, never per-VS").
          //
          // A HISTOGRAM, NOT A NEAR/FAR CLASSIFIER. A threshold here would bake
          // in the very split under investigation and report whatever it was
          // told to; the buckets are fixed powers of four over camera distance
          // so the whole distribution is visible and the reader draws the line,
          // not the instrument.
          //
          // THE QUESTION IT ANSWERS: on a frame where nothing renders, is the
          // near bucket EMPTY (those surfaces were never submitted - the defect
          // is upstream, in submission) or POPULATED BUT NOT ON SCREEN (they
          // exist and something makes them invisible - the defect is placement
          // or visibility)? Those are different bugs and nothing measured so
          // far separates them.
          // enum, not `static constexpr`: RcVs is a function-local class and
          // C++17 forbids static data members in one (this target builds at
          // /std:c++17, meson.build:123). An enumerator is a constant
          // expression and is legal in a local class, so it still works as the
          // array bound without a second place to keep the count in sync.
          enum : uint32_t { kCenBuckets = 6u };
          uint32_t cenDistN[kCenBuckets] = { 0u, 0u, 0u, 0u, 0u, 0u };
          uint32_t cenDistOn[kCenBuckets] = { 0u, 0u, 0u, 0u, 0u, 0u };
          float clipWMn = 1e30f, clipWMx = -1e30f;
          uint32_t clipWNeg = 0u;   // surfaces with w <= 0, i.e. behind the camera
          // -1, not 0, for "not measurable": a VS that genuinely did not move
          // reads 0, and the two must not be the same value.
          float cenJump = -1.0f;    // distance this VS's bbox centre moved since the previous frame
          uint32_t flkTrans = 0u;      // render-layer on<->off changes
          uint32_t flkAsTrans = 0u;    // acceleration-structure-layer changes
          uint32_t flkViewTrans = 0u;  // frustum-layer changes (prOnScreen)
          uint32_t flkSurfTrans = 0u;  // surface-table-layer changes
          uint32_t flkOn = 0u, flkFrames = 0u;  // frames rendered / frames tracked
          uint32_t flkOffRunMax = 0u;  // longest consecutive run of gone frames
          uint32_t flkSinceTrans = 0u; // frames since the last render-layer change
        };
        std::unordered_map<uint64_t, RcVs> byVs;
        uint64_t totOrdSeen = 0ull, totOrdFinal = 0ull, totUnoSeen = 0ull, totContinued = 0ull;

        // NV-DXVK [CenSplit]: camera position, hoisted above the surface scan
        // so per-surface distance can be bucketed inside it. Same accessor and
        // the same freecam=false the header line uses further down, so the two
        // cannot disagree about where the camera was.
        const Vector3 cenCamPos =
          getSceneManager().getCamera().getPosition(false);

        for (uint32_t s = 0; s < scanRc; ++s) {
          const RtInstance* inst = reorderedRc[s];
          const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
          if (blas == nullptr) {
            continue;
          }
          const uint32_t a = covRc[baseOrdSeen + s];
          const uint32_t b = covRc[baseOrdFinal + s];
          const uint32_t c = covRc[baseUnoSeen + s];
          const uint32_t d = covRc[baseContinued + s];

          // NV-DXVK [CensusAlign]: attribute GPU-written COUNTS to the VS that
          // owned this slot in the frame the GPU consumed, not the VS that
          // owns it at readback time. The slot's frame comes from the packed
          // nibble in COVERAGE_OBS_IDENT_REGION (written by the same hits
          // being counted); the slot->vs map for that frame comes from the
          // ring filled in dispatchTlasProbe. Slots without an identity write
          // carry zero hit counts, so live attribution is harmless for them.
          // Live-state fields (surfaces, pi*, blas-state, inst*) stay on the
          // live owner — they describe current CPU state, which has no lag.
          const uint64_t vsLive = uint64_t(blas->input.getTransformData().vertexShaderHash);
          uint64_t vsForCounts = vsLive;
          {
            const uint32_t obsIdentA = covRc[baseObsIdent + s];
            if ((obsIdentA & 0x80000000u) != 0u) {
              const uint32_t nib = (obsIdentA >> 27) & 0xFu;
              for (uint32_t r = 0; r < kIdentRingSize; ++r) {
                const uint32_t fp1 = s_identExpectedRingFramePlus1[r];
                if (fp1 == 0u || (((fp1 - 1u) & 0xFu) != nib)) {
                  continue;
                }
                if (s < s_identVsRing[r].size() && s_identVsRing[r][s] != 0ull) {
                  vsForCounts = s_identVsRing[r][s];
                }
                break;
              }
            }
          }

          // NV-DXVK [ProbeAlign]: same frame-nibble join as vsForCounts, but
          // for the GPU-written TLASPROBE_*/SURFMAP regions, whose stamp is
          // written by the probe shader itself. probeRingRow is kept so the
          // occluder surface INDEX below can be resolved through the same
          // frame's snapshot — an index from the GPU names a slot in the
          // GPU's table, not in today's.
          uint64_t vsForProbe = vsLive;
          uint32_t probeRingRow = UINT32_MAX;
          {
            const uint32_t probeStamp = covRc[baseProbeFrame + s];
            if ((probeStamp & 0x80000000u) != 0u) {
              const uint32_t nibP = (probeStamp >> 27) & 0xFu;
              for (uint32_t r = 0; r < kIdentRingSize; ++r) {
                const uint32_t fp1 = s_identExpectedRingFramePlus1[r];
                if (fp1 == 0u || (((fp1 - 1u) & 0xFu) != nibP)) {
                  continue;
                }
                if (s < s_identVsRing[r].size() && s_identVsRing[r][s] != 0ull) {
                  vsForProbe = s_identVsRing[r][s];
                  probeRingRow = r;
                }
                break;
              }
            }
          }

          // Insert all keys BEFORE taking references — a later operator[]
          // could rehash and invalidate an earlier reference.
          byVs[vsForCounts];
          byVs[vsForProbe];
          byVs[vsLive];
          RcVs& eCounts = byVs[vsForCounts];
          RcVs& eProbe = byVs[vsForProbe];
          RcVs& e = byVs[vsLive];
          ++e.surfaces;
          eCounts.ordSeen += a; eCounts.ordFinal += b; eCounts.unoSeen += c; eCounts.continued += d;
          totOrdSeen += a; totOrdFinal += b; totUnoSeen += c; totContinued += d;

          const uint32_t pm = covRc[basePiwMask + s];
          const uint32_t pf = covRc[basePiwFlags + s];
          const uint32_t pd = covRc[basePiwDist + s];
          const uint32_t pb = covRc[basePiwBlasLo + s];
          if (pm == 0u) {
            ++e.piNotWritten;      // culling shader never ran for this slot
          } else if (pm == 1u) {
            ++e.piMaskZero;        // it ran and wrote mask 0
          } else {
            ++e.piMaskLive;        // entry carries a live mask
          }
          if (pf & COVERAGE_PIW_FLAG_DIST_NONFINITE) { ++e.piDistNonFinite; }
          if (pf & COVERAGE_PIW_FLAG_POS_NONFINITE)  { ++e.piPosNonFinite; }
          if (pf & COVERAGE_PIW_FLAG_BLASREF_ZERO)   { ++e.piBlasRefZero; }
          // Distance range only over slots the shader actually wrote; an
          // unwritten slot reads 0 and would otherwise drag the min to zero and
          // make every VS look like it has an instance sitting on the camera.
          if (pm != 0u && pd != 0xFFFFFFFFu) {
            if (pd < e.piDistMin) { e.piDistMin = pd; }
            if (pd > e.piDistMax) { e.piDistMax = pd; }
          }
          if (pb != 0u && e.piBlasLoAny == 0u) { e.piBlasLoAny = pb; }

          // NV-DXVK [BlasState]: what the entry's reference actually points at.
          const PooledBlas* pooled = blas->dynamicBlas.ptr();
          if (pooled == nullptr || pooled->accelStructure == nullptr) {
            ++e.blasNull;
          } else {
            const uint64_t cpuRef = pooled->accelerationStructureReference;
            if (cpuRef == 0ull) {
              ++e.blasRefZeroCpu;
            } else if (e.blasRefAny == 0ull) {
              e.blasRefAny = cpuRef;
            }
            // Only meaningful where the GPU actually wrote an entry for this
            // slot (pm != 0); an unwritten slot has pb == 0 and comparing it
            // would manufacture a mismatch for every non-PI surface.
            if (pm != 0u && cpuRef != 0ull
                && pb != static_cast<uint32_t>(cpuRef & 0xFFFFFFFFull)) {
              ++e.blasRefMismatch;
            }
            const uint32_t touchAge = (pooled->frameLastTouched == kInvalidFrameIndex)
              ? UINT32_MAX : (frameRc - pooled->frameLastTouched);
            if (touchAge != UINT32_MAX && touchAge > e.blasTouchAgeMax) {
              e.blasTouchAgeMax = touchAge;
            }
          }
          const uint32_t prims = blas->buildRanges.empty() ? 0u : blas->buildRanges[0].primitiveCount;
          if (prims == 0u) { ++e.blasPrimsZero; }
          if (prims < e.blasPrimsMin) { e.blasPrimsMin = prims; }
          if (prims > e.blasPrimsMax) { e.blasPrimsMax = prims; }
          const uint32_t updAge = (blas->frameLastUpdated == kInvalidFrameIndex)
            ? UINT32_MAX : (frameRc - blas->frameLastUpdated);
          if (updAge != UINT32_MAX && updAge > e.blasUpdateAgeMax) {
            e.blasUpdateAgeMax = updAge;
          }

          // NV-DXVK [RawHit]: hits traversal committed on this surface, before
          // resolve had a chance to reject them.
          // NV-DXVK [CensusAlign]: GPU count — frame-correct attribution.
          eCounts.rawHit += covRc[baseRawHit + s];

          // NV-DXVK [IdentProbe] v2 — FRAME-ALIGNED. v1 compared the observed
          // identity against LIVE CPU surfaces and read a constant ~87%
          // mismatch on healthy frames: the readback lags GPU execution by
          // frames-in-flight while the table reshuffles every frame, so the
          // live table is the wrong reference by construction. The GPU now
          // stamps each observation with its submission frame (region 100 =
          // frameId+1, carried in cb.enableResolveCensus) and the comparison
          // joins against that frame's snapshot from the ring filled in
          // dispatchTlasProbe. A mismatch after this join means the ray
          // consumed surface bytes that were never what the CPU submitted for
          // that slot IN THAT SAME FRAME — a real desync, not readback lag.
          {
            const uint32_t obsIdent = covRc[baseObsIdent + s];
            if ((obsIdent & 0x80000000u) != 0u) {
              const uint32_t obsFrameNibble = (obsIdent >> 27) & 0xFu;
              // Find the ring snapshot whose frame matches the observation's
              // nibble. Unique within the last 16 frames; the ring holds 8.
              for (uint32_t r = 0; r < kIdentRingSize; ++r) {
                const uint32_t fp1 = s_identExpectedRingFramePlus1[r];
                if (fp1 == 0u || (((fp1 - 1u) & 0xFu) != obsFrameNibble)) {
                  continue;
                }
                if (s >= s_identExpectedRing[r].size()) {
                  break;
                }
                const uint32_t expIdent = s_identExpectedRing[r][s];
                if (expIdent == 0u) {
                  break;
                }
                // NV-DXVK [CensusAlign]: GPU observation — frame-correct owner.
                ++eCounts.identSeen;
                // Compare identity bits only (mask out the frame nibble).
                const uint32_t kIdentMask = ~(0xFu << 27);
                if ((obsIdent & kIdentMask) != (expIdent & kIdentMask)) {
                  ++eCounts.identBad;
                  if (eCounts.identBadFirstSlot == UINT32_MAX) {
                    eCounts.identBadFirstSlot = s;
                    eCounts.identBadExpected = expIdent;
                    eCounts.identBadObserved = obsIdent;
                  }
                }
                break;
              }
            }
          }

          // NV-DXVK [SurfMap]: stored as value+1, so 0 means the probe never
          // wrote this slot. SURFACE_INDEX_INVALID is 0x1FFFFF (the 21-bit max,
          // NOT 0xFFFF) and unmapped entries arrive as int32_t(-1), which the
          // 21-bit setter truncates to that same 0x1FFFFF - so both forms are
          // checked rather than assuming which one shows up.
          // NV-DXVK [ProbeAlign]: SURFMAP is written by tlas_probe (GPU), so
          // it is attributed through the probe-frame owner like every other
          // probe region.
          const uint32_t smRaw = covRc[baseSurfMap + s];
          if (smRaw == 0u) {
            ++eProbe.surfMapUnwritten;
          } else {
            const uint32_t sm = smRaw - 1u;
            if (sm == 0x1FFFFFu || sm == 0xFFFFFFFFu) {
              ++eProbe.surfMapInvalid;
            } else if (sm != s) {
              // Maps somewhere other than itself. Expected for the temporal
              // last-frame -> this-frame use, so this is reported, not judged.
              ++eProbe.surfMapMismatch;
            }
          }

          // NV-DXVK [InstMask]: stored as mask+1 so an unwritten slot is
          // distinguishable from a genuine mask of 0 - and a mask of 0 is
          // exactly the failure worth catching, because such an instance is
          // traced by nothing while looking correct everywhere else.
          const uint32_t imRaw = covRc[baseInstMask + s];
          if (imRaw == 0u) {
            ++e.instMaskUnwritten;
          } else {
            const uint32_t im = imRaw - 1u;
            if (im == 0u) {
              ++e.instMaskZero;
            } else {
              e.instMaskAny |= im;
            }
          }

          // NV-DXVK [CamTris]: outside the RAN gate on purpose. The triangle
          // sampling has its own denominator and its own skip conditions, and
          // tying it to the face probe's gate would silently drop surfaces the
          // face probe declined but the camera sampling handled fine.
          const uint32_t triTested = covRc[baseTriTested + s];
          if (triTested > 0u) {
            ++eProbe.camTrisSurfaces;
            eProbe.camTrisTested += triTested;
            eProbe.camTrisReached += covRc[baseTriReached + s];
            eProbe.camTrisReachedNoCull += covRc[baseTriNoCull + s];
            eProbe.camTrisMissed += covRc[baseTriMissed + s];
            const uint32_t radius = covRc[baseMaxRadius + s];
            if (radius > eProbe.maxRadius) { eProbe.maxRadius = radius; }

            // NV-DXVK [Occluder] + [ProbeAlign]: the blocking surface index
            // names a slot in the GPU-frame table, so it is resolved through
            // that frame's ring snapshot when available — resolving it
            // through the live reordered table named whichever VS holds the
            // slot TODAY (the [CamProbe] ringVs measurement showed live and
            // GPU-frame owners agree on ~0% of slots). Live fallback only
            // when the ring has no snapshot for the probe's frame.
            // Bounds-checked rather than trusted: the index comes from the
            // GPU and a stale or out-of-range slot must not read a
            // neighbouring instance and name an innocent shader.
            const uint32_t occPlusOne = covRc[baseOccluder + s];
            if (occPlusOne != 0u) {
              const uint32_t occIdx = occPlusOne - 1u;
              ++eProbe.occludedSurfaces;
              uint64_t occVs = 0ull;
              if (probeRingRow != UINT32_MAX && occIdx < s_identVsRing[probeRingRow].size()) {
                occVs = s_identVsRing[probeRingRow][occIdx];
              } else if (probeRingRow == UINT32_MAX && occIdx < scanRc) {
                const RtInstance* occInst = reorderedRc[occIdx];
                const BlasEntry* occBlas = (occInst != nullptr) ? occInst->getBlas() : nullptr;
                if (occBlas != nullptr) {
                  occVs = uint64_t(occBlas->input.getTransformData().vertexShaderHash);
                }
              }
              if (occVs != 0ull) {
                ++eProbe.occluderVs[occVs];
              }
            }
          }

          // NV-DXVK [TlasProbe] + [ProbeAlign]: every field in this block is
          // GPU probe output, so it accumulates under the probe-frame owner.
          const uint32_t pr = covRc[baseProbeFlags + s];
          if (pr & COVERAGE_TLASPROBE_FLAG_RAN) {
            ++eProbe.prRan;
            // NV-DXVK [Centroid]: gated on RAN and nothing else. The probe
            // writes these BEFORE the w > 0 branch, so they are populated even
            // for geometry behind the camera - which is the case that has to
            // survive, because "off screen" has to be separable into "behind
            // the camera" and "in front and outside the bounds".
            {
              const float cx = u2fRaw(covRc[baseCenX + s]);
              const float cy = u2fRaw(covRc[baseCenY + s]);
              const float cz = u2fRaw(covRc[baseCenZ + s]);
              const float cw = u2fRaw(covRc[baseClipW + s]);
              // A non-finite centroid is a different finding and must not be
              // allowed to collapse the bbox to +/-inf and hide every real
              // value in it.
              if (std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cz)) {
                ++eProbe.cenSeen;
                eProbe.cenMn[0] = std::min(eProbe.cenMn[0], cx);
                eProbe.cenMn[1] = std::min(eProbe.cenMn[1], cy);
                eProbe.cenMn[2] = std::min(eProbe.cenMn[2], cz);
                eProbe.cenMx[0] = std::max(eProbe.cenMx[0], cx);
                eProbe.cenMx[1] = std::max(eProbe.cenMx[1], cy);
                eProbe.cenMx[2] = std::max(eProbe.cenMx[2], cz);

                // NV-DXVK [CenSplit]: bucket THIS surface by its distance from
                // the camera, and record whether it is on screen. Per surface,
                // not per VS - that distinction is the whole point of the
                // field. Bucket edges are fixed powers of four (250, 1k, 4k,
                // 16k, 64k) so the reader sees the distribution and no
                // near/far line is drawn by the instrument.
                const float ddx = cx - cenCamPos.x;
                const float ddy = cy - cenCamPos.y;
                const float ddz = cz - cenCamPos.z;
                const float cdist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                const uint32_t b =
                    (cdist <   250.0f) ? 0u
                  : (cdist <  1000.0f) ? 1u
                  : (cdist <  4000.0f) ? 2u
                  : (cdist < 16000.0f) ? 3u
                  : (cdist < 64000.0f) ? 4u
                                       : 5u;
                ++eProbe.cenDistN[b];
                if (pr & COVERAGE_TLASPROBE_FLAG_ONSCREEN) {
                  ++eProbe.cenDistOn[b];
                }
              }
              if (std::isfinite(cw)) {
                eProbe.clipWMn = std::min(eProbe.clipWMn, cw);
                eProbe.clipWMx = std::max(eProbe.clipWMx, cw);
                if (cw <= 0.0f) {
                  ++eProbe.clipWNeg;
                }
              }
            }
            if (pr & COVERAGE_TLASPROBE_FLAG_STRICT_HIT)  { ++eProbe.prStrictHit; }
            if (pr & COVERAGE_TLASPROBE_FLAG_STRICT_SELF) { ++eProbe.prStrictSelf; }
            if (pr & COVERAGE_TLASPROBE_FLAG_ANY_SELF)    { ++eProbe.prAnySelf; }
            if (pr & COVERAGE_TLASPROBE_FLAG_ANY_HIT) {
              ++eProbe.prAnyHit;
            } else {
              // Nothing anywhere along the ray, with no mask and no culling.
              ++eProbe.prAnyMiss;
            }
            if (pr & COVERAGE_TLASPROBE_FLAG_ONSCREEN) {
              ++eProbe.prOnScreen;
              if (pr & COVERAGE_TLASPROBE_FLAG_ANY_SELF) {
                ++eProbe.prOnScreenSelf;
              }
            }

            // NV-DXVK [CamProbe]
            const uint32_t cdA = covRc[baseCamDotA + s];
            const uint32_t cdB = covRc[baseCamDotB + s];
            if (pr & COVERAGE_TLASPROBE_FLAG_CAMPROJ_RAN) {
              ++eProbe.camProjRan;
              if (cdA > eProbe.camDotAMax) { eProbe.camDotAMax = cdA; }
              if (cdB > eProbe.camDotBMax) { eProbe.camDotBMax = cdB; }
            }

            // The decisive population, and the only one these are read on: the
            // surface is on screen AND the face probe self-hits it, i.e. it is
            // provably there and provably in frame. Every other surface can be
            // legitimately absent from the primary ray for ordinary reasons and
            // would only dilute the number.
            const bool onScreenSelf =
              (pr & COVERAGE_TLASPROBE_FLAG_ONSCREEN) && (pr & COVERAGE_TLASPROBE_FLAG_ANY_SELF);
            if (onScreenSelf) {
              if (pr & COVERAGE_TLASPROBE_FLAG_CAMPROJ_RAN) {
                if (cdA > eProbe.camDotAMaxOnScreenSelf) { eProbe.camDotAMaxOnScreenSelf = cdA; }
                if (cdB > eProbe.camDotBMaxOnScreenSelf) { eProbe.camDotBMaxOnScreenSelf = cdB; }
              }
              if (pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_SELF) {
                ++eProbe.camRaySelf;
              } else if (pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_HIT) {
                ++eProbe.camRayOther;
              } else {
                ++eProbe.camRayMiss;
              }
              if (pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_SELF) {
                ++eProbe.camRayAnySelf;
                if (!(pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_SELF)) {
                  ++eProbe.camRayCulled;
                }
              }
              const uint32_t cdist = covRc[baseCamDist + s];
              if (cdist < eProbe.camDistMin) { eProbe.camDistMin = cdist; }
              if (cdist > eProbe.camDistMax) { eProbe.camDistMax = cdist; }
            }
          }
        }

        // NV-DXVK [XfMismatch]: the transform-divergence probe. Every
        // instance feeds the GPU TWO transforms: surface.objectToWorld
        // (packed into the surface buffer — what resolve shades with and
        // what the tlas_probe aims its self-ray through) and
        // m_vkInstance.transform (written into the TLAS instance entry —
        // where the rays actually find the geometry). If they disagree, the
        // mesh renders nowhere the surface claims to be: on screen per the
        // record, unreachable by a ray through its own triangle, sky behind
        // — the 2026-08-03 flicker signature, WITHOUT any content
        // corruption. (Capture-ordered bakes were proven landing on the
        // victims while the signature persisted unchanged — content is
        // exonerated; this is the remaining suspect.)
        // CPU-only and same-frame: both fields are read from the same
        // RtInstance in one walk — no readback lag, no attribution ring.
        // VkTransformMatrixKHR is row-major 3x4, Matrix4 is column-major:
        // vk[r][c] corresponds to o2w[c][r].
        {
          // PI slots' TLAS entries are written by the point-instancer path,
          // not necessarily from m_vkInstance — so their comparison is
          // reported SEPARATELY. A large piMismatch with clean nonPi says
          // "vkInstance is not the PI entry source; compare against the PI
          // placement records instead", not "every PI instance is displaced".
          uint32_t xfChecked = 0u, xfMismatchNonPi = 0u, xfMismatchPi = 0u;
          float xfWorstDelta = 0.0f;
          uint32_t xfWorstSurf = 0u;
          uint64_t xfWorstVs = 0ull;
          bool xfWorstIsPi = false;
          int xfRawBudget = 6;
          for (uint32_t s = 0; s < scanRc; ++s) {
            const RtInstance* instXf = reorderedRc[s];
            const BlasEntry* blasXf = (instXf != nullptr) ? instXf->getBlas() : nullptr;
            if (blasXf == nullptr) {
              continue;
            }
            const bool slotIsPi = covRc[basePiwMask + s] != 0u;
            const VkTransformMatrixKHR& vkXf = instXf->getVkInstance().transform;
            const Matrix4& sfXf = instXf->surface.objectToWorld;
            float dMax = 0.0f;
            for (uint32_t r = 0; r < 3; ++r) {
              for (uint32_t c = 0; c < 4; ++c) {
                const float d = std::abs(vkXf.matrix[r][c] - sfXf[c][r]);
                if (d > dMax) { dMax = d; }
              }
            }
            ++xfChecked;
            if (dMax > xfWorstDelta) {
              xfWorstDelta = dMax;
              xfWorstSurf = s;
              xfWorstVs = uint64_t(blasXf->input.getTransformData().vertexShaderHash);
              xfWorstIsPi = slotIsPi;
            }
            // 1.0 world units: far below the artifact scale (an instance
            // parked elsewhere), far above float noise on 30k-unit coords.
            if (dMax > 1.0f) {
              if (slotIsPi) { ++xfMismatchPi; } else { ++xfMismatchNonPi; }
              if (xfRawBudget > 0) {
                --xfRawBudget;
                Logger::info(str::format(
                  "[XfMismatch]   f=", frameRc,
                  " surf=", s,
                  " pi=", slotIsPi ? 1 : 0,
                  " vs=0x", std::hex, uint64_t(blasXf->input.getTransformData().vertexShaderHash), std::dec,
                  " surfT=(", sfXf[3][0], ",", sfXf[3][1], ",", sfXf[3][2], ")",
                  " tlasT=(", vkXf.matrix[0][3], ",", vkXf.matrix[1][3], ",", vkXf.matrix[2][3], ")",
                  " dMax=", dMax));
              }
            }
          }
          // One aggregate line per frame, ALWAYS (raw-data-first): maxDelta
          // on a clean frame measures the noise floor, so "no mismatch" is a
          // measurement rather than an absence of lines.
          Logger::info(str::format("[XfMismatch] f=", frameRc,
            " checked=", xfChecked,
            " mismatchNonPi=", xfMismatchNonPi,
            " mismatchPi=", xfMismatchPi,
            " maxDelta=", xfWorstDelta,
            " worstSurf=", xfWorstSurf,
            " worstPi=", xfWorstIsPi ? 1 : 0,
            " worstVs=0x", std::hex, xfWorstVs, std::dec));
        }

        // NV-DXVK [InstDiag]: walk the instance manager's own table, not the
        // reordered surface list, and ask which live instances made it into
        // this frame's TLAS and which of them moved since the last frame.
        //
        // Entirely CPU-side - no coverage region, no shader, no dispatch
        // ordering to get wrong. Three of the last four instrument bugs were
        // plumbing races between a CPU write and a GPU read; this reads data
        // that is already sitting in memory at readback time.
        //
        // Keyed by RtInstance::getId(), NOT by pointer. Instances are pooled and
        // a freed slot is reused, so a pointer that matches across frames can be
        // a different instance - and this fork has already lost a debugging
        // session to dereferencing a stale RtInstance. Nothing here dereferences
        // a stored pointer: positions are copied by value and only ids persist.
        {
          // id -> (position, frame it was seen). Pruned every frame so it cannot
          // grow without bound over a long capture.
          static std::unordered_map<uint64_t, std::pair<Vector3, uint32_t>> s_instPrev;

          // Shares the raw-line option and cap with [CamProbe] - one knob for
          // "raw per-item lines", one budget, so a pathological frame cannot
          // turn either dump into the frame's bottleneck.
          int rawInstBudget = RtxOptions::logResolveCensusRaw()
            ? RtxOptions::logResolveCensusRawPerFrame() : 0;

          std::unordered_set<uint64_t> inTlas;
          inTlas.reserve(reorderedRc.size());
          for (const RtInstance* inst : reorderedRc) {
            if (inst != nullptr) {
              inTlas.insert(inst->getId());
            }
          }

          const auto& liveInstances = getSceneManager().getInstanceManager().getInstanceTable();
          for (const RtInstance* inst : liveInstances) {
            if (inst == nullptr) {
              continue;
            }
            const BlasEntry* blas = inst->getBlas();
            if (blas == nullptr) {
              continue;
            }
            const uint64_t vsHash = uint64_t(blas->input.getTransformData().vertexShaderHash);
            // Only VS the census already knows about. A VS with live instances
            // but no surfaces this frame has no census line to attach to, and
            // inventing one here would change what "distinctVS" counts.
            const auto it = byVs.find(vsHash);
            if (it == byVs.end()) {
              continue;
            }
            RcVs& e = it->second;

            ++e.instLive;
            const uint64_t id = inst->getId();
            const bool isInTlas = inTlas.count(id) != 0u;
            if (isInTlas) {
              ++e.instInTlas;
            }
            if (inst->censusMask() == 0u) {
              ++e.instMaskLive0;
            }

            const Vector3 pos = inst->getWorldPosition();

            // Raw per-instance lines for the VS that recorded nothing this
            // frame. The verdict fields are already final here - they come from
            // the surface loop above - so this needs no second pass.
            //
            // Per instance, not per VS: the aggregate is what let five
            // hypotheses survive this investigation, and "one of 33 instances
            // left the TLAS" is invisible in a mean but is the entire bug.
            if (rawInstBudget > 0
                && e.ordSeen == 0ull && e.ordFinal == 0ull && e.unoSeen == 0ull) {
              const auto prevIt = s_instPrev.find(id);
              const bool hasPrev = (prevIt != s_instPrev.end())
                                && (prevIt->second.second + 1u == frameRc);
              Logger::info(str::format(
                "[InstDiag] f=", frameRc,
                " vs=0x", std::hex, vsHash, std::dec,
                " id=", id,
                " inTlas=", (isInTlas ? 1 : 0),
                " mask=0x", std::hex, inst->censusMask(), std::dec,
                " pos=(", pos.x, ",", pos.y, ",", pos.z, ")",
                " prevPos=", (hasPrev
                  ? str::format("(", prevIt->second.first.x, ",",
                                     prevIt->second.first.y, ",",
                                     prevIt->second.first.z, ")")
                  : std::string("none")),
                " moved=", (hasPrev
                  ? str::format(length(pos - prevIt->second.first))
                  : std::string("n/a"))));
              --rawInstBudget;
            }
            const auto prev = s_instPrev.find(id);
            // Only the immediately previous frame counts. Comparing against an
            // older sighting would report a legitimately moving object that was
            // absent for a few frames as a teleport.
            if (prev != s_instPrev.end() && prev->second.second + 1u == frameRc) {
              const Vector3 d = pos - prev->second.first;
              const float dist = length(d);
              if (dist > 0.0f) {
                ++e.instMoved;
                if (dist > e.instMoveMax) {
                  e.instMoveMax = dist;
                }
              }
            }
            s_instPrev[id] = { pos, frameRc };
          }

          // Drop anything not seen for a few frames. Bounded work, and it keeps
          // the map the size of the live scene rather than the whole session.
          if ((frameRc & 0x3Fu) == 0u) {
            for (auto it2 = s_instPrev.begin(); it2 != s_instPrev.end();) {
              it2 = (frameRc - it2->second.second > 4u) ? s_instPrev.erase(it2) : std::next(it2);
            }
          }
        }

        // NV-DXVK [FlickerTrack]: per-VS state history, and the events where it
        // changes.
        //
        // WHY THIS EXISTS. The reported defect is intermittent - groups of
        // meshes vanish for seconds and flash back - and every field this census
        // had described a SINGLE frame. On a single frame, geometry that has
        // been absent for the entire session and geometry that vanished eight
        // frames ago are the same reading, so a per-frame signature cannot tell
        // an intermittent fault from a permanent one. It has already selected
        // the wrong population once: the "NOTRAVERSED and on screen and not
        // self-hittable" signature fired on ~9% of lines and was read as the
        // flicker, when it was four VS that read that way on 100% of 1314
        // consecutive frames - a constant, and therefore provably not the thing
        // that comes back.
        //
        // A transition count settles that in one number per line. flkTrans>0 is
        // an intermittent fault; flkTrans==0 with flkOn==0 is a permanent one;
        // flkTrans==0 with flkOn==flkFrames is healthy. Nothing else on the line
        // distinguishes the three.
        //
        // THREE LAYERS, because "it vanished" has three different causes and
        // they need different fixes:
        //   SURFACE - the VS stopped producing surfaces at all. There is no
        //             census line on those frames, so no per-line field could
        //             ever report it; only a tracker that remembers the VS
        //             across the gap can. Cause is upstream of the accel
        //             manager entirely (draw stopped arriving / instance reaped).
        //   AS      - surfaces exist, but the face probe stopped self-hitting:
        //             the geometry left the built acceleration structure.
        //   VIEW    - in the structure, but nothing projects inside the frustum
        //             any more. Added 2026-08-03 after the first run with this
        //             tracker: 0x29d5f7de0ba76c66 produced 50 events, all
        //             classified RENDER, while prOnScreen was in fact flipping
        //             in perfect lockstep with ordSeen (26 frames both, 1201
        //             neither, ZERO rendered-while-off-screen). Without this
        //             layer every frustum failure is mislabelled as a resolve
        //             failure, which is a different file and a different bug.
        //   RENDER  - present in the structure, inside the frustum, and still
        //             not resolved by the primary ray.
        // Reported outermost-first: a VS that loses its surfaces also leaves the
        // structure and stops rendering, and three events for one cause is how a
        // log stops being readable.
        //
        // Cost: one hash lookup and a handful of integer compares per VS per
        // frame (~25 VS), plus a line only when something actually changes. It
        // is not gated separately - it is strictly cheaper than the census line
        // it annotates. There is deliberately no periodic roll-up: the running
        // totals ride every EVENT line and every census line, so a ranking is a
        // sort over `flkTrans=` rather than a second log format to maintain.
        {
          struct FlkVs {
            // 2 = never observed. Distinct from 0 so the first frame of a VS
            // cannot be counted as a transition into it.
            uint8_t lastSurf = 2u, lastAs = 2u, lastView = 2u, lastRender = 2u;
            uint32_t surfTrans = 0u, asTrans = 0u, viewTrans = 0u, trans = 0u;
            uint32_t on = 0u, frames = 0u;
            uint32_t offRun = 0u, offRunMax = 0u;
            uint32_t lastTransFrame = 0u;
            uint32_t lastLineFrame = 0u;
            uint32_t absentRun = 0u;
            // NV-DXVK [Centroid]: previous frame's centroid bbox centre, so the
            // event line can carry how far the geometry moved into or out of
            // view. Valid only when hasPrevCen - a zeroed centre and a genuine
            // one at the world origin are otherwise the same reading.
            bool hasPrevCen = false;
            float prevCen[3] = { 0.0f, 0.0f, 0.0f };
            // Last computed jump, kept so the per-line field can report the
            // same number the event line does. -1 = not measurable.
            float lastJump = -1.0f;
          };
          static std::unordered_map<uint64_t, FlkVs> s_flk;
          static uint32_t s_flkLastFrame = UINT32_MAX;

          // asState is -1 for "not observed this frame" (the face probe did not
          // run). An unrun probe reads identically to an absent one, and letting
          // it set the state would invent two AS transitions for every gap in
          // probe coverage - the same class of error as reading piNotWritten on
          // a non-point-instancer surface.
          auto flkStep = [&](uint64_t vs, FlkVs& t, int surfState, int asState,
                             int viewState, int renderState, uint32_t surfaces,
                             uint64_t ordSeen, uint32_t prAnySelf, uint32_t prRan,
                             uint32_t cenSeen, const float* cenCentre,
                             float clipWMn, float clipWMx, uint32_t clipWNeg) {
            ++t.frames;
            if (renderState > 0) {
              ++t.on;
              t.offRun = 0u;
            } else {
              ++t.offRun;
              t.offRunMax = std::max(t.offRunMax, t.offRun);
            }

            // NV-DXVK [Centroid]: how far this VS's centroid bbox centre moved
            // since the previous frame it was sampled on. This is the number
            // that decides the open question - the instance TRANSLATION does
            // not move (instMoved=0 on every frame, flash included), but the
            // ONSCREEN test runs on the centroid, and a rotation, a scale, or a
            // sub-view reprojection moves the centroid without moving the
            // translation at all.
            float jump = -1.0f;   // -1 = not measurable this frame
            if (cenSeen > 0u && cenCentre != nullptr) {
              if (t.hasPrevCen) {
                const float dx = cenCentre[0] - t.prevCen[0];
                const float dy = cenCentre[1] - t.prevCen[1];
                const float dz = cenCentre[2] - t.prevCen[2];
                jump = std::sqrt(dx * dx + dy * dy + dz * dz);
              }
              t.prevCen[0] = cenCentre[0];
              t.prevCen[1] = cenCentre[1];
              t.prevCen[2] = cenCentre[2];
              t.hasPrevCen = true;
            }
            t.lastJump = jump;

            const char* layer = nullptr;
            const char* dir = nullptr;
            if (t.lastSurf != 2u && surfState != int(t.lastSurf)) {
              ++t.surfTrans;
              layer = "SURFACE";
              dir = (surfState != 0) ? "BACK" : "GONE";
            } else if (asState >= 0 && t.lastAs != 2u && asState != int(t.lastAs)) {
              ++t.asTrans;
              layer = "AS";
              dir = (asState != 0) ? "BACK" : "GONE";
            } else if (viewState >= 0 && t.lastView != 2u && viewState != int(t.lastView)) {
              ++t.viewTrans;
              layer = "VIEW";
              dir = (viewState != 0) ? "BACK" : "GONE";
            }
            // Counted whatever layer is reported: flkTrans is the headline
            // number and it must mean "changed visible state", not "changed
            // visible state for a reason no outer layer explains".
            if (t.lastRender != 2u && renderState != int(t.lastRender)) {
              ++t.trans;
              if (layer == nullptr) {
                layer = "RENDER";
                dir = (renderState != 0) ? "BACK" : "GONE";
              }
            }

            if (layer != nullptr) {
              // heldFor: how long the state being left had lasted. On the first
              // event for a VS there is no previous transition to measure from,
              // so the tracked lifetime is the honest answer.
              const uint32_t heldFor = (t.lastTransFrame == 0u)
                ? t.frames : (frameRc - t.lastTransFrame);
              Logger::info(str::format(
                "[FlickerTrack] EVENT f=", frameRc,
                " vs=0x", std::hex, vs, std::dec,
                " ", dir, " layer=", layer,
                " heldFor=", heldFor,
                " surfaces=", surfaces,
                " ordSeen=", ordSeen,
                " prAnySelf=", prAnySelf, "/", prRan,
                " trans=", t.trans,
                " asTrans=", t.asTrans,
                " viewTrans=", t.viewTrans,
                " surfTrans=", t.surfTrans,
                " on=", t.on, "/", t.frames,
                " offRunMax=", t.offRunMax,
                // NV-DXVK [Centroid]: the state of the geometry AT the event.
                // cenJump is the whole point of the line on a VIEW event: a
                // large jump means the geometry moved out of frame, a jump of
                // ~0 means it did not and the projection or the camera basis
                // did. clipW separates "behind the camera" from "in front and
                // outside the bounds".
                " cenSeen=", cenSeen,
                " cenCentre=", (cenSeen > 0u && cenCentre != nullptr
                  ? str::format("(", cenCentre[0], ",", cenCentre[1], ",", cenCentre[2], ")")
                  : std::string("none")),
                " cenJump=", (jump < 0.0f ? std::string("n/a") : str::format(jump)),
                " clipW=", (cenSeen > 0u ? str::format(clipWMn, "..", clipWMx)
                                         : std::string("none")),
                " clipWNeg=", clipWNeg));
              t.lastTransFrame = frameRc;
            }

            t.lastSurf = uint8_t(surfState);
            if (asState >= 0) {
              t.lastAs = uint8_t(asState);
            }
            if (viewState >= 0) {
              t.lastView = uint8_t(viewState);
            }
            t.lastRender = uint8_t(renderState);
          };

          // Guarded so a census emitted twice for one frame cannot double-count
          // a transition and manufacture a flicker out of a logging artifact.
          if (frameRc != s_flkLastFrame) {
            s_flkLastFrame = frameRc;

            for (auto& kv : byVs) {
              const RcVs& e = kv.second;
              // byVs also holds keys inserted purely as ident-ring/probe-frame
              // aliases. Those are not drawn VS and stepping them would report
              // a flicker for a VS that never had geometry.
              if (e.surfaces == 0u) {
                continue;
              }
              FlkVs& t = s_flk[kv.first];
              t.lastLineFrame = frameRc;
              t.absentRun = 0u;
              // Centre of the centroid bbox, or null when nothing was sampled.
              float cenCentre[3];
              const bool haveCen = (e.cenSeen > 0u);
              if (haveCen) {
                for (int c = 0; c < 3; ++c) {
                  cenCentre[c] = 0.5f * (e.cenMn[c] + e.cenMx[c]);
                }
              }
              flkStep(kv.first, t, 1,
                      (e.prRan == 0u) ? -1 : ((e.prAnySelf > 0u) ? 1 : 0),
                      (e.prRan == 0u) ? -1 : ((e.prOnScreen > 0u) ? 1 : 0),
                      (e.ordSeen > 0ull) ? 1 : 0,
                      e.surfaces, e.ordSeen, e.prAnySelf, e.prRan,
                      e.cenSeen, haveCen ? cenCentre : nullptr,
                      e.clipWMn, e.clipWMx, e.clipWNeg);
            }

            // VS tracked earlier that produced no surfaces at all this frame.
            // This is the loudest way geometry disappears and the one a per-line
            // field structurally cannot report, because there is no line.
            for (auto& kv : s_flk) {
              FlkVs& t = kv.second;
              if (t.lastLineFrame == frameRc || t.lastSurf == 2u) {
                continue;
              }
              ++t.absentRun;
              flkStep(kv.first, t, 0, 0, 0, 0, 0u, 0ull, 0u, 0u,
                      0u, nullptr, 0.0f, 0.0f, 0u);
            }

            // Forget VS that have been gone long enough to be a scene change
            // rather than a flicker (level load, menu shader). Without this the
            // map charges every retired VS an offRun forever. This is a map-size
            // bound, not a log throttle - it emits nothing.
            for (auto it = s_flk.begin(); it != s_flk.end();) {
              it = (it->second.absentRun > 3600u) ? s_flk.erase(it) : std::next(it);
            }
          }

          // Copy out unconditionally, so the per-line fields are correct even on
          // a repeat emit that the advance guard skipped.
          for (auto& kv : byVs) {
            const auto itFlk = s_flk.find(kv.first);
            if (itFlk == s_flk.end()) {
              continue;
            }
            const FlkVs& t = itFlk->second;
            RcVs& e = kv.second;
            e.flkTrans = t.trans;
            e.flkAsTrans = t.asTrans;
            e.flkViewTrans = t.viewTrans;
            e.flkSurfTrans = t.surfTrans;
            e.flkOn = t.on;
            e.flkFrames = t.frames;
            e.flkOffRunMax = t.offRunMax;
            e.flkSinceTrans = (t.lastTransFrame == 0u) ? t.frames
                                                       : (frameRc - t.lastTransFrame);
            e.cenJump = t.lastJump;
          }
        }

        // Camera state on the header line (handoff V5 §10.2). Every conclusion
        // this census has produced about "gone frames" rests on the camera
        // having been still, and nothing in 85 MB of previous capture can
        // confirm it - a NOTRAVERSED frame was indistinguishable from the user
        // looking away. freecam=false to report the camera the game is actually
        // rendering from, which is the one the primary rays are built from.
        const RtCamera& censusCam = getSceneManager().getCamera();
        const Vector3 censusCamPos = censusCam.getPosition(false);
        const Vector3 censusCamDir = censusCam.getDirection(false);

        Logger::info(str::format(
          "[ResolveCensus] === f=", frameRc,
          " distinctVS=", byVs.size(),
          " orderedSize=", reorderedRc.size(),
          " ordSeen=", totOrdSeen,
          " ordFinal=", totOrdFinal,
          " unoSeen=", totUnoSeen,
          " continued=", totContinued,
          " camPos=(", censusCamPos.x, ",", censusCamPos.y, ",", censusCamPos.z, ")",
          " camDir=(", censusCamDir.x, ",", censusCamDir.y, ",", censusCamDir.z, ")",
          " ==="));

        // Sorted by ordSeen desc: the geometry the resolver touches most is the
        // geometry whose ordFinal==0 is most surprising, so it reads first.
        std::vector<std::pair<uint64_t, RcVs>> rankRc(byVs.begin(), byVs.end());
        std::sort(rankRc.begin(), rankRc.end(),
          [](const std::pair<uint64_t, RcVs>& x, const std::pair<uint64_t, RcVs>& y) {
            return x.second.ordSeen > y.second.ordSeen;
          });

        // NV-DXVK [Occluder]: the single biggest blocker for a VS, as
        // "0xhash xN", or "none". Reported as one entry plus its share rather
        // than the whole tally, because a line that lists every blocker is a
        // line nobody reads - and if the cause is one object, one entry names
        // it. occludedSurf alongside gives the denominator.
        auto occluderTop = [](const RcVs& e) -> std::string {
          uint64_t bestVs = 0ull;
          uint32_t bestN = 0u;
          for (const auto& kv : e.occluderVs) {
            if (kv.second > bestN) {
              bestN = kv.second;
              bestVs = kv.first;
            }
          }
          if (bestN == 0u) {
            return std::string("none");
          }
          return str::format("0x", std::hex, bestVs, std::dec, " x", bestN,
                             " of", e.occluderVs.size(), "distinct");
        };

        // [PIWrite] suffix, appended to every line so a NOTRAVERSED verdict and
        // the state of its TLAS entries are always read together. piDist is
        // omitted when no slot was written, rather than printed as a sentinel.
        // Captures occluderTop by reference - a non-capturing lambda cannot see
        // it, and the two are kept separate because the occluder tally needs a
        // loop that would otherwise sit awkwardly inside a format call.
        auto piSuffix = [&occluderTop](const RcVs& e) -> std::string {
          return str::format(
            " piNotWritten=", e.piNotWritten,
            " piMaskZero=", e.piMaskZero,
            " piMaskLive=", e.piMaskLive,
            " piDistNaN=", e.piDistNonFinite,
            " piPosNaN=", e.piPosNonFinite,
            " piBlasRef0=", e.piBlasRefZero,
            " piDist=", (e.piDistMin == UINT32_MAX
              ? std::string("none")
              : str::format(e.piDistMin, "..", e.piDistMax)),
            // piBlasLo was collected but never emitted in the previous build,
            // which is why the log could not say whether the entry's BLAS
            // address changes between render and vanish frames. It is the
            // cheapest discriminator between "stale acceleration structure" and
            // "TLAS build skipped it", so it is printed next to the CPU-side
            // reference it should equal.
            " piBlasLo=", str::format("0x", std::hex, e.piBlasLoAny, std::dec),
            " cpuBlasRef=", str::format("0x", std::hex, e.blasRefAny, std::dec),
            " blasRefMismatch=", e.blasRefMismatch,
            " blasNull=", e.blasNull,
            " blasRef0cpu=", e.blasRefZeroCpu,
            " blasPrims=", (e.blasPrimsMin == UINT32_MAX
              ? std::string("none")
              : str::format(e.blasPrimsMin, "..", e.blasPrimsMax)),
            " blasPrims0=", e.blasPrimsZero,
            " blasTouchAge=", e.blasTouchAgeMax,
            " blasUpdAge=", e.blasUpdateAgeMax,
            // [TlasProbe]. prAnyMiss is the one to read first on a NOTRAVERSED
            // line: it counts surfaces that a mask-0xFF, no-cull ray query
            // could not find anywhere along its path. Those are absent from the
            // structure the driver built.
            " prRan=", e.prRan,
            " prAnyHit=", e.prAnyHit,
            " prAnyMiss=", e.prAnyMiss,
            " prAnySelf=", e.prAnySelf,
            " prStrictHit=", e.prStrictHit,
            " prStrictSelf=", e.prStrictSelf,
            " prOnScreen=", e.prOnScreen,
            " prOnScreenSelf=", e.prOnScreenSelf,
            // NV-DXVK [CamProbe]. Read camDotA/camDotB FIRST, and calibrate
            // before concluding: on a WINS_PRIMARY line the geometry is
            // demonstrably rendering, so whichever of the two reads ~0 there is
            // the correct NDC Y convention and the other is meaningless. Then
            // read that same one on NOTRAVERSED lines. Non-zero there means the
            // projection that says "on screen" and the basis the primary ray is
            // built from disagree - the camera is not self-consistent, and that
            // is the defect.
            //
            // ossX suffix = restricted to on-screen self-hittable surfaces.
            // Those are the only surfaces provably in frame and provably in the
            // structure, so they are the only ones whose miss needs explaining.
            " camProjRan=", e.camProjRan,
            " camDotA=", e.camDotAMax,
            " camDotB=", e.camDotBMax,
            " camDotAoss=", e.camDotAMaxOnScreenSelf,
            " camDotBoss=", e.camDotBMaxOnScreenSelf,
            // camRaySelf > 0 on a NOTRAVERSED line is the sharpest single
            // number this census can produce: a ray from the real camera
            // origin, with the primary pass's own mask and ray flags, reached
            // geometry that the primary pass recorded zero interactions with on
            // the same frame in the same command buffer against the same TLAS.
            " camRaySelf=", e.camRaySelf,
            " camRayOther=", e.camRayOther,
            " camRayMiss=", e.camRayMiss,
            " camRayAnySelf=", e.camRayAnySelf,
            " camRayCulled=", e.camRayCulled,
            " camDist=", (e.camDistMin == UINT32_MAX
              ? std::string("none")
              : str::format(e.camDistMin, "..", e.camDistMax)),
            // NV-DXVK [CamTris]: THE line to read. camTrisReached > 0 on a
            // NOTRAVERSED line means part of this surface is the frontmost
            // thing along a camera ray - it owns pixels on screen - and the
            // resolver still recorded no interaction with it anywhere on the
            // frame. Calibrate on WINS_PRIMARY first: that is the ceiling this
            // sampling can report, and a NOTRAVERSED reading is only meaningful
            // against it.
            " camTrisSurf=", e.camTrisSurfaces,
            " camTrisTested=", e.camTrisTested,
            " camTrisReached=", e.camTrisReached,
            // NV-DXVK [RawHit] - READ THIS FIRST on a NOTRAVERSED line.
            // rawHit is what traversal committed; ordSeen is what survived
            // resolveVertex. rawHit > 0 with ordSeen == 0 means the ray found
            // the geometry and the resolve stage discarded it, which is a
            // completely different bug from the ray never arriving - and the
            // two have been indistinguishable for this entire investigation.
            " rawHit=", e.rawHit,
            // NV-DXVK [IdentProbe] — on a dropout line read identBad FIRST.
            // identBad > 0 says hits consumed a surface entry whose identity
            // differs from what the CPU uploaded this frame: the ray pass is
            // one frame out of phase with the surface table. identBad == 0
            // with the VS still at ordSeen=0 exonerates the surface buffer
            // and moves the staleness to the TLAS instance data itself.
            " identSeen=", e.identSeen,
            " identBad=", e.identBad,
            " identBadSlot=", (e.identBadFirstSlot == UINT32_MAX
              ? std::string("none")
              : str::format(e.identBadFirstSlot)),
            " identExp=0x", std::hex, e.identBadExpected,
            " identObs=0x", e.identBadObserved, std::dec,
            " surfMapUnwritten=", e.surfMapUnwritten,
            " surfMapInvalid=", e.surfMapInvalid,
            " surfMapMismatch=", e.surfMapMismatch,
            " instMask0=", e.instMaskZero,
            " instMaskUnwritten=", e.instMaskUnwritten,
            " instMaskAny=0x", std::hex, e.instMaskAny, std::dec,
            // NV-DXVK [InstDiag]. On a NOTRAVERSED line read instLive against
            // instInTlas first: if instances exist that the TLAS build did not
            // include, the geometry is untraceable for reasons that have
            // nothing to do with rays, and every probe in this file would still
            // report the structure as healthy. instMoveMax then says whether
            // the ones that ARE in the TLAS jumped since the previous frame.
            " instLive=", e.instLive,
            " instInTlas=", e.instInTlas,
            " instMissing=", (e.instLive > e.instInTlas) ? (e.instLive - e.instInTlas) : 0u,
            " instMaskLive0=", e.instMaskLive0,
            " instMoved=", e.instMoved,
            " instMoveMax=", e.instMoveMax,
            // NV-DXVK [Occluder]: the top blocker, by how many of this VS's
            // surfaces it stopped. On a NOTRAVERSED line where instances are
            // all present and unmoved, this names what is standing in front.
            " occludedSurf=", e.occludedSurfaces,
            " occluderTop=", occluderTop(e),
            // NV-DXVK [NoCull]: THE comparison. camTrisReached carries the
            // primary pass's cull flag; camTrisNoCull is the identical ray
            // without it. NoCull >> Reached on a dropout line means backface
            // rejection is discarding geometry that is present and in front,
            // and nothing else in this census could have seen that. camTrisMiss
            // is the remainder - rays that found nothing at all where the
            // surface buffer says geometry is.
            " camTrisNoCull=", e.camTrisReachedNoCull,
            " camTrisMiss=", e.camTrisMissed,
            // NV-DXVK [Spike]: geometric extent about the instance's own
            // origin, so it is immune to the object moving. Compare an
            // OCCLUDER's value on the frames the flicker VS drops against its
            // value on normal frames: a jump of hundreds of units with an
            // unchanged transform means the mesh deformed, which is the only
            // remaining way geometry covers more screen without moving.
            " maxRadius=", e.maxRadius,
            // NV-DXVK [FlickerTrack] - READ flkTrans BEFORE any other field on
            // this line when the question is the intermittent dropout.
            //
            // Every other field here is a snapshot, and a snapshot cannot say
            // whether what it is describing is a fault that comes and goes or a
            // condition that has held all session. These four can, and they are
            // the difference between chasing the flicker and chasing a constant
            // that merely looks like it on any single frame:
            //   flkTrans>0                      -> intermittent. THIS is the bug.
            //   flkTrans=0 and flkOn=0/N        -> permanent absence. Real, but
            //                                      not the flicker: it never
            //                                      came back to be seen going.
            //   flkTrans=0 and flkOn=N/N        -> healthy.
            // flkLayer splits an intermittent one by cause: surf>0 means the VS
            // stops producing surfaces (upstream of the accel manager), as>0
            // means it leaves the built structure, and neither raised while
            // flkTrans>0 means it stays in the structure and stops being
            // resolved. flkGap is the longest run of gone frames - "seconds"
            // at this frame rate should read in the hundreds, and a flkGap of 1
            // is a single-frame blink, a different symptom.
            " flkTrans=", e.flkTrans,
            " flkLayer=surf", e.flkSurfTrans, "/as", e.flkAsTrans,
                        "/view", e.flkViewTrans,
            " flkOn=", e.flkOn, "/", e.flkFrames,
            " flkGap=", e.flkOffRunMax,
            " flkSince=", e.flkSinceTrans,
            // NV-DXVK [Centroid]: the world position the ONSCREEN verdict on
            // this line was computed from, as a bbox over this VS's surfaces.
            // Read it against the SAME VS on an adjacent frame with the
            // opposite verdict - that pair is the measurement, and a single
            // line's value means nothing on its own.
            //
            // cenSeen is the denominator and must be checked first: 0 means no
            // surface of this VS was sampled, and the bbox is then "none"
            // rather than a position.
            //
            // clipW is the other half of an off-screen verdict. ONSCREEN is
            // `w > 0 && all(abs(ndc) <= 1)`, so clipWNeg>0 says the geometry is
            // BEHIND the camera and clipWNeg=0 with prOnScreen=0 says it is in
            // front and outside the frustum bounds. Different bugs.
            " cenSeen=", e.cenSeen,
            " cenMin=", (e.cenSeen > 0u
              ? str::format("(", e.cenMn[0], ",", e.cenMn[1], ",", e.cenMn[2], ")")
              : std::string("none")),
            " cenMax=", (e.cenSeen > 0u
              ? str::format("(", e.cenMx[0], ",", e.cenMx[1], ",", e.cenMx[2], ")")
              : std::string("none")),
            " cenJump=", e.cenJump,
            " clipW=", (e.cenSeen > 0u
              ? str::format(e.clipWMn, "..", e.clipWMx)
              : std::string("none")),
            " clipWNeg=", e.clipWNeg,
            // NV-DXVK [CenSplit]: per-surface camera-distance distribution as
            // total/onScreen per bucket. THIS IS THE FIELD TO READ, not
            // cenJump - cenJump moves when the surface SET changes and was
            // once misread as the geometry moving.
            //
            // Bucket edges: <250, <1k, <4k, <16k, <64k, >=64k world units.
            //
            // Compare the SAME bucket across a rendering frame and a
            // non-rendering one:
            //   bucket count drops to 0   -> those surfaces were not submitted
            //                                that frame. Defect is upstream, in
            //                                what produces the draws.
            //   count holds, onScreen 0   -> they exist and are not visible.
            //                                Defect is placement/visibility.
            // A count that holds while the geometry disappears on screen is
            // the more interesting of the two and the one no probe has shown.
            " cenDist=[",
              e.cenDistN[0], "/", e.cenDistOn[0], " ",
              e.cenDistN[1], "/", e.cenDistOn[1], " ",
              e.cenDistN[2], "/", e.cenDistOn[2], " ",
              e.cenDistN[3], "/", e.cenDistOn[3], " ",
              e.cenDistN[4], "/", e.cenDistOn[4], " ",
              e.cenDistN[5], "/", e.cenDistOn[5], "]");
        };

        for (const auto& kv : rankRc) {
          const RcVs& e = kv.second;
          if (e.ordSeen == 0ull && e.ordFinal == 0ull && e.unoSeen == 0ull) {
            // Present in the surface table but the primary ray never interacted
            // with it in any way this frame. Listed with verdict NOTRAVERSED
            // rather than skipped - "in the TLAS and never touched" is the
            // single most important state this census can report.
            Logger::info(str::format(
              "[ResolveCensus]   f=", frameRc,
              " vs=0x", std::hex, kv.first, std::dec,
              " surfaces=", e.surfaces,
              " ordSeen=0 ordFinal=0 unoSeen=0 continued=0",
              " verdict=NOTRAVERSED",
              piSuffix(e)));
            continue;
          }

          // Verdicts are the four cases the buckets partition into - see the
          // region block in common_binding_indices.h. Stated as a label rather
          // than left to the reader because the whole value of this probe is
          // that "never hit" and "hit then dropped" stop being conflated.
          const char* verdict =
            (e.ordSeen == 0ull && e.unoSeen > 0ull)   ? "UNORDERED_ONLY" :
            (e.ordFinal > 0ull)                       ? "WINS_PRIMARY" :
            (e.continued >= e.ordSeen)                ? "ALWAYS_RESOLVED_PAST" :
                                                        "LOST_AFTER_RESOLVE";

          Logger::info(str::format(
            "[ResolveCensus]   f=", frameRc,
            " vs=0x", std::hex, kv.first, std::dec,
            " surfaces=", e.surfaces,
            " ordSeen=", e.ordSeen,
            " ordFinal=", e.ordFinal,
            " unoSeen=", e.unoSeen,
            " continued=", e.continued,
            " winRate=", (e.ordSeen > 0ull) ? (double(e.ordFinal) / double(e.ordSeen)) : 0.0,
            " verdict=", verdict,
            piSuffix(e)));
        }

        // NV-DXVK [CamProbe]: raw per-surface lines, no rollup.
        //
        // Every aggregate in this census describes a VS, and the failure is a
        // per-object, single-frame event, so an aggregate is the wrong shape
        // for the last step no matter how carefully it is built - a max over
        // 258 surfaces still cannot say WHICH surface, at what distance, with
        // what error, hitting what instead. These lines say exactly that, for
        // exactly the surfaces whose absence needs explaining: their VS was
        // never traversed this frame, yet the surface is on screen and the face
        // probe self-hits it.
        //
        // Capped per frame because this runs on every frame by design (a
        // single-frame dropout is invisible to any even-stride sampler) and the
        // population is a handful of surfaces per frame in normal operation.
        // The cap protects against a pathological frame turning the log into
        // the bottleneck, not against the expected volume.
        if (RtxOptions::logResolveCensusRaw()) {
          int rawBudget = RtxOptions::logResolveCensusRawPerFrame();
          // NV-DXVK [CenSplit]: running count of AIMED, OFF-SCREEN surfaces
          // seen so far this frame, used to stride-sample them.
          //
          // The aimed VS has ~259 surfaces and the per-frame budget is far
          // smaller, so printing them in scan order would dump the lowest slot
          // indices and stop - the first-N bias this file's own probe comments
          // warn about twice, and the near surfaces sit anywhere in the range
          // (measured: slots 111..886). Striding instead spreads the sample
          // across the whole population.
          //
          // On-screen aimed surfaces are NEVER strided away: they are 2-13 per
          // frame and they are the entire signal. The stride applies only to
          // the far bulk, which is the part being sampled for comparison.
          uint32_t aimedFarSeen = 0u;
          constexpr uint32_t kAimedFarStride = 16u;
          for (uint32_t s = 0; s < scanRc && rawBudget > 0; ++s) {
            const RtInstance* inst = reorderedRc[s];
            const BlasEntry* blas = (inst != nullptr) ? inst->getBlas() : nullptr;
            if (blas == nullptr) {
              continue;
            }
            const uint32_t pr = covRc[baseProbeFlags + s];
            // NV-DXVK [CamTris]: the selection is now "owns at least one pixel",
            // not the old on-screen-self flag. That flag was derived from
            // triangle 0, which its own WINS_PRIMARY calibration showed is
            // frontmost only 16% of the time even for geometry that is plainly
            // visible - so it was selecting on the probe's aim rather than on
            // the geometry's visibility, and the raw lines it produced were
            // mostly surfaces that were legitimately behind something.
            //
            // The face probe self-hit is still required: it is what proves the
            // surface is genuinely in the built structure, which is the other
            // half of "present and visible yet never traversed".
            const uint32_t triReached = covRc[baseTriReached + s];
            const bool ownsPixels = (pr & COVERAGE_TLASPROBE_FLAG_RAN)
              && (pr & COVERAGE_TLASPROBE_FLAG_ANY_SELF)
              && triReached > 0u;
            // NV-DXVK [RawHit]: a committed hit on a surface whose VS recorded
            // no resolved interaction is reported unconditionally, whatever the
            // probe thinks. It is direct evidence from the renderer's own
            // traversal rather than from an instrument of mine, and the last
            // three false leads all came from trusting my aim over that.
            const bool hadRawHit = covRc[baseRawHit + s] > 0u;
            // NV-DXVK [CenSplit]: the aimed-VS bypass, applied here too. This
            // gate runs BEFORE the VS is even resolved, so relaxing only the
            // "was traversed" veto below would still have discarded every line
            // and the option would have gone on reading as enabled while
            // emitting nothing. ownsPixels requires triReached>0, i.e. the
            // surface is frontmost somewhere - the near surfaces on a flash
            // frame need not be, and requiring it would re-introduce the same
            // aim-based selection this block's own comment warns about.
            //
            // DELIBERATELY DOES NOT TEST THE VS HASH HERE. The first version
            // did, against the LIVE owner, and every line it produced was
            // internally mismatched: vs=0x29d5f7de0ba76c66 with
            // probeVs=0x28d6a5dc1284d8fd / 0x29566a60d473af50 / 0x28ea29dae516dbd7
            // on 5 of 5 sampled lines. A slot's live owner at readback is not
            // its owner in the frame the GPU wrote the probe data - that is the
            // whole reason [ProbeAlign] exists (V12 section 3.1), and selecting
            // on the live hash walks straight back into it.
            //
            // So admit any probed surface cheaply here, and let the second
            // gate below filter on attrVs, which IS the probe-frame owner.
            // Costs nothing: rawBudget is only consumed by lines that actually
            // print, so the extra candidates are a compare and a continue.
            //
            // The ONSCREEN requirement is gone as of the far/near comparison:
            // the aimed VS's OFF-screen surfaces are now wanted too, because
            // "where are these same surfaces on a frame where none of them
            // render" is the half of the comparison the on-screen-only dump
            // structurally could not produce.
            const bool aimedEarly =
              !RtxOptions::subViewGateProbeVsHashes().empty()
              && (pr & COVERAGE_TLASPROBE_FLAG_RAN);
            if (!ownsPixels && !hadRawHit && !aimedEarly) {
              continue;
            }
            const uint64_t vsHash = uint64_t(blas->input.getTransformData().vertexShaderHash);
            // NV-DXVK [ProbeAlign]: this line's probe data describes the
            // GPU-frame owner of slot s, not the live owner — so BOTH the
            // "was it traversed" filter and the headline attribution use the
            // probe-frame owner when the stamp resolves. vs= stays the live
            // owner for continuity with older logs; probeVs= is the surface
            // the fields are actually about.
            uint64_t probeVs = 0ull;
            // NV-DXVK [CenSplit]: resolved from the SAME ring row as probeVs,
            // in the same walk, so the identity and the probe data on this line
            // always describe the same surface in the same frame. Falling back
            // to the live instance when the stamp does not resolve would
            // silently reintroduce the frame mix on exactly the ambiguous
            // slots, so they stay 0 and the line says "unresolved" instead.
            uint64_t probeInstId = 0ull;
            uint32_t probePrevSlot = uint32_t(SURFACE_INDEX_INVALID);
            bool probeIdentResolved = false;
            {
              const uint32_t probeStampR = covRc[baseProbeFrame + s];
              if ((probeStampR & 0x80000000u) != 0u) {
                const uint32_t nibR2 = (probeStampR >> 27) & 0xFu;
                for (uint32_t r = 0; r < kIdentRingSize; ++r) {
                  const uint32_t fp1 = s_identExpectedRingFramePlus1[r];
                  if (fp1 == 0u || (((fp1 - 1u) & 0xFu) != nibR2)) {
                    continue;
                  }
                  if (s < s_identVsRing[r].size() && s_identVsRing[r][s] != 0ull) {
                    probeVs = s_identVsRing[r][s];
                  }
                  if (s < s_identInstIdRing[r].size()) {
                    probeInstId = s_identInstIdRing[r][s];
                    probeIdentResolved = true;
                  }
                  // NV-DXVK [CamProbe prevSurf]: same ring row r, so the
                  // predecessor named here belongs to the frame the GPU wrote
                  // this line's counts, not to the table at readback.
                  if (s < s_identPrevSlotRing[r].size()) {
                    probePrevSlot = s_identPrevSlotRing[r][s];
                  }
                  break;
                }
              }
            }
            const uint64_t attrVs = (probeVs != 0ull) ? probeVs : vsHash;
            const auto it = byVs.find(attrVs);
            if (it == byVs.end()) {
              continue;
            }
            const RcVs& e = it->second;
            // NV-DXVK [CenSplit]: bypass for a VS under active investigation.
            //
            // The two filters this block normally applies - "owns pixels or had
            // a raw hit" above, and "its VS resolved nothing this frame" below -
            // together select "visible yet never traversed", which is the
            // question they were built for. They are also mutually exclusive
            // with the question now being asked, and that cost a run: on the
            // frames where 0x29d5f7de0ba76c66's near surfaces appear, its VS
            // resolves ~19.6k pixels, so the veto discards exactly the lines
            // needed and the whole option emitted 0 lines all session while
            // reading as enabled (it defaults to true).
            //
            // For an aimed VS the population of interest is instead "on screen
            // this frame", which is 2-13 surfaces of its 259 - well inside the
            // per-frame budget - and naming those slots is what decides whether
            // the same surfaces relocate or different geometry lands in slots
            // attributed to this VS. The original filters are untouched for
            // every other VS.
            const bool aimedVs =
              !RtxOptions::subViewGateProbeVsHashes().empty()
              && RtxOptions::subViewGateProbeVsHashes().count(attrVs) != 0;
            const bool aimedOnScreen = aimedVs
              && (pr & COVERAGE_TLASPROBE_FLAG_RAN)
              && (pr & COVERAGE_TLASPROBE_FLAG_ONSCREEN);
            // Stride-sample the aimed VS's OFF-screen surfaces. Counted here,
            // after attrVs is known, so the stride runs over the aimed
            // population itself rather than over all slots - a stride over slot
            // index would sample whatever else happens to occupy those slots
            // and would drift as the surface table reshuffles, which it
            // demonstrably does (202 of 302 lines had live owner != probe
            // owner).
            bool aimedFarSample = false;
            if (aimedVs && !aimedOnScreen) {
              aimedFarSample = ((aimedFarSeen % kAimedFarStride) == 0u);
              ++aimedFarSeen;
            }
            // The early gate is now deliberately permissive (it cannot know the
            // probe-frame owner without this ring walk), so anything that got
            // in ONLY on that relaxation and is not actually the aimed VS has
            // to be dropped here. Without this the dump fills with unrelated
            // geometry and the aimed surfaces are lost in it - the budget would
            // be spent before reaching them.
            if (!ownsPixels && !hadRawHit && !aimedOnScreen && !aimedFarSample) {
              continue;
            }
            if (!aimedOnScreen && !aimedFarSample
                && (e.ordSeen != 0ull || e.ordFinal != 0ull || e.unoSeen != 0ull)) {
              continue;  // its VS was traversed this frame; nothing to explain
            }

            const uint32_t camHitPlusOne = covRc[baseCamHitSurf + s];
            // NV-DXVK [CamProbe ringVs]: who the ident ring says owned this
            // slot in the frame the GPU wrote these counts. The aggregate
            // credits rawHit/ordSeen to THIS owner (see the vsForCounts remap
            // in the rollup loop), while vs= above is the LIVE owner at
            // readback. 2026-08-02: rawHit=893 printed under vs=0x29d5... on a
            // NOTRAVERSED frame — without this field the line cannot say
            // whether those hits are the VS's own (census wrong) or a
            // reshuffled twin's (census right). ringVs=none means the slot
            // carries no identity observation this frame.
            uint64_t ringVs = 0ull;
            {
              const uint32_t obsIdentR = covRc[baseObsIdent + s];
              if ((obsIdentR & 0x80000000u) != 0u) {
                const uint32_t nibR = (obsIdentR >> 27) & 0xFu;
                for (uint32_t r = 0; r < kIdentRingSize; ++r) {
                  const uint32_t fp1 = s_identExpectedRingFramePlus1[r];
                  if (fp1 == 0u || (((fp1 - 1u) & 0xFu) != nibR)) {
                    continue;
                  }
                  if (s < s_identVsRing[r].size() && s_identVsRing[r][s] != 0ull) {
                    ringVs = s_identVsRing[r][s];
                  }
                  break;
                }
              }
            }
            Logger::info(str::format(
              "[CamProbe] f=", frameRc,
              " surf=", s,
              " vs=0x", std::hex, vsHash, std::dec,
              " probeVs=", (probeVs == 0ull
                ? std::string("none")
                : str::format("0x", std::hex, probeVs, std::dec)),
              " ringVs=", (ringVs == 0ull
                ? std::string("none")
                : str::format("0x", std::hex, ringVs, std::dec)),
              " camDotA=", covRc[baseCamDotA + s],
              " camDotB=", covRc[baseCamDotB + s],
              " camProjRan=", ((pr & COVERAGE_TLASPROBE_FLAG_CAMPROJ_RAN) ? 1 : 0),
              " camDist=", covRc[baseCamDist + s],
              " camRayHit=", ((pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_HIT) ? 1 : 0),
              " camRaySelf=", ((pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_SELF) ? 1 : 0),
              " camRayAnySelf=", ((pr & COVERAGE_TLASPROBE_FLAG_CAMRAY_ANY_SELF) ? 1 : 0),
              // 'none' rather than a sentinel: 0 in this region means the ray
              // committed no hit, and printing that as surface 0 would name an
              // innocent surface as the occluder.
              " camHitSurf=", (camHitPlusOne == 0u
                ? std::string("none")
                : str::format(camHitPlusOne - 1u)),
              " triTested=", covRc[baseTriTested + s],
              " triReached=", triReached,
              " primCount=", covRc[basePrimCount + s],
              // Raw, per surface, unreduced - the whole reason these lines
              // exist. surfMap and instMask print the stored value minus the
              // +1 bias, or "unwritten" when the slot was never touched.
              " rawHit=", covRc[baseRawHit + s],
              // NV-DXVK [CenSplit]: STABLE SURFACE IDENTITY. Read prevSurf, not
              // surf=, when tracking a surface across frames.
              //
              // surf= is a SLOT, and slots are frame-local: the accel manager
              // reassigns them on every TLAS build. Measured on the 15:59 run,
              // 721 distinct slots carried this VS's ~259 surfaces, and 252 of
              // those slots showed a camera-distance spread over 5000 units.
              // A slot-keyed comparison therefore cannot tell "this surface
              // moved" from "a different surface is in that slot now" - it was
              // the second ill-posed comparison in a row on this question.
              //
              // prevSurf is THE cross-frame key: the slot this slot occupied
              // last frame, per AccelManager::getProbePrevSurfaceSlots(). Chain
              // it backwards - (f,s) -> (f-1,prevSurf) -> (f-2,...) - to follow
              // one surface through a flash, and read it against cenDistCam:
              //
              //   prevSurf resolves + cenDistCam changes -> it MOVED.
              //   prevSurf=none on the near surfaces -> they are new that frame,
              //     nothing moved, and the cause is upstream of the table.
              //
              // instId is RtInstance::getId() and is kept only as a coarse
              // grouping. It is NOT an identity: on the 16:51 run ids 2604 and
              // 4586 carried bit-identical centroids and split the run 498/516,
              // i.e. one object destroyed and recreated reads as two.
              //
              // firstIndex is deliberately gone. It was the intended key and it
              // printed 0 on all 17462 lines of that run, because
              // RtSurface::firstIndex is assigned nowhere in the tree - see the
              // note on getProbePrevSurfaceSlots(). A field that cannot vary
              // cannot discriminate, and leaving it on the line invites a
              // fourth ill-posed comparison.
              " instId=", (probeIdentResolved
                ? str::format(probeInstId) : std::string("unresolved")),
              " prevSurf=", (probePrevSlot == uint32_t(SURFACE_INDEX_INVALID)
                ? std::string("none") : str::format(probePrevSlot)),
              // The live table's view of the same slot, kept only so a
              // disagreement is visible rather than silent. Never track a
              // surface by these - they are the slot's occupant NOW.
              " liveInstId=", (inst != nullptr ? inst->getId() : 0ull),
              // The centroid and its distance, so the line stands alone.
              " cen=(", u2fRaw(covRc[baseCenX + s]), ",",
                        u2fRaw(covRc[baseCenY + s]), ",",
                        u2fRaw(covRc[baseCenZ + s]), ")",
              " cenDistCam=", std::sqrt(
                  (u2fRaw(covRc[baseCenX + s]) - cenCamPos.x) * (u2fRaw(covRc[baseCenX + s]) - cenCamPos.x)
                + (u2fRaw(covRc[baseCenY + s]) - cenCamPos.y) * (u2fRaw(covRc[baseCenY + s]) - cenCamPos.y)
                + (u2fRaw(covRc[baseCenZ + s]) - cenCamPos.z) * (u2fRaw(covRc[baseCenZ + s]) - cenCamPos.z)),
              " clipW=", u2fRaw(covRc[baseClipW + s]),
              " onScreen=", ((pr & COVERAGE_TLASPROBE_FLAG_ONSCREEN) ? 1 : 0),
              " surfMap=", (covRc[baseSurfMap + s] == 0u
                ? std::string("unwritten")
                : str::format(int32_t(covRc[baseSurfMap + s] - 1u))),
              " instMask=", (covRc[baseInstMask + s] == 0u
                ? std::string("unwritten")
                : str::format("0x", std::hex, covRc[baseInstMask + s] - 1u, std::dec)),
              " prFlags=0x", std::hex, pr, std::dec));
            --rawBudget;
          }
        }

        // Clear the GPU-written regions - 75-78 (census), 79-82 ([PIWrite]),
        // 83-84 ([TlasProbe]) and 85-88 ([CamProbe]) - in one contiguous
        // memset, then 90-91 ([CamTris] results) in a second.
        //
        // Region 89 (the CPU-written triangle count) is SKIPPED, and the gap is
        // why there are two memsets instead of one. Clearing it here was a real
        // bug: this readback runs while the probe dispatch has only been
        // RECORDED, not executed, so the zeroing landed before the GPU ever
        // read the counts and the triangle sampling silently did nothing on
        // 2571 of 2580 frames - all-or-nothing per frame, exactly as a race
        // looks. Region 89 is owned end to end by dispatchTlasProbe, which
        // rewrites every live slot and zeroes the tail immediately before the
        // dispatch that consumes it, so nothing here needs to clear it and
        // nothing here may. Must
        // cover every slot the shaders can write
        // (their own bound is COVERAGE_SURFACE_SLOTS), not just the scanned
        // window - otherwise counts at high slots accumulate silently across
        // frames and every later census is a running total wearing a per-frame
        // label. Clearing the [PIWrite] regions matters just as much: their
        // whole meaning rests on 0 being "not written THIS frame", which a
        // stale value from an earlier frame would quietly destroy.
        std::memset(&covRc[baseOrdSeen], 0,
                    size_t(14u) * size_t(COVERAGE_SURFACE_SLOTS) * sizeof(uint32_t));
        // Regions 90-93: [CamTris] tested/reached, [RawHit] and [SurfMap].
        // Region 94 ([InstMask]) is CPU-owned like 89 and is excluded for the
        // same reason - it is written immediately before the dispatch that
        // reads it, and clearing it here would win that race every time.
        std::memset(&covRc[baseTriTested], 0,
                    size_t(4u) * size_t(COVERAGE_SURFACE_SLOTS) * sizeof(uint32_t));
        // Regions 95-99 ([Occluder], [NoCull] reached and missed, [Spike]
        // maxRadius, [IdentProbe] packed identity) sit past the CPU-owned 94,
        // so they need their own memset rather than extending the block above
        // across it. [IdentProbe] MUST be cleared each census: bit 31 means
        // "written since the last census", and a stale value would fabricate
        // identity comparisons for slots no ray touched.
        std::memset(&covRc[baseOccluder], 0,
                    size_t(5u) * size_t(COVERAGE_SURFACE_SLOTS) * sizeof(uint32_t));
        // NV-DXVK [Centroid]: regions 101-104. Cleared for the same reason as
        // everything else here - the probe writes them only for slots it
        // actually runs on, so a slot the probe skipped this frame would
        // otherwise hand the census a previous frame's world position and
        // report the geometry as stationary precisely when it moved. Region
        // 100 (FRAME) is deliberately NOT in this range: it carries its own
        // frame nibble and is how stale data is detected elsewhere.
        std::memset(&covRc[baseCenX], 0,
                    size_t(4u) * size_t(COVERAGE_SURFACE_SLOTS) * sizeof(uint32_t));
      }
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
        // NV-DXVK [Coverage compact]: the CPU no longer scans the 74 MB
        // coverage buffer directly. That mapping is HOST_VISIBLE|HOST_COHERENT
        // without HOST_CACHED, i.e. write-combined: every 4-byte CPU read is
        // an uncached memory transaction, and the full-buffer scan measured
        // ~1 second per frame ([Perf.Frame] finalBlit ~0.9-1.4s with the GPU
        // idle the whole time). Instead:
        //   (1) a compute pass (recordCoverageCompactDispatch, recorded at
        //       the exits of dispatchDebugView) appends every nonzero
        //       (flatIndex, value) pair to a small HOST_CACHED buffer, and
        //   (2) here we rebuild a CPU-side cached shadow array from the
        //       pairs of the LAST COMPLETED frame (same 1-2 frame staleness
        //       the direct read always had).
        // All existing dump logic below indexes `cov` exactly as before; it
        // just points at the shadow now. The shadow is sparse-cleared via a
        // touched-index list, so steady-state CPU cost is proportional to
        // the number of live surfaces, not the 18.6M-slot buffer size.
        uint32_t* cov = nullptr;
        constexpr size_t kCovTotalUints = size_t(COVERAGE_TOTAL_REGIONS) * COVERAGE_SURFACE_SLOTS;
        static std::vector<uint32_t> s_covShadow;
        static std::vector<uint32_t> s_covShadowTouched;
        const Rc<DxvkBuffer>& covCompactBuf = rtOutput.m_surfaceCoverageCompactBuffer;
        if (covCompactBuf.ptr() != nullptr) {
          // Rebuild the shadow from the last completed frame's pairs. (This
          // frame's compaction dispatch is recorded at the exits of
          // dispatchDebugView — see recordCoverageCompactDispatch above —
          // so it lands AFTER the postprocess PickRegion writes.)
          const uint32_t* comp = reinterpret_cast<const uint32_t*>(covCompactBuf->mapPtr(0));
          if (comp != nullptr) {
            if (s_covShadow.size() != kCovTotalUints) {
              s_covShadow.assign(kCovTotalUints, 0u);
              s_covShadowTouched.clear();
            }
            for (const uint32_t touchedIdx : s_covShadowTouched) {
              s_covShadow[touchedIdx] = 0u;
            }
            s_covShadowTouched.clear();

            const uint32_t rawCount = comp[0];
            const uint32_t entryCount = std::min(rawCount, uint32_t(COVERAGE_COMPACT_MAX_ENTRIES));
            if (rawCount > entryCount) {
              Logger::warn(str::format(
                "[Coverage] compact overflow: ", rawCount - entryCount,
                " nonzero slots dropped (capacity ", uint32_t(COVERAGE_COMPACT_MAX_ENTRIES), ")"));
            }
            const uint32_t* pairs = comp + COVERAGE_COMPACT_HEADER_UINTS;
            s_covShadowTouched.reserve(entryCount);
            for (uint32_t e = 0; e < entryCount; ++e) {
              const uint32_t flatIdx = pairs[2u * e];
              const uint32_t val = pairs[2u * e + 1u];
              // Torn-read guard: the dump has always tolerated racing an
              // in-flight GPU frame (deliberately loose), but a torn PAIR
              // could produce an out-of-range index — validate before the
              // scatter so a race costs precision, never memory safety.
              if (flatIdx < kCovTotalUints && val != 0u) {
                s_covShadow[flatIdx] = val;
                s_covShadowTouched.push_back(flatIdx);
              }
            }
            cov = s_covShadow.data();
          }
        }
        // Fallback: compact buffer unavailable — original direct (slow,
        // uncached) mapping so the diagnostic still functions.
        if (cov == nullptr) {
          cov = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
        }
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

          // NV-DXVK [Perf.UnorderedSteps]: raw census for the unordered resolve
          // stage. Placed before the coveragePickRegionOnly fast path so it is
          // never skipped, and it needs none of the per-VS attribution below -
          // these are three scalars in slot 0 of regions 71/72/73.
          //
          // Context for whoever reads this next. The compile-time stage ladder
          // (REMIX_GBSTOP in geometry_resolver.slangh) measured, on a static
          // camera at 1920x1080 on 2026-07-25:
          //
          //   traversal + raygen + G-buffer stores    3.0 ms
          //   unordered resolve                      42.5 ms   <- this stage
          //   material evaluation (resolveVertex)    74.5 ms
          //   post-material tail + resolve loop       ~0 ms
          //                                         -------
          //                                          ~120 ms  = the whole pass
          //
          // 42.5 ms divided by an unknown candidate count is not attributable:
          // kMaxUnorderedResolveSteps is 128 on primary rays, so "many cheap
          // candidates" and "few expensive candidates" are indistinguishable in a
          // pass timer. These counters settle that before any cut ladder result
          // gets interpreted.
          // The sweep overrides shader constants, not options, so its census step
          // has to be admitted explicitly here.
          if (RtxOptions::perfUnorderedStepCensus() || m_perfSweep.censusActive) {
            const uint64_t unoSteps =
              cov[uint32_t(COVERAGE_UNORDERED_STEPS_REGION) * uint32_t(COVERAGE_SURFACE_SLOTS)];
            const uint64_t unoInteractions =
              cov[uint32_t(COVERAGE_UNORDERED_INTERACTIONS_REGION) * uint32_t(COVERAGE_SURFACE_SLOTS)];
            const uint64_t unoPixels =
              cov[uint32_t(COVERAGE_UNORDERED_PIXELS_REGION) * uint32_t(COVERAGE_SURFACE_SLOTS)];
            const double rcpPixels = (unoPixels > 0u) ? (1.0 / double(unoPixels)) : 0.0;
            Logger::warn(str::format(
              "[Perf.UnorderedSteps] gpuFrame=", gpuFrame,
              " pixels=", unoPixels,
              " steps=", unoSteps,
              " interactions=", unoInteractions,
              " stepsPerPixel=", double(unoSteps) * rcpPixels,
              " interactionsPerPixel=", double(unoInteractions) * rcpPixels,
              " acceptRate=", (unoSteps > 0u) ? (double(unoInteractions) / double(unoSteps)) : 0.0));
          }

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
              // NV-DXVK [FogHideProbe]: PS hash of the pixel-winning draws. The
              // garbage stack shares one VS but the fog hide keys off the PS's
              // c_fogColorFactor read — the PS identity is what names the
              // FS_*.dxbc dump to disassemble.
              std::map<uint64_t, uint64_t> vsPs;
              std::map<uint64_t, VsBox2> vsBox;
              struct HashRec {
                uint64_t px = 0, vs = 0, mat = 0, tex = 0, geo = 0, ps = 0; VsBox2 box;
                bool xfDone = false;
                float oX = 0, oY = 0, oZ = 0;     // object-space AABB extent
                float cl0 = 0, cl1 = 0, cl2 = 0;  // objectToWorld column lengths (rigid => ~1)
                float wX = 0, wY = 0, wZ = 0;     // world-space AABB extent
                float tx = 0, ty = 0, tz = 0;     // objectToWorld translation
              };
              std::map<uint16_t, HashRec> hashAgg;
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
                  vsPs[vs] = uint64_t(blas->input.getTransformData().pixelShaderHash);
                  const uint64_t geo = uint64_t(inst->surface.associatedGeometryHash);
                  const uint16_t packed =
                      (uint16_t) (geo >> 48) ^ (uint16_t) (geo >> 32)
                    ^ (uint16_t) (geo >> 16) ^ (uint16_t) geo;
                  HashRec& hr = hashAgg[packed];
                  hr.px += px; hr.vs = vs; hr.mat = vsMat[vs]; hr.tex = vsTex[vs]; hr.geo = geo;
                  hr.ps = vsPs[vs];
                  hr.box.minX = std::min(hr.box.minX, sMinX); hr.box.maxX = std::max(hr.box.maxX, sMaxX);
                  hr.box.minY = std::min(hr.box.minY, sMinY); hr.box.maxY = std::max(hr.box.maxY, sMaxY);
                  // Geometry + transform diagnostics for EVERY surface (computed
                  // once per packed-hash group). object AABB (blas boundingBox),
                  // objectToWorld column lengths (rigid => ~1, else scale/shear),
                  // and world AABB extent (object box transformed). Lets the
                  // [PickHash] table show, per colour, whether a mesh is compact
                  // (objExt small) but flung huge in world (worldExt big, colLen!=1)
                  // = the blade, vs a real tall mesh (objExt big, colLen~1) = ships.
                  if (!hr.xfDone) {
                    hr.xfDone = true;
                    const AxisAlignedBoundingBox& bb = blas->input.getGeometryData().boundingBox;
                    const Matrix4& o2w = inst->surface.objectToWorld;
                    auto colLen = [&](int c) {
                      return std::sqrt(o2w[c][0]*o2w[c][0] + o2w[c][1]*o2w[c][1] + o2w[c][2]*o2w[c][2]);
                    };
                    hr.cl0 = colLen(0); hr.cl1 = colLen(1); hr.cl2 = colLen(2);
                    hr.tx = o2w[3][0]; hr.ty = o2w[3][1]; hr.tz = o2w[3][2];
                    hr.oX = bb.maxPos.x - bb.minPos.x;
                    hr.oY = bb.maxPos.y - bb.minPos.y;
                    hr.oZ = bb.maxPos.z - bb.minPos.z;
                    Vector3 wMin( 1e30f,  1e30f,  1e30f), wMax(-1e30f, -1e30f, -1e30f);
                    for (int ci = 0; ci < 8; ++ci) {
                      const Vector4 cc(
                        (ci & 1) ? bb.maxPos.x : bb.minPos.x,
                        (ci & 2) ? bb.maxPos.y : bb.minPos.y,
                        (ci & 4) ? bb.maxPos.z : bb.minPos.z, 1.0f);
                      const Vector4 w = o2w * cc;
                      wMin.x = std::min(wMin.x, w.x); wMin.y = std::min(wMin.y, w.y); wMin.z = std::min(wMin.z, w.z);
                      wMax.x = std::max(wMax.x, w.x); wMax.y = std::max(wMax.y, w.y); wMax.z = std::max(wMax.z, w.z);
                    }
                    hr.wX = wMax.x - wMin.x; hr.wY = wMax.y - wMin.y; hr.wZ = wMax.z - wMin.z;
                  }
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
                  char psHex[24]; snprintf(psHex, sizeof(psHex), "0x%016llx", (unsigned long long)vsPs[kv.first]);
                  const VsBox2& bx = vsBox[kv.first];
                  Logger::info(str::format(
                    "[Coverage]   PickRegion2VS VS=", vsHex, " pixels=", kv.second,
                    " box x=[", bx.minX, ",", bx.maxX, "] y=[", bx.minY, ",", bx.maxY, "]",
                    " w=", (bx.maxX - bx.minX), " h=", (bx.maxY - bx.minY),
                    " colorTexture=", texHex, " material=", matHex, " ps=", psHex));
                }
                // [PickHash] one line per GEOMETRY-hash colour key (the value the
                // GEOMETRY_HASH debug view 277 actually colours by:
                // packed = fold16(associatedGeometryHash); colour = R5G6B5 of it).
                // rgb is the EXACT debug-view colour (0-255) so a screen colour can
                // be matched to its surface with no aiming/guessing. Sorted by pixels.
                std::vector<std::pair<uint16_t, HashRec>> hsrt(hashAgg.begin(), hashAgg.end());
                std::sort(hsrt.begin(), hsrt.end(),
                          [](const std::pair<uint16_t, HashRec>& a,
                             const std::pair<uint16_t, HashRec>& b) { return a.second.px > b.second.px; });
                for (const auto& kv : hsrt) {
                  const uint16_t packed = kv.first;
                  const HashRec& hr = kv.second;
                  const int r = int(((packed >> 0)  & 0x1F) * 255 / 31);
                  const int g = int(((packed >> 5)  & 0x3F) * 255 / 63);
                  const int b = int(((packed >> 11) & 0x1F) * 255 / 31);
                  char gHex[24]; snprintf(gHex, sizeof(gHex), "0x%016llx", (unsigned long long) hr.geo);
                  char vHex[24]; snprintf(vHex, sizeof(vHex), "0x%016llx", (unsigned long long) hr.vs);
                  char mHex[24]; snprintf(mHex, sizeof(mHex), "0x%016llx", (unsigned long long) hr.mat);
                  char tHex[24]; snprintf(tHex, sizeof(tHex), "0x%016llx", (unsigned long long) hr.tex);
                  char fsHex[24]; snprintf(fsHex, sizeof(fsHex), "0x%016llx", (unsigned long long) hr.ps);
                  char pHex[8];  snprintf(pHex, sizeof(pHex), "0x%04x", (unsigned) packed);
                  Logger::info(str::format(
                    "[PickHash] packed=", pHex, " rgb=(", r, ",", g, ",", b, ")",
                    " pixels=", hr.px,
                    " box x=[", hr.box.minX, ",", hr.box.maxX, "] y=[", hr.box.minY, ",", hr.box.maxY, "]",
                    " objExt=(", hr.oX, ",", hr.oY, ",", hr.oZ, ")",
                    " colLen=(", hr.cl0, ",", hr.cl1, ",", hr.cl2, ")",
                    " worldExt=(", hr.wX, ",", hr.wY, ",", hr.wZ, ")",
                    " o2wT=(", hr.tx, ",", hr.ty, ",", hr.tz, ")",
                    " geometryHash=", gHex, " VS=", vHex, " material=", mHex, " colorTexture=", tHex,
                    " ps=", fsHex));
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
          // NV-DXVK [Coverage compact]: `cov` now points at the CPU shadow,
          // which is sparse-cleared via the touched-index list on the next
          // rebuild — the clear that matters is the REAL GPU buffer, so its
          // atomic accumulation restarts. Writes to write-combined memory
          // stream at full bandwidth (unlike reads), so this memset stays
          // cheap (~10 ms) even at 74 MB.
          uint32_t* covGpu = reinterpret_cast<uint32_t*>(rtOutput.m_surfaceCoverageBuffer->mapPtr(0));
          if (covGpu != nullptr) {
            memset(covGpu, 0, size_t(COVERAGE_TOTAL_REGIONS) * COVERAGE_SURFACE_SLOTS * sizeof(uint32_t));
          }
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
      // NV-DXVK [Coverage compact]: no debug-view pass this frame, so no
      // later coverage writers — record the compaction now.
      recordCoverageCompactDispatch();
      return;
    }

    debugView.dispatch(this,
      getResourceManager().getSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
      srcImage, rtOutput, *m_common);

    // NV-DXVK [Coverage compact]: recorded after debugView.dispatch so the
    // postprocess PickRegion writes are included in this frame's snapshot.
    recordCoverageCompactDispatch();

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
    //
    // NV-DXVK [perf]: gated off by default. This block is not a log — it drives
    // recordSkyDrawPositionsReadback below, which per gameplay frame creates up
    // to 8 host-visible buffers, emits 16 pipeline barriers, copies, signals a
    // timeline and spawns 8 std::async tasks, all inside this function's GPU
    // profile zone. Nsight put rasterizeToSkyMatte at p50 0.011 ms but p95 91 ms
    // with 655 >50 ms spikes across 661 frames (~1 per frame, ~99 ms/frame,
    // 65.5 s of a 203 s capture), none of it inside InjectRTX — i.e. this was
    // the whole gap between the ~55 ms InjectRTX span and the ~170 ms frame.
    // The [SkyTrace. prefix is dropped by log.cpp's filter, so the cost was
    // invisible in the log while still being paid every frame.
    if (RtxOptions::skyVertsReadbackEnable()) {
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
      // NV-DXVK [SkyPrefillCache]: m_skyProbeImage is passed so the prefill can
      // cache its analytic result and replay it by copy when the atmosphere
      // inputs have not changed. Gating here on m_skyClearDirty is NOT the same
      // question - that flag only says the game cleared a sky render target
      // (:10340, :10355), which TF2 does every frame, whereas the analytic sky
      // depends solely on sun/tint/turbidity. See rtx_atmosphere.h.
      m_atmosphere->dispatchCubeSkyPrefill(this, m_skyProbeCubePlaneStorageViews,
                                           m_skyProbeImage->info().extent.width,
                                           m_skyProbeImage);
      // Note: [SkyTrace.*] is dropped by the emitMsg prefix filter (log.cpp:308),
      // so this line never reaches the log. The per-window [Perf.SkyPrefill]
      // counter inside dispatchCubeSkyPrefill is the one to read.
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
