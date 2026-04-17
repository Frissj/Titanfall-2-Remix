#include "d3d11_rtx.h"
#include <set>
#include <chrono>
#include <fstream>
#include <mutex>
#include <unordered_map>

// Include dxvk_device.h before any rtx headers so that dxvk_buffer.h and
// sibling headers (included bare by rtx_utils.h) are already in the TU.
#include "../dxvk/dxvk_device.h"

#include "d3d11_context.h"
#include "d3d11_buffer.h"
#include "d3d11_input_layout.h"
#include "d3d11_device.h"
#include "d3d11_vs_classifier.h"
#include "d3d11_view_srv.h"
#include "d3d11_sampler.h"
#include "d3d11_depth_stencil.h"
#include "d3d11_blend.h"
#include "d3d11_rasterizer.h"

#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_point_instancer_system.h"
#include "../dxvk/rtx_render/rtx_materials.h"
#include "../dxvk/rtx_render/rtx_debug_view.h"
#include "../dxvk/rtx_render/rtx_camera.h"
#include "../dxvk/rtx_render/rtx_camera_manager.h"
#include "../dxvk/rtx_render/rtx_scene_manager.h"
#include "../dxvk/rtx_render/rtx_light_manager.h"
#include "../dxvk/rtx_render/rtx_matrix_helpers.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// NV-DXVK: scene dumper. Writes per-instance world-space triangles to
// scene_dump.obj for offline inspection. Triggered automatically once the game
// has been rendering for >5 seconds. Currently focused on BSP-style
// (R32G32_UINT packed position + g_modelInst SRV) draws.
namespace SceneDump {
  static std::ofstream g_obj;
  static std::mutex    g_mutex;
  static uint32_t      g_baseVtx       = 0;
  static uint32_t      g_objectsWritten = 0;
  static bool          g_done          = false;
  static std::chrono::steady_clock::time_point g_firstFrameTime;
  static bool          g_armed         = false;

  static const char* const kOutPath =
    "C:/Users/Friss/Downloads/Compressed/Titanfall-2-Digital-Deluxe-Edition-AnkerGames/Titanfall2/scene_dump.obj";

  static void armOnFirstGameplayFrame(uint32_t rawDraws) {
    if (g_done || g_armed) return;
    if (rawDraws < 50) return;  // skip menu frames
    g_firstFrameTime = std::chrono::steady_clock::now();
    g_armed = true;
  }
  static bool shouldDumpThisFrame() {
    // NV-DXVK: disabled. Capture path kept compiled for easy re-enable.
    return false;
  }
  static void open() {
    if (!g_obj.is_open()) {
      g_obj.open(kOutPath, std::ios::out | std::ios::trunc);
      if (g_obj.is_open()) {
        g_obj << "# Titanfall 2 BSP scene dump\n";
        dxvk::Logger::info(dxvk::str::format("[SceneDump] writing to ", kOutPath));
      } else {
        dxvk::Logger::err(dxvk::str::format("[SceneDump] FAILED to open ", kOutPath));
      }
    }
  }
  // Emit a small unit-cube at (0,0,0) — that's where the camera lives in this
  // dump's coordinate frame (geometry is camera-relative). Lets you eyeball
  // distance from camera to BSP chunks in Blender/MeshLab.
  static void writeCameraMarker() {
    if (!g_obj.is_open()) return;
    g_obj << "o CAMERA\n";
    const float s = 8.0f; // unit cube edge half-size in world units
    static const float corners[8][3] = {
      {-s,-s,-s},{ s,-s,-s},{ s, s,-s},{-s, s,-s},
      {-s,-s, s},{ s,-s, s},{ s, s, s},{-s, s, s},
    };
    for (int i = 0; i < 8; ++i)
      g_obj << "v " << corners[i][0] << " " << corners[i][1] << " " << corners[i][2] << "\n";
    const uint32_t b = g_baseVtx + 1;
    // 12 triangles via 6 quads — splitting each into two
    static const int faces[12][3] = {
      {0,1,2},{0,2,3},  {4,6,5},{4,7,6},
      {0,4,5},{0,5,1},  {2,6,7},{2,7,3},
      {1,5,6},{1,6,2},  {0,3,7},{0,7,4},
    };
    for (int i = 0; i < 12; ++i)
      g_obj << "f " << (b+faces[i][0]) << " " << (b+faces[i][1]) << " " << (b+faces[i][2]) << "\n";
    g_baseVtx += 8;
    ++g_objectsWritten;
  }
  static void close() {
    if (g_obj.is_open()) {
      g_obj.close();
      g_done = true;
      dxvk::Logger::info(dxvk::str::format(
        "[SceneDump] done. ", g_objectsWritten, " objects, ", g_baseVtx, " vertices"));
    }
  }
  static inline uint32_t decodeX(uint32_t u0)              { return u0 & 0x001FFFFFu; }
  static inline uint32_t decodeY(uint32_t u0, uint32_t u1) { return ((u0 >> 21) & 0x7FFu) | ((u1 & 0x3FFu) << 11u); }
  static inline uint32_t decodeZ(uint32_t u1)              { return u1 >> 10; }
}

namespace dxvk {

  // Map D3D11_BLEND → VkBlendFactor.  Mirrors D3D11BlendState::DecodeBlendFactor
  // but kept local to avoid exposing internal statics.
  static VkBlendFactor mapD3D11Blend(D3D11_BLEND b, bool isAlpha) {
    switch (b) {
      case D3D11_BLEND_ZERO:              return VK_BLEND_FACTOR_ZERO;
      case D3D11_BLEND_ONE:               return VK_BLEND_FACTOR_ONE;
      case D3D11_BLEND_SRC_COLOR:         return VK_BLEND_FACTOR_SRC_COLOR;
      case D3D11_BLEND_INV_SRC_COLOR:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
      case D3D11_BLEND_SRC_ALPHA:         return VK_BLEND_FACTOR_SRC_ALPHA;
      case D3D11_BLEND_INV_SRC_ALPHA:     return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      case D3D11_BLEND_DEST_ALPHA:        return VK_BLEND_FACTOR_DST_ALPHA;
      case D3D11_BLEND_INV_DEST_ALPHA:    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      case D3D11_BLEND_DEST_COLOR:        return VK_BLEND_FACTOR_DST_COLOR;
      case D3D11_BLEND_INV_DEST_COLOR:    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
      case D3D11_BLEND_SRC_ALPHA_SAT:     return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
      case D3D11_BLEND_BLEND_FACTOR:      return isAlpha ? VK_BLEND_FACTOR_CONSTANT_ALPHA : VK_BLEND_FACTOR_CONSTANT_COLOR;
      case D3D11_BLEND_INV_BLEND_FACTOR:  return isAlpha ? VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
      case D3D11_BLEND_SRC1_COLOR:        return VK_BLEND_FACTOR_SRC1_COLOR;
      case D3D11_BLEND_INV_SRC1_COLOR:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
      case D3D11_BLEND_SRC1_ALPHA:        return VK_BLEND_FACTOR_SRC1_ALPHA;
      case D3D11_BLEND_INV_SRC1_ALPHA:    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
      default:                            return VK_BLEND_FACTOR_ONE;
    }
  }

  // Map D3D11_BLEND_OP → VkBlendOp.
  static VkBlendOp mapD3D11BlendOp(D3D11_BLEND_OP op) {
    switch (op) {
      case D3D11_BLEND_OP_ADD:          return VK_BLEND_OP_ADD;
      case D3D11_BLEND_OP_SUBTRACT:     return VK_BLEND_OP_SUBTRACT;
      case D3D11_BLEND_OP_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
      case D3D11_BLEND_OP_MIN:          return VK_BLEND_OP_MIN;
      case D3D11_BLEND_OP_MAX:          return VK_BLEND_OP_MAX;
      default:                          return VK_BLEND_OP_ADD;
    }
  }

  D3D11Rtx::D3D11Rtx(D3D11DeviceContext* pContext)
    : m_context(pContext) {}

  Rc<DxvkSampler> D3D11Rtx::getDefaultSampler() const {
    if (m_defaultSampler == nullptr) {
      // D3D11 spec default: linear min/mag/mip, clamp UVW, no compare, no aniso
      DxvkSamplerCreateInfo info;
      info.magFilter      = VK_FILTER_LINEAR;
      info.minFilter      = VK_FILTER_LINEAR;
      info.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      info.mipmapLodBias  = 0.0f;
      info.mipmapLodMin   = -1000.0f;
      info.mipmapLodMax   =  1000.0f;
      info.useAnisotropy  = VK_FALSE;
      info.maxAnisotropy  = 1.0f;
      info.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      info.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      info.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      info.compareToDepth = VK_FALSE;
      info.compareOp      = VK_COMPARE_OP_NEVER;
      info.borderColor    = VkClearColorValue{};
      info.usePixelCoord  = VK_FALSE;
      m_defaultSampler = m_context->m_device->createSampler(info);
    }
    return m_defaultSampler;
  }

  void D3D11Rtx::Initialize() {
    // Scale geometry workers to available cores (min 2, max 6).
    // D3D11 games typically have high draw call counts, so more workers pay off.
    const uint32_t cores = std::max(2u, std::thread::hardware_concurrency());
    const uint32_t workers = std::min(std::max(cores / 2, 2u), 6u);
    m_pGeometryWorkers = std::make_unique<GeometryProcessor>(workers, "d3d11-geometry");

    // --- D3D11 sensible defaults (Default layer = lowest priority) ---
    // Written to the Default layer so rtx.conf, user.conf, and all other
    // config layers override them naturally.  Without this, setDeferred()
    // writes to the Derived layer (priority 5) which stomps rtx.conf (priority 3)
    // and makes per-game config files useless.
    const RtxOptionLayer* defaults = RtxOptionLayer::getDefaultLayer();

    // FusedWorldViewMode::View tells Remix to treat objectToView as the full
    // local-to-view transform.  In commitGeometryToRT it sets:
    //   objectToWorld = objectToView   (fused transform)
    //   worldToView   = identity       (camera at origin)
    // This works because we bake the worldToView into objectToView via
    // objectToView = worldToView * objectToWorld (line ~1787).  The camera
    // position is encoded in worldToView's translation, so after the fuse
    // geometry is centred near origin (camera-relative) and the RT camera
    // at origin sees it correctly.
    RtxOptions::fusedWorldViewModeObject().setDeferred(FusedWorldViewMode::View, defaults);

    // Anti-culling: D3D11 engines aggressively frustum-cull objects before
    // issuing draw calls.  Without anti-culling, off-screen objects vanish
    // from reflections, shadows, and GI.
    RtxOptions::AntiCulling::Object::enableObject().setDeferred(true, defaults);
    RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCullingObject().setDeferred(true, defaults);
    RtxOptions::AntiCulling::Object::numObjectsToKeepObject().setDeferred(20000u, defaults);
    RtxOptions::AntiCulling::Object::fovScaleObject().setDeferred(2.0f, defaults);
    RtxOptions::AntiCulling::Object::farPlaneScaleObject().setDeferred(10.0f, defaults);
    RtxOptions::AntiCulling::Light::enableObject().setDeferred(true, defaults);

    // Use incoming vertex buffers directly (skip copy to staging → saves VRAM + bandwidth).
    RtxOptions::useBuffersDirectlyObject().setDeferred(true, defaults);

    // --- Fallback lighting ---
    // D3D11 has no legacy lighting API — all lighting is shader-driven,
    // so Remix never receives explicit light definitions from the application.
    // Force the fallback light to Always so the scene is lit even if there are
    // no Remix USD light assets placed yet.  Use a bright distant light that
    // produces reasonable illumination for most indoor/outdoor scenes.
    LightManager::fallbackLightModeObject().setDeferred(LightManager::FallbackLightMode::Always, defaults);
    LightManager::fallbackLightTypeObject().setDeferred(LightManager::FallbackLightType::Distant, defaults);
    LightManager::fallbackLightRadianceObject().setDeferred(Vector3(4.0f, 4.0f, 4.0f), defaults);
    LightManager::fallbackLightDirectionObject().setDeferred(Vector3(-0.3f, -1.0f, 0.5f), defaults);
    LightManager::fallbackLightAngleObject().setDeferred(5.0f, defaults);

    // Start with the Remix developer menu visible so users can verify Remix is
    // active.  showUI has NoSave flag, so rtx.conf cannot override it through
    // the normal config layer path — setting it here on Default ensures the
    // menu is open on first frame.  Alt+X toggle still works (writes to Derived).
    RtxOptions::showUIObject().setDeferred(UIType::Advanced, defaults);
  }

  // NV-DXVK: return-value helper. True = skip D3D11 native rasterization
  // (RT already owns the output), false = emit native raster via EmitCs.
  // Once Remix is active we normally suppress ALL subsequent raster to avoid
  // the "shared RT target" write hazards documented on m_remixActiveThisFrame,
  // BUT UI/HUD draws must rasterize natively or they never appear — the RT
  // composite doesn't include them. So: if THIS draw was RT-captured, or
  // the frame already had RT activity AND this draw was NOT UI-classified,
  // return true. If this draw was filtered as UI, always return false so
  // the native raster path runs (and the HUD shows up).
  // NOTE: RT's final blit in rtx_context.cpp:725 currently copies the RT
  // output OVER the backbuffer, which can still clobber the UI pixels that
  // native raster just wrote. Full fix requires either deferring UI emits
  // past injectRTX or a masked composite; this change is the necessary-
  // but-not-sufficient first step.
  bool D3D11Rtx::OnDraw(UINT vertexCount, UINT startVertex) {
    ++m_rawDrawCount;
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    SubmitDraw(false, vertexCount, startVertex, 0);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawIndexed(UINT indexCount, UINT startIndex, INT baseVertex) {
    ++m_rawDrawCount;
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    SubmitDraw(true, indexCount, startIndex, baseVertex);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawInstanced(UINT vertexCountPerInstance, UINT instanceCount, UINT startVertex, UINT startInstance) {
    ++m_rawDrawCount;
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    SubmitInstancedDraw(false, vertexCountPerInstance, startVertex, 0, instanceCount, startInstance);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  bool D3D11Rtx::OnDrawIndexedInstanced(UINT indexCountPerInstance, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance) {
    ++m_rawDrawCount;
    m_lastDrawCaptured = false;
    m_lastDrawFilteredAsUI = false;
    SubmitInstancedDraw(true, indexCountPerInstance, startIndex, baseVertex, instanceCount, startInstance);
    if (m_lastDrawCaptured) m_remixActiveThisFrame = true;
    if (m_lastDrawFilteredAsUI) return false;
    return m_remixActiveThisFrame;
  }

  void D3D11Rtx::SubmitInstancedDraw(bool indexed, UINT count, UINT start, INT base,
                                      UINT instanceCount, UINT startInstance) {
    try {
    if (instanceCount <= 1) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    // Find per-instance float4 rows in the input layout that form a world matrix.
    // Engines encode this as 3 or 4 consecutive float4 elements with per-instance step rate,
    // using semantics like INSTANCETRANSFORM, WORLD, I, INST, or TEXCOORD at high indices.
    auto* layout = m_context->m_state.ia.inputLayout.ptr();
    if (!layout) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    const auto& semantics = layout->GetRtxSemantics();

    struct Float4Row {
      uint32_t inputSlot;
      uint32_t byteOffset;
    };

    std::vector<Float4Row> instRows;
    uint32_t instSlot = UINT32_MAX;

    for (const auto& s : semantics) {
      if (!s.perInstance) continue;
      if (s.format != VK_FORMAT_R32G32B32A32_SFLOAT) continue;

      // Accept any per-instance float4 — most engines use INSTANCETRANSFORM, WORLD, I, INST,
      // or repurpose TEXCOORD with high indices. The key signal is per-instance + float4.
      if (instSlot == UINT32_MAX)
        instSlot = s.inputSlot;

      // Only collect rows from the same input slot.
      if (s.inputSlot != instSlot) continue;
      instRows.push_back({s.inputSlot, s.byteOffset});
    }

    if (instRows.size() < 3) {
      // NV-DXVK: Source Engine 2 bone-index instancing.
      // Per-instance data is R16G16B16A16_UINT containing bone indices,
      // NOT float4 transform rows.  The actual transforms are in VS SRV t30
      // (g_boneMatrix, StructuredBuffer<float3x4>, stride=48).
      // Read the bone index for each instance, fetch the float3x4 from t30,
      // and use it as the instance world transform.
      bool handledAsBoneInstancing = false;
      const D3D11RtxSemantic* boneIdxSem = nullptr;
      ID3D11ShaderResourceView* boneSrv = nullptr;
      {
        // Find per-instance UINT semantic. Accepts both:
        //   R16G16B16A16_UINT — legacy skinned-character per-instance bone idx
        //   R32G32_UINT       — TF2 BSP / batched-prop per-instance modelInst idx
        for (const auto& s : semantics) {
          if (!s.perInstance) continue;
          if (s.format == VK_FORMAT_R16G16B16A16_UINT
              || s.format == VK_FORMAT_R32G32_UINT
              || s.format == VK_FORMAT_R32G32B32A32_UINT) {
            boneIdxSem = &s;
            break;
          }
        }
        // DEBUG: count fanout entries by semantic format
        if (boneIdxSem) {
          static uint32_t sFanoutR16 = 0, sFanoutR32x2 = 0, sFanoutR32x4 = 0;
          uint32_t* counter = (boneIdxSem->format == VK_FORMAT_R16G16B16A16_UINT) ? &sFanoutR16
                            : (boneIdxSem->format == VK_FORMAT_R32G32_UINT) ? &sFanoutR32x2
                            : &sFanoutR32x4;
          if ((*counter) < 5) {
            ++(*counter);
            Logger::info(str::format(
              "[D3D11Rtx] FanoutSem: fmt=", uint32_t(boneIdxSem->format),
              " perInst=", boneIdxSem->perInstance ? 1 : 0,
              " slot=", boneIdxSem->inputSlot,
              " byteOff=", boneIdxSem->byteOffset,
              " counts(R16=", sFanoutR16, " R32x2=", sFanoutR32x2,
              " R32x4=", sFanoutR32x4, ")"));
          }
        }
        // NV-DXVK: deterministic slot selection via RDEF — ask the VS itself
        // whether it reads g_modelInst (BSP) or g_boneMatrix (skinned). Falls
        // back to blind t31/t30 probing for shaders without RDEF.
        uint32_t usedSlot = 0;
        bool isModelInstFanout = false;
        {
          uint32_t modelInstSlot = UINT32_MAX, boneMatrixSlot = UINT32_MAX;
          auto vsPtr = m_context->m_state.vs.shader;
          if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
            const D3D11CommonShader* common = vsPtr->GetCommonShader();
            modelInstSlot  = common->FindResourceSlot("g_modelInst");
            boneMatrixSlot = common->FindResourceSlot("g_boneMatrix");
          }
          if (modelInstSlot != UINT32_MAX
              && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            boneSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
            if (boneSrv) { usedSlot = modelInstSlot; isModelInstFanout = true; }
          }
          if (!boneSrv && boneMatrixSlot != UINT32_MAX
              && boneMatrixSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            boneSrv = m_context->m_state.vs.shaderResources.views[boneMatrixSlot].ptr();
            if (boneSrv) usedSlot = boneMatrixSlot;
          }
          if (!boneSrv) {
            // Last-resort blind probe — RDEF didn't tell us which slot the VS
            // reads. Log loudly: every blind hit is a shader we can't classify
            // deterministically and may mis-route (BSP transforms vs bone
            // matrices). Either RDEF was stripped or the resource has a name
            // we don't recognize.
            const uint32_t kInstTransformSlot = 31, kBoneSrvSlot = 30;
            if (kInstTransformSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
              boneSrv = m_context->m_state.vs.shaderResources.views[kInstTransformSlot].ptr();
              if (boneSrv) { usedSlot = kInstTransformSlot; isModelInstFanout = true; }
            }
            if (!boneSrv && kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
              boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();
              if (boneSrv) usedSlot = kBoneSrvSlot;
            }
            if (boneSrv) {
              static std::unordered_set<uintptr_t> sBlindLogged;
              auto vsPtr = m_context->m_state.vs.shader;
              uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
              if (key && sBlindLogged.insert(key).second) {
                std::string vsHash = "?";
                if (vsPtr->GetCommonShader() != nullptr) {
                  auto& s = vsPtr->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash = s->getShaderKey().toString();
                }
                Logger::err(str::format(
                  "[D3D11Rtx] BLIND-PROBE fanout for VS=", vsHash,
                  " (RDEF lookup found neither g_modelInst nor g_boneMatrix)",
                  " — guessing slot=", usedSlot,
                  " isModelInst=", isModelInstFanout ? 1 : 0,
                  ". This shader will use heuristic routing and may be wrong."));
              }
            }
          }
        }
        static bool sLoggedSlot = false;
        if (!sLoggedSlot && boneSrv) {
          sLoggedSlot = true;
          Com<ID3D11Resource> r; boneSrv->GetResource(&r);
          auto* b = static_cast<D3D11Buffer*>(r.ptr());
          Logger::info(str::format("[D3D11Rtx] Using SRV slot ", usedSlot,
            " bufSize=", (b ? b->Desc()->ByteWidth : 0),
            " usage=", (b ? b->Desc()->Usage : 0),
            " hasImmData=", (b ? b->GetImmutableData().size() : 0)));
        }

        if (boneIdxSem && boneSrv) {
          // Get the bone matrix buffer
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          DxvkBufferSlice boneBufSlice = boneBuf ? boneBuf->GetBufferSlice() : DxvkBufferSlice();
          const uint8_t* bonePtr = boneBufSlice.defined() ?
            reinterpret_cast<const uint8_t*>(boneBufSlice.mapPtr(0)) : nullptr;
          const size_t boneBufLen = boneBufSlice.defined() ? boneBufSlice.length() : 0;

          // Get the per-instance index buffer
          const auto& instVb = m_context->m_state.ia.vertexBuffers[boneIdxSem->inputSlot];
          const uint8_t* boneReadPtr = bonePtr;
          size_t boneReadLen = boneBufLen;

          // Try multiple paths for bone buffer if direct mapPtr failed
          if (!boneReadPtr && boneBuf) {
            const auto mapped = boneBuf->GetMappedSlice();
            if (mapped.mapPtr && mapped.length >= 48) {
              boneReadPtr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              boneReadLen = mapped.length;
            }
          }
          if (!boneReadPtr && boneBuf) {
            void* p = boneBuf->GetBuffer()->mapPtr(0);
            if (p) {
              boneReadPtr = reinterpret_cast<const uint8_t*>(p);
              boneReadLen = boneBuf->GetBuffer()->info().size;
            }
          }

          // Read IMMUTABLE instance buffer data from CPU cache (set at CreateBuffer time).
          if (instVb.buffer != nullptr && m_cachedInstBufPtr != instVb.buffer.ptr()) {
            const auto& immData = instVb.buffer->GetImmutableData();
            if (!immData.empty()) {
              m_instBufCache = immData;
              m_cachedInstBufPtr = instVb.buffer.ptr();
            }
          }

          // Log VS hash on first bone-instanced draw (to find the shader)
          static bool sLoggedVsHash = false;
          if (!sLoggedVsHash) {
            sLoggedVsHash = true;
            auto vsShader = m_context->m_state.vs.shader;
            if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
              auto& shader = vsShader->GetCommonShader()->GetShader();
              if (shader != nullptr) {
                Logger::info(str::format(
                  "[D3D11Rtx] Bone-instanced VS hash: ",
                  shader->getShaderKey().toString()));
              }
            }
          }

          // 1 BLAS + N TLAS instances via instancesToObject.
          // Per-instance transforms are in the DYNAMIC t31 buffer (208 bytes/instance).
          m_boneInstBatches++;

          // Read the t31 buffer (per-instance world transforms) directly.
          // It's DYNAMIC (MAP_WRITE_DISCARD) so we use GetMappedSlice().
          const uint8_t* t31Data = nullptr;
          size_t t31Len = 0;
          {
            Com<ID3D11Resource> res;
            boneSrv->GetResource(&res);
            auto* t31Buf = static_cast<D3D11Buffer*>(res.ptr());
            if (t31Buf) {
              auto mapped = t31Buf->GetMappedSlice();
              if (mapped.mapPtr && mapped.length > 0) {
                t31Data = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                t31Len = mapped.length;
              } else {
                // Fallback: try the underlying DxvkBuffer's slice
                void* p = t31Buf->GetBuffer()->mapPtr(0);
                if (p) {
                  t31Data = reinterpret_cast<const uint8_t*>(p);
                  t31Len = t31Buf->GetBuffer()->info().size;
                }
              }
            }
          }

          // Debug logging — fires after frame 50 (when user has loaded into scene)
          static uint32_t sFrameCount = 0;
          static uint32_t sDumpedAll = 0;
          sFrameCount++;

          // Track a specific batch's t31 translation across frames to see
          // if it's view-dependent (changes when camera moves) or stable.
          static uint64_t sTrackedKey = 0;
          static uint32_t sTrackedCount = 0;
          if (t31Data && t31Len >= 48 && sTrackedCount < 30) {
            // Pick the first batch we see and track its t31[0] every frame
            uint64_t myKey = reinterpret_cast<uintptr_t>(boneSrv);
            if (sTrackedKey == 0) sTrackedKey = myKey;
            if (myKey == sTrackedKey) {
              const float* m = reinterpret_cast<const float*>(t31Data);
              ++sTrackedCount;
              Logger::info(str::format(
                "[D3D11Rtx] Track srv=", myKey, " frame=", sFrameCount,
                " t31[0].T=(", m[3], ",", m[7], ",", m[11], ")"));
            }
          }

          // #3: Dump FULL t31 matrices ONCE after 50 frames (rotation + translation)
          if (sFrameCount > 50 && sDumpedAll < 1 && t31Data && t31Len >= 48) {
            ++sDumpedAll;
            uint32_t numInst = static_cast<uint32_t>(t31Len / 208);
            // Full matrix dump for first 2 instances — check scale/rotation/translation
            for (uint32_t k = 0; k < std::min(numInst, 2u); ++k) {
              const float* m = reinterpret_cast<const float*>(t31Data + k * 208);
              // compute magnitude of each row (scale per axis)
              float mag0 = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
              float mag1 = std::sqrt(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);
              float mag2 = std::sqrt(m[8]*m[8] + m[9]*m[9] + m[10]*m[10]);
              Logger::info(str::format(
                "[D3D11Rtx] T31 mat[", k, "]:"
                " r0=(", m[0], ",", m[1], ",", m[2], ") T0=", m[3], " mag=", mag0,
                " r1=(", m[4], ",", m[5], ",", m[6], ") T1=", m[7], " mag=", mag1,
                " r2=(", m[8], ",", m[9], ",", m[10], ") T2=", m[11], " mag=", mag2));
            }
            std::string dump = str::format("inst=", numInst);
            for (uint32_t k = 0; k < numInst && k * 208 + 48 <= t31Len; ++k) {
              const float* m = reinterpret_cast<const float*>(t31Data + k * 208);
              dump += str::format(" [", k, "]=(", m[3], ",", m[7], ",", m[11], ")");
            }
            Logger::info(str::format("[D3D11Rtx] DumpAllT31: ", dump));

            // (vertex decode test removed — Z offset bug already fixed from shader decomp)

            // #5: log cb0, cb2, cb3 sizes
            const auto& vsCbs = m_context->m_state.vs.constantBuffers;
            for (uint32_t sl = 0; sl < 4; ++sl) {
              if (vsCbs[sl].buffer != nullptr) {
                Logger::info(str::format("[D3D11Rtx] VS cb[", sl, "] size=",
                  vsCbs[sl].buffer->Desc()->ByteWidth,
                  " off=", vsCbs[sl].constantOffset));
              }
            }
          }

          if (!m_instBufCache.empty() && t31Data) {
            const UINT maxInstances = std::min(instanceCount, RtxOptions::maxInstanceSubmissions());
            constexpr uint32_t BYTES_PER_INSTANCE = 208;

            // NV-DXVK (TF2 BSP): t31 stores objectToCameraRelative transforms
            // (vertex buffers hold world - cameraOrigin). Remix's camera, NRC,
            // denoisers, and motion-vector systems live in absolute world, so
            // we ADD c_cameraOrigin to each per-instance translation at push
            // time to shift BSP from camera-relative into absolute world.
            // camOrigin is read from CBufCommonPerCamera offset 4 below and
            // applied in the tforms loop.
            float camOrigin[3] = { 0.0f, 0.0f, 0.0f };
            bool haveCamOrigin = false;
            // DEBUG: log failure reason once per unique VS
            const char* failReason = nullptr;
            {
              auto vsPtr4 = m_context->m_state.vs.shader;
              if (vsPtr4 == nullptr || vsPtr4->GetCommonShader() == nullptr) {
                failReason = "no_common_shader";
              } else {
                const D3D11CommonShader* common = vsPtr4->GetCommonShader();
                auto camLoc = common->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
                if (!camLoc) {
                  failReason = "FindCBField_returned_null";
                } else if (camLoc->size < 12) {
                  failReason = "size<12";
                } else if (camLoc->slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                  failReason = "slot_oob";
                } else {
                  const auto& vsCbs2 = m_context->m_state.vs.constantBuffers;
                  const auto& cb = vsCbs2[camLoc->slot];
                  if (cb.buffer == nullptr) {
                    failReason = "cb_buffer_null";
                  } else {
                    const auto mapped = cb.buffer->GetMappedSlice();
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                    const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + camLoc->offset;
                    if (!p) {
                      failReason = "mapPtr_null";
                    } else if (base + 12 > cb.buffer->Desc()->ByteWidth) {
                      failReason = "base+12_oob";
                    } else {
                      const float* fp = reinterpret_cast<const float*>(p + base);
                      if (!std::isfinite(fp[0]) || !std::isfinite(fp[1]) || !std::isfinite(fp[2])) {
                        failReason = "non_finite";
                      } else {
                        camOrigin[0] = fp[0]; camOrigin[1] = fp[1]; camOrigin[2] = fp[2];
                        haveCamOrigin = true;
                        // NV-DXVK: publish to m_lastFanoutCamOrigin ONLY if
                        // this draw is from the MAIN gameplay camera pass, not
                        // a shadow cascade / reflection probe / cubemap etc.
                        // Heuristic: main pass has a non-square, target-sized
                        // viewport (e.g. 2560x1440). Shadow cascades use square
                        // viewports (1024x1024). Reflection probes use tiny
                        // off-screen RTs. Without this filter fanout publishes
                        // ~15+ different origins per frame and path 1/3 end up
                        // using whichever fanned out last → chaos.
                        bool isMainViewport = false;
                        {
                          const auto& vps = m_context->m_state.rs.viewports;
                          const float vw = vps[0].Width;
                          const float vh = vps[0].Height;
                          if (vw > 0.0f && vh > 0.0f) {
                            const float asp = vw / vh;
                            const bool nonSquare = std::abs(asp - 1.0f) > 0.02f;
                            const bool bigEnough = vw >= 1024.0f && vh >= 600.0f;
                            isMainViewport = nonSquare && bigEnough;
                          }
                        }
                        if (isMainViewport) {
                          const bool changed =
                            !m_hasFanoutCamOrigin
                            || std::abs(m_lastFanoutCamOrigin.x - fp[0]) > 0.5f
                            || std::abs(m_lastFanoutCamOrigin.y - fp[1]) > 0.5f
                            || std::abs(m_lastFanoutCamOrigin.z - fp[2]) > 0.5f;
                          m_lastFanoutCamOrigin = Vector3(fp[0], fp[1], fp[2]);
                          m_hasFanoutCamOrigin = true;
                          // NV-DXVK: capture the VP rows at the SAME cb/offset
                          // we just pulled camOrigin from. CBufCommonPerCamera
                          // lives at camLoc->slot; c_cameraRelativeToClip is at
                          // +16 (current frame VP) and ...PrevFrame at +96. We
                          // prefer +16 but fall back to +96 when +16 is still
                          // identity (very early frames). The same cb is
                          // authoritative for "the gameplay pose" when read
                          // from the fanout VS — path 3 should reuse this
                          // rather than reading cb2@96 of whichever VS it
                          // happens to be running under.
                          {
                            const size_t vpBaseCurr =
                              static_cast<size_t>(cb.constantOffset) * 16 + 16;
                            const size_t vpBasePrev =
                              static_cast<size_t>(cb.constantOffset) * 16 + 96;
                            const size_t bsz = cb.buffer->Desc()->ByteWidth;
                            auto tryReadVP = [&](size_t b) -> bool {
                              if (b + 64 > bsz) return false;
                              const float* vp = reinterpret_cast<const float*>(p + b);
                              for (int k = 0; k < 12; ++k)
                                if (!std::isfinite(vp[k])) return false;
                              Vector3 r0(vp[0], vp[1], vp[2]);
                              Vector3 r1(vp[4], vp[5], vp[6]);
                              Vector3 r2(vp[8], vp[9], vp[10]);
                              // Reject identity — means VP not yet populated.
                              if (std::abs(r0.x - 1.0f) < 1e-4f
                                  && std::abs(r1.y - 1.0f) < 1e-4f
                                  && std::abs(r2.z - 1.0f) < 1e-4f
                                  && std::abs(r0.y) < 1e-4f && std::abs(r0.z) < 1e-4f)
                                return false;
                              // Reject zero / degenerate rows.
                              const float l0 = length(r0), l1 = length(r1), l2 = length(r2);
                              if (l0 < 0.1f || l1 < 0.1f || l2 < 0.001f) return false;
                              m_lastFanoutVpRow0 = r0;
                              m_lastFanoutVpRow1 = r1;
                              m_lastFanoutVpRow2 = r2;
                              m_hasFanoutVpRows = true;
                              return true;
                            };
                            if (!tryReadVP(vpBaseCurr)) tryReadVP(vpBasePrev);
                          }
                          if (changed) {
                            static uint32_t sPublishLog = 0;
                            if (sPublishLog < 30) {
                              ++sPublishLog;
                              Logger::info(str::format(
                                "[D3D11Rtx.fanoutOri] publish #", sPublishLog,
                                " draw=", m_drawCallID,
                                " cam=(", fp[0], ",", fp[1], ",", fp[2], ")",
                                " vpRows=", m_hasFanoutVpRows ? 1 : 0));
                            }
                          }
                        } else {
                          // Log rejection once per unique non-main viewport so
                          // we can see what's being correctly filtered out.
                          static uint32_t sRejectLog = 0;
                          if (sRejectLog < 10) {
                            ++sRejectLog;
                            const auto& vps = m_context->m_state.rs.viewports;
                            Logger::info(str::format(
                              "[D3D11Rtx.fanoutOri] reject #", sRejectLog,
                              " vp=", int(vps[0].Width), "x", int(vps[0].Height),
                              " cam=(", fp[0], ",", fp[1], ",", fp[2], ")"));
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            if (failReason) {
              static std::unordered_set<uintptr_t> sFailLogged;
              auto vsPtr6 = m_context->m_state.vs.shader;
              uintptr_t key6 = (vsPtr6 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr6.ptr()) : 0;
              if (key6 && sFailLogged.insert(key6).second && sFailLogged.size() < 12) {
                std::string vsHash6 = "?";
                if (vsPtr6->GetCommonShader() != nullptr) {
                  auto& s = vsPtr6->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash6 = s->getShaderKey().toString();
                }
                Logger::warn(str::format(
                  "[D3D11Rtx] BSP camOrigin lookup FAILED VS=", vsHash6,
                  " reason=", failReason));
              }
            }
            // DEBUG: log camOrigin once per unique VS that uses BSP fanout.
            if (haveCamOrigin) {
              static std::unordered_set<uintptr_t> sCamOriginLogged;
              auto vsPtr5 = m_context->m_state.vs.shader;
              uintptr_t key5 = (vsPtr5 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr5.ptr()) : 0;
              if (key5 && sCamOriginLogged.insert(key5).second && sCamOriginLogged.size() < 12) {
                Logger::info(str::format(
                  "[D3D11Rtx] BSP camOrigin=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")"));
              }
            }

            // Track unique position vertex buffers this frame
            {
              uint32_t posSlot = UINT32_MAX;
              for (const auto& s : semantics) {
                if (!s.perInstance && s.format == VK_FORMAT_R32G32_UINT) { posSlot = s.inputSlot; break; }
              }
              if (posSlot != UINT32_MAX) {
                const auto& pvb = m_context->m_state.ia.vertexBuffers[posSlot];
                if (pvb.buffer != nullptr)
                  m_boneInstVbPtrs.insert(reinterpret_cast<uintptr_t>(pvb.buffer.ptr()));
              }
            }

            // ONE SubmitDraw per batch = matches original game's draw count.
            // Scene manager expands to N TLAS instances via instancesToObject.
            auto tforms = std::make_shared<std::vector<Matrix4>>();
            tforms->reserve(maxInstances);

            const uint8_t* instData = m_instBufCache.data();
            const uint32_t stride = instVb.stride;
            const uint32_t boneOff = boneIdxSem->byteOffset;
            // DEBUG: per-VS, dump the first few charIdx values + raw t31 matrix
            // so we can verify the per-instance VB actually contains valid
            // indices and the t31 lookups produce sensible matrices.
            std::string idxDumpLine;
            std::string t31DumpLine;
            bool dumpThisDraw = false;
            {
              static std::unordered_set<uintptr_t> sIdxDumpVsLogged;
              auto vsPtr2 = m_context->m_state.vs.shader;
              uintptr_t key2 = (vsPtr2 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr2.ptr()) : 0;
              if (key2 && sIdxDumpVsLogged.size() < 12 && sIdxDumpVsLogged.insert(key2).second)
                dumpThisDraw = true;
            }
            for (uint32_t i = 0; i < maxInstances; ++i) {
              size_t instOff = static_cast<size_t>(startInstance + i) * stride + boneOff;
              // Index width depends on the semantic format.
              // R16G16B16A16_UINT -> first uint16 (legacy bones)
              // R32G32_UINT / R32G32B32A32_UINT -> first uint32 (BSP)
              uint32_t charIdx = 0;
              if (boneIdxSem->format == VK_FORMAT_R16G16B16A16_UINT) {
                if (instOff + 2 <= m_instBufCache.size())
                  charIdx = *reinterpret_cast<const uint16_t*>(instData + instOff);
              } else {
                if (instOff + 4 <= m_instBufCache.size())
                  charIdx = *reinterpret_cast<const uint32_t*>(instData + instOff);
              }
              size_t t31Off = static_cast<size_t>(charIdx) * BYTES_PER_INSTANCE;
              if (dumpThisDraw && i < 6) {
                idxDumpLine += str::format(" [", i, "]=", charIdx);
                if (t31Off + 48 <= t31Len) {
                  const float* mm = reinterpret_cast<const float*>(t31Data + t31Off);
                  t31DumpLine += str::format(" T", i, "=(", mm[3], ",", mm[7], ",", mm[11], ")");
                } else {
                  t31DumpLine += str::format(" T", i, "=OOB");
                }
              }
              if (t31Off + 48 > t31Len) continue;

              const float* m = reinterpret_cast<const float*>(t31Data + t31Off);
              bool allFinite = true;
              for (int f = 0; f < 12; ++f) if (!std::isfinite(m[f])) { allFinite = false; break; }
              if (!allFinite) continue;
              if (m[0] == 0.f && m[1] == 0.f && m[2] == 0.f && m[3] == 0.f) continue;

              // NV-DXVK (fanout+camOrigin): t31 stores
              // objectToCameraRelative — a float3x4 whose translation column
              // is (mesh_world - cameraOrigin). Applying it to BLAS (plain-
              // decoded local positions) produces (world - cam), i.e. camera-
              // relative world. Remix's worldToView (kCameraAtOrigin=false)
              // expects absolute-world input and subtracts cam itself, so if
              // we leave BSP in camera-relative space, w2v double-subtracts
              // and geometry lands at (world - 2·cam) — usually entirely
              // behind the player. Shift to absolute world by adding
              // +cameraOrigin to the translation column.
              //
              // (Verified via DXBC disassembly of VS_597b7e49…: the VS does
              // clip = cb2.c_cameraRelativeToClip × (objectToCameraRelative ×
              // local + 1), which by construction produces camera-relative
              // world pre-projection. c_cameraOrigin is [unused] in the VS
              // itself — only the CB layout declares it — so reading cb2@4
              // here is safe and always gives the current camera pose.)
              const float adjTx = haveCamOrigin ? (m[3]  + camOrigin[0]) : m[3];
              const float adjTy = haveCamOrigin ? (m[7]  + camOrigin[1]) : m[7];
              const float adjTz = haveCamOrigin ? (m[11] + camOrigin[2]) : m[11];
              tforms->push_back(Matrix4(
                Vector4(m[0], m[4], m[8],  0.0f),
                Vector4(m[1], m[5], m[9],  0.0f),
                Vector4(m[2], m[6], m[10], 0.0f),
                Vector4(adjTx, adjTy, adjTz, 1.0f)));
            }

            if (dumpThisDraw) {
              std::string vsHash3 = "?";
              auto vsPtr3 = m_context->m_state.vs.shader;
              if (vsPtr3 != nullptr && vsPtr3->GetCommonShader() != nullptr) {
                auto& s = vsPtr3->GetCommonShader()->GetShader();
                if (s != nullptr) vsHash3 = s->getShaderKey().toString();
              }
              Logger::info(str::format(
                "[D3D11Rtx] InstIdxDump VS=", vsHash3,
                " maxInst=", maxInstances, " stride=", stride, " boneOff=", boneOff,
                " t31Len=", t31Len,
                " idx:", idxDumpLine,
                " t31_T:", t31DumpLine));
            }
            // NV-DXVK: scene dump. After 5s of gameplay, dump per-instance
            // BSP geometry to OBJ. Skips skinned characters (their VBs aren't
            // in immutable storage we can read here).
            if (isModelInstFanout && !tforms->empty() && SceneDump::shouldDumpThisFrame()) {
              std::lock_guard<std::mutex> lk(SceneDump::g_mutex);
              const bool firstOpen = !SceneDump::g_obj.is_open();
              SceneDump::open();
              if (firstOpen && SceneDump::g_obj.is_open()) {
                SceneDump::writeCameraMarker();
              }
              if (SceneDump::g_obj.is_open()) {
                // Find the position semantic + its VB.
                const D3D11RtxSemantic* posS = nullptr;
                for (const auto& s : semantics) {
                  if (!s.perInstance && s.format == VK_FORMAT_R32G32_UINT) { posS = &s; break; }
                }
                // Read VB + IB via immutable cache (BSP buffers are typically IMMUTABLE).
                const uint8_t* posData = nullptr; size_t posLen = 0;
                if (posS) {
                  const auto& pvb = m_context->m_state.ia.vertexBuffers[posS->inputSlot];
                  if (pvb.buffer != nullptr) {
                    const auto& imm = pvb.buffer->GetImmutableData();
                    if (!imm.empty()) {
                      posData = imm.data() + pvb.offset + posS->byteOffset;
                      posLen  = imm.size() - (pvb.offset + posS->byteOffset);
                    }
                  }
                }
                const uint8_t* idxData = nullptr; size_t idxLen = 0;
                VkIndexType ixType = VK_INDEX_TYPE_UINT16;
                if (indexed) {
                  const auto& ib = m_context->m_state.ia.indexBuffer;
                  if (ib.buffer != nullptr) {
                    const auto& imm = ib.buffer->GetImmutableData();
                    if (!imm.empty()) {
                      idxData = imm.data() + ib.offset;
                      idxLen  = imm.size() - ib.offset;
                      // ib.format is DXGI_FORMAT, map to VkIndexType.
                      ixType  = (ib.format == DXGI_FORMAT_R16_UINT)
                                  ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                    }
                  }
                }
                if (posData && (!indexed || idxData)) {
                  const uint32_t posStride = posS ? std::max<uint32_t>(8u, m_context->m_state.ia.vertexBuffers[posS->inputSlot].stride) : 8u;
                  // Decode constants — match the VS shader: scale 1/1024, bias (-1024,-1024,-2048)
                  const float kScale  = 1.0f / 1024.0f;
                  const float kBiasZ  = -2048.0f;
                  for (uint32_t inst = 0; inst < tforms->size(); ++inst) {
                    const Matrix4& T = (*tforms)[inst];
                    SceneDump::g_obj << "o BSP_" << SceneDump::g_objectsWritten++
                                     << "_inst" << inst << "\n";
                    // Determine vertex count: use either count (non-indexed) or
                    // max index seen + 1 (indexed). For simplicity, dump the first
                    // 'count' vertices for non-indexed; for indexed, dump every
                    // referenced vertex as positions and emit triangles via faces.
                    if (!indexed) {
                      for (uint32_t v = 0; v < count; ++v) {
                        size_t off = static_cast<size_t>(v) * posStride;
                        if (off + 8 > posLen) break;
                        const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
                        uint32_t xi = SceneDump::decodeX(up[0]);
                        uint32_t yi = SceneDump::decodeY(up[0], up[1]);
                        uint32_t zi = SceneDump::decodeZ(up[1]);
                        float lx = float(xi) * kScale - 1024.0f;
                        float ly = float(yi) * kScale - 1024.0f;
                        float lz = float(zi) * kScale + kBiasZ;
                        float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
                        float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
                        float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
                        SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
                      }
                      const uint32_t triCount = count / 3;
                      for (uint32_t t = 0; t < triCount; ++t) {
                        uint32_t a = SceneDump::g_baseVtx + t * 3 + 1;
                        SceneDump::g_obj << "f " << a << " " << (a+1) << " " << (a+2) << "\n";
                      }
                      SceneDump::g_baseVtx += count;
                    } else {
                      // Indexed: scan to find max used vertex, emit those, then faces.
                      const uint32_t idxStride = (ixType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
                      uint32_t maxV = 0;
                      for (uint32_t i = 0; i < count; ++i) {
                        size_t io = static_cast<size_t>(start + i) * idxStride;
                        if (io + idxStride > idxLen) { maxV = 0; break; }
                        uint32_t idx = (idxStride == 2)
                          ? *reinterpret_cast<const uint16_t*>(idxData + io)
                          : *reinterpret_cast<const uint32_t*>(idxData + io);
                        idx += static_cast<uint32_t>(std::max(base, 0));
                        if (idx > maxV) maxV = idx;
                      }
                      const uint32_t vCount = maxV + 1;
                      for (uint32_t v = 0; v < vCount; ++v) {
                        size_t off = static_cast<size_t>(v) * posStride;
                        if (off + 8 > posLen) break;
                        const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
                        uint32_t xi = SceneDump::decodeX(up[0]);
                        uint32_t yi = SceneDump::decodeY(up[0], up[1]);
                        uint32_t zi = SceneDump::decodeZ(up[1]);
                        float lx = float(xi) * kScale - 1024.0f;
                        float ly = float(yi) * kScale - 1024.0f;
                        float lz = float(zi) * kScale + kBiasZ;
                        float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
                        float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
                        float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
                        SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
                      }
                      const uint32_t triCount = count / 3;
                      for (uint32_t t = 0; t < triCount; ++t) {
                        uint32_t i0base = (start + t * 3);
                        size_t i0o = static_cast<size_t>(i0base + 0) * idxStride;
                        size_t i1o = static_cast<size_t>(i0base + 1) * idxStride;
                        size_t i2o = static_cast<size_t>(i0base + 2) * idxStride;
                        if (i2o + idxStride > idxLen) break;
                        uint32_t i0 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i0o) : *reinterpret_cast<const uint32_t*>(idxData + i0o);
                        uint32_t i1 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i1o) : *reinterpret_cast<const uint32_t*>(idxData + i1o);
                        uint32_t i2 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i2o) : *reinterpret_cast<const uint32_t*>(idxData + i2o);
                        i0 += static_cast<uint32_t>(std::max(base, 0));
                        i1 += static_cast<uint32_t>(std::max(base, 0));
                        i2 += static_cast<uint32_t>(std::max(base, 0));
                        SceneDump::g_obj << "f " << (SceneDump::g_baseVtx + i0 + 1) << " "
                                                  << (SceneDump::g_baseVtx + i1 + 1) << " "
                                                  << (SceneDump::g_baseVtx + i2 + 1) << "\n";
                      }
                      SceneDump::g_baseVtx += vCount;
                    }
                  }
                }
              }
            }

            // DEBUG: distance to closest geometry from camera, per VS.
            // In the camera-relative world frame Remix uses, the camera sits
            // at the origin — so |T| is the distance from camera to that
            // instance. We also log the absolute-world camera origin (read
            // from cb2.c_cameraOrigin earlier) for cross-reference with the
            // GPU cull shader's `cameraPosition` (which comes from
            // CameraManager and is in absolute world).
            if (isModelInstFanout && !tforms->empty()) {
              static std::unordered_set<uintptr_t> sDistLogged;
              auto vsPtr7 = m_context->m_state.vs.shader;
              uintptr_t key7 = (vsPtr7 != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr7.ptr()) : 0;
              if (key7 && sDistLogged.size() < 12 && sDistLogged.insert(key7).second) {
                float minDistSq = std::numeric_limits<float>::max();
                float maxDistSq = 0.0f;
                for (const Matrix4& tm : *tforms) {
                  const float dx = tm[3][0], dy = tm[3][1], dz = tm[3][2];
                  const float ds = dx*dx + dy*dy + dz*dz;
                  if (ds < minDistSq) minDistSq = ds;
                  if (ds > maxDistSq) maxDistSq = ds;
                }
                std::string vsHash7 = "?";
                if (vsPtr7->GetCommonShader() != nullptr) {
                  auto& s = vsPtr7->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash7 = s->getShaderKey().toString();
                }
                Logger::info(str::format(
                  "[D3D11Rtx] BSP dist VS=", vsHash7,
                  " tforms=", tforms->size(),
                  " closest=", std::sqrt(minDistSq),
                  " farthest=", std::sqrt(maxDistSq),
                  " camOriginAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                  " (camera in our frame is at origin; cullingRadius default is 5000)"));
              }
            }
            if (!tforms->empty()) {
              // NV-DXVK: log EVERY fanout submit (cap ~40 per session) so we can
              // see which camera context each PI batch belongs to — main view vs
              // shadow cascade. camOriginAbs is the absolute-world c_cameraOrigin
              // read from the VS's CBufCommonPerCamera; its value distinguishes
              // main camera from shadow cascades in TF2.
              static uint32_t sFanoutLogCount = 0;
              auto vsPtr = m_context->m_state.vs.shader;
              if (sFanoutLogCount < 40) {
                ++sFanoutLogCount;
                std::string vsHash = "?";
                if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
                  auto& s = vsPtr->GetCommonShader()->GetShader();
                  if (s != nullptr) vsHash = s->getShaderKey().toString();
                }
                Logger::info(str::format(
                  "[D3D11Rtx] FanoutSubmit #", sFanoutLogCount,
                  " VS=", vsHash,
                  " isModelInst=", isModelInstFanout ? 1 : 0,
                  " usedSlot=", usedSlot,
                  " tforms=", tforms->size(),
                  " idxFmt=", uint32_t(boneIdxSem->format),
                  " camOriginAbs=(", camOrigin[0], ",", camOrigin[1], ",", camOrigin[2], ")",
                  " sample T0=(", (*tforms)[0][3][0], ",", (*tforms)[0][3][1], ",", (*tforms)[0][3][2], ")"));
              }
              // Keep alive via ring buffer
              if (m_boneTransformRing.empty()) m_boneTransformRing.resize(4);
              m_boneTransformRing[m_boneInstFrameId % 4].push_back(tforms);

              m_currentInstancesToObject = tforms.get();
              // NV-DXVK: Carry ownership alongside the raw pointer so the RtInstance
              // consuming this survives beyond the 4-frame ring buffer.
              m_currentInstancesToObjectOwner = tforms;
              m_boneInstanceCount = static_cast<uint32_t>(tforms->size());
              m_boneInstTotal += m_boneInstanceCount;
              SubmitDraw(indexed, count, start, base);
              m_boneInstanceCount = 0;
              m_currentInstancesToObject = nullptr;
              m_currentInstancesToObjectOwner.reset();
            }
            handledAsBoneInstancing = true;
          } else {
            m_boneInstNoCache++;
            handledAsBoneInstancing = true;
          }
        }
      }

      // Old single-draw bone path removed — handled by async extract above.

      if (!handledAsBoneInstancing) {
        static uint32_t sNoInstXformLog = 0;
        if (sNoInstXformLog < 3) {
          ++sNoInstXformLog;
          Logger::info(str::format("[D3D11Rtx] Instanced draw (", instanceCount,
                                   " instances) has no per-instance transform (", instRows.size(),
                                   " float4 rows). Submitting single draw."));
        }
        SubmitDraw(indexed, count, start, base);
      }
      return;
    }

    // Read the instance buffer
    const auto& vb = m_context->m_state.ia.vertexBuffers[instSlot];
    if (vb.buffer == nullptr) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    DxvkBufferSlice instBufSlice = vb.buffer->GetBufferSlice(vb.offset);
    const uint32_t instStride = vb.stride;
    const size_t instBufLen = instBufSlice.length();
    if (instStride == 0) {
      SubmitDraw(indexed, count, start, base);
      return;
    }

    // Cap to avoid excessive submission — configurable via rtx.maxInstanceSubmissions
    const UINT maxInstances = std::min(instanceCount, RtxOptions::maxInstanceSubmissions());

    static uint32_t sInstLog = 0;
    if (sInstLog < 3) {
      ++sInstLog;
      Logger::info(str::format("[D3D11Rtx] Instanced draw: ", instanceCount,
                               " instances, ", instRows.size(), " float4 rows in slot ",
                               instSlot, ", stride=", instStride));
    }

    for (UINT i = 0; i < maxInstances; ++i) {
      UINT instIdx = startInstance + i;
      size_t instOffset = static_cast<size_t>(instIdx) * instStride;

      // Read 3 or 4 float4 rows to build a world matrix.
      // Row layout: each row is at instOffset + row.byteOffset within the instance buffer.
      float rows[4][4] = {};
      bool valid = true;

      for (size_t r = 0; r < std::min<size_t>(instRows.size(), 4); ++r) {
        size_t rowOff = instOffset + instRows[r].byteOffset;
        if (rowOff + 16 > instBufLen) { valid = false; break; }
        const void* ptr = instBufSlice.mapPtr(rowOff);
        if (!ptr) { valid = false; break; }
        std::memcpy(rows[r], ptr, 16);
        for (int c = 0; c < 4; ++c) {
          if (!std::isfinite(rows[r][c])) { valid = false; break; }
        }
        if (!valid) break;
      }

      if (!valid) continue;

      // If only 3 rows, the 4th row is (0,0,0,1) — affine transform.
      if (instRows.size() == 3) {
        rows[3][0] = 0.f; rows[3][1] = 0.f; rows[3][2] = 0.f; rows[3][3] = 1.f;
      }

      Matrix4 instMatrix(
        Vector4(rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
        Vector4(rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
        Vector4(rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
        Vector4(rows[3][0], rows[3][1], rows[3][2], rows[3][3]));

      SubmitDraw(indexed, count, start, base, &instMatrix);
    }
    } catch (const std::exception& e) {
      Logger::err(str::format("[D3D11Rtx] CRASH in SubmitInstancedDraw: ", e.what()));
    } catch (...) {
      Logger::err("[D3D11Rtx] CRASH in SubmitInstancedDraw: unknown exception");
    }
  }

  // Read a row-major float4x4 from a mapped cbuffer.  Returns identity on bounds violation
  // or if any element is NaN/Inf (corrupt GPU memory, emulator artifacts, etc.).
  static Matrix4 readCbMatrix(const uint8_t* ptr, size_t offset, size_t bufSize) {
    if (offset + 64 > bufSize)
      return Matrix4();
    float raw[4][4];
    std::memcpy(raw, ptr + offset, 64);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        if (!std::isfinite(raw[r][c]))
          return Matrix4();
    return Matrix4(
      Vector4(raw[0][0], raw[0][1], raw[0][2], raw[0][3]),
      Vector4(raw[1][0], raw[1][1], raw[1][2], raw[1][3]),
      Vector4(raw[2][0], raw[2][1], raw[2][2], raw[2][3]),
      Vector4(raw[3][0], raw[3][1], raw[3][2], raw[3][3]));
  }

  // Detect a perspective projection matrix in either memory layout.
  //
  // Row-major layout (D3D standard, CryEngine, id Tech, Source):
  //   m[0] = [±Sx, 0,   0,    0  ]
  //   m[1] = [0,  ±Sy,  0,    0  ]
  //   m[2] = [Jx,  Jy,  Q,   ±1 ]  ← perspective-divide at m[2][3]
  //   m[3] = [0,   0,   Wz,   0  ]
  //
  // Column-major read as row-major (UE4/UE5, Unity, Godot):
  //   m[0] = [±Sx, 0,   0,    0  ]
  //   m[1] = [0,  ±Sy,  0,    0  ]
  //   m[2] = [Jx,  Jy,  Q,   Wz ]  ← m[2][3] = nearPlane or 0
  //   m[3] = [0,   0,  ±1,    0  ]  ← perspective-divide at m[3][2]
  //
  // Returns: 0 = not perspective, 1 = row-major pure P, 2 = column-major-as-row pure P,
  //          3 = row-major combined View*Proj, 4 = column-major combined View*Proj.
  //
  // allowCombinedVP: when false, only cls 1/2 (pure projection) are returned.
  // This prevents false positives from degenerate matrices on splash/UI
  // frames that happen to have m[2][3]≈±1 and m[3][3]≈0 but are NOT real
  // view*projection matrices.  The caller should set this to true only when
  // the current frame has enough draws to be confident it's gameplay.
  static int classifyPerspective(const Matrix4& m, bool allowCombinedVP = true) {
    constexpr float kTol = 0.02f;
    constexpr float kJitterTol = 0.15f;

    // ---- Pure projection (diagonal rows 0-1) ----
    // These match a standalone projection matrix that the engine stores
    // separately from the view transform.  Rows 0-1 must be diagonal
    // (no off-axis terms), which is true for standard D3D perspective.
    const bool diag01 =
        std::abs(m[0][1]) <= kTol && std::abs(m[0][2]) <= kTol && std::abs(m[0][3]) <= kTol &&
        std::abs(m[1][0]) <= kTol && std::abs(m[1][2]) <= kTol && std::abs(m[1][3]) <= kTol &&
        std::abs(m[0][0]) >= 0.1f && std::abs(m[1][1]) >= 0.1f;

    if (diag01) {
      // Row-major check: m[2][3] ≈ ±1, m[3][3] ≈ 0.
      const bool r23 = std::abs(std::abs(m[2][3]) - 1.0f) < kTol;
      const bool r33z = std::abs(m[3][3]) < kTol;
      if (r23 && r33z) {
        if (std::abs(m[2][0]) <= kJitterTol && std::abs(m[2][1]) <= kJitterTol &&
            std::abs(m[3][0]) <= kTol && std::abs(m[3][1]) <= kTol)
          return 1;
      }

      // Column-major-as-row check: m[3][2] ≈ ±1, m[3][3] ≈ 0.
      const bool c32 = std::abs(std::abs(m[3][2]) - 1.0f) < kTol;
      const bool c33z = std::abs(m[3][3]) < kTol;
      if (c32 && c33z) {
        if (std::abs(m[2][0]) <= kJitterTol && std::abs(m[2][1]) <= kJitterTol &&
            std::abs(m[3][0]) <= kTol && std::abs(m[3][1]) <= kTol)
          return 2;
      }
    }

    // ---- Combined View*Projection (Source engine, Titanfall 2, etc.) ----
    //
    // Many engines (especially Source-based ones) store a pre-multiplied
    // View × Projection matrix in the VS cbuffer instead of separate V
    // and P matrices.  The view rotation is baked into ALL rows, so the
    // off-diagonal elements in rows 0-2 are large (they encode the camera
    // basis scaled by FOV and depth range).  The only invariant that
    // survives the multiplication is the perspective-divide signature:
    //
    //   Row-major VP:    m[2][3] ≈ ±1,  m[3][3] ≈ 0
    //   Col-major VP:    m[3][2] ≈ ±1,  m[3][3] ≈ 0
    //
    // We add a few lightweight sanity checks to avoid false positives:
    //   * All 16 entries must be finite (reject NaN/Inf garbage)
    //   * At least one entry in the first two rows must have magnitude
    //     > 0.01 (reject zero/near-zero matrices)
    //
    // When ExtractTransforms sees cls == 3 or 4, it treats the matrix as
    // a combined VP and uses it as viewToProjection directly, with
    // worldToView set to identity (since the view is already baked in).
    {
      // Finite-value check (reject garbage / padding / uninitialised data)
      bool allFinite = true;
      bool anySignificant = false;
      for (int r = 0; r < 4 && allFinite; ++r) {
        for (int c = 0; c < 4; ++c) {
          if (!std::isfinite(m[r][c])) { allFinite = false; break; }
          if (r < 2 && std::abs(m[r][c]) > 0.01f) anySignificant = true;
        }
      }

      if (allFinite && anySignificant && allowCombinedVP) {
        // Additional sanity: a real VP matrix has rows 0-1 with substantial
        // magnitudes (they encode camera right/up scaled by Sx/Sy ≈ 0.3-3.0
        // for typical FOVs).  Reject near-zero rows that would produce
        // degenerate Sx/Sy (≈0) and cause the decomposition to output
        // garbage (fwd=(0,0,-1), pos=(0,0,0)).  This prevents false
        // positives on orthographic/identity-like matrices that happen to
        // have the right signature in m[2][3] and m[3][3] (e.g. the 2D UI
        // ortho projection that Source stores at offset 0 of the same
        // cbuffer that holds the real VP at offset 96).
        constexpr float kMinRowMag = 0.1f;
        const float magR0 = std::sqrt(m[0][0]*m[0][0] + m[0][1]*m[0][1] + m[0][2]*m[0][2]);
        const float magR1 = std::sqrt(m[1][0]*m[1][0] + m[1][1]*m[1][1] + m[1][2]*m[1][2]);

        // Additional: at least one of rows 0-1 must have magnitude that
        // DIFFERS from 1.0 by more than 0.05.  A real VP matrix has
        // Sx = cot(fovY/2)/aspect and Sy = cot(fovY/2), which only both
        // equal 1.0 for the unlikely case of exactly 90° FOV on a 1:1
        // aspect display.  False positives from identity-like parameter
        // matrices (common in Source cbuffer slot 0) have BOTH row
        // magnitudes at exactly 1.0, which this check rejects.
        // The strongest false-positive filter: the extracted projection
        // scales Sx (= magR0) and Sy (= magR1) encode the FOV and
        // aspect ratio.  For a real VP matrix:
        //   Sx = cot(fovY/2) / viewportAspect
        //   Sy = cot(fovY/2)
        // So Sy/Sx ≈ viewportAspect (within ~20% to account for
        // non-square pixels, guard bands, etc.).
        //
        // False positives from game-parameter cbuffers have random
        // Sx/Sy ratios that almost never match the viewport.
        //
        // We can't access the viewport from this static function, so
        // we use the most common gaming aspect ratios (16:9 = 1.778,
        // 16:10 = 1.6, 21:9 = 2.333, 4:3 = 1.333) and accept if the
        // ratio is within the plausible range [1.0, 3.0].  This rejects
        // ratios like 0.25 or 4.0 that come from false positives.
        const float devFromUnit = std::max(
            std::abs(magR0 - 1.0f),
            std::abs(magR1 - 1.0f));
        const float aspectRatio = (magR0 > 0.001f) ? (magR1 / magR0) : 0.0f;

        if (magR0 >= kMinRowMag && magR1 >= kMinRowMag
            && devFromUnit > 0.05f
            && aspectRatio >= 1.0f && aspectRatio <= 3.0f) {
          // Row-major combined VP: m[2][3] ≈ ±1, m[3][3] ≈ 0
          if (std::abs(std::abs(m[2][3]) - 1.0f) < kTol && std::abs(m[3][3]) < kTol)
            return 3;

          // Column-major combined VP: m[3][2] ≈ ±1, m[3][3] ≈ 0
          if (std::abs(std::abs(m[3][2]) - 1.0f) < kTol && std::abs(m[3][3]) < kTol)
            return 4;
        }
      }
    }

    return 0;
  }

  // Return true if m looks like a camera view matrix (rigid-body: rotation + translation).
  // Expects row-major convention (or column-major already transposed by the caller).
  // The upper-left 3×3 should be approximately orthonormal and the last column [0,0,0,1].
  static bool isViewMatrix(const Matrix4& m) {
    // Row 3 must be [*, *, *, 1] (affine).
    if (std::abs(m[3][3] - 1.0f) > 0.01f) return false;
    // Columns 0-2 of rows 0-2 should have unit length (orthonormal rotation).
    for (int col = 0; col < 3; ++col) {
      float lenSq = m[0][col] * m[0][col] + m[1][col] * m[1][col] + m[2][col] * m[2][col];
      if (std::abs(lenSq - 1.0f) > 0.1f) return false;
    }
    // m[0][3], m[1][3], m[2][3] should be 0 (no perspective warp).
    if (std::abs(m[0][3]) > 0.01f || std::abs(m[1][3]) > 0.01f || std::abs(m[2][3]) > 0.01f)
      return false;
    // Reject identity — identity means "no view transform" which is not useful.
    if (isIdentityExact(m)) return false;
    return true;
  }

  DrawCallTransforms D3D11Rtx::ExtractTransforms() {
    DrawCallTransforms transforms;

    // NV-DXVK: Reset per-call.  Will be set to true below only if no real
    // perspective matrix is found in any cbuffer and the viewport fallback
    // block ends up running.  SubmitDraw reads this immediately after the
    // call returns to decide whether the draw is UI (fallback was used =>
    // skip RTX submission, let the native raster path handle it).
    m_lastExtractUsedFallback = false;
    m_lastClassifierSaidUi = false;
    m_currentDrawIsBoneTransformed = false;
    m_lastDrawCamOriginSet = false;
    m_lastWtvPathId = 0;
    m_lastO2wPathId = 0;
    m_skipViewMatrixScan = false;

    // NV-DXVK: SHADOW CLASSIFICATION — run the new pure classifier and log
    // what it says for each unique VS. This does NOT alter current behavior;
    // it only emits one log line per VS so we can A/B verify the classifier
    // matches reality before swapping the dispatcher over. Expected output:
    //   VS_6e3e6f28... → StaticWorld (rdef_cb3_CBufModelInstance)
    //   VS_ef94e6c7... → SkinnedChar (sem_blendindices_canonical_t30)
    //   VS_597b7e49... → InstancedBsp (sem_uint4+rdef_g_modelInst)
    //   VS_8027c7a1... → UI (no_signals)  [menu shaders]
    {
      auto vsPtrC = m_context->m_state.vs.shader;
      if (vsPtrC != nullptr) {
        const D3D11CommonShader* common = vsPtrC->GetCommonShader();
        static std::unordered_set<uintptr_t> sShadowLogged;
        uintptr_t key = reinterpret_cast<uintptr_t>(vsPtrC.ptr());
        if (sShadowLogged.insert(key).second) {
          const auto* il = m_context->m_state.ia.inputLayout.ptr();
          const std::vector<D3D11RtxSemantic> kEmpty;
          const auto& sems = il ? il->GetRtxSemantics() : kEmpty;
          auto cls = D3D11VsClassifier::classify(common, sems);
          std::string vsName = "?";
          if (common != nullptr) {
            auto& s = common->GetShader();
            if (s != nullptr) vsName = s->getShaderKey().toString();
          }
          Logger::info(str::format(
            "[VsClass] vs=", vsName,
            " kind=", D3D11VsClassifier::kindName(cls.kind),
            " reason=", cls.reason,
            " cb3=", cls.cb3Slot,
            " modelInst=", cls.modelInstSlot, (cls.modelInstFromRdef ? "(rdef)" : ""),
            " bonePal=",   cls.bonePaletteSlot, (cls.bonePaletteFromRdef ? "(rdef)" : "")));
        }
      }
    }

    // NV-DXVK: helper — read current bound VS hash (for per-path logging).
    // Returns truncated 16-char lowercase hex string or "<novs>".
    auto getVsHashShort = [this]() -> std::string {
      auto vsPtr = m_context->m_state.vs.shader;
      if (vsPtr == nullptr || vsPtr->GetCommonShader() == nullptr) return "<novs>";
      auto& s = vsPtr->GetCommonShader()->GetShader();
      if (s == nullptr) return "<novs>";
      std::string full = s->getShaderKey().toString();
      // Format is typically "VS_<40hexchars>". Shorten to first 19 chars
      // (VS_ + 16 hex) for log readability.
      return full.substr(0, std::min<size_t>(full.size(), 19));
    };

    // Maximum bytes to scan per cbuffer. Projection/view/world matrices are
    // always in the first few hundred bytes of a cbuffer — capping the scan
    // prevents multi-second stalls on emulators that pack all constants into
    // a single 64KB+ UBO (Xenia, Yuzu, RPCS3, Citra).
    static constexpr size_t kMaxScanBytes = 8192;  // 128 matrices

    // Compute the scannable byte range for a cbuffer binding: the intersection
    // of the bound range (constantOffset..constantOffset+constantCount) with
    // the buffer allocation, capped to kMaxScanBytes from the start of the range.
    auto cbRange = [](const D3D11ConstantBufferBinding& cb) -> std::pair<size_t, size_t> {
      const size_t bufSize = cb.buffer->Desc()->ByteWidth;
      const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
      if (base >= bufSize)
        return { 0, 0 };
      size_t end;
      if (cb.constantCount > 0)
        end = std::min(base + static_cast<size_t>(cb.constantCount) * 16, bufSize);
      else
        end = bufSize;
      if (end - base > kMaxScanBytes)
        end = base + kMaxScanBytes;
      return { base, end };
    };

    // Column-major engines (Unity, Godot) store matrices transposed in memory;
    // transposing after read normalizes them to row-major for all our checks.
    auto readMatrix = [this](const uint8_t* ptr, size_t offset, size_t bufSize) -> Matrix4 {
      Matrix4 m = readCbMatrix(ptr, offset, bufSize);
      return m_columnMajor ? transpose(m) : m;
    };

    // Viewport aspect ratio — used to score projection candidates and reject
    // shadow map / cubemap projections that don't match the screen.
    float viewportAspect = 0.0f;
    {
      const auto& vp = m_context->m_state.rs.viewports[0];
      if (vp.Height > 0.0f)
        viewportAspect = vp.Width / vp.Height;
    }

    // Score a perspective projection: higher = more likely main game camera.
    // Shadow maps have square aspect, cubemaps have 90° FOV, tool cameras
    // have extreme FOV — all score lower than a typical game camera.
    auto scorePerspective = [viewportAspect](const Matrix4& proj) -> float {
      float score = 1.0f;
      DecomposeProjectionParams dpp;
      decomposeProjection(proj, dpp);
      // Guard against degenerate decomposition (NaN/Inf from near-singular matrices).
      if (!std::isfinite(dpp.fov) || !std::isfinite(dpp.aspectRatio) || !std::isfinite(dpp.nearPlane))
        return score;
      float fovDeg = dpp.fov * (180.0f / 3.14159265f);
      if (fovDeg >= 30.0f && fovDeg <= 120.0f)
        score += 2.0f;
      else if (fovDeg >= 15.0f && fovDeg <= 150.0f)
        score += 1.0f;
      if (viewportAspect > 0.0f) {
        float diff = std::abs(std::abs(dpp.aspectRatio) - viewportAspect);
        if (diff < 0.15f)
          score += 2.0f;
        else if (diff < 0.5f)
          score += 1.0f;
      }
      if (dpp.nearPlane > 0.001f && dpp.nearPlane < 100.0f)
        score += 1.0f;
      return score;
    };

    // All shader stages to scan for camera matrices.
    // VS is most common; emulators (Dolphin, PCSX2, Xenia, Citra) and some
    // deferred renderers put camera matrices in GS, DS, or PS cbuffers.
    const D3D11ConstantBufferBindings* stageCbs[] = {
      &m_context->m_state.vs.constantBuffers,
      &m_context->m_state.gs.constantBuffers,
      &m_context->m_state.ds.constantBuffers,
      &m_context->m_state.ps.constantBuffers,
    };
    static constexpr int kNumStages = 4;
    static const char* kStageNames[] = { "VS", "GS", "DS", "PS" };

    // Scan one stage's cbuffers for the best-scoring perspective matrix.
    // classifyPerspective detects both row-major and column-major-as-row
    // layouts in a single pass, so no separate transpose pass is needed.
    auto scanStageForProj = [&](int stageIdx,
        uint32_t& outSlot, size_t& outOff, float& outScore,
        Matrix4& outMat, bool& outColMajor) -> bool
    {
      bool found = false;
      const auto& cbs = *stageCbs[stageIdx];
      for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) continue;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        auto [base, end] = cbRange(cb);
        for (size_t off = base; off + 64 <= end; off += 16) {
          Matrix4 m = readCbMatrix(ptr, off, bufSize);
          // NV-DXVK: Only allow combined VP (cls 3/4) detection once
          // we're confident this is a gameplay frame (50+ draws).
          // Splash/UI frames with 1-20 draws produce false positives
          // from degenerate ortho matrices that happen to have the
          // right m[2][3]/m[3][3] signature.
          const bool allowVP = (m_rawDrawCount > 250);
          int cls = classifyPerspective(m, allowVP);
          if (cls == 0) continue;
          // Column-major-as-row (cls==2 or 4): transpose to row-major for scoring/use.
          const bool isCol = (cls == 2 || cls == 4);
          Matrix4 normalized = isCol ? transpose(m) : m;
          float s;
          if (cls <= 2) {
            // Pure projection: use the existing FOV/aspect scorer.
            s = scorePerspective(normalized);
          } else {
            // Combined VP (cls 3/4): score by how well Sy/Sx matches
            // the viewport aspect ratio.  False positives from game
            // parameter cbuffers have random Sy/Sx ratios; the real VP
            // has Sy/Sx ≈ viewportAspect because Sx = cot(fov/2)/aspect
            // and Sy = cot(fov/2).
            const float r0mag = std::sqrt(normalized[0][0]*normalized[0][0]
                                        + normalized[0][1]*normalized[0][1]
                                        + normalized[0][2]*normalized[0][2]);
            const float r1mag = std::sqrt(normalized[1][0]*normalized[1][0]
                                        + normalized[1][1]*normalized[1][1]
                                        + normalized[1][2]*normalized[1][2]);
            const float vpAspect = (r0mag > 0.001f) ? (r1mag / r0mag) : 0.0f;
            const float diff = std::abs(vpAspect - viewportAspect);
            // Base 1.0 + up to 10.0 for perfect aspect match
            s = 1.0f + 10.0f / (1.0f + diff * 5.0f);
          }
          // NV-DXVK: Debug logging for every VP candidate found during
          // the scan.  Helps track down false positives by showing which
          // slot/offset/cls/score combinations the scanner is evaluating.
          // Gated on a one-shot latch + frame count so we don't spam on
          // every frame after the per-frame re-scan for combined VP.
          {
            static uint32_t s_scanLogCount = 0;
            if (s_scanLogCount < 20) {
              ++s_scanLogCount;
              Logger::info(str::format(
                  "[D3D11Rtx] Projection scan candidate: stage=",
                  kStageNames[stageIdx],
                  " slot=", slot, " off=", off, " cls=", cls,
                  " score=", s,
                  " diag=(", normalized[0][0], ",", normalized[1][1],
                  ",", normalized[2][2], ")",
                  " m23=", normalized[2][3], " m33=", normalized[3][3],
                  " rawDraw=", m_rawDrawCount));
            }
          }
          if (s > outScore) {
            outSlot     = slot;
            outOff      = off;
            outScore    = s;
            outMat      = normalized;
            outColMajor = isCol;
            found       = true;
          }
        }
      }
      return found;
    };

    uint32_t projSlot   = m_projSlot;
    size_t   projOffset = m_projOffset;
    int      projStage  = m_projStage;

    // --- PROJECTION: Source Engine 2 fast-path ---
    // From IDA/shader analysis: CBufCommonPerCamera (cb2) has
    // c_cameraRelativeToClipPrevFrame at offset 96 (always filled).
    // Use offset 96 (prev-frame VP) — offset 16 (current-frame VP) is
    // identity on early draws and can contain degenerate values during
    // loading/transitions that cause assertion failures in SetupByFrustum.
    // --- TF2 deterministic projection: CBufCommonPerCamera at cb2 VS.
    // Layout (from VS RDEF / shader disasm):
    //   offset  0: c_zNear
    //   offset  4: c_cameraOrigin
    //   offset 16: row_major float4x4 c_cameraRelativeToClip    ← CURRENT-FRAME VP
    //   offset 84: c_cameraOriginPrevFrame
    //   offset 96: row_major float4x4 c_cameraRelativeToClipPrevFrame ← PREV VP
    // The active VP for THIS draw is whichever the game wrote into offset 16
    // for that pass (gameplay/shadow/portal/fog/...). Remix classifies the
    // resulting camera downstream. No scoring, no multi-slot scan.
    if (projSlot == UINT32_MAX) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const uint32_t kSourceCamSlot = 2;
      const auto& srcCb = vsCbs[kSourceCamSlot];
      if (srcCb.buffer != nullptr) {
        const auto mapped = srcCb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        const size_t bufSize = srcCb.buffer->Desc()->ByteWidth;
        if (ptr && bufSize >= 160) {
          Matrix4 raw16 = readCbMatrix(ptr, 16, bufSize);
          const int cls16 = classifyPerspective(raw16, true);
          int usedCls = 0;
          if (cls16 > 0) {
            projSlot    = kSourceCamSlot;
            projOffset  = 16;
            projStage   = 0;
            m_projSlot    = kSourceCamSlot;
            m_projOffset  = 16;
            m_projStage   = 0;
            m_columnMajor = (cls16 == 2);
            usedCls = cls16;
          } else {
            Matrix4 raw96 = readCbMatrix(ptr, 96, bufSize);
            const int cls96 = classifyPerspective(raw96, true);
            if (cls96 > 0) {
              projSlot    = kSourceCamSlot;
              projOffset  = 96;
              projStage   = 0;
              m_projSlot    = kSourceCamSlot;
              m_projOffset  = 96;
              m_projStage   = 0;
              m_columnMajor = (cls96 == 2);
              usedCls = cls96;
            }
          }
          static uint32_t sFastLog = 0;
          static uint32_t sFirstPerspLog = 0;
          const bool isPersp = (usedCls > 0);
          // Log first 3 calls (including identity failures) AND the first 3
          // successful perspective picks separately, so we can see exactly
          // when real gameplay VPs start arriving at cb2@16.
          if (sFastLog < 3 || (isPersp && sFirstPerspLog < 3)) {
            if (sFastLog < 3) ++sFastLog;
            if (isPersp) ++sFirstPerspLog;
            Logger::info(str::format(
              "[D3D11Rtx] TF2 deterministic VP: offset=",
              (projSlot == kSourceCamSlot ? (int)projOffset : -1),
              " cls=", usedCls,
              " isPersp=", isPersp ? 1 : 0,
              " diag16=(", raw16[0][0], ",", raw16[1][1], ",", raw16[2][2], ")",
              " m23_16=", raw16[2][3], " m33_16=", raw16[3][3]));
          }
        }
      }
    }

    // --- PROJECTION: first-draw scan (cache miss) ---
    // Single pass across all stages — classifyPerspective handles both layouts.
    if (projSlot == UINT32_MAX) {
      float bestScore = 0.0f;
      Matrix4 bestMat;
      uint32_t bestSlot = UINT32_MAX;
      size_t bestOff = SIZE_MAX;
      int bestStage = -1;
      bool bestCol = false;

      for (int si = 0; si < kNumStages; ++si) {
        uint32_t ts = UINT32_MAX; size_t to = SIZE_MAX;
        float tsc = bestScore; Matrix4 tm; bool tc = false;
        if (scanStageForProj(si, ts, to, tsc, tm, tc) && tsc > bestScore) {
          bestScore = tsc;
          bestSlot = ts; bestOff = to; bestStage = si; bestMat = tm;
          bestCol = tc;
        }
      }

      if (bestSlot != UINT32_MAX) {
        projSlot   = bestSlot;
        projOffset = bestOff;
        projStage  = bestStage;
        m_projSlot   = bestSlot;
        m_projOffset = bestOff;
        m_projStage  = bestStage;
        m_columnMajor = bestCol;
        // NV-DXVK: Track whether the match was a combined VP so EndFrame
        // can invalidate the cache for next frame (combined VP must be
        // re-scanned because it changes with camera movement and is only
        // valid on certain draws within the frame).
        // Check: re-read the matrix and classify to determine if it's VP.
        {
          const auto& cbCheck = (*stageCbs[bestStage])[bestSlot];
          if (cbCheck.buffer != nullptr) {
            const auto mappedCheck = cbCheck.buffer->GetMappedSlice();
            const uint8_t* ptrCheck = reinterpret_cast<const uint8_t*>(mappedCheck.mapPtr);
            if (ptrCheck) {
              Matrix4 mCheck = readCbMatrix(ptrCheck, bestOff, cbCheck.buffer->Desc()->ByteWidth);
              int clsCheck = classifyPerspective(mCheck, true);
              m_projIsCombinedVP = (clsCheck >= 3);
            }
          }
        }
      }
    }

    // --- PROJECTION: validate cached location, re-scan on stale ---
    if (projSlot != UINT32_MAX && projStage >= 0 && projStage < kNumStages) {
      const auto& cbs = *stageCbs[projStage];
      const auto& cb = cbs[projSlot];
      Matrix4 proj;
      bool valid = false;
      if (cb.buffer != nullptr) {
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr) {
          Matrix4 raw = readCbMatrix(ptr, projOffset, cb.buffer->Desc()->ByteWidth);
          // NV-DXVK: Always allow VP in validation — the projection was
          // already found and classified on a previous draw.  Restricting
          // allowVP by rawDrawCount causes the cache to invalidate on
          // every early-frame draw, making all pre-250 draws fall to
          // uiFallback even though the cached projection is still valid.
          const bool allowVP = true;
          int cls = classifyPerspective(raw, allowVP);
          if (cls > 0) {
            proj = (cls == 2 || cls == 4) ? transpose(raw) : raw;
            valid = true;
            // NV-DXVK: For combined View*Proj matrices (cls 3/4), decompose
            // into a clean pure projection + a view matrix extracted from
            // the camera direction/position encoded in the VP rows.
            //
            // For row-major VP = V × P (D3D convention with P having
            // perspSign in m[2][3]):
            //
            //   Row 0 of VP = CamRight × ProjScale  (right dir scaled by Sx)
            //   Row 1 of VP = CamUp    × ProjScale  (up dir scaled by Sy)
            //   Row 2 of VP = CamFwd   × ProjScale  (fwd dir scaled by Q, + perspSign in w)
            //   Row 3 of VP = CamPos   × ProjScale  (position scaled)
            //
            // The forward direction is recoverable as normalize(VP[2][0:2]).
            // The right/up directions are recoverable as normalize(VP[0/1][0:2]).
            // The projection scales Sx, Sy are the magnitudes of those rows.
            //
            // With these we construct:
            //   P = standard perspective from Sx, Sy, conservative near/far
            //   V = rigid-body view matrix from the normalized directions + position
            if (cls == 3 || cls == 4) {
              // NV-DXVK: prefer the fanout-cached VP rows over proj[] when
              // available. The cached projection slot/offset may contain a
              // DIFFERENT VS's cb2 content than the gameplay fanout VS's, so
              // per-draw reads of proj[] can flip the basis by 90° between
              // draws. Fanout rows are captured once per frame from the
              // authoritative gameplay VS, so everyone sees the same pose.
              Vector3 vpRight = m_hasFanoutVpRows ? m_lastFanoutVpRow0
                                                  : Vector3(proj[0][0], proj[0][1], proj[0][2]);
              Vector3 vpUp    = m_hasFanoutVpRows ? m_lastFanoutVpRow1
                                                  : Vector3(proj[1][0], proj[1][1], proj[1][2]);
              Vector3 vpFwd   = m_hasFanoutVpRows ? m_lastFanoutVpRow2
                                                  : Vector3(proj[2][0], proj[2][1], proj[2][2]);

              const float magRight = length(vpRight);
              const float magUp    = length(vpUp);
              const float magFwd   = length(vpFwd);

              // Projection scales from row magnitudes.
              // Sx = |right row|, Sy = |up row|.  For the forward row
              // the magnitude encodes Q (depth scale) which we don't
              // directly need for building P -- we use conservative near/far.
              const float Sx = std::max(magRight, 0.001f);
              const float Sy = std::max(magUp,    0.001f);

              // Recover normalised camera basis.
              Vector3 fwd   = (magFwd   > 0.001f) ? vpFwd   / magFwd   : Vector3(0, 0, -1);
              Vector3 right = (magRight > 0.001f) ? vpRight / magRight : Vector3(1, 0, 0);
              // Re-derive up via RH cross. Source is right-handed; in RH
              // cross(right, fwd) = up. Previous LH variant cross(fwd, right)
              // produced up=(0,0,-1) — view inverted, rendered image rolled.
              Vector3 up = cross(right, fwd);
              const float upLen = length(up);
              if (upLen > 0.001f) up = up / upLen;
              else up = Vector3(0, 0, 1);
              // Re-derive right too so all three are perfectly orthonormal.
              right = cross(fwd, up);
              const float rightLen = length(right);
              if (rightLen > 0.001f) right = right / rightLen;

              // Camera world-space position: read c_cameraOrigin directly
              // from cb2 offset 4 (float3).  This is the current-frame
              // ground truth.  Previously we read from projOffset-16
              // (offset 80) which contained c_frameNum (garbage) +
              // c_cameraOriginPrevFrame (1 frame behind).  The heuristic
              // to skip c_frameNum was fragile and always 1 frame stale.
              //
              // CBufCommonPerCamera layout:
              //   offset  0: c_zNear        (float,  4 bytes)
              //   offset  4: c_cameraOrigin (float3, 12 bytes) ← THIS
              //   offset 16: c_cameraRelativeToClip (float4x4, 64 bytes)
              //   offset 80: c_frameNum     (int,    4 bytes)
              //   offset 84: c_cameraOriginPrevFrame (float3, 12 bytes)
              //   offset 96: c_cameraRelativeToClipPrevFrame (float4x4)
              const float perspSign = (proj[2][3] < 0.0f) ? -1.0f : 1.0f;
              float Tx = 0.0f, Ty = 0.0f, Tz = 0.0f;
              {
                bool gotCamPos = false;
                // NV-DXVK: prefer the canonical gameplay camera origin
                // captured by the BSP bone-fanout path (m_lastFanoutCamOrigin).
                // That value is authoritative — different VS permutations bind
                // different c_cameraOrigin values (reflection/shadow/mech/etc.),
                // and the VS that triggers Main latch isn't always the main
                // gameplay VS. Using the fanout-captured origin guarantees
                // path 1 and path 3 agree on Main's world position, which
                // stops NRC cache thrashing.
                if (m_hasFanoutCamOrigin) {
                  Tx = m_lastFanoutCamOrigin.x;
                  Ty = m_lastFanoutCamOrigin.y;
                  Tz = m_lastFanoutCamOrigin.z;
                  gotCamPos = true;
                }
                // Source attribution for diagnostic. "F" = fanout-cached,
                // "R" = per-shader RDEF lookup, "H" = hardcoded cb+offset-4
                // fallback, "-" = nothing (Main stays at origin for this draw).
                char sourceP1 = gotCamPos ? 'F' : '-';
                // NV-DXVK: authoritative camera origin via RDEF reflection —
                // ask the actual shader where c_cameraOrigin lives instead of
                // hardcoding offset 4 in cb2. Only used on the first frames
                // before fanout has populated m_lastFanoutCamOrigin.
                const auto vsPtrP1 = m_context->m_state.vs.shader;
                if (!gotCamPos && vsPtrP1 != nullptr && vsPtrP1->GetCommonShader() != nullptr) {
                  const auto* commonP1 = vsPtrP1->GetCommonShader();
                  auto camLocP1 = commonP1->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
                  if (camLocP1 && camLocP1->size >= 12
                      && camLocP1->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                    const auto& vsCbsP1 = m_context->m_state.vs.constantBuffers;
                    const auto& camCbP1 = vsCbsP1[camLocP1->slot];
                    if (camCbP1.buffer != nullptr) {
                      const auto mapP1 = camCbP1.buffer->GetMappedSlice();
                      const uint8_t* pP1 = reinterpret_cast<const uint8_t*>(mapP1.mapPtr);
                      const size_t baseP1 =
                        static_cast<size_t>(camCbP1.constantOffset) * 16 + camLocP1->offset;
                      if (pP1 && baseP1 + 12 <= camCbP1.buffer->Desc()->ByteWidth) {
                        const float* fp = reinterpret_cast<const float*>(pP1 + baseP1);
                        if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                          Tx = fp[0]; Ty = fp[1]; Tz = fp[2];
                          gotCamPos = true;
                          sourceP1 = 'R';
                        }
                      }
                    }
                  }
                }
                // Fallback: old hardcoded cb (the one we decomposed VP from).
                // Left in place for shaders that don't expose CBufCommonPerCamera.
                if (!gotCamPos) {
                  const size_t cbBase = static_cast<size_t>(cb.constantOffset) * 16;
                  const size_t bufSize = cb.buffer->Desc()->ByteWidth;
                  if (cbBase + 16 <= bufSize) {
                    const float* cam4 = reinterpret_cast<const float*>(ptr + cbBase + 4);
                    if (std::isfinite(cam4[0]) && std::isfinite(cam4[1]) && std::isfinite(cam4[2])
                        && (std::abs(cam4[0]) > 1.0f || std::abs(cam4[1]) > 1.0f || std::abs(cam4[2]) > 1.0f)) {
                      Tx = cam4[0]; Ty = cam4[1]; Tz = cam4[2];
                      gotCamPos = true;
                      sourceP1 = 'H';
                    }
                  }
                  if (!gotCamPos && cbBase + 96 <= bufSize) {
                    const float* cam84 = reinterpret_cast<const float*>(ptr + cbBase + 84);
                    if (std::isfinite(cam84[0]) && std::isfinite(cam84[1]) && std::isfinite(cam84[2])) {
                      Tx = cam84[0]; Ty = cam84[1]; Tz = cam84[2];
                      sourceP1 = 'H';
                    }
                  }
                }
                // Log which source path 1 used (first ~50 draws or changes).
                {
                  static uint32_t sP1Log = 0;
                  static char sLastSource = '?';
                  static Vector3 sLastValue(-1e9f, -1e9f, -1e9f);
                  const bool changed = sourceP1 != sLastSource
                    || std::abs(sLastValue.x - Tx) > 0.5f
                    || std::abs(sLastValue.y - Ty) > 0.5f
                    || std::abs(sLastValue.z - Tz) > 0.5f;
                  if (changed && sP1Log < 30) {
                    ++sP1Log;
                    sLastSource = sourceP1;
                    sLastValue = Vector3(Tx, Ty, Tz);
                    Logger::info(str::format(
                      "[D3D11Rtx.path1Cam] #", sP1Log,
                      " src=", sourceP1,
                      " cam=(", Tx, ",", Ty, ",", Tz, ")"));
                  }
                }
              }

              // Build the D3D row-major view matrix:
              //   V = [Rx  Ry  Rz  0]
              //       [Ux  Uy  Uz  0]
              //       [Fx  Fy  Fz  0]
              //       [Tx' Ty' Tz' 1]
              //
              // where T' = -dot(dir, pos) for each axis (the "eye-space translation").
              //
              // NV-DXVK EXPERIMENT: TF2 renders camera-relative — vertex buffers
              // hold (world - cameraOrigin) and t31 (objectToCameraRelative)
              // transforms place geometry relative to camera. Our TLAS therefore
              // sits in camera-at-origin space. If we encode c_cameraOrigin into
              // the view matrix, Remix's RtCamera::position = cameraOrigin in
              // world, but our TLAS entries are at small camera-relative coords —
              // rays fire from the wrong origin and miss everything. Force the
              // view translation to zero so Remix's camera sits at origin,
              // matching the TLAS frame. Only the viewmodel/particles (already
              // at identity in view space) were rendering before; with this,
              // world geometry should also be hit.
              // NV-DXVK: world-space Main camera (NOT camera-relative). Previously
              // this was true (camera at origin + all geo in camera-relative frame),
              // but NRC's spatial cache needs STABLE world coords — camera-relative
              // makes every position shift per-frame, invalidating NRC. Motion
              // vectors / denoisers also need real world-space camera motion.
              // With false, Main gets its actual world translation and BSP's
              // translate(cameraOrigin) o2w fallback produces matching world coords.
              constexpr bool kCameraAtOrigin = false;
              const float Tx_use = kCameraAtOrigin ? 0.0f : Tx;
              const float Ty_use = kCameraAtOrigin ? 0.0f : Ty;
              const float Tz_use = kCameraAtOrigin ? 0.0f : Tz;
              const float dotR = -(right.x * Tx_use + right.y * Ty_use + right.z * Tz_use);
              const float dotU = -(up.x    * Tx_use + up.y    * Ty_use + up.z    * Tz_use);
              const float dotF = -(fwd.x   * Tx_use + fwd.y   * Ty_use + fwd.z   * Tz_use);

              // NV-DXVK: construct the Matrix4 from COLUMNS.
              // dxvk's Matrix4 stores data[i] as column i, and its multiply
              // operator treats data[i] as column i. For a proper view matrix
              // where row 0 = right, row 1 = up, row 2 = fwd (so V*P produces
              // view-space coords via row-i · (P,1)), we must pass the
              // COLUMNS of that matrix to the constructor:
              //   column 0 = (right.x, up.x, fwd.x, 0)
              //   column 1 = (right.y, up.y, fwd.y, 0)
              //   column 2 = (right.z, up.z, fwd.z, 0)
              //   column 3 = (dotR, dotU, dotF, 1)
              // Previous code passed ROWS (right, up, fwd, translation) as
              // args, producing V^T instead of V. With camera at origin this
              // was invisible (V^T = V for translation-free rotations around
              // trivial axes), but with a real camera position inverse(V^T)[3]
              // != cameraPos, which is the bug the log shows.
              m_lastWtvPathId = 1; // path 1: generic VP-decomposition
              transforms.worldToView = Matrix4(
                Vector4(right.x, up.x, fwd.x, 0.0f),
                Vector4(right.y, up.y, fwd.y, 0.0f),
                Vector4(right.z, up.z, fwd.z, 0.0f),
                Vector4(dotR,    dotU,  dotF,  1.0f));

              // Build a clean pure perspective projection from the extracted scales.
              const float nearZ = 1.0f;
              const float farZ  = 20000.0f;
              const float Q     = farZ / (farZ - nearZ);
              // NV-DXVK: use D3D-style Q (positive). With V[2]=+fwd and
              // perspSign from game's own matrix, the overall view*proj
              // pipeline matches whatever convention TF2 itself uses.
              proj = Matrix4(
                Vector4(Sx,   0.0f, 0.0f,          0.0f),
                Vector4(0.0f, Sy,   0.0f,          0.0f),
                Vector4(0.0f, 0.0f, Q,             perspSign),
                Vector4(0.0f, 0.0f, -nearZ * Q,    0.0f));

              // Log decompositions periodically (every 100th) so we can
              // verify the camera position/direction tracks player movement
              // across frames without flooding the log.
              static uint32_t s_vpDecompLogCount = 0;
              ++s_vpDecompLogCount;
              if (s_vpDecompLogCount <= 3 || (s_vpDecompLogCount % 100) == 0) {
                Logger::info(str::format(
                    "[D3D11Rtx] Decomposed combined VP (cls=", cls,
                    "): Sx=", Sx, " Sy=", Sy,
                    " fwd=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
                    " pos=(", Tx, ",", Ty, ",", Tz, ")",
                    " perspSign=", perspSign));
              }
            }
          }
        }
      }

      if (!valid && projSlot == m_projSlot && projStage == m_projStage) {
        // Cached location is stale (different pass). Re-scan all stages.
        projSlot = UINT32_MAX;
        float bestScore = 0.0f;
        for (int si = 0; si < kNumStages; ++si) {
          uint32_t ts = UINT32_MAX; size_t to = SIZE_MAX;
          float tsc = bestScore; Matrix4 tm; bool tc = false;
          if (scanStageForProj(si, ts, to, tsc, tm, tc)) {
            projSlot = ts; projOffset = to; projStage = si;
            proj = tm; bestScore = tsc;
          }
        }
      }

      if (projSlot != UINT32_MAX) {
        // Strip TAA jitter — Remix does its own TAA.
        proj[2][0] = 0.0f;
        proj[2][1] = 0.0f;

        // --- AXIS AUTO-DETECTION (projection-derived) ---
        // Vote on Y-flip and LH/RH from each valid projection matrix.
        // Votes accumulate until a threshold is reached, then the setting
        // is permanently locked — no re-evaluation.  This guarantees that
        // objectToWorld transforms (and therefore geometry/spatial hashes)
        // see a consistent coordinate system for the entire session.
        {
          const bool canVote = !m_yFlipSettled || !m_lhSettled;

          if (canVote) {
            m_axisDetected = true;

            // Y-flip: negative Y scale in projection
            m_yFlipVotes += (proj[1][1] < 0.0f) ? 1 : -1;
            if (!m_yFlipSettled && std::abs(m_yFlipVotes) >= kVoteThreshold) {
              m_yFlipSettled = true;
              const bool yFlip = m_yFlipVotes > 0;
              RtCamera::correctProjectionYFlipObject().setDeferred(yFlip);
            }

            // LH/RH from projection decomposition
            DecomposeProjectionParams dpp;
            decomposeProjection(proj, dpp);
            if (std::isfinite(dpp.fov) && std::isfinite(dpp.aspectRatio)) {
              m_lhVotes += dpp.isLHS ? 1 : -1;
              if (!m_lhSettled && std::abs(m_lhVotes) >= kVoteThreshold) {
                m_lhSettled = true;
                const bool isLH = m_lhVotes > 0;
                RtxOptions::leftHandedCoordinateSystemObject().setDeferred(isLH);
              }
            }
          }

        }

        transforms.viewToProjection = proj;

        // NV-DXVK: Mark this frame as "has real projection" so subsequent
        // draws that hit the fallback path can reuse these transforms
        // instead of being filtered as UIFallback.
        m_foundRealProjThisFrame = true;
        m_hasEverFoundProj = true;
        m_lastGoodTransforms = transforms;
      }
    }

    // --- FALLBACK PROJECTION ---
    // If no perspective matrix was found in any cbuffer, synthesize one from
    // the viewport.  This gives Remix a valid camera so geometry renders at
    // roughly correct positions even when: (a) the engine packs matrices in
    // a format we don't recognize, (b) the game uses compute-based rendering,
    // or (c) all cbuffers are GPU-only / unmappable.  The fallback is
    // intentionally conservative (60° FOV, 0.1–10000 range) and is only used
    // when the real scan comes up empty.
    //
    // NV-DXVK: For Source-engine games (Titanfall 2, etc.), the main-menu
    // and HUD draws legitimately have NO perspective projection — they use
    // orthographic UI projections.  When that happens we flag the extract
    // as "used fallback" so SubmitDraw can drop the draw out of the RTX
    // pipeline (it still rasterizes natively via the EmitCs path in
    // D3D11DeviceContext::Draw*), and EndFrame's camera safety net will
    // skip firing for the frame.  That leaves injectRTX() with an invalid
    // camera, which early-returns (rtx_context.cpp:492), so the native
    // raster content in the backbuffer passes through unchanged instead of
    // being overwritten by a path-traced empty scene compressed into a
    // viewport-fallback corner.
    // NV-DXVK: Source Engine 2 last-resort — if no projection was found
    // by the generic scanner, try reading cb2@96 directly as a combined VP.
    // This is c_cameraRelativeToClipPrevFrame which is always populated
    // even on early draws where c_cameraRelativeToClip (cb2@16) is identity.
    if (projSlot == UINT32_MAX) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const auto& srcCb = vsCbs[2];
      if (srcCb.buffer != nullptr) {
        const auto mapped = srcCb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr && srcCb.buffer->Desc()->ByteWidth >= 160) {
          Matrix4 raw = readCbMatrix(ptr, 96, srcCb.buffer->Desc()->ByteWidth);
          // Check if it looks like a perspective matrix (m[2][3] == ±1)
          if (std::abs(std::abs(raw[2][3]) - 1.0f) < 0.1f) {
            projSlot = 2;
            projOffset = 96;
            projStage = 0;

            // Decompose the combined VP into projection + view
            Vector3 vpRight(raw[0][0], raw[0][1], raw[0][2]);
            Vector3 vpUp   (raw[1][0], raw[1][1], raw[1][2]);
            Vector3 vpFwd  (raw[2][0], raw[2][1], raw[2][2]);
            const float Sx = std::max(length(vpRight), 0.001f);
            const float Sy = std::max(length(vpUp),    0.001f);
            const float magFwd = length(vpFwd);
            Vector3 fwd   = magFwd > 0.001f ? vpFwd / magFwd : Vector3(0, 0, -1);
            Vector3 right = Sx > 0.001f ? vpRight / Sx : Vector3(1, 0, 0);
            // RH cross: up = right × fwd. Same fix as path 1.
            Vector3 up    = cross(right, fwd);
            float upLen = length(up);
            if (upLen > 0.001f) up = up / upLen; else up = Vector3(0, 0, 1);
            right = cross(fwd, up);
            float rightLen = length(right);
            if (rightLen > 0.001f) right = right / rightLen;

            // Camera position: try offset 4 (current frame), fall back to
            // offset 84 (prev frame) if current is zero (early draws).
            // NV-DXVK: respect cb.constantOffset — see path 1 fix comment.
            float Tx = 0, Ty = 0, Tz = 0;
            {
              const size_t cbBase = static_cast<size_t>(srcCb.constantOffset) * 16;
              const size_t bsz = srcCb.buffer->Desc()->ByteWidth;
              bool got = false;
              if (cbBase + 16 <= bsz) {
                const float* c4 = reinterpret_cast<const float*>(ptr + cbBase + 4);
                if (std::isfinite(c4[0]) && std::isfinite(c4[1]) && std::isfinite(c4[2])
                    && (std::abs(c4[0]) > 1.0f || std::abs(c4[1]) > 1.0f || std::abs(c4[2]) > 1.0f)) {
                  Tx = c4[0]; Ty = c4[1]; Tz = c4[2]; got = true;
                }
              }
              if (!got && cbBase + 96 <= bsz) {
                const float* c84 = reinterpret_cast<const float*>(ptr + cbBase + 84);
                if (std::isfinite(c84[0]) && std::isfinite(c84[1]) && std::isfinite(c84[2])) {
                  Tx = c84[0]; Ty = c84[1]; Tz = c84[2];
                }
              }
            }
            const float dotR = -(right.x*Tx + right.y*Ty + right.z*Tz);
            const float dotU = -(up.x*Tx    + up.y*Ty    + up.z*Tz);
            const float dotF = -(fwd.x*Tx   + fwd.y*Ty   + fwd.z*Tz);
            // NV-DXVK: store by columns — see path 1 fix.
            m_lastWtvPathId = 2; // path 2: TF2 cb2@96 last-resort VP-decomp
            transforms.worldToView = Matrix4(
              Vector4(right.x, up.x, fwd.x, 0),
              Vector4(right.y, up.y, fwd.y, 0),
              Vector4(right.z, up.z, fwd.z, 0),
              Vector4(dotR,    dotU, dotF, 1));

            const float perspSign = raw[2][3] < 0 ? -1.0f : 1.0f;
            const float nearZ = 1.0f, farZ = 20000.0f;
            const float Q = farZ / (farZ - nearZ);
            // NV-DXVK: D3D-style Q, matches path 3.
            transforms.viewToProjection = Matrix4(
              Vector4(Sx,   0, 0,          0),
              Vector4(0,    Sy, 0,         0),
              Vector4(0,    0, Q,          perspSign),
              Vector4(0,    0, -nearZ*Q,   0));

            m_foundRealProjThisFrame = true;
            m_lastGoodTransforms = transforms;
          }
        }
      }
    }

    // NV-DXVK: If no projection found but we've found one in a prior frame,
    // reuse the cached camera — BUT only for R32G32_UINT position draws
    // (main world geometry).  fmt=106 draws that fail projection detection
    // are shadow/depth passes with light-space transforms → applying the
    // main camera VP to them produces extreme BLAS → GPU TDR.
    if (projSlot == UINT32_MAX && m_hasEverFoundProj) {
      // Check if this draw uses R32G32_UINT position format
      bool isUintPosLayout = false;
      D3D11InputLayout* il = m_context->m_state.ia.inputLayout.ptr();
      if (il) {
        for (const auto& s : il->GetRtxSemantics()) {
          if (std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0
              && s.format == VK_FORMAT_R32G32_UINT) {
            isUintPosLayout = true;
            break;
          }
        }
      }
      if (isUintPosLayout) {
        transforms.viewToProjection = m_lastGoodTransforms.viewToProjection;
        // Use the cached worldToView from the last VP decomposition.
        // This contains the camera rotation (from VP rows) and the camera
        // position (from cb2@4).  objectToView = worldToView * objectToWorld
        // is computed at line ~1787.  With FusedWorldViewMode::View, Remix
        // then sets objectToWorld = objectToView (fusing the transforms) and
        // zeros worldToView, so geometry ends up in view space centred on
        // the camera.
        //
        // For GPU bone draws: objectToWorld = identity (interleaver applies
        // bones GPU-side → world-space output).  objectToView = worldToView.
        // After fuse: objectToWorld = worldToView, camera at origin.
        // Geometry goes world → view via the instance transform. Correct.
        // Bone matrices output CAMERA-RELATIVE positions (world pos minus
        // camera origin is baked into the bone matrix by the engine).
        // Set worldToView from the cached VP + c_cameraOrigin.
        // The view matrix scan will run later but finds nothing for early
        // draws. Setting it here ensures a valid camera for R32G32_UINT draws.
        {
          float camX = 0, camY = 0, camZ = 0;
          // NV-DXVK: prefer the canonical gameplay cameraOrigin captured by
          // BSP fanout (m_lastFanoutCamOrigin). Same reasoning as path 1.
          bool gotCamP3 = false;
          if (m_hasFanoutCamOrigin) {
            camX = m_lastFanoutCamOrigin.x;
            camY = m_lastFanoutCamOrigin.y;
            camZ = m_lastFanoutCamOrigin.z;
            gotCamP3 = true;
          }
          char sourceP3 = gotCamP3 ? 'F' : '-';
          {
            const auto vsPtrP3 = m_context->m_state.vs.shader;
            if (!gotCamP3 && vsPtrP3 != nullptr && vsPtrP3->GetCommonShader() != nullptr) {
              const auto* commonP3 = vsPtrP3->GetCommonShader();
              auto camLocP3 = commonP3->FindCBField("CBufCommonPerCamera", "c_cameraOrigin");
              if (camLocP3 && camLocP3->size >= 12
                  && camLocP3->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                const auto& vsCbsP3 = m_context->m_state.vs.constantBuffers;
                const auto& camCbP3 = vsCbsP3[camLocP3->slot];
                if (camCbP3.buffer != nullptr) {
                  const auto mapP3 = camCbP3.buffer->GetMappedSlice();
                  const uint8_t* pP3 = reinterpret_cast<const uint8_t*>(mapP3.mapPtr);
                  const size_t baseP3 =
                    static_cast<size_t>(camCbP3.constantOffset) * 16 + camLocP3->offset;
                  if (pP3 && baseP3 + 12 <= camCbP3.buffer->Desc()->ByteWidth) {
                    const float* fp = reinterpret_cast<const float*>(pP3 + baseP3);
                    if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                      camX = fp[0]; camY = fp[1]; camZ = fp[2];
                      gotCamP3 = true;
                      sourceP3 = 'R';
                    }
                  }
                }
              }
            }
          }
          const auto& camCb = m_context->m_state.vs.constantBuffers[2];
          if (!gotCamP3 && camCb.buffer != nullptr) {
            const auto mapped = camCb.buffer->GetMappedSlice();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            // Fallback: hardcoded cb2@4 with constantOffset.
            const size_t camCbBase = static_cast<size_t>(camCb.constantOffset) * 16;
            const size_t camBufSz  = camCb.buffer->Desc()->ByteWidth;
            if (p && camCbBase + 16 <= camBufSz) {
              const float* co = reinterpret_cast<const float*>(p + camCbBase + 4);
              camX = co[0]; camY = co[1]; camZ = co[2];
              sourceP3 = 'H';
            }
          }
          // Log which source path 3 used (capped, only on change).
          {
            static uint32_t sP3Log = 0;
            static char sLastSource = '?';
            static Vector3 sLastValue(-1e9f, -1e9f, -1e9f);
            const bool changed = sourceP3 != sLastSource
              || std::abs(sLastValue.x - camX) > 0.5f
              || std::abs(sLastValue.y - camY) > 0.5f
              || std::abs(sLastValue.z - camZ) > 0.5f;
            if (changed && sP3Log < 30) {
              ++sP3Log;
              sLastSource = sourceP3;
              sLastValue = Vector3(camX, camY, camZ);
              Logger::info(str::format(
                "[D3D11Rtx.path3Cam] #", sP3Log,
                " src=", sourceP3,
                " cam=(", camX, ",", camY, ",", camZ, ")"));
            }
          }
          // Read VP rotation. Prefer the cached fanout VP rows (captured at
          // the same moment as m_lastFanoutCamOrigin) so every path-3 draw
          // gets the SAME gameplay pose regardless of which VS's cb2 is
          // bound for this specific draw. Only fall back to per-draw cb2@96
          // when fanout hasn't published yet (very early boot frames).
          Vector3 right(0, -1, 0), up(0, 0, 1), fwd(1, 0, 0);  // defaults
          bool gotLiveRotation = false;
          bool usedFanoutVp = false;
          if (m_hasFanoutVpRows) {
            const Vector3& vpRight = m_lastFanoutVpRow0;
            const Vector3& vpUp    = m_lastFanoutVpRow1;
            const Vector3& vpFwd   = m_lastFanoutVpRow2;
            float magR = length(vpRight), magU = length(vpUp), magF = length(vpFwd);
            if (magR > 0.1f && magU > 0.1f && magF > 0.001f &&
                std::abs(magR - magU) > 0.01f) {
              right = vpRight / magR;
              up    = vpUp    / magU;
              fwd   = vpFwd   / magF;
              // Re-orthogonalize (same as path 1 and updated path 3).
              Vector3 up2 = cross(right, fwd);
              float upLen = length(up2);
              if (upLen > 0.001f) up = up2 / upLen;
              Vector3 right2 = cross(fwd, up);
              float rightLen = length(right2);
              if (rightLen > 0.001f) right = right2 / rightLen;
              gotLiveRotation = true;
              usedFanoutVp = true;
            }
          }
          const auto& camCb2 = m_context->m_state.vs.constantBuffers[2];
          if (!gotLiveRotation && camCb2.buffer != nullptr) {
            const auto camMapped2 = camCb2.buffer->GetMappedSlice();
            const uint8_t* camPtr = reinterpret_cast<const uint8_t*>(camMapped2.mapPtr);
            if (camPtr && camCb2.buffer->Desc()->ByteWidth >= 160) {
              const float* vp = reinterpret_cast<const float*>(camPtr + 96);
              Vector3 vpRight(vp[0], vp[1], vp[2]);
              Vector3 vpUp   (vp[4], vp[5], vp[6]);
              Vector3 vpFwd  (vp[8], vp[9], vp[10]);
              float magR = length(vpRight), magU = length(vpUp), magF = length(vpFwd);
              // Check if VP is valid (not identity — identity has mag ≈ 1 for all rows
              // and diagonal-dominant structure, while a real VP has different scales)
              if (magR > 0.1f && magU > 0.1f && magF > 0.001f &&
                  std::abs(magR - magU) > 0.01f) {  // real VP has different Sx vs Sy
                right = vpRight / magR;
                up    = vpUp    / magU;
                fwd   = vpFwd   / magF;
                // NV-DXVK (fix 6): re-orthogonalize the basis the same way path 1
                // does (VP-decomposition branch at line ~1624). TF2's combined
                // VP rows are NOT perfectly orthonormal after TAA jitter +
                // float rounding, and path 1 re-orthogonalizes while path 3
                // used the raw rows verbatim. That tiny skew was enough to
                // make the two paths produce bases that differed by a full
                // 90° permutation across successive frames (visible in the
                // log as the latch #1 vs latch #2 axis twist). Apply the same
                // RH cross-product pass here so both paths end up with
                // identical orthonormal bases and the hysteresis gate in
                // CameraManager sees a consistent forward direction.
                Vector3 up2 = cross(right, fwd);
                float upLen = length(up2);
                if (upLen > 0.001f) up = up2 / upLen;
                Vector3 right2 = cross(fwd, up);
                float rightLen = length(right2);
                if (rightLen > 0.001f) right = right2 / rightLen;
                // NO Y-flip — use raw VP rotation so inverse matches cb3
                gotLiveRotation = true;
              }
            }
          }
          // Log rotation once per frame (not per draw) to track mouse look
          static uint32_t sRotLogFrame = UINT32_MAX;
          static uint32_t sRotLogCount = 0;
          if (m_rawDrawCount < 15 && sRotLogCount < 200) {
            // Only log on first draw of each frame
            uint32_t frameApprox = sRotLogCount; // approximate
            ++sRotLogCount;
            Logger::info(str::format(
              "[D3D11Rtx] ViewRot: live=", gotLiveRotation ? 1 : 0,
              " R=(", right.x, ",", right.y, ",", right.z, ")",
              " U=(", up.x, ",", up.y, ",", up.z, ")",
              " F=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
              " cam=(", camX, ",", camY, ",", camZ, ")",
              " w2vT=(", transforms.worldToView[3][0], ",", transforms.worldToView[3][1], ",", transforms.worldToView[3][2], ")",
              " raw=", m_rawDrawCount));
          }
          // Fallback: use cached rotation from VP decomposition
          if (!gotLiveRotation) {
            const Matrix4& cachedView = m_lastGoodTransforms.worldToView;
            Vector3 cRight(cachedView[0][0], cachedView[0][1], cachedView[0][2]);
            Vector3 cUp   (cachedView[1][0], cachedView[1][1], cachedView[1][2]);
            Vector3 cFwd  (cachedView[2][0], cachedView[2][1], cachedView[2][2]);
            if (length(cRight) > 0.5f && length(cFwd) > 0.5f) {
              right = cRight; up = cUp; fwd = cFwd;
            }
          }
          // cb3 = objectToCameraRelative = objectToWorld × viewRotation.
          // We need Remix's camera to know the rotation so rays track the
          // camera direction. Set worldToView = liveRotation (from VP),
          // then objectToWorld = inverse(liveRotation) × cb3 to undo the
          // double rotation. No translation in worldToView (cb3 has it).
          //
          // Use the live VP rotation from cb2@96 as worldToView.
          // DON'T apply Y-flip — the VP rotation must match what cb3 was
          // built with so inverse(rotation) × cb3 cancels correctly.
          // The axis swap is handled by the VP rotation itself (it maps
          // Source axes to the projection's expected space).
          // Validate rotation vectors are finite
          bool rotValid = std::isfinite(right.x) && std::isfinite(right.y) && std::isfinite(right.z)
                       && std::isfinite(up.x) && std::isfinite(up.y) && std::isfinite(up.z)
                       && std::isfinite(fwd.x) && std::isfinite(fwd.y) && std::isfinite(fwd.z)
                       && length(right) > 0.5f && length(fwd) > 0.5f;
          if (rotValid) {
            // Full view matrix with rotation AND camera translation
            float tR = -(right.x*camX + right.y*camY + right.z*camZ);
            float tU = -(up.x*camX    + up.y*camY    + up.z*camZ);
            float tF = -(fwd.x*camX   + fwd.y*camY   + fwd.z*camZ);
            m_lastWtvPathId = 3; // path 3: bone-fanout primary, raw VP rotation + cb2@4 cam
            // NV-DXVK: store by columns — see path 1 fix.
            transforms.worldToView = Matrix4(
              Vector4(right.x, up.x, fwd.x, 0),
              Vector4(right.y, up.y, fwd.y, 0),
              Vector4(right.z, up.z, fwd.z, 0),
              Vector4(tR,      tU,   tF,   1));
          } else {
            // Fallback: fixed axis swap (identity-like; rows=cols for this case)
            m_lastWtvPathId = 4; // path 4: bone-fanout fallback, hardcoded axis swap
            transforms.worldToView = Matrix4(
              Vector4( 0,  0,  1, 0),
              Vector4(-1,  0,  0, 0),
              Vector4( 0,  1,  0, 0),
              Vector4( 0,  0,  0, 1));
          }
          // Skip the VIEW matrix scan only — worldToView is set.
          // Allow WORLD matrix scan to extract cb3 per-draw transforms.
          m_skipViewMatrixScan = true;
        }

        // NV-DXVK (non-instanced BSP t31 path): TF2 BSP shaders (verified
        // via DXBC disasm of VS_597b7e49…) do:
        //   clip = cb2.c_cameraRelativeToClip × (t31[v1.x].objectToCameraRelative × local + 1)
        // where t31 is g_modelInst (StructuredBuffer<ModelInstance>, stride
        // 208) and v1.x is COLOR1 (R16G16B16A16_UINT per-instance, first
        // uint16). The shader does NOT use a t30 bone matrix — that code
        // path was a wrong guess.
        //
        // For non-instanced draws (instanceCount=1), the fanout code above
        // doesn't run; we fetch t31[charIdx].objectToCameraRelative here
        // and use it as objectToWorld (plus +cameraOrigin on the translation
        // column to shift from camera-relative into absolute world, matching
        // the fanout path).
        bool gotBoneTransform = false;
        {
          // Heavy diagnostic logging: tag every step so we can see exactly
          // where/why the t31 path succeeds or fails.
          const char* t31SkipReason = nullptr;
          ID3D11ShaderResourceView* modelInstSrv = nullptr;
          uint32_t modelInstSlot = UINT32_MAX;
          bool rdefFound = false;
          {
            auto vsPtrT31 = m_context->m_state.vs.shader;
            if (vsPtrT31 != nullptr && vsPtrT31->GetCommonShader() != nullptr) {
              modelInstSlot = vsPtrT31->GetCommonShader()->FindResourceSlot("g_modelInst");
              if (modelInstSlot != UINT32_MAX) rdefFound = true;
            }
          }
          // NV-DXVK principled routing: the t31 path reads g_modelInst[idx]
          // where idx comes from a per-instance COLOR1/I:R16G16B16A16_UINT
          // semantic declared by the VS. That semantic is the authoritative
          // signal the shader is genuinely instanced against t31 — when it's
          // absent AND RDEF didn't name g_modelInst, the VS is a static mesh
          // and its transform lives in cb3.CBufModelInstance (PIX-confirmed
          // for VS_6e3e6f28). Previously we fell back to slot 31 blindly,
          // reading t31[0] from an unrelated buffer and corrupting cb3's
          // correct matrix on later frames.
          //
          // Semantic-based gate (no hardcoded hashes):
          //   - hasInstanceIdx: VS declares per-instance R16G16B16A16_UINT
          //     (COLOR1/I per the Source 2 convention) → real t31 indexing
          //   - rdefFound: shader self-declared g_modelInst → real t31
          // If neither, skip t31 path and let cb3/identity handle it.
          bool hasInstanceIdxSemantic = false;
          if (il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                hasInstanceIdxSemantic = true;
                break;
              }
            }
          }
          const bool t31PathEligible = rdefFound || hasInstanceIdxSemantic;
          // Also check if this VS has a cb3 CBufModelInstance — if so, the
          // downstream RDEF cb3 path will own the transform and we must NOT
          // let the "no bone transform → fallback" flag at line ~2806 fire.
          bool cb3OwnsTransform = false;
          {
            auto vsPtrCb3 = m_context->m_state.vs.shader;
            if (vsPtrCb3 != nullptr && vsPtrCb3->GetCommonShader() != nullptr) {
              auto cbInfo = vsPtrCb3->GetCommonShader()->FindCBuffer("CBufModelInstance");
              if (cbInfo && cbInfo->bindSlot != UINT32_MAX) cb3OwnsTransform = true;
            }
          }
          if (t31PathEligible && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
            modelInstSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
          } else if (!t31PathEligible) {
            // Skip the t31 fetch. If cb3 will provide the transform, pre-claim
            // gotBoneTransform so the end-of-block "no bone matrix" check
            // (line ~2806) doesn't set m_lastExtractUsedFallback=true and
            // cause SubmitDraw to filter this as UIFallback. The cb3 RDEF
            // path further down will write the real objectToWorld.
            if (cb3OwnsTransform) {
              gotBoneTransform = true;
            }
            static std::unordered_set<std::string> sT31SkipLogged;
            const std::string vkeyMiss = getVsHashShort();
            if (sT31SkipLogged.insert(vkeyMiss).second) {
              Logger::info(str::format(
                "[D3D11Rtx.o2w.t31.skip] vs=", vkeyMiss,
                " reason=no_rdef_g_modelInst_and_no_perinstance_uint4_idx",
                " cb3OwnsTransform=", cb3OwnsTransform ? 1 : 0,
                " (cb3 CBufModelInstance RDEF path owns this draw)"));
            }
          }
          // NV-DXVK: skinned-character discrimination. TF2 has TWO shader
          // families that both bind g_modelInst on t31:
          //
          //   1. BSP / batched props: POSITION0/V + COLOR1/I. t31 index is a
          //      per-INSTANCE COLOR1 value; one matrix per draw instance
          //      applies to the whole mesh. Our t31 fix is correct here.
          //   2. Skinned characters: POSITION0/V + BLENDWEIGHT0/V:fmt82 +
          //      BLENDINDICES0/V:fmt41. t31 is used as a BONE PALETTE — the
          //      VS reads t31[blendIdx[i]] PER-VERTEX and weighted-sums.
          //      Applying t31[0] to the whole character collapses every
          //      vertex onto bone 0 (pelvis/root), which is what caused the
          //      giant-face-stuck-on-camera visual.
          //
          // Detect category 2 by presence of BLENDINDICES per-vertex and skip
          // the t31 branch entirely — let the legacy t30 / skinning machinery
          // downstream handle these (as it does for classic characters).
          bool hasBlendIndices = false;
          if (il != nullptr) {
            for (const auto& sem : il->GetRtxSemantics()) {
              if (!sem.perInstance &&
                  std::strncmp(sem.name, "BLENDINDICES", 12) == 0 &&
                  sem.index == 0) {
                hasBlendIndices = true;
                break;
              }
            }
          }
          if (hasBlendIndices) {
            t31SkipReason = "has_blendindices_skinned_character";
            Logger::warn(str::format(
              "[D3D11Rtx.o2w.t31.skip] vs=", getVsHashShort(),
              " drawID=", m_drawCallID,
              " reason=", t31SkipReason,
              " (routing to legacy skinning path)"));
            // Fall through to legacy t30 bone path below.
            modelInstSrv = nullptr;
          }
          if (!modelInstSrv) {
            if (!t31SkipReason) t31SkipReason = "no_srv_bound_at_slot";
          } else {
            Com<ID3D11Resource> t31Res;
            modelInstSrv->GetResource(&t31Res);
            auto* t31Buf = static_cast<D3D11Buffer*>(t31Res.ptr());
            const uint8_t* t31Data = nullptr;
            size_t t31Len = 0;
            if (t31Buf) {
              auto t31Map = t31Buf->GetMappedSlice();
              if (t31Map.mapPtr && t31Map.length > 0) {
                t31Data = reinterpret_cast<const uint8_t*>(t31Map.mapPtr);
                t31Len  = t31Map.length;
              } else {
                void* p = t31Buf->GetBuffer()->mapPtr(0);
                if (p) {
                  t31Data = reinterpret_cast<const uint8_t*>(p);
                  t31Len  = t31Buf->GetBuffer()->info().size;
                }
              }
            }

            // Read charIdx from COLOR1 per-instance semantic (R16G16B16A16_UINT).
            // VS disasm uses v1.x which is the first uint16 of the 8-byte entry.
            uint32_t charIdx = 0;
            const char* charIdxReason = "no_perinstance_r16g16b16a16_uint_semantic";
            if (il != nullptr) {
              for (const auto& s : il->GetRtxSemantics()) {
                if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                  const auto& instVb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                  if (instVb.buffer == nullptr) {
                    charIdxReason = "instVb_buffer_null";
                  } else {
                    DxvkBufferSlice instSlice = instVb.buffer->GetBufferSlice(instVb.offset);
                    const uint8_t* instPtr =
                      instSlice.defined() ? reinterpret_cast<const uint8_t*>(instSlice.mapPtr(0)) : nullptr;
                    const size_t instOff =
                      static_cast<size_t>(m_currentInstanceIndex) * instVb.stride + s.byteOffset;
                    if (!instPtr) {
                      charIdxReason = "instPtr_null";
                    } else if (instSlice.length() < instOff + 2) {
                      charIdxReason = "instSlice_too_small";
                    } else {
                      charIdx = reinterpret_cast<const uint16_t*>(instPtr + instOff)[0];
                      charIdxReason = "ok";
                    }
                  }
                  break;
                }
              }
            }

            // Fetch t31[charIdx].objectToCameraRelative (float3x4 at entry+0).
            constexpr uint32_t BYTES_PER_INSTANCE = 208u;
            const size_t t31Off = static_cast<size_t>(charIdx) * BYTES_PER_INSTANCE;
            if (!t31Data) {
              t31SkipReason = "t31Data_null";
            } else if (t31Off + 48 > t31Len) {
              t31SkipReason = "t31Off_oob";
            } else {
              const float* m = reinterpret_cast<const float*>(t31Data + t31Off);
              bool finite = true;
              for (int k = 0; k < 12 && finite; ++k) if (!std::isfinite(m[k])) finite = false;
              const bool r0nz = m[0] != 0.f || m[1] != 0.f || m[2] != 0.f;
              const bool r1nz = m[4] != 0.f || m[5] != 0.f || m[6] != 0.f;
              const bool r2nz = m[8] != 0.f || m[9] != 0.f || m[10] != 0.f;
              if (!finite) {
                t31SkipReason = "non_finite_matrix";
              } else if (!(r0nz && r1nz && r2nz)) {
                t31SkipReason = "zero_row_in_matrix";
              } else {
                // +cameraOrigin to shift camera-relative → absolute world.
                float camOri[3] = { 0.f, 0.f, 0.f };
                bool haveCam = false;
                if (m_hasFanoutCamOrigin) {
                  camOri[0] = m_lastFanoutCamOrigin.x;
                  camOri[1] = m_lastFanoutCamOrigin.y;
                  camOri[2] = m_lastFanoutCamOrigin.z;
                  haveCam = true;
                } else {
                  const auto& cb2 = m_context->m_state.vs.constantBuffers[2];
                  if (cb2.buffer != nullptr) {
                    const auto cb2Map = cb2.buffer->GetMappedSlice();
                    const uint8_t* p2 = reinterpret_cast<const uint8_t*>(cb2Map.mapPtr);
                    const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
                    if (p2 && base + 16 <= cb2.buffer->Desc()->ByteWidth) {
                      const float* fp = reinterpret_cast<const float*>(p2 + base + 4);
                      if (std::isfinite(fp[0]) && std::isfinite(fp[1]) && std::isfinite(fp[2])) {
                        camOri[0] = fp[0]; camOri[1] = fp[1]; camOri[2] = fp[2];
                        haveCam = true;
                      }
                    }
                  }
                }
                const float tx = haveCam ? (m[3]  + camOri[0]) : m[3];
                const float ty = haveCam ? (m[7]  + camOri[1]) : m[7];
                const float tz = haveCam ? (m[11] + camOri[2]) : m[11];
                transforms.objectToWorld = Matrix4(
                  Vector4(m[0], m[4], m[8],  0.0f),
                  Vector4(m[1], m[5], m[9],  0.0f),
                  Vector4(m[2], m[6], m[10], 0.0f),
                  Vector4(tx,   ty,   tz,    1.0f));
                gotBoneTransform = true;
                m_lastO2wPathId = 1;

                // One-shot dump of the full t31 buffer contents the first time
                // each unique VS hits this path. Helps us see whether a shader
                // variant actually uses multiple entries or always idx 0.
                {
                  static std::unordered_set<std::string> sT31Dumped;
                  const std::string vkey = getVsHashShort();
                  if (sT31Dumped.insert(vkey).second) {
                    const uint32_t entries = std::min<uint32_t>(
                      static_cast<uint32_t>(t31Len / BYTES_PER_INSTANCE), 8u);
                    for (uint32_t e = 0; e < entries; ++e) {
                      const float* em = reinterpret_cast<const float*>(
                        t31Data + e * BYTES_PER_INSTANCE);
                      Logger::info(str::format(
                        "[D3D11Rtx.t31.dump] vs=", vkey, " entry=", e,
                        " T=(", em[3], ",", em[7], ",", em[11], ")",
                        " row0=(", em[0], ",", em[1], ",", em[2], ")",
                        " row1=(", em[4], ",", em[5], ",", em[6], ")",
                        " row2=(", em[8], ",", em[9], ",", em[10], ")"));
                    }
                  }
                }

                // Log every successful t31 draw (no cap) with full context +
                // VS hash so we can correlate which shader variants take this
                // path and disassemble representative ones.
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.t31.ok] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " rdef=", rdefFound ? 1 : 0,
                  " slot=", modelInstSlot,
                  " t31Len=", t31Len,
                  " charIdx=", charIdx,
                  " charIdxReason=", charIdxReason,
                  " haveCam=", haveCam ? 1 : 0,
                  " raw.T=(", m[3], ",", m[7], ",", m[11], ")",
                  " +cam=(", camOri[0], ",", camOri[1], ",", camOri[2], ")",
                  " final.T=(", tx, ",", ty, ",", tz, ")",
                  " row0=(", m[0], ",", m[1], ",", m[2], ")",
                  " row1=(", m[4], ",", m[5], ",", m[6], ")",
                  " row2=(", m[8], ",", m[9], ",", m[10], ")"));
              }
            }
            if (t31SkipReason) {
              Logger::warn(str::format(
                "[D3D11Rtx.o2w.t31.skip] vs=", getVsHashShort(),
                " drawID=", m_drawCallID,
                " rdef=", rdefFound ? 1 : 0,
                " slot=", modelInstSlot,
                " t31Len=", t31Len,
                " charIdx=", charIdx,
                " charIdxReason=", charIdxReason,
                " reason=", t31SkipReason));
            }
          }
          if (t31SkipReason && !strstr(t31SkipReason, "t31Data") && !modelInstSrv) {
            Logger::warn(str::format(
              "[D3D11Rtx.o2w.t31.nosrv] vs=", getVsHashShort(),
              " drawID=", m_drawCallID,
              " rdef=", rdefFound ? 1 : 0,
              " slot=", modelInstSlot,
              " reason=", t31SkipReason));
          }
        }

        // Legacy t30 bone path — only used when the t31 path above didn't
        // produce a transform (skinned characters / non-BSP draws).
        //
        // NV-DXVK principled gate: only enter this block when the VS actually
        // skins against t30. Signals (any of):
        //   - RDEF declares g_boneMatrix on the VS
        //   - VS has per-vertex BLENDINDICES semantic (skinned)
        // Without these, t30 being bound is coincidental app-state leftover;
        // reading bone[0] and using it as objectToWorld would displace static
        // meshes to whatever vestigial bone is at slot 0 (observed on
        // VS_6e3e6f28 — mesh translated to (-5223,835,32) instead of cb3's
        // correct (-5246,410,43)).
        bool t30PathEligible = false;
        {
          auto vsPtrBone = m_context->m_state.vs.shader;
          if (vsPtrBone != nullptr && vsPtrBone->GetCommonShader() != nullptr) {
            if (vsPtrBone->GetCommonShader()->FindResourceSlot("g_boneMatrix") != UINT32_MAX)
              t30PathEligible = true;
          }
          if (!t30PathEligible && il != nullptr) {
            for (const auto& s : il->GetRtxSemantics()) {
              if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0) {
                t30PathEligible = true;
                break;
              }
            }
          }
        }
        if (!t30PathEligible && !gotBoneTransform) {
          static std::unordered_set<std::string> sT30GateLogged;
          const std::string vkey = getVsHashShort();
          if (sT30GateLogged.insert(vkey).second) {
            Logger::info(str::format(
              "[D3D11Rtx.o2w.t30.skip] vs=", vkey,
              " reason=no_rdef_g_boneMatrix_and_no_blendindices",
              " (static mesh — cb3 CBufModelInstance path owns this draw)"));
          }
        }
        if (!gotBoneTransform && t30PathEligible) {
          const uint32_t kBoneSrvSlot = 30;
          ID3D11ShaderResourceView* boneSrv = nullptr;
          if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
            boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();
          if (boneSrv) {
            // Get bone matrix buffer
            Com<ID3D11Resource> boneRes;
            boneSrv->GetResource(&boneRes);
            auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
            DxvkBufferSlice boneBufSlice = boneBuf ? boneBuf->GetBufferSlice() : DxvkBufferSlice();
            const uint8_t* bonePtr = boneBufSlice.defined() ?
              reinterpret_cast<const uint8_t*>(boneBufSlice.mapPtr(0)) : nullptr;
            const size_t boneBufLen = boneBufSlice.defined() ? boneBufSlice.length() : 0;

            // Find the per-instance bone index from slot 1
            // (R16G16B16A16_UINT, perInstance=1, instance 0 for non-instanced draws)
            uint32_t boneIdx = 0;
            bool hasBoneIdx = false;
            for (const auto& s : il->GetRtxSemantics()) {
              if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
                const auto& instVb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
                if (instVb.buffer != nullptr) {
                  DxvkBufferSlice instSlice = instVb.buffer->GetBufferSlice(instVb.offset);
                  const uint8_t* instPtr = reinterpret_cast<const uint8_t*>(instSlice.mapPtr(0));
                  // Read COLOR1.y (second uint16) at the current instance index.
                  // The shader does: bone_index = BLENDINDICES(0) + COLOR1.y
                  // COLOR1 layout: [x=uint16, y=uint16, z=uint16, w=uint16]
                  // COLOR1.y = the second uint16 = byte offset +2 from semantic start
                  const size_t instOff = static_cast<size_t>(m_currentInstanceIndex) * instVb.stride + s.byteOffset;
                  if (instPtr && instSlice.length() >= instOff + 4) {
                    boneIdx = reinterpret_cast<const uint16_t*>(instPtr + instOff)[1]; // [1] = COLOR1.y
                    hasBoneIdx = true;
                  }
                }
                break;
              }
            }

            if (hasBoneIdx && bonePtr) {
              size_t boneOff = static_cast<size_t>(boneIdx) * 48;
              if (boneOff + 48 <= boneBufLen) {
                const float* m = reinterpret_cast<const float*>(bonePtr + boneOff);
                bool valid = true;
                for (int j = 0; j < 12; ++j) {
                  if (!std::isfinite(m[j])) { valid = false; break; }
                }
                if (valid) {
                  // Bone matrix is objectToWorld (float3x4, row-major).
                  transforms.objectToWorld = Matrix4(
                    Vector4(m[0], m[1], m[2],  0.0f),
                    Vector4(m[4], m[5], m[6],  0.0f),
                    Vector4(m[8], m[9], m[10], 0.0f),
                    Vector4(m[3], m[7], m[11], 1.0f));
                  gotBoneTransform = true;
                  m_lastO2wPathId = 2;
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.t30cpu] vs=", getVsHashShort(),
                    " drawID=", m_drawCallID,
                    " rawDraw=", m_rawDrawCount,
                    " inst=", m_currentInstanceIndex,
                    " boneIdx=", boneIdx,
                    " T=(", m[3], ",", m[7], ",", m[11], ")",
                    " row0=(", m[0], ",", m[1], ",", m[2], ")",
                    " row1=(", m[4], ",", m[5], ",", m[6], ")",
                    " row2=(", m[8], ",", m[9], ",", m[10], ")"));
                }
              }
            } else if (!bonePtr && boneSrv) {
              // Try multiple paths to read bone 0 from the D3D11Buffer:
              // 1. GetMappedSlice (WRITE_DISCARD mapped memory)
              // 2. DxvkBuffer direct mapPtr
              // 3. Cached from UpdateSubresource
              const float* bm = nullptr;
              // Path 1: D3D11Buffer mapped slice
              const auto mappedSlice = boneBuf->GetMappedSlice();
              if (mappedSlice.mapPtr && mappedSlice.length >= 48)
                bm = reinterpret_cast<const float*>(mappedSlice.mapPtr);
              // Path 2: DxvkBuffer direct map
              if (!bm) {
                void* p = boneBuf->GetBuffer()->mapPtr(0);
                if (p)
                  bm = reinterpret_cast<const float*>(p);
              }
              // Path 3: cached from UpdateSubresource
              if (!bm && m_hasCachedBone0)
                bm = m_cachedBone0;
              static uint32_t sBonePath = 0;
              if (sBonePath < 5 && bm) {
                ++sBonePath;
                Logger::info(str::format(
                  "[D3D11Rtx] Bone read: path=",
                  (bm == reinterpret_cast<const float*>(mappedSlice.mapPtr)) ? "mapped" :
                  (bm == m_cachedBone0) ? "cached" : "dxvkBuf",
                  " T=(", bm[3], ",", bm[7], ",", bm[11], ")",
                  " mapMode=", uint32_t(boneBuf->GetMapMode())));
              }
              if (bm) {
                bool valid = true;
                for (int j = 0; j < 12; ++j)
                  if (!std::isfinite(bm[j])) { valid = false; break; }
                if (valid) {
                  transforms.objectToWorld = Matrix4(
                    Vector4(bm[0], bm[1], bm[2],  0.0f),
                    Vector4(bm[4], bm[5], bm[6],  0.0f),
                    Vector4(bm[8], bm[9], bm[10], 0.0f),
                    Vector4(bm[3], bm[7], bm[11], 1.0f));
                  gotBoneTransform = true;
                  m_lastO2wPathId = 3;
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.t30slice] vs=", getVsHashShort(),
                    " drawID=", m_drawCallID,
                    " rawDraw=", m_rawDrawCount,
                    " T=(", bm[3], ",", bm[7], ",", bm[11], ")",
                    " row0=(", bm[0], ",", bm[1], ",", bm[2], ")"));
                  static uint32_t sBoneDiag2 = 0;
                  if (sBoneDiag2 < 10) {
                    ++sBoneDiag2;
                    Logger::info(str::format(
                      "[D3D11Rtx] Bone from MappedSlice: T=(",
                      bm[3], ",", bm[7], ",", bm[11], ")",
                      " mapPtr=", mappedSlice.mapPtr != nullptr ? 1 : 0));
                  }
                }
              }

              // Log per-draw bone info: buffer address + offset to see if it changes
              static uint32_t sGpuBoneDiag = 0;
              if (sGpuBoneDiag < 30) {
                ++sGpuBoneDiag;
                // Check the instance buffer (slot 1) pointer and offset per draw
                uintptr_t instBufAddr = 0;
                uint32_t instBufOff = 0;
                for (const auto& s3 : il->GetRtxSemantics()) {
                  if (s3.perInstance && s3.format == VK_FORMAT_R16G16B16A16_UINT) {
                    const auto& ivb3 = m_context->m_state.ia.vertexBuffers[s3.inputSlot];
                    if (ivb3.buffer != nullptr) {
                      instBufAddr = reinterpret_cast<uintptr_t>(ivb3.buffer.ptr());
                      instBufOff = ivb3.offset;
                    }
                    break;
                  }
                }
                // Also check if t30 buffer changes between draws
                uintptr_t boneBufAddr = reinterpret_cast<uintptr_t>(boneBuf);
                Logger::info(str::format(
                  "[D3D11Rtx] BoneDraw raw=", m_rawDrawCount,
                  " t30buf=", boneBufAddr,
                  " instBuf=", instBufAddr, "+", instBufOff,
                  " t30len=", boneBufLen));
              }
            }
          }
        }
        if (!gotBoneTransform) {
          // No bone matrix available — can't position this geometry
          m_lastExtractUsedFallback = true;
        }
        // NOTE: do NOT set m_foundRealProjThisFrame here — that would let
        // fmt=106 shadow draws bypass the uiFallback check in SubmitDraw.
      } else {
        // Non-R32G32_UINT draw without camera — mark as fallback so it
        // gets filtered as UI in SubmitDraw (shadow/depth pass).
        m_lastExtractUsedFallback = true;
      }
    }

    if (projSlot == UINT32_MAX && !m_hasEverFoundProj) {
      m_lastExtractUsedFallback = true;
      const auto& vp = m_context->m_state.rs.viewports[0];
      if (vp.Width > 0.0f && vp.Height > 0.0f) {
        const float aspect = vp.Width / vp.Height;
        const float fovY   = 60.0f * (3.14159265f / 180.0f);
        const float nearZ  = 0.1f;
        const float farZ   = 10000.0f;
        const float yScale = 1.0f / std::tan(fovY * 0.5f);
        const float xScale = yScale / aspect;
        const float Q      = farZ / (farZ - nearZ);
        transforms.viewToProjection = Matrix4(
          Vector4(xScale, 0.0f,   0.0f,         0.0f),
          Vector4(0.0f,   yScale, 0.0f,         0.0f),
          Vector4(0.0f,   0.0f,   Q,            1.0f),
          Vector4(0.0f,   0.0f,  -nearZ * Q,    0.0f));
        static bool s_fallbackLogged = false;
        if (!s_fallbackLogged) {
          s_fallbackLogged = true;
          Logger::info(str::format(
            "[D3D11Rtx] No projection found in cbuffers — using viewport fallback (",
            vp.Width, "x", vp.Height, " aspect=", aspect, ")"));
        }
      }
    }

    // --- VIEW MATRIX ---
    // NV-DXVK: Skip if worldToView was already set (cross-frame VP for R32G32_UINT)
    if (m_skipViewMatrixScan) goto skipViewScan;
    // Cached fast path: re-read from previously discovered location.
    // Only rescan when the cached location is invalid or doesn't contain
    // a view matrix anymore (shader change, different render pass).
    bool viewCacheHit = false;
    if (m_viewSlot != UINT32_MAX && m_viewStage >= 0 && m_viewStage < kNumStages) {
      const auto& cb = (*stageCbs[m_viewStage])[m_viewSlot];
      if (cb.buffer != nullptr) {
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr) {
          Matrix4 c = readMatrix(ptr, m_viewOffset, cb.buffer->Desc()->ByteWidth);
          if (isViewMatrix(c)) {
            m_lastWtvPathId = 5; // cached view-matrix slot
            // NV-DXVK: readCbMatrix stores rows-as-columns (passes raw[i][j]
            // to Matrix4 ctor with rows as args, which dxvk treats as cols).
            // The mathematical matrix in memory ends up stored as M^T in our
            // Matrix4. Transpose to recover the intended M, matching the
            // convention path 1/3 use after the column-storage fix.
            transforms.worldToView = transpose(c);
            viewCacheHit = true;
            // Diagnostic log: confirm path-5 latches now produce same Main.pos
            // as path 1/3. Cap to 30 to avoid spam.
            static uint32_t sPath5Log = 0;
            if (sPath5Log < 30) {
              ++sPath5Log;
              const auto& w = transforms.worldToView;
              Logger::info(str::format(
                "[D3D11Rtx.path5Cam] #", sPath5Log,
                " cam=(", w[3][0], ",", w[3][1], ",", w[3][2],
                ")  (raw t-col)"));
            }
          }
        }
      }
    }

    // Full scan fallback — same logic as before, but caches the result.
    if (!viewCacheHit && projSlot != UINT32_MAX) {
      if (projStage >= 0 && projStage < kNumStages) {
        const auto& cb = (*stageCbs[projStage])[projSlot];
        if (cb.buffer != nullptr) {
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (ptr) {
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            if (projOffset >= 64) {
              Matrix4 c = readMatrix(ptr, projOffset - 64, bufSize);
              if (isViewMatrix(c)) {
                m_lastWtvPathId = 6; // scan near projection (offset-64)
                transforms.worldToView = transpose(c); // see path 5 fix comment
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = projOffset - 64;
                static uint32_t sPath6Log = 0;
                if (sPath6Log < 30) {
                  ++sPath6Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path6Cam] #", sPath6Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
              }
            }
            if (isIdentityExact(transforms.worldToView)) {
              auto [vBase, vEnd] = cbRange(cb);
              for (size_t off = vBase; off + 64 <= vEnd; off += 16) {
                if (off >= projOffset && off < projOffset + 64) continue;
                Matrix4 c = readMatrix(ptr, off, bufSize);
                if (isViewMatrix(c)) {
                  m_lastWtvPathId = 7; // scan same-cb as projection
                  transforms.worldToView = transpose(c); // see path 5 fix comment
                  m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
                  static uint32_t sPath7Log = 0;
                  if (sPath7Log < 30) {
                    ++sPath7Log;
                    const auto& w = transforms.worldToView;
                    Logger::info(str::format(
                      "[D3D11Rtx.path7Cam] #", sPath7Log,
                      " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                  }
                  break;
                }
              }
            }
          }
        }
      }

      // Cross-stage fallback: scan all stages' cbuffers for a view matrix.
      if (isIdentityExact(transforms.worldToView)) {
        for (int si = 0; si < kNumStages && isIdentityExact(transforms.worldToView); ++si) {
          const auto& cbs = *stageCbs[si];
          for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
            if (si == projStage && slot == projSlot) continue;
            const auto& cb = cbs[slot];
            if (cb.buffer == nullptr) continue;
            const auto mapped = cb.buffer->GetMappedSlice();
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            if (!ptr) continue;
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            auto [csBase, csEnd] = cbRange(cb);
            for (size_t off = csBase; off + 64 <= csEnd; off += 16) {
              Matrix4 c = readMatrix(ptr, off, bufSize);
              if (isViewMatrix(c)) {
                m_lastWtvPathId = 8; // cross-stage all-cb scan
                transforms.worldToView = transpose(c); // see path 5 fix comment
                m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
                static uint32_t sPath8Log = 0;
                if (sPath8Log < 30) {
                  ++sPath8Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path8Cam] #", sPath8Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
                break;
              }
            }
            if (!isIdentityExact(transforms.worldToView)) break;
          }
        }
      }

      // Convention fallback: if no view matrix was found, the column-major
      // detection may be wrong (ambiguous when near plane ≈ 1). Retry with
      // the opposite convention, but only for the projection cbuffer.
      if (isIdentityExact(transforms.worldToView) && projStage >= 0 && projStage < kNumStages) {
        const auto& cb = (*stageCbs[projStage])[projSlot];
        if (cb.buffer != nullptr) {
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (ptr) {
            const size_t bufSize = cb.buffer->Desc()->ByteWidth;
            auto [fbBase, fbEnd] = cbRange(cb);
            for (size_t off = fbBase; off + 64 <= fbEnd; off += 16) {
              if (off >= projOffset && off < projOffset + 64) continue;
              Matrix4 raw = readCbMatrix(ptr, off, bufSize);
              Matrix4 flipped = m_columnMajor ? raw : transpose(raw);
              if (isViewMatrix(flipped)) {
                m_lastWtvPathId = 9; // convention-flip fallback
                transforms.worldToView = transpose(flipped); // see path 5 fix comment
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
                m_columnMajor = !m_columnMajor;
                static uint32_t sPath9Log = 0;
                if (sPath9Log < 30) {
                  ++sPath9Log;
                  const auto& w = transforms.worldToView;
                  Logger::info(str::format(
                    "[D3D11Rtx.path9Cam] #", sPath9Log,
                    " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
                }
                break;
              }
            }
          }
        }
      }
    }

    // When using fallback projection (projSlot == UINT32_MAX), still search
    // all stages for a view matrix so the camera position is correct.
    if (!viewCacheHit && projSlot == UINT32_MAX && isIdentityExact(transforms.worldToView)) {
      for (int si = 0; si < kNumStages && isIdentityExact(transforms.worldToView); ++si) {
        const auto& cbs = *stageCbs[si];
        for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
          const auto& cb = cbs[slot];
          if (cb.buffer == nullptr) continue;
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (!ptr) continue;
          const size_t bufSize = cb.buffer->Desc()->ByteWidth;
          auto [csBase, csEnd] = cbRange(cb);
          for (size_t off = csBase; off + 64 <= csEnd; off += 16) {
            Matrix4 c = readMatrix(ptr, off, bufSize);
            if (isViewMatrix(c)) {
              m_lastWtvPathId = 10; // fallback-projection branch cross-stage scan
              transforms.worldToView = transpose(c); // see path 5 fix comment
              m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
              static uint32_t sPath10Log = 0;
              if (sPath10Log < 30) {
                ++sPath10Log;
                const auto& w = transforms.worldToView;
                Logger::info(str::format(
                  "[D3D11Rtx.path10Cam] #", sPath10Log,
                  " cam=(", w[3][0], ",", w[3][1], ",", w[3][2], ")"));
              }
              break;
            }
          }
          if (!isIdentityExact(transforms.worldToView)) break;
        }
      }
    }

    skipViewScan:
    // --- Z-UP / Y-UP AUTO-DETECTION (view-matrix-derived) ---
    // In a Y-up world, the view matrix "up" column (col 1) has its largest
    // component in row 1 (Y). In a Z-up world, column 1's largest component
    // is in row 2 (Z). Vote on each valid view matrix and settle via threshold.
    if (!isIdentityExact(transforms.worldToView)) {
      if (!m_zUpSettled) {
        const float absY = std::abs(transforms.worldToView[1][1]);
        const float absZ = std::abs(transforms.worldToView[2][1]);
        // Only vote when there's a clear winner (avoid ambiguous 45° views)
        if (std::abs(absZ - absY) > 0.3f) {
          m_zUpVotes += (absZ > absY) ? 1 : -1;
          if (!m_zUpSettled && std::abs(m_zUpVotes) >= kVoteThreshold) {
            m_zUpSettled = true;
            const bool zUp = m_zUpVotes > 0;
            RtxOptions::zUpObject().setDeferred(zUp);
          }
        }
      }

      // Log settled axis conventions once.
      if (m_zUpSettled && m_yFlipSettled && m_lhSettled && !m_axisLogged) {
        m_axisLogged = true;
        Logger::info(str::format("[D3D11Rtx] Axis detection settled: ",
          m_lhVotes > 0 ? "LH" : "RH",
          m_yFlipVotes > 0 ? " Y-flipped" : "",
          m_zUpVotes > 0 ? " Z-up" : " Y-up",
          m_columnMajor ? " col-major" : " row-major",
          " (proj stage=", kStageNames[std::max(0, m_projStage)],
          " slot=", m_projSlot, " off=", m_projOffset, ")"));
      }
    }

    // --- CAMERA POSITION SMOOTHING ---
    // The view matrix encodes camera position in its translation row (row 3).
    // Floating-point rounding in cbuffer reads causes sub-pixel jitter between
    // draws/frames. Apply exponential moving average on the position to dampen
    // this without introducing visible lag. The rotation (upper 3x3) is left
    // untouched — rotation jitter is rare and smoothing it causes ghosting.
    //
    // NV-DXVK: Smoothing is ONLY applied to VP-decomposition paths (1, 2, 3)
    // where float rounding in the row-magnitude normalization + basis
    // re-derivation actually produces jitter. For the cached-slot scan paths
    // (5-10) the translation column is read verbatim from a real view matrix
    // cbuffer — no jitter — and smoothing just introduces lag. m_lastWtvPathId
    // lets us gate cleanly. Paths 0 and 11 are also excluded (bone-composite).
    //
    // D3D row-major view matrix layout:
    //   [R00 R01 R02  0]    pos = -R^T * t
    //   [R10 R11 R12  0]    where t = (V[3][0], V[3][1], V[3][2])
    //   [R20 R21 R22  0]
    //   [tx  ty  tz   1]
    const bool smoothingApplies =
      m_lastWtvPathId == 1 || m_lastWtvPathId == 2 || m_lastWtvPathId == 3;
    if (smoothingApplies && !isIdentityExact(transforms.worldToView) && !m_skipViewMatrixScan) {
      const auto& V = transforms.worldToView;
      // Camera world position: pos = -R^T * t for view matrix V = [R | 0; t | 1]
      Vector3 t(V[3][0], V[3][1], V[3][2]);
      Vector3 camPos(
        -(V[0][0] * t.x + V[1][0] * t.y + V[2][0] * t.z),
        -(V[0][1] * t.x + V[1][1] * t.y + V[2][1] * t.z),
        -(V[0][2] * t.x + V[1][2] * t.y + V[2][2] * t.z));

      constexpr float kSmoothAlpha = 0.8f; // 0 = full smooth (laggy), 1 = no smooth (jittery)
      constexpr float kTeleportThreshold = 5.0f; // snap on large jumps (cutscene, teleport)

      if (m_hasPrevCamPos) {
        Vector3 delta = camPos - m_smoothedCamPos;
        float distSq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (distSq < kTeleportThreshold * kTeleportThreshold) {
          m_smoothedCamPos = Vector3(
            m_smoothedCamPos.x + kSmoothAlpha * (camPos.x - m_smoothedCamPos.x),
            m_smoothedCamPos.y + kSmoothAlpha * (camPos.y - m_smoothedCamPos.y),
            m_smoothedCamPos.z + kSmoothAlpha * (camPos.z - m_smoothedCamPos.z));
        } else {
          m_smoothedCamPos = camPos;
        }
      } else {
        m_smoothedCamPos = camPos;
        m_hasPrevCamPos = true;
      }

      // Reconstruct translation row from smoothed position: t = -R * smoothPos
      transforms.worldToView[3][0] = -(V[0][0] * m_smoothedCamPos.x + V[0][1] * m_smoothedCamPos.y + V[0][2] * m_smoothedCamPos.z);
      transforms.worldToView[3][1] = -(V[1][0] * m_smoothedCamPos.x + V[1][1] * m_smoothedCamPos.y + V[1][2] * m_smoothedCamPos.z);
      transforms.worldToView[3][2] = -(V[2][0] * m_smoothedCamPos.x + V[2][1] * m_smoothedCamPos.y + V[2][2] * m_smoothedCamPos.z);
    }

    // --- WORLD MATRIX ---
    // NV-DXVK: For R32G32_UINT draws, read cb3 directly from its mapped memory.
    // cb3 is updated via Map/WRITE_DISCARD (not UpdateSubresource).
    // GetMappedSlice() returns the CPU-mapped pointer with current data.
    if (m_skipViewMatrixScan) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const auto& cb3 = vsCbs[3];
      const float* bm = nullptr;
      if (cb3.buffer != nullptr) {
        const auto mapped = cb3.buffer->GetMappedSlice();
        if (mapped.mapPtr && mapped.length >= static_cast<size_t>(cb3.constantOffset) * 16 + 48) {
          bm = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(mapped.mapPtr) + static_cast<size_t>(cb3.constantOffset) * 16);
        }
        static uint32_t sCb3Diag = 0;
        if (sCb3Diag < 10) {
          ++sCb3Diag;
          Logger::info(str::format(
            "[D3D11Rtx] CB3 read: mapPtr=", mapped.mapPtr != nullptr ? 1 : 0,
            " mapMode=", uint32_t(cb3.buffer->GetMapMode()),
            " usage=", uint32_t(cb3.buffer->Desc()->Usage),
            " size=", cb3.buffer->Desc()->ByteWidth,
            " off=", cb3.constantOffset,
            " bm=", bm != nullptr ? 1 : 0));
        }
      }
      // NV-DXVK: only run CB3→O2W if no upstream path (t31 at line 2342,
      // or legacy t30 bone paths at 2551/2604) already set objectToWorld.
      // Without this gate, CB3→O2W clobbers the t31-derived o2w with a
      // stale cb3 read for every R32G32_UINT draw — the histogram showed
      // cb3=32 commits per frame and t31=0 despite t31 firing successfully
      // thousands of times.
      if (bm && m_lastO2wPathId == 0) {
        Matrix4 cb3Mat(
          Vector4(bm[0], bm[1], bm[2],  0.0f),
          Vector4(bm[4], bm[5], bm[6],  0.0f),
          Vector4(bm[8], bm[9], bm[10], 0.0f),
          Vector4(bm[3], bm[7], bm[11], 1.0f));
        Matrix4 invView = inverse(transforms.worldToView);
        transforms.objectToWorld = invView * cb3Mat;
        m_lastO2wPathId = 4;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.cb3] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " cb3.T=(", cb3Mat[3][0], ",", cb3Mat[3][1], ",", cb3Mat[3][2], ")",
          " o2w.T=(", transforms.objectToWorld[3][0], ",",
          transforms.objectToWorld[3][1], ",",
          transforms.objectToWorld[3][2], ")"));
      }
    }

    // Scan VS cbuffers first (model matrices live in VS for virtually all engines),
    // then fall back to other stages for emulator compatibility.
    // Gated by useCBufferWorldMatrices — disable if CB layout causes wrong detections.
    // NV-DXVK: Skip for R32G32_UINT draws — cached cb3 is already set above.
    if (RtxOptions::useCBufferWorldMatrices() && !m_currentDrawIsBoneTransformed && !m_skipViewMatrixScan) {
      // --- Source-engine float3x4 world matrix (translation in column 3) ---
      // IDA analysis of materialsystem_dx11.dll confirms:
      //   VS slot 0 = per-draw texture/viewport constants (set by materialsystem)
      //   VS slot 1 = per-draw material/skinning constants (set by materialsystem)
      //   VS slots 2+ = set by engine/game code (not materialsystem)
      //   VS slot 2 = combined VP matrix at offset 96 (camera)
      //   VS slot 3 = [objectToWorld float3x4 | worldToView float3x4] (96 bytes total)
      //
      // Source/Titanfall 2 stores objectToWorld as a float3x4 at VS slot=3 offset=0.
      // Format:
      //   Row 0: [R00 R01 R02 Tx]   ← 16 bytes, translation in COLUMN 3
      //   Row 1: [R10 R11 R12 Ty]   ← 16 bytes
      //   Row 2: [R20 R21 R22 Tz]   ← 16 bytes
      //   (offset +48: second float3x4, the worldToView, NOT another object matrix)
      //
      // This is checked FIRST before the generic 4x4 scanner to prevent the
      // scanner from picking up false positives from materialsystem's slot 0/1 data.
      auto trySourceFloat3x4 = [&](const D3D11ConstantBufferBindings& cbs,
                                    uint32_t slot, uint32_t byteOffset = 0) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t cbBase  = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t base    = cbBase + byteOffset;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        // Need at least 48 bytes (3 rows × 16 bytes).
        if (base + 48 > bufSize) return false;
        const float* f = reinterpret_cast<const float*>(ptr + base);
        // Read the 3×4 matrix.
        const float R00 = f[0],  R01 = f[1],  R02 = f[2],  Tx = f[3];
        const float R10 = f[4],  R11 = f[5],  R12 = f[6],  Ty = f[7];
        const float R20 = f[8],  R21 = f[9],  R22 = f[10], Tz = f[11];
        // Sanity: all 12 entries must be finite.
        if (!std::isfinite(R00) || !std::isfinite(R01) || !std::isfinite(R02) ||
            !std::isfinite(R10) || !std::isfinite(R11) || !std::isfinite(R12) ||
            !std::isfinite(R20) || !std::isfinite(R21) || !std::isfinite(R22) ||
            !std::isfinite(Tx)  || !std::isfinite(Ty)  || !std::isfinite(Tz))
          return false;
        // Sanity: each column of the 3×3 rotation block must have unit length
        // (approximately orthonormal).  Degenerate zero-columns are rejected.
        const float col0Sq = R00*R00 + R10*R10 + R20*R20;
        const float col1Sq = R01*R01 + R11*R11 + R21*R21;
        const float col2Sq = R02*R02 + R12*R12 + R22*R22;
        if (col0Sq < 0.25f || col0Sq > 4.0f) return false;
        if (col1Sq < 0.25f || col1Sq > 4.0f) return false;
        if (col2Sq < 0.25f || col2Sq > 4.0f) return false;
        // Reject if it looks like a perspective matrix (would have large off-diagonal
        // entries and near-zero diagonal in the third row).
        if (classifyPerspective(Matrix4(
              Vector4(R00, R01, R02, 0.0f),
              Vector4(R10, R11, R12, 0.0f),
              Vector4(R20, R21, R22, 0.0f),
              Vector4(Tx,  Ty,  Tz,  1.0f))) != 0)
          return false;
        // NV-DXVK (Titanfall 2): reject identity. TF2 packs two float3x4 blocks
        // back-to-back in VS s3 where block 0 (offset 0) is identity and block 1
        // (offset 48) is the real objectToWorld. Accepting identity here made the
        // caller stop searching and every static world draw landed at origin.
        if (std::abs(R00 - 1.0f) < 1e-6f && std::abs(R11 - 1.0f) < 1e-6f && std::abs(R22 - 1.0f) < 1e-6f
            && std::abs(R01) < 1e-6f && std::abs(R02) < 1e-6f
            && std::abs(R10) < 1e-6f && std::abs(R12) < 1e-6f
            && std::abs(R20) < 1e-6f && std::abs(R21) < 1e-6f
            && std::abs(Tx) < 1e-6f && std::abs(Ty) < 1e-6f && std::abs(Tz) < 1e-6f)
          return false;
        // Row-major Matrix4 from row-major float3x4.
        transforms.objectToWorld = Matrix4(
          Vector4(R00, R01, R02, 0.0f),
          Vector4(R10, R11, R12, 0.0f),
          Vector4(R20, R21, R22, 0.0f),
          Vector4(Tx,  Ty,  Tz,  1.0f));
        m_lastO2wPathId = 6;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.sf3x4] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " slot=", slot, " off=", byteOffset,
          " T=(", Tx, ",", Ty, ",", Tz, ")"));
        return true;
      };

      // --- Standard 4x4 world matrix scan (D3D row-major convention) ---
      // Used as fallback for non-Source engines where the model matrix is stored
      // in the standard D3D convention (translation in row 3, column 3 = zero).
      // SKIPS VS slots 0-1 which are owned by materialsystem (per-draw constants
      // that contain viewport/texture data, not world transforms).
      auto tryWorldCb = [&](const D3D11ConstantBufferBindings& cbs, uint32_t slot,
                            int skipStage, uint32_t skipSlot) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        if (base + 64 > bufSize) return false;
        Matrix4 candidate = readMatrix(ptr, base, bufSize);
        if (isIdentityExact(candidate) || classifyPerspective(candidate) != 0 || isViewMatrix(candidate))
          return false;
        if (std::abs(candidate[3][3] - 1.0f) > 0.01f) return false;
        if (std::abs(candidate[0][3]) > 0.01f || std::abs(candidate[1][3]) > 0.01f || std::abs(candidate[2][3]) > 0.01f)
          return false;
        transforms.objectToWorld = candidate;
        m_lastO2wPathId = 7;
        Logger::info(str::format(
          "[D3D11Rtx.o2w.worldcb] vs=", getVsHashShort(),
          " drawID=", m_drawCallID,
          " slot=", slot,
          " T=(", candidate[3][0], ",", candidate[3][1], ",", candidate[3][2], ")"));
        return true;
      };

      // NV-DXVK: sync `found` with the path-id system so the world-matrix
      // scan below (RDEF, trySourceFloat3x4, tryWorldCb) doesn't overwrite
      // an o2w already set by an upstream path (t31=1, t30cpu=2, t30slice=3).
      bool found = (m_lastO2wPathId != 0);
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;

      // NV-DXVK: for shaders with a per-vertex BLENDINDICES semantic (skinned
      // characters), the world transform does NOT live in any cbuffer — it's
      // exclusively in the t30/t31 SRV bone/model palette. Scanning cbuffers
      // for character shaders produces garbage (typically picks up a cb3
      // region that happens to contain -cameraOrigin as a 3x4 translation,
      // which then stamps o2w=-cam and plasters the character's BLAS over
      // the camera origin). Short-circuit `found = true` when BLENDINDICES
      // is present so the entire scan block below is bypassed. If t30/t31
      // couldn't produce an o2w upstream, the draw stays at identity (and
      // gets filtered as UI-fallback downstream) which is strictly better
      // than a wrong non-identity matrix.
      if (!found) {
        auto* ilGate = m_context->m_state.ia.inputLayout.ptr();
        if (ilGate) {
          for (const auto& s : ilGate->GetRtxSemantics()) {
            if (!s.perInstance &&
                std::strncmp(s.name, "BLENDINDICES", 12) == 0 &&
                s.index == 0) {
              found = true;  // poison pill: skip the cbuffer scans
              static uint32_t sBiPoisonLog = 0;
              if (sBiPoisonLog < 20) {
                ++sBiPoisonLog;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.scan.skip] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " reason=has_blendindices_no_cbuffer_scan"));
              }
              break;
            }
          }
        }
      }

      // NV-DXVK: DETERMINISTIC EXTRACTION via DXBC RDEF reflection.
      // The VS itself declares the cbuffers it binds (names + slots) and their
      // field offsets. We look up by HLSL cbuffer/field name — no size or content
      // heuristics. Only guessing is replaced; legacy path retained below as a
      // fallback for shaders that stripped RDEF.
      const D3D11CommonShader* commonVS = nullptr;
      auto vsPtr = m_context->m_state.vs.shader;
      if (vsPtr != nullptr) commonVS = vsPtr->GetCommonShader();

      auto rdefReadFloats = [&](const D3D11ConstantBufferBindings& cbs,
                                uint32_t slot, uint32_t fieldOff,
                                uint32_t fieldSize, float* out) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = cbs[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + fieldOff;
        if (base + fieldSize > cb.buffer->Desc()->ByteWidth) return false;
        std::memcpy(out, ptr + base, fieldSize);
        return true;
      };

      if (commonVS != nullptr) {
        // Titanfall 2 Source Engine 2 transform chain (verified via RDEF).
        // Some shader variants expose a `CBufModelInstance` cbuffer with
        // `objectToCameraRelative` at offset 0. Most BSP shaders instead
        // use the g_modelInst SRV (t31); those are handled upstream in the
        // fanout path and the non-instanced t31 read at ~line 2342.
        auto modelCb = commonVS->FindCBuffer("CBufModelInstance");

        // NV-DXVK DIAG: PIX confirmed VS 0x298e12b3d5bcd082 (merged[Opaque][0]
        // warped-mesh VS, SHA1 6e3e6f28...) binds a valid 3x4 row-major
        // objectToCameraRelative at cb slot 3 with uniform scale 0.3. If
        // FindCBuffer("CBufModelInstance") misses for this VS, the merged
        // variant uses a different HLSL cbuffer name. Dump all declared
        // cbuffers + raw cb3 bytes so we can identify the correct name.
        {
          // One-shot per VS-match; safe to leave ungated.
          const std::string vsKeyDiag = getVsHashShort();
          const bool isWarpedMeshVs = vsKeyDiag.find("VS_6e3e6f28f2156ea2") != std::string::npos;
          if (isWarpedMeshVs) {
            static bool sLoggedOnce = false;
            if (!sLoggedOnce) {
              sLoggedOnce = true;
              auto names = commonVS->GetCBufferNamesAndSlots();
              std::string cbList;
              for (const auto& p : names) {
                cbList += p.first + "@slot" + std::to_string(p.second) + " ";
              }
              Logger::info(str::format(
                "[D3D11Rtx.o2w.warpedMesh.rdefDump] vs=", vsKeyDiag,
                " cbufferCount=", names.size(),
                " cbuffers={ ", cbList, "}",
                " modelCbFound=", (modelCb != nullptr) ? 1 : 0,
                " modelCbSlot=", modelCb ? modelCb->bindSlot : UINT32_MAX));

              // Dump raw cb3 bytes as 12 floats (the objectToCameraRelative we
              // know PIX has for this draw). Bypasses RDEF to confirm the raw
              // binding has the matrix.
              const uint32_t kRawSlot = 3;
              if (kRawSlot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                const auto& cb = vsCbs[kRawSlot];
                if (cb.buffer != nullptr) {
                  const auto mapped = cb.buffer->GetMappedSlice();
                  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
                  if (ptr) {
                    const size_t base = static_cast<size_t>(cb.constantOffset) * 16;
                    if (base + 48 <= cb.buffer->Desc()->ByteWidth) {
                      float raw[12];
                      std::memcpy(raw, ptr + base, 48);
                      Logger::info(str::format(
                        "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                        " bufSize=", cb.buffer->Desc()->ByteWidth,
                        " constOff=", cb.constantOffset,
                        " r0=(", raw[0], ",", raw[1], ",", raw[2], ",", raw[3], ")",
                        " r1=(", raw[4], ",", raw[5], ",", raw[6], ",", raw[7], ")",
                        " r2=(", raw[8], ",", raw[9], ",", raw[10], ",", raw[11], ")"));
                    } else {
                      Logger::info(str::format(
                        "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                        " base+48 exceeds bufSize=", cb.buffer->Desc()->ByteWidth,
                        " constOff=", cb.constantOffset));
                    }
                  } else {
                    Logger::info(str::format(
                      "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                      " slot3 buffer has no mapPtr"));
                  }
                } else {
                  Logger::info(str::format(
                    "[D3D11Rtx.o2w.warpedMesh.rawCb3] vs=", vsKeyDiag,
                    " slot3 has NO buffer bound"));
                }
              }
            }
          }
        }

        if (modelCb && modelCb->bindSlot != UINT32_MAX) {
          float m[12];
          if (rdefReadFloats(vsCbs, modelCb->bindSlot, 0, 48, m)) {
            bool ok = true;
            for (int k = 0; k < 12 && ok; ++k)
              if (!std::isfinite(m[k])) ok = false;
            if (ok) {
              // Also fetch c_cameraOrigin from CBufCommonPerCamera (offset 4, 3 floats).
              float camO[3] = { 0.f, 0.f, 0.f };
              bool haveCamO = false;
              if (auto camLoc = commonVS->FindCBField("CBufCommonPerCamera", "c_cameraOrigin")) {
                if (camLoc->size >= 12 && camLoc->slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) {
                  if (rdefReadFloats(vsCbs, camLoc->slot, camLoc->offset, 12, camO)) {
                    if (std::isfinite(camO[0]) && std::isfinite(camO[1]) && std::isfinite(camO[2])) {
                      haveCamO = true;
                    }
                  }
                }
              }

              // Detect "cb3 is identity-with-zero-translation": the BSP world render pass
              // uses cb3 = { I | 0 } because its vertex buffer is already in cam-relative
              // world space — the VS matmul passes vertices through unchanged to cam-relative.
              // For Remix RT we need objectToWorld = translate(+cameraOrigin) so those
              // cam-relative vertices land at absolute world in the BLAS.
              constexpr float kEps = 1e-4f;
              auto near_ = [&](float a, float b) { return std::abs(a - b) < kEps; };
              const bool cb3IsIdentityZeroT =
                   near_(m[0], 1.f) && near_(m[1], 0.f) && near_(m[2],  0.f) && near_(m[3],  0.f)
                && near_(m[4], 0.f) && near_(m[5], 1.f) && near_(m[6],  0.f) && near_(m[7],  0.f)
                && near_(m[8], 0.f) && near_(m[9], 0.f) && near_(m[10], 1.f) && near_(m[11], 0.f);
              const bool cb3IsAllZero =
                   m[0]==0.f && m[1]==0.f && m[2]==0.f && m[3]==0.f
                && m[4]==0.f && m[5]==0.f && m[6]==0.f && m[7]==0.f
                && m[8]==0.f && m[9]==0.f && m[10]==0.f && m[11]==0.f;
              const bool cb3IsZero = cb3IsIdentityZeroT || cb3IsAllZero;

              // c_cameraOrigin may be zero for this specific draw's cb2 binding
              // (BSP world pass). For the zeroCb3 path only, fall back to the
              // fanout-captured authoritative gameplay camera origin.
              // IMPORTANT: do NOT apply this fallback to the non-zero-cb3 path,
              // since those draws have their own real translation and adding
              // camOrigin on top would double-shift and displace them.
              float camOforZeroCb3[3] = { camO[0], camO[1], camO[2] };
              bool haveCamOforZeroCb3 = haveCamO;
              const bool rdefCamOValid = haveCamO
                && (camO[0] != 0.f || camO[1] != 0.f || camO[2] != 0.f);
              if (!rdefCamOValid && m_hasFanoutCamOrigin) {
                camOforZeroCb3[0] = m_lastFanoutCamOrigin.x;
                camOforZeroCb3[1] = m_lastFanoutCamOrigin.y;
                camOforZeroCb3[2] = m_lastFanoutCamOrigin.z;
                haveCamOforZeroCb3 = true;
              }

              // BSP world VS (VS_bb30826b) fanout path already puts absolute-world
              // translations in i2o (via adjTx = m[3]+camOrigin in the fanout code).
              // For that specific VS + fanout, leave objectToWorld=identity to avoid
              // double-shifting by camOrigin. Every other zeroCb3 case (including
              // fanout particles) gets translate(+camOrigin) as before.
              const bool isFanoutDraw = (m_currentInstancesToObject != nullptr);
              // Short VS key = "VS_" + first 16 hex of SHA1. VS_bb30826b's SHA1 starts
              // with "bb30826b03dc9a8b". Compare by SHA1 prefix string.
              std::string vsKey = getVsHashShort();
              const bool isBspWorldVsFanout = isFanoutDraw
                && vsKey.find("VS_bb30826b03dc9a8b") != std::string::npos;
              if (cb3IsZero && isBspWorldVsFanout) {
                transforms.objectToWorld = Matrix4();  // identity
                m_lastO2wPathId = 7;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.rdef.zeroCb3.bspFanout] vs=", vsKey,
                  " drawID=", m_drawCallID,
                  " → identity (i2o already has +camOrigin from fanout)"));
              } else if (cb3IsZero && haveCamOforZeroCb3) {
                transforms.objectToWorld = Matrix4(
                  Vector4(1.f, 0.f, 0.f, 0.f),
                  Vector4(0.f, 1.f, 0.f, 0.f),
                  Vector4(0.f, 0.f, 1.f, 0.f),
                  Vector4(camOforZeroCb3[0], camOforZeroCb3[1], camOforZeroCb3[2], 1.f));
                m_lastO2wPathId = 6;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.rdef.zeroCb3] vs=", vsKey,
                  " drawID=", m_drawCallID,
                  " isFanout=", isFanoutDraw ? 1 : 0,
                  " camO=(", camOforZeroCb3[0], ",", camOforZeroCb3[1], ",", camOforZeroCb3[2], ")"));
              } else {
                const float tx = haveCamO ? (m[3]  + camO[0]) : m[3];
                const float ty = haveCamO ? (m[7]  + camO[1]) : m[7];
                const float tz = haveCamO ? (m[11] + camO[2]) : m[11];
                // Row-major float3x4: col c of rotation = (m[c], m[4+c], m[8+c]).
                transforms.objectToWorld = Matrix4(
                  Vector4(m[0], m[4], m[8],  0.f),
                  Vector4(m[1], m[5], m[9],  0.f),
                  Vector4(m[2], m[6], m[10], 0.f),
                  Vector4(tx,   ty,   tz,    1.f));
                m_lastO2wPathId = 5;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.rdef] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " slot=", modelCb->bindSlot,
                  " r0=(", m[0], ",", m[1], ",", m[2], ") Tx=", m[3],
                  " r1=(", m[4], ",", m[5], ",", m[6], ") Ty=", m[7],
                  " r2=(", m[8], ",", m[9], ",", m[10], ") Tz=", m[11],
                  " camO=(", camO[0], ",", camO[1], ",", camO[2], ")",
                  " T_abs=(", tx, ",", ty, ",", tz, ")",
                  " haveCamO=", haveCamO ? 1 : 0));
              }
              found = true;
            }
          }
        }
        // Otherwise objectToWorld stays identity (correct for VBs already in
        // camera-relative coords — e.g. world mesh / screen-space passes).
      }

      // ======== LEGACY HEURISTIC FALLBACK (shaders without RDEF only) ========
      // Skipped entirely when we have commonVS-derived metadata above.
      if (!found && commonVS == nullptr) {
        const uint32_t kSourceModelSlot = 3;

        // === VERBOSE SLOT-3 DIAGNOSTIC ===
        // Log slot 3 state every Nth draw (per-frame counter resets outside).
        // Helps diagnose why objectToWorld stays identity in-game.
        static uint32_t s_slot3DiagFrame = UINT32_MAX;
        static uint32_t s_slot3DiagDrawInFrame = 0;
        {
          // Track frame transitions by watching m_drawCallID reset to 0.
          static uint32_t s_lastDrawCallID = UINT32_MAX;
          if (m_drawCallID == 0 || m_drawCallID < s_lastDrawCallID) {
            s_slot3DiagFrame++;
            s_slot3DiagDrawInFrame = 0;
          }
          s_lastDrawCallID = m_drawCallID;
          s_slot3DiagDrawInFrame++;
        }
        // Log: first 3 draws of every in-game frame (camValid via draws>0 approximation),
        // and any draw where slot3 is NULL or identity (unexpected).
        const bool isEarlyDraw = (s_slot3DiagDrawInFrame <= 3);
        const bool logThisDraw = (s_slot3DiagFrame < 600) && (isEarlyDraw || (m_drawCallID % 10 == 0));
        if (logThisDraw) {
          const auto& cb3 = vsCbs[kSourceModelSlot];
          if (cb3.buffer == nullptr) {
            Logger::warn(str::format(
              "[D3D11Rtx] slot3 NULL  frame=", s_slot3DiagFrame,
              " drawInFrame=", s_slot3DiagDrawInFrame,
              " m_drawCallID=", m_drawCallID));
          } else {
            const auto mapped = cb3.buffer->GetMappedSlice();
            const float* f = reinterpret_cast<const float*>(
              static_cast<const uint8_t*>(mapped.mapPtr) + static_cast<size_t>(cb3.constantOffset) * 16);
            if (f && cb3.buffer->Desc()->ByteWidth >= static_cast<size_t>(cb3.constantOffset) * 16 + 48) {
              Logger::info(str::format(
                "[D3D11Rtx] slot3 raw  frame=", s_slot3DiagFrame,
                " draw=", s_slot3DiagDrawInFrame, "/", m_drawCallID,
                " buf=", cb3.buffer->Desc()->ByteWidth,
                " off=", cb3.constantOffset,
                " R0=(", f[0], ",", f[1], ",", f[2], ",", f[3], ")",
                " R1=(", f[4], ",", f[5], ",", f[6], ",", f[7], ")",
                " R2=(", f[8], ",", f[9], ",", f[10], ",", f[11], ")"));
            }
          }
        }

        if (!found)
          found = trySourceFloat3x4(vsCbs, kSourceModelSlot, 0);

        // NV-DXVK (Titanfall 2): camera-relative rendering fallback.
        // Most TF2 VS (VS_759738774e, VS_ef94e6c7, ...) use CBufCommonPerCamera at
        // cb2 and have NO CBufModelInstance at cb3. Vertex buffers already contain
        // (world - c_cameraOrigin); the VS only multiplies by c_cameraRelativeToClip.
        // Remix defaults o2w=identity, builds BLAS at origin, camera sees nothing.
        // Restore world placement by using c_cameraOrigin (cb2 offset 4, float3)
        // as the o2w translation: BLAS_vert + cameraOrigin = (world - cameraOrigin)
        //                                                  + cameraOrigin = world.
        //
        // NV-DXVK: capture the draw's cameraOrigin for the TLAS-coherence filter
        // that runs at the SubmitInstancedDraw call site after ExtractTransforms
        // returns (can't bare-return here — this function returns a value).
        if (!found) {
          const auto& cb2 = vsCbs[2];
          if (cb2.buffer != nullptr) {
            const auto mapped = cb2.buffer->GetMappedSlice();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
            const size_t sz = cb2.buffer->Desc()->ByteWidth;
            if (p && base + 16 <= sz) {
              const float* f = reinterpret_cast<const float*>(p + base);
              // c_cameraOrigin at offset 4 (f[1..3]); f[0] is c_zNear.
              const float camX = f[1], camY = f[2], camZ = f[3];
              // NV-DXVK: diagnostic — log cb2 cameraOrigin for the first few
              // draws per frame, including the BSP gameplay hashes. Expected:
              // all BSP draws in one frame should read the SAME cameraOrigin;
              // if they disagree, we've got stale/inconsistent cb2 state.
              // Also read prev-frame origin at offset 84 for comparison.
              {
                static uint32_t sCamOrigLog = 0;
                if (sCamOrigLog < 60) {
                  // Get current VS hash
                  XXH64_hash_t vsH = 0;
                  if (m_context->m_state.vs.shader != nullptr
                      && m_context->m_state.vs.shader->GetCommonShader() != nullptr) {
                    auto& s = m_context->m_state.vs.shader->GetCommonShader()->GetShader();
                    if (s != nullptr) vsH = static_cast<XXH64_hash_t>(s->getHash());
                  }
                  // Read prev-frame cameraOrigin at offset 84 (float3)
                  float prevX = 0, prevY = 0, prevZ = 0;
                  bool havePrev = false;
                  if (base + 96 <= sz) {
                    const float* fp = reinterpret_cast<const float*>(p + base + 84);
                    prevX = fp[0]; prevY = fp[1]; prevZ = fp[2];
                    havePrev = true;
                  }
                  ++sCamOrigLog;
                  char vsHex[32];
                  std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                                static_cast<unsigned long long>(vsH));
                  Logger::info(str::format(
                    "[D3D11Rtx.camOri] #", sCamOrigLog,
                    " draw=", m_drawCallID,
                    " vs=", vsHex,
                    " cur=(", camX, ",", camY, ",", camZ, ")",
                    " prev=", havePrev ? "yes" : "no",
                    " prevPos=(", prevX, ",", prevY, ",", prevZ, ")",
                    " cb2base=", base, " bufSize=", sz));
                }
              }
              // NV-DXVK: different VS permutations store c_cameraOrigin with
              // different signs at cb2@4 (the fanout BSP VS sees +cam for
              // gameplay camera, while other VS permutations — shadow,
              // reflection, HUD — see NEGATED cam or their own pass-camera).
              // Using the per-draw cb2@4 produces inconsistent o2w.T across
              // draws in the same frame: BSP walls land at -cam, characters
              // at their own pass-cam, etc. Prefer the authoritative
              // m_lastFanoutCamOrigin (captured once per frame from the
              // gameplay BSP fanout VS). The per-draw cb2@4 is only used as
              // a last-resort fallback when fanout hasn't published yet.
              float useCamX = camX, useCamY = camY, useCamZ = camZ;
              bool camFromFanout = false;
              if (m_hasFanoutCamOrigin) {
                useCamX = m_lastFanoutCamOrigin.x;
                useCamY = m_lastFanoutCamOrigin.y;
                useCamZ = m_lastFanoutCamOrigin.z;
                camFromFanout = true;
              }
              if (std::isfinite(useCamX) && std::isfinite(useCamY) && std::isfinite(useCamZ)
                  && (std::abs(useCamX) + std::abs(useCamY) + std::abs(useCamZ)) > 1.0f) {
                transforms.objectToWorld = Matrix4(
                  Vector4(1.0f, 0.0f, 0.0f, 0.0f),
                  Vector4(0.0f, 1.0f, 0.0f, 0.0f),
                  Vector4(0.0f, 0.0f, 1.0f, 0.0f),
                  Vector4(useCamX, useCamY, useCamZ, 1.0f));
                found = true;
                m_lastO2wPathId = 8;
                Logger::info(str::format(
                  "[D3D11Rtx.o2w.cb2cam] vs=", getVsHashShort(),
                  " drawID=", m_drawCallID,
                  " T=(", useCamX, ",", useCamY, ",", useCamZ, ")",
                  " src=", camFromFanout ? "fanout" : "cb2@4"));
                m_lastDrawCamOrigin    = Vector3(useCamX, useCamY, useCamZ);
                m_lastDrawCamOriginSet = true;
                static uint32_t sCamFbLog = 0;
                if (sCamFbLog < 20) {
                  ++sCamFbLog;
                  Logger::info(str::format(
                    "[D3D11Rtx] o2w fallback (camOri): src=",
                    camFromFanout ? "fanout" : "cb2@4",
                    " cb2Cam=(", camX, ",", camY, ",", camZ, ")",
                    " use=(", useCamX, ",", useCamY, ",", useCamZ, ")"));
                }
              }
            }
          }
        }

        // Per-draw diagnostic — now logs first 8 hits PER FRAME rather than
        // per session, so in-game transforms are visible even after menu loading.
        static uint32_t s_f3x4LogFrame = UINT32_MAX;
        static uint32_t s_f3x4LogHitsThisFrame = 0;
        {
          static uint32_t s_f3x4PrevID = UINT32_MAX;
          if (m_drawCallID == 0 || m_drawCallID < s_f3x4PrevID) {
            s_f3x4LogFrame++;
            s_f3x4LogHitsThisFrame = 0;
          }
          s_f3x4PrevID = m_drawCallID;
        }
        if (found && s_f3x4LogHitsThisFrame < 8 && s_f3x4LogFrame < 600) {
          ++s_f3x4LogHitsThisFrame;
          const auto& m = transforms.objectToWorld;
          const bool isIdentity = isIdentityExact(transforms.objectToWorld);
          Logger::info(str::format(
            "[D3D11Rtx] objectToWorld slot=", kSourceModelSlot,
            " frame=", s_f3x4LogFrame, " draw=", m_drawCallID,
            isIdentity ? " IDENTITY" : "",
            " T=(", m[3][0], ",", m[3][1], ",", m[3][2], ")"
            " R=(", m[0][0], ",", m[1][1], ",", m[2][2], ")"));
        }
      }

      // Generic 4x4 scan — fallback for non-Source engines.
      // Skips VS slots 0 and 1 (materialsystem's per-draw/material cbuffers) to
      // avoid false positives from texture/viewport constants.
      if (!found) {
        // Try projSlot+1 first (common layout: proj in slot N, world in slot N+1).
        if (projSlot != UINT32_MAX && projStage == 0
            && projSlot + 1 < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT
            && projSlot + 1 > 1)   // skip slots 0 and 1 (materialsystem)
          found = tryWorldCb(vsCbs, projSlot + 1, projStage, projSlot);

        if (!found) {
          for (uint32_t s = 2; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++s) {
            if (projStage == 0 && s == projSlot) continue;
            if (tryWorldCb(vsCbs, s, projStage, projSlot)) { found = true; break; }
          }
        }
        // Scan non-VS stages for emulator compatibility.
        if (!found) {
          for (int si = 1; si < kNumStages && !found; ++si) {
            for (uint32_t s = 0; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++s) {
              if (si == projStage && s == projSlot) continue;
              if (tryWorldCb(*stageCbs[si], s, projStage, projSlot)) { found = true; break; }
            }
          }
        }
        // Last resort: try float3x4 scan over all VS slots (covers engines that
        // put the model matrix in a non-slot-3 location).
        // NV-DXVK (Titanfall 2): start at slot 1, not 2 — TF2 shaders commonly
        // leave VS s0 null and put a 48-byte float3x4 world matrix in VS s1.
        // trySourceFloat3x4 rejects identity/view/proj so materialsystem data
        // in slot 1 on other engines won't produce false positives.
        if (!found) {
          for (uint32_t s = 1; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT && !found; ++s) {
            if (projStage == 0 && s == projSlot) continue;
            if (trySourceFloat3x4(vsCbs, s)) found = true;
          }
        }

        // === LOG WHEN objectToWorld STAYS IDENTITY (search failed) ===
        if (!found) {
          static uint32_t s_noWorldCount = 0;
          if (s_noWorldCount < 200) {
            ++s_noWorldCount;
            // Dump which VS slots are actually bound so we can see what slot has what.
            std::string slotInfo = "";
            for (uint32_t s = 0; s < 8; ++s) {
              const auto& cb = vsCbs[s];
              if (cb.buffer != nullptr) {
                slotInfo += str::format(" s", s, "=", cb.buffer->Desc()->ByteWidth, "@", cb.constantOffset);
              }
            }
            Logger::warn(str::format(
              "[D3D11Rtx] objectToWorld NOT FOUND (identity) drawID=", m_drawCallID,
              " projSlot=", projSlot, " projStage=", projStage,
              " boundVS:", slotInfo));
          }
        }
      }
    }

    // NV-DXVK: For bone draws, use the worldToView from a fmt=106 draw
    // (which has the correct camera). Reset objectToWorld to identity
    // since the interleaver applies the bone matrix GPU-side.
    if (m_currentDrawIsBoneTransformed) {
      transforms.objectToWorld = Matrix4();
      // Build worldToView from c_cameraOrigin (cb2@4) with explicit
      // Source Engine coordinate system mapping:
      //   Source: X=forward, Y=left, Z=up
      //   D3D view: X=right, Y=up, Z=forward
      // View rotation: D3D_right = Source_-Y, D3D_up = Source_Z, D3D_fwd = Source_X
      float camX = 0, camY = 0, camZ = 0;
      const auto& camCb = m_context->m_state.vs.constantBuffers[2];
      if (camCb.buffer != nullptr) {
        const auto mapped = camCb.buffer->GetMappedSlice();
        const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (p && camCb.buffer->Desc()->ByteWidth >= 16) {
          const float* co = reinterpret_cast<const float*>(p + 4);
          camX = co[0]; camY = co[1]; camZ = co[2];
        }
      }
      // Source: X=forward, Y=left, Z=up
      // Use the cached VP's camera direction (fwd from decomposition)
      // but fix the up axis which the VP decomposition negates (Y-flip).
      const Matrix4& cachedView = m_lastGoodTransforms.worldToView;
      // Extract axes from cached view, fix the Y-flipped up
      Vector3 right(cachedView[0][0], cachedView[0][1], cachedView[0][2]);
      Vector3 up   (cachedView[1][0], cachedView[1][1], cachedView[1][2]);
      Vector3 fwd  (cachedView[2][0], cachedView[2][1], cachedView[2][2]);
      // Fix Y-flip: if up.z is negative, the VP decomposition flipped it
      if (up.z < 0) {
        up.x = -up.x; up.y = -up.y; up.z = -up.z;
      }
      // Check if cached rotation is valid (non-zero)
      bool hasCachedRotation = (length(right) > 0.5f && length(fwd) > 0.5f);
      if (!hasCachedRotation) {
        // Fallback: hardcoded Source→D3D axis mapping (camera looking +X)
        right = Vector3(0, -1, 0);
        up    = Vector3(0,  0, 1);
        fwd   = Vector3(1,  0, 0);
      }
      const float tR = -(right.x*camX + right.y*camY + right.z*camZ);
      const float tU = -(up.x*camX    + up.y*camY    + up.z*camZ);
      const float tF = -(fwd.x*camX   + fwd.y*camY   + fwd.z*camZ);
      m_lastWtvPathId = 11; // cached VP + live camX/Y/Z reuse path
      // NV-DXVK: store by columns — see path 1 fix.
      transforms.worldToView = Matrix4(
        Vector4(right.x, up.x, fwd.x, 0),
        Vector4(right.y, up.y, fwd.y, 0),
        Vector4(right.z, up.z, fwd.z, 0),
        Vector4(tR,      tU,   tF,   1));
      static uint32_t sViewDiag = 0;
      if (sViewDiag < 5) {
        ++sViewDiag;
        Logger::info(str::format(
          "[D3D11Rtx] Bone view: cam=(", camX, ",", camY, ",", camZ, ")",
          " R=(", right.x, ",", right.y, ",", right.z, ")",
          " U=(", up.x, ",", up.y, ",", up.z, ")",
          " F=(", fwd.x, ",", fwd.y, ",", fwd.z, ")",
          " cached=", hasCachedRotation ? 1 : 0));
      }
    }

    transforms.objectToView = transforms.objectToWorld;
    if (!isIdentityExact(transforms.worldToView))
      transforms.objectToView = transforms.worldToView * transforms.objectToWorld;

    transforms.sanitize();

    // Log camera discovery once.
    static bool s_cameraLogged = false;
    if (projSlot != UINT32_MAX && !s_cameraLogged) {
      s_cameraLogged = true;
      const auto& p = transforms.viewToProjection;
      Logger::info(str::format(
        "[D3D11Rtx] Camera found: stage=", kStageNames[projStage],
        " slot=", projSlot, " off=", projOffset,
        " proj diag=(", p[0][0], ",", p[1][1], ",", p[2][2], ")",
        " m[2][3]=", p[2][3],
        m_columnMajor ? " [column-major]" : " [row-major]"));
    }

    // DEBUG: dump info for non-bone draws returning identity o2w.
    // Per-frame counter so we don't saturate on drawID=0. Skip fallback draws
    // because those get rejected downstream — we want the REAL submissions.
    if (!m_currentDrawIsBoneTransformed && isIdentityExact(transforms.objectToWorld)
        && !m_lastExtractUsedFallback) {
      static uint32_t s_silentFrame = UINT32_MAX;
      static uint32_t s_silentPerFrame = 0;
      static uint32_t s_silentPrevID = UINT32_MAX;
      if (m_drawCallID == 0 || m_drawCallID < s_silentPrevID) {
        s_silentFrame++;
        s_silentPerFrame = 0;
      }
      s_silentPrevID = m_drawCallID;
      if (s_silentFrame < 3 && s_silentPerFrame < 3) {
        ++s_silentPerFrame;
        const auto& vsCbs = m_context->m_state.vs.constantBuffers;
        std::string vsHashStr = "?";
        auto vsShader = m_context->m_state.vs.shader;
        if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
          auto& shader = vsShader->GetCommonShader()->GetShader();
          if (shader != nullptr)
            vsHashStr = shader->getShaderKey().toString();
        }
        Logger::warn(str::format("[D3D11Rtx] === SILENT-ID drawID=", m_drawCallID, " VS=", vsHashStr, " cbuffer dump ==="));
        for (uint32_t s = 0; s < 8; ++s) {
          const auto& cb = vsCbs[s];
          if (cb.buffer == nullptr) continue;
          const auto mapped = cb.buffer->GetMappedSlice();
          const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
          if (!p) continue;
          const size_t base = static_cast<size_t>(cb.constantOffset) * 16;
          const size_t sz = cb.buffer->Desc()->ByteWidth;
          const size_t n = std::min<size_t>(sz - base, 256);
          const float* f = reinterpret_cast<const float*>(p + base);
          // Dump as float4 rows
          for (size_t r = 0; r * 16 + 16 <= n; ++r) {
            Logger::warn(str::format(
              "  VS s", s, " [", r, "] = (", f[r*4+0], ", ", f[r*4+1], ", ", f[r*4+2], ", ", f[r*4+3], ")"));
          }
        }
      }
    }

    // ================================================================
    // NV-DXVK: CLASSIFIER-DRIVEN OVERRIDE (V2 dispatcher, Phase 1)
    // ================================================================
    // Runs after the legacy path-selection tangle. For kinds the classifier
    // is confident about, we overwrite `transforms.objectToWorld` with a
    // deterministic RDEF-sourced value. Kinds whose legacy path already
    // works (InstancedBsp, SkinnedChar) keep whatever the legacy code set.
    //
    // This block is the single source of truth for:
    //   StaticWorld  → cb3.CBufModelInstance.objectToCameraRelative
    //                  + cb2.CBufCommonPerCamera.c_cameraOrigin (to shift to
    //                  absolute world). PIX-verified on VS_6e3e6f28f2156ea2.
    //   UI/Unknown   → mark fallback so SubmitDraw filters as UIFallback.
    //
    // Guarantee: once this block runs, no downstream code should modify
    // objectToWorld. If a legacy path set a different o2w earlier in this
    // function, the override replaces it.
    {
      auto vsPtrV2 = m_context->m_state.vs.shader;
      const D3D11CommonShader* commonV2 =
        (vsPtrV2 != nullptr) ? vsPtrV2->GetCommonShader() : nullptr;
      const auto* ilV2 = m_context->m_state.ia.inputLayout.ptr();
      const std::vector<D3D11RtxSemantic> kEmptySemsV2;
      const auto& semsV2 = ilV2 ? ilV2->GetRtxSemantics() : kEmptySemsV2;
      const auto clsV2 = D3D11VsClassifier::classify(commonV2, semsV2);

      auto readCbFloats = [&](uint32_t slot, uint32_t byteOff, uint32_t nBytes,
                              float* out) -> bool {
        if (slot >= D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT) return false;
        const auto& cb = m_context->m_state.vs.constantBuffers[slot];
        if (cb.buffer == nullptr) return false;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) return false;
        const size_t base = static_cast<size_t>(cb.constantOffset) * 16 + byteOff;
        if (base + nBytes > cb.buffer->Desc()->ByteWidth) return false;
        std::memcpy(out, ptr + base, nBytes);
        for (uint32_t i = 0; i < nBytes / 4; ++i)
          if (!std::isfinite(out[i])) return false;
        return true;
      };

      // Helper: detect "zero" matrix/vec3 (all components exactly 0.f).
      auto allZero = [](const float* p, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) if (p[i] != 0.f) return false;
        return true;
      };

      switch (clsV2.kind) {
        case D3D11VsClassification::Kind::StaticWorld: {
          // V2 StaticWorld scope: UI-demotion only.
          // Read cb3 + camO purely to detect the menu/pre-gameplay case
          // where shader declares CBufModelInstance but contents are
          // uninitialized. Do NOT override legacy objectToWorld. The
          // legacy cb3-RDEF path (o2wPath=5) already produces the right
          // o2w for real gameplay draws (PIX-verified on VS_6e3e6f28);
          // overriding it here submits draws that RT can't render into
          // usable pixels, which flipped m_remixActiveThisFrame=true and
          // caused the RT blit to clobber everything the native raster
          // had drawn.
          float m[12];
          if (readCbFloats(clsV2.cb3Slot, /*offset*/ 0, /*size*/ 48, m)) {
            // Read camera origin from CBufCommonPerCamera.c_cameraOrigin.
            // Field offset is RDEF-declared; if the shader doesn't expose it
            // explicitly, fall back to the canonical cb2 offset-4 location
            // used by every TF2 Source 2 shader observed so far.
            float camO[3] = { 0.f, 0.f, 0.f };
            bool haveCam = false;
            if (commonV2 != nullptr) {
              auto camLoc = commonV2->FindCBField(
                  "CBufCommonPerCamera", "c_cameraOrigin");
              if (camLoc && camLoc->size >= 12) {
                haveCam = readCbFloats(camLoc->slot, camLoc->offset, 12, camO);
              }
            }
            if (!haveCam) {
              haveCam = readCbFloats(/*cb2*/ 2, /*byteOff*/ 4, 12, camO);
            }
            // Fallback to the BSP-fanout captured camera if both above failed
            // (very early boot frames where cb2 isn't populated yet).
            if (!haveCam && m_hasFanoutCamOrigin) {
              camO[0] = m_lastFanoutCamOrigin.x;
              camO[1] = m_lastFanoutCamOrigin.y;
              camO[2] = m_lastFanoutCamOrigin.z;
              haveCam = true;
            }
            // Menu / pre-gameplay guard.
            // TF2's UI and menu shaders re-use the StaticWorld shader
            // template (same CBufModelInstance cbuffer declared in RDEF)
            // but the cbuffer contents are never written during menu
            // frames — cb3 holds identity or zeros, and the gameplay
            // camera origin hasn't been captured yet (camO all zero).
            // If we commit these to RT the BLASes pile up at world
            // origin, render nothing, and flip m_remixActiveThisFrame=
            // true so the RT blit runs and clobbers the native-raster
            // output the UI/menu buttons were drawn into.
            //
            // Signal: cameraOrigin == 0. No valid world placement is
            // possible without a real camera. Demote to UI (native
            // raster renders it, RT stays out of this frame).
            if (allZero(camO, 3)) {
              m_lastExtractUsedFallback = true;
              m_lastClassifierSaidUi    = true;
              static std::unordered_set<std::string> sV2LogUiDemote;
              const std::string vk = getVsHashShort();
              if (sV2LogUiDemote.insert(vk).second) {
                Logger::info(str::format(
                  "[VsClass.v2.StaticWorld.demoteUI] vs=", vk,
                  " reason=camO_all_zero_no_real_camera"));
              }
              break;
            }
            // (no override — legacy path owns objectToWorld for real
            // gameplay draws; we only demoted the menu case above)
            static std::unordered_set<std::string> sV2LogStatic;
            const std::string vk = getVsHashShort();
            if (sV2LogStatic.insert(vk).second) {
              Logger::info(str::format(
                "[VsClass.v2.StaticWorld.pass] vs=", vk,
                " cb3Slot=", clsV2.cb3Slot,
                " haveCam=", haveCam ? 1 : 0,
                " m[3,7,11]=(", m[3], ",", m[7], ",", m[11], ")",
                " camO=(", camO[0], ",", camO[1], ",", camO[2], ")"));
            }
          }
          break;
        }
        case D3D11VsClassification::Kind::InstancedBsp:
        case D3D11VsClassification::Kind::SkinnedChar: {
          // The classifier has definitively identified these as real
          // rendering geometry (InstancedBsp = per-instance t31; SkinnedChar
          // = BLENDINDICES bone palette). Neither is UI. The legacy o2w
          // extraction for these kinds is left intact, but we force the
          // fallback flag false so SubmitDraw's UIFallback filter does not
          // accidentally reject them when a legacy sub-branch hit a
          // per-draw edge case (e.g. t31 entry out of range, cached bone
          // slice null) and wrote `m_lastExtractUsedFallback = true` at
          // line ~2806. If legacy produced a bad o2w, the finiteness /
          // magnitude guard in SubmitDraw (line ~5340) still rejects it;
          // we're only undoing the UIFallback class of rejection.
          m_lastExtractUsedFallback = false;
          static std::unordered_set<std::string> sV2LogRecognized;
          const std::string vk = getVsHashShort();
          const std::string key =
              std::string(D3D11VsClassifier::kindName(clsV2.kind)) + "|" + vk;
          if (sV2LogRecognized.insert(key).second) {
            Logger::info(str::format(
              "[VsClass.v2.", D3D11VsClassifier::kindName(clsV2.kind), "] vs=", vk,
              " fallback_cleared legacy_o2wPath=", m_lastO2wPathId));
          }
          break;
        }
        case D3D11VsClassification::Kind::UI:
        case D3D11VsClassification::Kind::Unknown:
          // No recognized transform signals. Force UIFallback so the native
          // raster path handles the draw and RTX skips it. Setting
          // m_lastClassifierSaidUi = true forces SubmitDraw into the TRUE
          // UI branch (line ~5880) regardless of m_foundRealProjThisFrame,
          // which is what makes the menu buttons/HUD reach native raster
          // after any gameplay draw has latched a real projection.
          m_lastExtractUsedFallback = true;
          m_lastClassifierSaidUi    = true;
          break;
        default:
          // Skybox/Viewmodel/Particle/Sprite2D — not produced by the
          // classifier yet. Fall through silently.
          break;
      }
    }

    return transforms;
  }

  // NV-DXVK: latch set in EndFrame once real gameplay starts — drives the
  // per-draw Submit log so it prints during actual scene rendering, not boot.
  static uint32_t s_GameplayLogFrames = 0;

  // NV-DXVK: helper — bump m_filterCounts AND record the reject against the
  // current VS so EndFrame can show per-shader outcomes.
  void D3D11Rtx::BumpFilter(FilterReason r) {
    const uint32_t ri = static_cast<uint32_t>(r);
    ++m_filterCounts[ri];
    if (!m_currentVsHashCache.empty()) {
      auto& st = m_vsFrameStats[m_currentVsHashCache];
      ++st.rejects[ri];
    }
  }

  Future<GeometryHashes> D3D11Rtx::ComputeGeometryHashes(
      const RasterGeometry& geo, uint32_t vertexCount,
      uint32_t hashStartVertex, uint32_t hashVertexCount) const {

    const void* posData = geo.positionBuffer.mapPtr(geo.positionBuffer.offsetFromSlice());
    const void* tcData  = geo.texcoordBuffer.defined()
                        ? geo.texcoordBuffer.mapPtr(geo.texcoordBuffer.offsetFromSlice())
                        : nullptr;
    const void* idxData = geo.indexBuffer.defined() ? geo.indexBuffer.mapPtr(0) : nullptr;

    // D3D11 dynamic buffers can be discarded (Map WRITE_DISCARD) at any time,
    // which recycles the physical slice backing our raw pointers.  Pin each
    // buffer with incRef + acquire(Read) so the allocator won't reuse the
    // memory while the hash worker is reading it.  The lambda releases them.
    DxvkBuffer* posBuf = geo.positionBuffer.buffer().ptr();
    DxvkBuffer* tcBuf  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.buffer().ptr() : nullptr;
    DxvkBuffer* idxBuf = geo.indexBuffer.defined()    ? geo.indexBuffer.buffer().ptr()    : nullptr;

    if (posBuf) { posBuf->incRef(); posBuf->acquire(DxvkAccess::Read); }
    if (tcBuf)  { tcBuf->incRef();  tcBuf->acquire(DxvkAccess::Read);  }
    if (idxBuf) { idxBuf->incRef(); idxBuf->acquire(DxvkAccess::Read); }

    const uint32_t posStride = geo.positionBuffer.stride();
    const uint32_t tcStride  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.stride() : 0u;
    const uint32_t idxStride = geo.indexBuffer.defined()    ? geo.indexBuffer.stride()    : 0u;
    const uint32_t indexType = static_cast<uint32_t>(geo.indexBuffer.indexType());
    const uint32_t topology  = static_cast<uint32_t>(geo.topology);

    const uint32_t posOffset = geo.positionBuffer.offsetFromSlice();

    XXH64_hash_t descHash   = hashGeometryDescriptor(geo.indexCount, vertexCount, indexType, topology);
    // NV-DXVK: Mix bone instance index into hash so each instance gets a unique BLAS
    if (geo.boneInstanceIndex != 0) {
      const uint32_t bi = geo.boneInstanceIndex;
      descHash = XXH3_64bits_withSeed(&bi, sizeof(bi), descHash);
    }
    const XXH64_hash_t layoutHash = hashVertexLayout(geo);

    // Compute the safe byte range available for position and texcoord data.
    // Buffer pins guarantee the memory won't be recycled, but we must still
    // clamp to the actual buffer extent to avoid reading past the allocation.
    const size_t posLength = geo.positionBuffer.length();
    const size_t tcLength  = geo.texcoordBuffer.defined() ? geo.texcoordBuffer.length() : 0;
    const size_t idxLength = geo.indexBuffer.defined()    ? geo.indexBuffer.length()    : 0;

    auto future = m_pGeometryWorkers->Schedule([posData, tcData, idxData,
                                         posBuf, tcBuf, idxBuf,
                                         posStride, tcStride, idxStride,
                                         posLength, tcLength, idxLength,
                                         vertexCount, indexCount = geo.indexCount,
                                         posOffset,
                                         hashStartVertex, hashVertexCount,
                                         descHash, layoutHash]() -> GeometryHashes {
      GeometryHashes hashes;
      hashes[HashComponents::GeometryDescriptor] = descHash;
      hashes[HashComponents::VertexLayout]       = layoutHash;

      if (posData && posStride > 0) {
        // Hash only the drawn subrange [hashStartVertex, hashStartVertex + hashVertexCount).
        // Clamp to actual buffer length to prevent OOB reads on shared/dynamic VBs.
        const size_t startByte = static_cast<size_t>(hashStartVertex) * posStride;
        size_t posBytes = static_cast<size_t>(hashVertexCount) * posStride;
        if (startByte >= posLength) {
          posBytes = 0;
        } else if (startByte + posBytes > posLength) {
          posBytes = posLength - startByte;
        }
        if (posBytes > 0) {
          const auto* posBase = static_cast<const uint8_t*>(posData) + startByte;
          hashes[HashComponents::VertexPosition] =
            XXH3_64bits_withSeed(posBase, posBytes, static_cast<XXH64_hash_t>(hashStartVertex));
        } else {
          hashes[HashComponents::VertexPosition] =
            XXH3_64bits(&posOffset, sizeof(posOffset));
        }

        if (tcData && tcStride > 0) {
          const size_t tcStartByte = static_cast<size_t>(hashStartVertex) * tcStride;
          size_t tcBytes = static_cast<size_t>(hashVertexCount) * tcStride;
          if (tcStartByte >= tcLength) {
            tcBytes = 0;
          } else if (tcStartByte + tcBytes > tcLength) {
            tcBytes = tcLength - tcStartByte;
          }
          if (tcBytes > 0) {
            const auto* tcBase = static_cast<const uint8_t*>(tcData) + tcStartByte;
            hashes[HashComponents::VertexTexcoord] =
              XXH3_64bits_withSeed(tcBase, tcBytes, static_cast<XXH64_hash_t>(hashStartVertex));
          }
        }
        if (idxData && idxStride > 0) {
          const size_t idxBytes = static_cast<size_t>(indexCount) * idxStride;
          hashes[HashComponents::Indices] =
            hashContiguousMemory(idxData, std::min(idxBytes, idxLength));
        }
      } else {
        // GPU-only buffer: stable identity hash from buffer address and offset.
        XXH64_hash_t posHash = XXH3_64bits(&posBuf, sizeof(posBuf));
        posHash = XXH3_64bits_withSeed(&posOffset, sizeof(posOffset), posHash);
        hashes[HashComponents::VertexPosition] = posHash;
      }

      hashes.precombine();

      // Release buffer pins — allow slice recycling again.
      if (posBuf) { posBuf->release(DxvkAccess::Read); posBuf->decRef(); }
      if (tcBuf)  { tcBuf->release(DxvkAccess::Read);  tcBuf->decRef();  }
      if (idxBuf) { idxBuf->release(DxvkAccess::Read); idxBuf->decRef(); }

      return hashes;
    });

    // If the worker queue was full, the lambda never runs — release pins now
    // to prevent a VRAM leak (incRef/acquire above would never be undone).
    if (!future.valid()) {
      if (posBuf) { posBuf->release(DxvkAccess::Read); posBuf->decRef(); }
      if (tcBuf)  { tcBuf->release(DxvkAccess::Read);  tcBuf->decRef();  }
      if (idxBuf) { idxBuf->release(DxvkAccess::Read); idxBuf->decRef(); }
    }

    return future;
  }

  void D3D11Rtx::FillMaterialData(LegacyMaterialData& mat) const {
    const auto& ps = m_context->m_state.ps;
    uint32_t textureID = 0;

    static uint32_t s_logCount = 0;
    const bool doLog = (s_logCount < 10);
    std::string logMsg;

    auto isBlockCompressed = [](DXGI_FORMAT fmt) -> bool {
      return (fmt >= DXGI_FORMAT_BC1_TYPELESS && fmt <= DXGI_FORMAT_BC1_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC2_TYPELESS && fmt <= DXGI_FORMAT_BC2_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC3_TYPELESS && fmt <= DXGI_FORMAT_BC3_UNORM_SRGB)
          || (fmt >= DXGI_FORMAT_BC4_TYPELESS && fmt <= DXGI_FORMAT_BC4_SNORM)
          || (fmt >= DXGI_FORMAT_BC5_TYPELESS && fmt <= DXGI_FORMAT_BC5_SNORM)
          || (fmt >= DXGI_FORMAT_BC6H_TYPELESS && fmt <= DXGI_FORMAT_BC7_UNORM_SRGB);
    };

    // Collect currently-bound render target images AND their dimensions.
    // Only reject SRVs that point to images actively bound as RTs.
    // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT is set on most D3D11 textures
    // (engines create them with BIND_RENDER_TARGET for mip gen, dynamic
    // updates, etc.), so the flag alone is NOT a reliable RT indicator.
    const auto& omState = m_context->m_state.om;
    std::array<DxvkImage*, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> boundRTImages = {};
    uint32_t rtWidth = 0, rtHeight = 0;
    for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
      auto* rtv = omState.renderTargetViews[rt].ptr();
      if (rtv) {
        Rc<DxvkImageView> rtvView = rtv->GetImageView();
        if (rtvView != nullptr) {
          boundRTImages[rt] = rtvView->image().ptr();
          if (rt == 0) {
            rtWidth  = rtvView->image()->info().extent.width;
            rtHeight = rtvView->image()->info().extent.height;
          }
        }
      }
    }

    // First pass: collect candidate textures with scoring.
    // Score: BC=+10, mips=+5, non-RT-sized=+3, lower slot=+1.
    // This replaces the old binary accept/reject that was too aggressive.
    struct TexCandidate {
      uint32_t slot;
      Rc<DxvkImageView> view;
      int score;
      bool isCurrentRT;
      std::string info;
    };
    std::vector<TexCandidate> candidates;

    for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; ++slot) {
      D3D11ShaderResourceView* srv = ps.shaderResources.views[slot].ptr();
      if (!srv) continue;
      if (srv->GetResourceType() != D3D11_RESOURCE_DIMENSION_TEXTURE2D) continue;

      Rc<DxvkImageView> view = srv->GetImageView();
      if (view == nullptr) continue;

      const auto& imgInfo = view->image()->info();
      D3D11_SHADER_RESOURCE_VIEW_DESC1 srvDesc = {};
      srv->GetDesc1(&srvDesc);
      const DXGI_FORMAT fmt = srvDesc.Format;
      const bool bc = isBlockCompressed(fmt);
      const bool hasMips = imgInfo.mipLevels > 1;

      DxvkImage* srvImage = view->image().ptr();
      bool isCurrentRT = false;
      for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
        if (boundRTImages[rt] == srvImage) { isCurrentRT = true; break; }
      }

      // Skip tiny dummy textures (1x1 default white/black).
      if (imgInfo.extent.width <= 2 && imgInfo.extent.height <= 2)
        continue;

      // Check if texture dimensions match current render target (likely GBuffer/intermediate).
      const bool matchesRT = (rtWidth > 0 && rtHeight > 0
        && imgInfo.extent.width == rtWidth && imgInfo.extent.height == rtHeight);

      int score = 0;
      if (bc)                       score += 10;  // Block-compressed = always content
      if (hasMips)                  score += 5;   // Mipmapped = likely content
      if (!matchesRT)               score += 3;   // Different size from RT = likely content
      if (!isCurrentRT)             score += 2;   // Not actively rendering to it
      score += std::max(0, 16 - (int)slot);       // Prefer lower slots (albedo first)

      // Currently bound as active RT → negative score (only use as absolute last resort)
      if (isCurrentRT) score = -10;

      std::string info;
      if (doLog) {
        info = str::format("  slot=", slot,
          " fmt=", (uint32_t)fmt,
          " w=", imgInfo.extent.width, " h=", imgInfo.extent.height,
          " mips=", imgInfo.mipLevels,
          " score=", score,
          bc ? " [BC]" : "",
          hasMips ? " [MIPS]" : "",
          isCurrentRT ? " [BOUND-RT]" : "",
          matchesRT ? " [RT-SIZED]" : "", "\n");
      }

      candidates.push_back({ slot, std::move(view), score, isCurrentRT, std::move(info) });
    }

    // Sort by score descending — best content textures first.
    std::sort(candidates.begin(), candidates.end(),
      [](const TexCandidate& a, const TexCandidate& b) { return a.score > b.score; });

    // Pick up to kMaxSupportedTextures (or 1 if ignoreSecondaryTextures is set).
    // If all have negative scores (all are currently-bound RTs), accept the
    // least-bad one rather than submitting zero textures.
    const uint32_t maxTextures = RtxOptions::ignoreSecondaryTextures()
                                ? 1u : LegacyMaterialData::kMaxSupportedTextures;
    bool pickedAny = false;
    for (auto& c : candidates) {
      if (textureID >= maxTextures) break;
      // Skip currently-bound RTs unless we have no other option.
      if (c.isCurrentRT && !candidates.empty() && candidates[0].score > 0)
        continue;

      mat.colorTextures[textureID] = TextureRef(std::move(c.view));
      mat.colorTextureSlot[textureID] = c.slot;

      if (c.slot < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
        D3D11SamplerState* samp = ps.samplers[c.slot];
        mat.samplers[textureID] = samp ? samp->GetDXVKSampler() : getDefaultSampler();
      } else {
        mat.samplers[textureID] = getDefaultSampler();
      }

      pickedAny = true;
      ++textureID;
    }

    // If nothing was picked and there are candidates, take the best one anyway.
    // Remix with a dubious texture is better than Remix with no texture at all.
    if (!pickedAny && !candidates.empty()) {
      auto& c = candidates[0];
      mat.colorTextures[0] = TextureRef(Rc<DxvkImageView>(c.view));
      mat.colorTextureSlot[0] = c.slot;
      if (c.slot < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) {
        D3D11SamplerState* samp = ps.samplers[c.slot];
        mat.samplers[0] = samp ? samp->GetDXVKSampler() : getDefaultSampler();
      } else {
        mat.samplers[0] = getDefaultSampler();
      }
      textureID = 1;
    }

    if (doLog && !candidates.empty()) {
      for (auto& c : candidates)
        logMsg += c.info;
      Logger::info(str::format("[D3D11Rtx] FillMaterialData draw #", s_logCount,
        " picked ", textureID, " of ", candidates.size(), " candidate(s):\n", logMsg));
      ++s_logCount;
    }

    // Material defaults for the Remix legacy material pipeline.
    // D3D11 bakes blending/alpha into immutable state objects — we extract
    // what we can from BlendState and DepthStencilState below.
    mat.textureColorArg1Source  = RtTextureArgSource::Texture;
    mat.textureColorArg2Source  = RtTextureArgSource::None;
    mat.textureColorOperation   = DxvkRtTextureOperation::Modulate;
    mat.textureAlphaArg1Source  = RtTextureArgSource::Texture;
    mat.textureAlphaArg2Source  = RtTextureArgSource::None;
    mat.textureAlphaOperation   = DxvkRtTextureOperation::SelectArg1;
    mat.tFactor                 = 0xFFFFFFFF;  // Opaque white
    mat.diffuseColorSource      = RtTextureArgSource::None;
    mat.specularColorSource     = RtTextureArgSource::None;

    // --- Blend state ---
    D3D11BlendState* blendState = m_context->m_state.om.cbState;
    if (blendState) {
      D3D11_BLEND_DESC1 blendDesc;
      blendState->GetDesc1(&blendDesc);
      const auto& rt0 = blendDesc.RenderTarget[0];

      mat.blendMode.enableBlending = rt0.BlendEnable;
      mat.blendMode.colorSrcFactor = mapD3D11Blend(rt0.SrcBlend, false);
      mat.blendMode.colorDstFactor = mapD3D11Blend(rt0.DestBlend, false);
      mat.blendMode.colorBlendOp   = mapD3D11BlendOp(rt0.BlendOp);
      mat.blendMode.alphaSrcFactor = mapD3D11Blend(rt0.SrcBlendAlpha, true);
      mat.blendMode.alphaDstFactor = mapD3D11Blend(rt0.DestBlendAlpha, true);
      mat.blendMode.alphaBlendOp   = mapD3D11BlendOp(rt0.BlendOpAlpha);
      mat.blendMode.writeMask      = rt0.RenderTargetWriteMask;

      // AlphaToCoverage = D3D11's cutout transparency (foliage, fences, hair).
      if (blendDesc.AlphaToCoverageEnable) {
        mat.alphaTestEnabled       = true;
        mat.alphaTestCompareOp     = VK_COMPARE_OP_GREATER;
        mat.alphaTestReferenceValue = 128;
      }
    }

    // --- Alpha test from depth-stencil state ---
    // Some engines use stencil ops to simulate alpha test; detect write-mask-zero
    // with stencil as a proxy for "discard if alpha < ref".
    D3D11DepthStencilState* dsState = m_context->m_state.om.dsState;
    if (dsState && !mat.alphaTestEnabled) {
      D3D11_DEPTH_STENCIL_DESC dsDesc;
      dsState->GetDesc(&dsDesc);
      if (dsDesc.StencilEnable && dsDesc.FrontFace.StencilFunc == D3D11_COMPARISON_LESS) {
        mat.alphaTestEnabled        = true;
        mat.alphaTestCompareOp      = VK_COMPARE_OP_GREATER;
        mat.alphaTestReferenceValue  = dsDesc.StencilReadMask;
      }
    }

    mat.updateCachedHash();
  }

  void D3D11Rtx::SubmitDraw(bool indexed,
                             UINT count,
                             UINT start,
                             INT  base,
                             const Matrix4* instanceTransform) {
    // NV-DXVK: cache VS hash at entry so BumpFilter() / submit tracking can
    // attribute stats without re-fetching it at every reject site.
    m_currentVsHashCache.clear();
    m_skinnedCharNeedsCamOffset = false;
    const D3D11CommonShader* commonVsForLog = nullptr;
    {
      auto vsShader = m_context->m_state.vs.shader;
      if (vsShader != nullptr && vsShader->GetCommonShader() != nullptr) {
        commonVsForLog = vsShader->GetCommonShader();
        auto& s = commonVsForLog->GetShader();
        if (s != nullptr) m_currentVsHashCache = s->getShaderKey().toString();
      }
    }

    // NV-DXVK: one-shot per-VS signature dump — list the cbuffers + SRVs the
    // shader binds + bound VB layout, so we know what each unique shader
    // looks like without having to run fxc /dumpbin on every hash. Dumped
    // exactly once per unique VS per session.
    if (!m_currentVsHashCache.empty() && commonVsForLog != nullptr) {
      const std::string shortKey = m_currentVsHashCache.substr(0, 19);
      if (m_vsRdefDumped.insert(shortKey).second) {
        std::string cbLine;
        for (uint32_t s = 0; s < 14; ++s) {
          const auto& cb = m_context->m_state.vs.constantBuffers[s];
          if (cb.buffer == nullptr) continue;
          cbLine += str::format(" cb", s, "=", cb.buffer->Desc()->ByteWidth,
                                 "@", cb.constantOffset);
        }
        std::string srvLine;
        for (uint32_t s = 0; s < 32; ++s) {
          if (s >= D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) break;
          auto* srv = m_context->m_state.vs.shaderResources.views[s].ptr();
          if (!srv) continue;
          Com<ID3D11Resource> res; srv->GetResource(&res);
          auto* buf = static_cast<D3D11Buffer*>(res.ptr());
          size_t bsz = buf ? buf->Desc()->ByteWidth : 0;
          srvLine += str::format(" t", s, "=", bsz);
        }
        std::string semLine;
        auto* il = m_context->m_state.ia.inputLayout.ptr();
        if (il) {
          for (const auto& sem : il->GetRtxSemantics()) {
            semLine += str::format(" ", sem.name, sem.index,
                                    (sem.perInstance ? "/I" : "/V"),
                                    ":fmt", uint32_t(sem.format),
                                    ":slot", sem.inputSlot,
                                    ":off", sem.byteOffset);
          }
        }
        Logger::info(str::format(
          "[D3D11Rtx.vs.sig] vs=", shortKey, " cbuffers:", cbLine,
          " SRVs:", srvLine, " semantics:", semLine));
      }
    }

    // NV-DXVK: Diagnostic — confirm SubmitDraw is reached
    {
      static uint32_t sEntryLog = 0;
      if (sEntryLog < 3) {
        ++sEntryLog;
        Logger::info(str::format("[D3D11Rtx] SubmitDraw ENTERED count=", count,
          " indexed=", indexed ? 1 : 0, " raw=", m_rawDrawCount));
      }
    }

    // NV-DXVK: Previously this returned early on deferred contexts because
    // D3D11Rtx::Initialize() is only called on the immediate context, leaving
    // m_pGeometryWorkers null everywhere else.  That meant Source-engine
    // games like Titanfall 2 — which batch-record every material draw onto
    // deferred contexts via materialsystem_dx11's threaded queue — fed zero
    // geometry to the RTX pipeline: the main menu rendered as an empty
    // ray-traced clear color with no actual scene content.
    //
    // The deferred-context CS stream is already recorded into the
    // D3D11CommandList and replayed in order by D3D11ImmediateContext::
    // ExecuteCommandList, so any EmitCs callbacks posted here will run on the
    // CS thread at the correct point relative to the game's native draws.
    // The only thing we were missing was the worker pool.  Lazy-allocate one
    // on first use so the immediate context keeps its eagerly-allocated pool
    // while every deferred context gets its own on demand.
    if (m_pGeometryWorkers == nullptr) {
      const uint32_t cores = std::max(2u, std::thread::hardware_concurrency());
      const uint32_t workers = std::min(std::max(cores / 2, 2u), 6u);
      m_pGeometryWorkers = std::make_unique<GeometryProcessor>(workers, "d3d11-geometry-def");
    }

    // NV-DXVK: One-shot cbuffer dump to identify Source's projection matrix
    // layout.  Titanfall 2 gameplay frames show raw=500+ draws but every
    // single one is being rejected as UIFallback because
    // classifyPerspective() never matches any matrix in Source's cbuffers.
    //
    // Gating: trigger only once per session, on a draw that comes *late*
    // in a frame that has already seen many draws.  m_rawDrawCount is the
    // frame-level counter incremented at the top of OnDraw*/OnDrawIndexed*
    // BEFORE SubmitDraw is invoked, so by the time we reach here for the
    // Nth draw of the frame it's already >= N.  "> 300" safely clears the
    // UI frames (which top out around raw=27) and guarantees we're inside
    // a bona-fide 3D gameplay frame.  We also scan all 15 VS cbuffer slots
    // and dump 256 bytes each (enough for 2x 4x4 matrices + padding), and
    // cover the first 4 PS slots because deferred-renderer passes often
    // put the active camera in a PS cbuffer for lighting reconstruction.
    if (!m_gameplayCBuffersDumped && m_rawDrawCount > 300) {
      m_gameplayCBuffersDumped = true;
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      Logger::info(str::format(
          "[D3D11Rtx] First late-gameplay draw (count=", count,
          ", frameRawDraws=", m_rawDrawCount,
          ") -- dumping cbuffers to find Source's projection matrix layout:"));
      for (uint32_t slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot) {
        const auto& cb = vsCbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) {
          Logger::info(str::format(
              "[D3D11Rtx]   VS cb slot=", slot,
              " size=", cb.buffer->Desc()->ByteWidth,
              " mapPtr=NULL"));
          continue;
        }
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t dumpBytes = std::min<size_t>(256, bufSize > base ? bufSize - base : 0);
        Logger::info(str::format(
            "[D3D11Rtx]   VS cb slot=", slot,
            " size=", bufSize,
            " constOff=", base,
            " dumping=", dumpBytes, " bytes"));
        const float* f = reinterpret_cast<const float*>(ptr + base);
        for (size_t row = 0; row < dumpBytes / 16; ++row) {
          Logger::info(str::format(
              "[D3D11Rtx]     +", row * 16, ": ",
              f[row*4+0], ", ", f[row*4+1], ", ",
              f[row*4+2], ", ", f[row*4+3]));
        }
      }
      // Also dump the first 4 PS cbuffers — Source's deferred lighting
      // passes commonly stash the active scene camera in a PS cbuffer for
      // view-space reconstruction.
      const auto& psCbs = m_context->m_state.ps.constantBuffers;
      for (uint32_t slot = 0; slot < 4; ++slot) {
        const auto& cb = psCbs[slot];
        if (cb.buffer == nullptr) continue;
        const auto mapped = cb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (!ptr) continue;
        const size_t bufSize = cb.buffer->Desc()->ByteWidth;
        const size_t base    = static_cast<size_t>(cb.constantOffset) * 16;
        const size_t dumpBytes = std::min<size_t>(256, bufSize > base ? bufSize - base : 0);
        Logger::info(str::format(
            "[D3D11Rtx]   PS cb slot=", slot,
            " size=", bufSize,
            " dumping=", dumpBytes, " bytes"));
        const float* f = reinterpret_cast<const float*>(ptr + base);
        for (size_t row = 0; row < dumpBytes / 16; ++row) {
          Logger::info(str::format(
              "[D3D11Rtx]     +", row * 16, ": ",
              f[row*4+0], ", ", f[row*4+1], ", ",
              f[row*4+2], ", ", f[row*4+3]));
        }
      }
    }

    // Throttle: don't exceed the worker ring buffer capacity.
    // Beyond this point new futures would overwrite in-flight ones → corrupt hashes.
    if (m_drawCallID >= kMaxConcurrentDraws) {
      BumpFilter(FilterReason::Throttle);
      return;
    }

    // --- Cheap pre-filters: discard draws that cannot contribute to raytracing ---

    // Only triangle topologies are raytraceable. Skip points, lines, patch lists, etc.
    // This check is first: it costs a single comparison before any other state is read.
    const D3D11_PRIMITIVE_TOPOLOGY d3dTopology = m_context->m_state.ia.primitiveTopology;
    if (d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST &&
        d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP) {
      BumpFilter(FilterReason::NonTriTopology);
      return;
    }

    // Skip depth-only passes: no pixel shader means depth prepass or shadow map.
    // Most engines draw opaque geometry twice — once for depth prepass (PS == null)
    // and once for the color pass (PS != null) with the same vertices.
    if (m_context->m_state.ps.shader == nullptr) {
      BumpFilter(FilterReason::NoPixelShader);
      return;
    }

    // Skip draws with no color render target (shadow maps, depth-only, auxiliary passes).
    // NV-DXVK: Source-engine games (Titanfall 2) bind render targets to
    // non-zero slots — the old "check slot 0 only" heuristic rejected every
    // single menu draw because slot 0 was null even though slots 1–N were
    // bound.  Scan every MRT slot and keep the draw if any slot holds a
    // valid RTV, matching what FillMaterialData() below does.
    {
      bool anyRtvBound = false;
      for (uint32_t rt = 0; rt < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++rt) {
        if (m_context->m_state.om.renderTargetViews[rt].ptr() != nullptr) {
          anyRtvBound = true;
          break;
        }
      }
      if (!anyRtvBound) {
        BumpFilter(FilterReason::NoRenderTarget);
        return;
      }
    }

    // Skip trivially small draws (< 3 elements = 0 triangles).
    if (count < 3) {
      BumpFilter(FilterReason::CountTooSmall);
      return;
    }

    // Read actual depth/stencil state from the OM — don't hardcode.
    bool zEnable = true;
    bool zWriteEnable = true;
    bool stencilEnabled = false;
    D3D11DepthStencilState* dsState = m_context->m_state.om.dsState;
    if (dsState) {
      D3D11_DEPTH_STENCIL_DESC dsDesc;
      dsState->GetDesc(&dsDesc);
      zEnable         = dsDesc.DepthEnable != FALSE;
      zWriteEnable    = dsDesc.DepthWriteMask != D3D11_DEPTH_WRITE_MASK_ZERO;
      stencilEnabled  = dsDesc.StencilEnable != FALSE;
    }

    // Skip fullscreen quad / postprocess draws: depth disabled + 6 or fewer
    // elements (a fullscreen triangle or quad) + no depth write.
    // Only skip if BOTH depth test and write are off — some engines do
    // "depth off, write on" for sky or "depth on, write off" for decals.
    if (!zEnable && !zWriteEnable && count <= 6) {
      BumpFilter(FilterReason::FullscreenQuad);
      // Flag for native raster: these draws ARE UI/HUD/postprocess and
      // must rasterize natively once gameplay has made Remix active on
      // the frame; otherwise the menu/HUD never reaches the backbuffer.
      m_lastDrawFilteredAsUI = true;
      return;
    }

    D3D11InputLayout* layout = m_context->m_state.ia.inputLayout.ptr();
    if (!layout) {
      BumpFilter(FilterReason::NoInputLayout);
      m_lastDrawFilteredAsUI = true;
      return;
    }

    const auto& semantics = layout->GetRtxSemantics();

    if (semantics.empty()) {
      BumpFilter(FilterReason::NoSemantics);
      m_lastDrawFilteredAsUI = true;
      return;
    }

    const D3D11RtxSemantic* posSem = nullptr;
    const D3D11RtxSemantic* nrmSem = nullptr;
    const D3D11RtxSemantic* tcSem  = nullptr;
    const D3D11RtxSemantic* colSem = nullptr;
    const D3D11RtxSemantic* bwSem  = nullptr; // BLENDWEIGHT  — per-vertex bone weights
    const D3D11RtxSemantic* biSem  = nullptr; // BLENDINDICES — per-vertex bone indices

    static auto isTexcoordFmt = [](VkFormat f) {
      return f == VK_FORMAT_R32G32_SFLOAT        // 103 — standard 2-float UVs
          || f == VK_FORMAT_R16G16_SFLOAT         // 83  — half-float UVs
          || f == VK_FORMAT_R16G16_UNORM          // 77  — normalized 16-bit UVs (UE4, Unity HDRP)
          || f == VK_FORMAT_R16G16_SNORM          // 79  — signed normalized (some console ports)
          || f == VK_FORMAT_R8G8_UNORM;           // 16  — 8-bit packed UVs (mobile ports)
    };

    for (const auto& s : semantics) {
      if (s.perInstance) continue; // Skip per-instance data — only per-vertex geometry
      // Standard D3D semantic names
      if      (!posSem && std::strncmp(s.name, "POSITION", 8) == 0 && s.index == 0)
        posSem = &s;
      else if (!nrmSem && std::strncmp(s.name, "NORMAL",   6) == 0 && s.index == 0)
        nrmSem = &s;
      else if (!tcSem  && std::strncmp(s.name, "TEXCOORD", 8) == 0 && s.index == 0)
        tcSem  = &s;
      else if (!colSem && std::strncmp(s.name, "COLOR",    5) == 0 && s.index == 0)
        colSem = &s;
      else if (!bwSem  && std::strncmp(s.name, "BLENDWEIGHT",  11) == 0 && s.index == 0)
        bwSem  = &s;
      else if (!biSem  && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0)
        biSem  = &s;
    }

    // Fallback: accept TEXCOORD at any semantic index (some engines use
    // TEXCOORD1+ for primary UVs or start UV numbering at 1).
    if (!tcSem) {
      for (const auto& s : semantics) {
        if (std::strncmp(s.name, "TEXCOORD", 8) == 0) {
          tcSem = &s;
          break;
        }
      }
    }

    // Fallback: some engines use generic ATTRIBUTE semantics instead
    // of POSITION/NORMAL/TEXCOORD.  Identify by format heuristics.
    if (!posSem) {
      static auto isPositionFmt = [](VkFormat f) {
        return f == VK_FORMAT_R32G32B32_SFLOAT     // 106
            || f == VK_FORMAT_R32G32B32A32_SFLOAT  // 109
            || f == static_cast<VkFormat>(97);     // R16G16B16A16_SFLOAT
      };
      static auto isNormalFmt = [](VkFormat f) {
        return f == VK_FORMAT_R8G8B8A8_UNORM                     // 37
            || f == static_cast<VkFormat>(65); // A2B10G10R10_SNORM_PACK32
      };
      for (const auto& s : semantics) {
        if (s.perInstance) continue;
        if (std::strncmp(s.name, "ATTRIBUTE", 9) != 0) continue;
        if      (!posSem && isPositionFmt(s.format)) posSem = &s;
        else if (!tcSem  && isTexcoordFmt(s.format)) tcSem  = &s;
        else if (!nrmSem && isNormalFmt(s.format))   nrmSem = &s;
      }
    }

    // Format-based UV fallback: position was found by name but texcoord
    // wasn't (non-standard semantic name, custom engine, emulator port).
    // Scan remaining unmatched semantics for a 2-component float format.
    if (posSem && !tcSem) {
      for (const auto& s : semantics) {
        if (s.perInstance) continue;
        if (&s == posSem || &s == nrmSem || &s == colSem) continue;
        if (std::strncmp(s.name, "SV_", 3) == 0) continue;
        if (isTexcoordFmt(s.format)) {
          tcSem = &s;
          break;
        }
      }
    }

    if (!posSem) {
      BumpFilter(FilterReason::NoPosition);
      return;
    }

    // Log vertex layout once when texcoord is missing — diagnose UV issues
    if (!tcSem) {
      static uint32_t sNoTcLogCount = 0;
      if (sNoTcLogCount < 3) {
        ++sNoTcLogCount;
        Logger::info(str::format("[D3D11Rtx] SubmitDraw: no TEXCOORD found. Layout has ",
                                 semantics.size(), " semantics:"));
        for (const auto& s : semantics) {
          Logger::info(str::format("[D3D11Rtx]   name=", s.name, " idx=", s.index,
                                   " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
                                   " offset=", s.byteOffset));
        }
      }
    }

    // Skip 2D UI/HUD draws: if position is R32G32_SFLOAT it is in screen/clip space,
    // not world space, and cannot be raytraced.
    if (posSem->format == VK_FORMAT_R32G32_SFLOAT) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::Position2D)];
      return;
    }

    // NV-DXVK: Skip draws whose position format is not supported by
    // Remix's geometry interleaver.  Unsupported formats (e.g.
    // VK_FORMAT_R32G32_UINT = 101, which Source binds for compute-
    // style vertex readback passes) produce garbage positions that
    // build degenerate BLAS entries with NaN triangles → the GPU hangs
    // forever traversing them → TDR / VK_ERROR_DEVICE_LOST.  Only
    // accept formats the interleaver can actually convert to valid
    // float3 world-space positions.
    {
      const VkFormat pf = posSem->format;
      const bool supportedPosFmt =
          pf == VK_FORMAT_R32G32B32_SFLOAT       // 106 — standard 3-float
       || pf == VK_FORMAT_R32G32B32A32_SFLOAT    // 109 — 4-float (w ignored)
       || pf == VK_FORMAT_R16G16B16A16_SFLOAT    // 97  — half-float 4-component
       || pf == VK_FORMAT_R16G16B16_SFLOAT       // 90  — half-float 3-component
       || pf == VK_FORMAT_R32G32_UINT           // 101 — Source Engine 2 quantized positions
       ;
      if (!supportedPosFmt) {
        // NV-DXVK: Dump FULL input layout + vertex buffer info for R32G32_UINT draws
        static uint32_t sUnsupDiagCount = 0;
        if (pf == VK_FORMAT_R32G32_UINT && sUnsupDiagCount < 5) {
          ++sUnsupDiagCount;
          // Log all semantics in this layout
          Logger::info(str::format(
            "[D3D11Rtx] R32G32_UINT layout diag (", semantics.size(), " semantics, ",
            count, " verts):"));
          for (const auto& s : semantics) {
            Logger::info(str::format(
              "[D3D11Rtx]   elem: name=", s.name, " idx=", s.index,
              " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
              " off=", s.byteOffset, " perInst=", s.perInstance ? 1 : 0));
          }
          // Log all bound vertex buffers for slots 0-3
          for (uint32_t sl = 0; sl < 4; ++sl) {
            const auto& vb = m_context->m_state.ia.vertexBuffers[sl];
            if (vb.buffer != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   vbuf[", sl, "]: stride=", vb.stride,
                " offset=", vb.offset,
                " size=", vb.buffer->Desc()->ByteWidth,
                " usage=", uint32_t(vb.buffer->Desc()->Usage)));
            }
          }
          // Log VS shader info if available
          if (m_context->m_state.vs.shader != nullptr) {
            Logger::info(str::format(
              "[D3D11Rtx]   VS bound: yes"));
          }
          // Log VS cbuffers for transform inspection
          const auto& vsCbs = m_context->m_state.vs.constantBuffers;
          for (uint32_t sl = 0; sl < 8; ++sl) {
            if (vsCbs[sl].buffer != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   VS cb[", sl, "]: size=", vsCbs[sl].buffer->Desc()->ByteWidth,
                " off=", vsCbs[sl].constantOffset));
            }
          }
          // Log VS SRVs (structured buffers that might contain vertex data for GPU pulling)
          for (uint32_t sl = 0; sl < 8; ++sl) {
            const auto& srv = m_context->m_state.vs.shaderResources.views[sl];
            if (srv.ptr() != nullptr) {
              Logger::info(str::format(
                "[D3D11Rtx]   VS srv[", sl, "]: bound"));
            }
          }
        }
        BumpFilter(FilterReason::UnsupPosFmt);
        static uint32_t sUnsupPosLog = 0;
        if (sUnsupPosLog < 3) {
          ++sUnsupPosLog;
          Logger::warn(str::format(
              "[D3D11Rtx] Skipping draw with unsupported position format ",
              static_cast<uint32_t>(pf),
              " — only R32G32B32[A32]_SFLOAT and R16G16B16[A16]_SFLOAT "
              "are supported by the interleaver."));
        }
        return;
      }
    }

    auto makeVertexBuffer = [&](const D3D11RtxSemantic* sem) -> RasterBuffer {
      if (!sem)
        return RasterBuffer();
      const auto& vb = m_context->m_state.ia.vertexBuffers[sem->inputSlot];
      if (vb.buffer == nullptr)
        return RasterBuffer();
      DxvkBufferSlice slice = vb.buffer->GetBufferSlice(vb.offset);
      return RasterBuffer(slice, sem->byteOffset, vb.stride, sem->format);
    };

    RasterBuffer posBuffer = makeVertexBuffer(posSem);
    if (!posBuffer.defined()) {
      BumpFilter(FilterReason::NoPosBuffer);
      return;
    }

    // Detect NDC-space screen quads early (but defer the actual rejection until
    // after ExtractTransforms so the VP cache gets populated from these draws).
    bool isNdcScreenQuad = false;
    if (count <= 6 && posSem->format == VK_FORMAT_R32G32B32_SFLOAT) {
      const float* p = reinterpret_cast<const float*>(
        posBuffer.mapPtr(posBuffer.offsetFromSlice()));
      if (p && std::abs(p[0]) <= 1.5f && std::abs(p[1]) <= 1.5f && std::abs(p[2]) <= 1.0f)
        isNdcScreenQuad = true;
    }

    // Normal buffer: only submit if enabled and the interleaver can convert.
    // Supported: R16G16_SFLOAT(83), R32G32_SFLOAT(103), R32G32B32_SFLOAT(106),
    // R32G32B32A32_SFLOAT(109), R8G8B8A8_UNORM(37), A2B10G10R10_SNORM(65).
    // D3D11 normals are often R16G16B16A16_SFLOAT(97) or R16G16B16A16_SNORM(98)
    // which the interleaver rejects.  Remix regenerates normals when absent.
    RasterBuffer nrmBuffer;
    if (nrmSem && RtxOptions::useInputAssemblerNormals()) {
      VkFormat nf = nrmSem->format;
      if (nf == VK_FORMAT_R8G8B8A8_UNORM
       || nf == VK_FORMAT_R32G32B32_SFLOAT
       || nf == VK_FORMAT_R32G32B32A32_SFLOAT
       || nf == VK_FORMAT_R32G32_SFLOAT
       || nf == VK_FORMAT_R16G16_SFLOAT
       || nf == static_cast<VkFormat>(65)) {  // A2B10G10R10_SNORM_PACK32
        nrmBuffer = makeVertexBuffer(nrmSem);
      }
    }
    RasterBuffer tcBuffer  = makeVertexBuffer(tcSem);

    // Color0: the interleaver converts BGRA and RGBA packed-byte formats.
    // Both B8G8R8A8_UNORM (D3D9 D3DCOLOR) and R8G8B8A8_UNORM (D3D11) are
    // supported — the interleaver swaps R/B for RGBA.  Float vertex color
    // formats are not supported; Remix defaults to white when color0 is absent.
    RasterBuffer colBuffer;
    if (colSem && (colSem->format == VK_FORMAT_B8G8R8A8_UNORM
                || colSem->format == VK_FORMAT_R8G8B8A8_UNORM)) {
      colBuffer = makeVertexBuffer(colSem);
    }

    // NV-DXVK start: Per-vertex skinning buffers (BLENDWEIGHT + BLENDINDICES)
    RasterBuffer bwBuffer  = makeVertexBuffer(bwSem);
    RasterBuffer biBuffer  = makeVertexBuffer(biSem);
    // NV-DXVK end

    RasterBuffer idxBuffer;
    if (indexed) {
      const auto& ib = m_context->m_state.ia.indexBuffer;
      if (ib.buffer == nullptr) {
        BumpFilter(FilterReason::NoIndexBuffer);
        return;
      }
      VkIndexType idxType = (ib.format == DXGI_FORMAT_R32_UINT)
                          ? VK_INDEX_TYPE_UINT32
                          : VK_INDEX_TYPE_UINT16;
      uint32_t idxStride = (idxType == VK_INDEX_TYPE_UINT32) ? 4 : 2;
      // NV-DXVK: Bake startIndex into the slice offset. The BLAS builder and
      // the cacheIndexDataOnGPU copy both read from `slice.offset + 0`, not
      // `slice.offset + startIndex*stride`. Without this, draws with startIndex>0
      // get the wrong index range cached → BLAS sees stale indices from the
      // top of the IB → OOB vertex reads → MMU fault.
      idxBuffer = RasterBuffer(
        ib.buffer->GetBufferSlice(ib.offset + size_t(start) * idxStride),
        0, idxStride, idxType);
    }

    VkPrimitiveTopology vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (m_context->m_state.ia.primitiveTopology) {
      case D3D11_PRIMITIVE_TOPOLOGY_POINTLIST:     vkTopology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;     break;
      case D3D11_PRIMITIVE_TOPOLOGY_LINELIST:      vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      break;
      case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP:     vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     break;
      case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST:  vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  break;
      case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
      default: break;
    }

    RasterGeometry geo;
    geo.topology       = vkTopology;
    geo.frontFace      = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    geo.positionBuffer = posBuffer;
    geo.normalBuffer   = nrmBuffer;
    geo.texcoordBuffer = tcBuffer;
    geo.color0Buffer   = colBuffer;
    geo.indexBuffer    = idxBuffer;
    geo.indexCount     = indexed ? count : 0;

    // NV-DXVK: Snapshot index bytes NOW from the game's currently mapped slice.
    // Runs on the thread that owns the D3D11 state (deferred context replay on
    // CS thread, or immediate context) before any subsequent Map/DISCARD can
    // rename the physical slice. Without this, the deferred cacheIndexDataOnGPU
    // copy (later on CS thread) reads the renamed slice → garbage indices →
    // BLAS build OOB → MMU fault → TDR.
    //
    // Only snapshot DYNAMIC buffers (the only ones subject to renaming).
    // Static/immutable buffers have stable physical addresses — zero overhead.
    if (indexed) {
      static uint32_t sIdxSnapStats[4] = {0, 0, 0, 0};  // dyn_snapped, dyn_no_mapptr, static_skipped, null_buf
      static uint32_t sIdxSnapLog = 0;
      const auto& ib2 = m_context->m_state.ia.indexBuffer;
      bool snapped = false;
      if (ib2.buffer == nullptr) {
        ++sIdxSnapStats[3];
      } else if (ib2.buffer->Desc()->Usage != D3D11_USAGE_DYNAMIC) {
        ++sIdxSnapStats[2];
      } else {
        const auto mapped = ib2.buffer->GetMappedSlice();
        if (mapped.mapPtr == nullptr) {
          ++sIdxSnapStats[1];
        } else {
          const uint32_t idxStride2 = (ib2.format == DXGI_FORMAT_R32_UINT) ? 4u : 2u;
          const size_t snapLen = size_t(count) * idxStride2;
          // Draw reads indices [start, start+count) — snapshot must start at
          // start*stride, not 0, or cacheIndexDataOnGPU uploads the wrong range.
          const size_t snapOff = size_t(ib2.offset) + size_t(start) * idxStride2;
          const size_t bufLen  = ib2.buffer->Desc()->ByteWidth;
          if (snapLen > 0 && snapOff + snapLen <= bufLen) {
            geo.indexDataSnapshot = std::make_shared<std::vector<uint8_t>>(snapLen);
            std::memcpy(geo.indexDataSnapshot->data(),
                        reinterpret_cast<const uint8_t*>(mapped.mapPtr) + snapOff,
                        snapLen);
            ++sIdxSnapStats[0];
            snapped = true;
          }
        }
      }
      // Log first 30 draws + stats every 500 draws
      if (sIdxSnapLog < 30 || (sIdxSnapLog % 500) == 0) {
        Logger::info(str::format("[IDX-SNAP] snap=", snapped ? 1 : 0,
          " count=", count,
          " usage=", (ib2.buffer != nullptr ? uint32_t(ib2.buffer->Desc()->Usage) : 0u),
          " off=", ib2.offset,
          " stats: dynSnap=", sIdxSnapStats[0],
          " dynNoMap=", sIdxSnapStats[1],
          " static=", sIdxSnapStats[2],
          " null=", sIdxSnapStats[3]));
      }
      ++sIdxSnapLog;
    }

    // NV-DXVK start: Per-vertex skinning — populate blend buffers and bone count
    if (bwBuffer.defined() && biBuffer.defined()) {
      geo.blendWeightBuffer  = bwBuffer;
      geo.blendIndicesBuffer = biBuffer;
      // Derive bones-per-vertex from the blend weight format:
      // Each explicit float weight implies one bone; the last bone's weight is
      // implicit (1 - sum).  So 1 float = 2 bones, 2 floats = 3, etc.
      switch (bwSem->format) {
        case VK_FORMAT_R32_SFLOAT:                geo.numBonesPerVertex = 2; break;
        case VK_FORMAT_R32G32_SFLOAT:             geo.numBonesPerVertex = 3; break;
        case VK_FORMAT_R32G32B32_SFLOAT:          geo.numBonesPerVertex = 4; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:       geo.numBonesPerVertex = 4; break;
        default:                                  geo.numBonesPerVertex = 0; break;
      }
    }
    // NV-DXVK end

    // NV-DXVK start: Diagnostic — dump first N unique input layouts
    {
      static uint32_t sLayoutLog = 0;
      static uintptr_t sLastLayout = 0;
      uintptr_t layoutAddr = reinterpret_cast<uintptr_t>(m_context->m_state.ia.inputLayout.ptr());
      if (layoutAddr != sLastLayout && sLayoutLog < 20) {
        sLastLayout = layoutAddr;
        ++sLayoutLog;
        Logger::info(str::format("[D3D11Rtx] Layout #", sLayoutLog,
                                 " (", semantics.size(), " semantics):"));
        for (const auto& s : semantics) {
          Logger::info(str::format("[D3D11Rtx]   name=", s.name,
            " idx=", s.index, " fmt=", uint32_t(s.format),
            " slot=", s.inputSlot, " off=", s.byteOffset,
            " inst=", s.perInstance ? 1 : 0));
        }
      }
    }
    // NV-DXVK end

    // NV-DXVK: Track bone buffer and attach bone data for GPU instancing.
    // For R32G32_UINT positions AND for instanced bone draws (m_attachBoneBuffers),
    // attach a SRV-backed transform buffer + per-vertex/per-instance index source.
    //
    // Two TF2 patterns share most of the plumbing:
    //   (A) Skinned characters (g_boneMatrix at t30, stride=48):
    //       - Per-instance R16G16B16A16_UINT semantic (bone indices)
    //       - One bone matrix per draw (use index from semantic)
    //   (B) BSP / batched static props (g_modelInst at t31, stride=208):
    //       - Per-vertex COLOR1 R32G32B32A32_UINT semantic (instance indices)
    //       - Each vertex picks its own transform via cb.bonePerVertex path
    // DEBUG: log posSem format + SRV slot occupancy for first N draws of each
    // unique VS, so we can see why the BSP path doesn't fire.
    {
      static std::unordered_set<uintptr_t> sPosFmtLogged;
      auto vsPtr = m_context->m_state.vs.shader;
      uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
      if (key && sPosFmtLogged.size() < 40 && sPosFmtLogged.insert(key).second) {
        const auto& srvs = m_context->m_state.vs.shaderResources.views;
        std::string vsHash = "?";
        if (vsPtr->GetCommonShader() != nullptr) {
          auto& s = vsPtr->GetCommonShader()->GetShader();
          if (s != nullptr) vsHash = s->getShaderKey().toString();
        }
        Logger::info(str::format(
          "[D3D11Rtx] PosFmtProbe VS=", vsHash,
          " posFmt=", posSem ? uint32_t(posSem->format) : 0,
          " posPerInst=", posSem ? (posSem->perInstance ? 1 : 0) : 0,
          " m_attachBoneBuffers=", m_attachBoneBuffers ? 1 : 0,
          " t30=", srvs[30].ptr() ? 1 : 0,
          " t31=", srvs[31].ptr() ? 1 : 0,
          " bspGuard=", (posSem && posSem->format == VK_FORMAT_R32G32_UINT) || m_attachBoneBuffers ? 1 : 0));
      }
    }
    if (posSem->format == VK_FORMAT_R32G32_UINT || m_attachBoneBuffers) {
      // NV-DXVK: ask the VS RDEF which resource it actually declares — both
      // t30 and t31 may be bound by app state, but each VS only reads ONE.
      // Preferring t30 by default mis-routed BSP draws (which read g_modelInst
      // at t31) into the bone-skinning path and they ended up at origin.
      uint32_t modelInstSlot = UINT32_MAX;
      uint32_t boneMatrixSlot = UINT32_MAX;
      {
        auto vsPtr = m_context->m_state.vs.shader;
        if (vsPtr != nullptr && vsPtr->GetCommonShader() != nullptr) {
          const D3D11CommonShader* common = vsPtr->GetCommonShader();
          modelInstSlot  = common->FindResourceSlot("g_modelInst");
          boneMatrixSlot = common->FindResourceSlot("g_boneMatrix");
        }
        // DEBUG: log RDEF resource lookup result per unique VS
        static std::unordered_set<uintptr_t> sRdefLookupLogged;
        uintptr_t key = (vsPtr != nullptr) ? reinterpret_cast<uintptr_t>(vsPtr.ptr()) : 0;
        if (key && sRdefLookupLogged.size() < 30 && sRdefLookupLogged.insert(key).second) {
          std::string vsHash = "?";
          if (vsPtr->GetCommonShader() != nullptr) {
            auto& s = vsPtr->GetCommonShader()->GetShader();
            if (s != nullptr) vsHash = s->getShaderKey().toString();
          }
          Logger::info(str::format(
            "[D3D11Rtx] RdefLookup VS=", vsHash,
            " g_modelInst=", modelInstSlot,
            " g_boneMatrix=", boneMatrixSlot));
        }
      }
      // BSP / batched static props use g_modelInst when present. Otherwise
      // fall back to g_boneMatrix (skinned characters). Final fallback: scan
      // both slots blindly (covers shaders without RDEF).
      ID3D11ShaderResourceView* xformSrv = nullptr;
      bool isModelInst = false;
      if (modelInstSlot != UINT32_MAX
          && modelInstSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
        xformSrv = m_context->m_state.vs.shaderResources.views[modelInstSlot].ptr();
        if (xformSrv) isModelInst = true;
      }
      if (!xformSrv && boneMatrixSlot != UINT32_MAX
          && boneMatrixSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
        xformSrv = m_context->m_state.vs.shaderResources.views[boneMatrixSlot].ptr();
      }
      if (!xformSrv) {
        // NV-DXVK semantic-based blind probe.
        // RDEF missed both g_modelInst and g_boneMatrix. Use the VS's declared
        // input semantics to classify the shader and attach only when the
        // semantics prove t31 is needed:
        //   - BLENDINDICES per-vertex → skinned character; t31 = bone palette.
        //     Attach as bone buffer (xformSrv path below with !isModelInst).
        //   - COLOR1/I R16G16B16A16_UINT per-instance → instanced BSP/prop;
        //     t31 = g_modelInst. Attach with isModelInst=true.
        //   - Neither → static cb3-only mesh (e.g. VS_6e3e6f28). Skip — the
        //     cb3 RDEF path upstream already wrote the correct objectToWorld.
        //     Attaching t30/t31 here would route vertices through skinning or
        //     per-instance fanout and warp the mesh around the camera.
        auto* ilProbe = m_context->m_state.ia.inputLayout.ptr();
        bool semBlendIdx = false;
        bool semPerInstIdx = false;
        if (ilProbe != nullptr) {
          for (const auto& s : ilProbe->GetRtxSemantics()) {
            if (!s.perInstance && std::strncmp(s.name, "BLENDINDICES", 12) == 0 && s.index == 0)
              semBlendIdx = true;
            if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT)
              semPerInstIdx = true;
          }
        }
        constexpr uint32_t kT31Slot = 31;
        if ((semBlendIdx || semPerInstIdx) && kT31Slot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
          xformSrv = m_context->m_state.vs.shaderResources.views[kT31Slot].ptr();
          if (xformSrv && semPerInstIdx) isModelInst = true;
        }
        auto vsPtr = m_context->m_state.vs.shader;
        if (vsPtr != nullptr) {
          static std::unordered_set<uintptr_t> sBlindClassifyLogged;
          uintptr_t key = reinterpret_cast<uintptr_t>(vsPtr.ptr());
          if (sBlindClassifyLogged.insert(key).second) {
            std::string vsHash = "?";
            if (vsPtr->GetCommonShader() != nullptr) {
              auto& s = vsPtr->GetCommonShader()->GetShader();
              if (s != nullptr) vsHash = s->getShaderKey().toString();
            }
            const char* cls = semBlendIdx ? "skinned_char_t31_bone_palette"
                            : semPerInstIdx ? "instanced_bsp_t31_modelInst"
                            : "static_mesh_cb3_owns_transform_skip_attach";
            Logger::info(str::format(
              "[D3D11Rtx] BLIND-PROBE classify VS=", vsHash,
              " class=", cls,
              " attached=", xformSrv ? 1 : 0,
              " isModelInst=", isModelInst ? 1 : 0));
          }
        }
      }
      if (xformSrv && !isModelInst) {
        // Legacy skinning path only. For BSP / batched-prop draws (isModelInst)
        // we do NOT attach a bone matrix here — the per-instance fanout above
        // already creates one TLAS instance per modelInst row with the correct
        // transform. Letting the interleave shader also bone-multiply would
        // double-apply the matrix and put geometry at sqr(transform) * raw_pos.
        Com<ID3D11Resource> xformRes;
        xformSrv->GetResource(&xformRes);
        auto* xformBuf = static_cast<D3D11Buffer*>(xformRes.ptr());
        if (xformBuf) {
          m_lastBoneBuffer = xformBuf;
          const uint32_t matrixStride = 48u;
          geo.boneMatrixBuffer = RasterBuffer(
            xformBuf->GetBufferSlice(), 0, matrixStride, VK_FORMAT_UNDEFINED);
          geo.boneMatrixStrideBytes = matrixStride;
        }
      } else if (xformSrv && isModelInst) {
        // BSP-path: log once per unique buffer so we know fanout activated.
        Com<ID3D11Resource> xformRes; xformSrv->GetResource(&xformRes);
        auto* xformBuf = static_cast<D3D11Buffer*>(xformRes.ptr());
        static uint32_t sBspLogCount = 0;
        if (xformBuf && sBspLogCount < 20) {
          ++sBspLogCount;
          Logger::info(str::format(
            "[D3D11Rtx] BSP-fanout-path (t31): bufSize=",
            xformBuf->Desc()->ByteWidth, " (no boneMatrixBuffer attached)"));
        }
      }
      // DEBUG: dump every semantic for the first N BSP-path draws so we can see
      // what the per-vertex/per-instance index format actually is.
      if (isModelInst) {
        static uint32_t sBspSemDump = 0;
        if (sBspSemDump < 6) {
          ++sBspSemDump;
          for (const auto& s : semantics) {
            Logger::info(str::format(
              "[D3D11Rtx] BSP semantic dump: name=", s.name, " idx=", s.index,
              " fmt=", uint32_t(s.format), " slot=", s.inputSlot,
              " byteOff=", s.byteOffset,
              " perInst=", s.perInstance ? 1 : 0));
          }
        }
      }
      // NV-DXVK (TF2 skinned characters): detect the weighted-skinning
      // fingerprint — POSITION0/V + BLENDINDICES0/V:fmt41 (RGBA8_UINT) +
      // BLENDWEIGHT0/V:fmt82 (R16G16 UNORM) + t30 SRV (g_boneMatrix, stride
      // 48). Bind t30 as matrix buffer, BLENDINDICES VB as index buffer,
      // BLENDWEIGHT VB as weight buffer. Interleaver does Σ w_i bone[idx_i].
      bool didSkinnedChar = false;
      if (biSem != nullptr && bwSem != nullptr
          && biSem->format == VK_FORMAT_R8G8B8A8_UINT
          && !biSem->perInstance) {
        // Use t30 directly (g_boneMatrix). xformSrv above may have picked t31
        // for isModelInst=false non-instanced BSP — override to t30 here.
        ID3D11ShaderResourceView* boneSrv = nullptr;
        {
          uint32_t boneSlot = UINT32_MAX;
          auto vsPtr2 = m_context->m_state.vs.shader;
          if (vsPtr2 != nullptr && vsPtr2->GetCommonShader() != nullptr)
            boneSlot = vsPtr2->GetCommonShader()->FindResourceSlot("g_boneMatrix");
          if (boneSlot == UINT32_MAX) boneSlot = 30u;
          if (boneSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
            boneSrv = m_context->m_state.vs.shaderResources.views[boneSlot].ptr();
        }
        if (boneSrv && biBuffer.defined() && bwBuffer.defined()) {
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          if (boneBuf) {
            m_lastBoneBuffer = boneBuf;
            geo.boneMatrixBuffer = RasterBuffer(
              boneBuf->GetBufferSlice(), 0, 48u, VK_FORMAT_UNDEFINED);
            geo.boneMatrixStrideBytes = 48u;
            geo.boneIndexBuffer = biBuffer;
            geo.boneIndexStrideBytes = biBuffer.stride();
            geo.boneIndexMask = 0xFFu;  // per-byte index within packed RGBA8
            geo.boneIndexComponentCount = 4u;
            geo.bonePerVertex = true;
            geo.boneWeightBuffer = bwBuffer;
            didSkinnedChar = true;
            // NV-DXVK: bone matrices are in camera-relative space (TF2 VS
            // does `cb2.c_cameraRelativeToClip * t30[idx] * local`). After
            // weighted skinning the interleaver produces camera-relative
            // positions, so objectToWorld must translate by +fanoutCam to
            // land them in absolute world. We can't write dcs here because
            // dcs isn't constructed yet; flip a flag and apply after dcs.
            m_skinnedCharNeedsCamOffset = true;
            static uint32_t sSkinLog = 0;
            if (sSkinLog < 20) {
              ++sSkinLog;
              Logger::info(str::format(
                "[D3D11Rtx] TF2 skinned char bound: t30buf=",
                boneBuf->Desc()->ByteWidth,
                " biStride=", biBuffer.stride(),
                " bwStride=", bwBuffer.stride(),
                " biOff=", biBuffer.offsetFromSlice(),
                " bwOff=", bwBuffer.offsetFromSlice()));
            }
          }
        }
      }

      // Find the per-vertex/per-instance index semantic. Prefer R32G32B32A32_UINT
      // (BSP), accept R16G16B16A16_UINT (legacy bone). For BSP the semantic is
      // typically per-VERTEX (inst=0); for bone-draws it's per-instance (inst=1).
      // Slice starts at vb.offset + s.byteOffset so the interleave shader can
      // index by vertex without needing semantic-internal offset awareness.
      if (!didSkinnedChar)
      for (const auto& s : semantics) {
        if (s.format == VK_FORMAT_R32G32B32A32_UINT) {
          const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
          if (vb.buffer != nullptr) {
            geo.boneIndexBuffer = RasterBuffer(
              vb.buffer->GetBufferSlice(vb.offset + s.byteOffset),
              0, vb.stride, s.format);
            geo.bonePerVertex       = !s.perInstance;   // per-vertex for BSP
            geo.boneIndexStrideBytes = vb.stride;        // typically 16 (4x uint32)
            geo.boneIndexMask        = 0xFFFFFFFFu;      // full 32-bit index
            // DEBUG
            static uint32_t sBspIdxLog = 0;
            if (sBspIdxLog < 20) {
              ++sBspIdxLog;
              Logger::info(str::format(
                "[D3D11Rtx] BSP idx semantic: name=", s.name,
                " perInst=", s.perInstance ? 1 : 0,
                " stride=", vb.stride, " byteOff=", s.byteOffset,
                " bonePerVertex=", geo.bonePerVertex ? 1 : 0));
            }
          }
          break;
        }
        if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
          const auto& vb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
          if (vb.buffer != nullptr) {
            geo.boneIndexBuffer = RasterBuffer(
              vb.buffer->GetBufferSlice(vb.offset + s.byteOffset),
              0, vb.stride, s.format);
            geo.bonePerVertex       = false;             // legacy: one bone/draw
            geo.boneIndexStrideBytes = vb.stride;        // typically 8 (4x uint16)
            geo.boneIndexMask        = 0xFFFFu;
          }
          break;
        }
      }
      geo.boneInstanceIndex = m_currentInstanceIndex;
    }

    // Read cull mode from the immutable ID3D11RasterizerState object.
    // Default: no culling (safe fallback when no state is bound).
    geo.cullMode = VK_CULL_MODE_NONE;
    D3D11RasterizerState* rsState = m_context->m_state.rs.state;
    if (rsState) {
      const auto* rsDesc = rsState->Desc();
      switch (rsDesc->CullMode) {
        case D3D11_CULL_NONE:  geo.cullMode = VK_CULL_MODE_NONE;      break;
        case D3D11_CULL_FRONT: geo.cullMode = VK_CULL_MODE_FRONT_BIT; break;
        case D3D11_CULL_BACK:  geo.cullMode = VK_CULL_MODE_BACK_BIT;  break;
      }
      geo.frontFace = rsDesc->FrontCounterClockwise
        ? VK_FRONT_FACE_COUNTER_CLOCKWISE
        : VK_FRONT_FACE_CLOCKWISE;
    }

    // Compute vertex count — must cover the highest vertex index accessed by
    // this draw so Remix doesn't read out of bounds when building BLAS.
    // geo.vertexCount is the buffer capacity; the hash uses a tighter subrange.
    const uint32_t maxVBVertices = posBuffer.stride() > 0
      ? static_cast<uint32_t>(posBuffer.length() / posBuffer.stride())
      : count;
    uint32_t drawVertexCount;
    uint32_t hashStart, hashCount;
    if (!indexed) {
      // Non-indexed Draw(count, start): vertices [start, start+count) accessed.
      drawVertexCount = std::min(start + count, maxVBVertices);
      hashStart = std::min(start, maxVBVertices);
      hashCount = std::min(count, maxVBVertices - hashStart);
    } else {
      // Indexed DrawIndexed(indexCount, startIndex, base): vertex = index + base.
      // The OLD comment said we couldn't know max(index) without scanning the IB,
      // so it fell back to base + indexCount — which is wrong. BSP/static geometry
      // shares a large vertex buffer across many draws, and draws with few indices
      // frequently reference vertices far above `base + indexCount`. That caused
      // Remix to report vertexCount far smaller than the real range, the BLAS
      // builder saw "valid" primitiveCount but indices ≥ maxVertex → MMU fault
      // reading beyond the vertex buffer → VK_ERROR_DEVICE_LOST on frame 3+.
      //
      // FIX: scan the index buffer CPU-side to find the actual max index.
      // For DYNAMIC source, read mapped slice directly. For STATIC source,
      // try mapPtr (often host-visible for small buffers). If scan isn't
      // possible, fall back to full buffer capacity (safe: over-reports).
      const uint32_t baseU = static_cast<uint32_t>(std::max(base, 0));
      uint32_t maxIdxSeen = 0;
      bool scanned = false;
      if (indexed) {
        const auto& ib = m_context->m_state.ia.indexBuffer;
        if (ib.buffer != nullptr) {
          const uint32_t idxStride = (ib.format == DXGI_FORMAT_R32_UINT) ? 4u : 2u;
          const void* src = nullptr;
          // DYNAMIC: use current mapped slice (race-safe on our thread).
          if (ib.buffer->Desc()->Usage == D3D11_USAGE_DYNAMIC) {
            const auto mapped = ib.buffer->GetMappedSlice();
            src = mapped.mapPtr;
          }
          // STATIC: some immutable buffers are host-visible for staging.
          if (src == nullptr) {
            src = ib.buffer->GetBuffer()->mapPtr(0);
          }
          if (src != nullptr) {
            const size_t bufSize = ib.buffer->Desc()->ByteWidth;
            const size_t startOff = ib.offset + size_t(start) * idxStride;
            const size_t readLen = size_t(count) * idxStride;
            if (startOff + readLen <= bufSize) {
              const uint8_t* p = reinterpret_cast<const uint8_t*>(src) + startOff;
              if (idxStride == 2) {
                const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
                for (uint32_t i = 0; i < count; ++i)
                  if (q[i] > maxIdxSeen) maxIdxSeen = q[i];
              } else {
                const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
                for (uint32_t i = 0; i < count; ++i)
                  if (q[i] > maxIdxSeen) maxIdxSeen = q[i];
              }
              scanned = true;
            }
          }
        }
      }
      if (scanned) {
        // BLAS builder needs maxVertex = highest index + 1 (inclusive range).
        // Add base offset since BLAS input vertices are [base, base + maxVtx].
        drawVertexCount = std::min(baseU + maxIdxSeen + 1u, maxVBVertices);
        hashStart = std::min(baseU, maxVBVertices);
        hashCount = std::min(maxIdxSeen + 1u, maxVBVertices - hashStart);
      } else {
        // Couldn't scan — fall back to the FULL vertex buffer capacity
        // (over-reports but safe: BLAS builder can't read past buffer).
        drawVertexCount = maxVBVertices;
        hashStart = std::min(baseU, maxVBVertices);
        hashCount = std::min(count, maxVBVertices - hashStart);
        // Log fallbacks so we know if static idx buffers aren't mappable.
        static uint32_t sFallbackLog = 0, sFallbackCount = 0;
        ++sFallbackCount;
        if (sFallbackLog < 20 || (sFallbackCount % 500) == 0) {
          ++sFallbackLog;
          const auto& ib = m_context->m_state.ia.indexBuffer;
          Logger::warn(str::format("[IDX-SCAN-FALLBACK] #", sFallbackCount,
            " idxBuf=0x", std::hex,
            (uintptr_t)(ib.buffer != nullptr ? ib.buffer.ptr() : nullptr),
            std::dec,
            " usage=", (ib.buffer != nullptr ? uint32_t(ib.buffer->Desc()->Usage) : 0u),
            " count=", count, " start=", start, " base=", base,
            " maxVBVertices=", maxVBVertices,
            " → drawVertexCount=full buffer (over-reporting)"));
        }
      }
    }
    if (drawVertexCount == 0)
      drawVertexCount = count;
    if (hashCount == 0)
      hashCount = std::min(count, maxVBVertices);
    geo.vertexCount = drawVertexCount;

    geo.futureGeometryHashes = ComputeGeometryHashes(geo, drawVertexCount,
                                                     hashStart, hashCount);
    if (!geo.futureGeometryHashes.valid()) {
      BumpFilter(FilterReason::HashFailed);
      return;
    }

    DrawCallState dcs;
    dcs.geometryData     = geo;
    dcs.transformData    = ExtractTransforms();

    // NV-DXVK (TF2 skinned chars): if the earlier RasterGeometry setup flagged
    // this draw as a skinned character (BLENDINDICES+BLENDWEIGHT+t30), set
    // objectToWorld = identity. Verified via DXBC disassembly of
    // VS_ef94e6c7fcc3c144: the bone matrices in t30 store ABSOLUTE world
    // transforms (not camera-relative). The game's VS applies skinning then
    // subtracts c_cameraOrigin explicitly to get camera-relative positions
    // before the clip transform. Since the interleaver's weighted skinning
    // already produces world-space positions, objectToWorld should be
    // identity — adding fanoutCam would double the camera offset.
    if (m_skinnedCharNeedsCamOffset) {
      // dcs.transformData.objectToWorld = Matrix4();  // identity
      // dcs.transformData.objectToView = dcs.transformData.worldToView;  // o2v = w2v * I
      // m_lastO2wPathId = 11;  // skinned char: identity (BLAS is world)
      BumpFilter(FilterReason::UIFallback);
      return;  // do NOT set m_lastDrawFilteredAsUI — don't rasterize
    }

    // NV-DXVK: TLAS coherence filter + matrix finiteness guard.
    // Fires for BOTH non-instanced (OnDraw/OnDrawIndexed → SubmitDraw) and
    // instanced (OnDrawInstanced/OnDrawIndexedInstanced → SubmitInstancedDraw
    // → SubmitDraw) paths since everything funnels here.
    //
    // (1) TLAS coherence: reject draws whose c_cameraOrigin doesn't match the
    //     Main camera's world position within kEpsilon. Different cameras mean
    //     different BLAS placements → TLAS mixes coord spaces → pathological
    //     bounds → ray traversal can run effectively forever → GPU TDR.
    // (2) Finiteness guard: reject draws whose objectToWorld matrix has any
    //     non-finite component or absurd translation magnitude. Observed in TF2
    //     where game cbuffers occasionally contain NaN (VS s2[10]=(-nan,...)).
    {
      const auto& m = dcs.transformData.objectToWorld;
      bool badMatrix = false;
      for (int r = 0; r < 4 && !badMatrix; ++r) {
        for (int c = 0; c < 4 && !badMatrix; ++c) {
          if (!std::isfinite(m[r][c])) badMatrix = true;
        }
      }
      constexpr float kMaxComponentMagnitude = 1.0e7f; // TF2 coords are ~1e4
      for (int r = 0; r < 4 && !badMatrix; ++r) {
        if (std::abs(m[3][r]) > kMaxComponentMagnitude) badMatrix = true;
      }
      if (badMatrix) {
        static uint32_t sBadMatLog = 0;
        if (sBadMatLog < 20) {
          ++sBadMatLog;
          Logger::err(str::format(
            "[TLAS-FILTER] reject draw=", m_drawCallID,
            " non-finite/absurd o2w: T=(", m[3][0], ",", m[3][1], ",", m[3][2], ")",
            " diag=(", m[0][0], ",", m[1][1], ",", m[2][2], ")"));
        }
        BumpFilter(FilterReason::FullscreenQuad);
        return;
      }
    }

    // The filter compares EVERY draw's WORLD-SPACE camera position against
    // Main's. Per-draw position is derived as inverse(worldToView)[3].xyz()
    // — same construction RtCamera::getPosition uses internally — so both
    // sides share an identical coordinate convention. Comparing raw
    // worldToView[3] columns directly fails because RtCamera's matCache
    // sometimes overwrites WorldToView with identity (depending on
    // freeCameraViewRelative()), but getPosition() always returns a valid
    // world position derived from the original view-to-world.
    Vector3 drawCamPos;
    {
      const Matrix4 v2w = inverse(dcs.transformData.worldToView);
      drawCamPos = Vector3(v2w[3][0], v2w[3][1], v2w[3][2]);
    }

    if (m_context->m_device != nullptr) {
      auto& sceneMgr = m_context->m_device->getCommon()->getSceneManager();
      auto& camMgr = sceneMgr.getCameraManager();
      auto& mainCam = camMgr.getCamera(CameraType::Main);
      const uint32_t frameId = m_context->m_device->getCurrentFrameId();
      // Only trust Main's position if the CLASSIFIER (not safety net) latched
      // it in the last few frames. Safety-net Main is whatever ExtractTransforms
      // produced — often identity/(-1,-1,-1) during menus/cinematics — and
      // would otherwise cause us to reject every real world draw.
      const bool classifierLatched = camMgr.isMainSetByClassifier();
      const uint32_t lastClassifierFrame = camMgr.getMainClassifierFrameId();
      constexpr uint32_t kMaxStaleFrames = 5; // allow last ~5 frames after latch
      const bool mainRecentlyLatched =
        classifierLatched
        && (frameId <= lastClassifierFrame
            || (frameId - lastClassifierFrame) <= kMaxStaleFrames);
      const bool mainEverValid = mainRecentlyLatched;

      // Per-frame stats — reset when we see the drawCallID counter roll over.
      static uint32_t s_tlasFilterFrame = UINT32_MAX;
      static uint32_t s_tlasAccept = 0;
      static uint32_t s_tlasReject = 0;
      static uint32_t s_tlasNoMain = 0;
      static uint32_t s_tlasPrevID = UINT32_MAX;
      if (m_drawCallID == 0 || m_drawCallID < s_tlasPrevID) {
        if (s_tlasFilterFrame != UINT32_MAX
            && (s_tlasAccept + s_tlasReject + s_tlasNoMain) > 0
            && s_tlasFilterFrame < 600) {
          Logger::info(str::format(
            "[TLAS-FILTER] frame=", s_tlasFilterFrame,
            " accept=", s_tlasAccept,
            " reject=", s_tlasReject,
            " noMain=", s_tlasNoMain));
        }
        s_tlasFilterFrame = (s_tlasFilterFrame == UINT32_MAX) ? 0 : s_tlasFilterFrame + 1;
        s_tlasAccept = 0;
        s_tlasReject = 0;
        s_tlasNoMain = 0;
      }
      s_tlasPrevID = m_drawCallID;

      if (mainEverValid) {
        // RtCamera::getPosition returns the world-space camera position,
        // derived from inverse(worldToView). Same convention as drawCamPos
        // above.
        const Vector3 mainCamPos = mainCam.getPosition(/*freecam=*/false);
        const float dx = drawCamPos.x - mainCamPos.x;
        const float dy = drawCamPos.y - mainCamPos.y;
        const float dz = drawCamPos.z - mainCamPos.z;
        const float d2 = dx*dx + dy*dy + dz*dz;
        // Big epsilon. TF2 world coords are ~1e4; draws in Main's coord space
        // share Main's worldToView translation exactly. Draws from other
        // cameras (shadow, viewmodel, reflection, cinematic origin) differ
        // by hundreds-thousands of units. 100 cleanly separates clusters.
        constexpr float kEpsilon = 100.0f;
        const bool mismatch = d2 > (kEpsilon * kEpsilon);

        if (mismatch) {
          struct Key { int x, y, z; };
          static std::vector<Key> seenOrigins;
          Key k{ int(drawCamPos.x), int(drawCamPos.y), int(drawCamPos.z) };
          bool seen = false;
          for (const auto& s : seenOrigins) {
            if (s.x == k.x && s.y == k.y && s.z == k.z) { seen = true; break; }
          }
          if (!seen && seenOrigins.size() < 32) {
            seenOrigins.push_back(k);
            Logger::info(str::format(
              "[TLAS-FILTER] new foreign cam #", seenOrigins.size(),
              " draw=", m_drawCallID, " frame=", s_tlasFilterFrame,
              " drawCamPos=(", drawCamPos.x, ",", drawCamPos.y, ",", drawCamPos.z, ")",
              " mainCamPos=(", mainCamPos.x, ",", mainCamPos.y, ",", mainCamPos.z, ")",
              " |delta|=", std::sqrt(d2)));
          }

          // NV-DXVK: rejection DISABLED. Filter runs on the D3D11 thread
          // BEFORE classification (which happens on CS thread inside
          // processCameraData). Rejecting a draw here prevents it from
          // reaching the classifier, which prevents Main from re-latching
          // when the gameplay camera moves. Net effect: Main froze at the
          // first latch and every subsequent gameplay draw was "foreign"
          // by stale comparison. Keep counting/logging for diagnostic, but
          // let the draw through. Coord-space coherence has to be enforced
          // downstream of the classifier (e.g., in RtInstanceManager when
          // building the TLAS), not pre-classification.
          ++s_tlasReject;
          // BumpFilter(FilterReason::FullscreenQuad);
          // return;
        } else {
          ++s_tlasAccept;
        }
      } else {
        // No Main yet — permit the draw through. Log once per session so we
        // know the filter is observing but passing during the first-frame gap.
        static bool sNoMainLogged = false;
        if (!sNoMainLogged) {
          sNoMainLogged = true;
          Logger::info(str::format(
            "[TLAS-FILTER] no Main latched yet at frame ", frameId,
            " — passing draws through until Main is available"));
        }
        ++s_tlasNoMain;
      }
    }

    // NV-DXVK: scene dump for cbuffer-based BSP draws (non-fanout). The
    // bone-instance fanout dump above only catches g_modelInst-style draws.
    // Anything that uses CBufModelInstance (cbuffer-based world matrix)
    // never reaches the fanout — that's where ground/walls likely live.
    // Skip if fanout already handled this draw.
    if (m_currentInstancesToObject == nullptr
        && SceneDump::shouldDumpThisFrame()
        && posSem
        && posSem->format == VK_FORMAT_R32G32_UINT) {
      std::lock_guard<std::mutex> lk(SceneDump::g_mutex);
      const bool firstOpen = !SceneDump::g_obj.is_open();
      SceneDump::open();
      if (firstOpen && SceneDump::g_obj.is_open()) {
        SceneDump::writeCameraMarker();
      }
      if (SceneDump::g_obj.is_open()) {
        const auto& pvb = m_context->m_state.ia.vertexBuffers[posSem->inputSlot];
        const uint8_t* posData = nullptr; size_t posLen = 0;
        if (pvb.buffer != nullptr) {
          const auto& imm = pvb.buffer->GetImmutableData();
          if (!imm.empty()) {
            posData = imm.data() + pvb.offset + posSem->byteOffset;
            posLen  = imm.size() - (pvb.offset + posSem->byteOffset);
          }
        }
        const uint8_t* idxData = nullptr; size_t idxLen = 0;
        VkIndexType ixType = VK_INDEX_TYPE_UINT16;
        if (indexed) {
          const auto& ib = m_context->m_state.ia.indexBuffer;
          if (ib.buffer != nullptr) {
            const auto& imm = ib.buffer->GetImmutableData();
            if (!imm.empty()) {
              idxData = imm.data() + ib.offset;
              idxLen  = imm.size() - ib.offset;
              ixType  = (ib.format == DXGI_FORMAT_R16_UINT)
                          ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            }
          }
        }
        if (posData && (!indexed || idxData)) {
          const Matrix4& T = dcs.transformData.objectToWorld;
          const uint32_t posStride = std::max<uint32_t>(8u, pvb.stride);
          const float kScale = 1.0f / 1024.0f;
          const float kBiasZ = -2048.0f;
          SceneDump::g_obj << "o BSP_CB_" << SceneDump::g_objectsWritten++
                           << "_v" << dcs.geometryData.vertexCount << "\n";
          if (!indexed) {
            for (uint32_t v = 0; v < count; ++v) {
              size_t off = static_cast<size_t>(v) * posStride;
              if (off + 8 > posLen) break;
              const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
              uint32_t xi = SceneDump::decodeX(up[0]);
              uint32_t yi = SceneDump::decodeY(up[0], up[1]);
              uint32_t zi = SceneDump::decodeZ(up[1]);
              float lx = float(xi) * kScale - 1024.0f;
              float ly = float(yi) * kScale - 1024.0f;
              float lz = float(zi) * kScale + kBiasZ;
              float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
              float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
              float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
              SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
            }
            const uint32_t triCount = count / 3;
            for (uint32_t t = 0; t < triCount; ++t) {
              uint32_t a = SceneDump::g_baseVtx + t * 3 + 1;
              SceneDump::g_obj << "f " << a << " " << (a+1) << " " << (a+2) << "\n";
            }
            SceneDump::g_baseVtx += count;
          } else {
            const uint32_t idxStride = (ixType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
            uint32_t maxV = 0;
            for (uint32_t i = 0; i < count; ++i) {
              size_t io = static_cast<size_t>(start + i) * idxStride;
              if (io + idxStride > idxLen) { maxV = 0; break; }
              uint32_t idx = (idxStride == 2)
                ? *reinterpret_cast<const uint16_t*>(idxData + io)
                : *reinterpret_cast<const uint32_t*>(idxData + io);
              idx += static_cast<uint32_t>(std::max(base, 0));
              if (idx > maxV) maxV = idx;
            }
            const uint32_t vCount = maxV + 1;
            for (uint32_t v = 0; v < vCount; ++v) {
              size_t off = static_cast<size_t>(v) * posStride;
              if (off + 8 > posLen) break;
              const uint32_t* up = reinterpret_cast<const uint32_t*>(posData + off);
              uint32_t xi = SceneDump::decodeX(up[0]);
              uint32_t yi = SceneDump::decodeY(up[0], up[1]);
              uint32_t zi = SceneDump::decodeZ(up[1]);
              float lx = float(xi) * kScale - 1024.0f;
              float ly = float(yi) * kScale - 1024.0f;
              float lz = float(zi) * kScale + kBiasZ;
              float wx = T[0][0]*lx + T[1][0]*ly + T[2][0]*lz + T[3][0];
              float wy = T[0][1]*lx + T[1][1]*ly + T[2][1]*lz + T[3][1];
              float wz = T[0][2]*lx + T[1][2]*ly + T[2][2]*lz + T[3][2];
              SceneDump::g_obj << "v " << wx << " " << wy << " " << wz << "\n";
            }
            const uint32_t triCount = count / 3;
            for (uint32_t t = 0; t < triCount; ++t) {
              uint32_t i0base = (start + t * 3);
              size_t i0o = static_cast<size_t>(i0base + 0) * idxStride;
              size_t i1o = static_cast<size_t>(i0base + 1) * idxStride;
              size_t i2o = static_cast<size_t>(i0base + 2) * idxStride;
              if (i2o + idxStride > idxLen) break;
              uint32_t i0 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i0o) : *reinterpret_cast<const uint32_t*>(idxData + i0o);
              uint32_t i1 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i1o) : *reinterpret_cast<const uint32_t*>(idxData + i1o);
              uint32_t i2 = (idxStride == 2) ? *reinterpret_cast<const uint16_t*>(idxData + i2o) : *reinterpret_cast<const uint32_t*>(idxData + i2o);
              i0 += static_cast<uint32_t>(std::max(base, 0));
              i1 += static_cast<uint32_t>(std::max(base, 0));
              i2 += static_cast<uint32_t>(std::max(base, 0));
              SceneDump::g_obj << "f " << (SceneDump::g_baseVtx + i0 + 1) << " "
                                       << (SceneDump::g_baseVtx + i1 + 1) << " "
                                       << (SceneDump::g_baseVtx + i2 + 1) << "\n";
            }
            SceneDump::g_baseVtx += vCount;
          }
        }
      }
    }

    // Reject NDC-space screen quads now that ExtractTransforms has cached the VP.
    if (isNdcScreenQuad) {
      BumpFilter(FilterReason::FullscreenQuad);
      return;
    }

    // NV-DXVK: UI / overlay / pre-3D filter.  ExtractTransforms flips this
    // flag when it can't find any perspective projection in the game's
    // cbuffers and has to fall back to the viewport-derived synthetic
    // camera.  That situation means one of three things, all of which
    // should keep the draw OUT of the RTX pipeline:
    //   1. The game is rendering a 2D UI / HUD / menu (Source engine main
    //      menu — this is the Titanfall 2 case).
    //   2. The game is playing a video (Bink through a fullscreen quad or
    //      textured blit).
    //   3. The game hasn't bound any cbuffers yet (very early boot frame).
    // In every case the native DXVK D3D11 raster path — which was already
    // recorded by the EmitCs([=] (DxvkContext* ctx) { ctx->draw*(); })
    // call inside D3D11DeviceContext::Draw* BEFORE we were invoked — will
    // rasterize the draw correctly to the bound render target.  Skipping
    // RTX submission just means it won't be ray-traced.  Combined with the
    // drawCallID-gated safety net below, this lets the native backbuffer
    // content pass through injectRTX() unchanged (it no-ops when the
    // camera is invalid), matching exactly what D3D9 Remix does via
    // isRenderingUI() + RtxGeometryStatus::Rasterized.
    if (m_lastExtractUsedFallback) {
      // NV-DXVK: If a previous draw in this frame already found a real
      // VP, reuse those transforms instead of filtering this draw as UI.
      // Source only populates the VP cbuffer on draws 250+ (main opaque
      // pass), but draws 1-249 (shadows, depth prepass, etc.) are still
      // real gameplay geometry that should be ray-traced.
      //
      // EXCEPT: if the V2 classifier definitively identified this draw
      // as UI (screenspace 2D with no real transform), the cached-VP
      // reuse would put a 2D NDC quad into the world TLAS where it
      // renders as nothing. Force the TRUE UI branch so native raster
      // gets to draw the HUD/menu.
      if (m_foundRealProjThisFrame && !m_lastClassifierSaidUi) {
        // NV-DXVK: Guard against degenerate cached worldToView.  If the
        // translation column is all zeros the cached matrix has no real
        // camera position and submitting geometry with it produces
        // degenerate/coincident triangles that crash GPU RT hardware
        // (VK_ERROR_DEVICE_LOST).  Reject these as UIFallback instead.
        const auto& cached = m_lastGoodTransforms.worldToView;
        if (cached[3][0] == 0.0f && cached[3][1] == 0.0f && cached[3][2] == 0.0f) {
          BumpFilter(FilterReason::UIFallback);
          m_lastDrawFilteredAsUI = true;
          return;
        }
        // NV-DXVK: Only reuse the CAMERA transforms (viewToProjection,
        // worldToView) — NOT objectToWorld which is per-object and was
        // already extracted for THIS draw by ExtractTransforms.  The
        // previous version copied the entire m_lastGoodTransforms
        // including objectToWorld from draw #251, which gave every
        // subsequent draw the same world transform → all objects at
        // the same position → overlapping degenerate BLAS → GPU hang.
        dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
        dcs.transformData.worldToView      = m_lastGoodTransforms.worldToView;
        // Recompute objectToView from the corrected worldToView +
        // this draw's own objectToWorld.
        dcs.transformData.objectToView = dcs.transformData.objectToWorld;
        if (!isIdentityExact(dcs.transformData.worldToView))
          dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
        // NV-DXVK: Reject draws where objectToView translation is extreme.
        // Shadow/depth passes use light-space transforms that, when combined
        // with the main camera VP, produce positions far from the camera
        // → huge BLAS → GPU TDR.
        {
          const auto& o2v = dcs.transformData.objectToView;
          const float maxT = 100000.0f;
          if (std::abs(o2v[3][0]) > maxT || std::abs(o2v[3][1]) > maxT || std::abs(o2v[3][2]) > maxT ||
              !std::isfinite(o2v[3][0]) || !std::isfinite(o2v[3][1]) || !std::isfinite(o2v[3][2])) {
            BumpFilter(FilterReason::UIFallback);
            // NOTE: do NOT set m_lastDrawFilteredAsUI — this is shadow/depth
            // rejection not actual UI. Keep native raster suppressed.
            return;
          }
        }
        // Fall through to submit to RTX.
        static uint32_t s_reusedCount = 0;
        if (s_reusedCount < 100) {
          ++s_reusedCount;
          const auto& T = dcs.transformData;
          Logger::info(str::format(
              "[D3D11Rtx] Reusing frame VP for fallback draw #",
              m_drawCallID, " (rawDraw=", m_rawDrawCount, ")",
              " o2w T=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
              " w2v T=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")"));
        }
      } else {
        // NV-DXVK: TRUE UI-class draw — no real projection has been found in
        // any prior draw of this frame. Flag for OnDraw* to allow native
        // rasterization so the menu/HUD at least enters the backbuffer.
        BumpFilter(FilterReason::UIFallback);
        m_lastDrawFilteredAsUI = true;
        return;
      }
    }

    // Apply per-instance world transform when submitting instanced draws.
    if (instanceTransform) {
      dcs.transformData.objectToWorld = *instanceTransform;
      m_lastO2wPathId = 9;  // per-instance override (fanout tforms)
      // Recompute objectToView with the per-instance world matrix.
      dcs.transformData.objectToView = dcs.transformData.objectToWorld;
      if (!isIdentityExact(dcs.transformData.worldToView))
        dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
      std::string vsH = m_currentVsHashCache.empty()
        ? std::string("<novs>") : m_currentVsHashCache.substr(0, 19);
      Logger::info(str::format(
        "[D3D11Rtx.o2w.fanout] vs=", vsH,
        " drawID=", m_drawCallID,
        " inst.T=(", (*instanceTransform)[3][0], ",",
        (*instanceTransform)[3][1], ",", (*instanceTransform)[3][2], ")"));
    }

    // NV-DXVK: For bone-instanced draws with instancesToObject.
    // t31 matrix IS the world transform (from shader decompilation).
    // BLAS = localPos (bone buffers stripped), objectToWorld = identity,
    // instancesToObject[i] = t31_mat[i] places in world directly.
    if (m_boneInstanceCount > 0 && m_currentInstancesToObject) {
      static uint32_t sSubmitLog = 0;
      if (sSubmitLog < 10) {
        ++sSubmitLog;
        const auto& w2v = dcs.transformData.worldToView;
        const auto& o2w = dcs.transformData.objectToWorld;
        const auto& i2o0 = (*m_currentInstancesToObject)[0];
        Logger::info(str::format(
          "[D3D11Rtx] SubmitBone: camPos(w2v.T)=(", -w2v[3][0], ",", -w2v[3][1], ",", -w2v[3][2], ")",
          " origO2W.T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " i2o0.T=(", i2o0[3][0], ",", i2o0[3][1], ",", i2o0[3][2], ")",
          " finalInst0=(", o2w[3][0] + i2o0[3][0], ",",
          o2w[3][1] + i2o0[3][1], ",", o2w[3][2] + i2o0[3][2], ")"));
      }
      // t31 contains VIEW-SPACE transforms (view * model). Need inv(worldToView)
      // to get world-space matrices. Pre-compute inv(w2v) once per SubmitDraw
      // and multiply with each instance's transform.
      //
      // Game shader: clipPos = cb2 * t31[i] * localPos  (cb2 = projection only)
      // We want:     worldPos = inv(worldToView) * t31[i] * localPos
      // => instancesToObject[i] = inv(worldToView) * t31[i]
      //
      // Since we can't modify the vector here (it's shared, stable pointer),
      // we apply the inverse via objectToWorld. Each i2o[i] is already t31[i],
      // so objectToWorld = inv(worldToView) gives the same math.
      dcs.transformData.instancesToObject = m_currentInstancesToObject;
      // NV-DXVK: Pass ownership too so it flows into RtInstance via instance_manager.
      dcs.transformData.instancesToObjectOwner = m_currentInstancesToObjectOwner;
      // t31 is already the full world-space model transform.
      // objectToWorld = identity, instancesToObject = t31[i], done.
      dcs.transformData.objectToWorld = Matrix4();
      m_lastO2wPathId = 10;  // bone-instanced fanout: identity o2w

      // NV-DXVK CRITICAL: If ExtractTransforms produced an identity w2v
      // for this draw (observed: COMMIT w2vT=(0,0,0) o2vT=(0,0,0)), the
      // main RT camera ends up at world origin and rays never hit the
      // real-world geometry at (-5179, 279, 92). Fall back to the last
      // good cached w2v so the fanout path always has a real camera.
      if (isIdentityExact(dcs.transformData.worldToView)
          && !isIdentityExact(m_lastGoodTransforms.worldToView)) {
        dcs.transformData.worldToView      = m_lastGoodTransforms.worldToView;
        dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
        static uint32_t sPath10W2vRestore = 0;
        if (sPath10W2vRestore < 10) {
          ++sPath10W2vRestore;
          const auto& w2v = dcs.transformData.worldToView;
          Logger::info(str::format(
            "[D3D11Rtx.path10.w2vRestore] drawID=", m_drawCallID,
            " restored cached w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")"));
        }
      }
      dcs.transformData.objectToView = dcs.transformData.worldToView;

      // (TestPos log removed — inv(w2v) not used)

      static uint32_t sPostLog = 0;
      if (sPostLog < 5) {
        ++sPostLog;
        const auto& o2w = dcs.transformData.objectToWorld;
        // Compute magnitude of each row of the 3x3 rotation part
        auto col_mag = [&](int c) {
          float s = o2w[c][0]*o2w[c][0] + o2w[c][1]*o2w[c][1] + o2w[c][2]*o2w[c][2];
          return std::sqrt(s);
        };
        Logger::info(str::format(
          "[D3D11Rtx] InvW2V: T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")",
          " col0=(", o2w[0][0], ",", o2w[0][1], ",", o2w[0][2], ") mag=", col_mag(0),
          " col1=(", o2w[1][0], ",", o2w[1][1], ",", o2w[1][2], ") mag=", col_mag(1),
          " col2=(", o2w[2][0], ",", o2w[2][1], ",", o2w[2][2], ") mag=", col_mag(2)));
      }
      dcs.geometryData.boneMatrixBuffer = RasterBuffer();
      dcs.geometryData.boneIndexBuffer = RasterBuffer();
      geo.boneMatrixBuffer = RasterBuffer();
      geo.boneIndexBuffer = RasterBuffer();

      // DEBUG: skip actual RTX submit for bone-instanced draws (isolate non-instanced)
      if (m_debugHideBoneInstanced) {
        return;
      }
    }
    // NV-DXVK: For bone-instanced draws with attached bone buffers (N-draw path),
    // the interleaver applies the bone transform. Set objectToWorld to identity.
    else if (m_attachBoneBuffers && geo.boneMatrixBuffer.defined()) {
      dcs.transformData.objectToWorld = Matrix4();
      m_lastO2wPathId = 10;  // bone-instanced (N-draw path): identity o2w
      // Same camera-rescue as the fanout branch above.
      if (isIdentityExact(dcs.transformData.worldToView)
          && !isIdentityExact(m_lastGoodTransforms.worldToView)) {
        dcs.transformData.worldToView      = m_lastGoodTransforms.worldToView;
        dcs.transformData.viewToProjection = m_lastGoodTransforms.viewToProjection;
      }
      dcs.transformData.objectToView = dcs.transformData.worldToView;
    }

    // Let processCameraData() classify the camera from the matrices.
    // Hardcoding Main would bypass Remix's sky/portal/shadow detection.
    dcs.cameraType       = CameraType::Unknown;
    dcs.usesVertexShader = (m_context->m_state.vs.shader != nullptr);
    dcs.usesPixelShader  = (m_context->m_state.ps.shader != nullptr);

    // NV-DXVK: Deterministic pass classifier — pass the current D3D11 viewport
    // to Remix so camera_manager can distinguish gameplay draws (viewport ==
    // back buffer) from shadow cascades / cubemaps / RT targets (off-size or
    // square viewports). No matrix heuristics involved.
    {
      const auto& vps = m_context->m_state.rs.viewports;
      if (vps[0].Width > 0.0f && vps[0].Height > 0.0f) {
        dcs.transformData.viewportWidth  = vps[0].Width;
        dcs.transformData.viewportHeight = vps[0].Height;
      }
    }

    // NV-DXVK: Capture bound VS hash for game-native per-draw identification.
    // Gameplay-world passes use a small, stable set of vertex shaders; fullscreen /
    // post / UI draws use different ones even when they share a projection shape.
    // Keying Main-camera classification off this hash eliminates the need for
    // matrix-property heuristics (aspect/tinyScale/maxZ/w2vT).
    dcs.transformData.worldToViewPathId = m_lastWtvPathId;
    if (m_context->m_state.vs.shader != nullptr) {
      auto* common = m_context->m_state.vs.shader->GetCommonShader();
      if (common != nullptr) {
        const auto& dxvkShader = common->GetShader();
        if (dxvkShader != nullptr) {
          dcs.transformData.vertexShaderHash =
            static_cast<XXH64_hash_t>(dxvkShader->getHash());
        }
      }
    }

    // D3D11 shaders are always SM 4.0+.
    if (dcs.usesVertexShader)
      dcs.vertexShaderInfo = ShaderProgramInfo{4, 0};
    if (dcs.usesPixelShader)
      dcs.pixelShaderInfo = ShaderProgramInfo{4, 0};
    dcs.zWriteEnable     = zWriteEnable;
    dcs.zEnable          = zEnable;
    dcs.stencilEnabled   = stencilEnabled;
    dcs.drawCallID       = m_drawCallID++;
    m_lastDrawCaptured   = true;  // Signal caller to skip D3D11 rasterization
    // NV-DXVK: record the successful submit against the current VS hash.
    if (!m_currentVsHashCache.empty())
      ++m_vsFrameStats[m_currentVsHashCache].submitted;

    // NV-DXVK: orientation probe — log the world-space directions that each
    // object's LOCAL +X/+Y/+Z axes map to, plus translation. No identity
    // filter — BSP uses pure-translation objectToWorld and we want to see
    // where BSP chunks are placed too.
    //
    // Log only the FIRST occurrence per VS hash so BSP (high-count shader)
    // doesn't flood and we still see prop/foliage variety.
    {
      static std::unordered_set<XXH64_hash_t> sLoggedHashes;
      static uint32_t sOrientLog = 0;
      const XXH64_hash_t vsH = dcs.transformData.vertexShaderHash;
      if (sOrientLog < 50 && sLoggedHashes.count(vsH) == 0) {
        sLoggedHashes.insert(vsH);
        ++sOrientLog;
        const auto& o = dcs.transformData.objectToWorld;
        char vsHex[32];
        std::snprintf(vsHex, sizeof(vsHex), "0x%016llx",
                      static_cast<unsigned long long>(vsH));
        Logger::info(str::format(
          "[D3D11Rtx.orient] #", sOrientLog,
          " draw=", dcs.drawCallID,
          " vs=", vsHex,
          " localX_w=(", o[0][0], ",", o[0][1], ",", o[0][2], ")",
          " localY_w=(", o[1][0], ",", o[1][1], ",", o[1][2], ")",
          " localZ_w=(", o[2][0], ",", o[2][1], ",", o[2][2], ")",
          " T_w=(", o[3][0], ",", o[3][1], ",", o[3][2], ")"));
      }
    }

    // Viewport depth range from D3D11_VIEWPORT.MinDepth / MaxDepth.
    {
      const auto& vp = m_context->m_state.rs.viewports[0];
      dcs.minZ = std::clamp(vp.MinDepth, 0.0f, 1.0f);
      dcs.maxZ = std::clamp(vp.MaxDepth, 0.0f, 1.0f);
    }

    // D3D11 has no legacy fog — engines bake fog into shaders.
    // FogState defaults to mode=0 (none), which is correct.

    // Register this context as the active rendering context so the primary
    // swap chain routes EndFrame/OnPresent through us, not a video-playback
    // device that happened to present first.
    FillMaterialData(dcs.materialData);

    // NV-DXVK start: Per-vertex skinning — capture bone matrices from VS SRV t30
    if (geo.numBonesPerVertex > 0) {
      bool gotBones = false;
      const uint32_t kBoneSrvSlot = 30;
      ID3D11ShaderResourceView* boneSrv = nullptr;
      if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
        boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();

      if (boneSrv) {
        Com<ID3D11Resource> boneRes;
        boneSrv->GetResource(&boneRes);
        auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());

        if (boneBuf) {
          // Try multiple paths to access bone data (buffer may be GPU-only)
          const uint8_t* bonePtr = nullptr;
          size_t boneBufLen = 0;

          // Path 1: mapped slice (WRITE_DISCARD mapped memory)
          {
            const auto mapped = boneBuf->GetMappedSlice();
            if (mapped.mapPtr && mapped.length >= 48) {
              bonePtr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
              boneBufLen = mapped.length;
            }
          }

          // Path 2: DxvkBuffer direct mapPtr (host-visible buffers)
          if (!bonePtr) {
            DxvkBufferSlice boneSlice = boneBuf->GetBufferSlice();
            if (boneSlice.defined()) {
              void* p = boneSlice.buffer()->mapPtr(0);
              if (p) {
                bonePtr = reinterpret_cast<const uint8_t*>(p) + boneSlice.offset();
                boneBufLen = boneSlice.length();
              }
            }
          }

          // Path 3: cached from UpdateSubresource interception
          if (!bonePtr && m_hasCachedBone0 && m_lastBoneBuffer == boneBuf) {
            bonePtr = reinterpret_cast<const uint8_t*>(m_cachedBone0);
            boneBufLen = 48; // Only bone 0 is cached
          }

          if (bonePtr && boneBufLen >= 48) {
            const uint32_t numBones = static_cast<uint32_t>(boneBufLen / 48);
            const uint32_t maxBones = std::min(numBones, 256u); // SkinningArgs limit

            dcs.skinningData.numBonesPerVertex = geo.numBonesPerVertex;
            dcs.skinningData.numBones = maxBones;
            dcs.skinningData.minBoneIndex = 0;
            dcs.skinningData.pBoneMatrices.resize(maxBones);

            bool allValid = true;
            for (uint32_t b = 0; b < maxBones; ++b) {
              const float* m = reinterpret_cast<const float*>(bonePtr + b * 48);
              // Validate bone data isn't garbage
              if (!std::isfinite(m[0]) || !std::isfinite(m[3])) {
                allValid = false;
                break;
              }
              // float3x4 row-major → Matrix4
              dcs.skinningData.pBoneMatrices[b] = Matrix4(
                Vector4(m[0], m[1], m[2],  0.0f),
                Vector4(m[4], m[5], m[6],  0.0f),
                Vector4(m[8], m[9], m[10], 0.0f),
                Vector4(m[3], m[7], m[11], 1.0f));
            }

            if (allValid) {
              dcs.skinningData.computeHash();
              gotBones = true;
            }
          }
        }
      }

      // If we couldn't read bone matrices, disable skinning for this draw
      // to prevent dispatchSkinning from running with empty bone data.
      if (!gotBones) {
        static uint32_t sBoneFailLog = 0;
        if (sBoneFailLog < 5) {
          ++sBoneFailLog;
          Logger::warn(str::format(
            "[D3D11Rtx] Per-vertex skinning: could not read bone matrices from t30.",
            " SRV=", boneSrv ? "bound" : "null",
            " bonesPerVert=", geo.numBonesPerVertex,
            " bwFmt=", bwSem ? uint32_t(bwSem->format) : 0,
            " biFmt=", biSem ? uint32_t(biSem->format) : 0));
        }
        geo.numBonesPerVertex = 0;
        geo.blendWeightBuffer  = RasterBuffer();
        geo.blendIndicesBuffer = RasterBuffer();
        dcs.geometryData = geo;
      } else {
        static uint32_t sBoneOkLog = 0;
        if (sBoneOkLog < 3) {
          ++sBoneOkLog;
          Logger::info(str::format(
            "[D3D11Rtx] Per-vertex skinning: captured ", dcs.skinningData.numBones,
            " bones (", geo.numBonesPerVertex, " per vertex)",
            " bwFmt=", uint32_t(bwSem->format),
            " biFmt=", uint32_t(biSem->format)));
        }
      }
    }
    // NV-DXVK end

    DrawParameters params;
    params.instanceCount = 1;
    params.vertexCount   = indexed ? 0 : count;
    params.indexCount    = indexed ? count : 0;
    params.firstIndex    = indexed ? start : 0;
    params.vertexOffset  = indexed ? static_cast<uint32_t>(std::max(base, 0)) : start;

    // NV-DXVK DEBUG: Log draw parameters for fmt=101 draws
    if (posSem->format == VK_FORMAT_R32G32_UINT) {
      static uint32_t sDrawParamLog = 0;
      if (sDrawParamLog < 20) {
        ++sDrawParamLog;
        Logger::info(str::format(
          "[D3D11Rtx] DrawParams: indexed=", indexed ? 1 : 0,
          " count=", count, " start=", start, " base=", base,
          " vertCount=", geo.vertexCount,
          " idxCount=", params.indexCount,
          " firstIdx=", params.firstIndex,
          " vtxOff=", params.vertexOffset,
          " stride=", posBuffer.stride(),
          " idxFmt=", indexed ? uint32_t(m_context->m_state.ia.indexBuffer.format) : 0,
          " idxOff=", indexed ? m_context->m_state.ia.indexBuffer.offset : 0));
      }
    }

    // === PER-DRAW TRANSFORM + VERTEX DIAGNOSTIC ===
    // Log every draw for the first 5 in-game frames (m_drawCallID-based gate).
    {
      static uint32_t s_submitLogFrame = 0;
      static uint32_t s_submitPrevID   = UINT32_MAX;
      if (dcs.drawCallID == 0 || dcs.drawCallID < s_submitPrevID)
        ++s_submitLogFrame;
      s_submitPrevID = dcs.drawCallID;

      // NV-DXVK: log during first gameplay frames (after boot-time menus).
      // Tracked via global "gameplay started" latch in EndFrame.
      if (s_GameplayLogFrames > 0) {
        const auto& T = dcs.transformData;
        const bool o2wIdentity = isIdentityExact(T.objectToWorld);
        // VS hash
        std::string vsHash = "?";
        auto vsShaderCom = m_context->m_state.vs.shader;
        if (vsShaderCom != nullptr && vsShaderCom->GetCommonShader() != nullptr) {
          auto& s = vsShaderCom->GetCommonShader()->GetShader();
          if (s != nullptr) vsHash = s->getShaderKey().toString();
        }
        Logger::info(str::format(
          "[D3D11Rtx] Submit drawID=", dcs.drawCallID,
          " frame=", s_submitLogFrame,
          " VS=", vsHash,
          " verts=", geo.vertexCount,
          " o2w:", o2wIdentity ? "IDENTITY" : "nonId",
          " T=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
          " o2vT=(", T.objectToView[3][0], ",", T.objectToView[3][1], ",", T.objectToView[3][2], ")",
          " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")"));

        // Sample first vertex position from the position buffer.
        const auto& posBuf = geo.positionBuffer;
        if (posBuf.defined()) {
          const float* p = reinterpret_cast<const float*>(
            posBuf.mapPtr(posBuf.offsetFromSlice()));
          if (p) {
            Logger::info(str::format(
              "[D3D11Rtx]   pos[0]=(", p[0], ",", p[1], ",", p[2], ")",
              " stride=", posBuf.stride(),
              " fmt=", static_cast<uint32_t>(posBuf.vertexFormat())));
          }
        }
      }
    }

    // (stale transform filter removed — worldToView now set by cross-frame VP)

    // NV-DXVK: Log every submitted draw with key info for TDR diagnosis.
    // Logger flushes to disk so the last entry before a TDR is visible.
    {
      const auto& T = dcs.transformData;
      const auto& G = dcs.geometryData;

      // NV-DXVK: Log the VS hash of non-instanced bone draws (these work
      // correctly — their shader tells us the right cbuffer layout Remix reads).
      static uint32_t sLoggedNonInstBone = 0;
      const bool isBoneInst = (m_boneInstanceCount > 0 && m_currentInstancesToObject);
      if (sLoggedNonInstBone < 5 && G.boneMatrixBuffer.defined() && !isBoneInst
          && G.positionBuffer.vertexFormat() == VK_FORMAT_R32G32_UINT
          && std::abs(T.objectToWorld[3][0]) > 100.f) {  // real world translation
        ++sLoggedNonInstBone;
        const auto& o2w = T.objectToWorld;
        Logger::info(str::format(
          "[D3D11Rtx] Non-inst bone o2w:"
          " col0=(", o2w[0][0], ",", o2w[0][1], ",", o2w[0][2], ")",
          " col1=(", o2w[1][0], ",", o2w[1][1], ",", o2w[1][2], ")",
          " col2=(", o2w[2][0], ",", o2w[2][1], ",", o2w[2][2], ")",
          " T=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
      }

      // NV-DXVK: bump per-frame histogram using the path tag set by
      // whichever site most recently wrote to transforms.objectToWorld.
      {
        const uint32_t pid = (m_lastO2wPathId < 16) ? m_lastO2wPathId : 15;
        ++m_o2wPathCounts[pid];
        if (!m_currentVsHashCache.empty()) {
          const std::string vsKey = m_currentVsHashCache.substr(0, 19);
          auto& arr = m_vsO2wPathCounts[vsKey];
          ++arr[pid];
        }
      }
      Logger::info(str::format(
        "[D3D11Rtx] COMMIT vs=", m_currentVsHashCache.substr(0, 19),
        " id=", dcs.drawCallID,
        " verts=", G.vertexCount,
        " fmt=", uint32_t(G.positionBuffer.vertexFormat()),
        " stride=", G.positionBuffer.stride(),
        " bone=", G.boneMatrixBuffer.defined() ? 1 : 0,
        " inst=", G.boneInstanceIndex,
        " o2wPath=", m_lastO2wPathId,
        " o2wT=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
        " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")",
        " o2vT=(", T.objectToView[3][0], ",", T.objectToView[3][1], ",", T.objectToView[3][2], ")",
        " raw=", m_rawDrawCount));
      // Extra: log the o2w rotation too (identity vs rotated detection).
      {
        const auto& M = T.objectToWorld;
        const bool identRot =
          std::abs(M[0][0] - 1.f) < 1e-4f && std::abs(M[1][1] - 1.f) < 1e-4f && std::abs(M[2][2] - 1.f) < 1e-4f &&
          std::abs(M[0][1]) < 1e-4f && std::abs(M[0][2]) < 1e-4f &&
          std::abs(M[1][0]) < 1e-4f && std::abs(M[1][2]) < 1e-4f &&
          std::abs(M[2][0]) < 1e-4f && std::abs(M[2][1]) < 1e-4f;
        Logger::info(str::format(
          "[D3D11Rtx.o2wRot] id=", dcs.drawCallID,
          " identityRot=", identRot ? 1 : 0,
          " col0=(", M[0][0], ",", M[0][1], ",", M[0][2], ")",
          " col1=(", M[1][0], ",", M[1][1], ",", M[1][2], ")",
          " col2=(", M[2][0], ",", M[2][1], ",", M[2][2], ")"));
      }
    }

    m_context->EmitCs([params, dcs](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, dcs);
    });
  }

  void D3D11Rtx::OnUpdateSubresource(ID3D11Resource* pDstResource, const void* pSrcData, UINT SrcDataSize, UINT DstOffset, UINT BufSize) {
    if (!pSrcData) return;
    // Cache bone matrix buffer (t30 = 393216 bytes = 8192 bones × 48 bytes)
    // Intercept the FULL buffer on the CPU before it's uploaded to GPU.
    // This allows per-instance bone lookup without GPU readback.
    if (BufSize == 393216 && SrcDataSize >= 48) {
      m_lastBoneBuffer = static_cast<ID3D11Buffer*>(pDstResource);
      // Keep buffer at full size; write each update at DstOffset.
      if (m_fullBoneCache.size() != BufSize)
        m_fullBoneCache.resize(BufSize, 0);
      const size_t maxCopy = std::min(
        static_cast<size_t>(SrcDataSize),
        static_cast<size_t>(BufSize) - static_cast<size_t>(DstOffset));
      std::memcpy(m_fullBoneCache.data() + DstOffset, pSrcData, maxCopy);
      m_hasFullBoneCache = true;
      if (DstOffset == 0) {
        std::memcpy(m_cachedBone0, pSrcData, 48);
        m_hasCachedBone0 = true;
      }
    }
    // Log ALL buffer update sizes to find cb3
    static uint32_t sAllUpdateLog = 0;
    if (sAllUpdateLog < 50) {
      // Only log unique sizes
      static std::set<uint32_t> seenSizes;
      if (seenSizes.find(BufSize) == seenSizes.end()) {
        seenSizes.insert(BufSize);
        ++sAllUpdateLog;
        const float* fData = reinterpret_cast<const float*>(
          reinterpret_cast<const uint8_t*>(pSrcData) + DstOffset);
        Logger::info(str::format(
          "[D3D11Rtx] UpdateSub: bufSize=", BufSize,
          " off=", DstOffset, " len=", SrcDataSize,
          " f0=(", (SrcDataSize >= 16 ? fData[0] : 0), ",",
          (SrcDataSize >= 16 ? fData[1] : 0), ",",
          (SrcDataSize >= 16 ? fData[2] : 0), ",",
          (SrcDataSize >= 16 ? fData[3] : 0), ")"));
      }
    }
    // Cache cb3 — try multiple common sizes (208, 224, 256, 240)
    if ((BufSize == 208 || BufSize == 224 || BufSize == 240 || BufSize == 256)
        && DstOffset == 0 && SrcDataSize >= 48) {
      std::memcpy(m_cachedCb3, pSrcData, 48);
      m_hasCachedCb3 = true;
    }
  }

  void D3D11Rtx::EndFrame(const Rc<DxvkImage>& backbuffer) {
    const uint32_t draws = m_drawCallID;
    const uint32_t raw = m_rawDrawCount;
    // NV-DXVK: arm/finalize the scene dumper.
    SceneDump::armOnFirstGameplayFrame(raw);

    // NV-DXVK: dump key rtx.conf options once we hit a real gameplay frame so
    // we can verify the config file is actually being read.
    {
      static bool sCfgLogged = false;
      if (!sCfgLogged && raw > 50) {
        sCfgLogged = true;
        Logger::info(str::format("[D3D11Rtx] rtx.conf state at first gameplay frame:"));
        Logger::info(str::format("  rtx.pointInstancer.enable = ",
          RtxPointInstancerSystem::enable() ? "True" : "False"));
        Logger::info(str::format("  rtx.pointInstancer.cullingRadius = ",
          RtxPointInstancerSystem::cullingRadius()));
        Logger::info(str::format("  rtx.legacyMaterial.albedoConstant = (",
          LegacyMaterialDefaults::albedoConstant().x, ",",
          LegacyMaterialDefaults::albedoConstant().y, ",",
          LegacyMaterialDefaults::albedoConstant().z, ")"));
        Logger::info(str::format("  rtx.legacyMaterial.useAlbedoTextureIfPresent = ",
          LegacyMaterialDefaults::useAlbedoTextureIfPresent() ? "True" : "False"));
        Logger::info(str::format("  rtx.legacyMaterial.emissiveIntensity = ",
          LegacyMaterialDefaults::emissiveIntensity()));
        Logger::info(str::format("  rtx.debugView.debugViewIdx = ",
          DebugView::debugViewIdx()));
      }
    }
    if (SceneDump::g_obj.is_open() && !SceneDump::g_done) {
      // Closes after the dump frame ends.
      SceneDump::close();
    }
    // Latch: once we see a gameplay-scale frame, log details for N frames.
    {
      static bool s_latched = false;
      if (!s_latched && raw > 50) {
        s_latched = true;
        s_GameplayLogFrames = 5;
      } else if (s_GameplayLogFrames > 0) {
        --s_GameplayLogFrames;
      }
    }
    Logger::info(str::format("[D3D11Rtx] EndFrame: draws=", draws,
      " raw=", raw,
      " backbuffer=", backbuffer != nullptr ? 1 : 0));

    // NV-DXVK: diagnostic — if draws were issued but all filtered out,
    // dump the per-filter rejection counts so we know exactly which
    // SubmitDraw pre-filter is killing the game's main-menu draws.
    if (raw > 0 && draws < raw) {
      Logger::info(str::format("[D3D11Rtx]   filters:",
        " throttle=",       m_filterCounts[static_cast<uint32_t>(FilterReason::Throttle)],
        " nonTriTopo=",     m_filterCounts[static_cast<uint32_t>(FilterReason::NonTriTopology)],
        " noPS=",           m_filterCounts[static_cast<uint32_t>(FilterReason::NoPixelShader)],
        " noRTV=",          m_filterCounts[static_cast<uint32_t>(FilterReason::NoRenderTarget)],
        " count<3=",        m_filterCounts[static_cast<uint32_t>(FilterReason::CountTooSmall)],
        " fsQuad=",         m_filterCounts[static_cast<uint32_t>(FilterReason::FullscreenQuad)],
        " noLayout=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoInputLayout)],
        " noSemantics=",    m_filterCounts[static_cast<uint32_t>(FilterReason::NoSemantics)],
        " noPos=",          m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosition)],
        " pos2D=",          m_filterCounts[static_cast<uint32_t>(FilterReason::Position2D)],
        " noPosBuf=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosBuffer)],
        " noIdxBuf=",       m_filterCounts[static_cast<uint32_t>(FilterReason::NoIndexBuffer)],
        " hashFail=",       m_filterCounts[static_cast<uint32_t>(FilterReason::HashFailed)],
        " uiFallback=",     m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)],
        " unsupFmt=",       m_filterCounts[static_cast<uint32_t>(FilterReason::UnsupPosFmt)]));

      // NV-DXVK: o2w path histogram (which code path set objectToWorld per
      // committed draw). 0 = never set (identity), 1 = non-inst BSP t31,
      // 2 = t30 CPU Bone, 3 = t30 Bone-slice, 4 = CB3→O2W, 5 = RDEF,
      // 6 = trySourceFloat3x4, 7 = tryWorldCb, 8 = cb2@4 fallback,
      // 9 = per-instance (fanout), 10 = bone-instanced identity.
      Logger::info(str::format("[D3D11Rtx]   o2wPaths:",
        " identity=", m_o2wPathCounts[0],
        " t31=",      m_o2wPathCounts[1],
        " t30cpu=",   m_o2wPathCounts[2],
        " t30slice=", m_o2wPathCounts[3],
        " cb3=",      m_o2wPathCounts[4],
        " rdef=",     m_o2wPathCounts[5],
        " sf3x4=",    m_o2wPathCounts[6],
        " worldcb=",  m_o2wPathCounts[7],
        " cb2cam=",   m_o2wPathCounts[8],
        " fanout=",   m_o2wPathCounts[9],
        " boneInst=", m_o2wPathCounts[10],
        " skinnedChar=", m_o2wPathCounts[11]));

      // Per-VS o2w path breakdown — which shader took which path.
      // Sort by total draws desc so the noisiest shaders appear first.
      if (!m_vsO2wPathCounts.empty()) {
        std::vector<std::pair<std::string, std::array<uint32_t, 16>>> sv;
        sv.reserve(m_vsO2wPathCounts.size());
        for (auto& kv : m_vsO2wPathCounts) sv.push_back(kv);
        std::sort(sv.begin(), sv.end(), [](const auto& a, const auto& b) {
          uint32_t at = 0, bt = 0;
          for (uint32_t v : a.second) at += v;
          for (uint32_t v : b.second) bt += v;
          return at > bt;
        });
        Logger::info(str::format("[D3D11Rtx]   o2wPathsByVS (", sv.size(), " unique):"));
        for (const auto& kv : sv) {
          const auto& a = kv.second;
          uint32_t tot = 0; for (uint32_t v : a) tot += v;
          if (tot == 0) continue;
          std::string line = str::format("    ", kv.first, " n=", tot);
          static const char* kName[12] = {
            "id", "t31", "t30cpu", "t30slice", "cb3", "rdef", "sf3x4",
            "worldcb", "cb2cam", "fanout", "boneInst", "skinnedChar"
          };
          for (uint32_t p = 0; p < 12; ++p) {
            if (a[p] > 0) line += str::format(" ", kName[p], "=", a[p]);
          }
          Logger::info(line);
        }
      }
    }
    for (int i = 0; i < 16; ++i) m_o2wPathCounts[i] = 0;
    m_vsO2wPathCounts.clear();
    // NV-DXVK: per-VS outcome dump — each VS hash, #submits, #rejects per filter.
    // Gate on frames with meaningful draw counts so boot-time menus don't eat
    // the quota before gameplay starts.
    {
      static uint32_t s_vsDumpGameFrame = 0;
      const bool isGameplayFrame = (raw > 20);
      if (isGameplayFrame) ++s_vsDumpGameFrame;
      if (isGameplayFrame && s_vsDumpGameFrame <= 8 && !m_vsFrameStats.empty()) {
        static const char* kReasonName[] = {
          "Throttle","NonTriTopo","NoPS","NoRTV","TooSmall","FsQuad","NoLayout",
          "NoSem","NoPos","Pos2D","NoPosBuf","NoIdxBuf","HashFail","UIFallback","UnsupFmt"
        };
        Logger::info(str::format("[D3D11Rtx]   per-VS outcome (", m_vsFrameStats.size(), " unique):"));
        // Stable-ish order: sort by (submits+rejectTotal) desc by copying to vector.
        std::vector<std::pair<std::string, VsFrameStats>> sorted;
        sorted.reserve(m_vsFrameStats.size());
        for (const auto& kv : m_vsFrameStats) sorted.push_back(kv);
        std::sort(sorted.begin(), sorted.end(),
          [](const auto& a, const auto& b) {
            uint32_t at = a.second.submitted, bt = b.second.submitted;
            for (uint32_t r : a.second.rejects) at += r;
            for (uint32_t r : b.second.rejects) bt += r;
            return at > bt;
          });
        for (const auto& kv : sorted) {
          std::string line = str::format("    ", kv.first.substr(0, 18), " subm=", kv.second.submitted);
          for (uint32_t r = 0; r < static_cast<uint32_t>(FilterReason::Count); ++r) {
            if (kv.second.rejects[r] > 0)
              line += str::format(" ", kReasonName[r], "=", kv.second.rejects[r]);
          }
          Logger::info(line);
        }
      }
    }
    m_vsFrameStats.clear();

    for (uint32_t i = 0; i < static_cast<uint32_t>(FilterReason::Count); ++i)
      m_filterCounts[i] = 0;

    // NV-DXVK: Per-frame bone instancing summary
    // Rotate ring buffer: clear the slot we're about to reuse next frame.
    // Each slot holds transforms from 4 frames ago. Scene manager has
    // definitely finished with them.
    ++m_boneInstFrameId;
    if (!m_boneTransformRing.empty()) {
      uint32_t nextSlot = m_boneInstFrameId % 4;
      m_boneTransformRing[nextSlot].clear();
    }

    if (m_boneInstBatches > 0) {
      uint32_t ringSize = 0;
      for (const auto& slot : m_boneTransformRing) ringSize += static_cast<uint32_t>(slot.size());
      Logger::info(str::format(
        "[D3D11Rtx] BoneInst: batches=", m_boneInstBatches,
        " instances=", m_boneInstTotal,
        " uniqueVB=", m_boneInstVbPtrs.size(),
        " ringEntries=", ringSize));
    }
    m_boneInstVbPtrs.clear();
    m_boneInstBatches = 0;
    m_boneInstTotal = 0;
    m_boneInstSkipped = 0;
    m_boneInstNoCache = 0;
    m_boneInstCacheHits = 0;
    m_boneInstCacheMisses = 0;

    // NV-DXVK: removed dead safety net. It was preempting the classifier:
    // EndFrame's EmitCs lambda ran on the CS thread the NEXT frame, calling
    // processExternalCamera with frame N's last-extracted transforms (often
    // a UI/fallback matrix) stamped as frame N+1's frameId. The next frame's
    // gameplay draws then saw Main valid for frame N+1 already →
    // shouldUpdateMainCamera=false → classifier never re-latched. The
    // current classifier + hysteresis gate leaves Main invalid on frames
    // where no gameplay draw is found, which is correct — injectRTX
    // early-returns and the native raster content passes through unchanged.

    m_drawCallID = 0;
    m_rawDrawCount = 0;
    m_remixActiveThisFrame = false;
    m_foundRealProjThisFrame = false;
    // Projection cache (m_projSlot, m_projOffset, m_projStage, m_columnMajor)
    // is NOT reset for pure projections (cls 1/2) — the validation path at
    // the start of ExtractTransforms re-reads and re-scans only when the
    // cached location becomes stale.  Resetting every frame would force an
    // O(stages × slots × bufferBytes) scan on the first draw, which hangs
    // emulators with 64KB+ UBOs.
    //
    // NV-DXVK: Combined VP (cls 3/4) MUST be re-scanned each frame because
    // (a) the VP content changes with camera movement, and (b) Source only
    // binds the correct VP cbuffer during the main opaque pass (draws 200+),
    // not during early shadow/depth-prepass draws.  If we cached a false
    // positive from an early draw on the previous frame, resetting here gives
    // the next frame's late-draw scan a chance to find the real VP.
    if (m_projIsCombinedVP) {
      m_projSlot   = UINT32_MAX;
      m_projOffset = SIZE_MAX;
      m_projStage  = -1;
    }
    ++m_axisDetectFrame;

    m_context->EmitCs([backbuffer, draws](DxvkContext* ctx) {
      RtxContext* rtx = static_cast<RtxContext*>(ctx);
      const uint32_t fid = rtx->getDevice()->getCurrentFrameId();
      const bool camValid = rtx->getSceneManager().getCamera().isValid(fid);
      Logger::info(str::format("[D3D11Rtx] CS endFrame: frameId=", fid,
        " draws=", draws, " camValid=", camValid ? 1 : 0));
      rtx->endFrame(0, backbuffer, true);
    });
  }

  void D3D11Rtx::OnPresent(const Rc<DxvkImage>& swapchainImage) {
    m_context->EmitCs([swapchainImage](DxvkContext* ctx) {
      RtxContext* rtx = static_cast<RtxContext*>(ctx);
      rtx->onPresent(swapchainImage);
    });
  }

}
