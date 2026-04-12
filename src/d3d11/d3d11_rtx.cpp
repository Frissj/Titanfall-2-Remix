#include "d3d11_rtx.h"
#include <set>

// Include dxvk_device.h before any rtx headers so that dxvk_buffer.h and
// sibling headers (included bare by rtx_utils.h) are already in the TU.
#include "../dxvk/dxvk_device.h"

#include "d3d11_context.h"
#include "d3d11_buffer.h"
#include "d3d11_input_layout.h"
#include "d3d11_device.h"
#include "d3d11_view_srv.h"
#include "d3d11_sampler.h"
#include "d3d11_depth_stencil.h"
#include "d3d11_blend.h"
#include "d3d11_rasterizer.h"

#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_camera.h"
#include "../dxvk/rtx_render/rtx_camera_manager.h"
#include "../dxvk/rtx_render/rtx_scene_manager.h"
#include "../dxvk/rtx_render/rtx_light_manager.h"
#include "../dxvk/rtx_render/rtx_matrix_helpers.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

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

  void D3D11Rtx::OnDraw(UINT vertexCount, UINT startVertex) {
    ++m_rawDrawCount;
    SubmitDraw(false, vertexCount, startVertex, 0);
  }

  void D3D11Rtx::OnDrawIndexed(UINT indexCount, UINT startIndex, INT baseVertex) {
    ++m_rawDrawCount;
    SubmitDraw(true, indexCount, startIndex, baseVertex);
  }

  void D3D11Rtx::OnDrawInstanced(UINT vertexCountPerInstance, UINT instanceCount, UINT startVertex, UINT startInstance) {
    ++m_rawDrawCount;
    SubmitInstancedDraw(false, vertexCountPerInstance, startVertex, 0, instanceCount, startInstance);
  }

  void D3D11Rtx::OnDrawIndexedInstanced(UINT indexCountPerInstance, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance) {
    ++m_rawDrawCount;
    SubmitInstancedDraw(true, indexCountPerInstance, startIndex, baseVertex, instanceCount, startInstance);
  }

  void D3D11Rtx::SubmitInstancedDraw(bool indexed, UINT count, UINT start, INT base,
                                      UINT instanceCount, UINT startInstance) {
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
        // Find per-instance UINT semantic (bone indices)
        for (const auto& s : semantics) {
          if (!s.perInstance) continue;
          if (s.format == VK_FORMAT_R16G16B16A16_UINT) {
            boneIdxSem = &s;
            break;
          }
        }
        // Check if VS SRV t30 is bound (g_boneMatrix)
        const uint32_t kBoneSrvSlot = 30;
        if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
          boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();

        if (boneIdxSem && boneSrv) {
          // Get the bone matrix buffer
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          DxvkBufferSlice boneBufSlice = boneBuf ? boneBuf->GetBufferSlice() : DxvkBufferSlice();
          const uint8_t* bonePtr = boneBufSlice.defined() ?
            reinterpret_cast<const uint8_t*>(boneBufSlice.mapPtr(0)) : nullptr;
          const size_t boneBufLen = boneBufSlice.defined() ? boneBufSlice.length() : 0;

          // Get the per-instance index buffer (slot 1)
          const auto& instVb = m_context->m_state.ia.vertexBuffers[boneIdxSem->inputSlot];
          if (instVb.buffer != nullptr && bonePtr != nullptr && false) {
            // CPU bone readback path — disabled, both buffers are GPU-only
            DxvkBufferSlice instSlice = instVb.buffer->GetBufferSlice(instVb.offset);
            const uint8_t* instPtr = reinterpret_cast<const uint8_t*>(instSlice.mapPtr(0));
            const size_t instLen = instSlice.length();
            const uint32_t instStride = instVb.stride;

            // Also read the per-instance base bone offset from a second per-instance semantic
            // (v5.y in the shader = base offset added to bone indices)
            // For now, assume base offset = 0 and use the first bone index directly.

            if (instPtr && instStride >= 4) {
              const UINT maxInstances = std::min(instanceCount, RtxOptions::maxInstanceSubmissions());

              static uint32_t sBoneInstLog = 0;
              if (sBoneInstLog < 5) {
                ++sBoneInstLog;
                Logger::info(str::format(
                  "[D3D11Rtx] Source bone instancing: ", instanceCount, " instances",
                  " boneSlot=", boneIdxSem->inputSlot,
                  " instStride=", instStride,
                  " boneBufLen=", boneBufLen));
              }

              for (UINT i = 0; i < maxInstances; ++i) {
                UINT instIdx = startInstance + i;
                size_t instOff = static_cast<size_t>(instIdx) * instStride + boneIdxSem->byteOffset;
                if (instOff + 6 > instLen) continue;  // need 3 uint16

                const uint16_t* boneIndices = reinterpret_cast<const uint16_t*>(instPtr + instOff);
                // Use the first bone index (primary bone for this instance)
                uint32_t boneIdx = boneIndices[0];
                size_t boneOff = static_cast<size_t>(boneIdx) * 48;  // stride=48 for float3x4
                if (boneOff + 48 > boneBufLen) continue;

                // Read the float3x4 bone matrix (3 rows × 4 floats)
                const float* m = reinterpret_cast<const float*>(bonePtr + boneOff);
                // float3x4 row-major: row0=[m[0..3]], row1=[m[4..7]], row2=[m[8..11]]
                // Translation in column 3: (m[3], m[7], m[11])
                bool valid = true;
                for (int j = 0; j < 12; ++j) {
                  if (!std::isfinite(m[j])) { valid = false; break; }
                }
                if (!valid) continue;

                Matrix4 instMatrix(
                  Vector4(m[0], m[1], m[2],  0.0f),
                  Vector4(m[4], m[5], m[6],  0.0f),
                  Vector4(m[8], m[9], m[10], 0.0f),
                  Vector4(m[3], m[7], m[11], 1.0f));

                SubmitDraw(indexed, count, start, base, &instMatrix);
              }
              handledAsBoneInstancing = true;
            }
          }
        }
      }

      if (!handledAsBoneInstancing && boneIdxSem && boneSrv) {
        // GPU bone path: both buffers are GPU-only. Submit each instance
        // separately with instanceIndex as boneIndex. The interleaver will
        // read the bone index from the GPU instance buffer and fetch the
        // bone matrix from t30 on the GPU.
        const UINT maxInstances = std::min(instanceCount, RtxOptions::maxInstanceSubmissions());
        static uint32_t sGpuInstLog = 0;
        if (sGpuInstLog < 3) {
          ++sGpuInstLog;
          Logger::info(str::format(
            "[D3D11Rtx] GPU bone instancing: ", instanceCount, " instances",
            " (max ", maxInstances, ")"));
        }
        // All instances share bone 0 (COLOR1.y=0 for all). No need to
        // expand per-instance — submit once with bone 0.
        SubmitDraw(indexed, count, start, base);
        handledAsBoneInstancing = true;
      }

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
    m_currentDrawIsBoneTransformed = false;
    m_skipViewMatrixScan = false;

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
    if (projSlot == UINT32_MAX) {
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;
      const uint32_t kSourceCamSlot = 2;
      const size_t   kSourceCamOff  = 96;
      const auto& srcCb = vsCbs[kSourceCamSlot];
      if (srcCb.buffer != nullptr) {
        const auto mapped = srcCb.buffer->GetMappedSlice();
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
        if (ptr && srcCb.buffer->Desc()->ByteWidth >= kSourceCamOff + 64) {
          Matrix4 raw = readCbMatrix(ptr, kSourceCamOff, srcCb.buffer->Desc()->ByteWidth);
          int cls = classifyPerspective(raw, true);
          static uint32_t sFastLog = 0;
          if (sFastLog < 5) {
            ++sFastLog;
            Logger::info(str::format(
              "[D3D11Rtx] Source fast-path cb2@96: cls=", cls,
              " diag=(", raw[0][0], ",", raw[1][1], ",", raw[2][2], ")",
              " m23=", raw[2][3], " m33=", raw[3][3],
              " bufSize=", srcCb.buffer->Desc()->ByteWidth,
              " mapPtr=", (mapped.mapPtr != nullptr ? 1 : 0)));
          }
          if (cls > 0) {
            projSlot   = kSourceCamSlot;
            projOffset = kSourceCamOff;
            projStage  = 0;  // VS
            m_projSlot   = kSourceCamSlot;
            m_projOffset = kSourceCamOff;
            m_projStage  = 0;
            m_columnMajor = (cls == 2);
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
              // Extract camera basis vectors from VP rows (xyz only, ignoring w).
              Vector3 vpRight(proj[0][0], proj[0][1], proj[0][2]);
              Vector3 vpUp   (proj[1][0], proj[1][1], proj[1][2]);
              Vector3 vpFwd  (proj[2][0], proj[2][1], proj[2][2]);

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
              // Re-derive up from cross product so the basis is exactly orthonormal
              // (floating-point drift in the VP product can make the original up
              //  slightly non-perpendicular to right × fwd).
              Vector3 up = cross(fwd, right);
              const float upLen = length(up);
              if (upLen > 0.001f) up = up / upLen;
              else up = Vector3(0, 0, 1);
              // Re-derive right too so all three are perfectly orthonormal.
              right = cross(up, fwd);
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
                // Camera position: try c_cameraOrigin (offset 4, current frame)
                // first.  On early draws (1-249) this is zero because the engine
                // only fills it on the main opaque pass — fall back to
                // c_cameraOriginPrevFrame (offset 84, always populated alongside
                // the prevFrame VP at offset 96).
                const size_t bufSize = cb.buffer->Desc()->ByteWidth;
                bool gotCamPos = false;
                // Try current-frame camera origin at offset 4
                if (bufSize >= 16) {
                  const float* cam4 = reinterpret_cast<const float*>(ptr + 4);
                  if (std::isfinite(cam4[0]) && std::isfinite(cam4[1]) && std::isfinite(cam4[2])
                      && (std::abs(cam4[0]) > 1.0f || std::abs(cam4[1]) > 1.0f || std::abs(cam4[2]) > 1.0f)) {
                    Tx = cam4[0]; Ty = cam4[1]; Tz = cam4[2];
                    gotCamPos = true;
                  }
                }
                // Fall back to prev-frame camera origin at offset 84
                if (!gotCamPos && bufSize >= 96) {
                  const float* cam84 = reinterpret_cast<const float*>(ptr + 84);
                  if (std::isfinite(cam84[0]) && std::isfinite(cam84[1]) && std::isfinite(cam84[2])) {
                    Tx = cam84[0]; Ty = cam84[1]; Tz = cam84[2];
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
              const float dotR = -(right.x * Tx + right.y * Ty + right.z * Tz);
              const float dotU = -(up.x    * Tx + up.y    * Ty + up.z    * Tz);
              const float dotF = -(fwd.x   * Tx + fwd.y   * Ty + fwd.z   * Tz);

              transforms.worldToView = Matrix4(
                Vector4(right.x, right.y, right.z, 0.0f),
                Vector4(up.x,    up.y,    up.z,    0.0f),
                Vector4(fwd.x,   fwd.y,   fwd.z,   0.0f),
                Vector4(dotR,    dotU,    dotF,    1.0f));

              // Build a clean pure perspective projection from the extracted scales.
              const float nearZ = 1.0f;
              const float farZ  = 20000.0f;
              const float Q     = farZ / (farZ - nearZ);
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
            Vector3 up    = cross(fwd, right);
            float upLen = length(up);
            if (upLen > 0.001f) up = up / upLen; else up = Vector3(0, 0, 1);
            right = cross(up, fwd);
            float rightLen = length(right);
            if (rightLen > 0.001f) right = right / rightLen;

            // Camera position: try offset 4 (current frame), fall back to
            // offset 84 (prev frame) if current is zero (early draws).
            float Tx = 0, Ty = 0, Tz = 0;
            {
              const size_t bsz = srcCb.buffer->Desc()->ByteWidth;
              bool got = false;
              if (bsz >= 16) {
                const float* c4 = reinterpret_cast<const float*>(ptr + 4);
                if (std::isfinite(c4[0]) && std::isfinite(c4[1]) && std::isfinite(c4[2])
                    && (std::abs(c4[0]) > 1.0f || std::abs(c4[1]) > 1.0f || std::abs(c4[2]) > 1.0f)) {
                  Tx = c4[0]; Ty = c4[1]; Tz = c4[2]; got = true;
                }
              }
              if (!got && bsz >= 96) {
                const float* c84 = reinterpret_cast<const float*>(ptr + 84);
                if (std::isfinite(c84[0]) && std::isfinite(c84[1]) && std::isfinite(c84[2])) {
                  Tx = c84[0]; Ty = c84[1]; Tz = c84[2];
                }
              }
            }
            const float dotR = -(right.x*Tx + right.y*Ty + right.z*Tz);
            const float dotU = -(up.x*Tx    + up.y*Ty    + up.z*Tz);
            const float dotF = -(fwd.x*Tx   + fwd.y*Ty   + fwd.z*Tz);
            transforms.worldToView = Matrix4(
              Vector4(right.x, right.y, right.z, 0),
              Vector4(up.x,    up.y,    up.z,    0),
              Vector4(fwd.x,   fwd.y,   fwd.z,   0),
              Vector4(dotR,    dotU,    dotF,    1));

            const float perspSign = raw[2][3] < 0 ? -1.0f : 1.0f;
            const float nearZ = 1.0f, farZ = 20000.0f;
            const float Q = farZ / (farZ - nearZ);
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
          const auto& camCb = m_context->m_state.vs.constantBuffers[2];
          if (camCb.buffer != nullptr) {
            const auto mapped = camCb.buffer->GetMappedSlice();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.mapPtr);
            if (p && camCb.buffer->Desc()->ByteWidth >= 16) {
              const float* co = reinterpret_cast<const float*>(p + 4);
              camX = co[0]; camY = co[1]; camZ = co[2];
            }
          }
          // Read VP rotation from cb2@96. Use LIVE data if valid (non-identity
          // VP with proper axis magnitudes), otherwise fall back to cached
          // rotation from m_lastGoodTransforms (from a previous frame's detection).
          Vector3 right(0, -1, 0), up(0, 0, 1), fwd(1, 0, 0);  // defaults
          bool gotLiveRotation = false;
          const auto& camCb2 = m_context->m_state.vs.constantBuffers[2];
          if (camCb2.buffer != nullptr) {
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
            transforms.worldToView = Matrix4(
              Vector4(right.x, right.y, right.z, 0),
              Vector4(up.x,    up.y,    up.z,    0),
              Vector4(fwd.x,   fwd.y,   fwd.z,   0),
              Vector4(tR,      tU,      tF,      1));
          } else {
            // Fallback: fixed axis swap
            transforms.worldToView = Matrix4(
              Vector4( 0, -1,  0, 0),
              Vector4( 0,  0,  1, 0),
              Vector4( 1,  0,  0, 0),
              Vector4( 0,  0,  0, 1));
          }
          // Skip the VIEW matrix scan only — worldToView is set.
          // Allow WORLD matrix scan to extract cb3 per-draw transforms.
          m_skipViewMatrixScan = true;
        }

        // Source Engine 2: gameplay shaders use bone matrices from SRV t30,
        // NOT cb3.  cb3 is identity for these draws.  Read the bone index
        // from per-instance slot 1 (instance 0 for non-instanced draws)
        // and fetch the float3x4 bone matrix from t30 as objectToWorld.
        bool gotBoneTransform = false;
        {
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
                  // Row 0: [r00 r01 r02 tx], Row 1: [r10 r11 r12 ty], Row 2: [r20 r21 r22 tz]
                  // Translation is in world space.  worldToView (from VP
                  // decomposition, already set above) handles world → view.
                  // objectToView = worldToView * objectToWorld.
                  transforms.objectToWorld = Matrix4(
                    Vector4(m[0], m[1], m[2],  0.0f),
                    Vector4(m[4], m[5], m[6],  0.0f),
                    Vector4(m[8], m[9], m[10], 0.0f),
                    Vector4(m[3], m[7], m[11], 1.0f));
                  // worldToView already set to m_lastGoodTransforms.worldToView above
                  gotBoneTransform = true;

                  static uint32_t sBoneDiag = 0;
                  if (sBoneDiag < 30) {
                    ++sBoneDiag;
                    Logger::info(str::format(
                      "[D3D11Rtx] CPU Bone: raw=", m_rawDrawCount,
                      " inst=", m_currentInstanceIndex,
                      " boneIdx=", boneIdx,
                      " T=(", m[3], ",", m[7], ",", m[11], ")",
                      " R0=(", m[0], ",", m[1], ",", m[2], ")"));
                  }
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
            transforms.worldToView = c;
            viewCacheHit = true;
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
                transforms.worldToView = c;
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = projOffset - 64;
              }
            }
            if (isIdentityExact(transforms.worldToView)) {
              auto [vBase, vEnd] = cbRange(cb);
              for (size_t off = vBase; off + 64 <= vEnd; off += 16) {
                if (off >= projOffset && off < projOffset + 64) continue;
                Matrix4 c = readMatrix(ptr, off, bufSize);
                if (isViewMatrix(c)) {
                  transforms.worldToView = c;
                  m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
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
                transforms.worldToView = c;
                m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
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
                transforms.worldToView = flipped;
                m_viewStage = projStage; m_viewSlot = projSlot; m_viewOffset = off;
                m_columnMajor = !m_columnMajor;
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
              transforms.worldToView = c;
              m_viewStage = si; m_viewSlot = slot; m_viewOffset = off;
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
    // D3D row-major view matrix layout:
    //   [R00 R01 R02  0]    pos = -R^T * t
    //   [R10 R11 R12  0]    where t = (V[3][0], V[3][1], V[3][2])
    //   [R20 R21 R22  0]
    //   [tx  ty  tz   1]
    if (!isIdentityExact(transforms.worldToView) && !m_skipViewMatrixScan) {
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
      if (bm) {
        Matrix4 cb3Mat(
          Vector4(bm[0], bm[1], bm[2],  0.0f),
          Vector4(bm[4], bm[5], bm[6],  0.0f),
          Vector4(bm[8], bm[9], bm[10], 0.0f),
          Vector4(bm[3], bm[7], bm[11], 1.0f));
        // cb3 = fullViewMatrix × objectToWorld_real
        // objectToWorld = inverse(fullViewMatrix) × cb3
        Matrix4 invView = inverse(transforms.worldToView);
        transforms.objectToWorld = invView * cb3Mat;
        static uint32_t sO2wLog = 0;
        if (sO2wLog < 10) {
          ++sO2wLog;
          const auto& o = transforms.objectToWorld;
          const auto& c = cb3Mat;
          Logger::info(str::format(
            "[D3D11Rtx] CB3→O2W: cb3T=(", c[3][0], ",", c[3][1], ",", c[3][2], ")",
            " cb3R0=(", c[0][0], ",", c[0][1], ",", c[0][2], ")",
            " o2wT=(", o[3][0], ",", o[3][1], ",", o[3][2], ")",
            " o2wR0=(", o[0][0], ",", o[0][1], ",", o[0][2], ")"));
        }
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
        // Build the D3D row-major 4x4 with translation in the last row.
        transforms.objectToWorld = Matrix4(
          Vector4(R00, R01, R02, 0.0f),
          Vector4(R10, R11, R12, 0.0f),
          Vector4(R20, R21, R22, 0.0f),
          Vector4(Tx,  Ty,  Tz,  1.0f));
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
        return true;
      };

      bool found = false;
      const auto& vsCbs = m_context->m_state.vs.constantBuffers;

      // NV-DXVK: Source/Titanfall 2 PRIORITY PATH — try VS slot 3 as float3x4
      // FIRST, before the generic 4x4 scanner.
      // IDA confirms engine code binds slot 3 as [objectToWorld|worldToView]
      // (two float3x4 back-to-back, 96 bytes total).  We read only the first.
      // This must run before tryWorldCb's generic scan so that materialsystem's
      // slot 0/1 viewport data cannot produce false-positive world matrices.
      {
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
        if (!found) {
          for (uint32_t s = 2; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT && !found; ++s) {
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
      transforms.worldToView = Matrix4(
        Vector4(right.x, right.y, right.z, 0),
        Vector4(up.x,    up.y,    up.z,    0),
        Vector4(fwd.x,   fwd.y,   fwd.z,   0),
        Vector4(tR,      tU,      tF,      1));
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

    return transforms;
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
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::Throttle)];
      return;
    }

    // --- Cheap pre-filters: discard draws that cannot contribute to raytracing ---

    // Only triangle topologies are raytraceable. Skip points, lines, patch lists, etc.
    // This check is first: it costs a single comparison before any other state is read.
    const D3D11_PRIMITIVE_TOPOLOGY d3dTopology = m_context->m_state.ia.primitiveTopology;
    if (d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST &&
        d3dTopology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NonTriTopology)];
      return;
    }

    // Skip depth-only passes: no pixel shader means depth prepass or shadow map.
    // Most engines draw opaque geometry twice — once for depth prepass (PS == null)
    // and once for the color pass (PS != null) with the same vertices.
    if (m_context->m_state.ps.shader == nullptr) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoPixelShader)];
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
        ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoRenderTarget)];
        return;
      }
    }

    // Skip trivially small draws (< 3 elements = 0 triangles).
    if (count < 3) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::CountTooSmall)];
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
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::FullscreenQuad)];
      return;
    }

    D3D11InputLayout* layout = m_context->m_state.ia.inputLayout.ptr();
    if (!layout) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoInputLayout)];
      return;
    }

    const auto& semantics = layout->GetRtxSemantics();

    if (semantics.empty()) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoSemantics)];
      return;
    }

    const D3D11RtxSemantic* posSem = nullptr;
    const D3D11RtxSemantic* nrmSem = nullptr;
    const D3D11RtxSemantic* tcSem  = nullptr;
    const D3D11RtxSemantic* colSem = nullptr;

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
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosition)];
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
        ++m_filterCounts[static_cast<uint32_t>(FilterReason::UnsupPosFmt)];
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
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoPosBuffer)];
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

    RasterBuffer idxBuffer;
    if (indexed) {
      const auto& ib = m_context->m_state.ia.indexBuffer;
      if (ib.buffer == nullptr) {
        ++m_filterCounts[static_cast<uint32_t>(FilterReason::NoIndexBuffer)];
        return;
      }
      VkIndexType idxType = (ib.format == DXGI_FORMAT_R32_UINT)
                          ? VK_INDEX_TYPE_UINT32
                          : VK_INDEX_TYPE_UINT16;
      uint32_t idxStride = (idxType == VK_INDEX_TYPE_UINT32) ? 4 : 2;
      idxBuffer = RasterBuffer(ib.buffer->GetBufferSlice(ib.offset), 0, idxStride, idxType);
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

    // NV-DXVK: Track bone buffer for UpdateSubresource interception.
    // The per-draw transform comes from cb3 (extracted by the world matrix scan),
    // NOT from the bone system. The bone buffer is shared across all draws.
    if (posSem->format == VK_FORMAT_R32G32_UINT) {
      const uint32_t kBoneSrvSlot = 30;
      if (kBoneSrvSlot < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
        auto boneSrv = m_context->m_state.vs.shaderResources.views[kBoneSrvSlot].ptr();
        if (boneSrv) {
          Com<ID3D11Resource> boneRes;
          boneSrv->GetResource(&boneRes);
          auto* boneBuf = static_cast<D3D11Buffer*>(boneRes.ptr());
          if (boneBuf) m_lastBoneBuffer = boneBuf;
          // Don't bind bone buffers — rotation-only worldToView + cb3 handles transforms.
        }
      }
      // Per-instance bone index from slot 1
      for (const auto& s : semantics) {
        if (s.perInstance && s.format == VK_FORMAT_R16G16B16A16_UINT) {
          const auto& instVb = m_context->m_state.ia.vertexBuffers[s.inputSlot];
          if (instVb.buffer != nullptr) {
            geo.boneIndexBuffer = RasterBuffer(
              instVb.buffer->GetBufferSlice(instVb.offset),
              s.byteOffset, instVb.stride, s.format);
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
      // Without scanning the IB we don't know max(index), so use base + indexCount
      // as a conservative upper bound.
      const uint32_t baseU = static_cast<uint32_t>(std::max(base, 0));
      drawVertexCount = std::min(baseU + count, maxVBVertices);
      hashStart = std::min(baseU, maxVBVertices);
      hashCount = std::min(count, maxVBVertices - hashStart);
    }
    if (drawVertexCount == 0)
      drawVertexCount = count;
    if (hashCount == 0)
      hashCount = std::min(count, maxVBVertices);
    geo.vertexCount = drawVertexCount;

    geo.futureGeometryHashes = ComputeGeometryHashes(geo, drawVertexCount,
                                                     hashStart, hashCount);
    if (!geo.futureGeometryHashes.valid()) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::HashFailed)];
      return;
    }

    DrawCallState dcs;
    dcs.geometryData     = geo;
    dcs.transformData    = ExtractTransforms();

    // Reject NDC-space screen quads now that ExtractTransforms has cached the VP.
    if (isNdcScreenQuad) {
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::FullscreenQuad)];
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
      if (m_foundRealProjThisFrame) {
        // NV-DXVK: Guard against degenerate cached worldToView.  If the
        // translation column is all zeros the cached matrix has no real
        // camera position and submitting geometry with it produces
        // degenerate/coincident triangles that crash GPU RT hardware
        // (VK_ERROR_DEVICE_LOST).  Reject these as UIFallback instead.
        const auto& cached = m_lastGoodTransforms.worldToView;
        if (cached[3][0] == 0.0f && cached[3][1] == 0.0f && cached[3][2] == 0.0f) {
          ++m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)];
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
            ++m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)];
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
        ++m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)];
        return;
      }
    }

    // Apply per-instance world transform when submitting instanced draws.
    if (instanceTransform) {
      dcs.transformData.objectToWorld = *instanceTransform;
      // Recompute objectToView with the per-instance world matrix.
      dcs.transformData.objectToView = dcs.transformData.objectToWorld;
      if (!isIdentityExact(dcs.transformData.worldToView))
        dcs.transformData.objectToView = dcs.transformData.worldToView * dcs.transformData.objectToWorld;
    }

    // Let processCameraData() classify the camera from the matrices.
    // Hardcoding Main would bypass Remix's sky/portal/shadow detection.
    dcs.cameraType       = CameraType::Unknown;
    dcs.usesVertexShader = (m_context->m_state.vs.shader != nullptr);
    dcs.usesPixelShader  = (m_context->m_state.ps.shader != nullptr);

    // D3D11 shaders are always SM 4.0+.
    if (dcs.usesVertexShader)
      dcs.vertexShaderInfo = ShaderProgramInfo{4, 0};
    if (dcs.usesPixelShader)
      dcs.pixelShaderInfo = ShaderProgramInfo{4, 0};
    dcs.zWriteEnable     = zWriteEnable;
    dcs.zEnable          = zEnable;
    dcs.stencilEnabled   = stencilEnabled;
    dcs.drawCallID       = m_drawCallID++;

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

      if (s_submitLogFrame >= 1 && s_submitLogFrame <= 10) {
        const auto& T = dcs.transformData;
        const bool o2wIdentity = isIdentityExact(T.objectToWorld);
        Logger::info(str::format(
          "[D3D11Rtx] Submit drawID=", dcs.drawCallID,
          " frame=", s_submitLogFrame,
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
      Logger::info(str::format(
        "[D3D11Rtx] COMMIT id=", dcs.drawCallID,
        " verts=", G.vertexCount,
        " fmt=", uint32_t(G.positionBuffer.vertexFormat()),
        " stride=", G.positionBuffer.stride(),
        " bone=", G.boneMatrixBuffer.defined() ? 1 : 0,
        " inst=", G.boneInstanceIndex,
        " o2wT=(", T.objectToWorld[3][0], ",", T.objectToWorld[3][1], ",", T.objectToWorld[3][2], ")",
        " w2vT=(", T.worldToView[3][0], ",", T.worldToView[3][1], ",", T.worldToView[3][2], ")",
        " o2vT=(", T.objectToView[3][0], ",", T.objectToView[3][1], ",", T.objectToView[3][2], ")",
        " raw=", m_rawDrawCount));
    }

    m_context->EmitCs([params, dcs](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, dcs);
    });
  }

  void D3D11Rtx::OnUpdateSubresource(ID3D11Resource* pDstResource, const void* pSrcData, UINT SrcDataSize, UINT DstOffset, UINT BufSize) {
    if (!pSrcData) return;
    // Cache bone matrix buffer (t30 = 393216 bytes)
    if (BufSize == 393216 && DstOffset == 0 && SrcDataSize >= 48) {
      std::memcpy(m_cachedBone0, pSrcData, 48);
      m_hasCachedBone0 = true;
      m_lastBoneBuffer = static_cast<ID3D11Buffer*>(pDstResource);
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
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(FilterReason::Count); ++i)
      m_filterCounts[i] = 0;

    // Safety net: if no draw set the camera this frame (all filtered, empty
    // present, geometry-hash failures, etc.), push a viewport-derived camera
    // so Remix doesn't reject the frame with "not detecting a valid camera".
    // The check runs on the CS thread where the camera state is authoritative.
    //
    // NV-DXVK: Only fire the safety net when at least one RTX-bound draw
    // actually passed all the pre-filters this frame (draws > 0).  On
    // pure-UI frames (Source main menu, loading screens, video playback,
    // etc.) every draw is filtered out by the UIFallback reason, so
    // drawCallID stays zero and the RTX pipeline should not attempt to
    // composite anything.  If we fire the safety net anyway, we hand
    // injectRTX() a synthetic camera, isCameraValid becomes true, and
    // injectRTX starts path-tracing an empty scene — which compresses the
    // native backbuffer content into a tiny corner of the output.
    // Keeping the camera invalid lets injectRTX early-return (see
    // rtx_context.cpp:492) and the native raster content passes through
    // the presenter unchanged.
    if (draws > 0) {
      DrawCallTransforms t = ExtractTransforms();
      if (!isIdentityExact(t.viewToProjection)) {
        Matrix4 wtv = t.worldToView;
        Matrix4 vtp = t.viewToProjection;
        m_context->EmitCs([wtv, vtp](DxvkContext* ctx) {
          RtxContext* rtx = static_cast<RtxContext*>(ctx);
          auto& camMgr = rtx->getSceneManager().getCameraManager();
          auto& mainCam = camMgr.getCamera(CameraType::Main);
          const uint32_t frameId = rtx->getDevice()->getCurrentFrameId();
          if (!mainCam.isValid(frameId)) {
            Logger::info(str::format("[D3D11Rtx] Camera safety net fired: frameId=", frameId));
            camMgr.processExternalCamera(CameraType::Main, wtv, vtp);
          }
        });
      }
    }

    m_drawCallID = 0;
    m_rawDrawCount = 0;
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
