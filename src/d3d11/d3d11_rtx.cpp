#include "d3d11_rtx.h"

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
    // local-to-view transform and set worldToView=identity, bypassing the
    // FusedWorldViewMode::None rejection (objectToView==objectToWorld && !identity).
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
      // No instance transform found — submit once without instance data.
      // This handles instancing used for non-transform data (colors, etc.)
      static uint32_t sNoInstXformLog = 0;
      if (sNoInstXformLog < 3) {
        ++sNoInstXformLog;
        Logger::info(str::format("[D3D11Rtx] Instanced draw (", instanceCount,
                                 " instances) has no per-instance transform (", instRows.size(),
                                 " float4 rows). Submitting single draw."));
      }
      SubmitDraw(indexed, count, start, base);
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
          const bool allowVP = (m_rawDrawCount > 50);
          int cls = classifyPerspective(m, allowVP);
          if (cls == 0) continue;
          // Column-major-as-row (cls==2 or 4): transpose to row-major for scoring/use.
          const bool isCol = (cls == 2 || cls == 4);
          Matrix4 normalized = isCol ? transpose(m) : m;
          float s = scorePerspective(normalized);
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
          const bool allowVP = (m_rawDrawCount > 50);
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

              // Camera world-space position: recoverable from VP row 3.
              // VP[3] = camPos × P.  VP[3][0] = Tx*Sx, VP[3][1] = Ty*Sy.
              // So Tx = VP[3][0] / Sx, Ty = VP[3][1] / Sy.
              // Tz = VP[3][3] / perspSign (but VP[3][3]=0 for perspective →
              // camera Z position is in the cbuffer at a known offset instead).
              // For now use the VP-derived position components.
              const float perspSign = (proj[2][3] < 0.0f) ? -1.0f : 1.0f;
              float Tx = proj[3][0] / Sx;
              float Ty = proj[3][1] / Sy;
              // Tz recovery: VP[3][2] = Tz*Q + 1*(-nQ) with nQ = near*Q.
              // Without knowing Q and near exactly we approximate from the
              // camera position data sitting at offset-16 in the same cbuffer.
              // For a first pass, try Tz from VP[3][3]/perspSign if nonzero,
              // else fall back to 0.
              float Tz = (std::abs(proj[3][3]) > 0.001f)
                       ? proj[3][3] / perspSign
                       : 0.0f;

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
              const float farZ  = 10000.0f;
              const float Q     = farZ / (farZ - nearZ);
              proj = Matrix4(
                Vector4(Sx,   0.0f, 0.0f,          0.0f),
                Vector4(0.0f, Sy,   0.0f,          0.0f),
                Vector4(0.0f, 0.0f, Q,             perspSign),
                Vector4(0.0f, 0.0f, -nearZ * Q,    0.0f));

              static bool s_vpDecompLogged = false;
              if (!s_vpDecompLogged) {
                s_vpDecompLogged = true;
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
    if (projSlot == UINT32_MAX) {
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
    if (!isIdentityExact(transforms.worldToView)) {
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
    // Scan VS cbuffers first (model matrices live in VS for virtually all engines),
    // then fall back to other stages for emulator compatibility.
    // Gated by useCBufferWorldMatrices — disable if CB layout causes wrong detections.
    if (RtxOptions::useCBufferWorldMatrices()) {
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
      if (projSlot != UINT32_MAX && projStage == 0
          && projSlot + 1 < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
        found = tryWorldCb(vsCbs, projSlot + 1, projStage, projSlot);
      if (!found) {
        for (uint32_t s = 0; s < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++s) {
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

    const XXH64_hash_t descHash   = hashGeometryDescriptor(geo.indexCount, vertexCount, indexType, topology);
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
       || pf == VK_FORMAT_R16G16B16_SFLOAT;      // 90  — half-float 3-component
      if (!supportedPosFmt) {
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
      ++m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)];
      return;
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

    m_context->EmitCs([params, dcs](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, dcs);
    });
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
        " uiFallback=",     m_filterCounts[static_cast<uint32_t>(FilterReason::UIFallback)]));
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
    // Projection cache (m_projSlot, m_projOffset, m_projStage, m_columnMajor)
    // is NOT reset — the validation path at the start of ExtractTransforms
    // re-reads and re-scans only when the cached location becomes stale.
    // Resetting every frame would force an O(stages × slots × bufferBytes)
    // scan on the first draw, which hangs emulators with 64KB+ UBOs.
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
