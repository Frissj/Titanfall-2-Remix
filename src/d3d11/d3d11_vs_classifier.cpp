#include "d3d11_vs_classifier.h"
#include "d3d11_shader.h"
#include "d3d11_input_layout.h"

#include <cstring>

namespace dxvk {

  namespace {

    // HLSL names we recognize for the transform SRVs. Respawn's Source 2
    // shaders use `g_modelInst` (StructuredBuffer<ModelInstance>, stride 208)
    // and `g_boneMatrix` (StructuredBuffer<float3x4>, stride 48). Both are
    // declared in RDEF when present. If neither is named but the draw uses
    // per-instance/BLENDINDICES semantics, the SRVs are identified by
    // their canonical D3D slot (t31 / t30) at the dispatch site instead.
    constexpr const char* kModelInstNames[]   = { "g_modelInst" };
    constexpr const char* kBoneMatrixNames[]  = { "g_boneMatrix" };
    constexpr const char* kCbufModelInstance  = "CBufModelInstance";

    // Canonical D3D slots when RDEF misses the name (observed on TF2 VSes
    // where the shader was compiled with stripped resource names).
    constexpr uint32_t kCanonicalModelInstSlot  = 31;
    constexpr uint32_t kCanonicalBonePaletteSlot = 30;

    // VK_FORMAT_R16G16B16A16_UINT = 95 — the per-instance UINT4 semantic
    // that Source 2 BSP shaders use to index g_modelInst (usually COLOR1/I).
    constexpr uint32_t kVkFormatR16G16B16A16Uint = 95;

    bool hasPerVertexBlendIndices(const std::vector<D3D11RtxSemantic>& sems) {
      for (const auto& s : sems) {
        if (!s.perInstance
            && std::strncmp(s.name, "BLENDINDICES", 12) == 0
            && s.index == 0) {
          return true;
        }
      }
      return false;
    }

    bool hasPerInstanceUint4(const std::vector<D3D11RtxSemantic>& sems) {
      for (const auto& s : sems) {
        if (s.perInstance
            && static_cast<uint32_t>(s.format) == kVkFormatR16G16B16A16Uint) {
          return true;
        }
      }
      return false;
    }

    uint32_t lookupResourceSlot(const D3D11CommonShader* sh,
                                const char* const*       names,
                                size_t                   count) {
      if (!sh) return ~0u;
      for (size_t i = 0; i < count; ++i) {
        const uint32_t slot = sh->FindResourceSlot(names[i]);
        if (slot != ~0u) return slot;
      }
      return ~0u;
    }

  } // namespace

  D3D11VsClassification D3D11VsClassifier::classify(
      const D3D11CommonShader*              commonShader,
      const std::vector<D3D11RtxSemantic>&  semantics) {

    D3D11VsClassification out;

    // Unknown shader — no signals at all. Dispatch will filter as UI.
    if (commonShader == nullptr) {
      out.kind = D3D11VsClassification::Kind::Unknown;
      out.reason = "null_shader";
      return out;
    }

    // --- Signal gathering (cheap, all local) ---
    const bool semBlendIdx   = hasPerVertexBlendIndices(semantics);
    const bool semPerInstU4  = hasPerInstanceUint4(semantics);

    const uint32_t rdefModelInst =
        lookupResourceSlot(commonShader, kModelInstNames,
                           sizeof(kModelInstNames) / sizeof(kModelInstNames[0]));
    const uint32_t rdefBoneMat =
        lookupResourceSlot(commonShader, kBoneMatrixNames,
                           sizeof(kBoneMatrixNames) / sizeof(kBoneMatrixNames[0]));

    const auto* cbModelInst = commonShader->FindCBuffer(kCbufModelInstance);
    const bool hasCb3ModelInstance =
        (cbModelInst != nullptr) && (cbModelInst->bindSlot != ~0u);

    // --- Decision order (strongest signal first) ---
    //
    // 1) Skinned character: per-vertex BLENDINDICES is unambiguous. The VS
    //    reads bone palette per-vertex from t30 or t31 (RDEF name varies).
    //    We don't try to compute o2w here — dispatcher attaches the palette
    //    as a bone buffer and the interleaver does the weighted skinning.
    if (semBlendIdx) {
      out.kind = D3D11VsClassification::Kind::SkinnedChar;
      if (rdefBoneMat != ~0u) {
        out.bonePaletteSlot   = rdefBoneMat;
        out.bonePaletteFromRdef = true;
        out.reason = "sem_blendindices+rdef_g_boneMatrix";
      } else if (rdefModelInst != ~0u) {
        // Some skinned variants keep the palette in `g_modelInst`.
        out.bonePaletteSlot   = rdefModelInst;
        out.bonePaletteFromRdef = true;
        out.reason = "sem_blendindices+rdef_g_modelInst_as_palette";
      } else {
        out.bonePaletteSlot = kCanonicalBonePaletteSlot;
        out.reason = "sem_blendindices_canonical_t30";
      }
      return out;
    }

    // 2) Instanced BSP / prop fanout: per-instance UINT4 is the Source 2
    //    BSP-batch signal. One t31 row per instance, indexed by COLOR1.x.
    if (semPerInstU4 || rdefModelInst != ~0u) {
      out.kind = D3D11VsClassification::Kind::InstancedBsp;
      if (rdefModelInst != ~0u) {
        out.modelInstSlot    = rdefModelInst;
        out.modelInstFromRdef = true;
        out.reason = semPerInstU4
                       ? "sem_uint4+rdef_g_modelInst"
                       : "rdef_g_modelInst";
      } else {
        out.modelInstSlot = kCanonicalModelInstSlot;
        out.reason = "sem_uint4_canonical_t31";
      }
      return out;
    }

    // 3) Static world / prop: cb3 CBufModelInstance carries
    //    objectToCameraRelative. This is the PIX-verified path for the
    //    merged[Opaque][0] meshes (VS_6e3e6f28f2156ea2).
    if (hasCb3ModelInstance) {
      out.kind     = D3D11VsClassification::Kind::StaticWorld;
      out.cb3Slot  = cbModelInst->bindSlot;
      out.reason   = "rdef_cb3_CBufModelInstance";
      return out;
    }

    // 4) No recognized transform source. The dispatcher should mark the
    //    draw as UIFallback and let the rasterizer handle it.
    out.kind = D3D11VsClassification::Kind::UI;
    out.reason = "no_signals";
    return out;
  }

  const char* D3D11VsClassifier::kindName(D3D11VsClassification::Kind k) {
    switch (k) {
      case D3D11VsClassification::Kind::Unknown:       return "Unknown";
      case D3D11VsClassification::Kind::UI:            return "UI";
      case D3D11VsClassification::Kind::StaticWorld:   return "StaticWorld";
      case D3D11VsClassification::Kind::InstancedBsp:  return "InstancedBsp";
      case D3D11VsClassification::Kind::SkinnedChar:   return "SkinnedChar";
      case D3D11VsClassification::Kind::Skybox3D:      return "Skybox3D";
      case D3D11VsClassification::Kind::Skybox2D:      return "Skybox2D";
      case D3D11VsClassification::Kind::Viewmodel:     return "Viewmodel";
      case D3D11VsClassification::Kind::Particle:      return "Particle";
      case D3D11VsClassification::Kind::Sprite2D:      return "Sprite2D";
    }
    return "?";
  }

}
