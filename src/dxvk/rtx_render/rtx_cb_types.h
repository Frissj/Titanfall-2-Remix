#pragma once

// Constant buffer layout types for the RTX shared backend.
// GPU-facing struct layouts for vertex capture, VS/PS state, and per-slot
// pixel shader data.  Binary layout must stay in sync with the corresponding
// HLSL/Slang constant buffers.

#include "../util/util_matrix.h"
#include <array>
#include <cstdint>

namespace dxvk {

  // Sentinel for unbound resource slots (textures, samplers, etc.).
  constexpr uint32_t kInvalidResourceSlot = 0;

  // --------------------------------------------------------------------------
  // Vertex capture CB — programmable VS path
  // --------------------------------------------------------------------------
  struct RtxVertexCaptureData {
    Matrix4 normalTransform;
    Matrix4 customWorldToProjection;
    Matrix4 invProj;
    Matrix4 viewToWorld;
    Matrix4 worldToObject;
    uint32_t baseVertex = 0;
    float jitterX;
    float jitterY;
    uint32_t padding;
  };

  // --------------------------------------------------------------------------
  // Viewport info — needed for pre-transformed vertex support
  // --------------------------------------------------------------------------
  struct RtxViewportInfo {
    Vector4 inverseOffset;
    Vector4 inverseExtent;
  };

  // --------------------------------------------------------------------------
  // Legacy light — populated from D3D11 constant buffers by the bridge.
  // --------------------------------------------------------------------------
  enum RtxLegacyLightType : uint32_t {
    RtxLegacyLightType_Point       = 1,
    RtxLegacyLightType_Spot        = 2,
    RtxLegacyLightType_Directional = 3,
  };

  struct RtxLegacyLight {
    Vector4 Diffuse;
    Vector4 Specular;
    Vector4 Ambient;

    Vector4 Position;
    Vector4 Direction;

    uint32_t Type;     // RtxLegacyLightType
    float Range;
    float Falloff;
    float Attenuation0;
    float Attenuation1;
    float Attenuation2;
    float Theta;       // FULL inner cone angle in radians (D3D9 convention)
    float Phi;         // FULL outer cone angle in radians (D3D9 convention)
    // NV-DXVK [EngineLightsCapture]: optional physical emitter radius
    // (TF2 HardwareLight.emitterRadius at offset 92). When > 0 it
    // overrides LightManager::lightConversionSphereLightFixedRadius in
    // createFromPointSpot. For a path tracer this is THE field that
    // controls soft shadow penumbra width — a 0.5-unit lamp bulb gives
    // sharp shadows, a 50-unit area emitter gives broad soft penumbras.
    // Default 0 = use the legacy global radius.
    float EmitterRadius = 0.0f;
    // NV-DXVK [perf]: optional STABLE identity hash. When non-zero, the legacy
    // conversion (LightData::createFromPointSpot) keys the light's dedup hash on
    // this instead of its world position. Engine lights set it from their
    // proven-stable s_globalLights slot index so MOVING lights dedup by slot
    // (position-hash churns every frame -> pile-up -> VRAM OOM). 0 = position
    // hash (USD / remixapi / non-engine callers keep legacy behaviour).
    uint64_t forceHash = 0;
  };

  // --------------------------------------------------------------------------
  // Material — API-agnostic surface properties.
  // --------------------------------------------------------------------------
  struct RtxMaterial {
    float Diffuse[4];
    float Ambient[4];
    float Specular[4];
    float Emissive[4];
    float Power;
  };

  // --------------------------------------------------------------------------
  // VS constant buffer — transforms, lights, material for vertex shaders.
  // --------------------------------------------------------------------------
  constexpr uint32_t kRtxMaxEnabledLights = 8;
  constexpr uint32_t kRtxMaxTextureSlots  = 8;

  struct RtxVSConstants {
    Matrix4 World;
    Matrix4 View;
    Matrix4 WorldView;
    Matrix4 NormalMatrix;
    Matrix4 InverseView;
    Matrix4 Projection;

    std::array<Matrix4, kRtxMaxTextureSlots> TexcoordMatrices;

    RtxViewportInfo ViewportInfo;

    Vector4 GlobalAmbient;
    std::array<RtxLegacyLight, kRtxMaxEnabledLights> Lights;
    RtxMaterial Material;
    float TweenFactor;
  };

  // Named field indices for RtxVSConstants — use when building CB
  // layouts or mapping D3D11 cbuffer offsets to named transforms.
  enum class RtxVSConstantField : uint32_t {
    World = 0,
    View,
    WorldView,
    NormalMatrix,
    InverseView,
    Projection,
    TexcoordMatrices,
    ViewportInfo,
    GlobalAmbient,
    Lights,
    Material,
    TweenFactor,
    Count
  };

  // --------------------------------------------------------------------------
  // PS constant buffer — per-draw PS state extracted by the D3D11 bridge.
  // --------------------------------------------------------------------------
  struct RtxPSConstants {
    Vector4 textureFactor;       // Blend factor for texture combiner
    float   alphaReference;      // Alpha test reference value (0-255 normalized to 0-1)
    float   fogStart;            // Linear fog start distance
    float   fogEnd;              // Linear fog end distance
    float   fogDensity;          // Exponential fog density
    Vector4 fogColor;            // Fog color (RGBA)
  };

  // --------------------------------------------------------------------------
  // Shared PS per-slot data — texture scale/offset per slot.
  // --------------------------------------------------------------------------
  struct RtxSharedPS {
    struct Stage {
      float Constant[4];
      float BumpEnvMat[2][2];
      float BumpEnvLScale;
      float BumpEnvLOffset;
      float texturePreOffset;
      float textureScale;
      float texturePostOffset;
      float padding[3];
    } Stages[kRtxMaxTextureSlots];
  };

  // Named field indices into RtxSharedPS::Stage — use when programmatically
  // building or inspecting per-slot PS data in the terrain baker.
  enum class RtxPSSlotField : uint32_t {
    Constant = 0,
    BumpEnvMat0,
    BumpEnvMat1,
    BumpEnvLScale,
    BumpEnvLOffset,
    TexturePreOffset,
    TextureScale,
    TexturePostOffset,
    Count
  };

}
