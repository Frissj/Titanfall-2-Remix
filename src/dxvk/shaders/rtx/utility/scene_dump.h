/*
* NV-DXVK: Per-pixel scene dump element. One entry per primary-ray pixel
* on the frame the user triggers a capture from ImGui. Buffer is sized to
* `downscaledExtent.width * downscaledExtent.height` and written at index
* `pixel.y * sceneDumpStride + pixel.x`. CSV format on disk:
*   pixelX, pixelY, u, v, mip, aniso, texIdx, texW, texH, pathCode, flags
*
* Layout MUST stay in lockstep between C++ and slang. Total = 32 bytes.
*
* Fields:
*   uv             : sampled UV at the primary hit (post-textureTransform)
*   mip            : log2(minor_axis * texWidth)  — what HW would pick when
*                    aniso ratio <= aniMax. Negative on under-magnification.
*   aniso          : major / minor (Jacobian SVD ratio). >aniMax forces
*                    higher mip per HW LOD rules.
*   texIdx         : bound albedo texture index. Resolve to image hash via
*                    the existing [D3D11Rtx.SampPick] log lines.
*   texDimsPacked  : (W << 16) | H of the bound albedo at mip 0.
*   pathCode       : gradient-pipeline branch (0..5, 10) — same encoding as
*                    surface_interaction.slangh DEBUG_VIEW_GRADIENT_PATH_CHECKER.
*   flags          : bit 0 = hit valid (1 if any opaque surface was hit on
*                    this pixel; 0 = sky / miss / first-hit was non-opaque).
*                    bits 1..31 reserved.
*/
#pragma once

#include "shader_types.h"

struct SceneDumpElement
{
  float u;
  float v;
  float mip;
  float aniso;

  uint  texIdx;
  uint  texDimsPacked;
  uint  pathCode;
  // flags bit 0 = hit valid; bit 1 = surface had hasLightmap set so the
  // lightmap UV columns below carry decoded TEXCOORD1 (else they mirror
  // the albedo UV per the surface_interaction.slangh fallback).
  uint  flags;

  // NV-DXVK: lightmap UV (TEXCOORD1) at the same hit. Lets the user
  // verify end-to-end that the per-instance lightmap UV interpolation
  // produced sane [0,1]-range values rather than mirroring albedo's
  // tile-magnitude UVs. Adds 8 bytes per pixel (≈8MB at 1188×668).
  float lightmapU;
  float lightmapV;

  // NV-DXVK [VanishDiag-Shader]: clip.w spread + world hit position so
  // we can correlate pathCode=1 (clip.w<=1, behind near plane) pixels
  // with their actual world geometry. minClipW < 1 means at least one
  // of {V0, V1, V2, hit} fell behind the camera near plane. hitWorld
  // tells us whether those pixels are actually the floor (vs walls/
  // sky/etc.) — answering "which surface is producing the bad path".
  // +20 bytes per pixel ≈ 16MB at 1188×668.
  float minClipW;
  float maxClipW;
  float hitWorldX;
  float hitWorldY;
  float hitWorldZ;
};
