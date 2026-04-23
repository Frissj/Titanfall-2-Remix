/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
*/
#pragma once

// NV-DXVK: HUD-compositor shader binding slots. See
// hud_composite.comp.slang. Reads pre-HUD + post-HUD backbuffer
// snapshots and the RT composite; writes the merged result into a
// scratch output image that is then blitted to the real backbuffer
// (since swap-chain images lack STORAGE usage and can't be written
// directly by a compute shader).
#define HUD_COMPOSITE_BINDING_SCRATCH_PRE    0
#define HUD_COMPOSITE_BINDING_SCRATCH_POST   1
#define HUD_COMPOSITE_BINDING_RT_FINAL       2
#define HUD_COMPOSITE_BINDING_HUD_TARGET     3
#define HUD_COMPOSITE_BINDING_OUTPUT         4

#ifdef __cplusplus
#include <cstdint>
struct HudCompositeArgs {
  uint32_t width;
  uint32_t height;
  uint32_t hudTargetWidth;
  uint32_t hudTargetHeight;
  uint32_t hudTargetValid;   // 1 if HUD_TARGET binding holds real data
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
#else
struct HudCompositeArgs {
  uint32_t width;
  uint32_t height;
  uint32_t hudTargetWidth;
  uint32_t hudTargetHeight;
  uint32_t hudTargetValid;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
#endif
