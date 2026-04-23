/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
*
* HUD compositor — diff-based variant for menu/loading HUD.
*
* For TF2 gameplay HUD we use a different mechanism (composite-intercept;
* see RtxContext::blitSavedRtFinalToSceneColor). This shader only runs
* for menu/loading frames where the HUD rasters directly to the
* swap-chain backbuffer. We detect HUD pixels as (scratchPost != scratchPre)
* and preserve them over the RT composite.
*/
#include "hud_composite_binding_indices.h"

layout(binding = HUD_COMPOSITE_BINDING_SCRATCH_PRE)
Texture2D<float4> scratchPre;

layout(binding = HUD_COMPOSITE_BINDING_SCRATCH_POST)
Texture2D<float4> scratchPost;

layout(binding = HUD_COMPOSITE_BINDING_RT_FINAL)
Texture2D<float4> rtFinal;

layout(binding = HUD_COMPOSITE_BINDING_HUD_TARGET)
Texture2D<float4> hudTarget;  // unused in this variant; still bound for descriptor-set compatibility

layout(binding = HUD_COMPOSITE_BINDING_OUTPUT)
RWTexture2D<float4> outputImage;

layout(push_constant)
ConstantBuffer<HudCompositeArgs> cb;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID)
{
  if (tid.x >= cb.width || tid.y >= cb.height) {
    return;
  }

  float4 pre  = scratchPre.Load(int3(tid, 0));
  float4 post = scratchPost.Load(int3(tid, 0));
  float4 rt   = rtFinal.Load(int3(tid, 0));

  float3 diff = abs(post.rgb - pre.rgb);
  bool touched = max(diff.r, max(diff.g, diff.b)) > 0.004f;

  outputImage[tid] = touched
    ? float4(post.rgb, 1.0)
    : float4(saturate(rt.rgb), 1.0);
}
