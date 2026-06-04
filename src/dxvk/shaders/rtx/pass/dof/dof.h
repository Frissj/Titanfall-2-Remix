/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef DOF_H
#define DOF_H

#include "rtx/utility/shader_types.h"

// NV-DXVK [engine-post DoF, Route B]: gather depth-of-field computed from Remix's
// path-traced linear depth (m_primaryLinearViewZ). Reproduces the host game's DoF
// blend model (lerp sharp<->blurred by circle-of-confusion) but recomputes CoC
// from the path tracer's higher-quality depth. The pass reads the sharp final
// color, gathers a CoC-sized disk, and writes the result to a scratch image that
// is copied back over the final output.

#define DOF_COLOR_INPUT   0   // Sampler2D<float4> sharp color (copy of final output)
#define DOF_VIEWZ_INPUT   1   // Sampler2D<float>  primary linear viewZ (path-traced depth)
#define DOF_COLOR_OUTPUT  2   // RWTexture2D<float4> blurred result (scratch)

struct DepthOfFieldArgs {
  uint2 imageSize;
  vec2  imageSizeInverse;

  // The game's harvested c_dof / DoFParams. Layout + CoC formula verified by fxc
  // disassembly of the game's DoF pass (FS_ec047dec…):
  //   struct DoFParams { float nearDepthEnd; float3 unused3; float4 worldParams; }
  //   dofA.x = nearDepthEnd  (split depth between near and far coeffs)  [c_dof 0]
  //   dofB   = worldParams: .xy = NEAR (scale,bias), .zw = FAR (scale,bias) [c_dof 4..7]
  //   isNear = viewDepth < nearDepthEnd
  //   (scale,bias) = isNear ? worldParams.xy : worldParams.zw
  //   CoC = saturate(viewDepth*scale + bias)
  // With the harvested config (near=(-0.337,2.24), far=(0,0)) this is near-field
  // only: full blur at the camera fading to sharp by nearDepthEnd, and ZERO far
  // blur — so distance stays sharp, matching the game.
  vec4  dofA;            // c_dof[0..3]  (x = nearDepthEnd)
  vec4  dofB;            // c_dof[4..7]  (= worldParams: xy near scale/bias, zw far)

  float depthScale;      // |viewZ| * depthScale -> normalized depth [0,1] for the ramp
  float maxBlurRadius;   // hard cap on gather radius (px), independent of dofA.x
  float strength;        // 0..1 overall (0 = passthrough)
  uint  numTaps;         // gather tap count
};

#endif  // DOF_H
