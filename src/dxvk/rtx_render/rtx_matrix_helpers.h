/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "MathLib/MathLib.h"
#include "../util/util_matrix.h"

#include "../../lssusd/usd_include_begin.h"
#include <pxr/base/arch/math.h>
#include "../../lssusd/usd_include_end.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <intrin.h>  // _ReturnAddress (MSVC intrinsic)

static inline void copyDxvkMatrix4ToDouble4x4(const dxvk::Matrix4& src, double dest[][4]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      //Convert from floats to doubles for USD
      dest[i][j] = src[i][j];
    }
  }
};

static inline void copyDxvkMatrix4ToFloat4x4(const dxvk::Matrix4& src, float dest[][4]) {
  memcpy(&dest[0][0], &src, sizeof(float)*16);
};

static inline void decomposeProjection(const dxvk::Matrix4& matrix, float& aspectRatio, float& fov, float& nearPlane, float& farPlane, float& shearX, float& shearY, bool& isLHS, bool& isReverseZ, bool log = false) {
  // [NaNGuard] Validate input before calling MathLib's DecomposeProjection.
  // DecomposeProjection internally calls MvpToPlanes which calls Sqrt
  // (MathLib.h:1918) — that Sqrt's `x >= 0` debug-assert fires on NaN
  // (NaN >= 0 is false). Inf is NOT a problem (Inf >= 0 is true, and
  // reverse-Z infinite-far projections legitimately use Inf for far —
  // TF2 does this). So reject NaN ONLY here.
  //
  // Wall-clock gated: first 10 hits log immediately (catches short
  // sessions on low-FPS builds), then once per 500ms. The `since=` field
  // shows skip count between logs.
  {
    bool hasNaN = false;
    for (int r = 0; r < 4 && !hasNaN; ++r) {
      for (int c = 0; c < 4; ++c) {
        if (std::isnan(matrix[r][c])) { hasNaN = true; break; }
      }
    }
    if (hasNaN) {
      static uint64_t sNaNGuardN          = 0;
      static uint64_t sNaNGuardLastLogN   = 0;
      static std::chrono::steady_clock::time_point sNaNGuardLastLogT
        = std::chrono::steady_clock::time_point::min();
      ++sNaNGuardN;
      const auto now = std::chrono::steady_clock::now();
      const bool firstFew = sNaNGuardN <= 10;
      const bool wallClockDue =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          now - sNaNGuardLastLogT).count() >= 500;
      if (firstFew || wallClockDue) {
        const uint64_t since = sNaNGuardN - sNaNGuardLastLogN;
        sNaNGuardLastLogN = sNaNGuardN;
        sNaNGuardLastLogT = now;
        // Caller address: with /Oi off this is the return-to-caller PC.
        // Match it against the .map file to identify the call site.
        const void* caller = _ReturnAddress();
        dxvk::Logger::warn(dxvk::str::format(
          "[decomposeProjection.NaNGuard] n=", sNaNGuardN,
          " since=", since,
          " caller=0x", reinterpret_cast<uintptr_t>(caller),
          " m0=(", matrix[0][0], ",", matrix[0][1], ",", matrix[0][2], ",", matrix[0][3], ")",
          " m1=(", matrix[1][0], ",", matrix[1][1], ",", matrix[1][2], ",", matrix[1][3], ")",
          " m2=(", matrix[2][0], ",", matrix[2][1], ",", matrix[2][2], ",", matrix[2][3], ")",
          " m3=(", matrix[3][0], ",", matrix[3][1], ",", matrix[3][2], ",", matrix[3][3], ")"));
      }
      // Safe defaults: zero output. Callers downstream check
      // std::isfinite on these fields before using them (see e.g.
      // scorePerspective at d3d11_rtx.cpp:3185, camera_manager.cpp:226),
      // so zero short-circuits to "invalid camera, skip this draw".
      aspectRatio = 0.0f;
      fov         = 0.0f;
      nearPlane   = 0.0f;
      farPlane    = 0.0f;
      shearX      = 0.0f;
      shearY      = 0.0f;
      isLHS       = false;
      isReverseZ  = false;
      return;
    }
  }

  float cameraParams[PROJ_NUM];
  float4x4 cameraMatrix;
  // Check size since struct padding can impacy this memcpy
  assert(sizeof(matrix) == sizeof(cameraMatrix));
  memcpy(&cameraMatrix, &matrix, sizeof(matrix));
  uint32_t flags;
  DecomposeProjection(NDC_D3D, NDC_D3D, float4x4(cameraMatrix), &flags, cameraParams, nullptr, nullptr, nullptr, nullptr);
  // Extract the FOV and aspect ratio from the projection matrix
  aspectRatio = matrix[0][0] * matrix[1][1] > 0.0f ? cameraParams[PROJ_ASPECT] : -cameraParams[PROJ_ASPECT];
  // Note: FoV represents the vertical FoV in radians (as opposed to PROJ_FOVX which is the horizontal FoV).
  fov = cameraParams[PROJ_FOVY];
  nearPlane = cameraParams[PROJ_ZNEAR];
  farPlane = cameraParams[PROJ_ZFAR];
  shearX = cameraParams[PROJ_DIRX];
  shearY = cameraParams[PROJ_DIRY];
  isLHS = (flags & PROJ_LEFT_HANDED) ? 1 : 0;
  isReverseZ = (flags & PROJ_REVERSED_Z) ? 1 : 0;

#ifdef _DEBUG
  if (log) {
    dxvk::Logger::info(dxvk::str::format("Projection Info: \n\tFlags: ", flags,
                                         "\n\tPROJ_ZNEAR: ", cameraParams[PROJ_ZNEAR],
                                         "\n\tPROJ_ZFAR: ", cameraParams[PROJ_ZFAR],
                                         "\n\tPROJ_ASPECT: ", cameraParams[PROJ_ASPECT],
                                         "\n\tPROJ_FOVX: ", cameraParams[PROJ_FOVX],
                                         "\n\tPROJ_FOVY: ", cameraParams[PROJ_FOVY],
                                         "\n\tPROJ_MINX: ", cameraParams[PROJ_MINX],
                                         "\n\tPROJ_MAXX: ", cameraParams[PROJ_MAXX],
                                         "\n\tPROJ_MINY: ", cameraParams[PROJ_MINY],
                                         "\n\tPROJ_MAXY: ", cameraParams[PROJ_MAXY],
                                         "\n\tPROJ_DIRX: ", cameraParams[PROJ_DIRX],
                                         "\n\tPROJ_DIRY: ", cameraParams[PROJ_DIRY],
                                         "\n\tPROJ_ANGLEMINX: ", cameraParams[PROJ_ANGLEMINX],
                                         "\n\tPROJ_ANGLEMAXX: ", cameraParams[PROJ_ANGLEMAXX],
                                         "\n\tPROJ_ANGLEMINY: ", cameraParams[PROJ_ANGLEMINY],
                                         "\n\tPROJ_ANGLEMAXY: ", cameraParams[PROJ_ANGLEMAXY]));
  }
#endif
}

struct DecomposeProjectionParams {
  float fov;
  float aspectRatio;
  float nearPlane;
  float farPlane;
  float shearX;
  float shearY;
  bool isLHS;
  bool isReverseZ;
};

static inline void decomposeProjection(const dxvk::Matrix4& matrix, DecomposeProjectionParams& result, bool log = false) {
  decomposeProjection(matrix, result.aspectRatio, result.fov, result.nearPlane, result.farPlane, result.shearX, result.shearY, result.isLHS, result.isReverseZ, log);
}
