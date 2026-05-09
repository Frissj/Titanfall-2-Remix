/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*/
#pragma once

// NV-DXVK [EngineSunCapture]: small standalone header. We deliberately do
// NOT include rtx_atmosphere.h or rtx_resources.h here — the previous
// design routed publishEngineSunCapture/fetchEngineSunCapture through
// rtx_atmosphere.h, which dragged the entire RT pipeline state machine
// into d3d11.dll's translation unit when d3d11_rtx.cpp #included it.
// That broke engine init ("Could not load library client" before any of
// our code ran). This header pulls only util_vector.h (Vector3) plus
// stdint, so the d3d11 producer can include it cheaply.

#include "../../util/util_vector.h"
#include <cstdint>

namespace dxvk {

  // Snapshot of the game's per-frame sun direction + colour, captured by
  // the d3d11 producer (D3D11Rtx::CaptureEngineSunFromCb) from
  // CBufCommonPerCamera.c_sunDir and .c_sunColor and consumed by the
  // atmosphere (RtxAtmosphere::getAtmosphereArgs) to override the static
  // sunRotation/sunElevation/sunIlluminance sliders.
  //
  // worldDirection is in the GAME's world-space (Z-up for TF2/Source).
  // The atmosphere consumer applies the Z-up -> Y-up axis swap and the
  // optional towards-light flip based on RtxOptions::engineSunIsZUp /
  // engineSunDirIsTowardsLight before handing it to the LUTs.
  //
  // colorLinear is the literal vec3 the engine pushed; the consumer
  // multiplies it by RtxOptions::engineSunIntensityScale to bring it
  // into the same magnitude range as the slider default illuminance.
  struct EngineSunSnapshot {
    Vector3  worldDirection { 0.0f, 0.0f, 1.0f };
    Vector3  colorLinear    { 1.0f, 1.0f, 1.0f };
    uint64_t frameId        = 0;
    bool     valid          = false;
  };

  // Producer side (d3d11_rtx.cpp). Cheap to call per-draw - takes a
  // single mutex around a struct copy.
  void publishEngineSunCapture(const EngineSunSnapshot& snap);

  // Consumer side (rtx_atmosphere.cpp). Returns a copy so the caller
  // never holds the mutex.
  EngineSunSnapshot fetchEngineSunCapture();

} // namespace dxvk
