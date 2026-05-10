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
#pragma once

#include "rtx/utility/shader_types.h"

// Atmosphere parameters for Hillaire physically-based atmospheric scattering
struct AtmosphereArgs {
  vec3 sunDirection;
  float planetRadius;  // in km
  
  vec3 sunIlluminance;
  float atmosphereThickness;  // in km
  
  vec3 rayleighScattering;
  float mieAnisotropy;  // Henyey-Greenstein phase function g parameter [-1, 1]
  
  vec3 mieScattering;
  float sunRayBrightness;  // Multiplier for direct sun ray brightness
  
  // Ozone absorption (important for realistic sunset colors per Hillaire paper Section 3.4)
  vec3 ozoneAbsorption;  // Absorption coefficients (km^-1)
  float ozoneLayerAltitude;  // Peak altitude of ozone layer (km)
  
  uint transmittanceLutWidth;
  uint transmittanceLutHeight;
  uint multiscatteringLutSize;
  uint skyViewLutWidth;
  
  uint skyViewLutHeight;
  float ozoneLayerWidth;  // Width of ozone layer (km)
  float viewAltitude;     // Camera altitude offset (km)
  uint pad2;
  
  // Derived parameters (computed on CPU)
  float atmosphereRadius;  // planetRadius + atmosphereThickness
  float rayleighScaleHeight;  // exponential density falloff for Rayleigh (km)
  float mieScaleHeight;  // exponential density falloff for Mie (km)
  float sunAngularRadius; // Sun angular radius in radians

  // NV-DXVK [AerialPerspective]: 3D LUT parameters
  uint  aerialPerspectiveLutSize;       // cubic dimension (e.g. 32)
  float aerialPerspectiveMaxDistanceKm; // far plane along Z axis of the LUT (e.g. 32 km)
  float aerialPerspectiveStrength;      // user multiplier on the effect (1.0 default)
  float aerialPerspectiveWorldToKm;     // multiplier from game world units to km

  // NV-DXVK [SkyTint] artist-authored sky tint multiplier captured from
  // TF2's CBufCommonPerCamera (c_skyColor at off=256, scaled by
  // c_envMapLightScale at off=272). Path tracer multiplies Hillaire
  // IBL output by this so indirect bounces and reflections receive the
  // artist's intended overall sky colour and brightness. Defaults to
  // (1,1,1) when the snapshot is unavailable, leaving Hillaire physics
  // output unmodified.
  vec3  skyTint;
  float skyTintPad;

  // NV-DXVK [SkyTune.fog] TF2 fog params captured from c_fogParams +
  // c_fogColorFactor. fogColor is the artist's authored haze tint,
  // fogStrength is a 0..1 mix that the composite pass uses to lerp
  // Hillaire AP inscatter toward fogColor on heavy-fog maps. With
  // fogStrength == 0 the existing pure-physics AP is preserved
  // unchanged, so this is non-destructive when the snapshot doesn't
  // include valid fog data.
  vec3  fogColor;
  float fogStrength;

  // [SkyTrace.probePrefill] Side length (texels) of the SkyProbe cube
  // face, plus which face the current prefill dispatch targets. Each
  // cube-prefill dispatch binds a single-layer 2D storage view of one
  // cube face; probeFace tells the shader which face direction
  // (+X/-X/+Y/-Y/+Z/-Z) to compute world-space rays for. We use 6
  // separate dispatches instead of one 2D-array dispatch because the
  // multi-layer path produced writes only on layer 0 in this DXVK
  // build (SPIR-V looked correct but only the first layer received
  // writes — likely a driver/DXVK quirk).
  uint  probeSide;
  uint  probeFace;
  uint  probePad1;
  uint  probePad2;
};
