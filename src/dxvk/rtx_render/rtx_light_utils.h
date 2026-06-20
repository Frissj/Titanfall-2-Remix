/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_cb_types.h"
#include "../util/util_vector.h"

namespace dxvk {
  struct LightUtils {
    // Function to calculate a radiance value from a light.
    // This function will determine the distance from `light` that the brightness would fall below kLegacyLightEndValue, based on the attenuation function.
    // If the light would never attenuate to less than kLegacyLightEndValue, light.Range will be used instead.
    // It will then determine how bright the replacement light needs to be to have a brightness of kNewLightEndValue at the same distance.
    static float calculateIntensity(const RtxLegacyLight& light, const float radius);

    // Variant of calculateIntensity but also combine that intensity with the original light's diffuse color to determine the radiance.
    static Vector3 calculateRadiance(const RtxLegacyLight& light, const float radius);

    // NV-DXVK [EngineDerivedLighting]: engine-faithful sphere-light intensity for
    // TF2 s_globalLights point/spot lights. Returns color/(pi*r^2) (clamped to the
    // lightConversionMaxIntensity firefly ceiling) so the path tracer's physical
    // inverse-square reproduces the engine's own falloff with no D3D9-legacy
    // endDistance solve / kDistanceSqToRadiance / lightConversionIntensityFactor.
    // `emitterRadiusRaw` is the engine emitter radius in raw (pre-sceneScale) units.
    static float calculateEngineFaithfulIntensity(float originalBrightness, float emitterRadiusRaw);

    // Best fit light transform for a given legacy light
    static Matrix4 getLightTransform(const RtxLegacyLight& light);
  };
} // namespace dxvk
