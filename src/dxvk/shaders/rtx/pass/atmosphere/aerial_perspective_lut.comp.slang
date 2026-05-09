/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*/

// NV-DXVK [AerialPerspective]: 3D LUT precomputing in-scattered radiance
// + transmittance for arbitrary view direction and distance through the
// atmosphere. Sampled per-shading-point in the path tracer to apply
// atmospheric haze on geometry, decoupled from the visible-sky source.
// This is what makes Hillaire's physics complement TF2's authored sky
// instead of replacing it.
//
// LUT parameterization (32x32x32):
//   X axis (32): cos(angle between view dir and sun dir)  in [-1..1]
//                stored as 0..1 via (mu_view_sun + 1) * 0.5
//   Y axis (32): cos(angle between view dir and zenith)   in [-1..1]
//                stored as 0..1 via (mu_view_zenith + 1) * 0.5
//   Z axis (32): distance along ray, exponentially mapped to bias
//                resolution toward near values where most game geometry
//                lives.  T_z in [0..1]:
//                  distance = pow(T_z, 2.0) * maxDistanceKm
//
// Voxel format (RGBA16F):
//   .rgb = accumulated in-scattered radiance from camera to distance
//   .a   = mono transmittance (luminance of RGB transmittance) so we
//          can pack into one channel; the consumer computes per-channel
//          T analytically from extinction*distance for proper colour
//          attenuation.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
Texture2D<float4> multiscatteringLutInput;

layout(binding = 3)
SamplerState lutSampler;

layout(binding = 4, rgba16f)
RWTexture3D<float4> aerialPerspectiveLutOutput;

// Convert voxel index to atmospheric-ray parameters.
//   tx [0..1] -> mu_view_sun in [-1..1]
//   ty [0..1] -> mu_view_zenith in [-1..1]
//   tz [0..1] -> distance in [0..maxDistanceKm], exponentially biased
void voxelToParams(vec3 t, AtmosphereArgs args,
                   out float muViewSun, out float muViewZenith,
                   out float distanceKm) {
  muViewSun    = t.x * 2.0f - 1.0f;
  muViewZenith = t.y * 2.0f - 1.0f;
  // Squared parameterization concentrates samples near the camera
  // where small distance changes matter most for foreground geometry.
  distanceKm = (t.z * t.z) * args.aerialPerspectiveMaxDistanceKm;
}

// Reconstruct a representative view direction in world space from
// (mu_view_sun, mu_view_zenith) using a sun-aligned coordinate frame.
// We pick a deterministic azimuth so the LUT entry corresponds to a
// single ray; consumers re-derive from the actual view direction at
// sample time.
vec3 paramsToViewDir(float muViewSun, float muViewZenith,
                     vec3 sunDir, vec3 up) {
  // Build sun-aligned basis: forward = sunDir, right and up perpendicular.
  vec3 right = normalize(cross(up, sunDir));
  vec3 sunUp = cross(sunDir, right);

  // We have two cosines that don't fully determine direction (need an
  // azimuth around the sun axis). Use the projection onto the local
  // "up" direction to disambiguate: viewDir = a*sunDir + b*sunUp + c*right
  // where a = muViewSun and we want viewDir.up = muViewZenith.
  // Solve: muViewSun*dot(sunDir,up) + b*dot(sunUp,up) + c*dot(right,up) = muViewZenith
  // dot(right, up) = 0 by construction. dot(sunUp, up) is a known scalar.
  // Constraint: |viewDir| = 1, i.e. muViewSun^2 + b^2 + c^2 = 1.
  float sunDotUp = dot(sunDir, up);
  float sunUpDotUp = dot(sunUp, up);

  float aSq = muViewSun * muViewSun;
  float remainingSq = max(0.0f, 1.0f - aSq);
  float remaining = sqrt(remainingSq);

  // b: component along sunUp such that the projection onto world up
  // equals muViewZenith - muViewSun*sunDotUp.
  float bRaw = (sunUpDotUp != 0.0f)
    ? (muViewZenith - muViewSun * sunDotUp) / sunUpDotUp
    : 0.0f;
  float b = clamp(bRaw, -remaining, remaining);
  float c = sqrt(max(0.0f, remainingSq - b * b));

  return normalize(muViewSun * sunDir + b * sunUp + c * right);
}

// Raymarch along a single view ray from the camera (at world up*altitude)
// out to distanceKm, accumulating in-scattered radiance and integrated
// transmittance.  Returns vec4(inscatter.rgb, mono_transmittance).
vec4 raymarchAerialPerspective(vec3 viewDir, float distanceKm,
                               AtmosphereArgs args) {
  // Camera position: ground level + viewAltitude (already in km).
  vec3 cameraPos = vec3(0.0f, args.viewAltitude, 0.0f);
  vec3 sunDir = args.sunDirection;

  // Number of steps: more for longer rays, capped for cost.
  // 16 samples is enough for smooth aerial perspective up to 32 km.
  const int kMaxSteps = 16;
  int numSamples = max(2, min(kMaxSteps, int(ceil(distanceKm * 0.5f))));
  float dt = distanceKm / float(numSamples);

  vec3 L = vec3(0.0f);
  vec3 transmittance = vec3(1.0f);

  float cosTheta = dot(viewDir, sunDir);
  float phaseR = rayleighPhase(cosTheta);
  float phaseM = hgPhase(cosTheta, args.mieAnisotropy);

  for (int i = 0; i < kMaxSteps; ++i) {
    if (i >= numSamples) break;

    float t = (float(i) + 0.5f) * dt;
    vec3 samplePos = cameraPos + viewDir * t;

    // Altitude above sea level (clamped — the camera often sits at
    // 0 altitude in our local frame so altitude == samplePos.y).
    float altitude = max(0.0f, samplePos.y);
    if (altitude > args.atmosphereThickness) break;

    float densityR  = rayleighDensity(altitude, args);
    float densityM  = mieDensity(altitude, args);
    float densityO3 = ozoneDensity(altitude, args);

    vec3 scatteringR = args.rayleighScattering * densityR;
    vec3 scatteringM = args.mieScattering * densityM;
    vec3 scattering  = scatteringR + scatteringM;
    vec3 ozoneAbsorb = args.ozoneAbsorption * densityO3;
    vec3 extinction  = scattering + ozoneAbsorb;

    // Approximate sun transmittance to the sample using the existing
    // analytical Kasten-Young air mass model (matches sky_view_lut).
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    float sunZenithCosAngle = dot(up, sunDir);
    float airMass;
    if (sunZenithCosAngle > 0.01f) {
      float sunElevDeg = degrees(acos(clamp(sunZenithCosAngle, -1.0f, 1.0f)));
      airMass = 1.0f / (sunZenithCosAngle + 0.15f * pow(93.885f - sunElevDeg, -1.253f));
    } else {
      float horizonAirMass = 40.0f;
      airMass = horizonAirMass * exp(-sunZenithCosAngle * 10.0f);
    }
    airMass = min(airMass, 200.0f);
    float opticalDepthToSun = args.rayleighScaleHeight
                            * exp(-altitude / args.rayleighScaleHeight)
                            * airMass;
    float ozonePathLength = ozoneDensity(args.ozoneLayerAltitude, args) * airMass;
    vec3 sunTransmittance = exp(-args.rayleighScattering * opticalDepthToSun
                              - args.ozoneAbsorption * ozonePathLength * 0.15f);

    // Single-scatter contribution at this sample, weighted by phase.
    vec3 sampleScatter = sunTransmittance
                       * (scatteringR * phaseR + scatteringM * phaseM)
                       * args.sunIlluminance;

    // Riemann-sum integration with exponential transmittance.
    // Standard Hillaire form: L += T * S * dt where S is in-scatter.
    vec3 stepTransmittance = exp(-extinction * dt);
    L += transmittance * (sampleScatter - sampleScatter * stepTransmittance) / max(extinction, vec3(1e-6f));
    transmittance *= stepTransmittance;
  }

  // Mono transmittance via luminance weighting (Rec. 709).
  float monoT = dot(transmittance, vec3(0.2126f, 0.7152f, 0.0722f));

  return vec4(L, monoT);
}

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;
  uint dim = max(1u, args.aerialPerspectiveLutSize);

  if (any(threadId >= uint3(dim, dim, dim))) {
    return;
  }

  // Voxel center in normalised LUT coords [0..1].
  vec3 t = (vec3(threadId) + 0.5f) / vec3(float(dim));

  float muViewSun, muViewZenith, distanceKm;
  voxelToParams(t, args, muViewSun, muViewZenith, distanceKm);

  vec3 viewDir = paramsToViewDir(muViewSun, muViewZenith,
                                  args.sunDirection,
                                  vec3(0.0f, 1.0f, 0.0f));

  vec4 result = raymarchAerialPerspective(viewDir, distanceKm, args);

  aerialPerspectiveLutOutput[threadId] = result;
}
