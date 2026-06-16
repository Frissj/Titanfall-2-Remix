/*
* NV-DXVK: single gate for RTX-layer debug probes added during PI/BSP diagnostics.
* Flip to true to enable all probes below (and the force-rebuild-every-frame dynamic
* BLAS path + the shader miss-write marker in geometry_resolver.slangh).
*
* Probes gated by this flag:
*   [PI-routing]      rtx_accel_manager.cpp — per-frame routing stats (PI vs normal)
*   [PI-dump]         rtx_point_instancer_system.cpp — per-batch CB inputs + i2o dump
*   [PI-dispatch]     rtx_accel_manager.cpp — summary per buildTlas
*   [PI-readback]     rtx_accel_manager.cpp — probe D: GPU TLAS instance byte readback
*   [PI-vbreadback]   rtx_accel_manager.cpp — probe E: BLAS position buffer readback
*   [PI-blas-validate]rtx_accel_manager.cpp — per-batch BLAS liveness / ref validation
*   [BBI] / [BBI-readback] / [BBI-serialsize]  rtx_accel_manager.cpp — build-input audit
*   [PI-override-late]rtx_accel_manager.cpp — TLAS instance override identity test
*   [VisibleSurf]     rtx_context.cpp — per-frame SharedSurfaceIndex readback + dedupe
*   Dynamic BLAS force-rebuild (bypasses `update/build` reuse logic)
*   geometry_resolver.slangh SharedSurfaceIndex-on-miss invalidation write
*/
#pragma once

namespace dxvk {
  // Log-only probes (VisibleSurf, BBI, PI-blas-validate, PI-routing, PI-dump, probe D/E).
  // Safe: do not mutate TLAS instance buffer.
  // NV-DXVK [perf]: OFF. Several of these (VisibleSurf, probe D TLAS-instance readback,
  // probe E BLAS-position readback) issue per-frame GPU->CPU readbacks = forced
  // CPU/GPU syncs that stall the frame. Re-enable only while actively debugging PI/BBI/
  // surface-visibility; the force-rebuild-every-frame path is gated separately below.
  constexpr bool kEnableRtxDebugProbes = false;
  // Destructive probes (PI-override-late, BBI-serialsize query issue, force-rebuild-every-frame).
  // Modify rendering state — can hide real bugs. Keep off by default.
  constexpr bool kEnableRtxDebugDestructiveProbes = false;
}
