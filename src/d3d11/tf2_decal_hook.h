#pragma once

// NV-DXVK [tf2_decal_hook]: replace the AutoDecal heuristic in d3d11_rtx.cpp
// with a TF2-specific runtime hook on engine.dll's decal-render function
// (sub_1801B4330, signature-located at startup). The hook flips a thread-
// local "we are currently inside TF2's decal-render call tree" counter on
// entry and clears it on return; the AutoDecal block then becomes a single
// `if (IsInDecalRender())` check instead of the brittle blend-state pattern
// match that was misfiring on world geometry / view model / sky mountain.
//
// Why a hook and not a categoriser based on shader hash:
//   - The blend-state pattern (DepthBias<0 + DepthWrite=0 + ONE,INV_SRC_ALPHA)
//     matches many alpha-blended overlays in TF2, not just decals. Coverage
//     probe confirmed this — 21M+ pixels/frame of world geo were getting
//     flagged.
//   - The decal-emit function IS unambiguous in callstacks: only true decals
//     reach EmitCs via engine.dll's R_DrawDecals-equivalent (sub_1801B4330).
//     Bracketing that call with a TLS flag gives an exact per-thread signal
//     with zero false positives.
//
// The hook is best-effort: if the AOB scan fails (e.g. engine.dll patched
// by another mod, version mismatch), IsInDecalRender() always returns
// false and the caller falls through to whatever fallback the user
// configures (currently: do nothing, since the heuristic was the bug).

namespace tf2_decal_hook {

  // One-time install. Safe to call from any thread on any frame; the
  // first call wins via std::call_once. Returns whether the hook is now
  // active (true) or installation failed/was skipped (false). After a
  // successful first call, subsequent calls just return the cached state.
  bool EnsureInstalled();

  // Thread-local: are we currently inside TF2's decal-render call tree?
  // Cheap (single TLS load); safe to call from any draw site.
  bool IsInDecalRender();

}  // namespace tf2_decal_hook
