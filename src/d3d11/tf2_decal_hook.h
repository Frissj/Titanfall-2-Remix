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
  //
  // CAVEAT (2026-07-30, from IDA): sub_1801B4330 is NOT decal-only. It is
  // slot 14 of the vtable at engine.dll+0x604448, whose interface-name
  // string is "StaticPropMgrClient005" — i.e. it is the static prop
  // manager's MAIN-PASS draw command, which draws props AND their decals
  // merged in one sorted list (the merge is gated by the engine cvar
  // staticProp_drawDecalsInSortOrder). So this returns true for ordinary
  // static prop draws too, not just decals. See the header comment block
  // above for the original — and now known to be too strong — "zero false
  // positives" claim.
  bool IsInDecalRender();

}  // namespace tf2_decal_hook

// NV-DXVK [tf2_engine_cvars]: write TF2 engine.dll ConVar values directly.
//
// Titanfall2 retail has no usable developer console, so the engine cvars
// that gate static prop visibility cannot be swept the normal way. They
// are, however, plain ConVar objects at fixed RVAs in engine.dll's .data,
// all registered with flags = 0 (no FCVAR_CHEAT, no FCVAR_DEVELOPMENTONLY),
// so writing them is just a memory store into an already-writable section.
//
// Layout verified from engine.dll's own ConVar constructor (sub_180416A40):
//   +0x18 m_pszName   +0x28 m_nFlags   +0x38 m_pParent (self for a root
//   cvar)   +0x58 m_fValue (float)   +0x5C m_nValue (int)
//
// Every write validates m_pszName against the expected string first, so a
// different engine.dll build fails closed (logs and skips) instead of
// corrupting whatever happens to live at that RVA.
namespace tf2_engine_cvars {

  // Apply whatever rtx.conf currently asks for. Cheap and idempotent —
  // intended to be called once per engine frame so the values can be
  // swept at runtime via rtx.conf without a rebuild. Values are re-applied
  // every call because the engine resets cvars across map loads.
  void ApplyOverrides();

}  // namespace tf2_engine_cvars
