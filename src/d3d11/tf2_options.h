#pragma once

// NV-DXVK [tf2_options]: TF2/Titanfall2-specific RTX options that are NOT in
// rtx_options.h, purely to keep rebuild cost sane.
//
// WHY THIS FILE EXISTS
// --------------------
// rtx_options.h is included by src/dxvk/pch/dxvk.pch.h -- the precompiled
// header. Touching it invalidates the PCH, so adding or editing a single
// option there rebuilds ~213 translation units (measured 2026-07-30). This
// header is included by d3d11_rtx.cpp only, so editing it rebuilds one file.
//
// That matters because TF2 options are the ones that churn: they get added,
// renamed and deleted constantly during investigation, while the options in
// rtx_options.h are comparatively stable.
//
// HOW IT WORKS
// ------------
// RTX_OPTION expands to `public: inline static RtxOption<type> name = ...`,
// so it only needs to sit inside SOME class -- not inside RtxOptions
// specifically. The rtx.conf key comes from the category and name strings
// passed to the macro, so an option declared here as
//   RTX_OPTION("rtx", bool, foo, true, "...")
// is still read from and written to as `rtx.foo`, exactly as if it lived in
// rtx_options.h. The inline static initialiser self-registers with the option
// manager when this translation unit is linked in.
//
// WHEN NOT TO USE THIS
// --------------------
// If an option needs to be read from code outside src/d3d11 (rtx_render, the
// imgui menus, the shaders' host-side glue), put it in rtx_options.h and pay
// the rebuild -- reaching into this header from src/dxvk would recreate the
// coupling this file exists to avoid, and in the wrong direction.
//
// This header deliberately includes rtx_option.h rather than rtx_options.h:
// we need the macro, not the option set. Include AFTER dxvk_device.h in the
// including TU (the rtx headers have that ordering requirement -- see the
// comment near the top of d3d11_rtx.cpp).

#include "../dxvk/rtx_render/rtx_option.h"

namespace dxvk {

  struct Tf2Options {
    // ===== [tf2_engine_cvars]: direct writes to TF2 engine.dll ConVars =====
    //
    // Retail Titanfall2 has no usable developer console, so these engine
    // cvars cannot be swept in-game. They are plain ConVar objects at fixed
    // RVAs in engine.dll .data, all registered with flags = 0 (verified in
    // IDA: no FCVAR_CHEAT, no FCVAR_DEVELOPMENTONLY), so Remix can write
    // them directly. Each write verifies the ConVar's m_pszName first and
    // fails closed on mismatch.
    //
    // Why these five: [MeshTraceSite] resolved the mesh-drop emit sites to
    // engine.dll+0x1B4AD6 (39x) and +0x1B3BC8 (7x). Both are inside the
    // static prop manager -- vtable at engine.dll+0x604448, interface string
    // "StaticPropMgrClient005", slot 14 = main pass, slot 17 = early depth
    // prepass. Not world geometry and not decals; the v2 handoff's "engine.dll
    // world geometry" label is on the right addresses but the wrong system.
    // These five are every knob those two functions read.
    //
    // Engine defaults: earlyDepthPrepass=1, earlyDepthPrepassDist=1500000,
    // IncludeOpaques=1, IncludeOpaquesDist=1000, drawDecalsInSortOrder=1.
    //
    // The tight one is IncludeOpaquesDist (1000). In the prepass the per-prop
    // gate picks 1500000^2 or 1000^2 off a per-prop flag byte, and it is a
    // `while` over a distance-sorted run, so crossing the threshold terminates
    // the REST of the run rather than skipping one prop -- a burst-shaped cull,
    // which is the shape the 16-mesh bursts have.
    //
    // -1 on any of the five means "leave that engine cvar alone".
    RTX_OPTION("rtx", bool, tf2StaticPropCvarOverride, false,
               "TF2/Titanfall2 only. Master switch for writing static-prop "
               "engine.dll ConVars directly. Off = never touch engine memory.");
    RTX_OPTION("rtx", float, tf2StaticPropEarlyDepthPrepass, -1.0f,
               "TF2/Titanfall2 only. Override staticProp_earlyDepthPrepass "
               "(engine default 1). -1 = leave alone. Needs "
               "tf2StaticPropCvarOverride=True.");
    RTX_OPTION("rtx", float, tf2StaticPropEarlyDepthPrepassDist, -1.0f,
               "TF2/Titanfall2 only. Override staticProp_earlyDepthPrepassDist "
               "(engine default 1500000). -1 = leave alone. Needs "
               "tf2StaticPropCvarOverride=True.");
    RTX_OPTION("rtx", float, tf2StaticPropIncludeOpaques, -1.0f,
               "TF2/Titanfall2 only. Override "
               "staticProp_earlyDepthPrepassIncludeOpaques (engine default 1). "
               "-1 = leave alone. Needs tf2StaticPropCvarOverride=True.");
    RTX_OPTION("rtx", float, tf2StaticPropIncludeOpaquesDist, -1.0f,
               "TF2/Titanfall2 only. Override "
               "staticProp_earlyDepthPrepassIncludeOpaquesDist (engine default "
               "1000 -- the tight, burst-shaped prop distance cull). -1 = "
               "leave alone. Needs tf2StaticPropCvarOverride=True.");
    RTX_OPTION("rtx", float, tf2StaticPropDrawDecalsInSortOrder, -1.0f,
               "TF2/Titanfall2 only. Override "
               "staticProp_drawDecalsInSortOrder (engine default 1). -1 = "
               "leave alone. Needs tf2StaticPropCvarOverride=True.");

    // NOTE: other TF2 options still live in rtx_options.h. Moving them is a
    // behaviour-free but rebuild-heavy change, so migrate opportunistically,
    // a few at a time, when nothing else is being tested. useCamGeoLatch was
    // deliberately NOT moved: it is load-bearing for the zig-zag fix, and a
    // registration surprise there would look exactly like the fix regressing.
  };

}  // namespace dxvk
