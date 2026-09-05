/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// ============================================================================
// NV-DXVK [EngineSymbols] -- the Titanfall 2 symbol table.
//
// THIS FILE IS THE ONLY PLACE ALLOWED TO NAME A LOCATION IN A GAME MODULE.
// Everything else asks for a symbol by name. scripts-common/check_engine_rvas.py
// enforces that in CI.
//
// Each entry says HOW a thing is found on whatever build is actually running,
// never WHERE it lives on one particular build. In order of preference:
//
//   StringEnclosingFunction / StringAnchoredFunction
//       Anchored on a string literal. Strongest option available for code:
//       compiler output churns between builds (registers reallocated,
//       instructions reordered, inlining flipped), string literals do not.
//
//   Function / CodeSite / CallTarget / RipRelativeData
//       Masked byte signature. Exactly one match in .text or nothing.
//
//   pattern == nullptr
//       "Not re-established on this build." Resolves to 0, the dependent
//       feature disables itself and logs once. This is a DELIBERATE, VALID
//       state -- not a TODO to be papered over with an old RVA.
//
// -----------------------------------------------------------------------
// Why so many entries are currently unregistered
// -----------------------------------------------------------------------
// The RVAs these sites used to carry were reverse-engineered from a different
// compilation of client.dll than the one that ships with v2.0.11.0. Read back
// against the installed binary, every one of them lands mid-instruction:
//
//   site         old RVA      first bytes on v2.0.11.0    meaning
//   ---------------------------------------------------------------------
//   render list  0x1A8278     C1 E8 06 ...                shr eax, 6
//   gate call A  0x36E352     EC 30 49 ...                mid-instruction
//   gate call B  0x36E83E     01 00 4C ...                mid-instruction
//   gate call C  0x36E9CB     08 00 00 ...                mid-instruction
//   LodV10       0x26B21A     44 24 30 ...                mid-instruction
//   V9Probe      0x1A8657     48 C1 E8 06 ...             shr rax, 6
//   SetupBones   0xFE780      2B E0 83 ...                mid-instruction
//
// So these features were ALREADY dead: each one byte-checks its target and
// bails. Routing them through the resolver costs no working behaviour, makes
// the "off" state explicit and logged instead of silent, and removes the stale
// literals. When a site's identity is re-established on a build, fill in its
// signature here and it comes back everywhere at once.
//
// A signature is only worth registering when it is specific enough to be
// unambiguous. A thin one (say `33 D2 F6 46 24 0C` -- xor edx,edx; test) is
// NOT registered on purpose: the resolver fails safe on zero or multiple
// matches, but it cannot detect a single COINCIDENTAL match, and planting a
// mid-function detour on one would reproduce the original crash exactly.
// ============================================================================

#include "rtx_engine_symbols.h"

namespace dxvk {
  namespace tf2sym {

    using EngineSymbols::SymbolDesc;
    using EngineSymbols::SymbolKind;

    // ------------------------------------------------------------------
    // RESOLVED -- string-anchored, build-independent.
    // ------------------------------------------------------------------

    // client.dll's BuildRenderableRenderLists job. Registered with the job
    // system by name, so the registration site hands us both:
    //     lea rax, aBuildRenderableRenderLists   <- the anchor
    //     lea rdx, <the job function>            <- what we want
    //     call JTGuts_RegisterJobType
    inline const SymbolDesc kBuildRenderableRenderLists {
      /* name         */ "client.BuildRenderableRenderLists",
      /* moduleName   */ "client.dll",
      /* pattern      */ nullptr,
      /* anchorString */ "BuildRenderableRenderLists",
      /* kind         */ SymbolKind::StringAnchoredFunction,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 24,
      /* searchAfter  */ 24,
    };

    // C_BaseAnimating::SetupBones. Uses "SetupBonesOnBaseAnimating" internally
    // (a profiling/threading scope name), so we walk back from that reference
    // to the enclosing function entry.
    inline const SymbolDesc kSetupBones {
      /* name         */ "client.C_BaseAnimating::SetupBones",
      /* moduleName   */ "client.dll",
      /* pattern      */ nullptr,
      /* anchorString */ "SetupBonesOnBaseAnimating",
      /* kind         */ SymbolKind::StringEnclosingFunction,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0x1000,
      /* searchAfter  */ 0,
    };

    // engine.dll's R_DrawWorldMeshes. Remix trampolines its entry to capture
    // the authoritative view-setup struct (rcx), which is what drives the Main
    // camera when RtxOptions::useEngineHookMainCamera is on.
    //
    // It uses the literal "R_DrawWorldMeshes" internally as a profiling scope
    // name, referenced from exactly one place, so we walk back from that
    // reference to the enclosing function entry.
    //
    // This is a load-bearing repair, not a safety change. The old hardcoded
    // RVA 0xB7DD0 is +0xE0 INSIDE sub_1800B7CF0 on the shipped build, so the
    // trampoline's 7-byte prologue check failed, the hook never installed,
    // g_engineMainW2v was never written, and Main had no camera source at all.
    // The real entry is 0xB7F80 and does begin with the expected
    // 48 8B C4 44 89 40 18.
    //
    // Note this function is SPLIT into chained .pdata chunks -- its first
    // chunk is 0x18 bytes and the literal sits in a later one -- which is why
    // the resolver follows UNW_FLAG_CHAININFO rather than trusting the
    // containing chunk's BeginAddress.
    inline const SymbolDesc kRDrawWorldMeshes {
      /* name         */ "engine.R_DrawWorldMeshes",
      /* moduleName   */ "engine.dll",
      /* pattern      */ nullptr,
      /* anchorString */ "R_DrawWorldMeshes",
      /* kind         */ SymbolKind::StringEnclosingFunction,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0x400,
      /* searchAfter  */ 0,
    };

    // ------------------------------------------------------------------
    // UNREGISTERED -- identity not re-established on the shipped build.
    // Each resolves to 0; its feature disables itself and logs once.
    // ------------------------------------------------------------------

    inline const SymbolDesc kRenderListProbeCallSite {
      "client.RenderListProbe.callSite", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kRenderableDrawGate {
      "client.RenderableDrawGate", "client.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kRenderableDrawGateCallSiteA {
      "client.RenderableDrawGate.callSiteA", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };
    inline const SymbolDesc kRenderableDrawGateCallSiteB {
      "client.RenderableDrawGate.callSiteB", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };
    inline const SymbolDesc kRenderableDrawGateCallSiteC {
      "client.RenderableDrawGate.callSiteC", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kModelRenderDraw {
      "client.ModelRenderDraw", "client.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kModelRenderVtableOwner {
      "client.ModelRenderDraw.vtableOwnerPtr", "client.dll", nullptr, nullptr,
      SymbolKind::RipRelativeData, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kLodV10Site {
      "client.LodV10.detourSite", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kV9ProbeSite {
      "client.V9Probe.detourSite", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kVanishDiagRenderableWrapper {
      "client.VanishDiag.renderableWrapper", "client.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kVanishDiagBucketVis {
      "client.VanishDiag.bucketVisibilityTest", "client.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kStudioRenderContextPtr {
      "client.StudioRenderContextPtr", "client.dll", nullptr, nullptr,
      SymbolKind::RipRelativeData, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kEngineModelInfo {
      "engine.ModelInfoPtr", "engine.dll", nullptr, nullptr,
      SymbolKind::RipRelativeData, 0, 0, 0, 0, 0 };

    // ------------------------------------------------------------------
    // engine.dll / studiorender.dll -- all UNREGISTERED.
    //
    // Every one of these is mid-function on the shipped v2.0.11.0 engine.dll,
    // exactly like the client.dll set:
    //
    //   old RVA     lands in                        delta
    //   -----------------------------------------------------------
    //   0x1B2200    no function at all              (unmapped)
    //   0xB4870     sub_1800B4780                   +0xF0
    //   0xB84C0     sub_1800B81B0                   +0x310
    //   0x1B2476    sub_1801B2340                   +0x136
    //   0x1B23D6    sub_1801B2340                   +0x96
    //   0x1B32ED    sub_1801B32D0                   +0x1D
    //   0x1B320B    sub_1801B3100                   +0x10B
    //
    // They are additionally all in code that cannot run today: the
    // tf2patches::kHookSub* flags gating them are constexpr false, and the
    // remainder are held off by `static bool s_...Installed = true`. They are
    // diagnostic probes kept for re-enabling, so they are wired to the
    // resolver rather than deleted -- when one is wanted again, give it an
    // anchor here and it comes back correct instead of patching a stale RVA.
    inline const SymbolDesc kProducerMFenceSite {
      "engine.ProducerMFence.target", "engine.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kOrSiteCapture {
      "engine.OrSiteCapture.target", "engine.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kB84C0Capture {
      "engine.B84C0Capture.target", "engine.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kPropCullSite {
      "engine.PropCull.target", "engine.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kPropCullDecisionSite {
      "engine.PropCullDecision.site", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kBitmaskLoadSite {
      "engine.BitmaskLoad.site", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kDispatchSite {
      "engine.Dispatch.site", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kDispatchMFenceSite {
      "engine.DispatchMFence.site", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kVisibilityCounter {
      "engine.VisibilityCounter", "engine.dll", nullptr, nullptr,
      SymbolKind::RipRelativeData, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kBucketSourceTable {
      "engine.BucketSourceTable", "engine.dll", nullptr, nullptr,
      SymbolKind::RipRelativeData, 0, 0, 0, 0, 0 };

    // [Join] draw-span call sites in client.dll -- the two
    // IClientRenderable::DrawModel vcalls the renderable latch wraps. Same
    // stale region as the gate sites (0x36E36A / 0x36E9F9), so unregistered.
    //
    // Candidate signatures, NOT registered until verified unique against a
    // loaded client.dll, because a coincidental single match would redirect
    // the wrong virtual call:
    //   site A: 8B CF FF 50 48 48 8B 5C 24 40
    //   site C: CB 41 FF 52 48 40 84 F6
    inline const SymbolDesc kDrawSpanSiteA {
      "client.JoinDrawSpan.siteA", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };
    inline const SymbolDesc kDrawSpanSiteC {
      "client.JoinDrawSpan.siteC", "client.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    // engine.dll vertex-budget immediate (the 3.1M-vert cap raised to
    // 0x7FFFFFFF). Old RVA 0xB7100; the site verifies the immediate reads
    // 0x00300000 before writing, and on the shipped build it does not.
    inline const SymbolDesc kVertexBudgetCap {
      "engine.VertexBudgetImmediate", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    // engine.dll byte patches, both byte-verified before writing and both
    // stale on the shipped build (old RVAs 0x730DA, 0x1B32DF).
    inline const SymbolDesc kEntityMaskGate {
      "engine.EntityMaskGate.jz", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };
    inline const SymbolDesc kDispatchEntryE {
      "engine.DispatchEntryE.load", "engine.dll", nullptr, nullptr,
      SymbolKind::CodeSite, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kStudioDrawModelExecute {
      "studiorender.DrawModelExecute", "studiorender.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

    inline const SymbolDesc kStudioQueuedDraw {
      "studiorender.QueuedDrawEntry", "studiorender.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

  } // namespace tf2sym
} // namespace dxvk
