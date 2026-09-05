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

  } // namespace tf2sym
} // namespace dxvk
