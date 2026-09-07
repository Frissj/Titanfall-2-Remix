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

    // ------------------------------------------------------------------
    // sec 7 SLICE B -- THE PRE-CULL RENDERABLE REGISTRY.
    //
    // ARCHITECTURE_OVERHAUL sec 1.2 rung 5 records this as "the enumeration
    // site is unsolved: sub_1801A8350's return is post-cull". Both halves of
    // that are answered here, and the second one is answered by not needing it.
    //
    // WHAT THE OLD ADDRESS ACTUALLY WAS. 0x1A8350 is not a function on the
    // shipped build; it is a label INSIDE BuildRenderableRenderLists' body (a
    // jnz target, 0x40 bytes past the entry). Reading it as a function is what
    // made the site look unsolvable.
    //
    // WHY THE RETURN BEING POST-CULL NO LONGER MATTERS. The body takes the
    // registry as its FIRST ARGUMENT and culls a copy of it into the caller's
    // output. The post-cull list is what the function RETURNS; the pre-cull
    // list is what it was HANDED. So we take the argument and never call the
    // function:
    //
    //     lea r9,  [rbx+1D8h]
    //     lea rdx, [rsp+40h]
    //     lea rcx, <REGISTRY>          <- this, arg0 by the x64 ABI
    //     mov r8,  rdi
    //     call    <the list builder>
    //
    // The registry is a statically allocated object rather than a pointer, so
    // the lea's rip-relative target IS its base address. Layout is read off the
    // builder and lives in rtx_engine_renderables.h, which owns the offsets.
    //
    // The builder's own "enumerate everything" arm walks that allocation mask
    // and takes every slot with a non-null pointer -- no leaf test, no frustum
    // test. That arm is the existence list; reading the same structure directly
    // gets it with no hook, no detour and no convar.
    //
    // STILL ONLY A CANDIDATE. "Pre-cull by construction" is an argument, and
    // sec 3.1 does not accept arguments: ExistenceSourcePromotion wants listed=
    // flat for 600 frames on the map under test. Until that passes this feeds a
    // VisibilitySource and kills nothing.
    // ------------------------------------------------------------------

    // The registered job function is a 9-byte thunk:
    //     mov rcx, rdx ; mov edx, 20h ; jmp <body>
    // so the body is one rel32 away. Chained onto the string-anchored thunk,
    // which is why this pattern only has to be unique inside those 9 bytes.
    inline const SymbolDesc kBuildRenderableRenderListsBody {
      /* name         */ "client.BuildRenderableRenderLists.body",
      /* moduleName   */ "client.dll",
      /* pattern      */ "48 8B CA BA ?? ?? ?? ?? E9",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CallTarget,
      /* addend       */ 8,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0x20,
      /* base         */ &kBuildRenderableRenderLists,
    };

    // `48 8D 0D` is `lea rcx, [rip+disp32]`. rcx is arg0 under the Windows x64
    // ABI, so this is not a register-allocation coincidence that needs padding
    // out with neighbouring instructions -- it is the calling convention, and
    // it is the ONLY rip-relative lea in the body. Ambiguity fails safe, so a
    // build that grows a second one disables the feature and says so rather
    // than silently picking the wrong global.
    inline const SymbolDesc kRenderableRegistry {
      /* name         */ "client.RenderableRegistry",
      /* moduleName   */ "client.dll",
      /* pattern      */ "48 8D 0D",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::RipRelativeData,
      /* addend       */ 0,
      /* dispOffset   */ 3,
      /* instrLength  */ 7,
      /* searchBefore */ 0,
      /* searchAfter  */ 0x200,
      /* base         */ &kBuildRenderableRenderListsBody,
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

    // ------------------------------------------------------------------
    // [Join] draw-span call sites -- the two IClientRenderable::DrawModel
    // vcalls the renderable latch wraps. REGISTERED 2026-09-06.
    //
    // THE SIGNATURES WERE NEVER STALE. ONLY THE RVAs WERE. Both candidates
    // below were written down but left unregistered pending verification
    // against a loaded client.dll. Read back against the shipped v2.0.11.0
    // binary, both match byte-for-byte; they had simply moved:
    //
    //   site A   old 0x36E36A -> 0x36E3CA   in sub_18036E340
    //   site C   old 0x36E9F9 -> 0x36EA59   in sub_18036E930  (the draw loop)
    //
    // sub_18036E930 is the renderable draw loop: it walks a 16-byte-strided
    // list, calls the cloak predicate at vtable+0xE0 -- the same virtual the
    // leaf-system paths call on registry renderables -- then the gate
    // sub_180371610, then DrawModel at vtable+0x48. sub_18036E340 is the
    // single-renderable path and ends in the same vcall.
    //
    // In BOTH, rcx holds the renderable at the call, because it is the `this`
    // pointer. That is what makes a latch possible at all: the handle the
    // registry hands out and the pointer this call receives are the same
    // 64-bit token, so the join needs no translation table.
    //
    // WHY A BARE SIGNATURE IS ACCEPTABLE HERE, given sec 2 prefers anchors.
    // These are CodeSites that get PATCHED, so the installer already
    // memcmp()s all 10/8 bytes plus the instructions its island replays
    // before writing anything, and refuses on any mismatch. A coincidental
    // match is caught twice over: the resolver rejects 2+ matches, and the
    // installer re-verifies the exact bytes at the address it was handed.
    //
    // addend lands the symbol on the CALL, not on the mov that precedes it:
    // the patch word is siteA-2 / siteC-1 and both must stay 8-byte aligned,
    // which they are (0x36E3C8, 0x36EA58) -- the installer checks that too.
    // ------------------------------------------------------------------
    inline const SymbolDesc kDrawSpanSiteA {
      /* name         */ "client.JoinDrawSpan.siteA",
      /* moduleName   */ "client.dll",
      /* pattern      */ "8B CF FF 50 48 48 8B 5C 24 40",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 2,          // -> the FF 50 48
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };
    inline const SymbolDesc kDrawSpanSiteC {
      /* name         */ "client.JoinDrawSpan.siteC",
      /* moduleName   */ "client.dll",
      /* pattern      */ "CB 41 FF 52 48 40 84 F6",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 1,          // -> the 41 FF 52 48
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

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

    // ------------------------------------------------------------------
    // [Join] QUEUED-DRAW HOOK in materialsystem_dx11.dll. REGISTERED 2026-09-06.
    //
    // These replace eight hardcoded `matsys + 0xRVA` literals. Read back
    // against the shipped v2.0.11.0 materialsystem_dx11.dll, the whole cluster
    // had moved by exactly +0xF0:
    //
    //   dispatcher   0x87F80 -> 0x88070
    //   allocator    0x87960 -> 0x87A50
    //   wrap sentry  0x872D0 -> 0x873C0
    //
    // AND THE DATA GLOBALS MOVED BY A DIFFERENT DELTA, +0x1040, because they
    // live in another section:
    //
    //   descriptor   0x1BBA040 -> 0x1BBB080
    //   readCursor   0x1BBA054 -> 0x1BBB094
    //
    // That difference is the whole argument for sec 2's "derive second
    // addresses out of the instruction stream". A single code delta applied to
    // the data addresses would have produced two plausible-looking pointers
    // into the wrong place, and the hook writes through them. So the two
    // globals below are decoded from the rip-relative operands of the very
    // instructions the dispatcher signature already verified, and cannot
    // disagree with the code they belong to.
    // ------------------------------------------------------------------

    // The queue dispatcher. Signature spans the load plus the whole 24-byte
    // body the installer memcmp()s, with only the three rip displacements
    // wildcarded -- unique in .text. Everything else the installer needs
    // (body/word/patch/resume) is a fixed offset INSIDE these verified bytes.
    inline const SymbolDesc kQueuedDrawDispatcher {
      /* name         */ "matsys.QueuedDraw.dispatcher",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "48 8B 05 ?? ?? ?? ?? 8B CA 83 C2 08 48 8B 3C 01"
                         " 48 8D 0D ?? ?? ?? ?? 89 15 ?? ?? ?? ?? FF D7",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // qword_181BBA040 -> the drained-queue descriptor, from `lea rcx, [rip+d]`
    // at dispatcher+0x10. Chained, so `48 8D 0D` only has to be unique inside
    // the dispatcher rather than across 1.8 MB of .text.
    inline const SymbolDesc kQueuedDrawDescriptor {
      /* name         */ "matsys.QueuedDraw.descriptor",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "48 8D 0D",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::RipRelativeData,
      /* addend       */ 0,
      /* dispOffset   */ 3,
      /* instrLength  */ 7,
      /* searchBefore */ 0,
      /* searchAfter  */ 0x20,
      /* base         */ &kQueuedDrawDispatcher,
    };

    // dword_181BBA054 -> the read cursor, from `mov [rip+d], edx` at
    // dispatcher+0x17. `FF D7` is carried along so the pattern clears the
    // chained specificity floor on concrete bytes rather than on length.
    inline const SymbolDesc kQueuedDrawReadCursor {
      /* name         */ "matsys.QueuedDraw.readCursor",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "89 15 ?? ?? ?? ?? FF D7",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::RipRelativeData,
      /* addend       */ 0,
      /* dispOffset   */ 2,
      /* instrLength  */ 6,
      /* searchBefore */ 0,
      /* searchAfter  */ 0x20,
      /* base         */ &kQueuedDrawDispatcher,
    };

    // The queue-record allocator. Its prologue is a stock MSVC register save
    // that occurs 442 times in this module, and is still ambiguous at 24 bytes
    // (27 hits). Uniqueness only arrives past it, at the counter load plus the
    // gs:[0x58] TLS access -- so the signature runs to 36 bytes and wildcards
    // the one displacement in it. THIS IS WHY THE 5-BYTE `48 89 5C 24 08` THE
    // INSTALLER VERIFIES IS NOT A LOCATOR: it is a check, not an identity.
    inline const SymbolDesc kQueuedDrawAlloc {
      /* name         */ "matsys.QueuedDraw.allocator",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57"
                         " 41 56 41 57 48 83 EC 20 44 8B 0D ?? ?? ?? ??"
                         " 65 48 8B 04 25",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // The ring's wrap sentinel -- a 14-byte leaf, unique at its first 8 bytes,
    // taken whole so the signature ends on its own `retn`. CodeSite rather
    // than Function because a leaf this size carries no .pdata entry, and its
    // address is only ever compared, never called.
    inline const SymbolDesc kQueuedDrawWrapSentinel {
      /* name         */ "matsys.QueuedDraw.wrapSentinel",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "8B 51 14 48 8B 41 08 8B 14 02 89 51 14 C3",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // ------------------------------------------------------------------
    // [Join] WORLD BATCH DRAW -- engine.dll, the population slice 1 could
    // never name. Located 2026-09-06; NOT yet consumed by anything.
    //
    // [Join.who] attributes ~25-48% of all draws to two engine.dll producers
    // (+0x1b3692 and +0x1b47e2) whose replay bodies are world-surface BATCH
    // renderers. sub_1801B36E0 accumulates N surfaces into one global scratch
    // array and issues a SINGLE draw for all of them:
    //
    //     for each surface:  accumulate into unk_193B894C0
    //     call [r10+0x560](ctx, count, &unk_193B894C0, ...)
    //
    // SO THE DRAW HAS NO OBJECT. It legitimately covers many, and no latch can
    // say otherwise -- which is the same fact [RsChurn] was reporting from the
    // other end. Its drawStart/drawCount tiling contiguously (75450+11376 =
    // 86826, +4944 = 91770) is batch boundaries moving as visibility changes,
    // not identity churn in the objects themselves.
    //
    // WHAT MAKES THIS FIXABLE. Each surface enters the batch as
    //     v26 = qword_193F09850 + 112 * (*(uint32*)(entry + 4))
    // -- an INDEX into a persistent global table, not a position in this
    // frame's allocation. So the batch does have a stable name available: the
    // set of surface indices in it. Same view, same set, same key; a different
    // batch correctly gets a different key, and a camera that returns to a
    // previous view recurs instead of minting. That is precisely the property
    // drawStart lacked, and it is why this is worth building rather than
    // declaring world geometry unnameable.
    //
    // The site below is the batch draw itself, where count and the descriptor
    // array are both live. Patch word 0x1B3A60 is 8-byte aligned, so the same
    // single-aligned-store technique the other two [Join] patches use applies.
    //
    // REGISTERED BUT UNUSED ON PURPOSE. Nothing reads it yet: the identity
    // hypothesis above is an argument, and this tree does not wire arguments.
    // It gets measured first -- hash the set, watch it under a fixed-position
    // sweep -- exactly as slice B was made to earn its ExistenceSource.
    inline const SymbolDesc kWorldBatchDrawSite {
      /* name         */ "engine.WorldBatchDraw.site",
      /* moduleName   */ "engine.dll",
      /* pattern      */ "48 8B CB 48 89 44 24 20 41 FF 92 60",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::CodeSite,
      /* addend       */ 8,          // -> the 41 FF 92 60 05 00 00
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // ------------------------------------------------------------------
    // NV-DXVK [WorldBatch] SECOND PRODUCER -- the depth-only world mesh pass.
    //
    // WHY THIS IS NOT A FOURTH CASE, which is what the handoff expected it to
    // be. [Join.who] bills 14% of dispatched records to engine.dll+0xb81e7, and
    // that site is this function's own queue-alloc call: sub_1800B81B0 defers
    // ITSELF, writing a 24-byte record { fn=sub_1800B81B0, a1, a2 } that the
    // render thread later re-invokes with the defer test false. So the 14% is
    // not a producer of many objects, it is one PASS -- and the pass turns out
    // to have exactly the shape the surface batch had:
    //
    //   walk a visibility BITMASK at a1+344200, one bit per world mesh
    //   -> 24-byte entry at *(qword_1807CB410+104) + 24*bitIndex
    //   -> uint16 at entry+14 = an INDEX into the persistent mesh table
    //      at *(qword_1807CB410+168)
    //
    // An index into a persistent table is the property that made the surface
    // set nameable, so the same key works here, and the set is available at the
    // function's own ENTRY -- the bitmask IS the set. That is why this is one
    // entry detour and not a hook in the inner loop.
    //
    // ANCHOR. The literal is passed to the profile-scope call at +0x106 and has
    // exactly ONE reference in the module, so findStringRefs' agreement rule is
    // satisfied by a single site with nothing to contradict it. 0x400 covers
    // the +0x106 walk-back with room to spare.
    inline const SymbolDesc kDepthOnlyWorldMeshes {
      /* name         */ "engine.DrawWorldMeshesDepthOnly",
      /* moduleName   */ "engine.dll",
      /* pattern      */ nullptr,
      /* anchorString */ "DrawWorldMeshesDepthOnly",
      /* kind         */ SymbolKind::StringEnclosingFunction,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0x400,
      /* searchAfter  */ 0,
    };

    // THE WORLD RENDER DATA the bitmask is sized against. Only one field is
    // wanted -- the dword at +8, the total world mesh count -- because the
    // detour has to know how many bitmask words to hash and the count is the
    // only thing that says.
    //
    // CHAINED, AND THE PATTERN IS DELIBERATELY LONGER THAN THE LOAD. A bare
    // `48 8B 05` is hopeless even inside one function; what makes this unique
    // is welding it to the round-up-to-qwords idiom that consumes it:
    //
    //   48 8B 05 ?? ?? ?? ??   mov rax, cs:<global>
    //   48 89 B4 24 ?? ?? ?? ??  mov [rsp+..], rsi     (register save, masked)
    //   8B 70 08               mov esi, [rax+8]        <- the count
    //   83 C6 3F               add esi, 3Fh
    //   48 C1 EE 06            shr rsi, 6              <- words = (count+63)/64
    //
    // The last three instructions are what make it a bitmask sizer rather than
    // any other global load, and they are bound to THIS global by the rax
    // dependency. The displacement itself is masked -- it is what we are
    // decoding, and baking it would be the stale-literal defect the queue
    // hook's verification arrays already demonstrated.
    //
    // searchBefore == 0 means "start AT the base symbol"; the load sits at
    // +0xC7, well inside the 0x4B1-byte body.
    inline const SymbolDesc kDepthOnlyWorldData {
      /* name         */ "engine.DrawWorldMeshesDepthOnly.worldData",
      /* moduleName   */ "engine.dll",
      /* pattern      */ "48 8B 05 ?? ?? ?? ?? 48 89 B4 24 ?? ?? ?? ?? 8B 70 08 83 C6 3F 48 C1 EE 06",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::RipRelativeData,
      /* addend       */ 0,
      /* dispOffset   */ 3,
      /* instrLength  */ 7,
      /* searchBefore */ 0,
      /* searchAfter  */ 0x4C0,
      /* base         */ &kDepthOnlyWorldMeshes,
    };

    // ------------------------------------------------------------------
    // IStudioRenderContext::DrawModel -- studiorender.dll sub_180015D10, the
    // function sitting in context vtable slot +0xB8.
    //
    // WHY THIS IS A MASKED SIGNATURE AND NOT A STRING ANCHOR, which is the
    // order the working rules ask for. All three preferences were checked
    // against the shipped module before settling here:
    //
    //   CreateInterface   studiorender.dll has NO versioned interface literal
    //                     at all -- no "VStudioRender0xx", nothing matching
    //                     /VStudio|StudioRender0|IStudioRender/. The Titanfall
    //                     fork does not name its interfaces the Source way.
    //   string anchor     the function body contains no literal of any kind,
    //                     and it has NO CALLERS to chain from either -- it is
    //                     reached only through the vtable slot, which is the
    //                     whole reason this symbol exists.
    //   masked signature  what is left, and it is the declared third choice.
    //
    // WHAT MAKES IT SPECIFIC ENOUGH TO REGISTER MODULE-WIDE. The body opens
    // with a four-step null-check chain on the DrawModelInfo argument that its
    // two sibling entries (sub_180015A60, sub_180015C00) do not have:
    //
    //   48 89 2A            mov  [rdx], rbp        clear DrawModelResults
    //   49 39 28            cmp  [r8], rbp         info->hwdata
    //   49 8B 40 08         mov  rax, [r8+8]       info->loddata
    //   39 68 04            cmp  [rax+4], ebp      loddata->numLOD
    //   48 39 68 08         cmp  [rax+8], rbp      loddata->meshTbl
    //   ... mov rcx, cs:<global>; mov rax,[rcx]; call [rax+3B8h]
    //
    // Those are the same three gates [De15] enumerates as the pre-enqueue
    // early-outs, so the signature is anchored on the function's SEMANTICS
    // rather than on a register allocation that a rebuild would shuffle.
    //
    // WHAT IS MASKED, AND WHY EACH. Every `0F 84` displacement is an internal
    // rel32 that moves whenever anything between here and the target changes
    // size; the `48 8B 0D` displacement is the rip-relative global we are NOT
    // trying to name here; the `sub rsp` immediate and the `mov [rsp+..], rbx`
    // displacement are compiler-chosen frame layout. Baking any of them would
    // be the stale-literal defect the [Join] verification arrays already paid
    // for once. What stays concrete is opcodes and register operands only.
    // THE THIRD WORLD PRODUCER -- engine.dll sub_1800B8670, which [Join.who]
    // bills 5% of the frame to as engine.dll+0xb86b0.
    //
    // Same self-deferring shape as DrawWorldMeshesDepthOnly: +0xb86b0 is this
    // function's own queue-alloc call, writing { fn, a1, a2, a3 } for the render
    // thread to re-invoke. On replay it walks 16-byte entries over the range
    // [ *(int*)(a1 + 4*a3), *(int*)(a1 + 4*(a3+1)) ), keeps the ones matching
    // mask a2, groups them by the top 48 bits of the first qword and draws the
    // packed result. Every input to that is live at the entry, so one entry
    // detour names it -- the same reason the depth pass is hooked at its entry
    // rather than in its loop.
    //
    // Distinguished from the depth pass by its prologue (push rbx/rsi/r14 vs
    // push rdi/r12) and by `mov edx,18h` + `lea r8d,[rdx-11h]` -- a 24-byte
    // record with alignment 7, written that way by the compiler and unique
    // enough to anchor on. Every rel32, rip-relative displacement and the
    // __chkstk argument are masked.
    inline const SymbolDesc kShadowRangePass {
      /* name         */ "engine.ShadowRangePass",
      /* moduleName   */ "engine.dll",
      /* pattern      */ "40 53 56 41 56 B8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 48 2B E0 "
                         "41 8B D8 44 8B F2 48 8B F1 FF 15 ?? ?? ?? ?? 85 C0 74 ?? "
                         "E8 ?? ?? ?? ?? BA 18 00 00 00 48 8D 0D ?? ?? ?? ?? 44 8D 42 EF "
                         "FF 15 ?? ?? ?? ??",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::Function,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // THE FOURTH WORLD PRODUCER -- engine.dll sub_1800B7960, which [Join.who]
    // bills 1% of the frame to as engine.dll+0xb79a4 with iaOnly=30.
    //
    // SMALL, AND THE REASON IT IS WORTH A SYMBOL ANYWAY. [ResidentGate]'s by{}
    // split measured this population at hit=0 on EVERY window it has ever been
    // printed on, with draws pinned at 290 per ten frames -- a total, perfectly
    // reproducible failure rather than a low rate. It is the whole of the
    // iaOnly class: 30 of 31 draws a frame, and the only producer in the frame
    // with no upstream name left.
    //
    // SELF-DEFERRING, the third time this shape has turned up after
    // DrawWorldMeshesDepthOnly and ShadowRangePass. +0xb79a4 is this function's
    // own queue-alloc call, writing { fn = itself, a1, a2, a3, a4 } into a
    // 32-byte record for the render thread to re-invoke -- so the site
    // [Join.who] names is not a producer of draws at all, which is exactly why
    // its draws arrive with no span around them and fall to the IA key.
    //
    // REACHED ONLY THROUGH A VTABLE, like the studio DrawModel path. Its single
    // code xref is a five-byte thunk (sub_1800DCE60) whose own only reference is
    // a vtable slot, so there is no call site to hook and an entry detour is the
    // only thing that catches every dispatch. That also means no chained
    // derivation is available: nothing already resolved calls it, so this is a
    // masked signature or nothing.
    //
    // WHAT IS ANCHORED. The four-register prologue welded to its 0x8048 stack
    // probe, then the deferral block's `mov edx,20h` + `lea r8d,[rdx-19h]` --
    // a 32-byte record at alignment 7, the same compiler idiom ShadowRangePass
    // is anchored on one size down (0x18 / [rdx-11h]). Every rel32, every
    // rip-relative displacement and the short jz are masked; what stays concrete
    // is opcodes, register operands and the two record constants.
    inline const SymbolDesc kWorldMeshListPass {
      /* name         */ "engine.WorldMeshListPass",
      /* moduleName   */ "engine.dll",
      // `40 55` -- the push rbp carries a REX prefix. Taken from the live bytes,
      // not from the disassembly listing, because the listing prints one
      // mnemonic either way and the difference is a byte of steal length. That
      // is the same trap the first draw-callee table fell into by assuming five.
      /* pattern      */ "40 55 56 57 41 55 B8 48 80 00 00 E8 ?? ?? ?? ?? 48 2B E0 "
                         "41 8B F9 41 8B F0 4C 8B EA 48 8B E9 FF 15 ?? ?? ?? ?? "
                         "85 C0 74 ?? E8 ?? ?? ?? ?? BA 20 00 00 00 "
                         "48 8D 0D ?? ?? ?? ?? 44 8D 42 E7 FF 15 ?? ?? ?? ??",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::Function,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // materialsystem_dx11.dll sub_18006F130 -- the producer [Join.who] bills 2%
    // of the frame to as materialsystem_dx11.dll+0x6f152, the last named gap.
    //
    // A deferring stub again: it queues { fn = sub_18006E958, obj } where obj is
    // *(a1+264), and sub_18006E958 is a one-instruction thunk that calls
    // obj->vtable[0x220](obj). The thunk is nine bytes, too small to detour
    // safely, so the span goes on the STUB -- the queue record is allocated
    // inside it, so the record/replay join carries the key to the draws the
    // replay issues. That is the same route the studio path already takes, and
    // it is why that path reads keyed{replay=...} rather than direct.
    //
    // THE IDENTITY IS THE OBJECT ITSELF. There is no batch, no count and no
    // descriptor array here -- just a persistent render object reached through
    // a member pointer, so the pointer is the stable name, exactly as the client
    // renderable pointer is for [Join]. If one object turns out to produce all
    // of these draws they will share a key and be separated only by drawCount
    // and material; maxDrawRun is what says whether that happened.
    //
    // Anchored on `mov rbx,[rcx+108h]` (the 264 offset) welded to the 16-byte
    // record size and its alignment-7 `lea r8d,[rdx-9]`.
    inline const SymbolDesc kMatsysObjectFlush {
      /* name         */ "materialsystem.ObjectFlushEnqueue",
      /* moduleName   */ "materialsystem_dx11.dll",
      /* pattern      */ "40 53 48 83 EC 20 48 8B 99 08 01 00 00 BA 10 00 00 00 "
                         "48 8D 0D ?? ?? ?? ?? 44 8D 42 F7 E8 ?? ?? ?? ??",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::Function,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    inline const SymbolDesc kStudioDrawModelExecute {
      /* name         */ "studiorender.DrawModelExecute",
      /* moduleName   */ "studiorender.dll",
      /* pattern      */ "48 89 6C 24 10 48 89 74 24 18 57 41 56 41 57 48 81 EC ?? ?? ?? ?? "
                         "33 ED 4D 8B F9 49 8B F0 48 8B FA 4C 8B F1 48 85 D2 74 03 48 89 2A "
                         "49 39 28 0F 84 ?? ?? ?? ?? 49 8B 40 08 48 85 C0 0F 84 ?? ?? ?? ?? "
                         "39 68 04 0F 84 ?? ?? ?? ?? 48 39 68 08 0F 84 ?? ?? ?? ?? "
                         "48 8B 0D ?? ?? ?? ?? 48 89 9C 24 ?? ?? ?? ?? 48 8B 01 FF 90 B8 03 00 00",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::Function,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    // THE MODEL-ARRAY ENQUEUE -- studiorender.dll sub_180013F30, the producer
    // [Join.who] bills 10% of the frame to as studiorender.dll+0x13f74.
    //
    // WHY THE ENQUEUE AND NOT THE TWO ENTRIES THAT CALL IT. The array path has
    // TWO entry points, sub_180015A60 and sub_180015C00, and they are byte-for-
    // byte identical clones -- same prologue, same queue-mode test, same
    // register allocation, differing only in rip-relative displacements and the
    // rel32 of the worker they pass. No masked signature can separate them, and
    // EngineSymbols correctly refuses a 2-match resolve, so anchoring either one
    // resolves to nothing. Their shared enqueue is a single function, is where
    // +0x13f74 actually lives (0x13F30 + 0x44 is the queue-alloc call's return
    // address), and covers both clones with one hook.
    //
    // ARGS AT ENTRY: rcx = the worker to replay, rdx = the render-info pointer,
    // r8 = the context transform block, r9d = COUNT, and the copied 32-byte
    // instance array as stack argument 5. Both count and array are therefore in
    // hand at the entry, which is what the key needs.
    //
    // DISCRIMINATOR: `mov edx, 90h` -- the 144-byte record size -- welded to
    // `mov r8d, 7` and the queue-alloc call. sub_180013D10, the OTHER enqueue,
    // queues a different size, so this cannot match it.
    inline const SymbolDesc kStudioArrayEnqueue {
      /* name         */ "studiorender.ModelArrayEnqueue",
      /* moduleName   */ "studiorender.dll",
      /* pattern      */ "48 89 5C 24 08 48 89 74 24 10 48 89 7C 24 18 55 41 56 41 57 "
                         "48 8D 6C 24 D1 48 81 EC ?? ?? ?? ?? 48 8B D9 49 8B F0 48 8B FA "
                         "48 8D 0D ?? ?? ?? ?? BA 90 00 00 00 41 B8 07 00 00 00 45 8B F9 "
                         "FF 15 ?? ?? ?? ??",
      /* anchorString */ nullptr,
      /* kind         */ SymbolKind::Function,
      /* addend       */ 0,
      /* dispOffset   */ 0,
      /* instrLength  */ 0,
      /* searchBefore */ 0,
      /* searchAfter  */ 0,
    };

    inline const SymbolDesc kStudioQueuedDraw {
      "studiorender.QueuedDrawEntry", "studiorender.dll", nullptr, nullptr,
      SymbolKind::Function, 0, 0, 0, 0, 0 };

  } // namespace tf2sym
} // namespace dxvk
