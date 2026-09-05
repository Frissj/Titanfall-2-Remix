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
// NV-DXVK [EngineSymbols] -- build-independent engine symbol resolution.
//
// WHY THIS EXISTS
// ---------------
// Remix's TF2 integration reaches into the game's own modules (client.dll,
// engine.dll, studiorender.dll) to read camera state, patch cull sites and
// install trampolines. Historically every one of those reaches was a
// HARDCODED RVA reverse-engineered from one specific build:
//
//     auto getLp = (void*(*)())((uint8_t*)GetModuleHandleA("client.dll") + 0x14EAE0);
//     void* lp = getLp();                                   // <-- per-frame crash
//
// On the installed v2.0.11.0 client.dll that address is not a function entry:
// it is +0x70 into an unrelated float-lerp routine. Calling it runs a function
// body having skipped its prologue, so the epilogue's `movaps xmm6,[rsp]`
// faults on a misaligned rsp -- reported by Windows as
// "access violation reading 0xFFFFFFFFFFFFFFFF", the SSE alignment sentinel.
//
// It is not a one-offset typo. EVERY hardcoded client.dll RVA in the codebase
// lands mid-function on this build, by INCONSISTENT deltas (+0x20, +0x40,
// +0x70, +0x150, one unmapped entirely) -- i.e. the shipped client.dll is a
// different compilation than the one those offsets were derived from. A
// non-constant delta cannot be corrected by rebasing. Re-deriving the numbers
// would work exactly until the next game patch, which is precisely the failure
// we are already in.
//
// THE CONTRACT
// ------------
// Nothing in Remix may ever again call, patch or dereference a raw
// `module + 0xRVA`. Instead:
//
//   1. Prefer an exported, versioned ABI. Source modules export
//      `CreateInterface(name, rc)`; that is the intended extension point and
//      involves no offsets at all. See createInterface().
//
//   2. Otherwise resolve by MASKED SIGNATURE over the module's real executable
//      sections, with FAIL-SAFE semantics: exactly one match, or nothing. Zero
//      or two-plus matches resolve to null and log; we never touch an address
//      we could not uniquely identify. See Pattern / resolve().
//
//   3. Derive addresses and displacements OUT OF the instruction stream rather
//      than hardcoding them -- a `call rel32` gives up its callee, a
//      rip-relative `mov` gives up its global. See decodeRel32Target() and
//      decodeRipRelative(). A signature over a call site therefore also yields
//      the callee, which is the robust replacement for hardcoding both.
//
//   4. Anything still unresolved DISABLES ITS FEATURE and logs once. Remix
//      stays alive with one capability dark; it does not crash and it does not
//      call an unverified address.
//
// Everything is keyed on a module GENERATION that changes whenever the module
// is unloaded and reloaded (client.dll unloads on quit -- a pointer cached
// across that boundary is the other historical crash). Resolution results are
// cached per generation and dropped automatically when it moves.
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace dxvk {

  namespace EngineSymbols {

    // ------------------------------------------------------------------
    // Module view: base plus the real section bounds parsed from the PE
    // headers. Scanning the whole image would sweep resources, relocations
    // and import thunks and invent matches; every scan is bounded to the
    // sections that can legitimately hold the thing being looked for.
    // ------------------------------------------------------------------
    struct ModuleView {
      void*     handle        = nullptr;
      uintptr_t base          = 0;
      uint32_t  imageSize     = 0;
      uint32_t  timeDateStamp = 0;   // PE link timestamp -- part of the fingerprint
      uint32_t  peCheckSum    = 0;

      uintptr_t codeBegin  = 0, codeEnd  = 0;   // union of IMAGE_SCN_MEM_EXECUTE sections
      uintptr_t rdataBegin = 0, rdataEnd = 0;   // initialized, non-writable data (vtables, strings)
      uintptr_t dataBegin  = 0, dataEnd  = 0;   // writable data (globals, ConVar objects)

      // Bumps whenever the module is (re)loaded at a different base or with a
      // different link timestamp. Cached resolutions carry the generation they
      // were produced under and are discarded when it changes.
      uint64_t  generation   = 0;

      bool valid()                     const { return handle != nullptr && codeEnd > codeBegin; }
      bool containsCode (uintptr_t a)  const { return a >= codeBegin  && a < codeEnd;  }
      bool containsRData(uintptr_t a)  const { return a >= rdataBegin && a < rdataEnd; }
      bool containsData (uintptr_t a)  const { return a >= dataBegin  && a < dataEnd;  }
      bool containsImage(uintptr_t a)  const { return a >= base && a < base + imageSize; }
    };

    // Look a loaded module up by name and parse its section table. Returns
    // false when the module is not currently loaded -- callers must treat that
    // as "feature unavailable this frame", never as an error.
    bool queryModule(const char* moduleName, ModuleView& out);

    // ------------------------------------------------------------------
    // Build fingerprint. Cheap to compute, stable for a given binary, and
    // printable so an unknown build self-documents in the log.
    // ------------------------------------------------------------------
    struct Fingerprint {
      uint32_t imageSize     = 0;
      uint32_t timeDateStamp = 0;
      uint32_t peCheckSum    = 0;
      uint64_t codeHash      = 0;   // FNV-1a over a sparse sample of .text

      bool operator==(const Fingerprint& o) const {
        return imageSize == o.imageSize && timeDateStamp == o.timeDateStamp
            && peCheckSum == o.peCheckSum && codeHash == o.codeHash;
      }
      bool operator!=(const Fingerprint& o) const { return !(*this == o); }

      std::string toString() const;
    };

    Fingerprint fingerprintOf(const ModuleView& m);

    // ------------------------------------------------------------------
    // Masked byte pattern, IDA syntax: "48 8B C4 44 89 40 18" with '?' or
    // '??' for a wildcard byte. Wildcard every relocated, rip-relative and
    // rel32 displacement -- those are exactly the bytes that legitimately
    // move between builds, and pinning them is what makes a signature
    // brittle instead of robust.
    // ------------------------------------------------------------------
    class Pattern {
    public:
      Pattern() = default;
      explicit Pattern(const char* idaStyle);

      bool   valid() const { return !m_bytes.empty(); }
      size_t size()  const { return m_bytes.size(); }

      // Counts matches in [begin,end), stopping early once `stopAfter` have
      // been seen (2 is enough to prove non-uniqueness). Writes the first
      // match to `firstOut` when non-null.
      size_t countMatches(uintptr_t begin, uintptr_t end,
                          uintptr_t* firstOut, size_t stopAfter = 2) const;

    private:
      std::vector<uint8_t> m_bytes;
      std::vector<uint8_t> m_wild;   // 1 = wildcard. Not vector<bool>: the raw
                                     // scan loop needs a plain byte array.
    };

    // ------------------------------------------------------------------
    // Safe memory predicates. Every deref of a game address goes through
    // these first; the game can unload a module mid-frame while the dxvk CS
    // thread is still running this code.
    // ------------------------------------------------------------------
    bool readable  (const void* p, size_t bytes);
    bool writable  (const void* p, size_t bytes);
    bool executable(const void* p);

    // ------------------------------------------------------------------
    // Is `addr` plausibly the ENTRY of a function rather than a point inside
    // one? This is the check whose absence caused the crash: the old code
    // verified only that the page was committed and executable, which a
    // mid-function address trivially satisfies.
    //
    // Accepts when the address is 16-byte aligned (MSVC aligns function
    // starts) or immediately preceded by inter-function padding (int3 / nop)
    // or by a control-transfer terminator (ret / jmp), AND begins with a
    // recognised x64 prologue form.
    // ------------------------------------------------------------------
    bool looksLikeFunctionEntry(const ModuleView& m, uintptr_t addr);

    // ------------------------------------------------------------------
    // Instruction-stream decoding. Prefer these over hardcoding a second
    // address: given a signature-located call site you get its callee for
    // free, and it is correct by construction on every build.
    // ------------------------------------------------------------------

    // `site` points at an E8 (call) or E9 (jmp) opcode. Yields the branch
    // target and validates it lands in the module's executable range.
    bool decodeRel32Target(const ModuleView& m, uintptr_t site, uintptr_t& targetOut);

    // rip-relative operand: `dispAt` is the address of the disp32 field and
    // `instrEnd` the address of the next instruction. Yields the referenced
    // address and validates it lands somewhere in the module image.
    bool decodeRipRelative(const ModuleView& m, uintptr_t dispAt,
                           uintptr_t instrEnd, uintptr_t& targetOut);

    // ------------------------------------------------------------------
    // Exported, versioned interface ABI -- the strongest option, no offsets.
    //
    // Source's CreateInterface does an EXACT string compare against a
    // registry of InterfaceReg nodes, so there is no prefix matching in the
    // engine. We supply it by trying "<baseName>%03d" across a version range,
    // which keeps working when a build bumps an interface version.
    //
    // Confirmed present on TF2 v2.0.11.0 client.dll (export at RVA 0x73BA00):
    //   VClient018, VClientEntityList003  (exposed by client.dll)
    //   VEngineClient013, VEngineCvar007  (requested from engine.dll/vstdlib)
    // ------------------------------------------------------------------
    void* createInterface(const ModuleView& m, const char* baseName,
                          int minVersion, int maxVersion, int* versionOut = nullptr);

    // ------------------------------------------------------------------
    // String-anchored resolution -- in practice the most durable technique
    // available, and stronger than a byte signature.
    //
    // Compiler output churns between builds: registers get reallocated,
    // instructions reorder, inlining decisions flip. String literals do not.
    // A function that is registered, named or logged by a literal can be found
    // through that literal on any build that still contains it.
    //
    // findStringRef() locates the unique read-only occurrence of `text`, then
    // the unique instruction in .text whose rip-relative operand points at it.
    // findCodePointerNear() then reads the *other* rip-relative operand in the
    // surrounding window -- which is how a job/callback registration gives up
    // the function it registers:
    //
    //     lea rax, aBuildRenderableRenderLists   <- found by findStringRef
    //     lea rdx, sub_1801A82A0                 <- the job, by findCodePointerNear
    //     call JTGuts_RegisterJobType
    //
    // Both require exactly one candidate and return false otherwise.
    // ------------------------------------------------------------------
    bool findStringRef(const ModuleView& m, const char* text,
                       uintptr_t& stringAddrOut, uintptr_t& dispSiteOut);

    // Search [site - before, site + after] for rip-relative operands whose
    // target lands in the module's executable range. Exactly one must qualify.
    bool findCodePointerNear(const ModuleView& m, uintptr_t dispSite,
                             int32_t before, int32_t after, uintptr_t& functionOut);

    // Walk back from an address INSIDE a function to that function's entry,
    // for the other common case: the literal is used by the function (a VPROF
    // scope name, a warning string) rather than registering a pointer to it.
    //
    // Requires the entry to be int3-padded, 16-byte aligned AND prologue-
    // shaped -- 0xCC also appears as an immediate operand mid-function, and
    // any one of those signals alone would eventually land inside one.
    bool findEnclosingFunction(const ModuleView& m, uintptr_t insideAddr,
                               uint32_t maxScanBack, uintptr_t& functionOut);

    // Fetch vtable slot `index` off `object`, validating that the vtable
    // pointer lies in read-only data and the slot target in executable code.
    //
    // NOTE: this proves the slot is *callable*, not that the index is the
    // *right* one. A wrong index is a valid function with the wrong
    // signature, and calling it is the same class of bug as the original
    // crash. Only ever pass an index that has been derived for this build and
    // recorded; never probe.
    bool vtableSlot(const ModuleView& owner, void* object, uint32_t index, void*& fnOut);

    // ------------------------------------------------------------------
    // ConVar lookup by name, with no offsets and no ICvar vtable index.
    //
    // A registered Source ConVar holds a pointer to its own name string. So:
    // find the "<name>" literal in read-only data, find the single writable
    // qword pointing at it, then walk back to the object start by looking for
    // the m_pParent self-pointer (a registered ConVar parents to itself
    // unless it is a child of another, in which case the parent is a ConVar
    // too). Everything is cross-checked, and the whole thing fails safe.
    // ------------------------------------------------------------------
    struct ConVarHandle {
      uintptr_t object     = 0;   // ConVar*
      uintptr_t parent     = 0;   // ConVar* actually holding the value
      int*      intValue   = nullptr;
      float*    floatValue = nullptr;

      bool valid() const { return object && intValue && floatValue; }
    };

    bool findConVar(const ModuleView& m, const char* name, ConVarHandle& out);

    // ------------------------------------------------------------------
    // Symbol registry.
    //
    // Declare a symbol once with the signature that identifies it, then ask
    // for it whenever you need it. Results are cached per module generation,
    // failures are logged exactly once, and a symbol with no signature
    // (pattern == nullptr) always resolves to 0 -- which is how a site whose
    // identity has NOT been re-established on the current build declares
    // itself unavailable instead of guessing.
    // ------------------------------------------------------------------
    enum class SymbolKind : uint8_t {
      // A function entry. Additionally validated with looksLikeFunctionEntry.
      Function,
      // A code site to be patched or read (mid-function is legitimate here).
      CodeSite,
      // A code site whose rel32 branch target is what the caller wants; the
      // resolver decodes it and returns the callee.
      CallTarget,
      // A global reached through a rip-relative operand in the matched
      // instruction; `dispOffset`/`instrLength` describe where it sits.
      RipRelativeData,
      // String-anchored: `anchorString` names a literal, and the function
      // pointer sitting beside the instruction that loads it is the answer
      // (the job/callback registration shape). Preferred over a byte
      // signature wherever a literal is available.
      StringAnchoredFunction,
      // String-anchored the other way round: the literal is used INSIDE the
      // function we want (a VPROF scope name, a warning), so walk back from
      // the reference to the enclosing function entry.
      StringEnclosingFunction,
    };

    struct SymbolDesc {
      const char* name         = nullptr;  // for logging; also the cache key
      const char* moduleName   = nullptr;  // "client.dll", "engine.dll", ...
      const char* pattern      = nullptr;  // nullptr => no byte signature
      const char* anchorString = nullptr;  // StringAnchoredFunction only
      SymbolKind  kind         = SymbolKind::Function;
      int32_t     addend       = 0;        // added to the match address
      uint8_t     dispOffset   = 0;        // RipRelativeData: disp32 offset within the match
      uint8_t     instrLength  = 0;        // RipRelativeData: length of the matched instruction
      int32_t     searchBefore = 0;        // StringAnchoredFunction: window, bytes before
      int32_t     searchAfter  = 0;        // StringAnchoredFunction: window, bytes after
    };

    // Resolve (and cache) a symbol. Returns 0 when unavailable for ANY reason:
    // module not loaded, no signature, zero matches, ambiguous matches, failed
    // entry validation, failed decode. Callers must handle 0 by disabling the
    // feature -- there is deliberately no way to get an unverified address out
    // of this API.
    uintptr_t resolve(const SymbolDesc& desc);

    // Convenience: true when the symbol resolved this generation.
    inline bool available(const SymbolDesc& desc) { return resolve(desc) != 0; }

    // Drop every cached resolution. Called automatically when a module's
    // generation moves; exposed for tests and for the dev dump command.
    void invalidateCache();

    // ------------------------------------------------------------------
    // Self-documenting report. Prints the fingerprint of every module we
    // touch plus the state of every symbol that has been asked for, in a
    // form that can be pasted straight into a bug report when a new game
    // build lands. Safe to call at any time.
    // ------------------------------------------------------------------
    void dumpReport();

  } // namespace EngineSymbols

  // ==================================================================
  // NV-DXVK [EngineEye] -- the engine's ground-truth eye position.
  //
  // This is what the old GetEngineEyeCM() was trying to obtain by calling
  // client.dll+0x14EAE0 and reading player+0x3D6C. Both halves of that were
  // wrong on this build: the address is not a function entry (it crashed),
  // and +0x3D6C is a static script anchor rather than a live eye field --
  // the [classify-eye-truth] log showed it pinned at (14158,-10801,877)
  // across 46 km of player travel.
  //
  // The replacement takes the eye from the viewmodel pass's c_cameraOrigin,
  // which IS Source's authoritative eye on this build (it tracks correctly in
  // pilot-on-foot, titan-cockpit and rodeo), and which Remix already captures
  // per draw with no module offsets whatsoever.
  //
  // Returns false when no eye is available yet; never crashes, never calls
  // into game code.
  // ==================================================================
  bool getEngineEyePosition(float outXyz[3]);

} // namespace dxvk
