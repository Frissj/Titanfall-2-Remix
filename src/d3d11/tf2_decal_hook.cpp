#include "tf2_decal_hook.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <windows.h>

#include "../util/log/log.h"
#include "../util/util_string.h"
#include "../dxvk/rtx_render/rtx_options.h"

namespace tf2_decal_hook {

  // ===== TLS counter ======================================================
  // We use a thread-local DEPTH COUNTER (not a flag) so that:
  //   - Multiple draw threads work independently (TF2 issues decals from
  //     several render-job threads in parallel).
  //   - Re-entrancy from the dispatcher (sub_1801B11B0 walking the command
  //     buffer) into a nested decal-emit is handled correctly.
  // The detour increments on entry and decrements on the unique return
  // point at the bottom of the wrapper; multi-return cleanup is implicit
  // because we wrap the entire original-function call.
  thread_local int t_decalDepth = 0;

  bool IsInDecalRender() {
    return t_decalDepth > 0;
  }

  // ===== Hook state =======================================================
  namespace {
    // Function pointer the wrapper calls — points into the trampoline page
    // we allocated, which executes the saved prologue then JMPs back to
    // engine.dll!sub_1801B4330+kPrologueSize.
    using DecalRenderFn = std::int64_t(__fastcall*)(std::int64_t a1,
                                                    std::uint32_t* a2,
                                                    int a3);
    DecalRenderFn s_origDecalRender = nullptr;

    // ===== Signature ======================================================
    // AOB pattern for sub_1801B4330 prologue. Captured from IDA at
    // engine.dll RVA 0x1B4330 (preferred base 0x180000000, so
    // 0x1801B4330 - 0x180000000 = 0x1B4330; an earlier revision of this
    // comment said 0xB4330, which is short by 0x100000 — the scan below
    // never used it, but anything computing a base from it would land
    // 1 MB off):
    //   44 89 44 24 18     mov [rsp+18], r8d         ; param spill (3rd arg)
    //   48 89 54 24 10     mov [rsp+10], rdx         ; param spill (2nd arg)
    //   56                 push rsi
    //   41 55              push r13
    //   B8 C8 80 00 00     mov eax, 0x80C8           ; stack frame size — distinctive
    //   E8 ?? ?? ?? ??     call __alloca_probe       ; rel32 (wildcarded)
    //   48 2B E0           sub rsp, rax
    //   48 8B F2           mov rsi, rdx
    //   45 8B E8           mov r13d, r8d
    //
    // The mov-immediate 0x80C8 is the unique fingerprint — a 32968-byte stack
    // frame on a 3-arg __alloca_probe'd function is not going to collide
    // anywhere else. Pattern length 32 bytes, single wildcard run for the
    // rel32 inside the call.
    constexpr std::size_t kPatternLen = 32;
    static constexpr std::uint8_t kPattern[kPatternLen] = {
      0x44, 0x89, 0x44, 0x24, 0x18,
      0x48, 0x89, 0x54, 0x24, 0x10,
      0x56,
      0x41, 0x55,
      0xB8, 0xC8, 0x80, 0x00, 0x00,
      0xE8, 0x00, 0x00, 0x00, 0x00,
      0x48, 0x2B, 0xE0,
      0x48, 0x8B, 0xF2,
      0x45, 0x8B, 0xE8,
    };
    // 0 = match required, 1 = wildcard (rel32 of the alloca_probe call).
    static constexpr std::uint8_t kMask[kPatternLen] = {
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0,
      0, 0,
      0, 0, 0, 0, 0,
      0, 1, 1, 1, 1,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
    };

    // Number of bytes we overwrite at the function entry. We need at least
    // 14 (for a `JMP [RIP+0]; <8-byte abs target>` indirect jump). The
    // smallest instruction boundary in the prologue at or above 14 is at
    // 18 bytes — the end of `mov eax, 0x80C8`. We copy those 18 bytes to
    // the trampoline. None of the five instructions in that span are
    // RIP-relative or position-dependent (no `jmp`/`call`/`mov [rip+x]`),
    // so the trampoline copy executes correctly at its new address.
    constexpr std::size_t kPrologueSize = 18;
    // Detour stub: `JMP [RIP+0]; <8-byte abs target>` = 14 bytes total.
    constexpr std::size_t kJmpStubSize = 14;

    // ===== Pattern scanner ================================================
    // Linear AOB scan over the module's .text section. engine.dll is ~349MB
    // total but .text is much smaller; we walk the PE section table to find
    // it and scan only that. Returns nullptr on no-match.
    std::uint8_t* ScanForPattern(HMODULE engine) {
      auto base = reinterpret_cast<std::uint8_t*>(engine);
      auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return nullptr;
      auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE)
        return nullptr;

      const auto* section = IMAGE_FIRST_SECTION(nt);
      const std::uint8_t* textBegin = nullptr;
      std::size_t textSize = 0;
      for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if (std::memcmp(section->Name, ".text", 5) == 0) {
          textBegin = base + section->VirtualAddress;
          textSize = section->Misc.VirtualSize;
          break;
        }
      }
      if (textBegin == nullptr || textSize < kPatternLen)
        return nullptr;

      // Tight linear scan. ~30 MB / GHz cycles ~= sub-30ms one-shot.
      const std::uint8_t* const end = textBegin + textSize - kPatternLen;
      for (const std::uint8_t* p = textBegin; p <= end; ++p) {
        bool match = true;
        for (std::size_t k = 0; k < kPatternLen; ++k) {
          if (kMask[k] == 0 && p[k] != kPattern[k]) {
            match = false;
            break;
          }
        }
        if (match)
          return const_cast<std::uint8_t*>(p);
      }
      return nullptr;
    }

    // ===== Detour wrapper =================================================
    // Increments the TLS counter, calls the original (via trampoline),
    // decrements, returns the result. Single return point — RAII would
    // also work but adds noise; explicit dec is clearer here.
    std::int64_t __fastcall DecalRenderWrapper(std::int64_t a1,
                                               std::uint32_t* a2,
                                               int a3) {
      // SEH-safe counter management. If engine.dll's sub_1801B4330 ever
      // raises a structured exception (job-system aborts, asserts), a
      // bare increment/decrement pair would leak the counter into the
      // next call on this thread — every subsequent draw would then
      // wrongly classify as decal. __finally runs during SEH unwind,
      // unlike C++ destructors. The cost is one TEB-relative entry on
      // the exception chain per call, which is negligible compared to
      // the work the wrapped function does.
      ++t_decalDepth;
      std::int64_t r = 0;
      __try {
        r = s_origDecalRender(a1, a2, a3);
      } __finally {
        --t_decalDepth;
      }
      return r;
    }

    // ===== Trampoline allocation ==========================================
    // Allocate an executable page near the target so the JMP-back from the
    // trampoline to engine.dll+kPrologueSize can use a 32-bit rel32 if
    // desired (we use absolute jmp here for simplicity, so proximity is
    // not strictly required — but allocating near keeps the address space
    // tidy).
    std::uint8_t* AllocateTrampolinePage() {
      // 4 KB is plenty for 18 bytes + 14-byte JMP back. VirtualAlloc
      // returns 64KB-aligned regions; RWX is acceptable here as this page
      // never holds user data.
      auto* p = static_cast<std::uint8_t*>(VirtualAlloc(
          nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      return p;
    }

    // ===== Install ========================================================
    // Returns true iff the hook is now live.
    bool DoInstall() {
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] engine.dll not loaded yet, skipping install.");
        return false;
      }

      std::uint8_t* target = ScanForPattern(engine);
      if (target == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] sub_1801B4330 prologue pattern NOT FOUND in "
          "engine.dll .text. Hook NOT installed (will fall back to no-op). "
          "Probable cause: engine.dll version mismatch or another mod has "
          "already patched the function entry.");
        return false;
      }
      dxvk::Logger::info(dxvk::str::format(
        "[tf2_decal_hook] Located sub_1801B4330 at 0x", std::hex,
        reinterpret_cast<std::uintptr_t>(target), std::dec));

      // Allocate trampoline page. Layout will be:
      //   [0..kPrologueSize)        copy of original prologue bytes
      //   [kPrologueSize..+14)      JMP [RIP+0]; <abs target = engine+kPrologueSize>
      std::uint8_t* tramp = AllocateTrampolinePage();
      if (tramp == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] VirtualAlloc for trampoline failed, aborting.");
        return false;
      }

      // Copy original prologue bytes verbatim. None of them are RIP-rel.
      std::memcpy(tramp, target, kPrologueSize);

      // Append `JMP [RIP+0]; <abs target>` returning into the original
      // function past the patched bytes.
      // Bytes: FF 25 00 00 00 00  <8-byte absolute address>
      std::uint8_t* jmpBack = tramp + kPrologueSize;
      jmpBack[0] = 0xFF;
      jmpBack[1] = 0x25;
      jmpBack[2] = 0x00;
      jmpBack[3] = 0x00;
      jmpBack[4] = 0x00;
      jmpBack[5] = 0x00;
      auto absTarget = reinterpret_cast<std::uintptr_t>(target + kPrologueSize);
      std::memcpy(jmpBack + 6, &absTarget, sizeof(absTarget));

      // s_origDecalRender = entry of the trampoline page.
      s_origDecalRender = reinterpret_cast<DecalRenderFn>(tramp);

      // Now overwrite the function entry with a JMP to our wrapper. Use
      // the same 14-byte indirect-jmp form so we don't clobber a register
      // at function entry (matters: at function entry, RCX/RDX/R8/R9 are
      // live with the arguments; clobbering RAX is technically fine in
      // Win64 but cleaner to avoid).
      std::uint8_t patchBytes[kPrologueSize] = {};
      patchBytes[0] = 0xFF;
      patchBytes[1] = 0x25;
      patchBytes[2] = 0x00;
      patchBytes[3] = 0x00;
      patchBytes[4] = 0x00;
      patchBytes[5] = 0x00;
      auto absWrapper = reinterpret_cast<std::uintptr_t>(&DecalRenderWrapper);
      std::memcpy(patchBytes + 6, &absWrapper, sizeof(absWrapper));
      // Pad the remaining 4 bytes with NOP so disassemblers stay aligned
      // and any debug-time single-step lands on a sane instruction.
      patchBytes[14] = 0x90;
      patchBytes[15] = 0x90;
      patchBytes[16] = 0x90;
      patchBytes[17] = 0x90;

      DWORD oldProtect = 0;
      if (!VirtualProtect(target, kPrologueSize, PAGE_EXECUTE_READWRITE,
                          &oldProtect)) {
        dxvk::Logger::warn(dxvk::str::format(
          "[tf2_decal_hook] VirtualProtect(RWX) failed at 0x", std::hex,
          reinterpret_cast<std::uintptr_t>(target), std::dec,
          " — hook NOT installed."));
        return false;
      }
      std::memcpy(target, patchBytes, kPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kPrologueSize, oldProtect, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kPrologueSize);

      dxvk::Logger::info(dxvk::str::format(
        "[tf2_decal_hook] Installed. target=0x", std::hex,
        reinterpret_cast<std::uintptr_t>(target),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", absWrapper, std::dec));
      return true;
    }

    // ===== Install gate ===================================================
    std::once_flag s_onceInstall;
    std::atomic<bool> s_installed{false};
  }  // namespace

  bool EnsureInstalled() {
    std::call_once(s_onceInstall, []() {
      s_installed.store(DoInstall(), std::memory_order_release);
    });
    return s_installed.load(std::memory_order_acquire);
  }

}  // namespace tf2_decal_hook

// ===========================================================================
// NV-DXVK [tf2_engine_cvars]
// ===========================================================================
namespace tf2_engine_cvars {

  namespace {
    // ConVar field offsets — verified against engine.dll's ConVar ctor
    // (sub_180416A40), which stores: [a1+56] = a1 (m_pParent self-pointer),
    // [a1+88] = atof(default) (m_fValue), [a1+92] = (int)m_fValue (m_nValue),
    // [a1+24] = name pointer, [a1+40] = flags.
    constexpr std::size_t kOffName   = 0x18;
    constexpr std::size_t kOffParent = 0x38;
    constexpr std::size_t kOffFValue = 0x58;
    constexpr std::size_t kOffNValue = 0x5C;

    // RVAs of the ConVar OBJECTS (not the m_pParent slots) in engine.dll.
    // Taken from the static ConVar constructors:
    //   sub_1805B0380 -> unk_193EC7900  staticProp_earlyDepthPrepass
    //   sub_1805B03C0 -> unk_193B87470  staticProp_earlyDepthPrepassDist
    //   sub_1805B0400 -> unk_193B87350  ...IncludeOpaques
    //   sub_1805B0440 -> unk_193B87110  ...IncludeOpaquesDist
    //   sub_1805B0340 -> unk_193EC7870  staticProp_drawDecalsInSortOrder
    // minus the 0x180000000 preferred base.
    struct CvarDef {
      const char*   name;
      std::uintptr_t rva;
    };

    enum CvarIndex : int {
      kEarlyDepthPrepass = 0,
      kEarlyDepthPrepassDist,
      kIncludeOpaques,
      kIncludeOpaquesDist,
      kDrawDecalsInSortOrder,
      kCvarCount
    };

    constexpr CvarDef kCvars[kCvarCount] = {
      { "staticProp_earlyDepthPrepass",                   0x13EC7900 },
      { "staticProp_earlyDepthPrepassDist",               0x13B87470 },
      { "staticProp_earlyDepthPrepassIncludeOpaques",     0x13B87350 },
      { "staticProp_earlyDepthPrepassIncludeOpaquesDist", 0x13B87110 },
      { "staticProp_drawDecalsInSortOrder",               0x13EC7870 },
    };

    // True iff [p, p+len) is committed and readable.
    bool IsReadable(const void* p, std::size_t len) {
      if (p == nullptr)
        return false;
      MEMORY_BASIC_INFORMATION mbi = {};
      if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0)
        return false;
      if (mbi.State != MEM_COMMIT)
        return false;
      constexpr DWORD kNoAccess = PAGE_NOACCESS | PAGE_GUARD;
      if ((mbi.Protect & kNoAccess) != 0)
        return false;
      const auto start = reinterpret_cast<std::uintptr_t>(p);
      const auto regionEnd =
        reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
      return start + len <= regionEnd;
    }

    // Resolve a cvar object and confirm its identity by name. Returns the
    // address to write (the parent), or nullptr if anything looks wrong.
    std::uint8_t* ResolveAndVerify(int index) {
      // Never cache a foreign module's base — re-resolve every call. The
      // engine can unload/reload DLLs and a stale base is an AV.
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr)
        return nullptr;

      auto* obj = reinterpret_cast<std::uint8_t*>(engine) + kCvars[index].rva;
      if (!IsReadable(obj, kOffNValue + sizeof(std::int32_t)))
        return nullptr;

      const char* name = *reinterpret_cast<const char* const*>(obj + kOffName);
      if (!IsReadable(name, 1))
        return nullptr;
      if (std::strcmp(name, kCvars[index].name) != 0)
        return nullptr;

      auto* parent = *reinterpret_cast<std::uint8_t* const*>(obj + kOffParent);
      if (parent == nullptr)
        parent = obj;  // pre-registration, or a build without the self-link
      if (!IsReadable(parent, kOffNValue + sizeof(std::int32_t)))
        return nullptr;
      return parent;
    }

    // Sentinel: rtx.conf value < 0 means "leave the engine default alone".
    constexpr float kLeaveAlone = -1.0f;

    // Remember what we last wrote so the per-frame call only logs on change.
    float s_lastApplied[kCvarCount] = {
      kLeaveAlone, kLeaveAlone, kLeaveAlone, kLeaveAlone, kLeaveAlone
    };
    bool s_warnedUnresolved[kCvarCount] = { false, false, false, false, false };

    void ApplyOne(int index, float value) {
      if (value < 0.0f)
        return;  // sentinel — not overridden

      std::uint8_t* parent = ResolveAndVerify(index);
      if (parent == nullptr) {
        if (!s_warnedUnresolved[index]) {
          s_warnedUnresolved[index] = true;
          dxvk::Logger::warn(dxvk::str::format(
            "[tf2_engine_cvars] could NOT verify '", kCvars[index].name,
            "' at engine.dll+0x", std::hex, kCvars[index].rva, std::dec,
            " — name check failed or memory unreadable. Skipping the write "
            "(fail-closed). Probable cause: different engine.dll build."));
        }
        return;
      }

      // .data is already RW in a loaded PE image, so no VirtualProtect.
      *reinterpret_cast<float*>(parent + kOffFValue) = value;
      *reinterpret_cast<std::int32_t*>(parent + kOffNValue) =
        static_cast<std::int32_t>(value);

      if (s_lastApplied[index] != value) {
        s_lastApplied[index] = value;
        dxvk::Logger::info(dxvk::str::format(
          "[tf2_engine_cvars] ", kCvars[index].name, " = ", value));
      }
    }
  }  // namespace

  void ApplyOverrides() {
    if (!dxvk::RtxOptions::tf2StaticPropCvarOverride())
      return;

    ApplyOne(kEarlyDepthPrepass,
             dxvk::RtxOptions::tf2StaticPropEarlyDepthPrepass());
    ApplyOne(kEarlyDepthPrepassDist,
             dxvk::RtxOptions::tf2StaticPropEarlyDepthPrepassDist());
    ApplyOne(kIncludeOpaques,
             dxvk::RtxOptions::tf2StaticPropIncludeOpaques());
    ApplyOne(kIncludeOpaquesDist,
             dxvk::RtxOptions::tf2StaticPropIncludeOpaquesDist());
    ApplyOne(kDrawDecalsInSortOrder,
             dxvk::RtxOptions::tf2StaticPropDrawDecalsInSortOrder());
  }

}  // namespace tf2_engine_cvars
