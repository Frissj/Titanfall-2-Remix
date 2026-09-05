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
#include "rtx_engine_symbols.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include <Windows.h>

#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace tf2 {
    // Producer lives in rtx_camera_manager.cpp, written from the viewmodel
    // pass's c_cameraOrigin in d3d11_rtx.cpp. See rtx_engine_symbols.h
    // [EngineEye] for why this, and not client.dll+0x3D6C, is the eye.
    extern std::atomic<float> g_pilotEyeX;
    extern std::atomic<float> g_pilotEyeY;
    extern std::atomic<float> g_pilotEyeZ;
    extern std::atomic<bool>  g_pilotEyeValid;
  }

  namespace EngineSymbols {

    namespace {

      // ----------------------------------------------------------------
      // Module generation bookkeeping.
      //
      // client.dll is unloaded when the player quits to desktop while the
      // dxvk CS thread is still running EndFrame. Any address cached across
      // that boundary points into an unmapped page. Rather than re-resolve
      // everything every call (the old code's approach, which cost a
      // GetModuleHandleA + VirtualQuery per frame and still got it wrong),
      // we stamp every cached value with a generation that moves whenever
      // the module's identity changes.
      // ----------------------------------------------------------------
      struct ModuleState {
        uintptr_t base          = 0;
        uint32_t  timeDateStamp = 0;
        uint64_t  generation    = 0;
      };

      std::mutex&                          stateMutex()  { static std::mutex m;                          return m; }
      std::map<std::string, ModuleState>&  moduleStates(){ static std::map<std::string, ModuleState> s;  return s; }
      std::atomic<uint64_t>                g_generationCounter { 1 };

      // Parsed section bounds, keyed by module name and validated against the
      // HMODULE so a reload forces a re-parse.
      std::map<std::string, ModuleView>& parsedModules() {
        static std::map<std::string, ModuleView> p;
        return p;
      }

      // Cached symbol resolution. `address == 0` with a matching generation is
      // a cached FAILURE -- deliberately sticky, so a symbol that cannot be
      // resolved does not re-scan several megabytes of .text every frame.
      struct SymbolCacheEntry {
        uintptr_t address    = 0;
        uint64_t  generation = 0;
        bool      resolved   = false;
        bool      logged     = false;
        std::string detail;              // human-readable outcome, for dumpReport
      };

      std::map<std::string, SymbolCacheEntry>& symbolCache() {
        static std::map<std::string, SymbolCacheEntry> c;
        return c;
      }

      // Modules we have looked at, for dumpReport.
      std::map<std::string, Fingerprint>& seenFingerprints() {
        static std::map<std::string, Fingerprint> f;
        return f;
      }

      constexpr uint64_t kFnvOffset = 1469598103934665603ull;
      constexpr uint64_t kFnvPrime  = 1099511628211ull;

      // --------------------------------------------------------------
      // Raw scan, deliberately free of C++ objects so it can sit inside a
      // structured exception handler. The module can be torn down by another
      // thread mid-scan; a fault here must degrade to "no match", not to a
      // dead renderer.
      // --------------------------------------------------------------
      size_t scanRawGuarded(const uint8_t* begin, const uint8_t* end,
                            const uint8_t* bytes, const uint8_t* wild,
                            size_t patternLen, uintptr_t* firstOut, size_t stopAfter) {
        size_t hits = 0;
        if (patternLen == 0 || end < begin || static_cast<size_t>(end - begin) < patternLen)
          return 0;

        // Anchor on the first non-wildcard byte so the common case is a
        // single memchr-like inner loop rather than a full compare per offset.
        size_t anchor = 0;
        while (anchor < patternLen && wild[anchor])
          ++anchor;
        const bool     haveAnchor = anchor < patternLen;
        const uint8_t  anchorByte = haveAnchor ? bytes[anchor] : 0;

        const uint8_t* last = end - patternLen;

#ifdef _MSC_VER
        __try {
#endif
          for (const uint8_t* p = begin; p <= last; ++p) {
            if (haveAnchor && p[anchor] != anchorByte)
              continue;
            size_t i = 0;
            for (; i < patternLen; ++i) {
              if (!wild[i] && p[i] != bytes[i])
                break;
            }
            if (i != patternLen)
              continue;
            if (hits == 0 && firstOut != nullptr)
              *firstOut = reinterpret_cast<uintptr_t>(p);
            if (++hits >= stopAfter)
              break;
          }
#ifdef _MSC_VER
        } __except (EXCEPTION_EXECUTE_HANDLER) {
          // Module vanished under us. Report "no match" -- callers fail safe.
          return 0;
        }
#endif
        return hits;
      }

      // Literal-byte search, same fail-safe contract and same SEH guard as the
      // pattern scan. Used to find a ConVar's name string in read-only data.
      size_t scanLiteralGuarded(const uint8_t* begin, const uint8_t* end,
                                const uint8_t* needle, size_t needleLen,
                                uintptr_t* firstOut, size_t stopAfter) {
        size_t hits = 0;
        if (needleLen == 0 || end < begin || static_cast<size_t>(end - begin) < needleLen)
          return 0;

        const uint8_t* last = end - needleLen;
#ifdef _MSC_VER
        __try {
#endif
          for (const uint8_t* p = begin; p <= last; ++p) {
            if (p[0] != needle[0])
              continue;
            if (std::memcmp(p, needle, needleLen) != 0)
              continue;
            if (hits == 0 && firstOut != nullptr)
              *firstOut = reinterpret_cast<uintptr_t>(p);
            if (++hits >= stopAfter)
              break;
          }
#ifdef _MSC_VER
        } __except (EXCEPTION_EXECUTE_HANDLER) {
          return 0;
        }
#endif
        return hits;
      }

      // 8-aligned qword search over writable data. Used to find the single
      // pointer to a ConVar's name string, i.e. its m_pszName member.
      size_t scanQwordGuarded(uintptr_t begin, uintptr_t end, uintptr_t value,
                              uintptr_t* firstOut, size_t stopAfter) {
        size_t hits = 0;
        if (end <= begin || end - begin < 8)
          return 0;

#ifdef _MSC_VER
        __try {
#endif
          for (uintptr_t p = (begin + 7) & ~uintptr_t(7); p + 8 <= end; p += 8) {
            if (*reinterpret_cast<const uintptr_t*>(p) != value)
              continue;
            if (hits == 0 && firstOut != nullptr)
              *firstOut = p;
            if (++hits >= stopAfter)
              break;
          }
#ifdef _MSC_VER
        } __except (EXCEPTION_EXECUTE_HANDLER) {
          return 0;
        }
#endif
        return hits;
      }

    } // anonymous namespace

    // ==================================================================
    // Memory predicates
    // ==================================================================

    static bool queryProtect(const void* p, size_t bytes, DWORD wantMask) {
      if (p == nullptr || bytes == 0)
        return false;

      const uint8_t* cur = static_cast<const uint8_t*>(p);
      const uint8_t* end = cur + bytes;

      // A range can span several regions; every one of them must qualify.
      while (cur < end) {
        MEMORY_BASIC_INFORMATION mbi = {};
        if (VirtualQuery(cur, &mbi, sizeof(mbi)) == 0)
          return false;
        if (mbi.State != MEM_COMMIT)
          return false;
        if ((mbi.Protect & PAGE_GUARD) || (mbi.Protect & PAGE_NOACCESS))
          return false;
        if ((mbi.Protect & wantMask) == 0)
          return false;

        const uint8_t* regionEnd = static_cast<const uint8_t*>(mbi.BaseAddress) + mbi.RegionSize;
        if (regionEnd <= cur)
          return false;                 // no forward progress: bail rather than spin
        cur = regionEnd;
      }
      return true;
    }

    bool readable(const void* p, size_t bytes) {
      constexpr DWORD kRead = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY
                            | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
      return queryProtect(p, bytes, kRead);
    }

    bool writable(const void* p, size_t bytes) {
      constexpr DWORD kWrite = PAGE_READWRITE | PAGE_WRITECOPY
                             | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
      return queryProtect(p, bytes, kWrite);
    }

    bool executable(const void* p) {
      constexpr DWORD kExec = PAGE_EXECUTE | PAGE_EXECUTE_READ
                            | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
      return queryProtect(p, 1, kExec);
    }

    // ==================================================================
    // Module / PE parsing
    // ==================================================================

    bool queryModule(const char* moduleName, ModuleView& out) {
      out = ModuleView();
      if (moduleName == nullptr)
        return false;

      HMODULE mod = GetModuleHandleA(moduleName);
      if (mod == nullptr)
        return false;                   // not loaded (yet, or any more)

      // Fast path. resolve() is on the per-frame path, and re-walking the PE
      // section table (a dozen VirtualQuery calls) every frame for every
      // symbol would be a real cost for no information. The handle changing is
      // what tells us the module was reloaded and the parse must be redone.
      {
        std::lock_guard<std::mutex> lock(stateMutex());
        auto it = parsedModules().find(moduleName);
        if (it != parsedModules().end() && it->second.handle == mod) {
          out = it->second;
          return true;
        }
      }

      const uintptr_t base = reinterpret_cast<uintptr_t>(mod);
      if (!readable(reinterpret_cast<const void*>(base), sizeof(IMAGE_DOS_HEADER)))
        return false;

      const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
      if (dos->e_lfanew <= 0 || dos->e_lfanew > 0x10000)
        return false;

      const uintptr_t ntAddr = base + static_cast<uintptr_t>(dos->e_lfanew);
      if (!readable(reinterpret_cast<const void*>(ntAddr), sizeof(IMAGE_NT_HEADERS64)))
        return false;

      const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(ntAddr);
      if (nt->Signature != IMAGE_NT_SIGNATURE)
        return false;
      if (nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return false;

      out.handle        = mod;
      out.base          = base;
      out.imageSize     = nt->OptionalHeader.SizeOfImage;
      out.timeDateStamp = nt->FileHeader.TimeDateStamp;
      out.peCheckSum    = nt->OptionalHeader.CheckSum;

      // Exception directory (.pdata). Present on every x64 PE that contains
      // non-leaf functions; this is what makes function-entry validation exact.
      {
        const IMAGE_DATA_DIRECTORY& dir =
          nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        if (dir.VirtualAddress != 0 && dir.Size >= sizeof(RUNTIME_FUNCTION)) {
          const uintptr_t pb = base + dir.VirtualAddress;
          const uintptr_t pe = pb + (dir.Size - dir.Size % sizeof(RUNTIME_FUNCTION));
          if (pe > pb && readable(reinterpret_cast<const void*>(pb), pe - pb)) {
            out.pdataBegin = pb;
            out.pdataEnd   = pe;
          }
        }
      }

      const auto* sec = IMAGE_FIRST_SECTION(nt);
      const WORD  count = nt->FileHeader.NumberOfSections;
      if (!readable(sec, sizeof(IMAGE_SECTION_HEADER) * count))
        return false;

      for (WORD i = 0; i < count; ++i) {
        const uintptr_t sBegin = base + sec[i].VirtualAddress;
        const DWORD     sSize  = sec[i].Misc.VirtualSize ? sec[i].Misc.VirtualSize
                                                         : sec[i].SizeOfRawData;
        if (sSize == 0)
          continue;
        const uintptr_t sEnd = sBegin + sSize;
        const DWORD     ch   = sec[i].Characteristics;

        if (ch & IMAGE_SCN_MEM_EXECUTE) {
          out.codeBegin = out.codeBegin ? std::min(out.codeBegin, sBegin) : sBegin;
          out.codeEnd   = std::max(out.codeEnd, sEnd);
        } else if (ch & IMAGE_SCN_MEM_WRITE) {
          out.dataBegin = out.dataBegin ? std::min(out.dataBegin, sBegin) : sBegin;
          out.dataEnd   = std::max(out.dataEnd, sEnd);
        } else if (ch & IMAGE_SCN_MEM_READ) {
          out.rdataBegin = out.rdataBegin ? std::min(out.rdataBegin, sBegin) : sBegin;
          out.rdataEnd   = std::max(out.rdataEnd, sEnd);
        }
      }

      if (out.codeEnd <= out.codeBegin)
        return false;

      // Generation: a fresh number whenever base or link timestamp changes.
      {
        std::lock_guard<std::mutex> lock(stateMutex());
        ModuleState& st = moduleStates()[moduleName];
        if (st.base != base || st.timeDateStamp != out.timeDateStamp || st.generation == 0) {
          st.base          = base;
          st.timeDateStamp = out.timeDateStamp;
          st.generation    = g_generationCounter.fetch_add(1) + 1;
        }
        out.generation = st.generation;
        parsedModules()[moduleName] = out;
      }

      return true;
    }

    // ==================================================================
    // Fingerprint
    // ==================================================================

    Fingerprint fingerprintOf(const ModuleView& m) {
      Fingerprint fp;
      if (!m.valid())
        return fp;

      fp.imageSize     = m.imageSize;
      fp.timeDateStamp = m.timeDateStamp;
      fp.peCheckSum    = m.peCheckSum;

      // Sparse FNV-1a over .text. 4096 samples is enough to separate builds
      // and costs microseconds; hashing all 7 MB would not.
      const size_t span = m.codeEnd - m.codeBegin;
      if (span >= 4096 && readable(reinterpret_cast<const void*>(m.codeBegin), 4096)) {
        const size_t kSamples = 4096;
        const size_t stride   = span / kSamples;
        uint64_t     h        = kFnvOffset;
        for (size_t i = 0; i < kSamples; ++i) {
          const uintptr_t at = m.codeBegin + i * stride;
          if (!readable(reinterpret_cast<const void*>(at), 1)) {
            h = 0;
            break;
          }
          h ^= *reinterpret_cast<const uint8_t*>(at);
          h *= kFnvPrime;
        }
        fp.codeHash = h;
      }

      return fp;
    }

    std::string Fingerprint::toString() const {
      return str::format("size=0x", std::hex, imageSize,
                         " ts=0x", timeDateStamp,
                         " sum=0x", peCheckSum,
                         " code=0x", codeHash, std::dec);
    }

    // ==================================================================
    // Pattern
    // ==================================================================

    Pattern::Pattern(const char* idaStyle) {
      if (idaStyle == nullptr)
        return;

      auto hexVal = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };

      for (const char* p = idaStyle; *p != '\0'; ) {
        if (std::isspace(static_cast<unsigned char>(*p))) { ++p; continue; }

        if (*p == '?') {
          m_bytes.push_back(0);
          m_wild.push_back(1);
          ++p;
          if (*p == '?') ++p;           // accept both "?" and "??"
          continue;
        }

        const int hi = hexVal(p[0]);
        const int lo = (hi >= 0 && p[1] != '\0') ? hexVal(p[1]) : -1;
        if (hi < 0 || lo < 0) {         // malformed -- reject the whole pattern
          m_bytes.clear();
          m_wild.clear();
          return;
        }
        m_bytes.push_back(static_cast<uint8_t>((hi << 4) | lo));
        m_wild.push_back(0);
        p += 2;
      }

      // A pattern that is entirely wildcards, or shorter than 6 bytes, cannot
      // uniquely identify anything in a multi-megabyte .text. Reject it here
      // rather than let it produce a confident wrong answer.
      size_t concrete = 0;
      for (uint8_t w : m_wild) {
        if (!w) ++concrete;
      }
      if (m_bytes.size() < 6 || concrete < 5) {
        m_bytes.clear();
        m_wild.clear();
      }
    }

    size_t Pattern::countMatches(uintptr_t begin, uintptr_t end,
                                 uintptr_t* firstOut, size_t stopAfter) const {
      if (!valid() || end <= begin)
        return 0;

      // One readability check for the whole range, then a raw loop. Checking
      // per byte would be a VirtualQuery per byte across several megabytes.
      if (!readable(reinterpret_cast<const void*>(begin), end - begin))
        return 0;

      return scanRawGuarded(reinterpret_cast<const uint8_t*>(begin),
                            reinterpret_cast<const uint8_t*>(end),
                            m_bytes.data(), m_wild.data(), m_bytes.size(),
                            firstOut, stopAfter);
    }

    // ==================================================================
    // Function-entry validation
    //
    // This is the check whose absence is the whole bug: the old code proved
    // only that client.dll+0x14EAE0 was committed and executable, which is
    // true of every byte inside every function.
    // ==================================================================

    bool functionBounds(const ModuleView& m, uintptr_t addr,
                        uintptr_t& beginOut, uintptr_t& endOut) {
      beginOut = 0;
      endOut   = 0;
      if (!m.valid() || m.pdataEnd <= m.pdataBegin || !m.containsCode(addr))
        return false;

      const auto* table = reinterpret_cast<const RUNTIME_FUNCTION*>(m.pdataBegin);
      const size_t count = (m.pdataEnd - m.pdataBegin) / sizeof(RUNTIME_FUNCTION);
      if (count == 0)
        return false;

      const uint32_t rva = static_cast<uint32_t>(addr - m.base);

      // .pdata is sorted by BeginAddress, so this is a binary search.
      size_t lo = 0, hi = count - 1;
      const RUNTIME_FUNCTION* hit = nullptr;
      while (lo <= hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const RUNTIME_FUNCTION& rf = table[mid];
        if (rva < rf.BeginAddress) {
          if (mid == 0)
            break;
          hi = mid - 1;
        } else if (rva >= rf.EndAddress) {
          lo = mid + 1;
        } else {
          hit = &rf;
          break;
        }
      }
      if (hit == nullptr)
        return false;                   // leaf function, or not code we can bound

      uint32_t begin = hit->BeginAddress;
      uint32_t end   = hit->EndAddress;
      uint32_t unwind = hit->UnwindData;

      // Follow UNW_FLAG_CHAININFO to the primary entry. Bounded: a malformed
      // or hostile chain must not spin.
      for (int hop = 0; hop < 8; ++hop) {
        const uintptr_t ui = m.base + unwind;
        if (!m.containsImage(ui) || !readable(reinterpret_cast<const void*>(ui), 4))
          break;

        const uint8_t* info  = reinterpret_cast<const uint8_t*>(ui);
        const uint8_t  flags = static_cast<uint8_t>(info[0] >> 3);
        if ((flags & 0x4) == 0)         // UNW_FLAG_CHAININFO
          break;

        const uint8_t codes = info[2];
        const size_t  slots = (static_cast<size_t>(codes) + 1u) & ~size_t(1);
        const uintptr_t chained = ui + 4 + slots * 2;
        if (!m.containsImage(chained)
            || !readable(reinterpret_cast<const void*>(chained), sizeof(RUNTIME_FUNCTION)))
          break;

        const auto* rf = reinterpret_cast<const RUNTIME_FUNCTION*>(chained);
        if (rf->BeginAddress == 0 || rf->EndAddress <= rf->BeginAddress)
          break;
        begin  = rf->BeginAddress;
        end    = rf->EndAddress;
        unwind = rf->UnwindData;
      }

      const uintptr_t b = m.base + begin;
      const uintptr_t e = m.base + end;
      if (!m.containsCode(b) || e <= b)
        return false;

      beginOut = b;
      endOut   = e;
      return true;
    }

    bool looksLikeFunctionEntry(const ModuleView& m, uintptr_t addr) {
      if (!m.valid() || !m.containsCode(addr))
        return false;
      if (!readable(reinterpret_cast<const void*>(addr), 16))
        return false;

      // Authoritative path. .pdata knows exactly where each function starts,
      // so there is nothing to infer: the address either IS a function start
      // or it is not.
      if (m.pdataEnd > m.pdataBegin) {
        uintptr_t begin = 0, end = 0;
        if (!functionBounds(m, addr, begin, end))
          return false;                 // inside no known function -> refuse
        return begin == addr;
      }

      // Fallback for a module with no exception directory. Weaker, and only
      // reached if .pdata is absent entirely.
      const uint8_t* p = reinterpret_cast<const uint8_t*>(addr);

      // (1) Placement. A function entry is either 16-byte aligned (MSVC's
      //     default with /O2) or immediately preceded by inter-function
      //     padding.
      //
      //     Placement ALONE is not sufficient, and the crash this module
      //     exists to prevent is the proof: client.dll+0x14EAE0 is 16-byte
      //     aligned AND preceded by a `ret`, yet it is the middle of a
      //     function -- the byte before it is the tail of the PREVIOUS basic
      //     block, not the end of the previous function. The shape test in
      //     (2) is what rejects it.
      bool placementOk = (addr & 0xF) == 0;
      if (!placementOk && addr > m.codeBegin
          && readable(reinterpret_cast<const void*>(addr - 1), 1)) {
        const uint8_t prev = *reinterpret_cast<const uint8_t*>(addr - 1);
        placementOk = (prev == 0xCC) || (prev == 0x90);   // int3 / nop padding
      }
      if (!placementOk)
        return false;

      // (2) Shape. Accept only first instructions that MSVC actually emits at
      //     a function entry. Deliberately conservative: a false negative
      //     disables a feature, a false positive means calling or patching the
      //     middle of one.
      //
      //     0x14EAE0 begins `F3 0F 5C A9 ..` (subss xmm5, [rcx+disp32]) -- an
      //     SSE arithmetic op on a memory operand, which no prologue starts
      //     with, so it is rejected here.
      auto isRipOrRegModrm = [](uint8_t modrm) {
        // rip-relative (mod=00, rm=101) or register-direct (mod=11).
        return modrm >= 0xC0 || (modrm & 0xC7) == 0x05;
      };

      if (p[0] == 0xCC || p[0] == 0x00)                    return false; // padding / not code

      if (p[0] >= 0x50 && p[0] <= 0x57)                    return true;  // push r{ax..di}
      if ((p[0] == 0x41 || p[0] == 0x40) &&
          p[1] >= 0x50 && p[1] <= 0x57)                    return true;  // push r8..r15 / REX push

      if (p[0] == 0x48 && p[1] == 0x83 && p[2] == 0xEC)    return true;  // sub rsp, imm8
      if (p[0] == 0x48 && p[1] == 0x81 && p[2] == 0xEC)    return true;  // sub rsp, imm32

      if (p[0] == 0x48 && p[1] == 0x8B && isRipOrRegModrm(p[2])) return true; // mov r64, r64 / [rip+d]
      if (p[0] == 0x4C && p[1] == 0x8B && isRipOrRegModrm(p[2])) return true; // mov r8-15, ...
      if (p[0] == 0x48 && p[1] == 0x8D)                    return true;  // lea (tiny accessor / frame)

      // Register spills into the shadow store. 44 89 44 24 xx is exactly the
      // documented C_BaseAnimating::SetupBones prologue.
      if (p[0] == 0x48 && p[1] == 0x89 && p[3] == 0x24)    return true;  // mov [rsp+x], r64
      if (p[0] == 0x4C && p[1] == 0x89 && p[3] == 0x24)    return true;  // mov [rsp+x], r8-15
      if (p[0] == 0x44 && p[1] == 0x89 && p[3] == 0x24)    return true;  // mov [rsp+x], r8d-r15d
      if (p[0] == 0x89 && p[2] == 0x24)                    return true;  // mov [rsp+x], r32

      if (p[0] == 0x33 && p[1] >= 0xC0)                    return true;  // xor r32, r32
      if (p[0] == 0x31 && p[1] >= 0xC0)                    return true;  // xor r32, r32 (alt)
      if (p[0] == 0xB8)                                    return true;  // mov eax, imm32
      if (p[0] == 0x48 && p[1] == 0x85)                    return true;  // test r64, r64
      if (p[0] == 0x48 && p[1] == 0x83 && p[2] >= 0xF8)    return true;  // cmp r64, imm8
      if (p[0] == 0xE9 || p[0] == 0xEB)                    return true;  // ICF / thunk jump
      if (p[0] == 0xFF && p[1] == 0x25)                    return true;  // jmp [rip+d32] (import thunk)
      if (p[0] == 0xC3 && p[1] == 0xCC)                    return true;  // empty stub

      return false;
    }

    // ==================================================================
    // Instruction decoding
    // ==================================================================

    bool decodeRel32Target(const ModuleView& m, uintptr_t site, uintptr_t& targetOut) {
      targetOut = 0;
      if (!m.valid() || !m.containsCode(site))
        return false;
      if (!readable(reinterpret_cast<const void*>(site), 5))
        return false;

      const uint8_t op = *reinterpret_cast<const uint8_t*>(site);
      if (op != 0xE8 && op != 0xE9)
        return false;

      int32_t rel = 0;
      std::memcpy(&rel, reinterpret_cast<const void*>(site + 1), sizeof(rel));
      const uintptr_t target = site + 5 + static_cast<intptr_t>(rel);

      if (!m.containsCode(target))
        return false;

      targetOut = target;
      return true;
    }

    bool decodeRipRelative(const ModuleView& m, uintptr_t dispAt,
                           uintptr_t instrEnd, uintptr_t& targetOut) {
      targetOut = 0;
      if (!m.valid() || !m.containsCode(dispAt) || instrEnd <= dispAt)
        return false;
      if (!readable(reinterpret_cast<const void*>(dispAt), 4))
        return false;

      int32_t disp = 0;
      std::memcpy(&disp, reinterpret_cast<const void*>(dispAt), sizeof(disp));
      const uintptr_t target = instrEnd + static_cast<intptr_t>(disp);

      if (!m.containsImage(target))
        return false;

      targetOut = target;
      return true;
    }

    // ==================================================================
    // String-anchored resolution
    // ==================================================================

    namespace {
      // Find every position in [begin,end) holding a disp32 that, interpreted
      // as a rip-relative operand of an instruction ending 4 bytes later,
      // points at `target`. Requires the caller to have proven readability.
      size_t scanRipRefGuarded(uintptr_t begin, uintptr_t end, uintptr_t target,
                               uintptr_t* firstOut, size_t stopAfter) {
        size_t hits = 0;
        if (end <= begin || end - begin < 4)
          return 0;

#ifdef _MSC_VER
        __try {
#endif
          for (uintptr_t q = begin; q + 4 <= end; ++q) {
            int32_t disp;
            std::memcpy(&disp, reinterpret_cast<const void*>(q), sizeof(disp));
            if (q + 4 + static_cast<intptr_t>(disp) != static_cast<intptr_t>(target))
              continue;
            if (hits == 0 && firstOut != nullptr)
              *firstOut = q;
            if (++hits >= stopAfter)
              break;
          }
#ifdef _MSC_VER
        } __except (EXCEPTION_EXECUTE_HANDLER) {
          return 0;
        }
#endif
        return hits;
      }
    } // anonymous namespace

    bool findStringRef(const ModuleView& m, const char* text,
                       uintptr_t& stringAddrOut, uintptr_t& dispSiteOut) {
      stringAddrOut = 0;
      dispSiteOut   = 0;
      if (!m.valid() || text == nullptr)
        return false;

      const size_t len = std::strlen(text);
      if (len < 4 || len > 190)         // too short to be distinctive
        return false;
      if (m.rdataEnd <= m.rdataBegin)
        return false;
      if (!readable(reinterpret_cast<const void*>(m.rdataBegin), m.rdataEnd - m.rdataBegin))
        return false;

      // "\0<text>\0": the surrounding terminators keep a literal from matching
      // inside a longer one that happens to contain it.
      std::vector<char> needle(len + 3);
      needle[0] = '\0';
      std::memcpy(needle.data() + 1, text, len);
      needle[len + 1] = '\0';

      uintptr_t strAddr = 0;
      if (scanLiteralGuarded(reinterpret_cast<const uint8_t*>(m.rdataBegin),
                             reinterpret_cast<const uint8_t*>(m.rdataEnd),
                             reinterpret_cast<const uint8_t*>(needle.data()),
                             len + 2, &strAddr, 2) != 1)
        return false;
      strAddr += 1;

      if (!readable(reinterpret_cast<const void*>(m.codeBegin), m.codeEnd - m.codeBegin))
        return false;

      uintptr_t site = 0;
      if (scanRipRefGuarded(m.codeBegin, m.codeEnd, strAddr, &site, 2) != 1)
        return false;

      stringAddrOut = strAddr;
      dispSiteOut   = site;
      return true;
    }

    bool findEnclosingFunction(const ModuleView& m, uintptr_t insideAddr,
                               uint32_t maxScanBack, uintptr_t& functionOut) {
      functionOut = 0;
      if (!m.valid() || !m.containsCode(insideAddr))
        return false;
      if (maxScanBack == 0 || maxScanBack > 0x4000)
        return false;

      // Exact, via .pdata (including chained-chunk resolution). No scanning
      // and no heuristic: a padding walk-back would fail outright here anyway,
      // because R_DrawWorldMeshes' entry is preceded by the previous
      // function's tail `jmp qword ptr [rax+8]` rather than by int3 padding.
      uintptr_t begin = 0, end = 0;
      if (functionBounds(m, insideAddr, begin, end)) {
        // maxScanBack stays meaningful as a sanity bound: a literal an
        // implausible distance from the entry suggests we followed the wrong
        // chain, so fail safe rather than hand back a distant function.
        if (insideAddr >= begin && insideAddr - begin <= maxScanBack) {
          functionOut = begin;
          return true;
        }
        return false;
      }

      // Fallback for a leaf function or a module with no exception directory:
      // walk back to the nearest candidate that is int3-padded, 16-byte
      // aligned AND prologue-shaped. All three together, because 0xCC also
      // occurs as an immediate operand mid-function and landing on one would
      // put a hook inside a function -- the failure mode this module exists
      // to prevent.
      const uintptr_t lo = std::max(m.codeBegin, insideAddr > maxScanBack
                                                 ? insideAddr - maxScanBack : m.codeBegin);
      if (!readable(reinterpret_cast<const void*>(lo), insideAddr - lo + 16))
        return false;

      for (uintptr_t a = insideAddr & ~uintptr_t(0xF); a > lo; a -= 16) {
        if (*reinterpret_cast<const uint8_t*>(a - 1) != 0xCC)
          continue;
        if (!looksLikeFunctionEntry(m, a))
          continue;
        functionOut = a;
        return true;
      }
      return false;
    }

    bool findCodePointerNear(const ModuleView& m, uintptr_t dispSite,
                             int32_t before, int32_t after, uintptr_t& functionOut) {
      functionOut = 0;
      if (!m.valid() || !m.containsCode(dispSite))
        return false;
      if (before < 0 || after < 0 || before > 256 || after > 256)
        return false;

      const uintptr_t lo = (dispSite > static_cast<uintptr_t>(before))
                         ? std::max(m.codeBegin, dispSite - before) : m.codeBegin;
      const uintptr_t hi = std::min(m.codeEnd, dispSite + after);
      if (hi <= lo + 4)
        return false;
      if (!readable(reinterpret_cast<const void*>(lo), hi - lo))
        return false;

      uintptr_t found = 0;
      size_t    hits  = 0;

      for (uintptr_t q = lo; q + 4 <= hi; ++q) {
        // The string operand itself is not a candidate.
        if (q >= dispSite && q < dispSite + 4)
          continue;

        int32_t disp;
        std::memcpy(&disp, reinterpret_cast<const void*>(q), sizeof(disp));
        const uintptr_t target = q + 4 + static_cast<intptr_t>(disp);

        // Only executable targets that are also plausible function entries
        // count. That is what excludes IAT thunks, data references and the
        // incidental disp32 that happens to land inside .text.
        if (!m.containsCode(target))
          continue;
        if (!looksLikeFunctionEntry(m, target))
          continue;

        if (hits == 0)
          found = target;
        else if (target != found)
          return false;                 // two different candidates: ambiguous
        ++hits;
      }

      if (found == 0)
        return false;

      functionOut = found;
      return true;
    }

    // ==================================================================
    // CreateInterface
    // ==================================================================

    void* createInterface(const ModuleView& m, const char* baseName,
                          int minVersion, int maxVersion, int* versionOut) {
      if (versionOut != nullptr)
        *versionOut = 0;
      if (!m.valid() || baseName == nullptr)
        return nullptr;
      if (minVersion < 0 || maxVersion < minVersion || maxVersion > 999)
        return nullptr;

      using CreateInterfaceFn = void* (*)(const char*, int*);
      auto fn = reinterpret_cast<CreateInterfaceFn>(
        GetProcAddress(static_cast<HMODULE>(m.handle), "CreateInterface"));
      if (fn == nullptr)
        return nullptr;

      // Source compares interface names with an exact strcmp, so there is no
      // prefix match to lean on; sweeping the version range is what makes this
      // survive a version bump.
      char name[128];
      for (int v = minVersion; v <= maxVersion; ++v) {
        const int written = std::snprintf(name, sizeof(name), "%s%03d", baseName, v);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name))
          return nullptr;

        int rc = 0;
        void* iface = fn(name, &rc);
        if (iface == nullptr)
          continue;

        // An interface object must start with a vtable pointer into the
        // owning module's read-only data, and that vtable's first slot must
        // point at executable code. Anything else is not an interface.
        if (!readable(iface, sizeof(void*)))
          continue;
        const uintptr_t vtbl = *reinterpret_cast<const uintptr_t*>(iface);
        if (!m.containsRData(vtbl) || !readable(reinterpret_cast<const void*>(vtbl), sizeof(void*)))
          continue;
        const uintptr_t slot0 = *reinterpret_cast<const uintptr_t*>(vtbl);
        if (!executable(reinterpret_cast<const void*>(slot0)))
          continue;

        if (versionOut != nullptr)
          *versionOut = v;
        return iface;
      }

      return nullptr;
    }

    bool vtableSlot(const ModuleView& owner, void* object, uint32_t index, void*& fnOut) {
      fnOut = nullptr;
      if (!owner.valid() || object == nullptr || index > 512)
        return false;
      if (!readable(object, sizeof(void*)))
        return false;

      const uintptr_t vtbl = *reinterpret_cast<const uintptr_t*>(object);
      if (!owner.containsRData(vtbl))
        return false;

      const uintptr_t slotAddr = vtbl + static_cast<uintptr_t>(index) * sizeof(void*);
      if (!owner.containsRData(slotAddr) || !readable(reinterpret_cast<const void*>(slotAddr), sizeof(void*)))
        return false;

      const uintptr_t fn = *reinterpret_cast<const uintptr_t*>(slotAddr);
      if (!owner.containsCode(fn) || !executable(reinterpret_cast<const void*>(fn)))
        return false;

      fnOut = reinterpret_cast<void*>(fn);
      return true;
    }

    // ==================================================================
    // ConVar discovery by name
    //
    // Replaces "the r_lod ConVar object is client.dll+0x11C2E70" with
    // something that cannot go stale: the object points at its own name.
    // ==================================================================

    bool findConVar(const ModuleView& m, const char* name, ConVarHandle& out) {
      out = ConVarHandle();
      if (!m.valid() || name == nullptr)
        return false;

      const size_t nameLen = std::strlen(name);
      if (nameLen == 0 || nameLen > 64)
        return false;
      if (m.rdataEnd <= m.rdataBegin || m.dataEnd <= m.dataBegin)
        return false;

      // (1) Locate the name literal in read-only data, as "\0<name>\0". The
      //     leading terminator is what stops "r_lod" from matching inside
      //     "mat_r_lod"; string pools always separate literals with a NUL.
      if (!readable(reinterpret_cast<const void*>(m.rdataBegin), m.rdataEnd - m.rdataBegin))
        return false;

      char needle[66];
      needle[0] = '\0';
      std::memcpy(needle + 1, name, nameLen + 1);   // "\0" + name + "\0"

      uintptr_t strAddr = 0;
      const size_t strHits = scanLiteralGuarded(
        reinterpret_cast<const uint8_t*>(m.rdataBegin),
        reinterpret_cast<const uint8_t*>(m.rdataEnd),
        reinterpret_cast<const uint8_t*>(needle), nameLen + 2, &strAddr, 2);
      if (strHits != 1)
        return false;                   // absent or ambiguous -> fail safe
      strAddr += 1;                     // skip the leading terminator

      // (2) Find the single writable qword pointing at it. That is the
      //     registered ConVar's m_pszName, written by its constructor at
      //     runtime -- which is why this cannot be read out of a static IDB
      //     and must be done live.
      if (!readable(reinterpret_cast<const void*>(m.dataBegin), m.dataEnd - m.dataBegin))
        return false;

      uintptr_t nameSlot = 0;
      const size_t slotHits = scanQwordGuarded(m.dataBegin, m.dataEnd, strAddr, &nameSlot, 2);
      if (slotHits != 1)
        return false;

      // (3) Walk back to the object start. Rather than assume the Source
      //     ConCommandBase layout (m_pszName at +0x18), try every 8-aligned
      //     candidate and keep the one that passes the full structural check.
      //     Exactly one must qualify.
      //
      //     A registered ConVar looks like:
      //       +0x00  ConCommandBase vtable   -> read-only data
      //       +0x18  m_pszName               -> the slot we just found
      //       +0x30  IConVar vtable          -> read-only data
      //       +0x38  m_pParent               -> itself, or another ConVar
      //       +0x40  m_pszDefaultValue       -> a read-only string
      //       +0x58  m_fValue  (float)
      //       +0x5C  m_nValue  (int)
      uintptr_t object      = 0;
      uintptr_t parent      = 0;
      size_t    objectHits  = 0;

      for (size_t back = 0; back <= 0x40; back += 8) {
        if (nameSlot < back)
          break;
        const uintptr_t cand = nameSlot - back;
        if (!m.containsData(cand) || !readable(reinterpret_cast<const void*>(cand), 0x60))
          continue;

        const uintptr_t vtbl0 = *reinterpret_cast<const uintptr_t*>(cand);
        if (!m.containsRData(vtbl0))
          continue;

        // The parent slot: a qword within the object pointing at a ConVar --
        // most commonly the object itself.
        uintptr_t parentCand     = 0;
        size_t    parentSlotHits = 0;
        for (size_t off = 8; off < 0x60; off += 8) {
          const uintptr_t v = *reinterpret_cast<const uintptr_t*>(cand + off);
          if (v == cand) {
            parentCand = cand;
            ++parentSlotHits;
          } else if (m.containsData(v) && v != 0 && readable(reinterpret_cast<const void*>(v), 0x60)) {
            // A different object is only a plausible parent if it also looks
            // like a ConVar (read-only-data vtable in slot 0).
            const uintptr_t pv = *reinterpret_cast<const uintptr_t*>(v);
            if (m.containsRData(pv) && *reinterpret_cast<const uintptr_t*>(v + (nameSlot - cand)) == strAddr) {
              parentCand = v;
              ++parentSlotHits;
            }
          }
        }
        if (parentSlotHits != 1)
          continue;

        object     = cand;
        parent     = parentCand;
        ++objectHits;
        if (objectHits >= 2)
          break;
      }
      if (objectHits != 1 || object == 0 || parent == 0)
        return false;

      // (4) Anchor on m_pszDefaultValue: the one qword in the object pointing
      //     at a short, printable, NUMERIC string in read-only data. That
      //     string is the ConVar's own statement of what its value should be,
      //     and it is what lets us identify the value fields without knowing
      //     where they sit.
      if (!readable(reinterpret_cast<const void*>(parent), 0x80))
        return false;

      double defaultValue  = 0.0;
      bool   haveDefault   = false;
      size_t defaultHits   = 0;
      for (size_t off = 8; off + 8 <= 0x80; off += 8) {
        const uintptr_t sp = *reinterpret_cast<const uintptr_t*>(parent + off);
        if (!m.containsRData(sp) || !readable(reinterpret_cast<const void*>(sp), 2))
          continue;

        char buf[32] = {};
        size_t n = 0;
        for (; n < sizeof(buf) - 1; ++n) {
          if (!readable(reinterpret_cast<const void*>(sp + n), 1))
            break;
          const char c = *reinterpret_cast<const char*>(sp + n);
          if (c == '\0')
            break;
          buf[n] = c;
        }
        if (n == 0 || n >= sizeof(buf) - 1)
          continue;

        // Numeric literal only: optional sign, digits, optional single '.'.
        size_t idx = (buf[0] == '-' || buf[0] == '+') ? 1u : 0u;
        size_t digits = 0, dots = 0;
        bool   numeric = idx < n;
        for (size_t k = idx; k < n && numeric; ++k) {
          if (buf[k] >= '0' && buf[k] <= '9') { ++digits; }
          else if (buf[k] == '.')             { ++dots; }
          else                                { numeric = false; }
        }
        if (!numeric || digits == 0 || dots > 1)
          continue;

        if (defaultHits == 0) {
          defaultValue = std::atof(buf);
          haveDefault  = true;
        }
        if (++defaultHits >= 2)
          break;
      }
      if (defaultHits != 1 || !haveDefault)
        return false;

      // (5) The value pair is the one place where m_fValue and m_nValue sit
      //     adjacent, agree with each other, AND agree with the default. The
      //     default-value cross-check is what makes this unambiguous -- a
      //     plain "float matches int" test also matches every run of zero
      //     padding in the object, which would resolve to whichever came
      //     first, i.e. a guess.
      //
      //     Consequence, deliberately: if the ConVar has already been changed
      //     away from its default before we first look, nothing matches and
      //     the lookup fails safe. A disabled feature is the correct outcome;
      //     writing through an unverified pointer is not.
      const float  wantF = static_cast<float>(defaultValue);
      const int    wantI = static_cast<int>(defaultValue);

      uintptr_t valuePair = 0;
      size_t    pairHits  = 0;
      for (size_t off = 0; off + 8 <= 0x80; off += 4) {
        const float f = *reinterpret_cast<const float*>(parent + off);
        const int   i = *reinterpret_cast<const int*>(parent + off + 4);
        if (!std::isfinite(f))
          continue;
        if (std::fabs(f - static_cast<float>(i)) > 1e-6f)
          continue;
        if (i != wantI || std::fabs(f - wantF) > 1e-6f)
          continue;
        if (pairHits == 0)
          valuePair = parent + off;
        if (++pairHits >= 2)
          break;
      }
      if (pairHits != 1)
        return false;

      out.object     = object;
      out.parent     = parent;
      out.floatValue = reinterpret_cast<float*>(valuePair);
      out.intValue   = reinterpret_cast<int*>(valuePair + 4);
      return true;
    }

    // ==================================================================
    // Registry
    // ==================================================================

    void invalidateCache() {
      std::lock_guard<std::mutex> lock(stateMutex());
      symbolCache().clear();
    }

    uintptr_t resolve(const SymbolDesc& desc) {
      if (desc.name == nullptr || desc.moduleName == nullptr)
        return 0;

      ModuleView mod;
      const bool haveModule = queryModule(desc.moduleName, mod);

      std::lock_guard<std::mutex> lock(stateMutex());
      SymbolCacheEntry& entry = symbolCache()[desc.name];

      // Module gone: forget everything about this symbol so a reload resolves
      // from scratch rather than handing back a pointer into freed pages.
      if (!haveModule) {
        entry = SymbolCacheEntry();
        entry.detail = "module not loaded";
        return 0;
      }

      if (entry.generation == mod.generation)
        return entry.address;           // hit, including a cached failure (0)

      const bool wasLogged = entry.logged && entry.generation != 0;
      entry = SymbolCacheEntry();
      entry.generation = mod.generation;

      auto fail = [&](const std::string& why) -> uintptr_t {
        entry.address  = 0;
        entry.resolved = false;
        entry.detail   = why;
        if (!wasLogged) {
          entry.logged = true;
          Logger::warn(str::format("[EngineSymbols] '", desc.name, "' unresolved in ",
                                   desc.moduleName, ": ", why,
                                   " -- dependent feature disabled"));
        } else {
          entry.logged = true;
        }
        return 0;
      };

      // String-anchored symbols do not use a byte signature at all.
      if (desc.kind == SymbolKind::StringAnchoredFunction
       || desc.kind == SymbolKind::StringEnclosingFunction) {
        if (desc.anchorString == nullptr)
          return fail("string-anchored symbol has no anchor string");

        uintptr_t strAddr = 0, site = 0;
        if (!findStringRef(mod, desc.anchorString, strAddr, site))
          return fail(str::format("anchor string \"", desc.anchorString,
                                  "\" absent or referenced from 2+ sites"));

        uintptr_t fn = 0;
        const bool ok = (desc.kind == SymbolKind::StringAnchoredFunction)
          ? findCodePointerNear(mod, site, desc.searchBefore, desc.searchAfter, fn)
          : findEnclosingFunction(mod, site,
                                  desc.searchBefore ? uint32_t(desc.searchBefore) : 0x1000u, fn);
        if (!ok)
          return fail(desc.kind == SymbolKind::StringAnchoredFunction
                        ? "no unique function pointer beside the anchor reference"
                        : "could not walk back to an enclosing function entry");

        entry.address  = fn;
        entry.resolved = true;
        entry.detail   = str::format("0x", std::hex, fn, " (+0x", fn - mod.base,
                                     ") via anchor", std::dec);
        Logger::info(str::format("[EngineSymbols] '", desc.name, "' -> ",
                                 desc.moduleName, "+0x", std::hex, fn - mod.base,
                                 std::dec, " (string-anchored, gen ", mod.generation, ")"));
        return fn;
      }

      // No signature means the symbol's identity has not been re-established
      // on this build. That is a legitimate, declared state -- not an error to
      // paper over by falling back to an old RVA.
      if (desc.pattern == nullptr)
        return fail("no signature registered for this build");

      const Pattern pat(desc.pattern);
      if (!pat.valid())
        return fail("malformed or insufficiently specific signature");

      uintptr_t first = 0;
      const size_t hits = pat.countMatches(mod.codeBegin, mod.codeEnd, &first, 2);
      if (hits == 0)
        return fail("no match in .text");
      if (hits > 1)
        return fail("ambiguous: 2+ matches in .text");

      uintptr_t address = first + static_cast<intptr_t>(desc.addend);

      switch (desc.kind) {
      case SymbolKind::Function:
        if (!mod.containsCode(address))
          return fail("resolved address outside .text");
        if (!looksLikeFunctionEntry(mod, address))
          return fail("resolved address is not a function entry");
        break;

      case SymbolKind::CodeSite:
        if (!mod.containsCode(address))
          return fail("resolved address outside .text");
        break;

      case SymbolKind::CallTarget: {
        uintptr_t target = 0;
        if (!decodeRel32Target(mod, address, target))
          return fail("no decodable rel32 branch at the matched site");
        if (!looksLikeFunctionEntry(mod, target))
          return fail("rel32 target is not a function entry");
        address = target;
        break;
      }

      case SymbolKind::RipRelativeData: {
        if (desc.instrLength == 0 || desc.dispOffset + 4u > desc.instrLength)
          return fail("bad rip-relative descriptor");
        uintptr_t target = 0;
        if (!decodeRipRelative(mod, address + desc.dispOffset,
                               address + desc.instrLength, target))
          return fail("rip-relative operand does not resolve into the image");
        address = target;
        break;
      }

      case SymbolKind::StringAnchoredFunction:
      case SymbolKind::StringEnclosingFunction:
        // Handled above, before the signature scan. Listed so the build's
        // /we4062 (unhandled enumerator) keeps this switch exhaustive.
        return fail("internal: string-anchored symbol reached the signature path");
      }

      entry.address  = address;
      entry.resolved = true;
      entry.detail   = str::format("0x", std::hex, address,
                                   " (+0x", address - mod.base, ")", std::dec);

      // Self-documenting: one line per newly resolved symbol, carrying the RVA
      // so a new build's resolutions can be read straight out of the log.
      Logger::info(str::format("[EngineSymbols] '", desc.name, "' -> ",
                               desc.moduleName, "+0x", std::hex, address - mod.base,
                               std::dec, " (gen ", mod.generation, ")"));
      return address;
    }

    // ==================================================================
    // Report
    // ==================================================================

    void dumpReport() {
      static const char* kModules[] = {
        "client.dll", "engine.dll", "studiorender.dll", "materialsystem_dx11.dll"
      };

      Logger::info("[EngineSymbols] ---- resolver report ----");

      for (const char* name : kModules) {
        ModuleView m;
        if (!queryModule(name, m)) {
          Logger::info(str::format("[EngineSymbols]   ", name, ": not loaded"));
          continue;
        }
        const Fingerprint fp = fingerprintOf(m);
        {
          std::lock_guard<std::mutex> lock(stateMutex());
          seenFingerprints()[name] = fp;
        }
        Logger::info(str::format(
          "[EngineSymbols]   ", name, ": base=0x", std::hex, m.base,
          " text=[0x", m.codeBegin - m.base, ",0x", m.codeEnd - m.base, ")",
          " rdata=[0x", m.rdataBegin - m.base, ",0x", m.rdataEnd - m.base, ")",
          " data=[0x", m.dataBegin - m.base, ",0x", m.dataEnd - m.base, ")",
          std::dec, " gen=", m.generation, " ", fp.toString()));
      }

      std::lock_guard<std::mutex> lock(stateMutex());
      for (const auto& kv : symbolCache()) {
        Logger::info(str::format("[EngineSymbols]   ", kv.first, ": ",
                                 kv.second.resolved ? "OK " : "-- ",
                                 kv.second.detail));
      }
      Logger::info("[EngineSymbols] ---- end report ----");
    }

  } // namespace EngineSymbols

  // ====================================================================
  // Engine eye
  // ====================================================================

  bool getEngineEyePosition(float outXyz[3]) {
    if (outXyz == nullptr)
      return false;

    if (!tf2::g_pilotEyeValid.load(std::memory_order_relaxed))
      return false;

    const float x = tf2::g_pilotEyeX.load(std::memory_order_relaxed);
    const float y = tf2::g_pilotEyeY.load(std::memory_order_relaxed);
    const float z = tf2::g_pilotEyeZ.load(std::memory_order_relaxed);

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
      return false;

    outXyz[0] = x;
    outXyz[1] = y;
    outXyz[2] = z;
    return true;
  }

} // namespace dxvk
