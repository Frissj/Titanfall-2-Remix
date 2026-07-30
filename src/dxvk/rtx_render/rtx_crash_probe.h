/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* NV-DXVK [CrashProbe] — name the faulting site instead of guessing at it.
*
* Two crashes were recorded with NO diagnostic of any kind, both exception
* code 0xC0000409:
*   03:28:24  ucrtbase.dll  +0xa527e
*   01:28:15  tier0.dll     +0x26170   (the game's own module)
*
* 0xC0000409 is STATUS_STACK_BUFFER_OVERRUN, which Windows also uses for
* __fastfail(). That matters: __fastfail deliberately bypasses SEH and vectored
* handlers, so a VEH alone would catch nothing. The reachable paths that end in
* a CRT fast-fail are:
*
*   - an uncaught C++ exception  -> std::terminate -> abort
*   - abort() called directly    -> SIGABRT
*   - a CRT function given an invalid argument -> invalid parameter handler
*
* All three are hookable BEFORE the fast-fail, which is the whole point. A VEH
* is also installed for ordinary access violations (0xC0000005), first-chance,
* logged once, and always passing the exception on — it never changes control
* flow.
*
* Output is module+offset, resolvable against d3d11.pdb (which lives in
* Titanfall2\bin\x64_retail) via dbghelp. Symbolising in-process is avoided on
* purpose: SymFromAddr takes a loader-adjacent lock, and doing that from a
* crashing thread is a good way to hang instead of reporting.
*
* Costs nothing when nothing crashes: installation is one-shot, and no hook
* runs on the normal path.
*/
#pragma once

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include <windows.h>

#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include "../../util/util_error.h"

namespace dxvk::crashprobe {

  // Module+offset for each frame. Deliberately not symbolised in-process.
  inline std::string captureBacktrace(uint32_t skip = 2) {
    static constexpr DWORD kMaxFrames = 48;
    void* frames[kMaxFrames] = {};
    const USHORT n = RtlCaptureStackBackTrace(
      static_cast<DWORD>(skip), kMaxFrames, frames, nullptr);

    std::string out;
    for (USHORT i = 0; i < n; ++i) {
      HMODULE mod = nullptr;
      char name[MAX_PATH] = {};
      const char* shortName = "?";
      uintptr_t offset = reinterpret_cast<uintptr_t>(frames[i]);

      if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(frames[i]), &mod) && mod != nullptr) {
        if (GetModuleFileNameA(mod, name, sizeof(name) - 1) != 0) {
          const char* slash = strrchr(name, '\\');
          shortName = (slash != nullptr) ? slash + 1 : name;
        }
        offset -= reinterpret_cast<uintptr_t>(mod);
      }

      char line[MAX_PATH + 64] = {};
      std::snprintf(line, sizeof(line), "\n    #%02u %s+0x%llx",
                    static_cast<unsigned>(i), shortName,
                    static_cast<unsigned long long>(offset));
      out += line;
    }
    return out;
  }

  inline std::atomic<bool>& reported() {
    static std::atomic<bool> r { false };
    return r;
  }

  // Log at most one crash report. A second thread faulting while the first is
  // still formatting would otherwise interleave two stacks into an unreadable
  // mess, and the first one is the one that matters.
  inline void report(const char* what, const std::string& extra = std::string()) {
    bool expected = false;
    if (!reported().compare_exchange_strong(expected, true)) {
      return;
    }
    Logger::err(str::format(
      "[CrashProbe] ", what, extra,
      " tid=", static_cast<uint32_t>(GetCurrentThreadId()),
      captureBacktrace()));
    Logger::flush();
  }

  inline void onTerminate() {
    std::string detail;
    // If a C++ exception is in flight, its type/message is the single most
    // useful fact available and is lost the moment abort() runs.
    if (auto ex = std::current_exception()) {
      try {
        std::rethrow_exception(ex);
      } catch (const std::exception& e) {
        detail = std::string(" uncaught std::exception what=\"") + e.what() + "\"";
      } catch (const DxvkError& e) {
        detail = std::string(" uncaught DxvkError message=\"") + e.message() + "\"";
      } catch (...) {
        detail = " uncaught non-standard exception";
      }
    } else {
      detail = " terminate with no active exception";
    }
    report("std::terminate", detail);
    // Fall through to the previous handler / abort.
  }

  inline void onAbort(int) {
    report("SIGABRT (abort)");
  }

  inline void onInvalidParameter(const wchar_t*, const wchar_t*,
                                 const wchar_t*, unsigned int, uintptr_t) {
    // The CRT's default here is __fastfail, i.e. instant death with no
    // unwinding — so this hook is the only chance to say anything at all.
    report("CRT invalid parameter");
  }

  inline LONG CALLBACK onVectoredException(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    // Access violations only. First-chance handlers see a lot of ordinary
    // exceptions (C++ throws, debugger notifications) that are handled
    // perfectly well downstream; logging those would bury the real fault.
    if (code == EXCEPTION_ACCESS_VIOLATION) {
      const auto* rec = info->ExceptionRecord;
      const char* op = rec->NumberParameters >= 2
        ? (rec->ExceptionInformation[0] == 1 ? "write" : "read") : "?";
      report("ACCESS_VIOLATION", str::format(
        " ", op, " addr=0x", std::hex,
        static_cast<uint64_t>(rec->NumberParameters >= 2 ? rec->ExceptionInformation[1] : 0),
        " at=0x", reinterpret_cast<uintptr_t>(rec->ExceptionAddress), std::dec));
    }
    // Never swallow. This probe reports; it must not alter behaviour.
    return EXCEPTION_CONTINUE_SEARCH;
  }

  inline void ensureInstalled() {
    static std::atomic<bool> installed { false };
    bool expected = false;
    if (!installed.compare_exchange_strong(expected, true)) {
      return;
    }
    std::set_terminate(&onTerminate);
    std::signal(SIGABRT, &onAbort);
    _set_invalid_parameter_handler(&onInvalidParameter);
    AddVectoredExceptionHandler(1 /*first*/, &onVectoredException);
    Logger::warn("[CrashProbe] installed (terminate / SIGABRT / CRT-invalid-param / AV)");
  }

}
