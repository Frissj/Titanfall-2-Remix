#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
// <intrin.h> only - deliberately NOT com_include.h/windows.h. util_string.h
// includes this header, and util_string.h is included by nearly every TU in
// the project; dragging windows.h into all of them invites min/max and
// winsock-ordering breakage in files that have never seen it. The single
// Win32 symbol needed is GetCurrentThreadId, declared here with exactly the
// signature winbase.h uses, so a TU that also includes windows.h (in either
// order) sees a matching redeclaration rather than a conflict.
#include <intrin.h>
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
#endif

// NV-DXVK [Perf.FmtSite]: WHO is formatting floats, and on which thread?
//
// WHY (2026-08-06). [GapSampler], retargeted onto dxvk-cs, put
// std::_Floating_to_chars_general_precision<double> in the top 8 RVA buckets of
// the command-stream thread - ~1% of a thread that measures 96 ms of a 98.8 ms
// frame. Nothing in the render path should be converting doubles to text.
//
// The obvious suspicion was that rtx.logDenyTags suppresses the OUTPUT while
// the str::format call still runs - the exact lesson §5.3 of the CPU handoff
// paid for once already ("a bounded diagnostic is not a cheap diagnostic; bound
// the WORK, not just the output"). Reading call sites did NOT settle it: every
// candidate inspected in rtx_instance_manager.cpp / rtx_accel_manager.cpp /
// util_spatial_map.h turned out to be properly rate-limited (first N + every
// 4096th, one-shot per VS hash, top-15 caps). Guessing has been wrong three
// times in this session, so measure it instead - the same decision, and the
// same mechanism, that made [Perf.WcCopy] name SubmitInstancedDraw:10126 after
// three failed guesses.
//
// WHY POOLED PER THREAD, unlike [Perf.WcCopy]'s plain thread_local table.
// WcCopy only ever cared about the frame thread and is drained from it, so a
// thread_local table read by that same thread was sufficient. Here the thread
// of interest is dxvk-cs while the drain happens on the frame thread, so a
// thread_local table would report the WRONG THREAD'S sites - and would do it
// silently, looking exactly like a correct empty result. Blocks are claimed
// from a fixed pool with one atomic increment.
//
// SAFETY, load-bearing and inherited from util_cond_diag.h's deadlock (which
// cost one hang already): note() must never allocate, lock, or log. str::format
// is called from inside paths that hold DXVK locks and from the logger's own
// callers, so a heap allocation here could invert lock order against any thread
// holding the CRT heap lock. The pool is constant-initialised at load time (not
// a function-local static, which would take MSVC's thread-safe-static guard),
// blocks are claimed with a single fetch_add, and a thread that arrives after
// the pool is exhausted simply stops being accounted rather than allocating.
//
// LIMIT: _ReturnAddress() names whatever frame str::format was inlined into, so
// read a hit as "which region", not as proof of one line - identical to the
// caveat on wcNote. Over thousands of samples the real caller dominates.
namespace dxvk {

  namespace fmt_diag {

#ifdef _WIN32

    constexpr uint32_t kSites          = 64;
    constexpr uint32_t kProbe          = 4;
    constexpr uint32_t kMaxFmtThreads  = 96;

    struct FmtSite {
      const void* ret   = nullptr;
      uint64_t    calls = 0;
    };

    struct FmtThreadBlock {
      uint32_t tid = 0;
      uint64_t total = 0;
      FmtSite  sites[kSites] = {};
    };

    // Namespace-scope inline variables, constant-initialised at load time. See
    // the safety note above for why these are not function-local statics.
    inline FmtThreadBlock         g_fmtPool[kMaxFmtThreads];
    inline std::atomic<uint32_t>  g_fmtPoolUsed { 0 };

    // Default ON, RTX_FMT_DIAG=0 disables without a rebuild. Resolved during
    // dynamic init, never lazily: getenv takes a CRT lock and this runs on
    // threads holding DXVK locks. Until dynamic init runs this reads its
    // zero-initialised value (false), which is the safe answer.
    inline bool readFmtDiagEnv() {
      const char* v = std::getenv("RTX_FMT_DIAG");
      return (v == nullptr || v[0] == '\0') ? true : (std::strtol(v, nullptr, 10) != 0);
    }
    inline const bool g_fmtDiagEnabled = readFmtDiagEnv();

    inline FmtThreadBlock* fmtSelf() {
      static thread_local FmtThreadBlock* t_self = nullptr;
      static thread_local bool            t_tried = false;
      if (t_self == nullptr && !t_tried) {
        t_tried = true;
        const uint32_t idx = g_fmtPoolUsed.fetch_add(1u, std::memory_order_acq_rel);
        if (idx < kMaxFmtThreads) {
          t_self = &g_fmtPool[idx];
          t_self->tid = GetCurrentThreadId();
        }
      }
      return t_self;
    }

    // Same bucketing as wcNote: call sites cluster in a narrow RVA range and
    // hash to neighbouring slots, so a plain direct-mapped table would let two
    // real sites evict each other and both would under-report - silently
    // corrupting the exact measurement this exists to make. Probe 4, and when
    // all four are taken evict the SMALLEST, which keeps the heavy hitters
    // (the only ones being ranked) stable.
    // `weight` is the number of float/double arguments in the format call,
    // folded in at compile time by str::format. Counting them here rather than
    // calling once per float keeps one probe per CALL, which is what makes the
    // return address name the caller instead of util_string.h's own internals.
    inline void noteFloat(const void* ret, uint32_t weight = 1u) {
      if (!g_fmtDiagEnabled)
        return;
      FmtThreadBlock* blk = fmtSelf();
      if (blk == nullptr)
        return;                       // pool exhausted - never allocate
      blk->total += weight;

      const uintptr_t h    = reinterpret_cast<uintptr_t>(ret);
      const uint32_t  home = uint32_t((h ^ (h >> 7)) & (kSites - 1u));

      uint32_t victim  = home;
      uint64_t leastBy = ~0ull;

      for (uint32_t k = 0; k < kProbe; ++k) {
        FmtSite& c = blk->sites[(home + k) & (kSites - 1u)];
        if (c.ret == ret) { c.calls += weight; return; }
        if (c.ret == nullptr) { c.ret = ret; c.calls = weight; return; }
        if (c.calls < leastBy) { leastBy = c.calls; victim = (home + k) & (kSites - 1u); }
      }

      FmtSite& e = blk->sites[victim];
      e.ret   = ret;
      e.calls = weight;
    }

    struct FmtSnapshot {
      uint32_t    tid;
      uint64_t    total;
      const void* ret;
      uint64_t    calls;
    };

    // Copies and RESETS the counters so the reporter sees per-window numbers
    // directly comparable to the other [Perf.*] lines. Walks the whole pool, so
    // it reports every thread's sites regardless of which thread drains.
    inline uint32_t drain(FmtSnapshot* out, uint32_t maxOut) {
      const uint32_t used = g_fmtPoolUsed.load(std::memory_order_acquire);
      const uint32_t n    = used < kMaxFmtThreads ? used : kMaxFmtThreads;
      uint32_t w = 0;
      for (uint32_t i = 0; i < n; ++i) {
        FmtThreadBlock& b = g_fmtPool[i];
        const uint64_t bt = b.total;
        b.total = 0;
        for (uint32_t s = 0; s < kSites && w < maxOut; ++s) {
          if (b.sites[s].ret == nullptr || b.sites[s].calls == 0)
            continue;
          out[w].tid   = b.tid;
          out[w].total = bt;
          out[w].ret   = b.sites[s].ret;
          out[w].calls = b.sites[s].calls;
          b.sites[s].calls = 0;       // keep ret: the slot identity is stable
          ++w;
        }
      }
      return w;
    }

#else

    // Signature must match the Win32 version including the default, so a caller
    // added outside the _WIN32 guard fails to build on Windows too rather than
    // silently compiling on one platform only.
    inline void noteFloat(const void*, uint32_t = 1u) { }

#endif

  }

}
