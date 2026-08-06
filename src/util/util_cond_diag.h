#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

// thread.h has a POSIX branch where dxvk::condition_variable is just
// std::condition_variable and there is no SleepConditionVariableSRW to
// instrument, so everything here is Win32-only and compiles away otherwise.
#ifdef _WIN32

#include "./com/com_include.h"

// NV-DXVK [Perf.CondWait]: WHICH condition variable is the frame thread
// blocked in?
//
// WHY (2026-08-06). The present thread spends ~70 ms of a ~121 ms frame
// blocked ([Perf.Busy] blockedMs=69.6 / wallMs=120.9), and the [GapSampler]
// histogram parks 56% of its samples in ntdll!ZwWaitForAlertByThreadId
// reached through ntdll!RtlSleepConditionVariableSRW. So the block is a
// CONDITION VARIABLE wait - but not which one, and that is the whole
// question, because the two answers need opposite fixes:
//
//   ours   -> a DXVK condvar (dxvk::condition_variable is the ONLY user of
//             SleepConditionVariableSRW inside d3d11.dll), and the captured
//             stack names the exact call site to attack.
//   theirs -> materialsystem_dx11.dll and tier0.dll ALSO import
//             SleepConditionVariableSRW (verified from their import tables),
//             so the engine can block on its own condvar with no DXVK
//             involvement at all, and nothing we change on this side helps.
//
// The sampler cannot answer that. Its APPcaller line is a heuristic - it
// takes the first validated return address in a 64-slot window above RSP -
// and for this wait it reports d3d11.dll+0x430250, which the PDB resolves to
// DxvkCsThread::dispatchChunk+0xd0, i.e. dxvk_cs.cpp:137, the instruction
// after WakeConditionVariable. A frame that called Wake cannot be the frame
// blocked in Sleep, so that hit is a DEAD slot left by an earlier, deeper
// dispatchChunk call. It passes the return-address check (it genuinely is
// one) which is exactly why the check did not catch it. This instrument
// exists so the question stops depending on that heuristic: it measures the
// wait from inside the wait, so it cannot be attributed to the wrong frame.
//
// Cost: two steady_clock reads around a call that is, by construction,
// blocking. There is no measurement floor to worry about here (contrast the
// per-draw FillMat buckets, where the work was sub-microsecond and the timer
// dominated) - the events being timed are microseconds at minimum and the
// ones we care about are milliseconds.
//
// SAFETY, load-bearing: note() must NEVER log, ALLOCATE, or take any lock.
// SleepConditionVariableSRW re-acquires the caller's SRW lock before
// returning, so everything here runs WITH THAT LOCK HELD. Taking the logger
// mutex, or the CRT heap lock, inverts lock order against every thread that
// holds it while waiting on a DXVK lock.
//
// THIS WAS VIOLATED AND IT DEADLOCKED THE GAME (2026-08-06). The first
// version allocated its per-thread block with `new` on each thread's first
// wait - i.e. took the heap lock under a DXVK SRW lock. It hung on the first
// raytraced frame, which is exactly when many threads (RT pipeline creation,
// shader compile pool, texture upload) reach their first condvar wait at
// once. The per-thread blocks now come from a fixed static pool claimed with
// one atomic increment: no allocation on any path.
//
// RtlCaptureStackBackTrace is still called under that lock. It walks our own
// stack, but RtlLookupFunctionEntry can touch the dynamic-function-table lock,
// so it is not provably lock-free either - which is the second reason this
// whole instrument is now OPT-IN rather than on by default.
//
// DEFAULT OFF. It has already answered the question it was built for: the
// frame thread blocks 0.0009-0.13 ms/frame in DXVK condvars across three
// separate builds, so the frame-thread stall is NOT ours. Set RTX_COND_DIAG=1
// to turn it back on when there is a new question for it.
namespace dxvk {

  namespace cond_diag {

    constexpr uint32_t kMaxFrames = 24;

    // Only waits at least this long are worth a stack. A condvar handoff of a
    // few microseconds is healthy queue behaviour; we are hunting the ~10 ms
    // class, and capturing every wake would put a stack walk on the CS
    // thread's idle path.
    constexpr uint64_t kCaptureNs = 1000000ull;   // 1 ms

    struct ThreadStat {
      uint32_t              tid        = 0;
      std::atomic<uint64_t> waitNs     { 0 };
      std::atomic<uint64_t> waitCount  { 0 };
      std::atomic<uint64_t> maxNs      { 0 };
      // One stack per thread, captured on the first wait over kCaptureNs.
      // frames[] is written BEFORE stackValid is released, and never again,
      // so a reader that acquires stackValid sees a complete capture.
      std::atomic<bool>     stackValid { false };
      uint16_t              nFrames    = 0;
      void*                 frames[kMaxFrames] = {};
    };

    // Fixed pool - see the allocation note above. Threads that wait after the
    // pool is exhausted simply stop being accounted; they are not tracked and
    // nothing is allocated to try.
    constexpr uint32_t kMaxTrackedThreads = 128;

    struct Snapshot {
      uint32_t tid;
      uint64_t waitNs;
      uint64_t waitCount;
      uint64_t maxNs;
      uint16_t nFrames;
      void*    frames[kMaxFrames];
    };

    // NAMESPACE-SCOPE inline variables, NOT function-local statics, and that
    // distinction is a safety property here rather than a style choice.
    //
    // A function-local static is initialised on first use behind MSVC's
    // thread-safe-static guard (_Init_thread_header), which TAKES A LOCK. On
    // this path first use happens inside condition_variable::wait, with the
    // caller's SRW lock held - the exact inversion that deadlocked the game
    // when this file allocated with `new`. These are constant-initialised at
    // load time instead, so reading them takes no guard and no lock.
    inline ThreadStat           g_pool[kMaxTrackedThreads];
    inline std::atomic<uint32_t> g_poolUsed { 0 };

    inline ThreadStat*            pool()     { return g_pool; }
    inline std::atomic<uint32_t>& poolUsed() { return g_poolUsed; }

    // DEFAULT OFF - see the header note. RTX_COND_DIAG=1 enables.
    //
    // Resolved ONCE during DLL static initialisation, before the game has
    // created the threads that wait. getenv takes a CRT lock, so calling it
    // lazily from wait() would reintroduce the same inversion. Until dynamic
    // init runs this reads its zero-initialised value, false, which is the
    // safe answer.
    inline bool readCondDiagEnv() {
      const char* v = std::getenv("RTX_COND_DIAG");
      return v != nullptr && v[0] != '\0' && std::strtol(v, nullptr, 10) != 0;
    }
    inline const bool g_condDiagEnabled = readCondDiagEnv();

    inline bool enabled() { return g_condDiagEnabled; }

    // Claims one pool slot per thread with a single atomic increment. No
    // allocation, no lock, no list to publish - the drain walks the pool
    // prefix directly. Returns nullptr once the pool is full, and note()
    // treats that as "stop accounting this thread".
    inline ThreadStat* self() {
      static thread_local ThreadStat* t_self  = nullptr;
      static thread_local bool        t_tried = false;
      if (t_self == nullptr && !t_tried) {
        t_tried = true;
        const uint32_t idx = poolUsed().fetch_add(1u, std::memory_order_acq_rel);
        if (idx < kMaxTrackedThreads) {
          t_self = &pool()[idx];
          t_self->tid = GetCurrentThreadId();
        }
      }
      return t_self;
    }

    inline void note(uint64_t ns) {
      ThreadStat* st = self();
      if (st == nullptr)
        return;                       // pool exhausted - never allocate
      st->waitNs.fetch_add(ns, std::memory_order_relaxed);
      st->waitCount.fetch_add(1u, std::memory_order_relaxed);

      uint64_t prevMax = st->maxNs.load(std::memory_order_relaxed);
      while (ns > prevMax
          && !st->maxNs.compare_exchange_weak(prevMax, ns,
               std::memory_order_relaxed, std::memory_order_relaxed)) {
        // prevMax reloaded by compare_exchange_weak
      }

      if (ns >= kCaptureNs && !st->stackValid.load(std::memory_order_acquire)) {
        // Skip 1 frame (this helper). Under inlining the skip count is
        // approximate, which is harmless - the frames still name the DXVK
        // call site, and that is all we need from them.
        st->nFrames = RtlCaptureStackBackTrace(1u, kMaxFrames, st->frames, nullptr);
        st->stackValid.store(true, std::memory_order_release);
      }
    }

    // Copies and RESETS the accumulators, so the reporter sees per-window
    // numbers directly comparable to [Perf.Busy]. The stack is copied but
    // NOT cleared: it is the identity of the site, not a per-window quantity.
    inline uint32_t drain(Snapshot* out, uint32_t maxOut) {
      const uint32_t used = std::min<uint32_t>(
        poolUsed().load(std::memory_order_acquire), kMaxTrackedThreads);
      uint32_t n = 0;
      for (uint32_t i = 0; i < used && n < maxOut; ++i) {
        ThreadStat* s = &pool()[i];
        Snapshot& o  = out[n];
        o.tid        = s->tid;
        o.waitNs     = s->waitNs.exchange(0u, std::memory_order_relaxed);
        o.waitCount  = s->waitCount.exchange(0u, std::memory_order_relaxed);
        o.maxNs      = s->maxNs.exchange(0u, std::memory_order_relaxed);
        o.nFrames    = 0;
        if (s->stackValid.load(std::memory_order_acquire)) {
          o.nFrames = s->nFrames;
          for (uint16_t k = 0; k < s->nFrames && k < kMaxFrames; ++k)
            o.frames[k] = s->frames[k];
        }
        ++n;
      }
      return n;
    }

    // RAII timer for a single blocking wait. Constructed before the sleep,
    // destroyed after it returns.
    class ScopedWait {
    public:
      ScopedWait()
      : m_on(enabled()) {
        if (m_on)
          m_t0 = std::chrono::steady_clock::now();
      }

      ~ScopedWait() {
        if (!m_on)
          return;
        const auto t1 = std::chrono::steady_clock::now();
        note(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
               t1 - m_t0).count()));
      }

      ScopedWait(const ScopedWait&) = delete;
      ScopedWait& operator = (const ScopedWait&) = delete;

    private:
      bool                                  m_on;
      std::chrono::steady_clock::time_point m_t0;
    };

  }

}

#endif  // _WIN32
