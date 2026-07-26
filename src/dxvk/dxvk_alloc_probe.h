#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// NV-DXVK [perf]: allocation / lock-stall probe.
//
// WHY THIS EXISTS
// On 2026-07-26 the [Perf.PrepScene] and [Perf.Merge] sub-splits showed that
// exactly ONE bucket spiked per frame, and that it was a DIFFERENT bucket each
// frame — dynBlas 11.7x on one frame, loop 3.9x on the next, uploadSurfaceData
// 3.5x on the one after — while instance/surface counts stayed flat and the
// cpuSlowX probe in [Perf.Frame] confirmed the CPU was running at full speed.
//
// A hot loop inflates the SAME bucket every frame. A spike that migrates
// between unrelated stages is a stall: the stage that happens to be running
// when something blocks gets charged for it. Every one of those stages calls
// into DxvkContext (writeToBuffer / copyBuffer / buffer creation / BLAS
// scheduling), and [Perf.Block] already reports csSyncMs=0 and gpuSyncMs=0, so
// it is not the command-stream or GPU fences. That leaves the memory allocator
// and the buffer slice pools, which is what this measures.
//
// WHAT IT MEASURES  (five distinct mechanisms, so the winner is unambiguous)
//   memAlloc    DxvkMemoryAllocator::alloc, whole call. The umbrella number.
//   typeLock    Time lost waiting on DxvkMemoryType::mutex, and ONLY when
//               actually contended — the fast path uses try_lock and records
//               nothing, so an uncontended acquire costs no clock reads.
//   vkAlloc     vkAllocateMemory itself. A real driver call that can take
//               milliseconds; if this dominates, the fix is chunk sizing or
//               pooling, not anything in the renderer.
//   freeChunks  freeEmptyChunks. Called from inside tryAllocFromType with the
//               type mutex DROPPED and retaken, so it both costs time and
//               widens the window for other threads to contend.
//   slice       DxvkBuffer::allocSlice, but only its free-list-exhausted
//               branch, where it has to build a whole new backing buffer.
//               sliceRefills counts how often that happens.
//
// COST
// Every timed site is either rare (a real allocation) or already known to be
// contended. The hot path of allocSlice — which runs per draw for constant
// buffers — takes NO clock reads at all: only the refill branch is timed. So
// this can be left on during a timing run without distorting what it measures,
// which is the whole point of separating it from rtx.logSurfaceCoverage.
//
// THREADING
// These sites are hit from the CS thread, the submit thread and worker threads
// at once, so every counter is atomic. Relaxed ordering throughout: the values
// are diagnostic aggregates, nothing branches on them, and relaxed atomics add
// roughly a nanosecond on x86.

namespace dxvk::allocProbe {

  /// One measured mechanism: total time, worst single occurrence, and count.
  /// maxNs matters more than the average here — a 15 ms frame spike is one bad
  /// acquire, and an average over hundreds of fast calls would hide it.
  struct Bucket {
    std::atomic<uint64_t> totalNs { 0 };
    std::atomic<uint64_t> maxNs   { 0 };
    std::atomic<uint32_t> count   { 0 };

    void add(uint64_t ns) {
      totalNs.fetch_add(ns, std::memory_order_relaxed);
      count.fetch_add(1u, std::memory_order_relaxed);
      // CAS loop rather than a plain store: several threads can be reporting a
      // new maximum at once and a naive store would lose the larger value.
      uint64_t prev = maxNs.load(std::memory_order_relaxed);
      while (ns > prev
          && !maxNs.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {
        // prev is refreshed by compare_exchange_weak on failure.
      }
    }

    void reset() {
      totalNs.store(0, std::memory_order_relaxed);
      maxNs.store(0, std::memory_order_relaxed);
      count.store(0u, std::memory_order_relaxed);
    }
  };

  // C++17 inline variables: one definition across all translation units with no
  // .cpp file, so adding this probe needs no meson.build change.
  inline std::atomic<bool> g_enabled { false };

  inline Bucket g_memAlloc;
  inline Bucket g_typeLock;
  inline Bucket g_vkAlloc;
  inline Bucket g_freeChunks;
  inline Bucket g_slice;
  inline std::atomic<uint32_t> g_sliceRefills { 0 };

  inline void resetAll() {
    g_memAlloc.reset();
    g_typeLock.reset();
    g_vkAlloc.reset();
    g_freeChunks.reset();
    g_slice.reset();
    g_sliceRefills.store(0u, std::memory_order_relaxed);
  }

  /// RAII timer that reads the clock ONLY when the probe is enabled, so a
  /// disabled probe costs one relaxed atomic load per site and nothing else.
  class Timer {

  public:

    explicit Timer(Bucket& bucket) {
      if (g_enabled.load(std::memory_order_relaxed)) {
        m_bucket = &bucket;
        m_start = std::chrono::steady_clock::now();
      }
    }

    ~Timer() {
      if (m_bucket != nullptr) {
        const auto elapsed = std::chrono::steady_clock::now() - m_start;
        m_bucket->add(uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
      }
    }

    Timer             (const Timer&) = delete;
    Timer& operator = (const Timer&) = delete;

  private:

    Bucket*                               m_bucket = nullptr;
    std::chrono::steady_clock::time_point m_start;

  };

}
