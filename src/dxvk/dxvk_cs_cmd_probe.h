#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "../util/util_likely.h"

// NV-DXVK [Perf.CsCmd]: per-command-type accounting on the dxvk-cs thread.
//
// WHY THIS EXISTS
// dxvk-cs is the frame. Measured 2026-08-06 on a clean 44-minute window (flat
// memory, no gap sampler, 73.6 ms frames): [Perf.CsSplit] execMs/frame = 73.3
// of a 73.6 ms frame, busyPct 99.7. Its composition, per frame:
//     >=10ms    22.4 ms    1 chunk      the RTX scene-prep + accel build
//     <1ms      21.0 ms    86 chunks    244 us each
//     <100us    28.4 ms    1037 chunks  27 us each
//     <10us      0.9 ms    245 chunks
// The fat chunk is understood -- [Perf.PrepScene] splits it (merge 11.0,
// accelLight 4.7, gc 2.2, surfMat 1.2). The other ~50 ms across ~1370 small
// chunks has NEVER been attributed, and it is the larger half.
//
// WHY THE PREVIOUS ANSWER DOES NOT COUNT
// HANDOFF_PERF_2026-08-06_v3 sec 3b concluded "the histogram is flat, there is
// no hot function, the driver is scale". That was measured with
// rtx.perfGapSampler aimed at dxvk-cs -- and that sampler was consuming ~16
// ms/frame OF THE THREAD IT WAS PROFILING: the <10ms bucket fell 16.2 -> 0.64
// ms/frame the moment it was switched off. A profiler that suspends its own
// target 500x/s and eats 22% of it will smear that overhead across every
// bucket and produce exactly the flat histogram that was reported. The verdict
// is unsafe and this probe exists to redo it honestly.
//
// WHAT IT MEASURES, AND WHY IT IS NOT A SAMPLING PROFILER
// Every DxvkCsCmd is a DxvkCsTypedCmd<T>/DxvkCsDataCmd<T,M> instantiation, so
// each distinct EmitCs lambda -- i.e. each call site -- is a distinct C++ type
// with its own vtable. The vtable pointer therefore identifies the call site
// exactly, at zero cost: it is already in cache, sitting in the object header
// the virtual dispatch just read.
//   count[type]  EXACT. One increment per command. No estimation.
//   mean[type]   sampled on 1-in-64 executions OF THAT TYPE.
//   total[type]  count * mean.
// Counts being exact is the whole point: a hot type cannot hide behind a low
// sampling probability the way it can in a RIP sampler, and a type that runs
// once per frame is still reported with its true count rather than being
// missed entirely.
//
// COST, WHICH IS THE REASON FOR THE 1-IN-64
// ~103,000 commands/frame. Timestamping every one would be 2 clock reads x 41
// ns = ~8.4 ms/frame -- 11% of the frame, i.e. the exact mistake the gap
// sampler made. At 1-in-64 it is ~1,600 timed commands/frame = ~0.13 ms/frame,
// under 0.2%. When disabled the per-command cost is one load of a plain bool
// and a predicted branch.
// TURN IT OFF once the 50 ms is attributed. It is cheap, not free.
//
// READING THE OUTPUT
// Slots are reported as d3d11+0xRVA of the command's vtable, newest-hottest
// first, and resolve against d3d11.pdb (which IS present at
// Titanfall2\bin\x64_retail). An unresolved RVA is still useful: it separates
// the types, so "three slots own 80% of the 50 ms" is actionable before any
// symbol lookup. estMsPerFrame is the column to sort by; count alone will
// mislead, because the <100us bucket is 1037 chunks of genuinely tiny work.

namespace dxvk::csCmdProbe {

  // Number of distinct command types tracked. Overflow is reported rather than
  // silently wrapping -- a truncated table would look like a complete one.
  constexpr uint32_t kMaxSlots = 512;

  // Time 1 execution in this many, PER TYPE. Power of two so the test is a mask.
  // (count & kSampleMask) == 1 also times the FIRST execution of every type, so
  // a type that runs once per frame still gets a mean instead of a zero.
  constexpr uint64_t kSampleMask = 63;

  struct Slot {
    const void*           vptr    = nullptr;  // identifies the type / call site
    std::atomic<uint64_t> count   { 0 };      // exact executions
    std::atomic<uint64_t> samples { 0 };      // how many were timed
    std::atomic<uint64_t> ns      { 0 };      // summed duration of those
  };

  // Relaxed atomic rather than a plain bool: it is written by the reporting
  // thread and read by dxvk-cs, which as a plain bool is a data race. Relaxed
  // is sufficient -- a one-window delay in observing a flip is irrelevant to
  // what this measures -- and on x86 a relaxed load is the same single mov a
  // plain bool would have compiled to, so correctness here costs nothing.
  extern std::atomic<bool>     g_enabled;
  extern Slot                  g_slots[kMaxSlots];
  extern std::atomic<uint32_t> g_slotCount;
  extern std::atomic<uint64_t> g_overflow;   // commands whose type found no slot

  uint32_t registerSlot(const void* vptr);

  // One slot per instantiated command type, resolved once. The magic-static
  // guard costs a single predicted load after the first call.
  template<typename T>
  inline uint32_t slotFor(const void* vptr) {
    static const uint32_t id = registerSlot(vptr);
    return id;
  }

  // CALLERS MUST TEST g_enabled BEFORE CONSTRUCTING THIS. The check is not
  // repeated here deliberately: doing it inside would mean the caller had
  // already paid for slotFor's magic-static guard, which is the cost this
  // design exists to avoid. See DxvkCsTypedCmd::exec.
  class Scope {
  public:
    // vptr is read from the object header by the caller -- already hot, since
    // the virtual dispatch that got us here just loaded it.
    inline Scope(uint32_t slot)
    : m_slot(slot) {
      if (unlikely(m_slot >= kMaxSlots)) {
        g_overflow.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      Slot& s = g_slots[m_slot];
      const uint64_t n = s.count.fetch_add(1, std::memory_order_relaxed) + 1;

      if (unlikely((n & kSampleMask) == 1)) {
        m_timed = true;
        m_start = std::chrono::steady_clock::now();
      }
    }

    inline ~Scope() {
      if (unlikely(m_timed)) {
        const auto dt = std::chrono::steady_clock::now() - m_start;
        Slot& s = g_slots[m_slot];
        s.samples.fetch_add(1, std::memory_order_relaxed);
        s.ns.fetch_add(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()),
          std::memory_order_relaxed);
      }
    }

    Scope             (const Scope&) = delete;
    Scope& operator = (const Scope&) = delete;

  private:
    uint32_t                              m_slot;
    bool                                  m_timed = false;
    std::chrono::steady_clock::time_point m_start;
  };

  // Zeroes counts/samples/ns but KEEPS the slot->vptr mapping, so a slot's
  // identity is stable across windows and successive reports are comparable.
  void resetCounters();

}
