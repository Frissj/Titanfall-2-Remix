#include "dxvk_cs_cmd_probe.h"

#include <mutex>

namespace dxvk::csCmdProbe {

  std::atomic<bool>     g_enabled { false };
  Slot                  g_slots[kMaxSlots];
  std::atomic<uint32_t> g_slotCount { 0 };
  std::atomic<uint64_t> g_overflow  { 0 };

  uint32_t registerSlot(const void* vptr) {
    // Runs ONCE per command type, from a magic-static initialiser, so it is off
    // the hot path entirely and a plain mutex is the right tool. Contention is
    // bounded by the number of distinct EmitCs call sites in the binary.
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);

    const uint32_t id = g_slotCount.load(std::memory_order_relaxed);

    if (id >= kMaxSlots) {
      // Return an out-of-range id; Scope counts these into g_overflow so the
      // report can say the table was too small instead of quietly truncating.
      return kMaxSlots;
    }

    g_slots[id].vptr = vptr;
    g_slotCount.store(id + 1, std::memory_order_release);
    return id;
  }

  void resetCounters() {
    const uint32_t n = g_slotCount.load(std::memory_order_acquire);

    for (uint32_t i = 0; i < n; ++i) {
      g_slots[i].count.store(0, std::memory_order_relaxed);
      g_slots[i].samples.store(0, std::memory_order_relaxed);
      g_slots[i].ns.store(0, std::memory_order_relaxed);
    }

    g_overflow.store(0, std::memory_order_relaxed);
  }

}
