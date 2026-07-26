#pragma once

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdint>

#include "../../util/log/log.h"
#include "../../util/util_string.h"

// NV-DXVK [perf]: "was this thread BLOCKED, or was it BUSY?"
//
// WHY THIS EXISTS
// [Perf.PrepScene] / [Perf.Merge] / [Perf.BuildBlas] showed exactly one bucket
// spiking per frame, a DIFFERENT one each frame (upload 4.8x, then loop 3.9x,
// then loop 6.2x) at flat instance counts. Two explanations were already ruled
// out by measurement, not argument:
//   - the CPU downclocking      -> cpuSlowX in [Perf.Frame] pins at 1.00
//   - the Vulkan allocator      -> [Perf.Alloc] reads 36-42 us against a 16 ms
//                                  spike, with typeLock/vkAlloc/refills all 0
// A migrating spike is the signature of a stall being charged to whichever
// stage happened to be running. But wall-clock timers CANNOT distinguish "this
// code executed slowly" from "this thread was descheduled for 15 ms" — and
// those have completely different fixes. That is the gap this closes.
//
// HOW
// QueryThreadCycleTime counts only cycles the calling thread actually RETIRED,
// so time spent blocked or preempted is invisible to it. Pair it with a
// wall-clock reading over the same span:
//
//   cpuUs     = cycles / cyclesPerUs   -> time the thread genuinely executed
//   blockedUs = wallUs - cpuUs         -> time it was not on a core at all
//
// A spike with blockedUs ~= the spike is the thread waiting on something. A
// spike with cpuUs ~= wallUs is real work, and then the page-fault delta says
// whether that work is memory-stalled (first-touch on freshly grown vectors,
// working-set trimming) rather than compute.
//
// This mirrors the [SdStall] methodology already used for SubmitDraw stages in
// d3d11_rtx.cpp; its getCalibratedCyclesPerUs() is file-static there, so the
// calibration is duplicated here rather than shared. Keep the two consistent
// if either is changed.
//
// COST
// Two QueryThreadCycleTime calls (~0.44 us each) plus one page-fault query per
// prepareSceneData, i.e. ~1 us/frame. That per-call cost is why QTCT was
// previously refuted for PER-DRAW buckets; at two calls per frame it is noise.
// Calibration is one-time and costs ~8 ms at the first sample.

namespace dxvk::cpuStall {

  /// Measured cycles-per-microsecond for this machine, calibrated once.
  /// QueryThreadCycleTime only counts executed cycles, so preemption during a
  /// calibration spin can only LOWER the observed ratio — taking the max over
  /// eight spins converges on the true sustained boost rate.
  inline int64_t cyclesPerUs() {
    static const int64_t s_rate = []() -> int64_t {
      int64_t best = 0;
      for (int i = 0; i < 8; ++i) {
        ULONG64 c0 = 0, c1 = 0;
        QueryThreadCycleTime(GetCurrentThread(), &c0);
        const auto w0 = std::chrono::steady_clock::now();
        volatile uint64_t sink = 0;
        while (std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now() - w0).count() < 1000) {
          sink += 1;
        }
        QueryThreadCycleTime(GetCurrentThread(), &c1);
        const int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - w0).count();
        if (us > 0) {
          const int64_t rate = int64_t(c1 - c0) / us;
          if (rate > best) {
            best = rate;
          }
        }
      }
      if (best <= 0) {
        best = 5000;  // calibration failed; matches the d3d11_rtx.cpp fallback
      }
      Logger::info(str::format(
        "[Perf.CpuCalib] (cpuStall) cyclesPerUs=", best,
        " (~", best / 1000, ".", (best % 1000) / 100, " GHz sustained)"));
      return best;
    }();
    return s_rate;
  }

  /// Process-wide cumulative page-fault count, or 0 if unavailable.
  /// Uses K32GetProcessMemoryInfo, which kernel32 exports directly (Win7+), so
  /// this needs no psapi link and therefore no meson.build change. The struct is
  /// declared locally to the documented PROCESS_MEMORY_COUNTERS layout to avoid
  /// pulling psapi.h into this header.
  inline uint64_t pageFaultCount(uint64_t* workingSetBytes = nullptr) {
    struct PmcLayout {
      DWORD  cb;
      DWORD  PageFaultCount;
      SIZE_T PeakWorkingSetSize;
      SIZE_T WorkingSetSize;
      SIZE_T QuotaPeakPagedPoolUsage;
      SIZE_T QuotaPagedPoolUsage;
      SIZE_T QuotaPeakNonPagedPoolUsage;
      SIZE_T QuotaNonPagedPoolUsage;
      SIZE_T PagefileUsage;
      SIZE_T PeakPagefileUsage;
    };
    using PfnGetMemInfo = BOOL (WINAPI*)(HANDLE, PmcLayout*, DWORD);

    static const PfnGetMemInfo s_fn = []() -> PfnGetMemInfo {
      HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
      return k32 != nullptr
        ? reinterpret_cast<PfnGetMemInfo>(GetProcAddress(k32, "K32GetProcessMemoryInfo"))
        : nullptr;
    }();

    if (s_fn == nullptr) {
      return 0;
    }

    PmcLayout pmc {};
    pmc.cb = sizeof(pmc);
    if (!s_fn(GetCurrentProcess(), &pmc, sizeof(pmc))) {
      return 0;
    }
    if (workingSetBytes != nullptr) {
      *workingSetBytes = uint64_t(pmc.WorkingSetSize);
    }
    return uint64_t(pmc.PageFaultCount);
  }

  /// One end of a measured span: retired cycles + wall clock + fault count.
  struct Sample {
    uint64_t                              cycles     = 0;
    std::chrono::steady_clock::time_point wall       {};
    uint64_t                              faults     = 0;
    uint64_t                              workingSet = 0;
  };

  inline Sample take() {
    Sample s;
    ULONG64 cyc = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cyc);
    s.cycles = uint64_t(cyc);
    s.wall   = std::chrono::steady_clock::now();
    s.faults = pageFaultCount(&s.workingSet);
    return s;
  }

  /// Result of differencing two Samples.
  struct Delta {
    int64_t  wallUs     = 0;
    int64_t  cpuUs      = 0;   ///< time the thread actually executed
    int64_t  blockedUs  = 0;   ///< wallUs - cpuUs, clamped at 0
    int64_t  busyPct    = 0;   ///< 100 * cpuUs / wallUs
    uint64_t faults     = 0;   ///< page faults over the span
    uint64_t workingSet = 0;   ///< working set at the end, bytes
  };

  inline Delta diff(const Sample& a, const Sample& b) {
    Delta d;
    d.wallUs = std::chrono::duration_cast<std::chrono::microseconds>(b.wall - a.wall).count();
    const int64_t rate = cyclesPerUs();
    d.cpuUs = rate > 0 ? int64_t(b.cycles - a.cycles) / rate : 0;
    // Clamped: cpuUs can marginally exceed wallUs from rate calibration error
    // or a clock/counter mismatch, and a negative "blocked" reads as nonsense.
    d.blockedUs = d.wallUs > d.cpuUs ? (d.wallUs - d.cpuUs) : 0;
    d.busyPct = d.wallUs > 0 ? (100 * d.cpuUs) / d.wallUs : 0;
    d.faults = b.faults >= a.faults ? (b.faults - a.faults) : 0;
    d.workingSet = b.workingSet;
    return d;
  }

}
