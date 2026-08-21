#include "rtx_perf_report.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// NV-DXVK [Perf.SessionState]: GetCurrentProcessorNumber / GetThreadPriority /
// Get{Process,Thread}Information for the session-state probe. Kept out of the
// header on purpose -- windows.h stays local to this one TU.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dxvk {

  namespace perfreport {

    static std::array<Cell, size_t(Slot::kCount)> s_cells;

    // Bumped once per frame from the frame thread; read from every publish site
    // on every thread. Relaxed is enough -- a publisher racing the increment
    // stamps a value one frame early, which changes an age by 1.
    static std::atomic<uint32_t> s_frameCounter { 0u };

    uint32_t currentFrame() {
      return s_frameCounter.load(std::memory_order_relaxed);
    }

    void publish(const Slot slot, const double valuePerFrame, const uint32_t frameId) {
      Cell& c = s_cells[size_t(slot)];
      c.value.store(valuePerFrame, std::memory_order_relaxed);
      c.frame.store(frameId, std::memory_order_relaxed);
      c.seq.fetch_add(1u, std::memory_order_relaxed);
    }

    void publishWindow(const Slot slot, const double windowTotal, const uint64_t frames,
                       const uint32_t frameId) {
      if (frames == 0ull) {
        // A window with no frames in it would divide by zero and, worse, would
        // overwrite a good value with garbage. Warmup and the first window after
        // a device reset both hit this.
        return;
      }
      publish(slot, windowTotal / double(frames), frameId);
    }

    // [Perf.SessionState] -- contract and motivation at the declaration.
    // Field guide for reading the line:
    //   spinNs  wall ns for 50k iterations of a dependent mul/xor/shift chain.
    //           No memory traffic, not vectorisable -- inflates only when the
    //           core itself runs slower (EcoQoS, E-core, clock throttle).
    //   xMin    spinNs / this thread's session-best spinNs. ~1.0x healthy.
    //           The degraded sessions should read ~3x here on both threads.
    //   clkNs   ns per steady_clock::now(). ~15-30 on rdtsc-backed QPC;
    //           hundreds+ means an HPET fallback taxed every instrument.
    //   core    processor number at probe time -- a sample, not a census, but
    //           repeated E-core numbers across windows name a placement issue.
    //   ecoP/T  EXPLICIT process/thread EcoQoS flags. Windows can impose QoS
    //           without setting these, so 0 does not acquit QoS -- spinNs is
    //           the ground truth; 1 here names the mechanism. -1 = no API.
    void sessionStateProbe(const char* who) {
#ifdef _WIN32
      volatile uint64_t sink = 0x9E3779B97F4A7C15ull;
      uint64_t v = sink;
      const auto t0 = std::chrono::steady_clock::now();
      for (uint32_t i = 0; i < 50000u; ++i) {
        v = v * 6364136223846793005ull + 1442695040888963407ull;
        v ^= v >> 29;
      }
      const auto t1 = std::chrono::steady_clock::now();
      sink = v;
      (void) sink;
      const uint64_t spinNs = (uint64_t) std::chrono::duration_cast<
        std::chrono::nanoseconds>(t1 - t0).count();

      auto cp = t1;
      for (uint32_t i = 0; i < 1000u; ++i)
        cp = std::chrono::steady_clock::now();
      const uint64_t clkNs = (uint64_t) std::chrono::duration_cast<
        std::chrono::nanoseconds>(cp - t1).count() / 1000u;

      static thread_local uint64_t s_spinMin = UINT64_MAX;
      if (spinNs < s_spinMin)
        s_spinMin = spinNs;
      const double xMin = (s_spinMin != 0ull && s_spinMin != UINT64_MAX)
        ? double(spinNs) / double(s_spinMin) : 0.0;

      int ecoP = -1;
      int ecoT = -1;
#ifdef PROCESS_POWER_THROTTLING_CURRENT_VERSION
      {
        PROCESS_POWER_THROTTLING_STATE ps = {};
        ps.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        if (GetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                  &ps, sizeof(ps))) {
          ecoP = ((ps.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) != 0
               && (ps.StateMask   & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) != 0) ? 1 : 0;
        }
      }
#endif
#ifdef THREAD_POWER_THROTTLING_CURRENT_VERSION
      {
        THREAD_POWER_THROTTLING_STATE ts = {};
        ts.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        if (GetThreadInformation(GetCurrentThread(), ThreadPowerThrottling,
                                 &ts, sizeof(ts))) {
          ecoT = ((ts.ControlMask & THREAD_POWER_THROTTLING_EXECUTION_SPEED) != 0
               && (ts.StateMask   & THREAD_POWER_THROTTLING_EXECUTION_SPEED) != 0) ? 1 : 0;
        }
      }
#endif
      char xbuf[32];
      std::snprintf(xbuf, sizeof(xbuf), "%.2f", xMin);
      Logger::warn(str::format(
        "[Perf.SessionState] who=", who,
        " spinNs=", spinNs,
        " xMin=", xbuf,
        " clkNs=", clkNs,
        " core=", (uint32_t) GetCurrentProcessorNumber(),
        " thrPrio=", GetThreadPriority(GetCurrentThread()),
        " prioClass=", (uint32_t) GetPriorityClass(GetCurrentProcess()),
        " ecoP=", ecoP,
        " ecoT=", ecoT));
#else
      (void) who;
#endif
    }

    namespace {

      // Fixed-decimal formatter. str::format prints doubles at full precision,
      // which turns a 24-row table into unreadable noise; this keeps columns
      // aligned. snprintf rather than ostream so it cannot pick up a locale that
      // swaps the decimal separator and breaks log parsing.
      std::string fmtF(const double v, const int decimals = 2) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
        return std::string(buf);
      }

      struct Read {
        double   v    = 0.0;
        uint32_t age  = 0u;      // frames since published
        bool     seen = false;
      };

      Read read(const Slot slot, const uint32_t now) {
        const Cell& c = s_cells[size_t(slot)];
        Read r;
        r.seen = c.seq.load(std::memory_order_relaxed) != 0u;
        if (!r.seen) {
          return r;
        }
        r.v = c.value.load(std::memory_order_relaxed);
        const uint32_t f = c.frame.load(std::memory_order_relaxed);
        r.age = (now >= f) ? (now - f) : 0u;
        return r;
      }

      double val(const Slot slot, const uint32_t now) {
        return read(slot, now).v;
      }

      // "12.34" with two decimals, or "  n/a" when the source never published.
      // A zero and a never-measured are different facts and the report must not
      // conflate them -- that is how a gated-off probe gets read as "free".
      std::string num(const Slot slot, const uint32_t now, const int decimals = 2) {
        const Read r = read(slot, now);
        if (!r.seen) {
          return "n/a";
        }
        return fmtF(r.v, decimals);
      }

      std::string pctOf(const Slot slot, const uint32_t now, const double whole) {
        const Read r = read(slot, now);
        if (!r.seen || whole <= 0.0) {
          return "-";
        }
        return fmtF(100.0 * r.v / whole, 0) + "%";
      }

      // One row of the breakdown.
      //
      //   kTotal   a bucket that stands on its own
      //   kNested  already inside its parent's number; never add it into a total
      //            (PERF_INSTRUMENTATION_MAP section 2)
      //   kAltView the SAME time as the row above, measured by a DIFFERENT
      //            instrument. Not a sibling and not a component -- adding it to
      //            its neighbour double-counts. [Perf.Boundary]'s post-Present
      //            number and [Perf.Gap]'s afterQueryEnd are one gap seen twice,
      //            and printing them as two children of "between entry points"
      //            let a reader sum 11.4 + 11.4 into a 16.7 ms parent.
      enum class RowKind { kTotal, kNested, kAltView };

      struct Row {
        int         depth;
        const char* label;
        Slot        slot;
        RowKind     kind;
      };

      void emitRows(const Row* rows, const size_t count, const uint32_t now,
                    const double whole) {
        for (size_t i = 0; i < count; ++i) {
          const Row& r = rows[i];
          std::string indent;
          for (int d = 0; d < r.depth; ++d) {
            indent += "  ";
          }
          Logger::warn(str::format(
            "[Perf.Report]   ", indent, r.label,
            std::string(std::max<int>(1, 34 - int(indent.size()) - int(std::string(r.label).size())), ' '),
            num(r.slot, now), " ms  ", pctOf(r.slot, now, whole),
            r.kind == RowKind::kNested  ? "   (nested)" :
            r.kind == RowKind::kAltView ? "   (ALT VIEW of the row above -- same time, "
                                          "different instrument; do NOT add)" : ""));
        }
      }

      // A closure check between two instruments that measure the same span by
      // different routes. Two routes agreeing is what makes a number usable; a
      // single route agreeing with itself is not evidence of anything.
      void check(const char* what, const double a, const double b,
                 const double tolPct, const bool aSeen, const bool bSeen) {
        if (!aSeen || !bSeen) {
          Logger::warn(str::format("[Perf.Report]   SKIP  ", what,
                                   "  (source not published -- probe gated off?)"));
          return;
        }
        const double denom = std::max(std::fabs(a), std::fabs(b));
        const double diff  = (denom > 1e-9) ? (100.0 * std::fabs(a - b) / denom) : 0.0;
        Logger::warn(str::format(
          "[Perf.Report]   ", (diff <= tolPct ? "PASS  " : "FAIL  "), what,
          "  ", fmtF(a, 2), " vs ", fmtF(b, 2),
          "  (", fmtF(diff, 1), "%, tol ", fmtF(tolPct, 0), "%)"));
      }

    }

    void emitReport(const uint32_t now) {
      const double wall   = val(Slot::FrameWallMs, now);
      const double ftCpu  = val(Slot::FrameThreadCpuMs, now);
      const double csExec = val(Slot::CsExecMs, now);
      const double gpuBsy = val(Slot::GpuPassTotalMs, now);
      const double gpuIdl = val(Slot::GpuIdleMs, now);

      if (wall <= 0.0) {
        Logger::warn("[Perf.Report] no frame wall published yet -- "
                     "is rtx.perfThreadCensus / the [Perf.Busy] block enabled?");
        return;
      }

      const double fps = (wall > 0.0) ? (1000.0 / wall) : 0.0;

      Logger::warn("[Perf.Report] ================================================================");
      Logger::warn(str::format(
        "[Perf.Report] FRAME ", fmtF(wall, 2), " ms  (",
        fmtF(fps, 1), " fps)   frame=", now));

      // ---- hygiene first. Every number below is void without it (map section 6).
      {
        const Read matNew = read(Slot::HygMatNew, now);
        const bool dirty  = matNew.seen && matNew.v > 0.5;
        Logger::warn(str::format(
          "[Perf.Report] HYGIENE  matNew=", num(Slot::HygMatNew, now, 0),
          " matTotal=", num(Slot::HygMatTotal, now, 0),
          " texTotal=", num(Slot::HygTexTotal, now, 0),
          " inst=", num(Slot::HygInstances, now, 0),
          " uniqueBlas=", num(Slot::HygUniqueBlas, now, 0),
          " drawsIn=", num(Slot::HygDrawsSubmitted, now, 0),
          " drawsCommit=", num(Slot::HygDrawsCommitted, now, 0),
          dirty ? "   *** matNew>0: MATERIAL CHURN, NUMBERS BELOW ARE NOT COMPARABLE ***" : ""));
        const Read aligned = read(Slot::GpuAligned, now);
        if (aligned.seen && aligned.v < 0.5) {
          Logger::warn("[Perf.Report] HYGIENE  *** [Perf.GpuPass] reported SHIFTED -- "
                       "every GPU per-pass label below is meaningless ***");
        }

        // [Perf.EntryCensus]: the OBSERVER EFFECT, stated up front for the same
        // reason matNew is. ScopedCall costs two clock reads on every timed
        // immediate-context call, and TF2 makes hundreds of thousands of them a
        // frame, so a large probe figure means the frame-thread rows below are
        // measuring the measurement. This is a comparability gate, not a bug:
        // the counts and shares on the census line stay valid, the milliseconds
        // do not.
        const Read probe = read(Slot::EntryProbeMs, now);
        if (probe.seen && wall > 0.0) {
          const double probePct = probe.v * 100.0 / wall;
          Logger::warn(str::format(
            "[Perf.Report] HYGIENE  entryCalls=", num(Slot::EntryCallsPerFrame, now, 0),
            "/frame  mapKB=", num(Slot::EntryMapKbPerFrame, now, 0),
            "/frame  ScopedCall probe=", fmtF(probe.v, 2), " ms (",
            fmtF(probePct, 1), "% of frame)",
            probePct > 10.0
              ? "   *** THE INSTRUMENT IS >10% OF THE FRAME: frame-thread ms below"
                " are inflated; use [Perf.EntryCensus] net/shares ***"
              : ""));
        }
      }

      // ---- 0. the three timelines
      Logger::warn("[Perf.Report] --- 0. TIMELINES (these do NOT sum; frame ~= max of the CPU two) ---");
      Logger::warn(str::format(
        "[Perf.Report]   frame thread (PRESENT)      ", num(Slot::FrameThreadCpuMs, now),
        " ms  ", pctOf(Slot::FrameThreadCpuMs, now, wall),
        "  busy=", num(Slot::FrameThreadBusyPct, now, 1),
        "%  blocked=", num(Slot::FrameThreadBlockedMs, now), " ms"));
      Logger::warn(str::format(
        "[Perf.Report]   dxvk-cs                     ", num(Slot::CsExecMs, now),
        " ms  ", pctOf(Slot::CsExecMs, now, wall),
        "  busy=", num(Slot::CsBusyPct, now, 1),
        "%  idle=", num(Slot::CsIdleMs, now), " ms"));
      Logger::warn(str::format(
        "[Perf.Report]   GPU                         ", num(Slot::GpuPassTotalMs, now),
        " ms  ", pctOf(Slot::GpuPassTotalMs, now, wall),
        "  idle=", num(Slot::GpuIdleMs, now), " ms  fenceWait=",
        num(Slot::GpuFenceWaitMs, now), " ms"));

      // ---- 1. frame thread
      Logger::warn("[Perf.Report] --- 1. FRAME THREAD ---");
      static const Row kFrameRows[] = {
        { 0, "D3D11 entry points",        Slot::EntryTotalMs,        RowKind::kTotal },
        { 1, "DrawIndexedInstanced",      Slot::EntryDrawIdxInstMs,  RowKind::kNested },
        { 1, "DrawIndexed",               Slot::EntryDrawIndexedMs,  RowKind::kNested },
        { 1, "Draw",                      Slot::EntryDrawMs,         RowKind::kNested },
        { 1, "state-setting calls",       Slot::EntryStateMs,        RowKind::kNested },
        { 1, "-> Remix OnDraw* hook",     Slot::DrawEntryOnDrawMs,   RowKind::kNested },
        { 1, "-> device LockContext",     Slot::DrawEntryLockMs,     RowKind::kNested },
        // [Perf.EntryCensus]: the families inside "state-setting calls".
        // Counts and bytes for these are on the census line itself; only the
        // time comes here, because a count in an Ms column would be rescaled
        // against the wall like everything else in this table.
        { 1, "  queries (GetData etc)",   Slot::EntryQueryMs,        RowKind::kNested },
        { 1, "  Map/Unmap/UpdateSub",     Slot::EntryMapMs,          RowKind::kNested },
        { 1, "  *SetConstantBuffers",     Slot::EntryCbSetMs,        RowKind::kNested },
        { 1, "  *SetShaderResources",     Slot::EntrySrvSetMs,       RowKind::kNested },
        { 1, "  copies/clears",           Slot::EntryCopyMs,         RowKind::kNested },
        // Not a family: this is the ScopedCall instrument's own cost, spread
        // across every row above (including the draws). Nested for the same
        // reason they are -- it is already inside EntryTotalMs.
        { 1, "  of which: probe cost",    Slot::EntryProbeMs,        RowKind::kNested },
        { 0, "between entry points",      Slot::GapTotalMs,          RowKind::kTotal },
        { 1, "afterQueryEnd",             Slot::GapAfterQueryEndMs,  RowKind::kNested },
        { 1, "post-Present engine code",  Slot::BoundaryPostMs,      RowKind::kAltView },
      };
      emitRows(kFrameRows, sizeof(kFrameRows) / sizeof(kFrameRows[0]), now, wall);

      Logger::warn("[Perf.Report]   -- inside SubmitDraw ([Perf.SdStall]) --");
      static const Row kSdRows[] = {
        { 1, "xform",     Slot::SdXformMs,    RowKind::kTotal },
        { 1, "emit",      Slot::SdEmitMs,     RowKind::kTotal },
        { 1, "commit",    Slot::SdCommitMs,   RowKind::kTotal },
        { 1, "capture",   Slot::SdCaptureMs,  RowKind::kTotal },
        { 1, "head",      Slot::SdHeadMs,     RowKind::kTotal },
        { 1, "skinCull",  Slot::SdSkinCullMs, RowKind::kTotal },
        { 1, "vsIdx",     Slot::SdVsIdxMs,    RowKind::kTotal },
        { 1, "rest",      Slot::SdRestMs,     RowKind::kTotal },
      };
      emitRows(kSdRows, sizeof(kSdRows) / sizeof(kSdRows[0]), now, wall);

      // NV-DXVK [Perf.Report] v6.9g: STAGE ATTRIBUTION -- how much of each
      // stage its own named leaves actually account for. This is the check
      // that keeps finding real defects: the same "parent vs sum of children"
      // test on bt_extractXf exposed a 25.5% hole that had been silently
      // inflating xs_pre and had corrupted the replay tier's whole cost model.
      // Doing it by hand from the log got the mapping wrong twice, which is
      // why it lives here now.
      // READ IT: a stage at <10% unattributed is well understood and further
      // splitting will find nothing -- go after what its leaves are NAMED as
      // instead. A stage above ~25% has something real hiding in it.
      // seg7 "rest" has no marks at all, so it is 100% by construction.
      {
        Logger::warn("[Perf.Report]   -- stage attribution (wall - named leaves = residual) --");
        struct SA { const char* name; Slot wallSlot; Slot leafSlot; };
        static const SA kSa[] = {
          { "head",     Slot::SdHeadMs,     Slot::SdHeadLeavesMs     },
          { "vsIdx",    Slot::SdVsIdxMs,    Slot::SdVsIdxLeavesMs    },
          { "skinCull", Slot::SdSkinCullMs, Slot::SdSkinCullLeavesMs },
          { "xform",    Slot::SdXformMs,    Slot::SdXformLeavesMs    },
          { "commit",   Slot::SdCommitMs,   Slot::SdCommitLeavesMs   },
          { "capture",  Slot::SdCaptureMs,  Slot::SdCaptureLeavesMs  },
          { "emit",     Slot::SdEmitMs,     Slot::SdEmitLeavesMs     },
        };
        double residSum = 0.0;
        bool   anySeen  = false;
        for (const SA& sa : kSa) {
          const Read w = read(sa.wallSlot, now);
          const Read l = read(sa.leafSlot, now);
          if (!w.seen || !l.seen) {
            Logger::warn(str::format("[Perf.Report]     ", sa.name,
                                     "  (source not published -- probe gated off?)"));
            continue;
          }
          anySeen = true;
          const double resid = w.v - l.v;
          const double pct   = (w.v > 1e-9) ? (100.0 * resid / w.v) : 0.0;
          residSum += resid;
          char buf[192];
          snprintf(buf, sizeof(buf),
                   "[Perf.Report]     %-9s %6.2f ms  named %6.2f  residual %6.2f  (%3.0f%% unattributed)",
                   sa.name, w.v, l.v, resid, pct);
          Logger::warn(buf);
        }
        // "rest" carries no marks, so all of it is residual by construction.
        const Read rest = read(Slot::SdRestMs, now);
        if (rest.seen) {
          residSum += rest.v;
          char buf[192];
          snprintf(buf, sizeof(buf),
                   "[Perf.Report]     %-9s %6.2f ms  named %6.2f  residual %6.2f  (100%% by construction -- no marks)",
                   "rest", rest.v, 0.0, rest.v);
          Logger::warn(buf);
        }
        if (anySeen) {
          char buf[160];
          snprintf(buf, sizeof(buf),
                   "[Perf.Report]     %-9s                              %6.2f ms unattributed across SubmitDraw",
                   "TOTAL", residSum);
          Logger::warn(buf);
        }
      }

      // NV-DXVK [Perf.Report] v6.9g: THE REPLAY TIER, IN ONE PLACE.
      // These are all children of bt_extractXf. They are broken out because the
      // tier's cost was assembled by hand out of the raw acc line for weeks,
      // each field with a different denominator, and two of the six did not
      // exist at all -- rp_selMiss and xt_cap were 1.67 ms/frame charged to no
      // bucket. That is what allowed a model to predict the tier saved 3.6
      // ms/frame while a direct F7 A/B measured it costing 1.18.
      // READ IT AS: "tier tax TOTAL" is what the tier spends. What it saves is
      // (full-path cost - replay cost) x hits, which is NOT on this report --
      // so a total approaching bt_extractXf means the tier is paying for
      // itself only if nearly every draw hits.
      // rp_o2w is listed but NOT in the total: both paths pay it.
      Logger::warn("[Perf.Report]   -- replay tier (children of bt_extractXf) --");
      static const Row kRpRows[] = {
        { 1, "rp_key    (all draws)",        Slot::AccRpKeyMs,     RowKind::kTotal },
        { 1, "rp_sel    (hits)",             Slot::AccRpSelMs,     RowKind::kTotal },
        { 1, "rp_selMiss(searched, missed)", Slot::AccRpSelMissMs, RowKind::kTotal },
        { 1, "rp_proof  (route gates)",      Slot::AccRpProofMs,   RowKind::kTotal },
        { 1, "rp_commit (member restore)",   Slot::AccRpCommitMs,  RowKind::kTotal },
        { 1, "xt_cap    (record capture)",   Slot::AccXtCapMs,     RowKind::kTotal },
        { 0, "TIER TAX total",               Slot::RpTierTotalMs,  RowKind::kTotal },
        { 1, "rp_o2w  (BOTH paths, not tax)", Slot::AccRpO2wMs,    RowKind::kAltView },
      };
      emitRows(kRpRows, sizeof(kRpRows) / sizeof(kRpRows[0]), now, wall);

      // NV-DXVK [Perf.ServeSplit] 2026-08-20 -- THE SERVE HALF OF bt_extractXf.
      //
      // READ THIS BEFORE THE NAMED-LEAF TABLE BELOW. Every child of
      // bt_extractXf in that table is a markXt site INSIDE ExtractTransforms,
      // and the split cache's whole purpose is NOT to call ExtractTransforms.
      // So on a served draw the parent bills and not one child does. The
      // "children sum to 96% of the parent" the source comments quote was
      // measured when the serve rate was 0%; at a 74% serve rate the same
      // arithmetic leaves the majority of the bucket dark.
      //
      // HOW TO READ IT: SERVE TOTAL against bt_extractXf in the table below.
      // The remainder is the derivation half, and only THAT remainder is what
      // the markXt children are a breakdown of.
      //   sv_pre/keys/lookup   TIER TAX -- every split draw pays these, hit or
      //                        miss. For the ~26% that miss, the cache is pure
      //                        added cost and this is how much.
      //   sv_carrier           replaying the skipped derivation's SIDE EFFECTS
      //                        (cbLoc + the ~14 cam member writes). Not a
      //                        compose; work the cache cannot avoid, only move.
      //   sv_compose           the actual compose. If this is small, then
      //                        THE_OPTIMISATION_PLAN sec 0.2's candidate --
      //                        "the compose costs what the derivation did" --
      //                        is refuted and the serve cost is the keys and
      //                        the lookup, which are a different fix entirely.
      Logger::warn("[Perf.Report]   -- split-cache SERVE path (bt_extractXf's other half; NO markXt child covers it) --");
      static const Row kSvRows[] = {
        { 1, "sv_pre     (all split draws)", Slot::AccSvPreMs,     RowKind::kTotal },
        { 1, "sv_keys    (all, tier tax)",   Slot::AccSvKeysMs,    RowKind::kTotal },
        { 1, "sv_lookup  (all, tier tax)",   Slot::AccSvLookupMs,  RowKind::kTotal },
        { 1, "sv_carrier (serve: restores)", Slot::AccSvCarrierMs, RowKind::kTotal },
        { 1, "sv_subView (serve: re-derive)",Slot::AccSvSubViewMs, RowKind::kTotal },
        { 1, "sv_compose (serve: the compose)", Slot::AccSvComposeMs, RowKind::kTotal },
        { 1, "sv_store   (store + miss tail)", Slot::AccSvStoreMs, RowKind::kTotal },
        { 0, "SERVE TOTAL (vs bt_extractXf)", Slot::SvTotalMs,     RowKind::kTotal },
        // SUMMED OVER DRAWS, SO IT CAN EXCEED THE FRAME and that is not a bug:
        // ~1,330 draws each wait concurrently in the same queue. It is here to
        // be watched, not added -- it belongs to no stage and no total.
        { 0, "chain enqueue->exec WAIT (sum over draws; latency, not work)",
                                             Slot::XfDeferWaitMs,  RowKind::kTotal },
      };
      emitRows(kSvRows, sizeof(kSvRows) / sizeof(kSvRows[0]), now, wall);

      Logger::warn("[Perf.Report]   -- named leaves ([Perf.SubmitDraw.acc]; parent+children = bucket) --");
      static const Row kAccRows[] = {
        { 1, "bt_extractXf",   Slot::AccExtractXfMs,    RowKind::kTotal },
        { 1, "tail_emit",      Slot::AccTailEmitMs,     RowKind::kTotal },
        { 2, "te_census",      Slot::AccTeCensusMs,     RowKind::kNested },
        { 2, "te_camDiag",     Slot::AccTeCamDiagMs,    RowKind::kNested },
        { 2, "te_propId",      Slot::AccTePropIdMs,     RowKind::kNested },
        { 1, "w2vw_cb3",       Slot::AccW2vwCb3Ms,      RowKind::kTotal },
        { 1, "skyClassify",    Slot::AccSkyClassifyMs,  RowKind::kTotal },
        { 1, "pfs_guard",      Slot::AccPfsGuardMs,     RowKind::kTotal },
        { 1, "filters",        Slot::AccFiltersMs,      RowKind::kTotal },
        { 1, "cbc_rawUv",      Slot::AccCbcRawUvMs,     RowKind::kTotal },
        { 1, "bonePalette",    Slot::AccBonePaletteMs,  RowKind::kTotal },
        { 1, "tail_capture",   Slot::AccTailCaptureMs,  RowKind::kTotal },
        { 1, "vsAnalysis",     Slot::AccVsAnalysisMs,   RowKind::kTotal },
        { 1, "o2w_t31",        Slot::AccO2wT31Ms,       RowKind::kTotal },
        { 1, "bt_cullVtx",     Slot::AccCullVtxMs,      RowKind::kTotal },
      };
      emitRows(kAccRows, sizeof(kAccRows) / sizeof(kAccRows[0]), now, wall);

      // 2026-08-14: the inside of pfs_guard, which is the largest named item on
      // the frame thread. Its own section because it is the ONLY block here that
      // survives RTX_PERF_MARKS=0 -- every kAccRows entry above is markStg-gated
      // and reads 0 on a normal run, including pfs_guard itself. So on a normal
      // run these rows have real values and their parent above shows 0. That
      // inversion is expected; the guard is measured by independent clock pairs
      // precisely so it can be read without the 30-marks-per-draw tax.
      Logger::warn("[Perf.Report]   -- captureDrawSnapshot = the inside of"
                   " pfs_guard (LIVE WITH MARKS OFF; pfs_guard above reads 0"
                   " when they are) --");
      static const Row kSnapRows[] = {
        { 1, "drawSnap TOTAL",       Slot::AccDrawSnapMs,      RowKind::kTotal  },
        { 2, "alloc (arena slot)",   Slot::AccDrawSnapAllocMs, RowKind::kNested },
        { 2, "identity block",       Slot::AccDrawSnapIdMs,    RowKind::kNested },
        { 2, "cb span copy",         Slot::AccDrawSnapSpanMs,  RowKind::kNested },
        // Children of the span copy. sdep was 70% of it until the byte-at-a-time
        // FNV was replaced with a word-wise four-lane mix; if it climbs back the
        // hash regressed or a new field was added to a carrier group.
        { 3, "stageDepCarrierGroups", Slot::AccCsSdepMs,       RowKind::kNested },
        { 3, "named-span locate",     Slot::AccCsPreMfMs,      RowKind::kNested },
        { 3, "manifest block",        Slot::AccCsManifestMs,   RowKind::kNested },
        // NOT disjoint from the two above -- capture() is called from both, and
        // the WC fill is inside capture(). Alt views, so the depths above still
        // read as a partition of the span copy.
        { 3, "capture() all sites",   Slot::AccCsCaptureMs,    RowKind::kAltView },
        { 4, "write-combined fill",   Slot::AccCsWcMs,         RowKind::kAltView },
      };
      emitRows(kSnapRows, sizeof(kSnapRows) / sizeof(kSnapRows[0]), now, wall);

      Logger::warn("[Perf.Report]   -- fanout, which lives OUTSIDE SubmitDraw --");
      static const Row kInstRows[] = {
        { 1, "SubmitInstancedDraw total", Slot::InstTotalMs,      RowKind::kTotal },
        { 2, "inner SubmitDraw",          Slot::InstInnerMs,      RowKind::kNested },
        { 2, "build",                     Slot::InstBuildMs,      RowKind::kNested },
        { 3, "setupPost",                 Slot::InstSetupPostMs,  RowKind::kNested },
        { 4, "camOrigin total",           Slot::InstCamOriginMs,  RowKind::kNested },
        { 5, "co_cbRead",                 Slot::InstCbReadMs,     RowKind::kNested },
        { 5, "co_vpScan",                 Slot::InstVpScanMs,     RowKind::kNested },
        { 4, "dbgTrack",                  Slot::InstDbgTrackMs,   RowKind::kNested },
        { 4, "UNATTRIBUTED residual",     Slot::InstResidualMs,   RowKind::kNested },
        { 3, "loop",                      Slot::InstLoopMs,       RowKind::kNested },
        { 4, "inst_loop",                 Slot::InstInstLoopMs,   RowKind::kNested },
        { 4, "t31_copy",                  Slot::InstT31CopyMs,    RowKind::kNested },
        { 4, "t31_gather",                Slot::InstT31GatherMs,  RowKind::kNested },
      };
      emitRows(kInstRows, sizeof(kInstRows) / sizeof(kInstRows[0]), now, wall);

      // ---- 2. dxvk-cs
      Logger::warn("[Perf.Report] --- 2. DXVK-CS ---");
      static const Row kCsRows[] = {
        { 0, "commitGeometryToRT",     Slot::CommitRtMs,        RowKind::kTotal },
        { 1, "submitDrawState",        Slot::CommitSubmitMs,    RowKind::kNested },
        { 2, "processDrawCallState",   Slot::ProcDcsMs,         RowKind::kNested },
        { 3, "processSceneObject",     Slot::ProcDcsInstMs,     RowKind::kNested },
        { 4, "updateInstance",         Slot::SceneObjUpdateMs,  RowKind::kNested },
        // [Perf.UpdInst] stages. fastRet is the 92.5% fast-path body that was
        // timed into no stage before 2026-08-09; if it is large, that is where
        // the "unattributed" block inside processSceneObjectImpl went.
        { 5, "ui_fastRet",             Slot::UpdInstFastRetMs,  RowKind::kNested },
        { 5, "ui_entry",               Slot::UpdInstEntryMs,    RowKind::kNested },
        { 5, "ui_surf",                Slot::UpdInstSurfMs,     RowKind::kNested },
        { 5, "ui_xform",               Slot::UpdInstXformMs,    RowKind::kNested },
        { 5, "ui_flags",               Slot::UpdInstFlagsMs,    RowKind::kNested },
        { 5, "ui_tail",                Slot::UpdInstTailMs,     RowKind::kNested },
        { 5, "ui_rest (vm/bb/cen/ac)", Slot::UpdInstRestMs,     RowKind::kNested },
        { 4, "findSimilarInstance",    Slot::SceneObjFindMs,    RowKind::kNested },
        { 4, "mid",                    Slot::SceneObjMidMs,     RowKind::kNested },
        { 4, "addInstance",            Slot::SceneObjAddMs,     RowKind::kNested },
        { 3, "geom",                   Slot::ProcDcsGeomMs,     RowKind::kNested },
        { 2, "determineMaterialData",  Slot::MatDataMs,         RowKind::kNested },
        { 1, "finalize (worker join)", Slot::CommitFinalizeMs,  RowKind::kNested },
        { 0, "the once-per-frame fat chunk", Slot::CsFatChunkMs, RowKind::kTotal },
        { 1, "prepareSceneData",       Slot::PrepSceneMs,       RowKind::kNested },
        { 2, "merge",                  Slot::PrepMergeMs,       RowKind::kNested },
        { 3, "loop",                   Slot::MergeLoopMs,       RowKind::kNested },
        { 3, "buildBlases",            Slot::MergeBuildBlasMs,  RowKind::kNested },
        { 3, "dynBlas",                Slot::MergeDynBlasMs,    RowKind::kNested },
        { 3, "setup",                  Slot::MergeSetupMs,      RowKind::kNested },
        { 2, "gc",                     Slot::PrepGcMs,          RowKind::kNested },
        { 2, "surfMat",                Slot::PrepSurfMatMs,     RowKind::kNested },
        { 2, "setup1+instSetup",       Slot::PrepSetupMs,       RowKind::kNested },
        { 2, "tlas+accel+light",       Slot::PrepTlasMs,        RowKind::kNested },
      };
      emitRows(kCsRows, sizeof(kCsRows) / sizeof(kCsRows[0]), now, wall);

      // ---- 3. GPU
      Logger::warn("[Perf.Report] --- 3. GPU ---");
      static const Row kGpuRows[] = {
        { 0, "GpuPass total",     Slot::GpuPassTotalMs,   RowKind::kTotal },
        { 1, "pt_integrate",      Slot::GpuPtIntegrateMs, RowKind::kNested },
        { 1, "gb_primaryRays",    Slot::GpuPrimaryRaysMs, RowKind::kNested },
        { 1, "prepScene (GPU)",   Slot::GpuPrepSceneMs,   RowKind::kNested },
        { 1, "pt_neeCache",       Slot::GpuNeeCacheMs,    RowKind::kNested },
        { 1, "upscaler",          Slot::GpuUpscalerMs,    RowKind::kNested },
        { 1, "finalBlit",         Slot::GpuFinalBlitMs,   RowKind::kNested },
        { 1, "pt_rtxdi",          Slot::GpuRtxdiMs,       RowKind::kNested },
        { 0, "GPU IDLE",          Slot::GpuIdleMs,        RowKind::kTotal },
      };
      emitRows(kGpuRows, sizeof(kGpuRows) / sizeof(kGpuRows[0]), now, wall);

      // ---- 4. closure checks
      Logger::warn("[Perf.Report] --- 4. CROSS-VALIDATION (two routes to the same span) ---");
      {
        const Read entry = read(Slot::EntryTotalMs, now);
        const Read gap   = read(Slot::GapTotalMs, now);
        check("frame thread   Entry+Gap  vs wall",
              entry.v + gap.v, wall, 6.0, entry.seen && gap.seen, true);

        const Read sdSum = [&] {
          Read r;
          const Slot stages[] = { Slot::SdHeadMs, Slot::SdVsIdxMs, Slot::SdSkinCullMs,
                                  Slot::SdXformMs, Slot::SdCommitMs, Slot::SdCaptureMs,
                                  Slot::SdEmitMs, Slot::SdRestMs };
          r.seen = true;
          for (const Slot s : stages) {
            const Read x = read(s, now);
            r.seen = r.seen && x.seen;
            r.v += x.v;
          }
          return r;
        }();
        const Read instInner = read(Slot::InstInnerMs, now);
        const Read drawIdx   = read(Slot::EntryDrawIndexedMs, now);
        check("SubmitDraw     SdStall sum vs inner+DrawIndexed",
              sdSum.v, instInner.v + drawIdx.v, 12.0,
              sdSum.seen, instInner.seen && drawIdx.seen);

        const Read instTotal = read(Slot::InstTotalMs, now);
        const Read entryInst = read(Slot::EntryDrawIdxInstMs, now);
        check("fanout         InstDraw    vs DrawIdxInst entry",
              instTotal.v, entryInst.v, 10.0, instTotal.seen, entryInst.seen);

        const Read soSum = [&] {
          Read r;
          const Slot st[] = { Slot::SceneObjFindMs, Slot::SceneObjMidMs,
                              Slot::SceneObjAddMs,  Slot::SceneObjUpdateMs };
          r.seen = true;
          for (const Slot s : st) {
            const Read x = read(s, now);
            r.seen = r.seen && x.seen;
            r.v += x.v;
          }
          return r;
        }();
        const Read pdInst = read(Slot::ProcDcsInstMs, now);
        check("dxvk-cs        SceneObj    vs ProcDCS instMs",
              soSum.v, pdInst.v, 15.0, soSum.seen, pdInst.seen);

        // TWO INSTRUMENTS ON updateInstance. [Perf.SceneObj]'s `update` bucket
        // brackets the call; [Perf.UpdInst]'s stages split its inside. They are
        // independent, so they are worth holding against each other -- rtx.conf
        // ~1451 asked for exactly this check and never got a clean answer.
        //
        // History, so a FAIL here is read correctly: these disagreed 12.86 vs
        // 4.47 and the gap was blamed on unattributed work inside
        // processSceneObjectImpl. The real cause was a missing mark on the
        // 92.5% fast-path return, now stage `fastRet`. So if this FAILs again,
        // suspect a THIRD conditional exit between marks before believing in
        // work that no probe can see -- and read [Perf.UpdInst] reachPct, which
        // was added to make that failure mode visible directly.
        const Read uiTot = read(Slot::UpdInstTotalMs, now);
        const Read soUpd = read(Slot::SceneObjUpdateMs, now);
        check("dxvk-cs        UpdInst sum vs SceneObj update",
              uiTot.v, soUpd.v, 15.0, uiTot.seen, soUpd.seen);

        const Read prep = read(Slot::PrepSceneMs, now);
        const Read fat  = read(Slot::CsFatChunkMs, now);
        check("dxvk-cs        PrepScene   vs CsSplit fat chunk",
              prep.v, fat.v, 25.0, prep.seen, fat.seen);
      }

      // ---- 5. the verdict
      Logger::warn("[Perf.Report] --- 5. VERDICT ---");
      {
        const bool haveFt  = read(Slot::FrameThreadCpuMs, now).seen;
        const bool haveCs  = read(Slot::CsExecMs, now).seen;
        const bool haveGpu = read(Slot::GpuPassTotalMs, now).seen
                          && read(Slot::GpuIdleMs, now).seen;

        const char* poleName = "unknown";
        double poleMs   = 0.0;
        double secondMs = 0.0;

        if (haveFt || haveCs) {
          if (ftCpu >= csExec) {
            poleName = "frame thread"; poleMs = ftCpu; secondMs = csExec;
          } else {
            poleName = "dxvk-cs";      poleMs = csExec; secondMs = ftCpu;
          }
        }

        // GPU is only the pole when it has essentially no idle. The map's section 5
        // trap is the opposite reasoning: [Perf.Gpu]'s three states always sum to
        // the frame, and [Perf.GpuPass] outsideRtMs is GPU-timeline time that
        // INCLUDES idle, so neither can be used to call GPU-bound. Idle is the
        // only field that can, and it has to be small in absolute terms.
        const bool gpuBound = haveGpu
                           && (gpuIdl < 0.15 * wall)
                           && (gpuBsy > 0.80 * wall);

        if (gpuBound) {
          Logger::warn(str::format(
            "[Perf.Report]   BOTTLENECK: GPU.  busy ", fmtF(gpuBsy, 2),
            " ms of a ", fmtF(wall, 2), " ms frame, idle only ",
            fmtF(gpuIdl, 2), " ms."));
          Logger::warn("[Perf.Report]   CPU work is hidden behind it -- do not optimise CPU here.");
        } else {
          const double slack = std::max(0.0, poleMs - secondMs);
          Logger::warn(str::format(
            "[Perf.Report]   BOTTLENECK: CPU, on the ", poleName, " (",
            fmtF(poleMs, 2), " ms). Second is ",
            fmtF(secondMs, 2), " ms, so SLACK = ",
            fmtF(slack, 2), " ms."));
          if (haveGpu) {
            Logger::warn(str::format(
              "[Perf.Report]   GPU is NOT the limit: ", fmtF(gpuBsy, 2),
              " ms busy, ", fmtF(gpuIdl, 2), " ms IDLE (",
              fmtF(100.0 * gpuIdl / wall, 0), "% of frame). Headroom ~",
              fmtF(wall - gpuBsy, 2), " ms."));
          }
          if (slack < 1.0) {
            Logger::warn(str::format(
              "[Perf.Report]   *** THE TWO CPU THREADS ARE LEVEL (slack ",
              fmtF(slack, 2), " ms). Work removed from ", poleName,
              " alone buys ~NOTHING -- both must come down together. ***"));
          }

          // Rank what is actually on the pole thread, and cap each by the slack.
          // An 8 ms win on a thread with 2 ms of slack is a 2 ms win; that cap is
          // the whole point of this section and is what nobody was computing.
          struct Target { const char* name; Slot slot; bool onFrameThread; };
          static const Target kTargets[] = {
            // 2026-08-14: the guard's inside. These are the ONLY frame-thread
            // targets here that are live with RTX_PERF_MARKS=0 -- everything
            // else on this list is markStg-gated, which is why this section has
            // been printing "(none published)" on every normal run while
            // captureDrawSnapshot was ~15% of the frame thread. Ranked first in
            // the array only for readability; the sort below is by size.
            { "captureDrawSnapshot (IS pfs_guard)",       Slot::AccDrawSnapMs,    true  },
            // NESTED INSIDE the row above -- do not add them to it. Listed
            // because the split is where the decisions are: the span copy is the
            // part §4 of THE_OPTIMISATION_PLAN is about, and the identity block
            // is the part that has never been attacked at all.
            { "  cb span copy (INSIDE drawSnap)",         Slot::AccDrawSnapSpanMs, true },
            { "  identity block (INSIDE drawSnap)",       Slot::AccDrawSnapIdMs,  true  },
            // NESTED INSIDE the span copy. Kept on the list after being fixed
            // (byte-at-a-time FNV -> word-wise four-lane, ~9.7x) precisely so a
            // regression re-ranks itself here instead of hiding: it was 70% of
            // the span copy and nothing in this report could see it.
            { "    stageDepCarrierGroups (INSIDE span)",  Slot::AccCsSdepMs,      true  },
            { "fanout co_cbRead (BSP camOrigin CB read)", Slot::InstCbReadMs,     true  },
            { "fanout setupPost residual (UNATTRIBUTED)", Slot::InstResidualMs,   true  },
            { "fanout dbgTrack (diagnostic?)",            Slot::InstDbgTrackMs,   true  },
            { "bt_extractXf",                             Slot::AccExtractXfMs,   true  },
            // v6.9g: NESTED INSIDE bt_extractXf above -- do not add the two
            // together. Listed separately because it is the only entry here
            // that can be removed outright rather than optimised: an F7 A/B on
            // 2026-08-09 measured bt_extractXf at 6.12 ms with the tier ON and
            // 4.95 ms with it OFF (n=57/53, sd 0.39/0.30), so this tax is not
            // currently bought back by what the tier saves.
            { "  replay tier tax (INSIDE bt_extractXf)",  Slot::RpTierTotalMs,    true  },
            { "SubmitDraw emit stage",                    Slot::SdEmitMs,         true  },
            { "SubmitDraw xform stage",                   Slot::SdXformMs,        true  },
            { "SubmitDraw commit stage",                  Slot::SdCommitMs,       true  },
            { "SubmitDraw capture stage",                 Slot::SdCaptureMs,      true  },
            { "inst_loop",                                Slot::InstInstLoopMs,   true  },
            { "te_census + te_camDiag (diagnostics?)",    Slot::AccTeCensusMs,    true  },
            { "skyClassify",                              Slot::AccSkyClassifyMs, true  },
            { "w2vw_cb3",                                 Slot::AccW2vwCb3Ms,     true  },
            { "prepScene merge loop",                     Slot::MergeLoopMs,      false },
            { "prepScene buildBlases",                    Slot::MergeBuildBlasMs, false },
            { "updateInstance",                           Slot::SceneObjUpdateMs, false },
            // NESTED INSIDE updateInstance above -- do not add the two together.
            // Listed on its own because it is the one stage in here that is the
            // FAST path: work the fast path was believed to have skipped. Every
            // ms shown is work still being done on 92.5% of instances, and it was
            // invisible to every instrument until 2026-08-09.
            { "  ui_fastRet (INSIDE updateInstance)",     Slot::UpdInstFastRetMs, false },
            { "findSimilarInstance",                      Slot::SceneObjFindMs,   false },
            { "prepScene gc",                             Slot::PrepGcMs,         false },
            { "ProcDCS geom",                             Slot::ProcDcsGeomMs,    false },
          };

          const bool poleIsFrameThread = (ftCpu >= csExec);
          struct Ranked { const char* name; double raw; double realised; };
          std::vector<Ranked> ranked;
          for (const Target& t : kTargets) {
            if (t.onFrameThread != poleIsFrameThread) {
              continue;
            }
            const Read r = read(t.slot, now);
            if (!r.seen || r.v <= 0.0) {
              continue;
            }
            ranked.push_back({ t.name, r.v, std::min(r.v, slack) });
          }
          std::sort(ranked.begin(), ranked.end(),
                    [](const Ranked& a, const Ranked& b) { return a.raw > b.raw; });

          Logger::warn(str::format(
            "[Perf.Report]   Targets ON THE POLE (", poleName,
            "), 'realised' = min(size, slack):"));
          const size_t shown = std::min<size_t>(ranked.size(), 8u);
          for (size_t i = 0; i < shown; ++i) {
            Logger::warn(str::format(
              "[Perf.Report]     ", i + 1, ". ", ranked[i].name,
              std::string(std::max<int>(1, 44 - int(std::string(ranked[i].name).size())), ' '),
              fmtF(ranked[i].raw, 2), " ms  -> realised ",
              fmtF(ranked[i].realised, 2), " ms"));
          }
          if (ranked.empty()) {
            Logger::warn("[Perf.Report]     (none published -- the per-stage probes for "
                         "this thread are gated off)");
          }

          // What the frame would become if the top target were free.
          //
          // THE OVERHEAD TERM IS LOAD-BEARING. poleMs is the pole thread's CPU
          // time; the frame WALL is larger by whatever that thread spends
          // blocked (and by any serial gap neither thread covers). Deleting CPU
          // work does not delete that, so projecting to the new pole alone
          // reports a frame that is impossible -- and it reported it while the
          // slack banner directly above said the opposite, which is the worse
          // failure of the two because the optimistic line is the memorable one.
          if (!ranked.empty()) {
            const double overhead = std::max(0.0, wall - poleMs);
            const double newPole  = std::max(poleMs - ranked[0].raw, secondMs);
            const double newWall  = std::max(newPole + overhead, haveGpu ? gpuBsy : 0.0);
            Logger::warn(str::format(
              "[Perf.Report]   If #1 were free: frame ", fmtF(wall, 2),
              " -> ~", fmtF(newWall, 2), " ms  (",
              fmtF(1000.0 / std::max(newWall, 0.001), 1), " fps, from ",
              fmtF(fps, 1), ")   [pole ", fmtF(poleMs, 2), " -> ", fmtF(newPole, 2),
              ", + ", fmtF(overhead, 2), " ms not-CPU overhead that stays]"));
          }
        }
      }

      Logger::warn("[Perf.Report] ================================================================");
    }

    void onFrameEnd(const uint32_t) {
      // Own counter rather than the device's: publish sites on dxvk-cs and
      // dxvk-queue have no device in scope, and this only needs to be monotonic.
      const uint32_t f = s_frameCounter.fetch_add(1u, std::memory_order_relaxed) + 1u;

      const uint32_t every = RtxOptions::perfReportFrames();
      if (every == 0u) {
        return;
      }
      static uint32_t s_lastEmit = 0u;
      if (f - s_lastEmit < every) {
        return;
      }
      s_lastEmit = f;
      emitReport(f);
    }

  }

}
