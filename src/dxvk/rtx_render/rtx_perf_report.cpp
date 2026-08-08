#include "rtx_perf_report.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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
            { "fanout co_cbRead (BSP camOrigin CB read)", Slot::InstCbReadMs,     true  },
            { "fanout setupPost residual (UNATTRIBUTED)", Slot::InstResidualMs,   true  },
            { "fanout dbgTrack (diagnostic?)",            Slot::InstDbgTrackMs,   true  },
            { "bt_extractXf",                             Slot::AccExtractXfMs,   true  },
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
