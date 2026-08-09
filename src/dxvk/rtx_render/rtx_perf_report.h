#pragma once

// NV-DXVK [Perf.Report] — the assembled frame breakdown, and the bottleneck verdict.
//
// WHY THIS EXISTS. The repo has ~90 [Perf.*] instruments. Every one of them is
// correct and every one of them is PARTIAL: it reports its own window, on its own
// cadence, in its own units, on one of six concurrent timelines. Assembling them
// into "what is the frame" has been done by hand, in a text editor, once per
// session, and got the answer WRONG at least three times in recorded history:
//   - dxvk-cs named as the pole when the frame thread was higher
//   - [Perf.Gpu] idleMs+fenceWaitMs+reapMs == frame read as evidence of GPU-bound
//     when that identity is true by construction (PERF_INSTRUMENTATION_MAP §5)
//   - [Perf.GpuPass] outsideRtMs added to totalMs and matched against frame wall,
//     which is the same mistake wearing a different hat: outsideRtMs is
//     GPU-timeline time outside the RT passes and INCLUDES the GPU sitting idle
//
// So this does not measure anything. Every value here is already computed by an
// existing probe for its own line; the emit sites publish() what they just
// calculated, and this file only assembles, cross-checks and ranks. Adding a
// slot must never add a clock read.
//
// THE THREE RULES IT ENCODES, so they stop being re-derived:
//   1. The timelines do NOT sum. frame ~= max(frame thread, dxvk-cs). The GPU is
//      a third clock and is only the pole when it has no idle.
//   2. Nested instruments must not be added to their parents (§2 of the map).
//      Anything marked kNested here is a child and is excluded from totals.
//   3. A saving on the pole thread is worth min(saving, poleMs - secondMs).
//      Past that, the other thread becomes the pole and the rest is free time.
//      This is the number that should drive what gets worked on, and it is the
//      one nobody computed.

#include <atomic>
#include <cstdint>

namespace dxvk {

  namespace perfreport {

    // Every value the report can hold. Adding one here costs a double and a
    // uint32 of static storage and nothing at runtime.
    //
    // Units are NORMALISED AT THE PUBLISH SITE: every Ms slot is milliseconds
    // PER FRAME, every Pct slot is 0-100, every Count slot is per frame. The
    // source instruments disagree wildly on units (§5 of the map: acc is ns
    // except wallUs; PrepScene/Merge/Frame are us; Busy/Gpu/GpuPass/CsSplit are
    // ms) and normalising at the source is the only place the divisor is known.
    enum class Slot : uint32_t {
      // ---- frame wall + the two CPU poles ----
      FrameWallMs,             // [Perf.Busy] wallMs
      FrameThreadCpuMs,        // [Perf.Busy] cpuMs      <- the pole candidate
      FrameThreadBusyPct,      // [Perf.Busy] busyPct
      FrameThreadBlockedMs,    // [Perf.Busy] blockedMs
      CsExecMs,                // [Perf.CsSplit] execMs/frames  <- pole candidate
      CsIdleMs,
      CsBusyPct,

      // ---- frame thread: the top-level split that must close to the wall ----
      EntryTotalMs,            // [Perf.Entry] totalMs   (inside D3D11 entry points)
      GapTotalMs,              // [Perf.Gap]   totalMs   (between them)
      EntryDrawIdxInstMs,
      EntryDrawIndexedMs,
      EntryDrawMs,
      EntryStateMs,            // everything that is not a draw entry point
      GapAfterQueryEndMs,
      BoundaryPostMs,          // [Perf.Boundary] postMsAvg — Present exit -> next call
      DrawEntryLockMs,         // [Perf.DrawEntry] deLock, normalised
      DrawEntryOnDrawMs,       // [Perf.DrawEntry] deOnDraw, normalised

      // ---- frame thread: inside SubmitDraw ([Perf.SdStall], 8 stages) ----
      SdHeadMs, SdVsIdxMs, SdSkinCullMs, SdXformMs,
      SdCommitMs, SdCaptureMs, SdEmitMs, SdRestMs,

      // Sum of the NAMED leaves that fall inside each stage, so the report can
      // print stage - leaves = residual instead of that being reconstructed by
      // hand. Added 2026-08-09 after doing it by hand and getting it wrong
      // twice in a row: the first pass filed flt_gate/flt_probes under vsIdx
      // (they are in xform) and treated the nine te_* accumulators as children
      // of tail_emit rather than siblings inside emit. That produced a phantom
      // "emit is 42% unattributed, 2.7 ms/frame" which survived until the
      // mapping was derived from the actual markStg line numbers -- the true
      // figure is 8% and 0.51 ms/frame.
      // The mapping belongs HERE, next to the marks, not in whatever script
      // happens to be reading the log. A leaf added to a stage without being
      // added to its sum shows up as residual, which is the safe direction.
      SdHeadLeavesMs, SdVsIdxLeavesMs, SdSkinCullLeavesMs, SdXformLeavesMs,
      SdCommitLeavesMs, SdCaptureLeavesMs, SdEmitLeavesMs,

      // ---- frame thread: the fanout, which lives OUTSIDE SubmitDraw ----
      InstTotalMs, InstInnerMs, InstBuildMs, InstLoopMs,
      InstSetupPostMs, InstResidualMs,
      InstT31GatherMs, InstT31CopyMs, InstInstLoopMs,
      InstInstances, InstCalls,
      // setupPost's real children. These were ALREADY measured by [Perf.MapCut]
      // and [Perf.CamCut] while [Perf.InstDraw]'s residual_ms subtracted only
      // semScan and prep -- so "UNATTRIBUTED residual" was 62% one known item.
      // Whenever a residual is the biggest thing on a thread, check what the
      // subtraction actually covers before writing a new probe.
      InstCamOriginMs, InstCbReadMs, InstVpScanMs, InstDbgTrackMs,

      // ---- frame thread: named leaves from [Perf.SubmitDraw.acc] ----
      AccExtractXfMs, AccTailEmitMs, AccTeCensusMs, AccTeCamDiagMs,
      AccTePropIdMs, AccW2vwCb3Ms, AccSkyClassifyMs, AccPfsGuardMs,
      AccFiltersMs, AccCbcRawUvMs, AccBonePaletteMs, AccTailCaptureMs,
      AccVsAnalysisMs, AccO2wT31Ms, AccCullVtxMs,

      // ---- the replay tier, in one place (v6.9f/g) ----
      // These are children of AccExtractXfMs. They were invisible to this
      // report while the tier's whole cost model was being built on them, and
      // two of them did not exist as counters at all: rp_selMiss and xt_cap
      // together were 1.67 ms/frame charged to nothing, which is what let a
      // model predict the tier SAVED 3.6 ms/frame while an F7 A/B measured it
      // COSTING 1.18. Everything the tier spends is now on one screen, and
      // RpTierTotalMs is the number to hold against what it saves.
      // rp_o2w is deliberately OUTSIDE the total: it is paid on both the replay
      // and full paths, so it is not tier tax and must never be counted as a
      // saving (that mistake is the other half of the same wrong model).
      AccRpKeyMs, AccRpSelMs, AccRpSelMissMs, AccRpProofMs,
      AccRpCommitMs, AccXtCapMs, AccRpO2wMs, RpTierTotalMs,

      // ---- dxvk-cs: the per-draw chain (NESTED — see kNested) ----
      CommitRtMs, CommitSubmitMs, CommitFinalizeMs,
      ProcDcsMs, ProcDcsGeomMs, ProcDcsInstMs,
      SceneObjFindMs, SceneObjMidMs, SceneObjAddMs, SceneObjUpdateMs,
      MatDataMs,

      // ---- dxvk-cs: the once-per-frame serial pass ----
      CsFatChunkMs,            // [Perf.CsSplit] >=10ms bucket, per frame
      PrepSceneMs,             // [Perf.PrepScene] sum
      PrepMergeMs, PrepGcMs, PrepSurfMatMs, PrepSetupMs, PrepTlasMs,
      MergeLoopMs, MergeDynBlasMs, MergeBuildBlasMs, MergeSetupMs,

      // ---- GPU + queue ----
      GpuPassTotalMs,          // [Perf.GpuPass] totalMs   <- pole candidate
      GpuIdleMs,               // [Perf.Block]   gpuIdleMs
      GpuFenceWaitMs, GpuReapMs, GpuCmdLists,
      GpuPtIntegrateMs, GpuPrimaryRaysMs, GpuPrepSceneMs,
      GpuUpscalerMs, GpuFinalBlitMs, GpuNeeCacheMs, GpuRtxdiMs,
      GpuAligned,              // 1 = [Perf.GpuPass] said ALIGNED, 0 = SHIFTED

      // ---- hygiene: every number above is void without these (§6) ----
      HygMatNew, HygMatTotal, HygTexTotal,
      HygInstances, HygUniqueBlas, HygDrawsSubmitted, HygDrawsCommitted,

      kCount
    };

    // One published value. Two independent atomics rather than a lock: writers
    // are once-per-window per slot from at most three threads, and a torn read
    // between value and frame stamp costs at worst one stale age in a log line.
    struct Cell {
      std::atomic<double>   value { 0.0 };
      std::atomic<uint32_t> frame { 0u };
      std::atomic<uint32_t> seq   { 0u };   // publish count, for "never seen"
    };

    // The report's own frame counter, bumped by onFrameEnd. Publish sites use
    // this rather than reaching for DxvkDevice::getCurrentFrameId() -- they are
    // scattered across five files and three threads, and several of them (the
    // dxvk-cs and dxvk-queue ones) have no device handle in scope at all. It
    // only has to be monotonic and shared, which is exactly what staleness
    // needs.
    uint32_t currentFrame();

    // Publish a value the caller has ALREADY computed for its own log line.
    // Never compute something here that the caller did not otherwise need.
    void publish(Slot slot, double valuePerFrame, uint32_t frameId);

    inline void publish(const Slot slot, const double valuePerFrame) {
      publish(slot, valuePerFrame, currentFrame());
    }

    // Convenience for the common "window total -> per frame" normalisation.
    // Does nothing when frames == 0, so a caller mid-warmup cannot poison a slot.
    void publishWindow(Slot slot, double windowTotal, uint64_t frames, uint32_t frameId);

    inline void publishWindow(const Slot slot, const double windowTotal, const uint64_t frames) {
      publishWindow(slot, windowTotal, frames, currentFrame());
    }

    // For emit sites whose accumulators are WINDOW TOTALS with no frame count of
    // their own -- [Perf.DrawEntry], [Perf.LoopCut], [Perf.InstDraw] all count
    // draws or calls, which are not frames. Dividing a window total by a draw
    // count and calling the result "per frame" is the per-DRAW vs per-INSTANCE
    // error in another costume (map section 5), and it has recurred enough times
    // to deserve a type.
    //
    // Declare one `static` per call site; step() returns frames elapsed since
    // that site last published.
    struct WindowFrames {
      uint32_t last = 0u;
      uint32_t step() {
        const uint32_t now = currentFrame();
        const uint32_t d   = now - last;
        last = now;
        return d;
      }
    };

    // NV-DXVK [Perf.SessionState] -- "how fast is a millisecond right now".
    // Every instrument in this codebase measures execution TIME and silently
    // assumes execution SPEED is constant. Whole sessions have violated that
    // assumption (08-08: 39 vs 57 ms at one camera spot; 08-09: 45 vs ~150 ms,
    // every CPU slice on BOTH threads uniformly 3-4x, GPU pass identical,
    // draw/instance counts identical, staging hit rates identical, cleared by
    // a game restart) and the cause was unattributable because nothing
    // measured the millisecond itself. This does: a fixed dependent integer
    // spin (core speed), the cost of one clock read (QPC health), the current
    // processor number (E-core placement sample), and the explicit EcoQoS
    // state. Call it once per emit window ON the thread being measured --
    // ~50-80us per call, never per draw. The xMin field (spin vs this
    // thread's session-best) is the comparability verdict: a window with
    // xMin well above ~1.1 is not comparable to a healthy one, exactly like
    // the matNew rule.
    void sessionStateProbe(const char* who);

    // Called once per frame from the frame thread (D3D11Rtx::EndFrame). Emits the
    // assembled report every rtx.perfReportFrames frames. Cost when the option is
    // 0: one option read and an integer compare.
    void onFrameEnd(uint32_t frameId);

    // Exposed for tests / manual trigger.
    void emitReport(uint32_t frameId);

  }

}
