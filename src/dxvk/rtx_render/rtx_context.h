/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include "../dxvk_context.h"
#include "rtx_resources.h"
#include "rtx_asset_exporter.h"
#include "rtx_camera_manager.h"
#include "rtx_atmosphere.h"
#include "rtx/pass/nrd_args.h"
// NV-DXVK [perf]: DxvkDLFGTimestampQueryPool, reused for per-pass GPU timing.
#include "rtx_dlfg.h"

#include <cstdint>
#include <chrono>
#include <atomic>
#include "rtx_options.h"

struct VolumeArgs;
struct RaytraceArgs;

namespace dxvk {
  class DxvkContext;
  class AssetExporter;
  class SceneManager;
  class TerrainBaker;
  struct ExternalDrawState;

  struct RtxVertexCaptureData;
  struct RtxVSConstants;
  struct RtxPSConstants;
  struct RtxSharedPS;
  struct RtxLegacyLight;
  
  struct DrawParameters {
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndex = 0;
    uint32_t vertexOffset = 0;
  };
  /** 
   * \brief RTX context
   * 
   * Tracks pipeline state and records command lists.
   * This is where the actual rendering commands are
   * recorded.
   */

  class RtxContext : public DxvkContext {

  public:
    
    RtxContext(const Rc<DxvkDevice>& device);
    ~RtxContext();

    float getGpuIdleTimeSinceLastCall();

    /**
      * \brief Reset screen resolution, and resize all screen
      *        buffers to specified resolution if required.
      * 
      * \param [in] upscaleExtent: New desired resolution.
      */
    void resetScreenResolution(const VkExtent3D& upscaleExtent);

    /**
      * \brief Triggers the RTX renderer.  Writes to targetImage (if specified) 
      *        and the currently bound render target if not.
      *        Will flush the scene and perform other non rendering tasks.
      * 
      * \param [in] cachedReflexFrameId: The Reflex frame ID at the time of calling, cached so Reflex can have
      * consistent frame IDs throughout the dispatches of an application frame.
      * \param [in] targetImage: Image to store raytraced result in
      */
    // NV-DXVK: skipBackbufferBlit=true runs the full RT pipeline
    // (scene-mgmt, BVH, path trace, denoise) but skips the final
    // blit-to-backbuffer. Used by the D3D11 HUD-deferral early-inject
    // path when it fires before the camera has latched: we still want
    // scene state flushed (to avoid frame-over-frame accumulation), but
    // without the RT-over-HUD clobber. See rtx_context.cpp ~line 729.
    void injectRTX(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage = nullptr,
                   bool skipBackbufferBlit = false);
    void endFrame(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage = nullptr, bool callInjectRtx = true);

    void onPresent(Rc<DxvkImage> targetImage = nullptr);

    /**
      * \brief Adds a batch of lights to the scene context
      *
      * \param [in] pLights: array of light structures
      * \param [in] numLights: number of lights
      */
    void addLights(const RtxLegacyLight* pLights, const uint32_t numLights);

    void clearRenderTarget(const Rc<DxvkImageView>& imageView, VkImageAspectFlags clearAspects, VkClearValue clearValue);
    void clearImageView(const Rc<DxvkImageView>& imageView, VkOffset3D offset, VkExtent3D extent, VkImageAspectFlags aspect, VkClearValue value);

    void commitGeometryToRT(const DrawParameters& params, DrawCallState& drawCallState);
    void commitExternalGeometryToRT(ExternalDrawState&& state);

    static void blitImageHelper(Rc<DxvkContext> ctx, const Rc<DxvkImage>& srcImage, const Rc<DxvkImage>& dstImage, VkFilter filter);

    virtual void flushCommandList() override;

    SceneManager& getSceneManager();
    Resources& getResourceManager();
  
    static void triggerScreenshot() { s_triggerScreenshot = true; }
    static void triggerUsdCapture() { s_triggerUsdCapture = true; }
    // [SpawnGeomDiag.FloorObjDump] dedicated trigger — independent of
    // USD capture / scene capture infrastructure. Wired to its own
    // hotkey (Ctrl+Shift+O) in dxvk_imgui.cpp so it never piggybacks
    // on existing capture paths. Consumed by AccelManager.
    static void triggerObjDump() { s_triggerObjDump = true; }
    static bool consumeObjDumpRequest() {
      bool expected = true;
      return s_triggerObjDump.compare_exchange_strong(expected, false);
    }

    void bindCommonRayTracingResources(const Resources::RaytracingOutput& rtOutput);

    void bindResourceView(const uint32_t slot, const Rc<DxvkImageView>& imageView, const Rc<DxvkBufferView>& bufferView);

    void getDenoiseArgs(NrdArgs& outPrimaryDirectNrdArgs, NrdArgs& outPrimaryIndirectNrdArgs, NrdArgs& outSecondaryNrdArgs);
    void updateRaytraceArgsConstantBuffer(Resources::RaytracingOutput& rtOutput, const VkExtent3D& downscaledExtent, const VkExtent3D& targetExtent);

    RtxVertexCaptureData& allocAndMapVertexCaptureConstantBuffer();
    RtxVSConstants& allocAndMapVSConstantBuffer();
    RtxSharedPS& allocAndMapPSSharedStateConstantBuffer();

    static bool checkIsShaderExecutionReorderingSupported(DxvkDevice& device);

    const DxvkScInfo& getSpecConstantsInfo(VkPipelineBindPoint pipeline) const;
    void setSpecConstantsInfo(VkPipelineBindPoint pipeline, const DxvkScInfo& newSpecConstantInfo);

    bool useRayReconstruction() const;

#ifdef REMIX_DEVELOPMENT
    // Note: Cache image views for all resources that used by current frame, so we can do query for resource aliasing at the end of frame.
    //       This is automatically called when binding resources for passes, RtxContext::bindCommonRayTracingResources
    //       When we are not using the binding function in the passes such as DLSSRR, we need to manually cache the image views. Please reference the cache logic in DxvkRayReconstruction::dispatch
    void cacheResourceAliasingImageView(const Rc<DxvkImageView>& imageView);
#endif

    inline void setFramePassStage(const RtxFramePassStage currentFramePassStage) {
#ifdef REMIX_DEVELOPMENT
      m_currentPassStage = currentFramePassStage;
#endif
    }

    // NV-DXVK: suppress this frame's end-of-frame injectRTX.
    // Used by the D3D11 HUD-deferral early-inject path: if the early
    // inject fires but camera hasn't latched yet (camValid=0), we cannot
    // produce the RT composite this frame. Marking the frame as injected
    // prevents the endFrame injectRTX from blitting RT-over-HUD and
    // wiping the native HUD rasters — player sees at least HUD + raster
    // scene rather than RT scene with no HUD.
    inline void suppressInjectThisFrame(uint32_t frameId) {
      m_frameLastInjected = frameId;
    }

    // NV-DXVK: sticky flag for the D3D11 HUD-deferral path. When the
    // early-inject fires before the camera has latched, we CAN'T call
    // injectRTX early because scene-mgmt is gated behind isCameraValid
    // internally (nothing would happen). Instead, d3d11 sets this flag
    // and lets EndFrame's normal injectRTX run the full pipeline — but
    // the final blit-to-backbuffer step checks this flag and skips, so
    // the HUD rasters already written to the backbuffer survive. Reset
    // each frame inside injectRTX after the blit site.
    inline void requestSkipBackbufferBlit() {
      m_skipBackbufferBlitThisFrame = true;
    }

    // NV-DXVK [HUD-Option5 v4]: blit the POST-tonemap scratch over TF2's
    // composite-output target (the backbuffer image). Queued from the
    // D3D11Rtx side right after TF2's composite PS writes its tonemapped
    // RGB to the backbuffer, BEFORE TF2's HUD rasterizes on top, so HUD
    // draws land directly onto our RT.
    void blitPostTonemapScratchToCompositeOut(Rc<DxvkImage> compositeOut);
    // Marks the current frame as intercepted; injectRTX skips its own
    // backbuffer blit (TF2's native present-time copy carries the full
    // RT+HUD frame to the swap chain).
    inline void requestCompositeIntercept() { m_compositeInterceptedThisFrame = true; }
    inline bool rtFinalPostTonemapScratchHasContent() const { return m_rtFinalPostTonemapScratchHasContent; }


  protected:
    virtual void updateComputeShaderResources() override;
    virtual void updateRaytracingShaderResources() override;

  private:
    // NV-DXVK [perf]: GPU time per RT pass. See markGpuStage in injectRTX.
    //
    // The existing [Perf.Frame] stage timers measure CPU wall time around each
    // dispatch call — recording cost, not execution cost — and they total a few
    // ms while [Perf.Gpu] puts the GPU at ~350 ms/frame. These timestamps are
    // written into the command stream at the same boundaries so [Perf.GpuPass]
    // reads field-for-field against [Perf.Frame], and say which dispatch
    // actually owns the frame.
    //
    // Ring of kFrames so results are read back well after the GPU has passed
    // them; a slot that is not ready yet is skipped rather than waited on, so
    // this can never introduce a stall of its own.
    struct GpuStageTimers {
      static constexpr uint32_t kSlots  = 48;  // 27 used: t0 + 2 pre-branch + 15 stages + 5 path-tracer + 4 gbuffer sub-stages
      static constexpr uint32_t kFrames = 4;

      Rc<DxvkDLFGTimestampQueryPool> pool;
      uint32_t frameSlot  = 0;
      uint32_t writeCount = 0;
      uint32_t slotIndex[kFrames][kSlots] = {};
      uint32_t stageCount[kFrames] = {};

      // NV-DXVK [perf]: which command buffer each mark was recorded into.
      //
      // Why: two spans that record NO GPU commands at all can still read wildly
      // differently -- gpuDrain reads <0.01 ms while postComposite reads 84-115 ms,
      // and postComposite's span is provably empty (its copyBuffer is guarded by
      // bytesToCopy != 0 and [Perf.TexBudget] reports sfCount=0, so it never runs;
      // dispatchObjectPicking early-returns without a pick request). So "empty span
      // inflates" is false as a general rule and something must distinguish the two.
      // The candidate is a submit boundary: [Perf.Block] reports submits=5/frame, and
      // a timestamp pair straddling one measures the gap between two submissions
      // rather than the work between two commands.
      //
      // Handles are only ever COMPARED for equality here, never dereferenced, so it
      // does not matter that they are recycled by the time the ring is resolved.
      VkCommandBuffer markCmdBuf[kFrames][kSlots] = {};
      // NV-DXVK [perf]: source line each mark was emitted from. lastMarkLine is the
      // most recently resolved frame's, so [Perf.GpuPass] can print name=ms@line and
      // a positional misassignment becomes visible instead of silent.
      uint32_t markLine[kFrames][kSlots] = {};
      uint32_t lastMarkLine[kSlots] = {};
      // Per-stage count of resolved frames where that stage's interval crossed into
      // a different command buffer, i.e. a submit landed inside it.
      uint32_t accumSubmitSplit[kSlots] = {};

      // Accumulated over the log window.
      double   accumMs[kSlots] = {};
      // Per-stage sample count, not one global count. The marks now start before
      // prepareSceneData, so a frame that bails out of the RT branch (menu, or a
      // TLAS that is not built yet) contributes 3 marks while a full frame
      // contributes 27. Dividing every stage by a single frame count would
      // silently scale the RT stages down by the fraction of bailed frames.
      uint32_t samplesAt[kSlots] = {};
      double   accumTotalMs = 0.0;
      uint32_t samples = 0;

      // NV-DXVK [perf]: per-frame MARK COUNT, min/max over the log window.
      //
      // Why this matters: the resolve loop accumulates accumMs[i] BY POSITION and
      // kStageNames[i - 1] names it. That is only correct if every frame emits the
      // same marks in the same order. It does not: the gbuffer sub-marks go through
      // markGpuStageIfPending, which fires only when markGpuStageBeforeNextDispatch
      // armed it, and the PSR dispatches that consume it are conditional. A frame
      // that skips a MIDDLE mark shifts every later stage's name by one, silently
      // reporting one pass's time under its neighbour's label.
      //
      // (Frames that bail out of the RT branch early are NOT the problem - their 3
      // marks align with the first 3 names, which is what samplesAt handles.)
      //
      // Read it as: marks=28..28 means the table is aligned and the per-stage names
      // can be trusted. Anything less than 28, or a min != max, means names at and
      // after the gap are shifted and the attribution is suspect. 28 = t0 plus the
      // 27 kStageNames entries, since accumMs[i] is named kStageNames[i - 1].
      uint32_t marksMin = ~0u;
      uint32_t marksMax = 0;

      // GPU time between the END of the previous frame's last mark and the START
      // of this frame's first one — i.e. everything the GPU does outside the
      // instrumented region: the game's own raster, the present/blit, and any
      // driver work between submissions. [Perf.Gpu] fenceWaitMs (300-480) minus
      // [Perf.GpuPass] totalMs (155-250) left ~150 ms/frame unaccounted for and
      // this is the direct measurement of it.
      uint64_t prevLastTs      = 0;
      bool     prevLastValid   = false;
      double   accumOutsideMs  = 0.0;
      uint32_t outsideSamples  = 0;

      std::chrono::steady_clock::time_point lastLog {};
    };

    GpuStageTimers m_gpuStageTimers;

    // NV-DXVK [Perf.Sweep]: walks the sub-stage probes automatically inside one
    // run, holding each for rtx.perfAutoSweepSeconds and emitting a per-step
    // gb_primaryRays summary plus a final table.
    //
    // Exists because the alternative is ~11 separate runs, and every cross-run
    // comparison in this investigation has been weakened by something changing
    // between runs. Within a single run the camera, the scene, the streaming
    // state and the driver's compiled pipelines are all fixed by construction,
    // which is a far stronger guarantee than "the camera did not move".
    //
    // Steps are timing probes except the last, which is the raw counter census —
    // deliberately placed last because its per-pixel InterlockedAdds perturb the
    // pass timer, and putting it anywhere else would contaminate the steps after
    // it. Ignore its ms column; read [Perf.UnorderedSteps] instead.
    struct PerfSweep {
      // Index into kSteps in rtx_context.cpp. Held one step at a time.
      uint32_t step = 0;
      bool     active = false;
      bool     finished = false;

      // True only while the census step is the current one. The coverage
      // readback that prints [Perf.UnorderedSteps] checks the standalone
      // rtx.perfUnorderedStepCensus option, which the sweep does not write - it
      // overrides constants, not options - so without this flag the census step
      // would fill the counters and never print them.
      bool     censusActive = false;

      // Gameplay gate. The sweep must not start in a menu or on a loading
      // screen: those frames have no world instances, so every step would
      // measure an empty scene and the ladder would read as uniformly free.
      // Same signal the coverage readback uses - getOrderedInstances() being
      // non-empty - plus a warmup, because on the first frame it goes non-empty
      // the instance/BLAS wiring is still settling.
      //
      // The clock is FROZEN rather than reset while out of gameplay, so an
      // alt-tab or a mid-sweep load screen costs wall time but does not silently
      // eat a step or smear two steps together.
      uint32_t gameplayFrames = 0;
      bool     gameplayReady  = false;
      bool     waitingLogged  = false;
      std::chrono::steady_clock::time_point lastTick {};

      std::chrono::steady_clock::time_point stepStart {};

      // gb_primaryRays samples for the current step, gathered only after the
      // settle window so the first frames after a state change - which still
      // have in-flight work from the previous step in the timestamp ring
      // (kFrames deep) - cannot leak into the result.
      double   minMs    = 0.0;
      double   maxMs    = 0.0;
      uint32_t samples  = 0;

      // MEDIAN, not mean. The first sweep run reported four steps as SLOWER than
      // baseline - including one that removes every material texture fetch -
      // which is impossible for real work removal. The cause was single hitched
      // frames: step maxima of 282 and 320 ms against means of 141 and 137
      // dragged the averages by 10-20 ms, comfortably more than the effects
      // being measured. The minima over the same samples were already monotonic
      // and correct, which is what identified the mean as the problem.
      //
      // A ring this size holds ~7.5 s of post-settle frames at the ~3 fps this
      // scene runs at with room to spare; if it ever fills, the oldest samples
      // are kept and later ones dropped rather than wrapping, so a long step
      // degrades to "median of the first N" instead of silently mixing.
      static constexpr uint32_t kMaxStepSamples = 256;
      double   sampleMs[kMaxStepSamples] = {};

      // Sized with headroom over kPerfSweepSteps. The step table hit exactly 16
      // when the material rungs were added; a table that outgrows these arrays
      // would silently drop its last steps from the summary rather than fail.
      // Raised to 64 when the table went A-B-A: interleaving a baseline before
      // every probe roughly doubles the row count (39 today), which 32 would
      // have overrun silently.
      static constexpr uint32_t kMaxSteps = 64;
      double   resultMs[kMaxSteps] = {};       // median for the step
      double   resultMinMs[kMaxSteps] = {};
      uint32_t resultSamples[kMaxSteps] = {};

      // Latched at START from rtx.perfAutoSweepQuick. Selects the 3-step
      // baseline/probe/baseline table instead of the full one. Latched rather
      // than read per frame so toggling the option mid-run cannot swap the
      // table under the accumulated results.
      bool     quick = false;

      // What was actually uploaded for the step currently being measured.
      //
      // Recorded at the point of application rather than read back at step
      // close. Two earlier versions of this echo were both wrong: reading the
      // TABLE ROW misreports passthrough steps (the row is all zeros while the
      // config's values are what ran), and reading the OUT-PARAMS at close time
      // misreports every step (the close log runs before that frame's
      // applyStep, so the out-params still hold whatever the caller assigned
      // from rtx.conf). Only capturing at apply time is correct for both.
      uint32_t appliedUno = 0, appliedPom = 0, appliedTex = 0, appliedFilm = 0;
      uint32_t appliedCensus = 0, appliedMat = 0, appliedGb = 0, appliedGrad = 0;
      uint32_t appliedCoh = 0;
    };

    PerfSweep m_perfSweep;

    // NV-DXVK [NsysAuto]: unattended Nsight Systems capture director.
    //
    // Same gameplay gate as PerfSweep above, for the same reason and with the
    // same freeze-don't-reset behaviour: a capture has to cover the same point
    // in the run every time or two traces cannot be compared. Wall time from
    // process start will not do that - load times vary by seconds.
    //
    // This only advances a clock and prints markers. nsys start/stop is driven
    // by "Capture TF2 (Nsight Systems auto).ps1", which watches the log for
    // them, because confirming the .nsys-rep finished writing before the game is
    // closed can only be done from outside the process.
    struct NsysAutoCapture {
      enum class Phase : uint32_t {
        Settle = 0,   // counting gameplay seconds towards the trigger
        Capture,      // between CAPTURE-BEGIN and CAPTURE-END
        Drain,        // after CAPTURE-END, letting nsys flush
        Done,         // nothing further; toggling the option off re-arms
      };

      Phase    phase = Phase::Settle;
      bool     active = false;
      bool     armedLogged = false;

      uint32_t gameplayFrames = 0;
      bool     gameplayReady = false;

      double   settleSeconds = 0.0;
      double   captureSeconds = 0.0;
      double   drainSeconds = 0.0;
      double   lastHeartbeat = 0.0;
      uint32_t capturedFrames = 0;

      // NVTX range id bracketing the capture. Non-zero only between
      // CAPTURE-BEGIN and CAPTURE-END. Typed as uint64_t rather than
      // nvtxRangeId_t so this header does not have to pull in NVTX - the ids
      // are opaque handles and the underlying type is a 64-bit integer.
      uint64_t nvtxRange = 0;

      // Hash of the thread that opened the push/pop range. Compared at
      // CAPTURE-END so a capture that spans threads is reported rather than
      // silently popping the wrong stack.
      uint64_t nvtxThreadId = 0;

      std::chrono::steady_clock::time_point lastTick {};
    };

    NsysAutoCapture m_nsysAuto;

    // Advances the capture clock and emits the [NsysAuto] markers. Called once
    // per frame from the same place as updatePerfSweep.
    void updateNsysAutoCapture();

    // Advances the sweep clock and returns the overrides for the current step.
    // Called once per frame from the constants setup so the values it returns are
    // the ones actually uploaded that frame.
    void updatePerfSweep(uint32_t& outUnorderedStopAfter,
                         uint32_t& outSkipPom,
                         uint32_t& outSkipMaterialTextures,
                         uint32_t& outSkipThinFilm,
                         uint32_t& outStepCensus,
                         uint32_t& outMaterialStopAfter,
                         uint32_t& outGbStopAfter,
                         uint32_t& outCheapTextureGradients,
                         uint32_t& outCoherentUnorderedFetch);

    // Feeds one frame's resolved gb_primaryRays into the current step.
    void perfSweepAddSample(double gbPrimaryRaysMs);

    // Median of the current step's samples. See sampleMs above for why median.
    double perfSweepStepMedian() const;

  public:
    // Writes one timestamp into the current frame's ring slot. A method rather
    // than a lambda in injectRTX so dispatchPathTracing can subdivide itself —
    // [Perf.GpuPass] showed that single dispatch owning ~150 ms of a ~350 ms
    // frame while every other pass was under 1 ms, and the flags that would
    // normally isolate it (enableSecondaryBounces, integrateIndirectMode) both
    // left it unchanged. No-op before the pool exists.
    //
    // Public so DxvkPathtracerGbuffer can split its own three full-resolution
    // dispatches (Primary Rays / Reflection PSR / Transmission PSR), which are
    // issued unconditionally — enablePSRR/enablePSTR only reach the shader as
    // constants, so disabling them does not remove the dispatch.
    // NV-DXVK [perf]: 'line' defaults to the CALLER's line via __builtin_LINE(),
    // evaluated at each call site, so no call site needs editing.
    //
    // Why: kStageNames maps stages BY POSITION, and that mapping is wrong. Adding
    // a single mark at the end of the frame reshuffled every label in the table -
    // postComposite 104.8 -> 0.45 ms, rtxdi 0.64 -> 94.5, upscaler absent -> 24.3,
    // pt_gbuffer 13.1 -> 0.63 - which an end-of-frame insertion cannot legitimately
    // do. Two of the resulting readings are impossible on their face: volumetrics
    // reads 3.4 ms with rtx.volumetrics.enable=False, and upscaler reads 24.3 ms
    // with DLSS off. marks=N..N/N ALIGNED printed throughout, because that check
    // only compares mark COUNTS - and markGpuStageIfPending is designed to keep the
    // count constant when a conditional dispatch does not consume a pending mark,
    // which preserves the count while moving where the mark actually lands.
    //
    // Reporting the source line each timestamp was written from replaces the
    // positional assumption with the emission site itself.
    void markGpuStage(uint32_t line = __builtin_LINE());

    // Starts a new frame in the timestamp ring: creates the pool on first use,
    // advances the slot, resolves the oldest frame, and emits [Perf.GpuPass].
    // Split out of injectRTX so it can run BEFORE prepareSceneData — the BLAS and
    // TLAS builds live in there and were outside the old t0, which is one of the
    // two places the ~150 ms of unattributed GPU time can be hiding.
    void beginGpuStageFrame();

    // Requests a timestamp written AFTER the next dispatch's pre-dispatch barrier
    // flush and BEFORE the dispatch itself (see DxvkContext::dispatch).
    //
    // Why this is needed: markGpuStage writes at BOTTOM_OF_PIPE, so the interval
    // that ends at the primary-ray dispatch also contains whatever that dispatch
    // had to WAIT for — DXVK emits the accumulated execution barriers inside
    // dispatch(), immediately before vkCmdDispatch. A 130 ms bucket that is
    // invariant to every shading option is exactly what a pipeline drain waiting
    // on the acceleration-structure build looks like, and no CPU-side counter can
    // tell the two apart. This mark splits them.
    void markGpuStageBeforeNextDispatch();

    // Emits the pending pre-dispatch mark if the dispatch never consumed it —
    // commitComputeState/commitRaytracingState can return false when a pipeline
    // is not ready yet, which would skip the mark and shift every [Perf.GpuPass]
    // stage name after it by one for that frame. Stage names are positional, so a
    // conditional mark has to be made unconditional somewhere.
    void markGpuStageIfPending();

  private:

    // This enum is for internal use only.
    // There is a mode called UpscalerType in RtxOptions, but it doesn't contain DLSS-RR because RR is considered as a special mode of DLSS.
    enum class InternalUpscaler {
      None = 0,
      DLSS,
      NIS,
      TAAU,
      XeSS,
      DLSS_RR,
    };

    void reportCpuSimdSupport();

    void takeScreenshot(std::string imageName, Rc<DxvkImage> image);

    // NV-DXVK [OnScreenAlbedoDump]: one-shot dump of the albedo texture of
    // every on-screen instance, ~10s after gameplay starts, to a separate
    // folder (NOT the full capture). Cheap early-out until it fires once.
    void dumpOnScreenAlbedosOnce();

    void checkOpacityMicromapSupport();
    void checkShaderExecutionReorderingSupport();
    void checkNeuralRadianceCacheSupport();

    VkExtent3D setDownscaleExtent(const VkExtent3D& upscaleExtent);

    VkExtent3D onFrameBegin(const VkExtent3D& upscaleExtent);

    void dispatchVolumetrics(const Resources::RaytracingOutput& rtOutput);
    void dispatchIntegrate(const Resources::RaytracingOutput& rtOutput);
    void dispatchPathTracing(const Resources::RaytracingOutput& rtOutput);
    void dispatchDemodulate(const Resources::RaytracingOutput& rtOutput);
    void dispatchNeeCache(const Resources::RaytracingOutput& rtOutput);
    void dispatchDLSS(const Resources::RaytracingOutput& rtOutput);
    void dispatchRayReconstruction(const Resources::RaytracingOutput& rtOutput);
    void dispatchDenoise(const Resources::RaytracingOutput& rtOutput);
    // NV-DXVK [MtnRadiance]: diagnostic readback of primary-hit albedo/direct/indirect
    // for DISTANT-hit pixels (|linearViewZ| > threshold), to localise why the far
    // 3D-skybox mountain tops shade black — albedo≈0 (material), direct≈0 (shadow-ray
    // precision at ~6.5e6 units), or indirect≈0 (NRC scene-bounds). Gated on
    // logSurfaceCoverage + throttled (steady-state artifact, no need for every frame).
    void captureMountainRadianceProbe(const Resources::RaytracingOutput& rtOutput);
    // NV-DXVK [MtnComposite]: companion to the above, called AFTER dispatchComposite so
    // m_compositeOutput holds THIS frame's resolved on-screen radiance (emissive + lighting
    // + sky). Reads that + the per-pixel surfaceIndex on the same coarse grid and attributes
    // each pixel to a VS, so we can see which emissive backdrop VS renders BLACK on screen
    // (the radiance probe is blind to emissive — it only reads direct/indirect/albedo).
    void captureMountainCompositeProbe(const Resources::RaytracingOutput& rtOutput);
    // NV-DXVK [TonemapProbe]: read the tonemap input (m_compositeOutput, HDR) and
    // the tonemap output (m_finalOutput, post-operator display [0,1]) at a sparse
    // pixel grid and log the in->out pairs, so the tonemap operator's effect is
    // directly observable (not just which operator is selected). Call right after
    // the tonemapping dispatch, before bloom/post-fx touch m_finalOutput.
    void captureTonemapProbe(const Resources::RaytracingOutput& rtOutput);
    // NV-DXVK: measure the average scene radiance (mean luminance of the composite output)
    // and feed it to NRC for dynamicMaxExpectedRadiance. Throttled + EMA-smoothed; async
    // readback so it never stalls the frame. Called after dispatchComposite.
    void updateNrcDynamicRadiance(const Resources::RaytracingOutput& rtOutput);
    void dispatchComposite(const Resources::RaytracingOutput& rtOutput);
    void dispatchReplaceCompositeWithDebugView(const Resources::RaytracingOutput& rtOutput);
    void dispatchNIS(const Resources::RaytracingOutput& rtOutput);
    void dispatchXeSS(const Resources::RaytracingOutput& rtOutput);
    void dispatchTemporalAA(const Resources::RaytracingOutput& rtOutput);
    void dispatchToneMapping(const Resources::RaytracingOutput& rtOutput, bool performSRGBConversion);
    void dispatchBloom(const Resources::RaytracingOutput& rtOutput);
    void dispatchDepthOfField(const Resources::RaytracingOutput& rtOutput);
    void dispatchPostFx(Resources::RaytracingOutput& rtOutput);
    void dispatchDebugView(Rc<DxvkImage>& srcImage, const Resources::RaytracingOutput& rtOutput, bool captureScreenImage);
    void dispatchObjectPicking(Resources::RaytracingOutput& rtOutput, const VkExtent3D& srcExtent, const VkExtent3D& targetExtent);
    void dispatchDLFG();
    void updateMetrics(const float gpuIdleTimeMilliseconds) const;

    void rasterizeToSkyMatte(const DrawParameters& params, const DrawCallState& drawCallState);
    void initSkyProbe();
    void rasterizeToSkyProbe(const DrawParameters& params, const DrawCallState& drawCallState);
    void rasterizeSky(const DrawParameters& params, const DrawCallState& drawCallState);
    enum class TryHandleSkyResult {
      Default,
      SkipSubmit,
    };
    TryHandleSkyResult tryHandleSky(const DrawParameters* originalParams, DrawCallState* originalDrawCallState /* can be std::move-d */);

    void bakeTerrain(const DrawParameters& params, DrawCallState& drawCallState, const MaterialData** outOverrideMaterialData);

    InternalUpscaler getCurrentFrameUpscaler();

    InternalUpscaler m_currentUpscaler = InternalUpscaler::None;
    InternalUpscaler m_previousUpscaler = InternalUpscaler::None;

    uint32_t m_frameLastInjected = kInvalidFrameIndex;
    bool m_skipBackbufferBlitThisFrame = false;

    // NV-DXVK [HUD-Option5 v4]: persistent scratch that saves POST-tonemap
    // m_finalOutput each frame. Read next frame by the D3D11Rtx-side
    // blit lambda that queues between TF2's composite draw and its HUD
    // rasters. One-frame RT lag is inherent here.
    Rc<DxvkImage>      m_rtFinalPostTonemapScratch;
    VkExtent3D         m_rtFinalPostTonemapScratchExtent = { 0, 0, 0 };
    VkFormat           m_rtFinalPostTonemapScratchFormat = VK_FORMAT_UNDEFINED;
    bool               m_rtFinalPostTonemapScratchHasContent = false;
    bool               m_compositeInterceptedThisFrame = false;
    bool m_captureStateForRTX = true;

    Rc<DxvkImage> m_skyProbeImage;
    Rc<DxvkImageView> m_skyProbeCubePlanes[6];
    // [SkyTrace.probePrefill] Per-face single-layer STORAGE views. The
    // cube-prefill compute pass dispatches once per face with the matching
    // view bound at compute slot 1. We tried a single 2D-array view + one
    // dispatch with z=6 first; SPIR-V looks correct (OpTypeImage 2D
    // Arrayed, OpImageWrite at uint3 coord) but only layer 0 receives the
    // writes in practice — likely a DXVK/driver path that doesn't fan
    // multi-layer storage writes correctly through this Slang→SPIR-V
    // emit. Six single-layer dispatches sidestep the issue.
    Rc<DxvkImageView> m_skyProbeCubePlaneStorageViews[6];
    VkFormat m_skyColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    VkFormat m_skyRtColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    VkClearValue m_skyClearValue;
    bool m_skyClearDirty = false;
    // NV-DXVK [SkyProbe.cubeRender]: set true after rasterizeToSkyProbe
    // successfully renders TF2's sky into all 6 cube faces. The path
    // tracer's IBL sample sites consult this (via cb.skyProbePopulated)
    // to decide between sampling the populated cubemap (TF2's painted
    // sky in reflections) or the Hillaire LUT fallback.
    bool m_skyProbeCubemapPopulated = false;
    SkyMode m_lastSkyMode = SkyMode::SkyboxRasterization;

    std::unique_ptr<RtxAtmosphere> m_atmosphere;

  public:
    // NV-DXVK [AerialPerspective]: external accessor so passes outside
    // RtxContext (e.g. composite) can query the atmosphere LUT for
    // distance-based haze application.
    RtxAtmosphere* getAtmosphere() const { return m_atmosphere.get(); }
  private:

    bool shouldUseDLSS() const;
    bool shouldUseRayReconstruction() const;
    bool shouldUseNIS() const;
    bool shouldUseTAA() const;
    bool shouldUseXeSS() const;
    bool shouldUseUpscaler() const { return shouldUseDLSS() || shouldUseNIS() || shouldUseTAA() || shouldUseXeSS(); }

    inline static bool s_triggerScreenshot = false;
    inline static bool s_triggerUsdCapture = false;
    // [SpawnGeomDiag.FloorObjDump] one-shot flag; consumed by
    // AccelManager BBI-readback to dump the floor PI BLAS to disk
    // as paired local/world OBJ files. Atomic because the consumer
    // reads from the render thread.
    inline static std::atomic<bool> s_triggerObjDump{false};
    inline static const bool s_capturePrePresentTestScreenshot = env::getEnvVar("RTX_TAKE_PRE_PRESENT_SCREENSHOT_FRAME") != "";

    bool m_rayTracingSupported;
    bool m_dlssSupported;
    bool m_submitContainsInjectRtx = false;
    uint64_t m_cachedReflexFrameId = 0;

    bool m_resetHistory = true;    // Discards use of temporal data in passes

    std::chrono::time_point<std::chrono::steady_clock> m_prevRunningTime;
    uint64_t m_prevGpuIdleTicks;

    bool m_screenshotFrameEnabled = false;
    bool m_triggerDelayedTerminate = false;
    uint32_t m_screenshotFrameNum = -1;
    uint32_t m_terminateAppFrameNum = -1;
    uint32_t m_framesWithoutValidScene = 0;
    IntegrateIndirectMode m_prevIntegrateIndirectMode = IntegrateIndirectMode::Count;

    DxvkRaytracingInstanceState m_rtState;

    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_objectPickingReadback {};

    // NV-DXVK: per-frame log of unique surface indices visible at primary hit.
    // Gated by env var RTX_LOG_VISIBLE_SURFACES=1. Reads m_sharedSurfaceIndex.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
      bool                            envChecked = false;
      bool                            enabled = false;
    } m_visibleSurfacesReadback {};
    void recordVisibleSurfacesReadback(const Resources::RaytracingOutput& rtOutput);

    // NV-DXVK [Atmosphere.lut readback]: per-frame async copy of a slab
    // of the AerialPerspective 3D LUT to a HOST_VISIBLE buffer, decoded
    // and logged on a worker thread. Diagnoses why aerial perspective
    // produced a uniform cyan/blue ghost over geometry — strength=1
    // washed everything; we want to see actual LUT inscatter/transmittance
    // values to know if it's the LUT data or the consumer math.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_aerialPerspectiveLutReadback {};
    void recordAerialPerspectiveLutReadback();

    // [SkyTrace.matteContent]: per-frame async readback of the sky-matte
    // image at the moment composite is about to sample it. Confirms the
    // GPU-visible pixel content (post any TF2 raster, post our injectRTX
    // white-clear, post any aliasing) is what we think it is. Decodes
    // R8G8B8A8_SRGB and B10G11R11_UFLOAT_PACK32; logs min/avg/max RGB
    // plus a center texel.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_skyMatteReadback {};

    // [SkyTrace.primaryMiss] Counts how many pixels in a center tile of
    // PrimaryLinearViewZ equal the miss-sentinel. Tells us whether visible-
    // sky pixels are actually being classified as primaryMiss in composite.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_primaryMissCountReadback {};

    // [SkyTrace.probeContent] readback of all 6 SkyProbe cube faces (a
    // center tile from each). The cube is what IBL/PSR samples for world
    // surfaces in Hybrid mode when skyProbePopulated==1, so its content
    // is the actual color tinting world geometry.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_skyProbeReadback {};

    // [SkyTrace.skyVerts] async readback of a sky draw's position buffer.
    // Decodes TF2's 21/21/22-bit packed position format CPU-side, applies
    // objectToWorld to get world-space coords, computes AABB. Used to test
    // whether TF2's sky meshes span all 6 cube-face directions or cluster
    // on a few — answers the "is the cube architecturally fillable?" Q.
    struct {
      std::atomic<uint64_t>           signalValue = 1;
      Rc<sync::Fence>                 signal = new sync::Fence{};
      std::vector<std::future<void>>  asyncTasks = {};
    } m_skyVertsReadback {};
  public:
    // Public so dispatchComposite (RtxComposite::dispatch in rtx_composite.cpp)
    // can fire the readbacks at the moment SkyLight is bound for sampling.
    void recordSkyMatteReadback();
    void recordPrimaryMissCountReadback(const Resources::RaytracingOutput& rtOutput,
                                        float missLinearViewZ);
    void recordSkyProbeReadback();
    void recordSkyDrawPositionsReadback(const DrawCallState& drawCallState,
                                        uint32_t frameId, uint32_t drawIdx);
  private:

    std::vector<DrawCallState> m_delayedRayTracedSky;

#ifdef REMIX_DEVELOPMENT
    void queryAvailableResourceAliasing();
    void clearResourceAliasingCache();
    void analyzeResourceAliasing();

    struct ResourceCache {
      Rc<DxvkImageView> view;
      RtxFramePassStage beginPassStage = RtxFramePassStage::FrameBegin;
      RtxFramePassStage endPassStage = RtxFramePassStage::FrameEnd;
      std::unordered_set<std::string> names;
    };

    // We only have 5 types of format categories and we won't expect this will exceed 10 in near future. So we hard code the category to 10 types for better performance and easier development.
    std::vector<ResourceCache> m_resourceCacheTable[static_cast<uint32_t>(RtxTextureFormatCompatibilityCategory::Count)];
    std::unordered_map<const DxvkImageView*, std::string> m_viewMap;

    RtxFramePassStage m_currentPassStage = RtxFramePassStage::FrameBegin;
#endif
  };
} // namespace dxvk