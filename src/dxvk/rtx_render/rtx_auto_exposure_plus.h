/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "dxvk_format.h"
#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_resources.h"
#include "rtx_render/rtx_mipmap.h"
#include "rtx_render/rtx_dlfg.h"

#include "../spirv/spirv_code_buffer.h"
#include "../util/util_matrix.h"
#include "rtx_options.h"

namespace dxvk {

  class DxvkDevice;

  // The "Plus" half of rtx.autoExposure.mode.
  //
  // Base mode is Remix's stock histogram auto exposure: one scalar for the whole frame, which
  // has to compromise between a bright sky and a dark interior. Plus keeps that scalar and adds
  // a spatially varying correction on top of it, so both can be exposed correctly at once.
  //
  // This is an exposure pass only. Linear HDR radiance in, linear HDR radiance out, multiplied
  // by a spatially varying gain. It performs no display transform of its own, so the tonemapper
  // downstream - Remix's native curve, Hable, Psycho17 or GT7 - runs unchanged and simply
  // receives a better exposed image. The one exception is the *local* tonemapper, which this
  // pass overrides rather than stacks with; see forcesGlobalTonemapper() below for why.
  //
  // It runs before bloom rather than after. Bloom is a large radius smear of bright regions, so
  // measuring a bloomed image would inflate the key around every highlight and make the
  // operator darken a wider area than the highlight actually covers - a dark halo, which is the
  // exact artifact the edge-aware pyramid exists to avoid. Running first also means bloom
  // responds to the corrected exposure, which is the right order for a lens effect.
  //
  // Structurally it is a relative of DxvkLocalToneMapping - both fuse a pyramid - but the
  // pyramid here is built with an edge-aware downsample and collapsed with an edge-stopping
  // selector rather than summed as Laplacians off Gaussian mips, which is what lets it run at
  // much higher strength without haloing.
  class DxvkAutoExposurePlus : public RtxPass {
  public:
    explicit DxvkAutoExposurePlus(DxvkDevice* device);
    ~DxvkAutoExposurePlus();

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      const float frameTimeMilliseconds,
      bool resetHistory = false,
      bool enableAutoExposure = true);

    void showImguiSettings();

    // True when this pass should push the display transform onto the global tonemapper.
    //
    // Plus and the local tonemapper are both local dynamic range compressors; in series they
    // flatten the image twice. Rather than rewriting rtx.tonemappingMode - which would silently
    // edit a setting the user chose, and would have to be guessed back on the way out - this
    // overrides which tonemapper runs for as long as Plus is active, exactly the way selecting
    // a tonemap operator already forces the global path in RtxContext::dispatchToneMapping.
    // The stored value of rtx.tonemappingMode is never touched.
    //
    // Note that the override has to carry the local tonemapper's S-curve across with it. The
    // local path applies ACES by default (rtx.localtonemap.finalizeWithACES) while the global
    // path's native curve does not (rtx.tonemap.finalizeWithACES), and that curve is near
    // straight in log-log - it redistributes luminance but adds no toe and no shoulder. Swapping
    // paths without it trades a filmic image for a flat one, which reads as this pass washing
    // the image out when in fact this pass never ran on it. dispatchToneMapping forces the ACES
    // finalize for the native curve only; the other operators bring their own shoulder.
    static bool forcesGlobalTonemapper();

  private:

    virtual bool isEnabled() const override;

    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) override;
    virtual void releaseTargetResource() override;

    // Number of pyramid levels for a given full resolution. The pyramid stops once the coarsest
    // level is roughly 8 texels tall rather than continuing down to 1x1, because the last few
    // levels carry no usable spatial information but still cost a full pair of dispatches each.
    static uint32_t calcLevelCount(const VkExtent3D& targetExtent);

    // The pyramid runs at half the output resolution. The kernel is 23 taps wide, so this
    // halves the cost of every level without measurably changing the gain field, which is
    // reconstructed by the guided filter at full resolution anyway.
    static VkExtent3D calcPyramidExtent(const VkExtent3D& targetExtent);

    // Level k of the pyramid is mip k. Carries the exposure key in .y and the accumulating gain
    // in .x.
    RtxMipmap::Resource m_pyramid;
    // Scratch for the horizontal half of each separable downsample. Level k holds an
    // intermediate of (width(k) / 2) x height(k), which fits inside mip k.
    RtxMipmap::Resource m_pyramidTmp;
    // Collapsed gain field plus the guide signal, at pyramid resolution, before accumulation.
    Resources::Resource m_fused;
    // Ping-ponged accumulation targets for the temporal pass.
    Resources::Resource m_fusedAccum[2];

    uint32_t m_levelCount = 0;
    uint32_t m_accumIndex = 0;
    bool m_hasHistory = false;

    // GPU timing.
    //
    // Reuses DxvkDLFGTimestampQueryPool rather than adding a second pool implementation - it
    // already does reset-then-write per slot and a non-blocking read, and is what the existing
    // [Perf.GpuPass] instrumentation in RtxContext uses. Timestamps are written at the stage
    // boundaries and resolved kFrames later so the GPU has certainly passed them; a slot that
    // is not ready is dropped rather than stalling the frame.
    struct StageTimers {
      // init, downsample, fuse, temporal, apply -> 5 intervals, so 6 marks.
      static constexpr uint32_t kMarks = 6;
      static constexpr uint32_t kStages = kMarks - 1;
      static constexpr uint32_t kFrames = 4;

      Rc<DxvkDLFGTimestampQueryPool> pool;

      uint32_t slotIndex[kFrames][kMarks] = {};
      uint32_t markCount[kFrames] = {};
      uint32_t frameSlot = 0;
      uint32_t writeCount = 0;
      // Frames where logging was off wrote no timestamps, so the ring holds entries from
      // whenever it was last on. Resolving those would report the gap rather than a frame.
      bool wasLogging = false;

      double accumMs[kStages] = {};
      double accumTotalMs = 0.0;
      uint64_t samples = 0;
      uint32_t framesSinceReport = 0;
    };

    StageTimers m_timers;

    void beginTimerFrame(Rc<RtxContext> ctx);
    void markTimer(Rc<RtxContext> ctx);
    void reportTimers(const VkExtent3D& pyramidExtent, uint32_t levelCount);

    RTX_OPTION("rtx.autoExposurePlus", float, exposure, 1.0f,
               "Exposure factor the operator assumes is applied downstream, used only to locate middle grey while measuring.\n"
               "The default of 1.0 is correct unless the tonemapper in use applies an exposure of its own (rtx.tonemap.exposureBias, rtx.localtonemap.exposure); a mismatch shifts where the operator thinks middle grey is, which rtx.autoExposurePlus.targetEV can absorb.");
    // This is not a scene independent number. The gain is a fixed fraction of the key
    // (gain = -key * intensity * 0.5), so the absolute correction scales with however many stops
    // the key spans, and linear scene radiance floored at 1e-4 spans roughly 15 - a good deal
    // more than a display referred signal would. The default is picked to be gentle at that
    // range; a scene with less range than that can take more intensity, and one with more needs
    // less. Treat it as the starting point for a sweep.
    RTX_OPTION("rtx.autoExposurePlus", float, intensity, 0.35f,
               "Strength of the local exposure correction, in the range [-1, 1].\n"
               "Positive values lift dark areas and pull down bright ones, flattening the image. Negative values do the opposite. Zero disables the local correction, leaving targetEV as a plain global exposure offset.\n"
               "The correction is proportional to how far a region sits from the frame average, so the value that looks right depends on the scene's dynamic range. Raise it for more equalization; if the image looks washed out, it is too high for the range the scene covers.");
    RTX_OPTION("rtx.autoExposurePlus", float, equalizationStrength, 0.5f,
               "How readily the pyramid collapse switches from a coarse scale to a finer one, in the range [0, 1].\n"
               "Higher values let smaller features carry their own exposure, flattening the image further. Extreme values produce a deep fried look.");
    RTX_OPTION("rtx.autoExposurePlus", float, equalizationPivot, 1.0f,
               "Log luminance, in stops above middle grey, past which the equalization weight starts being measured relatively rather than absolutely.\n"
               "Raising it equalizes bright regions more aggressively; lowering it leaves them alone sooner. A value of 1.0 means one stop over middle grey, which is a reasonable anchor for midtones but sits low against linear HDR radiance, where a sky can reach eight or more stops over - so this is worth sweeping per game.");
    RTX_OPTION("rtx.autoExposurePlus", float, targetEV, 0.0f,
               "Global brightness offset in stops, applied on top of the local correction. Positive values brighten the whole image.\n"
               "Independent of intensity, so it still works as a plain exposure offset with the local correction turned off.");
    RTX_OPTION("rtx.autoExposurePlus", float, maxGainEV, 6.0f,
               "Clamp on the per-pixel exposure correction, in stops.\n"
               "Guards the guided-filter regression against unbounded HDR input; at the default it should not engage on well behaved content.");
    RTX_OPTION("rtx.autoExposurePlus", int, levels, 0,
               "Number of pyramid levels, or 0 to derive it from the render resolution.\n"
               "Fewer levels confine the operator to smaller features and cost less.");
    RTX_OPTION("rtx.autoExposurePlus", bool, forceGlobalTonemapper, true,
               "Routes the display transform through the global tonemapper while this pass is active, overriding rtx.tonemappingMode without modifying it.\n"
               "The local tonemapper is itself a local dynamic range compressor, so running it after this pass compresses the image twice and washes it out. Disable only to deliberately stack the two.\n"
               "While this is active the global tonemapper's native curve also gets an ACES finalize it would not otherwise have, so that the S-curve the local tonemapper applies by default is not lost in the swap. Selected operators (Hable, Psycho17, GT7) supply their own and are left alone.");

    // Temporal accumulation
    RTX_OPTION("rtx.autoExposurePlus", bool, temporalAccumulation, true,
               "Accumulates the gain field across frames with reprojection.\n"
               "The operator applies the most gain in dark regions, which is exactly where a path traced image retains the most residual denoiser noise, so without this the exposure visibly shimmers. Disable only to see what it is doing.");
    RTX_OPTION("rtx.autoExposurePlus", bool, matchBaseAdaptationSpeed, true,
               "Drives the local correction at the same speed as the base auto exposure (rtx.autoExposure.autoExposureSpeed) so the two settle together.\n"
               "With this off the local correction uses rtx.autoExposurePlus.adaptationSpeed instead, which lets the local and global halves visibly adapt at different rates.");
    RTX_OPTION("rtx.autoExposurePlus", float, adaptationSpeed, 5.0f,
               "Local correction adaptation speed in units per second, used when matchBaseAdaptationSpeed is disabled.");

    // Diagnostics
    RTX_OPTION("rtx.autoExposurePlus", int, debugPyramidLevel, 0,
               "Pyramid level shown by the 'Auto Exposure Plus Pyramid Level' debug view.");
    RTX_OPTION("rtx.autoExposurePlus", bool, perfLogging, false,
               "Writes per-stage GPU timings for this pass to the log under the [Perf.AutoExpPlus] tag.");
    RTX_OPTION("rtx.autoExposurePlus", int, perfLogIntervalFrames, 600,
               "How many frames to average over before emitting a [Perf.AutoExpPlus] line.");
  };

}
