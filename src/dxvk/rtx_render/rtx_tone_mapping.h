/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "../spirv/spirv_code_buffer.h"
#include "../util/util_matrix.h"
#include "rtx_options.h"

namespace dxvk {

  class DxvkDevice;

  class DxvkToneMapping: public CommonDeviceObject {
  public:
    explicit DxvkToneMapping(DxvkDevice* device);
    ~DxvkToneMapping();

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      const float frameTimeMilliseconds,
      bool performSRGBConversion = true,
      bool resetHistory = false,
      bool autoExposureEnabled = true,
      bool forceFinalizeWithACES = false);

    bool isEnabled() const { return tonemappingEnabled(); }

    void showImguiSettings();

    // NV-DXVK [tonemap operators]: draws the operator dropdown. Static because it
    // only touches the rtx.tonemap.tonemapOperator option, and it has to be drawn
    // from the Tonemapping section in dxvk_imgui.cpp *outside* the Global/Local
    // branch - selecting an operator overrides tonemappingMode, so the dropdown
    // must stay reachable even while the mode is set to Local.
    static void showOperatorImguiSetting();

  private:
    void createResources(Rc<RtxContext> ctx);

    void dispatchHistogram(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::Resource& colorBuffer,
      bool autoExposureEnabled);

    void dispatchToneCurve(
      Rc<RtxContext> ctx);

    void dispatchApplyToneMapping(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      Rc<DxvkImageView> exposureView,
      const Resources::Resource& inputBuffer,
      const Resources::Resource& colorBuffer,
      bool performSRGBConversion,
      bool autoExposureEnabled,
      bool forceFinalizeWithACES);

    // NV-DXVK [PSDT]: the Perceptual Scene Display Transform's own analysis
    // chain. It lives inside the global tonemapper rather than in a pass of its
    // own for one substantive reason: it has to read the luminance histogram
    // *between* dispatchHistogram and dispatchToneCurve, because the tone curve
    // pass zeroes the histogram when it is finished with it. Everything else
    // follows from being here - the operator selection already forces the
    // global path, the input colour buffer and exposure texture are already in
    // hand, and no new pass registration or dispatch ordering is introduced.
    void ensurePsdtResources(Rc<RtxContext> ctx, const VkExtent3D& targetExtent);
    void createPsdtResources(Rc<RtxContext> ctx, const VkExtent3D& targetExtent);
    void releasePsdtResources();

    void dispatchPsdt(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      const Resources::Resource& colorBuffer,
      const float frameTimeMilliseconds,
      bool autoExposureEnabled);

    // Adaptation field resolution: one texel per 16x16 source pixels. Coarse on
    // purpose - see PSDT_FIELD_BLOCK_SIZE.
    static VkExtent3D calcPsdtFieldExtent(const VkExtent3D& targetExtent);
    // Pyramid depth, capped both by PSDT_MAX_LEVELS and by what the field
    // extent can actually support (createImageResource asserts on the latter).
    static uint32_t calcPsdtLevelCount(const VkExtent3D& fieldExtent);

    bool psdtSelected() const;

    Rc<vk::DeviceFn> m_vkd;

    Resources::Resource m_toneHistogram;
    Resources::Resource m_toneCurve;

    // Ping-ponged: the analysis pass reprojects the previous frame's mip 0 with
    // motion vectors while writing this frame's, so the two cannot be the same
    // image - a workgroup would otherwise sample a texel another workgroup in
    // the same dispatch had already overwritten.
    RtxMipmap::Resource m_psdtField[2];
    RtxMipmap::Resource m_psdtChroma[2];
    Resources::Resource m_psdtState;

    VkExtent3D m_psdtFieldExtent = { 0, 0, 0 };
    uint32_t m_psdtLevelCount = 0;
    uint32_t m_psdtIndex = 0;
    bool m_psdtHasHistory = false;

    bool m_resetState = true;
    bool m_isCurveChanged = true;

    enum class ExposureAverageMode : uint32_t {
      Mean = 0,
      Median
    };

    enum class DitherMode : uint32_t {
      None = 0,
      Spatial,
      SpatialTemporal,
    };

    // NV-DXVK [PSDT]: the gamut the transform fits colours into. Only meaningful
    // if the output chain is actually that wide - on the default sRGB path
    // anything but Rec709 produces values the swapchain will clip.
    enum class PsdtGamut : uint32_t {
      Rec709 = 0,
      DisplayP3,
      Rec2020,
    };

    // NV-DXVK [PSDT]: which perceptual space the colour-volume stage works in.
    // Deliberately swappable - the choice is an experiment, not a conclusion.
    enum class PsdtSpace : uint32_t {
      ICtCp = 0,
      Jzazbz,
    };

    // NV-DXVK [tonemap operators]: selects a fork-ported tonemap operator that
    // replaces the native dynamic (histogram tone-curve) tonemapper. Numeric
    // values match the tonemapOperator* constants in
    // shaders/rtx/pass/tonemap/tonemapping.h and the RemixProjGroup fork.
    // Public so the dispatch in rtx_context.cpp can force the global path when an
    // operator is selected (the default tonemappingMode is Local, which would
    // otherwise bypass the operator entirely).
  public:
    enum class TonemapOperator : uint32_t {
      None        = 0,  // Native Remix dynamic tone curve.
      HableFilmic = 3,  // Uncharted 2 filmic.
      Psycho17    = 6,  // renodx Psycho Test 17 (perceptual / vision-model).
      GT7         = 7,  // Gran Turismo 7 (SDR, ICtCp chroma-preserving).
      // NV-DXVK [PSDT]: not a curve but a scene-adaptive display transform,
      // with its own analysis passes and its own settings group under
      // rtx.tonemap.psdt. The others stay in the list as controls to measure it
      // against - that is worth more than the couple of kilobytes they cost.
      PerceptualTF2 = 8,
    };

    RTX_OPTION("rtx.tonemap", float, exposureBias, 0.f, "The exposure value to use for the global tonemapper when auto exposure is disabled, or a bias multiplier on top of the auto exposure's calculated exposure value.");
    RTX_OPTION("rtx.tonemap", bool, tonemappingEnabled, true, "A flag to enable or disable the local tonemapper. Note this flag will only take effect when the global tonemapper is set to be used (as opposed to another option such as the local tonemapper).");
    RTX_OPTION("rtx.tonemap", bool, colorGradingEnabled, false, "A flag to enable or disable color grading after the global tonemapper's tonemapping pass, but before gamma correction and dithering (if enabled).");

    // Color grading settings
    RTX_OPTION("rtx.tonemap", Vector3, colorBalance, Vector3(1.0f, 1.0f, 1.0f), "The color tint to apply after tonemapping when color grading is enabled for the tonemapper (rtx.tonemap.colorGradingEnabled). Values should be in the range [0, 1].");
    RTX_OPTION("rtx.tonemap", float, contrast, 1.0f, "The contrast adjustment to apply after tonemapping when color grading is enabled for the tonemapper (rtx.tonemap.colorGradingEnabled). Values should be in the range [0, 1].");
    RTX_OPTION("rtx.tonemap", float, saturation, 1.0f, "The saturation adjustment to apply after tonemapping when color grading is enabled for the tonemapper (rtx.tonemap.colorGradingEnabled). Values should be in the range [0, 1].");

    // Tone curve settings
    // Important that the min/max here do now under/overflow the dyamic range of input, or visual errors will be noticeable
    RTX_OPTION("rtx.tonemap", float, toneCurveMinStops, -24.0f, "Low endpoint of the tone curve (in log2(linear)).");
    RTX_OPTION("rtx.tonemap", float, toneCurveMaxStops, 8.0f, "High endpoint of the tone curve (in log2(linear))."); 
    RTX_OPTION("rtx.tonemap", bool,  tuningMode, false, "A flag to enable a debug visualization to tune the tonemapping exposure curve with, as well as exposing parameters for tuning the tonemapping in the UI.");
    RTX_OPTION("rtx.tonemap", bool,  finalizeWithACES, false, "A flag to enable applying a final pass of ACES tonemapping to the tonemapped result.");
    RTX_OPTION("rtx.tonemap", float, dynamicRange, 15.f, "Range [0, inf). Without further adjustments, the tone curve will try to fit the entire luminance of the scene into the range [-dynamicRange, 0] in linear photographic stops. Higher values adjust for ambient monitor lighting; perfect conditions -> 17.587 stops.");
    RTX_OPTION("rtx.tonemap", float, shadowMinSlope, 0.f, "Range [0, inf). Forces the tone curve below a linear value of 0.18 to have at least this slope, making the tone darker.");
    RTX_OPTION("rtx.tonemap", float, shadowContrast, 0.f, "Range [0, inf). Additional gamma power to apply to the tone of the tone curve below shadowContrastEnd.");
    RTX_OPTION("rtx.tonemap", float, shadowContrastEnd, 0.f, "Range (-inf, 0]. High endpoint for the shadow contrast effect in linear stops; values above this are unaffected.");
    RTX_OPTION("rtx.tonemap", float, curveShift, 0.0f, "Range [0, inf). Amount by which to shift the tone curve up or down. Nonzero values will cause additional clipping.");
    RTX_OPTION("rtx.tonemap", float, maxExposureIncrease, 5.f, "Range [0, inf). Forces the tone curve to not increase luminance values at any point more than this value.");

    // NV-DXVK [tonemap operators]: choose the tonemap operator by NUMBER (enums
    // parse as their integer value in rtx.conf). None (0) keeps Remix's native
    // dynamic tone curve. Others replace it: 3 = Hable/Uncharted2 (closest to
    // Titanfall's own filmic look), 6 = Psycho17 (perceptual, best saturated-
    // highlight handling — default), 7 = GT7 (Gran Turismo 7, ICtCp chroma-
    // preserving). Set in rtx.conf, e.g. "rtx.tonemap.operator = 7" for GT7.
    RTX_OPTION("rtx.tonemap", TonemapOperator, tonemapOperator, TonemapOperator::Psycho17,
               "Tonemap operator (set by number in rtx.conf). 0 = None (Remix native dynamic curve), 3 = HableFilmic (Uncharted 2, closest to Titanfall's native look), 6 = Psycho17 (perceptual vision-model, best bright/saturated handling; default), 7 = GT7 (Gran Turismo 7, SDR ICtCp chroma-preserving), 8 = PerceptualTF2 (PSDT - scene-adaptive perceptual display transform; see the rtx.tonemap.psdt.* group).");

    // =====================================================================
    // PSDT - Perceptual Scene Display Transform (operator 8)
    // =====================================================================
    // Only read when rtx.tonemap.operator is 8. The defaults are meant to be a
    // usable starting point on an ordinary sRGB monitor in a dim room, not a
    // neutral identity - PSDT has no identity configuration, because a display
    // transform that does nothing is not a display transform.

    // --- display model ---------------------------------------------------
    // The transform is scene-referred throughout and only becomes
    // display-referred here, which is what lets the same transform target an
    // sRGB panel and an HDR one without being rewritten.
    RTX_OPTION("rtx.tonemap.psdt", float, displayPeakNits, 100.0f,
               "Peak luminance of the target display, in cd/m^2.\n"
               "Leave equal to displayRefWhiteNits for SDR - that is the exact, intended path, where the transform's output lands in [0,1] with no normalisation. Setting it higher declares headroom above diffuse white and PSDT will grade for it, but this pipeline ends in an 8-bit sRGB encode, so what you will see is an SDR-normalised preview of that grade rather than the grade itself.");
    RTX_OPTION("rtx.tonemap.psdt", float, displayRefWhiteNits, 100.0f,
               "Luminance of diffuse (paper) white on the target display, in cd/m^2. A framebuffer value of 1.0 means this.\n"
               "It is the reference the perceptual space is absolute against, so it changes how ICtCp/Jzazbz interpret every value - not just how bright the image is.");
    RTX_OPTION("rtx.tonemap.psdt", float, displayBlackNits, 0.05f,
               "Black level of the target display, in cd/m^2. Roughly 0.05 for a good LCD, 0.0005 for OLED.\n"
               "This is the asymptote the curve's toe approaches, so it sets where shadow detail stops being separable. Lowering it lengthens the linear shadow region and separates dark detail further; it is clamped internally so that a value of 0 does not remove the toe entirely.");
    RTX_OPTION("rtx.tonemap.psdt", float, surroundNits, 5.0f,
               "Luminance of the viewing environment, in cd/m^2. A dark room is ~1, a dim room ~5, an office ~50.\n"
               "Shifts the display-referred mid anchor: a bright surround needs the mid lifted or shadow detail disappears into the light reflecting off the screen.");
    RTX_OPTION("rtx.tonemap.psdt", PsdtGamut, displayGamut, PsdtGamut::Rec709,
               "Gamut the colour-volume stage fits colours into. 0 = Rec.709/sRGB, 1 = Display-P3, 2 = Rec.2020.\n"
               "Only set this wider than Rec709 if the output chain genuinely is wider. On the default sRGB swapchain a wider setting tells PSDT that colours which cannot be shown are fine, and they are then clipped by the swapchain instead of being mapped by the transform - which is the exact failure the colour-volume stage exists to avoid.");
    RTX_OPTION("rtx.tonemap.psdt", PsdtSpace, perceptualSpace, PsdtSpace::ICtCp,
               "Perceptual space for the colour-volume stage. 0 = ICtCp (BT.2100; cheap, good HDR intensity/chroma separation), 1 = Jzazbz (better hue linearity over a wider luminance range, two extra matrix multiplies).\n"
               "Swapping this changes nothing else in the transform - the space is behind an abstraction on purpose, so it stays an experiment.");

    // --- adaptation ------------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", bool, sceneAdaptive, true,
               "Lets the frame's measured statistics modulate the resolved curve, adaptation and pooling parameters (scene intent: night / indoor / outdoor / bright).\n"
               "Disable to hold every parameter at exactly what the settings below say, which is what you want while sweeping one of them - with this on, a parameter sweep is measuring the parameter and the scene classifier at once.");
    RTX_OPTION("rtx.tonemap.psdt", float, sourceExclusion, 0.75f,
               "How completely light sources are excluded from the adaptation anchor, in [0,1].\n"
               "At 0 the anchor is the plain (centre-weighted) histogram median and a sunlit window will drag the whole corridor's exposure down with it. At 1 the anchor ignores anything classified as a source entirely and exposes purely for surfaces. This is the single setting that most changes how a dark interior with a bright opening reads.");
    RTX_OPTION("rtx.tonemap.psdt", float, sourceThreshold, 4.0f,
               "Stops above middle grey at which a pixel starts being classified as a light source rather than a lit surface.\n"
               "Relative rather than absolute, which only works because the input is post-exposure. With auto exposure disabled this becomes an absolute threshold and wants re-tuning.");
    RTX_OPTION("rtx.tonemap.psdt", float, glareClassThreshold, 8.0f,
               "Stops above middle grey at which a pixel is too bright to be represented as a value at all, and its energy is handed to the glare model as spatial extent instead.");
    RTX_OPTION("rtx.tonemap.psdt", float, localAdaptation, 0.65f,
               "How much of the adaptation comes from the local field rather than the frame anchor, in [0,1].\n"
               "This is not a local exposure multiply - it moves where the curve thinks middle grey is for a neighbourhood, so a dark region gains shadow separation rather than just gain. It is bounded by contrastBudget below, which is what stops it flattening the image.");
    RTX_OPTION("rtx.tonemap.psdt", float, contrastBudget, 1.0f,
               "Scales how far local adaptation may move away from the frame anchor, in stops. At the default: shadows 2.0, midtones 0.55, highlights 0.25, emitters 0.05.\n"
               "The most important constraint in the system. Local adaptation with no budget converges on 'every region is middle grey', which improves visibility while destroying the brightness hierarchy - and that hierarchy carries as much information as the geometry does. Raise it for visibility, lower it for punch; a value of 0 disables local adaptation entirely.");
    RTX_OPTION("rtx.tonemap.psdt", float, adaptationSpeedUp, 3.0f,
               "How fast the adaptation anchor rises when the scene gets brighter, in units per second.\n"
               "Faster than the downward speed by default because that is the direction human light adaptation actually is: stepping into daylight settles in seconds, stepping into a dark room takes much longer.");
    RTX_OPTION("rtx.tonemap.psdt", float, adaptationSpeedDown, 1.2f,
               "How fast the adaptation anchor falls when the scene gets darker, in units per second. Slower than upward by design; it is also what makes an interior read as dark when you first step into it.");
    RTX_OPTION("rtx.tonemap.psdt", bool, temporalAccumulation, true,
               "Accumulates the adaptation field across frames with motion-vector reprojection.\n"
               "The field decides where the curve's anchor sits and the input is a denoised path-traced image whose residual noise is worst in exactly the dark regions that get the most adaptation. Disable only to see what the field is doing on its own.");
    RTX_OPTION("rtx.tonemap.psdt", float, fieldAdaptationSpeed, 8.0f,
               "Adaptation field accumulation speed, in units per second. Higher tracks the scene more closely and shimmers more.");

    // --- curve -----------------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", float, midtoneContrast, 1.15f,
               "Slope of the curve's linear segment, in display stops per scene stop.\n"
               "1.0 is a straight line in log-log, which redistributes luminance and adds no contrast at all - this is exactly why Remix's native dynamic curve renders flat on its own. Values slightly above 1 give the midtones their punch; the compression is done by the toe and shoulder, not by flattening the middle.");
    RTX_OPTION("rtx.tonemap.psdt", float, shadowDepth, 0.55f,
               "Fraction of the display range below the mid anchor that the linear segment may use before the toe takes over, in [0,1].\n"
               "Higher pushes the toe lower, lengthening the linear shadow region and separating dark detail. Lower brings the toe up and crushes towards the black floor sooner.");
    RTX_OPTION("rtx.tonemap.psdt", float, highlightRolloff, 0.45f,
               "Fraction of the display range above the mid anchor that the linear segment may use before the shoulder starts, in [0,1].\n"
               "Lower starts the shoulder earlier, which makes it longer and softer and hands more of the highlight range to the colour-volume and glare stages. Higher keeps the midtones linear further up and compresses harder when it finally rolls off.\n"
               "The default is measured rather than chosen: tools/psdt/psdt_sweep.py shows 0.45 dominating 0.60 on every axis it trades against - more highlight separation (0.19 vs 0.16 display stops per scene stop over +2..+5), more chroma retained, six points less clipping - while still keeping the full midtone slope. Below about 0.40 the shoulder starts eating into the +-1 stop midtone region and punch begins to drop.");
    RTX_OPTION("rtx.tonemap.psdt", float, detailStrength, 0.55f,
               "How much of the scene's local contrast is restored after luminance compression, in [0,1].\n"
               "The curve compresses a base layer; this decides how much of the detail riding on that base keeps its original contrast rather than the curve's compressed slope. It is what separates 'compressed dynamic range' from 'flat'. It is tapered away in the highlights and rolled off with detail magnitude, because re-expanding contrast the shoulder correctly removed is exactly how haloing appears around bright sources.");

    // --- colour volume ---------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", float, chromaPreservation, 2.5f,
               "How hard colour resists being compressed as the display runs out of colour volume. Higher preserves more.\n"
               "Compression is proportional to luminance x chroma x gamut pressure, not to luminance alone - a bright white wall keeps its (near zero) chroma untouched while a neon red at the same luminance is the thing that has to give. That product is why raising this does not simply oversaturate the whole image.");
    RTX_OPTION("rtx.tonemap.psdt", float, hueTrajectory, 0.60f,
               "How far a colour is allowed to travel along the film-like path to white as it runs out of display volume, in [0,1].\n"
               "At 0 colours desaturate with their hue held exactly, which is safe and reads as plastic. At 1 a bright red follows the same arc a per-channel curve would take it on - red, orange, yellow-white - without the rest of the image being handed to a per-channel curve. The rotation is along the shortest arc and scaled by pressure, so ordinary colours are untouched and there is no discontinuity at the hue wrap.");
    RTX_OPTION("rtx.tonemap.psdt", float, luminanceConcession, 0.8f,
               "How much brightness a saturated colour may give up in order to keep its colour, in [0,1].\n"
               "A saturated red cannot be as bright as a white - the most saturated Rec.709 red has a luminance of 0.21 - so when the tone stage asks for a brightness the hue cannot support, something has to give. At 0 the colour gives, and a bright red bleaches towards white. At 1 the brightness gives, and the red stays red but renders darker than a white at the same scene luminance. Film does the latter, which is most of why film highlights read as coloured light rather than as blown paper.\n"
               "Only saturated colours are affected at all: a neutral is never outside the volume, so it concedes nothing and the brightness of the rest of the frame is untouched.\n"
               "At the default a fully saturated colour renders about 0.28 stops below a neutral of the same scene luminance, and retains 0.44 of its scene chroma against 0.35 with the concession off. 1.0 buys another 0.03 of chroma for 0.09 more stops of darkening.");
    RTX_OPTION("rtx.tonemap.psdt", float, spatialWhite, 0.35f,
               "How much the local low-frequency mean colour, rather than D65, is used as the white that chroma compresses towards, in [0,1].\n"
               "Keeps a tungsten-lit interior warm as it desaturates instead of drifting towards daylight white, which is most of what makes an indoor-to-outdoor transition read correctly. Set to 0 for a strict D65 convergence.");
    RTX_OPTION("rtx.tonemap.psdt", float, colourfulness, 1.0f,
               "Couples perceived colourfulness to local adaptation.\n"
               "A region the viewer is dark-adapted to is shown at less absolute display luminance than the scene had and needs proportionally more chroma to look equally colourful. This is the coupling between the tone stage and the colour stage that treating the two as independent throws away. 0 disables it.");

    // --- glare -----------------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", float, glareStrength, 0.5f,
               "Strength of the perceptual glare model. 0 disables it.\n"
               "Not bloom - bloom is a lens artifact applied to the image, glare is how an intensity that cannot be shown as a value gets shown as an area instead. It is deliberately separate from rtx.bloom.*, and both can run.");
    RTX_OPTION("rtx.tonemap.psdt", float, glareThreshold, 5.0f,
               "Stops above the adaptation anchor at which a source starts to glare.\n"
               "Anchored to the adaptation state rather than to an absolute value, which is the whole reason a car headlight is blinding at night and invisible at noon.");
    RTX_OPTION("rtx.tonemap.psdt", float, glareFalloff, 0.65f,
               "Per-level weight ratio of the glare kernel, in (0,1). Higher spreads glare wider.\n"
               "The radius itself is not a parameter and cannot be: a dim source drops below the visibility threshold after one or two pyramid levels while a very bright one survives several more, so the halo grows with roughly the log of the source luminance on its own.");

    // --- diagnostics -----------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", int, levels, 0,
               "Adaptation pyramid levels, or 0 to derive from the render resolution. Fewer levels confine both the adaptation pooling and the glare kernel to smaller radii.");

    // Dithering settings
    RTX_OPTION("rtx.tonemap", DitherMode, ditherMode, DitherMode::SpatialTemporal,
               "Tonemap dither mode selection, dithering allows for reduction of banding artifacts in the final rendered output from quantization using a small amount of monochromatic noise. Impact typically most visible in darker regions with smooth lighting gradients.\n"
               "Enabling dithering will make the rendered image slightly noisier, though usually dither noise is fairly imperceptible in most cases without looking closely. Generally dithered results will also look better than the alternative of banding artifacts due to increasing perceptual precision of the signal.\n"
               "Note that temporal dithering may increase perceptual precision further but may also introduce more noticeable noise in the final output in some cases due to the noise pattern changing every frame unlike a purely spatial approach.\n"
               "Supported enum values are 0 = None (Disabled), 1 = Spatial (Enabled, Spatial dithering only), 2 = SpatialTemporal (Enabled, Spatial and temporal dithering).\n"
               "Generally enabling dithering is recommended, but disabling it may be useful in some niche cases for improving compression ratios in images or videos at the cost of quality (as noise while it may not be very visible may be more difficult to compress), or for capturing \"raw\" post-tonemapped data from the renderer.");
  };
  
}
