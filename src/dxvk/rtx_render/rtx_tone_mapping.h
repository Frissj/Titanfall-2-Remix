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

    // NV-DXVK [PSDT v0.3]: true when PSDT is the selected operator and is the
    // configured owner of local dynamic range compression.
    //
    // Read by DxvkAutoExposurePlus::isEnabled(), which is why it is static and
    // public: the answer is a pure function of two rtx options, so it needs no
    // object and creates no dependency on the tonemapper's state. Deciding it
    // through isEnabled() rather than at the dispatch call site is what makes
    // the suppression complete - RtxPass turns isEnabled() into isActive() once
    // per frame and releases the pass's resources on the transition, so Plus
    // gives back its pyramid rather than merely skipping work with it
    // allocated.
    static bool psdtOwnsLocalAdaptation();

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
    //
    // Three fields rather than v0.1's two, because the source energy became
    // per-channel (so glare carries the source's own colour) and the
    // illuminant became a measured quantity of its own rather than a mean
    // chroma riding along with it. Together they are about 780 KiB at 1080p.
    RtxMipmap::Resource m_psdtField[2];
    RtxMipmap::Resource m_psdtSource[2];
    RtxMipmap::Resource m_psdtIlluminant[2];
    // Body-only luminance histogram. Accumulated by psdt_analysis with atomics
    // and zeroed by psdt_state once it has been read, the same way the
    // tonemapper's own histogram is consumed by its tone curve pass.
    Resources::Resource m_psdtBodyHistogram;
    Resources::Resource m_psdtState;

    VkExtent3D m_psdtFieldExtent = { 0, 0, 0 };
    uint32_t m_psdtLevelCount = 0;
    uint32_t m_psdtIndex = 0;
    bool m_psdtHasHistory = false;

    // Camera state, for the transition classifier in psdt_state. Kept here
    // rather than read from RtCamera's own history because the tonemapper runs
    // once per frame and only needs the delta across that one step.
    Vector3 m_psdtPreviousCameraForward = Vector3(0.0f, 0.0f, 1.0f);
    bool m_psdtHasCameraHistory = false;

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

    // NV-DXVK [PSDT v0.3]: which pass performs local dynamic range compression.
    //
    // Exactly one of them should, and until v0.3 two of them did. Auto Exposure
    // Plus builds a pyramid, accumulates it temporally and multiplies the HDR
    // buffer by a spatially varying gain; PSDT builds a pyramid, accumulates it
    // temporally and moves the curve's anchor per neighbourhood. Running both
    // compresses the same signal twice, and v0.2's answer - scale PSDT down by
    // 1 - |plus.intensity| - divided the job between two implementations of it
    // rather than choosing one.
    //
    // The default is Psdt, which suppresses the Plus pass entirely rather than
    // weakening it: DxvkAutoExposurePlus::isEnabled() consults
    // psdtOwnsLocalAdaptation(), so the pass deactivates, releases its
    // resources, and its dispatches stop being submitted. Plus is untouched for
    // every other operator, which is the only reason it can be suppressed here
    // without being deleted there.
    enum class LocalAdaptationOwner : uint32_t {
      // PSDT owns it. The Plus pass does not run while this operator is
      // selected, and PSDT spends the whole local adaptation budget.
      Psdt = 0,
      // Plus owns it. PSDT's local strength and contrast budget go to zero, so
      // its curve is anchored globally and the spatial part of the job is done
      // upstream in linear radiance. Useful as a control: it isolates what
      // PSDT's own local adaptation is contributing.
      AutoExposurePlus,
      // Both, with the budget split by 1 - |plus.intensity|. This is exactly
      // v0.2's behaviour, kept so the change is measurable rather than merely
      // asserted.
      Both,
    };

    // NV-DXVK [PSDT]: in-image diagnostics. Every entry shows a value the
    // transform actually read or produced on that pixel on its way past,
    // rather than recomputing its own version of it - a debug view that
    // re-derives the quantity agrees with the shader right up until the moment
    // something is wrong, which is the only moment it was needed.
    enum class PsdtDebugView : uint32_t {
      Off = 0,
      Adaptation,     // Pooled adaptation, as stops about the frame anchor.
      Budget,         // Adaptation left after the contrast budget clamped it.
      Roles,          // BODY green, SOURCE red, SKY blue.
      SourceEnergy,   // The glare kernel's input, per channel.
      Illuminant,     // Measured local illuminant.
      Depth,          // Body depth in red; in green, how much of the coarse
                      // scales survived the depth AND luminance agreement
                      // tests. A dark frame here is one pooling on the
                      // finest scale alone.
      Demand,         // Colour volume demand. Green in gamut, red out of it.
      Pressure,       // Chroma actually given up.
      Glare,          // Glare contribution alone.
      Clipping,       // Signed pre-clamp excess: red over peak, blue below zero.
      CurveSlope,     // d(display)/d(scene) at this pixel's base.
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
      // Two more controls, added for the same reason. Reinhard is the
      // no-shoulder no-colour-management baseline; AgX is the canonical
      // controlled path to white, which is the interesting comparison for
      // PSDT's hue trajectory rather than the easy one. Numbered above PSDT so
      // the gaps at 1, 2, 4 and 5 stay reserved for the RemixProjGroup fork.
      Reinhard = 9,
      AgX = 10,
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
               "Tonemap operator (set by number in rtx.conf). 0 = None (Remix native dynamic curve), 3 = HableFilmic (Uncharted 2, closest to Titanfall's native look), 6 = Psycho17 (perceptual vision-model, best bright/saturated handling; default), 7 = GT7 (Gran Turismo 7, SDR ICtCp chroma-preserving), 8 = PerceptualTF2 (PSDT - scene-adaptive perceptual display transform; see the rtx.tonemap.psdt.* group), 9 = Reinhard (extended, white point 4.0), 10 = AgX (reference configuration).\n"
               "9 and 10 exist as scientific controls for PSDT rather than as recommendations: Reinhard has no shoulder and no colour management at all, and AgX is the canonical controlled path to white. Everything PSDT claims is measured against those two and against 6 and 7 in tools/psdt/.");

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
    RTX_OPTION("rtx.tonemap.psdt", bool, rendererSignals, true,
               "Classifies pixels using what the renderer knows - the per-pixel emissive bit, linear view depth and demodulated albedo - rather than from brightness alone.\n"
               "This is the difference between 'source' meaning 'an emitter' and 'source' meaning 'bright'. With it off, a sunlit white wall meters as a lamp, sky cannot be told from a bright ceiling, and the white point falls back to a grey-world estimate over the scene's mean colour rather than an estimate of the light itself. Disable only to reproduce v0.1 behaviour or if the gbuffer is unavailable.");
    RTX_OPTION("rtx.tonemap.psdt", float, sourceThreshold, 4.0f,
               "Stops above middle grey at which a pixel starts being classified as a light source rather than a lit surface.\n"
               "Relative rather than absolute, which only works because the input is post-exposure. With auto exposure disabled this becomes an absolute threshold and wants re-tuning.\n"
               "With rendererSignals on this is one input among several rather than the whole classifier: an emissive surface qualifies 2.5 stops sooner because the emissive bit is direct evidence, and a non-emissive one additionally has to be spatially compact, which is what keeps a large bright surface out of the source population.");
    RTX_OPTION("rtx.tonemap.psdt", float, skyViewZ, 100000.0f,
               "Linear view depth above which a pixel is sky rather than geometry.\n"
               "A ray miss writes NRD's miss depth (500001) and far backdrop geometry is clamped to 200000, while interior geometry is orders of magnitude below both, so anything between about 10000 and 150000 separates them identically. Sky is neither surface nor lamp: it gets a low fixed adaptation weight, contributes no glare energy unless it is bright enough to be the sun, and is what the outdoor scene-intent term is measured from.");
    RTX_OPTION("rtx.tonemap.psdt", float, illuminantMinAlbedo, 0.04f,
               "Albedo below which a surface contributes nothing to the local illuminant estimate.\n"
               "The estimate is radiance divided by albedo, which is the colour of the light rather than the colour of the paint - but only where the division means something. A near-black surface divides noise by noise and a fully saturated one has no information in its dark channels; both are weighted out rather than hard-tested, so the estimate degrades to 'no opinion' instead of flickering.");
    RTX_OPTION("rtx.tonemap.psdt", float, depthSensitivity, 0.35f,
               "How strongly the adaptation pooling refuses to average across a depth discontinuity. 0 disables it and restores a plain box pyramid.\n"
               "Depth is carried through the pyramid as log2(1+z), so this is measured in octaves of distance and one value works from a corridor to a skybox: two walls at 5 and 10 units are 0.9 apart and still pool together, while the same wall and a 200000-unit backdrop are 15 apart and do not. It only ever reduces the coarse scales' weight, so the worst case is that pooling collapses to the finest scale, which is a real adaptation neighbourhood in its own right.\n"
               "This is what stops a bright window from re-exposing the wall beside it.");
    RTX_OPTION("rtx.tonemap.psdt", LocalAdaptationOwner, localAdaptationOwner, LocalAdaptationOwner::Psdt,
               "Which pass performs local dynamic range compression. 0 = PSDT, 1 = Auto Exposure Plus, 2 = both.\n"
               "Exactly one of them should, and until v0.3 two of them did. Plus builds a pyramid over the HDR buffer, accumulates it temporally with reprojection and multiplies by a spatially varying gain; PSDT builds a pyramid over the same buffer, accumulates it temporally with reprojection and moves the curve's anchor per neighbourhood. Two implementations of one job, coordinated by a scalar - which is not a coordination.\n"
               "At the default PSDT owns it and the Plus pass does not run at all while this operator is selected: it deactivates, releases its pyramid, and stops submitting its dispatches (16 of them at 1080p, one a full-resolution read-modify-write of the HDR buffer). Plus is completely unaffected for every other operator, which is what makes suppressing it here reasonable. What Plus knew and PSDT did not - the edge-stopping pyramid collapse and the edge-aware downsample - moved into scaleCoherence and pyramidCoherence below, so nothing it was contributing is lost.\n"
               "1 is the control that isolates PSDT's own local adaptation by handing the spatial half of the job back to Plus. 2 reproduces v0.2's stacking exactly, so the change is measurable rather than merely asserted.");
    RTX_OPTION("rtx.tonemap.psdt", float, scaleCoherence, 1.20f,
               "How strongly the adaptation pooling refuses to average across a luminance boundary between scales. 0 disables it and restores fixed pooling weights.\n"
               "Depth already stops the pooling from averaging a wall together with the sky behind it. This stops it averaging a wall together with the shadow across the middle of it - one surface, one distance, two lighting conditions, which depth cannot see and which is where local adaptation matters most. Measured past a one-stop dead zone, so ordinary shading costs nothing, with a squared falloff after it.\n"
               "This is Auto Exposure Plus's edge-stopping pyramid collapse, moved here. It only ever reduces a coarse scale's weight, so the worst case is that pooling falls back to the finest scale.\n"
               "Transported as 8 bits of [0, 4]; the push constant block is exactly full.");
    RTX_OPTION("rtx.tonemap.psdt", float, pyramidCoherence, 1.00f,
               "Width, in stops, of the robust reweight in the pyramid's body reduction. 0 restores a plain weight-normalised mean.\n"
               "The other half of what came across from Auto Exposure Plus. Four children that are two dark walls and two sunlit ones average to a value describing neither, and every coarser level is then built out of values describing a scene that is not there. One Welsch step about the weighted mean makes a level report the population most of it belongs to instead; the bright minority is left to the finer level it came from, which still exists and is still pooled.\n"
               "Lower is more selective. Below about a stop it starts rejecting ordinary shading, at which point the coarse levels stop describing anything wider than the fine ones do.");
    RTX_OPTION("rtx.tonemap.psdt", float, sourceConfidence, 1.50f,
               "Octaves by which this frame's source energy may disagree with the reprojected history before the block is only half believed. 0 disables the test.\n"
               "The only place in the transform that knows its input is path traced. A firefly the denoiser missed, a speculative fireball and a specular hit on something that moved are all a hundredfold luminance spike in a few pixels for one frame, and none of them is distinguishable from a muzzle flash in a framebuffer - but they are distinguishable from where the same block was a frame ago. Disagreement widens the history window the block has to escape, so a source that persists is believed and one that appeared out of nothing is admitted at a fraction of its energy until it repeats itself.\n"
               "Only the appearing direction is slowed. A light going out is instant, because that is what a light going out is.\n"
               "Lower is stricter. It is observable through debug view 4, which is the glare kernel's actual input rather than a re-derivation of it.");
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
               "The curve compresses a base layer; this decides how much of the detail riding on that base keeps its original contrast rather than the curve's compressed slope. It is what separates 'compressed dynamic range' from 'flat'. It is tapered away in the highlights by detailProtect and rolled off with detail magnitude by detailKnee, because re-expanding contrast the shoulder correctly removed is exactly how haloing appears around bright sources.");
    RTX_OPTION("rtx.tonemap.psdt", float, detailProtect, 1.0f,
               "How hard highlights are protected from local contrast restoration. 0 restores contrast just as strongly in the shoulder as in the midtones.\n"
               "Scales how quickly restoration fades as the neighbourhood moves up the shoulder. The shoulder spent that contrast deliberately, on the grounds that the display cannot show it; putting it back is how a bright rim appears around every light in the frame.");
    RTX_OPTION("rtx.tonemap.psdt", float, detailKnee, 3.0f,
               "Detail magnitude, in stops away from the neighbourhood's own level, at which restoration has fallen to half.\n"
               "Texture sits a fraction of a stop from its surroundings and gets the full restoration. A lamp sits six stops from the wall behind it and gets almost none - it is not detail, it is a different object, and restoring the contrast between them is a halo rather than a texture. Lower values are more conservative.");

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
               "How far the observer is adapted towards the measured local illuminant rather than D65, in [0,1]. 0 is a strict D65 convergence.\n"
               "Keeps a tungsten-lit interior warm as it desaturates instead of drifting towards daylight white, which is most of what makes an indoor-to-outdoor transition read correctly.\n"
               "This is a von Kries chromatic adaptation in CAT02 cone space, not an offset applied to the chroma coordinates. The distinction is not academic: under an adaptation, a surface that is neutral under the local light lands exactly on the perceptual neutral axis, so compressing its chroma to zero takes it to the local white and a neutral is still guaranteed to fit inside the display volume. Under an offset - which is what v0.1 did - neither of those holds.\n"
               "The illuminant itself is measured as radiance over albedo across surfaces where that division is meaningful, so a room of coloured walls under a neutral light reports a neutral illuminant. The effective strength is additionally scaled by how much of the frame produced a usable estimate, so a scene with no evidence adapts to D65 rather than to noise.");
    RTX_OPTION("rtx.tonemap.psdt", float, pressureLuma, 0.50f,
               "How early a colour that is bright AND saturated begins easing towards the gamut boundary, relative to an ordinary colour, in [0,1]. 0 waits until the colour genuinely does not fit.\n"
               "Anticipatory compression: a transform does not have to wait for the boundary to arrive before it starts approaching it gracefully. Only the product of brightness and chroma moves this, so a bright neutral is never affected.");
    RTX_OPTION("rtx.tonemap.psdt", float, pressureContext, 0.6f,
               "How much a colour's contrast against its own neighbourhood brings that easing forward. 0 ignores spatial context.\n"
               "The fourth term in the compression decision, after luminance, chroma and gamut distance. A bright outlier against a dark surround - a lamp against a wall, a muzzle flash in a corridor - is somewhere the eye is not looking for colour fidelity and where a hard boundary clip is very visible, so it earns an earlier and gentler approach. A colour sitting in a large field of its own brightness is the opposite case: a hue shift there is obvious and there is no clipping artifact to trade against, so it keeps its colour longer.");
    RTX_OPTION("rtx.tonemap.psdt", float, whiteConvergence, 2.0f,
               "Exponent shaping how quickly the hue trajectory engages as a colour leaves the display volume. Higher keeps ordinary colours untouched for longer and concentrates the rotation into the colours that genuinely cannot fit.");
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
               "Per-level weight ratio of the glare kernel's wide lobe, in (0,1). Higher spreads the veil wider.\n"
               "The radius itself is not a parameter and cannot be: the visibility threshold is applied per pyramid level, so a dim source drops below it after one or two levels while a very bright one survives several more, and the halo grows with roughly the log of the source luminance on its own. v0.1 thresholded the summed energy once instead, which gave every source the same profile scaled by brightness - a bloom, not a glare.");
    RTX_OPTION("rtx.tonemap.psdt", float, glareNearField, 0.8f,
               "Amplitude of the glare kernel's narrow lobe, relative to the wide one, in [0,1].\n"
               "The eye's glare point-spread function is a narrow core sitting on a much wider, much fainter veil, and a single exponential cannot be both. This is the core, and it is also the near field a highlight needs between its clipped centre and its halo - the part that says a light source has an extent rather than just a position.");

    // --- diagnostics -----------------------------------------------------
    RTX_OPTION("rtx.tonemap.psdt", int, levels, 0,
               "Adaptation pyramid levels, or 0 to derive from the render resolution. Fewer levels confine both the adaptation pooling and the glare kernel to smaller radii.");
    RTX_OPTION("rtx.tonemap.psdt", PsdtDebugView, debugView, PsdtDebugView::Off,
               "Replaces the image with one of the transform's own intermediate values. 0 = Off, 1 = Adaptation, 2 = Budget, 3 = Roles, 4 = Source energy, 5 = Illuminant, 6 = Depth, 7 = Gamut demand, 8 = Pressure, 9 = Glare, 10 = Clipping, 11 = Curve slope.\n"
               "Every view shows a value the transform actually read or produced on that pixel on its way past, rather than recomputing its own version of it. A debug view that re-derives the quantity agrees with the shader right up until the moment something is wrong, which is the only moment it was needed.\n"
               "Roles is the one to look at first: it is the classification the whole architecture rests on, in green (surface), red (source) and blue (sky). Depth is the second: its green channel is how much of the coarse adaptation scales survived the depth and luminance agreement tests, so a frame that is dark there is one pooling on the finest scale alone - which is the way the v0.3 pooling terms fail. Clipping is the third, because it shows what the final gamut fit had to hide.");

    // Dithering settings
    RTX_OPTION("rtx.tonemap", DitherMode, ditherMode, DitherMode::SpatialTemporal,
               "Tonemap dither mode selection, dithering allows for reduction of banding artifacts in the final rendered output from quantization using a small amount of monochromatic noise. Impact typically most visible in darker regions with smooth lighting gradients.\n"
               "Enabling dithering will make the rendered image slightly noisier, though usually dither noise is fairly imperceptible in most cases without looking closely. Generally dithered results will also look better than the alternative of banding artifacts due to increasing perceptual precision of the signal.\n"
               "Note that temporal dithering may increase perceptual precision further but may also introduce more noticeable noise in the final output in some cases due to the noise pattern changing every frame unlike a purely spatial approach.\n"
               "Supported enum values are 0 = None (Disabled), 1 = Spatial (Enabled, Spatial dithering only), 2 = SpatialTemporal (Enabled, Spatial and temporal dithering).\n"
               "Generally enabling dithering is recommended, but disabling it may be useful in some niche cases for improving compression ratios in images or videos at the cost of quality (as noise while it may not be very visible may be more difficult to compress), or for capturing \"raw\" post-tonemapped data from the renderer.");
  };
  
}
