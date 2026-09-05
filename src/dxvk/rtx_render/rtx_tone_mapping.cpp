/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_tone_mapping.h"
#include "dxvk_device.h"

#include <algorithm>
#include <cmath>
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx/pass/psdt/psdt.h"
#include "rtx_camera.h"
#include "rtx_scene_manager.h"
#include "rtx_auto_exposure_plus.h"

#include <rtx_shaders/auto_exposure.h>
#include <rtx_shaders/auto_exposure_histogram.h>
#include <rtx_shaders/tonemapping_histogram.h>
#include <rtx_shaders/tonemapping_tone_curve.h>
#include <rtx_shaders/tonemapping_apply_tonemapping.h>
#include <rtx_shaders/psdt_analysis.h>
#include <rtx_shaders/psdt_downsample.h>
#include <rtx_shaders/psdt_state.h>
#include "rtx_imgui.h"
#include "rtx/utility/debug_view_indices.h"

static_assert((TONEMAPPING_TONE_CURVE_SAMPLE_COUNT & 1) == 0, "The shader expects a sample count that is a multiple of 2.");

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class HistogramShader : public ManagedShader
    {
      SHADER_SOURCE(HistogramShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_histogram)

      PUSH_CONSTANTS(ToneMappingHistogramArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D(TONEMAPPING_HISTOGRAM_HISTOGRAM_INPUT_OUTPUT)
        RW_TEXTURE2D_READONLY(TONEMAPPING_HISTOGRAM_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_HISTOGRAM_EXPOSURE_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(HistogramShader);

    class ToneCurveShader : public ManagedShader
    {
      SHADER_SOURCE(ToneCurveShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_tone_curve)

      PUSH_CONSTANTS(ToneMappingCurveArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D(TONEMAPPING_TONE_CURVE_HISTOGRAM_INPUT_OUTPUT)
        RW_TEXTURE1D(TONEMAPPING_TONE_CURVE_TONE_CURVE_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ToneCurveShader);

    class ApplyTonemappingShader : public ManagedShader
    {
      SHADER_SOURCE(ApplyTonemappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_apply_tonemapping)

      PUSH_CONSTANTS(ToneMappingApplyToneMappingArgs)

      BEGIN_PARAMETER()
        TEXTURE2DARRAY(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)
        SAMPLER1D(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)
        SAMPLER2D(TONEMAPPING_APPLY_PSDT_FIELD_INPUT)
        SAMPLER2D(TONEMAPPING_APPLY_PSDT_SOURCE_INPUT)
        SAMPLER2D(TONEMAPPING_APPLY_PSDT_ILLUM_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_PSDT_STATE_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ApplyTonemappingShader);

    // NV-DXVK [PSDT]: scene analysis, adaptation pyramid, adaptation state.
    class PsdtAnalysisShader : public ManagedShader
    {
      SHADER_SOURCE(PsdtAnalysisShader, VK_SHADER_STAGE_COMPUTE_BIT, psdt_analysis)

      PUSH_CONSTANTS(PsdtAnalysisArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(PSDT_ANALYSIS_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(PSDT_ANALYSIS_EXPOSURE_INPUT)
        TEXTURE2D(PSDT_ANALYSIS_MOTION_INPUT)
        TEXTURE2D(PSDT_ANALYSIS_SURFACE_FLAGS_INPUT)
        TEXTURE2D(PSDT_ANALYSIS_VIEW_Z_INPUT)
        TEXTURE2D(PSDT_ANALYSIS_ALBEDO_INPUT)
        RW_TEXTURE2D(PSDT_ANALYSIS_FIELD_OUTPUT)
        SAMPLER2D(PSDT_ANALYSIS_FIELD_HISTORY_INPUT)
        RW_TEXTURE2D(PSDT_ANALYSIS_SOURCE_OUTPUT)
        SAMPLER2D(PSDT_ANALYSIS_SOURCE_HISTORY_INPUT)
        RW_TEXTURE2D(PSDT_ANALYSIS_ILLUM_OUTPUT)
        SAMPLER2D(PSDT_ANALYSIS_ILLUM_HISTORY_INPUT)
        RW_TEXTURE1D(PSDT_ANALYSIS_BODY_HISTOGRAM_OUTPUT)
        RW_TEXTURE1D_READONLY(PSDT_ANALYSIS_STATE_INPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PsdtAnalysisShader);

    class PsdtDownsampleShader : public ManagedShader
    {
      SHADER_SOURCE(PsdtDownsampleShader, VK_SHADER_STAGE_COMPUTE_BIT, psdt_downsample)

      PUSH_CONSTANTS(PsdtDownsampleArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(PSDT_DOWNSAMPLE_FIELD_INPUT)
        RW_TEXTURE2D(PSDT_DOWNSAMPLE_FIELD_OUTPUT)
        TEXTURE2D(PSDT_DOWNSAMPLE_SOURCE_INPUT)
        RW_TEXTURE2D(PSDT_DOWNSAMPLE_SOURCE_OUTPUT)
        TEXTURE2D(PSDT_DOWNSAMPLE_ILLUM_INPUT)
        RW_TEXTURE2D(PSDT_DOWNSAMPLE_ILLUM_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PsdtDownsampleShader);

    class PsdtStateShader : public ManagedShader
    {
      SHADER_SOURCE(PsdtStateShader, VK_SHADER_STAGE_COMPUTE_BIT, psdt_state)

      PUSH_CONSTANTS(PsdtStateArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D_READONLY(PSDT_STATE_BINDING_HISTOGRAM)
        RW_TEXTURE1D(PSDT_STATE_BINDING_BODY_HISTOGRAM)
        SAMPLER2D(PSDT_STATE_BINDING_FIELD)
        SAMPLER2D(PSDT_STATE_BINDING_SOURCE)
        SAMPLER2D(PSDT_STATE_BINDING_ILLUM)
        RW_TEXTURE1D(PSDT_STATE_BINDING_STATE)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PsdtStateShader);
  }

  // NV-DXVK [tonemap operators]: dropdown for rtx.tonemap.tonemapOperator. Uses
  // ComboWithKey rather than a plain Combo because the enum values are sparse
  // (0/3/6/7/8/9/10 - the gaps at 1, 2, 4 and 5 are reserved by the upstream
  // fork's numbering), so the list index is not the option value.
  RemixGui::ComboWithKey<DxvkToneMapping::TonemapOperator> tonemapOperatorCombo {
    "Tonemap Operator",
    RemixGui::ComboWithKey<DxvkToneMapping::TonemapOperator>::ComboEntries { {
      { DxvkToneMapping::TonemapOperator::None, "None (Remix native)",
        "Remix's native dynamic tone curve, fitted per-frame from the luminance histogram.\n"
        "This is the only setting that leaves the Tonemapping Mode selection in charge: pick it if you\n"
        "want the Local (Laplacian-pyramid) tonemapper, which every other operator bypasses." },
      { DxvkToneMapping::TonemapOperator::HableFilmic, "Hable Filmic (Uncharted 2)",
        "Uncharted 2 filmic curve, using the Half-Life: Alyx parameters (exposure bias 2.0, W 4.0).\n"
        "Closest match to Titanfall's own filmic look." },
      { DxvkToneMapping::TonemapOperator::Psycho17, "Psycho17 (perceptual)",
        "renodx 'Psycho Test 17' perceptual/vision-model operator with neutral parameters and BT.2020\n"
        "gamut compression. Best handling of bright saturated highlights." },
      { DxvkToneMapping::TonemapOperator::GT7, "GT7 (Gran Turismo 7)",
        "Gran Turismo 7 SDR operator - chroma-preserving in ICtCp, peak 1.0, no parameters." },
      { DxvkToneMapping::TonemapOperator::PerceptualTF2, "PerceptualTF2 (PSDT)",
        "Perceptual Scene Display Transform. Not a curve: a scene-adaptive display transform with\n"
        "its own analysis passes - source-aware exposure anchoring, multi-scale local adaptation under\n"
        "a contrast budget, an exposure-aware perceptual curve, local contrast restoration, colour-volume\n"
        "pressure driving chroma compression and a controlled hue trajectory, a spatial white point, and\n"
        "a perceptual glare model. Settings live under rtx.tonemap.psdt.*; the other operators stay in\n"
        "this list as controls to measure it against." },
      { DxvkToneMapping::TonemapOperator::Reinhard, "Reinhard (control)",
        "Extended Reinhard, white point 4.0. No shoulder, no colour management, no path to white -\n"
        "it does not compress chroma, it clips. Here as the baseline PSDT's chroma retention and\n"
        "clipping figures are read against, not as a recommendation." },
      { DxvkToneMapping::TonemapOperator::AgX, "AgX (control)",
        "AgX at its reference configuration - no look transform, no contrast preset. The canonical\n"
        "controlled path to white, which is the interesting comparison for PSDT's hue trajectory\n"
        "rather than the easy one." },
    } }
  };

  // NV-DXVK [PSDT]: in-image diagnostics. Contiguous enum, but ComboWithKey is
  // used anyway so each entry carries the sentence explaining what a correct
  // picture looks like - a debug view nobody can read is a debug view nobody
  // uses.
  RemixGui::ComboWithKey<DxvkToneMapping::PsdtDebugView> psdtDebugViewCombo {
    "Debug View",
    RemixGui::ComboWithKey<DxvkToneMapping::PsdtDebugView>::ComboEntries { {
      { DxvkToneMapping::PsdtDebugView::Off, "Off", "The transformed image." },
      { DxvkToneMapping::PsdtDebugView::Adaptation, "Adaptation field",
        "Pooled local adaptation as stops about the frame anchor. Orange above, blue below.\n"
        "Should follow the scene's large-scale luminance layout and show no structure at object scale." },
      { DxvkToneMapping::PsdtDebugView::Budget, "Adaptation after budget",
        "The same field after the contrast budget clamped it, at 2 stops full scale.\n"
        "The difference between this and the previous view is the flattening the budget prevented." },
      { DxvkToneMapping::PsdtDebugView::Roles, "Role classification",
        "Green surface, red source, blue sky. The classification everything else is downstream of.\n"
        "Lamps and muzzle flashes should be red, sunlit walls green, anything past the far plane blue.\n"
        "If a bright wall is red, either Renderer Signals is off or Source Threshold is too low." },
      { DxvkToneMapping::PsdtDebugView::SourceEnergy, "Source energy",
        "The glare kernel's input, per channel, at mip 0. Should be black except at actual emitters,\n"
        "and should carry the emitter's own colour rather than the surrounding scene's." },
      { DxvkToneMapping::PsdtDebugView::Illuminant, "Local illuminant",
        "The measured colour of the light, as radiance over albedo. A room of coloured walls under a\n"
        "neutral light should read grey here; if it reads coloured, the estimate is picking up paint." },
      { DxvkToneMapping::PsdtDebugView::Depth, "Depth and pooling trust",
        "Red is body depth, green is how much of the coarse adaptation scales survived the depth test.\n"
        "Green should collapse at a window or a skyline and stay high across a continuous surface." },
      { DxvkToneMapping::PsdtDebugView::Demand, "Colour volume demand",
        "Green while the colour fits the display, red once it does not, with the crossover at exactly\n"
        "the boundary. The red area is what the colour-volume stage is actually working on." },
      { DxvkToneMapping::PsdtDebugView::Pressure, "Chroma given up",
        "How much chroma the compression actually removed. Should be zero over most of the frame." },
      { DxvkToneMapping::PsdtDebugView::Glare, "Glare only",
        "The glare contribution on its own, without the image underneath it." },
      { DxvkToneMapping::PsdtDebugView::Clipping, "Pre-clamp overflow",
        "Per-channel overflow measured before the final gamut fit had a chance to hide it.\n"
        "This is the honest picture of how much work the last clip is doing; it should be nearly black." },
      { DxvkToneMapping::PsdtDebugView::CurveSlope, "Curve slope",
        "d(display)/d(scene) at each pixel's base level, about 1.0. Orange above 1, blue below.\n"
        "Shows where the curve is adding contrast and where the shoulder and toe are taking it away." },
    } }
  };

  void DxvkToneMapping::showOperatorImguiSetting() {
    tonemapOperatorCombo.getKey(&tonemapOperatorObject());

    if (tonemapOperator() != TonemapOperator::None) {
      ImGui::Indent();
      ImGui::TextWrapped(
        "Operator overrides Tonemapping Mode: the global tonemapper is forced on and the local "
        "tonemapper is skipped. Per-operator parameters are fixed in tonemap_operators.slangh.");
      ImGui::Unindent();
    }

    // NV-DXVK [PSDT]: drawn here rather than in showImguiSettings for the same
    // reason the operator dropdown is - showImguiSettings is only reached from
    // the Global branch, and selecting an operator overrides Tonemapping Mode
    // without changing it, so the settings for the selected operator have to
    // stay reachable while the stored mode still says Local.
    if (tonemapOperator() != TonemapOperator::PerceptualTF2) {
      return;
    }

    ImGui::Indent();
    if (ImGui::CollapsingHeader("PSDT - Perceptual Scene Display Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();

      if (ImGui::CollapsingHeader("Display Model")) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "What the transform is targeting. Everything upstream is scene-referred; this is where it "
          "stops being so. Leave Peak equal to Reference White for SDR - that is the exact path.");
        RemixGui::DragFloat("Peak Luminance (nits)", &displayPeakNitsObject(), 1.0f, 20.f, 4000.f);
        RemixGui::DragFloat("Reference White (nits)", &displayRefWhiteNitsObject(), 1.0f, 20.f, 1000.f);
        RemixGui::DragFloat("Black Level (nits)", &displayBlackNitsObject(), 0.0005f, 0.f, 1.f);
        RemixGui::DragFloat("Surround (nits)", &surroundNitsObject(), 0.5f, 0.f, 200.f);
        RemixGui::Combo("Display Gamut", &displayGamutObject(), "Rec.709 / sRGB\0Display-P3\0Rec.2020\0");
        RemixGui::Combo("Perceptual Space", &perceptualSpaceObject(), "ICtCp\0Jzazbz\0");
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Scene Classification", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "Which pixels are surfaces, which are light sources and which are sky. Everything else in "
          "this transform is downstream of that question, and it is not answerable from the "
          "framebuffer - a sunlit white wall and a neon sign are both just bright. Renderer Signals "
          "reads the answer from the gbuffer instead: the per-pixel emissive bit, linear view depth, "
          "and the albedo that lets reflectance be divided out of radiance to leave the colour of "
          "the light. Turn it off to see how much of the image the guess was getting wrong.");
        RemixGui::Checkbox("Renderer Signals", &rendererSignalsObject());
        RemixGui::DragFloat("Source Threshold (stops)", &sourceThresholdObject(), 0.05f, 0.f, 16.f);
        RemixGui::DragFloat("Glare Class Threshold (stops)", &glareClassThresholdObject(), 0.05f, 0.f, 24.f);
        RemixGui::DragFloat("Sky View Depth", &skyViewZObject(), 1000.f, 1000.f, 400000.f);
        RemixGui::DragFloat("Illuminant Min Albedo", &illuminantMinAlbedoObject(), 0.002f, 0.005f, 0.5f);
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Adaptation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "What the viewer is adapted to. Source Exclusion decides whether a sunlit window is scene "
          "content or a highlight event; Contrast Budget bounds how far local adaptation may pull a "
          "region away from the frame anchor, and is what stops local adaptation flattening the image. "
          "Depth Sensitivity stops the pooling averaging a wall together with the sky behind it.");
        RemixGui::Checkbox("Scene Adaptive", &sceneAdaptiveObject());
        RemixGui::DragFloat("Source Exclusion", &sourceExclusionObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Local Adaptation", &localAdaptationObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Contrast Budget", &contrastBudgetObject(), 0.01f, 0.f, 4.f);
        RemixGui::DragFloat("Depth Sensitivity", &depthSensitivityObject(), 0.01f, 0.f, 2.f);
        RemixGui::Separator();
        RemixGui::DragFloat("Adaptation Speed Up", &adaptationSpeedUpObject(), 0.05f, 0.05f, 30.f);
        RemixGui::DragFloat("Adaptation Speed Down", &adaptationSpeedDownObject(), 0.05f, 0.05f, 30.f);
        RemixGui::Checkbox("Temporal Accumulation", &temporalAccumulationObject());
        RemixGui::DragFloat("Field Adaptation Speed", &fieldAdaptationSpeedObject(), 0.1f, 0.1f, 60.f);
        RemixGui::Separator();
        RemixGui::Checkbox("Coordinate With Auto Exposure Plus", &coordinateWithAutoExposurePlusObject());
        ImGui::TextWrapped(
          "Plus applies a local gain before this pass and PSDT applies local adaptation after it. "
          "Both are local dynamic range compression; at full strength in series they compress the "
          "same signal twice, which is the failure this architecture exists to avoid.");
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Luminance Curve", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "Midtone Contrast is the slope of the curve's linear segment. 1.0 is a straight line in "
          "log-log, which adds no contrast at all - the compression is done by the toe and shoulder, "
          "not by flattening the middle. Detail Strength restores local contrast the curve removed.");
        RemixGui::DragFloat("Midtone Contrast", &midtoneContrastObject(), 0.005f, 0.30f, 2.5f);
        RemixGui::DragFloat("Shadow Depth", &shadowDepthObject(), 0.01f, 0.05f, 0.98f);
        RemixGui::DragFloat("Highlight Rolloff", &highlightRolloffObject(), 0.01f, 0.05f, 0.98f);
        RemixGui::DragFloat("Detail Strength", &detailStrengthObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Detail Highlight Protection", &detailProtectObject(), 0.01f, 0.f, 4.f);
        RemixGui::DragFloat("Detail Knee (stops)", &detailKneeObject(), 0.05f, 0.25f, 12.f);
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Colour Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "Colour is compressed towards the gamut boundary - the most colourful thing the display can "
          "still show at that luminance - rather than towards white, and only when it does not fit. A "
          "bright white wall keeps its chroma untouched; a neon red at the same luminance is what has "
          "to give. Luminance Concession decides which of the two gives: a saturated red cannot be as "
          "bright as a white, so either it bleaches or it renders darker. Film renders it darker. "
          "Spatial White is a chromatic adaptation towards the measured local illuminant, so a "
          "tungsten interior desaturates towards its own warm white rather than towards daylight.");
        RemixGui::DragFloat("Chroma Preservation", &chromaPreservationObject(), 0.01f, 0.05f, 6.f);
        RemixGui::DragFloat("Luminance Concession", &luminanceConcessionObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Hue Trajectory", &hueTrajectoryObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("White Convergence", &whiteConvergenceObject(), 0.05f, 0.25f, 8.f);
        RemixGui::DragFloat("Spatial White", &spatialWhiteObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Colourfulness", &colourfulnessObject(), 0.01f, 0.f, 4.f);
        RemixGui::Separator();
        ImGui::TextWrapped(
          "Anticipatory compression: how early a colour starts easing towards the boundary rather "
          "than arriving at it. Pressure Luma is the brightness-times-chroma term; Pressure Context "
          "brings it forward for a colour that is a bright outlier against its own neighbourhood, "
          "where a hard clip would be visible and colour fidelity is not being judged.");
        RemixGui::DragFloat("Pressure Luma", &pressureLumaObject(), 0.01f, 0.f, 1.f);
        RemixGui::DragFloat("Pressure Context", &pressureContextObject(), 0.01f, 0.f, 3.f);
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Glare", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "How a source too bright to be a value becomes an area instead. Separate from rtx.bloom.* on "
          "purpose - bloom is a lens artifact on the image, this is a property of the light. The radius "
          "is not a setting: the threshold is applied per pyramid level, so a brighter source stays "
          "visible at more levels and spreads further on its own. Near Field is the narrow lobe of the "
          "kernel - the core a light has around it before the wide veil begins.");
        RemixGui::DragFloat("Glare Strength", &glareStrengthObject(), 0.01f, 0.f, 4.f);
        RemixGui::DragFloat("Glare Threshold (stops)", &glareThresholdObject(), 0.05f, 0.f, 20.f);
        RemixGui::DragFloat("Glare Falloff", &glareFalloffObject(), 0.01f, 0.05f, 0.95f);
        RemixGui::DragFloat("Glare Near Field", &glareNearFieldObject(), 0.01f, 0.f, 1.f);
        ImGui::Unindent();
      }

      if (ImGui::CollapsingHeader("Diagnostics")) {
        ImGui::Indent();
        ImGui::TextWrapped(
          "Each view replaces the image with a value the transform actually read or produced on that "
          "pixel on the way past, rather than recomputing its own version of it. Roles is the one to "
          "look at first - green surface, red source, blue sky - because the whole architecture rests "
          "on that classification. Clipping is the second, because it shows what the final gamut fit "
          "had to hide.");
        psdtDebugViewCombo.getKey(&debugViewObject());
        RemixGui::DragInt("Pyramid Levels (0 = auto)", &levelsObject(), 1.f, 0, PSDT_MAX_LEVELS);
        ImGui::Unindent();
      }

      ImGui::Unindent();
    }
    ImGui::Unindent();
  }

  DxvkToneMapping::DxvkToneMapping(DxvkDevice* device)
  : CommonDeviceObject(device), m_vkd(device->vkd())  {
  }
  
  DxvkToneMapping::~DxvkToneMapping()  {  }

  void DxvkToneMapping::showImguiSettings() {

    RemixGui::DragFloat("Global Exposure", &exposureBiasObject(), 0.01f, -4.f, 4.f);
    
    RemixGui::Checkbox("Color Grading Enabled", &colorGradingEnabledObject());
    if (colorGradingEnabled()) {
      ImGui::Indent();
      RemixGui::DragFloat("Contrast", &contrastObject(), 0.01f, 0.f, 1.f);
      RemixGui::DragFloat("Saturation", &saturationObject(), 0.01f, 0.f, 1.f);
      RemixGui::DragFloat3("Color Balance", &colorBalanceObject(), 0.01f, 0.f, 1.f);
      RemixGui::Separator();
      ImGui::Unindent();
    }

    RemixGui::Checkbox("Tonemapping Enabled", &tonemappingEnabledObject());
    if (tonemappingEnabled()) {
      ImGui::Indent();
      RemixGui::Checkbox("Finalize With ACES", &finalizeWithACESObject());

      RemixGui::Combo("Dither Mode", &ditherModeObject(), "Disabled\0Spatial\0Spatial + Temporal\0");

      RemixGui::Checkbox("Tuning Mode", &tuningModeObject());
      if (tuningMode()) {
        ImGui::Indent();

        RemixGui::DragFloat("Curve Shift", &curveShiftObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Shadow Min Slope", &shadowMinSlopeObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Shadow Contrast", &shadowContrastObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Shadow Contrast End", &shadowContrastEndObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Min Stops", &toneCurveMinStopsObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Max Stops", &toneCurveMaxStopsObject(), 0.01f, 0.f, 0.f);

        RemixGui::DragFloat("Max Exposure Increase", &maxExposureIncreaseObject(), 0.01f, 0.f, 0.f);
        RemixGui::DragFloat("Dynamic Range", &dynamicRangeObject(), 0.01f, 0.f, 0.f);

        ImGui::Unindent();
      }
      RemixGui::Separator();
      ImGui::Unindent();
    }
  }


  // =========================================================================
  // NV-DXVK [PSDT]: Perceptual Scene Display Transform
  // =========================================================================

  bool DxvkToneMapping::psdtSelected() const {
    return tonemapOperator() == TonemapOperator::PerceptualTF2 && tonemappingEnabled();
  }

  VkExtent3D DxvkToneMapping::calcPsdtFieldExtent(const VkExtent3D& targetExtent) {
    return VkExtent3D {
      std::max(1u, (targetExtent.width  + PSDT_FIELD_BLOCK_SIZE - 1) / PSDT_FIELD_BLOCK_SIZE),
      std::max(1u, (targetExtent.height + PSDT_FIELD_BLOCK_SIZE - 1) / PSDT_FIELD_BLOCK_SIZE),
      1
    };
  }

  uint32_t DxvkToneMapping::calcPsdtLevelCount(const VkExtent3D& fieldExtent) {
    // createImageResource asserts that the level count fits the extent, and at
    // low render resolutions the field is genuinely small - 320x180 gives a
    // 20x12 field, which supports five levels and not six.
    const uint32_t maxDim = std::max(fieldExtent.width, fieldExtent.height);
    uint32_t byExtent = 1;
    while ((maxDim >> byExtent) > 0) {
      ++byExtent;
    }

    uint32_t count = std::min<uint32_t>(PSDT_MAX_LEVELS, byExtent);
    if (levels() > 0) {
      count = std::min<uint32_t>(count, static_cast<uint32_t>(levels()));
    }
    return std::max(1u, count);
  }

  void DxvkToneMapping::releasePsdtResources() {
    for (uint32_t i = 0; i < 2; ++i) {
      m_psdtField[i].reset();
      m_psdtSource[i].reset();
      m_psdtIlluminant[i].reset();
    }
    m_psdtBodyHistogram.reset();
    m_psdtState.reset();
    m_psdtFieldExtent = VkExtent3D { 0, 0, 0 };
    m_psdtLevelCount = 0;
    m_psdtHasHistory = false;
    m_psdtHasCameraHistory = false;
  }

  void DxvkToneMapping::createPsdtResources(Rc<RtxContext> ctx, const VkExtent3D& targetExtent) {
    releasePsdtResources();

    m_psdtFieldExtent = calcPsdtFieldExtent(targetExtent);
    m_psdtLevelCount = calcPsdtLevelCount(m_psdtFieldExtent);

    // The mip chain is the adaptation pyramid and the glare kernel at once, so
    // all three fields need every level. See psdt_downsample.
    for (uint32_t i = 0; i < 2; ++i) {
      m_psdtField[i] = RtxMipmap::createResource(
        ctx, i == 0 ? "psdt adaptation field 0" : "psdt adaptation field 1",
        m_psdtFieldExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 0, { 0.f, 0.f, 0.f, 0.f }, m_psdtLevelCount);

      m_psdtSource[i] = RtxMipmap::createResource(
        ctx, i == 0 ? "psdt source field 0" : "psdt source field 1",
        m_psdtFieldExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 0, { 0.f, 0.f, 0.f, 0.f }, m_psdtLevelCount);

      m_psdtIlluminant[i] = RtxMipmap::createResource(
        ctx, i == 0 ? "psdt illuminant field 0" : "psdt illuminant field 1",
        m_psdtFieldExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 0, { 0.f, 0.f, 0.f, 0.f }, m_psdtLevelCount);
    }

    // Body-only luminance histogram. psdt_analysis accumulates into it with
    // atomics and psdt_state zeroes it once it has been read, so the only
    // clear needed is this one, for the first frame.
    {
      DxvkImageCreateInfo desc;
      desc.type = VK_IMAGE_TYPE_1D;
      desc.flags = 0;
      desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      desc.numLayers = 1;
      desc.mipLevels = 1;
      desc.extent = VkExtent3D { PSDT_BODY_HISTOGRAM_SIZE, 1, 1 };
      desc.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      desc.tiling = VK_IMAGE_TILING_OPTIMAL;
      desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
      desc.format = VK_FORMAT_R32_UINT;
      desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
      viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLevel = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;
      viewInfo.format = desc.format;
      viewInfo.usage = desc.usage;

      m_psdtBodyHistogram.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                        DxvkMemoryStats::Category::RTXRenderTarget, "psdt body histogram");
      m_psdtBodyHistogram.view = device()->createImageView(m_psdtBodyHistogram.image, viewInfo);
      ctx->changeImageLayout(m_psdtBodyHistogram.image, VK_IMAGE_LAYOUT_GENERAL);

      VkClearColorValue clearColor = {};
      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ctx->clearColorImage(m_psdtBodyHistogram.image, clearColor, subRange);
    }

    // AdaptationState. A 1D float image rather than a storage buffer, matching
    // how the exposure and tone curve textures already travel between these
    // passes - it is 64 scalars written by one thread once a frame, so the
    // shape of the resource is not where the interesting part is.
    {
      DxvkImageCreateInfo desc;
      desc.type = VK_IMAGE_TYPE_1D;
      desc.flags = 0;
      desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      desc.numLayers = 1;
      desc.mipLevels = 1;
      desc.extent = VkExtent3D { PSDT_STATE_SIZE, 1, 1 };
      desc.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      desc.tiling = VK_IMAGE_TILING_OPTIMAL;
      desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
      desc.format = VK_FORMAT_R32_SFLOAT;
      desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
      viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLevel = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;
      viewInfo.format = desc.format;
      viewInfo.usage = desc.usage;

      m_psdtState.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                DxvkMemoryStats::Category::RTXRenderTarget, "psdt adaptation state");
      m_psdtState.view = device()->createImageView(m_psdtState.image, viewInfo);
      ctx->changeImageLayout(m_psdtState.image, VK_IMAGE_LAYOUT_GENERAL);

      // psdt_state reads PSDT_STATE_HAS_HISTORY before it writes anything, so
      // the texture has to start at a defined zero or the first frame's
      // temporal blend reads uninitialised memory and can latch a nonsense
      // anchor that then eases back over seconds.
      VkClearColorValue clearColor = {};
      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ctx->clearColorImage(m_psdtState.image, clearColor, subRange);
    }

    m_psdtIndex = 0;
    m_psdtHasHistory = false;

    Logger::info(str::format(
      "[PSDT] adaptation field ", m_psdtFieldExtent.width, "x", m_psdtFieldExtent.height,
      " (", m_psdtLevelCount, " levels, finest ", PSDT_FIELD_BLOCK_SIZE,
      "px, coarsest ", PSDT_FIELD_BLOCK_SIZE << (m_psdtLevelCount - 1), "px)"));
  }

  // The PSDT resources are created whether or not the operator is selected,
  // because the apply shader's descriptor set layout does not change with the
  // operator and Vulkan will not accept an unbound binding. They cost a few
  // hundred KiB; the alternative is two pipeline layouts and a second apply
  // shader variant, which is a great deal more machinery for the same result.
  void DxvkToneMapping::ensurePsdtResources(Rc<RtxContext> ctx, const VkExtent3D& targetExtent) {
    const VkExtent3D wantedExtent = calcPsdtFieldExtent(targetExtent);

    if (m_psdtField[0].image.ptr() != nullptr
     && m_psdtFieldExtent.width == wantedExtent.width
     && m_psdtFieldExtent.height == wantedExtent.height
     && m_psdtLevelCount == calcPsdtLevelCount(wantedExtent)) {
      return;
    }

    createPsdtResources(ctx, targetExtent);
  }

  void DxvkToneMapping::dispatchPsdt(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    const Resources::Resource& colorBuffer,
    const float frameTimeMilliseconds,
    bool autoExposureEnabled) {

    if (!psdtSelected()) {
      // Resources stay allocated once created. They are under a megabyte and
      // they stay bound to the apply shader whatever operator is selected, so
      // freeing them would only buy a re-allocation the next time the operator
      // is switched back - which is exactly when someone is A/B comparing.
      return;
    }

    ScopedGpuProfileZone(ctx, "PSDT");

    const VkExtent3D targetExtent = colorBuffer.view->imageInfo().extent;
    const float deltaSeconds = 0.001f * frameTimeMilliseconds;

    if (m_resetState) {
      m_psdtHasHistory = false;
      m_psdtHasCameraHistory = false;
    }

    // Reprojection needs valid motion vectors; without them fall back to the
    // stateless field rather than reprojecting with garbage.
    const bool motionAvailable = rtOutput.m_primaryScreenSpaceMotionVector.image != nullptr;
    const bool accumulate = temporalAccumulation() && motionAvailable && m_psdtHasHistory;

    // The renderer-side classification signals. All three are plain
    // per-frame resources at the pre-upscale extent, written by the gbuffer
    // pass and still live at tonemap time. If any is missing the classifier
    // falls back to luminance and compactness alone, which is v0.1 behaviour -
    // worse, but not broken.
    const bool gbufferAvailable =
         rtOutput.m_primarySurfaceFlags.image != nullptr
      && rtOutput.m_primaryLinearViewZ.image != nullptr
      && rtOutput.m_primaryAlbedo.image != nullptr;
    const bool useRendererSignals = rendererSignals() && gbufferAvailable;

    const uint32_t readIndex = m_psdtIndex;
    const uint32_t writeIndex = 1 - m_psdtIndex;

    Rc<DxvkSampler> mipSampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // --- camera transition ------------------------------------------------
    // Three things look identical to a histogram and want three different
    // responses: a player sweeping their view, a teleport, and a scripted cut.
    // The camera separates the first two directly - isCameraCut is a position
    // jump larger than the scene's own unique-object distance, so it is a
    // teleport or a respawn by construction - and the angular delta across one
    // frame is the continuous signal for the third. The luminance test in
    // psdt_state stays as a second opinion for cuts that do not move the
    // camera at all, such as a light going out.
    float cameraAngularSpeed = 0.0f;
    bool cameraCut = false;
    bool cameraValid = false;
    {
      const RtCamera& camera = ctx->getSceneManager().getCamera();
      const Vector3 forward = camera.getDirection(true);
      const float forwardLength = length(forward);
      if (forwardLength > 0.5f) {
        cameraValid = true;
        const Vector3 unitForward = forward / forwardLength;
        if (m_psdtHasCameraHistory && deltaSeconds > 1e-5f) {
          const float cosAngle = std::clamp(dot(unitForward, m_psdtPreviousCameraForward), -1.0f, 1.0f);
          cameraAngularSpeed = std::acos(cosAngle) / deltaSeconds;
        }
        m_psdtPreviousCameraForward = unitForward;
        m_psdtHasCameraHistory = true;
      }
      cameraCut = camera.isCameraCut();
    }

    // --- Auto Exposure Plus coordination ----------------------------------
    // Plus applies a spatially varying gain before the tonemapper; PSDT
    // applies local adaptation after it. Both are local dynamic range
    // compression, and running both at full strength in series compresses the
    // same signal twice - which is the exact failure this architecture was
    // built to avoid, and which v0.1 did nothing about because it built its own
    // adaptation field beside Plus rather than coordinating with it.
    //
    // The honest coordination is to spend the local adaptation budget once
    // between the two passes. Plus reports how much of it it took as its own
    // intensity, so PSDT takes the remainder.
    float localScale = 1.0f;
    if (coordinateWithAutoExposurePlus()) {
      DxvkAutoExposurePlus& plus = ctx->getCommonObjects()->metaAutoExposurePlus();
      if (plus.isActive()) {
        localScale = std::max(0.0f, 1.0f - std::abs(DxvkAutoExposurePlus::intensity()));
      }
    }

    // --- scene analysis: full-res HDR + gbuffer -> field mip 0 ------------
    {
      ScopedGpuProfileZone(ctx, "PSDT: Scene Analysis");

      const VkExtent3D motionExtent = motionAvailable
        ? rtOutput.m_primaryScreenSpaceMotionVector.image->info().extent
        : m_psdtFieldExtent;
      const VkExtent3D gbufferExtent = useRendererSignals
        ? rtOutput.m_primaryLinearViewZ.image->info().extent
        : m_psdtFieldExtent;

      // Same exponential form the base auto exposure and Auto Exposure Plus
      // use, so that matching the speeds produces matching settling behaviour
      // rather than merely matching numbers.
      const float perFrameSpeed = fieldAdaptationSpeed() * deltaSeconds;

      PsdtAnalysisArgs pushArgs = {};
      pushArgs.colorExtent = uvec2 { targetExtent.width, targetExtent.height };
      pushArgs.fieldExtent = uvec2 { m_psdtFieldExtent.width, m_psdtFieldExtent.height };
      pushArgs.motionExtent = uvec2 { motionExtent.width, motionExtent.height };
      pushArgs.gbufferExtent = uvec2 { gbufferExtent.width, gbufferExtent.height };
      pushArgs.exposure = exp2f(exposureBias() + RtxOptions::calcUserEVBias());
      pushArgs.flags =
          (autoExposureEnabled ? PSDT_ANALYSIS_FLAG_AUTO_EXPOSURE : 0u)
        | (accumulate ? PSDT_ANALYSIS_FLAG_HAS_HISTORY : 0u)
        | (useRendererSignals ? PSDT_ANALYSIS_FLAG_RENDERER_SIGNALS : 0u);
      pushArgs.sourceThresholdStops = sourceThreshold();
      pushArgs.glareThresholdStops = glareClassThreshold();
      pushArgs.currentWeight = accumulate
        ? std::min(1.0f, std::max(0.05f, 1.0f - exp2f(-perFrameSpeed)))
        : 1.0f;
      pushArgs.skyViewZ = skyViewZ();
      pushArgs.illuminantMinAlbedo = std::max(illuminantMinAlbedo(), 1e-3f);
      pushArgs.toneCurveMinStops = toneCurveMinStops();
      pushArgs.toneCurveMaxStops = toneCurveMaxStops();
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      // Bindings the shader will not read still have to be satisfied. The
      // illuminant field is a four-component float image of the right shape,
      // so it stands in for whichever gbuffer input is absent.
      Rc<DxvkImageView> standIn = m_psdtIlluminant[readIndex].view;

      ctx->bindResourceView(PSDT_ANALYSIS_COLOR_INPUT, colorBuffer.view, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_EXPOSURE_INPUT, exposureView, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_MOTION_INPUT,
                            motionAvailable ? rtOutput.m_primaryScreenSpaceMotionVector.view
                                            : standIn, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_SURFACE_FLAGS_INPUT,
                            useRendererSignals ? rtOutput.m_primarySurfaceFlags.view : standIn, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_VIEW_Z_INPUT,
                            useRendererSignals ? rtOutput.m_primaryLinearViewZ.view : standIn, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_ALBEDO_INPUT,
                            useRendererSignals ? rtOutput.m_primaryAlbedo.view : standIn, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_FIELD_OUTPUT, m_psdtField[writeIndex].views[0], nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_FIELD_HISTORY_INPUT, m_psdtField[readIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_ANALYSIS_FIELD_HISTORY_INPUT, mipSampler);
      ctx->bindResourceView(PSDT_ANALYSIS_SOURCE_OUTPUT, m_psdtSource[writeIndex].views[0], nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_SOURCE_HISTORY_INPUT, m_psdtSource[readIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_ANALYSIS_SOURCE_HISTORY_INPUT, mipSampler);
      ctx->bindResourceView(PSDT_ANALYSIS_ILLUM_OUTPUT, m_psdtIlluminant[writeIndex].views[0], nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_ILLUM_HISTORY_INPUT, m_psdtIlluminant[readIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_ANALYSIS_ILLUM_HISTORY_INPUT, mipSampler);
      ctx->bindResourceView(PSDT_ANALYSIS_BODY_HISTOGRAM_OUTPUT, m_psdtBodyHistogram.view, nullptr);
      ctx->bindResourceView(PSDT_ANALYSIS_STATE_INPUT, m_psdtState.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PsdtAnalysisShader::getShader());

      // One workgroup per field texel: the 16x16 threads are the 16x16 source
      // pixels the texel covers, reduced in groupshared.
      ctx->dispatch(m_psdtFieldExtent.width, m_psdtFieldExtent.height, 1);
    }

    // --- adaptation pyramid ----------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "PSDT: Adaptation Pyramid");

      for (uint32_t level = 0; level + 1 < m_psdtLevelCount; ++level) {
        const VkExtent3D srcExtent = m_psdtField[writeIndex].image->mipLevelExtent(level);
        const VkExtent3D dstExtent = m_psdtField[writeIndex].image->mipLevelExtent(level + 1);

        PsdtDownsampleArgs pushArgs = {};
        pushArgs.srcExtent = uvec2 { srcExtent.width, srcExtent.height };
        pushArgs.dstExtent = uvec2 { dstExtent.width, dstExtent.height };
        ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

        ctx->bindResourceView(PSDT_DOWNSAMPLE_FIELD_INPUT, m_psdtField[writeIndex].views[level], nullptr);
        ctx->bindResourceView(PSDT_DOWNSAMPLE_FIELD_OUTPUT, m_psdtField[writeIndex].views[level + 1], nullptr);
        ctx->bindResourceView(PSDT_DOWNSAMPLE_SOURCE_INPUT, m_psdtSource[writeIndex].views[level], nullptr);
        ctx->bindResourceView(PSDT_DOWNSAMPLE_SOURCE_OUTPUT, m_psdtSource[writeIndex].views[level + 1], nullptr);
        ctx->bindResourceView(PSDT_DOWNSAMPLE_ILLUM_INPUT, m_psdtIlluminant[writeIndex].views[level], nullptr);
        ctx->bindResourceView(PSDT_DOWNSAMPLE_ILLUM_OUTPUT, m_psdtIlluminant[writeIndex].views[level + 1], nullptr);
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PsdtDownsampleShader::getShader());

        const VkExtent3D workgroups = util::computeBlockCount(dstExtent, VkExtent3D { PSDT_GROUP_SIZE, PSDT_GROUP_SIZE, 1 });
        ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
      }
    }

    // --- adaptation state -------------------------------------------------
    {
      ScopedGpuProfileZone(ctx, "PSDT: Adaptation State");

      PsdtStateArgs pushArgs = {};
      pushArgs.toneCurveMinStops = toneCurveMinStops();
      pushArgs.toneCurveMaxStops = toneCurveMaxStops();
      pushArgs.deltaTimeSeconds = deltaSeconds;

      // Everything that is not a float lives in the flags word: PsdtStateArgs
      // is exactly at the 128-byte push constant limit, so an enum that costs
      // a whole dword costs a parameter somewhere else.
      const uint32_t nearFieldFixed = static_cast<uint32_t>(
        std::clamp(glareNearField(), 0.0f, 1.0f) * 255.0f + 0.5f);
      pushArgs.flags =
          (m_psdtHasHistory ? PSDT_STATE_FLAG_HAS_HISTORY : 0u)
        | (sceneAdaptive() ? PSDT_STATE_FLAG_SCENE_ADAPTIVE : 0u)
        | (cameraCut ? PSDT_STATE_FLAG_CAMERA_CUT : 0u)
        | (cameraValid ? PSDT_STATE_FLAG_CAMERA_VALID : 0u)
        | (static_cast<uint32_t>(displayGamut()) << PSDT_STATE_SHIFT_GAMUT)
        | (static_cast<uint32_t>(perceptualSpace()) << PSDT_STATE_SHIFT_SPACE)
        | ((static_cast<uint32_t>(debugView()) & 0xFu) << PSDT_STATE_SHIFT_DEBUG)
        | ((m_psdtLevelCount & 0xFu) << PSDT_STATE_SHIFT_LEVELS)
        | ((nearFieldFixed & 0xFFu) << PSDT_STATE_SHIFT_NEARFIELD);

      pushArgs.displayPeakNits = displayPeakNits();
      pushArgs.displayBlackNits = displayBlackNits();
      pushArgs.displayRefWhiteNits = displayRefWhiteNits();
      pushArgs.surroundNits = surroundNits();

      pushArgs.sourceExclusion = sourceExclusion();
      pushArgs.localStrength = localAdaptation() * localScale;
      pushArgs.contrastBudget = contrastBudget() * localScale;
      pushArgs.depthSensitivity = useRendererSignals ? depthSensitivity() : 0.0f;

      pushArgs.adaptationSpeedUp = adaptationSpeedUp();
      pushArgs.adaptationSpeedDown = adaptationSpeedDown();
      pushArgs.cameraAngularSpeed = cameraAngularSpeed;

      pushArgs.midtoneContrast = midtoneContrast();
      pushArgs.shadowDepth = shadowDepth();
      pushArgs.highlightRolloff = highlightRolloff();
      pushArgs.detailStrength = detailStrength();
      pushArgs.detailProtect = detailProtect();
      pushArgs.detailKnee = detailKnee();

      pushArgs.chromaPreservation = chromaPreservation();
      pushArgs.hueTrajectory = hueTrajectory();
      pushArgs.pressureLuma = pressureLuma();
      pushArgs.pressureContext = pressureContext();
      pushArgs.whiteConvergence = whiteConvergence();
      pushArgs.illuminantStrength = spatialWhite();
      pushArgs.colourfulness = colourfulness();
      pushArgs.luminanceConcession = luminanceConcession();

      pushArgs.glareStrength = glareStrength();
      pushArgs.glareThresholdStops = glareThreshold();
      pushArgs.glareFalloff = glareFalloff();
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->bindResourceView(PSDT_STATE_BINDING_HISTOGRAM, m_toneHistogram.view, nullptr);
      ctx->bindResourceView(PSDT_STATE_BINDING_BODY_HISTOGRAM, m_psdtBodyHistogram.view, nullptr);
      ctx->bindResourceView(PSDT_STATE_BINDING_FIELD, m_psdtField[writeIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_STATE_BINDING_FIELD, mipSampler);
      ctx->bindResourceView(PSDT_STATE_BINDING_SOURCE, m_psdtSource[writeIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_STATE_BINDING_SOURCE, mipSampler);
      ctx->bindResourceView(PSDT_STATE_BINDING_ILLUM, m_psdtIlluminant[writeIndex].view, nullptr);
      ctx->bindResourceSampler(PSDT_STATE_BINDING_ILLUM, mipSampler);
      ctx->bindResourceView(PSDT_STATE_BINDING_STATE, m_psdtState.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PsdtStateShader::getShader());

      // One workgroup, one thread per histogram bucket. It has to run here -
      // after dispatchHistogram and before dispatchToneCurve - because the
      // tone curve pass zeroes the histogram when it is done with it.
      ctx->dispatch(1, 1, 1);
    }

    m_psdtIndex = writeIndex;
    m_psdtHasHistory = true;
  }

  void DxvkToneMapping::createResources(Rc<RtxContext> ctx) {
    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_1D;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;
    viewInfo.format = desc.format;

    desc.extent = VkExtent3D{ TONEMAPPING_TONE_CURVE_SAMPLE_COUNT, 1, 1 };

    viewInfo.format = desc.format = VK_FORMAT_R32_UINT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_toneHistogram.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "tone mapper histogram");
    m_toneHistogram.view = device()->createImageView(m_toneHistogram.image, viewInfo);
    ctx->changeImageLayout(m_toneHistogram.image, VK_IMAGE_LAYOUT_GENERAL);

    viewInfo.format = desc.format = VK_FORMAT_R32_SFLOAT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    m_toneCurve.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "tone mapper curve");
    m_toneCurve.view = device()->createImageView(m_toneCurve.image, viewInfo);
    ctx->changeImageLayout(m_toneCurve.image, VK_IMAGE_LAYOUT_GENERAL);
  }

  void DxvkToneMapping::dispatchHistogram(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& colorBuffer,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Tonemap: Generate Histogram");

    // Clear the histogram resource
    if(m_resetState) {
      VkClearColorValue clearColor;
      clearColor.float32[0] = clearColor.float32[1] = clearColor.float32[2] = clearColor.float32[3] = 0;

      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

      ctx->clearColorImage(m_toneHistogram.image, clearColor, subRange);
    }

    // Prepare shader arguments
    ToneMappingHistogramArgs pushArgs = {};
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.exposureFactor = exp2f(exposureBias() + RtxOptions::calcUserEVBias());

    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

    VkExtent3D workgroups = util::computeBlockCount(colorBuffer.view->imageInfo().extent, VkExtent3D{16, 16, 1 });

    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_COLOR_INPUT, colorBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_HISTOGRAM_INPUT_OUTPUT, m_toneHistogram.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_HISTOGRAM_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, HistogramShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatchToneCurve(
    Rc<RtxContext> ctx) {

    ScopedGpuProfileZone(ctx, "Tonemap: Calculate Tone Curve");

    // Prepare shader arguments
    ToneMappingCurveArgs pushArgs = {};
    pushArgs.dynamicRange = dynamicRange();
    pushArgs.shadowMinSlope = shadowMinSlope();
    pushArgs.shadowContrast = shadowContrast();
    pushArgs.shadowContrastEnd = shadowContrastEnd();
    pushArgs.maxExposureIncrease = maxExposureIncrease();
    pushArgs.curveShift = curveShift();
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.needsReset = m_resetState;

    VkExtent3D workgroups = VkExtent3D{ TONEMAPPING_TONE_CURVE_SAMPLE_COUNT, 1, 1 };

    ctx->bindResourceView(TONEMAPPING_TONE_CURVE_HISTOGRAM_INPUT_OUTPUT, m_toneHistogram.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_TONE_CURVE_TONE_CURVE_INPUT_OUTPUT, m_toneCurve.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ToneCurveShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatchApplyToneMapping(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& inputBuffer,
    const Resources::Resource& colorBuffer,
    bool performSRGBConversion,
    bool autoExposureEnabled,
    bool forceFinalizeWithACES) {

    ScopedGpuProfileZone(ctx, "Apply Tone Mapping");

    const VkExtent3D workgroups = util::computeBlockCount(colorBuffer.view->imageInfo().extent, VkExtent3D{ 16 , 16, 1 });

    // NV-DXVK [auto exposure plus]: the native dynamic tone curve is a near-straight line in
    // log-log across dynamicRange stops - it redistributes luminance but supplies no toe and no
    // shoulder, so on its own it renders flat. The LOCAL tonemapper never had that problem
    // because rtx.localtonemap.finalizeWithACES defaults to true and gives it an S-curve. When
    // something overrides the local path onto this one, that S-curve has to come with it, or
    // the override silently changes the look of the whole image. Overriding rather than writing
    // rtx.tonemap.finalizeWithACES keeps the user's stored value intact, the same way the
    // tonemappingMode override does.
    const bool acesFinalize = finalizeWithACES() || forceFinalizeWithACES;

    // Prepare shader arguments
    ToneMappingApplyToneMappingArgs pushArgs = {};
    pushArgs.toneMappingEnabled = tonemappingEnabled();
    pushArgs.colorGradingEnabled = colorGradingEnabled();
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.finalizeWithACES = acesFinalize;
    pushArgs.useLegacyACES = RtxOptions::useLegacyACES();

    // Tonemap args
    pushArgs.performSRGBConversion = performSRGBConversion;
    pushArgs.shadowContrast = shadowContrast();
    pushArgs.shadowContrastEnd = shadowContrastEnd();
    pushArgs.exposureFactor = exp2f(exposureBias() + RtxOptions::calcUserEVBias()); // ev100
    pushArgs.toneCurveMinStops = toneCurveMinStops();
    pushArgs.toneCurveMaxStops = toneCurveMaxStops();
    pushArgs.debugMode = tuningMode();

    // Color grad args
    pushArgs.colorBalance = colorBalance();
    pushArgs.contrast = contrast();
    pushArgs.saturation = saturation();

    // Dither args
    switch (ditherMode()) {
    case DitherMode::None: pushArgs.ditherMode = ditherModeNone; break;
    case DitherMode::Spatial: pushArgs.ditherMode = ditherModeSpatialOnly; break;
    case DitherMode::SpatialTemporal: pushArgs.ditherMode = ditherModeSpatialTemporal; break;
    }
    pushArgs.frameIndex = ctx->getDevice()->getCurrentFrameId();

    // NV-DXVK [tonemap operators]: select the fork operator (0 = native curve).
    pushArgs.tonemapOperator = static_cast<uint32_t>(tonemapOperator());

    // NV-DXVK [tonemap operators]: report which operator is live and whether the
    // native ACES finalize can actually take effect. ACES finalize ONLY runs on
    // the native path (operator == None, not in tuning/debug mode); any selected
    // operator bypasses it. Change-triggered so it does not spam.
    {
      static uint32_t s_lastOp = 0xFFFFFFFFu;
      static int s_lastAces = -1;
      const bool acesEffective =
        (pushArgs.tonemapOperator == tonemapOperatorNone) && !tuningMode() && acesFinalize;
      if (pushArgs.tonemapOperator != s_lastOp || (int) acesFinalize != s_lastAces) {
        s_lastOp = pushArgs.tonemapOperator;
        s_lastAces = (int) acesFinalize;
        Logger::info(str::format(
          "[Tonemap] operator=", pushArgs.tonemapOperator,
          " (0=native,3=Hable,6=Psycho17,7=GT7,8=PerceptualTF2) tonemappingEnabled=", tonemappingEnabled() ? 1 : 0,
          " finalizeWithACES=", acesFinalize ? 1 : 0,
          " (option=", finalizeWithACES() ? 1 : 0, " forced=", forceFinalizeWithACES ? 1 : 0, ")",
          " acesEffective=", acesEffective ? 1 : 0,
          acesEffective ? "" : " (ACES finalize bypassed: an operator is selected and/or tonemapping disabled)"));
      }
    }

    ctx->bindResourceView(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT, inputBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT, m_toneCurve.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceSampler(TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT, linearSampler);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT, colorBuffer.view, nullptr);

    // NV-DXVK [PSDT]: bound whatever the operator, because the descriptor set
    // layout does not change with the operator selection and Vulkan requires
    // every declared binding to be satisfied. The apply shader only reads them
    // on the PerceptualTF2 branch.
    {
      Rc<DxvkSampler> mipSampler = ctx->getResourceManager().getSampler(
        VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
      const uint32_t currentIndex = m_psdtIndex;

      ctx->bindResourceView(TONEMAPPING_APPLY_PSDT_FIELD_INPUT, m_psdtField[currentIndex].view, nullptr);
      ctx->bindResourceSampler(TONEMAPPING_APPLY_PSDT_FIELD_INPUT, mipSampler);
      ctx->bindResourceView(TONEMAPPING_APPLY_PSDT_SOURCE_INPUT, m_psdtSource[currentIndex].view, nullptr);
      ctx->bindResourceSampler(TONEMAPPING_APPLY_PSDT_SOURCE_INPUT, mipSampler);
      ctx->bindResourceView(TONEMAPPING_APPLY_PSDT_ILLUM_INPUT, m_psdtIlluminant[currentIndex].view, nullptr);
      ctx->bindResourceSampler(TONEMAPPING_APPLY_PSDT_ILLUM_INPUT, mipSampler);
      ctx->bindResourceView(TONEMAPPING_APPLY_PSDT_STATE_INPUT, m_psdtState.view, nullptr);
    }

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ApplyTonemappingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

  }

  void DxvkToneMapping::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds,
    bool performSRGBConversion,
    bool resetHistory,
    bool autoExposureEnabled,
    bool forceFinalizeWithACES) {

    ScopedGpuProfileZone(ctx, "Tone Mapping");

    m_resetState |= resetHistory;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    // TODO : set reset on significant camera changes as well
    if (m_toneHistogram.image.ptr() == nullptr) {
      createResources(ctx);
      m_resetState = true;
    }

    const Resources::Resource& inputColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);

    // NV-DXVK [PSDT]: has to happen before dispatchApplyToneMapping whatever the
    // operator, since the apply pass binds these unconditionally.
    ensurePsdtResources(ctx, inputColorBuffer.view->imageInfo().extent);
    if (tonemappingEnabled()) {
      dispatchHistogram(ctx, exposureView, inputColorBuffer, autoExposureEnabled);
      // NV-DXVK [PSDT]: strictly between these two. dispatchPsdt reads the
      // luminance histogram, and dispatchToneCurve zeroes it on its way out.
      dispatchPsdt(ctx, exposureView, rtOutput, inputColorBuffer, frameTimeMilliseconds, autoExposureEnabled);
      dispatchToneCurve(ctx);
    }

    dispatchApplyToneMapping(ctx, linearSampler, exposureView, inputColorBuffer, rtOutput.m_finalOutput.resource(Resources::AccessType::Write), performSRGBConversion, autoExposureEnabled, forceFinalizeWithACES);

    m_resetState = false;
  }
}
