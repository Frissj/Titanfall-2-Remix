/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <cstdint>

// NV-DXVK [engine-post forward]: shared producer/consumer state for forwarding
// the host game's final post-process composite (Titanfall2/Source "CBufEnginePost"
// pass: tonemap + bloom + DoF + color-correction) into Remix's own post pipeline.
//
// Producer: D3D11Rtx::SubmitDraw harvests the bound CBufEnginePost when it
// detects the engine post-composite draw, drops that draw (so it is not injected
// as scene geometry), and writes the parameters here once per frame.
//
// Consumers:
//   - Bloom / auto-exposure are driven directly via RTX_OPTION setDeferred at
//     harvest time (Remix has native equivalents) and do NOT need this struct.
//   - Tonemap (ported filmic curve), color-correction (3D LUT volume) and DoF
//     (Route B, path-traced-depth CoC) are GPU passes; they read their params
//     from here when building their constant buffers.
//
// Single producer (D3D11 submit thread) / single consumer (RT frame post). A
// frame stamp lets consumers tell whether the host actually ran its post pass
// this frame; when it did not, every effect's `active` flag is false and the
// consumers fall back to Remix's own behaviour.
namespace dxvk {

  struct EnginePostState {
    // Offsets below are the verified CBufEnginePost byte offsets (RDEF reflection
    // of the Titanfall2 post FS, cbuffer size 304). Kept as comments next to the
    // fields so the harvester and any future game can be cross-checked.

    // --- Tonemap (filmic toe/mid/shoulder). Ported into the apply shader. ---
    std::atomic<bool>  tonemapActive    { false };
    float tonemapToe        = 0.0f;  // c_debugTonemapToe       @208
    float tonemapMid1       = 0.0f;  // c_debugTonemapMid1      @212
    float tonemapMid2       = 0.0f;  // c_debugTonemapMid2      @216
    float tonemapShoulder   = 0.0f;  // c_debugTonemapShoulder  @220
    bool  tonemapTweaks     = false; // c_debugTonemapEnableTweaks @200 (!= 0)
    bool  tonemapDisabled   = false; // c_debugTonemapDisable   @204 (!= 0)

    // --- Color correction (Source 32^3 LUT volume). New LUT stage. ---
    std::atomic<bool>  ccActive         { false };
    float ccWeights[3]      = { 0.0f, 0.0f, 0.0f }; // c_colorCorrectionVolumeWeights @32
    float fadeToBlack       = 0.0f;  // c_fadeToBlackFactor     @28

    // --- Depth of field (Route B). New CoC-from-RT-depth pass. ---
    std::atomic<bool>  dofActive        { false };
    float dof[8]            = { 0,0,0,0, 0,0,0,0 }; // c_dof (8 floats) @272
    // Width of the game's DoFBlurSmallTexture (bound to the post composite, t10).
    // The game encodes blur strength as this texture's downsample resolution
    // rather than a radius, so the DoF pass derives its gather radius from
    // (renderWidth / dofBlurSmallWidth). 0 = not yet seen (fall back to a default).
    std::atomic<uint32_t> dofBlurSmallWidth { 0 };

    // Frame this state was last written by the producer; consumers compare
    // against the current frame to know if the host post pass ran.
    std::atomic<uint64_t> writtenFrame  { 0 };

    static EnginePostState& get() {
      static EnginePostState s_state;
      return s_state;
    }
  };

}
