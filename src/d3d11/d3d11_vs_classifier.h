#pragma once

#include <cstdint>
#include <vector>

namespace dxvk {

  class D3D11CommonShader;
  struct D3D11RtxSemantic;

  // NV-DXVK: principled VS classifier.
  // Pure function of the shader's RDEF + declared input semantics.
  // Returns the Kind of geometry the VS produces and which slots the caller
  // should read transforms from. No side effects — no state read or written
  // outside the return value.
  //
  // Design goal: replace the tangle of competing paths in ExtractTransforms
  // (t31 / t30cpu / t30slice / cb3 RDEF / legacy blind-probe) where each
  // path had unrelated side effects and overrode each other. Here:
  //
  //   auto cls = D3D11VsClassifier::classify(vsCommon, inputSemantics);
  //   switch (cls.kind) {
  //     case Kind::StaticWorld:   read cb3.CBufModelInstance.objectToCameraRelative
  //     case Kind::InstancedBsp:  read t31[charIdx] via per-instance UINT4
  //     case Kind::SkinnedChar:   attach t30 or t31 as bone palette, o2w=identity
  //     case Kind::UI / Unknown:  filter as UIFallback
  //   }
  //
  // Skybox3D, Skybox2D, Viewmodel, Particle, Sprite2D variants require
  // cross-draw context (cb2 comparison against MainWorld). Those are left as
  // extension points — the enum includes them but classify() won't return them
  // yet. A second-pass refiner that takes FrameCameraContext can upgrade a
  // StaticWorld to Skybox3D etc. without touching classify().

  struct D3D11VsClassification {
    enum class Kind : uint32_t {
      Unknown = 0,       // classifier failed — treat as UI
      UI,                // no real transform, 2D/screenspace
      StaticWorld,       // cb3 CBufModelInstance owns the transform
      InstancedBsp,      // per-instance UINT4 index → t31 g_modelInst
      SkinnedChar,       // BLENDINDICES + BLENDWEIGHT → t30/t31 bone palette

      // Extension points — require FrameCameraContext to disambiguate
      // from StaticWorld/InstancedBsp. Not produced by classify() yet.
      Skybox3D,
      Skybox2D,
      Viewmodel,
      Particle,
      Sprite2D,
    };

    Kind kind = Kind::Unknown;

    // Slot the transform buffer is bound on. UINT32_MAX if not applicable.
    uint32_t modelInstSlot   = ~0u; // t31 for InstancedBsp
    uint32_t bonePaletteSlot = ~0u; // t30 or t31 for SkinnedChar
    uint32_t cb3Slot         = ~0u; // cb3 for StaticWorld (CBufModelInstance)

    // True iff RDEF named the resource (high-confidence). False iff the
    // classifier relied on input semantics alone (still correct, but the
    // caller may want to log).
    bool modelInstFromRdef   = false;
    bool bonePaletteFromRdef = false;

    // Reason string for diagnostics. Short, stable, parsable.
    // e.g. "rdef_cb3_CBufModelInstance", "sem_color1_perinst_uint4",
    //      "sem_blendindices_perVertex", "no_signals".
    const char* reason = "no_signals";
  };

  class D3D11VsClassifier {
  public:
    // Classify a VS. `commonShader` may be null (unknown shader → Unknown).
    // `semantics` may be empty. Result is stable for a given (shader, il) pair.
    static D3D11VsClassification classify(
        const D3D11CommonShader*              commonShader,
        const std::vector<D3D11RtxSemantic>&  semantics);

    // Convert Kind to short string for logs.
    static const char* kindName(D3D11VsClassification::Kind k);
  };

}
