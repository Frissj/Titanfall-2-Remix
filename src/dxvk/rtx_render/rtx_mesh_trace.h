/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* NV-DXVK [MeshTrace] submission funnel.
*
* [MeshTrace] in rtx_accel_manager reports whether a mesh reached the TLAS.
* When it did not, that alone cannot say WHERE it was lost, and the existing
* [BulkPush] census counts per vertex shader — useless here, because one
* shader draws many meshes, so "the shader is busy" never attributes to the
* mesh that vanished. That ambiguity has now cost two rounds of investigation.
*
* This records the same mesh identity ((vertexShaderHash, vertexCount) — the
* only pair that survives a restart, see rtx_options.h traceVertexCounts) at
* each stage between the game's draw call and the TLAS, so the funnel names
* the stage the mesh died at:
*
*   Submitted    SceneManager::submitDrawState entry — the game issued it
*   ProcessDcs   reached processDrawCallState — survived submitDrawState filters
*   BlasReady    a BlasEntry exists for it
*   InstanceMade processSceneObject returned an RtInstance
*   InstanceNull processSceneObject returned nullptr
*   (TLAS)       counted by the accel-manager walk, not here
*
* Also carries the material hash seen AT SUBMISSION. Comparing that against
* the material hash the instance carries at TLAS time separates "the game/
* texture binding produced a new material" from "Remix re-keyed it later" —
* without dereferencing any texture. Deliberately: calling getImageHash() from
* a diagnostic walk races texture streaming freeing ManagedTexture's mip view,
* which is a known crash in this project.
*
* Header-only so no build-file change is needed; the inline function-local
* statics are one instance across all translation units under C++17.
*
* Cost when tracing is off: one atomic-free check of two empty option sets.
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtx_options.h"

namespace dxvk::meshtrace {

  // ---- Filter snapshot ----
  //
  // The first version read RtxOptions::traceVertexShaders() per draw from the
  // D3D11 submit thread. That returns a REFERENCE into the option system's
  // layered storage, which the main thread re-resolves every frame
  // ([Perf.applyPending] shows it running); iterating that set while it is
  // rebuilt is a use-after-free, and the build that introduced it crashed
  // after ~60 s with no error line. I do not have a dump proving that was the
  // cause, but it is a genuine hazard on a hot cross-thread path and it should
  // not exist either way.
  //
  // Instead the filter is snapshotted into an immutable object once per frame
  // on the render thread and published atomically. Readers take a shared_ptr
  // copy, so the object they read can never be mutated or freed underneath
  // them. It also removes an option lookup from every single draw call.
  struct FilterSnapshot {
    std::vector<uint64_t> vsHashes;
    std::vector<uint64_t> vertexCounts;
    bool any = false;
  };

  inline std::shared_ptr<const FilterSnapshot>& filterStorage() {
    static std::shared_ptr<const FilterSnapshot> f = std::make_shared<const FilterSnapshot>();
    return f;
  }

  inline std::mutex& filterMutex() {
    static std::mutex m;
    return m;
  }

  inline std::shared_ptr<const FilterSnapshot> loadFilter() {
    std::lock_guard<std::mutex> lock(filterMutex());
    return filterStorage();
  }

  // Render thread only, once per frame, before the tables are drained.
  inline void refreshFilter() {
    auto snap = std::make_shared<FilterSnapshot>();
    for (const auto& h : RtxOptions::traceVertexShaders()) {
      snap->vsHashes.push_back(h);
    }
    for (const auto& v : RtxOptions::traceVertexCounts()) {
      snap->vertexCounts.push_back(v);
    }
    snap->any = !snap->vsHashes.empty() || !snap->vertexCounts.empty();
    std::lock_guard<std::mutex> lock(filterMutex());
    filterStorage() = snap;
  }

  enum class Stage : uint32_t {
    Submitted = 0,
    ProcessDcs,
    BlasReady,
    InstanceMade,
    InstanceNull,
    // NV-DXVK [BatchSubmitDraw] blind spot, closed 2026-07-30.
    //
    // With rtx.batchSubmitDrawStages on, a draw does NOT go straight from
    // D3D11Rtx::SubmitDraw to SceneManager::submitDrawState. It is collected
    // into a per-frame arena and re-emitted by flushGeometryBatch() at
    // EndFrame. Nothing observed that hop, so a draw that entered the arena and
    // never arrived at submitDrawState was indistinguishable from one the
    // d3d11_rtx filter cascade rejected -- 15 of the 17 "DISCARDED INSIDE
    // d3d11_rtx" rows in the 13:30 run were in exactly that state, with no
    // reject line and no early-return marker to their name.
    //
    // Batched      = accepted into the arena (the commit point was reached)
    // BatchEmitted = Phase C handed it to the CS thread for commitGeometryToRT
    //
    // Batched>0 with BatchEmitted==0 means the arena lost it. Both >0 with
    // Submitted==0 means it was re-emitted but died between commitGeometryToRT
    // and submitDrawState. Either way the stage is named instead of inferred.
    Batched,
    BatchEmitted,
    // NV-DXVK [MeshTrace] earliest IDENTITY-keyed observation, added 2026-07-30.
    //
    // Pipeline order is GeometryDerived -> Batched -> BatchEmitted -> Submitted;
    // it is listed last only so the existing stage indices do not shift.
    //
    // Why it exists: stage 0 (drawTable) is keyed on (vs, indexCount), a draw
    // SHAPE, so it can never say whether a specific MESH was drawn -- a sibling
    // prop of identical shape makes it non-zero regardless. Every other stage is
    // keyed on (vs, vertexCount), the accel side's identity, but the earliest of
    // them sits at the batch commit point, i.e. AFTER the whole d3d11_rtx filter
    // cascade. That left a gap: "nothing arrived Remix-side" could mean either
    // the engine never emitted the draw, or it emitted it and d3d11_rtx dropped
    // it, and nothing could tell those apart.
    //
    // Recorded at the first function-scope point in SubmitDraw where the
    // identity actually exists (geo.vertexCount assignment). So:
    //   GeometryDerived == 0                      -> nothing of this identity got
    //                                                that far (engine did not emit
    //                                                it, or it was rejected before
    //                                                geometry derivation -- and
    //                                                BumpFilter names those)
    //   GeometryDerived > 0, Batched/Submitted 0  -> the draw provably existed and
    //                                                died inside the d3d11_rtx
    //                                                cascade, identity-confirmed
    GeometryDerived,
    kCount
  };

  struct Counts {
    uint32_t stage[static_cast<size_t>(Stage::kCount)] = {};
    uint64_t submitMat = 0;   // material hash as seen at submission time
    uint64_t vsHash = 0;
    uint32_t vertexCount = 0;

    // ---- Stage-0 attribution confidence (drawTable only) ----
    //
    // (vs, indexCount) is NOT unique per mesh. Two props that share a shader
    // and an index count land on ONE key, so stage[Submitted] counts both and
    // cannot be attributed to either. Measured on the 2026-07-30 log: 30 of the
    // 120 "GAME DID NOT ISSUE" rows also matched a reject line on the same
    // (frame, idx) -- a 25% false-join rate that was completely invisible in
    // the output, and enough on its own to explain the entire 16-row
    // "DISCARDED INSIDE d3d11_rtx" bucket.
    //
    // Widening the key is NOT possible: the join's other side has only
    // blasEntry->input.getGeometryData(), and RasterGeometry does not retain
    // firstIndex/baseVertex (they are baked into the RasterBuffer slices). A
    // widened key would therefore miss every lookup and report zero draws for
    // everything -- the same silent-join failure this file's stage-0 comment
    // already warns about.
    //
    // So instead of pretending the key is unique, the submit side counts how
    // many DISTINCT (firstIndex, baseVertex) draw ranges shared this key during
    // the frame. variants == 1 means stage[Submitted] belongs to a single mesh
    // and can be trusted; variants > 1 means it is a sum over that many
    // distinct draws and must not be used to decide whether the game drew THIS
    // mesh. The ambiguity becomes a printed number instead of a wrong verdict.
    static constexpr uint32_t kMaxVariants = 8;
    uint64_t variantIds[kMaxVariants] = {};
    uint32_t variantCount = 0;
    bool     variantOverflow = false;   // more than kMaxVariants distinct ranges
  };

  // Result of a stage-0 lookup. Bundles the count with the confidence in it so
  // a call site cannot read one without the other.
  struct DrawAttrib {
    uint32_t draws    = 0;
    uint32_t variants = 0;
    bool     overflow = false;
    // True when this key carried at most one distinct draw range THIS FRAME.
    //
    // Read the limit carefully -- I originally documented this as "attributable
    // to exactly one mesh" and that is WRONG. The key is (vs, indexCount), a
    // draw SHAPE. This returning true only rules out two ranges colliding within
    // the frame; it does NOT establish that the one draw under the key belongs to
    // the mesh being investigated. A prop that stops drawing while a sibling of
    // identical shape keeps drawing yields draws==1, variants==1, and tells you
    // nothing about the prop. Measured: that is exactly what produced 9 bogus
    // "DISCARDED INSIDE d3d11_rtx" verdicts in the 2026-07-30 13:43 run.
    //
    // Consequently `draws` may support a conclusion but must never decide one.
    // The funnel stages are keyed on (vs, vertexCount) -- the same identity the
    // accel-manager side uses -- and are the authoritative signal.
    bool attributable() const { return variants <= 1 && !overflow; }
  };

  using Table = std::unordered_map<uint64_t, Counts>;

  inline std::mutex& tableMutex() {
    static std::mutex m;
    return m;
  }

  inline Table& table() {
    static Table t;
    return t;
  }

  // Same mixer as the accel-manager side so both agree on identity.
  inline uint64_t makeKey(uint64_t vs, uint32_t verts) {
    uint64_t h = vs ^ (0x9e3779b97f4a7c15ull + (static_cast<uint64_t>(verts) << 1));
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27; h *= 0x94d049bb133111ebull;
    return h ^ (h >> 31);
  }

  // Material is intentionally NOT part of the filter here: a mesh that is being
  // dropped must still be followed while its material re-keys underneath it,
  // which is exactly the case observed (143 rehashes on meshes that never left).
  inline bool matches(uint64_t vsHash, uint32_t vertexCount) {
    const auto f = loadFilter();
    if (!f->any) {
      return false;
    }
    if (!f->vsHashes.empty()) {
      bool hit = false;
      for (const uint64_t h : f->vsHashes) { if (h == vsHash) { hit = true; break; } }
      if (!hit) return false;
    }
    if (!f->vertexCounts.empty()) {
      bool hit = false;
      for (const uint64_t v : f->vertexCounts) {
        if (v == static_cast<uint64_t>(vertexCount)) { hit = true; break; }
      }
      if (!hit) return false;
    }
    return true;
  }

  inline void record(uint64_t vsHash, uint32_t vertexCount, Stage s, uint64_t materialHash = 0) {
    if (!matches(vsHash, vertexCount)) {
      return;
    }
    const uint64_t k = makeKey(vsHash, vertexCount);
    std::lock_guard<std::mutex> lock(tableMutex());
    Counts& c = table()[k];
    c.vsHash = vsHash;
    c.vertexCount = vertexCount;
    ++c.stage[static_cast<size_t>(s)];
    if (s == Stage::Submitted) {
      c.submitMat = materialHash;
    }
  }

  // Called once per frame by the accel manager, which owns the reporting.
  inline void snapshotAndClear(Table& out) {
    std::lock_guard<std::mutex> lock(tableMutex());
    out = table();
    table().clear();
  }

  // ---- Stage 0: the game's own D3D11 draw call ----
  //
  // Separate table because it is keyed on (vs, INDEX count), not
  // (vs, vertex count): at D3D11Rtx::SubmitDraw the vertex count has not been
  // derived yet, only the raw index count from the draw argument. The trace
  // side joins using input.getGeometryData().indexCount, which is that same
  // raw value. Do NOT join against buildRanges[0].primitiveCount — that is
  // post-processing and does not equal indexCount/3; a previous version did,
  // the join silently missed, and present meshes reported zero draws.
  //
  // Without this stage the funnel's first entry is SceneManager, and calling
  // that "the game did not submit it" is unsupportable — the entire d3d11_rtx
  // filter cascade sits between the two.
  //
  // What this key can and cannot prove: (vs, indexCount) identifies a DRAW
  // SHAPE, not a mesh. A non-zero count proves some traced draw of that shape
  // happened; it does NOT prove it was the mesh under investigation. Read
  // DrawAttrib::attributable() (or the `variants=` field in the report) before
  // concluding anything from the count -- measured 25% false-join rate on the
  // 2026-07-30 log. See the Counts comment above for why the key cannot simply
  // be widened.
  inline Table& drawTable() {
    static Table t;
    return t;
  }

  // Cheap shader-only test, for call sites that need to know whether a draw is
  // worth tracking before its geometry has been derived.
  inline bool isTracedVs(uint64_t vsHash) {
    const auto f = loadFilter();
    if (f->vsHashes.empty()) {
      return false;
    }
    for (const uint64_t h : f->vsHashes) {
      if (h == vsHash) return true;
    }
    return false;
  }

  // firstIndex/baseVertex are the draw's own range arguments. They are NOT part
  // of the key (the join's other side cannot reconstruct them -- see Counts) and
  // are used only to count how many distinct draw ranges collide on this key.
  inline void recordD3D11Draw(uint64_t vsHash, uint32_t indexCount,
                              uint32_t firstIndex, int32_t baseVertex) {
    // Can only filter on the shader here: primCount is not what
    // traceVertexCounts holds, so applying that set would reject everything.
    const auto f = loadFilter();
    if (f->vsHashes.empty()) {
      return;
    }
    bool hit = false;
    for (const uint64_t h : f->vsHashes) { if (h == vsHash) { hit = true; break; } }
    if (!hit) {
      return;
    }
    const uint64_t k = makeKey(vsHash, indexCount);
    // baseVertex is signed and may legitimately be negative; cast through the
    // unsigned type of the same width so the packing is lossless either way.
    const uint64_t variant = (static_cast<uint64_t>(firstIndex) << 32)
                           |  static_cast<uint64_t>(static_cast<uint32_t>(baseVertex));
    std::lock_guard<std::mutex> lock(tableMutex());
    Counts& c = drawTable()[k];
    c.vsHash = vsHash;
    c.vertexCount = indexCount;  // holds the raw index count in this table
    ++c.stage[static_cast<size_t>(Stage::Submitted)];
    bool knownVariant = false;
    for (uint32_t i = 0; i < c.variantCount; ++i) {
      if (c.variantIds[i] == variant) { knownVariant = true; break; }
    }
    if (!knownVariant) {
      if (c.variantCount < Counts::kMaxVariants) {
        c.variantIds[c.variantCount++] = variant;
      } else {
        c.variantOverflow = true;
      }
    }
  }

  // Single accessor for the stage-0 table so every reader gets the count and its
  // attribution confidence together. Returns a zeroed DrawAttrib when the key is
  // absent, which is itself meaningful: the game issued no traced draw for it.
  inline DrawAttrib lookupDraw(const Table& t, uint64_t key) {
    DrawAttrib a;
    const auto it = t.find(key);
    if (it != t.end()) {
      a.draws    = it->second.stage[static_cast<size_t>(Stage::Submitted)];
      a.variants = it->second.variantCount;
      a.overflow = it->second.variantOverflow;
    }
    return a;
  }

  inline void snapshotDrawsAndClear(Table& out) {
    std::lock_guard<std::mutex> lock(tableMutex());
    out = drawTable();
    drawTable().clear();
  }

  inline const char* stageName(Stage s) {
    switch (s) {
    case Stage::Submitted:    return "Submitted";
    case Stage::ProcessDcs:   return "ProcessDcs";
    case Stage::BlasReady:    return "BlasReady";
    case Stage::InstanceMade: return "InstanceMade";
    case Stage::InstanceNull: return "InstanceNull";
    case Stage::Batched:         return "Batched";
    case Stage::BatchEmitted:    return "BatchEmitted";
    case Stage::GeometryDerived: return "GeometryDerived";
    default:                  return "?";
    }
  }

}
