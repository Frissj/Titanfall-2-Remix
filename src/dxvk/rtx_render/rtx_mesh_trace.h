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
    kCount
  };

  struct Counts {
    uint32_t stage[static_cast<size_t>(Stage::kCount)] = {};
    uint64_t submitMat = 0;   // material hash as seen at submission time
    uint64_t vsHash = 0;
    uint32_t vertexCount = 0;
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
  inline Table& drawTable() {
    static Table t;
    return t;
  }

  inline void recordD3D11Draw(uint64_t vsHash, uint32_t indexCount) {
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
    std::lock_guard<std::mutex> lock(tableMutex());
    Counts& c = drawTable()[k];
    c.vsHash = vsHash;
    c.vertexCount = indexCount;  // holds the raw index count in this table
    ++c.stage[static_cast<size_t>(Stage::Submitted)];
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
    default:                  return "?";
    }
  }

}
