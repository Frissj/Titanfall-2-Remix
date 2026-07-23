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

#include "rtx_types.h"
#include "rtx_options.h"
#include "rtx_terrain_baker.h"
#include "rtx_instance_manager.h"
#include "rtx_light_manager.h"
#include "graph/rtx_graph_instance.h"
#include "dxvk_scoped_annotation.h"

#include <mutex>
#include <unordered_set>
#include <cmath>

namespace dxvk {

  // NV-DXVK [BonePalOob probe]: see BonePalette::operator[] in rtx_types.h.
  std::atomic<uint64_t> g_bonePaletteOobReads { 0u };

  // Instance constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtInstance* instance) : m_type(Type::Instance) {
    m_ptr.instance = instance;
  }
  RtInstance* PrimInstance::getInstance() const {
    if (m_type != Type::Instance) {
      return nullptr;
    }
    return m_ptr.instance;
  }

  // Light constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtLight* light) : m_type(Type::Light) {
    m_ptr.light = light;
  }
  RtLight* PrimInstance::getLight() const {
    if (m_type != Type::Light) {
      return nullptr;
    }
    return m_ptr.light;
  }

  // Graph constructor, getter, and assignment operator
  PrimInstance::PrimInstance(GraphInstance* graph) : m_type(Type::Graph) {
    m_ptr.graph = graph;
  }
  GraphInstance* PrimInstance::getGraph() const {
    if (m_type != Type::Graph) {
      return nullptr;
    }
    return m_ptr.graph;
  }

  PrimInstance::Type PrimInstance::getType() const {
    if (m_ptr.untyped == nullptr) {
      return Type::None;
    }
    return m_type;
  }

  PrimInstance::PrimInstance(void* owner, Type type) : m_type(type) {
    m_ptr.untyped = owner;
  }

  void* PrimInstance::getUntyped() const {
    return m_ptr.untyped;
  }

  void PrimInstance::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex) {
    PrimInstanceOwner* prim = nullptr;
    if (m_type == Type::Instance) {
      prim = &m_ptr.instance->getPrimInstanceOwner();
    } else if (m_type == Type::Light) {
      prim = &m_ptr.light->getPrimInstanceOwner();
    } else if (m_type == Type::Graph) {
      prim = &m_ptr.graph->getPrimInstanceOwner();
    }

    if (prim) {
      prim->setReplacementInstance(replacementInstance, replacementIndex, m_ptr.untyped, m_type);
    }
  }

  std::ostream& operator << (std::ostream& os, PrimInstance::Type type) {
    switch (type) {
      ENUM_NAME(PrimInstance::Type::Instance);
      ENUM_NAME(PrimInstance::Type::Light);
      ENUM_NAME(PrimInstance::Type::Graph);
      ENUM_NAME(PrimInstance::Type::None);
    }
    return os << static_cast<uint8_t>(type);
  }

  std::ostream& operator << (std::ostream& os, FogMode mode) {
    switch (mode) {
      ENUM_NAME(FogMode::None);
      ENUM_NAME(FogMode::Exp);
      ENUM_NAME(FogMode::Exp2);
      ENUM_NAME(FogMode::Linear);
    }
    return os << static_cast<uint32_t>(mode);
  }

  ReplacementInstance* PrimInstanceOwner::getOrCreateReplacementInstance(void* owner, PrimInstance::Type type, size_t index,size_t numPrims) {
    if (m_replacementInstance != nullptr && !isRoot(owner)) {
      // This Prim is already a non-root member of another replacementInstance, but SceneManager is trying to use it as the root of a new replacement.
      ONCE(assert(false && "getOrCreateReplacementInstance should only be called on root prims"));
      // Try to handle it gracefully anyways by removing this from the previous replacement and making it the root of a new replacement.
      // This will cause m_replacmementInstance to be null, so it will enter the new ReplacementInstance case below.
      setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex, owner, type);
    }

    if (m_replacementInstance == nullptr) {
      ReplacementInstance* replacement = new ReplacementInstance();
      replacement->setup(PrimInstance(owner, type), numPrims);
      setReplacementInstance(replacement, index, owner, type);
    } else if (m_replacementInstance->prims.size() != numPrims) {
      // Number of prims changing generally means a new replacement asset has loaded in.
      // Need to unlink the old instances, and either re-link them (if they are returned as 
      // similar by findSimilarInstances) or create new ones.
      ReplacementInstance* replacement = m_replacementInstance;
      
      // Clear the root manually, so that `clear()` doesn't try to delete the replacement.
      replacement->root = PrimInstance();
      // Clear the link to this prim so that it doesn't get marked for GC.
      setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex, owner, type);

      // Wipe out all the links in the replacement, which will mark all of the old non-root replacements for cleanup.
      replacement->clear();
      
      // Redo replacement setup.
      replacement->setup(PrimInstance(owner, type), numPrims);
      setReplacementInstance(replacement, index, owner, type);
    }
    return m_replacementInstance;
  }

  ReplacementInstance::~ReplacementInstance() {
    root = PrimInstance();
    clear();
  }

  void ReplacementInstance::clear() {
    // clear up all references to this ReplacementInstance.
    for (size_t i = 0; i < prims.size(); i++) {
      RtInstance* subInstance = prims[i].getInstance();
      if (subInstance) {
        subInstance->markForGarbageCollection();
      }
      GraphInstance* graphInstance = prims[i].getGraph();
      if (graphInstance) {
        graphInstance->removeInstance();
      }
      RtLight* light = prims[i].getLight();
      if (light) {
        light->markForGarbageCollection();
      }
      prims[i].setReplacementInstance(nullptr, kInvalidReplacementIndex);
    }
  }

  void ReplacementInstance::setup(PrimInstance newRoot, size_t numPrims) {
    prims.resize(numPrims);
    root = newRoot;
  }
  
  bool PrimInstanceOwner::isRoot(const void* owner) const {
    return m_replacementInstance != nullptr
      && m_replacementIndex != ReplacementInstance::kInvalidReplacementIndex
      && m_replacementInstance->root.getUntyped() == owner;
  }

  void PrimInstanceOwner::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex, void* owner, PrimInstance::Type type) {
    // Early out if this is just re-applying the same values.
    if (m_replacementInstance != nullptr && m_replacementInstance == replacementInstance) {
      ONCE(assert(false && "single prim is being set to multiple replacement indices."));
      return;
    }

    // Then check if the owner is already in a replacement:
    if (m_replacementInstance && m_replacementIndex != ReplacementInstance::kInvalidReplacementIndex) {
      
      // Inside a replacement, check if it's the root:
      if (isRoot(owner)) {
        // This is the root of a replacement being deleted.
        // Clear the root, and delete the replacementInstance.
        // the ReplacementInstance destructor will call this function again, which will
        // actually clear m_replacementInstance and m_replacementIndex.
        delete m_replacementInstance;
        m_replacementInstance = nullptr;
        m_replacementIndex = ReplacementInstance::kInvalidReplacementIndex;
        return;
      }

      // Next, remove the prim from the replacementInstance.
      PrimInstance& prim = m_replacementInstance->prims[m_replacementIndex];
      if (prim.getType() == type && prim.getUntyped() == owner) {
        // clear up the old reference to this owner
        prim = PrimInstance();
      } else {
        // The prim believed it was in a slot, but something else was actually there.
        // This is a sign that something went wrong earlier, but shouldn't cause problems itself.
        ONCE(assert(false && "PrimInstance was not properly removed from its replacementInstance before something else took its place."));
      }
    }

    // Set this owner to the new replacementInstance.
    m_replacementInstance = replacementInstance;
    m_replacementIndex = replacementIndex;

    // Inform the replacementInstance that this owner is now in it.
    if (m_replacementInstance && replacementIndex != ReplacementInstance::kInvalidReplacementIndex) {
      PrimInstance& prim = m_replacementInstance->prims[replacementIndex];
      if (prim.getType() != type && prim.getType() != PrimInstance::Type::None) {
        // While specific pointers may change, the type of a slot should never change.
        ONCE(assert(false && "Trying to assign a primInstance to a replacementInstance slot that was not the same type."));
        m_replacementInstance = nullptr;
        m_replacementIndex = ReplacementInstance::kInvalidReplacementIndex;
        return;
      } else if (prim.getUntyped() != nullptr && prim.getUntyped() != owner) {
        // Another owner is already in this spot.  Clean that up properly before overriding it.
        if (m_replacementInstance->root.getUntyped() == prim.getUntyped()) {
          // Replacing the old root.  Shouldn't happen, but if it does we would want to
          // update the root before clearing the old root, to avoid triggering garbage collection.
          m_replacementInstance->root = PrimInstance(owner, type);
        }
        prim.setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex);
        ONCE(assert(false && "PrimInstance was not properly cleaned up before being replaced."));
      }
      prim = PrimInstance(owner, type);
    }
  }

  uint32_t RasterGeometry::calculatePrimitiveCount() const {
    const uint32_t elementCount = usesIndices() ? indexCount : vertexCount;
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return elementCount / 3;

    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return elementCount >= 3
        ? elementCount - 2
        : 0;

    default:
      assert(!"Unsupported primitive topology");
      return UINT32_MAX;
    }
  }

  bool DrawCallState::finalizePendingFutures(const RtCamera* pLastCamera) {
    ScopedCpuProfileZone();
    // Geometry hashes are vital, and cannot be disabled, so its important we get valid data (hence the return type)
    const bool valid = finalizeGeometryHashes();
    if (valid) {
      // Bounding boxes (if enabled) will be finalized here, default is FLT_MAX bounds
      finalizeGeometryBoundingBox();

      // Skinning processing will be finalized here, if object requires skinning
      finalizeSkinningData(pLastCamera);

      // Update any categories that require geometry hash
      setupCategoriesForGeometry();

      // [SpawnGeomDiag.FinalCats] At the end of finalisation, log the
      // category flags actually set on this DrawCallState along with
      // the source texture/material hash + vsHash so we can identify
      // every draw (especially the floor's draw) and see what
      // routing-altering tags it has BEFORE it reaches submitDrawState
      // / addBlas. Throttled to one log per (textureHash, vsHash)
      // tuple so log volume stays bounded.
      // NOTE: I'm intentionally NOT calling setupCategoriesForTexture()
      // here — that function is dead code in this branch (zero call
      // sites) and re-enabling it would change runtime semantics, not
      // just diagnostics. We surface the existing state instead.
      {
        const uint64_t textureHash = static_cast<uint64_t>(
          getMaterialData().getColorTexture().getImageHash());
        const uint64_t vsHi = static_cast<uint64_t>(
          getTransformData().vertexShaderHash);
        static std::mutex sFinalCatLogMu;
        static std::unordered_set<uint64_t> sFinalCatLogSeen;
        const uint64_t key = textureHash ^ (vsHi + 0x9e3779b97f4a7c15ull);
        bool first = false;
        {
          std::lock_guard<std::mutex> lk(sFinalCatLogMu);
          first = sFinalCatLogSeen.insert(key).second;
        }
        if (first) {
          std::string tags;
          auto add = [&](InstanceCategories c, const char* name) {
            if (categories.test(c)) { tags += name; tags += ","; }
          };
          add(InstanceCategories::Hidden,           "Hidden");
          add(InstanceCategories::Ignore,           "Ignore");
          add(InstanceCategories::Sky,              "Sky");
          add(InstanceCategories::Terrain,          "Terrain");
          add(InstanceCategories::Particle,         "Particle");
          add(InstanceCategories::WorldUI,          "WorldUI");
          add(InstanceCategories::WorldMatte,       "WorldMatte");
          add(InstanceCategories::DecalStatic,      "DecalStatic");
          add(InstanceCategories::DecalDynamic,     "DecalDynamic");
          add(InstanceCategories::DecalSingleOffset,"DecalSingleOffset");
          add(InstanceCategories::DecalNoOffset,    "DecalNoOffset");
          add(InstanceCategories::ThirdPersonPlayerModel, "PlayerModel");
          add(InstanceCategories::ThirdPersonPlayerBody,  "PlayerBody");
          add(InstanceCategories::Beam,             "Beam");
          add(InstanceCategories::ParticleEmitter,  "Emitter");
          add(InstanceCategories::AnimatedWater,    "AnimWater");
          add(InstanceCategories::IgnoreLights,         "IgnoreLights");
          add(InstanceCategories::IgnoreAlphaChannel,   "IgnoreAlpha");
          add(InstanceCategories::IgnoreBakedLighting,  "IgnoreBaked");
          add(InstanceCategories::IgnoreAntiCulling,    "IgnoreAntiCull");
          add(InstanceCategories::IgnoreMotionBlur,     "IgnoreMotionBlur");
          add(InstanceCategories::IgnoreOpacityMicromap,"IgnoreOMM");
          add(InstanceCategories::IgnoreTransparencyLayer, "IgnoreXparent");
          if (tags.empty()) tags = "none";
          else tags.pop_back();
          Logger::info(str::format(
            "[SpawnGeomDiag.FinalCats] textureHash=0x", std::hex, textureHash, std::dec,
            " vsHash=0x", std::hex, vsHi, std::dec,
            " primaryHash=0x", std::hex,
            static_cast<uint64_t>(getMaterialData().getHash()), std::dec,
            " cats=", tags));
        }
      }

      return true;
    }

    return false;
  }

  bool DrawCallState::isEye() const {
    if (RtxOptions::Eye::enable() && RtxOptions::Eye::assumeViewTexgenModeAsEye()) {
      return getTransformData().texgenMode == TexGenMode::ViewPositions;
    }
    return false;
  }

  bool DrawCallState::finalizeGeometryHashes() {
    if (!geometryData.futureGeometryHashes.valid()) {
      return false;
    }

    geometryData.hashes = geometryData.futureGeometryHashes.get();

    if (geometryData.hashes[HashComponents::VertexPosition] == kEmptyHash) {
      throw DxvkError("Position hash should never be empty");
    }

    return true;
  }

  void DrawCallState::finalizeGeometryBoundingBox() {
    if (geometryData.futureBoundingBox.valid())
      geometryData.boundingBox = geometryData.futureBoundingBox.get();

    // [SpikeBBox] s2s hull-trim "spikes": the bbox future resolves HERE, AFTER
    // SubmitDraw — which is why d3d11_rtx's [SpikeGeo]/[PickGeo] saw it invalid.
    // dxvk computes this OBJECT-space bbox by reading the draw's actual verts, so
    // a spike vertex blows it out. Gated on the two spike VS hashes (0x29566a60
    // s2s_metal_trims_01, 0x29a262 s2s_wall_trim_03); deduped per drawCallID
    // (cap 64). bbExt huge => a spike vertex IS in the geometry; bbExt normal
    // (trim-piece sized) => the "spike" is shading, not geometry.
    {
      const uint64_t vs = static_cast<uint64_t>(transformData.vertexShaderHash);
      if ((vs == 0x29566a60d473af50ull || vs == 0x29a262d2e574b21cull)
          && geometryData.boundingBox.isValid()) {
        static std::mutex s_sbMu; static std::unordered_set<uint32_t> s_sbSeen;
        bool fresh = false;
        { std::lock_guard<std::mutex> g(s_sbMu);
          if (s_sbSeen.size() < 64u && s_sbSeen.insert(drawCallID).second) fresh = true; }
        if (fresh) {
          const auto& bb = geometryData.boundingBox;
          const Vector3 ext = bb.maxPos - bb.minPos;
          Logger::warn(str::format(
            "[SpikeBBox] drawId=", drawCallID, " VS=0x", std::hex, vs, std::dec,
            " verts=", geometryData.vertexCount,
            " bbExt=", (ext.x + ext.y + ext.z),
            " bbMin=(", bb.minPos.x, ",", bb.minPos.y, ",", bb.minPos.z, ")",
            " bbMax=(", bb.maxPos.x, ",", bb.maxPos.y, ",", bb.maxPos.z, ")"));
        }
      }
    }
  }

  void DrawCallState::finalizeSkinningData(const RtCamera* pLastCamera) {
    // NV-DXVK start: Support both async (futureSkinningData) and pre-populated skinningData paths.
    // The D3D11 layer populates skinningData directly when capturing per-vertex bone data.
    if (futureSkinningData.valid()) {
      skinningData = futureSkinningData.get();
    }

    // Process skinning if we have bone data (from either path)
    if (skinningData.numBones > 0 && geometryData.blendWeightBuffer.defined()) {
      assert(skinningData.numBonesPerVertex <= 4);

      if (pLastCamera != nullptr) {
        // NV-DXVK [WidowBake]: save the draw's OWN worldToView before this
        // block overwrites it from pLastCamera (below). [WidowO2W] proved the
        // Widow leaves SubmitDraw with o2w=IDENTITY and a correct w2v, yet
        // [ShipBake] sees o2w teleported — because the rebuild here uses
        // pLastCamera, which on camera-motion frames is a STALE/WRONG camera.
        // This probe dumps the draw's own w2v vs pLastCamera's w2v vs the
        // resulting o2w, so we can confirm the mismatch at its injection point.
        const Matrix4 widowBakeDrawW2v = transformData.worldToView;
        const auto fusedMode = RtxOptions::fusedWorldViewMode();
        if (likely(fusedMode == FusedWorldViewMode::None)) {
          transformData.objectToView = transformData.worldToView;
          // Do not bother when transform is fused. Camera matrices are identity and so is worldToView.
        }
        bool usedDrawCamera = false;
        if (RtxOptions::tf2SkinnedUseDrawCamera()) {
          // NV-DXVK [StudioModelHook fix]: un-fuse the skinned objectToWorld
          // against the DRAW'S OWN worldToView — the camera this draw was
          // actually rendered with — instead of pLastCamera (the global last-
          // set / engine-hook Main). For normal titles the two are identical
          // so this is a no-op; under TF2's single-global engine-hook camera
          // they diverge and the Main-based decompose teleports the skinned
          // world-space BLAS (proven by [WidowCam]/[WidowBake]: the draw's own
          // camera isn't even registered as an RtCamera). Using the draw's own
          // camera is a mathematical identity — o2w = inverse(drawW2v) *
          // (drawW2v * trueWorld) = trueWorld — so the world placement is exact.
          //
          // SAFETY: inverse() of a degenerate/uninitialized worldToView (det~0)
          // yields NaN/inf. A NaN objectToWorld poisons the BLAS and triggers a
          // GPU device-loss (freeze/crash). So compute the candidate, and only
          // accept it if EVERY element is finite; otherwise fall through to the
          // pLastCamera path (which is always a valid camera → always finite).
          const Matrix4 candidateO2w =
            inverse(transformData.worldToView) * transformData.objectToView;
          bool finite = true;
          for (int c = 0; c < 4 && finite; ++c)
            for (int r = 0; r < 4 && finite; ++r)
              if (!std::isfinite(candidateO2w[c][r])) finite = false;
          if (finite) {
            transformData.objectToWorld = candidateO2w;
            // worldToView is left as the draw's own (NOT overwritten).
            usedDrawCamera = true;
          } else {
            // [StudioNaN] ROOT-CAUSE probe: the candidate o2w came out non-
            // finite. Dump the actual inputs so we can see WHY, not just THAT
            // it happened. Distinguishes the cases:
            //   w2vFinite=0            -> worldToView is ALREADY NaN (bug is
            //                            upstream in ExtractTransforms, not here)
            //   o2vFinite=0            -> objectToView is NaN (upstream)
            //   both finite, det3~0    -> worldToView is SINGULAR (rank-deficient
            //                            / all-zero rotation) -> inverse()=inf
            // Plus the draw identity (name/vtx/numBones/fusedMode) so we know
            // WHICH draws produce it. Capped at 30 distinct samples.
            static uint32_t s_studioNanN = 0;
            if (s_studioNanN < 30u) {
              ++s_studioNanN;
              const Matrix4& w2v = transformData.worldToView;
              const Matrix4& o2v = transformData.objectToView;
              auto isFin = [](const Matrix4& m) {
                for (int c = 0; c < 4; ++c)
                  for (int r = 0; r < 4; ++r)
                    if (!std::isfinite(m[c][r])) return false;
                return true; };
              // determinant of the 3x3 rotation block (columns 0,1,2)
              const Vector3 wc0(w2v[0][0], w2v[0][1], w2v[0][2]);
              const Vector3 wc1(w2v[1][0], w2v[1][1], w2v[1][2]);
              const Vector3 wc2(w2v[2][0], w2v[2][1], w2v[2][2]);
              const float det3 = dot(cross(wc0, wc1), wc2);
              Logger::warn(str::format(
                "[StudioNaN] n=", s_studioNanN,
                " name=", (studioModelName[0] ? studioModelName : "(none)"),
                " isWidow=", (isWidowModel ? 1 : 0),
                " vtx=", geometryData.vertexCount,
                " numBones=", skinningData.numBones,
                " fusedMode=", static_cast<uint32_t>(RtxOptions::fusedWorldViewMode()),
                " w2vFinite=", (isFin(w2v) ? 1 : 0),
                " o2vFinite=", (isFin(o2v) ? 1 : 0),
                " det3(w2v)=", det3,
                " w2vT=(", w2v[3][0], ",", w2v[3][1], ",", w2v[3][2], ")",
                " w2v_r0=(", w2v[0][0], ",", w2v[0][1], ",", w2v[0][2], ")",
                " w2v_r1=(", w2v[1][0], ",", w2v[1][1], ",", w2v[1][2], ")",
                " w2v_r2=(", w2v[2][0], ",", w2v[2][1], ",", w2v[2][2], ")"));
            }
          }
        }
        if (!usedDrawCamera) {
          transformData.objectToWorld = pLastCamera->getViewToWorld(false) * transformData.objectToView;
          transformData.worldToView = pLastCamera->getWorldToView(false);
        }

        // NV-DXVK [WidowBake] consumer: raw per-call dump for the Widow
        // (by-model tag), capped to bound volume. drawW2vT = the draw's own
        // (correct) camera; camW2vT = pLastCamera (the camera actually used);
        // a mismatch IS the teleport. Requires rtx.tf2DetectWidow.
        if (isWidowModel) {
          static uint32_t s_widowBakeN = 0;
          if (s_widowBakeN < 240u) {
            ++s_widowBakeN;
            const Matrix4& o2w = transformData.objectToWorld;
            const Matrix4& o2v = transformData.objectToView;
            Logger::info(str::format(
              "[WidowBake] n=", s_widowBakeN,
              " name=", (studioModelName[0] ? studioModelName : "(none)"),
              " numBones=", skinningData.numBones,
              " drawW2vT=(", widowBakeDrawW2v[3][0], ",", widowBakeDrawW2v[3][1], ",", widowBakeDrawW2v[3][2], ")",
              " camW2vT=(", transformData.worldToView[3][0], ",", transformData.worldToView[3][1], ",", transformData.worldToView[3][2], ")",
              " o2vT=(", o2v[3][0], ",", o2v[3][1], ",", o2v[3][2], ")",
              " o2wT=(", o2w[3][0], ",", o2w[3][1], ",", o2w[3][2], ")"));
          }
        }
      } else {
        ONCE(Logger::warn("[RTX-Compatibility-Warn] Cannot decompose the matrices for a skinned mesh because the camera is not set."));
      }

      // In rare cases when the mesh is skinned but has only one active bone, skip the skinning pass
      // and bake that single bone into the objectToWorld/View matrices.
      if (skinningData.minBoneIndex + 1 == skinningData.numBones) {
        const Matrix4& skinningMatrix = skinningData.pBoneMatrices[skinningData.minBoneIndex];

        transformData.objectToWorld = transformData.objectToWorld * skinningMatrix;
        transformData.objectToView = transformData.objectToView * skinningMatrix;

        skinningData.boneHash = 0;
        skinningData.numBones = 0;
        skinningData.numBonesPerVertex = 0;
      }

      // Store the numBonesPerVertex in the RasterGeometry as well to allow it to be overridden
      geometryData.numBonesPerVertex = skinningData.numBonesPerVertex;
    }
    // NV-DXVK end
  }

  void DrawCallState::setCategory(InstanceCategories category, bool doSet) {
    if (doSet) {
      categories.set(category);
    }
  }

  void DrawCallState::removeCategory(InstanceCategories category) {
    categories.clr(category);
  }

  void DrawCallState::setupCategoriesForTexture() {
    // TODO (REMIX-231): It would probably be much more efficient to use a map of texture hash to category flags, rather
    //                   than doing N lookups per texture hash for each category.
    const XXH64_hash_t& textureHash = materialData.getColorTexture().getImageHash();

    setCategory(InstanceCategories::WorldUI, lookupHash(RtxOptions::worldSpaceUiTextures(), textureHash));
    setCategory(InstanceCategories::WorldMatte, lookupHash(RtxOptions::worldSpaceUiBackgroundTextures(), textureHash));

    setCategory(InstanceCategories::Ignore, lookupHash(RtxOptions::ignoreTextures(), textureHash));
    setCategory(InstanceCategories::IgnoreLights, lookupHash(RtxOptions::ignoreLights(), textureHash));
    setCategory(InstanceCategories::IgnoreAntiCulling, lookupHash(RtxOptions::antiCullingTextures(), textureHash));
    setCategory(InstanceCategories::IgnoreMotionBlur, lookupHash(RtxOptions::motionBlurMaskOutTextures(), textureHash));
    setCategory(InstanceCategories::IgnoreOpacityMicromap, lookupHash(RtxOptions::opacityMicromapIgnoreTextures(), textureHash) || isUsingRaytracedRenderTarget);
    setCategory(InstanceCategories::IgnoreAlphaChannel, lookupHash(RtxOptions::ignoreAlphaOnTextures(), textureHash));
    setCategory(InstanceCategories::IgnoreBakedLighting, lookupHash(RtxOptions::ignoreBakedLightingTextures(), textureHash));

    setCategory(InstanceCategories::Hidden, lookupHash(RtxOptions::hideInstanceTextures(), textureHash));

    setCategory(InstanceCategories::Particle, lookupHash(RtxOptions::particleTextures(), textureHash));
    setCategory(InstanceCategories::Beam, lookupHash(RtxOptions::beamTextures(), textureHash));
    setCategory(InstanceCategories::IgnoreTransparencyLayer, lookupHash(RtxOptions::ignoreTransparencyLayerTextures(), textureHash));

    setCategory(InstanceCategories::DecalStatic, lookupHash(RtxOptions::decalTextures(), textureHash));
    setCategory(InstanceCategories::DecalDynamic, lookupHash(RtxOptions::dynamicDecalTextures(), textureHash));
    setCategory(InstanceCategories::DecalSingleOffset, lookupHash(RtxOptions::singleOffsetDecalTextures(), textureHash));
    setCategory(InstanceCategories::DecalNoOffset, lookupHash(RtxOptions::nonOffsetDecalTextures(), textureHash));

    setCategory(InstanceCategories::AnimatedWater, lookupHash(RtxOptions::animatedWaterTextures(), textureHash));

    setCategory(InstanceCategories::ThirdPersonPlayerModel, lookupHash(RtxOptions::playerModelTextures(), textureHash));
    setCategory(InstanceCategories::ThirdPersonPlayerBody, lookupHash(RtxOptions::playerModelBodyTextures(), textureHash));

    setCategory(InstanceCategories::Terrain, lookupHash(RtxOptions::terrainTextures(), textureHash));
    // NV-DXVK [EngineCam-Skybox]: respect the master kill-switch for Sky.
    // The texture-hash lookup path is independent of shouldBakeSky() and
    // would otherwise re-tag draws whose textures appear in the legacy
    // skyBoxTextures list (which Remix populates from rtx.conf or
    // captured-asset metadata). Gate it on the same option.
    setCategory(InstanceCategories::Sky,
                !RtxOptions::disableSkyTagging()
                && lookupHash(RtxOptions::skyBoxTextures(), textureHash));

    setCategory(InstanceCategories::ParticleEmitter, lookupHash(RtxOptions::particleEmitterTextures(), textureHash));

    // [SpawnGeomDiag.CatFlags] Log the texture-driven category result
    // ONCE per (textureHash, vsHash) tuple so we can correlate "the
    // missing floor" (worldspawn__008__world_vr_training_vr_marble_floor)
    // against whatever categorisation Remix is applying to its draw
    // calls. If the floor's draw call ends up with Hidden / Ignore /
    // WorldUI / Particle / Decal*, that's why it's absent from the
    // BLAS-stage scene OBJ — it was filtered out at categorisation
    // time before ever reaching addBlas / addPointInstancerBlas.
    {
      static std::mutex sCatLogMu;
      static std::unordered_set<uint64_t> sCatLogSeen;
      const uint64_t vsHi = static_cast<uint64_t>(
        getTransformData().vertexShaderHash);
      const uint64_t key =
        (static_cast<uint64_t>(textureHash) ^
         (vsHi + 0x9e3779b97f4a7c15ull));
      bool first = false;
      {
        std::lock_guard<std::mutex> lk(sCatLogMu);
        first = sCatLogSeen.insert(key).second;
      }
      if (first) {
        // Only flag the categories that would alter pipeline routing.
        const bool hidden  = categories.test(InstanceCategories::Hidden);
        const bool ignore  = categories.test(InstanceCategories::Ignore);
        const bool sky     = categories.test(InstanceCategories::Sky);
        const bool terrain = categories.test(InstanceCategories::Terrain);
        const bool particle= categories.test(InstanceCategories::Particle);
        const bool worldUi = categories.test(InstanceCategories::WorldUI);
        const bool worldMatte = categories.test(InstanceCategories::WorldMatte);
        const bool decalS  = categories.test(InstanceCategories::DecalStatic);
        const bool decalD  = categories.test(InstanceCategories::DecalDynamic);
        const bool decalSO = categories.test(InstanceCategories::DecalSingleOffset);
        const bool decalNO = categories.test(InstanceCategories::DecalNoOffset);
        const bool ppm     = categories.test(InstanceCategories::ThirdPersonPlayerModel);
        const bool ppb     = categories.test(InstanceCategories::ThirdPersonPlayerBody);
        const bool beam    = categories.test(InstanceCategories::Beam);
        const bool emitter = categories.test(InstanceCategories::ParticleEmitter);
        const bool animw   = categories.test(InstanceCategories::AnimatedWater);
        const bool ignoreA = categories.test(InstanceCategories::IgnoreAlphaChannel);
        const bool ignoreB = categories.test(InstanceCategories::IgnoreBakedLighting);
        const bool ignoreL = categories.test(InstanceCategories::IgnoreLights);
        // Build a compact comma-separated tag list of categories actually set.
        std::string tags;
        if (hidden)     tags += "Hidden,";
        if (ignore)     tags += "Ignore,";
        if (sky)        tags += "Sky,";
        if (terrain)    tags += "Terrain,";
        if (particle)   tags += "Particle,";
        if (worldUi)    tags += "WorldUI,";
        if (worldMatte) tags += "WorldMatte,";
        if (decalS)     tags += "DecalStatic,";
        if (decalD)     tags += "DecalDynamic,";
        if (decalSO)    tags += "DecalSingleOffset,";
        if (decalNO)    tags += "DecalNoOffset,";
        if (ppm)        tags += "PlayerModel,";
        if (ppb)        tags += "PlayerBody,";
        if (beam)       tags += "Beam,";
        if (emitter)    tags += "Emitter,";
        if (animw)      tags += "AnimWater,";
        if (ignoreA)    tags += "IgnoreAlpha,";
        if (ignoreB)    tags += "IgnoreBaked,";
        if (ignoreL)    tags += "IgnoreLights,";
        if (tags.empty()) tags = "none";
        else tags.pop_back(); // drop trailing comma
        Logger::info(str::format(
          "[SpawnGeomDiag.CatFlags] textureHash=0x", std::hex, textureHash, std::dec,
          " vsHash=0x", std::hex, vsHi, std::dec,
          " cats=", tags,
          " primaryHash=0x", std::hex,
          static_cast<uint64_t>(getMaterialData().getHash()), std::dec));
      }
    }
  }

  void DrawCallState::setupCategoriesForGeometry() {
    const XXH64_hash_t assetReplacementHash = getHash(RtxOptions::geometryAssetHashRule());
    // NV-DXVK [EngineCam-Skybox]: same kill-switch as the texture-hash
    // sky path above. Geometry-hash based skyBoxGeometries lookup is yet
    // another independent classifier that would otherwise re-tag draws.
    setCategory(InstanceCategories::Sky,
                !RtxOptions::disableSkyTagging()
                && lookupHash(RtxOptions::skyBoxGeometries(), assetReplacementHash));
    // NV-DXVK [debug.hideVertexShaders]: hide draws by VERTEX-SHADER hash.
    // Placed here (a LIVE category function) rather than in
    // setupCategoriesForTexture(), which is dead code in this branch
    // (zero call sites — see note at the finalize site ~line 289), which is
    // also why rtx.hideInstanceTextures is a no-op here. vertexShaderHash is
    // populated by finalize time (the [SpawnGeomDiag.FinalCats] log reads it
    // right after this call). Used to hide multi-material geometry no single
    // texture identifies (e.g. the misplaced sub-view BSP plane).
    const bool hiddenByVs =
        lookupHash(RtxOptions::hideVertexShaders(), transformData.vertexShaderHash);
    // NV-DXVK: also honor rtx.hideInstanceTextures HERE. The original
    // setupCategoriesForTexture() path that would apply it is dead code (zero
    // call sites — see note above), so without this the option is a silent
    // no-op. Match the albedo (colorTextures[0]) IMAGE hash — the same hash the
    // option is documented to key on, and the one the on-screen albedo dump
    // writes as <hash>_albedo.dds — so a single shared VS can no longer hide an
    // object; you hide exactly the texture you name.
    const auto& albedoTex = getMaterialData().getColorTexture();
    const bool hiddenByTex = albedoTex.isValid()
        && lookupHash(RtxOptions::hideInstanceTextures(), albedoTex.getImageHash());
    setCategory(InstanceCategories::Hidden, hiddenByVs || hiddenByTex);
  }

  static std::optional<Vector3> makeCameraPosition(const Matrix4& worldToView,
                                                   bool zWrite,
                                                   bool alphaBlend,
                                                   bool hasSkinning) {
    if (hasSkinning) {
      return std::nullopt;
    }
    // particles
    if (!zWrite && alphaBlend) {
      return std::nullopt;
    }
    // identity matrix
    if (isIdentityExact(worldToView)) {
      return std::nullopt;
    }

#define USE_TRUE_CAMERA_POSITION_FOR_COMPARISON 0

#if USE_TRUE_CAMERA_POSITION_FOR_COMPARISON
    return (inverse(worldToView))[3].xyz();
#else
    // as we compare the cameras relatively and don't need precise camera position:
    // just return a position-like vector, to avoid calculating heavy matrix inverse operation
    return worldToView[3].xyz();
#endif
  }

  static bool areCamerasClose(const Vector3& a, const Vector3& b) {
    const float distanceThreshold = RtxOptions::skyAutoDetectUniqueCameraDistance();
    return lengthSqr(a - b) < distanceThreshold * distanceThreshold;
  }

  bool checkSkyAutoDetect(bool depthTestEnable,
                          const std::optional<Vector3>& newCameraPos,
                          uint32_t prevFrameSeenCamerasCount,
                          const std::vector<Vector3>& seenCameraPositions) {

    if (RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPositionAndDepthFlags &&
        RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPosition) {
      return false;
    }
    const bool withDepthFlags = (RtxOptions::skyAutoDetect() == SkyAutoDetectMode::CameraPositionAndDepthFlags);


    const bool searchingForSkyCamera             = (seenCameraPositions.size() == 0);
    const bool skyFoundAndSearchingForMainCamera = (seenCameraPositions.size() == 1);
    const bool skyAndMainCameraFound             = (seenCameraPositions.size() >= 2);

    if (skyAndMainCameraFound) {
      // assume that subsequent draw calls can not be sky
      return false;
    }

    if (searchingForSkyCamera) {
      if (withDepthFlags) {
        // no depth test: frame starts with a sky
        // depth test: frame starts with a world, not a sky
        return !depthTestEnable;
      }
      // assume the first camera to be sky
      return true;
    }

    {
      // corner case: if there was no sky camera at all, fallback, but this would also
      // involve a one-frame (preceding to the current one) being rasterized (like a flicker)
      if (prevFrameSeenCamerasCount < 2) {
        if (withDepthFlags) {
          // no depth test: sky
          // depth test: world
          return !depthTestEnable;
        }
        // assume no sky
        return false;
      }
    }

    if (skyFoundAndSearchingForMainCamera) {
      // if draw call doesn't have a camera position
      if (!newCameraPos) {
        // it can't contain main camera, so assume that it's still a sky
        return true;
      }

      // if same as the existing sky camera
      if (areCamerasClose(seenCameraPositions[0], *newCameraPos)) {
        // still sky
        return true;
      }

      // found a new unique camera, which should be a main camera
      return false;
    }

    assert(0);
    return false;
  }

  enum class SkyDetectionSource {
    None,
    Explicit,   // minZ, texHash, geoHash, dcIdThreshold
    AutoDetect  // checkSkyAutoDetect
  };

  SkyDetectionSource shouldBakeSky(const DrawCallState& drawCallState,
                     bool hasSkinning,
                     uint32_t prevFrameSeenCamerasCount,
                     std::vector<Vector3>& seenCameraPositions) {
    // NV-DXVK [EngineCam-Skybox]: master kill-switch for sky tagging.
    // d3d11_rtx's SetSkyCategoryFromCb2 has its own gate on this option,
    // but there are FOUR independent sky-classification paths in remix
    // (cb2-origin / minZ-threshold / texture-hash / drawcall-id / camera-
    // position autoDetect) and the user's intent ("3D-skybox geometry
    // should reach TLAS as ray-traced content") requires ALL of them to
    // stand down. Returning None from this top-level function short-
    // circuits the call site in setupCategoriesForHeuristics, which
    // means setCategory(Sky, false) — no draw is tagged regardless of
    // which subclassifier would have triggered.
    //
    // Other sky-related machinery (Hillaire atmosphere, sun NEE, env_sky
    // cubemap sampling in shaders) keeps working — those pull from
    // [EngineSun] cb2 captures and don't depend on per-draw sky tags.
    if (RtxOptions::disableSkyTagging()) {
      return SkyDetectionSource::None;
    }

    const auto drawCallCameraPos =
      drawCallState.isDrawingToRaytracedRenderTarget
        ? std::optional<Vector3>{}
        : makeCameraPosition(
            drawCallState.getTransformData().worldToView,
            drawCallState.zWriteEnable,
            drawCallState.getMaterialData().blendMode.enableBlending,
            hasSkinning);

    auto l_addIfUnique = [&seenCameraPositions](const std::optional<Vector3>& newCameraPos) {
      if (!newCameraPos) {
        return;
      }
      for (const Vector3& seen : seenCameraPositions) {
        if (areCamerasClose(seen, *newCameraPos)) {
          return;
        }
      }
      seenCameraPositions.push_back(*newCameraPos);
    };
    l_addIfUnique(drawCallCameraPos);


    if (drawCallState.minZ >= RtxOptions::skyMinZThreshold()) {
      return SkyDetectionSource::Explicit;
    }

    // NOTE: we use color texture hash for sky detection, however the replacement is hashed with
    // the whole legacy material hash (which, as of 12/9/2022, equals to color texture hash). Adding a check just in case.
    assert(drawCallState.getMaterialData().getColorTexture().getImageHash() == drawCallState.getMaterialData().getHash() && "Texture or material hash method changed!");

    if (drawCallState.getMaterialData().usesTexture()) {
      if (lookupHash(RtxOptions::skyBoxTextures(), drawCallState.getMaterialData().getHash())) {
        return SkyDetectionSource::Explicit;
      }
    } else {
      if (drawCallState.drawCallID < RtxOptions::skyDrawcallIdThreshold()) {
        return SkyDetectionSource::Explicit;
      }
    }

    // don't track camera positions for Raytraced Render Targets, as they are a different camera position from main view
    const static auto renderTargetCameraPositions = std::vector<Vector3>{};

    if (checkSkyAutoDetect(drawCallState.zEnable,
                           drawCallCameraPos,
                           prevFrameSeenCamerasCount,
                           drawCallState.isDrawingToRaytracedRenderTarget ? renderTargetCameraPositions : seenCameraPositions)) {
      return SkyDetectionSource::AutoDetect;
    }

    return SkyDetectionSource::None;
  }

  bool shouldBakeTerrain(const DrawCallState& drawCallState) {
    if (!TerrainBaker::needsTerrainBaking())
      return false;

    return lookupHash(RtxOptions::terrainTextures(), drawCallState.getMaterialData().getHash());
  }

  void DrawCallState::setupCategoriesForHeuristics(uint32_t prevFrameSeenCamerasCount,
                                                   std::vector<Vector3>& seenCameraPositions) {
    const SkyDetectionSource skySource = shouldBakeSky(*this,
                                                       futureSkinningData.valid() || skinningData.numBones > 0,
                                                       prevFrameSeenCamerasCount,
                                                       seenCameraPositions);
    setCategory(InstanceCategories::Sky, skySource != SkyDetectionSource::None);
    skyAutoDetected = (skySource == SkyDetectionSource::AutoDetect);

    setCategory(InstanceCategories::Terrain, shouldBakeTerrain(*this));
  }

  BlasEntry::BlasEntry(const DrawCallState& input_)
    : input(input_), m_spatialMap(RtxOptions::uniqueObjectDistance() * 2.f) {
      if (RtxOptions::uniqueObjectDistance() <= 0.f) {
        ONCE(Logger::err("rtx.uniqueObjectDistance must be greater than 0."));
      }
    }

  void BlasEntry::unlinkInstance(RtInstance* instance) {
    instance->removeFromSpatialCache();
    auto it = std::find(m_linkedInstances.begin(), m_linkedInstances.end(), instance);
    if (it != m_linkedInstances.end()) {
      // Swap & pop - faster than "erase", but doesn't preserve order, which is fine here.
      std::swap(*it, m_linkedInstances.back());
      m_linkedInstances.pop_back();
    } else {
      ONCE(Logger::err("Tried to unlink an instance, which was never linked!"));
    }
  }

  void BlasEntry::rebuildSpatialMap() {
    m_spatialMap.rebuild(RtxOptions::uniqueObjectDistance() * 2.f);
  }

} // namespace dxvk
