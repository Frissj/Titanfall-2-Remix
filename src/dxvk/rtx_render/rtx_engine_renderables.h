/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <vector>

#include "rtx_resident_scene.h"
#include "../../util/util_vector.h"

namespace dxvk {

  // ==========================================================================
  // NV-DXVK [RenderableEnum] -- RESIDENT_SCENE_PLAN sec 7 slice B.
  //
  // THE ENGINE'S OWN LIST OF WHAT EXISTS, read rather than hooked.
  //
  // ARCHITECTURE_OVERHAUL sec 1.2 calls this the architecture's foundation
  // stone and records rung 5 as unsolved because "sub_1801A8350's return is
  // post-cull". Both parts of that turned out to be answerable:
  //
  //   - 0x1A8350 is not a function on the shipped build. It is a label inside
  //     BuildRenderableRenderLists' body, 0x40 past the entry, which is why
  //     reading it as one made the site look unsolvable.
  //
  //   - The body's RETURN is post-cull, but its FIRST ARGUMENT is not. It is
  //     handed the whole registry and culls a copy of it into the caller's
  //     output buffer. We take the argument and never call the function, so
  //     the cull never happens on our path.
  //
  // WHY THIS IS A READ AND NOT A HOOK. Nothing is detoured, nothing is
  // patched, and no convar is set. The registry is a statically allocated
  // object in client.dll reached through a rip-relative lea, so the whole
  // mechanism is one resolved address plus three offsets. There is no
  // execution-order dependency to get wrong and nothing to uninstall, which
  // also means a resolution failure is inert rather than fatal.
  //
  // -----------------------------------------------------------------------
  // LAYOUT, read off the list builder. This is the ONLY place that states it.
  // -----------------------------------------------------------------------
  //
  //     +0x030  uint32    nWords       size of the allocation bitmask, in
  //                                    64-bit words. Capacity = nWords * 64.
  //     +0x038  uint64[]  allocMask    one bit per slot
  //     +0x1038 Entry[]   entries      stride 16:
  //                                       +0  IClientRenderable*  (null = free)
  //                                       +8  uint32              (unread)
  //                                       +12 uint32              flags
  //
  // The builder's "enumerate everything" arm walks exactly this and selects
  // every slot whose pointer is non-null, with no leaf test and no frustum
  // test. That arm is the existence list. We reproduce it directly.
  //
  // -----------------------------------------------------------------------
  // THE POINTER IS NEVER DEREFERENCED, AND THAT IS A HARD RULE
  // -----------------------------------------------------------------------
  // This runs on a DXVK thread while the game owns that memory. A renderable
  // can be destroyed between the moment we read its slot and any moment after,
  // so calling a virtual on it -- which is what the builder does at vtable+248
  // -- would be a use-after-free with a race attached. The pointer is read as
  // an OPAQUE 64-BIT TOKEN and nothing else: it is the identity, not an object.
  //
  // For the same reason the walk is tolerant rather than assertive. A torn read
  // during a registry resize yields a wrong count for one frame, which the
  // promotion gate below is exactly the right instrument to notice -- a source
  // that tears is a source whose listed= is not flat, and it will not promote.
  //
  // -----------------------------------------------------------------------
  // WHAT THIS DELIBERATELY DOES NOT DO
  // -----------------------------------------------------------------------
  // It does not retire anything. Being pre-cull BY CONSTRUCTION is an argument,
  // and sec 3.1 does not accept arguments -- it accepts 600 flat frames under a
  // fixed-position sweep, measured on the map under test. So the list is
  // published as a VisibilitySource, which may never assert dead, and the
  // evidence for promotion is accumulated alongside it. When the gate passes,
  // promote() hands back the ExistenceSource that invalidateAbsent() wants.
  // ==========================================================================
  class RenderableEnum {
  public:
    struct Stats {
      uint32_t capacity     = 0u;   // nWords * 64
      uint32_t listed       = 0u;   // non-null slots this frame
      uint32_t maxSlot      = 0u;   // highest occupied slot seen
      uint32_t reads        = 0u;   // frames the registry was read
      uint32_t readFailures = 0u;   // frames it was unreadable or implausible
      bool     resolved     = false;
    };

    RenderableEnum() = default;

    // One frame's read. `cameraPos` is the MAIN camera origin and is used only
    // to decide whether the promotion evidence may be extended: the gate is
    // defined over a FIXED position, so translation resets it. Rotation does
    // not, which is the whole point of a pitch-and-yaw sweep.
    void update(uint32_t frame, const Vector3& cameraPos);

    // Post-cull-safe view of the list. Always available.
    const VisibilitySource& visibility() const { return m_visible; }

    // Evidence, and the only door to an ExistenceSource.
    const ExistenceSourcePromotion& promotion() const { return m_promotion; }

    // NV-DXVK slice 2: THE PROMOTED LIST, or null. Promoted ONCE, the first
    // frame the gate reads flat on this map, and kept after the camera moves:
    // the sweep proves a property of the SOURCE (the registry is pre-cull), not
    // of one frame, and a death signal that only armed while the camera stood
    // still would retire nothing. Dropped with the scene (clear) and on any
    // read failure, so a new map, or a registry we can no longer vouch for,
    // must earn it again.
    //
    // Null on a frame where it cannot be trusted even though promoted: the
    // frame it was promoted (its list is not filled until the next walk) and a
    // frame whose listed= collapsed below half the previous one -- a torn read
    // during a registry resize would otherwise be read as a mass death.
    const ExistenceSource* existence() const {
      return (m_existence != nullptr && m_existenceUsable) ? m_existence.get() : nullptr;
    }
    uint32_t collapseSkips() const { return m_collapseSkips; }

    const Stats& stats() const { return m_stats; }
    void clear();

  private:
    // Resolves the registry base and validates the header. Returns 0 when the
    // symbol is unresolved or the structure fails its plausibility checks.
    // resolveNsOut: time spent in EngineSymbols::resolve alone (see Cost).
    // Non-const: the readable() probes are cached per resolved address.
    uintptr_t registryBase(uint32_t& capacityOut, uint64_t& resolveNsOut);

    // The address and word count the readable() probes last passed for; 0 =
    // nothing proven. See registryBase for why this may be reused.
    uintptr_t m_probedBase  = 0;
    uint32_t  m_probedWords = 0u;

    // NV-DXVK [perf] 2026-09-12: WHERE update() SPENDS ITS TIME. Replacing the
    // per-frame unordered_set with a reused vector did not move [Perf.GcInst]
    // enum= (1164-1244 us before, 1156-1711 us after), so allocation was not the
    // cost. Summed over the log window, printed as per-frame averages. If the
    // phases sum far below enum=, the time is outside update() entirely.
    struct Cost {
      uint64_t resolveNs = 0ull;   // EngineSymbols::resolve, the three-link chain
      uint64_t probeNs   = 0ull;   // header check + the readable() probes
      uint64_t walkNs    = 0ull;   // bitmask walk + note()
      uint64_t closeNs   = 0ull;   // endFrame(): sort + dedupe, both lists
      uint64_t totalNs   = 0ull;   // whole update() on the read path
      uint64_t maxNs     = 0ull;   // worst single update() in the window
      uint32_t frames    = 0u;
      // Full readable() probe passes this window. 1 after a (re)resolve, 0 in
      // steady state; climbing every window = the registry address or size is
      // moving and the cache never hits.
      uint32_t probes    = 0u;
    };
    Cost m_cost;

    VisibilitySource         m_visible { "client.RenderableRegistry" };
    ExistenceSourcePromotion m_promotion;
    std::unique_ptr<ExistenceSource> m_existence;
    bool                     m_existenceUsable = false;
    uint32_t                 m_lastListed = 0u;
    uint32_t                 m_collapseSkips = 0u;   // cumulative
    Stats                    m_stats;

    Vector3  m_lastCameraPos { 0.0f, 0.0f, 0.0f };
    bool     m_haveCameraPos = false;
    uint32_t m_lastLogFrame  = 0u;
  };

} // namespace dxvk
