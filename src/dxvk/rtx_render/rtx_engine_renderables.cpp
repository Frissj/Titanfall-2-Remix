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
#include "rtx_engine_renderables.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "rtx_engine_symbols.h"
#include "rtx_engine_symbols_tf2.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    // These three offsets, and nothing else, are what this feature knows about
    // client.dll internals. Documented once in rtx_engine_renderables.h and
    // named here so a layout change is a one-line diff rather than a hunt.
    constexpr uintptr_t kOffNWords    = 0x030;
    constexpr uintptr_t kOffAllocMask = 0x038;
    constexpr uintptr_t kOffEntries   = 0x1038;
    constexpr size_t    kEntryStride  = 16;

    // PLAUSIBILITY, NOT TASTE. nWords counts 64-bit words, so this ceiling is
    // 262,144 renderables: far above anything the engine holds, and far below
    // a value that would make the entry-array readability probe span nonsense.
    // A registry reading outside this is a registry we did not actually find,
    // and the right response is to disable rather than to walk it.
    constexpr uint32_t kMaxWords = 4096u;

    // A camera that has not translated further than this counts as fixed.
    //
    // The promotion gate is defined over a fixed POSITION while pitch and yaw
    // sweep. A player standing still still jitters by sub-unit amounts from
    // view-bob and interpolation, and treating that as movement would mean the
    // gate could never accumulate 600 consecutive frames on a real machine.
    // Calibrated against the sweep that produced the slice 1 reading: standing
    // still and rotating spanned 10 units of X over 90 seconds, so this is
    // loose enough to absorb bob and far tighter than a walking step.
    constexpr float kFixedPosEpsilon = 16.0f;

    // Index of the lowest set bit. Written out rather than pulled from
    // <bit> because this file is compiled at the project baseline standard
    // and a countr_zero here is not worth raising it for.
    inline uint32_t lowestSetBit(uint64_t v) {
      uint32_t n = 0u;
      while ((v & 1ull) == 0ull) {
        v >>= 1;
        ++n;
      }
      return n;
    }
  }

  uintptr_t RenderableEnum::registryBase(uint32_t& capacityOut, uint64_t& resolveNsOut) {
    capacityOut = 0u;

    const auto tResolve = std::chrono::steady_clock::now();
    const uintptr_t base = EngineSymbols::resolve(tf2sym::kRenderableRegistry);
    resolveNsOut = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - tResolve).count());
    if (base == 0) {
      // Unresolved, or client.dll went away: whatever comes back next is a new
      // mapping and must be probed from scratch.
      m_probedBase  = 0;
      m_probedWords = 0u;
      return 0;   // unresolved; the resolver has already logged it once
    }

    // NV-DXVK [perf] 2026-09-12: THE PROBES ARE PAID ONCE PER MAPPING, NOT PER
    // FRAME. cost{} billed the three readable() calls at probe=1699-1807 us of
    // total=1813-1923 us per frame: VirtualQuery is not a cheap call on an
    // image section, and the walk it guards costs 42 us. What the probes prove
    // cannot change between frames while the address does not: the registry
    // is a static object inside client.dll's image (the resolver finds it
    // through a lea), so its page protection is the module mapping's, and the
    // mapping's lifetime is exactly what resolve() re-checks every frame -- it
    // returns 0 on unload and a new address on a new generation, and either
    // one drops this cache. A changed nWords re-probes the spans it sizes.
    const bool known = (base == m_probedBase);

    // Header first. If this is not readable, the address is not what we think
    // it is and every offset below would be a guess on top of a guess.
    if (!known) {
      m_probedBase  = 0;
      m_probedWords = 0u;
      m_cost.probes += 1u;
      if (!EngineSymbols::readable(reinterpret_cast<const void*>(base + kOffNWords),
                                  sizeof(uint32_t)))
        return 0;
    }

    const uint32_t nWords = *reinterpret_cast<const uint32_t*>(base + kOffNWords);
    if (nWords == 0u || nWords > kMaxWords)
      return 0;

    const uint32_t capacity = nWords * 64u;

    // Both spans probed before either is walked.
    if (!known || nWords != m_probedWords) {
      if (known) {
        m_cost.probes += 1u;
      }
      m_probedBase  = 0;
      m_probedWords = 0u;
      if (!EngineSymbols::readable(reinterpret_cast<const void*>(base + kOffAllocMask),
                                  static_cast<size_t>(nWords) * sizeof(uint64_t)))
        return 0;
      if (!EngineSymbols::readable(reinterpret_cast<const void*>(base + kOffEntries),
                                  static_cast<size_t>(capacity) * kEntryStride))
        return 0;
      m_probedBase  = base;
      m_probedWords = nWords;
    }

    capacityOut = capacity;
    return base;
  }

  void RenderableEnum::update(uint32_t frame, const Vector3& cameraPos) {
    if (!RtxOptions::RenderableEnum::enable()) {
      // Not read this frame: nothing may be retired on a list that is not this
      // frame's.
      m_existenceUsable = false;
      return;
    }

    using Clock = std::chrono::steady_clock;
    auto ns = [](Clock::time_point a, Clock::time_point b) {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };

    const Clock::time_point tBegin = Clock::now();
    uint32_t capacity = 0u;
    uint64_t resolveNs = 0ull;
    const uintptr_t base = registryBase(capacity, resolveNs);
    const Clock::time_point tBase = Clock::now();

    m_stats.resolved = (base != 0);
    if (base == 0) {
      m_stats.readFailures += 1u;
      // Evidence is only meaningful while the source is actually being read.
      // A gap would otherwise be indistinguishable from a flat window, which
      // is the one thing the gate must never confuse.
      m_promotion.reset();
      m_haveCameraPos = false;
      // A list we could not read is a list we can no longer vouch for: the
      // promoted source goes with it and has to be earned again.
      if (m_existence != nullptr) {
        Logger::warn("[ExistenceSource] demoted 'client.RenderableRegistry' -- the registry"
                     " became unreadable; absence stops meaning death until it re-promotes");
        m_existence.reset();
      }
      m_existenceUsable = false;
      return;
    }

    m_stats.reads   += 1u;
    m_stats.capacity = capacity;

    const uint64_t* mask    = reinterpret_cast<const uint64_t*>(base + kOffAllocMask);
    const uint8_t*  entries = reinterpret_cast<const uint8_t*>(base + kOffEntries);

    m_visible.beginFrame(frame);
    // The promoted list is the same walk, noted into both. It exists only
    // from the frame AFTER promotion, so it is always a complete frame.
    ExistenceSource* existence = m_existence.get();
    if (existence != nullptr) {
      existence->beginFrame(frame);
    }

    uint32_t listed  = 0u;
    uint32_t maxSlot = 0u;
    const uint32_t nWords = capacity / 64u;

    for (uint32_t w = 0; w < nWords; ++w) {
      uint64_t bits = mask[w];
      while (bits != 0ull) {
        const uint32_t slot = w * 64u + lowestSetBit(bits);
        bits &= bits - 1ull;

        if (slot >= capacity)
          break;

        // OPAQUE TOKEN. Copied out as an integer and never dereferenced -- see
        // the hard rule in the header. Null means the slot is free, which is
        // the same test the list builder applies.
        uint64_t handle = 0ull;
        std::memcpy(&handle, entries + static_cast<size_t>(slot) * kEntryStride,
                    sizeof(handle));
        if (handle == 0ull)
          continue;

        m_visible.note(handle);
        if (existence != nullptr) {
          existence->note(handle);
        }
        ++listed;
        if (slot > maxSlot)
          maxSlot = slot;
      }
    }

    const Clock::time_point tWalk = Clock::now();
    m_visible.endFrame();
    if (existence != nullptr) {
      existence->endFrame();
    }
    const Clock::time_point tClose = Clock::now();

    m_stats.listed  = listed;
    m_stats.maxSlot = maxSlot;

    // NV-DXVK slice 2: THE COLLAPSE GUARD. A torn read during a registry resize
    // yields a short list for one frame, and on a promoted source every handle
    // missing from it is a death. Half the previous count is far below what
    // any real frame does -- the sweep that promoted this source held listed=
    // flat to the unit -- so a drop that deep is the read, not the world.
    const bool collapsed = (m_lastListed != 0u) && (listed < m_lastListed / 2u);
    if (collapsed && existence != nullptr) {
      ++m_collapseSkips;
    }
    m_existenceUsable = (existence != nullptr) && !collapsed;
    m_lastListed = listed;

    // ------------------------------------------------------------------
    // THE PROMOTION EVIDENCE.
    //
    // Fed ONLY while the camera has not translated. sec 3.1 is explicit that
    // this belongs to the sweep harness rather than to the enumeration hook,
    // and the reason is that the gate's meaning depends on it: at a fixed
    // position the true object population cannot change, so any movement in
    // listed= is culling leaking into the list. Let the camera walk and a
    // perfectly good pre-cull list would fail the gate, while a bad one could
    // accidentally pass it.
    // ------------------------------------------------------------------
    const bool fixed =
      m_haveCameraPos &&
      std::fabs(cameraPos.x - m_lastCameraPos.x) <= kFixedPosEpsilon &&
      std::fabs(cameraPos.y - m_lastCameraPos.y) <= kFixedPosEpsilon &&
      std::fabs(cameraPos.z - m_lastCameraPos.z) <= kFixedPosEpsilon;

    if (fixed) {
      m_promotion.observe(listed);
    } else {
      m_promotion.reset();
      m_lastCameraPos = cameraPos;
      m_haveCameraPos = true;
    }

    // NV-DXVK slice 2: PROMOTE ONCE, the first frame the gate reads flat, and
    // keep it -- see existence(). promote() logs the promotion. The list is
    // first filled on the next walk, so it is not usable this frame.
    if (m_existence == nullptr && m_promotion.flat()) {
      m_existence = m_promotion.promote(m_visible.name());
    }

    const uint64_t totalNs = ns(tBegin, Clock::now());
    m_cost.resolveNs += resolveNs;
    m_cost.probeNs   += ns(tBegin, tBase) - std::min(resolveNs, ns(tBegin, tBase));
    m_cost.walkNs    += ns(tBase, tWalk);
    m_cost.closeNs   += ns(tWalk, tClose);
    m_cost.totalNs   += totalNs;
    m_cost.maxNs      = std::max(m_cost.maxNs, totalNs);
    m_cost.frames    += 1u;

    if (RtxOptions::RenderableEnum::logStats() && frame - m_lastLogFrame >= 60u) {
      m_lastLogFrame = frame;
      const uint64_t perFrame = 1000ull * std::max<uint64_t>(m_cost.frames, 1ull);
      Logger::warn(str::format(
        "[RenderableEnum] f=", frame,
        " listed=", listed,
        " capacity=", capacity,
        " maxSlot=", maxSlot,
        " fixedCam=", fixed ? 1u : 0u,
        // THE GATE. flatFrames must reach kRequiredFlatFrames under a
        // fixed-position pitch-and-yaw sweep before this list is allowed to
        // assert that anything is dead.
        " flatFrames=", m_promotion.flatFrames(),
        "/", ExistenceSourcePromotion::kRequiredFlatFrames,
        " breaks=", m_promotion.breaks(),
        " promotable=", m_promotion.flat() ? 1u : 0u,
        // Slice 2: promoted and trusted this frame -- absence is death for
        // records that carry a handle. collapseSkips: frames the guard refused.
        " existence=", (m_existence != nullptr && m_existenceUsable) ? 1u : 0u,
        " collapseSkips=", m_collapseSkips,
        " reads=", m_stats.reads,
        " readFail=", m_stats.readFailures,
        // us/frame averaged over the window. total << [Perf.GcInst] enum= means
        // the time is outside update(); otherwise the largest phase is the cost.
        " cost{resolve=", m_cost.resolveNs / perFrame,
        " probe=", m_cost.probeNs / perFrame,
        " walk=", m_cost.walkNs / perFrame,
        " close=", m_cost.closeNs / perFrame,
        " total=", m_cost.totalNs / perFrame,
        " max=", m_cost.maxNs / 1000ull,
        " probes=", m_cost.probes, "}",
        " | listed FLAT under a fixed-position sweep = pre-cull;"
        " listed moving = culling is leaking into the list"));
      m_cost = Cost();
    }
  }

  void RenderableEnum::clear() {
    m_promotion.reset();
    m_stats         = Stats();
    m_haveCameraPos = false;
    m_lastLogFrame  = 0u;
    // A new scene is a new map: the promotion was earned on the old one.
    m_existence.reset();
    m_existenceUsable = false;
    m_lastListed = 0u;
    m_cost = Cost();
    m_probedBase  = 0;
    m_probedWords = 0u;
  }

} // namespace dxvk
