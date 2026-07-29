/*
* Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../util_env.h"
#include "../util_filesys.h"
// NV-DXVK start: Fix some circular inclusion stuff
#include "../util_string.h"
// NV-DXVK end


// NV-DXVK start: Don't double print every line
namespace{
  // NV-DXVK [perf]: the double-print writes every line to std::cerr, which is
  // unit-buffered, and then flushes it again via std::endl. In a windowed game
  // there is usually no stderr attached at all, so those two flushes per line
  // are pure syscall overhead producing output nobody can see. Only double-print
  // when stderr is actually connected to something.
  bool stdErrIsConnected() {
#ifdef _WIN32
    const HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    return h != nullptr && h != INVALID_HANDLE_VALUE;
#else
    return true;
#endif
  }

  bool getDoublePrintToStdErr() {
    const std::string str = dxvk::env::getEnvVar("DXVK_LOG_NO_DOUBLE_PRINT_STDERR");
    return str.empty() && stdErrIsConnected();
  }

  template<int N>
  static inline void getLocalTimeString(char(&timeString)[N]) {
    // [HH:MM:SS.MS]
    static const char* format = "[%02d:%02d:%02d.%03d] ";

#ifdef _WIN32
    SYSTEMTIME lt;
    GetLocalTime(&lt);

    sprintf_s(timeString, format,
              lt.wHour, lt.wMinute, lt.wSecond, lt.wMilliseconds);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm* lt = localtime(&tv.tv_sec);

    sprintf_s(timeString, format,
              lt->tm_hour, lt->tm_min, lt->tm_sec, (tv.tv_usec / 1000) % 1000);
#endif
  }
}
// NV-DXVK end

namespace dxvk {

  Logger::Logger(const std::string& fileName, const LogLevel logLevel)
  : m_minLevel(logLevel)
  // NV-DXVK start: Don't double print every line
  , m_doublePrintToStdErr(getDoublePrintToStdErr())
  // NV-DXVK end
  {
    if (m_minLevel != LogLevel::None) {
      const auto path = getFilePath(fileName);

      if (!path.empty()) {
        m_fileStream = std::ofstream(str::tows(path.c_str()).c_str());
        assert(m_fileStream.is_open());
      }
    }
    // NV-DXVK [buffered log]: opt-out knob restoring flush-per-line.
    m_flushEveryLine = [](){
      const char* v = std::getenv("RTX_LOG_UNBUFFERED");
      return v != nullptr && v[0] == '1';
    }();
    m_lastFlushTime = std::chrono::steady_clock::now();
  }

  // NV-DXVK [buffered log]: drain whatever is still buffered on graceful
  // teardown (static-destruction order at process exit). Crash paths are
  // covered separately by the UnhandledExceptionFilter calling flush().
  Logger::~Logger() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    flushBufferLocked();
  }

  // NV-DXVK [buffered log]: single write + single OS flush for the whole
  // batch. Called with m_mutex held.
  void Logger::flushBufferLocked() {
    if (m_fileStream && !m_lineBuffer.empty()) {
      m_fileStream.write(m_lineBuffer.data(),
                         static_cast<std::streamsize>(m_lineBuffer.size()));
      m_fileStream.flush();
    }
    // clear() keeps capacity — no allocator churn at steady state.
    m_lineBuffer.clear();
    m_lastFlushTime = std::chrono::steady_clock::now();
  }
  
  void Logger::initRtxLog() {
    s_instance = std::move(Logger("remix-dxvk.log"));
  }

  void Logger::trace(const std::string& message) {
    s_instance.emitMsg(LogLevel::Trace, message);
  }
  
  void Logger::debug(const std::string& message) {
    s_instance.emitMsg(LogLevel::Debug, message);
  }

  void Logger::info(const std::string& message) {
    s_instance.emitMsg(LogLevel::Info, message);
  }

  void Logger::warn(const std::string& message) {
    s_instance.emitMsg(LogLevel::Warn, message);
  }

  void Logger::err(const std::string& message) {
    s_instance.emitMsg(LogLevel::Error, message);
  }

  void Logger::log(LogLevel level, const std::string& message) {
    s_instance.emitMsg(level, message);
  }

  void Logger::flush() {
    std::lock_guard<dxvk::mutex> lock(s_instance.m_mutex);
    // NV-DXVK [buffered log]: drain the batch buffer, not just the stream —
    // the crash filter depends on this getting the last lines to disk.
    s_instance.flushBufferLocked();
  }

  // NV-DXVK [Perf]: see the declaration in log.h. Same env var and the same
  // one-shot initialisation emitMsg uses, exposed so hot call sites can skip
  // building a string that emitMsg would only discard.
  bool Logger::d3d11DiagEnabled() {
    static const bool s_enabled = []() {
      const char* v = std::getenv("RTX_D3D11_DIAG");
      return v != nullptr && v[0] == '1';
    }();
    return s_enabled;
  }

  void Logger::emitMsg(LogLevel level, const std::string& message) {
    // NV-DXVK: drop high-volume diagnostic tags unless RTX_D3D11_DIAG=1.
    // These tags fire at per-draw rates (1000s/sec in TF2 main menu) and
    // the cumulative file-I/O stalls the game even with flush removed.
    // Filtering here is cheaper than patching ~25 individual call sites.
    // Applies to info AND warn — some diagnostics are emitted as warn
    // (e.g. [D3D11Rtx.o2w.t31.nosrv]) even though they're per-draw
    // volume. Error level always passes through.
    if (level == LogLevel::Info || level == LogLevel::Warn) {
      static const bool s_d3d11DiagEnabled = []() {
        const char* v = std::getenv("RTX_D3D11_DIAG");
        return v != nullptr && v[0] == '1';
      }();
      if (!s_d3d11DiagEnabled) {
        static constexpr const char* kFilteredTags[] = {
          "[D3D11Rtx.o2w.",
          "[D3D11Rtx.t31.",
          "[D3D11Rtx.path",
          "[D3D11Rtx.vs.",
          "[D3D11Rtx.UITex]",
          // NV-DXVK: [ShaderHashMap] intentionally NOT filtered — it fires
          // exactly once per unique shader (gated by D3D11ShaderModuleSet::
          // m_modules in d3d11_shader.cpp), and is the only way to map the
          // truncated 64-bit getHash() seen in [D3D11Rtx] FillMaterialData
          // logs to the SHA1-named .dxbc files in shader_dumps/. Removing
          // it from filtering is required for the gloss/spec disassembly
          // workflow.
          "[VMPass",
          "[VMHunt",
          "[VsClass",
          "[BBI]",
          "[BBI-",
          "[PI-",
          "[BUFMAP]",
          "[BUFMAP-",
          "[VisibleSurf]",
          "[Opaque]",
          "[Unord]",
          "[BVH-BUILD]",
          "[BLAS-TRACK]",
          "[AccelMgr.",
          "[ASMAP]",
          "[ASMAP-",
          "[TLAS-coh]",
          "[D3D11Rtx.orient",
          "[D3D11Rtx.o2wRot",
          "[D3D11SwapChain]",
          "  VS s",
          "  Bone from MappedSlice",
          "  name=",
          "  pos[",
          "  BSP-fanout-path",
          // NV-DXVK: re-enabled for the garbled-ship (bone-skinning) investigation
          // — these show the actual skinning state per draw.
          //   "[skin.histo",
          //   "[skin.vert",
          //   "[DrawSkin",
          //   "[BoneUploadFrame",
          "[IDX-SNAP",
          "[IDX-SCAN-FALLBACK",
          "[BVH-UPDATE",
          "[TLAS-FILTER]",
          // NV-DXVK: top per-frame FPS offenders identified from the
          // remix-dxvk.log FPS analysis (these alone were ~120 lines/frame
          // and dropped the game to ~4-8 FPS). Listed per-tag on purpose:
          // any NEW probe you add uses a tag that is NOT in this list, so it
          // still prints while these old ones stay silent. Set
          // RTX_D3D11_DIAG=1 to bring everything back.
          "[PhantomProbe]",
          "[HullCensus.Inst]",
          "[PsCBfields]",
          "[GateAll]",
          "[D3D11Rtx.SampPick]",
          "[MtnFanoutIdx]",
          "[RTX-InstMgr.UVx]",
          "[LodAll]",
          "[LodV10]",
          "[SpawnGeomDiag.BBI]",
          "[SpawnGeomDiag.DrawIn]",
          "[ZigGeoState]",
          "[ZigW2v]",
          "[ZigCam]",
          "[TC1Surface]",
          "[BlasFill]",
          "[pcdTrace]",
          "[D3D11RtxFrame]",
          "[VM.class]",
          "[VM.check]",
          "[PropIdHashInputs.Mtn2904]",
          "[FloorTrace.emit]",
          "[TLASEntry",            // catches [TLASEntry] and [TLASEntry-View]
          "[InstCounts]",
          // "[MtnDedup]",   // un-silenced 2026-07-29: now also driven by
          // rtx.findSimilarProbeVsHashes for the tree-billboard churn hunt.
          // Its two hardcoded mountain VSes (0x29146e1d, 0x28f7ffa9) do not
          // draw in this level (1 mention each in the log, both from the
          // shader-hash table), so nothing uncapped is being re-enabled; the
          // option path is capped at 4000 lines. Re-silence when done.
          "[VS2904Trace]",
          "[fanoutCamWrite]",
          "[fanoutCBRead]",
          "[StudioVcall]",
          "[MainCamPose",          // catches [MainCamPose] and [MainCamPoseOverride]
          "[VanishDiag-Raw]",
          // NV-DXVK: noise unrelated to the sky-triangle (VS 0x29566)
          // investigation — geometry-batch / skinning / dropship / camera-
          // cache / HUD plumbing. Disabled so the log reads cleanly for
          // [SkyTriAABB] + [SkyDiag] + [Coverage] + [Mtn*]. Kept ON: SkyDiag,
          // SkyTrace*, Coverage, Mtn*, SurfAlbedo, SkyTriAABB, CamMgr*,
          // SubView*. RTX_D3D11_DIAG=1 restores all of these.
          "[SpawnGeomDiag.",
          "[SceneInvalidRaw]",
          "[SceneClearRaw]",
          "[debobTimeline]",
          "[PropIdTrace]",
          "[VanishDiag",            // VanishDiag/-T/-Stack-Auto (separate bug)
          "[Widow",                 // dropship transform debugging
          "[Ship",                  // dropship hull (ShipBox/Bone/Bake/SrcVB...)
          "[LodNear]",
          "[Reflect",               // Reflect / Reflect.diag
          "[Path13Diag]",
          "[De10]",
          "[cachedSaveSkipNonPlayer]",
          "[VguiSurface]",
          "[GcKeep2904]",
          // NV-DXVK: 2026-06-16 perf pass. The remix-dxvk.log frequency
          // analysis showed these tags alone produced ~50k of 72k lines in a
          // 44s window (~820 lines/frame at 2 fps). Because the filter is a
          // DENYLIST, every probe added after the lists above kept printing —
          // these are the ones left on from the BLAS/skinning/floor/sky/
          // census bug hunts. They fire on the per-draw / per-instance /
          // per-frame hot path, and the shared log mutex + file I/O inflated
          // each submitDraw to ~300us (~340ms/frame for ~1000 draws). Gated
          // off here so perf can be measured cleanly. RTX_D3D11_DIAG=1 brings
          // them all back. NOTE [ShaderHashMap] is deliberately NOT here (see
          // above); [Coverage] is controlled by its own rtx option, not this.
          "[TrimCache]",
          "[BlasGeom]",
          "[FloorTrace.",           // .recv / .aabb / .emit (emit already above)
          "[TlasCensus]",
          "[PassCensus]",
          // "[InstReap]",          // un-silenced 2026-07-29: the reap decision
          //   is the open question for the shared-BlasEntry ping-pong (two props
          //   sharing one mesh, alternating CREATE every frame, mapSz never >1).
          //   Now carries blasPtr + mapSz. Already gameplay-gated and capped at
          //   24 lines/frame, so steady-state cost is bounded. Re-silence when
          //   done. Note [InstReapWave] is a DIFFERENT tag (mass-removal only)
          //   and was never filtered — the prefix match stops at the ']'.
          "[SubViewVsCensus]",
          "[SubViewVar]",
          "[T31Stale]",
          "[MtnMotion]",
          "[MtnAlphaState]",
          "[MtnPIAdd]",
          "[BonePropId",            // path10/11/12 + -fanout / -Ndraw
          "[ZigGun]",
          "[SkinAABB]",
          "[TonemapProbe]",
          // NV-DXVK: was the blanket prefix "[SkyTrace." which ALSO silenced
          // [SkyTrace.delayPush] / [SkyTrace.delayReplay] -- the two tags that
          // report 3D-skybox geometry entering the scene. They are per-frame and
          // low volume (delayPush caps at 3 lines/frame, delayReplay at 1), and
          // their absence cost a whole session: the tree-billboard investigation
          // read "0 SkyTrace lines" as "the replay does not run" when the lines
          // were simply being dropped. Deny the genuinely noisy sub-tags by name
          // instead, so a newly added [SkyTrace.*] tag is visible by default.
          "[SkyTrace.probeContent",
          "[SkyTrace.probePrefill",
          "[SkyTrace.matteContent",
          "[SkyTrace.matteRaster",
          "[SkyTrace.matteClear",
          "[SkyTrace.constants",
          "[SkyTrace.compositeBind",
          "[SkyTrace.domeArgs",
          "[SkyTrace.fogColor",
          "[SkyTrace.skyVerts",
          "[SkyTrace.primaryMiss",
          "[SkyTrace.topHit",       // topHitSurf / topHitSurfAliased / topHitWorld
          "  slot=",                // srvScore texture-pick dump (d3d11_rtx.cpp:14824)
          // NV-DXVK: 2026-07-28 perf pass. Frequency analysis of a 167 s run
          // (20.04 MB / 112190 lines / ~1745 gameplay frames) with the tags
          // binned by time, so front-loaded probes are not confused with
          // steady-state ones. Everything below was still firing in the LAST
          // 30 s of the session, i.e. it is per-frame gameplay cost, not
          // load-time. Measured per-tag volume in that run is in the comments.
          //
          // Deliberately NOT added, for the record:
          //   [ShaderHashMap]  9408 lines but ALL of them before t+61 s and
          //                    zero in the last 30 s — it is behaving exactly
          //                    as its exemption above claims (once per unique
          //                    shader, during load). Not a steady-state cost.
          //   [SubmitStall*]   owned by rtx.logSubmitStall; turn that False
          //                    rather than masking it here, or the option
          //                    silently stops meaning anything.
          //   [CamMgr.*]       unmasked on request, see the block below.
          //   [NsysAuto]       the capture script WATCHES the log for its
          //                    CAPTURE-BEGIN/END markers to drive the hotkey.
          //                    Filtering it would break unattended captures.
          //   [Perf.*]         the instrumentation this is all in aid of.
          //                    ([Perf.GeoChurn] is the one exception - it is a
          //                    per-frame churn counter, not a timing source.)
          "[SkyAutoCb2",            // 10496 lines / 2.90 MB — .classify is ~6/frame
          "[Zig",                   // whole Zig* family, ~3.0 MB: P1/MtxRows/MtxDiff/
                                    // Bone/Bone2/Sync/VmO2w/GunRB/VB/NDC. ZigGun,
                                    // ZigGeoState, ZigW2v, ZigCam already above.
          "[Atmosphere.",           // .live + .lut, ~1.50 MB (474-509 bytes/line)
          "[TlasMember]",           // 3488 lines / 1.05 MB, ~2/frame
          "[EngineCam",             // EngineCam + EngineCamFrame, ~0.90 MB
          "[EnginePost]",           // 0.48 MB
          "[NrcBounds]",            // 0.38 MB
          "[TF2Probe.",             // .CLIFF, 0.37 MB
          "[TLASFrame]",            // 0.34 MB
          "[SceneGcSummary]",       // 0.30 MB
          "[SkyProbeWaste]",        // 0.28 MB
          "[CullCmp]",              // 0.23 MB
          "[BoneIdProbe.",          // .Bulker, 0.19 MB
          "[cachedSave]",           // 0.17 MB  (cachedSaveSkipNonPlayer already above)
          "[GcExit]",               // 0.16 MB
          "[v2pWrite]",             // 0.16 MB
          "[LightContrib.",         // .all, 0.15 MB
          "[pcdEnter]",             // 0.14 MB  (pcdTrace already above)
          "[Perf.GeoChurn]",        // 0.14 MB — counter, not a timing source
          "[VM.create]",            // 0.13 MB  (VM.class / VM.check already above)
          // "[SubViewKey]",        // un-silenced 2026-07-29: the aggregate
          //   carries cand/hit/create/scaled per frame. [SubViewKey.create]
          //   only details the CREATE branch, so a sub-view draw that HITS
          //   (claims an already-existing instance — the propId-collision
          //   takeover) emits no detail line at all and is invisible without
          //   this counter. 0.13 MB.
          "[ReskinProbe]",          // 0.10 MB
          "[cachedConsume]",
          "[cacheWriteAccept]",
          "[BonePaletteShare]",
          "[SlotUpd]",
          "[w2vGuardCatch]",
          "[subPassUpd]",
          // Reverses the 2026-06 un-mask at the top of this list. Those two were
          // re-enabled for the garbled-ship / bone-skinning hunt, which has since
          // closed (the razor draws were identified and the s2s bone chain was
          // traced end to end). They are ~0.48 MB/run of per-frame upload state.
          // If that investigation reopens, delete these two lines rather than
          // setting RTX_D3D11_DIAG=1, which would also unmask everything else.
          "[BoneUploadFrame",
          "[BoneFresh]",
          // NV-DXVK: camera diagnostics UNMASKED on request — these are the
          // primary signal for camera-stability issues (rejections, re-latch,
          // per-camera probe) and were previously hidden. They are internally
          // throttled (hyst/latch capped at ~40, probeI per-unique-camera), so
          // unfiltering them does not spam. Re-add to filter if too noisy.
          // "[CamMgr.probeI",
          // "[CamMgr.hyst",
          // "[CamMgr.latch",
          // "[CamMgr.hist",
        };
        // NV-DXVK [perf]: this used to strlen + memcmp all ~90 tags for EVERY
        // info/warn message, filtered or not — ~90 calls into the CRT per log
        // call on the per-draw path. Index the list once by the tag's second
        // character (the first letter after '['), which spreads ~90 tags over
        // ~25 buckets, so a lookup touches a handful of entries instead of all
        // of them. Same list, same prefix semantics.
        struct TagEntry { const char* text; size_t len; };
        struct TagIndex {
          std::vector<TagEntry> buckets[256];

          TagIndex() {
            for (const char* tag : kFilteredTags) {
              const size_t n = std::strlen(tag);
              if (n < 2)
                continue;  // the list has no such entries; guards the index below
              buckets[static_cast<unsigned char>(tag[1])].push_back({ tag, n });
            }
          }
        };
        static const TagIndex s_tagIndex;

        if (message.size() >= 2) {
          for (const auto& tag : s_tagIndex.buckets[static_cast<unsigned char>(message[1])]) {
            if (message.size() >= tag.len
             && std::memcmp(message.data(), tag.text, tag.len) == 0) {
              return;
            }
          }
        }
        // Substring match — for messages that share the bare "[D3D11Rtx]"
        // tag with legitimate errors we can't blanket-filter on prefix.
        // These phrases identify specific high-volume diagnostic calls
        // inside that tag.
        static constexpr const char* kFilteredSubstrings[] = {
          "objectToWorld NOT FOUND",
          "o2wPaths: identity=",
          "filters: throttle=",
          "TF2 skinned char bound:",
          "EndFrame: draws=",
          "CB3 read:",
          "Reusing frame VP for fallback",
          "Decomposed combined VP",
          "FillMaterialData draw #",   // per-draw [D3D11Rtx] material-pick dump
        };
        // NV-DXVK [perf]: MSVC's std::string::find is a plain character loop, so
        // nine of them over a 200+ byte message was ~2k comparisons per log
        // call. memchr is SIMD-accelerated in the CRT, so seek candidate start
        // positions with it and only memcmp at those. Same substring semantics.
        for (const char* needle : kFilteredSubstrings) {
          const size_t nLen = std::strlen(needle);
          if (nLen == 0 || message.size() < nLen)
            continue;

          const char*  cur  = message.data();
          size_t       left = message.size() - nLen + 1;

          while (left != 0) {
            const char* hit = static_cast<const char*>(std::memchr(cur, needle[0], left));
            if (hit == nullptr)
              break;
            if (std::memcmp(hit, needle, nLen) == 0)
              return;
            left -= static_cast<size_t>(hit - cur) + 1;
            cur   = hit + 1;
          }
        }
      }
    }
    if (level >= m_minLevel) {
      // NV-DXVK: OutputDebugString acquires a PROCESS-WIDE kernel mutex
      // to deliver the message to any listening debugger (and to itself
      // even when none is attached). At high log throughput (2000+ lines
      // per second as seen during Titanfall 2 loading) this serialises
      // every thread in the process that tries to log — the CS thread,
      // the D3D11 submission thread, and the game's own audio thread all
      // block on the same mutex, so audio cuts out and the loading
      // screen visibly freezes even though no thread is actually
      // deadlocked. Gate behind an env-var opt-in — unset by default
      // means ODS never fires, which is the right behaviour when no
      // debugger is attached. Set DXVK_ODS_LOG=1 to re-enable when
      // attached to the process with a debugger.
      static const bool s_odsEnabled = []() {
        const char* v = std::getenv("DXVK_ODS_LOG");
        return v != nullptr && v[0] == '1';
      }();
      if (s_odsEnabled) {
        OutputDebugString((message + '\n').c_str());
      }

      std::lock_guard<dxvk::mutex> lock(m_mutex);
      
      constexpr std::array<const char*, 5> s_prefixes{
        "trace: ",
        "debug: ",
        "info:  ",
        "warn:  ",
        "err:   ",
      };
      const char* prefix = s_prefixes[static_cast<std::uint32_t>(level)];

      char timeString[64];
      getLocalTimeString(timeString);

      // NV-DXVK [perf]: this used to build a std::stringstream over a COPY of
      // the message and pull lines out with std::getline into a temporary
      // std::string — a stream construction plus two allocations for every
      // emitted line. Split on '\n' in place instead; memchr is the CRT's
      // SIMD scan. Line semantics are identical to getline's, including the
      // trailing '\n' not producing an extra empty line.
      //
      // std::cerr is unit-buffered and std::endl flushed again on top, so the
      // double-print cost N flushes per message. Accumulate and hand it over in
      // one insertion.
      std::string stdErrBatch;

      for (size_t pos = 0; pos < message.size(); ) {
        const char*  from = message.data() + pos;
        const size_t avail = message.size() - pos;
        const char*  nl   = static_cast<const char*>(std::memchr(from, '\n', avail));
        const size_t len  = (nl != nullptr) ? static_cast<size_t>(nl - from) : avail;

        // NV-DXVK start: Don't double print every line
        if (m_doublePrintToStdErr) {
          stdErrBatch.append(timeString);
          stdErrBatch.append(prefix);
          stdErrBatch.append(from, len);
          stdErrBatch.push_back('\n');
        }
        // NV-DXVK end

        // NV-DXVK [buffered log]: append the formatted line to the batch
        // buffer instead of `m_fileStream << ... << std::endl`. The old
        // per-line std::endl forced an OS-level flush for EVERY line while
        // holding the logger mutex; at the measured 570-860 lines/s the
        // flushes plus the mutex hold time serialised the CS thread and the
        // D3D11 submit thread against each other ([Perf.SdThreads] stallUs).
        // Lines keep their own timestamps — only the WRITE is deferred.
        m_lineBuffer.append(timeString);
        m_lineBuffer.append(prefix);
        m_lineBuffer.append(from, len);
        m_lineBuffer.push_back('\n');

        if (nl == nullptr)
          break;              // no trailing newline — that was the last line

        pos += len + 1;       // a trailing '\n' leaves pos == size and ends the
                              // loop, so it produces no extra empty line
      }

      if (!stdErrBatch.empty()) {
        std::cerr << stdErrBatch;
        std::cerr.flush();
      }

      // NV-DXVK [buffered log]: flush policy —
      //   * Warn/Error: immediately. These are the lines that matter when
      //     the process dies unexpectedly, and they are low-volume.
      //   * Buffer over 64 KB: bound memory and write in efficient chunks.
      //   * Older than 100 ms: bound how stale the on-disk log can be, so
      //     tail -f behavior and TDR forensics stay usable.
      //   * RTX_LOG_UNBUFFERED=1: restore flush-per-line behavior.
      // Everything else stays buffered. The UnhandledExceptionFilter in
      // d3d11_main.cpp calls Logger::flush() on crash, which drains this
      // buffer, and ~Logger drains it on graceful exit.
      constexpr size_t kFlushSizeThreshold = 64 * 1024;
      const bool flushNow =
           m_flushEveryLine
        || level >= LogLevel::Warn
        || m_lineBuffer.size() >= kFlushSizeThreshold
        || (std::chrono::steady_clock::now() - m_lastFlushTime) >= std::chrono::milliseconds(100);
      if (flushNow) {
        flushBufferLocked();
      }
    }
  }
  
  
  LogLevel Logger::getMinLogLevel() {
    const std::array<std::pair<const char*, LogLevel>, 6> logLevels{ {
      { "trace", LogLevel::Trace },
      { "debug", LogLevel::Debug },
      { "info",  LogLevel::Info  },
      { "warn",  LogLevel::Warn  },
      { "error", LogLevel::Error },
      { "none",  LogLevel::None  },
    } };
    
    const std::string logLevelStr = env::getEnvVar("DXVK_LOG_LEVEL");
    
    for (const auto& pair : logLevels) {
      if (logLevelStr == pair.first)
        return pair.second;
    }
    
    return LogLevel::Info;
  }
  
  std::string Logger::getFilePath(const std::string& fileName) {
    // NV-DXVK start: Use std::filesystem::path helpers + RtxFileSys
    auto path = util::RtxFileSys::path(util::RtxFileSys::Logs);

    // Note: If no path is specified to store log files in, simply use the current directory by returning
    // the specified log file name directly.
    if (path.empty()) {
      return fileName;
    }

    // Append the specified log file name to the logging directory.
    path /= fileName;

    return path.string();
    // NV-DXVK end
  }
  
  Logger& Logger::operator=(Logger&& other) {
    m_minLevel = other.m_minLevel;
    m_doublePrintToStdErr = other.m_doublePrintToStdErr;
    std::swap(m_fileStream, other.m_fileStream);
    // NV-DXVK [buffered log]: carry the batch state with the stream it
    // belongs to (initRtxLog() move-assigns over the static instance).
    std::swap(m_lineBuffer, other.m_lineBuffer);
    m_lastFlushTime = other.m_lastFlushTime;
    m_flushEveryLine = other.m_flushEveryLine;
    return *this;
  }
  
}
