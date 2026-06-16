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

#include "../util_env.h"
#include "../util_filesys.h"
// NV-DXVK start: Fix some circular inclusion stuff
#include "../util_string.h"
// NV-DXVK end


// NV-DXVK start: Don't double print every line
namespace{
  bool getDoublePrintToStdErr() {
    const std::string str = dxvk::env::getEnvVar("DXVK_LOG_NO_DOUBLE_PRINT_STDERR");
    return str.empty();
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
    if (s_instance.m_fileStream)
      s_instance.m_fileStream.flush();
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
          "[MtnDedup]",
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
          "[InstReap]",
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
          "[SkyTrace.",             // probeContent / probePrefill.face / ...
          "  slot=",                // srvScore texture-pick dump (d3d11_rtx.cpp:14824)
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
        // Prefix match — all filtered tags start at offset 0.
        for (const char* tag : kFilteredTags) {
          const size_t n = std::strlen(tag);
          if (message.size() >= n && std::memcmp(message.data(), tag, n) == 0) {
            return;
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
        for (const char* needle : kFilteredSubstrings) {
          if (message.find(needle) != std::string::npos) {
            return;
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

      std::stringstream stream(message);
      std::string       line;

      char timeString[64];
      getLocalTimeString(timeString);

      while (std::getline(stream, line, '\n')) {
        // NV-DXVK start: Don't double print every line
        if (m_doublePrintToStdErr) {
          std::cerr << timeString << prefix << line << std::endl;
        }
        // NV-DXVK end

        if (m_fileStream) {
          m_fileStream << timeString << prefix << line << std::endl;
          // NV-DXVK: The previous revision added an explicit
          // m_fileStream.flush() here to guarantee the last few log lines
          // before a hard access-violation crash reached disk. That was
          // useful for init-time crash chasing BUT on any high-throughput
          // log path (CS thread emitting per-present / per-draw stats
          // during a fast loading screen, ~3000+ lines/sec) the explicit
          // Windows flush cost 10-100μs per call and serialised the CS
          // thread behind file I/O — manifesting as a "stuck loading
          // screen" that nothing in the rest of the engine could explain.
          // std::endl above already flushes the C++ stream into the OS
          // page cache, which is enough for normal graceful-exit scenarios.
          // For crash-chasing, install an UnhandledExceptionFilter that
          // calls m_fileStream.flush() on the crash path instead of paying
          // the cost on every single line in steady state.
        }
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
    return *this;
  }
  
}
