/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <array>
#include <chrono>
#include <fstream>
#include <string>

#include "../thread.h"

namespace dxvk {
  
  enum class LogLevel : std::uint32_t {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    None  = 5,
  };

  /**
   * \brief Logger
   * 
   * Logger for one DLL. Creates a text file and
   * writes all log messages to that file.
   */
  class Logger {
    
  public:

    // NV-DXVK start: pass log level as param
    Logger(const std::string& fileName, const LogLevel logLevel = getMinLogLevel());
    // NV-DXVK end
    // NV-DXVK [buffered log]: drain the line buffer on graceful teardown.
    ~Logger();
    
    // NV-DXVK start: special init pathway for remix logs
    static void initRtxLog();
    // NV-DXVK end
    
    static void trace(const std::string& message);
    static void debug(const std::string& message);
    static void info (const std::string& message);
    static void warn (const std::string& message);
    static void err  (const std::string& message);
    static void log  (LogLevel level, const std::string& message);

    // NV-DXVK [Perf]: is the per-draw diagnostic channel (RTX_D3D11_DIAG=1) on?
    //
    // WHY THIS EXISTS. emitMsg's kFilteredTags list drops high-volume tags, but
    // it drops them INSIDE emitMsg — by which point the caller has already run
    // str::format: float->string conversions, several heap allocations, and then
    // a ~100-entry prefix scan, all to throw the result away. The filter saves
    // file I/O, not CPU.
    //
    // Measured: [fanoutCamWrite] and [fanoutCBRead] are both filtered, both emit
    // ZERO lines, and together with their neighbours accounted for ~19.5 us per
    // instanced draw ([Perf.CamCut] co_vpScan) — 65% of that span. The cost
    // ramped 5.7 -> 19.5 us over the first 10s of gameplay because the guarding
    // condition is "camera moved", so it only engages once the player moves.
    //
    // Guard hot-path log sites with this so str::format never runs:
    //   if (Logger::d3d11DiagEnabled()) Logger::info(str::format(...));
    // Cheap: one function-local static, initialised once.
    static bool d3d11DiagEnabled();

    // NV-DXVK: apply rtx.logDenyTags — the per-run edit to emitMsg's tag
    // denylist. Comma-separated prefixes; a leading '-' un-silences instead of
    // silencing, e.g. "-[SpawnGeomDiag., [BulkPush]". Called once per DLL from
    // the RtxOptions constructor once the conf files are parsed; safe to call
    // again (it rebuilds and republishes the filter atomically).
    //
    // Exists so a blanket denylist entry can be lifted for one run without a
    // rebuild. A silenced probe looks exactly like a probe that never ran, and
    // that has cost this fork two sessions — see the comment block above the
    // list in log.cpp.
    static void setDenyTags(const std::string& spec);

    // NV-DXVK: would emitMsg drop a message starting with this tag? For hot
    // call sites, so str::format never runs for a line the denylist would
    // discard anyway:
    //   if (!Logger::tagDenied("[MtnDedup]")) Logger::info(str::format(...));
    // Uses the same published tag index as emitMsg (a handful of memcmps).
    // Returns false while the index is not built yet (before the first
    // info/warn emit) — the line then goes through emitMsg's own filter, so
    // the answer is never wrong, only occasionally pessimistic about cost.
    // Denylist changes via setDenyTags are picked up on the next call, same
    // as emitMsg itself. Deliberately NOT gated on d3d11DiagEnabled() here:
    // RTX_D3D11_DIAG=1 clears the denylist, which this reflects naturally.
    static bool tagDenied(const char* tag);

    // NV-DXVK: force the log file to disk. Intended for the
    // UnhandledExceptionFilter path where the process is about to die and
    // the OS won't run ofstream destructors. Acquires the same mutex as
    // emitMsg, so it's safe to call concurrently with normal logging.
    static void flush();

    static LogLevel logLevel() {
      return s_instance.m_minLevel;
    }
    
  private:
    
    static Logger s_instance;
    
    LogLevel m_minLevel;
    // NV-DXVK start: Don't double print every line
    bool m_doublePrintToStdErr;
    // NV-DXVK end
    
    dxvk::mutex   m_mutex;
    std::ofstream m_fileStream;

    // NV-DXVK [buffered log]: formatted lines accumulate here and are
    // written to m_fileStream in batches (size/age/severity policy in
    // emitMsg) instead of paying a stream write + OS flush per line.
    // Warn/Error lines and Logger::flush() (called by the crash filter in
    // d3d11_main.cpp) drain immediately, so crash forensics keep working.
    std::string m_lineBuffer;
    std::chrono::steady_clock::time_point m_lastFlushTime {};
    bool m_flushEveryLine = false;

    // Must be called with m_mutex held.
    void flushBufferLocked();

    void emitMsg(LogLevel level, const std::string& message);
    
    static LogLevel getMinLogLevel();
    static std::string getFilePath(const std::string& fileName);

    Logger& operator=(Logger&& other);

  };
  
}
