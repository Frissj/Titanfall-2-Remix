#include "dxvk_cs.h"
#include "dxvk_scoped_annotation.h"

// NV-DXVK start: notify user and kill process on exception in CS thread to avoid silent hangs
#include "rtx_render/rtx_env.h"
// NV-DXVK end

#include "../tracy/TracyC.h"

// NV-DXVK [Perf.CsSplit]: per-chunk execution timing on the CS thread.
// str::format reaches here transitively through dxvk_context.h, but this file
// has never used it before - included explicitly so the report does not depend
// on someone else's include graph.
#include <chrono>
#include <algorithm>
#include "../util/util_string.h"
#include "../util/log/log.h"
// NV-DXVK [Perf.Report]: this thread is one of the two pole candidates, so the
// assembler cannot render a verdict without what [Perf.CsSplit] already knows.
#include "rtx_render/rtx_perf_report.h"

namespace dxvk {

  // NV-DXVK [Perf.CsSplit] 2026-08-06: is dxvk-cs's time ONE fat pass or a
  // million small ones? That distinction decides whether it can be threaded.
  //
  // [ThreadCensus] measures dxvk-cs at 95-98% of one core on a ~100 ms frame:
  // it IS the frame time, with the game thread blocked ~60 ms and the GPU idle
  // ~76 ms behind it. But "move it to the worker pool" is only possible for
  // some of that work:
  //   - Vulkan command recording is serial by construction (one VkCommandBuffer,
  //     D3D11-ordered state and draws). It cannot be spread across cores.
  //   - Remix's scene processing (instance manager, accel manager, spatial map,
  //     hashing) is data-parallel in principle - 16k mostly independent
  //     instances - and the d3d11-geometry pool that would run it already
  //     exists and idles at ~1.4%.
  // The two are indistinguishable in the [GapSampler] histogram because it is
  // flat (top bucket 1.9%, top eight under 9%).
  //
  // WHY PER-CHUNK, and not a timer around executeAll vs the rest of the loop:
  // threadFunc's only work IS executeAll, so that split would read ~100%/0% and
  // answer nothing. The shape of the per-chunk DISTRIBUTION does answer it:
  //   one chunk/frame at tens of ms -> a big serial Remix pass rides in a
  //     single command; that is the parallelizable case and the fat bucket
  //     names how much is on the table
  //   ~1400 chunks/frame at microseconds each -> the cost is command replay
  //     spread over the whole stream; nothing to hand to a pool, and the answer
  //     to "can we multithread this" is no
  // A frame is ~1364 chunks, so one clock pair per chunk is ~2700 steady_clock
  // reads (~41 ns each, measured) = ~0.11 ms/frame against a 96 ms thread:
  // ~0.1%, and it is measuring the thread that decides the frame rate.
  //
  // Compile-time constant next to the code it affects, per the convention the
  // other perf switches in this project follow.
  static constexpr bool kEnableCsSplit = true;

  DxvkCsChunk::DxvkCsChunk() {
    
  }
  
  
  DxvkCsChunk::~DxvkCsChunk() {
    this->reset();
  }
  
  
  void DxvkCsChunk::init(DxvkCsChunkFlags flags) {
    m_flags = flags;
  }


  void DxvkCsChunk::executeAll(DxvkContext* ctx) {
    ScopedCpuProfileZone();
    auto cmd = m_head;
    
    if (m_flags.test(DxvkCsChunkFlag::SingleUse)) {
      m_commandOffset = 0;
      
      while (cmd != nullptr) {
        auto next = cmd->next();
        cmd->exec(ctx);
        cmd->~DxvkCsCmd();
        cmd = next;
      }

      m_head = nullptr;
      m_tail = nullptr;
    } else {
      while (cmd != nullptr) {
        cmd->exec(ctx);
        cmd = cmd->next();
      }
    }
  }
  
  
  void DxvkCsChunk::reset() {
    auto cmd = m_head;

    while (cmd != nullptr) {
      auto next = cmd->next();
      cmd->~DxvkCsCmd();
      cmd = next;
    }
    
    m_head = nullptr;
    m_tail = nullptr;

    m_commandOffset = 0;
  }
  
  
  DxvkCsChunkPool::DxvkCsChunkPool() {
    
  }
  
  
  DxvkCsChunkPool::~DxvkCsChunkPool() {
    for (DxvkCsChunk* chunk : m_chunks)
      delete chunk;
  }
  
  
  DxvkCsChunk* DxvkCsChunkPool::allocChunk(DxvkCsChunkFlags flags) {
    ScopedCpuProfileZone();
    DxvkCsChunk* chunk = nullptr;

    { std::lock_guard<sync::Spinlock> lock(m_mutex);
      
      if (m_chunks.size() != 0) {
        chunk = m_chunks.back();
        m_chunks.pop_back();
      }
    }
    
    if (!chunk)
      chunk = new DxvkCsChunk();
    
    chunk->init(flags);
    return chunk;
  }
  
  
  void DxvkCsChunkPool::freeChunk(DxvkCsChunk* chunk) {
    ScopedCpuProfileZone();
    chunk->reset();
    
    std::lock_guard<sync::Spinlock> lock(m_mutex);
    m_chunks.push_back(chunk);
  }
  
  
  DxvkCsThread::DxvkCsThread(
    const Rc<DxvkDevice>&   device,
    const Rc<DxvkContext>&  context)
  : m_device(device), m_context(context),
    m_thread([this] { threadFunc(); }) {
    
  }
  
  
  DxvkCsThread::~DxvkCsThread() {
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      m_stopped.store(true);
    }
    
    m_condOnAdd.notify_one();
    m_thread.join();
  }
  
  
  uint64_t DxvkCsThread::dispatchChunk(DxvkCsChunkRef&& chunk) {
    ScopedCpuProfileZone();

    uint64_t seq;

    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      seq = ++m_chunksDispatched;
      m_chunksQueued.push(std::move(chunk));
    }
    
    m_condOnAdd.notify_one();
    return seq;
  }
  
  
  void DxvkCsThread::synchronize(uint64_t seq) {
    ScopedCpuProfileZone();

    // Avoid locking if we know the sync is a no-op, may
    // reduce overhead if this is being called frequently
    if (seq > m_chunksExecuted.load(std::memory_order_acquire)) {
      std::unique_lock<dxvk::mutex> lock(m_mutex);

      if (seq == SynchronizeAll)
        seq = m_chunksDispatched.load();

      auto t0 = dxvk::high_resolution_clock::now();
      m_condOnSync.wait(lock, [this, seq] {
        return m_chunksExecuted.load() >= seq;
      });
      auto t1 = dxvk::high_resolution_clock::now();
      auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

      m_device->addStatCtr(DxvkStatCounter::CsSyncCount, 1);
      m_device->addStatCtr(DxvkStatCounter::CsSyncTicks, ticks.count());
    }
  }
  
  
  void DxvkCsThread::threadFunc() {
    ScopedCpuProfileZone();

    env::setThreadName("dxvk-cs");

    DxvkCsChunkRef chunk;

    // [Perf.CsSplit] state. Thread-local by construction - only this thread
    // touches it - so no synchronisation and no atomics on the hot path.
    // Buckets are decade-spaced in microseconds; `fat` deliberately starts at
    // 1 ms because that is the scale at which a bucket could hold a whole
    // scene-processing pass rather than a run of draw commands.
    constexpr uint32_t kCsBuckets = 5;   // <10us, <100us, <1ms, <10ms, >=10ms
    uint64_t csBucketNs[kCsBuckets] = {};
    uint64_t csBucketN [kCsBuckets] = {};
    uint64_t csTotalNs = 0, csChunks = 0, csMaxNs = 0, csIdleNs = 0;
    auto     csLastReport = std::chrono::steady_clock::now();
    auto     csWaitStart  = csLastReport;
    // NV-DXVK [Perf.Report]: frame ordinal at the previous report, so this
    // thread can normalise its window totals to ms/FRAME. csChunks is a chunk
    // count, not a frame count -- dividing by it was the "billed per draw,
    // measured per instance" mistake in another costume.
    uint32_t csLastReportFrame = perfreport::currentFrame();

    try {
      while (!m_stopped.load()) {
        { 
          ScopedCpuProfileZoneN("waiting for work");
          std::unique_lock<dxvk::mutex> lock(m_mutex);
          if (chunk) {
            m_chunksExecuted++;
            m_condOnSync.notify_one();
            
            chunk = DxvkCsChunkRef();
          }
          
          if (m_chunksQueued.size() == 0) {
            m_condOnAdd.wait(lock, [this] {
              return (m_chunksQueued.size() != 0)
                  || (m_stopped.load());
            });
          }
          
          if (m_chunksQueued.size() != 0) {
            chunk = std::move(m_chunksQueued.front());
            m_chunksQueued.pop();
          }
        }
        
        if (chunk) {
          m_context->addStatCtr(DxvkStatCounter::CsChunkCount, 1);

          if constexpr (kEnableCsSplit) {
            const auto t0 = std::chrono::steady_clock::now();
            // Everything between leaving the lock above and here is queue
            // bookkeeping and the condvar wait; billing it as idle keeps
            // "busy" comparable with [ThreadCensus]'s cycle-based number.
            csIdleNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
              t0 - csWaitStart).count());

            chunk->executeAll(m_context.ptr());

            const auto t1 = std::chrono::steady_clock::now();
            csWaitStart = t1;
            const uint64_t dNs = uint64_t(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            csTotalNs += dNs;
            ++csChunks;
            if (dNs > csMaxNs)
              csMaxNs = dNs;

            const uint64_t dUs = dNs / 1000ull;
            const uint32_t b = dUs < 10ull      ? 0u
                             : dUs < 100ull     ? 1u
                             : dUs < 1000ull    ? 2u
                             : dUs < 10000ull   ? 3u
                                                : 4u;
            csBucketNs[b] += dNs;
            csBucketN [b] += 1;

            // Report on the same 5s cadence as [ThreadCensus]/[GapSampler] so
            // the three lines describe the same window and can be read together.
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                  t1 - csLastReport).count() >= 5000) {
              const double winMs = double(std::chrono::duration_cast<
                std::chrono::milliseconds>(t1 - csLastReport).count());
              csLastReport = t1;

              static constexpr const char* kBucketNames[kCsBuckets] = {
                "<10us", "<100us", "<1ms", "<10ms", ">=10ms"
              };
              // NV-DXVK [Perf.Report]: dxvk-cs is the second pole candidate, and
              // the >=10ms bucket is the once-per-frame serial Remix pass -- the
              // largest indivisible block anywhere in the frame. Normalised to
              // ms/frame by the report's own frame counter rather than this
              // thread's chunk count, which is not a frame count.
              {
                const uint32_t rframes = perfreport::currentFrame() - csLastReportFrame;
                csLastReportFrame = perfreport::currentFrame();
                if (rframes > 0u) {
                  perfreport::publishWindow(perfreport::Slot::CsExecMs,
                    double(csTotalNs) / 1.0e6, rframes);
                  perfreport::publishWindow(perfreport::Slot::CsIdleMs,
                    double(csIdleNs) / 1.0e6, rframes);
                  perfreport::publishWindow(perfreport::Slot::CsFatChunkMs,
                    double(csBucketNs[kCsBuckets - 1]) / 1.0e6, rframes);
                }
                perfreport::publish(perfreport::Slot::CsBusyPct,
                  (winMs > 0.0 ? (double(csTotalNs) / 1.0e6) * 100.0 / winMs : 0.0));
              }

              std::string line = str::format(
                "[Perf.CsSplit] window=", winMs, "ms chunks=", csChunks,
                " execMs=", double(csTotalNs) / 1e6,
                " idleMs=", double(csIdleNs) / 1e6,
                " busyPct=", winMs > 0.0 ? (double(csTotalNs) / 1e6) * 100.0 / winMs : 0.0,
                " maxChunkMs=", double(csMaxNs) / 1e6, " | byDuration:");
              for (uint32_t i = 0; i < kCsBuckets; ++i) {
                line += str::format(" ", kBucketNames[i], "=",
                  double(csBucketNs[i]) / 1e6, "ms/", csBucketN[i]);
              }
              Logger::warn(line);

              // NV-DXVK [Perf.SessionState]: measure THIS thread's execution
              // speed once per window -- the uniform 3-4x whole-session CPU
              // slowdowns (08-08, 08-09) were unattributable because every
              // instrument assumed a constant millisecond. Contract at the
              // declaration in rtx_perf_report.h.
              perfreport::sessionStateProbe("dxvk-cs");

              for (uint32_t i = 0; i < kCsBuckets; ++i) {
                csBucketNs[i] = 0;
                csBucketN[i]  = 0;
              }
              csTotalNs = csChunks = csMaxNs = csIdleNs = 0;
            }
          } else {
            chunk->executeAll(m_context.ptr());
          }
        }
      }
    } catch (const DxvkError& e) {
      Logger::err("Exception on CS thread!");
      Logger::err(e.message());

      // NV-DXVK start: notify user and kill process on exception in CS thread to avoid silent hangs
      char buf[2048];
      snprintf(buf, sizeof(buf), "Exception on CS thread: %s. The game will exit now.", e.message().c_str());
      messageBox(buf, "RTX Remix", MB_OK);
      exit(1);
      // NV-DXVK end
    }
  }
  
}