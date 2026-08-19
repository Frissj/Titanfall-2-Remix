#include "d3d11_cmdlist.h"
#include "d3d11_context_imm.h"
#include "d3d11_device.h"
#include "d3d11_texture.h"
#include "d3d11_vanish_diag.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef _WIN32
// RtlCaptureStackBackTrace / GetModuleHandleA - for the [Perf.SyncSite] diag.
#  include <windows.h>
#endif

// NV-DXVK [perf]: the implicit-flush policy, made runtime-tunable.
//
// [Perf.Gpu] accounts for the whole frame exactly (idle + fenceWait + reap ==
// frame time, every window), and GPU idle tracks the number of command-list
// submissions almost linearly:
//
//   cmdLists/frame  13.0  15.3  21.7  22.8  23.2
//   gpuIdleMs        36    91   217   259   253
//
// So the frame is being chopped into many small submissions with the GPU
// bubbling between them. What that correlation cannot say is which way the
// causality runs: more submissions could be starving the GPU, or a frame that
// is already long could simply give the 750us flush timer more chances to fire.
// These knobs break the tie — raising them forces fewer, larger submissions
// without touching anything else, so if idle falls the fragmentation was the
// cause, and if it does not the submissions were a symptom.
//
// Env-read once so a sweep needs no rebuild. Defaults are the stock DXVK values.
static uint32_t GetFlushTuning(const char* name, uint32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0')
    return fallback;
  const long parsed = std::strtol(v, nullptr, 10);
  return (parsed > 0) ? uint32_t(parsed) : fallback;
}

static const uint32_t MinFlushIntervalUs = GetFlushTuning("RTX_FLUSH_MIN_US",      750);
static const uint32_t IncFlushIntervalUs = GetFlushTuning("RTX_FLUSH_INC_US",      250);
static const uint32_t MaxPendingSubmits  = GetFlushTuning("RTX_FLUSH_MAX_PENDING",   6);

constexpr static VkDeviceSize MaxImplicitDiscardSize = 256ull << 10;

#include "../dxvk/dxvk_bone_diag.h"

namespace dxvk {

  // NV-DXVK [Stage0 pipeline probe]: game-thread frame-boundary block accounting.
  // The draw phase (~191ms, game thread) and the RTX inject (~102ms, CS thread)
  // currently run serially (~370ms/frame, ~2.7fps). Before redesigning the frame
  // pipeline we must know WHERE the game thread actually stalls at the boundary.
  // These count the two primitive waits the game thread can block on. Written only
  // on the (relatively rare) blocking paths, read+reset once per window by the
  // D3D11Rtx perf logger as [Perf.GameBlock].
  std::atomic<int64_t>  g_gtWaitResNs   { 0 };  // ns blocked in device->waitForResource (GPU/Map contention)
  std::atomic<uint32_t> g_gtWaitResN    { 0 };  // # of blocking resource waits
  std::atomic<uint32_t> g_gtWaitResDiscN{ 0 };  // ...of those, DISCARD maps (discard-rename SHOULD have avoided the wait)
  std::atomic<int64_t>  g_gtSyncCsNs    { 0 };  // ns blocked in m_csThread.synchronize (CS-thread catch-up)
  std::atomic<uint32_t> g_gtSyncCsN     { 0 };  // # of CS syncs that actually waited (>10us)

  D3D11ImmediateContext::D3D11ImmediateContext(
          D3D11Device*    pParent,
    const Rc<DxvkDevice>& Device)
  : D3D11DeviceContext(pParent, Device, DxvkCsChunkFlag::SingleUse),
    m_csThread(Device, Device->createRtxContext()),
    m_videoContext(this, Device) {
    EmitCs([
      cDevice                 = m_device,
      cRelaxedBarriers        = pParent->GetOptions()->relaxedBarriers,
      cIgnoreGraphicsBarriers = pParent->GetOptions()->ignoreGraphicsBarriers
    ] (DxvkContext* ctx) {
      ctx->beginRecording(cDevice->createCommandList());

      DxvkBarrierControlFlags barrierControl;

      if (cRelaxedBarriers)
        barrierControl.set(DxvkBarrierControl::IgnoreWriteAfterWrite);

      if (cIgnoreGraphicsBarriers)
        barrierControl.set(DxvkBarrierControl::IgnoreGraphicsBarriers);

      ctx->setBarrierControl(barrierControl);
    });
    
    ClearState();

    m_rtx.Initialize();
  }
  
  
  D3D11ImmediateContext::~D3D11ImmediateContext() {
    Flush();
    SynchronizeCsThread(DxvkCsThread::SynchronizeAll);
    SynchronizeDevice();
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::QueryInterface(REFIID riid, void** ppvObject) {
    if (riid == __uuidof(ID3D11VideoContext)) {
      *ppvObject = ref(&m_videoContext);
      return S_OK;
    }

    return D3D11DeviceContext::QueryInterface(riid, ppvObject);
  }


  D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE D3D11ImmediateContext::GetType() {
    return D3D11_DEVICE_CONTEXT_IMMEDIATE;
  }
  
  
  UINT STDMETHODCALLTYPE D3D11ImmediateContext::GetContextFlags() {
    return 0;
  }
  
  
  // NV-DXVK [Perf.SyncSite]: WHO is spinning, and what does each one cost?
  //
  // After the spin-burst flush fix, GetData is still ~43 ms/frame and ~90% of
  // all immediate-context CPU, at ~300-400k polls/frame, 99.998% not-ready,
  // 100% type0 = EVENT. [Perf.Query] objSwitches says that resolves to only
  // ~6-7 distinct query OBJECTS per frame, i.e. a handful of full CPU->GPU sync
  // points. Attributing the 43 ms to those call sites is the prerequisite for
  // removing any of them - a total tells us nothing about which sync to attack.
  //
  // The instrument captures a stack ONCE PER BURST, not per poll: ~7 captures a
  // frame against ~350k polls, so it cannot distort what it measures (this is
  // the trap the [Perf.Entry] ScopedCall on this same function fell into, where
  // 800k clock reads cost ~20 ms/frame to measure a 54 ms number). Sites are
  // deduped by FNV-1a over the engine frames, same idiom as [DropStack].
  //
  // Set RTX_SYNC_DIAG=0 to disable without a rebuild.
  static bool GetDiagFlag(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
      return fallback;
    return std::strtol(v, nullptr, 10) != 0;
  }

  static const bool SyncDiagEnabled = GetDiagFlag("RTX_SYNC_DIAG", true);

  namespace {
    struct SyncSite {
      uint64_t sig      = 0;
      uint64_t bursts   = 0;
      uint64_t polls    = 0;
      uint64_t micros   = 0;
      USHORT   nFrames  = 0;
      void*    frames[24] = {};
      bool     stackLogged = false;
    };

    constexpr uint32_t kMaxSyncSites = 16;
    SyncSite   g_syncSites[kMaxSyncSites];
    uint32_t   g_syncSiteCount = 0;
    std::mutex g_syncMutex;

    // Resolve the CURRENT stack to a site index, inserting if new. Returns -1
    // when the table is full or the capture failed - callers then just skip
    // accounting for that burst rather than mis-attributing it.
    int32_t SyncDiagResolveSite() {
      void* frames[24];
      // Skip 1: this helper itself. GetData and the d3d11 vtable thunk are kept
      // because they are cheap to eyeball and confirm the capture is sane.
      const USHORT n = RtlCaptureStackBackTrace(1u, 24u, frames, nullptr);
      if (n == 0)
        return -1;

      // Signature over the frames ABOVE dxvk (skip 3: helper's caller chain
      // through D3D11ImmediateContext::GetData and the ID3D11DeviceContext
      // thunk) so distinct ENGINE sync sites are what we dedup on.
      uint64_t sig = 1469598103934665603ull;
      for (USHORT k = (n > 3u ? 3u : 0u); k < n; ++k) {
        sig ^= reinterpret_cast<uint64_t>(frames[k]);
        sig *= 1099511628211ull;
      }

      std::lock_guard<std::mutex> g(g_syncMutex);
      for (uint32_t i = 0; i < g_syncSiteCount; ++i) {
        if (g_syncSites[i].sig == sig)
          return int32_t(i);
      }
      if (g_syncSiteCount >= kMaxSyncSites)
        return -1;

      const uint32_t idx = g_syncSiteCount++;
      g_syncSites[idx].sig     = sig;
      g_syncSites[idx].nFrames = n;
      for (USHORT k = 0; k < n; ++k)
        g_syncSites[idx].frames[k] = frames[k];
      return int32_t(idx);
    }

    // NV-DXVK [Perf.SyncSite]: real image size from the PE header, so a frame
    // is only attributed to a module if it actually lies inside it.
    //
    // The first capture used a flat 0x40000000 window per module and produced
    // "eng+0x1fe6ba3b" - a 536 MB RVA into an engine.dll that is nowhere near
    // that size. A too-permissive window silently claims addresses belonging to
    // unenumerated modules (ntdll, kernel32, ...), which is worse than leaving
    // them as "?" because it reads as a real answer.
    uint64_t SyncDiagModuleSize(uint64_t base) {
      if (!base)
        return 0;
      const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return 0;
      const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE)
        return 0;
      return nt->OptionalHeader.SizeOfImage;
    }

    std::string SyncDiagFormatStack(const SyncSite& site) {
      struct Mod { const char* name; uint64_t base; };
      Mod mods[7] = {
        { "eng",    reinterpret_cast<uint64_t>(GetModuleHandleA("engine.dll")) },
        { "cli",    reinterpret_cast<uint64_t>(GetModuleHandleA("client.dll")) },
        { "mat",    reinterpret_cast<uint64_t>(GetModuleHandleA("materialsystem_dx11.dll")) },
        { "shaderapi", reinterpret_cast<uint64_t>(GetModuleHandleA("shaderapidx11.dll")) },
        { "studio", reinterpret_cast<uint64_t>(GetModuleHandleA("studiorender.dll")) },
        { "server", reinterpret_cast<uint64_t>(GetModuleHandleA("server.dll")) },
        { "d3d11",  reinterpret_cast<uint64_t>(GetModuleHandleA("d3d11.dll")) },
      };
      uint64_t sizes[7];
      for (int m = 0; m < 7; ++m)
        sizes[m] = SyncDiagModuleSize(mods[m].base);

      std::string out;
      out.reserve(900);
      for (USHORT k = 0; k < site.nFrames; ++k) {
        const uint64_t addr = reinterpret_cast<uint64_t>(site.frames[k]);
        const char* mod = "?";
        uint64_t rva = addr, bestBase = 0;
        for (int m = 0; m < 7; ++m) {
          if (mods[m].base && sizes[m]
              && addr >= mods[m].base && addr < mods[m].base + sizes[m]
              && mods[m].base > bestBase) {
            bestBase = mods[m].base; mod = mods[m].name; rva = addr - mods[m].base;
          }
        }
        if (!out.empty()) out += " | ";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s+0x%llx", mod,
                      static_cast<unsigned long long>(rva));
        out += buf;
      }

      // Bases and sizes, so an RVA can be sanity-checked against the module it
      // was attributed to (and so "?" frames can be chased if one matters).
      out += "  [";
      for (int m = 0; m < 7; ++m) {
        if (!mods[m].base)
          continue;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s@0x%llx+0x%llx ", mods[m].name,
                      static_cast<unsigned long long>(mods[m].base),
                      static_cast<unsigned long long>(sizes[m]));
        out += buf;
      }
      out += "]";
      return out;
    }

    // NV-DXVK [Perf.QEndSite]: name the engine call sites that call End() on a
    // query, because the frame is spent immediately after them.
    //
    // Measured 2026-07-27 with draws finally timed:
    //   [Perf.Entry]  26 ms/frame   (draws 24 ms of it)
    //   [Perf.Gap]   104-113 ms/frame, of which afterQueryEnd = 101-111 ms
    //                spread over exactly 6 gaps -> ~17 ms per QueryEnd
    // and 26 + 104 reconstructs the ~130 ms frame, so nothing else is missing.
    // Adding draws to the timed set did NOT shrink afterQueryEnd, which killed
    // the theory that the gap was the untimed draw batch.
    //
    // So six times a frame the engine ends a query and then spends ~17 ms
    // somewhere that is not any D3D11 entry point. A CPU sampling capture put
    // the frame thread 80% in the kernel with d3d11.dll as the calling module,
    // so this is a blocking wait, not engine compute.
    //
    // Only the call site can say what it is waiting for. QueryEnd runs 6 times a
    // frame, so a full stack capture per call is free here - unlike the GetData
    // path above, which needed per-burst sampling to stay affordable.
    // Deliberately a separate table from g_syncSites: those entries carry
    // burst/poll accounting that [Perf.SyncSite] reports, and salting them with
    // zero-poll QueryEnd sites would make that report lie.
    constexpr uint32_t kMaxQEndSites = 12;
    SyncSite   g_qendSites[kMaxQEndSites];
    uint32_t   g_qendSiteCount = 0;
    std::mutex g_qendMutex;

    void QEndDiagNoteSite(uint32_t queryType) {
      void* frames[24];
      const USHORT n = RtlCaptureStackBackTrace(1u, 24u, frames, nullptr);
      if (n == 0)
        return;

      uint64_t sig = 1469598103934665603ull;
      for (USHORT k = (n > 3u ? 3u : 0u); k < n; ++k) {
        sig ^= reinterpret_cast<uint64_t>(frames[k]);
        sig *= 1099511628211ull;
      }

      std::string line;
      uint64_t    hits = 0;
      {
        std::lock_guard<std::mutex> g(g_qendMutex);

        uint32_t idx = kMaxQEndSites;
        for (uint32_t i = 0; i < g_qendSiteCount; ++i) {
          if (g_qendSites[i].sig == sig) { idx = i; break; }
        }

        if (idx == kMaxQEndSites) {
          if (g_qendSiteCount >= kMaxQEndSites)
            return;
          idx = g_qendSiteCount++;
          g_qendSites[idx].sig     = sig;
          g_qendSites[idx].nFrames = n;
          for (USHORT k = 0; k < n; ++k)
            g_qendSites[idx].frames[k] = frames[k];
        }

        g_qendSites[idx].polls += 1;
        hits = g_qendSites[idx].polls;

        // Log each distinct site once on discovery. Six sites a frame at 7 fps
        // would be 2500 lines/minute if logged unconditionally.
        if (!g_qendSites[idx].stackLogged) {
          g_qendSites[idx].stackLogged = true;
          line = SyncDiagFormatStack(g_qendSites[idx]);
        }
      }

      if (!line.empty()) {
        // type: 0=EVENT 1=OCCLUSION 2=TIMESTAMP 3=TIMESTAMP_DISJOINT
        //       4=PIPELINE_STATS 5=OCCLUSION_PREDICATE 6+=SO_STATS/OVERFLOW
        Logger::warn(str::format(
          "[Perf.QEndSite] newSite#", g_qendSiteCount - 1,
          " type=", queryType, " hits=", hits,
          " stack: ", line));
      }
    }
  }

  // Open burst state. Immediate-context calls are frame-thread only, but keep
  // this thread_local so a stray call from elsewhere cannot corrupt the totals.
  static thread_local int32_t  t_burstSite  = -1;
  static thread_local uint64_t t_burstPolls = 0;
  static thread_local dxvk::high_resolution_clock::time_point t_burstStart;

  static void SyncDiagEndBurst();

  static void SyncDiagBeginBurst() {
    // Close any burst still open (the app moved to a different query object
    // without the previous one ever reporting ready).
    SyncDiagEndBurst();

    t_burstSite  = SyncDiagResolveSite();
    t_burstPolls = 0;
    t_burstStart = dxvk::high_resolution_clock::now();
  }

  static void SyncDiagEndBurst() {
    if (t_burstSite < 0)
      return;

    const auto elapsed = dxvk::high_resolution_clock::now() - t_burstStart;
    const uint64_t us = uint64_t(std::chrono::duration_cast<
      std::chrono::microseconds>(elapsed).count());

    {
      std::lock_guard<std::mutex> g(g_syncMutex);
      SyncSite& site = g_syncSites[t_burstSite];
      site.bursts += 1;
      site.polls  += t_burstPolls;
      site.micros += us;
    }

    t_burstSite  = -1;
    t_burstPolls = 0;
  }

  static void SyncDiagMaybeLog() {
    static thread_local auto s_lastLog = dxvk::high_resolution_clock::now();
    const auto now = dxvk::high_resolution_clock::now();
    if (now - s_lastLog < std::chrono::seconds(5))
      return;
    s_lastLog = now;

    // Copy under the lock, format and log outside it - Logger::warn does I/O.
    SyncSite snapshot[kMaxSyncSites];
    uint32_t count = 0;
    {
      std::lock_guard<std::mutex> g(g_syncMutex);
      count = g_syncSiteCount;
      for (uint32_t i = 0; i < count; ++i) {
        snapshot[i] = g_syncSites[i];
        g_syncSites[i].bursts = g_syncSites[i].polls = g_syncSites[i].micros = 0;
        g_syncSites[i].stackLogged = true;
      }
    }

    for (uint32_t i = 0; i < count; ++i) {
      const SyncSite& s = snapshot[i];
      Logger::warn(str::format(
        "[Perf.SyncSite] site", i,
        " bursts=", s.bursts,
        " polls=", s.polls,
        " ms=", double(s.micros) / 1000.0,
        " msPerBurst=", (s.bursts ? double(s.micros) / 1000.0 / double(s.bursts) : 0.0)));

      // The stack is invariant per site, so print it only the first time.
      if (!s.stackLogged)
        Logger::warn(str::format("[Perf.SyncSite] site", i, " stack: ",
                                 SyncDiagFormatStack(s)));
    }
  }


  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::GetData(
          ID3D11Asynchronous*               pAsync,
          void*                             pData,
          UINT                              DataSize,
          UINT                              GetDataFlags) {
    // NV-DXVK [perf]: THIS INSTRUMENT IS EXPENSIVE AT THIS CALL RATE.
    //
    // ScopedCall takes steady_clock::now() in its constructor AND destructor
    // plus three relaxed atomics. TF2 calls GetData ~400k times per frame, so
    // this single line costs ~800k clock reads - on the order of 20 ms/frame.
    //
    // Note what that does to the number it reports: the measured span runs from
    // after the first now() to before the second, so the ~20 ms of clock reads
    // is NOT included in the "GetData=54ms" figure in [Perf.Entry]. The true
    // cost of this entry point to the frame is therefore ~54 ms of body PLUS
    // ~20 ms of measuring it. Do not treat [Perf.Entry] GetData as the whole
    // bill, and do not conclude the instrument is free because the number it
    // prints looks self-consistent.
    //
    // Left enabled deliberately - it is the only per-entry-point CPU breakdown
    // that exists, and GetData is the largest item in the frame, so blinding it
    // would be worse than paying for it. If the overhead needs to go, sample it
    // (time 1 poll in N, scale by N) rather than removing it; at 400k calls a
    // 1-in-64 sample still yields ~6000 samples/frame. That changes what every
    // [Perf.Entry] number means, so it is a deliberate decision, not a cleanup.
    vanish_diag::ScopedCall vdScope_GetData(vanish_diag::GetData);
    if (!pAsync || (DataSize && !pData))
      return E_INVALIDARG;
    
    // Check whether the data size is actually correct
    if (DataSize && DataSize != pAsync->GetDataSize())
      return E_INVALIDARG;
    
    // Passing a non-null pData is actually allowed if
    // DataSize is 0, but we should ignore that pointer
    pData = DataSize ? pData : nullptr;

    // Get query status directly from the query object
    auto query = static_cast<D3D11Query*>(pAsync);
    HRESULT hr = query->GetData(pData, GetDataFlags);

    // NV-DXVK [perf]: [Perf.Entry] measured ~1e6 GetData calls per frame costing
    // 80-130 ms — the largest single item in the frame, bigger than all of
    // OnDraw* or EndFrame. That is the game spinning on a query that is not
    // becoming ready. Attribute it: which query type, how many distinct query
    // objects, what fraction of polls return S_FALSE (not-ready), and whether the
    // game is passing DONOTFLUSH (which suppresses our stall handling). One line
    // per 5 s, on the frame thread only.
    {
      constexpr uint32_t kQueryTypes = 16;  // covers the whole D3D11_QUERY enum

      static thread_local uint64_t s_pollTotal = 0, s_pollNotReady = 0, s_pollDoNotFlush = 0;
      static thread_local uint64_t s_byType[kQueryTypes] = {};
      static thread_local const void* s_lastQuery = nullptr;
      static thread_local uint32_t s_lastQueryType = 0;
      static thread_local uint64_t s_queryChanges = 0;
      static thread_local auto s_lastLog = dxvk::high_resolution_clock::now();

      // GetDesc1 is a virtual call plus a struct copy; at ~1e6 calls/frame that
      // would itself distort what we are measuring. The type is a property of the
      // query object, and a spin loop hammers the same object, so memoise on the
      // pointer — GetDesc1 then runs once per switch, not once per poll.
      if (pAsync != s_lastQuery) {
        s_lastQuery = pAsync;
        ++s_queryChanges;

        D3D11_QUERY_DESC1 qd = {};
        query->GetDesc1(&qd);
        s_lastQueryType = (uint32_t(qd.Query) < kQueryTypes) ? uint32_t(qd.Query) : 0u;
      }

      ++s_pollTotal;
      ++s_byType[s_lastQueryType];
      if (hr == S_FALSE)                                  ++s_pollNotReady;
      if (GetDataFlags & D3D11_ASYNC_GETDATA_DONOTFLUSH)  ++s_pollDoNotFlush;

      // Reading the clock on every poll would cost more than everything else
      // here combined (~25 ns x 1e6 calls = ~25 ms/frame). Check it once every
      // 4096 polls instead; at these rates that is still many checks per frame.
      if ((s_pollTotal & 0xFFFu) == 0u
       && dxvk::high_resolution_clock::now() - s_lastLog >= std::chrono::seconds(5)) {
        s_lastLog = dxvk::high_resolution_clock::now();

        std::string types;
        for (uint32_t t = 0; t < kQueryTypes; ++t) {
          if (s_byType[t] != 0)
            types += str::format(" type", t, "=", s_byType[t]);
        }

        Logger::warn(str::format(
          "[Perf.Query] flushTuning(min=", MinFlushIntervalUs,
          " inc=", IncFlushIntervalUs,
          " maxPending=", MaxPendingSubmits, ")",
          " polls=", s_pollTotal,
          " notReady=", s_pollNotReady,
          " notReadyPct=", (s_pollTotal ? double(s_pollNotReady) * 100.0 / double(s_pollTotal) : 0.0),
          " doNotFlush=", s_pollDoNotFlush,
          " objSwitches=", s_queryChanges,
          " |", types,
          "  (type0=EVENT 1=OCCLUSION 2=TIMESTAMP 5=OCCLUSION_PREDICATE)"));

        s_pollTotal = s_pollNotReady = s_pollDoNotFlush = s_queryChanges = 0;
        for (auto& c : s_byType) c = 0;
      }
    }

    // If we're likely going to spin on the asynchronous object,
    // flush the context so that we're keeping the GPU busy.
    if (hr == S_FALSE) {
      // Don't mark the event query as stalling if the app does
      // not intend to spin on it. This reduces flushes on End.
      if (!(GetDataFlags & D3D11_ASYNC_GETDATA_DONOTFLUSH))
        query->NotifyStall();

      // Ignore the DONOTFLUSH flag here as some games will spin
      // on queries without ever flushing the context otherwise.
      //
      // NV-DXVK [perf]: rate-limited. This used to call FlushImplicit(FALSE)
      // unconditionally on every not-ready poll, and FlushImplicit reads
      // high_resolution_clock::now() every time it is entered.
      //
      // TF2 polls EVENT queries ~400k times per frame at 99.9984% not-ready
      // ([Perf.Query]), so that was ~400k clock reads per frame - roughly
      // 10 ms - to make a decision that CANNOT come out true more than once
      // per MinFlushIntervalUs (5000 us here, so ~36 times in a 180 ms frame).
      // The instrumentation block above this one already avoids exactly this
      // trap for its own clock read, with the note that ~25 ns x 1e6 calls
      // would cost more than everything else combined; the real code path was
      // doing it anyway twenty lines below.
      //
      // Gating on poll COUNT rather than time keeps the fast path free of any
      // clock read. Every 64th poll still leaves ~6000 FlushImplicit calls per
      // frame against a ceiling of ~36 possible flushes, so no flush that
      // would have happened is lost or meaningfully delayed.
      //
      // The counter resets when the polled query object changes, so the FIRST
      // poll of any new spin burst always flushes immediately. That is what
      // keeps this safe for applications that poll only a handful of times per
      // frame, where a pure every-64th gate would starve the flush.
      // NV-DXVK [Perf.Spin]: the FIRST not-ready poll of a spin burst flushes
      // NOW, bypassing FlushImplicit's timer entirely.
      //
      // FlushImplicit only submits if `now - m_lastFlush >= MinFlushIntervalUs
      // + IncFlushIntervalUs * pending`, and StrongHint does NOT bypass that
      // timer - it only bypasses the pending-submission count. So a game that
      // ends an EVENT query and immediately spins on it could not get its work
      // submitted for up to a full flush interval, no matter how hard it asked.
      //
      // Measured on TF2 ([Perf.Entry] / [Perf.Block] / [Perf.Query]):
      //   GetData   = 67.77 ms of a 74.29 ms immediate-context frame (91%)
      //   polls     = 404469/frame, 99.9986% not-ready, all type0 = EVENT
      //   submits   = 7 per frame
      //   gpuIdleMs = 44
      // The GPU was idle because we would not hand it work, and an idle laptop
      // GPU downclocks, which then inflates every GPU-side pass measurement.
      //
      // Rate limiting flushes is correct when the app is MAKING PROGRESS - it
      // stops the frame being chopped into tiny submissions. An app blocked on
      // a fence is the opposite case: there is no progress to protect, and the
      // only thing that can unblock it is the submission we are withholding.
      //
      // Cost is bounded by how often the polled OBJECT changes, not by poll
      // count: [Perf.Query] reports objSwitches=120-162 per 5 s against 7-11
      // MILLION polls, i.e. ~5-7 per frame. So this adds ~5-7 submits/frame on
      // top of the current 7 - still a normal D3D11 submission rate. Flush()
      // is itself a no-op when nothing is queued (it checks m_csIsBusy and the
      // chunk), so bursts with no pending work cost nothing.
      //
      // Subsequent polls within the same burst keep the cheap every-64th
      // rate-limited path: by then the work is already submitted and repeated
      // flushing would achieve nothing.
      {
        static thread_local const void* s_spinQuery = nullptr;
        static thread_local uint32_t    s_spinPolls = 0;

        const bool newSpinBurst = (pAsync != s_spinQuery);

        if (newSpinBurst) {
          s_spinQuery = pAsync;
          s_spinPolls = 0;

          // NV-DXVK [Perf.SyncSite]: one stack capture per burst, ~7/frame.
          if (SyncDiagEnabled)
            SyncDiagBeginBurst();
        }

        if (newSpinBurst)
          Flush();
        else if ((s_spinPolls & 0x3Fu) == 0u)
          FlushImplicit(FALSE);

        ++s_spinPolls;
      }

      if (SyncDiagEnabled)
        ++t_burstPolls;
    } else if (SyncDiagEnabled) {
      // Query became ready - the burst that was waiting on it is over. This is
      // the normal close; SyncDiagBeginBurst also force-closes a burst that the
      // app abandoned by moving to a different query object.
      SyncDiagEndBurst();
      SyncDiagMaybeLog();
    }

    return hr;
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::Begin(ID3D11Asynchronous* pAsync) {
    vanish_diag::ScopedCall vdScope_QueryBegin(vanish_diag::QueryBegin);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pAsync))
      return;

    auto query = static_cast<D3D11Query*>(pAsync);

    if (unlikely(!query->DoBegin()))
      return;

    EmitCs([cQuery = Com<D3D11Query, false>(query)]
    (DxvkContext* ctx) {
      cQuery->Begin(ctx);
    });
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::End(ID3D11Asynchronous* pAsync) {
    vanish_diag::ScopedCall vdScope_QueryEnd(vanish_diag::QueryEnd);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pAsync))
      return;

    auto query = static_cast<D3D11Query*>(pAsync);

    // [Perf.QEndSite] / [Perf.Gap]: which engine call site is this, and what
    // kind of query. GetDesc1 is a virtual call plus a struct copy -
    // unaffordable on the GetData spin path above, but this runs 6 times a
    // frame.
    //
    // t_lastQueryEndType is what lets [Perf.Gap] split its flat afterQueryEnd
    // bucket by type. It must be set unconditionally (not just when the diag is
    // on) or the split silently attributes every gap to type 0.
    {
      D3D11_QUERY_DESC1 qd = {};
      query->GetDesc1(&qd);
      vanish_diag::t_lastQueryEndType = uint32_t(qd.Query);

      if (SyncDiagEnabled)
        QEndDiagNoteSite(uint32_t(qd.Query));
    }

    if (unlikely(!query->DoEnd())) {
      EmitCs([cQuery = Com<D3D11Query, false>(query)]
      (DxvkContext* ctx) {
        cQuery->Begin(ctx);
      });
    }

    EmitCs([cQuery = Com<D3D11Query, false>(query)]
    (DxvkContext* ctx) {
      cQuery->End(ctx);
    });

    if (unlikely(query->IsEvent())) {
      query->NotifyEnd();

      // NV-DXVK [Perf.QEvent]: which branch an EVENT-query End actually takes.
      //
      // This is the decision that gates the frame. Measured 2026-07-27:
      //   [Perf.Busy] frame thread   blockedMs 105-113/frame, busyPct 26-33%
      //   [Perf.Gap]  afterQueryEnd  110-124 ms/frame across 6 gaps
      //   [Perf.SubmitGap]           5.3 submits/frame, inSubmitMs 0.03,
      //                              gapMsMax 130-197 ms
      //   [Perf.QEndSite]            both sites type=0 (EVENT), from
      //                              materialsystem_dx11, common root mat+0x87f9f
      // So the engine ends a fence and then BLOCKS ~110 ms while the GPU has
      // nothing queued. blockedMs and afterQueryEnd agreeing to within a few ms
      // is two independent instruments measuring the same wait.
      //
      // IsStalling() only becomes true once the app has been SEEN spinning in
      // GetData on this context. The [Perf.Spin] path above (first not-ready
      // poll flushes immediately) was built for exactly that case - but the
      // frame thread now issues ~1 GetData per frame, so neither that fix nor
      // this flag can engage. If stallingTrue is 0 while eventEnds is ~6/frame,
      // every fence is taking the timer-gated FlushImplicit path and the work
      // the engine is waiting for is being withheld until the flush timer
      // fires, which is the mechanism the comment block above predicted.
      //
      // Counted, not assumed: this distinguishes "we are withholding the
      // submission" from "we submitted promptly and the GPU genuinely took
      // 110 ms", and those two demand opposite fixes.
      const bool stalling = query->IsStalling();
      {
        static std::atomic<uint64_t> s_eventEnds { 0 };
        static std::atomic<uint64_t> s_stallingTrue { 0 };
        static std::atomic<uint64_t> s_lastReportNs { 0 };
        static const auto s_epoch = dxvk::high_resolution_clock::now();

        s_eventEnds.fetch_add(1, std::memory_order_relaxed);
        if (stalling)
          s_stallingTrue.fetch_add(1, std::memory_order_relaxed);

        const uint64_t nowNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          dxvk::high_resolution_clock::now() - s_epoch).count());
        uint64_t last = s_lastReportNs.load(std::memory_order_relaxed);

        if (nowNs - last >= 5000000000ull
         && s_lastReportNs.compare_exchange_strong(last, nowNs, std::memory_order_relaxed)) {
          const uint64_t ends  = s_eventEnds.exchange(0, std::memory_order_relaxed);
          const uint64_t stall = s_stallingTrue.exchange(0, std::memory_order_relaxed);
          Logger::warn(str::format(
            "[Perf.QEvent] window=", double(nowNs - last) / 1.0e9, "s",
            " eventEnds=", ends,
            " tookFlush(stalling)=", stall,
            " tookFlushImplicit=", ends - stall));
        }
      }

      stalling
        ? Flush()
        : FlushImplicit(TRUE);
    }
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::Flush() {
    Flush1(D3D11_CONTEXT_TYPE_ALL, nullptr);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::Flush1(
          D3D11_CONTEXT_TYPE          ContextType,
          HANDLE                      hEvent) {
    m_parent->FlushInitContext();

    if (hEvent)
      SignalEvent(hEvent);
    
    D3D11DeviceLock lock = LockContext();
    
    if (m_csIsBusy || !m_csChunk->empty()) {
      // Add commands to flush the threaded
      // context, then flush the command list
      EmitCs([] (DxvkContext* ctx) {
        ctx->flushCommandList();
      });
      
      FlushCsChunk();
      
      // Reset flush timer used for implicit flushes
      m_lastFlush = dxvk::high_resolution_clock::now();
      m_csIsBusy  = false;
    }
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Signal(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    Logger::err("D3D11ImmediateContext::Signal: Not implemented");
    return E_NOTIMPL;
  }


  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Wait(
          ID3D11Fence*                pFence,
          UINT64                      Value) {
    Logger::err("D3D11ImmediateContext::Wait: Not implemented");
    return E_NOTIMPL;
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::ExecuteCommandList(
          ID3D11CommandList*  pCommandList,
          BOOL                RestoreContextState) {
    vanish_diag::ScopedCall vdScope(vanish_diag::ExecCmdList);
    D3D11DeviceLock lock = LockContext();

    auto commandList = static_cast<D3D11CommandList*>(pCommandList);

    // Flush any outstanding commands so that
    // we don't mess up the execution order
    FlushCsChunk();

    // As an optimization, flush everything if the
    // number of pending draw calls is high enough.
    FlushImplicit(FALSE);

    // NV-DXVK: Fold the deferred-context RTX draw count recorded into the
    // command list (see D3D11DeferredContext::FinishCommandList) into our
    // own D3D11Rtx counter BEFORE replaying the chunks.  The chunks carry
    // EmitCs lambdas that call commitGeometryToRT on the CS thread, so the
    // geometry itself reaches the RTX pipeline via the normal CS path; the
    // only thing the immediate context needs to know is the total draw
    // count so that D3D11Rtx::EndFrame sees a non-zero value and the
    // kMaxConcurrentDraws throttle stays accurate.  This is what makes
    // Source-engine games (Titanfall 2, etc.) that record all of their
    // material draws onto worker-thread deferred contexts actually show
    // up in Remix's raytraced composite.
    const uint32_t clRtxDraws = commandList->GetRtxDrawCount();
    m_rtx.addDrawCallID(clRtxDraws);

    // NV-DXVK: diagnostic — first few ExecuteCommandList calls log the
    // recorded RTX draw count so we can see whether deferred contexts are
    // actually producing draws that pass SubmitDraw's pre-filters.
    static uint32_t s_execCount = 0;
    const uint32_t n = ++s_execCount;
    if (n <= 8) {
      Logger::info(str::format(
          "[D3D11ImmediateContext] ExecuteCommandList #", n,
          " rtxDraws=", clRtxDraws));
    }

    // NV-DXVK TF2 viewmodel diagnostic: snapshot the immediate context's
    // current VS cb2 contents at ExecuteCommandList time, BEFORE the
    // recorded deferred chunks fire. Compare against [VMHunt.cb2] which
    // captures cb2 at the deferred context's SubmitDraw recording time.
    //
    // Hypothesis: TF2 records gun + hands draws on a deferred context. Our
    // [VMHunt.cb2] capture happens on the deferred-context thread when the
    // draw is recorded. By the time ExecuteCommandList replays those chunks
    // on the immediate context, the game may have updated cb2 with the
    // CORRECT viewmodel camera (different origin, smaller maxZ frustum) via
    // the immediate context — and the GPU sees that update because cb
    // bindings carry through D3D11's versioning. If the immediate context's
    // cb2 origin here differs from what [VMHunt.cb2] logged for nearby
    // draws, we've been computing the viewmodel transform on stale data.
    //
    // Only logs when cb2 appears to hold a CBufCommonPerCamera-shaped
    // payload (offset 0 = zNear small, offset 4 = float3 cam origin
    // plausible). Throttled to ~120 calls per session and to changes (skip
    // if cb2 looks identical to the previous logged value) so this doesn't
    // flood the log.
    {
      static uint32_t sCb2LogCount = 0;
      static float sLastCamX = 1e30f, sLastCamY = 1e30f, sLastCamZ = 1e30f;
      const auto& vsCbs = m_state.vs.constantBuffers;
      // cb slot 2 is where Source engine binds CBufCommonPerCamera in TF2.
      if (sCb2LogCount < 120) {
        const auto& cb2 = vsCbs[2];
        if (cb2.buffer != nullptr) {
          const auto map = cb2.buffer->GetMappedSlice();
          const uint8_t* p = reinterpret_cast<const uint8_t*>(map.mapPtr);
          const size_t base = static_cast<size_t>(cb2.constantOffset) * 16;
          const size_t bufSize = cb2.buffer->Desc()->ByteWidth;
          // We need at least 16 bytes (zNear + camOrigin float3) to be
          // worth logging. Read 80 bytes if available so we can also dump
          // the c_cameraRelativeToClip first row for cross-checking.
          if (p && base + 16 <= bufSize) {
            const float* fp = reinterpret_cast<const float*>(p + base);
            const float zNear = fp[0];
            const float cx = fp[1], cy = fp[2], cz = fp[3];
            const bool plausible =
              std::isfinite(zNear) && std::isfinite(cx)
              && std::isfinite(cy) && std::isfinite(cz)
              && (std::abs(cx) > 1.0f || std::abs(cy) > 1.0f || std::abs(cz) > 1.0f);
            const bool changed =
              std::abs(sLastCamX - cx) > 0.5f
              || std::abs(sLastCamY - cy) > 0.5f
              || std::abs(sLastCamZ - cz) > 0.5f;
            if (plausible && changed) {
              ++sCb2LogCount;
              sLastCamX = cx; sLastCamY = cy; sLastCamZ = cz;
              // Also dump c2c row0 if buffer is large enough — helps confirm
              // whether projection matches the gameplay or viewmodel cam.
              float r0x = 0, r0y = 0, r0z = 0, r0w = 0;
              if (base + 32 <= bufSize) {
                const float* r0 = reinterpret_cast<const float*>(p + base + 16);
                r0x = r0[0]; r0y = r0[1]; r0z = r0[2]; r0w = r0[3];
              }
              Logger::info(str::format(
                "[ExecCL.cb2] #", sCb2LogCount,
                " execId=", n,
                " rtxDraws=", clRtxDraws,
                " cb2.constOff=", cb2.constantOffset,
                " cb2.bufSize=", bufSize,
                " zNear=", zNear,
                " camOrigin=(", cx, ",", cy, ",", cz, ")",
                " c2c_row0=(", r0x, ",", r0y, ",", r0z, ",", r0w, ")"));
            }
          }
        }
      }
    }

    // Dispatch command list to the CS thread and
    // restore the immediate context's state
    uint64_t csSeqNum = commandList->EmitToCsThread(&m_csThread);
    m_csSeqNum = std::max(m_csSeqNum, csSeqNum);

    if (RestoreContextState)
      RestoreState();
    else
      ClearState();

    // Mark CS thread as busy so that subsequent
    // flush operations get executed correctly.
    m_csIsBusy = true;
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::FinishCommandList(
          BOOL                RestoreDeferredContextState,
          ID3D11CommandList   **ppCommandList) {
    InitReturnPtr(ppCommandList);
    
    Logger::err("D3D11: FinishCommandList called on immediate context");
    return DXGI_ERROR_INVALID_CALL;
  }
  
  
  HRESULT STDMETHODCALLTYPE D3D11ImmediateContext::Map(
          ID3D11Resource*             pResource,
          UINT                        Subresource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    vanish_diag::ScopedCall vdScope(vanish_diag::Map);
    D3D11DeviceLock lock = LockContext();

    if (unlikely(!pResource))
      return E_INVALIDARG;
    
    D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pResource->GetType(&resourceDim);

    HRESULT hr;
    
    if (likely(resourceDim == D3D11_RESOURCE_DIMENSION_BUFFER)) {
      hr = MapBuffer(
        static_cast<D3D11Buffer*>(pResource),
        MapType, MapFlags, pMappedResource);
      // NV-DXVK: log Map calls on t30-sized buffers to find the bone
      // upload pattern (TF2's skinned-character bone matrices).
      auto* b = static_cast<D3D11Buffer*>(pResource);
      const uint32_t sz = b->Desc()->ByteWidth;

      // NV-DXVK [Perf.EntryCensus] / [Perf.EntryMap]: Map is one of the three
      // ways the ~50% "D3D11 entry points" share can be shaped, and the only
      // one whose fix is an upload-path change rather than a call-rate change.
      //
      // ByteWidth is the MAPPABLE size, not the bytes the game goes on to
      // write - that is unknowable from here, since the write happens in engine
      // code against the returned pointer. It is still the right axis: a
      // WRITE_DISCARD costs the rename of the whole allocation regardless of
      // how much of it is touched, which is exactly what this number is.
      vanish_diag::addBytes(vanish_diag::Map, sz);
      // Bind role, classified here because the D3D11_BIND_* constants are in
      // scope and the census header is not the place to know about them.
      // CONSTANT_BUFFER wins over VB/IB when a buffer carries both: the
      // question this bucket exists to answer is "are the buffers the
      // derivation reads being renamed", so a buffer that is ever a cbuffer
      // belongs in the cbuffer population.
      const UINT bindFlags = b->Desc()->BindFlags;
      const uint32_t bindRole =
          (bindFlags & D3D11_BIND_CONSTANT_BUFFER)                     ? 0u
        : (bindFlags & (D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_INDEX_BUFFER)) ? 1u
        :                                                                2u;
      vanish_diag::noteMapBuffer(uint32_t(MapType), sz, bindRole);

      if (tf2::boneDiagEnabled() && sz == 393216) {
        static uint32_t sMapBoneLog = 0;
        if (sMapBoneLog < 20) {
          ++sMapBoneLog;
          Logger::info(str::format(
            "[BoneMap.diag] Map t30-sized buffer",
            " ptr=", reinterpret_cast<uintptr_t>(b),
            " mapType=", uint32_t(MapType),
            " usage=", uint32_t(b->Desc()->Usage),
            " bindFlags=", uint32_t(b->Desc()->BindFlags),
            " mapPtrResult=", reinterpret_cast<uintptr_t>(pMappedResource ? pMappedResource->pData : nullptr)));
        }
      }
    } else {
      // [Perf.EntryMap]: counted separately and WITHOUT bytes. An image map
      // has no single byte width (it is a mip footprint), and MapImage is a
      // different code path with a different cost, so blending it into the
      // buffer KB/frame figure would corrupt the one number this axis exists
      // to produce.
      vanish_diag::noteMapImage();
      hr = MapImage(
        GetCommonTexture(pResource),
        Subresource, MapType, MapFlags,
        pMappedResource);
    }

    if (unlikely(FAILED(hr)))
      *pMappedResource = D3D11_MAPPED_SUBRESOURCE();

    return hr;
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::Unmap(
          ID3D11Resource*             pResource,
          UINT                        Subresource) {
    vanish_diag::ScopedCall vdScope(vanish_diag::Unmap);
    // Since it is very uncommon for images to be mapped compared
    // to buffers, we count the currently mapped images in order
    // to avoid a virtual method call in the common case.
    if (unlikely(m_mappedImageCount > 0)) {
      D3D11_RESOURCE_DIMENSION resourceDim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
      pResource->GetType(&resourceDim);

      if (resourceDim != D3D11_RESOURCE_DIMENSION_BUFFER) {
        D3D11DeviceLock lock = LockContext();
        UnmapImage(GetCommonTexture(pResource), Subresource);
      }
    }
  }

  // NV-DXVK [Perf.EntryCensus]: bytes moved by one UpdateSubresource.
  //
  // Buffers are exact - the box, or the whole buffer when there is no box.
  // Textures are an APPROXIMATION and are marked as such on the census line:
  // SrcDepthPitch is the full slice for a 2D/3D upload when the caller supplies
  // it, and SrcRowPitch is a floor of one row when it does not. Deliberately
  // not resolved through GetCommonTexture: this runs on the frame thread inside
  // the measured scope, and a wrong-by-a-mip byte count is worth far less than
  // not perturbing the thing being measured. If textures ever dominate this
  // bucket, that is the point to make it exact.
  static uint64_t UpdateSubBytes(
          ID3D11Resource*   pDstResource,
    const D3D11_BOX*        pDstBox,
          UINT              SrcRowPitch,
          UINT              SrcDepthPitch) {
    if (unlikely(pDstResource == nullptr))
      return 0;

    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pDstResource->GetType(&dim);

    if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) {
      if (pDstBox != nullptr)
        return pDstBox->right > pDstBox->left ? uint64_t(pDstBox->right - pDstBox->left) : 0;
      return uint64_t(static_cast<D3D11Buffer*>(pDstResource)->Desc()->ByteWidth);
    }

    return SrcDepthPitch != 0 ? uint64_t(SrcDepthPitch) : uint64_t(SrcRowPitch);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::UpdateSubresource(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch) {
    vanish_diag::ScopedCall vdScope(vanish_diag::UpdateSub);
    if (vanish_diag::t_isFrameThread)
      vanish_diag::addBytes(vanish_diag::UpdateSub,
        UpdateSubBytes(pDstResource, pDstBox, SrcRowPitch, SrcDepthPitch));
    UpdateResource<D3D11ImmediateContext>(this, pDstResource,
      DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, 0);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::UpdateSubresource1(
          ID3D11Resource*                   pDstResource,
          UINT                              DstSubresource,
    const D3D11_BOX*                        pDstBox,
    const void*                             pSrcData,
          UINT                              SrcRowPitch,
          UINT                              SrcDepthPitch,
          UINT                              CopyFlags) {
    vanish_diag::ScopedCall vdScope_UpdateSub(vanish_diag::UpdateSub);
    if (vanish_diag::t_isFrameThread)
      vanish_diag::addBytes(vanish_diag::UpdateSub,
        UpdateSubBytes(pDstResource, pDstBox, SrcRowPitch, SrcDepthPitch));
    UpdateResource<D3D11ImmediateContext>(this, pDstResource,
      DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch, CopyFlags);
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::OMSetRenderTargets(
          UINT                              NumViews,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView) {
    FlushImplicit(TRUE);
    
    D3D11DeviceContext::OMSetRenderTargets(
      NumViews, ppRenderTargetViews, pDepthStencilView);
  }
  
  
  void STDMETHODCALLTYPE D3D11ImmediateContext::OMSetRenderTargetsAndUnorderedAccessViews(
          UINT                              NumRTVs,
          ID3D11RenderTargetView* const*    ppRenderTargetViews,
          ID3D11DepthStencilView*           pDepthStencilView,
          UINT                              UAVStartSlot,
          UINT                              NumUAVs,
          ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT*                             pUAVInitialCounts) {
    FlushImplicit(TRUE);

    D3D11DeviceContext::OMSetRenderTargetsAndUnorderedAccessViews(
      NumRTVs, ppRenderTargetViews, pDepthStencilView,
      UAVStartSlot, NumUAVs, ppUnorderedAccessViews,
      pUAVInitialCounts);
  }
  
  
  HRESULT D3D11ImmediateContext::MapBuffer(
          D3D11Buffer*                pResource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    if (unlikely(!pMappedResource))
      return E_INVALIDARG;

    if (unlikely(pResource->GetMapMode() == D3D11_COMMON_BUFFER_MAP_MODE_NONE)) {
      Logger::err("D3D11: Cannot map a device-local buffer");
      return E_INVALIDARG;
    }

    VkDeviceSize bufferSize = pResource->Desc()->ByteWidth;

    if (likely(MapType == D3D11_MAP_WRITE_DISCARD)) {
      // Allocate a new backing slice for the buffer and set
      // it as the 'new' mapped slice. This assumes that the
      // only way to invalidate a buffer is by mapping it.
      auto physSlice = pResource->DiscardSlice();
      pMappedResource->pData      = physSlice.mapPtr;
      pMappedResource->RowPitch   = bufferSize;
      pMappedResource->DepthPitch = bufferSize;
      
      EmitCs([
        cBuffer      = pResource->GetBuffer(),
        cBufferSlice = physSlice
      ] (DxvkContext* ctx) {
        ctx->invalidateBuffer(cBuffer, cBufferSlice);
      });

      return S_OK;
    } else if (likely(MapType == D3D11_MAP_WRITE_NO_OVERWRITE)) {
      // Put this on a fast path without any extra checks since it's
      // a somewhat desired method to partially update large buffers
      // NV-DXVK [T31Cache]: this path hands back the EXISTING slice and the
      // game writes into it in place, so neither mapPtr nor the DiscardSlice
      // content generation moves. Anything caching these bytes needs a signal
      // that they may have changed; that is what m_mapGen is. One relaxed
      // atomic increment on a path that is about to have the game memcpy into
      // write-combined memory.
      pResource->NoteMapForWrite();
      DxvkBufferSliceHandle physSlice = pResource->GetMappedSlice();
      pMappedResource->pData      = physSlice.mapPtr;
      pMappedResource->RowPitch   = bufferSize;
      pMappedResource->DepthPitch = bufferSize;
      return S_OK;
    } else {
      // Quantum Break likes using MAP_WRITE on resources which would force
      // us to synchronize with the GPU multiple times per frame. In those
      // situations, if there are no pending GPU writes to the resource, we
      // can promote it to MAP_WRITE_DISCARD, but preserve the data by doing
      // a CPU copy from the previous buffer slice, to avoid the sync point.
      bool doInvalidatePreserve = false;

      auto buffer = pResource->GetBuffer();
      auto sequenceNumber = pResource->GetSequenceNumber();

      if (MapType != D3D11_MAP_READ && !MapFlags && bufferSize <= MaxImplicitDiscardSize) {
        SynchronizeCsThread(sequenceNumber);

        bool hasWoAccess = buffer->isInUse(DxvkAccess::Write);
        bool hasRwAccess = buffer->isInUse(DxvkAccess::Read);

        if (hasRwAccess && !hasWoAccess) {
          // Uncached reads can be so slow that a GPU sync may actually be faster
          doInvalidatePreserve = buffer->memFlags() & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
      }

      if (doInvalidatePreserve) {
        FlushImplicit(TRUE);

        auto prevSlice = pResource->GetMappedSlice();
        auto physSlice = pResource->DiscardSlice();

        EmitCs([
          cBuffer      = std::move(buffer),
          cBufferSlice = physSlice
        ] (DxvkContext* ctx) {
          ctx->invalidateBuffer(cBuffer, cBufferSlice);
        });

        std::memcpy(physSlice.mapPtr, prevSlice.mapPtr, physSlice.length);
        pMappedResource->pData      = physSlice.mapPtr;
        pMappedResource->RowPitch   = bufferSize;
        pMappedResource->DepthPitch = bufferSize;
        return S_OK;
      } else {
        if (!WaitForResource(buffer, sequenceNumber, MapType, MapFlags))
          return DXGI_ERROR_WAS_STILL_DRAWING;

        DxvkBufferSliceHandle physSlice = pResource->GetMappedSlice();
        pMappedResource->pData      = physSlice.mapPtr;
        pMappedResource->RowPitch   = bufferSize;
        pMappedResource->DepthPitch = bufferSize;
        return S_OK;
      }
    }
  }
  
  
  HRESULT D3D11ImmediateContext::MapImage(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource,
          D3D11_MAP                   MapType,
          UINT                        MapFlags,
          D3D11_MAPPED_SUBRESOURCE*   pMappedResource) {
    const Rc<DxvkImage>  mappedImage  = pResource->GetImage();
    const Rc<DxvkBuffer> mappedBuffer = pResource->GetMappedBuffer(Subresource);

    auto mapMode = pResource->GetMapMode();
    
    if (unlikely(mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_NONE)) {
      Logger::err("D3D11: Cannot map a device-local image");
      return E_INVALIDARG;
    }

    if (unlikely(Subresource >= pResource->CountSubresources()))
      return E_INVALIDARG;
    
    if (likely(pMappedResource != nullptr)) {
      // Resources with an unknown memory layout cannot return a pointer
      if (pResource->Desc()->Usage         == D3D11_USAGE_DEFAULT
       && pResource->Desc()->TextureLayout == D3D11_TEXTURE_LAYOUT_UNDEFINED)
        return E_INVALIDARG;
    } else {
      if (pResource->Desc()->Usage != D3D11_USAGE_DEFAULT)
        return E_INVALIDARG;
    }

    VkFormat packedFormat = m_parent->LookupPackedFormat(
      pResource->Desc()->Format, pResource->GetFormatMode()).Format;
    
    uint64_t sequenceNumber = pResource->GetSequenceNumber(Subresource);

    auto formatInfo = imageFormatInfo(packedFormat);
    void* mapPtr;

    if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_DIRECT) {
      // Wait for the resource to become available. We do not
      // support image renaming, so stall on DISCARD instead.
      if (MapType == D3D11_MAP_WRITE_DISCARD)
        MapFlags &= ~D3D11_MAP_FLAG_DO_NOT_WAIT;

      if (MapType != D3D11_MAP_WRITE_NO_OVERWRITE) {
        if (!WaitForResource(mappedImage, sequenceNumber, MapType, MapFlags))
          return DXGI_ERROR_WAS_STILL_DRAWING;
      }
      
      // Query the subresource's memory layout and hope that
      // the application respects the returned pitch values.
      mapPtr = mappedImage->mapPtr(0);
    } else {
      constexpr uint32_t DoInvalidate = (1u << 0);
      constexpr uint32_t DoPreserve   = (1u << 1);
      constexpr uint32_t DoWait       = (1u << 2);
      uint32_t doFlags;

      if (MapType == D3D11_MAP_READ) {
        // Reads will not change the image content, so we only need
        // to wait for the GPU to finish writing to the mapped buffer.
        doFlags = DoWait;
      } else if (MapType == D3D11_MAP_WRITE_DISCARD) {
        doFlags = DoInvalidate;

        // If we know for sure that the mapped buffer is currently not
        // in use by the GPU, we don't have to allocate a new slice.
        if (m_csThread.lastSequenceNumber() >= sequenceNumber && !mappedBuffer->isInUse(DxvkAccess::Read))
          doFlags = 0;
      } else if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_STAGING && (MapFlags & D3D11_MAP_FLAG_DO_NOT_WAIT)) {
        // Always respect DO_NOT_WAIT for mapped staging images
        doFlags = DoWait;
      } else if (MapType != D3D11_MAP_WRITE_NO_OVERWRITE || mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER) {
        // Need to synchronize thread to determine pending GPU accesses
        SynchronizeCsThread(sequenceNumber);

        // Don't implicitly discard large buffers or buffers of images with
        // multiple subresources, as that is likely to cause memory issues.
        VkDeviceSize bufferSize = pResource->CountSubresources() == 1
          ? pResource->GetMappedSlice(Subresource).length
          : MaxImplicitDiscardSize;

        if (bufferSize >= MaxImplicitDiscardSize) {
          // Don't check access flags, WaitForResource will return
          // early anyway if the resource is currently in use
          doFlags = DoWait;
        } else if (mappedBuffer->isInUse(DxvkAccess::Write)) {
          // There are pending GPU writes, need to wait for those
          doFlags = DoWait;
        } else if (mappedBuffer->isInUse(DxvkAccess::Read)) {
          // All pending GPU accesses are reads, so the buffer data
          // is still current, and we can prevent GPU synchronization
          // by creating a new slice with an exact copy of the data.
          doFlags = DoInvalidate | DoPreserve;
        } else {
          // There are no pending accesses, so we don't need to wait
          doFlags = 0;
        }
      } else {
        // No need to synchronize staging resources with NO_OVERWRITE
        // since the buffer will be used directly.
        doFlags = 0;
      }

      if (doFlags & DoInvalidate) {
        FlushImplicit(TRUE);

        DxvkBufferSliceHandle prevSlice = pResource->GetMappedSlice(Subresource);
        DxvkBufferSliceHandle physSlice = pResource->DiscardSlice(Subresource);

        EmitCs([
          cImageBuffer = mappedBuffer,
          cBufferSlice = physSlice
        ] (DxvkContext* ctx) {
          ctx->invalidateBuffer(cImageBuffer, cBufferSlice);
        });

        if (doFlags & DoPreserve)
          std::memcpy(physSlice.mapPtr, prevSlice.mapPtr, physSlice.length);

        mapPtr = physSlice.mapPtr;
      } else {
        if (doFlags & DoWait) {
          // We cannot respect DO_NOT_WAIT for buffer-mapped resources since
          // our internal copies need to be transparent to the application.
          if (mapMode == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER)
            MapFlags &= ~D3D11_MAP_FLAG_DO_NOT_WAIT;

          // Wait for mapped buffer to become available
          if (!WaitForResource(mappedBuffer, sequenceNumber, MapType, MapFlags))
            return DXGI_ERROR_WAS_STILL_DRAWING;
        }

        mapPtr = pResource->GetMappedSlice(Subresource).mapPtr;
      }
    }

    // Mark the given subresource as mapped
    pResource->SetMapType(Subresource, MapType);

    if (pMappedResource) {
      auto layout = pResource->GetSubresourceLayout(formatInfo->aspectMask, Subresource);
      pMappedResource->pData      = reinterpret_cast<char*>(mapPtr) + layout.Offset;
      pMappedResource->RowPitch   = layout.RowPitch;
      pMappedResource->DepthPitch = layout.DepthPitch;
    }

    m_mappedImageCount += 1;
    return S_OK;
  }
  
  
  void D3D11ImmediateContext::UnmapImage(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource) {
    D3D11_MAP mapType = pResource->GetMapType(Subresource);
    pResource->SetMapType(Subresource, D3D11_MAP(~0u));

    if (mapType == D3D11_MAP(~0u))
      return;

    // Decrement mapped image counter only after making sure
    // the given subresource is actually mapped right now
    m_mappedImageCount -= 1;

    if ((mapType != D3D11_MAP_READ) &&
        (pResource->GetMapMode() == D3D11_COMMON_TEXTURE_MAP_MODE_BUFFER)) {
      // Now that data has been written into the buffer,
      // we need to copy its contents into the image
      VkImageAspectFlags aspectMask = imageFormatInfo(pResource->GetPackedFormat())->aspectMask;
      VkImageSubresource subresource = pResource->GetSubresourceFromIndex(aspectMask, Subresource);

      UpdateImage(pResource, &subresource, VkOffset3D { 0, 0, 0 },
        pResource->MipLevelExtent(subresource.mipLevel),
        DxvkBufferSlice(pResource->GetMappedBuffer(Subresource)));
    }
  }
  
  
  void D3D11ImmediateContext::UpdateMappedBuffer(
          D3D11Buffer*                  pDstBuffer,
          UINT                          Offset,
          UINT                          Length,
    const void*                         pSrcData,
          UINT                          CopyFlags) {
    DxvkBufferSliceHandle slice;

    if (likely(CopyFlags != D3D11_COPY_NO_OVERWRITE)) {
      slice = pDstBuffer->DiscardSlice();

      EmitCs([
        cBuffer      = pDstBuffer->GetBuffer(),
        cBufferSlice = slice
      ] (DxvkContext* ctx) {
        ctx->invalidateBuffer(cBuffer, cBufferSlice);
      });
    } else {
      slice = pDstBuffer->GetMappedSlice();
      // NV-DXVK [T31Cache]/[Perf.FanoutCamCache] 2026-08-08: this is a CPU
      // write window onto the mapped bytes (UpdateSubresource NO_OVERWRITE
      // writes in place below), so it must bump the map generation like the
      // Map(WRITE_NO_OVERWRITE) path already does -- GetMapGeneration()'s
      // contract is "same (buf, mapGen) implies same bytes", and mapGen-keyed
      // caches would otherwise serve stale data after this write.
      pDstBuffer->NoteMapForWrite();
    }

    std::memcpy(reinterpret_cast<char*>(slice.mapPtr) + Offset, pSrcData, Length);
  }


  void STDMETHODCALLTYPE D3D11ImmediateContext::SwapDeviceContextState(
          ID3DDeviceContextState*           pState,
          ID3DDeviceContextState**          ppPreviousState) {
    InitReturnPtr(ppPreviousState);

    if (!pState)
      return;
    
    Com<D3D11DeviceContextState> oldState = std::move(m_stateObject);
    Com<D3D11DeviceContextState> newState = static_cast<D3D11DeviceContextState*>(pState);

    if (oldState == nullptr)
      oldState = new D3D11DeviceContextState(m_parent);
    
    if (ppPreviousState)
      *ppPreviousState = oldState.ref();
    
    m_stateObject = newState;

    oldState->SetState(m_state);
    newState->GetState(m_state);

    RestoreState();
  }


  void D3D11ImmediateContext::SynchronizeCsThread(uint64_t SequenceNumber) {
    D3D11DeviceLock lock = LockContext();

    // Dispatch current chunk so that all commands
    // recorded prior to this function will be run
    if (SequenceNumber > m_csSeqNum)
      FlushCsChunk();

    // NV-DXVK [Stage0 probe]: time the CS-thread catch-up wait. This is the
    // game thread waiting for the CS thread (which runs injectRTX) to reach a
    // sequence number — a candidate serialization point at the frame boundary.
    const auto tSync0 = std::chrono::steady_clock::now();
    m_csThread.synchronize(SequenceNumber);
    const auto tSync1 = std::chrono::steady_clock::now();
    const int64_t syncNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tSync1 - tSync0).count();
    g_gtSyncCsNs.fetch_add(syncNs, std::memory_order_relaxed);
    if (syncNs > 10000)
      g_gtSyncCsN.fetch_add(1u, std::memory_order_relaxed);
  }
  
  
  void D3D11ImmediateContext::SynchronizeDevice() {
    m_device->waitForIdle();
  }
  
  
  bool D3D11ImmediateContext::WaitForResource(
    const Rc<DxvkResource>&                 Resource,
          uint64_t                          SequenceNumber,
          D3D11_MAP                         MapType,
          UINT                              MapFlags) {
    // Determine access type to wait for based on map mode
    DxvkAccess access = MapType == D3D11_MAP_READ
      ? DxvkAccess::Write
      : DxvkAccess::Read;
    
    // Wait for any CS chunk using the resource to execute, since
    // otherwise we cannot accurately determine if the resource is
    // actually being used by the GPU right now.
    bool isInUse = Resource->isInUse(access);

    if (!isInUse) {
      SynchronizeCsThread(SequenceNumber);
      isInUse = Resource->isInUse(access);
    }

    if (MapFlags & D3D11_MAP_FLAG_DO_NOT_WAIT) {
      if (isInUse) {
        // We don't have to wait, but misbehaving games may
        // still try to spin on `Map` until the resource is
        // idle, so we should flush pending commands
        FlushImplicit(FALSE);
        return false;
      }
    } else {
      if (isInUse) {
        // Make sure pending commands using the resource get
        // executed on the the GPU if we have to wait for it
        Flush();
        SynchronizeCsThread(SequenceNumber);

        // NV-DXVK [Stage0 probe]: time the GPU resource wait. This is the block
        // that serializes a next-frame Map behind the CS-thread RTX inject still
        // reading the resource. A DISCARD map reaching here means discard-rename
        // was defeated (e.g. a pinned slice) — the prime suspect for the draw/
        // inject serialization.
        const auto tW0 = std::chrono::steady_clock::now();
        m_device->waitForResource(Resource, access);
        const auto tW1 = std::chrono::steady_clock::now();
        g_gtWaitResNs.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(tW1 - tW0).count(),
          std::memory_order_relaxed);
        g_gtWaitResN.fetch_add(1u, std::memory_order_relaxed);
        if (MapType == D3D11_MAP_WRITE_DISCARD)
          g_gtWaitResDiscN.fetch_add(1u, std::memory_order_relaxed);
      }
    }

    return true;
  }
  
  
  void D3D11ImmediateContext::EmitCsChunk(DxvkCsChunkRef&& chunk) {
    m_csSeqNum = m_csThread.dispatchChunk(std::move(chunk));
    m_csIsBusy = true;
  }


  void D3D11ImmediateContext::TrackTextureSequenceNumber(
          D3D11CommonTexture*         pResource,
          UINT                        Subresource) {
    pResource->TrackSequenceNumber(Subresource, m_csSeqNum + 1);
  }


  void D3D11ImmediateContext::TrackBufferSequenceNumber(
          D3D11Buffer*                pResource) {
    pResource->TrackSequenceNumber(m_csSeqNum + 1);
  }


  void D3D11ImmediateContext::FlushImplicit(BOOL StrongHint) {
    // Flush only if the GPU is about to go idle, in
    // order to keep the number of submissions low.
    uint32_t pending = m_device->pendingSubmissions();

    if (StrongHint || pending <= MaxPendingSubmits) {
      auto now = dxvk::high_resolution_clock::now();

      uint32_t delay = MinFlushIntervalUs
                     + IncFlushIntervalUs * pending;

      // Prevent flushing too often in short intervals.
      if (now - m_lastFlush >= std::chrono::microseconds(delay))
        Flush();
    }
  }


  void D3D11ImmediateContext::SignalEvent(HANDLE hEvent) {
    uint64_t value = ++m_eventCount;

    if (m_eventSignal == nullptr)
      m_eventSignal = new sync::CallbackFence();

    m_eventSignal->setCallback(value, [hEvent] {
      SetEvent(hEvent);
    });

    EmitCs([
      cSignal = m_eventSignal,
      cValue  = value
    ] (DxvkContext* ctx) {
      ctx->signal(cSignal, cValue);
    });
  }
  
}
